'''Build train-only OnHW fold-0 bigram vocabularies and verify their targets.'''

import argparse
import csv
import hashlib
import io
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from tva.tokenizers import get_tokenizer


FOLD = 0
PAIR_LIMIT = 100
ELIGIBLE_COUNT = 176
CHAR_VOCAB_SIZE = 60


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )


def build_vocab(chars, pairs: list[str], labels: list[str], path: Path):
    vocab = chars.vocab | {
        pair: idx for idx, pair in enumerate(pairs, start=chars.size)
    }
    inverse = {idx: token for token, idx in vocab.items()}
    path.parent.mkdir()
    write_json(path, {'vocab': vocab, 'idx_token': inverse})

    tokenizer = get_tokenizer('bigram')
    tokenizer.load(path)
    if (
        tokenizer.vocab != vocab
        or tokenizer.idx_token != inverse
        or tokenizer.size != CHAR_VOCAB_SIZE + PAIR_LIMIT
        or set(inverse) != set(range(tokenizer.size))
    ):
        raise ValueError(f'Invalid vocabulary IDs in {path}.')

    token_counts = Counter()
    for label in labels:
        ids = tokenizer.encode(label)
        if 0 in ids or tokenizer.decode(ids) != label:
            raise ValueError(f'Invalid round-trip in {path} for {label!r}.')
        token_counts.update(ids)
    return tokenizer, token_counts


def emitted_pairs(chars, pairs: list[str], word_types: set[str]) -> set[str]:
    tokenizer = get_tokenizer('bigram')
    pair_ids = {pair: idx for idx, pair in enumerate(pairs, start=chars.size)}
    tokenizer.vocab = chars.vocab | pair_ids
    used_ids = {idx for text in word_types for idx in tokenizer.encode(text)}
    return {pair for pair, idx in pair_ids.items() if idx in used_ids}


def select_emitting_pairs(chars, order: list[str], word_types: set[str], limit: int):
    """Keep active top-ranked pairs and refill without shadowing them."""
    initial = order[:limit]
    active = emitted_pairs(chars, initial, word_types)
    selected = [pair for pair in initial if pair in active]
    rejected = []
    for pair in order[limit:]:
        if len(selected) == limit:
            break
        trial = [*selected, pair]
        if len(emitted_pairs(chars, trial, word_types)) == len(trial):
            selected.append(pair)
        else:
            rejected.append(pair)
    if len(selected) != limit:
        raise ValueError(f'Only {len(selected)} pairs can be selected with this rule; need {limit}.')
    return selected, {
        'unused_in_initial_top_pairs': [pair for pair in initial if pair not in active],
        'replacements': selected[len(active):],
        'rejected_replacements': rejected,
    }


def matched_evidence(args, label_count: int, boundary_count: int, eligible_count: int) -> dict:
    report_path = args.force_ranking.with_name('continuity_score_summary.json')
    report = json.loads(report_path.read_bytes())
    provenance = report['provenance']
    config_path = Path(provenance['config'])
    config = yaml.safe_load(config_path.read_bytes())
    export_path = Path(report['input_jsonl'])
    summary_path = export_path.with_name('summary.json')
    export = json.loads(summary_path.read_bytes())
    checkpoint = Path(provenance['checkpoint'])
    if (
        args.train.resolve() != (Path(config['dir_dataset']) / 'train.json').resolve()
        or args.char_vocab.resolve() != (Path(config['dir_tokenizer']) / f'{FOLD}.json').resolve()
        or config['arch_de'] != 'unilstm_b'
        or provenance['fold'] != FOLD
        or provenance['split'] != 'train'
        or provenance['checkpoint_epoch'] != 299
        or any(export[key] != provenance[key] for key in
               ('config', 'checkpoint', 'checkpoint_epoch', 'fold', 'split'))
        or export['requested_samples'] != label_count
        or export['completed_samples'] != label_count
        or export['exported_boundaries'] != boundary_count
        or report['total_boundaries'] != boundary_count
        or report['eligible_pairs'] != eligible_count
    ):
        raise ValueError('Expected a complete same-split A0 epoch-299 training export.')
    expected_parameters = {'local_window_ms': 100, 'local_weight': 0.75,
                           'region_weight': 0.25, 'min_occurrences': 100,
                           'min_writers': 30, 'min_stratum_occurrences': 100}
    if any(report['parameters'][key] != value for key, value in expected_parameters.items()):
        raise ValueError('The matched pilot uses the existing force score and support gate.')
    return {
        'force_report': report_path,
        'alignment_config': config_path,
        'alignment_export_summary': summary_path,
        'alignment_checkpoint': checkpoint,
        'training_boundary_evidence': export_path,
    }


def build(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(f'Output already exists: {args.output_dir}')

    source_paths = {
        'train': args.train,
        'char_vocab': args.char_vocab,
        'force_ranking': args.force_ranking,
        'builder': Path(__file__),
        'tokenizer_code': Path(__file__).with_name('tokenizers.py'),
    }
    sources = {name: path.read_bytes() for name, path in source_paths.items()}
    labels = [
        row['label'] for row in json.loads(sources['train'])['annotations'][str(FOLD)]
    ]
    if not labels or any(not label for label in labels):
        raise ValueError('Expected nonempty labels in fold-0 training annotations.')

    chars = get_tokenizer('char')
    chars.load(args.char_vocab)
    if (
        chars.size != CHAR_VOCAB_SIZE
        or chars.vocab.get('') != 0
        or set(chars.idx_char) != set(range(CHAR_VOCAB_SIZE))
        or chars.idx_char != {idx: token for token, idx in chars.vocab.items()}
        or any(len(token) != 1 for token in chars.vocab if token)
    ):
        raise ValueError('Expected the fold-0 character vocabulary with blank ID 0.')
    if set(''.join(labels)) - chars.vocab.keys():
        raise ValueError('Training text contains characters outside the fallback vocabulary.')

    frequencies = Counter(
        label[index:index + 2]
        for label in labels
        for index in range(len(label) - 1)
    )
    ranking = csv.DictReader(io.StringIO(sources['force_ranking'].decode('utf-8')))
    eligible = sorted(
        (row for row in ranking if row['eligible'] == 'True'),
        key=lambda row: int(row['rank']),
    )
    force_order = [row['pair'] for row in eligible]
    if (
        len(force_order) < PAIR_LIMIT
        or len(set(force_order)) != len(force_order)
        or (not args.match_emitted and len(force_order) != ELIGIBLE_COUNT)
        or [int(row['rank']) for row in eligible] != list(range(1, len(eligible) + 1))
        or any(pair not in frequencies for pair in force_order)
    ):
        raise ValueError('Expected a complete, unique ranking of eligible pairs in fold-0 text.')
    frequency_order = sorted(force_order, key=lambda pair: (-frequencies[pair], pair))
    frequency_ranks = {pair: idx for idx, pair in enumerate(frequency_order, start=1)}
    orders = {'force': force_order, 'frequency': frequency_order}
    selection = {name: order[:PAIR_LIMIT] for name, order in orders.items()}
    matching_details = {}
    evidence_hashes = {}
    if args.match_emitted:
        evidence_paths = matched_evidence(
            args, len(labels), sum(len(label) - 1 for label in labels), len(eligible)
        )
        for name, path in evidence_paths.items():
            with path.open('rb') as file:
                digest = hashlib.file_digest(file, 'sha256').hexdigest()
            evidence_hashes[name] = {'path': str(path), 'sha256': digest}
        word_types = set(labels)
        for name, order in orders.items():
            selection[name], matching_details[name] = select_emitting_pairs(
                chars, order, word_types, PAIR_LIMIT
            )
    common = set(selection['force']) & set(selection['frequency'])
    num_chars = sum(map(len, labels))
    report = {
        'fold': FOLD,
        'split': 'train',
        'recordings': len(labels),
        'unique_labels': len(set(labels)),
        'characters': num_chars,
        'eligible_pairs': len(eligible),
        'selected_pairs_per_vocabulary': PAIR_LIMIT,
        'shared_pairs': len(common),
        'pair_jaccard': len(common) / (2 * PAIR_LIMIT - len(common)),
        'conditions': {},
        'examples': {},
    }

    if args.match_emitted:
        report['selection_policy'] = 'match-emitted'
        report['refill'] = matching_details

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output_dir.parent) as tmp:
        staging = Path(tmp) / 'build'
        staging.mkdir()
        tokenizers, counts = {}, {}
        for variant, pairs in selection.items():
            tokenizer, token_counts = build_vocab(
                chars, pairs, labels, staging / variant / f'{FOLD}.json'
            )
            tokenizers[variant], counts[variant] = tokenizer, token_counts
            num_tokens = token_counts.total()
            report['conditions'][variant] = {
                'vocab_size_including_blank': tokenizer.size,
                'round_trips_passed': len(labels),
                'tokens': num_tokens,
                'characters_per_token': num_chars / num_tokens,
                'token_reduction_fraction': (num_chars - num_tokens) / num_chars,
                'distinct_bigram_tokens_used': sum(
                    tokenizer.vocab[pair] in token_counts for pair in pairs
                ),
            }
            if args.match_emitted and report['conditions'][variant]['distinct_bigram_tokens_used'] != PAIR_LIMIT:
                raise ValueError(f'{variant} must emit all {PAIR_LIMIT} bigrams.')

        for label in ('Mittwoch', 'Programm', 'drei', 'Chief', 'Haus'):
            if label in labels:
                report['examples'][label] = {
                    variant: [tokenizer.idx_token[idx] for idx in tokenizer.encode(label)]
                    for variant, tokenizer in tokenizers.items()
                }

        pairs = []
        for row in eligible:
            pair = row['pair']
            record = {
                'pair': pair,
                'force_rank': int(row['rank']),
                'force_score': float(row['provisional_force_continuity_score']),
                'force_occurrences': int(row['occurrence_count']),
                'force_writers': int(row['writer_count']),
                'text_occurrences': frequencies[pair],
                'frequency_rank': frequency_ranks[pair],
            }
            for variant, tokenizer in tokenizers.items():
                token_id = tokenizer.vocab.get(pair)
                record[f'{variant}_id'] = token_id
                record[f'{variant}_tokens_emitted'] = counts[variant].get(token_id, 0)
            pairs.append(record)

        write_json(staging / 'summary.json', report)
        write_json(staging / 'pair_selection.json', {'pairs': pairs})
        input_hashes = {}
        for name, path in source_paths.items():
            if path.read_bytes() != sources[name]:
                raise ValueError(f'Input changed during the build: {path}')
            input_hashes[name] = {'path': str(path), 'sha256': sha256(sources[name])}

        if args.match_emitted:
            input_hashes.update(evidence_hashes)
        manifest = {
            'dataset': args.train.parent.name if args.match_emitted else 'OnHW-Words500 WI/RH',
            'fold': FOLD,
            'split': 'train',
            'frequency_counting': 'All adjacent positions across fold-0 recordings',
            'frequency_tie_break': 'Ascending case-sensitive Unicode pair order',
            'segmentation': 'TVA BigramTokenizer greedy left-to-right',
            'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'python': sys.version,
            'command': [sys.executable, *sys.orig_argv[1:]],
            'inputs': input_hashes,
            'output_sha256': {
                path.relative_to(staging).as_posix(): sha256(path.read_bytes())
                for path in sorted(staging.rglob('*.json'))
            },
        }
        if args.match_emitted:
            manifest['selection_policy'] = report['selection_policy']
        write_json(staging / 'run_manifest.json', manifest)
        staging.rename(args.output_dir)

    print(json.dumps(report, ensure_ascii=False, indent=2))


def verify_matched_vocabs(config: argparse.Namespace) -> None:
    """Check the two built vocabularies before a matched run."""
    vocab_dir = Path(config.dir_tokenizer)
    if config.tokenizer != 'bigram' or vocab_dir.name not in ('frequency', 'force'):
        raise ValueError('A matched run needs the force or frequency Bigram vocabulary.')
    root = vocab_dir.parent
    manifest = json.loads((root / 'run_manifest.json').read_bytes())
    summary = json.loads((root / 'summary.json').read_bytes())
    train = Path(config.dir_dataset) / 'train.json'
    if (
        manifest['dataset'] != train.parent.name
        or manifest['fold'] != config.idx_fold
        or manifest['split'] != 'train'
        or manifest.get('selection_policy') != 'match-emitted'
        or hashlib.sha256(train.read_bytes()).hexdigest() != manifest['inputs']['train']['sha256']
    ):
        raise ValueError('The matched vocabularies must come from this training split.')
    for condition in ('frequency', 'force'):
        relative = f'{condition}/{config.idx_fold}.json'
        info = summary['conditions'][condition]
        if (
            info['vocab_size_including_blank'] != CHAR_VOCAB_SIZE + PAIR_LIMIT
            or info['distinct_bigram_tokens_used'] != PAIR_LIMIT
            or info['round_trips_passed'] != summary['recordings']
            or hashlib.sha256((root / relative).read_bytes()).hexdigest()
            != manifest['output_sha256'][relative]
        ):
            raise ValueError(f'Invalid matched vocabulary: {root / relative}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, required=True, help='Training annotations JSON.')
    parser.add_argument('--char-vocab', type=Path, required=True, help='Fold-0 char vocabulary JSON.')
    parser.add_argument('--force-ranking', type=Path, required=True, help='Saved pair score CSV.')
    parser.add_argument('--output-dir', type=Path, required=True, help='New output directory.')
    parser.add_argument('--match-emitted', action='store_true',
                        help='Use same-split fixed-epoch evidence and require 100 emitted bigrams.')
    return parser.parse_args()
