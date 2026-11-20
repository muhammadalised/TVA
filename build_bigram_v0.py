'''Build the train-only OnHW WI/RH fold-0 Bigram v0 comparison.'''

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


def build(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(f'Output already exists: {args.output_dir}')

    source_paths = {
        'train': args.train,
        'char_vocab': args.char_vocab,
        'force_ranking': args.force_ranking,
        'builder': Path(__file__),
        'tokenizer_code': Path(__file__).with_name('tva') / 'tokenizers.py',
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
        len(set(force_order)) != ELIGIBLE_COUNT
        or [int(row['rank']) for row in eligible] != list(range(1, ELIGIBLE_COUNT + 1))
        or any(pair not in frequencies for pair in force_order)
    ):
        raise ValueError('Expected 176 uniquely ranked eligible pairs present in fold-0 text.')
    frequency_order = sorted(force_order, key=lambda pair: (-frequencies[pair], pair))
    frequency_ranks = {pair: idx for idx, pair in enumerate(frequency_order, start=1)}
    selection = {'force': force_order[:PAIR_LIMIT], 'frequency': frequency_order[:PAIR_LIMIT]}
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

        write_json(staging / 'run_manifest.json', {
            'dataset': 'OnHW-Words500 WI/RH',
            'fold': FOLD,
            'split': 'train',
            'frequency_counting': 'All adjacent positions across fold-0 recordings',
            'frequency_tie_break': 'Ascending case-sensitive Unicode pair order',
            'segmentation': 'TVA BigramTokenizer greedy left-to-right',
            'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'python': sys.version,
            'command': [sys.executable, *sys.argv],
            'inputs': input_hashes,
            'output_sha256': {
                path.relative_to(staging).as_posix(): sha256(path.read_bytes())
                for path in sorted(staging.rglob('*.json'))
            },
        })
        staging.rename(args.output_dir)

    print(json.dumps(report, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, required=True, help='Training annotations JSON.')
    parser.add_argument('--char-vocab', type=Path, required=True, help='Fold-0 char vocabulary JSON.')
    parser.add_argument('--force-ranking', type=Path, required=True, help='Saved pair score CSV.')
    parser.add_argument('--output-dir', type=Path, required=True, help='New output directory.')
    return parser.parse_args()


if __name__ == '__main__':
    build(parse_args())
