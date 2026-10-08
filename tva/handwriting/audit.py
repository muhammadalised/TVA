"""Audit ED label compatibility after tokenizer vocabularies are frozen."""

import argparse
import hashlib
import os
import json
from pathlib import Path
import unicodedata
import zipfile

from .ed import ED_ALPHABET
from tva.handwriting_tokenizers import GreedyHandwritingBigramTokenizer
from tva.handwriting_tokenizers import GreedyLinguisticBigramTokenizer
from tva.handwriting_tokenizers import HandwritingBPETokenizer, HandwritingUnigramTokenizer
from tva.dataset.ed import ED_ARCHIVE_SHA256


ARCHIVE_SHA256 = {
    f'gold_{distribution}': digest
    for distribution, digest in ED_ARCHIVE_SHA256.items()
}
DEFAULT_DATASET_DIRECTORY = Path(os.environ.get(
    'TVA_ED_DATASET_DIR', '/mnt/c/Users/Ali/Downloads/fau-english-dataset'
))
DEFAULT_ADAPTER = Path(
    'artifacts/tokenizers/ed_iam_handwriting_bigram_greedy_v1.json'
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_archive(path: Path, stem: str) -> dict:
    expected_digest = ARCHIVE_SHA256[stem]
    digest = _sha256(path)
    if digest != expected_digest:
        raise ValueError(
            f'{stem} archive SHA-256 mismatch: expected {expected_digest}, got {digest}'
        )
    with zipfile.ZipFile(path) as archive:
        value = json.loads(archive.read(f'{stem}/annotations.json'))
    if value.get('categories') != list(ED_ALPHABET):
        raise ValueError(f'{stem} declared categories do not match the frozen alphabet')
    return value


def audit(
    dataset_directory: Path,
    adapter_path: Path,
    tokenizer_kind: str = 'handwriting',
) -> dict[str, int]:
    """Check both distributions contain the same 2,550 encodable records."""
    from tva.tokenizers import BigramTokenizer

    tokenizer_classes = {
        'handwriting': GreedyHandwritingBigramTokenizer,
        'linguistic': GreedyLinguisticBigramTokenizer,
        'original': BigramTokenizer,
        'bpe': HandwritingBPETokenizer,
        'unigram': HandwritingUnigramTokenizer,
    }
    tokenizer = tokenizer_classes[tokenizer_kind]()
    tokenizer.load(adapter_path)
    records_by_split: dict[str, dict[int, str]] = {}
    encoded_tokens = 0
    for stem in ARCHIVE_SHA256:
        payload = read_archive(dataset_directory / f'{stem}.zip', stem)
        records: dict[int, str] = {}
        for annotation in payload['annotations']:
            sample_id = annotation['id']
            label = annotation['label']
            if sample_id in records:
                raise ValueError(f'{stem} contains duplicate sample ID {sample_id}')
            ids = tokenizer.encode(label)
            if not ids or 0 in ids:
                raise ValueError(f'{stem} sample {sample_id} emitted an invalid target')
            normalized = unicodedata.normalize('NFC', label)
            if tokenizer.decode(ids) != normalized:
                raise ValueError(f'{stem} sample {sample_id} failed round-trip')
            records[sample_id] = normalized
            encoded_tokens += len(ids)
        records_by_split[stem] = records
    if records_by_split['gold_wd'] != records_by_split['gold_wi']:
        raise ValueError('WD and WI archives do not contain identical labeled records')
    records = records_by_split['gold_wd']
    if len(records) != 2_550:
        raise ValueError(f'expected 2,550 records, found {len(records)}')
    return {
        'records_per_archive': len(records),
        'archives_checked': len(records_by_split),
        'label_encodings_checked': sum(map(len, records_by_split.values())),
        'encoded_tokens_checked': encoded_tokens,
        'blank_ids_emitted': 0,
        'round_trip_failures': 0,
    }


def compare(dataset_directory, *, labels_path=None, selection_path=None, output_path=None):
    """Compare all frozen vocabularies and ED targets without training a model."""
    from collections import Counter
    from itertools import combinations
    import string

    from tva.handwriting_tokenizers import get_tokenizer
    from .common import canonical_json_bytes as json_bytes, require, save_frozen, sha256_bytes as sha256
    from .iam import DEFAULT_LABELS, DEFAULT_SELECTION, read_iam_training

    REPO_ROOT = Path(__file__).resolve().parents[2]
    ED_DIRECTORY = Path(dataset_directory)
    OUTPUT = REPO_ROOT / 'artifacts/tokenizers/ed_iam_tva_original_bigram_greedy_v1.json'
    HANDWRITING = REPO_ROOT / 'artifacts/tokenizers/ed_iam_handwriting_bigram_greedy_v1.json'
    CUSTOM_LINGUISTIC = REPO_ROOT / 'artifacts/tokenizers/ed_iam_linguistic_bigram_greedy_v1.json'
    annotations = read_iam_training(labels_path or DEFAULT_LABELS, selection_path or DEFAULT_SELECTION)
    unique_labels = {row['label'] for row in annotations}
    vocab = json.loads(OUTPUT.read_bytes())['vocab']
    selected_bigrams = [
        token for token, index in sorted(vocab.items(), key=lambda item: item[1])
        if len(token) == 2
    ]
    tokenizer_bigram = get_tokenizer('bigram')
    tokenizer_bigram.load(str(OUTPUT))
    counts = Counter(
        text[i:i + 2] for text in unique_labels for i in range(len(text) - 1)
    )
    original_pairs = set(selected_bigrams)
    handwriting_pairs = {t for t in json.loads(HANDWRITING.read_bytes())['vocab'] if len(t) == 2}
    custom_pairs = {t for t in json.loads(CUSTOM_LINGUISTIC.read_bytes())['vocab'] if len(t) == 2}
    pair_sets = {'original_tva': original_pairs, 'handwriting': handwriting_pairs, 'custom_linguistic': custom_pairs}
    for name, pairs in pair_sets.items():
        require(len(pairs) == 145, f'{name} bigram count mismatch')
    nonletter_pairs = [t for t in selected_bigrams if any(c not in string.ascii_letters for c in t)]
    vocabulary_comparison = {
        'candidate_pair_count': len(counts),
        'selected_ascii_letter_pairs': 145 - len(nonletter_pairs),
        'selected_nonletter_pairs': nonletter_pairs,
        'selected_space_pairs': [t for t in selected_bigrams if ' ' in t],
        'selected_punctuation_pairs': [t for t in selected_bigrams if any(c in string.punctuation for c in t)],
        'selected_digit_pairs': [t for t in selected_bigrams if any(c in string.digits for c in t)],
        'pairwise_overlap': {
            f'{a} vs {b}': len(pair_sets[a] & pair_sets[b])
            for a, b in combinations(pair_sets, 2)
        },
        'selected_bigrams': [
            {'rank': rank, 'token': token, 'unique_label_occurrence_count': counts[token]}
            for rank, token in enumerate(selected_bigrams, start=1)
        ],
        'selection_boundary_count': counts[selected_bigrams[-1]],
        'pairs_tied_at_selection_boundary': sorted(t for t, count in counts.items() if count == counts[selected_bigrams[-1]]),
    }


    tokenizers = {'original_tva': tokenizer_bigram}
    for name, kind, artifact in [
        ('handwriting', 'handwriting_bigram_greedy', HANDWRITING),
        ('custom_linguistic', 'linguistic_bigram_greedy', CUSTOM_LINGUISTIC),
    ]:
        tokenizer = get_tokenizer(kind)
        tokenizer.load(str(artifact))
        require(tokenizer.size == 224 and tokenizer.vocab.get('') == 0, f'{name} dimensions mismatch')
        require({t for t in tokenizer.vocab if len(t) == 1} == set(ED_ALPHABET), f'{name} alphabet mismatch')
        tokenizers[name] = tokenizer


    def token_strings(tokenizer, text):
        ids = tokenizer.encode(text)
        require(bool(ids) and 0 not in ids, 'Empty target or CTC blank emitted')
        require(tokenizer.decode(ids) == text, 'Exact ED label reconstruction failed')
        return [tokenizer.decode([index]) for index in ids]


    archive_records = {}
    segmentation = {}
    examples = []
    for stem in ARCHIVE_SHA256:
        payload = read_archive(ED_DIRECTORY / f'{stem}.zip', stem)
        rows = payload['annotations']
        records = {row['id']: row['label'] for row in rows}
        require(len(records) == len(rows) == 2550, f'{stem}: expected 2,550 unique sample IDs')
        archive_records[stem] = records
        parts = {name: {} for name in tokenizers}
        for sample_id, text in records.items():
            for name, tokenizer in tokenizers.items():
                parts[name][sample_id] = token_strings(tokenizer, text)
            if stem == 'gold_wd' and len(examples) < 5 and parts['original_tva'][sample_id] != parts['custom_linguistic'][sample_id]:
                examples.append({'id': sample_id, 'label': text, 'tokens': {name: parts[name][sample_id] for name in tokenizers}})
        characters = sum(len(text) for text in records.values())
        summary = {}
        for name in tokenizers:
            total = sum(len(tokens) for tokens in parts[name].values())
            summary[name] = {
                'records': len(records), 'characters': characters, 'encoded_tokens': total,
                'mean_tokens_per_record': total / len(records),
                'token_reduction_vs_characters': 1 - total / characters,
                'bigram_tokens_used': sum(len(t) == 2 for tokens in parts[name].values() for t in tokens),
                'round_trip_failures': 0, 'blank_ids_emitted': 0,
            }
        differences = {
            f'{a} vs {b}': sum(parts[a][i] != parts[b][i] for i in records)
            for a, b in combinations(tokenizers, 2)
        }
        with zipfile.ZipFile(ED_DIRECTORY / f'{stem}.zip') as archive:
            split_payloads = {
                split: json.loads(archive.read(f'{stem}/{split}.json'))
                for split in ('train', 'val')
            }
        folds = {}
        for split, split_payload in split_payloads.items():
            require(split_payload['info']['num_fold'] == 5, 'Expected five ED folds')
            require(set(split_payload['annotations']) == {str(i) for i in range(5)}, 'Invalid ED fold keys')
            for fold_id, fold_rows in split_payload['annotations'].items():
                ids = [row['id'] for row in fold_rows]
                require(len(ids) == len(set(ids)) and set(ids) <= set(records), 'Invalid fold IDs')
                require(all(row['label'] == records[row['id']] for row in fold_rows), 'Fold label mismatch')
                folds.setdefault(fold_id, {})[split] = {
                    'records': len(ids),
                    'encoded_tokens': {name: sum(len(parts[name][i]) for i in ids) for name in tokenizers},
                }
        for fold_id in folds:
            train_ids = {row['id'] for row in split_payloads['train']['annotations'][fold_id]}
            val_ids = {row['id'] for row in split_payloads['val']['annotations'][fold_id]}
            require(not train_ids & val_ids and train_ids | val_ids == set(records), 'Invalid ED fold partition')
        segmentation[stem] = {'conditions': summary, 'different_segmentations': differences, 'folds': folds}
    require(archive_records['gold_wd'] == archive_records['gold_wi'], 'WD/WI records differ')
    audit_report = {
        'schema_version': 'tva.iam-original-bigram-comparison.v1',
        'tokenizer_sha256': sha256(OUTPUT.read_bytes()),
        'comparator_sha256': {'handwriting': sha256(HANDWRITING.read_bytes()), 'custom_linguistic': sha256(CUSTOM_LINGUISTIC.read_bytes())},
        'archive_sha256': dict(ARCHIVE_SHA256),
        'unique_ed_records': 2550, 'ed_labels_used_for_vocabulary_selection': False,
        'vocabulary_comparison': vocabulary_comparison, 'segmentation': segmentation,
        'examples': examples,
    }

    if output_path is not None:
        save_frozen(output_path, json_bytes(audit_report))
    return audit_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-directory', type=Path, default=DEFAULT_DATASET_DIRECTORY)
    parser.add_argument('--adapter', type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument('--tokenizer-kind', choices=('handwriting', 'linguistic', 'original', 'bpe', 'unigram'), default='handwriting')
    parser.add_argument('--compare', action='store_true', help='Compare all frozen vocabularies and ED targets.')
    parser.add_argument('--output', type=Path, help='Optional frozen JSON report for --compare.')
    args = parser.parse_args()
    if args.compare:
        report = compare(args.dataset_directory, output_path=args.output)
        summary = {
            'unique_ed_records': report['unique_ed_records'],
            'pairwise_overlap': report['vocabulary_comparison']['pairwise_overlap'],
            'segmentation': {
                stem: {k: v for k, v in value.items() if k != 'folds'}
                for stem, value in report['segmentation'].items()
            },
        }
    else:
        summary = audit(args.dataset_directory, args.adapter, args.tokenizer_kind)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
