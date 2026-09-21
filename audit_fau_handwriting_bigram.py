"""Audit all FAU English labels against the already-frozen greedy adapter."""

import argparse
import hashlib
import json
from pathlib import Path
import unicodedata
import zipfile

from tva.fau_handwriting_bigram import FAU_ALPHABET
from tva.handwriting_bigram_tokenizer import GreedyHandwritingBigramTokenizer
from tva.handwriting_bigram_tokenizer import GreedyLinguisticBigramTokenizer
from tva.dataset.fau import FAU_ARCHIVE_SHA256


ARCHIVE_SHA256 = {
    f'gold_{distribution}': digest
    for distribution, digest in FAU_ARCHIVE_SHA256.items()
}
DEFAULT_DATASET_DIRECTORY = Path('/mnt/c/Users/Ali/Downloads/fau-english-dataset')
DEFAULT_ADAPTER = Path(
    'artifacts/tokenizers/fau_english_iam_handwriting_bigram_greedy_v1.json'
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
    if value.get('categories') != list(FAU_ALPHABET):
        raise ValueError(f'{stem} declared categories do not match the frozen alphabet')
    return value


def audit(
    dataset_directory: Path,
    adapter_path: Path,
    tokenizer_kind: str = 'handwriting',
) -> dict[str, int]:
    """Check both distributions contain the same 2,550 encodable records."""
    tokenizer_classes = {
        'handwriting': GreedyHandwritingBigramTokenizer,
        'linguistic': GreedyLinguisticBigramTokenizer,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--dataset-directory', type=Path, default=DEFAULT_DATASET_DIRECTORY
    )
    parser.add_argument('--adapter', type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument(
        '--tokenizer-kind',
        choices=('handwriting', 'linguistic'),
        default='handwriting',
    )
    args = parser.parse_args()
    print(
        json.dumps(
            audit(args.dataset_directory, args.adapter, args.tokenizer_kind),
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
