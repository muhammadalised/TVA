"""Freeze IAM frequency evidence and the matched FAU linguistic comparator."""

import argparse
from pathlib import Path

from tva.fau_linguistic_bigram import (
    build_fau_linguistic_adapter,
    build_iam_frequency_evidence,
    write_json,
)


DEFAULT_EVIDENCE = Path(
    'artifacts/tokenizers/source/iam-train-letter-bigram-counts-v1.json'
)
DEFAULT_OUTPUT = Path(
    'artifacts/tokenizers/fau_english_iam_linguistic_bigram_greedy_v1.json'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path)
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--evidence', type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument(
        '--rebuild-evidence',
        action='store_true',
        help='Rebuild evidence from authenticated IAM files before the adapter.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rebuild_evidence:
        if args.labels is None or args.selection is None:
            raise ValueError(
                '--labels and --selection are required with --rebuild-evidence'
            )
        if args.evidence.exists() and not args.overwrite:
            raise FileExistsError(f'evidence already exists: {args.evidence}')
        evidence = build_iam_frequency_evidence(args.labels, args.selection)
        print(f'Evidence SHA-256: {write_json(evidence, args.evidence)}')
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f'output already exists: {args.output}')
    adapter = build_fau_linguistic_adapter(args.evidence)
    print(f'Adapter SHA-256: {write_json(adapter, args.output)}')
    print(f'Wrote {args.output} ({adapter["size"]} classes)')


if __name__ == '__main__':
    main()
