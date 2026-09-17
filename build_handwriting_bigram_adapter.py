"""Build the frozen OnHW adapter from the pinned IAM+READ DTLR model."""

import argparse
from pathlib import Path

from tva.handwriting_bigram_adapter import (
    ADAPTER_BIGRAM_COUNT,
    ADAPTER_POLICY_ID,
    ADAPTER_SIZE,
    build_frozen_onhw_adapter,
    write_adapter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--source',
        required=True,
        type=Path,
        help='Path to the pinned combined IAM+READ model.json.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('artifacts/tokenizers/onhw_words500_rh_iam_read_v1.json'),
        help='Canonical adapter output path.',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Replace an existing output file.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f'output already exists: {args.output}; pass --overwrite to replace it'
        )
    adapter = build_frozen_onhw_adapter(args.source)
    digest = write_adapter(adapter, args.output)
    print(f'Wrote {args.output}')
    print(f'Policy: {ADAPTER_POLICY_ID}')
    print(f'Classes: {ADAPTER_SIZE} ({ADAPTER_BIGRAM_COUNT} bigrams)')
    print(f'SHA-256: {digest}')


if __name__ == '__main__':
    main()
