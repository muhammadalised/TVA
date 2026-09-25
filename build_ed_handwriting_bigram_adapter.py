"""Build the frozen 224-class ED handwriting-bigram adapter."""

import argparse
from pathlib import Path

from tva.ed_handwriting_bigram import (
    ADAPTER_BIGRAM_COUNT,
    ADAPTER_POLICY_ID,
    ADAPTER_SIZE,
    build_ed_adapter,
    write_adapter,
)


DEFAULT_SOURCE = Path(
    'artifacts/tokenizers/source/iam-english-handwriting-bigram-v1.json'
)
DEFAULT_OUTPUT = Path(
    'artifacts/tokenizers/ed_iam_handwriting_bigram_greedy_v1.json'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f'output already exists: {args.output}; pass --overwrite to replace it'
        )
    adapter = build_ed_adapter(args.source)
    digest = write_adapter(adapter, args.output)
    print(f'Wrote {args.output}')
    print(f'Policy: {ADAPTER_POLICY_ID}')
    print(f'Classes: {ADAPTER_SIZE} ({ADAPTER_BIGRAM_COUNT} bigrams)')
    print(f'SHA-256: {digest}')


if __name__ == '__main__':
    main()
