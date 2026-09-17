"""Build matched fold-specific linguistic Bigram tokenizers."""

import argparse
from pathlib import Path

from tva.linguistic_bigram import build_dataset_models, write_dataset_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', required=True, type=Path)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output-directory', required=True, type=Path)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = build_dataset_models(args.train, dataset=args.dataset)
    manifest = write_dataset_models(
        models,
        args.output_directory,
        overwrite=args.overwrite,
    )
    print(f'Wrote {len(models)} folds to {args.output_directory}')
    for fold, artifact in manifest['artifacts'].items():
        print(f'Fold {fold}: {artifact["sha256"]}')


if __name__ == '__main__':
    main()
