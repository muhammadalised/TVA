#!/usr/bin/env python3
"""Run the frozen FAU tokenizer comparison sequentially and safely."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import yaml


REPO_ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = REPO_ROOT / 'configs' / 'thesis' / 'fau'
CONDITION_CONFIGS = {
    'handwriting_wd': CONFIG_ROOT / 'bigram_handwriting_wd.yaml',
    'linguistic_wd': CONFIG_ROOT / 'bigram_linguistic_wd.yaml',
    'handwriting_wi': CONFIG_ROOT / 'bigram_handwriting_wi.yaml',
    'linguistic_wi': CONFIG_ROOT / 'bigram_linguistic_wi.yaml',
}
DEFAULT_CONDITIONS = tuple(CONDITION_CONFIGS)
DEFAULT_FOLDS = (1, 2, 3, 4)
DEFAULT_BATCH_SIZE = 32
NUM_FOLDS = 5


def validate_folds(folds: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    """Return unique, valid FAU fold indices in the requested order."""
    normalized = tuple(folds)
    if not normalized:
        raise ValueError('At least one fold is required.')
    if len(set(normalized)) != len(normalized):
        raise ValueError('Fold indices must be unique.')
    invalid = [fold for fold in normalized if fold not in range(NUM_FOLDS)]
    if invalid:
        raise ValueError(f'FAU fold indices must be in 0..4; got {invalid}.')
    return normalized


def load_effective_configs(
    folds: list[int] | tuple[int, ...],
    conditions: list[str] | tuple[str, ...],
    batch_size: int,
) -> list[tuple[str, int, dict, Path]]:
    """Build effective configs without modifying the frozen base YAML files."""
    folds = validate_folds(folds)
    if batch_size < 1:
        raise ValueError('Batch size must be positive.')
    if not conditions:
        raise ValueError('At least one condition is required.')
    if len(set(conditions)) != len(conditions):
        raise ValueError('Condition names must be unique.')

    runs = []
    for fold in folds:
        for condition in conditions:
            path_config = CONDITION_CONFIGS[condition]
            with path_config.open('r', encoding='utf-8') as file:
                config = yaml.safe_load(file)

            _validate_base_config(condition, config)
            config['idx_fold'] = fold
            config['size_batch'] = batch_size
            output = REPO_ROOT / config['dir_work'] / str(fold)
            runs.append((condition, fold, config, output))
    return runs


def _validate_base_config(condition: str, config: dict) -> None:
    """Fail before training if a frozen FAU production invariant drifted."""
    expected_distribution = condition.rsplit('_', 1)[1]
    expected_tokenizer = (
        'handwriting_bigram_greedy'
        if condition.startswith('handwriting')
        else 'linguistic_bigram_greedy'
    )
    expected = {
        'checkpoint': None,
        'dataset_format': 'fau-zip',
        'device': 'cuda',
        'epoch': 300,
        'epoch_warmup': 30,
        'fau_distribution': expected_distribution,
        'num_channel': 7,
        'seed': 42,
        'test': False,
        'tokenizer': expected_tokenizer,
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f'Frozen config drift in {condition}: {mismatches}'
        )


def ensure_outputs_absent(runs: list[tuple[str, int, dict, Path]]) -> None:
    """Refuse the entire matrix before launch if any target already has data."""
    occupied = [
        output
        for _, _, _, output in runs
        if output.exists() and any(output.iterdir())
    ]
    if occupied:
        formatted = '\n'.join(f'  - {path}' for path in occupied)
        raise FileExistsError(
            'Refusing to overwrite existing FAU run directories:\n'
            f'{formatted}\n'
            'Inspect the existing run and select only untouched folds and '
            'conditions.'
        )


def validate_dataset() -> Path:
    """Validate the FAU archive override used by the dataset loader."""
    value = os.environ.get('TVA_FAU_DATASET_DIR')
    if not value:
        raise RuntimeError(
            'Set TVA_FAU_DATASET_DIR to the directory containing '
            'gold_wd.zip and gold_wi.zip.'
        )
    directory = Path(value).expanduser()
    missing = [
        name for name in ('gold_wd.zip', 'gold_wi.zip')
        if not (directory / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f'Missing FAU archives in {directory}: {", ".join(missing)}'
        )
    return directory


def run_matrix(
    runs: list[tuple[str, int, dict, Path]], dry_run: bool
) -> None:
    """Preview or execute the requested matrix in a deterministic order."""
    ensure_outputs_absent(runs)
    if not dry_run:
        validate_dataset()

    environment = os.environ.copy()
    environment['MPLBACKEND'] = 'Agg'
    total = len(runs)

    with tempfile.TemporaryDirectory(prefix='tva_fau_matrix_') as temporary:
        directory = Path(temporary)
        for index, (condition, fold, config, output) in enumerate(runs, 1):
            path_config = directory / f'{condition}_fold_{fold}.yaml'
            with path_config.open('w', encoding='utf-8') as file:
                yaml.safe_dump(config, file, allow_unicode=True)

            command = [
                sys.executable,
                str(REPO_ROOT / 'main.py'),
                '--config',
                str(path_config),
            ]
            print(
                f'[{index}/{total}] {condition}, fold {fold}, '
                f'batch {config["size_batch"]} -> {output}',
                flush=True,
            )
            print(f'  command: {" ".join(command)}', flush=True)
            if not dry_run:
                subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    env=environment,
                    check=True,
                )

    if dry_run:
        print(f'Preview complete: {total} runs; no training was started.')
        print(
            'Temporary preview configs were removed. Run this launcher '
            'without --dry-run to regenerate them and start training.'
        )
    else:
        print(f'FAU matrix complete: {total} runs finished successfully.')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Run the frozen FAU handwriting-aware versus linguistic Bigram '
            'matrix sequentially. Defaults preserve completed fold 0.'
        )
    )
    parser.add_argument(
        '--folds',
        nargs='+',
        type=int,
        default=list(DEFAULT_FOLDS),
        help='Fold indices to run (default: 1 2 3 4).',
    )
    parser.add_argument(
        '--conditions',
        nargs='+',
        choices=DEFAULT_CONDITIONS,
        default=list(DEFAULT_CONDITIONS),
        help='Conditions to run in the declared order.',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help='Physical batch size applied identically to every run.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate and preview the matrix without starting training.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        runs = load_effective_configs(
            args.folds,
            args.conditions,
            args.batch_size,
        )
        run_matrix(runs, args.dry_run)
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        raise SystemExit(f'ERROR: {error}') from error


if __name__ == '__main__':
    main()
