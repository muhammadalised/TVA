"""Run cross-validation folds sequentially with the active Python environment."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

import yaml


REPO_ROOT = Path(__file__).resolve().parent


def prepare_runs(cfgs: dict, folds: list[int] | None = None) -> list[dict]:
    """Validate all selected inputs and output paths before starting any fold."""
    if cfgs['idx_fold'] != -1:
        raise ValueError('Cross-validation requires idx_fold=-1.')

    if cfgs.get('dataset_format', 'json-directory') == 'ed-zip':
        from tva.dataset import get_ed_num_folds

        num_folds = get_ed_num_folds(
            cfgs['dir_dataset'], cfgs['ed_distribution']
        )
    else:
        documents = {}
        for partition in ('train', 'val'):
            path = Path(cfgs['dir_dataset']) / f'{partition}.json'
            with path.open('r', encoding='utf-8') as handle:
                documents[partition] = json.load(handle)
        num_folds = documents['train']['info']['num_fold']
        expected = {str(fold) for fold in range(num_folds)}
        for partition, document in documents.items():
            if document['info']['num_fold'] != num_folds or set(
                document['annotations']
            ) != expected:
                raise ValueError(
                    f'Inconsistent fold metadata in {partition}.json.'
                )

    selected = list(range(num_folds)) if folds is None else list(folds)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError('Select at least one fold, without duplicates.')
    if any(fold not in range(num_folds) for fold in selected):
        raise ValueError(f'Fold indices must be in 0..{num_folds - 1}.')

    runs = []
    for fold in selected:
        config = deepcopy(cfgs)
        config['idx_fold'] = fold
        checkpoint = config.get('checkpoint')
        if checkpoint:
            config['checkpoint'] = checkpoint.replace('-1', str(fold))
            if not Path(config['checkpoint']).is_file():
                raise FileNotFoundError(config['checkpoint'])

        tokenizer = config.get('dir_tokenizer')
        if tokenizer:
            path = Path(tokenizer)
            if not path.is_file():
                path = path / f'{fold}.json'
            if not path.is_file():
                raise FileNotFoundError(
                    f'Missing tokenizer for fold {fold}: {path}'
                )
        elif config['tokenizer'] != 'char' or not config.get('categories'):
            raise ValueError(
                'A tokenizer artifact or character categories are required.'
            )

        output = Path(config['dir_work']) / str(fold)
        if not checkpoint and not config.get('test', False) and output.exists():
            if not output.is_dir() or any(output.iterdir()):
                raise FileExistsError(
                    f'Refusing to overwrite fold {fold}: {output}. '
                    'Choose a new dir_work, select untouched --folds, or set '
                    'an explicit recovery checkpoint to resume.'
                )
        runs.append(config)
    return runs


def train_cv(
    cfgs: dict,
    path_main: str,
    *,
    dry_run: bool = False,
    folds: list[int] | None = None,
) -> None:
    """Run validated folds in order; a failed child stops the entire run."""
    runs = prepare_runs(cfgs, folds)
    main_path = Path(path_main).resolve()
    if not main_path.is_file():
        raise FileNotFoundError(main_path)
    environment = os.environ.copy()
    environment['MPLBACKEND'] = 'Agg'
    environment['PYTHONUNBUFFERED'] = '1'
    print(f'Python: {sys.executable}', flush=True)
    print(f'Folds: {[config["idx_fold"] for config in runs]}', flush=True)
    print(f'Results: {cfgs["dir_work"]}', flush=True)

    with tempfile.TemporaryDirectory(prefix='tva_cv_') as temporary:
        for index, config in enumerate(runs, 1):
            fold = config['idx_fold']
            path_config = Path(temporary) / f'f{fold}.yaml'
            with path_config.open('w', encoding='utf-8') as handle:
                yaml.safe_dump(config, handle, allow_unicode=True)
            command = [
                sys.executable, '-u', str(main_path), '-c', str(path_config)
            ]
            print(
                f'[{index}/{len(runs)}] Fold {fold}: {shlex.join(command)}',
                flush=True,
            )
            if not dry_run:
                subprocess.run(command, check=True, env=environment)

    print(
        'Dry run complete; no training started.' if dry_run else
        'All selected folds completed.',
        flush=True,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--config', required=True, help='Training YAML file.')
    parser.add_argument('-m', '--main', default='main.py', help='Training script.')
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Validate and preview without training.',
    )
    parser.add_argument(
        '--folds', type=int, nargs='+',
        help='Optional subset, e.g. --folds 2 3 4.',
    )
    args = parser.parse_args()
    # Dataset and result paths in repository configs are repository-relative.
    os.chdir(REPO_ROOT)
    try:
        with open(args.config, 'r', encoding='utf-8') as handle:
            cfgs = yaml.safe_load(handle)
        train_cv(cfgs, args.main, dry_run=args.dry_run, folds=args.folds)
    except subprocess.CalledProcessError as error:
        print(
            f'Training failed (exit {error.returncode}); '
            'later folds were not started.',
            file=sys.stderr,
        )
        sys.exit(error.returncode if error.returncode > 0 else 1)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.exit(1, f'{error}\n')
    except KeyboardInterrupt:
        sys.exit(130)
