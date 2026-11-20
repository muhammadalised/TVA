'''Launch one condition of the fold-0 Bigram v0 recognition pilot.'''

import argparse
from pathlib import Path

import yaml

from main import main


BASE_CONFIG = Path('configs/thesis/b0_char_wi_rh.yaml')
BIGRAM_VOCABS = Path(
    'results/thesis/forced_alignment_review/wi_rh_fold0_20261117_bigram_v0'
)
PILOT_RESULTS = Path('results/thesis/bigram/wi_rh_fold0_bigram_v0_pilot')
CONDITIONS = ('char', 'frequency', 'force')


def pilot_config(condition: str, seed: int) -> argparse.Namespace:
    config = yaml.safe_load(BASE_CONFIG.read_text(encoding='utf-8'))
    if condition != 'char':
        config['tokenizer'] = 'bigram'
        config['dir_tokenizer'] = str(BIGRAM_VOCABS / condition)
    config['dir_work'] = str(PILOT_RESULTS / f'seed_{seed}' / condition)
    config['seed'] = seed
    config['train_only'] = True
    return argparse.Namespace(**config)


def launch() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('condition', choices=CONDITIONS)
    parser.add_argument('--seed', type=int, default=42, help='Matched seed (default: 42).')
    args = parser.parse_args()
    cfg = pilot_config(args.condition, args.seed)
    vocab = Path(cfg.dir_tokenizer) / f'{cfg.idx_fold}.json'
    if not vocab.is_file():
        parser.error(f'Missing {vocab}. See the build command in docs/FORCED_ALIGNMENT.md.')
    try:
        Path(cfg.dir_work).mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f'This run directory already exists: {cfg.dir_work}')
    main(cfg)


if __name__ == '__main__':
    launch()
