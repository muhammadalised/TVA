'''Launch a fixed-budget OnHW fold-0 Bigram recognition pilot.'''

import argparse
from pathlib import Path

import yaml

from main import main
from tva.force_bigram import verify_matched_vocabs


BASE_CONFIG = Path('configs/thesis/b0_char_wi_rh.yaml')
BIGRAM_VOCABS = Path(
    'results/thesis/forced_alignment_review/wi_rh_fold0_20261117_bigram_v0'
)
PILOT_RESULTS = Path('results/thesis/bigram/wi_rh_fold0_bigram_v0_pilot')
MATCHED_CONFIGS = Path('configs/thesis/forced_alignment')
MATCHED_RESULTS = Path('results/thesis/bigram/bigram_matched_fold0')
CONDITIONS = ('char', 'frequency', 'force')


def pilot_config(
    condition: str, seed: int, *, setting: str = 'wi', matched: bool = False
) -> argparse.Namespace:
    if matched and condition != 'char':
        path = MATCHED_CONFIGS / setting / f'{condition}.yaml'
        config = yaml.safe_load(path.read_text(encoding='utf-8'))
        if seed != config['seed']:
            output = Path(config['dir_work'])
            config['dir_work'] = str(output.parent.parent / f'seed_{seed}' / output.name)
            config['seed'] = seed
        return argparse.Namespace(**config)

    base_config = BASE_CONFIG.with_name(f'b0_char_{setting}_rh.yaml')
    config = yaml.safe_load(base_config.read_text(encoding='utf-8'))
    output = MATCHED_RESULTS / setting if matched else PILOT_RESULTS
    if condition != 'char':
        config['tokenizer'] = 'bigram'
        config['dir_tokenizer'] = str(BIGRAM_VOCABS / condition)
    config['dir_work'] = str(output / f'seed_{seed}' / condition)
    config['seed'] = seed
    config['train_only'] = True
    return argparse.Namespace(**config)


def launch() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('condition', choices=CONDITIONS)
    parser.add_argument('--seed', type=int, default=42, help='Matched seed (default: 42).')
    parser.add_argument('--matched', action='store_true',
                        help='Use vocabularies with 100 emitted bigrams on each side.')
    parser.add_argument('--setting', choices=('wi', 'wd'), default='wi')
    parser.add_argument('--dry-run', action='store_true',
                        help='Check inputs and print the resolved config.')
    args = parser.parse_args()
    if args.setting == 'wd' and not args.matched:
        parser.error('The original Bigram v0 pilot was WI only. Use --matched for WD.')
    cfg = pilot_config(args.condition, args.seed, setting=args.setting, matched=args.matched)
    vocab = Path(cfg.dir_tokenizer) / f'{cfg.idx_fold}.json'
    if not vocab.is_file():
        parser.error(f'Missing {vocab}. See the build command in docs/FORCED_ALIGNMENT.md.')
    matched_bigram = getattr(cfg, 'verify_matched_bigrams', False)
    if args.dry_run:
        if matched_bigram:
            try:
                verify_matched_vocabs(cfg)
            except (OSError, KeyError, ValueError) as error:
                parser.error(f'{error}. See the matched-vocabulary setup in docs/FORCED_ALIGNMENT.md.')
        print(yaml.safe_dump(vars(cfg)), end='')
        return
    try:
        if not matched_bigram:
            Path(cfg.dir_work).mkdir(parents=True, exist_ok=False)
        main(cfg)
    except FileExistsError:
        parser.error(f'This run directory already exists: {cfg.dir_work}')


if __name__ == '__main__':
    launch()
