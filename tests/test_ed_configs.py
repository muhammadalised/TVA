"""Every ED condition trains on the same data with the same settings."""
from pathlib import Path
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / 'configs/thesis/ed'
CONFIGS = sorted(CONFIG_ROOT.rglob('*.yaml'))
EXPERIMENTS = {'character', 'subword_comparison', 'letters_only', 'compression_matched',
               'size_and_compression_matched'}
# What a condition may change: its tokenizer and where its results go.
PER_CONDITION = {'tokenizer', 'dir_tokenizer', 'dir_work'}
# Data location follows the split.
PER_SPLIT = {'dir_dataset', 'ed_distribution'}


def split_of(path):
    return next(s for s in ('wd', 'wi') if f'_{s}' in path.stem)


class EdConfigTests(unittest.TestCase):
    def test_every_config_belongs_to_one_experiment_folder(self):
        for path in CONFIGS:
            with self.subTest(config=path.name):
                self.assertIn(path.relative_to(CONFIG_ROOT).parts[0], EXPERIMENTS)
                self.assertEqual(len(path.relative_to(CONFIG_ROOT).parts), 2)
        self.assertEqual(len({path.name for path in CONFIGS}), len(CONFIGS))

    def test_every_condition_reads_the_extracted_files_of_its_split(self):
        self.assertGreater(len(CONFIGS), 20)
        for path in CONFIGS:
            with self.subTest(config=path.name):
                config = yaml.safe_load(path.read_text())
                split = split_of(path)
                self.assertEqual(config['dataset_format'], 'json-directory')
                self.assertEqual(config['dir_dataset'], f'data/tva/ed/gold_{split}')
                self.assertEqual(config['ed_distribution'], split)

    def test_conditions_differ_only_in_tokenizer_and_output(self):
        configs = {path.name: yaml.safe_load(path.read_text()) for path in CONFIGS}
        reference = configs['character_wd.yaml']
        for name, config in configs.items():
            with self.subTest(config=name):
                ignored = PER_CONDITION | PER_SPLIT
                if name.endswith('_cv.yaml'):
                    ignored = ignored | {'idx_fold'}  # all folds via train_cv.py
                differing = {k for k in reference.keys() | config.keys()
                             if k not in ignored and config.get(k) != reference.get(k)}
                self.assertEqual(differing, set())

    def test_result_directories_are_unique(self):
        work = [yaml.safe_load(path.read_text())['dir_work'] for path in CONFIGS]
        self.assertEqual(len(work), len(set(work)))


if __name__ == '__main__':
    unittest.main()
