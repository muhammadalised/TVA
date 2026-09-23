import argparse
import os
from pathlib import Path
from unittest.mock import patch
import unittest

import torch
import yaml

from main import build_dataset
from tva.dataset import FauZipDataset, fn_collate, get_fau_num_folds
from tva.fau_handwriting_bigram import FAU_ALPHABET
from tva.model import BaseModel
from tva.tokenizers import get_tokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / 'configs/thesis/fau'
FAU_DIRECTORY = Path('/mnt/c/Users/Ali/Downloads/fau-english-dataset')
SMOKE_CONFIGS = {
    f'{kind}_{distribution}': CONFIG_ROOT / f'smoke_bigram_{kind}_{distribution}.yaml'
    for kind in ('handwriting', 'linguistic')
    for distribution in ('wd', 'wi')
}
PRODUCTION_CONFIGS = {
    f'{kind}_{distribution}': CONFIG_ROOT / f'bigram_{kind}_{distribution}.yaml'
    for kind in ('handwriting', 'linguistic')
    for distribution in ('wd', 'wi')
}
TIMING_CONFIG = CONFIG_ROOT / 'timing_bigram_handwriting_wd.yaml'
MATCHED_KEYS = (
    'arch_de',
    'arch_en',
    'aug',
    'cache',
    'checkpoint',
    'dataset_format',
    'device',
    'dir_dataset',
    'epoch',
    'epoch_warmup',
    'freq_eval',
    'freq_log',
    'freq_save',
    'idx_fold',
    'len_seq',
    'lr',
    'max_train_samples',
    'max_val_samples',
    'num_channel',
    'num_concat',
    'num_worker',
    'seed',
    'size_batch',
    'test',
    'categories',
)


class FauSmokeConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.configs = {
            name: yaml.safe_load(path.read_text(encoding='utf-8'))
            for name, path in SMOKE_CONFIGS.items()
        }

    def test_all_four_smoke_conditions_are_strictly_matched(self):
        expected = {
            key: self.configs['handwriting_wd'][key] for key in MATCHED_KEYS
        }
        for name, config in self.configs.items():
            with self.subTest(config=name):
                self.assertEqual(
                    {key: config[key] for key in MATCHED_KEYS}, expected
                )
                expected_distribution = 'wd' if name.endswith('_wd') else 'wi'
                self.assertEqual(
                    config['fau_distribution'], expected_distribution
                )

    def test_each_config_builds_a_seven_channel_224_output_model(self):
        for name, config in self.configs.items():
            with self.subTest(config=name):
                tokenizer = get_tokenizer(config['tokenizer'])
                tokenizer.load(REPO_ROOT / config['dir_tokenizer'])
                model = BaseModel(
                    config['arch_en'],
                    config['arch_de'],
                    config['num_channel'],
                    tokenizer.size,
                    config['len_seq'],
                )
                self.assertEqual(config['num_channel'], 7)
                self.assertEqual(config['categories'], ['', *FAU_ALPHABET])
                self.assertEqual(tokenizer.size, 224)
                self.assertEqual(model.decoder.fc.out_features, 224)

    def test_tokenizer_pairings_are_explicit(self):
        for name, config in self.configs.items():
            expected = (
                'handwriting_bigram_greedy'
                if name.startswith('handwriting')
                else 'linguistic_bigram_greedy'
            )
            self.assertEqual(config['tokenizer'], expected)
            self.assertIn(name.split('_')[0], config['dir_tokenizer'])


class FauProductionConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.configs = {
            name: yaml.safe_load(path.read_text(encoding='utf-8'))
            for name, path in PRODUCTION_CONFIGS.items()
        }

    def test_all_four_production_conditions_are_strictly_matched(self):
        excluded = {
            'dir_tokenizer',
            'dir_work',
            'fau_distribution',
            'tokenizer',
        }
        reference = {
            key: value
            for key, value in self.configs['handwriting_wd'].items()
            if key not in excluded
        }
        for name, config in self.configs.items():
            with self.subTest(config=name):
                self.assertEqual(
                    {
                        key: value
                        for key, value in config.items()
                        if key not in excluded
                    },
                    reference,
                )
                self.assertEqual(
                    config['fau_distribution'],
                    'wd' if name.endswith('_wd') else 'wi',
                )

    def test_production_settings_and_tokenizers_are_frozen(self):
        for name, config in self.configs.items():
            with self.subTest(config=name):
                expected_tokenizer = (
                    'handwriting_bigram_greedy'
                    if name.startswith('handwriting')
                    else 'linguistic_bigram_greedy'
                )
                tokenizer = get_tokenizer(config['tokenizer'])
                tokenizer.load(REPO_ROOT / config['dir_tokenizer'])
                model = BaseModel(
                    config['arch_en'],
                    config['arch_de'],
                    config['num_channel'],
                    tokenizer.size,
                    config['len_seq'],
                )
                self.assertEqual(config['arch_en'], 'blconv_b')
                self.assertEqual(config['arch_de'], 'bilstm_b')
                self.assertEqual(config['num_channel'], 7)
                self.assertEqual(config['categories'], ['', *FAU_ALPHABET])
                self.assertEqual(config['tokenizer'], expected_tokenizer)
                self.assertEqual(tokenizer.size, 224)
                self.assertEqual(model.decoder.fc.out_features, 224)
                self.assertEqual(config['epoch'], 300)
                self.assertEqual(config['epoch_warmup'], 30)
                self.assertEqual(config['size_batch'], 8)
                self.assertTrue(config['aug'])
                self.assertIsNone(config['checkpoint'])

    def test_timing_config_matches_production_where_it_matters(self):
        timing = yaml.safe_load(TIMING_CONFIG.read_text(encoding='utf-8'))
        production = self.configs['handwriting_wd']
        intentional_differences = {
            'dir_work',
            'epoch',
            'epoch_warmup',
            'freq_save',
        }
        self.assertEqual(
            {
                key: value
                for key, value in timing.items()
                if key not in intentional_differences
            },
            {
                key: value
                for key, value in production.items()
                if key not in intentional_differences
            },
        )
        self.assertEqual(timing['epoch'], 1)
        self.assertEqual(timing['epoch_warmup'], 1)
        self.assertEqual(timing['freq_save'], 0)


class FauZipDatasetTests(unittest.TestCase):
    def setUp(self):
        if not all(
            (FAU_DIRECTORY / f'gold_{distribution}.zip').exists()
            for distribution in ('wd', 'wi')
        ):
            self.skipTest('local FAU archives are not available')

    def test_environment_override_and_fold_metadata(self):
        with patch.dict(os.environ, {'TVA_FAU_DATASET_DIR': str(FAU_DIRECTORY)}):
            self.assertEqual(get_fau_num_folds('unused', 'wd'), 5)
            self.assertEqual(get_fau_num_folds('unused', 'wi'), 5)

    def test_loader_reads_and_collates_seven_channel_samples(self):
        config = self._load_config('handwriting_wd')
        tokenizer = get_tokenizer(config.tokenizer)
        tokenizer.load(REPO_ROOT / config.dir_tokenizer)
        with patch.dict(os.environ, {'TVA_FAU_DATASET_DIR': str(FAU_DIRECTORY)}):
            dataset = build_dataset(config, 'train', tokenizer, ratio_ds=8)
        self.assertIsInstance(dataset, FauZipDataset)
        self.assertEqual(len(dataset), 8)
        first = dataset[0]
        second = dataset[1]
        self.assertEqual(first[0].shape[1], 7)
        self.assertEqual(
            tokenizer.decode(first[1].tolist()), dataset.annos[0]['label']
        )
        sequences, labels, sequence_lengths, label_lengths = fn_collate(
            [first, second]
        )
        self.assertEqual(sequences.shape[0:2], torch.Size([2, 7]))
        self.assertEqual(labels.shape[0], 2)
        self.assertTrue(torch.all(sequence_lengths > 0))
        self.assertTrue(torch.all(label_lengths > 0))

    def test_validation_split_is_bounded_independently(self):
        config = self._load_config('linguistic_wi')
        tokenizer = get_tokenizer(config.tokenizer)
        tokenizer.load(REPO_ROOT / config.dir_tokenizer)
        with patch.dict(os.environ, {'TVA_FAU_DATASET_DIR': str(FAU_DIRECTORY)}):
            dataset = build_dataset(config, 'val', tokenizer, ratio_ds=8)
        self.assertEqual(len(dataset), 4)
        self.assertEqual(dataset.distribution, 'wi')
        self.assertEqual(dataset.split, 'val')

    def _load_config(self, name: str) -> argparse.Namespace:
        return argparse.Namespace(
            **yaml.safe_load(SMOKE_CONFIGS[name].read_text())
        )


if __name__ == '__main__':
    unittest.main()
