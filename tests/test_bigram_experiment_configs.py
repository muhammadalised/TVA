import unittest
from pathlib import Path

import yaml

from tva.model import BaseModel
from tva.tokenizers import get_tokenizer, resolve_tokenizer_path


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / 'configs/thesis'

CONFIGS = {
    'handwriting_wd': CONFIG_ROOT / 'bigram_handwriting_wd_rh.yaml',
    'handwriting_wi': CONFIG_ROOT / 'bigram_handwriting_wi_rh.yaml',
    'linguistic_wd': CONFIG_ROOT / 'bigram_linguistic_wd_rh.yaml',
    'linguistic_wi': CONFIG_ROOT / 'bigram_linguistic_wi_rh.yaml',
}

CONTROLLED_KEYS = (
    'arch_de',
    'arch_en',
    'aug',
    'cache',
    'checkpoint',
    'device',
    'epoch',
    'epoch_warmup',
    'freq_eval',
    'freq_log',
    'freq_save',
    'idx_fold',
    'len_seq',
    'lr',
    'num_channel',
    'num_concat',
    'num_worker',
    'seed',
    'size_batch',
    'test',
    'categories',
)


class BigramExperimentConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.configs = {
            name: yaml.safe_load(path.read_text(encoding='utf-8'))
            for name, path in CONFIGS.items()
        }

    def test_non_tokenizer_settings_are_matched(self) -> None:
        expected = {
            key: self.configs['handwriting_wd'][key]
            for key in CONTROLLED_KEYS
        }
        for name, config in self.configs.items():
            with self.subTest(config=name):
                self.assertEqual(
                    {key: config[key] for key in CONTROLLED_KEYS},
                    expected,
                )

    def test_each_condition_loads_a_419_class_head(self) -> None:
        for name, config in self.configs.items():
            with self.subTest(config=name):
                tokenizer = get_tokenizer(config['tokenizer'])
                tokenizer.load(
                    resolve_tokenizer_path(
                        str(REPO_ROOT / config['dir_tokenizer']),
                        config['idx_fold'],
                    )
                )
                self.assertEqual(tokenizer.size, 419)

                model = BaseModel(
                    config['arch_en'],
                    config['arch_de'],
                    config['num_channel'],
                    tokenizer.size,
                    config['len_seq'],
                )
                self.assertEqual(model.decoder.fc.out_features, 419)

    def test_dataset_and_tokenizer_pairings_are_explicit(self) -> None:
        for name, config in self.configs.items():
            with self.subTest(config=name):
                split = 'wd' if '_wd' in name else 'wi'
                self.assertIn(f'_{split}_word_rh', config['dir_dataset'])

                if name.startswith('handwriting'):
                    self.assertEqual(config['tokenizer'], 'handwriting_bigram')
                    self.assertTrue(config['dir_tokenizer'].endswith('.json'))
                else:
                    self.assertEqual(config['tokenizer'], 'linguistic_bigram')
                    self.assertIn(
                        f'onhw_words500_{split}_word_rh',
                        config['dir_tokenizer'],
                    )


if __name__ == '__main__':
    unittest.main()
