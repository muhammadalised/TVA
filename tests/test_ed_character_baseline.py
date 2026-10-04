import argparse
from pathlib import Path
import unittest

import yaml

from main import load_configured_tokenizer
from tva.handwriting.ed import ED_ALPHABET
from tva.model import BaseModel
from tva.handwriting_tokenizers import CharacterTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / 'configs/thesis/ed'


class EdCharacterBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.configs = {
            distribution: yaml.safe_load(
                (CONFIG_ROOT / f'character_{distribution}.yaml').read_text(
                    encoding='utf-8'
                )
            )
            for distribution in ('wd', 'wi')
        }
        cls.bigram_configs = {
            distribution: yaml.safe_load(
                (
                    CONFIG_ROOT
                    / f'bigram_handwriting_{distribution}.yaml'
                ).read_text(encoding='utf-8')
            )
            for distribution in ('wd', 'wi')
        }

    def test_fold_zero_protocol_is_matched_to_actual_bigram_runs(self):
        for distribution, config in self.configs.items():
            with self.subTest(distribution=distribution):
                expected = dict(self.bigram_configs[distribution])
                expected.update(
                    {
                        'dir_tokenizer': None,
                        'dir_work': (
                            f'results/thesis/ed/character_{distribution}'
                        ),
                        'size_batch': 32,
                        'tokenizer': 'char',
                    }
                )
                self.assertEqual(config, expected)
                self.assertEqual(config['idx_fold'], 0)
                self.assertEqual(config['ed_distribution'], distribution)

    def test_configured_alphabet_has_blank_plus_78_characters(self):
        for distribution, config in self.configs.items():
            with self.subTest(distribution=distribution):
                cfgs = argparse.Namespace(**config)
                tokenizer = load_configured_tokenizer(cfgs)
                self.assertIsInstance(tokenizer, CharacterTokenizer)
                self.assertEqual(config['categories'], ['', *ED_ALPHABET])
                self.assertEqual(tokenizer.size, 79)
                self.assertEqual(tokenizer.vocab[''], 0)
                text = 'Hello, ED!'
                encoded = tokenizer.encode(text)
                self.assertNotIn(0, encoded)
                self.assertEqual(tokenizer.decode(encoded), text)

    def test_model_output_head_matches_character_vocabulary(self):
        for distribution, config in self.configs.items():
            with self.subTest(distribution=distribution):
                cfgs = argparse.Namespace(**config)
                tokenizer = load_configured_tokenizer(cfgs)
                model = BaseModel(
                    cfgs.arch_en,
                    cfgs.arch_de,
                    cfgs.num_channel,
                    tokenizer.size,
                    cfgs.len_seq,
                )
                self.assertEqual(model.decoder.fc.out_features, 79)

    def test_character_categories_reject_invalid_blank_and_duplicates(self):
        tokenizer = CharacterTokenizer()
        for categories in (['a'], ['', 'a', 'a'], ['', '', 'a']):
            with self.subTest(categories=categories):
                with self.assertRaises(ValueError):
                    tokenizer.load_categories(categories)


if __name__ == '__main__':
    unittest.main()
