import argparse
from pathlib import Path
import unittest
import json
import tempfile

import yaml

from main import load_configured_tokenizer
from tva.handwriting.ed import ED_ALPHABET
from tva.model import BaseModel
from tva.tokenizers import CharacterTokenizer
from tva.tokenizer_factory import get_tokenizer


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
                self.assertIs(type(tokenizer), CharacterTokenizer)
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

    def test_saved_ed_configs_reject_categories_that_change_character_ids(self):
        for categories in (['a'], ['', 'a', 'a'], ['', '', 'a']):
            with self.subTest(categories=categories):
                cfgs = argparse.Namespace(**dict(self.configs['wd'], categories=categories))
                with self.assertRaisesRegex(ValueError, 'character categories differ'):
                    load_configured_tokenizer(cfgs)

    def test_factory_uses_original_class_and_original_training_reproduces_file(self):
        self.assertIs(type(get_tokenizer('char')), CharacterTokenizer)
        artifact = REPO_ROOT / 'artifacts/tokenizers/ed/character/ed_characters_v1.json'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'train.json').write_text(json.dumps({'info': {'num_fold': 1}}))
            tokenizer = CharacterTokenizer()
            tokenizer.train(directory, ['', *ED_ALPHABET])
            self.assertEqual((root / 'tokenizers/char/0.json').read_bytes(), artifact.read_bytes())

    def test_explicit_tva_configs_match_existing_baseline_ids_and_settings(self):
        for distribution in ('wd', 'wi'):
            explicit = yaml.safe_load((CONFIG_ROOT / f'character_tva_{distribution}.yaml').read_text())
            existing = yaml.safe_load((CONFIG_ROOT / f'character_{distribution}_extracted_cv.yaml').read_text())
            excluded = {'dir_tokenizer', 'dir_work', 'idx_fold'}
            self.assertEqual({k: v for k, v in explicit.items() if k not in excluded},
                             {k: v for k, v in existing.items() if k not in excluded})
            original = load_configured_tokenizer(argparse.Namespace(**explicit))
            previous = load_configured_tokenizer(argparse.Namespace(**existing))
            self.assertIs(type(original), CharacterTokenizer)
            self.assertEqual(original.vocab, previous.vocab)
            self.assertEqual(original.idx_char, previous.idx_char)
            self.assertEqual(original.encode(''.join(ED_ALPHABET)), list(range(1, 79)))


if __name__ == '__main__':
    unittest.main()
