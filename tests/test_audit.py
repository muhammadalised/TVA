"""The ED audit loads tokenizers by the same keys and paths as recognition configs."""
from argparse import Namespace
from pathlib import Path
import unittest

import yaml

from main import load_configured_tokenizer
from tva.dataset.ed import ED_ALPHABET
from tva.handwriting.audit import DEFAULT_DIR_TOKENIZER, DEFAULT_TOKENIZER, load_tokenizer

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted((ROOT / 'configs/thesis/ed').rglob('*.yaml'))
TEXT = 'In short, there will be on TV! ' + ''.join(ED_ALPHABET)


class AuditTokenizerTests(unittest.TestCase):
    def test_every_config_tokenizer_loads_like_training(self):
        for path in CONFIGS:
            with self.subTest(config=path.name):
                config = yaml.safe_load(path.read_text())
                # null is the standard character file (main.load_configured_tokenizer)
                directory = config['dir_tokenizer'] or 'artifacts/tokenizers/ed/character/char'
                audited = load_tokenizer(config['tokenizer'], ROOT / directory)
                trained = load_configured_tokenizer(Namespace(**config))
                self.assertIs(type(audited), type(trained))
                self.assertEqual(audited.size, trained.size)
                self.assertEqual(audited.encode(TEXT), trained.encode(TEXT))
                self.assertEqual(audited.decode(audited.encode(TEXT)), TEXT)

    def test_default_is_the_handwriting_bigram(self):
        tokenizer = load_tokenizer(DEFAULT_TOKENIZER, ROOT / DEFAULT_DIR_TOKENIZER)
        self.assertEqual(tokenizer.size, 224)
        self.assertEqual(tokenizer.policy, 'greedy')

    def test_archived_linguistic_bigram_keeps_its_key(self):
        path = ROOT / 'artifacts/tokenizers/archive/ed/linguistic/ed_iam_linguistic_bigram_greedy_v1.json'
        tokenizer = load_tokenizer('linguistic_bigram_greedy', path)
        self.assertEqual(tokenizer.size, 224)
        self.assertEqual(tokenizer.decode(tokenizer.encode(TEXT)), TEXT)

    def test_unknown_key_is_rejected(self):
        with self.assertRaises(ValueError):
            load_tokenizer('handwriting', ROOT / DEFAULT_DIR_TOKENIZER)


if __name__ == '__main__':
    unittest.main()
