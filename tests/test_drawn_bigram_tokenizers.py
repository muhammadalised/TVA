"""Regression checks for the size- and compression-matched IAM text-only bigram draws."""
from argparse import Namespace
import json
from pathlib import Path
import unittest

import yaml

from main import load_configured_tokenizer
from tva.compression_matched_tokenizers import ARTIFACT_ROOT, characters_per_token, greedy_bigram_split, iam_training_texts
from tva.dataset.ed import ED_ALPHABET
from tva.drawn_bigram_tokenizers import NUM_DRAWS, OUTPUT, PARTNER, TOLERANCE, candidate_pool
from tva.handwriting.common import canonical_json_bytes
from tests.hash_seed import call_with_hash_seed
from tva.tokenizer_factory import get_tokenizer
from tva.tokenizers import BigramTokenizer

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ARTIFACT_ROOT / OUTPUT


def load(key, path):
    tokenizer = get_tokenizer(key)
    tokenizer.load(str(path))
    return tokenizer


class DrawnBigramTokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.texts = iam_training_texts()
        cls.partner = load('handwriting_bigram_greedy', ARTIFACT_ROOT / PARTNER)
        cls.draws = [load('bigram', DIRECTORY / f'draw{d}' / '0.json') for d in range(NUM_DRAWS)]

    def test_frozen_files_are_reproduced_exactly(self):
        models, summary = call_with_hash_seed('tva.drawn_bigram_tokenizers:build_drawn_models')
        self.assertEqual((DIRECTORY / 'draws.json').read_bytes(), canonical_json_bytes(summary))
        for index, (files, record) in enumerate(models):
            with self.subTest(draw=index):
                self.assertEqual(set(files), {'0.json', '0.scores.json'})
                self.assertEqual(json.loads((DIRECTORY / f'draw{index}' / 'selection.json').read_text()), record)
                for fold in range(5):
                    for name, content in files.items():
                        path = DIRECTORY / f'draw{index}' / name.replace('0.', f'{fold}.', 1)
                        self.assertEqual(path.read_bytes(), content)

    def test_built_like_the_handwriting_bigram(self):
        pool = set(candidate_pool())
        partner_bigrams = {t for t in self.partner.vocab if len(t) == 2}
        for index, tokenizer in enumerate(self.draws):
            with self.subTest(draw=index):
                self.assertIs(type(tokenizer), BigramTokenizer)
                self.assertEqual(tokenizer.size, self.partner.size)
                bigrams = [t for t in sorted(tokenizer.vocab, key=tokenizer.vocab.get) if len(t) == 2]
                self.assertEqual(len(bigrams), len(partner_bigrams))
                # TVA's trainer numbers the allowed bigrams by text frequency.
                self.assertEqual(list(tokenizer.scores), bigrams)
                self.assertLessEqual(set(bigrams), pool)
                self.assertNotEqual(set(bigrams), partner_bigrams)
                # Characters keep the handwriting bigram's (and TVA's) IDs.
                self.assertEqual({t: i for t, i in tokenizer.vocab.items() if len(t) < 2},
                                 {t: i for t, i in self.partner.vocab.items() if len(t) < 2})
        self.assertEqual(len({tuple(sorted(t.vocab)) for t in self.draws}), NUM_DRAWS)

    def test_compression_matches_the_handwriting_partner_on_iam(self):
        target = characters_per_token(self.partner, self.texts)
        for index, tokenizer in enumerate(self.draws):
            with self.subTest(draw=index):
                self.assertAlmostEqual(characters_per_token(tokenizer, self.texts), target, delta=TOLERANCE)

    def test_encodes_with_greedy_tva_splitting_and_round_trips(self):
        for tokenizer in self.draws:
            pieces = {t for t in tokenizer.vocab if len(t) == 2}
            for text in self.texts[:300]:
                self.assertEqual([tokenizer.idx_token[i] for i in tokenizer.encode(text)],
                                 greedy_bigram_split(text, pieces))
            for text in ['', ''.join(ED_ALPHABET), 'the  handwriting! 100% = 9 + 3?']:
                ids = tokenizer.encode(text)
                self.assertNotIn(0, ids)
                self.assertEqual(tokenizer.decode(ids), text)

    def test_configs_load_the_draws(self):
        for index in range(NUM_DRAWS):
            for split in ('wd', 'wi'):
                with self.subTest(draw=index, split=split):
                    path = ROOT / 'configs/thesis/ed/size_and_compression_matched' / f'bigram_linguistic_drawn{index}_{split}.yaml'
                    config = yaml.safe_load(path.read_text())
                    self.assertEqual(config['tokenizer'], 'bigram')
                    tokenizer = load_configured_tokenizer(Namespace(**config))
                    self.assertEqual(tokenizer.vocab, self.draws[index].vocab)


if __name__ == '__main__':
    unittest.main()
