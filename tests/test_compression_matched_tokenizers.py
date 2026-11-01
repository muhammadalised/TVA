"""Regression checks for the compression-matched IAM text-only ED controls."""
from argparse import Namespace
import json
from pathlib import Path
import tempfile
import unittest

import yaml

from main import load_configured_tokenizer
from tva.compression_matched_tokenizers import (
    ARTIFACT_ROOT, FAMILIES, MAX_PIECES, MIN_PIECES, NUM_CHARACTERS, characters_per_token,
    greedy_bigram_split, iam_training_texts, matched_size, train_tva, write_iam_dataset,
)
from tva.dataset.ed import ED_ALPHABET
from tva.tokenizers import BigramTokenizer, BPETokenizer, UnigramTokenizer
from tests.hash_seed import call_with_hash_seed

ROOT = Path(__file__).resolve().parents[1]
CLASSES = {'bigram': BigramTokenizer, 'bpe': BPETokenizer, 'unigram': UnigramTokenizer}
EXPECTED_K = {'bigram': 62, 'bpe': 71, 'unigram': 36}
# Unigram compression moves in coarser steps: every budget is a new training.
TOLERANCE = {'bigram': 0.002, 'bpe': 0.002, 'unigram': 0.003}
FULL = {'bigram': 'ed/linguistic/bigram_letters', 'bpe': 'ed/linguistic/bpe', 'unigram': 'ed/linguistic/unigram'}


def load(kind, directory, fold=0):
    tokenizer = CLASSES[kind]()
    tokenizer.load(str(ARTIFACT_ROOT / directory / f'{fold}.json'))
    return tokenizer


def pieces(tokenizer):
    if isinstance(tokenizer, BigramTokenizer):
        return [t for t in sorted(tokenizer.vocab, key=tokenizer.vocab.get) if len(t) == 2]
    return sorted(t for t in tokenizer.tokenizer.get_vocab() if len(t) > 1 and t != '<PAD>')


class CompressionMatchedTokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.texts = iam_training_texts()
        cls.directory = tempfile.TemporaryDirectory()
        write_iam_dataset(cls.directory.name)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def record(self, kind):
        return json.loads((ARTIFACT_ROOT / FAMILIES[kind]['output'] / 'selection.json').read_text())

    def test_recorded_k_is_the_closest_compression(self):
        # Rebuilding scans all 145 sizes (minutes), so the stored curve is
        # checked around k and the file is retrained at k.
        for kind in FAMILIES:
            with self.subTest(kind=kind):
                record = self.record(kind)
                k = record['subword_count']
                self.assertEqual(k, EXPECTED_K[kind])
                self.assertEqual(record['classes'], NUM_CHARACTERS + k)
                stored = record['selection']['iam_characters_per_token_by_k']
                self.assertEqual(stored['first_k'], MIN_PIECES)
                curve = dict(enumerate(stored['values'], start=MIN_PIECES))
                self.assertEqual(max(curve), MAX_PIECES)
                partner = load(kind, FAMILIES[kind]['partner'])
                target = characters_per_token(partner, self.texts)
                self.assertEqual(round(target, 6), record['selection']['iam_characters_per_token']['handwriting_partner'])
                self.assertEqual(matched_size(curve, target), k)
                for size in (k - 1, k, k + 1):
                    tokenizer, _ = train_tva(kind, self.directory.name, size)
                    measured = round(characters_per_token(tokenizer, self.texts), 6)
                    self.assertAlmostEqual(measured, curve[size], delta=0 if kind != 'unigram' else 1e-3)

    def test_frozen_files_are_tva_trained_at_k(self):
        for kind in FAMILIES:
            with self.subTest(kind=kind):
                directory = ARTIFACT_ROOT / FAMILIES[kind]['output']
                tokenizer, files = call_with_hash_seed('tva.compression_matched_tokenizers:train_tva',
                                                       kind, self.directory.name, EXPECTED_K[kind])
                if kind == 'unigram':
                    # Hugging Face's Unigram trainer is not bit-reproducible.
                    self.assertEqual(pieces(tokenizer), pieces(load(kind, FAMILIES[kind]['output'])))
                    continue
                for fold in range(5):
                    for name, content in files.items():
                        self.assertEqual((directory / name.replace('0.', f'{fold}.', 1)).read_bytes(), content)

    def test_bigram_and_bpe_controls_are_prefixes_of_tva(self):
        bigram = load('bigram', FAMILIES['bigram']['output'])
        self.assertEqual(pieces(bigram), pieces(load('bigram', FULL['bigram']))[:EXPECTED_K['bigram']])
        merges = lambda t: json.loads(t.tokenizer.to_str())['model']['merges']
        bpe = load('bpe', FAMILIES['bpe']['output'])
        self.assertEqual(merges(bpe), merges(load('bpe', FULL['bpe']))[:EXPECTED_K['bpe']])
        for kind, cls in CLASSES.items():
            with self.subTest(kind=kind):
                tokenizer = load(kind, FAMILIES[kind]['output'])
                self.assertIs(type(tokenizer), cls)
                self.assertEqual(tokenizer.size, 1 + len(ED_ALPHABET) + EXPECTED_K[kind])

    def test_bigram_control_encodes_with_greedy_tva_splitting(self):
        bigram = load('bigram', FAMILIES['bigram']['output'])
        bigrams = set(pieces(bigram))
        for text in self.texts[:500]:
            self.assertEqual([bigram.idx_token[i] for i in bigram.encode(text)], greedy_bigram_split(text, bigrams))

    def test_compression_matches_the_handwriting_partner_on_iam(self):
        for kind in FAMILIES:
            with self.subTest(kind=kind):
                matched = characters_per_token(load(kind, FAMILIES[kind]['output']), self.texts)
                partner = characters_per_token(load(kind, FAMILIES[kind]['partner']), self.texts)
                self.assertAlmostEqual(matched, partner, delta=TOLERANCE[kind])

    def test_round_trip_without_blank(self):
        for kind in FAMILIES:
            tokenizer = load(kind, FAMILIES[kind]['output'])
            for text in ['', ''.join(ED_ALPHABET), 'the  handwriting! 100% = 9 + 3?', *self.texts[:200]]:
                ids = tokenizer.encode(text)
                self.assertNotIn(0, ids)
                self.assertEqual(tokenizer.decode(ids), text)

    def test_configs_load_the_matched_tokenizers(self):
        for kind in FAMILIES:
            for split in ('wd', 'wi'):
                path = ROOT / 'configs/thesis/ed/compression_matched' / f'{kind}_linguistic_matched_{split}.yaml'
                config = yaml.safe_load(path.read_text())
                self.assertEqual(config['tokenizer'], kind)
                self.assertEqual(config['dir_tokenizer'], f"artifacts/tokenizers/{FAMILIES[kind]['output']}")
                tokenizer = load_configured_tokenizer(Namespace(**config))
                self.assertEqual(tokenizer.size, 1 + len(ED_ALPHABET) + EXPECTED_K[kind])


if __name__ == '__main__':
    unittest.main()
