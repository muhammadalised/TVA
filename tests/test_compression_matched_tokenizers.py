"""Regression checks for the compression-matched IAM text-only ED controls."""
from argparse import Namespace
import json
from pathlib import Path
import unittest

import yaml

from main import load_configured_tokenizer
from tva.compression_matched_tokenizers import (
    ARTIFACT_ROOT, FAMILIES, bpe_merges, build_matched_model, characters_per_token,
    greedy_bigram_split, iam_training_texts, matched_size, ranked_pieces,
    unigram_compression_curve, unigram_em_model,
)
from tva.dataset.ed import ED_ALPHABET
from tva.tokenizer_factory import get_tokenizer
from tva.tokenizers import BigramTokenizer, BPETokenizer, UnigramTokenizer

ROOT = Path(__file__).resolve().parents[1]
CLASSES = {'bigram': BigramTokenizer, 'bpe': BPETokenizer, 'unigram': UnigramTokenizer}
EXPECTED_K = {'bigram': 62, 'bpe': 59, 'unigram': 55}
# Unigram compression moves in coarser steps: every budget is a new EM training.
TOLERANCE = {'bigram': 0.002, 'bpe': 0.002, 'unigram': 0.003}


def load(kind, fold=0):
    tokenizer = get_tokenizer(kind)
    tokenizer.load(str(ARTIFACT_ROOT / FAMILIES[kind]['output'] / f'{fold}.json'))
    return tokenizer


def load_role(kind, role):
    key, path = FAMILIES[kind][role]
    tokenizer = get_tokenizer(key)
    tokenizer.load(str(ARTIFACT_ROOT / path))
    return tokenizer


class CompressionMatchedTokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.texts = iam_training_texts()

    def test_frozen_files_are_reproduced_exactly(self):
        for kind in ('bigram', 'bpe'):  # Unigram: see the next test
            with self.subTest(kind=kind):
                directory = ARTIFACT_ROOT / FAMILIES[kind]['output']
                content, record = build_matched_model(kind)
                self.assertEqual(record['subword_count'], EXPECTED_K[kind])
                self.assertEqual(json.loads((directory / 'selection.json').read_text()), record)
                for fold in range(5):
                    self.assertEqual((directory / f'{fold}.json').read_bytes(), content)

    def test_unigram_control_is_the_text_only_em_at_the_recorded_budget(self):
        # Rebuilding retrains the EM for all 146 budgets (minutes), so the
        # stored curve is checked around k and the file is retrained at k.
        directory = ARTIFACT_ROOT / FAMILIES['unigram']['output']
        record = json.loads((directory / 'selection.json').read_text())
        k = record['subword_count']
        self.assertEqual(k, EXPECTED_K['unigram'])
        curve = record['selection']['iam_characters_per_token_by_k']
        self.assertEqual(len(curve), 146)
        target = characters_per_token(load_role('unigram', 'partner'), self.texts)
        self.assertEqual(round(target, 6), record['selection']['iam_characters_per_token']['handwriting_partner'])
        self.assertEqual(matched_size(curve, target), k)
        self.assertEqual([round(c, 6) for c in unigram_compression_curve(self.texts, [k - 1, k, k + 1])],
                         curve[k - 1:k + 2])
        content = unigram_em_model(self.texts, k).to_str(pretty=True).encode('utf-8')
        for fold in range(5):
            self.assertEqual((directory / f'{fold}.json').read_bytes(), content)
        # The full budget is the frozen text-only Unigram EM itself.
        source = (ARTIFACT_ROOT / FAMILIES['unigram']['source'][1]).read_bytes()
        self.assertEqual(unigram_em_model(self.texts, 145).to_str(pretty=True).encode('utf-8'), source)

    def test_vocabularies_are_prefixes_of_the_text_only_sources(self):
        bigram, source = load('bigram'), load_role('bigram', 'source')
        self.assertIs(type(source), BigramTokenizer)
        ranked = ranked_pieces('bigram', source)
        self.assertEqual(len(ranked), 145)
        self.assertEqual([t for t in sorted(bigram.vocab, key=bigram.vocab.get) if len(t) == 2],
                         ranked[:EXPECTED_K['bigram']])
        # Same IDs as the source for the characters and the kept bigrams.
        self.assertEqual(bigram.vocab, {t: i for t, i in source.vocab.items() if i < bigram.size})
        bpe = load('bpe')
        self.assertEqual(bpe_merges(bpe), bpe_merges(load_role('bpe', 'source'))[:EXPECTED_K['bpe']])
        for kind, cls in CLASSES.items():
            with self.subTest(kind=kind):
                tokenizer = load(kind)
                self.assertIs(type(tokenizer), cls)
                self.assertEqual(tokenizer.size, 1 + len(ED_ALPHABET) + EXPECTED_K[kind])

    def test_bigram_control_encodes_with_greedy_tva_splitting(self):
        bigram = load('bigram')
        pieces = {t for t in bigram.vocab if len(t) == 2}
        for text in self.texts[:500]:
            self.assertEqual([bigram.idx_token[i] for i in bigram.encode(text)], greedy_bigram_split(text, pieces))

    def test_compression_matches_the_handwriting_partner_on_iam(self):
        for kind in FAMILIES:
            with self.subTest(kind=kind):
                matched = characters_per_token(load(kind), self.texts)
                partner = characters_per_token(load_role(kind, 'partner'), self.texts)
                self.assertAlmostEqual(matched, partner, delta=TOLERANCE[kind])

    def test_round_trip_without_blank(self):
        for kind in FAMILIES:
            tokenizer = load(kind)
            for text in ['', ''.join(ED_ALPHABET), 'the  handwriting! 100% = 9 + 3?', *self.texts[:200]]:
                ids = tokenizer.encode(text)
                self.assertNotIn(0, ids)
                self.assertEqual(tokenizer.decode(ids), text)

    def test_configs_load_the_matched_tokenizers(self):
        for kind in FAMILIES:
            for split in ('wd', 'wi'):
                path = ROOT / 'configs/thesis/ed/compression_matched' / f'{kind}_linguistic_matched_{split}.yaml'
                config = yaml.safe_load(path.read_text())
                partner = yaml.safe_load((ROOT / 'configs/thesis/ed/subword_comparison' / f'bpe_linguistic_{split}.yaml').read_text())
                differing = {key for key in config if config[key] != partner.get(key)}
                self.assertEqual(differing - {'tokenizer'}, {'dir_tokenizer', 'dir_work'})
                self.assertEqual(config['tokenizer'], kind)
                tokenizer = load_configured_tokenizer(Namespace(**config))
                self.assertEqual(tokenizer.size, 1 + len(ED_ALPHABET) + EXPECTED_K[kind])


if __name__ == '__main__':
    unittest.main()
