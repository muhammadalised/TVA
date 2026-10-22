"""Regression checks for the compression-matched IAM text-only ED controls."""
from argparse import Namespace
from pathlib import Path
import unittest

import yaml

from main import load_configured_tokenizer
from tva.compression_matched_tokenizers import (
    ARTIFACT_ROOT, FAMILIES, CompressionMatchedBigramTokenizer, CompressionMatchedBPETokenizer,
    build_matched_model, characters_per_token, iam_training_texts,
)
from tva.handwriting.common import canonical_json_bytes
from tva.handwriting.ed import ED_ALPHABET
from tva.linguistic_tokenizers import LinguisticBPETokenizer
from tva.tokenizer_factory import get_tokenizer

ROOT = Path(__file__).resolve().parents[1]
CLASSES = {'bigram': CompressionMatchedBigramTokenizer, 'bpe': CompressionMatchedBPETokenizer}
EXPECTED_K = {'bigram': 62, 'bpe': 59}


class CompressionMatchedTokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.texts = iam_training_texts()

    def load(self, kind, fold=0):
        tokenizer = get_tokenizer(f'linguistic_{kind}_matched')
        tokenizer.load(ARTIFACT_ROOT / FAMILIES[kind]['output'] / f'{fold}.json')
        return tokenizer

    def test_frozen_files_are_reproduced_exactly(self):
        for kind in FAMILIES:
            directory = ARTIFACT_ROOT / FAMILIES[kind]['output']
            content = canonical_json_bytes(build_matched_model(kind))
            for fold in range(5):
                self.assertEqual((directory / f'{fold}.json').read_bytes(), content)

    def test_vocabularies_are_prefixes_of_the_text_only_sources(self):
        for kind, cls in CLASSES.items():
            tokenizer = self.load(kind)
            self.assertIsInstance(tokenizer, cls)
            k = EXPECTED_K[kind]
            self.assertEqual(tokenizer.size, 1 + len(ED_ALPHABET) + k)
            self.assertEqual({c: tokenizer.vocab[c] for c in ['', *ED_ALPHABET]},
                             {c: i for i, c in enumerate(['', *ED_ALPHABET])})
        bigram = self.load('bigram')
        source = get_tokenizer('linguistic_bigram_greedy')
        source.load(ARTIFACT_ROOT / FAMILIES['bigram']['source'][1])
        ranked = [r['token'] for r in sorted(source.model['vocabulary'], key=lambda r: r['rank'])]
        self.assertEqual(bigram.model['pieces'], ranked[:EXPECTED_K['bigram']])
        bpe = self.load('bpe')
        text_bpe = LinguisticBPETokenizer()
        text_bpe.load(ARTIFACT_ROOT / FAMILIES['bpe']['source'][1])
        self.assertEqual(bpe.model['pieces'], text_bpe.source_model['merges'][:EXPECTED_K['bpe']])

    def test_compression_matches_the_handwriting_partner_on_iam(self):
        for kind in FAMILIES:
            partner = get_tokenizer(FAMILIES[kind]['partner'][0])
            partner.load(ARTIFACT_ROOT / FAMILIES[kind]['partner'][1])
            matched = characters_per_token(self.load(kind), self.texts)
            self.assertAlmostEqual(matched, characters_per_token(partner, self.texts), delta=0.002)

    def test_full_size_prefix_segments_like_the_source(self):
        # With every source piece kept, the runtime must equal the source tokenizer.
        for kind in FAMILIES:
            source = get_tokenizer(FAMILIES[kind]['source'][0])
            source.load(ARTIFACT_ROOT / FAMILIES[kind]['source'][1])
            control = self.load(kind)
            model = dict(control.model)
            if kind == 'bigram':
                model['pieces'] = [r['token'] for r in sorted(source.model['vocabulary'], key=lambda r: r['rank'])]
            else:
                model['pieces'] = source.source_model['merges']
                control.source_model = {'merges': model['pieces']}
            control.model = model
            for text in self.texts[:500]:
                self.assertEqual(control.segment(text)['tokens'],
                                 [source.idx_token[i] for i in source.encode(text)])

    def test_round_trip_without_blank(self):
        for kind in FAMILIES:
            tokenizer = self.load(kind)
            for text in ['', ''.join(ED_ALPHABET), 'the  handwriting! 100% = 9 + 3?', *self.texts[:200]]:
                ids = tokenizer.encode(text)
                self.assertNotIn(0, ids)
                self.assertEqual(tokenizer.decode(ids), text)
            with self.assertRaises(ValueError):
                tokenizer.encode('#')

    def test_configs_load_the_matched_tokenizers(self):
        for kind in FAMILIES:
            for split in ('wd', 'wi'):
                path = ROOT / 'configs/thesis/ed' / f'{kind}_linguistic_matched_{split}.yaml'
                config = yaml.safe_load(path.read_text())
                partner = yaml.safe_load((ROOT / 'configs/thesis/ed' / f'bpe_linguistic_{split}.yaml').read_text())
                differing = {key for key in config if config[key] != partner.get(key)}
                self.assertEqual(differing, {'dir_tokenizer', 'dir_work', 'tokenizer'})
                self.assertEqual(config['dir_work'], f'results/thesis/ed/linguistic_{kind}_matched_{split}')
                tokenizer = load_configured_tokenizer(Namespace(**config))
                self.assertEqual(tokenizer.size, 1 + len(ED_ALPHABET) + EXPECTED_K[kind])


if __name__ == '__main__':
    unittest.main()
