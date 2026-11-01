"""Linguistic (original TVA, letters only) and handwriting BPE/Unigram tokenizers for ED (D024)."""
from collections import defaultdict
import json
import os
from pathlib import Path
import tempfile
import unittest

from tva.dataset.ed import ED_ALPHABET
from tva.handwriting_tokenizers import (
    HandwritingBPETokenizer, HandwritingUnigramTokenizer,
    connected_pairs, read_bigram_scores, read_connectivity_lines,
)
from tva.tokenizer_factory import get_tokenizer
from tva.tokenizers import BPETokenizer, UnigramTokenizer

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'artifacts/evidence/iam/iam_connectivity_v1.json.gz'
ED = ROOT / 'artifacts/tokenizers/ed'
ONHW = ROOT / 'data/tva/onhw_words500_wd_word_rh'
CATEGORIES = ['', *ED_ALPHABET]
FILES = {
    'linguistic_bpe': ('bpe', ED / 'linguistic/bpe'),
    'linguistic_unigram': ('unigram', ED / 'linguistic/unigram'),
    'handwriting_bpe': ('bpe', ED / 'handwriting/bpe_v3'),
    'handwriting_unigram': ('unigram', ED / 'handwriting/unigram_v3'),
}


def iam_lines():
    return read_connectivity_lines(EVIDENCE)


def write_iam_dataset(directory):
    annotations = [{'id': i, 'label': line['text']} for i, line in enumerate(iam_lines())]
    with open(os.path.join(directory, 'train.json'), 'w', encoding='utf-8') as f:
        json.dump({'info': {'num_fold': 1}, 'annotations': {'0': annotations}}, f)


def load(name, fold=0):
    key, directory = FILES[name]
    tokenizer = get_tokenizer(key)
    tokenizer.load(str(directory / f'{fold}.json'))
    return tokenizer


def pieces(tokenizer):
    return {t for t in tokenizer.tokenizer.get_vocab() if len(t) > 1 and t != '<PAD>'}


class SubwordTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        write_iam_dataset(cls.directory.name)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def trained(self, name):
        return Path(self.directory.name, 'tokenizers', name, '0.json').read_bytes()

    def test_letters_only_linguistic_bpe_reproduces_its_files(self):
        get_tokenizer('bpe').train(self.directory.name, CATEGORIES, 224, letters_only=True)
        for fold in range(5):
            self.assertEqual((ED / f'linguistic/bpe/{fold}.json').read_bytes(), self.trained('bpe224_letters'))

    def test_letters_only_linguistic_unigram_selects_the_frozen_pieces(self):
        # Hugging Face's Unigram trainer is not bit-reproducible: IDs and scores vary slightly.
        tokenizer = get_tokenizer('unigram')
        tokenizer.train(self.directory.name, CATEGORIES, 224, letters_only=True)
        self.assertEqual(pieces(tokenizer), pieces(load('linguistic_unigram')))

    def test_allowing_every_pair_is_the_original_tva_trainer(self):
        letters = [c for c in ED_ALPHABET if c.isalpha()]
        every_pair = {a + b for a in letters for b in letters}
        BPETokenizer().train(self.directory.name, CATEGORIES, 224, letters_only=True,
                             allowed_pairs=every_pair, name='bpe_every_pair')
        self.assertEqual(self.trained('bpe_every_pair'), (ED / 'linguistic/bpe/0.json').read_bytes())

    def test_handwriting_bpe_reproduces_its_files(self):
        HandwritingBPETokenizer().train(self.directory.name, CATEGORIES, 224, path_evidence=EVIDENCE)
        for fold in range(5):
            self.assertEqual((ED / f'handwriting/bpe_v3/{fold}.json').read_bytes(), self.trained('handwriting_bpe224'))

    def test_handwriting_unigram_selects_the_frozen_pieces(self):
        tokenizer = HandwritingUnigramTokenizer()
        tokenizer.train(self.directory.name, CATEGORIES, 224, path_evidence=EVIDENCE)
        self.assertEqual(pieces(tokenizer), pieces(load('handwriting_unigram')))
        for fold in range(1, 5):
            self.assertEqual((ED / f'handwriting/unigram_v3/{fold}.json').read_bytes(),
                             (ED / 'handwriting/unigram_v3/0.json').read_bytes())

    @unittest.skipUnless((ONHW / 'tokenizers/bpe200/0.json').is_file(), 'needs the local OnHW dataset')
    def test_default_bpe_is_the_original_tva_trainer(self):
        with tempfile.TemporaryDirectory() as directory:
            os.symlink(ONHW / 'train.json', Path(directory, 'train.json'))
            categories = json.loads((ONHW / 'train.json').read_text())['categories']
            BPETokenizer().train(directory, ['', *categories], 200)
            for fold in range(5):
                original = json.loads((ONHW / f'tokenizers/bpe200/{fold}.json').read_text())
                trained = json.loads(Path(directory, f'tokenizers/bpe200/{fold}.json').read_text())
                self.assertEqual(trained['model'], original['model'])
                self.assertEqual(trained['pre_tokenizer'], original['pre_tokenizer'])


class SubwordEncodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.texts = [line['text'] for line in iam_lines() if set(line['text']) <= set(ED_ALPHABET)]
        cls.tokenizers = {name: load(name) for name in FILES}

    def test_classes_are_tva_classes(self):
        for name, tokenizer in self.tokenizers.items():
            with self.subTest(name=name):
                self.assertIs(type(tokenizer), BPETokenizer if 'bpe' in name else UnigramTokenizer)

    def test_same_layout_and_letter_only_pieces(self):
        for name, tokenizer in self.tokenizers.items():
            with self.subTest(name=name):
                vocab = tokenizer.tokenizer.get_vocab()
                self.assertEqual(tokenizer.size, 224)
                self.assertEqual(vocab['<PAD>'], 0)
                self.assertTrue(set(ED_ALPHABET) <= set(vocab))
                self.assertEqual(len(pieces(tokenizer)), 145)
                self.assertTrue(all(p.isalpha() for p in pieces(tokenizer)))

    def test_round_trip_keeps_spaces_and_never_emits_blank(self):
        for name, tokenizer in self.tokenizers.items():
            with self.subTest(name=name):
                for text in ['', ''.join(ED_ALPHABET), 'In short, there will be on TV!', *self.texts[:500]]:
                    ids = tokenizer.encode(text)
                    self.assertNotIn(0, ids)
                    self.assertEqual(tokenizer.decode(ids), text)
                tokens = [tokenizer.decode([i]) for i in tokenizer.encode('the cat, 42')]
                self.assertEqual([t for t in tokens if not t.isalpha()], [' ', ',', ' ', '4', '2'])

    def test_handwriting_pieces_only_join_allowed_pairs(self):
        allowed = set(connected_pairs(EVIDENCE, 145, CATEGORIES))
        self.assertEqual(min(connected_pairs(EVIDENCE, 145, CATEGORIES).values()), 0.5)
        for name in FILES:
            with self.subTest(name=name):
                joined = {p for p in pieces(self.tokenizers[name])
                          if all(p[i:i + 2] in allowed for i in range(len(p) - 1))}
                if name.startswith('handwriting'):
                    self.assertEqual(joined, pieces(self.tokenizers[name]))
                else:  # text-only pieces also join pairs usually written apart
                    self.assertLess(len(joined), 145)


class SubwordConfigTests(unittest.TestCase):
    def test_configs_load_their_tokenizers(self):
        from argparse import Namespace

        import yaml

        from main import load_configured_tokenizer

        expected = {
            'bpe_linguistic': ('bpe', 'linguistic/bpe', 'linguistic_bpe_v2'),
            'unigram_linguistic': ('unigram', 'linguistic/unigram', 'linguistic_unigram_v2'),
            'bpe_handwriting': ('bpe', 'handwriting/bpe_v3', 'handwriting_bpe_v3'),
            'unigram_handwriting': ('unigram', 'handwriting/unigram_v3', 'handwriting_unigram_v3'),
        }
        for stem, (key, directory, work) in expected.items():
            for split in ('wd', 'wi'):
                with self.subTest(config=f'{stem}_{split}'):
                    config = yaml.safe_load((ROOT / f'configs/thesis/ed/subword_comparison/{stem}_{split}.yaml').read_text())
                    self.assertEqual(config['tokenizer'], key)
                    self.assertEqual(config['dir_tokenizer'], f'artifacts/tokenizers/ed/{directory}')
                    self.assertEqual(config['dir_work'], f'results/thesis/ed/{work}_{split}')
                    self.assertEqual(load_configured_tokenizer(Namespace(**config)).size, 224)


class EvidenceTests(unittest.TestCase):
    def test_bigram_scores_summarize_the_line_evidence(self):
        counts = defaultdict(lambda: [0, 0])
        for line in iam_lines():
            for i, edge in line['edges'].items():
                if edge is not None:
                    pair = counts[line['text'][i:i + 2]]
                    pair[0] += 1
                    pair[1] += int(edge[0] == edge[1])
        for row in read_bigram_scores(EVIDENCE):
            exact, connected = counts[row['pair']]
            self.assertEqual(row['n_exact_alignment'], exact)
            if exact:
                self.assertEqual(row['exact_alignment_connected_rate'], connected / exact)


if __name__ == '__main__':
    unittest.main()
