"""Linguistic (original TVA) and handwriting bigram tokenizers for ED."""
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml

from tva.dataset.ed import ED_ALPHABET
from tva.handwriting_tokenizers import HandwritingBigramTokenizer, read_bigram_scores
from tva.tokenizer_factory import get_tokenizer
from tva.tokenizers import BigramTokenizer

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'artifacts/evidence/iam/iam_connectivity_v1.json.gz'
TOKENIZERS = ROOT / 'artifacts/tokenizers'
HANDWRITING = TOKENIZERS / 'ed/handwriting/bigram'
LINGUISTIC = TOKENIZERS / 'ed/linguistic/bigram'
LINGUISTIC_LETTERS = TOKENIZERS / 'ed/linguistic/bigram_letters'
CATEGORIES = ['', *ED_ALPHABET]
CONFIGS = ROOT / 'configs/thesis/ed'


def iam_texts():
    with gzip.open(EVIDENCE, 'rt', encoding='utf-8') as f:
        return [line['text'] for line in json.load(f)['lines']]


def write_iam_dataset(directory, num_fold=1):
    annotations = [{'id': i, 'label': text} for i, text in enumerate(iam_texts())]
    with open(os.path.join(directory, 'train.json'), 'w', encoding='utf-8') as f:
        json.dump({'info': {'num_fold': num_fold}, 'annotations': {'0': annotations}}, f)


def write_tokenizer(directory, utility):
    """A tiny tokenizer over 'abc' with the given bigram utilities."""
    vocab = {t: i for i, t in enumerate(['', 'a', 'b', 'c', *utility])}
    path = os.path.join(directory, 'tiny.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'vocab': vocab, 'idx_token': {v: k for k, v in vocab.items()}, 'utility': utility}, f)
    return path


class BigramTrainingTests(unittest.TestCase):
    def test_handwriting_training_reproduces_the_frozen_files(self):
        with tempfile.TemporaryDirectory() as directory:
            write_iam_dataset(directory)
            HandwritingBigramTokenizer().train(directory, CATEGORIES, 224, path_evidence=EVIDENCE)
            trained = Path(directory, 'tokenizers/handwriting_bigram224/0.json').read_bytes()
        for fold in range(5):
            self.assertEqual((HANDWRITING / f'{fold}.json').read_bytes(), trained)

    def train_tva_bigram(self, letters_only, size=224):
        # TVA breaks frequency ties in set() order, which depends on the hash seed.
        code = ('import json, sys; from tva.tokenizers import get_tokenizer; '
                "get_tokenizer('bigram').train(sys.argv[1], json.loads(sys.argv[2]), "
                f'{size}, letters_only={letters_only})')
        with tempfile.TemporaryDirectory() as directory:
            write_iam_dataset(directory)
            subprocess.run(
                [sys.executable, '-c', code, directory, json.dumps(CATEGORIES)],
                cwd=ROOT, env=dict(os.environ, PYTHONHASHSEED='42', PYTHONPATH=str(ROOT)),
                check=True, capture_output=True,
            )
            name = f'bigram{size}_letters' if letters_only else f'bigram{size}'
            return Path(directory, 'tokenizers', name, '0.json').read_bytes()

    def test_linguistic_training_is_the_unchanged_tva_trainer(self):
        trained = self.train_tva_bigram(letters_only=False)
        for fold in range(5):
            self.assertEqual((LINGUISTIC / f'{fold}.json').read_bytes(), trained)

    def test_compression_matched_bigram_is_tva_letters_only_training_at_141(self):
        # Keeping the top 62 bigrams equals training with a 62-bigram budget.
        trained = self.train_tva_bigram(letters_only=True, size=141)
        for fold in range(5):
            self.assertEqual((LINGUISTIC_LETTERS.parent / f'bigram_matched/{fold}.json').read_bytes(), trained)

    def test_letters_only_linguistic_training_reproduces_its_files(self):
        trained = self.train_tva_bigram(letters_only=True)
        for fold in range(5):
            self.assertEqual((LINGUISTIC_LETTERS / f'{fold}.json').read_bytes(), trained)
        vocab = json.loads(trained)['vocab']
        self.assertEqual(len(vocab), 224)
        self.assertEqual({t for t in vocab if len(t) < 2}, set(CATEGORIES))
        self.assertTrue(all(t.isalpha() for t in vocab if len(t) == 2))

    def test_handwriting_selects_the_most_connected_letter_bigrams(self):
        rows = [r for r in read_bigram_scores(EVIDENCE)
                if r['pair'].isalpha() and r['n_exact_alignment'] >= 20]
        ranked = sorted(rows, key=lambda r: (-r['exact_alignment_connected_rate'], r['pair']))
        with tempfile.TemporaryDirectory() as directory:
            write_iam_dataset(directory, num_fold=3)
            tokenizer = HandwritingBigramTokenizer()
            tokenizer.train(directory, CATEGORIES, len(CATEGORIES) + 10, path_evidence=EVIDENCE)
            saved = sorted(os.listdir(Path(directory, 'tokenizers/handwriting_bigram89')))
        self.assertEqual(saved, ['0.json', '1.json', '2.json'])
        self.assertEqual(tokenizer.size, 89)
        self.assertEqual(set(tokenizer.utility), {r['pair'] for r in ranked[:10]})
        self.assertEqual([tokenizer.idx_token[i] for i in range(79)], sorted(CATEGORIES))

    def test_evidence_can_be_a_plain_score_list(self):
        rows = [{'pair': 'ab', 'n_exact_alignment': 30, 'exact_alignment_connected_rate': 0.9},
                {'pair': 'bc', 'n_exact_alignment': 5, 'exact_alignment_connected_rate': 1.0},
                {'pair': 'c.', 'n_exact_alignment': 30, 'exact_alignment_connected_rate': 1.0}]
        with tempfile.TemporaryDirectory() as directory:
            with open(os.path.join(directory, 'train.json'), 'w') as f:
                json.dump({'info': {'num_fold': 1}, 'annotations': {}}, f)
            with open(os.path.join(directory, 'scores.json'), 'w') as f:
                json.dump(rows, f)
            tokenizer = HandwritingBigramTokenizer()
            tokenizer.train(directory, ['', 'a', 'b', 'c', '.'], 10,
                            path_evidence=os.path.join(directory, 'scores.json'))
        self.assertEqual(tokenizer.utility, {'ab': 0.9})


class BigramEncodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.texts = [t for t in iam_texts() if set(t) <= set(ED_ALPHABET)]

    def test_greedy_policy_encodes_exactly_like_tva(self):
        handwriting = HandwritingBigramTokenizer('greedy')
        handwriting.load(HANDWRITING / '0.json')
        original = BigramTokenizer()
        original.load(HANDWRITING / '0.json')
        for text in self.texts:
            self.assertEqual(handwriting.encode(text), original.encode(text))

    def test_both_policies_round_trip_without_blank(self):
        for policy in ('greedy', 'utility-based'):
            tokenizer = HandwritingBigramTokenizer(policy)
            tokenizer.load(HANDWRITING / '0.json')
            self.assertEqual(tokenizer.size, 224)
            for text in ['', ''.join(ED_ALPHABET), *self.texts[:500]]:
                ids = tokenizer.encode(text)
                self.assertNotIn(0, ids)
                self.assertEqual(tokenizer.decode(ids), text)

    def test_utility_based_policy_maximizes_total_utility(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_tokenizer(directory, {'ab': 0.4, 'bc': 0.5})
            tokens = {}
            for policy in ('greedy', 'utility-based'):
                tokenizer = HandwritingBigramTokenizer(policy)
                tokenizer.load(path)
                tokens[policy] = [tokenizer.idx_token[i] for i in tokenizer.encode('abc')]
            self.assertEqual(tokens, {'greedy': ['ab', 'c'], 'utility-based': ['a', 'bc']})

            # equal utility: prefer more bigrams, then the earlier bigram
            tokenizer = HandwritingBigramTokenizer('utility-based')
            tokenizer.load(write_tokenizer(directory, {'ab': 0.5, 'bc': 0.5}))
            self.assertEqual([tokenizer.idx_token[i] for i in tokenizer.encode('abc')], ['ab', 'c'])

    def test_policy_is_validated(self):
        with self.assertRaisesRegex(ValueError, 'policy must be'):
            HandwritingBigramTokenizer('dp')


class BigramFactoryTests(unittest.TestCase):
    def test_keys(self):
        expected = {
            'handwriting_bigram_greedy': 'greedy',
            'handwriting_bigram_utility': 'utility-based',
            'handwriting_bigram': 'utility-based',
            'linguistic_bigram': 'utility-based',
            'linguistic_bigram_greedy': 'greedy',
        }
        for key, policy in expected.items():
            tokenizer = get_tokenizer(key)
            self.assertIsInstance(tokenizer, HandwritingBigramTokenizer)
            self.assertEqual(tokenizer.policy, policy)
        self.assertIs(type(get_tokenizer('bigram')), BigramTokenizer)

    def test_letters_only_configs_only_change_the_tokenizer_and_output(self):
        for split in ('wd', 'wi'):
            with self.subTest(split=split):
                base = yaml.safe_load((CONFIGS / f'subword_comparison/bigram_linguistic_{split}.yaml').read_text())
                letters = yaml.safe_load((CONFIGS / f'letters_only/bigram_linguistic_letters_{split}.yaml').read_text())
                self.assertEqual({k for k in base if base[k] != letters[k]}, {'dir_tokenizer', 'dir_work'})
                self.assertEqual(letters['dir_tokenizer'], 'artifacts/tokenizers/ed/linguistic/bigram_letters')
                self.assertEqual(letters['dir_work'], f'results/thesis/ed/linguistic_bigram_letters_{split}')

    def test_archived_bigram_files_load(self):
        archived = {
            'archive/ed/linguistic/ed_iam_linguistic_bigram_greedy_v1.json': 224,
            'archive/onhw/handwriting/onhw_words500_rh_iam_read_v1.json': 419,
            'archive/onhw/linguistic_bigram/onhw_words500_wd_word_rh/0.json': 419,
            'archive/onhw/linguistic_bigram/onhw_words500_wi_word_rh/0.json': 419,
        }
        for path, size in archived.items():
            with self.subTest(path=path):
                tokenizer = HandwritingBigramTokenizer('utility-based')
                tokenizer.load(TOKENIZERS / path)
                self.assertEqual(tokenizer.size, size)
                self.assertEqual(len(tokenizer.utility), sum(len(t) == 2 for t in tokenizer.vocab))


if __name__ == '__main__':
    unittest.main()
