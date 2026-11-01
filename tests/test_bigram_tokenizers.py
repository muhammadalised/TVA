"""Linguistic (original TVA) and handwriting bigram tokenizers for ED (D024, D025)."""
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
from tva.handwriting_tokenizers import (
    HandwritingBigramTokenizer, LegacyBigramTokenizer, connected_pairs, read_bigram_scores,
)
from tva.tokenizer_factory import get_tokenizer
from tva.tokenizers import BigramTokenizer, split_at_pairs

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'artifacts/evidence/iam/iam_connectivity_v1.json.gz'
TOKENIZERS = ROOT / 'artifacts/tokenizers'
HANDWRITING = TOKENIZERS / 'ed/handwriting/bigram_v3'
HANDWRITING_PREVIOUS = TOKENIZERS / 'ed/handwriting/bigram'
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


def write_tokenizer(directory, scores):
    """A tiny tokenizer over 'abc' with the given bigram scores beside it."""
    vocab = {t: i for i, t in enumerate(['', 'a', 'b', 'c', *scores])}
    path = os.path.join(directory, 'tiny.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'vocab': vocab, 'idx_token': {v: k for k, v in vocab.items()}}, f)
    with open(os.path.join(directory, 'tiny.scores.json'), 'w', encoding='utf-8') as f:
        json.dump(scores, f)
    return path


def train_with_hash_seed(call, name):
    """Run a training call as TVA's notebook does: TVA breaks ties in set() order."""
    code = ('import json, sys; from tva.tokenizers import BigramTokenizer; '
            'from tva.handwriting_tokenizers import HandwritingBigramTokenizer; '
            f'categories = json.loads(sys.argv[2]); {call}')
    with tempfile.TemporaryDirectory() as directory:
        write_iam_dataset(directory)
        subprocess.run(
            [sys.executable, '-c', code, directory, json.dumps(CATEGORIES)],
            cwd=ROOT, env=dict(os.environ, PYTHONHASHSEED='42', PYTHONPATH=str(ROOT)),
            check=True, capture_output=True,
        )
        built = Path(directory, 'tokenizers', name)
        return {path.name: path.read_bytes() for path in built.glob('0.*')}


class BigramTrainingTests(unittest.TestCase):
    def assert_frozen(self, directory, files):
        self.assertEqual(set(files), {'0.json', '0.scores.json'})
        for fold in range(5):
            for name, content in files.items():
                self.assertEqual((directory / name.replace('0.', f'{fold}.', 1)).read_bytes(), content)

    def test_handwriting_training_reproduces_the_frozen_files(self):
        files = train_with_hash_seed(
            f"HandwritingBigramTokenizer().train(sys.argv[1], categories, 224, path_evidence='{EVIDENCE}')",
            'handwriting_bigram224')
        self.assert_frozen(HANDWRITING, files)

    def test_linguistic_training_is_the_unchanged_tva_trainer(self):
        files = train_with_hash_seed('BigramTokenizer().train(sys.argv[1], categories, 224)', 'bigram224')
        self.assert_frozen(LINGUISTIC, files)

    def test_letters_only_linguistic_training_reproduces_its_files(self):
        files = train_with_hash_seed(
            'BigramTokenizer().train(sys.argv[1], categories, 224, letters_only=True)', 'bigram224_letters')
        self.assert_frozen(LINGUISTIC_LETTERS, files)
        vocab = json.loads(files['0.json'])['vocab']
        self.assertEqual(len(vocab), 224)
        self.assertEqual({t for t in vocab if len(t) < 2}, set(CATEGORIES))
        self.assertTrue(all(t.isalpha() for t in vocab if len(t) == 2))

    def test_compression_matched_bigram_is_tva_letters_only_training_at_141(self):
        # Keeping the top 62 bigrams equals training with a 62-bigram budget.
        files = train_with_hash_seed(
            'BigramTokenizer().train(sys.argv[1], categories, 141, letters_only=True)', 'bigram141_letters')
        self.assert_frozen(LINGUISTIC_LETTERS.parent / 'bigram_matched', files)

    def test_handwriting_holds_the_same_pairs_as_before_in_tva_order(self):
        handwriting, previous = BigramTokenizer(), BigramTokenizer()
        handwriting.load(HANDWRITING / '0.json')
        previous.load(HANDWRITING_PREVIOUS / '0.json')
        bigrams = [t for t in sorted(handwriting.vocab, key=handwriting.vocab.get) if len(t) == 2]
        self.assertEqual(set(bigrams), {t for t in previous.vocab if len(t) == 2})
        self.assertEqual(set(handwriting.scores), set(bigrams))
        # TVA numbers bigrams by text frequency; the handwriting scores are connected rates.
        self.assertNotEqual(bigrams, sorted(bigrams))
        self.assertTrue(all(0.5 <= rate <= 1 for rate in handwriting.scores.values()))

    def test_handwriting_selects_the_most_connected_letter_bigrams(self):
        rows = [r for r in read_bigram_scores(EVIDENCE)
                if r['pair'].isalpha() and r['n_exact_alignment'] >= 20]
        ranked = sorted(rows, key=lambda r: (-r['exact_alignment_connected_rate'], r['pair']))
        with tempfile.TemporaryDirectory() as directory:
            write_iam_dataset(directory, num_fold=3)
            tokenizer = HandwritingBigramTokenizer()
            tokenizer.train(directory, CATEGORIES, len(CATEGORIES) + 10, path_evidence=EVIDENCE)
            saved = sorted(os.listdir(Path(directory, 'tokenizers/handwriting_bigram89')))
        self.assertEqual(saved, ['0.json', '0.scores.json'])
        self.assertEqual(tokenizer.size, 89)
        self.assertEqual(tokenizer.scores, {r['pair']: r['exact_alignment_connected_rate'] for r in ranked[:10]})
        self.assertEqual([tokenizer.idx_token[i] for i in range(79)], sorted(CATEGORIES))

    def test_size_is_limited_by_the_qualifying_pairs(self):
        self.assertEqual(len(connected_pairs(EVIDENCE, 389, CATEGORIES)), 389)
        with self.assertRaisesRegex(ValueError, 'Only 389 letter pairs'):
            connected_pairs(EVIDENCE, 390, CATEGORIES)
        self.assertGreater(len(connected_pairs(EVIDENCE, 390, CATEGORIES, min_count=10)), 389)

    def test_rate_column_is_a_setting(self):
        dominant = connected_pairs(EVIDENCE, 145, CATEGORIES)
        box = connected_pairs(EVIDENCE, 145, CATEGORIES, rate='box_intersection_v1_connected_rate')
        self.assertEqual(min(dominant.values()), 0.5)
        self.assertNotEqual(set(dominant), set(box))

    def test_evidence_can_be_a_plain_score_list(self):
        rows = [{'pair': 'ab', 'n_exact_alignment': 30, 'exact_alignment_connected_rate': 0.9},
                {'pair': 'bc', 'n_exact_alignment': 5, 'exact_alignment_connected_rate': 1.0},
                {'pair': 'c.', 'n_exact_alignment': 30, 'exact_alignment_connected_rate': 1.0}]
        with tempfile.TemporaryDirectory() as directory:
            with open(os.path.join(directory, 'train.json'), 'w') as f:
                json.dump({'info': {'num_fold': 1}, 'annotations': {'0': [{'id': 0, 'label': 'abc.'}]}}, f)
            with open(os.path.join(directory, 'scores.json'), 'w') as f:
                json.dump(rows, f)
            tokenizer = HandwritingBigramTokenizer()
            tokenizer.train(directory, ['', 'a', 'b', 'c', '.'], 6,
                            path_evidence=os.path.join(directory, 'scores.json'))
        self.assertEqual(tokenizer.scores, {'ab': 0.9})

    def test_split_at_pairs_cuts_between_letters_only(self):
        self.assertEqual(split_at_pairs('the cat', {'th'}), ['th', 'e c', 'a', 't'])
        self.assertEqual(split_at_pairs('', {'th'}), [''])
        self.assertEqual(''.join(split_at_pairs('In short, 42!', set())), 'In short, 42!')


class BigramEncodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.texts = [t for t in iam_texts() if set(t) <= set(ED_ALPHABET)]

    def load(self, policy, path, cls=BigramTokenizer):
        tokenizer = cls(policy)
        tokenizer.load(path)
        return tokenizer

    def test_handwriting_class_encodes_exactly_like_tva(self):
        handwriting = self.load('greedy', HANDWRITING / '0.json', HandwritingBigramTokenizer)
        original = self.load('greedy', HANDWRITING / '0.json')
        for text in self.texts:
            self.assertEqual(handwriting.encode(text), original.encode(text))

    def test_both_policies_round_trip_without_blank(self):
        for directory in (HANDWRITING, LINGUISTIC, LINGUISTIC_LETTERS):
            for policy in ('greedy', 'dp'):
                with self.subTest(directory=directory.name, policy=policy):
                    tokenizer = self.load(policy, directory / '0.json')
                    self.assertEqual(tokenizer.size, 224)
                    for text in ['', ''.join(ED_ALPHABET), *self.texts[:500]]:
                        ids = tokenizer.encode(text)
                        self.assertNotIn(0, ids)
                        self.assertEqual(tokenizer.decode(ids), text)

    def test_dp_keeps_the_previous_utility_based_segmentation(self):
        # Same pairs and the same connected rates, so the same tokens.
        current = self.load('dp', HANDWRITING / '0.json')
        previous = self.load('dp', HANDWRITING_PREVIOUS / '0.json', LegacyBigramTokenizer)
        self.assertEqual(current.scores, previous.scores)
        for text in self.texts:
            self.assertEqual(current.decode(current.encode(text)), text)
            self.assertEqual([current.idx_token[i] for i in current.encode(text)],
                             [previous.idx_token[i] for i in previous.encode(text)])

    def test_dp_maximizes_total_score(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_tokenizer(directory, {'ab': 0.4, 'bc': 0.5})
            tokens = {policy: [t.idx_token[i] for i in t.encode('abc')]
                      for policy in ('greedy', 'dp') for t in [self.load(policy, path)]}
            self.assertEqual(tokens, {'greedy': ['ab', 'c'], 'dp': ['a', 'bc']})

            # equal scores: prefer more bigrams, then the earlier bigram
            tokenizer = self.load('dp', write_tokenizer(directory, {'ab': 0.5, 'bc': 0.5}))
            self.assertEqual([tokenizer.idx_token[i] for i in tokenizer.encode('abc')], ['ab', 'c'])

    def test_dp_needs_saved_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_tokenizer(directory, {'ab': 1.0})
            os.remove(os.path.join(directory, 'tiny.scores.json'))
            self.load('greedy', path)
            with self.assertRaisesRegex(ValueError, 'dp policy needs'):
                self.load('dp', path)

    def test_policy_is_validated(self):
        with self.assertRaisesRegex(ValueError, 'policy must be'):
            BigramTokenizer('utility-based')


class BigramFactoryTests(unittest.TestCase):
    def test_keys(self):
        self.assertIs(type(get_tokenizer('bigram')), BigramTokenizer)
        self.assertEqual(get_tokenizer('bigram').policy, 'greedy')
        self.assertEqual(get_tokenizer('bigram_dp').policy, 'dp')
        expected = {
            'handwriting_bigram_greedy': (HandwritingBigramTokenizer, 'greedy'),
            'handwriting_bigram_utility': (LegacyBigramTokenizer, 'dp'),
            'handwriting_bigram': (LegacyBigramTokenizer, 'dp'),
            'linguistic_bigram': (LegacyBigramTokenizer, 'dp'),
            'linguistic_bigram_greedy': (LegacyBigramTokenizer, 'greedy'),
        }
        for key, (cls, policy) in expected.items():
            tokenizer = get_tokenizer(key)
            self.assertIs(type(tokenizer), cls)
            self.assertEqual(tokenizer.policy, policy)

    def test_bigram_configs_only_change_the_tokenizer_files_and_output(self):
        for split in ('wd', 'wi'):
            with self.subTest(split=split):
                base = yaml.safe_load((CONFIGS / f'subword_comparison/bigram_linguistic_{split}.yaml').read_text())
                letters = yaml.safe_load((CONFIGS / f'letters_only/bigram_linguistic_letters_{split}.yaml').read_text())
                handwriting = yaml.safe_load((CONFIGS / f'subword_comparison/bigram_handwriting_{split}.yaml').read_text())
                for config in (letters, handwriting):
                    self.assertEqual({k for k in base if base[k] != config[k]}, {'dir_tokenizer', 'dir_work'})
                self.assertEqual(letters['dir_tokenizer'], 'artifacts/tokenizers/ed/linguistic/bigram_letters')
                self.assertEqual(letters['dir_work'], f'results/thesis/ed/linguistic_bigram_letters_{split}')
                self.assertEqual(handwriting['dir_tokenizer'], 'artifacts/tokenizers/ed/handwriting/bigram_v3')
                self.assertEqual(handwriting['dir_work'], f'results/thesis/ed/handwriting_bigram_v3_{split}')

    def test_archived_bigram_files_load(self):
        archived = {
            'ed/handwriting/bigram/0.json': 224,
            'archive/ed/linguistic/ed_iam_linguistic_bigram_greedy_v1.json': 224,
            'archive/onhw/handwriting/onhw_words500_rh_iam_read_v1.json': 419,
            'archive/onhw/linguistic_bigram/onhw_words500_wd_word_rh/0.json': 419,
            'archive/onhw/linguistic_bigram/onhw_words500_wi_word_rh/0.json': 419,
        }
        for path, size in archived.items():
            with self.subTest(path=path):
                tokenizer = LegacyBigramTokenizer('dp')
                tokenizer.load(TOKENIZERS / path)
                self.assertEqual(tokenizer.size, size)
                self.assertEqual(len(tokenizer.scores), sum(len(t) == 2 for t in tokenizer.vocab))


if __name__ == '__main__':
    unittest.main()
