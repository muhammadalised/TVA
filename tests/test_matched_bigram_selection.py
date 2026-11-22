import unittest
from types import SimpleNamespace

from tva.force_bigram import select_emitting_pairs


class MatchedBigramSelectionTest(unittest.TestCase):
    def setUp(self):
        self.chars = SimpleNamespace(vocab={'': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4}, size=5)

    def test_refill_keeps_originally_emitted_pairs_and_rejects_a_shadowing_pair(self):
        # ab prevents bc from firing in abc. Adding da would block ab in dab.
        order = ['ab', 'bc', 'da', 'cd']
        selected, details = select_emitting_pairs(self.chars, order, {'dabc', 'ab', 'cd'}, 2)
        self.assertEqual(selected, ['ab', 'da'])
        self.assertEqual(details['unused_in_initial_top_pairs'], ['bc'])

        selected, details = select_emitting_pairs(self.chars, order, {'dabc', 'cd'}, 2)
        self.assertEqual(selected, ['ab', 'cd'])
        self.assertEqual(details['rejected_replacements'], ['da'])

    def test_raises_when_budget_cannot_be_filled(self):
        with self.assertRaisesRegex(ValueError, 'Only 1 pairs'):
            select_emitting_pairs(self.chars, ['ab', 'bc'], {'abc'}, 2)
