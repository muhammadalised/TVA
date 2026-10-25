import math
import unittest

from tva.handwriting.training import (
    collect_candidates, forward_backward, frequent_spans, removal_losses, span_connected,
    train_bpe, train_unigram,
)


def line(text, components):
    return {"text": text,
            "edges": {i: None if pair is None else tuple(pair) for i, pair in enumerate(components)}}


def connected_counts(candidates):
    """Handwriting Unigram seeds."""
    return {t: row["n_connected"] for t, row in candidates.items()}


class HandwritingSubwordsTests(unittest.TestCase):
    def test_long_span_requires_same_component_not_just_connected_edges(self):
        consistent = line("abc", [(1, 1), (1, 1)])
        inconsistent = line("abc", [(1, 1), (2, 2)])
        missing = line("abc", [(1, 1), None])
        self.assertTrue(span_connected(consistent, 0, 3))
        self.assertFalse(span_connected(inconsistent, 0, 3))
        self.assertIsNone(span_connected(missing, 0, 3))
        candidates = collect_candidates([consistent, inconsistent, missing], 3, 1, 0.5)
        self.assertEqual(candidates["abc"]["n_total"], 3)
        self.assertEqual(candidates["abc"]["n_exact_alignment"], 2)
        self.assertEqual(candidates["abc"]["connected_rate"], 0.5)

    def test_candidates_never_cross_space_or_punctuation(self):
        corpus = [line("ab cd!ef", [(1, 1)] * 7)]
        candidates = collect_candidates(corpus, 6, 1, 0.5)
        self.assertEqual(set(candidates), {"ab", "cd", "ef"})

    def test_bpe_prioritizes_visual_support_over_text_frequency(self):
        corpus = [line("ab", [(1, 2)]) for _ in range(10)]
        corpus += [line("cd", [(3, 3)]) for _ in range(3)]
        candidates = collect_candidates(corpus, 2, 1, 0)
        self.assertEqual(train_bpe(corpus, candidates, budget=1), [("c", "d")])

    def test_bpe_can_learn_longer_tokens_in_order(self):
        corpus = [line("abc abc", [(1, 1), (1, 1), None, None, (2, 2), (2, 2)])]
        candidates = collect_candidates(corpus, 3, 1, 0.5)
        self.assertEqual(train_bpe(corpus, candidates, budget=2), [("a", "b"), ("ab", "c")])

    def test_different_handwriting_changes_unigram_vocabulary(self):
        connected = [line("abc", [(1, 1), (1, 1)])]
        disconnected = [line("abc", [(1, 2), (2, 3)])]
        eligible = collect_candidates(connected, 3, 1, 0.5)
        ineligible = collect_candidates(disconnected, 3, 1, 0.5)
        self.assertIn("abc", train_unigram(connected, connected_counts(eligible), 3, iterations=3))
        control = train_unigram(disconnected, connected_counts(ineligible), 3, iterations=3)
        self.assertFalse(any(len(t) > 1 for t in control))

    def test_unigram_prunes_to_budget(self):
        corpus = [line("abcabc", [(1, 1)] * 5)] * 4
        candidates = collect_candidates(corpus, 4, 1, 0.5)
        log_probs = train_unigram(corpus, connected_counts(candidates), budget=2, iterations=5)
        self.assertEqual(sum(len(t) > 1 for t in log_probs), 2)
        self.assertAlmostEqual(sum(math.exp(v) for v in log_probs.values()), 1.0)

    def test_forward_backward_matches_exhaustive_segmentation(self):
        probabilities = {"a": 0.2, "b": 0.3, "ab": 0.5}
        score, counts = forward_backward("ab", {t: math.log(p) for t, p in probabilities.items()})
        self.assertAlmostEqual(math.exp(score), 0.5 + 0.2 * 0.3)
        self.assertAlmostEqual(counts["ab"], 0.5 / 0.56)
        self.assertAlmostEqual(counts["a"], 0.06 / 0.56)
        self.assertAlmostEqual(counts["b"], 0.06 / 0.56)

    def test_removal_loss_matches_full_forward_likelihood_difference(self):
        scores = {"a": math.log(0.2), "b": math.log(0.3), "ab": math.log(0.5)}
        loss = removal_losses({"ab": 7}, scores, {"ab"})["ab"]
        original, _ = forward_backward("ab", scores)
        without, _ = forward_backward("ab", {t: v for t, v in scores.items() if t != "ab"})
        self.assertAlmostEqual(loss, 7 * (original - without))

    def test_training_is_deterministic(self):
        corpus = [line("abab cab", [(1, 1)] * 3 + [None] * 2 + [(2, 2)] * 2)] * 3
        candidates = collect_candidates(corpus, 3, 1, 0.5)
        self.assertEqual(train_bpe(corpus, candidates, 2), train_bpe(corpus, candidates, 2))
        seeds = connected_counts(candidates)
        self.assertEqual(train_unigram(corpus, seeds, 2), train_unigram(corpus, seeds, 2))

    def test_frequent_spans_count_letter_spans_only(self):
        corpus = [{"text": "ab cd!ab"}, {"text": "abc"}]
        self.assertEqual(frequent_spans(corpus, 3, 1), {"ab": 3, "abc": 1, "bc": 1, "cd": 1})
        self.assertEqual(frequent_spans(corpus, 3, 2), {"ab": 3})

    def test_text_only_unigram_ignores_connectivity(self):
        # Same text written connected or not: frequency seeds give the same model.
        connected = [line("abc", [(1, 1), (1, 1)])] * 3
        disconnected = [line("abc", [(1, 2), (2, 3)])] * 3
        seeds = frequent_spans(connected, 3, 1)
        self.assertEqual(seeds, frequent_spans(disconnected, 3, 1))
        self.assertEqual(train_unigram(connected, seeds, 1, iterations=3),
                         train_unigram(disconnected, seeds, 1, iterations=3))
        self.assertEqual(train_unigram([{"text": "abc"}] * 3, seeds, 1, iterations=3),
                         train_unigram(connected, seeds, 1, iterations=3))


if __name__ == "__main__":
    unittest.main()
