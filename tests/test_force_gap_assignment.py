import copy
import unittest

import numpy as np

from tva.force_gap_assignment import (
    ForceEvent, LetterGap, assign_force_events, detect_force_events,
)


def event(name, start, end, length=100):
    return ForceEvent(name, start, end, (end-start)*10.0, start == 0, end == length)


class ForceGapAssignmentTests(unittest.TestCase):
    def test_detects_complete_runs_with_inclusive_threshold(self):
        force = np.array([5, 1, 0, 5, 0, 5], dtype=np.float32)
        found = detect_force_events(force, 1, 100, "sample")
        self.assertEqual([(e.start_sample, e.end_sample) for e in found], [(1, 3), (4, 5)])
        self.assertEqual([e.duration_ms for e in found], [20, 10])
        self.assertFalse(any(e.left_truncated or e.right_truncated for e in found))
        np.testing.assert_array_equal(force, [5, 1, 0, 5, 0, 5])

    def test_recording_edges_are_truncated(self):
        found = detect_force_events(np.array([0, 0, 5, 0]), 1, 100, "x")
        self.assertTrue(found[0].left_truncated)
        self.assertTrue(found[1].right_truncated)
        result = assign_force_events(found, [LetterGap("g", 0, 4)], 4)
        self.assertEqual([e["status"] for e in result["events"]], ["unassigned", "unassigned"])
        self.assertIn("truncated_event", result["gaps"][0]["reasons"])

    def test_unique_contained_event_gets_one_candidate_link(self):
        result = assign_force_events([event("e", 12, 20)], [LetterGap("g", 8, 30)], 100)
        self.assertEqual(result["gaps"][0]["status"], "single_candidate")
        self.assertEqual(result["events"][0]["candidate_gap_id"], "g")
        self.assertEqual(result["overlaps"][0]["overlap_samples"], 8)

    def test_crossing_preserves_full_event_duration(self):
        result = assign_force_events([event("e", 8, 16)], [LetterGap("g", 10, 30)], 100)
        self.assertEqual(result["gaps"][0]["status"], "ambiguous")
        self.assertIn("edge_crossing_or_touching", result["gaps"][0]["reasons"])
        self.assertEqual(result["events"][0]["duration_ms"], 80)
        self.assertEqual(result["overlaps"][0]["overlap_samples"], 6)

    def test_same_endpoint_inside_gap_is_not_strict_containment(self):
        for start, end in [(10, 16), (14, 30)]:
            with self.subTest(start=start, end=end):
                result = assign_force_events([event("e", start, end)], [LetterGap("g", 10, 30)], 100)
                self.assertEqual(result["gaps"][0]["status"], "ambiguous")

    def test_endpoint_touch_from_outside_is_not_positive_overlap(self):
        result = assign_force_events([event("e", 10, 20)], [LetterGap("g", 20, 30)], 100)
        self.assertEqual(result["overlaps"], [])
        self.assertEqual(result["gaps"][0]["status"], "no_observed_drop")
        self.assertTrue(result["gaps"][0]["near_edge_low"])
        self.assertEqual(result["events"][0]["status"], "unassigned")

    def test_event_shared_by_two_gaps_is_never_linked_twice(self):
        result = assign_force_events(
            [event("e", 10, 22)],
            [LetterGap("a", 5, 15), LetterGap("b", 18, 30)], 100)
        self.assertEqual(len(result["overlaps"]), 2)
        self.assertTrue(all(g["status"] == "ambiguous" for g in result["gaps"]))
        self.assertTrue(all("shared_event" in g["reasons"] for g in result["gaps"]))
        self.assertIsNone(result["events"][0]["candidate_gap_id"])

    def test_unreliable_competitor_stays_in_overlap_table(self):
        result = assign_force_events(
            [event("e", 10, 22)],
            [LetterGap("a", 5, 15), LetterGap("b", 18, 30, reliable=False)], 100)
        self.assertEqual(len(result["overlaps"]), 2)
        self.assertIn("unavailable_competing_gap", result["gaps"][0]["reasons"])
        self.assertEqual(result["gaps"][1]["status"], "unavailable")

    def test_multiple_events_in_one_gap_are_all_ambiguous(self):
        result = assign_force_events(
            [event("e1", 10, 15), event("e2", 18, 22)],
            [LetterGap("g", 5, 30)], 100)
        self.assertEqual(result["gaps"][0]["status"], "ambiguous")
        self.assertIn("multiple_events", result["gaps"][0]["reasons"])
        self.assertTrue(all(e["status"] == "ambiguous" for e in result["events"]))

    def test_empty_gap_has_no_fallback_assignment(self):
        result = assign_force_events([event("e", 10, 20)], [LetterGap("g", 15, 15)], 100)
        self.assertEqual(result["overlaps"], [])
        self.assertEqual(result["gaps"][0]["status"], "unavailable")
        self.assertIn("empty_gap", result["gaps"][0]["reasons"])
        self.assertEqual(result["events"][0]["status"], "unassigned")

    def test_unreliable_and_padded_gaps_cannot_give_clean_links(self):
        for gap in [LetterGap("g", 5, 30, reliable=False),
                    LetterGap("g", 5, 30, padded=True)]:
            with self.subTest(gap=gap):
                result = assign_force_events([event("e", 10, 20)], [gap], 100)
                self.assertEqual(result["gaps"][0]["status"], "unavailable")
                self.assertEqual(result["events"][0]["status"], "ambiguous")

    def test_out_of_recording_shift_is_unavailable_without_clipping(self):
        result = assign_force_events([event("e", 12, 20)], [LetterGap("g", 2, 30)], 100, shift_samples=-8)
        self.assertEqual(result["gaps"][0]["start_sample"], -6)
        self.assertEqual(result["gaps"][0]["status"], "unavailable")
        self.assertIn("outside_recording", result["gaps"][0]["reasons"])

    def test_timing_shift_changes_gap_match_not_event(self):
        events = [event("e", 12, 20)]
        gaps = [LetterGap("g", 8, 25)]
        base = assign_force_events(events, gaps, 100)
        shifted = assign_force_events(events, gaps, 100, shift_samples=8)
        self.assertEqual(base["gaps"][0]["status"], "single_candidate")
        self.assertEqual(shifted["gaps"][0]["status"], "ambiguous")
        for key in ["event_id", "start_sample", "end_sample", "duration_ms"]:
            self.assertEqual(base["events"][0][key], shifted["events"][0][key])

    def test_repeated_pair_text_does_not_merge_boundary_ids(self):
        result = assign_force_events(
            [event("e1", 12, 18), event("e2", 42, 48)],
            [LetterGap("sample:0", 8, 25), LetterGap("sample:1", 38, 55)], 100)
        self.assertEqual([e["candidate_gap_id"] for e in result["events"]], ["sample:0", "sample:1"])

    def test_no_drop_is_an_observation_not_a_connected_label(self):
        result = assign_force_events([], [LetterGap("g", 8, 30)], 100)
        self.assertEqual(result["gaps"][0]["status"], "no_observed_drop")
        self.assertNotIn("connected", result["gaps"][0])

    def test_input_records_are_not_mutated(self):
        events = [event("e", 12, 20)]
        gaps = [LetterGap("g", 8, 30)]
        before = copy.deepcopy((events, gaps))
        assign_force_events(events, gaps, 100, shift_samples=-8)
        self.assertEqual((events, gaps), before)

    def test_bad_inputs_and_duplicate_ids_are_rejected(self):
        for force in [np.array([]), np.array([[1]]), np.array([np.nan])]:
            with self.subTest(force=force):
                with self.assertRaises(ValueError):
                    detect_force_events(force, 1, 100, "x")
        with self.assertRaises(ValueError):
            detect_force_events(np.array([1]), 0, 0, "x")
        with self.assertRaises(ValueError):
            assign_force_events([event("e", 10, 20), event("e", 30, 40)], [], 100)
        with self.assertRaises(ValueError):
            assign_force_events([], [LetterGap("g", 1, 2), LetterGap("g", 3, 4)], 100)
        with self.assertRaises(ValueError):
            assign_force_events([event("a", 10, 20), event("b", 20, 30)], [], 100)

    def test_a098_empty_gap_keeps_next_low_sample_unassigned(self):
        result = assign_force_events([event("e", 117, 125, 200)], [LetterGap("g", 112, 112)], 200)
        self.assertEqual(result["gaps"][0]["status"], "unavailable")
        self.assertEqual(result["events"][0]["status"], "unassigned")
        self.assertTrue(result["gaps"][0]["near_edge_low"])


if __name__ == "__main__":
    unittest.main()
