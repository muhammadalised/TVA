import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tva.alignment_analysis import analyze_alignment
from tva.alignment_plot import plot_alignment
from tva.ctc_alignment import ctc_viterbi_align


def token_text(token_id: int) -> str:
    return {0: '<blank>', 1: 'A', 2: 'B'}[token_id]


class AlignmentAnalysisTest(unittest.TestCase):
    def test_maps_characters_and_blank_region_to_input_samples(self) -> None:
        probabilities = torch.tensor(
            [
                [0.90, 0.05, 0.05],
                [0.05, 0.90, 0.05],
                [0.05, 0.90, 0.05],
                [0.90, 0.05, 0.05],
                [0.05, 0.05, 0.90],
                [0.90, 0.05, 0.05],
            ]
        )
        alignment = ctc_viterbi_align(probabilities, target=[1, 2])

        analysis = analyze_alignment(
            alignment,
            probabilities,
            token_text,
            downsampling_ratio=8,
            num_model_input_samples=50,
            num_raw_samples=45,
            sample_rate_hz=100,
        )

        first, second = analysis.characters
        self.assertEqual(
            (first.input_start_sample, first.input_end_sample),
            (8, 24),
        )
        self.assertEqual(first.anchor_input_sample, 16)
        self.assertAlmostEqual(first.aligned_probability, 0.90)
        self.assertTrue(first.agrees_with_greedy)
        self.assertFalse(second.overlaps_padding)

        boundary = analysis.boundaries[0]
        self.assertEqual(
            (boundary.input_start_sample, boundary.input_end_sample),
            (24, 32),
        )
        self.assertEqual(boundary.num_blank_frames, 1)
        self.assertEqual(boundary.duration_ms, 80)
        self.assertEqual(analysis.trailing_unmodeled_samples, 2)

    def test_marks_a_forced_character_that_is_not_locally_preferred(self) -> None:
        probabilities = torch.tensor(
            [
                [0.90, 0.05, 0.05],
                [0.05, 0.90, 0.05],
                [0.05, 0.50, 0.45],
            ]
        )
        alignment = ctc_viterbi_align(probabilities, target=[1, 2])

        analysis = analyze_alignment(
            alignment,
            probabilities,
            token_text,
            downsampling_ratio=8,
            num_model_input_samples=24,
            num_raw_samples=24,
            sample_rate_hz=100,
        )

        forced_b = analysis.characters[1]
        self.assertEqual(forced_b.preferred_text, 'A')
        self.assertAlmostEqual(forced_b.aligned_probability, 0.45)
        self.assertAlmostEqual(forced_b.confidence_margin, -0.05)
        self.assertFalse(forced_b.agrees_with_greedy)

    def test_creates_visualization_file(self) -> None:
        probabilities = torch.tensor(
            [
                [0.90, 0.05, 0.05],
                [0.05, 0.90, 0.05],
                [0.90, 0.05, 0.05],
                [0.05, 0.05, 0.90],
            ]
        )
        alignment = ctc_viterbi_align(probabilities, target=[1, 2])
        analysis = analyze_alignment(
            alignment,
            probabilities,
            token_text,
            downsampling_ratio=8,
            num_model_input_samples=32,
            num_raw_samples=32,
            sample_rate_hz=100,
        )
        raw_signal = np.zeros((32, 13), dtype=np.float32)
        normalized_signal = np.zeros((32, 13), dtype=np.float32)

        with tempfile.TemporaryDirectory() as directory:
            path_plot = Path(directory) / 'alignment.png'
            plot_alignment(
                raw_signal,
                normalized_signal,
                probabilities,
                analysis,
                known_label='AB',
                greedy_prediction='AB',
                path_save=path_plot,
            )

            self.assertTrue(path_plot.is_file())
            self.assertGreater(path_plot.stat().st_size, 0)


if __name__ == '__main__':
    unittest.main()
