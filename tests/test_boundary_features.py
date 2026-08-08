import unittest

import numpy as np
import torch

from tva.alignment_analysis import analyze_alignment
from tva.boundary_features import extract_boundary_features
from tva.ctc_alignment import ctc_viterbi_align


def token_text(token_id: int) -> str:
    return {0: '<blank>', 1: 'A', 2: 'B'}[token_id]


class BoundaryFeatureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.probabilities = torch.tensor(
            [
                [0.90, 0.05, 0.05],
                [0.05, 0.90, 0.05],
                [0.90, 0.05, 0.05],
                [0.05, 0.05, 0.90],
            ]
        )
        alignment = ctc_viterbi_align(
            self.probabilities,
            target=[1, 2],
        )
        self.analysis = analyze_alignment(
            alignment,
            self.probabilities,
            token_text,
            downsampling_ratio=8,
            num_model_input_samples=32,
            num_raw_samples=32,
            sample_rate_hz=100,
        )

        self.raw_signal = np.zeros((32, 13), dtype=np.float32)
        self.raw_signal[:, 12] = 100
        self.raw_signal[18:22, 12] = 0

        # A simple ramp gives a non-zero derivative-energy check.
        ramp = np.arange(32, dtype=np.float32)[:, None]
        self.normalized_signal = np.repeat(ramp, 13, axis=1)

    def test_extracts_force_motion_and_reliability_features(self) -> None:
        result = extract_boundary_features(
            self.raw_signal,
            self.normalized_signal,
            self.analysis,
            window_sizes_ms=(100,),
        )

        self.assertEqual(result.force_reference_raw, 100)
        self.assertEqual(result.low_force_threshold_raw, 10)
        self.assertEqual(result.window_sizes_ms, [100])

        boundary = result.boundaries[0]
        self.assertEqual(boundary.pair, 'AB')
        self.assertEqual(boundary.center_input_sample, 20)
        self.assertTrue(boundary.both_anchors_agree_with_greedy)
        self.assertGreater(boundary.minimum_confidence_margin, 0)

        window = boundary.windows[0]
        self.assertEqual((window.start_sample, window.end_sample), (15, 25))
        self.assertEqual(window.force_min_relative, 0)
        self.assertEqual(window.force_drop_ratio, 1)
        self.assertAlmostEqual(window.low_force_fraction, 0.4)
        self.assertEqual(window.longest_low_force_ms, 40)
        self.assertGreater(window.motion_derivative_energy, 0)

    def test_rejects_invalid_window_size(self) -> None:
        with self.assertRaisesRegex(ValueError, 'window sizes'):
            extract_boundary_features(
                self.raw_signal,
                self.normalized_signal,
                self.analysis,
                window_sizes_ms=(0,),
            )

    def test_clips_a_window_whose_alignment_reaches_padding(self) -> None:
        short_raw_signal = self.raw_signal[:10]

        result = extract_boundary_features(
            short_raw_signal,
            self.normalized_signal,
            self.analysis,
            window_sizes_ms=(100,),
        )

        boundary = result.boundaries[0]
        window = boundary.windows[0]
        self.assertTrue(boundary.overlaps_padding)
        self.assertTrue(window.clipped_at_recording_edge)
        self.assertEqual(window.end_sample, len(short_raw_signal))
        self.assertGreater(window.actual_num_samples, 0)


if __name__ == '__main__':
    unittest.main()
