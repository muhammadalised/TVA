import unittest

import torch

from tva.ctc_alignment import ctc_viterbi_align


class CTCViterbiAlignmentTest(unittest.TestCase):
    def test_aligns_two_tokens_with_blanks(self) -> None:
        # The obvious path is: blank, A, A, blank, B, blank.
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

        self.assertEqual(alignment.expanded_target, [0, 1, 0, 2, 0])
        self.assertEqual(alignment.state_path, [0, 1, 1, 2, 3, 4])
        self.assertEqual(alignment.token_path, [0, 1, 1, 0, 2, 0])

        first, second = alignment.tokens
        self.assertEqual((first.start_frame, first.end_frame), (1, 3))
        self.assertEqual(first.num_frames, 2)
        self.assertEqual((second.start_frame, second.end_frame), (4, 5))

    def test_repeated_tokens_require_a_blank_between_them(self) -> None:
        # A blank between the two A states is required for CTC to produce AA.
        probabilities = torch.tensor(
            [
                [0.90, 0.10],
                [0.10, 0.90],
                [0.90, 0.10],
                [0.10, 0.90],
                [0.90, 0.10],
            ]
        )

        alignment = ctc_viterbi_align(probabilities, target=[1, 1])

        self.assertEqual(alignment.expanded_target, [0, 1, 0, 1, 0])
        self.assertEqual(alignment.state_path, [0, 1, 2, 3, 4])
        self.assertEqual(alignment.token_path, [0, 1, 0, 1, 0])
        self.assertEqual(
            [
                (token.start_frame, token.end_frame)
                for token in alignment.tokens
            ],
            [(1, 2), (3, 4)],
        )

    def test_known_target_constrains_an_uncertain_frame(self) -> None:
        # At frame 2 the model slightly prefers A, but the known target AB
        # forces the complete best path to advance to B.
        probabilities = torch.tensor(
            [
                [0.90, 0.05, 0.05],
                [0.05, 0.90, 0.05],
                [0.05, 0.50, 0.45],
            ]
        )

        alignment = ctc_viterbi_align(probabilities, target=[1, 2])

        self.assertEqual(alignment.token_path, [0, 1, 2])
        self.assertEqual(alignment.tokens[1].start_frame, 2)

    def test_rejects_too_few_frames_for_repeated_tokens(self) -> None:
        probabilities = torch.tensor(
            [
                [0.10, 0.90],
                [0.10, 0.90],
            ]
        )

        with self.assertRaisesRegex(ValueError, 'at least 3 frames'):
            ctc_viterbi_align(probabilities, target=[1, 1])

    def test_rejects_blank_inside_target(self) -> None:
        probabilities = torch.tensor(
            [
                [0.90, 0.10],
                [0.10, 0.90],
            ]
        )

        with self.assertRaisesRegex(ValueError, 'must not contain'):
            ctc_viterbi_align(probabilities, target=[0, 1])


if __name__ == '__main__':
    unittest.main()
