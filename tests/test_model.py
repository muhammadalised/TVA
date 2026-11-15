import unittest

import torch

from tva.model import build_decoder
from tva.model.unilstm import UniLSTM


class UniLSTMTest(unittest.TestCase):
    def test_returns_one_probability_vector_per_time_step(self) -> None:
        model = UniLSTM(
            size_in=3,
            num_cls=5,
            hidden_size=4,
            num_layers=1,
            r_drop=0.0,
        )
        output = model(torch.randn(2, 7, 3))

        self.assertEqual(output.shape, (2, 7, 5))
        torch.testing.assert_close(
            output.sum(dim=2),
            torch.ones(2, 7),
        )

    def test_future_input_does_not_change_prefix_outputs(self) -> None:
        torch.manual_seed(42)
        model = UniLSTM(
            size_in=3,
            num_cls=5,
            hidden_size=4,
            num_layers=1,
            r_drop=0.0,
        ).eval()
        prefix = torch.randn(1, 4, 3)
        future = torch.randn(1, 3, 3) * 100

        prefix_output = model(prefix)
        extended_output = model(torch.cat((prefix, future), dim=1))[:, :4]

        torch.testing.assert_close(prefix_output, extended_output)

    def test_factory_builds_the_unidirectional_decoder(self) -> None:
        decoder = build_decoder(8, 5, 'unilstm_b')

        self.assertIsInstance(decoder, UniLSTM)
        self.assertFalse(decoder.lstm.bidirectional)


if __name__ == '__main__':
    unittest.main()
