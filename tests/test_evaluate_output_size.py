import unittest
from unittest.mock import Mock, patch

import evaluate


class EvaluateOutputSizeTests(unittest.TestCase):
    @patch('evaluate.torch.randn', return_value=Mock(name='dummy_input'))
    @patch('evaluate.profile', return_value=(123, 456))
    @patch('evaluate.BaseModel')
    @patch('evaluate.resolve_tokenizer_path', return_value='adapter.json')
    @patch('evaluate.get_tokenizer')
    def test_complexity_model_uses_loaded_tokenizer_size(
        self,
        get_tokenizer,
        resolve_tokenizer_path,
        base_model,
        profile,
        randn,
    ):
        tokenizer = Mock(size=419)
        get_tokenizer.return_value = tokenizer
        model = Mock()
        model.eval.return_value = model
        base_model.return_value = model
        cfgs = {
            'tokenizer': 'handwriting_bigram',
            'dir_tokenizer': 'adapter.json',
            'idx_fold': 0,
            'arch_en': 'blconv_b',
            'arch_de': 'bilstm_b',
            'num_channel': 13,
            'len_seq': 1024,
            'num_concat': 0,
        }

        results = evaluate.get_macs_params(cfgs)

        resolve_tokenizer_path.assert_called_once_with('adapter.json', 0)
        tokenizer.load.assert_called_once_with('adapter.json')
        base_model.assert_called_once_with(
            'blconv_b',
            'bilstm_b',
            13,
            419,
            1024,
        )
        profile.assert_called_once_with(model, inputs=(randn.return_value,))
        self.assertEqual(results['macs'], 123)
        self.assertEqual(results['params'], 456)


if __name__ == '__main__':
    unittest.main()
