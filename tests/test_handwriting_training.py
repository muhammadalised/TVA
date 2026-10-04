"""Regression checks for the notebook-facing training API."""
from pathlib import Path
import tempfile
import unittest

from tva.handwriting.ed import ED_ALPHABET
from tva.handwriting.iam import DEFAULT_LABELS, DEFAULT_SELECTION
from tva.handwriting_tokenizers import GreedyHandwritingBigramTokenizer
from tva.handwriting_tokenizers import IAMBigramTokenizer

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / 'artifacts/tokenizers'
SOURCE = ARTIFACTS / 'source/iam-english-handwriting-bigram-v1.json'
CATEGORIES = ['', *ED_ALPHABET]


class NotebookTrainingTests(unittest.TestCase):
    def test_handwriting_training_reproduces_frozen_vocabulary(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'handwriting.json'
            tokenizer = GreedyHandwritingBigramTokenizer()
            tokenizer.train(SOURCE, CATEGORIES, output_path=output)
            self.assertEqual(output.read_bytes(), (ARTIFACTS / 'ed_iam_handwriting_bigram_greedy_v1.json').read_bytes())
            text = 'In short, there will be on TV!'
            self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
            self.assertEqual(tokenizer.size, 224)

    def test_training_preserves_an_existing_different_file(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'handwriting.json'
            output.write_bytes(b'existing file')
            with self.assertRaises(ValueError):
                GreedyHandwritingBigramTokenizer().train(SOURCE, CATEGORIES, output_path=output)
            self.assertEqual(output.read_bytes(), b'existing file')

    def test_wrong_alphabet_fails_before_creating_an_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'handwriting.json'
            with self.assertRaises(ValueError):
                GreedyHandwritingBigramTokenizer().train(SOURCE, ['', 'a'], output_path=output)
            self.assertFalse(output.exists())

    @unittest.skipUnless(DEFAULT_LABELS.is_file() and DEFAULT_SELECTION.is_file(), 'External IAM inputs unavailable')
    def test_iam_training_reproduces_original_vocabulary_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'linguistic.json'
            tokenizer = IAMBigramTokenizer()
            tokenizer.train(DEFAULT_LABELS, CATEGORIES, 224, selection_path=DEFAULT_SELECTION, output_path=output)
            original = ARTIFACTS / 'ed_iam_tva_original_bigram_greedy_v1.json'
            self.assertEqual(output.read_bytes(), original.read_bytes())
            self.assertEqual(output.with_suffix('.provenance.json').read_bytes(), original.with_suffix('.provenance.json').read_bytes())
            self.assertEqual(tokenizer.size, 224)


if __name__ == '__main__':
    unittest.main()
