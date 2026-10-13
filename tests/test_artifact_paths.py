"""Saved experiment configs keep working after the artifact directory move."""
from pathlib import Path
import tempfile
import unittest

from tva.handwriting.common import resolve_artifact_path
from tva.tokenizer_factory import get_tokenizer, resolve_tokenizer_path

ROOT = Path(__file__).resolve().parents[1]


class ArtifactPathTests(unittest.TestCase):
    def test_saved_run_paths_load_same_vocabulary_and_segmentation(self):
        conditions = [
            ('bigram', 'ed_iam_tva_original_bigram_greedy_v1.json', 'ed/linguistic', 224, 'the handwriting'),
            ('handwriting_bigram_greedy', 'ed_iam_handwriting_bigram_greedy_v1.json', 'ed/handwriting', 224, 'the handwriting'),
            ('handwriting_bpe', 'ed_iam_handwriting_bpe_v1.json', 'ed/handwriting', 224, 'the%=the'),
            ('handwriting_unigram', 'ed_iam_handwriting_unigram_v1.json', 'ed/handwriting', 224, 'the%=the'),
            ('linguistic_bigram_greedy', 'ed_iam_linguistic_bigram_greedy_v1.json', 'archive/ed/linguistic', 224, 'the handwriting'),
            ('handwriting_bigram', 'onhw_words500_rh_iam_read_v1.json', 'archive/onhw/handwriting', 419, 'Dabei'),
            ('linguistic_bigram', 'linguistic_bigram/onhw_words500_wd_word_rh/0.json', 'archive/onhw', 419, 'Dabei'),
        ]
        for kind, old, group, size, text in conditions:
            with self.subTest(kind=kind):
                old_path = ROOT / 'artifacts/tokenizers' / old
                name = old if kind == 'linguistic_bigram' else Path(old).name
                new_path = ROOT / 'artifacts/tokenizers' / group / name
                a, b = get_tokenizer(kind), get_tokenizer(kind)
                a.load(resolve_tokenizer_path(str(old_path), 0))
                b.load(new_path)
                self.assertEqual(a.size, size)
                self.assertEqual(a.encode(text), b.encode(text))
                self.assertEqual(a.decode(a.encode(text)), text)
        directory = ROOT / 'artifacts/tokenizers/linguistic_bigram/onhw_words500_wd_word_rh'
        self.assertEqual(resolve_tokenizer_path(str(directory), 0),
                         str(ROOT / 'artifacts/tokenizers/archive/onhw/linguistic_bigram/onhw_words500_wd_word_rh/0.json'))

    def test_missing_unrelated_files_do_not_redirect_to_bundled_models(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ed_iam_handwriting_bpe_v1.json'
            self.assertEqual(resolve_artifact_path(path), path)
            with self.assertRaises(FileNotFoundError):
                get_tokenizer('handwriting_bpe').load(path)


if __name__ == '__main__':
    unittest.main()
