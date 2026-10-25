"""Fold layout and compatibility with paths in completed recognition runs."""
from pathlib import Path
import tempfile
import unittest

from tva.handwriting.common import resolve_artifact_path, save_fold_tokenizers
from tva.tokenizer_factory import get_tokenizer, resolve_tokenizer_path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / 'artifacts/tokenizers'
CONDITIONS = [
    ('char', 'ed_characters_v1.json', 'ed/character/char', 79),
    ('bigram', 'ed_iam_tva_original_bigram_greedy_v1.json', 'ed/linguistic/bigram', 224),
    ('handwriting_bigram_greedy', 'ed_iam_handwriting_bigram_greedy_v1.json', 'ed/handwriting/bigram', 224),
]
# Folders without former file names (BPE/Unigram were replaced by TVA-based files).
FOLDERS = CONDITIONS + [
    ('bigram', None, 'ed/linguistic/bigram_letters', 224),
    ('bpe', None, 'ed/linguistic/bpe', 224),
    ('unigram', None, 'ed/linguistic/unigram', 224),
    ('unigram_em', None, 'ed/linguistic/unigram_em', 224),
    ('handwriting_bpe', None, 'ed/handwriting/bpe', 224),
    ('handwriting_unigram', None, 'ed/handwriting/unigram', 224),
]


class ArtifactPathTests(unittest.TestCase):
    def test_all_fold_files_preserve_vocabulary_and_segmentation(self):
        for key, _, directory, size in FOLDERS:
            reference = get_tokenizer(key)
            root = ARTIFACTS / directory
            reference.load(str(root / '0.json'))
            for fold in range(5):
                with self.subTest(key=key, fold=fold):
                    path = Path(resolve_tokenizer_path(str(root), fold))
                    self.assertEqual(path, root / f'{fold}.json')
                    self.assertEqual(path.read_bytes(), (root / '0.json').read_bytes())
                    tokenizer = get_tokenizer(key)
                    tokenizer.load(str(path))
                    self.assertEqual(tokenizer.size, size)
                    for text in ['the handwriting', '100% = 9 + 3? (Yes!)']:
                        self.assertEqual(tokenizer.encode(text), reference.encode(text))
                        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_saved_run_paths_load_same_vocabulary_and_segmentation(self):
        for key, old_name, directory, _ in CONDITIONS:
            new_path = ARTIFACTS / directory / '0.json'
            # Both layouts that predate the current fold directories are supported.
            old_paths = [ARTIFACTS / old_name, ARTIFACTS / Path(directory).parent / old_name]
            for old_path in old_paths:
                with self.subTest(key=key, path=old_path):
                    self.assertEqual(Path(resolve_tokenizer_path(str(old_path), 0)), new_path)
                    tokenizer = get_tokenizer(key)
                    tokenizer.load(resolve_tokenizer_path(str(old_path), 0))
                    self.assertEqual(tokenizer.decode(tokenizer.encode('the handwriting')), 'the handwriting')
        historical = [
            ('linguistic_bigram_greedy', 'ed_iam_linguistic_bigram_greedy_v1.json', 'archive/ed/linguistic/ed_iam_linguistic_bigram_greedy_v1.json', 224, 'the handwriting'),
            ('handwriting_bigram', 'onhw_words500_rh_iam_read_v1.json', 'archive/onhw/handwriting/onhw_words500_rh_iam_read_v1.json', 419, 'Dabei'),
            ('linguistic_bigram', 'linguistic_bigram/onhw_words500_wd_word_rh/0.json', 'archive/onhw/linguistic_bigram/onhw_words500_wd_word_rh/0.json', 419, 'Dabei'),
        ]
        for key, old, new, size, text in historical:
            with self.subTest(key=key):
                self.assertEqual(resolve_artifact_path(ARTIFACTS / old), ARTIFACTS / new)
                tokenizer = get_tokenizer(key)
                tokenizer.load(resolve_tokenizer_path(str(ARTIFACTS / old), 0))
                self.assertEqual(tokenizer.size, size)
                self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
        old_directory = ARTIFACTS / 'linguistic_bigram/onhw_words500_wd_word_rh'
        self.assertEqual(resolve_tokenizer_path(str(old_directory), 0),
                         str(ARTIFACTS / historical[-1][2]))

    def test_original_fau_english_saved_run_names_load(self):
        conditions = [
            ('handwriting_bigram_greedy', 'fau_english_iam_handwriting_bigram_greedy_v1.json', 'ed/handwriting/bigram/0.json'),
            ('linguistic_bigram_greedy', 'fau_english_iam_linguistic_bigram_greedy_v1.json', 'archive/ed/linguistic/ed_iam_linguistic_bigram_greedy_v1.json'),
        ]
        for key, old, new in conditions:
            with self.subTest(key=key):
                path = resolve_tokenizer_path(str(ARTIFACTS / old), 0)
                self.assertEqual(Path(path), ARTIFACTS / new)
                tokenizer = get_tokenizer(key)
                tokenizer.load(path)
                self.assertEqual(tokenizer.size, 224)
                text = 'In short, there will be on TV!'
                self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_missing_unrelated_files_do_not_redirect_to_bundled_models(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ed_iam_handwriting_bigram_greedy_v1.json'
            self.assertEqual(resolve_artifact_path(path), path)
            with self.assertRaises(FileNotFoundError):
                get_tokenizer('handwriting_bigram_greedy').load(path)
        missing = ARTIFACTS / 'ed/handwriting/bpe/5.json'
        self.assertEqual(resolve_artifact_path(missing), missing)
        self.assertFalse(missing.exists())

    def test_fold_publishing_preserves_conflicts_before_any_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / '0.json').write_bytes(b'original tokenizer')
            (root / '3.json').write_bytes(b'existing tokenizer')
            with self.assertRaisesRegex(ValueError, 'Existing frozen file differs'):
                save_fold_tokenizers(root)
            self.assertFalse((root / '1.json').exists())
            self.assertEqual((root / '3.json').read_bytes(), b'existing tokenizer')
            (root / '3.json').write_bytes(b'original tokenizer')
            save_fold_tokenizers(root)
            save_fold_tokenizers(root)
            self.assertEqual([p.name for p in sorted(root.glob('*.json'))],
                             [f'{fold}.json' for fold in range(5)])
            self.assertTrue(all(p.read_bytes() == b'original tokenizer' for p in root.glob('*.json')))


if __name__ == '__main__':
    unittest.main()
