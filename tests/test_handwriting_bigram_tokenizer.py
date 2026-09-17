import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
import unicodedata

from tva.handwriting_bigram_adapter import ADAPTER_SIZE
from tva.handwriting_bigram_tokenizer import HandwritingBigramTokenizer
from tva.tokenizers import get_tokenizer, resolve_tokenizer_path


REPO_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = (
    REPO_ROOT / 'artifacts/tokenizers/onhw_words500_rh_iam_read_v1.json'
)
DATASETS = (
    REPO_ROOT / 'data/tva/onhw_words500_wd_word_rh',
    REPO_ROOT / 'data/tva/onhw_words500_wi_word_rh',
)


def small_model(ab_utility: float, bc_utility: float) -> dict:
    rows = [
        {
            'token': 'ab',
            'utility': ab_utility,
            'n_exact_alignment': 10,
            'exact_alignment_connected_rate': ab_utility,
        },
        {
            'token': 'bc',
            'utility': bc_utility,
            'n_exact_alignment': 10,
            'exact_alignment_connected_rate': bc_utility,
        },
    ]
    tokens = ['', 'a', 'b', 'c', 'ab', 'bc']
    return {
        'schema_version': 'dtlr.handwriting-bigram-tokenizer.v2',
        'model_version': 'test-model',
        'text_normalization': 'NFC',
        'blank_token': '',
        'blank_id': 0,
        'policy': {
            'overlap_resolution': 'maximum-total-utility-non-overlapping-v1',
            'single_character_utility': 0.0,
        },
        'vocab': {token: index for index, token in enumerate(tokens)},
        'idx_token': {str(index): token for index, token in enumerate(tokens)},
        'size': len(tokens),
        'eligible_bigram_count': len(rows),
        'vocabulary': rows,
    }


class HandwritingBigramTokenizerTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = HandwritingBigramTokenizer()
        self.tokenizer.load(ADAPTER_PATH)

    def test_factory_loads_frozen_adapter_with_blank_in_output_size(self):
        tokenizer = get_tokenizer('handwriting_bigram')
        self.assertIsInstance(tokenizer, HandwritingBigramTokenizer)
        tokenizer.load(ADAPTER_PATH)
        self.assertEqual(tokenizer.size, ADAPTER_SIZE)
        self.assertEqual(tokenizer.vocab[''], 0)
        self.assertEqual(tokenizer.idx_token[0], '')

    def test_shared_artifact_and_legacy_fold_paths_both_resolve(self):
        self.assertEqual(
            resolve_tokenizer_path(str(ADAPTER_PATH), 3),
            str(ADAPTER_PATH),
        )
        legacy_directory = REPO_ROOT / 'data/tva/onhw_words500_wd_word_rh/tokenizers/char'
        self.assertEqual(
            resolve_tokenizer_path(str(legacy_directory), 3),
            str(legacy_directory / '3.json'),
        )

    def test_documented_overlap_examples_use_utility_dp(self):
        expected = {
            'Juni': ['J', 'u', 'ni'],
            'wir': ['w', 'ir'],
            'Teil': ['T', 'ei', 'l'],
            'Dabei': ['D', 'ab', 'ei'],
        }
        for label, tokens in expected.items():
            with self.subTest(label=label):
                self.assertEqual(self.tokenizer.segment(label)['tokens'], tokens)

    def test_dp_maximizes_utility_and_uses_dtlr_tie_breaking(self):
        tokenizer = HandwritingBigramTokenizer()
        tokenizer.load_model(small_model(ab_utility=0.4, bc_utility=0.9))
        self.assertEqual(tokenizer.segment('abc')['tokens'], ['a', 'bc'])

        tokenizer.load_model(small_model(ab_utility=0.9, bc_utility=0.9))
        self.assertEqual(tokenizer.segment('abc')['tokens'], ['ab', 'c'])

    def test_nfc_round_trip_and_blank_decode(self):
        decomposed = 'Ku\N{COMBINING DIAERESIS}'
        ids = self.tokenizer.encode(decomposed)
        self.assertNotIn(0, ids)
        self.assertEqual(self.tokenizer.decode(ids), 'Kü')
        self.assertEqual(self.tokenizer.decode([0, *ids, 0]), 'Kü')
        self.assertEqual(self.tokenizer.segment(decomposed)['text'], 'Kü')

    def test_unknown_character_and_id_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, 'absent from tokenizer vocabulary'):
            self.tokenizer.encode('two words')
        with self.assertRaisesRegex(ValueError, 'unknown token ID'):
            self.tokenizer.decode([ADAPTER_SIZE])

    def test_load_rejects_noncanonical_bytes(self):
        content = ADAPTER_PATH.read_bytes() + b'\n'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'changed.json'
            path.write_bytes(content)
            with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
                HandwritingBigramTokenizer().load(path)

    def test_all_available_onhw_labels_round_trip_without_blank(self):
        if not all((directory / 'train.json').exists() for directory in DATASETS):
            self.skipTest('local OnHW annotations are not available')
        checked = 0
        for directory in DATASETS:
            for split in ('train', 'val'):
                annotations = json.loads(
                    (directory / f'{split}.json').read_text(encoding='utf-8')
                )['annotations']
                for fold in range(5):
                    for annotation in annotations[str(fold)]:
                        normalized = unicodedata.normalize('NFC', annotation['label'])
                        ids = self.tokenizer.encode(annotation['label'])
                        self.assertTrue(ids)
                        self.assertTrue(all(0 < index < ADAPTER_SIZE for index in ids))
                        self.assertEqual(self.tokenizer.decode(ids), normalized)
                        checked += 1
        self.assertEqual(checked, 251_990)

    def test_matches_dtlr_reference_on_complete_unique_label_set_when_available(self):
        dtlr_poc = os.environ.get('TVA_DTLR_POC')
        if not dtlr_poc or not all(
            (directory / 'train.json').exists() for directory in DATASETS
        ):
            self.skipTest('DTLR reference path or local OnHW annotations unavailable')
        import sys

        sys.path.insert(0, dtlr_poc)
        try:
            from dtlr_poc.tokenizer import HandwritingBigramTokenizer as Reference
        finally:
            sys.path.pop(0)

        reference = Reference()
        reference.load(ADAPTER_PATH)
        labels = {'Juni', 'wir', 'Teil', 'Dabei'}
        for directory in DATASETS:
            for split in ('train', 'val'):
                annotations = json.loads(
                    (directory / f'{split}.json').read_text(encoding='utf-8')
                )['annotations']
                for fold_labels in annotations.values():
                    labels.update(annotation['label'] for annotation in fold_labels)
        self.assertEqual(len(labels), 501)
        for label in sorted(labels):
            with self.subTest(label=label):
                self.assertEqual(self.tokenizer.segment(label), reference.segment(label))
                self.assertEqual(self.tokenizer.encode(label), reference.encode(label))


if __name__ == '__main__':
    unittest.main()
