import copy
import json
import os
import unittest
from pathlib import Path

from tva.handwriting_bigram_adapter import (
    ADAPTER_BIGRAM_COUNT,
    ADAPTER_POLICY_ID,
    ADAPTER_SHA256,
    ADAPTER_SIZE,
    ONHW_ALPHABET,
    SOURCE_SHA256,
    build_frozen_onhw_adapter,
    canonical_json_bytes,
    project_source_model,
    sha256_bytes,
    validate_frozen_source,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = (
    REPO_ROOT / 'artifacts/tokenizers/onhw_words500_rh_iam_read_v1.json'
)


def small_source_model() -> dict:
    rows = [
        {'token': 'AB', 'utility': 0.8, 'evidence': {'source': 'first'}},
        {'token': 'Ax', 'utility': 0.7, 'evidence': {'source': 'excluded'}},
        {'token': 'BA', 'utility': 0.6, 'evidence': {'source': 'last'}},
    ]
    tokens = ['', 'A', 'B', 'x', 'AB', 'Ax', 'BA']
    return {
        'schema_version': 'example',
        'model_version': 'example',
        'text_normalization': 'NFC',
        'blank_token': '',
        'blank_id': 0,
        'policy': {'overlap_resolution': 'example'},
        'vocab': {token: index for index, token in enumerate(tokens)},
        'idx_token': {str(index): token for index, token in enumerate(tokens)},
        'size': len(tokens),
        'eligible_bigram_count': len(rows),
        'vocabulary': rows,
    }


class AdapterBuilderTests(unittest.TestCase):
    def test_projection_uses_only_fixed_alphabet_and_preserves_source_order(self):
        source = small_source_model()
        original = copy.deepcopy(source)
        adapter = project_source_model(source, 'example-sha', ('A', 'B', 'Ä'))

        self.assertEqual(source, original)
        self.assertEqual(list(adapter['vocab']), ['', 'A', 'B', 'Ä', 'AB', 'BA'])
        self.assertEqual([row['token'] for row in adapter['vocabulary']], ['AB', 'BA'])
        self.assertEqual(adapter['vocabulary'][0], source['vocabulary'][0])
        self.assertEqual(adapter['vocabulary'][1], source['vocabulary'][2])
        self.assertEqual(
            adapter['adapter']['fallback_characters_absent_from_source'], ['Ä']
        )
        self.assertFalse(adapter['adapter']['annotation_files_read'])

    def test_projection_rejects_duplicate_task_characters(self):
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            project_source_model(small_source_model(), 'example-sha', ('A', 'A'))

    def test_frozen_source_rejects_wrong_checksum_before_projection(self):
        with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
            validate_frozen_source({}, 'wrong')

    def test_canonical_serialization_is_order_independent_and_utf8(self):
        first = {'z': 'Ä', 'a': [1, 2]}
        second = {'a': [1, 2], 'z': 'Ä'}
        self.assertEqual(canonical_json_bytes(first), canonical_json_bytes(second))
        self.assertIn('Ä'.encode(), canonical_json_bytes(first))
        self.assertTrue(canonical_json_bytes(first).endswith(b'\n'))

    def test_committed_adapter_has_frozen_invariants(self):
        content = ADAPTER_PATH.read_bytes()
        adapter = json.loads(content)
        self.assertEqual(content, canonical_json_bytes(adapter))
        self.assertEqual(adapter['schema_version'], 'dtlr.handwriting-bigram-tokenizer.v2')
        self.assertEqual(adapter['model_version'], ADAPTER_POLICY_ID)
        self.assertEqual(adapter['size'], ADAPTER_SIZE)
        self.assertEqual(adapter['eligible_bigram_count'], ADAPTER_BIGRAM_COUNT)
        self.assertEqual(adapter['vocab'][''], 0)
        self.assertEqual(adapter['idx_token']['0'], '')
        self.assertEqual(
            [adapter['idx_token'][str(index)] for index in range(1, 60)],
            list(ONHW_ALPHABET),
        )
        self.assertEqual(
            adapter['adapter']['source_model']['sha256'], SOURCE_SHA256
        )
        self.assertEqual(
            adapter['adapter']['fallback_characters_absent_from_source'], ['Ä', 'Ü']
        )
        self.assertFalse(adapter['adapter']['annotation_files_read'])
        self.assertEqual(set(adapter['vocab'].values()), set(range(ADAPTER_SIZE)))
        self.assertEqual(
            {int(index): token for index, token in adapter['idx_token'].items()},
            {index: token for token, index in adapter['vocab'].items()},
        )
        self.assertEqual(sha256_bytes(content), ADAPTER_SHA256)
        self.assertNotEqual(ADAPTER_SHA256, SOURCE_SHA256)

    def test_pinned_source_rebuilds_committed_bytes_when_available(self):
        source_value = os.environ.get('TVA_DTLR_SOURCE')
        if not source_value:
            self.skipTest('TVA_DTLR_SOURCE does not identify the external source')
        source_path = Path(source_value)
        rebuilt = build_frozen_onhw_adapter(source_path)
        self.assertEqual(canonical_json_bytes(rebuilt), ADAPTER_PATH.read_bytes())


if __name__ == '__main__':
    unittest.main()
