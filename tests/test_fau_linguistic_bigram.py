import json
from pathlib import Path
import tempfile
import unittest

from audit_fau_handwriting_bigram import audit
from tva.fau_handwriting_bigram import ADAPTER_SHA256 as HANDWRITING_SHA256
from tva.fau_linguistic_bigram import (
    ADAPTER_BIGRAM_COUNT,
    ADAPTER_POLICY_ID,
    ADAPTER_SHA256,
    ADAPTER_SIZE,
    EVIDENCE_SHA256,
    LABELS_SHA256,
    SELECTION_SHA256,
    build_fau_linguistic_adapter,
    build_iam_frequency_evidence,
)
from tva.handwriting_bigram_adapter import canonical_json_bytes, sha256_bytes
from tva.handwriting_bigram_tokenizer import GreedyLinguisticBigramTokenizer
from tva.tokenizers import get_tokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = (
    REPO_ROOT / 'artifacts/tokenizers/source/iam-train-letter-bigram-counts-v1.json'
)
LINGUISTIC_PATH = (
    REPO_ROOT
    / 'artifacts/tokenizers/fau_english_iam_linguistic_bigram_greedy_v1.json'
)
HANDWRITING_PATH = (
    REPO_ROOT
    / 'artifacts/tokenizers/fau_english_iam_handwriting_bigram_greedy_v1.json'
)
FAU_DIRECTORY = Path('/mnt/c/Users/Ali/Downloads/fau-english-dataset')
IAM_LABELS = Path('/home/artellisys/DTLR/data/IAM_new/labels.pkl')
IAM_SELECTION = Path('/home/artellisys/dtlr-output/iam-train-full/selection.json')


class FauLinguisticBuilderTests(unittest.TestCase):
    def test_frequency_evidence_is_frozen_and_reproducible(self):
        content = EVIDENCE_PATH.read_bytes()
        evidence = json.loads(content)
        self.assertEqual(content, canonical_json_bytes(evidence))
        self.assertEqual(sha256_bytes(content), EVIDENCE_SHA256)
        self.assertEqual(evidence['line_count'], 5_694)
        self.assertEqual(evidence['candidate_bigram_count'], 861)
        self.assertEqual(evidence['sources']['labels_sha256'], LABELS_SHA256)
        self.assertEqual(
            evidence['sources']['selection_sha256'], SELECTION_SHA256
        )
        self.assertEqual(evidence['counts'][144], {'token': 'bu', 'count': 273})
        self.assertEqual(evidence['counts'][145], {'token': 'tu', 'count': 270})

    def test_external_iam_sources_rebuild_evidence_when_available(self):
        if not IAM_LABELS.exists() or not IAM_SELECTION.exists():
            self.skipTest('authenticated external IAM sources are unavailable')
        rebuilt = build_iam_frequency_evidence(IAM_LABELS, IAM_SELECTION)
        self.assertEqual(canonical_json_bytes(rebuilt), EVIDENCE_PATH.read_bytes())

    def test_adapter_is_canonical_reproducible_and_matched(self):
        content = LINGUISTIC_PATH.read_bytes()
        adapter = json.loads(content)
        handwriting = json.loads(HANDWRITING_PATH.read_bytes())
        self.assertEqual(
            sha256_bytes(HANDWRITING_PATH.read_bytes()), HANDWRITING_SHA256
        )
        self.assertEqual(content, canonical_json_bytes(adapter))
        self.assertEqual(sha256_bytes(content), ADAPTER_SHA256)
        self.assertEqual(
            canonical_json_bytes(build_fau_linguistic_adapter(EVIDENCE_PATH)),
            content,
        )
        self.assertEqual(adapter['model_version'], ADAPTER_POLICY_ID)
        self.assertEqual(adapter['size'], ADAPTER_SIZE)
        self.assertEqual(adapter['eligible_bigram_count'], ADAPTER_BIGRAM_COUNT)
        self.assertEqual(
            [adapter['idx_token'][str(index)] for index in range(79)],
            [handwriting['idx_token'][str(index)] for index in range(79)],
        )
        handwriting_pairs = {row['token'] for row in handwriting['vocabulary']}
        linguistic_pairs = {row['token'] for row in adapter['vocabulary']}
        self.assertEqual(len(handwriting_pairs & linguistic_pairs), 72)
        self.assertTrue(adapter['adapter']['iam_training_transcripts_read'])
        self.assertFalse(
            adapter['adapter']['fau_annotation_label_values_read_by_builder']
        )

    def test_changed_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'evidence.json'
            path.write_bytes(EVIDENCE_PATH.read_bytes() + b'\n')
            with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
                build_fau_linguistic_adapter(path)


class GreedyFauLinguisticTokenizerTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = GreedyLinguisticBigramTokenizer()
        self.tokenizer.load(LINGUISTIC_PATH)

    def test_factory_loads_matched_224_class_comparator(self):
        tokenizer = get_tokenizer('linguistic_bigram_greedy')
        self.assertIsInstance(tokenizer, GreedyLinguisticBigramTokenizer)
        tokenizer.load(LINGUISTIC_PATH)
        self.assertEqual(tokenizer.size, 224)
        self.assertEqual(tokenizer.vocab[''], 0)

    def test_runtime_uses_greedy_left_to_right(self):
        result = self.tokenizer.segment('short')
        self.assertEqual(result['tokens'], ['sh', 'or', 't'])
        self.assertEqual(result['segments'][0]['kind'], 'linguistic-bigram')
        self.assertEqual(self.tokenizer.decode(self.tokenizer.encode('short')), 'short')

    def test_all_fau_labels_round_trip_when_archives_are_available(self):
        stems = ('gold_wd', 'gold_wi')
        if not all((FAU_DIRECTORY / f'{stem}.zip').exists() for stem in stems):
            self.skipTest('local FAU archives are not available')
        result = audit(FAU_DIRECTORY, LINGUISTIC_PATH, 'linguistic')
        self.assertEqual(result['records_per_archive'], 2_550)
        self.assertEqual(result['label_encodings_checked'], 5_100)
        self.assertEqual(result['encoded_tokens_checked'], 158_156)
        self.assertEqual(result['blank_ids_emitted'], 0)
        self.assertEqual(result['round_trip_failures'], 0)


if __name__ == '__main__':
    unittest.main()
