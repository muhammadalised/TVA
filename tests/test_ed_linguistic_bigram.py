import json
from pathlib import Path
import tempfile
import unittest

from tva.handwriting.audit import audit
from tva.handwriting.ed import ADAPTER_SHA256 as HANDWRITING_SHA256
from tva.handwriting.common import canonical_json_bytes, sha256_bytes
from tva.handwriting_tokenizers import (
    ED_LINGUISTIC_BIGRAM_COUNT as ADAPTER_BIGRAM_COUNT,
    ED_LINGUISTIC_POLICY_ID as ADAPTER_POLICY_ID,
    ED_LINGUISTIC_SHA256 as ADAPTER_SHA256,
    ED_LINGUISTIC_SIZE as ADAPTER_SIZE,
    IAM_FREQUENCY_EVIDENCE_SHA256 as EVIDENCE_SHA256,
    GreedyLinguisticBigramTokenizer,
)
from tva.handwriting_tokenizers import get_tokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
LINGUISTIC_PATH = (
    REPO_ROOT
    / 'artifacts/tokenizers/ed_iam_linguistic_bigram_greedy_v1.json'
)
HANDWRITING_PATH = (
    REPO_ROOT
    / 'artifacts/tokenizers/ed_iam_handwriting_bigram_greedy_v1.json'
)
ED_DIRECTORY = Path('/mnt/c/Users/Ali/Downloads/ed-dataset')


class FrozenEdLinguisticArtifactTests(unittest.TestCase):
    def test_retained_adapter_records_authenticated_iam_provenance(self):
        adapter = json.loads(LINGUISTIC_PATH.read_bytes())
        evidence = adapter['adapter']['frequency_evidence']
        self.assertEqual(evidence['sha256'], EVIDENCE_SHA256)
        self.assertEqual(evidence['dataset'], 'IAM')
        self.assertEqual(evidence['split'], 'train')
        self.assertEqual(evidence['line_count'], 5694)
        self.assertEqual(
            evidence['labels_sha256'],
            '5ac34ad37ba0b125308fe1a2bc97095985e25dfa76495628c3bb3895c0b446ab',
        )
        self.assertEqual(
            evidence['selection_sha256'],
            '7893dfba4febe6df99cf0bdb0c74fabe7b736d2a6af05033d8638b90455bc1c2',
        )

    def test_adapter_is_frozen_and_matched(self):
        content = LINGUISTIC_PATH.read_bytes()
        adapter = json.loads(content)
        handwriting = json.loads(HANDWRITING_PATH.read_bytes())
        self.assertEqual(
            sha256_bytes(HANDWRITING_PATH.read_bytes()), HANDWRITING_SHA256
        )
        self.assertEqual(content, canonical_json_bytes(adapter))
        self.assertEqual(sha256_bytes(content), ADAPTER_SHA256)
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
            adapter['adapter']['annotation_label_values_read_by_builder']
        )

    def test_changed_frozen_adapter_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'adapter.json'
            path.write_bytes(LINGUISTIC_PATH.read_bytes() + b'\n')
            with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
                GreedyLinguisticBigramTokenizer().load(path)


class GreedyEdLinguisticTokenizerTests(unittest.TestCase):
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

    def test_all_ed_labels_round_trip_when_archives_are_available(self):
        stems = ('gold_wd', 'gold_wi')
        if not all((ED_DIRECTORY / f'{stem}.zip').exists() for stem in stems):
            self.skipTest('local ED archives are not available')
        result = audit(ED_DIRECTORY, LINGUISTIC_PATH, 'linguistic')
        self.assertEqual(result['records_per_archive'], 2_550)
        self.assertEqual(result['label_encodings_checked'], 5_100)
        self.assertEqual(result['encoded_tokens_checked'], 158_156)
        self.assertEqual(result['blank_ids_emitted'], 0)
        self.assertEqual(result['round_trip_failures'], 0)


if __name__ == '__main__':
    unittest.main()
