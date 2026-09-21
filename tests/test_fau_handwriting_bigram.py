import json
from pathlib import Path
import tempfile
import unittest

from audit_fau_handwriting_bigram import audit
from tva.fau_handwriting_bigram import (
    ADAPTER_BIGRAM_COUNT,
    ADAPTER_POLICY_ID,
    ADAPTER_SHA256,
    ADAPTER_SIZE,
    FAU_ALPHABET,
    SOURCE_BIGRAM_COUNT,
    SOURCE_SHA256,
    build_fau_adapter,
)
from tva.handwriting_bigram_adapter import canonical_json_bytes, sha256_bytes
from tva.handwriting_bigram_tokenizer import (
    GreedyHandwritingBigramTokenizer,
    HandwritingBigramTokenizer,
)
from tva.tokenizers import get_tokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = (
    REPO_ROOT / 'artifacts/tokenizers/source/iam-english-handwriting-bigram-v1.json'
)
ADAPTER_PATH = (
    REPO_ROOT
    / 'artifacts/tokenizers/fau_english_iam_handwriting_bigram_greedy_v1.json'
)
FAU_DIRECTORY = Path('/mnt/c/Users/Ali/Downloads/fau-english-dataset')


class FauAdapterBuilderTests(unittest.TestCase):
    def test_pinned_source_is_the_exact_dtlr_artifact(self):
        self.assertEqual(sha256_bytes(SOURCE_PATH.read_bytes()), SOURCE_SHA256)

    def test_committed_adapter_is_canonical_and_reproducible(self):
        content = ADAPTER_PATH.read_bytes()
        adapter = json.loads(content)
        self.assertEqual(content, canonical_json_bytes(adapter))
        self.assertEqual(sha256_bytes(content), ADAPTER_SHA256)
        self.assertEqual(canonical_json_bytes(build_fau_adapter(SOURCE_PATH)), content)

    def test_adapter_preserves_bigrams_and_projects_only_singletons(self):
        source = json.loads(SOURCE_PATH.read_bytes())
        adapter = json.loads(ADAPTER_PATH.read_bytes())
        self.assertEqual(adapter['model_version'], ADAPTER_POLICY_ID)
        self.assertEqual(adapter['size'], ADAPTER_SIZE)
        self.assertEqual(adapter['eligible_bigram_count'], ADAPTER_BIGRAM_COUNT)
        self.assertEqual(ADAPTER_BIGRAM_COUNT, SOURCE_BIGRAM_COUNT)
        self.assertEqual(adapter['vocabulary'], source['vocabulary'])
        self.assertEqual(
            [adapter['idx_token'][str(index)] for index in range(1, 79)],
            list(FAU_ALPHABET),
        )
        self.assertEqual(adapter['adapter']['source_singletons_excluded'], ['*'])
        self.assertEqual(adapter['adapter']['task_singletons_added'], ['%', '(', '='])
        self.assertFalse(adapter['adapter']['annotation_label_values_read_by_builder'])

    def test_wrong_source_bytes_are_rejected_before_projection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'source.json'
            path.write_bytes(SOURCE_PATH.read_bytes() + b'\n')
            with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
                build_fau_adapter(path)


class GreedyFauTokenizerTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = GreedyHandwritingBigramTokenizer()
        self.tokenizer.load(ADAPTER_PATH)

    def test_factory_and_output_size_include_blank(self):
        tokenizer = get_tokenizer('handwriting_bigram_greedy')
        self.assertIsInstance(tokenizer, GreedyHandwritingBigramTokenizer)
        tokenizer.load(ADAPTER_PATH)
        self.assertEqual(tokenizer.size, 224)
        self.assertEqual(tokenizer.vocab[''], 0)

    def test_greedy_overlap_is_distinct_from_utility_dp(self):
        self.assertEqual(
            self.tokenizer.segment('short')['tokens'], ['s', 'ho', 'r', 't']
        )
        dp = HandwritingBigramTokenizer()
        dp.load_model({
            **self.tokenizer.model,
            'schema_version': 'dtlr.handwriting-bigram-tokenizer.v2',
            'policy': {
                **self.tokenizer.model['policy'],
                'overlap_resolution': 'maximum-total-utility-non-overlapping-v1',
            },
        })
        self.assertEqual(dp.segment('short')['tokens'], ['s', 'h', 'or', 't'])

    def test_punctuation_round_trip_never_emits_blank(self):
        label = 'Is 9 + 3 = 12? (100% yes!)'
        ids = self.tokenizer.encode(label)
        self.assertNotIn(0, ids)
        self.assertEqual(self.tokenizer.decode(ids), label)

    def test_all_fau_labels_round_trip_when_archives_are_available(self):
        archive_stems = ('gold_wd', 'gold_wi')
        if not all(
            (FAU_DIRECTORY / f'{stem}.zip').exists() for stem in archive_stems
        ):
            self.skipTest('local FAU archives are not available')
        result = audit(FAU_DIRECTORY, ADAPTER_PATH)
        self.assertEqual(result['records_per_archive'], 2_550)
        self.assertEqual(result['label_encodings_checked'], 5_100)
        self.assertEqual(result['blank_ids_emitted'], 0)
        self.assertEqual(result['round_trip_failures'], 0)


if __name__ == '__main__':
    unittest.main()
