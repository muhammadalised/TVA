import json
from pathlib import Path
import tempfile
import unittest

from tva.handwriting_bigram_adapter import (
    ADAPTER_SIZE,
    canonical_json_bytes,
    sha256_bytes,
)
from tva.handwriting_bigram_tokenizer import (
    LINGUISTIC_SCHEMA,
    MANIFEST_SHA256,
    LinguisticBigramTokenizer,
)
from tva.tokenizers import get_tokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPO_ROOT / 'artifacts/tokenizers/linguistic_bigram'
DATASETS = {
    'onhw_words500_wd_word_rh': REPO_ROOT / 'data/tva/onhw_words500_wd_word_rh',
    'onhw_words500_wi_word_rh': REPO_ROOT / 'data/tva/onhw_words500_wi_word_rh',
}


class FrozenLinguisticArtifactTests(unittest.TestCase):
    def test_completed_fold_zero_matches_frozen_manifest_and_provenance(self):
        for dataset, data_directory in DATASETS.items():
            with self.subTest(dataset=dataset):
                artifact_directory = ARTIFACT_ROOT / dataset
                manifest_content = (artifact_directory / 'manifest.json').read_bytes()
                self.assertEqual(sha256_bytes(manifest_content), MANIFEST_SHA256[dataset])
                manifest = json.loads(manifest_content)
                self.assertEqual(manifest['dataset'], dataset)
                self.assertEqual(manifest['fold_count'], 5)
                for fold in (0,):
                    content = (artifact_directory / f'{fold}.json').read_bytes()
                    model = json.loads(content)
                    self.assertEqual(content, canonical_json_bytes(model))
                    self.assertEqual(model['schema_version'], LINGUISTIC_SCHEMA)
                    self.assertEqual(model['fold'], fold)
                    self.assertEqual(model['size'], ADAPTER_SIZE)
                    self.assertEqual(model['eligible_bigram_count'], 359)
                    self.assertEqual(
                        manifest['artifacts'][str(fold)]['sha256'],
                        sha256_bytes(content),
                    )
                    self.assertEqual(
                        manifest['artifacts'][str(fold)]['source_annotations_sha256'],
                        model['source_annotations']['sha256'],
                    )
                    self.assertFalse(model['policy']['validation_annotations_read'])
                    source = data_directory / 'train.json'
                    if source.exists():
                        self.assertEqual(model['source_annotations']['sha256'], sha256_bytes(source.read_bytes()))
                    tokenizer = LinguisticBigramTokenizer()
                    tokenizer.load(artifact_directory / f'{fold}.json')
                    self.assertEqual(tokenizer.size, ADAPTER_SIZE)


class LinguisticBigramRuntimeTests(unittest.TestCase):
    def test_factory_loads_fold_artifact_and_uses_linguistic_dp(self):
        tokenizer = get_tokenizer('linguistic_bigram')
        self.assertIsInstance(tokenizer, LinguisticBigramTokenizer)
        tokenizer.load(
            ARTIFACT_ROOT / 'onhw_words500_wd_word_rh/0.json'
        )
        self.assertEqual(tokenizer.size, 419)
        result = tokenizer.segment('Dabei')
        self.assertTrue(all(
            segment['kind'] in {'single', 'linguistic-bigram'}
            for segment in result['segments']
        ))
        self.assertEqual(tokenizer.decode(tokenizer.encode('Dabei')), 'Dabei')

    def test_runtime_rejects_artifact_changed_after_manifest(self):
        source_directory = ARTIFACT_ROOT / 'onhw_words500_wd_word_rh'
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            (temporary / 'manifest.json').write_bytes(
                (source_directory / 'manifest.json').read_bytes()
            )
            (temporary / '0.json').write_bytes(
                (source_directory / '0.json').read_bytes() + b'\n'
            )
            with self.assertRaisesRegex(ValueError, 'artifact SHA-256 mismatch'):
                LinguisticBigramTokenizer().load(temporary / '0.json')

    def test_completed_fold_zero_covers_its_train_and_validation_labels(self):
        if not all((path / 'train.json').exists() for path in DATASETS.values()):
            self.skipTest('local OnHW annotations are not available')
        checked = 0
        for dataset, data_directory in DATASETS.items():
            train = json.loads(
                (data_directory / 'train.json').read_text(encoding='utf-8')
            )['annotations']
            validation = json.loads(
                (data_directory / 'val.json').read_text(encoding='utf-8')
            )['annotations']
            for fold in (0,):
                tokenizer = LinguisticBigramTokenizer()
                tokenizer.load(ARTIFACT_ROOT / dataset / f'{fold}.json')
                for annotation in [*train[str(fold)], *validation[str(fold)]]:
                    ids = tokenizer.encode(annotation['label'])
                    self.assertTrue(ids)
                    self.assertTrue(all(0 < index < 419 for index in ids))
                    self.assertEqual(
                        tokenizer.decode(ids),
                        annotation['label'],
                    )
                    checked += 1
        self.assertEqual(checked, 50_398)


if __name__ == '__main__':
    unittest.main()
