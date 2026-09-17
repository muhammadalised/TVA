import itertools
import json
from pathlib import Path
import tempfile
import unittest

from tva.handwriting_bigram_adapter import (
    ADAPTER_SIZE,
    ONHW_ALPHABET,
    canonical_json_bytes,
    sha256_bytes,
)
from tva.handwriting_bigram_tokenizer import LinguisticBigramTokenizer
from tva.linguistic_bigram import (
    LINGUISTIC_SCHEMA,
    build_dataset_models,
    build_fold_model,
    write_dataset_models,
)
from tva.tokenizers import get_tokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPO_ROOT / 'artifacts/tokenizers/linguistic_bigram'
DATASETS = {
    'onhw_words500_wd_word_rh': REPO_ROOT / 'data/tva/onhw_words500_wd_word_rh',
    'onhw_words500_wi_word_rh': REPO_ROOT / 'data/tva/onhw_words500_wi_word_rh',
}


class LinguisticBigramBuilderTests(unittest.TestCase):
    def test_builder_collapses_duplicate_samples_and_breaks_ties_lexically(self):
        characters = ONHW_ALPHABET[:20]
        labels = [''.join(pair) for pair in itertools.product(characters, repeat=2)]
        annotations = [
            {'label': label, 'id': index}
            for index, label in enumerate([*labels, labels[0], labels[0]])
        ]
        model = build_fold_model(
            annotations,
            dataset='synthetic',
            fold=0,
            source_sha256='source',
        )

        expected = sorted(labels)[:359]
        self.assertEqual(model['size'], ADAPTER_SIZE)
        self.assertEqual(
            [row['token'] for row in model['vocabulary']],
            expected,
        )
        self.assertTrue(all(row['utility'] == 1.0 for row in model['vocabulary']))
        self.assertEqual(model['source_annotations']['sample_count'], 402)
        self.assertEqual(model['source_annotations']['unique_word_type_count'], 400)
        self.assertTrue(model['policy']['sample_duplicates_collapsed'])
        self.assertFalse(model['policy']['validation_annotations_read'])

    def test_dataset_builder_rejects_validation_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'val.json'
            path.write_text('{}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'train.json only'):
                build_dataset_models(path, dataset='synthetic')

    def test_committed_artifacts_match_manifests_and_training_provenance(self):
        for dataset, data_directory in DATASETS.items():
            with self.subTest(dataset=dataset):
                artifact_directory = ARTIFACT_ROOT / dataset
                manifest = json.loads(
                    (artifact_directory / 'manifest.json').read_text(encoding='utf-8')
                )
                self.assertEqual(manifest['dataset'], dataset)
                self.assertEqual(manifest['fold_count'], 5)
                rebuilt = build_dataset_models(
                    data_directory / 'train.json',
                    dataset=dataset,
                )
                for fold in range(5):
                    content = (artifact_directory / f'{fold}.json').read_bytes()
                    model = json.loads(content)
                    self.assertEqual(content, canonical_json_bytes(model))
                    self.assertEqual(model, rebuilt[fold])
                    self.assertEqual(model['schema_version'], LINGUISTIC_SCHEMA)
                    self.assertEqual(model['fold'], fold)
                    self.assertEqual(model['size'], 419)
                    self.assertEqual(model['eligible_bigram_count'], 359)
                    self.assertEqual(
                        manifest['artifacts'][str(fold)]['sha256'],
                        sha256_bytes(content),
                    )
                    self.assertEqual(
                        manifest['artifacts'][str(fold)]['source_annotations_sha256'],
                        model['source_annotations']['sha256'],
                    )
                with tempfile.TemporaryDirectory() as directory:
                    write_dataset_models(rebuilt, directory)
                    temporary = Path(directory)
                    self.assertEqual(
                        (temporary / 'manifest.json').read_bytes(),
                        (artifact_directory / 'manifest.json').read_bytes(),
                    )
                    for fold in range(5):
                        self.assertEqual(
                            (temporary / f'{fold}.json').read_bytes(),
                            (artifact_directory / f'{fold}.json').read_bytes(),
                        )


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

    def test_every_fold_covers_its_train_and_validation_labels(self):
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
            for fold in range(5):
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
        self.assertEqual(checked, 251_990)


if __name__ == '__main__':
    unittest.main()
