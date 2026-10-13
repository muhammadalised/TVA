"""Evidence leakage/corruption guards and full public training reproduction."""
import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from tva.handwriting.common import sha256_bytes
from tva.handwriting.ed import ED_ALPHABET
from tva.handwriting.evidence import DEFAULT_EVIDENCE, load_evidence, validate_evidence
from tva.handwriting_tokenizers import (
    GreedyHandwritingBigramTokenizer, HandwritingBPETokenizer, HandwritingUnigramTokenizer,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / 'artifacts/tokenizers/ed/handwriting'


class EvidenceGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus, cls.bundle = load_evidence()

    def test_evidence_preserves_full_iam_train_and_abstentions(self):
        self.assertEqual(len(self.corpus), 5694)
        self.assertEqual(sum(len(line['edges']) for line in self.corpus), 160536)
        self.assertTrue(any(edge is None for line in self.corpus for edge in line['edges'].values()))
        self.assertEqual(len(self.bundle['scores']), 1259)
        self.assertFalse(self.bundle['contract']['vocabulary_selection_performed'])
        self.assertNotIn('vocabulary', self.bundle)

    def test_changed_file_bytes_fail_authentication(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'evidence.json.gz'
            path.write_bytes(DEFAULT_EVIDENCE.read_bytes() + b'changed')
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                load_evidence(path)

    def test_leakage_changed_text_and_missing_or_duplicate_edges_fail(self):
        mutations = [
            lambda b: b.update(split='val'),
            lambda b: b['selection'].update(split='test'),
            lambda b: b['manifest'].update(datasets=['IAM', 'ED']),
            lambda b: b['lines'][0].update(text=b['lines'][0]['text'] + '!'),
            lambda b: b['lines'][0]['edges'].pop(),
            lambda b: b['lines'][0]['edges'].append(b['lines'][0]['edges'][0]),
            lambda b: b['lines'][0]['edges'][0].__setitem__(1, 0),
            lambda b: b['scores'][0].update(n_total=b['scores'][0]['n_total'] + 1),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                bundle = copy.deepcopy(self.bundle)
                mutate(bundle)
                with self.assertRaises(ValueError):
                    validate_evidence(bundle)


class FullEvidenceTrainingTests(unittest.TestCase):
    def test_all_public_training_apis_recompute_exact_frozen_artifacts(self):
        conditions = [
            ('bigram', GreedyHandwritingBigramTokenizer, 'ed_iam_handwriting_bigram_greedy_v1.json'),
            ('bpe', HandwritingBPETokenizer, 'ed_iam_handwriting_bpe_v1.json'),
            ('unigram', HandwritingUnigramTokenizer, 'ed_iam_handwriting_unigram_v1.json'),
        ]
        for kind, cls, filename in conditions:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / filename
                tokenizer = cls()
                tokenizer.train(DEFAULT_EVIDENCE, ['', *ED_ALPHABET], output_path=output)
                self.assertEqual(output.read_bytes(), (ARTIFACTS / filename).read_bytes())
                self.assertEqual(tokenizer.size, 224)
                report = json.loads(output.with_suffix('.training.json').read_text())
                self.assertTrue(report['learned_fields_recomputed_from_evidence'])
                self.assertTrue(report['frozen_native_bytes_reproduced'])
                self.assertFalse(report['annotation_label_values_read_from_ed'])
                self.assertEqual(report['trainer_repository'], 'TVA')
                self.assertEqual(report['ed_adapter_sha256'], sha256_bytes(output.read_bytes()))
                native_name = 'iam-english-handwriting-bigram-v1.json' if kind == 'bigram' else f'iam-english-handwriting-{kind}-v1.json'
                self.assertEqual((output.parent / 'source' / native_name).read_bytes(),
                                 (ARTIFACTS / 'source' / native_name).read_bytes())
                for text in ['the handwriting', 'the%=the', 'In short, there will be on TV!']:
                    ids = tokenizer.encode(text)
                    self.assertNotIn(0, ids)
                    self.assertEqual(tokenizer.decode(ids), text)

    def test_training_rejects_model_file_as_evidence_and_wrong_categories(self):
        for cls in (GreedyHandwritingBigramTokenizer, HandwritingBPETokenizer, HandwritingUnigramTokenizer):
            with self.subTest(cls=cls.__name__), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / 'model.json'
                with self.assertRaisesRegex(ValueError, 'evidence SHA-256'):
                    cls().train(ARTIFACTS / 'ed_iam_handwriting_bigram_greedy_v1.json',
                                ['', *ED_ALPHABET], output_path=output)
                with self.assertRaisesRegex(ValueError, 'declared ED alphabet'):
                    cls().train(DEFAULT_EVIDENCE, ['', 'a'], output_path=output)
                self.assertFalse(output.exists())

    def test_conflicting_files_are_preserved_before_any_output_write(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'bigram.json'
            report = output.with_suffix('.training.json')
            report.write_bytes(b'existing report')
            with self.assertRaisesRegex(ValueError, 'Existing frozen file differs'):
                GreedyHandwritingBigramTokenizer().train(DEFAULT_EVIDENCE, ['', *ED_ALPHABET], output_path=output)
            self.assertEqual(report.read_bytes(), b'existing report')
            self.assertFalse(output.exists())
            self.assertFalse((output.parent / 'source').exists())


if __name__ == '__main__':
    unittest.main()
