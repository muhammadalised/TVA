"""Regression checks for the matched IAM text-only ED controls."""
from argparse import Namespace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from main import load_configured_tokenizer
from tva.handwriting.common import canonical_json_bytes, sha256_bytes
from tva.handwriting.ed import ED_ALPHABET
from tva.handwriting_tokenizers import HandwritingBPETokenizer, HandwritingUnigramTokenizer
from tva.linguistic_tokenizers import (
    LinguisticBPETokenizer, LinguisticUnigramTokenizer, SOURCE_FILES, SOURCE_SHA256,
)
from tva.tokenizer_factory import get_tokenizer
from handwriting_training_reference import HandwritingSubwordTokenizer

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'artifacts/tokenizers/ed/linguistic'
CLASSES = {'bpe': LinguisticBPETokenizer, 'unigram': LinguisticUnigramTokenizer}
HW_CLASSES = {'bpe': HandwritingBPETokenizer, 'unigram': HandwritingUnigramTokenizer}


class MatchedLinguisticSubwordTests(unittest.TestCase):
    def load(self, kind):
        tokenizer = get_tokenizer(f'linguistic_{kind}')
        tokenizer.load(DIRECTORY / f'ed_iam_linguistic_{kind}_v1.json')
        return tokenizer

    def test_segmentation_is_the_same_function_for_each_matched_pair(self):
        for kind, cls in CLASSES.items():
            self.assertIs(cls.segment, HW_CLASSES[kind].segment)
            self.assertIs(cls.encode, HW_CLASSES[kind].encode)
            self.assertIs(cls.decode, HW_CLASSES[kind].decode)
            self.assertIsInstance(get_tokenizer(f'linguistic_{kind}'), cls)

    def test_frozen_models_preserve_spaces_punctuation_symbols_and_ctc_ids(self):
        for kind in CLASSES:
            tokenizer = self.load(kind)
            self.assertEqual(tokenizer.size, 224)
            self.assertEqual(len([t for t in tokenizer.vocab if len(t) > 1]), 145)
            self.assertEqual({c: tokenizer.vocab[c] for c in ['', *ED_ALPHABET]},
                             {c: i for i, c in enumerate(['', *ED_ALPHABET])})
            for text in ['', ''.join(ED_ALPHABET), 'the  handwriting! 100% = 9 + 3?', 'In short, there will be on TV!']:
                ids = tokenizer.encode(text)
                self.assertNotIn(0, ids)
                self.assertEqual(tokenizer.decode(ids), text)
                self.assertEqual(''.join(tokenizer.segment(text)['tokens']), text)
            with self.assertRaises(ValueError):
                tokenizer.encode('unknown#')
            with self.assertRaises(ValueError):
                tokenizer.decode([224])

    def test_native_segmentation_and_projection_keep_the_learned_parameters(self):
        for kind in CLASSES:
            tokenizer = self.load(kind)
            native = json.loads((DIRECTORY / 'source' / SOURCE_FILES[kind]).read_text())
            self.assertEqual(native['schema_version'], 'dtlr.text-subwords.v1')
            self.assertFalse(native['freeze']['handwriting_evidence_read'])
            self.assertFalse(native['freeze']['ed_labels_read'])
            self.assertEqual(native['freeze']['line_count'], 5694)
            self.assertTrue(native['freeze']['duplicate_transcriptions_retained'])
            self.assertNotIn('connectivity_method', native)
            reference = HandwritingSubwordTokenizer(native)
            for text in ['the handwriting', 'the  handwriting!', 'This is a test.', '123 (Yes!)']:
                self.assertEqual(tokenizer.segment(text), reference.segment(text))
            if kind == 'unigram':
                self.assertTrue(all(tokenizer._scores[t] == score for t, score in native['log_probs'].items() if t in tokenizer.vocab))
                self.assertEqual(tokenizer._scores['%'], 0.0)
                self.assertEqual(tokenizer._scores['='], 0.0)
            else:
                self.assertEqual(tokenizer.source_model['merges'], native['merges'])

    def test_recognition_configs_match_with_equivalent_extracted_loading(self):
        allowed = {'tokenizer', 'dir_tokenizer', 'dir_work', 'dataset_format', 'dir_dataset'}
        for kind in CLASSES:
            for dist in ['wd', 'wi']:
                a = yaml.safe_load((ROOT / f'configs/thesis/ed/{kind}_handwriting_{dist}.yaml').read_text())
                b = yaml.safe_load((ROOT / f'configs/thesis/ed/{kind}_linguistic_{dist}.yaml').read_text())
                self.assertEqual({k: v for k, v in a.items() if k not in allowed},
                                 {k: v for k, v in b.items() if k not in allowed})
                self.assertEqual(a['dataset_format'], 'ed-zip')
                self.assertEqual(b['dataset_format'], 'json-directory')
                self.assertEqual(b['dir_dataset'], f'data/tva/ed/gold_{dist}')
                b['dir_tokenizer'] = str(ROOT / b['dir_tokenizer'])
                self.assertEqual(load_configured_tokenizer(Namespace(**b)).size, 224)

    def test_tampered_native_or_adapter_is_rejected(self):
        for kind, cls in CLASSES.items():
            path = DIRECTORY / f'ed_iam_linguistic_{kind}_v1.json'
            source = DIRECTORY / 'source' / SOURCE_FILES[kind]
            self.assertEqual(sha256_bytes(source.read_bytes()), SOURCE_SHA256[kind])
            with tempfile.TemporaryDirectory() as directory:
                out = Path(directory) / path.name
                packaged = out.parent / 'source' / source.name
                packaged.parent.mkdir()
                packaged.write_bytes(source.read_bytes())
                out.write_bytes(path.read_bytes() + b'\n')
                with self.assertRaisesRegex(ValueError, 'adapter differs'):
                    cls().load(out)
                out.write_bytes(path.read_bytes())
                packaged.write_bytes(source.read_bytes() + b'\n')
                with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
                    cls().load(out)

    def test_training_preflights_outputs_and_rejects_wrong_alphabet(self):
        for kind, cls in CLASSES.items():
            model = self.load(kind)
            with self.assertRaisesRegex(ValueError, 'declared ED alphabet'):
                cls().train(None, ['', 'a'], selection_path=None, output_path='unused.json')
            with tempfile.TemporaryDirectory() as directory:
                out = Path(directory) / 'model.json'
                out.with_suffix('.training.json').write_bytes(b'existing')
                with patch('tva.linguistic_tokenizers.fit_iam_subwords',
                           return_value=(model.model, model.source_model, {})):
                    with self.assertRaisesRegex(ValueError, 'Existing frozen file differs'):
                        cls().train(None, ['', *ED_ALPHABET], selection_path=None, output_path=out)
                self.assertFalse(out.exists())
                self.assertFalse((out.parent / 'source').exists())
                self.assertEqual(out.with_suffix('.training.json').read_bytes(), b'existing')


if __name__ == '__main__':
    unittest.main()
