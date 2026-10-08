"""Protect frozen source identity, inference semantics, and matched ED runs."""
from argparse import Namespace
import json
from pathlib import Path
import tempfile
import unittest

import yaml

from main import load_configured_tokenizer
from tva.handwriting.common import canonical_json_bytes
from tva.handwriting.ed import ED_ALPHABET
from tva.handwriting.subwords import (
    ADAPTER_FILES, SOURCE_FILES, build_ed_subword_adapter,
)
from tva.handwriting_tokenizers import (
    HandwritingBPETokenizer, HandwritingUnigramTokenizer, get_tokenizer,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / 'artifacts/tokenizers'
CLASSES = {'bpe': HandwritingBPETokenizer, 'unigram': HandwritingUnigramTokenizer}


class HandwritingSubwordTests(unittest.TestCase):
    def tokenizer(self, kind):
        tokenizer = get_tokenizer(f'handwriting_{kind}')
        tokenizer.load(ARTIFACTS / ADAPTER_FILES[kind])
        return tokenizer

    def test_reproduction_and_preserved_inference_parameters(self):
        for kind, cls in CLASSES.items():
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / 'tokenizer.json'
                source = ARTIFACTS / 'source' / SOURCE_FILES[kind]
                tokenizer = cls()
                tokenizer.train(source, ['', *ED_ALPHABET], output_path=output)
                self.assertEqual(output.read_bytes(), (ARTIFACTS / ADAPTER_FILES[kind]).read_bytes())
                self.assertEqual((output.parent / 'source' / source.name).read_bytes(), source.read_bytes())
                self.assertEqual(tokenizer.size, 224)
                self.assertEqual({t for t in tokenizer.vocab if len(t) == 1}, set(ED_ALPHABET))
                self.assertEqual({t for t in tokenizer.vocab if len(t) > 1},
                                 {t for t in tokenizer.source_model['vocab'] if len(t) > 1})
                if kind == 'unigram':
                    for token, score in tokenizer._scores.items():
                        if token not in '%=':
                            self.assertEqual(score, tokenizer.source_model['log_probs'][token])

    def test_reference_segmentation_and_forced_symbol_boundaries(self):
        expected = {
            'bpe': ['the', ' ', 'h', 'an', 'd', 'w', 'r', 'i', 't', 'i', 'n', 'g'],
            'unigram': ['the', ' ', 'ha', 'n', 'd', 'w', 'r', 'i', 't', 'i', 'n', 'g'],
        }
        for kind in CLASSES:
            with self.subTest(kind=kind):
                tokenizer = self.tokenizer(kind)
                self.assertEqual(tokenizer.segment('the handwriting')['tokens'], expected[kind])
                tokens = tokenizer.segment('the%=the')['tokens']
                self.assertEqual(tokens, ['the', '%', '=', 'the'])
                for text in ['', ''.join(ED_ALPHABET), 'the%=the', '100% = 9 + 3? (Yes!)']:
                    ids = tokenizer.encode(text)
                    self.assertNotIn(0, ids)
                    self.assertEqual(tokenizer.decode(ids), text)
                with self.assertRaises(ValueError):
                    tokenizer.encode('unknown#')
                with self.assertRaises(ValueError):
                    tokenizer.decode([224])

    def test_tampering_adapter_or_source_is_rejected(self):
        for kind, cls in CLASSES.items():
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                source = ARTIFACTS / 'source' / SOURCE_FILES[kind]
                output = Path(directory) / 'model.json'
                cls().train(source, ['', *ED_ALPHABET], output_path=output)
                original = output.read_bytes()
                model = json.loads(original)
                model['policy']['inference'] = 'greedy'
                output.write_bytes(canonical_json_bytes(model))
                with self.assertRaisesRegex(ValueError, 'adapter differs'):
                    cls().load(output)
                output.write_bytes(original)
                packaged = output.parent / 'source' / source.name
                packaged.write_bytes(packaged.read_bytes() + b'\n')
                with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
                    cls().load(output)

    def test_invalid_categories_and_different_outputs_are_preserved(self):
        for kind, cls in CLASSES.items():
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                source = ARTIFACTS / 'source' / SOURCE_FILES[kind]
                output = Path(directory) / 'model.json'
                with self.assertRaises(ValueError):
                    cls().train(source, ['', 'a'], output_path=output)
                self.assertFalse(output.exists())
                output.write_bytes(b'existing')
                with self.assertRaises(ValueError):
                    cls().train(source, ['', *ED_ALPHABET], output_path=output)
                self.assertEqual(output.read_bytes(), b'existing')

    def test_fold_zero_configs_match_bigram_and_load_through_training_entry(self):
        allowed = {'dir_tokenizer', 'dir_work', 'tokenizer'}
        for kind in CLASSES:
            for distribution in ('wd', 'wi'):
                with self.subTest(kind=kind, distribution=distribution):
                    base = yaml.safe_load((ROOT / f'configs/thesis/ed/bigram_handwriting_{distribution}.yaml').read_text())
                    cfg = yaml.safe_load((ROOT / f'configs/thesis/ed/{kind}_handwriting_{distribution}.yaml').read_text())
                    self.assertEqual({k: v for k, v in base.items() if k not in allowed},
                                     {k: v for k, v in cfg.items() if k not in allowed})
                    self.assertEqual(cfg['idx_fold'], 0)
                    self.assertEqual(cfg['dir_work'], f'results/thesis/ed/handwriting_{kind}_{distribution}')
                    cfg['dir_tokenizer'] = str(ROOT / cfg['dir_tokenizer'])
                    self.assertEqual(load_configured_tokenizer(Namespace(**cfg)).size, 224)


if __name__ == '__main__':
    unittest.main()
