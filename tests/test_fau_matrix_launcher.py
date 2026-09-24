import tempfile
import unittest
from pathlib import Path

from run_fau_matrix import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CONDITIONS,
    DEFAULT_FOLDS,
    ensure_outputs_absent,
    load_effective_configs,
    validate_folds,
)


class FauMatrixLauncherTests(unittest.TestCase):
    def test_defaults_cover_remaining_four_by_four_matrix(self):
        runs = load_effective_configs(
            DEFAULT_FOLDS,
            DEFAULT_CONDITIONS,
            DEFAULT_BATCH_SIZE,
        )
        self.assertEqual(len(runs), 16)
        self.assertEqual({fold for _, fold, _, _ in runs}, {1, 2, 3, 4})
        self.assertNotIn(0, {fold for _, fold, _, _ in runs})
        self.assertEqual(
            {condition for condition, _, _, _ in runs},
            set(DEFAULT_CONDITIONS),
        )

    def test_effective_configs_freeze_laptop_protocol(self):
        runs = load_effective_configs([3], DEFAULT_CONDITIONS, 32)
        for condition, fold, config, output in runs:
            with self.subTest(condition=condition):
                self.assertEqual(fold, 3)
                self.assertEqual(config['idx_fold'], 3)
                self.assertEqual(config['size_batch'], 32)
                self.assertEqual(config['seed'], 42)
                self.assertEqual(config['epoch'], 300)
                self.assertEqual(config['epoch_warmup'], 30)
                self.assertIsNone(config['checkpoint'])
                self.assertEqual(output.name, '3')

    def test_fold_validation_rejects_duplicates_and_out_of_range(self):
        with self.assertRaises(ValueError):
            validate_folds([1, 1])
        with self.assertRaises(ValueError):
            validate_folds([5])
        with self.assertRaises(ValueError):
            validate_folds([])

    def test_preflight_rejects_existing_output_before_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'fold'
            output.mkdir()
            (output / 'partial.log').write_text('partial', encoding='utf-8')
            run = ('handwriting_wd', 1, {}, output)
            with self.assertRaises(FileExistsError):
                ensure_outputs_absent([run])

    def test_preflight_accepts_absent_and_empty_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            empty = Path(temporary) / 'empty'
            empty.mkdir()
            absent = Path(temporary) / 'absent'
            runs = [
                ('handwriting_wd', 1, {}, empty),
                ('linguistic_wd', 1, {}, absent),
            ]
            ensure_outputs_absent(runs)


if __name__ == '__main__':
    unittest.main()
