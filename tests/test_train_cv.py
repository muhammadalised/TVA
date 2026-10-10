import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

from train_cv import prepare_runs, train_cv


class CrossValidationLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='cv tests ')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset = self.root / 'dataset'
        self.dataset.mkdir()
        for partition in ('train', 'val'):
            (self.dataset / f'{partition}.json').write_text(
                json.dumps({
                    'info': {'num_fold': 2},
                    'annotations': {'0': [{'label': 'a'}], '1': [{'label': 'b'}]},
                }),
                encoding='utf-8',
            )
        self.config = {
            'idx_fold': -1, 'dir_dataset': str(self.dataset),
            'dir_work': str(self.root / 'results'), 'checkpoint': None,
            'tokenizer': 'char', 'dir_tokenizer': None,
            'categories': ['', 'a', 'b'], 'test': False,
        }
        self.main = self.root / 'training script.py'
        self.main.write_text(
            "import sys, yaml\n"
            "from pathlib import Path\n"
            "cfg = yaml.safe_load(Path(sys.argv[sys.argv.index('-c') + 1]).read_text())\n"
            "path = Path(cfg['dir_work'])\n"
            "path.mkdir(parents=True, exist_ok=True)\n"
            "with (path / 'started.txt').open('a') as f:\n"
            "    f.write(str(cfg['idx_fold']) + '\\n')\n"
            "sys.exit(cfg.get('failure_code', 0) if cfg['idx_fold'] == 0 else 0)\n",
            encoding='utf-8',
        )

    def run_quietly(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            train_cv(self.config, str(self.main), **kwargs)

    def test_dry_run_never_starts_training_or_creates_results(self):
        with patch('train_cv.subprocess.run') as child:
            self.run_quietly(dry_run=True)
        child.assert_not_called()
        self.assertFalse(Path(self.config['dir_work']).exists())

    def test_subprocess_handles_spaces_and_runs_folds_sequentially(self):
        self.run_quietly()
        record = Path(self.config['dir_work']) / 'started.txt'
        self.assertEqual(record.read_text().splitlines(), ['0', '1'])

    def test_failed_first_fold_stops_later_folds(self):
        self.config['failure_code'] = 13
        with self.assertRaises(subprocess.CalledProcessError) as raised:
            self.run_quietly()
        self.assertEqual(raised.exception.returncode, 13)
        record = Path(self.config['dir_work']) / 'started.txt'
        self.assertEqual(record.read_text().splitlines(), ['0'])

    def test_existing_later_fold_is_rejected_before_any_training(self):
        output = Path(self.config['dir_work']) / '1'
        output.mkdir(parents=True)
        (output / 'checkpoint.pth').write_bytes(b'previous run')
        with patch('train_cv.subprocess.run') as child:
            with self.assertRaises(FileExistsError):
                self.run_quietly()
        child.assert_not_called()

    def test_missing_later_tokenizer_is_rejected_before_training(self):
        tokenizers = self.root / 'tokenizers'
        tokenizers.mkdir()
        (tokenizers / '0.json').write_text('{}', encoding='utf-8')
        self.config['dir_tokenizer'] = str(tokenizers)
        with patch('train_cv.subprocess.run') as child:
            with self.assertRaises(FileNotFoundError):
                self.run_quietly()
        child.assert_not_called()

    def test_selected_folds_leave_base_configuration_unchanged(self):
        runs = prepare_runs(self.config, [1])
        self.assertEqual([run['idx_fold'] for run in runs], [1])
        self.assertEqual(self.config['idx_fold'], -1)
        for folds in ([], [0, 0], [2]):
            with self.subTest(folds=folds), self.assertRaises(ValueError):
                prepare_runs(self.config, folds)

    def test_cli_propagates_child_failure_exit_code(self):
        self.config['failure_code'] = 13
        config_path = self.root / 'test config.yaml'
        config_path.write_text(yaml.safe_dump(self.config), encoding='utf-8')
        command = [
            sys.executable, str(Path(__file__).resolve().parents[1] / 'train_cv.py'),
            '-c', str(config_path), '-m', str(self.main),
        ]
        result = subprocess.run(command, text=True, capture_output=True)
        self.assertEqual(result.returncode, 13, result.stderr)
        self.assertIn('later folds were not started', result.stderr)

    def test_workstation_config_only_changes_fold_selection_and_output(self):
        root = Path(__file__).resolve().parents[1]
        base = yaml.safe_load(
            (root / 'configs/thesis/b0_char_wi_rh.yaml').read_text(encoding='utf-8-sig')
        )
        cv = yaml.safe_load(
            (root / 'configs/thesis/b0_char_wi_rh_cv.yaml').read_text(encoding='utf-8-sig')
        )
        self.assertEqual(cv['idx_fold'], -1)
        self.assertNotEqual(cv['dir_work'], base['dir_work'])
        self.assertIsNone(cv['checkpoint'])
        self.assertEqual(
            {k: v for k, v in cv.items() if k not in ('idx_fold', 'dir_work')},
            {k: v for k, v in base.items() if k not in ('idx_fold', 'dir_work')},
        )


if __name__ == '__main__':
    unittest.main()
