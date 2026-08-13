import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tva.alignment_analysis import analyze_alignment
from tva.boundary_export import (
    BoundaryJSONLWriter,
    load_boundary_rows,
    make_boundary_rows,
)
from tva.boundary_features import extract_boundary_features
from tva.ctc_alignment import ctc_viterbi_align


def token_text(token_id: int) -> str:
    return {0: '<blank>', 1: 'Ä', 2: 'B'}[token_id]


class BoundaryExportTest(unittest.TestCase):
    def make_rows(self) -> list[dict]:
        probabilities = torch.tensor(
            [
                [0.05, 0.90, 0.05],
                [0.90, 0.05, 0.05],
                [0.05, 0.05, 0.90],
            ]
        )
        alignment = ctc_viterbi_align(probabilities, [1, 2])
        analysis = analyze_alignment(
            alignment,
            probabilities,
            token_text,
            downsampling_ratio=8,
            num_model_input_samples=24,
            num_raw_samples=24,
            sample_rate_hz=100,
        )
        raw_signal = np.ones((24, 13), dtype=np.float32) * 100
        normalized_signal = np.zeros((24, 13), dtype=np.float32)
        features = extract_boundary_features(
            raw_signal,
            normalized_signal,
            analysis,
            window_sizes_ms=(100,),
        )

        return make_boundary_rows(
            config_path='config.yaml',
            checkpoint_path='best.pth',
            checkpoint_epoch=5,
            split='train',
            fold=0,
            sample_index=7,
            annotation={
                'filename': 'data/0/train/00000007.csv',
                'id_writer': 3,
            },
            known_label='ÄB',
            greedy_prediction='ÄB',
            raw_num_samples=24,
            model_input_num_samples=24,
            alignment=alignment,
            analysis=analysis,
            features=features,
        )

    def test_builds_one_self_contained_row_per_boundary(self) -> None:
        rows = self.make_rows()

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row['boundary_id'], '0:train:7:0')
        self.assertEqual(row['pair'], 'ÄB')
        self.assertEqual(row['known_label'], 'ÄB')
        self.assertIn('100', row['windows_ms'])
        self.assertEqual(row['schema_version'], 2)
        self.assertIn('candidate_region', row)
        self.assertEqual(
            row['candidate_region']['candidate_duration_ms'],
            row['blank_duration_ms'],
        )
        self.assertEqual(
            row['windows_ms']['100']['force_min_relative'], 1.0
        )

    def test_writer_preserves_unicode_and_completed_samples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'boundaries.jsonl'
            rows = self.make_rows()

            with BoundaryJSONLWriter(output) as writer:
                writer.write_sample(7, rows)
                # A one-character label legitimately has no boundaries.
                writer.write_sample(8, [])

            loaded = load_boundary_rows(output)
            self.assertEqual(loaded[0]['pair'], 'ÄB')
            progress = [
                json.loads(line)
                for line in output.with_suffix('.jsonl.progress')
                .read_text(encoding='utf-8')
                .splitlines()
            ]
            self.assertEqual(progress[-1], {
                'sample_index': 8,
                'num_boundaries': 0,
            })

    def test_resume_removes_rows_from_an_unfinished_sample(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'boundaries.jsonl'
            rows = self.make_rows()

            with BoundaryJSONLWriter(output) as writer:
                writer.write_sample(7, rows)

            unfinished = dict(rows[0], sample_index=9)
            with open(output, 'a', encoding='utf-8') as file:
                file.write(json.dumps(unfinished) + '\n')

            with BoundaryJSONLWriter(output, resume=True) as writer:
                self.assertEqual(writer.completed_samples, {7})
                writer.write_sample(9, [unfinished])

            loaded = load_boundary_rows(output)
            self.assertEqual(
                [row['sample_index'] for row in loaded],
                [7, 9],
            )

    def test_existing_output_requires_an_explicit_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'boundaries.jsonl'
            with BoundaryJSONLWriter(output):
                pass

            with self.assertRaisesRegex(FileExistsError, '--resume'):
                BoundaryJSONLWriter(output)


if __name__ == '__main__':
    unittest.main()
