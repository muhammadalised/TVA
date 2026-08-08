import csv
import json
import tempfile
import unittest
from pathlib import Path

from tva.boundary_statistics import (
    SUBSET_AGREEMENT_FILTERED,
    analyze_boundary_jsonl,
    write_boundary_statistics,
)


def make_row(
    boundary_id: str,
    pair: str,
    sample_index: int,
    writer_id: int,
    *,
    agrees: bool = True,
    overlaps_padding: bool = False,
    clipped_100: bool = False,
    force_100: float = 0.5,
) -> dict:
    def window(window_ms: int, clipped: bool, force: float) -> dict:
        return {
            'window_ms': window_ms,
            'clipped_at_recording_edge': clipped,
            'force_min_relative': force,
            'force_at_center_relative': force + 0.1,
            'force_drop_ratio': 1 - force,
            'low_force_fraction': 0.0 if force > 0.1 else 0.4,
            'longest_low_force_ms': 0.0 if force > 0.1 else 40.0,
            'af_mean_magnitude': 1.0,
            'ar_mean_magnitude': 2.0,
            'gyro_mean_magnitude': 3.0,
            'motion_derivative_energy': 4.0,
        }

    return {
        'schema_version': 1,
        'config': 'config.yaml',
        'checkpoint': 'best.pth',
        'checkpoint_epoch': 5,
        'split': 'train',
        'fold': 0,
        'downsampling_ratio': 8,
        'sample_rate_hz': 100.0,
        'low_force_threshold_fraction': 0.1,
        'boundary_id': boundary_id,
        'sample_index': sample_index,
        'writer_id': writer_id,
        'pair': pair,
        'minimum_aligned_probability': 0.8,
        'minimum_confidence_margin': 0.6 if agrees else -0.2,
        'alignment_mean_log_score': -0.1,
        'blank_duration_ms': 160.0,
        'both_anchors_agree_with_greedy': agrees,
        'overlaps_padding': overlaps_padding,
        'windows_ms': {
            '50': window(50, False, 0.6),
            '100': window(100, clipped_100, force_100),
        },
    }


class BoundaryStatisticsTest(unittest.TestCase):
    def write_fixture(self, directory: str) -> Path:
        path = Path(directory) / 'boundaries.jsonl'
        rows = [
            make_row('0:train:0:0', 'ab', 0, 1, force_100=0.5),
            make_row(
                '0:train:1:0', 'ab', 1, 2,
                agrees=False, force_100=0.0,
            ),
            make_row(
                '0:train:2:0', 'ab', 2, 1,
                clipped_100=True, force_100=1.0,
            ),
            make_row('0:train:3:0', 'bc', 3, 2, force_100=0.4),
        ]
        with open(path, 'w', encoding='utf-8') as file:
            for row in rows:
                file.write(json.dumps(row) + '\n')

        with open(path.with_name('summary.json'), 'w', encoding='utf-8') as file:
            json.dump({'exported_boundaries': len(rows)}, file)
        return path

    def test_calculates_global_and_pair_subsets_without_margin_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_fixture(directory)
            summary, pair_rows = analyze_boundary_jsonl(path)

        self.assertEqual(summary['total_boundaries'], 4)
        self.assertEqual(summary['unique_pairs'], 2)
        self.assertEqual(summary['total_writers'], 2)

        global_100 = summary['global_statistics']['100']
        self.assertEqual(global_100['all']['occurrence_count'], 4)
        self.assertEqual(
            global_100[SUBSET_AGREEMENT_FILTERED]['occurrence_count'],
            2,
        )

        ab_all = next(
            row for row in pair_rows
            if row['pair'] == 'ab'
            and row['window_ms'] == 100
            and row['subset'] == 'all'
        )
        self.assertEqual(ab_all['occurrence_count'], 3)
        self.assertEqual(ab_all['writer_count'], 2)
        self.assertAlmostEqual(ab_all['anchor_agreement_rate'], 2 / 3)
        self.assertAlmostEqual(ab_all['force_min_relative_median'], 0.5)

        ab_filtered_50 = next(
            row for row in pair_rows
            if row['pair'] == 'ab'
            and row['window_ms'] == 50
            and row['subset'] == SUBSET_AGREEMENT_FILTERED
        )
        ab_filtered_100 = next(
            row for row in pair_rows
            if row['pair'] == 'ab'
            and row['window_ms'] == 100
            and row['subset'] == SUBSET_AGREEMENT_FILTERED
        )
        self.assertEqual(ab_filtered_50['occurrence_count'], 2)
        self.assertEqual(ab_filtered_100['occurrence_count'], 1)

    def test_writes_json_and_csv_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_fixture(directory)
            summary, pair_rows = analyze_boundary_jsonl(path)
            output_dir = Path(directory) / 'analysis'
            summary_path, pair_path, overview_path = write_boundary_statistics(
                output_dir, summary, pair_rows
            )

            with open(summary_path, 'r', encoding='utf-8') as file:
                saved_summary = json.load(file)
            with open(pair_path, 'r', encoding='utf-8') as file:
                saved_rows = list(csv.DictReader(file))
            with open(overview_path, 'r', encoding='utf-8') as file:
                overview_rows = list(csv.DictReader(file))

            self.assertEqual(saved_summary['total_boundaries'], 4)
            self.assertEqual(len(saved_rows), 2 * 2 * 2)
            self.assertEqual(saved_rows[0]['pair'], 'ab')
            self.assertEqual(len(overview_rows), len(saved_rows))
            self.assertLess(len(overview_rows[0]), len(saved_rows[0]))

            with self.assertRaisesRegex(FileExistsError, '--overwrite'):
                write_boundary_statistics(output_dir, summary, pair_rows)

    def test_rejects_duplicate_boundary_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'boundaries.jsonl'
            row = make_row('duplicate', 'ab', 0, 1)
            with open(path, 'w', encoding='utf-8') as file:
                file.write(json.dumps(row) + '\n')
                file.write(json.dumps(row) + '\n')

            with self.assertRaisesRegex(ValueError, 'Duplicate boundary'):
                analyze_boundary_jsonl(path)


if __name__ == '__main__':
    unittest.main()
