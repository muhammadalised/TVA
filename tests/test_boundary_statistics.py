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
    boundary_index: int = 0,
    num_boundaries: int = 3,
    center_input_sample: float = 20.0,
    raw_num_samples: int = 100,
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

    def candidate_region(force: float) -> dict:
        return {
            'candidate_start_sample': 16,
            'candidate_end_sample': 32,
            'candidate_num_samples': 16,
            'candidate_duration_ms': 160.0,
            'start_sample': 16,
            'end_sample': 32,
            'actual_num_samples': 16,
            'actual_duration_ms': 160.0,
            'used_fallback_window': False,
            'fallback_window_ms': 100,
            'clipped_at_recording_edge': clipped_100,
            'force_min_raw': force * 100,
            'force_mean_raw': force * 100,
            'force_at_center_raw': (force + 0.1) * 100,
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
        'schema_version': 2,
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
        'left_character': pair[0],
        'boundary_index': boundary_index,
        'num_boundaries_in_sample': num_boundaries,
        'center_input_sample': center_input_sample,
        'raw_num_samples': raw_num_samples,
        'minimum_aligned_probability': 0.8,
        'minimum_confidence_margin': 0.6 if agrees else -0.2,
        'alignment_mean_log_score': -0.1,
        'blank_duration_ms': 160.0,
        'both_anchors_agree_with_greedy': agrees,
        'overlaps_padding': overlaps_padding,
        'candidate_region': candidate_region(force_100),
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
                agrees=False, force_100=0.0, boundary_index=1,
            ),
            make_row(
                '0:train:2:0', 'ab', 2, 1,
                clipped_100=True, force_100=1.0, boundary_index=2,
            ),
            make_row(
                '0:train:3:0', 'bc', 3, 2,
                force_100=0.4, num_boundaries=1,
            ),
            make_row(
                '0:train:4:0', 'Ab', 4, 2,
                force_100=0.6, boundary_index=0, num_boundaries=2,
            ),
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
            (
                summary,
                pair_rows,
                position_rows,
                region_pair_rows,
                region_position_rows,
            ) = analyze_boundary_jsonl(path)

        self.assertEqual(summary['total_boundaries'], 5)
        self.assertEqual(summary['unique_pairs'], 3)
        self.assertEqual(summary['total_writers'], 2)
        self.assertTrue(summary['has_candidate_region_features'])

        global_100 = summary['global_statistics']['100']
        self.assertEqual(global_100['all']['occurrence_count'], 5)
        self.assertEqual(
            global_100[SUBSET_AGREEMENT_FILTERED]['occurrence_count'],
            3,
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

        positions_100 = {
            row['group_value']: row['occurrence_count']
            for row in position_rows
            if row['group_type'] == 'boundary_position'
            and row['window_ms'] == 100
            and row['subset'] == SUBSET_AGREEMENT_FILTERED
        }
        self.assertEqual(positions_100, {
            'only': 1,
            'first': 2,
            'middle': 0,
            'final': 0,
        })

        first_uppercase = next(
            row for row in position_rows
            if row['group_type'] == 'boundary_position_x_left_case'
            and row['group_value'] == 'first|uppercase'
            and row['window_ms'] == 100
            and row['subset'] == SUBSET_AGREEMENT_FILTERED
        )
        self.assertEqual(first_uppercase['occurrence_count'], 1)

        only = next(
            row for row in position_rows
            if row['group_type'] == 'boundary_position'
            and row['group_value'] == 'only'
            and row['window_ms'] == 100
            and row['subset'] == SUBSET_AGREEMENT_FILTERED
        )
        self.assertAlmostEqual(only['boundary_center_relative_median'], 0.2)
        self.assertAlmostEqual(only['uniform_reference_relative_median'], 0.5)
        self.assertAlmostEqual(only['boundary_relative_offset_median'], -0.3)
        self.assertAlmostEqual(
            only['boundary_absolute_relative_error_median'], 0.3
        )
        self.assertAlmostEqual(only['boundary_center_time_ms_median'], 200.0)
        self.assertAlmostEqual(only['boundary_time_offset_ms_median'], -300.0)

        ab_region = next(
            row for row in region_pair_rows
            if row['pair'] == 'ab' and row['subset'] == 'all'
        )
        self.assertEqual(ab_region['occurrence_count'], 3)
        self.assertEqual(ab_region['candidate_duration_ms_median'], 160.0)
        self.assertAlmostEqual(ab_region['force_min_relative_median'], 0.5)
        self.assertEqual(len(region_position_rows), 19 * 2)

    def test_writes_json_and_csv_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_fixture(directory)
            (
                summary,
                pair_rows,
                position_rows,
                region_pair_rows,
                region_position_rows,
            ) = analyze_boundary_jsonl(path)
            output_dir = Path(directory) / 'analysis'
            (
                summary_path,
                pair_path,
                overview_path,
                position_path,
                position_overview_path,
                region_pair_path,
                region_pair_overview_path,
                region_position_path,
                region_position_overview_path,
            ) = write_boundary_statistics(
                output_dir,
                summary,
                pair_rows,
                position_rows,
                region_pair_rows,
                region_position_rows,
            )

            with open(summary_path, 'r', encoding='utf-8') as file:
                saved_summary = json.load(file)
            with open(pair_path, 'r', encoding='utf-8') as file:
                saved_rows = list(csv.DictReader(file))
            with open(overview_path, 'r', encoding='utf-8') as file:
                overview_rows = list(csv.DictReader(file))
            with open(position_path, 'r', encoding='utf-8') as file:
                position_statistics = list(csv.DictReader(file))
            with open(position_overview_path, 'r', encoding='utf-8') as file:
                position_overview = list(csv.DictReader(file))
            with open(region_pair_path, 'r', encoding='utf-8') as file:
                region_statistics = list(csv.DictReader(file))
            with open(region_pair_overview_path, 'r', encoding='utf-8') as file:
                region_overview = list(csv.DictReader(file))
            with open(region_position_path, 'r', encoding='utf-8') as file:
                region_position_statistics = list(csv.DictReader(file))
            with open(
                region_position_overview_path, 'r', encoding='utf-8'
            ) as file:
                region_position_overview = list(csv.DictReader(file))

            self.assertEqual(saved_summary['total_boundaries'], 5)
            self.assertEqual(len(saved_rows), 3 * 2 * 2)
            self.assertEqual(saved_rows[0]['pair'], 'Ab')
            self.assertEqual(len(overview_rows), len(saved_rows))
            self.assertLess(len(overview_rows[0]), len(saved_rows[0]))
            self.assertEqual(len(position_statistics), 19 * 2 * 2)
            self.assertEqual(len(position_overview), len(position_statistics))
            self.assertLess(
                len(position_overview[0]), len(position_statistics[0])
            )
            self.assertEqual(len(region_statistics), 3 * 2)
            self.assertEqual(len(region_overview), len(region_statistics))
            self.assertLess(
                len(region_overview[0]), len(region_statistics[0])
            )
            self.assertEqual(len(region_position_statistics), 19 * 2)
            self.assertEqual(
                len(region_position_overview),
                len(region_position_statistics),
            )

            with self.assertRaisesRegex(FileExistsError, '--overwrite'):
                write_boundary_statistics(
                    output_dir,
                    summary,
                    pair_rows,
                    position_rows,
                    region_pair_rows,
                    region_position_rows,
                )

    def test_rejects_duplicate_boundary_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'boundaries.jsonl'
            row = make_row('duplicate', 'ab', 0, 1)
            with open(path, 'w', encoding='utf-8') as file:
                file.write(json.dumps(row) + '\n')
                file.write(json.dumps(row) + '\n')

            with self.assertRaisesRegex(ValueError, 'Duplicate boundary'):
                analyze_boundary_jsonl(path)

    def test_rejects_an_invalid_boundary_position(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'boundaries.jsonl'
            row = make_row(
                'invalid-position',
                'ab',
                0,
                1,
                boundary_index=3,
                num_boundaries=3,
            )
            with open(path, 'w', encoding='utf-8') as file:
                file.write(json.dumps(row) + '\n')

            with self.assertRaisesRegex(ValueError, 'Invalid boundary_index'):
                analyze_boundary_jsonl(path)

    def test_rejects_an_invalid_recording_length(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'boundaries.jsonl'
            row = make_row(
                'invalid-length', 'ab', 0, 1, raw_num_samples=0
            )
            with open(path, 'w', encoding='utf-8') as file:
                file.write(json.dumps(row) + '\n')

            with self.assertRaisesRegex(ValueError, 'raw_num_samples'):
                analyze_boundary_jsonl(path)


if __name__ == '__main__':
    unittest.main()
