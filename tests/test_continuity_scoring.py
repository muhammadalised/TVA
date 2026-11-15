import csv
import json
import tempfile
import unittest
from pathlib import Path

from tva.continuity_scoring import (
    score_boundary_jsonl,
    write_continuity_scores,
)


def make_boundary(
    boundary_id: str,
    pair: str,
    sample_index: int,
    writer_id: int,
    *,
    contact_preserved: bool,
    position: str,
    duration_ms: float,
    split: str = 'train',
) -> dict:
    if position == 'first':
        boundary_index, num_boundaries = 0, 3
    elif position == 'middle':
        boundary_index, num_boundaries = 1, 3
    elif position == 'final':
        boundary_index, num_boundaries = 2, 3
    else:
        boundary_index, num_boundaries = 0, 1

    low_force_fraction = 0.0 if contact_preserved else 0.5
    feature_values = {
        'clipped_at_recording_edge': False,
        'low_force_fraction': low_force_fraction,
    }
    return {
        'schema_version': 2,
        'config': 'config.yaml',
        'checkpoint': 'best.pth',
        'checkpoint_epoch': 5,
        'split': split,
        'fold': 0,
        'sample_rate_hz': 100.0,
        'low_force_threshold_fraction': 0.1,
        'boundary_id': boundary_id,
        'sample_index': sample_index,
        'writer_id': writer_id,
        'pair': pair,
        'left_character': pair[0],
        'boundary_index': boundary_index,
        'num_boundaries_in_sample': num_boundaries,
        'minimum_aligned_probability': 0.9,
        'minimum_confidence_margin': 0.8,
        'both_anchors_agree_with_greedy': True,
        'overlaps_padding': False,
        'windows_ms': {'100': dict(feature_values)},
        'candidate_region': {
            **feature_values,
            'candidate_duration_ms': duration_ms,
            'used_fallback_window': False,
        },
    }


class ContinuityScoringTest(unittest.TestCase):
    def write_fixture(self, directory: str, *, split: str = 'train') -> Path:
        rows = []
        definitions = (
            ('ab', True, 'first', 160.0),
            ('xy', False, 'first', 160.0),
            ('cd', True, 'middle', 400.0),
            ('uv', False, 'middle', 400.0),
        )
        sample_index = 0
        for pair, preserved, position, duration_ms in definitions:
            for writer_id in (1, 2):
                rows.append(
                    make_boundary(
                        f'{pair}:{writer_id}',
                        pair,
                        sample_index,
                        writer_id,
                        contact_preserved=preserved,
                        position=position,
                        duration_ms=duration_ms,
                        split=split,
                    )
                )
                sample_index += 1

        path = Path(directory) / 'boundaries.jsonl'
        with open(path, 'w', encoding='utf-8') as file:
            for row in rows:
                file.write(json.dumps(row) + '\n')
        return path

    def test_corrects_within_position_and_duration_and_ranks_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_fixture(directory)
            summary, rows = score_boundary_jsonl(
                path,
                min_occurrences=2,
                min_writers=2,
                min_stratum_occurrences=1,
            )

        by_pair = {row['pair']: row for row in rows}
        self.assertEqual(summary['reliable_boundaries'], 8)
        self.assertEqual(summary['eligible_pairs'], 4)
        self.assertAlmostEqual(
            by_pair['ab']['provisional_force_continuity_score'], 1.0
        )
        self.assertAlmostEqual(
            by_pair['cd']['provisional_force_continuity_score'], 1.0
        )
        self.assertAlmostEqual(
            by_pair['xy']['provisional_force_continuity_score'], 0.0
        )
        self.assertAlmostEqual(
            by_pair['uv']['provisional_force_continuity_score'], 0.0
        )
        self.assertEqual(by_pair['ab']['rank'], 1)
        self.assertEqual(by_pair['cd']['rank'], 2)

    def test_support_gate_marks_but_retains_ineligible_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_fixture(directory)
            summary, rows = score_boundary_jsonl(
                path,
                min_occurrences=3,
                min_writers=2,
                min_stratum_occurrences=1,
            )

        self.assertEqual(summary['eligible_pairs'], 0)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(not row['eligible'] for row in rows))
        self.assertTrue(all(row['rank'] is None for row in rows))
        self.assertTrue(
            all('fewer_than_3_occurrences' in row['ineligible_reasons']
                for row in rows)
        )

    def test_writes_method_report_and_pair_scores(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_fixture(directory)
            summary, rows = score_boundary_jsonl(
                path,
                min_occurrences=2,
                min_writers=2,
                min_stratum_occurrences=1,
            )
            output_dir = Path(directory) / 'scores'
            summary_path, score_path = write_continuity_scores(
                output_dir,
                summary,
                rows,
            )

            with open(summary_path, 'r', encoding='utf-8') as file:
                saved_summary = json.load(file)
            with open(score_path, 'r', encoding='utf-8') as file:
                saved_rows = list(csv.DictReader(file))

            self.assertEqual(saved_summary['eligible_pairs'], 4)
            self.assertEqual(saved_rows[0]['pair'], 'ab')
            self.assertIn('adjusted_local_contact_score', saved_rows[0])
            with self.assertRaisesRegex(FileExistsError, '--overwrite'):
                write_continuity_scores(output_dir, summary, rows)

    def test_rejects_non_training_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_fixture(directory, split='val')
            with self.assertRaisesRegex(ValueError, 'training split only'):
                score_boundary_jsonl(path)


if __name__ == '__main__':
    unittest.main()
