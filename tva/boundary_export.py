'''Helpers for exporting one JSONL record per character boundary.'''

import json
import os
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from .alignment_analysis import AlignmentAnalysis
from .boundary_features import BoundaryFeatureAnalysis
from .ctc_alignment import CTCAlignment

__all__ = [
    'BoundaryJSONLWriter',
    'load_boundary_rows',
    'make_boundary_rows',
]


def make_boundary_rows(
    *,
    config_path: str,
    checkpoint_path: str,
    checkpoint_epoch: int | None,
    split: str,
    fold: int | str,
    sample_index: int,
    annotation: dict[str, Any],
    known_label: str,
    greedy_prediction: str,
    raw_num_samples: int,
    model_input_num_samples: int,
    alignment: CTCAlignment,
    analysis: AlignmentAnalysis,
    features: BoundaryFeatureAnalysis,
) -> list[dict[str, Any]]:
    '''Build one self-contained JSON record for each adjacent-character pair.'''
    if len(analysis.boundaries) != len(features.boundaries):
        raise ValueError('Alignment and feature boundary counts do not match.')

    sample_fields = {
        'schema_version': 2,
        'config': config_path,
        'checkpoint': checkpoint_path,
        'checkpoint_epoch': checkpoint_epoch,
        'split': split,
        'fold': fold,
        'sample_index': sample_index,
        'source_file': annotation['filename'],
        'writer_id': annotation['id_writer'],
        'known_label': known_label,
        'greedy_prediction': greedy_prediction,
        'raw_num_samples': raw_num_samples,
        'model_input_num_samples': model_input_num_samples,
        'model_num_frames': analysis.num_model_frames,
        'downsampling_ratio': analysis.downsampling_ratio,
        'sample_rate_hz': analysis.sample_rate_hz,
        'trailing_unmodeled_samples': analysis.trailing_unmodeled_samples,
        'alignment_log_score': alignment.log_score,
        'alignment_mean_log_score': (
            alignment.log_score / analysis.num_model_frames
        ),
        'force_reference_raw': features.force_reference_raw,
        'low_force_threshold_raw': features.low_force_threshold_raw,
        'low_force_threshold_fraction': features.low_force_threshold_fraction,
        'empty_region_fallback_ms': features.empty_region_fallback_ms,
    }

    rows = []
    for region, boundary in zip(analysis.boundaries, features.boundaries):
        if region.boundary_index != boundary.boundary_index:
            raise ValueError('Alignment and feature boundary indices differ.')

        row = dict(sample_fields)
        row.update(
            {
                'boundary_id': (
                    f'{fold}:{split}:{sample_index}:'
                    f'{boundary.boundary_index}'
                ),
                'boundary_index': boundary.boundary_index,
                'num_boundaries_in_sample': len(features.boundaries),
                'left_character': region.left_text,
                'right_character': region.right_text,
                'pair': boundary.pair,
                'center_input_sample': boundary.center_input_sample,
                'blank_frames': boundary.blank_frames,
                'blank_duration_ms': boundary.blank_duration_ms,
                'left_aligned_probability': (
                    boundary.left_aligned_probability
                ),
                'right_aligned_probability': (
                    boundary.right_aligned_probability
                ),
                'minimum_aligned_probability': (
                    boundary.minimum_aligned_probability
                ),
                'left_confidence_margin': boundary.left_confidence_margin,
                'right_confidence_margin': boundary.right_confidence_margin,
                'minimum_confidence_margin': (
                    boundary.minimum_confidence_margin
                ),
                'both_anchors_agree_with_greedy': (
                    boundary.both_anchors_agree_with_greedy
                ),
                'overlaps_padding': boundary.overlaps_padding,
                'candidate_region': asdict(boundary.candidate_region),
                'windows_ms': {
                    str(window.window_ms): asdict(window)
                    for window in boundary.windows
                },
            }
        )
        rows.append(row)

    return rows


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    '''Read JSONL while tolerating only an incomplete final write.'''
    if not path.exists():
        return []

    lines = path.read_text(encoding='utf-8').splitlines()
    records = []
    for line_index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if line_index != len(lines) - 1:
                raise ValueError(
                    f'Malformed JSONL record in {path} at line '
                    f'{line_index + 1}.'
                )
            # A killed process can leave its final line half-written. It is
            # safe to discard because progress is recorded only afterwards.

    return records


def load_boundary_rows(path: str | Path) -> list[dict[str, Any]]:
    '''Load exported boundary rows for summaries or later aggregation.'''
    return _read_jsonl(Path(path))


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    '''Atomically replace a JSONL file with the supplied records.'''
    temporary = path.with_name(path.name + '.tmp')
    with open(temporary, 'w', encoding='utf-8') as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + '\n')
    os.replace(temporary, path)


class BoundaryJSONLWriter:
    '''Append sample boundaries and support safe restart after interruption.

    A small progress JSONL is written only after all boundary rows for a sample
    have been flushed. On resume, rows belonging to unfinished samples are
    removed before new work is appended.
    '''

    def __init__(
        self,
        output_path: str | Path,
        *,
        resume: bool = False,
        overwrite: bool = False,
    ) -> None:
        if resume and overwrite:
            raise ValueError('--resume and --overwrite cannot be used together.')

        self.output_path = Path(output_path)
        self.progress_path = self.output_path.with_suffix(
            self.output_path.suffix + '.progress'
        )
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        exists = self.output_path.exists() or self.progress_path.exists()
        if exists and not (resume or overwrite):
            raise FileExistsError(
                f'{self.output_path} already exists. Use --resume to '
                'continue it or --overwrite to start again.'
            )

        if overwrite:
            _write_jsonl(self.output_path, [])
            _write_jsonl(self.progress_path, [])

        self.completed_samples: set[int] = set()
        if resume:
            self._repair_for_resume()
        elif not exists:
            _write_jsonl(self.output_path, [])
            _write_jsonl(self.progress_path, [])

        self._output_file = open(
            self.output_path, 'a', encoding='utf-8'
        )
        self._progress_file = open(
            self.progress_path, 'a', encoding='utf-8'
        )

    def _repair_for_resume(self) -> None:
        '''Keep only samples proven complete by both output files.'''
        progress = _read_jsonl(self.progress_path)
        expected = {
            int(record['sample_index']): int(record['num_boundaries'])
            for record in progress
        }

        boundary_rows = _read_jsonl(self.output_path)
        rows_by_sample: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in boundary_rows:
            rows_by_sample[int(row['sample_index'])].append(row)

        self.completed_samples = {
            sample_index
            for sample_index, expected_count in expected.items()
            if len(rows_by_sample[sample_index]) == expected_count
        }

        repaired_rows = [
            row
            for row in boundary_rows
            if int(row['sample_index']) in self.completed_samples
        ]
        repaired_progress = [
            {
                'sample_index': sample_index,
                'num_boundaries': expected[sample_index],
            }
            for sample_index in sorted(self.completed_samples)
        ]
        _write_jsonl(self.output_path, repaired_rows)
        _write_jsonl(self.progress_path, repaired_progress)

    def write_sample(
        self,
        sample_index: int,
        rows: list[dict[str, Any]],
    ) -> None:
        '''Write every boundary of one sample, then mark the sample complete.'''
        if sample_index in self.completed_samples:
            raise ValueError(f'Sample {sample_index} was already exported.')
        if any(int(row['sample_index']) != sample_index for row in rows):
            raise ValueError('Every row must belong to the supplied sample.')

        for row in rows:
            self._output_file.write(
                json.dumps(row, ensure_ascii=False) + '\n'
            )
        self._output_file.flush()

        progress = {
            'sample_index': sample_index,
            'num_boundaries': len(rows),
        }
        self._progress_file.write(
            json.dumps(progress, ensure_ascii=False) + '\n'
        )
        self._progress_file.flush()
        self.completed_samples.add(sample_index)

    def close(self) -> None:
        self._output_file.close()
        self._progress_file.close()

    def __enter__(self) -> 'BoundaryJSONLWriter':
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
