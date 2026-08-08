'''Threshold-free summaries of exported handwriting boundary measurements.'''

import csv
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

__all__ = [
    'SUBSET_AGREEMENT_FILTERED',
    'analyze_boundary_jsonl',
    'write_boundary_statistics',
]


ALIGNMENT_METRICS = (
    'minimum_aligned_probability',
    'minimum_confidence_margin',
    'alignment_mean_log_score',
    'blank_duration_ms',
)

WINDOW_METRICS = (
    'force_min_relative',
    'force_at_center_relative',
    'force_drop_ratio',
    'low_force_fraction',
    'longest_low_force_ms',
    'af_mean_magnitude',
    'ar_mean_magnitude',
    'gyro_mean_magnitude',
    'motion_derivative_energy',
)

SUBSET_ALL = 'all'
SUBSET_AGREEMENT_FILTERED = 'agreement_nonpadding_unclipped'
SUBSETS = (SUBSET_ALL, SUBSET_AGREEMENT_FILTERED)

PROVENANCE_FIELDS = (
    'schema_version',
    'config',
    'checkpoint',
    'checkpoint_epoch',
    'split',
    'fold',
    'downsampling_ratio',
    'sample_rate_hz',
    'low_force_threshold_fraction',
)


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    '''Return exact descriptive statistics without selecting a threshold.'''
    if not values:
        return {
            'count': 0,
            'mean': None,
            'std': None,
            'min': None,
            'p10': None,
            'p25': None,
            'median': None,
            'p75': None,
            'p90': None,
            'max': None,
        }

    array = np.asarray(values, dtype=np.float64)
    quantiles = np.quantile(array, [0.10, 0.25, 0.50, 0.75, 0.90])
    return {
        'count': len(array),
        'mean': float(array.mean()),
        'std': float(array.std()),
        'min': float(array.min()),
        'p10': float(quantiles[0]),
        'p25': float(quantiles[1]),
        'median': float(quantiles[2]),
        'p75': float(quantiles[3]),
        'p90': float(quantiles[4]),
        'max': float(array.max()),
    }


@dataclass
class GroupAccumulator:
    '''Measurements for one pair, window size, and transparent subset.'''

    occurrence_count: int = 0
    sample_indices: set[int] = field(default_factory=set)
    writer_ids: set[str] = field(default_factory=set)
    anchor_agreement_count: int = 0
    padding_overlap_count: int = 0
    window_clipped_count: int = 0
    no_low_force_count: int = 0
    metrics: dict[str, list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def add(self, row: dict[str, Any], window: dict[str, Any]) -> None:
        self.occurrence_count += 1
        self.sample_indices.add(int(row['sample_index']))
        self.writer_ids.add(str(row['writer_id']))
        self.anchor_agreement_count += int(
            row['both_anchors_agree_with_greedy']
        )
        self.padding_overlap_count += int(row['overlaps_padding'])
        self.window_clipped_count += int(
            window['clipped_at_recording_edge']
        )
        self.no_low_force_count += int(window['low_force_fraction'] == 0)

        for name in ALIGNMENT_METRICS:
            self.metrics[name].append(float(row[name]))
        for name in WINDOW_METRICS:
            self.metrics[name].append(float(window[name]))

    def nested_summary(self, total_writers: int) -> dict[str, Any]:
        return {
            'occurrence_count': self.occurrence_count,
            'sample_count': len(self.sample_indices),
            'writer_count': len(self.writer_ids),
            'writer_coverage_fraction': _safe_rate(
                len(self.writer_ids), total_writers
            ),
            'anchor_agreement_rate': _safe_rate(
                self.anchor_agreement_count, self.occurrence_count
            ),
            'padding_overlap_rate': _safe_rate(
                self.padding_overlap_count, self.occurrence_count
            ),
            'window_clipped_rate': _safe_rate(
                self.window_clipped_count, self.occurrence_count
            ),
            'no_low_force_rate': _safe_rate(
                self.no_low_force_count, self.occurrence_count
            ),
            'metrics': {
                name: _distribution(self.metrics[name])
                for name in ALIGNMENT_METRICS + WINDOW_METRICS
            },
        }

    def flat_summary(
        self,
        pair: str,
        window_ms: int,
        subset: str,
        total_writers: int,
    ) -> dict[str, Any]:
        nested = self.nested_summary(total_writers)
        row = {
            'pair': pair,
            'window_ms': window_ms,
            'subset': subset,
            **{key: value for key, value in nested.items() if key != 'metrics'},
        }
        for metric_name, statistics in nested['metrics'].items():
            for statistic_name, value in statistics.items():
                if statistic_name == 'count':
                    continue
                row[f'{metric_name}_{statistic_name}'] = value
        return row


def _load_export_summary(input_path: Path) -> dict[str, Any] | None:
    summary_path = input_path.with_name('summary.json')
    if not summary_path.exists():
        return None
    with open(summary_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def _validate_provenance(
    expected: dict[str, Any],
    row: dict[str, Any],
    line_number: int,
) -> None:
    for field_name, expected_value in expected.items():
        if row.get(field_name) != expected_value:
            raise ValueError(
                f'Inconsistent {field_name!r} at JSONL line {line_number}.'
            )


def analyze_boundary_jsonl(
    input_path: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    '''Stream a boundary JSONL and return global plus per-pair summaries.'''
    input_path = Path(input_path)
    source_summary = _load_export_summary(input_path)
    expected_rows = (
        int(source_summary['exported_boundaries'])
        if source_summary is not None
        else None
    )

    groups: dict[tuple[str, int, str], GroupAccumulator] = defaultdict(
        GroupAccumulator
    )
    global_groups: dict[tuple[int, str], GroupAccumulator] = defaultdict(
        GroupAccumulator
    )
    pair_counts: Counter[str] = Counter()
    sample_indices: set[int] = set()
    writer_ids: set[str] = set()
    boundary_ids: set[str] = set()
    provenance: dict[str, Any] | None = None
    window_sizes: tuple[int, ...] | None = None
    row_count = 0

    with open(input_path, 'r', encoding='utf-8') as file:
        progress = tqdm(
            file,
            total=expected_rows,
            desc='Analyzing boundaries',
            unit='boundary',
        )
        for line_number, line in enumerate(progress, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f'Malformed JSON at line {line_number}.'
                ) from error

            if provenance is None:
                provenance = {
                    name: row.get(name) for name in PROVENANCE_FIELDS
                }
            else:
                _validate_provenance(provenance, row, line_number)

            row_windows = tuple(sorted(int(size) for size in row['windows_ms']))
            if window_sizes is None:
                window_sizes = row_windows
            elif row_windows != window_sizes:
                raise ValueError(
                    f'Inconsistent window sizes at JSONL line {line_number}.'
                )

            boundary_id = str(row['boundary_id'])
            if boundary_id in boundary_ids:
                raise ValueError(f'Duplicate boundary ID {boundary_id!r}.')
            boundary_ids.add(boundary_id)

            pair = str(row['pair'])
            pair_counts[pair] += 1
            sample_indices.add(int(row['sample_index']))
            writer_ids.add(str(row['writer_id']))
            row_count += 1

            for window_ms in window_sizes:
                window = row['windows_ms'][str(window_ms)]
                groups[(pair, window_ms, SUBSET_ALL)].add(row, window)
                global_groups[(window_ms, SUBSET_ALL)].add(row, window)

                include_in_agreement_subset = (
                    row['both_anchors_agree_with_greedy']
                    and not row['overlaps_padding']
                    and not window['clipped_at_recording_edge']
                )
                if include_in_agreement_subset:
                    groups[
                        (pair, window_ms, SUBSET_AGREEMENT_FILTERED)
                    ].add(row, window)
                    global_groups[
                        (window_ms, SUBSET_AGREEMENT_FILTERED)
                    ].add(row, window)

    if row_count == 0 or provenance is None or window_sizes is None:
        raise ValueError('Boundary JSONL contains no records.')
    if expected_rows is not None and row_count != expected_rows:
        raise ValueError(
            f'Export summary expects {expected_rows} boundaries, but the '
            f'JSONL contains {row_count}.'
        )
    if source_summary is not None:
        for field_name in (
            'config',
            'checkpoint',
            'checkpoint_epoch',
            'split',
            'fold',
        ):
            if (
                field_name in source_summary
                and source_summary[field_name] != provenance[field_name]
            ):
                raise ValueError(
                    f'Export summary and JSONL disagree on {field_name!r}.'
                )
        if 'pair_counts' in source_summary:
            expected_pair_counts = {
                str(pair): int(count)
                for pair, count in source_summary['pair_counts'].items()
            }
            if dict(pair_counts) != expected_pair_counts:
                raise ValueError(
                    'Export summary pair counts do not match the JSONL.'
                )

    total_writers = len(writer_ids)
    pair_rows = []
    for pair in sorted(pair_counts):
        for window_ms in window_sizes:
            for subset in SUBSETS:
                group = groups[(pair, window_ms, subset)]
                pair_rows.append(
                    group.flat_summary(
                        pair, window_ms, subset, total_writers
                    )
                )

    pair_support = list(pair_counts.values())
    summary = {
        'schema_version': 1,
        'input_jsonl': str(input_path),
        'provenance': provenance,
        'source_export_summary': source_summary,
        'total_boundaries': row_count,
        'samples_with_boundaries': len(sample_indices),
        'total_writers': total_writers,
        'unique_pairs': len(pair_counts),
        'window_sizes_ms': list(window_sizes),
        'subsets': {
            SUBSET_ALL: 'Every exported boundary occurrence.',
            SUBSET_AGREEMENT_FILTERED: (
                'Both target-character anchors agree with the local model '
                'choice, neither overlaps padding, and this window is not '
                'clipped. No probability or margin threshold is applied.'
            ),
        },
        'pair_occurrence_distribution': _distribution(pair_support),
        'pairs_with_at_least': {
            '40_occurrences': sum(count >= 40 for count in pair_support),
            '100_occurrences': sum(count >= 100 for count in pair_support),
            '500_occurrences': sum(count >= 500 for count in pair_support),
        },
        'global_statistics': {
            str(window_ms): {
                subset: global_groups[(window_ms, subset)].nested_summary(
                    total_writers
                )
                for subset in SUBSETS
            }
            for window_ms in window_sizes
        },
    }
    return summary, pair_rows


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + '.tmp')
    with open(temporary, 'w', encoding='utf-8') as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def _atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + '.tmp')
    with open(temporary, 'w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _make_overview_rows(pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    '''Keep the most interpretable columns in a compact companion table.'''
    columns = (
        'pair',
        'window_ms',
        'subset',
        'occurrence_count',
        'sample_count',
        'writer_count',
        'writer_coverage_fraction',
        'anchor_agreement_rate',
        'padding_overlap_rate',
        'window_clipped_rate',
        'no_low_force_rate',
        'minimum_aligned_probability_median',
        'minimum_confidence_margin_median',
        'blank_duration_ms_median',
        'force_min_relative_p25',
        'force_min_relative_median',
        'force_min_relative_p75',
        'low_force_fraction_median',
        'longest_low_force_ms_median',
        'motion_derivative_energy_p25',
        'motion_derivative_energy_median',
        'motion_derivative_energy_p75',
    )
    return [
        {column: row[column] for column in columns}
        for row in pair_rows
    ]


def write_boundary_statistics(
    output_dir: str | Path,
    summary: dict[str, Any],
    pair_rows: list[dict[str, Any]],
    *,
    overwrite: bool = False,
) -> tuple[Path, Path, Path]:
    '''Write the global report plus detailed and compact pair CSV files.'''
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / 'analysis_summary.json'
    pair_path = output_dir / 'pair_statistics.csv'
    overview_path = output_dir / 'pair_overview.csv'

    existing = [
        path
        for path in (summary_path, pair_path, overview_path)
        if path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            f'{existing[0]} already exists. Use --overwrite to replace the '
            'previous analysis.'
        )

    _atomic_write_json(summary_path, summary)
    _atomic_write_csv(pair_path, pair_rows)
    _atomic_write_csv(overview_path, _make_overview_rows(pair_rows))
    return summary_path, pair_path, overview_path
