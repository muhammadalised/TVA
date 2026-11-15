'''Provisional, interpretable force-continuity scores for character pairs.'''

import csv
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm


LOCAL_WEIGHT = 0.75
REGION_WEIGHT = 0.25
DEFAULT_LOCAL_WINDOW_MS = 100
DEFAULT_MIN_OCCURRENCES = 100
DEFAULT_MIN_WRITERS = 30
DEFAULT_MIN_STRATUM_OCCURRENCES = 100
POSITIONS = ('only', 'first', 'middle', 'final')


@dataclass(frozen=True)
class BoundaryEvidence:
    '''Small scoring record retained from one reliable aligned boundary.'''

    pair: str
    sample_index: int
    writer_id: str
    position: str
    left_is_uppercase: bool
    duration_bucket: str
    candidate_duration_ms: float
    used_fallback: bool
    local_contact_preserved: float
    region_contact_preserved: float
    minimum_aligned_probability: float
    minimum_confidence_margin: float


@dataclass
class WriterEvidence:
    '''Residual evidence from one writer for one character pair.'''

    count: int = 0
    local_residual_sum: float = 0.0
    region_residual_sum: float = 0.0

    def add(self, local_residual: float, region_residual: float) -> None:
        self.count += 1
        self.local_residual_sum += local_residual
        self.region_residual_sum += region_residual


@dataclass
class PairEvidence:
    '''Raw and adjusted evidence accumulated for one character pair.'''

    occurrence_count: int = 0
    sample_indices: set[int] = field(default_factory=set)
    writer_evidence: dict[str, WriterEvidence] = field(
        default_factory=lambda: defaultdict(WriterEvidence)
    )
    position_counts: Counter[str] = field(default_factory=Counter)
    uppercase_count: int = 0
    fallback_count: int = 0
    local_contact_sum: float = 0.0
    region_contact_sum: float = 0.0
    durations_ms: list[float] = field(default_factory=list)
    aligned_probabilities: list[float] = field(default_factory=list)
    confidence_margins: list[float] = field(default_factory=list)

    def add(
        self,
        evidence: BoundaryEvidence,
        local_residual: float,
        region_residual: float,
    ) -> None:
        self.occurrence_count += 1
        self.sample_indices.add(evidence.sample_index)
        self.position_counts[evidence.position] += 1
        self.uppercase_count += int(evidence.left_is_uppercase)
        self.fallback_count += int(evidence.used_fallback)
        self.local_contact_sum += evidence.local_contact_preserved
        self.region_contact_sum += evidence.region_contact_preserved
        self.durations_ms.append(evidence.candidate_duration_ms)
        self.aligned_probabilities.append(
            evidence.minimum_aligned_probability
        )
        self.confidence_margins.append(evidence.minimum_confidence_margin)
        self.writer_evidence[evidence.writer_id].add(
            local_residual,
            region_residual,
        )


def _boundary_position(row: dict[str, Any]) -> str:
    boundary_index = int(row['boundary_index'])
    num_boundaries = int(row['num_boundaries_in_sample'])
    if num_boundaries <= 0 or not 0 <= boundary_index < num_boundaries:
        raise ValueError('Invalid boundary position in JSONL record.')
    if num_boundaries == 1:
        return 'only'
    if boundary_index == 0:
        return 'first'
    if boundary_index == num_boundaries - 1:
        return 'final'
    return 'middle'


def _duration_bucket(region: dict[str, Any]) -> str:
    '''Use broad bins so the correction groups retain useful support.'''
    if region.get('used_fallback_window', False):
        return 'empty_region_fallback'
    duration_ms = float(region['candidate_duration_ms'])
    if duration_ms <= 160:
        return 'short_0_160_ms'
    if duration_ms <= 320:
        return 'medium_161_320_ms'
    if duration_ms <= 480:
        return 'long_321_480_ms'
    return 'very_long_over_480_ms'


def _read_reliable_evidence(
    input_path: Path,
    local_window_ms: int,
) -> tuple[list[BoundaryEvidence], dict[str, Any], int]:
    evidence: list[BoundaryEvidence] = []
    provenance: dict[str, Any] | None = None
    total_boundaries = 0
    seen_boundary_ids: set[str] = set()

    with open(input_path, 'r', encoding='utf-8') as file:
        progress = tqdm(file, desc='Reading continuity evidence', unit='boundary')
        for line_number, line in enumerate(progress, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f'Malformed JSON at line {line_number}.'
                ) from error

            total_boundaries += 1
            boundary_id = str(row['boundary_id'])
            if boundary_id in seen_boundary_ids:
                raise ValueError(f'Duplicate boundary ID {boundary_id!r}.')
            seen_boundary_ids.add(boundary_id)

            row_provenance = {
                name: row.get(name)
                for name in (
                    'schema_version',
                    'config',
                    'checkpoint',
                    'checkpoint_epoch',
                    'split',
                    'fold',
                    'sample_rate_hz',
                    'low_force_threshold_fraction',
                )
            }
            if provenance is None:
                provenance = row_provenance
                if provenance['split'] != 'train':
                    raise ValueError(
                        'Continuity scores must be learned from the training '
                        'split only.'
                    )
            elif row_provenance != provenance:
                raise ValueError(
                    f'Inconsistent provenance at JSONL line {line_number}.'
                )

            windows = row.get('windows_ms', {})
            local_window = windows.get(str(local_window_ms))
            region = row.get('candidate_region')
            if local_window is None:
                raise ValueError(
                    f'Missing {local_window_ms} ms window at line '
                    f'{line_number}.'
                )
            if region is None:
                raise ValueError(
                    f'Missing candidate-region features at line '
                    f'{line_number}.'
                )

            reliable = (
                row['both_anchors_agree_with_greedy']
                and not row['overlaps_padding']
                and not local_window['clipped_at_recording_edge']
                and not region['clipped_at_recording_edge']
            )
            if not reliable:
                continue

            evidence.append(
                BoundaryEvidence(
                    pair=str(row['pair']),
                    sample_index=int(row['sample_index']),
                    writer_id=str(row['writer_id']),
                    position=_boundary_position(row),
                    left_is_uppercase=str(row['left_character']).isupper(),
                    duration_bucket=_duration_bucket(region),
                    candidate_duration_ms=float(
                        region['candidate_duration_ms']
                    ),
                    used_fallback=bool(
                        region.get('used_fallback_window', False)
                    ),
                    local_contact_preserved=float(
                        local_window['low_force_fraction'] == 0
                    ),
                    region_contact_preserved=float(
                        region['low_force_fraction'] == 0
                    ),
                    minimum_aligned_probability=float(
                        row['minimum_aligned_probability']
                    ),
                    minimum_confidence_margin=float(
                        row['minimum_confidence_margin']
                    ),
                )
            )

    if total_boundaries == 0 or provenance is None:
        raise ValueError('Boundary JSONL contains no records.')
    if not evidence:
        raise ValueError('No boundaries passed the transparent quality filter.')
    return evidence, provenance, total_boundaries


def _rate_baselines(
    evidence: list[BoundaryEvidence],
) -> tuple[
    dict[tuple[str, str], tuple[int, float, float]],
    dict[str, tuple[int, float, float]],
    tuple[int, float, float],
]:
    '''Calculate expected contact rates for increasingly broad groups.'''
    strata: dict[tuple[str, str], list[float]] = defaultdict(
        lambda: [0.0, 0.0, 0.0]
    )
    positions: dict[str, list[float]] = defaultdict(
        lambda: [0.0, 0.0, 0.0]
    )
    global_values = [0.0, 0.0, 0.0]

    for item in evidence:
        key = (item.position, item.duration_bucket)
        for values in (strata[key], positions[item.position], global_values):
            values[0] += 1
            values[1] += item.local_contact_preserved
            values[2] += item.region_contact_preserved

    def finish(values: list[float]) -> tuple[int, float, float]:
        count = int(values[0])
        return count, values[1] / count, values[2] / count

    return (
        {key: finish(values) for key, values in strata.items()},
        {key: finish(values) for key, values in positions.items()},
        finish(global_values),
    )


def _expected_rates(
    item: BoundaryEvidence,
    strata: dict[tuple[str, str], tuple[int, float, float]],
    positions: dict[str, tuple[int, float, float]],
    global_rates: tuple[int, float, float],
    min_stratum_occurrences: int,
) -> tuple[float, float, str]:
    stratum = strata[(item.position, item.duration_bucket)]
    if stratum[0] >= min_stratum_occurrences:
        return stratum[1], stratum[2], 'position_x_duration'
    position = positions[item.position]
    if position[0] >= min_stratum_occurrences:
        return position[1], position[2], 'position'
    return global_rates[1], global_rates[2], 'global'


def _clip_probability(value: float) -> float:
    return min(1.0, max(0.0, value))


def _median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def score_boundary_jsonl(
    input_path: str | Path,
    *,
    local_window_ms: int = DEFAULT_LOCAL_WINDOW_MS,
    min_occurrences: int = DEFAULT_MIN_OCCURRENCES,
    min_writers: int = DEFAULT_MIN_WRITERS,
    min_stratum_occurrences: int = DEFAULT_MIN_STRATUM_OCCURRENCES,
    local_weight: float = LOCAL_WEIGHT,
    region_weight: float = REGION_WEIGHT,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    '''Create a provisional force-only ranking from reliable train boundaries.

    The correction subtracts the expected contact-preservation rate for the
    same boundary position and broad CTC-region duration. Pair residuals are
    averaged per writer before writers are averaged, preventing prolific
    writers from dominating a pair score.
    '''
    if min_occurrences <= 0 or min_writers <= 0:
        raise ValueError('Support thresholds must be positive.')
    if min_stratum_occurrences <= 0:
        raise ValueError('min_stratum_occurrences must be positive.')
    if local_weight < 0 or region_weight < 0:
        raise ValueError('Continuity weights cannot be negative.')
    if not np.isclose(local_weight + region_weight, 1.0):
        raise ValueError('Local and region weights must sum to one.')

    input_path = Path(input_path)
    evidence, provenance, total_boundaries = _read_reliable_evidence(
        input_path,
        local_window_ms,
    )
    strata, positions, global_rates = _rate_baselines(evidence)
    pair_evidence: dict[str, PairEvidence] = defaultdict(PairEvidence)
    baseline_source_counts: Counter[str] = Counter()

    for item in evidence:
        expected_local, expected_region, source = _expected_rates(
            item,
            strata,
            positions,
            global_rates,
            min_stratum_occurrences,
        )
        baseline_source_counts[source] += 1
        pair_evidence[item.pair].add(
            item,
            item.local_contact_preserved - expected_local,
            item.region_contact_preserved - expected_region,
        )

    rows: list[dict[str, Any]] = []
    for pair, accumulated in pair_evidence.items():
        writer_local_scores = []
        writer_region_scores = []
        for writer in accumulated.writer_evidence.values():
            writer_local_scores.append(
                _clip_probability(
                    global_rates[1]
                    + writer.local_residual_sum / writer.count
                )
            )
            writer_region_scores.append(
                _clip_probability(
                    global_rates[2]
                    + writer.region_residual_sum / writer.count
                )
            )

        adjusted_local = float(np.mean(writer_local_scores))
        adjusted_region = float(np.mean(writer_region_scores))
        score = local_weight * adjusted_local + region_weight * adjusted_region
        writer_scores = np.asarray(
            [
                local_weight * local + region_weight * region
                for local, region in zip(
                    writer_local_scores,
                    writer_region_scores,
                )
            ],
            dtype=np.float64,
        )
        writer_q25, writer_q75 = np.quantile(writer_scores, [0.25, 0.75])
        eligible = (
            accumulated.occurrence_count >= min_occurrences
            and len(accumulated.writer_evidence) >= min_writers
        )
        reasons = []
        if accumulated.occurrence_count < min_occurrences:
            reasons.append(f'fewer_than_{min_occurrences}_occurrences')
        if len(accumulated.writer_evidence) < min_writers:
            reasons.append(f'fewer_than_{min_writers}_writers')

        count = accumulated.occurrence_count
        row = {
            'rank': None,
            'pair': pair,
            'eligible': eligible,
            'ineligible_reasons': '|'.join(reasons),
            'provisional_force_continuity_score': score,
            'adjusted_local_contact_score': adjusted_local,
            'adjusted_region_contact_score': adjusted_region,
            'raw_local_contact_rate': accumulated.local_contact_sum / count,
            'raw_region_contact_rate': accumulated.region_contact_sum / count,
            'writer_score_median': float(np.median(writer_scores)),
            'writer_score_iqr': float(writer_q75 - writer_q25),
            'occurrence_count': count,
            'sample_count': len(accumulated.sample_indices),
            'writer_count': len(accumulated.writer_evidence),
            'minimum_aligned_probability_median': _median(
                accumulated.aligned_probabilities
            ),
            'minimum_confidence_margin_median': _median(
                accumulated.confidence_margins
            ),
            'candidate_duration_ms_median': _median(
                accumulated.durations_ms
            ),
            'fallback_window_rate': accumulated.fallback_count / count,
            'left_uppercase_rate': accumulated.uppercase_count / count,
        }
        for position in POSITIONS:
            row[f'position_{position}_rate'] = (
                accumulated.position_counts[position] / count
            )
        rows.append(row)

    eligible_rows = sorted(
        (row for row in rows if row['eligible']),
        key=lambda row: (
            -row['provisional_force_continuity_score'],
            row['pair'],
        ),
    )
    for rank, row in enumerate(eligible_rows, start=1):
        row['rank'] = rank
    rows.sort(
        key=lambda row: (
            not row['eligible'],
            row['rank'] if row['rank'] is not None else 0,
            row['pair'],
        )
    )

    summary = {
        'schema_version': 1,
        'input_jsonl': str(input_path),
        'provenance': provenance,
        'method_status': (
            'Provisional force-only development score; weights and thresholds '
            'are not frozen thesis hyperparameters.'
        ),
        'quality_filter': (
            'Both anchors agree with the greedy class; no padding overlap; '
            'the local window and candidate region are not clipped.'
        ),
        'total_boundaries': total_boundaries,
        'reliable_boundaries': len(evidence),
        'unique_scored_pairs': len(rows),
        'eligible_pairs': len(eligible_rows),
        'parameters': {
            'local_window_ms': local_window_ms,
            'local_weight': local_weight,
            'region_weight': region_weight,
            'min_occurrences': min_occurrences,
            'min_writers': min_writers,
            'min_stratum_occurrences': min_stratum_occurrences,
            'duration_buckets': {
                'empty_region_fallback': 'Marked empty-region fallback.',
                'short_0_160_ms': 'Non-empty duration <= 160 ms.',
                'medium_161_320_ms': 'Duration > 160 and <= 320 ms.',
                'long_321_480_ms': 'Duration > 320 and <= 480 ms.',
                'very_long_over_480_ms': 'Duration > 480 ms.',
            },
        },
        'correction': {
            'description': (
                'Subtract the expected contact-preservation rate for the same '
                'boundary position and duration bucket. Fall back to position '
                'or global expectation when a group is too small. Average '
                'residuals within writer, then give every writer equal weight.'
            ),
            'baseline_source_counts': dict(baseline_source_counts),
            'global_local_contact_rate': global_rates[1],
            'global_region_contact_rate': global_rates[2],
            'strata': [
                {
                    'position': position,
                    'duration_bucket': duration_bucket,
                    'occurrence_count': values[0],
                    'local_contact_rate': values[1],
                    'region_contact_rate': values[2],
                }
                for (position, duration_bucket), values in sorted(
                    strata.items()
                )
            ],
        },
        'top_eligible_pairs': [
            {
                'rank': row['rank'],
                'pair': row['pair'],
                'score': row['provisional_force_continuity_score'],
                'occurrence_count': row['occurrence_count'],
                'writer_count': row['writer_count'],
            }
            for row in eligible_rows[:20]
        ],
    }
    return summary, rows


def write_continuity_scores(
    output_dir: str | Path,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    '''Atomically write the method report and pair ranking.'''
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / 'continuity_score_summary.json'
    score_path = output_dir / 'pair_continuity_scores.csv'
    existing = [path for path in (summary_path, score_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f'{existing[0]} already exists. Use --overwrite to replace it.'
        )

    temporary_summary = summary_path.with_name(summary_path.name + '.tmp')
    with open(temporary_summary, 'w', encoding='utf-8') as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    os.replace(temporary_summary, summary_path)

    temporary_scores = score_path.with_name(score_path.name + '.tmp')
    with open(temporary_scores, 'w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_scores, score_path)
    return summary_path, score_path
