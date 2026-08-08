from dataclasses import dataclass

import numpy as np

from .alignment_analysis import AlignmentAnalysis, BoundaryRegion

__all__ = [
    'BoundaryFeatureAnalysis',
    'BoundaryFeatures',
    'BoundaryWindowFeatures',
    'extract_boundary_features',
]


@dataclass(frozen=True)
class BoundaryWindowFeatures:
    '''Sensor features from one fixed window around a boundary midpoint.'''

    window_ms: int
    requested_num_samples: int
    start_sample: int
    end_sample: int
    actual_num_samples: int
    clipped_at_recording_edge: bool
    force_min_raw: float
    force_mean_raw: float
    force_at_center_raw: float
    force_min_relative: float
    force_at_center_relative: float
    force_drop_ratio: float
    low_force_fraction: float
    longest_low_force_ms: float
    af_mean_magnitude: float
    ar_mean_magnitude: float
    gyro_mean_magnitude: float
    motion_derivative_energy: float


@dataclass(frozen=True)
class BoundaryFeatures:
    '''Alignment reliability and multi-window sensor features for one pair.'''

    boundary_index: int
    pair: str
    center_input_sample: float
    blank_frames: int
    blank_duration_ms: float
    left_aligned_probability: float
    right_aligned_probability: float
    minimum_aligned_probability: float
    left_confidence_margin: float
    right_confidence_margin: float
    minimum_confidence_margin: float
    both_anchors_agree_with_greedy: bool
    overlaps_padding: bool
    windows: list[BoundaryWindowFeatures]


@dataclass(frozen=True)
class BoundaryFeatureAnalysis:
    '''Recording-level settings and extracted features for all boundaries.'''

    force_reference_raw: float
    low_force_threshold_raw: float
    low_force_threshold_fraction: float
    window_sizes_ms: list[int]
    boundaries: list[BoundaryFeatures]


def _longest_true_run(values: np.ndarray) -> int:
    '''Length of the longest consecutive True section.'''
    longest = 0
    current = 0

    for value in values:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    return longest


def _window_bounds(
    center_sample: float,
    window_ms: int,
    sample_rate_hz: float,
    num_samples: int,
) -> tuple[int, int, int, bool]:
    '''Return a fixed-size, midpoint-centred, recording-clipped window.'''
    requested = max(1, round(window_ms / 1000 * sample_rate_hz))
    center_rounded = round(center_sample)
    center_clipped = min(max(center_rounded, 0), num_samples - 1)
    start_unclipped = center_clipped - requested // 2
    end_unclipped = start_unclipped + requested
    start = max(0, start_unclipped)
    end = min(num_samples, end_unclipped)
    clipped = (
        center_clipped != center_rounded
        or start != start_unclipped
        or end != end_unclipped
    )

    return start, end, requested, clipped


def _extract_window(
    boundary: BoundaryRegion,
    raw_signal: np.ndarray,
    normalized_signal: np.ndarray,
    sample_rate_hz: float,
    window_ms: int,
    force_reference: float,
    low_force_threshold: float,
) -> BoundaryWindowFeatures:
    '''Extract force and motion diagnostics for one boundary/window size.'''
    start, end, requested, clipped = _window_bounds(
        boundary.center_input_sample,
        window_ms,
        sample_rate_hz,
        raw_signal.shape[0],
    )
    raw_window = raw_signal[start:end]
    normalized_window = normalized_signal[start:end]
    force = raw_window[:, 12]

    center_index = int(round(boundary.center_input_sample))
    center_index = min(max(center_index, 0), raw_signal.shape[0] - 1)
    force_at_center = float(raw_signal[center_index, 12])
    force_min = float(force.min())
    low_force = force <= low_force_threshold

    af_magnitude = np.linalg.norm(normalized_window[:, 0:3], axis=1)
    ar_magnitude = np.linalg.norm(normalized_window[:, 3:6], axis=1)
    gyro_magnitude = np.linalg.norm(normalized_window[:, 6:9], axis=1)

    if len(normalized_window) > 1:
        derivatives = np.diff(normalized_window[:, 0:9], axis=0)
        derivative_energy = float(np.mean(derivatives**2))
    else:
        derivative_energy = 0.0

    return BoundaryWindowFeatures(
        window_ms=window_ms,
        requested_num_samples=requested,
        start_sample=start,
        end_sample=end,
        actual_num_samples=end - start,
        clipped_at_recording_edge=clipped,
        force_min_raw=force_min,
        force_mean_raw=float(force.mean()),
        force_at_center_raw=force_at_center,
        force_min_relative=force_min / force_reference,
        force_at_center_relative=force_at_center / force_reference,
        force_drop_ratio=float(
            np.clip(1 - force_min / force_reference, 0, 1)
        ),
        low_force_fraction=float(low_force.mean()),
        longest_low_force_ms=(
            _longest_true_run(low_force) / sample_rate_hz * 1000
        ),
        af_mean_magnitude=float(af_magnitude.mean()),
        ar_mean_magnitude=float(ar_magnitude.mean()),
        gyro_mean_magnitude=float(gyro_magnitude.mean()),
        motion_derivative_energy=derivative_energy,
    )


def extract_boundary_features(
    raw_signal: np.ndarray,
    normalized_signal: np.ndarray,
    alignment_analysis: AlignmentAnalysis,
    window_sizes_ms: tuple[int, ...] = (50, 100, 150),
    low_force_threshold_fraction: float = 0.10,
) -> BoundaryFeatureAnalysis:
    '''Extract transparent, unweighted features around boundary midpoints.

    No continuity score or acceptance threshold is applied here. The returned
    measurements are intended for development analysis before those choices
    are frozen using training-fold data only.
    '''
    if raw_signal.ndim != 2 or raw_signal.shape[1] != 13:
        raise ValueError('raw_signal must have shape (num_samples, 13).')
    if normalized_signal.ndim != 2 or normalized_signal.shape[1] != 13:
        raise ValueError(
            'normalized_signal must have shape (num_samples, 13).'
        )
    if len(normalized_signal) < len(raw_signal):
        raise ValueError(
            'normalized_signal cannot be shorter than the raw recording.'
        )
    if not window_sizes_ms or any(size <= 0 for size in window_sizes_ms):
        raise ValueError('window sizes must be positive.')
    if not 0 < low_force_threshold_fraction < 1:
        raise ValueError('low-force threshold fraction must be between 0 and 1.')

    normalized_signal = normalized_signal[: len(raw_signal)]
    force = raw_signal[:, 12]
    force_reference = max(float(np.percentile(force, 90)), 1e-6)
    low_force_threshold = (
        force_reference * low_force_threshold_fraction
    )

    boundaries = []
    for boundary in alignment_analysis.boundaries:
        left = alignment_analysis.characters[boundary.boundary_index]
        right = alignment_analysis.characters[boundary.boundary_index + 1]
        windows = [
            _extract_window(
                boundary,
                raw_signal,
                normalized_signal,
                alignment_analysis.sample_rate_hz,
                window_ms,
                force_reference,
                low_force_threshold,
            )
            for window_ms in window_sizes_ms
        ]

        boundaries.append(
            BoundaryFeatures(
                boundary_index=boundary.boundary_index,
                pair=boundary.pair,
                center_input_sample=boundary.center_input_sample,
                blank_frames=boundary.num_blank_frames,
                blank_duration_ms=boundary.duration_ms,
                left_aligned_probability=left.aligned_probability,
                right_aligned_probability=right.aligned_probability,
                minimum_aligned_probability=min(
                    left.aligned_probability,
                    right.aligned_probability,
                ),
                left_confidence_margin=left.confidence_margin,
                right_confidence_margin=right.confidence_margin,
                minimum_confidence_margin=min(
                    left.confidence_margin,
                    right.confidence_margin,
                ),
                both_anchors_agree_with_greedy=(
                    left.agrees_with_greedy and right.agrees_with_greedy
                ),
                overlaps_padding=(
                    left.overlaps_padding
                    or right.overlaps_padding
                    or left.input_end_sample > len(raw_signal)
                    or right.input_end_sample > len(raw_signal)
                ),
                windows=windows,
            )
        )

    return BoundaryFeatureAnalysis(
        force_reference_raw=force_reference,
        low_force_threshold_raw=low_force_threshold,
        low_force_threshold_fraction=low_force_threshold_fraction,
        window_sizes_ms=list(window_sizes_ms),
        boundaries=boundaries,
    )
