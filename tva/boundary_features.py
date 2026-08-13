from dataclasses import dataclass

import numpy as np

from .alignment_analysis import AlignmentAnalysis, BoundaryRegion

__all__ = [
    'BoundaryFeatureAnalysis',
    'BoundaryFeatures',
    'BoundaryRegionFeatures',
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
class BoundaryRegionFeatures:
    '''Sensor features from the complete CTC candidate boundary region.

    A CTC alignment can place two character emissions in consecutive frames,
    leaving an empty candidate region. In that case, ``start_sample`` and
    ``end_sample`` describe a small midpoint-centred fallback window, while
    ``candidate_start_sample`` and ``candidate_end_sample`` preserve the empty
    original interval.
    '''

    candidate_start_sample: int
    candidate_end_sample: int
    candidate_num_samples: int
    candidate_duration_ms: float
    start_sample: int
    end_sample: int
    actual_num_samples: int
    actual_duration_ms: float
    used_fallback_window: bool
    fallback_window_ms: int
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
    candidate_region: BoundaryRegionFeatures
    windows: list[BoundaryWindowFeatures]


@dataclass(frozen=True)
class BoundaryFeatureAnalysis:
    '''Recording-level settings and extracted features for all boundaries.'''

    force_reference_raw: float
    low_force_threshold_raw: float
    low_force_threshold_fraction: float
    empty_region_fallback_ms: int
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


def _candidate_region_bounds(
    boundary: BoundaryRegion,
    sample_rate_hz: float,
    num_samples: int,
    fallback_window_ms: int,
) -> tuple[int, int, bool, bool]:
    '''Return valid feature bounds for a complete CTC candidate region.'''
    candidate_start = boundary.input_start_sample
    candidate_end = boundary.input_end_sample
    used_fallback = candidate_end <= candidate_start

    if used_fallback:
        start, end, _, clipped = _window_bounds(
            boundary.center_input_sample,
            fallback_window_ms,
            sample_rate_hz,
            num_samples,
        )
        return start, end, True, clipped

    start = min(max(candidate_start, 0), num_samples)
    end = min(max(candidate_end, 0), num_samples)
    clipped = start != candidate_start or end != candidate_end
    return start, end, False, clipped


def _extract_candidate_region(
    boundary: BoundaryRegion,
    raw_signal: np.ndarray,
    normalized_signal: np.ndarray,
    sample_rate_hz: float,
    fallback_window_ms: int,
    force_reference: float,
    low_force_threshold: float,
) -> BoundaryRegionFeatures:
    '''Extract measurements across the complete inter-character region.'''
    start, end, used_fallback, clipped = _candidate_region_bounds(
        boundary,
        sample_rate_hz,
        raw_signal.shape[0],
        fallback_window_ms,
    )
    raw_region = raw_signal[start:end]
    normalized_region = normalized_signal[start:end]

    # The recording is non-empty and the fallback window always has at least
    # one sample, so a clipped region can only be empty if it lies wholly in
    # model padding. Use the last real sample for transparent diagnostics and
    # mark the region as clipped so it can be excluded during analysis.
    if len(raw_region) == 0:
        start = raw_signal.shape[0] - 1
        end = raw_signal.shape[0]
        raw_region = raw_signal[start:end]
        normalized_region = normalized_signal[start:end]
        clipped = True

    force = raw_region[:, 12]
    center_index = int(round(boundary.center_input_sample))
    center_index = min(max(center_index, 0), raw_signal.shape[0] - 1)
    force_at_center = float(raw_signal[center_index, 12])
    force_min = float(force.min())
    low_force = force <= low_force_threshold

    af_magnitude = np.linalg.norm(normalized_region[:, 0:3], axis=1)
    ar_magnitude = np.linalg.norm(normalized_region[:, 3:6], axis=1)
    gyro_magnitude = np.linalg.norm(normalized_region[:, 6:9], axis=1)

    if len(normalized_region) > 1:
        derivatives = np.diff(normalized_region[:, 0:9], axis=0)
        derivative_energy = float(np.mean(derivatives**2))
    else:
        derivative_energy = 0.0

    candidate_num_samples = max(
        0,
        boundary.input_end_sample - boundary.input_start_sample,
    )
    return BoundaryRegionFeatures(
        candidate_start_sample=boundary.input_start_sample,
        candidate_end_sample=boundary.input_end_sample,
        candidate_num_samples=candidate_num_samples,
        candidate_duration_ms=(
            candidate_num_samples / sample_rate_hz * 1000
        ),
        start_sample=start,
        end_sample=end,
        actual_num_samples=end - start,
        actual_duration_ms=(end - start) / sample_rate_hz * 1000,
        used_fallback_window=used_fallback,
        fallback_window_ms=fallback_window_ms,
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
    empty_region_fallback_ms: int = 100,
    low_force_threshold_fraction: float = 0.10,
) -> BoundaryFeatureAnalysis:
    '''Extract transparent, unweighted features around boundary midpoints.

    No continuity score or acceptance threshold is applied here. The returned
    measurements are intended for development analysis before those choices
    are frozen using training-fold data only.
    '''
    if raw_signal.ndim != 2 or raw_signal.shape[1] != 13:
        raise ValueError('raw_signal must have shape (num_samples, 13).')
    if len(raw_signal) == 0:
        raise ValueError('raw_signal must contain at least one sample.')
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
    if empty_region_fallback_ms <= 0:
        raise ValueError('empty-region fallback size must be positive.')
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
        candidate_region = _extract_candidate_region(
            boundary,
            raw_signal,
            normalized_signal,
            alignment_analysis.sample_rate_hz,
            empty_region_fallback_ms,
            force_reference,
            low_force_threshold,
        )

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
                candidate_region=candidate_region,
                windows=windows,
            )
        )

    return BoundaryFeatureAnalysis(
        force_reference_raw=force_reference,
        low_force_threshold_raw=low_force_threshold,
        low_force_threshold_fraction=low_force_threshold_fraction,
        empty_region_fallback_ms=empty_region_fallback_ms,
        window_sizes_ms=list(window_sizes_ms),
        boundaries=boundaries,
    )
