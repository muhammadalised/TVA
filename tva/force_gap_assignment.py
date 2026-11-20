"""Whole-recording force events and conservative links to CTC candidate gaps.

This module reports time-based candidates and uncertainty. It does not produce
physical pen-lift labels, pair scores, or tokenizer inputs.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ForceEvent:
    event_id: str
    start_sample: int
    end_sample: int
    duration_ms: float
    left_truncated: bool = False
    right_truncated: bool = False


@dataclass(frozen=True)
class LetterGap:
    gap_id: str
    start_sample: int
    end_sample: int
    reliable: bool = True
    padded: bool = False


def detect_force_events(
    force: np.ndarray,
    threshold: float,
    sample_rate_hz: float,
    recording_id: str,
) -> list[ForceEvent]:
    """Keep every complete observed low-force run, including 1-sample runs."""
    force = np.asarray(force)
    if force.ndim != 1 or len(force) == 0:
        raise ValueError("force must be a nonempty one-dimensional array")
    if not np.isfinite(force).all() or not math.isfinite(threshold):
        raise ValueError("force and threshold must be finite")
    if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be finite and positive")
    low = force <= threshold
    edges = np.diff(np.r_[False, low, False].astype(np.int8))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return [
        ForceEvent(
            event_id=f"{recording_id}:event:{i}",
            start_sample=int(start),
            end_sample=int(end),
            duration_ms=float((end - start) / sample_rate_hz * 1000),
            left_truncated=bool(start == 0),
            right_truncated=bool(end == len(force)),
        )
        for i, (start, end) in enumerate(zip(starts, ends))
    ]


def assign_force_events(
    events: list[ForceEvent],
    gaps: list[LetterGap],
    num_samples: int,
    *,
    shift_samples: int = 0,
    near_edge_samples: int = 8,
) -> dict:
    """Link only mutually unique, strictly contained, non-truncated events.

    All original gaps participate in overlap bookkeeping, even when unreliable
    or outside the raw recording. Empty intervals have no positive overlap.
    Near-edge flags never create an assignment. Every event receives exactly
    one ledger category: single_candidate, ambiguous, or unassigned.
    """
    if num_samples <= 0 or int(num_samples) != num_samples:
        raise ValueError("num_samples must be a positive integer")
    if int(shift_samples) != shift_samples:
        raise ValueError("shift_samples must be an integer")
    if near_edge_samples < 0 or int(near_edge_samples) != near_edge_samples:
        raise ValueError("near_edge_samples must be a nonnegative integer")
    if len({e.event_id for e in events}) != len(events):
        raise ValueError("duplicate event IDs")
    if len({g.gap_id for g in gaps}) != len(gaps):
        raise ValueError("duplicate gap IDs")
    ordered = sorted(events, key=lambda e: (e.start_sample, e.end_sample))
    for i, event in enumerate(ordered):
        if not 0 <= event.start_sample < event.end_sample <= num_samples:
            raise ValueError("event interval is outside the recording or empty")
        if not math.isfinite(event.duration_ms) or event.duration_ms <= 0:
            raise ValueError("event duration must be finite and positive")
        if event.left_truncated != (event.start_sample == 0):
            raise ValueError("incorrect left truncation flag")
        if event.right_truncated != (event.end_sample == num_samples):
            raise ValueError("incorrect right truncation flag")
        if i and ordered[i - 1].end_sample >= event.start_sample:
            raise ValueError("events must be separate maximal low-force runs")
    for gap in gaps:
        if gap.end_sample < gap.start_sample:
            raise ValueError("gap end precedes its start")

    shifted = {
        gap.gap_id: (
            gap.start_sample + int(shift_samples),
            gap.end_sample + int(shift_samples),
        )
        for gap in gaps
    }
    gap_by_id = {g.gap_id: g for g in gaps}
    event_by_id = {e.event_id: e for e in events}
    gap_reasons = {}
    for gap in gaps:
        start, end = shifted[gap.gap_id]
        reasons = []
        if start == end:
            reasons.append("empty_gap")
        if not gap.reliable:
            reasons.append("unreliable_gap")
        if gap.padded:
            reasons.append("source_padding")
        if start < 0 or end > num_samples:
            reasons.append("outside_recording")
        gap_reasons[gap.gap_id] = reasons

    by_gap = {g.gap_id: [] for g in gaps}
    by_event = {e.event_id: [] for e in events}
    overlaps = []
    for event in events:
        for gap in gaps:
            start, end = shifted[gap.gap_id]
            overlap_start = max(event.start_sample, start)
            overlap_end = min(event.end_sample, end)
            if overlap_start < overlap_end:
                by_gap[gap.gap_id].append(event.event_id)
                by_event[event.event_id].append(gap.gap_id)
                overlaps.append({
                    "event_id": event.event_id,
                    "gap_id": gap.gap_id,
                    "overlap_start_sample": overlap_start,
                    "overlap_end_sample": overlap_end,
                    "overlap_samples": overlap_end - overlap_start,
                })

    gap_rows = []
    clean_event_to_gap = {}
    for gap in gaps:
        start, end = shifted[gap.gap_id]
        unavailable = gap_reasons[gap.gap_id]
        reasons = list(unavailable)
        event_ids = by_gap[gap.gap_id]
        if len(event_ids) > 1:
            reasons.append("multiple_events")
        for event_id in event_ids:
            event = event_by_id[event_id]
            if event.left_truncated or event.right_truncated:
                reasons.append("truncated_event")
            if not (start < event.start_sample and event.end_sample < end):
                reasons.append("edge_crossing_or_touching")
            other_gaps = by_event[event_id]
            if len(other_gaps) > 1:
                reasons.append("shared_event")
                if any(gap_reasons[g] for g in other_gaps if g != gap.gap_id):
                    reasons.append("unavailable_competing_gap")
        reasons = sorted(set(reasons))
        if unavailable:
            status = "unavailable"
        elif not event_ids:
            status = "no_observed_drop"
        elif reasons:
            status = "ambiguous"
        else:
            status = "single_candidate"
            event_id = event_ids[0]
            if event_id in clean_event_to_gap:
                raise AssertionError("an event received two clean assignments")
            clean_event_to_gap[event_id] = gap.gap_id
        near = any(
            event.start_sample - near_edge_samples <= edge
            <= event.end_sample - 1 + near_edge_samples
            for event in events
            for edge in (start, end)
        )
        gap_rows.append({
            "gap_id": gap.gap_id,
            "original_start_sample": gap.start_sample,
            "original_end_sample": gap.end_sample,
            "start_sample": start,
            "end_sample": end,
            "reliable": gap.reliable,
            "padded": gap.padded,
            "status": status,
            "reasons": reasons,
            "overlap_event_ids": list(event_ids),
            "candidate_event_id": event_ids[0] if status == "single_candidate" else None,
            "near_edge_low": bool(near),
        })

    event_rows = []
    for event in events:
        gap_ids = by_event[event.event_id]
        truncated = event.left_truncated or event.right_truncated
        if truncated:
            status, reasons = "unassigned", ["recording_truncated"]
        elif event.event_id in clean_event_to_gap:
            status, reasons = "single_candidate", []
        elif not gap_ids:
            status, reasons = "unassigned", ["no_positive_gap_overlap"]
        else:
            status = "ambiguous"
            reasons = sorted({
                reason
                for row in gap_rows if row["gap_id"] in gap_ids
                for reason in row["reasons"]
            })
        event_rows.append({
            **asdict(event),
            "status": status,
            "reasons": reasons,
            "overlap_gap_ids": list(gap_ids),
            "candidate_gap_id": clean_event_to_gap.get(event.event_id),
        })
    assert len(event_rows) == len(events)
    assert len(clean_event_to_gap) == sum(
        row["status"] == "single_candidate" for row in gap_rows
    )
    return {
        "shift_samples": int(shift_samples),
        "near_edge_samples": int(near_edge_samples),
        "num_samples": int(num_samples),
        "events": event_rows,
        "gaps": gap_rows,
        "overlaps": overlaps,
    }
