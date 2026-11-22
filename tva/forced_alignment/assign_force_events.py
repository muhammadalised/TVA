"""Run the approved whole-force-event assignment pilot on the frozen audit.

This exports assignment bookkeeping only. It does not fit pair scores.
"""
import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from tva.force_gap_assignment import (
    LetterGap, assign_force_events, detect_force_events,
)

DEFAULT_INPUT = Path("results/thesis/boundary_exports/a0_char_wi_rh_unidirectional/fold0/train_region_v2/boundaries.jsonl")
DEFAULT_CASES = Path("results/thesis/forced_alignment_review/wi_rh_fold0_20261116_trace_audit/frozen_selection.json")
DEFAULT_DATA = Path("data/tva/onhw_words500_wi_word_rh")
SHIFTS = (-8, 0, 8)
DELAYED = set("ijtf x".replace(" ", "") + "\u00e4\u00f6\u00fcIJTFX\u00c4\u00d6\u00dc")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            })


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def signature(gap):
    """Association includes status, event IDs and reasons, not near-edge flag."""
    return (
        gap["status"], tuple(gap["overlap_event_ids"]),
        tuple(gap["reasons"]), gap["candidate_event_id"],
    )


def make_plots(output, records, cases, raw_by_sample):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    (output / "plots").mkdir()
    by_sample = {r["sample_index"]: r for r in records}
    for page in range(25):
        fig, axes = plt.subplots(
            4, 2, figsize=(16, 12.8), dpi=140,
            gridspec_kw={"width_ratios": [1, 1.7]},
        )
        for row_index, case in enumerate(cases[page*4:(page+1)*4]):
            source = case["row"]
            record = by_sample[source["sample_index"]]
            baseline = record["variants"]["0"]
            gap = next(g for g in baseline["gaps"] if g["gap_id"] == source["boundary_id"])
            force = raw_by_sample[source["sample_index"]]
            times = np.arange(len(force)) / 100
            intersecting = [e for e in baseline["events"] if e["event_id"] in gap["overlap_event_ids"]]
            left = max(0, min([gap["start_sample"], *[e["start_sample"] for e in intersecting]]) - 30)
            right = min(len(force), max([gap["end_sample"], *[e["end_sample"] for e in intersecting]]) + 30)
            local = source["windows_ms"]["100"]
            left = min(left, local["start_sample"])
            right = max(right, local["end_sample"])
            for column, axis in enumerate(axes[row_index]):
                start, end = (0, len(force)) if column == 0 else (left, right)
                axis.plot(times[start:end], force[start:end], color="#613c92", lw=1,
                          marker="." if column else None, markersize=2.5)
                axis.axhline(record["threshold"], color="#ac3035", linestyle="--", lw=1)
                for event in baseline["events"]:
                    if event["end_sample"] <= start or event["start_sample"] >= end:
                        continue
                    axis.axvspan(event["start_sample"]/100, event["end_sample"]/100,
                                 color="#ba3b46", alpha=.12)
                    if column:
                        axis.text((event["start_sample"]+event["end_sample"])/200, .72,
                                  "E"+event["event_id"].rsplit(":", 1)[1],
                                  transform=axis.get_xaxis_transform(), ha="center",
                                  fontsize=7, color="#9b2027", clip_on=True)
                for neighbour in baseline["gaps"]:
                    a, b = neighbour["start_sample"], neighbour["end_sample"]
                    if b < start or a > end:
                        continue
                    selected = neighbour["gap_id"] == gap["gap_id"]
                    color = "#bb7600" if selected else "#777777"
                    axis.axvline(a/100, color=color, linestyle=":", lw=1.5 if selected else .6)
                    axis.axvline(b/100, color=color, linestyle=":", lw=1.5 if selected else .6)
                    axis.axvspan(a/100, b/100, ymin=.90, ymax=.98, color=color, alpha=.35)
                    if column:
                        index = neighbour["gap_id"].rsplit(":", 1)[1]
                        axis.text((a+b)/200, .99, "G"+index, fontsize=7,
                                  transform=axis.get_xaxis_transform(), ha="center", va="top", clip_on=True)
                if gap["end_sample"] > gap["start_sample"]:
                    axis.axvspan(gap["start_sample"]/100, gap["end_sample"]/100,
                                 color="#edb64a", alpha=.12)
                axis.axvspan(local["start_sample"]/100, local["end_sample"]/100,
                             color="#1f77b4", alpha=.12)
                axis.set_xlim(times[start], max(times[end-1], times[start]+.01))
                axis.set_ylabel("Raw tip force", fontsize=9)
                axis.set_xlabel("Time (s)", fontsize=9)
                axis.tick_params(labelsize=8)
                axis.grid(alpha=.2)
            axes[row_index, 0].set_title(
                f'{case["case_id"]}  {source["known_label"]}  [{source["pair"]}]  {gap["status"]}',
                fontsize=10, loc="left")
            minus = next(g for g in record["variants"]["-80"]["gaps"] if g["gap_id"] == gap["gap_id"])
            plus = next(g for g in record["variants"]["80"]["gaps"] if g["gap_id"] == gap["gap_id"])
            axes[row_index, 1].set_title(
                f'-80 ms: {minus["status"]} | +80 ms: {plus["status"]}',
                fontsize=9, loc="left")
        fig.suptitle(f"Whole-force-event assignment pilot - sheet {page+1:02d}/25", fontsize=13)
        fig.text(.5, .005,
                 "Red shading/E IDs: complete recorded force drops. Orange: selected original CTC gap; blue: old local window.\n"
                 "Top bars/G IDs: all neighbouring gaps, including unreliable ones. A candidate is a time-based link, not physical ground truth.",
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .04, 1, .97), h_pad=2)
        fig.savefig(output / "plots" / f"sheet_{page+1:02d}.png")
        plt.close(fig)
        if (page+1) % 5 == 0:
            print("Rendered", page+1, "/ 25 sheets", flush=True)


def execute(args, output, manifest):
    cases = json.loads(args.cases.read_text())
    freeze = json.loads((args.cases.parent / "freeze_manifest.json").read_text())
    assert sha(args.cases) == freeze["frozen_selection_sha256"]
    assert sha(args.input) == freeze["source_sha256"][str(DEFAULT_INPUT)]
    assert len(cases) == len({c["row"]["sample_index"] for c in cases}) == 100
    selected = {c["row"]["sample_index"]: c for c in cases}
    source_gaps = defaultdict(list)
    with args.input.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["sample_index"] in selected:
                assert row["split"] == "train" and row["fold"] == 0
                source_gaps[row["sample_index"]].append(row)
    assert set(source_gaps) == set(selected)
    annotations = json.loads((args.dataset_dir / "train.json").read_text())["annotations"]["0"]
    trace_manifest = json.loads((args.cases.parent / "render_manifest.json").read_text())
    feature_index = {
        r["case_id"]: r for r in json.loads((args.cases.parent / "case_features.json").read_text())
    }
    records, event_rows, gap_rows, overlap_rows, case_rows = [], [], [], [], []
    raw_hashes, raw_by_sample = {}, {}
    for sample_index, case in selected.items():
        rows = sorted(source_gaps[sample_index], key=lambda r: r["boundary_index"])
        source = case["row"]
        annotation = annotations[sample_index]
        assert len(rows) == len(source["known_label"])-1
        assert [r["boundary_index"] for r in rows] == list(range(len(rows)))
        assert len({r["boundary_id"] for r in rows}) == len(rows)
        assert next(r for r in rows if r["boundary_id"] == source["boundary_id"]) == source
        raw_path = args.dataset_dir / source["source_file"]
        raw_hashes[str(raw_path)] = sha(raw_path)
        expected_hash = trace_manifest["raw_source_sha256"][str(DEFAULT_DATA / source["source_file"])]
        assert raw_hashes[str(raw_path)] == expected_hash
        signal = np.loadtxt(raw_path, delimiter=";", dtype=np.float32)
        assert signal.ndim == 2 and signal.shape[1] == 13 and np.isfinite(signal).all()
        assert len(signal) == source["raw_num_samples"]
        assert annotation["filename"] == source["source_file"] and annotation["label"] == source["known_label"]
        force = signal[:, 12]
        threshold = source["low_force_threshold_raw"]
        reference = max(float(np.percentile(force, 90)), 1e-6)
        assert np.isclose(reference, source["force_reference_raw"], rtol=1e-6)
        assert np.isclose(threshold, .1 * source["force_reference_raw"], rtol=1e-12)
        recording_id = f"0:train:{sample_index}"
        events = detect_force_events(force, threshold, 100, recording_id)
        gaps = []
        by_id = {}
        for row in rows:
            assert row["sample_index"] == sample_index and row["known_label"] == source["known_label"]
            assert row["writer_id"] == source["writer_id"] == annotation["id_writer"]
            assert row["sample_rate_hz"] == 100 and row["downsampling_ratio"] == 8
            assert row["checkpoint_epoch"] == 275 and row["low_force_threshold_raw"] == threshold
            assert row["force_reference_raw"] == source["force_reference_raw"]
            assert row["pair"] == source["known_label"][row["boundary_index"]:row["boundary_index"]+2]
            region = row["candidate_region"]
            gaps.append(LetterGap(
                row["boundary_id"], region["candidate_start_sample"], region["candidate_end_sample"],
                reliable=bool(row["both_anchors_agree_with_greedy"] and not region["clipped_at_recording_edge"]),
                padded=bool(row["overlaps_padding"]),
            ))
            by_id[row["boundary_id"]] = row
        variants = {
            str(shift*10): assign_force_events(events, gaps, len(force), shift_samples=shift)
            for shift in SHIFTS
        }
        metadata = {
            "sample_index": sample_index, "case_id": case["case_id"],
            "sampling_group": case["sampling_group"], "word": source["known_label"],
            "writer_id": source["writer_id"],
        }
        for shift, result in variants.items():
            assert len(result["events"]) == len(events)
            for event, computed in zip(events, result["events"]):
                assert all(computed[key] == value for key, value in asdict(event).items())
                event_rows.append({**metadata, "shift_ms": int(shift), **computed})
            for gap in result["gaps"]:
                original = by_id[gap["gap_id"]]
                gap_rows.append({
                    **metadata, "shift_ms": int(shift), "audited_gap": gap["gap_id"] == source["boundary_id"],
                    "pair": original["pair"], "left_character": original["left_character"],
                    "right_character": original["right_character"],
                    "possible_delayed_left": original["left_character"] in DELAYED,
                    "possible_delayed_either_neighbour": any(ch in DELAYED for ch in original["pair"]),
                    **gap,
                })
            overlap_rows.extend({**metadata, "shift_ms": int(shift), **edge} for edge in result["overlaps"])
        target = {
            shift: next(g for g in result["gaps"] if g["gap_id"] == source["boundary_id"])
            for shift, result in variants.items()
        }
        baseline = target["0"]
        old = feature_index[case["case_id"]]
        case_rows.append({
            **metadata, "gap_id": source["boundary_id"], "pair": source["pair"],
            "pair_rank": case["pair_rank"], "pair_score_unchanged": case["pair_score"],
            "old_local_evidence": old["local"]["category"], "old_region_evidence": old["region"]["category"],
            "old_local_longest_low_ms": old["local"]["longest_low_ms"],
            "old_region_longest_low_ms": old["region"]["longest_low_ms"],
            "source_fallback": old["fallback"], "possible_delayed_left": old["possible_delayed_left"],
            "possible_delayed_either_neighbour": any(ch in DELAYED for ch in source["pair"]),
            "baseline_status": baseline["status"], "baseline_reasons": baseline["reasons"],
            "baseline_event_ids": baseline["overlap_event_ids"],
            "candidate_event_id": baseline["candidate_event_id"], "baseline_near_edge_low": baseline["near_edge_low"],
            "minus80_status": target["-80"]["status"], "minus80_reasons": target["-80"]["reasons"],
            "minus80_event_ids": target["-80"]["overlap_event_ids"],
            "plus80_status": target["80"]["status"], "plus80_reasons": target["80"]["reasons"],
            "plus80_event_ids": target["80"]["overlap_event_ids"],
            "minus80_association_changed": signature(baseline) != signature(target["-80"]),
            "plus80_association_changed": signature(baseline) != signature(target["80"]),
            "either_association_changed": any(signature(baseline) != signature(target[t]) for t in ["-80", "80"]),
            "either_status_changed": any(baseline["status"] != target[t]["status"] for t in ["-80", "80"]),
            "old_local_missed_region_drop": old["local_region_contact_disagreement"] and old["local"]["category"] == "no_low",
            "old_region_multiple_low_runs": old["region"]["low_runs"] >= 2,
            "physical_assignment": "unverified",
        })
        records.append({
            **metadata, "source_file": str(raw_path), "raw_sha256": raw_hashes[str(raw_path)],
            "threshold": threshold, "force_reference": reference, "num_samples": len(force),
            "audited_gap_id": source["boundary_id"], "variants": variants,
        })
        raw_by_sample[sample_index] = force
    assert len(case_rows) == 100
    write_csv(output / "case_assignments.csv", case_rows)
    write_csv(output / "event_assignments.csv", event_rows)
    write_csv(output / "gap_assignments.csv", gap_rows)
    write_csv(output / "event_gap_overlaps.csv", overlap_rows)
    with (output / "recording_assignments.jsonl").open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    groups = []
    coverage = []
    for group in ["high", "middle", "low", "targeted"]:
        subset = [r for r in case_rows if r["sampling_group"] == group]
        reasons = Counter(reason for row in subset for reason in row["baseline_reasons"])
        groups.append({
            "group": group, "cases": len(subset),
            **{status: sum(r["baseline_status"] == status for r in subset)
               for status in ["single_candidate", "no_observed_drop", "ambiguous", "unavailable"]},
            "timing_sensitive_association": sum(r["either_association_changed"] for r in subset),
            "timing_sensitive_status": sum(r["either_status_changed"] for r in subset),
            "baseline_candidate_changed_by_shift": sum(r["baseline_status"] == "single_candidate" and r["either_association_changed"] for r in subset),
            "baseline_no_drop_changed_by_shift": sum(r["baseline_status"] == "no_observed_drop" and r["either_association_changed"] for r in subset),
            "reason_counts": dict(reasons),
        })
    for kind in ["pair", "word", "writer_id"]:
        for value in sorted({str(r[kind]) for r in case_rows}):
            subset = [r for r in case_rows if str(r[kind]) == value]
            coverage.append({
                "kind": kind, "value": value, "cases": len(subset),
                **{status: sum(r["baseline_status"] == status for r in subset)
                   for status in ["single_candidate", "no_observed_drop", "ambiguous", "unavailable"]},
                "timing_sensitive_association": sum(r["either_association_changed"] for r in subset),
            })
    write_csv(output / "group_summary.csv", groups)
    write_csv(output / "coverage_by_pair_word_writer.csv", coverage)
    delayed_groups = []
    for flag in ["possible_delayed_left", "possible_delayed_either_neighbour"]:
        for value in [False, True]:
            subset = [r for r in case_rows if r[flag] == value]
            delayed_groups.append({
                "flag": flag, "value": value, "cases": len(subset),
                "statuses": dict(Counter(r["baseline_status"] for r in subset)),
                "timing_sensitive_association": sum(r["either_association_changed"] for r in subset),
            })
    base_events = [r for r in event_rows if r["shift_ms"] == 0]
    base_gaps = [r for r in gap_rows if r["shift_ms"] == 0]
    summary = {
        "status": "completed_assignment_diagnostic", "audited_cases": len(case_rows),
        "recordings": len(records), "all_neighbouring_gaps": len(base_gaps),
        "unavailable_neighbouring_gaps": sum(r["status"] == "unavailable" for r in base_gaps),
        "groups": groups, "possible_delayed_groups": delayed_groups,
        "baseline_all_recording_event_categories": dict(Counter(r["status"] for r in base_events)),
        "baseline_all_recording_event_count": len(base_events),
        "baseline_all_gap_statuses": dict(Counter(r["status"] for r in base_gaps)),
        "timing_signature": "status, overlapping event IDs, uncertainty reasons, candidate event ID; near-edge flag excluded",
        "case_statuses": dict(Counter(r["baseline_status"] for r in case_rows)),
        "cases_timing_sensitive": sum(r["either_association_changed"] for r in case_rows),
        "cases_status_sensitive": sum(r["either_status_changed"] for r in case_rows),
        "candidate_cases": [r["case_id"] for r in case_rows if r["baseline_status"] == "single_candidate"],
        "stable_candidate_cases": [r["case_id"] for r in case_rows if r["baseline_status"] == "single_candidate" and not r["either_association_changed"]],
        "no_drop_cases": [r["case_id"] for r in case_rows if r["baseline_status"] == "no_observed_drop"],
        "stable_no_drop_cases": [r["case_id"] for r in case_rows if r["baseline_status"] == "no_observed_drop" and not r["either_association_changed"]],
        "score_changed": False, "physical_accuracy_established": False,
        "minimum_duration_filter": None,
        "interpretation": "Same development cases as prior review; candidate links are temporal bookkeeping, not independently validated physical joins.",
    }
    write_json(output / "assignment_summary.json", summary)
    make_plots(output, records, cases, raw_by_sample)
    for name, digest in raw_hashes.items():
        assert sha(name) == digest
    for name, digest in manifest["input_sha256"].items():
        assert sha(name) == digest, name
    manifest.update(
        status="completed", finished_at_utc=datetime.now(timezone.utc).isoformat(),
        raw_source_sha256=raw_hashes,
        result_sha256={str(p.relative_to(output)): sha(p) for p in output.rglob("*")
                       if p.is_file() and p.name != "run_manifest.json"},
    )
    write_json(output / "run_manifest.json", manifest)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    old = json.loads((args.cases.parent / "audit_manifest.json").read_text())
    inputs = dict(old["input_sha256"])
    for name, digest in inputs.items():
        assert sha(name) == digest, name
    for path in [args.input, args.cases, args.cases.parent / "freeze_manifest.json",
                 args.cases.parent / "render_manifest.json", args.cases.parent / "audit_summary.json",
                 args.cases.parent / "case_features.json",
                 Path("docs/FORCED_ALIGNMENT_TRACE_AUDIT_20261116.csv"),
                 args.dataset_dir / "train.json"]:
        inputs[str(path)] = sha(path)
    sources = [Path(__file__), Path("tva/force_gap_assignment.py"), Path("tests/test_force_gap_assignment.py")]
    (output / "source").mkdir()
    for source in sources:
        shutil.copyfile(source, output / "source" / source.name)
    manifest = {
        "status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "user_date": "2026-11-16", "code_revision": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"), "git_status": git("status", "--porcelain=v1", "--untracked-files=all"),
        "python": sys.executable, "python_version": sys.version,
        "packages": {name: importlib.metadata.version(name) for name in ["numpy", "matplotlib"]},
        "command": [sys.executable, *sys.orig_argv[1:]], "input_sha256": inputs,
        "source_sha256": {str(p): sha(p) for p in sources},
        "procedure": {
            "force_threshold": "stored original 0.10 x recording 90th percentile, independently reconciled",
            "recordings": 100, "gap_quality": "both anchors agree, source candidate region unclipped, original padding absent, shifted raw bounds valid",
            "shifts_ms": [-80, 0, 80], "near_edge_ms": 80, "minimum_duration_ms": None,
            "candidate": "one whole non-truncated event strictly contained in one reliable gap; mutually unique overlap",
            "score_change": False, "realignment": False, "model_inference": False,
            "random_sampling": False, "physical_truth": False,
            "gap_reason_flags": "nonexclusive; unavailable takes precedence, then ambiguity",
            "truncated_events": "remain unassigned even when overlapping gaps",
        },
    }
    (output / "working_tree_before_run.patch").write_text(subprocess.check_output(["git", "diff", "--binary"], text=True))
    (output / "protocol_before_run.md").write_text(Path("docs/FORCED_ALIGNMENT.md").read_text())
    write_json(output / "run_manifest.json", manifest)
    try:
        execute(args, output, manifest)
    except Exception as error:
        manifest.update(status="failed", error=repr(error), failed_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(output / "run_manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
