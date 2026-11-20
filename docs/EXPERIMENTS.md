# Thesis Experiment Log

This file records scientific runs and important development experiments. All
reported values must be traceable to the saved configuration, log, metrics,
predictions, code version, and experiment backup.

## B0 character baseline — WD/RH fold 0

### Setup

- Date completed: 2026-08-02
- Dataset: OnHW-Words500, writer-dependent, right-handed, fold 0
- Method: raw 13-channel IMU to character-level CTC labels
- Architecture: BLConv-B + BiLSTM-B
- Seed: 42
- Epochs: 300 (epochs 0–299)
- Batch size: 64
- Training machine: WSL laptop with NVIDIA GeForce RTX 4060
- Configuration: `configs/thesis/b0_char_wd_rh.yaml`

### Best validation results

| Metric | Value | Epoch |
| --- | ---: | ---: |
| Levenshtein distance | 0.722597 | 286 |
| Character error rate | 0.127563 (12.76%) | 286 |
| Word error rate | 0.359809 (35.98%) | 289 |
| Average reference length | 5.664615 characters | Not optimized |

The original paper's WD character result is aggregated across five folds, so
it must not be directly compared with this single-fold value.

### Artifacts and caveats

- The complete run directory was archived and backed up by the researcher.
- `latest.pth` and the epoch-299 milestone checkpoint were retained.
- This run preceded automatic best-CER/best-WER checkpoint saving. Metrics and
  predictions for epochs 286 and 289 are preserved, but their exact model
  weights are not. The epoch-299 model can be used for initial forced-alignment
  development.
- Backup archive checksum: not yet recorded in this log.

## B0 character baseline — WI/RH fold 0

### Setup

- Date completed: 2026-08-02
- Dataset: OnHW-Words500, writer-independent, right-handed, fold 0
- Method: raw 13-channel IMU to character-level CTC labels
- Architecture: BLConv-B + BiLSTM-B
- Seed: 42
- Epochs: 300 (epochs 0–299)
- Batch size: 64
- Training machine: WSL laptop with NVIDIA GeForce RTX 4060
- Configuration: `configs/thesis/b0_char_wi_rh.yaml`

### Best validation results

| Metric | Value | Epoch |
| --- | ---: | ---: |
| Levenshtein distance | 0.851474 | 244 |
| Character error rate | 0.156041 (15.60%) | 244 |
| Word error rate | 0.279478 (27.95%) | 269 |
| Average reference length | 5.456727 characters | Not optimized |

These are single-fold development results. They must not be presented as the
final writer-independent result, which will require the agreed five-fold
evaluation.

### Artifacts and use

- The run used automatic validation-selected checkpoint saving.
- `best_cer.pth` from epoch 244 is the preferred checkpoint for WI forced
  alignment because alignment quality depends most directly on character
  recognition.
- `best_wer.pth` from epoch 269 is retained for recognition comparison, while
  `latest.pth` remains the resumable end-of-training checkpoint.
- The complete run directory should be archived before alignment development.
- Backup archive checksum: not yet recorded in this log.

## A0 unidirectional character aligner — WI/RH fold 0

### Setup

- Date prepared: 2026-08-09
- Date completed: 2026-08-09
- Dataset: OnHW-Words500, writer-independent, right-handed, fold 0
- Purpose: alignment-only temporal-localization comparison
- Architecture: BLConv-B + UniLSTM-B
- Controlled change from B0: recurrent decoder direction only
- Seed: 42
- Maximum epochs: 300 (validation-selected checkpoint)
- Batch size: 64
- Configuration: `configs/thesis/a0_char_wi_rh_unidirectional.yaml`
- Output: `results/thesis/alignment_models/A0_char_wi_rh_unidirectional`

### Best validation results

| Metric | Value | Epoch |
| --- | ---: | ---: |
| Levenshtein distance | 0.945578 | 275 |
| Character error rate | 0.173287 (17.33%) | 275 |
| Word error rate | 0.327664 (32.77%) | 293 |
| Average reference length | 5.456727 characters | Not optimized |

The recognition results are worse than B0 WI (15.60% CER and 27.95% WER), as
expected when the recurrent decoder cannot use future frames. A0 is not the
recognition baseline.

### Alignment acceptance result

Use `best_cer.pth` to repeat the sample alignments and full training boundary
analysis. Compare reliable `only`, `first`, `middle`, and `final` timing offsets,
blank durations, anchor agreement, and sample support with B0. The main question
is whether A0 removes the 80 ms word-initial concentration; recognition metrics
are secondary because A0 does not replace the B0 recognition baseline.

A0 is not fully causal because BLConv retains centred convolutions and
sequence-wide instance normalization. This experiment isolates the effect of
the recurrent decoder direction before considering a larger encoder change.

The complete fold-0 training export contained 88,292 boundaries from 19,907
samples and 42 writers. The transparent agreement/non-padding/unclipped subset
retained 78,953 boundaries (89.4%). Reliable median timing changed as follows:

| Position | B0 offset | A0 offset |
| --- | ---: | ---: |
| Only boundary | -315 ms | -65 ms |
| First boundary | -300 ms | -47 ms |
| Middle boundary | -98 ms | +63 ms |
| Final boundary | -50 ms | +38 ms |

The severe word-initial concentration therefore decreased substantially while
retaining broad sample and writer coverage. A0 is accepted as the alignment
model for fold-0 tokenizer-evidence development. The equal-spacing reference
remains a diagnostic rather than character-boundary ground truth.

## 2026-11-15 — A0 force baseline reproduction and word-support audit

### Purpose and scope

Reproduce the existing development ranking before testing its validity.
The user approved the staged plan and ongoing alignment-document updates.
This run changes no scientific setting. Use OnHW-Words500, right-handed,
WI fold 0, training boundaries only. No validation labels or recognition
scores are used for pair selection.

### Inputs and resolved settings

- Code revision: b7e4d5f4f542d8c73e967c13984cac339d349f02.
- Branch: forced-alignment. Only docs/FORCED_ALIGNMENT.md had tracked changes
  before the run; the existing untracked compatibility audit was preserved.
- Interpreter: /home/artellisys/miniconda3/envs/tva/bin/python, Python 3.11.15.
- PYTHONHASHSEED=42; no stochastic scoring operation or new training seed.
  The existing A0 configuration uses training seed 42.
- Checkpoint-selection rule: reuse the existing A0 best-CER export, epoch 275.
  No model was loaded or trained during scoring.
- Input: results/thesis/boundary_exports/a0_char_wi_rh_unidirectional/fold0/train_region_v2/boundaries.jsonl.
- Baseline: 100 ms local window, 0.75 local / 0.25 complete-region weights,
  at least 100 accepted occurrences and 30 writers per eligible pair,
  at least 100 occurrences per position-duration correction stratum.
  The export force threshold remains 0.10 of the recording reference.
- Output:
  [wi_rh_fold0_20261115_baseline/](../results/thesis/forced_alignment_review/wi_rh_fold0_20261115_baseline/).
  Local results are Git-ignored; preserve this directory.
- Full command and provenance:
  [run_manifest.json](../results/thesis/forced_alignment_review/wi_rh_fold0_20261115_baseline/run_manifest.json).
  It includes the resolved parameters, package versions, current hashes and
  pre-run working-tree state. The pre-run patch is saved beside it.

### Observed results

- Both continuity_score_summary.json and pair_continuity_scores.csv match the
  August outputs byte for byte. All 422 pair rows and all CSV fields reconcile.
- Independently verified 88,292 exported boundaries against annotations:
  sample IDs, labels, source files, writer IDs, pair positions and unique,
  complete boundary IDs. Export provenance is consistent with training fold 0.
- Training labels contain 19,907 recordings and 501 word types.
- Quality filtering retains 78,953 boundaries, 422 pairs and 42 writers.
  There are 2,170 marked empty-region fallbacks.
- Before filtering, 192 of 426 pairs pass the support gate; 78 of those occur
  in three or four words. After filtering, 176 pairs pass; 62 occur in three
  or four words (35.23%). A proposed five-word gate would leave 114 pairs.
  This is a coverage calculation, not an adopted setting.
- Per-pair word counts and exact supporting words are in
  [word_support.csv](../results/thesis/forced_alignment_review/wi_rh_fold0_20261115_baseline/word_support.csv).
  [validation_report.json](../results/thesis/forced_alignment_review/wi_rh_fold0_20261115_baseline/validation_report.json)
  records the reconciliation checks.
- Before/after hashes match for all recorded source inputs and old outputs.

### Provenance limits and next step

Current hashes identify the checkpoint, configuration, export and source files
present now. The historical export stores checkpoint path and epoch, but does
not store a checkpoint hash or original source revision. These cannot be
reconstructed as verified historical identities from current files alone.

Reproduction does not validate physical character timing or contact accuracy.
The score is not a calibrated contact probability. Word and writer stability,
trace review and any score variant remain planned. No model inference,
recognition training, tokenizer construction or English frozen-artifact update
occurred. Runtime verification was the unchanged scorer plus independent
row/support/output/hash checks; the full test suite was not rerun.

Next: Step 2 of [the alignment plan](FORCED_ALIGNMENT.md#step-by-step-validation-plan--2026-11-15).

## 2026-11-15 — A0 ranking stability across words and writers

### Purpose and authorization

The user approved Step 2 to check how stable the existing force ranking is
across training words and writers. The exact procedure was recorded in
[FORCED_ALIGNMENT.md](FORCED_ALIGNMENT.md#step-2-procedure-fixed-before-execution)
before execution. This is a sensitivity diagnostic of a fixed model/export.
It is not a recognition experiment or a change to tokenizer construction.

### Reproducibility

- Code revision: b7e4d5f4f542d8c73e967c13984cac339d349f02, forced-alignment.
  Existing uncommitted changes: docs/FORCED_ALIGNMENT.md, docs/PROGRESS.md,
  docs/EXPERIMENTS.md; existing untracked compatibility audit preserved.
- Python: /home/artellisys/miniconda3/envs/tva/bin/python, 3.11.15.
  PYTHONHASHSEED=42. Exhaustive deterministic omissions; no random sampling.
- Inputs and selected checkpoint are the Step 1 A0 WI/RH fold-0 train export,
  recording epoch 275 of the existing best-CER checkpoint. Fixed seed-42
  training history; no model inference, retraining or realignment.
- Unchanged settings: 100 ms local force; 0.75/0.25 local/region weights;
  occurrence/writer gates 100/30; stratum support 100; export force fraction
  0.10. Recompute all correction means/fallbacks and writer residuals per run.
- Command: PYTHONHASHSEED=42 /home/artellisys/miniconda3/envs/tva/bin/python
  results/thesis/forced_alignment_review/wi_rh_fold0_20261115_stability/analysis.py
- [Run manifest](../results/thesis/forced_alignment_review/wi_rh_fold0_20261115_stability/run_manifest.json)
  includes resolved settings, source/input/output SHA-256 hashes, package
  versions, exact command and pre-run Git state/diff. Analysis and independent
  export-check source files are saved beside the scientific outputs.
- Result directory:
  [wi_rh_fold0_20261115_stability/](../results/thesis/forced_alignment_review/wi_rh_fold0_20261115_stability/).
  Local outputs are Git-ignored; preserve them. A repeat analysis requires a
  new output directory via --output-dir; existing results are not overwritten.

### Results

501 word-type omissions and 42 writer omissions completed over the same
78,953 accepted training boundaries. The fixed comparison set contains all
176 baseline-eligible pairs. It retains diagnostic ranks after eligibility
loss. Ties use average ranks for Spearman and alphabetical order for displayed
integer ranks. Spearman describes order similarity, not contact correctness.

| Measure | Word omission | Writer omission |
| --- | ---: | ---: |
| Median Spearman, fixed 176 | 0.999971 | 0.998998 |
| Minimum Spearman, fixed 176 | 0.996586 | 0.997636 |
| Minimum Spearman, per-run common eligible pairs | 0.999238 | 0.997597 |
| Worst fixed-set top-20 retention | 19/20 | 19/20 |
| Worst fixed-set top-50 retention | 49/50 | 48/50 |
| Worst actual eligible-list top-20 retention (supplement) | 19/20 | 19/20 |
| Worst actual eligible-list top-50 retention (supplement) | 48/50 | 47/50 |
| Pairs losing eligibility at least once | 42 | 2 |
| Pairs always eligible within omission type | 134 | 174 |
| Largest absolute score change, 0-1 scale | 0.097608 | 0.023269 |
| Median across pairs of worst absolute score change | 0.013270 | 0.017145 |
| Largest fixed-set rank movement | 55 | 17 |

- dr: removing drei changes score 0.397002 to 0.299393 and diagnostic rank
  61 to 116. It loses eligibility; it has only three distinct word contexts.
- gr and hi show large word effects and lose eligibility after some context
  removals. Among word-gate failures, 33 pairs have three word types, eight
  have four, and one has five. Three baseline top-20 pairs (Di/gr/hi) can fail.
- wo: removing Mittwoch changes 0.500588 to 0.570879 and rank 28 to 14
  while retaining eligibility. Its six contexts show that five words alone
  would not guarantee score stability.
- Writer gate failures concern Ge and av, both starting at 101 accepted
  occurrences. The largest writer score change is sa, -0.023269 without
  writer 1. The largest writer rank movement is 17 places for ät.
- All 134 pairs always eligible across word omissions also survive every
  writer omission. This is a diagnostic intersection, not a new support gate.
- [stability_summary.json](../results/thesis/forced_alignment_review/wi_rh_fold0_20261115_stability/stability_summary.json)
  contains distributions and worst examples.
  [omission_summary.csv](../results/thesis/forced_alignment_review/wi_rh_fold0_20261115_stability/omission_summary.csv)
  records each omission;
  [pair_sensitivity.csv](../results/thesis/forced_alignment_review/wi_rh_fold0_20261115_stability/pair_sensitivity.csv)
  records each pair's range, worst responsible IDs and support losses.
  omission_pair_scores.csv.gz retains every pair/run row, including missing
  evidence. Eligible-list overlap supplements were added after the initial
  result inspection to expose support effects; the original metrics remain.

### Verification and limits

Ten comparisons against the saved/native scorer passed, including eight
word/writer omissions chosen from IDs and accepted counts before score
inspection. Maximum error is 3.33e-16 against the predeclared 1e-12 tolerance.
Support, eligibility, ranks and correction fallback counts match.
No production scoring function was edited; reference checks supplied the
accepted evidence through an in-memory reader.

Independent export verification checks all 543 omissions and 229,146 pair rows.
Gate decisions, ranks, sensitivity maxima and gate-loss counts reconcile.
An alternative rank calculation confirms Spearman within 2.22e-16.
[export_audit.json](../results/thesis/forced_alignment_review/wi_rh_fold0_20261115_stability/export_audit.json)
records those checks and supplemental eligible-list overlap. All recorded
input hashes match, including old results and the unrelated local audit.
The full test suite was not rerun.

The global order survives single omissions, but some pairs depend on specific
words. Most word omissions remove no direct observations of a particular
pair; high overall/median correlation is not sufficient evidence for that pair.
Direct-context changes and worst omissions are retained. Individual writer
extremes are smaller, but not every aggregate word-sensitivity measure is larger.

The aligner was trained on the complete training fold. These checks do not
measure retrained unseen-word/writer generalization, several simultaneous
omissions, calibrated uncertainty or physical boundary/contact accuracy.
The original export's missing historical checkpoint hash/source revision
remains a provenance limit. No new gate, score, dataset, tokenizer or
recognition condition was adopted.

Next: freeze the Step 3 trace-audit cases and review rules. No further
experimental stage was executed.

## 2026-11-16 — Fixed A0 force-trace audit: provisional AI review

### Scope and provenance

The user approved Step 3. The procedure in docs/FORCED_ALIGNMENT.md was saved
before raw trace loading. No score, threshold, model, alignment, split or
support gate changed. This uses only accepted OnHW-Words500 RH WI fold-0
training boundaries and their original raw 13-channel, 100 Hz recordings.

- Code revision: b7e4d5f4f542d8c73e967c13984cac339d349f02, forced-alignment.
  Existing three-document edits and the unrelated audit were preserved.
- Environment: /home/artellisys/miniconda3/envs/tva/bin/python, Python 3.11.15;
  NumPy 2.4.6 and Matplotlib 3.11.1. No dependencies were added.
- Frozen aligner/export: existing seed-42 A0 best-CER export, epoch 275.
  No model loading, checkpoint selection, inference or realignment occurred.
- Existing force fraction 0.10; local window 100 ms. The baseline's 0.75/0.25
  local/region weights, position-duration correction and 100/30 support
  gates remain unchanged. No continuity ranking was refitted.
- Sample: 30 different pairs per rank band (1-59, 60-118, 119-176),
  with one random accepted recording each, plus ten targeted cases.
  Python selection seed 42; review-order seed 4201. Exactly 100 distinct
  recordings, 96 pairs and 38 writers. No case replacement after inspection.
- Frozen selection SHA-256:
  d6bd59ebedd6ff0a0c3fe935e0ce0bb2b24398cc59dc331622dd75469002c722.
- [Local audit artifacts](../results/thesis/forced_alignment_review/wi_rh_fold0_20261116_trace_audit/)
  include selection/render/audit manifests, source for all three stages,
  raw feature checks, full-run extent checks, 25 PNG sheets, original visual
  notes and the final case ledger. The source protocol at freezing is preserved.
- [Review ledger](FORCED_ALIGNMENT_TRACE_AUDIT_20261116.csv) contains
  the exact 100 IDs, raw-source hashes, counts, visual notes and uncertainties.
  Its plot paths are relative to the local audit directory. Local results
  remain Git-ignored.

### Observed force measurements

These categories describe the existing force threshold. They are not physical
pen-lift labels: no low sample; longest low run 10-20 ms; or at least 30 ms.
Thirty milliseconds is a reporting category, not an adopted method setting.

| Sample group | n | Local no / brief / sustained | Region no / brief / sustained | Local no, region low | Multiple region low runs |
| --- | ---: | --- | --- | ---: | ---: |
| High | 30 | 19 / 2 / 9 | 9 / 0 / 21 | 10 | 4 |
| Middle | 30 | 8 / 2 / 20 | 6 / 0 / 24 | 2 | 6 |
| Low | 30 | 7 / 3 / 20 | 3 / 0 / 27 | 4 | 10 |
| Targeted | 10 | 5 / 1 / 4 | 4 / 1 / 5 | 1 | 3 |

The main 90 cases contain 16 local/region contact disagreements and 20
multi-run regions. These are diagnostic counts, not a population error rate.
All eight locally brief cases in the complete sample overlap full-trace
events of 60-150 ms. A047's region has longest low run 20 ms but intersects
two 80 ms full-trace events. These support interval-clipping as an explanation
for those short measurements; they do not establish the true letter boundary.

Three empty gaps use the retained 100 ms fallback and stay above threshold.
In A098, sample 117 is below threshold but lies just outside [107,117).
The 17 possible delayed-stroke cases remain unassigned: dots/crosses cannot
be confirmed from force alone. All 100 physical letter-event assignments
remain unverified.

### Review, validation and limits

Codex AI visually inspected all 25 sheets, four cases per sheet, with full
recording context and a detailed interval view. Images hide pair score, rank
band and selection reason. Labels, letters and writer IDs are visible.
Earlier aggregate findings were known. Review is partially masked, not blind,
not independently human-reviewed and not physical timing ground truth.

All 100 raw recordings match the original exported force references, local and
region minima/means, low fractions and longest runs. Each has finite 13-channel
data of the expected length. Frozen IDs and input hashes remain unchanged.
Current checkpoint, export, Step 1 and Step 2 results and production source
hashes reconcile. No full test suite, model inference or training ran.
The old export's absent historical code revision/checkpoint hash remains
unresolved; current hashes cannot retrospectively authenticate them.

Pair-balanced samples and targeted cases are not pooled into population
accuracy. Rank-band differences use the same evidence that created the
ranking and are internal consistency only. Visual force inspection cannot
independently validate tip-contact calibration or letter-specific event
assignment.

Interpretation: event extent and assignment need attention before a minimum
duration is adopted. A 30 ms rule inside a clipped window could discard
fragments of longer drops. Keep the baseline; Step 4 needs a concrete change
and comparison proposal. No tokenizer or recognition condition was selected.

Detailed examples, counts and the execution status are in
[FORCED_ALIGNMENT.md](FORCED_ALIGNMENT.md#2026-11-16--step-3-fixed-trace-audit-completed-as-provisional-ai-review).

## 2026-11-16 — Whole-force-event assignment pilot

### Approval and provenance

The user approved the Step 4 diagnostic. It uses the same frozen 100
OnHW-Words500 RH WI fold-0 training recordings and every gap in those words.
It is not recognition training or a new tokenizer construction condition.

- Revision: b7e4d5f4f542d8c73e967c13984cac339d349f02 plus saved uncommitted
  implementation/docs. Added tva/force_gap_assignment.py, assign_force_events.py
  and tests/test_force_gap_assignment.py.
- Python: /home/artellisys/miniconda3/envs/tva/bin/python, 3.11.15.
  Existing NumPy/Matplotlib; no dependency added.
- Existing seed-42 A0 best-CER export, epoch 275. No model loading,
  checkpoint selection, inference, realignment or training.
- Original 0.10 x recording-90th-percentile force threshold, independently
  reconciled with raw CSVs. No duration filter, smoothing, event merging,
  midpoint snapping or new support gate.
- Candidate links require mutually unique overlap, whole-event strict
  containment and a usable gap. Truncated/shared/multiple/empty/unreliable
  cases stay explicit. The legacy fallback supplies no physical gap.
- Shifts: -8/0/+8 samples (-80/0/+80 ms), with events fixed.
  Signature: category, event IDs, reasons and candidate ID; near-edge
  changes alone do not count.
- Command: PYTHONHASHSEED=42 python assign_force_events.py --output-dir
  results/thesis/forced_alignment_review/wi_rh_fold0_20261116_assignment_pilot
- [Pilot artifacts](../results/thesis/forced_alignment_review/wi_rh_fold0_20261116_assignment_pilot/)
  preserve exact source/input hashes, source snapshots, pre-run Git state,
  protocol, full CSV/JSON ledgers, plot sheets and coverage summaries.
  Local results are Git-ignored and protected against overwrite.

### Observed results

The 100 recordings contain 547 gaps and 556 recorded force events.
Events divide into 242 unique candidates, 282 ambiguous and 32 without
positive gap overlap. Gaps divide into 242 candidates, 113 no-drop,
161 ambiguous and 31 unavailable. Unavailable includes 21 unreliable and
12 empty gaps, with two sharing both flags. Every event and gap is retained.

| Audited group | n | Candidate | No drop | Ambiguous | Unavailable | Association-sensitive under either shift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| High | 30 | 11 | 9 | 10 | 0 | 16 |
| Middle | 30 | 16 | 5 | 8 | 1 | 14 |
| Low | 30 | 15 | 3 | 12 | 0 | 18 |
| Targeted | 10 | 3 | 2 | 3 | 2 | 2 |

Audited totals: 45 candidates, 19 no-drop, 33 ambiguous and three unavailable
empty gaps. These are diagnostic counts, not population accuracy or contact
probabilities.

50/100 associations change under at least one shift. In 36 cases the category
changes; another 14 change event IDs/reasons within the same category.
Only 21/45 candidates and 18/19 no-drop observations retain their association
under both shifts. Excluding unstable links would remove candidate drops
disproportionately in this sample. No such exclusion or new pair rate was used.

Examples: A001 retains a full 90 ms event despite the old local window missing
it. A002's 140 ms candidate becomes ambiguous in either shift. A047 intersects
two 80 ms events and stays ambiguous, but the overlapping IDs change.
A084's no-drop observation becomes ambiguous after the later shift.
The three empty gaps remain unavailable.

The 17 possible-delayed-left cases contain four candidates and 13 ambiguous
gaps. Flagging either neighbour gives 34 cases: nine candidates, four no-drop,
20 ambiguous and one unavailable. These are confounded descriptive groups,
not established dot/cross effects.

### Verification, interpretation and next step

- PYTHONHASHSEED=42 python -m unittest tests.test_force_gap_assignment
  tests.test_continuity_scoring: 22 pass (18 new, four existing).
- [Independent checks](../results/thesis/forced_alignment_review/wi_rh_fold0_20261116_assignment_pilot/verification_report.json)
  reconcile 1,668 event, 1,641 gap and 1,534 overlap rows across all variants.
  Separate raw scans/sample-set intersections reproduce event extents and
  overlaps. Category accounting, clean-link conditions, CSV fields and
  timing/group summaries match. Existing-output protection passes.
- All recorded original/raw/baseline/audit/Step 2 hashes match. No production
  scorer, model, tokenizer file or training config changed. Frozen IDs and
  the case ledger are preserved.
- An off-axis text-label issue was corrected only in plotting. Revised
  sheets have a separate plot_revision_v2/ source/hash manifest; original
  plots and numerical exports remain. Revised sheets 1/12/25 inspected.
  Scientific source functions remain identical to the saved run.
- No full test suite, inference, training or pair-score fitting ran.

The cases already informed this design. No independent physical timing or
contact truth was introduced. The +/-80 ms shifts are declared stress
checks, not calibrated error bounds. A robust temporal candidate still need
not be an actual inter-letter lift. The prior export's absent historical
source revision/checkpoint hash remains a provenance limit.

Outcome: full-event accounting and explicit uncertainty are implemented.
Timing sensitivity prevents promotion to a pair-ranking method at this stage.
Next: determine independent timing evidence or propose a controlled alignment
improvement before aggregation. Keep the old score; Steps 5-6 remain pending.

## 2026-11-17 — OnHW WI/RH fold-0 Bigram v0 prototype

- Used the 176 eligible pairs in the reproduced Step 1 force ranking as a
  common candidate set. The force vocabulary took the top 100 saved ranks.
  The control took the top 100 by adjacent position counts over all fold-0
  training recordings, using case-sensitive Unicode order for ties.
- Both vocabularies preserve 59 character IDs and blank ID 0. The existing
  greedy tokenizer round-tripped all 19,907 training labels with each.
  They share 58 pairs. The force vocabulary emitted 75,766 tokens (29.98%
  reduction from character targets), and the control 70,841 (34.53%).
- Independently checked the pair selections, per-pair emissions and token
  counts. Rebuilding under a second hash seed reproduced both vocabularies,
  the selection ledger and the summary byte for byte. The run manifest
  records source and input hashes and the exact command.
- This is a train-only tokenizer check. Recognition accuracy has yet to be
  measured. [Run artifacts](../results/thesis/forced_alignment_review/wi_rh_fold0_20261117_bigram_v0/) ·
  [Method and verification](FORCED_ALIGNMENT.md#2026-11-17--bigram-v0-built-for-onhw-wirh-fold-0)

## 2026-11-19 — OnHW WI/RH fold-0 Bigram v0 recognition pilot

- Scored seed-42 final checkpoints at epoch 299 on the same 5,292 recordings
  from 11 writers. The existing B0 character checkpoint was reused and all
  archived predictions reproduced exactly. The two new bigram trainings
  ran in train-only mode.
- Character CER/WER were 15.80%/28.10%. Frequency-ranked bigrams gave
  16.74%/25.42%. Force-ranked bigrams gave 16.28%/25.53%.
- The primary force-minus-frequency CER delta was −0.46 percentage points.
  Its paired writer-bootstrap 95% interval was [−1.15, +0.06] points.
  WER was +0.11 points with an interval of [−0.73, +0.90] points.
- This is one fold and seed. The A0 checkpoint used to build the force
  ranking was selected on fold-0 validation. The pilot does not establish
  a general recognition advantage from force ranking.
  [Run artifacts](../results/thesis/bigram/wi_rh_fold0_bigram_v0_pilot/seed_42/evaluation_20261119/) ·
  [Method and verification](FORCED_ALIGNMENT.md#2026-11-19--onhw-bigram-v0-recognition-pilot-completed)
