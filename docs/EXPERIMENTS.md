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

## 2026-11-21 — DHSD DTLR checkpoint diagnostic

The user requested trying DTLR on DHSD. Created and executed
[DHSD_DTLR_Check.ipynb](../../DTLR/DHSD_DTLR_Check.ipynb) using the existing
WSL dtlr-poc kernel. This is an image-domain diagnostic, separate from the
OnHW IMU forced-alignment work. It does not change that method, its scores,
its tokenizer files or any recognition run.

**Data and fixed sample.** Local DHSD v1.0.0 root:
/mnt/c/Users/Ali/Downloads/DHSD-v1.0.0/nauman-riaz-DHSD-c64d620.
[Publisher record](https://doi.org/10.5281/zenodo.18743313), CC BY 4.0.
All 5,939 annotated images decode as 256x64 RGBA. All are opaque. The supplied
4,745 training and 1,194 test files are disjoint and cover the full CSV, but
all 37 writers occur in both sets: the supplied split is not writer-disjoint.
The checks establish structural validity, not transcript correctness.

Before inference, fix 12 random training images from sorted paths with seed 42.
Then add seven distinct training images, one for each äöüÄÖÜß, using the same
RNG and preferring writers not yet selected. There are 19 images from 18 writers:
12 random images from 11 writers; seven targeted images from seven additional
writers. No test image, prediction-based selection or later exclusion is used.
A source-only overview was inspected before predictions were reviewed. Some
crops are faint and possible case/punctuation differences remain. Labels are
unchanged; no independent human reference audit or corrected reference is claimed.

**Models and settings.** Strictly load the same three local checkpoints used
in the ScaDS diagnostic, with native alphabets and saved decoder layouts:

- German synthetic pretrained, epoch 20, 166 classes; SHA-256
  840c39a672539b7a6f45b758801d668f27d95ace4f93fc334b42f74cfd6351de.
- READ-finetuned, epoch 3043, 85 classes, saved single-linear decoder classifier;
  SHA-256 3b04f884ef527321df798325987ac34def2990237ad7a8d7cbe2db23676ddea8.
- IAM-finetuned, epoch 1799, 166 classes; SHA-256
  d076f1276d973c2e3a945238738174b4ce9ebc87257633832bf32be00e473b8f.

Use seed 42, batch one, score >0.3, NMS IoU 0.5, top-k 900, RGB conversion,
deterministic test resize (short side up to 800, long side up to 1333) and
ImageNet normalization. Do not crop, repair spelling, compose Unicode, replace
class IDs or remove unsupported characters. Existing monotonic edit alignment,
Otsu and dominant-core-v3 code is unchanged. There are 57 new forward passes
and 763 saved boxes: German 287, READ 153, IAM 323. This is ordinary DTLR
inference plus text alignment, not CTC forced alignment or model training.

**Unchanged supplied-label results:**

| Group | Model | Exact letters | Alignment (%) | Usable exact letter pairs | Pair usability (%) | Nonspace edits / reference characters | Text disagreement (%) |
| --- | --- | --- | ---: | --- | ---: | --- | ---: |
| random (12) | German pretrained | 116/194 | 59.79 | 70/172 | 40.70 | 79/196 | 40.31 |
| random (12) | READ-finetuned | 33/194 | 17.01 | 6/172 | 3.49 | 162/196 | 82.65 |
| random (12) | IAM-finetuned | 142/194 | 73.20 | 91/172 | 52.91 | 71/196 | 36.22 |
| German-letter examples (7) | German pretrained | 59/98 | 60.20 | 32/85 | 37.65 | 40/100 | 40.00 |
| German-letter examples (7) | READ-finetuned | 25/98 | 25.51 | 7/85 | 8.24 | 75/100 | 75.00 |
| German-letter examples (7) | IAM-finetuned | 69/98 | 70.41 | 42/85 | 49.41 | 34/100 | 34.00 |

Groups have different selection rules; do not pool them into a representative
corpus score. Nonspace disagreement is case-sensitive edit distance/reference
length after removing whitespace, not cleaned benchmark CER. None of the three
models matches any of the 19 complete supplied transcripts after whitespace
removal. Text alignment and component usability do not measure geometric box
accuracy or verified ink-connection labels.

Across all selected labels, special-letter matches are German 4/34 (ä 1/3,
ö 2/6, ü 1/6), READ 1/34 (ü 1/6), IAM 0/34. All checkpoints match 0/16 ß
occurrences. IAM has no ß class; READ lacks separate Ä and Ü classes. German
pretrained has no native alphabet gaps on this sample but still fails ß. Keep
all occurrences in the denominators. In the complete supplied training labels,
ß appears 3,999 times across 3,380 of 4,745 images; this is a label count, not a
verified visible-character count.

IAM remains strongest on the random sample, with substantial remaining errors.
Changing to DHSD does not establish a ready-to-use German evidence pipeline.
DHSD words/phrases, preprocessing and label mix differ from ScaDS lines, so
cross-pilot percentage differences cannot isolate corpus quality or transfer.
Different model alphabets and training histories also prevent a causal style
claim. No checkpoint, dataset or production evidence condition is adopted.

**Outputs and reproducibility.**
[outputs/dhsd_dtlr_checkpoint_check_20261121_v1/](../../DTLR/outputs/dhsd_dtlr_checkpoint_check_20261121_v1/)
contains 35 local Git-ignored files: CSV/image inspection, fixed selection,
resolved run/checkpoint manifests, detections, 57 metric rows, 795 pair rows,
paired predictions, nine PNG/PDF figure pairs and an executed notebook snapshot.
The notebook embeds all nine figures and every selected case. Source image,
CSV, code and checkpoint hashes are stored in the manifests.

- DTLR revision: 4a74692e16d11c9744fa71cd01ee5a982f6aa67b (iam-handwriting-bpe-unigram).
- TVA revision: dfb95e375ffb5dc084ab45b6243bdfe3ef0bd604 (forced-alignment).
- Existing uncommitted DTLR ScaDS notebook/docs work and TVA's untracked
  TOKENIZER_COMPATIBILITY_AUDIT.md are preserved and recorded in the manifest.
- dataset_inspection.json SHA-256:
  e2734e62fd5c713879fa8b2b35cf33841c2e286fac33f471019773f97ff09283.
- selection.json SHA-256:
  4d5e14198ddb7d1ceb841c37884fd4f71afcfd27c9aca5ab7395388030cc5c5b.
- manifest.json SHA-256:
  ba736056ed9be8a231ab74fbdbfb3e9649633831478b6501a3f403dcb7b9789d.
- line_metrics.csv SHA-256:
  613a55077e539e9253df417bf425c9679d832755742fae9746c53b8dc6d53c81.
- group_summary.json SHA-256:
  1aa1c613a99fd6c5b0395a2fa8b1ce1947fc31728a578ed706cba3a24972c04d.
- pair_diagnostics.json SHA-256:
  21c41f65a07fb345d04f821617ba60465f8c0b1c6b5e3644562b7e203cba498f.
- completion.json SHA-256:
  727c7cd3bc41b8aa1b58a1fc5c431c7619af20e5d55b4d51316b17cd573a186d.
- Notebook source SHA-256:
  31b527476c330c596e0333e5ad11c27d5b6e777e01a3059ebbb83c65cd582b0b.

**Verification.** All six code cells pass in the real dtlr-poc kernel
(Python 3.11.0, torch 2.1.0+cu118, RTX 4060 Laptop GPU). An independent
rolling-row edit-distance implementation confirms all 57 metric rows. Native
class IDs, pair counts, summary denominators, selected image/CSV hashes and
all nine embedded/exported PNGs reconcile; nine PDFs exist. Source-only and
representative box/chart layouts were inspected. All 220 prior ScaDS notebook/
output files retain identical hashes. A targeted cache check reruns the unchanged
model, score, plot and completion cells with authenticated frozen inputs; it
reuses all three prediction caches and preserves all 35 output files. The slow
full-image audit is not repeated in that targeted check. The first execution
attempt hit a runner output timeout during the full image audit, before inference;
a rerun with progress monitoring completed. No training, dependency, model-library
change, threshold tuning, source-data edit, full test suite or construction
export was needed. Next: human reference/box review and a declared adaptation or
image-alignment plan before any full German evidence extraction.

## 2026-11-21 — A100 IAM-to-DHSD setup (software verification)

The user requested a notebook/setup ready to clone and train over SSH on one
A100. Added an isolated package under DTLR training/dhsd and
[DHSD_A100_Training.ipynb](../../DTLR/DHSD_A100_Training.ipynb). The
[setup guide](../../DTLR/training/dhsd/README.md) contains exact transfer,
Conda/CUDA build, tmux, preparation, preflight, benchmark, training, resume and
explicit final-evaluation commands. This is implementation validation, not a
completed German training experiment or A100 benchmark.

The launcher reuses DTLR's unmodified model and adapted CTC loss. It avoids the
upstream loop's silent exception skipping and image-count iteration limit.
Successful optimizer updates have an explicit budget. The pilot uses 1,000
classifier updates then 19,000 network updates, batch four, seed 42, AdamW,
head/network/backbone learning rates 1e-4/1e-5/1e-6, norm limit 0.01 and
validation/save every 1,000 updates. Float32 with TF32 is the default; custom
CUDA ops are not assumed AMP-safe. A100 extensions are built for architecture
8.0 with the existing pinned Python 3.11.0 / torch 2.1.0+cu118 / CUDA 11.8 runtime.

Writer-disjoint preparation replaces the publisher's overlapping split:
27/5/5 writers and 4,354/805/780 images. Test writer IDs are 24,30,31,6,8;
none appeared in the earlier 18-writer preview. Source CSV/image checksums,
raw supplied labels and split definitions are frozen. No independent human
label verification is claimed. Test labels/frequencies are not used for model
alphabet construction: preserve all native IAM IDs and append declared German
characters plus unsupported characters observed in training. For this release
this appends ß and š, making 168 classes. New rows use seeded existing-row
initialization; all original rows/IDs and decoder sharing are preserved.

Native IAM is evaluated once on the same validation images. Checkpoint
selection uses minimum raw validation CER; ties keep the earlier step and WER
comes from the same checkpoint. Detection scoring remains score >0.3, NMS 0.5,
top-k 900. The test set is never evaluated automatically. The companion
notebook shows commands and reviews logs/validation boxes without auto-launching
training. Lower text error does not validate box geometry or ink connectivity.

Local verification used the RTX 4060 dtlr-poc environment. Thirteen focused tests
pass. A four-update, batch-one, reduced-resolution smoke job covers two head
and two network updates. Another job used batch two and two workers, stopped
at step 2 and resumed to step 4. Its step-2 checkpoint stays byte-identical;
model/optimizer/RNG resume, stage changes, finite parameters and strict 168-class
checkpoint loading pass. A separate validation invocation reproduces the saved
selected-checkpoint error counts/CER/WER. These four-image smoke scores are
software checks, not a German recognition result. An initial preflight stopped
before updates on unsupported training š; the corrected extension policy is
covered by a test. All five notebook cells execute and both rendered figures
were inspected. Shell syntax checks pass. Data preparation reproduces byte for
byte in a clean clone. No upstream source, prior image result, OnHW artifact,
source dataset or original checkpoint was changed. A100-specific compilation,
production batch memory, throughput, convergence and full training remain pending.
A final CUDA stop/resume job also recovered deliberately missing latest pointers
and stale best metadata from complete step files, then reproduced the saved
validation metrics. Session-specific validation files protect replayed attempts.

Code-only local branch: codex/dhsd-a100-training; commit 87651f819daa15b20801d871cd1eef09d1060718, based on 4a74692e16d11c9744fa71cd01ee5a982f6aa67b through two setup commits. The cumulative change contains only the 15 new training/notebook files. A temporary Git index preserved the active branch and unrelated work; no GitHub/GitLab push occurred.

Private transfer artifact:
[dtlr-dhsd-a100-kit.tar](../../DTLR/outputs/dhsd_a100_setup_20261121_v2/dtlr-dhsd-a100-kit.tar),
650,670,080 bytes. It includes a full-history Git bundle, original DHSD ZIP,
original IAM checkpoint1799.pth, prepared manifests, START_HERE.sh and checksums.
It contains no compiled RTX binaries or local smoke checkpoints. The starter
was tested in a clean deployment directory: payload authentication, clone,
data extraction and weight paths pass; all thirteen tests also pass from that
clone. Original weights/data are included for private transfer, not publication.

- Transfer archive SHA-256:
  21cc6d3814488207105b8c23c726f509d50b6ef5b16d207d6d566dcf8317d323.
- Initial IAM checkpoint SHA-256:
  d076f1276d973c2e3a945238738174b4ce9ebc87257633832bf32be00e473b8f.
- Prepared train/validation/test JSON SHA-256:
  9b37c4a0eb558af162852a7ab0ee26a93d666d4ffa94464591c9e99d60d7be16;
  434ef04f28d15b893f9ffe410c0b978615387484ba8d636a307e2d46101bfadc;
  bde689a5bd1767dd9433ed70bc6e5be2ed1c3ff94ebbeb2a62b27cb1dde1c2a9.
- Further code/asset hashes and bundle identity are in
  [KIT_MANIFEST.json](../../DTLR/outputs/dhsd_a100_setup_20261121_v2/kit/KIT_MANIFEST.json).
- [Verification limits](../../DTLR/training/dhsd/VERIFICATION.md) distinguish
  local software checks from unrun A100/full-training work.

Next: copy the private kit to the allocated A100, build the pinned runtime and
native extensions, run the four-update preflight and 200-update production-size
benchmark, inspect memory/speed, then launch the bounded pilot. The earlier
6–12 GPU-hour estimate is not a measured A100 runtime. No adaptation gain is
claimed by preparing this setup.
