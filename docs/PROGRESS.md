# Thesis Progress Log

This file records completed work, important observations, and immediate next
steps. Add new dated entries below the current snapshot, newest first.
Preserve the older entries, including failed attempts.

## Current status - updated 2026-11-21

- An SSH/A100 IAM-to-DHSD training kit is prepared. It includes a cloneable
  local Git bundle, original data/weights, fixed writer-disjoint splits,
  staged training, resume support and a companion notebook. Local software
  checks pass; A100 compilation, speed and full adaptation remain pending.
  [Setup guide](../../DTLR/training/dhsd/README.md)
- A separate DHSD image diagnostic is complete. On 12 fixed random training
  images, IAM DTLR reaches 73.20% exact letter alignment and 52.91% usable
  exact pairs; substantial errors and German special-letter failures remain.
  This does not change the OnHW IMU method or adopt a German image corpus.
  [DHSD experiment record](EXPERIMENTS.md#2026-11-21--dhsd-dtlr-checkpoint-diagnostic)
- OnHW WI/RH fold-0 Bigram v0 was evaluated at fixed epoch 299, seed 42.
  The completed bigram trainings were compared with the existing B0 character
  checkpoint on the same 5,292 recordings from 11 writers.
- CER/WER were 15.80%/28.10% for characters, 16.74%/25.42% for frequency
  ranking and 16.28%/25.53% for force ranking. Force reduced CER by 0.46
  points relative to frequency. The paired writer interval includes zero.
  [Evaluation record](FORCED_ALIGNMENT.md#2026-11-19--onhw-bigram-v0-recognition-pilot-completed)
- Both 100-pair bigram vocabularies keep the character fallbacks and use the
  Step 1 ranking evidence. The Step 4 assignments remain sensitive to gap
  timing. Step 5 broader consistency work remains open.

## 2026-11-21 — Cloneable IAM-to-DHSD A100 setup prepared

- Prepared [the training setup](../../DTLR/training/dhsd/README.md) and
  [companion notebook](../../DTLR/DHSD_A100_Training.ipynb). A private transfer
  archive contains the code bundle, original IAM checkpoint, original DHSD ZIP,
  frozen prepared splits and checksums. Its starter authenticates and clones
  the code without requiring a remote push or changing the current checkout.
- The seed-42 split has 4,354/805/780 images from 27/5/5 disjoint writers.
  Held-out test writers exclude the 18 writers shown in the earlier preview.
  The native 166 IAM IDs remain intact; missing ß and training-observed š are
  appended. No label is dropped or silently repaired. Default pilot budget:
  1,000 classifier updates plus 19,000 network updates, batch four.
- Thirteen tests, real reduced-resolution CUDA head/network updates, a two-worker
  stop/resume job, strict trained-checkpoint reload and all five notebook cells
  pass. Clean clone/materialization and byte-identical preparation also pass.
  The initial preflight caught unsupported š before any update; this is retained
  in the verification note. Smoke scores are not adaptation-performance evidence.
- [The implementation record](EXPERIMENTS.md#2026-11-21--a100-iam-to-dhsd-setup-software-verification)
  links provenance and limitations. A100-specific bootstrap, production memory/
  speed and full 20,000-update training remain unverified. OnHW and prior image
  results remain unchanged. Next: transfer the kit, build on the allocated A100,
  run preflight and benchmark, then launch the pilot inside tmux.

## 2026-11-21 — DTLR tested on a fixed DHSD sample

- Created and executed [DHSD_DTLR_Check.ipynb](../../DTLR/DHSD_DTLR_Check.ipynb):
  12 random training images plus seven German-letter examples, 19 images from
  18 writers. Compared German pretrained, READ-finetuned and IAM-finetuned
  checkpoints with matched settings and unchanged supplied labels.
- Random-sample exact letter alignment / usable exact pair rates are
  59.79%/40.70% German, 17.01%/3.49% READ and 73.20%/52.91% IAM. None of the
  19 complete transcripts matches exactly under any checkpoint after removing
  whitespace. All models fail all 16 selected ß occurrences; IAM lacks that class.
- All 5,939 images decode. Supplied train/test files are disjoint, but all 37
  writers occur in both sets. This is a development diagnostic, not independent
  writer evaluation. Supplied labels and geometric boxes lack human validation.
- All six code cells pass. Independent checks confirm 57 metric rows, pairs,
  native classes, input hashes and nine figures. Targeted cache verification
  preserves all 35 outputs; all 220 earlier ScaDS files are unchanged. No model
  training, data cleaning, full evidence export or IMU method change occurred.
- [The experiment record](EXPERIMENTS.md#2026-11-21--dhsd-dtlr-checkpoint-diagnostic)
  retains protocol, counts, provenance and the initial runner timeout. Next:
  human box/reference review before selecting an adaptation or alignment plan.

## 2026-11-19 — Bigram v0 recognition pilot completed

- Checked all three checkpoints and reconciled per-recording errors against
  original references. The existing B0 predictions reproduced exactly.
- Saved configs, checkpoint and vocabulary hashes, paired predictions and
  writer-bootstrap summaries to the local results directory.
  [Evaluation artifacts](../results/thesis/bigram/wi_rh_fold0_bigram_v0_pilot/seed_42/evaluation_20261119/)

## 2026-11-17 — Bigram v0 vocabularies built

- Built and reloaded force-ranked and frequency-ranked vocabularies with the
  existing greedy tokenizer. Independently checked selection and token counts
  and reproduced the data files byte for byte with a second hash seed.
- Both vocabularies and the provenance manifest are in the local results
  directory. No held-out labels, inference or training were used.
  [Run artifacts](../results/thesis/forced_alignment_review/wi_rh_fold0_20261117_bigram_v0/)

## 2026-11-16 — Whole-force-event assignment pilot completed

- The user approved Step 4. Added the standalone detector/matcher, fixed-case
  runner and 18 focused tests. No dependency or production score changed.
- Ran the 100 frozen recordings and all 547 word gaps, keeping 556 complete
  recorded events. The 100 audited gaps give 45 candidates, 19 no-drop,
  33 ambiguous and three unavailable cases.
- The [alignment execution record](FORCED_ALIGNMENT.md#2026-11-16--step-4-pilot-completed-explicit-uncertainty-and-timing-sensitivity)
  reports group counts and examples. Timing shifts change 50 associations
  and 36 categories. Only 21/45 candidates versus 18/19 no-drop observations
  retain their association in both shifts. No uncertainty-based exclusion
  or new connectivity percentage was adopted.
- Verification: 22 focused tests pass. Independent raw scans/sample-set
  intersections reconcile 1,668 event, 1,641 gap and 1,534 overlap rows.
  CSV fields, category accounting and input hashes match. Overwrite guard
  passes. Plot-only clipping corrected additively; original exports remain.
- [The experiment record](EXPERIMENTS.md#2026-11-16--whole-force-event-assignment-pilot)
  links source snapshots, command, manifests and findings. No inference,
  training, pair-score fitting or tokenizer construction ran.
- Next: independent timing reference or an alignment-improvement proposal
  before aggregation. Keep the old ranking. Steps 5-6 remain pending.

## 2026-11-16 — Whole-event assignment proposal documented

- At the user's request, added the simple rule and a concrete Step 4 pilot
  design to [FORCED_ALIGNMENT.md](FORCED_ALIGNMENT.md#proposed-rule-whole-drops-conservative-links).
  This is a proposal; no implementation or new experiment was run.
- Detect complete low-force runs using the existing threshold. Compare each
  event against every gap in its recording. Keep edge crossings, competing
  gaps, multiple drops, empty gaps and recording-edge events uncertain.
- Preserve full event durations and IDs. Do not snap events to midpoints,
  double-count them, or treat an uncertain gap as a connected join.
- The proposed 100-recording pilot reports assignment coverage and changes
  under declared -80/+80 ms gap shifts. It preserves the old score and sample.
- No new pair-contact percentage is specified. Excluding ambiguous cases
  could bias pair comparisons; coverage and missingness must be checked first.
- Next: review the proposal. Independent physical timing/contact validation
  remains open. Existing code, raw data, audit ledger and results are preserved.

## 2026-11-16 — Fixed force-trace audit inspected and recorded

- The user approved Step 3. Froze 100 cases before raw trace loading:
  30 pair-balanced cases per rank band and 10 targeted cases, seeds 42/4201.
  Sample coverage: 100 recordings, 96 pairs, 38 writers and all position groups.
- Codex AI inspected all 25 four-case sheets with scores/groups hidden.
  Earlier aggregate findings were known; the review is partially masked,
  provisional and not independently human-verified.
- [The 100-case ledger](FORCED_ALIGNMENT_TRACE_AUDIT_20261116.csv) preserves
  individual visual notes, raw hashes, exact force counts and unresolved
  physical assignments. The [alignment execution record](FORCED_ALIGNMENT.md#2026-11-16--step-3-fixed-trace-audit-completed-as-provisional-ai-review)
  gives group-specific counts, failure examples and the frozen sample hash.
- Among the 90 main cases, 16 local windows stay above threshold while the
  wider region has sustained low force; 20 regions have multiple low runs.
  Eight brief local runs across all 100 are fragments of 60-150 ms full-trace
  events. A047's short region cuts two 80 ms events. A simple 30 ms local
  rule would therefore conflate interval clipping with short sensor dips.
- All 100 raw recordings reproduce saved force summaries. Frozen sample,
  raw sources, old results and code hashes are preserved. No score change,
  model inference or training occurred.
- [The experiment record](EXPERIMENTS.md#2026-11-16--fixed-a0-force-trace-audit-provisional-ai-review)
  links local plots, manifests and reproducible source. This audit does not
  establish physical contact accuracy. Next: a concrete Step 4 proposal.

## 2026-11-15 — Ranking stability checked across every word and writer

- The user explicitly authorized Step 2. Recorded its exact population,
  omission procedure, rank metrics and numerical checks before execution in
  [FORCED_ALIGNMENT.md](FORCED_ALIGNMENT.md#step-2-procedure-fixed-before-execution).
- Completed 501 word-type and 42 writer omissions. Recomputed all correction
  baselines and pair scores with the same model/export, weights and gates.
  Fixed-set ranks keep the original 176 pairs; support losses are separate.
- Broad order is stable: minimum Spearman correlations 0.996586 and 0.997636.
  Fixed-set top-20 overlap never falls below 19/20 in either mode.
- Some word effects are large. Removing drei moves dr from 61 to 116 and
  below the gate. Removing Mittwoch moves still-eligible wo from 28 to 14.
  Eligibility fails for 42 pairs in at least one word omission and for only
  Ge/av in writer omissions; Ge/av begin at 101 occurrences.
- [The experiment entry](EXPERIMENTS.md#2026-11-15--a0-ranking-stability-across-words-and-writers)
  links all results and the reproducible local analysis. Ten scorer
  comparisons agree within 3.33e-16; independent checks reconcile all
  229,146 saved pair rows and every omission summary. Input hashes match.
- Documentation verification: all new local links/anchors resolve and
  git diff --check passes. Checked the summary support counts against the
  baseline: Ge and av have 101 observations, not exactly the 100 gate.
- These are fixed-model, single-omission diagnostics. They do not prove
  physical continuity or new-word/writer model generalization. No score,
  support gate, tokenizer or recognition condition was changed.
- Next: Step 3 fixed trace audit. Its cases/review rules remain to be frozen.
  Full test suite not rerun; no model inference or training occurred.

## 2026-11-15 — Alignment validation plan and unchanged baseline reproduction

- The user asked for a step-by-step plan and ongoing updates in
  [FORCED_ALIGNMENT.md](FORCED_ALIGNMENT.md). Preserved the linked branch's
  original method review there, with source revision and branch-specific
  decision context. Added interpretation limits and six staged checks.
- Reproduced the existing scorer in a new, separate local directory.
  Both saved outputs match the original byte for byte. All 422 scored pairs,
  176 eligibility flags and every CSV field reconcile.
- Checked all 88,292 boundary records against training annotations and
  verified complete unique boundary IDs. Retained 78,953 quality-filtered
  boundaries across 42 writers. All source and prior-result hashes are unchanged.
- The review's unfiltered support counts are confirmed. After the actual
  filter, 62 of 176 eligible pairs occur in three or four words; a proposed
  five-word gate would leave 114. No gate or score setting changed.
- [The experiment record](EXPERIMENTS.md#2026-11-15--a0-force-baseline-reproduction-and-word-support-audit)
  links the local manifest, exact command, checksums, word-support table and
  limitations. No checkpoint inference, tokenizer fitting or training occurred.
- Verification: saved artifact hashes reconcile; all new local links and
  anchors resolve; the source review is preserved exactly; git diff --check
  passes. No scoring/model code changed. The full test suite was not rerun.
- Next: Step 2 stability across words and writers. The remaining diagnostics
  are planned, not completed. Model timing and physical continuity are not
  established by baseline reproduction.

## 2026-08-01 — Initial setup and dataset preparation

### Completed

- Corrected the thesis direction: raw IMU remains the model input, while
  handwriting-informed tokens are used as text-label outputs.
- Confirmed the dataset scope with the supervisor: right-handed data only.
- Created the `tva-thesis` Conda environment with Python 3.11 and installed the
  repository dependencies.
- Prepared the right-handed writer-dependent (WD) and writer-independent (WI)
  OnHW-Words500 datasets in the TVA CSV/JSON format.
- Normalized repository-relative dataset paths in the YAML configurations.

### Dataset validation

Both processed datasets report 13 channels, a 100 Hz target rate, five folds,
and the correct WD/WI metadata flag.

| Dataset | Train samples by fold | Validation samples by fold | Writer check |
| --- | --- | --- | --- |
| WD/RH | 20,163; 20,160; 20,160; 20,158; 20,161 | 5,036; 5,039; 5,039; 5,041; 5,038 | All 53 writers occur in both splits, as expected for WD |
| WI/RH | 19,907; 20,078; 20,202; 20,520; 20,089 | 5,292; 5,121; 4,997; 4,679; 5,110 | Zero train/validation writer overlap in every fold |

For each dataset:

- 125,995 CSV files are referenced and present;
- no annotation references are missing or duplicated;
- annotation IDs are contiguous within every fold and split;
- no label is empty or contains a character outside the configured alphabet;
- 501 unique word labels occur across the complete dataset; and
- a deterministic sample of 200 CSV files contained 13 numeric, finite columns
  and valid sequence lengths.

The CSV content check was sampled rather than a complete scan of all 251,990
files. The structural and annotation-reference checks covered the complete
datasets.

### Immediate next steps

1. Create the fold-specific character tokenizer files for WD and WI.
2. Add dedicated, committed B0 character-baseline configurations.
3. Make the training code's mixed-precision context follow the selected device
   instead of assuming CUDA.
4. Add a true small-sample smoke-test option; two full CPU epochs are not a
   genuinely quick smoke test with roughly 20,000 training samples per fold.
5. Run a tiny local pipeline test, followed by the full fold-0 B0 baseline on
   an NVIDIA training machine.

## 2026-08-01 — Character tokenizer validation

- WD/RH character tokenizer files exist for folds 0–4.
- Each file contains 60 contiguous IDs, with the CTC blank assigned to ID 0.
- All five WD files have the same SHA-256 hash, as expected because the
  character vocabulary is fixed rather than learned from fold frequencies.
- Encode/decode round trips succeeded for all 25,199 train-plus-validation
  labels in every WD fold.
- The tokenizer notebook also generated WD Bigram, BPE, and Unigram variants;
  these are not yet needed for the B0 character baseline.
- WI/RH character tokenizer files were subsequently generated for folds 0–4.
  They use the same valid 60-token vocabulary and passed encode/decode checks
  for all 25,199 train-plus-validation labels in every fold.
- Character-tokenizer preparation for both WD/RH and WI/RH is complete.

## 2026-08-01 — B0 configurations and local CPU smoke test

### Completed

- Added dedicated fold-0 B0 character-baseline configurations for the WD/RH
  and WI/RH datasets under `configs/thesis/`.
- Made mixed precision device-aware: CUDA training uses mixed precision, while
  CPU training runs in normal precision.
- Added optional `max_train_samples` and `max_val_samples` configuration keys.
  Missing or zero values keep the complete dataset, so scientific experiment
  configurations are unaffected.
- Updated the ignored Mac-local configuration to use 16 training samples,
  eight validation samples, batch size two, and one epoch.
- Protected the learning-rate scheduler against a zero-length cosine phase in
  a one-epoch development test.

### Smoke-test result

The complete WD/RH fold-0 pipeline ran successfully on the Mac CPU:

- eight training batches and four validation batches completed;
- the model received the raw 13-channel signals;
- character tokenization and CTC loss worked;
- backward propagation and optimizer updates completed;
- validation decoding and CER/WER evaluation completed; and
- metrics, predictions, a visualization, and a checkpoint were written under
  `results/thesis/development/mac_cpu_smoke_char_wd/0/`.

The smoke-test CER and WER were both 1.0. This is expected after one epoch on
only 16 training examples and is not a thesis result.

### Next step

Set up the CUDA environment on the RTX 4060 laptop and run a short GPU test
before starting the full WD/RH fold-0 B0 experiment.

## 2026-08-02 — Resumable checkpointing

- Extended the repository's model-only checkpoints to preserve the optimizer,
  learning-rate scheduler, mixed-precision scaler, completed epoch, metric
  history, random-number generators, and DataLoader shuffle generator.
- Training now writes `latest.pth` atomically after every completed epoch.
  Numbered milestone checkpoints continue to follow `freq_save`.
- Older model-only checkpoint files remain usable for loading weights, but are
  clearly logged as non-resumable.
- A controlled CPU test trained epochs 0 and 1, then launched a fresh process
  from the saved epoch-0 checkpoint. The new process correctly began at epoch
  1 and reproduced the uninterrupted run's learning rates and batch losses
  exactly. This validates deterministic continuation for the tested setup.
- The first CUDA resume attempt exposed that loading the whole checkpoint
  directly onto the GPU also moved RNG state tensors to CUDA, while PyTorch
  requires CPU `ByteTensor` RNG states. Checkpoints now load through CPU and
  defensively convert RNG and DataLoader generator states to CPU before
  restoration. The existing checkpoint remains valid.

## 2026-08-02 — Validation-selected checkpoints

- Added automatic `best_cer.pth` and `best_wer.pth` checkpoints.
- Each file is replaced only for a strict improvement over all previous
  validation epochs. Restored metric history makes this rule resume-aware.
- `latest.pth` remains the recovery checkpoint, while the best-CER checkpoint
  is the preferred model for forced alignment.

## 2026-08-02 — Tokenizer-family scope clarification (historical IMU-first plan)

- Clarified that the thesis plans handwriting-aware Bigram, BPE, and Unigram
  tokenizers, not only one motion-selected pair vocabulary.
- Each proposed variant will be compared with its linguistic counterpart at a
  matched vocabulary size; the character baseline remains the common reference.
- At that stage, motion continuity was the primary token-selection evidence.
  The approved September 2026 proposal supersedes this with image-derived ink
  connectivity as the primary evidence. Frequency,
  occurrence count, and writer coverage are retained as reliability conditions.
- Implementation remains staged: establish forced alignment and motion-aware
  Bigram first, then reuse the shared evidence for BPE and Unigram.
- Added method, decision, and proposal-planning documents so this clarification
  is carried into the formal thesis proposal.

## 2026-08-02 — WD/RH fold-0 character baseline

- Completed 300 epochs of B0 character training on the RTX 4060 laptop.
- Best validation CER was 0.127563 (12.76%) at epoch 286.
- Best validation WER was 0.359809 (35.98%) at epoch 289.
- The complete run was archived and backed up by the researcher. The final
  `latest.pth` checkpoint was retained for initial forced-alignment work.
- This run preceded automatic best-model checkpointing, so the exact weights
  from epochs 286 and 289 are not available. This limitation is recorded in
  `docs/EXPERIMENTS.md`.

## 2026-08-02 — WI/RH fold-0 character baseline

- Completed 300 epochs of B0 character training on the RTX 4060 laptop.
- Best validation CER was 0.156041 (15.60%) at epoch 244.
- Best validation WER was 0.279478 (27.95%) at epoch 269.
- The WI run used automatic best-model checkpointing. Use `best_cer.pth`
  from epoch 244 for WI forced-alignment development; retain `best_wer.pth`
  for recognition comparison and `latest.pth` for resumable recovery.
- WD/RH and WI/RH fold-0 character baselines are now complete, so development
  can move to target-constrained CTC forced alignment.

## 2026-08-08 — Core CTC Viterbi alignment

- Created the `forced-alignment` development branch from the documented
  baseline state.
- Added a standalone target-constrained CTC Viterbi implementation in
  `tva/ctc_alignment.py`.
- The implementation returns the expanded CTC target, best state/token path,
  total log score, and a half-open model-frame interval for each known target
  token.
- Added five focused unit tests covering an ordinary two-character alignment,
  repeated characters, target-constrained advancement, insufficient frames,
  and an invalid blank inside the target.
- All five tests pass on the Mac `tva-thesis` environment.
- The implementation currently stops at model-output frames. Loading real
  checkpoints, mapping frames to raw IMU positions, visualization, and an
  alignment-confidence definition are intentionally left for the next stage.

## 2026-08-08 — Single-sample alignment runner

- Added `align_sample.py` to connect a character configuration, checkpoint,
  real dataset sample, model inference, greedy decoding, and constrained
  Viterbi alignment in one readable command.
- The runner prints sample metadata, the known label, greedy prediction,
  timeline sizes, alignment scores, and one model-frame interval per target
  character. An optional flag prints the complete frame-level token path.
- Completed an end-to-end CPU check using the Mac WD smoke checkpoint and a
  real processed validation sample. The run produced all six intervals for
  `gerade`; its positions are only a software check because the smoke model was
  trained for one epoch on 16 examples.
- The next meaningful check uses the trained WI/RH fold-0 `best_cer.pth` on the
  RTX machine.

## 2026-08-08 — Alignment diagnostics and visualization

- Verified real WI/RH fold-0 alignments for `gerade`, `Juni`, `immer`, and
  `Ich`. The repeated `mm` in `immer` was correctly separated by CTC blanks.
- The examples showed that a good whole-path score can hide an uncertain
  forced character, so the runner now reports character-level probability,
  preferred class, competing probability, margin, and local greedy agreement.
- Added approximate output-frame-to-model-input mapping using BLConv's 8x
  temporal reduction, including explicit reporting of trailing samples that
  do not form a complete output frame.
- Added candidate boundary regions from the CTC blank frames between adjacent
  character emission anchors. These are search regions, not claimed physical
  boundaries.
- Added synchronized PNG plots of normalized AF/AR/G magnitude, raw F values,
  and model probabilities. Green anchors agree locally, red anchors are forced,
  and orange spans are blank-frame regions.
- Added reusable JSON artifacts with sample metadata, paths, confidence
  diagnostics, approximate positions, and boundary regions.
- Eight focused tests now pass, including mapping, confidence, and PNG creation.
  An end-to-end Mac smoke run also produced a valid JSON/PNG pair; scientific
  inspection must use the trained RTX checkpoint.

## 2026-08-08 — Midpoint boundary feature extractor

- Manual plots showed why the complete CTC blank region is too broad: long
  regions can contain pen lifts or internal strokes unrelated to the actual
  adjacent-character boundary.
- Adopted the midpoint between adjacent character emission anchors as the
  initial boundary location and implemented 50, 100, and 150 ms local windows.
- Added unweighted raw-force, relative-force, low-force-duration, AF/AR/G
  magnitude, and motion-derivative features for every window.
- The provisional low-force threshold is 10% of each recording's raw-force
  90th percentile. It is saved with the features and remains subject to
  training-fold validation.
- Each boundary also stores both neighbouring character probabilities,
  confidence margins, their minima, and local greedy agreement. No final
  acceptance filter or continuity score has been hard-coded.
- The runner prints a compact 100 ms summary and writes all window sizes into
  the JSON artifact. Dashed black plot lines mark boundary midpoints.
- Thirteen focused tests pass, including safe handling of an alignment window
  that reaches model padding. The complete runner/JSON/PNG path succeeds with
  the local smoke checkpoint.

## 2026-08-08 — Training-split boundary exporter

- The inspected `gerade`, `immer`, and `Ich` examples confirmed that local
  force features distinguish obvious contact losses while long CTC blank
  regions alone do not. Unreliable forced anchors remain explicitly visible.
- Added `export_boundaries.py` to run the established alignment and feature
  pipeline across a selected fold and write one JSONL row per adjacent-character
  occurrence. It defaults to the training split so tokenizer evidence does not
  leak from validation data.
- Every record contains checkpoint/sample provenance, CTC measurements,
  alignment reliability, padding status, and nested 50/100/150 ms force and
  motion features. No continuity score or acceptance threshold is imposed.
- Added a progress sidecar and sample-safe `--resume` behavior. Existing output
  requires an explicit `--resume` or `--overwrite`, preventing accidental loss
  or duplication.
- The scientific inference default is batch size one because padding unequal
  words can affect bidirectional LSTM predictions. Larger batches remain an
  explicit exploratory option.
- All 15 focused tests pass. A real three-sample CPU smoke export produced 12
  boundary rows, and a second run resumed without duplication. A separate
  two-sample export was also extended to three samples using `--resume`.

## 2026-08-08 — Pair-level descriptive analysis

- The complete WI/RH fold-0 training export finished successfully: all 19,907
  requested samples completed, producing 88,292 boundaries across 426 distinct
  case-sensitive character pairs. Pair counts sum exactly to the boundary
  total, and the implied average word length is 5.435 characters.
- Added a streaming analysis tool that verifies provenance, window consistency,
  boundary-ID uniqueness, and agreement with the exporter summary before
  calculating statistics.
- Added global and per-pair counts, sample/writer coverage, alignment-quality
  rates, contact-preservation rates, and descriptive distributions for all
  alignment, force, and motion measurements at 50, 100, and 150 ms.
- Results are reported for all occurrences and for an explicitly named
  agreement/non-padding/unclipped subset. The subset intentionally uses no
  probability or confidence-margin threshold and is not presented as the final
  reliability definition.
- The analysis writes a JSON report, a compact pair overview CSV, and a full
  statistics CSV. No continuity score, ranking, or tokenizer merge is created
  yet.
- All 18 focused tests pass. The tool also completed against the real local
  smoke export, verified all 12 rows, and produced the three expected analysis
  files.

## 2026-08-09 — Boundary-position and case diagnostics

- Inspection of the full WI pair overview found a possible position confound:
  supported pairs beginning with uppercase characters preserved contact more
  often than supported lowercase pairs. Because uppercase characters normally
  occur at the start of words, pair identity alone cannot explain this result.
- Extended the existing analyzer to assign every occurrence one unambiguous
  position: `only` for the sole boundary of a two-character word, otherwise
  `first`, `middle`, or `final`.
- Added Unicode-aware left-character case groups and position-by-case groups.
  These are descriptive diagnostics and do not change the quality subset or
  create a continuity score.
- The analyzer now writes compact and full position-statistics CSV files in
  addition to the existing JSON and pair tables. Existing boundary exports can
  be reused; model inference and data re-export are not required.
- All 19 focused tests pass. The updated analyzer also completed on the real
  local smoke export and produced all five expected output files.

## 2026-08-09 — CTC boundary timing-bias diagnostic

- The full WI position report showed very different behaviour at word-initial
  boundaries: `only` and `first` boundaries had a median CTC blank duration of
  zero and preserved contact in about 92% of reliable occurrences, whereas
  `middle` and `final` boundaries had longer blank regions and lower contact
  preservation. This suggests that temporal localization, rather than only
  handwriting, may influence the boundary measurements.
- Added normalized boundary-centre timing and a simple evenly spaced reference,
  plus signed relative offset, absolute relative error, centre time, and signed
  millisecond offset. Negative signed offsets mean the estimate occurs earlier
  than the reference.
- The equal-spacing reference is documented as a rough sanity check, not a true
  physical character boundary. Existing JSONL exports contain all required
  fields, so no model inference or boundary re-export is needed.
- The compact pair and position overviews now include these timing diagnostics,
  and the CLI prints them beside the 100 ms position summary.
- All 20 focused alignment and boundary-analysis tests pass, and the analyzer
  completed successfully on the real local smoke export.

## 2026-08-09 — Unidirectional recurrent alignment experiment prepared

- Full WI results confirmed a strong word-initial localization bias. Reliable
  `only` and `first` boundaries both had a median centre time of 80 ms, median
  blank duration of zero, and median offsets of about -315 and -300 ms from the
  rough uniform references. The unfiltered and case-crossed summaries showed
  the same pattern, so neither the reliability filter nor uppercase letters
  explain it.
- Added a simple UniLSTM decoder and `unilstm_b`/`unilstm_s` factory keys. A
  focused test verifies that changing future decoder inputs cannot change its
  earlier outputs.
- Added `configs/thesis/a0_char_wi_rh_unidirectional.yaml`. It keeps all B0 WI
  settings fixed except the recurrent direction and writes to a separate
  alignment-model directory, so the completed B0 recognition baseline remains
  untouched.
- A0 removes whole-recording future recurrent context, but BLConv retains
  centred convolutions and sequence-wide instance normalization. It is not
  fully causal, so its timestamps require the same empirical position
  diagnostic and are not assumed to be ground truth.
- All 23 focused tests pass. A CPU smoke step also completed a forward pass,
  CTC loss, backward pass, and optimizer update with the new decoder.

## 2026-08-13 — A0 accepted and complete-region features implemented

- Completed A0 WI/RH fold-0 training. Its best validation CER was 17.33% at
  epoch 275 and its best WER was 32.77% at epoch 293. These are worse than B0,
  so A0 remains alignment-only rather than replacing the recognition baseline.
- Exported all 19,907 training samples: 88,292 boundaries across 426 pairs and
  42 writers. The agreement/non-padding/unclipped subset retained 78,953
  boundaries (89.4%).
- Reliable median timing offsets changed from B0's approximately -315/-300 ms
  for `only`/`first` boundaries to -65/-47 ms with A0. The severe word-initial
  concentration was therefore materially reduced without losing broad data or
  writer coverage.
- Added sensor features over the complete CTC candidate interval while keeping
  the existing 50/100/150 ms midpoint features. Each region stores original
  and actual duration, force/contact measurements, AF/AR/G magnitudes, motion
  derivative energy, edge clipping, and fallback usage.
- Empty candidate intervals now use a clearly marked 100 ms midpoint fallback;
  their original zero duration remains recorded.
- Extended the exporter schema and descriptive analyzer. New compact and full
  pair/position region tables are written separately so region measurements do
  not get mixed with fixed-window results.

## 2026-08-13 — Corrected provisional force-continuity ranking

- Complete-region comparison found contact preservation in 28.1% of 78,953
  reliable boundaries, versus 37.2% in the centred 100 ms view. The complete
  interval therefore finds additional force losses, but its duration is a
  serious confound: reliable first boundaries have a 480 ms median candidate
  region, compared with 240 ms for middle and final boundaries.
- Added `score_continuity.py` and `tva/continuity_scoring.py`. They create the
  first pair ranking without rerunning the recognition model.
- The scorer uses the established agreement/non-padding/unclipped evidence,
  subtracts expected contact rates for matching boundary-position and broad
  duration groups, and balances pair residuals equally across writers.
- The explicit development score uses 75% corrected 100 ms local contact and
  25% corrected complete-region contact. It reports raw and corrected
  components so the effect of the correction remains visible.
- The tentative support gate is 100 reliable occurrences and 30 writers.
  Frequency decides whether a pair is reliable enough to rank but does not
  increase its continuity score.
- Output includes a JSON method report and an auditable CSV with eligibility,
  writer variability, alignment confidence, duration, fallback use, case, and
  word-position proportions. The score is rejected for non-training splits.
- Four focused scorer tests cover correction/ranking, support gates, atomic
  output writing, and training-only enforcement. All 29 repository tests pass
  in the `tva-thesis` environment.

## 2026-09-15 — Approved proposal makes image connectivity primary

- Reconciled the repository documentation with the approved proposal,
  *Handwriting-Aware Tokenization for IMU-Based Online Handwriting Recognition*.
- The primary evidence pipeline now uses IAM and READ separately, pretrained
  DTLR character identities and boxes, transcript matching with uncertainty
  rejection, binarization, and connected-component labelling.
- Static connected ink is documented as a cross-modal handwriting prior for
  OnHW IMU recognition, not as proof of continuous IMU pen motion.
- Bigram is the required first tokenizer experiment. BPE and Unigram are
  conditional extensions if Bigram results are promising. Individual
  characters always remain fallback tokens.
- The core evaluation remains separate right-handed WD and WI OnHW-Words500
  recognition with fixed BLConv-B + BiLSTM-B + CTC models, matched linguistic
  baselines, fresh training per tokenizer, and CER/WER on reconstructed text.
- Added the proposal's explicit hypothesis that any benefit will be larger in
  WI than WD; this must be tested rather than assumed.
- Reclassified the completed CTC alignment and force-continuity work as a
  secondary/fallback comparison if the image method is unreliable and time
  remains. Historical implementation and experiment records were preserved.
- Optional image-domain recognition remains distinct from the primary use of
  image data to construct tokenizers.

### Immediate next steps

1. Prepare IAM and READ as separate evidence datasets.
2. Integrate the pretrained DTLR model and define auditable
   detection-to-transcription matching and rejection rules.
3. Pilot binarization, CCL, box-to-component association, and pair scoring on a
   small manually reviewed subset from each dataset.
4. Freeze the first connectivity definition and build matched handwriting- and
   frequency-based Bigram vocabularies before recognition training.
