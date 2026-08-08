# Thesis Progress Log

This file records completed work, important observations, and immediate next
steps. Add new entries chronologically. Do not remove failed attempts; they are
part of the research record.

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

## 2026-08-02 — Tokenizer-family scope clarification

- Clarified that the thesis plans handwriting-aware Bigram, BPE, and Unigram
  tokenizers, not only one motion-selected pair vocabulary.
- Each proposed variant will be compared with its linguistic counterpart at a
  matched vocabulary size; the character baseline remains the common reference.
- Motion continuity is the primary token-selection evidence. Frequency,
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
- Eleven focused tests pass, including safe handling of an alignment window
  that reaches model padding. The complete runner/JSON/PNG path succeeds with
  the local smoke checkpoint.
