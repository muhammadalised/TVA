# Thesis Progress Log

This file records completed work, important observations, and immediate next
steps. Add new entries chronologically. Do not remove failed attempts; they are
part of the research record.

For thesis traceability, record every material compatibility finding,
methodological decision, failed or negative result, data-quality issue,
experimental caveat, frozen parameter, and reproducibility identifier as the
work occurs. Put detailed evidence in a focused document when needed and link
it from this log and `docs/DECISIONS.md`.

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
- Eleven focused tests pass, including safe handling of an alignment window
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

## 2026-09-15 — Combined tokenizer compatibility audit

- Inspected TVA's tokenizer classes, OnHW label loader, fold-specific
  vocabulary construction, CTC blank convention, decoder, and output-head
  sizing before changing implementation code.
- Pinned the combined IAM+READ artifact
  `iam-read-combined-v1` at SHA-256
  `5c5d9f1689a4802fc5e9451e5afe8abdfb090562587ab78ddd343211feb94dd4`.
  It contains 494 classes: blank ID 0, 91 non-empty single characters, and 402
  handwriting bigrams.
- Submitted every OnHW training label in all five WD and all five WI folds to
  the unmodified combined tokenizer. Every successful encoding round-tripped
  to the NFC label, and NFC changed no OnHW label.
- Complete coverage failed because uppercase `Ä` and `Ü` are absent. In each
  complete 25,199-sample train-plus-validation partition, 608 samples (2.41%)
  contain one of these characters. Every WD and WI fold is affected.
- Confirmed that TVA's current Bigram tokenizer cannot be reused for this
  artifact: it applies greedy left-to-right matching, whereas the frozen model
  maximizes total handwriting utility with dynamic programming. The two
  segmentations differed for 9,009 of 24,591 otherwise encodable labels
  (36.64%).
- Confirmed that TVA correctly passes `tokenizer.size`, including blank, to the
  training output head. Found a separate reporting issue: `evaluate.py`
  hard-codes 500 classes for complexity calculation.
- Designed a leakage-safe adapter option using only the fixed OnHW alphabet:
  blank + all 59 OnHW characters + 359 frozen in-alphabet bigrams, for 419
  classes. A 496-class full-artifact-plus-fallback form remains a possible
  sensitivity condition. The primary policy must be frozen before training.
- Recorded the complete evidence, per-fold failure counts, methodological
  implications, and required conformance tests in
  `docs/TOKENIZER_COMPATIBILITY_AUDIT.md`.

### Immediate next steps

1. Implement a dedicated handwriting-bigram tokenizer that exactly preserves
   the DTLR NFC and dynamic-programming behavior.
2. Implement and checksum the now-frozen deterministic 419-class OnHW
   compatibility adapter.
3. Add complete coverage, round-trip, blank, ID, provenance, and DTLR-reference
   conformance tests before any recognition training.
4. Build matched fold-specific linguistic Bigram vocabularies from OnHW
   training labels only.

## 2026-09-17 — Primary handwriting adapter policy frozen

- Froze policy `onhw-words500-rh-iam-read-v1` at 419 output classes: blank ID
  0, the exact pre-existing 59-character OnHW alphabet at IDs 1–59, and the
  359 source bigrams whose two characters are in that alphabet at IDs 60–418.
- The construction preserves source-vocabulary order, utilities, NFC, and the
  DTLR maximum-total-utility dynamic program. It adds `Ä` and `Ü` only as
  fallback singles and derives no new bigrams from OnHW labels.
- Excluded the 496-class full-artifact extension from the primary experiment.
  It may be used only as a separately named, predeclared sensitivity analysis.
- Validated an in-memory reference construction against every train and
  validation label in all five WD and all five WI folds. All 20 fold/split
  checks passed coverage, non-empty target, no-blank target, ID-range, and NFC
  encode/decode round-trip invariants.
- This was a compatibility validation, not vocabulary selection: construction
  used the fixed task alphabet and frozen source only, without OnHW label
  frequencies, validation membership, or recognition results.
- Recorded a comparator-design caveat: a size-matched linguistic vocabulary
  does not isolate token membership if it uses greedy segmentation while the
  handwriting tokenizer uses utility-maximizing dynamic programming. This
  segmentation-policy choice must be predeclared before comparative training.
- Full policy: `docs/HANDWRITING_BIGRAM_ADAPTER_V1.md`.

### Immediate next steps

1. Implement the deterministic adapter builder and canonical artifact, then
   record the derived artifact SHA-256.
2. Implement the dedicated tokenizer and DTLR-reference conformance tests.
3. Add complete coverage, round-trip, blank, ID, and provenance tests.
4. Resolve and predeclare the linguistic comparator's segmentation policy.
5. Correct `evaluate.py`'s hard-coded 500-class complexity head before model
   size or MAC reporting.

## 2026-09-17 — Canonical 419-class adapter implemented

- Added a deterministic builder and CLI in
  `tva/handwriting_bigram_adapter.py` and
  `build_handwriting_bigram_adapter.py`.
- The builder authenticates the source checksum and audited schema/model
  invariants before projection. It rejects changed provenance, non-contiguous
  mappings, invalid blank conventions, altered DP policy, and unexpected
  source or output counts. It also checks the completed canonical bytes against
  the frozen adapter checksum, preventing silent regeneration drift under the
  same policy ID.
- Generated the canonical artifact at
  `artifacts/tokenizers/onhw_words500_rh_iam_read_v1.json` with SHA-256
  `12ce25d8bedc552e6b3497ffb1d07e506b01b34296cc21b970f82a550cbf2bfe`.
- Confirmed that rebuilding from the pinned source produces byte-identical
  canonical JSON. The artifact records that no annotation files were read.
- Ran the canonical artifact through the independent DTLR reference tokenizer
  on every WD/WI train and validation label. All 251,990 fold/split label
  instances across the 20 checks passed coverage, non-empty target, no-blank
  target, ID-range, and NFC round-trip assertions.
- Added six unit/integration tests covering alphabet-only projection, source
  order and metadata preservation, duplicate-alphabet rejection, source hash
  rejection, canonical UTF-8 serialization, frozen mapping invariants, and a
  byte-identical rebuild when the external source is available.

### Immediate next steps

1. Implement TVA's dedicated handwriting-bigram tokenizer using the artifact's
   NFC and maximum-total-utility DP policy.
2. Add DTLR-reference segmentation conformance and complete OnHW label tests
   for that runtime tokenizer.
3. Resolve and predeclare the matched linguistic comparator's segmentation
   policy.
4. Correct `evaluate.py`'s hard-coded 500-class complexity head.

## 2026-09-18 — TVA handwriting-bigram runtime implemented

- Added `HandwritingBigramTokenizer` in
  `tva/handwriting_bigram_tokenizer.py` and exposed it through
  `get_tokenizer('handwriting_bigram')`.
- Updated tokenizer-path resolution so the canonical shared artifact can be
  supplied directly through the existing `dir_tokenizer` configuration field;
  legacy per-fold tokenizer directories retain their previous behavior. The
  shared resolver is used consistently by training, evaluation, and token
  counting.
- The training load path authenticates the entire canonical adapter SHA-256,
  then validates schema, inverse and contiguous mappings, blank ID 0,
  normalization, overlap policy, finite utility values, output size, bigram
  count, source checksum, policy ID, and the leakage-safe construction flag.
- Reimplemented the DTLR right-to-left maximum-total-utility dynamic program,
  including its preference for more bigrams and its deterministic pair choice
  when utility and bigram count tie. NFC normalization occurs before
  segmentation.
- Confirmed exact TVA-versus-DTLR equality of both segmentation dictionaries
  and encoded IDs for all 501 unique OnHW words, including the previously
  documented overlap cases.
- Confirmed complete runtime coverage on all 251,990 WD/WI fold/split label
  instances: targets were non-empty, used IDs 1–418 only, and decoded to the
  NFC-normalized label.
- Removed `evaluate.py`'s hard-coded 500-class complexity head. Training and
  complexity reporting now both size the output head from `tokenizer.size`, so
  the frozen condition uses exactly 419 logits including blank.
- The complete suite now has 16 passing tests: six adapter tests and ten
  runtime/integration tests. No recognition training was run.

### Immediate next steps

1. Resolve and predeclare the matched linguistic comparator's segmentation
   policy before constructing its fold-specific vocabularies.
2. Add a thesis experiment configuration using tokenizer key
   `handwriting_bigram` and the canonical shared artifact, then run a small
   pipeline smoke test before full training.

## 2026-09-18 — Matched linguistic Bigram baseline frozen and generated

- Froze D016: the linguistic baseline uses the same NFC and
  maximum-total-utility dynamic program as the handwriting condition. Its pair
  membership and utilities come only from fold-specific OnHW training text.
- Preserved TVA's existing distinct-word convention so repeated writers and
  samples do not dominate language frequency. Pairs are ranked by descending
  adjacency count and lexical tie-breaking; utilities are counts divided by
  the fold maximum.
- Generated ten 419-class artifacts: five WD/RH and five WI/RH, each containing
  blank ID 0, the same 59 characters, and 359 training-derived bigrams. Full
  checksums are recorded in per-dataset manifests and
  `docs/LINGUISTIC_BIGRAM_BASELINE_V1.md`.
- Runtime loading verifies the pinned dataset-manifest checksum and its
  per-fold artifact checksum before accepting a linguistic tokenizer.
- Confirmed the builder accepts only `train.json`; no validation annotation or
  recognition result participates in vocabulary construction.
- Found an important baseline limitation before training: every fold reaches a
  frequency cutoff of one. Between 87 and 97 selected pairs per fold come from
  a larger 147–155-pair count-one tie, resolved deterministically by token.
- The linguistic and handwriting vocabularies share 223–225 of 359 pairs.
  Their DP segmentations differ on 52.28%–53.29% of complete fold samples, and
  their mean target lengths are approximately 3.221 and 3.321 tokens,
  respectively. These are composition measurements, not recognition results.
- Every linguistic artifact passed train/validation coverage, no-blank target,
  ID-range, and round-trip checks over 251,990 fold/split label instances.
- The suite now has 22 passing tests. No recognition training was run.
- Predeclared a possible greedy-for-both fallback as a secondary segmentation
  ablation if DP training is unsuccessful. It must retain both DP results and
  cannot be promoted post hoc based on validation performance.

### Immediate next steps

1. Add matched fold-0 WD and WI experiment configurations for handwriting and
   linguistic Bigram conditions.
2. Run bounded pipeline smoke tests in the `tva` environment.
3. Verify saved model heads contain 419 outputs and preserve smoke-test logs.
4. Only after smoke tests pass, begin the predeclared fold-0 development runs.

## 2026-09-18 — Matched fold-0 Bigram configurations and smoke tests

- Added four fold-0 experiment configurations under `configs/thesis/`: the
  handwriting-aware and matched linguistic Bigram conditions for WD/RH and
  WI/RH. Architecture, augmentation, optimizer schedule, epoch budget, batch
  size, seed, and evaluation settings match across all four conditions.
- Loaded each frozen tokenizer through its production configuration and
  confirmed `tokenizer.size == 419` and a 419-output model head.
- Ran a bounded end-to-end CPU smoke test for every condition with 16 training
  and eight validation samples, one epoch, batch size two, and seed 42. All
  four completed data loading, augmentation, encoding, CTC loss, backward
  updates, validation decoding, evaluation, and checkpoint saving.
- Inspected each saved `latest.pth`: `decoder.fc.weight` has shape `(419, 256)`
  and `decoder.fc.bias` has shape `(419,)` in all four runs.
- Smoke logs, resolved configuration snapshots, metrics, predictions,
  visualizations, and checkpoints are preserved under
  `results/thesis/development/smoke_bigram_*/0/`. These bounded CPU results
  (CER/WER both 1.0 after one tiny epoch) are software checks, not thesis
  recognition results.
- The initial sandboxed process could not see CUDA, but a permitted local run
  detected the NVIDIA GeForce RTX 4060 Laptop GPU and completed the same
  bounded handwriting-WD smoke test with CUDA mixed precision enabled.
- Added regression tests that keep the four non-tokenizer configurations
  matched and require every configured tokenizer and output head to have 419
  classes. The complete suite passes 25 tests.
- A manual full-data, one-epoch timing run of the handwriting-aware WD/RH
  fold-0 configuration on the RTX 4060 measured 29 seconds for training and
  four seconds for validation. This implies approximately 2 hours 45 minutes
  per 300-epoch fold and about 11 hours for the four fold-0 Bigram development
  runs when executed sequentially. It is a runtime measurement, not a
  recognition result.

### Immediate next steps

1. Start the four predeclared fold-0 development runs from
   scratch; do not initialize them from character or smoke checkpoints.
2. Compare CER/WER only after all matched fold-0 runs finish, preserving each
   configuration, log, checkpoint, prediction file, and commit identifier.

## 2026-09-18 — Handwriting-aware tokenizer notebook

- Added `train_handwriting_tokenizer.ipynb` as the handwriting-aware
  counterpart to TVA's baseline `train_tokenizers.ipynb`.
- The notebook explicitly distinguishes external IAM+READ tokenizer training
  from the deterministic OnHW compatibility projection: it does not train or
  select handwriting bigrams from OnHW labels.
- It authenticates the pinned DTLR source checksum, reproduces the frozen
  419-class adapter in memory, verifies byte identity with the canonical
  artifact, demonstrates production DP segmentation, and runs the complete
  post-construction WD/WI label compatibility audit.
- Canonical writes are disabled by default. Enabling the write flag can only
  emit the already-frozen artifact because the builder rejects any unexpected
  source checksum, schema, class count, bigram count, or final artifact hash.
