# CTC Forced Alignment

> **Thesis-scope note (September 2026):** This is a completed secondary and
> fallback pipeline. The approved proposal uses DTLR-assisted connected ink in
> IAM and READ images as the primary handwriting prior. Use this IMU alignment
> work as an additional comparison only if the image method is unreliable and
> time remains. CTC alignments are not ground-truth physical boundaries.

## Purpose

The character model predicts a probability for every character at every
model-output frame. During training we also know the correct word label. CTC
Viterbi alignment finds the highest-probability path through those predictions
that is allowed to produce the known label.

For the label `Haus`, the algorithm searches through this expanded sequence:

```text
blank, H, blank, a, blank, u, blank, s, blank
```

At every frame, a path may stay at its current state, advance by one state, or
skip a blank when CTC permits it. The implementation stores the best score and
previous state for every frame/state combination, then follows those saved
choices backwards to recover the alignment.

Repeated characters need special handling. For example, the two `l` characters
in `Hallo` must have a blank between them; otherwise standard CTC collapsing
would turn them into one `l`.

## Implementation

The standalone implementation is in `tva/ctc_alignment.py`. Its main function
accepts one sample at a time:

```python
from tva.ctc_alignment import ctc_viterbi_align

# probabilities: (number of model frames, number of character classes)
# target_ids: known character IDs without CTC blanks
alignment = ctc_viterbi_align(probabilities, target_ids, blank_id=0)

for token in alignment.tokens:
    print(
        token.target_index,
        token.token_id,
        token.start_frame,
        token.end_frame,
    )
```

Intervals use normal Python half-open indexing: `[start_frame, end_frame)`.
Thus an interval `[3, 6)` contains frames 3, 4, and 5.

The returned object also exposes:

- `log_score`: total log-probability of the selected path;
- `expanded_target`: target IDs with CTC blanks inserted;
- `state_path`: expanded-target state selected at every frame; and
- `token_path`: corresponding token ID selected at every frame.

## Single-sample runner

The [forced-alignment walkthrough](../forced_alignment_demo.ipynb) starts
with one `Programm` recording and follows the project from CTC timing and
force export through the pair ranking, timing audits, Bigram v0 selection
and recognition pilot. Its executed plots and tables include the seven
evidence rows for the example and the final three-way CER/WER comparison.

`tva/forced_alignment/align_sample.py` connects the alignment algorithm to a trained TVA character
model and a real dataset sample. It deliberately processes only one sample so
checkpoint loading, preprocessing, model inference, greedy decoding, and
forced alignment remain easy to inspect.

Run WI/RH fold 0 with its best-CER checkpoint:

```bash
python -m tva.forced_alignment.align_sample \
  --config configs/thesis/b0_char_wi_rh.yaml \
  --checkpoint results/thesis/baselines/B0_char_wi_rh/0/checkpoints/best_cer.pth \
  --split val \
  --sample-index 0 \
  --device cuda
```

Run WD/RH fold 0 with the retained final checkpoint:

```bash
python -m tva.forced_alignment.align_sample \
  --config configs/thesis/b0_char_wd_rh.yaml \
  --checkpoint results/thesis/baselines/B0_char_wd_rh/0/checkpoints/latest.pth \
  --split val \
  --sample-index 0 \
  --device cuda
```

Add `--show-path` when the complete frame-by-frame CTC path is useful for
debugging. Without it, the runner prints compact character and boundary tables.
Every run also writes a reusable JSON record and PNG plot under
`results/thesis/alignment_debug/` by default.

## Confidence and approximate IMU mapping

For each target character, the analysis records its mean model probability,
the locally preferred class, the best competing probability, their margin, and
whether the target agrees with the local greedy choice. These values are useful
diagnostics, but they are not yet treated as calibrated confidence or used as a
final exclusion threshold.

BLConv reduces the time dimension by a factor of eight. The tool maps each
model-frame edge to an approximate model-input sample using this ratio and
reports any trailing samples that were not represented by a complete output
frame. These positions are called character emission anchors, not exact
physical stroke start/end times, because the convolutional and bidirectional
layers use neighbouring context.

The blank frames between adjacent character anchors form a candidate boundary
region. A zero-width region means the CTC path moved directly from one
character to the next; it does not by itself prove that the physical motion was
continuous.

## Observed word-initial timing bias

The full WI/RH fold-0 training analysis showed that the bidirectional B0 model
usually placed both `only` and `first` boundaries at 80 ms. In the reliable
subset, their median offsets from the rough uniform reference were -315 ms and
-300 ms, and their median intervening CTC blank duration was zero. The same
pattern appeared without the reliability filter and for both uppercase and
lowercase word starts.

This is consistent with the BiLSTM using later parts of the recording to emit
the first characters early. Alignment probability does not solve the problem:
the median first-boundary probability was approximately 0.999. Therefore the
current word-initial sensor windows must not be treated as physical continuity
measurements.

The A0 experiment keeps BLConv-B but replaces BiLSTM-B with a unidirectional
LSTM. On the full WI/RH fold-0 training export, it reduced reliable median
`only` and `first` offsets from about -315/-300 ms to -65/-47 ms and retained
78,953 reliable boundaries across all 42 writers. A0 is therefore accepted as
the fold-0 alignment model. It does not replace B0 for recognition because its
validation CER and WER are worse. BLConv still uses centred convolutions and
sequence-wide instance normalization, so A0 is not fully causal end to end and
its timestamps remain estimates rather than ground truth.

## Visualization and JSON

The PNG contains three synchronized timelines:

- normalized magnitudes for the AF, AR, and G sensor groups;
- the raw F channel so force/contact structure remains visible; and
- the model's highest class probability plus each forced character probability.

Green character anchors agree with the local model choice. Red anchors were
inserted by the target constraint despite another locally preferred class.
Orange regions contain intervening CTC blank frames. The dataset metadata order
is `AF(0-2), AR(3-5), G(6-8), M(9-11), F(12)`.

The JSON contains sample/checkpoint metadata, complete CTC paths, character
diagnostics, approximate input positions, and candidate boundary regions. Raw
sensor arrays remain in the dataset and are not duplicated in the JSON.

## Boundary feature extraction

The feature extractor keeps two complementary measurements. It measures
separate 50, 100, and 150 ms windows around the midpoint between adjacent
character emission anchors, and it measures the complete intervening CTC
candidate region. Local windows avoid assigning every event in a long blank
region to the boundary; the complete region avoids missing a force drop near a
region edge. Their difference is an explicit development comparison.

When two characters are emitted in consecutive CTC frames, the candidate
region has zero samples. The extractor uses a 100 ms midpoint-centred fallback
in that case and saves `used_fallback_window=true`. The original zero duration
is also retained, so fallback evidence cannot be mistaken for a true non-empty
region.

Each boundary stores alignment reliability from both neighbouring anchors:

- aligned probabilities and confidence margins;
- the minimum of the two values; and
- whether both anchors agree with their local greedy classes.

Each window stores transparent sensor measurements:

- raw force minimum, mean, and centre value;
- force values relative to the recording's 90th-percentile reference;
- force-drop ratio;
- fraction and longest duration below a provisional low-force threshold;
- mean AF, AR, and G magnitudes; and
- derivative energy across the nine AF/AR/G axes.

The complete candidate-region record stores the same sensor measurements plus
the original and actual interval sizes/durations, whether a fallback was used,
and whether clipping at the recording edge occurred.

The provisional low-force threshold is 10% of the recording-level force
reference. It is saved in every JSON file so the calculation is reproducible.
This value, the window size, the alignment-quality filter, and the eventual
continuity score remain development choices; the extractor does not yet accept,
reject, or rank a boundary.

If model padding places an anchor outside the real sensor recording, the local
window is clipped to valid raw samples and the boundary is explicitly marked as
overlapping padding. Such cases can later be excluded rather than being
mistaken for real zero-force data.

## Tests

Run the focused tests from the repository root:

```bash
python -m unittest discover -s tests -v
```

The tests cover ordinary and repeated characters, target-constrained decisions,
invalid targets, insufficient frames, confidence diagnostics, IMU mapping,
candidate boundary regions, force/motion window features, JSONL records,
interrupted-export recovery, Unicode labels, and PNG creation.

## Training-split boundary export

`tva/forced_alignment/export_boundaries.py` applies the same alignment and feature extraction to a
complete dataset split. The default split is `train`, because tokenizer
statistics must be learned without looking at validation recordings. It writes
one JSONL record per adjacent-character occurrence, plus a compact summary and
a progress file.

Run the selected A0 WI/RH training-fold export on the RTX machine:

```bash
python -m tva.forced_alignment.export_boundaries \
  --config configs/thesis/a0_char_wi_rh_unidirectional.yaml \
  --checkpoint results/thesis/alignment_models/A0_char_wi_rh_unidirectional/0/checkpoints/best_cer.pth \
  --device cuda \
  --overwrite
```

The default output is:

```text
results/thesis/boundary_exports/a0_char_wi_rh_unidirectional/fold0/train/boundaries.jsonl
```

The first export after adding complete-region features must use `--overwrite`.
The older JSONL rows do not contain those measurements, so `--resume` would
correctly skip them rather than upgrade them. Later interruptions of the new
export can use `--resume` normally.

Use `--max-samples 100 --overwrite` for a small development export. Use
`--resume` after an interruption. Resume is sample-safe: the progress marker is
written only after every boundary of that sample has been flushed, and any
unfinished rows are removed before continuing.

Inference defaults to batch size one. This avoids allowing zero-padding from
other, longer words to influence the bidirectional LSTM output. A larger
`--batch-size` is available for exploratory speed tests, but the final export
should retain the reproducible default unless equivalence is demonstrated.

Each boundary row contains sample and checkpoint provenance, the character
pair, CTC blank information, alignment reliability, padding status, the
recording-level force reference, and nested 50/100/150 ms sensor features. No
continuity class or score is added at this stage.

## Pair-level descriptive analysis

`tva/forced_alignment/analyze_boundaries.py` reads the completed JSONL export and calculates exact
descriptive statistics globally and separately for every character pair. It is
a CPU analysis and does not run the recognition model again:

```bash
python -m tva.forced_alignment.analyze_boundaries \
  --input results/thesis/boundary_exports/a0_char_wi_rh_unidirectional/fold0/train/boundaries.jsonl \
  --overwrite
```

The default output directory is `pair_analysis/` beside the input file. It
contains:

- `analysis_summary.json`: provenance, data-integrity totals, pair-support
  distribution, and global statistics for every window;
- `pair_overview.csv`: a compact table of the most interpretable pair
  measurements; and
- `pair_statistics.csv`: the full means, standard deviations, ranges, and
  10th/25th/50th/75th/90th percentiles used for later scoring and ablations.
- `position_overview.csv`: compact position, case, and position-by-case
  diagnostics; and
- `position_statistics.csv`: the corresponding full descriptive statistics.
- `pair_region_overview.csv` and `pair_region_statistics.csv`: compact and
  full pair summaries over complete candidate regions; and
- `position_region_overview.csv` and `position_region_statistics.csv`: the
  corresponding complete-region position/case diagnostics.

Every pair/window is reported twice. The `all` subset contains every exported
occurrence. The `agreement_nonpadding_unclipped` subset includes an occurrence
only when both forced character anchors agree with the local model choice,
neither anchor overlaps padding, and that sensor window is not clipped at the
recording edge. This is a transparent quality subset, not the final alignment
filter: it applies no probability or confidence-margin threshold.

The `no_low_force_rate` is the fraction of occurrences whose complete local
window remained above the provisional recording-relative force threshold. It
is descriptive contact evidence and is not yet a continuity label.

Position groups are mutually exclusive. `only` means a two-character word has
exactly one boundary, which is simultaneously its first and final boundary.
Longer words use `first`, `middle`, and `final`. The report also separates the
case of the character to the left of the boundary and crosses case with
position. Unicode-aware case checks support German uppercase characters.

The position report also checks for systematic timing bias. It expresses each
estimated boundary centre as a fraction of the recording and compares it with
the simple evenly spaced reference
`(boundary_index + 1) / (number_of_boundaries + 1)`. The signed offset is
negative when the CTC estimate is earlier than this reference, and the absolute
error ignores direction. The report also gives the signed offset in
milliseconds. This equal-spacing reference is only a sanity check: real
characters do not take equal time, so it is not physical boundary ground truth
and must not be used as a training label by itself.

## Provisional corrected force-continuity ranking

`tva/forced_alignment/score_continuity.py` reads the completed training JSONL directly. It does not
run the recognizer or require a GPU:

```bash
python -m tva.forced_alignment.score_continuity \
  --input results/thesis/boundary_exports/a0_char_wi_rh_unidirectional/fold0/train_region_v2/boundaries.jsonl \
  --overwrite
```

The default `continuity_scores/` directory beside the JSONL contains:

- `continuity_score_summary.json`: exact method parameters, global and
  position-duration reference rates, data counts, and the top eligible pairs;
  and
- `pair_continuity_scores.csv`: all pairs, their eligibility reason, raw and
  corrected local/region contact evidence, provisional force score,
  writer-to-writer variability, support, alignment confidence, region duration,
  fallback rate, case rate, and boundary-position proportions.

The score is learned only from records whose split is `train`. Its transparent
quality filter requires greedy agreement for both anchors, no padding overlap,
and unclipped local and region measurements. Expected contact rates are formed
within boundary-position and broad duration groups. Pair residuals are first
averaged within writer and then across writers. The defaults rank only pairs
with at least 100 reliable occurrences and 30 writers.

The provisional score is 75% corrected 100 ms local contact and 25% corrected
complete-region contact. It is a force-only development baseline for a
possible secondary IMU-informed Bigram experiment, not the primary proposed
image-informed score or a final combined force/motion score.

## Historical next steps (superseded by the 2026-11-15 plan)

1. Run the corrected force scorer on the complete A0 WI training export and
   inspect its top, middle, and bottom supported pairs.
2. Check ranking sensitivity to local/region weights, duration bins, and the
   support gate without using validation or test recognition results.
3. Define a motion-only component from accelerometer and gyroscope evidence.
4. Compare force-only, motion-only, and combined scores before constructing a
   secondary IMU-informed Bigram vocabulary.

## Method review — 2026-11-15 (preserved source)

The review below is copied from `origin/forced-alignment-onhw` at
`e1bc2a7771a3c787f912c3f58df4b51d862cb4b0`. Its D027 reference belongs to
that branch. This original branch does not contain that decision entry.
The source review remains a proposal. The follow-up plan below records the
current authorization and qualifies claims that have not been validated.

- Status: **proposed, not approved.** Claude wrote this review on the Mac.
  Ask the user to approve each change before you implement it (AGENTS.md,
  "Get approval").
- Scope: secondary experiment D027, OnHW right-handed, WI fold 0 first.
- Evidence: the counts below come from
  `data/tva/onhw_words500_wi_word_rh/train.json`, fold 0, training split
  only. Nothing here uses the A0 export, because it is on the RTX laptop.

### Verdict

The approach is good enough for a secondary experiment. The engineering is
sound: training split only, quality subset, provenance, tests, the A0 fix for
early timing, and corrections for position, gap length, and writer.

Force is the right signal. The OnHW pen measures tip force, so "pen stays
down between two letters" is the sensor version of "ink is connected" in an
image. This makes the comparison with the DTLR image evidence clean.

The weak point: **nothing yet shows that the pair score measures real pen
connectivity.** Validate before you build any tokenizer.

### Problems, most serious first

1. **Each pair score comes from very few words.**
   - The fold-0 WI training split has 19,907 recordings but only 501
     different words. Each word is written about 40 times.
   - 192 of 426 pairs pass the 100-occurrence / 30-writer gate. Of these,
     78 (41%) come from only 3–4 different words (48 from 3, 30 from 4).
   - The 100-occurrence gate therefore means only "at least 3 words".
   - Effect: a pair's score partly shows how people write those few words,
     not the pair in general.
   - Proposed fix: add a minimum number of distinct words (for example 5),
     and average residuals per word as well as per writer. Report word
     variability next to writer variability.
2. **Coarse timing.**
   - The encoder has three stride-2 convolutions (8× downsampling). At
     100 Hz, one CTC frame is 80 ms.
   - The 50 ms window is smaller than one frame, so it has no real meaning.
     Every anchor is uncertain by at least ±40 ms.
   - A0 still places word-start boundaries about 50–65 ms early.
   - Proposed fix: drop the 50 ms window. Detect pen lifts as events (force
     below the threshold for at least about 30 ms) at 10 ms resolution.
     Count lift events inside the whole gap between the two character
     anchors, instead of using a fixed window around the midpoint.
3. **Delayed strokes.**
   - i/j dots, t/f/x crosses, and umlaut dots (ä, ö, ü, Ä, Ö, Ü) cause pen
     lifts that are not real breaks between letters.
   - 37 of the 192 gated pairs have such a letter on the left. 30 have an
     uppercase letter on the left, which is mostly word-initial.
   - The current correction handles position and gap length, but not the
     letter.
   - Proposed fix: add a per-letter expected lift rate to the correction, or
     at least report these pairs as a separate group.
4. **Drop motion for now.**
   - D005 says motion is primary. But force measures contact directly.
     Motion is indirect, noisier, and adds more settings to tune.
   - Proposed fix: make force the primary evidence for D027. Keep motion as a
     later ablation, or drop it.

### Proposed validation before building tokenizers

1. **Manual audit:** inspect about 100 boundaries with the existing PNG plots
   (`tva/forced_alignment/align_sample.py`). Pick pairs from the top, middle, and bottom of the
   ranking. Record whether the pen is really down or up at each boundary.
   This gives a defensible accuracy number for the thesis.
2. **Word-level consistency check:** for each word, compare the pen-lift
   count measured from force with the count that the pair scores predict.
   Measuring lifts needs no alignment model. If the two do not agree, the
   alignment step is the problem.
3. **Cross-check with image evidence:** compare the force ranking with the
   IAM image connectivity (`dominant-core-v3`) for the lowercase pairs that
   both have. Expect only partial agreement: German and English school
   scripts join letters differently.

An alignment-free regression (pen lifts per word fitted from letter and pair
terms) was considered and rejected as the main method: 501 words give too
few independent equations for about 485 unknowns. It is useful only as the
consistency check in item 2.

### Proposed order for the RTX laptop

1. Run `tva/forced_alignment/score_continuity.py` on the existing A0 WI fold-0 export as it is,
   and save the result as the baseline ranking.
2. After approval, add the distinct-word gate and the per-word balancing,
   then rerun. Report how the ranking changes.
3. Run the validation checks above.
4. Only then propose the open D027 settings (construction corpus, vocabulary
   size, number of allowed pairs) for approval.

## Step-by-step validation plan — 2026-11-15

### Purpose and authorization

Find out whether the existing A0 force ranking gives useful evidence about
letter-pair contact. Use right-handed OnHW-Words500, WI fold 0, training
recordings only. This is a development diagnostic, not a recognition result.

The user approved planning and maintaining this document after the review.
The immediate work preserves and reproduces the existing baseline and records
word support. The user then explicitly approved Step 2, which is now complete.
The user approved Step 3 on 2026-11-16; its provisional AI inspection is
complete. Steps 4-6 remain planned. No approval here freezes a new score,
threshold, support gate, or tokenizer design.
Keep the current `forced-alignment` branch and the saved documentation stash.
Do not merge branch-specific decision numbers during this work.

Keep the existing scorer, A0 checkpoint, split, force threshold (10% of the
recording force reference), 100 ms local window, 75%/25% local/region weights,
and 100-occurrence/30-writer gate unchanged for the baseline. Save new outputs
in a separate directory. Do not overwrite the August outputs.

### Step 1 — Preserve and reproduce the baseline

- Status: completed on 2026-11-15; see the execution record below.
- Verify the export metadata, input hashes, installed interpreter and code
  revision. Save the resolved scoring settings and existing Git changes.
- Run the unchanged `tva/forced_alignment/score_continuity.py` on the existing training export.
- Compare the saved ranking and method report with the August outputs.
  Check every score and eligibility flag, not only the top pairs.
- Independently check boundary IDs, training annotation labels, writers and
  pair positions. Count word types before and after the scorer quality filter.
- Save a run manifest and word-support table beside the reproduced ranking.
- Completion condition: all comparisons reconcile, or each difference is
  explained. Record the result below before proceeding.

### Step 2 — Check stability across words and writers

- Status: completed on 2026-11-15. The user explicitly approved Step 2.
  See the execution record for results and limits.
- Use the same accepted boundaries and unchanged score. Remove one word type
  at a time and recompute all pair scores. Repeat for one writer at
  a time. Recompute the correction baselines consistently for each omission.
- Report score changes, rank changes, support losses and the words/writers
  responsible for large changes. Compare ranks on a fixed common pair set;
  report eligibility changes separately.
- Report uncertainty from resampling whole words or writers if needed;
  repeated recordings must not be treated as independent word contexts.
- Do not choose a new word gate or balancing rule from recognition scores.
  A five-word gate is only a proposal; report its coverage cost first.
- Completion condition: identify which rankings depend strongly on a single
  word or writer. Do not invent a pass threshold after viewing the results.

#### Step 2 procedure fixed before execution

- Authorization: the user asked to check ranking stability across words and
  writers and keep this document updated. This does not change the score.
- Population: use the same 78,953 accepted training boundaries. Omit all
  accepted observations of each of the 501 training word types in turn.
  Separately omit each of the 42 training writers. No random sampling.
- Keep the A0 model and existing alignment export fixed. These are sensitivity
  checks conditional on that model, not unseen-word or unseen-writer model
  evaluations. Do not retrain, realign, or read validation recordings.
- For every omission, recompute position-duration, position and global contact
  baselines. Preserve fallback rules, per-writer averaging, clipping, weights
  and support gates exactly. Recompute all pair scores.
- Primary rank comparison: the fixed set of 176 baseline-eligible pairs.
  Keep its members for the rank diagnostic even if an omission makes one
  ineligible. Report those eligibility losses separately.
- Report Spearman correlation using average ranks for exact score ties.
  Displayed integer ranks use alphabetical tie-breaking, as in the scorer.
  Report top-20 and top-50 overlap divided by 20 or 50 on that fixed set.
  These are descriptive cutoffs, not selected vocabulary sizes.
- Also compare baseline and omitted rankings on their common eligible set,
  and report the intersection that remains eligible in every omission.
- Save every omitted word/writer ID, all retained pair scores and support,
  per-omission summary, and per-pair worst changes with the responsible ID.
  Report score differences on the 0-1 score scale, integer rank shifts and
  gate failures. Keep word and writer results separate.
- Numerical validation: compare the array calculation with the unchanged
  existing scorer, first without an omission, then after removing the
  alphabetically first, middle and last word and the most frequent accepted
  word. Check first, middle, last and most frequent accepted writers too.
  Collapse duplicate validation cases. Selection uses IDs/counts, not scores.
  Require score agreement within 1e-12, matching support, eligibility and
  displayed ranks. These checks must pass before accepting results.
- No post-hoc stability pass/fail cutoff or new word gate. Exhaustive
  leave-one-group changes are sensitivity results, not confidence intervals.
- Output:
  results/thesis/forced_alignment_review/wi_rh_fold0_20261115_stability/.
  Save the analysis source, resolved procedure, input hashes, code revision,
  working-tree diff and package versions with the results. Preserve Step 1,
  the August ranking, the checkpoint and the input export.

### Step 3 — Inspect a fixed sample of sensor traces

- Status: provisional Codex AI inspection completed on 2026-11-16.
  Independent human and physical-timing validation remain open.
- Select about 100 boundaries across high, middle and low eligible ranks.
  Include only/first/middle/final positions, short/long regions, fallback cases,
  uppercase letters and letters with possible delayed strokes.
- Save the seed, exact boundary IDs, sampling groups and input hashes first.
  Do not replace difficult cases after seeing the traces.
- Inspect raw force and surrounding context, with alignment anchors visible.
  Record sustained low-force events, short dips, uncertain timing and possible
  delayed strokes separately. Hide the pair score during review where possible.
- A reviewer can assess the force rule and plausible event assignment. The
  same force trace is not independent physical character-boundary ground truth.
  State who reviewed it and whether the review was blind. AI review alone is
  provisional and must not be described as independent human validation.
- Completion condition: retain a case ledger, unresolved examples and review
  limits. Report results by sampling group; do not present a stratified audit
  as unweighted accuracy over all training boundaries.

#### Step 3 procedure fixed before trace inspection - 2026-11-16

- Scope: the same accepted WI/RH fold-0 training boundary pool and unchanged
  scorer/export. No model inference, realignment, score fitting or training.
- Freeze 100 different recordings and boundary IDs using Python Random seed
  42. Sort IDs before drawing. Candidate selection reads metadata, rank,
  interval duration and letter identity, not force values or trace images.
- Main sample: split the 176 eligible pair ranks into consecutive bands
  1-59, 60-118 and 119-176. Draw 30 different pairs uniformly per band.
  For each pair, draw one accepted occurrence uniformly from unused recordings.
  These are pair-balanced diagnostic samples, not occurrence-weighted estimates.
- Supplement: draw two fallback cases, two two-letter-word (only-position)
  cases, two regions longer than 480 ms, one uppercase-left case, one possible
  delayed-stroke-left case, and one case each for the known-sensitive dr and wo.
  Draw from remaining recordings in this order. Do not replace difficult cases.
- Possible delayed-stroke letters are i/j/t/f/x and umlaut letters, including
  uppercase forms. This is a screening flag, not an observed stroke label.
- Shuffle review order with separate seed 4201. Give cases A001-A100.
  Freeze the selected IDs, exact rules, source hashes and full selected rows
  before opening any raw force trace. Record group sizes and coverage.
- Plot raw 100 Hz tip force without smoothing, with the unchanged 10% force
  reference threshold, original CTC gap edges, local 100 ms window and any
  fallback interval. Show the full recording plus a detailed surrounding view.
  Gap edges are emission-span edges, not verified physical letter boundaries.
- Hide pair score, rank band and sampling reason in the review plots. Words,
  letters and writer ID remain visible. Earlier aggregate results were seen;
  this is partially masked Codex AI review, not blind human validation.
- Recompute raw force summaries and check against saved export values.
  Describe local and region evidence separately: no below-threshold sample,
  brief low force (one or two 10 ms samples), or a low run of at least three
  samples. Thirty milliseconds is descriptive here, not a new scoring rule
  or established physical pen-lift threshold. Keep the original score unchanged.
- Record multiple low runs, events at/near interval edges, local/region
  disagreement, empty-gap fallbacks and uncertain assignment. A low run within
  80 ms of an edge is an edge-sensitivity flag, not a physical error bound.
- Inspect every frozen case visually. Keep case-specific notes and unresolved
  interpretations. Do not call dots/crosses confirmed from force alone.
  Without independent character timing or writing video, inter-letter event
  assignment and delayed strokes remain unverified.
- Report main rank-band groups separately from targeted supplements.
  No pooled population accuracy, independent ground truth or calibrated
  continuity probabilities can be obtained from this audit.
- Save frozen selection, raw checksums, plots, feature checks, review ledger
  and source with results/thesis/forced_alignment_review/wi_rh_fold0_20261116_trace_audit/.
  Update this record with outcomes and limits after inspection.

### Step 4 — Decide whether one score change is justified

- Status: the user-approved assignment diagnostic completed on 2026-11-16.
  Implementation and verification passed. No new pair score was adopted.
- Use Steps 2–3 to decide whether word support, a sustained-lift rule, or
  delayed-stroke reporting needs attention first. Change one factor at a time.
- Do not automatically adopt five words, 30 ms, per-letter subtraction, or
  removal of the 50 ms features. Preserve the baseline and existing exports.
- Specify the new rule, reason, expected failure cases and comparison before
  implementation. Separate delayed-stroke groups before subtracting letter
  effects, which could remove useful pair differences.
- Compare the approved variant with the baseline on the fixed diagnostics.
  Preserve negative findings. Keep motion optional unless explicitly approved.
- Completion condition: retain the baseline if evidence does not support a
  change. Otherwise document the approved rule and its actual limits.

#### Proposed rule: whole drops, conservative links

Status at writing: proposal only, 2026-11-16. The user then approved this
assignment pilot. The rule below is the approved diagnostic, not an adopted
pair-scoring or tokenizer method. See the execution record for its outcome.

Purpose: keep a force drop whole, then ask which model-estimated letter gap it
could belong to. Do not force every drop into a gap. A temporal match remains
a candidate association, not proof of a physical inter-letter lift.

In simple terms:

1. Find the whole drop in the recording.
2. Compare it with all letter gaps in that word.
3. Keep a candidate link only when one whole drop fits strictly inside one
   gap and there is no competing drop or gap.
4. Otherwise record why the link is uncertain. Keep the case visible.

##### 4A. Find events before looking at gaps

Use the unchanged raw-force threshold, 10% of that recording's 90th-percentile
force reference. An event is one consecutive run of samples at or below this
threshold in the full recording. Save its start, end and full recorded duration.
The end is the first subsequent sample above threshold.

For this first assignment diagnostic, retain even 10-20 ms runs and report
their durations. Do not add a 30 ms filter, smooth the signal, join separate
runs, expand windows or move alignment anchors. This isolates the assignment
change. A run touching the start or end of the recording is truncated; its
full physical extent is unknown and it cannot receive a clean assignment.

Example: a drop lasts 80 ms, but only its final 10 ms falls inside the old
local window. It remains an 80 ms recorded event. It does not become a 10 ms
event because the window cuts it.

##### 4B. Compare with every gap in the word

Load all boundary records for each selected recording, not only the audited
boundary. The same event must not acquire two supposedly independent labels
because neighbouring gaps were inspected separately.

Use the original CTC candidate intervals:
end of the left character-emission span to start of the right span.
Do not use the 100 ms midpoint fallback as a physical gap. Keep zero-width
gaps explicitly unavailable for interval assignment.

Build and save a table of every positive-duration event/gap overlap before
choosing links. Use half-open sample intervals [start,end): merely touching
an interval endpoint does not create positive overlap. Such an outside event
can still carry the separate near-edge warning. Identify events by recording
and event index, and gaps by their existing boundary_id. Repeated pair text
does not identify a unique gap.

Keep unreliable or padded neighbouring gaps visible as possible conflicts;
do not delete them to make a match look unique. They cannot supply a clean
assignment. Events lying wholly within a character-emission span, outside all
gaps, or at recording edges remain in an unassigned-event ledger. Do not move
them to the nearest gap.

##### 4C. Record candidates and ambiguity separately

| Situation | Proposed record |
| --- | --- |
| Exactly one non-truncated drop fits strictly inside one reliable, nonempty gap; neither has another overlap | Single force-drop candidate for that gap |
| An overlapping drop touches or crosses a gap edge, including equality of its start/end with the gap start/end | Edge assignment uncertain; keep its full recorded duration |
| One drop overlaps two or more gaps, including an unreliable competing gap | Shared event; assignment uncertain |
| Two or more drops overlap one gap | Multiple events; assignment uncertain |
| A reliable nonempty gap overlaps no drop | No drop observed in that model interval |
| Empty gap, unreliable anchors, padding, or truncated event | Assignment unavailable or uncertain |

More than one uncertainty reason may apply. Retain all reasons and event IDs.
A single candidate is only a transparent time-based link. It does not rule
out a within-letter stroke or a delayed dot/cross.

No drop observed is not proof that the physical letter join is connected.
A nearby drop outside the gap may indicate a timing problem. Preserve the
existing 80 ms near-edge flag as a diagnostic, not an automatic matching
tolerance or a known timing-error bound.

All events must reconcile: uniquely linked candidates, ambiguous events and
unassigned events account for every detected run once. Store the full overlap
table separately, because an ambiguous event can overlap several gaps.

##### 4D. Keep delayed strokes and timing uncertainty visible

Retain the existing possible-delayed-left flag. Also record the right letter
and note whether either neighbour may have a dot, cross or umlaut stroke.
These are review flags, not automatic explanations or exclusions. Delayed
strokes can concern other letters in the word too; force alone cannot resolve
them. Do not subtract per-letter effects or silently call those events breaks.

For a declared timing stress check, shift all gaps of a word together by
-80 ms and +80 ms and repeat the overlap bookkeeping, keeping events fixed.
If a candidate or no-drop observation changes, flag it as timing-sensitive.
These are one-frame diagnostic shifts, not estimated physical uncertainty.
Do not choose whichever shift produces the preferred ranking. Intervals
shifted outside the recording become unavailable; preserve that result.

##### 4E. Pilot on the same frozen 100 recordings before changing scores

The proposed implementation would create an assignment report beside the
baseline, using the existing 100 cases and every neighbouring gap in those
recordings. It would not retrain the model, realign samples, overwrite the
audit ledger or change the current pair ranking.

For each audited gap, show the old local/region measurements, full event
intervals, overlap table, candidate/uncertain status, uncertainty reasons and
the timing stress result. Reuse the fixed case IDs. Plot complete events and
all nearby gap edges so crossings and competitors are visible.

Report separately for high/middle/low and targeted groups:

- Counts of single candidates, no-drop observations, ambiguous and unavailable
  gaps, with overlapping ambiguity reasons reported separately.
- Fraction of cases in which the assignment changes under either timing shift.
- Coverage by pair, word and writer, plus possible delayed-stroke groups.
- Cases where the old window clipped an event, missed an event elsewhere in
  the gap, or combined several events.

Verification must reconcile event IDs and durations with raw samples, enforce
no duplicate clean assignment, preserve half-open indexing, and retain every
ambiguous/unavailable case. Check gap endpoints, zero gaps, recording edges,
repeated letters, competing gaps and multiple drops explicitly.

The 100 cases already informed this proposal. Rechecking them is development,
not independent validation. Human review with independent character timing or
writing video would be needed before claiming physical assignment accuracy.
The same force plot cannot supply that ground truth by itself.

Do not compute a new pair contact percentage by silently dropping ambiguous
cases: uncertainty may depend on pair identity, word, writer and gap duration.
First report coverage and missingness. Any later aggregation and matched
recognition comparison need their own stated design. The current 75%/25%
score, support gates and tokenizer choices remain unchanged.

Expected benefit: recorded event duration no longer depends on how a window
cuts the drop, and uncertain attribution becomes explicit. Remaining limit:
this bookkeeping cannot make the CTC gap a verified physical letter boundary.
If useful assignments cover too little data, report that failure and retain
the baseline rather than forcing a link.

##### Execution conventions fixed before the pilot

- Read the same 100 frozen recording IDs. Load every neighbouring boundary
  record, including unreliable ones, from the unchanged original export.
- A usable gap has agreeing anchors, an unclipped source candidate region,
  no original padding, positive width and shifted bounds inside raw data.
  Do not repair source quality by shifting a gap.
- Use original raw thresholds after reconciling them with the recording
  force reference. Detect full runs once; all three timing variants use the
  same event IDs, sample bounds and recorded durations.
- Events touching recording boundaries stay unassigned, even when they
  overlap a gap. Other events with conflicting overlaps are ambiguous.
  An unavailable gap can still compete in the overlap table.
- Gap categories are single_candidate, no_observed_drop, ambiguous and
  unavailable. Unavailable takes precedence; preserve all overlapping
  reason flags separately. These are not physical contact labels.
- The timing signature consists of category, overlapping event IDs,
  uncertainty reasons and candidate event ID. Any change under -80 or
  +80 ms is association-sensitive. Category-only changes are also reported.
  A near-edge warning alone does not count as an assignment change.
- Save full JSON/CSV ledgers, 100-case comparison, group/coverage summaries,
  source snapshots, hashes and 25 plot sheets in a new
  results/thesis/forced_alignment_review/wi_rh_fold0_20261116_assignment_pilot/
  directory. Refuse to overwrite existing results.

### Step 5 — Check broader consistency without treating it as proof

- Status: planned; define each comparison before execution.
- Compare whole-word force-lift summaries with pair evidence only after
  defining a valid prediction model. Corrected scores are not calibrated
  probabilities and cannot simply be summed into expected lift counts.
- Disagreement can reflect force thresholds, delayed strokes, word context or
  alignment. It does not identify alignment as the sole cause.
- An optional comparison with existing IAM image evidence uses only shared
  pairs. Language, writers and measurement methods differ. Agreement supports
  consistency; disagreement alone does not invalidate either measurement.
- Completion condition: state what each check can and cannot establish.
  Do not select thresholds from held-out recognition labels or scores.

### Step 6 — Build the provisional tokenizer and scope a recognition comparison

- The fold-0 vocabulary build is complete. The v0 execution record below
  gives the selection rules, vocabulary size, artifacts and verification.
- A recognition comparison would hold the fold, model and training settings
  fixed across character, force-ranked and frequency-ranked targets. Define
  the evaluation protocol and metrics before starting the runs.
- This OnHW sensor experiment is secondary to the image-derived thesis method.
  Extending it beyond fold 0 would require each fold's own training evidence.

#### Proposed recognition comparison for Bigram v0

Compare three CTC recognizers on the same OnHW-Words500 writer-independent,
right-handed split. The primary comparison is force-ranked vs. frequency-ranked
bigrams. These two conditions share the same 176 eligible candidates, 100-pair
budget, 160 output classes and greedy tokenization. Only the pair ranking rule
changes. The character model has 60 output classes and provides the reference
for the overall effect of using bigram targets.

1. **Run a matched fold-0 pilot.** Use the two frozen v0 vocabularies and
   retrain the character condition alongside them. Start from the B0 BLConv-B
   + BiLSTM-B recipe. Use the same 13 input channels, train recordings,
   preprocessing, augmentation, optimizer, schedule, batch settings and CTC
   loss. Use fresh models with matching seeds, beginning with seed 42. Keep
   the 300-epoch budget and score the epoch-299 checkpoint for every condition.
   For a repeated pilot, add seeds 43 and 44 for all three conditions.
2. **Freeze evaluation before training.** Check training-label round-trips,
   blank ID 0, output sizes and whether CTC can fit each target given its
   frame count, including the blank needed between identical adjacent IDs.
   Train without running the scoring split during the epoch loop, then score
   all final checkpoints on the same recordings. The pilot launcher opts
   into train-only mode and skips loading `val.json`. Use identical greedy
   CTC decoding with each model's own vocabulary and compare the resulting
   case-sensitive strings directly with the original references.
   [CTC best-path decoding](https://www.cs.utoronto.ca/~graves/icml_2006.pdf)
3. **Measure accuracy on paired predictions.** Use corpus character error rate
   (CER), total character edits divided by total reference characters, as the
   primary metric. Report word error rate (WER) from the same checkpoint.
   Save case ID, writer ID, reference, prediction and edit counts for every
   condition and seed. Report the force-minus-frequency CER difference in
   percentage points and both comparisons against characters. Resample
   whole evaluation writers jointly across conditions for a paired interval.
   With multiple seeds, compute the interval for the mean paired delta with
   the same writer resample in every seed and report individual seed deltas.
   Token counts and CTC loss remain diagnostics.
4. **Validate a promising result across writer splits.** The v0 ranking uses
   training boundary rows from A0, whose epoch-275 checkpoint was selected by
   CER on fold-0 validation. This indirect exposure makes the v0 comparison
   a development pilot. For a benchmark-wide comparison, fix the protocol
   before scoring and build new force and frequency vocabularies inside each
   outer training fold. Select the aligner checkpoint at a preset epoch or
   using only inner, writer-disjoint validation. Keep each outer validation
   set out of ranking, vocabulary and checkpoint selection. Report paired
   fold-level results. The OnHW benchmark reports over five folds.
   [OnHW benchmark paper](https://link.springer.com/article/10.1007/s10032-022-00415-6)

A negative force-minus-frequency CER difference favors force ranking under
this shared candidate gate. Look for a consistent improvement across runs and
writers and check WER for a practical regression. If both bigram systems beat
characters and frequency has the better CER, the pilot supports bigram targets
and favors the frequency ranking. A small or inconsistent gap is inconclusive.
The seed-42 fold-0 pilot is complete. See the execution record for results.

##### Launch the fold-0 pilot

From the TVA repository root in the WSL `tva` environment, run the seed-42
conditions one at a time on one GPU.

```bash
export PYTHONHASHSEED=42
python -m tva.forced_alignment.train_bigram_v0 char
python -m tva.forced_alignment.train_bigram_v0 frequency
python -m tva.forced_alignment.train_bigram_v0 force
```

The launcher reads the same B0 YAML for every condition and loads the
appropriate existing fold-0 vocabulary. Use `--seed 43` or `--seed 44` on
all three commands for a matched repeat. The run saves the resolved config and
checkpoints under
`results/thesis/bigram/wi_rh_fold0_bigram_v0_pilot/seed_42/<condition>/0/`.
Choose the fixed endpoint at `checkpoints/299.pth` for later evaluation.

The seed-42 launch configs were exercised on synthetic CPU data with the real
vocabularies and B0 architecture. Each wrote a finite-loss checkpoint with
only training metrics. The training-only preflight round-tripped all 19,907
labels in every condition. The existing B0 sequence padding makes CTC targets
feasible and adds frames to five character, three frequency and four force
training recordings. The raw signals without sufficient CTC frames number
three, two and three, respectively.

#### Match the emitted vocabularies for WI and WD fold 0

Use [train_force_tokenizers.ipynb](../train_force_tokenizers.ipynb) to run
the vocabulary-building steps. The
[config map](../configs/thesis/forced_alignment/README.md) shows the WI and
WD YAMLs side by side and links to the builder and scorer.

The November WI run used 160 output classes on both sides, including blank.
Force training emitted 97 of its 100 selected bigrams. Frequency training
emitted all 100. The new comparison matches both counts. Each vocabulary
keeps the same 60 character and blank IDs and adds 100 bigrams that each
appear in at least one encoded training label.

Use the same within-fold candidate pool and quality gate for both rankings.
Start with the top 100 for each. Keep the pairs the greedy encoder emits.
Replace unused pairs in rank order, accepting a replacement only when all
already selected pairs still appear in the encoded training labels. The
builder checks full label round-trips, ID ranges and the emitted count
before writing either vocabulary. Total training token counts can differ
because the rankers choose different pairs.

Export the WI and WD scores separately, using the final A0 epoch-299
checkpoint for each split. The WI checkpoint already exists. The WD
aligner trains on WD fold-0 training recordings with validation disabled.
The builder checks provenance and requires a complete training export
from the same dataset and fold. Both use the existing force score and
support thresholds. WI fold 0 remains a development split that we have
already evaluated. Rebuilding from epoch 299 also changes the source of
its force ranking compared with the previous pilot.

##### Verified WI build

The 2026-11-22 WI run exported all 88,292 gap rows from epoch 299.
Independent checks matched every row to its training annotation.
The score retained 79,099 reliable boundaries and 175 eligible pairs.
The raw top 100 force list had two unused pairs, `tw` and `gs`.
The refill chose `ta` and `ua`. The frequency list needed no refill.

| WI vocabulary | Output classes | Emitted bigram types | Training tokens |
| --- | ---: | ---: | ---: |
| Force | 160 | 100 | 75,450 |
| Frequency | 160 | 100 | 70,841 |

Both vocabularies round-trip all 19,907 training labels. Both launchers pass
their dry run, and their resolved settings differ only in the vocabulary
and output paths. The WD aligner and export passed a smoke test using four
WD training recordings. The full WD build requires the checkpoint below.
All 49 tests passed. The original WI files retain their saved hashes.
[WI build summary](../results/thesis/forced_alignment_review/bigram_matched_fold0/wi/tokenizers/summary.json) ·
[WI provenance](../results/thesis/forced_alignment_review/bigram_matched_fold0/wi/tokenizers/run_manifest.json)

##### WD alignment and vocabulary setup

From the TVA root in the `tva` environment, run the WD aligner once.
Run the remaining commands after it writes `299.pth`.

```bash
export PYTHONHASHSEED=42
TVA_WD_A0=results/thesis/alignment_models/A0_char_wd_rh_unidirectional
test ! -e "$TVA_WD_A0/0/checkpoints/latest.pth" && \
  python main.py --config configs/thesis/a0_char_wd_rh_unidirectional.yaml
```

```bash
export PYTHONHASHSEED=42
TVA_WD_A0=results/thesis/alignment_models/A0_char_wd_rh_unidirectional
TVA_WD_DATA=data/tva/onhw_words500_wd_word_rh
TVA_WD_MATCHED=results/thesis/forced_alignment_review/bigram_matched_fold0/wd

OMP_NUM_THREADS=1 python -m tva.forced_alignment.export_boundaries \
  --config configs/thesis/a0_char_wd_rh_unidirectional.yaml \
  --checkpoint "$TVA_WD_A0/0/checkpoints/299.pth" \
  --split train --device cuda --num-workers 4 \
  --output "$TVA_WD_MATCHED/evidence/boundaries.jsonl" && \
python -m tva.forced_alignment.score_continuity --input "$TVA_WD_MATCHED/evidence/boundaries.jsonl" && \
python -m tva.forced_alignment.build_bigram_v0 --match-emitted \
  --train "$TVA_WD_DATA/train.json" \
  --char-vocab "$TVA_WD_DATA/tokenizers/char/0.json" \
  --force-ranking "$TVA_WD_MATCHED/evidence/continuity_scores/pair_continuity_scores.csv" \
  --output-dir "$TVA_WD_MATCHED/tokenizers"
```

##### Launch the matched recognition runs

Run one command at a time on one GPU. Each uses seed 42 and trains for
300 epochs without reading validation. Use the existing character baselines
for reference. Score epoch 299 for each condition.

```bash
export PYTHONHASHSEED=42
python main.py --config configs/thesis/forced_alignment/wi/frequency.yaml
python main.py --config configs/thesis/forced_alignment/wi/force.yaml
python main.py --config configs/thesis/forced_alignment/wd/frequency.yaml
python main.py --config configs/thesis/forced_alignment/wd/force.yaml
```

The runs are saved under
`results/thesis/bigram/bigram_matched_fold0/<setting>/seed_42/<condition>/0/`.

#### 2026-11-19 — Proposed ED sentence recognition comparison

ED consists of English sentences. The OnHW-Words500 labels used for the
current ranking are predominantly German and include words such as
`Mittwoch` and `drei`. Pair counts, word positions and surrounding letters
can shift across the two corpora. For an English sentence recognition
comparison, derive the two rankings from ED training data.
[OnHW benchmark paper](https://link.springer.com/article/10.1007/s10032-022-00415-6)

Start with ED `gold_wi` fold 0. It has 2,025 training recordings from 81
writers and 525 validation recordings from 21 disjoint writers. Use the ED
BLConv-B + BiLSTM-B recipe and the same training settings and fixed
checkpoint rule in each condition. The previous ED character results used
validation-selected checkpoints and provide historical context.
[ED character rerun summary](../results/thesis/ed/ed_char_cv_rerun_summary_20261017.json)

- **ED characters.** Use the 79-class ED vocabulary, including CTC blank ID 0
  and space ID 1. Preserve case, spaces, digits and punctuation.
- **ED force-ranked bigrams.** Compute force continuity on reliable
  within-word letter boundaries in the ED training recordings.
- **ED frequency-ranked bigrams.** Rank the same eligible ED pairs by
  frequency in the ED training labels. Give both bigram vocabularies the
  same pair budget, character fallbacks and greedy segmentation. Set the
  budget after checking training support and before scoring recognition.

Begin with a training-only force and alignment feasibility check. ED stores
seven channels at 100 Hz and lists force as the seventh saved channel.
The current exporter reads force at column 12 of the 13-channel OnHW
layout and assumes OnHW motion channel indices. Map the ED channels
explicitly, verify pen-contact behavior and audit alignments and pauses with an
ED character aligner whose checkpoint is chosen using training data only
or fixed in advance. Compute word-position corrections and pair support
from ED training writers and only allow bigrams between adjacent letters.

Use case-sensitive corpus CER with spaces retained as the primary metric and
sentence WER as the secondary metric. Compare paired predictions on the
same recordings using greedy CTC decoding, report writer-level uncertainty
and rebuild the evidence and vocabularies separately inside later folds.

##### Optional transfer comparison from German words to English sentences

The earlier transfer feasibility check reused the saved OnHW rank ledger
and ED character mapping from `handwriting-tokenizers-v1`. It projected both
OnHW ranks onto the same 160 candidates observed in ED fold-0 training,
took the top 100 per rank and round-tripped all 2,025 training labels in
memory. The selections shared 64 bigrams. The ED frequency row uses that
same OnHW-derived candidate pool.

| Training targets | Tokens | Reduction vs. characters |
| --- | ---: | ---: |
| ED characters | 84,158 | — |
| Transferred OnHW force rank | 71,299 | 15.28% |
| Transferred OnHW frequency rank | 67,839 | 19.39% |
| ED frequency on the OnHW-derived candidate pool | 66,492 | 20.99% |

This would test whether an OnHW ranking transfers across language, writing
task and corpus. The feasibility check measured target length only and
saved no vocabularies. No ED recognition training or evaluation ran.

### Interpretation limits retained throughout

- The 80 ms output spacing is confirmed by the encoder. It does not establish
  a physical timestamp error bound of ±40 ms. A 50 ms force window still
  measures force, but its placement may be wrong.
- Offsets from evenly spaced letters are sanity checks, not measured errors
  against true character boundaries. A0 reduces that diagnostic bias; it does
  not establish accurate physical timing.
- A CTC blank interval is a model interval. A physical pen lift can occur
  outside it. Empty intervals use a marked 100 ms fallback in the baseline.
- Tip contact and connected image ink are related but distinct measurements.
- Five hundred and one word types do not by themselves prove that a regression
  with about 485 terms is impossible. Rank, overlap, regularization and an
  independent validation design would matter; no such model is adopted here.

### Execution record

#### 2026-11-19 — OnHW Bigram v0 recognition pilot completed

The user trained the frequency and force conditions for 300 epochs and
reused the existing B0 character run. Evaluated all three at fixed,
zero-based epoch 299, seed 42, on the same 5,292 WI/RH fold-0 recordings
from 11 held-out writers (28,877 reference characters). Both bigrams used
`299.pth`. The B0 character `latest.pth` contained epoch 299. The saved
training settings match apart from tokenizers, paths and validation mode.
B0 logged validation during training. All its archived epoch-299 predictions
were reproduced exactly during the new evaluation.

| Condition | CER | WER | Character edits | Wrong words |
| --- | ---: | ---: | ---: | ---: |
| Character | 15.80% | 28.10% | 4,563 | 1,487 |
| Frequency-ranked | 16.74% | 25.42% | 4,834 | 1,345 |
| Force-ranked | 16.28% | 25.53% | 4,701 | 1,351 |

The primary force-minus-frequency result was −0.46 percentage points CER,
or 133 fewer character edits. The paired 95% writer-bootstrap interval was
[−1.15, +0.06] points (20,000 draws, seed 42). Force had lower CER for four
writers, tied for two and had higher CER for five. Its WER was +0.11 points
versus frequency with an interval of [−0.73, +0.90] points. Both intervals
include zero. The comparison is inconclusive across writers at this seed.

The character baseline had the lowest CER. Both bigram models had lower
WER than characters, by 2.57 points for force and 2.68 points for frequency.
The resulting trade-off depends on whether character edits or exact-word
recognition matters more for the application. The pilot covers one seed
and one fold. The original A0 aligner was selected using this validation
fold, which limits the strength of the result for the force method.

All three checkpoints loaded strictly, each evaluation label round-tripped,
and every prediction was reconciled against the original annotation order.
Per-recording edit counts reproduced the evaluator's CER and WER. The
standard B0 target-length padding adds frames to six evaluation recordings
for characters and three for each bigram. The run directory contains the
commands, configuration and input hashes, paired predictions, writer
counts, analysis code and bootstrap summary.
[Evaluation artifacts](../results/thesis/bigram/wi_rh_fold0_bigram_v0_pilot/seed_42/evaluation_20261119/)

#### 2026-11-17 — Bigram v0 built for OnHW WI/RH fold 0

Built the first provisional tokenizer comparison from the existing Step 1 force
ranking. The Step 4 pilot found substantial sensitivity to an 80 ms shift.
Its event assignments stay diagnostic, and v0 uses the reproduced ranking.
The broader consistency work in Step 5 remains open.

Both vocabularies select 100 pairs from the same 176 pairs with sufficient
force evidence. The force condition takes the first 100 saved ranks. The
frequency control ranks the same candidates by adjacent positions in all
19,907 fold-0 training recordings. Each recorded label counts once per
recording and repeated pair positions count separately. Ties use ascending
case-sensitive Unicode pair order. TVA's general Bigram trainer counts each
distinct label once. This control measures frequency per training recording
and is conditional on the shared eligibility gate.

Both files retain all 59 character IDs and blank ID 0. Bigram IDs 60–159
follow the condition's rank order. TVA's existing Bigram tokenizer reads both
files and matches pairs greedily from left to right, with character fallback.

| Training measure | Force-ranked | Frequency control |
| --- | ---: | ---: |
| Vocabulary IDs including blank | 160 | 160 |
| Exact label round-trips | 19,907/19,907 | 19,907/19,907 |
| Encoded tokens | 75,766 | 70,841 |
| Characters per token | 1.428 | 1.527 |
| Token reduction from the 108,199-character baseline | 29.98% | 34.53% |
| Selected pairs that are emitted at least once | 97/100 | 100/100 |

The vocabularies share 58 bigrams and each selects 42 different ones (Jaccard
0.408). For example, the force vocabulary splits `Mittwoch` as
`Mi | tt | wo | ch` and the control splits it as `M | it | t | wo | ch`.
The selected force pairs `tw`, `gs` and `ön` occur in training labels but are
never emitted because the greedy scan consumes an overlapping earlier pair.
They stay in v0 to preserve the selection rule and fixed budget. The
frequency control emits 4,925 fewer training tokens. This measures target
length and leaves recognition accuracy and physical boundary validity open.

The [Bigram v0 demo notebook](../bigram_v0_demo.ipynb) loads the saved
vocabularies, compares word splits side by side and lets you enter a word to
see its tokens, IDs and decoded text.

If the run directory is absent, rebuild it from the TVA root with the TVA
Python environment and the saved Step 1 ranking:

```bash
PYTHONHASHSEED=42 python -m tva.forced_alignment.build_bigram_v0 \
  --train data/tva/onhw_words500_wi_word_rh/train.json \
  --char-vocab data/tva/onhw_words500_wi_word_rh/tokenizers/char/0.json \
  --force-ranking results/thesis/forced_alignment_review/wi_rh_fold0_20261115_baseline/pair_continuity_scores.csv \
  --output-dir results/thesis/forced_alignment_review/wi_rh_fold0_20261117_bigram_v0
```

- The standalone builder writes both fold files, the full 176-pair selection
  ledger, summary and provenance manifest. The manifest records the command,
  code revision and SHA-256 hashes for the inputs, source files and outputs.
  [Builder](../tva/forced_alignment/build_bigram_v0.py) · [Run artifacts](../results/thesis/forced_alignment_review/wi_rh_fold0_20261117_bigram_v0/)
- Verification loaded the saved vocabularies through the repository factory,
  checked contiguous IDs and round-tripped every training recording. An
  independent frequency count and segmentation reconciled both selections,
  every pair's emission count and the token totals. A second build under a
  different Python hash seed reproduced all four data files byte for byte.
  The existing-output guard passed and 22 alignment and scoring tests passed.
- The run used fold-0 training files only. The vocabularies and reports sit
  under Git-ignored results. No held-out labels were read and no recognition
  training or evaluation ran. A recognition comparison remains a separate
  design decision.

#### 2026-11-16 — Step 4 pilot completed: explicit uncertainty and timing sensitivity

The user approved implementation of the whole-drop proposal. Added
[force_gap_assignment.py](../tva/force_gap_assignment.py) and
[assign_force_events.py](../tva/forced_alignment/assign_force_events.py), with
[18 focused tests](../tests/test_force_gap_assignment.py).
The module detects full recorded low-force runs and builds the complete
event/gap overlap table. The runner exports every category and compares
the declared -80/0/+80 ms gap timings. It does not calculate a new pair score.

Original command, recorded before the command files moved under `tva/`:

```bash
PYTHONHASHSEED=42 python assign_force_events.py --output-dir results/thesis/forced_alignment_review/wi_rh_fold0_20261116_assignment_pilot
```

The module path for a new run is `python -m tva.forced_alignment.assign_force_events`.

Used WSL TVA Python 3.11.15 and existing NumPy/Matplotlib. No dependency was
added. This remains seed-42 A0 WI/RH fold-0 training evidence, checkpoint
epoch 275, with no model inference or realignment. Source revision is b7e4d5f
plus saved uncommitted code/docs. The manifest and snapshots record exact
source and input hashes.

The same 100 recordings contain 547 original word gaps, including the 100
audited gaps. All participate, including unreliable neighbours. There are
556 full recorded force runs: 242 unique candidates across all gaps, 282
ambiguous and 32 without positive gap overlap. Every event appears once in
that ledger. At original timing the 547 gaps contain 242 candidates,
113 no-drop observations, 161 ambiguous and 31 unavailable cases.
Unavailable gaps include 21 unreliable and 12 empty gaps; two have both flags.
Nothing was discarded.

For the fixed 100 audited gaps:

| Frozen group | n | Single candidate | No observed drop | Ambiguous | Unavailable | Association changes under either shift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| High | 30 | 11 | 9 | 10 | 0 | 16 |
| Middle | 30 | 16 | 5 | 8 | 1 | 14 |
| Low | 30 | 15 | 3 | 12 | 0 | 18 |
| Targeted | 10 | 3 | 2 | 3 | 2 | 2 |

These are diagnostic counts, not accuracy or population contact rates.
Totals are 45 candidates, 19 no-drop observations, 33 ambiguous and three
unavailable empty gaps.

Timing is the main limitation:

- 50/100 associations change under at least one declared shift. In 36 cases
  the category changes. In another 14, event IDs or uncertainty reasons
  change while the broad category stays the same. The near-edge flag alone
  does not count as an association change.
- Only 21/45 original candidates keep their link/category/reasons in both
  shifts. In contrast, 18/19 no-drop observations remain unchanged.
- Removing unstable cases would remove candidate drops much more often than
  no-drop observations in this sample. A contact percentage from the remainder
  could be biased toward apparent continuity. No such exclusion was adopted.
- These one-frame shifts are stress checks, not measured physical timing
  errors. Passing them still does not prove correctness.

Examples:

- A001 (dem): one full 90 ms event fits the gap in all variants, although the
  old midpoint window misses it.
- A002 (wie): its 140 ms candidate becomes ambiguous in both shifts. The
  earlier gap cuts it; the later gap also admits another event.
- A006: full events of 90 and 80 ms remain separate. The gap is ambiguous
  for multiple events rather than gaining a short invented lift.
- A047 (Teil): its original gap intersects two different 80 ms events.
  Either shift intersects a different crossing event. It remains ambiguous,
  but its event association is timing-sensitive.
- A084 (Region): no drop overlaps the original gap; the later shift reaches
  a crossing event. No-drop evidence is not automatically robust.
- A037/A098/A099: empty gaps stay unavailable in every variant. The legacy
  local fallback is shown for comparison but supplies no assignment.

Delayed-stroke flags remain descriptive. Among 17 left-letter flagged cases,
four have candidates and 13 are ambiguous. Flagging either neighbour gives
34 cases: nine candidates, four no-drop, 20 ambiguous and one unavailable.
These confounded groups do not prove that dots/crosses caused ambiguity.

Artifacts and verification:

- [Pilot directory](../results/thesis/forced_alignment_review/wi_rh_fold0_20261116_assignment_pilot/)
  contains case_assignments.csv, event_assignments.csv, gap_assignments.csv,
  event_gap_overlaps.csv, recording_assignments.jsonl, group_summary.csv,
  coverage_by_pair_word_writer.csv, assignment_summary.json and run_manifest.json.
  It includes exact source snapshots, pre-run Git diff and protocol.
- [Revised plot sheets](../results/thesis/forced_alignment_review/wi_rh_fold0_20261116_assignment_pilot/plot_revision_v2/plots/)
  show full events and every nearby gap. An off-axis label issue was fixed
  in the renderer only. Original plots and numerical exports remain.
  The plot revision has a separate source/hash manifest. Revised sheets
  1/12/25 were inspected, including candidate, multiple-event and empty cases.
- The 18 new tests and four existing scoring tests pass (22 total):
  PYTHONHASHSEED=42 python -m unittest tests.test_force_gap_assignment
  tests.test_continuity_scoring.
- [Independent verification](../results/thesis/forced_alignment_review/wi_rh_fold0_20261116_assignment_pilot/verification_report.json)
  checks all three variants: 1,668 event, 1,641 gap and 1,534 overlap rows.
  A separate raw scan reproduces events. Sample-set intersections reproduce
  the overlap graph. Mutual uniqueness, containment, category accounting,
  timing signatures, group summaries and all CSV fields reconcile.
- The existing-output guard refuses a repeat run without altering its manifest.
  Original input/raw/baseline/audit/Step 2 hashes match. Frozen sample and
  review ledger are unchanged. Scientific source matches the saved run;
  the only later runner change is plot clipping.
- No production scorer, model, frozen tokenizer or training config changed.
  No full test suite, inference, training or tokenizer construction ran.
  Physical event-to-letter assignment remains unverified.

Outcome: the approved bookkeeping and bounded pilot are complete. They
preserve whole event durations and expose uncertainty. This does not justify
a replacement pair-ranking method. Next: determine independent timing
evidence or propose a controlled alignment improvement before aggregation.
Keep the baseline. Do not silently drop ambiguous/unstable cases and call
the remaining fraction a connectivity probability. Steps 5-6 remain pending.

#### 2026-11-16 — Step 4 assignment proposal recorded; implementation pending

- The user requested a simple proposal for assigning force drops to letter
  gaps. The [proposed rule](#proposed-rule-whole-drops-conservative-links)
  detects complete low-force runs first, then checks every gap in the word.
- A single contained event can receive a candidate time-based link. Crossing,
  shared, multiple, truncated and unavailable cases remain explicit.
  Zero gaps do not gain physical evidence from the legacy midpoint fallback.
- The proposed pilot reuses the frozen 100 cases, retains all nearby gap
  records and reports coverage and one-frame timing sensitivity. It does not
  yet change the pair score, minimum duration, support gate or tokenizer.
- Only this documentation/proposal was completed. No assignment pilot or new
  scientific result is claimed. Physical event assignment remains unverified.
- Next: review this concrete proposal before implementation. A later scoring
  rule must address ambiguity and selection bias explicitly.

#### 2026-11-16 — Step 3: fixed trace audit completed as provisional AI review

The user approved the audit. Codex AI inspected all 100 frozen cases in 25
four-case sheets, with full raw-force traces and detailed interval views.
Scores, rank bands and selection reasons were hidden in those views. The
review was not blind: earlier aggregate results and some sensitive pairs were
already known. This is not independent human validation.

Sampling was frozen before loading raw recordings: 30 different pairs from
each rank band plus 10 targeted cases, using selection seed 42 and review-order
seed 4201. The sample covers 100 different recordings, 96 pairs and 38 writers.
Positions: 30 first, 45 middle, 22 final and three only-position cases.
It includes three empty-gap fallbacks, 22 regions longer than 480 ms,
12 uppercase-left cases and 17 possible delayed-stroke-left cases.
No case was replaced after viewing a trace.

- Frozen selection SHA-256:
  d6bd59ebedd6ff0a0c3fe935e0ce0bb2b24398cc59dc331622dd75469002c722.
- [100-case ledger](FORCED_ALIGNMENT_TRACE_AUDIT_20261116.csv)
  contains IDs, raw-file hashes, exact force categories, case-specific visual
  notes and unresolved letter/stroke assignments. Its plot column is relative
  to the local audit directory.
- [Local audit directory](../results/thesis/forced_alignment_review/wi_rh_fold0_20261116_trace_audit/)
  contains frozen_selection.json, freeze_manifest.json, protocol_at_freeze.md,
  render_manifest.json, visual_notes_v1.json, review_ledger.csv, audit_summary.json,
  group_summary.csv, full_run_extent_checks.json, audit_manifest.json and all
  25 PNG sheets. Selection/render/finalization source is saved beside them.

The table gives exact raw-force observations under the original threshold.
No low means no sample at/below that threshold; brief means a longest run of
10-20 ms; sustained means at least 30 ms. These are descriptive categories,
not verified pen-lift or physical connectivity labels. The original score was
not changed. Sampling groups must stay separate.

| Frozen group | Cases | Local: no / brief / sustained low | Region: no / brief / sustained low | Local no-low, region low | Region with multiple low runs |
| --- | ---: | --- | --- | ---: | ---: |
| High ranks 1-59 | 30 | 19 / 2 / 9 | 9 / 0 / 21 | 10 | 4 |
| Middle ranks 60-118 | 30 | 8 / 2 / 20 | 6 / 0 / 24 | 2 | 6 |
| Low ranks 119-176 | 30 | 7 / 3 / 20 | 3 / 0 / 27 | 4 | 10 |
| Targeted supplement | 10 | 5 / 1 / 4 | 4 / 1 / 5 | 1 | 3 |

Observed failure modes and examples:

- In 16/90 main sampled cases, the local window remains above threshold but
  the wider region contains sustained low force. Examples A001 (dem), A004
  (fünf), A012 (Da) and A070 (hier) show drops away from the midpoint.
  This is disagreement between two measurements, not a measured error rate.
- In 20/90 main cases, the region contains multiple separate low runs.
  A008 (kaum), A009 (Zum), A024 (können) and A077 (macht) show why a
  whole-gap event cannot automatically be assigned to one letter transition.
- Across the complete diagnostic sample, eight local windows have only
  10-20 ms of low force: A006/A010/A042/A058/A061/A065/A071/A088.
  Every one overlaps a full-trace low run of 60-150 ms. A 30 ms minimum
  applied inside the fixed local window would discard those fragments even
  though the underlying event is longer. This is not evidence for isolated
  single-sample sensor noise.
- A047 (Teil) has only 20 ms as its longest region low run, but that short
  interval intersects two different 80 ms full-trace runs. Duration computed
  after clipping an event to a CTC interval can hide its full extent.
- The 80 ms edge-sensitivity flag occurs in 17/30 high, 14/30 middle,
  18/30 low and 3/10 targeted cases. It only marks low samples near an edge.
  It is not a calibrated timestamp error or proof that the edge is wrong.
- A018 (sehen), A026 (Unternehmen) and A089 (Haus) show clear above-threshold
  intervals. A034 (Anderen) and A084 (Region) show large visible depressions
  that still stay above threshold. A visible force dip is not automatically
  a low-force event under the current rule.
- All three empty-gap fallbacks (A037/A098/A099) stay above threshold.
  They remain artificial 100 ms measurements, not observed nonempty gaps.
  In A098, the first low sample is index 117, just outside [107,117).
  Exact half-open counts resolve the visually borderline edge.
- Dots, crosses and umlaut events cannot be identified from force alone.
  All physical letter-event assignments remain unverified, including the
  17 cases flagged for possible delayed strokes. No false accuracy number
  was calculated.

Interpretation and next step:

High-ranked pairs more often keep force above threshold in this sample, but
the ranking and audit use the same training evidence. That pattern is internal
consistency, not independent validation. The pair-balanced and targeted sample
also does not estimate occurrence-weighted population accuracy.

The review exposes interval-placement and event-clipping problems. It does not
establish which drop is a true inter-letter lift. Before adopting a minimum
duration, Step 4 should consider event extent and event-to-letter assignment.
Retain the baseline. A concrete change and comparison still need a stated
proposal; no new gate, threshold, score, tokenizer or recognition condition
was selected by this audit.

Verification:

- Raw force from all 100 source CSVs matches the saved force reference and
  local/region minima, means, low fractions and longest low runs. All arrays
  are finite, 13-channel recordings of the expected length.
- The frozen IDs and hashes are unchanged after rendering/review. All 25
  sheets were visually inspected; all cases have individual notes.
- Current raw-file, checkpoint, export, baseline and Step 2 result hashes
  reconcile. Frozen English artifacts and production scoring/model code
  were not modified. No model inference, training or full test suite ran.
- The audit manifest identifies code revision b7e4d5f, Python/package
  versions, source, raw-data and result hashes. The original export's missing
  historical source revision/checkpoint hash remains a provenance limit.
- Independent physical timing/contact references and human confirmation are
  still needed before calling this a validated pen-connectivity measure.
  Step 3's provisional AI inspection is complete; Steps 4-6 remain planned.

#### 2026-11-15 — Step 2 completed: stable overall order, word-sensitive pairs

The user authorized this diagnostic explicitly. All 501 word omissions and
42 writer omissions completed. The scorer, A0 model, export, thresholds,
weights, correction rules and training split stayed fixed. The calculation
recomputed correction baselines and all pair scores after every omission.

The primary comparison retains the same 176 baseline-eligible pairs, including
diagnostic ranks for pairs that lose support. Spearman correlation measures
similarity of rank order; 1 means identical order. It does not measure physical
contact accuracy.

| Measure | Remove one word (501 checks) | Remove one writer (42 checks) |
| --- | ---: | ---: |
| Median fixed-set Spearman correlation | 0.999971 | 0.998998 |
| Lowest fixed-set Spearman correlation | 0.996586 | 0.997636 |
| Minimum retained fixed-set top 20 | 19/20 | 19/20 |
| Minimum retained fixed-set top 50 | 49/50 | 48/50 |
| Largest absolute pair score change (0-1 scale) | 0.097608 | 0.023269 |
| Largest fixed-set rank movement | 55 places | 17 places |
| Pairs that lose eligibility at least once | 42/176 | 2/176 |
| Pairs eligible in every omission of this type | 134/176 | 174/176 |
| Most pairs losing eligibility in one omission | 4 | 2 |

As a supplementary check, reapplying the support gate before forming each
eligible list retains at least 19/20 and 48/50 baseline leaders for word
omissions, and 19/20 and 47/50 for writer omissions. This supplements the
predeclared fixed-set comparison; it does not replace it.

Important examples:

- Removing the word drei changes dr from 0.397002 to 0.299393.
  Its diagnostic rank moves from 61 to 116, and it loses eligibility.
  It occurs in only three training word types. Rank 116 is a fixed-set
  diagnostic; the operational eligible list excludes it.
- Removing Programm changes gr from 0.627902 to 0.722789. It also loses
  eligibility. Removing Chief changes hi from 0.558802 to 0.651382 and
  removes it from the eligible list. Score sensitivity and support loss
  must be reported together.
- Removing Mittwoch changes wo from 0.500588 to 0.570879 and from rank
  28 to 14. It stays eligible and has six word types. A five-word rule
  would therefore not remove all word-sensitive cases.
- Across word omissions, 33 of the 42 pairs with support losses have three
  word types, eight have four, and one has five. Di, gr and hi are baseline
  top-20 pairs that lose eligibility in at least one word omission.
- Writer omissions affect Ge and av eligibility. Both start at
  101 accepted occurrences, so losing contributing observations breaks
  that occurrence gate; this does not show that their score is writer-driven.
  The largest writer-driven score change is sa, 0.182944 to 0.159675
  after removing writer 1. The largest rank movement is 17 places for
  the pair ät after removing writer 2007.
- Across pairs, the median worst absolute score change is 0.013270 for
  word omissions and 0.017145 for writer omissions. The larger word
  extremes concern particular pairs; word effects are not larger by
  every summary measure.

Interpretation:

The broad ranking and its leading set survive any single omission with little
change. Some individual pairs depend on specific words, and support eligibility
is less robust across word contexts than across writers. High overall
correlation must not hide these cases: removing most of 501 words removes no
direct observations of any particular pair. Per-pair direct-context changes
and worst omissions are saved separately.

All checks are conditional on an A0 model already trained with the complete
training fold. They do not test model generalization to unseen words or writers,
joint removal of several groups, or correct physical character boundaries.
They are sensitivity measurements, not confidence intervals. There is no
post-hoc pass threshold, new gate, balancing rule or selected vocabulary.

Artifacts and verification:

- [stability results](../results/thesis/forced_alignment_review/wi_rh_fold0_20261115_stability/)
  contain analysis.py, verify_exports.py, run_manifest.json,
  stability_summary.json, numerical_validation.json, omission_summary.csv,
  pair_sensitivity.csv and omission_pair_scores.csv.gz.
  The last file retains all 422 baseline-observed pair rows per omission,
  with missing evidence explicit. Local results remain Git-ignored.
- The manifest records the exact command, source hashes, code revision,
  uncommitted changes, interpreter, package versions, parameters and input
  hashes. It preserves the pre-run documentation diff and the fixed
  validation-case IDs. No model inference or training was run.
- Ten numerical comparisons passed: the saved Step 1 baseline, the unchanged
  native scorer baseline, and eight count/ID-selected omissions. Maximum
  score difference was 3.33e-16 against the declared 1e-12 tolerance.
  Support, eligibility, eligible-list ranks, fixed-set ranks and correction
  fallback counts all matched.
- The reference scorer used its unchanged calculation with an in-memory
  reader for the already accepted evidence; no production function was edited.
- Independent export checks reconciled all 543 omissions and 229,146 saved
  pair rows. All gate decisions, ranks, pair-sensitivity maxima and gate-loss
  counts reconcile. An independently calculated Spearman coefficient differs
  by at most 2.22e-16. Supplemental eligible-list overlap is saved separately.
- All recorded input hashes match before/after, including Step 1 and August
  outputs, the checkpoint, export, annotations, scoring source and the existing
  untracked compatibility audit. The full test suite was not rerun.
- Original-export provenance limits from Step 1 remain. Current input hashes
  do not retrospectively authenticate an unrecorded August code revision.

Next: Step 3, a fixed sensor-trace audit. Freeze its cases and review rules
before inspection. Do not change the score or construct tokenizers from this
stability check alone. Steps 3-6 remain planned.

#### 2026-11-15 — Step 1 completed: baseline reproduced unchanged

- Run directory:
  [wi_rh_fold0_20261115_baseline/](../results/thesis/forced_alignment_review/wi_rh_fold0_20261115_baseline/).
  These local artifacts are Git-ignored. The tracked result summary is here
  and in [the experiment log](EXPERIMENTS.md#2026-11-15--a0-force-baseline-reproduction-and-word-support-audit).
- Ran the unchanged scorer with PYTHONHASHSEED=42 in the installed WSL
  environment, /home/artellisys/miniconda3/envs/tva/bin/python
  (Python 3.11.15). No model inference or training was needed.
- Kept all settings stated in Step 1. The JSON report and full 422-row CSV are
  byte-for-byte identical to the preserved August outputs. Every CSV field,
  score, rank and eligibility flag reconciles.
- Independently checked all 88,292 boundary rows against the training
  annotation index, label, source file, writer, pair position and boundary ID.
  IDs are unique and complete. All rows are WI fold-0 training data, with the
  A0 export recording checkpoint epoch 275 and 100 Hz / 8x downsampling.
- Of 426 raw pairs, 422 retain quality-filtered evidence. There are 78,953
  accepted boundaries across 42 writers. Of these, 2,170 use the marked empty
  region fallback. The baseline support gate admits 176 pairs.
- The original review counted support before filtering: 192 pairs qualify;
  78 occur in three or four words. After filtering, the figures are 176 and
  62 (35.23%). The five-word proposal would leave 114 eligible pairs.
  It is not adopted. No 145-pair vocabulary should be assumed feasible
  under that proposal.
- The run saves word_support.csv, validation_report.json, console.log,
  the pre-run documentation diff, and run_manifest.json. The manifest records
  the exact scorer command, resolved settings, code revision, working-tree
  state, package versions, input/output SHA-256 hashes and checkpoint rule.
- Code revision: b7e4d5f. Before scoring, only this alignment document had
  tracked changes. The existing untracked TOKENIZER_COMPATIBILITY_AUDIT.md
  stayed untouched. No scoring or model code changed.
- All recorded input hashes, including the checkpoint, export, annotations,
  configuration and old ranking/report, match before and after the run.
- Provenance limit: the original export records a checkpoint path and epoch,
  but no checkpoint hash or original source revision. Current hashes identify
  the local files now; they do not retrospectively prove the August inputs.
- Interpretation: reproducibility and word support are verified. Physical
  timing accuracy, true inter-letter contact and ranking stability remain
  unverified. Scores are development rankings, not calibrated probabilities.
- Next: specify Step 2's leave-one-word/leave-one-writer calculations and
  common-pair reporting, then run them with the baseline unchanged. Steps
  2-6 remain planned. No method change, tokenizer or recognition run started.
