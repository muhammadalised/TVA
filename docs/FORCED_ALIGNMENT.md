# CTC Forced Alignment

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

`align_sample.py` connects the alignment algorithm to a trained TVA character
model and a real dataset sample. It deliberately processes only one sample so
checkpoint loading, preprocessing, model inference, greedy decoding, and
forced alignment remain easy to inspect.

Run WI/RH fold 0 with its best-CER checkpoint:

```bash
python align_sample.py \
  --config configs/thesis/b0_char_wi_rh.yaml \
  --checkpoint results/thesis/baselines/B0_char_wi_rh/0/checkpoints/best_cer.pth \
  --split val \
  --sample-index 0 \
  --device cuda
```

Run WD/RH fold 0 with the retained final checkpoint:

```bash
python align_sample.py \
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

`export_boundaries.py` applies the same alignment and feature extraction to a
complete dataset split. The default split is `train`, because tokenizer
statistics must be learned without looking at validation recordings. It writes
one JSONL record per adjacent-character occurrence, plus a compact summary and
a progress file.

Run the selected A0 WI/RH training-fold export on the RTX machine:

```bash
python export_boundaries.py \
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

`analyze_boundaries.py` reads the completed JSONL export and calculates exact
descriptive statistics globally and separately for every character pair. It is
a CPU analysis and does not run the recognition model again:

```bash
python analyze_boundaries.py \
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

`score_continuity.py` reads the completed training JSONL directly. It does not
run the recognizer or require a GPU:

```bash
python score_continuity.py \
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
complete-region contact. It is a force-only development baseline for the first
handwriting-aware Bigram experiment, not the final combined force/motion score.

## Next development steps

1. Run the corrected force scorer on the complete A0 WI training export and
   inspect its top, middle, and bottom supported pairs.
2. Check ranking sensitivity to local/region weights, duration bins, and the
   support gate without using validation or test recognition results.
3. Define a motion-only component from accelerometer and gyroscope evidence.
4. Compare force-only, motion-only, and combined scores before constructing the
   first handwriting-aware Bigram vocabulary.
