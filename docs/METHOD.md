# Planned Handwriting-Aware Tokenization Method

This document describes the current technical plan. Exact feature definitions,
thresholds, and vocabulary sizes remain experimental parameters until they are
validated and frozen.

## Inputs and outputs

The recognizer input remains a raw 13-channel IMU word recording. The changed
research variable is the token sequence used as the CTC target. Every token
maps to a text string, and decoded token strings are concatenated before CER and
WER are calculated.

## 1. Character alignment

For each fold, train a character CTC alignment model from that fold's training
data. Run target-constrained CTC Viterbi forced alignment using each training
recording and its known word label. The alignment must support CTC blanks and
repeated characters and provide character intervals plus a confidence measure.

Map encoder-frame positions back to approximate raw-signal positions using the
model's temporal downsampling ratio. Exclude or separately analyze low-confidence
alignments rather than treating all estimated boundaries as equally reliable.

Implementation status on 2026-08-08: the standalone target-constrained Viterbi
algorithm supports repeated characters and is connected to real checkpoints.
The single-sample runner now reports local probability diagnostics, maps
emission anchors approximately to model-input samples, identifies intervening
blank-frame regions, and saves synchronized sensor/confidence plots plus JSON.
Manual validation and the final alignment-quality filter remain open. See
`docs/FORCED_ALIGNMENT.md` for the interface and limitations.

The full WI timing diagnostic later found that the bidirectional B0 model
usually placed `only` and `first` boundaries at 80 ms, about 315 and 300 ms
earlier than the rough uniform references. High local confidence did not expose
this shift. The separate A0 BLConv-B + UniLSTM-B alignment model reduced these
reliable median offsets to about 65 and 47 ms early while retaining 78,953 of
88,292 boundaries in the transparent quality subset. A0 is therefore used for
fold-0 tokenizer-evidence development. BLConv can still use neighbouring
samples and sequence-wide instance-normalization statistics, so A0 is not
fully causal or assumed ground truth.

The batch exporter processes the training partition without augmentation and
writes one occurrence record per adjacent-character boundary. Its scientific
default is one sample per inference call because padding unequal word lengths
can otherwise influence a bidirectional model. Interrupted exports are
resumable at sample boundaries.

## 2. Boundary features

For every adjacent-character boundary, retain two transparent views of the
same candidate:

- fixed 50, 100, and 150 ms windows around the candidate-region midpoint; and
- the complete CTC interval between the neighbouring character emissions.

The complete interval can reveal a force loss near an edge that a midpoint
window misses. Conversely, a long interval can include unrelated internal
motion, so neither view is assumed correct before comparison. If CTC emits two
characters in consecutive frames, the complete interval is empty; extraction
then uses a marked 100 ms midpoint fallback and reports its usage rate.

A recording-adaptive provisional low-force threshold uses 10% of the
recording's 90th-percentile raw force value. This is an inspectable development
rule, not a frozen thesis threshold.

Candidate feature groups are:

- force/contact: low-force fraction, longest low-force duration, and force
  change around the boundary;
- accelerometer: changes and derivative energy from AF and AR channels;
- gyroscope: changes in rotational motion and angular derivative energy;
- pause/energy: duration of unusually low movement energy; and
- learned representation: similarity of encoder features before and after the
  boundary.

Raw or lightly filtered force should be used for contact analysis because
per-word normalization removes absolute force meaning. Contact thresholds may
need writer- or recording-level calibration.

## 3. Boundary-continuity score

Normalize heterogeneous feature values using training-fold statistics. Define
a score in `[0, 1]`, where a larger value means stronger evidence of continuity.
An interpretable starting point is a weighted combination:

```text
C = w_force*C_force + w_acc*C_acc + w_gyro*C_gyro + w_pause*C_pause
```

Initial weights, thresholding, and feature groups must be evaluated through
ablation rather than presented as known facts. Planned comparisons include
force only, motion without force, and combined evidence.

For every character pair, aggregate occurrence-level scores using robust
statistics such as median and interquartile range. Retain count, alignment
confidence, and writer coverage. All statistics and thresholds must come only
from the fold's training partition.

The implemented descriptive analysis reports all occurrences alongside a
transparent agreement/non-padding/unclipped subset. It calculates means,
standard deviations, ranges, and the 10th, 25th, 50th, 75th, and 90th
percentiles for alignment, contact, and motion measurements at every window
size. No probability threshold, margin threshold, window choice, feature
weight, or continuity cutoff is selected by this analysis.

Before ranking pairs, the same occurrence features are summarized by boundary
position (`only`, `first`, `middle`, or `final`), left-character case, and their
cross-product. This diagnostic checks whether apparently continuous uppercase
pairs are actually explained by their usual word-initial position. The groups
do not alter or reweight the exported measurements.

The position analysis also compares the normalized estimated boundary centre
with a rough evenly spaced reference. For boundary index `k` in a word with
`B` boundaries, the reference is `(k + 1) / (B + 1)`. Signed relative and
millisecond offsets reveal whether the CTC anchors are systematically early or
late; absolute relative error describes offset size. This is explicitly not
ground truth, because character durations vary. It is only a diagnostic for
temporal localization bias before boundary evidence is turned into tokenizer
scores.

## 4. Handwriting-aware Bigram

Keep all characters as fallback tokens. Reject pairs below minimum occurrence,
alignment-confidence, or writer-coverage requirements. Rank the remaining
pairs primarily by continuity and consistency, then add the highest-ranked
pairs until the target vocabulary size is reached.

## 5. Handwriting-aware BPE

Start from characters. At each iteration, score candidate merges using motion
cohesion and reliability rather than raw pair frequency. For a multi-character
candidate, derive cohesion from its internal aligned boundaries. The initial
rule should use the weakest internal boundary; mean and geometric mean are
possible ablations. Repeat until the matched vocabulary size is reached.

## 6. Handwriting-aware Unigram

Generate bounded-length substring candidates from training labels. Assign each
candidate a utility based on internal motion cohesion, consistency, support,
writer coverage, and a documented length penalty. Retain every character,
iteratively prune weak candidates, and use dynamic programming to select the
best tokenization of each label.

Because conventional Unigram training is probabilistic and text-statistical,
the exact proposed algorithm should be called motion-aware or
motion-regularized Unigram, with its objective stated explicitly.

## 7. Fair comparisons

Train a fresh BLConv-B + BiLSTM-B + CTC model for every tokenizer. Match each
handwriting-aware vocabulary size with its linguistic counterpart. Keep data
splits, input preprocessing, augmentation, optimizer, schedule, epochs, seed,
batch size, and decoding/evaluation constant.

The planned conditions are character, linguistic Bigram, handwriting-aware
Bigram, linguistic BPE, handwriting-aware BPE, linguistic Unigram, and
handwriting-aware Unigram. Develop on fold 0, freeze the method, then perform
the agreed five-fold WD/RH and WI/RH evaluations.
