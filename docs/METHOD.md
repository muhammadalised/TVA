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

For each fold, train the B0 character CTC model from that fold's training data.
Run target-constrained CTC Viterbi forced alignment using each training
recording and its known word label. The alignment must support CTC blanks and
repeated characters and provide character intervals plus a confidence measure.

Map encoder-frame positions back to approximate raw-signal positions using the
model's temporal downsampling ratio. Exclude or separately analyze low-confidence
alignments rather than treating all estimated boundaries as equally reliable.

Implementation status on 2026-08-08: the standalone target-constrained Viterbi
algorithm is implemented in `tva/ctc_alignment.py` and tested on transparent
toy cases, including repeated characters. Real-checkpoint integration,
output-to-IMU frame mapping, visualization, and confidence filtering remain the
next validation steps. See `docs/FORCED_ALIGNMENT.md` for the interface.

## 2. Boundary features

For every adjacent-character boundary, examine fixed windows before and after
the estimated position. Candidate window sizes include 50, 100, and 150 ms at
the dataset's 100 Hz sample rate.

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
