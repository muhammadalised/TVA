# Planned Handwriting-Aware Tokenization Method

This document describes the technical method implied by the approved thesis
proposal. Exact thresholds, localization filters, connectivity definitions,
and vocabulary sizes remain experimental parameters until validated and
frozen.

## 1. Inputs and outputs

The final recognizer input is a raw 13-channel OnHW-Words500 IMU word
recording. The research variable is the text-token sequence used as its CTC
target. Every token maps to a text string; decoded token strings are
concatenated before CER and WER are calculated.

Offline images supply evidence for selecting the token vocabulary. They do not
replace, segment, or augment the IMU input.

## 2. Image evidence datasets

Use IAM for English handwriting and READ for German handwriting. Prepare and
analyse the two datasets separately so language, style, annotation, and writer
coverage differences remain visible. For every accepted sample, retain image,
transcription, writer identity when available, dataset identity, and split
provenance.

Do not silently pool IAM and READ. A combined evidence condition, if useful,
is a separately named experiment with a documented weighting rule.

## 3. Character localization and transcript matching

Run a pretrained DTLR model to obtain character identities and bounding boxes.
Match its ordered output to the known transcription before extracting
connectivity evidence. Store localization confidence and the reason for any
rejection.

The matching stage must define how it handles insertions, deletions,
substitutions, repeated characters, spaces, punctuation, case, and characters
outside the OnHW vocabulary. Samples or boundaries with uncertain identity or
ordering must be excluded under a predeclared rule.

Before large-scale extraction, manually audit a representative sample from
both datasets, including cursive joins, detached marks such as the dot of `i`,
overlapping boxes, touching words, and low-quality scans.

## 4. Binarization and connected-component evidence

Binarize each accepted handwriting image and run connected-component labelling
on the ink mask. Relate components to adjacent DTLR character boxes using an
explicit, testable association rule.

For each adjacent transcript pair, record at least:

- whether the two localized character regions share connected ink;
- the relevant component identifiers and box geometry;
- a gap or component-distance measure for disconnected ink;
- DTLR/transcript-matching confidence;
- dataset and writer provenance; and
- exclusion or ambiguity flags.

CCL identifies connected regions of pixels, not characters. Several cursive
characters may share a component, while one character may contain multiple
components. The method must therefore use DTLR identities/boxes and must not
interpret component count alone as character segmentation.

## 5. Pair-connectivity aggregation

Aggregate accepted adjacent-character observations separately for IAM and READ.
For every case-sensitive pair, report occurrence count, writer coverage,
connected proportion, uncertainty/exclusion rates, variability across writers,
and any secondary gap statistic.

Give writers equal influence where writer IDs permit this, so prolific writers
do not dominate a pair. Minimum occurrence and writer-support thresholds are
reliability gates. Frequency must not become the primary reason a pair receives
a high handwriting score.

Freeze all image-processing settings and evidence tables before using OnHW
recognition validation results to compare tokenizers.

## 6. Handwriting-aware Bigram

Retain every individual task-alphabet character as a fallback token. Rank
eligible adjacent character pairs primarily by image-derived connectivity,
with a documented tie-breaking rule. Add the highest-ranked pairs until the
target vocabulary size is reached.

The prepared combined IAM+READ model does not use left-to-right greedy
matching. It chooses non-overlapping bigrams by maximum-total-utility dynamic
programming, with single-character utility zero and a deterministic preference
for more bigrams when total utility ties. TVA must preserve that algorithm for
the handwriting-aware condition. Its existing greedy Bigram implementation is
appropriate only for a separately defined comparator, not for loading the
frozen handwriting model.

Before training, project or extend the frozen vocabulary using a predeclared
task-alphabet rule so every OnHW character remains representable. This rule may
use the fixed 59-character OnHW alphabet but must not use label frequencies or
validation-label presence. The 2026-09-15 coverage and segmentation audit is
summarized in `docs/PROGRESS.md`. The resulting 419-class primary policy was
frozen on 2026-09-17; its exact deterministic construction is in
`docs/HANDWRITING_BIGRAM_ADAPTER_V1.md`.

Create a frequency-based Bigram vocabulary of the same or nearly the same size
as the direct comparator. First analyse token coverage and composition, then
train the recognition models. The frozen comparator uses 359 pairs selected
from distinct training word types, normalized frequency as utility, and the
same dynamic program as the handwriting condition. This controls the
segmentation algorithm while changing the evidence used for pair membership
and utility. Exact rules, fold artifacts, checksums, and the count-one cutoff
limitation are in `docs/LINGUISTIC_BIGRAM_BASELINE_V1.md`.

## 7. Conditional BPE and Unigram extensions

Proceed only if the Bigram experiment is promising.

For handwriting-aware BPE, begin with characters and use image-connectivity
evidence to guide successive merges instead of selecting merges primarily by
frequency. For a multi-character candidate, define cohesion transparently from
its internal adjacent-pair scores.

For handwriting-aware Unigram, generate bounded-length candidates, score their
internal connectivity and reliability, preserve all characters, prune weak
candidates, and use a documented segmentation objective. Because conventional
Unigram training is probabilistic and text-statistical, state precisely how
the proposed objective differs.

Both families require matched frequency-based baselines and explicit rules for
support, length, internal-boundary aggregation, and deterministic evaluation.

## 8. Recognition experiment

For every tokenizer, train a fresh BLConv-B + BiLSTM-B + CTC model on the same
raw OnHW recordings. Keep architecture, preprocessing, augmentation, optimizer,
learning-rate schedule, epochs, batch size, seed, split, and decoding constant
unless a change is the named subject of an ablation.

Evaluate right-handed WD and WI settings separately. Development begins on
fold 0; final claims require five-fold evaluation. Report CER and WER on
reconstructed plain text and preserve all configurations, vocabularies,
checkpoints, predictions, code commits, and dataset/evidence versions.

The core comparison is character versus matched frequency-based and
image-connectivity-based tokenizers. The proposal specifically predicts a
larger benefit in WI than WD, which must be tested rather than assumed.

## 9. Optional image-domain evaluation

If time and resources permit, use a fixed offline handwriting recognizer to
evaluate the same frozen tokenizers on image data. This is a secondary test of
whether handwriting-aware labels help directly in the image domain. It must
not alter the primary IMU-recognition comparison after observing its results.
