# Thesis Proposal Plan

This document mirrors the approved proposal, *Handwriting-Aware Tokenization
for IMU-Based Online Handwriting Recognition*. The proposal is the authority
for thesis scope; implementation notes elsewhere must not silently broaden or
reverse it.

## Problem and objective

Conventional Bigram, BPE, and Unigram tokenizers form text-label tokens mainly
from linguistic frequency. A character group can therefore become one output
token even when its characters are normally written with separate movements.
The thesis investigates whether visible handwriting structure provides more
suitable output labels for IMU-based online handwriting recognition.

The recognizer continues to receive raw 13-channel IMU signals. Tokenization
changes only its output vocabulary and CTC target labels.

## Primary research method

Use offline handwriting images to estimate how often neighbouring characters
are visibly connected:

1. Prepare the IAM English and READ German datasets and analyse them
   separately.
2. Use a pretrained DTLR model to obtain character identities and bounding
   boxes.
3. Match the detected characters and boxes to each known transcription.
4. Exclude uncertain localizations rather than treating them as evidence.
5. Binarize the accepted writing and apply connected-component labelling
   (CCL).
6. Determine whether the ink associated with each adjacent character pair is
   connected and aggregate the observations into pair-connectivity scores.
7. Require sufficient occurrences and writer support for a score to be
   eligible for tokenizer construction.

IAM and READ must produce separate evidence tables and results. Combining them
may be studied only as an explicit additional condition.

This is cross-modal transfer: static-image ink connectivity is a handwriting
prior for an IMU recognizer. It is not proof of continuous pen motion in any
particular IMU recording, and CCL components are not character identities by
themselves.

## Tokenizer development

Construct and analyse a handwriting-aware Bigram vocabulary first. Retain all
individual characters as fallback tokens and compare the vocabulary with a
frequency-based Bigram baseline of the same or nearly the same size.

Extend the connectivity evidence to handwriting-aware BPE and Unigram only if
the Bigram results are promising. Each proposed family must be compared with
its corresponding frequency-based tokenizer. The precise rules for deriving
longer-token cohesion from internal pair scores must be documented before
those experiments.

## Recognition evaluation

Encode right-handed OnHW-Words500 labels with each tokenizer and train a
separate BLConv-B + BiLSTM-B + CTC recognizer from scratch. Compare
character-level, frequency-based, and handwriting-aware labels separately in:

- writer-dependent (WD/RH) evaluation; and
- writer-independent (WI/RH) evaluation.

Keep raw IMU input, architecture, data splits, preprocessing, augmentation,
training settings, and decoding/evaluation fixed. Reconstruct ordinary text
before calculating character error rate (CER) and word error rate (WER).

Development may screen a small set of choices on fold 0. Final conclusions
require the agreed five-fold evaluation, with the method and vocabulary sizes
frozen in advance.

## Hypotheses

1. Tokenizers trained with handwriting patterns will outperform corresponding
   tokenizers trained using only linguistic frequency information.
2. The benefit of handwriting-aware tokenization will be greater in the
   writer-independent setting than in the writer-dependent setting.

These are hypotheses to test, not assumed outcomes.

## Leakage control

The image datasets provide external handwriting priors rather than OnHW
validation evidence. Do not use OnHW validation recordings or recognition
scores to construct a fold's tokenizer, choose pair thresholds, or tune its
connectivity weights. Record the exact image corpus split, DTLR checkpoint,
matching rules, uncertainty filters, binarization/CCL settings, vocabulary,
and code commit for every experiment.

## Optional and fallback studies

- If time and resources permit, evaluate the same tokenizers with a fixed
  offline image-recognition model to test whether they also help directly in
  the image domain.
- If the image method is unreliable and time remains, use the implemented CTC
  forced-alignment pipeline on OnHW as an additional comparison. Force,
  acceleration, and rotation around estimated boundaries may provide an
  IMU-derived continuity score, but those alignments are not physical-boundary
  ground truth.

Neither optional study replaces the primary image-to-IMU transfer experiment.

## Milestone order

1. Prepare IAM and READ and obtain reliable DTLR character localizations.
2. Implement and validate transcript matching, binarization, CCL, and
   pair-connectivity aggregation.
3. Construct and analyse the handwriting-aware Bigram vocabulary.
4. Train matched Bigram recognition comparisons on WD/RH and WI/RH.
5. Extend to BPE and Unigram only if Bigram results are promising.
6. Complete the five-fold comparison and thesis analysis.
7. Attempt the optional image-recognition or IMU-alignment comparison only if
   time and evidence justify it.

## Main risks and controls

- **Incorrect DTLR/transcript matching:** reject uncertain samples and audit a
  representative set manually.
- **CCL ambiguity:** remember that multiple characters can share one component
  and one character can contain several; define the box-to-component rule
  explicitly and report edge cases.
- **Corpus/language effects:** analyse IAM and READ separately and report
  coverage for every candidate pair.
- **Rare candidates:** use minimum occurrence and writer-support gates without
  turning frequency into the primary ranking criterion.
- **Too many training conditions:** evaluate Bigram first and expand to BPE and
  Unigram only after a promising result.
