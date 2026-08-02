# Thesis Proposal Plan

## Working title

**Handwriting-Aware Label Tokenization for IMU-Based Online Handwriting
Recognition**

## Problem and motivation

IMU handwriting recognizers commonly predict individual characters or use
subword tokenizers learned from text frequency. Linguistic frequency does not
necessarily describe how characters are physically connected during writing.
The thesis investigates whether token vocabularies learned from handwriting
motion can provide more suitable CTC output labels.

## Objective

Develop handwriting-aware Bigram, BPE, and Unigram tokenizers whose candidate
selection or merge decisions are guided primarily by motion continuity, then
compare them fairly with character labels and corresponding linguistic
tokenizers.

## Research question

Can handwriting-aware Bigram, BPE, and Unigram text-label tokenizers learned
from IMU motion continuity improve character and word recognition compared with
character labels and linguistically trained tokenizers?

## Planned approach

1. Train the existing BLConv-B + BiLSTM-B + CTC character baseline on raw
   13-channel OnHW-Words500 IMU recordings.
2. Use target-constrained CTC forced alignment to estimate character intervals
   in training recordings.
3. Measure continuity around adjacent-character boundaries using force,
   accelerometer, gyroscope, pause/energy, and possibly encoder features.
4. Aggregate reliable motion evidence across occurrences and writers.
5. Use the shared evidence to train handwriting-aware Bigram, BPE, and Unigram
   text-label tokenizers.
6. Retrain the same recognizer from scratch for every tokenizer and evaluate
   reconstructed text with CER and WER.

## Experimental comparison

The main comparison contains character, linguistic Bigram, handwriting-aware
Bigram, linguistic BPE, handwriting-aware BPE, linguistic Unigram, and
handwriting-aware Unigram conditions. Linguistic and handwriting-aware variants
within each family will use matched vocabulary sizes and otherwise identical
training settings. Candidate vocabulary sizes may be screened during fold-0
development; a small final set will be fixed before cross-validation.

Experiments use the right-handed writer-dependent and writer-independent
OnHW-Words500 splits. Development begins on fold 0; final conclusions will use
the agreed five-fold evaluation. Tokenizers and motion thresholds are learned
from each training partition only to prevent validation leakage.

## Expected contribution

The thesis will contribute a method for converting aligned handwriting-motion
patterns into text-token vocabularies, adaptations of established tokenizer
families to this motion evidence, and a controlled evaluation of whether
handwriting-aware tokenization adds value beyond linguistic token statistics.

## Scope

The raw IMU signal remains the model input; the thesis changes output-label
tokenization. Right-handed data is the core dataset. Optional DTLR and connected
component experiments on offline handwriting images are cross-modal extensions
and will be attempted only after the IMU comparison is complete.

## Staged implementation

Implement and validate character alignment and motion scoring first, followed
by handwriting-aware Bigram, BPE, and Unigram in that order. This staging
reduces technical risk while preserving all three tokenizer families in the
planned thesis objective.

## Six-month work plan

| Period | Planned work |
| --- | --- |
| Month 1 | Complete character baselines; implement and validate CTC forced alignment |
| Month 2 | Define continuity features and ablations; implement handwriting-aware Bigram |
| Month 3 | Implement handwriting-aware BPE and motion-aware Unigram |
| Month 4 | Run fold-0 matched comparisons; refine and freeze the methodology |
| Month 5 | Run final WD/RH and WI/RH cross-validation experiments; optional image study only if core work is complete |
| Month 6 | Analyze results, document limitations, write and revise the thesis |

## Main risks and mitigations

- **Noisy character alignments:** use target-constrained Viterbi alignment,
  confidence filtering, and manual inspection of representative examples.
- **Writer-dependent force levels:** calibrate contact evidence per writer or
  recording and evaluate force-only, motion-only, and combined scores.
- **Rare token candidates:** require minimum count and writer coverage while
  keeping motion continuity as the primary selection evidence.
- **Too many experimental combinations:** develop on fold 0, freeze decisions
  before cross-validation, and keep the image experiment optional.
