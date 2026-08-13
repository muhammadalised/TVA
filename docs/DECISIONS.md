# Thesis Method Decisions

This file records decisions that affect the scientific meaning or scope of the
thesis. Revisit a decision only when new evidence or supervisor guidance makes
the reason no longer valid.

## D001 — Tokenize labels, not IMU input

- Status: confirmed by supervisor
- Decision: The recognizer receives raw 13-channel IMU signals. Tokenization is
  applied to output text labels and CTC ground truth.
- Reason: This corrects the original misunderstanding and preserves the TVA
  recognition architecture.

## D002 — Use right-handed WD and WI data

- Status: confirmed by supervisor
- Decision: Use the right-handed OnHW-Words500 writer-dependent and
  writer-independent datasets. Exclude left-handed data.
- Reason: The available left-handed dataset is too small for the planned
  comparison.

## D003 — Develop matched handwriting-aware tokenizer families

- Status: clarified on 2026-08-02
- Decision: Plan handwriting-aware Bigram, BPE, and Unigram variants and compare
  each with the corresponding linguistic tokenizer at matched vocabulary size.
- Reason: The thesis idea is to train established tokenizer families from
  handwriting patterns instead of linguistic patterns, not merely to create a
  single pair vocabulary.

## D004 — Use a character CTC model for forced alignment

- Status: planned
- Decision: Use target-constrained CTC Viterbi forced alignment from a trained
  character baseline to estimate character intervals in training recordings.
- Reason: Word labels are available, but true character timestamps are not.
  Linguistic tokenizers are not required for this alignment stage.

## D005 — Motion is primary; frequency is a reliability condition

- Status: planned
- Decision: Rank or merge candidates primarily by aligned motion continuity.
  Use occurrence count and writer coverage to reject unreliable estimates.
- Reason: Completely ignoring sample support would allow rare, noisy candidates
  to dominate, while ranking primarily by frequency would revert to linguistic
  tokenization.

## D006 — Select a checkpoint within one architecture by validation CER

- Status: revised on 2026-08-09
- Decision: Prefer `best_cer.pth` when choosing among checkpoints from the same
  alignment architecture. Use `latest.pth` when a best-CER model is unavailable
  and document that exception. CER alone must not choose between architectures.
- Reason: Character accuracy is relevant to forced alignment, but the WI timing
  diagnostic showed that a highly confident bidirectional model can still place
  character emissions too early to represent physical handwriting boundaries.

## D007 — Keep image OCR evaluation optional

- Status: planned extension
- Decision: DTLR plus connected-component analysis may be used for optional
  cross-modal validation only after the IMU study works.
- Reason: Static connected ink is not temporal motion continuity and should not
  displace the core IMU contribution.

## D008 — Final vocabulary sizes

- Status: unresolved
- Current plan: Compare linguistic and handwriting-aware variants at matched
  sizes. Screen a limited set on fold 0, then predeclare a small final set before
  cross-validation.
- Reason: Vocabulary size affects recognition independently of the token-learning
  method, but a large grid across three families, two dataset settings, and five
  folds would make the experiment matrix unnecessarily expensive.

## D009 — Validate a unidirectional recurrent decoder for alignment

- Status: accepted for tokenizer-evidence development on 2026-08-13
- Decision: Keep the BLConv-B + BiLSTM-B character model as the recognition
  baseline. Use the separately trained BLConv-B + UniLSTM-B A0 model for
  fold-0 alignment and tokenizer-evidence development.
- Reason: In the reliable WI training subset, `only` and `first` boundaries had
  a median centre time of 80 ms and were about 315 and 300 ms earlier than the
  rough uniform references. Their median blank duration was zero. This makes
  the current word-initial sensor windows unsuitable for continuity scoring.
- Limitation: BLConv uses centred convolutions and sequence-wide instance
  normalization, so A0 is not fully causal end to end. It is a controlled test
  of recurrent direction.
- Evidence: A0 changed reliable median timing offsets for `only`/`first`
  boundaries from approximately -315/-300 ms to -65/-47 ms. It retained
  78,953 reliable boundaries (89.4% of all 88,292), spanning all 42 writers.
  Its validation CER was worse than B0, so A0 is an alignment model and does
  not replace the recognition baseline.

## D010 — Retain midpoint and whole-region boundary measurements

- Status: implemented for comparison on 2026-08-13
- Decision: Store the existing 50/100/150 ms midpoint windows and a second set
  of features over the complete CTC candidate region. If the candidate region
  is empty, use a clearly marked 100 ms midpoint fallback instead of silently
  treating an empty array as physical continuity.
- Reason: A0 provides plausible candidate regions, but their duration varies
  substantially. A small window can miss a pen lift near one edge, while the
  complete region can include unrelated motion. Keeping both makes this choice
  testable through descriptive comparison and later ablation.

## D011 — Correct provisional force scores for position and region duration

- Status: implemented for fold-0 development on 2026-08-13
- Decision: Use the corrected 100 ms contact-preservation component as the
  primary force evidence and the corrected complete-region component as
  secondary evidence. Estimate expected contact separately for boundary
  position and broad region-duration groups, then aggregate residual evidence
  with equal weight per writer.
- Development defaults: 75% local and 25% complete-region evidence; at least
  100 reliable occurrences from at least 30 writers; at least 100 occurrences
  before a position-duration baseline is used.
- Reason: Complete-region analysis finds pressure losses missed by a local
  window, but longer regions and word-initial positions show lower apparent
  continuity independent of pair identity. Directly ranking raw region rates
  would therefore reward or penalize some pairs for alignment geometry.
- Limitation: This is the first interpretable force-only baseline. Weighting,
  force threshold, duration bins, support gates, motion components, and their
  ablations remain development choices rather than frozen thesis settings.
