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

## D006 — Select the alignment model by validation CER

- Status: implemented for future runs
- Decision: Prefer `best_cer.pth` for forced alignment. Use `latest.pth` when a
  best-CER model is unavailable, and document that exception.
- Reason: Character accuracy is more directly relevant to character-time
  alignment than word-level accuracy.

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
