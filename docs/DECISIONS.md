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

- Status: confirmed by the September 2026 proposal, with staged scope
- Decision: Plan handwriting-aware Bigram, BPE, and Unigram variants and compare
  each implemented family with the corresponding linguistic tokenizer at
  matched vocabulary size. Bigram is required first; BPE and Unigram follow
  only if the Bigram results are promising.
- Reason: The thesis idea is to train established tokenizer families from
  handwriting patterns instead of linguistic patterns, not merely to create a
  single pair vocabulary.

## D004 — Use a character CTC model for forced alignment

- Status: superseded as the primary method; retained as fallback/comparison
- Decision: Use target-constrained CTC Viterbi forced alignment from a trained
  character baseline to estimate character intervals in training recordings
  only if the image method proves unreliable and time remains, or as an
  explicitly secondary comparison.
- Reason: Word labels are available, but true character timestamps are not.
  Linguistic tokenizers are not required for this alignment stage.

## D005 — Motion was primary; frequency is a reliability condition

- Status: superseded by D012 for the primary study
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

## D007 — Keep image-domain recognition evaluation optional

- Status: revised by the September 2026 proposal
- Decision: Using a fixed offline recognizer to evaluate the finished
  tokenizers in the image domain is optional. DTLR plus connected-component
  analysis for constructing the tokenizers is not optional; it is the primary
  evidence pipeline.
- Reason: The proposal distinguishes primary image-derived tokenizer
  construction from optional image-domain recognition evaluation.

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

## D012 — Use image connectivity as the primary handwriting prior

- Status: confirmed by the approved September 2026 proposal
- Decision: Analyse IAM English and READ German handwriting separately. Use a
  pretrained DTLR model for character identities and bounding boxes, match the
  output to known transcriptions, exclude uncertain localizations, and use
  binarization plus connected-component labelling to estimate adjacent-pair
  ink connectivity. Aggregate the accepted observations with occurrence and
  writer-support requirements.
- Reason: The approved contribution is a transparent cross-modal method that
  constructs output-label tokenizers from visible handwriting structure rather
  than primarily from linguistic frequency.
- Limitation: Connected ink in a static image is a handwriting prior, not proof
  of continuous pen motion in an OnHW recording. CCL components are not
  characters without the localization and transcript-matching context.

## D013 — Keep corpora and experiment stages distinct

- Status: confirmed by the approved September 2026 proposal
- Decision: Produce separate IAM- and READ-derived connectivity analyses.
  Construct and evaluate Bigram first. Extend to BPE and Unigram only if the
  Bigram outcome is promising. Treat any pooled IAM+READ evidence as a separate
  documented condition.
- Reason: Separate analysis exposes language and dataset effects, while staged
  tokenizer development controls experimental cost and technical risk.

## D014 — Test the proposed WI advantage explicitly

- Status: confirmed by the approved September 2026 proposal
- Decision: Evaluate the same controlled tokenizer comparisons separately in
  WD/RH and WI/RH and test whether the handwriting-aware benefit is larger in
  WI. Do not assume that outcome from fold-0 development.
- Reason: A stronger benefit in writer-independent recognition is one of the
  proposal's two stated hypotheses.

## D015 — Do not load the combined artifact through TVA's greedy tokenizer

- Status: compatibility finding confirmed on 2026-09-15; adapter policy frozen
  on 2026-09-17
- Decision: Integrate the frozen combined IAM+READ model through a dedicated
  tokenizer that preserves its NFC normalization and maximum-total-utility
  dynamic-programming segmentation. Do not load it through TVA's existing
  greedy `BigramTokenizer`. Add missing OnHW characters only as fallback
  singles and never derive new handwriting bigrams from OnHW labels.
- Reason: `Ä` and `Ü` are absent from the frozen vocabulary, affecting 608 of
  25,199 OnHW samples. Greedy and frozen-model segmentations differ for 9,009
  of 24,591 otherwise encodable labels, so direct loading would silently test a
  different tokenizer.
- Leakage rule: A compatibility projection may use the fixed 59-character
  OnHW task alphabet, but not OnHW label frequencies, validation-label
  presence, or recognition results. The same frozen handwriting adapter must
  be used for every WD/WI fold. Matched linguistic vocabularies remain
  fold-specific and training-only.
- Frozen primary adapter: policy `onhw-words500-rh-iam-read-v1`, containing
  419 classes: blank ID 0, the fixed 59 OnHW characters in configuration order,
  and 359 frozen bigrams whose two characters belong to that alphabet. Source
  bigram order and utilities are preserved. Retaining the complete artifact
  and appending `Ä`/`Ü` would give 496 classes; that is excluded from the
  primary experiment and may be used only as a separately predeclared
  sensitivity analysis.
- Comparator caveat: The matched linguistic condition's segmentation policy
  must be predeclared. Equal output size does not isolate vocabulary membership
  if the linguistic tokenizer remains greedy while the handwriting tokenizer
  uses utility-maximizing dynamic programming.
- Evidence and exact construction:
  `docs/TOKENIZER_COMPATIBILITY_AUDIT.md` and
  `docs/HANDWRITING_BIGRAM_ADAPTER_V1.md`. The canonical adapter SHA-256 is
  `12ce25d8bedc552e6b3497ffb1d07e506b01b34296cc21b970f82a550cbf2bfe`.

## D016 — Match the linguistic baseline's segmentation algorithm

- Status: frozen on 2026-09-18 before recognition training
- Decision: Build 419-class linguistic Bigram tokenizers separately from each
  WD/WI fold's training annotations. Collapse repeated samples to distinct word
  types, rank adjacent pairs by descending type-weighted occurrence count with
  lexical tie-breaking, and retain 359 pairs. Use normalized pair frequency as
  utility in the same maximum-total-utility dynamic program as the handwriting
  tokenizer.
- Reason: Using the existing greedy tokenizer would confound evidence source
  with segmentation algorithm. Distinct-word weighting follows TVA's previous
  Bigram construction and avoids treating writer/sample multiplicity as
  linguistic evidence.
- Leakage rule: Read `train.json` only. Never inspect validation labels or
  recognition results during construction. Validation may be used afterward
  only for compatibility checks.
- Interpretation: The controlled factor is the source of pair evidence and
  utility—training-text frequency versus IAM+READ handwriting connectivity.
  This is not a token-membership-only comparison because the utilities also
  differ by evidence source.
- Limitation: The matched 359-bigram size reaches a count-one tie in every
  fold, so 87–97 selected tail pairs are chosen lexically from larger tied
  groups. This deterministic but weakly supported tail must be reported.
- Contingency: If DP-based training is unsuccessful, greedy left-to-right
  segmentation may be evaluated for both handwriting and linguistic
  vocabularies as a separately named secondary ablation. It cannot replace or
  suppress the predeclared primary DP result based on observed performance.
- Specification: `docs/LINGUISTIC_BIGRAM_BASELINE_V1.md`.
