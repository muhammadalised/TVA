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

- Status: planned
- Scope (2026-11-15): secondary forced-alignment experiment only; see D027.
- Decision: Use target-constrained CTC Viterbi forced alignment from a trained
  character baseline to estimate character intervals in training recordings.
- Reason: Word labels are available, but true character timestamps are not.
  Linguistic tokenizers are not required for this alignment stage.

## D005 — Motion is primary; frequency is a reliability condition

- Status: planned
- Scope (2026-11-15): secondary forced-alignment experiment only; see D027.
- Decision: Rank or merge candidates primarily by aligned motion continuity.
  Use occurrence count and writer coverage to reject unreliable estimates.
- Reason: Completely ignoring sample support would allow rare, noisy candidates
  to dominate, while ranking primarily by frequency would revert to linguistic
  tokenization.

## D006 — Select a checkpoint within one architecture by validation CER

- Status: revised on 2026-08-09
- Scope (2026-11-15): secondary forced-alignment experiment only; see D027.
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
- Construction at other sizes and its limits: D024.

## D009 — Validate a unidirectional recurrent decoder for alignment

- Status: accepted for tokenizer-evidence development on 2026-08-13
- Scope (2026-11-15): secondary forced-alignment experiment only; see D027.
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
- Scope (2026-11-15): secondary forced-alignment experiment only; see D027.
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
- Scope (2026-11-15): secondary forced-alignment experiment only; see D027.
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

## D017 — Freeze a separate greedy IAM-handwriting adapter for ED

- Status: accepted and implemented on 2026-09-21 before ED recognition
  training
- Decision: Keep the DTLR IAM-only 145-bigram vocabulary frozen, project only
  its singleton layer to ED's declared 78-character alphabet, and use greedy
  left-to-right segmentation under the separate TVA key
  `handwriting_bigram_greedy`. The result has 224 classes including blank ID 0.
- Reason: The supervisor requested an English-only handwriting tokenizer for
  the independent ED dataset. Greedy matches TVA's established Bigram
  behavior, while a separate key prevents any retrospective change to the DP
  OnHW condition.
- Leakage rule: The adapter builder may use only the already-frozen IAM source
  and the declared ED character schema. ED label values, frequencies,
  validation results, and recognition scores must not influence handwriting
  bigram membership, ranking, or thresholds. Full labels are read only after
  freezing for compatibility audit and later recognizer training/evaluation.
- Fairness rule: The matched ED linguistic comparator must use the same
  224-class structure and greedy segmentation. Any DP comparison must be a
  separately declared symmetric ablation for both vocabularies.
- Frozen artifact SHA-256:
  `d79062a252ac1ce05e1a309dab9dd439ee06a06e497024cc9321ecce401e40b7`.
- Validation: all 2,550 canonical labels in both WD and WI packages round-trip
  exactly; 5,100 encodings emit zero blank IDs and have zero failures.

## D018 — Match the ED comparator with IAM training-text frequency

- Status: accepted and implemented on 2026-09-21 before recognition training
- Decision: Select 145 linguistic bigrams from the complete 5,694-line IAM
  training transcript split. Count case-sensitive adjacent ASCII-letter pair
  occurrences after NFC normalization, rank by descending count with lexical
  tie-breaking, and use the same ordered ED singletons and greedy segmentation
  as the handwriting condition.
- Reason: Both vocabularies then use the same English corpus and downstream
  alphabet while differing in the intended evidence source: linguistic
  occurrence frequency versus DTLR-derived handwriting connectivity.
- Leakage rule: IAM train is the sole selection source. IAM validation/test
  text, ED labels, and all recognition results are excluded. ED labels are
  used only in the post-freeze compatibility audit and subsequent recognition
  task.
- Matching invariants: both conditions have blank ID 0, 78 identical singleton
  IDs, 145 bigrams, 224 total outputs, NFC normalization, and greedy
  left-to-right segmentation. The vocabularies share 72 pairs.
- Limitation: equal vocabulary size does not imply equal target length. Across
  one canonical 2,550-label ED set, the linguistic vocabulary emits 79,078
  tokens and the handwriting vocabulary emits 85,153. This 7.13% compression
  difference is part of the vocabulary effect and must be reported.
- Frozen evidence SHA-256:
  `7808d5d7b58354be982b042c8f60db4d63bf2ee12aa03cbeca6c3fe628cd1ebb`.
- Frozen adapter SHA-256:
  `1da363e43fe9d300ece7ad5e3183d84a15fab4bdc6d5b2239b450c222b175abd`.

## D019 — Consume ED as authenticated seven-channel ZIP data

- Status: implemented and smoke-tested on 2026-09-21
- Decision: Stream the supplied WD/WI CSV members directly from their ZIP
  archives after authenticating the recorded archive SHA-256. Treat the seven
  columns and 100 Hz rate as already prepared dataset inputs; do not apply an
  OnHW 13-channel selection or a second resampling step.
- Reason: The supplied archives already contain the supervisor-defined folds
  and processed signals. Direct ZIP loading preserves their identity, avoids a
  redundant roughly 500 MB extracted copy, and prevents accidental mixing
  with OnHW preprocessing.
- Preprocessing: verify finite float32 seven-channel arrays, apply TVA's
  existing per-sample/per-channel standardization, and use the same subsequent
  model pipeline for both tokenizers. Dataset augmentation is disabled in
  smoke tests; the production augmentation policy must be frozen before full
  training.
- Portability: configs use `data/tva/ed`; local machines may override
  it with `TVA_ED_DATASET_DIR`. Absolute local paths must not be committed.
- Validation: all four tokenizer × distribution smoke conditions completed
  one bounded CPU epoch and saved 224-output checkpoints. These runs validate
  plumbing only and are not recognition results.

## D020 — Initial matched ED fold-0 production settings

- Status: superseded by D021's A6000 batch-64 deployment setting before any
  production training began
- Decision: Use BLConv-B + BiLSTM-B + CTC, 300 epochs, a 30-epoch linear
  warmup followed by cosine decay, AdamW at learning rate 0.001, seed 42,
  TVA's existing training augmentation, and batch size 8 for all four
  handwriting/linguistic × WD/WI fold-0 conditions. Every model starts from
  scratch and has seven input channels and 224 outputs including blank ID 0.
- Reason: The architecture and optimization schedule match the established
  TVA recognition experiment, while the smaller batch is appropriate for ED
  sentence sequences (mean 2,138 and maximum 7,314 input samples) rather than
  OnHW word sequences. The fold-0 train sets contain 2,047 WD and 2,025 WI
  recordings, so batch 8 still provides a comparable number of optimizer
  updates to the earlier OnHW runs.
- Timing gate: Before starting a 300-epoch job, run one complete WD fold-0
  epoch with the production architecture and batch size. This checks peak CUDA
  memory and gives a wall-clock estimate; its metrics are diagnostics only and
  the checkpoint must not initialize a production run.
- Fairness rule: the four production configurations may differ only in the
  frozen tokenizer, its artifact path, WD/WI distribution, and output
  directory. If batch size must change because the timing run exhausts GPU
  memory, change it identically in all four configs and document the change
  before any full run begins.
- Timing outcome: batch size 8 completed a full WD fold-0 epoch on the RTX
  4060 with CUDA mixed precision: 82 seconds training and 4 seconds
  validation. No out-of-memory failure occurred. Peak VRAM was not captured.

## D021 — Use physical batch size 64 on the A6000

- Status: superseded for the final ED matrix by D022 on 2026-09-24; retained
  as the A6000 systems benchmark
- Trigger: The same complete WD fold-0 batch-8 timing run took 100 seconds for
  training and 2 seconds for validation on the RTX A6000. Observed memory
  peaked at only 2,234 MiB of 48 GB, while one-second GPU-utilization samples
  ranged from 0% to 43%. The stronger GPU was therefore not used efficiently
  by this configuration.
- Decision: use physical batch size 64 for every A6000 production run. Apply
  it identically to every tokenizer, distribution, and fold. The choice was
  made from systems measurements and the predeclared TVA batch convention;
  diagnostic CER/WER was ignored.
- Scientific constraint: changing batch size changes the number of optimizer
  updates per epoch. Throughput alone is not sufficient justification for an
  arbitrarily large batch; the final choice must state the resulting updates
  per epoch and whether epochs, learning rate, or schedule remain unchanged.
- Interpretation: low sampled utilization suggests an input/launch or
  small-batch bottleneck, but the one-second samples do not identify its exact
  cause. This benchmark is a systems result, not recognition evidence.
- Batch-64 outcome: the complete WD fold ran in 16 seconds training plus 1
  second validation, with maximum observed memory 7,248 MiB and sampled GPU
  utilization up to 47%. It completed without an out-of-memory error and is
  5.9 times faster in training than batch 8 on the same workstation.
- Update budget: WD fold 0 has 32 training updates per epoch, or 9,600 over
  300 epochs; WI fold 0 also has 32 updates per epoch. This is fewer updates
  than the provisional batch-8 setting, but batch 64 restores TVA's established
  production batch size and every epoch still covers the full training fold.
  It would have kept the comparison controlled only if all 20 runs used that
  setting; D022 instead freezes all 20 runs at batch 32.

## D022 — Complete the ED five-fold matrix on the laptop at batch 32

- Status: accepted and frozen on 2026-09-24 after all four fold-0 conditions
  completed
- Trigger: The A6000 workstation became unavailable after its filesystem
  returned a read-only error and remote SSH reset before key exchange. The RTX
  4060 laptop completed every matched WD/WI fold-0 condition at batch 32 in
  approximately 69--71 minutes per run, with a measured peak of 5,211 MiB in
  the timing run. These four valid results already share one protocol.
- Decision: use physical batch size 32 on the RTX 4060 laptop for the complete
  ED five-fold comparison. Retain the completed fold-0 runs and train folds
  1--4 for handwriting-aware and linguistic Bigram under both WD and WI. Keep
  architecture, 300 epochs, 30-epoch warmup, learning rate 0.001, augmentation,
  seed 42, greedy segmentation, and all other non-condition fields unchanged.
- Update budget: the fold-0 WD run has 64 optimizer updates per epoch, or
  19,200 across 300 epochs. Fold sizes may cause the last batch and exact step
  count to differ slightly, but every paired condition within a fold uses the
  same batch size and split.
- Scientific constraint: do not mix any batch-64 run into this five-fold
  aggregate. A future A6000 batch-64 study would be a separate complete matrix
  and would require rerunning fold 0.
- Reproducibility: `run_ed_matrix.py` loads the four frozen production
  templates, overrides only `idx_fold` and `size_batch`, defaults to folds
  1--4 at batch 32, and executes the 16 remaining runs sequentially. It
  validates frozen invariants and all output paths before starting, refuses to
  overwrite non-empty result directories, and provides `--dry-run` preview.
- Failure policy: if a run fails, stop the matrix and preserve its directory
  for diagnosis. Do not delete or silently resume it. Restart only explicitly
  selected untouched conditions after documenting the failure or define a
  reviewed checkpoint-resume procedure.

## D023 — Use ED as the repository-wide dataset identifier

- Status: accepted and implemented on 2026-09-25
- Decision: refer to the supervisor-supplied English Dataset exclusively as
  `ED` in active code, configuration, artifacts, tests, notebooks,
  documentation, environment variables, and result-directory names.
- API migration: use `ed-zip`, `ed_distribution`, `TVA_ED_DATASET_DIR`,
  `EdZipDataset`, the `tva.dataset.ed` loader, `tva.ed_*` tokenizer modules,
  `configs/thesis/ed/`, and `run_ed_matrix.py`.
- Scientific invariants: this is a terminology and organization change only.
  The two 224-class vocabularies, token IDs, 78-character singleton alphabet,
  145 selected bigrams, segmentation policy, dataset archives, fold
  assignments, model settings, and reported recognition results are
  unchanged.
- Artifact identity: terminology metadata changed the canonical JSON bytes.
  The ED handwriting adapter SHA-256 is
  `d79062a252ac1ce05e1a309dab9dd439ee06a06e497024cc9321ecce401e40b7`;
  the IAM frequency-evidence SHA-256 is
  `7808d5d7b58354be982b042c8f60db4d63bf2ee12aa03cbeca6c3fe628cd1ebb`;
  and the ED linguistic adapter SHA-256 is
  `1da363e43fe9d300ece7ad5e3183d84a15fab4bdc6d5b2239b450c222b175abd`.
- Provenance rule: previously generated logs and checkpoint payloads remain
  immutable historical evidence. Their enclosing result directories may use
  the ED naming convention, but their recorded execution metadata must not be
  rewritten retrospectively.

## D024 — Freeze tokenizer construction: TVA's trainers plus allowed pairs

- Status: accepted on 2026-11-01. This is the final construction method.
  Ablations change only the settings listed below, never the construction
  code.
- Decision, in one sentence: every learned tokenizer is built by **TVA's own
  `.train()`, unchanged**; a handwriting-aware tokenizer gets one extra input,
  a **list of letter pairs that are allowed to join** inside a token, and that
  list is the only place connectivity enters.
- How the list is used (`allowed_pairs` in `tva/tokenizers.py`):
  - BPE and Unigram: before TVA's trainer sees the text, each line is cut
    between two letters whose pair is not allowed. Pieces can therefore only
    form across allowed pairs.
  - Bigram: only allowed pairs are counted, so only they can be selected.
  - `allowed_pairs=None` (the default) allows every pair and reproduces the
    original TVA tokenizer exactly.
- Linguistic tokenizer: every pair allowed, TVA's `letters_only=True`. This is
  the matched partner, because handwriting tokenizers never join spaces or
  punctuation. The original TVA bigram (`letters_only=False`) stays as a
  reference result.
- Handwriting tokenizer: the allowed list is the **N most-connected letter
  pairs** in the IAM evidence, ranked by `exact_alignment_connected_rate`
  (connectivity method `dominant-core-v3`), among pairs with at least 20
  aligned observations; ties are broken alphabetically.
  - BPE and Unigram: N = 145 by default, independent of the vocabulary size.
    These are exactly the pairs written connected at least half of the time.
  - Bigram: N equals the number of bigram pieces, so a size-k handwriting
    bigram is the k most-connected pairs, and its linguistic partner is TVA's
    k most frequent pairs. At k = 145 this is the earlier handwriting bigram.
- Shared by both sides: the 5,694 IAM training lines, deduplicated and
  filtered by TVA's own code; 224 classes by default; TVA's encoders (greedy
  bigram, Hugging Face BPE and Unigram).
- Settings that ablations may change (no code changes needed):
  - the allowed list: N, the minimum observation count, the rate column (for
    example `box_intersection_v1_connected_rate`), the evidence file, or a
    control list (random, frequency-drawn, least connected, noisy);
  - the vocabulary size. Limits at the defaults: the handwriting bigram can
    have at most 389 pieces (468 classes), because only 389 letter pairs have
    20 observations; BPE reaches 500 classes; Unigram on cut text stops at
    about 470 classes (asked for 500, got 478), so record the size reached;
  - the bigram segmentation policy (D025);
  - compression-matched controls: TVA's trainer at a smaller vocabulary size.
- Reason: the TVA baseline stays the published method, untouched, and the
  handwriting tokenizer is "TVA plus connectivity" with nothing else changed.
  Ranking pairs by how often they are written connected was rejected: it is
  mostly text frequency and keeps 136 of TVA's 145 bigrams. Restricting which
  pairs may join keeps connectivity as the deciding factor (the 145 allowed
  pairs share only 72 with TVA's list).
- Fairness: matched construction is necessary but not sufficient. A
  handwriting gain is attributed to connectivity only after the
  compression-matched control, a control list of the same size chosen without
  connectivity, and five folds.
- Reproducibility: `tokenizers==0.23.1` (pinned in `requirements.txt` and
  `environment.yml`) and `PYTHONHASHSEED=42` (TVA breaks ties in `set()`
  order). Hugging Face's Unigram trainer is not bit-reproducible, even run to
  run; Unigram files are checked by their piece set, and the frozen file is
  authoritative.
- Supersedes: the custom handwriting BPE/Unigram trainers
  (`tva/handwriting/training.py`), `UnigramEMTokenizer` and the
  `unigram_trainer` experiment, and the handwriting bigram's alphabetical ID
  order (IDs now follow TVA's frequency order). Earlier files stay at their
  paths so that completed runs still decode correctly; the new handwriting and
  compression-matched files are in `*_v3` directories, and their runs write to
  `*_v3_{wd,wi}`.

## D025 — Bigram segmentation is an encoding setting for both sides

- Status: accepted on 2026-11-01.
- Decision: every bigram tokenizer, linguistic and handwriting, can encode
  with either policy:
  - `greedy` (default, config key `bigram`): TVA's left-to-right splitting.
  - `dp` (config key `bigram_dp`): the split with the highest total score;
    ties prefer more bigrams. The score is each tokenizer's own ranking
    score: IAM text count for linguistic bigrams, connected rate for
    handwriting bigrams. It is saved next to the vocabulary as
    `<fold>.scores.json`, so TVA's vocabulary files stay unchanged.
- Fairness rule: a policy is always applied to both sides of a comparison
  (as already required by D016 and D017). Greedy is the primary result; DP is
  a named ablation.
- Scope: BPE (learned merge order) and Unigram (most likely split) have their
  own fixed splitting and no policy choice.
- Expectation: on ED, the September 2026 segmentation audit found the same
  total token count for both policies but different tokens for 70% of labels,
  so this ablation tests which bigrams are used, not compression.

## D026 — Language-specific construction and evaluation

- Status: confirmed by the user on 2026-11-14.
- English: construct linguistic and handwriting-aware English tokenizers using
  IAM training data; evaluate their recognition performance on ED.
- German: construct German tokenizers using a suitable German handwriting
  dataset, still to be selected and approved; evaluate on right-handed
  OnHW-Words500 in the existing WD/WI settings.
- READ: retain its experiments, evidence, and findings as historical records,
  but do not use it for future construction or evaluation. Its historical
  German handwriting style does not suit the intended German study.
- Pending: DHSD and FH-SWF German handwriting have been considered, but their
  comparison is deferred to another session. Neither dataset is adopted, and
  no new German experiment is authorized by this decision alone.
- Reason: use language-matched external handwriting evidence for construction
  and a separate recognition dataset for evaluation. This separates the English
  and planned German tracks without treating them as a direct cross-language
  performance comparison.
- Supersedes: the OnHW-only recognition scope and active IAM/READ construction
  plan in earlier scope, method, and handoff notes, including D002's exclusive
  recognition-dataset scope. The right-handed restriction for OnHW remains.
  D024/D025 continue to govern current frozen English construction and
  segmentation; German-specific construction settings require approval.
- Historical integrity: this clarification does not alter previous datasets,
  vocabularies, runs, scores, or their original experimental meaning. Do not
  rewrite earlier records or apply ED's seven-channel settings to OnHW's
  thirteen-channel recordings.


## D027 — Sensor forced alignment as a secondary OnHW experiment

- Status: approved by the user on 2026-11-15 as the secondary experiment named
  in the proposal. The primary method does not change (D012, D024–D026).
- Decision: continue the August forced-alignment pipeline (D004–D006,
  D009–D011) on right-handed OnHW-Words500. The evidence is a per-pair
  continuity score from the sensor recordings of the fold's training split.
- Construction: the pair ranking becomes an `allowed_pairs` list for TVA's
  unchanged trainers, as D024 allows for a new evidence file. No construction
  code changes.
- Leakage rule: evidence comes only from the fold's training split. The scorer
  already rejects other splits. Validation labels, frequencies, and scores must
  not select pairs, thresholds, or weights.
- Difference from D026: D026 builds German tokenizers from a separate German
  handwriting dataset. This experiment takes evidence from OnHW's own training
  recordings. It is a separate, named condition, not the German main track.
- Pending (approve before runs): construction corpus, vocabulary sizes, number
  of allowed pairs, support gate, force-only or force plus motion, and the
  comparison set.
- Review: the 2026-11-15 method review proposes a distinct-word gate, event-based
  lift detection, a delayed-stroke correction, force as the primary evidence,
  and three validation checks. See
  [FORCED_ALIGNMENT.md](FORCED_ALIGNMENT.md#method-review--2026-11-15).
  These are proposals, not approved changes.
