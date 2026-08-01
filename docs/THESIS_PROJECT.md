# Handwriting-Informed Label Tokenization for IMU Handwriting Recognition

## Purpose of this document

Read this file before proposing changes, experiments, or code for this thesis.
It records the agreed research direction, the baseline project, the planned
method, and the rules for a fair comparison. Update it whenever the thesis
scope or methodology changes.

## Baseline repository and paper

- Repository: `muhammadalised/TVA` (the user's fork of the TVA project).
- Paper: *Tokenization vs. Augmentation: A Systematic Study of Writer Variance
  in IMU-Based Online Handwriting Recognition*.
- Baseline architecture: BLConv encoder + BiLSTM + CTC.
- Model input: raw 13-channel IMU time-series signals.
- Paper data: right-handed OnHW-Words500, evaluated in writer-dependent (WD/RH)
  and writer-independent (WI/RH) settings using 5-fold cross-validation.
- Evaluation metrics: character error rate (CER) and word error rate (WER).

### Important correction about tokenization

Both the TVA paper and this thesis tokenize **text labels**, not the raw IMU
signals. The model must continue to receive the raw IMU time series as input.
Tokenization changes the target sequence that the CTC model learns to predict.

For example, the word `the` may be represented as:

```text
Character labels:  [t] [h] [e]
Multi-character labels: [th] [e]
```

The token `[th]` is still a text-label token. It is not a clustered IMU segment
or an extra input to the recognizer.

## Corrected thesis idea

The thesis will investigate **handwriting-informed text-label tokenization for
raw-IMU handwriting recognition**.

Existing Bigram, BPE, and Unigram tokenizers normally select multi-character
tokens mainly from text occurrence statistics. The proposed tokenizer will
instead use evidence from handwriting motion to decide which adjacent
characters are suitable to combine. A pair such as `th` should become one token
when the transition from `t` to `h` is repeatedly written as a smooth,
continuous motion—not simply because `th` is frequent in written language.

The complete recognition pipeline remains:

```text
raw 13-channel IMU signal
    -> BLConv encoder
    -> BiLSTM
    -> CTC output probabilities over text-label tokens
    -> CTC decoding
    -> predicted text
```

Only the output vocabulary and the tokenized ground-truth labels change between
the baseline and proposed approaches.

## Main research question

> Can text-label tokens selected using handwriting-motion continuity improve
> raw-IMU handwriting recognition compared with character-level labels and
> linguistically derived tokenization?

## Core hypotheses

1. Some character transitions have repeatable and continuous handwriting
   patterns that make them useful as multi-character output tokens.
2. A tokenizer built from this motion evidence may produce labels that are
   easier for the IMU recognizer to learn than tokens selected only by text
   frequency.
3. The model must be trained from scratch for every tokenizer because changing
   its target labels can also change what useful features its encoder learns.

These are hypotheses to test, not claims that are already proven.

## Proposed method

### 1. Reproduce the character baseline

Train the existing BLConv + BiLSTM + CTC model using raw IMU input and
character-level text labels. This verifies the dataset, training code, and
evaluation procedure before introducing a new method.

### 2. Estimate where characters occur in time

The recordings contain word transcriptions but do not provide exact start and
end times for every character. Train a character-level CTC model and use
target-constrained CTC forced alignment to estimate a time interval for each
known character in each training word.

A simple greedy CTC path can be useful for an early prototype, but the intended
method is CTC Viterbi forced alignment. It constrains the alignment to the known
ground-truth word and correctly handles CTC blanks and repeated characters.
Low-confidence alignments should be excluded from tokenizer construction.

### 3. Measure continuity at adjacent-character boundaries

For every aligned pair of adjacent characters, inspect a small time window
around their estimated boundary. Possible continuity evidence includes:

- whether force or pressure remains active or drops;
- the presence and duration of a pause or pen lift;
- changes in accelerometer and gyroscope signals;
- changes in direction, speed, or signal energy; and
- changes in the encoder's learned frame-level features.

These measurements do not independently reveal character boundaries. They are
used only after forced alignment has estimated a boundary from the recording
and its known text label.

### 4. Build a handwriting-informed text vocabulary

Aggregate the boundary measurements for each candidate character pair across
the training data. A pair should be considered as a multi-character token only
when it has:

- enough training examples;
- strong average motion continuity;
- reasonably consistent continuity across examples; and
- ideally, consistency across different writers.

The first proposed tokenizer should use character pairs only. This keeps the
method understandable and limits the number of decisions. Longer tokens can be
tested later if time permits.

All individual characters must remain in the vocabulary as fallback tokens, so
every word can always be represented. Once the vocabulary is selected, labels
can be tokenized deterministically using a documented rule such as
left-to-right longest matching or dynamic programming.

### 5. Train the proposed recognizer from scratch

Initialize a new BLConv + BiLSTM + CTC model and train it on the same raw IMU
recordings, but use the handwriting-informed text tokens as its output labels.
Do not reuse the trained weights from the character model for the main fair
comparison.

### 6. Decode and evaluate as ordinary text

Convert predicted token sequences back to text by concatenating their token
strings. For example, `[th] [e]` becomes `the`. Calculate CER and WER on this
reconstructed text so that every tokenizer is evaluated in the same way.

## Fair experiment design

Use the same data splits, preprocessing, augmentations, training duration,
optimizer, learning-rate schedule, batch size, seed, and CER/WER evaluation for
each comparison unless the changed item is the explicit research variable.

For every cross-validation fold:

- train the character alignment model using that fold's training recordings;
- estimate alignments for training recordings only;
- calculate motion-continuity statistics from training recordings only;
- construct and freeze the tokenizer using the training partition only; and
- apply the frozen tokenizer and model to the held-out validation partition.

The validation labels may be encoded using the frozen tokenizer, but validation
recordings must never influence token selection, thresholds, or continuity
statistics. This prevents data leakage, especially in writer-independent
experiments.

### Core comparison matrix

| ID | Model input | CTC output labels | Purpose |
| --- | --- | --- | --- |
| B0 | Raw IMU | Individual characters | Essential reference baseline |
| B1 | Raw IMU | Linguistic Bigram/BPE/Unigram tokens | Existing TVA comparison |
| P1 | Raw IMU | Handwriting-continuity-informed text tokens | Primary thesis method |

Run B0 first. Then implement and evaluate P1 against B0. Add the most relevant
B1 comparison to determine whether motion information adds value beyond
ordinary text statistics. Avoid expanding to many tokenizer variants before
the core pipeline works reliably.

## Dataset and local paths

- Raw WD data: `data/raw/Words500_dep_R`
- Raw WI data: `data/raw/Words500_indep_R`
- Processed WD data: `data/tva/onhw_words500_wd_word_rh`
- Processed WI data: `data/tva/onhw_words500_wi_word_rh`
- WD tokenizer directory: `data/tva/onhw_words500_wd_word_rh/tokenizers/`
- Current training configuration: `configs/train.yaml`
- Results: `results/tva/`

The thesis will use the right-handed writer-dependent and writer-independent
OnHW-Words500 datasets. The supervisor confirmed that left-handed data will not
be included because the available left-handed dataset is too small for the
planned comparison.

The user will manage the dataset and result directories. Code must not assume
that large datasets, checkpoints, or result folders are committed to Git.

## Development and experiment workflow

- Use this repository as the single source of truth for thesis code.
- Develop primarily on the Mac, use short smoke tests on available machines,
  and use the office workstation for full or final runs when possible.
- Commit and push code before moving an experiment to another machine. Pull the
  exact commit on that machine instead of copying modified files manually.
- Keep scientific settings in committed configuration files. Keep local data
  paths and machine-specific settings separate.
- Run final B0/B1/P1 comparisons on the same hardware and software environment
  where practical.
- Never allow two machines to write to the same result directory.

A reproducible experiment record should include the experiment ID, date, Git
commit, resolved configuration, dataset split or manifest hash, tokenizer
files, random seed, machine and software environment, CER/WER, predictions,
checkpoints, and short observations.

The existing checkpoint code restores model weights only. It is not yet safe
for exact training continuation on another machine because it does not restore
the optimizer, scheduler, gradient scaler, epoch, and random-number states.
Proper resumable checkpointing should be implemented before moving an
unfinished training run between machines.

## Initial implementation order

1. Create and verify a dedicated character-baseline configuration.
2. Run a very short fold-0 smoke test.
3. Train and evaluate the full fold-0 character baseline.
4. Implement and test target-constrained CTC forced alignment.
5. Extract adjacent-character boundary measurements from training data.
6. Implement the first pair-based handwriting-informed tokenizer.
7. Train P1 from scratch on fold 0 and compare it with B0.
8. Refine the method only after inspecting alignment quality and initial
   results.
9. Run the agreed final cross-validation experiments.
10. Attempt optional extensions only if the core experiments are complete.

## Current training observations

- A 4060 Laptop GPU takes roughly 2 minutes per train-and-validation epoch for
  one WD fold at batch size 64.
- At 300 epochs, this is roughly 10 hours per fold or 50 hours for five folds.
- During development, use fold 0 and very few epochs. Use all five folds only
  for established or final experiments.
- The paper's reported result is an aggregate over five folds, so a single fold
  is not expected to reproduce that number exactly.

## Optional image-based extension

If the core IMU study is completed early, the general label-tokenization idea
may also be evaluated with offline handwriting images. One possible approach is
to combine character localization from a detector such as DTLR with connected
component labeling (CCL).

CCL finds connected regions of ink, not characters. Several cursive characters
may form one component, while a character such as `i` can contain multiple
components. Character boxes or locations are therefore needed to relate ink
components to adjacent text characters.

This extension could select text tokens when adjacent localized characters
consistently share connected ink. It should be described as
**stroke- or ink-connectivity-informed tokenization**, not true motion
continuity, because a static image does not contain the temporal pen trajectory.

The image experiment is optional cross-modal validation, not part of the minimum
viable thesis.

## Scope boundaries and unresolved decisions

- Core thesis: raw-IMU handwriting recognition with handwriting-informed
  text-label tokenization.
- Essential comparison: character baseline versus the proposed tokenizer.
- Important comparison: the proposed tokenizer versus a linguistically derived
  tokenizer of comparable vocabulary size.
- Left-handed IMU data: excluded from the thesis because the available dataset
  is too small; this scope decision was confirmed by the supervisor.
- Image/DTLR/CCL evaluation: optional and only after the IMU method works.
- Longer-than-pair tokens and complex end-to-end token learning: optional.
- Do not cluster or tokenize the IMU input as the primary thesis method.
- Do not claim that pen lifts, force changes, or CCL components directly reveal
  character boundaries.

## Documentation files to maintain

As implementation begins, maintain the following supporting records:

- `docs/PROGRESS.md`: chronological research diary and next steps;
- `docs/EXPERIMENTS.md`: experiment configurations, results, and conclusions;
- `docs/DECISIONS.md`: important methodological decisions and their reasons;
- `docs/SETUP.md`: reproducible setup for each training machine; and
- `docs/METHOD.md`: detailed explanation of alignment and tokenizer design.

Record failed experiments as well as successful ones. A failed attempt can
still provide useful evidence and will make the final thesis easier to write.
