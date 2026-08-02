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
tokens mainly from text occurrence statistics. This thesis will develop
**handwriting-aware counterparts for these tokenizer families**. Their token
inventories or merge decisions will be guided primarily by handwriting-motion
continuity rather than linguistic frequency. A pair such as `th` should become
one token when the transition from `t` to `h` is repeatedly written as a
smooth, continuous motion—not simply because `th` is frequent in the text.

The planned proposed variants are:

- a handwriting-aware Bigram tokenizer that selects character pairs using
  reliable boundary-continuity scores;
- a handwriting-aware BPE tokenizer that performs iterative motion-guided
  merges; and
- a handwriting-aware Unigram tokenizer that generates, scores, and prunes
  substring candidates using motion cohesion.

Frequency still has a secondary role as a reliability requirement: a candidate
must occur often enough, and across enough writers, for its motion score to be
credible. Frequency is not the primary ranking or merge objective.

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

> Can handwriting-aware Bigram, BPE, and Unigram text-label tokenizers learned
> from motion continuity improve raw-IMU handwriting recognition compared with
> character-level labels and their linguistically trained counterparts?

Supporting questions are:

1. Which aligned sensor features provide reliable evidence of continuity at
   adjacent-character boundaries?
2. Does handwriting-aware tokenization improve CER or WER when vocabulary size,
   architecture, data split, and training procedure are controlled?
3. Do the Bigram, BPE, and Unigram families benefit equally from handwriting
   information, or is one family better suited to motion-guided token learning?

## Core hypotheses

1. Some character transitions have repeatable and continuous handwriting
   patterns that make them useful as multi-character output tokens.
2. Bigram, BPE, and Unigram tokenizers adapted to this motion evidence may
   produce labels that are easier for the IMU recognizer to learn than tokens
   selected only by text statistics.
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

### 4. Build shared motion-continuity evidence

Aggregate the boundary measurements for each adjacent character pair across the
training data. A candidate should influence token construction only when it has:

- enough training examples;
- strong average motion continuity;
- reasonably consistent continuity across examples; and
- ideally, consistency across different writers.

The output of this stage is a fold-specific table containing pair counts,
writer coverage, median continuity, variability, and alignment confidence.
This shared table supplies handwriting evidence to all proposed tokenizer
families.

### 5. Train handwriting-aware tokenizer families

Construct three proposed tokenizer variants:

1. **Handwriting-aware Bigram:** keep characters and add reliable pairs ranked
   by motion continuity rather than occurrence count.
2. **Handwriting-aware BPE:** begin with characters and iteratively merge the
   adjacent token pair with the strongest reliable motion-cohesion score rather
   than the highest linguistic frequency.
3. **Handwriting-aware Unigram:** generate bounded-length substring candidates,
   assign utilities from internal continuity, consistency, support, and writer
   coverage, then prune candidates and tokenize with dynamic programming.

For a longer candidate, its cohesion can be derived from its internal character
boundaries. The initial interpretable choice is the minimum internal continuity
score, so one clearly discontinuous boundary cannot be hidden by other smooth
boundaries. Mean or geometric-mean aggregation can be evaluated as an ablation.

All individual characters must remain in the vocabulary as fallback tokens, so
every word can always be represented. Once the vocabulary is selected, labels
can be tokenized deterministically using a documented rule such as
left-to-right longest matching or dynamic programming.

### 6. Train each recognizer from scratch

Initialize a new BLConv + BiLSTM + CTC model and train it on the same raw IMU
recordings for every linguistic and handwriting-aware tokenizer, changing only
the text-label representation. Do not reuse the trained character-model weights
for the main fair comparisons.

### 7. Decode and evaluate as ordinary text

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
| B1-Bigram | Raw IMU | Linguistic Bigram tokens | Matched linguistic baseline |
| P1-Bigram | Raw IMU | Handwriting-aware Bigram tokens | Motion-guided pair method |
| B1-BPE | Raw IMU | Linguistic BPE tokens | Matched linguistic baseline |
| P1-BPE | Raw IMU | Handwriting-aware BPE tokens | Motion-guided merge method |
| B1-Unigram | Raw IMU | Linguistic Unigram tokens | Matched linguistic baseline |
| P1-Unigram | Raw IMU | Handwriting-aware Unigram tokens | Motion-guided candidate method |

Each handwriting-aware method must be compared with its linguistic counterpart
at the same or closely matched vocabulary size. Development is staged—Bigram
first, followed by BPE and Unigram—so that alignment and continuity errors can
be corrected before they affect more complex tokenizers. The proposal includes
all three families; staging describes implementation order, not a different
research objective. Exact final vocabulary sizes remain to be agreed. Candidate
sizes may be screened on fold 0, but the final cross-validation matrix should
use a small, predeclared set to keep the comparison feasible.

## Dataset and local paths

- Raw WD data: `data/raw/Words500_dep_R`
- Raw WI data: `data/raw/Words500_indep_R`
- Processed WD data: `data/tva/onhw_words500_wd_word_rh`
- Processed WI data: `data/tva/onhw_words500_wi_word_rh`
- WD tokenizer directory: `data/tva/onhw_words500_wd_word_rh/tokenizers/`
- Character-baseline configurations: `configs/thesis/b0_char_wd_rh.yaml` and
  `configs/thesis/b0_char_wi_rh.yaml`
- Thesis results: `results/thesis/`

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
- Run the final character, linguistic, and handwriting-aware tokenizer-family
  comparisons on the same hardware and software environment where practical.
- Never allow two machines to write to the same result directory.

A reproducible experiment record should include the experiment ID, date, Git
commit, resolved configuration, dataset split or manifest hash, tokenizer
files, random seed, machine and software environment, CER/WER, predictions,
checkpoints, and short observations.

Training checkpoints now preserve the model, optimizer, scheduler, gradient
scaler, epoch, metrics, and random states. `latest.pth` supports continuation,
while `best_cer.pth` and `best_wer.pth` preserve validation-selected models.

## Initial implementation order

1. Create and verify a dedicated character-baseline configuration.
2. Run a very short fold-0 smoke test.
3. Train and evaluate the full fold-0 character baseline.
4. Implement and test target-constrained CTC forced alignment.
5. Extract adjacent-character boundary measurements from training data.
6. Implement handwriting-aware Bigram and compare it with linguistic Bigram.
7. Adapt the shared motion evidence to handwriting-aware BPE and Unigram.
8. Train all matched linguistic and handwriting-aware variants from scratch on
   fold 0 and refine the method after inspecting the results.
9. Freeze the methodology and run the agreed final cross-validation matrix.
10. Attempt optional image extensions only after the IMU comparisons are
    complete.

## Current training observations

- The observed WD/RH fold-0 run on the RTX 4060 Laptop GPU took roughly 20
  seconds per train-and-validation epoch at batch size 64.
- At that observed rate, 300 epochs take roughly 100 minutes per fold. Runtime
  may differ by tokenizer, fold, caching state, WSL resources, and machine.
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
- Essential comparison: character baseline plus matched linguistic and
  handwriting-aware Bigram, BPE, and Unigram tokenizers.
- Vocabulary sizes must be matched within each linguistic/handwriting-aware
  tokenizer-family comparison.
- Exact final vocabulary sizes are unresolved and must be frozen before the
  final cross-validation runs.
- Left-handed IMU data: excluded from the thesis because the available dataset
  is too small; this scope decision was confirmed by the supervisor.
- Image/DTLR/CCL evaluation: optional and only after the IMU method works.
- BPE and Unigram provide the planned longer-than-pair variants. Unbounded or
  end-to-end neural tokenizer learning remains outside the core scope.
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
- `docs/PROPOSAL_PLAN.md`: proposal-ready summary of the agreed plan.

Record failed experiments as well as successful ones. A failed attempt can
still provide useful evidence and will make the final thesis easier to write.
