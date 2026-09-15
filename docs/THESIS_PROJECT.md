# Handwriting-Aware Tokenization for IMU-Based Online Handwriting Recognition

## Purpose and authority

This is the repository's canonical thesis-scope document. It follows the
approved Pattern Recognition Lab proposal dated September 2026. Historical
forced-alignment work remains documented, but it is not the primary proposed
method.

## Thesis in plain language

The model receives the same raw 13-channel IMU time series in every recognition
experiment. Tokenization changes only the output text labels learned through
CTC. For example, `the` may be encoded as `[t] [h] [e]` or `[th] [e]`; `[th]`
is a text token, not an IMU segment.

Conventional Bigram, BPE, and Unigram tokenizers choose character groups mainly
from linguistic frequency. This thesis asks whether visible handwriting
structure can produce more suitable token vocabularies.

The fixed recognition path is:

```text
raw 13-channel IMU
  -> BLConv-B encoder
  -> BiLSTM-B decoder
  -> CTC probabilities over text-label tokens
  -> decoded ordinary text
```

## Research question and hypotheses

The main question is whether handwriting-aware label tokenizers, constructed
from visible character connectivity in handwriting images, improve IMU-based
online handwriting recognition over corresponding frequency-based tokenizers.

The proposal tests two hypotheses:

1. Tokenizers trained with handwriting patterns achieve better recognition
   performance than corresponding tokenizers trained only with linguistic
   frequency information.
2. Their benefit is more pronounced in writer-independent recognition than in
   writer-dependent recognition.

These are hypotheses, not established claims.

## Primary method: image-derived connectivity

### Evidence sources

Use the IAM English and READ German offline handwriting datasets. Analyse them
separately; do not silently pool their evidence. An explicit combined-corpus
condition may be added later only with a documented weighting rule.

### Character localization

Use a pretrained DTLR model to obtain character identities and bounding boxes.
Match the ordered detections to the known transcription and exclude uncertain
localizations. Record confidence, matching outcome, dataset, sample, writer
when available, and rejection reasons.

DTLR/transcript agreement is a critical validity check. The method must specify
how it treats recognition errors, repeated characters, punctuation, spaces,
case, and symbols not represented in OnHW.

### Connected-component evidence

Binarize each accepted image and apply connected-component labelling (CCL).
For neighbouring characters in reading order, relate their DTLR boxes to ink
components and record whether their visible ink is connected. A component-gap
or distance measurement can provide secondary evidence.

CCL components are not characters: multiple cursive characters can share one
component, and one character can contain detached components. Character boxes
and identities are therefore required to interpret connectivity.

### Pair aggregation

Aggregate the accepted observations into a connectivity score for each
case-sensitive adjacent-character pair. Retain occurrence count, connected
proportion, writer coverage, writer-level variability, uncertainty rate, and
corpus provenance. Where writer identities exist, balance writers so prolific
writers do not dominate.

Frequency and writer coverage are reliability gates. They must not become the
primary ranking objective.

Static image connectivity is used as a cross-modal handwriting prior for an
IMU recognizer. It is not evidence that a particular IMU recording contains a
continuous pen trajectory at that boundary.

## Tokenizer plan

### Bigram first

Construct and analyse a handwriting-aware Bigram vocabulary from image-derived
pair scores. Keep every individual character as a fallback token and use a
documented deterministic encoding rule. Compare against both the character
baseline and a frequency-based Bigram vocabulary of matched size.

### Conditional BPE and Unigram extensions

Extend the approach to BPE and Unigram only if the Bigram results are
promising, as specified in the proposal.

- Handwriting-aware BPE starts from characters and guides successive merges
  with connectivity rather than primarily with frequency.
- Handwriting-aware Unigram generates and prunes bounded-length candidates
  using their internal connectivity and reliability.

For longer tokens, the rule that combines internal pair scores must be defined
and frozen before final experiments. All characters remain fallback tokens.
Each proposed tokenizer is compared with its frequency-based counterpart at
the same or nearly the same vocabulary size.

## Recognition data and evaluation

- Recognition dataset: right-handed OnHW-Words500.
- Settings: writer-dependent (WD/RH) and writer-independent (WI/RH).
- Final evaluation: five folds.
- Metrics: character error rate (CER) and word error rate (WER) after decoded
  tokens are concatenated into ordinary text.
- Recognition model: BLConv-B + BiLSTM-B + CTC.

Left-handed data is excluded because it is too small for the planned
comparison.

Train a new recognizer from scratch for every tokenizer. Do not initialize a
main comparison from a character-model checkpoint. Keep raw input,
architecture, preprocessing, augmentation, epochs, optimizer, learning-rate
schedule, batch size, seed, split, and evaluation fixed unless the changed
item is explicitly being studied.

Development may use fold 0 and a small predeclared vocabulary-size screen.
Freeze the method and a limited final size set before five-fold evaluation.
Fold-0 results alone are development findings, not final thesis results.

## Leakage and reproducibility rules

Image connectivity is external evidence, but its construction still needs
fixed corpus splits and provenance. Never use OnHW validation recordings or
recognition scores to construct a tokenizer, choose connectivity thresholds,
or tune evidence weights for that same evaluation.

Every completed experiment should preserve:

- code commit and resolved configuration;
- OnHW split and data version;
- IAM/READ split and sample provenance;
- DTLR model/checkpoint and matching rules;
- binarization, CCL, uncertainty, and support settings;
- frozen tokenizer vocabulary and encoding rule;
- seed, checkpoint, predictions, CER/WER, result directory, and backup
  checksum.

## Completed development work

### Data and training infrastructure

The processed right-handed WD and WI datasets have been validated: 13 numeric
channels, 100 Hz target rate, five folds, 25,199 samples per setting, 501 word
labels, expected WD writer overlap, and zero WI train/validation writer overlap.

Training supports device-aware mixed precision, atomic resumable `latest.pth`
checkpoints, validation-selected `best_cer.pth` and `best_wer.pth`, random-state
restoration, and deterministic DataLoader continuation.

### Fold-0 character baselines

These are development results, not final five-fold results:

| Run | Architecture | Best CER | Best WER |
| --- | --- | ---: | ---: |
| B0 WD/RH | BLConv-B + BiLSTM-B + CTC | 12.76% | 35.98% |
| B0 WI/RH | BLConv-B + BiLSTM-B + CTC | 15.60% | 27.95% |
| A0 WI alignment model | BLConv-B + UniLSTM-B + CTC | 17.33% | 32.77% |

A0 is an alignment-development model, not a recognition baseline replacement.

## Secondary/fallback IMU forced-alignment study

An earlier thesis direction has already produced a functional fold-0 WI
pipeline for target-constrained CTC alignment and sensor-boundary analysis. It
includes alignment diagnostics, approximate frame-to-input mapping, local and
whole-region force/motion features, resumable training-only JSONL exports,
pair/position analysis, and a writer-balanced corrected force score.

This work is retained as an additional comparison if the image-based method is
unreliable and time remains. It may also help compare image-derived and
IMU-derived rankings. It does not define the primary tokenizer in the approved
proposal.

Important limitations are:

- CTC emissions are model alignments, not physical character-boundary truth.
- The B0 BiLSTM placed many word-initial emissions implausibly early.
- A0 reduced that bias but is not fully causal because BLConv has centred
  convolutions and sequence-wide instance normalization.
- The current 75% local/25% whole-region force score is a provisional
  development baseline, not proof of motion continuity.

Technical details and commands remain in `docs/FORCED_ALIGNMENT.md`.

## Optional image-domain recognition

If time and resources permit, evaluate the same frozen tokenizers with a fixed
offline handwriting-recognition model. This asks whether the labels also help
directly in the image domain. It is optional and must not displace or
retroactively tune the primary image-to-IMU transfer study.

## Current implementation order

1. Prepare IAM and READ separately.
2. Obtain and validate DTLR character identities and boxes against
   transcriptions; exclude uncertain localizations.
3. Implement binarization, CCL, box-to-component association, and auditable
   pair-connectivity aggregation.
4. Build and analyse handwriting-aware and matched frequency-based Bigram
   vocabularies.
5. Train fold-0 Bigram comparisons on WD/RH and WI/RH.
6. If promising, define handwriting-aware BPE and Unigram and their matched
   baselines.
7. Freeze the method and run the selected five-fold comparison.
8. Attempt optional image-domain recognition or the secondary IMU-alignment
   comparison only if time and evidence justify it.

## Local project paths

```text
data/raw/Words500_dep_R/
data/raw/Words500_indep_R/
data/tva/onhw_words500_wd_word_rh/
data/tva/onhw_words500_wi_word_rh/
results/thesis/
```

IAM, READ, DTLR artifacts, and image-connectivity outputs need dedicated,
portable paths before implementation. Large data and results remain
Git-ignored and should be trained/read from local storage, with completed runs
backed up separately.

## Documentation map

- `docs/PROPOSAL_PLAN.md`: concise proposal-aligned scope.
- `docs/METHOD.md`: technical image-connectivity method.
- `docs/PROGRESS.md`: chronological record, including superseded work.
- `docs/DECISIONS.md`: methodological decisions and supersessions.
- `docs/EXPERIMENTS.md`: configurations, results, and caveats.
- `docs/FORCED_ALIGNMENT.md`: secondary IMU pipeline documentation.
- `docs/SETUP.md`: reproducible machine setup.
