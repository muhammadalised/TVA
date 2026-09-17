# Thesis Project Handoff — RTX Machine

**Last updated:** 2026-09-15

**Repository:** [muhammadalised/TVA](https://github.com/muhammadalised/TVA)

**Current experiment branch:** `handwriting-bigram-onhw-v1`
**Documentation baseline commit:** `d6f99c6` — *Align thesis documentation with proposal*

This is the starting document for continuing the thesis on another machine. Read this file first, then read `docs/THESIS_PROJECT.md`, `docs/PROGRESS.md`, and `docs/DECISIONS.md` for the full technical record.

## 1. The thesis in plain language

The recognition model continues to receive a **raw 13-channel IMU time series** from the pen. We do **not** tokenize, segment, or replace the IMU input.

What changes is the **text label at the model output**. A normal character model represents `the` as `[t] [h] [e]`. A Bigram tokenizer might represent it as `[th] [e]`. The thesis asks whether pairs/subwords should be chosen because they have a meaningful handwriting relationship, rather than simply because they occur frequently in language.

The recognition architecture is kept fixed for fair comparison:

```text
raw 13-channel IMU
  -> BLConv encoder
  -> BiLSTM decoder
  -> CTC probabilities over output label tokens
  -> decoded text
```

For every tokenizer experiment, train a **new model from scratch**. Only the output vocabulary and ground-truth label encoding may change.

### The three planned handwriting-aware tokenizers

1. **Bigram (first implementation):** retain all characters and add selected two-character tokens, for example `[th]`, based on handwriting evidence.
2. **BPE (conditional extension):** if Bigram is promising, start with
   characters and make successive handwriting-guided merges rather than
   frequency-guided merges.
3. **Unigram (conditional extension):** if Bigram is promising, generate and
   score longer candidate strings from their internal handwriting cohesion,
   then prune/select a vocabulary.

Frequency is still useful as a **reliability gate**: a pair needs enough examples and writer coverage for its handwriting score to be trustworthy. It must not become the main reason a token is selected.

## 2. Approved scope: primary and fallback evidence sources

There are two related approaches discussed with the supervisor. They must not be confused.

### A. Primary proposal method: offline handwriting-image connectivity

The approved proposal uses offline handwriting images as a source of a
*handwriting prior*:

1. Analyse **IAM** English and **READ** German handwriting separately.
2. Obtain character locations/identities with a pretrained **DTLR** model.
3. Binarize the writing and use connected-component labelling (CCL).
4. For adjacent characters in reading order, measure whether their visible ink belongs to the same connected component.
5. Aggregate this across image samples to score character-pair connectivity.
6. Use the resulting scores to make handwriting-aware Bigram choices first, followed by BPE and Unigram if the Bigram result is promising.

This is a **cross-modal transfer study**. Connected ink in a static image is used as a handwriting-informed prior for the IMU recognizer; it is *not* proof that a particular IMU recording has continuous pen motion at that point.

The proposed evaluation is on right-handed OnHW-Words500 IMU data in both writer-dependent (WD) and writer-independent (WI) settings. An optional, lower-priority image-domain evaluation can check whether the same tokenizers also help an offline image recognizer.

### B. Implemented fallback / complementary direction: IMU CTC alignment

The existing `forced-alignment` branch implements the alternative suggested earlier: learn character timing from the IMU data itself using CTC forced alignment, then measure force and motion around estimated character boundaries. This is fully functional for **fold-0 WI development** and can become a fallback if the image-based method is not feasible or does not give useful token rankings.

It is not wasted work. It provides:

- a complete character-level baseline;
- an auditable way to inspect IMU character boundaries;
- a force-based candidate pair ranking for a first IMU-informed Bigram; and
- a future comparison source against image-derived connectivity.

### Scope documentation status

The approved September 2026 proposal makes **image-derived connectivity
primary** and forced alignment an additional fallback/comparison only if the
image method is unreliable and time remains. The repository planning,
methodology, decision, progress, and forced-alignment documents have been
updated to reflect that scope. Do not claim that either static image
connectivity or CTC alignment gives ground-truth physical pen-stroke
boundaries.

## 3. Data and evaluation scope

- **Recognition dataset:** OnHW-Words500.
- **Hand:** right-handed only. Left-handed data is excluded because it is too small; this was confirmed by the supervisor.
- **Settings:** writer-dependent (WD/RH) and writer-independent (WI/RH).
- **Cross-validation:** five folds for final reported results.
- **Metrics:** character error rate (CER) and word error rate (WER), measured after predicted tokens have been concatenated back into ordinary text.
- **Model:** BLConv-B + BiLSTM-B + CTC for the recognition comparisons.

Never let OnHW validation recordings or recognition scores influence tokenizer
construction. Preserve explicit IAM/READ evidence splits and provenance. A
frequency-based comparator learned from OnHW text must use only that fold's
training labels; then freeze each tokenizer before training and evaluation.

## 4. Completed, verified work

### Environment and data preparation

- Conda environment: `tva-thesis`, Python 3.11, defined in `environment.yml`.
- A Mac CPU smoke configuration exists only for quick development checks.
- CUDA smoke configuration exists for the RTX 4060.
- Processed OnHW-Words500 datasets were validated:
  - 13 numeric IMU channels;
  - 100 Hz target rate;
  - five folds;
  - 25,199 samples per setting in total;
  - 501 unique word labels;
  - WD has the expected writer overlap; WI has zero train/validation writer overlap in every fold.
- Portable paths are used; no machine-specific path should be committed.

Expected local data paths:

```text
data/raw/Words500_dep_R/
data/raw/Words500_indep_R/
data/tva/onhw_words500_wd_word_rh/
data/tva/onhw_words500_wi_word_rh/
```

The `data/` and `results/` directories are deliberately Git-ignored. Keep master archives and completed-run backups on Google Drive, but train from the local SSD.

### Reproducible and resumable training

- CUDA mixed precision is enabled only when the selected device supports it.
- `latest.pth` is atomically written after each completed epoch.
- It contains the model, optimizer, scheduler, mixed-precision scaler, epoch, metric history, random-number states, and DataLoader shuffle state.
- `best_cer.pth` and `best_wer.pth` are saved automatically for newer runs.
- A CUDA resume issue involving RNG states was fixed: checkpoints load through CPU and restore PyTorch RNG byte tensors correctly.

For a stopped run, use its `latest.pth` as the configuration checkpoint and resume on the same software environment whenever possible.

### Character baselines completed (fold 0 only)

These are valid **development** results, not final five-fold thesis results.

| Run | Dataset / architecture | Best CER | Epoch | Best WER | Epoch |
| --- | --- | ---: | ---: | ---: | ---: |
| B0 character WD/RH | BLConv-B + BiLSTM-B + CTC | 12.76% | 286 | 35.98% | 289 |
| B0 character WI/RH | BLConv-B + BiLSTM-B + CTC | 15.60% | 244 | 27.95% | 269 |
| A0 WI alignment model | BLConv-B + UniLSTM-B + CTC | 17.33% | 275 | 32.77% | 293 |

Details:

- B0 WD configuration: `configs/thesis/b0_char_wd_rh.yaml`.
- B0 WI configuration: `configs/thesis/b0_char_wi_rh.yaml`.
- B0 uses the intended recognition architecture and 300 epochs, seed 42, batch size 64.
- WD's original best-epoch weights were not preserved because that run happened before automatic best-checkpoint saving. Its retained `latest.pth` is still usable for development but not identical to the best-CER epoch.
- WI has `best_cer.pth` (epoch 244), `best_wer.pth` (epoch 269), and `latest.pth`.
- A0 has worse recognition accuracy and is **not** a baseline replacement. It was made only to obtain less biased alignment timing.

### CTC forced alignment implementation

The following is implemented, tested, and pushed on `forced-alignment`:

- target-constrained CTC Viterbi alignment: `tva/ctc_alignment.py`;
- simple single-sample runner: `align_sample.py`;
- character confidence/margin and greedy-agreement diagnostics;
- frame-to-raw-input approximate mapping (BLConv temporal reduction is 8x);
- JSON and PNG inspection artifacts;
- local midpoint and complete-region sensor features;
- resumable train-split JSONL boundary exporter: `export_boundaries.py`;
- pair/position descriptive analysis: `analyze_boundaries.py`;
- corrected, writer-balanced force-continuity scorer: `score_continuity.py` and `tva/continuity_scoring.py`.

The unit-test suite had **29 passing tests** at commit `52868fe`.

### Key IMU-alignment findings

1. B0's bidirectional LSTM placed word-initial character boundaries very early, even when very confident. This is expected behaviour when a bidirectional model can use future frames, but makes those timestamps poor physical-boundary estimates.
2. A0 replaces only the recurrent decoder with a UniLSTM. It reduced the median estimated timing offsets for `only`/`first` boundaries from about **-315/-300 ms** to **-65/-47 ms** on WI fold-0 development data.
3. A0 retained 78,953 reliable boundaries (89.4% of 88,292) from 19,907 WI training samples and 42 writers.
4. The complete blank-region measurement finds more contact loss than a fixed 100 ms midpoint window, but region duration and word position confound raw rates. Therefore the scorer corrects for broad position/duration groups.
5. The current IMU score is deliberately an interpretable **force-only development baseline**: 75% corrected local contact and 25% corrected whole-region contact. It is not a final proof of motion continuity.

### Combined IAM+READ tokenizer compatibility audit

The combined tokenizer is already prepared outside TVA. Its frozen artifact is
`iam-read-combined-v1`, SHA-256
`5c5d9f1689a4802fc5e9451e5afe8abdfb090562587ab78ddd343211feb94dd4`,
with 494 classes and CTC blank ID 0.

The 2026-09-15 audit ran every OnHW training label in all five WD and WI folds
through the unmodified tokenizer. The artifact lacks uppercase `Ä` and `Ü`, so
608 of the complete 25,199 samples cannot be encoded. All otherwise encodable
labels passed NFC encode/decode round trips.

Do not load this artifact through TVA's existing `BigramTokenizer`. TVA uses
greedy matching, while the frozen model uses maximum-total-utility dynamic
programming; their segmentations differed on 36.64% of encodable OnHW labels.
A dedicated tokenizer and explicit compatibility adapter are required before
training. The full audit, per-fold counts, leakage rules, and frozen
419-class alphabet projection are in
`docs/TOKENIZER_COMPATIBILITY_AUDIT.md`. The primary adapter was frozen on
2026-09-17 as policy `onhw-words500-rh-iam-read-v1`; exact construction and
all-fold validation results are in `docs/HANDWRITING_BIGRAM_ADAPTER_V1.md`.
The implemented canonical artifact is
`artifacts/tokenizers/onhw_words500_rh_iam_read_v1.json`, SHA-256
`12ce25d8bedc552e6b3497ffb1d07e506b01b34296cc21b970f82a550cbf2bfe`.
TVA's runtime implementation is now available under tokenizer key
`handwriting_bigram`. It matched the DTLR reference exactly on all 501 unique
OnHW words and passed all-label checks over 251,990 fold/split instances.

The matched linguistic baseline is also frozen and generated for all five WD
and five WI folds under
`artifacts/tokenizers/linguistic_bigram/`. It uses tokenizer key
`linguistic_bigram`, selects 359 pairs from distinct fold-training word types,
and applies normalized frequency utility through the same dynamic program. No
validation labels were used for construction. The count-one vocabulary-tail
limitation and complete checksums are recorded in
`docs/LINGUISTIC_BIGRAM_BASELINE_V1.md`.

## 5. Current branch and important files

```text
handwriting-bigram-onhw-v1       Current Bigram integration branch
main                             Stable branch with proposal-aligned docs
forced-alignment                 Isolated IMU alignment implementation branch
configs/thesis/                  Baseline and alignment YAML configurations
docs/THESIS_PROJECT.md           Original detailed IMU-first thesis plan
docs/PROGRESS.md                 Chronological work record
docs/EXPERIMENTS.md              Experiment values and caveats
docs/DECISIONS.md                Scientific decisions and limitations
docs/TOKENIZER_COMPATIBILITY_AUDIT.md  Frozen-model/OnHW compatibility evidence
docs/HANDWRITING_BIGRAM_ADAPTER_V1.md  Frozen 419-class adapter policy
artifacts/tokenizers/onhw_words500_rh_iam_read_v1.json  Canonical adapter
build_handwriting_bigram_adapter.py  Deterministic adapter builder CLI
tva/handwriting_bigram_tokenizer.py  NFC + utility-DP runtime tokenizer
docs/LINGUISTIC_BIGRAM_BASELINE_V1.md  Frozen matched-baseline policy and audit
build_linguistic_bigram_tokenizers.py  Fold-specific baseline builder
docs/FORCED_ALIGNMENT.md         Commands and technical explanation
docs/PROJECT_HANDOFF_RTX.md      This handoff
```

The editable proposal sources mentioned during the earlier handoff are not
present in this clone:

```text
docs/image_primary_thesis_proposal.tex
docs/references.bib
```

The approved PDF has been reviewed separately. Preserve the PDF and its
editable `.tex` and `.bib` sources outside Git unless they are deliberately
added to the repository.

## 6. Bring-up checklist on the RTX PC

1. Clone/pull the repository and check out the branch:

   ```bash
   git fetch origin
   git switch handwriting-bigram-onhw-v1
   git pull --ff-only origin handwriting-bigram-onhw-v1
   ```

2. Create or update the environment:

   ```bash
   conda env create -f environment.yml
   conda activate tva
   ```

   If it already exists:

   ```bash
   conda env update -n tva -f environment.yml --prune
   conda activate tva
   ```

   The checked-in environment file currently declares the historical name
   `tva-thesis`; the RTX machine uses the existing environment name `tva`.

3. Verify that PyTorch sees the NVIDIA GPU before a long run:

   ```bash
   python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA GPU')"
   ```

4. Copy/extract the processed datasets to the expected local `data/tva/...` paths. Do not train from Google Drive.

5. Copy the necessary existing result directories if continuing the IMU path:

   ```text
   results/thesis/baselines/B0_char_wi_rh/
   results/thesis/alignment_models/A0_char_wi_rh_unidirectional/
   results/thesis/boundary_exports/a0_char_wi_rh_unidirectional/
   ```

   At minimum, the A0 `best_cer.pth` and the complete training boundary export are needed to continue its existing fold-0 scoring without retraining.

6. Run the tests before new work:

   ```bash
   python -m unittest discover -s tests -v
   ```

## 7. Useful existing commands

### Train a character B0 run

```bash
python main.py --config configs/thesis/b0_char_wi_rh.yaml
python main.py --config configs/thesis/b0_char_wd_rh.yaml
```

The YAML currently targets fold 0. Change `idx_fold`, result directory, and tokenizer path deliberately for another fold. Never reuse a fold-0 result directory for another fold.

### Inspect one aligned sample (IMU path)

```bash
python align_sample.py \
  --config configs/thesis/a0_char_wi_rh_unidirectional.yaml \
  --checkpoint results/thesis/alignment_models/A0_char_wi_rh_unidirectional/0/checkpoints/best_cer.pth \
  --split val \
  --sample-index 0 \
  --device cuda
```

### Export IMU boundaries (only when a new export is really needed)

```bash
python export_boundaries.py \
  --config configs/thesis/a0_char_wi_rh_unidirectional.yaml \
  --checkpoint results/thesis/alignment_models/A0_char_wi_rh_unidirectional/0/checkpoints/best_cer.pth \
  --device cuda \
  --overwrite
```

Use `--resume` only after an interrupted run with the same output schema. Older exports without complete-region fields need `--overwrite`, not `--resume`.

### Analyse and score an existing IMU export (CPU only)

```bash
python analyze_boundaries.py \
  --input results/thesis/boundary_exports/a0_char_wi_rh_unidirectional/fold0/train_region_v2/boundaries.jsonl \
  --overwrite

python score_continuity.py \
  --input results/thesis/boundary_exports/a0_char_wi_rh_unidirectional/fold0/train_region_v2/boundaries.jsonl \
  --overwrite
```

The scorer writes an auditable `pair_continuity_scores.csv` and a method JSON report beside the export. Do not use a validation export: the program rejects non-training splits by design.

## 8. Recommended next steps

### First: integrate the completed combined tokenizer

The IAM and READ evidence pipelines and the combined tokenizer were completed
in the separate DTLR repository. Do not repeat evidence extraction as the next
TVA step.

1. Use the implemented frozen 419-class adapter and dedicated runtime; do not
   regenerate it from OnHW annotation lists or change its recorded checksum.
2. Use the implemented matched linguistic artifacts. They use the same dynamic
   program with fold-training frequency utilities; full policy and checksums
   are in `docs/LINGUISTIC_BIGRAM_BASELINE_V1.md`.
3. Add matched fold-0 handwriting and linguistic configurations and run
   bounded pipeline smoke tests before full training.

### Controlled OnHW Bigram experiment

1. Use the frozen fold-specific matched linguistic Bigram artifacts; never
   rebuild them using validation labels.
2. Preserve the completed deterministic pre-training coverage report for each
   condition.
3. Run small matched pipeline smoke tests in the `tva` environment.
4. Train fresh BLConv-B + BiLSTM-B + CTC models with all non-tokenizer settings
   held fixed, decode to plain text, and compare CER/WER.
5. Develop on fold 0, freeze the final comparison, then run all five WD and WI
   folds.

### Training-budget estimate

Current full runs use 300 epochs. The RTX 4060 measurement was about **20 seconds per epoch**, so one fold takes roughly **2 hours** including overhead.

| Experiment scope | Approximate training time |
| --- | ---: |
| One tokenizer, one setting, five folds | 10–12 hours |
| Handwriting-aware Bigram for WD + WI | 20–24 hours |
| Linguistic Bigram comparison for WD + WI | another 20–24 hours |
| Bigram handwriting-aware + linguistic, all WD/WI folds | about 40–48 hours |

Tokenizer construction is quick; model training is the expensive part. Run one setting/fold at a time, use unique result directories, and back up each completed run.

### Secondary IMU forced-alignment path

If the image method proves unreliable and time remains, the existing A0 WI
fold-0 pipeline can be used as an additional comparison:

1. Inspect the top/middle/bottom pairs in `pair_continuity_scores.csv` for obvious artifacts (position, case, force fallbacks, low writer coverage).
2. Run sensitivity checks for support thresholds, duration bins, and the 75/25 local/region weights without looking at recognition validation scores.
3. Add a separately reported motion-only score from accelerometer/gyroscope features; then compare force-only, motion-only, and combined variants.
4. Construct a first IMU-informed Bigram tokenizer from eligible pairs.
5. Train a fresh model and compare it with character and linguistic-Bigram baselines before expanding to BPE/Unigram.

## 9. Fair-comparison rules — do not break these

- Keep raw IMU input unchanged across B0, linguistic, and handwriting-aware tokenizer runs.
- Keep architecture, preprocessing, augmentation, epochs, optimizer, learning-rate schedule, batch size, seed, and data split the same unless a change is explicitly being studied.
- Train each tokenizer model from scratch; never initialize its main result from a character-model checkpoint.
- Construct each fold's tokenizer from **training evidence only**.
- Compare handwriting-aware Bigram with a **matched-vocabulary-size linguistic Bigram**, not only with the character model.
- Convert predicted tokens back to plain text before computing CER/WER.
- Do not report fold-0 development values as final thesis results.
- Preserve code commit, configuration, seed, checkpoint, data version, result directory, and backup checksum for every completed run.

## 10. References that the proposal should include

- TVA continuation paper: Li et al., *Tokenization vs. Augmentation: A Systematic Study of Writer Variance in IMU-Based Online Handwriting Recognition* (arXiv:2603.16883).
- OnHW-Words500 benchmark: Ott et al., *Benchmarking Online Sequence-to-Sequence and Character-Based Handwriting Recognition from IMU-Enhanced Pens*, IJDAR, 2022, DOI: 10.1007/s10032-022-00415-6.
- IAM: Marti and Bunke, *The IAM Database: An English Sentence Database for Offline Handwriting Recognition*, IJDAR, 2002.
- READ: Sánchez et al., *ICDAR 2017 Competition on Handwritten Text Recognition on the READ Dataset*, ICDAR 2017, DOI: 10.1109/ICDAR.2017.226.
- DTLR: Baena, Kalleli, and Aubry, *General Detection-Based Text Line Recognition*, NeurIPS 2024.
- REWI if discussing the implementation lineage: Li et al., *Robust and Efficient Writer-Independent IMU-Based Handwriting Recognition*, 2026.

The working BibTeX source is not currently present in this clone and should be
recovered from the proposal source archive if repository-local references are
needed.

## 11. Short status summary for the next chat

> We are continuing a master’s thesis in the TVA repository. The model always
> receives raw 13-channel IMU handwriting signals; only output text labels are
> tokenized. We have completed valid fold-0 character baselines (WD CER 12.76%,
> WI CER 15.60%) and a substantial forced-alignment/IMU-boundary-analysis
> branch. The approved proposal makes connected-component connectivity from
> IAM and READ images, supported by pretrained DTLR character localization,
> the primary handwriting prior. Bigram comes first; BPE and Unigram follow
> only if Bigram is promising. The tokenizers are evaluated on OnHW-Words500
> WD and WI. Image connectivity is not proof of IMU pen continuity. The IMU
> CTC forced-alignment method remains an implemented secondary/fallback path.
