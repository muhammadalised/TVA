# Thesis Project Handoff — RTX Machine

**Last updated:** 2026-09-15

**Repository:** [muhammadalised/TVA](https://github.com/muhammadalised/TVA)

**Current branch:** `forced-alignment`
**Current code commit:** `52868fe` — *Add corrected force continuity ranking*

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
2. **BPE (later extension):** start with characters and make successive handwriting-guided merges rather than frequency-guided merges.
3. **Unigram (later extension):** generate and score longer candidate strings from their internal handwriting cohesion, then prune/select a vocabulary.

Frequency is still useful as a **reliability gate**: a pair needs enough examples and writer coverage for its handwriting score to be trustworthy. It must not become the main reason a token is selected.

## 2. Important scope update: two possible evidence sources

There are two related approaches discussed with the supervisor. They must not be confused.

### A. Primary proposal direction: offline handwriting-image connectivity

The current proposal direction is to use offline handwriting images as a source of a *handwriting prior*:

1. Use an image dataset such as **IAM** and/or **READ**.
2. Obtain character locations/identities with **DTLR** or compatible text-line-recognition output.
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

### Scope discrepancy to resolve before final experiments

The older repository documents (`THESIS_PROJECT.md` and `DECISIONS.md`) still describe **IMU forced alignment as the main method** and image connectivity as an optional extension. The newer supervisor/proposal discussion makes **image-derived connectivity primary** and forced alignment the fallback.

Before starting expensive five-fold training, update those documents to match the supervisor's final preferred wording. Do not claim that either static image connectivity or CTC alignment gives ground-truth physical pen-stroke boundaries.

## 3. Data and evaluation scope

- **Recognition dataset:** OnHW-Words500.
- **Hand:** right-handed only. Left-handed data is excluded because it is too small; this was confirmed by the supervisor.
- **Settings:** writer-dependent (WD/RH) and writer-independent (WI/RH).
- **Cross-validation:** five folds for final reported results.
- **Metrics:** character error rate (CER) and word error rate (WER), measured after predicted tokens have been concatenated back into ordinary text.
- **Model:** BLConv-B + BiLSTM-B + CTC for the recognition comparisons.

Never let validation recordings influence tokenizer construction. For each fold, derive image/IMU evidence and freeze the tokenizer using the training partition only; then train and evaluate on that fold normally.

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

## 5. Current branch and important files

```text
forced-alignment                 Current branch and remote branch
main                             Original / stable project branch
configs/thesis/                  Baseline and alignment YAML configurations
docs/THESIS_PROJECT.md           Original detailed IMU-first thesis plan
docs/PROGRESS.md                 Chronological work record
docs/EXPERIMENTS.md              Experiment values and caveats
docs/DECISIONS.md                Scientific decisions and limitations
docs/FORCED_ALIGNMENT.md         Commands and technical explanation
docs/PROJECT_HANDOFF_RTX.md      This handoff
```

Two proposal assets are currently **untracked** in this clone:

```text
docs/image_primary_thesis_proposal.tex
docs/references.bib
```

They will not travel through Git unless deliberately added and committed. The formal PR-lab proposal was edited in Overleaf/Downloads, not necessarily in this repository. Preserve its `.tex` and `.bib` files separately.

## 6. Bring-up checklist on the RTX PC

1. Clone/pull the repository and check out the branch:

   ```bash
   git fetch origin
   git switch forced-alignment
   git pull --ff-only origin forced-alignment
   ```

2. Create or update the environment:

   ```bash
   conda env create -f environment.yml
   conda activate tva-thesis
   ```

   If it already exists:

   ```bash
   conda env update -n tva-thesis -f environment.yml --prune
   conda activate tva-thesis
   ```

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

### First: settle and document the primary method

1. Confirm with the supervisor that the **image-derived connectivity method is primary**, while IMU forced alignment is a secondary/fallback experiment.
2. Update `THESIS_PROJECT.md`, `DECISIONS.md`, `PROGRESS.md`, and the formal proposal to make this consistent.
3. Define a small pilot: likely IAM first because its transcriptions and line images are widely used; add READ if German/historical coverage is needed.

### Primary image-based Bigram pilot

1. Obtain/prepare image line samples and their transcripts.
2. Run DTLR (or use suitable character-localization output) and verify that character boxes/identities align with the transcript. This alignment is a critical quality check.
3. Implement transparent CCL-based pair connectivity features:
   - same connected component or not;
   - distance/gap between adjacent components as a secondary measure;
   - support count and writer coverage;
   - filters for uncertain detector/transcript alignments.
4. Build **one handwriting-aware Bigram vocabulary** from training evidence only. Preserve individual characters as fallback tokens.
5. Build a **matched-size linguistic Bigram** vocabulary as the proper baseline. The two vocabularies must have the same/near-identical size.
6. Tokenize the OnHW labels deterministically, train a fresh BLConv-B + BiLSTM-B + CTC model, decode back to text, and compare CER/WER.
7. Screen vocabulary sizes on fold 0, predeclare a small final size set, then run all five folds in both WD and WI settings.

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

If the image method is delayed or weak, continue with the existing A0 WI fold-0 pipeline:

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

The repository's `docs/references.bib` contains the working BibTeX entries.

## 11. Short status summary for the next chat

> We are continuing a master’s thesis in the TVA repository. The model always receives raw 13-channel IMU handwriting signals; only output text labels are tokenized. We have completed valid fold-0 character baselines (WD CER 12.76%, WI CER 15.60%) and a substantial forced-alignment/IMU-boundary-analysis branch. The supervisor's newer preferred method is to make handwriting-aware Bigram/BPE/Unigram tokenizers using connected-component connectivity from offline handwriting images (IAM/READ, with DTLR support) and evaluate them on OnHW-Words500 WD and WI. Bigram comes first. Image connectivity is a cross-modal handwriting prior, not proof of IMU pen continuity. The IMU CTC forced-alignment method remains an implemented fallback/comparison path.
