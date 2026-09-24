# Thesis Project Handoff — RTX Machine

**Last updated:** 2026-09-23

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

### New supervisor-supplied FAU English evaluation

An additional English sentence-level IMU dataset is available at
`/mnt/c/Users/Ali/Downloads/fau-english-dataset/` as `gold_wd.zip` and
`gold_wi.zip`. It contains 2,550 seven-channel, 100 Hz recordings from 102
writers and supplies five WD and five WI folds. It is independent of IAM;
IAM is only the external offline-handwriting evidence source for the English
handwriting tokenizer.

Keep the repository boundary strict: DTLR freezes a generic IAM-only English
handwriting vocabulary from IAM training evidence and must not read FAU
annotations. TVA owns the FAU-specific character projection, runtime
segmentation, post-freeze compatibility audit, and recognition training. Only
the dataset's declared `categories` schema—not FAU label frequencies or
validation outcomes—may influence singleton coverage.

Supervisor direction is to evaluate English handwriting tokenizers first and
not use the combined IAM+READ tokenizer for this initial FAU study. DTLR commit
`d2f4631` froze the generic 145-bigram IAM source artifact at
`/home/artellisys/DTLR/poc/frozen/iam-english-handwriting-bigram-v1.json`;
its SHA-256 is
`2fbb81479211454d8ad028d8af76eb26e76b75b80408a1a1e1f4efa15ab26a92`.
Its count-20/rate-0.5 selection thresholds are reproducible but remain
provisional rather than statistically optimal. TVA pinned the exact source and
froze the FAU adapter on 2026-09-21. The adapter has 224 outputs including blank
(78 FAU singletons plus all 145 frozen IAM bigrams), uses greedy left-to-right
segmentation, and has SHA-256
`4c06828ac3b0ae03e98d569b0f3fea1cdfbc0a125f6eeffcc7ffb2a4935f3f52`.
It removes unused IAM `*` and adds `%`, `(`, and `=` as fallback singletons.
The post-freeze audit encoded all 2,550 labels from each archive with zero
blank emissions and zero round-trip failures. The matched 224-class IAM-text
linguistic comparator is also frozen, under tokenizer key
`linguistic_bigram_greedy`, with SHA-256
`0a7e173afdd2a911780c14517e651b9787c3b8b3c0fdedb74f915920b4d4f392`.
It uses 145 frequency-ranked IAM-train bigrams and the identical singleton and
greedy policies. Do not train until the seven-channel preprocessing policy and
data path are fixed and smoke-tested.

This is an additional external-transfer experiment, not permission to replace
or reinterpret the completed OnHW fold-0 results. It also needs a dedicated
seven-channel data configuration; OnHW recognition checkpoints are not
compatible with its input shape. Full audit details and archive hashes are in
`docs/EXPERIMENTS.md` under “FAU English dataset compatibility audit.”

The four FAU fold-0 production configurations are now frozen under
`configs/thesis/fau/`. They use fresh BLConv-B + BiLSTM-B models, seven input
channels, 224 outputs, 300 epochs, augmentation, seed 42, and batch size 64.
Before starting them, run the complete one-epoch systems check:

```bash
conda activate tva
export TVA_FAU_DATASET_DIR=/mnt/c/Users/Ali/Downloads/fau-english-dataset
MPLBACKEND=Agg python main.py \
  --config configs/thesis/fau/timing_bigram_handwriting_wd.yaml
```

Record the reported training and validation time and use `nvidia-smi` in a
second terminal to observe peak memory. The timing checkpoint is diagnostic
and must not initialize a production model.

This timing gate passed on 2026-09-23. The RTX 4060 completed the full WD fold
at batch size 8 with CUDA mixed precision in 82 seconds training plus 4 seconds
validation, without an out-of-memory error. Peak VRAM was not captured. The
projected budget is about 7 hours 10 minutes per 300-epoch condition, or 28
hours 40 minutes for the four fold-0 conditions when run sequentially on that
GPU. The timing checkpoint is excluded from production initialization.

The same batch-8 condition was then measured on the 48 GB RTX A6000: 100
seconds training, 2 seconds validation, and only 2,234 MiB maximum observed
memory, with sampled utilization between 0% and 43%. It was slower than the
4060 run and did not use the A6000 efficiently. No production run has started.
Benchmark larger physical batches before deployment, ignore their diagnostic
CER/WER, account for the reduced optimizer-step count, and freeze one setting
for all 20 five-fold runs.

That benchmark is now complete. Batch 64 took 16 seconds training plus 1
second validation, peaked at 7,248 MiB, and completed without an out-of-memory
error. It is frozen for all 20 runs. This gives 32 updates per fold-0 epoch and
9,600 over 300 epochs; all other training settings remain unchanged. Estimated
sequential A6000 time is about 1 hour 25 minutes per run or 28 hours 20 minutes
for the complete matrix, excluding caching and backup overhead.

As a workstation-outage fallback, batch 32 was also timed on the RTX 4060
laptop. The complete WD fold-0 epoch took 26 seconds training plus 2 seconds
validation and peaked at 5,211 MiB without an out-of-memory error, projecting
to roughly 2 hours 20 minutes for 300 epochs. This fallback changes the update
budget to 64 steps per WD fold-0 epoch. Use it only for conditions compared at
the same batch size; do not combine a laptop batch-32 fold 0 with A6000
batch-64 folds 1--4. A final batch-64 five-fold matrix must rerun fold 0.

The laptop handwriting-aware WD fold-0 fallback run subsequently completed all
300 epochs under commit `9761142e491dc631c2411763fae3677edbfa861a` in about
1 hour 9 minutes. Best validation CER was 0.2427056328 at epoch 206 and best
validation WER was 0.6480386889 at epoch 245. The best-CER and best-WER
checkpoint SHA-256 values are respectively
`35a1f5ce12362bfbb9e2e50ccba249b5e09d517f974b667dcf5cfccf25c4db7b`
and `83f583c05d13d8b1972b880e58311294df35eab332fc623e24a2ccd26e07f993`.
This is only half of the controlled comparison; run the linguistic-Bigram WD
fold-0 condition with the identical batch-32 protocol before interpretation.

The matched linguistic-Bigram run is now complete. Its best CER was
0.3219891586 at epoch 203 and best WER was 0.7117141322 at epoch 270. Against
those values, handwriting-aware Bigram reduced CER by 7.93 percentage points
(24.62% relative) and WER by 6.37 points (8.95% relative). The comparison is
controlled for commit, architecture, WD fold, batch 32, seed, schedule,
augmentation, and greedy segmentation. Treat it as promising one-fold evidence,
not a five-fold or statistically significant conclusion. Do not combine these
batch-32 results with batch-64 folds in the final aggregate.

The matched WI fold-0 pair is also complete at batch 32. Handwriting-aware
achieved CER/WER 0.1642117744/0.4483709273; linguistic achieved
0.1859637922/0.4749373434. This is a 2.18-point CER reduction (11.70% relative)
and 2.66-point WER reduction (5.59% relative). Together with WD fold 0, the
handwriting-aware tokenizer leads on both split regimes, but the evidence is
still limited to one fold and seed. The next scientific step is a complete
five-fold study under one batch protocol, rerunning fold 0 if batch 64 is used.

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

### Handwriting-aware Bigram completed (WI/RH fold 0)

The frozen 419-class handwriting-aware condition completed 300 epochs. Best
validation CER is **15.99% at epoch 272** and best validation WER is **24.62%
at epoch 274**. The two values come from separate saved checkpoints. Against
the fold-0 WI character baseline, WER improves by 3.33 percentage points
(11.9% relative), while CER is 0.38 percentage points worse. This is a mixed
development result; do not attribute it to handwriting evidence until the
matched linguistic Bigram run is complete, and do not make a final claim from
one fold.

The run was interrupted after the training phase of epoch 233 by a
Tkinter/Matplotlib GUI-backend cleanup failure. Its complete epoch-232 state
was resumed at epoch 233 with `MPLBACKEND=Agg` and finished normally. Preserve
both run logs and configuration snapshots. Full metrics, checkpoint hashes,
and recovery details are recorded in `docs/EXPERIMENTS.md`.

### Matched linguistic Bigram completed (WI/RH fold 0)

The 419-class matched linguistic condition completed 300 epochs. Its best CER
and WER occur at epoch 262: **15.97% CER** and **24.91% WER**. Compared with
handwriting Bigram, linguistic has a negligible 0.014-percentage-point CER
advantage, while handwriting has a 0.28-point WER advantage. At their
respective best-WER checkpoints, handwriting recognizes 15 more of the 5,292
validation words exactly. An exploratory paired exact test gives `p = 0.497`,
so the fold-0 gap is not reliable evidence of superiority. Treat the current
WI result as both Bigram conditions improving WER relative to the character
baseline, with no established difference between their evidence sources.
Complete the remaining folds before making the thesis claim.

### Handwriting-aware Bigram completed (WD/RH fold 0)

The frozen handwriting-aware condition completed 300 epochs with **20.24% CER
at epoch 276** and **39.87% WER at epoch 281**. Relative to the fold-0 WD
character baseline, this is 7.49 CER points and 3.89 WER points worse. At their
respective best-WER checkpoints, handwriting recognizes 196 fewer of the 5,036
validation words exactly; an exploratory paired exact test gives `p =
6.02e-13`. This is a substantial negative WD fold-0 result. It contrasts with
the positive WI WER change and is directionally compatible with the proposed
larger WI benefit, but the matched linguistic WD run is required before
interpreting the tokenizer-evidence source. Full checkpoint hashes are in
`docs/EXPERIMENTS.md`.

### Matched linguistic Bigram completed (WD/RH fold 0)

The 419-class matched linguistic condition completed 300 epochs with **20.02%
CER at epoch 282** and **40.07% WER at epoch 278**. Linguistic is 7.26 CER
points and 4.09 WER points worse than the fold-0 WD character baseline.
Against handwriting Bigram, linguistic has a 0.22-point CER advantage and
handwriting has a 0.20-point WER advantage—only ten additional exactly correct
words among 5,036. An exploratory paired exact test gives `p = 0.716`.

The completed fold-0 pattern is therefore: both Bigram conditions improve WER
over character in WI, both degrade WD, and handwriting versus linguistic is a
near-tie in both settings. Fold 0 does not support a handwriting-specific
advantage. Run the remaining folds before making the final thesis claim. Full
metrics and hashes are in `docs/EXPERIMENTS.md`.

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
train_handwriting_tokenizer.ipynb  Reproducible handwriting adapter notebook
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

### FAU English external-transfer experiment

The IAM handwriting adapter, matched IAM-text comparator, and both greedy
runtimes are now complete. The authenticated seven-channel ZIP loader and all
four bounded WD/WI smoke runs are also complete. Next:

1. Freeze the full-training architecture, augmentation, batch size, and epoch
   schedule for both tokenizers and both distributions.
2. Add four full fold-0 configurations using fresh seven-channel models; do
   not reuse OnHW's 13-channel checkpoints.
3. Run one bounded full-architecture CUDA timing check, then launch fresh
   matched recognition runs.

Canonical files:

- `artifacts/tokenizers/source/iam-english-handwriting-bigram-v1.json`
- `artifacts/tokenizers/fau_english_iam_handwriting_bigram_greedy_v1.json`
- `artifacts/tokenizers/source/iam-train-letter-bigram-counts-v1.json`
- `artifacts/tokenizers/fau_english_iam_linguistic_bigram_greedy_v1.json`
- `build_fau_handwriting_bigram_adapter.py`
- `build_fau_linguistic_bigram_adapter.py`
- `audit_fau_handwriting_bigram.py`
- `tva/fau_handwriting_bigram.py`
- `tva/fau_linguistic_bigram.py`
- `tests/test_fau_handwriting_bigram.py`
- `tests/test_fau_linguistic_bigram.py`
- `tva/dataset/fau.py`
- `tests/test_fau_dataset_pipeline.py`
- `configs/thesis/fau/smoke_bigram_{handwriting,linguistic}_{wd,wi}.yaml`

To run a smoke test from the current dataset location:

```bash
export TVA_FAU_DATASET_DIR=/mnt/c/Users/Ali/Downloads/fau-english-dataset
MPLBACKEND=Agg python main.py \
  --config configs/thesis/fau/smoke_bigram_handwriting_wd.yaml
```

The smoke configurations deliberately use BLConv-S + BiLSTM-S and tiny sample
limits. They are pipeline diagnostics and must not be reused as thesis-result
configurations.

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

This integration step is complete as of 2026-09-18. The four configurations
are under `configs/thesis/`. All four passed bounded CPU end-to-end smoke tests,
and their saved output heads were verified as 419 classes. A subsequent
bounded handwriting-WD run passed on the RTX 4060 with CUDA mixed precision.
The full fold-0 jobs may now start from scratch.

### Controlled OnHW Bigram experiment

1. Use the frozen fold-specific matched linguistic Bigram artifacts; never
   rebuild them using validation labels.
2. Preserve the completed deterministic pre-training coverage report for each
   condition.
3. Preserve the completed CPU and CUDA pipeline smoke checks.
4. Train fresh BLConv-B + BiLSTM-B + CTC models with all non-tokenizer settings
   held fixed, decode to plain text, and compare CER/WER.
5. Develop on fold 0, freeze the final comparison, then run all five WD and WI
   folds.

### Training-budget estimate

Current full runs use 300 epochs. A full-data handwriting-aware WD/RH fold-0
timing run on the RTX 4060 measured about **29 seconds for training plus 4
seconds for validation per epoch**. At roughly 33 seconds per complete epoch,
one 300-epoch fold is approximately **2 hours 45 minutes**, excluding one-time
startup, dataset caching, and backup overhead.

| Experiment scope | Approximate training time |
| --- | ---: |
| Four fold-0 Bigram development runs | about 11 hours |
| One tokenizer, one setting, five folds | about 14–15 hours |
| Handwriting-aware Bigram for WD + WI | about 28–30 hours |
| Linguistic Bigram comparison for WD + WI | another 28–30 hours |
| Bigram handwriting-aware + linguistic, all WD/WI folds | about 56–60 hours |

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
