# Tokenization vs. Augmentation: A Systematic Study of Writer Variance in IMU-Based Online Handwriting Recognition

This repository contains the official implementation of our paper, [**"Tokenization vs. Augmentation: A Systematic Study of Writer Variance in IMU-Based Online Handwriting Recognition"**](https://arxiv.org/abs/2603.16883), accepted for presentation at **Machine Learning Workshop** of the **20th International Conference on Document Analysis and Recognition (ICDAR 2026)**.

## Introduction

This paper investigates two strategies to address the challenges of uneven character distributions and high inter-writer variability in IMU-based online handwriting recognition: sub-word tokenization and concatenation-based data augmentation. Building upon the robust CNN-BiLSTM baseline, the models evaluate Bigram, Byte-Pair Encoding (BPE), and Unigram tokenizers alongside a novel concatenation strategy.

### Results on the right-handed OnHW-words500 dataset

**Tokenizer Performance on Writer-Independent Split**
| Model / Tokenizer | Vocab Size | CER (%)  | WER (%)   |
| ----------------- | ---------- | -------- | --------- |
| Baseline          | -          | 7.41     | 15.40     |
| BPE               | 300        | 7.95     | 13.45     |
| Unigram           | 500        | 7.90     | 13.29     |
| **Bigram**        | **500**    | **7.20** | **12.99** |

**Concatenation Augmentation Performance on Writer-Dependent Split**
| # Concat | CER (%)  | WER (%)   |
| -------- | -------- | --------- |
| 0        | 14.86    | 45.10     |
| 1        | 11.44    | 38.65     |
| 2        | 10.04    | 34.52     |
| **3**    | **9.73** | **33.63** |

## Installation

1. **Install PyTorch**: Please follow the instructions on the official PyTorch website to install the version appropriate for your system (CUDA/CPU).

2. **Install Dependencies**: Install the remaining required packages using `requirements.txt`.
```bash
pip install -r requirements.txt
```

## Dataset

For commercial reasons, our datasets will not be published. Alternatively, you can use the OnHW public dataset for training and evaluation. In the paper, we use the right-handed subset of the OnHW-words500 dataset. To download the dataset, please visit: https://www.iis.fraunhofer.de/de/ff/lv/dataanalytics/anwproj/schreibtrainer/onhw-dataset.html.

We use a MSCOCO-like structure for the training and evaluation of our dataset. After the OnHW dataset is downloaded, please convert the original dataset to the desired structure with the notebook `prepare_dataset.ipynb`. Please adjust the variables `dir_raw`, `dir_out`, and `writer_indep` accordingly.

## Usage

### Analysis notebooks

All seven analysis and inspection notebooks are in
[`notebooks/analysis/`](notebooks/analysis/README.md). Select the TVA Python
kernel, adjust the settings near the top, then run the cells. The first code
cell locates the repository automatically, whether the kernel starts in the
repository root or the notebook folder. Shared ED loading code is in
`tva/ed_analysis.py`. These notebooks perform analysis and inference only.

| Notebook | What it shows | Needs |
| --- | --- | --- |
| [analyze_ed.ipynb](notebooks/analysis/analyze_ed.ipynb) | ED dataset, folds, signals, tokenization and completed character baselines | Extracted ED data; saved results for recognition summaries |
| [analyze_onhw.ipynb](notebooks/analysis/analyze_onhw.ipynb) | OnHW writer and character distributions and supplied folds | Prepared OnHW datasets |
| [analyze_handwriting_predictions.ipynb](notebooks/analysis/analyze_handwriting_predictions.ipynb) | One recording's predicted token IDs, CTC decisions and decoded text | Extracted ED data and a saved checkpoint |
| [analyze_ed_token_distribution.ipynb](notebooks/analysis/analyze_ed_token_distribution.ipynb) | Token frequencies of training labels, validation labels and model predictions; over- and under-predicted tokens; optional comparison across tokenizers | Extracted ED data and a saved run |
| [analyze_ed_linguistic_bigrams.ipynb](notebooks/analysis/analyze_ed_linguistic_bigrams.ipynb) | How often each linguistic bigram is used in ED, how concentrated use is, unused bigrams, train vs validation | Extracted ED data |
| [inspect_handwriting_tokenizers.ipynb](notebooks/analysis/inspect_handwriting_tokenizers.ipynb) | Encoding and decoding of one ED sentence (or your own text) with the handwriting Bigram, BPE and Unigram tokenizers | Frozen tokenizer files; ED data for real samples |
| [analyze_ed_compression.ipynb](notebooks/analysis/analyze_ed_compression.ipynb) | Characters per token of every tokenizer, recognition error plotted against compression, and the text-only bigram/BPE sizes that compress like their handwriting partners | Frozen tokenizer files; ED data and saved run scores for sections 1–2 |

Predicted tokens are the IDs the model actually emits (best CTC path), not
re-encoded predicted text. `TVA_ED_EXTRACTED_DIR` and `TVA_ED_RESULTS_DIR`
override the default `data/tva/ed` and `results/thesis/ed` locations.

To inspect actual handwriting-model predictions one recording at a time, open
[analyze_handwriting_predictions.ipynb](notebooks/analysis/analyze_handwriting_predictions.ipynb).
Select Bigram/BPE/Unigram, WD/WI, fold and validation index. Its five code cells load the saved recognizer,
run one extracted ED recording, show actual predicted token IDs and CTC frame
decisions, and compare decoded text with the reference and saved training
prediction. It performs inference only and requires local saved checkpoints.
Fresh single-record results are labelled separately from saved batched scores.

### Thesis ED tokenizers

Start with **train_ed_tokenizers.ipynb** in the WSL TVA Python environment.
It follows the original `train_tokenizers.ipynb`: declare the ED categories,
put the training dataset in TVA format, and call `.train(dir_ds, categories, size)`.
The thesis learns every vocabulary from IAM and evaluates it on ED:

```python
from tva.dataset.iam import prepare_iam
from tva.tokenizers import get_tokenizer
from tva.handwriting_tokenizers import HandwritingBigramTokenizer

prepare_iam(iam_labels, iam_selection, 'data/tva/iam')   # IAM -> train.json

linguistic = get_tokenizer('bigram')                      # original TVA, unchanged
linguistic.train('data/tva/iam', categories, 224, letters_only=True)

handwriting = HandwritingBigramTokenizer()                # TVA's train() + allowed pairs
handwriting.train('data/tva/iam', categories, 224, path_evidence=iam_evidence)
```

**Tokenizer construction is frozen** ([docs/DECISIONS.md](docs/DECISIONS.md),
D024 and D025). Every tokenizer, linguistic or handwriting, is trained by TVA's
own `.train()` in `tva/tokenizers.py`. A handwriting tokenizer adds one input,
`allowed_pairs`: the letter pairs that may join inside a token. That list is
the only place connectivity enters:

- **Bigram:** only allowed pairs are counted. A size-k handwriting bigram holds
  the k most connected IAM letter pairs (highest `exact_alignment_connected_rate`
  among pairs with at least 20 aligned observations); its linguistic partner
  holds TVA's k most frequent letter pairs.
- **BPE and Unigram:** each IAM line is cut between two letters whose pair is not
  allowed before TVA's trainer sees it. The allowed pairs are the 145 most
  connected ones (written connected at least half of the time).
- `allowed_pairs=None` (the default) is the original TVA tokenizer. The matched
  linguistic partner is TVA's `letters_only=True` version, because handwriting
  tokenizers never join spaces or punctuation.

Ablations change only settings, never this code: the allowed list (its size,
the minimum count, the rate column, the evidence file, or a control list), the
vocabulary size, and the bigram segmentation policy. Every bigram file can be
encoded greedily (config key `bigram`, TVA's default) or with the `dp` policy
(key `bigram_dp`), which keeps the bigrams with the highest total score saved
next to the vocabulary in `<fold>.scores.json` (text count for linguistic
bigrams, connected rate for handwriting bigrams). A policy is always applied
to both sides of a comparison.

All learned ED tokenizers have 145 pieces, 78 ED characters and the CTC blank
(224 classes); none reads ED labels. Run the notebook with `PYTHONHASHSEED=42`:
TVA breaks bigram frequency ties in `set()` order. Hugging Face's Unigram
trainer is not bit-reproducible, so Unigram files are checked by their pieces.
`tokenizers==0.23.1` is pinned. The character baseline uses the original TVA
class with a fixed character JSON file (79 classes).

```text
train_ed_tokenizers.ipynb            ED categories, IAM preparation, train, example
train_tokenizers.ipynb               Original TVA OnHW notebook, unchanged
tva/
  tokenizers.py                     Original TVA tokenizers (+ letters_only, allowed_pairs and the
                                    bigram dp policy, all off by default)
  handwriting_tokenizers.py         Handwriting bigram, BPE and Unigram: evidence -> allowed pairs
  tokenizer_factory.py              Config key -> tokenizer
  dataset/iam.py                    IAM transcriptions -> TVA train.json
  dataset/ed.py                     ED loader and ED_ALPHABET
  compression_matched_tokenizers.py Compression-matched text-only controls
  drawn_bigram_tokenizers.py        Size- and compression-matched bigram draws
  handwriting/
    common.py                       Shared frozen-file helpers
    audit.py                        Optional ED compatibility/comparison audits
```

Tokenizer files already exist in `artifacts/tokenizers/ed/`. See
[the artifact index](artifacts/tokenizers/README.md) for current and archived
vocabularies. Each ED method has its own directory (for example
`ed/handwriting/bigram_v3/`) containing `0.json` through `4.json`. Configs point to
the directory, and TVA loads the requested fold. The five files contain the same
IAM vocabulary; WD/WI share them. The notebook refuses to overwrite a differing
frozen file. Detailed audits are separate from the training notebook:

```bash
python -m tva.handwriting.audit --compare --dataset-directory /mnt/c/Users/Ali/Downloads/fau-english-dataset
```

To check that one tokenizer encodes and decodes every ED label, pass the
`tokenizer` and `dir_tokenizer` values of its config (default: the handwriting
bigram):

```bash
python -m tva.handwriting.audit --tokenizer bpe --dir-tokenizer artifacts/tokenizers/ed/linguistic/bpe
```

Files from before D024 stay at their paths so completed runs still decode
correctly. Archived bigram files (custom ED linguistic bigram, OnHW conditions,
the previous ED handwriting bigram) load with `LegacyBigramTokenizer`; their
saved config keys `handwriting_bigram`, `handwriting_bigram_greedy`,
`linguistic_bigram` and `linguistic_bigram_greedy` still work, and `unigram_em`,
`handwriting_bpe` and `handwriting_unigram` load the previous Hugging Face files.
See docs/EXPERIMENTS.md for provenance and methodological differences.

Recognition training is a separate manual step. ED configs live in
[configs/thesis/ed/](configs/thesis/ed/README.md), one folder per experiment
(`character`, `subword_comparison`, `letters_only`, `compression_matched`,
`size_and_compression_matched`), all using batch 32. Run one config at a time, for example:

```bash
python main.py -c configs/thesis/ed/subword_comparison/bigram_linguistic_wd.yaml
```

Character configuration selection (`configs/thesis/ed/character/`):

- `character_wd.yaml` / `character_wi.yaml`: fold 0 (set `idx_fold` for other folds).
- `character_wd_cv.yaml` / `character_wi_cv.yaml`:
  all-five-fold character runs using the extracted dataset (see docs/SETUP.md).

Superseded character CV/rerun launch templates were removed. Completed
experiments retain their exact saved configs under results/; analysis reads
those snapshots rather than requiring historical launch templates.

### Original TVA character tokenizer — ED

`tokenizer_factory.py` delegates `char` directly to TVA's unchanged
`CharacterTokenizer`; there is no character subclass. The standard artifact
`artifacts/tokenizers/ed/character/char/0.json` was generated with
TVA's own `.train()` from the declared ED alphabet and is shared across folds.
All 79 IDs and all 5,100 WD/WI label encodings match the earlier ED tokenizer.
Existing completed configs with `dir_tokenizer: null` remain compatible by
loading this same file through `main.load_configured_tokenizer`.

The character configs name this file explicitly
(`dir_tokenizer: artifacts/tokenizers/ed/character/char`):

```bash
python main.py -c configs/thesis/ed/character/character_wd.yaml
python main.py -c configs/thesis/ed/character/character_wi.yaml
```

The former `character_tva_{wd,wi}` configs were identical apart from their
result directories and were removed.

### BPE and Unigram — ED

The linguistic tokenizers are TVA's own `BPETokenizer` and `UnigramTokenizer`
with `letters_only=True`: learned pieces only join letters, while spaces,
digits and punctuation stay single tokens (TVA's default whitespace splitting
drops the spaces of ED sentences). The handwriting tokenizers call the same
trainers with the 145 most connected letter pairs as `allowed_pairs` (D024).
Every tokenizer has the `<PAD>` blank, 78 ED characters and 145 learned pieces
(224 classes); encoding and decoding are TVA's Hugging Face code for both sides.

```python
get_tokenizer('bpe').train('data/tva/iam', categories, 224, letters_only=True)
get_tokenizer('unigram').train('data/tva/iam', categories, 224, letters_only=True)
HandwritingBPETokenizer().train('data/tva/iam', categories, 224, path_evidence=iam_evidence)
HandwritingUnigramTokenizer().train('data/tva/iam', categories, 224, path_evidence=iam_evidence)
```

On IAM, the handwriting BPE keeps 71 of TVA's 145 pieces (1.41 vs 1.55
characters per token) and the handwriting Unigram 39 (1.28 vs 1.54). Every
handwriting piece joins only allowed pairs; about half of TVA's BPE pieces and
61% of its Unigram pieces join a pair that is usually written apart.

Run fold 0 from the TVA root. Linguistic results go to
`results/thesis/ed/linguistic_{bpe,unigram}_v2_{wd,wi}`, handwriting results to
`results/thesis/ed/handwriting_{bpe,unigram}_v3_{wd,wi}`:

```bash
python main.py -c configs/thesis/ed/subword_comparison/bpe_linguistic_wd.yaml
python main.py -c configs/thesis/ed/subword_comparison/bpe_linguistic_wi.yaml
python main.py -c configs/thesis/ed/subword_comparison/unigram_linguistic_wd.yaml
python main.py -c configs/thesis/ed/subword_comparison/unigram_linguistic_wi.yaml
python main.py -c configs/thesis/ed/subword_comparison/bpe_handwriting_wd.yaml
python main.py -c configs/thesis/ed/subword_comparison/bpe_handwriting_wi.yaml
python main.py -c configs/thesis/ed/subword_comparison/unigram_handwriting_wd.yaml
python main.py -c configs/thesis/ed/subword_comparison/unigram_handwriting_wi.yaml
```

The handwriting BPE/Unigram built with custom trainers before D024 (files in
`ed/handwriting/{bpe,unigram}/`, runs in `handwriting_{bpe,unigram}_v2_*`) and
the text-only Unigram EM (`ed/linguistic/unigram_em/`, runs in
`linguistic_unigram_em_*`) are superseded. Their files stay so those runs can
still be decoded; the code that built them is in git history.

### Compression-matched text-only controls — ED

Handwriting tokenizers merge less than their 145-piece text-only partners,
and on ED fewer merges go with lower error. These controls remove that
difference: each is TVA's letters-only tokenizer trained with a smaller
vocabulary, 79 + *k* classes, with *k* chosen so that it covers as many
characters per token on IAM training text as its handwriting partner.

| Control | TVA tokenizer with | CTC classes | IAM characters per token (handwriting partner) |
| --- | --- | --- | --- |
| `linguistic_bigram_matched` | 62 bigrams (the top 62 of `bigram_letters`) | 141 | 1.336 (1.335) |
| `linguistic_bpe_matched` | 71 merges (the first 71 of `bpe`) | 150 | 1.409 (1.410) |
| `linguistic_unigram_matched` | 36 pieces (retrained) | 115 | 1.286 (1.284) |

`tva/compression_matched_tokenizers.py` (`build_fold_tokenizers`) trains every
*k* = 1..145 without reading ED labels and keeps the whole compression curve in
`selection.json`. Files are in `artifacts/tokenizers/ed/linguistic/bigram_matched/`
(unchanged by D024: the handwriting bigram kept its pairs) and
`{bpe,unigram}_matched_v3/`. Run fold 0 from the TVA root:

```bash
python main.py -c configs/thesis/ed/compression_matched/bigram_linguistic_matched_wd.yaml
python main.py -c configs/thesis/ed/compression_matched/bigram_linguistic_matched_wi.yaml
python main.py -c configs/thesis/ed/compression_matched/bpe_linguistic_matched_wd.yaml
python main.py -c configs/thesis/ed/compression_matched/bpe_linguistic_matched_wi.yaml
python main.py -c configs/thesis/ed/compression_matched/unigram_linguistic_matched_wd.yaml
python main.py -c configs/thesis/ed/compression_matched/unigram_linguistic_matched_wi.yaml
```

Settings equal the other ED configs; set `idx_fold` to 1–4 for further folds,
or to -1 for `train_cv.py`. Results go to
`results/thesis/ed/linguistic_bigram_matched_v2_{wd,wi}` and
`results/thesis/ed/linguistic_{bpe,unigram}_matched_v3_{wd,wi}`, which
`analyze_ed_compression.ipynb` picks up automatically. Compare each control with
its handwriting partner: if the handwriting tokenizer still has lower error, the
connectivity evidence helps beyond compression.

### Size- and compression-matched bigram draws — ED

The compression-matched bigram has 141 classes against the handwriting
bigram's 224, so it differs in two ways: which bigrams and how many. These
controls keep 145 bigrams (224 classes) and the handwriting bigram's
compression, so only the choice of bigrams differs. Each is a random draw from
the handwriting bigram's own candidates (389 letter bigrams with at least 20
aligned IAM observations), weighted by text frequency (probability ∝
count^α, Gumbel top-k); α is set per draw so that the draw compresses IAM text
like the handwriting bigram. A uniform draw (α = 0) compresses much less
(1.22 ± 0.03), so the weighting is needed.

| Draw | Seed | α | IAM characters per token (handwriting 1.3353) | Bigrams shared with handwriting |
| --- | --- | --- | --- | --- |
| `linguistic_bigram_drawn0` | 2 | 0.542 | 1.3355 | 61 of 145 |
| `linguistic_bigram_drawn1` | 3 | 0.466 | 1.3348 | 62 of 145 |
| `linguistic_bigram_drawn2` | 4 | 0.496 | 1.3346 | 60 of 145 |

Seeds 0 and 1 got no closer than 0.0013 and 0.0022 and were skipped (tolerance
0.001, as close as the compression-matched bigram). Files are in
`artifacts/tokenizers/ed/linguistic/bigram_drawn/draw{0,1,2}/`, built by
`tva/drawn_bigram_tokenizers.py` without reading ED labels. Each is built like
the handwriting bigram: TVA's `BigramTokenizer` with the drawn bigrams as its
allowed pairs, greedy encoding. Run fold 0 from the TVA root:

```bash
python main.py -c configs/thesis/ed/size_and_compression_matched/bigram_linguistic_drawn0_wd.yaml
python main.py -c configs/thesis/ed/size_and_compression_matched/bigram_linguistic_drawn0_wi.yaml
python main.py -c configs/thesis/ed/size_and_compression_matched/bigram_linguistic_drawn1_wd.yaml
python main.py -c configs/thesis/ed/size_and_compression_matched/bigram_linguistic_drawn1_wi.yaml
python main.py -c configs/thesis/ed/size_and_compression_matched/bigram_linguistic_drawn2_wd.yaml
python main.py -c configs/thesis/ed/size_and_compression_matched/bigram_linguistic_drawn2_wi.yaml
```

Results go to `results/thesis/ed/linguistic_bigram_drawn{0,1,2}_{wd,wi}`. If
the handwriting bigram beats all three draws, choosing bigrams by connectivity
helps beyond vocabulary size and compression; if it falls within their range,
it does not.

### Training

In the paper, models are trained in a 5-fold cross validation style, which can be done using the `main.py` to train each fold individually. Please adjust the configurations in the `configs/train.yaml` configuration file accordingly.
```bash
python main.py -c configs/train.yaml
```

Alternatively, you can also train all folds at once sequentially with `train_cv.py`. The script will generate configuration files for all folds in a `temp*` directory and run `main.py` with these configuration files sequentially. After the training is finished, the `temp*` directory will be deleted automatically.
```bash
python train_cv.py -c configs/train.yaml
```

NOTE: Before the training with `train_cv.py`, please make sure the `idx_fold` in `configs/train.yaml` is set to -1.

### OnHW WI character baseline on the workstation

`configs/thesis/b0_char_wi_rh_cv.yaml` runs all five WI/RH folds from scratch.
It retains the fold-0 baseline's model, seed 42, batch 64, 300 epochs,
30-epoch warmup and standard augmentation without concatenation. Its output
directory is `results/thesis/baselines/B0_char_wi_rh_cv_workstation`, so the
original development results remain available.

Sync the updated `train_cv.py` and CV config to the workstation first. The
workstation needs the prepared WI dataset and character tokenizer files
`0.json` through `4.json` under the configured dataset directory.

From the workstation repository, create a persistent terminal:

```bash
cd ~/TVA  # adjust to the workstation checkout
tmux new -s onhw-wi-char-cv
```

Inside tmux, activate the workstation's existing TVA environment (shown here
as `tva-thesis`), preview the run, then start training:

```bash
conda activate tva-thesis
python train_cv.py -c configs/thesis/b0_char_wi_rh_cv.yaml --dry-run
mkdir -p results/thesis/baselines/B0_char_wi_rh_cv_workstation
git rev-parse HEAD > results/thesis/baselines/B0_char_wi_rh_cv_workstation/code_revision.txt
set -o pipefail
python -u train_cv.py -c configs/thesis/b0_char_wi_rh_cv.yaml 2>&1 | tee results/thesis/baselines/B0_char_wi_rh_cv_workstation/console_$(date +%Y%m%d_%H%M%S).log
```

Proceed after the dry run validates folds `[0, 1, 2, 3, 4]`. Detach with
**Ctrl+b**, then **d**; reconnect with `tmux attach -t onhw-wi-char-cv`.
Folds run sequentially on one GPU. `latest.pth`, `best_cer.pth`, and
`best_wer.pth` are saved in each fold's `checkpoints/` directory.

The launcher stops on failure and refuses to overwrite populated fold
directories. `--folds 2 3 4` selects untouched folds if only earlier folds
have completed. To resume an interrupted fold, use a copy of the CV config
with `checkpoint` set to the existing `latest.pth` path and select only that
fold with `--folds N`; then launch the remaining untouched folds with the
original CV config. Do not change the total epoch budget when resuming.

### Evaluation

As we are using cross validation, the results are already given in the output files of training. However, you can always re-evaluate the model with the configuration and weight you want. In that case, please adjust the `test.yaml` file accordingly and run `main.py` with it.
```bash
python main.py -c configs/test.yaml
```

After you get all results of all folds, you can summarize the results and also calculate the #Params and MACs with `evaluate.py`.
```bash
python evaluate.py -c configs/train.yaml
# or
python evaluate.py -c path_to_config_in_work_dir
```

## License

This project is released under the MIT license. Please see the `LICENSE` file for more information.
