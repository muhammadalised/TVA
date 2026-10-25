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
linguistic.train('data/tva/iam', categories, 224)

handwriting = HandwritingBigramTokenizer(policy='greedy')
handwriting.train('data/tva/iam', categories, 224, path_evidence=iam_evidence)
```

`HandwritingBigramTokenizer` subclasses TVA's `BigramTokenizer` and keeps its
vocabulary layout, files, `load`, `encode` and `decode`. Only `train` differs:
instead of the most frequent label bigrams it keeps the letter bigrams most
often written connected (highest `exact_alignment_connected_rate` among pairs
with at least `min_count=20` aligned observations). Any dataset works the same
way: a `train.json` for the folds plus a per-bigram connectivity score file.

The segmentation policy is a constructor parameter:

```python
greedy = HandwritingBigramTokenizer(policy='greedy')            # identical to BigramTokenizer
utility_based = HandwritingBigramTokenizer(policy='utility-based')  # max total connected rate
```

Both load the same file; the policy changes only how text is split. ED
experiments use `greedy` so that linguistic and handwriting bigrams differ only
in their vocabularies. Config keys: `bigram` (linguistic),
`handwriting_bigram_greedy`, `handwriting_bigram_utility`.

Both bigram vocabularies have 145 bigrams, 78 ED characters and the CTC blank
(224 classes); neither reads ED labels. TVA's linguistic selection counts all
adjacent pairs, so 35 of its 145 bigrams contain a space, while handwriting
selection only ranks letter pairs. The letters-only control removes that
difference: `train(..., letters_only=True)` counts only letter pairs; spaces
and punctuation stay single-character tokens (`linguistic/bigram_letters`,
configs `bigram_linguistic_letters_{wd,wi}.yaml`). TVA breaks frequency ties in `set()` order,
so the linguistic file is reproduced with `PYTHONHASHSEED=42`. The character
baseline uses the original TVA class with a fixed character JSON file (79 classes).

```text
train_ed_tokenizers.ipynb            ED categories, IAM preparation, train, example
train_tokenizers.ipynb               Original TVA OnHW notebook, unchanged
tva/
  tokenizers.py                     Original TVA tokenizers (+ letters_only option, off by default)
  handwriting_tokenizers.py         Handwriting bigram, BPE and Unigram, text-only Unigram EM (TVA subclasses)
  tokenizer_factory.py              Config key -> tokenizer
  dataset/iam.py                    IAM transcriptions -> TVA train.json
  dataset/ed.py                     ED loader and ED_ALPHABET
  compression_matched_tokenizers.py Compression-matched text-only controls
  handwriting/
    common.py                       Shared frozen-file helpers
    training.py                     Handwriting BPE and Unigram learning (Unigram EM also text-only)
    audit.py                        Optional ED compatibility/comparison audits
```

Tokenizer files already exist in `artifacts/tokenizers/ed/`. See
[the artifact index](artifacts/tokenizers/README.md) for current and archived
vocabularies. Each ED method has its own directory (for example
`ed/handwriting/bigram/`) containing `0.json` through `4.json`. Configs point to
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

Archived bigram files (custom ED linguistic bigram, OnHW conditions) load with
`HandwritingBigramTokenizer`; their saved config keys `handwriting_bigram`,
`linguistic_bigram` and `linguistic_bigram_greedy` still work.
See docs/EXPERIMENTS.md for provenance and methodological differences.

Recognition training is a separate manual step. ED configs live in
[configs/thesis/ed/](configs/thesis/ed/README.md), one folder per experiment
(`character`, `subword_comparison`, `letters_only`, `compression_matched`,
`unigram_trainer`), all using batch 32. Run one config at a time, for example:

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

The same rule as for the bigrams: the linguistic tokenizers are TVA's own
`BPETokenizer` and `UnigramTokenizer`, and the handwriting tokenizers subclass
them and only replace `train()`. Both are trained with letters-only splitting:
learned pieces only join letters, while spaces, digits and punctuation stay
single tokens. (TVA's default whitespace splitting drops the spaces of ED
sentences, so `letters_only=True` is required on ED.) Every tokenizer has the
`<PAD>` blank, 78 ED characters and 145 learned pieces (224 classes); encoding
and decoding are TVA's Hugging Face code for both sides.

```python
get_tokenizer('bpe').train('data/tva/iam', categories, 224, letters_only=True)
get_tokenizer('unigram').train('data/tva/iam', categories, 224, letters_only=True)
UnigramEMTokenizer().train('data/tva/iam', categories, 224)
HandwritingBPETokenizer().train('data/tva/iam', categories, 224, path_evidence=iam_evidence)
HandwritingUnigramTokenizer().train('data/tva/iam', categories, 224, path_evidence=iam_evidence)
```

Handwriting candidates are letter spans of 2–6 characters with at least 20
aligned IAM observations, written connected at least half the time. Handwriting
BPE merges the pair with the most *connected* occurrences at each step;
handwriting Unigram seeds and initializes pieces with connected counts, then
runs EM and likelihood pruning (`tva/handwriting/training.py`).

With connectivity switched off, the handwriting BPE trainer reproduces TVA's
BPE (138 of 145 merges, 1.554 vs 1.553 characters per token on IAM), but the
Unigram EM shares only 103 of 145 pieces with Hugging Face's Unigram trainer.
`UnigramEMTokenizer` is therefore the linguistic partner of the handwriting
Unigram: the same EM, pruning and IAM lines, with every letter span of 2–6
characters seen at least 20 times as a candidate, seeded by its frequency
instead of its connected count. Only the connectivity evidence separates the
two. TVA's Unigram stays as the reference baseline. Hugging Face's Unigram
trainer is also not bit-reproducible (same pieces, slightly different IDs and
scores), so its file is frozen once.

Run fold 0 from the TVA root; results go to `results/thesis/ed/*_v2_{wd,wi}`
(`linguistic_unigram_em_{wd,wi}` for the text-only EM):

```bash
python main.py -c configs/thesis/ed/subword_comparison/bpe_linguistic_wd.yaml
python main.py -c configs/thesis/ed/subword_comparison/bpe_linguistic_wi.yaml
python main.py -c configs/thesis/ed/subword_comparison/unigram_linguistic_wd.yaml
python main.py -c configs/thesis/ed/subword_comparison/unigram_linguistic_wi.yaml
python main.py -c configs/thesis/ed/unigram_trainer/unigram_linguistic_em_wd.yaml
python main.py -c configs/thesis/ed/unigram_trainer/unigram_linguistic_em_wi.yaml
python main.py -c configs/thesis/ed/subword_comparison/bpe_handwriting_wd.yaml
python main.py -c configs/thesis/ed/subword_comparison/bpe_handwriting_wi.yaml
python main.py -c configs/thesis/ed/subword_comparison/unigram_handwriting_wd.yaml
python main.py -c configs/thesis/ed/subword_comparison/unigram_handwriting_wi.yaml
```

The earlier custom BPE/Unigram tokenizers (and their `linguistic_bpe`,
`linguistic_unigram` runs) are superseded; their files remain in git history.

### Compression-matched text-only controls — ED

Handwriting tokenizers merge less than their 145-piece text-only partners
(on IAM, 1.38 vs 1.55 characters per token for BPE, 1.34 vs 1.47 for the
letters-only frequency bigram), and on ED fewer merges
go with lower error. These controls remove that difference: each keeps only the
first *k* pieces of a text-only tokenizer, with *k* chosen so that it covers as
many characters per token on IAM training text as its handwriting partner.

| Control | Pieces kept | CTC classes | IAM characters per token (handwriting partner) |
| --- | --- | --- | --- |
| `linguistic_bigram_matched` | top 62 letters-only TVA bigrams | 141 | 1.336 (1.335) |
| `linguistic_bpe_matched` | first 59 letters-only TVA BPE merges | 138 | 1.379 (1.379) |

Files are in `artifacts/tokenizers/ed/linguistic/{bigram,bpe}_matched/`; they
were built by `tva/compression_matched_tokenizers.py` (`build_fold_tokenizers`)
without reading ED labels and load with TVA's `BigramTokenizer` / `BPETokenizer`
(`selection.json` records the choice of *k*). TVA ranks bigrams by frequency, so
the bigram control is byte-identical to
`BigramTokenizer.train(..., 141, letters_only=True)`. Both controls were rebuilt
on the letters-only TVA tokenizers, so they need new runs. Run fold 0 from the
TVA root:

```bash
python main.py -c configs/thesis/ed/compression_matched/bigram_linguistic_matched_wd.yaml
python main.py -c configs/thesis/ed/compression_matched/bigram_linguistic_matched_wi.yaml
python main.py -c configs/thesis/ed/compression_matched/bpe_linguistic_matched_wd.yaml
python main.py -c configs/thesis/ed/compression_matched/bpe_linguistic_matched_wi.yaml
```

Settings equal the other ED configs; set `idx_fold` to 1–4 for further folds,
or to -1 for `train_cv.py`. Results go to
`results/thesis/ed/linguistic_bigram_matched_v2_{wd,wi}` and
`results/thesis/ed/linguistic_bpe_matched_v2_{wd,wi}`, which
`analyze_ed_compression.ipynb` picks up automatically. Compare each control with
its handwriting partner: if the handwriting tokenizer still has lower error, the
connectivity evidence helps beyond compression.

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
