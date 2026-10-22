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

### ED analysis notebooks

Each notebook has one settings cell at the top and ends with a short
plain-language summary generated from its numbers. Shared loading code is in
`tva/ed_analysis.py`. None of them trains anything.

| Notebook | What it shows | Needs |
| --- | --- | --- |
| `analyze_ed_token_distribution.ipynb` | Token frequencies of training labels, validation labels and model predictions; over- and under-predicted tokens; optional comparison across tokenizers | Extracted ED data and a saved run |
| `analyze_ed_linguistic_bigrams.ipynb` | How often each linguistic bigram is used in ED, how concentrated use is, unused bigrams, train vs validation | Extracted ED data |
| `inspect_handwriting_tokenizers.ipynb` | Encoding and decoding of one ED sentence (or your own text) with the handwriting Bigram, BPE and Unigram tokenizers | Frozen tokenizer files; ED data for real samples |

Predicted tokens are the IDs the model actually emits (best CTC path), not
re-encoded predicted text. `TVA_ED_EXTRACTED_DIR` and `TVA_ED_RESULTS_DIR`
override the default `data/tva/ed` and `results/thesis/ed` locations.

To inspect actual handwriting-model predictions one recording at a time, open
`analyze_handwriting_predictions.ipynb`. Select Bigram/BPE/Unigram, WD/WI,
fold and validation index. Its five code cells load the saved recognizer,
run one extracted ED recording, show actual predicted token IDs and CTC frame
decisions, and compare decoded text with the reference and saved training
prediction. It performs inference only and requires local saved checkpoints.
Fresh single-record results are labelled separately from saved batched scores.

### Thesis ED tokenizers

Start with **train_ed_tokenizers.ipynb** in the WSL TVA Python environment.
The notebook declares the ED categories and source paths, imports the tokenizer
classes, and calls their training methods:

```python
from tva.iam_tokenizers import IAMBigramTokenizer
from tva.handwriting_tokenizers import GreedyHandwritingBigramTokenizer

linguistic = IAMBigramTokenizer()
linguistic.train(iam_labels, categories, 224,
                 selection_path=iam_selection, output_path=linguistic_output)

handwriting = GreedyHandwritingBigramTokenizer()
handwriting.train(iam_evidence, categories, output_path=handwriting_output)
```

The linguistic tokenizer runs the original TVA bigram algorithm on IAM training
transcripts. The handwriting tokenizer selects its vocabulary in TVA from the
portable IAM connectivity evidence exported by DTLR. Both have 145 bigrams,
78 ED characters, and
CTC blank (224 classes). Neither selects bigrams from ED labels. The character
baseline uses the original TVA class with a fixed character JSON file (79 classes).

```text
train_ed_tokenizers.ipynb            ED categories, imports, train, example
train_tokenizers.ipynb               Original TVA OnHW notebook, unchanged
train_onhw_handwriting_tokenizer.ipynb  Historical IAM+READ OnHW condition
tva/
  tokenizers.py                     Original TVA tokenizers, unchanged
  iam_tokenizers.py                 IAM input wrapper for original TVA bigram training
  handwriting_tokenizers.py         Handwriting-aware tokenizer classes
  tokenizer_factory.py              Shared recognition/analysis initialization
  tokenizer_compatibility.py        Load-only historical linguistic tokenizers
  handwriting/
    common.py                       Shared frozen-file helpers
    evidence.py                     Portable IAM connectivity export reader
    training.py                     Bigram selection, supported BPE, Unigram EM
    ed.py                           ED bigram alphabet/ID projection
    onhw.py                         Historical OnHW vocabulary preparation
    audit.py                        Optional reconstruction/comparison audits
```

Tokenizer files already exist in `artifacts/tokenizers/ed/handwriting/` and
`artifacts/tokenizers/ed/linguistic/`. See [the artifact index](artifacts/tokenizers/README.md)
for current and archived vocabularies. Each ED method now has its own directory
(for example `ed/handwriting/bpe/`) containing `0.json` through `4.json`.
Configs point to the directory, and TVA loads the requested fold. The five files
contain the same frozen IAM vocabulary; WD/WI share them. Running the ED notebook
relearns the handwriting vocabularies from evidence, verifies exact reproduction,
and refuses to overwrite a differing frozen file. Detailed
audits are separate from the training notebook:

```bash
python -m tva.handwriting.audit --compare --dataset-directory /mnt/c/Users/Ali/Downloads/fau-english-dataset
```

Historical custom linguistic artifacts and loaders in
`tva/tokenizer_compatibility.py` remain for saved experiments and analyze_ed.ipynb. Their retired builders are not needed for new training.
See docs/EXPERIMENTS.md for provenance and methodological differences.

Recognition training is a separate manual step. Current ED configs live in
configs/thesis/ed/: character, handwriting, and original TVA linguistic WD/WI,
all using batch 32. Preview the linguistic recognition runs without training:

```bash
python run_ed_matrix.py --conditions tva_original_wd tva_original_wi --folds 0 1 2 3 4 --dry-run
```

Character configuration selection:

- `character_wd.yaml` / `character_wi.yaml`: single-fold ZIP-backed templates.
- `character_wd_extracted_cv.yaml` / `character_wi_extracted_cv.yaml`:
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

Explicit original-TVA comparison configs are available for fold 0:

```bash
python main.py -c configs/thesis/ed/character_tva_wd.yaml
python main.py -c configs/thesis/ed/character_tva_wi.yaml
```

They use extracted ED data and separate output directories. Existing scores
remain valid because the mappings and encoding/decoding are identical; these
extra recognition runs are optional.

### Matched linguistic BPE and Unigram — ED

The trained text-only controls are ready in `artifacts/tokenizers/ed/linguistic/`.
They use the same 5,694 IAM training lines as handwriting BPE/Unigram, retaining
repeated transcriptions. Both sides call the same training functions and share
the exact BPE/Viterbi segmentation implementation in `tva/subword_tokenizers.py`.
Spaces and punctuation remain explicit tokens; NFC, 2–6-letter pieces, the
145-piece budget, 78 ED characters and CTC blank are matched (224 classes).
The text controls use text occurrence counts with handwriting evidence disabled.
No ED labels or connectivity evidence are read during text-control fitting.
These are custom matched text-only controls; original TVA Hugging Face `bpe`
and `unigram` keys remain available separately.

`train_ed_tokenizers.ipynb` contains simple `LinguisticBPETokenizer.train()` and
`LinguisticUnigramTokenizer.train()` cells. Files already exist, so the notebook
is optional. Start recognition manually, from the TVA root with the `tva` WSL
Python environment. These four configs read the extracted files in
`data/tva/ed/gold_wd` and `data/tva/ed/gold_wi`; ZIP files and
`TVA_ED_DATASET_DIR` are not required:

```bash
python main.py -c configs/thesis/ed/bpe_linguistic_wd.yaml
python main.py -c configs/thesis/ed/bpe_linguistic_wi.yaml
python main.py -c configs/thesis/ed/unigram_linguistic_wd.yaml
python main.py -c configs/thesis/ed/unigram_linguistic_wi.yaml
```

Each config runs fold 0 with the same settings as its handwriting counterpart,
using a separate results directory. Recognition training is not started by
preparing the tokenizer files. Existing handwriting models and results remain
valid.

### ED handwriting BPE and Unigram — fold 0

The frozen IAM sources and ED tokenizer files are already bundled; no notebook
run is required. `train_ed_tokenizers.ipynb` can reproduce their training
with simple class imports and `.train()` calls. These now fit candidates and
merges/probabilities from the bundled IAM evidence, then verify exact frozen-v1
reproduction. Use `.load()` to reuse a tokenizer or `.prepare()` for projection
without fitting. No ED labels are used to select
or fit the vocabularies. Each method has 224 ED classes and 145 selected pieces.
BPE keeps ordered merges; Unigram keeps its original Viterbi scores and uses
forced character fallback for added `%` and `=`.

From the TVA root in the WSL `tva` environment, run manually:

```bash
export TVA_ED_DATASET_DIR=/mnt/c/Users/Ali/Downloads/fau-english-dataset
python main.py -c configs/thesis/ed/bpe_handwriting_wd.yaml
python main.py -c configs/thesis/ed/bpe_handwriting_wi.yaml
python main.py -c configs/thesis/ed/unigram_handwriting_wd.yaml
python main.py -c configs/thesis/ed/unigram_handwriting_wi.yaml
```

These configs match the existing handwriting bigram fold-0 settings and use
separate output directories. Hashes, compatibility checks, fallback policy and
comparison scope are recorded in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md#2026-10-08--frozen-iam-handwriting-bpeunigram-integrated-for-ed-fold-0).

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
