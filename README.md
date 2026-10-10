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

### Thesis ED tokenizers

Start with **train_ed_tokenizers.ipynb** in the WSL TVA Python environment.
The notebook declares the ED categories and source paths, imports the tokenizer
classes, and calls their training methods:

```python
from tva.handwriting_tokenizers import IAMBigramTokenizer, GreedyHandwritingBigramTokenizer

linguistic = IAMBigramTokenizer()
linguistic.train(iam_labels, categories, 224,
                 selection_path=iam_selection, output_path=linguistic_output)

handwriting = GreedyHandwritingBigramTokenizer()
handwriting.train(iam_handwriting, categories, output_path=handwriting_output)
```

The linguistic tokenizer runs the original TVA bigram algorithm on IAM training
transcripts. The handwriting tokenizer prepares the ED vocabulary from the
frozen IAM handwriting evidence. Both have 145 bigrams, 78 ED characters, and
CTC blank (224 classes). Neither selects bigrams from ED labels. The character
baseline uses the configured alphabet directly (79 classes).

```text
train_ed_tokenizers.ipynb            ED categories, imports, train, example
train_tokenizers.ipynb               Original TVA OnHW notebook, unchanged
train_onhw_handwriting_tokenizer.ipynb  Historical IAM+READ OnHW condition
tva/
  tokenizers.py                     Original TVA tokenizers, unchanged
  handwriting_tokenizers.py         Thesis tokenizer classes and integration
  handwriting/
    common.py                       Shared frozen-file helpers
    iam.py                          IAM transcript validation and TVA training
    ed.py                           IAM handwriting vocabulary for ED
    onhw.py                         Historical OnHW vocabulary preparation
    audit.py                        Optional reconstruction/comparison audits
```

Tokenizer files already exist in artifacts/tokenizers/. Running the ED notebook
reproduces them and refuses to overwrite a differing frozen file. Detailed
audits are separate from the training notebook:

```bash
python -m tva.handwriting.audit --compare --dataset-directory /mnt/c/Users/Ali/Downloads/fau-english-dataset
```

Historical custom linguistic artifacts and loaders remain for saved experiments
and analyze_ed.ipynb. Their retired builders are not needed for new training.
See docs/EXPERIMENTS.md for provenance and methodological differences.

Recognition training is a separate manual step. Current ED configs live in
configs/thesis/ed/: character, handwriting, and original TVA linguistic WD/WI,
all using batch 32. Preview the linguistic recognition runs without training:

```bash
python run_ed_matrix.py --conditions tva_original_wd tva_original_wi --folds 0 1 2 3 4 --dry-run
```

Completed experiments retain their exact saved configs under results/.

### ED handwriting BPE and Unigram — fold 0

The frozen IAM sources and ED tokenizer files are already bundled; no notebook
run is required. `train_ed_tokenizers.ipynb` can reproduce the ED preparation
with simple class imports and `.train()` calls. No ED labels are used to select
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
