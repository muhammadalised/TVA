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
