# Thesis Experiment Log

This file records scientific runs and important development experiments. All
reported values must be traceable to the saved configuration, log, metrics,
predictions, code version, and experiment backup.

## Bigram pipeline smoke validation — WD/RH and WI/RH fold 0

### Setup and result

- Date completed: 2026-09-18
- Conditions: handwriting-aware Bigram and matched linguistic Bigram, each on
  WD/RH and WI/RH fold 0
- Scope: 16 training samples, eight validation samples, one CPU epoch, batch
  size two, seed 42
- Result: all four conditions completed training and validation and saved
  resumable checkpoints.
- Checkpoint verification: all four output heads contain 419 classes;
  `decoder.fc.weight` is `(419, 256)` and `decoder.fc.bias` is `(419,)`.
- Artifacts: `results/thesis/development/smoke_bigram_*/0/`

CER and WER were both 1.0 in every run, which is expected for a single epoch on
16 samples. These values are pipeline diagnostics and must not be reported as
recognition evidence. A subsequent bounded run detected the NVIDIA GeForce RTX
4060 Laptop GPU and successfully exercised CUDA mixed precision for the
handwriting-aware WD condition. The pipeline is therefore cleared for full
fold-0 training.

### Full-data timing measurement

A separate one-epoch timing run used the complete handwriting-aware WD/RH
fold-0 training and validation splits on the RTX 4060. Training took 29
seconds and validation took four seconds. The resulting estimate is about 2
hours 45 minutes for one 300-epoch fold, or about 11 hours for the four fold-0
Bigram conditions run sequentially. This run measures computational cost only;
its one-epoch recognition metrics and checkpoint are not experimental results
and must not be used to initialize a full run.

## Handwriting-aware Bigram — WI/RH fold 0

### Setup

- Date completed: 2026-09-18
- Dataset: OnHW-Words500, writer-independent, right-handed, fold 0
- Tokenizer: frozen IAM+READ handwriting-aware Bigram adapter
- Tokenizer classes: 419 including CTC blank ID 0
- Segmentation: frozen maximum-total-utility dynamic program
- Architecture: BLConv-B + BiLSTM-B + CTC
- Seed: 42
- Epochs: 300 (epochs 0–299)
- Batch size: 64
- Training machine: WSL laptop with NVIDIA GeForce RTX 4060
- Configuration: `configs/thesis/bigram_handwriting_wi_rh.yaml`
- Code commit at training start: `0298374`

### Best validation results

| Metric | Value | Epoch |
| --- | ---: | ---: |
| Levenshtein distance | 0.872260 | 272 |
| Character error rate | 0.159850 (15.99%) | 272 |
| Word error rate | 0.246221 (24.62%) | 274 |
| Average reference length | 5.456727 characters | Not optimized |

The metric optima belong to different checkpoints. At the best-CER checkpoint
(epoch 272), WER is 0.248110 (24.81%). At the best-WER checkpoint (epoch 274),
CER is 0.160266 (16.03%). Do not present the independently best CER and WER as
if they came from one model state.

### Fold-0 character comparison

Compared with the B0 WI/RH character run, independently best WER improves from
27.95% to 24.62%: a 3.33-percentage-point reduction, or an 11.9% relative
error reduction. Independently best CER changes from 15.60% to 15.99%, a
0.38-percentage-point increase. This mixed fold-0 result suggests that the
handwriting-aware labels improve exact-word recognition without improving
character error. It is a development observation, not a final thesis claim.
The matched linguistic Bigram run and all five folds are still required before
attributing the WER change to handwriting-derived pair evidence.

### Interruption, recovery, and artifacts

The first process completed training epoch 233 but stopped before its
validation/checkpoint stage because Tkinter objects from Matplotlib's GUI
backend were destroyed outside the main loop. The intact `latest.pth` from
epoch 232 contained the complete model, optimizer, scheduler, scaler, random,
DataLoader-generator, and metrics states. Training resumed at epoch 233 with
`MPLBACKEND=Agg` and completed epoch 299. Epoch 233 was therefore repeated;
the complete resumed metrics file contains exactly epochs 0–299.

Checkpoint SHA-256 values:

| Checkpoint | Epoch | SHA-256 |
| --- | ---: | --- |
| `best_cer.pth` | 272 | `e8037512f745b2105767ba519cf5118394f8e23fda163ba791a99fea688fdeef` |
| `best_wer.pth` | 274 | `3ee167a19f829a43436f63dfd843368207b160ad13e52f2d828100080438f3d8` |
| `latest.pth` | 299 | `a7beef28b9fd29095f52ec5a8729898280c560d51727c8456f25572abc77d10f` |

The complete run directory is
`results/thesis/bigram/handwriting_wi_rh/0/`. Preserve both logs and resolved
configuration snapshots so the interruption and resume remain auditable.
External backup location and checksum: not yet recorded.

## B0 character baseline — WD/RH fold 0

### Setup

- Date completed: 2026-08-02
- Dataset: OnHW-Words500, writer-dependent, right-handed, fold 0
- Method: raw 13-channel IMU to character-level CTC labels
- Architecture: BLConv-B + BiLSTM-B
- Seed: 42
- Epochs: 300 (epochs 0–299)
- Batch size: 64
- Training machine: WSL laptop with NVIDIA GeForce RTX 4060
- Configuration: `configs/thesis/b0_char_wd_rh.yaml`

### Best validation results

| Metric | Value | Epoch |
| --- | ---: | ---: |
| Levenshtein distance | 0.722597 | 286 |
| Character error rate | 0.127563 (12.76%) | 286 |
| Word error rate | 0.359809 (35.98%) | 289 |
| Average reference length | 5.664615 characters | Not optimized |

The original paper's WD character result is aggregated across five folds, so
it must not be directly compared with this single-fold value.

### Artifacts and caveats

- The complete run directory was archived and backed up by the researcher.
- `latest.pth` and the epoch-299 milestone checkpoint were retained.
- This run preceded automatic best-CER/best-WER checkpoint saving. Metrics and
  predictions for epochs 286 and 289 are preserved, but their exact model
  weights are not. The epoch-299 model can be used for initial forced-alignment
  development.
- Backup archive checksum: not yet recorded in this log.

## B0 character baseline — WI/RH fold 0

### Setup

- Date completed: 2026-08-02
- Dataset: OnHW-Words500, writer-independent, right-handed, fold 0
- Method: raw 13-channel IMU to character-level CTC labels
- Architecture: BLConv-B + BiLSTM-B
- Seed: 42
- Epochs: 300 (epochs 0–299)
- Batch size: 64
- Training machine: WSL laptop with NVIDIA GeForce RTX 4060
- Configuration: `configs/thesis/b0_char_wi_rh.yaml`

### Best validation results

| Metric | Value | Epoch |
| --- | ---: | ---: |
| Levenshtein distance | 0.851474 | 244 |
| Character error rate | 0.156041 (15.60%) | 244 |
| Word error rate | 0.279478 (27.95%) | 269 |
| Average reference length | 5.456727 characters | Not optimized |

These are single-fold development results. They must not be presented as the
final writer-independent result, which will require the agreed five-fold
evaluation.

### Artifacts and use

- The run used automatic validation-selected checkpoint saving.
- `best_cer.pth` from epoch 244 is the preferred checkpoint for WI forced
  alignment because alignment quality depends most directly on character
  recognition.
- `best_wer.pth` from epoch 269 is retained for recognition comparison, while
  `latest.pth` remains the resumable end-of-training checkpoint.
- The complete run directory should be archived before alignment development.
- Backup archive checksum: not yet recorded in this log.
