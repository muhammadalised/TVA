# Thesis Experiment Log

This file records scientific runs and important development experiments. All
reported values must be traceable to the saved configuration, log, metrics,
predictions, code version, and experiment backup.

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

## A0 unidirectional character aligner — WI/RH fold 0

### Planned setup

- Date prepared: 2026-08-09
- Dataset: OnHW-Words500, writer-independent, right-handed, fold 0
- Purpose: alignment-only temporal-localization comparison
- Architecture: BLConv-B + UniLSTM-B
- Controlled change from B0: recurrent decoder direction only
- Seed: 42
- Maximum epochs: 300 (validation-selected checkpoint)
- Batch size: 64
- Configuration: `configs/thesis/a0_char_wi_rh_unidirectional.yaml`
- Output: `results/thesis/alignment_models/A0_char_wi_rh_unidirectional`

### Acceptance check

Use `best_cer.pth` to repeat the sample alignments and full training boundary
analysis. Compare reliable `only`, `first`, `middle`, and `final` timing offsets,
blank durations, anchor agreement, and sample support with B0. The main question
is whether A0 removes the 80 ms word-initial concentration; recognition metrics
are secondary because A0 does not replace the B0 recognition baseline.

A0 is not fully causal because BLConv retains centred convolutions and
sequence-wide instance normalization. This experiment isolates the effect of
the recurrent decoder direction before considering a larger encoder change.
