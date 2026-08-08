# CTC Forced Alignment

## Purpose

The character model predicts a probability for every character at every
model-output frame. During training we also know the correct word label. CTC
Viterbi alignment finds the highest-probability path through those predictions
that is allowed to produce the known label.

For the label `Haus`, the algorithm searches through this expanded sequence:

```text
blank, H, blank, a, blank, u, blank, s, blank
```

At every frame, a path may stay at its current state, advance by one state, or
skip a blank when CTC permits it. The implementation stores the best score and
previous state for every frame/state combination, then follows those saved
choices backwards to recover the alignment.

Repeated characters need special handling. For example, the two `l` characters
in `Hallo` must have a blank between them; otherwise standard CTC collapsing
would turn them into one `l`.

## Implementation

The standalone implementation is in `tva/ctc_alignment.py`. Its main function
accepts one sample at a time:

```python
from tva.ctc_alignment import ctc_viterbi_align

# probabilities: (number of model frames, number of character classes)
# target_ids: known character IDs without CTC blanks
alignment = ctc_viterbi_align(probabilities, target_ids, blank_id=0)

for token in alignment.tokens:
    print(
        token.target_index,
        token.token_id,
        token.start_frame,
        token.end_frame,
    )
```

Intervals use normal Python half-open indexing: `[start_frame, end_frame)`.
Thus an interval `[3, 6)` contains frames 3, 4, and 5.

The returned object also exposes:

- `log_score`: total log-probability of the selected path;
- `expanded_target`: target IDs with CTC blanks inserted;
- `state_path`: expanded-target state selected at every frame; and
- `token_path`: corresponding token ID selected at every frame.

## Single-sample runner

`align_sample.py` connects the alignment algorithm to a trained TVA character
model and a real dataset sample. It deliberately processes only one sample so
checkpoint loading, preprocessing, model inference, greedy decoding, and
forced alignment remain easy to inspect.

Run WI/RH fold 0 with its best-CER checkpoint:

```bash
python align_sample.py \
  --config configs/thesis/b0_char_wi_rh.yaml \
  --checkpoint results/thesis/baselines/B0_char_wi_rh/0/checkpoints/best_cer.pth \
  --split val \
  --sample-index 0 \
  --device cuda
```

Run WD/RH fold 0 with the retained final checkpoint:

```bash
python align_sample.py \
  --config configs/thesis/b0_char_wd_rh.yaml \
  --checkpoint results/thesis/baselines/B0_char_wd_rh/0/checkpoints/latest.pth \
  --split val \
  --sample-index 0 \
  --device cuda
```

Add `--show-path` when the complete frame-by-frame CTC path is useful for
debugging. Without it, the runner prints compact character and boundary tables.
Every run also writes a reusable JSON record and PNG plot under
`results/thesis/alignment_debug/` by default.

## Confidence and approximate IMU mapping

For each target character, the analysis records its mean model probability,
the locally preferred class, the best competing probability, their margin, and
whether the target agrees with the local greedy choice. These values are useful
diagnostics, but they are not yet treated as calibrated confidence or used as a
final exclusion threshold.

BLConv reduces the time dimension by a factor of eight. The tool maps each
model-frame edge to an approximate model-input sample using this ratio and
reports any trailing samples that were not represented by a complete output
frame. These positions are called character emission anchors, not exact
physical stroke start/end times, because the convolutional and bidirectional
layers use neighbouring context.

The blank frames between adjacent character anchors form a candidate boundary
region. A zero-width region means the CTC path moved directly from one
character to the next; it does not by itself prove that the physical motion was
continuous.

## Visualization and JSON

The PNG contains three synchronized timelines:

- normalized magnitudes for the AF, AR, and G sensor groups;
- the raw F channel so force/contact structure remains visible; and
- the model's highest class probability plus each forced character probability.

Green character anchors agree with the local model choice. Red anchors were
inserted by the target constraint despite another locally preferred class.
Orange regions contain intervening CTC blank frames. The dataset metadata order
is `AF(0-2), AR(3-5), G(6-8), M(9-11), F(12)`.

The JSON contains sample/checkpoint metadata, complete CTC paths, character
diagnostics, approximate input positions, and candidate boundary regions. Raw
sensor arrays remain in the dataset and are not duplicated in the JSON.

## Tests

Run the focused tests from the repository root:

```bash
python -m unittest discover -s tests -v
```

The tests cover ordinary and repeated characters, target-constrained decisions,
invalid targets, insufficient frames, confidence diagnostics, IMU mapping,
candidate boundary regions, and PNG creation.

## Next development steps

1. Run the updated single-sample tool with the trained WI best-CER checkpoint
   and inspect several plots.
2. Compare correct greedy predictions with red, target-forced characters.
3. Decide and document an initial alignment-quality filter from development
   examples only.
4. Build the batched exporter after the single-sample representation is stable.
