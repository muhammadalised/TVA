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

## Boundary feature extraction

The first automatic feature extractor uses the midpoint between adjacent
character emission anchors. It measures separate 50, 100, and 150 ms windows
around that point. Using local windows avoids treating a long CTC blank region
as though every pen lift inside it belonged to the same character boundary.

Each boundary stores alignment reliability from both neighbouring anchors:

- aligned probabilities and confidence margins;
- the minimum of the two values; and
- whether both anchors agree with their local greedy classes.

Each window stores transparent sensor measurements:

- raw force minimum, mean, and centre value;
- force values relative to the recording's 90th-percentile reference;
- force-drop ratio;
- fraction and longest duration below a provisional low-force threshold;
- mean AF, AR, and G magnitudes; and
- derivative energy across the nine AF/AR/G axes.

The provisional low-force threshold is 10% of the recording-level force
reference. It is saved in every JSON file so the calculation is reproducible.
This value, the window size, the alignment-quality filter, and the eventual
continuity score remain development choices; the extractor does not yet accept,
reject, or rank a boundary.

If model padding places an anchor outside the real sensor recording, the local
window is clipped to valid raw samples and the boundary is explicitly marked as
overlapping padding. Such cases can later be excluded rather than being
mistaken for real zero-force data.

## Tests

Run the focused tests from the repository root:

```bash
python -m unittest discover -s tests -v
```

The tests cover ordinary and repeated characters, target-constrained decisions,
invalid targets, insufficient frames, confidence diagnostics, IMU mapping,
candidate boundary regions, force/motion window features, and PNG creation.

## Next development steps

1. Run the updated tool on the three inspected WI samples and compare their
   boundary-feature tables with the plots.
2. Inspect a broader training-fold development sample before selecting the
   low-force rule, window size, or alignment-quality filter.
3. Build the batched exporter after the single-sample feature representation is
   stable.
4. Aggregate reliable occurrence features by character pair before defining
   the final continuity score.
