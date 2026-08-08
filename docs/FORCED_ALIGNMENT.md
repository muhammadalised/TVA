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

## Current boundary

This first implementation returns positions on the model-output timeline. It
does not yet map positions to raw IMU samples. BLConv reduces the time dimension
by a factor of eight, so that conversion must be added and verified explicitly
before motion features are measured.

It also does not yet define the alignment-confidence filter. A useful
confidence measure will be added after inspecting alignments from real samples.

## Tests

Run the focused tests from the repository root:

```bash
python -m unittest discover -s tests -v
```

The tests cover ordinary characters, repeated characters, target-constrained
decisions, invalid targets, and recordings with too few model frames.

## Next development steps

1. Load a trained character checkpoint and run inference on a few fold-0
   samples.
2. Decode target IDs and alignment intervals into readable characters.
3. Map output-frame intervals back to approximate raw IMU positions.
4. Plot character intervals over selected IMU channels.
5. Inspect successful and suspicious examples before defining confidence.
