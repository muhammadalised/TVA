# OnHW forced-alignment Bigram configs

Open [train_force_tokenizers.ipynb](../../../train_force_tokenizers.ipynb)
from the TVA root. Choose `wi` for writer-independent or `wd` for
writer-dependent. The notebook uses the existing export, scoring and
vocabulary-building commands. It shows the inputs and checks the two
vocabularies before you train a recognizer.

| File | WI | WD |
| --- | --- | --- |
| Build both vocabularies | [build_tokenizers.yaml](wi/build_tokenizers.yaml) | [build_tokenizers.yaml](wd/build_tokenizers.yaml) |
| Train the frequency-ranked recognizer | [frequency.yaml](wi/frequency.yaml) | [frequency.yaml](wd/frequency.yaml) |
| Train the force-ranked recognizer | [force.yaml](wi/force.yaml) | [force.yaml](wd/force.yaml) |
| A0 character aligner | [A0 WI config](../a0_char_wi_rh_unidirectional.yaml) | [A0 WD config](../a0_char_wd_rh_unidirectional.yaml) |
| B0 character reference | [B0 WI config](../b0_char_wi_rh.yaml) | [B0 WD config](../b0_char_wd_rh.yaml) |

The two recognition YAMLs for each split differ only in `dir_tokenizer`
and `dir_work`. Use TVA's normal entry point. Run one condition at a time
from the TVA root in the `tva` environment.

```bash
export PYTHONHASHSEED=42
python main.py --config configs/thesis/forced_alignment/wi/frequency.yaml
python main.py --config configs/thesis/forced_alignment/wi/force.yaml
```

Use `wd` in the config path after the WD vocabulary build completes. The
runs train for 300 epochs with validation disabled. Compare their final
epoch-299 checkpoints. The launcher checks both vocabulary files against
the build manifest and requires a fresh output directory for a new run.

| Where to look | What it does |
| --- | --- |
| [tva/force_bigram.py](../../../tva/force_bigram.py) | Selects and verifies the two matched vocabularies |
| [tva/tokenizers.py](../../../tva/tokenizers.py) | Loads the vocabulary and applies TVA's greedy Bigram encoder |
| [tva/continuity_scoring.py](../../../tva/continuity_scoring.py) | Computes the force ranking from the saved training boundaries |
| [export_boundaries.py](../../../tva/forced_alignment/export_boundaries.py) | Exports the training evidence from the configured aligner checkpoint |
| [Forced-alignment notes](../../../docs/FORCED_ALIGNMENT.md#match-the-emitted-vocabularies-for-wi-and-wd-fold-0) | Describes the protocol, saved results and shell commands |

For an interrupted boundary export, rerun the same exporter command with
`--resume`, using the same config and checkpoint. The build notebook
reuses completed outputs. Find the command modules in
[tva/forced_alignment/](../../../tva/forced_alignment/). Run them from the
TVA root with `python -m tva.forced_alignment.<module>`. For example,
`python -m tva.forced_alignment.build_bigram_v0 --help` lists the builder's
inputs. Recognition still uses `main.py --config`.
