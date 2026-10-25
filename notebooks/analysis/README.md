# Analysis notebooks

Select the TVA Python kernel, edit each notebook's settings near the top, and
run the cells in order. The first code cell finds the repository root and makes
TVA imports and data paths work from either the root or this directory.
Existing outputs stay under the repository's `results/` directory.

| Notebook | Purpose |
| --- | --- |
| [analyze_ed.ipynb](analyze_ed.ipynb) | ED dataset, folds, signals, tokenization and character-baseline results |
| [analyze_onhw.ipynb](analyze_onhw.ipynb) | OnHW writer/character distributions and fold structure |
| [analyze_ed_compression.ipynb](analyze_ed_compression.ipynb) | Tokenizer compression, recognition-error trends and compression-matched controls |
| [analyze_ed_linguistic_bigrams.ipynb](analyze_ed_linguistic_bigrams.ipynb) | Linguistic bigram usage and train/validation token distributions |
| [analyze_ed_token_distribution.ipynb](analyze_ed_token_distribution.ipynb) | Training-label, validation-label and actual model-prediction token frequencies |
| [analyze_handwriting_predictions.ipynb](analyze_handwriting_predictions.ipynb) | A saved recognizer's prediction for one handwriting recording |
| [inspect_handwriting_tokenizers.ipynb](inspect_handwriting_tokenizers.ipynb) | Token IDs, segmentation and decoding for an ED sentence or custom text |

Data and checkpoints are local and are not bundled with the notebooks. See
[dataset setup](../../docs/SETUP.md) and the
[tokenizer artifact index](../../artifacts/tokenizers/README.md).
These notebooks do not train tokenizers or recognizers. Training and dataset
preparation notebooks remain in the repository root.
