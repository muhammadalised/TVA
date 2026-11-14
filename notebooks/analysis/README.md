# Analysis notebooks

Select the TVA Python kernel, edit each notebook's settings near the top, and
run the cells in order. The first code cell finds the repository root and makes
TVA imports and data paths work from either the root or this directory.
Existing outputs stay under the repository's `results/` directory.

| Notebook | Purpose |
| --- | --- |
| [compare_dataset_token_distributions.ipynb](compare_dataset_token_distributions.ipynb) | IAM/ED token distributions and separate OnHW character charts; no checkpoint needed |
| [analyze_ed.ipynb](analyze_ed.ipynb) | ED dataset, folds, signals, tokenization and character-baseline results |
| [analyze_onhw.ipynb](analyze_onhw.ipynb) | OnHW writer/character distributions and fold structure |
| [analyze_ed_compression.ipynb](analyze_ed_compression.ipynb) | Tokenizer compression, recognition-error trends and compression-matched controls |
| [analyze_ed_linguistic_bigrams.ipynb](analyze_ed_linguistic_bigrams.ipynb) | Letters-only vs original and handwriting bigrams: vocabulary, spaces, compression, recognition scores and token usage |
| [analyze_ed_token_distribution.ipynb](analyze_ed_token_distribution.ipynb) | Training-label, validation-label and actual model-prediction token frequencies |
| [analyze_handwriting_predictions.ipynb](analyze_handwriting_predictions.ipynb) | A saved recognizer's prediction for one handwriting recording |
| [inspect_handwriting_tokenizers.ipynb](inspect_handwriting_tokenizers.ipynb) | Token IDs, segmentation and decoding for an ED sentence or custom text |

Data and checkpoints are local and are not bundled with the notebooks. See
[dataset setup](../../docs/SETUP.md) and the
[tokenizer artifact index](../../artifacts/tokenizers/README.md).
These notebooks do not train tokenizers or recognizers. Training and dataset
preparation notebooks remain in the repository root.

For a simple dataset comparison, start with
[compare_dataset_token_distributions.ipynb](compare_dataset_token_distributions.ipynb).
Run All in the TVA kernel. The default is ED WI fold 0 with detailed Bigram charts;
change FAMILY to bpe or unigram for those views. The final summary covers all six
current English tokenizers. Exports go to results/dataset_token_distributions/.
English token counts use the IAM lines compatible with the ED alphabet; excluded
lines are reported. German subword comparisons wait for an approved construction
dataset. No model or tokenizer is trained.
