# ED recognition configs

One folder per experiment. Every config trains one fold (`idx_fold: 0`) on the
extracted ED files (`data/tva/ed/gold_wd` or `gold_wi`) with identical settings;
configs differ only in the tokenizer and the result directory
(`tests/test_ed_configs.py` checks this). Each condition has a WD and a WI file.
Run from the TVA root, for example:

```bash
python main.py -c configs/thesis/ed/subword_comparison/bigram_handwriting_wd.yaml
```

| Folder | Question | Configs |
| --- | --- | --- |
| [character/](character) | Baseline without learned pieces | `character` (fold 0), `character_*_cv` (all five folds via `train_cv.py`); both TVA's `CharacterTokenizer` with the standard 79-class file |
| [subword_comparison/](subword_comparison) | Do handwriting-aware tokenizers beat linguistic (text-only) ones? | `bigram_linguistic` (TVA's original bigram) vs `bigram_handwriting`, `bpe_linguistic` vs `bpe_handwriting`, `unigram_linguistic` vs `unigram_handwriting` |
| [letters_only/](letters_only) | Does the bigram result hold when spaces and punctuation are left out of training? | `bigram_linguistic_letters` (TVA bigram, `letters_only=True`), compared with `bigram_handwriting` |
| [compression_matched/](compression_matched) | Is the difference just compression? | `bigram_linguistic_matched` (top 62 letters-only bigrams), `bpe_linguistic_matched` (first 59 BPE merges), each compressing IAM text like its handwriting partner |
| [unigram_trainer/](unigram_trainer) | Is the Unigram difference due to the trainer? | `unigram_linguistic_em`: the handwriting Unigram's EM without connectivity, compared with `unigram_handwriting` |

Partners are always the handwriting configs in `subword_comparison/`.
