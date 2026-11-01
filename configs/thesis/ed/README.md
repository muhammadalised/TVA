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
| [letters_only/](letters_only) | Does the bigram result hold when spaces and punctuation are left out of training? | `bigram_linguistic_letters` (TVA bigram, `letters_only=True`, the matched partner), compared with `bigram_handwriting` |
| [compression_matched/](compression_matched) | Is the difference just compression? | TVA's letters-only tokenizers with fewer pieces, each compressing IAM text like its handwriting partner: `bigram_linguistic_matched` (62 bigrams), `bpe_linguistic_matched` (71 merges), `unigram_linguistic_matched` (36 pieces) |
| [size_and_compression_matched/](size_and_compression_matched) | Is it the choice of bigrams, once size and compression are equal? | `bigram_linguistic_drawn{0,1,2}`: 145 bigrams (224 classes) drawn by text frequency from the handwriting bigram's candidates, each compressing IAM text like the handwriting bigram |

Partners are always the handwriting configs in `subword_comparison/`.

Every tokenizer is built by TVA's own trainer; a handwriting tokenizer only
restricts which letter pairs may join ([docs/DECISIONS.md](../../../docs/DECISIONS.md),
D024). All tokenizer files load with the config keys `bigram`, `bpe` and
`unigram`, so a handwriting config differs from its linguistic partner only in
`dir_tokenizer` and `dir_work`. For the bigram segmentation ablation (D025), set
`tokenizer: bigram_dp` on **both** sides of a comparison.

Runs of the handwriting and BPE/Unigram compression-matched tokenizers built
before D024 are in `*_v2_*` (and `handwriting_{wd,wi}`, `linguistic_unigram_em_*`,
`linguistic_unigram_matched_*`); their saved configs still load their files.
The `unigram_trainer/` experiment (text-only Unigram EM) was retired with them.
