# Tokenizer artifacts

The ED tokenizers are already prepared. You can train recognizers without running
`train_ed_tokenizers.ipynb` again.

```text
artifacts/tokenizers/
  ed/
    character/
      char/               0.json ... 4.json
    linguistic/
      bigram/             0.json ... 4.json, 0.scores.json ... 4.scores.json
      bigram_letters/     0.json ... 4.json, 0.scores.json ... 4.scores.json
      bpe/                0.json ... 4.json
      unigram/            0.json ... 4.json
      bigram_matched/     0.json ... 4.json, scores, selection.json
      bpe_matched_v3/     0.json ... 4.json, selection.json
      unigram_matched_v3/ 0.json ... 4.json, selection.json
      bigram_drawn/       draw0/, draw1/, draw2/ (each 0.json ... 4.json, scores, selection.json), draws.json
      unigram_em/         superseded (runs before D024)
      bpe_matched/        superseded (runs before D024)
      unigram_matched/    superseded (runs before D024)
    handwriting/
      bigram_v3/          0.json ... 4.json, 0.scores.json ... 4.scores.json
      bpe_v3/             0.json ... 4.json
      unigram_v3/         0.json ... 4.json
      bigram/             superseded (runs before D024)
      bpe/                superseded (runs before D024)
      unigram/            superseded (runs before D024)
  archive/                Tokenizers for historical ED and OnHW experiments
```

Configs set `dir_tokenizer` to a method directory, for example
`artifacts/tokenizers/ed/handwriting/bpe_v3`. TVA loads `<idx_fold>.json` from it.
An explicit JSON path is also supported.

The five fold files within each directory are **byte-identical copies**. Learned
vocabularies use IAM training data independently of ED folds, and the character
tokenizer uses the declared alphabet. WD/WI share these files; the file names
do not imply separate fitting on ED training labels.

## How they are built (frozen, docs/DECISIONS.md D024)

Every learned file is written by **TVA's own `.train()`** on IAM, in TVA's
format, and loads with the config keys `bigram` (or `bigram_dp`), `bpe` and
`unigram`. A handwriting tokenizer passes one extra input, `allowed_pairs`: the
most connected IAM letter pairs from the
[connectivity evidence](../evidence/README.md). A bigram file's
`<fold>.scores.json` holds the score that selected each bigram (text count, or
connected rate for handwriting); only the `dp` segmentation policy reads it (D025).

| Family | Method | Fold 0 file | Built with |
| --- | --- | --- | --- |
| Character | Original TVA character | [char/0.json](ed/character/char/0.json) | [Comparison](ed/character/char/0.comparison.json) |
| Linguistic | Original TVA bigram | [bigram/0.json](ed/linguistic/bigram/0.json) | `BigramTokenizer.train(..., 224)`; [provenance](ed/linguistic/bigram/0.provenance.json), [audit](ed/linguistic/bigram/0.audit.json) |
| Linguistic | Original TVA bigram, letters only (matched partner) | [bigram_letters/0.json](ed/linguistic/bigram_letters/0.json) | `letters_only=True` |
| Linguistic | Original TVA BPE, letters only | [bpe/0.json](ed/linguistic/bpe/0.json) | `letters_only=True` |
| Linguistic | Original TVA Unigram, letters only | [unigram/0.json](ed/linguistic/unigram/0.json) | `letters_only=True` |
| Handwriting | Bigram: the 145 most connected letter pairs | [bigram_v3/0.json](ed/handwriting/bigram_v3/0.json) | `HandwritingBigramTokenizer` |
| Handwriting | BPE across the 145 most connected pairs | [bpe_v3/0.json](ed/handwriting/bpe_v3/0.json) | `HandwritingBPETokenizer` |
| Handwriting | Unigram across the 145 most connected pairs | [unigram_v3/0.json](ed/handwriting/unigram_v3/0.json) | `HandwritingUnigramTokenizer` |
| Linguistic | Compression-matched bigram (62 bigrams, 141 classes) | [bigram_matched/0.json](ed/linguistic/bigram_matched/0.json) | [Selection](ed/linguistic/bigram_matched/selection.json) |
| Linguistic | Compression-matched BPE (71 merges, 150 classes) | [bpe_matched_v3/0.json](ed/linguistic/bpe_matched_v3/0.json) | [Selection](ed/linguistic/bpe_matched_v3/selection.json) |
| Linguistic | Compression-matched Unigram (36 pieces, 115 classes) | [unigram_matched_v3/0.json](ed/linguistic/unigram_matched_v3/0.json) | [Selection](ed/linguistic/unigram_matched_v3/selection.json) |
| Linguistic | Size- and compression-matched bigrams (3 draws of 145 by text frequency) | [draw0/0.json](ed/linguistic/bigram_drawn/draw0/0.json), [draw1](ed/linguistic/bigram_drawn/draw1/0.json), [draw2](ed/linguistic/bigram_drawn/draw2/0.json) | [Draws](ed/linguistic/bigram_drawn/draws.json), `selection.json` per draw |

Records alongside `0.json` describe the shared tokenizer, so they are not
replicated for the other four copies.

`train_ed_tokenizers.ipynb` declares the categories and calls `.train()` on IAM
(`PYTHONHASHSEED=42`, because TVA breaks bigram ties in `set()` order). Its
final cell publishes `0.json` through `4.json` (and the scores files) with
`save_fold_tokenizers`. The compression-matched controls are TVA's tokenizers
trained with 79 + *k* classes (`tva/compression_matched_tokenizers.py`); the
drawn bigrams are TVA's bigram with drawn allowed pairs
(`tva/drawn_bigram_tokenizers.py`). Hugging Face's Unigram trainer is not
bit-reproducible, so Unigram files are frozen once and checked by their pieces.

## Superseded files

Files from before D024 stay at their paths, so saved configs of completed runs
still decode correctly:

- `ed/handwriting/bigram/`: the same 145 pairs as `bigram_v3`, with IDs in
  alphabetical order and connected rates stored inside the file as `utility`
  (key `handwriting_bigram_greedy`, or `LegacyBigramTokenizer`).
- `ed/handwriting/{bpe,unigram}/`, `ed/linguistic/unigram_em/`: built by the
  retired custom trainers (`tva/handwriting/training.py`, in git history).
- `ed/linguistic/{bpe,unigram}_matched/`: controls for those handwriting files.

## Historical runs and compatibility

`archive/` retains the custom ED linguistic bigram, the IAM+READ OnHW handwriting
bigram and the old OnHW linguistic fold files. They keep their historical layout.
Completed recognition configs and experiment logs are preserved as recorded.
The path resolver redirects their known former artifact paths to the current
locations, including the preceding versioned ED filenames.

This migration preserves all tokenizer and native-model bytes. The handwriting
trainer's reference-directory lookup changed to follow the new layout; current
training reports record its updated implementation checksum. Corpus, evidence,
model and projection checksums are unchanged. Historical report bytes remain in
Git history.
