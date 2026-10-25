# Tokenizer artifacts

The ED tokenizers are already prepared. You can train recognizers without running
`train_ed_tokenizers.ipynb` again.

```text
artifacts/tokenizers/
  ed/
    character/
      char/             0.json, 1.json, 2.json, 3.json, 4.json
    linguistic/
      bigram/           0.json, 1.json, 2.json, 3.json, 4.json
      bigram_letters/   0.json, 1.json, 2.json, 3.json, 4.json
      bpe/              0.json, 1.json, 2.json, 3.json, 4.json
      unigram/          0.json, 1.json, 2.json, 3.json, 4.json
      unigram_em/       0.json, 1.json, 2.json, 3.json, 4.json
      bigram_matched/   0.json, 1.json, 2.json, 3.json, 4.json
      bpe_matched/      0.json, 1.json, 2.json, 3.json, 4.json
      unigram_matched/  0.json, 1.json, 2.json, 3.json, 4.json
    handwriting/
      bigram/           0.json, 1.json, 2.json, 3.json, 4.json
      bpe/              0.json, 1.json, 2.json, 3.json, 4.json
      unigram/          0.json, 1.json, 2.json, 3.json, 4.json
  archive/              Tokenizers for historical ED and OnHW experiments
```

Configs set `dir_tokenizer` to a method directory, for example
`artifacts/tokenizers/ed/handwriting/bpe`. TVA loads `<idx_fold>.json` from it.
An explicit JSON path is also supported.

The five fold files within each directory are **byte-identical copies**. Learned
vocabularies use IAM training data independently of ED folds, and the character
tokenizer uses the declared alphabet. WD and WI share these files; the file names
do not imply separate fitting on ED training labels. Existing vocabulary IDs,
merges, probabilities and recognition results are unchanged.

## Current ED tokenizers

| Family | Method | Fold 0 file | Supporting records |
| --- | --- | --- | --- |
| Character | Original TVA character | [char/0.json](ed/character/char/0.json) | [Comparison](ed/character/char/0.comparison.json) |
| Linguistic | Original TVA bigram | [bigram/0.json](ed/linguistic/bigram/0.json) | [Provenance](ed/linguistic/bigram/0.provenance.json), [audit](ed/linguistic/bigram/0.audit.json) |
| Linguistic | Original TVA bigram, `letters_only=True` | [bigram_letters/0.json](ed/linguistic/bigram_letters/0.json) | Letter pairs only, like the handwriting bigram |
| Linguistic | Original TVA BPE, `letters_only=True` | [bpe/0.json](ed/linguistic/bpe/0.json) | Hugging Face BPE file |
| Linguistic | Original TVA Unigram, `letters_only=True` | [unigram/0.json](ed/linguistic/unigram/0.json) | Hugging Face Unigram file |
| Linguistic | Handwriting Unigram's EM on text alone (frequency seeds) | [unigram_em/0.json](ed/linguistic/unigram_em/0.json) | Hugging Face Unigram file; partner of the handwriting Unigram |
| Linguistic | Compression-matched bigram (top 62 of `bigram_letters`) | [bigram_matched/0.json](ed/linguistic/bigram_matched/0.json) | [Selection](ed/linguistic/bigram_matched/selection.json) |
| Linguistic | Compression-matched BPE (first 59) | [bpe_matched/0.json](ed/linguistic/bpe_matched/0.json) | [Selection](ed/linguistic/bpe_matched/selection.json) |
| Linguistic | Compression-matched Unigram (text-only EM, 55 pieces) | [unigram_matched/0.json](ed/linguistic/unigram_matched/0.json) | [Selection](ed/linguistic/unigram_matched/selection.json) |
| Handwriting | Bigram (most connected IAM letter pairs) | [bigram/0.json](ed/handwriting/bigram/0.json) | Connectivity rate stored as `utility` inside the file |
| Handwriting | BPE (merges ranked by connected occurrences) | [bpe/0.json](ed/handwriting/bpe/0.json) | Hugging Face BPE file |
| Handwriting | Unigram (pieces seeded by connected counts) | [unigram/0.json](ed/handwriting/unigram/0.json) | Hugging Face Unigram file |

Records alongside `0.json` describe the shared tokenizer, so they are not
replicated for the other four copies.

## Construction and reuse

`train_ed_tokenizers.ipynb` declares categories, imports tokenizer classes and
calls `.train()` using IAM transcripts or the bundled
[IAM connectivity evidence](../evidence/README.md). Its final cell publishes the
same fitted vocabulary as `0.json` through `4.json` using `save_fold_tokenizers`.
Every learned file is written by `.train(dir_ds, categories, 224, ...)` on IAM
in TVA's own format, like the original TVA notebook; `.load()` reuses a saved
file. BPE/Unigram files are Hugging Face tokenizers with letters-only splitting.
The character file is produced by the unchanged original TVA class, with 79 IDs.
All learned ED methods have 145 extra pieces and 224 CTC classes, except the
two compression-matched controls. They keep a prefix of a text-only vocabulary
(62 bigrams, 141 classes; 59 BPE merges, 138 classes) and are built by
`tva/compression_matched_tokenizers.py`.

The linguistic tokenizers are TVA's own classes; the handwriting tokenizers
subclass them and only replace `.train()`. `UnigramEMTokenizer` also subclasses
TVA's `UnigramTokenizer`: it runs the handwriting Unigram's EM without
connectivity. The BPE/Unigram files replaced the
earlier custom (DTLR-format) ones, which remain in git history.

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
