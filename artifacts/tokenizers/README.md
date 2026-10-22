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
      bpe/              0.json, 1.json, 2.json, 3.json, 4.json
      unigram/          0.json, 1.json, 2.json, 3.json, 4.json
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
| Linguistic | Matched BPE | [bpe/0.json](ed/linguistic/bpe/0.json) | [Training](ed/linguistic/bpe/0.training.json) |
| Linguistic | Matched Unigram | [unigram/0.json](ed/linguistic/unigram/0.json) | [Training](ed/linguistic/unigram/0.training.json) |
| Handwriting | Bigram | [bigram/0.json](ed/handwriting/bigram/0.json) | [Training](ed/handwriting/bigram/0.training.json) |
| Handwriting | BPE | [bpe/0.json](ed/handwriting/bpe/0.json) | [Training](ed/handwriting/bpe/0.training.json) |
| Handwriting | Unigram | [unigram/0.json](ed/handwriting/unigram/0.json) | [Training](ed/handwriting/unigram/0.training.json) |

Reports alongside `0.json` describe the shared tokenizer, so they are not
replicated for the other four copies. Native IAM sources live in each learned
method's `source/` directory. Their original versioned release names and JSON
metadata are retained for authentication and provenance.

The handwriting BPE/Unigram [release manifest](ed/handwriting/source/iam-english-handwriting-subwords-v1.manifest.json)
is shared. Family-level compatibility audits are
[handwriting](ed/handwriting/subwords.audit.json) and
[linguistic](ed/linguistic/subwords.audit.json).

## Construction and reuse

`train_ed_tokenizers.ipynb` declares categories, imports tokenizer classes and
calls `.train()` using IAM transcripts or the bundled
[IAM connectivity evidence](../evidence/README.md). Its final cell publishes the
same fitted vocabulary as `0.json` through `4.json` using `save_fold_tokenizers`.
Calling `.prepare()` adapts a frozen native model; `.load()` reuses a saved file.
The character file is produced by the unchanged original TVA class, with 79 IDs.
All learned ED methods have 145 extra pieces and 224 CTC classes.

The linguistic BPE/Unigram controls use the shared custom trainers and segmentation
with handwriting evidence disabled. Original TVA's Hugging Face implementations
remain available separately. Training policies and authenticated source models
retain their existing version identities inside the JSON records.

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
