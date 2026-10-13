# Tokenizer artifacts

The tokenizer files are already trained and prepared. Recognition training does not require
rerunning `train_ed_tokenizers.ipynb`. The current ED vocabulary files are shared
across WD/WI and all five folds. Calling `.train()` now selects/fits handwriting
vocabularies in TVA from [the portable IAM evidence](../evidence/README.md).
Calling `.prepare()` only adapts an already frozen source; `.load()` reuses a
saved tokenizer directly.

```text
artifacts/tokenizers/
  ed/
    handwriting/          Current ED handwriting bigram, BPE, and Unigram
      source/             Exact frozen IAM releases from DTLR and release manifest
    linguistic/           Original TVA bigram trained on IAM, with audit/provenance
  archive/
    ed/linguistic/         Historical custom linguistic bigram
    onhw/handwriting/     Historical IAM+READ handwriting bigram
    onhw/linguistic_bigram/ Historical OnHW linguistic fold files and manifests
```

## Current ED tokenizers

| Method | Vocabulary file | Supporting files |
| --- | --- | --- |
| Handwriting bigram | [Bigram](ed/handwriting/ed_iam_handwriting_bigram_greedy_v1.json) | [Frozen IAM source](ed/handwriting/source/iam-english-handwriting-bigram-v1.json) |
| Handwriting BPE | [BPE](ed/handwriting/ed_iam_handwriting_bpe_v1.json) | [Frozen IAM source](ed/handwriting/source/iam-english-handwriting-bpe-v1.json) |
| Handwriting Unigram | [Unigram](ed/handwriting/ed_iam_handwriting_unigram_v1.json) | [Frozen IAM source](ed/handwriting/source/iam-english-handwriting-unigram-v1.json) |
| Original TVA linguistic bigram | [Bigram](ed/linguistic/ed_iam_tva_original_bigram_greedy_v1.json) | [Provenance](ed/linguistic/ed_iam_tva_original_bigram_greedy_v1.provenance.json), [audit](ed/linguistic/ed_iam_tva_original_bigram_greedy_v1.audit.json) |

The handwriting BPE and Unigram share a [release manifest](ed/handwriting/source/iam-english-handwriting-subwords-v1.manifest.json)
and an [ED compatibility audit](ed/handwriting/ed_iam_handwriting_subwords_v1.audit.json).
Provenance records where a file came from and how it was produced. An audit
records the checks performed on it.

The character baseline uses the alphabet in its configuration and needs no
learned tokenizer artifact. Matched text-only BPE/Unigram training functions moved with the shared
implementation, but their ED artifacts and recognition configs are not yet
integrated here.

## Historical artifacts

`archive/` retains vocabulary files needed to reproduce or analyze completed
experiments. The custom ED linguistic bigram and the OnHW utility-based
conditions are historical and are not the current ED training defaults.

## Compatibility and reproducibility

This directory reorganization preserves every frozen JSON file byte for byte,
including vocabulary IDs, learned parameters, hashes, and provenance. Frozen
reports and older experiment notes can therefore mention the previous locations.
The tokenizer path resolver redirects those former repository paths when loading
saved experiment configurations. Current configs and preparation notebooks use
the directories above.

Recognition results remain under `results/thesis/ed/` and the OnHW result paths
specified by their configs. Completed run configs are preserved as recorded.

## TVA training reports

Each handwriting `.train()` call writes a `.training.json` report alongside its
ED vocabulary. This records TVA's implementation/evidence hashes and confirms
that candidates and learned parameters were recomputed from IAM evidence and
that both the historical native release and the ED projection reproduce exactly.
The frozen source's original DTLR provenance stays unchanged; the new report
identifies the TVA fitting implementation separately.

The v1 training interface is deliberately fixed to the existing experimental
policy. A different corpus, thresholds, budget, or learned result requires a
new version rather than silently replacing artifacts used in completed runs.

The current fitting reports are [bigram](ed/handwriting/ed_iam_handwriting_bigram_greedy_v1.training.json),
[BPE](ed/handwriting/ed_iam_handwriting_bpe_v1.training.json), and
[Unigram](ed/handwriting/ed_iam_handwriting_unigram_v1.training.json).
