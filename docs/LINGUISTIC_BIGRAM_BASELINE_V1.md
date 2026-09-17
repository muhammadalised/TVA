# Matched Linguistic Bigram Baseline v1

**Policy frozen:** 2026-09-18  
**Model version:** `onhw-frequency-bigram-v1`  
**Output size:** 419 classes per fold

This document freezes the frequency-based comparator for the primary
handwriting-bigram experiment. It is trained independently for every WD and WI
fold using that fold's `train.json` annotations only. Validation labels and
recognition results do not influence construction.

## Construction policy

For each training fold:

1. Normalize labels to NFC and verify that every character belongs to the
   predeclared 59-character OnHW alphabet.
2. Collapse duplicate label samples to distinct word types. This gives every
   training word equal linguistic weight and prevents the number of writers or
   repeated recordings from being treated as language-frequency evidence.
3. Count every adjacent pair occurrence across those distinct word types.
4. Rank pairs by descending adjacency count, then by ascending Unicode token
   string for deterministic ties.
5. Retain the first 359 pairs. Assign each selected pair utility equal to its
   count divided by the largest selected count in that fold. Division by one
   common positive value does not change the dynamic program's choices.
6. Assign IDs as blank ID 0, the fixed 59 characters at IDs 1–59, and the 359
   ranked bigrams at IDs 60–418.

The tokenizer uses the same NFC normalization and maximum-total-utility
non-overlapping dynamic program as the handwriting-aware condition, including
the same bigram-count and pair-choice tie-breaking. The comparison therefore
controls the segmentation algorithm. It intentionally changes the evidence
used for both pair selection and overlap utility: training-text frequency for
the linguistic condition versus frozen IAM+READ connectivity for the
handwriting condition. It must not be described as a token-membership-only
comparison.

TVA loads these artifacts with tokenizer key `linguistic_bigram`.

## Leakage controls

- The builder accepts only a file named `train.json`.
- WD and WI are built separately, and every artifact records its dataset and
  fold.
- Each artifact stores the complete source `train.json` SHA-256, training
  sample count, and distinct-word count.
- The schema records `validation_annotations_read: false`.
- Validation labels are used only after construction for coverage and
  round-trip checks; they never rank or select tokens.
- Recognition metrics cannot modify the frozen vocabulary or utility rule.

## Generated artifacts

```text
artifacts/tokenizers/linguistic_bigram/onhw_words500_wd_word_rh/{0..4}.json
artifacts/tokenizers/linguistic_bigram/onhw_words500_wi_word_rh/{0..4}.json
```

Each directory contains a manifest with full per-fold artifact checksums.
Runtime loading authenticates the pinned manifest checksum and then the chosen
fold artifact checksum before accepting the tokenizer.

| Dataset | Source `train.json` SHA-256 | Manifest SHA-256 |
| --- | --- | --- |
| WD/RH | `badc0ce8972780de5f1cd8d7aacdf1ea360e89efac480138ed417ebccc4e81cd` | `c6895783d3da442e80ddf169be2e6397839ff2b0158df19b2c6ecb1bc40a2faa` |
| WI/RH | `3d8355292880518095f9c1a44d207339c4f18dcfb8ba0b0c189e1d7aaf9949cf` | `605b0f627d3044340111d7ee0b6ba364790b51130d431640d8eca3e1cc9bb392` |

Rebuild commands:

```bash
python build_linguistic_bigram_tokenizers.py \
  --train data/tva/onhw_words500_wd_word_rh/train.json \
  --dataset onhw_words500_wd_word_rh \
  --output-directory artifacts/tokenizers/linguistic_bigram/onhw_words500_wd_word_rh \
  --overwrite

python build_linguistic_bigram_tokenizers.py \
  --train data/tva/onhw_words500_wi_word_rh/train.json \
  --dataset onhw_words500_wi_word_rh \
  --output-directory artifacts/tokenizers/linguistic_bigram/onhw_words500_wi_word_rh \
  --overwrite
```

## Pre-training audit

| Split/fold | Training word types | Candidate pairs | Count above cutoff | Selected / total tied at cutoff | Overlap with 359 handwriting pairs | Different segmentations / 25,199 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| WD 0 | 479 | 413 | 266 | 93 / 147 | 224 | 13,174 |
| WD 1 | 501 | 426 | 272 | 87 / 154 | 224 | 13,276 |
| WD 2 | 465 | 410 | 262 | 97 / 148 | 225 | 13,428 |
| WD 3 | 484 | 425 | 270 | 89 / 155 | 223 | 13,326 |
| WD 4 | 499 | 425 | 272 | 87 / 153 | 224 | 13,277 |
| WI 0 | 501 | 426 | 272 | 87 / 154 | 224 | 13,276 |
| WI 1 | 501 | 426 | 272 | 87 / 154 | 224 | 13,276 |
| WI 2 | 500 | 425 | 272 | 87 / 153 | 224 | 13,277 |
| WI 3 | 501 | 426 | 272 | 87 / 154 | 224 | 13,276 |
| WI 4 | 501 | 426 | 272 | 87 / 154 | 224 | 13,276 |

The cutoff count is one in every fold. Consequently, 87–97 selected tokens
come from a larger tied group of 147–155 once-occurring pairs. Lexical
tie-breaking makes this tail reproducible but not strongly supported by
frequency. This is a direct consequence of matching the handwriting
condition's large 359-bigram inventory and must be reported as a baseline
limitation. WD produces five distinct vocabularies; WI produces two because
folds 0, 1, 3, and 4 contain the same 501 word types.

The two conditions share 223–225 of their 359 bigrams. Their segmentations
differ on 13,174–13,428 of 25,199 samples per fold (52.28%–53.29%). Mean target
length is approximately 3.321 tokens for handwriting and 3.221 for linguistic
tokenization. These are pre-training composition measurements, not recognition
results.

All ten artifacts successfully encoded their corresponding training and
validation labels. Across 251,990 fold/split label instances, targets were
non-empty, used IDs 1–418 only, and decoded exactly to their NFC labels.

## Contingent greedy ablation

If the primary DP-based training is unstable or performs poorly, a secondary
experiment may apply the same deterministic greedy left-to-right rule to both
the frozen handwriting and matched linguistic vocabularies. It must be named
and reported as a separate segmentation ablation. It must not replace the
predeclared DP result or be selected as the primary method after comparing
validation performance. All other training settings should remain matched.
