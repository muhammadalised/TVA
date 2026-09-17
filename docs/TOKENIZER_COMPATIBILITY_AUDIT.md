# Frozen IAM+READ Tokenizer Compatibility Audit

**Audit date:** 2026-09-15  
**Target:** right-handed OnHW-Words500 WD and WI datasets  
**Source artifact:** `combined-iam-read-v1/model.json` from the DTLR output
bundle

This note preserves the evidence gathered before integrating the combined
IAM+READ handwriting-bigram tokenizer into TVA. It is an audit result, not a
recognition result: no OnHW model was trained and no tokenizer was selected or
tuned using recognition validation scores.

## 1. Frozen artifact provenance

| Property | Audited value |
| --- | --- |
| Schema | `dtlr.handwriting-bigram-tokenizer.v2` |
| Model version | `iam-read-combined-v1` |
| SHA-256 | `5c5d9f1689a4802fc5e9451e5afe8abdfb090562587ab78ddd343211feb94dd4` |
| Evidence datasets | IAM train and READ train |
| Connectivity method | `dominant-core-v3` |
| Unicode normalization | NFC |
| Minimum evidence count | 20 exact alignments |
| Connectivity-rate threshold | 0.5 |
| Pair policy | letters only |
| Overlap resolution | maximum-total-utility non-overlapping dynamic program |
| Vocabulary | 494 classes: one blank, 91 non-empty single characters, 402 bigrams |

The IDs are contiguous from 0 through 493. The empty string is the CTC blank
at ID 0, matching TVA's loss and decoder convention. The source artifact must
remain immutable and be identified by the checksum above in every derived TVA
configuration or adapter.

## 2. TVA path inspected

The audit traced labels from `train.json` and `val.json` through
`HRDataset`, tokenizer encoding, CTC loss, decoding, and model construction.
Important implementation facts are:

- `HRDataset` passes each annotation's raw `label` directly to
  `tokenizer.encode`; it performs no normalization or vocabulary validation.
- With dataset caching enabled, incompatible labels fail while the dataset is
  being constructed. TVA constructs the validation dataset before the training
  dataset.
- `tokenizer.size` is passed directly to the decoder output head. The blank is
  already part of that size; another class must not be added.
- CTC loss uses blank ID 0. Batch-label padding also uses zero, but the supplied
  target lengths exclude padding, so this is compatible.
- TVA's existing `BigramTokenizer` can read the artifact's `vocab` and
  `idx_token` keys, but it would apply greedy left-to-right matching. That is
  not the segmentation algorithm defined by the frozen model.
- The complexity calculation in `evaluate.py` currently uses a hard-coded
  500-class head rather than `tokenizer.size`; it must be corrected before
  reporting parameter or MAC comparisons for this experiment.

## 3. Complete OnHW label audit

Every training label in all five WD folds and all five WI folds was submitted
to the unmodified combined tokenizer. Validation labels were also checked as a
compatibility diagnostic. For every encodable label, the following invariant
was verified:

```text
decode(encode(label)) == NFC(label)
```

All successful encodings passed this round trip, and none of the OnHW labels
changed under NFC. Complete coverage failed only because the artifact lacks the
uppercase OnHW characters `Ä` and `Ü`.

| Fold | WD training failures | WI training failures |
| ---: | ---: | ---: |
| 0 | 353 / 20,163 | 480 / 19,907 |
| 1 | 507 / 20,160 | 483 / 20,078 |
| 2 | 606 / 20,160 | 488 / 20,202 |
| 3 | 457 / 20,158 | 498 / 20,520 |
| 4 | 509 / 20,161 | 483 / 20,089 |

Each fold's train-plus-validation partition contains the same 25,199 physical
samples. Across that complete set, 608 samples, or 2.41%, contain `Ä` or `Ü`
and cannot be encoded by the unmodified artifact. Every WD and WI fold is
affected, so the raw artifact cannot be used for a complete training run.

## 4. Segmentation compatibility

The DTLR tokenizer selects a non-overlapping set of bigrams that maximizes
total handwriting utility. TVA's current Bigram tokenizer instead takes the
first available pair while scanning from left to right.

Among the 24,591 labels that do not contain the two missing characters, the
algorithms produced different token sequences for 9,009 labels, or 36.64%.
Examples include:

| Label | TVA greedy segmentation | Frozen-model DP segmentation |
| --- | --- | --- |
| `Juni` | `[J, un, i]` | `[J, u, ni]` |
| `wir` | `[wi, r]` | `[w, ir]` |
| `Teil` | `[Te, i, l]` | `[T, ei, l]` |
| `Dabei` | `[Da, be, i]` | `[D, ab, ei]` |

Therefore, loading the JSON into TVA's existing `BigramTokenizer` would be a
silent methodological error even after character coverage is repaired. A
dedicated tokenizer must preserve the DTLR dynamic program and its deterministic
tie-breaking behavior.

## 5. Vocabulary utilization and output-head implications

Of the frozen model's 402 eligible bigrams, 220 occur in the encodable OnHW
labels and 182 are unused by this dataset. The artifact also contains
punctuation, digits, historical characters, combining marks, and other single
characters outside the declared 59-character OnHW alphabet.

Three integration sizes are consequently relevant:

| Integration | Output classes | Interpretation |
| --- | ---: | --- |
| Unmodified artifact | 494 | Preserves artifact exactly but fails label coverage |
| Full artifact plus `Ä`, `Ü` fallbacks | 496 | Complete coverage, but retains all out-of-task classes |
| OnHW-alphabet projection | 419 | Blank + 59 fallback characters + 359 frozen in-alphabet bigrams |

The 419-class projection removes 43 bigrams containing characters outside the
declared OnHW alphabet. This projection can be constructed without label or
frequency inspection: it uses only the fixed task alphabet already present in
the dataset metadata/configuration. On 2026-09-17 it was frozen as the primary
adapter under policy ID `onhw-words500-rh-iam-read-v1`. The 496-class form is
excluded from the primary experiment and may be used only as a separately
predeclared sensitivity condition. The exact construction and ID assignment
are recorded in `docs/HANDWRITING_BIGRAM_ADAPTER_V1.md`. The implemented
canonical artifact has SHA-256
`12ce25d8bedc552e6b3497ffb1d07e506b01b34296cc21b970f82a550cbf2bfe`.

Existing character or differently sized tokenizer checkpoints cannot be loaded
as complete TVA models because their output-head shapes differ. Main comparison
models must be trained from scratch.

## 6. Leakage-safe integration requirements

The integration must satisfy all of the following before recognition training:

1. Validate the frozen schema, model version, checksum, contiguous IDs, inverse
   mappings, and blank convention at load time.
2. Preserve NFC normalization, utilities, eligible-pair decisions, dynamic
   programming, and deterministic tie-breaking from the frozen artifact.
3. If an OnHW adapter is used, derive it only from the predeclared 59-character
   task alphabet. Do not inspect OnHW train or validation label frequencies to
   select, rank, or remove handwriting bigrams.
4. Add `Ä` and `Ü` only as character fallbacks; do not infer new handwriting
   bigrams for them from OnHW labels.
5. Freeze one handwriting adapter before experiments and use it unchanged in
   every WD and WI fold.
6. Build the matched linguistic Bigram separately for each fold using only that
   fold's OnHW training labels. Validation labels and recognition scores must
   not influence its vocabulary.
7. Match output vocabulary size between the handwriting-aware and linguistic
   Bigram conditions and keep the character model as the common reference.
8. Record both source and derived-adapter checksums in configurations and
   experiment results.

The complete-label inspection reported here is a compatibility audit. Its
measurements must not be used to adapt token membership; only the fixed OnHW
alphabet may inform the compatibility projection.

## 7. Required pre-training tests

- Every WD/WI training and validation label encodes successfully.
- Every label satisfies the NFC encode/decode round trip.
- No encoded target contains blank ID 0.
- IDs are unique, contiguous, and within the model-head range.
- The TVA implementation matches the DTLR reference segmentation on a fixed
  conformance corpus, including overlapping eligible pairs and ties.
- Adapter construction produces identical bytes/checksums without reading any
  OnHW annotation list.
- All folds use the same frozen handwriting adapter.
- The matched linguistic tokenizer reads only its fold's training annotations.

## 8. Audit conclusion

The artifact's structure, provenance metadata, CTC blank convention, and
round-trip behavior passed. Complete OnHW coverage and direct compatibility
with TVA's current greedy tokenizer failed. The frozen artifact is therefore
valid input to a dedicated integration, but it is not directly trainable in
TVA. The 419-class adapter policy was subsequently frozen and passed an
in-memory all-fold reference validation on 2026-09-17. The next milestone is
its implementation as a canonical artifact, a handwriting-aware tokenizer,
and the conformance/coverage test suite.
