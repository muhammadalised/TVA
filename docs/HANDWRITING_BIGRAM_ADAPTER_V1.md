# OnHW Handwriting-Bigram Adapter Policy v1

**Policy frozen:** 2026-09-17  
**Policy ID:** `onhw-words500-rh-iam-read-v1`  
**Primary output size:** 419 classes  
**Canonical artifact SHA-256:**
`12ce25d8bedc552e6b3497ffb1d07e506b01b34296cc21b970f82a550cbf2bfe`

This document freezes the compatibility transformation between the external
IAM+READ handwriting-bigram model and the right-handed OnHW-Words500
experiments. The policy was fixed before recognition training and does not use
OnHW token frequencies, validation-label membership, or recognition results.

## Source model

The only accepted source is the DTLR artifact with all of these properties:

| Property | Required value |
| --- | --- |
| Schema | `dtlr.handwriting-bigram-tokenizer.v2` |
| Model version | `iam-read-combined-v1` |
| SHA-256 | `5c5d9f1689a4802fc5e9451e5afe8abdfb090562587ab78ddd343211feb94dd4` |
| Normalization | NFC |
| Blank token and ID | empty string, 0 |
| Segmentation | maximum-total-utility non-overlapping dynamic program |

The implementation must reject a source that does not match this provenance.
The source artifact remains immutable.

## Fixed task alphabet

The adapter uses the pre-existing 59-character OnHW alphabet, in this exact
order:

```text
A B C D E F G H I J K L M N O P Q R S T U V W X Y Z Ä Ö Ü
a b c d e f g h i j k l m n o p q r s t u v w x y z ä ö ü ß
```

This alphabet is part of the task configuration, not inferred from annotation
contents. It is embedded in the adapter definition so adapter construction
does not need to read an OnHW train or validation annotation file.

## Deterministic construction

Construct the adapter as follows:

1. Verify the source provenance, schema invariants, inverse mappings, and
   contiguous source IDs.
2. Start the derived vocabulary with the empty CTC blank at ID 0.
3. Assign the 59 fixed characters above to IDs 1 through 59 in exactly the
   displayed order. This retains every task character as a fallback; `Ä` and
   `Ü` are the only fallback singles absent from the source.
4. Traverse source bigrams in source-vocabulary order. Keep a bigram if and
   only if both of its characters are members of the fixed task alphabet.
5. Append the 359 retained bigrams without re-ranking them, assigning IDs 60
   through 418. Preserve their frozen source utilities and other segmentation
   metadata.

The result is exactly 419 classes: one blank, 59 single characters, and 359
source-derived bigrams. Relative source order and utility values are not
changed. No bigram containing `Ä` or `Ü` is invented from OnHW data.

The canonical artifact is
`artifacts/tokenizers/onhw_words500_rh_iam_read_v1.json`. It is serialized as
sorted-key, indented UTF-8 JSON with a final newline. Its SHA-256 is recorded
above beside the source checksum. Rebuilding it from the pinned source produces
identical bytes without access to OnHW annotations.

The reproducible build command is:

```bash
python build_handwriting_bigram_adapter.py \
  --source /path/to/combined-iam-read-v1/model.json \
  --output artifacts/tokenizers/onhw_words500_rh_iam_read_v1.json \
  --overwrite
```

## Frozen experimental use

- This 419-class adapter is the primary handwriting-aware Bigram condition.
- The same artifact must be used unchanged for all five WD and all five WI
  folds.
- The previously considered 496-class full-source-plus-fallback construction
  is not part of the primary experiment. It may be run only as a separately
  named, predeclared sensitivity analysis; it must not replace the primary
  condition in response to recognition results.
- The model output head must contain exactly `tokenizer.size == 419` logits,
  including blank. CTC blank remains ID 0.
- Existing checkpoints with differently sized output heads are not compatible
  complete-model initializations.

## Validation completed before implementation

An in-memory reference construction on 2026-09-17 produced the required 419
classes. The committed builder subsequently reproduced the same counts. It
added the two missing fallback singles, removed 34 out-of-task source singles
and 43 out-of-task source bigrams, and retained 359 bigrams.

The derived vocabulary was exercised with the DTLR reference tokenizer on
every train and validation label in every WD and WI fold. All 20 fold/split
checks passed for the canonical artifact, covering 251,990 fold/split label
instances: every label encoded, every target was non-empty, no target contained
blank ID 0, every ID was in the interval 1 through 418, and
`decode(encode(label)) == NFC(label)`.

This complete-label pass establishes compatibility only. Token membership was
already determined by the fixed alphabet and source model, so the pass did not
tune or select the adapter from OnHW label statistics.

## Comparator caveat to resolve before training

The handwriting condition must preserve the DTLR utility-maximizing dynamic
program. TVA's existing linguistic Bigram uses greedy left-to-right matching.
Consequently, equal vocabulary sizes alone do not make the conditions differ
only in bigram membership. Before the matched linguistic experiment is run,
its segmentation policy must be explicitly predeclared and the thesis must
either control this algorithmic factor or report it as a limitation. It must
not be silently treated as a vocabulary-only comparison.

Detailed compatibility evidence is in
`docs/TOKENIZER_COMPATIBILITY_AUDIT.md`; the governing decision is D015 in
`docs/DECISIONS.md`.

## TVA runtime integration

TVA loads this condition with tokenizer key `handwriting_bigram`. The runtime
implementation is `tva/handwriting_bigram_tokenizer.py`; unlike TVA's legacy
`bigram` condition, it uses NFC normalization and the frozen
maximum-total-utility dynamic program. File loading authenticates the complete
canonical artifact checksum before accepting the model, validates its source
provenance and leakage declaration, and exposes `size == 419` including blank
ID 0.

On 2026-09-17, the TVA implementation matched the independent DTLR reference
tokenizer exactly—both complete segmentation dictionaries and encoded IDs—for
all 501 unique OnHW words. It also passed coverage, ID-range, no-blank-target,
and NFC round-trip checks for all 251,990 WD/WI fold/split label instances.
