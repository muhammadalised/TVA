# Tokenizer training evidence

`iam/iam_connectivity_v1.json.gz` is the compact DTLR export consumed by TVA's
handwriting tokenizer `.train()` methods. It includes only the pinned IAM
training split: 5,694 transcriptions, 160,536 non-whitespace pair observations,
and 1,259 aggregate connectivity score rows. No vocabulary has been selected
in this file, and no ED labels, images, or sensor recordings enter its export.

For each line, the export preserves original text offsets and within-image
component IDs for exact usable observations. Null component IDs are explicit
abstentions. This permits TVA to count longer spans sharing one component and
to recompute supported BPE merges as token segmentation changes.

The file is 859,995 bytes. Compressed SHA-256:
`815304d2591bc187a2eea7dd76593499905c11645c8b9a146e7f1699c0e636fa`.
Decompressed SHA-256:
`1b44fddd0fa76dbb4ed14517f4361c522a4415139d16307b3468e14d122b8d0c`.
Both hashes and the original five input hashes are checked by
`tva/handwriting/evidence.py`. It also validates line identities/transcription
hashes, coverage, offsets, component semantics, and aggregate reconstruction.

DTLR regenerates this evidence without importing TVA or fitting tokenizers:

```bash
python poc/scripts/export_iam_tokenizer_evidence.py \
  --iam-train-dir /home/artellisys/dtlr-output/iam-train-full \
  --output /home/artellisys/TVA/artifacts/evidence/iam/iam_connectivity_v1.json.gz
```

TVA training does not need a DTLR checkout or raw IAM image files. The original
TVA linguistic bigram wrapper still reads its authenticated IAM labels and
selection inputs, as declared in the ED notebook.
