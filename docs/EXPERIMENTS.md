# Thesis Experiment Log

This file records scientific runs and important development experiments. All
reported values must be traceable to the saved configuration, log, metrics,
predictions, code version, and experiment backup.

## FAU English dataset compatibility audit — pre-training

### Scope and dataset identity

- Audit date: 2026-09-21
- Source supplied by supervisor:
  `/mnt/c/Users/Ali/Downloads/fau-english-dataset/`
- `gold_wd.zip` SHA-256:
  `edaf78b0f422f719bbb13153249a5a6667a814d6f5a8d7d0d2ba02535ae55a3e`
- `gold_wi.zip` SHA-256:
  `0ba1b8e5d53c7208bbdc4a43cfb701ed6a1d27be382a23ba0aaade8cb3283304`
- No training code or dataset file was changed during this audit.

Both archives describe the same 2,550 sentence recordings from 102 writers,
with 501 unique labels. Each sample is a seven-channel IMU sequence resampled
to 100 Hz; labels range from 10 to 63 characters and average 41.60 characters.
The task alphabet contains 78 characters: space, 52 ASCII letters, ten digits,
and 15 punctuation/symbol characters. This is therefore a sentence-level,
seven-channel task, not a drop-in replacement for the 13-channel,
word-level OnHW-Words500 experiment.

The supplied five-fold splits are structurally complete: every fold has zero
sample-ID overlap between train and validation and their union contains all
2,550 samples. In WD, all or nearly all writers occur in both sides while
validation labels are disjoint from training labels. Validation fold sizes are
503, 511, 514, 531, and 491. In WI, train and validation writer sets are
disjoint; validation fold sizes are 525, 525, 500, 500, and 500. Repeated
prompts cause substantial train/validation label overlap in WI, which is a
property of the supplied split and must be reported when interpreting transfer
results.

### Supervisor-directed tokenizer scope

The supervisor requested that the initial FAU evaluation use English
handwriting tokenizers, not the combined IAM+READ tokenizer. Consequently, the
frozen 419-class OnHW IAM+READ adapter is out of scope for this first FAU
experiment. It also cannot encode any raw FAU label because it lacks spaces,
digits, and punctuation.

DTLR commit `d2f4631c5adbc0a721e01f3bb2e179b877934976` froze the
generic IAM-only source artifact at
`/home/artellisys/DTLR/poc/frozen/iam-english-handwriting-bigram-v1.json`.
Its SHA-256 is
`2fbb81479211454d8ad028d8af76eb26e76b75b80408a1a1e1f4efa15ab26a92`.
It contains CTC blank ID 0, 76 singleton characters, and 145 IAM-derived
handwriting bigrams, for 222 classes. The source artifact was reconstructed
byte-identically from the prior demonstration model and passed all 43 DTLR
tests. Its complete vocabulary audit also passed without emitting blank.

The frozen source records IAM train as its sole evidence source: 5,694 lines,
160,536 pair observations, and 1,259 scored pairs using `dominant-core-v3`.
The score evidence SHA-256 is
`45c86a9f703ebc3684563ec2a840e7c34241aa76e84d692be46cf11bfec27055`,
the evidence-manifest SHA-256 is
`82cb95a14e71ab5b70c868ca1676b3f628552fabac395fbd30039fc7798bf190`,
and the reconstructed source-model SHA-256 is
`36db922860cbd501a6c5a1297dbf7d998abdbe1a337513acb00b44e74fbd9334`.
It explicitly assigns downstream task-alphabet and segmentation decisions to
the consumer repository.

The selection thresholds remain a scientific limitation: at least 20 exact
alignment observations and connected rate at least 0.5 were provisional
demonstration settings. They are frozen before downstream evaluation for
reproducibility and leakage control, but are not claimed to be statistically
optimal or held-out validated.

The IAM-only model covers nearly all of the FAU alphabet but lacks `%`, `(`,
and `=`. Those characters occur 416, 414, and 429 times respectively and make
906 of the 2,550 raw labels unencodable. IAM's `*` singleton does not occur in
FAU. Silently deleting unsupported symbols or evaluating only the encodable
subset would change the task and bias the result.

The leakage-safe TVA adapter was frozen on 2026-09-21. It keeps all 145
IAM-derived bigrams and their score rows byte-for-byte at the JSON-value level,
then projects only the singleton layer to the declared 78-character FAU task
alphabet. It has 224 output classes: blank ID 0, 78 singletons, and 145
bigrams. The pinned source is
`artifacts/tokenizers/source/iam-english-handwriting-bigram-v1.json`; the
canonical adapter is
`artifacts/tokenizers/fau_english_iam_handwriting_bigram_greedy_v1.json`, with
SHA-256
`4c06828ac3b0ae03e98d569b0f3fea1cdfbc0a125f6eeffcc7ffb2a4935f3f52`.
Construction excludes IAM-only `*` and adds FAU-only `%`, `(`, and `=` as
single-character fallbacks. No FAU label value, label frequency, or recognition
result selects, ranks, or removes a handwriting bigram.

The repository boundary is explicit: DTLR constructs and freezes the generic
IAM-only English handwriting vocabulary using IAM training evidence only. It
must not read FAU annotations or create an FAU-specific vocabulary. TVA owns
the downstream FAU task adapter, whose only task-specific construction input
is the predeclared 78-character `categories` schema. Complete FAU labels may
be read only after the adapter is frozen, for encode/decode compatibility
auditing and recognizer training/evaluation—not for handwriting-token
selection, ranking, thresholds, or segmentation-policy tuning.

### Segmentation-policy audit

DTLR's IAM-only tokenizer does not use TVA's legacy greedy encoder. It uses a
maximum-total-utility dynamic program: for overlapping eligible pairs, it
chooses the non-overlapping set with the greatest sum of IAM handwriting
connectivity scores. TVA's legacy Bigram encoder instead scans left to right
and immediately consumes any available pair, without considering its score.

Applying both policies to all 2,550 FAU labels with the same 145 IAM bigrams
produces different token identities for 1,778 samples (69.73%) and 350 of the
501 unique labels (69.86%). Both policies produce 85,153 target tokens in
total, so this difference changes which bigrams are learned rather than the
aggregate target length. For example, in `short`, DP selects `or`, whereas
greedy consumes `ho` first.

Therefore, switching the IAM tokenizer to greedy defines a new derived
segmentation variant, not simply loading the original DTLR model unchanged.
Greedy left-to-right was frozen as the FAU primary policy on 2026-09-21 for
compatibility with established TVA Bigram behavior. It is exposed under the
separate tokenizer key `handwriting_bigram_greedy`; the existing
`handwriting_bigram` DP runtime and all completed OnHW experiments remain
unchanged. The matched FAU comparator must also use greedy segmentation. DP may
be evaluated later only as an explicitly named ablation applied to both
vocabularies.

After the adapter bytes were frozen, `audit_fau_handwriting_bigram.py`
authenticated both source archives and encoded every canonical annotation in
both packages. The audit checked 2,550 records per archive (5,100 encodings),
170,306 total emitted target tokens, zero blank IDs, zero reconstruction
failures, identical WD/WI record-label mappings, and exact declared-alphabet
agreement. This is a compatibility result, not recognition evidence. The
focused regression suite passed 21 applicable tests; two unrelated optional
external-reference tests were skipped.

### Matched IAM-text linguistic comparator

The matched linguistic comparator was frozen on 2026-09-21 before FAU model
training. Its vocabulary evidence comes only from the same complete 5,694-line
IAM training split used by the handwriting pipeline. The authenticated IAM
`labels.pkl` SHA-256 is
`5ac34ad37ba0b125308fe1a2bc97095985e25dfa76495628c3bb3895c0b446ab`;
the complete-train selection-manifest SHA-256 is
`7893dfba4febe6df99cf0bdb0c74fabe7b736d2a6af05033d8638b90455bc1c2`.
No IAM validation/test text and no FAU label were used to select pairs.

After NFC normalization, the builder counts every case-sensitive adjacent
ASCII-letter pair occurrence within each IAM training line. It found 861
candidate pairs and 153,350 eligible occurrences. Pairs are ranked by
descending occurrence count with lexical token tie-breaking, and the first 145
are retained to match the handwriting vocabulary. The last selected pair is
`bu` with count 273; the first excluded pair is `tu` with count 270, so the
selection boundary is not tied. The frozen evidence file is
`artifacts/tokenizers/source/iam-train-letter-bigram-counts-v1.json`, SHA-256
`3a2d90f7233dd550f399e3296f92390291201482ca690c9cd85eb30d8b8079ca`.

The canonical comparator is
`artifacts/tokenizers/fau_english_iam_linguistic_bigram_greedy_v1.json`,
SHA-256
`0a7e173afdd2a911780c14517e651b9787c3b8b3c0fdedb74f915920b4d4f392`.
It uses the identical blank ID, ordered 78-character singleton layer,
145-bigram count, 224-class output size, NFC normalization, and greedy
left-to-right segmentation as the handwriting condition. Pair utilities are
normalized IAM counts retained for audit metadata; greedy segmentation does
not consult them. The two vocabularies share 72 bigrams and differ in 73 pairs
per condition.

The post-freeze compatibility audit encoded all 2,550 canonical FAU labels in
both archives. The linguistic comparator emitted 79,078 target tokens per
archive, with zero blank IDs and zero round-trip failures. The handwriting
condition emits 85,153 per archive, so the linguistic vocabulary shortens the
FAU targets by 6,075 tokens (7.13%). This is an intrinsic consequence of the
different frozen pair memberships—not a segmentation-policy difference—but it
must be considered when interpreting recognition results.

The complete TVA regression suite passed after integration: 40 tests ran, 38
passed, and two unrelated optional external-reference checks were skipped.

One provenance anomaly should be retained in the record: `gold_wd/build.log`
mentions a `gold_wi/meta.json` save path even though the archive's split
metadata correctly says `writer_dependent`. The fold contents themselves pass
the structural WD checks above.

### FAU ZIP loader and matched pipeline smoke validation

- Date completed: 2026-09-21
- Scope: fold 0, eight training and four validation samples per condition
- Conditions: handwriting/linguistic Bigram × WD/WI
- Architecture: diagnostic BLConv-S + BiLSTM-S + CTC
- Input: archive-provided seven channels at 100 Hz
- Output: 224 classes including blank ID 0
- Device: CPU
- Epochs: one
- Augmentation: disabled
- Result: all four conditions completed training, validation, decoding, metric
  calculation, and resumable checkpoint saving.

`FauZipDataset` authenticates the WD/WI source archive SHA-256 before use and
streams semicolon-delimited float32 sequences directly from the ZIP. It checks
the declared distribution, five-fold structure, 100 Hz rate, seven-channel
shape, complete member paths, 78-character alphabet, finite signal values, and
fold index. The source archives remain untouched and need not be extracted.
The standard TVA per-sample, per-channel normalization is then applied. The
loader does not resample or select channels because the archive already
contains the declared seven 100 Hz channels.

All saved decoder heads have shape `(224, 128)`. The one-epoch CER and WER are
1.0 in every condition, which is expected for this tiny diagnostic run and is
not recognition evidence.

| Condition | Latest checkpoint SHA-256 |
| --- | --- |
| Handwriting WD | `b4a69b58f769de17c8b639b6320b62307d2decac59badc480b99ea23fa114397` |
| Linguistic WD | `7339cc0f8c151119de718c245004770997a65b50daee4a132475b5b2e80a9ba7` |
| Handwriting WI | `8f7eaca9df7341a31f478271d081b973a78dd6981931c5cac2afcc4f254bf7b1` |
| Linguistic WI | `b3a54dab2c5e1953849bb399364bdcd1de185824b8f2fb9679f54c874c2dc284` |

The committed smoke configurations use the portable directory
`data/tva/fau_english`. On the current machine they were executed by setting
`TVA_FAU_DATASET_DIR=/mnt/c/Users/Ali/Downloads/fau-english-dataset`, so no
machine-specific source path is stored in configuration files.

After loader integration, the complete TVA suite ran 46 tests: 44 passed and
two unrelated optional external-reference checks were skipped.

## Bigram pipeline smoke validation — WD/RH and WI/RH fold 0

### Setup and result

- Date completed: 2026-09-18
- Conditions: handwriting-aware Bigram and matched linguistic Bigram, each on
  WD/RH and WI/RH fold 0
- Scope: 16 training samples, eight validation samples, one CPU epoch, batch
  size two, seed 42
- Result: all four conditions completed training and validation and saved
  resumable checkpoints.
- Checkpoint verification: all four output heads contain 419 classes;
  `decoder.fc.weight` is `(419, 256)` and `decoder.fc.bias` is `(419,)`.
- Artifacts: `results/thesis/development/smoke_bigram_*/0/`

CER and WER were both 1.0 in every run, which is expected for a single epoch on
16 samples. These values are pipeline diagnostics and must not be reported as
recognition evidence. A subsequent bounded run detected the NVIDIA GeForce RTX
4060 Laptop GPU and successfully exercised CUDA mixed precision for the
handwriting-aware WD condition. The pipeline is therefore cleared for full
fold-0 training.

### Full-data timing measurement

A separate one-epoch timing run used the complete handwriting-aware WD/RH
fold-0 training and validation splits on the RTX 4060. Training took 29
seconds and validation took four seconds. The resulting estimate is about 2
hours 45 minutes for one 300-epoch fold, or about 11 hours for the four fold-0
Bigram conditions run sequentially. This run measures computational cost only;
its one-epoch recognition metrics and checkpoint are not experimental results
and must not be used to initialize a full run.

## Handwriting-aware Bigram — WI/RH fold 0

### Setup

- Date completed: 2026-09-18
- Dataset: OnHW-Words500, writer-independent, right-handed, fold 0
- Tokenizer: frozen IAM+READ handwriting-aware Bigram adapter
- Tokenizer classes: 419 including CTC blank ID 0
- Segmentation: frozen maximum-total-utility dynamic program
- Architecture: BLConv-B + BiLSTM-B + CTC
- Seed: 42
- Epochs: 300 (epochs 0–299)
- Batch size: 64
- Training machine: WSL laptop with NVIDIA GeForce RTX 4060
- Configuration: `configs/thesis/bigram_handwriting_wi_rh.yaml`
- Code commit at training start: `0298374`

### Best validation results

| Metric | Value | Epoch |
| --- | ---: | ---: |
| Levenshtein distance | 0.872260 | 272 |
| Character error rate | 0.159850 (15.99%) | 272 |
| Word error rate | 0.246221 (24.62%) | 274 |
| Average reference length | 5.456727 characters | Not optimized |

The metric optima belong to different checkpoints. At the best-CER checkpoint
(epoch 272), WER is 0.248110 (24.81%). At the best-WER checkpoint (epoch 274),
CER is 0.160266 (16.03%). Do not present the independently best CER and WER as
if they came from one model state.

### Fold-0 character comparison

Compared with the B0 WI/RH character run, independently best WER improves from
27.95% to 24.62%: a 3.33-percentage-point reduction, or an 11.9% relative
error reduction. Independently best CER changes from 15.60% to 15.99%, a
0.38-percentage-point increase. This mixed fold-0 result suggests that the
handwriting-aware labels improve exact-word recognition without improving
character error. It is a development observation, not a final thesis claim.
The matched linguistic Bigram run and all five folds are still required before
attributing the WER change to handwriting-derived pair evidence.

### Interruption, recovery, and artifacts

The first process completed training epoch 233 but stopped before its
validation/checkpoint stage because Tkinter objects from Matplotlib's GUI
backend were destroyed outside the main loop. The intact `latest.pth` from
epoch 232 contained the complete model, optimizer, scheduler, scaler, random,
DataLoader-generator, and metrics states. Training resumed at epoch 233 with
`MPLBACKEND=Agg` and completed epoch 299. Epoch 233 was therefore repeated;
the complete resumed metrics file contains exactly epochs 0–299.

Checkpoint SHA-256 values:

| Checkpoint | Epoch | SHA-256 |
| --- | ---: | --- |
| `best_cer.pth` | 272 | `e8037512f745b2105767ba519cf5118394f8e23fda163ba791a99fea688fdeef` |
| `best_wer.pth` | 274 | `3ee167a19f829a43436f63dfd843368207b160ad13e52f2d828100080438f3d8` |
| `latest.pth` | 299 | `a7beef28b9fd29095f52ec5a8729898280c560d51727c8456f25572abc77d10f` |

The complete run directory is
`results/thesis/bigram/handwriting_wi_rh/0/`. Preserve both logs and resolved
configuration snapshots so the interruption and resume remain auditable.
External backup location and checksum: not yet recorded.

## Matched linguistic Bigram — WI/RH fold 0

### Setup

- Date completed: 2026-09-18
- Dataset: OnHW-Words500, writer-independent, right-handed, fold 0
- Tokenizer: fold-0 training-text frequency Bigram
- Tokenizer classes: 419 including CTC blank ID 0
- Segmentation: matched maximum-total-utility dynamic program
- Architecture: BLConv-B + BiLSTM-B + CTC
- Seed: 42
- Epochs: 300 (epochs 0–299)
- Batch size: 64
- Training machine: WSL laptop with NVIDIA GeForce RTX 4060
- Configuration: `configs/thesis/bigram_linguistic_wi_rh.yaml`
- Code commit at training start: `04ba7f8`

### Best validation results

| Metric | Value | Epoch |
| --- | ---: | ---: |
| Levenshtein distance | 0.871504 | 262 |
| Character error rate | 0.159712 (15.97%) | 262 |
| Word error rate | 0.249055 (24.91%) | 262 |
| Average reference length | 5.456727 characters | Not optimized |

Unlike the handwriting run, both optimized metrics occur in the same epoch-262
checkpoint. The final epoch has CER 16.09% and WER 25.17%.

### Matched WI fold-0 comparison

| Condition | Best CER | Best WER |
| --- | ---: | ---: |
| Character B0 | **15.60%** | 27.95% |
| Matched linguistic Bigram | 15.97% | 24.91% |
| Handwriting-aware Bigram | 15.99% | **24.62%** |

Relative to the character baseline, linguistic Bigram reduces WER by 3.04
percentage points (10.9% relative) but increases CER by 0.37 points. Relative
to linguistic Bigram, handwriting Bigram reduces WER by only 0.28 points
(1.14% relative), while linguistic Bigram has a 0.014-point CER advantage.

At the respective best-WER checkpoints, the two Bigram models were compared on
the same 5,292 validation words. Handwriting was correct on 3,989 and
linguistic on 3,974, a difference of 15 words. They were both correct on 3,769;
handwriting alone was correct on 220, linguistic alone on 205, and both were
wrong on 1,098. An exploratory two-sided exact McNemar/binomial test on the
discordant pairs gives `p = 0.497`. Because checkpoints were selected and
tested on this same validation fold, this is descriptive, post-selection
evidence rather than a confirmatory significance test. Fold 0 does not support
a reliable difference between the two Bigram evidence sources; five-fold
evaluation is required.

### Artifacts

The run completed uninterrupted and its metrics file contains exactly epochs
0–299. All saved checkpoints have 419-output heads and complete resumable
state.

| Checkpoint | Epoch | SHA-256 |
| --- | ---: | --- |
| `best_cer.pth` | 262 | `159451a4df6727429a6bcb70cb2878a5327da98df69d388d4f685385b87b332e` |
| `best_wer.pth` | 262 | `159451a4df6727429a6bcb70cb2878a5327da98df69d388d4f685385b87b332e` |
| `latest.pth` | 299 | `cb88e59cf942185c625012ddeaf18cfacf24d052eac69b6f7fa5184e4a27b5fa` |

The complete run directory is
`results/thesis/bigram/linguistic_wi_rh/0/`. External backup location and
checksum: not yet recorded.

## Handwriting-aware Bigram — WD/RH fold 0

### Setup and results

- Date completed: 2026-09-18
- Dataset: OnHW-Words500, writer-dependent, right-handed, fold 0
- Tokenizer: frozen IAM+READ handwriting-aware Bigram adapter
- Tokenizer classes: 419 including CTC blank ID 0
- Segmentation: frozen maximum-total-utility dynamic program
- Architecture: BLConv-B + BiLSTM-B + CTC
- Seed: 42
- Epochs: 300 (epochs 0–299)
- Batch size: 64
- Training machine: WSL laptop with NVIDIA GeForce RTX 4060
- Configuration: `configs/thesis/bigram_handwriting_wd_rh.yaml`
- Code commit at training start: `04ba7f8`

| Metric | Value | Epoch |
| --- | ---: | ---: |
| Levenshtein distance | 1.146743 | 276 |
| Character error rate | 0.202440 (20.24%) | 276 |
| Word error rate | 0.398729 (39.87%) | 281 |
| Average reference length | 5.664615 characters | Not optimized |

The optima occur in different checkpoints. At epoch 276, WER is 39.95%. At
the best-WER epoch 281, CER is 20.41%. The final epoch has CER 20.45% and WER
40.29%.

### Fold-0 character comparison

The WD/RH character baseline has independently best CER 12.76% and WER 35.98%.
Handwriting Bigram therefore increases CER by 7.49 percentage points (58.7%
relative) and WER by 3.89 points (10.8% relative). At the respective best-WER
checkpoints, handwriting recognizes 3,028 of 5,036 validation words exactly,
versus 3,224 for character—a deficit of 196 words. Character alone is correct
on 469 discordant samples and handwriting alone on 273; an exploratory paired
exact test gives `p = 6.02e-13`.

The paired test is post-selection analysis on the validation fold, not a final
confirmatory test. Nevertheless, the effect is large and clearly negative on
WD fold 0. Together with the positive WI WER change, it is directionally
consistent with the proposal's expectation that handwriting-aware labels may
be more useful for writer-independent recognition. The matched linguistic WD
run and remaining folds are required before attributing this pattern to the
source of Bigram evidence.

### Artifacts

The run completed uninterrupted with exactly epochs 0–299. All checkpoints
have 419-output heads and complete resumable state.

| Checkpoint | Epoch | SHA-256 |
| --- | ---: | --- |
| `best_cer.pth` | 276 | `307c01cfaa7514d217806513800efa55267186338654c2b8714f286b692bbe6c` |
| `best_wer.pth` | 281 | `2d945fee160be6abed5924b8ac9e9c514c2d0190a9381bdcc6077f72105bb967` |
| `latest.pth` | 299 | `3de1f8d35dda39dd230ae30073b652985105db2e4d822d5079359e852323722e` |

The complete run directory is
`results/thesis/bigram/handwriting_wd_rh/0/`. External backup location and
checksum: not yet recorded.

## Matched linguistic Bigram — WD/RH fold 0

### Setup and results

- Date completed: 2026-09-19
- Dataset: OnHW-Words500, writer-dependent, right-handed, fold 0
- Tokenizer: fold-0 training-text frequency Bigram
- Tokenizer classes: 419 including CTC blank ID 0
- Segmentation: matched maximum-total-utility dynamic program
- Architecture: BLConv-B + BiLSTM-B + CTC
- Seed: 42
- Epochs: 300 (epochs 0–299)
- Batch size: 64
- Training machine: WSL laptop with NVIDIA GeForce RTX 4060
- Configuration: `configs/thesis/bigram_linguistic_wd_rh.yaml`
- Code commit at training start: `04ba7f8`

| Metric | Value | Epoch |
| --- | ---: | ---: |
| Levenshtein distance | 1.134035 | 282 |
| Character error rate | 0.200196 (20.02%) | 282 |
| Word error rate | 0.400715 (40.07%) | 278 |
| Average reference length | 5.664615 characters | Not optimized |

The metric optima belong to different checkpoints. At epoch 282, WER is
40.19%. At the best-WER epoch 278, CER is 20.05%. The final epoch has CER
20.18% and WER 40.31%.

### Matched WD fold-0 comparison

| Condition | Best CER | Best WER |
| --- | ---: | ---: |
| Character B0 | **12.76%** | **35.98%** |
| Matched linguistic Bigram | 20.02% | 40.07% |
| Handwriting-aware Bigram | 20.24% | 39.87% |

Relative to character, linguistic Bigram increases CER by 7.26 percentage
points and WER by 4.09 points. Relative to handwriting Bigram, linguistic has
a 0.22-point CER advantage, while handwriting has a 0.20-point WER advantage.

At their respective best-WER checkpoints, handwriting recognizes 3,028 of
5,036 validation words exactly and linguistic recognizes 3,018, a difference
of only ten words. Handwriting alone is correct on 312 discordant samples and
linguistic alone on 302; an exploratory paired exact test gives `p = 0.716`.
Thus fold 0 provides no reliable evidence of a difference between Bigram
evidence sources in WD. Both are substantially worse than character, making a
common effect of the large Bigram label space or segmentation the more
plausible fold-0 interpretation. This is not a final conclusion without the
remaining folds.

### Artifacts

The run completed uninterrupted with exactly epochs 0–299. All checkpoints
have 419-output heads and complete resumable state.

| Checkpoint | Epoch | SHA-256 |
| --- | ---: | --- |
| `best_cer.pth` | 282 | `a9fc407649838426a7b2aceec2f14ae67421b8600f502eb53abe892bed8d684f` |
| `best_wer.pth` | 278 | `891bbaf8c9d49d816a61adfc392f01d30ef88b3d59ba76ce3758a47da62dac18` |
| `latest.pth` | 299 | `18fb40e94e019ff2f843101b6a321ad1ab7f667373f37b8277250f4fec8da9d7` |

The complete run directory is
`results/thesis/bigram/linguistic_wd_rh/0/`. External backup location and
checksum: not yet recorded.

## B0 character baseline — WD/RH fold 0

### Setup

- Date completed: 2026-08-02
- Dataset: OnHW-Words500, writer-dependent, right-handed, fold 0
- Method: raw 13-channel IMU to character-level CTC labels
- Architecture: BLConv-B + BiLSTM-B
- Seed: 42
- Epochs: 300 (epochs 0–299)
- Batch size: 64
- Training machine: WSL laptop with NVIDIA GeForce RTX 4060
- Configuration: `configs/thesis/b0_char_wd_rh.yaml`

### Best validation results

| Metric | Value | Epoch |
| --- | ---: | ---: |
| Levenshtein distance | 0.722597 | 286 |
| Character error rate | 0.127563 (12.76%) | 286 |
| Word error rate | 0.359809 (35.98%) | 289 |
| Average reference length | 5.664615 characters | Not optimized |

The original paper's WD character result is aggregated across five folds, so
it must not be directly compared with this single-fold value.

### Artifacts and caveats

- The complete run directory was archived and backed up by the researcher.
- `latest.pth` and the epoch-299 milestone checkpoint were retained.
- This run preceded automatic best-CER/best-WER checkpoint saving. Metrics and
  predictions for epochs 286 and 289 are preserved, but their exact model
  weights are not. The epoch-299 model can be used for initial forced-alignment
  development.
- Backup archive checksum: not yet recorded in this log.

## B0 character baseline — WI/RH fold 0

### Setup

- Date completed: 2026-08-02
- Dataset: OnHW-Words500, writer-independent, right-handed, fold 0
- Method: raw 13-channel IMU to character-level CTC labels
- Architecture: BLConv-B + BiLSTM-B
- Seed: 42
- Epochs: 300 (epochs 0–299)
- Batch size: 64
- Training machine: WSL laptop with NVIDIA GeForce RTX 4060
- Configuration: `configs/thesis/b0_char_wi_rh.yaml`

### Best validation results

| Metric | Value | Epoch |
| --- | ---: | ---: |
| Levenshtein distance | 0.851474 | 244 |
| Character error rate | 0.156041 (15.60%) | 244 |
| Word error rate | 0.279478 (27.95%) | 269 |
| Average reference length | 5.456727 characters | Not optimized |

These are single-fold development results. They must not be presented as the
final writer-independent result, which will require the agreed five-fold
evaluation.

### Artifacts and use

- The run used automatic validation-selected checkpoint saving.
- `best_cer.pth` from epoch 244 is the preferred checkpoint for WI forced
  alignment because alignment quality depends most directly on character
  recognition.
- `best_wer.pth` from epoch 269 is retained for recognition comparison, while
  `latest.pth` remains the resumable end-of-training checkpoint.
- The complete run directory should be archived before alignment development.
- Backup archive checksum: not yet recorded in this log.
