# Thesis Experiment Log

This file records scientific runs and important development experiments. All
reported values must be traceable to the saved configuration, log, metrics,
predictions, code version, and experiment backup.

## ED dataset compatibility audit — pre-training

### Scope and dataset identity

- Audit date: 2026-09-21
- Source supplied by supervisor:
  `/mnt/c/Users/Ali/Downloads/ed-dataset/`
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

The supervisor requested that the initial ED evaluation use English
handwriting tokenizers, not the combined IAM+READ tokenizer. Consequently, the
frozen 419-class OnHW IAM+READ adapter is out of scope for this first ED
experiment. It also cannot encode any raw ED label because it lacks spaces,
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

The IAM-only model covers nearly all of the ED alphabet but lacks `%`, `(`,
and `=`. Those characters occur 416, 414, and 429 times respectively and make
906 of the 2,550 raw labels unencodable. IAM's `*` singleton does not occur in
ED. Silently deleting unsupported symbols or evaluating only the encodable
subset would change the task and bias the result.

The leakage-safe TVA adapter was frozen on 2026-09-21. It keeps all 145
IAM-derived bigrams and their score rows byte-for-byte at the JSON-value level,
then projects only the singleton layer to the declared 78-character ED task
alphabet. It has 224 output classes: blank ID 0, 78 singletons, and 145
bigrams. The pinned source is
`artifacts/tokenizers/source/iam-english-handwriting-bigram-v1.json`; the
canonical adapter is
`artifacts/tokenizers/ed_iam_handwriting_bigram_greedy_v1.json`, with
SHA-256
`d79062a252ac1ce05e1a309dab9dd439ee06a06e497024cc9321ecce401e40b7`.
Construction excludes IAM-only `*` and adds ED-only `%`, `(`, and `=` as
single-character fallbacks. No ED label value, label frequency, or recognition
result selects, ranks, or removes a handwriting bigram.

The repository boundary is explicit: DTLR constructs and freezes the generic
IAM-only English handwriting vocabulary using IAM training evidence only. It
must not read ED annotations or create an ED-specific vocabulary. TVA owns
the downstream ED task adapter, whose only task-specific construction input
is the predeclared 78-character `categories` schema. Complete ED labels may
be read only after the adapter is frozen, for encode/decode compatibility
auditing and recognizer training/evaluation—not for handwriting-token
selection, ranking, thresholds, or segmentation-policy tuning.

### Segmentation-policy audit

DTLR's IAM-only tokenizer does not use TVA's legacy greedy encoder. It uses a
maximum-total-utility dynamic program: for overlapping eligible pairs, it
chooses the non-overlapping set with the greatest sum of IAM handwriting
connectivity scores. TVA's legacy Bigram encoder instead scans left to right
and immediately consumes any available pair, without considering its score.

Applying both policies to all 2,550 ED labels with the same 145 IAM bigrams
produces different token identities for 1,778 samples (69.73%) and 350 of the
501 unique labels (69.86%). Both policies produce 85,153 target tokens in
total, so this difference changes which bigrams are learned rather than the
aggregate target length. For example, in `short`, DP selects `or`, whereas
greedy consumes `ho` first.

Therefore, switching the IAM tokenizer to greedy defines a new derived
segmentation variant, not simply loading the original DTLR model unchanged.
Greedy left-to-right was frozen as the ED primary policy on 2026-09-21 for
compatibility with established TVA Bigram behavior. It is exposed under the
separate tokenizer key `handwriting_bigram_greedy`; the existing
`handwriting_bigram` DP runtime and all completed OnHW experiments remain
unchanged. The matched ED comparator must also use greedy segmentation. DP may
be evaluated later only as an explicitly named ablation applied to both
vocabularies.

After the adapter bytes were frozen, `audit_ed_handwriting_bigram.py`
authenticated both source archives and encoded every canonical annotation in
both packages. The audit checked 2,550 records per archive (5,100 encodings),
170,306 total emitted target tokens, zero blank IDs, zero reconstruction
failures, identical WD/WI record-label mappings, and exact declared-alphabet
agreement. This is a compatibility result, not recognition evidence. The
focused regression suite passed 21 applicable tests; two unrelated optional
external-reference tests were skipped.

### Matched IAM-text linguistic comparator

The matched linguistic comparator was frozen on 2026-09-21 before ED model
training. Its vocabulary evidence comes only from the same complete 5,694-line
IAM training split used by the handwriting pipeline. The authenticated IAM
`labels.pkl` SHA-256 is
`5ac34ad37ba0b125308fe1a2bc97095985e25dfa76495628c3bb3895c0b446ab`;
the complete-train selection-manifest SHA-256 is
`7893dfba4febe6df99cf0bdb0c74fabe7b736d2a6af05033d8638b90455bc1c2`.
No IAM validation/test text and no ED label were used to select pairs.

The separate builder is required by this comparison policy, not because TVA
lacked greedy Bigram encoding. TVA's original `BigramTokenizer.train()` learns
a fold-specific vocabulary directly from the recognition dataset's
`train.json`, collapses labels to distinct word types, and saves only the token
mappings. Applying it to ED would create an in-domain, ED-training-label
baseline. That would not be validation leakage when implemented separately per
fold, but it would answer a different question and would not match the external
IAM evidence source used by the handwriting condition. The ED comparator
therefore uses a deterministic IAM-only builder, is frozen once for all WD/WI
folds, applies explicit lexical tie-breaking, and records source hashes and
leakage declarations. The additional metadata and artifact authentication do
not affect model predictions; the selected bigram inventory and resulting
target segmentation can affect training and recognition.

After NFC normalization, the builder counts every case-sensitive adjacent
ASCII-letter pair occurrence within each IAM training line. It found 861
candidate pairs and 153,350 eligible occurrences. Pairs are ranked by
descending occurrence count with lexical token tie-breaking, and the first 145
are retained to match the handwriting vocabulary. The last selected pair is
`bu` with count 273; the first excluded pair is `tu` with count 270, so the
selection boundary is not tied. The frozen evidence file is
`artifacts/tokenizers/source/iam-train-letter-bigram-counts-v1.json`, SHA-256
`7808d5d7b58354be982b042c8f60db4d63bf2ee12aa03cbeca6c3fe628cd1ebb`.

The canonical comparator is
`artifacts/tokenizers/ed_iam_linguistic_bigram_greedy_v1.json`,
SHA-256
`1da363e43fe9d300ece7ad5e3183d84a15fab4bdc6d5b2239b450c222b175abd`.
It uses the identical blank ID, ordered 78-character singleton layer,
145-bigram count, 224-class output size, NFC normalization, and greedy
left-to-right segmentation as the handwriting condition. Pair utilities are
normalized IAM counts retained for audit metadata; greedy segmentation does
not consult them. The two vocabularies share 72 bigrams and differ in 73 pairs
per condition.

The post-freeze compatibility audit encoded all 2,550 canonical ED labels in
both archives. The linguistic comparator emitted 79,078 target tokens per
archive, with zero blank IDs and zero round-trip failures. The handwriting
condition emits 85,153 per archive, so the linguistic vocabulary shortens the
ED targets by 6,075 tokens (7.13%). This is an intrinsic consequence of the
different frozen pair memberships—not a segmentation-policy difference—but it
must be considered when interpreting recognition results.

Accordingly, the reported ED comparison supports only the narrow claim
"IAM handwriting evidence versus IAM training-text frequency at matched size
and greedy policy." It does not establish superiority over a linguistic
tokenizer optimized directly from each ED training fold. Such an ED-trained
linguistic tokenizer is a valid additional baseline and could produce different
CER/WER, but it must be named and reported as a separate condition rather than
silently replacing the frozen IAM-matched control.

The complete TVA regression suite passed after integration: 40 tests ran, 38
passed, and two unrelated optional external-reference checks were skipped.

One provenance anomaly should be retained in the record: `gold_wd/build.log`
mentions a `gold_wi/meta.json` save path even though the archive's split
metadata correctly says `writer_dependent`. The fold contents themselves pass
the structural WD checks above.

### ED ZIP loader and matched pipeline smoke validation

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

`EdZipDataset` authenticates the WD/WI source archive SHA-256 before use and
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
`data/tva/ed`. On the current machine they were executed by setting
`TVA_ED_DATASET_DIR=/mnt/c/Users/Ali/Downloads/ed-dataset`, so no
machine-specific source path is stored in configuration files.

After loader integration, the complete TVA suite ran 46 tests: 44 passed and
two unrelated optional external-reference checks were skipped.

### ED fold-0 production configuration freeze

- Date frozen: 2026-09-23
- Conditions: handwriting/linguistic Bigram × WD/WI
- Architecture: BLConv-B + BiLSTM-B + CTC
- Input/output: seven 100 Hz channels; 224 classes including blank ID 0
- Training: 300 epochs, 30-epoch warmup, AdamW, learning rate 0.001,
  augmentation enabled, seed 42, batch size 64
- Evaluation/checkpoints: validation every epoch; resumable latest and
  best-CER/best-WER checkpoints; numbered checkpoint every 25 epochs

The four production configurations are under `configs/thesis/ed/` and are
matched in every non-condition field. All runs must start from scratch. The
ED signals are substantially longer than OnHW word signals: the archive
reports a mean of 2,137.65 samples, median 2,055, minimum 393, and maximum
7,314. Fold 0 contains 2,047 WD or 2,025 WI training recordings and 503 WD or
525 WI validation recordings. Batch size 8 was the conservative RTX 4060
timing setting; before production it was superseded by batch size 64 after the
A6000 systems benchmark below. The base YAML files retain that batch-64 A6000
setting. D022 later selected a separate batch-32 laptop protocol for the actual
five-fold matrix after all four fold-0 runs completed consistently at batch 32.

Before a production run, execute
`configs/thesis/ed/timing_bigram_handwriting_wd.yaml`. It differs from the
handwriting-WD production configuration only in running one epoch, using a
development result directory, a one-epoch warmup, and disabling numbered
milestone saves. It consumes the complete fold, so its training and validation
times can estimate the production budget. Its CER/WER and checkpoint are not
thesis recognition evidence and must not be reused for initialization.

The post-freeze regression suite ran 49 tests: 47 passed and the same two
optional external-reference checks were skipped.

### ED character baseline — WD fold 0 protocol

The character baseline is the reference condition for measuring whether either
Bigram vocabulary improves recognition. Its vocabulary is fixed directly from
the declared ED alphabet: blank ID 0 followed by 78 characters, for a 79-class
CTC output head. It contains no learned bigrams and reads no label frequencies,
so tokenizer construction cannot leak validation-label statistics.

The fold-0 configuration is `configs/thesis/ed/character_wd.yaml`. It matches
the actual batch-32 Bigram study in architecture (BLConv-B + BiLSTM-B), seven
input channels, WD fold 0, augmentation, seed 42, learning rate 0.001, and the
300-epoch schedule with 30 warmup epochs. Training starts from scratch and
writes to `results/thesis/ed/character_wd/0/`. This condition changes only the
token vocabulary and corresponding output-head width; it must not initialize
from a Bigram checkpoint.

The run completed on 2026-09-25 through epoch 299. Its independently best
validation CER was 0.158331 (15.83%) at epoch 241, and its independently best
validation WER was 0.546480 (54.65%) at epoch 199. The best-CER checkpoint has
SHA-256 `de86cac7b3c89966344d62bb8b1dc6d12b31df4bc5da3712ba3949543613d122`;
the best-WER checkpoint has SHA-256
`df8e3a2d5c52314bc2367fd3392a46d4ab3545c2c1c34e45280079aa1569011d`.

| ED WD fold-0 condition | Best CER | Best WER |
| --- | ---: | ---: |
| Character | **15.83%** | **54.65%** |
| IAM handwriting-aware Bigram | 24.27% | 64.80% |
| IAM linguistic Bigram | 32.20% | 71.17% |

On fold 0, character improves over handwriting-aware Bigram by 8.44 absolute
CER points (34.76% relative) and 10.16 WER points (15.67% relative). It
improves over linguistic Bigram by 16.37 CER points (50.83% relative) and
16.52 WER points (23.22% relative). This is strong fold-0 evidence that the
larger 224-class Bigram output space does not help this ED condition, despite
the handwriting-aware vocabulary remaining better than the linguistic Bigram
control. A final tokenizer conclusion requires the character baseline on all
five WD folds; this single-fold comparison is not a five-fold aggregate.

### ED character baseline — WI fold 0 protocol

The matched writer-independent character run uses
`configs/thesis/ed/character_wi.yaml`. It changes the distribution to WI and
writes to `results/thesis/ed/character_wi/0/`; all other scientific settings
match the WD character run and the completed batch-32 WI Bigram fold-0 pair.
The vocabulary remains blank ID 0 plus the fixed 78-character ED alphabet, for
a 79-class output head.

The run completed on 2026-09-25 through epoch 299. Its independently best
validation CER was 0.137717 (13.77%) at epoch 276, and its independently best
validation WER was 0.423058 (42.31%) at epoch 285. The best-CER checkpoint has
SHA-256 `d8831121878ab80f6639a9eee8e9b0a2d26a337293e75b9205324eb64698c548`;
the best-WER checkpoint has SHA-256
`6fd88eac1f4274b41904e628cf7b797f2c0aecb92627f2aee1995aa4486d68fa`.

| ED WI fold-0 condition | Best CER | Best WER |
| --- | ---: | ---: |
| Character | **13.77%** | **42.31%** |
| IAM handwriting-aware Bigram | 16.42% | 44.84% |
| IAM linguistic Bigram | 18.60% | 47.49% |

On fold 0, character improves over handwriting-aware Bigram by 2.65 absolute
CER points (16.13% relative) and 2.53 WER points (5.65% relative). It improves
over linguistic Bigram by 4.82 CER points (25.94% relative) and 5.19 WER points
(10.92% relative). Character therefore leads both Bigram systems on both WD
and WI fold 0. The WI margin over handwriting-aware Bigram is smaller than the
WD margin, and no five-fold character conclusion should be drawn until the
remaining folds are complete.

### ED full-architecture CUDA timing result

- Date: 2026-09-23
- Config: `configs/thesis/ed/timing_bigram_handwriting_wd.yaml`
- Hardware: NVIDIA GeForce RTX 4060 Laptop GPU
- Scope: complete WD fold 0; 2,047 train and 503 validation recordings
- Effective batches: 256 train and 63 validation at batch size 8
- Runtime: 82 seconds training and 4 seconds validation
- Startup caching: approximately 16 seconds total for validation and training
- CUDA mode: automatic mixed precision enabled
- Outcome: completed without a CUDA out-of-memory error
- Peak VRAM: not recorded
- Diagnostic checkpoint SHA-256:
  `c12625f50404673ccce98c514e782a841a37463894c18fcf293e521aeef47a53`

The one-epoch CER and WER were both 1.0, as expected for an untrained
sentence-level recognizer. They are not recognition evidence. The timing
checkpoint must not initialize any production model.

At the observed 86 seconds per train-plus-validation epoch, one 300-epoch run
is approximately 7 hours 10 minutes, excluding one-time caching and backup
overhead. The four fold-0 conditions would take roughly 28 hours 40 minutes if
run sequentially on the same RTX 4060. This is a planning estimate, not an
A6000 benchmark.

### RTX 4060 batch-32 fallback timing

- Date: 2026-09-23
- Hardware: NVIDIA GeForce RTX 4060 Laptop GPU
- Scope: complete WD fold 0; 2,047 train and 503 validation recordings
- Effective batches: 64 train and 16 validation
- Runtime: 26 seconds training and 2 seconds validation
- Observed GPU memory: maximum 5,211 MiB
- CUDA mode: automatic mixed precision enabled
- Outcome: completed without an out-of-memory error

Batch 32 is a practical laptop fallback and projects to about 2 hours 20
minutes for 300 train-plus-validation epochs, excluding caching and checkpoint
overhead. It produces 64 optimizer updates per WD fold-0 epoch, versus 32 for
the frozen A6000 batch-64 protocol. Consequently, a laptop batch-32 result may
be compared only with conditions trained using the same batch-32 protocol. It
must not be combined as fold 0 with batch-64 folds 1--4 in a five-fold summary;
such a final batch-64 matrix requires rerunning fold 0 at batch 64. The
one-epoch diagnostic metrics are not recognition evidence.

### ED handwriting-aware Bigram — WD fold 0, batch 32

- Completed: 2026-09-24
- Code commit: `9761142e491dc631c2411763fae3677edbfa861a`
- Hardware: NVIDIA GeForce RTX 4060 Laptop GPU
- Tokenizer: frozen IAM handwriting-aware Bigram artifact, greedy left-to-right
  segmentation
- Data: ED WD fold 0; no ED labels were used to select bigrams
- Training: 300 epochs, batch 32, seed 42, CUDA mixed precision
- Wall-clock interval including initial caching: approximately 1 hour 9 minutes
  25 seconds
- Best validation CER: 0.2427056328 at epoch 206
- Best validation WER: 0.6480386889 at epoch 245
- Mean validation reference length: 42.176938 characters
- Best-CER checkpoint SHA-256:
  `35a1f5ce12362bfbb9e2e50ccba249b5e09d517f974b667dcf5cfccf25c4db7b`
- Best-WER checkpoint SHA-256:
  `83f583c05d13d8b1972b880e58311294df35eab332fc623e24a2ccd26e07f993`
- Final resumable checkpoint SHA-256:
  `8d93f9f463d1e8d6e044abfa6dc8f052b1a5b5cb1b1c043d043e48aa93c11bcd`

The final epoch CER was 0.2484091 and WER was 0.6552929, both slightly worse
than their respective best values. This shows why checkpoint selection must be
metric-specific: `best_cer.pth` and `best_wer.pth` represent different epochs,
and `latest.pth` is not the reported optimum. This is the first half of the
controlled WD fold-0 comparison. No tokenizer conclusion is valid until the
matched IAM linguistic-Bigram condition is trained with the same batch size,
fold, seed, architecture, and schedule. Because this run used batch 32, it is
not interchangeable with a batch-64 fold in the planned A6000 five-fold matrix.

### ED linguistic Bigram — WD fold 0, batch 32

- Completed: 2026-09-24
- Code commit: `9761142e491dc631c2411763fae3677edbfa861a`
- Hardware: NVIDIA GeForce RTX 4060 Laptop GPU
- Tokenizer: frozen IAM linguistic-frequency Bigram artifact, greedy
  left-to-right segmentation
- Data: ED WD fold 0; no ED labels were used to select bigrams
- Training: 300 epochs, batch 32, seed 42, CUDA mixed precision
- Wall-clock interval including initial caching: approximately 1 hour 9 minutes
  36 seconds
- Best validation CER: 0.3219891586 at epoch 203
- Best validation WER: 0.7117141322 at epoch 270
- Mean validation reference length: 42.176938 characters
- Best-CER checkpoint SHA-256:
  `6a3270ac9e7b3733a941a6e9177d51b90a7f266282123e2b3594ea8d0dd8bc53`
- Best-WER checkpoint SHA-256:
  `35d1f0dd9fef8677822407e83c62e1481c915b70bea63799612cdbde1ce99c7f`
- Final resumable checkpoint SHA-256:
  `c2cf32c732097546bca16406bcc79cd620189e056ea24462c845fb8419b49d3b`

Under this matched WD fold-0 protocol, the handwriting-aware tokenizer reduced
CER from 0.3219891586 to 0.2427056328: an absolute reduction of 0.0792835258,
or 7.93 percentage points and 24.62% relative. It reduced WER from
0.7117141322 to 0.6480386889: an absolute reduction of 0.0636754433, or 6.37
percentage points and 8.95% relative. Mean Levenshtein distance decreased by
3.343936 characters per validation sentence. The result supports the
handwriting-aware hypothesis on this one WD fold, but does not establish
five-fold generalization or statistical significance. Both conditions used a
single seed and metric-specific best epochs. The batch-32 comparison also
remains separate from the frozen batch-64 A6000 matrix.

### ED handwriting-aware Bigram — WI fold 0, batch 32

- Completed: 2026-09-24
- Code commit: `9761142e491dc631c2411763fae3677edbfa861a`
- Hardware: NVIDIA GeForce RTX 4060 Laptop GPU
- Tokenizer: frozen IAM handwriting-aware Bigram artifact, greedy left-to-right
  segmentation
- Data: ED WI fold 0; no ED labels were used to select bigrams
- Training: 300 epochs, batch 32, seed 42, CUDA mixed precision
- Runtime including initial caching: approximately 1 hour 11 minutes 7 seconds
- Best validation CER: 0.1642117744 at epoch 264
- Best validation WER: 0.4483709273 at epoch 265
- Best-CER checkpoint SHA-256:
  `bc22bedd7636193c661f55515cdd4d54dbd5b6824bfe2415f939449e9e12bdc5`
- Best-WER checkpoint SHA-256:
  `2cd1c3c0d05e13194c823de2587b9a64083794d001f0e407f95e8c196b22eeaf`

### ED linguistic Bigram — WI fold 0, batch 32

- Completed: 2026-09-24
- Code commit: `9761142e491dc631c2411763fae3677edbfa861a`
- Hardware: NVIDIA GeForce RTX 4060 Laptop GPU
- Tokenizer: frozen IAM linguistic-frequency Bigram artifact, greedy
  left-to-right segmentation
- Data: ED WI fold 0; no ED labels were used to select bigrams
- Training: 300 epochs, batch 32, seed 42, CUDA mixed precision
- Runtime including initial caching: approximately 1 hour 10 minutes 38 seconds
- Best validation CER: 0.1859637922 at epoch 262
- Best validation WER: 0.4749373434 at epoch 262
- Best-CER and best-WER checkpoint SHA-256 (the same epoch and bytes):
  `25ccf0c05377e0f2157c589874646716162ae72e7fb07c77b8ea69151e708365`
- Final resumable checkpoint SHA-256:
  `24819bd58fffa6c0dd7e067cd9fac5d18a6bba8bedb543835c8a4f3717276baa`

Under the matched WI fold-0 protocol, handwriting-aware Bigram reduced CER by
0.0217520179, or 2.18 percentage points and 11.70% relative, and reduced WER
by 0.0265664160, or 2.66 points and 5.59% relative. Mean Levenshtein distance
decreased by 0.908571 characters per validation sentence. Handwriting-aware
Bigram therefore outperformed the linguistic control on both WD and WI fold 0,
although effect size was larger on WD. This agreement across two distributions
is promising, but a single fold and seed do not establish generalization or
statistical significance. These batch-32 results must remain separate from a
batch-64 five-fold aggregate.

### ED five-fold laptop protocol and launcher

The final ED matrix is frozen at physical batch size 32 on the RTX 4060
laptop. The four completed fold-0 conditions are retained; the remaining
matrix comprises handwriting-aware and linguistic Bigram under WD and WI for
folds 1--4, or 16 sequential runs. This choice avoids mixing batch protocols
and supersedes the provisional A6000 batch-64 deployment plan for this matrix.
The A6000 timings remain valid systems evidence but are not recognition runs.

`run_ed_matrix.py` makes the effective protocol explicit without changing the
four batch-64 base templates. It overrides only `idx_fold` and `size_batch`,
defaults to batch 32 and folds 1--4, validates the frozen training invariants,
checks all 16 target directories before launching anything, and refuses to
overwrite any non-empty result directory. Each `main.py` run saves its actual
effective YAML in its result directory. Preview the complete plan with:

```bash
python run_ed_matrix.py --dry-run
```

After setting `TVA_ED_DATASET_DIR`, start the sequential matrix with:

```bash
MPLBACKEND=Agg python run_ed_matrix.py
```

The launcher stops on the first nonzero training exit and leaves all existing
outputs intact. Continue only with explicitly selected untouched folds and
conditions after diagnosing and documenting a failure. At the observed fold-0
runtime of about 69--71 minutes, the 16 remaining runs require roughly 18--19
hours sequentially, excluding interruption and backup overhead.

### ED WD five-fold comparison completed

- Completed: 2026-09-25
- Code for folds 1--4: commit `1047487` (`Add safe ED five-fold launcher`)
- Protocol: ED WD, folds 0--4, batch 32, seed 42, 300 epochs,
  BLConv-B + BiLSTM-B + CTC, CUDA mixed precision, and greedy left-to-right
  segmentation
- Controlled variable: the frozen IAM bigram-selection source
  (handwriting-aware evidence versus linguistic frequency)
- Result directories: `results/thesis/ed/handwriting_wd/{0..4}/` and
  `results/thesis/ed/linguistic_wd/{0..4}/`

Every run reached epoch 299 and retained `latest.pth`, `best_cer.pth`, and
`best_wer.pth`. The values below are the independently best validation metrics;
CER and WER can therefore come from different epochs and checkpoints.

| Fold | Handwriting CER (epoch) | Linguistic CER (epoch) | CER reduction | Handwriting WER (epoch) | Linguistic WER (epoch) | WER reduction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 24.27% (206) | 32.20% (203) | 7.93 pp | 64.80% (245) | 71.17% (270) | 6.37 pp |
| 1 | 23.04% (202) | 27.35% (220) | 4.31 pp | 62.22% (272) | 65.69% (205) | 3.46 pp |
| 2 | 23.45% (257) | 27.80% (225) | 4.35 pp | 61.88% (187) | 65.38% (225) | 3.50 pp |
| 3 | 25.82% (234) | 32.47% (165) | 6.65 pp | 66.41% (244) | 70.99% (248) | 4.59 pp |
| 4 | 24.05% (251) | 30.39% (268) | 6.33 pp | 65.99% (254) | 69.57% (276) | 3.59 pp |
| **Unweighted mean ± sample SD** | **24.13 ± 1.06%** | **30.04 ± 2.39%** | **5.91 pp** | **64.26 ± 2.10%** | **68.56 ± 2.84%** | **4.30 pp** |

On the unweighted five-fold mean, handwriting-aware bigrams reduce CER by
5.91 percentage points (19.69% relative) and WER by 4.30 points (6.27%
relative) compared with the matched linguistic bigrams. The direction is
consistent in every fold for both metrics. This is substantially stronger
evidence than the earlier fold-0 observation, but it is still one seed and is
specific to ED's writer-dependent split. The corresponding five-fold WI
comparison remains necessary before drawing a writer-independent conclusion.

The linguistic fold-2 process was killed by the host OOM handler after epoch
208 while another memory-intensive process overlapped it. It was resumed from
that run's `latest.pth` with the saved effective configuration and completed
through epoch 299. The other WD runs completed without a recorded training-log
error.

| Condition / fold | Best-CER checkpoint SHA-256 | Best-WER checkpoint SHA-256 |
| --- | --- | --- |
| Handwriting 0 | `35a1f5ce12362bfbb9e2e50ccba249b5e09d517f974b667dcf5cfccf25c4db7b` | `83f583c05d13d8b1972b880e58311294df35eab332fc623e24a2ccd26e07f993` |
| Handwriting 1 | `d051dfd553c3e46891dd2328c88e255f3ebd73174531e7ac8b566b4407933834` | `8a75c045ce51f28f2deb388f3921597afa44b41edbfa28f900ac4130d6aa25ca` |
| Handwriting 2 | `649bcb2cc917f4de816c3c8d500989a05a389c813f90722bb9726bfc27b3f6b1` | `fa7fa42cc6276c1f6377d71e3ef0c720d59450f528a76e10587e6dc5a58dcb5b` |
| Handwriting 3 | `5e7181e8eaabef25e8b00f8ef21e3eef1739c898f2f84452b4eca6b593542345` | `a9edaa61291ba7b2e7bb7ec571188c95c5acbca142ec718bc01c15102c529699` |
| Handwriting 4 | `8e64ebe165b538a9fc12cfdc897313fe6c2a6525b6242cca8039d8f1935e88ee` | `577d60901aa460eb8c07b06ae86d50ced16912fe7060a9ac3c553afaa9ef0571` |
| Linguistic 0 | `6a3270ac9e7b3733a941a6e9177d51b90a7f266282123e2b3594ea8d0dd8bc53` | `35d1f0dd9fef8677822407e83c62e1481c915b70bea63799612cdbde1ce99c7f` |
| Linguistic 1 | `daa6b062cdbd16e7d06f55d65b95f59924d62ab21ede63572216055cafb2a312` | `b33ba6f0d2a1fe97382f4f76cbcde150bd26d22053e980702b4f25a5cc2066bb` |
| Linguistic 2 | `0153de2d843a785d8c194983df0c2965ea93b6387b62b98fe1d07f04bef35f58` | `0153de2d843a785d8c194983df0c2965ea93b6387b62b98fe1d07f04bef35f58` |
| Linguistic 3 | `578c905d84ef465ec28d5fd66af55e7bda48fa26b384483615fb3636082eb216` | `15954a74b00ad39a3bb37b3815a77deef7f6b209ae973cc0c2606f8f1460c416` |
| Linguistic 4 | `8665a07bb1f9c99d204369a24c7d6d8fbe3debcc540d426f147de441885436a0` | `06cf73bd27e30f94fabf24a2fdd5f579776803ab84a6afbfb395d93ee6673a05` |

### A6000 batch-8 deployment timing

- Date: 2026-09-23
- Hardware: NVIDIA RTX A6000, 48 GB
- Config and data: the same complete WD fold-0 batch-8 timing condition
- Runtime: 100 seconds training and 2 seconds validation
- Startup caching: approximately 3 seconds validation and 10 seconds training
- Observed GPU memory: maximum 2,234 MiB
- Sampled utilization: 0–43% at one-second intervals
- Outcome: completed without an out-of-memory error

This run was slower in training than the RTX 4060 measurement despite much
more available memory. The observation suggests that batch 8 does not keep the
A6000 busy, although the utilization samples alone cannot distinguish data
loading, augmentation, padding, kernel-launch, or recurrent-model limitations.
No recognition conclusion may be drawn from the one-epoch metrics.

### A6000 batch-64 deployment timing and provisional choice

- Date: 2026-09-23
- Scope: complete WD fold 0; 2,047 train and 503 validation recordings
- Effective batches: 32 train and 8 validation
- Runtime: 16 seconds training and 1 second validation
- Observed GPU memory: maximum 7,248 MiB of 48 GB
- Sampled utilization: maximum 47% at one-second resolution
- Outcome: completed without an out-of-memory error

Batch 64 reduced A6000 training time by 84 seconds per epoch, a 5.9× speedup
over batch 8. It also matches TVA's established production batch size. The
tradeoff is fewer optimizer updates: 32 per epoch and 9,600 across 300 epochs
for WD fold 0, compared with 76,800 under batch 8. Each epoch still covers the
complete fold. The 300-epoch schedule, 30-epoch warmup, learning rate, and all
other settings remain unchanged. Batch 64 was provisionally frozen identically
across all 20 runs, but no production run used it before D022 selected the
complete laptop batch-32 protocol. Diagnostic CER/WER was not used for the
A6000 decision.

At 17 seconds per train-plus-validation epoch, one 300-epoch run is estimated
at 1 hour 25 minutes. Four fold-0 conditions are approximately 5 hours 40
minutes, and the complete 20-run five-fold matrix is approximately 28 hours 20
minutes when executed sequentially, excluding caching and backup overhead.

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
  weights are not. The epoch-299 model remains available for reproducibility.
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

### Artifacts

- The run used automatic validation-selected checkpoint saving.
- `best_cer.pth` from epoch 244 and `best_wer.pth` from epoch 269 are retained
  for recognition comparison, while
  `latest.pth` remains the resumable end-of-training checkpoint.
- The complete run directory should be archived.
- Backup archive checksum: not yet recorded in this log.

## Original TVA linguistic bigram trained on IAM — notebook and pre-training audit

- Date: 2026-10-03
- Notebook: `train_iam_tva_bigram_tokenizer.ipynb`, following the existing
  `train_tokenizers.ipynb` factory/train/load workflow.
- Native artifact:
  `artifacts/tokenizers/ed_iam_tva_original_bigram_greedy_v1.json`.
- Artifact SHA-256:
  `796a676bcfe4087acfdebf795a5e9797a8dd05ff93054d9e72de14e4d44eb73f`.
- Provenance and full comparison are saved beside the artifact as
  `ed_iam_tva_original_bigram_greedy_v1.provenance.json` and
  `ed_iam_tva_original_bigram_greedy_v1.audit.json`.

### Method

The notebook authenticates the existing IAM `labels.pkl` and complete
training selection manifest using their pinned hashes and every selected
transcription hash. All 5,694 training lines are represented in a temporary,
one-fold TVA `train.json`; the original `BigramTokenizer.train()` reduces
them to 5,337 unique complete labels. No IAM validation/test labels enter
tokenizer training. Raw training text is retained without additional
normalization, filtering, lowercasing, or stripping.

The original trainer counts every adjacent pair and uses
`Counter.most_common`, with original greedy encoding at runtime. Its
implementation is unchanged. Blank plus the existing 78-character ED alphabet
leave 145 bigram slots in a 224-class vocabulary. Selected unrestricted pairs
are checked for ED alphabet compatibility, with no silent filtering.

A subprocess launched with `PYTHONHASHSEED=42` makes the original set iteration
and encounter-order tie behavior reproducible under recorded Python 3.11.15.
Two training executions produced byte-identical artifacts. This retains the
original algorithm; it does not introduce lexical tie-breaking. There are
1,397 candidate pairs. The selection boundary count is 385, shared by
`fi`, `ig`, and `ke`; hash seed/Python details therefore matter to exact
membership and token IDs. Provenance records the trainer-source hash.

### Vocabulary and ED segmentation comparison

Original TVA selects 109 ASCII-letter pairs and 36 non-letter pairs.
Of the latter, 35 contain a space and three contain punctuation; these
categories overlap. No selected pair contains a digit. This candidate policy
differs from the handwriting and custom linguistic letter-pair conditions.

| Vocabulary pair | Shared bigrams (of 145 each) |
| --- | ---: |
| Original TVA / handwriting | 58 |
| Original TVA / custom linguistic | 109 |
| Handwriting / custom linguistic | 72 |

Both authenticated ED archives contain the same 2,550 labeled records with
106,087 characters. All three tokenizers reconstructed all records exactly,
with zero blank targets and zero round-trip failures. Per-archive totals are
identical because the records are identical; they are not independent samples.

| Condition | Encoded tokens per archive | Mean tokens per record | Reduction from characters |
| --- | ---: | ---: | ---: |
| Original TVA on IAM | 70,766 | 27.75 | 33.29% |
| Custom linguistic on IAM | 79,078 | 31.01 | 25.46% |
| Handwriting on IAM | 85,153 | 33.39 | 19.73% |

Original TVA differs in token-string segmentation from the custom comparator
on 2,427 records and from handwriting on 2,505 records. The audit also saves
all five supplied WD/WI train/validation-fold token totals and example
segmentations. ED labels are first read after tokenizer selection and freezing.

### Interpretation and next step

This is a separate linguistic baseline, not a replacement for the custom
comparator or its completed results. Both counting and candidate eligibility
differ, so outcomes cannot isolate either factor alone. Target compression
does not establish recognition accuracy. The handwriting minimum-count 20 and
connected-rate 0.5 selection thresholds remain provisional limitations.

The artifact loads using `tokenizer: bigram` and the native JSON path above.
Recognition training has not been launched and no recognition config was
created for this condition. A future manual run must use a separate result
directory and match the saved ED study protocol: batch 32, seed 42, 300 epochs,
BLConv-B + BiLSTM-B, seven channels, standard augmentation, no concatenation.
Some older templates still specify batch 64. The supervisor's OnHW-trained
transfer setup still requires its exact tokenizer/config/scores for comparison.
## 2026-10-03 — Retired custom linguistic vocabulary builders

Removed the superseded construction paths:
`build_ed_linguistic_bigram_adapter.py`, `tva/ed_linguistic_bigram.py`,
`build_linguistic_bigram_tokenizers.py`, and `tva/linguistic_bigram.py`.
New IAM linguistic vocabularies are trained with the original TVA algorithm
through `train_iam_tva_bigram_tokenizer.ipynb`.

Retained frozen custom ED/OnHW vocabularies, frequency evidence, saved configs,
results, and the small compatibility loaders in
`tva/handwriting_bigram_tokenizer.py`. Those loaders still authenticate and
encode historical artifacts with their original greedy or utility-DP policy;
they do not select vocabularies. Their pinned artifact identities now live
beside the loaders rather than importing deleted construction modules. The
analysis notebook and historical experiment configs continue to work.

Builder-only tests were retired; frozen-artifact integrity, source-provenance,
round-trip, and runtime tests remain. The IAM notebook carries its two pinned
source hashes directly and no longer imports the retired ED builder.
All frozen JSON files remained byte-identical. Existing character-baseline
changes were preserved.

Validation: 55 repository tests completed successfully (seven optional skips).
Every IAM notebook code cell was then executed successfully, reproducing the
same tokenizer, provenance, and complete ED audit files. No recognition
training was launched. For reconstruction of the retired historical policies,
the removed source is available in Git revision
`339a2f49a3538c48aa4206c988048fd38534109b`; it is no longer an active builder
entry point. No commit or push was performed.
## 2026-10-03 — Config and artifact cleanup

Removed 11 redundant config templates: the four OnHW handwriting/custom
linguistic Bigram templates, two ED custom linguistic production templates,
four ED smoke templates, and the ED timing template. Exact historical run
configs remain in their original result directories. The related OnHW
template-only tests were retired; historical tokenizer/runtime tests remain.
ED smoke checks now derive bounded in-memory settings from the active
production configs rather than retaining duplicate YAML files.

Removed nine unused artifacts: the eight custom OnHW linguistic vocabularies
for WD/WI folds 1–4 (no corresponding saved runs), and the full IAM
letter-pair count evidence used only by the retired custom builder.
Retained both completed OnHW linguistic fold-0 vocabularies and their
unchanged original checksum manifests. Those original manifests still record
all five original folds; only the completed fold-0 artifact is retained
locally. Removed tracked configs/evidence/vocabularies are recoverable from
Git revision `339a2f49a3538c48aa4206c988048fd38534109b`.

The ED custom linguistic tokenizer remains for completed-run analysis and
the IAM notebook's three-condition comparison. Its source hashes and selected
counts remain in its frozen metadata. The IAM-only handwriting source,
ED handwriting adapter, native original-TVA tokenizer, provenance, audit,
and historical IAM+READ handwriting artifact are also retained. All retained
artifact bytes and every saved result YAML remained unchanged.

The six active ED production configs now cover character, handwriting, and
original TVA-on-IAM, each in WD/WI. Added explicitly named
`bigram_tva_original_wd.yaml` and `bigram_tva_original_wi.yaml`, with
`tokenizer: bigram`, the shared native IAM artifact, batch 32, and isolated
`results/thesis/ed/tva_original_wd|wi` directories. The two current
handwriting templates now specify batch 32, matching the actual completed ED
study and character configs. Saved historical batch settings were not edited.

The matrix launcher uses `tva_original_wd|wi` instead of the removed custom
linguistic config names. Existing output protection is preserved. Include
fold 0 explicitly when starting the new baseline; launcher defaults still
start at fold 1. Recognition training remains manual.

Validation: 52 repository tests completed successfully, with seven optional
external-data/reference skips. The original IAM notebook passed end to end against the available ED archives,
reproducing the frozen outputs. The launcher dry run validated all ten new
WD/WI fold configurations without launching recognition training. No
recognition training, commit, or push was performed.
## 2026-10-03 — Fold-0 original-TVA linguistic comparison prepared

The next comparison uses original TVA-on-IAM linguistic Bigram against the
completed IAM handwriting Bigram condition on ED fold 0. Both WD and WI
linguistic configs passed launcher preflight without launching recognition
training. The native tokenizer is already frozen; the new recognition models
have no saved runs yet.

Existing handwriting run JSON files report independently best validation
CER/WER: WD 24.270563% / 64.803869% (epochs 206 / 245), and
WI 16.421177% / 44.837093% (epochs 264 / 265). Saved run YAMLs confirm the
batch-32, seed-42, 300-epoch, seven-channel BLConv-B + BiLSTM-B protocol with
augmentation and no concatenation. Their historical dataset/tokenizer names
predate the documented ED metadata rename; those records were left unchanged.
The existing handwriting results can be reused for this matched comparison.

Manual WSL command after setting `TVA_ED_DATASET_DIR`:

```bash
/home/artellisys/miniconda3/envs/tva/bin/python run_ed_matrix.py --conditions tva_original_wd tva_original_wi --folds 0
```

Run from `/home/artellisys/TVA`. The launcher trains the two new recognizers
sequentially and evaluates validation data each epoch. It uses isolated
`tva_original_wd/0` and `tva_original_wi/0` result directories and refuses
occupied outputs. Do not use the earlier custom linguistic result directories
for this condition. Original-TVA CER/WER remain pending the manual runs.

## 2026-10-04 — Original TVA-on-IAM ED fold-0 results verified

Recorded the completed manual original-TVA linguistic recognition runs for
ED WD and WI fold 0. Both result files contain exactly epochs 0–299.
Their saved configs confirm batch 32, seed 42, 300 epochs, 30 warmup epochs,
learning rate 0.001, BLConv-B + BiLSTM-B, seven input channels, standard
augmentation, no concatenation, and the shared frozen IAM native bigram
tokenizer. Vocabulary selection did not use ED labels.

Native tokenizer SHA-256:
`796a676bcfe4087acfdebf795a5e9797a8dd05ff93054d9e72de14e4d44eb73f`.

### Original TVA linguistic metrics

| Split | Metric | Raw value | Percent | Epoch |
| --- | --- | ---: | ---: | ---: |
| WD | CER | 0.3595569172755126 | 35.96% | 191 |
| WD | WER | 0.7923159591617410 | 79.23% | 224 |
| WI | CER | 0.20931187012631675 | 20.93% | 250 |
| WI | WER | 0.5243107769423558 | 52.43% | 258 |

Minimum validation Levenshtein distance was 15.165009940357853 for WD
(epoch 191) and 8.742857142857142 for WI (epoch 250).
Average validation reference length was 42.17693836978131 characters for WD
and 41.76952380952381 for WI. These lengths are reference statistics, not
optimized model scores. Best CER and WER are selected independently and
need not occur at the same epoch.

### Matched fold-0 comparison

All values below are independently best validation CER/WER from the saved
300-epoch runs. The custom comparator is a retained historical condition;
its results are not reassigned to original TVA.

| Condition | WD CER | WD WER | WI CER | WI WER |
| --- | ---: | ---: | ---: | ---: |
| Character | 15.83% | 54.65% | 13.77% | 42.31% |
| IAM handwriting bigram | 24.27% | 64.80% | 16.42% | 44.84% |
| Custom IAM linguistic bigram | 32.20% | 71.17% | 18.60% | 47.49% |
| Original TVA-on-IAM linguistic bigram | 35.96% | 79.23% | 20.93% | 52.43% |

Handwriting has lower error than original TVA by 11.69 CER and 14.43 WER
percentage points on WD, and 4.51 CER and 7.59 WER points on WI.
Character remains the best tested condition on both splits. Original TVA is
also worse than the earlier custom linguistic comparator on this fold.

This confirms that using the original TVA algorithm on IAM does not improve
the fold-0 linguistic baseline in these runs. Original TVA has shorter
encoded ED targets than both other bigram vocabularies, but lower target
counts did not translate into lower recognition error here. This is an
observed association, not an explanation of the cause.

The evidence is one fold and one seed for each split. Complete the remaining
folds before generalizing the ranking. Vocabulary selection conditions differ
in candidate eligibility and counting/connection evidence; these results do
not isolate a causal effect of handwriting connectivity. The handwriting
selection thresholds remain provisional. The supervisor's OnHW-trained
transfer experiment still requires its artifact, config, and scores for a
direct comparison.

### Saved run evidence

WD directory: `results/thesis/ed/tva_original_wd/0/`.

| File | SHA-256 |
| --- | --- |
| `train_20261003230725.json` | `9a8001737de89c96a99ee7613edbb7084a05538bb33c27750d7da1842026ae52` |
| `train_20261003230725.yaml` | `2fd3627b42468032d9c04cd9abd0b11cec39b662d7aafe11d9bb9a98a417b59b` |
| `checkpoints/best_cer.pth` | `40ca32849769ed7a7bcd971ed94930dcd60daea31beb27be9b73470dbffb89c1` |
| `checkpoints/best_wer.pth` | `b571999fb3f8f76544bc6f7627b81994f80a2b57920b724287351dc736e4822d` |

WI directory: `results/thesis/ed/tva_original_wi/0/`.

| File | SHA-256 |
| --- | --- |
| `train_20261004001614.json` | `c25d59a60ee3db7b6edf586da6c44b8c3b135ad7638fa553d49bbb9832238c07` |
| `train_20261004001614.yaml` | `60d9d9c41de45064c0b6af5c56513036cd0e3c915c53da4960d87abc79be8f15` |
| `checkpoints/best_cer.pth` | `48e17ec8c5d1a9f6b1a32440948f2afb90f93df6ed2087f63a42d45cbd4b1e28` |
| `checkpoints/best_wer.pth` | `304b16fc435bfa51ed6fc85228a24edb6c7e83554de9e34721acde49c8018865` |

Metrics were read from the saved per-run JSON `best` entries; the generic
`evaluate.py` summary currently reports WER at the best-CER epoch for
training runs, which is a different convention. Use the independent minima
above consistently when comparing with the previous thesis tables.
External backup location/checksum is not yet recorded.

No tokenizer policy, training config, saved result, or checkpoint was changed
during verification. No additional recognition run, commit, or push was
performed.

## 2026-10-04 — Simplified tokenizer layout and ED notebook

The ED entry point is now train_ed_tokenizers.ipynb: declare categories and
source paths, import classes, call train, and inspect an encoding example.
The original train_tokenizers.ipynb remains unchanged for OnHW. The historical
IAM+READ notebook is now train_onhw_handwriting_tokenizer.ipynb.

Runtime handwriting classes live in tva/handwriting_tokenizers.py. Source
validation, vocabulary preparation, IAM training, and optional audits live
in tva/handwriting/. The old root builder/audit scripts and scattered
handwriting modules were replaced by this package; their consumers were
updated. Original TVA training and segmentation algorithms are unchanged.

Executed the new ED notebook and reproduced the existing tokenizer/provenance
bytes. The relocated full comparison audit reproduced the existing audit
report exactly. Frozen artifacts, recognition configs, and saved results
remain unchanged. No recognition training was launched.

Validation: both simplified notebooks executed successfully; unittest discovery
ran 56 tests (49 passed, 7 skipped for optional external inputs).

### Original TVA tokenizer module restored

Restored tva/tokenizers.py byte for byte from the initial TVA commit cff221c.
All thesis extensions (IAM training wrapper, explicit character alphabet,
handwriting/historical linguistic classes, extended factory, and shared-file
path resolution) live in tva/handwriting_tokenizers.py. Recognition entry
points, the ED notebook, audits, and tests import the extensions there.
Original tokenizer algorithms are inherited or called through the original
factory. Existing configuration keys and artifact formats remain unchanged.

Verified both notebooks and the byte-identical comparison audit after restoring
the upstream file. All 56 tests completed: 49 passed, 7 optional skips.

## 2026-10-08 — Frozen IAM handwriting BPE/Unigram integrated for ED fold 0

Imported the DTLR release described in
`/home/artellisys/DTLR/poc/provenance/iam-handwriting-subwords-v1-freeze.md`.
The original released files and manifest are bundled under
`artifacts/tokenizers/source/` with their exact bytes. DTLR's own frozen loader
validated both sources; its reconstructed release manifest matched the pinned
manifest exactly. No DTLR files were modified.

| Artifact | SHA-256 |
| --- | --- |
| IAM BPE source | `a504c2306200faf6eab98f3feef6f98b81e6e6c6c7f707cf31aa3399a660be47` |
| IAM Unigram source | `af2bc8455afd796367737857f3a36f1deb45f07ec9783a4bbd7f15057f2b9476` |
| Release manifest | `e8b42352a286dde90d6956ae22e92a33d8a94fcf8d251d2ce1cfe224492b6528` |
| ED BPE adapter | `fcf244d112f55fc7e30edde4eaaabda70d282ea193b7834acbdcdac6cbc4c4ea` |
| ED Unigram adapter | `e1628c0d2c30cdc9d05898a848c80bfb055d282cf78d39aa27d6d1a3cf8471f0` |

### Alphabet and inference policy

The separately versioned ED adapters each contain blank ID 0, the 78 declared
ED singleton characters, and all 145 selected IAM pieces: 224 classes, matching
the existing ED bigram comparison. They remove IAM-only singleton `#`, `&`, `*`,
add `%`, `=`, and remap IDs. The source releases remain unchanged at 225 classes.
Only the declared ED alphabet enters adapter construction; no ED label values,
frequencies, folds, or recognition metrics enter vocabulary selection or fitting.

BPE applies the frozen ordered merge list with left-to-right merge application.
Unigram uses the frozen log probabilities and DTLR's Viterbi traversal/tie rule.
No learned score is changed or renormalized. Added `%` and `=` have one forced
singleton arc with score zero. Since all learned multi-character pieces consist
of letters, these symbols cannot join a competing piece. The resulting scores
are inference weights, not a newly normalized ED probability distribution.
Symbols outside the declared alphabet raise an error; encoding never emits blank.

This compares complete tokenizer methods: greedy handwriting bigram, ordered
BPE merges, and Unigram Viterbi. It does not isolate vocabulary membership alone.
All share IAM train evidence, the 145-piece budget and the ED recognition setup.
DTLR thresholds (count 20/rate 0.5), length cap 6 and training settings remain
exploratory. Unigram uses handwriting for candidate eligibility/initialization,
then text likelihood for fitting. Text-only BPE/Unigram controls are separate
future experiments. Fold-0 recognition results remain pending.

### Code and manual runs

`HandwritingBPETokenizer` and `HandwritingUnigramTokenizer` live in
`tva/handwriting_tokenizers.py`; authentication and deterministic preparation
live in `tva/handwriting/subwords.py`. Loading uses the bundled source beside
the adapter and verifies exact adapter reconstruction. No DTLR dependency or
checkout is needed for recognition. `tva/tokenizers.py` remains byte-identical
to initial commit `cff221c`. Earlier artifacts, configs and results are unchanged.

The simple `train_ed_tokenizers.ipynb` now prepares both adapters through `.train()`
and demonstrates encoding. Files are already present; running it is optional.
The call adapts alphabet/IDs and accepts only byte-identical existing outputs;
it does not retrain the released IAM model.

Run manually from the TVA root in the WSL `tva` environment:

```bash
export TVA_ED_DATASET_DIR=/mnt/c/Users/Ali/Downloads/fau-english-dataset
python main.py -c configs/thesis/ed/bpe_handwriting_wd.yaml
python main.py -c configs/thesis/ed/bpe_handwriting_wi.yaml
python main.py -c configs/thesis/ed/unigram_handwriting_wd.yaml
python main.py -c configs/thesis/ed/unigram_handwriting_wi.yaml
```

All four configs have `idx_fold: 0`, 300 epochs, 30 warmup epochs, batch 32,
seed 42, learning rate 0.001, `blconv_b`/`bilstm_b`, seven channels,
augmentation enabled and no concatenation. Apart from tokenizer/file/output
paths they exactly match the existing handwriting bigram WD/WI configs.
Outputs are isolated under `results/thesis/ed/handwriting_{bpe,unigram}_{wd,wi}/`.
`run_ed_matrix.py` retains its existing bigram conditions; use these configs
directly for the new fold-0 runs. No recognition training was launched here.

### Validation

- TVA unittest discovery: 61 tests, 54 passed and 7 optional-input skips.
- The complete tokenizer-preparation notebook executed successfully and
  reproduced all existing tokenizer bytes.
- TVA and DTLR segmentations match exactly on all 5,612 IAM training lines
  covered by the ED alphabet; the other 82 contain IAM-only symbols.
- All 5,100 ED encodings per tokenizer (2,550 records in each of WD and WI)
  decode exactly with no blank. They match reference chunk encoding between
  added forced symbols, including the DTLR BPE merge order/Unigram choices.
- Per unique ED set: BPE uses 83,989 tokens; Unigram uses 84,818 tokens for
  106,087 characters. These are compatibility statistics, not recognition scores.
- Two real samples per train/validation split, WD/WI and tokenizer: all eight
  CPU forward batches output 224 classes and finite CTC loss. No optimizer step,
  checkpoint or recognition result was created.

The frozen compatibility report is
`artifacts/tokenizers/ed_iam_handwriting_subwords_v1.audit.json`.
The existing audit CLI can also recheck either adapter:

```bash
python -m tva.handwriting.audit --tokenizer-kind bpe --adapter artifacts/tokenizers/ed_iam_handwriting_bpe_v1.json
python -m tva.handwriting.audit --tokenizer-kind unigram --adapter artifacts/tokenizers/ed_iam_handwriting_unigram_v1.json
```
