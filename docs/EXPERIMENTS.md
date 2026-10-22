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

## 2026-10-08 — Handwriting BPE ED WD fold 0 completed

The user manually ran `python main.py -c configs/thesis/ed/bpe_handwriting_wd.yaml`.
Verified the supplied best metrics against the saved 300-epoch run (indices
0–299), run tag `20261008183405`, under
`results/thesis/ed/handwriting_bpe_wd/0/`. The log records CUDA mixed precision,
ED `gold_wd.zip` from `/mnt/c/Users/Ali/Downloads/fau-english-dataset/`,
and `HandwritingBPETokenizer` loading the frozen ED BPE adapter. The saved
configuration exactly matches the declared fold-0 config. Model and training
settings match the completed handwriting bigram run; its saved config retains
historical FAU dataset/tokenizer/output names from before the ED rename. Both
logs identify the same WD ZIP archive. All 300 prediction epochs are present,
with 503 validation labels per epoch.

The source release and derived adapter remain the versions frozen before the
run: IAM BPE SHA-256
`a504c2306200faf6eab98f3feef6f98b81e6e6c6c7f707cf31aa3399a660be47`,
ED adapter SHA-256
`fcf244d112f55fc7e30edde4eaaabda70d282ea193b7834acbdcdac6cbc4c4ea`.
No vocabulary or inference policy was changed after seeing recognition results.

### Independent best validation metrics

| WD fold-0 method | Best CER (%) | CER epoch | Best WER (%) | WER epoch |
| --- | --- | --- | --- | --- |
| Character | 15.83313693 | 241 | 54.64803869 | 199 |
| Handwriting bigram | 24.27056328 | 206 | 64.80386889 | 245 |
| Handwriting BPE | 24.45439547 | 185 | 63.40677055 | 261 |
| Original TVA linguistic bigram | 35.95569173 | 191 | 79.23159592 | 224 |

Epochs are zero-based saved indices. Each metric is its own minimum over the
300 epochs; these columns do not describe one common checkpoint. For BPE,
best CER is 0.24454395474899834 at epoch 185, while best WER is
0.6340677055346587 at epoch 261. WER at the best-CER checkpoint is
64.88447071%; CER at the best-WER checkpoint is 24.62408673%. Use the
independent minima when comparing with previous tables; evaluating
`best_cer.pth` will not reproduce the independent minimum WER.

BPE has 0.18383219 percentage points higher CER and 1.39709833 points lower WER
than handwriting bigram. The character baseline still has lower CER and WER.
These are one fold and seed; neither the small CER difference nor the WER
improvement establishes a general ranking. BPE WI and Unigram WD/WI remain
pending in this result report; their manual runs are the next comparison step.

### Saved evidence hashes

All paths below are relative to `results/thesis/ed/handwriting_bpe_wd/0/`.

| File | SHA-256 |
| --- | --- |
| `train_20261008183405.yaml` | `d67db550f66c34f65c6b40de50270e9c5d5f3a74b8b026b878c06bebccc29100` |
| `train_20261008183405.json` | `0b18d45c9d388f4d48d54aa05d9e0f0125797f608381c88b69e1b204a0ba0284` |
| `train_20261008183405.log` | `4b0bb6e8f2a46ac1820a870a6f9231b969e65142c285e85f482fde32ee7f2d4d` |
| `train_preds_20261008183405.json` | `d95fd65bf64f76ef010f0b4bd73321855ba3db2bff09dcb76e8bbab0afc5169c` |
| `checkpoints/best_cer.pth` | `a7ef4e1845f50b39028287d7fd8d6dcacbedb1ba33f54ce91f7c4da89680403f` |
| `checkpoints/best_wer.pth` | `a4e8ab5e32be3bf63e5351e2787293824c1faf1b524edf390671cf255b5663a8` |

## 2026-10-08 — Handwriting BPE ED WI fold 0 completed

Verified the user's reported result against run `20261008194420` under
`results/thesis/ed/handwriting_bpe_wi/0/`. All 300 epochs (0–299) and their
prediction entries are present, with 525 WI validation labels per epoch.
The saved YAML exactly matches `configs/thesis/ed/bpe_handwriting_wi.yaml`.
Model/training hyperparameters match the completed handwriting bigram WI run;
its saved dataset names predate the ED metadata rename. The BPE log confirms
CUDA mixed precision, the frozen ED BPE tokenizer and `gold_wi.zip` from
`/mnt/c/Users/Ali/Downloads/fau-english-dataset/`.

The ED BPE adapter remains byte-identical to its pre-evaluation freeze
(SHA-256 `fcf244d112f55fc7e30edde4eaaabda70d282ea193b7834acbdcdac6cbc4c4ea`).
No tokenizer selection or fitting used these recognition results.

### Independent best validation metrics

| WI fold-0 method | Best CER (%) | CER epoch | Best WER (%) | WER epoch |
| --- | --- | --- | --- | --- |
| Character | 13.77171782 | 276 | 42.30576441 | 285 |
| Handwriting bigram | 16.42117744 | 264 | 44.83709273 | 265 |
| Handwriting BPE | 17.17816590 | 248 | 44.73684211 | 257 |
| Original TVA linguistic bigram | 20.93118701 | 250 | 52.43107769 | 258 |

Epochs are zero-based saved indices. BPE's best CER is
0.17178165899037803, first recorded at epoch 248; best WER is
0.4473684210526316 at epoch 257. WER at the saved best-CER epoch is
45.11278195%. Unlike BPE WD, the best-WER epoch also ties the minimum CER
(17.17816590%), so epoch 257 achieves both reported minima.

Relative to handwriting bigram WI fold 0, BPE has 0.75698846 percentage
points higher CER and 0.10025063 points lower WER. Both WD and WI fold-0
results show higher BPE CER and lower BPE WER than handwriting bigram; the
WI WER difference is only 0.10 points. Character remains lower on both metrics.
These are descriptive one-fold/one-seed comparisons, not an established
cross-fold ranking. Handwriting BPE fold 0 is now complete for WD and WI;
Unigram WD/WI is the next planned comparison.

### Saved evidence hashes

Paths are relative to `results/thesis/ed/handwriting_bpe_wi/0/`.

| File | SHA-256 |
| --- | --- |
| `train_20261008194420.yaml` | `cc5ad8c2efa032a6b6480532e4171667a8b8330d1278f4add12e50128fca287d` |
| `train_20261008194420.json` | `c8274e380950002537ba5abc0a7a3703696efc452500feaa31661f3edad7afd9` |
| `train_20261008194420.log` | `25d09a908a3d96be86e0090c9930c265bba6aec4424ef22712dc0d12ba4eb853` |
| `train_preds_20261008194420.json` | `9cef0c114f51959002efdc530acc1fada2c53cf10655b53fb4ced64fd8830151` |
| `checkpoints/best_cer.pth` | `334ba257df029a043fd3bc13caad34e026edeaf71767d43e041b5e95516a4c79` |
| `checkpoints/best_wer.pth` | `4eb407159482ac5c65530a6cffd53dd4fceac9a3877c6a9076b70a1da2c72b2e` |

## 2026-10-08 — Handwriting Unigram ED WD fold 0 completed

Verified the user's reported best values against run `20261008210048` under
`results/thesis/ed/handwriting_unigram_wd/0/`. All 300 epochs (0–299) and
prediction entries are present, with 503 WD validation labels per epoch.
The saved config exactly matches `configs/thesis/ed/unigram_handwriting_wd.yaml`
and matches the completed handwriting BPE WD model/training settings except
for tokenizer/artifact/output paths. The log confirms CUDA mixed precision,
`HandwritingUnigramTokenizer` and ED `gold_wd.zip` from
`/mnt/c/Users/Ali/Downloads/fau-english-dataset/`.

The ED Unigram adapter remains byte-identical to its pre-evaluation freeze:
SHA-256 `e1628c0d2c30cdc9d05898a848c80bfb055d282cf78d39aa27d6d1a3cf8471f0`.
Its 145 IAM pieces, learned scores and declared forced-symbol policy were
not changed after recognition evaluation.

### Independent best validation metrics

| WD fold-0 method | Best CER (%) | CER epoch | Best WER (%) | WER epoch |
| --- | --- | --- | --- | --- |
| Character | 15.83313693 | 241 | 54.64803869 | 199 |
| Handwriting bigram | 24.27056328 | 206 | 64.80386889 | 245 |
| Handwriting BPE | 24.45439547 | 185 | 63.40677055 | 261 |
| Handwriting Unigram | 24.06787650 | 265 | 63.78291241 | 229 |
| Original TVA linguistic bigram | 35.95569173 | 191 | 79.23159592 | 224 |

Epochs are zero-based saved indices. Unigram's minimum CER is
0.24067876502474664 at epoch 265 and minimum WER is
0.6378291241268136 at epoch 229. These are separate selected checkpoints.
WER at the best-CER epoch is 64.13218700%;
CER at the best-WER epoch is 24.27056328%.

Against handwriting bigram, Unigram has 0.20268678
percentage points lower CER and 1.02095648 points
lower WER. Against handwriting BPE, its CER is
0.38651897 points lower and WER is
0.37614186 points higher. Among the three
handwriting methods, Unigram has the lowest CER and BPE the lowest WER on this
fold; character has lower error on both metrics. These are one-fold/one-seed
results, not an established cross-fold ranking. Unigram WI is next, followed
by freezing and integrating the matched text-only controls discussed with the user.

### Saved evidence hashes

Paths are relative to `results/thesis/ed/handwriting_unigram_wd/0/`.

| File | SHA-256 |
| --- | --- |
| `train_20261008210048.yaml` | `bf2f97f98cd5b552db58bcf40631fa6a5370063f909f922c399f76a1e3af9ff7` |
| `train_20261008210048.json` | `982301735479486e71727b3615c9db1c5c39c16731b0d142fb9de063b4bd497e` |
| `train_20261008210048.log` | `28d878228d48f79b3c34dac489c3c2f1239789ccf67c1300c2daea8ed5da1f73` |
| `train_preds_20261008210048.json` | `4755fddb5a72822a49151f16b813e94afb864361e17ce7a049fe356f47cebce2` |
| `checkpoints/best_cer.pth` | `a73137e379214bd50ba423b98c8e16f50caf73eacc24bc6c70799861df4795bf` |
| `checkpoints/best_wer.pth` | `7579b498c555a3598954e792c6b26027461b935a1bda37ee4dd01746fa0133e8` |

## 2026-10-08 — Handwriting Unigram ED WI fold 0 completed

Verified the user's reported best values against run `20261008221323` under
`results/thesis/ed/handwriting_unigram_wi/0/`. All 300 epochs (0–299) and their
prediction entries are present, with 525 WI validation labels per epoch.
The saved YAML exactly matches `configs/thesis/ed/unigram_handwriting_wi.yaml`
and matches the completed BPE WI model/training settings except tokenizer,
artifact and output paths. The log confirms CUDA mixed precision,
`HandwritingUnigramTokenizer` and ED `gold_wi.zip` from
`/mnt/c/Users/Ali/Downloads/fau-english-dataset/`.

The Unigram adapter remains the pre-evaluation frozen artifact, SHA-256
`e1628c0d2c30cdc9d05898a848c80bfb055d282cf78d39aa27d6d1a3cf8471f0`.
No pieces, learned scores or forced-symbol policy changed after evaluation.

### Independent best WI validation metrics

| WI fold-0 method | Best CER (%) | CER epoch | Best WER (%) | WER epoch |
| --- | --- | --- | --- | --- |
| Character | 13.77171782 | 276 | 42.30576441 | 285 |
| Handwriting bigram | 16.42117744 | 264 | 44.83709273 | 265 |
| Handwriting BPE | 17.17816590 | 248 | 44.73684211 | 257 |
| Handwriting Unigram | 16.84071321 | 277 | 45.41353383 | 261 |
| Original TVA linguistic bigram | 20.93118701 | 250 | 52.43107769 | 258 |

Epochs are zero-based saved indices. Unigram's best CER is
0.16840713210816727 at epoch 277 and best WER is
0.45413533834586467 at epoch 261, from separate selected checkpoints.
WER at the best-CER epoch is 45.96491228%;
CER at the best-WER epoch is 16.95471750%.

Compared with handwriting bigram WI, Unigram CER is
0.41953577 percentage points higher and WER is
0.57644110 points higher. Compared with handwriting
BPE WI, Unigram CER is 0.33745269 points lower
and WER is 0.67669173 points higher.

### Complete current fold-0 comparison

| Method | WD CER (%) | WD WER (%) | WI CER (%) | WI WER (%) |
| --- | --- | --- | --- | --- |
| Character | 15.83313693 | 54.64803869 | 13.77171782 | 42.30576441 |
| Handwriting bigram | 24.27056328 | 64.80386889 | 16.42117744 | 44.83709273 |
| Handwriting BPE | 24.45439547 | 63.40677055 | 17.17816590 | 44.73684211 |
| Handwriting Unigram | 24.06787650 | 63.78291241 | 16.84071321 | 45.41353383 |
| Original TVA linguistic bigram | 35.95569173 | 79.23159592 | 20.93118701 | 52.43107769 |

Character has the lowest CER and WER in both distributions among these current
conditions. Among handwriting methods, Unigram has lowest WD CER, bigram has
lowest WI CER, and BPE has lowest WER in both. These are independent minima
on one fold and one seed, not a cross-fold or statistical ranking. Original TVA
linguistic bigram is an existing implementation baseline; it is not the matched
text-only BPE/Unigram control discussed with the user. All four new handwriting
BPE/Unigram WD/WI fold-0 runs are complete. Next planned work is to freeze and
integrate the IAM text-only BPE/Unigram controls for the matched comparison.

### Saved evidence hashes

Paths are relative to `results/thesis/ed/handwriting_unigram_wi/0/`.

| File | SHA-256 |
| --- | --- |
| `train_20261008221323.yaml` | `b844cb38dc2847fb15a31f40b0c39338836e7223b3f2506abce83d1e28f7a9ea` |
| `train_20261008221323.json` | `163bf0ffb94a823fa63f6f4e90bc9eba9241cc4af224ffee1cceffaa20d3292a` |
| `train_20261008221323.log` | `eba1613c8c95359cb1666b46f422e9df0f35bd4ef6f0f8f31b562f22acfcaeef` |
| `train_preds_20261008221323.json` | `d02fb41fbda62e33ec248eeb07a0cecabc056ef413a179f58ffad9172a25eff8` |
| `checkpoints/best_cer.pth` | `5a6f503afced4f4ef6974ce28a529eb6502e4df1efa8f117abaa84e64f7a27e2` |
| `checkpoints/best_wer.pth` | `e35eaded1f414b56ecead62f3697d7d692c56dc747f5c7f7469bc35ce5913016` |


## 2026-10-10 — Tokenizer artifact directories organized

Current ED handwriting vocabularies and their bundled IAM sources are now under
`artifacts/tokenizers/ed/handwriting/`; the original TVA IAM linguistic bigram
and its supporting reports are under `artifacts/tokenizers/ed/linguistic/`.
Historical custom ED and OnHW vocabularies are retained under
`artifacts/tokenizers/archive/`. See `artifacts/tokenizers/README.md` for the index.

All 17 frozen artifact files were moved without changing their bytes or hashes.
Active configurations, preparation notebooks, and audit defaults use the new
paths. The tokenizer resolver supports previous repository paths recorded in
saved recognition configurations. Historical experiment entries and frozen
reports retain their original paths. Recognition results and running character
baseline outputs are not moved; no recognition training is launched.

Validation: the CPU unittest suite ran 71 tests (64 passed, 7 optional skips).
Saved-path regression checks load every tokenizer condition with identical
encodings, and unrelated missing files still fail. All ED preparation notebook
code cells reproduced their existing files at the new locations; all 17
artifact hashes remained unchanged. Artifact index links were checked.


## 2026-10-10 — Handwriting tokenizer training moved into TVA

DTLR now supplies a compact, authenticated IAM training connectivity export.
TVA owns the selection/fitting algorithms and the current ED tokenizer training
APIs. `GreedyHandwritingBigramTokenizer.train()` selects from aggregate scores;
`HandwritingBPETokenizer.train()` collects eligible spans and learns supported
ordered merges; `HandwritingUnigramTokenizer.train()` collects candidates,
initializes from connected counts, and runs exact EM/removal-loss pruning.
The original DTLR algorithms, thresholds, length cap, 145-piece budget,
deterministic ordering, EM settings, and greedy/BPE/Viterbi ED inference policies
are preserved. Shared text-only trainer functions moved with this code, but
matched text-only ED artifacts/configs are not added by this migration.

The evidence input is `artifacts/evidence/iam/iam_connectivity_v1.json.gz`:
5,694 IAM train transcriptions, 160,536 non-whitespace pair observations, and
1,259 aggregate score rows. It retains exact usable component IDs and explicit
abstentions, without detector boxes/images or vocabulary selection. Compressed
SHA-256: `815304d2591bc187a2eea7dd76593499905c11645c8b9a146e7f1699c0e636fa`;
decompressed SHA-256:
`1b44fddd0fa76dbb4ed14517f4361c522a4415139d16307b3468e14d122b8d0c`.
DTLR authenticates all five original input exports and reconstructs aggregate
scores; TVA validates bundle hashes, split/line identities, transcription
hashes, text offsets, coverage, component semantics, and aggregate agreement.

`train_ed_tokenizers.ipynb` retains the categories/imports/`.train()` structure
and supplies this evidence path to all three handwriting methods. `.train()`
now actually recomputes candidates and learned parameters. `.prepare()` names
the previous frozen IAM-to-ED projection operation; `.load()` reuses a saved
file. No ED transcription labels enter selection or fitting. No recognition
training is launched, and recognition configs/checkpoints/results are unchanged.

The current v1 training wrapper checks recomputed learned fields against the
existing frozen experiment and retains its historical release metadata only
after exact reproduction. New `.training.json` reports alongside the three
ED handwriting vocabularies record TVA implementation/evidence hashes and
reproduction checks separately from the historical DTLR provenance. Different
corpora/policies/results require a separate version rather than replacing v1.
Original `tva/tokenizers.py` remains byte-identical to the upstream initial
commit; the linguistic IAM bigram trainer still calls its original algorithm.

Validation: the complete TVA suite ran 93 tests (86 passed, 7 optional skips),
including mathematical EM/Viterbi/removal-loss checks, connectivity-dependent
selection, text-only training without image edges, input corruption/leakage
rejection, all three full public fitting APIs, and saved-config compatibility.
All three newly fitted native models and ED projections reproduce their frozen
files byte for byte, including BPE merge order, Unigram scores/training curve,
vocabulary evidence, and IDs. Every ED notebook code cell executed successfully;
all 17 pre-existing artifact checksums stayed unchanged. DTLR's 78 tests pass,
and its exporter independently regenerated identical bundle bytes without
importing TVA. DTLR's former training CLIs point to TVA; historical Python APIs
use lazy compatibility wrappers rather than duplicate trainer implementations.
Frozen DTLR inference, prior snapshots, provenance, and evidence generation
remain available. Neither repository is committed or pushed by this change.

### 2026-10-10 — Separate linguistic helpers from handwriting tokenizers

Moved authenticated IAM input preparation and `IAMBigramTokenizer` to
`tva/iam_tokenizers.py`. It still delegates vocabulary selection and text
encoding to the unchanged original `tva/tokenizers.py` implementation.
Moved the load-only historical OnHW and ED custom linguistic classes and their
frozen identities to `tva/tokenizer_compatibility.py`; saved experiment keys
and artifacts remain supported. `tva/tokenizer_factory.py` now supplies the
shared factory, artifact path resolution, and explicit character alphabet
initialization for recognition and analysis. Updated entry points, tests, ED
notebooks, and README imports. `tva/handwriting_tokenizers.py` now contains only
handwriting-aware tokenizer classes. Tokenizer files and training algorithms
are unchanged; no recognition experiments were launched.

Validation: TVA unittest suite passed (93 tests: 86 passed, 7 optional checks
skipped). Verified all 17 frozen tokenizer artifacts remain byte-identical
and `tva/tokenizers.py` matches the original TVA commit `cff221c`. Updated
Python modules and ED notebook cells parse successfully.

## 2026-10-11 — OnHW WI/RH character baseline: all five folds completed

Verified the user's completed fresh workstation run under
`results/thesis/baselines/B0_char_wi_rh_cv_workstation/`. These are the
OnHW-Words500 right-handed writer-independent character baselines, separate
from the ED experiments. Each fold completed all 300 epochs (0–299), with
300 complete saved prediction sets and retained best-CER, best-WER and latest
checkpoints. Independently recomputed the selected-epoch CER/WER from saved
predictions and confirmed validation labels exactly match dataset annotations.

All five saved configs differ only in `idx_fold`: BLConv-B + BiLSTM-B,
13 channels at 100 Hz, 60 character classes including blank, batch 64,
300 epochs, learning rate 0.001, 30-epoch warmup, augmentation enabled,
no concatenation, seed 42, full datasets, and CUDA mixed precision. Each starts
with `checkpoint: null`; the CV launcher runs a fresh Python process per fold.
Train/validation writer sets are disjoint in every fold. Core saved settings
also match the earlier OnHW WI fold-0 baseline.

### Independent best validation metrics

| Fold | Run ID | Train samples | Validation samples | Best CER (%) | CER epoch | Best WER (%) | WER epoch |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `20261010155328` | 19907 | 5292 | 15.60411400 | 244 | 27.94784580 | 269 |
| 1 | `20261010173703` | 20078 | 5121 | 1.80383719 | 271 | 6.95176723 | 260 |
| 2 | `20261010192827` | 20202 | 4997 | 2.10158263 | 268 | 7.10426256 | 270 |
| 3 | `20261010211425` | 20520 | 4679 | 13.90545397 | 251 | 24.42829664 | 251 |
| 4 | `20261010230222` | 20089 | 5110 | 3.11972109 | 241 | 9.80430528 | 237 |
| **Unweighted mean** | — | — | — | **7.30694178** | — | **15.24729550** | — |
| **Fold standard deviation (population, ddof=0)** | — | — | — | 6.12039494 | — | 9.05920170 | — |

Epochs are zero-based. Best CER and best WER are independent validation minima;
they generally correspond to different checkpoints. Using the best-CER
checkpoint for both metrics instead gives mean CER 7.30694178% and mean
WER 15.38229305%. The final-epoch means are CER
7.49877830% and WER
15.56679715%.

### Paper comparison

[TVA paper, Appendix B Table 3](https://arxiv.org/html/2603.16883v1)
reports the WI character/no-concatenation baseline at **7.41% CER / 15.40% WER**.
Our unweighted fold means are lower by 0.10305822 CER percentage points
and 0.15270450 WER points. WER averaged at the best-CER checkpoint is
15.38229305%, also close to the reported 15.40%.

This is a close aggregate reproduction and supports consistency of the WI
baseline setup. It does not establish byte-identical runs or environment
identity; the paper does not provide its per-fold results or exact checkpoint
selection rule here. Large fold variability is retained in the report rather
than inferred to be a pipeline failure. Complete the corresponding five-fold
WD character baseline to assess that protocol separately. No new recognition
training was launched and no tokenizer files were changed during verification.

### Saved evidence hashes

Hashes cover saved configs, epoch metrics, logs and complete prediction files;
paths are repository-relative. Checkpoint files were confirmed present but
were not reserialized or modified.

| File | SHA-256 |
| --- | --- |
| `results/thesis/baselines/B0_char_wi_rh_cv_workstation/0/train_20261010155328.json` | `1524751085ec78526d4309d42b0d4e79cc56274e86c0ef63e88c2972abda3aff` |
| `results/thesis/baselines/B0_char_wi_rh_cv_workstation/0/train_20261010155328.log` | `ecbf881ba387f5332b4e2eb1ec42c68124d5c65780cb7dd584bf63ef7d8238a9` |
| `results/thesis/baselines/B0_char_wi_rh_cv_workstation/0/train_20261010155328.yaml` | `268c94be26222e8b4e605734fa0f764a9816d4c9d7f7994f9d13fd68babfe7c4` |
| `results/thesis/baselines/B0_char_wi_rh_cv_workstation/0/train_preds_20261010155328.json` | `0106bb57a1c03a227ba0836edf2517172644a92f340bf2bdbc6d37710db85f5f` |
| `results/thesis/baselines/B0_char_wi_rh_cv_workstation/1/train_20261010173703.json` | `60ad42438f862d76da108e9076d84416eb594cf5efde1709d13edfffe517020b` |
| `results/thesis/baselines/B0_char_wi_rh_cv_workstation/1/train_20261010173703.log` | `26b94cf70f78246f79fe41f0b69c2ff24344556e2ea8bdaacce9b8d33d0c6b09` |
| `results/thesis/baselines/B0_char_wi_rh_cv_workstation/1/train_20261010173703.yaml` | `aade00e5709eb4a11ab9cd31dddf4b71ca54658eace31e6a483e209d23186b9c` |
| `results/thesis/baselines/B0_char_wi_rh_cv_workstation/1/train_preds_20261010173703.json` | `e0699dbb473e5bb9a7f75b19e3f33319300b8a3d5d6ea36989744fb42a97f30a` |
| `results/thesis/baselines/B0_char_wi_rh_cv_workstation/2/train_20261010192827.json` | `48921c0e868d08dc5dad8ed444a83dfabf8aceacdb3e5b279510941b5e46b093` |
| `results/thesis/baselines/B0_char_wi_rh_cv_workstation/2/train_20261010192827.log` | `11a09ce945a062fc0666863cedef48a4479ba6ce0eab352766fcb5290606275e` |
| `results/thesis/baselines/B0_char_wi_rh_cv_workstation/2/train_20261010192827.yaml` | `3e37a3d0ce4d50142e765e3826fb90cf30bb0d83a10aa3a4b8ae1c832274fd1f` |
| `results/thesis/baselines/B0_char_wi_rh_cv_workstation/2/train_preds_20261010192827.json` | `5e0960096cedb9ea07d8a446125ec6df977da42cd741d57839b8d8addb50075b` |
| `results/thesis/baselines/B0_char_wi_rh_cv_workstation/3/train_20261010211425.json` | `640079e1e150b3eaa566719c99e642ae59bd2b64f43b0a94ae138f1b23ddfac9` |
| `results/thesis/baselines/B0_char_wi_rh_cv_workstation/3/train_20261010211425.log` | `2d7304d82dd14910a893cf7553b9ba709041404ead691e06be1710e08a537b02` |
| `results/thesis/baselines/B0_char_wi_rh_cv_workstation/3/train_20261010211425.yaml` | `9138e7c323a4445f65e321586adc2dea2faa8664376894c819a3bae32159aa70` |
| `results/thesis/baselines/B0_char_wi_rh_cv_workstation/3/train_preds_20261010211425.json` | `1fa8ce646ee483425ae5b65b4960314d517def633aa0c9e2f0944c80ad259f08` |
| `results/thesis/baselines/B0_char_wi_rh_cv_workstation/4/train_20261010230222.json` | `6ea609f78c4029494e796f1f8460acf9c40bdb21fcf73f03dcddfb7b8934d7b9` |
| `results/thesis/baselines/B0_char_wi_rh_cv_workstation/4/train_20261010230222.log` | `fb97f611f3c57fbfb8252eea9db7d1d90a195a18a1438668124c85551514fb61` |
| `results/thesis/baselines/B0_char_wi_rh_cv_workstation/4/train_20261010230222.yaml` | `d0ddc54ce3c8e03638a60545b7b7a40548b1369aebf09af180af4602220e0016` |
| `results/thesis/baselines/B0_char_wi_rh_cv_workstation/4/train_preds_20261010230222.json` | `2df9a35eb50e23b0e7217d8b6f03af48c2bc53a5d54e83e12ae795034d8a0510` |

## 2026-10-11 — OnHW WD/WI character baseline: combined five-fold results

Both right-handed OnHW-Words500 character-baseline CV runs are complete under
`results/thesis/baselines/B0_char_wi_rh_cv_workstation/` and
`results/thesis/baselines/B0_char_wd_rh_cv_workstation/`. All ten folds contain
300 consecutive epochs (0–299). Each saved configuration matches its CV YAML
after substituting the fold index. Best-CER and latest checkpoints are present
for every fold. These results are separate from the ED experiments.

The launch configs are `configs/thesis/b0_char_wi_rh_cv.yaml` and
`configs/thesis/b0_char_wd_rh_cv.yaml`. Shared settings are BLConv-B + BiLSTM-B,
13 input channels, 60 character classes including CTC blank, batch size 64,
300 epochs, learning rate 0.001, 30-epoch warmup, standard augmentation enabled,
no concatenation (`num_concat: 0`), seed 42, and fresh initialization
(`checkpoint: null`). The configs differ only in dataset/tokenizer/output paths.

### Paired metrics at the best-CER epoch

For each fold, select the lowest validation CER and report both CER and WER
from that same epoch, following `evaluate.py:get_mean_std_cv`. Epoch indices
are zero-based. Aggregate values are equal-weight means over five folds;
standard deviations use the population convention (`ddof=0`). This convention
is explicit and does not assume the paper disclosed the same selection rule.

| Split | Fold | Run ID | Selected epoch | Validation samples | CER (%) | WER (%) |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| WI | 0 | `20261010155328` | 244 | 5292 | 15.60411400 | 28.28798186 |
| WI | 1 | `20261010173703` | 271 | 5121 | 1.80383719 | 6.97129467 |
| WI | 2 | `20261010192827` | 268 | 4997 | 2.10158263 | 7.20432259 |
| WI | 3 | `20261010211425` | 251 | 4679 | 13.90545397 | 24.42829664 |
| WI | 4 | `20261010230222` | 241 | 5110 | 3.11972109 | 10.01956947 |
| WD | 0 | `20261011113536` | 259 | 5036 | 12.48992183 | 35.56393963 |
| WD | 1 | `20261011131746` | 271 | 5039 | 13.03032340 | 45.24707283 |
| WD | 2 | `20261011145904` | 233 | 5039 | 21.29301274 | 57.43203016 |
| WD | 3 | `20261011164301` | 258 | 5041 | 14.34185472 | 41.73775045 |
| WD | 4 | `20261011182730` | 286 | 5038 | 12.54794328 | 41.98094482 |

### Aggregate comparison with the paper

[TVA paper, Appendix B Table 3](https://arxiv.org/html/2603.16883v1)
reports the character baseline without concatenation (C0). Values below are
percentages; differences are ours minus paper in percentage points.

| Split | Our CER mean ± SD | Paper CER | CER difference | Our WER mean ± SD | Paper WER | WER difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| WI | 7.30694178 ± 6.12039494 | 7.41 | -0.10305822 | 15.38229305 ± 9.10790737 | 15.40 | -0.01770695 |
| WD | 14.74061119 ± 3.34339505 | 14.86 | -0.11938881 | 44.39234758 ± 7.23279217 | 45.10 | -0.70765242 |

The initial WI fold-0 discrepancy is resolved at the five-fold aggregate:
WI folds 0 and 3 have substantially higher errors than folds 1, 2 and 4.
Our WI mean is 7.31% CER / 15.38% WER versus 7.41% / 15.40% in the paper;
WD is 14.74% / 44.39% versus 14.86% / 45.10%. This supports a close aggregate
reproduction of both character baselines; it does not establish identical
dataset bytes, runtime environments, or per-fold paper results. The earlier
WI-only entry remains a historical record; its request to complete WD is now
satisfied by this entry.

Independent best-WER minima remain separate: their five-fold means are
15.24729550% for WI and 44.26136447% for WD. In WI fold 0, the independent
27.94784580% WER occurs at epoch 269, whereas the paired best-CER result is
28.28798186% WER at epoch 244. Do not mix these selection conventions.

### Verification and saved evidence

All ten paired CER/WER values were independently recalculated using `jiwer`
from the saved validation labels and selected-epoch predictions and matched
the recorded metrics within 1e-12 (fractional error rates). Original configs,
logs, prediction files and checkpoints were preserved. No recognition training
was launched and no tokenizer or training configuration was changed.

The machine-readable summary is
`results/thesis/baselines/onhw_char_cv_summary_20261011.json`. It records
selected epochs, sample counts, independent WER minima, aggregate statistics,
paper differences, and the original metrics paths and SHA-256 hashes.
WI evidence hashes are retained in the preceding WI-only section. Additional
WD evidence hashes are recorded below; all paths are repository-relative.

| File | SHA-256 |
| --- | --- |
| `results/thesis/baselines/B0_char_wd_rh_cv_workstation/0/train_20261011113536.json` | `f7a43fbe9ce981bcddab9f2338731aa72e3ccdeb42362a665a17eea26dd56213` |
| `results/thesis/baselines/B0_char_wd_rh_cv_workstation/0/train_20261011113536.log` | `2d2de3dd3703456cc399137794cdcbed2fc404db52e6289233d50a62e23d6f03` |
| `results/thesis/baselines/B0_char_wd_rh_cv_workstation/0/train_20261011113536.yaml` | `55195b4733736079adef2620f8ca8bb2be5fdac240b2be2cefcc9bcd78d9c795` |
| `results/thesis/baselines/B0_char_wd_rh_cv_workstation/0/train_preds_20261011113536.json` | `29d259436c08288da3e3216ef7002738bca78b856416b5926708d561d296fc52` |
| `results/thesis/baselines/B0_char_wd_rh_cv_workstation/1/train_20261011131746.json` | `20f2ce04abc69c5d0c3c82fbc33715f5a94ee875736e15ed8f34ac3322388dbd` |
| `results/thesis/baselines/B0_char_wd_rh_cv_workstation/1/train_20261011131746.log` | `89906eec182d7f59dd3a6f518868dd4b8f25a8a50cf7bce0e8e5efd0ecd6bdba` |
| `results/thesis/baselines/B0_char_wd_rh_cv_workstation/1/train_20261011131746.yaml` | `d779dcb59d6ec4e20592b76d86377486a4b5893b9882b4beb83b9876cd1fbd4b` |
| `results/thesis/baselines/B0_char_wd_rh_cv_workstation/1/train_preds_20261011131746.json` | `5adaf192aba8dd7ce5d06f1a0046b51591d64090fdf45c3adeb286f17c7012ec` |
| `results/thesis/baselines/B0_char_wd_rh_cv_workstation/2/train_20261011145904.json` | `0f8c8f347f9782e3babefc0f09d1d75a0f8b0d739fc1c0d237faf17e65a5abd6` |
| `results/thesis/baselines/B0_char_wd_rh_cv_workstation/2/train_20261011145904.log` | `bc2f42d59abcd8d3bf902469f0acf304aaf62aaff89fbd45bf10568cfdb958a8` |
| `results/thesis/baselines/B0_char_wd_rh_cv_workstation/2/train_20261011145904.yaml` | `9cc619fd58ccd4ffa5967bd6449fc8c542de8495422a5dc7ab7c7a3d536ab8fd` |
| `results/thesis/baselines/B0_char_wd_rh_cv_workstation/2/train_preds_20261011145904.json` | `de19df31f0fbf7c0dd690814186ea2e6fdc8b342e9cecd5121bc2b280427eef4` |
| `results/thesis/baselines/B0_char_wd_rh_cv_workstation/3/train_20261011164301.json` | `eb63a4b471ea6aa0301c3f5a1b94167ee2e3be9fe64d47d30f47d6f95c464e25` |
| `results/thesis/baselines/B0_char_wd_rh_cv_workstation/3/train_20261011164301.log` | `4706b31df5e627de38517f4e985275fad58494e251c9516c3c506828cb04b903` |
| `results/thesis/baselines/B0_char_wd_rh_cv_workstation/3/train_20261011164301.yaml` | `28c9a9786dd297a7667d78ec07ccf7bef832d4ed0f2a26c92d0566158264316b` |
| `results/thesis/baselines/B0_char_wd_rh_cv_workstation/3/train_preds_20261011164301.json` | `b04ffe14d892e4844e032cd4ea35b52f90039d0c8fcc89a9fd328783b65d2e5f` |
| `results/thesis/baselines/B0_char_wd_rh_cv_workstation/4/train_20261011182730.json` | `9099defc7d6c1d193a9d0e7e9c928eeb841f9a3b3bd031a3b99bf70ef414217d` |
| `results/thesis/baselines/B0_char_wd_rh_cv_workstation/4/train_20261011182730.log` | `1a3851d77584fdee09e511ac4f79c359ac54aa218bc63f1e456558c08fd106e1` |
| `results/thesis/baselines/B0_char_wd_rh_cv_workstation/4/train_20261011182730.yaml` | `50879d55b65ea4402fa90c65ea8ead19963c644404c4bdb94249aef38d4a2553` |
| `results/thesis/baselines/B0_char_wd_rh_cv_workstation/4/train_preds_20261011182730.json` | `d501401c720b75f18bc6d885130440e40446bd60892c54f3b67dda09b7d2b613` |

## 2026-10-17 — English ED character baseline: fresh WD/WI five-fold reruns completed

All ten fresh character-baseline rerun folds completed 300 consecutive epochs
(0–299). These results use only `character_wi_cv_rerun` and
`character_wd_cv_rerun` under `results/thesis/ed/`; the historical ED fold-0
runs and the OnHW experiments remain separate.

Launch configs: `configs/thesis/ed/character_wi_cv_rerun.yaml` and
`configs/thesis/ed/character_wd_cv_rerun.yaml`. Every saved fold config matches
its launch config after substituting the fold index. Protocol: BLConv-B +
BiLSTM-B, seven input channels, fixed ED alphabet (79 classes including blank),
batch 32, eight DataLoader workers, 300 epochs, 30 warmup epochs, learning rate
0.001, standard augmentation enabled, no concatenation, cache enabled, seed 42,
CUDA, and fresh initialization (`checkpoint: null`).

### Validation results at each fold's best-CER epoch

Both metrics in each row come from the same selected epoch, following
`evaluate.py:get_mean_std_cv`. Epochs are zero-based. Means give equal weight
to each fold; standard deviations use the population convention (`ddof=0`).

| Split | Fold | Run ID | Selected epoch | Train samples | Validation samples | CER (%) | WER (%) |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| WI | 0 | `20261015232734` | 266 | 2025 | 525 | 13.70787542 | 42.73182957 |
| WI | 1 | `20261016003749` | 287 | 2025 | 525 | 15.50582751 | 45.19083969 |
| WI | 2 | `20261016025635` | 291 | 2050 | 500 | 10.87687716 | 40.18858308 |
| WI | 3 | `20261016131221` | 299 | 2050 | 500 | 9.72188516 | 35.87301587 |
| WI | 4 | `20261016141958` | 255 | 2050 | 500 | 16.04115070 | 46.04519774 |
| WD | 0 | `20261016152921` | 247 | 2047 | 503 | 15.84727787 | 55.40032241 |
| WD | 1 | `20261016164333` | 254 | 2039 | 511 | 15.57899784 | 52.47944181 |
| WD | 2 | `20261016175116` | 264 | 2036 | 514 | 15.74721818 | 53.21009243 |
| WD | 3 | `20261016185730` | 205 | 2019 | 531 | 16.74362308 | 56.37960478 |
| WD | 4 | `20261016200528` | 233 | 2059 | 491 | 16.04037039 | 55.55555556 |

| Split | CER mean ± SD (%) | WER mean ± SD (%) |
| --- | ---: | ---: |
| WI | **13.17072319 ± 2.49546838** | **42.00589319 ± 3.68514991** |
| WD | **15.99149747 ± 0.40465024** | **54.60500340 ± 1.49325687** |

Independent best-WER minima are a separate selection convention: their means
are 41.90490906% for WI and 54.27923472% for WD. They must not replace the
paired WER values in the table above without relabeling the comparison.

WI has lower five-fold error than WD in this character-baseline experiment.
WI CER varies more across folds (population SD 2.50 percentage points) than
WD CER (0.40 points). The OnHW paper scores concern a different dataset and
are not direct benchmarks for these English sentence-level results.

### WI fold-2 slowdown and recovery

On 2026-10-16 the user reported a stall near epoch 231. During inspection,
WSL commands timed out and Windows committed-memory usage reached 97%; memory
pressure was a suspected contributor, not a proven root cause. Training
recovered in the same process without a checkpoint restart. The epoch-232
training log records an elapsed 2:37:31, followed by roughly 9–10 second
epochs. The completed record now contains all epochs through 299; its selected
epoch is 291 (CER 10.87687716%, WER 40.18858308%). The interruption affected
elapsed runtime; no missing epoch or selected-prediction mismatch was found.

### Verification and evidence

For all ten runs, independently recalculated the selected CER/WER using
`jiwer` and matched saved metrics within 1e-12 fractional error rate. Verified
all 300 prediction sets have the expected sample count and stored labels
exactly match the ordered validation annotations for the corresponding fold.
Best-CER, best-WER, and latest checkpoint files are present for every fold.
Both source archives match their authenticated SHA-256 values. No training
was launched and original run files were not modified during this check.

Machine-readable summary:
`results/thesis/ed/ed_char_cv_rerun_summary_20261017.json`.
It includes all fold scores, selection epochs, independent WER minima, sample
counts, dataset hashes, and SHA-256 hashes for every saved config, metrics,
log, and complete prediction file.

Summary SHA-256: `49933bfe4584e3f0ca97566dd0771c014d38b88170bdb03d96b57a50b19907bf`.

## 2026-10-17 — English ED extracted-file dataset preparation

Extracted the authenticated English ED archives into `data/tva/ed/gold_wd/`
and `data/tva/ed/gold_wi/` for use with the standard OnHW-style `HRDataset`
loader. Source archives remain unchanged. This changes storage access only:
no resampling, channel selection, label normalization, split regeneration,
or signal conversion was applied. Original filenames, CSV bytes, annotations,
statistics and fold order were preserved.

Each distribution contains 2,550 recordings and 2,561 archived files, about
933.5 MiB uncompressed (about 1.82 GiB combined). All 5,122 extracted archive
files passed SHA-256 comparison against their decompressed source bytes.
Parsed all 5,100 CSV files as float32 and verified nonempty, finite,
seven-channel sequences. Both annotation files in each distribution retain
the original five-fold metadata and every referenced recording exists.

| Distribution | Source archive SHA-256 | Extraction manifest SHA-256 |
| --- | --- | --- |
| WD | `edaf78b0f422f719bbb13153249a5a6667a814d6f5a8d7d0d2ba02535ae55a3e` | `397677f2d5a03e31b3c6a7154ccd44bc3a689b77a0b902a6b5eb4aa4f7c6fca1` |
| WI | `0ba1b8e5d53c7208bbdc4a43cfb701ed6a1d27be382a23ba0aaade8cb3283304` | `3f91aa7b2cf1dfd4afbada8d4d72f884a7ea6db323d66801d875db8170905ca0` |

The per-distribution `extraction_manifest.json` records every archived file's
relative path, byte count, and SHA-256. Combined verification is recorded in
`data/tva/ed/extraction_verification.json`. Data and manifests remain under the
Git-ignored dataset directory.

Added `configs/thesis/ed/character_wi_extracted_cv.yaml` and
`configs/thesis/ed/character_wd_extracted_cv.yaml`. Compared with the respective
completed rerun protocol, only `dataset_format`, `dir_dataset`, and `dir_work`
differ: `json-directory`, the extracted distribution directory, and a new
`character_wi_extracted_cv` or `character_wd_extracted_cv` results directory.
All training hyperparameters and token vocabularies remain identical. These
new configurations do not use the ZIP-only `TVA_ED_DATASET_DIR` override.

Validation through `main.build_dataset`: 120 exact comparisons across WD/WI,
all five folds, train/validation, cache on/off, using three samples per
combination. Raw signal arrays, encoded labels and normalized/padded tensors
matched the ZIP loader exactly; annotations matched and batch collation
returned seven-channel inputs. Both full five-fold CV dry runs passed without
creating recognition results. No model training was launched. Existing run
configs, checkpoints, scores, tokenizer files and ZIP loader code were not
changed. Extraction and configuration usage are documented in `docs/SETUP.md`.

## 2026-10-17 — ED analysis notebook updated for extracted data

Updated `analyze_ed.ipynb` to read `data/tva/ed/gold_wd/` and `gold_wi/`
through normal files rather than opening the source ZIP archives. Removed the
stale Windows Downloads ZIP default and stale tokenizer artifact paths. The
optional `TVA_ED_EXTRACTED_DIR` override points to the extracted root;
`TVA_ED_DATASET_DIR` remains specific to ZIP-backed training.

The notebook now verifies all 5,122 extracted archived files against their
extraction manifests, dataset metadata and canonical raw-record identity,
all five train/validation partitions, sample disjointness, WI writer separation,
and exactly-once validation coverage. It reports shared label prompts
separately from writer/sample leakage and keeps raw labels unchanged.

Added WD/WI train/validation character-frequency plots for the configurable
fold (default 0), plus all-fold total variation, base-2 Jensen–Shannon
divergence and missing-character tables. Retained label-length, global
sequence-length and deterministic 128-recording channel analyses, reading all
inputs from the extracted dataset. Channel estimates remain labelled as
sampled statistics.

Updated coverage comparisons to the configured 79-class character tokenizer,
current IAM handwriting-aware Bigram and original TVA-on-IAM linguistic
Bigram. All 2,550 raw labels round-trip without blank IDs for each tokenizer.
No tokenizer is fitted or selected using ED labels. The retired custom
linguistic condition is not substituted for the active original-TVA condition.

Added the completed `character_wi_cv_rerun` and `character_wd_cv_rerun`
results, explicitly retaining their historical ZIP-backed provenance. Reads
original metrics, checks 300 epochs and saved configs, verifies validation
label order, and recomputes paired CER/WER at the minimum-CER epoch. The means
and population standard deviations match the independently verified
`ed_char_cv_rerun_summary_20261017.json` within 1e-12 percentage points.
Missing recognition files skip that entire split rather than producing an
incomplete-fold aggregate, so dataset-only analysis remains usable on another
machine.

Executed all ten code cells successfully in the TVA Jupyter kernel, saved
outputs inline, and inspected the new distribution and recognition plots.
Six PNG figures and `analysis_summary.json` are exported under the Git-ignored
`results/ed_analysis/` directory, including input/artifact/result provenance
hashes and computed values. No model training, extraction, signal rewriting,
or tokenizer fitting occurs in the notebook. Dataset manifests, tokenizer
artifacts and completed training results remain unchanged.

## 2026-10-17 — Remove superseded ED character launch templates

Removed four redundant YAML files from `configs/thesis/ed/`:
`character_wd_cv.yaml`, `character_wi_cv.yaml`,
`character_wd_cv_rerun.yaml`, and `character_wi_cv_rerun.yaml`.
The first pair differed from the character fold-0 templates only in fold
selection; the second pair additionally selected the completed rerun output
directories. Their launch role is superseded by the extracted-data CV configs.

Retained 12 configs: WD/WI pairs for single-fold ZIP-backed character,
all-fold extracted character, handwriting Bigram, original TVA/IAM text
Bigram, handwriting BPE, and handwriting Unigram. Existing methods and
scientific settings were not changed.

Before removal, verified all ten completed rerun YAML snapshots exactly
match their corresponding historical launch protocol and fold index.
`analyze_ed.ipynb` now uses the saved fold-0 snapshot as its historical
reference, checks all other fold snapshots against it, and checks shared
scientific settings against the current extracted-character protocol. Thus
historical score verification does not require obsolete launch templates.
Historical paths in earlier experiment-log entries remain the paths recorded
at the time of those experiments; the exact effective configs remain with
their original results.

Validation: all four ED character-baseline tests and all six ED matrix-launcher
tests passed. Re-executed all ten notebook code cells successfully after
removing the files; aggregate CER/WER still match the verified result summary.
Remaining tokenizer artifact references and extracted dataset paths resolve.
No active source references to the removed templates remain. Updated README
configuration guidance; saved results, datasets and tokenizers were preserved.

## 2026-10-17 — Handwriting tokenizer encoding/CTC inspection notebook

Added `inspect_handwriting_tokenizers.ipynb`, a CPU-only interactive notebook
for manually inspecting frozen handwriting tokenizer encoding and prediction
decoding. It loads the actual ED greedy Bigram, handwriting BPE, handwriting
Unigram, and historical OnHW utility-Bigram artifacts through TVA's factory.
Editable text/model settings display NFC normalization, segmentation pieces,
character offsets, IDs, metadata where available, and decoded text. A compact
comparison handles unsupported alphabets explicitly without deleting text.

The prediction example calls the production `BestPath` decoder on simulated
scores or user-provided frame IDs / [time, vocabulary] tensors, with optional
valid-length trimming. Intermediate argmax IDs, adjacent-repeat collapse,
blank removal and final text are visible. It distinguishes target decoding
from frame-level CTC decoding and warns that token IDs/artifacts must match the
recognizer checkpoint. Simulated scores demonstrate mechanics and are not
recognition evidence; no recognizer or tokenizer is trained.

Validation: executed all six code cells and saved outputs inline. Checked
English sentence/punctuation/digit input under all three ED models, a word
under the historical OnHW model, and empty input. All 16 all-blank/repeated/
blank-separated CTC cases passed, repeated target IDs remained intact, and
invalid IDs were rejected for all four tokenizers. SHA-256 checks confirmed
frozen artifact files remained unchanged. No dataset or checkpoint is required.
README links to the new playground.

## 2026-10-17 — Simple ED linguistic Bigram frequency notebook

Added `analyze_ed_linguistic_bigrams.ipynb` with four code cells: select the
WD/WI protocol and fold, load the original TVA/IAM linguistic Bigram, count
emitted bigram tokens, and display a table plus frequency plots. It reads the
extracted ED annotations directly and uses the actual greedy encoder.
Singleton and CTC blank IDs are excluded; raw case, spaces, punctuation and
recording multiplicity are retained. Frequencies describe emitted tokens,
not all overlapping adjacent-character candidates. No vocabulary is fitted.

The table includes token IDs, train/validation/total counts and overall
relative frequency. Plots compare train/validation frequencies normalized
by their respective bigram totals and show a count histogram across all
145 vocabulary bigrams, including zero-count entries if present. `TOP_N`
controls the displayed rows and rank plot; the histogram always covers the
full vocabulary. No export framework, recognition loading or extra audit
pipeline is added to this notebook.

Executed all four code cells and retained outputs inline. Default WD fold 0
contains 2,047 training and 503 validation recordings, emitting 28,080 and
7,241 bigram tokens respectively (35,321 total). All 145 vocabulary bigrams
occur. The five most common emitted tokens are `e ` (1,427), `s ` (942),
`t ` (877), `th` (739) and `n ` (697), with spaces preserved.

Independent verification outside the notebook confirmed identical pooled
counts against all 2,550 canonical records for every WD/WI fold. Training
and validation together therefore count the dataset once. Original data,
vocabularies and recognizer results remain unchanged. README links to the
notebook.

Frequency-notebook follow-up: WD and WI are now both displayed by default
using `SPLITS = ['wd', 'wi']`, retaining the four-code-cell structure. Each
protocol has its own count table and a side-by-side train/validation frequency
plot. The pooled histogram is shown once because both protocols contain the
same recordings and emit the same 35,321 total bigram tokens. Default WI fold 0
contains 2,025 train and 525 validation records, emitting 27,961 and 7,360
bigram tokens respectively. Re-executed all four cells and confirmed WD and WI
tables plus both figures are saved inline.

## 2026-10-17 — Individual handwriting prediction-decoding notebook

Added `analyze_handwriting_predictions.ipynb`, a simple five-code-cell
notebook for actual per-record inference and decoding. Editable settings select
ED handwriting Bigram/BPE/Unigram, WD/WI, a completed fold, validation sample,
checkpoint and device (CPU by default). It loads a saved model configuration
and checkpoint, uses the matching canonical frozen tokenizer and extracted
validation recording, and applies standard HRDataset normalization/padding
with augmentation disabled. Older saved Bigram runs predate the ZIP-loader
`ed_distribution` field and refer to former artifact names; the notebook uses
the selected run directory and current matching method config to locate the
frozen artifact, with tokenizer-key and category agreement checks. Original
saved configs and artifacts are not rewritten.

Displays the reference text, actual CTC-surviving prediction IDs and pieces,
reference token IDs/pieces, per-record CER/WER and a short frame-level table.
The frame table shows argmax IDs, blank tokens, and whether a frame contributes
after adjacent-repeat collapse. The final text is decoded with production
`BestPath` and checked against direct decoding of the surviving IDs. Predicted
IDs come from actual model scores, not re-encoding decoded prediction text.

The final cell reads the existing prediction JSON at the loaded checkpoint
epoch and same validation index, verifies validation-label order, and displays
its stored text separately. Existing JSONs contain decoded strings and cannot
recover original logits or frame IDs. Fresh single-record CPU inference can
differ from historical batched/mixed-precision results; this is explicitly
labelled and does not replace experiment scores. The default Bigram WD fold-0
sample showed such a difference, so exact saved-text reproduction is not
claimed. No aggregate recognition result is recalculated by this notebook.

Validation: all five cells executed with outputs retained inline. Actual
checkpoint inference, target-token reconstruction, CTC decoding and stored
prediction lookup passed for all six method/distribution combinations at
fold 0. Also tested validation index 1 with the Bigram best-WER checkpoint.
Default Bigram best-CER checkpoint is epoch 206. No training was launched and
no checkpoint, dataset, tokenizer artifact or original result was modified.

## 2026-10-18 — Matched linguistic BPE/Unigram controls prepared

Prepared and froze two IAM text-only controls for the existing handwriting
BPE/Unigram experiments. These use the same custom training functions with
`handwriting_aware=False`, rather than original TVA's Hugging Face trainers.
Original TVA `bpe`/`unigram` keys and `tva/tokenizers.py` remain unchanged.

### Matched construction and inference

- Both sides use exactly the same ordered 5,694 IAM training lines, including
  repeated transcriptions. Verified line IDs and text against the bundled
  handwriting training corpus. Linguistic fitting reads only authenticated IAM
  labels/selection, with no connectivity evidence or ED labels.
- The text controls count letter spans of length 2–6 with at least 20 text
  occurrences. Their pool has 3,325 candidates; the connectivity-filtered pool
  has 303. The intentional difference is text-based versus handwriting-based
  candidate evidence, merge counts and Unigram initialization.
- BPE learns 145 merges. Unigram uses eight EM iterations per stage, a seed
  multiplier of four, shrink factor 0.75 and the identical pruning procedure.
  The same seed-cap rule starts Unigram with 580 text candidates versus 303
  handwriting candidates, reflecting the different eligible pools.
- Each ED tokenizer has 145 pieces + 78 declared characters + blank = 224
  classes. The IAM-to-ED projection removes `#`, `&`, `*` and adds `%`, `=`.
  Unigram retains learned IAM scores without renormalization and gives added
  forced singleton symbols zero score, matching the handwriting projection.
- Both families now inherit the identical segmentation/encode/decode functions
  from `tva/subword_tokenizers.py`: ordered BPE merges or maximum-score Unigram
  Viterbi, preserving NFC, spaces and punctuation. Verified extracted methods
  are AST-identical to the pre-refactor handwriting methods.
- IAM models are pinned by SHA-256; ED adapters are reconstructed exactly on
  load. Frozen output preflight prevents partial overwrite of differing files.
  Separate training reports record source, candidate and implementation hashes.

### Frozen artifacts

| Algorithm | Native IAM SHA-256 | ED adapter SHA-256 | Pieces shared with handwriting |
| --- | --- | --- | ---: |
| bpe | `b378e1e93b86d3483f15d5bbe5865c31c9581112f75ded56f45589f130b6c1d3` | `c757f01c98f9756b190cc82140a3d454b11fe78bcb707a8924ca72634fba16ea` | 59/145 |
| unigram | `cce6b18818a71b43d74c98e5e2ae334698b8c491c704539aed04e8a02edba0b3` | `b475f4d8fde5320eba2a0b302c97dfc4efbc0f7246f375a35ff4298062776cab` | 50/145 |

The ready ED files live at
`artifacts/tokenizers/ed/linguistic/ed_iam_linguistic_{bpe,unigram}_v1.json`;
corresponding native models are in `source/`, with `.training.json` reports and
one combined `.audit.json` beside the adapters. The existing simple
`train_ed_tokenizers.ipynb` has two added class/import/train sections.
The complete notebook executed successfully, including a second byte-identical
text fit and exact reproduction of the prior handwriting artifacts.

### Recognition commands (manual, fold 0)

```bash
python main.py -c configs/thesis/ed/bpe_linguistic_wd.yaml
python main.py -c configs/thesis/ed/bpe_linguistic_wi.yaml
python main.py -c configs/thesis/ed/unigram_linguistic_wd.yaml
python main.py -c configs/thesis/ed/unigram_linguistic_wi.yaml
```

The four configs match their respective handwriting configs in every field
except tokenizer key, tokenizer artifact path and output directory. Models use
224 output classes, batch 32, seed 42 and 300 epochs. Both distributions use
the same frozen tokenizer. All five fold inputs/outputs pass CV launch preflight.
No recognition training was started.

### Validation

Full unittest suite: 99 tests completed successfully (92 passed, seven optional
skips). New checks cover shared runtime identity, CTC IDs, sentence/symbol
round-trips, native segmentation parity, config matching, source/adapter
corruption and immutable output preflight. Each tokenizer round-trips all
5,100 ED WD/WI label encodings and all 5,612 common-alphabet IAM lines. Eight
real ED train/validation CPU batches (both methods and distributions) produced
finite 224-class outputs and CTC loss using the production architectures.
The combined compatibility audit reads ED labels only after vocabulary fitting.

All pre-existing handwriting artifacts/reports, handwriting training/evidence
source and original TVA tokenizers retain their exact bytes. No existing
recognition runs were changed; no handwriting retraining is required. Preserved
unrelated workspace/editor and inspection-notebook changes. No commit or push
was performed for this preparation.

## 2026-10-19 — Linguistic subword recognition uses extracted ED files

Corrected the four new `bpe_linguistic_{wd,wi}.yaml` and
`unigram_linguistic_{wd,wi}.yaml` configs to use `dataset_format: json-directory`
and `dir_dataset: data/tva/ed/gold_wd` or `gold_wi`. They previously inherited
the older handwriting configs' ZIP loader, despite verified extracted copies
being available since 2026-10-17. No ZIP path or `TVA_ED_DATASET_DIR` override
is needed for these four linguistic recognition runs. Training settings,
tokenizers, folds and result directories are unchanged.

Verified extracted train/validation annotation files are byte-identical to
archive members. Across both tokenizers, both distributions, all five folds,
both partitions and cached/noncached modes, 160 representative comparisons
matched raw sensor arrays, target token IDs and processed tensors exactly.
The extracted loader also worked with an intentionally nonexistent ZIP override.
All four five-fold launch inputs pass preflight; six focused regression tests
passed. Updated the config comparison to allow the equivalent data-container
settings while retaining strict matching of recognition hyperparameters.
No recognition training was launched, and no completed runs were changed.

## 2026-10-19 — Matched linguistic BPE/Unigram ED fold 0 completed

Verified all four manually launched recognition runs completed 300 epochs
(0–299), with complete epoch metrics and predictions. Each WD epoch contains
503 validation labels and each WI epoch 525. Saved configs exactly match the
corresponding `configs/thesis/ed/{bpe,unigram}_linguistic_{wd,wi}.yaml` templates.
The runs use the verified extracted ED data and frozen IAM text-only tokenizers.
Their training settings match the completed handwriting counterparts after
accounting for equivalent ZIP/extracted input settings, tokenizer/artifact and
output paths. Both tokenizers retain the pre-evaluation frozen identities:

- BPE ED adapter SHA-256:
  `c757f01c98f9756b190cc82140a3d454b11fe78bcb707a8924ca72634fba16ea`.
- Unigram ED adapter SHA-256:
  `b475f4d8fde5320eba2a0b302c97dfc4efbc0f7246f375a35ff4298062776cab`.

Independently recomputed CER, WER, mean Levenshtein distance and mean reference
length from saved predictions at each selected CER/WER epoch. Validation labels
match the extracted fold-0 annotations exactly. Best-CER and best-WER checkpoints
are present, with evidence hashes below. The earlier BPE WD attempt
`20261019100231` failed during dataset lookup before training and is excluded;
the completed BPE WD run is `20261019101032`.

### Independent best validation scores

| Method | Split | Run ID | Best CER (%) | CER epoch | Best WER (%) | WER epoch |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Linguistic BPE | WD | `20261019101032` | 28.97478199 | 214 | 67.27565825 | 280 |
| Linguistic BPE | WI | `20261019111405` | 18.00811711 | 245 | 45.21303258 | 278 |
| Linguistic Unigram | WD | `20261019121946` | 30.12491162 | 260 | 70.04298764 | 241 |
| Linguistic Unigram | WI | `20261019132426` | 18.71038351 | 279 | 46.59147870 | 264 |

Epochs are zero-based indices. CER and WER are independent validation minima;
they generally come from different checkpoints. They are not a joint score
pair measured at one common checkpoint.

### Metrics at the selected checkpoints

| Method | Split | WER at best-CER epoch (%) | CER at best-WER epoch (%) |
| --- | --- | ---: | ---: |
| bpe | WD | 68.35034927 | 29.39901013 |
| bpe | WI | 45.41353383 | 18.14036208 |
| unigram | WD | 70.41912950 | 30.52557153 |
| unigram | WI | 46.59147870 | 18.92015140 |

### Matched text-only versus handwriting-aware comparison

All values are independent fold-0 validation minima. Improvements are
linguistic error minus handwriting error, in percentage points; positive values
favor handwriting. Handwriting reference results were read from their original
completed run files rather than transcribed from rounded values.

| Method | Split | Linguistic CER (%) | Handwriting CER (%) | CER improvement (pp) | Linguistic WER (%) | Handwriting WER (%) | WER improvement (pp) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bpe | WD | 28.97478199 | 24.45439547 | 4.52038652 | 67.27565825 | 63.40677055 | 3.86888769 |
| bpe | WI | 18.00811711 | 17.17816590 | 0.82995121 | 45.21303258 | 44.73684211 | 0.47619048 |
| unigram | WD | 30.12491162 | 24.06787650 | 6.05703512 | 70.04298764 | 63.78291241 | 6.26007523 |
| unigram | WI | 18.71038351 | 16.84071321 | 1.86967030 | 46.59147870 | 45.41353383 | 1.17794486 |

Handwriting-aware BPE and Unigram have lower CER and WER than their matched
text-only controls on WD and WI fold 0. The gaps are larger on WD. These are
single-fold, seed-42 results; they do not establish a five-fold or statistical
ranking. Connectivity filtering, evidence-based ranking/initialization and the
resulting candidate-pool sizes are combined in the handwriting treatment;
this comparison does not separate those contributions individually. These
text-only controls use our shared custom trainers and are distinct from
original TVA's Hugging Face BPE/Unigram implementations.

### Saved evidence hashes

Paths are repository-relative. Each run's hashes cover its saved configuration,
metrics, log, full predictions and two selected checkpoints.

| File | SHA-256 |
| --- | --- |
| `results/thesis/ed/linguistic_bpe_wd/0/train_20261019101032.yaml` | `c8b2ebc207ae0fe72d44888a56932adeb0bd2124d4c84c20aaee3d1f2f621d4f` |
| `results/thesis/ed/linguistic_bpe_wd/0/train_20261019101032.json` | `d1d26e1f00673a16c18a19022dbae56287bb8f9b0b5d23356848b10f60e52a46` |
| `results/thesis/ed/linguistic_bpe_wd/0/train_preds_20261019101032.json` | `104a31fbfde7ffeb0d4d783285e80208092bc029eedf9006ed46b700e5a6a64f` |
| `results/thesis/ed/linguistic_bpe_wd/0/train_20261019101032.log` | `6bfae4d08328fdc6a362928211a2c9e987a1d2a1e1d79974dae8d5c643e09f69` |
| `results/thesis/ed/linguistic_bpe_wd/0/checkpoints/best_cer.pth` | `dc3b741c2f88c12c3c696b42e0388b4c1f59bbe593f28176c025bd6793a9dbbf` |
| `results/thesis/ed/linguistic_bpe_wd/0/checkpoints/best_wer.pth` | `8152617322db2e04f46ad817e1d78d97ebd38501841661479e5ff11460d79a4a` |
| `results/thesis/ed/linguistic_bpe_wi/0/train_20261019111405.yaml` | `27d5cb448b6fa2fd3f7d69da7d9b771e02c63d820f11c89c56c08f7740a5feaa` |
| `results/thesis/ed/linguistic_bpe_wi/0/train_20261019111405.json` | `35091bf5df4820590be3b55f81f6546b8a3acbfd17aea15d78ebf15f597383c0` |
| `results/thesis/ed/linguistic_bpe_wi/0/train_preds_20261019111405.json` | `d57e78a8a6ba5775e9deb93133490d9c80e2633ff9ba49eda7ba54b271ddc563` |
| `results/thesis/ed/linguistic_bpe_wi/0/train_20261019111405.log` | `41a498c8e88ed4ebc092fc5ef1cf28b65a90ecc9e249536fc36953799ecdd645` |
| `results/thesis/ed/linguistic_bpe_wi/0/checkpoints/best_cer.pth` | `bdf091a1faee40d1ec7714fb2115f1f62e55aa1d3837bdf113b59717bf2e8c3e` |
| `results/thesis/ed/linguistic_bpe_wi/0/checkpoints/best_wer.pth` | `764f4ae32ba569eca60449374a409a260612324d85f7e44518f39a96dc8e987d` |
| `results/thesis/ed/linguistic_unigram_wd/0/train_20261019121946.yaml` | `b38e94b6905cdb64851cfbea21d4ee5b01df0d5f9ec4d9b4b9587e9f627db67f` |
| `results/thesis/ed/linguistic_unigram_wd/0/train_20261019121946.json` | `6f2c9beb3fdca3398b6e0ce9214fb54f5f7dfed01bb4ba04cc61d34611750d35` |
| `results/thesis/ed/linguistic_unigram_wd/0/train_preds_20261019121946.json` | `0933d47934c96cb4aa743fc4dd1f2a886d3bfe3f300d9ae7e5b82c4312e757de` |
| `results/thesis/ed/linguistic_unigram_wd/0/train_20261019121946.log` | `7d56e35591786d02adff82c10dbb619170fc5eefe0f1f8098a2dae0cd98b6721` |
| `results/thesis/ed/linguistic_unigram_wd/0/checkpoints/best_cer.pth` | `643c55b271339b885407c0cf96e79437c4afd91c165aa0f0b762a34b94822e7d` |
| `results/thesis/ed/linguistic_unigram_wd/0/checkpoints/best_wer.pth` | `30d14442240b3383fe7c0dfdc2a5b64474a9917b7931c22986c946355ebee905` |
| `results/thesis/ed/linguistic_unigram_wi/0/train_20261019132426.yaml` | `4932d52b87b37c3c27922842ca0c6dc3c51c1b7ae64769d47b14a615499569dd` |
| `results/thesis/ed/linguistic_unigram_wi/0/train_20261019132426.json` | `db7acebee06e14084126027ea070c211de1ef6154c8dfdd73dc3888764dfa785` |
| `results/thesis/ed/linguistic_unigram_wi/0/train_preds_20261019132426.json` | `66034a7f5737bec1bd8cbffd1fa67651848267743db3feabd39ef965c5e3d233` |
| `results/thesis/ed/linguistic_unigram_wi/0/train_20261019132426.log` | `7fdedae06cc5af4d0f9908c9f3a1ba80e195ef1b8cacb54d88d2199b3963d491` |
| `results/thesis/ed/linguistic_unigram_wi/0/checkpoints/best_cer.pth` | `3d9662c8830c3d0c16a164d470dc45cb8a17790473c65afb57f53a936c790cfd` |
| `results/thesis/ed/linguistic_unigram_wi/0/checkpoints/best_wer.pth` | `c397ac194fb78d28bc449b19eb8b1994cd8e90f5f1111d300f0f0b586b7fa50e` |

Recorded the completed results without modifying datasets, tokenizers, models
or recognition outputs. No recognition training, commit or push was performed
while verifying these scores. Remaining folds are needed for a full matched
cross-validation comparison.

## 2026-10-19 — Original TVA character configs and ED equivalence verified

Generated a standard character JSON through the unchanged original
`tva.tokenizers.CharacterTokenizer.train()` method, using the declared ED
alphabet and temporary fold-count metadata only. No transcription-derived
vocabulary selection occurs. The shared file is
`artifacts/tokenizers/ed/character/ed_characters_v1.json`, SHA-256
`d83bf465262d65246907518d43ff9b4788a54c82a2209868189aecbecd124751`.

Before removing the previous subclass, compared it with the original TVA
instance: all 79 forward/inverse IDs (78 characters plus blank ID 0) match
exactly, as do encodings and decodings for all 5,100 WD/WI label entries,
the full declared alphabet, empty input and punctuation/symbol examples.
The machine-readable result is saved beside the artifact as
`ed_characters_v1.comparison.json`.

Added `configs/thesis/ed/character_tva_wd.yaml` and
`configs/thesis/ed/character_tva_wi.yaml`. They load the standard file and
extracted ED folders, use fold 0, and have separate `character_tva_wd` and
`character_tva_wi` result directories. Their model/training settings match the
existing extracted character CV configs; only artifact/output paths and fold
selection differ. Optional manual recognition commands:

```bash
python main.py -c configs/thesis/ed/character_tva_wd.yaml
python main.py -c configs/thesis/ed/character_tva_wi.yaml
```

Removed the character subclass and factory override. `get_tokenizer('char')`
now returns exactly TVA's original class. Completed ED configs with
`dir_tokenizer: null` remain compatible: `main.load_configured_tokenizer` loads
the shared file and verifies its IDs against the saved categories. Updated
ED preparation/analysis notebooks to use original `.load()`. Existing character
configs and recognition results were preserved. Identical mappings and
encoding/decoding mean no baseline retraining is required.

Validation: 101 unittest cases completed successfully (94 passed, seven optional
skips), including exact class identity, original `.train()` byte reproduction,
legacy config loading and explicit config matching. New configs loaded real
seven-channel extracted train/validation samples and passed five-fold input
preflight. Notebook code parses after normal IPython magic transformation.
Original TVA source, existing frozen artifacts and subword implementation
fingerprints are unchanged. Preserved unrelated user changes. No recognition
training, commit or push was performed.


## 2026-10-22 — ED tokenizer artifacts use TVA-style fold directories

Reorganized the seven current ED tokenizers under
`artifacts/tokenizers/ed/{family}/{method}/{fold}.json`:
`character/char`, `linguistic/{bigram,bpe,unigram}` and
`handwriting/{bigram,bpe,unigram}`. Each method has real `0.json` through `4.json`
files. These are byte-identical copies of its existing frozen tokenizer, not
models fitted separately on ED folds. IAM remains the learned-vocabulary source;
ED supplies only the declared task alphabet. WD and WI share the files.

Updated 14 recognition configs to point to the method directories, the tokenizer
notebook to write `0.json` and publish fold copies, and analysis notebooks/helpers
to resolve the requested fold. Native IAM sources now live inside their method's
`source/` directory. Shared manifests/audits stay at family level; comparison,
provenance and fitting reports accompany the canonical `0.json`. Archived OnHW
and superseded ED tokenizers retain their historical archive layout.

Known paths from both previous artifact layouts redirect to the new locations.
Completed run configs, metrics, checkpoints and historical experiment entries
remain unchanged. CV input validation uses the same resolver as recognition.
No recognition training was started, and no score needs to be replaced.

All existing tokenizer/native-model bytes and their vocabulary IDs, merges,
probabilities and checksums are unchanged. Of the 29 pre-move JSON artifacts,
24 remain byte-identical; the five fitting reports differ only in the recorded
implementation checksum for `training.py`, whose frozen-reference directory
lookup follows the new method folders. Selection/segmentation code is unchanged,
and original `tva/tokenizers.py` remains byte-identical to upstream `cff221c`.

Validation: 103 unittest cases completed (96 passed, seven optional skips),
including all 35 fold files, old saved paths, segmentation/mapping equivalence,
and preservation of conflicting fold destinations. Executed every code cell of
`train_ed_tokenizers.ipynb`: every output reproduced byte for byte. All 14 updated
recognition configs passed five-fold launcher input validation (70 fold inputs),
without launching recognizer training. Changed notebooks passed syntax checks
with normal IPython transformation. No commit or push performed.
