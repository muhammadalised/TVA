# Thesis Storage and Machine Setup

## Storage roles

The thesis uses three separate storage locations with different purposes:

- **GitHub:** code, committed configurations, and Markdown documentation.
- **Google Drive:** the master dataset archives and backups of completed
  experiments.
- **Local SSD on each machine:** extracted datasets and active training runs.

Training must use the local SSD copy. Do not train directly from a Google Drive
folder because synchronization can slow down file access or expose incomplete
files while they are being transferred.

## Conda environment

The dedicated environment is named `tva-thesis` and uses Python 3.11. The
reproducible package definition is stored in `environment.yml`.

Create it on a new machine from the repository root with:

```bash
conda env create -f environment.yml
conda activate tva-thesis
```

If the environment already exists, update it with:

```bash
conda env update -n tva-thesis -f environment.yml --prune
```

Register it for the repository notebooks with:

```bash
python -m ipykernel install --user --name tva-thesis \
  --display-name "Python (tva-thesis)"
```

Each GPU training machine must be checked separately for a compatible NVIDIA
driver and CUDA-enabled PyTorch build before full training. Do not assume that
an environment created on one operating system automatically has GPU support
on another.

Activate the environment before running any repository command:

```bash
conda activate tva-thesis
python main.py -c configs/train.yaml
```

## Google Drive structure

Create the following folder hierarchy inside Google Drive:

```text
Handwriting_Thesis/
├── 01_datasets/
│   └── onhw_words500/
│       └── v1/
│           ├── source_archives/
│           │   ├── OnHW-words500_dep.zip
│           │   └── OnHW-words500_indep.zip
│           ├── processed_archives/
│           │   ├── onhw_words500_wd_word_rh_v1.zip
│           │   └── onhw_words500_wi_word_rh_v1.zip
│           ├── documentation/
│           │   ├── README.pdf
│           │   └── DATASET_INFO.md
│           └── MANIFEST.sha256
├── 02_experiment_backups/
│   ├── baseline_character/
│   ├── baseline_linguistic/
│   └── proposed_handwriting_informed/
├── 03_thesis_documents/
└── 04_meeting_notes/
```

### Source archives

The two original Fraunhofer ZIP files are the permanent source data. Keep their
original filenames and do not modify their contents.

### Processed archives

After preprocessing is verified, create one ZIP archive for each processed
dataset. These archives allow every machine to start from the exact same
processed data without independently repeating preprocessing.

If preprocessing or dataset splits change, create a new `v2` directory. Never
silently replace files inside `v1` after experiments have started.

### Checksum manifest

`MANIFEST.sha256` records file hashes for the archives. A checksum is a digital
fingerprint: matching hashes confirm that two machines have identical files.
The exact command for generating and checking this file will be added after the
archives are placed locally.

## Local repository structure

Use the same structure inside every clone of the TVA repository:

```text
TVA/
├── data/
│   ├── raw/
│   │   ├── Words500_dep_R/
│   │   │   ├── 0/
│   │   │   ├── 1/
│   │   │   ├── 2/
│   │   │   ├── 3/
│   │   │   └── 4/
│   │   └── Words500_indep_R/
│   │       ├── 0/
│   │       ├── 1/
│   │       ├── 2/
│   │       ├── 3/
│   │       └── 4/
│   └── tva/
│       ├── onhw_words500_wd_word_rh/
│       │   ├── train.json
│       │   ├── val.json
│       │   └── tokenizers/
│       └── onhw_words500_wi_word_rh/
│           ├── train.json
│           ├── val.json
│           └── tokenizers/
├── results/
│   ├── tva/
│   │   └── rewi/
│   └── thesis/
│       ├── development/
│       └── final/
├── configs/
├── docs/
└── tva/
```

The contents of `data/` and `results/` stay outside Git. They are already
ignored by the repository. The relative paths inside the repository remain the
same on macOS and Windows, even when the absolute repository location differs.

The downloaded ZIP files contain extra top-level directories named
`Words500_dep_02` and `Words500_indep_02`. When extracting them, place the
contents of those directories directly inside `Words500_dep_R` and
`Words500_indep_R`. The preparation notebook expects fold directories `0`
through `4` directly below each configured raw directory.

Use `results/tva/` for reproductions of existing TVA experiments. Use
`results/thesis/` for the newly developed handwriting-informed method and its
controlled comparisons.

## Dataset configuration paths

Use these portable relative paths in experiment configurations:

```yaml
# Writer-dependent, right-handed
dir_dataset: data/tva/onhw_words500_wd_word_rh
```

```yaml
# Writer-independent, right-handed
dir_dataset: data/tva/onhw_words500_wi_word_rh
```

This avoids embedding a machine-specific path such as a macOS username or a
Windows drive letter in a scientific configuration.

## Experiment directory names

Every run must have a unique, descriptive directory. Use this pattern:

```text
<method>_<setting>_fold<fold>_seed<seed>_<YYYYMMDD-HHMM>_<machine>
```

Examples:

```text
B0_char_wd_fold0_seed42_20260801-1400_workstation
P1_motion_pair_wd_fold0_seed42_20260810-0930_laptop
```

The timestamp and machine name prevent two machines from overwriting the same
run. Store active runs locally. When a run finishes, compress its complete
directory and upload it to the matching folder under
`02_experiment_backups/`.

## Normal workflow across machines

1. Write and test code on the main development machine.
2. Commit and push the code to GitHub.
3. Pull the exact commit on the training machine.
4. Confirm that the local dataset version and checksum match.
5. Start a uniquely named local result directory.
6. Record the Git commit, configuration, seed, and machine information.
7. Train using the local dataset copy.
8. Upload a compressed backup after the run has finished.

Do not use Google Drive to synchronize source code between machines; GitHub is
the source of truth for code. Prefer resuming an unfinished run on the same
machine and with the same software environment.

## Resuming interrupted training

Every completed epoch atomically updates this recovery checkpoint:

```text
<dir_work>/<fold>/checkpoints/latest.pth
```

The checkpoint contains the model, optimizer, learning-rate scheduler,
mixed-precision scaler, completed epoch, metric history, and random states.
Numbered milestone checkpoints are additionally retained according to
`freq_save`.

To resume, set `checkpoint` in the same experiment configuration while leaving
`epoch` at the original total number of intended epochs:

```yaml
checkpoint: results/thesis/baselines/B0_char_wd_rh/0/checkpoints/latest.pth
epoch: 300
```

Then run the normal training command. The log must state the next epoch, for
example `continuing at epoch 75`. An older checkpoint containing only model
weights can still be loaded, but it starts a new optimizer and schedule and is
therefore not a true resume.
