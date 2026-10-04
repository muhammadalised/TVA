"""Authenticate IAM input and train using the unchanged original TVA algorithm."""
from pathlib import Path
import inspect
import json
import os
import pickle
import platform
import subprocess
import sys
import tempfile
import textwrap

from .common import canonical_json_bytes as json_bytes, require, save_frozen, sha256_bytes as sha256
from .ed import ED_ALPHABET

LABELS_SHA256 = '5ac34ad37ba0b125308fe1a2bc97095985e25dfa76495628c3bb3895c0b446ab'
SELECTION_SHA256 = '7893dfba4febe6df99cf0bdb0c74fabe7b736d2a6af05033d8638b90455bc1c2'
DEFAULT_LABELS = Path('/home/artellisys/DTLR/data/IAM_new/labels.pkl')
DEFAULT_SELECTION = Path('/home/artellisys/dtlr-output/iam-train-full/selection.json')


def read_iam_training(labels_path, selection_path):
    """Authenticate every selected transcript and return all 5,694 train rows."""
    IAM_LABELS = Path(labels_path)
    IAM_SELECTION = Path(selection_path)
    labels_bytes = IAM_LABELS.read_bytes()
    selection_bytes = IAM_SELECTION.read_bytes()
    require(sha256(labels_bytes) == LABELS_SHA256, 'IAM labels SHA-256 mismatch')
    require(sha256(selection_bytes) == SELECTION_SHA256, 'IAM selection SHA-256 mismatch')
    selection = json.loads(selection_bytes)
    require(
        selection.get('schema_version') == 'dtlr.iam-selection.v1'
        and selection.get('dataset') == 'IAM'
        and selection.get('split') == 'train'
        and selection.get('requested_count') == 5694
        and selection.get('labels_sha256') == LABELS_SHA256,
        'Manifest must identify the complete IAM training split',
    )
    labels = pickle.loads(labels_bytes)
    train_rows = labels['ground_truth']['train']
    require(isinstance(train_rows, list) and len(train_rows) == 5694, 'Unexpected IAM train split')
    by_id = {row['id']: row for row in train_rows}
    require(len(by_id) == len(train_rows), 'Duplicate IAM training line ID')
    selected_ids = [row['id'] for row in selection['lines']]
    require(
        len(selected_ids) == 5694 and len(set(selected_ids)) == 5694
        and set(selected_ids) == set(by_id),
        'Manifest must cover every IAM training line exactly once',
    )
    annotations = []
    for selected in selection['lines']:
        row = by_id[selected['id']]
        text = row['text']
        require(isinstance(text, str) and bool(text), 'Invalid IAM transcript')
        require(
            sha256(text.encode('utf-8')) == selected['transcription_sha256'],
            f'Transcription hash mismatch: {row["id"]}',
        )
        annotations.append({'id': row['id'], 'label': text})
    return annotations


def train_iam_bigram(labels_path, selection_path, categories, size, output_path):
    """Run BigramTokenizer.train in a fresh, hash-seeded Python process."""
    from tva.tokenizers import BigramTokenizer

    REPO_ROOT = Path(__file__).resolve().parents[2]
    CATEGORIES = list(categories)
    require(
        CATEGORIES == ['', *ED_ALPHABET] and size == 224,
        'This frozen ED experiment requires 78 singletons, blank, and 145 bigrams',
    )
    VOCAB_SIZE = size
    BIGRAM_COUNT = 145
    HASH_SEED = '42'
    OUTPUT = Path(output_path)
    PROVENANCE = OUTPUT.with_suffix('.provenance.json')
    annotations = read_iam_training(labels_path, selection_path)
    unique_labels = {row['label'] for row in annotations}
    training_code = textwrap.dedent("""
    import json
    import sys
    from tva.tokenizers import get_tokenizer

    dir_ds, categories_json, size = sys.argv[1:]
    categories = json.loads(categories_json)
    tokenizer_bigram = get_tokenizer('bigram')
    tokenizer_bigram.train(dir_ds, categories, int(size))
    """)
    training_env = dict(os.environ, PYTHONHASHSEED=HASH_SEED)
    with tempfile.TemporaryDirectory(prefix='iam-tva-bigram-') as directory:
        dir_ds = Path(directory)
        train_json_bytes = json.dumps(
            {'info': {'num_fold': 1}, 'annotations': {'0': annotations}},
            ensure_ascii=False, indent=2,
        ).encode('utf-8')
        (dir_ds / 'train.json').write_bytes(train_json_bytes)
        result = subprocess.run(
            [sys.executable, '-c', training_code, str(dir_ds),
             json.dumps(CATEGORIES), str(VOCAB_SIZE)],
            cwd=REPO_ROOT, env=training_env, capture_output=True, text=True,
        )
        print(result.stdout, end='')
        print(result.stderr, end='')
        result.check_returncode()
        trained_bytes = (dir_ds / 'tokenizers' / f'bigram{VOCAB_SIZE}' / '0.json').read_bytes()
        # Re-run the original trainer to verify deterministic artifact bytes.
        repeated = subprocess.run(
            [sys.executable, '-c', training_code, str(dir_ds),
             json.dumps(CATEGORIES), str(VOCAB_SIZE)],
            cwd=REPO_ROOT, env=training_env, capture_output=True, text=True,
        )
        repeated.check_returncode()
        require(
            trained_bytes == (dir_ds / 'tokenizers' / f'bigram{VOCAB_SIZE}' / '0.json').read_bytes(),
            'Repeated original-TVA training produced different bytes',
        )
    model = json.loads(trained_bytes)
    vocab = model['vocab']
    selected_bigrams = [token for token, index in sorted(vocab.items(), key=lambda item: item[1]) if len(token) == 2]
    require(len(vocab) == VOCAB_SIZE, 'Unexpected output class count')
    require(vocab.get('') == 0, 'CTC blank must have ID 0')
    require({t for t in vocab if len(t) == 1} == set(ED_ALPHABET), 'Singleton alphabet mismatch')
    require(len(selected_bigrams) == BIGRAM_COUNT, 'Expected 145 selected bigrams')
    require(set(vocab.values()) == set(range(VOCAB_SIZE)), 'Invalid vocabulary IDs')
    require(
        model['idx_token'] == {str(index): token for token, index in vocab.items()},
        'Forward and inverse mappings disagree',
    )
    require(
        all(char in ED_ALPHABET for token in selected_bigrams for char in token),
        'A selected unrestricted IAM pair falls outside the ED alphabet; do not silently filter it',
    )

    provenance = {
        'schema_version': 'tva.iam-original-bigram-provenance.v1',
        'condition': 'original-TVA-on-IAM linguistic bigram',
        'source_dataset': 'IAM', 'source_split': 'train', 'downstream_dataset': 'ED',
        'labels_sha256': LABELS_SHA256, 'selection_sha256': SELECTION_SHA256,
        'training_annotations_sha256': sha256(train_json_bytes),
        'line_count': len(annotations), 'unique_complete_label_count': len(unique_labels),
        'trainer': 'tva.tokenizers.BigramTokenizer.train',
        'trainer_source_sha256': sha256(inspect.getsource(BigramTokenizer.train).encode('utf-8')),
        'python_version': platform.python_version(), 'python_hash_seed': HASH_SEED,
        'selection': 'unique-complete-labels; all adjacent pairs; Counter.most_common',
        'text_preprocessing': 'none; raw IAM transcripts retained',
        'encoding': 'original-TVA greedy left-to-right',
        'categories': sorted(set(CATEGORIES)), 'size': VOCAB_SIZE, 'bigram_count': BIGRAM_COUNT,
        'blank_token': '', 'blank_id': 0,
        'ed_labels_read_for_selection': False,
        'repeated_training_byte_identical': True,
        'tokenizer_sha256': sha256(trained_bytes),
    }
    # Preflight both files before writing either one.
    for path, content in [(OUTPUT, trained_bytes), (PROVENANCE, json_bytes(provenance))]:
        if path.exists():
            require(path.read_bytes() == content, f'Existing frozen file differs: {path}')
    save_frozen(OUTPUT, trained_bytes)
    save_frozen(PROVENANCE, json_bytes(provenance))

    return OUTPUT
