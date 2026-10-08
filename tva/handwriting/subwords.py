"""ED alphabet projection of the frozen IAM BPE/Unigram release.

Only declared categories and authenticated source files enter construction.
DTLR training, evidence generation and ED annotation reading live elsewhere.
"""
import json
from pathlib import Path

from .common import canonical_json_bytes, require, sha256_bytes
from .ed import ED_ALPHABET

SOURCE_FILES = {
    'bpe': 'iam-english-handwriting-bpe-v1.json',
    'unigram': 'iam-english-handwriting-unigram-v1.json',
}
SOURCE_SHA256 = {
    'bpe': 'a504c2306200faf6eab98f3feef6f98b81e6e6c6c7f707cf31aa3399a660be47',
    'unigram': 'af2bc8455afd796367737857f3a36f1deb45f07ec9783a4bbd7f15057f2b9476',
}
MANIFEST_FILENAME = 'iam-english-handwriting-subwords-v1.manifest.json'
MANIFEST_SHA256 = 'e8b42352a286dde90d6956ae22e92a33d8a94fcf8d251d2ce1cfe224492b6528'
ADAPTER_SCHEMA = 'tva.ed-handwriting-subword-adapter.v1'
ADAPTER_FILES = {
    kind: f'ed_iam_handwriting_{kind}_v1.json' for kind in SOURCE_FILES
}


def load_source(path, kind):
    """Authenticate exact released bytes, including all learned/provenance fields."""
    require(kind in SOURCE_FILES, 'Unsupported handwriting subword algorithm')
    raw = Path(path).read_bytes()
    require(sha256_bytes(raw) == SOURCE_SHA256[kind], 'Frozen IAM source SHA-256 mismatch')
    model = json.loads(raw)
    require(model['status'] == 'frozen-experiment-policy', 'Expected a frozen source')
    require(model['algorithm'] == f'handwriting-aware-{kind}', 'Wrong source algorithm')
    require(model['dataset'] == 'IAM' and model['training_split'] == 'train', 'Expected IAM train only')
    require(model['size'] == 225 and len(model['vocabulary']) == 145, 'Unexpected source budget')
    return model


def build_ed_subword_adapter(source_path, kind):
    """Retain all 145 pieces and inference parameters; change only alphabet/IDs."""
    source = load_source(source_path, kind)
    pieces = [t for t, i in sorted(source['vocab'].items(), key=lambda item: item[1]) if len(t) > 1]
    require(len(pieces) == 145, 'Expected 145 selected pieces')
    require(all(all(c in ED_ALPHABET and c.isalpha() for c in t) for t in pieces),
            'Selected pieces must be ED letters with singleton fallback')
    singles = {t for t in source['vocab'] if len(t) == 1}
    added = sorted(set(ED_ALPHABET) - singles)
    removed = sorted(singles - set(ED_ALPHABET))
    require(added == ['%', '='] and removed == ['#', '&', '*'], 'Unexpected alphabet projection')
    tokens = ['', *ED_ALPHABET, *pieces]
    require(len(tokens) == len(set(tokens)) == 224, 'Invalid ED vocabulary')
    return {
        'schema_version': ADAPTER_SCHEMA,
        'model_version': f'ed-iam-handwriting-{kind}-v1',
        'status': 'frozen-downstream-adapter',
        'frozen_date': '2026-10-08',
        'dataset': 'ED',
        'algorithm': f'handwriting-aware-{kind}',
        'unicode_normalization': 'NFC',
        'blank_id': 0, 'blank_token': '',
        'size': 224, 'subword_count': 145,
        'vocab': {t: i for i, t in enumerate(tokens)},
        'idx_token': {str(i): t for i, t in enumerate(tokens)},
        'source_model': {
            'filename': SOURCE_FILES[kind], 'sha256': SOURCE_SHA256[kind],
            'model_version': source['model_version'],
            'release_manifest_sha256': MANIFEST_SHA256,
        },
        'policy': {
            'vocabulary_selection': 'unchanged-frozen-IAM-train-pieces',
            'inference': source['policy']['inference'],
            'ordered_merges_unchanged': kind == 'bpe',
            'learned_log_probs_unchanged': kind == 'unigram',
            'unigram_probability_renormalization': False,
            'added_singletons': added, 'removed_singletons': removed,
            'added_singleton_behavior': 'forced-one-character-token; boundary-for-letter-pieces',
            'added_singleton_viterbi_score': 0.0 if kind == 'unigram' else None,
            'score_interpretation': (
                'Original IAM token log probabilities plus zero-score forced symbols; '
                'these are segmentation weights, not a new normalized ED probability model.'
                if kind == 'unigram' else 'Original ordered BPE merges.'
            ),
            'source_ids_preserved': False,
            'annotation_label_values_read_by_builder': False,
            'unknown_characters': 'raise-error',
        },
    }


def load_ed_subword_adapter(path, kind):
    """Authenticate a derived file by exact reconstruction from its pinned source."""
    path = Path(path)
    raw = path.read_bytes()
    source_path = path.parent / 'source' / SOURCE_FILES[kind]
    expected = build_ed_subword_adapter(source_path, kind)
    require(raw == canonical_json_bytes(expected), 'Frozen ED subword adapter differs from v1')
    return expected, load_source(source_path, kind)
