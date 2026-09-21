"""Leakage-safe FAU projection of the frozen IAM handwriting vocabulary."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from tva.handwriting_bigram_adapter import canonical_json_bytes, sha256_bytes


SOURCE_SCHEMA = 'dtlr.handwriting-bigram-tokenizer.v1'
SOURCE_MODEL_VERSION = 'iam-english-handwriting-bigram-v1'
SOURCE_SHA256 = '2fbb81479211454d8ad028d8af76eb26e76b75b80408a1a1e1f4efa15ab26a92'
SOURCE_SIZE = 222
SOURCE_SINGLETON_COUNT = 76
SOURCE_BIGRAM_COUNT = 145

ADAPTER_SCHEMA = 'tva.fau-handwriting-bigram-adapter.v1'
ADAPTER_POLICY_ID = 'fau-english-iam-handwriting-bigram-greedy-v1'
ADAPTER_FROZEN_DATE = '2026-09-21'
ADAPTER_SIZE = 224
ADAPTER_BIGRAM_COUNT = 145
ADAPTER_SHA256 = '4c06828ac3b0ae03e98d569b0f3fea1cdfbc0a125f6eeffcc7ffb2a4935f3f52'
GREEDY_POLICY = 'greedy-left-to-right-v1'

FAU_ALPHABET = (
    ' ', '!', '"', '%', "'", '(', ')', '+', ',', '-', '.', '/',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '=', '?',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
)


def _validate_mappings(model: dict[str, Any]) -> None:
    vocab = model.get('vocab')
    raw_idx_token = model.get('idx_token')
    if not isinstance(vocab, dict) or not isinstance(raw_idx_token, dict):
        raise ValueError('model must contain vocab and idx_token mappings')
    expected_ids = set(range(len(vocab)))
    if set(vocab.values()) != expected_ids:
        raise ValueError('vocabulary IDs must be unique, contiguous, and zero-based')
    if set(raw_idx_token) != {str(index) for index in expected_ids}:
        raise ValueError('idx_token keys must cover every vocabulary ID')
    if any(raw_idx_token[str(index)] != token for token, index in vocab.items()):
        raise ValueError('vocab and idx_token must be inverse mappings')
    if model.get('size') != len(vocab):
        raise ValueError('declared size does not match vocabulary size')


def validate_frozen_iam_source(model: dict[str, Any], digest: str) -> None:
    """Authenticate the exact DTLR IAM-only artifact and its core invariants."""
    if digest != SOURCE_SHA256:
        raise ValueError(
            'frozen IAM source SHA-256 mismatch: expected '
            f'{SOURCE_SHA256}, got {digest}'
        )
    required = {
        'schema_version': SOURCE_SCHEMA,
        'model_version': SOURCE_MODEL_VERSION,
        'status': 'frozen-experiment-policy',
        'dataset': 'IAM',
        'training_split': 'train',
        'blank_token': '',
        'blank_id': 0,
        'unicode_normalization': 'NFC',
        'size': SOURCE_SIZE,
        'eligible_bigram_count': SOURCE_BIGRAM_COUNT,
    }
    for key, expected in required.items():
        if model.get(key) != expected:
            raise ValueError(
                f'frozen IAM source {key} mismatch: expected {expected!r}, '
                f'got {model.get(key)!r}'
            )
    _validate_mappings(model)
    if model['vocab'].get('') != 0 or model['idx_token'].get('0') != '':
        raise ValueError('frozen IAM source blank must be empty token ID 0')

    singletons = [token for token in model['vocab'] if len(token) == 1]
    if len(singletons) != SOURCE_SINGLETON_COUNT:
        raise ValueError('frozen IAM source has an unexpected singleton count')
    rows = model.get('vocabulary')
    if not isinstance(rows, list) or len(rows) != SOURCE_BIGRAM_COUNT:
        raise ValueError('frozen IAM source has an invalid bigram vocabulary')
    row_tokens = [row.get('token') for row in rows if isinstance(row, dict)]
    mapped_bigrams = [
        token
        for token, index in sorted(
            model['vocab'].items(), key=lambda item: item[1]
        )
        if len(token) == 2
    ]
    if len(row_tokens) != len(rows) or row_tokens != mapped_bigrams:
        raise ValueError('source bigram rows do not match source vocabulary order')
    if any(
        not isinstance(row.get('utility'), (int, float))
        for row in rows
    ):
        raise ValueError('source bigram rows must contain numeric utility values')

    provenance = model.get('evidence_provenance')
    if not isinstance(provenance, dict):
        raise ValueError('frozen IAM source lacks evidence provenance')
    if (
        provenance.get('sole_dataset') != 'IAM'
        or provenance.get('sole_split') != 'train'
    ):
        raise ValueError('frozen IAM source is not IAM-train-only')


def build_fau_adapter(source_path: str | Path) -> dict[str, Any]:
    """Build the adapter without reading FAU annotation labels or frequencies."""
    source_bytes = Path(source_path).read_bytes()
    digest = sha256_bytes(source_bytes)
    try:
        source = json.loads(source_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError('frozen IAM source is not valid UTF-8 JSON') from error
    if not isinstance(source, dict):
        raise ValueError('frozen IAM source root must be a JSON object')
    validate_frozen_iam_source(source, digest)

    rows = copy.deepcopy(source['vocabulary'])
    if any(any(char not in FAU_ALPHABET for char in row['token']) for row in rows):
        raise ValueError('a frozen IAM bigram is outside the declared FAU alphabet')
    ordered_tokens = ['', *FAU_ALPHABET, *(row['token'] for row in rows)]
    if len(set(ordered_tokens)) != len(ordered_tokens):
        raise ValueError('projected adapter tokens are not unique')

    adapter = {
        'schema_version': ADAPTER_SCHEMA,
        'model_version': ADAPTER_POLICY_ID,
        'status': 'frozen-downstream-adapter',
        'dataset': 'FAU English',
        'text_normalization': 'NFC',
        'blank_token': '',
        'blank_id': 0,
        'policy': {
            'vocabulary_selection': 'frozen-iam-train-handwriting-evidence-only',
            'overlap_resolution': GREEDY_POLICY,
            'single_character_utility': 0.0,
            'required_characters': list(FAU_ALPHABET),
        },
        'vocab': {token: index for index, token in enumerate(ordered_tokens)},
        'idx_token': {
            str(index): token for index, token in enumerate(ordered_tokens)
        },
        'size': len(ordered_tokens),
        'eligible_bigram_count': len(rows),
        'vocabulary': rows,
        'adapter': {
            'policy_id': ADAPTER_POLICY_ID,
            'frozen_date': ADAPTER_FROZEN_DATE,
            'task': 'FAU English sentence recognition, WD and WI',
            'source_model': {
                'schema_version': source['schema_version'],
                'model_version': source['model_version'],
                'sha256': digest,
                'size': source['size'],
                'eligible_bigram_count': source['eligible_bigram_count'],
                'source_scores_sha256': source['source_scores_sha256'],
                'source_manifest_sha256': source['evidence_provenance'][
                    'source_manifest_sha256'
                ],
            },
            'projection': 'declared-task-alphabet-v1',
            'task_alphabet': list(FAU_ALPHABET),
            'task_alphabet_source': (
                'identical categories fields in gold_wd/annotations.json and '
                'gold_wi/annotations.json'
            ),
            'annotation_label_values_read_by_builder': False,
            'retained_source_bigram_count': len(rows),
            'source_singletons_excluded': sorted(
                token for token in source['vocab']
                if len(token) == 1 and token not in FAU_ALPHABET
            ),
            'task_singletons_added': sorted(
                char for char in FAU_ALPHABET if char not in source['vocab']
            ),
        },
        'note': (
            'FAU adapter of the frozen IAM-only handwriting bigram vocabulary. '
            'The 145 bigrams and their scores are unchanged. FAU labels and '
            'recognition results did not select or rank tokens.'
        ),
    }
    _validate_mappings(adapter)
    if adapter['size'] != ADAPTER_SIZE:
        raise ValueError('FAU adapter does not have the expected 224 classes')
    if adapter['eligible_bigram_count'] != ADAPTER_BIGRAM_COUNT:
        raise ValueError('FAU adapter does not preserve all 145 IAM bigrams')
    if adapter['adapter']['source_singletons_excluded'] != ['*']:
        raise ValueError('unexpected IAM-only singleton difference')
    if adapter['adapter']['task_singletons_added'] != ['%', '(', '=']:
        raise ValueError('unexpected FAU-only singleton difference')
    digest = sha256_bytes(canonical_json_bytes(adapter))
    if digest != ADAPTER_SHA256:
        raise ValueError(
            f'frozen FAU adapter SHA-256 mismatch: expected {ADAPTER_SHA256}, '
            f'got {digest}'
        )
    return adapter


def write_adapter(adapter: dict[str, Any], output_path: str | Path) -> str:
    """Write canonical adapter bytes and return their SHA-256 digest."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    content = canonical_json_bytes(adapter)
    output.write_bytes(content)
    return sha256_bytes(content)
