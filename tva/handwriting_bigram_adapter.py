"""Build the frozen IAM+READ-to-OnHW handwriting-bigram adapter."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


SOURCE_SCHEMA = 'dtlr.handwriting-bigram-tokenizer.v2'
SOURCE_MODEL_VERSION = 'iam-read-combined-v1'
SOURCE_SHA256 = '5c5d9f1689a4802fc5e9451e5afe8abdfb090562587ab78ddd343211feb94dd4'
SOURCE_SIZE = 494
SOURCE_BIGRAM_COUNT = 402

ADAPTER_POLICY_ID = 'onhw-words500-rh-iam-read-v1'
ADAPTER_FROZEN_DATE = '2026-09-17'
ADAPTER_SIZE = 419
ADAPTER_BIGRAM_COUNT = 359
ADAPTER_SHA256 = '12ce25d8bedc552e6b3497ffb1d07e506b01b34296cc21b970f82a550cbf2bfe'

ONHW_ALPHABET = tuple(
    'ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜabcdefghijklmnopqrstuvwxyzäöüß'
)


def sha256_bytes(content: bytes) -> str:
    """Return the lowercase SHA-256 digest for bytes."""
    return hashlib.sha256(content).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON deterministically for a stable artifact checksum."""
    text = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )
    return f'{text}\n'.encode('utf-8')


def _validate_mappings(model: dict[str, Any]) -> None:
    vocab = model.get('vocab')
    idx_token = model.get('idx_token')
    if not isinstance(vocab, dict) or not isinstance(idx_token, dict):
        raise ValueError('source model must contain vocab and idx_token mappings')
    if any(not isinstance(token, str) or not isinstance(index, int)
           for token, index in vocab.items()):
        raise ValueError('vocab must map strings to integer IDs')
    expected_ids = set(range(len(vocab)))
    if set(vocab.values()) != expected_ids:
        raise ValueError('vocabulary IDs must be unique, contiguous, and zero-based')
    if set(idx_token) != {str(index) for index in expected_ids}:
        raise ValueError('idx_token keys must cover every vocabulary ID')
    if any(idx_token[str(index)] != token for token, index in vocab.items()):
        raise ValueError('vocab and idx_token must be inverse mappings')
    if model.get('size') != len(vocab):
        raise ValueError('declared model size does not match vocabulary size')


def validate_frozen_source(model: dict[str, Any], source_sha256: str) -> None:
    """Reject any input other than the audited combined IAM+READ artifact."""
    if source_sha256 != SOURCE_SHA256:
        raise ValueError(
            f'frozen source SHA-256 mismatch: expected {SOURCE_SHA256}, '
            f'got {source_sha256}'
        )
    required = {
        'schema_version': SOURCE_SCHEMA,
        'model_version': SOURCE_MODEL_VERSION,
        'text_normalization': 'NFC',
        'blank_token': '',
        'blank_id': 0,
        'size': SOURCE_SIZE,
        'eligible_bigram_count': SOURCE_BIGRAM_COUNT,
    }
    for key, expected in required.items():
        if model.get(key) != expected:
            raise ValueError(
                f'frozen source {key} mismatch: expected {expected!r}, '
                f'got {model.get(key)!r}'
            )
    policy = model.get('policy')
    if not isinstance(policy, dict):
        raise ValueError('frozen source lacks its policy object')
    if policy.get('overlap_resolution') != 'maximum-total-utility-non-overlapping-v1':
        raise ValueError('frozen source has an unsupported overlap-resolution policy')
    if policy.get('single_character_utility') != 0.0:
        raise ValueError('frozen source must assign zero utility to single characters')

    _validate_mappings(model)
    if model['vocab'].get('') != 0 or model['idx_token'].get('0') != '':
        raise ValueError('frozen source blank must be the empty token at ID 0')

    vocabulary = model.get('vocabulary')
    if not isinstance(vocabulary, list) or len(vocabulary) != SOURCE_BIGRAM_COUNT:
        raise ValueError('frozen source has an invalid bigram vocabulary list')
    tokens = []
    for row in vocabulary:
        if not isinstance(row, dict) or not isinstance(row.get('token'), str):
            raise ValueError('each bigram vocabulary row must contain a token string')
        token = row['token']
        if len(token) != 2:
            raise ValueError(f'eligible token is not a two-character bigram: {token!r}')
        if not isinstance(row.get('utility'), (int, float)):
            raise ValueError(f'bigram lacks numeric utility: {token!r}')
        tokens.append(token)
    if len(set(tokens)) != len(tokens):
        raise ValueError('frozen source contains duplicate bigram rows')
    mapped_bigrams = [
        token for token, _ in sorted(model['vocab'].items(), key=lambda item: item[1])
        if len(token) == 2
    ]
    if tokens != mapped_bigrams:
        raise ValueError('bigram rows do not match source vocabulary order')


def project_source_model(
    source_model: dict[str, Any],
    source_sha256: str,
    alphabet: Sequence[str] = ONHW_ALPHABET,
) -> dict[str, Any]:
    """Project a validated source model using only a predeclared alphabet."""
    alphabet = tuple(alphabet)
    if not alphabet or any(not isinstance(char, str) or len(char) != 1 for char in alphabet):
        raise ValueError('task alphabet must contain one-character strings')
    if len(set(alphabet)) != len(alphabet):
        raise ValueError('task alphabet contains duplicate characters')
    if '' in alphabet:
        raise ValueError('task alphabet must not contain the CTC blank')

    _validate_mappings(source_model)
    vocabulary = source_model.get('vocabulary')
    if not isinstance(vocabulary, list):
        raise ValueError('source model lacks its ordered bigram vocabulary')
    allowed = set(alphabet)
    retained_rows = [
        copy.deepcopy(row)
        for row in vocabulary
        if isinstance(row, dict)
        and isinstance(row.get('token'), str)
        and len(row['token']) == 2
        and all(char in allowed for char in row['token'])
    ]
    ordered_tokens = ['', *alphabet, *(row['token'] for row in retained_rows)]
    if len(set(ordered_tokens)) != len(ordered_tokens):
        raise ValueError('projected tokens are not unique')

    adapter = copy.deepcopy(source_model)
    adapter['model_version'] = ADAPTER_POLICY_ID
    adapter['status'] = 'frozen-onhw-adapter'
    adapter['vocab'] = {token: index for index, token in enumerate(ordered_tokens)}
    adapter['idx_token'] = {
        str(index): token for index, token in enumerate(ordered_tokens)
    }
    adapter['size'] = len(ordered_tokens)
    adapter['eligible_bigram_count'] = len(retained_rows)
    adapter['vocabulary'] = retained_rows
    adapter['policy']['required_characters'] = list(alphabet)
    adapter['adapter'] = {
        'policy_id': ADAPTER_POLICY_ID,
        'frozen_date': ADAPTER_FROZEN_DATE,
        'task': 'OnHW-Words500 right-handed WD and WI',
        'source_model': {
            'schema_version': source_model.get('schema_version'),
            'model_version': source_model.get('model_version'),
            'sha256': source_sha256,
            'size': source_model.get('size'),
            'eligible_bigram_count': source_model.get('eligible_bigram_count'),
        },
        'projection': 'fixed-task-alphabet-membership-v1',
        'task_alphabet': list(alphabet),
        'task_alphabet_source': 'pre-existing TVA OnHW configuration',
        'annotation_files_read': False,
        'fallback_characters_absent_from_source': [
            char for char in alphabet if char not in source_model['vocab']
        ],
        'excluded_source_single_count': sum(
            len(token) == 1 and token not in allowed for token in source_model['vocab']
        ),
        'excluded_source_bigram_count': len(vocabulary) - len(retained_rows),
    }
    adapter['note'] = (
        'Frozen leakage-safe OnHW projection of the IAM+READ handwriting-bigram '
        'model. Bigram membership uses only the fixed task alphabet.'
    )
    _validate_mappings(adapter)
    return adapter


def build_frozen_onhw_adapter(source_path: str | Path) -> dict[str, Any]:
    """Load, authenticate, and project the pinned source artifact."""
    source_bytes = Path(source_path).read_bytes()
    source_sha256 = sha256_bytes(source_bytes)
    try:
        source_model = json.loads(source_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError('source model is not valid UTF-8 JSON') from error
    if not isinstance(source_model, dict):
        raise ValueError('source model root must be a JSON object')
    validate_frozen_source(source_model, source_sha256)
    adapter = project_source_model(source_model, source_sha256)

    if adapter['size'] != ADAPTER_SIZE:
        raise ValueError(
            f'adapter size mismatch: expected {ADAPTER_SIZE}, got {adapter["size"]}'
        )
    if adapter['eligible_bigram_count'] != ADAPTER_BIGRAM_COUNT:
        raise ValueError(
            'adapter bigram count mismatch: expected '
            f'{ADAPTER_BIGRAM_COUNT}, got {adapter["eligible_bigram_count"]}'
        )
    if adapter['adapter']['fallback_characters_absent_from_source'] != ['Ä', 'Ü']:
        raise ValueError('source fallback-character difference is not the audited pair')
    if adapter['adapter']['excluded_source_single_count'] != 34:
        raise ValueError('unexpected excluded source-single count')
    if adapter['adapter']['excluded_source_bigram_count'] != 43:
        raise ValueError('unexpected excluded source-bigram count')
    adapter_sha256 = sha256_bytes(canonical_json_bytes(adapter))
    if adapter_sha256 != ADAPTER_SHA256:
        raise ValueError(
            f'frozen adapter SHA-256 mismatch: expected {ADAPTER_SHA256}, '
            f'got {adapter_sha256}'
        )
    return adapter


def write_adapter(adapter: dict[str, Any], output_path: str | Path) -> str:
    """Write canonical adapter bytes and return their SHA-256 digest."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    content = canonical_json_bytes(adapter)
    output.write_bytes(content)
    return sha256_bytes(content)
