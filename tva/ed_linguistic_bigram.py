"""Matched IAM-text linguistic Bigram comparator for ED."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle
import string
from typing import Any
import unicodedata

from tva.ed_handwriting_bigram import ED_ALPHABET, GREEDY_POLICY
from tva.handwriting_bigram_adapter import canonical_json_bytes, sha256_bytes


LABELS_SHA256 = '5ac34ad37ba0b125308fe1a2bc97095985e25dfa76495628c3bb3895c0b446ab'
SELECTION_SHA256 = '7893dfba4febe6df99cf0bdb0c74fabe7b736d2a6af05033d8638b90455bc1c2'
EVIDENCE_SCHEMA = 'tva.iam-letter-bigram-counts.v1'
EVIDENCE_SHA256 = '7808d5d7b58354be982b042c8f60db4d63bf2ee12aa03cbeca6c3fe628cd1ebb'

ADAPTER_SCHEMA = 'tva.ed-frequency-bigram-adapter.v1'
ADAPTER_POLICY_ID = 'ed-iam-frequency-bigram-greedy-v1'
ADAPTER_FROZEN_DATE = '2026-09-21'
ADAPTER_SIZE = 224
ADAPTER_BIGRAM_COUNT = 145
ADAPTER_SHA256 = '1da363e43fe9d300ece7ad5e3183d84a15fab4bdc6d5b2239b450c222b175abd'


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def build_iam_frequency_evidence(
    labels_path: str | Path,
    selection_path: str | Path,
) -> dict[str, Any]:
    """Count ASCII letter bigrams in the pinned complete IAM train split."""
    labels_path = Path(labels_path)
    selection_path = Path(selection_path)
    labels_digest = _file_sha256(labels_path)
    selection_digest = _file_sha256(selection_path)
    if labels_digest != LABELS_SHA256:
        raise ValueError('IAM labels.pkl SHA-256 mismatch')
    if selection_digest != SELECTION_SHA256:
        raise ValueError('IAM train selection SHA-256 mismatch')

    selection = json.loads(selection_path.read_bytes())
    if (
        selection.get('schema_version') != 'dtlr.iam-selection.v1'
        or selection.get('dataset') != 'IAM'
        or selection.get('split') != 'train'
        or selection.get('requested_count') != 5_694
        or selection.get('labels_sha256') != labels_digest
    ):
        raise ValueError('IAM selection manifest does not identify full IAM train')

    labels = pickle.loads(labels_path.read_bytes())
    train_rows = labels.get('ground_truth', {}).get('train')
    if not isinstance(train_rows, list) or len(train_rows) != 5_694:
        raise ValueError('IAM labels do not contain the expected train split')
    by_id = {row['id']: row for row in train_rows}
    if len(by_id) != len(train_rows):
        raise ValueError('IAM train contains duplicate line IDs')

    selected_rows = []
    for selected in selection.get('lines', []):
        row = by_id.get(selected.get('id'))
        if row is None:
            raise ValueError('selected IAM line is absent from labels.pkl')
        if selected.get('transcription_sha256') != _text_sha256(row['text']):
            raise ValueError('selected IAM transcription hash mismatch')
        selected_rows.append(row)
    if len(selected_rows) != 5_694:
        raise ValueError('IAM selection does not contain every training line')

    letters = set(string.ascii_letters)
    counts: Counter[str] = Counter()
    for row in selected_rows:
        text = unicodedata.normalize('NFC', row['text'])
        counts.update(
            text[index:index + 2]
            for index in range(len(text) - 1)
            if text[index] in letters and text[index + 1] in letters
        )
    rows = [
        {'token': token, 'count': count}
        for token, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    return {
        'schema_version': EVIDENCE_SCHEMA,
        'status': 'frozen-training-text-evidence',
        'dataset': 'IAM',
        'split': 'train',
        'text_normalization': 'NFC',
        'pair_policy': 'case-sensitive-ascii-letters-only',
        'counting_unit': 'adjacent-pair-occurrence-across-lines',
        'ranking': 'descending-count-then-lexical-token',
        'line_count': len(selected_rows),
        'candidate_bigram_count': len(rows),
        'total_candidate_occurrences': sum(counts.values()),
        'sources': {
            'labels_sha256': labels_digest,
            'selection_sha256': selection_digest,
            'selection_rule': selection['selection_rule'],
            'selection_seed': selection['seed'],
        },
        'counts': rows,
        'note': (
            'Frequency evidence uses IAM training transcripts only. No ED '
            'labels or recognition results were read.'
        ),
    }


def validate_evidence(evidence: dict[str, Any], digest: str) -> None:
    if digest != EVIDENCE_SHA256:
        raise ValueError('frozen IAM frequency-evidence SHA-256 mismatch')
    required = {
        'schema_version': EVIDENCE_SCHEMA,
        'status': 'frozen-training-text-evidence',
        'dataset': 'IAM',
        'split': 'train',
        'text_normalization': 'NFC',
        'pair_policy': 'case-sensitive-ascii-letters-only',
        'counting_unit': 'adjacent-pair-occurrence-across-lines',
        'ranking': 'descending-count-then-lexical-token',
        'line_count': 5_694,
        'candidate_bigram_count': 861,
    }
    for key, expected in required.items():
        if evidence.get(key) != expected:
            raise ValueError(f'IAM frequency evidence {key} mismatch')
    sources = evidence.get('sources')
    if not isinstance(sources, dict):
        raise ValueError('IAM frequency evidence lacks source provenance')
    if (
        sources.get('labels_sha256') != LABELS_SHA256
        or sources.get('selection_sha256') != SELECTION_SHA256
    ):
        raise ValueError('IAM frequency evidence source provenance mismatch')
    rows = evidence.get('counts')
    if not isinstance(rows, list) or len(rows) != 861:
        raise ValueError('IAM frequency evidence has invalid count rows')
    expected_order = sorted(
        rows, key=lambda row: (-row.get('count', -1), row.get('token', ''))
    )
    if rows != expected_order:
        raise ValueError('IAM frequency evidence rows are not canonically ranked')
    if len({row.get('token') for row in rows}) != len(rows):
        raise ValueError('IAM frequency evidence contains duplicate tokens')
    if any(
        not isinstance(row.get('token'), str)
        or len(row['token']) != 2
        or any(char not in string.ascii_letters for char in row['token'])
        or isinstance(row.get('count'), bool)
        or not isinstance(row.get('count'), int)
        or row['count'] <= 0
        for row in rows
    ):
        raise ValueError('IAM frequency evidence contains an invalid count row')
    if evidence.get('total_candidate_occurrences') != sum(
        row['count'] for row in rows
    ):
        raise ValueError('IAM frequency evidence total count mismatch')


def build_ed_linguistic_adapter(evidence_path: str | Path) -> dict[str, Any]:
    """Build the matched 224-class comparator from frozen IAM text counts."""
    content = Path(evidence_path).read_bytes()
    digest = sha256_bytes(content)
    try:
        evidence = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError('IAM frequency evidence is not valid UTF-8 JSON') from error
    if not isinstance(evidence, dict):
        raise ValueError('IAM frequency evidence root must be an object')
    validate_evidence(evidence, digest)

    selected = evidence['counts'][:ADAPTER_BIGRAM_COUNT]
    maximum_count = selected[0]['count']
    rows = [
        {
            'token': row['token'],
            'iam_training_occurrence_count': row['count'],
            'rank': rank,
            'utility': row['count'] / maximum_count,
        }
        for rank, row in enumerate(selected, start=1)
    ]
    ordered_tokens = ['', *ED_ALPHABET, *(row['token'] for row in rows)]
    if len(set(ordered_tokens)) != len(ordered_tokens):
        raise ValueError('linguistic adapter tokens are not unique')
    adapter = {
        'schema_version': ADAPTER_SCHEMA,
        'model_version': ADAPTER_POLICY_ID,
        'status': 'frozen-matched-comparator',
        'dataset': 'ED',
        'text_normalization': 'NFC',
        'blank_token': '',
        'blank_id': 0,
        'policy': {
            'vocabulary_selection': 'iam-train-adjacent-pair-frequency-v1',
            'overlap_resolution': GREEDY_POLICY,
            'single_character_utility': 0.0,
            'required_characters': list(ED_ALPHABET),
            'selection_count': ADAPTER_BIGRAM_COUNT,
            'tie_breaking': 'descending-count-then-lexical-token',
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
            'task': 'ED sentence recognition, WD and WI',
            'frequency_evidence': {
                'schema_version': EVIDENCE_SCHEMA,
                'sha256': digest,
                'dataset': 'IAM',
                'split': 'train',
                'line_count': evidence['line_count'],
                'labels_sha256': LABELS_SHA256,
                'selection_sha256': SELECTION_SHA256,
            },
            'projection': 'declared-task-alphabet-v1',
            'task_alphabet': list(ED_ALPHABET),
            'iam_training_transcripts_read': True,
            'annotation_label_values_read_by_builder': False,
        },
        'note': (
            'Matched linguistic comparator selected from IAM training-text '
            'frequency only. ED labels and recognition results did not select '
            'or rank tokens.'
        ),
    }
    if adapter['size'] != ADAPTER_SIZE or len(rows) != ADAPTER_BIGRAM_COUNT:
        raise ValueError('matched comparator does not have the required dimensions')
    adapter_digest = sha256_bytes(canonical_json_bytes(adapter))
    if ADAPTER_SHA256 and adapter_digest != ADAPTER_SHA256:
        raise ValueError('frozen ED linguistic adapter SHA-256 mismatch')
    return adapter


def write_json(value: dict[str, Any], output_path: str | Path) -> str:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    content = canonical_json_bytes(value)
    output.write_bytes(content)
    return sha256_bytes(content)
