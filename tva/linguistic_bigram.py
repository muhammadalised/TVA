"""Build matched, training-only linguistic Bigram tokenizers."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import unicodedata
from typing import Any

from tva.handwriting_bigram_adapter import (
    ADAPTER_BIGRAM_COUNT,
    ADAPTER_SIZE,
    ONHW_ALPHABET,
    canonical_json_bytes,
    sha256_bytes,
)


LINGUISTIC_SCHEMA = 'tva.frequency-bigram-tokenizer.v1'
LINGUISTIC_MODEL_VERSION = 'onhw-frequency-bigram-v1'
SELECTION_POLICY = 'training-word-type-adjacency-frequency-v1'
MANIFEST_SHA256 = {
    'onhw_words500_wd_word_rh': (
        'c6895783d3da442e80ddf169be2e6397839ff2b0158df19b2c6ecb1bc40a2faa'
    ),
    'onhw_words500_wi_word_rh': (
        '605b0f627d3044340111d7ee0b6ba364790b51130d431640d8eca3e1cc9bb392'
    ),
}


def _normalized_training_labels(annotations: list[dict[str, Any]]) -> list[str]:
    labels = []
    for annotation in annotations:
        label = annotation.get('label')
        if not isinstance(label, str) or not label:
            raise ValueError('every training annotation must have a non-empty label')
        normalized = unicodedata.normalize('NFC', label)
        unknown = sorted(set(normalized) - set(ONHW_ALPHABET))
        if unknown:
            raise ValueError(
                'training label contains characters outside the task alphabet: '
                f'{unknown!r}'
            )
        labels.append(normalized)
    return labels


def build_fold_model(
    annotations: list[dict[str, Any]],
    *,
    dataset: str,
    fold: int,
    source_sha256: str,
) -> dict[str, Any]:
    """Build one 419-class baseline from one fold's training annotations."""
    if not annotations:
        raise ValueError('training annotation fold is empty')
    labels = _normalized_training_labels(annotations)
    word_types = sorted(set(labels))
    adjacency_count: Counter[str] = Counter()
    word_type_support: Counter[str] = Counter()
    for label in word_types:
        pairs = [label[index:index + 2] for index in range(len(label) - 1)]
        adjacency_count.update(pairs)
        word_type_support.update(set(pairs))

    ranked_pairs = sorted(
        adjacency_count,
        key=lambda pair: (-adjacency_count[pair], pair),
    )
    if len(ranked_pairs) < ADAPTER_BIGRAM_COUNT:
        raise ValueError(
            f'fold has only {len(ranked_pairs)} candidate bigrams; '
            f'{ADAPTER_BIGRAM_COUNT} are required'
        )
    selected = ranked_pairs[:ADAPTER_BIGRAM_COUNT]
    maximum_count = adjacency_count[selected[0]]
    rows = [
        {
            'token': pair,
            'rank': rank,
            'training_word_type_adjacency_count': adjacency_count[pair],
            'training_word_type_support': word_type_support[pair],
            'utility': adjacency_count[pair] / maximum_count,
        }
        for rank, pair in enumerate(selected, start=1)
    ]
    ordered_tokens = ['', *ONHW_ALPHABET, *selected]
    vocab = {token: index for index, token in enumerate(ordered_tokens)}
    idx_token = {str(index): token for token, index in vocab.items()}
    model = {
        'schema_version': LINGUISTIC_SCHEMA,
        'model_version': LINGUISTIC_MODEL_VERSION,
        'status': 'frozen-matched-linguistic-baseline',
        'dataset': dataset,
        'training_split': 'train',
        'fold': fold,
        'text_normalization': 'NFC',
        'blank_token': '',
        'blank_id': 0,
        'source_annotations': {
            'sha256': source_sha256,
            'sample_count': len(labels),
            'unique_word_type_count': len(word_types),
        },
        'policy': {
            'selection': SELECTION_POLICY,
            'sample_duplicates_collapsed': True,
            'ranking': 'adjacency-count-descending_then-token-ascending-v1',
            'utility': 'adjacency-count-divided-by-maximum-selected-count-v1',
            'pair_policy': 'fixed-task-alphabet-only',
            'target_bigram_count': ADAPTER_BIGRAM_COUNT,
            'overlap_resolution': 'maximum-total-utility-non-overlapping-v1',
            'single_character_utility': 0.0,
            'required_characters': list(ONHW_ALPHABET),
            'validation_annotations_read': False,
        },
        'candidate_bigram_count': len(ranked_pairs),
        'selection_cutoff_count': adjacency_count[selected[-1]],
        'vocab': vocab,
        'idx_token': idx_token,
        'size': len(vocab),
        'eligible_bigram_count': len(rows),
        'vocabulary': rows,
        'note': (
            'Matched linguistic baseline built from distinct word types in this '
            'fold training split only. Validation labels are excluded.'
        ),
    }
    if model['size'] != ADAPTER_SIZE:
        raise ValueError(f'linguistic tokenizer size is {model["size"]}, not {ADAPTER_SIZE}')
    return model


def build_dataset_models(
    train_path: str | Path,
    *,
    dataset: str,
) -> dict[int, dict[str, Any]]:
    """Build every fold while accepting only a file named ``train.json``."""
    path = Path(train_path)
    if path.name != 'train.json':
        raise ValueError('linguistic vocabularies must be built from train.json only')
    content = path.read_bytes()
    try:
        source = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError('training annotations are not valid UTF-8 JSON') from error
    if not isinstance(source, dict) or not isinstance(source.get('annotations'), dict):
        raise ValueError('training annotation file has an invalid structure')
    info = source.get('info')
    if not isinstance(info, dict) or not isinstance(info.get('num_fold'), int):
        raise ValueError('training annotation file lacks num_fold metadata')
    folds = source['annotations']
    expected_keys = {str(fold) for fold in range(info['num_fold'])}
    if set(folds) != expected_keys:
        raise ValueError('training annotation folds do not match num_fold metadata')
    source_sha256 = sha256_bytes(content)
    return {
        fold: build_fold_model(
            folds[str(fold)],
            dataset=dataset,
            fold=fold,
            source_sha256=source_sha256,
        )
        for fold in range(info['num_fold'])
    }


def write_dataset_models(
    models: dict[int, dict[str, Any]],
    output_directory: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write fold artifacts and a checksum manifest."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paths = {fold: output / f'{fold}.json' for fold in sorted(models)}
    manifest_path = output / 'manifest.json'
    existing = [path for path in [*paths.values(), manifest_path] if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f'output already exists: {existing[0]}; pass overwrite=True to replace it'
        )

    artifacts = {}
    for fold, path in paths.items():
        content = canonical_json_bytes(models[fold])
        path.write_bytes(content)
        artifacts[str(fold)] = {
            'filename': path.name,
            'sha256': sha256_bytes(content),
            'size': models[fold]['size'],
            'eligible_bigram_count': models[fold]['eligible_bigram_count'],
            'source_annotations_sha256': models[fold]['source_annotations']['sha256'],
        }
    manifest = {
        'schema_version': 'tva.frequency-bigram-manifest.v1',
        'model_version': LINGUISTIC_MODEL_VERSION,
        'dataset': models[min(models)]['dataset'],
        'fold_count': len(models),
        'artifacts': artifacts,
    }
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    return manifest
