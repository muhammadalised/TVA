"""IAM training transcriptions in TVA's dataset format.

IAM is only used to learn tokenizer vocabularies, so only the labels are read.
"""

import hashlib
import json
import os
from pathlib import Path
import pickle

from loguru import logger

LABELS_SHA256 = '5ac34ad37ba0b125308fe1a2bc97095985e25dfa76495628c3bb3895c0b446ab'
SELECTION_SHA256 = '7893dfba4febe6df99cf0bdb0c74fabe7b736d2a6af05033d8638b90455bc1c2'
DEFAULT_LABELS = Path('/home/artellisys/DTLR/data/IAM_new/labels.pkl')
DEFAULT_SELECTION = Path('/home/artellisys/dtlr-output/iam-train-full/selection.json')


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_iam_training(labels_path, selection_path) -> list[dict]:
    '''Reads all 5,694 IAM training lines in the frozen selection order.

    Args:
        labels_path: DTLR's IAM labels.pkl.
        selection_path: selection.json listing every training line and the
            SHA-256 of its transcription.

    Returns:
        List of {'id', 'label'} annotations.
    '''
    labels_bytes = Path(labels_path).read_bytes()
    selection_bytes = Path(selection_path).read_bytes()
    _require(_sha256(labels_bytes) == LABELS_SHA256, 'IAM labels SHA-256 mismatch')
    _require(_sha256(selection_bytes) == SELECTION_SHA256, 'IAM selection SHA-256 mismatch')
    selection = json.loads(selection_bytes)
    train_rows = pickle.loads(labels_bytes)['ground_truth']['train']
    by_id = {row['id']: row for row in train_rows}
    selected_ids = [line['id'] for line in selection['lines']]
    _require(
        len(train_rows) == len(by_id) == len(selected_ids) == 5694
        and set(selected_ids) == set(by_id),
        'Selection must cover every IAM training line exactly once',
    )

    annotations = []
    for line in selection['lines']:
        text = by_id[line['id']]['text']
        _require(
            _sha256(text.encode('utf-8')) == line['transcription_sha256'],
            f'Transcription hash mismatch: {line["id"]}',
        )
        annotations.append({'id': line['id'], 'label': text})
    return annotations


def prepare_iam(labels_path, selection_path, dir_ds: str) -> None:
    '''Writes the IAM training lines as a one-fold TVA dataset.

    The resulting 'train.json' is what TVA's tokenizer .train() methods read.

    Args:
        labels_path: DTLR's IAM labels.pkl.
        selection_path: Frozen IAM training selection.json.
        dir_ds: Output dataset directory.
    '''
    annotations = read_iam_training(labels_path, selection_path)
    os.makedirs(dir_ds, exist_ok=True)
    with open(os.path.join(dir_ds, 'train.json'), 'w', encoding='utf-8') as f:
        json.dump(
            {'info': {'num_fold': 1}, 'annotations': {'0': annotations}},
            f,
            ensure_ascii=False,
            indent=2,
        )

    logger.info(f'{len(annotations)} IAM training lines are saved at {dir_ds}.')
