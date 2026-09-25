"""Authenticated ZIP-backed loader for the ED IMU dataset."""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any
import zipfile

from loguru import logger
import numpy as np
from tqdm import tqdm

from tva.ed_handwriting_bigram import ED_ALPHABET

from . import HRDataset
from .transforms import AddNoise, Drift, Dropout, TimeWarp


ED_ARCHIVE_SHA256 = {
    'wd': 'edaf78b0f422f719bbb13153249a5a6667a814d6f5a8d7d0d2ba02535ae55a3e',
    'wi': '0ba1b8e5d53c7208bbdc4a43cfb701ed6a1d27be382a23ba0aaade8cb3283304',
}
ED_ARCHIVE_STEM = {'wd': 'gold_wd', 'wi': 'gold_wi'}
ED_NUM_CHANNELS = 7
ED_SAMPLE_RATE = 100
ED_NUM_FOLDS = 5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=4)
def _inspect_archive(path_value: str, distribution: str) -> dict[str, Any]:
    """Authenticate one archive and cache its immutable metadata payloads."""
    path = Path(path_value)
    expected_digest = ED_ARCHIVE_SHA256[distribution]
    digest = _sha256(path)
    if digest != expected_digest:
        raise ValueError(
            f'ED {distribution.upper()} archive SHA-256 mismatch: '
            f'expected {expected_digest}, got {digest}'
        )
    stem = ED_ARCHIVE_STEM[distribution]
    with zipfile.ZipFile(path) as archive:
        names = frozenset(archive.namelist())
        payloads = {
            split: json.loads(archive.read(f'{stem}/{split}.json'))
            for split in ('train', 'val')
        }

    expected_split = {
        'wd': 'writer_dependent',
        'wi': 'writer_independent',
    }[distribution]
    for split, payload in payloads.items():
        info = payload.get('info', {})
        if info.get('num_channel') != ED_NUM_CHANNELS:
            raise ValueError(f'ED {distribution} {split} channel count mismatch')
        if info.get('rate_sample_target') != ED_SAMPLE_RATE:
            raise ValueError(f'ED {distribution} {split} sample rate mismatch')
        if info.get('num_fold') != ED_NUM_FOLDS:
            raise ValueError(f'ED {distribution} {split} fold count mismatch')
        if info.get('split') != expected_split:
            raise ValueError(f'ED {distribution} {split} distribution mismatch')
        if payload.get('categories') != list(ED_ALPHABET):
            raise ValueError(f'ED {distribution} {split} alphabet mismatch')
        annotations = payload.get('annotations')
        if set(annotations or {}) != {str(index) for index in range(ED_NUM_FOLDS)}:
            raise ValueError(f'ED {distribution} {split} fold keys mismatch')
        for fold_annotations in annotations.values():
            for annotation in fold_annotations:
                member = f'{stem}/{annotation["filename"]}'
                if member not in names:
                    raise ValueError(f'ED archive member is missing: {member}')
    return {
        'digest': digest,
        'stem': stem,
        'names': names,
        'payloads': payloads,
    }


def resolve_ed_dataset_directory(configured_directory: str | Path) -> Path:
    """Allow a local environment override without committing machine paths."""
    override = os.environ.get('TVA_ED_DATASET_DIR')
    return Path(override) if override else Path(configured_directory)


def resolve_ed_archive(
    configured_directory: str | Path,
    distribution: str,
) -> Path:
    if distribution not in ED_ARCHIVE_SHA256:
        raise ValueError('ED distribution must be "wd" or "wi"')
    directory = resolve_ed_dataset_directory(configured_directory)
    return directory / f'{ED_ARCHIVE_STEM[distribution]}.zip'


def get_ed_num_folds(
    configured_directory: str | Path,
    distribution: str,
) -> int:
    archive_path = resolve_ed_archive(configured_directory, distribution)
    _inspect_archive(str(archive_path.resolve()), distribution)
    return ED_NUM_FOLDS


class EdZipDataset(HRDataset):
    """Read one ED WD/WI fold directly from its authenticated ZIP archive."""

    def __init__(
        self,
        archive_path: str | Path,
        split: str,
        distribution: str,
        tokenizer: Any,
        ratio_ds: int,
        idx_fold: str | int,
        len_seq: int = 0,
        aug: bool = False,
        cache: bool = False,
        num_concat: int = 0,
        max_samples: int = 0,
    ) -> None:
        if split not in {'train', 'val'}:
            raise ValueError('ED dataset split must be "train" or "val"')
        fold = int(idx_fold)
        if fold not in range(ED_NUM_FOLDS):
            raise ValueError('ED fold index must be between 0 and 4')
        if distribution not in ED_ARCHIVE_SHA256:
            raise ValueError('ED distribution must be "wd" or "wi"')

        self.archive_path = Path(archive_path).resolve()
        inspected = _inspect_archive(str(self.archive_path), distribution)
        self.archive_digest = inspected['digest']
        self.archive_stem = inspected['stem']
        self.distribution = distribution
        self.split = split
        self.dir_ds = str(self.archive_path.parent)
        self.tokenizer = tokenizer
        self.ratio_ds = ratio_ds
        self.idx_fold = fold
        self.len_seq = len_seq
        self.cache = cache
        self.num_concat = num_concat
        self._archive: zipfile.ZipFile | None = None
        self._archive_pid: int | None = None

        self.augs = (
            [
                AddNoise(scale=0.05, kind='multiplicative'),
                Drift(0.1, 40, 'multiplicative'),
                Dropout(size=(5, 10), per_channel=True),
                TimeWarp(5, 4),
            ]
            if aug
            else None
        )
        self.annos = list(inspected['payloads'][split]['annotations'][str(fold)])
        if max_samples > 0:
            self.annos = self.annos[:max_samples]

        if self.num_concat > 0:
            self.indices_writer = defaultdict(list)
            for index, annotation in enumerate(self.annos):
                self.indices_writer[annotation['id_writer']].append(index)

        if cache:
            self.data_cache = [
                [
                    self._load_sequence(annotation),
                    self.tokenizer.encode(annotation['label']),
                ]
                for annotation in tqdm(self.annos)
            ]
            self.close()
            logger.info(
                f'Cached ED {distribution.upper()} {split} fold {fold} '
                f'from {self.archive_path}'
            )

    def _get_archive(self) -> zipfile.ZipFile:
        process_id = os.getpid()
        if self._archive is None or self._archive_pid != process_id:
            self.close()
            self._archive = zipfile.ZipFile(self.archive_path)
            self._archive_pid = process_id
        return self._archive

    def close(self) -> None:
        archive = getattr(self, '_archive', None)
        if archive is not None:
            archive.close()
        self._archive = None
        self._archive_pid = None

    def __del__(self) -> None:
        self.close()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state['_archive'] = None
        state['_archive_pid'] = None
        return state

    def _load_sequence(self, annotation: dict[str, Any]) -> np.ndarray:
        member = f'{self.archive_stem}/{annotation["filename"]}'
        content = self._get_archive().read(member)
        sequence = np.loadtxt(
            io.BytesIO(content), delimiter=';', dtype=np.float32
        )
        if sequence.ndim != 2 or sequence.shape[1] != ED_NUM_CHANNELS:
            raise ValueError(
                f'ED sample {annotation["id"]} does not have seven channels'
            )
        if not len(sequence) or not np.isfinite(sequence).all():
            raise ValueError(
                f'ED sample {annotation["id"]} contains invalid signal values'
            )
        return sequence

    def _get_raw_sample(self, idx: int) -> tuple[np.ndarray, list[int]]:
        if self.cache:
            return self.data_cache[idx]
        annotation = self.annos[idx]
        return (
            self._load_sequence(annotation),
            self.tokenizer.encode(annotation['label']),
        )
