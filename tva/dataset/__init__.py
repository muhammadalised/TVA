import json
import os
from typing import Any
from collections import defaultdict

import numpy as np
import torch
from loguru import logger
from torch.nn.functional import pad
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm

from .transforms import AddNoise, Drift, Dropout, TimeWarp

__all__ = ['fn_collate', 'HRDataset']


def fn_collate(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    '''Collate function for aligning the shape of IMU sequences and labels.

    This function pads sequences and labels to the maximum length within the
    batch. IMU sequences are permuted to match typical 1D-convolutional
    input shapes.

    Args:
        batch: List of tuples containing (sequence, label) tensors.

    Returns:
        A tuple of four tensors:
        - seqs (torch.Tensor): Padded IMU sequences.
            Shape: (batch_size, num_channels, max_seq_len).
        - labels (torch.Tensor): Padded token labels.
            Shape: (batch_size, max_label_len).
        - lens_sig (torch.Tensor): Original lengths of IMU signals.
        - lens_label (torch.Tensor): Original lengths of labels.
    '''
    seqs, labels, lens_seq, lens_label = [], [], [], []

    for x, y in batch:
        seqs.append(x)
        labels.append(y)
        lens_seq.append(len(x))
        lens_label.append(len(y))

    # [batch, len, chan] -> [batch, chan, len]
    seqs = pad_sequence(seqs, True).permute(0, 2, 1)
    labels = pad_sequence(labels, True)
    lens_seq = torch.tensor(lens_seq)
    lens_label = torch.tensor(lens_label)

    return seqs, labels, lens_seq, lens_label


class HRDataset(Dataset):
    '''Dataset for IMU-based handwriting recognition.

    Handles loading, augmenting, and normalizing inertial sensor data
    sequences and their corresponding text labels.

    Args:
        path_anno: Path to the annotation file.
        tokenizer: Tokenizer instance for label encoding.
        ratio_ds: Downsampling ratio of the target model.
        idx_fold: Fold index for cross validation.
        len_seq: Fixed length for padding sequences. Defaults to 0.
        aug: Whether to apply online data augmentation. Defaults to False.
        cache: Whether to cache loaded data in memory. Defaults to False.
        num_concat: Number of additional samples to concatenate. Defaults to 0.

    Attributes:
        dir_ds: Root directory of the dataset.
        tokenizer: Tokenizer for text processing.
        ratio_ds: Model downsampling ratio.
        idx_fold: Current cross-validation fold.
        len_seq: Target sequence length for padding.
        cache: Memory caching flag.
        num_concat: Number of additional samples to concatenate.
        augs: List of active augmentation transforms.
        annos: List of annotation entries for the fold.
        data_cache: In-memory storage for cached data.
        writer_indices: Mapping of writer IDs to sample indices.
    '''

    def __init__(
        self,
        path_anno: str,
        tokenizer: Any,
        ratio_ds: int,
        idx_fold: str | int,
        len_seq: int = 0,
        aug: bool = False,
        cache: bool = False,
        num_concat: int = 0,
    ) -> None:
        self.dir_ds = os.path.dirname(path_anno)
        self.tokenizer = tokenizer
        self.ratio_ds = ratio_ds
        self.idx_fold = idx_fold
        self.len_seq = len_seq
        self.cache = cache
        self.num_concat = num_concat

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

        with open(path_anno, 'r') as f:
            annos = json.load(f)
            self.annos = annos['annotations'][str(idx_fold)]

        # group indices by writer for concatenation
        if self.num_concat > 0:
            self.indices_writer = defaultdict(list)

            for idx, anno in enumerate(self.annos):
                self.indices_writer[anno['id_writer']].append(idx)

        if cache:
            self.data_cache = [
                [
                    np.loadtxt(
                        os.path.join(self.dir_ds, anno['filename']),
                        delimiter=';',
                        dtype=np.float32,
                    ),
                    self.tokenizer.encode(anno['label']),
                ]
                for anno in tqdm(self.annos)
            ]
            logger.info(f'Cached dataset {path_anno}')

    def __len__(self) -> int:
        '''Returns the total number of samples in the fold.'''
        return len(self.annos)

    def _get_raw_sample(self, idx: int) -> tuple[np.ndarray, list[int]]:
        '''Fetches raw sequence and label (list of tokens) for a given index.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Tuple of raw numpy sequence and list of token ids.
        '''
        if self.cache:
            return self.data_cache[idx]

        anno = self.annos[idx]
        seq = np.loadtxt(
            os.path.join(self.dir_ds, anno['filename']),
            delimiter=';',
            dtype=np.float32,
        )
        label = self.tokenizer.encode(anno['label'])

        return seq, label

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        '''Fetches and processes a single IMU-label pair.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Tuple containing the processed IMU signal tensor and label tensor.
        '''
        indices = [idx]

        if self.num_concat > 0:
            id_writer = self.annos[idx]['id_writer']
            indices_candidate = self.indices_writer[id_writer]
            indices_aug = np.random.choice(
                indices_candidate, self.num_concat
            ).tolist()
            indices = (
                indices_aug[: self.num_concat // 2]
                + indices
                + indices_aug[self.num_concat // 2 :]
            )

        seq, label = [], []

        for i in indices:
            s, l = self._get_raw_sample(i)
            seq.append(s)
            label.extend(l)

        # concatenate sequences along time axis
        seq = np.concatenate(seq, axis=0)

        # label pre-processing
        label = torch.tensor(label, dtype=torch.int32)

        # sequence pre-processing
        seq = self._process(seq, len(label))

        return seq, label

    def _process(self, seq: np.ndarray, len_label: int) -> torch.Tensor:
        '''Applies augmentation, normalization, and padding to the signal.

        Normalization: seq_norm = (seq - mean) / (std + eps)

        Args:
            seq: Raw inertial sensor data.
            len_label: Length of the corresponding token label.

        Returns:
            Processed and padded signal tensor.
        '''
        # data augmentation
        if self.augs is not None:
            for aug in self.augs:
                if np.random.random() < 0.25:
                    seq = aug(seq)

        # normalize
        seq = (seq - np.mean(seq, 0)) / (np.std(seq, 0) + 1e-6)
        seq = torch.from_numpy(seq).to(torch.float32)

        # padding to fixed length
        if self.len_seq and len(seq) < self.len_seq:
            seq = pad(seq.T, (0, self.len_seq - len(seq))).T

        # ensure sequence length is compatible with CTC loss constraints
        if len(seq) < (len_min := len_label * 2 * self.ratio_ds):
            seq = pad(seq.T, (0, len_min - len(seq))).T

        return seq
