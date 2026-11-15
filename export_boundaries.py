'''Export alignment and sensor features for every character boundary.'''

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from tva.alignment_analysis import analyze_alignment
from tva.alignment_runtime import load_config, load_model, token_text
from tva.boundary_export import (
    BoundaryJSONLWriter,
    load_boundary_rows,
    make_boundary_rows,
)
from tva.boundary_features import extract_boundary_features
from tva.ctc_alignment import ctc_viterbi_align
from tva.dataset import HRDataset, fn_collate
from tva.decoder_ctc import BestPath
from tva.tokenizers import get_tokenizer


class IndexedDataset(Dataset):
    '''Return the original sample index along with each dataset item.'''

    def __init__(self, dataset: HRDataset, indices: list[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[int, torch.Tensor, torch.Tensor]:
        sample_index = self.indices[index]
        signal, target = self.dataset[sample_index]
        return sample_index, signal, target


def collate_indexed(
    batch: list[tuple[int, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    '''Pad a batch while keeping its original dataset indices.'''
    indices, signals, targets = zip(*batch)
    x, y, signal_lengths, target_lengths = fn_collate(
        list(zip(signals, targets))
    )
    return torch.tensor(indices), x, y, signal_lengths, target_lengths


def default_output_path(
    config_path: str,
    fold: int | str,
    split: str,
) -> Path:
    return (
        Path('results/thesis/boundary_exports')
        / Path(config_path).stem
        / f'fold{fold}'
        / split
        / 'boundaries.jsonl'
    )


def write_summary(
    output_path: Path,
    metadata: dict[str, Any],
    pair_counts: Counter[str],
    completed_samples: int,
    boundary_count: int,
) -> Path:
    '''Write a compact, human-readable description of the export.'''
    summary_path = output_path.with_name('summary.json')
    payload = dict(metadata)
    payload.update(
        {
            'completed_samples': completed_samples,
            'exported_boundaries': boundary_count,
            'unique_pairs': len(pair_counts),
            'pair_counts': dict(sorted(pair_counts.items())),
        }
    )
    temporary = summary_path.with_name(summary_path.name + '.tmp')
    with open(temporary, 'w', encoding='utf-8') as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    os.replace(temporary, summary_path)
    return summary_path


def export_boundaries(args: argparse.Namespace) -> None:
    '''Run batched inference and export one JSONL row per boundary.'''
    config = load_config(args.config)
    if config.tokenizer != 'char':
        raise ValueError('Boundary export requires a character-model config.')

    device = torch.device(args.device or config.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(
            'CUDA was requested but is not available. Use --device cpu or '
            'check the CUDA-enabled PyTorch installation.'
        )

    tokenizer = get_tokenizer(config.tokenizer)
    tokenizer.load(
        os.path.join(config.dir_tokenizer, f'{config.idx_fold}.json')
    )
    model, checkpoint_epoch = load_model(
        config, tokenizer, args.checkpoint, device
    )
    decoder = BestPath(tokenizer)

    annotation_path = os.path.join(config.dir_dataset, f'{args.split}.json')
    with open(annotation_path, 'r', encoding='utf-8') as file:
        dataset_document = json.load(file)
    sample_rate_hz = float(dataset_document['info']['rate_sample_target'])

    dataset = HRDataset(
        annotation_path,
        tokenizer,
        model.ratio_ds,
        config.idx_fold,
        config.len_seq,
        aug=False,
        cache=False,
    )
    all_indices = list(range(len(dataset)))
    if args.max_samples > 0:
        all_indices = all_indices[: args.max_samples]

    output_path = Path(args.output) if args.output else default_output_path(
        args.config, config.idx_fold, args.split
    )
    batch_size = args.batch_size or config.size_batch
    num_workers = (
        args.num_workers
        if args.num_workers is not None
        else config.num_worker
    )

    added_boundaries = 0

    with BoundaryJSONLWriter(
        output_path,
        resume=args.resume,
        overwrite=args.overwrite,
    ) as writer:
        indices = [
            index
            for index in all_indices
            if index not in writer.completed_samples
        ]
        indexed_dataset = IndexedDataset(dataset, indices)
        dataloader = DataLoader(
            indexed_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_indexed,
        )

        progress = tqdm(
            total=len(indices),
            desc='Exporting samples',
            unit='sample',
        )
        with torch.no_grad():
            for (
                sample_indices,
                batch_signal,
                batch_target,
                signal_lengths,
                target_lengths,
            ) in dataloader:
                batch_probabilities = model(batch_signal.to(device)).cpu()

                for position, sample_index_tensor in enumerate(sample_indices):
                    sample_index = int(sample_index_tensor)
                    signal_length = int(signal_lengths[position])
                    target_length = int(target_lengths[position])
                    num_frames = signal_length // model.ratio_ds

                    probabilities = batch_probabilities[
                        position, :num_frames
                    ]
                    signal = batch_signal[
                        position, :, :signal_length
                    ].transpose(0, 1)
                    target = batch_target[position, :target_length]
                    annotation = dataset.annos[sample_index]

                    raw_signal = np.loadtxt(
                        os.path.join(
                            config.dir_dataset, annotation['filename']
                        ),
                        delimiter=';',
                        dtype=np.float32,
                    )
                    alignment = ctc_viterbi_align(
                        probabilities, target, blank_id=0
                    )
                    analysis = analyze_alignment(
                        alignment=alignment,
                        probabilities=probabilities,
                        token_to_text=lambda token_id: token_text(
                            tokenizer, token_id
                        ),
                        downsampling_ratio=model.ratio_ds,
                        num_model_input_samples=signal_length,
                        num_raw_samples=raw_signal.shape[0],
                        sample_rate_hz=sample_rate_hz,
                    )
                    features = extract_boundary_features(
                        raw_signal=raw_signal,
                        normalized_signal=signal.numpy(),
                        alignment_analysis=analysis,
                    )
                    known_label = tokenizer.decode(target.tolist())
                    greedy_prediction = decoder.decode(probabilities)
                    rows = make_boundary_rows(
                        config_path=args.config,
                        checkpoint_path=args.checkpoint,
                        checkpoint_epoch=checkpoint_epoch,
                        split=args.split,
                        fold=config.idx_fold,
                        sample_index=sample_index,
                        annotation=annotation,
                        known_label=known_label,
                        greedy_prediction=greedy_prediction,
                        raw_num_samples=raw_signal.shape[0],
                        model_input_num_samples=signal_length,
                        alignment=alignment,
                        analysis=analysis,
                        features=features,
                    )
                    writer.write_sample(sample_index, rows)
                    added_boundaries += len(rows)
                    progress.update(1)
        progress.close()

        completed_samples = len(writer.completed_samples)

    all_rows = load_boundary_rows(output_path)
    pair_counts = Counter(row['pair'] for row in all_rows)
    metadata = {
        'schema_version': 1,
        'config': args.config,
        'checkpoint': args.checkpoint,
        'checkpoint_epoch': checkpoint_epoch,
        'split': args.split,
        'fold': config.idx_fold,
        'device': str(device),
        'batch_size': batch_size,
        'num_workers': num_workers,
        'requested_samples': len(all_indices),
        'resumed': args.resume,
    }
    summary_path = write_summary(
        output_path,
        metadata,
        pair_counts,
        completed_samples,
        len(all_rows),
    )

    print('\nBoundary export complete')
    print(f'  samples complete: {completed_samples}')
    print(f'  boundaries total: {len(all_rows)}')
    print(f'  boundaries added this run: {added_boundaries}')
    print(f'  JSONL:   {output_path}')
    print(f'  summary: {summary_path}')
    print(f'  progress: {output_path.with_suffix(output_path.suffix + ".progress")}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Export alignment, force, motion, and reliability measurements '
            'for every adjacent-character boundary.'
        )
    )
    parser.add_argument(
        '-c', '--config', required=True,
        help='Character-model YAML configuration.',
    )
    parser.add_argument(
        '--checkpoint', required=True,
        help='Trained character-model checkpoint.',
    )
    parser.add_argument(
        '--split', choices=('train', 'val'), default='train',
        help='Dataset split to export. Defaults to train.',
    )
    parser.add_argument(
        '--device', choices=('cpu', 'cuda'),
        help='Override the device from the YAML configuration.',
    )
    parser.add_argument(
        '--batch-size', type=int, default=1,
        help=(
            'Inference batch size. Defaults to 1 so bidirectional-model '
            'outputs cannot depend on padding from neighbouring samples.'
        ),
    )
    parser.add_argument(
        '--num-workers', type=int,
        help='DataLoader workers. Defaults to the configuration value.',
    )
    parser.add_argument(
        '--max-samples', type=int, default=0,
        help='Process only the first N samples; 0 means the complete fold.',
    )
    parser.add_argument(
        '--output',
        help='Exact JSONL output path. A thesis results path is used by default.',
    )
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument(
        '--resume', action='store_true',
        help='Continue an interrupted export and repair unfinished rows.',
    )
    output_mode.add_argument(
        '--overwrite', action='store_true',
        help='Replace an existing export and start from sample zero.',
    )
    args = parser.parse_args()

    if args.batch_size is not None and args.batch_size <= 0:
        parser.error('--batch-size must be positive.')
    if args.num_workers is not None and args.num_workers < 0:
        parser.error('--num-workers cannot be negative.')
    if args.max_samples < 0:
        parser.error('--max-samples cannot be negative.')

    return args


if __name__ == '__main__':
    export_boundaries(parse_args())
