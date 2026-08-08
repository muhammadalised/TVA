'''Run target-constrained CTC alignment for one handwriting sample.'''

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from tva.alignment_analysis import AlignmentAnalysis, analyze_alignment
from tva.alignment_plot import plot_alignment
from tva.boundary_features import (
    BoundaryFeatureAnalysis,
    extract_boundary_features,
)
from tva.ctc_alignment import CTCAlignment, ctc_viterbi_align
from tva.dataset import HRDataset
from tva.decoder_ctc import BestPath
from tva.model import BaseModel
from tva.tokenizers import get_tokenizer


def load_config(path: str) -> argparse.Namespace:
    '''Load the same YAML configuration used to train the model.'''
    with open(path, 'r', encoding='utf-8') as file:
        return argparse.Namespace(**yaml.safe_load(file))


def load_model(
    config: argparse.Namespace,
    tokenizer: Any,
    checkpoint_path: str,
    device: torch.device,
) -> tuple[BaseModel, int | None]:
    '''Build the character model and restore its trained weights.'''
    model = BaseModel(
        config.arch_en,
        config.arch_de,
        config.num_channel,
        tokenizer.size,
        config.len_seq,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location='cpu',
        weights_only=False,
    )

    # New checkpoints contain training state under named keys. Supporting a
    # plain state dictionary also keeps the runner useful for older weights.
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model_state = checkpoint['model']
        checkpoint_epoch = checkpoint.get('epoch')
    else:
        model_state = checkpoint
        checkpoint_epoch = None

    model.load_state_dict(model_state, strict=True)
    model.to(device)
    model.eval()

    return model, checkpoint_epoch


def token_text(tokenizer: Any, token_id: int) -> str:
    '''Convert one token ID to readable text, including the blank token.'''
    text = tokenizer.decode([token_id])
    return text if text else '<blank>'


def print_alignment(
    analysis: AlignmentAnalysis,
) -> None:
    '''Print character confidence and candidate boundary regions.'''
    print('\nCharacter anchors and confidence')
    print('index char frames  input_anchor aligned preferred margin  agrees')
    print('----- ---- ------- ------------ ------- --------- ------- -------')

    for character in analysis.characters:
        frames = (
            f'{character.model_start_frame}:{character.model_end_frame}'
        )
        print(
            f'{character.target_index:>5} '
            f'{character.text:^4} '
            f'{frames:>7} '
            f'{character.anchor_input_sample:>12.1f} '
            f'{character.aligned_probability:>7.3f} '
            f'{character.preferred_text:^9} '
            f'{character.confidence_margin:>7.3f} '
            f'{"yes" if character.agrees_with_greedy else "NO":>7}'
        )

    print('\nCandidate boundary regions')
    print('index pair model_frames input_samples blank_frames duration_ms')
    print('----- ---- ------------ ------------- ------------ -----------')

    for boundary in analysis.boundaries:
        model_frames = (
            f'{boundary.model_start_frame}:{boundary.model_end_frame}'
        )
        input_samples = (
            f'{boundary.input_start_sample}:{boundary.input_end_sample}'
        )
        print(
            f'{boundary.boundary_index:>5} '
            f'{boundary.pair:^4} '
            f'{model_frames:>12} '
            f'{input_samples:>13} '
            f'{boundary.num_blank_frames:>12} '
            f'{boundary.duration_ms:>11.1f}'
        )


def save_analysis(
    args: argparse.Namespace,
    config: argparse.Namespace,
    annotation: dict[str, Any],
    dataset_info: dict[str, Any],
    checkpoint_epoch: int | None,
    known_label: str,
    greedy_prediction: str,
    alignment: CTCAlignment,
    analysis: AlignmentAnalysis,
    boundary_features: BoundaryFeatureAnalysis,
) -> tuple[Path, Path]:
    '''Save reusable JSON metadata and return JSON/PNG output paths.'''
    output_dir = (
        Path(args.output_dir)
        / Path(args.config).stem
        / f'fold{config.idx_fold}'
        / args.split
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = f'sample_{args.sample_index:05d}'
    path_json = output_dir / f'{output_stem}.json'
    path_plot = output_dir / f'{output_stem}.png'

    payload = {
        'schema_version': 1,
        'sample': {
            'config': args.config,
            'checkpoint': args.checkpoint,
            'checkpoint_epoch': checkpoint_epoch,
            'split': args.split,
            'fold': config.idx_fold,
            'sample_index': args.sample_index,
            'source_file': annotation['filename'],
            'writer_id': annotation['id_writer'],
            'known_label': known_label,
            'greedy_prediction': greedy_prediction,
        },
        'dataset': {
            'sample_rate_hz': dataset_info['rate_sample_target'],
            'sensor_groups': dataset_info['sensors'],
            'channel_indices': dataset_info['idxs_channel'],
        },
        'alignment': {
            'log_score': alignment.log_score,
            'mean_log_score': (
                alignment.log_score / analysis.num_model_frames
            ),
            'expanded_target': alignment.expanded_target,
            'state_path': alignment.state_path,
            'token_path': alignment.token_path,
        },
        'analysis': asdict(analysis),
        'boundary_feature_analysis': asdict(boundary_features),
    }

    with open(path_json, 'w', encoding='utf-8') as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    return path_json, path_plot


def print_boundary_features(
    features: BoundaryFeatureAnalysis,
    window_ms: int = 100,
) -> None:
    '''Print the most interpretable features for one selected window size.'''
    print(f'\nBoundary features ({window_ms} ms midpoint window)')
    print('pair center force_min_rel low_force longest_low motion_dE margin agree')
    print('---- ------ ------------- --------- ----------- --------- ------ -----')

    for boundary in features.boundaries:
        window = next(
            item for item in boundary.windows if item.window_ms == window_ms
        )
        print(
            f'{boundary.pair:^4} '
            f'{boundary.center_input_sample:>6.1f} '
            f'{window.force_min_relative:>9.3f} '
            f'{window.low_force_fraction:>9.3f} '
            f'{window.longest_low_force_ms:>11.1f} '
            f'{window.motion_derivative_energy:>9.3f} '
            f'{boundary.minimum_confidence_margin:>10.3f} '
            f'{"yes" if boundary.both_anchors_agree_with_greedy else "NO":>5}'
        )


def align_one_sample(args: argparse.Namespace) -> CTCAlignment:
    '''Load one sample, run inference, align it, and print the result.'''
    config = load_config(args.config)

    if config.tokenizer != 'char':
        raise ValueError(
            'Forced character alignment requires a character-model config.'
        )

    device = torch.device(args.device or config.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(
            'CUDA was requested but is not available. Use --device cpu or '
            'check the CUDA-enabled PyTorch installation.'
        )

    tokenizer = get_tokenizer(config.tokenizer)
    tokenizer_path = os.path.join(
        config.dir_tokenizer, f'{config.idx_fold}.json'
    )
    tokenizer.load(tokenizer_path)

    model, checkpoint_epoch = load_model(
        config,
        tokenizer,
        args.checkpoint,
        device,
    )

    annotation_path = os.path.join(
        config.dir_dataset, f'{args.split}.json'
    )
    with open(annotation_path, 'r', encoding='utf-8') as file:
        dataset_document = json.load(file)

    dataset = HRDataset(
        annotation_path,
        tokenizer,
        model.ratio_ds,
        config.idx_fold,
        config.len_seq,
        aug=False,
        cache=False,
    )

    if not 0 <= args.sample_index < len(dataset):
        raise IndexError(
            f'sample index {args.sample_index} is outside the {args.split} '
            f'set, which contains {len(dataset)} samples.'
        )

    signal, target = dataset[args.sample_index]
    annotation = dataset.annos[args.sample_index]
    path_raw_signal = os.path.join(
        config.dir_dataset, annotation['filename']
    )
    raw_signal = np.loadtxt(
        path_raw_signal,
        delimiter=';',
        dtype=np.float32,
    )

    # HRDataset returns (time, channels); Conv1d expects (batch, channels, time).
    model_input = signal.transpose(0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = model(model_input)[0].cpu()

    alignment = ctc_viterbi_align(
        probabilities,
        target,
        blank_id=0,
    )
    greedy_prediction = BestPath(tokenizer).decode(probabilities)
    known_label = tokenizer.decode(target.tolist())
    sample_rate_hz = float(dataset_document['info']['rate_sample_target'])
    analysis = analyze_alignment(
        alignment=alignment,
        probabilities=probabilities,
        token_to_text=lambda token_id: token_text(tokenizer, token_id),
        downsampling_ratio=model.ratio_ds,
        num_model_input_samples=signal.shape[0],
        num_raw_samples=raw_signal.shape[0],
        sample_rate_hz=sample_rate_hz,
    )
    boundary_features = extract_boundary_features(
        raw_signal=raw_signal,
        normalized_signal=signal.numpy(),
        alignment_analysis=analysis,
    )

    path_json, path_plot = save_analysis(
        args,
        config,
        annotation,
        dataset_document['info'],
        checkpoint_epoch,
        known_label,
        greedy_prediction,
        alignment,
        analysis,
        boundary_features,
    )
    plot_alignment(
        raw_signal=raw_signal,
        normalized_signal=signal.numpy(),
        probabilities=probabilities,
        analysis=analysis,
        known_label=known_label,
        greedy_prediction=greedy_prediction,
        path_save=path_plot,
    )

    print('Sample information')
    print(f'  configuration: {args.config}')
    print(f'  checkpoint:    {args.checkpoint}')
    print(f'  checkpoint epoch: {checkpoint_epoch}')
    print(f'  device:        {device}')
    print(f'  split/fold:    {args.split}/{config.idx_fold}')
    print(f'  sample index:  {args.sample_index}')
    print(f'  source file:   {annotation["filename"]}')
    print(f'  writer ID:     {annotation["id_writer"]}')
    print(f'  known label:   {known_label}')
    print(f'  greedy output: {greedy_prediction}')
    print(f'  raw IMU samples:   {raw_signal.shape[0]}')
    print(f'  model input samples: {signal.shape[0]}')
    print(f'  model frames:  {probabilities.shape[0]}')
    print(f'  downsampling:  {model.ratio_ds}x')
    print(f'  sample rate:   {sample_rate_hz:g} Hz')
    print(
        f'  trailing unmodeled samples: '
        f'{analysis.trailing_unmodeled_samples}'
    )
    print(f'  path log score: {alignment.log_score:.4f}')
    print(
        '  mean log score: '
        f'{alignment.log_score / probabilities.shape[0]:.4f}'
    )

    print_alignment(analysis)
    print_boundary_features(boundary_features)

    if args.show_path:
        readable_path = [
            token_text(tokenizer, token_id)
            for token_id in alignment.token_path
        ]
        print('\nFrame-by-frame token path')
        print(' | '.join(readable_path))

    print('\nSaved outputs')
    print(f'  JSON: {path_json}')
    print(f'  plot: {path_plot}')

    return alignment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Align one known character label to its model-output frames.'
        )
    )
    parser.add_argument(
        '-c',
        '--config',
        required=True,
        help='Character-model YAML configuration.',
    )
    parser.add_argument(
        '--checkpoint',
        required=True,
        help='Trained character-model checkpoint.',
    )
    parser.add_argument(
        '--split',
        choices=('train', 'val'),
        default='val',
        help='Dataset split containing the sample. Defaults to val.',
    )
    parser.add_argument(
        '--sample-index',
        type=int,
        default=0,
        help='Zero-based sample index within the selected fold. Defaults to 0.',
    )
    parser.add_argument(
        '--device',
        choices=('cpu', 'cuda'),
        help='Override the device from the YAML configuration.',
    )
    parser.add_argument(
        '--show-path',
        action='store_true',
        help='Also print the selected token at every model frame.',
    )
    parser.add_argument(
        '--output-dir',
        default='results/thesis/alignment_debug',
        help=(
            'Root directory for JSON and PNG outputs. Defaults to '
            'results/thesis/alignment_debug.'
        ),
    )
    return parser.parse_args()


if __name__ == '__main__':
    align_one_sample(parse_args())
