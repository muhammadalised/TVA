'''Run target-constrained CTC alignment for one handwriting sample.'''

import argparse
import os
from typing import Any

import torch
import yaml

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
    alignment: CTCAlignment,
    tokenizer: Any,
) -> None:
    '''Print one readable row for every target character.'''
    print('\nForced alignment (model-output frames)')
    print('index  char  token_id  start  end  frames')
    print('-----  ----  --------  -----  ---  ------')

    for token in alignment.tokens:
        character = token_text(tokenizer, token.token_id)
        print(
            f'{token.target_index:>5}  '
            f'{character:^4}  '
            f'{token.token_id:>8}  '
            f'{token.start_frame:>5}  '
            f'{token.end_frame:>3}  '
            f'{token.num_frames:>6}'
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
    print(f'  IMU samples:   {signal.shape[0]}')
    print(f'  model frames:  {probabilities.shape[0]}')
    print(f'  downsampling:  {model.ratio_ds}x')
    print(f'  path log score: {alignment.log_score:.4f}')
    print(
        '  mean log score: '
        f'{alignment.log_score / probabilities.shape[0]:.4f}'
    )

    print_alignment(alignment, tokenizer)

    if args.show_path:
        readable_path = [
            token_text(tokenizer, token_id)
            for token_id in alignment.token_path
        ]
        print('\nFrame-by-frame token path')
        print(' | '.join(readable_path))

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
    return parser.parse_args()


if __name__ == '__main__':
    align_one_sample(parse_args())
