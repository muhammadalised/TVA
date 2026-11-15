'''Shared setup helpers for alignment command-line tools.'''

import argparse
from typing import Any

import torch
import yaml

from .model import BaseModel

__all__ = ['load_config', 'load_model', 'token_text']


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
    # plain state dictionary also keeps the tools useful for older weights.
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
