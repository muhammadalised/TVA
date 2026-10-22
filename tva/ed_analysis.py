"""Shared helpers for the ED analysis notebooks.

Loads extracted ED labels, saved recognition runs and their frozen tokenizers,
and runs inference only. Nothing here trains a model or fits a tokenizer.
"""

from argparse import Namespace
from collections import Counter
import json
import math
import os
from pathlib import Path

import torch
import yaml
from IPython.display import Markdown, display
from torch.utils.data import DataLoader

from tva.dataset import HRDataset, fn_collate
from tva.model import BaseModel
from tva.tokenizer_factory import get_tokenizer, resolve_tokenizer_path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _root(variable: str, default: str) -> Path:
    path = Path(os.environ.get(variable, default)).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


# Optional overrides for other machines; defaults match docs/SETUP.md.
DATA_ROOT = _root('TVA_ED_EXTRACTED_DIR', 'data/tva/ed')
RESULTS_ROOT = _root('TVA_ED_RESULTS_DIR', 'results/thesis/ed')

# Readable condition name -> saved result directory under RESULTS_ROOT.
CONDITIONS = {
    'character': 'character_{split}_cv_rerun',
    'handwriting_bigram': 'handwriting_{split}',
    'handwriting_bpe': 'handwriting_bpe_{split}',
    'handwriting_unigram': 'handwriting_unigram_{split}',
    'linguistic_bigram_tva': 'tva_original_{split}',
    'linguistic_bigram_iam_frequency': 'linguistic_{split}',
    'linguistic_bpe': 'linguistic_bpe_{split}',
    'linguistic_unigram': 'linguistic_unigram_{split}',
}

# Frozen tokenizer files, for analyses that need no trained model.
TOKENIZERS = {
    'character': ('char', 'artifacts/tokenizers/ed/character/ed_characters_v1.json'),
    'handwriting_bigram': ('handwriting_bigram_greedy', 'artifacts/tokenizers/ed/handwriting/ed_iam_handwriting_bigram_greedy_v1.json'),
    'handwriting_bpe': ('handwriting_bpe', 'artifacts/tokenizers/ed/handwriting/ed_iam_handwriting_bpe_v1.json'),
    'handwriting_unigram': ('handwriting_unigram', 'artifacts/tokenizers/ed/handwriting/ed_iam_handwriting_unigram_v1.json'),
    'linguistic_bigram_tva': ('bigram', 'artifacts/tokenizers/ed/linguistic/ed_iam_tva_original_bigram_greedy_v1.json'),
    'linguistic_bigram_iam_frequency': ('linguistic_bigram_greedy', 'artifacts/tokenizers/archive/ed/linguistic/ed_iam_linguistic_bigram_greedy_v1.json'),
    'linguistic_bpe': ('linguistic_bpe', 'artifacts/tokenizers/ed/linguistic/ed_iam_linguistic_bpe_v1.json'),
    'linguistic_unigram': ('linguistic_unigram', 'artifacts/tokenizers/ed/linguistic/ed_iam_linguistic_unigram_v1.json'),
}


def load_tokenizer(condition: str):
    """Load a frozen ED tokenizer by condition name."""
    key, path = TOKENIZERS[condition]
    tokenizer = get_tokenizer(key)
    tokenizer.load(resolve_tokenizer_path(str(REPO_ROOT / path), 0))
    return tokenizer


def load_records(split: str, fold: int, part: str) -> list[dict]:
    """Return one fold's 'train' or 'val' annotations from extracted ED."""
    path = DATA_ROOT / f'gold_{split}' / f'{part}.json'
    if not path.is_file():
        raise FileNotFoundError(f'Missing extracted ED file {path}. See docs/SETUP.md.')
    return json.loads(path.read_text())['annotations'][str(fold)]


def _repo_path(value: str | None) -> str | None:
    """Make a saved artifact path usable on this machine."""
    if not value:
        return value
    path = Path(value)
    if not path.is_absolute():
        return str(REPO_ROOT / path)
    if not path.exists() and 'artifacts' in path.parts:
        # Saved on another machine: rebase onto this checkout.
        return str(REPO_ROOT.joinpath(*path.parts[path.parts.index('artifacts'):]))
    return value


def load_run(condition: str, split: str, fold: int,
             checkpoint: str = 'best_cer.pth', device: str = 'cpu') -> dict:
    """Load a saved recognizer, its tokenizer and its validation dataset."""
    from main import load_configured_tokenizer

    run_dir = RESULTS_ROOT / CONDITIONS[condition].format(split=split) / str(fold)
    configs = sorted(run_dir.glob('train_[0-9]*.yaml'))
    if not configs:
        raise FileNotFoundError(f'No saved run for {condition} {split.upper()} fold {fold}: {run_dir}')
    # The newest config belongs to the completed (or resumed) run.
    config = yaml.safe_load(configs[-1].read_text())
    config['dir_tokenizer'] = _repo_path(config.get('dir_tokenizer'))
    tokenizer = load_configured_tokenizer(Namespace(**config))

    state = torch.load(run_dir / 'checkpoints' / checkpoint, map_location='cpu', weights_only=False)
    model = BaseModel(config['arch_en'], config['arch_de'], config['num_channel'],
                      tokenizer.size, config['len_seq'])
    model.load_state_dict(state['model'])
    model.to(device).eval()

    dataset = HRDataset(str(DATA_ROOT / f'gold_{split}' / 'val.json'), tokenizer,
                        model.ratio_ds, fold, len_seq=config['len_seq'])
    return {'condition': condition, 'split': split, 'fold': fold, 'run_dir': run_dir,
            'config': config, 'tokenizer': tokenizer, 'model': model,
            'epoch': state['epoch'], 'checkpoint': checkpoint,
            'dataset': dataset, 'device': device}


def predict_frames(run: dict, indices: list[int] | None = None,
                   batch_size: int = 16) -> list[list[int]]:
    """Return the winning token ID of every output frame, per validation sample.

    Uses the same preprocessing and valid-frame lengths as validation in main.py.
    """
    dataset = run['dataset']
    indices = list(range(len(dataset))) if indices is None else list(indices)
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(subset, batch_size, collate_fn=fn_collate)
    model, frames = run['model'], []
    with torch.inference_mode():
        for x, _, len_x, _ in loader:
            winners = model(x.to(run['device'])).argmax(-1).cpu()
            for row, length in zip(winners, len_x // model.ratio_ds):
                frames.append(row[:length].tolist())
    return frames


def collapse(frames: list[int]) -> list[int]:
    """CTC best path: merge repeated frames, then drop blank ID 0."""
    merged = [i for n, i in enumerate(frames) if n == 0 or i != frames[n - 1]]
    return [i for i in merged if i != 0]


def saved_predictions(run: dict) -> list[str] | None:
    """Text predictions saved during training for the checkpoint's epoch."""
    for path in sorted(run['run_dir'].glob('train_preds_*.json'), reverse=True):
        predictions = json.loads(path.read_text())
        if str(run['epoch']) in predictions:
            return predictions[str(run['epoch'])]
    return None


def token_counts(tokenizer, id_lists) -> Counter:
    """Count token strings over many encoded or predicted sequences."""
    # TVA's CharacterTokenizer names its ID mapping idx_char.
    mapping = getattr(tokenizer, 'idx_token', None) or tokenizer.idx_char
    return Counter(mapping[i] for ids in id_lists for i in ids)


def js_divergence(a: Counter, b: Counter) -> float:
    """Base-2 Jensen-Shannon divergence: 0 = identical, 1 = no shared tokens."""
    total_a, total_b = sum(a.values()), sum(b.values())
    if not total_a or not total_b:
        return float('nan')
    value = 0.0
    for token in set(a) | set(b):
        p, q = a[token] / total_a, b[token] / total_b
        m = (p + q) / 2
        value += sum(0.5 * x * math.log2(x / m) for x in (p, q) if x)
    return value


def show_table(rows: list[dict]) -> None:
    """Display a list of dictionaries as a Markdown table."""
    if not rows:
        print('(nothing to show)')
        return
    columns = list(rows[0])
    def cell(value):
        return str(value).replace('|', '\\|').replace('\n', ' ')
    lines = ['| ' + ' | '.join(columns) + ' |', '|' + ' --- |' * len(columns)]
    lines += ['| ' + ' | '.join(cell(row[c]) for c in columns) + ' |' for row in rows]
    display(Markdown('\n'.join(lines)))


def show(token: str) -> str:
    """Make spaces visible in token names."""
    return repr(token)
