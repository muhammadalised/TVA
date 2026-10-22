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
    'character': ('char', 'artifacts/tokenizers/ed/character/char/0.json'),
    'handwriting_bigram': ('handwriting_bigram_greedy', 'artifacts/tokenizers/ed/handwriting/bigram/0.json'),
    'handwriting_bpe': ('handwriting_bpe', 'artifacts/tokenizers/ed/handwriting/bpe/0.json'),
    'handwriting_unigram': ('handwriting_unigram', 'artifacts/tokenizers/ed/handwriting/unigram/0.json'),
    'linguistic_bigram_tva': ('bigram', 'artifacts/tokenizers/ed/linguistic/bigram/0.json'),
    'linguistic_bigram_iam_frequency': ('linguistic_bigram_greedy', 'artifacts/tokenizers/archive/ed/linguistic/ed_iam_linguistic_bigram_greedy_v1.json'),
    'linguistic_bpe': ('linguistic_bpe', 'artifacts/tokenizers/ed/linguistic/bpe/0.json'),
    'linguistic_unigram': ('linguistic_unigram', 'artifacts/tokenizers/ed/linguistic/unigram/0.json'),
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
                   batch_size: int | None = None) -> list[list[int]]:
    """Return the winning token ID of every output frame, per validation sample.

    Uses the same preprocessing and valid-frame lengths as validation in main.py.
    The run's own batch size is the default: padding within a batch changes the
    encoder's instance normalization, so other sizes give slightly different outputs.
    """
    dataset = run['dataset']
    indices = list(range(len(dataset))) if indices is None else list(indices)
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(subset, batch_size or run['config']['size_batch'], collate_fn=fn_collate)
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


def saved_scores(condition: str, split: str, fold: int) -> dict | None:
    """Validation CER/WER saved during training, at the epoch with the lowest CER.

    Both metrics come from the same epoch, as in evaluate.py:get_mean_std_cv.
    Returns None when the run does not exist on this machine.
    """
    run_dir = RESULTS_ROOT / CONDITIONS[condition].format(split=split) / str(fold)
    files = sorted(run_dir.glob('train_[0-9]*.json'))
    if not files:
        return None
    # The newest file of a resumed run also contains the restored earlier epochs.
    metrics = json.loads(files[-1].read_text())
    epochs = {int(e): m['evaluation'] for e, m in metrics.items()
              if e.isdigit() and 'evaluation' in m}
    if not epochs:
        return None
    epoch = min(epochs, key=lambda e: (epochs[e]['character_error_rate'], e))
    return {'epoch': epoch, 'epochs': len(epochs),
            'cer': epochs[epoch]['character_error_rate'],
            'wer': epochs[epoch]['word_error_rate']}


def iam_training_texts() -> list[str]:
    """The 5,694 IAM training lines every ED tokenizer learned from.

    Read from the bundled, hash-checked connectivity evidence. Lines with
    characters outside the ED alphabet (#, &, *) are left out.
    """
    from tva.handwriting.ed import ED_ALPHABET
    from tva.handwriting.evidence import load_evidence

    corpus, _ = load_evidence()
    allowed = set(ED_ALPHABET)
    return [line['text'] for line in corpus if set(line['text']) <= allowed]


def characters_per_token(tokenizer, texts) -> float:
    """Compression: characters divided by tokens (1.00 = plain characters)."""
    texts = list(texts)
    return sum(map(len, texts)) / sum(len(tokenizer.encode(t)) for t in texts)


def greedy_bigram_split(text: str, bigrams: set[str]) -> list[str]:
    """Split like TVA's greedy bigram encoder, using only the given bigrams."""
    tokens, i = [], 0
    while i < len(text):
        step = 2 if text[i:i + 2] in bigrams else 1
        tokens.append(text[i:i + step])
        i += step
    return tokens


def _apply_merge(tokens: list[str], merge: dict) -> list[str]:
    merged, i = [], 0
    while i < len(tokens):
        if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == (merge['left'], merge['right']):
            merged.append(merge['token'])
            i += 2
        else:
            merged.append(tokens[i])
            i += 1
    return merged


def bpe_split(text: str, merges: list[dict]) -> list[str]:
    """Apply BPE merges in learned order, exactly like the frozen tokenizers."""
    tokens = list(text)
    for merge in merges:
        tokens = _apply_merge(tokens, merge)
    return tokens


def bpe_compression_curve(texts: list[str], merges: list[dict]) -> list[float]:
    """Characters per token when only the first k merges are kept, for k = 0..len(merges).

    BPE learns merges one at a time, so keeping the first k merges equals
    training with a budget of k.
    """
    sequences = [list(text) for text in texts]
    characters = sum(map(len, texts))
    curve = [characters / sum(map(len, sequences))]
    for merge in merges:
        sequences = [_apply_merge(tokens, merge) for tokens in sequences]
        curve.append(characters / sum(map(len, sequences)))
    return curve


def bigram_compression_curve(texts: list[str], ranked_bigrams: list[str]) -> list[float]:
    """Characters per token when only the top k bigrams are kept, for k = 0..len."""
    characters = sum(map(len, texts))
    return [characters / sum(len(greedy_bigram_split(t, set(ranked_bigrams[:k]))) for t in texts)
            for k in range(len(ranked_bigrams) + 1)]


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
