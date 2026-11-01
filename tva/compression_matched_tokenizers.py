"""Compression-matched IAM text-only controls for ED recognition.

Each control is TVA's own letters-only tokenizer (bigram, BPE or Unigram)
trained on IAM with a smaller vocabulary: 79 characters plus k learned pieces
(D024). k is chosen so that the control covers as many characters per token
on IAM training text as its handwriting partner. Handwriting and text-only
tokens then merge equally often, and only the evidence that chose the pieces
differs.

For bigram and BPE a smaller budget keeps a prefix of the full vocabulary
(the top k bigrams, the first k merges); Unigram is retrained for every k.
No ED labels or recognition results are read.
"""

import json
from pathlib import Path
import tempfile

from tva.dataset.ed import ED_ALPHABET
from tva.handwriting.common import canonical_json_bytes, save_fold_tokenizers, save_frozen, sha256_bytes
from tva.handwriting_tokenizers import read_connectivity_lines
from tva.tokenizers import BigramTokenizer, BPETokenizer, UnigramTokenizer

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / 'artifacts' / 'tokenizers'
IAM_EVIDENCE = ROOT / 'artifacts' / 'evidence' / 'iam' / 'iam_connectivity_v1.json.gz'
SCHEMA = 'tva.ed-compression-matched-text-control.v2'
CATEGORIES = ['', *ED_ALPHABET]
NUM_CHARACTERS = len(set(CATEGORIES))  # blank and the 78 ED characters
MIN_PIECES, MAX_PIECES = 1, 145  # TVA's bigram trainer needs at least one bigram

# kind -> TVA class, handwriting partner and output directory.
FAMILIES = {
    'bigram': {
        'class': BigramTokenizer,
        'partner': 'ed/handwriting/bigram_v3',
        'output': 'ed/linguistic/bigram_matched',
    },
    'bpe': {
        'class': BPETokenizer,
        'partner': 'ed/handwriting/bpe_v3',
        'output': 'ed/linguistic/bpe_matched_v3',
    },
    'unigram': {
        'class': UnigramTokenizer,
        'partner': 'ed/handwriting/unigram_v3',
        'output': 'ed/linguistic/unigram_matched_v3',
    },
}


def iam_training_texts() -> list[str]:
    """The IAM training lines every ED tokenizer learned from.

    Read from the bundled connectivity evidence (text only). Lines with
    characters outside the ED alphabet (#, &, *) are left out.
    """
    allowed = set(ED_ALPHABET)
    return [line['text'] for line in read_connectivity_lines(IAM_EVIDENCE) if set(line['text']) <= allowed]


def write_iam_dataset(directory) -> None:
    """IAM training lines as a one-fold TVA dataset, like tva.dataset.iam.prepare_iam()."""
    annotations = [{'id': i, 'label': line['text']} for i, line in enumerate(read_connectivity_lines(IAM_EVIDENCE))]
    with open(Path(directory) / 'train.json', 'w', encoding='utf-8') as f:
        json.dump({'info': {'num_fold': 1}, 'annotations': {'0': annotations}}, f)


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


def train_tva(kind: str, dir_ds, num_pieces: int):
    """TVA's letters-only tokenizer with num_pieces learned pieces; returns it and its files."""
    size = NUM_CHARACTERS + num_pieces
    tokenizer = FAMILIES[kind]['class']()
    tokenizer.train(str(dir_ds), CATEGORIES, size, letters_only=True, name=f'{kind}{size}')
    directory = Path(dir_ds) / 'tokenizers' / f'{kind}{size}'
    files = {path.name: path.read_bytes() for path in sorted(directory.glob('0.*'))}
    return tokenizer, files


def matched_size(curve: dict[int, float], target: float) -> int:
    """Size k whose compression is closest to the target; ties go to the smaller k."""
    return min(curve, key=lambda k: (abs(curve[k] - target), k))


def build_matched_model(kind: str) -> tuple[dict[str, bytes], dict]:
    """Select k on IAM training text; return the ED tokenizer files and the choice."""
    family = FAMILIES[kind]
    partner_path = ARTIFACT_ROOT / family['partner'] / '0.json'
    partner = family['class']()
    partner.load(str(partner_path))

    texts = iam_training_texts()
    target = characters_per_token(partner, texts)
    curve, files = {}, {}
    with tempfile.TemporaryDirectory() as directory:
        write_iam_dataset(directory)
        for k in range(MIN_PIECES, MAX_PIECES + 1):
            tokenizer, files[k] = train_tva(kind, directory, k)
            curve[k] = characters_per_token(tokenizer, texts)
    k = matched_size(curve, target)
    record = {
        'schema_version': SCHEMA,
        'kind': kind, 'subword_count': k, 'classes': NUM_CHARACTERS + k,
        'handwriting_partner': {'path': f"{family['partner']}/0.json",
                                'sha256': sha256_bytes(partner_path.read_bytes())},
        'selection': {
            'rule': (f"TVA's letters-only {kind} tokenizer trained on IAM with {NUM_CHARACTERS} + k classes; "
                     'k minimizes |IAM characters per token - handwriting partner| (ties: smaller k)'),
            'iam_text_lines': len(texts),
            'iam_characters_per_token': {
                'handwriting_partner': round(target, 6),
                'matched': round(curve[k], 6),
                'full_text_only': round(curve[MAX_PIECES], 6),
            },
            'iam_characters_per_token_by_k': {'first_k': MIN_PIECES,
                                              'values': [round(curve[k], 6) for k in sorted(curve)]},
            'ed_labels_read': False,
        },
    }
    return files[k], record


def build_fold_tokenizers(kind: str, num_folds: int = 5) -> Path:
    """Write 0.json (and the selection record) and copy it to every fold."""
    directory = ARTIFACT_ROOT / FAMILIES[kind]['output']
    files, record = build_matched_model(kind)
    for name, content in files.items():
        save_frozen(directory / name, content)
    save_frozen(directory / 'selection.json', canonical_json_bytes(record))
    save_fold_tokenizers(directory, num_folds)
    return directory
