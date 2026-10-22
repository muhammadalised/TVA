"""Compression-matched IAM text-only controls for ED recognition.

Each control keeps a prefix of an existing text-only vocabulary: the top k
IAM-count bigrams, or the first k text BPE merges. k is chosen so that the
control covers as many characters per token on IAM training text as its
handwriting partner. Handwriting and text-only tokens then merge equally often,
and only the evidence that chose the pieces differs.

No ED labels or recognition results are read.
"""

import json
from pathlib import Path
import unicodedata

from loguru import logger

from tva.handwriting.common import (
    canonical_json_bytes, require, resolve_artifact_path, save_fold_tokenizers, save_frozen, sha256_bytes,
)
from tva.handwriting.ed import ED_ALPHABET
from tva.subword_tokenizers import SubwordTokenizer

ARTIFACT_ROOT = Path(__file__).resolve().parents[1] / 'artifacts' / 'tokenizers'
SCHEMA = 'tva.ed-compression-matched-text-control.v1'

# kind -> text-only source, handwriting partner and output directory.
FAMILIES = {
    'bigram': {
        'source': ('linguistic_bigram_greedy', 'archive/ed/linguistic/ed_iam_linguistic_bigram_greedy_v1.json'),
        'partner': ('handwriting_bigram_greedy', 'ed/handwriting/bigram/0.json'),
        'output': 'ed/linguistic/bigram_matched',
        'selection': 'top-k IAM-count bigrams (source rank order)',
    },
    'bpe': {
        'source': ('linguistic_bpe', 'ed/linguistic/bpe/0.json'),
        'partner': ('handwriting_bpe', 'ed/handwriting/bpe/0.json'),
        'output': 'ed/linguistic/bpe_matched',
        'selection': 'first k text BPE merges (learned order)',
    },
}


def iam_training_texts() -> list[str]:
    """The IAM training lines every ED tokenizer learned from.

    Read from the bundled, hash-checked connectivity evidence (text only).
    Lines with characters outside the ED alphabet (#, &, *) are left out.
    """
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


def ranked_pieces(kind: str, source) -> list:
    """The source's learned pieces in selection order: bigram strings or BPE merges."""
    if kind == 'bigram':
        return [row['token'] for row in sorted(source.model['vocabulary'], key=lambda r: r['rank'])]
    return list(source.source_model['merges'])


def compression_curve(kind: str, texts: list[str], pieces: list) -> list[float]:
    return (bigram_compression_curve if kind == 'bigram' else bpe_compression_curve)(texts, pieces)


def matched_size(curve: list[float], target: float) -> int:
    """Size k whose compression is closest to the target; ties go to the smaller k."""
    return min(range(len(curve)), key=lambda k: (abs(curve[k] - target), k))


def build_matched_model(kind: str) -> dict:
    """Select k on IAM training text and return the frozen ED tokenizer model."""
    from tva.tokenizer_factory import get_tokenizer

    require(kind in FAMILIES, f'Unsupported compression-matched family: {kind!r}')
    family = FAMILIES[kind]
    loaded = {}
    for role in ('source', 'partner'):
        key, path = family[role]
        loaded[role] = get_tokenizer(key)
        loaded[role].load(ARTIFACT_ROOT / path)

    texts = iam_training_texts()
    pieces = ranked_pieces(kind, loaded['source'])
    curve = compression_curve(kind, texts, pieces)
    target = characters_per_token(loaded['partner'], texts)
    k = matched_size(curve, target)
    selected = pieces[:k]
    tokens = ['', *ED_ALPHABET, *(p if kind == 'bigram' else p['token'] for p in selected)]
    require(len(tokens) == len(set(tokens)), 'Duplicate token in matched vocabulary')
    return {
        'schema_version': SCHEMA,
        'model_version': f'ed-iam-linguistic-{kind}-compression-matched-v1',
        'status': 'frozen-compression-matched-text-control',
        'dataset': 'ED', 'kind': kind,
        'unicode_normalization': 'NFC', 'blank_id': 0, 'blank_token': '',
        'size': len(tokens), 'subword_count': k,
        'vocab': {t: i for i, t in enumerate(tokens)},
        'idx_token': {str(i): t for i, t in enumerate(tokens)},
        'pieces': selected,
        'source': {'path': family['source'][1],
                   'sha256': sha256_bytes((ARTIFACT_ROOT / family['source'][1]).read_bytes()),
                   'subword_count': len(pieces)},
        'handwriting_partner': {'path': family['partner'][1],
                                'sha256': sha256_bytes((ARTIFACT_ROOT / family['partner'][1]).read_bytes())},
        'selection': {
            'rule': family['selection'] + '; k minimizes |IAM characters per token - handwriting partner|',
            'iam_text_lines': len(texts),
            'iam_characters_per_token': {
                'handwriting_partner': round(target, 6),
                'matched': round(curve[k], 6),
                'full_text_only': round(curve[-1], 6),
            },
            'ed_labels_read': False,
        },
    }


def build_fold_tokenizers(kind: str, num_folds: int = 5) -> Path:
    """Write 0.json for a family and copy it to every recognition fold."""
    directory = ARTIFACT_ROOT / FAMILIES[kind]['output']
    save_frozen(directory / '0.json', canonical_json_bytes(build_matched_model(kind)))
    save_fold_tokenizers(directory, num_folds)
    return directory


class _CompressionMatchedTokenizer(SubwordTokenizer):
    """Load-and-run tokenizer for a frozen compression-matched control."""

    def load(self, path_config):
        path = resolve_artifact_path(path_config)
        model = json.loads(path.read_bytes())
        require(model.get('schema_version') == SCHEMA and model.get('kind') == self.kind,
                f'Not a compression-matched {self.kind} tokenizer: {path}')
        source = ARTIFACT_ROOT / model['source']['path']
        require(sha256_bytes(source.read_bytes()) == model['source']['sha256'],
                f'Text-only source changed since the control was built: {source}')
        tokens = [model['idx_token'][str(i)] for i in range(model['size'])]
        require(tokens[:len(ED_ALPHABET) + 1] == ['', *ED_ALPHABET], 'Unexpected ED character IDs')
        require(model['vocab'] == {t: i for i, t in enumerate(tokens)}, 'Inconsistent vocabulary')
        self.model = model
        self.vocab = dict(model['vocab'])
        self.idx_token = dict(enumerate(tokens))
        if self.kind == 'bpe':
            self.source_model = {'merges': model['pieces']}
        logger.info(f'{type(self).__name__} with {model["subword_count"]} pieces is loaded from {path}.')


class CompressionMatchedBigramTokenizer(_CompressionMatchedTokenizer):
    """Top-k IAM-count bigrams, segmented greedily left to right."""

    kind = 'bigram'

    def segment(self, text: str) -> dict:
        if not self.model:
            raise ValueError('Tokenizer not trained or loaded.')
        text = unicodedata.normalize('NFC', text)
        missing = sorted(set(text) - set(self.vocab))
        require(not missing, f'characters are absent from tokenizer vocabulary: {missing!r}')
        tokens = greedy_bigram_split(text, set(self.model['pieces']))
        return {'text': text, 'tokens': tokens}


class CompressionMatchedBPETokenizer(_CompressionMatchedTokenizer):
    """First k text BPE merges, applied with the shared BPE segmentation."""

    kind = 'bpe'
