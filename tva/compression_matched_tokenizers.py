"""Compression-matched IAM text-only controls for ED recognition.

Each control keeps a prefix of a text-only TVA vocabulary: the top k bigrams
of the letters-only bigram, or the first k BPE merges. A Unigram vocabulary has
no such order, so the Unigram control retrains the text-only Unigram EM with a
budget of k pieces. k is chosen so that the control covers as many characters
per token on IAM training text as its handwriting partner.
Handwriting and text-only tokens then merge equally often, and only the
evidence that chose the pieces differs.

The controls are saved in TVA's own formats: the bigram control loads with
BigramTokenizer, the BPE control with BPETokenizer and the Unigram control
with UnigramTokenizer. No ED labels or
recognition results are read.
"""

import json
from pathlib import Path

from tva.dataset.ed import ED_ALPHABET
from tva.handwriting.common import canonical_json_bytes, save_fold_tokenizers, save_frozen, sha256_bytes
from tva.handwriting.training import frequent_spans, train_unigram
from tva.handwriting_tokenizers import bpe_from_merges, read_connectivity_lines, unigram_from_log_probs

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / 'artifacts' / 'tokenizers'
IAM_EVIDENCE = ROOT / 'artifacts' / 'evidence' / 'iam' / 'iam_connectivity_v1.json.gz'
SCHEMA = 'tva.ed-compression-matched-text-control.v1'

# kind -> text-only source, handwriting partner and output directory.
FAMILIES = {
    'bigram': {
        'source': ('bigram', 'ed/linguistic/bigram_letters/0.json'),
        'partner': ('handwriting_bigram_greedy', 'ed/handwriting/bigram/0.json'),
        'output': 'ed/linguistic/bigram_matched',
        'selection': 'top-k letters-only TVA bigrams (TVA rank order = ID order)',
    },
    'bpe': {
        'source': ('bpe', 'ed/linguistic/bpe/0.json'),
        'partner': ('handwriting_bpe', 'ed/handwriting/bpe/0.json'),
        'output': 'ed/linguistic/bpe_matched',
        'selection': 'first k text BPE merges (learned order)',
    },
    'unigram': {
        'source': ('unigram_em', 'ed/linguistic/unigram_em/0.json'),
        'partner': ('handwriting_unigram', 'ed/handwriting/unigram/0.json'),
        'output': 'ed/linguistic/unigram_matched',
        'selection': 'text-only Unigram EM (UnigramEMTokenizer) retrained with a budget of k pieces',
    },
}


def iam_training_texts() -> list[str]:
    """The IAM training lines every ED tokenizer learned from.

    Read from the bundled connectivity evidence (text only). Lines with
    characters outside the ED alphabet (#, &, *) are left out.
    """
    allowed = set(ED_ALPHABET)
    return [line['text'] for line in read_connectivity_lines(IAM_EVIDENCE) if set(line['text']) <= allowed]


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


def bigram_compression_curve(texts: list[str], ranked_bigrams: list[str]) -> list[float]:
    """Characters per token when only the top k bigrams are kept, for k = 0..len."""
    characters = sum(map(len, texts))
    return [characters / sum(len(greedy_bigram_split(t, set(ranked_bigrams[:k]))) for t in texts)
            for k in range(len(ranked_bigrams) + 1)]


def bpe_merges(tokenizer) -> list[tuple[str, str]]:
    """The ordered merges of a loaded BPETokenizer."""
    merges = json.loads(tokenizer.tokenizer.to_str())['model']['merges']
    return [tuple(m.split(' ')) if isinstance(m, str) else tuple(m) for m in merges]


def bpe_compression_curve(texts: list[str], merges: list[tuple[str, str]]) -> list[float]:
    """Characters per token when only the first k merges are kept, for k = 0..len(merges).

    BPE learns merges one at a time, so keeping the first k merges equals
    training with a budget of k.
    """
    chars = sorted(set(''.join(texts)))
    characters = sum(map(len, texts))
    curve = []
    for k in range(len(merges) + 1):
        encodings = bpe_from_merges(chars, merges[:k]).encode_batch(texts)
        curve.append(characters / sum(len(e.ids) for e in encodings))
    return curve


def unigram_em_model(texts: list[str], budget: int):
    """The text-only Unigram EM with budget pieces, as UnigramEMTokenizer trains it."""
    lines = [{'text': text} for text in texts]
    log_probs = train_unigram(lines, frequent_spans(lines), budget)
    return unigram_from_log_probs(sorted(ED_ALPHABET), log_probs)


def unigram_compression_curve(texts: list[str], budgets) -> list[float]:
    """Characters per token of the text-only Unigram EM for each budget.

    Unlike bigram and BPE, every budget is a separate EM training.
    """
    characters = sum(map(len, texts))
    return [characters / sum(len(e.ids) for e in unigram_em_model(texts, k).encode_batch(texts))
            for k in budgets]


def ranked_pieces(kind: str, source) -> list:
    """The source's learned pieces in selection order: bigram strings or BPE merges.

    TVA's BigramTokenizer numbers its bigrams by frequency rank; handwriting
    bigram files rank by their stored utility instead.
    """
    if kind == 'bpe':
        return bpe_merges(source)
    utility = getattr(source, 'utility', None)
    if utility:
        return sorted(utility, key=lambda t: (-utility[t], t))
    return [source.idx_token[i] for i in sorted(source.idx_token) if len(source.idx_token[i]) == 2]


def compression_curve(kind: str, texts: list[str], pieces: list) -> list[float]:
    return (bigram_compression_curve if kind == 'bigram' else bpe_compression_curve)(texts, pieces)


def matched_size(curve: list[float], target: float) -> int:
    """Size k whose compression is closest to the target; ties go to the smaller k."""
    return min(range(len(curve)), key=lambda k: (abs(curve[k] - target), k))


def build_matched_model(kind: str) -> tuple[bytes, dict]:
    """Select k on IAM training text; return the ED tokenizer file and the choice."""
    from tva.tokenizer_factory import get_tokenizer

    family = FAMILIES[kind]
    loaded = {}
    for role in ('source', 'partner'):
        key, path = family[role]
        loaded[role] = get_tokenizer(key)
        loaded[role].load(str(ARTIFACT_ROOT / path))

    texts = iam_training_texts()
    if kind == 'unigram':
        source_count = loaded['source'].size - 1 - len(ED_ALPHABET)
        curve = unigram_compression_curve(texts, range(source_count + 1))
    else:
        pieces = ranked_pieces(kind, loaded['source'])
        source_count = len(pieces)
        curve = compression_curve(kind, texts, pieces)
    target = characters_per_token(loaded['partner'], texts)
    k = matched_size(curve, target)
    record = {
        'schema_version': SCHEMA,
        'kind': kind, 'subword_count': k,
        'source': {'path': family['source'][1],
                   'sha256': sha256_bytes((ARTIFACT_ROOT / family['source'][1]).read_bytes()),
                   'subword_count': source_count},
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
    if kind == 'unigram':
        # Every budget is a separate training, so the whole curve is kept.
        record['selection']['iam_characters_per_token_by_k'] = [round(c, 6) for c in curve]
        return unigram_em_model(texts, k).to_str(pretty=True).encode('utf-8'), record
    selected = pieces[:k]
    if kind == 'bpe':
        content = bpe_from_merges(list(ED_ALPHABET), selected).to_str(pretty=True)
        return content.encode('utf-8'), record

    # Exactly what TVA's BigramTokenizer.train() writes for this vocabulary.
    vocab = {t: i for i, t in enumerate(['', *ED_ALPHABET, *selected])}
    idx_token = {i: t for t, i in vocab.items()}
    return json.dumps({'vocab': vocab, 'idx_token': idx_token}, ensure_ascii=False).encode('utf-8'), record


def build_fold_tokenizers(kind: str, num_folds: int = 5) -> Path:
    """Write 0.json (and the selection record) and copy it to every fold."""
    directory = ARTIFACT_ROOT / FAMILIES[kind]['output']
    content, record = build_matched_model(kind)
    save_frozen(directory / '0.json', content)
    save_frozen(directory / 'selection.json', canonical_json_bytes(record))
    save_fold_tokenizers(directory, num_folds)
    return directory
