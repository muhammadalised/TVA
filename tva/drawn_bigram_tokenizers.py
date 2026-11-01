"""Size- and compression-matched IAM text-only bigram controls for ED recognition.

The compression-matched bigram (tva/compression_matched_tokenizers.py) matches
the handwriting bigram's characters per token with only 62 bigrams, because
the most frequent bigrams compress more. These controls match both: 145
bigrams (224 classes, like the handwriting bigram) that compress IAM text as
much as the handwriting bigram, chosen without connectivity.

Each control is a random draw from the handwriting bigram's own candidates
(letter bigrams with at least 20 aligned IAM observations), weighted by text
frequency: bigram p is drawn without replacement with probability
proportional to count(p) ** alpha (Gumbel top-k). alpha = 0 draws uniformly
(compresses less than the handwriting bigram), large alpha keeps the most
frequent bigrams (compresses more); per draw, alpha is the grid value whose
compression is closest to the handwriting bigram's. Draws that cannot get
within TOLERANCE are skipped, so seeds are tried in order until enough are
accepted. Each control is then built exactly like the handwriting bigram
(D024): TVA's letters-only BigramTokenizer with the drawn bigrams as its
allowed pairs, so it differs from the handwriting bigram only in which
bigrams it holds.

No ED labels or recognition results are read.
"""

from collections import Counter
import json
import math
from pathlib import Path
import random
import tempfile

from tva.compression_matched_tokenizers import (
    ARTIFACT_ROOT, CATEGORIES, IAM_EVIDENCE, greedy_bigram_split, iam_training_texts, write_iam_dataset,
)
from tva.dataset.ed import ED_ALPHABET
from tva.handwriting.common import canonical_json_bytes, save_fold_tokenizers, save_frozen, sha256_bytes
from tva.handwriting_tokenizers import MIN_COUNT, read_bigram_scores
from tva.tokenizers import BigramTokenizer

SCHEMA = 'tva.ed-size-compression-matched-bigram.v2'
PARTNER = 'ed/handwriting/bigram_v3/0.json'
OUTPUT = 'ed/linguistic/bigram_drawn'
ALPHAS = [i / 1000 for i in range(2001)]
TOLERANCE = 0.001
NUM_DRAWS = 3


def candidate_pool() -> list[str]:
    """The letter bigrams the handwriting bigram ranked, alphabetically."""
    return sorted(row['pair'] for row in read_bigram_scores(IAM_EVIDENCE)
                  if row['pair'].isalpha() and set(row['pair']) <= set(ED_ALPHABET)
                  and row['n_exact_alignment'] >= MIN_COUNT)


def bigram_counts(texts: list[str], pool: list[str]) -> Counter:
    """Occurrences in the unique IAM lines, counted like TVA's BigramTokenizer."""
    allowed, counts = set(pool), Counter()
    for text in set(texts):
        counts.update(text[i:i + 2] for i in range(len(text) - 1) if text[i:i + 2] in allowed)
    return counts


def handwriting_partner() -> list[str]:
    """The handwriting bigram's bigrams, alphabetically."""
    vocab = json.loads((ARTIFACT_ROOT / PARTNER).read_text())['vocab']
    return sorted(t for t in vocab if len(t) == 2)


def tva_bigram_files(dir_ds, bigrams: list[str]) -> dict[str, bytes]:
    """TVA's letters-only BigramTokenizer files with the given bigrams as allowed pairs."""
    size = len(CATEGORIES) + len(bigrams)
    BigramTokenizer().train(str(dir_ds), CATEGORIES, size, letters_only=True,
                            allowed_pairs=set(bigrams), name='drawn')
    directory = Path(dir_ds) / 'tokenizers' / 'drawn'
    return {path.name: path.read_bytes() for path in sorted(directory.glob('0.*'))}


def characters_per_token(texts: list[str], bigrams) -> float:
    bigrams = set(bigrams)
    return sum(map(len, texts)) / sum(len(greedy_bigram_split(t, bigrams)) for t in texts)


def draw(seed: int, texts: list[str], pool: list[str], counts: Counter, size: int, target: float) -> dict:
    """One frequency-weighted draw, its alpha chosen to match the target compression."""
    rng = random.Random(seed)
    gumbel = {p: -math.log(-math.log(rng.random())) for p in pool}
    compression = {}  # selections repeat across neighbouring alphas
    best = None
    for alpha in ALPHAS:
        selected = tuple(sorted(sorted(pool, key=lambda p: (-(alpha * math.log(counts[p]) + gumbel[p]), p))[:size]))
        if selected not in compression:
            compression[selected] = characters_per_token(texts, selected)
        # Ties go to the smaller alpha.
        if best is None or abs(compression[selected] - target) < abs(best['compression'] - target):
            best = {'seed': seed, 'alpha': alpha, 'compression': compression[selected], 'bigrams': list(selected)}
    return best


def build_drawn_models(num_draws: int = NUM_DRAWS) -> tuple[list[tuple[dict[str, bytes], dict]], dict]:
    """Draw until num_draws are accepted; return each draw's tokenizer files with its record, and a summary."""
    texts = iam_training_texts()
    pool = candidate_pool()
    counts = bigram_counts(texts, pool)
    partner = handwriting_partner()
    target = characters_per_token(texts, partner)
    accepted, rejected, seed = [], [], 0
    while len(accepted) < num_draws:
        result = draw(seed, texts, pool, counts, len(partner), target)
        if abs(result['compression'] - target) <= TOLERANCE:
            accepted.append(result)
        else:
            rejected.append({'seed': seed, 'iam_characters_per_token': round(result['compression'], 6)})
        seed += 1

    rule = (f'{len(partner)} bigrams drawn without replacement from the handwriting bigram candidates '
            f'(letter bigrams with >= {MIN_COUNT} aligned IAM observations), with probability proportional to '
            'IAM text count ** alpha (Gumbel top-k, random.Random(seed)); alpha on a 0.001 grid in [0, 2] '
            f'minimizes |IAM characters per token - handwriting partner|; draws further than {TOLERANCE} '
            'from the partner are skipped')
    partner_record = {'path': PARTNER, 'sha256': sha256_bytes((ARTIFACT_ROOT / PARTNER).read_bytes()),
                      'iam_characters_per_token': round(target, 6)}
    models = []
    for index, result in enumerate(accepted):
        with tempfile.TemporaryDirectory() as directory:
            write_iam_dataset(directory)
            files = tva_bigram_files(directory, result['bigrams'])
        models.append((files, {
            'schema_version': SCHEMA, 'draw': index, 'seed': result['seed'], 'alpha': result['alpha'],
            'subword_count': len(result['bigrams']),
            'handwriting_partner': partner_record,
            'selection': {
                'rule': rule, 'candidate_count': len(pool), 'iam_text_lines': len(texts),
                'iam_characters_per_token': round(result['compression'], 6),
                'shared_with_handwriting_partner': len(set(result['bigrams']) & set(partner)),
                'ed_labels_read': False,
            },
        }))
    summary = {'schema_version': SCHEMA, 'rule': rule, 'handwriting_partner': partner_record,
               'tolerance': TOLERANCE, 'accepted_seeds': [r['seed'] for r in accepted], 'rejected': rejected}
    return models, summary


def build_fold_tokenizers(num_draws: int = NUM_DRAWS, num_folds: int = 5) -> Path:
    """Write draw<i>/0.json (and its record) for each draw, copied to every fold."""
    directory = ARTIFACT_ROOT / OUTPUT
    models, summary = build_drawn_models(num_draws)
    for index, (files, record) in enumerate(models):
        for name, content in files.items():
            save_frozen(directory / f'draw{index}' / name, content)
        save_frozen(directory / f'draw{index}' / 'selection.json', canonical_json_bytes(record))
        save_fold_tokenizers(directory / f'draw{index}', num_folds)
    save_frozen(directory / 'draws.json', canonical_json_bytes(summary))
    return directory
