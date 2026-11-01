"""Handwriting-aware counterparts of TVA's tokenizers.

Construction is frozen (docs/DECISIONS.md, D024): every tokenizer is built
by TVA's own train(), unchanged. A handwriting tokenizer only adds one input,
the letter pairs allowed to join inside a token: the most connected letter
pairs in the handwriting evidence. Files, loading, encoding and decoding stay
TVA's.
"""

import gzip
import json
import os

from tva.tokenizers import (
    BigramTokenizer,
    BPETokenizer,
    UnigramTokenizer,
)

__all__ = [
    'HandwritingBigramTokenizer',
    'HandwritingBPETokenizer',
    'HandwritingUnigramTokenizer',
    'LegacyBigramTokenizer',
    'connected_pairs',
    'read_bigram_scores',
    'read_connectivity_lines',
]

# Defaults of the frozen construction (D024).
NUM_PAIRS = 145
MIN_COUNT = 20
RATE = 'exact_alignment_connected_rate'


def _read_json(path: str):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as f:
        return json.load(f)


def read_bigram_scores(path_evidence: str) -> list[dict]:
    '''Reads per-bigram connectivity scores from a JSON or JSON.GZ file.

    The file is either a list of score rows or an object with a 'scores' list
    (the DTLR export). Each row needs:
        pair: The two characters, e.g. 'th'.
        n_exact_alignment: How often the pair was observed with both
            characters reliably located in the handwriting.
        exact_alignment_connected_rate: Fraction of those observations in
            which the two characters were written connected.

    Args:
        path_evidence: Path to the connectivity evidence.

    Returns:
        List of score rows.
    '''
    evidence = _read_json(path_evidence)
    return evidence['scores'] if isinstance(evidence, dict) else evidence


def read_connectivity_lines(path_evidence: str) -> list[dict]:
    '''Reads per-line connectivity evidence from a JSON or JSON.GZ file.

    The file has a 'lines' list (the DTLR export). Each line needs:
        text: The transcription.
        edges: [i, left, right] for adjacent characters i and i + 1, where
            left/right are the IDs of the connected components (strokes) the
            two characters belong to, or None if they could not be located.

    Args:
        path_evidence: Path to the connectivity evidence.

    Returns:
        Lines as {'text', 'edges': {i: (left, right) or None}}.
    '''
    return [
        {
            'text': line['text'],
            'edges': {
                i: None if left is None else (left, right)
                for i, left, right in line['edges']
            },
        }
        for line in _read_json(path_evidence)['lines']
    ]


def connected_pairs(
    path_evidence: str,
    num_pairs: int,
    categories: list[str] | None = None,
    min_count: int = MIN_COUNT,
    rate: str = RATE,
) -> dict[str, float]:
    '''The num_pairs most connected letter pairs, most connected first.

    Pairs are letter pairs observed at least min_count times (exactly
    aligned), ranked by their connected rate; ties go alphabetically.

    Args:
        path_evidence: Connectivity scores, see read_bigram_scores().
        num_pairs: How many pairs to keep.
        categories: Characters both letters must belong to. Defaults to any.
        min_count: Minimum aligned observations of a pair. Defaults to 20.
        rate: Score column to rank by. Defaults to
            'exact_alignment_connected_rate'.

    Returns:
        Pair -> connected rate.

    Raises:
        ValueError: If fewer than num_pairs pairs qualify.
    '''
    rows = [
        row for row in read_bigram_scores(path_evidence)
        if row['pair'].isalpha()
        and (categories is None or set(row['pair']) <= set(categories))
        and row['n_exact_alignment'] >= min_count
    ]
    if len(rows) < num_pairs:
        raise ValueError(
            f'Only {len(rows)} letter pairs have {min_count} observations; '
            f'{num_pairs} requested. Lower min_count to allow more.'
        )
    rows.sort(key=lambda r: (-r[rate], r['pair']))
    return {row['pair']: row[rate] for row in rows[:num_pairs]}


def _evidence_path(dir_ds: str, path_evidence: str | None) -> str:
    if path_evidence is None:
        return os.path.join(dir_ds, 'connectivity.json.gz')
    return path_evidence


class HandwritingBigramTokenizer(BigramTokenizer):
    '''TVA's BigramTokenizer restricted to the most connected letter pairs.

    A vocabulary of size k holds the k most connected letter pairs (the
    allowed pairs); TVA's train() then counts and numbers them exactly as it
    does for its own bigrams. The 'dp' policy scores bigrams by their
    connected rate.
    '''

    def train(
        self,
        dir_ds: str,
        categories: list[str],
        size: int = 200,
        path_evidence: str | None = None,
        min_count: int = MIN_COUNT,
    ) -> None:
        '''Trains a vocabulary containing characters and connected bigrams.

        Args:
            dir_ds: Path to the directory containing 'train.json'.
            categories: List of allowed characters (unigrams).
            size: Target vocabulary size. Defaults to 200.
            path_evidence: Connectivity scores, see read_bigram_scores().
                Defaults to '<dir_ds>/connectivity.json.gz'.
            min_count: Minimum observations for a pair to be ranked.
                Defaults to 20.
        '''
        num_pairs = size - len(set(categories))
        pairs = connected_pairs(
            _evidence_path(dir_ds, path_evidence), num_pairs, categories,
            min_count,
        )
        super().train(
            dir_ds, categories, size, letters_only=True,
            allowed_pairs=set(pairs), name=f'handwriting_bigram{size}',
        )
        assert self.scores.keys() == pairs.keys(), \
            'an allowed pair never occurs in the training labels.'

        # the 'dp' policy uses the connected rate, not the text count
        self.scores = {t: pairs[t] for t in self.scores}
        dir_tokenizers = os.path.join(
            dir_ds, 'tokenizers', f'handwriting_bigram{size}'
        )
        with open(
            os.path.join(dir_ds, 'train.json'), 'r', encoding='utf-8'
        ) as f:
            folds = json.load(f)['annotations']
        for idx_fold in folds:
            path_scores = os.path.join(
                dir_tokenizers, f'{idx_fold}.scores.json'
            )
            with open(path_scores, 'w', encoding='utf-8') as f:
                json.dump(self.scores, f, ensure_ascii=False)


class HandwritingBPETokenizer(BPETokenizer):
    '''TVA's BPETokenizer whose pieces only join connected letter pairs.'''

    def train(
        self,
        dir_ds: str,
        categories: list[str],
        size: int = 200,
        path_evidence: str | None = None,
        num_pairs: int = NUM_PAIRS,
        min_count: int = MIN_COUNT,
    ) -> None:
        '''Trains TVA's letters-only BPE on text cut at unconnected pairs.

        Args:
            dir_ds: Path to the directory containing 'train.json'.
            categories: List of allowed characters for the initial alphabet.
            size: Target vocabulary size. Defaults to 200.
            path_evidence: Connectivity scores, see read_bigram_scores().
                Defaults to '<dir_ds>/connectivity.json.gz'.
            num_pairs: Number of most connected pairs allowed. Defaults to
                145 (the pairs written connected at least half of the time).
            min_count: Minimum observations for a pair to be ranked.
                Defaults to 20.
        '''
        pairs = connected_pairs(
            _evidence_path(dir_ds, path_evidence), num_pairs, categories,
            min_count,
        )
        super().train(
            dir_ds, categories, size, letters_only=True,
            allowed_pairs=set(pairs), name=f'handwriting_bpe{size}',
        )


class HandwritingUnigramTokenizer(UnigramTokenizer):
    '''TVA's UnigramTokenizer whose pieces only join connected letter pairs.'''

    def train(
        self,
        dir_ds: str,
        categories: list[str],
        size: int = 200,
        path_evidence: str | None = None,
        num_pairs: int = NUM_PAIRS,
        min_count: int = MIN_COUNT,
    ) -> None:
        '''Trains TVA's letters-only Unigram on text cut at unconnected pairs.

        Args:
            dir_ds: Path to the directory containing 'train.json'.
            categories: List of allowed characters for the initial alphabet.
            size: Target vocabulary size. Defaults to 200.
            path_evidence: Connectivity scores, see read_bigram_scores().
                Defaults to '<dir_ds>/connectivity.json.gz'.
            num_pairs: Number of most connected pairs allowed. Defaults to
                145 (the pairs written connected at least half of the time).
            min_count: Minimum observations for a pair to be ranked.
                Defaults to 20.
        '''
        pairs = connected_pairs(
            _evidence_path(dir_ds, path_evidence), num_pairs, categories,
            min_count,
        )
        super().train(
            dir_ds, categories, size, letters_only=True,
            allowed_pairs=set(pairs), name=f'handwriting_unigram{size}',
        )


class LegacyBigramTokenizer(BigramTokenizer):
    '''Loads bigram files saved before D024, with scores inside the file.

    Earlier handwriting and custom linguistic bigram files store each
    bigram's score as 'utility' (or as 'vocabulary' rows). They only load and
    encode; new files are trained with the classes above.
    '''

    def load(self, path_config: str) -> None:
        '''Loads vocabulary and embedded bigram scores from a JSON file.

        Args:
            path_config: Path to the tokenizer JSON file.
        '''
        with open(path_config, 'r', encoding='utf-8') as f:
            configs = json.load(f)

        self.vocab = configs['vocab']
        self.idx_token = {int(k): v for k, v in configs['idx_token'].items()}
        if 'utility' in configs:
            self.scores = configs['utility']
        else:
            self.scores = {
                row['token']: row['utility'] for row in configs['vocabulary']
            }
