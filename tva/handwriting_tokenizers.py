"""Handwriting-aware counterparts of TVA's tokenizers.

Each class subclasses the original TVA tokenizer and only replaces train():
vocabularies are learned from handwriting connectivity evidence instead of
label frequencies. Files, loading, encoding and decoding stay TVA's.
UnigramEMTokenizer runs the handwriting Unigram's EM on the labels alone, as
the linguistic partner that lacks only the connectivity.
"""

import gzip
import json
import os

from loguru import logger
from tokenizers import Tokenizer, decoders, models

from tva.handwriting.training import (
    collect_candidates,
    frequent_spans,
    train_bpe,
    train_unigram,
)
from tva.tokenizers import (
    BigramTokenizer,
    BPETokenizer,
    UnigramTokenizer,
    letters_only_pre_tokenizer,
)

__all__ = [
    'HandwritingBigramTokenizer',
    'HandwritingBPETokenizer',
    'HandwritingUnigramTokenizer',
    'UnigramEMTokenizer',
    'bpe_from_merges',
    'read_bigram_scores',
    'read_connectivity_lines',
    'unigram_from_log_probs',
]

POLICIES = ('greedy', 'utility-based')


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


def _letters_only(tokenizer: Tokenizer) -> Tokenizer:
    '''Gives a built model the pipeline of TVA's letters_only training.'''
    tokenizer.add_special_tokens(['<PAD>'])
    tokenizer.pre_tokenizer = letters_only_pre_tokenizer()
    tokenizer.decoder = decoders.Fuse()
    return tokenizer


def bpe_from_merges(chars: list[str], merges: list[tuple[str, str]]) -> Tokenizer:
    '''Builds a letters-only BPE tokenizer from ordered merges.

    IDs follow TVA's layout: '<PAD>' (CTC blank) is 0, then the characters,
    then one token per merge in learned order.
    '''
    tokens = ['<PAD>', *chars, *(left + right for left, right in merges)]
    vocab = {t: i for i, t in enumerate(tokens)}
    return _letters_only(Tokenizer(models.BPE(vocab=vocab, merges=list(merges))))


def unigram_from_log_probs(
    chars: list[str], log_probs: dict[str, float]
) -> Tokenizer:
    '''Builds a letters-only Unigram tokenizer from learned log-probabilities.

    IDs follow TVA's layout: '<PAD>' (CTC blank) is 0, then the characters,
    then the learned pieces in alphabetical order.
    '''
    pieces = sorted(t for t in log_probs if len(t) > 1)
    # rounded so that files are identical across platforms; characters
    # never seen in training are always single tokens, so 0.0 is unused
    vocab = [('<PAD>', 0.0)]
    vocab += [(c, round(log_probs.get(c, 0.0), 10)) for c in chars]
    vocab += [(t, round(log_probs[t], 10)) for t in pieces]
    return _letters_only(
        Tokenizer(models.Unigram(vocab, unk_id=None, byte_fallback=False))
    )


class HandwritingBigramTokenizer(BigramTokenizer):
    '''Implements a Unigram + Bigram Tokenizer with handwriting-aware bigrams.

    Identical to TVA's BigramTokenizer (vocabulary layout, files, encoding and
    decoding) except for how bigrams are selected: instead of the most frequent
    bigrams in the labels, it keeps the letter bigrams most often written
    connected in a handwriting dataset.

    Segmentation policies:
        'greedy': Left-to-right bigram matching, exactly as BigramTokenizer.
        'utility-based': Non-overlapping bigrams with the highest total
            connectivity rate (ties prefer more bigrams).

    Attributes:
        vocab: Mapping from token to ID.
        idx_token: Mapping from ID to token.
        utility: Connectivity rate of each bigram in the vocabulary.
        policy: Segmentation policy used by encode().
        size: Property return of the vocabulary size.
    '''

    def __init__(self, policy: str = 'greedy') -> None:
        super().__init__()
        if policy not in POLICIES:
            raise ValueError(f'policy must be one of {POLICIES}, got {policy!r}')

        self.policy = policy
        self.utility: dict[str, float] = {}

    def train(
        self,
        dir_ds: str,
        categories: list[str],
        size: int = 200,
        path_evidence: str | None = None,
        min_count: int = 20,
    ) -> None:
        '''Trains a vocabulary containing characters and top connected bigrams.

        Bigrams are ranked by connectivity rate (ties alphabetically) among
        letter pairs observed at least min_count times. The evidence describes
        the whole handwriting dataset, so every fold gets the same vocabulary.

        Args:
            dir_ds: Path to the directory containing 'train.json' (used to
                read fold info). Tokenizers are saved under it.
            categories: List of allowed characters (unigrams).
            size: Target vocabulary size. Defaults to 200.
            path_evidence: Connectivity scores, see read_bigram_scores().
                Defaults to '<dir_ds>/connectivity.json.gz'.
            min_count: Minimum observations for a bigram to be ranked.
                Defaults to 20.
        '''
        if path_evidence is None:
            path_evidence = os.path.join(dir_ds, 'connectivity.json.gz')

        dir_tokenizers = os.path.join(
            dir_ds, 'tokenizers', f'handwriting_bigram{size}'
        )
        os.makedirs(dir_tokenizers, exist_ok=True)

        with open(
            os.path.join(dir_ds, 'train.json'), 'r', encoding='utf-8'
        ) as f:
            num_fold = json.load(f)['info']['num_fold']

        chars = sorted(list(set(categories)))
        limit_bigram = size - len(chars)
        assert limit_bigram > 0, 'size smaller than number of categories.'

        # rank connected letter bigrams
        rows = [
            row for row in read_bigram_scores(path_evidence)
            if row['pair'].isalpha()
            and set(row['pair']) <= set(chars)
            and row['n_exact_alignment'] >= min_count
        ]
        rows.sort(
            key=lambda r: (-r['exact_alignment_connected_rate'], r['pair'])
        )
        rows_top = rows[:limit_bigram]

        # bigram IDs follow alphabetical order, as in the frozen ED vocabulary
        bigrams_top = sorted(row['pair'] for row in rows_top)

        self.vocab = {t: i for i, t in enumerate(chars + bigrams_top)}
        self.idx_token = {v: k for k, v in self.vocab.items()}
        self.utility = {
            row['pair']: row['exact_alignment_connected_rate']
            for row in sorted(rows_top, key=lambda r: r['pair'])
        }

        for idx_fold in range(num_fold):
            path_save = os.path.join(dir_tokenizers, f'{idx_fold}.json')
            with open(path_save, 'w', encoding='utf-8') as f:
                json.dump(
                    {
                        'vocab': self.vocab,
                        'idx_token': self.idx_token,
                        'utility': self.utility,
                    },
                    f,
                    ensure_ascii=False,
                )

        logger.info(
            f'HandwritingBigramTokenizers are saved at {dir_tokenizers}.'
        )

    def load(self, path_config: str) -> None:
        '''Loads vocabulary and bigram utilities from a JSON file.

        Args:
            path_config: Path to the tokenizer JSON file.
        '''
        with open(path_config, 'r', encoding='utf-8') as f:
            configs = json.load(f)

        self.vocab = configs['vocab']
        self.idx_token = {int(k): v for k, v in configs['idx_token'].items()}
        if 'utility' in configs:
            self.utility = configs['utility']
        else:
            # archived tokenizers list bigrams as 'vocabulary' rows
            self.utility = {
                row['token']: row['utility'] for row in configs['vocabulary']
            }

        logger.info(
            f'HandwritingBigramTokenizer ({self.policy}) is loaded from '
            f'{path_config}.'
        )

    def encode(self, text: str) -> list[int]:
        '''Encodes text using the selected segmentation policy.

        Args:
            text: Input text string.

        Returns:
            List of token IDs.

        Raises:
            ValueError: If the tokenizer has not been trained or loaded.
        '''
        if self.policy == 'greedy':
            return super().encode(text)

        if not self.vocab:
            raise ValueError('Tokenizer not trained or loaded.')

        # best[i] = (total utility, bigram count, token IDs) for text[i:]
        best = [(0.0, 0, [])] * (len(text) + 1)

        for i in range(len(text) - 1, -1, -1):
            utility, count, ids = best[i + 1]
            winner = (utility, count, [self.vocab[text[i]]] + ids)

            bigram = text[i : i + 2]
            if len(bigram) == 2 and bigram in self.utility:
                utility, count, ids = best[i + 2]
                candidate = (
                    utility + self.utility[bigram],
                    count + 1,
                    [self.vocab[bigram]] + ids,
                )
                if candidate[:2] >= winner[:2]:
                    winner = candidate

            best[i] = winner

        return best[0][2]


class _HandwritingSubwordTokenizer:
    """Shared training steps of the handwriting BPE and Unigram tokenizers."""

    kind: str

    def train(
        self,
        dir_ds: str,
        categories: list[str],
        size: int = 200,
        path_evidence: str | None = None,
        min_count: int = 20,
        rate_threshold: float = 0.5,
        max_length: int = 6,
    ) -> None:
        '''Learns pieces from handwriting connectivity.

        Candidate pieces are letter spans of 2..max_length characters observed
        at least min_count times and written connected at least rate_threshold
        of the time. The vocabulary then holds the blank, the categories and
        size - len(categories) learned pieces. Every non-letter character
        stays its own token, like letters_only=True for TVA's tokenizers.
        The evidence describes the whole handwriting dataset, so every fold
        gets the same vocabulary.

        Args:
            dir_ds: Path to the directory containing 'train.json' (used to
                read fold info). Tokenizers are saved under it.
            categories: List of allowed characters for the alphabet.
            size: Target vocabulary size. Defaults to 200.
            path_evidence: Per-line connectivity, see read_connectivity_lines().
                Defaults to '<dir_ds>/connectivity.json.gz'.
            min_count: Minimum aligned observations of a piece. Defaults to 20.
            rate_threshold: Minimum connected rate of a piece. Defaults to 0.5.
            max_length: Maximum piece length. Defaults to 6.
        '''
        if path_evidence is None:
            path_evidence = os.path.join(dir_ds, 'connectivity.json.gz')

        dir_tokenizers = os.path.join(
            dir_ds, 'tokenizers', f'handwriting_{self.kind}{size}'
        )
        os.makedirs(dir_tokenizers, exist_ok=True)

        with open(
            os.path.join(dir_ds, 'train.json'), 'r', encoding='utf-8'
        ) as f:
            num_fold = json.load(f)['info']['num_fold']

        chars = sorted(set(categories) - {''})
        limit_piece = size - 1 - len(chars)
        assert limit_piece > 0, 'size smaller than number of categories.'

        # learn only from lines that the categories can spell
        lines = [
            line for line in read_connectivity_lines(path_evidence)
            if set(line['text']) <= set(chars)
        ]
        candidates = collect_candidates(
            lines, max_length, min_count, rate_threshold
        )

        self.tokenizer = self._build(lines, candidates, chars, limit_piece)

        for idx_fold in range(num_fold):
            self.tokenizer.save(
                os.path.join(dir_tokenizers, f'{idx_fold}.json')
            )

        self._size = self.tokenizer.get_vocab_size()
        logger.info(
            f'{type(self).__name__}s are saved at {dir_tokenizers}.'
        )


class HandwritingBPETokenizer(_HandwritingSubwordTokenizer, BPETokenizer):
    '''BPE whose merges are ranked by connected occurrences.

    Like BPE training, it repeatedly merges the most supported adjacent pair,
    but support counts only occurrences written connected, and only candidate
    pieces may form. Encoding is TVA's Hugging Face BPE.
    '''

    kind = 'bpe'

    @staticmethod
    def _build(lines, candidates, chars, limit_piece) -> Tokenizer:
        return bpe_from_merges(chars, train_bpe(lines, candidates, limit_piece))


class HandwritingUnigramTokenizer(_HandwritingSubwordTokenizer, UnigramTokenizer):
    '''Unigram LM whose pieces are seeded and initialized by connectivity.

    Candidates start from their connected counts; EM on the training text and
    likelihood-based pruning then reduce them to the budget. Encoding is
    TVA's Hugging Face Unigram (Viterbi).
    '''

    kind = 'unigram'

    @staticmethod
    def _build(lines, candidates, chars, limit_piece) -> Tokenizer:
        seeds = {t: row['n_connected'] for t, row in candidates.items()}
        return unigram_from_log_probs(
            chars, train_unigram(lines, seeds, limit_piece)
        )


class UnigramEMTokenizer(UnigramTokenizer):
    '''The handwriting Unigram's EM trained on text alone.

    Linguistic partner of HandwritingUnigramTokenizer: every letter span seen
    often enough is a candidate and starts from its frequency instead of its
    connected count. Lines, EM, pruning and encoding are the same, so the two
    vocabularies differ only by the connectivity evidence.
    '''

    def train(
        self,
        dir_ds: str,
        categories: list[str],
        size: int = 200,
        min_count: int = 20,
        max_length: int = 6,
    ) -> None:
        '''Learns pieces from the labels of each fold.

        Like letters_only=True for TVA's tokenizers, only labels spelled by
        the categories are used and every non-letter character stays its own
        token. Repeated labels are kept, as the handwriting Unigram sees each
        handwritten line once.

        Args:
            dir_ds: Path to the directory containing 'train.json'.
            categories: List of allowed characters for the alphabet.
            size: Target vocabulary size. Defaults to 200.
            min_count: Minimum occurrences of a piece. Defaults to 20.
            max_length: Maximum piece length. Defaults to 6.
        '''
        dir_tokenizers = os.path.join(dir_ds, 'tokenizers', f'unigram_em{size}')
        os.makedirs(dir_tokenizers, exist_ok=True)

        with open(
            os.path.join(dir_ds, 'train.json'), 'r', encoding='utf-8'
        ) as f:
            annos = json.load(f)['annotations']

        chars = sorted(set(categories) - {''})
        limit_piece = size - 1 - len(chars)
        assert limit_piece > 0, 'size smaller than number of categories.'

        for idx_fold, annos_fold in annos.items():
            lines = [
                {'text': anno['label']} for anno in annos_fold
                if set(anno['label']) <= set(chars)
            ]
            seeds = frequent_spans(lines, max_length, min_count)
            self.tokenizer = unigram_from_log_probs(
                chars, train_unigram(lines, seeds, limit_piece)
            )
            self.tokenizer.save(
                os.path.join(dir_tokenizers, f'{idx_fold}.json')
            )

        self._size = self.tokenizer.get_vocab_size()
        logger.info(f'UnigramEMTokenizers are saved at {dir_tokenizers}.')
