"""Identical text segmentation for matched linguistic and handwriting subwords."""

import math
import unicodedata

from tva.handwriting.common import require


class SubwordTokenizer:
    """Ordered BPE merges or maximum-score Unigram Viterbi segmentation."""

    kind: str

    def __init__(self) -> None:
        self.model = {}
        self.source_model = {}
        self.vocab = {}
        self.idx_token = {}


    @property
    def size(self) -> int:
        if not self.model:
            raise ValueError('Tokenizer not trained or loaded.')
        return len(self.vocab)


    def segment(self, text: str) -> dict:
        if not self.model:
            raise ValueError('Tokenizer not trained or loaded.')
        if not isinstance(text, str):
            raise TypeError('text must be a string')
        text = unicodedata.normalize('NFC', text)
        missing = sorted(set(text) - set(self.vocab))
        require(not missing, f'characters are absent from tokenizer vocabulary: {missing!r}')
        if self.kind == 'bpe':
            sequence = [(c, i, i + 1) for i, c in enumerate(text)]
            for merge in self.source_model['merges']:
                merged, i = [], 0
                while i < len(sequence):
                    if (i + 1 < len(sequence)
                            and sequence[i][0] == merge['left']
                            and sequence[i + 1][0] == merge['right']):
                        merged.append((merge['token'], sequence[i][1], sequence[i + 1][2]))
                        i += 2
                    else:
                        merged.append(sequence[i])
                        i += 1
                sequence = merged
        else:
            # Match DTLR's start/end traversal and strict improvement tie rule.
            best = [-math.inf] * (len(text) + 1)
            back = [None] * (len(text) + 1)
            best[0] = 0.0
            for start in range(len(text)):
                for end in range(start + 1, min(len(text), start + self._max_length) + 1):
                    token = text[start:end]
                    if token in self._scores:
                        candidate = best[start] + self._scores[token]
                        if candidate > best[end]:
                            best[end], back[end] = candidate, (token, start, end)
            sequence, end = [], len(text)
            while end:
                step = back[end]
                require(step is not None, 'text cannot be segmented')
                sequence.append(step)
                end = step[1]
            sequence.reverse()
        return {
            'text': text, 'tokens': [t for t, _, _ in sequence],
            'segments': [{'token': t, 'start': s, 'end': e} for t, s, e in sequence],
        }


    def encode(self, text: str) -> list[int]:
        return [self.vocab[t] for t in self.segment(text)['tokens']]


    def decode(self, ids: list[int]) -> str:
        if not self.model:
            raise ValueError('Tokenizer not trained or loaded.')
        try:
            return ''.join(self.idx_token[i] for i in ids)
        except KeyError as error:
            raise ValueError(f'unknown token ID: {error.args[0]}') from error
