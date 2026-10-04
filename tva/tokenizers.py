import json
import os
from collections import Counter
from typing import Any

from loguru import logger
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

__all__ = [
    'CharacterTokenizer',
    'BigramTokenizer',
    'BPETokenizer',
    'UnigramTokenizer'
    'get_tokenizer',
]


class CharacterTokenizer:
    '''Implements a simple Character-Level Tokenizer.

    Strategy: Vocabulary is defined strictly by the provided 'categories' list.
    Encoding is done via direct list indexing.

    Attributes:
        vocab: Mapping from character to ID.
        idx_char: Mapping from ID to character.
        size: Property return of the vocabulary size.
    '''

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.idx_char: dict[int, str] = {}

    @property
    def size(self) -> int:
        '''The size of the vocabulary.

        Raises:
            ValueError: If the tokenizer has not been trained or loaded.
        '''
        if not self.vocab:
            raise ValueError('Tokenizer not trained or loaded.')

        return len(self.vocab)

    def train(self, dir_ds: str, categories: list[str]) -> None:
        '''Constructs the tokenizer from the provided categories list.

        Saves separate tokenizer JSON files for each fold found in the dataset
        configuration.

        Args:
            dir_ds: Path to the directory containing the dataset (used to read
                fold info).
            categories: List of all allowed characters. Must cover all
                characters in the dataset.
        '''
        dir_tokenizers = os.path.join(dir_ds, 'tokenizers', 'char')
        os.makedirs(dir_tokenizers, exist_ok=True)

        self.vocab = {char: i for i, char in enumerate(categories)}
        self.idx_char = {v: k for k, v in self.vocab.items()}

        with open(os.path.join(dir_ds, 'train.json'), 'r') as f:
            num_fold = json.load(f)['info']['num_fold']

        for idx_fold in range(num_fold):
            path_save = os.path.join(dir_tokenizers, f'{idx_fold}.json')

            with open(path_save, 'w', encoding='utf-8') as f:
                json.dump(
                    {'vocab': self.vocab, 'idx_char': self.idx_char},
                    f,
                    ensure_ascii=False,
                )

        logger.info(f'CharacterTokenizers are saved at {dir_tokenizers}.')

    def load(self, path_config: str) -> None:
        '''Loads vocabulary from a JSON file.

        Args:
            path_config: Path to the tokenizer JSON file.
        '''
        with open(path_config, 'r', encoding='utf-8') as f:
            configs = json.load(f)

        self.vocab = configs['vocab']
        self.idx_char = {int(k): v for k, v in configs['idx_char'].items()}

        logger.info(f'CharacterTokenizer is loaded from {path_config}.')

    def encode(self, text: str) -> list[int]:
        '''Encodes text using strict list indexing.

        Args:
            text: Input text string.

        Returns:
            List of token IDs.

        Raises:
            ValueError: If the tokenizer has not been trained or loaded.
        '''
        if not self.vocab:
            raise ValueError('Tokenizer not trained or loaded.')

        return [self.vocab[char] for char in text]

    def decode(self, ids: list[int]) -> str:
        '''Decodes token IDs back to string.

        Args:
            ids: List of token IDs.

        Returns:
            Decoded string.

        Raises:
            ValueError: If the tokenizer has not been trained or loaded.
        '''
        if not self.vocab:
            raise ValueError('Tokenizer not trained or loaded.')

        return ''.join([self.idx_char[i] for i in ids])


class BigramTokenizer:
    '''Implements a simplified Hybrid Unigram + Bigram Tokenizer.

    Attributes:
        vocab: Mapping from token to ID.
        idx_token: Mapping from ID to token.
        size: Property return of the vocabulary size.
    '''

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.idx_token: dict[int, str] = {}

    @property
    def size(self) -> int:
        '''The size of the vocabulary.

        Raises:
            ValueError: If the tokenizer has not been trained or loaded.
        '''
        if not self.vocab:
            raise ValueError('Tokenizer not trained or loaded.')

        return len(self.vocab)

    def train(
        self, dir_ds: str, categories: list[str], size: int = 200
    ) -> None:
        '''Trains a vocabulary containing characters and top frequent bigrams.

        Args:
            dir_ds: Path to the directory containing 'train.json'.
            categories: List of allowed characters (unigrams).
            size: Target vocabulary size. Defaults to 200.
        '''
        dir_tokenizers = os.path.join(dir_ds, 'tokenizers', f'bigram{size}')
        os.makedirs(dir_tokenizers, exist_ok=True)

        with open(
            os.path.join(dir_ds, 'train.json'), 'r', encoding='utf-8'
        ) as f:
            annos = json.load(f)['annotations']

        for idx_fold, annos_fold in annos.items():
            texts = list(set(anno['label'] for anno in annos_fold))
            chars = sorted(list(set(categories)))
            counts_bigram = Counter()

            # count bigrams in text labels
            for text in texts:
                for i in range(len(text) - 1):
                    bigram = text[i : i + 2]
                    counts_bigram[bigram] += 1

            limit_bigram = size - len(chars)
            assert limit_bigram > 0, 'size smaller than number of categories.'

            bigrams_top = [
                t for t, c in counts_bigram.most_common(limit_bigram)
            ]

            self.vocab = {t: i for i, t in enumerate(chars + bigrams_top)}
            self.idx_token = {v: k for k, v in self.vocab.items()}

            path_save = os.path.join(dir_tokenizers, f'{idx_fold}.json')
            with open(path_save, 'w', encoding='utf-8') as f:
                json.dump(
                    {'vocab': self.vocab, 'idx_token': self.idx_token},
                    f,
                    ensure_ascii=False,
                )

        logger.info(f'BigramTokenizers are saved at {dir_tokenizers}.')

    def load(self, path_config: str) -> None:
        '''Loads vocabulary from a JSON file.

        Args:
            path_config: Path to the tokenizer JSON file.
        '''
        with open(path_config, 'r', encoding='utf-8') as f:
            configs = json.load(f)

        self.vocab = configs['vocab']
        self.idx_token = {int(k): v for k, v in configs['idx_token'].items()}

        logger.info(f'BigramTokenizer is loaded from {path_config}.')

    def encode(self, text: str) -> list[int]:
        '''Encodes text using greedy bigram matching.

        Args:
            text: Input text string.

        Returns:
            List of token IDs.

        Raises:
            ValueError: If the tokenizer has not been trained or loaded.
        '''
        if not self.vocab:
            raise ValueError('Tokenizer not trained or loaded.')

        tokens = []
        i = 0
        n = len(text)

        while i < n:
            # check for bigram match first
            if i + 1 < n:
                bigram = text[i : i + 2]
                if bigram in self.vocab:
                    tokens.append(self.vocab[bigram])
                    i += 2
                    continue

            # fallback to unigram
            char = text[i]
            tokens.append(self.vocab[char])
            i += 1

        return tokens

    def decode(self, ids: list[int]) -> str:
        '''Decodes token IDs back to string.

        Args:
            ids: List of token IDs.

        Returns:
            Decoded string.

        Raises:
            ValueError: If the tokenizer has not been trained or loaded.
        '''
        if not self.vocab:
            raise ValueError('Tokenizer not trained or loaded.')

        return ''.join([self.idx_token[i] for i in ids])


class BPETokenizer:
    '''Implements a trainable Character-Level Tokenizer for Charformer.

    Strategy: Trains a vocabulary using Byte-Pair Encoding (BPE).

    Attributes:
        tokenizer: The underlying Hugging Face Tokenizer.
        _size: The current size of the vocabulary.
        size: Property return of the vocabulary size.
    '''

    def __init__(self) -> None:
        self.tokenizer: Tokenizer | None = None
        self._size: int = 0

    @property
    def size(self) -> int:
        '''The size of the vocabulary.

        Raises:
            ValueError: If the tokenizer has not been trained or loaded.
        '''
        if not self._size:
            raise ValueError('Tokenizer not trained or loaded.')

        return self._size

    def train(
        self, dir_ds: str, categories: list[str], size: int = 200
    ) -> None:
        '''Trains a BPE tokenizer using an iterator of texts.

        Args:
            dir_ds: Path to the directory containing 'train.json'.
            categories: List of allowed characters for the initial alphabet.
            size: Target vocabulary size. Defaults to 200.
        '''
        dir_tokenizers = os.path.join(dir_ds, 'tokenizers', f'bpe{size}')
        os.makedirs(dir_tokenizers, exist_ok=True)

        with open(
            os.path.join(dir_ds, 'train.json'), 'r', encoding='utf-8'
        ) as f:
            annos = json.load(f)['annotations']

        for idx_fold, annos_fold in annos.items():
            texts = list(set(anno['label'] for anno in annos_fold))

            self.tokenizer = Tokenizer(models.BPE())
            self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

            trainer = trainers.BpeTrainer(
                vocab_size=size,
                min_frequency=1,
                special_tokens=['<PAD>'],
                initial_alphabet=categories,
            )

            self.tokenizer.train_from_iterator(texts, trainer)
            self.tokenizer.decoder = decoders.BPEDecoder()
            self.tokenizer.save(
                os.path.join(dir_tokenizers, f'{idx_fold}.json')
            )

        # set vocab size from the last fold trained
        self._size = self.tokenizer.get_vocab_size()
        logger.info(f'BPETokenizers are saved at {dir_tokenizers}.')

    def load(self, path_json: str) -> None:
        '''Loads a pre-trained tokenizer from a JSON file.

        Args:
            path_json: Path to the tokenizer JSON file.
        '''
        self.tokenizer = Tokenizer.from_file(path_json)
        self._size = self.tokenizer.get_vocab_size()
        logger.info(f'BPETokenizer is loaded from {path_json}.')

    def encode(self, text: str) -> list[int]:
        '''Encodes text using the trained BPE model.

        Args:
            text: Input text string.

        Returns:
            List of token IDs.

        Raises:
            ValueError: If the tokenizer has not been loaded.
        '''
        if not self.tokenizer:
            raise ValueError('Tokenizer not loaded.')

        return self.tokenizer.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        '''Decodes token IDs back to string.

        Args:
            ids: List of token IDs.

        Returns:
            Decoded string.

        Raises:
            ValueError: If the tokenizer has not been loaded.
        '''
        if not self.tokenizer:
            raise ValueError('Tokenizer not loaded.')

        return self.tokenizer.decode(ids)


class UnigramTokenizer:
    '''Implements a trainable Character-Level Tokenizer for Charformer.

    Strategy: Trains a vocabulary using the Unigram language model.

    Attributes:
        tokenizer: The underlying Hugging Face Tokenizer.
        _size: The current size of the vocabulary.
        size: Property return of the vocabulary size.
    '''

    def __init__(self) -> None:
        self.tokenizer: Tokenizer | None = None
        self._size: int = 0

    @property
    def size(self) -> int:
        '''The size of the vocabulary.

        Raises:
            ValueError: If the tokenizer has not been trained or loaded.
        '''
        if not self._size:
            raise ValueError('Tokenizer not trained or loaded.')

        return self._size

    def train(
        self, dir_ds: str, categories: list[str], size: int = 200
    ) -> None:
        '''Trains a Unigram tokenizer using an iterator of texts.

        Args:
            dir_ds: Path to the directory containing 'train.json'.
            categories: List of allowed characters for the initial alphabet.
            size: Target vocabulary size. Defaults to 200.
        '''
        dir_tokenizers = os.path.join(dir_ds, 'tokenizers', f'unigram{size}')
        os.makedirs(dir_tokenizers, exist_ok=True)

        with open(
            os.path.join(dir_ds, 'train.json'), 'r', encoding='utf-8'
        ) as f:
            annos = json.load(f)['annotations']

        for idx_fold, annos_fold in annos.items():
            texts = list(set(anno['label'] for anno in annos_fold))

            self.tokenizer = Tokenizer(models.Unigram())
            self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

            trainer = trainers.UnigramTrainer(
                vocab_size=size,
                special_tokens=['<PAD>'],
                initial_alphabet=categories,
            )

            self.tokenizer.train_from_iterator(texts, trainer)
            self.tokenizer.decoder = decoders.BPEDecoder()
            self.tokenizer.save(
                os.path.join(dir_tokenizers, f'{idx_fold}.json')
            )

        # set vocab size from the last fold trained
        self._size = self.tokenizer.get_vocab_size()
        logger.info(f'UnigramTokenizers are saved at {dir_tokenizers}.')

    def load(self, path_json: str) -> None:
        '''Loads a pre-trained tokenizer from a JSON file.

        Args:
            path_json: Path to the tokenizer JSON file.
        '''
        self.tokenizer = Tokenizer.from_file(path_json)
        self._size = self.tokenizer.get_vocab_size()
        logger.info(f'UnigramTokenizer is loaded from {path_json}.')

    def encode(self, text: str) -> list[int]:
        '''Encodes text using the trained Unigram model.

        Args:
            text: Input text string.

        Returns:
            List of token IDs.

        Raises:
            ValueError: If the tokenizer has not been loaded.
        '''
        if not self.tokenizer:
            raise ValueError('Tokenizer not loaded.')

        return self.tokenizer.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        '''Decodes token IDs back to string.

        Args:
            ids: List of token IDs.

        Returns:
            Decoded string.

        Raises:
            ValueError: If the tokenizer has not been loaded.
        '''
        if not self.tokenizer:
            raise ValueError('Tokenizer not loaded.')

        return self.tokenizer.decode(ids)


def get_tokenizer(tokenizer: str) -> Any:
    '''Factory function to get a tokenizer instance.

    Args:
        tokenizer: Key of the tokenizer. Options: 'char', 'bigram', 'bpe',
            'embed'.

    Returns:
        The requested tokenizer object.
    '''
    match tokenizer:
        case 'char':
            return CharacterTokenizer()
        case 'bigram':
            return BigramTokenizer()
        case 'bpe':
            return BPETokenizer()
        case 'unigram':
            return UnigramTokenizer()
        case _:
            raise ValueError(
                f'Unknown loss function: "{tokenizer}". '
                'Supported: ["char", "bigram", "bpe", "unigram"]'
            )
