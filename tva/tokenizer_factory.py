"""Shared tokenizer initialization for recognition and analysis.

Linguistic methods delegate to the unchanged original TVA factory.
Historical custom linguistic keys are retained for saved experiments.
"""

from typing import Any

from loguru import logger

from tva.tokenizers import (
    CharacterTokenizer as OriginalCharacterTokenizer,
    get_tokenizer as get_original_tokenizer,
)
from tva.handwriting.common import resolve_artifact_path
from tva.handwriting_tokenizers import (
    HandwritingBigramTokenizer, GreedyHandwritingBigramTokenizer,
    HandwritingBPETokenizer, HandwritingUnigramTokenizer,
)
from tva.tokenizer_compatibility import (
    LinguisticBigramTokenizer, GreedyLinguisticBigramTokenizer,
)


class CharacterTokenizer(OriginalCharacterTokenizer):
    """Original character encoding with explicit alphabet initialization."""

    def load_categories(self, categories: list[str]) -> None:
        '''Initialize a character vocabulary from an explicit ordered list.

        The first category must be the empty CTC blank token. This path is
        useful when the complete alphabet is part of a frozen experiment
        configuration and no data-derived tokenizer artifact is required.

        Args:
            categories: Ordered character vocabulary including blank ID 0.

        Raises:
            ValueError: If the vocabulary is empty, duplicated, malformed, or
                does not reserve ID 0 for the CTC blank.
        '''
        if not isinstance(categories, list) or not categories:
            raise ValueError('categories must be a non-empty list')
        if any(not isinstance(char, str) for char in categories):
            raise ValueError('every category must be a string')
        if categories[0] != '':
            raise ValueError('CTC blank must be the empty category at ID 0')
        if any(char == '' for char in categories[1:]):
            raise ValueError('CTC blank may appear only at ID 0')
        if len(set(categories)) != len(categories):
            raise ValueError('categories must not contain duplicates')

        self.vocab = {char: i for i, char in enumerate(categories)}
        self.idx_char = {index: char for char, index in self.vocab.items()}

        logger.info(
            f'CharacterTokenizer initialized from {len(categories)} '
            'configured categories.'
        )


def get_tokenizer(tokenizer: str) -> Any:
    """Use thesis extensions or delegate to the original TVA factory."""
    extensions = {
        'char': CharacterTokenizer,
        'handwriting_bpe': HandwritingBPETokenizer,
        'handwriting_unigram': HandwritingUnigramTokenizer,
        'handwriting_bigram': HandwritingBigramTokenizer,
        'handwriting_bigram_greedy': GreedyHandwritingBigramTokenizer,
        'linguistic_bigram': LinguisticBigramTokenizer,
        'linguistic_bigram_greedy': GreedyLinguisticBigramTokenizer,
    }
    if tokenizer in extensions:
        return extensions[tokenizer]()
    return get_original_tokenizer(tokenizer)


def resolve_tokenizer_path(path_config: str, idx_fold: int) -> str:
    """Resolve a shared tokenizer file or a legacy fold directory."""
    path = resolve_artifact_path(path_config)
    if path.is_file():
        return str(path)
    return str(resolve_artifact_path(path / f'{idx_fold}.json'))
