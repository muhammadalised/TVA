"""Shared tokenizer initialization for recognition and analysis.

Original linguistic keys delegate to TVA; matched text controls use shared trainers.
Historical custom linguistic keys are retained for saved experiments.
"""

from typing import Any

from tva.tokenizers import (
    get_tokenizer as get_original_tokenizer,
)
from tva.handwriting.common import resolve_artifact_path
from tva.handwriting_tokenizers import (
    HandwritingBigramTokenizer, GreedyHandwritingBigramTokenizer,
    HandwritingBPETokenizer, HandwritingUnigramTokenizer,
)
from tva.linguistic_tokenizers import LinguisticBPETokenizer, LinguisticUnigramTokenizer
from tva.tokenizer_compatibility import (
    LinguisticBigramTokenizer, GreedyLinguisticBigramTokenizer,
)


def get_tokenizer(tokenizer: str) -> Any:
    """Use thesis extensions or delegate to the original TVA factory."""
    extensions = {
        'linguistic_bpe': LinguisticBPETokenizer,
        'linguistic_unigram': LinguisticUnigramTokenizer,
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
