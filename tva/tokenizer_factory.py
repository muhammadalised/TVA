"""Shared tokenizer initialization for recognition and analysis.

Original TVA keys ('char', 'bigram', 'bpe', 'unigram') delegate to TVA's own
factory and are the linguistic tokenizers; the keys below add the handwriting
tokenizers, the text-only Unigram EM ('unigram_em') and keys used by earlier
experiments.
"""

from typing import Any

from tva.tokenizers import (
    BigramTokenizer, BPETokenizer,
    get_tokenizer as get_original_tokenizer,
)
from tva.handwriting.common import resolve_artifact_path
from tva.handwriting_tokenizers import (
    HandwritingBigramTokenizer,
    HandwritingBPETokenizer, HandwritingUnigramTokenizer,
    UnigramEMTokenizer,
)


def get_tokenizer(tokenizer: str) -> Any:
    """Use thesis extensions or delegate to the original TVA factory."""
    extensions = {
        'handwriting_bigram_greedy': lambda: HandwritingBigramTokenizer('greedy'),
        'handwriting_bigram_utility': lambda: HandwritingBigramTokenizer('utility-based'),
        'handwriting_bpe': HandwritingBPETokenizer,
        'handwriting_unigram': HandwritingUnigramTokenizer,
        'unigram_em': UnigramEMTokenizer,
        # Keys recorded by earlier experiments; their files load with these classes.
        'handwriting_bigram': lambda: HandwritingBigramTokenizer('utility-based'),
        'linguistic_bigram': lambda: HandwritingBigramTokenizer('utility-based'),
        'linguistic_bigram_greedy': lambda: HandwritingBigramTokenizer('greedy'),
        'linguistic_bigram_matched': BigramTokenizer,
        'linguistic_bpe_matched': BPETokenizer,
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
