"""Shared tokenizer initialization for recognition and analysis.

Original TVA keys ('char', 'bigram', 'bpe', 'unigram') delegate to TVA's own
factory. Since D024 every tokenizer file is in TVA's format, so the frozen
handwriting tokenizers load with these keys too; 'bigram_dp' selects the
dp segmentation policy (D025). The other keys load files of earlier
experiments.
"""

from typing import Any

from tva.tokenizers import (
    BigramTokenizer, BPETokenizer, UnigramTokenizer,
    get_tokenizer as get_original_tokenizer,
)
from tva.handwriting.common import resolve_artifact_path
from tva.handwriting_tokenizers import (
    HandwritingBigramTokenizer,
    HandwritingBPETokenizer, HandwritingUnigramTokenizer,
    LegacyBigramTokenizer,
)


def get_tokenizer(tokenizer: str) -> Any:
    """Use thesis extensions or delegate to the original TVA factory."""
    extensions = {
        'bigram_dp': lambda: BigramTokenizer('dp'),
        # Keys recorded by earlier experiments; their files load with these classes.
        'handwriting_bigram_greedy': lambda: HandwritingBigramTokenizer('greedy'),
        'handwriting_bigram_utility': lambda: LegacyBigramTokenizer('dp'),
        'handwriting_bpe': HandwritingBPETokenizer,
        'handwriting_unigram': HandwritingUnigramTokenizer,
        'unigram_em': UnigramTokenizer,
        'handwriting_bigram': lambda: LegacyBigramTokenizer('dp'),
        'linguistic_bigram': lambda: LegacyBigramTokenizer('dp'),
        'linguistic_bigram_greedy': lambda: LegacyBigramTokenizer('greedy'),
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
