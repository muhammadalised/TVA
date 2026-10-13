"""Shared serialization; no dataset or vocabulary-selection logic."""
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_bytes(content: bytes) -> str:
    """Return the lowercase SHA-256 digest for bytes."""
    return hashlib.sha256(content).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON deterministically for a stable artifact checksum."""
    text = json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False,
    )
    return f'{text}\n'.encode('utf-8')


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def save_frozen(path: str | Path, content: bytes) -> None:
    """Create an artifact, or accept an identical existing artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == content, f'Existing frozen file differs: {path}')
    else:
        with path.open('xb') as handle:
            handle.write(content)


# Only known former repository paths are redirected; other missing files still fail.
_ARTIFACT_ROOT = Path(__file__).resolve().parents[2] / 'artifacts/tokenizers'
_LEGACY_ARTIFACT_PATHS = {
    'onhw_words500_rh_iam_read_v1.json':
        'archive/onhw/handwriting/onhw_words500_rh_iam_read_v1.json',
    'ed_iam_handwriting_bigram_greedy_v1.json':
        'ed/handwriting/ed_iam_handwriting_bigram_greedy_v1.json',
    'ed_iam_linguistic_bigram_greedy_v1.json':
        'archive/ed/linguistic/ed_iam_linguistic_bigram_greedy_v1.json',
    'ed_iam_tva_original_bigram_greedy_v1.audit.json':
        'ed/linguistic/ed_iam_tva_original_bigram_greedy_v1.audit.json',
    'ed_iam_handwriting_subwords_v1.audit.json':
        'ed/handwriting/ed_iam_handwriting_subwords_v1.audit.json',
    'ed_iam_handwriting_unigram_v1.json':
        'ed/handwriting/ed_iam_handwriting_unigram_v1.json',
    'ed_iam_tva_original_bigram_greedy_v1.json':
        'ed/linguistic/ed_iam_tva_original_bigram_greedy_v1.json',
    'ed_iam_handwriting_bpe_v1.json':
        'ed/handwriting/ed_iam_handwriting_bpe_v1.json',
    'ed_iam_tva_original_bigram_greedy_v1.provenance.json':
        'ed/linguistic/ed_iam_tva_original_bigram_greedy_v1.provenance.json',
    'source/iam-english-handwriting-bpe-v1.json':
        'ed/handwriting/source/iam-english-handwriting-bpe-v1.json',
    'source/iam-english-handwriting-bigram-v1.json':
        'ed/handwriting/source/iam-english-handwriting-bigram-v1.json',
    'source/iam-english-handwriting-unigram-v1.json':
        'ed/handwriting/source/iam-english-handwriting-unigram-v1.json',
    'source/iam-english-handwriting-subwords-v1.manifest.json':
        'ed/handwriting/source/iam-english-handwriting-subwords-v1.manifest.json',
    'linguistic_bigram/onhw_words500_wd_word_rh/manifest.json':
        'archive/onhw/linguistic_bigram/onhw_words500_wd_word_rh/manifest.json',
    'linguistic_bigram/onhw_words500_wd_word_rh/0.json':
        'archive/onhw/linguistic_bigram/onhw_words500_wd_word_rh/0.json',
    'linguistic_bigram/onhw_words500_wi_word_rh/manifest.json':
        'archive/onhw/linguistic_bigram/onhw_words500_wi_word_rh/manifest.json',
    'linguistic_bigram/onhw_words500_wi_word_rh/0.json':
        'archive/onhw/linguistic_bigram/onhw_words500_wi_word_rh/0.json',
}


def resolve_artifact_path(path: str | Path) -> Path:
    """Locate a moved bundled artifact referenced by a saved experiment config."""
    path = Path(path)
    if path.exists():
        return path
    try:
        relative = path.absolute().relative_to(_ARTIFACT_ROOT).as_posix()
    except ValueError:
        return path
    target = _LEGACY_ARTIFACT_PATHS.get(relative)
    return _ARTIFACT_ROOT / target if target is not None else path
