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
