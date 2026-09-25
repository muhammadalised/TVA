"""Runtime tokenizer for frozen handwriting-aware bigram models."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import unicodedata
from typing import Any

from loguru import logger

from tva.handwriting_bigram_adapter import (
    ADAPTER_BIGRAM_COUNT,
    ADAPTER_POLICY_ID,
    ADAPTER_SHA256,
    ADAPTER_SIZE,
    SOURCE_SHA256,
    sha256_bytes,
)
from tva.ed_handwriting_bigram import (
    ADAPTER_BIGRAM_COUNT as ED_ADAPTER_BIGRAM_COUNT,
    ADAPTER_POLICY_ID as ED_ADAPTER_POLICY_ID,
    ADAPTER_SCHEMA as ED_ADAPTER_SCHEMA,
    ADAPTER_SHA256 as ED_ADAPTER_SHA256,
    ADAPTER_SIZE as ED_ADAPTER_SIZE,
    GREEDY_POLICY,
    SOURCE_SHA256 as ED_SOURCE_SHA256,
)
from tva.ed_linguistic_bigram import (
    ADAPTER_BIGRAM_COUNT as ED_LINGUISTIC_BIGRAM_COUNT,
    ADAPTER_POLICY_ID as ED_LINGUISTIC_POLICY_ID,
    ADAPTER_SCHEMA as ED_LINGUISTIC_SCHEMA,
    ADAPTER_SHA256 as ED_LINGUISTIC_SHA256,
    ADAPTER_SIZE as ED_LINGUISTIC_SIZE,
    EVIDENCE_SHA256 as IAM_FREQUENCY_EVIDENCE_SHA256,
)
from tva.linguistic_bigram import MANIFEST_SHA256


HANDWRITING_SCHEMA = 'dtlr.handwriting-bigram-tokenizer.v2'
LINGUISTIC_SCHEMA = 'tva.frequency-bigram-tokenizer.v1'
SUPPORTED_SCHEMAS = {HANDWRITING_SCHEMA, LINGUISTIC_SCHEMA}
SUPPORTED_OVERLAP_POLICY = 'maximum-total-utility-non-overlapping-v1'


@dataclass(frozen=True)
class _Segmentation:
    utility: float
    bigram_count: int
    segments: tuple[dict[str, Any], ...]


def _normalize_text(text: str, policy: str) -> str:
    if policy == 'none':
        return text
    if policy == 'NFC':
        return unicodedata.normalize('NFC', text)
    raise ValueError(f'unsupported Unicode normalization policy: {policy}')


class HandwritingBigramTokenizer:
    """Tokenize text with the DTLR maximum-total-utility dynamic program."""

    supported_schemas = SUPPORTED_SCHEMAS
    supported_overlap_policy = SUPPORTED_OVERLAP_POLICY

    def __init__(self) -> None:
        self.model: dict[str, Any] = {}
        self.vocab: dict[str, int] = {}
        self.idx_token: dict[int, str] = {}
        self._bigram_rows: dict[str, dict[str, Any]] = {}

    @property
    def size(self) -> int:
        """Return the vocabulary size, including CTC blank ID 0."""
        if not self.vocab:
            raise ValueError('Tokenizer not trained or loaded.')
        return len(self.vocab)

    def load(self, path_config: str | Path) -> None:
        """Load and authenticate the canonical frozen OnHW adapter."""
        path = Path(path_config)
        content = path.read_bytes()
        digest = sha256_bytes(content)
        if digest != ADAPTER_SHA256:
            raise ValueError(
                f'frozen adapter SHA-256 mismatch: expected {ADAPTER_SHA256}, '
                f'got {digest}'
            )
        try:
            model = json.loads(content)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError('tokenizer model is not valid UTF-8 JSON') from error
        self.load_model(model, require_frozen_adapter=True)
        logger.info(f'HandwritingBigramTokenizer is loaded from {path}.')

    def load_model(
        self,
        model: dict[str, Any],
        *,
        require_frozen_adapter: bool = False,
    ) -> None:
        """Load a structurally valid DTLR-v2 model from memory.

        The in-memory path supports small algorithm tests. TVA's training path
        calls :meth:`load`, which additionally authenticates the frozen bytes.
        """
        if not isinstance(model, dict):
            raise ValueError('tokenizer model root must be a JSON object')
        if model.get('schema_version') not in self.supported_schemas:
            raise ValueError('unsupported utility-bigram tokenizer schema')
        if model.get('blank_token') != '' or model.get('blank_id') != 0:
            raise ValueError('CTC blank must be the empty token at ID 0')
        if model.get('text_normalization') not in {'none', 'NFC'}:
            raise ValueError('tokenizer normalization must be none or NFC')
        policy = model.get('policy')
        if not isinstance(policy, dict):
            raise ValueError('tokenizer model lacks its policy object')
        if policy.get('overlap_resolution') != self.supported_overlap_policy:
            raise ValueError('tokenizer model has an unsupported overlap policy')
        if policy.get('single_character_utility') != 0.0:
            raise ValueError('single-character utility must be zero')

        vocab = model.get('vocab')
        raw_idx_token = model.get('idx_token')
        if not isinstance(vocab, dict) or not isinstance(raw_idx_token, dict):
            raise ValueError('tokenizer model lacks vocab or idx_token mappings')
        if any(
            not isinstance(token, str) or not isinstance(index, int)
            for token, index in vocab.items()
        ):
            raise ValueError('vocab must map token strings to integer IDs')
        expected_ids = set(range(len(vocab)))
        if set(vocab.values()) != expected_ids:
            raise ValueError('vocabulary IDs must be unique, contiguous, and zero-based')
        if set(raw_idx_token) != {str(index) for index in expected_ids}:
            raise ValueError('idx_token keys must cover every vocabulary ID')
        idx_token = {int(index): token for index, token in raw_idx_token.items()}
        if any(idx_token[index] != token for token, index in vocab.items()):
            raise ValueError('vocab and idx_token must be inverse mappings')
        if model.get('size') != len(vocab):
            raise ValueError('declared tokenizer size does not match its vocabulary')
        if vocab.get('') != 0 or idx_token.get(0) != '':
            raise ValueError('CTC blank mapping must be the empty token at ID 0')

        vocabulary = model.get('vocabulary')
        if not isinstance(vocabulary, list):
            raise ValueError('tokenizer model lacks its bigram vocabulary rows')
        bigram_rows: dict[str, dict[str, Any]] = {}
        for row in vocabulary:
            if not isinstance(row, dict):
                raise ValueError('bigram vocabulary rows must be objects')
            token = row.get('token')
            utility = row.get('utility')
            if not isinstance(token, str) or len(token) != 2:
                raise ValueError(f'invalid bigram token: {token!r}')
            if token in bigram_rows:
                raise ValueError(f'duplicate bigram token: {token!r}')
            if (
                isinstance(utility, bool)
                or not isinstance(utility, (int, float))
                or not math.isfinite(utility)
            ):
                raise ValueError(f'bigram utility must be finite: {token!r}')
            if token not in vocab:
                raise ValueError(f'bigram row is absent from vocab: {token!r}')
            bigram_rows[token] = row
        if model.get('eligible_bigram_count') != len(bigram_rows):
            raise ValueError('declared bigram count does not match vocabulary rows')
        mapped_bigrams = {token for token in vocab if len(token) == 2}
        if set(bigram_rows) != mapped_bigrams:
            raise ValueError('bigram rows do not match the mapped bigram tokens')

        if require_frozen_adapter:
            self._validate_frozen_adapter(model)

        self.model = model
        self.vocab = dict(vocab)
        self.idx_token = idx_token
        self._bigram_rows = bigram_rows

    @staticmethod
    def _validate_frozen_adapter(model: dict[str, Any]) -> None:
        if model.get('model_version') != ADAPTER_POLICY_ID:
            raise ValueError('tokenizer is not the frozen OnHW adapter policy')
        if model.get('size') != ADAPTER_SIZE:
            raise ValueError('frozen adapter has an unexpected output size')
        if model.get('eligible_bigram_count') != ADAPTER_BIGRAM_COUNT:
            raise ValueError('frozen adapter has an unexpected bigram count')
        if model.get('text_normalization') != 'NFC':
            raise ValueError('frozen adapter must use NFC normalization')
        adapter = model.get('adapter')
        if not isinstance(adapter, dict):
            raise ValueError('frozen adapter lacks provenance metadata')
        if adapter.get('policy_id') != ADAPTER_POLICY_ID:
            raise ValueError('frozen adapter policy metadata does not match')
        source = adapter.get('source_model')
        if not isinstance(source, dict) or source.get('sha256') != SOURCE_SHA256:
            raise ValueError('frozen adapter source provenance does not match')
        if adapter.get('annotation_files_read') is not False:
            raise ValueError('frozen adapter does not declare leakage-safe construction')

    def segment(self, text: str) -> dict[str, Any]:
        """Return the exact DTLR segmentation and its utility metadata."""
        if not self.model:
            raise ValueError('Tokenizer not trained or loaded.')
        if not isinstance(text, str):
            raise TypeError('text must be a string')
        input_text = text
        text = _normalize_text(text, self.model['text_normalization'])
        best: list[_Segmentation | None] = [None] * (len(text) + 1)
        best[len(text)] = _Segmentation(0.0, 0, ())

        for index in range(len(text) - 1, -1, -1):
            suffix = best[index + 1]
            assert suffix is not None
            single = {
                'token': text[index],
                'start': index,
                'end': index + 1,
                'kind': 'single',
                'utility': 0.0,
            }
            winner = _Segmentation(
                suffix.utility,
                suffix.bigram_count,
                (single, *suffix.segments),
            )

            pair = text[index:index + 2]
            if len(pair) == 2 and pair in self._bigram_rows:
                suffix = best[index + 2]
                assert suffix is not None
                row = self._bigram_rows[pair]
                segment = {
                    'token': pair,
                    'start': index,
                    'end': index + 2,
                    'kind': (
                        'handwriting-bigram'
                        if self.model['schema_version'] != LINGUISTIC_SCHEMA
                        else 'linguistic-bigram'
                    ),
                    'utility': row['utility'],
                }
                for metadata_key in (
                    'n_exact_alignment',
                    'exact_alignment_connected_rate',
                    'training_word_type_adjacency_count',
                    'training_word_type_support',
                    'rank',
                ):
                    if metadata_key in row:
                        segment[metadata_key] = row[metadata_key]
                candidate = _Segmentation(
                    suffix.utility + row['utility'],
                    suffix.bigram_count + 1,
                    (segment, *suffix.segments),
                )
                if (
                    candidate.utility,
                    candidate.bigram_count,
                ) >= (
                    winner.utility,
                    winner.bigram_count,
                ):
                    winner = candidate
            best[index] = winner

        result = best[0]
        assert result is not None
        output = {
            'text': text,
            'tokens': [segment['token'] for segment in result.segments],
            'total_utility': result.utility,
            'bigram_token_count': result.bigram_count,
            'segments': list(result.segments),
        }
        if input_text != text:
            output['input_text'] = input_text
        return output

    def encode(self, text: str) -> list[int]:
        """Encode text without ever emitting the CTC blank ID."""
        result = self.segment(text)
        missing = sorted({
            token for token in result['tokens'] if token not in self.vocab
        })
        if missing:
            raise ValueError(
                f'characters are absent from tokenizer vocabulary: {missing!r}'
            )
        return [self.vocab[token] for token in result['tokens']]

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs; blank ID 0 contributes the empty string."""
        if not self.idx_token:
            raise ValueError('Tokenizer not trained or loaded.')
        try:
            return ''.join(self.idx_token[index] for index in ids)
        except KeyError as error:
            raise ValueError(f'unknown token ID: {error.args[0]}') from error


class LinguisticBigramTokenizer(HandwritingBigramTokenizer):
    """Runtime for fold-specific frequency-derived Bigram baselines."""

    def load(self, path_config: str | Path) -> None:
        """Load and validate one fold's linguistic tokenizer artifact."""
        path = Path(path_config)
        content = path.read_bytes()
        try:
            model = json.loads(content)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError('tokenizer model is not valid UTF-8 JSON') from error
        dataset = model.get('dataset')
        expected_manifest_sha256 = MANIFEST_SHA256.get(dataset)
        if expected_manifest_sha256 is None:
            raise ValueError('linguistic Bigram dataset is not a frozen condition')
        manifest_path = path.parent / 'manifest.json'
        manifest_content = manifest_path.read_bytes()
        manifest_sha256 = sha256_bytes(manifest_content)
        if manifest_sha256 != expected_manifest_sha256:
            raise ValueError(
                'linguistic Bigram manifest SHA-256 mismatch: expected '
                f'{expected_manifest_sha256}, got {manifest_sha256}'
            )
        try:
            manifest = json.loads(manifest_content)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError('linguistic Bigram manifest is not valid JSON') from error
        artifact = manifest.get('artifacts', {}).get(str(model.get('fold')))
        if not isinstance(artifact, dict) or artifact.get('filename') != path.name:
            raise ValueError('linguistic Bigram artifact is absent from its manifest')
        digest = sha256_bytes(content)
        if artifact.get('sha256') != digest:
            raise ValueError('linguistic Bigram artifact SHA-256 mismatch')
        self.load_model(model)
        if model.get('schema_version') != LINGUISTIC_SCHEMA:
            raise ValueError('tokenizer is not a linguistic Bigram artifact')
        if model.get('model_version') != 'onhw-frequency-bigram-v1':
            raise ValueError('unsupported linguistic Bigram model version')
        if model.get('size') != ADAPTER_SIZE:
            raise ValueError('linguistic Bigram output size is not matched at 419')
        if model.get('eligible_bigram_count') != ADAPTER_BIGRAM_COUNT:
            raise ValueError('linguistic Bigram count is not matched at 359')
        if model.get('training_split') != 'train':
            raise ValueError('linguistic Bigram was not derived from the training split')
        policy = model['policy']
        if policy.get('selection') != 'training-word-type-adjacency-frequency-v1':
            raise ValueError('linguistic Bigram has an unsupported selection policy')
        if policy.get('validation_annotations_read') is not False:
            raise ValueError('linguistic Bigram does not declare validation exclusion')
        logger.info(f'LinguisticBigramTokenizer is loaded from {path}.')


class GreedyHandwritingBigramTokenizer(HandwritingBigramTokenizer):
    """Greedy left-to-right runtime for the frozen ED adapter."""

    supported_schemas = {ED_ADAPTER_SCHEMA}
    supported_overlap_policy = GREEDY_POLICY
    bigram_kind = 'handwriting-bigram'

    def load(self, path_config: str | Path) -> None:
        """Load and authenticate the canonical frozen ED adapter."""
        path = Path(path_config)
        content = path.read_bytes()
        digest = sha256_bytes(content)
        if digest != ED_ADAPTER_SHA256:
            raise ValueError(
                'frozen ED adapter SHA-256 mismatch: expected '
                f'{ED_ADAPTER_SHA256}, got {digest}'
            )
        try:
            model = json.loads(content)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError('tokenizer model is not valid UTF-8 JSON') from error
        self.load_model(model)
        self._validate_ed_adapter(model)
        logger.info(f'GreedyHandwritingBigramTokenizer is loaded from {path}.')

    @staticmethod
    def _validate_ed_adapter(model: dict[str, Any]) -> None:
        if model.get('model_version') != ED_ADAPTER_POLICY_ID:
            raise ValueError('tokenizer is not the frozen ED adapter policy')
        if model.get('size') != ED_ADAPTER_SIZE:
            raise ValueError('frozen ED adapter has an unexpected output size')
        if model.get('eligible_bigram_count') != ED_ADAPTER_BIGRAM_COUNT:
            raise ValueError('frozen ED adapter has an unexpected bigram count')
        if model.get('text_normalization') != 'NFC':
            raise ValueError('frozen ED adapter must use NFC normalization')
        adapter = model.get('adapter')
        if not isinstance(adapter, dict):
            raise ValueError('frozen ED adapter lacks provenance metadata')
        if adapter.get('policy_id') != ED_ADAPTER_POLICY_ID:
            raise ValueError('frozen ED adapter policy metadata does not match')
        source = adapter.get('source_model')
        if not isinstance(source, dict) or source.get('sha256') != ED_SOURCE_SHA256:
            raise ValueError('frozen ED adapter source provenance does not match')
        if adapter.get('annotation_label_values_read_by_builder') is not False:
            raise ValueError(
                'frozen ED adapter does not declare leakage-safe construction'
            )

    def segment(self, text: str) -> dict[str, Any]:
        """Segment by taking an available bigram at each leftmost position."""
        if not self.model:
            raise ValueError('Tokenizer not trained or loaded.')
        if not isinstance(text, str):
            raise TypeError('text must be a string')
        input_text = text
        text = _normalize_text(text, self.model['text_normalization'])
        segments: list[dict[str, Any]] = []
        index = 0
        total_utility = 0.0
        while index < len(text):
            pair = text[index:index + 2]
            row = self._bigram_rows.get(pair) if len(pair) == 2 else None
            if row is not None:
                segment = {
                    'token': pair,
                    'start': index,
                    'end': index + 2,
                    'kind': self.bigram_kind,
                    'utility': row['utility'],
                }
                for metadata_key in (
                    'n_exact_alignment',
                    'exact_alignment_connected_rate',
                    'iam_training_occurrence_count',
                    'rank',
                ):
                    if metadata_key in row:
                        segment[metadata_key] = row[metadata_key]
                segments.append(segment)
                total_utility += row['utility']
                index += 2
                continue
            segments.append({
                'token': text[index],
                'start': index,
                'end': index + 1,
                'kind': 'single',
                'utility': 0.0,
            })
            index += 1

        output = {
            'text': text,
            'tokens': [segment['token'] for segment in segments],
            'total_utility': total_utility,
            'bigram_token_count': sum(
                segment['kind'] == self.bigram_kind for segment in segments
            ),
            'segments': segments,
        }
        if input_text != text:
            output['input_text'] = input_text
        return output


class GreedyLinguisticBigramTokenizer(GreedyHandwritingBigramTokenizer):
    """Greedy runtime for the frozen IAM-frequency ED comparator."""

    supported_schemas = {ED_LINGUISTIC_SCHEMA}
    bigram_kind = 'linguistic-bigram'

    def load(self, path_config: str | Path) -> None:
        path = Path(path_config)
        content = path.read_bytes()
        digest = sha256_bytes(content)
        if digest != ED_LINGUISTIC_SHA256:
            raise ValueError(
                'frozen ED linguistic adapter SHA-256 mismatch: expected '
                f'{ED_LINGUISTIC_SHA256}, got {digest}'
            )
        try:
            model = json.loads(content)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError('tokenizer model is not valid UTF-8 JSON') from error
        self.load_model(model)
        self._validate_ed_linguistic_adapter(model)
        logger.info(f'GreedyLinguisticBigramTokenizer is loaded from {path}.')

    @staticmethod
    def _validate_ed_linguistic_adapter(model: dict[str, Any]) -> None:
        if model.get('model_version') != ED_LINGUISTIC_POLICY_ID:
            raise ValueError('tokenizer is not the frozen ED linguistic policy')
        if model.get('size') != ED_LINGUISTIC_SIZE:
            raise ValueError('ED linguistic tokenizer has an unexpected size')
        if model.get('eligible_bigram_count') != ED_LINGUISTIC_BIGRAM_COUNT:
            raise ValueError('ED linguistic tokenizer has an unexpected pair count')
        adapter = model.get('adapter')
        if not isinstance(adapter, dict):
            raise ValueError('ED linguistic tokenizer lacks provenance metadata')
        evidence = adapter.get('frequency_evidence')
        if (
            not isinstance(evidence, dict)
            or evidence.get('sha256') != IAM_FREQUENCY_EVIDENCE_SHA256
            or evidence.get('dataset') != 'IAM'
            or evidence.get('split') != 'train'
        ):
            raise ValueError('ED linguistic evidence provenance does not match')
        if adapter.get('iam_training_transcripts_read') is not True:
            raise ValueError('ED linguistic adapter omits its IAM text input')
        if adapter.get('annotation_label_values_read_by_builder') is not False:
            raise ValueError(
                'ED linguistic adapter does not declare leakage-safe construction'
            )
