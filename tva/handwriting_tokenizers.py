"""Handwriting-aware tokenizer training and runtime classes."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import unicodedata
from typing import Any

from loguru import logger

from tva.handwriting.common import canonical_json_bytes, require, resolve_artifact_path, save_frozen, sha256_bytes

from tva.handwriting.onhw import (
    ADAPTER_BIGRAM_COUNT,
    ADAPTER_POLICY_ID,
    ADAPTER_SHA256,
    ADAPTER_SIZE,
    SOURCE_SHA256,
)
from tva.handwriting.ed import (
    ADAPTER_BIGRAM_COUNT as ED_ADAPTER_BIGRAM_COUNT,
    ADAPTER_POLICY_ID as ED_ADAPTER_POLICY_ID,
    ADAPTER_SCHEMA as ED_ADAPTER_SCHEMA,
    ADAPTER_SHA256 as ED_ADAPTER_SHA256,
    ADAPTER_SIZE as ED_ADAPTER_SIZE,
    GREEDY_POLICY,
    SOURCE_SHA256 as ED_SOURCE_SHA256,
)


LINGUISTIC_SCHEMA = 'tva.ed-frequency-bigram-adapter.v1'
ED_LINGUISTIC_SHA256 = '1da363e43fe9d300ece7ad5e3183d84a15fab4bdc6d5b2239b450c222b175abd'
ED_LINGUISTIC_SIZE = 224
IAM_FREQUENCY_EVIDENCE_SHA256 = '7808d5d7b58354be982b042c8f60db4d63bf2ee12aa03cbeca6c3fe628cd1ebb'
MANIFEST_SHA256 = {
    'onhw_words500_wd_word_rh': (
        'c6895783d3da442e80ddf169be2e6397839ff2b0158df19b2c6ecb1bc40a2faa'
    ),
    'onhw_words500_wi_word_rh': (
        '605b0f627d3044340111d7ee0b6ba364790b51130d431640d8eca3e1cc9bb392'
    ),
}


HANDWRITING_SCHEMA = 'dtlr.handwriting-bigram-tokenizer.v2'
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

    supported_schemas = {HANDWRITING_SCHEMA}
    bigram_kind = 'handwriting-bigram'
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

    def train(self, source_path, categories, *, output_path) -> None:
        """Prepare the historical IAM+READ-to-OnHW vocabulary and load it."""
        from tva.handwriting.onhw import ONHW_ALPHABET, build_frozen_onhw_adapter

        require(list(categories) == ['', *ONHW_ALPHABET], 'Expected the declared OnHW alphabet')
        model = build_frozen_onhw_adapter(source_path)
        save_frozen(output_path, canonical_json_bytes(model))
        self.load(output_path)

    def load(self, path_config: str | Path) -> None:
        """Load and authenticate the canonical frozen OnHW adapter."""
        path = resolve_artifact_path(path_config)
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
                    'kind': self.bigram_kind,
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


class GreedyHandwritingBigramTokenizer(HandwritingBigramTokenizer):
    """Greedy left-to-right runtime for the frozen ED adapter."""

    supported_schemas = {ED_ADAPTER_SCHEMA}
    supported_overlap_policy = GREEDY_POLICY
    bigram_kind = 'handwriting-bigram'

    def train(self, evidence_path, categories, *, output_path) -> None:
        """Select handwriting bigrams from IAM connectivity scores in TVA."""
        from tva.handwriting.ed import ED_ALPHABET
        from tva.handwriting.training import fit_ed_tokenizer, save_training_outputs

        require(list(categories) == ['', *ED_ALPHABET], 'Expected the declared ED alphabet')
        adapter, native, report = fit_ed_tokenizer(evidence_path, 'bigram')
        save_training_outputs(output_path, 'bigram', adapter, native, report)
        self.load(output_path)

    def prepare(self, source_path, categories, *, output_path) -> None:
        """Prepare ED IDs/alphabet from already-frozen IAM handwriting evidence.

        Preparation reuses an existing vocabulary; use train() for selection.
        All 145 selected bigrams are retained; no ED label values are read.
        """
        from tva.handwriting.ed import ED_ALPHABET, build_ed_adapter

        require(list(categories) == ['', *ED_ALPHABET], 'Expected the declared ED alphabet')
        model = build_ed_adapter(source_path)
        save_frozen(output_path, canonical_json_bytes(model))
        self.load(output_path)

    def load(self, path_config: str | Path) -> None:
        """Load and authenticate the canonical frozen ED adapter."""
        path = resolve_artifact_path(path_config)
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


class _HandwritingSubwordTokenizer:
    """ED runtime using the frozen DTLR inference rules and ED class IDs."""

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

    def train(self, evidence_path, categories, *, output_path) -> None:
        """Learn candidates and merges/probabilities from IAM evidence in TVA."""
        from tva.handwriting.ed import ED_ALPHABET
        from tva.handwriting.training import fit_ed_tokenizer, save_training_outputs

        require(list(categories) == ['', *ED_ALPHABET], 'Expected the declared ED alphabet')
        adapter, native, report = fit_ed_tokenizer(evidence_path, self.kind)
        save_training_outputs(output_path, self.kind, adapter, native, report)
        self.load(output_path)

    def prepare(self, source_path, categories, *, output_path) -> None:
        """Reuse frozen pieces/parameters; use train() to learn them from evidence."""
        from tva.handwriting.ed import ED_ALPHABET
        from tva.handwriting.subwords import SOURCE_FILES, build_ed_subword_adapter

        require(list(categories) == ['', *ED_ALPHABET], 'Expected the declared ED alphabet')
        model = build_ed_subword_adapter(source_path, self.kind)
        output_path = Path(output_path)
        # Package the source beside the adapter so no DTLR checkout is needed.
        save_frozen(output_path.parent / 'source' / SOURCE_FILES[self.kind],
                    Path(source_path).read_bytes())
        save_frozen(output_path, canonical_json_bytes(model))
        self.load(output_path)

    def load(self, path_config) -> None:
        from tva.handwriting.subwords import load_ed_subword_adapter

        model, source = load_ed_subword_adapter(path_config, self.kind)
        self.model, self.source_model = model, source
        self.vocab = dict(model['vocab'])
        self.idx_token = {int(i): t for i, t in model['idx_token'].items()}
        if self.kind == 'unigram':
            # Keep learned scores exactly; new nonletters have one forced arc.
            self._scores = {t: v for t, v in source['log_probs'].items() if t in self.vocab}
            self._scores.update({t: 0.0 for t in model['policy']['added_singletons']})
            self._max_length = max(map(len, self._scores))
        logger.info(f'{type(self).__name__} is loaded from {path_config}.')

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


class HandwritingBPETokenizer(_HandwritingSubwordTokenizer):
    """Apply all frozen BPE merges in their learned order."""

    kind = 'bpe'


class HandwritingUnigramTokenizer(_HandwritingSubwordTokenizer):
    """Viterbi segmentation with frozen IAM scores and forced ED symbols."""

    kind = 'unigram'
