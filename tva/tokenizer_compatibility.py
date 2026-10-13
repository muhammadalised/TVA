"""Load-only linguistic tokenizers retained for completed experiments.

New linguistic training uses tva.tokenizers through tva.iam_tokenizers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

from tva.handwriting.common import resolve_artifact_path, sha256_bytes
from tva.handwriting.onhw import ADAPTER_BIGRAM_COUNT, ADAPTER_SIZE
from tva.handwriting_tokenizers import (
    HandwritingBigramTokenizer, GreedyHandwritingBigramTokenizer,
)

# Frozen historical artifact identities; no vocabulary construction here.
ED_LINGUISTIC_BIGRAM_COUNT = 145
ED_LINGUISTIC_POLICY_ID = 'ed-iam-frequency-bigram-greedy-v1'
ED_LINGUISTIC_SCHEMA = 'tva.ed-frequency-bigram-adapter.v1'
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


LINGUISTIC_SCHEMA = 'tva.frequency-bigram-tokenizer.v1'


class LinguisticBigramTokenizer(HandwritingBigramTokenizer):
    """Compatibility loader for completed OnHW frequency/utility experiments.

    New linguistic vocabularies use the original TVA BigramTokenizer.
    """

    supported_schemas = {LINGUISTIC_SCHEMA}
    bigram_kind = 'linguistic-bigram'

    def train(self, *args, **kwargs) -> None:
        """Historical artifacts are load-only."""
        raise NotImplementedError('Use IAMBigramTokenizer for new linguistic training')

    def load(self, path_config: str | Path) -> None:
        """Load and validate one fold's linguistic tokenizer artifact."""
        path = resolve_artifact_path(path_config)
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


class GreedyLinguisticBigramTokenizer(GreedyHandwritingBigramTokenizer):
    """Compatibility loader for the completed custom IAM-frequency ED runs.

    Kept for historical configs and analysis, not new vocabulary training.
    """

    supported_schemas = {ED_LINGUISTIC_SCHEMA}
    bigram_kind = 'linguistic-bigram'

    def train(self, *args, **kwargs) -> None:
        """Historical artifacts are load-only."""
        raise NotImplementedError('Use IAMBigramTokenizer for new linguistic training')

    def load(self, path_config: str | Path) -> None:
        path = resolve_artifact_path(path_config)
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
