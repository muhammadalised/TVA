"""Matched IAM text-only BPE/Unigram controls for ED recognition.

Use the same trainers and segmentation as the frozen handwriting variants.
Only candidate evidence and merge/initialization counts switch to text-only.
"""

import json
from pathlib import Path
import unicodedata

from loguru import logger

from tva.handwriting.common import canonical_json_bytes, require, resolve_artifact_path, save_frozen, sha256_bytes
from tva.handwriting.ed import ED_ALPHABET
from tva.handwriting.training import collect_text_candidates, train_bpe, train_unigram
from tva.iam_tokenizers import LABELS_SHA256, SELECTION_SHA256, read_iam_training
from tva.subword_tokenizers import SubwordTokenizer

SOURCE_FILES = {kind: f'iam-english-text-{kind}-v1.json' for kind in ('bpe', 'unigram')}
# Pin learned models after fitting; ED projections are reconstructed on load.
SOURCE_SHA256 = {'bpe': 'b378e1e93b86d3483f15d5bbe5865c31c9581112f75ded56f45589f130b6c1d3', 'unigram': 'cce6b18818a71b43d74c98e5e2ae334698b8c491c704539aed04e8a02edba0b3'}
ADAPTER_SCHEMA = 'tva.ed-text-subword-adapter.v1'


def project_ed_model(source, kind):
    """Apply exactly the handwriting release's alphabet/ID projection policy."""
    require(kind in SOURCE_FILES and source['algorithm'] == f'text-only-{kind}', 'Wrong text-only algorithm')
    pieces = [t for t, i in sorted(source['vocab'].items(), key=lambda item: item[1]) if len(t) > 1]
    require(len(pieces) == 145, 'Expected 145 selected pieces')
    require(all(2 <= len(t) <= 6 and all(c in ED_ALPHABET and c.isalpha() for c in t) for t in pieces),
            'Selected pieces must be 2–6 ED letters')
    singles = {t for t in source['vocab'] if len(t) == 1}
    added, removed = sorted(set(ED_ALPHABET) - singles), sorted(singles - set(ED_ALPHABET))
    require(added == ['%', '='] and removed == ['#', '&', '*'], 'Unexpected IAM-to-ED alphabet projection')
    tokens = ['', *ED_ALPHABET, *pieces]
    require(len(tokens) == len(set(tokens)) == 224, 'Invalid ED vocabulary')
    return {
        'schema_version': ADAPTER_SCHEMA,
        'model_version': f'ed-iam-linguistic-{kind}-v1',
        'status': 'frozen-matched-text-control',
        'frozen_date': '2026-10-18',
        'dataset': 'ED', 'algorithm': f'text-only-{kind}',
        'unicode_normalization': 'NFC', 'blank_id': 0, 'blank_token': '',
        'size': 224, 'subword_count': 145,
        'vocab': {t: i for i, t in enumerate(tokens)},
        'idx_token': {str(i): t for i, t in enumerate(tokens)},
        'source_model': {
            'filename': SOURCE_FILES[kind],
            'sha256': sha256_bytes(canonical_json_bytes(source)),
            'model_version': source['model_version'],
        },
        'policy': {
            'vocabulary_selection': 'IAM-train-text-only; matched trainer with handwriting disabled',
            'inference': source['policy']['inference'],
            'ordered_merges_unchanged': kind == 'bpe',
            'learned_log_probs_unchanged': kind == 'unigram',
            'unigram_probability_renormalization': False,
            'added_singletons': added, 'removed_singletons': removed,
            'added_singleton_behavior': 'forced-one-character-token; boundary-for-letter-pieces',
            'added_singleton_viterbi_score': 0.0 if kind == 'unigram' else None,
            'annotation_label_values_read_by_builder': False,
            'unknown_characters': 'raise-error',
        },
    }


def fit_iam_subwords(labels_path, selection_path, kind):
    """Fit on all authenticated IAM train rows, retaining duplicate transcriptions.

    This path never opens connectivity evidence or ED annotations.
    """
    require(kind in SOURCE_FILES, 'Unsupported text-only subword algorithm')
    rows = read_iam_training(labels_path, selection_path)
    corpus = [{'line_id': row['id'], 'text': row['label']} for row in rows]
    require(len(corpus) == 5694, 'Expected all 5,694 IAM train lines')
    require(all(unicodedata.normalize('NFC', row['text']) == row['text'] for row in corpus),
            'Expected NFC IAM transcripts to match handwriting training')
    candidates = collect_text_candidates(corpus, max_length=6, minimum_count=20)
    if kind == 'bpe':
        native = train_bpe(corpus, candidates, budget=145, handwriting_aware=False)
    else:
        native = train_unigram(corpus, candidates, budget=145, iterations=8,
                              seed_multiplier=4, shrink_factor=0.75, handwriting_aware=False)
    require(len(native['vocabulary']) == 145, 'Training did not reach the matched piece budget')
    native.update({
        'status': 'frozen-experiment-policy', 'model_version': f'iam-english-text-{kind}-v1',
        'trainer_repository': 'TVA',
        'freeze': {
            'labels_sha256': LABELS_SHA256, 'selection_sha256': SELECTION_SHA256,
            'ordered_corpus_sha256': sha256_bytes(canonical_json_bytes(corpus)),
            'candidate_pool_sha256': sha256_bytes(canonical_json_bytes(candidates)),
            'candidate_count': len(candidates), 'line_count': len(corpus),
            'duplicate_transcriptions_retained': True,
            'max_piece_length': 6, 'minimum_text_occurrences': 20,
            'handwriting_evidence_read': False, 'ed_labels_read': False,
        },
    })
    adapter = project_ed_model(native, kind)
    report = {
        'schema_version': 'tva.matched-text-tokenizer-training.v1',
        'trainer_repository': 'TVA', 'algorithm': kind, 'dataset': 'IAM', 'split': 'train',
        **native['freeze'],
        'handwriting_aware': False, 'selected_piece_count': 145, 'ed_ctc_classes': 224,
        'native_sha256': sha256_bytes(canonical_json_bytes(native)),
        'ed_adapter_sha256': sha256_bytes(canonical_json_bytes(adapter)),
        'matched_handwriting_model_version': f'ed-iam-handwriting-{kind}-v1',
        'implementation_sha256': {
            name: sha256_bytes((Path(__file__).parent / name).read_bytes())
            for name in ('handwriting/training.py', 'linguistic_tokenizers.py', 'subword_tokenizers.py')
        },
    }
    return adapter, native, report


class _LinguisticSubwordTokenizer(SubwordTokenizer):
    """IAM text control using the identical handwriting segmentation methods."""

    def train(self, labels_path, categories, *, selection_path, output_path):
        require(list(categories) == ['', *ED_ALPHABET], 'Expected the declared ED alphabet')
        adapter, native, report = fit_iam_subwords(labels_path, selection_path, self.kind)
        require(sha256_bytes(canonical_json_bytes(native)) == SOURCE_SHA256[self.kind],
                'Fitted text-only model differs from frozen v1')
        output = Path(output_path)
        files = {
            output: canonical_json_bytes(adapter),
            output.parent / 'source' / SOURCE_FILES[self.kind]: canonical_json_bytes(native),
            output.with_suffix('.training.json'): canonical_json_bytes(report),
        }
        for path, content in files.items():
            require(not path.exists() or path.read_bytes() == content, f'Existing frozen file differs: {path}')
        for path, content in files.items():
            save_frozen(path, content)
        self.load(output)

    def load(self, path_config):
        path = resolve_artifact_path(path_config)
        source_raw = (path.parent / 'source' / SOURCE_FILES[self.kind]).read_bytes()
        require(sha256_bytes(source_raw) == SOURCE_SHA256[self.kind], 'Frozen IAM text source SHA-256 mismatch')
        source = json.loads(source_raw)
        model = project_ed_model(source, self.kind)
        require(path.read_bytes() == canonical_json_bytes(model), 'Frozen ED text adapter differs from v1')
        self.model, self.source_model = model, source
        self.vocab = dict(model['vocab'])
        self.idx_token = {int(i): t for i, t in model['idx_token'].items()}
        if self.kind == 'unigram':
            self._scores = {t: v for t, v in source['log_probs'].items() if t in self.vocab}
            self._scores.update({t: 0.0 for t in model['policy']['added_singletons']})
            self._max_length = max(map(len, self._scores))
        logger.info(f'{type(self).__name__} is loaded from {path}.')


class LinguisticBPETokenizer(_LinguisticSubwordTokenizer):
    """Text-frequency BPE trained and segmented like the handwriting control."""

    kind = 'bpe'


class LinguisticUnigramTokenizer(_LinguisticSubwordTokenizer):
    """Text-only Unigram with the matched EM/pruning and Viterbi implementation."""

    kind = 'unigram'
