"""Read the portable DTLR IAM connectivity export; no DTLR checkout is needed."""
import gzip
import json
from pathlib import Path
import unicodedata

from .common import require, sha256_bytes

DEFAULT_EVIDENCE = Path(__file__).resolve().parents[2] / 'artifacts/evidence/iam/iam_connectivity_v1.json.gz'
EVIDENCE_SHA256 = '815304d2591bc187a2eea7dd76593499905c11645c8b9a146e7f1699c0e636fa'
UNCOMPRESSED_SHA256 = '1b44fddd0fa76dbb4ed14517f4361c522a4415139d16307b3468e14d122b8d0c'
SOURCE_SHA256 = {
    'detections': '5e68b4c2ef1db15b5d1776ba1d4b7696ba9289f81dd4dd018598dd7d6aea2b32',
    'evidence': '9c75d6d240127c2e6a6c4dffd83c491beab6a31dc4be204461861c1913740bd3',
    'manifest': '82cb95a14e71ab5b70c868ca1676b3f628552fabac395fbd30039fc7798bf190',
    'scores': '45c86a9f703ebc3684563ec2a840e7c34241aa76e84d692be46cf11bfec27055',
    'selection': '7893dfba4febe6df99cf0bdb0c74fabe7b736d2a6af05033d8638b90455bc1c2',
}


def validate_evidence(bundle):
    """Validate identities, text offsets, abstentions and aggregate reconstruction."""
    require(bundle.get('schema_version') == 'dtlr.iam-tokenizer-evidence.v1', 'Unsupported evidence schema')
    require(bundle.get('dataset') == 'IAM' and bundle.get('split') == 'train', 'Evidence must be IAM train only')
    require(bundle.get('connectivity_method') == 'dominant-core-v3', 'Wrong connectivity method')
    require(bundle.get('source_sha256') == SOURCE_SHA256, 'Pinned source hashes differ')
    selection, manifest = bundle['selection'], bundle['manifest']
    require(selection['schema_version'] == 'dtlr.iam-selection.v1'
            and selection['dataset'] == 'IAM' and selection['split'] == 'train', 'Invalid IAM training selection')
    require(manifest['schema_version'] == 'dtlr.export-manifest.v5'
            and manifest['datasets'] == ['IAM'] and manifest['splits'] == ['train']
            and manifest['primary_connectivity_method'] == 'dominant-core-v3', 'Invalid evidence manifest')
    require(len(bundle['lines']) == len(selection['lines']) == selection['requested_count']
            == manifest['line_count'] == 5694, 'Expected all 5,694 IAM train lines')
    corpus, ids, counts = [], set(), {}
    for line, item in zip(bundle['lines'], selection['lines']):
        text = line['text']
        require(line['line_id'] == item['id'] and line['line_id'] not in ids, 'Line identity/order mismatch')
        require(sha256_bytes(text.encode('utf-8')) == item['transcription_sha256'], 'Training text hash mismatch')
        require(unicodedata.normalize('NFC', text) == text, 'NFC offsets must be preserved')
        ids.add(line['line_id'])
        edges = {}
        for start, left, right in line['edges']:
            require(type(start) is int and 0 <= start < len(text) - 1 and start not in edges, 'Invalid/duplicate edge offset')
            require((left is None and right is None)
                    or (type(left) is int and left > 0 and type(right) is int and right > 0), 'Invalid component evidence')
            edges[start] = None if left is None else (left, right)
            pair = text[start:start + 2]
            totals = counts.setdefault(pair, [0, 0, 0])
            totals[0] += 1
            if left is not None:
                totals[1] += 1
                totals[2] += int(left == right)
        expected = {i for i in range(len(text) - 1) if not any(c.isspace() for c in text[i:i + 2])}
        require(set(edges) == expected, 'Incomplete non-whitespace occurrence evidence')
        corpus.append({'line_id': line['line_id'], 'text': text, 'edges': edges})
    require(sum(len(line['edges']) for line in corpus) == manifest['evidence_count'] == 160536, 'Evidence count differs')
    scores = bundle['scores']
    require(len(scores) == manifest['score_count'] == 1259
            and len({row['pair'] for row in scores}) == len(scores)
            and set(counts) == {row['pair'] for row in scores}, 'Aggregate inventory differs')
    for row in scores:
        require(row['schema_version'] == 'dtlr.bigram-scores.v5'
                and row['dataset'] == 'IAM' and row['split'] == 'train'
                and row['connectivity_method'] == 'dominant-core-v3', 'Invalid score source')
        total, exact, connected = counts[row['pair']]
        require(row['n_total'] == total and row['n_exact_alignment'] == exact
                and row['exact_alignment_connected_rate'] == (connected / exact if exact else None),
                'Aggregate connectivity scores differ from occurrence evidence')
    return corpus


def load_evidence(path=DEFAULT_EVIDENCE):
    """Authenticate the complete v1 evidence before vocabulary fitting."""
    packed = Path(path).read_bytes()
    require(sha256_bytes(packed) == EVIDENCE_SHA256, 'IAM evidence SHA-256 mismatch')
    raw = gzip.decompress(packed)
    require(sha256_bytes(raw) == UNCOMPRESSED_SHA256, 'Decompressed IAM evidence SHA-256 mismatch')
    bundle = json.loads(raw)
    corpus = validate_evidence(bundle)
    return corpus, bundle
