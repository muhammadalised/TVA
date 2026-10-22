"""Handwriting vocabulary selection and fitting, moved from DTLR into TVA.

The algorithms and deterministic tie rules reproduce the frozen IAM v1 policy.
DTLR provides localization/connectivity evidence; no image model is imported here.
Original source: DTLR poc/dtlr_poc/{handwriting_subwords,tokenizer}.py (Apache-2.0).
"""
from collections import Counter, defaultdict
import math
import unicodedata

from .common import sha256_bytes

SCHEMA = "dtlr.handwriting-subwords.v1"
SPAN_METHOD = "pairwise-consistent-dominant-component-v1"
TEXT_SCHEMA = "dtlr.text-subwords.v1"
MODEL_SCHEMA = "dtlr.handwriting-bigram-tokenizer.v1"


def text_sha256(text):
    return sha256_bytes(text.encode("utf-8"))


def prepare_corpus(records, rows, selection, manifest):
    if (selection.get("schema_version") != "dtlr.iam-selection.v1"
            or selection.get("dataset") != "IAM" or selection.get("split") != "train"):
        raise ValueError("selection must be IAM train")
    if (manifest.get("schema_version") != "dtlr.export-manifest.v5"
            or manifest.get("datasets") != ["IAM"] or manifest.get("splits") != ["train"]
            or manifest.get("primary_connectivity_method") != "dominant-core-v3"):
        raise ValueError("manifest must declare sole IAM train dominant-core-v3 evidence")
    selected = selection["lines"]
    if (len(selected) != selection["requested_count"] or len(records) != len(selected)
            or len(records) != manifest["line_count"] or len(rows) != manifest["evidence_count"]):
        raise ValueError("selection, line, or evidence counts disagree")
    corpus, by_id = [], {}
    for record, item in zip(records, selected):
        if (record.get("dataset") != "IAM" or record.get("split") != "train"
                or record["line_id"] != item["id"]
                or text_sha256(record["transcription"]) != item["transcription_sha256"]):
            raise ValueError("detection text/order differs from frozen IAM train selection")
        text = record["transcription"]
        if unicodedata.normalize("NFC", text) != text:
            raise ValueError("non-NFC evidence text needs an explicit offset remapping")
        if record["line_id"] in by_id:
            raise ValueError("duplicate line ID")
        line = {"line_id": record["line_id"], "text": text, "edges": {}}
        corpus.append(line)
        by_id[line["line_id"]] = line
    for row in rows:
        if (row.get("schema_version") != "dtlr.bigram-evidence.v5"
                or row.get("dataset") != "IAM" or row.get("split") != "train"
                or row.get("connectivity_method") != "dominant-core-v3"):
            raise ValueError("row is not IAM train dominant-core-v3 evidence")
        if row["line_id"] not in by_id:
            raise ValueError("evidence line is outside frozen selection")
        line = by_id[row["line_id"]]
        start = row["left_gt_index"]
        if (not isinstance(start, int) or start < 0 or start + 1 >= len(line["text"])
                or row["right_gt_index"] != start + 1
                or line["text"][start:start + 2] != row["pair"]
                or any(c.isspace() for c in row["pair"])):
            raise ValueError("evidence offsets/text do not match transcription")
        if start in line["edges"]:
            raise ValueError("duplicate edge evidence")
        usable = (row["usable"] is True
                  and row["left_alignment"] == row["right_alignment"] == "match")
        left, right = row.get("left_dominant_component_id"), row.get("right_dominant_component_id")
        if usable and (type(left) is not int or left <= 0 or type(right) is not int or right <= 0
                       or row["connected"] is not (left == right)):
            raise ValueError("usable edge has inconsistent dominant component evidence")
        line["edges"][start] = (left, right) if usable else None
    for line in corpus:
        expected = {i for i in range(len(line["text"]) - 1)
                    if not any(c.isspace() for c in line["text"][i:i + 2])}
        if set(line["edges"]) != expected:
            raise ValueError("incomplete per-line non-whitespace edge evidence")
    return corpus

def span_connected(line, start, end):
    """Return None for unusable/non-exact evidence, rather than False."""
    edges = [line["edges"].get(i) for i in range(start, end - 1)]
    if not edges or any(edge is None for edge in edges):
        return None
    return len({component for edge in edges for component in edge}) == 1

def collect_candidates(corpus, max_length=6, minimum_count=20, rate_threshold=0.5):
    if type(max_length) is not int or max_length < 2:
        raise ValueError("max_length must be at least two")
    if type(minimum_count) is not int or minimum_count < 1:
        raise ValueError("minimum_count must be positive")
    if not math.isfinite(rate_threshold) or not 0 <= rate_threshold <= 1:
        raise ValueError("rate_threshold must be between zero and one")
    counts = defaultdict(lambda: [0, 0, 0])
    for line in corpus:
        text = line["text"]
        for start, char in enumerate(text):
            if not char.isalpha():
                continue
            for end in range(start + 2, min(len(text), start + max_length) + 1):
                if not text[end - 1].isalpha():
                    break
                row = counts[text[start:end]]
                row[0] += 1
                connected = span_connected(line, start, end)
                if connected is not None:
                    row[1] += 1
                    row[2] += int(connected)
    result = {}
    for token, (total, exact, connected) in sorted(counts.items()):
        if exact >= minimum_count and connected / exact >= rate_threshold:
            result[token] = {"token": token, "n_total": total, "n_exact_alignment": exact,
                             "n_connected": connected, "connected_rate": connected / exact}
    return result

def collect_text_candidates(corpus, max_length=6, minimum_count=20):
    """Count letter-only spans without accessing any handwriting edges."""
    if type(max_length) is not int or max_length < 2:
        raise ValueError("max_length must be at least two")
    if type(minimum_count) is not int or minimum_count < 1:
        raise ValueError("minimum_count must be positive")
    counts = Counter()
    for line in corpus:
        text = line["text"]
        for start, char in enumerate(text):
            if not char.isalpha():
                continue
            for end in range(start + 2, min(len(text), start + max_length) + 1):
                if not text[end - 1].isalpha():
                    break
                counts[text[start:end]] += 1
    return {token: {"token": token, "n_total": count}
            for token, count in sorted(counts.items()) if count >= minimum_count}

def _model(algorithm, singletons, candidates, selected, policy):
    ordered = [""] + sorted(singletons) + sorted(selected)
    model = {"schema_version": SCHEMA, "status": "experimental-not-frozen",
            "algorithm": algorithm, "dataset": "IAM", "training_split": "train",
            "unicode_normalization": "NFC", "connectivity_method": "dominant-core-v3",
            "span_connectivity_method": SPAN_METHOD, "blank_token": "", "blank_id": 0,
            "size": len(ordered), "vocab": {t: i for i, t in enumerate(ordered)},
            "idx_token": {str(i): t for i, t in enumerate(ordered)},
            "vocabulary": [candidates[t] for t in sorted(selected)], "policy": policy}
    if algorithm.startswith("text-only-"):
        model["schema_version"] = TEXT_SCHEMA
        del model["connectivity_method"], model["span_connectivity_method"]
    return model

def _alphabet(corpus, singletons):
    observed = {c for line in corpus for c in line["text"]}
    singletons = observed if singletons is None else set(singletons)
    if any(len(c) != 1 for c in singletons) or not observed <= singletons:
        raise ValueError("singleton alphabet does not cover all training characters")
    return singletons

def _apply_merge(sequence, pair):
    result, i = [], 0
    while i < len(sequence):
        if i + 1 < len(sequence) and (sequence[i][0], sequence[i + 1][0]) == pair:
            result.append((pair[0] + pair[1], sequence[i][1], sequence[i + 1][2]))
            i += 2
        else:
            result.append(sequence[i])
            i += 1
    return result

def train_bpe(corpus, candidates, budget=145, singletons=None, handwriting_aware=True):
    """Rank current pairs by visual support, or text frequency for a control."""
    if type(handwriting_aware) is not bool:
        raise ValueError("handwriting_aware must be boolean")
    if type(budget) is not int or budget < 0:
        raise ValueError("budget must be nonnegative")
    singletons = _alphabet(corpus, singletons)
    sequences = [[(c, i, i + 1) for i, c in enumerate(line["text"])] for line in corpus]
    merges, selected = [], set()
    for _ in range(budget):
        support, frequencies = Counter(), Counter()
        for line, sequence in zip(corpus, sequences):
            for left, right in zip(sequence, sequence[1:]):
                token = left[0] + right[0]
                if token in candidates and token not in selected:
                    pair = (left[0], right[0])
                    frequencies[pair] += 1
                    if handwriting_aware and span_connected(line, left[1], right[2]) is True:
                        support[pair] += 1
        priority = support if handwriting_aware else frequencies
        if not priority:
            break
        pair = min(priority, key=lambda p: (-priority[p], p[0] + p[1], p))
        selected.add(pair[0] + pair[1])
        merge = {"left": pair[0], "right": pair[1], "token": pair[0] + pair[1],
                 "text_occurrences_at_merge": frequencies[pair]}
        if handwriting_aware:
            merge["connected_occurrences_at_merge"] = support[pair]
        merges.append(merge)
        sequences = [_apply_merge(sequence, pair) for sequence in sequences]
    algorithm = "handwriting-aware-bpe" if handwriting_aware else "text-only-bpe"
    model = _model(algorithm, singletons, candidates, selected,
                   {"subword_budget": budget, "merge_priority":
                    "current-connected-occurrence-count" if handwriting_aware else "current-text-occurrence-count",
                    "merge_application": "all-text-occurrences-left-to-right",
                    "inference": "ordered-BPE-merges"})
    model["merges"] = merges
    return model

def _logadd(a, b):
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    high, low = max(a, b), min(a, b)
    return high + math.log1p(math.exp(low - high))

def _lattice(text, log_probs):
    max_length = max(map(len, log_probs))
    return [[(end, text[start:end], log_probs[text[start:end]])
             for end in range(start + 1, min(len(text), start + max_length) + 1)
             if text[start:end] in log_probs] for start in range(len(text))]

def forward_backward(text, log_probs):
    """Exact likelihood and expected counts over all possible segmentations."""
    arcs = _lattice(text, log_probs)
    forward = [-math.inf] * (len(text) + 1)
    forward[0] = 0.0
    for start, edges in enumerate(arcs):
        for end, token, score in edges:
            forward[end] = _logadd(forward[end], forward[start] + score)
    if forward[-1] == -math.inf:
        raise ValueError("text cannot be covered by unigram vocabulary")
    backward = [-math.inf] * (len(text) + 1)
    backward[-1] = 0.0
    for start in range(len(text) - 1, -1, -1):
        for end, token, score in arcs[start]:
            backward[start] = _logadd(backward[start], score + backward[end])
    expected = Counter()
    for start, edges in enumerate(arcs):
        for end, token, score in edges:
            expected[token] += math.exp(forward[start] + score + backward[end] - forward[-1])
    return forward[-1], expected

def _em(word_counts, weights, iterations):
    total = sum(weights.values())
    probs = {t: w / total for t, w in sorted(weights.items())}
    history, expected = [], Counter()
    for _ in range(iterations):
        expected, likelihood = Counter(), 0.0
        log_probs = {t: math.log(p) for t, p in probs.items()}
        for word, count in sorted(word_counts.items()):
            score, counts = forward_backward(word, log_probs)
            likelihood += count * score
            for token, value in counts.items():
                expected[token] += count * value
        history.append(likelihood)
        weights = {t: max(expected[t], 1e-12) for t in probs}
        total = sum(weights.values())
        probs = {t: w / total for t, w in weights.items()}
    return probs, expected, history

def train_unigram(corpus, candidates, budget=145, singletons=None, iterations=8,
                  seed_multiplier=4, shrink_factor=0.75, handwriting_aware=True):
    """CCL-constrained unigram LM with EM and fixed-weight removal-loss pruning.

    This custom prototype uses exact individual removal losses at fixed weights; it is not SentencePiece.
    Images influence candidate eligibility and connected-count initialization.
    With handwriting_aware=False, use text-frequency seeds/initialization only.
    """
    if (type(budget) is not int or budget < 0 or type(iterations) is not int or iterations < 1
            or type(seed_multiplier) is not int or seed_multiplier < 1):
        raise ValueError("invalid unigram budget/iterations/seed multiplier")
    if not math.isfinite(shrink_factor) or not 0 < shrink_factor < 1:
        raise ValueError("shrink_factor must be strictly between zero and one")
    if type(handwriting_aware) is not bool:
        raise ValueError("handwriting_aware must be boolean")
    singletons = _alphabet(corpus, singletons)
    char_counts = Counter(c for line in corpus for c in line["text"])
    # All non-letter boundaries force singletons, exactly factorizing likelihood.
    words = Counter()
    for line in corpus:
        run = ""
        for c in line["text"]:
            if c.isalpha():
                run += c
            else:
                if run:
                    words[run] += 1
                    run = ""
                words[c] += 1
        if run:
            words[run] += 1
    count_key = "n_connected" if handwriting_aware else "n_total"
    ranked = sorted(candidates, key=lambda t: (-candidates[t][count_key], t))
    active = set(ranked[:budget * seed_multiplier])
    weights = {c: max(char_counts[c], 1) for c in sorted(singletons)}
    weights.update({t: max(candidates[t][count_key], 1e-12) for t in sorted(active)})
    curve = []
    while True:
        probs, expected, history = _em(words, weights, iterations)
        curve.append({"subword_count": len(active), "pre_update_log_likelihoods": history})
        if len(active) <= budget:
            break
        keep = max(budget, int(len(active) * shrink_factor))
        losses = removal_losses(words, {t: math.log(p) for t, p in probs.items()}, active)
        active = set(sorted(active, key=lambda t: (-losses[t], t))[:keep])
        weights = {t: p for t, p in probs.items() if len(t) == 1 or t in active}
    algorithm = "handwriting-aware-unigram" if handwriting_aware else "text-only-unigram"
    model = _model(algorithm, singletons, candidates, active,
                   {"subword_budget": budget, "iterations_per_stage": iterations,
                    "seed_multiplier": seed_multiplier, "shrink_factor": shrink_factor,
                    "seed_priority": "connected-occurrence-count" if handwriting_aware else "text-occurrence-count",
                    "pruning": "individual-fixed-weight-removal-log-likelihood-loss",
                    "training": "exact-forward-backward-EM-on-training-text",
                    "inference": "maximum-log-probability-Viterbi"})
    model["log_probs"] = {t: math.log(p) for t, p in sorted(probs.items())}
    model["training_curve"] = curve
    return model

def removal_losses(word_counts, log_probs, removable):
    """Exact one-piece removal loss with current weights held fixed.

    Each piece's loss is measured individually. A batch removes the pieces with
    the least loss, then EM refits/normalizes probabilities. Batch interactions
    are not accounted for; this is a declared prototype pruning policy.
    """
    losses = {t: 0.0 for t in sorted(removable)}
    def score_lattice(arcs, removed=None):
        forward = [-math.inf] * (len(arcs) + 1)
        forward[0] = 0.0
        for start, edges in enumerate(arcs):
            for end, token, score in edges:
                if token != removed:
                    forward[end] = _logadd(forward[end], forward[start] + score)
        return forward[-1]
    for word, count in sorted(word_counts.items()):
        arcs = _lattice(word, log_probs)
        relevant = {token for edges in arcs for _, token, _ in edges} & removable
        if not relevant:
            continue
        original = score_lattice(arcs)
        for token in sorted(relevant):
            without = score_lattice(arcs, token)
            if without == -math.inf:
                raise ValueError("removal would eliminate character coverage")
            losses[token] += count * (original - without)
    return losses


def pair_allowed(pair: str, policy: str) -> bool:
    if len(pair) != 2:
        return False
    if policy == "letters-only":
        return all(char.isalpha() for char in pair)
    if policy == "non-whitespace":
        return not any(char.isspace() for char in pair)
    raise ValueError(f"unsupported pair policy: {policy}")

def build_bigram_model(
    scores: list[dict],
    source_sha256: str,
    minimum_count: int,
    rate_threshold: float,
    pair_policy: str = "letters-only",
    required_characters: tuple[str, ...] = (" ",),
) -> dict:
    if minimum_count <= 0:
        raise ValueError("minimum_count must be positive")
    if not 0.0 <= rate_threshold <= 1.0:
        raise ValueError("rate_threshold must be between zero and one")
    if not scores:
        raise ValueError("bigram score input is empty")
    if any(len(char) != 1 for char in required_characters):
        raise ValueError("required characters must each contain exactly one character")
    pairs = set()
    characters = set(required_characters)
    vocabulary = []
    for row in scores:
        if row.get("schema_version") != "dtlr.bigram-scores.v5":
            raise ValueError("unsupported bigram score schema")
        if row.get("dataset") != "IAM" or row.get("split") != "train":
            raise ValueError("tokenizer vocabulary must be derived from IAM train scores only")
        if row.get("connectivity_method") != "dominant-core-v3":
            raise ValueError("tokenizer scores do not use frozen dominant-core-v3")
        pair = row["pair"]
        if len(pair) != 2:
            raise ValueError(f"score pair must contain exactly two characters: {pair!r}")
        if pair in pairs:
            raise ValueError(f"duplicate score row for pair {pair!r}")
        pairs.add(pair)
        characters.update(pair)
        count = row.get("n_exact_alignment", 0)
        rate = row.get("exact_alignment_connected_rate")
        if (
            pair_allowed(pair, pair_policy)
            and count >= minimum_count
            and rate is not None
            and rate >= rate_threshold
        ):
            vocabulary.append({
                "token": pair,
                "n_exact_alignment": count,
                "exact_alignment_connected_rate": rate,
                "utility": rate,
            })
    vocabulary.sort(key=lambda row: row["token"])
    # TVA reserves ID 0 for the CTC blank, represented by the empty string.
    ordered_tokens = [""] + sorted(characters) + [row["token"] for row in vocabulary]
    vocab = {token: index for index, token in enumerate(ordered_tokens)}
    idx_token = {str(index): token for token, index in vocab.items()}
    return {
        "schema_version": MODEL_SCHEMA,
        "status": "provisional-demo-policy",
        "dataset": "IAM",
        "training_split": "train",
        "connectivity_method": "dominant-core-v3",
        "score_field": "exact_alignment_connected_rate",
        "count_field": "n_exact_alignment",
        "blank_token": "",
        "blank_id": 0,
        "source_scores_sha256": source_sha256,
        "policy": {
            "minimum_count": minimum_count,
            "rate_threshold": rate_threshold,
            "pair_policy": pair_policy,
            "overlap_resolution": "maximum-total-utility-non-overlapping-v1",
            "single_character_utility": 0.0,
            "required_characters": list(required_characters),
        },
        "vocab": vocab,
        "idx_token": idx_token,
        "size": len(vocab),
        "eligible_bigram_count": len(vocabulary),
        "vocabulary": vocabulary,
        "note": (
            "Training-derived demonstration model. Policy thresholds are not held-out "
            "performance and must be validated before thesis claims."
        ),
    }



def native_model_bytes(model, kind):
    """Preserve each release's original JSON serialization convention."""
    import json
    return (json.dumps(model, ensure_ascii=False, indent=2, sort_keys=kind != 'bigram',
                       allow_nan=False) + '\n').encode('utf-8')


def fit_ed_tokenizer(evidence_path, kind):
    """Fit the IAM v1 policy in TVA and return its byte-identical ED projection.

    Frozen references supply historical release metadata and verification only.
    Candidates, merges, probabilities and training curves are recomputed from
    the evidence, without ED labels or imports from DTLR.
    """
    import json
    from pathlib import Path
    import tempfile
    from .common import canonical_json_bytes, require
    from .evidence import load_evidence, EVIDENCE_SHA256, SOURCE_SHA256
    from .ed import build_ed_adapter, validate_frozen_iam_source, SOURCE_SHA256 as BIGRAM_SHA256
    from .subwords import load_source, build_ed_subword_adapter, SOURCE_FILES

    require(kind in ('bigram', 'bpe', 'unigram'), 'Unsupported handwriting training algorithm')
    corpus, bundle = load_evidence(evidence_path)
    source_directory = Path(__file__).resolve().parents[2] / 'artifacts/tokenizers/ed/handwriting' / kind / 'source'
    if kind == 'bigram':
        path = source_directory / 'iam-english-handwriting-bigram-v1.json'
        raw = path.read_bytes()
        reference = json.loads(raw)
        validate_frozen_iam_source(reference, sha256_bytes(raw))
        trained = build_bigram_model(bundle['scores'], SOURCE_SHA256['scores'], 20, 0.5)
        ignored = {'status', 'note'}
        candidate_count = len(bundle['scores'])
    else:
        path = source_directory / SOURCE_FILES[kind]
        reference = load_source(path, kind)
        candidates = collect_candidates(corpus, max_length=6, minimum_count=20, rate_threshold=0.5)
        require(len(candidates) == 303, 'Frozen candidate count differs')
        require(sha256_bytes(canonical_json_bytes(candidates)) == reference['freeze']['candidate_pool_sha256'],
                'Recomputed candidate pool differs from the frozen policy')
        trained = train_bpe(corpus, candidates, budget=145) if kind == 'bpe' else train_unigram(
            corpus, candidates, budget=145, iterations=8, seed_multiplier=4, shrink_factor=0.75)
        ignored = {'status'}
        candidate_count = len(candidates)
    for key, value in trained.items():
        if key not in ignored:
            require(value == reference.get(key), f'Recomputed {kind} field differs from frozen v1: {key}')
    # This is the newly fitted model, with the original freeze metadata unchanged.
    reproduced = dict(reference)
    reproduced.update({key: value for key, value in trained.items() if key not in ignored})
    require(native_model_bytes(reproduced, kind) == path.read_bytes(), 'Native frozen model reproduction differs')
    with tempfile.TemporaryDirectory(prefix='tva-handwriting-fit-') as directory:
        native = Path(directory) / path.name
        native.write_bytes(native_model_bytes(reproduced, kind))
        adapter = build_ed_adapter(native) if kind == 'bigram' else build_ed_subword_adapter(native, kind)
    report = {
        'schema_version': 'tva.handwriting-tokenizer-training.v1', 'trainer_repository': 'TVA',
        'algorithm': kind, 'dataset': 'IAM', 'split': 'train', 'line_count': len(corpus),
        'evidence_sha256': EVIDENCE_SHA256, 'evidence_source_sha256': SOURCE_SHA256,
        'candidate_count': candidate_count, 'selected_piece_count': 145, 'ed_ctc_classes': adapter['size'],
        'learned_fields_recomputed_from_evidence': True, 'frozen_native_bytes_reproduced': True,
        'frozen_native_sha256': sha256_bytes(native_model_bytes(reproduced, kind)),
        'ed_adapter_sha256': sha256_bytes(canonical_json_bytes(adapter)),
        'historical_source_provenance_preserved': True,
        'annotation_label_values_read_from_ed': False,
        'implementation_sha256': {
            name: sha256_bytes((Path(__file__).parent / name).read_bytes())
            for name in ('training.py', 'evidence.py')
        },
    }
    return adapter, reproduced, report


def save_training_outputs(output_path, kind, adapter, native, report):
    """Preflight every file before writing the fitted tokenizer and training report."""
    from pathlib import Path
    from .common import canonical_json_bytes, require, save_frozen
    from .subwords import SOURCE_FILES

    output = Path(output_path)
    source_name = 'iam-english-handwriting-bigram-v1.json' if kind == 'bigram' else SOURCE_FILES[kind]
    files = {
        output: canonical_json_bytes(adapter),
        output.parent / 'source' / source_name: native_model_bytes(native, kind),
        output.with_suffix('.training.json'): canonical_json_bytes(report),
    }
    for path, content in files.items():
        require(not path.exists() or path.read_bytes() == content, f'Existing frozen file differs: {path}')
    for path, content in files.items():
        save_frozen(path, content)
