"""Handwriting-aware BPE and Unigram vocabulary learning.

Connectivity evidence decides which letter spans may become pieces and how
they are ranked; the rest follows standard BPE and Unigram training. The
Unigram EM can also run on text alone (frequent_spans() seeds), as the
linguistic partner that differs only in the missing connectivity.
Original source: DTLR poc/dtlr_poc/handwriting_subwords.py (Apache-2.0).

A corpus is a list of lines {'text': str, 'edges': {i: (left, right) | None}}
where edges[i] gives the connected-component IDs of characters i and i + 1,
or None when the alignment of that pair is unreliable.
"""
from collections import Counter, defaultdict
import math


def span_connected(line, start, end):
    """Return None for unusable/non-exact evidence, rather than False."""
    edges = [line["edges"].get(i) for i in range(start, end - 1)]
    if not edges or any(edge is None for edge in edges):
        return None
    return len({component for edge in edges for component in edge}) == 1


def _letter_spans(text, max_length):
    """(start, end) of every letter-only span of 2..max_length characters."""
    for start, char in enumerate(text):
        if not char.isalpha():
            continue
        for end in range(start + 2, min(len(text), start + max_length) + 1):
            if not text[end - 1].isalpha():
                break
            yield start, end


def frequent_spans(corpus, max_length=6, minimum_count=20):
    """Letter spans of 2..max_length seen at least minimum_count times.

    The text-only counterpart of collect_candidates(): every frequent span is
    a candidate, and its frequency (not its connected count) is its seed.

    Returns:
        Span -> number of occurrences.
    """
    if type(max_length) is not int or max_length < 2:
        raise ValueError("max_length must be at least two")
    if type(minimum_count) is not int or minimum_count < 1:
        raise ValueError("minimum_count must be positive")
    counts = Counter(line["text"][start:end] for line in corpus
                     for start, end in _letter_spans(line["text"], max_length))
    return {token: n for token, n in sorted(counts.items()) if n >= minimum_count}


def collect_candidates(corpus, max_length=6, minimum_count=20, rate_threshold=0.5):
    """Letter spans of 2..max_length written connected often enough."""
    if type(max_length) is not int or max_length < 2:
        raise ValueError("max_length must be at least two")
    if type(minimum_count) is not int or minimum_count < 1:
        raise ValueError("minimum_count must be positive")
    if not math.isfinite(rate_threshold) or not 0 <= rate_threshold <= 1:
        raise ValueError("rate_threshold must be between zero and one")
    counts = defaultdict(lambda: [0, 0, 0])
    for line in corpus:
        for start, end in _letter_spans(line["text"], max_length):
            row = counts[line["text"][start:end]]
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


def train_bpe(corpus, candidates, budget=145):
    """Learn up to budget ordered merges, ranked by connected occurrences.

    Returns:
        List of (left, right) merges in learned order.
    """
    if type(budget) is not int or budget < 0:
        raise ValueError("budget must be nonnegative")
    sequences = [[(c, i, i + 1) for i, c in enumerate(line["text"])] for line in corpus]
    merges, selected = [], set()
    for _ in range(budget):
        support = Counter()
        for line, sequence in zip(corpus, sequences):
            for left, right in zip(sequence, sequence[1:]):
                token = left[0] + right[0]
                if (token in candidates and token not in selected
                        and span_connected(line, left[1], right[2]) is True):
                    support[(left[0], right[0])] += 1
        if not support:
            break
        pair = min(support, key=lambda p: (-support[p], p[0] + p[1], p))
        selected.add(pair[0] + pair[1])
        merges.append(pair)
        sequences = [_apply_merge(sequence, pair) for sequence in sequences]
    return merges


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
    for _ in range(iterations):
        expected = Counter()
        log_probs = {t: math.log(p) for t, p in probs.items()}
        for word, count in sorted(word_counts.items()):
            _, counts = forward_backward(word, log_probs)
            for token, value in counts.items():
                expected[token] += count * value
        weights = {t: max(expected[t], 1e-12) for t in probs}
        total = sum(weights.values())
        probs = {t: w / total for t, w in weights.items()}
    return probs


def train_unigram(corpus, seeds, budget=145, iterations=8,
                  seed_multiplier=4, shrink_factor=0.75):
    """Unigram LM seeded by the given counts, fitted with EM, pruned to budget.

    Pieces start from the budget * seed_multiplier candidates with the highest
    seed counts (connected counts for the handwriting Unigram, frequencies for
    the text-only one), initialized by those counts. Each stage runs EM on the
    training text, then keeps the shrink_factor share of pieces whose
    individual removal would lose the most likelihood. Only line['text'] is
    read, so connectivity enters through the seeds alone.

    Args:
        corpus: Lines {'text': str, ...}.
        seeds: Candidate piece -> starting count.

    Returns:
        Log-probabilities of the single characters and selected pieces.
    """
    if (type(budget) is not int or budget < 0 or type(iterations) is not int or iterations < 1
            or type(seed_multiplier) is not int or seed_multiplier < 1):
        raise ValueError("invalid unigram budget/iterations/seed multiplier")
    if not math.isfinite(shrink_factor) or not 0 < shrink_factor < 1:
        raise ValueError("shrink_factor must be strictly between zero and one")
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
    ranked = sorted(seeds, key=lambda t: (-seeds[t], t))
    active = set(ranked[:budget * seed_multiplier])
    weights = {c: max(char_counts[c], 1) for c in sorted(char_counts)}
    weights.update({t: max(seeds[t], 1e-12) for t in sorted(active)})
    while True:
        probs = _em(words, weights, iterations)
        if len(active) <= budget:
            break
        keep = max(budget, int(len(active) * shrink_factor))
        losses = removal_losses(words, {t: math.log(p) for t, p in probs.items()}, active)
        active = set(sorted(active, key=lambda t: (-losses[t], t))[:keep])
        weights = {t: p for t, p in probs.items() if len(t) == 1 or t in active}
    return {t: math.log(p) for t, p in sorted(probs.items())}


def removal_losses(word_counts, log_probs, removable):
    """Exact one-piece removal loss with current weights held fixed.

    Each piece's loss is measured individually. A batch removes the pieces with
    the least loss, then EM refits/normalizes probabilities. Batch interactions
    are not accounted for.
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
