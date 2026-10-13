"""Native-IAM encoder for mathematical trainer tests (not an ED runtime)."""
import math
import unicodedata
from tva.handwriting.training import SCHEMA, TEXT_SCHEMA, _apply_merge, _lattice

class HandwritingSubwordTokenizer:
    def __init__(self, model):
        algorithms = {SCHEMA: ("handwriting-aware-bpe", "handwriting-aware-unigram"),
                      TEXT_SCHEMA: ("text-only-bpe", "text-only-unigram")}
        if model.get("algorithm") not in algorithms.get(model.get("schema_version"), ()):
            raise ValueError("unsupported handwriting subword model")
        vocab = model["vocab"]
        inverse = {int(i): t for i, t in model["idx_token"].items()}
        if (vocab.get("") != 0 or model.get("blank_id") != 0 or model.get("blank_token") != ""
                or model.get("size") != len(vocab) or set(vocab.values()) != set(range(len(vocab)))
                or inverse != {i: t for t, i in vocab.items()}
                or model.get("unicode_normalization") != "NFC"):
            raise ValueError("invalid CTC vocabulary or normalization")
        subwords = {t for t in vocab if len(t) > 1}
        if any(not all(c.isalpha() and c in vocab for c in t) for t in subwords):
            raise ValueError("subwords must be letters with singleton fallback")
        if model["algorithm"].endswith("-bpe"):
            available = {t for t in vocab if len(t) == 1}
            for merge in model["merges"]:
                if (merge["left"] not in available or merge["right"] not in available
                        or merge["token"] != merge["left"] + merge["right"]
                        or merge["token"] not in subwords or merge["token"] in available):
                    raise ValueError("invalid BPE merge order")
                available.add(merge["token"])
            if available != set(vocab) - {""}:
                raise ValueError("BPE vocabulary differs from learned merges")
        else:
            scores = model["log_probs"]
            if (set(scores) != set(vocab) - {""}
                    or any(not math.isfinite(v) or v > 0 for v in scores.values())
                    or not math.isclose(sum(math.exp(v) for v in scores.values()), 1, abs_tol=1e-8)):
                raise ValueError("invalid unigram probabilities")
        self.model, self.vocab, self.idx_token = model, vocab, inverse
        self.size = len(vocab)

    def segment(self, text):
        text = unicodedata.normalize("NFC", text)
        missing = sorted(set(text) - set(self.vocab))
        if missing:
            raise ValueError(f"characters are absent from tokenizer vocabulary: {missing!r}")
        if self.model["algorithm"].endswith("-bpe"):
            sequence = [(c, i, i + 1) for i, c in enumerate(text)]
            for merge in self.model["merges"]:
                sequence = _apply_merge(sequence, (merge["left"], merge["right"]))
        else:
            arcs = _lattice(text, self.model["log_probs"])
            best, back = [-math.inf] * (len(text) + 1), [None] * (len(text) + 1)
            best[0] = 0.0
            for start, edges in enumerate(arcs):
                for end, token, score in edges:
                    candidate = best[start] + score
                    if candidate > best[end]:
                        best[end], back[end] = candidate, (token, start, end)
            sequence, end = [], len(text)
            while end:
                step = back[end]
                if step is None:
                    raise ValueError("text cannot be segmented")
                sequence.append(step)
                end = step[1]
            sequence.reverse()
        return {"text": text, "tokens": [t for t, _, _ in sequence],
                "segments": [{"token": t, "start": s, "end": e} for t, s, e in sequence]}

    def encode(self, text):
        return [self.vocab[t] for t in self.segment(text)["tokens"]]

    def decode(self, ids):
        try:
            return "".join(self.idx_token[i] for i in ids)
        except KeyError as error:
            raise ValueError(f"unknown token ID: {error.args[0]}") from error
