"""From-scratch byte-level BPE (GPT-2-style regex pre-tokenization for tractable training).
Minbpe-inspired; written for the Glass-LLM pilot. No external tokenizer libraries."""
import json
from collections import Counter
from pathlib import Path
import regex as re

# GPT-2 pre-tokenization pattern: keeps words, numbers, punctuation runs separate.
PAT = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def _chunks_to_words(text):
    """Return {tuple_of_byte_ids: count} over regex-pretokenized chunks."""
    words = Counter()
    for chunk in PAT.findall(text):
        words[tuple(chunk.encode("utf-8"))] += 1
    return words


def _pair_stats(words):
    stats = Counter()
    for word, c in words.items():
        for pair in zip(word, word[1:]):
            stats[pair] += c
    return stats


def _merge_word(word, pair, new_id):
    out, i, n = [], 0, len(word)
    while i < n:
        if i < n - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
            out.append(new_id); i += 2
        else:
            out.append(word[i]); i += 1
    return tuple(out)


class BPETokenizer:
    def __init__(self):
        self.merges = {}                                   # (a, b) -> new_id
        self.vocab = {i: bytes([i]) for i in range(256)}   # id -> bytes
        self.pattern = PAT

    # ---- training ----
    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        words = _chunks_to_words(text)
        num_merges = vocab_size - 256
        for i in range(num_merges):
            stats = _pair_stats(words)
            if not stats:
                break
            best = max(stats, key=lambda p: (stats[p], -p[0], -p[1]))  # deterministic
            new_id = 256 + i
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            words = {_merge_word(w, best, new_id): c for w, c in words.items()}
            if verbose and i % 500 == 0:
                print(f"  merge {i}/{num_merges} {best} -> {new_id} (count {stats[best]})")
        return self

    # ---- encoding ----
    def _encode_chunk(self, ids):
        while len(ids) >= 2:
            pairs = set(zip(ids, ids[1:]))
            pair = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = list(_merge_word(tuple(ids), pair, self.merges[pair]))
        return ids

    def encode(self, text):
        out = []
        for chunk in self.pattern.findall(text):
            out.extend(self._encode_chunk(list(chunk.encode("utf-8"))))
        return out

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    @property
    def vocab_size(self):
        return len(self.vocab)

    # ---- persistence ----
    def save(self, path):
        path = Path(path)
        obj = {"merges": [[a, b, nid] for (a, b), nid in self.merges.items()],
               "vocab_size": self.vocab_size}
        path.write_text(json.dumps(obj), encoding="utf-8")

    @classmethod
    def load(cls, path):
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        t = cls()
        for a, b, nid in obj["merges"]:
            t.merges[(a, b)] = nid
            t.vocab[nid] = t.vocab[a] + t.vocab[b]
        return t
