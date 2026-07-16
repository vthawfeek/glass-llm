"""Tokenizer audit — one uniform interface over our from-scratch tokenizers and real production
tokenizers (GPT-4, GPT-4o, GPT-2, BERT). Loads entirely from the repo-local cache (offline)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# force offline: use only the pre-cached blobs (R4) — no runtime network on the deploy host
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(ROOT / "assets" / "tiktoken_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import sys
sys.path.insert(0, str(ROOT / "src"))
from bpe import BPETokenizer

# name -> (kind, spec).  Order defines dropdown order.
REGISTRY = {
    "Ours · clinical trials (domain)": ("ours", "dom_4096"),
    # NB: `domain_4096` (the v1 tokenizer) is deliberately absent. It is superseded by `dom_4096`,
    # which is what build_zoo.py trains and every models_v2 checkpoint uses. Listing both put two
    # near-identically-labelled entries in the dropdown that disagreed (pembrolizumab: 4 vs 2).
    "Ours · general (English)": ("ours", "general_4096"),
    "GPT-4 / GPT-3.5 (cl100k)": ("tiktoken", "cl100k_base"),
    "GPT-4o (o200k)":           ("tiktoken", "o200k_base"),
    "GPT-2":                    ("hf", "gpt2"),
    "BERT (WordPiece)":         ("hf", "bert"),
}
_CACHE = {}


def _load(name):
    if name in _CACHE:
        return _CACHE[name]
    kind, spec = REGISTRY[name]
    if kind == "ours":
        obj = BPETokenizer.load(ROOT / "pilot" / "tokenizers" / f"{spec}.json")
    elif kind == "tiktoken":
        import tiktoken
        obj = tiktoken.get_encoding(spec)
    else:
        # lightweight: load the saved tokenizer.json directly (no full `transformers`)
        from tokenizers import Tokenizer
        obj = Tokenizer.from_file(str(ROOT / "assets" / "hf" / spec / "tokenizer.json"))
    _CACHE[name] = (kind, obj)
    return _CACHE[name]


def pieces(name, text):
    """Return the list of human-readable token pieces for `text` under tokenizer `name`."""
    kind, obj = _load(name)
    if not text:
        return []
    if kind == "ours":
        return [obj.vocab[i].decode("utf-8", "replace").replace(" ", "␣") for i in obj.encode(text)]
    if kind == "tiktoken":
        return [obj.decode_single_token_bytes(i).decode("utf-8", "replace").replace(" ", "␣")
                for i in obj.encode(text)]
    # hf tokenizer.json via the `tokenizers` lib (exclude [CLS]/[SEP] so we measure the word)
    return [t.replace("Ġ", "␣").replace("▁", "␣")
            for t in obj.encode(text, add_special_tokens=False).tokens]


def count(name, text):
    return len(pieces(name, text))


def fertility(name, terms):
    """Mean tokens per term over a probe list (drug names, gene variants, phrases)."""
    ns = [count(name, t) for t in terms if t.strip()]
    return sum(ns) / max(1, len(ns))


def names():
    return list(REGISTRY)
