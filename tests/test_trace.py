"""R2 gate: the trace() API is the linchpin — if it's wrong, every panel is wrong.
These tests do NOT need trained weights (a random-init model exercises the same paths)."""
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bpe import BPETokenizer
from model import GPT
from glass import GlassModel

VOCAB = 512
CFG = dict(d_model=64, n_layer=3, n_head=4, ctx=64)
PROMPT = "Inclusion Criteria: histologically confirmed metastatic cancer; pembrolizumab 200 mg"


def _glass():
    tok = BPETokenizer().train(
        "Inclusion Criteria: metastatic cancer pembrolizumab osimertinib ECOG 0-1 " * 30,
        vocab_size=VOCAB)
    model = GPT(vocab_size=tok.vocab_size, **CFG)
    return GlassModel(model, tok)


def test_shapes_consistent():
    g = _glass(); tr = g.trace(PROMPT)
    T = tr.n_tokens
    assert T >= 1 and T <= CFG["ctx"]
    assert len(tr.token_texts) == T
    assert tr.embeddings.shape == (T, CFG["d_model"])
    assert len(tr.hiddens) == CFG["n_layer"] + 1
    for h in tr.hiddens:
        assert h.shape == (T, CFG["d_model"])
    assert len(tr.attentions) == CFG["n_layer"]
    for a in tr.attentions:
        assert a.shape == (CFG["n_head"], T, T)


def test_attention_rows_sum_to_one():
    g = _glass(); tr = g.trace(PROMPT)
    for layer, a in enumerate(tr.attentions):
        row_sums = a.sum(axis=-1)                 # (n_head, T)
        assert np.allclose(row_sums, 1.0, atol=1e-4), f"layer {layer} rows don't sum to 1"


def test_attention_is_causal():
    g = _glass(); tr = g.trace(PROMPT)
    for a in tr.attentions:
        for head in a:
            upper = np.triu(head, k=1)            # strictly-future positions
            assert np.allclose(upper, 0.0, atol=1e-6), "attends to future tokens (mask broken)"


def test_logits_and_topk():
    g = _glass(); tr = g.trace(PROMPT)
    assert tr.logits.shape == (g.tok.vocab_size,)   # BPE may stop early on tiny corpora
    probs = [p for _, p, _ in tr.topk]
    assert probs == sorted(probs, reverse=True)   # descending
    assert 0.0 <= sum(probs) <= 1.0 + 1e-5


def test_token_alignment_roundtrip():
    # concatenated token pieces must reconstruct the (un-truncated) prompt
    g = _glass()
    ids = g.tok.encode(PROMPT)
    assert g.tok.decode(ids) == PROMPT
    tr = g.trace(PROMPT)
    assert tr.token_ids == ids[-CFG["ctx"]:]


def test_generate_length_and_determinism():
    g = _glass()
    a = g.generate(PROMPT, max_new_tokens=12, temperature=0.8, top_k=20, seed=1)
    b = g.generate(PROMPT, max_new_tokens=12, temperature=0.8, top_k=20, seed=1)
    assert len(a["steps"]) == 12
    assert a["text"] == b["text"]                 # same seed -> identical
    assert a["watermark"].startswith("SYNTHETIC")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"PASS {name}")
    print("all trace tests passed")
