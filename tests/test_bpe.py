"""Round-trip and invariant tests for the from-scratch BPE tokenizer."""
import sys, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.bpe import BPETokenizer

TRAIN = (
    "Inclusion Criteria:\n- Histologically confirmed metastatic non-small cell lung cancer\n"
    "- ECOG performance status 0-1\n- Adequate organ function\n"
    "Patients received pembrolizumab, osimertinib, or trastuzumab. EGFR T790M and BRAF V600E.\n"
) * 40

SAMPLES = [
    "pembrolizumab 200 mg",
    "EGFR exon 19 deletion; BRAF V600E",
    "ECOG performance status 0-1",
    "",                      # empty string
    "  ",                   # whitespace only
    "naïve café — résumé",  # non-ASCII (multi-byte UTF-8)
    "NCT01234567\n\nInformed consent required.",
    "αβγ 中文 🧬",            # emoji + non-Latin scripts
]


def _tok(vocab=512):
    return BPETokenizer().train(TRAIN, vocab_size=vocab)


def test_roundtrip_samples():
    t = _tok()
    for s in SAMPLES:
        assert t.decode(t.encode(s)) == s, f"round-trip failed: {s!r}"


def test_roundtrip_random():
    t = _tok()
    rng = random.Random(0)
    alphabet = "abcdefghijklmnopqrstuvwxyz ABC-0123:,\n"
    for _ in range(200):
        s = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 60)))
        assert t.decode(t.encode(s)) == s


def test_vocab_size_grows():
    # BPE may stop early if the corpus runs out of mergeable pairs; it must never
    # exceed the requested size and must grow beyond the 256-byte base.
    t = _tok(vocab=400)
    assert 256 < t.vocab_size <= 400
    assert all(isinstance(v, bytes) for v in t.vocab.values())


def test_save_load_roundtrip(tmp_path=Path("pilot/tokenizers/_test.json")):
    t = _tok()
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    t.save(tmp_path)
    t2 = BPETokenizer.load(tmp_path)
    for s in SAMPLES:
        assert t.encode(s) == t2.encode(s)
        assert t2.decode(t2.encode(s)) == s
    tmp_path.unlink()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"PASS {name}")
    print("all BPE tests passed")
