"""Unit tests for the from-scratch MiniLM encoder (src/minilm.py): shapes, determinism, and a
coarse relevance sanity check on a tiny hand-built index (independent of the real 5960-chunk
pilot/rag_dom index, so this runs fast with no data dependency).
Run: python tests/test_minilm.py
"""
import sys
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from minilm import MiniLMEmbedder, D_MODEL

passed = failed = 0


def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
        print(f"PASS  {name}")
    else:
        failed += 1
        print(f"FAIL  {name}")


def main():
    embedder = MiniLMEmbedder()

    v = embedder.embed("HER2-positive metastatic breast cancer eligibility")
    check("embed() returns the right shape", v.shape == (D_MODEL,))
    check("embed() returns finite values", np.isfinite(v).all())
    check("embed() is non-degenerate (nonzero norm)", np.linalg.norm(v) > 1e-6)

    v2 = embedder.embed("HER2-positive metastatic breast cancer eligibility")
    check("embed() is deterministic for the same input", np.allclose(v, v2))

    batch = embedder.embed_batch(["pembrolizumab melanoma trial", "HER2-positive breast cancer"])
    single0 = embedder.embed("pembrolizumab melanoma trial")
    cos = (batch[0] @ single0) / (np.linalg.norm(batch[0]) * np.linalg.norm(single0) + 1e-9)
    check("batched and single-item encoding agree (padding doesn't leak)", cos > 0.999)

    # tiny synthetic index: relevance sanity check independent of the real corpus
    docs = [
        ("oncology", "Eligibility: histologically confirmed HER2-positive metastatic breast cancer, ECOG 0-1."),
        ("cardiology", "Eligibility: NYHA class II-III heart failure with reduced ejection fraction."),
        ("respiratory", "Eligibility: physician-diagnosed asthma requiring daily inhaled corticosteroids."),
        ("nutrition", "A recipe study comparing chocolate chip cookie recipes for taste preference."),
    ]
    doc_embs = np.stack([embedder.embed(t) for _, t in docs])
    doc_embs = doc_embs / np.linalg.norm(doc_embs, axis=1, keepdims=True)

    def top1(query):
        q = embedder.embed(query)
        q = q / (np.linalg.norm(q) + 1e-9)
        sims = doc_embs @ q
        return docs[int(np.argmax(sims))][0], sims

    label, sims = top1("breast cancer HER2 treatment")
    check("retrieves the oncology doc for a breast-cancer query", label == "oncology")
    label, sims = top1("congestive heart failure ejection fraction")
    check("retrieves the cardiology doc for a heart-failure query", label == "cardiology")
    label, sims = top1("asthma inhaler corticosteroid")
    check("retrieves the respiratory doc for an asthma query", label == "respiratory")

    print(f"\n{passed}/{passed + failed} passed.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
