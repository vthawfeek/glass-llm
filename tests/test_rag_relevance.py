"""Regression guard for RAG retrieval quality on the real 5,960-chunk index
(pilot/rag_dom). This encodes the finding that motivated the MiniLM embedding option: the
model's own self-embedding (next-token training objective, never trained for topical
similarity) frequently retrieves off-topic chunks and fails to separate on-topic from
off-topic queries by score, while MiniLM (trained for sentence similarity) does. Run after
the zoo build + `python src/build_minilm_index.py`:  python tests/test_rag_relevance.py
"""
import sys, re
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import rag
from minilm import MiniLMEmbedder

passed = failed = 0


def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
        print(f"PASS  {name}")
    else:
        failed += 1
        print(f"FAIL  {name}")


def keyword_hit_rate(hits, pattern):
    rx = re.compile(pattern, re.I)
    return sum(1 for _, text, _ in hits if rx.search(text)) / len(hits)


def main():
    embs_minilm = rag.load_minilm_embeddings(ROOT / "pilot" / "rag_dom")
    _, chunks = rag.load_index(ROOT / "pilot" / "rag_dom")
    if embs_minilm is None:
        print("embeddings_minilm.npy not built yet; run src/build_minilm_index.py first.")
        sys.exit(1)

    embedder = MiniLMEmbedder()

    # on-topic queries: MiniLM top-5 should be substantially on-topic
    probes = [
        ("breast cancer", r"breast"),
        ("melanoma", r"melanoma"),
        ("pembrolizumab", r"pembrolizumab"),
        ("type 2 diabetes", r"diabet"),
        ("asthma", r"asthma"),
    ]
    for query, pattern in probes:
        hits = rag.query(embedder, embs_minilm, chunks, query, k=5)
        rate = keyword_hit_rate(hits, pattern)
        check(f"MiniLM: {rate*100:.0f}% of top-5 for {query!r} mention the topic (need >=60%)",
              rate >= 0.6)

    # off-topic queries should score noticeably lower than on-topic ones (a usable "no match"
    # signal, the property self-embedding was found to lack -- it scored garbage HIGHER)
    def top_score(query):
        return rag.query(embedder, embs_minilm, chunks, query, k=1)[0][2]

    on_topic_scores = [top_score(q) for q, _ in probes]
    off_topic_scores = [top_score(q) for q in
                         ["quantum computing error correction", "recipe for chocolate chip cookies"]]
    check(f"MiniLM separates on-topic ({min(on_topic_scores):.3f}+) from off-topic "
          f"({max(off_topic_scores):.3f} max) by a real margin",
          min(on_topic_scores) > max(off_topic_scores) + 0.1)

    # exact-match sanity: a chunk's own distinctive topic phrase should retrieve it at rank 1
    target = chunks[0]
    assert "Shoulder Dislocation" in target["text"], "fixture assumption changed, update this test"
    hits = rag.query(embedder, embs_minilm, chunks, "shoulder dislocation self reduction", k=1)
    check("MiniLM ranks a chunk's own topic query as its #1 nearest neighbour",
          hits[0][0] == target["nct"])

    print(f"\n{passed}/{passed + failed} passed.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
