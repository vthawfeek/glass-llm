"""Transparent RAG retriever — a vector index over real trial chunks (with NCT IDs), embedded
with OUR model. Cosine top-k, shown with scores. Same operation a vector DB (FAISS/Chroma)
performs, kept transparent and dependency-free. Index is built offline and loaded from disk."""
import json
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

RAG_DIR = Path("pilot/rag")


def _chunks_from_body(nct, body, max_chars=320):
    """Split a trial record into short passages, tagged with the NCT ID."""
    out, cur = [], ""
    for line in body.split("\n"):
        line = line.strip()
        if not line or line == "<|endoftext|>":
            continue
        if len(cur) + len(line) > max_chars and cur:
            out.append(cur.strip()); cur = ""
        cur += " " + line
    if cur.strip():
        out.append(cur.strip())
    return [{"nct": nct, "text": c} for c in out if len(c) > 40]


def build_index(model, n_trials=500, out_dir=RAG_DIR, area="all"):
    """Fetch trials (with NCT IDs), chunk, embed with `model`, and persist the index."""
    from fetch_data import fetch_trials
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    docs = fetch_trials(area, n_trials, quick=False)
    chunks = []
    for nct, body in docs.items():
        chunks.extend(_chunks_from_body(nct, body))
    embs = np.stack([model.embed(c["text"]) for c in chunks]).astype(np.float32)
    embs /= (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)     # unit-normalize
    np.save(out_dir / "embeddings.npy", embs)
    (out_dir / "chunks.json").write_text(json.dumps(chunks), encoding="utf-8")
    print(f"[rag] built index: {len(chunks)} chunks from {len(docs)} trials -> {out_dir}")
    return len(chunks)


def load_index(out_dir=RAG_DIR):
    out_dir = Path(out_dir)
    embs = np.load(out_dir / "embeddings.npy")
    chunks = json.loads((out_dir / "chunks.json").read_text(encoding="utf-8"))
    return embs, chunks


def query(model, embs, chunks, text, k=5, unique_trials=True):
    """Cosine top-k retrieval. Returns list of (nct, text, score).
    unique_trials keeps only the best-scoring chunk per NCT so results show diverse trials."""
    q = model.embed(text).astype(np.float32)
    q /= (np.linalg.norm(q) + 1e-9)
    sims = embs @ q
    out, seen = [], set()
    for i in np.argsort(-sims):
        nct = chunks[i]["nct"]
        if unique_trials and nct in seen:
            continue
        seen.add(nct)
        out.append((nct, chunks[i]["text"], float(sims[i])))
        if len(out) >= k:
            break
    return out


def build_grounded_prompt(question, hits):
    ctx = "\n".join(f"[{nct}] {text}" for nct, text, _ in hits)
    return (f"Context from real trials:\n{ctx}\n\nQuestion: {question}\nAnswer using only the context above:")


def exists(out_dir=RAG_DIR):
    out_dir = Path(out_dir)
    return (out_dir / "embeddings.npy").exists() and (out_dir / "chunks.json").exists()
