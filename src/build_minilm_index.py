"""Build the MiniLM embedding index over the SAME chunks as the self-embedding RAG index
(pilot/rag_dom/chunks.json), so the dashboard can compare both embedding sources on identical
data. Writes pilot/rag_dom/embeddings_minilm.npy.

Resumable/checkpointed on purpose: this dev machine has shown intermittent access violations
under low free memory when doing sustained torch work (see src/minilm.py docstring). Progress is
saved after every batch to .minilm_ckpt/ next to the output, so re-running this script after a
crash picks up where it left off instead of restarting. Run: python src/build_minilm_index.py
"""
import sys
import time
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

torch.set_num_threads(1)  # smaller peak memory footprint; see src/minilm.py docstring

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import rag
from minilm import MiniLMEmbedder

RAG_DIR = ROOT / "pilot" / "rag_dom"
OUT_FILE = RAG_DIR / "embeddings_minilm.npy"
CKPT_DIR = RAG_DIR / ".minilm_ckpt"
BATCH_SIZE = 8


def main():
    if OUT_FILE.exists():
        print(f"{OUT_FILE} already exists, nothing to do (delete it to force a rebuild).")
        return

    _, chunks = rag.load_index(RAG_DIR)
    texts = [c["text"] for c in chunks]
    n = len(texts)

    CKPT_DIR.mkdir(exist_ok=True)
    done_file = CKPT_DIR / "n_done.txt"
    arr_file = CKPT_DIR / "embs_partial.npy"
    n_done = int(done_file.read_text()) if done_file.exists() else 0
    results = list(np.load(arr_file)) if arr_file.exists() else []
    print(f"Embedding {n} chunks with MiniLM (resuming from {n_done})...")

    embedder = MiniLMEmbedder()
    i = n_done
    with tqdm(total=n, initial=n_done, unit="chunk", desc="MiniLM index") as pbar:
        while i < n:
            batch = texts[i:i + BATCH_SIZE]
            vecs = embedder.embed_batch(batch)
            results.extend(vecs)
            i += len(batch)
            np.save(arr_file, np.array(results, dtype=np.float32))
            done_file.write_text(str(i))
            pbar.update(len(batch))

    final = np.array(results, dtype=np.float32)
    final /= (np.linalg.norm(final, axis=1, keepdims=True) + 1e-9)  # unit-normalize, matches
    # rag.build_index()'s convention: rag.query() assumes index rows are already unit vectors
    # and only normalizes the query, so an un-normalized index silently corrupts the ranking
    # (scores get scaled by each chunk's raw vector norm instead of being pure cosine similarity).
    np.save(OUT_FILE, final)
    print(f"Saved {OUT_FILE}, shape {final.shape}")
    # clean up the checkpoint now that the real output exists
    arr_file.unlink(missing_ok=True)
    done_file.unlink(missing_ok=True)
    try:
        CKPT_DIR.rmdir()
    except OSError:
        pass  # leave it if anything else got dropped in there


if __name__ == "__main__":
    main()
