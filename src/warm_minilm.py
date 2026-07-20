"""Pre-download the MiniLM weights + tokenizer into a repo-local cache (assets/minilm/) so the
RAG embedding-source comparison works OFFLINE at deploy time, same rationale as warm_tokenizers.py
(R4). Uses huggingface_hub directly (NOT transformers.AutoModel, which segfaults on this machine
when loading cached weights, see src/minilm.py's docstring). Run once online."""
from pathlib import Path
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "minilm"
OUT.mkdir(parents=True, exist_ok=True)

REPO = "sentence-transformers/all-MiniLM-L6-v2"
FILES = ["model.safetensors", "tokenizer.json", "config.json"]

for fname in FILES:
    p = hf_hub_download(repo_id=REPO, filename=fname)
    (OUT / fname).write_bytes(Path(p).read_bytes())
    print("cached:", fname, "->", OUT / fname)

print("done. assets dir:", OUT)
