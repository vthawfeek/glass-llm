"""R4 mitigation: pre-download tiktoken + HF tokenizer vocabs into a repo-local cache so the
audit panel works OFFLINE (and on Hugging Face Spaces without runtime fetches). Run once online."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TIK_CACHE = ROOT / "assets" / "tiktoken_cache"
HF_CACHE = ROOT / "assets" / "hf"
TIK_CACHE.mkdir(parents=True, exist_ok=True)
HF_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["TIKTOKEN_CACHE_DIR"] = str(TIK_CACHE)

import tiktoken
for enc in ["cl100k_base", "o200k_base"]:
    tiktoken.get_encoding(enc).encode("warm")
    print("tiktoken cached:", enc)

from transformers import AutoTokenizer
for repo, short in [("gpt2", "gpt2"), ("bert-base-uncased", "bert")]:
    AutoTokenizer.from_pretrained(repo).save_pretrained(HF_CACHE / short)
    print("hf saved:", short)

print("done. cache dirs:", TIK_CACHE, HF_CACHE)
