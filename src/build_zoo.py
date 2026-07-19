"""Build the model zoo for the v2 clinical-trials glass-box dashboard, idempotent.

Base LMs (12) = {domain=clinical trials, generic=English} x {low,med,high data} x {4,8 heads}
  - domain models: dom_4096 tokenizer, trained on ClinicalTrials.gov text (all specialties)
  - generic models: general_4096 tokenizer, trained on Project Gutenberg English
  - VOLUME varies training-data bytes with the tokenizer + steps held FIXED, so the
    low/med/high contrast is a clean data-volume experiment (not confounded by steps).
Biomarker heads (6) = a span-tagging head per DOMAIN base (the fine-tune mechanism).
RAG index (1)       = cosine index over real clinical-trial chunks, embedded by d_hi_4h.

Writes to models_v2/ so the original frozen models/ (old app + drafted content) stay intact.
Run in background:  python src/build_zoo.py
"""
import os, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bpe import BPETokenizer
from model import GPT
from train import set_seed, encode_cached, get_batch, eval_bpb, truncated_text
import biomarker as B
from glass import GlassModel
import rag

ROOT = Path(__file__).resolve().parents[1]
TOK = ROOT / "pilot" / "tokenizers"
# Output dirs are overridable so a retrain can build into a temp dir and be swapped in
# atomically (keeps a live app serving the previous zoo until the new one is ready).
MODELS = Path(os.environ.get("GLASS_MODELS", str(ROOT / "models_v2")))
CACHE = Path(os.environ.get("GLASS_CACHE", str(ROOT / "pilot" / "models" / "_cache_v2")))
RAGDIR = Path(os.environ.get("GLASS_RAG", str(ROOT / "pilot" / "rag_dom")))
DATA = {"d": ROOT / "pilot" / "data_dom", "g": ROOT / "pilot" / "data_gen"}
TOKENIZER = {"d": "dom_4096", "g": "general_4096"}
CFG = dict(d_model=128, n_layer=4, ctx=128)
VOL = {"lo": 1_000_000, "md": 8_000_000, "hi": 32_000_000}   # matched domain/generic corpora
STEPS = 4000                                                 # 4x compute for less-gibberish output


def log(m):
    print(m, flush=True)


def ensure_tokenizer():
    out = TOK / "dom_4096.json"
    if out.exists():
        log("[tok] dom_4096 exists, skip")
        return
    log("[tok] training dom_4096 (cap 3MB) ...")
    t0 = time.time()
    text = truncated_text(str(DATA["d"] / "train.txt"), 3_000_000)
    tok = BPETokenizer().train(text, vocab_size=4096)
    tok.save(out)
    log(f"[tok] dom_4096 saved  vocab={tok.vocab_size}  ({time.time()-t0:.0f}s)")


def train_base(tag, tokenizer, data_dir, max_bytes, n_head, seed=0):
    ckpt = MODELS / f"{tag}.pt"
    if ckpt.exists():
        log(f"[base] {tag}: exists, skip")
        return
    set_seed(seed)
    tok = BPETokenizer.load(TOK / f"{tokenizer}.json")
    train_ids = encode_cached(tokenizer, tok, f"{data_dir}/train.txt", str(CACHE), max_bytes)
    test_ids = encode_cached(tokenizer, tok, f"{data_dir}/test.txt", str(CACHE), 1_000_000)
    test_bytes = len(truncated_text(f"{data_dir}/test.txt", 1_000_000).encode("utf-8"))
    cfg = dict(CFG, n_head=n_head)
    model = GPT(vocab_size=tok.vocab_size, **cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    log(f"[base] {tag}: tok={tokenizer} vocab={tok.vocab_size} "
        f"params={model.num_params()/1e6:.2f}M cfg={cfg} bytes={max_bytes} "
        f"train_tokens={len(train_ids)}")
    model.train(); t0 = time.time()
    for step in range(STEPS):
        x, y = get_batch(train_ids, cfg["ctx"], 16, "cpu")
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 250 == 0 or step == STEPS - 1:
            log(f"  [base] {tag} step {step:4d}  loss {loss.item():.3f}")
    bpb = eval_bpb(model, test_ids, cfg["ctx"], "cpu", test_bytes)
    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "cfg": cfg, "tokenizer": tokenizer,
                "vocab_size": tok.vocab_size, "tag": tag, "steps": STEPS,
                "test_bpb": round(bpb, 4), "train_bytes": max_bytes,
                "wall_s": round(time.time() - t0, 1)}, ckpt)
    log(f"[base] saved {tag}  test_bpb={bpb:.4f}  ({time.time()-t0:.0f}s)")


def bio_training_texts(n=1400, seed=0):
    """Balanced-ish training lines: those with >=1 biomarker + a sample of lines without."""
    raw = (DATA["d"] / "train.txt").read_text(encoding="utf-8", errors="replace")
    pos, neg = [], []
    for rec in raw.split("<|endoftext|>"):
        for line in rec.split("\n"):
            line = line.strip()
            if not (20 <= len(line) <= 300):
                continue
            (pos if B.find_spans(line) else neg).append(line)
            if len(pos) >= n and len(neg) >= n // 2:
                break
        if len(pos) >= n and len(neg) >= n // 2:
            break
    rng = np.random.default_rng(seed)
    pos = pos[:n]
    neg = list(rng.choice(neg, size=min(len(neg), n // 2), replace=False)) if neg else []
    texts = pos + neg
    rng.shuffle(texts)
    log(f"[bio] training corpus: {len(pos)} biomarker lines + {len(neg)} plain lines")
    return texts


def train_bio_heads():
    texts = bio_training_texts()
    for vol in VOL:
        for h in (4, 8):
            tag = f"d_{vol}_{h}h"
            ckpt = MODELS / f"{tag}.pt"
            out = MODELS / f"{tag}_bio.pt"
            if not ckpt.exists() or out.exists():
                if out.exists():
                    log(f"[bio] {tag}: head exists, skip")
                continue
            base = GlassModel.load(ckpt, tokenizer_dir=str(TOK))
            B.train_head(base, texts, out, tag, steps=150, log=log)


def build_rag():
    if rag.exists(RAGDIR):
        log("[rag] index exists, skip"); return
    ckpt = MODELS / "d_hi_4h.pt"
    if not ckpt.exists():
        log("[rag] base d_hi_4h missing, skip"); return
    base = GlassModel.load(ckpt, tokenizer_dir=str(TOK))
    rag.build_index(base, n_trials=500, out_dir=RAGDIR, area="all")


def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    ensure_tokenizer()
    grid = [(dom, vol, h) for dom in ("d", "g") for vol in VOL for h in (4, 8)]
    t0 = time.time()
    for dom, vol, h in grid:
        train_base(f"{dom}_{vol}_{h}h", TOKENIZER[dom], DATA[dom], VOL[vol], h)
    log(f"[zoo] all base models done ({time.time()-t0:.0f}s)")
    train_bio_heads()
    build_rag()
    log("[zoo] COMPLETE")


if __name__ == "__main__":
    main()
