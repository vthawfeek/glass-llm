"""Stage 4/5: train a tiny GPT under one (tokenizer, seed) condition and record BPB.
Compute-matched (same steps/corpus across conditions). Idempotent per (tag, seed)."""
import argparse, csv, math, os, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bpe import BPETokenizer
from model import GPT

TIERS = {"cpu": dict(d_model=128, n_layer=4, n_head=4, ctx=128),
         "gpu": dict(d_model=256, n_layer=6, n_head=8, ctx=256)}


def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def truncated_text(text_path, max_bytes):
    text = Path(text_path).read_text(encoding="utf-8", errors="replace")
    if max_bytes and len(text.encode()) > max_bytes:
        text = text.encode("utf-8")[:max_bytes].decode("utf-8", "ignore")
    return text


def encode_cached(tok_name, tok, text_path, cache_dir, max_bytes):
    # cache key includes parent dir + byte cap so different corpora/caps never collide
    tag = str(max_bytes) if max_bytes else "full"
    parent = Path(text_path).parent.name
    cache = Path(cache_dir) / f"{tok_name}__{parent}__{Path(text_path).stem}__{tag}.npy"
    if cache.exists():
        return np.load(cache)
    ids = np.array(tok.encode(truncated_text(text_path, max_bytes)), dtype=np.int32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, ids)
    return ids


def get_batch(ids, ctx, bs, device):
    ix = torch.randint(0, len(ids) - ctx - 1, (bs,))
    x = torch.stack([torch.from_numpy(ids[i:i+ctx].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(ids[i+1:i+1+ctx].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def eval_bpb(model, test_ids, ctx, device, test_bytes):
    """Total test NLL (nats) over non-overlapping windows -> bits-per-byte."""
    model.eval()
    total_nats, n = 0.0, len(test_ids)
    for i in range(0, n - 1, ctx):
        chunk = test_ids[i:i+ctx+1]
        if len(chunk) < 2:
            break
        x = torch.from_numpy(chunk[:-1].astype(np.int64))[None].to(device)
        y = torch.from_numpy(chunk[1:].astype(np.int64))[None].to(device)
        logits, _ = model(x)
        ll = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum")
        total_nats += ll.item()
    bpb = (total_nats / math.log(2)) / test_bytes
    return bpb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)      # e.g. domain_4096
    ap.add_argument("--tag", required=True)            # condition label for runs.csv
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tier", default="cpu")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--max-train-bytes", type=int, default=800000)
    ap.add_argument("--max-test-bytes", type=int, default=0, help="cap test text (0=all)")
    ap.add_argument("--data", default="pilot/data")
    ap.add_argument("--out", default="pilot/results/runs.csv")
    a = ap.parse_args()

    device = "cuda" if (a.tier == "gpu" and torch.cuda.is_available()) else "cpu"
    cfg = TIERS[a.tier]; ctx = cfg["ctx"]
    ckpt = Path("pilot/models") / f"{a.tag}_seed{a.seed}.pt"

    # idempotency: skip if this (tag, seed) already recorded
    if Path(a.out).exists():
        for r in csv.DictReader(open(a.out, encoding="utf-8")):
            if r["tag"] == a.tag and int(r["seed"]) == a.seed:
                print(f"[train] {a.tag} seed{a.seed}: already in runs.csv, skip"); return

    set_seed(a.seed)
    tok = BPETokenizer.load(Path("pilot/tokenizers") / f"{a.tokenizer}.json")
    train_ids = encode_cached(a.tokenizer, tok, f"{a.data}/train.txt",
                              "pilot/models/_cache", a.max_train_bytes)
    test_ids = encode_cached(a.tokenizer, tok, f"{a.data}/test.txt",
                             "pilot/models/_cache", a.max_test_bytes)
    # BPB denominator must be the bytes of the SAME (possibly truncated) test text
    test_bytes = len(truncated_text(f"{a.data}/test.txt", a.max_test_bytes).encode("utf-8"))

    model = GPT(vocab_size=tok.vocab_size, **cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    nparams = model.num_params()
    print(f"[train] {a.tag} seed{a.seed} tok={a.tokenizer} vocab={tok.vocab_size} "
          f"params={nparams/1e6:.2f}M train_tokens={len(train_ids)} device={device}")

    model.train(); t0 = time.time(); tok_seen = 0
    for step in range(a.steps):
        x, y = get_batch(train_ids, ctx, a.batch_size, device)
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward(); opt.step()
        tok_seen += x.numel()
        if step % 100 == 0 or step == a.steps - 1:
            print(f"  step {step:4d}  loss {loss.item():.3f}")
    wall = time.time() - t0
    tps = tok_seen / wall

    bpb = eval_bpb(model, test_ids, ctx, device, test_bytes)
    torch.save({"cfg": cfg, "tok": a.tokenizer, "seed": a.seed}, ckpt)

    row = {"tag": a.tag, "tokenizer": a.tokenizer, "seed": a.seed, "tier": a.tier,
           "vocab": tok.vocab_size, "params_M": round(nparams/1e6, 3),
           "steps": a.steps, "final_loss": round(loss.item(), 4),
           "test_bpb": round(bpb, 4), "wall_s": round(wall, 1),
           "tokens_per_s": round(tps, 1), "device": device}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    new = not Path(a.out).exists()
    with open(a.out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"[train] {a.tag} seed{a.seed}: test_bpb={bpb:.4f}  wall={wall:.1f}s  tps={tps:.0f}")


if __name__ == "__main__":
    main()
