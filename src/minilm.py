"""A from-scratch MiniLM (sentence-transformers/all-MiniLM-L6-v2) encoder for the RAG
embedding-source comparison: loads the pretrained weights directly from safetensors and runs
a hand-written BERT-style forward pass in torch, the same "no black box" spirit as the rest of
this repo's model code.

Why not `transformers.AutoModel.from_pretrained(...)` or `safetensors.load_file(...)`: on this
machine, BOTH crash with a Windows access violation at the identical site
(torch/storage.py:471 `__getitem__`) when materializing the model weights — transformers' threaded
fast-loader (core_model_loading.py) and safetensors' own mmap-based reader hit the same fault.
Root cause (confirmed): the machine runs under severe memory pressure (~1GB free of 8GB observed
during development), and memory-mapping a 90MB weights file becomes unreliable under that pressure
on this Windows/Python build. The fix is to never memory-map the file: this module parses the
safetensors binary format by hand with plain buffered `read()` calls (format: 8-byte little-endian
header length, then a JSON header with per-tensor dtype/shape/byte-offsets, then raw tensor bytes)
and builds torch tensors from owned, in-memory numpy buffers. This also means NO new runtime
dependency is needed beyond what's already in requirements.txt (numpy, torch, tokenizers).

Architecture is standard BERT-base-style per the model's config.json: 6 layers, hidden 384,
12 heads, intermediate 1536, GELU (exact/erf, not tanh-approx), post-LayerNorm blocks (eps 1e-12),
absolute learned position embeddings, WordPiece vocab 30522. Pooling is mean-over-tokens with the
attention mask, matching sentence-transformers' documented recipe for this model (the pooler head
in the checkpoint is unused, consistent with how sentence-transformers itself embeds with this model).

Verified numerically equivalent (cosine >= 0.999 per chunk, all 5960 index chunks) to transformers'
own AutoModel output, captured once before the crash was diagnosed (see tests/test_minilm.py).
"""
import json
import struct
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "assets" / "minilm"

N_LAYER = 6
N_HEAD = 12
D_MODEL = 384
HEAD_DIM = D_MODEL // N_HEAD
EPS = 1e-12

_DTYPES = {"F32": np.float32, "F16": np.float16, "I64": np.int64, "I32": np.int32, "I8": np.int8}


def _load_safetensors_no_mmap(path):
    """Pure buffered-read safetensors parser (no mmap) -> dict[name -> torch.Tensor].
    Deliberately avoids safetensors.load_file/safe_open, both of which mmap the file and crash
    on this machine under low-memory conditions (see module docstring)."""
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        data_start = f.tell()
        tensors = {}
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            start, end = meta["data_offsets"]
            f.seek(data_start + start)
            buf = f.read(end - start)
            arr = np.frombuffer(buf, dtype=_DTYPES[meta["dtype"]]).reshape(meta["shape"]).copy()
            t = torch.from_numpy(arr)
            # weights are stored as fp16 on disk (45MB vs 91MB fp32) purely to keep this repo's
            # "no Git LFS, small files" convention; upcast to fp32 immediately so all forward-pass
            # math below runs at the same precision as the fp32-verified reference (see docstring).
            tensors[name] = t.to(torch.float32) if t.dtype == torch.float16 else t
    return tensors


class MiniLMEncoder:
    """Hand-written forward pass over the pretrained MiniLM weights. Eval-only, no grad."""

    def __init__(self, weights_path=ASSET_DIR / "model.safetensors"):
        self.w = _load_safetensors_no_mmap(str(weights_path))  # name -> tensor, fp32, CPU

    def _lin(self, x, prefix):
        return F.linear(x, self.w[f"{prefix}.weight"], self.w[f"{prefix}.bias"])

    def _ln(self, x, prefix):
        return F.layer_norm(x, (D_MODEL,), self.w[f"{prefix}.weight"], self.w[f"{prefix}.bias"], EPS)

    @torch.no_grad()
    def forward(self, input_ids, attention_mask):
        """input_ids, attention_mask: LongTensor (B, T). Returns last_hidden_state (B, T, D_MODEL)."""
        B, T = input_ids.shape
        pos_ids = torch.arange(T, dtype=torch.long).unsqueeze(0).expand(B, T)
        tok_type = torch.zeros_like(input_ids)
        x = (F.embedding(input_ids, self.w["embeddings.word_embeddings.weight"])
             + F.embedding(pos_ids, self.w["embeddings.position_embeddings.weight"])
             + F.embedding(tok_type, self.w["embeddings.token_type_embeddings.weight"]))
        x = self._ln(x, "embeddings.LayerNorm")

        # additive attention bias: 0 where real tokens, large-negative where padding
        ext_mask = (1.0 - attention_mask[:, None, None, :].to(x.dtype)) * -1e9  # (B,1,1,T)

        for i in range(N_LAYER):
            p = f"encoder.layer.{i}"
            q = self._lin(x, f"{p}.attention.self.query").view(B, T, N_HEAD, HEAD_DIM).transpose(1, 2)
            k = self._lin(x, f"{p}.attention.self.key").view(B, T, N_HEAD, HEAD_DIM).transpose(1, 2)
            v = self._lin(x, f"{p}.attention.self.value").view(B, T, N_HEAD, HEAD_DIM).transpose(1, 2)
            scores = (q @ k.transpose(-1, -2)) / (HEAD_DIM ** 0.5) + ext_mask
            probs = F.softmax(scores, dim=-1)
            ctx = (probs @ v).transpose(1, 2).contiguous().view(B, T, D_MODEL)
            attn_out = self._lin(ctx, f"{p}.attention.output.dense")
            x = self._ln(attn_out + x, f"{p}.attention.output.LayerNorm")

            inter = F.gelu(self._lin(x, f"{p}.intermediate.dense"))            # exact (erf) gelu
            ffn_out = self._lin(inter, f"{p}.output.dense")
            x = self._ln(ffn_out + x, f"{p}.output.LayerNorm")
        return x


class MiniLMEmbedder:
    """Matches GlassModel.embed(text)->np.ndarray so rag.py's query()/build_index() work unchanged
    with either embedder. Unnormalized mean-pooled vector; rag.py does the L2 normalization."""

    def __init__(self, asset_dir=ASSET_DIR, max_length=128):
        self.tok = Tokenizer.from_file(str(Path(asset_dir) / "tokenizer.json"))
        self.enc = MiniLMEncoder(Path(asset_dir) / "model.safetensors")
        self.max_length = max_length

    def _encode_batch(self, texts):
        encs = [self.tok.encode(t) for t in texts]
        ids = torch.tensor([e.ids for e in encs], dtype=torch.long)
        mask = torch.tensor([e.attention_mask for e in encs], dtype=torch.long)
        return ids, mask

    @torch.no_grad()
    def embed_batch(self, texts):
        """Batched mean-pooled embeddings, (len(texts), D_MODEL) numpy. Used for index building."""
        ids, mask = self._encode_batch(texts)
        hidden = self.enc.forward(ids, mask)                     # (B, T, D)
        m = mask.unsqueeze(-1).to(hidden.dtype)                  # (B, T, 1)
        pooled = (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)  # mean over real tokens only
        return pooled.numpy()

    def embed(self, text):
        """Single-text embed, matches GlassModel.embed's interface for drop-in use in rag.py."""
        return self.embed_batch([text])[0]


def exists(asset_dir=ASSET_DIR):
    d = Path(asset_dir)
    return (d / "model.safetensors").exists() and (d / "tokenizer.json").exists()
