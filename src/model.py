"""Minimal decoder-only GPT (nanoGPT-style, compact) written for the Glass-LLM pilot."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, ctx):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head, self.d_model = n_head, d_model
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.register_buffer("mask", torch.tril(torch.ones(ctx, ctx)).view(1, 1, ctx, ctx))
        self.store_attn = False   # set True only when tracing (avoids training overhead)
        self.last_attn = None     # (B, n_head, T, T) softmax weights from the last forward

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.d_model, dim=2)
        h = self.n_head
        q = q.view(B, T, h, C // h).transpose(1, 2)
        k = k.view(B, T, h, C // h).transpose(1, 2)
        v = v.view(B, T, h, C // h).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(C // h)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        if self.store_attn:
            self.last_attn = att.detach()
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, d_model, n_head, ctx):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, ctx)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                 nn.Linear(4 * d_model, d_model))
        self.store_contrib = False   # set True only when tracing (avoids training overhead)
        self.last_attn_out = None    # (B, T, C) attention sublayer's write into the residual stream
        self.last_ffn_out = None     # (B, T, C) feed-forward sublayer's write into the residual stream

    def forward(self, x):
        a = self.attn(self.ln1(x))       # attention sublayer output
        x = x + a                        # residual stream after attention
        f = self.mlp(self.ln2(x))        # feed-forward sublayer output
        x = x + f                        # residual stream after feed-forward
        if self.store_contrib:
            self.last_attn_out = a.detach()
            self.last_ffn_out = f.detach()
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_layer=4, n_head=4, ctx=128):
        super().__init__()
        self.ctx = ctx
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(ctx, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_head, ctx) for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok.weight   # weight tying
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx, targets=None, return_hidden=False):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None]
        hiddens = [x] if return_hidden else None      # residual stream after embedding
        for b in self.blocks:
            x = b(x)
            if return_hidden:
                hiddens.append(x)
        logits = self.head(self.lnf(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        if return_hidden:
            return logits, loss, hiddens
        return logits, loss

    def set_trace(self, flag):
        """Toggle attention-weight and per-sublayer contribution capture on every block
        (tracing only, no effect on the forward math, so training/generation are untouched)."""
        for b in self.blocks:
            b.attn.store_attn = flag
            b.store_contrib = flag

    def attentions(self):
        """Per-layer attention from the most recent traced forward: list of (B, n_head, T, T)."""
        return [b.attn.last_attn for b in self.blocks]

    def contributions(self):
        """Per-layer sublayer writes from the most recent traced forward:
        (attn_outs, ffn_outs), each a list of length n_layer with (B, T, d_model) tensors."""
        return ([b.last_attn_out for b in self.blocks],
                [b.last_ffn_out for b in self.blocks])

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
