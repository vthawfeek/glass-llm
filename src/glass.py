"""The trace(prompt) API, the single interface every dashboard panel reads from.
Returns tokens, embeddings, per-layer/head attention, hidden states, and next-token logits
in one structured object, plus a transparent generate() for the generation panel."""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bpe import BPETokenizer
from model import GPT

WATERMARK = "SYNTHETIC: interpretability demo, not medical information"


class Trace:
    """Structured result of one forward pass. All arrays are numpy for easy plotting."""
    def __init__(self, prompt, token_ids, token_texts, embeddings, hiddens, attentions,
                 logits, topk, attn_contrib=None, ffn_contrib=None):
        self.prompt = prompt
        self.token_ids = token_ids            # list[int]
        self.token_texts = token_texts        # list[str] (per-token display strings)
        self.embeddings = embeddings          # (T, d_model) token embeddings
        self.hiddens = hiddens                # list length n_layer+1 of (T, d_model)
        self.attentions = attentions          # list length n_layer of (n_head, T, T)
        self.logits = logits                  # (vocab,) next-token logits at last position
        self.topk = topk                      # list[(token_text, prob, token_id)]
        self.attn_contrib = attn_contrib      # list length n_layer of (T, d_model) attention writes
        self.ffn_contrib = ffn_contrib        # list length n_layer of (T, d_model) feed-forward writes

    @property
    def n_tokens(self):
        return len(self.token_ids)


class GlassModel:
    def __init__(self, model, tokenizer, meta=None):
        self.model = model.eval()
        self.tok = tokenizer
        self.meta = meta or {}
        self.ctx = model.ctx

    @classmethod
    def load(cls, ckpt_path, tokenizer_dir="pilot/tokenizers"):
        ck = torch.load(ckpt_path, map_location="cpu")
        tok = BPETokenizer.load(Path(tokenizer_dir) / f"{ck['tokenizer']}.json")
        model = GPT(vocab_size=ck["vocab_size"], **ck["cfg"])
        model.load_state_dict(ck["state_dict"])
        return cls(model, tok, meta=ck)

    # ---- display helpers ----
    def token_text(self, tid):
        """Human-readable piece for one token id; leading space shown as '␣'."""
        s = self.tok.vocab[tid].decode("utf-8", "replace")
        return s.replace(" ", "␣")

    def _encode_capped(self, prompt):
        ids = self.tok.encode(prompt) or [0]      # never empty
        return ids[-self.ctx:]                     # keep last ctx tokens

    # ---- embedding-space helpers (embeddings panel) ----
    def embedding_matrix(self):
        return self.model.tok.weight.detach().numpy()   # (vocab, d_model)

    def nearest(self, tid, k=10):
        """Cosine nearest neighbors of a token in embedding space."""
        E = self.embedding_matrix()
        v = E[tid]
        sims = (E @ v) / (np.linalg.norm(E, axis=1) * np.linalg.norm(v) + 1e-9)
        order = np.argsort(-sims)
        out = [(self.token_text(int(i)), float(sims[i]), int(i)) for i in order if i != tid]
        return out[:k]

    @torch.no_grad()
    def embed(self, text):
        """Mean-pooled final hidden state, a sentence vector for retrieval (RAG index)."""
        idx = torch.tensor([self._encode_capped(text)], dtype=torch.long)
        _, _, hiddens = self.model(idx, return_hidden=True)
        return hiddens[-1][0].mean(0).numpy()

    # ---- the core API ----
    @torch.no_grad()
    def trace(self, prompt, topk=10):
        ids = self._encode_capped(prompt)
        idx = torch.tensor([ids], dtype=torch.long)
        self.model.set_trace(True)
        logits, _, hiddens = self.model(idx, return_hidden=True)
        attns = self.model.attentions()
        attn_outs, ffn_outs = self.model.contributions()
        self.model.set_trace(False)

        last = logits[0, -1]                       # next-token logits
        probs = F.softmax(last, dim=-1)
        p, i = torch.topk(probs, min(topk, probs.numel()))
        topk_list = [(self.token_text(int(t)), float(pr), int(t)) for pr, t in zip(p, i)]

        return Trace(
            prompt=prompt,
            token_ids=ids,
            token_texts=[self.token_text(t) for t in ids],
            embeddings=self.model.tok.weight[ids].detach().numpy(),
            hiddens=[h[0].detach().numpy() for h in hiddens],
            attentions=[a[0].numpy() for a in attns],       # drop batch dim -> (n_head, T, T)
            logits=last.numpy(),
            topk=topk_list,
            attn_contrib=[a[0].numpy() for a in attn_outs],  # per-layer (T, d_model) attention write
            ffn_contrib=[f[0].numpy() for f in ffn_outs],    # per-layer (T, d_model) feed-forward write
        )

    @torch.no_grad()
    def dist(self, logits, temperature=1.0, top_k=0, show=10):
        """Next-token display distribution AFTER temperature + top-k, mirroring generate().

        `logits` is the raw (vocab,) logit vector (e.g. `trace.logits`). Returns
        list[(token_text, prob)] for the top-`show` candidates, high→low. This is what the
        model actually samples from, so the generation panel's bars respond to the sliders
        (low temperature sharpens toward one token; a small top-k zeroes out the tail)."""
        lg = torch.as_tensor(np.asarray(logits, dtype=np.float32)).clone()
        lg = lg / max(float(temperature), 1e-6)
        if top_k:
            v, _ = torch.topk(lg, min(int(top_k), lg.numel()))
            lg[lg < v[-1]] = -float("inf")
        probs = F.softmax(lg, dim=-1)
        pv, pi = torch.topk(probs, min(int(show), probs.numel()))
        return [(self.token_text(int(t)), float(pr)) for pr, t in zip(pv, pi)]

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=40, temperature=0.8, top_k=40, seed=0, show=8):
        """Autoregressive generation. Returns the full text plus, for every step, the candidate
        distribution (AFTER temperature + top-k) with the sampled token flagged, so the
        dashboard can replay the loop one token at a time. Sampling is unchanged, so `text` is
        still deterministic for a given seed.

        Each `steps` entry is a dict: {token, prob, dist}, where dist is a list of
        (token_text, prob, is_sampled) for the shown candidates (the sampled token is always
        included, even if it fell outside the top-`show`)."""
        torch.manual_seed(seed)
        ids = self._encode_capped(prompt)
        steps = []
        for _ in range(max_new_tokens):
            idx = torch.tensor([ids[-self.ctx:]], dtype=torch.long)
            logits, _ = self.model(idx)
            logits = logits[0, -1] / max(temperature, 1e-6)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.numel()))
                logits[logits < v[-1]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            nxt = int(torch.multinomial(probs, 1))
            pv, pi = torch.topk(probs, min(show, probs.numel()))
            cand = [(self.token_text(int(t)), float(pr), int(t) == nxt) for pr, t in zip(pv, pi)]
            if not any(picked for *_, picked in cand):     # keep the sampled token visible
                cand.append((self.token_text(nxt), float(probs[nxt]), True))
            steps.append({"token": self.token_text(nxt), "prob": float(probs[nxt]), "dist": cand})
            ids.append(nxt)
        text = self.tok.decode(ids)
        return {"text": text, "steps": steps, "watermark": WATERMARK}
