"""Biomarker span-tagging, the dashboard's 'fine-tune for a task' mechanism.

Honest framing (see README): a 1.3M from-scratch model CANNOT do real structured
biomarker extraction, the state of the art needs ~7B models (Alkhoury et al., npj
Digital Medicine 2025). What we build here is a *mechanism demo*: distant supervision
from an oncology gene/biomarker lexicon labels each token, and a small linear head on
top of the FROZEN base model learns to TAG biomarker mentions. You watch fine-tuning
add a task head and light up biomarker tokens, you must NOT read it as an extractor.
The head imitates the lexicon; it will miss novel terms and mishandle negation.
"""
import json
from pathlib import Path
import numpy as np
import regex as re
import torch
import torch.nn as nn

# ---- distant-supervision lexicon (oncology) -------------------------------------------
# Gene symbols are matched CASE-SENSITIVELY (so "MET"/"KIT"/"ER" don't hit common words).
GENES = [
    "EGFR", "ALK", "ROS1", "BRAF", "KRAS", "NRAS", "HRAS", "ERBB2", "HER2", "MET", "RET",
    "NTRK1", "NTRK2", "NTRK3", "NTRK", "PIK3CA", "PTEN", "TP53", "BRCA1", "BRCA2", "KIT",
    "PDGFRA", "FLT3", "IDH1", "IDH2", "FGFR1", "FGFR2", "FGFR3", "FGFR", "CDK4", "CDK6",
    "ESR1", "ABL1", "BCR", "JAK2", "MYC", "VHL", "SMAD4", "APC", "CDKN2A", "MDM2", "NF1",
    "RB1", "STK11", "KEAP1", "ERBB3", "MAP2K1", "AKT1", "MTOR", "CD274", "PDCD1", "MSH2",
    "MSH6", "MLH1", "PMS2", "ATM", "PALB2", "CDH1", "GNAQ", "GNA11", "DNMT3A", "TET2",
    "NPM1", "CEBPA", "SF3B1", "CALR", "MPL", "EZH2", "BCL2", "CCND1", "AURKA", "POLE",
    "CTNNB1", "GATA3", "MAP2K2", "SMO", "PTCH1", "WT1", "ARID1A", "SETD2", "BAP1",
]
# Multi-token biomarkers / immuno-oncology markers.
BIOMARKERS = [
    "PD-L1", "PD-1", "PDL1", "MSI-H", "MSI-High", "MSI", "dMMR", "pMMR", "MMR", "TMB-H",
    "TMB", "HRD", "Ki-67", "Ki67", "BCR-ABL", "HER2/neu", "microsatellite instability",
    "mismatch repair", "tumor mutational burden", "homologous recombination deficiency",
]
# Specific alteration codes, always a biomarker wherever they appear.
VARIANT_CODES = [
    "V600E", "V600K", "T790M", "L858R", "C797S", "G12C", "G12D", "G12V", "G13D", "Q61K",
    "Q61R", "G719X", "S768I", "exon 19 deletion", "exon 20 insertion", "exon 14 skipping",
    "del19", "T315I", "D816V", "V617F",
]
# Qualifiers only count as biomarkers when they trail a gene/marker (context-gated).
QUALIFIERS = [
    "mutation", "mutations", "mutant", "mutated", "amplification", "amplified", "fusion",
    "fusions", "rearrangement", "rearranged", "translocation", "positive", "negative",
    "overexpression", "overexpressed", "expression", "deletion", "insertion", "wild-type",
    "wildtype", "wild type", "methylation", "alteration", "alterations",
]

_ALWAYS = sorted(set(GENES + BIOMARKERS + VARIANT_CODES), key=len, reverse=True)
_ALWAYS_RE = re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(re.escape(t) for t in _ALWAYS) +
                        r")(?![A-Za-z0-9])")
_QUAL_RE = re.compile(r"[\s\-]{0,2}(?:" + "|".join(re.escape(q) for q in
                      sorted(QUALIFIERS, key=len, reverse=True)) + r")", re.IGNORECASE)


def find_spans(text):
    """Return merged character spans [start, end) of biomarker mentions in `text`."""
    spans = []
    for m in _ALWAYS_RE.finditer(text):
        spans.append([m.start(), m.end()])
        qm = _QUAL_RE.match(text, m.end())          # extend onto a trailing qualifier
        if qm and qm.end() > qm.start():
            spans.append([m.end(), qm.end()])
    if not spans:
        return []
    spans.sort()
    merged = [spans[0]]
    for s, e in spans[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [tuple(x) for x in merged]


def label_ids(tok, ids):
    """Binary biomarker label per BPE token, derived from the EXACT token ids given.
    Reconstructs the text those ids cover (robust to context-capping/truncation) and
    aligns biomarker spans by UTF-8 byte offset. Oncology text is ASCII so byte==char."""
    piece_bytes = [tok.vocab[i] for i in ids]
    text = b"".join(piece_bytes).decode("utf-8", "replace")
    char_spans = find_spans(text)
    byte_spans = [(len(text[:s].encode("utf-8")), len(text[:e].encode("utf-8")))
                  for s, e in char_spans]
    labels, off = [], 0
    for pb in piece_bytes:
        t0, t1 = off, off + len(pb)
        labels.append(1 if any(t0 < b1 and b0 < t1 for b0, b1 in byte_spans) else 0)
        off = t1
    return labels


def label_tokens(tok, text):
    """Convenience: encode `text` and label each token. Returns (ids, labels)."""
    ids = tok.encode(text)
    return ids, label_ids(tok, ids)


# ---- the task head (a linear probe on the frozen base model) --------------------------
class SpanHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.lin = nn.Linear(d_model, 2)

    def forward(self, feats):
        return self.lin(feats)


@torch.no_grad()
def _features(base, ids):
    """Frozen, layer-normalized final hidden states for token ids -> (T, d_model) tensor."""
    idx = torch.tensor([ids], dtype=torch.long)
    _, _, hiddens = base.model(idx, return_hidden=True)
    return base.model.lnf(hiddens[-1])[0]            # (T, d_model)


def train_head(base, texts, out_path, tag, steps=300, lr=5e-3, seed=0, log=print):
    """Train a biomarker-tagging head on FROZEN base features. Idempotent (skips if exists)."""
    out_path = Path(out_path)
    if out_path.exists():
        log(f"[bio] {tag}: head exists, skip")
        return
    torch.manual_seed(seed)
    d_model = base.model.tok.weight.shape[1]
    # pre-featurize once (base is frozen), keep only chunks that fit the context window
    samples = []
    pos = 0
    for t in texts:
        ids = base._encode_capped(t)
        if len(ids) < 2:
            continue
        labs = label_ids(base.tok, ids)              # labels derived from the exact ids used
        y = torch.tensor(labs, dtype=torch.long)
        pos += int(y.sum())
        samples.append((_features(base, ids), y))
    if not samples:
        log(f"[bio] {tag}: no usable samples"); return
    total = sum(len(y) for _, y in samples)
    w1 = total / max(1, pos)                          # up-weight the rare biomarker class
    weight = torch.tensor([1.0, min(w1, 30.0)])
    head = SpanHead(d_model)
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    order = np.arange(len(samples))
    for step in range(steps):
        rng.shuffle(order)
        tot = 0.0
        for i in order:
            feats, y = samples[i]
            logits = head(feats)
            loss = nn.functional.cross_entropy(logits, y, weight=weight)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        if step % 100 == 0 or step == steps - 1:
            log(f"  [bio] {tag} step {step:3d}  loss {tot/len(samples):.3f}  "
                f"(pos {pos}/{total}, w+={weight[1]:.1f})")
    torch.save({"head": head.state_dict(), "d_model": d_model, "base": tag,
                "pos": pos, "total": total}, out_path)
    log(f"[bio] saved {out_path}  (biomarker tokens {pos}/{total})")


def load_head(path):
    ck = torch.load(path, map_location="cpu")
    head = SpanHead(ck["d_model"])
    head.load_state_dict(ck["head"])
    head.eval()
    return head, ck


@torch.no_grad()
def tag(base, head, prompt):
    """Per-token tagging aligned to base.trace(prompt). Returns list of
    (piece, prob_biomarker, is_biomarker) plus the token ids used."""
    ids = base._encode_capped(prompt)
    feats = _features(base, ids)
    probs = torch.softmax(head(feats), dim=-1)[:, 1].numpy()
    out = [(base.token_text(t), float(p), bool(p >= 0.5)) for t, p in zip(ids, probs)]
    return out, ids
