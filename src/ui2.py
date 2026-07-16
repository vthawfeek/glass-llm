"""UI helpers for the v2 clinical-trials glass-box dashboard.

Adds two things the v1 helpers don't have:
  * architecture_figure_v2 — a decoder schematic driven by ALL FIVE knobs (domain, volume,
    heads, fine-tune head, RAG), lighting up whatever the current selection changes.
  * pipeline_sankey — the literal text -> tokens -> vectors -> (model) -> re-tokenised ->
    text flow, with REAL token pieces and REAL embedding values on the nodes.
Everything else (plain-language glossary, EXPLAIN cards, dataset card, journey CSS) is
reused from ui.py so the two dashboards share one look and voice.
"""
import html
import numpy as np
import plotly.graph_objects as go
import ui  # reuse GLOSSARY / EXPLAIN / dataset_card_html / journey helpers / step CSS

# ---- knob metadata: layman labels + plain-English tooltips (help=) --------------------
KNOBS = {
    "domain": {
        "label": "1 · Training data",
        "options": ["🏥 Domain — clinical trials", "📖 Generic — English books"],
        "help": "What text the model learned from. Domain = ClinicalTrials.gov trials, all "
                "specialties (medical words are familiar). Generic = ordinary English novels "
                "(medical words look foreign). Watch tokenization, embeddings and generation change.",
    },
    "volume": {
        "label": "2 · Training-data volume",
        "options": ["Low (1 MB)", "Medium (8 MB)", "High (32 MB)"],
        "help": "How MUCH text the model trained on — same tokenizer and same 4,000 training "
                "steps, only the amount of data changes. Low = 1 MB (~0.25M tokens), "
                "Medium = 8 MB (~2M tokens), High = 32 MB (~8M tokens). Too little data → the "
                "model memorises and predicts held-out text worse (higher bits-per-byte).",
    },
    "heads": {
        "label": "3 · Attention heads",
        "options": ["4 heads", "8 heads"],
        "help": "How many parallel 'lenses' each attention layer uses. More heads track more "
                "relationships between tokens at once. Watch the Attention panel gain more heads.",
    },
    "finetune": {
        "label": "4 · Fine-tuned (biomarker tagging)",
        "options": ["No", "Yes"],
        "help": "Adds a small task head on top of the FROZEN model, trained to TAG biomarker "
                "mentions (EGFR, PD-L1, BRAF V600E…). A mechanism demo of fine-tuning — NOT a real "
                "extractor (a 1.3M model can't structure biomarkers; that needs ~7B models).",
    },
    "rag": {
        "label": "5 · Retrieval (RAG)",
        "options": ["No", "Yes"],
        "help": "Retrieval-Augmented Generation: before generating, fetch real trial passages "
                "and put them in front of the model. Watch invented ('hallucinated') text on the "
                "left become grounded in real retrieved trials on the right.",
    },
}


def tag_for(domain_generic, volume, heads):
    """Map the three model-shaping knobs to a checkpoint tag in models_v2/.
    Robust to the volume labels carrying a size suffix, e.g. 'Low (0.5 MB)'."""
    dom = "d" if domain_generic.startswith("🏥") else "g"
    vol = "lo" if volume.startswith("Low") else ("md" if volume.startswith("Med") else "hi")
    h = 8 if heads.startswith("8") else 4
    return f"{dom}_{vol}_{h}h"


# ---- per-step math for the architecture figure (formula on the box + plain-English hover) ----
ARCH_MATH = {
    "input":     ("", "The raw text you typed, before the model touches it."),
    "tokenizer": ("id → E[id] ∈ ℝ¹²⁸",
                  "Each text piece is matched to a fixed vocabulary to get an id, and each id "
                  "looks up a learned 128-number vector (its embedding)."),
    "pos":       ("x = E[tok] + P[pos]",
                  "A position vector is added so the model knows token order — otherwise "
                  "'A then B' and 'B then A' would look the same."),
    "attention": ("softmax(QKᵀ/√d)·V",
                  "Each token makes a Query and matches earlier tokens' Keys; the best matches "
                  "pass on their Values. 'Masked' = it can't look at future tokens."),
    "an1":       ("LayerNorm(x + sublayer)",
                  "Add the sublayer's output back onto its input (a residual shortcut) and "
                  "normalize — this keeps training stable."),
    "ff":        ("W₂·GELU(W₁x)",
                  "A small 2-layer network reshapes each token's vector on its own, storing "
                  "patterns the model learned from data."),
    "an2":       ("LayerNorm(x + sublayer)",
                  "Another residual + normalize, closing the transformer block. The block "
                  "repeats ×4 (the 4 layers)."),
    "linear":    ("z = x · Eᵀ",
                  "The final vector is projected back onto the whole vocabulary, giving a raw "
                  "score (logit) for every possible next token."),
    "softmax":   ("p = softmax(z / T)",
                  "Scores become probabilities that sum to 1; temperature T sharpens (low) or "
                  "flattens (high) the odds."),
    "out":       ("next ~ p",
                  "One token is sampled from those probabilities, appended to the text, and the "
                  "whole loop repeats to generate more."),
}

# ordered (name, formula, plain) for the app's 'math at each step' table
ARCH_TABLE = [
    ("Tokenize", "text → ids", "Split the text into vocabulary pieces (tokens) and map each to an id."),
    ("Embed", "id → E[id] ∈ ℝ¹²⁸", "Each id looks up a learned 128-number vector."),
    ("Add position", "x = E[tok] + P[pos]", "Add a position vector so token order matters."),
    ("Attention", "softmax(QKᵀ/√d)·V", "Tokens query earlier tokens' keys and mix their values (masked to the past)."),
    ("Add & Norm", "LayerNorm(x + sublayer)", "Residual shortcut + normalization; keeps training stable. Block repeats ×4."),
    ("Feed Forward", "W₂·GELU(W₁x)", "A 2-layer network reshapes each token vector independently."),
    ("LM head", "z = x · Eᵀ", "Project the final vector onto the vocabulary → a score (logit) per next token."),
    ("Softmax", "p = softmax(z / T)", "Turn scores into probabilities; temperature T sharpens or flattens them."),
    ("Sample", "next ~ p", "Pick the next token, append it, and repeat → output text."),
]


_C = {"pink": "rgba(209,73,120,0.18)", "orange": "rgba(199,125,52,0.20)",
      "blue": "rgba(91,107,191,0.18)", "green": "rgba(106,141,63,0.20)",
      "grey": "rgba(150,150,150,0.12)", "hot": "rgba(209,73,91,0.24)"}
_HOT_LINE, _COLD_LINE = "#d1495b", "#5f7f88"


def architecture_figure_v2(is_generic, volume, n_head, fine_tuned, rag, vocab, height=560):
    """Decoder-only schematic reflecting the knob selection, with the math for each step shown
    on its box and a plain-English explanation on hover (the ⓘ markers). Red = a component this
    selection changes from the default (domain · High · 4 heads · no FT · no RAG)."""
    dom_txt = "English books" if is_generic else "Clinical trials"
    tok_txt = "general_4096" if is_generic else "dom_4096"
    fig = go.Figure()
    boxes = [
        ("input",     f"Input text · {dom_txt} · {volume}", _C["grey"], 0.12, 0.92),
        ("tokenizer", f"Token Embedding · {tok_txt} · vocab {vocab:,}", _C["pink"], 1.05, 1.98),
        ("pos",       "+ Positional Encoding", _C["grey"], 2.1, 2.85),
        ("attention", f"Masked Multi-Head Attention · {n_head} heads", _C["orange"], 3.0, 3.95),
        ("an1",       "Add & Norm", _C["grey"], 4.05, 4.55),
        ("ff",        "Feed Forward", _C["blue"], 4.68, 5.45),
        ("an2",       "Add & Norm", _C["grey"], 5.55, 6.05),
        ("linear",    "Linear (LM head)", _C["grey"], 6.35, 7.05),
        ("softmax",   "Softmax", _C["green"], 7.15, 7.8),
        ("out",       "Next-token probabilities", _C["grey"], 7.9, 8.65),
    ]
    hot = set()
    if is_generic:
        hot.add("tokenizer")           # different tokenizer + learned weights
    if n_head == 8:
        hot.add("attention")
    x0, x1 = 0.7, 4.3
    hx, hy, htext = [], [], []
    for key, label, fill, y0, y1 in boxes:
        h = key in hot
        ymid = (y0 + y1) / 2
        formula, plain = ARCH_MATH[key]
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                      line=dict(color=_HOT_LINE if h else _COLD_LINE, width=3 if h else 1),
                      fillcolor=_C["hot"] if h else fill)
        if formula:
            fig.add_annotation(x=(x0 + x1) / 2, y=ymid + 0.15, text=label, showarrow=False,
                               font=dict(size=10.5, color="#f2f2f2"))
            fig.add_annotation(x=(x0 + x1) / 2, y=ymid - 0.16, text=formula, showarrow=False,
                               font=dict(size=9.5, color="#bfe3e3"))
        else:
            fig.add_annotation(x=(x0 + x1) / 2, y=ymid, text=label, showarrow=False,
                               font=dict(size=10.5, color="#f2f2f2"))
        hx.append(x1 - 0.13); hy.append(ymid)
        htext.append(f"<b>{label}</b><br>{formula + '<br>' if formula else ''}{plain}")
    for i in range(len(boxes) - 1):
        fig.add_annotation(x=(x0 + x1) / 2, y=boxes[i + 1][3], ax=(x0 + x1) / 2, ay=boxes[i][4],
                           xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                           arrowhead=2, arrowsize=1, arrowwidth=1, arrowcolor="#6a8")
    fig.add_shape(type="rect", x0=x1 + 0.15, x1=x1 + 0.25, y0=3.0, y1=6.05,
                  line=dict(color="#7aa", width=1), fillcolor="rgba(0,0,0,0)")
    fig.add_annotation(x=x1 + 0.62, y=4.5, text="× 4<br>layers", showarrow=False,
                       font=dict(size=11, color="#9cc"))
    # RAG feeds retrieved context INTO the input (external box, left of input)
    if rag:
        fig.add_shape(type="rect", x0=-1.15, x1=0.4, y0=0.12, y1=0.92, line=dict(color=_HOT_LINE, width=2),
                      fillcolor=_C["hot"])
        fig.add_annotation(x=-0.38, y=0.52, text="Retrieved<br>trials (RAG)", showarrow=False,
                           font=dict(size=10, color="#f2f2f2"))
        fig.add_annotation(x=0.55, y=0.52, ax=0.4, ay=0.52, xref="x", yref="y", axref="x",
                           ayref="y", showarrow=True, arrowhead=2, arrowwidth=1.5, arrowcolor=_HOT_LINE)
    # Fine-tune biomarker head branches off the residual stream (external box, right)
    if fine_tuned:
        fig.add_shape(type="rect", x0=x1 + 0.9, x1=x1 + 2.5, y0=4.05, y1=4.95,
                      line=dict(color=_HOT_LINE, width=2), fillcolor=_C["hot"])
        fig.add_annotation(x=x1 + 1.7, y=4.5, text="Biomarker head<br>(fine-tuned)", showarrow=False,
                           font=dict(size=10, color="#f2f2f2"))
        fig.add_annotation(x=x1 + 0.9, y=4.5, ax=x1, ay=4.5, xref="x", yref="y", axref="x",
                           ayref="y", showarrow=True, arrowhead=2, arrowwidth=1.5, arrowcolor=_HOT_LINE)
    # ⓘ hover markers carrying the plain-English explanation of each step's math
    fig.add_trace(go.Scatter(x=hx, y=hy, mode="markers+text", text=["ⓘ"] * len(hx),
                  textfont=dict(size=11, color="#9ad"), marker=dict(size=18, color="rgba(0,0,0,0)"),
                  hovertext=htext, hoverinfo="text",
                  hoverlabel=dict(bgcolor="#0e2a2a", bordercolor="#2e8b8b",
                                  font=dict(size=12, color="#eaeaea")),
                  showlegend=False, cliponaxis=False))
    fig.update_xaxes(visible=False, range=[-1.3, 7.2])
    fig.update_yaxes(visible=False, range=[0, 8.85])
    fig.update_layout(height=height, margin=dict(l=4, r=4, t=6, b=4),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def pipeline_sankey(input_text, in_pieces, emb0, attn_last, topk, out_text, height=470, max_in=8):
    """Literal pipeline as a Sankey with REAL data on the nodes:
        input text → tokens → vectors(e₀) → [attention + FFN] → predicted tokens(prob) → output text.
    input_text = the raw prompt (first node); emb0 = first embedding component per input token;
    attn_last = last-token attention over inputs.
    """
    attn_last = np.asarray(attn_last, dtype=float)
    order = sorted(np.argsort(-attn_last)[:max_in])          # keep most-attended inputs, in order
    toks = [in_pieces[i] for i in order]
    e0 = [float(emb0[i]) for i in order]
    aw = [float(attn_last[i]) for i in order]
    n = len(toks)

    labels, xs, ys, colors = [], [], [], []

    def col(items, x, color):
        base = len(labels)
        m = len(items)
        yy = list(np.linspace(0.06, 0.94, m)) if m > 1 else [0.5]
        for lab in items:
            labels.append(lab); xs.append(x); colors.append(color)
        ys.extend(yy)
        return base

    inp_i = col([f"“{(input_text or '…').strip()[:20]}…”"], 0.01, "#8a6bbf")   # input text (1 node)
    tok_i = col(toks, 0.20, "#2e8b8b")                                          # tokenise
    vec_i = col([f"e₀={v:+.2f}" for v in e0], 0.40, "#3f8f9e")                  # vectorise
    hub_i = col(["attention + FFN"], 0.58, "#c77d34")                           # transformer
    prd_i = col([f"{t}·{p:.0%}" for t, p in topk], 0.79, "#d1495b")             # re-tokenise
    out_i = col([f"“{(out_text or '…')[:22]}”"], 0.99, "#6a8d3f")               # output text

    src, tgt, val, lc = [], [], [], []
    for j in range(n):                                        # input text → each token
        src.append(inp_i); tgt.append(tok_i + j)
        val.append(1.0); lc.append("rgba(138,107,191,0.30)")
    for j in range(n):                                        # token → its vector
        src.append(tok_i + j); tgt.append(vec_i + j)
        val.append(max(abs(e0[j]), 0.02)); lc.append("rgba(46,139,139,0.35)")
    for j in range(n):                                        # vector → model hub (by attention)
        src.append(vec_i + j); tgt.append(hub_i)
        val.append(max(aw[j], 1e-3)); lc.append("rgba(63,143,158,0.40)")
    for j, (_, p) in enumerate(topk):                         # hub → predicted token (by prob)
        src.append(hub_i); tgt.append(prd_i + j)
        val.append(max(float(p), 1e-3)); lc.append("rgba(209,73,91,0.40)")
    if topk:                                                  # best predicted → output text
        src.append(prd_i); tgt.append(out_i)
        val.append(max(float(topk[0][1]), 1e-3)); lc.append("rgba(106,141,63,0.45)")

    fig = go.Figure(go.Sankey(
        arrangement="fixed",
        node=dict(label=labels, x=xs, y=ys, color=colors, pad=12, thickness=15,
                  line=dict(color="#222", width=0.5)),
        link=dict(source=src, target=tgt, value=val, color=lc)))
    fig.update_layout(
        height=height, margin=dict(l=6, r=6, t=34, b=6),
        title=dict(text="input text → tokenise → vectorise → attention+FFN → re-tokenise → output text",
                   font=dict(size=12, color="#cdd")),
        font=dict(color="#e6e6e6", size=11), paper_bgcolor="rgba(0,0,0,0)")
    return fig
