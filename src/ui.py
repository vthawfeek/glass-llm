"""UI helpers for the dashboard: plain-language glossary, a clean 2D architecture schematic,
and a staged fade-in "pipeline" renderer. Kept separate so app.py stays readable."""
import html
import numpy as np
import plotly.graph_objects as go

# ---- plain-language tooltips (used via st widget help= and metric help=) ----
GLOSSARY = {
    "temperature": "Randomness of generation. Low (0.2) → the model almost always picks its top "
                   "guess (safe, repetitive). High (1.4) → flatter odds, more variety and surprises.",
    "top_k": "Only the k most likely next tokens are kept as candidates; the rest are dropped. "
             "Smaller k → more focused, safer text.",
    "max_new_tokens": "How many tokens the model generates before stopping. One token is roughly a "
                      "short word or word-fragment.",
    "bpb": "Bits-per-byte: how well the model predicts held-out text, normalized by raw bytes so "
           "tokenizers with different vocabularies are comparable. Lower is better.",
    "token": "The unit a model actually reads, a whole word or a word-fragment drawn from a fixed "
             "vocabulary. Models never see raw letters as words; they see tokens.",
    "tokens_per_word": "Average number of tokens each word is split into. Lower = less fragmentation "
                       "= cheaper and more context left for real content.",
    "attention": "Each layer has several attention 'heads', parallel lenses that learn to focus on "
                 "different relationships between tokens (e.g. a word and the one it depends on).",
    "cosine": "How aligned two vectors are, from -1 to 1. 1 = same direction (very similar meaning "
              "in the model's embedding space).",
    "embedding": "A learned vector (list of numbers) standing for a token's meaning. Similar tokens "
                 "sit close together in this space.",
    "logits": "The model's raw score for every possible next token, before they're turned into "
              "probabilities by softmax.",
    "rag": "Retrieval-Augmented Generation: fetch relevant real documents first, then hand them to "
           "the model as context, so answers are grounded in sources instead of invented.",
    "fertility": "Tokens per term. A tokenizer with lower fertility splits your vocabulary into "
                 "fewer pieces.",
    "pca": "Principal Component Analysis: squashes the high-dimensional embedding space down to 3 "
           "axes so it can be drawn, keeping as much spread as possible.",
}


def help_for(key):
    return GLOSSARY.get(key, "")


# ---- per-step explanations: a one-line headline (always shown) + a plain-English
# "how it works" with an analogy (tucked in an expander). Inspired by Transformer Explainer. ----
EXPLAIN = {
    "tokenization": {
        "headline": "**First, your text is chopped into _tokens_**, whole words or word-fragments "
                    "drawn from a fixed list of 4,096 pieces this model knows.",
        "how": "A language model can't read raw letters, it only knows a fixed vocabulary. A "
               "byte-pair tokenizer builds that vocabulary by repeatedly merging the most common "
               "character pairs. So common words like *metastatic* become **one** token, while a "
               "rare word like *pembrolizumab* gets shattered into several. Fewer, cleaner tokens "
               "means the model spends less effort just recognising a word, and it's why a "
               "clinical tokenizer beats a general one on biomedical text.",
    },
    "embeddings": {
        "headline": "**Each token becomes a _vector_**, a list of 128 numbers that stands for its "
                    "meaning. Tokens with similar meanings sit close together.",
        "how": "Think of a map of meaning: *cancer*, *tumor*, and *carcinoma* end up in the same "
               "neighbourhood; *the* and *of* end up somewhere else. The model *learned* these "
               "positions during training. The real space has 128 dimensions, we flatten it to 3 "
               "with PCA so you can see it. Drag to rotate; your prompt's tokens are the red dots.",
    },
    "attention": {
        "headline": "**Attention lets each token look back** at earlier tokens and decide which "
                    "ones matter for predicting what comes next.",
        "how": "It works like a search engine. Each token writes a **Query** ('what am I looking "
               "for?'), every earlier token offers a **Key** ('here's what I am'), and the best "
               "matches pass along their **Value** ('here's my content'). Each **row** of the "
               "heatmap sums to 1, it shows how much that token attends to every earlier token "
               "(it can't peek ahead, so the top-right is blank). A model with more **heads** "
               "tracks more of these relationships at once.",
    },
    "generation": {
        "headline": "**The final vector becomes a score for every possible next token.** Softmax "
                    "turns the scores into probabilities, then one token is sampled.",
        "how": "Those raw scores are called *logits*. **Temperature** divides them before softmax: "
               "low (÷0.2) sharpens toward the single best guess (confident, repetitive); high "
               "(÷1.4) flattens the odds (creative, riskier). **top-k** keeps only the k most "
               "likely candidates. The chosen token is turned back into text, then the whole loop "
               "repeats to produce the next token.",
    },
    "audit": {
        "headline": "**Same terms, different tokenizers.** See how real LLMs (GPT-4, GPT-4o, GPT-2, "
                    "BERT) split _your_ vocabulary, and what that costs.",
        "how": "Every tokenizer has a different vocabulary, so it fragments the same word "
               "differently, *pembrolizumab* is 2 tokens for our clinical tokenizer, 6 for "
               "GPT-4's. **Cost and context are facts**: you pay per token and your context window "
               "is counted in tokens. **Quality is only a flag**, heavy fragmentation is a risk "
               "to test for rare terms, not proof the model understands them worse.",
    },
    "rag": {
        "headline": "**Retrieval happens _outside_ the model.** Before answering, we fetch real "
                    "trial records and hand them to the model as context.",
        "how": "The question is turned into a vector, then compared (by cosine similarity) against "
               "a stored index of real trial chunks, the same nearest-neighbour operation a "
               "vector database (FAISS, Chroma) performs. The best matches, with their NCT IDs, "
               "become context. **Left:** the model alone invents a plausible but fake trial. "
               "**Right:** grounded in retrieved real records. This is why RAG reduces "
               "hallucination.",
    },
}



# ---- model metadata: layman name + plain tooltip + which component differs ----
MODELS = {
    "hero_domain":  {"short": "🏥 Clinical (default)",
                     "diff": "Our main model. Its tokenizer learned from clinical-trial text, so "
                             "medical words stay whole. 4 attention heads.",
                     "highlight": None, "tok": "domain_4096"},
    "hero_general": {"short": "📖 Everyday English",
                     "diff": "Same model, but its tokenizer learned from ordinary English books "
                             "instead of clinical text. Watch medical words shatter into more "
                             "pieces in the Tokenization step.",
                     "highlight": "tokenizer", "tok": "general_4096"},
    "arch_8head":   {"short": "🔎 More attention",
                     "diff": "Same as Clinical, but with 8 attention heads instead of 4, more "
                             "parallel 'lenses' looking at the text. Watch the Attention step change.",
                     "highlight": "attention", "tok": "domain_4096"},
    "ft_biologics": {"short": "💉 Biologics-tuned",
                     "diff": "The Clinical model, trained further on biologics trials. Its learned "
                             "weights shift; the tokenizer stays identical (watch: Tokenization "
                             "doesn't change, Generation does).",
                     "highlight": "weights", "tok": "domain_4096"},
}

_C = {"pink": "rgba(209,73,120,0.18)", "orange": "rgba(199,125,52,0.20)",
      "blue": "rgba(91,107,191,0.18)", "green": "rgba(106,141,63,0.20)",
      "grey": "rgba(150,150,150,0.12)"}


def architecture_figure(n_layer, n_head, tok_label, vocab, highlight, height=430):
    """Vertical decoder-only schematic (à la the classic Transformer figure), with the
    component that this model changes lit up in red."""
    fig = go.Figure()
    # (key, label, fill, y0, y1)
    boxes = [
        ("input",     "Input text", _C["grey"], 0.15, 0.9),
        ("tokenizer", f"Token Embedding · vocab {vocab:,}", _C["pink"], 1.05, 1.9),
        ("pos",       "+ Positional Encoding", _C["grey"], 2.05, 2.7),
        ("attention", f"Masked Multi-Head Attention · {n_head} heads", _C["orange"], 3.0, 3.95),
        ("an1",       "Add & Norm", _C["grey"], 4.05, 4.5),
        ("ff",        "Feed Forward", _C["blue"], 4.65, 5.35),
        ("an2",       "Add & Norm", _C["grey"], 5.45, 5.9),
        ("linear",    "Linear", _C["grey"], 6.25, 6.9),
        ("softmax",   "Softmax", _C["green"], 7.05, 7.7),
        ("out",       "Next-token probabilities", _C["grey"], 7.85, 8.6),
    ]
    x0, x1 = 0.7, 4.3
    hot_keys = {highlight} if highlight else set()
    if highlight == "weights":
        hot_keys = {"attention", "ff"}
    for key, label, fill, yy0, yy1 in boxes:
        hot = key in hot_keys
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=yy0, y1=yy1,
                      line=dict(color="#d1495b" if hot else "#5f7f88", width=3 if hot else 1),
                      fillcolor="rgba(209,73,91,0.22)" if hot else fill)
        fig.add_annotation(x=(x0 + x1) / 2, y=(yy0 + yy1) / 2, text=label, showarrow=False,
                           font=dict(size=11, color="#f2f2f2"))
    # up-arrows between boxes
    for i in range(len(boxes) - 1):
        fig.add_annotation(x=x0 - 0.02 + (x1 - x0) / 2, y=boxes[i + 1][3], ax=(x1 + x0) / 2,
                           ay=boxes[i][4], xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1, arrowcolor="#6a8")
    # "× N" bracket beside the transformer block (attention..an2)
    fig.add_shape(type="rect", x0=x1 + 0.15, x1=x1 + 0.25, y0=3.0, y1=5.9,
                  line=dict(color="#7aa", width=1), fillcolor="rgba(0,0,0,0)")
    fig.add_annotation(x=x1 + 0.6, y=4.45, text=f"× {n_layer}<br>layers", showarrow=False,
                       font=dict(size=11, color="#9cc"))
    # a couple of high-level math labels on the left
    fig.add_annotation(x=x0 - 0.35, y=3.5, text="softmax(QKᵀ/√d)·V", showarrow=False,
                       textangle=-90, font=dict(size=9, color="#8ab"))
    fig.add_annotation(x=x0 - 0.35, y=7.4, text="p = softmax(z / T)", showarrow=False,
                       textangle=-90, font=dict(size=9, color="#8ab"))
    fig.update_xaxes(visible=False, range=[-0.2, 5.3])
    fig.update_yaxes(visible=False, range=[0, 8.8])
    fig.update_layout(height=height, margin=dict(l=4, r=4, t=6, b=4),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def dataset_card_html(title, rows, tint="#2e8b8b"):
    """A small 'trained on' card."""
    items = "".join(f"<div style='display:flex;justify-content:space-between;gap:10px;"
                    f"padding:2px 0'><span style='color:#9ab'>{html.escape(k)}</span>"
                    f"<span style='color:#eaeaea;text-align:right'>{html.escape(str(v))}</span></div>"
                    for k, v in rows)
    return (f"<div style='border:1px solid {tint}55;border-left:4px solid {tint};border-radius:8px;"
            f"padding:10px 12px;background:#0e2a2a22'>"
            f"<div style='font-weight:600;color:#eaeaea;margin-bottom:6px'>{html.escape(title)}</div>"
            f"{items}</div>")


_STEP_CSS = """
<style>
.glstep{border:1px solid #2e8b8b55;border-left:4px solid #2e8b8b;border-radius:9px;
  padding:9px 11px;margin-bottom:2px;background:#0e2a2a22;transition:opacity .3s}
.glstep.on{border-left-color:#d1495b;background:#d1495b14}
.glstep .h{display:flex;align-items:center;gap:7px;margin-bottom:4px}
.glstep .num{background:#2e8b8b;color:#fff;border-radius:50%;width:20px;height:20px;display:flex;
  align-items:center;justify-content:center;font-size:.72rem;flex:none}
.glstep.on .num{background:#d1495b}
.glstep .ti{font-weight:600;color:#eaeaea;font-size:.9rem}
.glstep .bd{font-family:ui-monospace,monospace;font-size:.82rem;color:#dfe7e7;word-break:break-word;line-height:1.5}
.glmath{text-align:center;color:#8ab;font-family:ui-monospace,monospace;font-size:.74rem;margin:1px 0 3px 0}
</style>
"""


def step_card_html(num, title, body, math=None, on=False):
    """One vertical journey step; `math` is the high-level transition shown as a ↓ arrow below."""
    arrow = f"<div class='glmath'>↓&nbsp; {math}</div>" if math else ""
    return (f"<div class='glstep {'on' if on else ''}'>"
            f"<div class='h'><div class='num'>{num}</div><div class='ti'>{html.escape(title)}</div></div>"
            f"<div class='bd'>{body}</div></div>{arrow}")


def journey_html(steps, upto=None):
    """Render the full vertical journey. steps = list of (num,title,body,math). `upto` dims the rest."""
    n = len(steps) if upto is None else upto
    out = [_STEP_CSS]
    for i, (num, title, body, math) in enumerate(steps):
        if i < n:
            out.append(step_card_html(num, title, body, math, on=(i == n - 1 and upto is not None)))
        else:
            out.append(f"<div style='opacity:.25'>{step_card_html(num, title, '…', math)}</div>")
    return "".join(out)


def sankey_figure(in_labels, in_flows, topk, height=360):
    """Sankey: input tokens ─(attention)→ prediction ─(probability)→ next-token candidates.
    Driven by real trace data (attention of the last token; next-token probabilities)."""
    # keep the chart legible: show the strongest-attended input tokens
    order = np.argsort(-np.asarray(in_flows))[:10]
    order = sorted(order)
    in_labels = [in_labels[i] for i in order]
    in_flows = [float(in_flows[i]) for i in order]
    n_in, n_c = len(in_labels), len(topk)
    labels = in_labels + ["· prediction ·"] + [t for t, _ in topk]
    node_color = ["#2e8b8b"] * n_in + ["#c77d34"] + ["#d1495b"] * n_c
    xs = [0.001] * n_in + [0.5] + [0.999] * n_c

    def lin(n):
        return list(np.linspace(0.05, 0.95, n)) if n > 1 else [0.5]
    ys = lin(n_in) + [0.5] + lin(n_c)
    pred = n_in
    src, tgt, val, lc = [], [], [], []
    for i, f in enumerate(in_flows):
        src.append(i); tgt.append(pred); val.append(max(f, 1e-3)); lc.append("rgba(46,139,139,0.35)")
    for j, (_, p) in enumerate(topk):
        src.append(pred); tgt.append(n_in + 1 + j); val.append(max(float(p), 1e-3))
        lc.append("rgba(209,73,91,0.40)")
    fig = go.Figure(go.Sankey(
        arrangement="fixed",
        node=dict(label=labels, color=node_color, x=xs, y=ys, pad=11, thickness=14,
                  line=dict(color="#222", width=0.5)),
        link=dict(source=src, target=tgt, value=val, color=lc)))
    fig.update_layout(height=height, margin=dict(l=6, r=6, t=30, b=6),
                      title=dict(text="input tokens ─(attention)→ prediction ─(probability)→ next token",
                                 font=dict(size=12, color="#cdd")),
                      font=dict(color="#e6e6e6", size=11), paper_bgcolor="rgba(0,0,0,0)")
    return fig


STEP_CSS = _STEP_CSS   # public alias for animating step cards from app.py
