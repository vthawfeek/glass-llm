"""Glass-LLM v2 — an interactive glass box for language models on CLINICAL-TRIAL text.

Two tabs:
  • Model explorer — a collapsible left panel with 5 dropdowns (data domain, data volume,
    attention heads, fine-tuning, RAG). Every change re-selects a real trained model and
    re-renders the architecture, a token→vector→text Sankey, and the tokenization /
    embeddings / attention / generation panels, plus biomarker tagging and retrieval.
  • Tokenizer audit — how every knob-combination and real LLMs (GPT-4/4o/2, BERT) tokenize
    your terms.

Highlight is TRANSPARENCY (watch each internal step), not model intelligence. Every
generated string is watermarked SYNTHETIC and scoped to trial metadata — not medical advice.
"""
import sys, os, html, importlib
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import bpe, model, glass, audit, rag, ui, ui2, biomarker
for _n in ("bpe", "model", "glass", "audit", "rag", "ui", "ui2", "biomarker"):
    importlib.reload(sys.modules[_n])
GlassModel, WATERMARK = glass.GlassModel, glass.WATERMARK

TOK_DIR = ROOT / "pilot" / "tokenizers"
MODEL_DIR = ROOT / "models_v2"
RAG_DIR = ROOT / "pilot" / "rag_dom"
RAG_EMBED_TAG = "d_hi_4h"          # index was built with this model; embed queries with it too
PALETTE = ["#2e8b8b", "#d1495b", "#6a8d3f", "#c77d34", "#5b6bbf", "#9a5ba6",
           "#3f8f9e", "#b5651d", "#7a7f35", "#8a4f7d"]


def _srcver():
    return max(os.path.getmtime(ROOT / "src" / f) for f in
               ("bpe.py", "model.py", "glass.py", "audit.py", "rag.py", "ui.py", "ui2.py",
                "biomarker.py"))
SRCVER = _srcver()

st.set_page_config(page_title="Glass-LLM · Clinical Trials", page_icon="🧬", layout="wide",
                   initial_sidebar_state="expanded")


# ---------------- cached loaders (SRCVER busts them on any src edit) ----------------
@st.cache_resource
def load_model(tag, srcver):
    p = MODEL_DIR / f"{tag}.pt"
    return GlassModel.load(p, tokenizer_dir=str(TOK_DIR)) if p.exists() else None

@st.cache_resource
def load_bio(tag, srcver):
    p = MODEL_DIR / f"{tag}_bio.pt"
    return biomarker.load_head(p) if p.exists() else None

@st.cache_resource
def load_tok(name, srcver):
    p = TOK_DIR / f"{name}.json"
    return bpe.BPETokenizer.load(p) if p.exists() else None

@st.cache_resource
def load_rag(srcver):
    return rag.load_index(RAG_DIR) if rag.exists(RAG_DIR) else (None, None)

@st.cache_data
def pca3(tag, srcver):
    E = load_model(tag, srcver).embedding_matrix()
    Ec = E - E.mean(0)
    _, _, Vt = np.linalg.svd(Ec, full_matrices=False)
    return Ec @ Vt[:3].T

@st.cache_data
def gen_cached(tag, prompt, max_new, temperature, top_k, srcver):
    return load_model(tag, srcver).generate(prompt, max_new_tokens=max_new,
                                            temperature=temperature, top_k=top_k)


def chips(g, ids, flags=None, limit=None):
    """Colored token chips; if `flags` given, biomarker tokens are highlighted red."""
    out = []
    ids = ids[:limit] if limit else ids
    for i, tid in enumerate(ids):
        piece = html.escape(g.token_text(tid) or "∅")
        if flags is not None:
            bg = "#d1495b" if flags[i] else "#33484d"
        else:
            bg = PALETTE[i % len(PALETTE)]
        out.append(f"<span style='background:{bg};color:#fff;padding:1px 5px;margin:1px;"
                   f"border-radius:4px;font-family:monospace;font-size:.85em'>{piece}</span>")
    return "".join(out)

def esc(s):
    return html.escape(str(s))

def explain(key):
    e = ui.EXPLAIN[key]
    st.markdown(e["headline"])
    with st.expander("ⓘ How does this work? (plain English)"):
        st.markdown(e["how"])

def output_box(text, tint="#2e8b8b"):
    """A large, obvious box for generated text."""
    st.markdown(
        f"<div style='border:2px solid {tint};border-radius:10px;padding:16px 18px;"
        f"background:#0e2a2a55;font-size:1.15rem;line-height:1.6;white-space:pre-wrap;"
        f"font-family:ui-monospace,monospace;color:#eef5f5'>{html.escape(text)}</div>",
        unsafe_allow_html=True)


# ================================= header =================================
st.title("🧬 Glass-LLM · Clinical Trials")
st.caption("Pick a model on the left; watch text flow through it, step by step. All five "
           "controls select a **real trained model** — nothing here is a mock-up.")
st.warning(f"⚠️ {WATERMARK}. Scope: clinical-trial metadata only — not medical advice.")


# ============================ left panel: 5 dropdowns ============================
with st.sidebar:
    st.header("Model controls")
    st.caption("Open/close this panel with the ‹ arrow. Each control changes the model "
               "and the panels on the right.")
    K = ui2.KNOBS
    d_domain = st.selectbox(K["domain"]["label"], K["domain"]["options"], key="k_domain",
                            help=K["domain"]["help"])
    d_volume = st.selectbox(K["volume"]["label"], K["volume"]["options"], index=2, key="k_volume",
                            help=K["volume"]["help"])
    d_heads = st.selectbox(K["heads"]["label"], K["heads"]["options"], key="k_heads",
                           help=K["heads"]["help"])
    d_ft = st.selectbox(K["finetune"]["label"], K["finetune"]["options"], key="k_ft",
                        help=K["finetune"]["help"])
    d_rag = st.selectbox(K["rag"]["label"], K["rag"]["options"], key="k_rag", help=K["rag"]["help"])

is_generic = d_domain.startswith("📖")
n_head = 8 if d_heads.startswith("8") else 4
want_ft = d_ft == "Yes"
want_rag = d_rag == "Yes"
tag = ui2.tag_for(d_domain, d_volume, d_heads)

tab_a, tab_b = st.tabs(["🔬  Model explorer", "🔤  Tokenizer audit (all combinations)"])

# =============================================================== TAB A
with tab_a:
    g = load_model(tag, SRCVER)
    if g is None:
        st.info(f"⏳ Model **{tag}** isn’t built yet. The zoo (12 models + heads + index) is "
                f"training — rerun once `models_v2/{tag}.pt` exists.")
        st.stop()
    cfg = g.meta["cfg"]

    # --- (a) INPUT TEXT — the first thing under the SYNTHETIC banner ---
    st.info("This is a small **text-completion** model (~1.3M parameters, like an early GPT-2) — "
            "**not** a chatbot. It predicts what text comes next.")
    default_prompt = ("What are the eligibility criteria for an osimertinib trial in "
                      "EGFR T790M-positive lung cancer?")
    st.markdown("#### ✍️ Input text")
    prompt = st.text_area("Input text", default_prompt, height=110, label_visibility="collapsed",
                          help="This is a text-completion model, not a chatbot: it continues your "
                               "text rather than answering it. A trial-style question or opener "
                               "with a drug and disease reads best, and a biomarker (EGFR, PD-L1, "
                               "BRAF V600E) lights up the fine-tuning panel.")
    st.caption("Not sure what to type? This model completes text, so a trial-style question or "
               "opener with a drug and disease works best. Try:\n\n"
               "- Which trials study pembrolizumab in metastatic melanoma?\n"
               "- What is the purpose of a trastuzumab study in HER2-positive breast cancer?\n"
               "- Study of nivolumab in advanced renal cell carcinoma:\n"
               "- Inclusion Criteria: patients with BRAF V600E mutated melanoma")
    c1, c2, c3 = st.columns(3)
    temperature = c1.slider("temperature", 0.1, 1.5, 0.7, 0.1, help=ui.help_for("temperature"))
    top_k = c2.slider("top-k", 1, 100, 40, 1, help=ui.help_for("top_k"))
    max_new = c3.slider("max new tokens", 5, 80, 40, 5, help=ui.help_for("max_new_tokens"))
    tr = g.trace(prompt)
    out_text = gen_cached(tag, prompt, 14, temperature, top_k, SRCVER)["text"][len(prompt):][:30]

    # --- (b) architecture figure driven by the 5 knobs (now after the input text) ---
    st.markdown(f"##### Your model:  `{tag}`  ·  {d_domain} · {d_volume} data · {n_head} heads"
                f"{' · fine-tuned' if want_ft else ''}{' · RAG' if want_rag else ''}")
    ac, dc = st.columns([3, 2])
    ac.plotly_chart(ui2.architecture_figure_v2(is_generic, d_volume, n_head, want_ft, want_rag,
                    g.meta["vocab_size"]), width="stretch", key="arch_v2")
    rows = [("Data domain", "English books" if is_generic else "Clinical trials"),
            ("Data volume", f"{d_volume} (~{g.meta.get('train_bytes', 0)//1000:,} KB)"),
            ("Tokenizer", f"{g.meta['tokenizer']} · {g.meta['vocab_size']:,} tokens"),
            ("Model", f"{sum(p.numel() for p in g.model.parameters())/1e6:.1f}M · "
                      f"{cfg['n_layer']}L × {cfg['n_head']}H"),
            ("Test BPB", g.meta.get("test_bpb", "?")),
            ("Fine-tuned", "Biomarker tagging head" if want_ft else "No"),
            ("Retrieval", "RAG (on)" if want_rag else "No")]
    dc.markdown(ui.dataset_card_html("Architecture & training data", rows), unsafe_allow_html=True)
    st.caption("🔴 Red = what this selection changes from the default (domain · High · 4 heads). "
               "Hover the ⓘ marks on the diagram for the math at each step, in plain English.")
    with st.expander("ⓘ The math at each step (plain English)"):
        st.table(pd.DataFrame(ui2.ARCH_TABLE, columns=["Step", "Formula", "What it means"]))

    # --- (c) the pipeline Sankey — real per-layer Attention→FFN chain (collapsible), no black box ---
    st.markdown("##### The flow, as one picture")
    expand_layers = st.toggle(
        "Expand the transformer layers (Attention + Feed-Forward, one pair per layer)",
        value=True, key="sankey_expand",
        help="On: every layer’s Attention and Feed-Forward node, each labelled with the real amount "
             "‖Δ‖ it writes into the residual stream. Off: collapse all layers into one "
             "Attention+Feed-Forward node for a clean, uncrowded overview.")
    last = tr.n_tokens - 1
    n_layer = cfg["n_layer"]
    entry_attn = tr.attentions[0].mean(axis=0)[last]          # layer-1 cross-token mixing (last token)
    emb0 = tr.embeddings[:, 0]
    h = [hh[last] for hh in tr.hiddens]                       # residual stream at the predictive token
    a = [aa[last] for aa in tr.attn_contrib]
    f = [ff[last] for ff in tr.ffn_contrib]
    attn_norms = [float(np.linalg.norm(v)) for v in a]
    ffn_norms = [float(np.linalg.norm(v)) for v in f]
    stream = [float(np.linalg.norm(h[0]))]
    for l in range(n_layer):
        stream.append(float(np.linalg.norm(h[l] + a[l])))    # residual norm after attention
        stream.append(float(np.linalg.norm(h[l + 1])))       # residual norm after feed-forward
    st.plotly_chart(ui2.pipeline_sankey(prompt, [g.token_text(t) for t in tr.token_ids], emb0,
                    entry_attn, attn_norms, ffn_norms, stream,
                    [(t, p) for t, p, _ in tr.topk[:6]], out_text, expand=expand_layers),
                    width="stretch", key="pipe")
    st.caption("Left→right: your **tokens** → their **vectors** (e₀ = first embedding number) → the "
               "transformer (**expanded** into each layer’s Attention→Feed-Forward, or **collapsed** "
               "into one hub via the toggle above) → the **LM head** → the **next-token** candidates → "
               "**text**. When expanded, each node’s Δ is how much that sublayer writes into the "
               "residual stream. The spine follows the final token (the position that predicts the "
               "next one); the fan-in on the left is where the other tokens enter, via attention. "
               "Note the **attention node sums over all heads** — switch control 3 (4↔8 heads) to see "
               "the per-head detail in the **Attention** panel below; here it changes the numbers "
               "(a different trained model), not the diagram’s shape.")

    # --- (d) tokenization ---
    st.subheader("1 · Tokenization"); explain("tokenization")
    st.markdown(chips(g, tr.token_ids), unsafe_allow_html=True)
    nwords = max(1, len(prompt.split()))
    m = st.columns(3)
    m[0].metric("tokens", tr.n_tokens, help=ui.help_for("token"))
    m[1].metric("words", nwords)
    m[2].metric("tokens / word", f"{tr.n_tokens/nwords:.2f}", help=ui.help_for("tokens_per_word"))

    # --- (e) embeddings ---
    st.subheader("2 · Embeddings"); explain("embeddings")
    coords = pca3(tag, SRCVER)
    rng = np.random.default_rng(0)
    samp = rng.choice(coords.shape[0], size=min(700, coords.shape[0]), replace=False)
    uniq = list(dict.fromkeys(tr.token_ids))
    fig_e = go.Figure()
    fig_e.add_trace(go.Scatter3d(x=coords[samp, 0], y=coords[samp, 1], z=coords[samp, 2],
                    mode="markers", marker=dict(size=2, color="#c9c9c9", opacity=.5),
                    name="vocabulary", hoverinfo="skip"))
    fig_e.add_trace(go.Scatter3d(x=coords[uniq, 0], y=coords[uniq, 1], z=coords[uniq, 2],
                    mode="markers+text", marker=dict(size=5, color="#d1495b"),
                    text=[g.token_text(t) for t in uniq], textposition="top center",
                    name="prompt tokens"))
    fig_e.update_layout(height=380, margin=dict(l=0, r=0, t=0, b=0),
                        scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
                        legend=dict(orientation="h"))
    st.plotly_chart(fig_e, width="stretch", key="emb3d")
    if uniq:
        pick = st.selectbox("Cosine nearest neighbors of", uniq,
                            format_func=lambda t: g.token_text(t), help=ui.help_for("cosine"))
        nn = g.nearest(pick, 6)
        st.table({"token": [t for t, _, _ in nn], "cosine": [round(s, 3) for _, s, _ in nn]})

    # --- (f) attention ---
    st.subheader("3 · Attention"); explain("attention")
    a1, a2 = st.columns(2)
    layer = a1.slider("layer", 0, cfg["n_layer"] - 1, 0, help="Transformer block, bottom to top.")
    head = a2.slider("head", 0, cfg["n_head"] - 1, 0, help=ui.help_for("attention"))
    labels = [g.token_text(t) for t in tr.token_ids]
    fig_a = go.Figure(go.Heatmap(z=tr.attentions[layer][head], x=labels, y=labels, colorscale="Teal"))
    fig_a.update_layout(height=430, margin=dict(l=6, r=6, t=6, b=6),
                        yaxis=dict(autorange="reversed"),
                        xaxis_title="attended-to (key)", yaxis_title="query")
    st.plotly_chart(fig_a, width="stretch", key="attn")
    st.caption(f"This model has **{cfg['n_head']} heads** per layer — change control 3 to compare.")

    # --- (4) fine-tuning: biomarker tagging (ALWAYS shown so numbering stays 1-2-3-4-5) ---
    st.subheader("4 · Fine-tuning — biomarker tagging")
    st.markdown("**A task head added on top of the frozen model** learns to tag biomarker "
                "mentions. This is a *mechanism* demo — watch which tokens light up.")
    with st.expander("ⓘ How does this work? (plain English)"):
        st.markdown("Fine-tuning here adds a small classifier on top of the model’s final "
                    "vectors, trained on clinical text where genes/biomarkers were "
                    "auto-labeled (distant supervision). It shows *how* fine-tuning bolts a "
                    "task onto a base model. It is **not** a real biomarker extractor: a "
                    "1.3M model can’t structure biomarkers (the state of the art uses ~7B "
                    "models), and this head just imitates the lexicon — it misses novel terms "
                    "and mishandles negation like *EGFR wild-type*.")
    if not want_ft:
        st.info("This step is **inactive** because the model isn’t fine-tuned. Turn on "
                "**control 4 · Fine-tuned = Yes** in the left panel to add the biomarker-tagging "
                "head and highlight biomarkers in your prompt.")
    else:
        bio = load_bio(tag, SRCVER)
        if is_generic or bio is None:
            st.info("Biomarker tagging is fine-tuned on **clinical** text. Switch control 1 to "
                    "🏥 Domain to see it (there are no biomarkers in English novels).")
        else:
            head_m, meta = bio
            tags, ids = biomarker.tag(g, head_m, prompt)
            flags = [t[2] for t in tags]
            st.markdown(chips(g, ids, flags=flags), unsafe_allow_html=True)
            hitnames = [t[0].replace("␣", " ").strip() for t in tags if t[2]]
            st.caption(("🔴 tagged as biomarker: " + ", ".join(f"`{h}`" for h in hitnames))
                       if hitnames else "No biomarker tokens tagged in this prompt.")

    # --- (g) generation + (h) output text ---
    st.subheader("5 · Generation" + (" + Retrieval (RAG)" if want_rag else ""))
    explain("generation")
    lab = [t for t, _, _ in tr.topk][::-1]; val = [p for _, p, _ in tr.topk][::-1]
    fig_g = go.Figure(go.Bar(x=val, y=lab, orientation="h", marker_color="#2e8b8b"))
    fig_g.update_layout(height=300, margin=dict(l=6, r=6, t=28, b=6),
                        title="Next-token probability (top-k)", xaxis_title="probability")
    st.plotly_chart(fig_g, width="stretch", key="gen")

    if not want_rag:
        st.markdown("#### 📤 Output text — the model continues your prompt")
        out = gen_cached(tag, prompt, max_new, temperature, top_k, SRCVER)
        output_box(out["text"])
        st.caption(f"⚠️ {out['watermark']}")
    else:
        explain("rag")
        embs, chunks = load_rag(SRCVER)
        if embs is None:
            st.info("RAG index not built yet — it’s created at the end of the zoo build.")
        else:
            st.caption("Retrieval uses the **same prompt** from the top of the page — one input "
                       "drives every step. A topic/keyword prompt (a condition, drug, or biomarker) "
                       "retrieves best.")
            k = st.slider("retrieved trials (top-k)", 1, 8, 4,
                          help="How many nearest trial chunks to pull from the index.")
            embed_model = load_model(RAG_EMBED_TAG, SRCVER) or g
            hits = rag.query(embed_model, embs, chunks, prompt, k=k)
            colL, colR = st.columns(2)
            with colL:
                st.markdown("##### ❌ Without retrieval — the model alone")
                out = gen_cached(tag, prompt, 40, temperature, top_k, SRCVER)
                output_box(out["text"], tint="#d1495b")
                st.caption("⚠️ Invented, plausible-sounding specifics (hallucination).")
            with colR:
                st.markdown("##### ✅ With retrieval — grounded in real trials")
                for nct, text, score in hits:
                    st.markdown(f"**[{nct}]** · similarity `{score:.3f}`")
                    st.caption(text[:200] + ("…" if len(text) > 200 else ""))
            with st.expander("See the assembled grounded prompt"):
                st.code(rag.build_grounded_prompt(prompt, hits))
            st.caption(f"⚠️ {WATERMARK}")

# =============================================================== TAB B
with tab_b:
    st.markdown("#### How every model + real LLMs tokenize *your* terms")
    explain("audit")
    default_terms = ("pembrolizumab\nosimertinib\ntrastuzumab\nEGFR T790M\nBRAF V600E\n"
                     "ALK rearrangement\nPD-L1 expression\nmicrosatellite instability")
    terms_text = st.text_area("Terms to audit (one per line)", default_terms, height=140)
    terms = [t for t in terms_text.splitlines() if t.strip()]

    # distinct tokenizers reachable from the 5 knobs = the two 'ours' + the 4 real LLMs
    OURS = ["Ours · clinical trials (domain)", "Ours · general (English)"]
    LLMS = ["GPT-4 / GPT-3.5 (cl100k)", "GPT-4o (o200k)", "GPT-2", "BERT (WordPiece)"]
    have = []
    for n in OURS + LLMS:
        try:
            audit.count(n, "test"); have.append(n)
        except Exception:
            pass
    if terms and have:
        table = {n: [audit.count(n, t) for t in terms] for n in have}
        df = pd.DataFrame(table, index=terms)
        df.loc["— mean tokens/term —"] = [round(audit.fertility(n, terms), 2) for n in have]
        st.dataframe(df, width="stretch")
        means = [audit.fertility(n, terms) for n in have]
        colors = ["#2e8b8b" if n.startswith("Ours") else "#d1495b" for n in have]
        fig = go.Figure(go.Bar(x=have, y=means, marker_color=colors))
        fig.update_layout(height=340, margin=dict(l=6, r=6, t=30, b=110), xaxis_tickangle=-25,
                          title="Mean tokens per term (lower = less fragmentation)")
        st.plotly_chart(fig, width="stretch", key="audit_bar")
        st.info("**Key insight:** among your five controls, only **control 1 (data domain)** "
                "changes tokenization. Volume, attention heads, fine-tuning and RAG do **not** "
                "touch the tokenizer — so all their combinations give identical columns. That’s "
                "why there are two ‘Ours’ columns, not 48.")
        with st.expander("Show all 48 parameter combinations (proving the invariance)"):
            combos, tok_of = [], {}
            for dom in ("🏥 Domain", "📖 Generic"):
                for vol in ("Low", "Med", "High"):
                    for h in ("4h", "8h"):
                        for ft in ("ft", "base"):
                            for rg in ("rag", "norag"):
                                name = f"{'D' if dom.startswith('🏥') else 'G'}·{vol}·{h}·{ft}·{rg}"
                                combos.append(name)
                                tok_of[name] = OURS[0] if dom.startswith("🏥") else OURS[1]
            wide = {c: [audit.count(tok_of[c], t) for t in terms] for c in combos}
            st.dataframe(pd.DataFrame(wide, index=terms), width="stretch")
            st.caption("Every ‘D·…’ column is identical (the clinical-trials tokenizer); every "
                       "‘G·…’ column is identical (the English tokenizer). Tokenization depends "
                       "only on the tokenizer — a real, load-bearing teaching point.")
    elif not have:
        st.info("Tokenizers not built yet — the clinical-trials (domain) tokenizer is created "
                "during the zoo build.")
    st.caption("⚠️ Fragmentation flags a risk for rare terms and exact-match tasks; it does not "
               "by itself prove worse quality. Cost & context are facts (you pay per token); "
               "quality is a flag to test.")
