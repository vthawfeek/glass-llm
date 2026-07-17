---
title: Glass LLM
emoji: 🧬
colorFrom: green
colorTo: red
sdk: docker
app_port: 8501
short_description: A glass box for LLMs on clinical-trial text
pinned: false
license: mit
---

# 🧬 Glass-LLM — a glass box for language models on clinical-trial text

**Live demo:** https://glass-llm.streamlit.app

A decoder-only GPT and a BPE tokenizer, both **written from scratch** and trained on
ClinicalTrials.gov text (all curated sections), wrapped in a dashboard with **five dials** — each
one selects a real trained model and visibly reshapes what happens inside.

> The point of this project is not a smart model — it's a **transparent** one. The model is
> ~1.3M parameters; everything it generates is watermarked synthetic and is **not** medical
> information.

## The five dials

A collapsible left panel with five dropdowns; changing any of them re-renders every panel with the
model's real numbers. Hover the architecture diagram for the math at each step, in plain English.

| Dial | What it changes |
|---|---|
| 1 · Training data (clinical ⇄ general English) | tokenization, embeddings, and generation |
| 2 · Data volume (1 ⇄ 8 ⇄ 32 MB) | how much the model memorises vs. generalises |
| 3 · Attention heads (4 ⇄ 8) | the attention heatmaps |
| 4 · Fine-tuning (on/off) | adds a biomarker-tagging task head (a *mechanism* demo) |
| 5 · Retrieval / RAG (on/off) | hallucination vs. grounding in real trials |

Two tabs: **🔬 Model explorer** (the five dials + tokenization, embeddings, attention, fine-tuning,
generation, and an input-text→tokens→vectors→text Sankey) and **🔤 Tokenizer audit** (paste your
own terms; compare GPT-4 / GPT-4o / GPT-2 / BERT).

## Findings you can check

**Data volume — memorise vs. generalise.** Same architecture, tokenizer, and 4,000 training steps;
only the data changes. The 1 MB model looks *most* fluent but scores *worst* on held-out text —
it has memorised the corpus. Bits-per-byte (lower = better):

| Training data | Held-out bits-per-byte |
|---|---|
| 1 MB (low) | 2.24 |
| 8 MB (medium) | 1.49 |
| 32 MB (high) | 1.47 |

**Tokenization — real, stated carefully.** *pembrolizumab* is **4** tokens with the clinical
tokenizer (vocab 4,096), **6** with GPT-4's (`p·emb·rol·iz·um·ab`), 5 with GPT-4o, 6 with GPT-2/BERT.

- **Cost & context are facts.** A full eligibility criterion: **30 tokens (domain) vs 55 (general)**
  at equal vocab size — cheaper, more usable context.
- **"Domain always wins" is false.** GPT-4's 100k+ vocab is competitive (**4.22** tokens/term vs
  **4.94** on the 90-term probe set in `pilot/probe_terms.txt`). The clean win is domain-vs-general
  *at equal vocab* (~20–45% fewer), **not** "beats GPT-4."
- **Quality is a flag, not a verdict** — a risk to test for rare/exact-match terms, not proof of
  worse understanding. (This is why [PubMedBERT](https://arxiv.org/abs/2007.15779) trained its own
  vocabulary; trade-offs in the [tokenizer literature](https://arxiv.org/abs/2310.08754).)

**Fine-tuning is a mechanism, not an extractor.** A 1.3M model cannot do structured biomarker
extraction — the state of the art is fine-tuned ~7B models
([Alkhoury et al., npj Digital Medicine 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12053753/)).
The fine-tune dial shows a task head *tagging* biomarker mentions (distant supervision), labelled
plainly as a demo.

## Architecture

- **Tokenizer:** byte-level BPE from scratch (`src/bpe.py`), ~150 lines, no `tiktoken`/HF.
- **Model:** decoder-only transformer from scratch (`src/model.py`), ~1.3M params
  (d_model 128, 4 layers, 4 or 8 heads, context 128), weight-tied head.
- **Trace API** (`src/glass.py`): one call returns tokens, embeddings, per-layer/head attention,
  hidden states, and next-token logits — the single interface every panel reads.
- **Fine-tune head** (`src/biomarker.py`): distant-supervision biomarker labels + a linear task
  head on the frozen model.
- **RAG** (`src/rag.py`): transparent cosine vector index over 500 real trial chunks (same
  operation FAISS/Chroma perform, kept visible).

## Limitations

- ~1.3M parameters. It produces trial-*shaped* text with invented specifics — a demonstration of
  the mechanism, not a capable model. All output is watermarked **synthetic**.
- Small-scale findings; the tokenizer quality gap attenuates with scale and may not transfer to
  frontier models.
- Scope is trial **metadata** only. This is not medical advice.

## Reproduce

```bash
uv venv --python 3.12 .venv && uv pip install -r requirements-lock.txt
python src/fetch_data.py --area all --out pilot/data_dom --no-general   # ClinicalTrials.gov (API v2)
# (build a generic English corpus into pilot/data_gen; see fetch_data.fetch_general)
python src/build_zoo.py           # trains the tokenizer + 12-model zoo + biomarker heads + RAG index
python src/warm_tokenizers.py     # cache GPT-4/GPT-2/BERT tokenizers for the offline audit
streamlit run app_v2.py
```

Tests: `python tests/test_bpe.py` · `python tests/test_trace.py` · `python tests/test_biomarker.py`
· `python tests/verify_app_v2.py`.

_Built with references to nanoGPT (architecture) and minbpe (tokenizer). Prior interactive
explainers: [Transformer Explainer](https://poloclub.github.io/transformer-explainer/)._
