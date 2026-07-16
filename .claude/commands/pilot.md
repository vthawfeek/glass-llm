---
description: Run the Glass-LLM Phase-0 pilot end-to-end and produce a GREEN/AMBER/RED go-no-go verdict with real numbers.
argument-hint: "[cpu|gpu] [therapeutic-area e.g. oncology|all] [--quick]"
allowed-tools: Bash, Read, Write, Edit, Glob, Grep, WebFetch, WebSearch, TodoWrite
---

# /pilot — Glass-LLM feasibility pilot (Option A)

You are executing the **Phase-0 pilot** for the Glass-LLM project: a from-scratch, glass-box
language-model study on ClinicalTrials.gov text. This pilot doubles as the project's walking
skeleton. Its ONLY job is to decide, with real preliminary numbers, whether the embedded
controlled result (tokenizer choice changes what a tiny LM represents) is achievable at laptop
scale — and to produce reusable code + data for the full build.

Run agentically to completion. Use TodoWrite to track the stages below. Long training runs go
to the **background** (run_in_background); when they finish you will be re-invoked — resume where
you left off. Every stage checks for its own outputs and **skips/resumes** if already done, so
re-running `/pilot` continues rather than restarting.

## Arguments
- `$1` = compute tier: `cpu` or `gpu`. If omitted, auto-detect via `torch.cuda.is_available()`
  and available VRAM, then print the chosen tier.
- `$2` = therapeutic-area filter for the trial pull (e.g. `oncology`), or `all`. Default `all`.
- `--quick` anywhere in `$ARGUMENTS` = smoke mode: tiny data slice + a few hundred training steps,
  finishes in ~10 min, validates the whole pipeline end-to-end before the real multi-hour run.
  Mark the RESULTS as `MODE: QUICK (not a valid verdict)` when in this mode.

## Non-negotiable scientific rules (embed these; do not deviate)
1. **Bits-per-byte (BPB), never raw perplexity**, for every cross-tokenizer number.
   BPB = (sum of test cross-entropy in nats / ln 2) / (UTF-8 byte count of the raw test text).
   Byte count is fixed across tokenizers → the numbers are comparable.
2. **Matched vocabulary size = 4096** for the two compared BPE tokenizers (equal embedding params).
3. **Split by NCT ID before tokenizing** (80/10/10 train/val/test). No trial may cross the split.
   Assert zero NCT-ID overlap between splits and fail loudly if violated.
4. **≥3 seeds** for each compared condition; report mean ± SD.
5. **Compute-matched**: same corpus, same number of epochs across conditions; log wall-clock,
   tokens/sec, peak memory as covariates.
6. **Integrity**: watermark any sample text the model generates with
   `SYNTHETIC — interpretability demo, not medical information`. Generate no health advice.

## Setup
- Create the project skeleton under the repo root:
  `pilot/`, `pilot/data/`, `pilot/tokenizers/`, `pilot/models/`, `pilot/results/`, `src/`.
- Use a venv (`python -m venv .venv`) or `uv`. Install pinned deps and write a lockfile
  (`requirements.txt` with exact versions): `torch`, `numpy`, `requests`, `regex`, `tqdm`,
  `matplotlib`, `pandas`. (No ChromaDB needed in the pilot.)
- Set all RNG seeds (Python, NumPy, torch) from an explicit `--seed`. Record every config used to
  `pilot/results/config.json`.

## Stage 1 — Data (produces the first, model-free result)
1. Pull ClinicalTrials.gov **API v2** (`https://clinicaltrials.gov/api/v2/studies`), paginated via
   `pageToken`, `pageSize=1000`. Request fields: `eligibilityCriteria`, `briefSummary`,
   `conditions`, `interventions`. Filter to interventional drug trials; apply `$2` area filter if given.
   Cap at ~30k trials for the pilot (fewer with `--quick`).
2. Clean (strip boilerplate/HTML, normalize whitespace), dedupe, and **split by NCT ID** 80/10/10.
   Write `pilot/data/{train,val,test}.txt` + `pilot/data/manifest.json` (trial counts, byte sizes,
   overlap assertion result, source URL, pull date).
3. Acquire ~10–20 MB of **general English** from a public-domain source (e.g. a few Project
   Gutenberg texts via WebFetch) for the general-tokenizer condition; record provenance in the manifest.
4. Build `pilot/probe_terms.txt` (~150 terms): `-mab`/`-tinib`/`-sartan`/`-statin`/`-prazole`/`-vir`/
   `-ciclib` drugs, gene symbols (BRAF, EGFR, TP53), variant notation (V600E), and stock eligibility
   phrases (e.g. "ECOG performance status", "estimated glomerular filtration rate", "informed consent").

## Stage 2 — Tokenizers (from scratch)
1. Implement byte-level **BPE from scratch** in `src/bpe.py` (minbpe-style). Add
   `tests/test_bpe.py` asserting `decode(encode(x)) == x` on held-out samples; **all tests must pass**.
2. Train three tokenizers into `pilot/tokenizers/`: `general_4096` (on the general corpus),
   `domain_4096` (on CT.gov train split), and `char` (byte/char-level, vocab ~100). Save merges+vocab.

## Stage 3 — Fertility analysis (MODEL-FREE — do this immediately, it is the Day-1 result)
- Tokenize `probe_terms.txt` and the test split with each tokenizer; compute tokens/term and
  tokens/word. Write `pilot/results/fertility.csv` and a bar chart `pilot/results/fertility.png`
  (domain vs general vs char). This result holds even if the trained-model signal is weak.

## Stage 4 — Model + training loop
- Implement a minimal decoder-only GPT in `src/model.py` (token+positional embeddings, N transformer
  blocks, tied output head) and `src/train.py` with checkpoint/resume, seed control, and a **BPB
  evaluator** on the test split.
- Config by tier — CPU: `d_model=128, n_layer=4, n_head=4, ctx=128`; GPU: `d_model=256, n_layer=6,
  n_head=8, ctx=256`; both `vocab=4096`. `--quick` shrinks steps to a few hundred.

## Stage 5 — Controlled micro-runs (compute-matched)
- Train the matrix: conditions {`domain_4096`, `general_4096`} × seeds {0,1,2}, plus one `char` run,
  **same corpus, same epochs**. Long runs → background; checkpoint so re-invocation resumes.
- Record per-run BPB (test), wall-clock, tokens/sec, peak memory into `pilot/results/runs.csv`.
- Also benchmark tokens/sec + memory to **recommend the full-project model tier**.

## Stage 6 — Analysis + verdict (the final result)
Compute mean ± SD BPB per condition and the domain-vs-general gap. Apply the **pre-registered gates**:
- 🟢 **GREEN**  — `(BPB_general_mean − BPB_domain_mean) > 2 × max(SD_domain, SD_general)`
  **and** domain fertility on probe terms ≥ ~15% lower → full quantitative embedded result is viable.
- 🟡 **AMBER** — fertility gap clear but BPB gap within noise → proceed, lean the embedded result on
  fertility + embedding-clustering, soften the BPB claim, consider scaling one tier.
- 🔴 **RED**   — neither signal → keep the dashboard as pure education, drop the quantitative
  tokenizer thread, and flag for human review.

Write **`pilot/RESULTS.md`** containing: the verdict banner (GREEN/AMBER/RED, or QUICK), the fertility
table + chart, the BPB table (mean ± SD per condition, gap, threshold), the tokens/sec benchmark and
recommended model tier, the exact configs/seeds used, data provenance, and a "what this shows / what it
does NOT show" honesty paragraph (small scale; may attenuate at frontier scale). Print the verdict
banner to the console as the final action.

## Stop and ask the human only if
- The CT.gov pull returns far less data than expected (< ~5 MB train) — report and ask before proceeding.
- The verdict is 🔴 RED — surface it prominently and recommend reassessment rather than silently continuing.
- A dependency cannot be installed on this machine — report the blocker; do not fabricate results.

Never invent numbers. If a stage did not run, say so in RESULTS.md. Reproducibility is the whole point.
