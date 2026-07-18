<!-- DRAFT for your review. Series post 1 (intro). Pre-publication gate: model size stated;
mechanism not intelligence; prior art nod; numbers verifiable in repo; no medical advice / SYNTHETIC framing.
Attach: a short screen capture of changing a dropdown and the panels updating. -->

You can't see inside a language model, so I built one you can.

It's a decoder-only transformer and a BPE tokenizer, both written from scratch (about 1.3M parameters), trained on ClinicalTrials.gov text. I wrapped it in a dashboard with five controls, and each one changes something you can watch:

1. Training data: clinical trials vs general English
2. Data volume: 1 MB, 8 MB, or 32 MB
3. Attention heads: 4 or 8
4. Fine-tuning: on or off (biomarker tagging)
5. Retrieval (RAG): on or off

Change any one of them and the tokenization, the embedding map, the attention heatmaps, the next-token probabilities, and the generated text all update with the model's real numbers. And you don't just see the new state: flip a control and the step it affects shows the before and after side by side. You can step the generation one token at a time, watching the next-token odds sharpen or flatten as you move the temperature. Hover the diagram and each step shows its math in plain English.

The model is deliberately tiny, so this was never about making it smart. Everything it writes is watermarked SYNTHETIC and is not medical information. What I wanted was to see, not just assert, what each design choice does.

Good interactive LLM explainers already exist; Transformer Explainer is one. The part I hadn't seen in one place is the whole pipeline (from-scratch tokenizer and model, data and architecture controls, fine-tuning, retrieval) on the kind of text a drug-discovery team actually works with, with every number reproducible.

A few honest findings this week, then the live demo.

#MachineLearning #LLM #NLP #Interpretability #DrugDiscovery #ClinicalTrials
