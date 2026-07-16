<!-- DRAFT. The launch post. Gate: model size + mechanism not intelligence; cost/context = fact,
quality = flag; prior art; SYNTHETIC watermark; no medical advice; live link works before posting.
Fill in the links. Attach: the 60-90s demo video. -->

Watch a language model invent a clinical trial, then watch retrieval stop it. Live, in a glass box.

For the last few weeks I've been building a language model on clinical-trial text from scratch: my own BPE tokenizer, my own transformer. Not to make it smart, but to make it transparent. It's live now, and you can click through every step.

Try it: <SPACE_URL>
Write-up: <BLOG_URL>
Code: <REPO_URL>

Five controls (data domain, data volume, attention heads, fine-tuning, retrieval) each reshape the model in front of you. Two things I'm most pleased with, because they're honest about the limits.

Hallucination vs grounding. Give the model a topic and it will write a fluent, completely invented trial. Turn on retrieval and the panel rebuilds the answer from real trials, shown with their NCT IDs and similarity scores. A model this small can't synthesise grounded prose, so the panel shows the mechanism of grounding and says as much.

Fine-tuning, honestly. I tried to fine-tune it to extract biomarkers (EGFR, PD-L1, HER2). It can't. Structured biomarker extraction is a task where the state of the art is fine-tuned 7B-parameter models (Alkhoury et al., npj Digital Medicine 2025). So the dashboard shows a fine-tuned task head tagging biomarker mentions, labelled clearly as a demo rather than an extractor.

Everything it generates is watermarked SYNTHETIC and is not medical information. Interactive LLM explainers already exist; Transformer Explainer is a good one. What's new here is the whole pipeline (data, architecture, fine-tuning, retrieval) on clinical-trial text, with every number reproducible.

Feedback from people who work in this area is exactly what I'm after.

#MachineLearning #LLM #NLP #DrugDiscovery #Interpretability #RAG
