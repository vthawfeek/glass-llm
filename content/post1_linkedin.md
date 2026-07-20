<!-- DRAFT for your review. Series post 1 (intro). Pre-publication gate: model size stated;
mechanism not intelligence; prior art nod; numbers verifiable in repo; no medical advice / SYNTHETIC framing.
Attach (upload as NATIVE VIDEO, not a link): demo_media/square/01_pipeline_no_black_box.mp4 (1:1, the whole flow, no black box). Alt: square/04_attention_layers_and_head_compare.mp4 or square/06_temperature_cautious_vs_creative.mp4. -->

𝗬𝗼𝘂 𝗰𝗮𝗻'𝘁 𝘀𝗲𝗲 𝗶𝗻𝘀𝗶𝗱𝗲 𝗮 𝗹𝗮𝗻𝗴𝘂𝗮𝗴𝗲 𝗺𝗼𝗱𝗲𝗹, 𝘀𝗼 𝗜 𝗯𝘂𝗶𝗹𝘁 𝗼𝗻𝗲 𝘆𝗼𝘂 𝗰𝗮𝗻.

It's a decoder-only transformer and a BPE tokenizer, both 𝘄𝗿𝗶𝘁𝘁𝗲𝗻 𝗳𝗿𝗼𝗺 𝘀𝗰𝗿𝗮𝘁𝗰𝗵 (about 𝟭.𝟯𝗠 𝗽𝗮𝗿𝗮𝗺𝗲𝘁𝗲𝗿𝘀), trained on ClinicalTrials.gov text. I wrapped it in a dashboard with 𝗳𝗶𝘃𝗲 𝗰𝗼𝗻𝘁𝗿𝗼𝗹𝘀, and each one changes something you can watch:

1. Training data: clinical trials vs general English
2. Data volume: 1 MB, 8 MB, or 32 MB
3. Attention heads: 4 or 8
4. Fine-tuning: on or off (biomarker tagging)
5. Retrieval (RAG): on or off

Change any one of them and the tokenization, the embedding map, the attention heatmaps, the next-token probabilities, and the generated text 𝗮𝗹𝗹 𝘂𝗽𝗱𝗮𝘁𝗲 𝘄𝗶𝘁𝗵 𝘁𝗵𝗲 𝗺𝗼𝗱𝗲𝗹'𝘀 𝗿𝗲𝗮𝗹 𝗻𝘂𝗺𝗯𝗲𝗿𝘀. And you don't just see the new state: 𝗳𝗹𝗶𝗽 𝗮 𝗰𝗼𝗻𝘁𝗿𝗼𝗹 𝗮𝗻𝗱 𝘁𝗵𝗲 𝘀𝘁𝗲𝗽 𝗶𝘁 𝗮𝗳𝗳𝗲𝗰𝘁𝘀 𝘀𝗵𝗼𝘄𝘀 𝘁𝗵𝗲 𝗯𝗲𝗳𝗼𝗿𝗲 𝗮𝗻𝗱 𝗮𝗳𝘁𝗲𝗿 𝘀𝗶𝗱𝗲 𝗯𝘆 𝘀𝗶𝗱𝗲. You can 𝘀𝘁𝗲𝗽 𝘁𝗵𝗲 𝗴𝗲𝗻𝗲𝗿𝗮𝘁𝗶𝗼𝗻 𝗼𝗻𝗲 𝘁𝗼𝗸𝗲𝗻 𝗮𝘁 𝗮 𝘁𝗶𝗺𝗲, watching the next-token odds sharpen or flatten as you move the temperature. Hover the diagram and each step shows its 𝗺𝗮𝘁𝗵 𝗶𝗻 𝗽𝗹𝗮𝗶𝗻 𝗘𝗻𝗴𝗹𝗶𝘀𝗵.

The model is 𝗱𝗲𝗹𝗶𝗯𝗲𝗿𝗮𝘁𝗲𝗹𝘆 𝘁𝗶𝗻𝘆, so this was never about making it smart. Everything it writes is watermarked 𝗦𝗬𝗡𝗧𝗛𝗘𝗧𝗜𝗖 𝗮𝗻𝗱 𝗶𝘀 𝗻𝗼𝘁 𝗺𝗲𝗱𝗶𝗰𝗮𝗹 𝗶𝗻𝗳𝗼𝗿𝗺𝗮𝘁𝗶𝗼𝗻. What I wanted was to 𝘀𝗲𝗲, 𝗻𝗼𝘁 𝗷𝘂𝘀𝘁 𝗮𝘀𝘀𝗲𝗿𝘁, what each design choice does.

Good interactive LLM explainers already exist; Transformer Explainer is one. The part I hadn't seen in one place is the 𝘄𝗵𝗼𝗹𝗲 𝗽𝗶𝗽𝗲𝗹𝗶𝗻𝗲 (from-scratch tokenizer and model, data and architecture controls, fine-tuning, retrieval) on the kind of text a drug-discovery team actually works with, with 𝗲𝘃𝗲𝗿𝘆 𝗻𝘂𝗺𝗯𝗲𝗿 𝗿𝗲𝗽𝗿𝗼𝗱𝘂𝗰𝗶𝗯𝗹𝗲.

A few honest findings this week, then the live demo.

#MachineLearning #LLM #NLP #Interpretability #DrugDiscovery

<!-- First comment (per plan, lines 66-69): keep the body link-free; drop the write-up link here. Live-app link is held for the launch (post 4) by design. -->
First comment:
Full write-up: https://rokpayprsizors.wordpress.com/2026/07/17/two-things-a-1-3-million-parameter-clinical-language-model-taught-me/
