<!-- DRAFT. The launch post. Gate: model size + mechanism not intelligence; cost/context = fact,
quality = flag; prior art; SYNTHETIC watermark; no medical advice; live link works before posting.
Attach (upload as NATIVE VIDEO, not a link): demo_media/square/00_launch_reel_63s.mp4 (1:1, best for mobile; use demo_media/mp4/00_launch_reel_63s.mp4 landscape only if your audience is desktop-heavy). The 63s reel: black box -> tokenizer -> memorise/generalise -> attention -> hallucinate-vs-ground -> live link. Alt single clip: square/07_rag_hallucinate_vs_ground.mp4. -->

No product this time, 𝗷𝘂𝘀𝘁 𝘁𝗵𝗲 𝗽𝗮𝗿𝘁𝘀 𝗹𝗮𝗶𝗱 𝗯𝗮𝗿𝗲. 𝗪𝗮𝘁𝗰𝗵 𝗮 𝗹𝗮𝗻𝗴𝘂𝗮𝗴𝗲 𝗺𝗼𝗱𝗲𝗹 𝗶𝗻𝘃𝗲𝗻𝘁 𝗮 𝗰𝗹𝗶𝗻𝗶𝗰𝗮𝗹 𝘁𝗿𝗶𝗮𝗹, 𝘁𝗵𝗲𝗻 𝘄𝗮𝘁𝗰𝗵 𝗿𝗲𝘁𝗿𝗶𝗲𝘃𝗮𝗹 𝘀𝘁𝗼𝗽 𝗶𝘁. Live, in a glass box.

For the last few weeks I've been building a language model on clinical-trial text 𝗳𝗿𝗼𝗺 𝘀𝗰𝗿𝗮𝘁𝗰𝗵: my own BPE tokenizer, my own transformer, about 1.3M parameters. 𝗡𝗼𝘁 𝘁𝗼 𝗺𝗮𝗸𝗲 𝗶𝘁 𝘀𝗺𝗮𝗿𝘁, 𝗯𝘂𝘁 𝘁𝗼 𝗺𝗮𝗸𝗲 𝗶𝘁 𝘁𝗿𝗮𝗻𝘀𝗽𝗮𝗿𝗲𝗻𝘁. It's live now, and you can click through every step.

Try it: https://glass-llm.streamlit.app

𝗙𝗶𝘃𝗲 𝗰𝗼𝗻𝘁𝗿𝗼𝗹𝘀 (data domain, data volume, attention heads, fine-tuning, retrieval) each reshape the model in front of you. Every dial shows its effect as a 𝗯𝗲𝗳𝗼𝗿𝗲-𝗮𝗻𝗱-𝗮𝗳𝘁𝗲𝗿, not just a re-render, and you can 𝘀𝘁𝗲𝗽 𝘁𝗵𝗲 𝗴𝗲𝗻𝗲𝗿𝗮𝘁𝗶𝗼𝗻 𝗼𝗻𝗲 𝘁𝗼𝗸𝗲𝗻 𝗮𝘁 𝗮 𝘁𝗶𝗺𝗲 as the temperature sharpens or flattens the odds. Two things I'm most pleased with, because they're honest about the limits.

𝗛𝗮𝗹𝗹𝘂𝗰𝗶𝗻𝗮𝘁𝗶𝗼𝗻 𝘃𝘀 𝗴𝗿𝗼𝘂𝗻𝗱𝗶𝗻𝗴. Give the model a topic and it will write a 𝗳𝗹𝘂𝗲𝗻𝘁, 𝗰𝗼𝗺𝗽𝗹𝗲𝘁𝗲𝗹𝘆 𝗶𝗻𝘃𝗲𝗻𝘁𝗲𝗱 𝘁𝗿𝗶𝗮𝗹. Turn on retrieval and the panel 𝗿𝗲𝗯𝘂𝗶𝗹𝗱𝘀 𝘁𝗵𝗲 𝗮𝗻𝘀𝘄𝗲𝗿 𝗳𝗿𝗼𝗺 𝗿𝗲𝗮𝗹 𝘁𝗿𝗶𝗮𝗹𝘀, shown with their NCT IDs and similarity scores. A model this small can't synthesise grounded prose, so the panel shows the 𝗺𝗲𝗰𝗵𝗮𝗻𝗶𝘀𝗺 of grounding and says as much.

𝗙𝗶𝗻𝗲-𝘁𝘂𝗻𝗶𝗻𝗴, 𝗵𝗼𝗻𝗲𝘀𝘁𝗹𝘆. I tried to fine-tune it to extract biomarkers (EGFR, PD-L1, HER2). 𝗜𝘁 𝗰𝗮𝗻'𝘁. Structured biomarker extraction is a task where the state of the art is 𝗳𝗶𝗻𝗲-𝘁𝘂𝗻𝗲𝗱 𝟳𝗕-𝗽𝗮𝗿𝗮𝗺𝗲𝘁𝗲𝗿 𝗺𝗼𝗱𝗲𝗹𝘀 (Alkhoury et al., npj Digital Medicine 2025). So the dashboard shows a fine-tuned task head tagging biomarker mentions, 𝗹𝗮𝗯𝗲𝗹𝗹𝗲𝗱 𝗰𝗹𝗲𝗮𝗿𝗹𝘆 𝗮𝘀 𝗮 𝗱𝗲𝗺𝗼 rather than an extractor.

Everything it generates is watermarked 𝗦𝗬𝗡𝗧𝗛𝗘𝗧𝗜𝗖 𝗮𝗻𝗱 𝗶𝘀 𝗻𝗼𝘁 𝗺𝗲𝗱𝗶𝗰𝗮𝗹 𝗶𝗻𝗳𝗼𝗿𝗺𝗮𝘁𝗶𝗼𝗻. Interactive LLM explainers already exist; Transformer Explainer is a good one. What's new here is the 𝘄𝗵𝗼𝗹𝗲 𝗽𝗶𝗽𝗲𝗹𝗶𝗻𝗲 (data, architecture, fine-tuning, retrieval) on clinical-trial text, with 𝗲𝘃𝗲𝗿𝘆 𝗻𝘂𝗺𝗯𝗲𝗿 𝗿𝗲𝗽𝗿𝗼𝗱𝘂𝗰𝗶𝗯𝗹𝗲.

Feedback from people who work in this area is exactly what I'm after.

#MachineLearning #LLM #NLP #DrugDiscovery #RAG

<!-- First comment (per plan, lines 66-69): body carries only the live-demo link; blog + code go in your own first comment. Pin this post to your profile. -->
First comment:
Write-up: https://rokpayprsizors.wordpress.com/2026/07/17/a-glass-box-for-clinical-trial-language-models-five-controls-every-step-visible/
Code: https://github.com/vthawfeek/glass-llm
