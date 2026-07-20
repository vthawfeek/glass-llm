<!-- DRAFT for your review. Series post 2 (the counterintuitive finding). Gate: model size;
mechanism not intelligence; no medical advice; numbers match repo.
Attach (upload as NATIVE VIDEO, not a link): demo_media/square/03_data_volume_memorise_vs_generalise.mp4 (1:1, the held-out BPB gauge flipping green<->red as volume changes). -->

I trained the same tiny model on 1 MB and on 32 MB of clinical trials. 𝗧𝗵𝗲 𝘀𝗺𝗮𝗹𝗹-𝗱𝗮𝘁𝗮 𝘃𝗲𝗿𝘀𝗶𝗼𝗻 𝘀𝗼𝘂𝗻𝗱𝗲𝗱 𝗺𝗼𝗿𝗲 𝗳𝗹𝘂𝗲𝗻𝘁. 𝗧𝗵𝗮𝘁 𝘁𝘂𝗿𝗻𝘀 𝗼𝘂𝘁 𝘁𝗼 𝗯𝗲 𝘁𝗵𝗲 𝘁𝗿𝗮𝗽.

Same architecture (about 1.3M parameters), same tokenizer, same 4,000 training steps. The 𝗼𝗻𝗹𝘆 𝘁𝗵𝗶𝗻𝗴 𝗜 𝗰𝗵𝗮𝗻𝗴𝗲𝗱 was how much data the model saw. Measured on 𝗵𝗲𝗹𝗱-𝗼𝘂𝘁 𝘁𝗿𝗶𝗮𝗹𝘀 𝗶𝘁 𝗻𝗲𝘃𝗲𝗿 𝘁𝗿𝗮𝗶𝗻𝗲𝗱 𝗼𝗻, in bits-per-byte (lower is better):

- 𝟭 𝗠𝗕: 𝟮.𝟮𝟰
- 𝟴 𝗠𝗕: 𝟭.𝟰𝟵
- 𝟯𝟮 𝗠𝗕: 𝟭.𝟰𝟳

The 1 MB model produces clean lines like "Age >= 18 to 80 years old", so 𝗯𝘆 𝗲𝘆𝗲 𝘆𝗼𝘂 𝘄𝗼𝘂𝗹𝗱 𝗰𝗮𝗹𝗹 𝗶𝘁 𝘁𝗵𝗲 𝗯𝗲𝘀𝘁 𝗼𝗳 𝘁𝗵𝗲 𝘁𝗵𝗿𝗲𝗲. 𝗜𝘁 𝗶𝘀𝗻'𝘁. With only 1 MB and 4,000 steps it passes over that tiny corpus about 30 times and 𝗺𝗲𝗺𝗼𝗿𝗶𝘀𝗲𝘀 𝗶𝘁. It isn't generating, 𝗶𝘁'𝘀 𝗿𝗲𝗰𝗶𝘁𝗶𝗻𝗴, and on text it hasn't seen 𝗶𝘁 𝗱𝗼𝗲𝘀 𝘁𝗵𝗲 𝘄𝗼𝗿𝘀𝘁.

This is why we 𝗺𝗲𝗮𝘀𝘂𝗿𝗲 𝗵𝗲𝗹𝗱-𝗼𝘂𝘁 𝗽𝗲𝗿𝗽𝗹𝗲𝘅𝗶𝘁𝘆 instead of trusting how the output reads. 𝗙𝗹𝘂𝗲𝗻𝘁 𝗶𝘀 𝗻𝗼𝘁 𝘁𝗵𝗲 𝘀𝗮𝗺𝗲 𝗮𝘀 𝗹𝗲𝗮𝗿𝗻𝗲𝗱. A model that has memorised its training set demos beautifully and generalises badly, and in a small-data setting (a rare disease, a new target, one site's records) 𝘁𝗵𝗮𝘁 𝘁𝗿𝗮𝗽 𝗶𝘀 𝘁𝗵𝗲 𝗱𝗲𝗳𝗮𝘂𝗹𝘁 rather than the exception.

In the dashboard, all three sizes sit on one held-out bits-per-byte scale, with the size you pick highlighted and a plain 𝗺𝗲𝗺𝗼𝗿𝗶𝘀𝗲𝗱 𝘃𝗲𝗿𝘀𝘂𝘀 𝗴𝗲𝗻𝗲𝗿𝗮𝗹𝗶𝘀𝗲𝘀 verdict, so you see the trap instead of trusting how fluent the output looks.

The model is about 1.3M parameters and everything it writes is watermarked 𝗦𝗬𝗡𝗧𝗛𝗘𝗧𝗜𝗖. 𝗜𝘁 𝗶𝘀 𝗻𝗼𝘁 𝗺𝗲𝗱𝗶𝗰𝗮𝗹 𝗶𝗻𝗳𝗼𝗿𝗺𝗮𝘁𝗶𝗼𝗻.

#MachineLearning #LLM #NLP #Interpretability #DataScience

<!-- First comment (per plan, lines 66-69): keep the body link-free; drop the write-up link here. Live-app link is held for the launch (post 4) by design. -->
First comment:
Full write-up: https://rokpayprsizors.wordpress.com/2026/07/17/two-things-a-1-3-million-parameter-clinical-language-model-taught-me/
