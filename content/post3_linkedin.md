<!-- DRAFT for your review. Series post 3 (the tokenizer tool, honestly). Gate: cost/context = fact,
quality = flag; no overclaim (GPT-4 is competitive); prior art; numbers match repo; no medical advice.
Attach (upload as NATIVE VIDEO, not a link): demo_media/square/08_tokenizer_audit_vs_gpt4.mp4 (1:1, audit table + bar chart vs GPT-4/4o/2/BERT). In a follow-up comment, drop demo_media/gif/02_tokenizer_clinical_vs_english.gif (same text through both tokenizers, 27 vs 43). -->

GPT-4 doesn't read the word "pembrolizumab". 𝗜𝘁 𝗿𝗲𝗮𝗱𝘀 𝗽, 𝗲𝗺𝗯, 𝗿𝗼𝗹, 𝗶𝘇, 𝘂𝗺, 𝗮𝗯: 𝘀𝗶𝘅 𝗳𝗿𝗮𝗴𝗺𝗲𝗻𝘁𝘀. Here is what that costs, and what it doesn't.

Before any model reasons about text, it 𝘀𝗽𝗹𝗶𝘁𝘀 𝗶𝘁 𝗶𝗻𝘁𝗼 𝘁𝗼𝗸𝗲𝗻𝘀. I ran cancer-drug names through real tokenizers:

- GPT-4 / GPT-3.5: 6 tokens
- GPT-4o: 5
- GPT-2, BERT: 6
- 𝗔 𝘁𝗼𝗸𝗲𝗻𝗶𝘇𝗲𝗿 𝗜 𝘁𝗿𝗮𝗶𝗻𝗲𝗱 𝗼𝗻 𝗰𝗹𝗶𝗻𝗶𝗰𝗮𝗹 𝘁𝗿𝗶𝗮𝗹𝘀 (vocab 4,096): 𝟰

𝗖𝗼𝘀𝘁 𝗮𝗻𝗱 𝗰𝗼𝗻𝘁𝗲𝘅𝘁 𝗮𝗿𝗲 𝘀𝗶𝗺𝗽𝗹𝗲 𝗳𝗮𝗰𝘁𝘀. You pay per token and your context window is counted in tokens. On a full eligibility criterion, my domain tokenizer used 30 tokens against 55 for a general-English tokenizer of the same vocabulary size: 𝗿𝗼𝘂𝗴𝗵𝗹𝘆 𝟰𝟱% 𝗰𝗵𝗲𝗮𝗽𝗲𝗿, with more room for real content.

"𝗗𝗼𝗺𝗮𝗶𝗻 𝗮𝗹𝘄𝗮𝘆𝘀 𝘄𝗶𝗻𝘀" 𝗶𝘀 𝘄𝗿𝗼𝗻𝗴, and it's worth being exact about that. GPT-4's 100k-plus vocabulary is 𝗴𝗲𝗻𝘂𝗶𝗻𝗲𝗹𝘆 𝗰𝗼𝗺𝗽𝗲𝘁𝗶𝘁𝗶𝘃𝗲. On a 90-term mixed probe set it averaged 4.22 tokens per term against my domain tokenizer's 4.94. A big vocabulary buys back most of the specialised advantage. The clean, controlled result is 𝗱𝗼𝗺𝗮𝗶𝗻 𝗮𝗴𝗮𝗶𝗻𝘀𝘁 𝗴𝗲𝗻𝗲𝗿𝗮𝗹 𝗮𝘁 𝗲𝗾𝘂𝗮𝗹 𝘃𝗼𝗰𝗮𝗯 (about 20-45% fewer tokens). It is 𝗻𝗼𝘁 "𝗯𝗲𝗮𝘁𝘀 𝗚𝗣𝗧-𝟰".

And 𝗾𝘂𝗮𝗹𝗶𝘁𝘆 𝗶𝘀 𝗮 𝗳𝗹𝗮𝗴, 𝗻𝗼𝘁 𝗮 𝘃𝗲𝗿𝗱𝗶𝗰𝘁. Heavy fragmentation is a risk to test for rare terms and exact-match tasks. It is 𝗻𝗼𝘁 proof that the model understands the term worse, because large models absorb a lot of it.

The dashboard has a panel where you 𝗽𝗮𝘀𝘁𝗲 𝘆𝗼𝘂𝗿 𝗼𝘄𝗻 𝘁𝗲𝗿𝗺𝘀 and see how GPT-4, GPT-4o, GPT-2, and BERT split them, and what that costs. Alongside it, the model view runs the same text through the 𝗰𝗹𝗶𝗻𝗶𝗰𝗮𝗹 𝗮𝗻𝗱 𝘁𝗵𝗲 𝗴𝗲𝗻𝗲𝗿𝗮𝗹 𝘁𝗼𝗸𝗲𝗻𝗶𝘇𝗲𝗿 𝗮𝘁 𝗼𝗻𝗰𝗲, so you can watch a drug name stay whole or shatter. It's a 𝗰𝗵𝗲𝗮𝗽 𝗰𝗵𝗲𝗰𝗸 before trusting an LLM on clinical text, and it's why PubMedBERT trained its vocabulary from scratch (arxiv.org/abs/2007.15779).

#LLM #NLP #DrugDiscovery #ClinicalNLP #Tokenization

<!-- First comment (per plan, lines 66-69): keep the body link-free; drop the write-up link here. Optional: move the PubMedBERT citation from the body into this comment too (plan line 80). Live-app link is held for the launch (post 4) by design. -->
First comment:
Full write-up: https://rokpayprsizors.wordpress.com/2026/07/17/two-things-a-1-3-million-parameter-clinical-language-model-taught-me/
