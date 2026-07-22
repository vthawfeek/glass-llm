<!-- DRAFT for your review. Series post 3 (feature 3: how GPT-4 sees a drug name). PLAIN-ENGLISH rewrite for a general audience. Gate: cost/context = fact, quality = flag; GPT-4 is competitive (NO "beats GPT-4"); prior art (PubMedBERT) in the first comment for advanced readers; not medical. Attach (upload as NATIVE VIDEO, not a link): demo_media/square/08_tokenizer_audit_vs_gpt4.mp4 (1:1, the term-by-term comparison vs GPT-4/4o/2/BERT). -->

To the AI you use, the cancer drug "pembrolizumab" is not one word. 𝗚𝗣𝗧-𝟰 𝘀𝗲𝗲𝘀 𝗶𝘁 𝗮𝘀 𝘀𝗶𝘅 𝗹𝗶𝘁𝘁𝗹𝗲 𝗳𝗿𝗮𝗴𝗺𝗲𝗻𝘁𝘀: 𝗽, 𝗲𝗺𝗯, 𝗿𝗼𝗹, 𝗶𝘇, 𝘂𝗺, 𝗮𝗯.

Remember, more fragments means more cost and less room. So on unusual words like drug names and gene names, 𝘀𝗽𝗲𝗰𝗶𝗮𝗹𝗶𝘀𝘁 𝘁𝗲𝘅𝘁 𝗰𝗮𝗻 𝗾𝘂𝗶𝗲𝘁𝗹𝘆 𝗴𝗲𝘁 𝗲𝘅𝗽𝗲𝗻𝘀𝗶𝘃𝗲.

But here is the part people get wrong. "𝗔 𝘀𝗽𝗲𝗰𝗶𝗮𝗹𝗶𝘀𝘁 𝘁𝗼𝗼𝗹 𝗮𝗹𝘄𝗮𝘆𝘀 𝘄𝗶𝗻𝘀" 𝗶𝘀 𝗻𝗼𝘁 𝘁𝗿𝘂𝗲. GPT-4's built-in word list is huge, and on a mixed batch of 90 tricky terms it was basically neck and neck with my medical version (about 4.2 fragments per term against my 4.9). A big word list buys back most of the advantage, so I would never claim this "beats GPT-4."

And one more honest point. 𝗟𝗼𝘁𝘀 𝗼𝗳 𝗳𝗿𝗮𝗴𝗺𝗲𝗻𝘁𝘀 𝗶𝘀 𝗮 𝘆𝗲𝗹𝗹𝗼𝘄 𝗳𝗹𝗮𝗴, 𝗻𝗼𝘁 𝗮 𝘃𝗲𝗿𝗱𝗶𝗰𝘁. It is a reason to double-check the AI on rare, exact terms. It is not proof the AI understands the word worse, because big models cope with it surprisingly well.

So I built a small tool where 𝘆𝗼𝘂 𝗽𝗮𝘀𝘁𝗲 𝘆𝗼𝘂𝗿 𝗼𝘄𝗻 𝘁𝗲𝗿𝗺𝘀 𝗮𝗻𝗱 𝘀𝗲𝗲, 𝘀𝗶𝗱𝗲 𝗯𝘆 𝘀𝗶𝗱𝗲, 𝗵𝗼𝘄 𝘁𝗵𝗲 𝗯𝗶𝗴 𝗔𝗜𝘀 𝗰𝗵𝗼𝗽 𝘁𝗵𝗲𝗺 𝘂𝗽 𝗮𝗻𝗱 𝘄𝗵𝗮𝘁 𝘁𝗵𝗮𝘁 𝗰𝗼𝘀𝘁𝘀. A cheap sanity check before you trust an AI on specialist text.

The AI in my demo is tiny and see-through, and everything it writes is made up and is not medical advice.

#AI #LLM #DrugDiscovery #Tokenization

<!-- First comment: keep the body link-free; write-up + a citation for advanced readers here (tightens the body). Live-app link is held for the launch (post 6). -->
First comment:
Full write-up: https://rokpayprsizors.wordpress.com/2026/07/17/two-things-a-1-3-million-parameter-clinical-language-model-taught-me/
Why some teams build a specialist word list from scratch (PubMedBERT): https://arxiv.org/abs/2007.15779
