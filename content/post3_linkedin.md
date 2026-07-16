<!-- DRAFT for your review. Series post 3 (the tokenizer tool, honestly). Gate: cost/context = fact,
quality = flag; no overclaim (GPT-4 is competitive); prior art; numbers match repo; no medical advice.
Attach: the audit-panel table. -->

GPT-4 doesn't read the word "pembrolizumab". It reads p, emb, rol, iz, um, ab: six fragments. Here is what that costs, and what it doesn't.

Before any model reasons about text, it splits it into tokens. I ran cancer-drug names through real tokenizers:

- GPT-4 / GPT-3.5: 6 tokens
- GPT-4o: 5
- GPT-2, BERT: 6
- A tokenizer I trained on clinical trials (vocab 4,096): 4

Cost and context are simple facts. You pay per token and your context window is counted in tokens. On a full eligibility criterion, my domain tokenizer used 30 tokens against 55 for a general-English tokenizer of the same vocabulary size: roughly 45% cheaper, with more room for real content.

"Domain always wins" is wrong, and it's worth being exact about that. GPT-4's 100k-plus vocabulary is genuinely competitive. On a mixed probe set it averaged 4.75 tokens per term against my domain tokenizer's 5.75. A big vocabulary buys back most of the specialised advantage. The clean, controlled result is domain against general at equal vocab (about 20-45% fewer tokens). It is not "beats GPT-4".

And quality is a flag, not a verdict. Heavy fragmentation is a risk to test for rare terms and exact-match tasks. It is not proof that the model understands the term worse, because large models absorb a lot of it.

The dashboard has a panel where you paste your own terms and see how GPT-4, GPT-4o, GPT-2, and BERT split them, and what that costs. It's a cheap check before trusting an LLM on clinical text, and it's why PubMedBERT trained its vocabulary from scratch (arxiv.org/abs/2007.15779).

#LLM #NLP #DrugDiscovery #ClinicalNLP #Tokenization
