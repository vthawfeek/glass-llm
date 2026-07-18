<!-- DRAFT for your review. Series post 2 (the counterintuitive finding). Gate: model size;
mechanism not intelligence; no medical advice; numbers match repo.
Attach: a side-by-side of the low-data vs high-data generation. -->

I trained the same tiny model on 1 MB and on 32 MB of clinical trials. The small-data version sounded more fluent. That turns out to be the trap.

Same architecture (about 1.3M parameters), same tokenizer, same 4,000 training steps. The only thing I changed was how much data the model saw. Measured on held-out trials it never trained on, in bits-per-byte (lower is better):

- 1 MB: 2.24
- 8 MB: 1.49
- 32 MB: 1.47

The 1 MB model produces clean lines like "Age >= 18 to 80 years old", so by eye you would call it the best of the three. It isn't. With only 1 MB and 4,000 steps it passes over that tiny corpus about 30 times and memorises it. It isn't generating, it's reciting, and on text it hasn't seen it does the worst.

This is why we measure held-out perplexity instead of trusting how the output reads. Fluent is not the same as learned. A model that has memorised its training set demos beautifully and generalises badly, and in a small-data setting (a rare disease, a new target, one site's records) that trap is the default rather than the exception.

In the dashboard, all three sizes sit on one held-out bits-per-byte scale — the size you pick highlighted, with a plain "memorised" versus "generalises" verdict — so you see the trap instead of trusting how fluent the output looks.

The model is about 1.3M parameters and everything it writes is watermarked SYNTHETIC. It is not medical information.

#MachineLearning #LLM #NLP #Interpretability #DataScience
