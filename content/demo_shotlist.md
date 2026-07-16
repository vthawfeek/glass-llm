# Demo video shot-list (60-90s, screen recording plus optional voiceover)

Goal: show transparency, not intelligence. Lead with the five controls, and land two moments: memorise vs generalise, and invent vs ground. Record at 1080p, hide the bookmarks bar, and keep the SYNTHETIC banner visible in at least one frame.

| # | Time | On screen | Say (voiceover / caption) |
|---|---|---|---|
| 1 | 0:00-0:08 | App header plus the SYNTHETIC warning; sweep the five-dropdown left panel | "A language model on clinical-trial text, built from scratch, with five controls you can turn and watch every step change." |
| 2 | 0:08-0:22 | Model explorer: the input text, tokens, vectors, attention, text Sankey. Then the Tokenizer-audit tab: paste drug names, show GPT-4, GPT-4o, GPT-2, BERT | "Before any model reasons, it splits text into tokens. GPT-4 reads pembrolizumab as six fragments; my clinical tokenizer, four. Cost and context are facts, quality is a flag." |
| 3 | 0:22-0:40 | Data volume control. Switch Low (1 MB) to High (32 MB) on the domain model; show the generation and the Test-BPB on the card | "Here is the trap: the 1 MB model sounds more fluent, but it memorised its data. Held-out bits-per-byte: 2.24 at 1 MB against 1.47 at 32 MB. Fluent is not the same as learned." |
| 4 | 0:40-0:52 | Attention control (4 to 8 heads), then Fine-tuned set to Yes on the domain model: biomarker tokens light up | "Turn the architecture control and the attention maps change. Turn on fine-tuning and a task head tags biomarkers, a mechanism demo, not an extractor." |
| 5 | 0:52-1:10 | RAG set to Yes. Same prompt: an invented trial on the left, real trials with NCT IDs and similarity scores on the right | "Ask the 1.3-million-parameter model and it invents a fluent, fake trial. Turn on retrieval and the answer is rebuilt from real trials, with NCT IDs. Watch it hallucinate, then watch grounding fix it." |
| 6 | 1:10-1:25 | Zoom on the URL or a "Try it" card | "It is live and open-source, and every number is reproducible. Link below." |

Editing notes:
- The two key moments are the volume switch (memorise vs generalise) and the invent-then-ground contrast. Add a half-second pause on each.
- Keep at least one frame with the SYNTHETIC and "not medical information" banner.
- Export under 100 MB so it embeds natively in the LinkedIn launch post (post 4).
- End card: Space URL, repo URL, and "SYNTHETIC: interpretability demo, not medical information."
- A topic or keyword prompt (a condition or drug) retrieves best in the RAG shot; the long default criterion still works but matches more loosely.
