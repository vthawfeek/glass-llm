# LinkedIn publishing plan: Glass-LLM launch

Assets in hand: 6 LinkedIn posts (drafted, em-dash-free), 2 blogs (live on WordPress), and the
demo media in `demo_media/`. The media set: 8 captioned clips, 8 square (1:1) clips, the 63s launch
reel in both landscape and 1:1, and 8 GIFs. Goal: land the work with a scientific / biomedical-NLP /
drug-discovery audience without over-posting.

## The one rule that drives everything
**One post = one idea = one clip.** The video *shows* the demo; the text says *why it matters* and
gives the one number; the blog holds the depth (linked in the first comment). Don't narrate the
video in the caption. If the clip shows it, cut it from the text. Front-load the payoff into the
first line, because LinkedIn hides everything after ~140 characters behind "...see more".

## Content map (what pairs with what)

The five feature posts follow the launch-reel order (attention, reel feature 5, is held for the
tail), and the launch closes the series.

| Post | Idea | Clip | Blog it leans on |
|------|------|------|------------------|
| 1 · Intro | "You can't see inside an LLM, so I built one you can" | `01_pipeline_no_black_box` | Blog 1 (teaser) |
| 2 · Clinical vs English tokenizer | Same sentence, equal 4k vocab: 27 vs 43 tokens | `02_tokenizer_clinical_vs_english` | Blog 1 (tokenization) |
| 3 · How GPT-4 sees a drug name | GPT-4 reads *pembrolizumab* as 6 fragments (cost = fact, quality = flag) | `08_tokenizer_audit_vs_gpt4` | Blog 1 (tokenization) |
| 4 · Fluent is not learned | Fluent is not the same as learned (1.47 vs 2.24 BPB) | `03_data_volume_memorise_vs_generalise` | Blog 1 |
| 5 · Hallucinate, then ground it | Watch it invent a trial, then watch retrieval ground it | `07_rag_hallucinate_vs_ground` | Blog 2 |
| 6 · Launch | It's live: five controls, every step visible | `00_launch_reel_63s` | Blog 2 (full write-up) |

Held back for a post-launch tail: `04_attention`, `05_generation`, `06_temperature`.

## The calendar (3-week Tue/Thu arc)

Two posts a week, 48 h apart, launch as the finale. Concrete dates below; shift the start week if
you prefer, but keep the Tue/Thu spacing.

| Slot | Date (2026) | Post | Media | Notes |
|------|-------------|------|-------|-------|
| 1 | **Tue Jul 21** | Post 1 (intro) | `square/01_pipeline…` | Sets up the series; no hard CTA |
| 2 | **Thu Jul 23** | Post 2 (clinical vs English) | `square/02_tokenizer_clinical…` | The controlled, equal-vocab win |
| 3 | **Tue Jul 28** | Post 3 (GPT-4 drug name) | `square/08_tokenizer_audit…` | The concrete, provocative fact |
| 4 | **Thu Jul 30** | Post 4 (fluent is not learned) | `square/03_data_volume…` | The shareable, counterintuitive one |
| 5 | **Tue Aug 4** | Post 5 (hallucinate, then ground) | `square/07_rag_hallucinate…` | The honest limit, then the fix |
| 6 | **Thu Aug 6** | Post 6 (LAUNCH) | `square/00_launch_reel_63s` (mobile) or `mp4/…` (desktop) | The peak. Pin it. Live link + blog + code |

Launch on Thu Aug 6 leaves a buffer before the Aug 11 deadline. Note: Post 1's line "a few honest
findings this week" reads fine treated loosely across the arc; if it bothers you, change it to "a
few honest findings over the next few weeks."

## Time of day
- **Days:** Tuesday and Thursday. Best for a professional audience. Avoid Monday mornings, Friday
  afternoons, and weekends.
- **Time:** post between **8:30 and 9:30 a.m. in the timezone where most of your audience sits.**
  If your reviewers are split US and Europe, ~**8:30 a.m. ET** (about 1:30 p.m. UK) catches European
  afternoon and the US morning at once. Confirm with LinkedIn's *your follower activity* view if you
  have it; otherwise the Tue/Thu 9 a.m. slot is a safe default.
- **Warm the app before the launch:** Streamlit Community Cloud sleeps after inactivity. Open
  glass-llm.streamlit.app ~15 min before Post 6 so the first visitors don't hit the "wake this app"
  screen, and revisit it during the first hour.

## The golden hour (per post)
- Be free for **60-90 minutes after posting** to reply to every comment. Early engagement drives reach.
- **Don't edit the post in the first hour.** Edits can suppress distribution, so proof-read first.
- Seed the conversation: end each post with a genuine question ("where have you seen this bite?") and
  answer the first serious replies with substance, not just thanks.

## Format & link mechanics
- **Upload the MP4 natively** (never a YouTube link). LinkedIn autoplays muted, which is exactly why
  the captions are burned in.
- **Aspect ratio:** use the **`square/` (1:1)** files for posts 1-5. They take the most feed space on
  mobile and keep the title and caption bands on screen the whole time. The launch reel now exists in
  both **landscape** (`mp4/00_launch_reel_63s.mp4`, best on desktop) and **1:1**
  (`square/00_launch_reel_63s.mp4`, best for the mobile feed, with a persistent SYNTHETIC + URL
  watermark). Post the 1:1 unless you know your audience is desktop-heavy.
- **Links kill reach in the post body.** For posts 1-5, keep the body link-free and drop the blog
  link in your **own first comment** ("full write-up in the comments"). For Post 6 (the launch), you
  have three links; put the **live-demo link in the body** (it's the whole point) and the **blog and
  code links in the first comment**. Pin Post 6 to your profile.
- **Hashtags:** 3-5, which your drafts already have. Keep the mix of broad (#MachineLearning #LLM
  #NLP) and niche (#Interpretability #ClinicalNLP #DrugDiscovery #Tokenization #RAG).

## How much text per post (right-sizing)
Your drafts are close. Trim to: **hook line, then the one number, then the honest caveat, then a soft
CTA**, with a blank line between each so it scans on mobile. Specifics:
- **Post 1:** strong as is. It's the "manifesto", so the length is earned. End on the series promise.
- **Post 2:** keep it to the equal-vocab comparison (27 vs 43, and 30 vs 55 on a criterion). Leave the
  GPT-4 numbers for Post 3. The "matched conditions, not 'domain always wins'" line is the honesty.
- **Post 3:** long but substantive; the honesty ("it is not 'beats GPT-4'") is the credibility-maker,
  so keep it. PubMedBERT citation now lives in the first comment to tighten the body.
- **Post 4:** the plain version leads with "fluent is not the same as learned" and leaves the exact
  scores for the write-up (raw numbers read as jargon in-feed). This is your most reshareable post;
  make sure the hook and the trap land above the fold.
- **Post 5:** lead with invent-then-ground; keep the NCT IDs and similarity detail and the honest
  "a model this small shows the mechanism of grounding, not a QA system."
- **Post 6:** the launch. Lead with the hallucinate-vs-ground hook, then the two honest limits
  (grounding mechanism, fine-tuning can't extract). Links as above.

## The post-launch tail (keep momentum without fatigue)
Don't cram the spare clips into the launch weeks. Run them as **one lightweight follow-up a
week** after launch, so the campaign has a long tail and the algorithm keeps you visible:
- Week of Aug 11: `06_temperature`, "one slider between a cautious model and a wild one."
- Week of Aug 18: `04_attention`, "watch attention sharpen with depth; 4 heads vs 8."
- Later, or as replies: `05_generation` as a GIF inside comments on the relevant posts.
- About a week after launch: a short reflection post ("what I learned building an LLM from scratch")
  that reshares the reel, good for the people who missed the launch.

## Pre-post checklist (run before each one)
- [ ] Model size stated (~1.3M params) and framed as mechanism, not intelligence.
- [ ] Cost/context = fact, quality = flag (Posts 2-3 especially); no "beats GPT-4" overclaim.
- [ ] Prior art nodded to (Transformer Explainer; PubMedBERT for tokenization).
- [ ] SYNTHETIC / not-medical-information line present.
- [ ] Every number matches the repo.
- [ ] (Post 6) live link, blog link, code link all open correctly, and the app is warm.
- [x] Consistency: the six post texts and every video overlay are em-dash-free. (Still open, your
      call: the app's own on-screen UI copy uses em dashes that show in the footage; removing those
      needs an app edit plus a re-record and redeploy.)
