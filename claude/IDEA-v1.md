# BitCliff — PLAN.md

The complete plan. Everything about the idea, the product, the measurements, the phases, the launch, and the money. No implementation detail; the technical layer lives in bitcliff-idea.md, bitcliff-questions-answered.md, and bitcliff-aws-budget.md.

---

## 1. What BitCliff is

**One line:** watch an LLM lose its mind as you drag the bits out, then get per-capability numbers on what each GGUF file actually deletes, so you know which one to download.

**The name is a definition.** For each capability on each model, **the cliff is the highest-precision file at which that capability is measurably damaged beyond the registered margin.** "Retrieval cliff: Q3_K_M" is a sentence, a badge, a table row, and the thing people will argue about. The name is load-bearing: cliffs drive the tables, the Picker, the durability comparison, and the recurring language of every post.

**Two jobs, one site:**
1. **Spectacle.** A slider that compresses a model in front of you until it forgets facts, loops, and finally speaks salad. This is marketing. It gets shared once.
2. **Reference.** Per-capability retention tables with pre-registered statistics, answering the question every local-LLM user has and nobody measures: which file do I download. This is the product. It gets linked forever.

The slider buys traffic. The tables keep it.

---

## 2. Purpose

**For the ecosystem.** GGUF selection runs on folklore. A model page shows fifteen files and their sizes; the only quality signals are perplexity and corpus KLD, both token-averaged, neither capability-aware. People pick by file size and Reddit memory. BitCliff replaces that with a place to see the damage with your own eyes and a place to measure it per capability. Invisible loss becomes visible, then quantified.

**For the research.** The site doubles as an instrument. Five questions, registered publicly before any confirmatory data exists:

1. Does the divergence metric everyone cites (corpus KLD) actually rank files in the order of their capability damage?
2. Does divergence measured at the moment of answering predict which specific questions a quant gets wrong?
3. Can the uncompressed model alone predict which of its own answers are fragile, before any quant is ever run?
4. Which direction does training-data contamination bend arithmetic measurements: does memorization make quantization damage look bigger than it is, or smaller?
5. Does the published finding (retrieval damaged ~3.2x more than arithmetic at 3-bit) transfer from the lab quantization schemes it was measured on to the k-quants people actually download?

**For me.** This operationalizes the published quantization research into a running public artifact: an interactive companion to the arXiv extension, a public pre-registration whose results visibly match it, a distribution asset, and one stronger line in every cold email. The floor outcome is guaranteed even if nothing goes viral.

---

## 3. Audiences

1. **Scrollers** (X, r/LocalLLaMA). Want 15 seconds of entertainment. Served by the slider, the failure gallery, and share cards. Converted by permalinks.
2. **Devs picking a GGUF for an 8B-class model.** Currently choose by file size and folklore. Served by the matrices, the cliffs, and the Picker.
3. **Quant uploaders.** The people whose HF pages the audience already lives on. Served by free, rigorous quality tables for their catalogs. The realistic win is one link line in an uploader's card template, which propagates across their entire catalog. Their adoption is upside, never a dependency.

---

## 4. The models and the ladder

**Three models, two roles:**

- **Spectacle: Qwen2.5-1.5B-Instruct.** Small models degrade dramatically, which is the content. Cheap enough to serve live. This is the only model with a live prompt box.
- **Reference pair: Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct.** The size class people quantize out of necessity; nobody agonizes over which 1.5B file to download, so the reference tables must be 8B-class or they answer nothing. Two models, size-comparable, so the cross-family durability comparison is legitimate. Precompute-only, no live inference.

**The ladder** runs from full precision down to the deranged zone: F16 (converted locally as the reference, never a download recommendation), then Q8, Q6, Q5, Q4, Q3, down to the lowest level the tracked uploaders actually publish, expected around IQ2_XXS, with IQ1_S included only where it exists.

**A quant level is a file, not a label.** "Q4_K_M" is not one thing; bartowski's, unsloth's, and mradermacher's Q4_K_M are different files with different calibration. Every output and every table row shows uploader, imatrix status, and the exact file hash. The canonical ladder uses bartowski's imatrix quants. Results are pinned to file hashes; when an uploader re-quantizes, pages get a "newer file exists" badge rather than silently pointing at stale data.

**One exception for spectacle:** if published 1.5B repos stop above the truly deranged levels, the missing bottom rungs get quantized in-house and clearly labeled as such. The published-files-only rule exists to protect download recommendations, and the 1.5B is not one.

---

## 5. The product, page by page

### Playground (home)

- Model tabs. 1.5B has a live prompt box; the 8B-class tabs are browse-only over precomputed outputs, with a banner saying so.
- **Ladder view:** one prompt, every quant level's output stacked, full precision at top. The screenshot format.
- **A/B view:** any two levels side by side with the slider between them. The slider snaps between levels; this is what the launch clip records.
- Every output card shows: quant label, uploader, a grade badge (correct / partial / wrong / truncated / gibberish), generation speed, and the first token where it diverged from full precision, highlighted.
- **Prompt sources:** a curated set of 50 (15 factual retrieval, 10 arithmetic word problems, 10 instruction adherence, 5 long-context needles, 10 pure-spectacle prompts that are labeled unscored entertainment) plus a prominent **Random button** that draws from the full evaluation set. The curated set is allowed to be dramatic precisely because the Random button exists; only the Random pool's integrity is sacred.
- Live box rules: a handful of prompts per hour per person, short outputs, visible queue position, and a graceful fallback where the whole site drops to precomputed-only mode if load spikes. Shared-IP users (VPNs, campus networks) may hit limits together; accepted at launch and said on the page.
- Every comparison has a stable permalink.

### The Numbers (per model)

One matrix per model. Rows are quant files; columns are per-capability retention versus full precision (with confidence intervals), the divergence metrics people already cite (so any mismatch is visible in a single screenshot), file size, estimated RAM, speed, uploader, imatrix status, and file hash.

**Every cell is one of four states, and the distinction is the product's honesty:**
1. **Damaged.** Statistically worse, and worse than the registered margin. Red. Cliffs are drawn here.
2. **Small real loss, within margin.** Statistically detectable, but smaller than the margin. Shows the signed loss, e.g. "-1.2pp." This is the expected verdict for the popular Q4 files, and it is a better headline than "free": "costs about a point, never more than three" is more credible and more quotable.
3. **Equivalent.** No detectable loss, and the data is strong enough to bound any loss under the margin. Green.
4. **Indeterminate.** Not enough evidence either way. Grey, and never rounded up to "free."

- **The cliff line** is drawn across the matrix where each capability first enters Damaged, with badges above the table: "Retrieval cliff: Q3_K_M. Arithmetic cliff: Q2_K." Odd cases where a lower file passes after a higher one failed are flagged, never smoothed.
- **Drill-down per cell shows both directions:** the questions full precision got right that the quant lost, AND the questions the quant randomly gained. Showing the gains closes the cherry-picking hole, and "the 2-bit file beats full precision on these five questions" is its own delightful post.
- Every table downloads as CSV and JSON.

### Durability

The two reference models' retention curves overlaid per capability, each normalized to its own full-precision baseline (which removes the differences in tokenizers, templates, and starting accuracy), with cliff positions compared directly. The claim is class-level ("7-8B"), the residual size gap is disclosed in one line, and the 1.5B appears nowhere on this page. This page exists because "which family takes a punch better" is a real argument people have with no data.

### The Picker (Phase 2)

Input: how much RAM you have and what you use the model for (retrieval-heavy chat, math and code, agents and long documents). Output: one specific file, with reasoning: the smallest file where every capability you care about is above its cliff with margin, tie-broken by speed, linking to the exact file on HF. Page titles match the literal search query ("Llama 3.1 8B GGUF comparison: which quant to download") because that query is the whole reference audience.

### Guess the Quant (Phase 1, if it earns its way in)

See one output, guess the compression level. Scored by distance (full points for exact, partial for adjacent, nothing otherwise) because the top of the ladder is genuinely indistinguishable and exact-only scoring would make the game feel broken. Ten rounds, shareable score card. It ships only if a blind test proves the mid-ladder is actually guessable by humans (see 8. Honesty features).

### Failure Gallery (Phase 3)

Community-submitted degradations from the live box, with voting and tags: forgot a fact, infinite loop, gibberish, ignored the instruction, botched the math. Loops and gibberish are tagged mechanically, not by mood. The "infinite loop" tag carries a note that loops are partly an artifact of the deterministic decoding the site uses for reproducibility; normal sampling escapes some of them. A monthly "best failures" post keeps the account alive between releases.

### Methodology and the public registration

A methodology page that states every test, every margin, every rule, in full, including the scope disclaimer that results are per model and size and must not be extrapolated. The pre-registration is rendered verbatim with its commit hash, timestamped by a third-party mechanism, so "you decided the rules after seeing the data" is impossible by construction.

### Share cards

Every permalink renders a proper social preview: the prompt, the outputs, the quant labels, the uploader, the site. Launch-blocking, not polish, because the card is the growth loop. One-tap copy-as-image for mobile. Every share URL carries tracking tags so the loop's health is measurable.

---

## 6. What gets measured, in plain language

**Three capability suites at launch** (two more, instruction following and long context, in Phase 2):

- **Retrieval:** the multivalue2 task from the published paper.
- **Arithmetic, twice.** The standard GSM8K problems, and a twin of each problem with the names and numbers changed and the answer recomputed, built so the twins match the originals in difficulty. The twins cannot have been memorized during training, because they didn't exist.

**Why twins matter (registered question 4).** If models partially memorized GSM8K, then "arithmetic damage" partly measures how memorization degrades, not how calculation degrades. Two stories are both plausible and point opposite ways: memorized answers might be fragile like other recall (making the paper's 3.2x asymmetry an underestimate) or robust like overlearned habits while multi-step calculation compounds noise (making it an overestimate). The registration commits to testing this two-sided, with no preferred outcome. The comparison is restricted to problems where the model at full precision solves both the original and the twin, because comparing on different populations would rig the result mechanically. A cut headline ships as its own honest post; that outcome fits the brand better than a padded one.

**One embargo:** the twin problems stay out of the public Random pool and out of the published dataset until the paper extension ships, because publishing them would leak the templates into future training data and to competitors. Their construction method is public from day one; the instantiated problems are not.

**The divergence questions (registered questions 1-3).** The community's current quality metric is a divergence score computed on Wikipedia text. The registration tests it at two levels: does it rank files correctly (its actual job), and does divergence measured at the moment of answering predict which individual questions flip. It also tests whether the full-precision model alone can predict its own fragile answers. All branches are informative: if the metric works, the site publishes the conversion table and a screening trick; if it fails, the site publishes exactly what it hides; if it only fails because it's computed on the wrong text, that is the sharpest post of all.

**Truncation is its own verdict.** Outputs get a generous length budget for evaluation; an answer that never arrives within it counts as wrong (an endless loop is damage), but truncation is tracked separately and reported per cell so "wrong" never silently absorbs "cut off." The pilot checks that the budget is genuinely generous for the full-precision model before the number is frozen.

**Reproducibility, honestly scoped.** All evaluation uses fully deterministic generation with every relevant setting recorded, and the site states plainly that determinism holds within a fixed hardware and software configuration, not across arbitrary machines. The 50 curated playground prompts are regenerated on the exact serving machine so that anyone re-running a curated prompt live sees the same output the page shows.

**The k-quant transfer question (registered question 5).** The paper's asymmetry was measured on lab quantization schemes. The downloadable k-quants are a different technique, and the headline number is not assumed to transfer; the launch copy never cites 3.2x for k-quants. If the k-quant profile differs, that is a better post and direct material for the paper extension, not a failure.

---

## 7. The uploader shootout

Same model, same quant label, different uploaders: bartowski, unsloth, and mradermacher side by side, at Q4 and at Q3. Q3 is included because calibration effects grow as bits drop, and testing only at Q4 would produce "uploader doesn't matter," true and useless, at exactly the level where nobody was torn. mradermacher publishes both calibrated and uncalibrated versions through the same pipeline, which is the cleanest possible isolation of whether calibration data matters. Separately, Qwen ships official GGUFs and Meta does not, so the official-versus-community question runs on the Qwen side. Either result is a post: "same label, different model," or "relax, the uploader doesn't matter."

---

## 8. Honesty as a feature set

These are product features, not paperwork, and they exist because the project's brand is the pre-registration habit:

- **The Random button.** Anyone can pull undramatic items, so the dramatic curated set can't be called a rig.
- **Both flip directions.** Losses and lucky gains, always shown together.
- **Four cell states.** The matrix never claims "no damage" from mere absence of evidence, and never hides a real-but-tiny loss behind "equivalent."
- **The public registration, frozen before confirmatory data, third-party timestamped.** It includes an explicit declaration that the pilot run was exploratory and informed the parameter choices, so nobody discovers that themselves.
- **Blind testing done blind.** The guess-the-quant gate uses two raters who have never seen any outputs, because the person who ran the pipeline is not blind. The gate is run on the ambiguous middle of the ladder only, since letting raters score points on obvious gibberish would pass the test trivially. Fallback if a second naive rater can't be found in time: the game stays out until one passes it.
- **Scope labels everywhere.** Per model, per size, no extrapolation, residual gaps disclosed.
- **Speed and hardware labeled.** Performance numbers say what machine they came from.

---

## 9. The 70B expansion and the iso-RAM answer

Funded by credits, scheduled two to three weeks after launch: the full ladder for a 70B-class Llama. This is the folklore battleground ("is a heavily compressed 70B still better than a lightly compressed 8B?") and it unlocks the strongest artifact of the whole project: **the iso-RAM view.** Pick a memory budget; see the best (model size, quant) pair per capability at that budget. 70B at 2-bit versus 8B at 6-bit, same RAM, who survives. Cross-size comparisons use absolute accuracy rather than within-model retention, because retention is defined relative to each model's own baseline. It also makes the "small models degrade worse" claim measured instead of asserted, within one family.

It deliberately does not enter the launch: a delayed launch costs more than a spectacular week-two follow-up earns.

---

## 10. Phases and gates

**Phase 0 decides whether the project exists.**

- **0A, the pilot.** One weekend, own hardware. Run the 1.5B ladder end to end. Purpose: shake out the pipeline, see the exploratory curves, pick curation, sanity-check the length budget. Declared exploratory; its numbers are thrown away.
- **The freeze.** The registration is committed and timestamped, with the verified file lists and hashes, the twin problems round-trip-checked, and the licenses audited (the twin-template source, GSM8K, and the terms on redistributing model outputs). Nothing confirmatory runs before this commit, and no GPU money is spent before it either.
- **0B.** The confirmatory runs: both reference ladders, the shootout files, the official-versus-community files, all the divergence measurements.
- **0B'.** The 1.5B confirmatory run on the actual serving machine, plus the curated 50 regenerated there.
- **0C.** The blind guessability test with two naive raters.

**Bars.** The 1.5B must show visible drama on curated prompts, or the spectacle half has no content. The 8B-class curves must be informative, and flat-then-cliff counts as informative: "Q4 costs about a point, the cliff starts at Q3" is the service headline devs actually want, claimable only through the registered equivalence test. The project dies only if every curve is noise.

**Phase 1 (~two weeks of evenings): launch.** Playground for all three models, share cards, permalinks, the matrices with cliff lines, the durability page, methodology, the public registration, and the game if it passed its gate.

**Phase 2.** The Picker, the model-card snippet generator and the uploader template ask, the instruction-following suite, the shootout post.

**Phase 3.** Live-prompt hardening, the gallery, voting.

**Phase 4.** The 70B ladder and the iso-RAM view, then the long-context suite.

---

## 11. Launch

**Before launch, quietly:**
- DM bartowski with the 8B table, the methodology, and the registration link. Framed as a sanity check, not a favor; a private review also catches configuration mistakes before they can be public ones. After it checks out, the one small ask: a single link line in his card template. Wait a week or one nudge, then launch regardless.
- No cold tags, ever. If the divergence result lands cleanly, a llama.cpp GitHub discussion is the native venue where that crowd actually reads.

**Launch day:**
- Asset #1 is a 15-second screen recording of the slider, made before any polish, using a built-in demo mode so the recording is clean. Everything else serves this clip.
- The X thread leads with whatever Phase 0 actually found. Candidate hooks: "Every GGUF page shows you file sizes. None shows you what got deleted." and "I measured where each capability falls off the cliff as you compress Llama 8B."
- Same day, the r/LocalLLaMA post in numbers framing: "Where the cliff is for Llama-3.1-8B, per capability, pre-registered, CSVs inside." That crowd wants tables, not the meme.
- Show HN as the backup venue: "See exactly what quantization deletes from an LLM."

**The follow-up ladder, each its own post, spaced over the following weeks:** the divergence verdict (whichever branch), the fragility-screening trick if it works, the uploader shootout, official versus community, the contamination verdict whichever way it cuts, the durability pair, the "2-bit beats full precision on these" noise flips, the 70B cliff and the iso-RAM table as the season finale, then monthly gallery highlights. Launch is not the event; the drip is.

---

## 12. Money and operations

- Funded by AWS credits: $5,000 of a $10,000 pool allocated to this project. Rough split: a year of the always-on serving box (~$1,600), the confirmatory GPU runs (~$150), the 70B run (~$800-1,500), an optional launch-week live-8B stunt (~$150), and small change for storage and the domain. Analytics is the one small cash cost.
- **Guardrails:** burn alerts at 50/80/100% of the allocation, credits excluded from the calculation so usage is visible as cost, nothing GPU-shaped ever left running idle, spend gates the agent must stop at, and a leash on what the automation account is even allowed to touch.
- **What the budget must not fund:** a permanently live 8B, replatforming the site onto managed cloud services, or any bigger model creeping into the launch phases. The gates exist so spectacle can never delay measurement or vice versa.
- **Credit expiry is a planning input:** use-it-or-lose-it argues for the 70B run and the live-8B week, not for hoarding.

---

## 13. Outputs, each with a terminal state

1. **The site**: done when all Phase 1 pages are public, permalinks are stable, and share cards unfurl correctly.
2. **The dataset**: every per-item record (outputs, grades, divergence, fragility signals) published for others to analyze; done when it's live with working downloads. Twins embargoed until the paper.
3. **The registration and the results report**: done when both are public and the hashes match.
4. **Answers to the five questions**: done when each has a verdict with statistics, including null and inconvenient verdicts.
5. **The cliff tables for three models**: done when published with every cell carrying its state and evidence.
6. **The open pipeline**: done when a stranger can rerun a ladder from a config file and the README.
7. **The content set**: the clip, the launch pair, and the follow-up calendar; done when the launch pair ships same-day and the calendar exists.
8. **The distribution artifacts**: the snippet generator, and the template-link ask made and its outcome recorded either way.
9. **The paper inputs**: the contamination and transfer results delivered in paper-ready form.

---

## 14. Risks, each with its answer

- The divergence metric turns out to work fine: publish the conversion table and the screening feature. Still content.
- The asymmetry doesn't transfer to k-quants: scheme-dependence is a better post and paper material.
- Contamination cuts the headline number: publish the cut. It fits the brand better than the pad.
- Q4 shows a small real loss instead of "free": expected, and the better headline.
- Twin construction stalls: fall back to the smaller vetted template set, smaller but clean.
- Uploaders re-quantize: hash pinning and badges, never silent staleness.
- Cherry-picking accusations: the Random button, both flip directions, full downloads, the registration.
- Abuse or cost spikes on the live box: graceful fallback to precomputed-only.
- Uploaders ignore the ask: the tables work standalone; the link was upside.
- All curves are noise: the only true kill, discovered for the price of one pilot weekend.

---

## 15. Success criteria

- **Viral:** the 15-second clip clears ~100k impressions, or the LocalLLaMA post front-pages.
- **Useful:** the template link lands or two model cards embed a table; steady search traffic to the Picker; anyone besides me cites a cliff in an argument.
- **Portfolio floor, guaranteed:** all nine outputs exist, the results match the registration in public, one pinned thread, one stronger line in every cold email.

---

## 16. Non-goals

- Not a general evaluation harness, and not competing with one; the point is paired compression diffs. Said once on the methodology page, never defensively.
- Not a model-quality leaderboard; retention is within-model by construction.
- No live serving above the 1.5B (except the optional one-week stunt). No accounts, no auth, no user data at launch. No training or fine-tuning claims. No managed-cloud sprawl.

---

## 17. The name

**BitCliff**, because the product's central finding is a shape: flat, then cliff. The community can adopt it as shorthand ("where's the cliff for Llama 8B") and every table reinforces it. "Lobotomized" stays in thread copy where it belongs; the domain stays clean. Runner-up was QuantDecay; WhichQuant was the working title and survives only as the Picker's SEO phrasing.

---

## 18. First move

0A. One weekend, own hardware, the existing suites, the 1.5B ladder. The curves decide everything else, and every hour spent on anything above this line before those curves exist is procrastination with better branding.