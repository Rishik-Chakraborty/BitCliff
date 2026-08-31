BitCliff — idea.md

v2. The complete plan, rewritten after review. This version absorbs the decisions v1 left implicit — suite sizes, what a rung actually is, the statistics, the build protocol — because every one of them turned out to be load-bearing. PLAN.md v1 is superseded by this document.

Changed from v1, in one place:

Suite sizes are no longer tradition; they come out of a power analysis run in the pilot and go into the registration. The registration gains a multiple-comparisons policy.
Launch is precomputed-only. The live prompt box moves to Phase 2 in a constrained form (evaluation-set prompts only, as a reproducibility demo); free-form input moves to Phase 3 with the gallery.
A rung on the ladder is an exact variant (Q4_K_M, never "Q4"), and the ladder's ordering rule is measured bits per weight. Cliff sentences name exact variants.
The reference pair stays the paper's models. Product currency comes from a third model added in Phase 2 by a written selection rule, kept outside the registered science.
Twins split into two types; the both-solved filter's bias direction is stated in the registration rather than discovered by a reviewer.
The answer-time divergence measurement (question 2) is pinned to a single frozen definition.
The uploader shootout gets a diplomacy protocol, and mradermacher becomes a second pre-launch reviewer, not just a subject.
The RAM column states its assumptions. The cliff is defined as a position, not a steepness claim, and each capability gets a slope tag.
New section: the build protocol — what the coding agent may build before the freeze and what must wait, and why the brief, not this document, is the prompt.
1. What BitCliff is

One line: watch an LLM lose its mind as you drag the bits out, then get per-capability numbers on what each GGUF file actually deletes, so you know which one to download.

The name is a definition. For each capability on each model, the cliff is the highest-precision file at which that capability is measurably damaged beyond the registered margin — named by its exact variant. "Retrieval cliff: Q3_K_M" is a sentence, a badge, a table row, and the thing people will argue about. A cliff is a position on the ladder, not a promise about steepness; a capability that erodes gradually still has a first rung past the margin, and the table says which kind it is. The name is load-bearing: cliffs drive the tables, the Picker, the durability comparison, and the recurring language of every post.

Two jobs, one site:

Spectacle. A slider that compresses a model in front of you until it forgets facts, loops, and finally speaks salad. This is marketing. It gets shared once.
Reference. Per-capability retention tables with pre-registered statistics, answering the question every local-LLM user has and nobody measures: which file do I download. This is the product. It gets linked forever.

The slider buys traffic. The tables keep it.

2. Purpose

For the ecosystem. GGUF selection runs on folklore. A model page shows fifteen files and their sizes; the only quality signals are perplexity and corpus KLD, both token-averaged, neither capability-aware. People pick by file size and Reddit memory. BitCliff replaces that with a place to see the damage with your own eyes and a place to measure it per capability. Invisible loss becomes visible, then quantified.

For the research. The site doubles as an instrument. Five questions, registered publicly before any confirmatory data exists:

Does the divergence metric everyone cites (corpus KLD) actually rank files in the order of their capability damage?
Does divergence measured along the answer itself predict which specific questions a quant gets wrong?
Can the uncompressed model alone predict which of its own answers are fragile, before any quant is ever run?
Which direction does training-data contamination bend arithmetic measurements: does memorization make quantization damage look bigger than it is, or smaller?
Does the published finding (retrieval damaged ~3.2x more than arithmetic at 3-bit) transfer from the lab quantization schemes it was measured on to the k-quants people actually download?

For me. This operationalizes the published quantization research into a running public artifact: an interactive companion to the arXiv extension, a public pre-registration whose results visibly match it, a distribution asset, and one stronger line in every cold email. The floor outcome is guaranteed even if nothing goes viral.

3. Audiences
Scrollers (X, r/LocalLLaMA). Want 15 seconds of entertainment. Served by the slider, the failure gallery, and share cards. Converted by permalinks.
Devs picking a GGUF for an 8B-class model. Currently choose by file size and folklore. Served by the matrices, the cliffs, and the Picker.
Quant uploaders. The people whose HF pages the audience already lives on. Served by free, rigorous quality tables for their catalogs. The realistic win is one link line in an uploader's card template, which propagates across their entire catalog. Their adoption is upside, never a dependency.
4. The models and the ladder

Three models at launch, two roles; a fourth model later, by rule:

Spectacle: Qwen2.5-1.5B-Instruct. Small models degrade dramatically, which is the content. Cheap enough that a live box is plausible later. This is the only model that will ever get one.
Reference pair: Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct. These are the paper's models, kept deliberately: question 5 is a transfer question, and continuity with the published measurements is worth more to the registered science than currency. They are size-comparable, so the cross-family durability comparison is legitimate. Precompute-only.
The currency model (Phase 2). The reference pair will read as dated to the Picker's search audience, so a third 8B-class model joins when the Picker ships — chosen by a rule written now, not a mood then: the most-downloaded 8B-class instruct GGUF repo on Hugging Face over the trailing 30 days, excluding models whose chat template forces a reasoning mode that can't be deterministically disabled, and models covered by fewer than two tracked uploaders. It gets a full ladder, a matrix, and Picker entries. It never enters the five registered questions; it is tables, not science, and the methodology page says so.

A rung is an exact variant, ordered by measured bits. "Q4" is a neighborhood — Q4_K_M, Q4_K_S, Q4_0, IQ4_XS are different files with different damage — so no bare quant level ever appears where a number could attach to it. The canonical ladder per model, frozen with hashes at registration: F16 (converted locally as the reference, never a download recommendation), Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_M, Q2_K, IQ2_XXS, and IQ1_S where the tracked uploaders publish it. Ladder order is measured bits per weight (file size divided by parameter count), which also settles how I-quants interleave with K-quants: by measurement, not by the number in the name.

A quant level is a file, not a label. bartowski's, unsloth's, and mradermacher's Q4_K_M are different files with different calibration. Every output and every table row shows uploader, imatrix status, and the exact file hash. The canonical ladder uses bartowski's imatrix quants. Results are pinned to file hashes; when an uploader re-quantizes, pages get a "newer file exists" badge rather than silently pointing at stale data.

One exception for spectacle: if published 1.5B repos stop above the truly deranged levels, the missing bottom rungs get quantized in-house and clearly labeled as such. The published-files-only rule exists to protect download recommendations, and the 1.5B is not one.

5. The product, page by page
Playground (home)
Model tabs. All three models are browse-only over precomputed outputs at launch, with a banner saying so. The live box is not a launch feature; see Phase 2.
Ladder view: one prompt, every rung's output stacked, full precision at top. The screenshot format.
A/B view: any two rungs side by side with the slider between them. The slider snaps between rungs; this is what the launch clip records, and nothing about the clip ever required live inference.
Every output card shows: exact variant, uploader, a grade badge (correct / partial / wrong / truncated / unparseable), generation speed, and the first token where it diverged from full precision, highlighted.
Prompt sources: a curated set of 50 (15 factual retrieval, 10 arithmetic word problems, 10 instruction adherence, 5 long-context needles, 10 pure-spectacle prompts labeled as unscored entertainment) plus a prominent Random button that draws from the full evaluation set. The curated set is allowed to be dramatic precisely because the Random button exists; only the Random pool's integrity is sacred.
Every comparison has a stable permalink.

The live box, staged honestly. Phase 2 ships it constrained: run any prompt from the evaluation set, live, and watch it reproduce the published output token for token. That is a reproducibility demo that doubles as the feature, with no free-form abuse surface, no jailbreak screenshots, and no moderation problem. Free-form input arrives in Phase 3 with the gallery, behind v1's rules: a handful of prompts per hour per person, short outputs, visible queue position, graceful fallback to precomputed-only under load, and the shared-IP caveat said on the page.

The Numbers (per model)

One matrix per model. Rows are quant files; columns are per-capability retention versus full precision (with confidence intervals), the divergence metrics people already cite (so any mismatch is visible in a single screenshot), file size, estimated RAM, speed, uploader, imatrix status, and file hash.

The RAM column states its assumptions: file size plus KV cache at 8,192 tokens of context in the default llama.cpp KV precision for the pinned build, formula published, context length in the column header. File size is not residency, and the column never pretends it is.

Every cell is one of four states, and the distinction is the product's honesty:

Damaged. Statistically worse after the multiplicity correction (section 7), and worse than the registered margin. Red. Cliffs are drawn here.
Small real loss, within margin. Statistically detectable, but smaller than the margin. Shows the signed loss, e.g. "-1.2pp." This is the expected verdict for the popular Q4 files, and it is a better headline than "free": "costs about a point, never more than three" is more credible and more quotable.
Equivalent. No detectable loss, and the data is strong enough to bound any loss under the margin. Green.
Indeterminate. Not enough evidence either way. Grey, and never rounded up to "free." Section 7 exists so this state cannot appear on the rungs everyone actually looks at.
The cliff line is drawn across the matrix where each capability first enters Damaged, with badges above the table: "Retrieval cliff: Q3_K_M. Arithmetic cliff: Q2_K." Next to each badge, a slope tag — abrupt or gradual — computed from the retention curve, because a cliff marks where the margin is crossed, not how fast. Odd cases where a lower file passes after a higher one failed are flagged, never smoothed.
Drill-down per cell shows both directions: the questions full precision got right that the quant lost, AND the questions the quant randomly gained. Showing the gains closes the cherry-picking hole, and "the 2-bit file beats full precision on these five questions" is its own delightful post.
Every table downloads as CSV and JSON.
Durability

The two reference models' retention curves overlaid per capability, each normalized to its own full-precision baseline (which removes the differences in tokenizers, templates, and starting accuracy), with cliff positions compared directly. The claim is class-level ("7-8B"), the residual size gap is disclosed in one line, and the 1.5B appears nowhere on this page. This page exists because "which family takes a punch better" is a real argument people have with no data.

The Picker (Phase 2)

Input: how much RAM you have and what you use the model for (retrieval-heavy chat, math and code, agents and long documents). Output: one specific file, with reasoning: the smallest file where every capability you care about is above its cliff with margin, tie-broken by speed, linking to the exact file on HF. The Picker covers the reference pair and the currency model, and the currency model is what makes its search thesis honest. Page titles match the literal search query ("Llama 3.1 8B GGUF comparison: which quant to download") because that query is the whole reference audience.

Guess the Quant (Phase 1, if it earns its way in)

See one output, guess the compression level. Scored by distance (full points for exact, partial for adjacent, nothing otherwise) because the top of the ladder is genuinely indistinguishable and exact-only scoring would make the game feel broken. Ten rounds, shareable score card. It ships only if a blind test proves the mid-ladder is actually guessable by humans (see section 9).

Failure Gallery (Phase 3)

Community-submitted degradations from the free-form live box, with voting and tags: forgot a fact, infinite loop, gibberish, ignored the instruction, botched the math. Loops and gibberish are tagged mechanically, not by mood. The "infinite loop" tag carries a note that loops are partly an artifact of the deterministic decoding the site uses for reproducibility; normal sampling escapes some of them. A monthly "best failures" post keeps the account alive between releases.

Methodology and the public registration

A methodology page that states every test, every margin, every rule, in full, including the scope disclaimer that results are per model and size and must not be extrapolated. It lists everything the freeze locked: suite sizes, margins, the exact variant list and ordering rule, the multiplicity policy, the question-2 definition, the twin policy and its stated biases, the currency-model selection rule. The pre-registration is rendered verbatim with its commit hash, timestamped by a third-party mechanism, so "you decided the rules after seeing the data" is impossible by construction.

Share cards

Every permalink renders a proper social preview: the prompt, the outputs, the variant labels, the uploader, the site. Launch-blocking, not polish, because the card is the growth loop. One-tap copy-as-image for mobile. Every share URL carries tracking tags so the loop's health is measurable.

6. What gets measured, in plain language

Three capability suites at launch (two more, instruction following and long context, in Phase 2):

Retrieval: the multivalue2 task from the published paper.
Arithmetic, twice. The standard GSM8K problems, and twins that cannot have been memorized during training, because they didn't exist.

Twins come in two types (registered question 4).

Type A: entities renamed, numbers untouched. Same arithmetic, same answer, difficulty preserved exactly by construction. Primary for the contamination question, because nothing about the problem changed except the surface a memorizer would have latched onto.
Type B: entities and numbers changed, answer recomputed. Secondary; catches number-specific memorization; difficulty approximately matched rather than identical, and treated accordingly.

If models partially memorized GSM8K, then "arithmetic damage" partly measures how memorization degrades, not how calculation degrades. Two stories are both plausible and point opposite ways: memorized answers might be fragile like other recall (making the paper's asymmetry an underestimate) or robust like overlearned habits while multi-step calculation compounds noise (making it an overestimate). The registration commits to testing this two-sided, with no preferred outcome. The comparison is restricted to problems the full-precision model solves in both original and twin form, because comparing on different populations would rig the result mechanically — and the registration states this filter's bias out loud: restricting to both-solved items selects for robustly solved problems and therefore attenuates the memorization signal, so a null on question 4 reads as "no effect large enough to survive a conservative filter," never as proof of absence. A cut headline ships as its own honest post; that outcome fits the brand better than a padded one.

One embargo: the twin problems stay out of the public Random pool and out of the published dataset until the paper extension ships, because publishing them would leak the templates into future training data and to competitors. Their construction method is public from day one; the instantiated problems are not.

The divergence questions (registered questions 1-3). The community's current quality metric is a divergence score computed on Wikipedia text. The registration tests it at two levels: does it rank files correctly (its actual job), and does divergence at the moment of answering predict which individual questions flip. Question 2's measurement is pinned before any data exists: answer-time divergence is the mean per-token KL divergence between quant and full precision, teacher-forced along the full-precision model's own answer tokens — both models scored on the same trajectory, so the measurement can't be contaminated by the quant wandering onto its own path. Secondary: first-token divergence conditioned on the prompt alone. Both frozen in the registration; no third definition gets computed after seeing the data. Question 3 asks whether the full-precision model alone can flag its own fragile answers. All branches are informative: if the metric works, the site publishes the conversion table and a screening trick; if it fails, the site publishes exactly what it hides; if it only fails because it's computed on the wrong text, that is the sharpest post of all.

Truncation and unparseability are their own verdicts. Outputs get a generous length budget; an answer that never arrives within it counts as wrong (an endless loop is damage), but truncation is tracked separately and reported per cell so "wrong" never silently absorbs "cut off." Likewise, unparseable is a grade, not an exception: the site's own content is adversarial input to its own parser, and answer extraction must survive the deranged zone — an output that defeats extraction is recorded as unparseable and displayed as such, never crashed on, never mislabeled. The pilot checks that the length budget is genuinely generous for the full-precision model before the number is frozen.

Suite sizes come from the power analysis, not from tradition. Bounding a loss under a roughly three-point margin with paired right/wrong outcomes takes high hundreds of items per suite, and the popular rungs must be adjudicable — a matrix that's grey exactly where everyone looks has failed at its one job. The exact N per suite is an output of the pilot and an input to the freeze; the mechanics live in section 7.

Reproducibility, honestly scoped. All evaluation uses fully deterministic generation. One manifest per run records every generation setting and the applied chat template, and the pipeline verifies that every file on a ladder resolves to the same template and tokenizer behavior before anything is graded — a ladder whose rungs format their prompts differently measures formatting drift and calls it damage. The site states plainly that determinism holds within a fixed hardware and software configuration, not across arbitrary machines. The 50 curated playground prompts are regenerated on the exact machine that will serve the Phase 2 live box, so anyone re-running a curated prompt live sees the same output the page shows.

The k-quant transfer question (registered question 5). The paper's asymmetry was measured on lab quantization schemes. The downloadable k-quants are a different technique, and the headline number is not assumed to transfer; the launch copy never cites 3.2x for k-quants. If the k-quant profile differs, that is a better post and direct material for the paper extension, not a failure.

7. The statistics, in plain language

This section exists because the four cell states, the cliffs, and the headline are all statistical claims, and v1 never said how they'd be earned.

The margin is the per-capability number separating "small real loss" from "damaged." Exact values frozen at registration, chosen before confirmatory data, informed by the pilot only in the declared-exploratory sense.
Damaged means a paired comparison against full precision — same items, same model, only the bits differ — comes back worse than the margin, and stays significant after the multiplicity correction.
Equivalent is earned, not assumed: the interval for the loss must sit entirely inside the margin (two one-sided tests). Absence of evidence stays grey.
Multiplicity: launch freezes roughly sixty confirmatory cells (rungs × capabilities × models). The registration commits to a false-discovery-rate correction within each model's family of cells, so a wall of tests can't manufacture red, and the promised non-monotonicity flags mark real oddities instead of noise.
Power is sized backward from the product. The suites must be large enough that Q4_K_M and Q5_K_M resolve to a definite state; that requirement, not habit, sets N. The power analysis is a simulation over paired outcomes at candidate suite sizes, run during the pilot, its script published with the pipeline.
Power is equalized across suites, because a cliff's position partly depends on it. Two capabilities with identical true damage but different suite sizes would show different cliffs — the instrument lying. Where equalization is impossible, the methodology page discloses it in the same breath as the badge.
8. The uploader shootout

Same model, same quant label, different uploaders: bartowski, unsloth, and mradermacher side by side, at Q4_K_M and at Q3_K_M. Q3 is included because calibration effects grow as bits drop, and testing only at Q4 would produce "uploader doesn't matter," true and useless, at exactly the level where nobody was torn. mradermacher publishes both calibrated and uncalibrated versions through the same pipeline, which is the cleanest possible isolation of whether calibration data matters. Separately, Qwen ships official GGUFs and Meta does not, so the official-versus-community question runs on the Qwen side. Either result is a post: "same label, different model," or "relax, the uploader doesn't matter."

The diplomacy protocol, decided now rather than in the awkward moment. The shootout can embarrass the same person being asked for a template link, and v1 pretended those two facts lived on different pages. The rules: every named uploader receives their table privately, with methodology and raw outputs, a week before the shootout post; the post frames results as properties of calibration choices — imatrix datasets, variant construction — not report cards on people; nothing is softened; and if the numbers cost the ask, the ask was always upside and the numbers are the product. mradermacher's calibrated/uncalibrated pairs also make them a natural second pre-launch reviewer, not just a subject.

9. Honesty as a feature set

These are product features, not paperwork, and they exist because the project's brand is the pre-registration habit:

The Random button. Anyone can pull undramatic items, so the dramatic curated set can't be called a rig.
Both flip directions. Losses and lucky gains, always shown together.
Four cell states. The matrix never claims "no damage" from mere absence of evidence, and never hides a real-but-tiny loss behind "equivalent."
The multiplicity policy is public, so sixty cells can't quietly become sixty chances to find red.
Bias directions are stated by the author, not discovered by reviewers: the both-solved filter's attenuation, the residual size gap on the durability page, the loop-tag decoding artifact.
Slope tags, so "cliff" never oversells abruptness the data doesn't show.
The public registration, frozen before confirmatory data, third-party timestamped. It includes an explicit declaration that the pilot run was exploratory and informed the parameter choices, so nobody discovers that themselves.
Blind testing done blind. The guess-the-quant gate uses two raters who have never seen any outputs, because the person who ran the pipeline is not blind. The gate runs on the ambiguous middle of the ladder only, since letting raters score points on obvious gibberish would pass the test trivially. Fallback if a second naive rater can't be found in time: the game stays out until one passes it.
Scope labels everywhere. Per model, per size, no extrapolation, exact variants named.
Speed and hardware labeled. Performance numbers say what machine they came from.
10. The 70B expansion and the iso-RAM answer

Funded by credits, scheduled two to three weeks after launch: the full ladder for a 70B-class Llama. This is the folklore battleground ("is a heavily compressed 70B still better than a lightly compressed 8B?") and it unlocks the strongest artifact of the whole project: the iso-RAM view. Pick a memory budget; see the best (model size, quant) pair per capability at that budget. 70B at 2-bit versus 8B at 6-bit, same RAM, who survives. Cross-size comparisons use absolute accuracy rather than within-model retention, because retention is defined relative to each model's own baseline. It also makes the "small models degrade worse" claim measured instead of asserted, within one family.

It deliberately does not enter the launch: a delayed launch costs more than a spectacular week-two follow-up earns. The precomputed-only launch decision frees serving budget, and this is where that budget lands.

11. Phases and gates

Phase 0 decides whether the project exists.

0A, the pilot. One weekend, own hardware, agent-supervised (section 12). Run the 1.5B ladder end to end. Purpose: shake out the pipeline, see the exploratory curves, pick curation, sanity-check the length budget. Declared exploratory; its numbers are thrown away. Two artifacts outlive it: the power analysis (candidate Ns simulated, suite sizes chosen) and the per-item record schema, designed as if it is already the published dataset. Numbers die at the freeze; the schema and the Ns are freeze inputs.
The freeze. The registration is committed and timestamped. The checklist, expanded from v1: the registration text; the margins; the suite sizes from the power analysis; the multiplicity policy; the exact variant list and the bits-per-weight ordering rule; the verified file hashes; the question-2 divergence definition; the twin policy with its bias statements; the currency-model selection rule; the twin problems round-trip-checked; the licenses audited (the twin-template source, GSM8K, the terms on redistributing model outputs). Nothing confirmatory runs before this commit, no GPU money is spent before it, and no code that hardcodes any frozen parameter is written before it.
0B. The confirmatory runs: both reference ladders at the frozen suite sizes, the shootout files, the official-versus-community files, all the divergence measurements.
0C. The blind guessability test with two naive raters.
(v1's 0B' — regenerating the curated prompts on the serving machine — moves to Phase 2, where it becomes the reproduction gate the constrained live box must pass before it ships.)

Bars. The 1.5B must show visible drama on curated prompts, or the spectacle half has no content. The 8B-class curves must be informative, and flat-then-cliff counts as informative: "Q4_K_M costs about a point, the cliff starts at Q3_K_M" is the service headline devs actually want, claimable only through the registered equivalence test. The power analysis can resize suites; it cannot lower bars. The project dies only if every curve is noise.

Phase 1 (~two weeks of evenings — now an honest estimate, because the live box left): launch. Precomputed playground for all three models, the slider, share cards, permalinks, the matrices with cliff lines and slope tags, the durability page, methodology, the public registration, and the game if it passed its gate.

Phase 2. The Picker; the currency model's full ladder and matrix; the constrained live box behind the reproduction gate; the model-card snippet generator and the uploader template ask; the instruction-following suite; the shootout post, protocol applied.

Phase 3. Free-form live input with the v1 rate limits and fallback, the gallery, voting.

Phase 4. The 70B ladder and the iso-RAM view, then the long-context suite.

12. The build protocol

The coding agent changes the schedule, not the gates. Rules, so the gates survive contact with an assistant that never stalls:

Sendable now: the 0A pipeline (downloader, runner, grader, plotter), the power-analysis simulation, and the playground UI built against pilot outputs as fixture data — the one Phase 1 surface that is decision-independent.
Waits for the freeze: anything that would hardcode a registration parameter. An agent does not stall on an undecided question; it decides it, silently and plausibly, and the result is a schema that pre-empts the registration. So every frozen parameter — suite sizes, margins, variant lists, budgets — lives in one config file the code reads and never defines. The v1 rule that nothing confirmatory runs before the freeze extends to code.
The brief is the prompt; this document is context. An agent handed the grand vision will helpfully scaffold the grand vision. The 0A brief contains: explicit non-goals (no website beyond fixtures, no share cards, no AWS); the exact 1.5B file list; pointers to the existing suite code; the grade taxonomy with truncated and unparseable as first-class outcomes; the full generation-settings manifest to record; and the record schema, the one 0A artifact that ships.
Two known traps, written into the brief because the agent will hit both: the grader must survive the deranged zone (unparseable is a grade path, not an exception), and the template/tokenizer verification runs before anything is graded.
Spend rule: agent work happens on own hardware until the freeze; after it, the automation account operates under the section 14 leash and stops at the spend gates.
13. Launch

Before launch, quietly:

DM bartowski with the 8B table, the methodology, and the registration link. Framed as a sanity check, not a favor; a private review also catches configuration mistakes before they can be public ones. After it checks out, the one small ask: a single link line in his card template. Wait a week or one nudge, then launch regardless.
The same private review goes to mradermacher, whose calibrated/uncalibrated pipeline makes them the reviewer most likely to catch a calibration-related mistake. (Shootout previews to all named uploaders happen in Phase 2, per the section 8 protocol.)
No cold tags, ever. If the divergence result lands cleanly, a llama.cpp GitHub discussion is the native venue where that crowd actually reads.

Launch day:

Asset #1 is a 15-second screen recording of the slider, made before any polish, using a built-in demo mode over precomputed outputs so the recording is clean. Everything else serves this clip.
The X thread leads with whatever Phase 0 actually found. Candidate hooks: "Every GGUF page shows you file sizes. None shows you what got deleted." and "I measured where each capability falls off the cliff as you compress Llama 8B."
Same day, the r/LocalLLaMA post in numbers framing: "Where the cliff is for Llama-3.1-8B, per capability, pre-registered, CSVs inside." That crowd wants tables, not the meme.
Show HN as the backup venue: "See exactly what quantization deletes from an LLM."

The follow-up ladder, each its own post, spaced over the following weeks: the divergence verdict (whichever branch), the fragility-screening trick if it works, the constrained live box ("run the eval yourself, watch it match the published output"), the uploader shootout, official versus community, the contamination verdict whichever way it cuts, the currency model's ladder, the durability pair, the "2-bit beats full precision on these" noise flips, the 70B cliff and the iso-RAM table as the season finale, then monthly gallery highlights. Launch is not the event; the drip is.

14. Money and operations
Funded by AWS credits: $5,000 of a $10,000 pool allocated to this project. The precomputed-only launch changes the shape of the spend: the site launches static or near-static (approximately nothing per month), and the always-on GPU box becomes a Phase 2 line item tied to the live box, budgeted by the months it actually runs rather than reserved for a year by default. Rough split: the confirmatory GPU runs (~$150 at v1 suite sizes — item counts scale linearly and inference is cheap, so even power-tripled suites stay in the low hundreds), the Phase 2+ serving box, the 70B run (~$800-1,500), an optional launch-week live-8B stunt (~$150, Phase 2 or later), and small change for storage and the domain. Analytics is the one small cash cost.
Guardrails: burn alerts at 50/80/100% of the allocation, credits excluded from the calculation so usage is visible as cost, nothing GPU-shaped ever left running idle, spend gates the agent must stop at, and a leash on what the automation account is even allowed to touch.
What the budget must not fund: a permanently live 8B, a GPU serving box before the phase that needs one, replatforming the site onto managed cloud services, or any bigger model creeping into the launch phases. The gates exist so spectacle can never delay measurement or vice versa.
Credit expiry is a planning input: use-it-or-lose-it argues for the 70B run and the live-8B week, not for hoarding.
15. Outputs, each with a terminal state
The site: done when all Phase 1 pages are public, permalinks are stable, and share cards unfurl correctly.
The dataset: every per-item record (outputs, grades, divergence, fragility signals) published for others to analyze, in the schema frozen at 0A; done when it's live with working downloads. Twins embargoed until the paper.
The registration and the results report: done when both are public, the hashes match, and the statistics policy in the report is the one in the registration.
Answers to the five questions: done when each has a verdict with statistics, including null and inconvenient verdicts.
The cliff tables for the reference models: done when published with every cell carrying its state and evidence.
The open pipeline: done when a stranger can rerun a ladder from a config file and the README; includes the power-analysis script.
The content set: the clip, the launch pair, and the follow-up calendar; done when the launch pair ships same-day and the calendar exists.
The distribution artifacts: the snippet generator, and the template-link ask made and its outcome recorded either way.
The paper inputs: the contamination and transfer results delivered in paper-ready form.
16. Risks, each with its answer
The divergence metric turns out to work fine: publish the conversion table and the screening feature. Still content.
The asymmetry doesn't transfer to k-quants: scheme-dependence is a better post and paper material.
Contamination cuts the headline number: publish the cut. It fits the brand better than the pad.
Q4_K_M shows a small real loss instead of "free": expected, and the better headline.
The power analysis demands much bigger suites: costs scale linearly and stay trivial; the schedule absorbs it, the freeze waits for it.
The popular rungs land Indeterminate anyway: cannot happen by design — that is what the sizing was for. If it happens regardless, the margin was mis-set, and the honest move is a re-registration labeled as such, never a quiet reinterpretation.
The currency rule picks a thinking-mode model: the written exclusions handle it without a judgment call in the moment.
Twin construction stalls: fall back to the smaller vetted template set, smaller but clean.
Uploaders re-quantize: hash pinning and badges, never silent staleness.
The shootout embarrasses an uploader mid-ask: the section 8 protocol was written for exactly this; the numbers ship, the ask was upside.
Cherry-picking accusations: the Random button, both flip directions, full downloads, the registration.
Abuse or cost spikes on the live box: it doesn't exist until Phase 2, is eval-set-constrained when it arrives, and free-form input keeps the graceful fallback.
Uploaders ignore the ask: the tables work standalone; the link was upside.
All curves are noise: the only true kill, discovered for the price of one pilot weekend.
17. Success criteria
Viral: the 15-second clip clears ~100k impressions, or the LocalLLaMA post front-pages.
Useful: the template link lands or two model cards embed a table; steady search traffic to the Picker; anyone besides me cites a cliff in an argument.
Portfolio floor, guaranteed: all nine outputs exist, the results match the registration in public, one pinned thread, one stronger line in every cold email.
18. Non-goals
Not a general evaluation harness, and not competing with one; the point is paired compression diffs. Said once on the methodology page, never defensively.
Not a model-quality leaderboard; retention is within-model by construction.
No free-form live input before Phase 3, and no live serving above the 1.5B (except the optional one-week stunt).
The currency model never enters the registered questions; it is product, not science.
No rung is ever labeled with a bare "Q4."
No accounts, no auth, no user data at launch. No training or fine-tuning claims. No managed-cloud sprawl.
19. The name

BitCliff, because the product's central question is a position: where does the damage start. The community can adopt it as shorthand ("where's the cliff for Llama 8B") and every table reinforces it. A cliff is a place, not a gradient — if the curves come back gradual, the badge still marks where the margin is crossed and the slope tag says gradual, so the name survives its own data either way. "Lobotomized" stays in thread copy where it belongs; the domain stays clean. Runner-up was QuantDecay; WhichQuant was the working title and survives only as the Picker's SEO phrasing.

20. First move

0A. One weekend, own hardware, supervising: the agent gets the brief (section 12), the human gets the exploratory curves, the power script gets run, and the freeze checklist gets filled in. The curves decide everything else, and every hour spent on anything above this line before those curves exist is still procrastination with better branding.