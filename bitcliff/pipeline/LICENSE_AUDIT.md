# License audit — closed-book factual QA suite (Task 3, ruling 1a)

Executed before any suite code was written, per binding order: Task 3 Step 1
must run and commit this audit first. Verification date for every entry
below: **2026-08-26**. All raw snippets are pasted verbatim from the cited
API/HTTP responses, fetched on the verification date.

## Verdict (see §1): MIT — via canonical source release

`akariasai/PopQA` — the dataset user ruling 1 designates as the primary
source — carries **no license metadata at all** on the Hugging Face Hub
mirror: no `license:` tag, no `cardData.license`, and no license section in
the dataset-card README (finding kept below, unchanged). That mirror-level
gap initially triggered the task brief's "missing license" stop condition
and this task was reported BLOCKED pending a controller decision.

**Controller decision (recorded 2026-08-26):** the license resolves via the
canonical source release, not the HF mirror. `github.com/AlexTMallen/
adaptive-retrieval` is the original PopQA release by the paper's authors
(Mallen et al. 2023). It has a `LICENSE` file: **MIT**. The dataset file
`data/popQA.tsv` ships IN that repo, so the MIT license covers the data.
The repo's README names the HF mirror `akariasai/PopQA` (maintained by
paper co-author Akari Asai) as an access path for the same data. HF mirror
card itself carries no license tag (recorded above); the MIT grant attaches
to the canonical release. Independently re-verified via the GitHub API
(`curl https://api.github.com/repos/AlexTMallen/adaptive-retrieval`
returns `"license":{"key":"mit","name":"MIT License","spdx_id":"MIT"}`) and
by fetching the LICENSE file directly (`curl https://raw.githubusercontent.
com/AlexTMallen/adaptive-retrieval/main/LICENSE` — "MIT License / Copyright
(c) 2023 Alex Mallen ..."), and by confirming `data/popQA.tsv` is present
in the repo tree via `curl https://api.github.com/repos/AlexTMallen/
adaptive-retrieval/contents/data`. Verified 2026-08-26.

TriviaQA (`mandarjoshi/trivia_qa`) remains the named fallback per ruling 1
and is unused — PopQA (MIT via the canonical release) is the suite's data
source. This unblocks Task 3; the suite build proceeds below.

---

## 1. `akariasai/PopQA` (primary source, user ruling 1) — MIT via canonical release

Query: `curl https://huggingface.co/api/datasets/akariasai/PopQA`

Raw JSON (full response):

```json
{"_id":"63a3a6bf3618ae17d667acf2","id":"akariasai/PopQA","author":"akariasai","sha":"098765c79ea10a2cb19c828324e33281b8336ec0","lastModified":"2022-12-22T01:01:20.000Z","private":false,"gated":false,"disabled":false,"tags":["size_categories:10K<n<100K","format:csv","modality:tabular","modality:text","library:datasets","library:pandas","library:mlcroissant","library:polars","region:us"],"description":"... Dataset Card for PopQA ...","downloads":9092,"likes":45,"siblings":[{"rfilename":".gitattributes"},{"rfilename":"README.md"},{"rfilename":"test.tsv"}],"createdAt":"2022-12-22T00:37:19.000Z","usedStorage":61513963}
```

Observations:
- `tags` contains no `license:*` entry (compare with GSM8K's `license:mit`
  and TriviaQA's `license:unknown`, both below — HF renders a `license:*`
  tag whenever the repo's YAML frontmatter declares one, or `license:other`/
  `license:unknown` when it's ambiguous; PopQA has none of these, meaning no
  license field was ever set).
- There is no `cardData` key in the API response at all (compare GSM8K and
  TriviaQA, which both have a populated `cardData` block).
- Fetched the dataset card directly to rule out an HF metadata-indexing gap:
  `curl https://huggingface.co/datasets/akariasai/PopQA/raw/main/README.md`
  — the full card (Dataset Summary, Languages, Dataset Structure, Data
  Fields, Citation Information) contains no License section and no license
  mention anywhere in the text.

**Mirror-level verdict: license missing** on the `akariasai/PopQA` HF card
itself. Per the task brief's stop condition ("If PopQA's license is missing
or restrictive, STOP and report BLOCKED"), Task 3 was initially halted here
and reported BLOCKED.

**Final verdict: MIT — via canonical source release.** Controller decision
(2026-08-26): the license resolves via `github.com/AlexTMallen/
adaptive-retrieval`, the original PopQA release by the paper's authors
(Mallen et al. 2023), which carries a `LICENSE` file of MIT and ships
`data/popQA.tsv` in-repo — the MIT grant covers the data. The HF mirror's
README names `akariasai/PopQA` as an access path for the same data,
maintained by paper co-author Akari Asai. Independently re-verified via
`curl https://api.github.com/repos/AlexTMallen/adaptive-retrieval`
(`"license":{"spdx_id":"MIT", ...}`), the raw `LICENSE` file content, and
the presence of `data/popQA.tsv` in the repo's `data/` directory listing.
This unblocks the suite build; `factual_qa` proceeds against PopQA below.

### Schema (recorded for whoever resolves this, via `datasets-server`)

Query: `curl 'https://datasets-server.huggingface.co/first-rows?dataset=akariasai/PopQA&config=default&split=test'`

Fields present: `id` (int), `subj` (str), `prop` (str), `obj` (str),
`subj_id`, `prop_id`, `obj_id` (Wikidata ids), `s_aliases` / `o_aliases`
(JSON-encoded list strings), `s_uri` / `o_uri`, `s_wiki_title` /
`o_wiki_title`, `s_pop` / `o_pop` (int — Wikipedia monthly pageviews for the
subject/object entity), `question` (str), `possible_answers` (JSON-encoded
list string of gold aliases).

Example row:

```json
{"id":4222362,"subj":"George Rankin","prop":"occupation","obj":"politician","subj_id":1850297,"prop_id":22,"obj_id":2834605,"s_aliases":"[\"George James Rankin\"]","o_aliases":"[\"political leader\",\"political figure\",\"polit.\",\"pol\"]","s_uri":"http://www.wikidata.org/entity/Q5543720","o_uri":"http://www.wikidata.org/entity/Q82955","s_wiki_title":"George Rankin","o_wiki_title":"Politician","s_pop":142,"o_pop":25692,"question":"What is George Rankin's occupation?","possible_answers":"[\"politician\", \"political leader\", \"political figure\", \"polit.\", \"pol\"]"}
```

So: question field is `question`; gold aliases live in `possible_answers`
(a JSON-encoded list string, matching the task brief's expectation);
popularity field is `s_pop` (subject entity's Wikipedia monthly pageviews —
matches the brief's "likely `s_pop`" guess). Now that the license question
is resolved (see updated verdict above), this schema is exactly what
`factual_qa.py` uses.

---

## 2. `mandarjoshi/trivia_qa` (named fallback, per user ruling 1 / freeze-plan §3)

Query: `curl https://huggingface.co/api/datasets/mandarjoshi/trivia_qa`

Relevant excerpt of the raw JSON (tags and cardData.license):

```json
"tags":["task_categories:question-answering","task_ids:open-domain-qa","task_ids:open-domain-abstractive-qa","task_ids:extractive-qa","task_ids:abstractive-qa","annotations_creators:crowdsourced","language_creators:machine-generated","multilinguality:monolingual","source_datasets:original","language:en","license:unknown","size_categories:100K<n<1M","format:parquet","modality:text","library:datasets","library:dask","library:polars","library:mlcroissant","arxiv:1705.03551","region:us"]
```

```json
"cardData":{"annotations_creators":["crowdsourced"],"language_creators":["machine-generated"],"language":["en"],"license":["unknown"],"multilinguality":["monolingual"], ...}
```

**Verdict: license explicitly `unknown`** on the Hub for this repo (the
`unfiltered.nocontext` config referenced in freeze-plan §3 as "Apache-2.0" —
that characterization traces to the original TriviaQA release terms, not
the HF repo's own `license:` tag, which reads `unknown`). Recorded here as
the named fallback per ruling 1; not selected — see verdict above, PopQA's
resolution (fix the license, or a controller-approved switch to this
fallback) is a decision for the controller, not this task.

---

## 3. `openai/gsm8k` (arithmetic suite dependency, already in use)

Query: `curl https://huggingface.co/api/datasets/openai/gsm8k`

Relevant excerpt:

```json
"tags":["benchmark:official","benchmark:eval-yaml","task_categories:text-generation","annotations_creators:crowdsourced","language_creators:crowdsourced","multilinguality:monolingual","source_datasets:original","language:en","license:mit","size_categories:10K<n<100K","format:parquet","modality:text","library:datasets","library:pandas","library:polars","library:mlcroissant","arxiv:2110.14168","region:us","math-word-problems"]
```

```json
"cardData":{"license":["mit"], "pretty_name":"Grade School Math 8K", ...}
```

**Verdict: MIT.** Verified; matches freeze-plan §4's "GSM8K (MIT — verify)."

---

## 4. `sgoel9/paul_graham_essays` — quarantined per freeze-plan §4

Query: `curl https://huggingface.co/api/datasets/sgoel9/paul_graham_essays`

Raw JSON (full response):

```json
{"_id":"661791519f00e704a682b380","id":"sgoel9/paul_graham_essays","author":"sgoel9","sha":"0c7155a53c25e24c9b9858314460e0ee1f5c3e4e","lastModified":"2024-04-20T03:49:51.000Z","private":false,"gated":false,"disabled":false,"tags":["task_categories:question-answering","task_categories:summarization","task_categories:text-generation","language:en","license:mit","size_categories:n<1K","format:csv","modality:text","library:datasets","library:pandas","library:mlcroissant","library:polars","doi:10.57967/hf/2212","region:us"],"description":"... Dataset Card for Paul Graham Essay Collection Dataset ...","downloads":415,"likes":5,"cardData":{"license":"mit","task_categories":["question-answering","summarization","text2text-generation","text-generation"],"language":["en"],"pretty_name":"Paul Graham Essay Collection","size_categories":["n<1K"]},"siblings":[{"rfilename":".gitattributes"},{"rfilename":"README.md"},{"rfilename":"pual_graham_essays.csv"}],"createdAt":"2024-04-11T07:29:21.000Z","usedStorage":10400961}
```

**Verdict: the HF *repo/uploader* tags this MIT** (`license:mit`,
`cardData.license: "mit"`) — but that MIT tag covers only sgoel9's upload
wrapper, not the underlying copyright in Paul Graham's original essay text,
which sgoel9 does not hold and which this uploader tag cannot license on
Paul Graham's behalf. The underlying essays' actual redistribution terms
are **unresolved**. Per freeze-plan §4, this dataset stays quarantined: it
may be used only as the source corpus for the never-published
configuration-2a `longctx_retrieval` (multivalue2) run's **outputs and
statistics**, and must not be redistributed or published as raw text. Not
used by the `factual_qa` suite in this task.

---

## 5. `Qwen/Qwen2.5-1.5B-Instruct` and `Qwen/Qwen2.5-7B-Instruct`

Query: `curl https://huggingface.co/api/models/Qwen/Qwen2.5-1.5B-Instruct`
and the `-7B-Instruct` equivalent.

```json
{
  "id": "Qwen/Qwen2.5-1.5B-Instruct",
  "tags": ["transformers","safetensors","qwen2","text-generation","chat","conversational","en","arxiv:2407.10671","base_model:Qwen/Qwen2.5-1.5B","base_model:finetune:Qwen/Qwen2.5-1.5B","license:apache-2.0","text-generation-inference","endpoints_compatible","deploy:azure","region:us"]
}
```

```json
{
  "id": "Qwen/Qwen2.5-7B-Instruct",
  "tags": ["transformers","safetensors","qwen2","text-generation","chat","conversational","en","arxiv:2309.00071","arxiv:2407.10671","base_model:Qwen/Qwen2.5-7B","base_model:finetune:Qwen/Qwen2.5-7B","license:apache-2.0","eval-results","text-generation-inference","endpoints_compatible","deploy:sagemaker","deploy:azure","region:us"]
}
```

**Verdict: Apache-2.0** for both, confirmed via the `license:apache-2.0` tag
on each model repo. Matches freeze-plan §4.

---

## 6. `meta-llama/Llama-3.1-8B-Instruct` — Llama 3.1 Community License

Query: `curl https://huggingface.co/api/models/meta-llama/Llama-3.1-8B-Instruct`

```json
{"_id":"6698d8a0653e4babe21e1e7d","id":"meta-llama/Llama-3.1-8B-Instruct","private":false,"pipeline_tag":"text-generation","library_name":"transformers","tags":["transformers","safetensors","llama","text-generation","facebook","meta","pytorch","llama-3","conversational","en","de","fr","it","pt","hi","es","th","arxiv:2204.05149","base_model:meta-llama/Llama-3.1-8B","base_model:finetune:meta-llama/Llama-3.1-8B","license:llama3.1","eval-results","text-generation-inference","endpoints_compatible","deploy:sagemaker","region:us"],"downloads":6375575,"likes":6675,"modelId":"meta-llama/Llama-3.1-8B-Instruct","author":"meta-llama","sha":"0e9e39f249a16976918f6564b8830bc894c89659","lastModified":"2024-09-25T17:00:57.000Z","gated":"manual", ...}
```

`license:llama3.1` tag confirms the repo license is the Llama 3.1 Community
License. The repo is gated (`"gated":"manual"`) and its own hosted
`LICENSE` file is not fetchable without an accepted-access, authenticated
token (`curl .../Llama-3.1-8B-Instruct/raw/main/LICENSE` returned "Access to
model ... is restricted. You must have access to it and be authenticated to
access it."). The Llama 3.1 Community License Agreement text is otherwise
published verbatim by Meta at the model-family level (llama.com and Meta's
`llama-models` GitHub repo mirror it identically as the license all
Llama-3.1 checkpoints — including 8B-Instruct — ship under). Fetched from:

`curl https://raw.githubusercontent.com/meta-llama/llama-models/main/models/llama3_1/LICENSE`

The output/derivatives-relevant clause, quoted verbatim (Section 1.b.i):

> i. If you distribute or make available the Llama Materials (or any
> derivative works thereof), or a product or service (including another AI
> model) that contains any of them, you shall (A) provide a copy of this
> Agreement with any such Llama Materials; and (B) prominently display
> "Built with Llama" on a related website, user interface, blogpost, about
> page, or product documentation. If you use the Llama Materials or any
> outputs or results of the Llama Materials to create, train, fine tune, or
> otherwise improve an AI model, which is distributed or made available,
> you shall also include "Llama" at the beginning of any such AI model
> name.

Also relevant — Section 3, Disclaimer of Warranty, which covers outputs
explicitly:

> UNLESS REQUIRED BY APPLICABLE LAW, THE LLAMA MATERIALS AND ANY OUTPUT AND
> RESULTS THEREFROM ARE PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF
> ANY KIND ...

**Verdict:** Llama 3.1 Community License (not Apache/MIT). Output use for
this pipeline (running closed-book QA prompts and grading/aggregating
retention statistics — no redistribution of the Llama Materials themselves,
no derivative model published, no "Built with Llama" trigger since we don't
ship a model) is compatible; the naming/attribution and "Built with Llama"
obligations only bite if a derivative model is distributed, which this
pipeline does not do. Flagging for controller awareness: if any pipeline
output artifact (e.g. raw generations) is ever published/redistributed
standalone, revisit Section 1.b.i and the Acceptable Use Policy at
`https://llama.meta.com/llama3_1/use-policy`.

---

## 7. Quantization repo (local: `/Users/rishikchakraborty/Documents/Coding/quantization`)

Verified locally via `git log` / `git show` on commit `0d1885e`:

```
commit 0d1885e97e626fdc9af5ad14d154a78cfa394f0e
Author: Rishik Chakraborty <rishikchak2008@gmail.com>
Date:   Wed Aug 26 12:43:01 2026 -0400

    Add Apache License 2.0

    Copyright 2026 Rishik Chakraborty. Repository previously had no license,
    which defaulted to all rights reserved and blocked any redistribution of
    extracted artefacts (flagged in multivalue2-bundle/PROVENANCE.md).

 LICENSE | 201 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 1 file changed, 201 insertions(+)
```

`LICENSE` file content begins:

```
                                Apache License
                          Version 2.0, January 2004
                       http://www.apache.org/licenses/
```

**Verdict: Apache-2.0**, added at commit `0d1885e` (2026-08-26). Matches
freeze-plan §4's resolved item #4.

---

## Summary table

| Repo | Type | License | Status |
|---|---|---|---|
| `akariasai/PopQA` | dataset | HF mirror card: none/missing; **canonical release (AlexTMallen/adaptive-retrieval): MIT** | resolved — MIT via canonical source, in use |
| `mandarjoshi/trivia_qa` | dataset | `unknown` (HF tag) | named fallback, unused |
| `openai/gsm8k` | dataset | MIT | in use (arithmetic suite) |
| `sgoel9/paul_graham_essays` | dataset | MIT uploader tag; underlying essays unresolved | quarantined to never-published 2a run |
| `Qwen/Qwen2.5-1.5B-Instruct` | model | Apache-2.0 | clear |
| `Qwen/Qwen2.5-7B-Instruct` | model | Apache-2.0 | clear |
| `meta-llama/Llama-3.1-8B-Instruct` | model | Llama 3.1 Community License | clear for pipeline's non-redistributive use; flagged obligations above |
| quantization repo (local, `0d1885e`) | code | Apache-2.0 | clear |
