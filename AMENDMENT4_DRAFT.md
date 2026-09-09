# AMENDMENT4_DRAFT — the 1.5B confirmatory ladder (slot fill)

**STATUS: DRAFT FOR USER REVIEW. Not appended to PREREG. Not committed as
an amendment. Not OTS-stamped. Appended and stamped only on the user's
"stamp it".**

The text below is what gets appended to PREREG.md, following Amendment 3,
upon the user's "stamp it".

---

# Amendment 4 (2026-09-08): the registered 1.5B confirmatory ladder

Appended per the disclosed-gap mechanics of Amendment 1 §A/§A2 and
Amendment 2 (a registration-gap fix beyond §7's knob-amendment scope,
disclosed in its own right), after user review of the standalone draft
(`AMENDMENT4_DRAFT.md`) and explicit user approval ("stamp it"). Committed
and OpenTimestamps-stamped like the original; this amendment's own commit
hash and receipt are recorded by the follow-up commit:

Amendment 4 commit: [filled by the follow-up commit]
Amendment 4 OpenTimestamps proof: `freeze/amendment4-commit-hash.txt.ots`
[filled by the follow-up commit]

## A. The slot this fills

Amendment 2 §F reserved it: "Does not change the shootout/official arms (§5)
or the 1.5B ladder (0B′ scope)." §4 registers Qwen2.5-1.5B-Instruct as the
spectacle model and pins the two reference ladders to committed manifests,
but no manifest for the 1.5B's tracked-uploader repo was ever committed, and
`configs/qwen2.5-1.5b-pilot.yaml` names files without hashes. §10's
blind-check materials and §3.1 configuration 2a both presuppose a
confirmatory 1.5B run, so the 1.5B ladder must be enumerated and pinned
before that run exists. This amendment fills only that slot: the rung set
and the file pins. It decides nothing about which suites run, on which
machine, with which divergence passes, or when.

## B. The rule applied

Amendment 2 §C's registered rule, applied unchanged to the 1.5B (quoted
once, verbatim):

> **Per reference model, the confirmatory ladder is: F16 (the local
> official-repo conversion, baseline only, never a download
> recommendation) + Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_M, Q2_K, plus the
> lowest rung the tracked uploader publishes below Q2_K** — one file per
> rung, drawn from the pinned bartowski imatrix manifests (PREREG §4
> table), each result pinned to that file's sha256.

Tracked uploader repo: `bartowski/Qwen2.5-1.5B-Instruct-GGUF`, enumerated
2026-09-08 via the Hugging Face API at its current revision
`9eadc66189c7641e1ddd226b8267a9119b2ce2d4` (repo last modified 2024-09-19),
per-file sha256 taken from the LFS metadata, no file downloaded. The repo
ships `Qwen2.5-1.5B-Instruct.imatrix`; every quant in it is imatrix-calibrated.
"Below Q2_K" is resolved by file size, the same way Amendment 2 §C resolved
IQ2_M for the reference models.

**Every candidate rung, published or not, included or not:**

| candidate | published at `9eadc661…`? | rule verdict |
|---|---|---|
| F16 | (local conversion, not an upload) | INCLUDED — baseline, Amendment 2 §C convention |
| Q8_0 | yes | INCLUDED — named rung |
| Q6_K | yes | INCLUDED — named rung |
| Q5_K_M | yes | INCLUDED — named rung |
| Q4_K_M | yes | INCLUDED — named rung |
| Q3_K_M | yes | INCLUDED — named rung |
| Q2_K | yes | INCLUDED — named rung |
| IQ2_M | yes (601,054,816 B) | INCLUDED — the only published file smaller than Q2_K (676,304,992 B); lowest published below Q2_K |
| IQ2_XXS | no | not published; cannot be a rung |
| IQ2_XS | no | not published |
| IQ2_S | no | not published |
| IQ1_M | no | not published |
| IQ1_S | no | not published |
| Q2_K_L | yes | EXCLUDED — size variant, larger than Q2_K, so not "below Q2_K" |
| IQ3_M, IQ3_XS, IQ4_XS | yes | EXCLUDED — IQ mid-rungs outside the named set (Amendment 2 §E rejected adding them) |
| Q3_K_S, Q3_K_L, Q3_K_XL, Q4_K_S, Q4_K_L, Q5_K_S, Q5_K_L, Q6_K_L | yes | EXCLUDED — size variants outside the named set |
| Q4_0, Q4_0_4_4, Q4_0_4_8, Q4_0_8_8 | yes | EXCLUDED — Q4_0 family; the three repacked files are also unloadable by the pinned llama.cpp build |
| f16 (uploader's upload) | yes | EXCLUDED — not a quant; the F16 baseline is the local conversion, per §C convention |

## C. The registered 1.5B confirmatory ladder (8 rows)

F16 (local) · Q8_0 · Q6_K · Q5_K_M · Q4_K_M · Q3_K_M · Q2_K · IQ2_M

| rung | file | sha256 |
|---|---|---|
| F16 | `models/f16/Qwen2.5-1.5B-Instruct-f16.gguf` (local conversion; `PILOT_RUNBOOK.md` recipe; same sha256 recorded in `runs/pilot-0a/manifest.json` and `calibration/qwen2.5-1.5b-instruct/summary.json`) | `954b449288dd7989c3ddc193666c88d1bb43857c1a2468d04e422b47cc234fe4` |
| Q8_0 | `Qwen2.5-1.5B-Instruct-Q8_0.gguf` | `7185d306cf45956c8c017cd0d3b05ecc6bc18b3ea8eb5c240dce40e87563db7f` |
| Q6_K | `Qwen2.5-1.5B-Instruct-Q6_K.gguf` | `1b01b4ea4ccdd5aa6a5972790002e120a5d500a5175be571114d642a8db4d14e` |
| Q5_K_M | `Qwen2.5-1.5B-Instruct-Q5_K_M.gguf` | `cf240adc57e126e86102335f6565fb23e523b28d287c75bdb0759f064e8bb572` |
| Q4_K_M | `Qwen2.5-1.5B-Instruct-Q4_K_M.gguf` | `1adf0b11065d8ad2e8123ea110d1ec956dab4ab038eab665614adba04b6c3370` |
| Q3_K_M | `Qwen2.5-1.5B-Instruct-Q3_K_M.gguf` | `7437ad04011a14fb890074cc783df4c1d537197942337353a824ff7e6115ef9b` |
| Q2_K | `Qwen2.5-1.5B-Instruct-Q2_K.gguf` | `a8880f0de2348db67d00519ef7f4b40326ef67012bf5f2e90bd1d47474e2355c` |
| IQ2_M | `Qwen2.5-1.5B-Instruct-IQ2_M.gguf` | `cee720c998e71ff3f02fbb3d392c7598bc0f845ae08bbb585ef2ff5fcbd45b81` |

The Q4_K_M and IQ2_M LFS hashes match the pilot's local copies of those
files byte-for-byte. The pilot ladder (`plan.md`, verified 2026-08-26
against this same repo) named the identical seven quant files.

## D. The manifest

`bitcliff/pipeline/reference-manifests/qwen2.5-1.5b-bartowski.json`,
sha256 `d4849ea64d88cc3af14e42c8a6ef0e3a7d49a2fd4f95b7d34899e526118a2b47`:
repo, revision sha, retrieval timestamp, enumeration date, imatrix status
and its evidence, the rule text, the resulting ladder, the F16 local
conversion block, and all 25 published GGUF files with size, sha256,
imatrix flag, and in-ladder verdict. Every other file in it remains
provenance-pinned but is not a confirmatory rung and enters no reference
table.

## E. Exclusion

The three in-house sub-2-bit files (IQ2_XXS, IQ1_M, IQ1_S; uploader
`bitcliff-inhouse`; `reference-manifests/qwen2.5-1.5b-inhouse.json`, sha256
`e4f44feb134586daf0812ef07efca5b68563654ed90af66d82203e173f6fd8a0`) are
`spectacle_only` per §4 and are not part of this ladder.

## F. What this amendment does NOT do

- Does not decide which suites the 1.5B runs, on which machine or
  fingerprint, whether any teacher-forced divergence pass runs, or when;
  those are run-plan decisions, not registration, and live in
  `RUN_0B_PRIME.md`.
- Does not modify any registered rule, seed, n, grading rule, budget,
  margin, or the §4 manifests for the reference models; it fills the
  1.5B rung-enumeration slot Amendment 2 §F reserved.
- Does not register the in-house files as rungs, does not promote the
  uploader's f16 upload to a rung, and does not change the 1.5B's
  spectacle status (§4).
- Does not authorize any spend.
