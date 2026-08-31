# AMENDMENT2_DRAFT — registered reference ladder (registration-gap fix)

**STATUS: DRAFT FOR USER REVIEW. Not appended to PREREG. Not committed as
an amendment. Not OTS-stamped. Nothing confirmatory runs before this is
appended and stamped ("stamp it").**

The text below is what gets appended to PREREG.md, following Amendment 1,
upon the user's "stamp it".

---

# Amendment 2 (2026-08-30): the registered reference ladder

Appended per the disclosed-gap mechanics of Amendment 1 §A/§A2 (a
registration-gap fix beyond §7's knob-amendment scope, disclosed in its
own right), after user review of the standalone draft
(`AMENDMENT2_DRAFT.md`) and explicit user approval ("stamp it",
2026-08-30). Committed and OpenTimestamps-stamped like the original; this
amendment's own commit hash and receipt are recorded by the follow-up
commit:

Amendment 2 commit: [TO BE FILLED by the follow-up commit]
Amendment 2 OpenTimestamps proof: `freeze/amendment2-commit-hash.txt.ots`
[TO BE FILLED after stamping]

## A. What PREREG omitted

§4 registers the reference ladders as "pinned to the committed manifests
(`bitcliff/pipeline/reference-manifests/`)" — provenance pinning, file by
file, sha256 by sha256 — but never enumerates WHICH of each manifest's 24
files constitute the confirmatory ladder. The manifests carry everything
the uploader ships (including _S/_L/_XL size variants, the Q4_0 family,
ARM-repacked files, and full-precision conversions); no registered
sentence selects the run matrix from them. The ladder is the
row-definition of every reference table, so this gap must be closed
before any confirmatory generation. Logged as `OPEN_QUESTIONS.md` §6
before any confirmatory run; resolved by user ruling 2026-08-30.

## B. Sources the ruling carries into registered text

The design doc as of the freeze (`claude/IDEA-v1.md` §4 — at
`claude/IDEA.md` until 2026-08-30, preserved verbatim at the v1 path;
unregistered until now) defines the ladder shape, quoted verbatim:

> **The ladder** runs from full precision down to the deranged zone: F16
> (converted locally as the reference, never a download recommendation),
> then Q8, Q6, Q5, Q4, Q3, down to the lowest level the tracked uploaders
> actually publish, expected around IQ2_XXS, with IQ1_S included only
> where it exists.

and, for the file-not-label rule and the canonical uploader (ibid.):

> **A quant level is a file, not a label.** […] The canonical ladder uses
> bartowski's imatrix quants.

The file-level pilot precedent (`plan.md`, verified against the live 1.5B
repo 2026-08-26, quoted verbatim):

> **Pilot ladder (verified against the live HF repo
> `bartowski/Qwen2.5-1.5B-Instruct-GGUF` on 2026-08-26):** Q8_0, Q6_K,
> Q5_K_M, Q4_K_M, Q3_K_M, Q2_K, IQ2_M. The repo publishes nothing below
> IQ2_M […]

## C. The registered rule (user ruling, 2026-08-30)

**Per reference model, the confirmatory ladder is: F16 (the local
official-repo conversion, baseline only, never a download
recommendation) + Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_M, Q2_K, plus the
lowest rung the tracked uploader publishes below Q2_K** — one file per
rung, drawn from the pinned bartowski imatrix manifests (PREREG §4
table), each result pinned to that file's sha256.

**Manifest resolution, stated plainly:** IDEA-v1.md §4 expected the bottom
"around IQ2_XXS, with IQ1_S included only where it exists." The pinned
manifests (llama-3.1-8b-bartowski.json rev `bf5b95e9…`,
qwen2.5-7b-bartowski.json rev `8911e8a4…`) carry **no IQ2_XXS, no IQ1_S,
no IQ1_M for either model**; the lowest published rung below Q2_K is
**IQ2_M in both**. By the "lowest published" rule, **the registered
bottom rung is IQ2_M for both reference models.** The expectation named
IQ2_XXS; the manifest reality is IQ2_M; the rule, not the expectation,
governs.

**Registered reference ladder, both models (8 rows):**
F16 (local) · Q8_0 · Q6_K · Q5_K_M · Q4_K_M · Q3_K_M · Q2_K · IQ2_M

Every other manifest file (IQ3/IQ4 variants, _S/_L/_XL sizes, Q4_0
family, ARM-repacked Q4_0_x_x, f32/f16 uploads) remains
provenance-pinned by §4 but is **not** a confirmatory rung and enters no
reference table. The shootout and official-vs-community arms (§5) are
unchanged by this amendment.

## D. Cell family and multiplicity

The confirmatory reference family is **2 models × 7 quant rungs × 4
scored suites = 56 cells**, plus 32 arm-scoped cells (8 arm files × 4
suites) outside the reference matrix. The 56-cell family matches the
launch-scope anticipation this ruling itself states (no external source
is cited for the anticipation). §8's dual rule applies unchanged:
per-cell verdicts descriptive at α = 0.05; any cross-cell headline
survives Holm over the family it aggregates across.

## E. Alternatives considered and rejected (user ruling, 2026-08-30)

- **Full manifest ladder (20 rungs/model, 160 reference cells)** and
  **family ladder + IQ3/IQ4 mid-rungs (10–11 rungs, 80–88 cells)**:
  REJECTED — both inflate the Holm family well beyond the anticipated
  launch scope.
- **In-house quantization of IQ2_XXS/IQ1_S for the reference models**:
  REJECTED — in-house rungs are 1.5B-spectacle-only (IDEA-v1.md §4's
  exception "exists to protect download recommendations, and the 1.5B is
  not one"; PREREG §4 bars `bitcliff-inhouse` rungs from reference
  tables, enforced in code).

## F. What this amendment does NOT do

- Does not modify any registered rule, seed, n, grading rule, budget, or
  the §4 manifests; it fills the rung-enumeration slot §4 never carried.
- Does not change the shootout/official arms (§5) or the 1.5B ladder
  (0B′ scope).
- Does not promote IDEA-v1.md to registered status beyond the sentences
  quoted here, and cites no other design document.
