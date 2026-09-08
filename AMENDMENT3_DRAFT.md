# AMENDMENT3_DRAFT — PREREG §5 shootout trigger fired

**Status: SUPERSEDED — stamped into PREREG.md as Amendment 3 on 2026-09-08 (commit `9f94913d322b009b214e2bfde80f1217e9d1da17`, receipt `freeze/amendment3-commit-hash.txt.ots`, pending Bitcoin attestation). This file is the reviewed draft, retained as history; PREREG.md's Amendment 3 is the registered text.**

The text below is what gets appended to PREREG.md, following Amendment 2,
upon the user's "stamp it".

---

# Amendment 3 (2026-09-08): PREREG §5 shootout trigger fired

Appended per §15.5 ("any registered follow-up trigger firing per §5" is
appended as a dated section, committed, and stamped). Records a fact
produced by the registered §8 machinery; alters no registered rule.

Amendment 3 commit: [filled by the follow-up commit]
Amendment 3 OpenTimestamps proof: `freeze/amendment3-commit-hash.txt.ots`
[filled by the follow-up commit]

## A. What fired, and when

The §5 trigger was first evaluated on 2026-09-03 (`scripts/analyze_0b.py`,
commit `be3b07d`, merge `b2f29c1`) and read FIRED. It read FIRED again on
the final registered data (factual_qa re-sourced from run 0b2 per
OPEN_QUESTIONS §8; commits `bbb1ef7` 2026-09-04 and `47e8462` 2026-09-06)
and under all four robustness seeds (FINDINGS_0B.md, seed sweep). Eleven
of the sixteen firing pairs lie in suites untouched by the §8 item-set
substitution, so the firing does not depend on it.

## B. The pairs that fired it (FINDINGS_0B.md, "Uploader shootout trigger")

Condition (i), CI excludes 0 — one pair:
- Arm 1, Q3_K_M, arithmetic_twins: unsloth_Q3_K_M vs
  mradermacher_static_Q3_K_M, Δ = −0.0957, 95% CI [−0.1915, −0.0106];
  states indeterminate vs damaged.

Condition (ii), same-label files in different §8 states — fifteen further
pairs (Δ, CI, state A / state B):
- Arm 1, Q4_K_M, arithmetic: bartowski vs unsloth +0.0060 [−0.0200, 0.0320]
  equivalent / indeterminate; bartowski vs mradermacher_static +0.0060
  [−0.0240, 0.0360] equivalent / indeterminate; bartowski vs
  mradermacher_i1 +0.0060 [−0.0220, 0.0360] equivalent / indeterminate.
- Arm 1, Q4_K_M, factual_qa: bartowski vs mradermacher_static +0.0140
  [−0.0040, 0.0340] indeterminate / equivalent; unsloth vs
  mradermacher_static +0.0060 [−0.0160, 0.0280] indeterminate / equivalent;
  mradermacher_static vs mradermacher_i1 −0.0020 [−0.0220, 0.0160]
  equivalent / indeterminate.
- Arm 1, Q3_K_M, arithmetic: bartowski vs unsloth −0.0200 [−0.0520, 0.0120]
  indeterminate / damaged; bartowski vs mradermacher_static −0.0160
  [−0.0480, 0.0160] indeterminate / damaged; unsloth vs mradermacher_i1
  +0.0040 [−0.0280, 0.0380] damaged / indeterminate; mradermacher_static vs
  mradermacher_i1 0.0000 [−0.0300, 0.0300] damaged / indeterminate.
- Arm 1, Q3_K_M, arithmetic_twins: bartowski vs unsloth +0.0745 [−0.0106,
  0.1596] damaged / indeterminate; unsloth vs mradermacher_i1 −0.0957
  [−0.1915, 0.0000] indeterminate / damaged.
- Arm 2, Q4_K_M, factual_qa: official vs bartowski −0.0080 [−0.0220,
  0.0060] equivalent / small_real_loss.
- Arm 2, Q3_K_M, arithmetic: official vs bartowski −0.0060 [−0.0280,
  0.0160] equivalent / indeterminate.
- Arm 2, Q3_K_M, factual_qa: official vs bartowski +0.0060 [−0.0140,
  0.0260] damaged / small_real_loss.

Full table: `bitcliff/pipeline/analysis/0b/FINDINGS_0B.md`, section
"Uploader shootout trigger (PREREG §5)".

## C. The mandated follow-up (§5, verbatim consequence)

PREREG §5's registered consequence, verbatim:

> extending the
> uploader shootout to Qwen2.5-7B-Instruct becomes a **registered follow-up
> measurement** under the same rules, run after the launch analyses.

The Qwen2.5-7B shootout file manifest will be committed, hashed, and stamped as its own dated section before that run; it is not part of this amendment.

## D. What this amendment does NOT do

- Does not change §5's scope, the trigger definition, or any registered
  rule, seed, n, or grading rule.
- Does not schedule or authorize the follow-up spend; timing is bound only
  by §5's "run after the launch analyses."
- Does not alter any 0B cell, cliff, or Holm verdict.
