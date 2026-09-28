"""Boot-time item-set hash gate (`assert_item_sets_match_registered`,
pre-rerun hardening OPEN_QUESTIONS §8) vs. every COMMITTED confirmatory run
record under `runs-cloud/pipeline/runs/`: the same carrier-honesty spirit as
`test_0b_configs.py`'s config-vs-PREREG cross-check, but here checking the
gate against the actual item sets those runs generated on, loaded straight
from each run's committed `items.jsonl` (no suite builders, no network, no
model -- just JSON + the gate function itself).

Controller ruling (this task's brief, citing OPEN_QUESTIONS §8): the
first-pass `configs/0b/*.yaml` runs' factual_qa items are the DISCLOSED
non-registered set (item-set sha256 `ac5cb282...`) -- `configs/0b/*.yaml`
hand-copied PREREG §7's M3 weight vector in prose order instead of the
ascending-s_pop convention order `factual_qa.items_from_records` needs. The
corrected `configs/0b2/*.yaml` reruns (factual_qa only) reproduce the
REGISTERED set (`2e53ca0e...`). So for each `configs/0b/*.yaml` config:
gate its own run's longctx_retrieval / arithmetic / arithmetic_twins items
(unaffected by the bug), gate the matching `0b2-*` run's factual_qa items
(the corrected set), and separately assert the first-pass run's OWN
factual_qa items are REJECTED by the gate -- this documents §8 as a
regression test, not just prose. `configs/0b2/*.yaml` configs carry only a
factual_qa suite block and gate against their own run's items.
"""

import json
from pathlib import Path

import pytest

from bitcliff_pipeline.__main__ import assert_item_sets_match_registered
from bitcliff_pipeline.config import load_config
from bitcliff_pipeline.items import EvalItem

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_0B_DIR = PIPELINE_ROOT / "configs" / "0b"
CONFIGS_0B2_DIR = PIPELINE_ROOT / "configs" / "0b2"
RUNS_DIR = PIPELINE_ROOT / "runs-cloud" / "pipeline" / "runs"

ALL_0B_CONFIG_PATHS = sorted(CONFIGS_0B_DIR.glob("*.yaml"))
ALL_0B2_CONFIG_PATHS = sorted(CONFIGS_0B2_DIR.glob("*.yaml"))


def _load_run_items(run_id: str) -> list[EvalItem]:
    """Loads a committed run's `items.jsonl` back into `EvalItem`s -- the
    exact shape `build_items` would have produced, so the gate sees the
    same thing it saw (or would have seen) at boot time for that run.
    `expected`/`prompt_tokens` are JSON lists on disk; `EvalItem` declares
    them as tuples (and `hashing.item_set_sha256` sorts `it.expected`,
    which a `None` survives untouched either way), so both are converted
    back to tuples here rather than left as lists.
    """
    path = RUNS_DIR / run_id / "items.jsonl"
    items = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        items.append(
            EvalItem(
                id=row["id"],
                suite=row["suite"],
                prompt=row["prompt"],
                expected=tuple(row["expected"]) if row["expected"] is not None else None,
                prompt_tokens=(
                    tuple(row["prompt_tokens"]) if row["prompt_tokens"] is not None else None
                ),
            )
        )
    return items


def _by_suite(items: list[EvalItem], suite: str) -> list[EvalItem]:
    return [it for it in items if it.suite == suite]


@pytest.mark.parametrize("path", ALL_0B_CONFIG_PATHS, ids=lambda p: p.stem)
def test_0b_config_gates_its_own_unaffected_suites_against_its_committed_run(path):
    """longctx_retrieval, arithmetic, and arithmetic_twins were unaffected
    by the OPEN_QUESTIONS §8 bug (only factual_qa's weight vector was
    hand-copied wrong) -- this run's own committed items for those three
    suites must pass the gate as-is."""
    cfg = load_config(path)
    run_id = path.stem
    items = _load_run_items(run_id)
    unaffected = [it for it in items if it.suite != "factual_qa"]
    assert {it.suite for it in unaffected} == {
        "longctx_retrieval",
        "arithmetic",
        "arithmetic_twins",
    }
    assert_item_sets_match_registered(unaffected, cfg.model_id)  # must not raise


@pytest.mark.parametrize("path", ALL_0B_CONFIG_PATHS, ids=lambda p: p.stem)
def test_0b_config_gates_the_matching_0b2_reruns_corrected_factual_qa_items(path):
    """The matching `0b2-*` rerun's factual_qa items (the corrected,
    registered draw) must pass the gate under this `0b-*` config's own
    model_id -- factual_qa's registered item-set hash is model-independent
    (Amendment 1 §C), but this cross-checks the actual committed bytes,
    not just the constant."""
    cfg = load_config(path)
    run_id = path.stem
    corrected_run_id = "0b2-" + run_id[len("0b-") :]
    corrected_items = _load_run_items(corrected_run_id)
    factual_qa = _by_suite(corrected_items, "factual_qa")
    assert len(factual_qa) == 500
    assert_item_sets_match_registered(factual_qa, cfg.model_id)  # must not raise


@pytest.mark.parametrize("path", ALL_0B_CONFIG_PATHS, ids=lambda p: p.stem)
def test_0b_configs_own_first_pass_factual_qa_items_are_rejected_by_the_gate(path):
    """Regression test for OPEN_QUESTIONS §8 itself: this `0b-*` run's OWN
    factual_qa items (item-set sha256 `ac5cb282...`, the hand-copied-wrong
    M3 vector's draw) must NOT pass the gate under the registered hash --
    if this ever silently started passing, the gate has stopped doing its
    job."""
    cfg = load_config(path)
    run_id = path.stem
    items = _load_run_items(run_id)
    factual_qa = _by_suite(items, "factual_qa")
    assert len(factual_qa) == 500
    with pytest.raises(RuntimeError, match="factual_qa"):
        assert_item_sets_match_registered(factual_qa, cfg.model_id)


@pytest.mark.parametrize("path", ALL_0B2_CONFIG_PATHS, ids=lambda p: p.stem)
def test_0b2_config_gates_its_own_committed_factual_qa_items(path):
    """`configs/0b2/*.yaml` carries only a factual_qa suite block (the
    targeted rerun) -- gate its own run's factual_qa items directly."""
    cfg = load_config(path)
    run_id = path.stem
    items = _load_run_items(run_id)
    assert {it.suite for it in items} == {"factual_qa"}
    assert_item_sets_match_registered(items, cfg.model_id)  # must not raise
