import dataclasses
import hashlib
import json
import re
from pathlib import Path

import pytest

import bitcliff_pipeline.__main__ as main_mod
import bitcliff_pipeline.generate as gen_mod
import bitcliff_pipeline.grading as grading_mod
from bitcliff_pipeline.__main__ import build_items, run_pipeline
from bitcliff_pipeline.config import GenSettings, LadderConfig, QuantFile
from bitcliff_pipeline.items import EvalItem
from bitcliff_pipeline.suites import arithmetic, factual_qa

# A tiny synthetic gsm8k-shaped record set, injected via
# `arithmetic.items_from_records` in place of the real (networked) GSM8K
# dataset — see the `fake_gsm8k` fixture below. Kept in the same shape as
# test_arithmetic.py's RECORDS so it exercises the real extraction/grading
# logic, not a mock of it.
ARITHMETIC_RECORDS = [
    {
        "question": f"Q{i}: If x = {i} and y = {i}, what is x + y?",
        "answer": f"x + y = {i} + {i} = {2 * i}\n#### {2 * i}",
    }
    for i in range(1, 21)
]


def make_config(tmp_path: Path, spectacle_only: bool = False) -> LadderConfig:
    spectacle = tmp_path / "configs" / "prompts_spectacle.yaml"
    spectacle.parent.mkdir(parents=True)
    spectacle.write_text(
        "prompts:\n  - {id: spec-001, prompt: 'Say hi.'}\n"
        "  - {id: spec-002, prompt: 'Say bye.'}\n"
    )
    return LadderConfig(
        model_id="test-model",
        hf_repo="fake/repo",
        f16_path=tmp_path / "models" / "f16.gguf",
        quants=(QuantFile("Q4_K_M", "m-Q4_K_M.gguf", "bartowski", True, spectacle_only=spectacle_only),),
        generation=GenSettings(42, 0.0, 1, 640, 4096),
        suites={
            "arithmetic": {"n_items": 4, "seed": 1},
            "spectacle": {"path": "configs/prompts_spectacle.yaml"},
        },
    )


class FakeLlm:
    """Echoes the prompt back for spectacle (unscored) items. For arithmetic
    items it solves the embedded "x = N and y = N" pattern so the scored-
    suite path (aggregate/retention) is still exercised end-to-end without a
    real model or network access.
    """

    _ARITH_RE = re.compile(r"x = (\d+) and y = (\d+)")

    def tokenize(self, text, add_bos=True, special=False):
        return list(range(len(text.split())))

    def create_chat_completion(self, messages, **kwargs):
        prompt = messages[0]["content"]
        m = self._ARITH_RE.search(prompt)
        content = f"#### {int(m.group(1)) + int(m.group(2))}" if m else prompt
        return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}

    def create_completion(self, prompt, max_tokens, **kwargs):
        # the truncation preflight (generate stage) probes via the token
        # path with a small budget and expects an honest "length"
        return {"choices": [{"text": "1, 2, 3", "finish_reason": "length"}]}


@pytest.fixture(autouse=True)
def fake_gsm8k(monkeypatch):
    """build_items()'s arithmetic branch normally calls arithmetic.load_gsm8k_items,
    which hits the network (HF `datasets`). Redirect it to a tiny local record
    set via items_from_records, exactly as suggested for testing that function.
    """
    monkeypatch.setattr(
        arithmetic,
        "load_gsm8k_items",
        lambda n_items, seed: arithmetic.items_from_records(
            ARITHMETIC_RECORDS, n_items, seed
        ),
    )


def test_pipeline_end_to_end(tmp_path):
    cfg = make_config(tmp_path)
    models_dir = tmp_path / "models"
    models_dir.mkdir(exist_ok=True)
    (models_dir / "m-Q4_K_M.gguf").write_bytes(b"quant bytes")
    cfg.f16_path.write_bytes(b"f16 bytes")
    runs_dir = tmp_path / "runs"

    run_pipeline(
        cfg, run_id="pilot-test", models_dir=models_dir, runs_dir=runs_dir,
        stage="all", llm_factory=lambda path, gen: FakeLlm(), base_dir=tmp_path,
    )

    run = runs_dir / "pilot-test"
    assert (run / "manifest.json").exists()
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["Q4_K_M"]["uploader"] == "bartowski"
    assert manifest["Q4_K_M"]["imatrix"] is True
    assert manifest["Q4_K_M"]["spectacle_only"] is False
    assert manifest["F16"]["uploader"] == "bitcliff-local-f16-conversion"
    assert manifest["F16"]["spectacle_only"] is False
    assert (run / "outputs" / "F16.jsonl").exists()
    assert (run / "outputs" / "Q4_K_M.jsonl").exists()
    grades = [json.loads(l) for l in (run / "grades.jsonl").read_text().splitlines()]
    # (4 arithmetic + 2 spectacle) items, for 2 ladder rungs
    assert len(grades) == 12
    assert {g["quant_label"] for g in grades} == {"F16", "Q4_K_M"}
    assert all("divergence" in g for g in grades)
    results = json.loads((run / "results.json").read_text())
    assert results  # the arithmetic suite produced scored rows
    assert all(r["retention"] == 1.0 for r in results)  # identical fake outputs
    assert (run / "retention.png").exists()


def test_grade_stage_rejects_stale_outputs(tmp_path):
    cfg = make_config(tmp_path)
    models_dir = tmp_path / "models"
    models_dir.mkdir(exist_ok=True)
    (models_dir / "m-Q4_K_M.gguf").write_bytes(b"quant bytes")
    cfg.f16_path.write_bytes(b"f16 bytes")
    runs_dir = tmp_path / "runs"

    run_pipeline(
        cfg, run_id="pilot-test", models_dir=models_dir, runs_dir=runs_dir,
        stage="all", llm_factory=lambda path, gen: FakeLlm(), base_dir=tmp_path,
    )

    # Simulate a stale items.jsonl: rewrite one item's prompt while keeping its id.
    # Outputs from the earlier "all" run still hold the old prompt.
    run = runs_dir / "pilot-test"
    items_path = run / "items.jsonl"
    items = [json.loads(l) for l in items_path.read_text().splitlines()]
    items[0]["prompt"] = items[0]["prompt"] + " (edited)"
    items_path.write_text("".join(json.dumps(i) + "\n" for i in items))

    with pytest.raises(RuntimeError, match="prompt mismatch"):
        run_pipeline(
            cfg, run_id="pilot-test", models_dir=models_dir, runs_dir=runs_dir,
            stage="grade", llm_factory=lambda path, gen: FakeLlm(), base_dir=tmp_path,
        )


def test_manifest_records_run_config(tmp_path):
    """Pre-0B ticket (freeze-plan §10): manifest.json carries the run's
    suite config so the packager can positively identify corpus provenance
    (e.g. a longctx run's corpus_sha256) instead of the seed heuristic."""
    cfg = make_config(tmp_path)
    cfg = dataclasses.replace(
        cfg,
        suites={
            **cfg.suites,
            "longctx_retrieval": {
                "variant": "multivalue4",
                "target_tokens": 8192,
                "seed": 2024,
                "n_items": 96,
                "corpus_sha256": "0a21a13834b5215876bd4019af8fbc436abbfbb61b2826db62223eb990071443",
                "max_tokens": 32,
            },
        },
    )
    models_dir = tmp_path / "models"
    models_dir.mkdir(exist_ok=True)
    (models_dir / "m-Q4_K_M.gguf").write_bytes(b"quant bytes")
    cfg.f16_path.write_bytes(b"f16 bytes")
    runs_dir = tmp_path / "runs"

    run_pipeline(
        cfg, run_id="runconfig-test", models_dir=models_dir, runs_dir=runs_dir,
        stage="download", llm_factory=lambda path, gen: FakeLlm(), base_dir=tmp_path,
    )

    manifest = json.loads((runs_dir / "runconfig-test" / "manifest.json").read_text())
    rc = manifest["_run_config"]
    assert rc["model_id"] == "test-model"
    assert rc["suites"]["longctx_retrieval"]["corpus_sha256"] == (
        "0a21a13834b5215876bd4019af8fbc436abbfbb61b2826db62223eb990071443"
    )
    # metadata key must not look like a rung to downstream consumers
    assert set(manifest) - {"_run_config"} == {"F16", "Q4_K_M"}


def test_build_items_covers_configured_suites(tmp_path):
    cfg = make_config(tmp_path)
    items = build_items(cfg, base_dir=tmp_path)
    suites = [i.suite for i in items]
    assert suites.count("arithmetic") == 4
    assert suites.count("spectacle") == 2


def test_build_items_factual_qa_forwards_alias_augmentation_path(tmp_path, monkeypatch):
    """build_items' factual_qa branch reads an optional
    alias_augmentation_path key from the suite config, resolved against
    base_dir (same pattern as the spectacle suite's `path` key), and
    forwards it to load_popqa_items."""
    captured = {}

    def fake_load_popqa_items(n_items, seed, alias_augmentation_path=None, weights=None):
        captured["args"] = (n_items, seed, alias_augmentation_path, weights)
        return []

    monkeypatch.setattr(factual_qa, "load_popqa_items", fake_load_popqa_items)

    cfg = make_config(tmp_path)
    cfg = dataclasses.replace(
        cfg,
        suites={
            **cfg.suites,
            "factual_qa": {
                "n_items": 500,
                "seed": 7411,
                "alias_augmentation_path": "data/aliases.json",
            },
        },
    )
    build_items(cfg, base_dir=tmp_path)
    assert captured["args"] == (500, 7411, tmp_path / "data/aliases.json", None)


def test_build_items_factual_qa_without_alias_augmentation_path_passes_none(tmp_path, monkeypatch):
    captured = {}

    def fake_load_popqa_items(n_items, seed, alias_augmentation_path=None, weights=None):
        captured["args"] = (n_items, seed, alias_augmentation_path, weights)
        return []

    monkeypatch.setattr(factual_qa, "load_popqa_items", fake_load_popqa_items)

    cfg = make_config(tmp_path)
    cfg = dataclasses.replace(
        cfg,
        suites={**cfg.suites, "factual_qa": {"n_items": 500, "seed": 7411}},
    )
    build_items(cfg, base_dir=tmp_path)
    assert captured["args"] == (500, 7411, None, None)


def test_build_items_factual_qa_forwards_weights(tmp_path, monkeypatch):
    """0B P3 wiring gap: `items_from_records` already accepted a `weights`
    parameter (PREREG §7's popularity-mix knob), but `__main__.build_items`
    never read a `weights` key from the config block or passed it through.
    """
    captured = {}

    def fake_load_popqa_items(n_items, seed, alias_augmentation_path=None, weights=None):
        captured["args"] = (n_items, seed, alias_augmentation_path, weights)
        return []

    monkeypatch.setattr(factual_qa, "load_popqa_items", fake_load_popqa_items)

    m3 = [0.16, 0.16, 0.16, 0.16, 0.16, 0.04, 0.04, 0.04, 0.04, 0.04]
    cfg = make_config(tmp_path)
    cfg = dataclasses.replace(
        cfg,
        suites={
            **cfg.suites,
            "factual_qa": {"n_items": 500, "seed": 2718, "weights": m3},
        },
    )
    build_items(cfg, base_dir=tmp_path)
    assert captured["args"] == (500, 2718, None, tuple(m3))


class TokenAwareFakeLlm(FakeLlm):
    """Extends FakeLlm with create_completion for token-id items. Records
    the exact token list it was called with. Answers the truncation
    preflight (identified by its registered probe budget) honestly."""

    def create_completion(self, prompt, max_tokens=None, **kwargs):
        if max_tokens == gen_mod.TRUNCATION_PREFLIGHT_MAX_TOKENS:
            return super().create_completion(prompt, max_tokens, **kwargs)
        return {"choices": [{"text": "The passcode is 42.", "finish_reason": "stop"}]}


def test_token_id_item_roundtrips_through_items_jsonl_and_grades(tmp_path, monkeypatch):
    """Covers the Task-1 deferred gap: a token-id item (prompt_tokens set)
    must survive write -> items.jsonl -> read (reconstructed as a tuple) and
    the grade stage must actually run the longctx_retrieval grader over its
    generated record — not just parse it.
    """
    cfg = make_config(tmp_path)
    models_dir = tmp_path / "models"
    models_dir.mkdir(exist_ok=True)
    (models_dir / "m-Q4_K_M.gguf").write_bytes(b"quant bytes")
    cfg.f16_path.write_bytes(b"f16 bytes")
    runs_dir = tmp_path / "runs"

    token_item = EvalItem(
        "longctx_retrieval-1-0000",
        "longctx_retrieval",
        "Alice's code?",
        ("42",),
        prompt_tokens=(11, 12, 13),
    )
    original_build_items = main_mod.build_items

    def build_items_with_token_item(
        config, base_dir, longctx_tokenizer=None, longctx_answer_specs=None
    ):
        return original_build_items(
            config, base_dir, longctx_tokenizer=longctx_tokenizer,
            longctx_answer_specs=longctx_answer_specs,
        ) + [token_item]

    monkeypatch.setattr(main_mod, "build_items", build_items_with_token_item)

    # Spy on the real grader so we can assert, from inside the grade stage
    # itself, that item.prompt_tokens came back as a tuple after the
    # items.jsonl roundtrip — not just that grading happened to succeed.
    captured = {}
    original_grade = grading_mod.GRADERS["longctx_retrieval"]

    def spy_grade(item, text):
        captured["prompt_tokens"] = item.prompt_tokens
        return original_grade(item, text)

    monkeypatch.setitem(grading_mod.GRADERS, "longctx_retrieval", spy_grade)

    run_pipeline(
        cfg, run_id="token-test", models_dir=models_dir, runs_dir=runs_dir,
        stage="all", llm_factory=lambda path, gen: TokenAwareFakeLlm(), base_dir=tmp_path,
    )

    run = runs_dir / "token-test"
    items = [json.loads(l) for l in (run / "items.jsonl").read_text().splitlines()]
    on_disk = next(i for i in items if i["id"] == "longctx_retrieval-1-0000")
    assert on_disk["prompt_tokens"] == [11, 12, 13]  # JSON has no tuple type

    grades = [json.loads(l) for l in (run / "grades.jsonl").read_text().splitlines()]
    token_grades = [g for g in grades if g["item_id"] == "longctx_retrieval-1-0000"]
    assert len(token_grades) == 2  # one per ladder rung (F16, Q4_K_M)
    assert all(g["state"] == "correct" for g in token_grades)

    # the grade stage actually reconstructed prompt_tokens as a tuple, not a list
    assert captured["prompt_tokens"] == (11, 12, 13)
    assert isinstance(captured["prompt_tokens"], tuple)


# ---------------------------------------------------------------------------
# longctx_retrieval wiring — P1 (RUN_0B.md §2, PREREG §3.1). No network, no
# real HF/GGUF tokenizer or model: a word-splitting stub tokenizer (same
# shape as test_longctx_retrieval.py's StubTokenizer) stands in for the HF
# side, and FakeLlm subclasses stand in for the llama-cpp side.
# ---------------------------------------------------------------------------


class StubHfTokenizer:
    """Deterministic word<->id tokenizer exposing exactly the surface
    `longctx_retrieval.build_items` / `assert_tokenizer_match` need:
    `__call__`, `decode`, `apply_chat_template`, `name_or_path`.
    """

    def __init__(self, name: str = "stub-hf-tokenizer"):
        self.name_or_path = name
        self._word_to_id: dict[str, int] = {}
        self._id_to_word: dict[int, str] = {}

    def _id_for(self, word: str) -> int:
        if word not in self._word_to_id:
            i = len(self._word_to_id)
            self._word_to_id[word] = i
            self._id_to_word[i] = word
        return self._word_to_id[word]

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        return {"input_ids": [self._id_for(w) for w in text.split(" ")]}

    def decode(self, ids) -> str:
        return " ".join(self._id_to_word[i] for i in ids)

    def apply_chat_template(self, messages, tokenize: bool = False, add_generation_prompt: bool = True) -> str:
        return f"<|user|>{messages[0]['content']}<|assistant|>"


LONGCTX_CORPUS_TEXT = " ".join(f"corpusword{i}" for i in range(500))
LONGCTX_CORPUS_SHA256 = hashlib.sha256(LONGCTX_CORPUS_TEXT.encode("utf-8")).hexdigest()


def _longctx_config(tmp_path: Path, cfg: LadderConfig) -> LadderConfig:
    (tmp_path / "corpus.txt").write_text(LONGCTX_CORPUS_TEXT)
    return dataclasses.replace(
        cfg,
        suites={
            **cfg.suites,
            "longctx_retrieval": {
                "variant": "multivalue2",
                "target_tokens": 64,
                "seed": 7,
                "n_items": 3,
                "corpus_path": "corpus.txt",
                "corpus_sha256": LONGCTX_CORPUS_SHA256,
                "tokenizer_path": "tok",  # never actually loaded in these tests
                "max_tokens": 32,
            },
        },
    )


class LongctxFakeLlm(FakeLlm):
    """A llama-cpp-shaped fake whose `.tokenize` agrees word-for-word with
    a given StubHfTokenizer — i.e. its GGUF-side tokenizer matches the HF
    side, so PREREG §3.1's gate passes."""

    def __init__(self, hf_tokenizer: StubHfTokenizer):
        self._hf = hf_tokenizer

    def tokenize(self, text: bytes, add_bos: bool = True, special: bool = False):
        s = text.decode("utf-8") if isinstance(text, bytes) else text
        return self._hf(s, add_special_tokens=False)["input_ids"]

    def create_completion(self, prompt, max_tokens=None, **kwargs):
        if max_tokens == gen_mod.TRUNCATION_PREFLIGHT_MAX_TOKENS:
            return {"choices": [{"text": "1, 2, 3", "finish_reason": "length"}]}
        return {"choices": [{"text": "irrelevant completion", "finish_reason": "stop"}]}


class MismatchingLongctxFakeLlm(LongctxFakeLlm):
    """Diverges from the HF tokenizer on every sample — simulates a GGUF
    tokenizer that disagrees with the HF one, which the §3.1 gate must
    catch and abort on."""

    def tokenize(self, text, add_bos: bool = True, special: bool = False):
        return super().tokenize(text, add_bos=add_bos, special=special) + [999999]


def test_build_items_longctx_retrieval_wiring(tmp_path):
    cfg = _longctx_config(tmp_path, make_config(tmp_path))
    stub = StubHfTokenizer()

    items = build_items(cfg, base_dir=tmp_path, longctx_tokenizer=stub)

    longctx_items = [i for i in items if i.suite == "longctx_retrieval"]
    assert len(longctx_items) == 3
    assert all(i.prompt_tokens is not None for i in longctx_items)
    assert all(isinstance(i.prompt_tokens, tuple) for i in longctx_items)


def test_build_items_longctx_retrieval_corpus_hash_mismatch_raises(tmp_path):
    cfg = _longctx_config(tmp_path, make_config(tmp_path))
    cfg = dataclasses.replace(
        cfg,
        suites={**cfg.suites, "longctx_retrieval": {**cfg.suites["longctx_retrieval"], "corpus_sha256": "0" * 64}},
    )
    stub = StubHfTokenizer()

    with pytest.raises(AssertionError, match="sha256"):
        build_items(cfg, base_dir=tmp_path, longctx_tokenizer=stub)


def test_generate_stage_runs_tokenizer_gate_before_first_longctx_generation(tmp_path, monkeypatch):
    hf_tokenizer = StubHfTokenizer()
    monkeypatch.setattr(main_mod, "_load_hf_tokenizer", lambda path: hf_tokenizer)

    cfg = _longctx_config(tmp_path, make_config(tmp_path))
    models_dir = tmp_path / "models"
    models_dir.mkdir(exist_ok=True)
    (models_dir / "m-Q4_K_M.gguf").write_bytes(b"quant bytes")
    cfg.f16_path.write_bytes(b"f16 bytes")
    runs_dir = tmp_path / "runs"

    run_pipeline(
        cfg, run_id="longctx-gate-test", models_dir=models_dir, runs_dir=runs_dir,
        stage="all", llm_factory=lambda path, gen: LongctxFakeLlm(hf_tokenizer), base_dir=tmp_path,
    )

    run = runs_dir / "longctx-gate-test"
    assert (run / "outputs" / "F16.jsonl").exists()
    assert (run / "outputs" / "Q4_K_M.jsonl").exists()


def test_generate_stage_aborts_on_tokenizer_mismatch(tmp_path, monkeypatch):
    hf_tokenizer = StubHfTokenizer()
    monkeypatch.setattr(main_mod, "_load_hf_tokenizer", lambda path: hf_tokenizer)

    cfg = _longctx_config(tmp_path, make_config(tmp_path))
    models_dir = tmp_path / "models"
    models_dir.mkdir(exist_ok=True)
    (models_dir / "m-Q4_K_M.gguf").write_bytes(b"quant bytes")
    cfg.f16_path.write_bytes(b"f16 bytes")
    runs_dir = tmp_path / "runs"

    with pytest.raises(AssertionError, match="tokenizer mismatch"):
        run_pipeline(
            cfg, run_id="longctx-mismatch-test", models_dir=models_dir, runs_dir=runs_dir,
            stage="all", llm_factory=lambda path, gen: MismatchingLongctxFakeLlm(hf_tokenizer),
            base_dir=tmp_path,
        )

    run = runs_dir / "longctx-mismatch-test"
    # F16 is the first file gated in this invocation (paths.items() yields
    # F16 first, see models.resolve_all) and its fake tokenizer mismatches
    # here, so the gate fires on it before any output is written for any
    # file this run would otherwise generate -- not because F16 is special,
    # just because it happens to be first in this run's file order.
    assert not (run / "outputs" / "F16.jsonl").exists()
    assert not (run / "outputs" / "Q4_K_M.jsonl").exists()


def test_generate_stage_gates_second_file_even_after_first_files_gate_passed(tmp_path, monkeypatch):
    """Review finding (P1 follow-up): a once-per-invocation boolean gate
    would check only the first-loaded GGUF's tokenizer and let every other
    file in the same run_pipeline call generate unchecked. PREREG §4 ("a
    quant level is a file, not a label") means each file needs its own
    check: F16's fake tokenizer matches here (its gate passes, it
    generates), but the SECOND rung (Q4_K_M) uses a fake llm whose
    tokenizer diverges -- the per-file gate must still catch that and abort
    before Q4_K_M's generation, even though some other file in this same
    run already passed the gate.
    """
    hf_tokenizer = StubHfTokenizer()
    monkeypatch.setattr(main_mod, "_load_hf_tokenizer", lambda path: hf_tokenizer)

    cfg = _longctx_config(tmp_path, make_config(tmp_path))
    models_dir = tmp_path / "models"
    models_dir.mkdir(exist_ok=True)
    (models_dir / "m-Q4_K_M.gguf").write_bytes(b"quant bytes")
    cfg.f16_path.write_bytes(b"f16 bytes")
    runs_dir = tmp_path / "runs"

    def llm_factory(path, gen):
        if path == cfg.f16_path:
            return LongctxFakeLlm(hf_tokenizer)  # agrees with the HF tokenizer -> gate passes
        return MismatchingLongctxFakeLlm(hf_tokenizer)  # Q4_K_M's GGUF tokenizer diverges

    with pytest.raises(AssertionError, match="tokenizer mismatch"):
        run_pipeline(
            cfg, run_id="longctx-second-rung-mismatch-test", models_dir=models_dir, runs_dir=runs_dir,
            stage="all", llm_factory=llm_factory, base_dir=tmp_path,
        )

    run = runs_dir / "longctx-second-rung-mismatch-test"
    # F16's own gate passed, so F16 generated successfully...
    assert (run / "outputs" / "F16.jsonl").exists()
    # ...but Q4_K_M's gate must still fire on Q4_K_M's own (mismatching)
    # tokenizer and abort before Q4_K_M ever generates.
    assert not (run / "outputs" / "Q4_K_M.jsonl").exists()


def test_generate_stage_writes_aligned_answer_spans_for_longctx(tmp_path, monkeypatch):
    """0B P3 carried item (a): run_pipeline's generate stage must persist
    an answer_spans.jsonl sidecar aligned with items.jsonl's
    longctx_retrieval items (RUN_0B.md §2 P3(a) -- so a later P2 divergence
    pass can consume this run's items without rebuilding them)."""
    hf_tokenizer = StubHfTokenizer()
    monkeypatch.setattr(main_mod, "_load_hf_tokenizer", lambda path: hf_tokenizer)

    cfg = _longctx_config(tmp_path, make_config(tmp_path))
    models_dir = tmp_path / "models"
    models_dir.mkdir(exist_ok=True)
    (models_dir / "m-Q4_K_M.gguf").write_bytes(b"quant bytes")
    cfg.f16_path.write_bytes(b"f16 bytes")
    runs_dir = tmp_path / "runs"

    run_pipeline(
        cfg, run_id="answer-spans-test", models_dir=models_dir, runs_dir=runs_dir,
        stage="all", llm_factory=lambda path, gen: LongctxFakeLlm(hf_tokenizer), base_dir=tmp_path,
    )

    run = runs_dir / "answer-spans-test"
    spans = [json.loads(l) for l in (run / "answer_spans.jsonl").read_text().splitlines()]
    items = [json.loads(l) for l in (run / "items.jsonl").read_text().splitlines()]
    longctx_items = [i for i in items if i["suite"] == "longctx_retrieval"]

    assert len(spans) == len(longctx_items) == 3
    assert [s["item_id"] for s in spans] == [i["id"] for i in longctx_items]
    for s in spans:
        assert s["n_answer_tokens"] == len(s["answer_ids"])
        assert s["n_answer_tokens"] > 0


def test_generate_stage_no_answer_spans_file_without_longctx_suite(tmp_path):
    """No longctx_retrieval suite configured -> no answer_spans.jsonl at
    all (rather than an empty file), matching items.jsonl's own
    suite-conditional behavior."""
    cfg = make_config(tmp_path)
    models_dir = tmp_path / "models"
    models_dir.mkdir(exist_ok=True)
    (models_dir / "m-Q4_K_M.gguf").write_bytes(b"quant bytes")
    cfg.f16_path.write_bytes(b"f16 bytes")
    runs_dir = tmp_path / "runs"

    run_pipeline(
        cfg, run_id="no-longctx-test", models_dir=models_dir, runs_dir=runs_dir,
        stage="all", llm_factory=lambda path, gen: FakeLlm(), base_dir=tmp_path,
    )

    assert not (runs_dir / "no-longctx-test" / "answer_spans.jsonl").exists()


def test_spectacle_only_flag_in_results(tmp_path):
    """Test that spectacle_only=True in config creates rows with spectacle_only=True."""
    cfg = make_config(tmp_path, spectacle_only=True)
    models_dir = tmp_path / "models"
    models_dir.mkdir(exist_ok=True)
    (models_dir / "m-Q4_K_M.gguf").write_bytes(b"quant bytes")
    cfg.f16_path.write_bytes(b"f16 bytes")
    runs_dir = tmp_path / "runs"

    run_pipeline(
        cfg, run_id="spectacle-test", models_dir=models_dir, runs_dir=runs_dir,
        stage="all", llm_factory=lambda path, gen: FakeLlm(), base_dir=tmp_path,
    )

    run = runs_dir / "spectacle-test"
    results = json.loads((run / "results.json").read_text())
    f16_rows = [r for r in results if r["quant_label"] == "F16"]
    q4_rows = [r for r in results if r["quant_label"] == "Q4_K_M"]

    # F16 should have spectacle_only=False
    assert all(r["spectacle_only"] is False for r in f16_rows)
    # Q4_K_M should have spectacle_only=True
    assert all(r["spectacle_only"] is True for r in q4_rows)
