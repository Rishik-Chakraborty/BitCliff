"""Tests for scripts/nll_driver.py (teacher-forced answer-span NLL driver,
replacing the uncommitted 0B box driver -- OPEN_QUESTIONS §16).

Loaded by file path (same pattern as tests/test_repro_gate.py): the script
has no package `__init__.py`.

No network, no real model: a digit-splitting stub HF tokenizer stands in
for `models/hf/...` (injected via `__main__._load_hf_tokenizer`, the same
seam tests/test_cli.py uses), the real `__main__.build_items` + vendored
multivalue generator build the longctx items from a tiny synthetic corpus,
and a fake llama-cpp object (`reset()/eval(tokens)/eval_logits`, the shape
`nll_scorer.make_llama_logits_provider` reads) serves deterministic,
per-model-distinct logits. Expected NLL values are computed by calling
`nll_scorer.score_records` directly on independently built NLLItems -- the
driver must reproduce the library's numbers, never a re-derivation.
"""

import dataclasses
import hashlib
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import pytest

import bitcliff_pipeline.__main__ as main_mod
from bitcliff_pipeline import nll_scorer
from bitcliff_pipeline.config import GenSettings, LadderConfig, NLLConfig, QuantFile
from bitcliff_pipeline.hashing import item_set_sha256, sha256_file, write_manifest

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = PIPELINE_ROOT / "scripts" / "nll_driver.py"
COMMITTED_NLL = (
    PIPELINE_ROOT / "runs-cloud" / "pipeline" / "runs" / "0b-qwen-7b-ladder" / "nll" / "F16.jsonl"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("nll_driver", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["nll_driver"] = module
    spec.loader.exec_module(module)
    return module


nd = _load_module()


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class DigitStubTokenizer:
    """HF-shaped stub (the surface `longctx_retrieval.build_items_with_
    answer_spec` needs, plus `__len__`) that splits every ASCII digit into
    its own token -- like Qwen2.5 -- so passcode answers yield single-digit
    tokens and `digit_token_ids_from_decode` finds them."""

    _TOKEN_RE = re.compile(r"\d|[^\d ]+")

    def __init__(self, split_digits: bool = True, name: str = "digit-stub"):
        self.name_or_path = name
        self._split_digits = split_digits
        self._w2i: dict[str, int] = {}
        self._i2w: dict[int, str] = {}

    def _id(self, w: str) -> int:
        if w not in self._w2i:
            i = len(self._w2i)
            self._w2i[w] = i
            self._i2w[i] = w
        return self._w2i[w]

    def _pieces(self, text: str) -> list[str]:
        if self._split_digits:
            return self._TOKEN_RE.findall(text)
        return [p for p in text.split(" ") if p]

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        return {"input_ids": [self._id(p) for p in self._pieces(text)]}

    def decode(self, ids) -> str:
        return " ".join(self._i2w[i] for i in ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True) -> str:
        return f"<|user|> {messages[0]['content']} <|assistant|>"

    def __len__(self) -> int:
        return len(self._w2i)


class FakeLogitsLlm:
    """llama-cpp-shaped fake: `reset()`, `eval(tokens)`, `eval_logits`.
    Row at position p is a deterministic function of (model salt, token at
    p, p) -- distinct models give distinct NLLs, identical models identical
    ones."""

    def __init__(self, vocab_size: int, salt: int):
        self.vocab_size = vocab_size
        self.salt = salt
        self._tokens: list[int] = []

    def reset(self):
        self._tokens = []

    def eval(self, tokens):
        self._tokens = list(tokens)

    @property
    def eval_logits(self):
        v = self.vocab_size
        return [
            [((tok * 31 + j * 7 + p * 3 + self.salt) % 17) / 3.0 for j in range(v)]
            for p, tok in enumerate(self._tokens)
        ]


CORPUS_TEXT = " ".join(f"corpusword{i}" for i in range(400))
CORPUS_SHA256 = hashlib.sha256(CORPUS_TEXT.encode("utf-8")).hexdigest()


def _config(tmp_path: Path, *, nll=("longctx_retrieval",), exploratory=True) -> LadderConfig:
    (tmp_path / "corpus.txt").write_text(CORPUS_TEXT)
    models = tmp_path / "models"
    models.mkdir(exist_ok=True)
    (models / "f16.gguf").write_bytes(b"fake f16 weights")
    (models / "m-Q4_K_M.gguf").write_bytes(b"fake q4 weights")
    (models / "m-Q3_K_M.gguf").write_bytes(b"fake q3 weights")
    return LadderConfig(
        model_id="test-model",
        hf_repo="fake/repo",
        f16_path=models / "f16.gguf",
        quants=(
            QuantFile("Q4_K_M", "m-Q4_K_M.gguf", "bartowski", True),
            QuantFile("Q3_K_M", "m-Q3_K_M.gguf", "bartowski", True),
        ),
        generation=GenSettings(42, 0.0, 1, 1024, 16384),
        suites={
            "longctx_retrieval": {
                "variant": "multivalue2",
                "target_tokens": 64,
                "seed": 7,
                "n_items": 3,
                "corpus_path": "corpus.txt",
                "corpus_sha256": CORPUS_SHA256,
                "tokenizer_path": "tok",
                "max_tokens": 32,
            },
            "arithmetic": {"n_items": 4, "seed": 1},
        },
        exploratory=exploratory,
        nll=NLLConfig(suites=tuple(nll)) if nll is not None else None,
    )


@pytest.fixture
def tok(monkeypatch):
    t = DigitStubTokenizer()
    monkeypatch.setattr(main_mod, "_load_hf_tokenizer", lambda path: t)
    return t


@pytest.fixture(autouse=True)
def no_arithmetic(monkeypatch):
    """The driver must build ONLY the named suites; building arithmetic
    would hit the network (GSM8K) -- make any such call fail loudly."""
    from bitcliff_pipeline.suites import arithmetic

    def boom(*a, **k):
        raise AssertionError("arithmetic suite built but not named for NLL")

    monkeypatch.setattr(arithmetic, "load_gsm8k_items", boom)


SALTS = {"f16.gguf": 0, "m-Q4_K_M.gguf": 5, "m-Q3_K_M.gguf": 11}


def _factory(tok, calls=None):
    def factory(model_path, n_ctx, seed):
        if calls is not None:
            calls.append((Path(model_path).name, n_ctx, seed))
        return FakeLogitsLlm(len(tok), SALTS[Path(model_path).name])

    return factory


def _run(tmp_path, tok, cfg=None, calls=None, **kw):
    cfg = cfg or _config(tmp_path)
    nd.run_nll(
        cfg,
        run_id="r1",
        runs_dir=tmp_path / "runs",
        models_dir=tmp_path / "models",
        base_dir=tmp_path,
        llm_factory=_factory(tok, calls),
        **kw,
    )
    return tmp_path / "runs" / "r1" / "nll"


def _expected_items(tmp_path, tok, cfg):
    trimmed = dataclasses.replace(
        cfg, suites={"longctx_retrieval": cfg.suites["longctx_retrieval"]}
    )
    specs: list = []
    items = main_mod.build_items(trimmed, tmp_path, longctx_tokenizer=tok, longctx_answer_specs=specs)
    return items, [
        nll_scorer.NLLItem(id=s.item_id, suite=i.suite, gen_prompt_ids=i.prompt_tokens, answer_ids=s.answer_ids)
        for i, s in zip(items, specs)
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_output_schema_equals_committed_records(tmp_path, tok):
    out = _run(tmp_path, tok)
    committed_keys = list(json.loads(COMMITTED_NLL.read_text().splitlines()[0]).keys())
    for label in ("F16", "Q4_K_M", "Q3_K_M"):
        rows = [json.loads(line) for line in (out / f"{label}.jsonl").read_text().splitlines()]
        assert rows
        for row in rows:
            assert list(row.keys()) == committed_keys


def test_nll_values_equal_scorer_for_same_fake_logits(tmp_path, tok):
    cfg = _config(tmp_path)
    out = _run(tmp_path, tok, cfg=cfg)
    _, nll_items = _expected_items(tmp_path, tok, cfg)
    digits = nll_scorer.digit_token_ids_from_decode(tok.decode, len(tok))
    for label, fname in (("F16", "f16.gguf"), ("Q4_K_M", "m-Q4_K_M.gguf")):
        expected = nll_scorer.score_records(
            FakeLogitsLlm(len(tok), SALTS[fname]), nll_items, label,
            sha256_file(tmp_path / "models" / fname), digits,
        )
        got = nll_scorer.read_records(out / f"{label}.jsonl")
        assert got == expected
    # sanity: F16 and Q4 fakes really differ, so equality above is not vacuous
    f16 = nll_scorer.read_records(out / "F16.jsonl")
    q4 = nll_scorer.read_records(out / "Q4_K_M.jsonl")
    assert [r.span_mean_nll for r in f16] != [r.span_mean_nll for r in q4]


def test_digits_only_populated_when_answer_has_digit_tokens(tmp_path, tok):
    out = _run(tmp_path, tok)
    digits = nll_scorer.digit_token_ids_from_decode(tok.decode, len(tok))
    assert digits
    _, nll_items = _expected_items(tmp_path, tok, _config(tmp_path))
    by_id = {i.id: i for i in nll_items}
    for label in ("F16", "Q4_K_M", "Q3_K_M"):
        for r in nll_scorer.read_records(out / f"{label}.jsonl"):
            has_digit = any(t in digits for t in by_id[r.item_id].answer_ids)
            assert has_digit  # passcode answers always carry digits here
            assert r.digits_only_mean_nll is not None
            assert math.isfinite(r.digits_only_mean_nll)


def test_empty_digit_set_raises(tmp_path, monkeypatch):
    t = DigitStubTokenizer(split_digits=False)  # 4-digit passcodes stay whole
    monkeypatch.setattr(main_mod, "_load_hf_tokenizer", lambda path: t)
    with pytest.raises(RuntimeError, match="digit token"):
        _run(tmp_path, t)
    assert not (tmp_path / "runs" / "r1" / "nll" / "F16.jsonl").exists()


def test_settings_n_ctx_seed_and_logs(tmp_path, tok, capsys):
    calls: list = []
    cfg = _config(tmp_path)
    _run(tmp_path, tok, cfg=cfg, calls=calls)
    _, nll_items = _expected_items(tmp_path, tok, cfg)
    max_seq = max(len(i.gen_prompt_ids) + len(i.answer_ids) for i in nll_items)
    n_ctx = math.ceil(max_seq / 256) * 256
    assert calls == [("f16.gguf", n_ctx, 42), ("m-Q4_K_M.gguf", n_ctx, 42), ("m-Q3_K_M.gguf", n_ctx, 42)]
    log = capsys.readouterr().out
    assert f"nll items: {len(nll_items)}" in log
    assert f"n_ctx sized to {n_ctx} (max seq {max_seq})" in log
    assert re.search(r"^digit token ids: \d+$", log, re.M)
    assert "NLL_START F16 f16.gguf" in log
    assert f"NLL_DONE Q4_K_M n={len(nll_items)}" in log
    assert log.rstrip().endswith("NLL_PASS_COMPLETE")


def test_labels_subset_and_order(tmp_path, tok):
    calls: list = []
    out = _run(tmp_path, tok, calls=calls, labels=["Q3_K_M", "Q4_K_M"])
    assert [c[0] for c in calls] == ["m-Q4_K_M.gguf", "m-Q3_K_M.gguf"]  # config order
    assert sorted(p.name for p in out.glob("*.jsonl")) == ["Q3_K_M.jsonl", "Q4_K_M.jsonl"]
    with pytest.raises(ValueError, match="unknown label"):
        _run(tmp_path, tok, labels=["Q9_X"])


def test_resume_skips_existing_label(tmp_path, tok, capsys):
    out = _run(tmp_path, tok, labels=["F16"])
    before = (out / "F16.jsonl").read_bytes()
    manifest_before = json.loads((out / "manifest.json").read_text())
    calls: list = []
    _run(tmp_path, tok, calls=calls)
    assert [c[0] for c in calls] == ["m-Q4_K_M.gguf", "m-Q3_K_M.gguf"]
    assert "skip F16:" in capsys.readouterr().out
    assert (out / "F16.jsonl").read_bytes() == before
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["labels"]["F16"] == manifest_before["labels"]["F16"]
    assert set(manifest["labels"]) == {"F16", "Q4_K_M", "Q3_K_M"}


def test_resume_refuses_settings_drift(tmp_path, tok):
    out = _run(tmp_path, tok, labels=["F16"])
    m = json.loads((out / "manifest.json").read_text())
    m["seed"] = 999
    (out / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(RuntimeError, match="manifest"):
        _run(tmp_path, tok)


def test_manifest_contents(tmp_path, tok):
    cfg = _config(tmp_path)
    out = _run(tmp_path, tok, cfg=cfg)
    m = json.loads((out / "manifest.json").read_text())
    items, nll_items = _expected_items(tmp_path, tok, cfg)
    digits = sorted(nll_scorer.digit_token_ids_from_decode(tok.decode, len(tok)))
    max_seq = max(len(i.gen_prompt_ids) + len(i.answer_ids) for i in nll_items)
    assert m["suites"] == ["longctx_retrieval"]
    assert m["n_items"] == {"longctx_retrieval": 3}
    assert m["item_set_sha256"] == {"longctx_retrieval": item_set_sha256(items)}
    assert m["n_ctx"] == {
        "rule": "ceil(max_seq/256)*256",
        "max_seq": max_seq,
        "value": math.ceil(max_seq / 256) * 256,
    }
    assert m["logits_all"] is True
    assert m["seed"] == 42
    assert m["n_gpu_layers"] == -1
    assert m["digit_token_ids"]["count"] == len(digits)
    assert m["digit_token_ids"]["ids"] == digits
    assert m["machine"] == nll_scorer._machine_fingerprint()
    assert m["driver_sha256"] == sha256_file(SCRIPT_PATH)
    assert m["model_id"] == "test-model"
    assert m["item_set_gate"] == "skipped (exploratory config)"
    for label, fname in (("F16", "f16.gguf"), ("Q4_K_M", "m-Q4_K_M.gguf"), ("Q3_K_M", "m-Q3_K_M.gguf")):
        assert m["labels"][label] == {
            "filename": fname,
            "model_sha256": sha256_file(tmp_path / "models" / fname),
            "sha256_source": "hashed",
        }
    # every record carries the manifest's model sha for its label
    for label in m["labels"]:
        for r in nll_scorer.read_records(out / f"{label}.jsonl"):
            assert r.model_sha256 == m["labels"][label]["model_sha256"]
            assert r.machine == m["machine"]


def test_model_sha_from_run_manifest_and_mismatch_refused(tmp_path, tok):
    cfg = _config(tmp_path)
    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True)
    f16_sha = sha256_file(tmp_path / "models" / "f16.gguf")
    write_manifest({"F16": {"filename": "f16.gguf", "sha256": f16_sha}}, run_dir / "manifest.json")
    out = _run(tmp_path, tok, cfg=cfg, labels=["F16"])
    m = json.loads((out / "manifest.json").read_text())
    assert m["labels"]["F16"]["sha256_source"] == "run manifest.json (verified)"
    (out / "F16.jsonl").unlink()
    (out / "manifest.json").unlink()
    write_manifest({"F16": {"filename": "f16.gguf", "sha256": "0" * 64}}, run_dir / "manifest.json")
    with pytest.raises(RuntimeError, match="sha256"):
        _run(tmp_path, tok, cfg=cfg, labels=["F16"])


def test_determinism_two_runs_byte_identical(tmp_path, tok):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    out_a = _run(a, tok, cfg=_config(a))
    out_b = _run(b, tok, cfg=_config(b))
    names = sorted(p.name for p in out_a.iterdir())
    assert names == sorted(p.name for p in out_b.iterdir())
    for n in names:
        assert (out_a / n).read_bytes() == (out_b / n).read_bytes(), n


def test_missing_nll_block_and_no_suites_errors(tmp_path, tok):
    cfg = _config(tmp_path, nll=None)
    with pytest.raises(ValueError, match="nll"):
        _run(tmp_path, tok, cfg=cfg)
    # --suites override works without an nll block
    out = _run(tmp_path, tok, cfg=cfg, suites=["longctx_retrieval"])
    assert (out / "F16.jsonl").exists()


def test_suites_override_validation(tmp_path, tok):
    with pytest.raises(ValueError, match="not in the config"):
        _run(tmp_path, tok, suites=["longctx_retrieval_2a"])
    with pytest.raises(ValueError, match="answer-span"):
        _run(tmp_path, tok, suites=["arithmetic"])


def test_item_set_gate_runs_for_non_exploratory_config(tmp_path, tok):
    cfg = _config(tmp_path, exploratory=False)
    with pytest.raises(RuntimeError, match="item-set hash gate"):
        _run(tmp_path, tok, cfg=cfg)
    assert not (tmp_path / "runs" / "r1" / "nll" / "F16.jsonl").exists()


def test_cli_main_parses_and_dispatches(tmp_path, monkeypatch):
    seen = {}

    def fake_run(config, **kw):
        seen["config"] = config
        seen.update(kw)

    monkeypatch.setattr(nd, "run_nll", fake_run)
    cfg_dir = tmp_path / "configs" / "0b"
    cfg_dir.mkdir(parents=True)
    cfg_path = cfg_dir / "x.yaml"
    cfg_path.write_text(
        "model_id: m\nhf_repo: r\nf16_path: models/f16.gguf\nquants: []\n"
        "generation: {seed: 42, temperature: 0.0, top_k: 1, max_tokens: 8, n_ctx: 64}\n"
        "suites: {spectacle: {path: p.yaml}}\n"
    )
    rc = nd.main([
        str(cfg_path), "--run-id", "g1", "--runs-dir", "rr", "--models-dir", "mm",
        "--suites", "longctx_retrieval", "--labels", "Q4_K_M",
    ])
    assert rc == 0
    assert seen["run_id"] == "g1"
    assert seen["runs_dir"] == Path("rr")
    assert seen["models_dir"] == Path("mm")
    assert seen["suites"] == ["longctx_retrieval"]
    assert seen["labels"] == ["Q4_K_M"]
    assert seen["base_dir"] == tmp_path.resolve()
    assert seen["config_sha256"] == sha256_file(cfg_path)
