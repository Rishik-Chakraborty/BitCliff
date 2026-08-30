from dataclasses import asdict

from bitcliff_pipeline.config import GenSettings
from bitcliff_pipeline.generate import (
    OutputRecord,
    read_records,
    run_items,
    write_records,
)
from bitcliff_pipeline.items import EvalItem

GEN = GenSettings(seed=42, temperature=0.0, top_k=1, max_tokens=640, n_ctx=4096)

ITEMS = [
    EvalItem("arithmetic-1-000", "arithmetic", "What is 12 * 34?", ("408",)),
    EvalItem("spec-001", "spectacle", "Write a haiku.", None),
]

TOKEN_ITEMS = [
    EvalItem(
        "longctx_retrieval-1-000",
        "longctx_retrieval",
        "What is Alice's code?",
        ("1234",),
        prompt_tokens=(1, 2, 3, 4, 5),
    ),
    EvalItem("spec-001", "spectacle", "Write a haiku.", None),
]


class FakeLlm:
    def __init__(self):
        self.calls = []
        self.completion_calls = []

    def create_chat_completion(self, messages, max_tokens, temperature, top_k, seed):
        self.calls.append(
            {"messages": messages, "max_tokens": max_tokens,
             "temperature": temperature, "top_k": top_k, "seed": seed}
        )
        return {
            "choices": [
                {"message": {"content": f"echo: {messages[0]['content'][:10]}"},
                 "finish_reason": "stop"}
            ]
        }

    def create_completion(self, prompt, max_tokens, temperature, top_k, seed):
        self.completion_calls.append(
            {"prompt": prompt, "max_tokens": max_tokens,
             "temperature": temperature, "top_k": top_k, "seed": seed}
        )
        return {
            "choices": [
                {"text": f"tok-echo: {prompt[:3]}", "finish_reason": "stop"}
            ]
        }


def test_run_items_builds_records_with_settings():
    llm = FakeLlm()
    records = run_items(llm, ITEMS, quant_label="Q4_K_M", model_sha256="abc123", gen=GEN)
    assert len(records) == 2
    r = records[0]
    assert r.item_id == "arithmetic-1-000"
    assert r.suite == "arithmetic"
    assert r.quant_label == "Q4_K_M"
    assert r.model_sha256 == "abc123"
    assert r.finish_reason == "stop"
    assert r.gen_settings == {**asdict(GEN), "max_tokens_effective": GEN.max_tokens}
    assert r.machine  # non-empty platform string
    assert "llama-cpp-python" in r.machine
    # deterministic settings actually passed through to the model
    assert llm.calls[0]["temperature"] == 0.0
    assert llm.calls[0]["top_k"] == 1
    assert llm.calls[0]["seed"] == 42


def test_run_items_routes_token_id_items_to_create_completion():
    llm = FakeLlm()
    records = run_items(llm, TOKEN_ITEMS, quant_label="Q4_K_M", model_sha256="abc123", gen=GEN)
    assert len(records) == 2

    # first item has prompt_tokens -> create_completion, exact token list passed
    assert len(llm.completion_calls) == 1
    assert llm.completion_calls[0]["prompt"] == [1, 2, 3, 4, 5]
    assert llm.completion_calls[0]["max_tokens"] == GEN.max_tokens
    assert llm.completion_calls[0]["temperature"] == GEN.temperature
    assert llm.completion_calls[0]["top_k"] == GEN.top_k
    assert llm.completion_calls[0]["seed"] == GEN.seed
    r0 = records[0]
    assert r0.item_id == "longctx_retrieval-1-000"
    assert r0.text == "tok-echo: [1, 2, 3]"
    assert r0.finish_reason == "stop"
    assert r0.gen_settings["max_tokens_effective"] == GEN.max_tokens

    # second item has no prompt_tokens -> stays on the chat path
    assert len(llm.calls) == 1
    r1 = records[1]
    assert r1.item_id == "spec-001"
    assert r1.text.startswith("echo:")
    assert r1.gen_settings["max_tokens_effective"] == GEN.max_tokens


def test_run_items_honors_per_suite_max_tokens_override():
    llm = FakeLlm()
    records = run_items(
        llm, TOKEN_ITEMS, quant_label="Q4_K_M", model_sha256="abc123", gen=GEN,
        max_tokens_by_suite={"longctx_retrieval": 32},
    )
    assert llm.completion_calls[0]["max_tokens"] == 32
    assert records[0].gen_settings["max_tokens_effective"] == 32
    # spectacle isn't in the override map -> falls back to gen.max_tokens
    assert llm.calls[0]["max_tokens"] == GEN.max_tokens
    assert records[1].gen_settings["max_tokens_effective"] == GEN.max_tokens


def test_run_items_max_tokens_by_suite_none_is_treated_as_empty():
    llm = FakeLlm()
    records = run_items(
        llm, TOKEN_ITEMS, quant_label="Q4_K_M", model_sha256="abc123", gen=GEN,
        max_tokens_by_suite=None,
    )
    assert records[0].gen_settings["max_tokens_effective"] == GEN.max_tokens


def test_jsonl_roundtrip(tmp_path):
    llm = FakeLlm()
    records = run_items(llm, ITEMS, "Q8_0", "def456", GEN)
    path = tmp_path / "Q8_0.jsonl"
    write_records(records, path)
    loaded = read_records(path)
    assert loaded == records
    assert isinstance(loaded[0], OutputRecord)


import pytest

from bitcliff_pipeline.generate import (
    TRUNCATION_PREFLIGHT_MAX_TOKENS,
    assert_truncation_finish_reason,
)


class PreflightFakeLlm:
    def __init__(self, finish_reason):
        self._finish_reason = finish_reason
        self.completion_kwargs = None

    def tokenize(self, text, add_bos=True, special=False):
        return list(range(len(text.split())))

    def create_completion(self, prompt, **kwargs):
        self.completion_kwargs = {"prompt": prompt, **kwargs}
        return {"choices": [{"text": "1, 2, 3", "finish_reason": self._finish_reason}]}


def test_truncation_preflight_passes_on_length():
    llm = PreflightFakeLlm("length")
    assert_truncation_finish_reason(llm)  # must not raise
    # the probe really was a deliberately-truncated token-path completion
    assert llm.completion_kwargs["max_tokens"] == TRUNCATION_PREFLIGHT_MAX_TOKENS
    assert isinstance(llm.completion_kwargs["prompt"], list)
    # deterministic decode settings, PREREG §6
    assert llm.completion_kwargs["temperature"] == 0.0
    assert llm.completion_kwargs["top_k"] == 1


@pytest.mark.parametrize("bad", ["stop", None])
def test_truncation_preflight_raises_on_wrong_finish_reason(bad):
    with pytest.raises(RuntimeError, match="finish_reason"):
        assert_truncation_finish_reason(PreflightFakeLlm(bad))
