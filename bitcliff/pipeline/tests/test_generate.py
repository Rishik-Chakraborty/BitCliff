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
    EvalItem("retrieval-1-000", "retrieval", "What is Alice's code?", ("1234", "5678")),
    EvalItem("spec-001", "spectacle", "Write a haiku.", None),
]


class FakeLlm:
    def __init__(self):
        self.calls = []

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


def test_run_items_builds_records_with_settings():
    llm = FakeLlm()
    records = run_items(llm, ITEMS, quant_label="Q4_K_M", model_sha256="abc123", gen=GEN)
    assert len(records) == 2
    r = records[0]
    assert r.item_id == "retrieval-1-000"
    assert r.suite == "retrieval"
    assert r.quant_label == "Q4_K_M"
    assert r.model_sha256 == "abc123"
    assert r.finish_reason == "stop"
    assert r.gen_settings == asdict(GEN)
    assert r.machine  # non-empty platform string
    assert "llama-cpp-python" in r.machine
    # deterministic settings actually passed through to the model
    assert llm.calls[0]["temperature"] == 0.0
    assert llm.calls[0]["top_k"] == 1
    assert llm.calls[0]["seed"] == 42


def test_jsonl_roundtrip(tmp_path):
    llm = FakeLlm()
    records = run_items(llm, ITEMS, "Q8_0", "def456", GEN)
    path = tmp_path / "Q8_0.jsonl"
    write_records(records, path)
    loaded = read_records(path)
    assert loaded == records
    assert isinstance(loaded[0], OutputRecord)
