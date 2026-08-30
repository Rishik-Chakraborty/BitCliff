"""Pre-0B ticket (freeze-plan §10): verify the REAL llama-cpp-python build
populates finish_reason='length' on truncation via the token path
(create_completion with a token-id prompt) — grading.py's `truncated` flag
depends on it. Runs against the smallest local GGUF; skipped when the model
file is absent (e.g. CI without model assets). The same
assert_truncation_finish_reason preflight runs on the 0B cloud instance
before every confirmatory generation, so this check re-executes on the
exact serving build too.
"""

from pathlib import Path

import pytest

from bitcliff_pipeline.generate import (
    TRUNCATION_PREFLIGHT_MAX_TOKENS,
    assert_truncation_finish_reason,
)

MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "Qwen2.5-1.5B-Instruct-IQ1_S-bitcliff-inhouse.gguf"
)

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason=f"local model not present: {MODEL_PATH}",
)


@pytest.fixture(scope="module")
def llm():
    from llama_cpp import Llama

    return Llama(model_path=str(MODEL_PATH), n_ctx=512, seed=42, verbose=False)


def test_real_llama_cpp_reports_length_on_truncation(llm):
    assert_truncation_finish_reason(llm)  # raises if finish_reason != "length"


def test_real_llama_cpp_token_path_truncation_direct(llm):
    """Belt-and-braces: same assertion without going through the preflight
    helper, so a bug in the helper cannot mask a llama-cpp regression."""
    tokens = llm.tokenize(b"Count upward forever: 1, 2, 3, 4, 5, 6, 7,", add_bos=True)
    out = llm.create_completion(
        prompt=list(tokens),
        max_tokens=TRUNCATION_PREFLIGHT_MAX_TOKENS,
        temperature=0.0,
        top_k=1,
        seed=42,
    )
    assert out["choices"][0]["finish_reason"] == "length"
