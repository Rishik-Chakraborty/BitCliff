import dataclasses
import importlib.metadata
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path

from .config import GenSettings
from .items import EvalItem


# Pre-0B ticket (freeze-plan §10): grading.py derives every record's
# `truncated` flag from `finish_reason == "length"`. Verify, on the actual
# llm instance about to generate, that a deliberately-truncated
# create_completion (the token path confirmatory longctx runs use) really
# reports "length" — if a llama-cpp-python version ever stops populating
# it, every truncation would silently grade as a clean stop.
TRUNCATION_PREFLIGHT_PROMPT = "Count upward forever: 1, 2, 3, 4, 5, 6, 7,"
TRUNCATION_PREFLIGHT_MAX_TOKENS = 8


def assert_truncation_finish_reason(llm) -> None:
    tokens = llm.tokenize(
        TRUNCATION_PREFLIGHT_PROMPT.encode("utf-8"), add_bos=True, special=False
    )
    out = llm.create_completion(
        prompt=list(tokens),
        max_tokens=TRUNCATION_PREFLIGHT_MAX_TOKENS,
        temperature=0.0,
        top_k=1,
        seed=42,
    )
    finish_reason = out["choices"][0].get("finish_reason")
    if finish_reason != "length":
        raise RuntimeError(
            f"truncation preflight failed: a create_completion capped at "
            f"{TRUNCATION_PREFLIGHT_MAX_TOKENS} tokens reported "
            f"finish_reason={finish_reason!r}, not 'length' — the truncated "
            f"flag (grading.py) cannot be trusted on this "
            f"llama-cpp-python build; aborting before any confirmatory "
            f"generation (pre-0B ticket, freeze-plan §10)"
        )


@dataclass(frozen=True)
class OutputRecord:
    item_id: str
    suite: str
    quant_label: str
    model_sha256: str
    prompt: str
    text: str
    finish_reason: str
    gen_settings: dict
    machine: str
    gen_wall_seconds: float | None = None
    """Task 5: wall-clock time of this item's single model call
    (create_completion / create_chat_completion), measured with
    time.perf_counter() around exactly that call -- computed AFTER the call
    returns, from a value nothing in the call itself can see, so it cannot
    alter what got generated. Defaults to None (Task 5 fix round 1) so
    `read_records` can still load pre-Task-5 output files (every record
    written before commit 1b6f418, e.g. runs-cloud/pipeline/runs/*/outputs/
    *.jsonl) -- `run_items` always sets a real float on every record it
    produces; None only ever appears on a legacy record loaded from disk."""
    gen_tokens: int | None = None
    """Task 5: the completion's generated-token count, read from the
    response's `usage.completion_tokens` (llama-cpp-python populates this
    on both create_completion and create_chat_completion) -- also read only
    after the call returns, never passed into it. Defaults to None for the
    same legacy-record-compatibility reason as `gen_wall_seconds` above."""


def make_llm(model_path: Path, gen: GenSettings):
    from llama_cpp import Llama

    return Llama(
        model_path=str(model_path),
        n_ctx=gen.n_ctx,
        seed=gen.seed,
        n_gpu_layers=-1,
        verbose=False,
    )


def run_items(
    llm,
    items: list[EvalItem],
    quant_label: str,
    model_sha256: str,
    gen: GenSettings,
    max_tokens_by_suite: dict[str, int] | None = None,
) -> list[OutputRecord]:
    max_tokens_by_suite = max_tokens_by_suite or {}
    try:
        llama_cpp_version = importlib.metadata.version("llama-cpp-python")
    except importlib.metadata.PackageNotFoundError:
        llama_cpp_version = "unknown"
    machine = (
        f"{platform.platform()} / {platform.machine()} "
        f"/ llama-cpp-python {llama_cpp_version}"
    )
    records = []
    for item in items:
        budget = max_tokens_by_suite.get(item.suite, gen.max_tokens)
        start = time.perf_counter()
        if item.prompt_tokens is not None:
            out = llm.create_completion(
                prompt=list(item.prompt_tokens),
                max_tokens=budget,
                temperature=gen.temperature,
                top_k=gen.top_k,
                seed=gen.seed,
            )
            choice = out["choices"][0]
            text = choice["text"]
        else:
            out = llm.create_chat_completion(
                messages=[{"role": "user", "content": item.prompt}],
                max_tokens=budget,
                temperature=gen.temperature,
                top_k=gen.top_k,
                seed=gen.seed,
            )
            choice = out["choices"][0]
            text = choice["message"]["content"]
        # Task 5 (runner timing field): both fields are read AFTER the call
        # returns, from values the call itself already produced -- neither
        # is threaded into any create_completion/create_chat_completion
        # argument above, so neither can alter the generated text.
        gen_wall_seconds = time.perf_counter() - start
        gen_tokens = out["usage"]["completion_tokens"]
        records.append(
            OutputRecord(
                item_id=item.id,
                suite=item.suite,
                quant_label=quant_label,
                model_sha256=model_sha256,
                prompt=item.prompt,
                text=text,
                finish_reason=choice.get("finish_reason") or "stop",
                gen_settings={**dataclasses.asdict(gen), "max_tokens_effective": budget},
                machine=machine,
                gen_wall_seconds=gen_wall_seconds,
                gen_tokens=gen_tokens,
            )
        )
    return records


def write_records(records: list[OutputRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(dataclasses.asdict(r)) + "\n")


def read_records(path: Path) -> list[OutputRecord]:
    return [OutputRecord(**json.loads(line)) for line in path.read_text().splitlines()]
