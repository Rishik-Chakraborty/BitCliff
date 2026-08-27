import dataclasses
import importlib.metadata
import json
import platform
from dataclasses import dataclass
from pathlib import Path

from .config import GenSettings
from .items import EvalItem


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
