# gsm8k_first60.jsonl provenance

The 60 records in `gsm8k_first60.jsonl` are the first 60 rows of the
`openai/gsm8k` ("main" config, "test" split) dataset, vendored here as a
fallback so `tests/test_twins.py` does not hard-depend on the local
HuggingFace `datasets` cache being warm.

GSM8K (`openai/gsm8k`) is released by OpenAI under the **MIT License**
(see the dataset repository's `LICENSE` file:
https://github.com/openai/grade-school-math). Vendoring a 60-row excerpt
of MIT-licensed test data for offline test execution is permitted under
that license.

This file is used only as a fallback when
`datasets.load_dataset("openai/gsm8k", "main", split="test")` cannot load
offline; the primary path in `tests/test_twins.py` loads from the
`datasets` cache directly, matching the pilot's usage
(`suites/arithmetic.py::load_gsm8k_items`).
