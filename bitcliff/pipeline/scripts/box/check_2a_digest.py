# Ran on the 0B-prime box (i-04147e175eba11337) on 2026-10-14; committed for the
# record (OPEN_QUESTIONS §18). Body unchanged from the copy shipped to the box,
# sha256 96f6ee1f383937c118bc99c5e3f5cc44fae2aed7a020d3a1fbae5efcd92fb285.
"""PREREG §3.1 2a pre-generation check (RUN_0B_PRIME §2): the 2a corpus hashes to
the registered CORPUS_2A_SHA256 and the vendored generator with the real
Qwen2.5-1.5B tokenizer reproduces the registered first-20-item digest."""
import hashlib, sys
from transformers import AutoTokenizer
from bitcliff_pipeline import registered
from bitcliff_pipeline.suites import longctx_retrieval as lr
from bitcliff_pipeline.vendor import generate_multivalue2 as mv2
tok = AutoTokenizer.from_pretrained("models/hf/Qwen2.5-1.5B-Instruct")
text = lr.load_2a_corpus_text()
c = hashlib.sha256(text.encode("utf-8")).hexdigest()
print("CORPUS_2A", c, "match", c == registered.CORPUS_2A_SHA256)
mv2._STREAM_CACHE[tok.name_or_path] = tok(text, add_special_tokens=False)["input_ids"]
d = mv2.items_digest(mv2.build_items(tok, "multivalue2", 20, 4096, 2024))
ok = d == registered.LONGCTX_2A_FIRST20_DIGEST_SHA256
print("FIRST20_DIGEST", d, "match", ok)
sys.exit(0 if ok and c == registered.CORPUS_2A_SHA256 else 1)
