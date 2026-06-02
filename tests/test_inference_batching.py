"""Correctness test for batched decode vs. per-request decode.

Batched decode must produce identical greedy tokens to decoding each request
on its own — that equivalence is the whole point of left-padded batching.

This test loads a real (small) model, so it is skipped by default. Enable it
with ``LITESERVE_MODEL_TESTS=1 pytest`` (downloads TinyLlama on first run).
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("LITESERVE_MODEL_TESTS") != "1",
    reason="set LITESERVE_MODEL_TESTS=1 to run model-loading tests",
)

MODEL = os.environ.get("LITESERVE_TEST_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

PROMPTS = [
    "The capital of France is",
    "Once upon a time, in a small village near the",
    "Photosynthesis is the process by which",
    "A neural network learns by adjusting its",
]


@pytest.fixture(scope="module")
def loaded_model():
    from liteserve.models.loader import ModelLoader

    return ModelLoader().load_model(name="test", model_path=MODEL, is_default=True)


def _greedy_requests(engine, tokens):
    from liteserve.engine.types import Request

    reqs = []
    for p in PROMPTS:
        r = Request(prompt=p, max_new_tokens=tokens, temperature=0.0)
        r.prompt_tokens = engine.tokenize(p)
        reqs.append(r)
    return reqs


def test_batched_decode_matches_sequential(loaded_model):
    from liteserve.engine.inference import InferenceEngine

    tokens = 16

    # Per-request reference.
    engine = InferenceEngine(model=loaded_model)
    seq_reqs = _greedy_requests(engine, tokens)
    for r in seq_reqs:
        engine.prefill(r)
        while not r.is_finished:
            engine.decode_step(r)
    reference = [list(r.generated_tokens) for r in seq_reqs]

    # Batched decode.
    engine = InferenceEngine(model=loaded_model)
    bat_reqs = _greedy_requests(engine, tokens)
    for r in bat_reqs:
        engine.prefill(r)
    while any(not r.is_finished for r in bat_reqs):
        engine.decode_batch([r for r in bat_reqs if not r.is_finished])
    batched = [list(r.generated_tokens) for r in bat_reqs]

    assert batched == reference, "batched greedy decode must match per-request decode"
    # Every request's KV must be freed on completion.
    assert engine.get_stats()["active_kv_caches"] == 0


def test_batched_decode_handles_single_request(loaded_model):
    from liteserve.engine.inference import InferenceEngine

    engine = InferenceEngine(model=loaded_model)
    reqs = _greedy_requests(engine, 8)[:1]
    engine.prefill(reqs[0])
    while not reqs[0].is_finished:
        engine.decode_batch(reqs)
    assert reqs[0].num_generated >= 1
