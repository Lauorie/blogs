"""E3: Time-to-first-token (prefill cost) vs prompt length.

For each prompt length L, time the prefill forward pass (building the KV cache
for all L tokens at once), then time a few single-token decode steps on top of
that cache for contrast. TTFT here is pure model compute — no tokenization,
scheduling or network overhead.
"""

from __future__ import annotations

import statistics

import torch
from transformers import DynamicCache, PreTrainedModel

from common import (
    cuda_timed,
    detect_logits_kw,
    load_model,
    log_environment,
    logger,
    make_input_ids,
    save_results,
    set_seed,
    timed_median,
)

PROMPT_LENGTHS = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
DECODE_PROBE_STEPS = 5


@torch.inference_mode()
def bench_prefill(model: PreTrainedModel, ids: torch.Tensor, kw: str | None) -> float:
    """Median prefill time for one prompt (fresh cache each run)."""
    kwargs = {kw: 1} if kw else {}

    def prefill() -> None:
        model(input_ids=ids, past_key_values=DynamicCache(), use_cache=True, **kwargs)

    length = ids.shape[1]
    warmup = 1 if length >= 8192 else 2
    reps = 3 if length >= 4096 else 5
    return timed_median(prefill, warmup=warmup, reps=reps)


@torch.inference_mode()
def bench_decode_after(model: PreTrainedModel, ids: torch.Tensor, kw: str | None) -> float:
    """Median single-token decode latency right after prefilling ids."""
    kwargs = {kw: 1} if kw else {}
    cache = DynamicCache()
    out = model(input_ids=ids, past_key_values=cache, use_cache=True, **kwargs)
    nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    times = []
    for _ in range(DECODE_PROBE_STEPS):

        def step() -> None:
            nonlocal nxt
            o = model(input_ids=nxt, past_key_values=cache, use_cache=True)
            nxt = o.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        times.append(cuda_timed(step))
    return statistics.median(times)


def main() -> None:
    set_seed(42)
    env = log_environment()
    model, tok = load_model(torch.bfloat16)
    kw = detect_logits_kw(model)

    rows = []
    for length in PROMPT_LENGTHS:
        ids = make_input_ids(tok, length)
        prefill_s = bench_prefill(model, ids, kw)
        decode_s = bench_decode_after(model, ids, kw)
        rows.append(
            {"prompt_len": length, "prefill_seconds": prefill_s, "decode_step_seconds": decode_s}
        )
        logger.info(
            f"L={length}: prefill {prefill_s * 1e3:.1f} ms, "
            f"decode step {decode_s * 1e3:.2f} ms"
        )

    save_results("e3_ttft", {"env": env, "rows": rows})


if __name__ == "__main__":
    main()
