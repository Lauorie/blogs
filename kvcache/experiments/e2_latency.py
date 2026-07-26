"""E2: Per-token generation latency, with vs without KV cache.

Three measurements on Qwen2.5-3B (bf16, greedy):
1. no_cache_forward: cost of one full forward over L tokens (what "generate the
   next token without a cache" costs), for a range of context lengths L.
   The lm_head is restricted to the last position so we measure transformer
   recomputation, not trivially avoidable logits waste.
2. with_cache_steps: per-step decode latency as the cache grows 128 -> 4096.
3. end_to_end: total wall clock to generate 1024 tokens from a 128-token
   prompt, both ways, and the resulting speedup.
"""

from __future__ import annotations

import time

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

PROMPT_LEN = 128
MAX_CTX = 4096
NO_CACHE_LENGTHS = [128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]
END_TO_END_NEW_TOKENS = 1024


@torch.inference_mode()
def bench_no_cache_forward(
    model: PreTrainedModel, full_ids: torch.Tensor, kw: str | None
) -> list[dict]:
    """Median time of a full forward pass over the first L tokens."""
    rows = []
    for length in NO_CACHE_LENGTHS:
        ids = full_ids[:, :length]
        kwargs = {kw: 1} if kw else {}

        def fwd() -> None:
            model(input_ids=ids, use_cache=False, **kwargs)

        reps = 3 if length >= 2048 else 5
        sec = timed_median(fwd, warmup=2, reps=reps)
        rows.append({"ctx": length, "seconds": sec})
        logger.info(f"no-cache forward ctx={length}: {sec * 1e3:.1f} ms")
    return rows


@torch.inference_mode()
def bench_with_cache_decode(
    model: PreTrainedModel, prompt_ids: torch.Tensor
) -> list[dict]:
    """Per-step decode latency as context grows from PROMPT_LEN to MAX_CTX."""
    cache = DynamicCache()
    out = model(input_ids=prompt_ids, past_key_values=cache, use_cache=True)
    nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    rows = []
    # Warm up decode kernels before timing.
    for _ in range(10):
        out = model(input_ids=nxt, past_key_values=cache, use_cache=True)
        nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    while cache.get_seq_length() < MAX_CTX:
        ctx = cache.get_seq_length() + 1  # sequence length attended over this step

        def step() -> None:
            nonlocal nxt
            o = model(input_ids=nxt, past_key_values=cache, use_cache=True)
            nxt = o.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        rows.append({"ctx": ctx, "seconds": cuda_timed(step)})
    logger.info(
        f"with-cache decode: {len(rows)} steps, "
        f"first {rows[0]['seconds'] * 1e3:.2f} ms, last {rows[-1]['seconds'] * 1e3:.2f} ms"
    )
    return rows


@torch.inference_mode()
def end_to_end(
    model: PreTrainedModel, prompt_ids: torch.Tensor, kw: str | None
) -> dict:
    """Total time to greedily generate END_TO_END_NEW_TOKENS, both ways."""
    kwargs = {kw: 1} if kw else {}

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    cache = DynamicCache()
    out = model(input_ids=prompt_ids, past_key_values=cache, use_cache=True)
    nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    cached_tokens = [int(nxt)]
    for _ in range(END_TO_END_NEW_TOKENS - 1):
        out = model(input_ids=nxt, past_key_values=cache, use_cache=True)
        nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        cached_tokens.append(int(nxt))
    torch.cuda.synchronize()
    with_cache_s = time.perf_counter() - t0
    logger.info(f"end-to-end WITH cache: {with_cache_s:.2f}s")

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    seq = prompt_ids
    plain_tokens = []
    for _ in range(END_TO_END_NEW_TOKENS):
        logits = model(input_ids=seq, use_cache=False, **kwargs).logits[:, -1, :]
        nxt = logits.argmax(dim=-1, keepdim=True)
        plain_tokens.append(int(nxt))
        seq = torch.cat([seq, nxt], dim=1)
    torch.cuda.synchronize()
    no_cache_s = time.perf_counter() - t0
    logger.info(f"end-to-end WITHOUT cache: {no_cache_s:.2f}s")

    return {
        "prompt_len": PROMPT_LEN,
        "num_new_tokens": END_TO_END_NEW_TOKENS,
        "with_cache_seconds": with_cache_s,
        "no_cache_seconds": no_cache_s,
        "speedup": no_cache_s / with_cache_s,
        "with_cache_tokens_per_s": END_TO_END_NEW_TOKENS / with_cache_s,
        "no_cache_tokens_per_s": END_TO_END_NEW_TOKENS / no_cache_s,
        "tokens_match": cached_tokens == plain_tokens,
    }


def main() -> None:
    set_seed(42)
    env = log_environment()
    model, tok = load_model(torch.bfloat16)
    kw = detect_logits_kw(model)

    prompt_ids = make_input_ids(tok, PROMPT_LEN)
    with_cache_steps = bench_with_cache_decode(model, prompt_ids)
    full_ids = make_input_ids(tok, MAX_CTX)
    no_cache_rows = bench_no_cache_forward(model, full_ids, kw)
    e2e = end_to_end(model, prompt_ids, kw)

    save_results(
        "e2_latency",
        {
            "env": env,
            "no_cache_forward": no_cache_rows,
            "with_cache_steps": with_cache_steps,
            "end_to_end": e2e,
        },
    )


if __name__ == "__main__":
    main()
