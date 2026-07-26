"""E4: KV cache GPU memory — measured vs formula.

Formula per token:
    bytes = 2 (K and V) x num_layers x num_kv_heads x head_dim x dtype_bytes

For each sequence length L we prefill a fresh cache, drop every other tensor,
and read the increase in torch.cuda.memory_allocated(). Also reports the
hypothetical no-GQA (MHA, 16 kv heads) size for comparison.
"""

from __future__ import annotations

import gc

import torch
from transformers import DynamicCache, PreTrainedModel

from common import (
    config_summary,
    detect_logits_kw,
    load_model,
    log_environment,
    logger,
    make_input_ids,
    save_results,
    set_seed,
)

SEQ_LENGTHS = [1024, 2048, 4096, 8192, 16384, 32768]
DTYPE_BYTES = 2  # bf16


@torch.inference_mode()
def measure_cache_bytes(model: PreTrainedModel, ids: torch.Tensor, kw: str | None) -> int:
    """Allocated-memory delta attributable to the KV cache alone."""
    kwargs = {kw: 1} if kw else {}
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    cache = DynamicCache()
    out = model(input_ids=ids, past_key_values=cache, use_cache=True, **kwargs)
    del out
    gc.collect()
    torch.cuda.synchronize()
    measured = torch.cuda.memory_allocated() - baseline
    del cache
    gc.collect()
    return measured


def main() -> None:
    set_seed(42)
    env = log_environment()
    model, tok = load_model(torch.bfloat16)
    kw = detect_logits_kw(model)
    cfg = config_summary(model)

    per_token_gqa = 2 * cfg["num_hidden_layers"] * cfg["num_key_value_heads"] * cfg["head_dim"] * DTYPE_BYTES
    per_token_mha = 2 * cfg["num_hidden_layers"] * cfg["num_attention_heads"] * cfg["head_dim"] * DTYPE_BYTES
    weights_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    rows = []
    for length in SEQ_LENGTHS:
        ids = make_input_ids(tok, length)
        measured = measure_cache_bytes(model, ids, kw)
        formula = per_token_gqa * length
        rows.append(
            {
                "seq_len": length,
                "measured_bytes": measured,
                "formula_bytes": formula,
                "mha_hypothetical_bytes": per_token_mha * length,
                "relative_error": (measured - formula) / formula,
            }
        )
        logger.info(
            f"L={length}: measured {measured / 2**20:.1f} MiB, "
            f"formula {formula / 2**20:.1f} MiB "
            f"(err {rows[-1]['relative_error']:+.2%})"
        )

    save_results(
        "e4_memory",
        {
            "env": env,
            "config": cfg,
            "dtype_bytes": DTYPE_BYTES,
            "per_token_bytes_gqa": per_token_gqa,
            "per_token_bytes_mha_hypothetical": per_token_mha,
            "model_weights_bytes": weights_bytes,
            "rows": rows,
        },
    )


if __name__ == "__main__":
    main()
