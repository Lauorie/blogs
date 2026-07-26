"""E1: Verify that KV caching is numerically equivalent to full recomputation.

Greedy-decode the same prompt twice — once recomputing the full sequence every
step (use_cache=False), once with a DynamicCache — and compare the generated
token ids and the per-step next-token logits, in bf16 and fp32.
"""

from __future__ import annotations

import torch
from transformers import DynamicCache, PreTrainedModel, PreTrainedTokenizerBase

from common import load_model, log_environment, logger, save_results, set_seed

PROMPT = "In a transformer language model, the attention mechanism works as follows:"
NUM_NEW_TOKENS = 64


@torch.inference_mode()
def greedy_no_cache(
    model: PreTrainedModel, prompt_ids: torch.Tensor, num_new: int
) -> tuple[list[int], list[torch.Tensor]]:
    """Generate greedily, recomputing the whole sequence at every step."""
    seq = prompt_ids
    tokens: list[int] = []
    step_logits: list[torch.Tensor] = []
    for _ in range(num_new):
        logits = model(input_ids=seq, use_cache=False).logits[:, -1, :]
        step_logits.append(logits.float().cpu())
        nxt = int(logits.argmax(dim=-1))
        tokens.append(nxt)
        seq = torch.cat([seq, torch.tensor([[nxt]], device=seq.device)], dim=1)
    return tokens, step_logits


@torch.inference_mode()
def greedy_with_cache(
    model: PreTrainedModel, prompt_ids: torch.Tensor, num_new: int
) -> tuple[list[int], list[torch.Tensor]]:
    """Generate greedily, forwarding only the newest token each step."""
    cache = DynamicCache()
    out = model(input_ids=prompt_ids, past_key_values=cache, use_cache=True)
    tokens: list[int] = []
    step_logits: list[torch.Tensor] = []
    logits = out.logits[:, -1, :]
    for _ in range(num_new):
        step_logits.append(logits.float().cpu())
        nxt = int(logits.argmax(dim=-1))
        tokens.append(nxt)
        step_ids = torch.tensor([[nxt]], device=prompt_ids.device)
        out = model(input_ids=step_ids, past_key_values=cache, use_cache=True)
        logits = out.logits[:, -1, :]
    return tokens, step_logits


def compare(
    model: PreTrainedModel, tok: PreTrainedTokenizerBase, dtype_name: str
) -> dict:
    """Run both decoders and report token agreement and logits differences."""
    prompt_ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()
    toks_a, logits_a = greedy_no_cache(model, prompt_ids, NUM_NEW_TOKENS)
    toks_b, logits_b = greedy_with_cache(model, prompt_ids, NUM_NEW_TOKENS)

    n_match = sum(a == b for a, b in zip(toks_a, toks_b))
    diffs = [float((la - lb).abs().max()) for la, lb in zip(logits_a, logits_b)]
    result = {
        "dtype": dtype_name,
        "prompt": PROMPT,
        "num_new_tokens": NUM_NEW_TOKENS,
        "tokens_identical": toks_a == toks_b,
        "num_matching_tokens": n_match,
        "max_abs_logits_diff": max(diffs),
        "mean_abs_logits_diff_of_max": sum(diffs) / len(diffs),
        "text_no_cache": tok.decode(toks_a),
        "text_with_cache": tok.decode(toks_b),
    }
    logger.info(
        f"[{dtype_name}] tokens identical: {result['tokens_identical']} "
        f"({n_match}/{NUM_NEW_TOKENS}), max |Δlogits| = {result['max_abs_logits_diff']:.3e}"
    )
    return result


def main() -> None:
    set_seed(42)
    env = log_environment()
    model, tok = load_model(torch.bfloat16)
    runs = [compare(model, tok, "bfloat16")]

    model = model.float()
    runs.append(compare(model, tok, "float32"))

    save_results("e1_equivalence", {"env": env, "runs": runs})


if __name__ == "__main__":
    main()
