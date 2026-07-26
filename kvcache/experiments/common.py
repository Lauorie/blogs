"""Shared utilities for KV cache experiments on Qwen2.5-3B.

All experiments run on a single GPU with greedy decoding, fixed seeds,
and CUDA-synchronized wall-clock timing.
"""

from __future__ import annotations

import json
import logging
import os
import random
import statistics
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger("kvcache-exp")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(_h)

MODEL_PATH = "/root/autodl-fs/models/Qwen2.5-3B"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# A long, natural paragraph used to build prompts of arbitrary token length.
FILLER_TEXT = (
    "The history of computing is a story of layered abstractions. Machine code "
    "gave way to assembly, assembly to compilers, and compilers to interpreters "
    "and virtual machines. Each layer trades a little efficiency for a lot of "
    "productivity, and each generation of engineers rediscovers the cost of that "
    "trade when performance matters. Modern deep learning sits at the top of this "
    "tower: a few lines of Python orchestrate billions of floating point "
    "operations on specialized hardware, scheduled by runtimes that most users "
    "never read. Understanding one layer beneath your daily work is often the "
    "highest-leverage learning investment an engineer can make. "
)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def log_environment() -> dict:
    """Record environment information for reproducibility."""
    import platform

    import transformers

    env = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "gpu_count": torch.cuda.device_count(),
        "model_path": MODEL_PATH,
    }
    logger.info(f"Environment: {json.dumps(env, indent=2)}")
    return env


def load_model(
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load Qwen2.5-3B onto cuda:0 in the given dtype."""
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=dtype).to("cuda").eval()
    logger.info(f"Model loaded in {time.perf_counter() - t0:.1f}s (dtype={dtype})")
    return model, tok


def config_summary(model: PreTrainedModel) -> dict:
    """Extract the config fields that determine KV cache size."""
    cfg = model.config
    return {
        "num_hidden_layers": cfg.num_hidden_layers,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "hidden_size": cfg.hidden_size,
        "head_dim": cfg.hidden_size // cfg.num_attention_heads,
        "vocab_size": cfg.vocab_size,
        "max_position_embeddings": cfg.max_position_embeddings,
    }


def make_input_ids(tok: PreTrainedTokenizerBase, length: int) -> torch.Tensor:
    """Build a [1, length] input_ids tensor of natural text on cuda."""
    reps = length // 100 + 2
    ids = tok(FILLER_TEXT * reps, return_tensors="pt").input_ids[:, :length]
    if ids.shape[1] < length:
        raise ValueError(f"Filler text too short for length {length}")
    return ids.cuda()


def cuda_timed(fn: Callable[[], object]) -> float:
    """Run fn once with CUDA sync before/after; return elapsed seconds."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def timed_median(fn: Callable[[], object], warmup: int, reps: int) -> float:
    """Median wall-clock seconds over reps runs after warmup runs."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    return statistics.median(cuda_timed(fn) for _ in range(reps))


@torch.inference_mode()
def detect_logits_kw(model: PreTrainedModel) -> str | None:
    """Find the kwarg that limits lm_head to the last position(s).

    transformers v5 uses `logits_to_keep`; some v4 releases used
    `num_logits_to_keep`. Returns None if neither is accepted.
    """
    probe = torch.tensor([[1, 2]], device="cuda")
    for kw in ("logits_to_keep", "num_logits_to_keep"):
        try:
            out = model(input_ids=probe, use_cache=False, **{kw: 1})
            if out.logits.shape[1] == 1:
                logger.info(f"Using lm_head limiting kwarg: {kw}")
                return kw
        except TypeError:
            continue
    logger.warning("No logits-limiting kwarg accepted; full logits will be computed")
    return None


def save_results(name: str, payload: dict) -> Path:
    """Write a result dict to results/<name>.json."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(f"Saved {path}")
    return path
