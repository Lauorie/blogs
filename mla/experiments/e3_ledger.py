"""E3：KV 缓存账本——从真实 config.json 逐字段算出各架构每 token 的缓存字节数。

只下载各模型的 config.json（几 KB），不下权重。输出 results/e3_ledger.json。
"""

from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import hf_hub_download

RESULTS = Path(__file__).resolve().parent / "results"
DTYPE_BYTES = 2  # bf16

MODELS = [
    ("NousResearch/Llama-2-7b-hf", "MHA"),  # meta-llama 原仓库需要授权，用镜像取 config
    ("Qwen/Qwen2.5-3B", "GQA"),
    ("Qwen/Qwen2.5-72B", "GQA"),
    ("openbmb/MiniCPM3-4B", "MLA"),
    ("deepseek-ai/DeepSeek-V2", "MLA"),
    ("deepseek-ai/DeepSeek-V3", "MLA"),
]


def per_token_bytes(cfg: dict, kind: str) -> tuple[int, str]:
    """每 token 每层的 KV 缓存字节数，附计算式说明。"""
    if kind == "MLA":
        d_c, d_r = cfg["kv_lora_rank"], cfg["qk_rope_head_dim"]
        return (d_c + d_r) * DTYPE_BYTES, f"({d_c}+{d_r})x{DTYPE_BYTES}B"
    n_kv = cfg.get("num_key_value_heads", cfg["num_attention_heads"])
    d_h = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    return 2 * n_kv * d_h * DTYPE_BYTES, f"2x{n_kv}x{d_h}x{DTYPE_BYTES}B"


def main() -> None:
    rows = []
    for repo, kind in MODELS:
        try:
            path = hf_hub_download(repo, "config.json")
        except Exception as exc:  # 私有仓库/网络问题：记录并跳过，不编数
            rows.append({"model": repo, "kind": kind, "error": str(exc)[:120]})
            continue
        cfg = json.loads(Path(path).read_text())
        per_layer, formula = per_token_bytes(cfg, kind)
        layers = cfg["num_hidden_layers"]
        rows.append({
            "model": repo, "kind": kind, "layers": layers,
            "heads": cfg["num_attention_heads"],
            "kv_heads": cfg.get("num_key_value_heads"),
            "kv_lora_rank": cfg.get("kv_lora_rank"),
            "qk_rope_head_dim": cfg.get("qk_rope_head_dim"),
            "per_token_per_layer_bytes": per_layer,
            "per_token_bytes": per_layer * layers,
            "bytes_at_32k": per_layer * layers * 32768,
            "formula": formula,
        })
        print(f"{repo:36s} {kind:4s} 每token {per_layer * layers / 1024:8.1f} KiB "
              f"(每层 {formula} x {layers} 层)")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "e3_ledger.json").write_text(json.dumps(rows, indent=2))
    print(f"saved {RESULTS / 'e3_ledger.json'}")


if __name__ == "__main__":
    main()
