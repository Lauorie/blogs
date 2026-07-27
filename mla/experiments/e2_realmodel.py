"""E2：真机对账——MiniCPM3-4B（MLA）vs Qwen2.5-3B（GQA）。

同一张卡上分别测量：
- 不同 prefill 长度下 KV 缓存显存（差值法，每长度测两遍取第二遍：首遍会混入
  该长度的一次性缓冲，第二遍的增量就是缓存本体；两遍都落盘）
- prefill 后单步 decode 延迟（中位数）

逐长度捕获 OOM（记为一行数据并停止加长，已测行保留）；每测完一个长度立即落盘。
输出 results/e2_realmodel.json。MiniCPM3 需 trust_remote_code（transformers 4.46）。
"""

from __future__ import annotations

import gc
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

RESULTS = Path(__file__).resolve().parent / "results"
OUT = RESULTS / "e2_realmodel.json"
LENGTHS = [1024, 4096, 8192, 16384]
FILLER = ("The history of computing is a story of layered abstractions. " * 1600)

MODELS = [
    ("MiniCPM3-4B", "/root/autodl-fs/models/MiniCPM3-4B", True),
    ("Qwen2.5-3B", "/root/autodl-fs/models/Qwen2_5-3B", False),
]


def formula_bytes_per_token(cfg) -> tuple[int, str]:
    if getattr(cfg, "kv_lora_rank", None):  # MLA
        per = (cfg.kv_lora_rank + cfg.qk_rope_head_dim) * 2 * cfg.num_hidden_layers
        return per, f"({cfg.kv_lora_rank}+{cfg.qk_rope_head_dim})x2Bx{cfg.num_hidden_layers}层"
    n_kv = cfg.num_key_value_heads
    d_h = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    per = 2 * n_kv * d_h * 2 * cfg.num_hidden_layers
    return per, f"2x{n_kv}x{d_h}x2Bx{cfg.num_hidden_layers}层"


def cleanup() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


@torch.inference_mode()
def cache_delta(model, ids) -> int:
    """一次 prefill 的显存增量（只保留缓存对象后读数）。"""
    cleanup()
    base = torch.cuda.memory_allocated()
    out = model(input_ids=ids, use_cache=True)
    cache = out.past_key_values
    del out
    gc.collect()
    torch.cuda.synchronize()
    delta = torch.cuda.memory_allocated() - base
    del cache
    return delta


@torch.inference_mode()
def probe(name: str, path: str, remote: bool, save) -> dict:
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=remote)
    # v5 用 dtype，v4 用 torch_dtype（trust_remote_code 的老模型代码只认后者）
    dtype_kw = ("dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype")
    model = AutoModelForCausalLM.from_pretrained(
        path, trust_remote_code=remote, **{dtype_kw: torch.bfloat16}).to("cuda").eval()
    per_formula, formula = formula_bytes_per_token(model.config)

    ids_full = tok(FILLER, return_tensors="pt").input_ids
    result = {"model": name,
              "params_b": sum(p.numel() for p in model.parameters()) / 1e9,
              "transformers": transformers.__version__,
              "formula": formula, "formula_per_token": per_formula, "rows": []}

    for length in LENGTHS:
        if ids_full.shape[1] < length:
            break
        ids = ids_full[:, :length].cuda()
        try:
            first = cache_delta(model, ids)    # 首遍：缓存 + 该长度的一次性缓冲
            second = cache_delta(model, ids)   # 第二遍：一次性缓冲已在位，增量≈缓存本体

            # 单步 decode 延迟（重新建缓存，连续 5 步取中位数）
            cleanup()
            out = model(input_ids=ids, use_cache=True)
            cache, nxt = out.past_key_values, ids[:, -1:]
            del out
            times = []
            for _ in range(3):
                o = model(input_ids=nxt, past_key_values=cache, use_cache=True)
                cache, nxt = o.past_key_values, o.logits[:, -1:].argmax(-1)
            for _ in range(5):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                o = model(input_ids=nxt, past_key_values=cache, use_cache=True)
                torch.cuda.synchronize(); times.append(time.perf_counter() - t0)
                cache, nxt = o.past_key_values, o.logits[:, -1:].argmax(-1)
            del cache, o
            cleanup()
        except torch.OutOfMemoryError as exc:
            cleanup()
            result["rows"].append({"len": length, "error": f"OOM: {str(exc)[:100]}"})
            print(f"[{name}] L={length}: OOM，保留已测行并停止加长")
            save(result)
            break

        row = {"len": length, "cache_bytes": second, "first_pass_bytes": first,
               "measured_per_token": second / length,
               "first_pass_per_token": first / length,
               "formula_per_token": per_formula,
               "rel_err": second / length / per_formula - 1,
               "decode_ms": statistics.median(times) * 1e3}
        result["rows"].append(row)
        save(result)  # 每个长度立即落盘
        print(f"[{name}] L={length:6d} 缓存 {second / 2**20:7.1f} MiB "
              f"({second / length:8.0f} B/tok，首遍 {first / length:8.0f}) "
              f"vs 公式 {per_formula} B/tok ({row['rel_err']:+.1%})  "
              f"decode {row['decode_ms']:.2f} ms")

    del model
    cleanup()
    return result


def main() -> None:
    want = sys.argv[1] if len(sys.argv) > 1 else None
    prev = {}
    if OUT.exists():
        for m in json.loads(OUT.read_text()).get("models", []):
            prev[m["model"]] = m

    out = {"env": {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__},
           "models": []}

    def save(current=None) -> None:
        models = []
        for name, _, _ in MODELS:
            if current is not None and current["model"] == name:
                models.append(current)
            else:
                entry = next((m for m in out["models"] if m["model"] == name),
                             prev.get(name))
                if entry is not None:
                    models.append(entry)
        RESULTS.mkdir(exist_ok=True)
        OUT.write_text(json.dumps({**out, "models": models}, indent=2))

    for name, path, remote in MODELS:
        if want and name != want:
            reused = prev.get(name)
            if reused is not None:
                if not reused.get("rows"):
                    print(f"[warn] 沿用的 {name} 旧条目没有数据行：{reused.get('error', '')[:80]}")
                out["models"].append(reused)
            continue
        entry = probe(name, path, remote, save)
        out["models"].append(entry)
        save()
    save()
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
