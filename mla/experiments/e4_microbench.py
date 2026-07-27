"""E4：解码微基准——"解压式 MLA" vs "吸收式 MLA" vs "标准 MHA 缓存"。

结构复现 DeepSeek-V2 单层注意力（128 头 x 128 维、d_c=512、d_R=64），权重随机、
bf16、单步解码（batch=1），测不同上下文长度下的每步延迟与解压路径的临时显存。
权重随机不影响耗时/显存结论（计算图与真实模型同形）。输出 results/e4_microbench.json。
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch

torch.manual_seed(42)
DEV, DT = "cuda", torch.bfloat16
D, H, DH, DC, DR = 5120, 128, 128, 512, 64
CTX = [1024, 4096, 8192, 16384, 32768]
REPS, WARMUP = 20, 5

W_uk = (torch.randn(H, DC, DH) / DC**0.5).to(DEV, DT)
W_uv = (torch.randn(H, DC, DH) / DC**0.5).to(DEV, DT)


def timed(fn) -> float:
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


@torch.inference_mode()
def bench(t: int) -> dict:
    c = torch.randn(t, DC, device=DEV, dtype=DT)          # latent 缓存
    kr = torch.randn(t, DR, device=DEV, dtype=DT)         # 解耦 RoPE 缓存
    K = torch.randn(t, H, DH, device=DEV, dtype=DT)       # MHA 基线的完整缓存
    V = torch.randn(t, H, DH, device=DEV, dtype=DT)
    q = torch.randn(H, DH, device=DEV, dtype=DT)
    qr = torch.randn(H, DR, device=DEV, dtype=DT)

    def mha_step() -> None:
        w = (torch.einsum("he,the->ht", q, K) / DH**0.5).softmax(-1)
        torch.einsum("ht,the->he", w, V)

    def decompress_step() -> None:                        # 每步把整段 K/V 从 latent 重建
        Kd = torch.einsum("tc,hce->the", c, W_uk)
        Vd = torch.einsum("tc,hce->the", c, W_uv)
        s = torch.einsum("he,the->ht", q, Kd) + torch.einsum("he,te->ht", qr, kr)
        w = (s / (DH + DR) ** 0.5).softmax(-1)
        torch.einsum("ht,the->he", w, Vd)

    def absorbed_step() -> None:                          # 吸收：全程在 latent 空间
        qt = torch.einsum("he,hce->hc", q, W_uk)
        s = torch.einsum("hc,tc->ht", qt, c) + torch.einsum("he,te->ht", qr, kr)
        w = (s / (DH + DR) ** 0.5).softmax(-1)
        torch.einsum("hc,hce->he", torch.einsum("ht,tc->hc", w, c), W_uv)

    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    t_dec = timed(decompress_step)
    peak_dec = torch.cuda.max_memory_allocated() - base   # 解压的临时大张量

    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    t_abs = timed(absorbed_step)
    peak_abs = torch.cuda.max_memory_allocated() - base

    return {"ctx": t, "mha_ms": timed(mha_step) * 1e3,
            "decompress_ms": t_dec * 1e3, "absorbed_ms": t_abs * 1e3,
            "decompress_peak_mb": peak_dec / 2**20, "absorbed_peak_mb": peak_abs / 2**20,
            "cache_mha_mb": (K.nbytes + V.nbytes) / 2**20,
            "cache_mla_mb": (c.nbytes + kr.nbytes) / 2**20}


def main() -> None:
    rows = []
    for t in CTX:
        r = bench(t)
        rows.append(r)
        print(f"ctx={t:6d}  MHA {r['mha_ms']:6.2f} ms | 解压 {r['decompress_ms']:7.2f} ms "
              f"(临时峰值 {r['decompress_peak_mb']:7.1f} MB) | 吸收 {r['absorbed_ms']:6.2f} ms "
              f"(峰值 {r['absorbed_peak_mb']:5.1f} MB)")
    out = Path(__file__).resolve().parent / "results"
    out.mkdir(exist_ok=True)
    (out / "e4_microbench.json").write_text(json.dumps(
        {"env": {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__},
         "dims": {"D": D, "H": H, "DH": DH, "DC": DC, "DR": DR}, "rows": rows}, indent=2))
    print("saved results/e4_microbench.json")


if __name__ == "__main__":
    main()
