"""绘制 MLA 博客的两张数据图：架构账本（E3）与解码微基准（E4）。

Usage: python draw_data_figs.py  (reads ../experiments/results, writes ../figures)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "experiments" / "results"
FIGURES = ROOT / "figures"

BLUE, LBLUE, RED, TEAL, GRAY = "#0F4D92", "#3775BA", "#B64342", "#42949E", "#8a8a8a"


def style() -> None:
    otf = Path.home() / ".fonts" / "NotoSansCJKsc-Regular.otf"
    if otf.exists():
        font_manager.fontManager.addfont(str(otf))
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Noto Sans CJK SC", "DejaVu Sans"],
        "axes.unicode_minus": False, "font.size": 12.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "savefig.bbox": "tight",
    })


def fig_ledger() -> None:
    rows = [r for r in json.loads((RESULTS / "e3_ledger.json").read_text()) if "error" not in r]
    label = {"NousResearch/Llama-2-7b-hf": "Llama-2-7B\nMHA·32头",
             "Qwen/Qwen2.5-3B": "Qwen2.5-3B\nGQA·2组",
             "Qwen/Qwen2.5-72B": "Qwen2.5-72B\nGQA·8组",
             "openbmb/MiniCPM3-4B": "MiniCPM3-4B\nMLA·d_c=256",
             "deepseek-ai/DeepSeek-V2": "DeepSeek-V2\n236B·MLA",
             "deepseek-ai/DeepSeek-V3": "DeepSeek-V3\n671B·MLA"}
    rows.sort(key=lambda r: -r["per_token_bytes"])
    names = [label[r["model"]] for r in rows]
    kib = [r["per_token_bytes"] / 1024 for r in rows]
    colors = [RED if "MHA" in label[r["model"]] else (LBLUE if r["kind"] == "GQA" else TEAL)
              for r in rows]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    bars = ax.bar(names, kib, color=colors, width=0.62)
    for b, v in zip(bars, kib):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.06, f"{v:.0f}", ha="center", fontsize=12)
    ax.set_yscale("log")
    ax.set_ylim(20, 900)
    ax.set_ylabel("每 token 的 KV 缓存（KiB，bf16，全部层）")
    ax.set_title("从各模型 config.json 逐字段算出（公式见正文）", fontsize=11, color="#555")
    ax.annotate("671B 的 V3 比 7B 的 MHA 还省 7.5 倍", xy=(2.15, 78), xytext=(2.9, 420),
                arrowprops=dict(arrowstyle="->", color="#444", lw=1.2), fontsize=12.5)
    fig.savefig(FIGURES / "fig_ledger.png", dpi=200)
    plt.close(fig)
    print("saved fig_ledger.png")


def fig_microbench() -> None:
    data = json.loads((RESULTS / "e4_microbench.json").read_text())
    rows = data["rows"]
    x = [r["ctx"] for r in rows]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))

    ax.plot(x, [r["decompress_ms"] for r in rows], "o-", color=RED, lw=2.2,
            label="MLA 解压式：每步重建 K/V")
    ax.plot(x, [r["mha_ms"] for r in rows], "s-", color=GRAY, lw=2,
            label="MHA：读完整 K/V 缓存")
    ax.plot(x, [r["absorbed_ms"] for r in rows], "^-", color=TEAL, lw=2.2,
            label="MLA 吸收式：全程 latent")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(x, [f"{t//1024}K" for t in x])
    ax.set_xlabel("上下文长度")
    ax.set_ylabel("单步注意力耗时（ms，单层）")
    ax.legend(fontsize=11)

    ax2.plot(x, [r["decompress_peak_mb"] for r in rows], "o-", color=RED, lw=2.2,
             label="解压式临时显存峰值")
    ax2.plot(x, [r["absorbed_peak_mb"] for r in rows], "^-", color=TEAL, lw=2.2,
             label="吸收式临时显存峰值")
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.set_xticks(x, [f"{t//1024}K" for t in x])
    ax2.set_xlabel("上下文长度")
    ax2.set_ylabel("单步临时显存（MiB，单层）")
    ax2.annotate("32K 时 2 GiB / 步", xy=(32768, 2072), xytext=(4000, 700),
                 arrowprops=dict(arrowstyle="->", color="#444", lw=1.2), fontsize=12)
    ax2.legend(fontsize=11)

    fig.suptitle("DeepSeek-V2 维度的单层注意力微基准（128头×128维，d_c=512，RTX 5090，bf16，batch=1）",
                 fontsize=11.5, color="#555", y=1.02)
    fig.savefig(FIGURES / "fig_microbench.png", dpi=200)
    plt.close(fig)
    print("saved fig_microbench.png")


if __name__ == "__main__":
    style()
    FIGURES.mkdir(exist_ok=True)
    fig_ledger()
    fig_microbench()
