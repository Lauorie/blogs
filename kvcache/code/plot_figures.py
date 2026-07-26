"""Render the three data figures for the KV cache blog from experiment JSONs.

Usage: python plot_figures.py  (reads ../experiments/results, writes ../figures)
"""

from __future__ import annotations

import json
import logging
import statistics
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "experiments" / "results"
FIGURES = ROOT / "figures"

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_3": "#8BCF8B",
    "red_strong": "#B64342",
    "neutral": "#CFCECE",
    "teal": "#42949E",
}


@dataclass(frozen=True)
class FigureStyle:
    font_size: int = 13
    axes_linewidth: float = 1.6


def apply_publication_style(style: FigureStyle) -> None:
    """Configure rcParams: CJK-capable sans fonts, no top/right spines."""
    cjk_otf = Path.home() / ".fonts" / "NotoSansCJKsc-Regular.otf"
    if cjk_otf.exists():
        font_manager.fontManager.addfont(str(cjk_otf))
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Noto Sans CJK SC", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": style.font_size,
            "axes.linewidth": style.axes_linewidth,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "savefig.bbox": "tight",
        }
    )


def finalize_figure(fig: plt.Figure, out_stem: Path, dpi: int = 300) -> None:
    """Save PNG and close."""
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    path = out_stem.with_suffix(".png")
    fig.savefig(path, dpi=dpi, pad_inches=0.06)
    plt.close(fig)
    logger.info(f"saved {path}")


def load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())


def bucket_median(steps: list[dict], width: int = 128) -> tuple[list[int], list[float]]:
    """Median per-step decode latency within ctx buckets of the given width."""
    buckets: dict[int, list[float]] = {}
    for s in steps:
        buckets.setdefault(s["ctx"] // width, []).append(s["seconds"])
    xs = [b * width + width // 2 for b in sorted(buckets)]
    ys = [statistics.median(buckets[b]) * 1e3 for b in sorted(buckets)]
    return xs, ys


def fig_latency(e2: dict) -> None:
    """Panel A: per-token latency vs context; Panel B: end-to-end totals."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.4), width_ratios=[1.7, 1])

    nc_x = [r["ctx"] for r in e2["no_cache_forward"]]
    nc_y = [r["seconds"] * 1e3 for r in e2["no_cache_forward"]]
    wc_x, wc_y = bucket_median(e2["with_cache_steps"])

    ax.plot(nc_x, nc_y, "o-", color=PALETTE["red_strong"], lw=2.2, ms=5,
            label="无缓存：整段重算一次前向")
    ax.plot(wc_x, wc_y, "-", color=PALETTE["blue_main"], lw=2.2,
            label="有 KV Cache：每步只算 1 个 token")
    ax.set_xlabel("上下文长度（token 数）")
    ax.set_ylabel("生成 1 个 token 的耗时（ms）")
    ax.set_xlim(0, 4200)
    ax.set_ylim(0, 160)
    ax.annotate(
        f"ctx=4096 时相差 {nc_y[-1] / wc_y[-1]:.1f} 倍",
        xy=(4096, nc_y[-1]), xytext=(2350, 128),
        arrowprops=dict(arrowstyle="->", color="#444", lw=1.2), fontsize=12,
    )
    ax.legend(loc="upper left", fontsize=11.5)

    e = e2["end_to_end"]
    bars = ax2.bar(
        ["无缓存", "有 KV Cache"],
        [e["no_cache_seconds"], e["with_cache_seconds"]],
        color=[PALETTE["red_strong"], PALETTE["blue_main"]], width=0.55,
    )
    for b, v in zip(bars, [e["no_cache_seconds"], e["with_cache_seconds"]]):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f} s",
                 ha="center", fontsize=12)
    ax2.set_ylabel("生成 1024 个 token 总耗时（s）")
    ax2.set_ylim(0, 33)
    ax2.set_title(f"端到端 {e['speedup']:.2f}×（且随长度继续拉大）", fontsize=12)

    fig.suptitle("Qwen2.5-3B / RTX 5090 / bf16 / 贪心解码", fontsize=11, y=1.02, color="#555")
    finalize_figure(fig, FIGURES / "fig_latency")


def fig_ttft(e3: dict) -> None:
    """Prefill time vs prompt length, with flat decode-step latency for contrast."""
    fig, ax = plt.subplots(figsize=(8, 4.4))
    xs = [r["prompt_len"] for r in e3["rows"]]
    prefill = [r["prefill_seconds"] * 1e3 for r in e3["rows"]]
    decode = [r["decode_step_seconds"] * 1e3 for r in e3["rows"]]

    ax.plot(xs, prefill, "o-", color=PALETTE["blue_main"], lw=2.2, ms=5,
            label="prefill：一次并行吃掉整个 prompt（≈TTFT）")
    ax.plot(xs, decode, "s--", color=PALETTE["teal"], lw=2, ms=4,
            label="decode：之后每个 token 的耗时")
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs, [str(x) for x in xs])
    ax.set_xlabel("prompt 长度（token 数）")
    ax.set_ylabel("耗时（ms）")
    last = e3["rows"][-1]
    per_tok = last["prefill_seconds"] * 1e3 / last["prompt_len"]
    ax.annotate(
        f"16K token 的 prompt 只要 {last['prefill_seconds'] * 1e3:.0f} ms\n"
        f"摊到每个 token 仅 {per_tok:.3f} ms",
        xy=(16384, last["prefill_seconds"] * 1e3), xytext=(1400, 560),
        arrowprops=dict(arrowstyle="->", color="#444", lw=1.2), fontsize=11.5,
    )
    ax.legend(loc="upper left", fontsize=11.5)
    ax.set_title("Qwen2.5-3B / RTX 5090 / bf16", fontsize=11, color="#555")
    finalize_figure(fig, FIGURES / "fig_ttft")


def fig_memory(e4: dict) -> None:
    """KV cache memory: formula line, measured points, MHA hypothetical, weights."""
    fig, ax = plt.subplots(figsize=(8, 4.6))
    xs = [r["seq_len"] for r in e4["rows"]]
    measured = [r["measured_bytes"] / 2**30 for r in e4["rows"]]
    formula = [r["formula_bytes"] / 2**30 for r in e4["rows"]]
    mha = [r["mha_hypothetical_bytes"] / 2**30 for r in e4["rows"]]
    weights_gib = e4["model_weights_bytes"] / 2**30

    ax.plot(xs, mha, "^--", color=PALETTE["red_strong"], lw=2, ms=6,
            label="假设不用 GQA（16 个 KV 头）")
    ax.plot(xs, formula, "-", color=PALETTE["blue_main"], lw=2.2,
            label="公式：2×36层×2头×128维×2字节×L")
    ax.plot(xs, measured, "o", color=PALETTE["blue_secondary"], ms=7, mfc="white", mew=2,
            label="实测（torch.cuda.memory_allocated）")
    ax.axhline(weights_gib, color="#888", ls=":", lw=1.8)
    ax.text(xs[0], weights_gib + 0.25, f"模型权重 {weights_gib:.2f} GiB", fontsize=11, color="#555")

    ax.set_xscale("log", base=2)
    ax.set_xticks(xs, [str(x) for x in xs])
    ax.set_xlabel("序列长度 L（token 数）")
    ax.set_ylabel("单条请求的 KV Cache 显存（GiB）")
    ax.annotate("GQA 省 8×", xy=(32768, mha[-1]), xytext=(9000, 7.2),
                arrowprops=dict(arrowstyle="->", color="#444", lw=1.2), fontsize=12)
    ax.legend(loc="upper left", fontsize=11)
    ax.set_title("Qwen2.5-3B（GQA：16 个 Q 头共享 2 个 KV 头）/ bf16", fontsize=11, color="#555")
    finalize_figure(fig, FIGURES / "fig_memory")


def main() -> None:
    apply_publication_style(FigureStyle())
    fig_latency(load("e2_latency"))
    fig_ttft(load("e3_ttft"))
    fig_memory(load("e4_memory"))


if __name__ == "__main__":
    main()
