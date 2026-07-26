"""Draw fig1b: the autoregressive loop with the concrete "天空为什么是" example.

Chinese labels must be exact, so this schematic is drawn programmatically.
Usage: python draw_fig1b.py  (writes ../figures/fig1b_autoregressive_loop.png)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BLUE_FILL, BLUE_EDGE = "#DCEBFA", "#3775BA"
ORANGE_FILL, ORANGE_EDGE = "#FDEBD0", "#E67E22"
DARK = "#3A4450"
OUT = Path(__file__).resolve().parent.parent / "figures" / "fig1b_autoregressive_loop.png"

BOX_W, BOX_H, GAP = 0.92, 0.86, 0.14
ROWS = [
    ("第 1 步", list("天空为什么是"), None, "蓝"),
    ("第 2 步", list("天空为什么是蓝"), 6, "色"),
    ("第 3 步", list("天空为什么是蓝色"), 7, "的"),
]


def box(ax: plt.Axes, x: float, y: float, w: float, h: float, fc: str, ec: str,
        lw: float = 1.8) -> None:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.10",
                                fc=fc, ec=ec, lw=lw))


def arrow(ax: plt.Axes, p0: tuple, p1: tuple, style: str = "-", curve: float = 0.0) -> None:
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=16,
                                 lw=1.7, color=DARK, linestyle=style,
                                 connectionstyle=f"arc3,rad={curve}"))


def main() -> None:
    cjk = Path.home() / ".fonts" / "NotoSansCJKsc-Regular.otf"
    if cjk.exists():
        font_manager.fontManager.addfont(str(cjk))
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Noto Sans CJK SC", "DejaVu Sans"]})

    fig, ax = plt.subplots(figsize=(13.76, 6.6))
    ax.set_xlim(0, 17.4)
    ax.set_ylim(0, 8.6)
    ax.set_axis_off()

    row_y = [6.6, 4.4, 2.2]
    out_centers = []
    new_centers = []
    for (label, tokens, new_idx, out_tok), y in zip(ROWS, row_y):
        ax.text(0.25, y + BOX_H / 2, label, fontsize=14, color=DARK, va="center")
        x = 1.55
        for i, t in enumerate(tokens):
            hl = new_idx is not None and i == new_idx
            box(ax, x, y, BOX_W, BOX_H, BLUE_FILL, ORANGE_EDGE if hl else BLUE_EDGE,
                2.6 if hl else 1.8)
            ax.text(x + BOX_W / 2, y + BOX_H / 2, t, fontsize=17, color="#1d3550",
                    ha="center", va="center")
            if hl:
                new_centers.append((x + BOX_W / 2, y + BOX_H))
            x += BOX_W + GAP
        # Transformer box + prediction
        tx = 10.9
        arrow(ax, (x + 0.06, y + BOX_H / 2), (tx - 0.08, y + BOX_H / 2))
        box(ax, tx, y - 0.08, 2.75, BOX_H + 0.16, "white", DARK)
        ax.text(tx + 1.375, y + BOX_H / 2, "Transformer\n一次前向", fontsize=12.5,
                color=DARK, ha="center", va="center")
        ox = tx + 2.75 + 0.7
        arrow(ax, (tx + 2.83, y + BOX_H / 2), (ox - 0.08, y + BOX_H / 2))
        box(ax, ox, y, BOX_W, BOX_H, ORANGE_FILL, ORANGE_EDGE, 2.2)
        ax.text(ox + BOX_W / 2, y + BOX_H / 2, out_tok, fontsize=17, color="#8a4b08",
                ha="center", va="center")
        ax.text(ox + BOX_W / 2, y + BOX_H + 0.22, "新字", fontsize=12, color="#8a4b08",
                ha="center")
        out_centers.append((ox + BOX_W / 2, y))

    # Dashed "append back" route: drop from the output box into the free band
    # between rows, then slant left into the next row's highlighted token.
    for (ox, oy), (nx, ny) in zip(out_centers[:2], new_centers):
        mid_y = oy - 0.65
        ax.plot([ox, ox], [oy - 0.06, mid_y], ls=(0, (5, 4)), lw=1.7, color=DARK)
        arrow(ax, (ox, mid_y), (nx, ny + 0.06), style=(0, (5, 4)))
        ax.text(nx + 1.1, ny + 0.62, "拼回输入", fontsize=12.5, color=DARK)

    ax.text(1.55, 1.15, "……如此循环，直到生成结束", fontsize=13.5, color=DARK)
    ax.text(8.7, 0.35, "自回归生成：每个新字都建立在全部旧字之上",
            fontsize=14.5, color=DARK, ha="center", fontweight="bold")

    fig.savefig(OUT, dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
