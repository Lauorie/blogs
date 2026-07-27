"""绘制矩阵吸收示意图：同一个分数的两种算法（解压式 vs 吸收式）。

Usage: python draw_absorb.py  (writes ../figures/fig_absorb.png)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

TEAL, TEAL_D = "#D5EDE8", "#2A9D8F"
ORANGE, ORANGE_D = "#FDEBD0", "#E67E22"
RED, GREEN, DARK = "#B64342", "#1F9D55", "#3A4450"
OUT = Path(__file__).resolve().parent.parent / "figures" / "fig_absorb.png"


def box(ax, x, y, w, h, fc, ec, lw=1.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.10",
                                fc=fc, ec=ec, lw=lw))


def arrow(ax, p0, p1):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=15,
                                 lw=1.7, color=DARK))


def main() -> None:
    otf = Path.home() / ".fonts" / "NotoSansCJKsc-Regular.otf"
    if otf.exists():
        font_manager.fontManager.addfont(str(otf))
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Noto Sans CJK SC", "DejaVu Sans"]})

    fig, ax = plt.subplots(figsize=(13.2, 6.8))
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 8.8)
    ax.set_axis_off()

    ax.text(6.6, 8.35, r"同一个注意力分数：  $q\cdot(W^{UK}c_j)\;=\;(W^{UK\top}q)\cdot c_j$",
            fontsize=17, ha="center", color=DARK)
    ax.text(6.6, 7.75, "括号放哪边，代价天差地别", fontsize=13, ha="center", color="#777")

    # 左：解压式——c 列在左，升维成 K 列，q 在 K 列右侧，短箭头逐一对照（不穿盒）
    ax.text(3.1, 7.0, "解压式：把每条速记展开成全文", fontsize=13.5, ha="center",
            color=RED, fontweight="bold")
    for i in range(3):
        y = 5.6 - i * 1.05
        box(ax, 0.35, y, 1.15, 0.7, TEAL, TEAL_D)
        ax.text(0.925, y + 0.35, f"$c_{{{i + 1}}}$·512维", fontsize=11, ha="center", va="center")
        arrow(ax, (1.6, y + 0.35), (2.35, y + 0.35))
        box(ax, 2.45, y, 2.0, 0.7, "white", TEAL_D)
        ax.text(3.45, y + 0.35, f"$k_{{{i + 1}}}$·16384维", fontsize=11, ha="center", va="center")
        arrow(ax, (5.35, 4.5 - (1 - i) * 0.35), (4.55, y + 0.35))
    ax.text(2.0, 6.2, "$W^{UK}$ 升维", fontsize=11, ha="center", color="#666")
    ax.text(0.925, 2.55, "…共 T 条", fontsize=11, ha="center", color="#666")
    box(ax, 5.4, 4.15, 0.95, 0.7, ORANGE, ORANGE_D)
    ax.text(5.875, 4.5, "q", fontsize=13, ha="center", va="center")
    ax.text(3.1, 0.7, "每生成一步，T 条速记全部展开一遍", fontsize=12, ha="center", color=RED)

    # 右：吸收式——q 先翻译成 q̃，三条 c 水平排开，扇形箭头（不穿盒）
    ax.text(9.9, 7.0, "吸收式：把问题翻译成速记的语言", fontsize=13.5, ha="center",
            color=GREEN, fontweight="bold")
    box(ax, 7.5, 5.3, 1.5, 0.7, ORANGE, ORANGE_D)
    ax.text(8.25, 5.65, "q·16384维", fontsize=11, ha="center", va="center")
    arrow(ax, (9.05, 5.65), (9.95, 5.65))
    ax.text(9.5, 5.95, "$W^{UK\\top}$", fontsize=11, ha="center", color="#666")
    box(ax, 10.05, 5.3, 1.6, 0.7, ORANGE, ORANGE_D)
    ax.text(10.85, 5.65, r"$\tilde{q}$·512维/头", fontsize=10.5, ha="center", va="center")
    ax.text(10.85, 6.25, "只翻译这一次", fontsize=11, ha="center", color=GREEN)
    for i in range(3):
        x = 8.15 + i * 1.85
        box(ax, x, 3.1, 1.3, 0.7, TEAL, TEAL_D)
        ax.text(x + 0.65, 3.45, f"$c_{{{i + 1}}}$·512维", fontsize=11, ha="center", va="center")
        arrow(ax, (10.85, 5.25), (x + 0.65, 3.85))
    ax.text(10.5, 2.55, "…共 T 条，原样对照，不展开", fontsize=11, ha="center", color="#666")
    ax.text(9.9, 0.7, "K 从头到尾没被造出来", fontsize=12, ha="center", color=GREEN)

    ax.plot([6.6, 6.6], [0.5, 7.3], ls=":", lw=1.4, color="#bbb")
    ax.text(6.6, 0.18, "维度口径：q、k 为 128 头合计（128×128）；$\\tilde{q}$ 每头 512 维；c 为全部头共享的一条",
            fontsize=10.5, ha="center", color="#888")

    fig.savefig(OUT, dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
