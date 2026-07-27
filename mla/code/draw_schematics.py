"""绘制 MLA 博客的两张概念示意图：速记压缩（fig_shorthand）与解耦 RoPE（fig_rope_lane）。

Usage: python draw_schematics.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

TEAL, TEAL_D = "#D5EDE8", "#2A9D8F"
PURPLE, PURPLE_D = "#EAE3F6", "#8E6DC7"
ORANGE, ORANGE_D = "#FDEBD0", "#E67E22"
GREEN_BG, GREEN_D = "#E8F5E9", "#4C9A57"
RED = "#B64342"
DARK = "#3A4450"
FIGS = Path(__file__).resolve().parent.parent / "figures"


def setup() -> None:
    otf = Path.home() / ".fonts" / "NotoSansCJKsc-Regular.otf"
    if otf.exists():
        font_manager.fontManager.addfont(str(otf))
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Noto Sans CJK SC", "DejaVu Sans"]})


def box(ax, x, y, w, h, fc, ec, lw=1.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=lw))


def arrow(ax, p0, p1, style="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=15,
                                 lw=1.7, color=DARK, linestyle=style))


def fig_shorthand() -> None:
    fig, ax = plt.subplots(figsize=(13.2, 6.4))
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 8.2)
    ax.set_axis_off()

    # 左：全文抄录（每人一大排 K/V 卡）
    ax.text(2.6, 7.6, "全文抄录（经典 KV 缓存）", fontsize=13.5, ha="center",
            color=DARK, fontweight="bold")
    box(ax, 0.5, 1.7, 4.2, 5.4, "white", DARK, 2.0)
    for i in range(4):
        y = 6.1 - i * 1.1
        ax.text(0.85, y + 0.32, f"{i + 1}号", fontsize=10.5, va="center", color="#666")
        box(ax, 1.45, y, 1.4, 0.64, TEAL, TEAL_D)
        ax.text(2.15, y + 0.32, "K·全文", fontsize=10.5, ha="center", va="center")
        box(ax, 2.95, y, 1.4, 0.64, PURPLE, PURPLE_D)
        ax.text(3.65, y + 0.32, "V·全文", fontsize=10.5, ha="center", va="center")
    ax.text(2.6, 2.05, "每位发言者 65,536 B", fontsize=11.5, ha="center", color=RED)

    # 中：压缩
    arrow(ax, (5.0, 4.4), (6.2, 4.4))
    ax.text(5.6, 4.9, "只记速记", fontsize=12, ha="center", color=DARK)
    ax.text(5.6, 3.95, "$W^{DKV}$ 降维", fontsize=10.5, ha="center", color="#666")

    # 右：速记本
    ax.text(8.6, 7.6, "速记本（MLA 的 latent 缓存）", fontsize=13.5, ha="center",
            color=GREEN_D, fontweight="bold")
    box(ax, 6.5, 1.7, 4.2, 5.4, GREEN_BG, GREEN_D, 2.0)
    for i in range(4):
        y = 6.1 - i * 1.1
        ax.text(6.85, y + 0.32, f"{i + 1}号", fontsize=10.5, va="center", color="#666")
        box(ax, 7.45, y, 1.7, 0.64, TEAL, TEAL_D)
        ax.text(8.3, y + 0.32, "c·速记 512 维", fontsize=10, ha="center", va="center")
        box(ax, 9.35, y, 1.05, 0.64, ORANGE, ORANGE_D)
        ax.text(9.875, y + 0.32, "时戳 64", fontsize=9.5, ha="center", va="center")
    ax.text(8.6, 2.05, "每位发言者 1,152 B（省 56.9 倍）", fontsize=11.5, ha="center",
            color=GREEN_D)
    ax.text(8.6, 1.35, "（\"时戳\"一小格的来历见第五节：解耦 RoPE）", fontsize=10, ha="center", color="#888")

    # 还原能力（K 上 V 下，与公式顺序一致；虚线分叉指向两个盒子）
    arrow(ax, (10.8, 4.4), (11.55, 4.87), style=(0, (5, 4)))
    arrow(ax, (10.8, 4.4), (11.55, 3.67), style=(0, (5, 4)))
    box(ax, 11.6, 4.55, 1.35, 0.64, TEAL, TEAL_D)
    ax.text(12.275, 4.87, "K·全文", fontsize=10.5, ha="center", va="center")
    box(ax, 11.6, 3.35, 1.35, 0.64, PURPLE, PURPLE_D)
    ax.text(12.275, 3.67, "V·全文", fontsize=10.5, ha="center", va="center")
    ax.text(12.25, 5.5, "需要时可还原\n（$W^{UK}$/$W^{UV}$）", fontsize=10, ha="center", color="#666")

    ax.text(6.6, 0.7, "速记不是丢信息的摘要：全文随时可以从速记严格重建（正文有逐位验证）",
            fontsize=12.5, ha="center", color=DARK)
    fig.savefig(FIGS / "fig_shorthand.png", dpi=200, facecolor="white",
                bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("saved fig_shorthand.png")


def fig_rope_lane() -> None:
    fig, ax = plt.subplots(figsize=(13.2, 6.8))
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 9.0)
    ax.set_axis_off()

    # 新发言者带两个问题
    ax.text(1.65, 6.4, "新发言者", fontsize=12.5, ha="center", color=DARK, fontweight="bold")
    box(ax, 0.5, 5.15, 2.3, 0.7, ORANGE, ORANGE_D, 2.2)
    ax.text(1.65, 5.5, "内容问题 $\\tilde{q}$", fontsize=12, ha="center", va="center")
    box(ax, 0.5, 3.85, 2.3, 0.7, ORANGE, ORANGE_D, 2.2)
    ax.text(1.65, 4.2, "时间问题 $q^R$", fontsize=12, ha="center", va="center")

    # 速记本（无时间概念）
    ax.text(5.65, 7.55, "速记本：只有内容，没有时间", fontsize=13, ha="center",
            color=GREEN_D, fontweight="bold")
    box(ax, 4.3, 1.6, 2.7, 5.5, GREEN_BG, GREEN_D, 2.0)
    for i in range(4):
        y = 6.1 - i * 1.15
        box(ax, 4.65, y, 2.0, 0.7, TEAL, TEAL_D)
        ax.text(5.65, y + 0.35, f"$c_{{{i + 1}}}$", fontsize=12, ha="center", va="center")
    arrow(ax, (2.9, 5.5), (4.32, 5.15))
    ax.text(3.6, 5.85, "对内容打分", fontsize=11, ha="center", color=DARK)

    # 时间旁路
    ax.text(8.65, 7.55, "时间旁路：单独一小栏", fontsize=13, ha="center",
            color=ORANGE_D, fontweight="bold")
    box(ax, 7.6, 1.6, 2.1, 5.5, "#FFF6E9", ORANGE_D, 2.0)
    for i in range(4):
        y = 6.1 - i * 1.15
        box(ax, 7.85, y, 1.6, 0.7, ORANGE, ORANGE_D)
        ax.text(8.65, y + 0.35, f"$k^R_{{{i + 1}}}$·64维", fontsize=10.5, ha="center", va="center")
    # 时间问题沿底部绕行进时间栏（折线，不穿速记本）
    ax.plot([2.9, 3.7, 3.7, 8.65], [4.2, 4.2, 1.1, 1.1], lw=1.7, color=DARK)
    arrow(ax, (8.65, 1.1), (8.65, 1.55))
    ax.text(5.3, 1.32, "对时间戳打分", fontsize=11, ha="center", color=DARK)

    # 汇合：速记本分数沿顶部绕行，时间栏分数直连
    box(ax, 10.6, 3.9, 1.9, 1.1, "white", DARK, 2.0)
    ax.text(11.55, 4.45, "两路分数\n相加", fontsize=11.5, ha="center", va="center")
    ax.plot([4.3, 3.6, 3.6, 11.55], [6.5, 6.5, 8.45, 8.45], lw=1.7, color=DARK)
    ax.plot([4.3], [6.5], marker="o", ms=5, color=DARK)
    arrow(ax, (11.55, 8.45), (11.55, 5.05))
    arrow(ax, (9.75, 4.45), (10.55, 4.45))

    ax.text(6.6, 0.3, "旋转的时间戳没法折进速记（正文有翻车实验）——那就让它走自己的小道",
            fontsize=12.5, ha="center", color=DARK)
    fig.savefig(FIGS / "fig_rope_lane.png", dpi=200, facecolor="white",
                bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("saved fig_rope_lane.png")


if __name__ == "__main__":
    setup()
    fig_shorthand()
    fig_rope_lane()
