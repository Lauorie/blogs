"""Draw fig3 (redundant K/V recomputation across steps 8/9/10) with matplotlib.

Generative models kept miscounting rows, so this schematic is drawn
programmatically: counts are exact by construction.
Usage: python draw_fig3.py  (writes ../figures/fig3_redundancy.png)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

TEAL = "#34A08C"
PURPLE = "#B8A2E0"
RED_EDGE = "#D98880"
RED_TEXT = "#8F2F28"
RED_FILL = "#FDF1F0"
GREEN = "#1F9D55"
DARK = "#3A4450"
OUT = Path(__file__).resolve().parent.parent / "figures" / "fig3_redundancy.png"

ROW_H, ROW_GAP, BOX_W, BOX_GAP = 0.62, 0.16, 1.05, 0.14
ROWS_W = 2 * BOX_W + BOX_GAP          # width of one K/V row
LABEL_W = 2.45                        # empty zone right of the rows for labels


def rounded(ax: plt.Axes, x: float, y: float, w: float, h: float, fc: str,
            ec: str, lw: float = 1.6, hatch: str | None = None) -> None:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=lw, hatch=hatch))


def panel(ax: plt.Axes, x0: float, title: str, n_rows: int, top: float) -> None:
    """One step panel: n_rows K/V rows, all but the last hatched as recomputed."""
    pw = 0.35 + ROWS_W + LABEL_W + 0.25
    ph = 10 * (ROW_H + ROW_GAP) + 1.15  # sized for the tallest panel (10 rows)
    rounded(ax, x0, top - ph, pw, ph, "white", DARK, 2.0)
    ax.text(x0 + pw / 2, top - 0.42, title, ha="center", va="center",
            fontsize=15, color=DARK, fontweight="bold")

    bx = x0 + 0.35
    n_old = n_rows - 1
    # Hatched overlay spanning the recomputed rows plus the label zone.
    oy_top = top - 0.78
    oy_bot = oy_top - n_old * (ROW_H + ROW_GAP) + ROW_GAP - 0.08
    rounded(ax, bx - 0.12, oy_bot, ROWS_W + LABEL_W + 0.12, oy_top - oy_bot,
            RED_FILL, RED_EDGE, 1.3, hatch="//")

    for i in range(n_rows):
        ry = top - 0.86 - i * (ROW_H + ROW_GAP) - ROW_H
        rounded(ax, bx, ry, BOX_W, ROW_H, TEAL, DARK)
        rounded(ax, bx + BOX_W + BOX_GAP, ry, BOX_W, ROW_H, PURPLE, DARK)
        for dx, label in ((0, "K"), (BOX_W + BOX_GAP, "V")):
            ax.text(bx + dx + BOX_W / 2, ry + ROW_H / 2, label, ha="center",
                    va="center", fontsize=13, color="#20313d")

    # Overlay label centred in the empty zone right of the rows.
    ax.text(bx + ROWS_W + 0.18 + (LABEL_W - 0.3) / 2, (oy_top + oy_bot) / 2,
            "recomputed -\nidentical result", ha="center", va="center",
            fontsize=11.5, color=RED_TEXT,
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=RED_EDGE, lw=1.1))
    # Green check + "new" beside the last row.
    ly = top - 0.86 - n_old * (ROW_H + ROW_GAP) - ROW_H / 2
    ax.text(bx + ROWS_W + 0.25, ly, "✓ new", ha="left", va="center",
            fontsize=12.5, color=GREEN, fontweight="bold")


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "hatch.linewidth": 0.9})
    fig, ax = plt.subplots(figsize=(13.76, 7.74))
    ax.set_xlim(0, 17.7)
    ax.set_ylim(0, 10.7)
    ax.set_axis_off()

    top = 10.05
    for i, (title, rows) in enumerate([("step 8", 8), ("step 9", 9), ("step 10", 10)]):
        panel(ax, 0.5 + i * 5.75, title, rows, top)

    ax.text(8.85, 0.42, "Old tokens always produce the same K and V - "
            "yet they are recomputed at every step. O(n²) wasted work.",
            ha="center", va="center", fontsize=13.5, color=DARK)

    fig.savefig(OUT, dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
