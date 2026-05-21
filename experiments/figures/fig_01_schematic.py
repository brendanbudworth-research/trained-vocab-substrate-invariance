"""Figure 1: schematic of the substrate-invariance setup.

A purely illustrative diagram (no data). The figure shows:

  - Left column:  the two surface notations the paper compares (NEUTRAL
                  metalinguistic frame and FUNC-PFX programming-style frame)
                  applied to a canonical operator (`and`) and an invented
                  operator (`bliq`).
  - Middle:       both stimulus families pass through the same LM forward
                  pass; we extract the residual stream at a single
                  (anchor, layer) coordinate.
  - Right column: a linear probe trained on NEUTRAL canonical activations
                  is evaluated on (a) FUNC-PFX canonical activations,
                  yielding Fact 1's measurement (M2-canonical: cross-notation
                  canonical-identity transfer), and (b) FUNC-PFX invented
                  activations, yielding Fact 2's measurement (M4b:
                  intended-arity agreement on invented words).

Numbers / verdicts come from the paper's Tables 1 and 4; rendered as
illustrative annotations rather than measurements.
"""

from __future__ import annotations

import matplotlib.patches as mpatches

from _shared import setup_matplotlib, save_figure


def draw_box(ax, x, y, w, h, label, *, facecolor="white", edgecolor="black",
             fontsize=9, fontweight="normal", lw=0.9, sublabel=None):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw,
    )
    ax.add_patch(rect)
    if sublabel is not None:
        ax.text(x + w / 2, y + h / 2 + 0.018, label,
                ha="center", va="center", fontsize=fontsize,
                fontweight=fontweight)
        ax.text(x + w / 2, y + h / 2 - 0.018, sublabel,
                ha="center", va="center", fontsize=fontsize - 1.5,
                style="italic", color="#444")
    else:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fontsize, fontweight=fontweight)


def arrow(ax, x0, y0, x1, y1, *, color="black", lw=1.0,
          arrowstyle="-|>"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=arrowstyle, color=color,
                                lw=lw, shrinkA=2, shrinkB=2))


def main():
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_aspect("auto")
    ax.axis("off")

    # ---- LEFT COLUMN: stimuli ------------------------------------------
    ax.text(0.155, 0.96, "Stimuli (paired by structure)",
            ha="center", va="center", fontsize=10.5, fontweight="bold")

    ax.text(0.06, 0.88, "NEUTRAL\nmetalinguistic", ha="center", va="center",
            fontsize=9, color="#1f6f43")
    ax.text(0.255, 0.88, "FUNC-PFX\nprogramming-style", ha="center",
            va="center", fontsize=9, color="#9a4a00")

    # canonical row
    ax.text(0.005, 0.78, "canonical\nop $c=$ `and`", ha="left", va="center",
            fontsize=8.5, style="italic")
    draw_box(ax, 0.005, 0.65, 0.13, 0.075,
             "Consider the word\n`and` in this sentence.",
             facecolor="#e8f5e9", fontsize=8.2)
    draw_box(ax, 0.165, 0.65, 0.18, 0.075,
             "The function `and(p, q)`\nevaluates to true.",
             facecolor="#fbe9d7", fontsize=8.2)

    # invented row
    ax.text(0.005, 0.55, "invented\nop $w=$ `bliq`", ha="left", va="center",
            fontsize=8.5, style="italic")
    draw_box(ax, 0.005, 0.42, 0.13, 0.075,
             "Consider the word\n`bliq` in this sentence.",
             facecolor="#e8f5e9", fontsize=8.2)
    draw_box(ax, 0.165, 0.42, 0.18, 0.075,
             "The function `bliq(p, q)`\nevaluates to true.",
             facecolor="#fbe9d7", fontsize=8.2)

    # arrow into LM
    arrow(ax, 0.355, 0.69, 0.435, 0.66, color="black", lw=1.2)
    arrow(ax, 0.355, 0.46, 0.435, 0.49, color="black", lw=1.2)

    # ---- MIDDLE: LM forward pass ---------------------------------------
    ax.text(0.515, 0.96, "Frozen base LM",
            ha="center", va="center", fontsize=10.5, fontweight="bold")
    ax.text(0.515, 0.92,
            "Gemma 2 9B / OLMo 2 7B / Pythia 6.9B-d",
            ha="center", va="center", fontsize=8.5, color="#333")
    draw_box(ax, 0.435, 0.42, 0.16, 0.32,
             "",
             facecolor="#f5f5f5", fontsize=9, lw=1.3)
    ax.text(0.515, 0.68, "LM forward pass", ha="center", va="center",
            fontsize=10, fontweight="bold")
    ax.text(0.515, 0.60, "extract residual stream\nat (anchor, layer)",
            ha="center", va="center", fontsize=8.8)
    ax.text(0.515, 0.50, "e.g. operator-after, L 4",
            ha="center", va="center", fontsize=8.5, style="italic",
            color="#555")

    # arrow out of LM, splitting into canonical / invented paths
    arrow(ax, 0.595, 0.66, 0.66, 0.7, lw=1.2)
    arrow(ax, 0.595, 0.5, 0.66, 0.46, lw=1.2)

    # ---- RIGHT COLUMN: probes and measurements -------------------------
    ax.text(0.82, 0.96, "Linear-probe readout (M-battery)",
            ha="center", va="center", fontsize=10.5, fontweight="bold")

    rect_f1 = mpatches.FancyBboxPatch(
        (0.66, 0.66), 0.33, 0.13,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor="#e3f2fd", edgecolor="#1565c0", linewidth=1.3,
    )
    ax.add_patch(rect_f1)
    ax.text(0.825, 0.765, "Fact 1  /  M2-canonical",
            ha="center", va="center", fontsize=10.5, fontweight="bold")
    ax.text(0.825, 0.728, "train: NEUT canonical  •  test: FUNC canonical",
            ha="center", va="center", fontsize=7.8, style="italic",
            color="#444")
    ax.text(0.825, 0.685, "= 1.000  [1.000, 1.000]",
            ha="center", va="center", fontsize=10, color="#0b5394",
            fontweight="bold")

    rect_f2 = mpatches.FancyBboxPatch(
        (0.66, 0.42), 0.33, 0.13,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor="#fdecea", edgecolor="#b00020", linewidth=1.3,
    )
    ax.add_patch(rect_f2)
    ax.text(0.825, 0.525, "Fact 2  /  M4b",
            ha="center", va="center", fontsize=10.5, fontweight="bold")
    ax.text(0.825, 0.488, "train: NEUT canonical  •  test: FUNC invented",
            ha="center", va="center", fontsize=7.8, style="italic",
            color="#444")
    ax.text(0.825, 0.445,
            "= 0.50  (chance; canonical-set expansion)",
            ha="center", va="center", fontsize=10, color="#9a1a1a",
            fontweight="bold")

    # ---- BOTTOM strip: methodological battery --------------------------
    ax.text(0.5, 0.33, "Methodological battery (§3.4)",
            ha="center", va="center", fontsize=10, fontweight="bold")

    battery = [
        ("M1", "within-condition\n5-fold CV"),
        ("M2-canonical", "cross-notation\n1-of-K identity"),
        ("M2-arity", "cross-notation\nbinary-vs-unary"),
        ("M4b", "invented intended-arity\nagreement"),
        ("M4c / pwmin", "lucky-default\ndetector"),
    ]
    bx0 = 0.04
    bw = 0.184
    bh = 0.13
    for i, (name, sub) in enumerate(battery):
        x = bx0 + i * (bw + 0.005)
        draw_box(ax, x, 0.16, bw, bh, name, sublabel=sub,
                 facecolor="#fafafa", fontsize=9, fontweight="bold")

    # ---- BOTTOM annotation ---------------------------------------------
    ax.text(0.5, 0.07,
            "All three model families return Fact 1 = ceiling and Fact 2 = chance "
            "at the principal operator-after L 4 cell under the v6 scope (15 "
            "canonicals × 16 invented). The §4.1.1 content-word control extends "
            "Fact 1 to in-vocabulary content words (trained-vocabulary "
            "substrate-invariance, not specifically operator-class).",
            ha="center", va="center", fontsize=8.5, style="italic", color="#333")

    ax.text(0.5, 0.02,
            "Figure 1. Schematic of the substrate-invariance setup. "
            "Notations / probes / measurements at a glance.",
            ha="center", va="center", fontsize=9, color="#222")

    save_figure(fig, "fig_01_schematic")


if __name__ == "__main__":
    main()
