"""Figure 4: causal patching cross-cell synthesis.

Five-panel barplot, one per (source, target, layer) cell tested in script 25a.
Each panel shows the mean behavioural ΔKL on the `ref_not` and `ref_and`
axes for the four patch conditions (BASELINE = 0 by construction, the
"PATCH_c" condition, and the "RANDOM_NORM" norm-matched negative control).
A patch is causally arity-respecting when the targeted PATCH bar is large
positive AND clearly exceeds the RANDOM_NORM control on the same reference
axis. The figure visualises the one PASS cell, the one WEAK PASS cell,
and the three FAIL cells in side-by-side comparison.

Numbers are the canonical Table 5 values from paper.md §4.5; they are
hard-coded here so the figure remains the single source of truth even if
log files are pruned. Log files of record:
  - 25a_20260520_211030.log  (original 3 cells, v2 corrected probe-causality)
  - 25a_20260521_085745.log  (the two reviewer-extra cells)
"""

from __future__ import annotations

from _shared import setup_matplotlib, save_figure


# (cell_short, source_anchor, target_anchor, layer,
#  ΔKL_not, ΔKL_and, RND_not, RND_and, verdict)
CELLS = [
    ("Gemma 2 9B\nopera→close L 2",
     "opera-after", "close-paren", 2,
     +0.048, +0.038, +0.001, +0.003, "CLEAN PASS"),
    ("Gemma 2 9B\nopera→opera L 4 (extra)",
     "opera-after", "opera-after", 4,
     +0.033, +0.027, +0.017, +0.021, "WEAK PASS"),
    ("Gemma 2 9B\nsente→close L 2",
     "sentence-final", "close-paren", 2,
     -0.020, -0.012, +0.011, +0.009, "FAIL"),
    ("OLMo 2 7B\nsente→close L 10",
     "sentence-final", "close-paren", 10,
     -0.012, -0.017, +0.019, +0.012, "FAIL"),
    ("OLMo 2 7B\nopera→close L 10 (extra)",
     "opera-after", "close-paren", 10,
     +0.013, +0.003, +0.023, +0.016, "FAIL"),
]

VERDICT_COLOR = {
    "CLEAN PASS": "#1b9e77",
    "WEAK PASS":  "#e6ab02",
    "FAIL":       "#888888",
}


def main():
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 5, figsize=(13.5, 3.0), sharey=True)

    bar_labels = [r"$\Delta\mathrm{KL}_{\,\mathrm{not}}$",
                  r"$\Delta\mathrm{KL}_{\,\mathrm{and}}$"]
    bar_xs = [0.0, 1.0]
    bar_width = 0.35

    for ax, cell in zip(axes, CELLS):
        title, src, tgt, layer, dnot, dand, rnot, rand_, verdict = cell
        color = VERDICT_COLOR[verdict]
        rnd_color = "#bdbdbd"

        ax.bar([x - bar_width / 2 for x in bar_xs],
               [dnot, dand], width=bar_width,
               color=color, edgecolor="black", linewidth=0.5,
               label="targeted PATCH")
        ax.bar([x + bar_width / 2 for x in bar_xs],
               [rnot, rand_], width=bar_width,
               color=rnd_color, edgecolor="black", linewidth=0.5,
               hatch="///", label="RANDOM_NORM")

        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xticks(bar_xs)
        ax.set_xticklabels(bar_labels, fontsize=8.5)
        ax.set_title(title, fontsize=9, pad=4)
        ax.set_ylim(-0.04, 0.06)

        ax.text(0.5, 0.97, verdict, transform=ax.transAxes,
                ha="center", va="top",
                fontsize=8.5, fontweight="bold", color=color,
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec=color, lw=0.8))

    axes[0].set_ylabel(r"$\Delta\mathrm{KL}$ "
                       r"(positive $=$ behaviour pulled toward $c$)")
    axes[-1].legend(loc="lower right", fontsize=7.5)

    fig.suptitle(
        "Figure 4. Causal patching across 5 (source, target, layer) cells. "
        "Targeted PATCH (coloured) vs. norm-matched RANDOM_NORM control (grey, "
        "hatched), per reference axis. A cell is causally arity-respecting only "
        "when targeted ΔKL clearly exceeds RANDOM_NORM on the same axis. "
        "(extra) = reviewer-round-1 follow-up cell.",
        fontsize=8.5, y=1.04,
    )
    fig.tight_layout()
    save_figure(fig, "fig_04_causal_patching")


if __name__ == "__main__":
    main()
