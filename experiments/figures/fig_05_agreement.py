"""Figure 5: 25b per-layer embedding-similarity vs probe agreement.

Three-panel line plot, one panel per model family. Each panel shows three
agreement curves across the model's L 0 + focus-layer set: `agree-all`
(unconstrained cosine argmax matches probe top), `agree-arity` (cosine
argmax restricted to canonicals of matching intended arity matches probe top),
and `arity-match` (cosine argmax has correct binary-vs-unary intended arity).

Chance baselines and the pre-registered "mechanism gap closed" threshold
(agree-arity ≥ 60%) are marked as horizontal dotted/solid lines.

The headline visual finding the figure must convey: L 0 agreement is at
floor (0-3%) for `agree-all` and `agree-arity` across all three models —
the operator-region attractor structure is constructed by intermediate-layer
processing, NOT inherited from token-embedding geometry. Mid-layer peaks
at L 4 (Gemma) and L 10 (OLMo, Pythia); late-layer L 24 collapses for
identity but `arity-match` stays elevated.

Numbers are extracted by hand from the per-model L0 + focus-layer block of
`outputs/25b_20260520_213935.log` (lines 201-206, 399-404, 597-602).
"""

from __future__ import annotations

from _shared import setup_matplotlib, save_figure, MODEL_PALETTE


PER_LAYER = {
    "Gemma 2 9B": {
        "layers": [0, 2, 4, 8, 16, 17],
        "agree_all":   [0.8, 12.1, 27.0,  6.2,  6.6,  6.2],
        "agree_arity": [0.0, 11.3, 18.4,  6.2,  8.6, 12.9],
        "arity_match": [16.4, 58.2, 66.4, 60.5, 43.8, 41.4],
        "focus_min": 2,
    },
    "OLMo 2 7B": {
        "layers": [0, 4, 7, 10, 16, 24],
        "agree_all":   [0.0, 32.8, 27.7, 41.0, 29.7,  2.3],
        "agree_arity": [0.0, 33.2, 27.7, 27.0, 15.2,  4.3],
        "arity_match": [1.6, 43.8, 55.9, 77.0, 82.4, 73.0],
        "focus_min": 4,
    },
    "Pythia 6.9B-d": {
        "layers": [0, 4, 7, 10, 16, 24],
        "agree_all":   [0.0, 26.6, 22.7, 38.7, 28.9,  3.1],
        "agree_arity": [0.0, 25.8, 16.0, 26.6, 24.6,  2.3],
        "arity_match": [2.3, 47.7, 55.9, 69.1, 64.8, 63.3],
        "focus_min": 4,
    },
}

CHANCE_ALL = 6.7
CHANCE_ARITY = 13.0
CHANCE_ARITY_MATCH = 53.0
GAP_CLOSED_THRESHOLD = 60.0


def main():
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.4), sharey=True)

    for ax, (model, data) in zip(axes, PER_LAYER.items()):
        layers = data["layers"]
        color = MODEL_PALETTE[model]

        ax.plot(layers, data["agree_all"], marker="o", color="#4477AA",
                label="agree-all", zorder=3)
        ax.plot(layers, data["agree_arity"], marker="s", color="#EE6677",
                label="agree-arity", zorder=3)
        ax.plot(layers, data["arity_match"], marker="^", color="#228833",
                label="arity-match", zorder=3)

        ax.axhline(CHANCE_ALL, color="#4477AA",
                   linestyle=":", linewidth=0.7, zorder=1)
        ax.axhline(CHANCE_ARITY, color="#EE6677",
                   linestyle=":", linewidth=0.7, zorder=1)
        ax.axhline(CHANCE_ARITY_MATCH, color="#228833",
                   linestyle=":", linewidth=0.7, zorder=1)

        ax.axhline(GAP_CLOSED_THRESHOLD, color="black",
                   linestyle="-", linewidth=0.6, alpha=0.55, zorder=1)
        ax.text(layers[-1] - 0.5, GAP_CLOSED_THRESHOLD + 1.5,
                "pre-reg gap-closed (60%)",
                ha="right", va="bottom", fontsize=7, color="black", alpha=0.7)

        ax.axvspan(-0.5, 0.5, color="grey", alpha=0.10, zorder=0)
        ax.text(0, 92, "L 0\nfloor", ha="center", va="top",
                fontsize=7.5, color="dimgrey")

        ax.set_xlim(-0.7, max(layers) + 0.7)
        ax.set_ylim(-3, 95)
        ax.set_xticks(layers)
        ax.set_xticklabels([f"L{L}" for L in layers], fontsize=8)
        ax.set_xlabel("layer")
        ax.set_title(model, fontsize=10, color=color)

    axes[0].set_ylabel("agreement (%)")
    axes[-1].legend(loc="lower left", fontsize=8, bbox_to_anchor=(0.02, 0.02))

    # No fig.suptitle: LaTeX \caption{} carries the explanatory prose.
    fig.tight_layout()
    save_figure(fig, "fig_05_agreement")


if __name__ == "__main__":
    main()
