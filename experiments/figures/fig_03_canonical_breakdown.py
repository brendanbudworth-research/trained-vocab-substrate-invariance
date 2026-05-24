"""Figure 3: per-canonical breakdown of invented mass at OLMo §3.7.9 cell.

Four-panel barplot showing the % of invented-stimulus predictions landing
on each canonical at the OLMo 2 7B `N→F sente→close L 10` cell, tracked
across the four pre-registered scopes (v3 → v6). The cell is the same one
documented in Table 3 of the paper.

The figure recomputes the invented-mass distribution live from the
script 24 OLMo v6 cache (`outputs/cache/24_OLMo_2_7B_*.npz`) by subsetting
the canonical readout to each scope's canonical set and predicting on
the v3 (5-word) or v4+ (16-word) invented set. This is the exact same
computation script 24 does in its sweep tables (lines 978-1401), so the
figure's numbers match the §4.2 / Table 3 verdict reading.

Headline visual finding: at v3 the invented mass appears arity-respecting
(intended-binary words land on `and`, intended-unary on `not`). At v5 / v6
with `nand` / `unprovably` in the readout, 100% of invented mass collapses
to a single attractor — the v3 mass distribution was a coincidence between
the model's default-canonical attractor and the intended arities of the
small 5-word invented set, exactly as the lucky-default detector predicts.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from _shared import (
    setup_matplotlib, save_figure, load_v6_cache,
    train_probe, V6_CANONICALS, V5_CANONICALS, V4_CANONICALS, V3_CANONICALS,
    INVENTED_16, INVENTED_5, UNARY_V6,
)


CELL_MODEL_SHORT = "OLMo_2_7B"
CELL_TRAIN_ANCHOR = "sentence-final"
CELL_TEST_ANCHOR = "close-paren"
CELL_LAYER = 10
SCOPE_LIST = ["v3", "v4", "v5", "v6"]

SCOPE_CANONICALS = {
    "v3": V3_CANONICALS,
    "v4": V4_CANONICALS,
    "v5": V5_CANONICALS,
    "v6": V6_CANONICALS,
}
SCOPE_INVENTED = {
    "v3": INVENTED_5,
    "v4": INVENTED_16,
    "v5": INVENTED_16,
    "v6": INVENTED_16,
}


def compute_breakdown(scope: str):
    """Return (canonical_list, breakdown_pct, m4b_arity_agree)."""
    neutral = load_v6_cache(CELL_MODEL_SHORT, "NEUTRAL")
    funcpfx = load_v6_cache(CELL_MODEL_SHORT, "FUNC-PFX")
    assert neutral is not None and funcpfx is not None, "OLMo v6 caches missing"

    X_tr, y_tr = neutral.slice_canonical(
        CELL_TRAIN_ANCHOR, CELL_LAYER, scope=scope,
    )
    X_inv, w_inv = funcpfx.slice_invented(
        CELL_TEST_ANCHOR, CELL_LAYER, scope=scope,
    )

    clf = train_probe(X_tr, y_tr)
    preds = clf.predict(X_inv)
    canonicals = SCOPE_CANONICALS[scope]
    breakdown = {c: float(np.mean(preds == c)) for c in canonicals}

    arity_agree = 0
    for w, p in zip(w_inv, preds):
        intended_arity = 2 if w in INVENTED_16[:8] else 1
        pred_arity = 1 if str(p) in UNARY_V6 else 2
        if intended_arity == pred_arity:
            arity_agree += 1
    m4b = arity_agree / max(len(preds), 1)
    return canonicals, breakdown, m4b


def main():
    plt = setup_matplotlib()
    # 2x2 grid so each panel gets ~2x the linear scale of the previous 1x4
    # strip. v6 in particular has 15 canonicals on its x-axis, which were
    # illegible at fontsize=7 in the strip layout.  Kept wider than tall so
    # \includegraphics[width=\linewidth] doesn't overflow a page slot.
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.0), sharey=True)
    axes_flat = axes.flatten()

    color_unary = "#1f77b4"
    color_binary = "#d62728"

    for ax, scope in zip(axes_flat, SCOPE_LIST):
        canonicals, breakdown, m4b = compute_breakdown(scope)
        ys = [breakdown[c] * 100 for c in canonicals]
        colors = [color_unary if c in UNARY_V6 else color_binary
                  for c in canonicals]
        bars = ax.bar(range(len(canonicals)), ys, color=colors,
                      edgecolor="black", linewidth=0.5)

        # Bar-label font shrinks with canonical-set size so v5/v6's narrow
        # bars don't get text wider than the bar.
        n = len(canonicals)
        label_fs = 10 if n <= 5 else (9 if n <= 10 else 8)

        for bar, c, v in zip(bars, canonicals, ys):
            if v >= 1:
                # Uniform placement: percentage always sits just above the
                # bar in plain black text; canonical name is on the x-axis.
                ax.text(bar.get_x() + bar.get_width() / 2, v + 2,
                        f"{v:.0f}%", ha="center", va="bottom",
                        fontsize=label_fs, color="black",
                        fontweight="bold")

        ax.set_xticks(range(len(canonicals)))
        ax.set_xticklabels(canonicals, rotation=45, ha="right", fontsize=10)
        ax.set_ylim(0, 122)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=10)
        n_words = len(SCOPE_INVENTED[scope])
        n_canon = len(canonicals)
        ax.set_title(f"{scope}  ({n_canon} canonicals × {n_words} invented)   "
                     f"M4b = {m4b * 100:.0f}%",
                     fontsize=12, pad=6)

    # Shared y-label on the leftmost panels of each row only.
    axes[0, 0].set_ylabel("share of invented predictions", fontsize=11)
    axes[1, 0].set_ylabel("share of invented predictions", fontsize=11)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color_unary, label="unary canonical"),
        plt.Rectangle((0, 0), 1, 1, color=color_binary, label="binary canonical"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.01), fontsize=11, frameon=False)

    # No fig.suptitle: LaTeX \caption{} carries the explanatory prose.
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save_figure(fig, "fig_03_canonical_breakdown")


if __name__ == "__main__":
    main()
