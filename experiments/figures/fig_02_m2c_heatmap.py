"""Figure 2: per-model M2-canonical heatmap across the v6 80-cell sweep.

Three side-by-side panels, one per model family. Each panel is a 16 × 5
heatmap with rows = (direction, anchor-pair) and columns = focus layers.
The 80 cells per model are extracted by parsing the v6 sweep table from
`outputs/24_20260520_185537.log` (script 24, the pre-registered v6 run).

Cells passing the M2 gate (M2-canonical ≥ 0.65) cluster at the
`operator-after → operator-after` anchor pair across the focus layers
(L 4 in Gemma, L 4-L 10 in OLMo and Pythia). The figure visualises the
cluster directly: PASS cells are bright; the post-call anchors and the
F→N reverse direction are systematically darker.
"""

from __future__ import annotations

import os
import re

import numpy as np

from _shared import setup_matplotlib, save_figure, LOG_DIR


LOG_PATH = os.path.join(LOG_DIR, "24_20260520_185537.log")

ROW_RE = re.compile(
    r"v6\s+([NF])->([NF])\s+(\S+)->(\S+)\s+L\s*(\d+)\s+\|\s+"
    r"\d\.\d+\s+\|\s+\d\.\d+\s+\|\s+(\d\.\d+)\s+\|\s+(\d\.\d+)\s+\|"
)

MODEL_HEADERS = {
    "Gemma 2 9B": "MODEL: Gemma 2 9B",
    "OLMo 2 7B": "MODEL: OLMo 2 7B",
    "Pythia 6.9B-d": "MODEL: Pythia 6.9B-deduped",
}

MODEL_LAYERS = {
    "Gemma 2 9B": [2, 4, 8, 16, 17],
    "OLMo 2 7B": [4, 7, 10, 16, 24],
    "Pythia 6.9B-d": [4, 7, 10, 16, 24],
}

ANCHORS_NEUTRAL = ["opera", "sente"]
ANCHORS_FUNC = ["opera", "first", "close", "sente"]


def parse_v6_sweep(log_path: str):
    with open(log_path, "r") as f:
        text = f.read()

    blocks = {}
    for model, header in MODEL_HEADERS.items():
        idx = text.find(header)
        if idx < 0:
            continue
        next_idx = min(
            (text.find(h, idx + 1) for h in MODEL_HEADERS.values() if text.find(h, idx + 1) >= 0),
            default=len(text),
        )
        blocks[model] = text[idx:next_idx]

    per_model = {}
    for model, body in blocks.items():
        cells: dict[tuple[str, str, str, int], tuple[float, float]] = {}
        for m in ROW_RE.finditer(body):
            d_src, d_tgt, src_anchor, tgt_anchor, layer, m2c, m2a = m.groups()
            direction = f"{d_src}->{d_tgt}"
            cells[(direction, src_anchor, tgt_anchor, int(layer))] = (
                float(m2c), float(m2a),
            )
        per_model[model] = cells
    return per_model


def build_grid(cells, layers, model):
    """Return (M, row_labels) with rows = (direction, anchor_pair), cols = layers."""
    row_labels = []
    row_keys = []
    for direction, src_set, tgt_set in [
        ("N->F", ANCHORS_NEUTRAL, ANCHORS_FUNC),
        ("F->N", ANCHORS_FUNC, ANCHORS_NEUTRAL),
    ]:
        for src in src_set:
            for tgt in tgt_set:
                row_labels.append(f"{direction.replace('->','→')} {src}→{tgt}")
                row_keys.append((direction, src, tgt))

    M = np.full((len(row_keys), len(layers)), np.nan)
    for i, (direction, src, tgt) in enumerate(row_keys):
        for j, L in enumerate(layers):
            v = cells.get((direction, src, tgt, L))
            if v is not None:
                M[i, j] = v[0]
    return M, row_labels


def main():
    plt = setup_matplotlib()
    import matplotlib.colors as mcolors

    per_model = parse_v6_sweep(LOG_PATH)
    n_cells = sum(len(c) for c in per_model.values())
    print(f"  parsed {n_cells} cells across {len(per_model)} models")

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 6.0), sharey=True)
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    cmap = "viridis"

    for ax, (model, cells) in zip(axes, per_model.items()):
        layers = MODEL_LAYERS[model]
        M, row_labels = build_grid(cells, layers, model)

        im = ax.imshow(M, aspect="auto", cmap=cmap, norm=norm,
                       interpolation="nearest")

        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}",
                            ha="center", va="center",
                            fontsize=6.5,
                            color="white" if v < 0.55 else "black")

        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels([f"L{L}" for L in layers], fontsize=8.5)
        ax.set_xlabel("focus layer")
        ax.set_yticks(range(len(row_labels)))
        if ax is axes[0]:
            ax.set_yticklabels(row_labels, fontsize=7.5)
            ax.set_ylabel("direction × (source anchor → target anchor)")
        ax.set_title(model, fontsize=10.5)
        ax.axhline(7.5, color="white", linewidth=0.9, alpha=0.85)
        ax.text(len(layers) - 0.4, 3.5, "N→F",
                color="white", ha="right", va="center",
                fontsize=8.5, alpha=0.9)
        ax.text(len(layers) - 0.4, 11.5, "F→N",
                color="white", ha="right", va="center",
                fontsize=8.5, alpha=0.9)

    cbar = fig.colorbar(im, ax=axes, fraction=0.022, pad=0.025)
    cbar.set_label("M2-canonical (15-class; chance = 0.067; PASS ≥ 0.65)",
                   fontsize=9)
    cbar.ax.axhline(0.65, color="red", linewidth=1.0)

    # No fig.suptitle: LaTeX \caption{} carries the explanatory prose.
    save_figure(fig, "fig_02_m2c_heatmap")


if __name__ == "__main__":
    main()
