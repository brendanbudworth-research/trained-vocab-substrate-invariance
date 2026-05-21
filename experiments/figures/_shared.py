"""Shared utilities for figure generation.

All figure scripts in `experiments/figures/` import from this module. The
module deliberately keeps a small surface area (cache loading + probe training
+ matplotlib helpers) so that each figure script's body reads as a description
of the figure itself rather than as plumbing.

Caches loaded by this module are produced by upstream experiment scripts
(21, 22c, 22d, 23, 24, 25c). The expected on-disk layout is
`experiments/outputs/cache/<script_id>_<model_short>_<condition>_npc50_<vN>-<tag>.npz`.

Probes follow the same convention as `experiments/24_v6_canonical_expansion.py:m2_metrics`:
sklearn LogisticRegression with C=1.0, lbfgs, max_iter=5000, no class
balancing or scaling. Seeds are fixed at SEED = 1337 to match script 24.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

SEED = 1337

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
CACHE_DIR = os.path.join(REPO_ROOT, "experiments", "outputs", "cache")
LOG_DIR = os.path.join(REPO_ROOT, "experiments", "outputs")
OUT_DIR = os.path.join(THIS_DIR, "out")

os.makedirs(OUT_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(THIS_DIR, "_mpl_cache"))


# ---------------------------------------------------------------------------
# v6 canonical / invented set
# ---------------------------------------------------------------------------
V6_CANONICALS = [
    "and", "or", "not", "implies", "necessarily",
    "xor", "nand", "possibly", "always", "negate",
    "nor", "iff", "unless", "definitely", "unprovably",
]
V5_CANONICALS = V6_CANONICALS[:10]
V3_CANONICALS = V6_CANONICALS[:5]
V4_CANONICALS = V3_CANONICALS

UNARY_V6 = {"not", "necessarily", "possibly", "always", "negate", "definitely", "unprovably"}
BINARY_V6 = {c for c in V6_CANONICALS if c not in UNARY_V6}

CANONICAL_ARITY: dict[str, int] = {c: (1 if c in UNARY_V6 else 2) for c in V6_CANONICALS}

INVENTED_16 = [
    "bliq", "dren", "molex", "krev", "sond", "glin", "twiv", "fump",
    "vusp", "perph", "kelm", "zorf", "gleph", "drelth", "vrith", "nilph",
]
INVENTED_5 = ["bliq", "dren", "vusp", "molex", "perph"]
INVENTED_BINARY_8 = INVENTED_16[:8]
INVENTED_UNARY_8 = INVENTED_16[8:]

MODELS = [
    {"short": "Gemma_2_9B", "label": "Gemma 2 9B", "focus_layers": [2, 4, 8, 16, 17]},
    {"short": "OLMo_2_7B", "label": "OLMo 2 7B", "focus_layers": [4, 7, 10, 16, 24]},
    {"short": "Pythia_6.9B-deduped", "label": "Pythia 6.9B-d", "focus_layers": [4, 7, 10, 16, 24]},
]

ANCHORS_NEUTRAL = ["operator-after", "sentence-final"]
ANCHORS_FUNC = ["operator-after", "first-arg", "close-paren", "sentence-final"]

ANCHOR_SHORT = {
    "operator-after": "opera",
    "first-arg": "first",
    "close-paren": "close",
    "sentence-final": "sente",
}


# ---------------------------------------------------------------------------
# Cache loading (matches the on-disk format produced by script 24)
# ---------------------------------------------------------------------------
@dataclass
class V6Cache:
    canonical_X: np.ndarray  # (n_anchors, n_stim, n_layers, hidden)
    canonical_labels: np.ndarray  # (n_stim,) of str
    invented_X: np.ndarray
    invented_word_per_stim: np.ndarray
    anchor_names: list[str]
    n_layers: int

    def slice_canonical(self, anchor: str, layer: int, *, scope: str = "v6"):
        a_idx = self.anchor_names.index(anchor)
        X = self.canonical_X[a_idx, :, layer, :]
        y = self.canonical_labels.astype(str)
        keep = {"v3": V3_CANONICALS, "v4": V4_CANONICALS,
                "v5": V5_CANONICALS, "v6": V6_CANONICALS}[scope]
        mask = np.isin(y, keep)
        return X[mask], y[mask]

    def slice_invented(self, anchor: str, layer: int, *, scope: str = "v6"):
        a_idx = self.anchor_names.index(anchor)
        X = self.invented_X[a_idx, :, layer, :]
        w = self.invented_word_per_stim.astype(str)
        keep = INVENTED_5 if scope == "v3" else INVENTED_16
        mask = np.isin(w, keep)
        return X[mask], w[mask]


def load_v6_cache(model_short: str, condition: str) -> Optional[V6Cache]:
    """Load script 24's v6-expanded-canonical cache for a model + condition.

    Returns None if the cache is missing; figure scripts should treat this as
    a soft error and skip the affected panel rather than crash."""
    path = os.path.join(
        CACHE_DIR,
        f"24_{model_short}_{condition}_npc50_v6-expanded-canonical.npz",
    )
    if not os.path.exists(path):
        return None
    z = np.load(path, allow_pickle=False)
    cx = z["canonical_X"].astype(np.float32)
    return V6Cache(
        canonical_X=cx,
        canonical_labels=z["canonical_labels"],
        invented_X=z["invented_X"].astype(np.float32),
        invented_word_per_stim=z["invented_word_per_stim"],
        anchor_names=list(z["anchor_names"]),
        n_layers=cx.shape[2],
    )


# ---------------------------------------------------------------------------
# Probe (LogisticRegression, identical hyperparams to script 24)
# ---------------------------------------------------------------------------
def train_probe(X_train: np.ndarray, y_train: np.ndarray):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs").fit(X_train, y_train)


def m2_canonical(X_tr, y_tr, X_te, y_te) -> tuple[float, float]:
    """Returns (M2-canonical, M2-arity)."""
    clf = train_probe(X_tr, y_tr)
    preds = clf.predict(X_te)
    m2c = float(np.mean(preds == y_te))
    m2a = float(np.mean([
        CANONICAL_ARITY[str(t)] == CANONICAL_ARITY[str(p)]
        for t, p in zip(y_te, preds)
    ]))
    return m2c, m2a


# ---------------------------------------------------------------------------
# matplotlib helpers
# ---------------------------------------------------------------------------
def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 220,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "serif"],
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "lines.linewidth": 1.4,
        "pdf.fonttype": 42,
    })
    return plt


MODEL_PALETTE = {
    "Gemma 2 9B": "#1f77b4",
    "OLMo 2 7B": "#d62728",
    "Pythia 6.9B-d": "#2ca02c",
}


def save_figure(fig, name: str) -> str:
    pdf_path = os.path.join(OUT_DIR, f"{name}.pdf")
    png_path = os.path.join(OUT_DIR, f"{name}.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    print(f"  wrote {pdf_path}")
    print(f"  wrote {png_path}")
    return pdf_path


# ---------------------------------------------------------------------------
# Log parsing utilities (used by fig_04, fig_05)
# ---------------------------------------------------------------------------
def newest_log(pattern_prefix: str) -> Optional[str]:
    """Return the newest log under outputs/ whose basename starts with the prefix."""
    if not os.path.isdir(LOG_DIR):
        return None
    candidates = [
        os.path.join(LOG_DIR, f)
        for f in os.listdir(LOG_DIR)
        if f.startswith(pattern_prefix) and f.endswith(".log")
    ]
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]
