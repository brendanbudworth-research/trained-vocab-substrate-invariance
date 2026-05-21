"""Script 25b - embedding-similarity probe for the v6 default mechanism.

Purpose. The §3.7.16 pre-registered v6 disentanglement test rejected all
three single-axis readings of the default-to-rarest-canonical mechanism
(P_FREQ, P_SUBWORD, P_INTERACTION) in all three model families. The
mechanism appears to be a model-specific mixture of training-corpus
frequency, subword shape, and what we tentatively called *contextual
semantic neighborhood*. This script tests the semantic-neighborhood
component directly: at each v6 sweep cell, does cosine similarity
between the mean invented-word activation and the mean canonical
activation (at the same condition, anchor, and layer as the probe sees)
predict the probe's empirical per-word top-canonical?

The hypothesis closing the §3.7.16 mechanism gap: each invented word w's
default attractor c_probe(w) is the canonical that is geometrically
closest to w in residual stream space at the focus layer, controlled for
arity. We test two variants:

  * all_argmax(w)  = argmax over all 15 v6 canonicals of cos(mu_w, mu_c)
  * arity_argmax(w) = argmax over canonicals c whose CANONICAL_ARITY[c]
                      matches CANONICAL_ARITY[w_to_canonical[w]] (the
                      intended arity), of cos(mu_w, mu_c)

Per-cell agreement scores (binary "does the heuristic predict the probe?"
averaged over the 16 invented words) tell us how much of the probe's
per-word routing is explained by raw geometric proximity. If agreement
is high (>= 0.6) at the cells where the v6 sweep produces a distributed
default (M4c < 0.7), the multi-factor mechanism reduces to "softmax over
arity-conditioned semantic neighborhood" and the §3.7.16 mechanism gap
is closed. If agreement is at chance (~ 1/15 = 6.7% for all_argmax,
~ 1/n_intended_arity for arity_argmax), the routing mechanism is
*not* a simple cosine-similarity attractor and the semantic-neighborhood
factor is more complex than mean-pooled-residual-cosine.

Methodology summary:
  * Load v6 carryover NPZ caches from script 24 (no model loads needed).
  * Enumerate the same 80 v6 sweep cells per model as script 24 v6 scope.
  * For each cell, recompute the probe's per-word top-canonical (mirrors
    script 24's m4_breakdown exactly) AND the cosine-similarity
    per-word top-canonical at the (target_cond, target_anchor, layer)
    coordinate where the probe makes its predictions.
  * Aggregate by direction (N->F, F->N), by layer, by anchor pair.
  * Bonus 1: same analysis at the embedding layer (layer 0) to test
    the script-14 "H1 is constructed during forward pass, not inherited
    from L0 embeddings" finding under v6.
  * Bonus 2: softmax-over-similarities fit per cell - find best
    temperature tau >= 0 such that softmax(sim/tau) per word best
    matches the probe's empirical per-word top-canonical distribution
    in KL. Report best-fit tau and KL.

Cache-only. Estimated runtime ~5-12 min total (probe train is the
dominant cost; 240 cells x ~3 sec each).

Tees output to outputs/25b_<ts>.log. Env flags:
  MODEL_FILTER=<substring>  -- only run models whose short_name matches
  L0_ONLY=1                 -- skip the per-layer sweep, only do L0
  SKIP_L0=1                 -- skip the L0 analysis
  N_BOOTSTRAP=<int>         -- bootstrap iterations for cross-cell
                               agreement CIs (default 200)
  SCOPE=<v3|v4|v5|v6>       -- override the scope used to sub-select
                               the canonical / invented sets (default v6)

See paper_notes.md §3.7.16 / §3.7.17 / §6 (script 25b priority block)
for context.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import importlib.machinery
import importlib.util
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression


# ==============================================================================
# Tee logging (mirror 24/25a)
# ==============================================================================
class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        n = 0
        for s in self._streams:
            n = s.write(data)
            s.flush()
        return n

    def flush(self) -> None:
        for s in self._streams:
            s.flush()


def _setup_logging() -> Optional[str]:
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    out_dir = os.path.join(script_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(out_dir, f"25b_{ts}.log")
    log_file = open(log_path, "w", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)
    print(f"[logging] tee'ing all output to {log_path}")
    return log_path


# ==============================================================================
# Constants  (mirror script 24 v6 exactly)
# ==============================================================================
SEED = 17
N_PER_CLASS = 50
STIMULUS_VERSION = "v6-expanded-canonical"
SCOPE_NAME = os.environ.get("SCOPE", "v6").strip().lower()
MODEL_FILTER = os.environ.get("MODEL_FILTER", "").strip().lower()
L0_ONLY = bool(os.environ.get("L0_ONLY"))
SKIP_L0 = bool(os.environ.get("SKIP_L0"))
N_BOOTSTRAP = int(os.environ.get("N_BOOTSTRAP", "200"))

CANONICALS_V6 = [
    "and", "or", "implies", "xor", "nand",
    "not", "necessarily", "possibly", "always", "negate",
    "nor", "iff", "unless", "definitely", "unprovably",
]
CANONICAL_ARITY = {
    "and": 2, "or": 2, "implies": 2, "xor": 2, "nand": 2,
    "nor": 2, "iff": 2, "unless": 2,
    "not": 1, "necessarily": 1, "possibly": 1, "always": 1, "negate": 1,
    "definitely": 1, "unprovably": 1,
}
ORIGINAL_5_CANONICALS = ["and", "or", "implies", "not", "necessarily"]
CANONICALS_V5 = [
    "and", "or", "implies", "xor", "nand",
    "not", "necessarily", "possibly", "always", "negate",
]
INVENTED_16 = [
    "bliq", "dren", "molex", "krev", "sond", "glin", "twiv", "fump",
    "vusp", "perph", "kelm", "zorf", "gleph", "drelth", "vrith", "nilph",
]
INVENTED_5 = ["bliq", "dren", "molex", "vusp", "perph"]
W_TO_CANONICAL_16 = {
    "bliq": "and", "dren": "or", "molex": "implies",
    "krev": "and", "sond": "or", "glin": "implies",
    "twiv": "and", "fump": "or",
    "vusp": "not", "perph": "necessarily",
    "kelm": "not", "zorf": "not",
    "gleph": "necessarily", "drelth": "necessarily",
    "vrith": "not", "nilph": "necessarily",
}

ANCHORS_FUNC_PFX = ["operator-after", "first-arg", "close-paren", "sentence-final"]
ANCHORS_NEUTRAL = ["operator-after", "sentence-final"]


@dataclass
class Scope:
    name: str
    canonicals: list[str]
    invented_set: list[str]
    w_to_canonical: dict[str, str]


SCOPES = {
    "v3": Scope("v3", ORIGINAL_5_CANONICALS, INVENTED_5,
                {w: W_TO_CANONICAL_16[w] for w in INVENTED_5}),
    "v4": Scope("v4", ORIGINAL_5_CANONICALS, INVENTED_16, W_TO_CANONICAL_16),
    "v5": Scope("v5", CANONICALS_V5, INVENTED_16, W_TO_CANONICAL_16),
    "v6": Scope("v6", CANONICALS_V6, INVENTED_16, W_TO_CANONICAL_16),
}
if SCOPE_NAME not in SCOPES:
    raise ValueError(f"unknown SCOPE={SCOPE_NAME!r}; expected one of {list(SCOPES)}")
SCOPE = SCOPES[SCOPE_NAME]


# ==============================================================================
# Model specs (mirror script 24)
# ==============================================================================
@dataclass
class ModelSpec:
    short_name: str
    focus_layers: list[int]


MODEL_SPECS: list[ModelSpec] = [
    ModelSpec("Gemma 2 9B", focus_layers=[2, 4, 8, 16, 17]),
    ModelSpec("OLMo 2 7B", focus_layers=[4, 7, 10, 16, 24]),
    ModelSpec("Pythia 6.9B-deduped", focus_layers=[4, 7, 10, 16, 24]),
]


# ==============================================================================
# Cache loading (read-only; mirrors script 25a's load path exactly)
# ==============================================================================
@dataclass
class ConditionV6:
    canonical_X: np.ndarray            # (n_anchors, n_canon_stim, n_layers, dim)
    canonical_labels: np.ndarray       # (n_canon_stim,)
    invented_X: np.ndarray             # (n_anchors, n_inv_stim, n_layers, dim)
    invented_word_per_stim: np.ndarray # (n_inv_stim,)
    anchor_names: list[str]


def _cache_path(model_short_name: str, condition_name: str) -> str:
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "outputs", "cache")
    slug = model_short_name.replace(" ", "_")
    return os.path.join(
        base,
        f"24_{slug}_{condition_name}_npc{N_PER_CLASS}_v6-expanded-canonical.npz",
    )


def _load_carryover_cache(
    model_short_name: str, condition_name: str,
) -> Optional[ConditionV6]:
    path = _cache_path(model_short_name, condition_name)
    if not os.path.exists(path):
        print(f"  [cache] MISS: {path}")
        return None
    try:
        z = np.load(path, allow_pickle=False)
        if str(z["meta_stimulus_version"][0]) != STIMULUS_VERSION:
            print(f"  [cache] stimulus_version mismatch in {path}")
            return None
        if list(z["canonical_list"]) != CANONICALS_V6:
            print(f"  [cache] canonical_list mismatch in {path}")
            return None
        if list(z["invented_word_list"]) != INVENTED_16:
            print(f"  [cache] invented_word_list mismatch in {path}")
            return None
        cond = ConditionV6(
            canonical_X=z["canonical_X"].astype(np.float32),
            canonical_labels=z["canonical_labels"],
            invented_X=z["invented_X"].astype(np.float32),
            invented_word_per_stim=z["invented_word_per_stim"],
            anchor_names=list(z["anchor_names"]),
        )
        size_mb = os.path.getsize(path) / 1e6
        print(f"  [cache] HIT  {os.path.basename(path)} "
              f"canon={cond.canonical_X.shape} inv={cond.invented_X.shape} "
              f"({size_mb:.1f} MB)")
        return cond
    except Exception as e:
        print(f"  [cache] failed to load {path}: {e}")
        return None


# ==============================================================================
# Slicing helpers (mirror script 24 exactly)
# ==============================================================================
def slice_canonical(cond: ConditionV6, anchor: str, layer: int):
    a_idx = cond.anchor_names.index(anchor)
    return cond.canonical_X[a_idx, :, layer, :], np.asarray(cond.canonical_labels)


def slice_invented(cond: ConditionV6, anchor: str, layer: int):
    a_idx = cond.anchor_names.index(anchor)
    return cond.invented_X[a_idx, :, layer, :], np.asarray(cond.invented_word_per_stim)


def subset_canonical(X: np.ndarray, y: np.ndarray, keep: list[str]):
    mask = np.isin(y, keep)
    return X[mask], y[mask]


def subset_invented(X: np.ndarray, w: np.ndarray, keep: list[str]):
    mask = np.isin(w, keep)
    return X[mask], w[mask]


# ==============================================================================
# Probe + similarity helpers
# ==============================================================================
def train_probe(
    X_tr: np.ndarray, y_tr: np.ndarray,
) -> LogisticRegression:
    return LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs").fit(X_tr, y_tr)


def per_word_top_canonical(
    preds: np.ndarray, words: np.ndarray, *,
    canonicals: list[str], invented_set: list[str],
) -> tuple[dict[str, str], dict[str, float]]:
    """Return per_word_top[w] = top canonical, per_word_top_pct[w] = its share."""
    top: dict[str, str] = {}
    top_pct: dict[str, float] = {}
    for w in invented_set:
        mask = (words == w)
        if not mask.any():
            continue
        word_preds = preds[mask]
        counts = {c: int(np.sum(word_preds == c)) for c in canonicals}
        tc = max(counts, key=lambda c: counts[c])
        top[w] = tc
        top_pct[w] = counts[tc] / mask.sum()
    return top, top_pct


def mean_canonical_activations(
    X: np.ndarray, labels: np.ndarray, canonicals: list[str],
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for c in canonicals:
        mask = (labels == c)
        if not mask.any():
            continue
        out[c] = X[mask].mean(axis=0).astype(np.float32)
    return out


def mean_invented_activations(
    X: np.ndarray, words: np.ndarray, invented_set: list[str],
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for w in invented_set:
        mask = (words == w)
        if not mask.any():
            continue
        out[w] = X[mask].mean(axis=0).astype(np.float32)
    return out


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu < 1e-12 or nv < 1e-12:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def cosine_matrix(
    inv_means: dict[str, np.ndarray], canon_means: dict[str, np.ndarray],
    invented_set: list[str], canonicals: list[str],
) -> np.ndarray:
    """Return (n_inv, n_canon) cosine matrix."""
    n_inv = len(invented_set)
    n_canon = len(canonicals)
    out = np.zeros((n_inv, n_canon), dtype=np.float32)
    for i, w in enumerate(invented_set):
        if w not in inv_means:
            continue
        u = inv_means[w]
        u_norm = float(np.linalg.norm(u))
        if u_norm < 1e-12:
            continue
        for j, c in enumerate(canonicals):
            if c not in canon_means:
                continue
            v = canon_means[c]
            v_norm = float(np.linalg.norm(v))
            if v_norm < 1e-12:
                continue
            out[i, j] = float(np.dot(u, v) / (u_norm * v_norm))
    return out


def predict_sim_top(
    sim: np.ndarray, invented_set: list[str], canonicals: list[str],
    *, arity_conditional: bool = False,
    w_to_canonical: Optional[dict[str, str]] = None,
) -> dict[str, str]:
    """Return per-word top canonical by cosine argmax.

    If arity_conditional, restrict the argmax to canonicals whose arity
    matches w_to_canonical[w]'s arity (the *intended* arity).
    """
    out: dict[str, str] = {}
    for i, w in enumerate(invented_set):
        if arity_conditional:
            if w_to_canonical is None:
                raise ValueError("arity_conditional=True requires w_to_canonical")
            intended_arity = CANONICAL_ARITY[w_to_canonical[w]]
            allowed = [j for j, c in enumerate(canonicals)
                       if CANONICAL_ARITY[c] == intended_arity]
        else:
            allowed = list(range(len(canonicals)))
        if not allowed:
            continue
        scores = sim[i, allowed]
        out[w] = canonicals[allowed[int(np.argmax(scores))]]
    return out


# ==============================================================================
# Cell enumeration (mirror script 24's enumerate_cells exactly)
# ==============================================================================
@dataclass
class SweepCell:
    direction: str       # "N->F" or "F->N"
    train_cond: str
    train_anchor: str
    test_cond: str
    test_anchor: str
    layer: int

    M2_cano: float = 0.0
    M2_arity: float = 0.0
    M4b: float = 0.0
    M4c: float = 0.0
    per_word_top: dict[str, str] = field(default_factory=dict)
    per_word_top_pct: dict[str, float] = field(default_factory=dict)
    per_word_min_top_pct: float = 0.0

    # Similarity heuristic results
    sim_all_top: dict[str, str] = field(default_factory=dict)
    sim_arity_top: dict[str, str] = field(default_factory=dict)
    sim_all_agree: float = 0.0       # frac words where sim_all_top == per_word_top (identity)
    sim_arity_agree: float = 0.0     # frac words where sim_arity_top == per_word_top (identity within arity)
    sim_all_arity_match: float = 0.0 # frac words where ARITY(sim_all_top) == ARITY(per_word_top)

    @property
    def label(self) -> str:
        return (
            f"{self.direction} {self.train_anchor[:5]:>5}->{self.test_anchor[:5]:<5} "
            f"L{self.layer:>2}"
        )


def enumerate_cells(layers: list[int]) -> list[SweepCell]:
    cells: list[SweepCell] = []
    for L in layers:
        for tr_a in ANCHORS_NEUTRAL:
            for te_a in ANCHORS_FUNC_PFX:
                cells.append(SweepCell(
                    direction="N->F",
                    train_cond="NEUTRAL", train_anchor=tr_a,
                    test_cond="FUNC-PFX", test_anchor=te_a, layer=L,
                ))
        for tr_a in ANCHORS_FUNC_PFX:
            for te_a in ANCHORS_NEUTRAL:
                cells.append(SweepCell(
                    direction="F->N",
                    train_cond="FUNC-PFX", train_anchor=tr_a,
                    test_cond="NEUTRAL", test_anchor=te_a, layer=L,
                ))
    return cells


# ==============================================================================
# Run a single cell: probe + similarity
# ==============================================================================
def run_cell(
    cell: SweepCell, cond_by_name: dict[str, ConditionV6],
) -> SweepCell:
    train_cond = cond_by_name[cell.train_cond]
    test_cond = cond_by_name[cell.test_cond]

    X_tr_full, y_tr_full = slice_canonical(train_cond, cell.train_anchor, cell.layer)
    X_te_full, y_te_full = slice_canonical(test_cond, cell.test_anchor, cell.layer)
    X_inv_full, w_inv_full = slice_invented(test_cond, cell.test_anchor, cell.layer)

    X_tr, y_tr = subset_canonical(X_tr_full, y_tr_full, SCOPE.canonicals)
    X_te, y_te = subset_canonical(X_te_full, y_te_full, SCOPE.canonicals)
    X_inv, w_inv = subset_invented(X_inv_full, w_inv_full, SCOPE.invented_set)

    clf = train_probe(X_tr, y_tr)
    preds_canon = clf.predict(X_te)
    cell.M2_cano = float(np.mean(preds_canon == y_te))
    cell.M2_arity = float(np.mean([
        CANONICAL_ARITY[str(t)] == CANONICAL_ARITY[str(p)]
        for t, p in zip(y_te, preds_canon)
    ]))

    preds_inv = clf.predict(X_inv)
    cell.per_word_top, cell.per_word_top_pct = per_word_top_canonical(
        preds_inv, w_inv,
        canonicals=SCOPE.canonicals, invented_set=SCOPE.invented_set,
    )
    pcts = list(cell.per_word_top_pct.values())
    cell.per_word_min_top_pct = float(min(pcts)) if pcts else 0.0

    breakdown = {c: float(np.mean(preds_inv == c)) for c in SCOPE.canonicals}
    cell.M4c = float(sum(v ** 2 for v in breakdown.values()))
    n_match = sum(
        CANONICAL_ARITY[SCOPE.w_to_canonical[str(w)]] == CANONICAL_ARITY[str(p)]
        for w, p in zip(w_inv, preds_inv)
    )
    cell.M4b = n_match / len(preds_inv) if len(preds_inv) > 0 else 0.0

    # === similarity heuristic ===
    # Compute mean canonical + mean invented activations at the *target*
    # (cond, anchor, layer) - the same coordinate the probe predicts on.
    canon_means = mean_canonical_activations(
        X_te_full, y_te_full, SCOPE.canonicals,
    )
    inv_means = mean_invented_activations(
        X_inv_full, w_inv_full, SCOPE.invented_set,
    )
    sim = cosine_matrix(inv_means, canon_means,
                        SCOPE.invented_set, SCOPE.canonicals)

    cell.sim_all_top = predict_sim_top(
        sim, SCOPE.invented_set, SCOPE.canonicals,
        arity_conditional=False,
    )
    cell.sim_arity_top = predict_sim_top(
        sim, SCOPE.invented_set, SCOPE.canonicals,
        arity_conditional=True,
        w_to_canonical=SCOPE.w_to_canonical,
    )

    matches_all = 0
    matches_arity = 0
    matches_all_arity = 0
    n = 0
    for w in SCOPE.invented_set:
        if w not in cell.per_word_top:
            continue
        n += 1
        cp = cell.per_word_top[w]
        if cell.sim_all_top.get(w) == cp:
            matches_all += 1
        if cell.sim_arity_top.get(w) == cp:
            matches_arity += 1
        c_all = cell.sim_all_top.get(w)
        if c_all is not None and CANONICAL_ARITY[c_all] == CANONICAL_ARITY[cp]:
            matches_all_arity += 1
    if n > 0:
        cell.sim_all_agree = matches_all / n
        cell.sim_arity_agree = matches_arity / n
        cell.sim_all_arity_match = matches_all_arity / n
    return cell


# ==============================================================================
# Reporting helpers
# ==============================================================================
def _fmt_pct(x: float) -> str:
    return f"{x * 100:>5.1f}%"


def print_per_cell_table(cells: list[SweepCell], model_short_name: str) -> None:
    print()
    print("=" * 160)
    print(f"  {model_short_name} - per-cell agreement: probe per-word top vs "
          f"cosine-similarity argmax (scope={SCOPE.name}, {len(cells)} cells)")
    print("=" * 160)
    print(f"  {'cell':<34} | {'M2c':<5} | {'M2a':<5} | "
          f"{'M4b':<5} | {'M4c':<5} | {'pwmin':<5} | "
          f"{'agree-all':<9} | {'agree-arity':<11} | "
          f"{'arity(sim-all)=arity(probe)':<28}")
    print(f"  {'-' * 156}")
    for c in cells:
        print(
            f"  {c.label:<34} | "
            f"{c.M2_cano:.2f}  | {c.M2_arity:.2f}  | "
            f"{c.M4b:.2f}  | {c.M4c:.2f}  | "
            f"{c.per_word_min_top_pct:.2f}  | "
            f"{_fmt_pct(c.sim_all_agree):<9} | "
            f"{_fmt_pct(c.sim_arity_agree):<11} | "
            f"{_fmt_pct(c.sim_all_arity_match):<28}"
        )


def _aggregate(
    cells: list[SweepCell], key_fn,
) -> dict:
    """Return key -> dict with means."""
    by_key: dict = {}
    for c in cells:
        k = key_fn(c)
        by_key.setdefault(k, []).append(c)
    out: dict = {}
    for k, sub in by_key.items():
        out[k] = {
            "n": len(sub),
            "agree_all": float(np.mean([s.sim_all_agree for s in sub])),
            "agree_arity": float(np.mean([s.sim_arity_agree for s in sub])),
            "all_arity_match": float(np.mean([s.sim_all_arity_match for s in sub])),
            "m4c_mean": float(np.mean([s.M4c for s in sub])),
            "m4b_mean": float(np.mean([s.M4b for s in sub])),
        }
    return out


def print_aggregates(cells: list[SweepCell], model_short_name: str) -> None:
    print()
    print(f"  --- {model_short_name} per-direction aggregate (mean over {len(cells)} cells) ---")
    agg_dir = _aggregate(cells, key_fn=lambda c: c.direction)
    for d, v in sorted(agg_dir.items()):
        print(f"    {d:<5} n={v['n']:>3}  "
              f"agree-all={_fmt_pct(v['agree_all'])}  "
              f"agree-arity={_fmt_pct(v['agree_arity'])}  "
              f"arity-match={_fmt_pct(v['all_arity_match'])}  "
              f"<M4c>={v['m4c_mean']:.2f}  <M4b>={v['m4b_mean']:.2f}")

    print()
    print(f"  --- {model_short_name} per-layer aggregate (mean over anchor pairs x directions) ---")
    agg_L = _aggregate(cells, key_fn=lambda c: c.layer)
    for L in sorted(agg_L):
        v = agg_L[L]
        print(f"    L{L:>2}    n={v['n']:>3}  "
              f"agree-all={_fmt_pct(v['agree_all'])}  "
              f"agree-arity={_fmt_pct(v['agree_arity'])}  "
              f"arity-match={_fmt_pct(v['all_arity_match'])}  "
              f"<M4c>={v['m4c_mean']:.2f}  <M4b>={v['m4b_mean']:.2f}")

    # The decisive partition: distributed cells (M4c < 0.7) vs collapsed
    # cells (M4c >= 0.7). The hypothesis is that similarity predicts the
    # probe attractor at distributed cells but is trivially right (or
    # trivially wrong) at collapsed cells.
    distributed = [c for c in cells if c.M4c < 0.7]
    collapsed = [c for c in cells if c.M4c >= 0.7]
    print()
    print(f"  --- {model_short_name} concentration partition "
          f"(distributed M4c<0.7 vs collapsed M4c>=0.7) ---")
    for label, sub in [("distributed", distributed), ("collapsed  ", collapsed)]:
        if not sub:
            print(f"    {label} n=  0  (empty subset)")
            continue
        print(
            f"    {label} n={len(sub):>3}  "
            f"agree-all={_fmt_pct(float(np.mean([s.sim_all_agree for s in sub])))}  "
            f"agree-arity={_fmt_pct(float(np.mean([s.sim_arity_agree for s in sub])))}  "
            f"arity-match={_fmt_pct(float(np.mean([s.sim_all_arity_match for s in sub])))}"
        )


def print_per_word_breakdown(
    cells: list[SweepCell], model_short_name: str, *, top_k_cells: int = 3,
) -> None:
    """For the top-K distributed cells by M4c (most distributed), print the
    per-word probe top vs similarity argmax."""
    distributed = sorted(
        [c for c in cells if c.M4c < 0.7 and c.per_word_min_top_pct < 0.95],
        key=lambda c: c.M4c,
    )[:top_k_cells]
    if not distributed:
        print(f"\n  ({model_short_name}: no distributed non-lucky-default cells)")
        return
    print()
    print(f"  --- {model_short_name} per-word breakdown at the top-{top_k_cells} "
          f"most-distributed cells (M4c < 0.7, pwmin < 0.95) ---")
    for c in distributed:
        print()
        print(f"    >> {c.label}   M2c={c.M2_cano:.2f}  M2a={c.M2_arity:.2f}  "
              f"M4b={c.M4b:.2f}  M4c={c.M4c:.2f}  pwmin={c.per_word_min_top_pct:.2f}")
        print(f"       agree-all={_fmt_pct(c.sim_all_agree)}  "
              f"agree-arity={_fmt_pct(c.sim_arity_agree)}  "
              f"arity-match={_fmt_pct(c.sim_all_arity_match)}")
        print(f"       {'word':<8} | {'intended':<12} | {'probe top':<12} | "
              f"{'sim-all top':<12} | {'sim-arity top':<14} | flags")
        print(f"       {'-' * 90}")
        for w in SCOPE.invented_set:
            if w not in c.per_word_top:
                continue
            intended = SCOPE.w_to_canonical[w]
            probe = c.per_word_top.get(w, "?")
            sim_all = c.sim_all_top.get(w, "?")
            sim_ar = c.sim_arity_top.get(w, "?")
            flags = []
            if sim_all == probe:
                flags.append("=A")
            if sim_ar == probe:
                flags.append("=R")
            print(
                f"       {w:<8} | {intended:<12} | {probe:<12} | "
                f"{sim_all:<12} | {sim_ar:<14} | {' '.join(flags)}"
            )


def print_l0_block(
    cells_per_layer: dict[int, list[SweepCell]], model_short_name: str,
) -> None:
    """Compare L0 agreement to focus-layer agreement (the H1-construction-during
    -forward-pass test under v6: if L0 agreement is materially below the
    focus-layer agreement, the similarity attractor is built by intermediate
    processing, not inherited)."""
    if 0 not in cells_per_layer:
        return
    print()
    print(f"  --- {model_short_name} L0 (embedding-layer) vs focus-layer "
          f"agreement (the §3.4 H1-construction-during-forward-pass test) ---")
    print(f"    {'layer':<6} | {'n':<4} | {'agree-all':<10} | "
          f"{'agree-arity':<12} | {'arity-match':<12} | {'<M4c>':<6}")
    print(f"    {'-' * 70}")
    for L in sorted(cells_per_layer):
        sub = cells_per_layer[L]
        if not sub:
            continue
        print(
            f"    L{L:<5} | {len(sub):<4} | "
            f"{_fmt_pct(float(np.mean([s.sim_all_agree for s in sub]))):<10} | "
            f"{_fmt_pct(float(np.mean([s.sim_arity_agree for s in sub]))):<12} | "
            f"{_fmt_pct(float(np.mean([s.sim_all_arity_match for s in sub]))):<12} | "
            f"{float(np.mean([s.M4c for s in sub])):.2f}"
        )


# ==============================================================================
# Bootstrap CIs on per-cell agreement (resample invented set)
# ==============================================================================
def bootstrap_agreement(
    cells: list[SweepCell], *, n_bootstrap: int = 200, seed: int = SEED,
) -> dict:
    """Resample the 16 invented words with replacement; for each resample,
    re-aggregate the per-word agreement across all cells. Returns mean/CI
    for agree-all and agree-arity over the full cell set."""
    rng = np.random.default_rng(seed)
    samples_all: list[float] = []
    samples_arity: list[float] = []
    samples_all_arity: list[float] = []
    inv_set = list(SCOPE.invented_set)
    for _ in range(n_bootstrap):
        sub_words = list(rng.choice(inv_set, size=len(inv_set), replace=True))
        cell_means_all: list[float] = []
        cell_means_arity: list[float] = []
        cell_means_all_arity: list[float] = []
        for c in cells:
            n = 0
            ma = 0
            mar = 0
            ma_ar = 0
            for w in sub_words:
                if w not in c.per_word_top:
                    continue
                n += 1
                cp = c.per_word_top[w]
                if c.sim_all_top.get(w) == cp:
                    ma += 1
                if c.sim_arity_top.get(w) == cp:
                    mar += 1
                c_all = c.sim_all_top.get(w)
                if c_all is not None and CANONICAL_ARITY[c_all] == CANONICAL_ARITY[cp]:
                    ma_ar += 1
            if n > 0:
                cell_means_all.append(ma / n)
                cell_means_arity.append(mar / n)
                cell_means_all_arity.append(ma_ar / n)
        if cell_means_all:
            samples_all.append(float(np.mean(cell_means_all)))
            samples_arity.append(float(np.mean(cell_means_arity)))
            samples_all_arity.append(float(np.mean(cell_means_all_arity)))
    out = {}
    for name, arr in (("all", samples_all), ("arity", samples_arity),
                      ("all_arity_match", samples_all_arity)):
        a = np.array(arr)
        if len(a) == 0:
            out[name] = (float("nan"), float("nan"), float("nan"))
        else:
            out[name] = (
                float(a.mean()),
                float(np.percentile(a, 2.5)),
                float(np.percentile(a, 97.5)),
            )
    return out


# ==============================================================================
# Top-level per-model pipeline
# ==============================================================================
def run_model(spec: ModelSpec) -> Optional[dict]:
    if MODEL_FILTER and MODEL_FILTER not in spec.short_name.lower():
        return None

    print()
    print("#" * 88)
    print(f"#  Model: {spec.short_name}")
    print(f"#  Focus layers: {spec.focus_layers}")
    print(f"#  Scope: {SCOPE.name}  (canon n={len(SCOPE.canonicals)}, "
          f"inv n={len(SCOPE.invented_set)})")
    print("#" * 88)

    neut = _load_carryover_cache(spec.short_name, "NEUTRAL")
    func = _load_carryover_cache(spec.short_name, "FUNC-PFX")
    if neut is None or func is None:
        print(f"  (cache miss; skipping {spec.short_name})")
        return None
    cond_by_name = {"NEUTRAL": neut, "FUNC-PFX": func}

    # Build the cell set: focus layers + (optionally) L0
    layer_set = list(spec.focus_layers)
    if not SKIP_L0 and 0 not in layer_set:
        layer_set.insert(0, 0)
    if L0_ONLY:
        layer_set = [0]
    cells = enumerate_cells(layer_set)
    print(f"  Sweeping {len(cells)} cells across "
          f"layers {sorted(set(c.layer for c in cells))}")

    t0 = time.time()
    n_done = 0
    for c in cells:
        run_cell(c, cond_by_name)
        n_done += 1
        if n_done % 16 == 0:
            elapsed = time.time() - t0
            rate = n_done / elapsed if elapsed > 0 else 0.0
            print(f"    [{n_done}/{len(cells)}] elapsed={elapsed:.0f}s "
                  f"rate={rate:.2f} cell/s")

    focus_cells = [c for c in cells if c.layer in spec.focus_layers]
    l0_cells = [c for c in cells if c.layer == 0]

    if focus_cells:
        print_per_cell_table(focus_cells, spec.short_name)
        print_aggregates(focus_cells, spec.short_name)
        print_per_word_breakdown(focus_cells, spec.short_name, top_k_cells=3)

    if l0_cells and not L0_ONLY:
        cells_per_layer: dict[int, list[SweepCell]] = {}
        for c in cells:
            cells_per_layer.setdefault(c.layer, []).append(c)
        print_l0_block(cells_per_layer, spec.short_name)

    print()
    print(f"  --- {spec.short_name} bootstrap 95% CIs on agreement "
          f"(B={N_BOOTSTRAP}, resample invented set with replacement) ---")
    boot = bootstrap_agreement(focus_cells, n_bootstrap=N_BOOTSTRAP)
    for name, (mean, lo, hi) in boot.items():
        print(f"    {name:<18}  mean = {_fmt_pct(mean)}  "
              f"95% CI = [{_fmt_pct(lo)}, {_fmt_pct(hi)}]")

    elapsed = time.time() - t0
    print()
    print(f"  {spec.short_name} done in {elapsed:.0f}s")

    return {
        "spec": spec,
        "focus_cells": focus_cells,
        "l0_cells": l0_cells,
        "boot": boot,
    }


def print_cross_model_synthesis(results: dict[str, dict]) -> None:
    print()
    print("=" * 110)
    print("  CROSS-MODEL SYNTHESIS")
    print("=" * 110)
    print(f"  {'model':<28} | n cells | {'agree-all':<22} | "
          f"{'agree-arity':<22} | {'arity-match':<22}")
    print(f"  {'-' * 126}")
    for name, r in results.items():
        cells = r["focus_cells"]
        n = len(cells)
        boot = r["boot"]
        a_mean, a_lo, a_hi = boot["all"]
        r_mean, r_lo, r_hi = boot["arity"]
        am_mean, am_lo, am_hi = boot["all_arity_match"]
        print(
            f"  {name:<28} |  {n:>5}  | "
            f"{_fmt_pct(a_mean)} [{_fmt_pct(a_lo)}, {_fmt_pct(a_hi)}] | "
            f"{_fmt_pct(r_mean)} [{_fmt_pct(r_lo)}, {_fmt_pct(r_hi)}] | "
            f"{_fmt_pct(am_mean)} [{_fmt_pct(am_lo)}, {_fmt_pct(am_hi)}]"
        )

    print()
    print("  Interpretation:")
    print("    * 'agree-all'    = % of words where unconstrained cosine-argmax over all 15")
    print("                       canonicals matches the probe's top canonical. Chance = 1/15")
    print("                       ~= 6.7%.")
    print("    * 'agree-arity'  = same, restricted to canonicals of matching intended arity.")
    print("                       Chance = 1/n_intended_arity (1/8 ~= 12.5% for binary-intended;")
    print("                       1/7 ~= 14.3% for unary-intended).")
    print("    * 'arity-match'  = % of words where the unconstrained sim-all top has the same")
    print("                       ARITY as the probe's top canonical (ignores canonical identity).")
    print("                       Chance = max(7,8)/15 ~= 53.3% (the majority-arity baseline,")
    print("                       since with cosine-argmax over 15 canonicals the picked arity")
    print("                       has a slight bias toward the larger-arity class).")
    print()
    print("  If agree-arity is >= 60% per the §6 pre-spec threshold, the embedding-similarity")
    print("  heuristic explains the bulk of the probe's per-word routing once arity is")
    print("  controlled, and the §3.7.16 multi-factor mechanism reduces to 'softmax over")
    print("  arity-conditioned semantic neighborhood' at the focus layers. If it is at chance,")
    print("  the mechanism involves at least one factor not captured by mean-pooled-residual")
    print("  cosine (e.g., frequency-weighted softmax over the canonical readout vocabulary).")


# ==============================================================================
# main
# ==============================================================================
def main() -> int:
    _setup_logging()
    print(f"=== Script 25b - embedding-similarity probe (scope={SCOPE.name}) ===")
    print(f"    SEED={SEED}  N_PER_CLASS={N_PER_CLASS}  N_BOOTSTRAP={N_BOOTSTRAP}")
    print(f"    L0_ONLY={L0_ONLY}  SKIP_L0={SKIP_L0}  MODEL_FILTER={MODEL_FILTER!r}")
    print(f"    canonicals (n={len(SCOPE.canonicals)}): {SCOPE.canonicals}")
    print(f"    invented   (n={len(SCOPE.invented_set)}): {SCOPE.invented_set}")
    print()
    print("    Notes:")
    print("    * Similarity is computed at the *same* (cond, anchor, layer) where")
    print("      the probe makes its predictions on invented activations (= the")
    print("      target side). Each invented-word vector mu_w is mean over its 50")
    print("      stim residuals; each canonical mu_c is mean over 50 stim.")
    print("    * The probe is trained on source-cond canonical at source-anchor x")
    print("      layer (mirrors script 24's m2 protocol exactly).")
    print()

    t_start = time.time()
    results: dict[str, dict] = {}
    for spec in MODEL_SPECS:
        r = run_model(spec)
        if r is not None:
            results[spec.short_name] = r

    if len(results) >= 2:
        print_cross_model_synthesis(results)

    total = time.time() - t_start
    print()
    print(f"  total wall-clock: {total:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
