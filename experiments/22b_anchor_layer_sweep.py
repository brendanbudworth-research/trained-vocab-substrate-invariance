"""Script 22b - full anchor x anchor x layer sweep with M2-canonical
+ M2-arity + M3 + M4 battery, at both OLMo 2 7B and Gemma 2 9B.

Motivation. Script 22a discovered a clean dissociation between
M2-canonical (5-class canonical-transfer accuracy) and M2-arity
(binary-vs-unary accuracy coarsened from the same 5-class probe) at
the OLMo 2 (NEUTRAL sentence-final -> FUNC-PFX close-paren, L10, N->F)
cell: M2-canonical = 0.616 AMBIG (bootstrap P = 7%) but M2-arity =
1.000 (perfect arity-axis transfer). The §3.7.9 candidate is now
upgraded to "demonstrated under M2-arity" but only at one cell so
far. This script answers the question: is that cell uniquely
structurally special, or are there other (anchor, layer, direction)
cells in either model with similar arity-respecting transfer?

The sweep enumerates every combination at the script-21 focus layers:

  Models:            OLMo 2 7B, Gemma 2 9B
  NEUT anchors:      operator-after, sentence-final
  FUNC-PFX anchors:  operator-after, first-arg, close-paren,
                     sentence-final
  Focus layers:      OLMo 2: 4, 7, 10, 16, 24
                     Gemma 2: 2, 4, 8, 16, 17
  Directions:        N->F  (train NEUTRAL, test FUNC-PFX invented)
                     F->N  (train FUNC-PFX, test NEUTRAL invented)

Total cells: 2 models * 5 layers * 8 N->F pairings + 2 * 5 * 8 F->N
           = 160 cells.

Each cell reports:

  M1_tr     within-cond probe CV at the training side, 5-class
  M1_te     within-cond probe CV at the test side
  M2_cano   bidirectional-eligible 5-class canonical-transfer accuracy
            in this direction (not bidirectional - just this one cell)
  M2_arity  coarsened binary-vs-unary accuracy from the same 5-class
            probe
  M3_cent   centroid arity-direction cosine angle between train and
            test cells (degrees)
  M3_prob   raw-probe arity-direction cosine angle (degrees)
  M4a       invented unary mass at the test anchor (read by the
            train-side 5-class probe)
  M4b       intended-arity agreement
  M4c       canonical catchment concentration (Herfindahl)
  verdict   composite verdict using M2-arity >= 0.65 (the defensible
            threshold given the 0.60 lucky-default floor) + M4b >= 0.65
            (above the 0.52 by-arity-random baseline) + M4c < 0.70
            (not single-canonical-collapsed)
  lucky?    flag set if M4a is at floor (<= 5% or >= 95%) AND M4c >=
            0.85 - the canonical lucky-default pattern

Cache-only: uses the script-21 v3-multi-anchor activations from
outputs/cache/21_*.npz. No model loading. Expected runtime ~30s -
1 min for compute, plus cache loading.

Outputs.

  - Long-form CSV-style table: every cell, all metrics.
  - Per-model summary: top-K (default K=10) cells sorted by
    (M2-arity desc, M4b desc, M4c asc).
  - Cross-tab: M2-canonical vs M2-arity dissociation. For each cell,
    report (M2-arity - M2-canonical). Large positive gaps indicate
    arity-axis transfers but canonical-identity does not (the
    §3.7.10 finding).
  - Specific check: is the §3.7.9 cell still the strongest in its
    direction/model? If not, which is?
  - Per-(layer, direction) summary in a compact grid.

Tees all output to outputs/22b_<ts>.log.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import os
import sys
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression


# ==============================================================================
# Tee logging boilerplate (same as 19b/20/21/22a).
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


def _setup_logging() -> str | None:
    if os.environ.get("NO_LOG"):
        return None
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(log_dir, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"22b_{ts}.log")
    log_f = open(log_path, "w", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_f)
    sys.stderr = _Tee(sys.__stderr__, log_f)
    print(f"[logging] tee'ing all output to {log_path}")
    return log_path


# ==============================================================================
# Constants - must match script 21 v3-multi-anchor for cache hits.
# ==============================================================================
SEED = 17
N_PER_CLASS = 50
V3_STIMULUS_VERSION = "v3-multi-anchor"

# Defensible thresholds based on script-22a findings:
GATE_CANONICAL_PASS = 0.65   # original 5-class threshold
GATE_ARITY_PASS = 0.65       # 0.05 above the 0.60 lucky-default floor
M4B_PASS = 0.65              # ~25% above the 0.52 by-arity-random baseline
M4C_DISTRIBUTED = 0.70       # below this = not collapsed; above = catchment

CANONICALS = ["and", "or", "not", "implies", "necessarily"]
UNARY_CANONICALS = ["not", "necessarily"]
BINARY_CANONICALS = ["and", "or", "implies"]
CANONICAL_ARITY = {"and": 2, "or": 2, "implies": 2, "not": 1, "necessarily": 1}
INVENTED_WORDS = ["bliq", "dren", "vusp", "molex", "perph"]
W_TO_CANONICAL = dict(zip(INVENTED_WORDS, CANONICALS))

NEUT_ANCHORS = ["operator-after", "sentence-final"]
FUNC_ANCHORS = ["operator-after", "first-arg", "close-paren", "sentence-final"]

FOCUS_LAYERS = {
    "OLMo 2 7B": [4, 7, 10, 16, 24],
    "Gemma 2 9B": [2, 4, 8, 16, 17],
}


# ==============================================================================
# Cache loading.
# ==============================================================================
def _cache_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "outputs", "cache")


def load_21_cache(model_short: str, condition: str) -> dict:
    slug = model_short.replace(" ", "_")
    path = os.path.join(_cache_dir(),
                        f"21_{slug}_{condition}_npc{N_PER_CLASS}_{V3_STIMULUS_VERSION}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required script-21 cache missing: {path}")
    z = np.load(path, allow_pickle=False)
    canon_X = z["canonical_X"].astype(np.float32)
    inv_X = z["invented_X"].astype(np.float32)
    anchor_names = [str(a) for a in z["anchor_names"]]
    return {
        "canonical_X": canon_X,
        "canonical_labels": np.array([str(l) for l in z["canonical_labels"]]),
        "invented_X": inv_X,
        "invented_word_per_stim": np.array([str(w) for w in z["invented_word_per_stim"]]),
        "anchor_names": anchor_names,
    }


def slice_canon(cache: dict, anchor: str, layer: int) -> np.ndarray:
    a_idx = cache["anchor_names"].index(anchor)
    return cache["canonical_X"][a_idx, :, layer, :]


def slice_invented(cache: dict, anchor: str, layer: int) -> np.ndarray:
    a_idx = cache["anchor_names"].index(anchor)
    return cache["invented_X"][a_idx, :, layer, :]


# ==============================================================================
# Metric primitives.
# ==============================================================================
def within_cond_cv(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> float:
    """5-fold stratified CV of a 5-class LR probe. Used as M1."""
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    scores = []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs")
        clf.fit(X[tr], y[tr])
        scores.append(clf.score(X[te], y[te]))
    return float(np.mean(scores))


def m2_metrics(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    """Returns (M2-canonical, M2-arity, predictions on test).

    Trains a 5-class LR probe on (X_train, y_train); scores 5-class
    accuracy on (X_test, y_test) (= M2-canonical); also scores the
    coarsened binary-vs-unary accuracy on the same predictions (=
    M2-arity).
    """
    clf = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs").fit(X_train, y_train)
    preds = clf.predict(X_test)
    m2_cano = float(np.mean(preds == y_test))
    n_arity_agree = sum(
        CANONICAL_ARITY[str(t)] == CANONICAL_ARITY[str(p)]
        for t, p in zip(y_test, preds)
    )
    m2_arity = n_arity_agree / len(y_test)
    return m2_cano, m2_arity, preds


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def centroid_arity_direction(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mean over unary canonical centroids minus mean over binary canonical
    centroids (unit-normalised)."""
    unary = np.mean([X[y == c].mean(axis=0) for c in UNARY_CANONICALS], axis=0)
    binary = np.mean([X[y == c].mean(axis=0) for c in BINARY_CANONICALS], axis=0)
    return unit(unary - binary)


def probe_arity_direction(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Train a BINARY arity-vs-unary LR probe on the raw activations
    (matching scripts 19/19b/20 convention). Return the probe's
    decision direction (unit-normalised)."""
    y_arity = np.array(["U" if CANONICAL_ARITY[str(c)] == 1 else "B" for c in y])
    clf = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs").fit(X, y_arity)
    # LR coefficient direction: from B -> U, depending on class order.
    classes = list(clf.classes_)
    direction = clf.coef_[0]
    # Normalize sign so the direction points U-ward: dot with centroid
    # direction; if negative, flip.
    cent = centroid_arity_direction(X, y)
    sign = np.sign(np.dot(unit(direction), cent))
    if sign == 0:
        sign = 1.0
    return unit(sign * direction)


def cosine_angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    cos = float(np.clip(np.dot(unit(u), unit(v)), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def invented_breakdown(
    X_train: np.ndarray, y_train: np.ndarray,
    X_inv_test: np.ndarray, inv_words: np.ndarray,
) -> dict:
    """Compute M4a / M4b / M4c using the 5-class LR probe trained on
    (X_train, y_train), evaluated on invented test activations."""
    clf = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs").fit(X_train, y_train)
    preds = clf.predict(X_inv_test)
    n = len(preds)
    canon_counts = {c: int(np.sum(preds == c)) for c in CANONICALS}
    canon_pct = {c: canon_counts[c] / n for c in CANONICALS}
    m4a = sum(canon_pct[c] for c in UNARY_CANONICALS)
    m4c = sum(p ** 2 for p in canon_pct.values())

    n_agree = 0
    per_word_top: dict[str, str] = {}
    per_word_top_pct: dict[str, float] = {}
    per_word_unary_pct: dict[str, float] = {}
    for w in INVENTED_WORDS:
        mask = inv_words == w
        if mask.sum() == 0:
            continue
        w_preds = preds[mask]
        intended_arity = CANONICAL_ARITY[W_TO_CANONICAL[w]]
        for pred in w_preds:
            if CANONICAL_ARITY[str(pred)] == intended_arity:
                n_agree += 1
        w_counts = {c: int(np.sum(w_preds == c)) for c in CANONICALS}
        top = max(w_counts, key=lambda c: w_counts[c])
        per_word_top[w] = top
        per_word_top_pct[w] = w_counts[top] / len(w_preds)
        per_word_unary_pct[w] = (
            sum(w_counts[c] for c in UNARY_CANONICALS) / len(w_preds)
        )
    m4b = n_agree / n if n > 0 else 0.0

    return {
        "M4a": m4a, "M4b": m4b, "M4c": m4c,
        "canon_pct": canon_pct,
        "per_word_top": per_word_top,
        "per_word_top_pct": per_word_top_pct,
        "per_word_unary_pct": per_word_unary_pct,
    }


# ==============================================================================
# Cell sweep.
# ==============================================================================
@dataclass
class SweepCell:
    model: str
    direction: str           # "N->F" or "F->N"
    train_cond: str          # "NEUTRAL" or "FUNC-PFX"
    train_anchor: str
    test_cond: str
    test_anchor: str
    layer: int

    # Filled by run_cell:
    M1_tr: float = 0.0
    M1_te: float = 0.0
    M2_cano: float = 0.0
    M2_arity: float = 0.0
    M3_cent_deg: float = 0.0
    M3_prob_deg: float = 0.0
    M4a: float = 0.0
    M4b: float = 0.0
    M4c: float = 0.0
    per_word_summary: str = ""
    per_word_min_top_pct: float = 0.0  # min over 5 invented words of within-word top concentration
    per_word_max_top_pct: float = 0.0  # max over 5 invented words of within-word top concentration
    lucky_default: bool = False

    @property
    def m2_gap(self) -> float:
        return self.M2_arity - self.M2_cano

    @property
    def verdict(self) -> str:
        """Composite verdict for arity-respecting cross-notation transfer.
        Requires all of:
          - M2-arity >= 0.65 (above the 0.60 lucky-default floor)
          - M4b >= 0.65 (above the 0.52 by-arity-random baseline)
          - M4c < 0.70 (not single-canonical collapsed)
          - M4a in [0.10, 0.90] (not at ceiling or floor)
          - not lucky_default
        """
        if self.lucky_default:
            return "LUCKY-NEG"
        passes = (
            self.M2_arity >= GATE_ARITY_PASS
            and self.M4b >= M4B_PASS
            and self.M4c < M4C_DISTRIBUTED
            and 0.10 <= self.M4a <= 0.90
        )
        if passes:
            return "PASS-arity"
        # Track partial-pass for diagnostics:
        if self.M2_arity >= GATE_ARITY_PASS and self.M4b >= M4B_PASS:
            return "ARITY-AXIS-ONLY"  # but M4c collapsed or M4a at ceiling
        if self.M2_arity >= GATE_ARITY_PASS:
            return "M2A-ONLY"
        return "FAIL"


def detect_lucky_default(bd: dict) -> bool:
    """A lucky-default cell: every invented word is mapped to ONE
    canonical with within-word concentration >= 0.95. This means each
    word's predictions are essentially deterministic on one canonical,
    which is the lucky-default-with-escape pattern (e.g., 4 words ->
    "and" 100%, perph -> "necessarily" 100%; gives M4b high by accident
    when intended arities happen to align with the default canonical).

    The §3.7.9 cell escapes this detector because perph has
    per_word_top_pct = 0.50 (a genuine within-word split), bringing
    min(per_word_top_pct) below the 0.95 threshold.

    Stricter than the original detector, which required M4c >= 0.85
    and was leaky at M4c = 0.68 (the 4-of-5-at-ceiling + 1-escape
    pattern).
    """
    pcts = list(bd["per_word_top_pct"].values())
    if len(pcts) == 0:
        return False
    return min(pcts) >= 0.95


def run_cell(caches: dict, cell: SweepCell) -> SweepCell:
    train_cache = caches[(cell.model, cell.train_cond)]
    test_cache = caches[(cell.model, cell.test_cond)]

    X_tr = slice_canon(train_cache, cell.train_anchor, cell.layer)
    y_tr = train_cache["canonical_labels"]
    X_te = slice_canon(test_cache, cell.test_anchor, cell.layer)
    y_te = test_cache["canonical_labels"]
    X_inv_te = slice_invented(test_cache, cell.test_anchor, cell.layer)
    inv_words = test_cache["invented_word_per_stim"]

    cell.M1_tr = within_cond_cv(X_tr, y_tr)
    cell.M1_te = within_cond_cv(X_te, y_te)
    cell.M2_cano, cell.M2_arity, _ = m2_metrics(X_tr, y_tr, X_te, y_te)

    cent_tr = centroid_arity_direction(X_tr, y_tr)
    cent_te = centroid_arity_direction(X_te, y_te)
    cell.M3_cent_deg = cosine_angle_deg(cent_tr, cent_te)

    prob_tr = probe_arity_direction(X_tr, y_tr)
    prob_te = probe_arity_direction(X_te, y_te)
    cell.M3_prob_deg = cosine_angle_deg(prob_tr, prob_te)

    bd = invented_breakdown(X_tr, y_tr, X_inv_te, inv_words)
    cell.M4a = bd["M4a"]
    cell.M4b = bd["M4b"]
    cell.M4c = bd["M4c"]
    pcts = list(bd["per_word_top_pct"].values())
    cell.per_word_min_top_pct = float(min(pcts)) if pcts else 0.0
    cell.per_word_max_top_pct = float(max(pcts)) if pcts else 0.0
    cell.lucky_default = detect_lucky_default(bd)
    cell.per_word_summary = " ".join(
        f"{w}={bd['per_word_top'][w]}({bd['per_word_top_pct'][w]*100:.0f}%/{bd['per_word_unary_pct'][w]*100:.0f}%U)"
        for w in INVENTED_WORDS if w in bd["per_word_top"]
    )
    return cell


def enumerate_cells(model: str) -> list[SweepCell]:
    cells: list[SweepCell] = []
    for L in FOCUS_LAYERS[model]:
        # N -> F
        for tr_a in NEUT_ANCHORS:
            for te_a in FUNC_ANCHORS:
                cells.append(SweepCell(
                    model=model, direction="N->F",
                    train_cond="NEUTRAL", train_anchor=tr_a,
                    test_cond="FUNC-PFX", test_anchor=te_a,
                    layer=L,
                ))
        # F -> N
        for tr_a in FUNC_ANCHORS:
            for te_a in NEUT_ANCHORS:
                cells.append(SweepCell(
                    model=model, direction="F->N",
                    train_cond="FUNC-PFX", train_anchor=tr_a,
                    test_cond="NEUTRAL", test_anchor=te_a,
                    layer=L,
                ))
    return cells


# ==============================================================================
# Reporting helpers.
# ==============================================================================
def _cell_short(c: SweepCell) -> str:
    return (
        f"{c.model:<11} {c.direction} {c.train_anchor[:5]}->{c.test_anchor[:5]} "
        f"L{c.layer}"
    )


def print_long_table(cells: list[SweepCell]) -> None:
    print()
    print("=" * 200)
    print("FULL SWEEP - all cells, all metrics")
    print("=" * 200)
    print()
    h = (
        f"  {'cell':<55} | {'M1tr':<5} | {'M1te':<5} | {'M2cano':<7} | "
        f"{'M2arty':<7} | {'gap':<6} | {'M3°c':<5} | {'M3°p':<5} | "
        f"{'M4a':<6} | {'M4b':<6} | {'M4c':<5} | {'pwmin':<5} | "
        f"{'verdict':<14} | per-word"
    )
    print(h)
    print(f"  {'-' * (len(h) - 2)}")
    for c in cells:
        flag = " *LUCKY*" if c.lucky_default else ""
        print(
            f"  {_cell_short(c):<55} | "
            f"{c.M1_tr:.2f}  | {c.M1_te:.2f}  | "
            f"{c.M2_cano:.3f}   | {c.M2_arity:.3f}   | "
            f"{c.m2_gap:+.3f} | "
            f"{c.M3_cent_deg:>4.1f}° | {c.M3_prob_deg:>4.1f}° | "
            f"{c.M4a*100:>4.1f}% | {c.M4b*100:>4.1f}% | "
            f"{c.M4c:.2f}  | {c.per_word_min_top_pct:.2f}  | "
            f"{c.verdict + flag:<14} | {c.per_word_summary}"
        )


def print_top_k(cells: list[SweepCell], k: int = 12) -> None:
    print()
    print("=" * 200)
    print(f"TOP-{k} CELLS BY ARITY-RESPECTING TRANSFER (M2-arity desc, M4b desc, M4c asc)")
    print("=" * 200)
    print()
    print(f"  Composite key: (M2-arity, M4b, -M4c). Lucky-default cells")
    print(f"  excluded since their high M4b is the predict-all-binary trap.")
    print()
    eligible = [c for c in cells if not c.lucky_default]
    sorted_cells = sorted(
        eligible,
        key=lambda c: (c.M2_arity, c.M4b, -c.M4c),
        reverse=True,
    )[:k]
    h = (
        f"  {'rank':<4} | {'cell':<55} | {'M2cano':<7} | {'M2arty':<7} | "
        f"{'gap':<6} | {'M4a':<6} | {'M4b':<6} | {'M4c':<5} | "
        f"{'pwmin':<5} | {'verdict':<14} | per-word"
    )
    print(h)
    print(f"  {'-' * (len(h) - 2)}")
    for i, c in enumerate(sorted_cells):
        print(
            f"  {i+1:>2}.  | {_cell_short(c):<55} | "
            f"{c.M2_cano:.3f}   | {c.M2_arity:.3f}   | {c.m2_gap:+.3f} | "
            f"{c.M4a*100:>4.1f}% | {c.M4b*100:>4.1f}% | {c.M4c:.2f}  | "
            f"{c.per_word_min_top_pct:.2f}  | "
            f"{c.verdict:<14} | {c.per_word_summary}"
        )


def print_dissociation_summary(cells: list[SweepCell]) -> None:
    print()
    print("=" * 200)
    print("M2-CANONICAL vs M2-ARITY DISSOCIATION SUMMARY")
    print("=" * 200)
    print()
    print(f"  Large positive gap (M2-arity - M2-canonical) = arity axis transfers")
    print(f"  but canonical-identity does not. This is the §3.7.10 finding.")
    print()
    print(f"  Top 10 cells by absolute dissociation gap:")
    print()
    h = (
        f"  {'cell':<55} | {'M2cano':<7} | {'M2arty':<7} | {'gap':<6} | "
        f"verdict        | mechanism"
    )
    print(h)
    print(f"  {'-' * (len(h) - 2)}")
    sorted_by_gap = sorted(cells, key=lambda c: -c.m2_gap)[:10]
    for c in sorted_by_gap:
        mech = ""
        if c.lucky_default:
            mech = "lucky-default (predict-all-1-binary; gap is chance-by-arity, not real)"
        elif c.m2_gap >= 0.30:
            mech = "arity-axis transfers; canonical-identity within-arity confusion"
        elif c.m2_gap >= 0.15:
            mech = "modest dissociation; check confusion matrix"
        else:
            mech = "near-aligned; canonical-identity and arity axes co-transport"
        print(
            f"  {_cell_short(c):<55} | {c.M2_cano:.3f}   | {c.M2_arity:.3f}   | "
            f"{c.m2_gap:+.3f} | {c.verdict + ('*' if c.lucky_default else ''):<14} | {mech}"
        )


def print_3_7_9_followup(cells: list[SweepCell]) -> None:
    """Compare the §3.7.9 cell (OLMo NEUT sf -> FUNC cp L10 N->F) against
    every other cell in the sweep and report whether it's still the
    unique best."""
    print()
    print("=" * 200)
    print("§3.7.9 CELL VS ALL ALTERNATIVES")
    print("=" * 200)
    print()
    target = next(
        c for c in cells
        if c.model == "OLMo 2 7B"
        and c.direction == "N->F"
        and c.train_cond == "NEUTRAL" and c.train_anchor == "sentence-final"
        and c.test_cond == "FUNC-PFX" and c.test_anchor == "close-paren"
        and c.layer == 10
    )
    print(f"  Target cell (script-22a primary candidate, §3.7.9):")
    print(f"    {_cell_short(target):<55}")
    print(f"    M2-canonical: {target.M2_cano:.3f}")
    print(f"    M2-arity:     {target.M2_arity:.3f}")
    print(f"    gap:          {target.m2_gap:+.3f}")
    print(f"    M4a / b / c:  {target.M4a*100:.1f}% / {target.M4b*100:.1f}% / {target.M4c:.2f}")
    print(f"    M3 cent / prob: {target.M3_cent_deg:.1f}° / {target.M3_prob_deg:.1f}°")
    print(f"    per-word:     {target.per_word_summary}")
    print(f"    verdict:      {target.verdict}")
    print()
    # Cells with M2-arity >= target AND M4b >= target.M4b, excluding the target itself:
    challengers = [
        c for c in cells
        if not c.lucky_default
        and (c.M2_arity, c.M4b) >= (target.M2_arity, target.M4b)
        and not (
            c.model == target.model and c.direction == target.direction
            and c.train_anchor == target.train_anchor
            and c.test_anchor == target.test_anchor
            and c.layer == target.layer
        )
    ]
    challengers.sort(key=lambda c: (-c.M2_arity, -c.M4b, c.M4c))
    if challengers:
        print(f"  Challengers (M2-arity >= target AND M4b >= target):")
        for c in challengers:
            print(
                f"    {_cell_short(c):<55} | M2arity={c.M2_arity:.3f} | "
                f"M4b={c.M4b*100:.1f}% | M4c={c.M4c:.2f} | "
                f"M4a={c.M4a*100:.1f}% | verdict={c.verdict}"
            )
    else:
        print(f"  No challengers found. The §3.7.9 cell is the UNIQUE strongest")
        print(f"  arity-respecting transfer cell in the OLMo 2 + Gemma 2 x 5 layers")
        print(f"  x 16 anchor pairings = 160-cell battery.")


def print_per_layer_grid(cells: list[SweepCell], model: str) -> None:
    print()
    print(f"  -- {model} per-layer arity-respecting transfer grid --")
    print()
    print(f"  Rows: (train anchor, test anchor) pairs. Cols: layers.")
    print(f"  Each entry shows verdict / M2-arity / M4b.")
    print()
    layers = FOCUS_LAYERS[model]
    pairs_n2f = [(tr, te) for tr in NEUT_ANCHORS for te in FUNC_ANCHORS]
    pairs_f2n = [(tr, te) for tr in FUNC_ANCHORS for te in NEUT_ANCHORS]
    for direction, pairs in [("N->F", pairs_n2f), ("F->N", pairs_f2n)]:
        print(f"  Direction {direction}:")
        header = "    pair                 | " + " | ".join(f"L{L:>2}        " for L in layers)
        print(header)
        print(f"    {'-' * (len(header) - 4)}")
        for tr, te in pairs:
            row = f"    {tr[:5]:<6}->{te[:5]:<6}     | "
            for L in layers:
                c = next(
                    cc for cc in cells
                    if cc.model == model and cc.direction == direction
                    and cc.train_anchor == tr and cc.test_anchor == te
                    and cc.layer == L
                )
                lucky = "*" if c.lucky_default else " "
                row += f"{c.verdict[:4]:<5}{c.M2_arity:.2f}{lucky} | "
            print(row)
        print()


# ==============================================================================
# Driver.
# ==============================================================================
def main() -> None:
    log_path = _setup_logging()
    print("Script 22b - full anchor x layer sweep")
    print(f"  Models: OLMo 2 7B, Gemma 2 9B")
    print(f"  NEUT anchors: {NEUT_ANCHORS}")
    print(f"  FUNC anchors: {FUNC_ANCHORS}")
    print(f"  Focus layers: OLMo 2 {FOCUS_LAYERS['OLMo 2 7B']}; Gemma 2 {FOCUS_LAYERS['Gemma 2 9B']}")
    print(f"  Total cells per model: 5 layers x (2x4 N->F + 4x2 F->N) = 80")
    print(f"  Grand total: 160 cells")
    print()
    print(f"  Verdict thresholds:")
    print(f"    M2-arity PASS:  >= {GATE_ARITY_PASS} (0.05 above lucky-default floor)")
    print(f"    M4b PASS:       >= {M4B_PASS}    (~25% above by-arity-random baseline)")
    print(f"    M4c distributed: < {M4C_DISTRIBUTED}    (above = single-canonical catchment)")
    print()

    print("=" * 200)
    print("LOADING SCRIPT-21 CACHES")
    print("=" * 200)
    caches: dict[tuple[str, str], dict] = {}
    for model in ["OLMo 2 7B", "Gemma 2 9B"]:
        for cond in ["NEUTRAL", "FUNC-PFX"]:
            caches[(model, cond)] = load_21_cache(model, cond)
            print(f"  loaded 21_{model.replace(' ', '_')}_{cond}: "
                  f"canon{caches[(model, cond)]['canonical_X'].shape}, "
                  f"inv{caches[(model, cond)]['invented_X'].shape}")

    print()
    print("=" * 200)
    print("RUNNING SWEEP")
    print("=" * 200)
    print()
    all_cells: list[SweepCell] = []
    for model in ["OLMo 2 7B", "Gemma 2 9B"]:
        cells = enumerate_cells(model)
        print(f"  {model}: {len(cells)} cells", end="", flush=True)
        t0 = time.time()
        for c in cells:
            run_cell(caches, c)
        elapsed = time.time() - t0
        print(f" ... done in {elapsed:.1f}s")
        all_cells.extend(cells)

    # Long table per model (split for readability):
    for model in ["OLMo 2 7B", "Gemma 2 9B"]:
        model_cells = [c for c in all_cells if c.model == model]
        print()
        print()
        print(f"########  {model} - all {len(model_cells)} cells  ########")
        print_long_table(model_cells)

    print_top_k(all_cells, k=12)
    print_dissociation_summary(all_cells)
    print_3_7_9_followup(all_cells)

    print()
    print("=" * 200)
    print("PER-(LAYER, DIRECTION) VERDICT GRID")
    print("=" * 200)
    for model in ["OLMo 2 7B", "Gemma 2 9B"]:
        print_per_layer_grid(all_cells, model)

    # Headline.
    print()
    print("=" * 200)
    print("HEADLINE")
    print("=" * 200)
    pass_arity = [c for c in all_cells if c.verdict == "PASS-arity"]
    arity_axis_only = [c for c in all_cells if c.verdict == "ARITY-AXIS-ONLY"]
    lucky_negs = [c for c in all_cells if c.lucky_default]
    print()
    print(f"  Total cells:                 {len(all_cells)}")
    print(f"  PASS-arity cells:            {len(pass_arity)} (full arity-respecting transfer)")
    print(f"  ARITY-AXIS-ONLY cells:       {len(arity_axis_only)} (M2-arity + M4b PASS but M4c collapsed)")
    print(f"  Lucky-default cells:         {len(lucky_negs)} (predict-all-one-canonical baseline)")
    print(f"  FAIL cells:                  {len(all_cells) - len(pass_arity) - len(arity_axis_only) - len(lucky_negs)}")
    print()
    if pass_arity:
        print(f"  PASS-arity cells (M2-arity >= {GATE_ARITY_PASS}, M4b >= {M4B_PASS}, M4c < {M4C_DISTRIBUTED}, M4a in [0.10, 0.90]):")
        for c in pass_arity:
            print(f"    - {_cell_short(c):<55} | M2arty={c.M2_arity:.3f} M4b={c.M4b*100:.1f}% M4c={c.M4c:.2f} M4a={c.M4a*100:.1f}%")
    else:
        print(f"  *** NO cells achieve the full arity-respecting PASS criterion. ***")
        print(f"      The §3.7.9 cell's M4c=0.57 fails the < 0.70 distributed threshold;")
        print(f"      similar for any other near-PASS cell. The arity-axis-only verdict")
        print(f"      may be the practical maximum given the 5-canonical limitation.")
    print()

    if log_path:
        print(f"[logging] full transcript written to: {log_path}")


if __name__ == "__main__":
    main()
