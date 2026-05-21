"""Script 22a - bootstrap CI on M2 canonical-transfer gate accuracy at the
script-21 borderline cells.

Motivation. Script 21 identifies a candidate cross-notation arity-
respecting transfer at OLMo 2 close-paren L10 N->F (M4b = 90%, 4 of 5
invented words track intended arity, mass distributed across "and" +
"necessarily"). The M2 canonical-transfer gate at this cell is 0.616 —
exactly 0.034 below the 0.65 PASS threshold. The threshold-based
verdict (AMBIG) is therefore fragile to stimulus-sampling noise.

Separately, script 19b found that the Gemma 2 L17 same-layer gate F->N
flipped from 0.644 (AMBIG, v1 seeds) to 0.652 (PASS, v2 seeds) — a
0.008 shift across nominally-identical seed regimes. This indicates the
gate accuracy itself has a non-trivial variance and the threshold
verdict can flip on stimulus noise.

This script puts confidence intervals on the gate accuracy at the
script-21 borderline cells, using stim-resampling bootstrap on the
5-class canonical-transfer test:

  At each (model, anchor, layer, direction) target cell:
    For B = 500 bootstrap iterations:
      1. Resample (with replacement) within each canonical class on
         the *training* side to preserve class balance.
      2. Fit a 5-class logistic-regression probe on the resampled
         training activations.
      3. Score on the (un-resampled) test activations.
      4. Record the test accuracy.
    Report the mean, 95% CI (percentile method), and the fraction of
    bootstrap samples that pass the 0.65 threshold (= bootstrap PASS
    probability).

Cells tested (all loaded from script 21's v3-multi-anchor cache):

  Primary candidate:
    OLMo 2 7B, close-paren anchor, L10, NEUTRAL -> FUNC-PFX  (M2 = 0.616 in script 21)

  Validation pairs in the same cell:
    OLMo 2 7B, close-paren anchor, L10, FUNC-PFX -> NEUTRAL  (M2 = 0.332 in script 21; reverse-direction
                                                              that FAILED — useful as a known-negative)
    OLMo 2 7B, operator-after, L10, NEUTRAL -> FUNC-PFX  (M2 = 0.800 in 19b/21; known PASS for symmetry)
    Gemma 2 9B, operator-after, L17, FUNC-PFX -> NEUTRAL  (M2 = 0.652 in 19b v2; threshold-boundary
                                                           case from the §3.7 reproducibility note)
    Gemma 2 9B, operator-after, L17, FUNC-PFX -> NEUTRAL  (using 19b cache — same answer should obtain
                                                           but it tests cross-script reproducibility)

  Lucky-default cells (to verify M2 stays at floor under bootstrap):
    OLMo 2 7B, operator-after, L7, FUNC-PFX -> NEUTRAL  (M2 = 0.212 / 0.276 across runs; should stay
                                                        FAIL)
    Gemma 2 9B, first-arg, L8, FUNC-PFX -> NEUTRAL  (M2 = 0.200; should stay FAIL)

Cache-only: loads script-21 multi-anchor caches and the 19b v2 same-
anchor cache. No model loading. Estimated runtime ~2-5 min for 500
bootstrap iterations across all cells.

Output. Per-cell: bootstrap mean, [95% CI low, high], P(M2 >= 0.65)
under the bootstrap distribution. Also report the cross-canonical
confusion matrix at the point estimate to see *which* canonicals are
mis-classified at the borderline cells.

Stable-seed stimulus generation guaranteed by the cache metadata
checksum (inherited from 19b v2 and script 21 v3).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression


# ==============================================================================
# Tee logging - same pattern as 19b/20/21.
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
    log_path = os.path.join(log_dir, f"22a_{ts}.log")
    log_f = open(log_path, "w", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_f)
    sys.stderr = _Tee(sys.__stderr__, log_f)
    print(f"[logging] tee'ing all output to {log_path}")
    return log_path


# ==============================================================================
# Constants - must match script 19b v2 / script 21 v3 for cache hits.
# ==============================================================================
SEED = 17
N_PER_CLASS = 50
N_BOOTSTRAP = 500
GATE_THRESHOLD = 0.65

V2_STIMULUS_VERSION = "v2-stable-seeds"        # 19b cache
V3_STIMULUS_VERSION = "v3-multi-anchor"        # 21 cache

CANONICALS = ["and", "or", "not", "implies", "necessarily"]
UNARY_CANONICALS = ["not", "necessarily"]
BINARY_CANONICALS = ["and", "or", "implies"]
CANONICAL_ARITY = {"and": 2, "or": 2, "implies": 2, "not": 1, "necessarily": 1}
INVENTED_WORDS = ["bliq", "dren", "vusp", "molex", "perph"]
W_TO_CANONICAL = dict(zip(INVENTED_WORDS, CANONICALS))


def stable_seed(*parts, base: int = SEED, modulo: int = 100_000) -> int:
    s = "::".join(map(str, parts)).encode("utf-8")
    h = int(hashlib.blake2b(s, digest_size=8).hexdigest(), 16)
    return base + (h % modulo)


def prompts_checksum(prompts: list[str]) -> str:
    h = hashlib.blake2b(digest_size=16)
    for p in prompts:
        h.update(p.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


# Import stimulus generators from 19b (templates etc.) for cache
# verification. The cache hashes must match what 19b/21 produced.
def _load_19b():
    import importlib.machinery
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "19b_directional_angle_gated.py")
    loader = importlib.machinery.SourceFileLoader("_m19b", path)
    spec = importlib.util.spec_from_loader("_m19b", loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_m19b"] = mod
    loader.exec_module(mod)
    return mod


_M19B = _load_19b()
make_neutral_stimuli = _M19B.make_neutral_stimuli
make_functional_canonical_stimuli = _M19B.make_functional_canonical_stimuli
make_functional_invented_stimuli = _M19B.make_functional_invented_stimuli
_generate_prompts = _M19B._generate_prompts


# ==============================================================================
# Cache loaders.
# ==============================================================================
def _cache_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "outputs", "cache")


def load_21_cache(model_short: str, condition: str) -> dict:
    """Load script-21 multi-anchor cache. Returns a dict with keys:
    canonical_X (n_anchors, n_stim, n_layers, dim), canonical_labels,
    anchor_names, etc."""
    slug = model_short.replace(" ", "_")
    path = os.path.join(_cache_dir(),
                        f"21_{slug}_{condition}_npc{N_PER_CLASS}_{V3_STIMULUS_VERSION}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required script-21 cache missing: {path}")
    z = np.load(path, allow_pickle=False)
    canon_X = z["canonical_X"].astype(np.float32)
    inv_X = z["invented_X"].astype(np.float32)
    anchor_names = list(z["anchor_names"])
    print(f"  [cache] 21_{slug}_{condition}: shape={canon_X.shape}, "
          f"anchors={anchor_names}")
    return {
        "canonical_X": canon_X,
        "canonical_labels": z["canonical_labels"],
        "invented_X": inv_X,
        "invented_word_per_stim": z["invented_word_per_stim"],
        "anchor_names": anchor_names,
    }


def load_19b_cache(model_short: str, condition: str) -> dict:
    """Load script-19b cache (single-anchor, operator-after only)."""
    slug = model_short.replace(" ", "_")
    path = os.path.join(_cache_dir(),
                        f"19b_{slug}_{condition}_npc{N_PER_CLASS}_{V2_STIMULUS_VERSION}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required script-19b cache missing: {path}")
    z = np.load(path, allow_pickle=False)
    canon_X = z["canonical_X"].astype(np.float32)  # (n_layers, n_stim, dim)
    print(f"  [cache] 19b_{slug}_{condition}: shape={canon_X.shape}")
    return {
        "canonical_X_per_layer": canon_X,
        "canonical_labels": z["canonical_labels"],
    }


def slice_21(cache: dict, anchor: str, layer: int) -> np.ndarray:
    """Returns (n_stim, dim) canonical slice at (anchor, layer)."""
    a_idx = cache["anchor_names"].index(anchor)
    return cache["canonical_X"][a_idx, :, layer, :]


def slice_21_invented(cache: dict, anchor: str, layer: int) -> np.ndarray:
    """Returns (n_stim, dim) invented slice at (anchor, layer)."""
    a_idx = cache["anchor_names"].index(anchor)
    return cache["invented_X"][a_idx, :, layer, :]


def invented_arity_breakdown(
    X_train: np.ndarray, y_train: np.ndarray,
    X_inv_test: np.ndarray, inv_words: np.ndarray,
) -> dict:
    """Compute M4a (unary mass), M4b (intended-arity agreement), M4c
    (Herfindahl) and per-word predicted-canonical breakdown at the same
    (train, test) cell."""
    clf = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs").fit(X_train, y_train)
    preds = clf.predict(X_inv_test)
    n = len(preds)
    canon_counts = {c: int(np.sum(preds == c)) for c in CANONICALS}
    canon_pct = {c: canon_counts[c] / n for c in CANONICALS}

    n_unary = sum(canon_counts[c] for c in UNARY_CANONICALS)
    m4a = n_unary / n

    n_agree = 0
    per_word_top: dict[str, str] = {}
    per_word_unary_pct: dict[str, float] = {}
    for w in INVENTED_WORDS:
        mask = np.array([str(iw) == w for iw in inv_words])
        if mask.sum() == 0:
            continue
        w_preds = preds[mask]
        intended_arity = CANONICAL_ARITY[W_TO_CANONICAL[w]]
        for pred in w_preds:
            if CANONICAL_ARITY[str(pred)] == intended_arity:
                n_agree += 1
        w_counts = {c: int(np.sum(w_preds == c)) for c in CANONICALS}
        per_word_top[w] = max(w_counts, key=lambda c: w_counts[c])
        per_word_unary_pct[w] = (
            sum(w_counts[c] for c in UNARY_CANONICALS) / len(w_preds)
        )
    m4b = n_agree / n if n > 0 else 0.0
    m4c = sum(p ** 2 for p in canon_pct.values())

    return {
        "M4a": m4a, "M4b": m4b, "M4c": m4c,
        "canon_pct": canon_pct,
        "per_word_top": per_word_top,
        "per_word_unary_pct": per_word_unary_pct,
    }


# ==============================================================================
# Bootstrap M2.
# ==============================================================================
def gate_accuracy(X_train, y_train, X_test, y_test) -> float:
    clf = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs")
    clf.fit(X_train, y_train)
    return float(clf.score(X_test, y_test))


def bootstrap_gate(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    n_boot: int = N_BOOTSTRAP, seed: int = 0,
) -> dict:
    """Stim-resample bootstrap on M2 gate accuracy.

    Resamples (with replacement) within each canonical class on the
    TRAINING side; test set fixed. Refits the 5-class probe each
    iteration. Reports mean, 95% CI (percentile), and the bootstrap
    probability of exceeding the GATE_THRESHOLD.
    """
    rng = np.random.RandomState(seed)
    classes = np.unique(y_train)
    class_idx = {c: np.where(y_train == c)[0] for c in classes}

    boots = np.zeros(n_boot, dtype=np.float64)
    for b in range(n_boot):
        resampled = []
        for c in classes:
            idx = class_idx[c]
            sample = rng.choice(idx, size=len(idx), replace=True)
            resampled.append(sample)
        sel = np.concatenate(resampled)
        boots[b] = gate_accuracy(X_train[sel], y_train[sel], X_test, y_test)

    point = gate_accuracy(X_train, y_train, X_test, y_test)
    ci_low, ci_high = np.percentile(boots, [2.5, 97.5])
    p_pass = float(np.mean(boots >= GATE_THRESHOLD))
    return {
        "point": point,
        "mean": float(np.mean(boots)),
        "std": float(np.std(boots)),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "p_pass_065": p_pass,
        "samples": boots,
    }


def confusion_matrix_5class(
    X_train, y_train, X_test, y_test, classes: list[str],
) -> np.ndarray:
    clf = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs")
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    for true, pred in zip(y_test, preds):
        i = classes.index(true)
        j = classes.index(pred)
        cm[i, j] += 1
    return cm


def m2_arity_from_5class_probe(
    X_train, y_train, X_test, y_test,
) -> float:
    """M2-arity: train a 5-class canonical probe, but score on the
    coarsened binary-vs-unary partition. Tests whether the canonical
    probe's predictions preserve arity class membership across the
    notation shift, even when 5-class accuracy is low.

    A high M2-arity with a low M2-canonical indicates partial transfer:
    the arity axis transfers but binary-canonical identity is lost.
    """
    clf = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs").fit(X_train, y_train)
    preds = clf.predict(X_test)
    n_agree = 0
    for true, pred in zip(y_test, preds):
        if CANONICAL_ARITY[str(true)] == CANONICAL_ARITY[str(pred)]:
            n_agree += 1
    return n_agree / len(y_test)


def bootstrap_m2_arity(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    n_boot: int = N_BOOTSTRAP, seed: int = 0,
) -> dict:
    """Stim-resample bootstrap on M2-arity. Same protocol as
    bootstrap_gate but scores on the coarsened binary-vs-unary
    partition."""
    rng = np.random.RandomState(seed)
    classes = np.unique(y_train)
    class_idx = {c: np.where(y_train == c)[0] for c in classes}

    boots = np.zeros(n_boot, dtype=np.float64)
    for b in range(n_boot):
        resampled = []
        for c in classes:
            idx = class_idx[c]
            sample = rng.choice(idx, size=len(idx), replace=True)
            resampled.append(sample)
        sel = np.concatenate(resampled)
        boots[b] = m2_arity_from_5class_probe(
            X_train[sel], y_train[sel], X_test, y_test
        )

    point = m2_arity_from_5class_probe(X_train, y_train, X_test, y_test)
    ci_low, ci_high = np.percentile(boots, [2.5, 97.5])
    return {
        "point": point,
        "mean": float(np.mean(boots)),
        "std": float(np.std(boots)),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
    }


def print_confusion(cm: np.ndarray, classes: list[str], header: str) -> None:
    print(f"    {header}")
    print(f"      pred ->  | " + " | ".join(f"{c:<12}" for c in classes))
    for i, c in enumerate(classes):
        row = " | ".join(f"{cm[i, j]:<12}" for j in range(len(classes)))
        print(f"      true {c:<5} | {row}")


# ==============================================================================
# Target cells.
# ==============================================================================
@dataclass
class Cell:
    name: str
    model: str
    train_source: str  # "21" or "19b"
    train_cond: str    # "NEUTRAL" or "FUNC-PFX"
    train_anchor: str | None  # for 21 only
    train_layer: int
    test_source: str
    test_cond: str
    test_anchor: str | None
    test_layer: int
    expected_m2: float  # from prior runs
    cell_role: str     # "primary-candidate", "neg-control", "pos-control", "threshold-boundary", "lucky-default"


TARGET_CELLS: list[Cell] = [
    # ===== Primary candidate from §3.7.9 - exact script-21 anchor mapping =====
    # Script 21 forces FUNC-PFX `close-paren` to map to NEUTRAL `sentence-final`
    # since NEUTRAL has no close-paren. This is the cell that reported M2=0.616
    # with M4b=90% in script 21.
    Cell(
        name="[CANDIDATE] OLMo sf->cp L10 N->F (script-21 §3.7.9 cell)",
        model="OLMo 2 7B",
        train_source="21", train_cond="NEUTRAL", train_anchor="sentence-final", train_layer=10,
        test_source="21", test_cond="FUNC-PFX", test_anchor="close-paren", test_layer=10,
        expected_m2=0.616,
        cell_role="primary-candidate",
    ),
    # ===== Same test-anchor but train at operator-after (the canonical-rich
    # NEUTRAL position). This was mistakenly run as the "primary" in v1; turns
    # out to be a MUCH stronger cell (M2=0.924) and worth investigating.
    Cell(
        name="[ALT-TRAIN] OLMo opAft->cp L10 N->F (corrected: NEUT op-after train)",
        model="OLMo 2 7B",
        train_source="21", train_cond="NEUTRAL", train_anchor="operator-after", train_layer=10,
        test_source="21", test_cond="FUNC-PFX", test_anchor="close-paren", test_layer=10,
        expected_m2=0.924,
        cell_role="alt-train",
    ),
    # ===== Reverse direction =====
    Cell(
        name="[REV] OLMo cp->sf L10 F->N (reverse of §3.7.9, known-FAIL)",
        model="OLMo 2 7B",
        train_source="21", train_cond="FUNC-PFX", train_anchor="close-paren", train_layer=10,
        test_source="21", test_cond="NEUTRAL", test_anchor="sentence-final", test_layer=10,
        expected_m2=0.332,
        cell_role="neg-control",
    ),
    Cell(
        name="[REV] OLMo cp->opAft L10 F->N (reverse of alt-train)",
        model="OLMo 2 7B",
        train_source="21", train_cond="FUNC-PFX", train_anchor="close-paren", train_layer=10,
        test_source="21", test_cond="NEUTRAL", test_anchor="operator-after", test_layer=10,
        expected_m2=0.668,
        cell_role="neg-control",
    ),
    # ===== Positive control =====
    Cell(
        name="[POS] OLMo opAft->opAft L10 N->F (known PASS)",
        model="OLMo 2 7B",
        train_source="21", train_cond="NEUTRAL", train_anchor="operator-after", train_layer=10,
        test_source="21", test_cond="FUNC-PFX", test_anchor="operator-after", test_layer=10,
        expected_m2=0.800,
        cell_role="pos-control",
    ),
    # ===== Threshold boundary case from 19b v2 =====
    Cell(
        name="[BOUNDARY] Gemma opAft->opAft L17 F->N (boundary at 0.652)",
        model="Gemma 2 9B",
        train_source="21", train_cond="FUNC-PFX", train_anchor="operator-after", train_layer=17,
        test_source="21", test_cond="NEUTRAL", test_anchor="operator-after", test_layer=17,
        expected_m2=0.652,
        cell_role="threshold-boundary",
    ),
    # ===== Lucky-default known-fails (should stay at floor under bootstrap) =====
    Cell(
        name="[LUCKY-NEG] OLMo opAft->opAft L7 F->N (lucky-default M4b=80%)",
        model="OLMo 2 7B",
        train_source="21", train_cond="FUNC-PFX", train_anchor="operator-after", train_layer=7,
        test_source="21", test_cond="NEUTRAL", test_anchor="operator-after", test_layer=7,
        expected_m2=0.276,
        cell_role="lucky-default-neg",
    ),
    Cell(
        name="[LUCKY-NEG] Gemma firstArg->sf L8 F->N (lucky-default M4b=73.6%)",
        model="Gemma 2 9B",
        train_source="21", train_cond="FUNC-PFX", train_anchor="first-arg", train_layer=8,
        test_source="21", test_cond="NEUTRAL", test_anchor="sentence-final", test_layer=8,
        expected_m2=0.200,
        cell_role="lucky-default-neg",
    ),
    # ===== Three additional PASS-arity cells identified by script 22b =====
    # Bootstrap CIs on M2-canonical AND M2-arity for each.
    Cell(
        name="[NEW-CAND] OLMo firstArg->opAft L7 F->N (dual M2 PASS)",
        model="OLMo 2 7B",
        train_source="21", train_cond="FUNC-PFX", train_anchor="first-arg", train_layer=7,
        test_source="21", test_cond="NEUTRAL", test_anchor="operator-after", test_layer=7,
        expected_m2=0.980,
        cell_role="new-candidate",
    ),
    Cell(
        name="[NEW-CAND] Gemma sf->opAft L4 N->F (Gemma reinstated, L4)",
        model="Gemma 2 9B",
        train_source="21", train_cond="NEUTRAL", train_anchor="sentence-final", train_layer=4,
        test_source="21", test_cond="FUNC-PFX", test_anchor="operator-after", test_layer=4,
        expected_m2=1.000,
        cell_role="new-candidate",
    ),
    Cell(
        name="[NEW-CAND] Gemma sf->firstArg L8 N->F (Gemma reinstated, L8)",
        model="Gemma 2 9B",
        train_source="21", train_cond="NEUTRAL", train_anchor="sentence-final", train_layer=8,
        test_source="21", test_cond="FUNC-PFX", test_anchor="first-arg", test_layer=8,
        expected_m2=1.000,
        cell_role="new-candidate",
    ),
]


# ==============================================================================
# Driver.
# ==============================================================================
def get_slice_from_cell(caches: dict, cell: Cell, side: str) -> tuple[np.ndarray, np.ndarray]:
    """side in {'train', 'test'}. Returns (X, y) for the requested side."""
    if side == "train":
        cond, anchor, layer = cell.train_cond, cell.train_anchor, cell.train_layer
    else:
        cond, anchor, layer = cell.test_cond, cell.test_anchor, cell.test_layer

    cache = caches[(cell.model, cond)]
    X = slice_21(cache, anchor, layer)
    y = cache["canonical_labels"]
    return X, y


def main() -> None:
    log_path = _setup_logging()
    print(f"Script 22a - bootstrap CI on M2 canonical-transfer gate")
    print(f"  N_BOOTSTRAP: {N_BOOTSTRAP}")
    print(f"  GATE_THRESHOLD: {GATE_THRESHOLD}")
    print(f"  Cache requirement: script-21 v3-multi-anchor caches")
    print()

    print("=" * 92)
    print("LOADING SCRIPT-21 CACHES")
    print("=" * 92)
    caches: dict[tuple[str, str], dict] = {}
    for model in ["OLMo 2 7B", "Gemma 2 9B"]:
        for cond in ["NEUTRAL", "FUNC-PFX"]:
            caches[(model, cond)] = load_21_cache(model, cond)

    print()
    print("=" * 92)
    print("BOOTSTRAP M2 CANONICAL-TRANSFER GATE (500 stim-resamples per cell)")
    print("=" * 92)
    print()
    print(f"  Cell role legend:")
    print(f"    primary-candidate    cross-notation arity-respecting transfer candidate from §3.7.9")
    print(f"    neg-control          known M2-FAIL reverse direction")
    print(f"    pos-control          known M2-PASS pairing")
    print(f"    threshold-boundary   M2 just above 0.65 in 19b v2; tests stability")
    print(f"    lucky-default-neg    high-M4b lucky-default cell; M2 should stay at floor")
    print()

    print(f"  {'cell':<55} | point | mean  | 95% CI         | P(>= {GATE_THRESHOLD:.2f}) | role")
    print(f"  {'-' * 120}")

    all_results: list[tuple[Cell, dict]] = []
    for cell in TARGET_CELLS:
        X_tr, y_tr = get_slice_from_cell(caches, cell, "train")
        X_te, y_te = get_slice_from_cell(caches, cell, "test")
        t0 = time.time()
        r = bootstrap_gate(X_tr, y_tr, X_te, y_te,
                           n_boot=N_BOOTSTRAP, seed=stable_seed(cell.name))
        elapsed = time.time() - t0
        all_results.append((cell, r))
        print(f"  {cell.name:<55} | {r['point']:.3f} | {r['mean']:.3f} | "
              f"[{r['ci95_low']:.3f}, {r['ci95_high']:.3f}] | "
              f"{r['p_pass_065']:>8.1%} | {cell.cell_role} ({elapsed:.1f}s)")

    print()
    print("=" * 92)
    print("M2-ARITY (coarsened 2-class binary-vs-unary readout from the 5-class probe)")
    print("=" * 92)
    print()
    print(f"  Train the same 5-class canonical probe used for M2, then score on the")
    print(f"  coarsened binary-vs-unary partition. M2-arity HIGH with M2 LOW indicates")
    print(f"  partial transfer: the arity axis transfers but binary-canonical identity")
    print(f"  is lost across the notation shift.")
    print()
    print(f"  {'cell':<55} | M2-cano | M2-arity | CI95 (M2-arity)")
    print(f"  {'-' * 110}")
    m2_arity_results: dict[str, dict] = {}
    for cell, r in all_results:
        X_tr, y_tr = get_slice_from_cell(caches, cell, "train")
        X_te, y_te = get_slice_from_cell(caches, cell, "test")
        ar = bootstrap_m2_arity(X_tr, y_tr, X_te, y_te,
                                n_boot=N_BOOTSTRAP, seed=stable_seed(cell.name, "arity"))
        m2_arity_results[cell.name] = ar
        print(f"  {cell.name:<55} | {r['point']:.3f}   | {ar['point']:.3f}    | "
              f"[{ar['ci95_low']:.3f}, {ar['ci95_high']:.3f}]")

    print()
    print("=" * 92)
    print("M4a/b/c AT CANDIDATE CELLS (invented_breakdown using the SAME train anchor)")
    print("=" * 92)
    print()
    print(f"  This re-runs script-21's invented_breakdown using each cell's actual")
    print(f"  train anchor on the canonical side. The original §3.7.9 candidate uses")
    print(f"  NEUTRAL `sentence-final` as the train anchor; the [ALT-TRAIN] cells use")
    print(f"  NEUTRAL `operator-after`. M4b differs at the two cells (per-word readout")
    print(f"  depends on which canonical-discrimination axis the model uses).")
    print()
    print(f"  {'cell':<55} | M4a   | M4b   | M4c   | per-word top (unary%)")
    print(f"  {'-' * 130}")
    m4_per_cell: dict[str, dict] = {}
    for cell, r in all_results:
        if cell.test_cond == "NEUTRAL":
            inv_cache = caches[(cell.model, "NEUTRAL")]
        else:
            inv_cache = caches[(cell.model, "FUNC-PFX")]
        inv_X = slice_21_invented(inv_cache, cell.test_anchor, cell.test_layer)
        inv_words = inv_cache["invented_word_per_stim"]
        X_tr, y_tr = get_slice_from_cell(caches, cell, "train")
        bd = invented_arity_breakdown(X_tr, y_tr, inv_X, inv_words)
        m4_per_cell[cell.name] = bd
        per_word_summary = " ".join(
            f"{w}=>{bd['per_word_top'][w]}({bd['per_word_unary_pct'][w]*100:.0f}%U)"
            for w in INVENTED_WORDS
        )
        print(f"  {cell.name:<55} | {bd['M4a']:>5.1%} | {bd['M4b']:>5.1%} | "
              f"{bd['M4c']:>5.2f} | {per_word_summary}")

    print()
    print("=" * 92)
    print("CONFUSION MATRIX AT POINT ESTIMATE (5x5, true class -> predicted class)")
    print("=" * 92)
    for cell, _ in all_results:
        if cell.cell_role not in ["primary-candidate", "alt-train", "neg-control",
                                  "threshold-boundary", "new-candidate"]:
            continue
        print()
        print(f"  {cell.name}")
        X_tr, y_tr = get_slice_from_cell(caches, cell, "train")
        X_te, y_te = get_slice_from_cell(caches, cell, "test")
        cm = confusion_matrix_5class(X_tr, y_tr, X_te, y_te, CANONICALS)
        print_confusion(cm, CANONICALS, header="")

    print()
    print("=" * 92)
    print("INTERPRETATION")
    print("=" * 92)
    primary = [(c, r) for c, r in all_results if c.cell_role == "primary-candidate"][0]
    alt_train = [(c, r) for c, r in all_results if c.cell_role == "alt-train"][0]
    cell, r = primary
    cell_alt, r_alt = alt_train
    bd_primary = m4_per_cell[cell.name]
    bd_alt = m4_per_cell[cell_alt.name]
    print()
    print(f"  PRIMARY CANDIDATE (§3.7.9, train @ NEUTRAL sentence-final):")
    print(f"    {cell.name}")
    print(f"    point M2:        {r['point']:.3f}")
    print(f"    bootstrap mean:  {r['mean']:.3f}  (std {r['std']:.3f})")
    print(f"    95% CI:          [{r['ci95_low']:.3f}, {r['ci95_high']:.3f}]")
    print(f"    P(M2 >= 0.65):   {r['p_pass_065']:.1%}")
    print(f"    M4a / M4b / M4c: {bd_primary['M4a']:.1%} / {bd_primary['M4b']:.1%} / "
          f"{bd_primary['M4c']:.2f}")
    print()
    print(f"  ALTERNATIVE CANDIDATE (train @ NEUTRAL operator-after):")
    print(f"    {cell_alt.name}")
    print(f"    point M2:        {r_alt['point']:.3f}")
    print(f"    bootstrap mean:  {r_alt['mean']:.3f}  (std {r_alt['std']:.3f})")
    print(f"    95% CI:          [{r_alt['ci95_low']:.3f}, {r_alt['ci95_high']:.3f}]")
    print(f"    P(M2 >= 0.65):   {r_alt['p_pass_065']:.1%}")
    print(f"    M4a / M4b / M4c: {bd_alt['M4a']:.1%} / {bd_alt['M4b']:.1%} / "
          f"{bd_alt['M4c']:.2f}")
    print()
    print(f"  These two cells differ ONLY in the NEUTRAL training anchor used for the")
    print(f"  5-class canonical probe. Both test on FUNC-PFX close-paren L10.")
    print()
    ar_primary = m2_arity_results[cell.name]
    ar_alt = m2_arity_results[cell_alt.name]
    print(f"  M2 vs M2-arity comparison at the candidate cells:")
    print(f"    PRIMARY (NEUT sf train):   M2={r['point']:.3f} "
          f"(bootstrap-PASS prob {r['p_pass_065']:.1%})   "
          f"M2-arity={ar_primary['point']:.3f} (CI [{ar_primary['ci95_low']:.3f}, "
          f"{ar_primary['ci95_high']:.3f}])")
    print(f"    ALT-TRAIN (NEUT op-aft):    M2={r_alt['point']:.3f} "
          f"(bootstrap-PASS prob {r_alt['p_pass_065']:.1%})   "
          f"M2-arity={ar_alt['point']:.3f} (CI [{ar_alt['ci95_low']:.3f}, "
          f"{ar_alt['ci95_high']:.3f}])")
    print()
    print(f"  At the PRIMARY cell, M2 - M2-arity gap = "
          f"{ar_primary['point'] - r['point']:+.3f}.")
    print(f"  At the ALT-TRAIN cell, M2 - M2-arity gap = "
          f"{ar_alt['point'] - r_alt['point']:+.3f}.")
    print(f"  Large positive gap = arity axis transfers but 5-class canonical identity does not.")
    print()
    if r["ci95_high"] >= GATE_THRESHOLD:
        if r["p_pass_065"] >= 0.5:
            print(f"  *** Bootstrap UPGRADE the verdict to 'PASS within statistical noise'")
            print(f"      (P(M2 >= {GATE_THRESHOLD:.2f}) = {r['p_pass_065']:.1%} >= 50% means more")
            print(f"      than half of bootstrap samples PASS the threshold).")
        else:
            print(f"  --- Bootstrap supports the AMBIG verdict: 95% CI extends across the")
            print(f"      0.65 threshold but the bootstrap is biased below ({r['p_pass_065']:.1%}")
            print(f"      of samples PASS). Threshold itself is fragile here.")
    else:
        print(f"  Bootstrap CONFIRMS the AMBIG verdict: 95% CI [")
        print(f"      {r['ci95_low']:.3f}, {r['ci95_high']:.3f}] is entirely below 0.65.")
        print(f"      The M2 gate at the §3.7.9 primary candidate cell is not just slightly")
        print(f"      below threshold; it's robustly below.")
    print()
    print(f"  IMPLICATION FOR §3.7.9:")
    if r["p_pass_065"] >= 0.5:
        print(f"    The candidate finding at OLMo 2 close-paren L10 N->F is upgraded.")
        print(f"    Update paper_notes §3.7.9 to reflect 'bootstrap-PASS within statistical")
        print(f"    noise' rather than 'AMBIG at the threshold'.")
    elif r["ci95_high"] >= GATE_THRESHOLD:
        print(f"    The candidate remains AMBIG but the threshold itself is fragile.")
        print(f"    Add the bootstrap CI to paper_notes §3.7.9; flag the threshold as a")
        print(f"    methodological concern for future M2 verdicts.")
    else:
        print(f"    The candidate stays AMBIG, robustly. The M4b = 90% per-word pattern")
        print(f"    is real but the underlying 5-class canonical-transfer does NOT pass")
        print(f"    bootstrap calibration. The candidate's status depends on whether")
        print(f"    'arity-respecting per-word transfer without canonical-transfer gate")
        print(f"    PASS' is a coherent finding. Discuss this in the paper_notes update.")

    # ====== New candidates (script 22b additions) ======
    new_cands = [(c, r) for c, r in all_results if c.cell_role == "new-candidate"]
    if new_cands:
        print()
        print("=" * 92)
        print("NEW PASS-ARITY CANDIDATES from script 22b sweep")
        print("=" * 92)
        print()
        for c, r in new_cands:
            ar = m2_arity_results[c.name]
            bd = m4_per_cell[c.name]
            print(f"  {c.name}")
            print(f"    Point M2-canonical: {r['point']:.3f} (bootstrap mean {r['mean']:.3f}, "
                  f"95% CI [{r['ci95_low']:.3f}, {r['ci95_high']:.3f}], "
                  f"P(>= {GATE_THRESHOLD:.2f}) = {r['p_pass_065']:.1%})")
            print(f"    Point M2-arity:     {ar['point']:.3f} (CI [{ar['ci95_low']:.3f}, "
                  f"{ar['ci95_high']:.3f}])")
            print(f"    M2-arity - M2-cano gap: {ar['point'] - r['point']:+.3f}")
            print(f"    M4a / M4b / M4c:    {bd['M4a']*100:.1f}% / {bd['M4b']*100:.1f}% / "
                  f"{bd['M4c']:.2f}")
            if r["p_pass_065"] >= 0.5 and ar["ci95_low"] >= GATE_THRESHOLD:
                print(f"    => CONFIRMED dual-PASS under bootstrap (M2-canonical AND M2-arity)")
            elif ar["ci95_low"] >= GATE_THRESHOLD and r["p_pass_065"] < 0.5:
                print(f"    => M2-arity PASS robust; M2-canonical AMBIG-or-below (§3.7.10-style dissociation)")
            elif r["p_pass_065"] >= 0.5:
                print(f"    => M2-canonical PASS robust; check M2-arity separately")
            else:
                print(f"    => BOTH gates below threshold under bootstrap; not a clean PASS")
            print()

    if log_path:
        print()
        print(f"[logging] full transcript written to: {log_path}")


if __name__ == "__main__":
    main()
