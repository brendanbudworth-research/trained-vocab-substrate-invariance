"""Script 22c - expanded invented-word set (16 invented words; 8 binary, 8 unary).

Phase 1 falsification test for the §3.7.11 / §3.7.12 finding: with only 5
invented words, the headline "cross-notation arity-respecting transfer" rests
on patterns like "4 of 5 invented words have predicted-canonical arity matching
intended arity" (the §3.7.9 cell) and "8 of 16 binary + 8 of 16 unary should
match if the model is truly tracking arity". A 5-word sample can sometimes
produce illusory arity tracking by chance + lucky-default. With 16 invented
words (8 intended-binary + 8 intended-unary), the M4b ≥ 0.65 threshold becomes
a much more discriminating test, and per-word patterns become statistically
meaningful.

Expanded invented set (16 words; original 5 kept for direct comparison):

  Intended-binary (8):
    bliq → and       (orig)
    dren → or        (orig)
    molex → implies  (orig)
    krev → and       (new)
    sond → or        (new)
    glin → implies   (new)
    twiv → and       (new)
    fump → or        (new)

  Intended-unary (8):
    vusp → not          (orig)
    perph → necessarily (orig)
    kelm → not          (new)
    zorf → not          (new)
    gleph → necessarily (new)
    drelth → necessarily (new)
    vrith → not         (new)
    nilph → necessarily (new)

Selection criteria for the 11 new words: phonotactically plausible English /
Latin-script tokens that avoid obvious morphemic semantic loadings (no Latin
roots, no "mole"/"x" style hidden semantics, no common English content-word
fragments). The tokenizer-level subword decomposition for each word is
printed at extraction time so subword-semantic confounds can be inspected.

Test cells: the four PASS-arity cells identified in §3.7.11 + bootstrap-
confirmed in §3.7.12. We re-run the M1-M4 + M2-arity battery at each cell
with the 16-word invented set and report:

  - M1, M2-canonical, M2-arity                 (same as 22a/22b; canonical
                                                stims unchanged, so M1 and M2
                                                should reproduce script 22b's
                                                values exactly)
  - M4a (invented unary mass over 16 × 50 = 800 invented stims)
  - M4b (intended-arity agreement over 800 invented stims)
  - M4c (Herfindahl over predicted-canonical fractions)
  - Per-word breakdown for all 16 words (top canonical, top %, intended arity,
    predicted arity, match)
  - Bootstrap M4b CI: B = 500, with-replacement resampling within each
    invented word (preserves the 8/8 binary/unary split)

Headline question: does M4b stay ≥ 0.65 at the 4 PASS-arity cells with 16
invented words? If yes at ≥ 3 of 4 cells, the cross-notation arity-respecting
transfer finding is real. If it drops to chance (M4b ≈ 0.50) at most cells,
the 4-of-5 pattern was a 5-word sampling artifact.

Cache: invented stimulus set differs from script 21's v3-multi-anchor, so
extraction is required. We bump STIMULUS_VERSION to v4-expanded-invented and
write a separate cache file. Canonical stimuli are identical to 21's, so
canonical activations could in principle be reloaded from 21's cache - but the
cache file structure combines canonical + invented, so we just re-extract both
and treat 22c's cache as standalone. Total extraction: 5 × 50 (canon) +
16 × 50 (invented) = 1050 stims per (model, condition) × 2 conditions × 2
models = 4200 forward passes. At ~0.5s per forward on MPS, expected runtime
~35 min (5-7× script 21's ~9 min).

Tees full output to outputs/22c_<ts>.log.
"""

from __future__ import annotations

import datetime as _dt
import gc
import hashlib
import os
import random
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModelForCausalLM, AutoTokenizer


# ==============================================================================
# Tee logging
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
    log_path = os.path.join(log_dir, f"22c_{ts}.log")
    log_f = open(log_path, "w", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_f)
    sys.stderr = _Tee(sys.__stderr__, log_f)
    print(f"[logging] tee'ing all output to {log_path}")
    return log_path


# ==============================================================================
# Constants
# ==============================================================================
SEED = 17
N_PER_CLASS = 50
N_BOOTSTRAP = 500
GATE_THRESHOLD = 0.65
M4B_PASS_THRESHOLD = 0.65

STIMULUS_VERSION = "v4-expanded-invented"

CANONICALS = ["and", "or", "not", "implies", "necessarily"]
UNARY_CANONICALS = ["not", "necessarily"]
BINARY_CANONICALS = ["and", "or", "implies"]
CANONICAL_ARITY = {"and": 2, "or": 2, "implies": 2, "not": 1, "necessarily": 1}

# Expanded invented set: 16 words, 8 intended-binary + 8 intended-unary.
INVENTED_WORDS = [
    "bliq", "dren", "molex",                    # original binary (3)
    "krev", "sond", "glin", "twiv", "fump",     # new binary (5)
    "vusp", "perph",                            # original unary (2)
    "kelm", "zorf", "gleph", "drelth", "vrith", "nilph",  # new unary (6)
]
W_TO_CANONICAL = {
    # binary intended:
    "bliq": "and", "dren": "or", "molex": "implies",
    "krev": "and", "sond": "or", "glin": "implies",
    "twiv": "and", "fump": "or",
    # unary intended:
    "vusp": "not", "perph": "necessarily",
    "kelm": "not", "zorf": "not",
    "gleph": "necessarily", "drelth": "necessarily",
    "vrith": "not", "nilph": "necessarily",
}
assert set(W_TO_CANONICAL.keys()) == set(INVENTED_WORDS)
assert sum(1 for w in INVENTED_WORDS if CANONICAL_ARITY[W_TO_CANONICAL[w]] == 2) == 8
assert sum(1 for w in INVENTED_WORDS if CANONICAL_ARITY[W_TO_CANONICAL[w]] == 1) == 8

ORIGINAL_5 = ["bliq", "dren", "vusp", "molex", "perph"]
NEW_11 = [w for w in INVENTED_WORDS if w not in ORIGINAL_5]
assert len(NEW_11) == 11

# Anchor definitions (same as 21)
ANCHORS_FUNC_PFX = ["operator-after", "first-arg", "close-paren", "sentence-final"]
ANCHORS_NEUTRAL = ["operator-after", "sentence-final"]


# ==============================================================================
# Stable seeding helpers (same as 19b/21).
# ==============================================================================
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


# ==============================================================================
# Stimulus generation - patch the 19b module's INVENTED_WORDS / W_TO_CANONICAL
# in place so make_functional_invented_stimuli uses our expanded set.
# ==============================================================================
def _load_19b_module():
    import importlib.machinery
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "19b_directional_angle_gated.py")
    loader = importlib.machinery.SourceFileLoader("_m19b_22c", path)
    spec = importlib.util.spec_from_loader("_m19b_22c", loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_m19b_22c"] = mod
    loader.exec_module(mod)
    return mod


_M19B = _load_19b_module()
# Patch the 19b module's globals before any stimulus generation.
_M19B.INVENTED_WORDS = INVENTED_WORDS
_M19B.W_TO_CANONICAL = W_TO_CANONICAL
make_neutral_stimuli = _M19B.make_neutral_stimuli
make_functional_canonical_stimuli = _M19B.make_functional_canonical_stimuli
make_functional_invented_stimuli = _M19B.make_functional_invented_stimuli
assert _M19B.SEED == SEED
assert _M19B.N_PER_CLASS == N_PER_CLASS


def _generate_prompts(name: str) -> tuple[list[str], list[str], list[str], list[str]]:
    """Generate canonical + invented prompts for one condition.
    Uses stable_seed; identical to 19b's logic but patches in expanded
    invented set via the monkey-patch above."""
    if name == "NEUTRAL":
        canonical_stim_fn = make_neutral_stimuli
        invented_stim_fn = make_neutral_stimuli
    elif name == "FUNC-PFX":
        canonical_stim_fn = make_functional_canonical_stimuli
        invented_stim_fn = make_functional_invented_stimuli
    else:
        raise ValueError(name)

    canon_prompts: list[str] = []
    canon_labels: list[str] = []
    for op in CANONICALS:
        op_rng = random.Random(stable_seed(name, "canon", op))
        canon_prompts.extend(canonical_stim_fn(op, op_rng, N_PER_CLASS))
        canon_labels.extend([op] * N_PER_CLASS)

    inv_prompts: list[str] = []
    inv_words: list[str] = []
    for w in INVENTED_WORDS:
        w_rng = random.Random(stable_seed(name, "inv", w))
        inv_prompts.extend(invented_stim_fn(w, w_rng, N_PER_CLASS))
        inv_words.extend([w] * N_PER_CLASS)

    return canon_prompts, canon_labels, inv_prompts, inv_words


# ==============================================================================
# Model spec (same as 21)
# ==============================================================================
@dataclass
class ModelSpec:
    short_name: str
    model_id: str
    dtype: "torch.dtype"
    focus_layers: list[int]


MODEL_SPECS: list[ModelSpec] = [
    ModelSpec(
        short_name="Gemma 2 9B",
        model_id="google/gemma-2-9b",
        dtype=torch.bfloat16,
        focus_layers=[2, 4, 8, 16, 17],
    ),
    ModelSpec(
        short_name="OLMo 2 7B",
        model_id="allenai/OLMo-2-1124-7B",
        dtype=torch.float16,
        focus_layers=[4, 7, 10, 16, 24],
    ),
]


# ==============================================================================
# Anchor finder + multi-anchor extraction - import from script 21.
# ==============================================================================
def _load_21_module():
    import importlib.machinery
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "21_multi_anchor_battery.py")
    loader = importlib.machinery.SourceFileLoader("_m21_22c", path)
    spec = importlib.util.spec_from_loader("_m21_22c", loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_m21_22c"] = mod
    loader.exec_module(mod)
    return mod


_M21 = _load_21_module()
find_operator_subword_idx = _M21.find_operator_subword_idx
compute_anchor_positions = _M21.compute_anchor_positions
extract_multi_anchor_activations = _M21.extract_multi_anchor_activations
ConditionMultiAnchor = _M21.ConditionMultiAnchor


# ==============================================================================
# Cache (v4 = standalone from 21's v3 cache)
# ==============================================================================
def _cache_path(model_short_name: str, condition_name: str) -> str:
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "outputs", "cache")
    os.makedirs(base, exist_ok=True)
    slug = model_short_name.replace(" ", "_")
    return os.path.join(
        base,
        f"22c_{slug}_{condition_name}_npc{N_PER_CLASS}_{STIMULUS_VERSION}.npz",
    )


def _cache_save(
    path: str, cond: ConditionMultiAnchor,
    *, model_id: str, condition_name: str,
    canon_prompts_hash: str, inv_prompts_hash: str,
    dtype_before_cache: str,
) -> None:
    canon_stack = cond.canonical_X.astype(np.float16)
    inv_stack = cond.invented_X.astype(np.float16)
    np.savez_compressed(
        path,
        canonical_X=canon_stack,
        canonical_labels=cond.canonical_labels,
        invented_X=inv_stack,
        invented_word_per_stim=cond.invented_word_per_stim,
        anchor_names=np.array(cond.anchor_names),
        canon_anchor_positions=np.array(cond.canon_anchor_positions),
        inv_anchor_positions=np.array(cond.inv_anchor_positions),
        meta_stimulus_version=np.array([STIMULUS_VERSION]),
        meta_model_id=np.array([model_id]),
        meta_condition=np.array([condition_name]),
        meta_canon_prompts_hash=np.array([canon_prompts_hash]),
        meta_inv_prompts_hash=np.array([inv_prompts_hash]),
        meta_dtype_before_cache=np.array([dtype_before_cache]),
        n_per_class=np.array([N_PER_CLASS]),
        invented_word_list=np.array(INVENTED_WORDS),
    )
    size_mb = os.path.getsize(path) / 1e6
    print(f"    [cache] saved {os.path.basename(path)} ({size_mb:.1f} MB)")


def _cache_load(
    path: str, *,
    expected_canon_hash: str, expected_inv_hash: str,
    expected_anchors: list[str],
) -> ConditionMultiAnchor | None:
    if not os.path.exists(path):
        return None
    try:
        z = np.load(path, allow_pickle=False)
        required = [
            "n_per_class", "meta_stimulus_version", "anchor_names",
            "meta_canon_prompts_hash", "meta_inv_prompts_hash",
            "invented_word_list",
        ]
        for k in required:
            if k not in z.files:
                return None
        if int(z["n_per_class"][0]) != N_PER_CLASS:
            return None
        if str(z["meta_stimulus_version"][0]) != STIMULUS_VERSION:
            return None
        if str(z["meta_canon_prompts_hash"][0]) != expected_canon_hash:
            return None
        if str(z["meta_inv_prompts_hash"][0]) != expected_inv_hash:
            return None
        cached_anchors = list(z["anchor_names"])
        if cached_anchors != expected_anchors:
            return None
        cached_words = list(z["invented_word_list"])
        if cached_words != INVENTED_WORDS:
            return None
        canon_X = z["canonical_X"].astype(np.float32)
        inv_X = z["invented_X"].astype(np.float32)
        cond = ConditionMultiAnchor(
            canonical_X=canon_X,
            canonical_labels=z["canonical_labels"],
            invented_X=inv_X,
            invented_word_per_stim=z["invented_word_per_stim"],
            anchor_names=cached_anchors,
            canon_anchor_positions=z["canon_anchor_positions"].tolist(),
            inv_anchor_positions=z["inv_anchor_positions"].tolist(),
        )
        print(f"    [cache] hit {os.path.basename(path)} "
              f"shape={canon_X.shape}, inv={inv_X.shape}")
        return cond
    except Exception as e:
        print(f"    [cache] failed to load {path}: {e}")
        return None


# ==============================================================================
# Build / load per-(model, condition) multi-anchor activations.
# ==============================================================================
def build_condition(
    spec: ModelSpec, condition_name: str, anchor_names: list[str],
    model, tok, device: str,
) -> ConditionMultiAnchor:
    canon_prompts, canon_labels, inv_prompts, inv_words = _generate_prompts(condition_name)
    canon_hash = prompts_checksum(canon_prompts)
    inv_hash = prompts_checksum(inv_prompts)

    path = _cache_path(spec.short_name, condition_name)
    cached = _cache_load(
        path,
        expected_canon_hash=canon_hash, expected_inv_hash=inv_hash,
        expected_anchors=anchor_names,
    )
    if cached is not None:
        return cached

    assert model is not None, "cache miss but model not loaded"
    print(f"\n  Building condition: {condition_name} "
          f"(anchors={anchor_names}; extracting)")

    print(f"    extracting canonical activations ({len(canon_prompts)} stim)...")
    t0 = time.time()
    canon_X, canon_pos, n_layers = extract_multi_anchor_activations(
        model, tok, canon_prompts, CANONICALS, anchor_names, condition_name, device
    )
    print(f"      {time.time() - t0:.1f}s, shape={canon_X.shape}")

    print(f"    extracting invented activations ({len(inv_prompts)} stim)...")
    t0 = time.time()
    inv_X, inv_pos, _ = extract_multi_anchor_activations(
        model, tok, inv_prompts, INVENTED_WORDS, anchor_names, condition_name, device
    )
    print(f"      {time.time() - t0:.1f}s, shape={inv_X.shape}")

    cond = ConditionMultiAnchor(
        canonical_X=canon_X,
        canonical_labels=np.array(canon_labels),
        invented_X=inv_X,
        invented_word_per_stim=np.array(inv_words),
        anchor_names=list(anchor_names),
        canon_anchor_positions=canon_pos,
        inv_anchor_positions=inv_pos,
    )
    _cache_save(
        path, cond,
        model_id=spec.model_id, condition_name=condition_name,
        canon_prompts_hash=canon_hash, inv_prompts_hash=inv_hash,
        dtype_before_cache=str(spec.dtype).replace("torch.", ""),
    )
    return cond


# ==============================================================================
# Cell + metric definitions
# ==============================================================================
@dataclass
class TestCell:
    """One PASS-arity cell to test under the expanded invented set."""
    name: str
    model_short: str
    train_cond: str
    train_anchor: str
    train_layer: int
    test_cond: str
    test_anchor: str
    test_layer: int
    expected_m4b_5word: float  # M4b under the original 5-word set


# §3.7.11/§3.7.12 PASS-arity cells + 2 lucky-default negative controls.
TEST_CELLS: list[TestCell] = [
    TestCell(
        name="§3.7.9: OLMo sf->cp L10 N->F",
        model_short="OLMo 2 7B",
        train_cond="NEUTRAL", train_anchor="sentence-final", train_layer=10,
        test_cond="FUNC-PFX", test_anchor="close-paren", test_layer=10,
        expected_m4b_5word=0.900,
    ),
    TestCell(
        name="Gemma sf->opa L4 N->F",
        model_short="Gemma 2 9B",
        train_cond="NEUTRAL", train_anchor="sentence-final", train_layer=4,
        test_cond="FUNC-PFX", test_anchor="operator-after", test_layer=4,
        expected_m4b_5word=0.728,
    ),
    TestCell(
        name="Gemma sf->first L8 N->F",
        model_short="Gemma 2 9B",
        train_cond="NEUTRAL", train_anchor="sentence-final", train_layer=8,
        test_cond="FUNC-PFX", test_anchor="first-arg", test_layer=8,
        expected_m4b_5word=0.736,
    ),
    TestCell(
        name="OLMo first->opa L7 F->N",
        model_short="OLMo 2 7B",
        train_cond="FUNC-PFX", train_anchor="first-arg", train_layer=7,
        test_cond="NEUTRAL", test_anchor="operator-after", test_layer=7,
        expected_m4b_5word=0.656,
    ),
    # Lucky-default negative controls (should stay at floor with 16 words)
    TestCell(
        name="[LUCKY-NEG] OLMo opa->opa L7 F->N",
        model_short="OLMo 2 7B",
        train_cond="FUNC-PFX", train_anchor="operator-after", train_layer=7,
        test_cond="NEUTRAL", test_anchor="operator-after", test_layer=7,
        expected_m4b_5word=0.800,  # 5-word lucky-default
    ),
    TestCell(
        name="[LUCKY-NEG] Gemma first->sf L8 F->N",
        model_short="Gemma 2 9B",
        train_cond="FUNC-PFX", train_anchor="first-arg", train_layer=8,
        test_cond="NEUTRAL", test_anchor="sentence-final", test_layer=8,
        expected_m4b_5word=0.736,  # 5-word lucky-default
    ),
]


# ==============================================================================
# M1-M4 primitives.
# ==============================================================================
def m1_cv(X: np.ndarray, y: np.ndarray) -> float:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    accs = []
    for tr_idx, te_idx in skf.split(X, y):
        clf = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs").fit(X[tr_idx], y[tr_idx])
        accs.append(clf.score(X[te_idx], y[te_idx]))
    return float(np.mean(accs))


def m2_metrics(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
) -> tuple[float, float, np.ndarray, LogisticRegression]:
    """Returns (M2-canonical, M2-arity, test_preds, fitted_classifier)."""
    clf = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs").fit(X_train, y_train)
    preds = clf.predict(X_test)
    m2_cano = float(np.mean(preds == y_test))
    n_ar = sum(
        CANONICAL_ARITY[str(t)] == CANONICAL_ARITY[str(p)]
        for t, p in zip(y_test, preds)
    )
    m2_arity = n_ar / len(y_test)
    return m2_cano, m2_arity, preds, clf


@dataclass
class M4Result:
    m4a: float  # invented unary mass
    m4b: float  # intended-arity agreement
    m4c: float  # Herfindahl over predicted canonicals
    breakdown_pct: dict[str, float]
    per_word_top: dict[str, str]
    per_word_top_pct: dict[str, float]
    per_word_unary_pct: dict[str, float]
    n_intended_binary_correct: int
    n_intended_unary_correct: int


def m4_breakdown(
    clf: LogisticRegression, X_inv: np.ndarray, words: np.ndarray,
) -> M4Result:
    preds = clf.predict(X_inv)
    n_total = len(preds)

    # Overall predicted-canonical breakdown.
    breakdown_pct: dict[str, float] = {}
    for c in CANONICALS:
        breakdown_pct[c] = float(np.mean(preds == c))

    # M4a: invented unary mass = predicted in {not, necessarily}.
    m4a = sum(breakdown_pct[c] for c in UNARY_CANONICALS)

    # M4c: Herfindahl over breakdown_pct.
    m4c = float(sum(v ** 2 for v in breakdown_pct.values()))

    # M4b: per-stimulus intended-arity agreement.
    n_match = 0
    n_intended_binary_correct = 0
    n_intended_unary_correct = 0
    for w, p in zip(words, preds):
        intended_can = W_TO_CANONICAL[str(w)]
        if CANONICAL_ARITY[intended_can] == CANONICAL_ARITY[str(p)]:
            n_match += 1
            if CANONICAL_ARITY[intended_can] == 2:
                n_intended_binary_correct += 1
            else:
                n_intended_unary_correct += 1
    m4b = n_match / n_total

    # Per-word top canonical + within-word concentration.
    per_word_top: dict[str, str] = {}
    per_word_top_pct: dict[str, float] = {}
    per_word_unary_pct: dict[str, float] = {}
    for w in INVENTED_WORDS:
        mask = (words == w)
        if not mask.any():
            continue
        word_preds = preds[mask]
        counts = {c: int(np.sum(word_preds == c)) for c in CANONICALS}
        top_c = max(counts, key=lambda c: counts[c])
        per_word_top[w] = top_c
        per_word_top_pct[w] = counts[top_c] / mask.sum()
        per_word_unary_pct[w] = sum(counts[c] for c in UNARY_CANONICALS) / mask.sum()

    return M4Result(
        m4a=m4a, m4b=m4b, m4c=m4c,
        breakdown_pct=breakdown_pct,
        per_word_top=per_word_top,
        per_word_top_pct=per_word_top_pct,
        per_word_unary_pct=per_word_unary_pct,
        n_intended_binary_correct=n_intended_binary_correct,
        n_intended_unary_correct=n_intended_unary_correct,
    )


def bootstrap_m4b(
    clf: LogisticRegression, X_inv: np.ndarray, words: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
) -> dict[str, float]:
    """Bootstrap CI on M4b: resample with replacement within each invented
    word. Preserves the 8/8 binary/unary split."""
    preds = clf.predict(X_inv)
    word_to_idx = {w: np.where(words == w)[0] for w in INVENTED_WORDS}

    m4b_samples = []
    rng = np.random.default_rng(SEED)
    for _ in range(n_bootstrap):
        idx_list = []
        for w in INVENTED_WORDS:
            w_idx = word_to_idx[w]
            sample = rng.choice(w_idx, size=len(w_idx), replace=True)
            idx_list.append(sample)
        all_idx = np.concatenate(idx_list)
        sub_preds = preds[all_idx]
        sub_words = words[all_idx]
        n_match = sum(
            CANONICAL_ARITY[W_TO_CANONICAL[str(w)]] == CANONICAL_ARITY[str(p)]
            for w, p in zip(sub_words, sub_preds)
        )
        m4b_samples.append(n_match / len(sub_preds))

    arr = np.array(m4b_samples)
    return {
        "point": None,  # filled in by the caller from the M4Result it just computed
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "ci95_low": float(np.percentile(arr, 2.5)),
        "ci95_high": float(np.percentile(arr, 97.5)),
        "p_pass_065": float(np.mean(arr >= M4B_PASS_THRESHOLD)),
    }


def detect_lucky_default(per_word_top_pct: dict[str, float]) -> bool:
    """Same detector as 22b: lucky-default if every word has within-word
    top concentration >= 0.95 (deterministic single-canonical mapping)."""
    pcts = list(per_word_top_pct.values())
    if not pcts:
        return False
    return min(pcts) >= 0.95


# ==============================================================================
# Slice helpers - take (anchor, layer) slice from ConditionMultiAnchor
# ==============================================================================
def slice_canonical(
    cond: ConditionMultiAnchor, anchor: str, layer: int,
) -> tuple[np.ndarray, np.ndarray]:
    a_idx = cond.anchor_names.index(anchor)
    X = cond.canonical_X[a_idx, :, layer, :]
    return X, np.asarray(cond.canonical_labels)


def slice_invented(
    cond: ConditionMultiAnchor, anchor: str, layer: int,
) -> tuple[np.ndarray, np.ndarray]:
    a_idx = cond.anchor_names.index(anchor)
    X = cond.invented_X[a_idx, :, layer, :]
    return X, np.asarray(cond.invented_word_per_stim)


# ==============================================================================
# Cell evaluation
# ==============================================================================
@dataclass
class CellResult:
    name: str
    m1_train: float
    m1_test: float
    m2_canonical: float
    m2_arity: float
    m4: M4Result
    bootstrap_m4b: dict[str, float]
    lucky_default: bool


def evaluate_cell(
    cell: TestCell,
    train_cond: ConditionMultiAnchor, test_cond: ConditionMultiAnchor,
) -> CellResult:
    X_train, y_train = slice_canonical(train_cond, cell.train_anchor, cell.train_layer)
    X_test, y_test = slice_canonical(test_cond, cell.test_anchor, cell.test_layer)
    X_inv, w_inv = slice_invented(test_cond, cell.test_anchor, cell.test_layer)

    m1_train = m1_cv(X_train, y_train)
    m1_test = m1_cv(X_test, y_test)
    m2_cano, m2_arity, _, clf = m2_metrics(X_train, y_train, X_test, y_test)
    m4 = m4_breakdown(clf, X_inv, w_inv)
    bs = bootstrap_m4b(clf, X_inv, w_inv)
    bs["point"] = m4.m4b  # overwrite the placeholder with the actual point
    lucky = detect_lucky_default(m4.per_word_top_pct)

    return CellResult(
        name=cell.name,
        m1_train=m1_train, m1_test=m1_test,
        m2_canonical=m2_cano, m2_arity=m2_arity,
        m4=m4, bootstrap_m4b=bs, lucky_default=lucky,
    )


# ==============================================================================
# Reporting
# ==============================================================================
def print_cell_result(cell: TestCell, res: CellResult) -> None:
    print()
    print("=" * 92)
    print(f"  {cell.name}")
    print(f"  model={cell.model_short}  "
          f"train={cell.train_cond}/{cell.train_anchor}@L{cell.train_layer}  "
          f"test={cell.test_cond}/{cell.test_anchor}@L{cell.test_layer}")
    print("=" * 92)
    print()
    print(f"  M1-train CV:   {res.m1_train:.3f}")
    print(f"  M1-test  CV:   {res.m1_test:.3f}")
    print(f"  M2-canonical:  {res.m2_canonical:.3f}  (5-class accuracy on test canonicals)")
    print(f"  M2-arity:      {res.m2_arity:.3f}  (binary-vs-unary coarsened)")
    print()
    print(f"  M4a (inv unary mass):        {res.m4.m4a*100:5.1f}%")
    print(f"  M4b (intended-arity agree):  {res.m4.m4b*100:5.1f}%   "
          f"(expected at 5-word {cell.expected_m4b_5word*100:.1f}%)")
    print(f"  M4c (Herfindahl):            {res.m4.m4c:5.2f}   "
          f"(0.20 = uniform across 5 canonicals)")
    print()
    print(f"  Bootstrap M4b: mean {res.bootstrap_m4b['mean']:.3f} "
          f"std {res.bootstrap_m4b['std']:.3f} "
          f"95% CI [{res.bootstrap_m4b['ci95_low']:.3f}, "
          f"{res.bootstrap_m4b['ci95_high']:.3f}]  "
          f"P(M4b >= {M4B_PASS_THRESHOLD:.2f}) = {res.bootstrap_m4b['p_pass_065']:.1%}")
    print()
    print(f"  Breakdown by predicted canonical (over 800 invented stims):")
    for c in CANONICALS:
        bar = int(res.m4.breakdown_pct[c] * 40)
        print(f"    {c:13s} {res.m4.breakdown_pct[c]*100:5.1f}%  {'#' * bar}")
    print()
    print(f"  Per-word breakdown (16 words, sorted by intended arity):")
    print(f"    {'word':<10} {'int.':>4} {'int.cano':<11} {'top':<13} {'top%':>5} {'unary%':>6} {'arity match':<5}")
    print(f"    {'-'*10} {'-'*4} {'-'*11} {'-'*13} {'-'*5} {'-'*6} {'-'*11}")
    for w in INVENTED_WORDS:
        intended_can = W_TO_CANONICAL[w]
        intended_arity = "B" if CANONICAL_ARITY[intended_can] == 2 else "U"
        top_c = res.m4.per_word_top.get(w, "—")
        top_pct = res.m4.per_word_top_pct.get(w, 0.0)
        unary_pct = res.m4.per_word_unary_pct.get(w, 0.0)
        pred_arity_match = (
            CANONICAL_ARITY[top_c] == CANONICAL_ARITY[intended_can]
            if top_c in CANONICAL_ARITY else False
        )
        match_str = "YES" if pred_arity_match else "no"
        is_orig = " (orig)" if w in ORIGINAL_5 else ""
        print(f"    {w:<10} {intended_arity:>4} {intended_can:<11} {top_c:<13} "
              f"{top_pct*100:5.1f} {unary_pct*100:5.1f}  {match_str:<5}{is_orig}")

    print()
    print(f"  Per-intended-arity correct counts:")
    print(f"    intended-binary correct (predicted binary): "
          f"{res.m4.n_intended_binary_correct}/{8 * N_PER_CLASS}")
    print(f"    intended-unary  correct (predicted unary):  "
          f"{res.m4.n_intended_unary_correct}/{8 * N_PER_CLASS}")
    print()
    print(f"  Lucky-default flag: {'YES (all per-word top_pct >= 0.95)' if res.lucky_default else 'no'}")
    print()


def print_summary_table(cells: list[TestCell], results: list[CellResult]) -> None:
    print()
    print("=" * 92)
    print("SUMMARY: M4b at the 4 PASS-arity cells + 2 lucky-default controls")
    print("=" * 92)
    print()
    print(f"  {'cell':<48} {'M4b(5w)':>8} {'M4b(16w)':>8} {'CI95':<20} {'lucky?':>7}")
    print(f"  {'-'*48} {'-'*8} {'-'*8} {'-'*20} {'-'*7}")
    for cell, res in zip(cells, results):
        ci = f"[{res.bootstrap_m4b['ci95_low']:.3f}, {res.bootstrap_m4b['ci95_high']:.3f}]"
        lucky_str = "YES" if res.lucky_default else "no"
        print(f"  {cell.name[:48]:<48} {cell.expected_m4b_5word:8.3f} "
              f"{res.m4.m4b:8.3f} {ci:<20} {lucky_str:>7}")
    print()

    pass_arity_cells = cells[:4]
    pass_arity_results = results[:4]
    n_still_pass = sum(1 for r in pass_arity_results
                       if r.bootstrap_m4b["ci95_low"] >= M4B_PASS_THRESHOLD)
    n_still_pass_point = sum(1 for r in pass_arity_results
                             if r.m4.m4b >= M4B_PASS_THRESHOLD)
    print(f"  At the 4 PASS-arity cells:")
    print(f"    Cells with point M4b >= {M4B_PASS_THRESHOLD:.2f}: "
          f"{n_still_pass_point}/4")
    print(f"    Cells with bootstrap M4b CI lower bound >= {M4B_PASS_THRESHOLD:.2f}: "
          f"{n_still_pass}/4")
    print()


# ==============================================================================
# Device detection (same as 21)
# ==============================================================================
def get_device() -> tuple[str, str]:
    if torch.backends.mps.is_available():
        return "mps", "MPS (Apple Silicon)"
    if torch.cuda.is_available():
        return "cuda", "CUDA"
    return "cpu", "CPU"


def free_model(model) -> None:
    try:
        model.cpu()
    except Exception:
        pass
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ==============================================================================
# Tokenization audit - print the BPE decomposition of each invented word
# under both models' tokenizers.
# ==============================================================================
def audit_tokenization() -> None:
    print("=" * 92)
    print("TOKENIZATION AUDIT (16 invented words under both tokenizers)")
    print("=" * 92)
    print()
    for spec in MODEL_SPECS:
        print(f"  {spec.short_name} ({spec.model_id})")
        try:
            tok = AutoTokenizer.from_pretrained(spec.model_id)
        except Exception as e:
            print(f"    [tokenizer load failed: {e}]")
            continue
        print(f"    {'word':<10} {'subwords':<40} {'n':>3}")
        for w in INVENTED_WORDS:
            ids = tok.encode(" " + w, add_special_tokens=False)
            subs = [tok.decode([i]) for i in ids]
            tag = " (orig)" if w in ORIGINAL_5 else ""
            print(f"    {w:<10} {str(subs):<40} {len(ids):>3}{tag}")
        print()


# ==============================================================================
# Main entry
# ==============================================================================
def main() -> None:
    log_path = _setup_logging()
    print(f"Script 22c - expanded invented-word set (16 words, 8B + 8U)")
    print(f"STIMULUS_VERSION={STIMULUS_VERSION}  SEED={SEED}  N_PER_CLASS={N_PER_CLASS}")
    print()
    print(f"Invented set: {INVENTED_WORDS}")
    print(f"  ({sum(1 for w in INVENTED_WORDS if CANONICAL_ARITY[W_TO_CANONICAL[w]] == 2)} "
          f"intended-binary, "
          f"{sum(1 for w in INVENTED_WORDS if CANONICAL_ARITY[W_TO_CANONICAL[w]] == 1)} "
          f"intended-unary)")
    print(f"Original 5: {ORIGINAL_5}")
    print(f"New 11:     {NEW_11}")
    print()
    audit_tokenization()

    device, device_name = get_device()
    print(f"Device: {device_name}")
    print()

    # Group cells by model so we only load each model once.
    cells_by_model: dict[str, list[TestCell]] = {}
    for cell in TEST_CELLS:
        cells_by_model.setdefault(cell.model_short, []).append(cell)

    all_results: list[CellResult] = []
    cells_in_order: list[TestCell] = []

    for spec in MODEL_SPECS:
        if spec.short_name not in cells_by_model:
            continue
        cells = cells_by_model[spec.short_name]
        print("=" * 92)
        print(f"MODEL: {spec.short_name}")
        print("=" * 92)
        t_total = time.time()

        canon_neut_prompts, _, inv_neut_prompts, _ = _generate_prompts("NEUTRAL")
        canon_func_prompts, _, inv_func_prompts, _ = _generate_prompts("FUNC-PFX")

        neut_cache = _cache_load(
            _cache_path(spec.short_name, "NEUTRAL"),
            expected_canon_hash=prompts_checksum(canon_neut_prompts),
            expected_inv_hash=prompts_checksum(inv_neut_prompts),
            expected_anchors=ANCHORS_NEUTRAL,
        )
        func_cache = _cache_load(
            _cache_path(spec.short_name, "FUNC-PFX"),
            expected_canon_hash=prompts_checksum(canon_func_prompts),
            expected_inv_hash=prompts_checksum(inv_func_prompts),
            expected_anchors=ANCHORS_FUNC_PFX,
        )

        model = None
        tok = None
        if neut_cache is None or func_cache is None:
            print(f"\n  Loading model: {spec.model_id}")
            t0 = time.time()
            tok = AutoTokenizer.from_pretrained(spec.model_id)
            model = AutoModelForCausalLM.from_pretrained(
                spec.model_id, torch_dtype=spec.dtype, low_cpu_mem_usage=True,
            ).to(device).eval()
            print(f"    loaded in {time.time() - t0:.1f}s")

        if neut_cache is None:
            print(f"\n  -- Building NEUTRAL condition --")
            neut = build_condition(spec, "NEUTRAL", ANCHORS_NEUTRAL,
                                   model, tok, device)
        else:
            neut = neut_cache

        if func_cache is None:
            print(f"\n  -- Building FUNC-PFX condition --")
            func = build_condition(spec, "FUNC-PFX", ANCHORS_FUNC_PFX,
                                   model, tok, device)
        else:
            func = func_cache

        if model is not None:
            free_model(model)
        if tok is not None:
            del tok
        gc.collect()

        cond_by_name = {"NEUTRAL": neut, "FUNC-PFX": func}

        for cell in cells:
            train_cond = cond_by_name[cell.train_cond]
            test_cond = cond_by_name[cell.test_cond]
            res = evaluate_cell(cell, train_cond, test_cond)
            all_results.append(res)
            cells_in_order.append(cell)
            print_cell_result(cell, res)

        print(f"\n  -- {spec.short_name} total time: {time.time() - t_total:.1f}s --\n")

    print_summary_table(cells_in_order, all_results)

    # Survives / fails-with-expansion summary.
    print("=" * 92)
    print("INTERPRETATION")
    print("=" * 92)
    print()
    pass_cells = [c for c in cells_in_order if not c.name.startswith("[LUCKY-NEG]")]
    pass_results = [r for c, r in zip(cells_in_order, all_results)
                    if not c.name.startswith("[LUCKY-NEG]")]
    n_point = sum(1 for r in pass_results if r.m4.m4b >= M4B_PASS_THRESHOLD)
    n_ci = sum(1 for r in pass_results
               if r.bootstrap_m4b["ci95_low"] >= M4B_PASS_THRESHOLD)
    print(f"  Phase 1 falsification test: did the 4 PASS-arity cells survive the")
    print(f"  expansion from 5 to 16 invented words?")
    print()
    print(f"  Point M4b >= 0.65 with 16 words: {n_point}/4 cells")
    print(f"  Bootstrap M4b CI floor >= 0.65 with 16 words: {n_ci}/4 cells")
    print()
    if n_ci == 4:
        print(f"  => All 4 PASS-arity cells survive expansion. The cross-notation")
        print(f"     arity-respecting transfer finding is real and is NOT a 5-word")
        print(f"     sampling artifact.")
    elif n_ci >= 2:
        print(f"  => {n_ci} of 4 PASS-arity cells survive expansion under the strict")
        print(f"     bootstrap CI floor criterion. The cross-notation arity-")
        print(f"     respecting transfer finding holds at a subset of cells; the")
        print(f"     others were partially dependent on the 5-word sample.")
        print(f"     Identify the surviving cell(s) as the principal Phase 1")
        print(f"     finding(s) and document the non-surviving cells as 5-word-")
        print(f"     specific.")
    elif n_point >= 2:
        print(f"  => Point M4b >= 0.65 at {n_point}/4 cells but bootstrap CI is")
        print(f"     weak. The finding is fragile under stimulus resampling and")
        print(f"     requires further validation (e.g., 22d expanded canonicals).")
    else:
        print(f"  => The 4-of-5 (resp. 11-13 of 16) pattern does NOT survive")
        print(f"     expansion. The 5-word PASS-arity cells were a sampling")
        print(f"     artifact. The §3.7.11/§3.7.12 cross-notation arity-respecting")
        print(f"     transfer finding does NOT generalize and must be retracted.")
    print()

    # Lucky-default controls.
    lucky_cells = [c for c in cells_in_order if c.name.startswith("[LUCKY-NEG]")]
    lucky_results = [r for c, r in zip(cells_in_order, all_results)
                     if c.name.startswith("[LUCKY-NEG]")]
    if lucky_cells:
        print(f"  Lucky-default negative controls (should stay at floor with 16 words):")
        for c, r in zip(lucky_cells, lucky_results):
            verdict = "REMAINS lucky-default" if r.lucky_default else "no longer lucky-default with 16 words"
            print(f"    {c.name}: M4b 5w={c.expected_m4b_5word:.3f} -> 16w={r.m4.m4b:.3f}  ({verdict})")
        print()
        print(f"  If lucky-default cells drop from M4b ~ 0.80 (5-word) to ~ 0.50 (16-word),")
        print(f"  the M4b ≥ 0.65 threshold is the right discriminator: it accepts PASS-")
        print(f"  arity transfer and rejects lucky-default when the invented set is")
        print(f"  balanced 8/8. If lucky-default cells STAY high, they're not lucky-")
        print(f"  default but something genuinely arity-preserving (which would be a")
        print(f"  surprise).")
        print()

    if log_path:
        print(f"[logging] full transcript written to: {log_path}")


if __name__ == "__main__":
    main()
