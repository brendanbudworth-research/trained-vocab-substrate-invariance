"""Script 23 - Pythia 6.9B-deduped cross-model replication.

Phase 2 principal experiment: replicate the OLMo 2 7B / Gemma 2 9B
operator-set-bound substrate-invariance finding on a third model
family (Pythia 6.9B-deduped, EleutherAI, GPT-NeoX architecture, trained
on the Pile).

Hypothesis under test. The Phase 1 result on OLMo + Gemma (§3.7.10 -
§3.7.14) is operator-set-bound substrate-invariance:

  (Fact 1) M2-canonical cross-notation transfer is robust at multiple
  (anchor, layer) cells in both models. Probes trained on canonical-
  operator activations under one notation predict canonical-operator
  identity under the other notation at substantially above chance
  (~8-10x chance, 95% bootstrap CIs above the M2 gate threshold).

  (Fact 2) Novel-operator generalization fails. All 4 originally-PASS-
  arity cells from script 22b are retracted under either the
  16-invented expansion (22c, §3.7.13) or the 10-canonical expansion
  (22d, §3.7.14). The mechanism is "default-to-rarest-canonical":
  invented mass concentrates on whichever canonical sits in the
  highest-entropy decision region of the readout vocabulary; that
  target shifts wholesale when the canonical set is expanded with new
  near-zero-frequency canonicals.

The Phase 2 replication question is: does Pythia 6.9B-deduped show the
same two-part pattern? Specifically:

  (P1) Does Pythia exhibit M2-canonical PASS (>= 0.65 on 10-class
       readout, ~7-8x chance) at any cross-notation (anchor, layer)
       cell?
  (P2) Does Pythia exhibit M4b PASS (intended-arity agreement >= 0.65)
       under the v3 5-canonical/5-invented readout at any cell? If yes,
       does that cell retract under v4 (16-invented) or v5 (10-canonical)
       like OLMo/Gemma?
  (P3) Does Pythia's default-to-rarest-canonical mechanism point at the
       same operators as OLMo/Gemma (nand/negate dominance under v5)?
       Or does Pythia have a distinct compression target?

A "yes" on (P1) + retraction on (P2) + similar default on (P3) is the
clean three-model replication. Any deviation is interesting and gets a
section in §3.7.15.

This script extracts a single v5-expanded-canonical Pythia cache
(NEUTRAL + FUNC-PFX, 10 canonicals + 16 invented at all anchors), then
runs the full M1/M2/M3/M4 sweep at THREE scopes:

  v3-scope: 5 original canonicals + 5 original invented (data
            subselected from v5 cache; comparable to 22b on OLMo/Gemma)
  v4-scope: 5 original canonicals + 16 invented (subselected;
            comparable to 22c)
  v5-scope: 10 canonicals + 16 invented (the principal analysis;
            comparable to 22d)

Each scope runs a full anchor x layer x direction sweep, identifies
PASS-arity cells, and reports the same headlines as 22b/22c/22d.

Total runtime estimate (M4 MPS fp16):
  Pythia 6.9B model load:              ~30 s
  v5 cache extraction (1300 stim x 2):  ~30-50 min
  Sweep + analysis (cache-only):       ~1-2 min

Total: ~35-55 minutes. Cache is ~1.3 GB (similar to 22d's OLMo cache).
Re-runs from cache: ~2 min.

Tees all output to outputs/23_<ts>.log.
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
from typing import Iterable

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
    log_path = os.path.join(log_dir, f"23_{ts}.log")
    log_f = open(log_path, "w", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_f)
    sys.stderr = _Tee(sys.__stderr__, log_f)
    print(f"[logging] tee'ing all output to {log_path}")
    return log_path


# ==============================================================================
# Constants - mirror 22d for direct cross-model comparability
# ==============================================================================
SEED = 17
N_PER_CLASS = 50
N_BOOTSTRAP = 500

STIMULUS_VERSION = "v5-expanded-canonical"

# Canonical set: 10 (same as 22d).
CANONICALS = [
    "and", "or", "implies", "xor", "nand",
    "not", "necessarily", "possibly", "always", "negate",
]
ORIGINAL_5_CANONICALS = ["and", "or", "not", "implies", "necessarily"]
NEW_5_CANONICALS = ["xor", "nand", "possibly", "always", "negate"]
UNARY_CANONICALS_10 = ["not", "necessarily", "possibly", "always", "negate"]
BINARY_CANONICALS_10 = ["and", "or", "implies", "xor", "nand"]
UNARY_CANONICALS_5 = ["not", "necessarily"]
BINARY_CANONICALS_5 = ["and", "or", "implies"]
MODAL_ADVERBIAL_UNARY = ["necessarily", "possibly", "always"]
OPERATIONAL_UNARY = ["not", "negate"]

CANONICAL_ARITY = {
    "and": 2, "or": 2, "implies": 2, "xor": 2, "nand": 2,
    "not": 1, "necessarily": 1, "possibly": 1, "always": 1, "negate": 1,
}

# Invented set: 16 words (same as 22c/22d).
INVENTED_16 = [
    "bliq", "dren", "molex",
    "krev", "sond", "glin", "twiv", "fump",
    "vusp", "perph",
    "kelm", "zorf", "gleph", "drelth", "vrith", "nilph",
]
INVENTED_5 = ["bliq", "dren", "vusp", "molex", "perph"]
W_TO_CANONICAL_16 = {
    "bliq": "and", "dren": "or", "molex": "implies",
    "krev": "and", "sond": "or", "glin": "implies",
    "twiv": "and", "fump": "or",
    "vusp": "not", "perph": "necessarily",
    "kelm": "not", "zorf": "not",
    "gleph": "necessarily", "drelth": "necessarily",
    "vrith": "not", "nilph": "necessarily",
}

# Anchor definitions.
ANCHORS_FUNC_PFX = ["operator-after", "first-arg", "close-paren", "sentence-final"]
ANCHORS_NEUTRAL = ["operator-after", "sentence-final"]

# Verdict thresholds (mirror 22b).
GATE_CANONICAL_PASS = 0.65
GATE_ARITY_PASS = 0.65
M4B_PASS = 0.65
M4C_DISTRIBUTED = 0.70


# ==============================================================================
# Model spec - Pythia 6.9B-deduped
# ==============================================================================
@dataclass
class ModelSpec:
    short_name: str
    model_id: str
    dtype: "torch.dtype"
    focus_layers: list[int]


PYTHIA_SPEC = ModelSpec(
    short_name="Pythia 6.9B-deduped",
    model_id="EleutherAI/pythia-6.9b-deduped",
    dtype=torch.float16,
    focus_layers=[4, 7, 10, 16, 24],  # same proportional positions as OLMo 2 7B
)


# ==============================================================================
# Stable seeding helpers (same as 22d).
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
# Load 19b module and monkey-patch the expanded sets (same approach as 22d)
# ==============================================================================
def _load_19b_module():
    import importlib.machinery
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "19b_directional_angle_gated.py")
    loader = importlib.machinery.SourceFileLoader("_m19b_23", path)
    spec = importlib.util.spec_from_loader("_m19b_23", loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_m19b_23"] = mod
    loader.exec_module(mod)
    return mod


_M19B = _load_19b_module()
_M19B.CANONICALS = CANONICALS
_M19B.CANONICAL_ARITY = CANONICAL_ARITY
_M19B.UNARY_CANONICALS = UNARY_CANONICALS_10
_M19B.BINARY_CANONICALS = BINARY_CANONICALS_10
_M19B.INVENTED_WORDS = INVENTED_16
_M19B.W_TO_CANONICAL = W_TO_CANONICAL_16
make_neutral_stimuli = _M19B.make_neutral_stimuli
make_functional_canonical_stimuli = _M19B.make_functional_canonical_stimuli
make_functional_invented_stimuli = _M19B.make_functional_invented_stimuli
assert _M19B.SEED == SEED
assert _M19B.N_PER_CLASS == N_PER_CLASS


def _generate_prompts(name: str) -> tuple[list[str], list[str], list[str], list[str]]:
    """Generate canonical (10 × N) + invented (16 × N) prompts.
    Identical generation logic to 22d so cache fingerprints align."""
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
    for w in INVENTED_16:
        w_rng = random.Random(stable_seed(name, "inv", w))
        inv_prompts.extend(invented_stim_fn(w, w_rng, N_PER_CLASS))
        inv_words.extend([w] * N_PER_CLASS)

    return canon_prompts, canon_labels, inv_prompts, inv_words


# ==============================================================================
# Import anchor finder + extraction from script 21
# ==============================================================================
def _load_21_module():
    import importlib.machinery
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "21_multi_anchor_battery.py")
    loader = importlib.machinery.SourceFileLoader("_m21_23", path)
    spec = importlib.util.spec_from_loader("_m21_23", loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_m21_23"] = mod
    loader.exec_module(mod)
    return mod


_M21 = _load_21_module()
extract_multi_anchor_activations = _M21.extract_multi_anchor_activations
ConditionMultiAnchor = _M21.ConditionMultiAnchor


# ==============================================================================
# Cache (23 = standalone; Pythia v5 cache)
# ==============================================================================
def _cache_path(model_short_name: str, condition_name: str) -> str:
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "outputs", "cache")
    os.makedirs(base, exist_ok=True)
    slug = model_short_name.replace(" ", "_")
    return os.path.join(
        base,
        f"23_{slug}_{condition_name}_npc{N_PER_CLASS}_{STIMULUS_VERSION}.npz",
    )


def _cache_save(
    path: str, cond: ConditionMultiAnchor,
    *, model_id: str, condition_name: str,
    canon_prompts_hash: str, inv_prompts_hash: str,
    dtype_before_cache: str,
) -> None:
    np.savez_compressed(
        path,
        canonical_X=cond.canonical_X.astype(np.float16),
        canonical_labels=cond.canonical_labels,
        invented_X=cond.invented_X.astype(np.float16),
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
        canonical_list=np.array(CANONICALS),
        invented_word_list=np.array(INVENTED_16),
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
            "canonical_list", "invented_word_list",
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
        if list(z["anchor_names"]) != expected_anchors:
            return None
        if list(z["canonical_list"]) != CANONICALS:
            return None
        if list(z["invented_word_list"]) != INVENTED_16:
            return None
        cond = ConditionMultiAnchor(
            canonical_X=z["canonical_X"].astype(np.float32),
            canonical_labels=z["canonical_labels"],
            invented_X=z["invented_X"].astype(np.float32),
            invented_word_per_stim=z["invented_word_per_stim"],
            anchor_names=list(z["anchor_names"]),
            canon_anchor_positions=z["canon_anchor_positions"].tolist(),
            inv_anchor_positions=z["inv_anchor_positions"].tolist(),
        )
        print(f"    [cache] hit {os.path.basename(path)} "
              f"canon={cond.canonical_X.shape}, inv={cond.invented_X.shape}")
        return cond
    except Exception as e:
        print(f"    [cache] failed to load {path}: {e}")
        return None


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
    canon_X, canon_pos, _ = extract_multi_anchor_activations(
        model, tok, canon_prompts, CANONICALS, anchor_names, condition_name, device
    )
    print(f"      {time.time() - t0:.1f}s, shape={canon_X.shape}")

    print(f"    extracting invented activations ({len(inv_prompts)} stim)...")
    t0 = time.time()
    inv_X, inv_pos, _ = extract_multi_anchor_activations(
        model, tok, inv_prompts, INVENTED_16, anchor_names, condition_name, device
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
# Slice helpers + subset constructors
# ==============================================================================
def slice_canonical(cond: ConditionMultiAnchor, anchor: str, layer: int):
    a_idx = cond.anchor_names.index(anchor)
    return cond.canonical_X[a_idx, :, layer, :], np.asarray(cond.canonical_labels)


def slice_invented(cond: ConditionMultiAnchor, anchor: str, layer: int):
    a_idx = cond.anchor_names.index(anchor)
    return cond.invented_X[a_idx, :, layer, :], np.asarray(cond.invented_word_per_stim)


def subset_canonical(
    X: np.ndarray, y: np.ndarray, keep: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isin(y, keep)
    return X[mask], y[mask]


def subset_invented(
    X: np.ndarray, w: np.ndarray, keep: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isin(w, keep)
    return X[mask], w[mask]


# ==============================================================================
# Metric primitives (parameterized by canonical set)
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
    m4a: float
    m4b: float
    m4c: float
    breakdown_pct: dict[str, float]
    per_word_top: dict[str, str]
    per_word_top_pct: dict[str, float]
    per_word_unary_pct: dict[str, float]


def m4_breakdown(
    clf: LogisticRegression, X_inv: np.ndarray, words: np.ndarray,
    *, canonicals: list[str], unary_canonicals: list[str],
    invented_set: list[str], w_to_canonical: dict[str, str],
) -> M4Result:
    preds = clf.predict(X_inv)
    n_total = len(preds)
    breakdown_pct = {c: float(np.mean(preds == c)) for c in canonicals}
    m4a = sum(breakdown_pct[c] for c in unary_canonicals)
    m4c = float(sum(v ** 2 for v in breakdown_pct.values()))

    n_match = 0
    for w, p in zip(words, preds):
        intended_can = w_to_canonical[str(w)]
        if CANONICAL_ARITY[intended_can] == CANONICAL_ARITY[str(p)]:
            n_match += 1
    m4b = n_match / n_total

    per_word_top: dict[str, str] = {}
    per_word_top_pct: dict[str, float] = {}
    per_word_unary_pct: dict[str, float] = {}
    for w in invented_set:
        mask = (words == w)
        if not mask.any():
            continue
        word_preds = preds[mask]
        counts = {c: int(np.sum(word_preds == c)) for c in canonicals}
        top_c = max(counts, key=lambda c: counts[c])
        per_word_top[w] = top_c
        per_word_top_pct[w] = counts[top_c] / mask.sum()
        per_word_unary_pct[w] = sum(counts[c] for c in unary_canonicals) / mask.sum()

    return M4Result(
        m4a=m4a, m4b=m4b, m4c=m4c,
        breakdown_pct=breakdown_pct,
        per_word_top=per_word_top,
        per_word_top_pct=per_word_top_pct,
        per_word_unary_pct=per_word_unary_pct,
    )


def bootstrap_m4b(
    clf: LogisticRegression, X_inv: np.ndarray, words: np.ndarray,
    *, invented_set: list[str], w_to_canonical: dict[str, str],
    n_bootstrap: int = N_BOOTSTRAP,
) -> dict[str, float]:
    preds = clf.predict(X_inv)
    word_to_idx = {w: np.where(words == w)[0] for w in invented_set}
    m4b_samples = []
    rng = np.random.default_rng(SEED)
    for _ in range(n_bootstrap):
        idx_list = []
        for w in invented_set:
            w_idx = word_to_idx[w]
            if len(w_idx) == 0:
                continue
            idx_list.append(rng.choice(w_idx, size=len(w_idx), replace=True))
        all_idx = np.concatenate(idx_list)
        sub_preds = preds[all_idx]
        sub_words = words[all_idx]
        n_match = sum(
            CANONICAL_ARITY[w_to_canonical[str(w)]] == CANONICAL_ARITY[str(p)]
            for w, p in zip(sub_words, sub_preds)
        )
        m4b_samples.append(n_match / len(sub_preds))
    arr = np.array(m4b_samples)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "ci95_low": float(np.percentile(arr, 2.5)),
        "ci95_high": float(np.percentile(arr, 97.5)),
        "p_pass_065": float(np.mean(arr >= M4B_PASS)),
    }


def detect_lucky_default(per_word_top_pct: dict[str, float]) -> bool:
    pcts = list(per_word_top_pct.values())
    if not pcts:
        return False
    return min(pcts) >= 0.95


# ==============================================================================
# Sweep cell with scoped metrics
# ==============================================================================
@dataclass
class SweepCell:
    scope: str  # "v3", "v4", or "v5"
    direction: str  # "N->F" or "F->N"
    train_cond: str
    train_anchor: str
    test_cond: str
    test_anchor: str
    layer: int

    M1_tr: float = 0.0
    M1_te: float = 0.0
    M2_cano: float = 0.0
    M2_arity: float = 0.0
    M4a: float = 0.0
    M4b: float = 0.0
    M4c: float = 0.0
    per_word_min_top_pct: float = 0.0
    per_word_max_top_pct: float = 0.0
    per_word_top: dict[str, str] = field(default_factory=dict)
    per_word_top_pct: dict[str, float] = field(default_factory=dict)
    lucky_default: bool = False

    @property
    def m2_gap(self) -> float:
        return self.M2_arity - self.M2_cano

    @property
    def verdict(self) -> str:
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
        if self.M2_arity >= GATE_ARITY_PASS and self.M4b >= M4B_PASS:
            return "ARITY-AXIS-ONLY"
        if self.M2_arity >= GATE_ARITY_PASS:
            return "M2A-ONLY"
        if self.M2_cano >= GATE_CANONICAL_PASS:
            return "M2C-ONLY"
        return "FAIL"


def enumerate_cells(scope: str, layers: list[int]) -> list[SweepCell]:
    cells: list[SweepCell] = []
    for L in layers:
        for tr_a in ANCHORS_NEUTRAL:
            for te_a in ANCHORS_FUNC_PFX:
                cells.append(SweepCell(
                    scope=scope, direction="N->F",
                    train_cond="NEUTRAL", train_anchor=tr_a,
                    test_cond="FUNC-PFX", test_anchor=te_a,
                    layer=L,
                ))
        for tr_a in ANCHORS_FUNC_PFX:
            for te_a in ANCHORS_NEUTRAL:
                cells.append(SweepCell(
                    scope=scope, direction="F->N",
                    train_cond="FUNC-PFX", train_anchor=tr_a,
                    test_cond="NEUTRAL", test_anchor=te_a,
                    layer=L,
                ))
    return cells


@dataclass
class Scope:
    name: str
    canonicals: list[str]
    unary_canonicals: list[str]
    binary_canonicals: list[str]
    invented_set: list[str]
    w_to_canonical: dict[str, str]


SCOPE_V3 = Scope(
    name="v3", canonicals=ORIGINAL_5_CANONICALS,
    unary_canonicals=UNARY_CANONICALS_5, binary_canonicals=BINARY_CANONICALS_5,
    invented_set=INVENTED_5,
    w_to_canonical={w: W_TO_CANONICAL_16[w] for w in INVENTED_5},
)
SCOPE_V4 = Scope(
    name="v4", canonicals=ORIGINAL_5_CANONICALS,
    unary_canonicals=UNARY_CANONICALS_5, binary_canonicals=BINARY_CANONICALS_5,
    invented_set=INVENTED_16,
    w_to_canonical=W_TO_CANONICAL_16,
)
SCOPE_V5 = Scope(
    name="v5", canonicals=CANONICALS,
    unary_canonicals=UNARY_CANONICALS_10, binary_canonicals=BINARY_CANONICALS_10,
    invented_set=INVENTED_16,
    w_to_canonical=W_TO_CANONICAL_16,
)


def run_cell(
    cell: SweepCell,
    cond_by_name: dict[str, ConditionMultiAnchor],
    scope: Scope,
) -> SweepCell:
    train_cond = cond_by_name[cell.train_cond]
    test_cond = cond_by_name[cell.test_cond]

    X_tr_full, y_tr_full = slice_canonical(train_cond, cell.train_anchor, cell.layer)
    X_te_full, y_te_full = slice_canonical(test_cond, cell.test_anchor, cell.layer)
    X_inv_full, w_inv_full = slice_invented(test_cond, cell.test_anchor, cell.layer)

    X_tr, y_tr = subset_canonical(X_tr_full, y_tr_full, scope.canonicals)
    X_te, y_te = subset_canonical(X_te_full, y_te_full, scope.canonicals)
    X_inv, w_inv = subset_invented(X_inv_full, w_inv_full, scope.invented_set)

    cell.M1_tr = m1_cv(X_tr, y_tr)
    cell.M1_te = m1_cv(X_te, y_te)
    cell.M2_cano, cell.M2_arity, _, clf = m2_metrics(X_tr, y_tr, X_te, y_te)

    m4 = m4_breakdown(
        clf, X_inv, w_inv,
        canonicals=scope.canonicals,
        unary_canonicals=scope.unary_canonicals,
        invented_set=scope.invented_set,
        w_to_canonical=scope.w_to_canonical,
    )
    cell.M4a = m4.m4a
    cell.M4b = m4.m4b
    cell.M4c = m4.m4c
    cell.per_word_top = m4.per_word_top
    cell.per_word_top_pct = m4.per_word_top_pct
    pcts = list(m4.per_word_top_pct.values())
    cell.per_word_min_top_pct = float(min(pcts)) if pcts else 0.0
    cell.per_word_max_top_pct = float(max(pcts)) if pcts else 0.0
    cell.lucky_default = detect_lucky_default(m4.per_word_top_pct)
    return cell


# ==============================================================================
# Reporting
# ==============================================================================
def _cell_short(c: SweepCell) -> str:
    return (
        f"{c.scope} {c.direction} {c.train_anchor[:5]:>5}->{c.test_anchor[:5]:<5} "
        f"L{c.layer:>2}"
    )


def print_sweep_table(cells: list[SweepCell], scope_name: str) -> None:
    print()
    print("=" * 200)
    print(f"  Pythia 6.9B-deduped - {scope_name} scope - full sweep "
          f"({len(cells)} cells)")
    print("=" * 200)
    h = (
        f"  {'cell':<32} | {'M1tr':<5} | {'M1te':<5} | {'M2cano':<7} | "
        f"{'M2arty':<7} | {'gap':<6} | {'M4a':<6} | {'M4b':<6} | "
        f"{'M4c':<5} | {'pwmin':<5} | {'verdict':<14}"
    )
    print(h)
    print(f"  {'-' * (len(h) - 2)}")
    for c in cells:
        flag = " *L" if c.lucky_default else ""
        print(
            f"  {_cell_short(c):<32} | "
            f"{c.M1_tr:.2f}  | {c.M1_te:.2f}  | "
            f"{c.M2_cano:.3f}   | {c.M2_arity:.3f}   | "
            f"{c.m2_gap:+.3f} | "
            f"{c.M4a*100:>4.1f}% | {c.M4b*100:>4.1f}% | "
            f"{c.M4c:.2f}  | {c.per_word_min_top_pct:.2f}  | "
            f"{c.verdict + flag:<14}"
        )


def print_top_k(cells: list[SweepCell], scope_name: str, k: int = 8) -> list[SweepCell]:
    eligible = [c for c in cells if not c.lucky_default]
    sorted_cells = sorted(
        eligible,
        key=lambda c: (c.M2_arity, c.M4b, -c.M4c),
        reverse=True,
    )[:k]
    print()
    print(f"  TOP-{k} cells under {scope_name} (sort: M2-arity desc, M4b desc, M4c asc):")
    for i, c in enumerate(sorted_cells):
        print(
            f"    {i+1:>2}. {_cell_short(c):<32} | M2c={c.M2_cano:.3f} | "
            f"M2a={c.M2_arity:.3f} | M4b={c.M4b*100:>5.1f}% | "
            f"M4c={c.M4c:.2f} | M4a={c.M4a*100:>5.1f}% | verdict={c.verdict}"
        )
    return sorted_cells


def print_canonical_breakdown(
    cells: list[SweepCell], scope: Scope, *, top_cell_count: int = 3,
) -> None:
    """At the top M2-cano cells, print the full canonical breakdown for
    invented words. Used to detect the default-to-rarest-canonical
    mechanism."""
    sorted_by_m2c = sorted(cells, key=lambda c: -c.M2_cano)[:top_cell_count]
    print()
    print(f"  Canonical-readout breakdown of invented mass at the top-"
          f"{top_cell_count} M2-canonical cells under {scope.name}:")
    print(f"  (Strict-arity prediction: invented binary words spread across binary "
          f"canonicals; invented unary words spread across unary canonicals.)")
    print(f"  (Default-to-rarest-canonical prediction: mass concentrates on one or "
          f"two low-frequency canonicals regardless of intended arity.)")
    for c in sorted_by_m2c:
        print()
        print(f"    >> {_cell_short(c)} (M2c={c.M2_cano:.3f}, M4b={c.M4b:.3f}, "
              f"M4c={c.M4c:.2f})")
        for w in scope.invented_set:
            if w not in c.per_word_top:
                continue
            intended_can = scope.w_to_canonical[w]
            intended_arity_tag = "B" if CANONICAL_ARITY[intended_can] == 2 else "U"
            top_c = c.per_word_top[w]
            top_pct = c.per_word_top_pct[w]
            match = (CANONICAL_ARITY[top_c] == CANONICAL_ARITY[intended_can]
                     if top_c in CANONICAL_ARITY else False)
            print(f"       {w:<10} (int.{intended_arity_tag}: {intended_can:<11}) "
                  f"-> {top_c:<13} {top_pct*100:5.1f}%   "
                  f"{'arity-match' if match else 'arity-MISMATCH'}")


def print_headline(
    cells_by_scope: dict[str, list[SweepCell]],
    olmo_summary: str | None = None,
    gemma_summary: str | None = None,
) -> None:
    print()
    print("=" * 200)
    print("PYTHIA 6.9B-deduped: PHASE 2 REPLICATION HEADLINE")
    print("=" * 200)
    print()
    print(f"  Question 1 (P1): does Pythia exhibit M2-canonical PASS "
          f"(>= {GATE_CANONICAL_PASS}) at any cross-notation (anchor, layer) cell?")
    for scope_name in ("v3", "v4", "v5"):
        cells = cells_by_scope[scope_name]
        pass_cano = [c for c in cells if c.M2_cano >= GATE_CANONICAL_PASS]
        n_c = len(cells[0].per_word_top) if cells else 0
        if pass_cano:
            best = max(pass_cano, key=lambda c: c.M2_cano)
            n_classes = len(SCOPE_V5.canonicals) if scope_name == "v5" else 5
            print(f"    {scope_name} ({n_classes}-class): {len(pass_cano)}/{len(cells)} "
                  f"cells PASS; best = {best.M2_cano:.3f} at {_cell_short(best)}")
        else:
            print(f"    {scope_name}: NO cells PASS M2-canonical")
    print()

    print(f"  Question 2 (P2): does Pythia exhibit PASS-arity verdict at v3, and is "
          f"it retracted at v4/v5?")
    v3_pass_arity = [c for c in cells_by_scope["v3"] if c.verdict == "PASS-arity"]
    v4_pass_arity = [c for c in cells_by_scope["v4"] if c.verdict == "PASS-arity"]
    v5_pass_arity = [c for c in cells_by_scope["v5"] if c.verdict == "PASS-arity"]
    print(f"    v3 PASS-arity cells: {len(v3_pass_arity)}")
    print(f"    v4 PASS-arity cells: {len(v4_pass_arity)}")
    print(f"    v5 PASS-arity cells: {len(v5_pass_arity)}")
    if v3_pass_arity:
        print(f"    v3 PASS-arity candidates:")
        for c in v3_pass_arity:
            print(f"      - {_cell_short(c)} | M2a={c.M2_arity:.3f} M4b={c.M4b:.3f} M4c={c.M4c:.2f}")
    print()

    print(f"  Question 3 (P3): does Pythia's v5 default-to-rarest-canonical mechanism")
    print(f"  point at the new canonicals (xor/nand/possibly/always/negate)?")
    v5_cells = cells_by_scope["v5"]
    new_canonical_tops: dict[str, int] = {c: 0 for c in NEW_5_CANONICALS}
    orig_canonical_tops: dict[str, int] = {c: 0 for c in ORIGINAL_5_CANONICALS}
    for c in v5_cells:
        for w, top_c in c.per_word_top.items():
            if top_c in NEW_5_CANONICALS:
                new_canonical_tops[top_c] += 1
            elif top_c in ORIGINAL_5_CANONICALS:
                orig_canonical_tops[top_c] += 1
    total = sum(new_canonical_tops.values()) + sum(orig_canonical_tops.values())
    if total > 0:
        print(f"    Aggregate per-word top canonical across all v5 cells:")
        for c in CANONICALS:
            n = new_canonical_tops.get(c, 0) + orig_canonical_tops.get(c, 0)
            is_new = " (NEW)" if c in NEW_5_CANONICALS else ""
            print(f"      {c:<13}{is_new:<6}: {n}/{total} ({n/total*100:.1f}%)")
    print()

    print(f"  CROSS-MODEL COMPARISON:")
    if olmo_summary:
        print(f"    OLMo 2 7B:        {olmo_summary}")
    if gemma_summary:
        print(f"    Gemma 2 9B:       {gemma_summary}")
    print(f"    Pythia 6.9B-d:    {_pythia_one_line_summary(cells_by_scope)}")


def _pythia_one_line_summary(cells_by_scope: dict[str, list[SweepCell]]) -> str:
    v3_cells = cells_by_scope["v3"]
    v5_cells = cells_by_scope["v5"]
    best_v3_m2c = max(c.M2_cano for c in v3_cells)
    best_v5_m2c = max(c.M2_cano for c in v5_cells)
    best_v3_m2a = max(c.M2_arity for c in v3_cells)
    best_v5_m4b = max(c.M4b for c in v5_cells)
    v3_pa = sum(1 for c in v3_cells if c.verdict == "PASS-arity")
    v5_pa = sum(1 for c in v5_cells if c.verdict == "PASS-arity")
    return (
        f"best M2c v3={best_v3_m2c:.2f} v5={best_v5_m2c:.2f}; "
        f"best M2a v3={best_v3_m2a:.2f}; best v5 M4b={best_v5_m4b:.2f}; "
        f"PASS-arity cells: v3={v3_pa} v5={v5_pa}"
    )


# ==============================================================================
# Tokenization audit
# ==============================================================================
def audit_tokenization() -> None:
    print("=" * 100)
    print("PYTHIA TOKENIZATION AUDIT (10 canonicals + 16 invented words)")
    print("=" * 100)
    try:
        tok = AutoTokenizer.from_pretrained(PYTHIA_SPEC.model_id)
    except Exception as e:
        print(f"  [tokenizer load failed: {e}]")
        return
    print(f"  {'word':<14} {'subwords':<48} {'n':>3}")
    for c in CANONICALS:
        ids = tok.encode(" " + c, add_special_tokens=False)
        subs = [tok.decode([i]) for i in ids]
        is_new = " (NEW)" if c in NEW_5_CANONICALS else ""
        arity_tag = " (B)" if CANONICAL_ARITY[c] == 2 else " (U)"
        print(f"  {c+arity_tag:<14} {str(subs):<48} {len(ids):>3}{is_new}")
    print()
    for w in INVENTED_16:
        ids = tok.encode(" " + w, add_special_tokens=False)
        subs = [tok.decode([i]) for i in ids]
        print(f"  {w+' (I)':<14} {str(subs):<48} {len(ids):>3}")
    print()


# ==============================================================================
# Device + cleanup
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
# OLMo / Gemma reference summaries (for comparison printout)
# ==============================================================================
OLMO_REFERENCE_SUMMARY = (
    "v3 1 PASS-arity (sentf->cp L10) then retracted by v4; v4 still 1 cell "
    "(opa->opa L7 lucky); v5 0 PASS-arity. Default canonical at v5 shifts to "
    "nand (100% on most cells)."
)
GEMMA_REFERENCE_SUMMARY = (
    "v3 1 PASS-arity (sentf->opa L4) then retracted by v4; v4 1 PASS-arity "
    "(first->sf L8 lucky) retracted by v5; v5 0 PASS-arity. Default canonical "
    "at v5 shifts to nand or negate."
)


# ==============================================================================
# Main
# ==============================================================================
def main() -> None:
    log_path = _setup_logging()
    print(f"Script 23 - Pythia 6.9B-deduped cross-model replication")
    print(f"STIMULUS_VERSION={STIMULUS_VERSION}  SEED={SEED}  N_PER_CLASS={N_PER_CLASS}")
    print()
    print(f"Model: {PYTHIA_SPEC.short_name} ({PYTHIA_SPEC.model_id})")
    print(f"  dtype: {PYTHIA_SPEC.dtype}")
    print(f"  focus_layers: {PYTHIA_SPEC.focus_layers}")
    print()
    print(f"Canonical set (10): {CANONICALS}")
    print(f"Invented set (16):  {INVENTED_16}")
    print()
    print(f"Scopes under test:")
    print(f"  v3: {len(SCOPE_V3.canonicals)} canonicals + {len(SCOPE_V3.invented_set)} invented "
          f"(comparable to 22b on OLMo/Gemma)")
    print(f"  v4: {len(SCOPE_V4.canonicals)} canonicals + {len(SCOPE_V4.invented_set)} invented "
          f"(comparable to 22c)")
    print(f"  v5: {len(SCOPE_V5.canonicals)} canonicals + {len(SCOPE_V5.invented_set)} invented "
          f"(comparable to 22d; principal analysis)")
    print()

    audit_tokenization()

    device, device_name = get_device()
    print(f"Device: {device_name}")
    print()

    print("=" * 100)
    print("PHASE A: cache extraction")
    print("=" * 100)
    t_total = time.time()

    canon_neut, _, inv_neut, _ = _generate_prompts("NEUTRAL")
    canon_func, _, inv_func, _ = _generate_prompts("FUNC-PFX")

    neut_cache = _cache_load(
        _cache_path(PYTHIA_SPEC.short_name, "NEUTRAL"),
        expected_canon_hash=prompts_checksum(canon_neut),
        expected_inv_hash=prompts_checksum(inv_neut),
        expected_anchors=ANCHORS_NEUTRAL,
    )
    func_cache = _cache_load(
        _cache_path(PYTHIA_SPEC.short_name, "FUNC-PFX"),
        expected_canon_hash=prompts_checksum(canon_func),
        expected_inv_hash=prompts_checksum(inv_func),
        expected_anchors=ANCHORS_FUNC_PFX,
    )

    model = None
    tok = None
    if neut_cache is None or func_cache is None:
        print(f"\n  Loading model: {PYTHIA_SPEC.model_id}")
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(PYTHIA_SPEC.model_id)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            PYTHIA_SPEC.model_id, torch_dtype=PYTHIA_SPEC.dtype, low_cpu_mem_usage=True,
        ).to(device).eval()
        print(f"    loaded in {time.time() - t0:.1f}s "
              f"({model.config.num_hidden_layers} layers, "
              f"hidden_size={model.config.hidden_size})")

    if neut_cache is None:
        print(f"\n  -- Building NEUTRAL condition --")
        neut = build_condition(PYTHIA_SPEC, "NEUTRAL", ANCHORS_NEUTRAL, model, tok, device)
    else:
        neut = neut_cache

    if func_cache is None:
        print(f"\n  -- Building FUNC-PFX condition --")
        func = build_condition(PYTHIA_SPEC, "FUNC-PFX", ANCHORS_FUNC_PFX, model, tok, device)
    else:
        func = func_cache

    if model is not None:
        free_model(model)
    if tok is not None:
        del tok
    gc.collect()

    print(f"\n  -- Phase A total time: {time.time() - t_total:.1f}s --")

    cond_by_name = {"NEUTRAL": neut, "FUNC-PFX": func}

    print()
    print("=" * 100)
    print("PHASE B: full anchor x layer sweep at three scopes (v3, v4, v5)")
    print("=" * 100)

    cells_by_scope: dict[str, list[SweepCell]] = {}
    for scope in (SCOPE_V3, SCOPE_V4, SCOPE_V5):
        t0 = time.time()
        cells = enumerate_cells(scope.name, PYTHIA_SPEC.focus_layers)
        for c in cells:
            run_cell(c, cond_by_name, scope)
        print(f"  scope {scope.name}: {len(cells)} cells in {time.time() - t0:.1f}s")
        cells_by_scope[scope.name] = cells

    print()
    print("=" * 100)
    print("PHASE C: per-scope sweep tables + top-K cells + breakdown")
    print("=" * 100)
    scope_objs = {"v3": SCOPE_V3, "v4": SCOPE_V4, "v5": SCOPE_V5}
    for scope_name in ("v3", "v4", "v5"):
        cells = cells_by_scope[scope_name]
        scope = scope_objs[scope_name]
        print_sweep_table(cells, scope_name)
        top_cells = print_top_k(cells, scope_name, k=8)
        print_canonical_breakdown(cells, scope, top_cell_count=3)

    print()
    print("=" * 100)
    print("PHASE D: candidate-cell falsification chain (within Pythia)")
    print("=" * 100)
    v3_candidates = [c for c in cells_by_scope["v3"] if c.verdict == "PASS-arity"]
    if not v3_candidates:
        print()
        print(f"  No v3-scope PASS-arity cells in Pythia. The falsification chain is")
        print(f"  vacuous: there are no candidate cells to retract under v4/v5. This")
        print(f"  is itself an informative result -- Pythia does not even produce the")
        print(f"  surface-level positive that OLMo/Gemma produced at v3.")
    else:
        print()
        print(f"  v3-PASS candidates ({len(v3_candidates)}). Checking same-cell v4 and v5 verdicts:")
        for v3c in v3_candidates:
            same_v4 = next(
                (c for c in cells_by_scope["v4"]
                 if c.direction == v3c.direction
                 and c.train_anchor == v3c.train_anchor
                 and c.test_anchor == v3c.test_anchor
                 and c.layer == v3c.layer),
                None,
            )
            same_v5 = next(
                (c for c in cells_by_scope["v5"]
                 if c.direction == v3c.direction
                 and c.train_anchor == v3c.train_anchor
                 and c.test_anchor == v3c.test_anchor
                 and c.layer == v3c.layer),
                None,
            )
            print(f"    >> {_cell_short(v3c)}")
            print(f"       v3: M2a={v3c.M2_arity:.3f} M4b={v3c.M4b:.3f} M4c={v3c.M4c:.2f} "
                  f"verdict={v3c.verdict}")
            if same_v4:
                print(f"       v4: M2a={same_v4.M2_arity:.3f} M4b={same_v4.M4b:.3f} "
                      f"M4c={same_v4.M4c:.2f} verdict={same_v4.verdict}")
            if same_v5:
                print(f"       v5: M2a={same_v5.M2_arity:.3f} M4b={same_v5.M4b:.3f} "
                      f"M4c={same_v5.M4c:.2f} verdict={same_v5.verdict}")
            survives_v4 = (same_v4 is not None and same_v4.verdict == "PASS-arity")
            survives_v5 = (same_v5 is not None and same_v5.verdict == "PASS-arity")
            if survives_v5:
                tag = "*** SURVIVES BOTH v4 AND v5 *** (would be a true positive)"
            elif survives_v4:
                tag = "survives v4 but retracted by v5 (10-canonical expansion)"
            elif same_v4 is not None and not survives_v4:
                tag = "retracted by v4 (16-invented expansion)"
            else:
                tag = "incomplete; check above"
            print(f"       verdict: {tag}")

    print()
    print("=" * 100)
    print("PHASE D2: bootstrap M4b on the best v5 cell (if any cross-notation M2-cano PASS)")
    print("=" * 100)
    # Bootstrap M4b on the v5 cell with the highest M2-canonical to characterize
    # the default-to-rarest-canonical mechanism precisely.
    v5_cells = cells_by_scope["v5"]
    eligible = sorted(v5_cells, key=lambda c: -c.M2_cano)
    top_v5 = next(
        (c for c in eligible if c.M2_cano >= GATE_CANONICAL_PASS),
        None,
    )
    if top_v5 is None:
        print(f"  No v5 cell passes M2-canonical >= {GATE_CANONICAL_PASS}; bootstrap skipped.")
    else:
        print(f"  Bootstrap M4b on top-M2c v5 cell: {_cell_short(top_v5)}")
        train_cond_obj = cond_by_name[top_v5.train_cond]
        test_cond_obj = cond_by_name[top_v5.test_cond]
        X_tr, y_tr = slice_canonical(train_cond_obj, top_v5.train_anchor, top_v5.layer)
        X_te, y_te = slice_canonical(test_cond_obj, top_v5.test_anchor, top_v5.layer)
        X_inv, w_inv = slice_invented(test_cond_obj, top_v5.test_anchor, top_v5.layer)
        clf = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs").fit(X_tr, y_tr)
        bs = bootstrap_m4b(
            clf, X_inv, w_inv,
            invented_set=INVENTED_16, w_to_canonical=W_TO_CANONICAL_16,
        )
        print(f"    M2c={top_v5.M2_cano:.3f} M2a={top_v5.M2_arity:.3f}")
        print(f"    M4b point={top_v5.M4b:.3f}, bootstrap mean={bs['mean']:.3f}, "
              f"std={bs['std']:.3f}, 95%CI [{bs['ci95_low']:.3f}, {bs['ci95_high']:.3f}], "
              f"P(>=0.65)={bs['p_pass_065']*100:.1f}%")

    print_headline(
        cells_by_scope,
        olmo_summary=OLMO_REFERENCE_SUMMARY,
        gemma_summary=GEMMA_REFERENCE_SUMMARY,
    )

    if log_path:
        print(f"\n[logging] full transcript written to: {log_path}")


if __name__ == "__main__":
    main()
