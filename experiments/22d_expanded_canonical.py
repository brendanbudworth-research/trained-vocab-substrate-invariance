"""Script 22d - expanded canonical set (10 canonicals: 5 binary + 5 unary).

The principal Platonic-vs-modifier-basin disambiguation experiment.

After script 22c (§3.7.13) tightened the Phase 1 headline to two surviving
PASS-arity cells (§3.7.9 OLMo sente->close L10 and Gemma sente->opera L4),
two interpretive questions remain open:

  1. At the Gemma 2 cell, 67.9% of invented-operator mass lands on
     "necessarily". Is this because the model has converged on a strict
     "logical-unary" abstraction (in which case adding more unary
     canonicals like `possibly`, `always`, `negate` should redistribute
     the mass roughly proportionally), or because the model has converged
     on a generic-modifier abstraction (in which case mass should stay
     concentrated on the modal-adverbial unaries `necessarily` /
     `possibly` / `always` and avoid the operational unaries `not` /
     `negate`)?

  2. At the OLMo 2 §3.7.9 cell, mass is distributed across "and" (65%) +
     "necessarily" (27%) + "not" (8%). Does this distribution persist when
     the canonical set is expanded? Does any of the 5 new canonicals
     attract invented mass at this cell?

Script 22d expands the canonical set to 10 (5 binary + 5 unary, balanced):

  Binary canonicals (5):
    and       (orig)  - high-frequency conjunction
    or        (orig)  - high-frequency disjunction
    implies   (orig)  - logical-binary, low-frequency English
    xor       (new)   - logical-binary, very low-frequency English
    nand      (new)   - logical-binary, very low-frequency English

  Unary canonicals (5):
    not       (orig)  - operational unary, common English negation
    necessarily (orig) - modal-adverbial unary
    possibly  (new)   - modal-adverbial unary (different modality)
    always    (new)   - temporal-quantifier adverbial unary
    negate    (new)   - operational/verb-like unary

The lexical-profile split is deliberate: modal-adverbial unaries
(necessarily, possibly, always) vs operational unaries (not, negate). If
the cross-notation transfer at the Gemma cell is a generic-modifier
abstraction, mass should concentrate on the modal-adverbial subset; if it
is a strict-logical-arity abstraction, mass should distribute across all
5 unaries proportional to their canonical-prior weights.

Invented set is unchanged from 22c (16 words: 8 intended-binary + 8
intended-unary). All 16 invented words map to canonicals in the original
5 (no invented word maps to the 5 new canonicals); this is intentional --
the new canonicals are decoy targets that the model can choose to attract
invented mass to under either hypothesis.

Cells tested: just the 2 survivors from §3.7.13 + 2 lucky-default negative
controls.

New metrics:
  - within-arity Herfindahl (m4c_unary, m4c_binary): concentration of
    mass within the unary (or binary) canonicals only. m4c_unary = 1/5 =
    0.20 means uniform across 5 unaries (strict-arity prediction); 1.00
    means collapsed to one (modifier-basin or default-canonical
    prediction).
  - modal_adverbial_fraction: of the unary mass, what fraction lands on
    {necessarily, possibly, always} vs {not, negate}. Under modifier-
    basin: >= 0.80. Under strict-arity (uniform 3/5 split): ~0.60.
  - new-canonical-uptake: total fraction of invented mass landing on any
    of the 5 new canonicals.

Tees full output to outputs/22d_<ts>.log.
"""

from __future__ import annotations

import datetime as _dt
import gc
import hashlib
import os
import random
import sys
import time
from dataclasses import dataclass

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
    log_path = os.path.join(log_dir, f"22d_{ts}.log")
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

STIMULUS_VERSION = "v5-expanded-canonical"

# Expanded canonical set: 10 canonicals, 5 binary + 5 unary.
CANONICALS = [
    # binary (5):
    "and", "or", "implies", "xor", "nand",
    # unary (5):
    "not", "necessarily", "possibly", "always", "negate",
]
ORIGINAL_5_CANONICALS = ["and", "or", "not", "implies", "necessarily"]
NEW_5_CANONICALS = ["xor", "nand", "possibly", "always", "negate"]

UNARY_CANONICALS = ["not", "necessarily", "possibly", "always", "negate"]
BINARY_CANONICALS = ["and", "or", "implies", "xor", "nand"]
MODAL_ADVERBIAL_UNARY = ["necessarily", "possibly", "always"]
OPERATIONAL_UNARY = ["not", "negate"]

CANONICAL_ARITY = {
    "and": 2, "or": 2, "implies": 2, "xor": 2, "nand": 2,
    "not": 1, "necessarily": 1, "possibly": 1, "always": 1, "negate": 1,
}
assert all(c in CANONICAL_ARITY for c in CANONICALS)

# Invented set: SAME 16 words as 22c (unchanged), each mapped to one of the
# original 5 canonicals (so no invented word maps to a new canonical -- the
# new canonicals are *decoy targets* the model can attract mass to under
# either the strict-arity or modifier-basin hypothesis).
INVENTED_WORDS = [
    "bliq", "dren", "molex",
    "krev", "sond", "glin", "twiv", "fump",
    "vusp", "perph",
    "kelm", "zorf", "gleph", "drelth", "vrith", "nilph",
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
# All invented words map to original-5 canonicals (no new-canonical mapping)
for w, c in W_TO_CANONICAL.items():
    assert c in ORIGINAL_5_CANONICALS, f"{w}->{c} maps to a new canonical"

# Anchor definitions (same as 21/22c)
ANCHORS_FUNC_PFX = ["operator-after", "first-arg", "close-paren", "sentence-final"]
ANCHORS_NEUTRAL = ["operator-after", "sentence-final"]


# ==============================================================================
# Stable seeding helpers (same as 19b/21/22c).
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
# Load 19b module; monkey-patch CANONICALS, CANONICAL_ARITY, INVENTED_WORDS,
# W_TO_CANONICAL so its stimulus generators use the expanded sets.
# ==============================================================================
def _load_19b_module():
    import importlib.machinery
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "19b_directional_angle_gated.py")
    loader = importlib.machinery.SourceFileLoader("_m19b_22d", path)
    spec = importlib.util.spec_from_loader("_m19b_22d", loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_m19b_22d"] = mod
    loader.exec_module(mod)
    return mod


_M19B = _load_19b_module()
_M19B.CANONICALS = CANONICALS
_M19B.CANONICAL_ARITY = CANONICAL_ARITY
_M19B.UNARY_CANONICALS = UNARY_CANONICALS
_M19B.BINARY_CANONICALS = BINARY_CANONICALS
_M19B.INVENTED_WORDS = INVENTED_WORDS
_M19B.W_TO_CANONICAL = W_TO_CANONICAL
make_neutral_stimuli = _M19B.make_neutral_stimuli
make_functional_canonical_stimuli = _M19B.make_functional_canonical_stimuli
make_functional_invented_stimuli = _M19B.make_functional_invented_stimuli
assert _M19B.SEED == SEED
assert _M19B.N_PER_CLASS == N_PER_CLASS


def _generate_prompts(name: str) -> tuple[list[str], list[str], list[str], list[str]]:
    """Generate canonical (10 × N_PER_CLASS) + invented (16 × N_PER_CLASS)
    prompts for one condition. Identical logic to 22c/19b; uses the
    expanded canonical set."""
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
# Model spec (same as 21/22c)
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
# Import anchor finder + extraction from script 21.
# ==============================================================================
def _load_21_module():
    import importlib.machinery
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "21_multi_anchor_battery.py")
    loader = importlib.machinery.SourceFileLoader("_m21_22d", path)
    spec = importlib.util.spec_from_loader("_m21_22d", loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_m21_22d"] = mod
    loader.exec_module(mod)
    return mod


_M21 = _load_21_module()
extract_multi_anchor_activations = _M21.extract_multi_anchor_activations
ConditionMultiAnchor = _M21.ConditionMultiAnchor


# ==============================================================================
# Cache (v5 = standalone from 22c/21 caches; canonical set differs)
# ==============================================================================
def _cache_path(model_short_name: str, condition_name: str) -> str:
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "outputs", "cache")
    os.makedirs(base, exist_ok=True)
    slug = model_short_name.replace(" ", "_")
    return os.path.join(
        base,
        f"22d_{slug}_{condition_name}_npc{N_PER_CLASS}_{STIMULUS_VERSION}.npz",
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
        if list(z["invented_word_list"]) != INVENTED_WORDS:
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
# Cells under test (just the 2 §3.7.13 survivors + 2 lucky-default controls)
# ==============================================================================
@dataclass
class TestCell:
    name: str
    model_short: str
    train_cond: str
    train_anchor: str
    train_layer: int
    test_cond: str
    test_anchor: str
    test_layer: int
    role: str
    # 22c summary (16-word, 5-canonical) for direct comparison
    m4b_22c: float
    m4a_22c: float
    necessarily_share_22c: float


TEST_CELLS: list[TestCell] = [
    TestCell(
        name="§3.7.9 OLMo sente->cp L10 N->F",
        model_short="OLMo 2 7B",
        train_cond="NEUTRAL", train_anchor="sentence-final", train_layer=10,
        test_cond="FUNC-PFX", test_anchor="close-paren", test_layer=10,
        role="survivor-distributed",
        m4b_22c=0.796, m4a_22c=0.349, necessarily_share_22c=0.274,
    ),
    TestCell(
        name="Gemma sente->opa L4 N->F",
        model_short="Gemma 2 9B",
        train_cond="NEUTRAL", train_anchor="sentence-final", train_layer=4,
        test_cond="FUNC-PFX", test_anchor="operator-after", test_layer=4,
        role="survivor-basin",
        m4b_22c=0.669, m4a_22c=0.679, necessarily_share_22c=0.679,
    ),
    TestCell(
        name="[LUCKY-NEG] OLMo opa->opa L7 F->N",
        model_short="OLMo 2 7B",
        train_cond="FUNC-PFX", train_anchor="operator-after", train_layer=7,
        test_cond="NEUTRAL", test_anchor="operator-after", test_layer=7,
        role="lucky-default-control",
        m4b_22c=0.500, m4a_22c=0.000, necessarily_share_22c=0.000,
    ),
    TestCell(
        name="[LUCKY-NEG] Gemma first->sf L8 F->N",
        model_short="Gemma 2 9B",
        train_cond="FUNC-PFX", train_anchor="first-arg", train_layer=8,
        test_cond="NEUTRAL", test_anchor="sentence-final", test_layer=8,
        role="lucky-default-control",
        m4b_22c=0.500, m4a_22c=0.000, necessarily_share_22c=0.000,
    ),
]


# ==============================================================================
# M1-M4 primitives + the new disambiguating metrics
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
    m4c_unary_within: float
    m4c_binary_within: float
    new_canonical_uptake: float
    modal_adverbial_fraction_of_unary: float
    operational_fraction_of_unary: float
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
    breakdown_pct = {c: float(np.mean(preds == c)) for c in CANONICALS}
    m4a = sum(breakdown_pct[c] for c in UNARY_CANONICALS)
    m4c = float(sum(v ** 2 for v in breakdown_pct.values()))

    # Within-arity Herfindahl (only over the 5 unaries or 5 binaries).
    if m4a > 0:
        unary_shares = {c: breakdown_pct[c] / m4a for c in UNARY_CANONICALS}
        m4c_unary = float(sum(v ** 2 for v in unary_shares.values()))
    else:
        m4c_unary = 0.0
    m4_binary_mass = sum(breakdown_pct[c] for c in BINARY_CANONICALS)
    if m4_binary_mass > 0:
        binary_shares = {c: breakdown_pct[c] / m4_binary_mass for c in BINARY_CANONICALS}
        m4c_binary = float(sum(v ** 2 for v in binary_shares.values()))
    else:
        m4c_binary = 0.0

    # New-canonical uptake.
    new_canonical_uptake = sum(breakdown_pct[c] for c in NEW_5_CANONICALS)

    # Modal-adverbial vs operational fraction within the unary mass.
    if m4a > 0:
        modal_share = sum(breakdown_pct[c] for c in MODAL_ADVERBIAL_UNARY) / m4a
        operational_share = sum(breakdown_pct[c] for c in OPERATIONAL_UNARY) / m4a
    else:
        modal_share = 0.0
        operational_share = 0.0

    # Per-stim intended-arity agreement.
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
        m4c_unary_within=m4c_unary,
        m4c_binary_within=m4c_binary,
        new_canonical_uptake=new_canonical_uptake,
        modal_adverbial_fraction_of_unary=modal_share,
        operational_fraction_of_unary=operational_share,
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
    preds = clf.predict(X_inv)
    word_to_idx = {w: np.where(words == w)[0] for w in INVENTED_WORDS}

    m4b_samples = []
    rng = np.random.default_rng(SEED)
    for _ in range(n_bootstrap):
        idx_list = []
        for w in INVENTED_WORDS:
            w_idx = word_to_idx[w]
            idx_list.append(rng.choice(w_idx, size=len(w_idx), replace=True))
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
        "point": None,
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "ci95_low": float(np.percentile(arr, 2.5)),
        "ci95_high": float(np.percentile(arr, 97.5)),
        "p_pass_065": float(np.mean(arr >= M4B_PASS_THRESHOLD)),
    }


def detect_lucky_default(per_word_top_pct: dict[str, float]) -> bool:
    pcts = list(per_word_top_pct.values())
    if not pcts:
        return False
    return min(pcts) >= 0.95


# ==============================================================================
# Slice helpers
# ==============================================================================
def slice_canonical(cond: ConditionMultiAnchor, anchor: str, layer: int):
    a_idx = cond.anchor_names.index(anchor)
    return cond.canonical_X[a_idx, :, layer, :], np.asarray(cond.canonical_labels)


def slice_invented(cond: ConditionMultiAnchor, anchor: str, layer: int):
    a_idx = cond.anchor_names.index(anchor)
    return cond.invented_X[a_idx, :, layer, :], np.asarray(cond.invented_word_per_stim)


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
    bs["point"] = m4.m4b
    lucky = detect_lucky_default(m4.per_word_top_pct)

    return CellResult(
        name=cell.name, m1_train=m1_train, m1_test=m1_test,
        m2_canonical=m2_cano, m2_arity=m2_arity,
        m4=m4, bootstrap_m4b=bs, lucky_default=lucky,
    )


# ==============================================================================
# Reporting
# ==============================================================================
def print_cell_result(cell: TestCell, res: CellResult) -> None:
    print()
    print("=" * 100)
    print(f"  {cell.name}   [{cell.role}]")
    print(f"  model={cell.model_short}  "
          f"train={cell.train_cond}/{cell.train_anchor}@L{cell.train_layer}  "
          f"test={cell.test_cond}/{cell.test_anchor}@L{cell.test_layer}")
    print("=" * 100)
    print()
    print(f"  M1-train CV: {res.m1_train:.3f}  M1-test CV: {res.m1_test:.3f}")
    print(f"  M2-canonical: {res.m2_canonical:.3f}  (10-class accuracy; chance = 0.10; PASS >= 0.65)")
    print(f"  M2-arity:     {res.m2_arity:.3f}  (binary-vs-unary; chance = 0.50; PASS >= 0.65)")
    print()
    print(f"  ===== M4 (16 × 50 = 800 invented stims, 10-canonical readout) =====")
    print(f"  M4a (inv unary mass):           {res.m4.m4a*100:5.1f}%   (22c-5cano: {cell.m4a_22c*100:.1f}%)")
    print(f"  M4b (intended-arity agree):     {res.m4.m4b*100:5.1f}%   (22c-5cano: {cell.m4b_22c*100:.1f}%)")
    print(f"  M4c (Herfindahl over 10):       {res.m4.m4c:5.2f}   (0.10 = uniform; 1.00 = collapsed)")
    print(f"  M4c-unary (within 5 unaries):   {res.m4.m4c_unary_within:5.2f}   "
          f"(0.20 = uniform across 5 unaries; 1.00 = all on one)")
    print(f"  M4c-binary (within 5 binaries): {res.m4.m4c_binary_within:5.2f}   "
          f"(0.20 = uniform across 5 binaries; 1.00 = all on one)")
    print()
    print(f"  Bootstrap M4b: mean {res.bootstrap_m4b['mean']:.3f} "
          f"std {res.bootstrap_m4b['std']:.3f}  "
          f"95% CI [{res.bootstrap_m4b['ci95_low']:.3f}, {res.bootstrap_m4b['ci95_high']:.3f}]  "
          f"P(>= {M4B_PASS_THRESHOLD:.2f}) = {res.bootstrap_m4b['p_pass_065']:.1%}")
    print()
    print(f"  ===== Disambiguating metrics =====")
    print(f"  new-canonical uptake (mass on the 5 NEW canonicals):")
    print(f"    total: {res.m4.new_canonical_uptake*100:5.1f}%")
    new_breakdown = {c: res.m4.breakdown_pct[c] for c in NEW_5_CANONICALS}
    for c in NEW_5_CANONICALS:
        arity_tag = "B" if CANONICAL_ARITY[c] == 2 else "U"
        bar = int(new_breakdown[c] * 40)
        print(f"      {c:12s} ({arity_tag}) {new_breakdown[c]*100:5.1f}%  {'#' * bar}")
    print()
    if res.m4.m4a > 0:
        print(f"  Within-unary breakdown:")
        print(f"    modal-adverbial fraction ({', '.join(MODAL_ADVERBIAL_UNARY)}): "
              f"{res.m4.modal_adverbial_fraction_of_unary*100:5.1f}%")
        print(f"    operational fraction    ({', '.join(OPERATIONAL_UNARY)}): "
              f"{res.m4.operational_fraction_of_unary*100:5.1f}%")
    else:
        print(f"  Within-unary breakdown: N/A (M4a = 0%)")
    print()
    print(f"  Full predicted-canonical breakdown (over 800 invented stims):")
    for c in CANONICALS:
        arity_tag = "B" if CANONICAL_ARITY[c] == 2 else "U"
        is_new = " (NEW)" if c in NEW_5_CANONICALS else ""
        bar = int(res.m4.breakdown_pct[c] * 40)
        print(f"    {c:13s} ({arity_tag}){is_new:6s} {res.m4.breakdown_pct[c]*100:5.1f}%  {'#' * bar}")
    print()
    print(f"  Per-word breakdown (16 words):")
    print(f"    {'word':<10} {'int.':>4} {'int.cano':<11} {'top':<13} {'top%':>5} {'unary%':>6} {'match':<5}")
    print(f"    {'-'*10} {'-'*4} {'-'*11} {'-'*13} {'-'*5} {'-'*6} {'-'*5}")
    for w in INVENTED_WORDS:
        intended_can = W_TO_CANONICAL[w]
        intended_arity = "B" if CANONICAL_ARITY[intended_can] == 2 else "U"
        top_c = res.m4.per_word_top.get(w, "—")
        top_pct = res.m4.per_word_top_pct.get(w, 0.0)
        unary_pct = res.m4.per_word_unary_pct.get(w, 0.0)
        is_match = (
            CANONICAL_ARITY[top_c] == CANONICAL_ARITY[intended_can]
            if top_c in CANONICAL_ARITY else False
        )
        is_new_top = "*" if top_c in NEW_5_CANONICALS else " "
        print(f"    {w:<10} {intended_arity:>4} {intended_can:<11} "
              f"{top_c:<13} {top_pct*100:5.1f} {unary_pct*100:5.1f}  "
              f"{'YES' if is_match else 'no':<5} {is_new_top}")
    print()
    print(f"  Lucky-default flag: {'YES' if res.lucky_default else 'no'}")
    print()


def print_summary_and_interpretation(
    cells: list[TestCell], results: list[CellResult],
) -> None:
    print()
    print("=" * 100)
    print("SUMMARY: M4b across 22c (5-canonical) and 22d (10-canonical) readouts")
    print("=" * 100)
    print()
    print(f"  {'cell':<48} {'role':<22} {'M4b(22c)':>9} {'M4b(22d)':>9} "
          f"{'new-cano%':>10} {'unary-Hfdl':>11}")
    print(f"  {'-'*48} {'-'*22} {'-'*9} {'-'*9} {'-'*10} {'-'*11}")
    for cell, res in zip(cells, results):
        print(f"  {cell.name[:48]:<48} {cell.role:<22} "
              f"{cell.m4b_22c:9.3f} {res.m4.m4b:9.3f} "
              f"{res.m4.new_canonical_uptake*100:9.1f}% {res.m4.m4c_unary_within:11.2f}")
    print()

    print("=" * 100)
    print("INTERPRETATION: strict-logical-arity vs generic-modifier-basin")
    print("=" * 100)
    print()
    print("  Reading guide:")
    print(f"    M4b 22d ≈ M4b 22c: arity-respecting transfer is invariant to canonical")
    print(f"                       set expansion (necessary for the strict-arity claim).")
    print(f"    M4b 22d << M4b 22c: arity claim was canonical-set-specific (5-word artifact).")
    print()
    print(f"    new-cano % > 5%:   model is using the new canonicals (some redistribution).")
    print(f"    new-cano % near 0: invented mass stays on the original 5 canonicals.")
    print()
    print(f"    M4c-unary near 0.20: invented unary mass distributes uniformly across the 5")
    print(f"                         unaries (strict-logical-arity prediction).")
    print(f"    M4c-unary near 1.00: invented unary mass collapses onto one canonical")
    print(f"                         (modifier-basin or default-canonical prediction).")
    print()
    print(f"    modal-adv frac of unary > 0.80: modifier-basin reading (mass on modal-")
    print(f"                                    adverbial necessarily/possibly/always).")
    print(f"    modal-adv frac of unary ≈ 0.60: strict-arity reading (uniform 3/5 split).")
    print(f"    modal-adv frac of unary < 0.40: operational reading (mass on not/negate).")
    print()
    print("  Per-cell verdict:")
    for cell, res in zip(cells, results):
        if cell.role == "lucky-default-control":
            continue
        print()
        print(f"  >> {cell.name}  [{cell.role}]")
        m4b_delta = res.m4.m4b - cell.m4b_22c
        m4b_survives = res.m4.m4b >= M4B_PASS_THRESHOLD
        print(f"     M4b 22d = {res.m4.m4b:.3f}  (22c was {cell.m4b_22c:.3f}; delta {m4b_delta:+.3f})")
        print(f"     M4b PASS at 22d: {'YES' if m4b_survives else 'NO'}")
        print(f"     new-canonical uptake: {res.m4.new_canonical_uptake*100:.1f}% of invented mass")
        print(f"     M4c-unary: {res.m4.m4c_unary_within:.2f} (0.20=uniform, 1.00=collapsed)")
        print(f"     modal-adverbial fraction of unary: "
              f"{res.m4.modal_adverbial_fraction_of_unary*100:.1f}%")
        print(f"     operational fraction of unary:     "
              f"{res.m4.operational_fraction_of_unary*100:.1f}%")
        print()

        # Heuristic verdict.
        if not m4b_survives:
            verdict = ("ARITY CLAIM FAILS 22d: M4b drops below 0.65 when canonical set is "
                       "expanded. The 22c PASS was specific to the original 5-canonical "
                       "readout. Retract the cross-notation arity-respecting transfer claim "
                       "at this cell.")
        elif res.m4.m4c_unary_within <= 0.30 and res.m4.modal_adverbial_fraction_of_unary <= 0.70:
            verdict = ("STRICT-ARITY: invented unary mass distributes broadly across the 5 "
                       "unaries (M4c-unary near 0.20) with mixed modal-adverbial and "
                       "operational components. Consistent with the model encoding logical "
                       "arity as an abstract feature, not a generic modifier prior.")
        elif res.m4.m4c_unary_within >= 0.60 or res.m4.modal_adverbial_fraction_of_unary >= 0.85:
            verdict = ("MODIFIER-BASIN: invented unary mass collapses onto one (or the modal-"
                       "adverbial subset of) unary canonicals. The cross-notation transfer "
                       "reads as a generic-modifier abstraction rather than a strict logical-"
                       "arity abstraction. Section 4.1's partial-Platonic framing needs the "
                       "modifier-basin caveat.")
        else:
            verdict = ("MIXED / UNDETERMINED: the within-unary Herfindahl and modal-adverbial "
                       "fraction land in the intermediate range. Neither strict-arity nor "
                       "modifier-basin reading is cleanly supported; report numbers without "
                       "a strong qualitative interpretation.")
        print(f"     => Verdict: {verdict}")
    print()

    # Lucky-default sanity check.
    lucky_cells = [(c, r) for c, r in zip(cells, results) if c.role == "lucky-default-control"]
    if lucky_cells:
        print()
        print(f"  Lucky-default negative controls (sanity check):")
        for c, r in lucky_cells:
            still_lucky = r.lucky_default
            print(f"    {c.name}: M4b 22c={c.m4b_22c:.3f}, M4b 22d={r.m4.m4b:.3f}, "
                  f"lucky-default-flag={'YES' if still_lucky else 'no'}, "
                  f"new-cano%={r.m4.new_canonical_uptake*100:.1f}%")
        print(f"    (Expected: M4b stays at floor 0.50; lucky-default flag may flip if the")
        print(f"     default-canonical shifts when extra canonicals are added.)")
        print()


# ==============================================================================
# Tokenization audit
# ==============================================================================
def audit_tokenization() -> None:
    print("=" * 100)
    print("TOKENIZATION AUDIT (10 canonicals + 16 invented words under both tokenizers)")
    print("=" * 100)
    print()
    for spec in MODEL_SPECS:
        print(f"  {spec.short_name} ({spec.model_id})")
        try:
            tok = AutoTokenizer.from_pretrained(spec.model_id)
        except Exception as e:
            print(f"    [tokenizer load failed: {e}]")
            continue
        print(f"    CANONICALS:")
        print(f"      {'word':<14} {'subwords':<40} {'n':>3}")
        for c in CANONICALS:
            ids = tok.encode(" " + c, add_special_tokens=False)
            subs = [tok.decode([i]) for i in ids]
            is_new = " (NEW)" if c in NEW_5_CANONICALS else ""
            arity_tag = " (B)" if CANONICAL_ARITY[c] == 2 else " (U)"
            print(f"      {c+arity_tag:<14} {str(subs):<40} {len(ids):>3}{is_new}")
        print(f"    (invented words: see 22c tokenization audit -- unchanged here)")
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
# Main
# ==============================================================================
def main() -> None:
    log_path = _setup_logging()
    print(f"Script 22d - expanded canonical set (10 canonicals: 5B + 5U)")
    print(f"STIMULUS_VERSION={STIMULUS_VERSION}  SEED={SEED}  N_PER_CLASS={N_PER_CLASS}")
    print()
    print(f"Canonical set:   {CANONICALS}")
    print(f"  ({len([c for c in CANONICALS if CANONICAL_ARITY[c]==2])} binary, "
          f"{len([c for c in CANONICALS if CANONICAL_ARITY[c]==1])} unary)")
    print(f"Original 5:      {ORIGINAL_5_CANONICALS}")
    print(f"New 5:           {NEW_5_CANONICALS}")
    print(f"  - new binaries:  xor, nand")
    print(f"  - new unaries:   possibly, always, negate")
    print(f"  - modal-adv U:   {MODAL_ADVERBIAL_UNARY}")
    print(f"  - operational U: {OPERATIONAL_UNARY}")
    print()
    print(f"Invented set (16, unchanged from 22c): {INVENTED_WORDS}")
    print()
    audit_tokenization()

    device, device_name = get_device()
    print(f"Device: {device_name}")
    print()

    cells_by_model: dict[str, list[TestCell]] = {}
    for cell in TEST_CELLS:
        cells_by_model.setdefault(cell.model_short, []).append(cell)

    all_results: list[CellResult] = []
    cells_in_order: list[TestCell] = []

    for spec in MODEL_SPECS:
        if spec.short_name not in cells_by_model:
            continue
        cells = cells_by_model[spec.short_name]
        print("=" * 100)
        print(f"MODEL: {spec.short_name}")
        print("=" * 100)
        t_total = time.time()

        canon_neut, _, inv_neut, _ = _generate_prompts("NEUTRAL")
        canon_func, _, inv_func, _ = _generate_prompts("FUNC-PFX")

        neut_cache = _cache_load(
            _cache_path(spec.short_name, "NEUTRAL"),
            expected_canon_hash=prompts_checksum(canon_neut),
            expected_inv_hash=prompts_checksum(inv_neut),
            expected_anchors=ANCHORS_NEUTRAL,
        )
        func_cache = _cache_load(
            _cache_path(spec.short_name, "FUNC-PFX"),
            expected_canon_hash=prompts_checksum(canon_func),
            expected_inv_hash=prompts_checksum(inv_func),
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
            neut = build_condition(spec, "NEUTRAL", ANCHORS_NEUTRAL, model, tok, device)
        else:
            neut = neut_cache

        if func_cache is None:
            print(f"\n  -- Building FUNC-PFX condition --")
            func = build_condition(spec, "FUNC-PFX", ANCHORS_FUNC_PFX, model, tok, device)
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

    print_summary_and_interpretation(cells_in_order, all_results)

    if log_path:
        print(f"[logging] full transcript written to: {log_path}")


if __name__ == "__main__":
    main()
