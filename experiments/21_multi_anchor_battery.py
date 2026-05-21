"""Script 21 - multi-anchor M1-M4 battery (post-call anchor re-test).

Highest-priority Phase 1 experiment, derived from the script 20 peer review.

The script 17-20 measurement window is the operator-anchored position
(i+1 after the last operator subword). In functional-prefix notation this
position lands on the `(` token *before* the argument list. In a causal
LM the residual stream at that position cannot attend to future tokens
(`p`, `,`, `q`, `)`). For canonical operators the model has a pre-
training lexical prior to fall back on; for invented operators no such
prior exists. The script-20 negative result on M4b (per-invented-word
intended-arity agreement) in FUNC-PFX is therefore current strong
evidence about early post-operator catchment geometry but only weak
evidence about full functional-call arity induction.

Script 21 re-runs the M1-M4 battery at multiple anchor positions:

  FUNC-PFX anchors (4):
    operator-after  i+1 after last operator subword     (control vs 17-20)
    first-arg       position of the first argument token (model has seen first arg)
    close-paren     position of the closing ')' token   (model has seen full call)
    sentence-final  last token of full sentence         (globally integrated)

  NEUTRAL anchors (2):
    operator-after  i+1 after last operator subword     (control vs 17-20)
    sentence-final  last token of full sentence         (post-integration)

  (NEUTRAL has no argument list, so first-arg / close-paren are not
   meaningful in that condition.)

One forward pass per stimulus extracts all layer activations at every
needed anchor position simultaneously. Cache stores per-condition
(n_anchors, n_stim, n_layers, dim) float16 with anchor metadata so any
sub-cell (anchor, layer) can be re-loaded instantly.

Reports the full M1-M4 battery per (model, anchor, focus_layer):

  M1   within-condition probe CV accuracy
  M2   bidirectional canonical-transfer gate (acc both directions)
  M3   cross-notation directional angle (centroid + probe)
  M4a  invented unary mass under cross-condition probe transfer
  M4b  intended-arity agreement (per-invented-word predicted canonical
       matches the intended-canonical's arity)
  M4c  canonical catchment concentration (Herfindahl over predicted-
       canonical fractions; 1.0 = total single-canonical collapse,
       0.20 = perfectly uniform across 5 canonicals)

Plus per-invented-word per-anchor predicted canonical (so we can see
exactly which words switch arity classes as anchor depth changes).

Headline question: does M4b pass (e.g., ≥ 0.65) at any anchor in
either model? If yes at close-paren / sentence-final, the §3.7.8
negative-result headline is retracted and a positive cross-notation
arity-respecting transfer finding takes its place. If no at any anchor,
the negative result becomes substantially stronger.

Uses the same stable-seed stimulus generation as 19b/20 (v3-multi-anchor
since the cache shape changes). Tees output to outputs/21_<ts>.log.
"""

from __future__ import annotations

import datetime as _dt
import gc
import hashlib
import os
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
    log_path = os.path.join(log_dir, f"21_{ts}.log")
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
N_BOOTSTRAP = 100
GATE_THRESHOLD = 0.65
M4B_PASS_THRESHOLD = 0.65  # context-grounded arity-respecting transfer

STIMULUS_VERSION = "v3-multi-anchor"  # bumped from 19b's v2 (cache shape changes)

CANONICALS = ["and", "or", "not", "implies", "necessarily"]
UNARY_CANONICALS = ["not", "necessarily"]
BINARY_CANONICALS = ["and", "or", "implies"]
CANONICAL_ARITY = {"and": 2, "or": 2, "implies": 2, "not": 1, "necessarily": 1}

INVENTED_WORDS = ["bliq", "dren", "vusp", "molex", "perph"]
W_TO_CANONICAL = dict(zip(INVENTED_WORDS, CANONICALS))

# Anchor definitions
ANCHORS_FUNC_PFX = ["operator-after", "first-arg", "close-paren", "sentence-final"]
ANCHORS_NEUTRAL = ["operator-after", "sentence-final"]


# ==============================================================================
# Stable seeding helpers (same as 19b v2).
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


# Reuse stimulus-generation primitives from 19b (templates, _generate_prompts,
# etc.) via importlib to avoid duplication. The 19b stimulus templates and
# generators are stable across v2/v3 (we only bump the cache shape, not the
# stimuli).
def _load_19b_module():
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


_M19B = _load_19b_module()
make_neutral_stimuli = _M19B.make_neutral_stimuli
make_functional_canonical_stimuli = _M19B.make_functional_canonical_stimuli
make_functional_invented_stimuli = _M19B.make_functional_invented_stimuli
_generate_prompts = _M19B._generate_prompts

assert _M19B.SEED == SEED
assert _M19B.N_PER_CLASS == N_PER_CLASS


# ==============================================================================
# Model spec - same as 19b.
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
# Multi-anchor position finder.
# ==============================================================================
def find_operator_subword_idx(decoded_tokens: list[str], operators: list[str]) -> int | None:
    """Return the index of the LAST subword of any operator in the prompt.
    Same matching logic as 19b's find_operator_anchor but returns the
    operator-subword index itself, not i+1.
    """
    best_idx = len(decoded_tokens)
    for op in operators:
        target = " " + op
        joined = ""
        for i, t in enumerate(decoded_tokens):
            joined += t
            if joined.endswith(target):
                if i < best_idx:
                    best_idx = i
                break
            # Reset joined when the running suffix can no longer match
            # any operator's leading-space prefix.
            # (Simple version: always extend; the suffix-match handles it.)
    return best_idx if best_idx < len(decoded_tokens) else None


def _find_token_after(decoded_tokens: list[str], char: str, start: int) -> int | None:
    """Find the index of the first token containing `char` at or after `start`."""
    for i in range(start, len(decoded_tokens)):
        if char in decoded_tokens[i]:
            return i
    return None


def compute_anchor_positions(
    tok, prompt: str, operators: list[str], condition: str,
) -> dict[str, int]:
    """Return a dict {anchor_name -> token index}.

    Missing anchors (e.g., close-paren not findable) are not present in
    the dict; callers should default to `sentence-final` for missing
    anchors but log it as anomalous since the stimulus generator should
    produce well-formed FUNC-PFX strings.
    """
    ids = tok(prompt, return_tensors="pt").input_ids[0].tolist()
    decoded = [tok.decode([i]) for i in ids]
    n_tok = len(ids)

    out: dict[str, int] = {}
    out["sentence-final"] = n_tok - 1

    op_subword_idx = find_operator_subword_idx(decoded, operators)
    if op_subword_idx is None:
        return out  # operator not found - use sentence-final as fallback everywhere

    op_after = op_subword_idx + 1 if op_subword_idx + 1 < n_tok else op_subword_idx
    out["operator-after"] = op_after

    if condition == "FUNC-PFX":
        paren_open_idx = _find_token_after(decoded, "(", op_subword_idx)
        if paren_open_idx is not None:
            first_arg_idx = paren_open_idx + 1 if paren_open_idx + 1 < n_tok else None
            paren_close_idx = _find_token_after(decoded, ")", paren_open_idx + 1)

            if first_arg_idx is not None:
                out["first-arg"] = first_arg_idx
            if paren_close_idx is not None:
                out["close-paren"] = paren_close_idx

    return out


# ==============================================================================
# Multi-anchor activation extraction.
# Output shape: (n_anchors, n_stim, n_layers, dim) float32 (downcast to fp16
# for disk).
# ==============================================================================
def extract_multi_anchor_activations(
    model, tok, prompts: list[str], operators: list[str],
    anchor_names: list[str], condition: str, device: str,
) -> tuple[np.ndarray, list[list[int]], int]:
    """Returns (activations, anchor_positions_per_stim, n_layers).

    activations.shape == (n_anchors, n_stim, n_layers, dim).
    anchor_positions_per_stim[stim_idx][anchor_idx] = token index actually
        used for this stimulus / anchor (sentence-final fallback if the
        intended anchor wasn't findable, with a warning printed).
    """
    n_anchors = len(anchor_names)
    n_stim = len(prompts)
    n_layers = None
    dim = None
    activations = None
    anchor_positions_per_stim: list[list[int]] = []
    n_fallback_per_anchor = {a: 0 for a in anchor_names}

    for stim_idx, p in enumerate(prompts):
        enc = tok(p, return_tensors="pt").to(device)
        seq_len = enc.input_ids.shape[1]
        anchors = compute_anchor_positions(tok, p, operators, condition)

        positions_per_anchor: list[int] = []
        for a in anchor_names:
            pos = anchors.get(a, None)
            if pos is None or pos >= seq_len:
                pos = seq_len - 1
                n_fallback_per_anchor[a] += 1
            positions_per_anchor.append(pos)
        anchor_positions_per_stim.append(positions_per_anchor)

        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        if device == "mps":
            torch.mps.synchronize()

        if activations is None:
            n_layers = len(out.hidden_states)
            dim = out.hidden_states[0].shape[-1]
            activations = np.zeros((n_anchors, n_stim, n_layers, dim), dtype=np.float32)

        for L, h in enumerate(out.hidden_states):
            for a_idx, pos in enumerate(positions_per_anchor):
                activations[a_idx, stim_idx, L, :] = h[0, pos, :].float().cpu().numpy()

    for a, n in n_fallback_per_anchor.items():
        if n > 0:
            print(f"    [anchor] {a}: {n}/{n_stim} stimuli fell back to "
                  f"sentence-final (anchor not findable)")

    return activations, anchor_positions_per_stim, n_layers


@dataclass
class ConditionMultiAnchor:
    """One model + one condition's multi-anchor activations + metadata."""
    canonical_X: np.ndarray  # (n_anchors, n_stim, n_layers, dim)
    canonical_labels: np.ndarray  # (n_stim,)
    invented_X: np.ndarray  # (n_anchors, n_stim, n_layers, dim)
    invented_word_per_stim: np.ndarray  # (n_stim,)
    anchor_names: list[str]
    canon_anchor_positions: list[list[int]]
    inv_anchor_positions: list[list[int]]


# ==============================================================================
# Disk cache for multi-anchor activations.
# ==============================================================================
def _cache_path(model_short_name: str, condition_name: str) -> str:
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "outputs", "cache")
    os.makedirs(base, exist_ok=True)
    slug = model_short_name.replace(" ", "_")
    return os.path.join(
        base,
        f"21_{slug}_{condition_name}_npc{N_PER_CLASS}_{STIMULUS_VERSION}.npz",
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
              f"shape={canon_X.shape}, anchors={cached_anchors}")
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
    canonical_stim_fn, invented_stim_fn,
) -> ConditionMultiAnchor:
    canon_prompts, canon_labels, inv_prompts, inv_words = _generate_prompts(
        condition_name, canonical_stim_fn, invented_stim_fn
    )
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
# M1-M4 primitives.
# ==============================================================================
def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v if n < 1e-12 else v / n


def cosine_angle_deg(u: np.ndarray, v: np.ndarray) -> tuple[float, float]:
    cos = float(np.dot(u, v))
    cos = max(-1.0, min(1.0, cos))
    return cos, float(np.degrees(np.arccos(cos)))


def centroid_unary_direction(canon_X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    u = np.stack([canon_X[labels == op].mean(axis=0) for op in UNARY_CANONICALS])
    b = np.stack([canon_X[labels == op].mean(axis=0) for op in BINARY_CANONICALS])
    return unit(u.mean(axis=0) - b.mean(axis=0))


def raw_binary_probe_direction(canon_X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    y = np.array([1 if op in UNARY_CANONICALS else 0 for op in labels])
    clf = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs").fit(canon_X, y)
    return unit(clf.coef_[0])


def within_cond_5class_cv(X: np.ndarray, y: np.ndarray, seed: int = 0) -> float:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = []
    for train_idx, test_idx in skf.split(X, y):
        clf = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs")
        clf.fit(X[train_idx], y[train_idx])
        scores.append(clf.score(X[test_idx], y[test_idx]))
    return float(np.mean(scores))


def canonical_transfer_accuracy(
    X_train, y_train, X_test, y_test,
) -> float:
    clf = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs")
    clf.fit(X_train, y_train)
    return float(clf.score(X_test, y_test))


def gate_verdict(acc_n2f: float, acc_f2n: float) -> str:
    pass_thr = GATE_THRESHOLD
    ambig_thr = 0.30
    n_pass = (acc_n2f >= pass_thr) + (acc_f2n >= pass_thr)
    n_fail = (acc_n2f < ambig_thr) + (acc_f2n < ambig_thr)
    if n_pass == 2:
        return "PASS"
    if n_fail >= 1:
        return "FAIL"
    return "AMBIG"


# ==============================================================================
# M4a / M4b / M4c.
# ==============================================================================
def invented_breakdown(
    train_X: np.ndarray, train_y: np.ndarray,
    test_inv_X: np.ndarray, inv_words: np.ndarray,
) -> dict:
    clf = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs").fit(train_X, train_y)
    preds = clf.predict(test_inv_X)
    n = len(preds)
    n_unary = int(np.sum(np.isin(preds, UNARY_CANONICALS)))
    canon_counts = {c: int(np.sum(preds == c)) for c in CANONICALS}
    canon_pct = {c: canon_counts[c] / n for c in CANONICALS}

    # M4b: intended-arity agreement
    n_agree = 0
    per_word_top: dict[str, str] = {}
    per_word_unary_pct: dict[str, float] = {}
    for w in INVENTED_WORDS:
        mask = np.array([iw == w for iw in inv_words])
        if mask.sum() == 0:
            continue
        w_preds = preds[mask]
        intended_canonical = W_TO_CANONICAL[w]
        intended_arity = CANONICAL_ARITY[intended_canonical]
        for pred in w_preds:
            if CANONICAL_ARITY[pred] == intended_arity:
                n_agree += 1
        w_counts = {c: int(np.sum(w_preds == c)) for c in CANONICALS}
        per_word_top[w] = max(w_counts, key=lambda c: w_counts[c])
        per_word_unary_pct[w] = sum(w_counts[c] for c in UNARY_CANONICALS) / len(w_preds)

    m4b = n_agree / n if n > 0 else 0.0

    # M4c: Herfindahl over canonical fractions
    m4c = sum(p ** 2 for p in canon_pct.values())

    return {
        "n": n,
        "M4a_unary_mass": n_unary / n,
        "M4b_intended_arity_agree": m4b,
        "M4c_herfindahl": m4c,
        "canon_pct": canon_pct,
        "per_word_top": per_word_top,
        "per_word_unary_pct": per_word_unary_pct,
    }


# ==============================================================================
# Per-anchor x per-focus-layer analysis.
# Each cell reports: M1 (within-cond CV both notations), M2 (gate both
# directions + verdict), M3 (centroid + probe angle), and the full M4
# breakdown for N->F and F->N invented transfer.
# ==============================================================================
def slice_anchor_layer(cond: ConditionMultiAnchor, anchor_idx: int, layer: int):
    """Return (canonical_X_slice, invented_X_slice) for one (anchor, layer)."""
    return cond.canonical_X[anchor_idx, :, layer, :], cond.invented_X[anchor_idx, :, layer, :]


def analyse_model(
    spec: ModelSpec, neut: ConditionMultiAnchor, func: ConditionMultiAnchor,
) -> dict:
    print()
    print("=" * 100)
    print(f"  {spec.short_name} - MULTI-ANCHOR M1-M4 BATTERY")
    print("=" * 100)

    results: list[dict] = []

    # For each FUNC-PFX anchor, we test against the SAME-NAMED NEUTRAL
    # anchor where possible. NEUTRAL has only operator-after and
    # sentence-final, so:
    #   FUNC-PFX operator-after  <-> NEUTRAL operator-after
    #   FUNC-PFX first-arg       <-> NEUTRAL sentence-final  (no NEUTRAL equiv)
    #   FUNC-PFX close-paren     <-> NEUTRAL sentence-final  (no NEUTRAL equiv)
    #   FUNC-PFX sentence-final  <-> NEUTRAL sentence-final
    #
    # For first-arg / close-paren the NEUTRAL counterpart is technically
    # sentence-final, since NEUTRAL doesn't have the intermediate
    # positions. We report this explicitly so the reviewer can see the
    # asymmetry.
    func_to_neut: dict[str, str] = {
        "operator-after": "operator-after",
        "first-arg": "sentence-final",  # no NEUTRAL equivalent
        "close-paren": "sentence-final",  # no NEUTRAL equivalent
        "sentence-final": "sentence-final",
    }

    print()
    print(f"  Per-anchor x per-focus-layer M1-M4 battery.")
    print(f"  NEUTRAL anchor mapping: " + ", ".join(
        f"{f}->{n}" for f, n in func_to_neut.items()
    ))
    print(f"  Focus layers: {spec.focus_layers}")
    print()
    header = (
        "  anchor (F-PFX <- NEUT)        | L  | M1_N | M1_F | M2_N->F | M2_F->N | M3_cent | "
        "M3_prob | M4a_N->F | M4b_N->F | M4c_N->F | M4a_F->N | M4b_F->N | M4c_F->N | M2"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for f_anchor in func.anchor_names:
        n_anchor = func_to_neut[f_anchor]
        f_a_idx = func.anchor_names.index(f_anchor)
        n_a_idx = neut.anchor_names.index(n_anchor)

        for L in spec.focus_layers:
            Xc_n, Xi_n = slice_anchor_layer(neut, n_a_idx, L)
            Xc_f, Xi_f = slice_anchor_layer(func, f_a_idx, L)
            y_n = neut.canonical_labels
            y_f = func.canonical_labels

            m1_n = within_cond_5class_cv(Xc_n, y_n)
            m1_f = within_cond_5class_cv(Xc_f, y_f)
            gate_n2f = canonical_transfer_accuracy(Xc_n, y_n, Xc_f, y_f)
            gate_f2n = canonical_transfer_accuracy(Xc_f, y_f, Xc_n, y_n)
            verdict = gate_verdict(gate_n2f, gate_f2n)

            d_n = centroid_unary_direction(Xc_n, y_n)
            d_f = centroid_unary_direction(Xc_f, y_f)
            _, deg_c = cosine_angle_deg(d_n, d_f)
            w_n = raw_binary_probe_direction(Xc_n, y_n)
            w_f = raw_binary_probe_direction(Xc_f, y_f)
            _, deg_p = cosine_angle_deg(w_n, w_f)

            bd_n2f = invented_breakdown(
                Xc_n, y_n, Xi_f, func.invented_word_per_stim
            )
            bd_f2n = invented_breakdown(
                Xc_f, y_f, Xi_n, neut.invented_word_per_stim
            )

            print(f"  {f_anchor:<13} <- {n_anchor:<13} | {L:>2} | "
                  f"{m1_n:.2f} | {m1_f:.2f} | {gate_n2f:>7.3f} | {gate_f2n:>7.3f} | "
                  f"{deg_c:>5.1f}° | {deg_p:>5.1f}° | "
                  f"{bd_n2f['M4a_unary_mass']:>7.1%} | {bd_n2f['M4b_intended_arity_agree']:>7.1%} | "
                  f"{bd_n2f['M4c_herfindahl']:>7.2f} | "
                  f"{bd_f2n['M4a_unary_mass']:>7.1%} | {bd_f2n['M4b_intended_arity_agree']:>7.1%} | "
                  f"{bd_f2n['M4c_herfindahl']:>7.2f} | {verdict}")

            results.append({
                "model": spec.short_name,
                "f_anchor": f_anchor, "n_anchor": n_anchor, "layer": L,
                "m1_n": m1_n, "m1_f": m1_f,
                "gate_n2f": gate_n2f, "gate_f2n": gate_f2n,
                "verdict": verdict,
                "deg_centroid": deg_c, "deg_probe": deg_p,
                "n2f": bd_n2f, "f2n": bd_f2n,
            })

        print()  # blank between anchor blocks

    # Per-anchor x per-word predicted-canonical detail tables
    print()
    print(f"  Per-invented-word predicted canonical (FUNC-PFX direction only,")
    print(f"  N->F transfer at each anchor; intended-arity at top of each column):")
    print()
    intended = " ".join(
        f"{w}={'U' if CANONICAL_ARITY[W_TO_CANONICAL[w]] == 1 else 'B'}"
        for w in INVENTED_WORDS
    )
    print(f"  ({intended})")
    print()
    detail_header = "  anchor | L  | " + " | ".join(f"{w:<14}" for w in INVENTED_WORDS)
    print(detail_header)
    print("  " + "-" * (len(detail_header) - 2))
    for r in results:
        cells = []
        for w in INVENTED_WORDS:
            top = r["n2f"]["per_word_top"].get(w, "?")
            unary_pct = r["n2f"]["per_word_unary_pct"].get(w, 0.0)
            cells.append(f"{top:<10}{unary_pct:>3.0%}")
        print(f"  {r['f_anchor']:<13} | {r['layer']:>2} | "
              + " | ".join(f"{c:<14}" for c in cells))

    return {"model": spec.short_name, "rows": results}


# ==============================================================================
# Cross-anchor headline summary.
# ==============================================================================
def headline_summary(per_model_results: list[dict]) -> None:
    print()
    print("=" * 100)
    print("CROSS-ANCHOR HEADLINE: does M4b (intended-arity agreement) pass anywhere?")
    print("=" * 100)
    print()
    print(f"  Threshold for 'M4b PASS': intended-arity agreement >= "
          f"{M4B_PASS_THRESHOLD:.2f}")
    print(f"  Random-by-arity baseline: 2 unary + 3 binary canonicals; "
          f"if invented split 2-unary / 3-binary intended, random matching = "
          f"(2*2 + 3*3) / (5*5) = 0.52")
    print(f"  So M4b PASS at >= 0.65 is ~25% above random-by-arity baseline.")
    print()
    print(f"  Best M4b across all (model, anchor, layer, direction) cells:")
    print()

    all_cells: list[tuple] = []
    for pmr in per_model_results:
        for r in pmr["rows"]:
            all_cells.append((
                r["model"], r["f_anchor"], r["layer"], "N->F",
                r["n2f"]["M4b_intended_arity_agree"],
                r["n2f"]["M4a_unary_mass"], r["n2f"]["M4c_herfindahl"],
                r["gate_n2f"], r["verdict"],
            ))
            all_cells.append((
                r["model"], r["f_anchor"], r["layer"], "F->N",
                r["f2n"]["M4b_intended_arity_agree"],
                r["f2n"]["M4a_unary_mass"], r["f2n"]["M4c_herfindahl"],
                r["gate_f2n"], r["verdict"],
            ))

    all_cells.sort(key=lambda x: -x[4])

    print(f"    {'model':<12} | {'anchor':<14} | {'L':<3} | dir  | M4b     | M4a     | M4c   | gate    | verdict")
    print(f"    {'-' * 90}")
    for cell in all_cells[:10]:
        print(f"    {cell[0]:<12} | {cell[1]:<14} | {cell[2]:>3} | "
              f"{cell[3]:<4} | {cell[4]:>5.1%}  | {cell[5]:>5.1%}  | "
              f"{cell[6]:>5.2f} | {cell[7]:>7.3f} | {cell[8]}")

    best_m4b = max(c[4] for c in all_cells)
    print()
    if best_m4b >= M4B_PASS_THRESHOLD:
        print(f"  *** M4b PASSES at best cell ({best_m4b:.1%} >= {M4B_PASS_THRESHOLD:.0%}). ***")
        print(f"  This RETRACTS the §3.7.8 negative-result headline in favour of")
        print(f"  a positive cross-notation arity-respecting transfer finding at")
        print(f"  the anchor / layer / direction above. Update paper_notes §3.7.8.")
    elif best_m4b >= 0.52:
        print(f"  M4b best = {best_m4b:.1%}, above the ~52% random-by-arity baseline but")
        print(f"  below the {M4B_PASS_THRESHOLD:.0%} PASS threshold. Suggestive of partial")
        print(f"  context-grounded arity transfer at this anchor; warrants stronger")
        print(f"  validation (more invented words, multiple unary canonicals, etc.).")
    else:
        print(f"  M4b best = {best_m4b:.1%}, at or below the ~52% random-by-arity baseline.")
        print(f"  The §3.7.8 negative-result headline SURVIVES the multi-anchor re-test:")
        print(f"  cross-notation arity-respecting transfer is not demonstrated at any")
        print(f"  tested anchor in either model. Substantially strengthens the")
        print(f"  negative result.")


# ==============================================================================
# Driver
# ==============================================================================
def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def free_model(model) -> None:
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_one_model(spec: ModelSpec, device: str) -> dict:
    print()
    print(f"########  {spec.short_name}  ({spec.model_id})  ########")
    t_total = time.time()

    # Up-front cache check (regenerate prompts deterministically; fast).
    neut_canon, _, neut_inv, _ = _generate_prompts(
        "NEUTRAL", make_neutral_stimuli, make_neutral_stimuli
    )
    func_canon, _, func_inv, _ = _generate_prompts(
        "FUNC-PFX", make_functional_canonical_stimuli, make_functional_invented_stimuli
    )
    neut_cache = _cache_load(
        _cache_path(spec.short_name, "NEUTRAL"),
        expected_canon_hash=prompts_checksum(neut_canon),
        expected_inv_hash=prompts_checksum(neut_inv),
        expected_anchors=ANCHORS_NEUTRAL,
    )
    func_cache = _cache_load(
        _cache_path(spec.short_name, "FUNC-PFX"),
        expected_canon_hash=prompts_checksum(func_canon),
        expected_inv_hash=prompts_checksum(func_inv),
        expected_anchors=ANCHORS_FUNC_PFX,
    )

    model = None
    tok = None
    if neut_cache is None or func_cache is None:
        print(f"  loading tokenizer + model ({spec.dtype}) on {device}...")
        tok = AutoTokenizer.from_pretrained(spec.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            spec.model_id, torch_dtype=spec.dtype, low_cpu_mem_usage=True
        ).to(device).eval()
        print(f"  model loaded, n_layers={model.config.num_hidden_layers}, "
              f"hidden={model.config.hidden_size}")
    else:
        print(f"  both conditions cached; skipping model load")

    if neut_cache is None:
        cond_neut = build_condition(
            spec, "NEUTRAL", ANCHORS_NEUTRAL, model, tok, device,
            make_neutral_stimuli, make_neutral_stimuli,
        )
    else:
        cond_neut = neut_cache

    if func_cache is None:
        cond_func = build_condition(
            spec, "FUNC-PFX", ANCHORS_FUNC_PFX, model, tok, device,
            make_functional_canonical_stimuli, make_functional_invented_stimuli,
        )
    else:
        cond_func = func_cache

    if model is not None:
        free_model(model)
    if tok is not None:
        del tok
    gc.collect()

    out = analyse_model(spec, cond_neut, cond_func)
    print(f"\n  -- {spec.short_name} total time: {time.time() - t_total:.1f}s --")
    return out


def main() -> None:
    log_path = _setup_logging()
    print(f"Script 21 - multi-anchor M1-M4 battery (post-call anchor re-test)")
    print(f"  STIMULUS_VERSION: {STIMULUS_VERSION}")
    print(f"  ANCHORS (NEUTRAL): {ANCHORS_NEUTRAL}")
    print(f"  ANCHORS (FUNC-PFX): {ANCHORS_FUNC_PFX}")
    print(f"  N_PER_CLASS: {N_PER_CLASS}")
    print(f"  GATE_THRESHOLD: {GATE_THRESHOLD}")
    print(f"  M4B_PASS_THRESHOLD: {M4B_PASS_THRESHOLD}")
    device = pick_device()
    print(f"  device: {device}")

    all_results: list[dict] = []
    for spec in MODEL_SPECS:
        r = run_one_model(spec, device)
        all_results.append(r)

    headline_summary(all_results)

    if log_path:
        print()
        print(f"[logging] full transcript written to: {log_path}")


if __name__ == "__main__":
    main()
