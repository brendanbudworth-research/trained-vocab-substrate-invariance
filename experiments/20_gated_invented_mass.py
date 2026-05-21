"""Script 20 - gated-invented-mass re-test at script-19b-identified pairings.

Closes the principal open empirical question at Phase 1 entry: when the
canonical-transfer gate PASSES, does the cross-condition invented transfer
actually deliver invented words into the unary region, or does the
gate-PASS-but-angle-wide failure mode (script 19b finding) generalise?

Targeted re-test of script-18 Diagnostic A at the specific gate-PASS
pairings that 19b identified as interesting:

  OLMo 2 7B:
    - (NEUTRAL, L10) -> (FUNC-PFX, L10) invented   [NEW; never tested]
    - (FUNC-PFX, L10) -> (NEUTRAL, L10) invented   [NEW]
    - (NEUTRAL, L10) -> (FUNC-PFX, L7) invented    [cross-layer, gate 0.800]
    - (FUNC-PFX, L10) -> (NEUTRAL, L7) invented    [cross-layer, gate 0.664]
    - (NEUTRAL, L7) -> (FUNC-PFX, L7) invented     [script-18 baseline = 0%]

  Gemma 2 9B:
    - (NEUTRAL, L2) -> (FUNC-PFX, L2) invented     [the 19b sweet spot]
    - (FUNC-PFX, L2) -> (NEUTRAL, L2) invented     [gate 0.884]
    - (NEUTRAL, L8) -> (FUNC-PFX, L8) invented     [gate-PASS-angle-wide test]
    - (FUNC-PFX, L8) -> (NEUTRAL, L8) invented     [gate 0.864]
    - (NEUTRAL, L4) -> (FUNC-PFX, L4) invented     [script-18 baseline = 99.6%]

Uses the 19b disk cache exclusively - no model load required. Runs in
seconds (probe fits dominate, ~20 pairings * <1s each).

For each pairing, reports:
  - The bidirectional gate accuracy (recomputed for self-consistency).
  - The directional cosine angle (centroid + probe).
  - The invented-unary mass: fraction of invented predictions in
    {not, necessarily}.
  - The full predicted-canonical breakdown (% per class).
  - Per-invented-word predicted canonical and unary/binary verdict.

Outputs tee'd to outputs/20_<ts>.log.

Three possible outcomes for the OLMo 2 L10 result:
  (i)  L10 invented unary mass ~0% in both directions: "notation-local"
       framing reinforced. Gate-PASS is necessary but doesn't even
       deliver any signal at the OLMo 2 best gate-passing layer.
  (ii) L10 invented unary mass non-trivial in one direction (e.g.,
       30-70%): OLMo 2 has weakly cross-notation-aligned arity at L10;
       framing softens to "L7-asymmetric / L10-partially-aligned".
  (iii) L10 invented unary mass >= 80% in both directions: OLMo 2 has
       cross-notation arity transfer too, just at a different depth
       than Gemma 2. Refines the cross-model claim substantially.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import os
import random
import sys
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression


# ==============================================================================
# Tee logging.
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
    log_path = os.path.join(log_dir, f"20_{ts}.log")
    log_f = open(log_path, "w", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_f)
    sys.stderr = _Tee(sys.__stderr__, log_f)
    print(f"[logging] tee'ing all output to {log_path}")
    print(f"[logging] (set NO_LOG=1 to disable)")
    return log_path


# ==============================================================================
# Shared constants - identical to scripts 17/18/19/19b.
# ==============================================================================
SEED = 17
N_PER_CLASS = 50
GATE_THRESHOLD = 0.65

# Must match the values in 19b for cache hits.
STIMULUS_VERSION = "v2-stable-seeds"
ANCHOR_MODE = "operator-after"

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


# Shared stimulus generation: load from script 19b via importlib so the
# templates / stimulus functions are not duplicated (and so any future
# stimulus change in 19b automatically invalidates 20 via the cache
# metadata check). Script name starts with a digit so we can't use a
# normal `import`; SourceFileLoader handles this.
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

# Sanity: our local constants must match 19b's, or the imported stimulus
# functions will silently produce different prompts than 19b cached.
assert _M19B.SEED == SEED, "SEED drift between 19b and 20"
assert _M19B.N_PER_CLASS == N_PER_CLASS, "N_PER_CLASS drift"
assert _M19B.STIMULUS_VERSION == STIMULUS_VERSION, "STIMULUS_VERSION drift"
assert _M19B.ANCHOR_MODE == ANCHOR_MODE, "ANCHOR_MODE drift"


# ==============================================================================
# Cache loader - identical interface to 19b's _cache_load.
# ==============================================================================
@dataclass
class ConditionActivations:
    canonical_X: list[np.ndarray]
    canonical_labels: np.ndarray
    invented_X: list[np.ndarray]
    invented_word_per_stim: list[str]


def _cache_path(model_short_name: str, condition_name: str) -> str:
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "outputs", "cache")
    slug = model_short_name.replace(" ", "_")
    return os.path.join(
        base,
        f"19b_{slug}_{condition_name}_npc{N_PER_CLASS}_{STIMULUS_VERSION}.npz",
    )


def _verify_meta(z, *, condition_name: str,
                 expected_canon_hash: str, expected_inv_hash: str,
                 path: str) -> None:
    """Hard-fail if cached metadata does not exactly match what script 20
    expects. Script 20 is downstream of 19b; it must not silently consume
    a cache that was built with different stimuli / anchor / version."""
    required = [
        "n_per_class", "meta_stimulus_version", "meta_anchor_mode",
        "meta_canon_prompts_hash", "meta_inv_prompts_hash",
    ]
    for k in required:
        if k not in z.files:
            raise RuntimeError(
                f"cache at {path} is missing key {k!r}; this is likely "
                f"a pre-v2 cache. Delete it and re-run script 19b to "
                f"regenerate with stable seeds."
            )
    npc = int(z["n_per_class"][0])
    if npc != N_PER_CLASS:
        raise RuntimeError(
            f"cache n_per_class mismatch: {npc} != {N_PER_CLASS}")
    sv = str(z["meta_stimulus_version"][0])
    if sv != STIMULUS_VERSION:
        raise RuntimeError(
            f"cache stimulus_version mismatch: {sv} != {STIMULUS_VERSION}")
    am = str(z["meta_anchor_mode"][0])
    if am != ANCHOR_MODE:
        raise RuntimeError(
            f"cache anchor_mode mismatch: {am} != {ANCHOR_MODE}")
    ch = str(z["meta_canon_prompts_hash"][0])
    if ch != expected_canon_hash:
        raise RuntimeError(
            f"cache canon_prompts_hash mismatch for {condition_name}: "
            f"{ch[:16]}... != {expected_canon_hash[:16]}... "
            f"(stimulus drift between 19b and 20)")
    ih = str(z["meta_inv_prompts_hash"][0])
    if ih != expected_inv_hash:
        raise RuntimeError(
            f"cache inv_prompts_hash mismatch for {condition_name}: "
            f"{ih[:16]}... != {expected_inv_hash[:16]}... "
            f"(stimulus drift between 19b and 20)")


def load_condition(
    model_short_name: str, condition_name: str,
    *, expected_canon_hash: str, expected_inv_hash: str,
) -> ConditionActivations:
    path = _cache_path(model_short_name, condition_name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required cache file missing: {path}\n"
            f"Run script 19b first to generate the cache."
        )
    z = np.load(path, allow_pickle=False)
    _verify_meta(z, condition_name=condition_name,
                 expected_canon_hash=expected_canon_hash,
                 expected_inv_hash=expected_inv_hash, path=path)
    canon_stack = z["canonical_X"].astype(np.float32)
    inv_stack = z["invented_X"].astype(np.float32)
    cond = ConditionActivations(
        canonical_X=[canon_stack[i] for i in range(canon_stack.shape[0])],
        canonical_labels=z["canonical_labels"],
        invented_X=[inv_stack[i] for i in range(inv_stack.shape[0])],
        invented_word_per_stim=list(z["invented_word_per_stim"]),
    )
    print(f"  [cache] loaded {os.path.basename(path)} "
          f"(n_layers={canon_stack.shape[0]}, dim={canon_stack.shape[2]})")
    print(f"  [cache]   meta verified: stimulus_version={STIMULUS_VERSION}, "
          f"anchor_mode={ANCHOR_MODE}")
    return cond


# ==============================================================================
# Direction + gate primitives (subset of 19b).
# ==============================================================================
def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return v
    return v / n


def cosine_angle_deg(u: np.ndarray, v: np.ndarray) -> tuple[float, float]:
    cos = float(np.dot(u, v))
    cos = max(-1.0, min(1.0, cos))
    deg = float(np.degrees(np.arccos(cos)))
    return cos, deg


def centroid_unary_direction(canon_X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    u = np.stack([canon_X[labels == op].mean(axis=0) for op in UNARY_CANONICALS])
    b = np.stack([canon_X[labels == op].mean(axis=0) for op in BINARY_CANONICALS])
    return unit(u.mean(axis=0) - b.mean(axis=0))


def raw_binary_probe_direction(canon_X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    y = np.array([1 if op in UNARY_CANONICALS else 0 for op in labels])
    clf = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs").fit(canon_X, y)
    return unit(clf.coef_[0])


def canonical_transfer_accuracy(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
) -> float:
    clf = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs")
    clf.fit(X_train, y_train)
    return float(clf.score(X_test, y_test))


def gate_verdict(acc: float) -> str:
    if acc >= GATE_THRESHOLD:
        return "PASS"
    if acc >= 0.30:
        return "AMBIG"
    return "FAIL"


# ==============================================================================
# The central measurement: invented-canonical predicted breakdown.
# ==============================================================================
def invented_breakdown(
    train_X: np.ndarray, train_y: np.ndarray, test_invented_X: np.ndarray,
    invented_words: list[str],
) -> dict:
    clf = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs")
    clf.fit(train_X, train_y)
    preds = clf.predict(test_invented_X)
    n_total = len(preds)
    n_unary = int(np.sum(np.isin(preds, UNARY_CANONICALS)))
    unary_mass = n_unary / n_total
    canon_counts = {c: int(np.sum(preds == c)) for c in CANONICALS}
    canon_pct = {c: canon_counts[c] / n_total for c in CANONICALS}

    per_word: dict[str, dict] = {}
    for w in INVENTED_WORDS:
        mask = np.array([iw == w for iw in invented_words])
        w_preds = preds[mask]
        w_n = len(w_preds)
        w_canon_counts = {c: int(np.sum(w_preds == c)) for c in CANONICALS}
        w_unary = sum(w_canon_counts[c] for c in UNARY_CANONICALS)
        w_top_canon = max(w_canon_counts, key=lambda c: w_canon_counts[c])
        per_word[w] = {
            "n": w_n,
            "top_pred": w_top_canon,
            "top_pred_pct": w_canon_counts[w_top_canon] / w_n if w_n > 0 else 0.0,
            "unary_pct": w_unary / w_n if w_n > 0 else 0.0,
            "intended_arity": CANONICAL_ARITY[W_TO_CANONICAL[w]],
        }

    return {
        "n_total": n_total,
        "unary_mass": unary_mass,
        "canon_pct": canon_pct,
        "per_word": per_word,
    }


# ==============================================================================
# Pairings to test.
# ==============================================================================
@dataclass
class Pairing:
    label: str
    src_cond: str
    src_L: int
    tgt_cond: str
    tgt_L: int
    note: str = ""


OLMO_PAIRINGS: list[Pairing] = [
    Pairing("baseline-1", "NEUTRAL", 7, "FUNC-PFX", 7,
            "script-18 baseline (was 0% unary)"),
    Pairing("baseline-2", "FUNC-PFX", 7, "NEUTRAL", 7,
            "script-18 reverse baseline (was 0% unary)"),
    Pairing("L10-same-N->F", "NEUTRAL", 10, "FUNC-PFX", 10,
            "NEW: 19b gate-PASS 0.800; never tested"),
    Pairing("L10-same-F->N", "FUNC-PFX", 10, "NEUTRAL", 10,
            "NEW: 19b gate-PASS 0.688; never tested"),
    Pairing("L10->L7-N->F", "NEUTRAL", 10, "FUNC-PFX", 7,
            "cross-layer 19b gate-PASS 0.800"),
    Pairing("L10->L7-F->N", "FUNC-PFX", 10, "NEUTRAL", 7,
            "cross-layer 19b gate-PASS 0.664"),
]

GEMMA_PAIRINGS: list[Pairing] = [
    Pairing("baseline-L4", "NEUTRAL", 4, "FUNC-PFX", 4,
            "script-18 / 19b baseline (was 99.6% unary)"),
    Pairing("L2-sweet-N->F", "NEUTRAL", 2, "FUNC-PFX", 2,
            "19b sweet spot; gate 1.000"),
    Pairing("L2-sweet-F->N", "FUNC-PFX", 2, "NEUTRAL", 2,
            "19b gate 0.884"),
    Pairing("L8-failmode-N->F", "NEUTRAL", 8, "FUNC-PFX", 8,
            "gate-PASS-angle-wide test; 19b gate 1.000"),
    Pairing("L8-failmode-F->N", "FUNC-PFX", 8, "NEUTRAL", 8,
            "gate-PASS-angle-wide test; 19b gate 0.864"),
]


def get_layer(cond: ConditionActivations, L: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return cond.canonical_X[L], cond.canonical_labels, cond.invented_X[L]


def analyse_pairings(
    model_name: str, neut: ConditionActivations, func: ConditionActivations,
    pairings: list[Pairing],
) -> list[dict]:
    print(f"\n{'=' * 92}")
    print(f"  {model_name} - GATED INVENTED-MASS RE-TEST")
    print(f"{'=' * 92}")
    print()
    print(f"  {'pairing':<22} | {'gate acc':>8} | "
          f"{'unary mass':>10} | {'centroid':>8} | {'probe':>6} | "
          f"{'verdict':<8} | note")
    print(f"  {'-' * 110}")

    cond_map = {"NEUTRAL": neut, "FUNC-PFX": func}
    results: list[dict] = []

    for p in pairings:
        src = cond_map[p.src_cond]
        tgt = cond_map[p.tgt_cond]
        Xs_canon, ys, _ = get_layer(src, p.src_L)
        Xt_canon, yt, Xt_inv = get_layer(tgt, p.tgt_L)
        inv_words = tgt.invented_word_per_stim

        gate_acc = canonical_transfer_accuracy(Xs_canon, ys, Xt_canon, yt)
        verdict = gate_verdict(gate_acc)

        d_s = centroid_unary_direction(Xs_canon, ys)
        d_t = centroid_unary_direction(Xt_canon, yt)
        _, deg_c = cosine_angle_deg(d_s, d_t)

        w_s = raw_binary_probe_direction(Xs_canon, ys)
        w_t = raw_binary_probe_direction(Xt_canon, yt)
        _, deg_p = cosine_angle_deg(w_s, w_t)

        bd = invented_breakdown(Xs_canon, ys, Xt_inv, inv_words)

        print(f"  {p.label:<22} | "
              f"{gate_acc:>8.3f} | "
              f"{bd['unary_mass']:>9.1%} | "
              f"{deg_c:>7.1f}° | {deg_p:>5.1f}° | "
              f"{verdict:<8} | {p.note}")

        results.append({
            "pairing": p.label,
            "src_cond": p.src_cond, "src_L": p.src_L,
            "tgt_cond": p.tgt_cond, "tgt_L": p.tgt_L,
            "gate_acc": gate_acc, "verdict": verdict,
            "centroid_deg": deg_c, "probe_deg": deg_p,
            "unary_mass": bd["unary_mass"],
            "canon_pct": bd["canon_pct"],
            "per_word": bd["per_word"],
            "note": p.note,
        })

    # Detail: canonical breakdown per pairing
    print()
    print(f"  Predicted-canonical breakdown (% of invented predictions per class):")
    print()
    print(f"  {'pairing':<22} | " + " | ".join(f"{c:>11}" for c in CANONICALS))
    print(f"  {'-' * 92}")
    for r in results:
        row = " | ".join(f"{r['canon_pct'][c]:>11.1%}" for c in CANONICALS)
        print(f"  {r['pairing']:<22} | {row}")

    # Detail: per-invented-word for each pairing
    print()
    print(f"  Per-invented-word predicted canonical (and unary share per word):")
    print()
    word_intended_arity = " ".join(
        f"{w}={'1' if CANONICAL_ARITY[W_TO_CANONICAL[w]] == 1 else '2'}"
        for w in INVENTED_WORDS
    )
    print(f"  (intended arity per word from W_TO_CANONICAL mapping: {word_intended_arity})")
    print()
    print(f"  {'pairing':<22} | " + " | ".join(f"{w:<14}" for w in INVENTED_WORDS))
    print(f"  {'-' * 110}")
    for r in results:
        cells = []
        for w in INVENTED_WORDS:
            pw = r["per_word"][w]
            cells.append(f"{pw['top_pred']:<10}{pw['unary_pct']:>3.0%}")
        print(f"  {r['pairing']:<22} | " + " | ".join(f"{c:<14}" for c in cells))

    return results


def cross_model_summary(olmo_results: list[dict], gemma_results: list[dict]) -> None:
    print()
    print("=" * 92)
    print("CROSS-MODEL SYNTHESIS")
    print("=" * 92)
    print()

    olmo_base_n2f = next(r for r in olmo_results if r["pairing"] == "baseline-1")
    olmo_base_f2n = next(r for r in olmo_results if r["pairing"] == "baseline-2")
    olmo_L10_n2f = next(r for r in olmo_results if r["pairing"] == "L10-same-N->F")
    olmo_L10_f2n = next(r for r in olmo_results if r["pairing"] == "L10-same-F->N")
    olmo_L10_L7_n2f = next(r for r in olmo_results if r["pairing"] == "L10->L7-N->F")
    olmo_L10_L7_f2n = next(r for r in olmo_results if r["pairing"] == "L10->L7-F->N")

    gemma_base = next(r for r in gemma_results if r["pairing"] == "baseline-L4")
    gemma_L2_n2f = next(r for r in gemma_results if r["pairing"] == "L2-sweet-N->F")
    gemma_L2_f2n = next(r for r in gemma_results if r["pairing"] == "L2-sweet-F->N")
    gemma_L8_n2f = next(r for r in gemma_results if r["pairing"] == "L8-failmode-N->F")
    gemma_L8_f2n = next(r for r in gemma_results if r["pairing"] == "L8-failmode-F->N")

    print("  Gemma 2 9B (script-18 reproducibility check):")
    print(f"    L4 (baseline, script-18 was 99.6%):       "
          f"unary={gemma_base['unary_mass']:.1%}, gate={gemma_base['gate_acc']:.3f} ({gemma_base['verdict']})")
    print()
    print("  Gemma 2 9B (NEW: L2 sweet spot vs L8 gate-PASS-angle-wide):")
    print(f"    L2 N->F (gate 1.000, centroid 58, probe 47): "
          f"unary={gemma_L2_n2f['unary_mass']:.1%}")
    print(f"    L2 F->N (gate 0.884, centroid 58, probe 47): "
          f"unary={gemma_L2_f2n['unary_mass']:.1%}")
    print(f"    L8 N->F (gate 1.000, centroid 69, probe 67): "
          f"unary={gemma_L8_n2f['unary_mass']:.1%}")
    print(f"    L8 F->N (gate 0.864, centroid 69, probe 67): "
          f"unary={gemma_L8_f2n['unary_mass']:.1%}")
    print()
    print("  OLMo 2 7B (script-18 reproducibility check):")
    print(f"    L7 N->F (baseline, script-18 was ~0%):    "
          f"unary={olmo_base_n2f['unary_mass']:.1%}, gate={olmo_base_n2f['gate_acc']:.3f} ({olmo_base_n2f['verdict']})")
    print(f"    L7 F->N (baseline, script-18 was ~0%):    "
          f"unary={olmo_base_f2n['unary_mass']:.1%}, gate={olmo_base_f2n['gate_acc']:.3f} ({olmo_base_f2n['verdict']})")
    print()
    print("  OLMo 2 7B (NEW: L10 gate-PASS layer never tested for invented mass):")
    print(f"    L10 same N->F (gate 0.800, centroid 72, probe 75): "
          f"unary={olmo_L10_n2f['unary_mass']:.1%}")
    print(f"    L10 same F->N (gate 0.688, centroid 72, probe 75): "
          f"unary={olmo_L10_f2n['unary_mass']:.1%}")
    print(f"    L10 -> L7 N->F (cross-layer):                       "
          f"unary={olmo_L10_L7_n2f['unary_mass']:.1%}")
    print(f"    L10 -> L7 F->N (cross-layer):                       "
          f"unary={olmo_L10_L7_f2n['unary_mass']:.1%}")
    print()

    # Decision table for the OLMo 2 L10 outcome
    olmo_L10_mean = float(np.mean([
        olmo_L10_n2f["unary_mass"], olmo_L10_f2n["unary_mass"]
    ]))
    if olmo_L10_mean < 0.20:
        outcome = "(i)  OLMo 2 L10 invented mass at floor"
        framing = ("'notation-local' framing reinforced. OLMo 2 has NO cross-notation arity "
                   "transfer in residual stream space even at the bidirectionally gate-PASS "
                   "layer L10. The gate-PASS at L10 represents a 5-class canonical-discrimination "
                   "axis that transfers, but the binary unary-vs-binary axis does not.")
    elif olmo_L10_mean < 0.65:
        outcome = "(ii) OLMo 2 L10 invented mass non-trivial"
        framing = ("'notation-local' framing weakens. OLMo 2 has weakly cross-notation-aligned "
                   "arity at L10. Framing should be: 'OLMo 2 has gate-PASS + partial invented "
                   "transfer at L10; Gemma 2 has gate-PASS + tight directional alignment + "
                   "near-complete invented transfer at L4. Cross-model contrast is gradient, "
                   "not binary.'")
    else:
        outcome = "(iii) OLMo 2 L10 invented mass high"
        framing = ("OLMo 2 has cross-notation arity transfer at L10, not L7. Substantially "
                   "refines the cross-model picture: BOTH models have cross-notation arity "
                   "transfer; they differ in WHICH layer carries it. The 'OLMo 2 is notation-"
                   "local' claim from §3.7.3 / §3.7.4 was wrong about depth: L10 is OLMo 2's "
                   "cross-notation transfer layer, not L7.")

    print(f"  OLMo 2 L10 outcome: {outcome}")
    print(f"  (mean L10 same-layer invented unary mass = {olmo_L10_mean:.1%})")
    print()
    print("  Interpretation guidance for paper_notes.md:")
    for line in framing.split(". "):
        if line.strip():
            print(f"    {line.strip()}{'.' if not line.strip().endswith('.') else ''}")


def main() -> None:
    log_path = _setup_logging()

    print(f"Script 20 - gated invented-mass re-test at 19b-identified pairings")
    print(f"  cache dir: experiments/outputs/cache/")
    print(f"  threshold: gate PASS >= {GATE_THRESHOLD} (~3x chance for 5-class)")
    print()

    # Compute expected prompt hashes for each (condition) so the cache
    # loader can verify metadata before we trust the activations.
    neut_canon, _, neut_inv, _ = _generate_prompts(
        "NEUTRAL", make_neutral_stimuli, make_neutral_stimuli
    )
    func_canon, _, func_inv, _ = _generate_prompts(
        "FUNC-PFX", make_functional_canonical_stimuli, make_functional_invented_stimuli
    )
    neut_canon_hash = prompts_checksum(neut_canon)
    neut_inv_hash = prompts_checksum(neut_inv)
    func_canon_hash = prompts_checksum(func_canon)
    func_inv_hash = prompts_checksum(func_inv)

    print("=" * 92)
    print("LOADING CACHES")
    print("=" * 92)
    print(f"  Expected NEUTRAL  canon hash: {neut_canon_hash[:16]}...,  "
          f"inv hash: {neut_inv_hash[:16]}...")
    print(f"  Expected FUNC-PFX canon hash: {func_canon_hash[:16]}...,  "
          f"inv hash: {func_inv_hash[:16]}...")
    print()
    olmo_neut = load_condition(
        "OLMo 2 7B", "NEUTRAL",
        expected_canon_hash=neut_canon_hash, expected_inv_hash=neut_inv_hash,
    )
    olmo_func = load_condition(
        "OLMo 2 7B", "FUNC-PFX",
        expected_canon_hash=func_canon_hash, expected_inv_hash=func_inv_hash,
    )
    gemma_neut = load_condition(
        "Gemma 2 9B", "NEUTRAL",
        expected_canon_hash=neut_canon_hash, expected_inv_hash=neut_inv_hash,
    )
    gemma_func = load_condition(
        "Gemma 2 9B", "FUNC-PFX",
        expected_canon_hash=func_canon_hash, expected_inv_hash=func_inv_hash,
    )

    olmo_results = analyse_pairings("OLMo 2 7B", olmo_neut, olmo_func, OLMO_PAIRINGS)
    gemma_results = analyse_pairings("Gemma 2 9B", gemma_neut, gemma_func, GEMMA_PAIRINGS)

    cross_model_summary(olmo_results, gemma_results)

    if log_path:
        print()
        print(f"[logging] full transcript written to: {log_path}")


if __name__ == "__main__":
    main()
