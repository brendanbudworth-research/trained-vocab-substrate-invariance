"""Script 24b -- frozen-criterion re-derivation.

Why this script exists
----------------------
The v6 pre-registration (``experiments/preregistration_v6.md``, §5) freezes
the PASS-arity cell verdict as the conjunction::

    M2-arity   >= 0.65
    M4b        >= 0.65
    M4c (max)  <= 0.85       <-- max fraction of invented mass on any single canonical
    M4a in       [0.20, 0.80]
    pwmin      <  0.95

The running code in ``24_v6_canonical_expansion.py`` drifted to a tighter
M4c definition and tighter M4a band::

    M2-arity   >= 0.65
    M4b        >= 0.65
    M4c (HHI)  <  0.70       <-- Herfindahl-Hirschman index Sigma p_c^2
    M4a in       [0.10, 0.90]
    pwmin      <  0.95

External reviewer (round 1) flagged the criterion-drift as a high-priority
provenance issue. This script replays the v6 four-scope sweep from the
existing cache files under BOTH criteria and prints:

  (a) per-model per-scope PASS-arity verdict count under each criterion
  (b) per-cell verdict comparison for every cell that is PASS-arity under
      EITHER criterion (so the reader can see if the two criteria ever
      disagree)
  (c) the headline retraction-chain (v3/v4/v5 PASS-arity candidates -> v6
      verdict) under both criteria

No model inference; cache-only. Probe fits are deterministic given the
seed in script 24. Runtime: ~2-4 min per model on CPU (no bootstrap CIs).

Usage
-----
    python experiments/24b_frozen_criterion_rederivation.py

The script auto-imports ``24_v6_canonical_expansion`` via ``importlib`` to
sidestep the leading-digit-in-module-name restriction, and reuses its
``run_cell``, ``enumerate_cells``, ``Scope``, ``SweepCell`` machinery
unchanged.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

# --- import script 24 by file path (leading digit in module name) -----------
# NB: Python 3.14's @dataclass requires the module to be registered in
# sys.modules BEFORE exec_module runs, because _process_class calls
# sys.modules.get(cls.__module__).__dict__ during decoration. Without
# the sys.modules entry that lookup returns None and the decorator
# crashes with AttributeError.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_S24_PATH = os.path.join(_THIS_DIR, "24_v6_canonical_expansion.py")
_spec = importlib.util.spec_from_file_location("s24", _S24_PATH)
assert _spec is not None and _spec.loader is not None
s24 = importlib.util.module_from_spec(_spec)
sys.modules["s24"] = s24
_spec.loader.exec_module(s24)


# ===========================================================================
# Pre-registered (FROZEN) criterion thresholds
# ===========================================================================
FROZEN_M2_ARITY_PASS = 0.65
FROZEN_M4B_PASS = 0.65
FROZEN_M4C_MAX_FRACTION = 0.85
FROZEN_M4A_LO = 0.20
FROZEN_M4A_HI = 0.80
FROZEN_PWMIN_LUCKY = 0.95


def frozen_verdict(cell: s24.SweepCell) -> str:
    """Pre-reg PASS-arity verdict (max-fraction M4c, narrower M4a band)."""
    if cell.lucky_default:
        return "LUCKY-NEG"
    max_fraction = max(cell.breakdown_pct.values()) if cell.breakdown_pct else 1.0
    passes = (
        cell.M2_arity >= FROZEN_M2_ARITY_PASS
        and cell.M4b >= FROZEN_M4B_PASS
        and max_fraction <= FROZEN_M4C_MAX_FRACTION
        and FROZEN_M4A_LO <= cell.M4a <= FROZEN_M4A_HI
    )
    if passes:
        return "PASS-arity"
    if cell.M2_arity >= FROZEN_M2_ARITY_PASS and cell.M4b >= FROZEN_M4B_PASS:
        return "ARITY-AXIS-ONLY"
    if cell.M2_arity >= FROZEN_M2_ARITY_PASS:
        return "M2A-ONLY"
    if cell.M2_cano >= s24.GATE_CANONICAL_PASS:
        return "M2C-ONLY"
    return "FAIL"


def m4c_max_fraction(cell: s24.SweepCell) -> float:
    if not cell.breakdown_pct:
        return float("nan")
    return max(cell.breakdown_pct.values())


# ===========================================================================
# Cache-only loader (skip hash validation; trust the existing cache)
# ===========================================================================
def load_carryover_cache_unchecked(
    model_short_name: str, condition_name: str,
) -> Optional[s24.ConditionMultiAnchor]:
    path = s24._cache_path(model_short_name, condition_name, "carryover")
    if not os.path.exists(path):
        print(f"    [cache miss] {os.path.basename(path)}")
        return None
    z = np.load(path, allow_pickle=False)
    cond = s24.ConditionMultiAnchor(
        canonical_X=z["canonical_X"].astype(np.float32),
        canonical_labels=z["canonical_labels"],
        invented_X=z["invented_X"].astype(np.float32),
        invented_word_per_stim=z["invented_word_per_stim"],
        anchor_names=list(z["anchor_names"]),
        canon_anchor_positions=z["canon_anchor_positions"].tolist(),
        inv_anchor_positions=z["inv_anchor_positions"].tolist(),
    )
    print(f"    [cache hit] {os.path.basename(path)} "
          f"canon={cond.canonical_X.shape}, inv={cond.invented_X.shape}")
    return cond


# ===========================================================================
# Per-model sweep
# ===========================================================================
@dataclass
class ScopeComparison:
    scope: str
    n_cells: int
    pass_running: int
    pass_frozen: int
    diff_cells: list[s24.SweepCell]


def run_model(spec: s24.ModelSpec) -> dict:
    print()
    print("#" * 100)
    print(f"# {spec.short_name}")
    print("#" * 100)

    neut = load_carryover_cache_unchecked(spec.short_name, "NEUTRAL")
    func = load_carryover_cache_unchecked(spec.short_name, "FUNC-PFX")
    if neut is None or func is None:
        print(f"  Skipping {spec.short_name} -- missing cache(s).")
        return {"spec": spec, "skipped": True}

    carryover = {"NEUTRAL": neut, "FUNC-PFX": func}

    cells_by_scope: dict[str, list[s24.SweepCell]] = {}
    comparisons: list[ScopeComparison] = []
    for scope in s24.ALL_SCOPES:
        t0 = time.time()
        cells = s24.enumerate_cells(scope.name, spec.focus_layers)
        for c in cells:
            s24.run_cell(c, carryover, scope)
        cells_by_scope[scope.name] = cells
        pass_running = sum(1 for c in cells if c.verdict == "PASS-arity")
        pass_frozen = sum(1 for c in cells if frozen_verdict(c) == "PASS-arity")
        diff = [c for c in cells if c.verdict != frozen_verdict(c)]
        comparisons.append(ScopeComparison(
            scope=scope.name, n_cells=len(cells),
            pass_running=pass_running, pass_frozen=pass_frozen,
            diff_cells=diff,
        ))
        print(f"  scope {scope.name}: {len(cells)} cells in {time.time() - t0:.1f}s "
              f"-- PASS-arity (running) = {pass_running}, "
              f"PASS-arity (frozen) = {pass_frozen}, "
              f"verdict diffs = {len(diff)}")
    return {
        "spec": spec, "skipped": False,
        "cells_by_scope": cells_by_scope, "comparisons": comparisons,
    }


# ===========================================================================
# Reporting
# ===========================================================================
def _cell_label(c: s24.SweepCell) -> str:
    return (
        f"{c.scope} {c.direction} "
        f"{c.train_anchor[:5]:>5}->{c.test_anchor[:5]:<5} L{c.layer:>2}"
    )


def print_per_model_comparison(result: dict) -> None:
    if result.get("skipped"):
        return
    spec = result["spec"]
    print()
    print("=" * 100)
    print(f"  {spec.short_name} -- per-scope PASS-arity counts under both criteria")
    print("=" * 100)
    print(f"  {'scope':<5} | {'cells':>5} | {'running':>9} | {'frozen':>8} | "
          f"{'verdict-diff':>12}")
    print("  " + "-" * 50)
    for comp in result["comparisons"]:
        print(f"  {comp.scope:<5} | {comp.n_cells:>5} | "
              f"{comp.pass_running:>9} | {comp.pass_frozen:>8} | "
              f"{len(comp.diff_cells):>12}")

    # Show every cell that is PASS-arity under EITHER criterion
    print()
    print(f"  Cells PASS-arity under EITHER criterion (any scope):")
    print(f"  {'cell':<32} | {'M2a':>5} | {'M4b':>5} | {'M4c-HHI':>7} | "
          f"{'max-c':>6} | {'M4a':>5} | {'pwmin':>5} | "
          f"{'running':<15} | {'frozen':<15}")
    print("  " + "-" * 130)
    any_pass = False
    for comp in result["comparisons"]:
        for c in result["cells_by_scope"][comp.scope]:
            r = c.verdict
            f = frozen_verdict(c)
            if r == "PASS-arity" or f == "PASS-arity":
                any_pass = True
                print(f"  {_cell_label(c):<32} | "
                      f"{c.M2_arity:>5.3f} | {c.M4b:>5.3f} | "
                      f"{c.M4c:>7.3f} | {m4c_max_fraction(c):>6.3f} | "
                      f"{c.M4a:>5.3f} | {c.per_word_min_top_pct:>5.3f} | "
                      f"{r:<15} | {f:<15}")
    if not any_pass:
        print("  (none)")


def print_headline_summary(results: list[dict]) -> None:
    print()
    print()
    print("=" * 120)
    print("  HEADLINE: PASS-arity verdict robustness under both criteria")
    print("=" * 120)
    print(f"  {'Model':<22} | {'v6 PASS (running)':<18} | {'v6 PASS (frozen)':<17} | "
          f"verdict")
    print("  " + "-" * 100)
    any_disagreement = False
    for r in results:
        if r.get("skipped"):
            continue
        spec = r["spec"]
        v6 = r["cells_by_scope"]["v6"]
        pr = sum(1 for c in v6 if c.verdict == "PASS-arity")
        pf = sum(1 for c in v6 if frozen_verdict(c) == "PASS-arity")
        if pr != pf:
            any_disagreement = True
            note = f"[DRIFT] {pr - pf:+d} cells differ"
        else:
            note = "consistent"
        print(f"  {spec.short_name:<22} | {pr:<18} | {pf:<17} | {note}")
    print()
    if any_disagreement:
        print("  >> Criterion-drift impact: at least one cell's PASS-arity verdict "
              "depends on which M4c definition is used. Headline claim "
              "'no PASS-arity cells at v6 under any model' must be qualified.")
    else:
        print("  >> No criterion-drift impact: the headline PASS-arity verdicts are "
              "robust to which M4c definition is used in all three model families.")
    print()


# ===========================================================================
# Cross-scope retraction chain under both criteria
# ===========================================================================
def cross_scope_retraction(results: list[dict]) -> None:
    print()
    print("=" * 120)
    print("  CROSS-SCOPE RETRACTION CHAIN (per-criterion)")
    print("=" * 120)
    print(f"  For each cell that is PASS-arity at v3 under EITHER criterion, "
          f"show its v3-v4-v5-v6 verdict trajectory under both.")
    print()
    for r in results:
        if r.get("skipped"):
            continue
        spec = r["spec"]
        scope_lookup = {s.name: s for s in s24.ALL_SCOPES}
        cells_by_key: dict[tuple, dict[str, s24.SweepCell]] = {}
        for scope_name, cells in r["cells_by_scope"].items():
            for c in cells:
                key = (c.direction, c.train_anchor, c.test_anchor, c.layer)
                cells_by_key.setdefault(key, {})[scope_name] = c

        v3_pass_keys = []
        for key, by_scope in cells_by_key.items():
            v3 = by_scope.get("v3")
            if v3 is None:
                continue
            if v3.verdict == "PASS-arity" or frozen_verdict(v3) == "PASS-arity":
                v3_pass_keys.append(key)

        if not v3_pass_keys:
            print(f"  {spec.short_name}: no v3 PASS-arity cells under either criterion.")
            continue

        print(f"  {spec.short_name}:")
        for key in v3_pass_keys:
            dirn, tr_a, te_a, L = key
            label = f"{dirn} {tr_a[:5]:>5}->{te_a[:5]:<5} L{L:>2}"
            print(f"    {label}:")
            for scope_name in ("v3", "v4", "v5", "v6"):
                c = cells_by_key[key].get(scope_name)
                if c is None:
                    print(f"      {scope_name}: (missing)")
                    continue
                print(f"      {scope_name}: "
                      f"running={c.verdict:<15} frozen={frozen_verdict(c):<15} "
                      f"M2a={c.M2_arity:.3f} M4b={c.M4b:.3f} "
                      f"HHI={c.M4c:.2f} max-c={m4c_max_fraction(c):.2f} "
                      f"M4a={c.M4a:.2f} pwmin={c.per_word_min_top_pct:.2f}")
        print()


# ===========================================================================
# Main
# ===========================================================================
def main() -> None:
    out_dir = os.path.join(_THIS_DIR, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(out_dir, f"24b_{ts}.log")
    print(f"[24b] frozen-criterion re-derivation -- logging to {log_path}")

    # Tee stdout to log file
    import sys
    log_f = open(log_path, "w", encoding="utf-8", buffering=1)

    class Tee:
        def __init__(self, *streams): self.streams = streams
        def write(self, s):
            for st in self.streams: st.write(s)
        def flush(self):
            for st in self.streams: st.flush()
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_f)

    try:
        print(f"[24b] start = {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        print(f"Running-code criterion (script 24):")
        print(f"  M2-arity >= 0.65, M4b >= 0.65, M4c (HHI) < 0.70, "
              f"0.10 <= M4a <= 0.90, pwmin < 0.95")
        print(f"Frozen pre-registered criterion (preregistration_v6.md §5):")
        print(f"  M2-arity >= 0.65, M4b >= 0.65, M4c (max fraction) <= 0.85, "
              f"0.20 <= M4a <= 0.80, pwmin < 0.95")
        print()

        results = [run_model(spec) for spec in s24.MODEL_SPECS]
        for r in results:
            print_per_model_comparison(r)

        print_headline_summary(results)
        cross_scope_retraction(results)

        print()
        print(f"[24b] done = {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[24b] log saved to {log_path}")
    finally:
        sys.stdout = original_stdout
        log_f.close()


if __name__ == "__main__":
    main()
