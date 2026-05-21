"""Script 25d -- Delta_specific bootstrap for the Cell-2 WEAK PASS verdict.

Why this script exists
----------------------
Paper.md Table 5 verdict for Cell 2 (Gemma 2 9B ``opera->opera L 4``, the
principal cross-family Fact-1 cell) is WEAK PASS under the 25a behavioural
KL adjudication: targeted Delta KL exceeds RANDOM_NORM by approximately
1.94x on ``not`` and 1.29x on ``and``. The 1.29x on ``and`` sits at the
soft boundary of the 1.3-2x heuristic band stated in the verdict criterion.

Paper.md Section 6 ("Causal evidence is partial") explicitly flags this as
the cleanest follow-up:

    "The Cell-2 WEAK PASS verdict also sits at the soft boundary of the
     1.3-2x heuristic (1.94x on `not`, 1.29x on `and`); a per-word
     bootstrap on Delta_specific = Delta KL_targeted - Delta KL_random
     across the 16 invented words is the cleaner statistic and would
     either firm up the WEAK PASS verdict or move it toward FAIL --
     single short follow-up using existing 25a outputs, no extraction
     required."

This script runs that follow-up. The cleaner statistic is

    Delta_specific(word) = Delta KL_targeted(word) - Delta KL_random_mean

per invented word, where Delta KL_targeted is the per-word value already
logged by script 25a's results table, and Delta KL_random_mean is the
aggregate (16-word mean) RANDOM_NORM Delta KL reported in the same log's
aggregate summary block. Bootstrap (B = 500) the mean Delta_specific
across the 16 invented words; the WEAK PASS verdict is robust if both the
``not`` and ``and`` axes' 95% bootstrap CIs lie strictly above zero, and
moves toward FAIL if either CI includes zero.

What this script does NOT do
----------------------------
The most rigorous available statistic would be a per-(word, stim) bootstrap
on Delta_specific where the RANDOM_NORM baseline is computed per-word
rather than as an aggregate offset. Script 25a accumulates per-word
RANDOM_NORM Delta KL values internally (lines 836-837 of 25a_causal_patching.py)
but only summarises them as mean/median/positive-count in the final block,
not as a per-word table. Recovering the per-word RANDOM_NORM data requires
re-running the patching at Cell 2 with extended logging (~6 min on MPS); we
defer this to a follow-up. The aggregate-offset approach used here treats
the RANDOM_NORM baseline as a fixed reference, which is slightly weaker
than a per-word baseline but is the strongest claim cache-only can support.

Inputs
------
* ``outputs/25a_20260521_085745.log`` -- the reviewer-round-1 follow-up
  run containing Cell 2 (Gemma 2 9B ``extra: opera->opera L 4``).

Outputs
-------
* ``outputs/25d_<ts>.log`` -- per-axis bootstrap distributions, CIs, and
  verdict update.

Usage
-----
    python experiments/25d_delta_specific_bootstrap.py

Cache-only; no model loads; runtime < 5 s on CPU.
"""
from __future__ import annotations

import os
import re
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(THIS_DIR, "outputs")

# The Cell-2 (Gemma opera->opera L 4) results live in the reviewer-round-1
# follow-up log. The other 25a log is the original 3-cell run.
LOG_CELL_2 = os.path.join(OUT_DIR, "25a_20260521_085745.log")

# Pre-spec adjudication thresholds (declared here; mirrors the paper.md
# Table 5 verdict criterion + §6 follow-up text).
B_BOOTSTRAP = 500
RNG_SEED = 20260522
WEAK_PASS_BAND_LOW = 1.3   # ratio of mean delta_targeted to mean delta_random
WEAK_PASS_BAND_HIGH = 2.0
CLEAN_PASS_FLOOR = 2.0     # mean ratio above which we'd call CLEAN PASS


@dataclass
class CellData:
    """Per-cell, per-axis delta-KL data parsed from a 25a log."""
    model: str
    cell_label: str
    per_word_delta_not: List[float]      # per-word delta KL under PATCH_not
    per_word_delta_and: List[float]      # per-word delta KL under PATCH_and
    random_mean_not: Optional[float]     # aggregate RANDOM_NORM mean
    random_mean_and: Optional[float]
    targeted_mean_not: Optional[float]
    targeted_mean_and: Optional[float]


def _parse_per_word_block(lines: List[str], start_idx: int) -> tuple[List[float], List[float], int]:
    """Parse the per-word `word arity | ... Δ(not) | ... Δ(and)` table.

    The block looks like::

        word     arity | KL(BASE||ref_not)  KL(PATCH_not||ref_not) Δ(not)     | KL(BASE||ref_and)  KL(PATCH_and||ref_and) Δ(and)
        ------...
          bliq     B     |          0.330                0.301        +0.029   |          0.318                0.286        +0.032
          ... (16 rows) ...
          nilph    U     |          0.231                0.179        +0.052   |          0.254                0.205        +0.049

    Returns (delta_not_list, delta_and_list, next_line_idx).
    """
    delta_not: List[float] = []
    delta_and: List[float] = []
    # Find the dashed separator line directly after the header
    sep_idx = start_idx
    while sep_idx < len(lines) and not lines[sep_idx].lstrip().startswith("---"):
        sep_idx += 1
    # Data rows follow the separator until a blank line
    i = sep_idx + 1
    while i < len(lines):
        line = lines[i].rstrip()
        if not line.strip():
            break
        # Row format: `  bliq     B     |  ...  +0.029   | ...  +0.032`
        # Match the two signed floats around the pipe separators.
        # Robust: find all signed floats and take positions 2 & 5 (the two
        # rightmost values in the | KL_base | KL_patch | Δ | KL_base | KL_patch | Δ | layout).
        signed = re.findall(r"[+-]\d+\.\d+", line)
        if len(signed) < 2:
            i += 1
            continue
        # Layout: KL_BASE_not, KL_PATCH_not, Δ_not, KL_BASE_and, KL_PATCH_and, Δ_and
        # KL values are unsigned so re.findall above only matches signed Δs.
        # We need exactly two signed floats per row (Δ_not, Δ_and).
        if len(signed) == 2:
            delta_not.append(float(signed[0]))
            delta_and.append(float(signed[1]))
        i += 1
    return delta_not, delta_and, i


def _parse_aggregate_means(lines: List[str], start_idx: int) -> dict:
    """Parse the `--- (D) Aggregate ΔKL summary ---` block following start_idx.

    Lines look like::
        Δ PATCH_not               mean = +0.033   median = +0.035   13/16 positive   ...
        Δ PATCH_and               mean = +0.027   median = +0.030   12/16 positive   ...
        Δ RANDOM_NORM_not         mean = +0.017   median = +0.008   11/16 positive   ...
        Δ RANDOM_NORM_and         mean = +0.021   median = +0.017   12/16 positive   ...

    Returns dict keyed by condition label -> {"mean": float, "median": float}.
    """
    out: dict = {}
    for i in range(start_idx, min(start_idx + 25, len(lines))):
        line = lines[i]
        m = re.search(r"Δ\s+(\S+)\s+mean\s*=\s*([+-]?\d+\.\d+)\s+median\s*=\s*([+-]?\d+\.\d+)", line)
        if m:
            cond = m.group(1)
            out[cond] = {"mean": float(m.group(2)), "median": float(m.group(3))}
    return out


def parse_cell_2_from_log(log_path: str) -> CellData:
    """Extract Cell-2 (Gemma opera->opera L 4) per-word + aggregate data."""
    with open(log_path, "r", encoding="utf-8") as f:
        text = f.read()
    lines = text.split("\n")

    # Find the Cell-2 RESULTS block. The Gemma extras run hits Cell 2 first
    # (CELL_FILTER=extra runs both extras; opera->opera is the second Gemma
    # extra). We anchor on the unique RESULTS header.
    results_idx = None
    for i, line in enumerate(lines):
        if "RESULTS:" in line and "opera->opera" in line and "L 4" in line:
            results_idx = i
            break
    if results_idx is None:
        raise RuntimeError(f"Cell-2 RESULTS block not found in {log_path}")

    # Section (C) header lives between RESULTS and the per-word table.
    c_header_idx = None
    for i in range(results_idx, min(results_idx + 50, len(lines))):
        if "(C) Behavioural shift" in lines[i]:
            c_header_idx = i
            break
    if c_header_idx is None:
        raise RuntimeError(f"Section (C) header not found after Cell-2 RESULTS at line {results_idx}")

    # Find the `word     arity |` table header after (C).
    table_header_idx = None
    for i in range(c_header_idx, min(c_header_idx + 30, len(lines))):
        if re.match(r"\s*word\s+arity\s*\|", lines[i]):
            table_header_idx = i
            break
    if table_header_idx is None:
        raise RuntimeError(f"Per-word table header not found after Section (C) at line {c_header_idx}")

    # Parse the per-word block.
    delta_not, delta_and, after_table_idx = _parse_per_word_block(lines, table_header_idx)
    if len(delta_not) != 16 or len(delta_and) != 16:
        raise RuntimeError(
            f"Expected 16 invented words; got |Δ(not)|={len(delta_not)}, "
            f"|Δ(and)|={len(delta_and)}"
        )

    # Find Section (D) aggregate summary.
    d_header_idx = None
    for i in range(after_table_idx, min(after_table_idx + 20, len(lines))):
        if "(D) Aggregate" in lines[i]:
            d_header_idx = i
            break
    if d_header_idx is None:
        raise RuntimeError(f"Section (D) header not found after Section (C) per-word table")
    aggs = _parse_aggregate_means(lines, d_header_idx)

    return CellData(
        model="Gemma 2 9B",
        cell_label="extra: opera->opera L 4 (principal Fact-1 anchor causal test)",
        per_word_delta_not=delta_not,
        per_word_delta_and=delta_and,
        random_mean_not=aggs.get("RANDOM_NORM_not", {}).get("mean"),
        random_mean_and=aggs.get("RANDOM_NORM_and", {}).get("mean"),
        targeted_mean_not=aggs.get("PATCH_not", {}).get("mean"),
        targeted_mean_and=aggs.get("PATCH_and", {}).get("mean"),
    )


def bootstrap_mean_ci(
    per_word: List[float],
    n_bootstrap: int = B_BOOTSTRAP,
    seed: int = RNG_SEED,
    ci: float = 0.95,
) -> tuple[float, float, float, np.ndarray]:
    """Bootstrap the mean of a per-word vector with replacement.

    Returns ``(point_mean, ci_lo, ci_hi, bootstrap_distribution)``.
    """
    rng = np.random.default_rng(seed)
    arr = np.asarray(per_word, dtype=np.float64)
    n = arr.size
    boots = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boots[b] = arr[idx].mean()
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(boots, alpha))
    hi = float(np.quantile(boots, 1.0 - alpha))
    return float(arr.mean()), lo, hi, boots


def verdict_for_axis(
    targeted_per_word: List[float],
    random_mean: float,
    label: str,
) -> dict:
    """Compute Δ_specific bootstrap CI and verdict for a single axis."""
    delta_specific = [v - random_mean for v in targeted_per_word]
    mean, lo, hi, _ = bootstrap_mean_ci(delta_specific)

    # Ratio diagnostics (matches the paper.md WEAK PASS band language).
    targeted_mean = float(np.mean(targeted_per_word))
    if random_mean > 0:
        ratio = targeted_mean / random_mean
    else:
        ratio = float("nan")

    # Verdict logic on Δ_specific 95% CI:
    #   * CI lower bound > 0  -> firms up WEAK PASS (or CLEAN PASS if mean ratio >= 2)
    #   * CI includes 0       -> verdict moves toward FAIL on this axis
    if lo > 0:
        if not np.isnan(ratio) and ratio >= CLEAN_PASS_FLOOR:
            axis_verdict = "FIRMS UP -> CLEAN PASS on this axis"
        else:
            axis_verdict = "FIRMS UP -> WEAK PASS on this axis"
    else:
        axis_verdict = "FAILS -> CI on Δ_specific includes 0 on this axis"

    return {
        "label": label,
        "targeted_mean": targeted_mean,
        "random_mean": random_mean,
        "ratio": ratio,
        "delta_specific_mean": mean,
        "delta_specific_ci_lo": lo,
        "delta_specific_ci_hi": hi,
        "verdict": axis_verdict,
        "per_word_delta_specific": delta_specific,
    }


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(OUT_DIR, f"25d_{ts}.log")
    print(f"[25d] Δ_specific bootstrap for Cell-2 WEAK PASS verdict -- log: {log_path}")

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
        print(f"[25d] start = {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[25d] source log = {LOG_CELL_2}")
        print(f"[25d] B_BOOTSTRAP = {B_BOOTSTRAP}, RNG_SEED = {RNG_SEED}")
        print()

        if not os.path.exists(LOG_CELL_2):
            print(f"[25d] FATAL: source log not found at {LOG_CELL_2}")
            return

        cell = parse_cell_2_from_log(LOG_CELL_2)

        print("=" * 100)
        print(f"  PARSED Cell-2 data ({cell.model})")
        print(f"  cell label: {cell.cell_label}")
        print("=" * 100)
        print(f"  16 invented words, per-axis ΔKL recovered from log")
        print(f"  Aggregate means from log's (D) block:")
        print(f"    targeted PATCH_not    mean = {cell.targeted_mean_not:+.4f}")
        print(f"    targeted PATCH_and    mean = {cell.targeted_mean_and:+.4f}")
        print(f"    RANDOM_NORM_not       mean = {cell.random_mean_not:+.4f}")
        print(f"    RANDOM_NORM_and       mean = {cell.random_mean_and:+.4f}")
        print()
        print(f"  Per-word ΔKL(not): {[f'{v:+.3f}' for v in cell.per_word_delta_not]}")
        print(f"  Per-word ΔKL(and): {[f'{v:+.3f}' for v in cell.per_word_delta_and]}")
        print()

        if cell.random_mean_not is None or cell.random_mean_and is None:
            print("[25d] FATAL: RANDOM_NORM aggregate means missing from log")
            return

        print("=" * 100)
        print(f"  BOOTSTRAP Δ_specific = ΔKL_targeted - ΔKL_random_mean per axis")
        print(f"  B = {B_BOOTSTRAP} resamples with replacement over the 16 invented words")
        print("=" * 100)
        print()

        not_verdict = verdict_for_axis(
            cell.per_word_delta_not, cell.random_mean_not, "PATCH_not vs RANDOM_NORM_not")
        and_verdict = verdict_for_axis(
            cell.per_word_delta_and, cell.random_mean_and, "PATCH_and vs RANDOM_NORM_and")

        for v in (not_verdict, and_verdict):
            print(f"  Axis: {v['label']}")
            print(f"    targeted ΔKL mean        = {v['targeted_mean']:+.4f}")
            print(f"    RANDOM_NORM ΔKL mean     = {v['random_mean']:+.4f}")
            print(f"    targeted / random ratio  = {v['ratio']:.3f}x")
            print(f"    Δ_specific mean          = {v['delta_specific_mean']:+.4f}")
            print(f"    Δ_specific 95% CI        = [{v['delta_specific_ci_lo']:+.4f}, "
                  f"{v['delta_specific_ci_hi']:+.4f}]")
            print(f"    axis verdict             = {v['verdict']}")
            print()

        # Joint verdict
        not_pass = not_verdict["delta_specific_ci_lo"] > 0
        and_pass = and_verdict["delta_specific_ci_lo"] > 0
        print("=" * 100)
        print("  JOINT VERDICT")
        print("=" * 100)
        print(f"  not-axis Δ_specific CI lower bound > 0?  {not_pass}")
        print(f"  and-axis Δ_specific CI lower bound > 0?  {and_pass}")
        if not_pass and and_pass:
            print(f"  >> Cell-2 WEAK PASS verdict FIRMS UP: both axes' Δ_specific 95% CI")
            print(f"     strictly above zero. The 1.29x ratio on `and` is no longer at the")
            print(f"     boundary -- the per-word distribution of Δ_specific(and) is")
            print(f"     significantly positive even though its mean is close to the")
            print(f"     RANDOM_NORM baseline. The paper.md Table 5 / §4.5 Cell 2 / §6 text")
            print(f"     should be updated to cite this Δ_specific CI as the cleaner statistic.")
        elif not_pass and not and_pass:
            print(f"  >> Cell-2 verdict downgraded on the `and` axis: Δ_specific(and) 95% CI")
            print(f"     includes zero. The 1.29x ratio on `and` is not robust under per-word")
            print(f"     bootstrap. Cell 2 remains WEAK PASS only on the `not` axis; the")
            print(f"     overall verdict should be downgraded to AMBIG or moved to FAIL on")
            print(f"     `and`, depending on the editorial choice for joint adjudication.")
        elif and_pass and not not_pass:
            print(f"  >> Unexpected result: `not` axis CI includes zero despite higher mean.")
            print(f"     Inspect the per-word distribution.")
        else:
            print(f"  >> Cell-2 WEAK PASS verdict FAILS under per-word bootstrap: both axes'")
            print(f"     Δ_specific 95% CI includes zero. Cell 2 should be downgraded to")
            print(f"     FAIL in paper.md Table 5 / §4.5 / §6.")
        print()

        # Per-word breakdown of which words drive the verdict
        print("=" * 100)
        print("  PER-WORD Δ_specific BREAKDOWN")
        print("=" * 100)
        print(f"  word | Δ_specific(not) | Δ_specific(and)")
        print(f"  ---- | --------------- | ---------------")
        WORDS = [
            ("bliq", "B"), ("dren", "B"), ("molex", "B"), ("krev", "B"),
            ("sond", "B"), ("glin", "B"), ("twiv", "B"), ("fump", "B"),
            ("vusp", "U"), ("perph", "U"), ("kelm", "U"), ("zorf", "U"),
            ("gleph", "U"), ("drelth", "U"), ("vrith", "U"), ("nilph", "U"),
        ]
        n_pos_not = 0
        n_pos_and = 0
        for (w, a), d_not_s, d_and_s in zip(
            WORDS, not_verdict["per_word_delta_specific"], and_verdict["per_word_delta_specific"]
        ):
            if d_not_s > 0: n_pos_not += 1
            if d_and_s > 0: n_pos_and += 1
            print(f"  {w:<6} {a}  | {d_not_s:+.4f}        | {d_and_s:+.4f}")
        print()
        print(f"  Per-word Δ_specific > 0:  {n_pos_not}/16 on `not`, {n_pos_and}/16 on `and`")
        print()
        print(f"[25d] done = {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[25d] log saved to {log_path}")
    finally:
        sys.stdout = original_stdout
        log_f.close()


if __name__ == "__main__":
    main()
