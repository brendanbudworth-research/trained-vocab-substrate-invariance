"""Regenerate all paper figures (fig_01 through fig_05).

Usage:
    cd experiments/figures
    python run_all.py

Per-figure compute:
    fig_01 (schematic):           ~2 s (matplotlib only)
    fig_02 (M2c heatmap):         ~2 s (log parse only)
    fig_03 (canonical breakdown): ~25 s (loads OLMo v6 cache, 4 probe fits)
    fig_04 (causal patching):     ~2 s (hard-coded Table 5 numbers)
    fig_05 (per-layer agreement): ~2 s (hard-coded log-extracted numbers)
"""

from __future__ import annotations

import importlib
import time


SCRIPTS = [
    "fig_01_schematic",
    "fig_02_m2c_heatmap",
    "fig_03_canonical_breakdown",
    "fig_04_causal_patching",
    "fig_05_agreement",
]


def main():
    for name in SCRIPTS:
        print(f"==> {name}")
        t0 = time.time()
        mod = importlib.import_module(name)
        mod.main()
        print(f"    [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
