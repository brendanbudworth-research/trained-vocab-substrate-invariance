# Paper figures

Five figures referenced in `paper.md`. Each script in this folder produces
the matching `out/fig_NN_*.{pdf,png}` artefact.

| ID | Script | What it shows | Data source |
|---|---|---|---|
| Figure 1 | `fig_01_schematic.py` | Schematic of the substrate-invariance setup: NEUTRAL vs FUNC-PFX stimulus pairs, residual-stream extraction, M2-canonical (Fact 1) vs M4b (Fact 2) probe readouts, and the methodological battery. | (no data; illustration) |
| Figure 2 | `fig_02_m2c_heatmap.py` | Per-model heatmap of M2-canonical across the v6 80-cell sweep (16 anchor pairs × 5 focus layers). Visualises the `operator-after → operator-after` PASS cluster across all three model families. | `outputs/24_20260520_185537.log` (sweep tables, parsed live) |
| Figure 3 | `fig_03_canonical_breakdown.py` | Per-canonical breakdown of invented-word predictions at OLMo `N→F sente→close L 10` across the four scopes (v3 → v6). v3 looks arity-respecting; v5 and v6 collapse 100% to a single attractor — the lucky-default detector's vindication. | `outputs/cache/24_OLMo_2_7B_*_v6-expanded-canonical.npz` (live probe re-fit per scope) |
| Figure 4 | `fig_04_causal_patching.py` | Cross-cell synthesis of the script 25a causal patching experiment: 5 (source, target, layer) cells × {targeted PATCH, RANDOM_NORM control} × {ref_not, ref_and}. One CLEAN PASS, one WEAK PASS, three FAIL. | Table 5 of `paper.md` (canonical numbers; `outputs/25a_20260520_211030.log` and `25a_20260521_085745.log` of record) |
| Figure 5 | `fig_05_agreement.py` | Per-layer agreement curves from script 25b: `agree-all`, `agree-arity`, `arity-match`, with chance baselines and the pre-registered 60% "mechanism gap closed" threshold. L 0 floor is the headline visual. | `outputs/25b_20260520_213935.log` (per-model L 0 + focus-layer block) |

## Regeneration

```bash
cd experiments/figures
python run_all.py
```

Each script can also be run individually. Outputs go to `experiments/figures/out/`.

## Reproducibility notes

- Probe hyperparameters in `_shared.py:train_probe` match
  `experiments/24_v6_canonical_expansion.py:m2_metrics` exactly
  (sklearn `LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs")`).
- The v6 caches under `experiments/outputs/cache/` are the canonical source for
  fig_03's live probe fits. If those caches are pruned, the script will exit
  cleanly with a `caches missing` assertion rather than producing a fake figure.
- fig_04 and fig_05 hard-code numbers from log files of record (cited inline in
  the script docstrings). If the upstream experiments are re-run with different
  seeds, update the constants in `CELLS` / `PER_LAYER`.
- All figures use deterministic colourmaps and seeds (`SEED = 1337` in
  `_shared.py`); regeneration is bit-stable.
