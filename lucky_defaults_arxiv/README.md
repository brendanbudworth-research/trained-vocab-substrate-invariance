# Lucky Defaults: A Failure Mode in Probe-Based Generalisation Metrics

A short methods note describing a probe-based generalisation failure mode
("lucky-default" attractor collapse) and a per-item top-class concentration
diagnostic (\pwmin) that detects it.

The case study draws on the v6 four-scope sweep data from the companion
substrate-invariance paper:

  https://github.com/brendanbudworth-research/trained-vocab-substrate-invariance

## Layout

```
.
├── Makefile             # `make` builds paper.pdf
├── README.md            # this file
├── paper.tex            # manuscript source
├── references.bib       # BibTeX
├── figures/             # PDFs for \includegraphics
└── .gitignore           # TeX intermediates
```

## Build

Requires `pdflatex` and `bibtex` (MacTeX / TeX Live / BasicTeX).

```bash
make              # full build (pdflatex + bibtex + pdflatex x 2)
make quick        # single pdflatex pass while drafting (refs may be unset)
make clean        # remove intermediates, keep paper.pdf
make arxiv        # bundle .tex + .bib + figures into arxiv_submission.tar.gz
```

## Status

Current draft (May 2026): 12-page PDF including references.
The case study draws on the v6 four-scope sweep data from the companion
substrate-invariance paper; no new experiments are required to reproduce.

## Reproducing the case-study numbers

The manuscript contains four tables; the two with case-study numerics are:

- **Table 3** — the four development-sweep LUCKY-NEG cells (3 OLMo + 1 Gemma)
  showing $A_\pi$, $\mathrm{HHI}$, $\max_c p_c$, $\pwmin{}$, dominant
  prototypes, and routing pattern. Sourced from the script-22b log in the
  companion repository:

  ```
  trained-vocab-substrate-invariance/experiments/outputs/22b_20260520_083957.log
  ```

  Filter rows by `verdict = LUCKY-NEG` AND `M4b >= 0.65` AND `M4c < 0.70`
  AND `pwmin >= 0.95`; the four matching rows are the table entries.

- **Table 4** — the eight-cell $\pwmin{}$ trajectory across the four
  pre-registered scopes ($v3 \to v4 \to v5 \to v6$). Sourced from the v6
  sweep log:

  ```
  trained-vocab-substrate-invariance/experiments/outputs/24b_20260521_120258.log
  ```

  The relevant block runs from line 117 ("N->F opera->first L 4:") through
  line 165 ("N->F opera->close L16:"); each cell's
  `v3 -> v4 -> v5 -> v6` $\pwmin{}$ values can be read off directly.
