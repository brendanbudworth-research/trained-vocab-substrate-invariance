# `paper_arxiv/` — arXiv v1 LaTeX source

Self-contained LaTeX build of `paper.md` for arXiv submission. Everything
needed to rebuild the PDF lives in this directory.

## Layout

```
paper_arxiv/
├── paper.tex         # full manuscript (sections, tables, figures, appendix)
├── references.bib    # BibTeX entries (mirror of repo-root references.bib)
├── Makefile          # build targets (see below)
├── README.md         # this file
└── figures/
    ├── fig_01_schematic.pdf
    ├── fig_02_m2c_heatmap.pdf
    ├── fig_03_canonical_breakdown.pdf
    ├── fig_04_causal_patching.pdf
    └── fig_05_agreement.pdf
```

The figure PDFs are byte-identical copies of
`experiments/figures/out/*.pdf` at the arXiv-v1 commit; regenerate them
upstream and re-copy if their source scripts change.

## Build

You need a TeX install with `pdflatex` and `bibtex`. On macOS the
fastest path is BasicTeX (≈ 100 MB) plus the packages used in
`paper.tex`:

```bash
brew install --cask basictex
eval "$(/usr/libexec/path_helper)"   # add /Library/TeX/texbin to PATH
sudo tlmgr update --self
sudo tlmgr install microtype lm xcolor enumitem caption \
                    natbib booktabs array longtable hyperref \
                    geometry amsmath amssymb
```

(MacTeX is the full ≈ 4 GB distribution if you'd rather not micromanage
packages.)

Build targets:

```bash
make            # full build: pdflatex + bibtex + pdflatex × 2  -> paper.pdf
make quick      # single pdflatex pass (fast, refs may be broken)
make clean      # remove TeX intermediates (.aux/.log/.bbl/etc.); keep PDF
make distclean  # also remove paper.pdf and arxiv_submission.tar.gz
make arxiv      # bundle paper.tex + .bib + .bbl + figures into a tarball
```

The first `make` typically takes ~10–20 s on a modern machine.

## arXiv submission

arXiv accepts a `.tar.gz` containing the LaTeX source; they re-run
`pdflatex` server-side. Use:

```bash
make arxiv
```

That produces `arxiv_submission.tar.gz` with `paper.tex`,
`references.bib`, the pre-built `paper.bbl` (so arXiv doesn't need to
run `bibtex`), and all five figure PDFs. Upload at
<https://arxiv.org/submit>.

Recommended submission settings:

- **Primary category**: `cs.CL` (Computation and Language)
- **Cross-listings**: `cs.LG` (Machine Learning), optionally `cs.AI`
- **License**: CC BY 4.0 (per repository-root discussion;
  permissive-first is the right default for this paper)
- **Comments**: e.g.\ "v1; 17 pages incl. appendix, 5 figures, 6 tables"

## Source provenance

`paper.tex` is hand-converted from `paper.md` at repository commit
`e6afe5e` + arXiv-v1 polish commit. Conventions used in the conversion:

- Inline backtick code (`\`and\``) → `\texttt{and}`.
- Bold (`**...**`) → `\textbf{...}`; italic (`*...*`) → `\emph{...}`.
- Markdown pipe tables → `\begin{tabular}` + `booktabs` rules.
- Math: `Δ`/`→`/`∈`/`≤`/`≥`/`×`/`²` rendered via proper math commands
  (`\Delta`, `\to`, `\in`, `\leq`, …) inside `$...$`.
- Custom shorthand commands in the preamble:
  - `\dkl` → `ΔKL`
  - `\dspec` → `Δ_specific`
  - `\NtoF` / `\FtoN` → `N→F` / `F→N`
  - `\Mc` / `\Ma` → `M2-canonical` / `M2-arity` (small caps)
  - `\Mbfb` / `\Mbfc` → `M4b` / `M4c`
  - `\pwmin` → per-word minimum top-canonical concentration
- Prose-style citations (`Author, Year`) → `\citet{key}` / `\citep{key}`
  against `references.bib`.
- Section / subsection labels use `\label{sec:...}` and `\label{tab:...}`
  / `\label{fig:...}` / `\label{app:...}`; cross-references via
  `\S\ref{...}`, `Table~\ref{...}`, `Figure~\ref{...}`.

If you re-edit `paper.md` and want the changes to propagate here, the
delta is mechanical (find/replace + hand-edit any new tables/figures).
The source-of-truth manuscript remains `paper.md`; this directory is
the LaTeX rendering of it for arXiv.
