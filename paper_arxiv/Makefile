# Build the arXiv-v1 PDF.
#   `make`         -> full build (pdflatex + bibtex + pdflatex x 2)
#   `make quick`   -> single pdflatex pass (fast, refs may be broken)
#   `make clean`   -> remove TeX intermediates (keeps paper.pdf)
#   `make distclean` -> remove everything including paper.pdf
#   `make arxiv`   -> produce arxiv_submission.tar.gz with sources + figures
#                    (everything arXiv needs to rebuild from source).
#
# Requires a TeX install with pdflatex + bibtex (e.g. MacTeX, TeX Live, or
# BasicTeX + manual `tlmgr install` for the packages listed in paper.tex).

MAIN     := paper
TEX      := pdflatex -interaction=nonstopmode -halt-on-error
BIB      := bibtex
INTERMEDIATES := $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log \
                 $(MAIN).out $(MAIN).toc $(MAIN).fdb_latexmk \
                 $(MAIN).fls $(MAIN).synctex.gz $(MAIN).nav \
                 $(MAIN).snm $(MAIN).vrb

.PHONY: all quick clean distclean arxiv

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib figures/*.pdf
	$(TEX) $(MAIN).tex
	$(BIB) $(MAIN)
	$(TEX) $(MAIN).tex
	$(TEX) $(MAIN).tex
	@echo
	@echo "Build OK -> $(MAIN).pdf"

quick:
	$(TEX) $(MAIN).tex

clean:
	rm -f $(INTERMEDIATES)

distclean: clean
	rm -f $(MAIN).pdf
	rm -f arxiv_submission.tar.gz

arxiv: $(MAIN).pdf
	@# Bundle exactly what arXiv needs: .tex source, .bib, all figures.
	@# arXiv will re-run pdflatex on this on their end.
	tar -czvf arxiv_submission.tar.gz \
	    $(MAIN).tex \
	    references.bib \
	    $(MAIN).bbl \
	    figures/fig_01_schematic.pdf \
	    figures/fig_02_m2c_heatmap.pdf \
	    figures/fig_03_canonical_breakdown.pdf \
	    figures/fig_04_causal_patching.pdf \
	    figures/fig_05_agreement.pdf
	@echo
	@echo "arXiv submission bundle -> arxiv_submission.tar.gz"
	@echo "Upload this file at https://arxiv.org/submit"
