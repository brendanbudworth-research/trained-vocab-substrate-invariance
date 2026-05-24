# Build the arXiv methods-note PDF.
#   `make`         -> full build (pdflatex + bibtex + pdflatex x 2)
#   `make quick`   -> single pdflatex pass (fast, refs may be broken)
#   `make clean`   -> remove TeX intermediates (keeps paper.pdf)
#   `make distclean` -> remove everything including paper.pdf
#   `make arxiv`   -> produce arxiv_submission.tar.gz with sources + figures
#
# Requires a TeX install with pdflatex + bibtex (MacTeX, TeX Live, or
# BasicTeX + tlmgr-installed packages listed in paper.tex).

MAIN     := paper
TEX      := pdflatex -interaction=nonstopmode -halt-on-error
BIB      := bibtex
INTERMEDIATES := $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log \
                 $(MAIN).out $(MAIN).toc $(MAIN).fdb_latexmk \
                 $(MAIN).fls $(MAIN).synctex.gz

.PHONY: all quick clean distclean arxiv

all: $(MAIN).pdf

FIGURES := $(wildcard figures/*.pdf)

$(MAIN).pdf: $(MAIN).tex references.bib $(FIGURES)
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
	tar -czvf arxiv_submission.tar.gz \
	    $(MAIN).tex \
	    references.bib \
	    $(MAIN).bbl \
	    figures/*.pdf
	@echo
	@echo "arXiv submission bundle -> arxiv_submission.tar.gz"
	@echo "Upload this file at https://arxiv.org/submit"
