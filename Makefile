PDFLATEXOPTS = -file-line-error -interaction=nonstopmode -halt-on-error -synctex=1

all: slides.pdf

.venv/bin/python:
	python3.12 -m venv .venv
	./.venv/bin/python -m pip install -r requirements.txt

slides.pdf: slides.tex .venv/bin/python $(wildcard images/*.py images/*.tex images/*.pdf images/*.tikz images/*.png) arlwide_theme/theme.tex
	$(MAKE) -C images all
	pdflatex $(PDFLATEXOPTS) slides
	pdflatex $(PDFLATEXOPTS) slides

handout.pdf: slides.pdf
	pdfjam slides.pdf 1,7,8,9,11,12,15,17,18,21,23,25,27,35,37,38,40,42,43,44,46,48,49,50,52,54,55,56,63,68,70,72,79,82,83,84,85,86,92,94,96,97,98,99,101,102,107,108,110,113,114,116,119,121,125,127,128,133,139,145,146,172 --fitpaper true -o handout.pdf

pdflatex:
	@echo "Compiling Main File ..."
	pdflatex $(PDFLATEXOPTS) slides
	@echo "Done"

update:
	pdflatex $(PDFLATEXOPTS) slides

clean:
	@echo "Cleaning up files from LaTeX compilation ..."
	$(MAKE) -C images clean
	rm -f *.aux
	rm -f *.log
	rm -f *.toc
	rm -f *.bbl
	rm -f *.blg
	rm -rf *.out
	rm -f *.bak
	rm -f *.ilg
	rm -f *.snm
	rm -f *.nav
	rm -f *.fls
	rm -f *.table
	rm -f *.gnuplot
	rm -f *.fdb_latexmk
	rm -f *.synctex.gz
	@echo "Done"

distclean: clean
	$(MAKE) -C images distclean
	rm -rf .venv
	rm -f slides.pdf
	rm -f handout.pdf

.PHONY: all pdflatex pdf png clean distclean
