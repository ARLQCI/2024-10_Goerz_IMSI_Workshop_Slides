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
	pdfjam slides.pdf 1,2,3,7,9,10,11,12,13,14,15,16,18,20,24,25,26,28,29,30,31,32,43,47,48,53,56,57,61,62,64,68,69,73,74,78,81,84,88,89,96,100,103,104,105,106 --fitpaper true -o handout.pdf

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
