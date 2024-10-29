ALL = guess_dynamics.pdf opt_dynamics.pdf adiabatic_dynamics_50πps.pdf animate_rottai.mp4 animate_rottai_1.mp4 animate_rottai_2.mp4 animate_rottai_3.mp4 cn_sim_results.pdf fidelity_map.pdf guess_sagnac.pdf opt_sagnac.pdf ramsey.pdf
PDFLATEXOPTS = -file-line-error -interaction=nonstopmode -halt-on-error
PYTHON = ../.venv/bin/python

all: $(ALL)

%.pdf: %.tex
	@echo "\n** generate figure: $< -> $@"
	pdflatex $(PDFLATEXOPTS) $<

%.pdf: %.py matplotlibrc
	@echo "\n** generate figure: $< -> $@"
	$(PYTHON) $<

%.png: %.py matplotlibrc
	@echo "\n** generate figure: $< -> $@"
	$(PYTHON) $<

%.tex: %.py
	$(PYTHON) $<


animate_rottai: animate_rottai.py
	$(PYTHON) $<
	touch $@

animate_rottai.mp4: animate_rottai
	ffmpeg -pattern_type glob -framerate 25 -i './animate_rottai/frame*.png' -vf "scale=890:540" -c:v libx264 -r 30 -pix_fmt yuv420p -y $@

animate_rottai_1.mp4: animate_rottai
	ffmpeg -pattern_type glob -framerate 25 -i './animate_rottai/1/frame*.png' -vf "scale=890:540" -c:v libx264 -r 30 -pix_fmt yuv420p -y $@

animate_rottai_2.mp4: animate_rottai
	ffmpeg -pattern_type glob -framerate 25 -i './animate_rottai/2/frame*.png' -vf "scale=890:540" -c:v libx264 -r 30 -pix_fmt yuv420p -y $@

animate_rottai_3.mp4: animate_rottai
	ffmpeg -pattern_type glob -framerate 25 -i './animate_rottai/3/frame*.png' -vf "scale=890:540" -c:v libx264 -r 30 -pix_fmt yuv420p -y $@

clean:
	@rm -f *.aux
	@rm -f *.log
	@rm -f *.toc
	@rm -f *.blg
	@rm -rf *.out
	@rm -f *.bak
	@rm -f *.ilg
	@rm -f *.snm
	@rm -f *.nav
	@rm -f *.table
	@rm -f *.dvi
	@rm -f *.fls
	@rm -f *.bbl
	@rm -f *~
	@rm -f *Notes.bib
	@rm -f *-eps-converted-to.pdf
	@rm -f *.fdb_latexmk
	@rm -f *.synctex.gz*
	@rm -f ./latexrun
	@rm -rf __pycache__

distclean: clean
	@rm -rf animate_rottai
	@rm -f benchmark_cheby_*.pdf
	@rm -f rottai_control_fields_*.pdf
	@rm -f ramsey_*.pdf
	@rm -f cn_sim_results_*.pdf
	@rm -f opt_dynamics_*.pdf
	@rm -f fidelity_map_*.pdf
	@rm -f guess_sagnac_*.pdf
	@rm -f guess_dynamics_*.pdf
	@rm -f opt_sagnac_*.pdf
	@rm -f adiabatic_dynamics_50πps_*.pdf
	@rm -f $(ALL)

.PHONY: all clean distclean
