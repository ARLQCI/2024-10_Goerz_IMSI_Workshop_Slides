# Optimal Control Techniques for Quantum Interferometry

This repository contains the slides for the talk "Optimal Control Techniques for Quantum Interferometry" presented at the [Quantum Hardware IMSI Workshop](https://www.imsi.institute/activities/statistical-methods-and-mathematical-analysis-for-quantum-information-science/quantum-hardware/) held October 28 – 31, 2024, in Chicago.

## Prerequisites

* Python 3.12
* LaTeX installation like [TeX Live](https://www.tug.org/texlive/)
* `ffmpeg`
* `make` on a Unix system

## Compilation

* Run `make` to generate all figures and create `slides.pdf`.
* Run `make handout.pdf` to create the "handout" version of the slides.

If `make` is not available or you are not running on a Unix system, inspect the `Makefile` for the commands required to compile the slides.


## Cleanup

* Run `make distclean` to delete all generated files, including the local `.venv` folder with the Python environment for creating the figures.
