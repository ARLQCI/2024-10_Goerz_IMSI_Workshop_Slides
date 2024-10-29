import os
import shutil
import functools
import subprocess
from tempfile import TemporaryDirectory
from pathlib import Path
from guess_dynamics import read_csv

import concurrent.futures


TEX = r"""
\documentclass[compress, aspectratio=169]{beamer}
\usepackage{amsmath}
\renewcommand{\familydefault}{\sfdefault}

\usepackage{tikz}
\usetikzlibrary{calc}
\usepackage{xcolor}

\definecolor{DarkBlue}{rgb}{0.1,0.1,0.5}
\definecolor{DarkRed}{rgb}{0.75,0.,0.}

\usepackage[psfixbb,graphics,tightpage,active]{preview}
\PreviewEnvironment{tikzpicture}

\begin{document}

\newcommand{\xangle}{7}
\newcommand{\yangle}{137.5}
\newcommand{\zangle}{90}

\newcommand{\xlength}{1}
\newcommand{\ylength}{0.5}
\newcommand{\zlength}{1}

\newcommand{\R}{2.5}

\pgfmathsetmacro{\xx}{\xlength*cos(\xangle)}
\pgfmathsetmacro{\xy}{\xlength*sin(\xangle)}
\pgfmathsetmacro{\yx}{\ylength*cos(\yangle)}
\pgfmathsetmacro{\yy}{\ylength*sin(\yangle)}
\pgfmathsetmacro{\zx}{\zlength*cos(\zangle)}
\pgfmathsetmacro{\zy}{\zlength*sin(\zangle)}

\tikzset{
  A/.pic = {
    \node[inner sep = 0pt, anchor = south] at (0,0) {%
      \includegraphics[width=1.3cm]{trapped_atom_A}
    };
    \node[fill = white, inner sep=0pt, rounded corners, fill opacity = 0.9]
    at (-8pt, 1.8cm) {\color{DarkRed}$V_{+}$};
  }
}
\tikzset{
  B/.pic = {
    \node[inner sep = 0pt, anchor = south] at (0,0) {%
      \includegraphics[width=1.3cm]{trapped_atom_B}
    };
    \node[fill = white, inner sep=0pt, rounded corners, fill opacity = 0.9]
    at (8pt, 1.8cm) {\color{DarkBlue}$V_{-}$};
  }
}


\begin{frame}
\begin{tikzpicture}
[   x={(\xx cm,\xy cm)},
    y={(\yx cm,\yy cm)},
    z={(\zx cm,\zy cm)},
]

  \fill[color = white] (-3.5cm,-1cm) rectangle (4cm, 3.5cm);
  \node[below] at (0, 3.5cm) {\small $t = #TIME#$~ms};

  \fill[gray!30] (0,0,0) circle (\R);
  \draw[dashed, line width = 1pt] (0,0,0) circle (\R);
  \fill[color = black!70] (0,0,0) circle (1pt);
  \draw[->, line width = 1pt, color = black!70]
    (0,0,0) -- node[midway, below] {$R$} (0:\R);

  \path (-#PHI#:\R) pic {A};
  \path (#PHI#:\R) pic {B};


\end{tikzpicture}
\end{frame}

\end{document}
"""


def texcode(t, phi):  # t in ms, phi in π
    phi_degree = 180.0 * phi
    return TEX.replace("#PHI#", "%.2f" % phi_degree).replace(
        "#TIME#", "%d" % t
    )


def render_frame(t, ϕ, rootdir=Path.cwd(), outdir=Path.cwd()):
    basename = "frame_%03d" % t
    texname = basename + ".tex"
    pdfname = basename + ".pdf"
    pngname = basename + ".png"

    if (outdir / pngname).is_file():
        return

    with TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / texname).write_text(texcode(t, ϕ))
        shutil.copy(rootdir / "trapped_atom_A.pdf", tmp)
        shutil.copy(rootdir / "trapped_atom_B.pdf", tmp)

        cmd = ["pdflatex", "-interaction=batchmode", texname]
        subprocess.run(cmd, cwd=tmp)

        cmd = [
            "convert",
            "-density",
            "300",
            pdfname,
            "-quality",
            "100",
            pngname,
        ]
        subprocess.run(cmd, cwd=tmp)

        if int(t) == 0 or int(t) == 100 or int(t) == 200 or int(t) == 300:
            shutil.copy(tmp / pdfname, outdir)
        shutil.copy(tmp / pngname, outdir)
        # for the three different phases:
        if t <= 100:
            shutil.copy(tmp / pngname, outdir / "1")
        if 100 <= t <= 200:
            shutil.copy(tmp / pngname, outdir / "2")
        if 200 <= t:
            shutil.copy(tmp / pngname, outdir / "3")


def main():
    scriptfile = os.path.abspath(__file__)
    outdir = Path(os.path.splitext(scriptfile)[0])
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "1").mkdir(parents=True, exist_ok=True)
    (outdir / "2").mkdir(parents=True, exist_ok=True)
    (outdir / "3").mkdir(parents=True, exist_ok=True)
    rootdir = outdir.parent
    datadir = rootdir / "adiabatic_dynamics_50πps"
    lab_data = read_csv(datadir / "dynamics_adiabatic_lab.csv")

    t = lab_data["time (ms)"]
    Δθ = lab_data["Δθ (π)"]

    data = list(zip(t, Δθ))

    def worker(t_ϕ):
        render_frame(t_ϕ[0], t_ϕ[1], rootdir=rootdir, outdir=outdir)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(worker, data)
        for result in results:
            pass


if __name__ == "__main__":
    main()
