import os
from pathlib import Path

from guess_dynamics import read_csv
from guess_sagnac import plot_sagnac


def transform_sagnac_data(sagnac_data, guess_sagnac_data):
    P = sagnac_data["P_right"]
    del sagnac_data["P_right"]
    sagnac_data["optimized"] = 1 - P  # "left" = 1 - "right"
    sagnac_data["unoptimized"] = 1 - guess_sagnac_data["P_right"]
    return sagnac_data


def main():
    """Produce an output PDF file (same name as script)."""
    scriptfile = os.path.abspath(__file__)
    datadir = Path(os.path.splitext(scriptfile)[0]).parent / "opt_dynamics"
    outfile = Path(os.path.splitext(scriptfile)[0] + ".pdf")

    sagnac_data = read_csv(datadir / "sagnac_opt.csv")
    datadir_guess = datadir / ".." / "guess_dynamics"
    guess_sagnac_data = read_csv(datadir_guess / "sagnac_guess.csv")
    sagnac_data = transform_sagnac_data(sagnac_data, guess_sagnac_data)
    fig = plot_sagnac(sagnac_data)
    fig.savefig(outfile, transparent=True, dpi=600)
    for frame in [1, 2]:
        fig = plot_sagnac(sagnac_data, frame=frame)
        outfile = Path(os.path.splitext(scriptfile)[0] + f"_{frame}.pdf")
        fig.savefig(outfile, transparent=True, dpi=600)


if __name__ == "__main__":
    main()
