import os

from pathlib import Path

import guess_dynamics
from guess_dynamics import read_csv, plot_dynamics

guess_dynamics.SHOWING_GUESS = False


def transform_sagnac_data(sagnac_data, guess_sagnac_data):
    P = sagnac_data["P_right"]
    del sagnac_data["P_right"]
    sagnac_data["optimized"] = 1 - P  # "left" = 1 - "right"
    sagnac_data["unoptimized"] = 1 - guess_sagnac_data["P_right"]
    return sagnac_data


def main():
    """Produce an output PDF file (same name as script)."""
    scriptfile = os.path.abspath(__file__)
    datadir = Path(os.path.splitext(scriptfile)[0])
    outfile = Path(os.path.splitext(scriptfile)[0] + ".pdf")
    lab_data = read_csv(datadir / "dynamics_opt_lab.csv")
    lab_data_theta = read_csv(datadir / "dynamics_opt_lab_theta.csv")
    moving_data = read_csv(datadir / "dynamics_opt_moving.csv")
    fig = plot_dynamics(
        lab_data,
        moving_data,
        t_r=0.15,
        lab_data_theta=lab_data_theta,
    )
    # fig.suptitle("optimized dynamics", y=1.0)
    fig.savefig(outfile, transparent=True, dpi=600)
    for frame in [1, 2, 3, 4, 5, 6]:
        fig = plot_dynamics(
            lab_data,
            moving_data,
            t_r=0.15,
            lab_data_theta=lab_data_theta,
            frame=frame,
        )
        outfile = Path(os.path.splitext(scriptfile)[0] + f"_{frame}.pdf")
        fig.savefig(outfile, transparent=True, dpi=600)


if __name__ == "__main__":
    main()
