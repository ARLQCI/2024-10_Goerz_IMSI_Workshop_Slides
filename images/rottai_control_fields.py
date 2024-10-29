import os
import mpl
from functools import partial
from scipy.interpolate import interp1d
from pathlib import Path

from guess_dynamics import read_csv


def plot_control_fields(
    time,
    ω_guess,
    ω_opt,
    frame=2,
):
    fig_width = 6
    left_margin = 1.25
    right_margin = 0.45
    bottom_margin = 0.80
    top_margin = 0.25
    h = 2.5

    fig_height = h + bottom_margin + top_margin
    w = (fig_width - left_margin - right_margin)

    fig = mpl.new_figure(fig_width, fig_height)

    ax = fig.add_axes(
        [
            left_margin / fig_width,
            bottom_margin / fig_height,
            w / fig_width,
            h / fig_height,
        ]
    )

    ax.plot(time, ω_guess, label="guess")
    if frame == 2:
        ax.plot(time, ω_opt, clip_on=False, label="opt")
    ax.axhline(y=0, lw=0.5, ls="--", color="black")
    ax.axhline(y=50, lw=0.5, ls="--", color="black")
    ax.legend(labelspacing=0.5)

    y0 = -252
    y1 = 252
    mpl.set_axis(
        ax,
        "y",
        range=(y0, y1),
        bounds=(y0, y1),
        show_opposite=False,
        ticks=[y0, -50, 0, 50, y1],
        position=("outward", 2),
        ticklabels=["", "", "0", "50", str(y1)],
        tick_params=dict(length=2, direction="inout"),
        minor_tick_params=dict(length=0),
        label="amplitude (MHz)",
        lw=0.5,
    )

    t_r = time[-1]
    mpl.set_axis(
        ax,
        "x",
        range=(0, t_r),
        bounds=(0, t_r),
        show_opposite=False,
        ticks=[0, t_r],
        position=("outward", 2),
        ticklabels=["0", str(int(t_r))],
        tick_params=dict(length=2, direction="inout"),
        minor_tick_params=dict(length=0),
        labelpad=0,
        label="time (μs)",
        lw=0.5,
    )
    ax.tick_params(right=False)

    return fig


def main():
    """Produce an output PDF file (same name as script)."""
    scriptfile = os.path.abspath(__file__)
    datadir = Path(os.path.splitext(scriptfile)[0])
    data = read_csv(datadir / "2023-05-17_OCT_tr=150μs_V0=0.2MHz_R=26μm_ω=50πps_guess_opt_controls.csv")

    outfile = Path(os.path.splitext(scriptfile)[0] + ".pdf")

    for frame in [1, 2]:

        fig = plot_control_fields(
            data["time (μs)"],
            data["ω_guess (π/sec)"],
            data["ω_opt (π/sec)"],
            frame=frame,
        )
        # fig.suptitle("unoptimized nonadiabatic dynamics", y=1.0)
        outfile_frame = Path(os.path.splitext(scriptfile)[0] + f"_{frame}.pdf")
        fig.savefig(outfile_frame, transparent=True, dpi=600)
        fig.savefig(outfile, transparent=True, dpi=600)


if __name__ == "__main__":
    main()
