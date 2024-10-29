import os
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

import mpl
from guess_dynamics import read_csv, LEFT_LABEL


COLUMN_WIDTH = 7.7  # cm


def render_and_format_sagnac(
    ax,
    sagnac_data,
    frame=2,
):
    keys = list(sagnac_data.keys())
    key = keys[1]
    P_func = interp1d(
        100 * sagnac_data["Ω (π/sec)"], sagnac_data[key], kind="cubic"
    )
    Ω = np.linspace(0, 100 * sagnac_data["Ω (π/sec)"][-1], 200)
    P = np.array([P_func(val) for val in Ω])
    Ω_0 = Ω[np.argmin(P)]
    if frame >= 2:
        ax.plot(Ω, P, label=key, clip_on=False)
    if len(keys) > 2:
        key = keys[2]
        extra_func = interp1d(
            100 * sagnac_data["Ω (π/sec)"], sagnac_data[key], kind="cubic"
        )
        extra = np.array([extra_func(val) for val in Ω])
        ax.plot(Ω, extra, label=key, clip_on=False, color="black", ls="dotted")
    ax.legend(loc="center", bbox_to_anchor=[0.5, 1.2], ncols=2)

    print(f"Sagnac population in range {np.min(P)}, {np.max(P)}")
    print(f"  for Ω ∈ [0, {Ω[-1]}]")
    print(f"  min(P) at Ω = {Ω_0}")
    mpl.set_axis(
        ax,
        "y",
        range=(0, 1),
        bounds=(0, 1),
        show_opposite=False,
        ticks=[0, 0.38, 0.62, 1],  # 11 is σ_p(0)
        position=("outward", 2),
        ticklabels=["0", ".38", ".62", "1"],
        tick_params=dict(length=2, direction="inout"),
        minor_tick_params=dict(length=0),
        label=rf"$\vert c_{{{LEFT_LABEL}}} \vert^2$",
    )
    xticks = [
        0,
        1.8,
        3.6,
    ]
    mpl.set_axis(
        ax,
        "x",
        range=(0, Ω[-1]),
        bounds=(0, Ω[-1]),
        show_opposite=False,
        ticks=xticks,
        position=("outward", 2),
        ticklabels=["%.1f" % v for v in xticks],
        tick_params=dict(length=2, direction="inout"),
        minor_tick_params=dict(length=0),
        label=r"$\Omega$ ($10^{-2} \pi$/s)",
        labelpad=0,
    )


def plot_sagnac(sagnac_data, frame=2):

    fig_width = COLUMN_WIDTH
    left_margin = 1.2
    right_margin = 0.5
    bottom_margin = 0.80
    top_margin = 0.65
    h = 2.0  # height of main axes

    w = fig_width - left_margin - right_margin
    fig_height = bottom_margin + h + top_margin

    fig = mpl.new_figure(fig_width, fig_height)

    ax = fig.add_axes(
        [
            left_margin / fig_width,
            bottom_margin / fig_height,
            w / fig_width,
            h / fig_height,
        ]
    )

    render_and_format_sagnac(ax, sagnac_data, frame=frame)

    return fig


def transform_sagnac_data(sagnac_data):
    P = sagnac_data["P_right"]
    del sagnac_data["P_right"]
    sagnac_data["unoptimized"] = 1-P  # "left" = 1 - "right"
    Δ = P - 0.5
    Δ /= np.max(Δ)
    sagnac_data["Sagnac"] = (0.5 * (1 + Δ))
    # we're cheating a bit here by simply setting the Sagnac curve simply to
    # the stretched data from the guess, but we've checked independently that
    # this indeed matches the analytical Sagnac curve exactly.
    return sagnac_data


def main():
    """Produce an output PDF file (same name as script)."""
    scriptfile = os.path.abspath(__file__)
    datadir = Path(os.path.splitext(scriptfile)[0]).parent / "guess_dynamics"
    outfile = Path(os.path.splitext(scriptfile)[0] + ".pdf")
    sagnac_data = read_csv(datadir / "sagnac_guess.csv")
    sagnac_data = transform_sagnac_data(sagnac_data)
    fig = plot_sagnac(sagnac_data)
    fig.savefig(outfile, transparent=True, dpi=600)
    for frame in [1, 2]:
        fig = plot_sagnac(sagnac_data, frame=frame)
        outfile = Path(os.path.splitext(scriptfile)[0] + f"_{frame}.pdf")
        fig.savefig(outfile, transparent=True, dpi=600)


if __name__ == "__main__":
    main()
