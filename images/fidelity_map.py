"""Plot script for contrast"""
import os
import mpl
import numpy as np
import matplotlib.ticker as ticker
from pathlib import Path

COLUMN_WIDTH = 7.7  # cm

μs = 1.0
sec = 1e6 * μs
MHz = 2 * np.pi


def plot_fidelity_map(
    separation_time_values_sec, potential_depth_values_MHz, data,
    frame=2
):
    fig_width = COLUMN_WIDTH

    left_margin = 1.2
    top_margin = 0.65
    bottom_margin = 0.9
    w = 5.20
    h = 3.45
    cbar_hgap = 0.35
    cbar_width = 0.25

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

    contour = ax.contourf(
        separation_time_values_sec,
        potential_depth_values_MHz,
        np.clip(data, 0, 1.0),
        cmap="cubehelix",
        vmin=0.0,
        vmax=1.0,
        levels=np.linspace(0, 1.0, num=21),
    )

    print(f"minimum fidelity: {np.min(data)}")
    print(f"maximum fidelity: {np.max(data)}")

    ax.set_xscale("log")
    ax.set_xlabel("separation time (seconds)")
    mpl.set_axis(
        ax, "y", start=0.1, stop=2.2, label=r"potential depth $V_0$ (MHz)"
    )
    ax.set_yticks([0.1, 0.5, 1.0, 1.5, 2.0, 2.2])
    ax.set_yticks(np.arange(0.1, 2.2, 0.1), minor=True)
    ax.yaxis.set_ticks_position('both')
    ax.set_title(
        r"Separation Fidelity $\vert\langle \Psi(t_r) | \Psi_{\mathrm{tgt}}\rangle\vert^2$",
        fontsize=r"medium",
        pad=8.0,
    )
    ax.tick_params(axis="both", direction="out", which="both", right=False, top=True)
    ax.tick_params(axis="both", which="minor", direction="out", length=1.5, right=False, top=True)
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=8))
    ax.xaxis.set_minor_locator(
        ticker.LogLocator(base=10.0, subs=np.arange(0.1, 1, 0.1), numticks=10)
    )

    ax_cbar = fig.add_axes(
        [
            (left_margin + w + cbar_hgap) / fig_width,
            bottom_margin / fig_height,
            cbar_width / fig_width,
            h / fig_height,
        ]
    )
    fig.colorbar(contour, cax=ax_cbar, ticks=np.linspace(0, 1.0, num=11))

    ax_cbar.set_yticks(np.linspace(0, 1.0, num=21), minor=True)
    ax_cbar.tick_params(axis="both", which="minor", direction="out", length=1.5)
    ax_cbar.tick_params(axis="both", direction="out", which="both")

    point_adiabatic = (0.1, 2.2)
    point_nonadiabatic = (1.5e-4, 0.2)
    ax.plot(*point_adiabatic, "s", color="red", transform=ax.transData, markersize=2, clip_on=False)

    if frame > 1:
        ax.axvline(x=1.5e-4, lw=0.5, ls="--", color="black")
        ax.plot(*point_nonadiabatic, "D", color="red", transform=ax.transData, markersize=2)

    return fig


def main():
    """Produce an output PDF file (same name as script)."""
    scriptfile = os.path.abspath(__file__)
    datadir = Path(os.path.splitext(scriptfile)[0])
    outfile = Path(os.path.splitext(scriptfile)[0] + ".pdf")
    datafile = datadir / "2023-05-16_50pps_map_splitting_fidelity.npz"
    data = np.load(datafile)
    separation_time_orders_of_magnitude = np.linspace(-1, 5, num=121)
    separation_time_values = np.array(
        [(10**x * μs) for x in separation_time_orders_of_magnitude]
    )
    potential_depth_values = np.linspace(0.1 * MHz, 2.2 * MHz, num=106)
    fig = plot_fidelity_map(
        separation_time_values / sec, potential_depth_values / MHz, data
    )
    fig.savefig(outfile, transparent=True, dpi=600)
    for frame in [1, 2]:
        fig = plot_fidelity_map(
            separation_time_values / sec, potential_depth_values / MHz, data,
            frame=frame
        )
        outfile = Path(os.path.splitext(scriptfile)[0] + f"_{frame}.pdf")
        fig.savefig(outfile, transparent=True, dpi=600)


if __name__ == "__main__":
    main()
