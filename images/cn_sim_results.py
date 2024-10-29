"""Plot script for Sagnac phase"""
import os
import mpl
import numpy as np
from scipy.interpolate import interp1d
from pathlib import Path

from guess_dynamics import (
    read_csv,
    set_panel_title,
    LEFT_LABEL,
    get_minima,
)


def interpolate(x, y, n):
    """Return interpolated x, y with length n"""
    f = interp1d(x, y)
    x_new = np.linspace(x[0], x[-1], num=n)
    y_new = np.array([f(xval) for xval in x_new])
    return x_new, y_new


def render_dataset(
    ax, dataset, title=None, xticks=None, _is_inset=False
):
    """Render the given SagnacDataSet onto the given Axes."""
    Ω, P = interpolate(
        dataset.sc_data["Ω (rad/s)"] / (1e-2 * np.pi),
        dataset.sc_data["|c₋|²"],
        301
    )
    ax.plot(
        Ω,
        P,
        lw=0.8,
        clip_on=_is_inset,
        label="Sagnac",
    )
    if dataset.cn_data is not None:
        ax.scatter(
            dataset.cn_data["Ω (rad/s)"] / (1e-2 * np.pi),
            dataset.cn_data["|c₋|²"],
            s=1.0,
            color=mpl.get_color("red"),
            marker="^",
            clip_on=_is_inset,
            label="CN",
        )
    if dataset.sp_data is not None:
        ax.scatter(
            dataset.sp_data["Ω (rad/s)"] / (1e-2 * np.pi),
            dataset.sp_data["|c₋|²"],
            s=1.0,
            color=mpl.get_color("purple"),
            clip_on=_is_inset,
            label="Quantum",
        )

    if not _is_inset:
        if xticks is None:
            Ω_0 = get_minima(Ω, P, x0=0.1 * Ω[-1], x1=Ω[-1])[-1] / 2.0
            print(f"{Ω_0=}")
            xticks = [
                0,
                Ω_0,
                2 * Ω_0,
            ]
        mpl.set_axis(
            ax,
            "y",
            range=(0, 1),
            bounds=(0, 1),
            show_opposite=False,
            ticks=[0, 1],
            position=("outward", 2),
            ticklabels=["0", "1"],
            tick_params=dict(length=2, direction="inout"),
            minor_tick_params=dict(length=0),
            labelpad=-2,
            label=rf"$\vert c_{{{LEFT_LABEL}}} \vert^2$",
        )
        mpl.set_axis(
            ax,
            "x",
            range=(0, Ω[-1]),
            bounds=(0, Ω[-1]),
            show_opposite=False,
            ticks=[float(v) for v in xticks],
            position=("outward", 2),
            ticklabels=[("%s" % v) for v in xticks],
            tick_params=dict(length=2, direction="inout"),
            minor_tick_params=dict(length=0),
            label=r"$\Omega$ ($10^{-2} \pi$/s)",
            labelpad=0,
        )
        set_panel_title(ax, voffset=10, title=title, centered=True, hoffset=0)


def _inset_y_lbl(v):
    assert 0 < v < 1
    lbl = "%.3f" % v
    return lbl[1:]  # "0.1" → ".1"


def render_inset(ax, bounds, ranges, dataset, yticks=None):
    """Put an inset on the given axes as axes-position `bounds`, showing data
    from `dataset` in the region specified by `ranges`.
    """
    x0, y0, width, height = bounds
    Ω0, Ω1, P0, P1 = ranges
    axins = ax.inset_axes([x0, y0, width, height])
    render_dataset(axins, dataset, _is_inset=True)
    axins.set_xlim(Ω0, Ω1)
    axins.set_ylim(P0, P1)
    axins.set_xticks([])
    axins.set_yticks([])
    if yticks is not None:
        mpl.set_axis(
            axins,
            "y",
            P0,
            P1,
            ticks=yticks,
            ticklabels=[_inset_y_lbl(v) for v in yticks],
            tick_params=dict(
                length=2,
                width=0.25,
                direction="inout",
                labelsize="xx-small",
                pad=0.5,
            ),
            minor_tick_params=dict(length=0),
            label="",
        )
    for axis in ["top", "bottom", "left", "right"]:
        axins.spines[axis].set_linewidth(0.25)
    rp, lps = ax.indicate_inset_zoom(
        axins, edgecolor="black", lw=0.25, alpha=1.0, clip_on=False
    )
    for lp in lps:
        lp.set(linewidth=0.25)


def plot_cn_sim_results(dataset_2cycles, dataset_10cycles, frame=2):
    """Plot the figure."""
    fig_width = 10.75 # cm

    left_margin = 0.65
    right_margin = 2.0
    top_margin = 1.00
    bottom_margin = 0.9
    h_gap = 0.5
    h = 1.6

    fig_height = bottom_margin + h + top_margin
    w = (fig_width - left_margin - right_margin - h_gap) / 2.0

    fig = mpl.new_figure(fig_width, fig_height)

    # 2 cycles
    ax = fig.add_axes(
        [
            left_margin / fig_width,
            bottom_margin / fig_height,
            w / fig_width,
            h / fig_height,
        ]
    )
    render_dataset(ax, dataset_2cycles, xticks=[0, 18], title="2 cycles @ $\omega_0 = 10$ π/s")
    Ωmax = ax.get_xlim()[1]

    if frame > 1:

        # 10 cycles
        ax = fig.add_axes(
            [
                (left_margin + w + h_gap) / fig_width,
                bottom_margin / fig_height,
                w / fig_width,
                h / fig_height,
            ]
        )
        render_dataset(
            ax, dataset_10cycles, xticks=[0, 3.6, 18], title="10 cycles @ $\omega_0 = 50$ π/s"
        )
        ax.set_ylabel("")

    ax.legend(
        loc='lower left', bbox_to_anchor=(1.1, 0.5), ncol=1,
        columnspacing=0.5, handlelength=1.5, borderaxespad=0
    )

    return fig


class SagnacDataSet:
    """Data for one Sagnac curve, multiple methods."""

    def __init__(self, *, sc_data, cn_data=None, sp_data=None):
        # all data are a dicts column name => numpy array
        self.sc_data = sc_data  # Semi-Classical ("Sagnac")
        self.cn_data = cn_data  # Crank Nicoloson (optional)
        self.sp_data = sp_data  # Split Propagator (optional)

    @classmethod
    def read(cls, *, sc_file, cn_file=None, sp_file=None):
        """Instantiate from file names"""
        kwargs = {}
        kwargs["sc_data"] = read_csv(sc_file)
        if cn_file is not None:
            kwargs["cn_data"] = read_csv(cn_file)
        if sp_file is not None:
            kwargs["sp_data"] = read_csv(sp_file)
        for data in kwargs.values():
            assert "Ω (rad/s)" in data.keys()
            assert "|c₋|²" in data.keys()
        return cls(**kwargs)


def main():
    """Produce an output PDF file (same name as script)."""
    scriptfile = os.path.abspath(__file__)
    datadir = Path(os.path.splitext(scriptfile)[0])
    outfile = Path(os.path.splitext(scriptfile)[0] + ".pdf")
    dataset_2cycles = SagnacDataSet.read(
        # cn_file=(datadir / "quantum_1.csv"),
        sc_file=(datadir / "sc_10πps_2cyc.csv"),
        sp_file=(datadir / "sagnac_adiabatic_10πps_2cyc.csv"),
    )
    dataset_10cycles = SagnacDataSet.read(
        sc_file=(datadir / "sc_50πps_10cyc.csv"),
        sp_file=(datadir / "sagnac_adiabatic_50πps_10cyc.csv"),
    )
    fig = plot_cn_sim_results(dataset_2cycles, dataset_10cycles)
    fig.savefig(outfile, transparent=True, dpi=600)
    for frame in [1, 2]:
        fig = plot_cn_sim_results(dataset_2cycles, dataset_10cycles, frame=frame)
        outfile = Path(os.path.splitext(scriptfile)[0] + f"_{frame}.pdf")
        fig.savefig(outfile, transparent=True, dpi=600)


if __name__ == "__main__":
    main()
