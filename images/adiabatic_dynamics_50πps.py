import os
import mpl

from pathlib import Path

from guess_dynamics import (
    set_xlabel,
    read_csv,
    render_expvals,
    set_panel_title,
    set_yaxis_scale_label,
    render_control_field,
)

COLUMN_WIDTH = 7.7  # cm


def set_x_axis(ax, t):
    mpl.set_axis(
        ax,
        "x",
        t[0],
        t[-1],
        range=(t[0], t[-1]),
        show_opposite=False,
        ticks=[t[0], 100, 200, t[-1]],
        position=("outward", 2),
        ticklabels=["0", "100", "200", str(int(t[-1]))],
        tick_params=dict(length=2, direction="inout"),
        minor_tick_params=dict(length=0),
        label="",  # label should be set with `set_xlabel`
    )


def plot_dynamics(lab_data, moving_data, t_r, frame=2):
    fig_width = COLUMN_WIDTH
    left_margin = 1.0
    right_margin = 0.5
    bottom_margin = 0.80
    top_margin = 0.8
    h = 1.4  # height of main axes (p, θ)
    vgap = 1.25
    hgap = 0.9

    fig_height = 2 * h + bottom_margin + top_margin + vgap
    w = (fig_width - left_margin - right_margin - hgap) / 2

    fig = mpl.new_figure(fig_width, fig_height)

    axs = []

    # bottom left: lab frame momentum #########################################
    ax = fig.add_axes(
        [
            left_margin / fig_width,
            bottom_margin / fig_height,
            w / fig_width,
            h / fig_height,
        ]
    )
    t = lab_data["time (ms)"]
    p = lab_data["p (Mπ/sec)"]
    σp = lab_data["σ_p (Mπ/sec)"]
    render_expvals(ax, t, p, σp, name="p (lab)", line_labels_at=140)
    render_control_field(
        ax,
        t,
        lab_data["ω (π/sec)"],
        xy_label_left=(4, 1.5),  # offset of line label (pt)
        line_label_at=250,
    )
    y0 = -93
    y1 = 93
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
        label=r"$p$ ($M \pi$/s)",
    )
    set_x_axis(ax, t)
    set_panel_title(ax, title="lab frame momentum")
    set_xlabel(ax, "time (ms)")
    axs.append(ax)

    # top left: lab frame position ############################################

    ax = fig.add_axes(
        [
            left_margin / fig_width,
            (bottom_margin + h + vgap) / fig_height,
            w / fig_width,
            h / fig_height,
        ]
    )
    t = lab_data["time (ms)"]
    Δθ = lab_data["Δθ (π)"]
    σθ = lab_data["σ_θ (π)"]
    if abs(abs(Δθ[-1]) - 2) < 1e-15:
        # fix floating point rounding errors resulting in the trajectory not
        # being perfectly closed
        Δθ[-1] = 0.0
    render_expvals(
        ax,
        t,
        Δθ,
        σθ,
        name="Δθ (lab)",
        clip_on=False,
        line_labels_at=50,
        xy_label_left=(-1, -4),  # offset of line label (pt)
        xy_label_right=(-1, 4),  # offset of line label (pt)
    )
    ax.annotate(
        r"$t_r$",
        xy=(100, 2),
        xycoords="data",
        xytext=(1, 3),
        textcoords="offset points",
        ha="left",
        va="top",
        size="small",
    )
    # set_title(ax, "lab frame", ax2=ax2, labelpad_pt=3)
    mpl.set_axis(
        ax,
        "y",
        -2,
        2,
        step=1,
        bounds=(-2, 2),
        show_opposite=False,
        ticks=[-2, 0, 2],
        ticklabels=["", "0", "2"],
        position=("outward", 2),
        tick_params=dict(length=2, direction="inout"),
        minor_tick_params=dict(length=0),
        label=r"$\Delta\theta$ ($\pi$)",
    )
    set_x_axis(ax, t)
    set_panel_title(ax, title="lab frame position")
    axs.append(ax)

    # bottom right: moving frame momentum #####################################

    if frame > 1:

        ax = fig.add_axes(
            [
                (left_margin + w + hgap) / fig_width,
                bottom_margin / fig_height,
                w / fig_width,
                h / fig_height,
            ]
        )
        t = moving_data["time (ms)"]
        p = moving_data["p (Mπ/sec)"]
        σp = moving_data["σ_p (Mπ/sec)"]
        render_expvals(
            ax,
            t,
            p,
            σp,
            name="p (moving)",
            line_labels_at=None,
        )
        y0 = -93
        y1 = 93
        mpl.set_axis(
            ax,
            "y",
            range=(y0, y1),
            bounds=(-43, 43),
            show_opposite=False,
            ticks=[-43, 0, 43],
            position=("outward", 2),
            ticklabels=["", "0", "43"],
            tick_params=dict(length=2, direction="inout"),
            minor_tick_params=dict(length=0),
            label="",
        )
        set_x_axis(ax, t)
        set_panel_title(ax, title="moving frame momentum")
        set_xlabel(ax, "time (ms)")
        axs.append(ax)

    # top right: moving frame position ########################################

    if frame > 1:

        ax = fig.add_axes(
            [
                (left_margin + w + hgap) / fig_width,
                (bottom_margin + h + vgap) / fig_height,
                w / fig_width,
                h / fig_height,
            ]
        )
        t = moving_data["time (ms)"]
        Δθ = moving_data["Δθ (π)"]
        σθ = moving_data["σ_θ (π)"]
        render_expvals(
            ax,
            t,
            Δθ,
            σθ,
            name="Δθ (moving)",
            scale=1000,
            line_labels_at=None,
        )
        y0 = -1.5
        y1 = 1.5
        mpl.set_axis(
            ax,
            "y",
            range=(y0, y1),
            bounds=(-1.35, 1.35),
            show_opposite=False,
            ticks=[-1.35, 0, 1.35],
            position=("outward", 2),
            ticklabels=["", "0", "1.35"],
            tick_params=dict(length=2, direction="inout"),
            minor_tick_params=dict(length=0),
            label="",
        )
        set_yaxis_scale_label(ax, r"$\times\!10^{-3}$")
        set_x_axis(ax, t)
        set_panel_title(ax, title="moving frame position")
        axs.append(ax)

    ###########################################################################

    for ax in axs:
        ax.axvline(x=(t[0] + t_r), lw=0.5, ls="--", color="black")
        ax.axvline(x=(t[-1] - t_r), lw=0.5, ls="--", color="black")

    return fig


def main():
    """Produce an output PDF file (same name as script)."""
    scriptfile = os.path.abspath(__file__)
    datadir = Path(os.path.splitext(scriptfile)[0])
    outfile = Path(os.path.splitext(scriptfile)[0] + ".pdf")
    lab_data = read_csv(datadir / "dynamics_adiabatic_lab.csv")
    moving_data = read_csv(datadir / "dynamics_adiabatic_moving.csv")
    fig = plot_dynamics(lab_data, moving_data, t_r=100.0)
    fig.savefig(outfile, transparent=True, dpi=600)
    for frame in [1, 2]:
        fig = plot_dynamics(lab_data, moving_data, t_r=100.0, frame=frame)
        outfile = Path(os.path.splitext(scriptfile)[0] + f"_{frame}.pdf")
        fig.savefig(outfile, transparent=True, dpi=600)


if __name__ == "__main__":
    main()
