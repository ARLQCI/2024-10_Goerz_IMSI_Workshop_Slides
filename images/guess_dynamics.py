import os
import mpl
import csv
import numpy as np
from functools import partial
from scipy.interpolate import interp1d
from matplotlib.transforms import blended_transform_factory
from pathlib import Path

COLUMN_WIDTH = 7.7  # cm
RASTERIZED = False

RIGHT_LABEL = r"+"  # should be initial state!
LEFT_LABEL = r"-"

SHOWING_GUESS = True
# SHOWING_GUESS=False means we're calling plot_dynamics from opt_dynamics.py
# This is used for some minor differences in how line labels are placed.

#_OMEGA_OPT_OF_T = r"$\omega_{\kern-0.5pt\mathrm{opt}}\kern-0.7pt(\kern-0.3pt{t}\kern-0.3pt)$"
_OMEGA_OPT_OF_T = r"$\omega_{\mathrm{opt}}(t)$"


def read_csv(file):
    """Read the given csv `file` into a dict mapping column names to numpy
    arrays."""
    with open(file, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        data_dict = {}
        for row in reader:
            for column, value in row.items():
                if column not in data_dict:
                    data_dict[column] = []
                data_dict[column].append(float(value))

        for column, values in data_dict.items():
            data_dict[column] = np.array(values)
        return data_dict


def set_xlabel(ax, text, ax2=None, labelpad_pt=12):
    """Set an xlabel as a label on the figure canvas.

    The `text` will be placed centered on the bottom axis of `ax`, or centered
    on the combined `ax` and `ax2` if `ax2` is given. Vertically, the top of
    the `text` will be moved down by `labelpad_pt`.

    This allows for exact and consistent placement for for normal and broken
    axes.
    """
    bb = ax.get_position().bounds
    x0 = bb[0]
    x1 = x0 + bb[2]
    y_pos = bb[1]
    if ax2 is not None:
        bb = ax2.get_position().bounds
        x1 = bb[0] + bb[2]
    x_pos = x0 + 0.5 * (x1 - x0)
    ax.annotate(
        text,
        xy=(x_pos, y_pos),
        xycoords="figure fraction",
        ha="center",
        va="top",
        xytext=(0, -labelpad_pt),
        textcoords="offset points",
    )


def set_panel_title(ax, title, hoffset=18, voffset=9, centered=False):
    """Set a panel label."""
    ax.annotate(
        title,
        xy=(0.5 if centered else 0, 1),
        xycoords="axes fraction",
        ha=("center" if centered else "left"),
        va="bottom",
        xytext=(-hoffset, voffset),
        textcoords="offset points",
    )


def set_broken_time_axis(ax1, ax2, Δt=1.5, T=200.15):
    offset = 2  # pt
    # offset must match the `position=("outward", offset)` for set_axis(…)
    mpl.set_axis(
        ax1,
        "x",
        range=(0, Δt),
        bounds=(0, Δt),
        show_opposite=False,
        ticks=[0, 1],
        position=("outward", offset),
        tick_params=dict(length=2, direction="inout"),
        minor_tick_params=dict(length=0),
        label="",
    )
    mpl.set_axis(
        ax2,
        "x",
        range=(T - Δt, T),
        bounds=(T - Δt, T),
        show_opposite=False,
        ticks=[199, T],
        position=("outward", offset),
        ticklabels=["199", str(T)],
        tick_params=dict(length=2, direction="inout"),
        minor_tick_params=dict(length=0),
        label="",
    )
    ax1.spines["right"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.set_ylim(ax1.get_ylim())
    ax2.yaxis.set_ticks([])
    ax1.annotate(
        "/",
        xy=(1, 0),
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize="x-small",
        xytext=(0, -offset),
        textcoords="offset points",
    )
    ax2.annotate(
        "/",
        xy=(0, 0),
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize="x-small",
        xytext=(0, -offset),
        textcoords="offset points",
    )


def set_title(ax, text, ax2=None, labelpad_pt=6):
    """Set `text` as an axis title on the figure canvas.

    The `text` will be placed centered on the top of `ax`, or centered
    on the combined `ax` and `ax2` if `ax2` is given. Vertically, the bottom of
    the `text` will be moved up by `labelpad_pt`.

    This allows for exact and consistent placement for for normal and broken
    axes.
    """
    bb = ax.get_position().bounds
    x0 = bb[0]
    x1 = x0 + bb[2]
    y_pos = bb[1] + bb[3]
    if ax2 is not None:
        bb = ax2.get_position().bounds
        x1 = bb[0] + bb[2]
    x_pos = x0 + 0.5 * (x1 - x0)
    ax.annotate(
        text,
        xy=(x_pos, y_pos),
        xycoords="figure fraction",
        ha="center",
        va="bottom",
        xytext=(0, labelpad_pt),
        textcoords="offset points",
    )


def set_yaxis_scale_label(ax, text):
    """Set `text` above the y-axis to indicate a scale."""
    ax.annotate(
        text,
        xy=(0, 1),
        xycoords="axes fraction",
        ha="left",
        va="bottom",
        xytext=(-5, 0.1),
        # xytext=(-16, 3),
        fontsize="xx-small",
        textcoords="offset points",
    )


def left_bottom_right_top(bounds):
    x0, y0, w, h = bounds
    return (x0, y0, x0 + w, y0 + h)


def _add_inset(
    ax,
    xlim,
    position="top right",
    *,
    h_inset,  # cm
    w_inset,  # cm
    hoffset_inset,  # cm
    voffset_inset,  # cm
    extra_bottom_offset,  # cm; extra space for x-axis labels of parent
):
    bb = left_bottom_right_top(ax.get_position().bounds)
    fig = ax.get_figure()
    fig_width = fig.get_figwidth() / mpl.cm2inch
    fig_height = fig.get_figheight() / mpl.cm2inch
    pos = [  # top right
        bb[0] + hoffset_inset / fig_width,  # x
        bb[3] + voffset_inset / fig_height,  # y
        w_inset / fig_width,  # width
        h_inset / fig_height,  # height
    ]
    if "bottom" in position:
        # adjust x to be below parent axis
        pos[1] = (
            bb[1] - (extra_bottom_offset + voffset_inset + h_inset) / fig_width
        )
    if "left" in position:
        # adjust x to be left of right edge of parent axis
        pos[0] = bb[2] - (hoffset_inset + w_inset) / fig_width
    ax_inset = fig.add_axes(pos)
    ax_inset.set_xlim(xlim)

    # draw the inset indicator

    def yoffset(pt):
        """pt to axis fraction"""
        h = ax.get_position().bounds[-1]
        return (pt / 72.0) / (h * fig.get_figheight())

    if "top" in position:
        # arrow_start in inset axes coordinates
        arrow_start = (0.5, 0)
        # arrow_end in parent data/axes coordinates
        arrow_end = (0.5 * (xlim[0] + xlim[1]), 1 - yoffset(2))
    else:
        arrow_start = (0.5, 1)
        arrow_end = (0.5 * (xlim[0] + xlim[1]), yoffset(2))

    arrowprops = dict(
        arrowstyle="-",
        color="black",
        shrinkA=0,
        shrinkB=0,
        linewidth=0.5,
        connectionstyle="angle3,angleA=90,angleB=0",
    )
    xy_trans = blended_transform_factory(ax.transData, ax.transAxes)
    ax.annotate(
        "",
        xy=arrow_end,
        xycoords=xy_trans,
        xytext=arrow_start,
        textcoords=ax_inset.transAxes,
        arrowprops=arrowprops,
    )
    ax.plot(  # bullet as manual arrowhead
        *arrow_end,
        "o",
        color="black",
        transform=xy_trans,
        markersize=1.41,
        label="",
    )

    return ax_inset


def format_inset(ax, ymax=None):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    if ymax is None:
        ymax = np.ceil(y1)
        ymax_lbl = str(int(ymax))
    else:
        ymax_lbl = str(ymax)

    def x_lbl(v):
        return str(int(v)) if int(v) == v else str(v)

    mpl.set_axis(
        ax,
        "x",
        x0,
        x1,
        ticks=[x0, x1],
        ticklabels=[x_lbl(v) for v in (x0, x1)],
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

    mpl.set_axis(
        ax,
        "y",
        -ymax,
        ymax,
        ticks=[-ymax, 0, ymax],
        ticklabels=["", "0", ymax_lbl],
        tick_params=dict(
            length=2,
            width=0.25,
            direction="inout",
            labelsize="xx-small",
            right=True,
            pad=0.5,
        ),
        minor_tick_params=dict(length=0),
        label="",
    )

    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(0.25)


def render_expvals(
    ax,
    t,
    val,
    σ,
    scale=1,
    name="",
    summarize=True,
    clip_on=True,
    line_labels_at=None,
    xy_label_left=(0, -3),  # offset of line label (pt)
    xy_label_right=(0, 3),  # offset of line label (pt)
    for_inset=False,
):
    if summarize:
        n = len(val)
        val1 = val[1 : n // 2]
        val2 = val[n // 2 : -1]
        σ1 = σ[1 : n // 2]
        σ2 = σ[n // 2 : -1]
        print(f"Summary {name}".strip())
        print(f"  1st half max(|val|) = {max(abs(val1))}")
        print(f"  1st half max(|val|+|σ|) = {max(abs(val1) + abs(σ1))}")
        print(f"  2nd half max(|val|) = {max(abs(val2))}")
        print(f"  2nd half max(|val|+|σ|) = {max(abs(val2) + abs(σ2))}")
    val = scale * val
    σ = scale * σ
    (l_left,) = ax.plot(
        t, val, label="left", rasterized=RASTERIZED, clip_on=clip_on
    )
    ax.fill_between(
        t, val - σ, val + σ, alpha=0.2, label="", rasterized=RASTERIZED
    )
    (l_right,) = ax.plot(
        t, -val, label="right", rasterized=RASTERIZED, clip_on=clip_on
    )
    ax.fill_between(
        t, -val - σ, -val + σ, alpha=0.2, label="", rasterized=RASTERIZED
    )
    if line_labels_at is not None:
        x = line_labels_at
        y = interp1d(t, val)(x)
        left_color = l_left.get_color()
        right_color = l_right.get_color()
        ax.annotate(
            rf"${LEFT_LABEL}$",
            xy=(x, y),
            xytext=xy_label_left,
            ha="center",
            va=("top" if xy_label_left[1] <= 0 else "bottom"),
            textcoords="offset points",
            color=left_color,
            #bbox=dict(
                #facecolor="none",
                #edgecolor=left_color,
                #boxstyle="circle",
                #linewidth=0.5,
                #pad=0.1,
            #),
            size=("xx-small" if for_inset else "small"),
        )
        ax.annotate(
            rf"${RIGHT_LABEL}$",
            xy=(x, -y),
            xytext=xy_label_right,
            ha="center",
            va=("bottom" if xy_label_right[1] >= 0 else "top"),
            textcoords="offset points",
            color=right_color,
            #bbox=dict(
                #facecolor="none",
                #edgecolor=right_color,
                #boxstyle="circle",
                #linewidth=0.5,
                #pad=0.1,
            #),
            size=("xx-small" if for_inset else "small"),
        )


def render_control_field(
    ax,
    t,
    val,
    line_label_at=None,
    xy_label_left=(0, 2),  # offset of line label (pt)
    for_inset=False,
):
    label = r"$\omega(t)$" if SHOWING_GUESS else _OMEGA_OPT_OF_T
    ax.plot(t, val, lw=0.75, color="black", rasterized=RASTERIZED, ls="dotted")
    if line_label_at is not None:
        x = line_label_at
        y = interp1d(t, val)(x)
        ax.annotate(
            label,
            xy=(x, y),
            xytext=xy_label_left,
            ha="center",
            va=("top" if xy_label_left[1] <= 0 else "bottom"),
            textcoords="offset points",
            color="black",
            size=("xx-small" if for_inset else "small"),
        )


def get_minima(x, y, x0, x1):
    # Find indices of x within the range [x0, x1]
    indices = np.where((x >= x0) & (x <= x1))[0]
    if len(indices) == 0:
        return []
    # Find local minima of y within the selected x range
    minima_indices = []
    for i in range(1, len(indices) - 1):
        if (
            y[indices[i]] <= y[indices[i - 1]]
            and y[indices[i]] <= y[indices[i + 1]]
        ):
            minima_indices.append(i)
    # Return the positions of the minima
    return x[indices[minima_indices]]


def render_harmonic_period(ax, x, y, t_ho=0.66):  # 1.089 is *harmonic& period
    arrowprops = dict(
        arrowstyle="|-|",
        color="black",
        shrinkA=0,
        shrinkB=0,
        mutation_scale=1,
        linewidth=0.5,
    )
    ax.annotate(
        "",
        xy=(x, y),
        xycoords="data",
        xytext=(x + t_ho, y),
        textcoords="data",
        arrowprops=arrowprops,
    )
    ax.annotate(
        str(t_ho),
        xy=(x + 0.5 * t_ho, y),
        xytext=(0, 2),
        textcoords="offset points",
        size="xx-small",
        va="bottom",
        ha="center",
    )


def plot_dynamics(
    lab_data,
    moving_data,
    t_r,
    lab_data_theta=None,  # alternative data for "lab frame position"
    frame=6,
):
    if lab_data_theta is None:
        lab_data_theta = lab_data
    fig_width = COLUMN_WIDTH
    left_margin = 1.0
    right_margin = 0.45
    bottom_margin = 0.80
    top_margin = 0.65
    brokengap = 0.1
    h = 1.4  # default height of main axes (p, θ)
    vgap_right = 3.3  # cm between top of p-axes and θ-axes, moving frame
    vgap_left = 3.3  # cm between top of p-axes and θ-axes, lab frame
    hgap = 0.9

    inset_geometry = dict(
        h_inset=0.75,  # cm
        w_inset=1.0,  # cm
        hoffset_inset=0.0,  # cm
        voffset_inset=0.3,  # cm
        extra_bottom_offset=0.3,  # cm; extra space for x-axis labels of parent
    )
    add_inset = partial(_add_inset, **inset_geometry)

    h_lab_position = h  # inset_geometry["h_inset"]

    # vgap_left = vgap_right - (
    #     inset_geometry["h_inset"]
    #     + inset_geometry["voffset_inset"]
    #     + inset_geometry["extra_bottom_offset"]
    # )

    fig_height = (
        2 * h + bottom_margin + top_margin + max(vgap_left, vgap_right)
    )

    w = (fig_width - left_margin - right_margin - hgap) / 2
    ws = (w - brokengap) / 2

    fig = mpl.new_figure(fig_width, fig_height)

    axs = []

    # bottom left: lab frame momentum #########################################

    if frame >= 5:

        ax1 = fig.add_axes(
            [
                left_margin / fig_width,
                bottom_margin / fig_height,
                ws / fig_width,
                h / fig_height,
            ]
        )
        ax2 = fig.add_axes(
            [
                (left_margin + ws + brokengap) / fig_width,
                bottom_margin / fig_height,
                ws / fig_width,
                h / fig_height,
            ]
        )
        t = lab_data["time (ms)"]
        p = lab_data["p (Mπ/sec)"]
        σp = lab_data["σ_p (Mπ/sec)"]
        render_expvals(ax1, t, p, σp, name="p (lab)")
        ax1.axvline(x=(t[0] + t_r), lw=0.5, ls="--", color="black")
        y0 = -121
        y1 = 121
        mpl.set_axis(
            ax1,
            "y",
            range=(y0, y1),
            bounds=(y0, y1),
            show_opposite=False,
            ticks=[y0, -50, 0, 50, y1],
            position=("outward", 2),
            ticklabels=["", "", "0", "50", str(y1)],
            tick_params=dict(length=2, direction="inout"),
            minor_tick_params=dict(length=0),
            label="",
        )
        render_expvals(ax2, t, p, σp, line_labels_at=198.85, summarize=False)
        ax2.set_ylim(ax1.get_ylim())
        print(f"  minima in [198.65, 200.15]: {get_minima(t, p, 198.65, 200.15)}")
        ax2.axvline(x=(t[-1] - t_r), lw=0.5, ls="--", color="black")
        render_control_field(
            ax1,
            t,
            lab_data["ω (π/sec)"],
            xy_label_left=(3.5, 1),  # offset of line label (pt)
            line_label_at=None,
        )
        if SHOWING_GUESS:
            # minima in [198.65, 200.15]: [199.32 199.98]
            render_harmonic_period(ax2, 198.99, -20)  # adjusted for maxima
        render_control_field(ax2, t, lab_data["ω (π/sec)"])

        set_broken_time_axis(ax1, ax2)
        axs.extend([ax1, ax2])
        ax1.set_ylabel(r"$p$ ($M \pi$/s)")
        set_xlabel(ax1, "time (ms)", ax2=ax2)
        set_panel_title(
            ax1,
            voffset=(4 + 1.1 * 28.346),
            title="lab frame momentum",
        )

        ax_inset1 = add_inset(ax1, (0, 0.15), "top right")
        render_expvals(ax_inset1, t, p, σp, summarize=False)
        render_control_field(
            ax_inset1,
            t,
            lab_data["ω (π/sec)"],
            xy_label_left=(
                (2, 2) if SHOWING_GUESS else (5.5, 3.5)
            ),  # offset of line label (pt)
            line_label_at=(0.045 if SHOWING_GUESS else 0.075),
            for_inset=True,
        )
        format_inset(ax_inset1)

        if frame >= 6:
            ax_inset2 = add_inset(ax2, (200, 200.15), "top left")
            render_expvals(ax_inset2, t, p, σp, summarize=False)
            render_control_field(
                ax_inset2,
                t,
                lab_data["ω (π/sec)"],
                xy_label_left=(
                    (-2, -1) if SHOWING_GUESS else (-5.5, 3.5)
                ),  # offset of line label (pt)
                line_label_at=(200.045 if SHOWING_GUESS else 200.075),
                for_inset=True,
            )
            format_inset(ax_inset2)

    # top left: lab frame position ############################################

    ax = fig.add_axes(
        [
            left_margin / fig_width,
            (bottom_margin + h + vgap_left) / fig_height,
            w / fig_width,
            h_lab_position / fig_height,
        ]
    )
    t = lab_data_theta["time (ms)"]
    Δθ = lab_data_theta["Δθ (π)"]
    σθ = lab_data_theta["σ_θ (π)"]
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
        line_labels_at=170,
        xy_label_left=(0.5, -3),  # offset of line label (pt)
        xy_label_right=(0.5, 3),  # offset of line label (pt)
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
    mpl.set_axis(
        ax,
        "x",
        t[0],
        t[-1],
        range=(t[0], t[-1]),
        show_opposite=False,
        ticks=[t[0], t[-1]],
        position=("outward", 2),
        ticklabels=["0", "200.15"],
        tick_params=dict(length=2, direction="inout"),
        minor_tick_params=dict(length=0),
        label="",
    )
    set_panel_title(ax, title="lab frame position")
    axs.append(ax)

    if frame >= 2:
        ax_inset1 = add_inset(ax, (0, 0.15), "bottom right")
        render_expvals(
            ax_inset1,
            lab_data["time (ms)"],
            lab_data["Δθ (π)"],
            lab_data["σ_θ (π)"],
            summarize=False,
            line_labels_at=(None if SHOWING_GUESS else 0.075),
            xy_label_left=(2, -2),
            xy_label_right=(-2, 2),
            for_inset=True,
        )
        format_inset(ax_inset1, ymax=9e-3)

    # bottom right: moving frame momentum #####################################

    if frame >= 5:

        ax1 = fig.add_axes(
            [
                (left_margin + w + hgap) / fig_width,
                bottom_margin / fig_height,
                ws / fig_width,
                h / fig_height,
            ]
        )
        ax2 = fig.add_axes(
            [
                (left_margin + w + hgap + ws + brokengap) / fig_width,
                bottom_margin / fig_height,
                ws / fig_width,
                h / fig_height,
            ]
        )
        t = moving_data["time (ms)"]
        p = moving_data["p (Mπ/sec)"]
        σp = moving_data["σ_p (Mπ/sec)"]
        render_expvals(
            ax1,
            t,
            p,
            σp,
            name="p (moving)",
        )
        ax1.axvline(x=(t[0] + t_r), lw=0.5, ls="--", color="black")
        y0 = -82
        y1 = 82
        mpl.set_axis(
            ax1,
            "y",
            range=(-82, 82),
            bounds=(y0, y1),
            show_opposite=False,
            ticks=[y0, -23, 0, 23, y1],  # 23 is σ_p(0)
            position=("outward", 2),
            ticklabels=["", "", "0", "23", str(y1)],
            tick_params=dict(length=2, direction="inout"),
            minor_tick_params=dict(length=0),
            label="",
        )
        render_expvals(
            ax2,
            t,
            p,
            σp,
            line_labels_at=(199.40 if SHOWING_GUESS else None),
            summarize=False,
        )
        ax2.axvline(x=(t[-1] - t_r), lw=0.5, ls="--", color="black")
        set_broken_time_axis(ax1, ax2)
        set_xlabel(ax1, "time (ms)", ax2=ax2)
        axs.extend([ax1, ax2])
        set_panel_title(
            ax1,
            voffset=(4 + 1.1 * 28.346),
            title="moving frame momentum",
        )

        ax_inset1 = add_inset(ax1, (0, 0.15), "top right")
        render_expvals(
            ax_inset1,
            t,
            p,
            σp,
            summarize=False,
            line_labels_at=(None if SHOWING_GUESS else 0.06),
            xy_label_left=(2, 2),
            xy_label_right=(2, -2),
            for_inset=True,
        )
        format_inset(ax_inset1)

        if frame >= 6:
            ax_inset2 = add_inset(ax2, (200, 200.15), "top left")
            render_expvals(
                ax_inset2,
                t,
                p,
                σp,
                summarize=False,
                line_labels_at=(None if SHOWING_GUESS else 200.06),
                xy_label_left=(2, -2),
                xy_label_right=(2, 2),
                for_inset=True,
            )
            format_inset(ax_inset2)

    # top right: moving frame position ########################################

    if frame >= 3:

        ax1 = fig.add_axes(
            [
                (left_margin + w + hgap) / fig_width,
                (bottom_margin + h + vgap_right) / fig_height,
                ws / fig_width,
                h / fig_height,
            ]
        )
        ax2 = fig.add_axes(
            [
                (left_margin + w + hgap + ws + brokengap) / fig_width,
                (bottom_margin + h + vgap_right) / fig_height,
                ws / fig_width,
                h / fig_height,
            ]
        )
        t = moving_data["time (ms)"]
        Δθ = moving_data["Δθ (π)"]
        σθ = moving_data["σ_θ (π)"]
        render_expvals(
            ax1,
            t,
            Δθ,
            σθ,
            name="Δθ (moving)",
            scale=1000,
        )
        ax1.axvline(x=(t[0] + t_r), lw=0.5, ls="--", color="black")
        set_yaxis_scale_label(ax1, r"$\times\!10^{-3}$")
        y0 = -9.0
        y1 = 9.0
        mpl.set_axis(
            ax1,
            "y",
            range=(y0, y1),
            bounds=(y0, y1),
            show_opposite=False,
            ticks=[y0, -2.5, 0, 2.5, y1],  # 1.9 is width of initial wavepacket
            position=("outward", 2),
            ticklabels=["", "", "0", "2.5", "9.0"],
            tick_params=dict(length=2, direction="inout"),
            minor_tick_params=dict(length=0),
            label=r"",
            # label=r"$\Delta\theta$ ($10^{-3} \pi$)",
        )
        ax1.annotate(
            r"$t_r =$ 150 μs",
            xy=(t_r, ax1.get_ylim()[1]),
            xycoords="data",
            xytext=(1, 2),
            textcoords="offset points",
            ha="left",
            va="top",
            size="small",
        )
        render_expvals(
            ax2,
            t,
            Δθ,
            σθ,
            line_labels_at=(198.92 if SHOWING_GUESS else None),
            summarize=False,
            scale=1000,
        )
        ax2.axvline(x=(t[-1] - t_r), lw=0.5, ls="--", color="black")
        set_broken_time_axis(ax1, ax2)
        if SHOWING_GUESS and frame >= 4:
            render_harmonic_period(ax2, 199.155, 2.5)  # adjusted for maxima
        # set_title(ax1, "moving frame", ax2=ax2)
        set_panel_title(ax1, title="moving frame position")
        axs.extend([ax1, ax2])

        ax_inset1 = add_inset(ax1, (0, 0.15), "bottom right")
        render_expvals(
            ax_inset1,
            t,
            Δθ,
            σθ,
            summarize=False,
            scale=1000,
            line_labels_at=(None if SHOWING_GUESS else 0.075),
            xy_label_left=(2, -2),
            xy_label_right=(-2, 2),
            for_inset=True,
        )
        format_inset(ax_inset1)

        if frame >= 6:
            ax_inset2 = add_inset(ax2, (200, 200.15), "bottom left")
            render_expvals(
                ax_inset2,
                t,
                Δθ,
                σθ,
                summarize=False,
                scale=1000,
                line_labels_at=(None if SHOWING_GUESS else 200.075),
                xy_label_left=(-2, +2),
                xy_label_right=(2, -2),
                for_inset=True,
            )
            format_inset(ax_inset2)

    ###########################################################################

    return fig


def main():
    """Produce an output PDF file (same name as script)."""
    scriptfile = os.path.abspath(__file__)
    datadir = Path(os.path.splitext(scriptfile)[0])
    outfile = Path(os.path.splitext(scriptfile)[0] + ".pdf")
    lab_data = read_csv(datadir / "dynamics_guess_lab.csv")
    lab_data_theta = read_csv(datadir / "dynamics_guess_lab_theta.csv")
    moving_data = read_csv(datadir / "dynamics_guess_moving.csv")
    fig = plot_dynamics(
        lab_data,
        moving_data,
        t_r=0.15,
        lab_data_theta=lab_data_theta,
    )
    # fig.suptitle("unoptimized nonadiabatic dynamics", y=1.0)
    fig.savefig(outfile, transparent=True, dpi=600)
    for frame in [1, 2, 3, 4, 5, 6]:
        fig = plot_dynamics(
            lab_data,
            moving_data,
            t_r=0.15,
            lab_data_theta=lab_data_theta,
            frame=frame
        )
        outfile = Path(os.path.splitext(scriptfile)[0] + f"_{frame}.pdf")
        fig.savefig(outfile, transparent=True, dpi=600)


if __name__ == "__main__":
    main()
