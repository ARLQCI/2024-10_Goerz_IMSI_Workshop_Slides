import os
import mpl
import math
from pathlib import Path

import numpy as np
from scipy.special import genlaguerre
import matplotlib
from matplotlib.patches import FancyArrowPatch


def set_panel_label(
      ax, label, fmt="({label})", hoffset=0, voffset=7, title=None
  ):
    """Set a panel label."""
    text = fmt.format(label=label)
    if title is not None:
        text += f" {title}"
        #text = title  # disable labels
    ax.annotate(
        text,
        xy=(0, 1),
        xycoords="axes fraction",
        ha="left",
        va="baseline",
        xytext=(-hoffset, voffset),
        textcoords="offset points",
    )



COLUMN_WIDTH = 7.8  # cm

TAI_RADIUS = 25.46  # μm

#MAP_RES = 100  # use a low resolution for a quick preview
MAP_RES = 500

# TCBCOLBACK = (0.953,0.953,0.953)
TCBCOLBACK = (1.0, 1.0, 1.0)


def zR(w0, omega):
    c = 3e8
    return omega * w0**2 / 2 / c


def w(z, w0, omega):
    return w0 * np.sqrt(1 + (z / zR(w0, omega)) ** 2)


def R(z, w0, omega):
    epsilon = 1e-15  # small regulator to avoid z=0
    return z * (1 + (zR(w0, omega) / (z + epsilon)) ** 2)


def GPhi(z, l, w0, omega):
    return (np.abs(l) + 1) * np.arctan(z / zR(w0, omega))


def E_LG(rho, theta, z, p=0, l=20, w0=10e-6, P=1e-3, omega=2.371e15, t=0):
    c = 3e8
    wst = w(z, w0, omega)
    return (
        np.sqrt(
            (2 * math.factorial(p))
            / (np.pi * math.factorial(p + np.abs(l)))
        )
        * np.sqrt(P / wst**2)
        * np.exp(-(rho**2) / wst**2)
        * ((np.sqrt(2) * rho) / wst) ** np.abs(l)
        * genlaguerre(p, l, 2 * rho**2 / wst**2)
        * np.exp(-1j * l * theta)
        * np.exp(-1j * omega * t)
        * np.exp(1j * omega / c * (z - rho**2 / (2 * R(z, w0, omega))))
        * np.exp(1j * GPhi(z, l, w0, omega))
    )


def LG_int(
    rho,
    theta,
    z,
    l1=20,
    l2=28,
    eta=1.67,
    w1=5.785e-6,
    omega=2.371e15,
    domega=2 * np.pi,
    t=0,
    P1=1e-3,
):
    MHz = 1 / 12345.1378229937
    # MHz unit is chosen so that max potential at TAI_RADIUS is 2.2MHz

    V = np.array([0.0])
    if abs(z) < 1e-15:
        z = 1e-15
    if abs(rho) > 1e-15:
        E1 = E_LG(rho, theta, z, 0, l1, w1, P1, omega, t)
        E2 = E_LG(
            rho,
            theta,
            z,
            0,
            l2,
            w1 * eta,
            P1 * eta**2 * np.sqrt(l2 / l1),
            omega + domega,
            t,
        )
        V = np.abs(E1 + E2) ** 2 * MHz
    return V


def find_z_maxima(f, z0, z1, r0, r1, n_samples=MAP_RES):
    """Given a function f(z, r), find maxima."""
    z_vals = np.linspace(z0, z1, n_samples)
    r_vals = np.linspace(r0, r1, n_samples)
    maxima = []
    for z in z_vals:
        f_over_r = np.array([f(z, r) for r in r_vals])
        maxima.append(get_maxima(r_vals, f_over_r, r0, r1))
    return z_vals, np.array(maxima)


def render_LG_maxima(ax):

    def f(z, r):
        return LG_int(rho=r*1e-6, z=z*1e-6, theta=np.pi/8)[0]

    z_vals, maxima = find_z_maxima(f, -150, 150, 10, 50, n_samples=MAP_RES)
    assert maxima.shape == (MAP_RES, 2), maxima.shape
    ax.plot(maxima[:, 0], z_vals, lw=0.72, color="orange")
    ax.plot(maxima[:, 1], z_vals, lw=0.72, color="orange")


def render_vertical_potential(ax, ax_cbar):
    r0 = 0
    r1 = 60
    R_max1 = 18.31831831831832  # maxima ...
    R_max2 = 36.15615615615616  # ... in panel (c)
    rs = np.linspace(r0, r1, MAP_RES)
    zs = np.linspace(-150, 150, MAP_RES)
    θ = np.pi / 8

    LG_int_profile = np.zeros((rs.size, zs.size), dtype=float)
    for i in range(zs.size):
        for j in range(rs.size):
            LG_int_profile[i, j] = LG_int(rs[j] * 1e-6, θ, zs[i] * 1e-6)

    print(f"Max val in vertical_potential: {np.max(LG_int_profile)}")

    lg_int_profile_log = np.log10(LG_int_profile)
    log_min = -2  # Minimum log scale value (1e-1)
    log_max = np.log10(137)

    ax.set_facecolor("black")
    im = ax.imshow(
        lg_int_profile_log,
        cmap="cubehelix",
        extent=[rs.min(), rs.max(), zs.min(), zs.max()],
        aspect="auto",
        origin="lower",
        rasterized=True,
        vmin=log_min,
        vmax=log_max,
    )

    ax.axvline(x=R_max1, lw=0.5, ls="--", color=(0, 0, 0, 0.5))
    ax.axvline(x=TAI_RADIUS, lw=0.5, ls="--", color=(0, 0, 0, 0.5))
    ax.axvline(x=R_max2, lw=0.5, ls="--", color=(0, 0, 0, 0.5))
    ax.axhline(y=0, lw=0.5, ls="--", color=(0, 0, 0, 0.5))

    ax.tick_params(
        axis="both", direction="out", which="both", right=False, top=True
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction="out",
        length=1.5,
        right=False,
        top=True,
    )
    ax.set_yticklabels(["", "-100", "0", "100"])
    ax.set_xticklabels(["0", "20", "40", "60"])
    ax.set_xlabel(r"$r$ (μm)", labelpad=0)
    ax.set_ylabel(r"$z$ (μm)", labelpad=-6)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    fig = ax.get_figure()
    cbar = fig.colorbar(im, ax=ax, cax=ax_cbar, extend="min")
    ax_cbar.tick_params(
        axis="both", which="minor", direction="out", length=1.5
    )
    ax_cbar.tick_params(axis="both", direction="out", which="both")
    ax_cbar.set_yticks([-2, -1, 0, 1, 2])
    ax_cbar.set_yticks(
        np.concatenate(
            (
                np.log10(np.linspace(0.01, 0.1, 10)),
                np.log10(np.linspace(0.1, 1, 10)),
                np.log10(np.linspace(1, 10, 10)),
                np.log10(np.linspace(10, 100, 10)),
            )
        ),
        minor=True,
    )
    ax_cbar.set_yticklabels(
        [r"10⁻²", r"10⁻¹", r"10⁰", r"10¹", "10²"]
    )
    ax_cbar.set_title(r"MHz ", loc="left", ha="left", fontsize="small", pad=2)
    for spine in ax_cbar.spines.values():
        spine.set_linewidth(0.5)


def render_polar_potential(ax):
    thetas = np.linspace(0, 2 * np.pi, MAP_RES)
    r0 = 18.31831831831832  # maxima ...
    r1 = 36.15615615615616  # ... in panel (c)
    rmax = 45  # how far the radial axis-spine extends
    rs = np.linspace(r0, r1, MAP_RES)

    LG_int_profile = np.zeros((rs.size, thetas.size), dtype=float)
    for i in range(thetas.size):
        for j in range(rs.size):
            LG_int_profile[i, j] = LG_int(
                rs[j] * 1e-6, thetas[i], 1e-15
            )  # z=0

    print(f"Max val in polar_potential: {np.max(LG_int_profile)}")

    lg_int_profile_log = np.log10(LG_int_profile)
    log_min = -2  # Minimum log scale value (1e-1)
    log_max = np.log10(137)

    ax.pcolormesh(
        thetas,
        rs,
        lg_int_profile_log.transpose(),
        cmap="cubehelix",
        rasterized=True,
        vmin=log_min,
        vmax=log_max,
    )

    # manual axis-spines and ticks

    # r-spine
    spines_style = "Simple, tail_width=0.5, head_width=2, head_length=2.5"
    spines_kw = dict(
        arrowstyle=spines_style,
        facecolor="black",
        edgecolor="black",
        linewidth=0,
        shrinkA=0,
        shrinkB=0,
        clip_on=False,
        zorder=100,
    )
    tick_box = dict(
        boxstyle="round,pad=0.05,rounding_size=0.3",
        facecolor=TCBCOLBACK,
        edgecolor="none",
        pad=0,
    )
    r_axis_spine = FancyArrowPatch((0, 0), (0, rmax), **spines_kw)
    ax.add_patch(r_axis_spine)
    # θ-spine
    theta_axis_spine = FancyArrowPatch(
        (0, r1 + 0.3 * (rmax - r1)),
        (0.8 * np.pi / 4, r1 + 0.3 * (rmax - r1)),
        connectionstyle="arc3,rad=0.16",
        **spines_kw,
    )
    ax.add_patch(theta_axis_spine)
    ax.annotate(
        r"$\theta$",
        xy=(0.8 * np.pi / 4, r1 + 0.3 * (rmax - r1)),
        xytext=(2, 0),
        ha="left",
        va="bottom",
        textcoords="offset points",
        fontsize="small",
        bbox=dict(alpha=0.5, **tick_box),
        annotation_clip=False,
    )
    # θ=π/8 marker
    ax.plot(
        np.array([0, np.pi / 8]),
        np.array([0, rmax]),
        color="black",
        ls="--",
        linewidth=0.5,
        clip_on=False,
    )
    ax.scatter(
        [np.pi / 8],
        [r1 + 0.3 * (rmax - r1)],
        s=1.5,
        color="black",
        clip_on=False,
    )
    ax.annotate(
        #r"$\pi \kern-0.8pt / \kern-0.5pt 8$",
        r"π/8",
        xy=(np.pi / 8, r1 + 0.3 * (rmax - r1)),
        xytext=(2.0, -2.0),
        ha="left",
        va="center",
        textcoords="offset points",
        fontsize="x-small",
        bbox=dict(alpha=0.5, **tick_box),
        annotation_clip=False,
    )
    # r-ticks
    ax.annotate(
        str(round(r0)),
        xy=(0, r0),
        xytext=(0, -5),
        ha="right",
        va="center",
        textcoords="offset points",
        fontsize="x-small",
        bbox=tick_box,
        annotation_clip=False,
    )
    ax.annotate(
        r"$R$",
        xy=(0, TAI_RADIUS),
        xytext=(0, -5),
        ha="center",
        va="center",
        textcoords="offset points",
        fontsize="x-small",
        bbox=dict(alpha=0.5, **tick_box),
        annotation_clip=False,
    )
    ax.annotate(
        str(round(r1)),
        xy=(0, r1),
        xytext=(0, -5),
        ha="center",
        va="center",
        textcoords="offset points",
        fontsize="x-small",
        bbox=tick_box,
        annotation_clip=False,
    )
    ax.annotate(
        r"$r$",
        xy=(0, rmax),
        xytext=(-3, 2),
        ha="left",
        va="bottom",
        textcoords="offset points",
        fontsize="small",
        bbox=tick_box,
        annotation_clip=False,
    )
    ax.annotate(
        r"μm",
        xy=(0, rmax),
        xytext=(-4, -6),
        ha="left",
        va="center",
        textcoords="offset points",
        fontsize="x-small",
        bbox=tick_box,
        annotation_clip=False,
    )
    ax.scatter(
        [0, 0], [TAI_RADIUS, r1], marker="|", linewidth=0.5, color="black"
    )  # Ticks
    # ax.set_rticks([TAI_RADIUS, ])  # Less radial ticks
    # ax.set_rlabel_position(0)

    ax.set_yticks([])
    # ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_rlim(r0, r1)
    ax.set_rorigin(0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.grid(True)


def render_inset(ax, data, bounds, ranges, xticks=None, yticks=None):
    """Put an inset on the given axes as axes-position `bounds`, showing data
    in the region specified by `ranges`.
    """
    rs, vals, vals0 = data
    X0, Y0, width, height = bounds
    x0, x1, y0, y1 = ranges
    axins = ax.inset_axes([X0, Y0, width, height])

    axins.plot(rs, vals, clip_on=True)
    axins.plot(rs, vals0, clip_on=True, dashes=[2, 2])

    axins.set_xlim(x0, x1)
    axins.set_ylim(y0, y1)
    axins.set_xticks([])
    if xticks is not None:
        mpl.set_axis(
            axins,
            "x",
            x0,
            x1,
            ticks=xticks,
            ticklabels=[str(v) for v in xticks],
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
    axins.set_yticks([])
    if yticks is not None:
        mpl.set_axis(
            axins,
            "y",
            y0,
            y1,
            ticks=yticks,
            ticklabels=[str(v) for v in yticks],
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
        bbox = dict(boxstyle="round", color=TCBCOLBACK, alpha=0.9, pad=0)
        matplotlib.pyplot.setp(axins.get_yticklabels(), bbox=bbox)
    for axis in ["top", "bottom", "left", "right"]:
        axins.spines[axis].set_linewidth(0.25)
    rp, lps = ax.indicate_inset_zoom(
        axins, edgecolor="black", lw=0.25, alpha=1.0, clip_on=False
    )
    for lp in lps:
        lp.set(linewidth=0.25)
        lp.set_zorder(0)
    axins.patch.set_alpha(0.0)
    return axins


def get_maxima(x, y, x0, x1):
    # Find indices of x within the range [x0, x1]
    indices = np.where((x >= x0) & (x <= x1))[0]
    if len(indices) == 0:
        return []
    # Find local amxima of y within the selected x range
    maxima_indices = []
    for i in range(1, len(indices) - 1):
        if (
            y[indices[i]] >= y[indices[i - 1]]
            and y[indices[i]] >= y[indices[i + 1]]
        ):
            maxima_indices.append(i)
    # Return the positions of the maxima_indices
    return x[indices[maxima_indices]]


def render_radial_potential(ax, extend=1.0):
    r0 = 0
    r1 = 60
    r1_ext = r1 * extend  # account for 2D panel being wider than 3D panel
    rs = np.linspace(r0, r1, 1000)
    θ = np.pi / 8
    z = 1e-15
    vals = np.array([LG_int(r * 1e-6, θ, z) for r in rs])
    R_max1 = get_maxima(rs, vals, 0, TAI_RADIUS)[0]
    R_max2 = get_maxima(rs, vals, TAI_RADIUS, r1)[0]
    print(f"Radial maxima at {R_max1}, {R_max2}")
    ax.plot(
        rs,
        vals,
        clip_on=False,
        #label=r"$\theta \kern-0.8pt=\kern-0.8pt \pi\kern-0.8pt / \kern-0.5pt 8$",
        label=r"$\theta=\pi/8$",
    )
    vals0 = np.array([LG_int(r * 1e-6, 0.0, z) for r in rs])
    ax.plot(
        rs,
        vals0,
        clip_on=False,
        dashes=[2, 2],
        #label=r"$\theta \kern-0.8pt=\kern-0.8pt 0",
        label=r"$\theta=0$",
    )
    ax.axvline(x=R_max1, lw=0.5, ls="--", color="black")
    ax.annotate(
        r"LG$_{20}$",
        xy=(R_max1, 137),
        xytext=(0, 1),
        textcoords="offset points",
        va="bottom",
        ha="center",
        fontsize="xx-small",
    )
    ax.axvline(x=TAI_RADIUS, lw=0.5, ls="--", color="black")
    ax.axvline(x=R_max2, lw=0.5, ls="--", color="black")
    ax.annotate(
        r"LG$_{28}$",
        xy=(R_max2, 137),
        xytext=(0, 1),
        textcoords="offset points",
        va="bottom",
        ha="center",
        fontsize="xx-small",
    )
    xticks = [0, R_max1, TAI_RADIUS, R_max2, r1]
    xtick_labels = [
        "0",
        str(round(R_max1)),
        "$R$",
        str(round(R_max2)),
        str(int(r1)),
    ]
    mpl.set_axis(
        ax,
        "y",
        range=(0, 137),
        bounds=(0, 137),
        show_opposite=False,
        ticks=[0, 137],
        position=("outward", 2),
        ticklabels=["0", "137"],
        tick_params=dict(length=2, direction="inout"),
        minor_tick_params=dict(length=0),
        labelpad=0,
        label=r"$V$ (MHz)",
        lw=0.5,
    )
    ax.tick_params(right=False)
    mpl.set_axis(
        ax,
        "x",
        range=(r0, r1_ext),
        bounds=(r0, r1),
        show_opposite=False,
        ticks=[float(v) for v in xticks],
        position=("outward", 2),
        ticklabels=xtick_labels,
        tick_params=dict(length=2, direction="inout"),
        minor_tick_params=dict(length=0),
        label=r"$r$ (μm)",
        labelpad=0,
        lw=0.5,
    )

    axins = render_inset(
        ax,
        (rs, vals, vals0),
        [0.68, 0.6, 0.3, 0.5],
        [TAI_RADIUS - 2, TAI_RADIUS + 2, 0, 4.4],
        yticks=[0, 2.2, 4.4],
        xticks=[TAI_RADIUS],
    )
    axins.set_xticklabels([""])
    axins.set_yticklabels(["0", "2.2", ""])
    axins.axvline(x=TAI_RADIUS, lw=0.25, ls="--", color="black")
    axins.axhline(y=2.2, lw=0.25, ls="--", color="black")

    axins.annotate(
        "",
        xy=(0, 1.15),
        xycoords="axes fraction",
        xytext=(1, 1.15),
        textcoords="axes fraction",
        arrowprops=dict(
            arrowstyle="|-|", mutation_scale=1, lw=0.25, shrinkA=0, shrinkB=0
        ),
    )
    axins.annotate(
        r"4μm",
        xy=(0.5, 1.15),
        xycoords="axes fraction",
        fontsize="xx-small",
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.15,rounding_size=0.3",
            facecolor=TCBCOLBACK,
            edgecolor="none",
            pad=0,
        ),
    )

    legend = ax.legend(
        loc=(0.62, 0.22),
        handlelength=1.8,
        frameon=True,
        framealpha=0.9,
        borderpad=0,
        edgecolor="none",
        facecolor=TCBCOLBACK,
        fontsize="xx-small",
    )


def nicefrac(a, b, fontsize=6):
    """The dirtiest hack to get a nicefrac without loading a package."""
    return "".join(
        [
            r"{\fontsize{FS}{FS}\selectfont".replace("FS", str(fontsize)),
            r"\raisebox{0.5ex}{\kern-0.1em A}".replace("A", str(a)),
            #r"\kern-0.2em/\kern-0.1em",
            r"\raisebox{-0.5ex}{B}".replace("B", str(b)),
            r"}",
        ]
    )


def render_angular_potential(ax):
    thetas = np.linspace(0, 2 * np.pi, 1000)
    vals = np.array(
        [LG_int(TAI_RADIUS * 1e-6, theta, 1e-9) for theta in thetas]
    )
    ax.plot(thetas / np.pi, vals, clip_on=False)
    mpl.set_axis(
        ax,
        "y",
        range=(0, 2.2),
        bounds=(0, 2.2),
        show_opposite=False,
        ticks=[0, 2.2],
        position=("outward", 2),
        ticklabels=["0", "2.2"],
        tick_params=dict(length=2, direction="inout"),
        minor_tick_params=dict(length=0),
        labelpad=-2,
        label=r"$V$ (MHz)",
    )
    ax.tick_params(right=False)
    xticks = [0, 2]
    #xtick_labels = ["0", nicefrac(r"π", 4), r"2π"]
    xtick_labels = ["0", r"2π"]
    mpl.set_axis(
        ax,
        "x",
        range=(0, 2),
        bounds=(0, 2),
        show_opposite=False,
        ticks=[float(v) for v in xticks],
        position=("outward", 2),
        ticklabels=xtick_labels,
        tick_params=dict(length=2, direction="inout"),
        minor_tick_params=dict(length=0),
        label=r"$\theta$",
        labelpad=0,
        lw=0.5,
    )


def plot_pinwheel():
    """Plot the figure."""
    fig_width = COLUMN_WIDTH

    left_margin = 1.0
    right_margin = 0.2
    top_margin = 0.75
    bottom_margin = 0.9

    h = 1.6  # height of 2D plots
    h_3D = 2  # height of 3D plots
    vgap = 1.6  # vertical gap between 2D and 3D plots
    hgap = 1.0  # horizontal gap

    w_cbar = 0.25  # width of colorbar
    gap_cbar = 0.25  # gap between 3D panel and colorbar

    w = (fig_width - left_margin - right_margin - hgap) / 2.0

    w_3D = w - w_cbar - gap_cbar
    h_3D = w_3D

    fig_height = bottom_margin + h + vgap + h_3D + top_margin

    fig = mpl.new_figure(fig_width, fig_height)

    # top left: vertical potential (3D) #######################################

    ax = fig.add_axes(
        [
            left_margin / fig_width,
            (bottom_margin + h + vgap) / fig_height,
            w_3D / fig_width,
            h_3D / fig_height,
        ]
    )

    ax_cbar = fig.add_axes(
        [
            (left_margin + w_3D + gap_cbar) / fig_width,
            (bottom_margin + h + vgap) / fig_height,
            w_cbar / fig_width,
            h_3D / fig_height,
        ]
    )

    render_vertical_potential(ax, ax_cbar)
    set_panel_label(ax, "a", title=r"V(r, z; θ=π/8)", voffset=10)

    # top right: polar potential (3D) #########################################

    ax = fig.add_axes(
        [
            (left_margin + hgap + w) / fig_width,
            (bottom_margin + h + vgap) / fig_height,
            w_3D / fig_width,
            h_3D / fig_height,
        ],
        projection="polar",
    )
    set_panel_label(ax, "b", title=r"V(r, θ; z=0)", voffset=10)

    render_polar_potential(ax)

    # bottom left: radial potential (2D) ######################################

    ax = fig.add_axes(
        [
            left_margin / fig_width,
            bottom_margin / fig_height,
            w / fig_width,
            h / fig_height,
        ]
    )

    render_radial_potential(ax, extend=w / w_3D)
    set_panel_label(ax, "c", title=r"V(r; θ, z=0)", voffset=10)

    # bottom right: angular potential (2D) ####################################

    ax = fig.add_axes(
        [
            (left_margin + hgap + w) / fig_width,
            bottom_margin / fig_height,
            w / fig_width,
            h / fig_height,
        ]
    )

    render_angular_potential(ax)
    set_panel_label(ax, "d", title=r"V(θ; r=R, z=0)", voffset=10)

    return fig


def main():
    """Produce an output PDF file (same name as script)."""
    scriptfile = os.path.abspath(__file__)
    # datadir = Path(os.path.splitext(scriptfile)[0])
    outfile = Path(os.path.splitext(scriptfile)[0] + ".pdf")
    fig = plot_pinwheel()
    fig.savefig(outfile, transparent=True, dpi=600)


if __name__ == "__main__":
    main()
