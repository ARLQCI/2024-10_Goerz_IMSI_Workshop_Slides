from pathlib import Path
import csv
import mpl
import numpy as np
import os


def plot_response_spectrum(
    response_data_guess,
    spectrum_data_guess,
    response_data_opt,
    spectrum_data_opt,
    response_xcol="τ_vals",
    spectrum_xcol="freq (MHz)",
    ycols=["μ=1.0", "μ=0.9", "μ=0.8", "μ=0.5"],
    frame=0,
):
    fig_width = 15
    fig_height = 7.5
    left_margin = 1.0
    right_margin = 0.25
    top_margin = 0.5
    hgap = 1.25
    vgap = 0.5
    bottom_margin = 0.75

    if frame == 0:
        frame = 2 * len(ycols)

    w = (fig_width - left_margin - right_margin - hgap) / 2.0
    h = (fig_height - bottom_margin - top_margin - vgap) / 2.0

    fig = mpl.new_figure(fig_width, fig_height)

    # top left: response to guess pulse

    ax = fig.add_axes(
        [
            left_margin / fig_width,
            (bottom_margin + h + vgap) / fig_height,
            w / fig_width,
            h / fig_height,
        ]
    )

    data = response_data_guess
    for (i, ycol) in enumerate(ycols):
        if (i + 1) <= frame:
            ax.plot(
                data[response_xcol] / 1000.0,
                data[ycol],
                label=ycol,
                lw=(1.5 if i == 0 else (1.0 / np.sqrt(i))),
            )
    mpl.set_axis(
        ax,
        "y",
        range=(0, 1),
        ticks=[0, 0.25, 0.5, 0.75, 1.0],
        tick_params=dict(length=2, direction="inout"),
        minor_tick_params=dict(length=0),
        label="final time population in |0⟩",
    )
    mpl.set_axis(
        ax,
        "x",
        tick_params=dict(length=2, direction="inout"),
        minor_tick_params=dict(length=0),
        label=("time of flight τ (ms)" if frame <= len(ycols) else ""),
    )
    ax.set_title("population response", fontsize=8)

    # top right: spectrum for guess pulse response

    ax = fig.add_axes(
        [
            (left_margin + w + hgap) / fig_width,
            (bottom_margin + h + vgap) / fig_height,
            w / fig_width,
            h / fig_height,
        ]
    )

    data = spectrum_data_guess
    for (i, ycol) in enumerate(ycols):
        if (i + 1) <= frame:
            ax.plot(
                data[spectrum_xcol][1:],
                data[ycol][1:],
                label=ycol,
                lw=(1.5 if i == 0 else (1.0 / np.sqrt(i))),
            )
    mpl.set_axis(
        ax,
        "y",
        # range=(0, 0.24),
        tick_params=dict(length=2, direction="inout"),
        minor_tick_params=dict(length=0),
        label="spectrum (arb. units)",
    )
    mpl.set_axis(
        ax,
        "x",
        range=(0, 5.5),
        tick_params=dict(length=2, direction="inout"),
        minor_tick_params=dict(length=0),
        label=("frequency (MHz)" if frame <= len(ycols) else ""),
    )
    ax.set_title("spectrum of population response", fontsize=8)
    ax.annotate(
        "unoptimized",
        xy=(0.5, 1.0),
        xycoords='axes fraction',
        xytext=(0, -3),
        textcoords='offset points',
        ha='center',
        va='top',
        fontsize=8,
    )

    ax.legend(
        loc='center',
        bbox_to_anchor=(0.5, 0.5),
        ncols=2,
    )

    if frame > len(ycols):

        # bottom left: response to opt pulse

        ax = fig.add_axes(
            [
                left_margin / fig_width,
                bottom_margin / fig_height,
                w / fig_width,
                h / fig_height,
            ]
        )

        data = response_data_opt
        for (i, ycol) in enumerate(ycols):
            if (i + len(ycols) + 1) <= frame:
                ax.plot(
                    data[response_xcol] / 1000.0,
                    data[ycol],
                    label=ycol,
                    lw=(1.5 if i == 0 else (1.0 / np.sqrt(i))),
                )
        mpl.set_axis(
            ax,
            "y",
            range=(0, 1),
            ticks=[0, 0.25, 0.5, 0.75, 1.0],
            tick_params=dict(length=2, direction="inout"),
            minor_tick_params=dict(length=0),
            label="final time population in |0⟩",
        )
        mpl.set_axis(
            ax,
            "x",
            tick_params=dict(length=2, direction="inout"),
            minor_tick_params=dict(length=0),
            label="time of flight τ (ms)",
        )

        # bottom right: spectrum for opt pulse response

        ax = fig.add_axes(
            [
                (left_margin + w + hgap) / fig_width,
                bottom_margin / fig_height,
                w / fig_width,
                h / fig_height,
            ]
        )

        data = spectrum_data_opt
        for (i, ycol) in enumerate(ycols):
            if (i + len(ycols) + 1) <= frame:
                ax.plot(
                    data[spectrum_xcol][1:],
                    data[ycol][1:],
                    label=ycol,
                    lw=(1.5 if i == 0 else (1.0 / np.sqrt(i))),
                )
        mpl.set_axis(
            ax,
            "y",
            # range=(0, 0.24),
            tick_params=dict(length=2, direction="inout"),
            minor_tick_params=dict(length=0),
            label="spectrum (arb. units)",
        )
        mpl.set_axis(
            ax,
            "x",
            range=(0, 5.5),
            tick_params=dict(length=2, direction="inout"),
            minor_tick_params=dict(length=0),
            label="frequency (MHz)",
        )
        ax.annotate(
            "optimized",
            xy=(0.5, 1.0),
            xycoords='axes fraction',
            xytext=(0, -3),
            textcoords='offset points',
            ha='center',
            va='top',
            fontsize=8,
        )

    return fig


DATADIR = Path(".") / "2024-10-26_PresentationFigsData"


def read_csv(filename):
    data_dict = {}
    with open(filename, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for fieldname in csv_reader.fieldnames:
            data_dict[fieldname] = []
        for row in csv_reader:
            # Append the data to the corresponding list in the dictionary
            for fieldname in csv_reader.fieldnames:
                data_dict[fieldname].append(float(row[fieldname]))
        for fieldname in data_dict:
            data_dict[fieldname] = np.array(data_dict[fieldname])
    return data_dict


def main():
    response_data_guess = read_csv(DATADIR / "guess_responses.csv")
    spectrum_data_guess = read_csv(DATADIR / "guess_spectra.csv")
    response_data_opt = read_csv(DATADIR / "opt_responses.csv")
    spectrum_data_opt = read_csv(DATADIR / "opt_spectra.csv")
    fig = plot_response_spectrum(
        response_data_guess,
        spectrum_data_guess,
        response_data_opt,
        spectrum_data_opt,
    )
    scriptfile = os.path.abspath(__file__)
    outfile = Path(os.path.splitext(scriptfile)[0] + ".pdf")
    fig.savefig(outfile, transparent=True, dpi=600)

    for _frame in range(8):
        frame = _frame + 1
        fig = plot_response_spectrum(
            response_data_guess,
            spectrum_data_guess,
            response_data_opt,
            spectrum_data_opt,
            frame=frame,
        )
        outfile = Path(os.path.splitext(scriptfile)[0] + f"_{frame}.pdf")
        fig.savefig(outfile, transparent=True, dpi=600)


if __name__ == "__main__":
    main()
