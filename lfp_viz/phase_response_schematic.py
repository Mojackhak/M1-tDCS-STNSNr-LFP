"""Draw a compact, editable schematic of the six Phase response classes."""

from pathlib import Path
import argparse
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

WIDTH_MM, HEIGHT_MM = 60, 40
MM_PER_INCH = 25.4
PHASES = ("Pre", "Early", "Late", "Post")
PROFILES = (
    ("SUS", "Sustained\nresponse", "#0072B2", (0, 0.55, 1, 1)),
    ("PSR", "Post-stimulation\nreversal", "#D55E00", (0, 1, 1, 0.25)),
    ("ISR", "Intra-stimulation\nreversal", "#E69F00", (0, 1, 0.25, 0.25)),
    ("REB", "Rebound\nresponse", "#CC79A7", (0, 1, 0.25, 0.85)),
    ("DEL", "Delayed-onset\nresponse", "#7A8F3A", (0, 0, 0, 1)),
    ("STB", "Stable\nprofile", "#A8ADB4", (0, 0, 0, 0)),
)
DESCRIPTION = (
    "Six representative, direction-oriented Phase response profiles: sustained "
    "response (SUS), post-stimulation reversal (PSR), intra-stimulation reversal "
    "(ISR), rebound response (REB), delayed-onset response (DEL), and stable "
    "profile (STB). Positions are Pre, Early, Late, and Post. Gray shading marks "
    "the stimulation phases. Amplitudes are arbitrary schematic coordinates, "
    "not measured data. The first nonzero change is oriented positive. Zero "
    "means an absolute adjacent change within the Pre-based Q75 bound, not "
    "statistical nonsignificance. STB also requires the total four-phase range "
    "to be within Q75. Allowed patterns are defined in phase_response_profiles.py."
)


def draw(output_root: Path):
    """Render the PDF and an inspection preview of the actual PDF page."""
    output_root.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        target = output_root / f"phase_response_schematic.{suffix}"
        if target.exists():
            raise FileExistsError(f"Output exists: {target}")
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6,
            "text.color": "#202020",
            "axes.labelcolor": "#202020",
            "xtick.color": "#202020",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    fig = plt.figure(figsize=(WIDTH_MM / MM_PER_INCH, HEIGHT_MM / MM_PER_INCH), dpi=160)

    for index, (code, name, color, values) in enumerate(PROFILES):
        row, column = divmod(index, 3)
        left_mm = 0.40 + column * 20.17
        top_mm = 0.20 + row * 20.65
        center_mm = left_mm + 9.15
        fig.text(
            center_mm / WIDTH_MM,
            1 - top_mm / HEIGHT_MM,
            code,
            ha="center",
            va="top",
            fontsize=7,
            fontweight="bold",
        )
        fig.text(
            center_mm / WIDTH_MM,
            1 - (top_mm + 2.6) / HEIGHT_MM,
            name,
            ha="center",
            va="top",
            fontsize=6,
            linespacing=1.06,
        )
        plot_top_mm = top_mm + 7.7
        plot_height_mm = 8.70
        axis = fig.add_axes(
            [
                left_mm / WIDTH_MM,
                1 - (plot_top_mm + plot_height_mm) / HEIGHT_MM,
                18.3 / WIDTH_MM,
                plot_height_mm / HEIGHT_MM,
            ],
            label=code,
        )
        axis.set_gid(f"profile-{code.lower()}")
        axis.set_xlim(-0.25, 3.25)
        axis.set_ylim(-0.19, 1.23)
        axis.axvspan(0.5, 2.5, facecolor="#EFEFEF", edgecolor="none", zorder=0)
        (curve,) = axis.plot(
            range(4),
            values,
            color=color,
            linewidth=1,
            solid_capstyle="round",
            solid_joinstyle="round",
            marker="o",
            markersize=2.5,
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=0.35,
            zorder=3,
        )
        curve.set_gid(f"curve-{code.lower()}")
        axis.set_xticks(range(4), PHASES)
        axis.tick_params(axis="x", length=0.9, width=0.3, pad=1.0, labelsize=6)
        axis.set_yticks([])
        for side in ("left", "right", "top"):
            axis.spines[side].set_visible(False)
        axis.spines["bottom"].set_color("#959595")
        axis.spines["bottom"].set_linewidth(0.35)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels = list(fig.texts)
    labels.extend(label for axis in fig.axes for label in axis.get_xticklabels())
    for label in labels:
        box = label.get_window_extent(renderer)
        if (
            box.x0 < 0
            or box.y0 < 0
            or box.x1 > fig.bbox.width
            or box.y1 > fig.bbox.height
        ):
            raise ValueError(f"Text lies outside the figure: {label.get_text()}")

    pdf_path = output_root / "phase_response_schematic.pdf"
    with pdf_path.open("xb") as stream:
        fig.savefig(
            stream,
            format="pdf",
            metadata={
                "Title": "Schematic definitions of Phase response profiles",
                "Subject": DESCRIPTION,
                "CreationDate": None,
                "ModDate": None,
            },
        )
    plt.close(fig)
    subprocess.run(
        [
            "pdftoppm",
            "-png",
            "-r",
            "600",
            "-singlefile",
            str(pdf_path),
            str(output_root / "phase_response_schematic"),
        ],
        check=True,
    )
    print(pdf_path)
    print(output_root / "phase_response_schematic.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    draw(parser.parse_args().output_root)
