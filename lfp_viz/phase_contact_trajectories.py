"""Render the 16 significant Ipsi contact-paired trajectories for Supplementary Figure 5."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from lfp_stats.config import load_stats_config
from lfp_stats.runner import build_stats_plan
from .phase_response_profiles import PHASES, require_new_outputs


def render(model_data, profiles, output_root):
    """Keep model values and contact membership; use the manuscript panel order."""
    ROOT = Path(output_root)
    names = ["contact_paired_trajectories." + ext for ext in ("pdf", "svg", "png")]
    names += [
        "preview.png",
        "contact_phase_values.csv",
        "patient_colors.csv",
        "validation.json",
    ]
    require_new_outputs([ROOT / name for name in names])
    ROOT.mkdir(parents=True, exist_ok=True)
    profiles = profiles.assign(
        _domain=profiles.Domain.map({"local": 0, "connectivity": 1})
    )
    profiles = profiles.sort_values(
        ["_domain", "Polar", "RegionValue", "Metric", "FeatureOutput", "Band"],
        kind="stable",
    ).to_dict("records")
    ids = {r["ModelID"] for r in profiles}
    rows = model_data.loc[
        model_data.ModelID.isin(ids) & model_data.Lat.eq("Ipsi")
    ].to_dict("records")
    phases = PHASES
    units = defaultdict(dict)
    for r in rows:
        key = (r["ModelID"], r["ID"], r["ContactUnitID"])
        if r["Phase"] in units[key]:
            raise ValueError("Duplicate recording-unit phase observation.")
        units[key][r["Phase"]] = float(r["Value"])
    if len(profiles) != 16 or len(ids) != 16 or {r["ModelID"] for r in rows} != ids:
        raise ValueError(
            "The manuscript grid requires 16 unique profiles with matching model inputs."
        )
    if not all(
        set(x) == set(phases) and np.isfinite(list(x.values())).all()
        for x in units.values()
    ):
        raise ValueError(
            "Each recording unit must have four finite phase observations."
        )
    patients = sorted({r["ID"] for r in rows})
    colors = {
        p: plt.colormaps["viridis"](t)
        for p, t in zip(patients, np.linspace(0, 1, len(patients)))
    }
    symbols = ["o", "s", "^", "D", "v", "P", "X", "h"]
    markers = {p: symbols[i % len(symbols)] for i, p in enumerate(patients)}
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6,
            "axes.titlesize": 6,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    boxsize = (25.0, 25.0)
    page_mm = (180.0, 184.0)
    fig = plt.figure(figsize=(page_mm[0] / 25.4, page_mm[1] / 25.4))
    axes = np.array(
        [
            fig.add_axes(
                [
                    (14 + 45 * c) / page_mm[0],
                    (22 + 44 * (3 - r)) / page_mm[1],
                    boxsize[0] / page_mm[0],
                    boxsize[1] / page_mm[1],
                ]
            )
            for r in range(4)
            for c in range(4)
        ]
    ).reshape(4, 4)
    bands = {
        "delta": "δ",
        "theta": "θ",
        "alpha": "α",
        "beta_low": "low-β",
        "beta_high": "high-β",
        "gamma_high": "high-γ",
    }
    counts = []
    for i, (ax, r) in enumerate(zip(axes.flat, profiles)):
        selected = {key: vals for key, vals in units.items() if key[0] == r["ModelID"]}
        for (_, patient, contact), vals in sorted(selected.items()):
            ax.plot(
                range(4),
                [vals[p] for p in phases],
                color=colors[patient],
                marker=markers[patient],
                markersize=3,
                linewidth=0.85,
                alpha=0.9,
                markeredgewidth=0.35,
                markeredgecolor=colors[patient],
            )
        n = len({k[1] for k in selected})
        m = len(selected)
        if r["Metric"] == "aperiodic":
            metric = "Aperiodic " + r["Band"]
            scale = r["Band"].capitalize()
        else:
            metric = {
                "raw_power": "Total power",
                "ciplv": "ciPLV",
                "wpli": "wPLI",
                "imcoh_abs": "|ImCoh|",
            }.get(r["Metric"])
            scale = {
                "raw_power": "Power (dB)",
                "ciplv": "ciPLV (logit)",
                "wpli": "wPLI (logit)",
                "imcoh_abs": "|ImCoh| (Fisher z)",
            }.get(r["Metric"])
            if r["Metric"] == "burst":
                part = r["FeatureOutput"].replace("-scalar", "")
                metric = "Burst " + part
                scale = {
                    "duration": "Duration (log10 s)",
                    "rate": "Rate (asinh)",
                    "occupancy": "Occupancy (%)",
                }[part]
            metric = bands[r["Band"]] + " " + metric
        ax.text(
            -0.42,
            1.06,
            chr(97 + i),
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=8,
        )
        suffix = scale[scale.index("(") :] if "(" in scale else ""
        ax.set_ylabel(metric + ("\n" + suffix if suffix else ""), labelpad=3)
        ax.set_xlabel(
            f'{r["Polar"]} · Ipsi · {r["RegionValue"].replace("-","–")}', labelpad=4
        )
        ax.set_xticks(range(4), phases)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("center")
        ax.set_xlim(-0.15, 3.15)
        ax.margins(y=0.12)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2.5, width=0.6, pad=2)
        counts.append(dict(ModelID=r["ModelID"], Patients=n, RecordingUnits=m))
    handles = [
        Line2D(
            [],
            [],
            color=colors[p],
            marker=markers[p],
            markersize=4,
            lw=1,
            label=p.replace("sub-", ""),
        )
        for p in patients
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=4,
        frameon=False,
        handlelength=1.4,
        columnspacing=1.25,
        fontsize=6,
    )
    fig.canvas.draw()
    measured_boxes = [
        (
            ax.get_window_extent().width / fig.dpi * 25.4,
            ax.get_window_extent().height / fig.dpi * 25.4,
        )
        for ax in axes.flat
    ]
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(
            ROOT / f"contact_paired_trajectories.{ext}", dpi=600, facecolor="white"
        )
    fig.savefig(ROOT / "preview.png", dpi=160, facecolor="white")
    with (ROOT / "contact_phase_values.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (ROOT / "patient_colors.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Color", "Marker"])
        writer.writerows((p, to_hex(colors[p]), markers[p]) for p in patients)
    (ROOT / "validation.json").write_text(
        json.dumps(
            {
                "profiles": 16,
                "observations": len(rows),
                "recording_unit_trajectories": len(units),
                "all_four_phases_complete": True,
                "duplicate_phase_keys": 0,
                "values": "Unchanged model inputs",
                "panels": counts,
                "axes_boxsize_mm": measured_boxes,
                "page_mm": page_mm,
            },
            indent=2,
        )
        + "\n"
    )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats-config", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    profiles = pd.read_csv(args.profiles)
    plan = build_stats_plan(load_stats_config(args.stats_config))
    render(plan.model_data, profiles, args.output_root)


if __name__ == "__main__":
    main()
