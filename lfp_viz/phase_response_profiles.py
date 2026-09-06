"""Classify and visualize significant Ipsi Phase trajectories using Pre Q75."""

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

PHASES = ["Pre", "Early", "Late", "Post"]
CONTRASTS = ["EarlyMinusPre", "LateMinusEarly", "PostMinusLate"]
QUANTILE = 0.75
CLASSES = ["SUS", "PSR", "ISR", "REB", "DEL", "STB"]
NAMES = dict(
    zip(
        CLASSES,
        [
            "Sustained response",
            "Post-stimulation reversal",
            "Intra-stimulation reversal",
            "Rebound response",
            "Delayed-onset response",
            "Stable profile",
        ],
    )
)
COLORS = dict(
    zip(CLASSES, ["#0072B2", "#D55E00", "#E69F00", "#CC79A7", "#7A8F3A", "#A8ADB4"])
)
PATTERNS = {
    **dict.fromkeys(["+++", "++0", "+0+", "+00", "0++", "0+0"], "SUS"),
    **dict.fromkeys(["++-", "+0-", "0+-"], "PSR"),
    **dict.fromkeys(["+-0", "+--"], "ISR"),
    "+-+": "REB",
    "00+": "DEL",
    "000": "STB",
}
BANDS = [
    "exponent",
    "offset",
    "delta",
    "theta",
    "alpha",
    "beta_low",
    "beta_high",
    "gamma_low",
    "gamma_high",
]
METRICS = [
    "aperiodic",
    "periodic",
    "raw_power",
    "burst",
    "ciplv",
    "imcoh_abs",
    "wpli",
    "psi",
    "trgc",
]
FIGURES = ["significant_profile_distribution", "significant_profile_patterns"]
TABLES = ["profiles.csv", "significant_contrasts.csv", "class_counts.csv"]
MUTED, GRID = "#737B85", "#D9DDE3"


def classify(differences, threshold, phase_range):
    """Apply the accepted three-step signs and full-range stable rule."""
    signs = np.where(
        differences > threshold, 1, np.where(differences < -threshold, -1, 0)
    )
    raw = "".join({1: "+", 0: "0", -1: "-"}[value] for value in signs)
    nonzero = np.flatnonzero(signs)
    multiplier = -1 if len(nonzero) and signs[nonzero[0]] < 0 else 1
    oriented = "".join({1: "+", 0: "0", -1: "-"}[value] for value in signs * multiplier)
    if oriented == "000" and phase_range > threshold:
        raise ValueError(
            "All adjacent changes are subthreshold but the full range exceeds Q75; classification is unresolved."
        )
    return raw, oriented, PATTERNS[oriented], multiplier


def read_table(stats_root, registry, kind):
    frames = []
    for relative in registry.loc[registry.TableType == kind, "Path"]:
        path = stats_root / relative
        frame = pd.read_csv(path)
        frame["SourceCSV"] = str(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def select_phase_lat(frame):
    return frame[
        (frame.AnalysisID == "phase-by-lat-contact-paired")
        & (frame.Engine == "lmer")
        & (frame.TestKind == "emm_pairwise")
        & (frame.TestID == "Phase-Lat")
        & (frame.Lat == "Ipsi")
    ]


def metric_label(metric, output, band):
    if metric == "aperiodic":
        return "Aperiodic " + band
    if metric == "burst":
        return (
            "Burst "
            + {
                "mean-scalar": "amplitude",
                "duration-scalar": "duration",
                "rate-scalar": "rate",
                "occupancy-scalar": "occupancy",
            }[output]
        )
    return {
        "raw_power": "Total power",
        "periodic": "Periodic power",
        "ciplv": "ciPLV",
        "imcoh_abs": "|ImCoh|",
        "wpli": "wPLI",
        "psi": "PSI",
        "trgc": "TRGC",
    }[metric]


def derive_tables(stats_root: Path, calibration_root: Path):
    registry = pd.read_csv(stats_root / "statistics.csv")
    tukey = select_phase_lat(read_table(stats_root, registry, "tukey"))
    emm = select_phase_lat(read_table(stats_root, registry, "emmeans"))
    significant = tukey[tukey.p_tukey < 0.05].copy()
    mids = significant.ModelID.unique()
    models = read_table(stats_root, registry, "models")
    models = models[
        (models.AnalysisID == "phase-by-lat-contact-paired")
        & (models.Engine == "lmer")
        & models.ModelID.isin(mids)
    ]
    pairing = read_table(stats_root, registry, "contact_pairing")
    pairing = pairing[
        pairing.ModelID.isin(mids) & (pairing.Lat == "Ipsi") & pairing.Included
    ]
    if models.ModelID.duplicated().any() or not pairing.CompletePhase.all():
        raise ValueError(
            "Unique models with complete included Ipsi contacts are required."
        )
    index = pd.read_csv(calibration_root / "bootstrap_contact_index.csv")
    pre = pd.read_csv(calibration_root / "pre_reconstruction.csv")
    bootstrap_info = json.loads((calibration_root / "validation.json").read_text())
    if bootstrap_info["replicates"] != 10000 or bootstrap_info["block_seconds"] != 10:
        raise ValueError(
            "The stored calibration does not match the accepted bootstrap design."
        )
    records = []
    max_contrast_error = 0.0
    max_pre_error = 0.0
    with np.load(
        calibration_root / "bootstrap_contact_values.npz", allow_pickle=False
    ) as bootstrap:
        for model in models.itertuples():
            mid = model.ModelID
            e = emm[emm.ModelID == mid]
            t = tukey[tukey.ModelID == mid]
            if (
                len(e) != 4
                or e.Phase.nunique() != 4
                or len(t) != 6
                or t[["group1", "group2"]].duplicated().any()
            ):
                raise ValueError(
                    f"Expected four EMMs and six unique pairwise contrasts: {mid}"
                )
            mu = e.set_index("Phase").loc[PHASES, "emmean"].to_numpy()
            differences = np.diff(mu)
            if not np.isfinite(mu).all():
                raise ValueError(f"Nonfinite EMMs: {mid}")
            for row in t.itertuples():
                error = abs(
                    mu[PHASES.index(row.group1)]
                    - mu[PHASES.index(row.group2)]
                    - row.estimate
                )
                max_contrast_error = max(max_contrast_error, error)
                if error > 1e-10:
                    raise ValueError(f"EMM and Tukey estimates disagree: {mid}")
            selected = index[index.ModelID == mid]
            contacts = pairing[pairing.ModelID == mid]
            unit_columns = ["ID", "ContactUnitID"]
            expected = set(contacts[unit_columns].itertuples(index=False, name=None))
            actual = set(selected[unit_columns].itertuples(index=False, name=None))
            if actual != expected or len(selected) != len(expected):
                raise ValueError(
                    f"Bootstrap membership differs from included Ipsi contacts: {mid}"
                )
            baseline = pre[pre.ModelID == mid]
            if set(
                baseline[unit_columns].itertuples(index=False, name=None)
            ) != expected or not np.allclose(baseline.PreEnd - baseline.PreStart, 60):
                raise ValueError(f"Baseline membership or duration differs: {mid}")
            max_pre_error = max(max_pre_error, float(baseline.PreError.abs().max()))
            if baseline.PreError.abs().max() > 1e-8:
                raise ValueError(f"Stored baseline reconstruction failed: {mid}")
            arrays = np.stack([bootstrap[str(i)] for i in selected.index])
            if arrays.shape[1:] != (2, 10000):
                raise ValueError(f"Unexpected bootstrap array dimensions: {mid}")
            finite = np.isfinite(arrays).all(axis=(0, 1))
            if finite.sum() != 10000:
                raise ValueError(
                    f"The selected trajectory does not have 10000 valid replicates: {mid}"
                )
            repeated_difference = (arrays[:, 0] - arrays[:, 1]).mean(axis=0)
            threshold = float(
                np.quantile(np.abs(repeated_difference), QUANTILE, method="linear")
            )
            phase_range = float(np.ptp(mu))
            raw, oriented, code, multiplier = classify(
                differences, threshold, phase_range
            )
            identity = {
                key: getattr(model, key)
                for key in [
                    "AnalysisID",
                    "ModelID",
                    "Engine",
                    "Formula",
                    "Domain",
                    "Metric",
                    "FeatureOutput",
                    "Band",
                    "RegionVariable",
                    "RegionValue",
                    "Polar",
                    "OutputGroup",
                ]
            }
            records.append(
                {
                    **identity,
                    "Lat": "Ipsi",
                    "MetricLabel": metric_label(
                        model.Metric, model.FeatureOutput, model.Band
                    ),
                    "BandLabel": (
                        "-"
                        if model.Metric == "aperiodic"
                        else model.Band.replace("_", "-")
                        .replace("beta-low", "low-beta")
                        .replace("beta-high", "high-beta")
                        .replace("gamma-low", "low-gamma")
                        .replace("gamma-high", "high-gamma")
                    ),
                    **{f"EMM_{phase}": value for phase, value in zip(PHASES, mu)},
                    **dict(zip(CONTRASTS, differences)),
                    "Quantile": QUANTILE,
                    "Q75": threshold,
                    "PhaseRange": phase_range,
                    "RawPattern": raw,
                    "OrientedPattern": oriented,
                    "OrientationMultiplier": multiplier,
                    "ClassCode": code,
                    "ClassName": NAMES[code],
                    "ModelStatus": model.Status,
                    "ModelSingular": model.Singular,
                    "ModelN_ID": model.n_ID,
                    "IpsiN_ID": selected.ID.nunique(),
                    "IpsiContactUnits": len(selected),
                    "ValidReplicates": int(finite.sum()),
                    "SignificantContrastCount": int((t.p_tukey < 0.05).sum()),
                    "MinPTukey": float(t.p_tukey.min()),
                    "SourceEMMCSV": e.SourceCSV.iloc[0],
                    "SourceTukeyCSV": t.SourceCSV.iloc[0],
                    "BootstrapValuesNPZ": str(
                        calibration_root / "bootstrap_contact_values.npz"
                    ),
                }
            )
    profiles = pd.DataFrame(records)
    profiles["_domain"] = profiles.Domain.map({"local": 0, "connectivity": 1})
    profiles["_class"] = profiles.ClassCode.map(dict(zip(CLASSES, range(6))))
    profiles["_polar"] = profiles.Polar.map({"Anodal": 0, "Cathodal": 1})
    profiles["_region"] = profiles.RegionValue.map(
        {"SNr": 0, "STN": 1, "SNr-STN": 2, "SNr\u2192STN": 3}
    )
    profiles["_metric"] = profiles.Metric.map(dict(zip(METRICS, range(len(METRICS)))))
    profiles["_band"] = profiles.Band.map(dict(zip(BANDS, range(len(BANDS)))))
    profiles = (
        profiles.sort_values(
            [
                "_domain",
                "_class",
                "_polar",
                "_region",
                "_metric",
                "FeatureOutput",
                "_band",
            ],
            kind="stable",
        )
        .drop(columns=[c for c in profiles if c.startswith("_")])
        .reset_index(drop=True)
    )
    rows = []
    for domain in ["local", "connectivity"]:
        group = profiles[profiles.Domain == domain]
        if group.empty:
            raise ValueError(f"No significant response profiles for domain: {domain}")
        for code in CLASSES:
            count = int(group.ClassCode.eq(code).sum())
            rows.append(
                {
                    "Domain": domain,
                    "Lat": "Ipsi",
                    "Quantile": QUANTILE,
                    "ClassCode": code,
                    "ClassName": NAMES[code],
                    "Count": count,
                    "Denominator": len(group),
                    "Percent": 100 * count / len(group),
                    "Color": COLORS[code],
                }
            )
    counts = pd.DataFrame(rows)
    order = dict(zip(profiles.ModelID, range(len(profiles))))
    significant["_order"] = significant.ModelID.map(order)
    significant = (
        significant.sort_values(["_order", "group2", "group1"], kind="stable")
        .drop(columns="_order")
        .reset_index(drop=True)
    )
    print(
        json.dumps(
            {
                "trajectories": len(profiles),
                "significant_contrasts": len(significant),
                "max_emm_contrast_error": max_contrast_error,
                "max_stored_pre_error": max_pre_error,
                "valid_replicates_per_trajectory": 10000,
                "model_statuses": profiles.ModelStatus.value_counts().to_dict(),
            }
        ),
        flush=True,
    )
    return profiles, significant, counts


def require_new_outputs(paths):
    """Refuse to replace generated outputs; use a fresh output directory."""
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"Outputs already exist: {existing}")


def plot_outputs(output_root):
    """Render the saved Q75 tables using the publication display controls."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from lfp_viz import visualdf

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6.5,
            "text.color": "black",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )
    profiles = pd.read_csv(
        output_root / "profiles.csv", dtype={"RawPattern": str, "OrientedPattern": str}
    )
    counts = pd.read_csv(output_root / "class_counts.csv")
    require_new_outputs(
        [
            output_root / (stem + "." + suffix)
            for stem in FIGURES
            for suffix in ["pdf", "svg", "png"]
        ]
    )

    donut_box_mm = 30
    fig = visualdf.plot_donut_grid_df(
        counts,
        category_var="ClassCode",
        value_col="Count",
        panel_var="Domain",
        category_levels=CLASSES,
        panel_levels=["local", "connectivity"],
        palette=COLORS,
        boxsize=(donut_box_mm, donut_box_mm),
        panel_gap=(4, 0),
        ncols=2,
        ring_width=0.35,
        startangle=90,
        counterclock=False,
        label_format="{category}\n{count:g} ({percent:.1f}%)",
        label_distance=1.10,
        label_fontsize=6,
        center_format="N = {total:g}",
        center_fontsize=7,
        label_top=["Local", "Connectivity"],
        panel_label_fontsize=7,
        legend_loc="outside_bottom",
        legend_ncol=6,
        legend_fontsize=6,
        font_family="Arial",
        transparent=False,
        dpi=600,
    )
    # Place the existing artists on the fixed publication page without scaling text.
    fig.set_size_inches(95 / 25.4, 40 / 25.4, forward=True)
    for ax, center_x_mm in zip(fig.axes, [21.6, 72.85]):
        ax.set_position(
            [
                (center_x_mm - donut_box_mm / 2) / 95,
                6.25 / 40,
                donut_box_mm / 95,
                donut_box_mm / 40,
            ]
        )
    legend = fig.legends[0]
    legend.borderpad = 0
    legend.set_bbox_to_anchor((0.5, -0.7 / 40), transform=fig.transFigure)
    for suffix in ["pdf", "svg", "png"]:
        fig.savefig(output_root / (FIGURES[0] + "." + suffix), dpi=600)
    plt.close(fig)

    domains = [("local", "Local"), ("connectivity", "Connectivity")]
    width_mm, row_mm = 180, 4
    height_mm = 10.5 + len(profiles) * row_mm + len(domains) * 5.5
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width_mm)
    ax.set_ylim(height_mm, 0)
    ax.axis("off")
    headers = [
        ("Polar", 2, "left"),
        ("Region", 22, "left"),
        ("Band", 40, "left"),
        ("Metric", 60, "left"),
        ("Early\u2212Pre", 103, "center"),
        ("Late\u2212Early", 119, "center"),
        ("Post\u2212Late", 135, "center"),
        ("Oriented", 153, "center"),
        ("Class", 174, "center"),
    ]
    for label, x, align in headers:
        ax.text(
            x,
            3,
            label,
            ha=align,
            va="center",
            fontsize=6.5,
            fontweight="bold",
            color="black",
        )
    ax.plot([2, width_mm - 2], [6, 6], color="black", linewidth=0.5)
    y = 9.5
    for domain, title in domains:
        group = profiles[profiles.Domain == domain]
        ax.text(2, y, title, fontsize=7, fontweight="bold", va="center", color="black")
        y += 4
        for row in group.itertuples():
            for x, value in [
                (2, row.Polar),
                (22, row.RegionValue),
                (40, row.BandLabel),
                (60, row.MetricLabel),
            ]:
                ax.text(x, y, value, fontsize=6.5, va="center", color="black")
            for x, sign in zip([103, 119, 135], row.RawPattern):
                ax.text(
                    x,
                    y,
                    sign.replace("-", "\u2212"),
                    ha="center",
                    va="center",
                    fontsize=7,
                    fontweight="bold" if sign != "0" else "normal",
                    color=MUTED if sign == "0" else "black",
                )
            ax.text(
                153,
                y,
                " ".join(row.OrientedPattern).replace("-", "\u2212"),
                ha="center",
                va="center",
                fontsize=6.5,
                color="black",
            )
            ax.add_patch(
                Rectangle(
                    (168, y - 1),
                    2,
                    2,
                    facecolor=COLORS[row.ClassCode],
                    edgecolor="none",
                )
            )
            ax.text(
                171.2,
                y,
                row.ClassCode,
                ha="left",
                va="center",
                fontsize=6.5,
                color="black",
            )
            y += row_mm
        ax.plot(
            [2, width_mm - 2],
            [y - row_mm / 2, y - row_mm / 2],
            color=GRID,
            linewidth=0.4,
        )
        y += 1.5
    for suffix in ["pdf", "svg", "png"]:
        fig.savefig(output_root / (FIGURES[1] + "." + suffix), dpi=600)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats-root", type=Path, required=True)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output_root = args.output_root
    require_new_outputs(
        [output_root / name for name in TABLES]
        + [
            output_root / (stem + "." + suffix)
            for stem in FIGURES
            for suffix in ["pdf", "svg", "png"]
        ]
    )
    profiles, significant, counts = derive_tables(
        args.stats_root, args.calibration_root
    )
    output_root.mkdir(parents=True, exist_ok=True)
    for frame, name in zip([profiles, significant, counts], TABLES):
        frame.to_csv(output_root / name, index=False)
    plot_outputs(output_root)
    print(
        counts[["Domain", "ClassCode", "Count", "Denominator", "Percent"]].to_string(
            index=False
        ),
        flush=True,
    )
    print(f"Outputs: {output_root}", flush=True)


if __name__ == "__main__":
    main()
