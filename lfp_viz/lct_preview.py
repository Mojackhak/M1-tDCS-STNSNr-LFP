"""Render LCT association fits and heatmaps from saved inference."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from lfp_clinical.runner import _trash_existing
from lfp_cohort.io import atomic_csv
from .config import VizConfig
from .correlation_fit import _raw_linear_model_curve, render_correlation_fit_figure
from .correlation_heatmap import (
    _build_scale_correlation_heatmap_plan, render_correlation_heatmap_figure,
)


def draw_fit(points, row, config, *, top, right, xlabel, ylabel, additional_covariates=(), method="partial_spearman"):
    """Render raw observations and an adjusted OLS mean-response ribbon."""
    points = points.copy()
    points["StratumLabel"] = top
    curve = _raw_linear_model_curve(
        points, method=method, x_column="PredictorValue", y_column="Outcome",
        baseline_reference=float(points.Baseline.median()), confidence_level=.95,
        additional_covariates=additional_covariates,
    )
    style = config["style"]
    style["fit"]["show_ribbon"] = True
    p_column = style["annotation"]["p_column"]
    figure = {
        "StratumLabel": top, "Lat": right, "Method": method,
        "BaselineReference": float(points.Baseline.median()), "ConfidenceLevel": .95,
        "rho": row.rho, p_column: row[p_column], "XLabel": xlabel, "YLabel": ylabel,
    }
    fig = render_correlation_fit_figure(points, figure, config, prepared_curve=curve)
    return fig, points



def lct_fit_config(config):
    """Use uncorrected endpoint P values without changing other fit styles."""
    config = deepcopy(config)
    annotation = config["style"]["annotation"]
    annotation["p_column"] = "p_raw"
    annotation["text_format"] = "ρ = {beta:.2f}\nP = {p:.3f} ({stars})"
    annotation["small_p_text_format"] = "ρ = {beta:.2f}\nP < 0.001 (***)"
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--heatmap-only", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    paths = yaml.safe_load(args.config.read_text())["paths"]
    if args.all:
        run_all(paths)
        return
    root, output = Path(paths["stats_root"]), Path(paths["output_root"])
    if args.heatmap_only:
        heat_fig, cells = draw_heatmap(paths)
        targets = [output / HEATMAP_NAME, output / "heatmap-cells.csv"]
        existing = [target for target in targets if target.exists()]
        if existing:
            _trash_existing(existing)
        output.mkdir(parents=True, exist_ok=True)
        heat_fig.savefig(targets[0], format="pdf", dpi=600, bbox_inches="tight")
        plt.close(heat_fig)
        atomic_csv(cells, targets[1], overwrite=False)
        print(targets[0])
        return
    coverage = pd.read_csv(root / "coverage.csv")
    lct = pd.read_csv(root / "lct-outcome/results.csv")
    tdcs = pd.read_csv(root / "tdcs-adjusted/results.csv")

    def fit_config():
        path = Path(paths["fit_style"])
        data = yaml.safe_load(path.read_text())
        data["inputs"].update(x_column="PredictorValue", y_column="Outcome")
        return VizConfig(path, data)

    def selected_points(row, predictor):
        return coverage[coverage.ScaleLabel.eq(row.ScaleLabel)
                        & coverage.EndpointID.eq(row.EndpointID)
                        & coverage.PredictorID.eq(predictor) & coverage.Included].copy()

    lct_row = lct[lct.ScaleLabel.eq("UPDRS-III") & lct.EndpointID.eq("stn-3m")].iloc[0]
    lct_points = selected_points(lct_row, "LCT")
    lct_fig, lct_points = draw_fit(
        lct_points, lct_row, lct_fit_config(fit_config()), top="LCT | UPDRS-III", right="STN-DBS",
        xlabel="LCT improvement (%)", ylabel="UPDRS-III (STN-DBS)",
    )
    row = tdcs[tdcs.ScaleLabel.eq("UPDRS-III") & tdcs.EndpointID.eq("stn-snr-3m")
               & tdcs.Metric.eq("ciplv") & tdcs.Band.eq("delta")
               & tdcs.Polar.eq("Anodal") & tdcs.Lat.eq("Ipsi")
               & tdcs.Phase.eq("Early")].iloc[0]
    points = selected_points(row, row.PredictorID)
    fit_fig, points = draw_fit(
        points, row, fit_config(), top="SNr–STN | ⊕ | Early−Pre", right="Ipsi | LCT adjusted",
        xlabel="Δδ ciPLV", ylabel="UPDRS-III (STN+SNr-DBS)",
        additional_covariates=("LCT",),
    )

    heat_fig, cells = draw_heatmap(paths)
    names = {
        "lct-UPDRS-III_STN_baseline-adjusted_fit.pdf": lct_fig,
        "tdcs-UPDRS-III_STN-SNr_ciplv-delta_SNr-STN_Anodal_Ipsi_Early-Pre_fit.pdf": fit_fig,
        HEATMAP_NAME: heat_fig,
    }
    tables = {"lct-fit-points.csv": lct_points, "tdcs-fit-points.csv": points,
              "heatmap-cells.csv": cells}
    tables["figures.csv"] = pd.DataFrame({"OutputPath": [str(output/name) for name in names],
                                        "Status": "preview", "StatsRoot": str(root)})
    existing = [output / name for name in [*names, *tables] if (output / name).exists()]
    if existing:
        _trash_existing(existing)
    output.mkdir(parents=True, exist_ok=True)
    for name, fig in names.items():
        fig.savefig(output / name, format="pdf", dpi=600, bbox_inches="tight")
        plt.close(fig)
        print(output / name)
    for name, table in tables.items():
        atomic_csv(table, output / name, overwrite=False)
    print(f"Heatmap: {len(cells)} cells; {cells.p_holm.lt(.05).sum()} significant")


HEATMAP_NAME = "scale-PDQ-39_domain-local_region-STN_polar-Anodal_lat-Ipsi_figure-clinical-correlation-heatmap copy.pdf"


def draw_heatmap(paths):
    """Reuse the reference composite unchanged except for values and colorbar."""
    tdcs = pd.read_csv(Path(paths["stats_root"]) / "tdcs-adjusted/results.csv")
    config, plan = heatmap_plan(paths, tdcs)
    spec = plan.figures[plan.figures.FigureKind.eq("composite")
                        & plan.figures.ScaleLabel.eq("PDQ-39")
                        & plan.figures.RegionValue.eq("STN")].iloc[0]
    cells = plan.coverage[plan.coverage.FigureID.eq(spec.FigureID)].copy()
    return render_correlation_heatmap_figure(cells, spec, config), cells


def heatmap_plan(paths, tdcs):
    """Build all approved atomic and composite layouts through the shared planner."""
    heat_path = Path(paths["heatmap_style"])
    heat_data = yaml.safe_load(heat_path.read_text())
    heat_data["visualization"]["id"] = "clinical-correlation-lct-preview"
    heat_data["paths"]["output_root"] = paths["output_root"]
    heat_data["inputs"]["stats_analysis_id"] = "clinical-correlation-lct"
    for scale in ["UPDRS-III", "PDQ-39"]:
        heat_data["labels"]["colorbar"][scale] = "Partial Spearman’s ρ (baseline + LCT-adjusted)"
    config = VizConfig(heat_path, heat_data)
    tdcs = tdcs.copy()
    tdcs["Scale"] = tdcs.ScaleLabel.map({"UPDRS-III": "MDS-UPDRS III score", "PDQ-39": "PDQ39 score"})
    tdcs["CorrelationID"] = tdcs.ScaleLabel + "|" + tdcs.EndpointID + "|" + tdcs.PredictorID
    plan = _build_scale_correlation_heatmap_plan(config, tdcs)
    plan.coverage["PhaseStripLabel"] = plan.coverage.Phase
    return config, plan


def run_all(paths):
    """Export approved fits and heatmaps without modifying inference outputs."""
    root, output = Path(paths["stats_root"]), Path(paths["output_root"])
    coverage = pd.read_csv(root / "coverage.csv")
    lct = pd.read_csv(root / "lct-outcome/results.csv")
    tdcs = pd.read_csv(root / "tdcs-adjusted/results.csv")
    selected = tdcs[tdcs.Status.eq("ok") & tdcs.p_holm.lt(.05)]
    fit_path = Path(paths["fit_style"])
    fit_data = yaml.safe_load(fit_path.read_text())
    fit_data["inputs"].update(x_column="PredictorValue", y_column="Outcome")
    fit_config = VizConfig(fit_path, fit_data)
    records, point_tables = [], []

    def export(fig, target):
        if target.exists():
            _trash_existing([target])
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, format="pdf", dpi=600, bbox_inches="tight")
        plt.close(fig)

    def fit_points(row, predictor):
        points = coverage[coverage.ScaleLabel.eq(row.ScaleLabel)
                          & coverage.EndpointID.eq(row.EndpointID)
                          & coverage.PredictorID.eq(predictor) & coverage.Included].copy()
        if len(points) != int(row.n) or points.ID.duplicated().any():
            raise ValueError("Saved fit coverage does not match statistics.")
        return points

    lct_config = lct_fit_config(fit_config)
    for _, source in lct.iterrows():
        for variant, method in [("ordinary", "spearman"), ("baseline-adjusted", "partial_spearman")]:
            row = source.copy()
            if variant == "ordinary":
                row["rho"], row["p_raw"] = row.rho_ordinary, row.p_raw_ordinary
            endpoint = fit_config["labels"]["endpoint"][row.EndpointID]
            points = fit_points(row, "LCT")
            fig, points = draw_fit(
                points, row, lct_config, top=f"LCT | {row.ScaleLabel}", right=endpoint,
                xlabel="LCT improvement (%)", ylabel=f"{row.ScaleLabel} ({endpoint})", method=method,
            )
            target = output / "lct-outcome/fit" / variant / f"{row.ScaleLabel}_{endpoint}.pdf"
            export(fig, target)
            record = row.to_dict()
            record.update(OutputPath=str(target), FigureType="lct-fit", Variant=variant, Status="ok")
            records.append(record)
            points["OutputPath"] = str(target)
            point_tables.append(points)
    print("LCT fits: 8", flush=True)

    for index, (_, row) in enumerate(selected.iterrows(), 1):
        labels = fit_config["labels"]
        region, polar = labels["region"][row.RegionValue], labels["polarity"][row.Polar]
        endpoint = labels["endpoint"][row.EndpointID]
        points = fit_points(row, row.PredictorID)
        fig, points = draw_fit(
            points, row, fit_config, top=f"{region} | {polar} | {row.PhaseContrast}",
            right=f"{row.Lat} | LCT adjusted", xlabel="Δ" + row.FeatureLabel,
            ylabel=f"{row.ScaleLabel} ({endpoint})", additional_covariates=("LCT",),
        )
        filename = f"{row.Metric}_{row.FeatureOutput}_{row.Band}_{row.Polar}_{row.PhaseContrast}_{row.Lat}.pdf"
        target = output / "tdcs-adjusted/fit" / row.ScaleLabel / row.EndpointID / row.Domain / row.RegionValue / filename
        export(fig, target)
        record = row.to_dict()
        record.update(OutputPath=str(target), FigureType="tdcs-fit", Variant="baseline-lct-adjusted", Status="ok")
        records.append(record)
        points["OutputPath"] = str(target)
        point_tables.append(points)
        if index % 10 == 0:
            print(f"tDCS fits: {index}/{len(selected)}", flush=True)

    heat_config, plan = heatmap_plan(paths, tdcs)
    heat_cells = plan.coverage.copy()
    for _, row in plan.figures.iterrows():
        cells = heat_cells[heat_cells.FigureID.eq(row.FigureID)]
        fig = render_correlation_heatmap_figure(cells, row, heat_config)
        target = output / "tdcs-adjusted/heatmap" / row.ScaleLabel / row.Domain / Path(row.OutputPath).name
        export(fig, target)
        record = row.to_dict()
        record.update(OutputPath=str(target), FigureType="heatmap", Variant=row.FigureKind, Status="ok")
        records.append(record)
        heat_cells.loc[heat_cells.FigureID.eq(row.FigureID), "OutputPath"] = str(target)
        print(f"Heatmap: {row.ScaleLabel}/{row.RegionValue}/{row.FigureKind}/{row.Polar}/{row.Lat}", flush=True)
    tables = {"figures.csv": pd.DataFrame(records), "fit-points.csv": pd.concat(point_tables, ignore_index=True),
              "heatmap-cells.csv": heat_cells}
    for name, table in tables.items():
        target = output / name
        if target.exists():
            _trash_existing([target])
        atomic_csv(table, target, overwrite=False)
    print(f"Exported {len(records)} PDFs to {output}", flush=True)


if __name__ == "__main__":
    main()
