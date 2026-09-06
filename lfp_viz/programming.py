"""Visualdf adapters for saved LFP-programming intensity analyses."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from . import visualdf
from .config import VizConfig
from .correlation_fit import (
    _fit_annotation_format,
    _fit_annotation_location,
    _raw_linear_model_curve,
    render_correlation_fit_figure,
)
from .data import path_below, read_csv, require_columns
from .plan import VizPlan

ENDPOINTS = ("stn-only-stn", "combined-stn", "combined-snr")
ENDPOINT_SLUGS = ("stn-dbs-stn", "stn-snr-dbs-stn", "stn-snr-dbs-snr")
MODELS = ("M0", "M1", "M2")
PHASES = ("Early", "Late", "Post")
LATS = ("Ipsi", "Contra")
POLARS = ("Anodal", "Cathodal")
BANDS = ("delta", "theta", "alpha", "beta_low", "beta_high", "gamma_low", "gamma_high")
BAND_LABELS = ("δ", "θ", "α", "low-β", "high-β", "low-γ", "high-γ")
LOCAL = (
    ("periodic", "mean-scalar", "Periodic power"),
    ("raw_power", "mean-scalar", "Total power"),
    ("burst", "mean-scalar", "Burst amplitude"),
    ("burst", "duration-scalar", "Burst duration"),
    ("burst", "rate-scalar", "Burst rate"),
    ("burst", "occupancy-scalar", "Burst occupancy"),
)
CONNECTIVITY = tuple(
    (metric, "mean-scalar", label)
    for metric, label in (
        ("ciplv", "ciPLV"),
        ("imcoh_abs", "|ImCoh|"),
        ("wpli", "wPLI"),
        ("psi", "PSI"),
        ("trgc", "TRGC"),
    )
)
MODEL_LABELS = {
    "M0": "M0: unadjusted",
    "M1": "M1: baseline UPDRS-III MedOFF",
    "M2": "M2: baseline PDQ-39 total",
}
ADJUSTED_FIT_MODELS = (
    ("M1", "UPDRSIII_MedOFF", "UPDRS-III adjusted"),
    ("M2", "PDQ39_Total", "PDQ-39 adjusted"),
)
CONTACT_FOCUSED_LOW_BETA_KEYS = (
    (
        "combined-snr",
        "domain-local_metric-periodic_feature-output-mean-scalar_band-beta-low_"
        "region-SNr_polar-Anodal_phase-Early_lat-Contra",
    ),
    (
        "combined-snr",
        "domain-local_metric-burst_feature-output-occupancy-scalar_band-beta-low_"
        "region-SNr_polar-Cathodal_phase-Early_lat-Ipsi",
    ),
)
CVS = (0.0, 0.1, 0.2, 0.3, 0.5)
RESULT_COLUMNS = {
    "AnalysisID",
    "CorrelationID",
    "PredictorID",
    "EndpointID",
    "Model",
    "Domain",
    "Metric",
    "FeatureOutput",
    "FeatureLabel",
    "Band",
    "RegionValue",
    "PairRegion",
    "PairDirection",
    "Target",
    "Polar",
    "Phase",
    "PhaseContrast",
    "Lat",
    "rho",
    "p_holm",
    "n",
    "SubjectIDs",
    "Status",
    "CorrectionFamilySize",
    "CorrectionFamily",
    "Method",
    "Covariate",
    "AssessmentMonthAfterDBSActivation",
}


def validate_programming_config(data: dict) -> None:
    """Validate the fixed figure contract at the YAML boundary."""
    paths = data["paths"]
    for key in ("correlations", "model_input", "jitter_summary"):
        path_below(Path(paths[key]), Path(paths["stats_root"]), key)
    output_root = Path(paths["output_root"]).resolve()
    stats_root = Path(paths["stats_root"]).resolve()
    if output_root.is_relative_to(stats_root) or stats_root.is_relative_to(output_root):
        raise ValueError(
            "Programming visualization and statistics roots must be separate."
        )
    if data["factor_levels"] != {
        "Phase": list(PHASES),
        "Lat": list(LATS),
        "Polar": list(POLARS),
        "Model": list(MODELS),
    }:
        raise ValueError("Programming figure factor levels do not match the contract.")
    strategy = data["inputs"].get("strategy")
    if strategy not in {"region", "contact-matched"}:
        raise ValueError(
            "Programming figures require a region or contact-matched strategy."
        )
    if data["inputs"] != {
        "strategy": strategy,
        "stats_analysis_id": "lfp-programming-intensity",
        "effect_column": "rho",
        "p_column": "p_holm",
        "fit_style_config": "submission_clinical_correlation_adjacent_phase_fit.yaml",
        "selection": {"model": "M0", "status": "ok", "p_lt": 0.05},
    }:
        raise ValueError("Programming figures must use saved rho and Holm inference.")
    if list(data["labels"]["endpoint"]) != list(ENDPOINTS):
        raise ValueError(
            "Programming endpoint labels must preserve the three-endpoint order."
        )
    if list(data["labels"]["jitter_adjustment"]) != list(MODELS):
        raise ValueError(
            "Programming jitter adjustment labels must preserve the model order."
        )
    jitter_geometry = data["style"]["geometry_mm"]
    if {
        "jitter_box": jitter_geometry["jitter_box"],
        "jitter_panel_gap": jitter_geometry["jitter_panel_gap"],
        "jitter_adjustment_strip_height": jitter_geometry[
            "jitter_adjustment_strip_height"
        ],
        "jitter_right_strip_width": jitter_geometry["jitter_right_strip_width"],
        "jitter_strip_pad": jitter_geometry["jitter_strip_pad"],
    } != {
        "jitter_box": [30.0, 25.0],
        "jitter_panel_gap": [5.0, 0.0],
        "jitter_adjustment_strip_height": 4.5,
        "jitter_right_strip_width": 4.5,
        "jitter_strip_pad": 2.0,
    }:
        raise ValueError(
            "Programming jitter geometry does not match the fixed contract."
        )
    if data["style"]["heatmap"]["value_limits"] != [-1.0, 1.0]:
        raise ValueError("Programming heatmap limits must be [-1, 1].")
    if data["outputs"]["filename_template"] != "{FigureID}.pdf":
        raise ValueError(
            "Programming PDF filenames must use the fixed figure identity."
        )


def _load_sources(config: VizConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tables = [
        read_csv(Path(config["paths"][key]), key)
        for key in ("correlations", "model_input", "jitter_summary")
    ]
    correlations, points, jitter = tables
    for table in tables:
        require_columns(table, {"Strategy"}, "Programming visualization source")
        if not table.Strategy.eq(config["inputs"]["strategy"]).all():
            raise ValueError(
                "Programming visualization sources mix exploratory strategies."
            )
    require_columns(correlations, RESULT_COLUMNS, "Programming correlations")
    require_columns(
        points,
        {
            "EndpointID",
            "PredictorID",
            "ID",
            "Value",
            "ProgrammingIntensity",
            "SourcePath",
            "ProgrammingSourcePath",
            "BaselineSourcePath",
        },
        "Programming model inputs",
    )
    require_columns(
        jitter,
        (RESULT_COLUMNS - {"p_holm"})
        | {
            "JitterCV",
            "rho_median",
            "rho_p025",
            "rho_p975",
            "Iterations",
            "Seed",
            "PrRhoNegative",
            "NominalSignAgreement",
            "DoseRankRhoMedian",
            "DoseRankRhoP025",
            "DoseRankRhoP975",
            "ValidIterations",
            "ValidFraction",
            "JitterStatus",
        },
        "Programming jitter summaries",
    )
    if any({"p_raw", "q", "LOSO"}.intersection(table.columns) for table in tables):
        raise ValueError(
            "Programming inputs must not include raw p, q, or LOSO columns."
        )
    if not correlations.AnalysisID.eq(config.stats_analysis_id).all():
        raise ValueError(
            "Programming correlations have an unexpected analysis identity."
        )
    identity = [
        "EndpointID",
        "Model",
        "Domain",
        "Metric",
        "FeatureOutput",
        "Band",
        "Polar",
        "Phase",
        "Lat",
    ]
    if (
        correlations.CorrelationID.duplicated().any()
        or correlations.duplicated(identity).any()
    ):
        raise ValueError(
            "Programming correlations contain duplicate planned positions."
        )
    if len(correlations) != 8532 or not correlations.CorrectionFamilySize.eq(6).all():
        raise ValueError("Expected 8,532 positions with six-comparison Holm families.")
    for column, expected in (
        ("EndpointID", ENDPOINTS),
        ("Model", MODELS),
        ("Phase", PHASES),
        ("Lat", LATS),
        ("Polar", POLARS),
    ):
        if set(correlations[column]) != set(expected):
            raise ValueError(f"Unexpected programming correlation {column} values.")
    local = correlations.Domain.eq("local")
    if (
        not correlations.loc[local, "RegionValue"]
        .eq(correlations.loc[local, "Target"])
        .all()
    ):
        raise ValueError("Local LFP regions must match the dose target.")
    for metric, _, _ in CONNECTIVITY:
        selected = correlations.loc[
            correlations.Domain.eq("connectivity") & correlations.Metric.eq(metric)
        ]
        directed = metric in {"psi", "trgc"}
        if not (
            selected.PairRegion.eq("SNr→STN" if directed else "SNr-STN").all()
            and selected.PairDirection.eq(
                "directed" if directed else "undirected"
            ).all()
        ):
            raise ValueError(f"Unexpected connectivity pair/direction for {metric}.")
    ok = correlations.Status.eq("ok")
    if (
        not np.isfinite(correlations.loc[ok, ["rho", "p_holm"]]).all().all()
        or correlations.loc[~ok, ["rho", "p_holm"]].notna().any().any()
    ):
        raise ValueError(
            "Finite effects and p values must agree with estimability status."
        )
    if points.duplicated(["EndpointID", "PredictorID", "ID"]).any():
        raise ValueError(
            "Model inputs contain repeated patients within an analysis cell."
        )
    if jitter.duplicated(["CorrelationID", "JitterCV"]).any():
        raise ValueError("Jitter summaries contain duplicate correlation/CV positions.")
    return correlations, points, jitter


def _heatmap_cells(correlations: pd.DataFrame, config: VizConfig) -> pd.DataFrame:
    """Assign every source result to exactly one visualdf matrix position."""
    cells = correlations.copy()
    combine_local_polarities = config["inputs"]["strategy"] == "contact-matched"
    assignments = []
    for row in cells.itertuples(index=False):
        endpoint_index = ENDPOINTS.index(row.EndpointID)
        phase_index = PHASES.index(row.Phase)
        lat_index = LATS.index(row.Lat)
        is_local = row.Domain == "local"
        if is_local:
            view = "local" if combine_local_polarities else f"{row.Polar}-local"
        else:
            view = "connectivity"
        polarity_offset = (
            POLARS.index(row.Polar) * 3
            if not is_local or combine_local_polarities
            else 0
        )
        if is_local and row.Metric == "aperiodic":
            block = "aperiodic"
            panel_row, panel_column = lat_index, phase_index + polarity_offset
            matrix_row = ("exponent", "offset").index(row.Band)
            matrix_column = endpoint_index
            row_label = row.Band.title()
            column_label = config["labels"]["endpoint"][row.EndpointID]
        else:
            block = "banded"
            metrics = LOCAL if is_local else CONNECTIVITY
            matrix_column = [(m, f) for m, f, _ in metrics].index(
                (row.Metric, row.FeatureOutput)
            )
            matrix_row = BANDS.index(row.Band)
            row_label, column_label = BAND_LABELS[matrix_row], metrics[matrix_column][2]
            panel_row = endpoint_index * 2 + lat_index
            panel_column = phase_index + polarity_offset
        assignments.append(
            {
                "FigureID": f"model-{row.Model}_view-{view}_rho-heatmap",
                "Block": block,
                "PanelRow": panel_row,
                "PanelColumn": panel_column,
                "MatrixRow": matrix_row,
                "MatrixColumn": matrix_column,
                "MatrixRowLabel": row_label,
                "MatrixColumnLabel": column_label,
                "EndpointLabel": config["labels"]["endpoint"][row.EndpointID],
            }
        )
    cells = pd.concat([cells.reset_index(drop=True), pd.DataFrame(assignments)], axis=1)
    key = ["FigureID", "Block", "PanelRow", "PanelColumn", "MatrixRow", "MatrixColumn"]
    if cells.duplicated(key).any():
        raise ValueError("Heatmap mapping would overwrite a result position.")
    sizes = cells.groupby("FigureID").size()
    expected_figure_count = 6 if combine_local_polarities else 9
    expected_local_size = 1584 if combine_local_polarities else 792
    if len(sizes) != expected_figure_count or any(
        count != (1260 if "connectivity" in name else expected_local_size)
        for name, count in sizes.items()
    ):
        raise ValueError("Heatmap layout does not cover the prescribed complete grids.")
    cells["CorrelationSourcePath"] = str(config.correlations_path)
    return cells.sort_values(key).reset_index(drop=True)


def build_programming_plan(config: VizConfig) -> VizPlan:
    """Adapt stored inference and patient data without statistical recomputation."""
    correlations, model_inputs, jitter = _load_sources(config)
    cells = _heatmap_cells(correlations, config)
    figures, fit_points, fit_curves, jitter_rows = [], [], [], []
    model_data, result_tables = {}, {}
    for figure_id, group in cells.groupby("FigureID", sort=False):
        row = group.iloc[0]
        figures.append(
            {
                "FigureID": figure_id,
                "PlotKind": "heatmap",
                "Model": row.Model,
                "Domain": row.Domain,
                "Polar": "Both" if group.Polar.nunique() == 2 else row.Polar,
                "n_positions": len(group),
                "OutputPath": str(
                    config.output_root
                    / "heatmap"
                    / row.Model
                    / config["outputs"]["filename_template"].format(FigureID=figure_id)
                ),
            }
        )
    selected = correlations.loc[
        correlations.Model.eq("M0")
        & correlations.Status.eq("ok")
        & correlations.p_holm.lt(0.05)
    ].copy()
    selected["EndpointOrder"] = selected.EndpointID.map(dict(zip(ENDPOINTS, range(3))))
    selected = selected.sort_values(
        [
            "EndpointOrder",
            "Domain",
            "Metric",
            "FeatureOutput",
            "Band",
            "Polar",
            "Phase",
            "Lat",
        ]
    )
    for _, row in selected.iterrows():
        slug = ENDPOINT_SLUGS[ENDPOINTS.index(row.EndpointID)]
        identity = f"endpoint-{slug}_{row.Domain}_{row.Metric}_{row.FeatureOutput}_{row.Band}_{row.Polar}_{row.Phase}_{row.Lat}"
        fit_id = f"{identity}_ols-fit"
        points = model_inputs.loc[
            model_inputs.EndpointID.eq(row.EndpointID)
            & model_inputs.PredictorID.eq(row.PredictorID)
        ].copy()
        ids = str(row.SubjectIDs).split(";")
        points = points.loc[points.ID.isin(ids)].sort_values("ID")
        if set(points.ID) != set(ids) or len(points) != row.n:
            raise ValueError(
                f"Fit patient set differs from nominal inference: {row.CorrelationID}"
            )
        if not np.isfinite(points[["Value", "ProgrammingIntensity"]]).all().all():
            raise ValueError(
                f"Selected fit contains nonfinite inputs: {row.CorrelationID}"
            )
        points["FigureID"], points["CorrelationID"] = fit_id, row.CorrelationID
        curve = _raw_linear_model_curve(
            points,
            method="spearman",
            x_column="Value",
            y_column="ProgrammingIntensity",
            baseline_reference=None,
            confidence_level=0.95,
        )
        curve["FigureID"], curve["CorrelationID"] = fit_id, row.CorrelationID
        curve["EndpointID"] = row.EndpointID
        curve["IntervalType"] = "pointwise_95pct_OLS_mean_response"
        curve["Strategy"] = config["inputs"]["strategy"]
        model_data[fit_id] = points
        result_tables[(fit_id, "fit", "curve")] = curve
        fit_points.append(points)
        fit_curves.append(curve)
        metadata = row.to_dict()
        metadata["EndpointLabel"] = config["labels"]["endpoint"][row.EndpointID]
        metadata["XLabel"] = f"Δ{row.FeatureLabel} ({row.PhaseContrast})"
        metadata["YLabel"] = metadata["EndpointLabel"] + "\n(×10⁴ V² × μs × Hz)"
        metadata["DoseDisplayDivisor"] = 10000
        metadata["StratumLabel"] = (
            f"{config['labels']['region'][row.RegionValue]} | {config['labels']['polarity'][row.Polar]} | {row.PhaseContrast}"
        )
        metadata["ConfidenceLevel"] = 0.95
        points["StratumLabel"] = metadata["StratumLabel"]
        figures.append(
            {
                **metadata,
                "FigureID": fit_id,
                "PlotKind": "fit",
                "n_positions": len(points),
                "OutputPath": str(
                    config.output_root
                    / "fit"
                    / config["outputs"]["filename_template"].format(FigureID=fit_id)
                ),
            }
        )
        include_jitter = (
            config["inputs"]["strategy"] != "contact-matched"
            or (row.EndpointID, row.PredictorID) in CONTACT_FOCUSED_LOW_BETA_KEYS
        )
        if include_jitter:
            jitter_id = f"{identity}_jitter"
            summaries = jitter.loc[
                jitter.EndpointID.eq(row.EndpointID)
                & jitter.PredictorID.eq(row.PredictorID)
            ].copy()
            expected = {(model, cv) for model in MODELS for cv in CVS}
            if set(zip(summaries.Model, summaries.JitterCV)) != expected:
                raise ValueError(
                    f"Incomplete selected jitter model/CV set: {row.CorrelationID}"
                )
            nominal = correlations.loc[
                correlations.EndpointID.eq(row.EndpointID)
                & correlations.PredictorID.eq(row.PredictorID)
            ]
            paired = summaries.merge(
                nominal[["CorrelationID", "rho", "n", "SubjectIDs", "Status"]],
                on="CorrelationID",
                validate="many_to_one",
                suffixes=("", "_source"),
            )
            if not (
                np.allclose(paired.rho, paired.rho_source, equal_nan=True)
                and paired.n.eq(paired.n_source).all()
                and paired.SubjectIDs.eq(paired.SubjectIDs_source).all()
                and paired.Status.eq(paired.Status_source).all()
            ):
                raise ValueError(
                    "Selected jitter summaries disagree with nominal inference."
                )
            summaries["FigureID"] = jitter_id
            summaries["JitterSourcePath"] = config["paths"]["jitter_summary"]
            model_data[jitter_id] = summaries
            jitter_rows.append(summaries)
            figures.append(
                {
                    **metadata,
                    "FigureID": jitter_id,
                    "PlotKind": "jitter",
                    "Model": "M0;M1;M2",
                    "n_positions": len(summaries),
                    "OutputPath": str(
                        config.output_root
                        / "jitter"
                        / config["outputs"]["filename_template"].format(
                            FigureID=jitter_id
                        )
                    ),
                }
            )

    endpoint_fit_limits: dict[str, tuple[float, float]] = {}
    if fit_points:
        nominal_points = pd.concat(fit_points, ignore_index=True)
        nominal_curves = pd.concat(fit_curves, ignore_index=True)
        for endpoint, group in nominal_curves.groupby("EndpointID"):
            observed = nominal_points.loc[
                nominal_points.EndpointID.eq(endpoint), "ProgrammingIntensity"
            ]
            lower = min(float(group["lower.CL"].min()), float(observed.min()))
            upper = max(float(group["upper.CL"].max()), float(observed.max()))
            margin = 0.08 * (upper - lower)
            endpoint_fit_limits[str(endpoint)] = (lower - margin, upper + margin)

    if config["inputs"]["strategy"] == "contact-matched":
        selected_keys = set(zip(selected.EndpointID, selected.PredictorID))
        for endpoint, predictor_id in CONTACT_FOCUSED_LOW_BETA_KEYS:
            adjusted_rows = correlations.loc[
                correlations.EndpointID.eq(endpoint)
                & correlations.PredictorID.eq(predictor_id)
                & correlations.Model.isin([item[0] for item in ADJUSTED_FIT_MODELS])
            ].copy()
            if set(adjusted_rows.Model) != {item[0] for item in ADJUSTED_FIT_MODELS}:
                raise ValueError(
                    f"Incomplete adjusted-fit model set: {endpoint}/{predictor_id}"
                )
            if (endpoint, predictor_id) not in selected_keys:
                continue

            base_row = adjusted_rows.loc[adjusted_rows.Model.eq("M1")].iloc[0]
            slug = ENDPOINT_SLUGS[ENDPOINTS.index(endpoint)]
            identity = (
                f"endpoint-{slug}_{base_row.Domain}_{base_row.Metric}_"
                f"{base_row.FeatureOutput}_{base_row.Band}_{base_row.Polar}_"
                f"{base_row.Phase}_{base_row.Lat}"
            )
            fit_id = f"{identity}_adjusted-ols-fit"
            panel_points, panel_curves, panel_statistics = [], [], []
            for model, covariate, adjustment_label in ADJUSTED_FIT_MODELS:
                result = adjusted_rows.loc[adjusted_rows.Model.eq(model)].iloc[0]
                if (
                    result.Status != "ok"
                    or result.Method != "partial_spearman"
                    or result.Covariate != covariate
                ):
                    raise ValueError(
                        f"Adjusted-fit inference is not estimable as specified: "
                        f"{result.CorrelationID}"
                    )
                ids = str(result.SubjectIDs).split(";")
                points = model_inputs.loc[
                    model_inputs.EndpointID.eq(endpoint)
                    & model_inputs.PredictorID.eq(predictor_id)
                    & model_inputs.ID.isin(ids)
                ].copy()
                points = points.sort_values("ID")
                if set(points.ID) != set(ids) or len(points) != result.n:
                    raise ValueError(
                        f"Adjusted-fit patient set differs from inference: "
                        f"{result.CorrelationID}"
                    )
                points["Baseline"] = pd.to_numeric(points[covariate], errors="coerce")
                if (
                    not np.isfinite(
                        points[["Value", "ProgrammingIntensity", "Baseline"]]
                    )
                    .all()
                    .all()
                ):
                    raise ValueError(
                        f"Adjusted fit contains nonfinite inputs: "
                        f"{result.CorrelationID}"
                    )
                baseline_reference = float(points.Baseline.median())
                points["FigureID"] = fit_id
                points["CorrelationID"] = result.CorrelationID
                points["Model"] = model
                points["Covariate"] = covariate
                points["AdjustmentLabel"] = adjustment_label
                points["BaselineReference"] = baseline_reference
                points["Lat"] = result.Lat
                curve = _raw_linear_model_curve(
                    points,
                    method="partial_spearman",
                    x_column="Value",
                    y_column="ProgrammingIntensity",
                    baseline_reference=baseline_reference,
                    confidence_level=0.95,
                )
                curve["FigureID"] = fit_id
                curve["CorrelationID"] = result.CorrelationID
                curve["EndpointID"] = endpoint
                curve["Model"] = model
                curve["Covariate"] = covariate
                curve["AdjustmentLabel"] = adjustment_label
                curve["BaselineReference"] = baseline_reference
                curve["Lat"] = result.Lat
                curve["IntervalType"] = "pointwise_95pct_OLS_mean_response"
                curve["Strategy"] = config["inputs"]["strategy"]
                panel_points.append(points)
                panel_curves.append(curve)
                panel_statistics.append(
                    {
                        "FigureID": fit_id,
                        "CorrelationID": result.CorrelationID,
                        "Model": model,
                        "Covariate": covariate,
                        "AdjustmentLabel": adjustment_label,
                        "BaselineReference": baseline_reference,
                        "Lat": result.Lat,
                        "n": int(result.n),
                        "SubjectIDs": result.SubjectIDs,
                        "rho": float(result.rho),
                        "p_holm": float(result.p_holm),
                    }
                )

            adjusted_points = pd.concat(panel_points, ignore_index=True)
            adjusted_curves = pd.concat(panel_curves, ignore_index=True)
            statistics = pd.DataFrame(panel_statistics)
            if endpoint not in endpoint_fit_limits:
                lower = min(
                    float(adjusted_curves["lower.CL"].min()),
                    float(adjusted_points.ProgrammingIntensity.min()),
                )
                upper = max(
                    float(adjusted_curves["upper.CL"].max()),
                    float(adjusted_points.ProgrammingIntensity.max()),
                )
                margin = 0.08 * (upper - lower)
                endpoint_fit_limits[endpoint] = (lower - margin, upper + margin)
            y_min, y_max = endpoint_fit_limits[endpoint]
            model_data[fit_id] = adjusted_points
            result_tables[(fit_id, "fit", "curve")] = adjusted_curves
            result_tables[(fit_id, "fit", "statistics")] = statistics
            fit_points.append(adjusted_points)
            fit_curves.append(adjusted_curves)

            metadata = base_row.to_dict()
            endpoint_label = config["labels"]["endpoint"][endpoint]
            metadata.update(
                {
                    "Model": "M1;M2",
                    "Method": "partial_spearman",
                    "Covariate": "UPDRSIII_MedOFF;PDQ39_Total",
                    "CorrelationID": ";".join(statistics.CorrelationID),
                    "rho": np.nan,
                    "p_holm": np.nan,
                    "EndpointLabel": endpoint_label,
                    "XLabel": (
                        f"Δ{base_row.RegionValue} {base_row.FeatureLabel} "
                        f"({base_row.PhaseContrast}; {base_row.Polar})"
                    ),
                    "YLabel": endpoint_label + "\n(×10⁴ V² × μs × Hz)",
                    "DoseDisplayDivisor": 10000,
                    "StratumLabel": (
                        f"{config['labels']['region'][base_row.RegionValue]} | "
                        f"{config['labels']['polarity'][base_row.Polar]} | "
                        f"{base_row.PhaseContrast}"
                    ),
                    "ConfidenceLevel": 0.95,
                    "Layout": "adjusted_two_panel",
                    "PanelModels": "M1;M2",
                    "PanelLabels": ";".join(
                        label.replace("\n", " ") for _, _, label in ADJUSTED_FIT_MODELS
                    ),
                    "FitFormula": "ProgrammingIntensity ~ 1 + Value + Baseline",
                    "BaselineReferenceMode": "median",
                    "YMin": y_min,
                    "YMax": y_max,
                }
            )
            figures.append(
                {
                    **metadata,
                    "FigureID": fit_id,
                    "PlotKind": "fit",
                    "n_positions": len(adjusted_points),
                    "OutputPath": str(
                        config.output_root
                        / "fit"
                        / "adjusted"
                        / config["outputs"]["filename_template"].format(FigureID=fit_id)
                    ),
                }
            )

    points_table = (
        pd.concat(fit_points, ignore_index=True)
        if fit_points
        else model_inputs.iloc[:0].assign(
            FigureID="", CorrelationID="", StratumLabel=""
        )
    )
    curves_table = (
        pd.concat(fit_curves, ignore_index=True)
        if fit_curves
        else pd.DataFrame(
            columns=[
                "FigureID",
                "CorrelationID",
                "EndpointID",
                "Strategy",
                "Value",
                "emmean",
                "lower.CL",
                "upper.CL",
                "IntervalType",
            ]
        )
    )
    jitter_table = (
        pd.concat(jitter_rows, ignore_index=True)
        if jitter_rows
        else jitter.iloc[:0].assign(FigureID="", JitterSourcePath="")
    )
    figures = pd.DataFrame(figures)
    # Reuse the unadjusted endpoint limits so existing M0 fit geometry is unchanged.
    for column in ("YMin", "YMax"):
        if column not in figures:
            figures[column] = np.nan
    for endpoint, (lower, upper) in endpoint_fit_limits.items():
        mask = (
            figures.PlotKind.eq("fit")
            & figures.EndpointID.eq(endpoint)
            & figures.YMin.isna()
        )
        figures.loc[mask, "YMin"] = lower
        figures.loc[mask, "YMax"] = upper
    significant_keys = correlations.loc[
        correlations.Status.eq("ok") & correlations.p_holm.lt(0.05),
        ["EndpointID", "PredictorID"],
    ].drop_duplicates()
    comparison = correlations.merge(
        significant_keys, on=["EndpointID", "PredictorID"], validate="many_to_one"
    )
    comparison["EndpointLabel"] = comparison.EndpointID.map(
        config["labels"]["endpoint"]
    )
    figures["AnalysisID"], figures["TestID"], figures["Status"] = (
        config.analysis_id,
        "programming-intensity",
        "planned",
    )
    figures["Strategy"] = config["inputs"]["strategy"]
    provenance = pd.DataFrame(
        [
            {"Input": key, "Path": config["paths"][key], "Rows": len(table)}
            for key, table in zip(
                ("correlations", "model_input", "jitter_summary"),
                (correlations, model_inputs, jitter),
            )
        ]
    )
    provenance.loc[len(provenance)] = {
        "Input": "fit_style_config",
        "Path": str(config.path.parent / config["inputs"]["fit_style_config"]),
        "Rows": np.nan,
    }
    provenance["Strategy"] = config["inputs"]["strategy"]
    data_root = config.output_root / "data"
    return VizPlan(
        figures=figures,
        coverage=cells,
        model_data=model_data,
        result_tables=result_tables,
        submission_tables={
            data_root / "heatmap_cells.csv": cells,
            data_root / "fit_points.csv": points_table,
            data_root / "ols_curves.csv": curves_table,
            data_root / "jitter_selected.csv": jitter_table,
            data_root / "model_comparison.csv": comparison,
            config.manifest_root / "inputs.csv": provenance,
        },
    )


def _matrix(panel: pd.DataFrame, value: str) -> pd.DataFrame:
    matrix = (
        panel.pivot(index="MatrixRow", columns="MatrixColumn", values=value)
        .sort_index()
        .sort_index(axis=1)
    )
    matrix.index = (
        panel.drop_duplicates("MatrixRow")
        .set_index("MatrixRow")
        .loc[matrix.index, "MatrixRowLabel"]
    )
    matrix.columns = (
        panel.drop_duplicates("MatrixColumn")
        .set_index("MatrixColumn")
        .loc[matrix.columns, "MatrixColumnLabel"]
    )
    return matrix


def _extend_canvas(fig: plt.Figure, *, bottom_mm: float, top_mm: float) -> None:
    """Reserve matrix/header space without scaling the visualdf cell geometry."""
    width, old_height = fig.get_size_inches()
    bottom = bottom_mm / 25.4
    height = old_height + bottom + top_mm / 25.4
    for ax in fig.axes:
        x, y, w, h = ax.get_position().bounds
        ax.set_position(
            (x, (y * old_height + bottom) / height, w, h * old_height / height)
        )
    for artist in fig.texts:
        x, y = artist.get_position()
        artist.set_position((x, (y * old_height + bottom) / height))
    fig.set_size_inches(width, height, forward=False)


def _mark_na(ax: plt.Axes, values: pd.DataFrame, color: str) -> None:
    ax.set_facecolor(color)
    for y, x in np.argwhere(~np.isfinite(values.to_numpy(dtype=float))):
        ax.text(
            x + 0.5,
            y + 0.5,
            "NA",
            ha="center",
            va="center",
            fontsize=4,
            color="#555555",
        )


def _right_strip(fig, bounds, label, style, *, horizontal=False):
    """Add grouped or short-matrix strips using the visualdf strip styling."""
    strip = fig.add_axes(bounds)
    strip._visualdf_strip_axis = True
    strip.set_facecolor(style["strips"]["right_background"])
    strip.text(
        0.5,
        0.5,
        label,
        ha="center",
        va="center",
        rotation=0 if horizontal else -90,
        fontsize=style["font"]["strip_pt"],
        fontfamily=style["font"]["family"],
    )
    strip.set_xticks([])
    strip.set_yticks([])
    for spine in strip.spines.values():
        spine.set_visible(False)


def _polarity_strip(fig, bounds, polarity, label, style):
    """Add an adjustment-and-polarity strip above three adjacent Phase panels."""
    strip = fig.add_axes(bounds)
    strip._visualdf_strip_axis = True
    strip._programming_polarity = polarity
    strip.set_facecolor(style["strips"]["top_background"])
    strip.text(
        0.5,
        0.5,
        label,
        ha="center",
        va="center",
        fontsize=style["font"]["strip_pt"],
        fontfamily=style["font"]["family"],
        fontweight="bold",
    )
    strip.set_xticks([])
    strip.set_yticks([])
    for spine in strip.spines.values():
        spine.set_visible(False)
    return strip


def render_programming_heatmap(
    cells: pd.DataFrame, row: pd.Series, config: VizConfig
) -> plt.Figure:
    """Compose local footers or polarity super-headers on visualdf grids."""
    style, labels = config["style"], config["labels"]
    font = style["font"]["family"]
    connectivity = row.Domain == "connectivity"
    both_polarities = cells.Polar.nunique() == 2
    banded = cells.loc[cells.Block.eq("banded")]
    count = 6 if both_polarities else 3
    panels = [
        [
            banded.loc[banded.PanelRow.eq(i) & banded.PanelColumn.eq(j)]
            for j in range(count)
        ]
        for i in range(6)
    ]
    values = [[_matrix(panel, "rho") for panel in panels_row] for panels_row in panels]
    ps = [[_matrix(panel, "p_holm") for panel in panels_row] for panels_row in panels]
    endpoint_labels = [
        labels["endpoint"][endpoint].split(": ") for endpoint in ENDPOINTS
    ]
    row_labels = [f"{dose} | {lat}" for _, dose in endpoint_labels for lat in LATS]
    top_labels = [labels["phase_contrast"][phase] for phase in PHASES] * (
        2 if both_polarities else 1
    )
    fig = visualdf.plot_well_heatmap_grid_df(
        values,
        ps,
        label_top=top_labels,
        label_right=row_labels,
        font_family=font,
        cmap=getattr(visualdf.cm, style["heatmap"]["colormap"]).with_extremes(
            bad=style["heatmap"]["na_color"]
        ),
        vmin=-1,
        vmax=1,
        cellsize=tuple(style["geometry_mm"]["cell_size"]),
        panel_gap_mm=tuple(style["geometry_mm"]["panel_gap"]),
        label_top_height_mm=4.5,
        label_right_width_mm=4.5,
        strip_pad_mm=2,
        label_top_bg_color=style["strips"]["top_background"],
        label_right_bg_color=style["strips"]["right_background"],
        label_fontsize=style["font"]["strip_pt"],
        tick_label_fontsize=style["font"]["tick_pt"],
        axis_label_fontsize=style["font"]["colorbar_pt"],
        colorbar_label="Spearman's ρ" if row.Model == "M0" else "Partial Spearman's ρ",
        colorbar_ticks=[-1, -0.5, 0, 0.5, 1],
        colorbar_length_mm=None,
        colorbar_width_mm=3,
        colorbar_pad_mm=2 + 4.5 + 3,
        cbar_label_offset_mm=5,
        xtick_rotation=90,
        show_x_ticklabels="bottom",
        show_y_ticklabels="left",
        grid_lines=None,
        gline_color=style["grid"]["color"],
        gline_width=style["grid"]["width_pt"],
        gline_alpha=style["grid"]["alpha"],
        show_p=True,
        hide_ns=True,
        p_text_color="white",
        p_text_size=style["font"]["star_pt"],
        dpi=style["dpi"],
        transparent=style["transparent"],
    )
    axes = fig.axes[: 6 * count]
    for ax, matrix in zip(axes, (m for matrices in values for m in matrices)):
        _mark_na(ax, matrix, style["heatmap"]["na_color"])
    _extend_canvas(
        fig,
        bottom_mm=0 if connectivity else 70,
        top_mm=6.5 if both_polarities else 0,
    )
    colorbar = next(ax for ax in fig.axes if getattr(ax, "_colorbar", None) is not None)
    position = colorbar.get_position()
    bottom, top = axes[-1].get_position().y0, axes[0].get_position().y1
    colorbar.set_position([position.x0, bottom, position.width, top - bottom])
    colorbar_label = "Spearman's ρ" if row.Model == "M0" else "Partial Spearman's ρ"
    next(text for text in fig.texts if text.get_text() == colorbar_label).set_y(
        (bottom + top) / 2
    )
    width_mm = fig.get_figwidth() * 25.4
    for start, stop, dbs in (
        (0, 2, endpoint_labels[0][0]),
        (2, 6, endpoint_labels[1][0]),
    ):
        top = axes[(start + 1) * count - 1].get_position()
        bottom = axes[stop * count - 1].get_position()
        _right_strip(
            fig,
            [
                top.x1 + (2 + 4.5 + 2) / width_mm,
                bottom.y0,
                4.5 / width_mm,
                top.y1 - bottom.y0,
            ],
            dbs,
            style,
        )
    if both_polarities:
        height_mm = fig.get_figheight() * 25.4
        strip_y = axes[0].get_position().y1 + (2 + 4.5 + 2) / height_mm
        adjustment_label = labels["jitter_adjustment"][row.Model]
        for start, polarity in ((0, "Anodal"), (3, "Cathodal")):
            left, right = (
                axes[start].get_position().x0,
                axes[start + 2].get_position().x1,
            )
            _polarity_strip(
                fig,
                [left, strip_y, right - left, 4.5 / height_mm],
                polarity,
                f"{adjustment_label} | {labels['polarity'][polarity]}",
                style,
            )
    if not connectivity:
        footer = cells.loc[cells.Block.eq("aperiodic")]
        height_mm = fig.get_figheight() * 25.4
        footer_top = axes[-1].get_position().y0 - 28 / height_mm
        for i in range(2):
            for j in range(count):
                panel = footer.loc[footer.PanelRow.eq(i) & footer.PanelColumn.eq(j)]
                matrix, p_matrix = _matrix(panel, "rho"), _matrix(panel, "p_holm")
                anchor = axes[j].get_position()
                footer_width = (
                    3
                    * style["geometry_mm"]["cell_size"][0]
                    / (fig.get_figwidth() * 25.4)
                )
                ax = fig.add_axes(
                    [
                        (anchor.x0 + anchor.x1 - footer_width) / 2,
                        footer_top - (i * 9 + 6) / height_mm,
                        footer_width,
                        6 / height_mm,
                    ]
                )
                visualdf._draw_well_heatmap_panel(
                    ax,
                    df_value=matrix,
                    z_values=matrix.to_numpy(dtype=float),
                    df_p_aligned=p_matrix,
                    cmap=axes[0].collections[0].cmap,
                    norm=mpl.colors.Normalize(-1, 1),
                    xtick_rotation=90,
                    ytick_rotation=0,
                    show_x_ticklabels=i == 1,
                    show_y_ticklabels=j == 0,
                    grid_lines=None,
                    gline_color=style["grid"]["color"],
                    gline_width=style["grid"]["width_pt"],
                    gline_alpha=style["grid"]["alpha"],
                    tick_label_fontsize=6,
                    show_p=True,
                    hide_ns=True,
                    p_text_color="white",
                    p_text_size=7,
                )
                _mark_na(ax, matrix, style["heatmap"]["na_color"])
                if j == count - 1:
                    position = ax.get_position()
                    _right_strip(
                        fig,
                        [
                            position.x1 + 2 / width_mm,
                            position.y0,
                            12 / width_mm,
                            position.height,
                        ],
                        LATS[i],
                        style,
                        horizontal=True,
                    )
    return fig


def _jitter_feature_label(row: pd.Series) -> str:
    metrics = LOCAL if row.Domain == "local" else CONNECTIVITY
    if row.Metric == "aperiodic":
        return row.Band.title()
    metric = next(
        label for m, f, label in metrics if (m, f) == (row.Metric, row.FeatureOutput)
    )
    return f"{BAND_LABELS[BANDS.index(row.Band)]} {metric}"


def _jitter_right_strip_label(row: pd.Series, config: VizConfig) -> str:
    region = (
        config["labels"]["region"][row.RegionValue]
        if row.Domain == "local"
        else config["labels"]["region"][row.PairRegion]
    )
    return f"{region} | {config['labels']['polarity'][row.Polar]} | {row.Lat}"


def render_programming_fit(
    points: pd.DataFrame, row: pd.Series, config: VizConfig
) -> plt.Figure:
    """Use the clinical Spearman renderer with the inherited reference style."""
    fit_config = VizConfig(
        path=config.path,
        data={
            **config.data,
            "inputs": {
                **config["inputs"],
                "x_column": "Value",
                "y_column": "ProgrammingIntensity",
            },
            "style": config["style"]["fit_reference"],
        },
    )
    fig = render_correlation_fit_figure(points, row, fit_config)
    fig.axes[0].set_ylim(row.YMin, row.YMax)
    fig.axes[0].yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value / row.DoseDisplayDivisor:g}")
    )
    return fig


def render_programming_adjusted_fit(
    points: pd.DataFrame,
    curves: pd.DataFrame,
    statistics: pd.DataFrame,
    row: pd.Series,
    config: VizConfig,
) -> plt.Figure:
    """Render the two reader-labeled covariate-adjusted panels via visualdf."""

    style = config["style"]["fit_reference"]
    font = style["font"]
    geometry = style["geometry_mm"]
    strips = style["strips"]
    point_style = style["points"]
    fit_style = style["fit"]
    axes_style = style["axes"]
    adjustment_labels = {model: label for model, _, label in ADJUSTED_FIT_MODELS}
    points = points.assign(AdjustmentLabel=points.Model.map(adjustment_labels))
    curves = curves.assign(AdjustmentLabel=curves.Model.map(adjustment_labels))
    panel_labels = list(adjustment_labels.values())
    lat = str(row.Lat)
    fig = visualdf.plot_triple_interaction_fit(
        df=points,
        curve=curves,
        slope=pd.DataFrame(),
        value_col="ProgrammingIntensity",
        x_var="Value",
        panel_var="AdjustmentLabel",
        facet_var="Lat",
        color_var=point_style["color_var"],
        panel_levels=panel_labels,
        facet_levels=[lat],
        x_label=str(row.XLabel),
        y_label=str(row.YLabel),
        y_limits=(float(row.YMin), float(row.YMax)),
        font_family=font["family"],
        palette=point_style["palette"],
        jitter_alpha=point_style["alpha"],
        jitter_size=point_style["size"],
        curve_line_width=fit_style["line_width"],
        curve_line_color=fit_style["line_color"],
        ribbon_alpha=fit_style["ribbon_alpha"],
        show_ribbon=fit_style["show_ribbon"],
        grid=axes_style["grid"],
        dpi=style["dpi"],
        show_top_right_axes=axes_style["show_top_right"],
        label_fontsize=font["strip_pt"],
        label_top_bg_color=strips["top_background"],
        label_right_bg_color=strips["right_background"],
        label_text_color=strips["text_color"],
        label_fontweight=strips["font_weight"],
        strip_top_height_mm=geometry["strip_top_height"],
        strip_right_width_mm=geometry["strip_right_width"],
        strip_pad_mm=geometry["strip_pad"],
        title_fontsize=font["axis_pt"],
        axis_label_fontsize=font["axis_pt"],
        tick_label_fontsize=font["tick_pt"],
        legend_loc=style["legend"]["location"],
        boxsize=tuple(geometry["boxsize"]),
        panel_gap=tuple(config["style"]["geometry_mm"]["panel_gap"]),
        x_label_offset_mm=geometry["x_label_offset"],
        y_label_offset_mm=geometry["y_label_offset"],
        transparent=style["transparent"],
    )

    annotation = style["annotation"]
    anchors = {
        "upper left": (0.05, 0.95, "left", "top"),
        "upper right": (0.95, 0.95, "right", "top"),
        "lower left": (0.05, 0.05, "left", "bottom"),
        "lower right": (0.95, 0.05, "right", "bottom"),
    }
    for axis, (model, _, _) in zip(
        fig.axes[: len(ADJUSTED_FIT_MODELS)], ADJUSTED_FIT_MODELS
    ):
        panel_curve = curves.loc[curves.Model.eq(model)]
        panel_statistic = statistics.loc[statistics.Model.eq(model)].iloc[0]
        p_value = float(panel_statistic.p_holm)
        text = _fit_annotation_format(p_value, annotation).format(
            beta=float(panel_statistic.rho),
            p=p_value,
            stars=visualdf.p_to_stars(p_value),
        )
        location = (
            _fit_annotation_location(panel_curve, "Value")
            if annotation["location"] == "auto_by_fit_slope"
            else str(annotation["location"])
        )
        x, y, horizontal, vertical = anchors[location]
        axis.text(
            x,
            y,
            text,
            transform=axis.transAxes,
            ha=horizontal,
            va=vertical,
            fontsize=font["tick_pt"],
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": annotation["box_alpha"],
            },
            zorder=10,
            clip_on=False,
        )
        axis.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{value / row.DoseDisplayDivisor:g}")
        )
    return fig


def render_programming_jitter(
    summaries: pd.DataFrame, row: pd.Series, config: VizConfig
) -> plt.Figure:
    style = config["style"]
    geometry = style["geometry_mm"]
    labels = config["labels"]
    adjustment_labels = [labels["jitter_adjustment"][model] for model in MODELS]
    feature_label = _jitter_feature_label(row)
    phase_label = labels["phase_contrast"][row.Phase]
    right_strip_label = _jitter_right_strip_label(row, config)
    plot = summaries.copy()
    plot["CVPosition"] = plot.JitterCV.map(dict(zip(CVS, range(5))))
    plot["AdjustmentLabel"] = plot.Model.map(labels["jitter_adjustment"])
    plot["JitterStratum"] = right_strip_label
    fig = visualdf.plot_triple_interaction_fit(
        plot,
        pd.DataFrame(),
        pd.DataFrame(),
        value_col="rho_median",
        x_var="CVPosition",
        panel_var="AdjustmentLabel",
        facet_var="JitterStratum",
        panel_levels=adjustment_labels,
        facet_levels=[right_strip_label],
        color_var=None,
        x_label=f"{labels['endpoint'][row.EndpointID]}, assumed jitter CV (%)",
        y_label=f"Δ{feature_label} ({phase_label})\nCorrelation coefficient (ρ)",
        x_limits=(-0.4, 4.4),
        y_limits=(-1.05, 1.05),
        font_family=style["font"]["family"],
        boxsize=tuple(geometry["jitter_box"]),
        panel_gap=tuple(geometry["jitter_panel_gap"]),
        jitter_alpha=0,
        jitter_size=0,
        show_ribbon=False,
        label_fontsize=style["font"]["strip_pt"],
        label_top_bg_color=style["strips"]["top_background"],
        label_right_bg_color=style["strips"]["right_background"],
        label_text_color=style["strips"]["jitter_text_color"],
        label_fontweight=style["strips"]["jitter_font_weight"],
        strip_top_height_mm=geometry["jitter_adjustment_strip_height"],
        strip_right_width_mm=geometry["jitter_right_strip_width"],
        strip_pad_mm=geometry["jitter_strip_pad"],
        grid=False,
        show_top_right_axes=False,
        axis_label_fontsize=style["font"]["axis_pt"],
        tick_label_fontsize=style["font"]["tick_pt"],
        x_label_offset_mm=5,
        y_label_offset_mm=5,
        dpi=style["dpi"],
        transparent=style["transparent"],
    )
    for model, ax in zip(MODELS, fig.axes[:3]):
        subset = plot.loc[plot.Model.eq(model)].sort_values("CVPosition")
        ax.set_xticks(range(5), ["0", "10", "20", "30", "50"])
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        valid = subset.loc[subset.rho_median.notna()]
        ax.vlines(valid.CVPosition, valid.rho_p025, valid.rho_p975, color="black", lw=1)
        ax.plot(valid.CVPosition, valid.rho_median, "o", color="black", markersize=3)
        nominal = float(subset.iloc[0].rho)
        if np.isfinite(nominal):
            ax.axhline(nominal, color="#777777", linestyle="--", lw=0.8)
        ax.axhline(0, color="#BBBBBB", lw=0.5, zorder=0)
    return fig


def render_programming_figure(
    plan: VizPlan, row: pd.Series, config: VizConfig
) -> plt.Figure:
    if row.PlotKind == "heatmap":
        return render_programming_heatmap(
            plan.coverage.loc[plan.coverage.FigureID.eq(row.FigureID)], row, config
        )
    if row.PlotKind == "fit":
        if row.get("Layout") == "adjusted_two_panel":
            return render_programming_adjusted_fit(
                plan.model_data[row.FigureID],
                plan.result_tables[(row.FigureID, "fit", "curve")],
                plan.result_tables[(row.FigureID, "fit", "statistics")],
                row,
                config,
            )
        return render_programming_fit(
            plan.model_data[row.FigureID],
            row,
            config,
        )
    return render_programming_jitter(plan.model_data[row.FigureID], row, config)


def programming_text_outputs(config: VizConfig) -> dict[Path, str]:
    """Return provenance text for the existing runner's overwrite protection."""
    metadata = {
        "analysis": config.analysis_id,
        "strategy": config["inputs"]["strategy"],
        "cross_strategy_comparison": False,
        "config": str(config.path),
        "inputs": config["paths"],
        "endpoint_labels": config["labels"]["endpoint"],
        "models": MODEL_LABELS,
        "fit_style_source": str(
            config.path.parent / config["inputs"]["fit_style_config"]
        ),
        "heatmap": {
            "figure_count": (
                6 if config["inputs"]["strategy"] == "contact-matched" else 9
            ),
            "value": "rho",
            "limits": [-1, 1],
            "p": "p_holm",
            "family_size": 6,
            "nonestimable": "gray_NA",
            "local_polarity_layout": (
                "combined"
                if config["inputs"]["strategy"] == "contact-matched"
                else "separate"
            ),
            "polarity_strips": {
                "polarity_labels": config["labels"]["polarity"],
                "adjustment_labels": config["labels"]["jitter_adjustment"],
                "format": "{adjustment} | {polarity_symbol}",
                "height_mm": 4.5,
                "polarity_words": False,
            },
            "right_strip_tiers": ["dose | laterality", "DBS type"],
            "dbs_row_groups": {"STN-DBS": [0, 1], "STN+SNr-DBS": [2, 3, 4, 5]},
            "aperiodic_heading": False,
            "aperiodic_laterality_strips": True,
            "overall_title": False,
            "colorbar_extent": "six_row_main_grid_excluding_aperiodic",
            "colorbar_labels": {
                "M0": "Spearman's ρ",
                "M1": "Partial Spearman's ρ",
                "M2": "Partial Spearman's ρ",
            },
            "colorbar_label_fontsize_pt": config["style"]["font"]["colorbar_pt"],
        },
        "selection": config["inputs"]["selection"],
        "ols": {
            "formula": "ProgrammingIntensity ~ 1 + Value",
            "confidence_level": 0.95,
            "interval": "pointwise_mean_response",
            "x_grid": "observed_range",
            "reported_inference": "stored_Spearman_only",
            "dose_tick_divisor": 10000,
        },
        "jitter": {
            "source": "saved_summary",
            "cv": CVS,
            "iterations": 10000,
            "seed": 20260903,
            "interval": "simulation_percentiles_not_sampling_CI",
            "adjustment_labels": config["labels"]["jitter_adjustment"],
            "overall_title": False,
            "top_strip": "adjustment_only",
            "right_strip": "region_polarity_symbol_laterality",
            "axis_context": ["endpoint_and_jitter_cv", "feature_phase_and_rho"],
            "contact_matched_associations": (
                [
                    {"endpoint": endpoint, "predictor": predictor}
                    for endpoint, predictor in CONTACT_FOCUSED_LOW_BETA_KEYS
                ]
                if config["inputs"]["strategy"] == "contact-matched"
                else "all_selected_M0_associations"
            ),
        },
        "style": config["style"],
        "upstream_recomputation": False,
    }
    if config["inputs"]["strategy"] == "contact-matched":
        metadata["adjusted_ols"] = {
            "associations": [
                {"endpoint": endpoint, "predictor": predictor}
                for endpoint, predictor in CONTACT_FOCUSED_LOW_BETA_KEYS
            ],
            "internal_models": [item[0] for item in ADJUSTED_FIT_MODELS],
            "reader_panel_labels": [
                item[2].replace("\n", " ") for item in ADJUSTED_FIT_MODELS
            ],
            "formula": "ProgrammingIntensity ~ 1 + Value + Baseline",
            "baseline_reference": "panel_covariate_median",
            "confidence_level": 0.95,
            "interval": "pointwise_mean_response",
            "reported_inference": "stored_partial_Spearman_only",
            "layout": "two_side_by_side_panels",
            "panel_gap_mm": config["style"]["geometry_mm"]["panel_gap"][0],
        }
    documentation = Path(__file__).resolve().parents[1] / "README.md"
    documentation_text = documentation.read_text(encoding="utf-8")
    return {
        config.manifest_root
        / "visualization.json": json.dumps(metadata, indent=2, ensure_ascii=False)
        + "\n",
        config.output_root / "README.md": documentation_text,
    }
