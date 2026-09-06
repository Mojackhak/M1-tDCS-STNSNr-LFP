"""Plan and render baseline-consistent clinical correlation fit plots."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import t

from . import visualdf
from .config import (
    SUBMISSION_CLINICAL_CORRELATION_ADJACENT_PHASE_FIT_ID,
    SUBMISSION_CLINICAL_CORRELATION_FIT_ID,
    VizConfig,
)
from .correlation_heatmap import load_correlation_results
from .data import path_below, read_csv, require_columns
from .plan import VizPlan

FIT_POINT_COLUMNS = {
    "AnalysisID",
    "CorrelationID",
    "EndpointID",
    "Scale",
    "PredictorID",
    "ID",
    "Method",
    "Transform",
    "XRaw",
    "YRaw",
    "Baseline",
    "XPlot",
    "YPlot",
}


def _slug(value: Any) -> str:
    return str(value).replace("_", "-").replace(" ", "-").replace("/", "-")


def _load_fit_points(config: VizConfig) -> pd.DataFrame:
    path = path_below(config.fit_points_path, config.stats_root, "Fit-point result")
    points = read_csv(path, "Fit-point result")
    require_columns(points, FIT_POINT_COLUMNS, f"Fit-point result {path}")
    if not points["AnalysisID"].eq(config.stats_analysis_id).all():
        raise ValueError("Fit-point AnalysisID does not match stats_analysis_id.")
    if points.duplicated(["CorrelationID", "ID"]).any():
        raise ValueError("Fit-point result contains duplicate patient coordinates.")
    return points


def _validate_selected_points(
    selected: pd.DataFrame, points: pd.DataFrame
) -> pd.DataFrame:
    selected_ids = set(selected["CorrelationID"])
    selected_points = points.loc[points["CorrelationID"].isin(selected_ids)].copy()
    if set(selected_points["CorrelationID"]) != selected_ids:
        raise ValueError("Selected correlations do not all have fit-point rows.")
    point_counts = selected_points.groupby("CorrelationID", observed=True).size()
    expected_counts = selected.set_index("CorrelationID")["n"].astype(int)
    if not point_counts.equals(expected_counts.reindex(point_counts.index)):
        raise ValueError("Selected fit-point counts do not match correlation n.")
    expected_subjects = selected.set_index("CorrelationID")["SubjectIDs"].map(
        lambda value: set(str(value).split(";"))
    )
    observed_subjects = selected_points.groupby("CorrelationID", observed=True)[
        "ID"
    ].agg(lambda values: set(values.astype(str)))
    if any(
        observed_subjects.loc[correlation_id] != expected_subjects.loc[correlation_id]
        for correlation_id in observed_subjects.index
    ):
        raise ValueError("Selected fit-point patient IDs do not match SubjectIDs.")
    return selected_points


def _stratum_label(row: pd.Series, config: VizConfig) -> str:
    region = config["labels"]["region"][str(row["RegionValue"])]
    polar = config["labels"]["polarity"][str(row["Polar"])]
    phase_label = (
        config["labels"]["phase_contrast"][str(row["Phase"])]
        if config.stats_analysis_id == "clinical-correlation-adjacent-phase"
        else str(row["Phase"])
    )
    return f"{region} | {polar} | {phase_label}"


def _raw_linear_model_curve(
    points: pd.DataFrame,
    *,
    method: str,
    x_column: str,
    y_column: str,
    baseline_reference: float | None,
    confidence_level: float,
) -> pd.DataFrame:
    """Fit the descriptive raw-space model and return its mean confidence band."""

    x_values = points[x_column].to_numpy(dtype=float)
    y_values = points[y_column].to_numpy(dtype=float)
    x_grid = np.linspace(float(np.min(x_values)), float(np.max(x_values)), 100)
    if method == "spearman":
        design = np.column_stack((np.ones(len(points)), x_values))
        grid_design = np.column_stack((np.ones(len(x_grid)), x_grid))
    elif method == "partial_spearman":
        if baseline_reference is None:
            raise ValueError("Partial fit requires a Baseline reference.")
        baseline = points["Baseline"].to_numpy(dtype=float)
        design = np.column_stack((np.ones(len(points)), x_values, baseline))
        grid_design = np.column_stack(
            (
                np.ones(len(x_grid)),
                x_grid,
                np.full(len(x_grid), baseline_reference),
            )
        )
    else:
        raise ValueError(f"Unsupported correlation fit method: {method}")

    coefficients, _, rank, _ = np.linalg.lstsq(design, y_values, rcond=None)
    n_parameters = design.shape[1]
    if rank != n_parameters:
        raise ValueError("Raw-space linear-model design is rank deficient.")
    residual_df = len(points) - n_parameters
    if residual_df <= 0:
        raise ValueError("Raw-space linear model has no residual degrees of freedom.")

    residuals = y_values - design @ coefficients
    residual_variance = float(residuals @ residuals) / residual_df
    coefficient_covariance = residual_variance * np.linalg.inv(design.T @ design)
    fitted = grid_design @ coefficients
    mean_variance = np.einsum(
        "ij,jk,ik->i", grid_design, coefficient_covariance, grid_design
    )
    mean_se = np.sqrt(np.maximum(mean_variance, 0.0))
    critical_value = float(t.ppf(1.0 - (1.0 - confidence_level) / 2.0, residual_df))
    return pd.DataFrame(
        {
            x_column: x_grid,
            "emmean": fitted,
            "lower.CL": fitted - critical_value * mean_se,
            "upper.CL": fitted + critical_value * mean_se,
        }
    )


def _fit_annotation_location(curve: pd.DataFrame, x_column: str) -> str:
    """Place inference text opposite the fitted curve's left-side trajectory."""

    fitted = curve.sort_values(x_column)["emmean"].to_numpy(dtype=float)
    return "upper left" if fitted[-1] > fitted[0] else "lower left"


def _fit_annotation_format(p_value: float, annotation: Mapping[str, Any]) -> str:
    """Select the configured inference annotation format."""

    if p_value < float(annotation["small_p_threshold"]):
        return str(annotation["small_p_text_format"])
    return str(annotation["text_format"])


def _fit_y_label(row: Mapping[str, Any], config: VizConfig) -> str:
    """Return a concise postoperative label without changing endpoint semantics."""

    endpoint_id = str(row["EndpointID"])
    endpoint_label = config["labels"]["endpoint"].get(endpoint_id)
    if endpoint_label is None:
        return f"{row['EndpointLabel']} {row['ScaleLabel']}"
    return f"{row['ScaleLabel']} ({endpoint_label})"


def build_correlation_fit_plan(config: VizConfig) -> VizPlan:
    """Select Holm-significant correlations and construct one fit per cell."""

    correlations = load_correlation_results(config)
    require_columns(
        correlations,
        {"Method", "ScaleDirection"},
        "Correlation result for fit plots",
    )
    adjacent_phase = config.stats_analysis_id == ("clinical-correlation-adjacent-phase")
    if adjacent_phase:
        require_columns(
            correlations,
            {"FromPhase", "ToPhase", "PhaseContrast", "PredictorDefinition"},
            "Adjacent-phase correlation result for fit plots",
        )
    selection = config["inputs"]["selection"]
    selected = correlations.loc[
        correlations["Status"].eq("ok")
        & pd.to_numeric(correlations[selection["column"]], errors="coerce").lt(
            float(selection["value"])
        )
    ].copy()
    if selected.empty:
        raise ValueError("No clinical correlations satisfy the fit selection.")
    selected = selected.sort_values(
        [
            "EndpointOrder",
            "ScaleOrder",
            "Domain",
            "RegionValue",
            "Polar",
            "Phase",
            "Lat",
            "FeatureOrder",
        ]
    )
    points = _validate_selected_points(selected, _load_fit_points(config))
    fit_style = config["style"]["fit"]

    figure_rows: list[dict[str, Any]] = []
    point_frames: list[pd.DataFrame] = []
    for _, row in selected.iterrows():
        correlation_id = str(row["CorrelationID"])
        figure_id = f"analysis-{config.analysis_id}_correlation-{_slug(correlation_id)}"
        figure_points = points.loc[points["CorrelationID"].eq(correlation_id)].copy()
        point_methods = figure_points["Method"].astype(str).unique()
        if len(point_methods) != 1 or point_methods[0] != row["Method"]:
            raise ValueError(f"Fit-point method does not match: {correlation_id}")
        stratum_label = _stratum_label(row, config)
        feature_label = config["labels"]["delta_prefix"] + str(row["FeatureLabel"])
        phase_contrast = (
            str(row["PhaseContrast"]) if adjacent_phase else str(row["Phase"])
        )
        if (
            adjacent_phase
            and phase_contrast != config["labels"]["phase_contrast"][str(row["Phase"])]
        ):
            raise ValueError(f"Adjacent-phase label does not match: {correlation_id}")
        x_label = feature_label
        y_label = _fit_y_label(row, config)
        baseline_reference = (
            float(figure_points["Baseline"].median())
            if row["Method"] == "partial_spearman"
            else np.nan
        )
        fit_formula = (
            fit_style["ordinary_formula"]
            if row["Method"] == "spearman"
            else fit_style["partial_formula"]
        )
        output_parent = config.output_root / "fit"
        if config.analysis_id not in {
            SUBMISSION_CLINICAL_CORRELATION_FIT_ID,
            SUBMISSION_CLINICAL_CORRELATION_ADJACENT_PHASE_FIT_ID,
        }:
            output_parent /= config.stats_analysis_id
        output_parent = (
            output_parent
            / str(row["EndpointID"])
            / f"scale-{_slug(row['ScaleLabel'])}"
            / str(row["Domain"])
            / _slug(row["RegionValue"])
            / str(row["Polar"])
            / _slug(phase_contrast)
            / str(row["Lat"])
        )
        filename = config["outputs"]["filename_template"].format(
            AnalysisID=config.analysis_id,
            CorrelationID=_slug(correlation_id),
        )
        figure_rows.append(
            {
                "AnalysisID": config.analysis_id,
                "StatsAnalysisID": config.stats_analysis_id,
                "FigureID": figure_id,
                "CorrelationID": correlation_id,
                "TestKind": "correlation_fit",
                "TestID": config["figures"][0]["test_id"],
                "Layout": "stratum_single",
                "EndpointID": row["EndpointID"],
                "EndpointLabel": row["EndpointLabel"],
                "Scale": row["Scale"],
                "ScaleLabel": row["ScaleLabel"],
                "Method": row["Method"],
                "Domain": row["Domain"],
                "Metric": row["Metric"],
                "FeatureOutput": row["FeatureOutput"],
                "Band": row["Band"],
                "RegionValue": row["RegionValue"],
                "Polar": row["Polar"],
                "Phase": row["Phase"],
                "PhaseContrast": phase_contrast,
                "Lat": row["Lat"],
                "StratumLabel": stratum_label,
                "XLabel": x_label,
                "YLabel": y_label,
                "FitFormula": fit_formula,
                "BaselineReference": baseline_reference,
                "BaselineReferenceMode": fit_style["baseline_reference"],
                "ConfidenceLevel": float(fit_style["confidence_level"]),
                "IntervalType": fit_style["interval_type"],
                "n": int(row["n"]),
                "rho": float(row["rho"]),
                "p_raw": float(row["p_raw"]),
                "p_holm": float(row["p_holm"]),
                "OutputPath": str(output_parent / filename),
                "Status": "planned",
                "Message": "",
            }
        )
        figure_points["FigureID"] = figure_id
        figure_points["StratumLabel"] = stratum_label
        figure_points["Lat"] = str(row["Lat"])
        point_frames.append(figure_points)

    figures = pd.DataFrame(figure_rows)
    coverage = pd.concat(point_frames, ignore_index=True)
    if (
        figures["FigureID"].duplicated().any()
        or figures["OutputPath"].duplicated().any()
    ):
        raise ValueError("Correlation fit plan contains duplicate figures or paths.")
    if coverage.duplicated(["FigureID", "ID"]).any():
        raise ValueError("Correlation fit plan contains duplicate plotted patients.")
    return VizPlan(
        figures=figures,
        coverage=coverage,
        model_data={},
        result_tables={},
    )


def render_correlation_fit_figure(
    points: pd.DataFrame,
    figure: Mapping[str, Any],
    config: VizConfig,
):
    """Render one correlation fit through the shared visualdf fit backbone."""

    x_column = config["inputs"]["x_column"]
    y_column = config["inputs"]["y_column"]
    stratum_label = str(figure["StratumLabel"])
    lat = str(figure["Lat"])
    points = points.copy()
    points["Lat"] = lat
    baseline_reference = (
        float(figure["BaselineReference"])
        if str(figure["Method"]) == "partial_spearman"
        else None
    )
    curve = _raw_linear_model_curve(
        points,
        method=str(figure["Method"]),
        x_column=x_column,
        y_column=y_column,
        baseline_reference=baseline_reference,
        confidence_level=float(figure["ConfidenceLevel"]),
    )
    curve["StratumLabel"] = stratum_label
    curve["Lat"] = lat
    selected_p_column = str(config["style"]["annotation"]["p_column"])
    slope = pd.DataFrame(
        {
            "rho": [float(figure["rho"])],
            selected_p_column: [float(figure[selected_p_column])],
            "StratumLabel": [stratum_label],
            "Lat": [lat],
        }
    )
    style = config["style"]
    font = style["font"]
    geometry = style["geometry_mm"]
    strips = style["strips"]
    point_style = style["points"]
    fit_style = style["fit"]
    annotation = style["annotation"]
    annotation_p_column = str(annotation["p_column"])
    annotation_p_value = float(figure[annotation_p_column])
    axes = style["axes"]
    annotation_location = (
        _fit_annotation_location(curve, x_column)
        if annotation["location"] == "auto_by_fit_slope"
        else annotation["location"]
    )
    return visualdf.plot_triple_interaction_fit(
        df=points,
        curve=curve,
        slope=slope,
        value_col=y_column,
        x_var=x_column,
        panel_var="StratumLabel",
        facet_var="Lat",
        color_var=point_style["color_var"],
        panel_levels=[stratum_label],
        facet_levels=[lat],
        x_label=str(figure["XLabel"]),
        y_label=str(figure["YLabel"]),
        font_family=font["family"],
        palette=point_style["palette"],
        jitter_alpha=point_style["alpha"],
        jitter_size=point_style["size"],
        curve_line_width=fit_style["line_width"],
        curve_line_color=fit_style["line_color"],
        ribbon_alpha=fit_style["ribbon_alpha"],
        show_ribbon=fit_style["show_ribbon"],
        grid=axes["grid"],
        dpi=style["dpi"],
        show_top_right_axes=axes["show_top_right"],
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
        panel_gap=tuple(geometry["panel_gap"]),
        x_label_offset_mm=geometry["x_label_offset"],
        y_label_offset_mm=geometry["y_label_offset"],
        slope_beta_col="rho",
        slope_p_col=annotation_p_column,
        slope_text_fmt=_fit_annotation_format(annotation_p_value, annotation),
        slope_text_loc=annotation_location,
        slope_text_box_alpha=annotation["box_alpha"],
        transparent=style["transparent"],
    )
