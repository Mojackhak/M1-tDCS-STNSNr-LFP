"""Render registered Spearman and cortical-allocation result figures."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lfp_viz import visualdf
from lfp_viz.correlation_fit import (
    _fit_annotation_location,
    _raw_linear_model_curve,
)
from lfp_viz.scalar import y_axis_label

from .artifacts import atomic_csv
from .config import StructuralConnectivityConfig
from .contact_lmm import (
    HEATMAP_ORIENTATIONS,
    _aperiodic_heatmap_panel,
    _aperiodic_parameter_contract,
    _atomic_pdf,
    _display_metric_key,
    _format_combined_atlas_grid_heatmap,
    _heatmap_panel,
    _heatmap_metric_order,
    _metric_labels,
    _pad_aperiodic_heatmap_panel,
    _pdf_page_dimensions_mm,
    _sanitize,
    _visual_styles,
)
from .statistics import build_analysis_points


SPEARMAN_DISPLAY_LIMIT = 1.0
CORTICAL_ALLOCATION_DISPLAY_LIMIT = 2.0
SPEARMAN_CONTEXTS = ("STN", "SNr", "RegionMean")
CORTICAL_ALLOCATION_CONTEXTS = ("STN", "SNr", "Connectivity")
CORTICAL_ADJACENT_ORIENTATION = "metric-by-band"
CORTICAL_ADJACENT_MAX_WIDTH_MM = 150.0
CORTICAL_ADJACENT_MAX_HEIGHT_MM = 210.0
ANALYSIS_LEVEL_LABELS = {
    "subject_aggregated": "Subject",
    "row_unaggregated": "Contact/ChannelPair",
}


def _read_csv(path: Path, required: Iterable[str], label: str) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{label} is missing: {path}")
    table = pd.read_csv(path, low_memory=False)
    missing = sorted(set(required).difference(table.columns))
    if missing:
        raise KeyError(f"{label} is missing columns: {missing}")
    return table


def _spearman_root(config: StructuralConnectivityConfig) -> Path:
    return config.summary_root / "spearman"


def _cortical_figure_root(config: StructuralConnectivityConfig) -> Path:
    return config.cortical_allocation_lmm_root


def _band_and_metric_labels(
    config: StructuralConnectivityConfig,
) -> tuple[Any, dict[str, str], dict[str, str], list[str]]:
    _, style_config = _visual_styles(config)
    band_labels = {
        str(row["value"]): str(row["label"])
        for row in style_config["matrices"]["banded"]["rows"]
    }
    return (
        style_config,
        band_labels,
        _metric_labels(style_config),
        _heatmap_metric_order(config, style_config),
    )


def _render_heatmap(
    cells: pd.DataFrame,
    spec: Mapping[str, Any],
    config: StructuralConnectivityConfig,
    *,
    value_column: str,
    significance_column: str,
    display_limit: float,
    colorbar_label: str,
    top_labels: list[str],
    right_labels: list[str] | None = None,
    y_label: str | None = None,
):
    phases = list(config["outcomes"]["phases"])
    bands = list(config["outcomes"]["bands"])
    style_config, band_labels, metric_labels, metric_order = _band_and_metric_labels(
        config
    )
    style = style_config["style"]
    geometry = style["geometry_mm"]
    strips = style["strips"]
    heatmap = style["heatmap"]
    cmap = getattr(visualdf.cm, heatmap["colormap"]).copy()
    cmap.set_bad("#d9d9d9")
    value_grid: list[pd.DataFrame] = []
    adjusted_p_grid: list[pd.DataFrame] = []
    for phase in phases:
        phase_rows = cells.loc[cells["Phase"].eq(phase)]
        value = phase_rows.pivot(
            index="Band", columns="Metric", values=value_column
        ).reindex(index=bands, columns=metric_order)
        adjusted_p = phase_rows.pivot(
            index="Band", columns="Metric", values=significance_column
        ).reindex(index=bands, columns=metric_order)
        value.index = [band_labels[str(band)] for band in value.index]
        value.columns = [metric_labels[str(metric)] for metric in value.columns]
        adjusted_p.index = value.index
        adjusted_p.columns = value.columns
        if spec["Orientation"] == "metric-by-band":
            value = value.T
            adjusted_p = adjusted_p.T
        value_grid.append(value)
        adjusted_p_grid.append(adjusted_p)
    return visualdf.plot_well_heatmap_grid_df(
        df_values=[value_grid],
        df_ps=[adjusted_p_grid],
        x_label=None,
        y_label=y_label,
        xtick_rotation=90,
        ytick_rotation=0,
        title=None,
        label_top=top_labels,
        label_right=right_labels,
        font_family=style["font"]["family"],
        cmap=cmap,
        vmin=-display_limit,
        vmax=display_limit,
        vmode="sym",
        dpi=style["dpi"],
        cellsize=tuple(float(value) for value in geometry["cell_size"]["banded"]),
        panel_gap_mm=tuple(geometry["panel_gap"]["lat_rows_phase_columns"]),
        label_top_height_mm=geometry["strip_top_height"],
        label_right_width_mm=geometry["strip_right_width"],
        strip_pad_mm=geometry["strip_pad"],
        label_top_bg_color=strips["top_background"],
        label_right_bg_color=strips["right_background"],
        label_text_color=strips["text_color"],
        label_fontsize=style["font"]["strip_pt"],
        label_fontweight=strips["font_weight"],
        include_global_label_margins=False,
        cbar_label_offset_mm=geometry["colorbar_label_offset"],
        grid_lines=None,
        gline_color=style["grid"]["color"],
        gline_width=style["grid"]["width_pt"],
        gline_alpha=style["grid"]["alpha"],
        axis_label_fontsize=style["font"]["strip_pt"],
        tick_label_fontsize=style["font"]["tick_pt"],
        colorbar_label=colorbar_label,
        colorbar_ticks=[-display_limit, 0.0, display_limit],
        colorbar_width_mm=geometry["colorbar_width"],
        colorbar_pad_mm=geometry["colorbar_pad"],
        show_p=True,
        hide_ns=style["significance"]["hide_non_significant"],
        p_text_color=style["significance"]["text_color"],
        p_text_size=style["font"]["star_pt"],
        show_x_ticklabels="bottom",
        show_y_ticklabels="left",
        transparent=style["transparent"],
    )


def _spearman_heatmap_specs(correlations: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    keys = ["AnalysisLevel", "ConnectomeSource", "LiftMetric", "Polar"]
    for values, group in correlations.groupby(
        keys, sort=True, dropna=False, observed=True
    ):
        analysis_level, source, lift_metric, polar = values
        local_contexts = set(
            group.loc[group["OutcomeDomain"].eq("local"), "Region"].astype(str)
        )
        network_contexts = set(
            group.loc[
                group["OutcomeDomain"].eq("connectivity"), "LiftPredictor"
            ].astype(str)
        )
        for context in SPEARMAN_CONTEXTS:
            if context not in local_contexts and context not in network_contexts:
                continue
            for orientation in HEATMAP_ORIENTATIONS:
                records.append(
                    {
                        "AnalysisLevel": analysis_level,
                        "OutcomeDomain": (
                            "combined" if context in {"STN", "SNr"} else "connectivity"
                        ),
                        "ConnectomeSource": source,
                        "LiftMetric": lift_metric,
                        "LiftPredictor": context,
                        "DisplayContext": context,
                        "Polar": polar,
                        "Region": context if context in {"STN", "SNr"} else "",
                        "Orientation": orientation,
                    }
                )
    for index, record in enumerate(records, start=1):
        filename = (
            "_".join(
                _sanitize(value)
                for value in (
                    f"level-{record['AnalysisLevel']}",
                    f"connectome-{record['ConnectomeSource']}",
                    f"lift-{record['LiftMetric']}",
                    f"context-{record['DisplayContext']}",
                    f"polar-{record['Polar']}",
                    f"orientation-{record['Orientation']}",
                )
            )
            + ".pdf"
        )
        record.update(
            {
                "FigureID": f"SPEARHEAT{index:03d}",
                "FigureType": "heatmap",
                "SelectionRole": "all_heatmap",
                "Filename": filename,
            }
        )
    return records


def _spearman_heatmap_cells(
    correlations: pd.DataFrame, spec: Mapping[str, Any]
) -> pd.DataFrame:
    base = (
        correlations["AnalysisLevel"].eq(spec["AnalysisLevel"])
        & correlations["ConnectomeSource"].eq(spec["ConnectomeSource"])
        & correlations["LiftMetric"].eq(spec["LiftMetric"])
        & correlations["Polar"].eq(spec["Polar"])
    )
    context = str(spec["DisplayContext"])
    frames = [
        correlations.loc[
            base
            & correlations["OutcomeDomain"].eq("connectivity")
            & correlations["LiftPredictor"].eq(context)
        ].copy()
    ]
    if context in {"STN", "SNr"}:
        frames.append(
            correlations.loc[
                base
                & correlations["OutcomeDomain"].eq("local")
                & correlations["Region"].eq(context)
                & correlations["LiftPredictor"].eq(context)
            ].copy()
        )
    return pd.concat(frames, ignore_index=True, sort=False)


def _render_spearman_heatmaps(
    correlations: pd.DataFrame,
    config: StructuralConnectivityConfig,
    *,
    overwrite: bool,
) -> pd.DataFrame:
    _, style_config = _visual_styles(config)
    phases = list(config["outcomes"]["phases"])
    figure_root = _spearman_root(config) / "figures" / "heatmap"
    records: list[dict[str, Any]] = []
    visualdf.set_greek_symbol_font_enabled(False)
    for spec in _spearman_heatmap_specs(correlations):
        cells = _spearman_heatmap_cells(correlations, spec)
        q_column = (
            "q_exact" if spec["AnalysisLevel"] == "subject_aggregated" else "q_rowwise"
        )
        context_label = str(spec["DisplayContext"])
        if context_label == "RegionMean":
            context_label = "Mean"
        polar_label = style_config["labels"]["polarity"][str(spec["Polar"])]
        level_label = ANALYSIS_LEVEL_LABELS[str(spec["AnalysisLevel"])]
        connectome_label = config["inference"]["contact_lmm"]["visualization"][
            "connectome_labels"
        ][str(spec["ConnectomeSource"])]
        figure = _render_heatmap(
            cells,
            spec,
            config,
            value_column="rho",
            significance_column=q_column,
            display_limit=SPEARMAN_DISPLAY_LIMIT,
            colorbar_label="Spearman rho",
            top_labels=[
                f"{context_label} | {polar_label} | {phase}" for phase in phases
            ],
            right_labels=[level_label],
            y_label=connectome_label,
        )
        source_path = figure_root / str(spec["Filename"])
        try:
            _atomic_pdf(
                figure,
                source_path,
                overwrite=overwrite,
                dpi=style_config["style"]["dpi"],
            )
        finally:
            visualdf.plt.close(figure)
        records.append(
            {
                **spec,
                "SourcePath": str(source_path),
                "PublishedRelativePath": f"heatmap/{spec['Filename']}",
                "SourceResultTable": str(config.output_path("correlations")),
                "PhasePanels": ";".join(phases),
                "ValueColumn": "rho",
                "SignificanceColumn": q_column,
                "YAxisLabel": connectome_label,
                "Renderer": "lfp_viz.visualdf.plot_well_heatmap_grid_df",
                "StyleConfig": str(style_config.path),
                "ColorMap": style_config["style"]["heatmap"]["colormap"],
                "DisplayMin": -SPEARMAN_DISPLAY_LIMIT,
                "DisplayMax": SPEARMAN_DISPLAY_LIMIT,
                "Status": "complete",
            }
        )
    return pd.DataFrame(records)


def _row_analysis_points(
    local_rows: pd.DataFrame,
    connectivity_rows: pd.DataFrame,
    lift_metrics: Iterable[str],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for lift_metric in lift_metrics:
        _, rows = build_analysis_points(
            local_rows.loc[local_rows["LiftMetric"].eq(lift_metric)],
            connectivity_rows.loc[connectivity_rows["LiftMetric"].eq(lift_metric)],
        )
        rows.insert(1, "LiftMetric", lift_metric)
        frames.append(rows)
    return pd.concat(frames, ignore_index=True, sort=False)


def _matching_points(points: pd.DataFrame, result: Mapping[str, Any]) -> pd.DataFrame:
    keys = (
        "AnalysisLevel",
        "LiftMetric",
        "OutcomeDomain",
        "Metric",
        "OutcomeScale",
        "ConnectomeSource",
        "Polar",
        "Region",
        "PairRegion",
        "Phase",
        "Band",
        "LiftPredictor",
    )
    selector = pd.Series(True, index=points.index)
    for key in keys:
        expected = result[key]
        observed = points[key]
        if pd.isna(expected) or str(expected) == "":
            selector &= observed.fillna("").astype(str).eq("")
        else:
            selector &= observed.astype(str).eq(str(expected))
    return points.loc[selector].copy()


def _spearman_fit_specs(correlations: pd.DataFrame, alpha: float) -> pd.DataFrame:
    adjusted = pd.Series(
        np.where(
            correlations["AnalysisLevel"].eq("subject_aggregated"),
            pd.to_numeric(correlations["q_exact"], errors="coerce"),
            pd.to_numeric(correlations["q_rowwise"], errors="coerce"),
        ),
        index=correlations.index,
    )
    selected = correlations.loc[
        correlations["OutcomeDomain"].eq("connectivity")
        & correlations["Status"].eq("ok")
        & np.isfinite(adjusted)
        & (adjusted < alpha)
    ].copy()
    selected["AdjustedP"] = adjusted.loc[selected.index]
    selected["SignificanceColumn"] = np.where(
        selected["AnalysisLevel"].eq("subject_aggregated"),
        "q_exact",
        "q_rowwise",
    )
    return selected.sort_values(
        [
            "AnalysisLevel",
            "ConnectomeSource",
            "LiftMetric",
            "LiftPredictor",
            "Polar",
            "Metric",
            "PairRegion",
            "Phase",
            "Band",
        ],
        kind="mergesort",
    ).reset_index(drop=True)


def _render_spearman_fits(
    selected: pd.DataFrame,
    subject_points: pd.DataFrame,
    row_points: pd.DataFrame,
    config: StructuralConnectivityConfig,
    *,
    overwrite: bool,
) -> pd.DataFrame:
    scalar_style, _ = _visual_styles(config)
    style = scalar_style["style"]
    font = style["font"]
    geometry = style["geometry_mm"]
    strips = style["strips"]
    widths = style["line_width_pt"]
    visualization = config["inference"]["contact_lmm"]["visualization"]
    figure_root = _spearman_root(config) / "figures" / "fit"
    records: list[dict[str, Any]] = []
    visualdf.set_greek_symbol_font_enabled(False)
    for index, result in selected.iterrows():
        point_source = (
            subject_points
            if result["AnalysisLevel"] == "subject_aggregated"
            else row_points
        )
        points = _matching_points(point_source, result)
        expected_n = int(
            result["n_subjects"]
            if result["AnalysisLevel"] == "subject_aggregated"
            else result["n_rows"]
        )
        if len(points) != expected_n:
            raise ValueError(
                "Spearman fit-point count does not match the registered result: "
                f"expected {expected_n}, found {len(points)}."
            )
        curve = _raw_linear_model_curve(
            points,
            method="spearman",
            x_column="X",
            y_column="Y",
            baseline_reference=None,
            confidence_level=0.95,
        )
        pair_region = str(result["PairRegion"])
        region_label = scalar_style["labels"]["region"][pair_region]
        polar_label = scalar_style["labels"]["polarity"][str(result["Polar"])]
        short_level = (
            "Subject" if result["AnalysisLevel"] == "subject_aggregated" else "Row"
        )
        panel_label = (
            f"{short_level} | {region_label} | {polar_label} | {result['Phase']}"
        )
        connectome_label = visualization["connectome_labels"][
            str(result["ConnectomeSource"])
        ]
        for table in (points, curve):
            table["Panel"] = panel_label
            table["Facet"] = connectome_label
        slope = pd.DataFrame(
            {
                "rho": [float(result["rho"])],
                "AdjustedP": [float(result["AdjustedP"])],
                "Panel": [panel_label],
                "Facet": [connectome_label],
            }
        )
        predictor_label = visualization["lift_predictor_labels"][
            str(result["LiftPredictor"])
        ]
        lift_label = visualization["lift_metric_labels"][str(result["LiftMetric"])]
        figure = visualdf.plot_triple_interaction_fit(
            df=points,
            curve=curve,
            slope=slope,
            value_col="Y",
            x_var="X",
            panel_var="Panel",
            facet_var="Facet",
            color_var="ID",
            panel_levels=[panel_label],
            facet_levels=[connectome_label],
            x_label=f"{predictor_label} {lift_label}",
            y_label=(f"{y_axis_label(result, scalar_style)}\n{connectome_label}"),
            title=None,
            font_family=font["latin"],
            palette="tab20",
            jitter_alpha=style["jitter"]["alpha"],
            jitter_size=style["jitter"]["size"],
            curve_line_width=widths["emm_line"],
            curve_line_color=style["emm"]["color"],
            ribbon_alpha=style["box"]["fill_alpha"],
            show_ribbon=True,
            grid=style["axes"]["grid"],
            dpi=style["dpi"],
            seed=style["jitter"]["seed"],
            show_top_right_axes=style["axes"]["show_top_right_spines"],
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
            legend_loc="none",
            boxsize=(30.0, float(geometry["panel_height"])),
            panel_gap=tuple(geometry["panel_gap"]),
            x_label_offset_mm=geometry["x_label_offset"],
            y_label_offset_mm=geometry["y_label_offset"],
            slope_beta_col="rho",
            slope_p_col="AdjustedP",
            slope_text_fmt=(
                f"ρ = {{beta:.2f}}\nq = {{p:.3g}} ({{stars}})\nn = {expected_n}"
            ),
            slope_text_loc=_fit_annotation_location(curve, "X"),
            slope_text_box_alpha=0.0,
            transparent=style["transparent"],
        )
        bits = [
            f"level-{result['AnalysisLevel']}",
            f"connectome-{result['ConnectomeSource']}",
            f"lift-{result['LiftMetric']}",
            f"predictor-{result['LiftPredictor']}",
            f"polar-{result['Polar']}",
            f"metric-{result['Metric']}",
            f"pair-{result['PairRegion']}",
            f"phase-{result['Phase']}",
            f"band-{result['Band']}",
        ]
        filename = "_".join(_sanitize(value) for value in bits) + ".pdf"
        source_path = figure_root / filename
        try:
            _atomic_pdf(
                figure,
                source_path,
                overwrite=overwrite,
                dpi=style["dpi"],
            )
        finally:
            visualdf.plt.close(figure)
        records.append(
            {
                "FigureID": f"SPEARFIT{index + 1:03d}",
                "FigureType": "fit",
                "SelectionRole": "corrected_significant_connectivity_fit",
                **{
                    key: result[key]
                    for key in (
                        "AnalysisLevel",
                        "OutcomeDomain",
                        "Metric",
                        "OutcomeScale",
                        "ConnectomeSource",
                        "LiftMetric",
                        "LiftPredictor",
                        "Polar",
                        "Region",
                        "PairRegion",
                        "Phase",
                        "Band",
                        "rho",
                        "Status",
                    )
                },
                "AdjustedP": float(result["AdjustedP"]),
                "SignificanceColumn": result["SignificanceColumn"],
                "n": expected_n,
                "PatientIDs": ";".join(sorted(points["ID"].astype(str).unique())),
                "SourcePath": str(source_path),
                "PublishedRelativePath": f"fit/{filename}",
                "SourcePointTable": (
                    str(config.output_path("subject_aggregates"))
                    if result["AnalysisLevel"] == "subject_aggregated"
                    else (
                        f"{config.output_path('local_rows')};"
                        f"{config.output_path('connectivity_rows')}"
                    )
                ),
                "SourceResultTable": str(config.output_path("correlations")),
                "CurveRole": "descriptive_raw_space_linear_mean_ci",
                "YAxisLabel": (
                    f"{y_axis_label(result, scalar_style)}\n{connectome_label}"
                ),
                "Renderer": "lfp_viz.visualdf.plot_triple_interaction_fit",
                "StyleConfig": str(scalar_style.path),
                "Status": "complete",
            }
        )
    return pd.DataFrame(records)


def run_spearman_figures(
    config: StructuralConnectivityConfig,
    *,
    dry_run: bool,
    overwrite: bool,
) -> dict[str, Any]:
    """Render registered figures from existing Spearman result tables."""

    correlations = _read_csv(
        config.output_path("correlations"),
        {
            "AnalysisLevel",
            "LiftMetric",
            "OutcomeDomain",
            "Metric",
            "OutcomeScale",
            "ConnectomeSource",
            "Polar",
            "Region",
            "PairRegion",
            "Phase",
            "Band",
            "LiftPredictor",
            "rho",
            "Status",
            "n_subjects",
            "n_rows",
            "q_exact",
            "q_rowwise",
        },
        "Spearman result table",
    )
    subject_points = _read_csv(
        config.output_path("subject_aggregates"),
        {
            "AnalysisLevel",
            "LiftMetric",
            "OutcomeDomain",
            "Metric",
            "OutcomeScale",
            "ConnectomeSource",
            "Polar",
            "Region",
            "PairRegion",
            "Phase",
            "Band",
            "LiftPredictor",
            "ID",
            "X",
            "Y",
        },
        "Subject aggregate point table",
    )
    local_rows = _read_csv(
        config.output_path("local_rows"),
        {"LiftMetric", "EligibilityStatus", "Lift", "Value"},
        "Local analysis rows",
    )
    connectivity_rows = _read_csv(
        config.output_path("connectivity_rows"),
        {
            "LiftMetric",
            "EligibilityStatus",
            "STNLift",
            "SNrLift",
            "RegionMeanLift",
            "Value",
        },
        "Connectivity analysis rows",
    )
    _visual_styles(config)
    selected = _spearman_fit_specs(correlations, float(config["inference"]["alpha"]))
    heatmap_count = len(_spearman_heatmap_specs(correlations))
    summary = {
        "status": "ready" if dry_run else "complete",
        "heatmap_figures": int(heatmap_count),
        "fit_figures": int(len(selected)),
        "figure_rows": int(heatmap_count + len(selected)),
        "statistics_recomputed": False,
        "writes": 0,
    }
    if dry_run:
        return summary
    registry_path = _spearman_root(config) / "figures.csv"
    if registry_path.exists() and not overwrite:
        raise FileExistsError(
            "Spearman figure registry exists; pass --overwrite to replace it: "
            f"{registry_path}"
        )
    row_points = _row_analysis_points(
        local_rows,
        connectivity_rows,
        config["inference"]["lift_metrics"],
    )
    heatmaps = _render_spearman_heatmaps(correlations, config, overwrite=overwrite)
    fits = _render_spearman_fits(
        selected,
        subject_points,
        row_points,
        config,
        overwrite=overwrite,
    )
    figures = pd.concat([heatmaps, fits], ignore_index=True, sort=False)
    if figures["FigureID"].duplicated().any():
        raise ValueError("Spearman figure IDs must be unique.")
    if figures["PublishedRelativePath"].duplicated().any():
        raise ValueError("Spearman publication paths must be unique.")
    atomic_csv(figures, registry_path, overwrite=overwrite)
    summary["writes"] = int(len(figures) + 1)
    return summary


def _cortical_heatmap_specs(slopes: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    metadata = slopes[["CorticalRegion", "Polar"]].drop_duplicates()
    metadata = metadata.sort_values(["CorticalRegion", "Polar"], kind="mergesort")
    for row in metadata.itertuples(index=False):
        for context in CORTICAL_ALLOCATION_CONTEXTS:
            for orientation in HEATMAP_ORIENTATIONS:
                records.append(
                    {
                        "CorticalRegion": row.CorticalRegion,
                        "Polar": row.Polar,
                        "DisplayContext": context,
                        "OutcomeDomain": (
                            "combined" if context in {"STN", "SNr"} else "connectivity"
                        ),
                        "Orientation": orientation,
                    }
                )
    for index, record in enumerate(records, start=1):
        filename = (
            "_".join(
                _sanitize(value)
                for value in (
                    f"cortical-{record['CorticalRegion']}",
                    f"context-{record['DisplayContext']}",
                    f"polar-{record['Polar']}",
                    f"orientation-{record['Orientation']}",
                )
            )
            + ".pdf"
        )
        record.update(
            {
                "FigureID": f"CALHEAT{index:03d}",
                "FigureType": "heatmap",
                "SelectionRole": "all_heatmap",
                "Filename": filename,
            }
        )
    return records


def _cortical_heatmap_cells(
    slopes: pd.DataFrame, spec: Mapping[str, Any]
) -> pd.DataFrame:
    base = slopes["CorticalRegion"].eq(spec["CorticalRegion"]) & slopes["Polar"].eq(
        spec["Polar"]
    )
    network = slopes.loc[base & slopes["OutcomeDomain"].eq("connectivity")].copy()
    context = str(spec["DisplayContext"])
    if context == "Connectivity":
        return network
    local = slopes.loc[
        base & slopes["OutcomeDomain"].eq("local") & slopes["Region"].eq(context)
    ].copy()
    return pd.concat([network, local], ignore_index=True, sort=False)


def _render_cortical_heatmaps(
    slopes: pd.DataFrame,
    config: StructuralConnectivityConfig,
    *,
    overwrite: bool,
) -> pd.DataFrame:
    _, style_config = _visual_styles(config)
    phases = list(config["outcomes"]["phases"])
    figure_root = _cortical_figure_root(config) / "figures" / "heatmap"
    records: list[dict[str, Any]] = []
    visualdf.set_greek_symbol_font_enabled(False)
    for spec in _cortical_heatmap_specs(slopes):
        cells = _cortical_heatmap_cells(slopes, spec)
        polar_label = style_config["labels"]["polarity"][str(spec["Polar"])]
        figure = _render_heatmap(
            cells,
            spec,
            config,
            value_column="OutcomeSDEffectPer10PctAllocation",
            significance_column="P_holm",
            display_limit=CORTICAL_ALLOCATION_DISPLAY_LIMIT,
            colorbar_label="Outcome SD per +10 pp allocation",
            top_labels=[
                f"{spec['DisplayContext']} | {polar_label} | {phase}"
                for phase in phases
            ],
            right_labels=[str(spec["CorticalRegion"])],
        )
        source_path = figure_root / str(spec["Filename"])
        try:
            _atomic_pdf(
                figure,
                source_path,
                overwrite=overwrite,
                dpi=style_config["style"]["dpi"],
            )
        finally:
            visualdf.plt.close(figure)
        records.append(
            {
                **spec,
                "SourcePath": str(source_path),
                "PublishedRelativePath": f"heatmap/{spec['Filename']}",
                "SourceSlopeTable": str(
                    config.cortical_allocation_lmm_root / "results" / "phase_slopes.csv"
                ),
                "PhasePanels": ";".join(phases),
                "ValueColumn": "OutcomeSDEffectPer10PctAllocation",
                "SignificanceColumn": "P_holm",
                "Renderer": "lfp_viz.visualdf.plot_well_heatmap_grid_df",
                "StyleConfig": str(style_config.path),
                "ColorMap": style_config["style"]["heatmap"]["colormap"],
                "DisplayMin": -CORTICAL_ALLOCATION_DISPLAY_LIMIT,
                "DisplayMax": CORTICAL_ALLOCATION_DISPLAY_LIMIT,
                "DisplayClippedCells": int(
                    pd.to_numeric(
                        cells["OutcomeSDEffectPer10PctAllocation"], errors="coerce"
                    )
                    .abs()
                    .gt(CORTICAL_ALLOCATION_DISPLAY_LIMIT)
                    .sum()
                ),
                "Status": "complete",
            }
        )
    return pd.DataFrame(records)


def _cortical_adjacent_grid_specs(
    config: StructuralConnectivityConfig,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for polar in ("Anodal", "Cathodal"):
        for context in CORTICAL_ALLOCATION_CONTEXTS:
            has_aperiodic = context in {"STN", "SNr"}
            suffix = "-plus-aperiodic" if has_aperiodic else ""
            filename = (
                "_".join(
                    _sanitize(value)
                    for value in (
                        "allocation-grid",
                        f"context-{context}",
                        f"polar-{polar}",
                        f"orientation-{CORTICAL_ADJACENT_ORIENTATION}{suffix}",
                    )
                )
                + ".pdf"
            )
            records.append(
                {
                    "OutcomeChangeBasis": "adjacent_phase",
                    "DisplayContext": context,
                    "OutcomeDomain": (
                        "local" if context in {"STN", "SNr"} else "connectivity"
                    ),
                    "Polar": polar,
                    "Lat": "Ipsi",
                    "Orientation": (
                        "metric-by-band-plus-aperiodic"
                        if has_aperiodic
                        else CORTICAL_ADJACENT_ORIENTATION
                    ),
                    "Layout": "cortical-region-grid",
                    "FigureType": "heatmap",
                    "SelectionRole": "all_heatmap",
                    "Filename": filename,
                }
            )
    for index, record in enumerate(records, start=1):
        record["FigureID"] = f"CALAHEAT{index:03d}"
    return records


def _cortical_adjacent_grid_cells(
    slopes: pd.DataFrame,
    spec: Mapping[str, Any],
    *,
    regions: list[str],
) -> pd.DataFrame:
    selector = (
        slopes["CorticalRegion"].isin(regions)
        & slopes["Polar"].eq(spec["Polar"])
        & slopes["Lat"].eq(spec["Lat"])
        & slopes["OutcomeDomain"].eq(spec["OutcomeDomain"])
    )
    context = str(spec["DisplayContext"])
    if context in {"STN", "SNr"}:
        selector &= slopes["Region"].eq(context)
    cells = slopes.loc[selector].copy()
    cells["DisplayMetric"] = [
        _display_metric_key(metric, feature_output)
        for metric, feature_output in cells[["Metric", "FeatureOutput"]].itertuples(
            index=False, name=None
        )
    ]
    cells["StandardizedSlopeDisplay"] = pd.to_numeric(
        cells["OutcomeSDEffectPer10PctAllocation"], errors="coerce"
    )
    return cells


def _cortical_adjacent_metric_contract(
    style_config: Any,
    *,
    context: str,
) -> tuple[list[str], dict[str, str], list[Mapping[str, Any]]]:
    group = "local" if context in {"STN", "SNr"} else "connectivity"
    columns = list(style_config["matrices"]["banded"]["groups"][group]["columns"])
    order = [
        _display_metric_key(column["metric"], column["feature_output"])
        for column in columns
    ]
    labels = {
        _display_metric_key(column["metric"], column["feature_output"]): str(
            column["label"]
        )
        for column in columns
    }
    return order, labels, columns


def _validate_cortical_adjacent_grid_cells(
    cells: pd.DataFrame,
    *,
    context: str,
    regions: list[str],
    phases: list[str],
    bands: list[str],
    metric_columns: list[Mapping[str, Any]],
    aperiodic_parameters: list[str],
) -> None:
    key = ["CorticalRegion", "Phase", "Metric", "FeatureOutput", "Band"]
    if cells.duplicated(key).any():
        raise ValueError(
            "Adjacent cortical-allocation heatmap cells must be unique within "
            "each cortical region and Phase."
        )
    expected = {
        (region, phase, str(column["metric"]), str(column["feature_output"]), band)
        for region in regions
        for phase in phases
        for column in metric_columns
        for band in bands
    }
    if context in {"STN", "SNr"}:
        expected.update(
            {
                (region, phase, "aperiodic", "mean-scalar", parameter)
                for region in regions
                for phase in phases
                for parameter in aperiodic_parameters
            }
        )
    observed = set(cells[key].itertuples(index=False, name=None))
    if observed != expected:
        missing = sorted(expected.difference(observed))
        extra = sorted(observed.difference(expected))
        raise ValueError(
            "Adjacent cortical-allocation heatmap cells do not match the "
            f"registered contract; missing={missing[:3]}, extra={extra[:3]}."
        )


def _render_cortical_adjacent_heatmaps(
    slopes: pd.DataFrame,
    config: StructuralConnectivityConfig,
    *,
    overwrite: bool,
) -> pd.DataFrame:
    _, style_config = _visual_styles(config)
    style = style_config["style"]
    geometry = style["geometry_mm"]
    strips = style["strips"]
    heatmap = style["heatmap"]
    settings = config["inference"]["cortical_allocation_lmm_adjacent"]
    regions = [str(value) for value in settings["regions"]]
    region_labels = ["Premotor" if value == "premotor" else value for value in regions]
    phases = [str(value) for value in settings["phase_levels"]]
    phase_labels = [
        config["inference"]["contact_lmm_adjacent"]["visualization"][
            "phase_display_labels"
        ].get(phase, phase)
        for phase in phases
    ]
    bands = [str(value) for value in config["outcomes"]["bands"]]
    band_labels = {
        str(row["value"]): str(row["label"])
        for row in style_config["matrices"]["banded"]["rows"]
    }
    aperiodic_parameters, aperiodic_labels = _aperiodic_parameter_contract(style_config)
    limit = CORTICAL_ALLOCATION_DISPLAY_LIMIT
    cmap = getattr(visualdf.cm, heatmap["colormap"]).copy()
    cmap.set_bad("#d9d9d9")
    cell_size = tuple(float(value) for value in geometry["cell_size"]["banded"])
    figure_root = config.cortical_allocation_lmm_adjacent_root / "figures" / "heatmap"
    records: list[dict[str, Any]] = []
    visualdf.set_greek_symbol_font_enabled(False)

    for spec in _cortical_adjacent_grid_specs(config):
        context = str(spec["DisplayContext"])
        cells = _cortical_adjacent_grid_cells(
            slopes,
            spec,
            regions=regions,
        )
        metric_order, metric_labels, metric_columns = (
            _cortical_adjacent_metric_contract(
                style_config,
                context=context,
            )
        )
        _validate_cortical_adjacent_grid_cells(
            cells,
            context=context,
            regions=regions,
            phases=phases,
            bands=bands,
            metric_columns=metric_columns,
            aperiodic_parameters=aperiodic_parameters,
        )

        value_grid: list[list[pd.DataFrame]] = []
        adjusted_p_grid: list[list[pd.DataFrame]] = []
        aperiodic_value_grid: list[list[pd.DataFrame]] = []
        aperiodic_adjusted_p_grid: list[list[pd.DataFrame]] = []
        for phase in phases:
            value_row: list[pd.DataFrame] = []
            adjusted_p_row: list[pd.DataFrame] = []
            aperiodic_value_row: list[pd.DataFrame] = []
            aperiodic_adjusted_p_row: list[pd.DataFrame] = []
            for region in regions:
                panel = cells.loc[cells["CorticalRegion"].eq(region)]
                value, adjusted_p = _heatmap_panel(
                    panel,
                    phase=phase,
                    bands=bands,
                    metric_order=metric_order,
                    band_labels=band_labels,
                    metric_labels=metric_labels,
                    orientation=CORTICAL_ADJACENT_ORIENTATION,
                )
                value_row.append(value)
                adjusted_p_row.append(adjusted_p)
                if context in {"STN", "SNr"}:
                    aperiodic_value, aperiodic_adjusted_p = _aperiodic_heatmap_panel(
                        panel,
                        phase=phase,
                        parameters=aperiodic_parameters,
                        parameter_labels=aperiodic_labels,
                    )
                    aperiodic_value_row.append(aperiodic_value)
                    aperiodic_adjusted_p_row.append(aperiodic_adjusted_p)
            value_grid.append(value_row)
            adjusted_p_grid.append(adjusted_p_row)
            if context in {"STN", "SNr"}:
                aperiodic_value_grid.append(aperiodic_value_row)
                aperiodic_adjusted_p_grid.append(aperiodic_adjusted_p_row)

        plot_values = value_grid
        plot_adjusted_p = adjusted_p_grid
        plot_right_labels = phase_labels
        plot_panel_gap = tuple(
            float(value) for value in geometry["panel_gap"]["lat_rows_phase_columns"]
        )
        plot_show_x_ticklabels = "bottom"
        aperiodic_column = -1
        if context in {"STN", "SNr"}:
            plot_values = []
            plot_adjusted_p = []
            for phase_index, (banded_row, banded_p_row) in enumerate(
                zip(value_grid, adjusted_p_grid, strict=True)
            ):
                plot_values.append(banded_row)
                plot_adjusted_p.append(banded_p_row)
                padded_value_row: list[pd.DataFrame] = []
                padded_p_row: list[pd.DataFrame] = []
                for region_index, (aperiodic_value, aperiodic_p) in enumerate(
                    zip(
                        aperiodic_value_grid[phase_index],
                        aperiodic_adjusted_p_grid[phase_index],
                        strict=True,
                    )
                ):
                    padded_value, padded_p, center = _pad_aperiodic_heatmap_panel(
                        aperiodic_value,
                        aperiodic_p,
                        band_columns=banded_row[region_index].columns,
                    )
                    if aperiodic_column not in {-1, center}:
                        raise ValueError(
                            "Adjacent cortical-allocation aperiodic panels have "
                            "inconsistent centers."
                        )
                    aperiodic_column = center
                    padded_value_row.append(padded_value)
                    padded_p_row.append(padded_p)
                plot_values.append(padded_value_row)
                plot_adjusted_p.append(padded_p_row)
            plot_right_labels = [""] * len(plot_values)
            plot_panel_gap = (
                float(geometry["panel_gap"]["lat_rows_phase_columns"][0]),
                4.0,
            )
            plot_show_x_ticklabels = "all"

        polar_symbol = {
            "Anodal": r"$\oplus$",
            "Cathodal": r"$\ominus$",
        }[str(spec["Polar"])]
        colorbar_label = (
            f"Standardized slope per +10 pp allocation | {context} | {polar_symbol}"
        )
        figure = visualdf.plot_well_heatmap_grid_df(
            df_values=plot_values,
            df_ps=plot_adjusted_p,
            x_label=None,
            y_label=None,
            xtick_rotation=90,
            ytick_rotation=0,
            title=None,
            label_top=region_labels,
            label_right=plot_right_labels,
            font_family=style["font"]["family"],
            cmap=cmap,
            vmin=-limit,
            vmax=limit,
            vmode="sym",
            dpi=style["dpi"],
            cellsize=cell_size,
            panel_gap_mm=plot_panel_gap,
            label_top_height_mm=geometry["strip_top_height"],
            label_right_width_mm=geometry["strip_right_width"],
            strip_pad_mm=geometry["strip_pad"],
            label_top_bg_color=strips["top_background"],
            label_right_bg_color=strips["right_background"],
            label_text_color=strips["text_color"],
            label_fontsize=style["font"]["strip_pt"],
            label_fontweight=strips["font_weight"],
            include_global_label_margins=False,
            cbar_label_offset_mm=geometry["colorbar_label_offset"],
            grid_lines=None,
            gline_color=style["grid"]["color"],
            gline_width=style["grid"]["width_pt"],
            gline_alpha=style["grid"]["alpha"],
            axis_label_fontsize=style["font"]["strip_pt"],
            tick_label_fontsize=style["font"]["tick_pt"],
            colorbar_label=colorbar_label,
            colorbar_ticks=[-limit, 0.0, limit],
            colorbar_width_mm=geometry["colorbar_width"],
            colorbar_pad_mm=geometry["colorbar_pad"],
            show_p=True,
            hide_ns=style["significance"]["hide_non_significant"],
            p_text_color=style["significance"]["text_color"],
            p_text_size=style["font"]["star_pt"],
            show_x_ticklabels=plot_show_x_ticklabels,
            show_y_ticklabels="left",
            transparent=style["transparent"],
        )
        if context in {"STN", "SNr"}:
            _format_combined_atlas_grid_heatmap(
                figure,
                phase_labels=phase_labels,
                atlas_count=len(regions),
                aperiodic_column=aperiodic_column,
                cell_width_mm=cell_size[0],
                label_right_bg_color=strips["right_background"],
                label_text_color=strips["text_color"],
                label_fontsize=style["font"]["strip_pt"],
                label_fontweight=strips["font_weight"],
            )

        source_path = figure_root / str(spec["Filename"])
        try:
            _atomic_pdf(
                figure,
                source_path,
                overwrite=overwrite,
                dpi=style["dpi"],
            )
        finally:
            visualdf.plt.close(figure)
        width_mm, height_mm = _pdf_page_dimensions_mm(source_path)
        if (
            width_mm > CORTICAL_ADJACENT_MAX_WIDTH_MM
            or height_mm > CORTICAL_ADJACENT_MAX_HEIGHT_MM
        ):
            raise ValueError(
                "Adjacent cortical-allocation heatmap exceeds the registered "
                f"PDF size: {width_mm:.2f} x {height_mm:.2f} mm."
            )

        has_aperiodic = context in {"STN", "SNr"}
        displayed_metrics = list(metric_order)
        if has_aperiodic:
            displayed_metrics.append("aperiodic/mean-scalar")
        records.append(
            {
                **spec,
                "SourcePath": str(source_path),
                "PublishedRelativePath": (f"adjacent_phase/heatmap/{spec['Filename']}"),
                "SourceSlopeTable": str(
                    config.cortical_allocation_lmm_adjacent_root
                    / "results"
                    / "phase_slopes.csv"
                ),
                "CorticalRegionColumns": ";".join(regions),
                "PhaseRows": ";".join(phases),
                "Bands": ";".join(bands),
                "Metrics": ";".join(displayed_metrics),
                "MetricBlocks": ("banded;aperiodic" if has_aperiodic else "banded"),
                "AperiodicParameters": (
                    ";".join(aperiodic_parameters) if has_aperiodic else ""
                ),
                "ValueColumn": "OutcomeSDEffectPer10PctAllocation",
                "SignificanceColumn": "P_holm",
                "ColorbarLabel": colorbar_label,
                "Renderer": "lfp_viz.visualdf.plot_well_heatmap_grid_df",
                "StyleConfig": str(style_config.path),
                "ColorMap": heatmap["colormap"],
                "DisplayMin": -limit,
                "DisplayMax": limit,
                "DisplayClippedCells": int(
                    cells["StandardizedSlopeDisplay"].abs().gt(limit).sum()
                ),
                "WidthMM": width_mm,
                "HeightMM": height_mm,
                "Status": "complete",
            }
        )
    return pd.DataFrame(records)


def _merge_cortical_figure_registries(
    current: pd.DataFrame,
    adjacent: pd.DataFrame,
) -> pd.DataFrame:
    if current.empty:
        combined = adjacent.copy()
    else:
        selector = ~current["PublishedRelativePath"].astype(str).str.startswith(
            "adjacent_phase/"
        )
        combined = pd.concat(
            [current.loc[selector].copy(), adjacent.copy()],
            ignore_index=True,
            sort=False,
        )
    if combined["FigureID"].duplicated().any():
        raise ValueError("Cortical-allocation figure IDs must be unique.")
    if combined["PublishedRelativePath"].duplicated().any():
        raise ValueError("Cortical-allocation publication paths must be unique.")
    return combined


def run_cortical_allocation_adjacent_figures(
    config: StructuralConnectivityConfig,
    *,
    dry_run: bool,
    overwrite: bool,
) -> dict[str, Any]:
    """Render six grids from completed adjacent cortical-allocation slopes."""

    root = config.cortical_allocation_lmm_adjacent_root
    slopes = _read_csv(
        root / "results" / "phase_slopes.csv",
        {
            "CorticalRegion",
            "Polar",
            "OutcomeDomain",
            "Metric",
            "FeatureOutput",
            "Region",
            "PairRegion",
            "Band",
            "Lat",
            "Phase",
            "P_holm",
            "OutcomeChangeBasis",
            "OutcomeSDEffectPer10PctAllocation",
        },
        "Adjacent cortical-allocation Phase slopes",
    )
    if set(slopes["OutcomeChangeBasis"].astype(str)) != {"adjacent_phase"}:
        raise ValueError(
            "Adjacent cortical-allocation figures require adjacent_phase slopes."
        )
    specs = _cortical_adjacent_grid_specs(config)
    summary = {
        "status": "ready" if dry_run else "complete",
        "heatmap_figures": int(len(specs)),
        "figure_rows": int(len(specs)),
        "mixed_model_refit": False,
        "multiplicity_recalculation": False,
        "writes": 0,
    }
    if dry_run:
        return summary

    adjacent_registry_path = root / "figures.csv"
    combined_registry_path = config.cortical_allocation_lmm_root / "figures.csv"
    existing_paths = [
        path
        for path in (adjacent_registry_path, combined_registry_path)
        if path.exists()
    ]
    if existing_paths and not overwrite:
        raise FileExistsError(
            "Cortical-allocation figure registry exists; pass --overwrite to "
            f"replace it: {existing_paths[0]}"
        )

    adjacent = _render_cortical_adjacent_heatmaps(
        slopes,
        config,
        overwrite=overwrite,
    )
    if len(adjacent) != len(specs):
        raise ValueError(
            "Adjacent cortical-allocation renderer did not produce six figures."
        )
    current = (
        pd.read_csv(combined_registry_path, low_memory=False)
        if combined_registry_path.is_file()
        else pd.DataFrame()
    )
    combined = _merge_cortical_figure_registries(current, adjacent)
    atomic_csv(
        adjacent,
        adjacent_registry_path,
        overwrite=overwrite,
    )
    atomic_csv(
        combined,
        combined_registry_path,
        overwrite=overwrite,
    )
    summary["combined_registry_rows"] = int(len(combined))
    summary["writes"] = int(len(adjacent) + 2)
    return summary


def _cortical_fit_models(slopes: pd.DataFrame, alpha: float) -> list[str]:
    significant = slopes.loc[
        slopes["OutcomeDomain"].eq("connectivity")
        & pd.to_numeric(slopes["P_holm"], errors="coerce").lt(alpha)
    ]
    return sorted(significant["ModelID"].astype(str).unique())


def _render_cortical_fits(
    model_ids: list[str],
    model_rows: pd.DataFrame,
    curves: pd.DataFrame,
    slopes: pd.DataFrame,
    config: StructuralConnectivityConfig,
    *,
    overwrite: bool,
) -> pd.DataFrame:
    scalar_style, _ = _visual_styles(config)
    style = scalar_style["style"]
    font = style["font"]
    widths = style["line_width_pt"]
    geometry = style["geometry_mm"]
    strips = style["strips"]
    phases = list(config["outcomes"]["phases"])
    figure_root = _cortical_figure_root(config) / "figures" / "fit"
    records: list[dict[str, Any]] = []
    visualdf.set_greek_symbol_font_enabled(False)
    for index, model_id in enumerate(model_ids, start=1):
        raw = model_rows.loc[
            model_rows["ModelID"].eq(model_id)
            & model_rows["RowEligibilityStatus"].eq("eligible")
        ].copy()
        curve = curves.loc[curves["ModelID"].eq(model_id)].copy()
        slope = slopes.loc[slopes["ModelID"].eq(model_id)].copy()
        if raw.empty or curve.empty or len(slope) != len(phases):
            raise ValueError(
                f"Cortical-allocation figure inputs are incomplete for {model_id}."
            )
        metadata = slope.iloc[0]
        pair_region = str(metadata["PairRegion"])
        region_label = scalar_style["labels"]["region"][pair_region]
        polar_label = scalar_style["labels"]["polarity"][str(metadata["Polar"])]
        cortical_region = str(metadata["CorticalRegion"])
        panel_levels = [
            f"{cortical_region} | {region_label} | {polar_label} | {phase}"
            for phase in phases
        ]
        panel_map = dict(zip(phases, panel_levels, strict=True))
        for table in (raw, curve, slope):
            table["PhasePanel"] = table["Phase"].map(panel_map)
            table["CorticalRegionLabel"] = cortical_region
        curve["emmean"] = pd.to_numeric(curve["PredictedValue"], errors="coerce")
        curve["lower.CL"] = pd.to_numeric(curve["Lower"], errors="coerce")
        curve["upper.CL"] = pd.to_numeric(curve["Upper"], errors="coerce")
        slope["EffectLowerPer10Pct"] = pd.to_numeric(
            slope["LowerPer10PctAllocation"], errors="coerce"
        ) / pd.to_numeric(slope["ValueSD"], errors="coerce")
        slope["EffectUpperPer10Pct"] = pd.to_numeric(
            slope["UpperPer10PctAllocation"], errors="coerce"
        ) / pd.to_numeric(slope["ValueSD"], errors="coerce")
        figure = visualdf.plot_triple_interaction_fit(
            df=raw,
            curve=curve,
            slope=slope,
            value_col="Value",
            x_var="RelativeAllocation",
            panel_var="PhasePanel",
            facet_var="CorticalRegionLabel",
            color_var="Phase",
            panel_levels=panel_levels,
            facet_levels=[cortical_region],
            x_label=f"{cortical_region} relative allocation",
            y_label=y_axis_label(metadata, scalar_style),
            title=None,
            font_family=font["latin"],
            palette=style["phase_palette"],
            jitter_alpha=style["jitter"]["alpha"],
            jitter_size=style["jitter"]["size"],
            curve_line_width=widths["emm_line"],
            curve_line_color=style["emm"]["color"],
            ribbon_alpha=style["box"]["fill_alpha"],
            show_ribbon=True,
            grid=style["axes"]["grid"],
            dpi=style["dpi"],
            seed=style["jitter"]["seed"],
            show_top_right_axes=style["axes"]["show_top_right_spines"],
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
            legend_loc="none",
            boxsize=(30.0, float(geometry["panel_height"])),
            panel_gap=tuple(geometry["panel_gap"]),
            x_label_offset_mm=geometry["x_label_offset"],
            y_label_offset_mm=geometry["y_label_offset"],
            slope_beta_col="OutcomeSDEffectPer10PctAllocation",
            slope_lower_col="EffectLowerPer10Pct",
            slope_upper_col="EffectUpperPer10Pct",
            slope_p_col="P_holm",
            slope_text_fmt="ΔSD/+10 pp = {beta:.2f}\npHolm = {p:.3g} ({stars})",
            slope_text_loc="upper left",
            slope_text_box_alpha=0.0,
            transparent=style["transparent"],
        )
        bits = [
            f"model-{model_id}",
            f"cortical-{cortical_region}",
            f"polar-{metadata['Polar']}",
            f"metric-{metadata['Metric']}",
            f"pair-{metadata['PairRegion']}",
            f"band-{metadata['Band']}",
        ]
        filename = "_".join(_sanitize(value) for value in bits) + ".pdf"
        source_path = figure_root / filename
        try:
            _atomic_pdf(
                figure,
                source_path,
                overwrite=overwrite,
                dpi=style["dpi"],
            )
        finally:
            visualdf.plt.close(figure)
        records.append(
            {
                "FigureID": f"CALFIT{index:03d}",
                "FigureType": "fit",
                "SelectionRole": "corrected_significant_connectivity_fit",
                "ModelID": model_id,
                "CorticalRegion": cortical_region,
                "OutcomeDomain": metadata["OutcomeDomain"],
                "Metric": metadata["Metric"],
                "OutcomeScale": metadata["OutcomeScale"],
                "Polar": metadata["Polar"],
                "Region": metadata["Region"],
                "PairRegion": metadata["PairRegion"],
                "Band": metadata["Band"],
                "FitStatus": metadata["FitStatus"],
                "SourcePath": str(source_path),
                "PublishedRelativePath": f"fit/{filename}",
                "SourcePointTable": str(
                    config.cortical_allocation_lmm_root
                    / "tables"
                    / "contact_model_rows.csv"
                ),
                "SourceCurveTable": str(
                    config.cortical_allocation_lmm_root
                    / "results"
                    / "prediction_grid.csv"
                ),
                "SourceSlopeTable": str(
                    config.cortical_allocation_lmm_root / "results" / "phase_slopes.csv"
                ),
                "PhasePanels": ";".join(phases),
                "ValueColumn": "Value",
                "PredictorColumn": "RelativeAllocation",
                "SlopeColumn": "OutcomeSDEffectPer10PctAllocation",
                "SignificanceColumn": "P_holm",
                "Renderer": "lfp_viz.visualdf.plot_triple_interaction_fit",
                "StyleConfig": str(scalar_style.path),
                "Status": "complete",
            }
        )
    return pd.DataFrame(records)


def run_cortical_allocation_figures(
    config: StructuralConnectivityConfig,
    *,
    dry_run: bool,
    overwrite: bool,
) -> dict[str, Any]:
    """Render registered figures from existing cortical-allocation tables."""

    root = config.cortical_allocation_lmm_root
    model_rows = _read_csv(
        root / "tables" / "contact_model_rows.csv",
        {
            "ModelID",
            "ID",
            "Phase",
            "Value",
            "RelativeAllocation",
            "RowEligibilityStatus",
        },
        "Cortical-allocation model rows",
    )
    curves = _read_csv(
        root / "results" / "prediction_grid.csv",
        {
            "ModelID",
            "Phase",
            "RelativeAllocation",
            "PredictedValue",
            "Lower",
            "Upper",
        },
        "Cortical-allocation prediction grid",
    )
    slopes = _read_csv(
        root / "results" / "phase_slopes.csv",
        {
            "ModelID",
            "CorticalRegion",
            "Polar",
            "OutcomeDomain",
            "Metric",
            "OutcomeScale",
            "Region",
            "PairRegion",
            "Band",
            "Phase",
            "P_holm",
            "Status",
            "FitStatus",
            "ValueSD",
            "LowerPer10PctAllocation",
            "UpperPer10PctAllocation",
            "OutcomeSDEffectPer10PctAllocation",
        },
        "Cortical-allocation Phase slopes",
    )
    _visual_styles(config)
    model_ids = _cortical_fit_models(slopes, float(config["inference"]["alpha"]))
    heatmap_count = len(_cortical_heatmap_specs(slopes))
    summary = {
        "status": "ready" if dry_run else "complete",
        "heatmap_figures": int(heatmap_count),
        "fit_figures": int(len(model_ids)),
        "figure_rows": int(heatmap_count + len(model_ids)),
        "mixed_model_refit": False,
        "writes": 0,
    }
    if dry_run:
        return summary
    registry_path = root / "figures.csv"
    if registry_path.exists() and not overwrite:
        raise FileExistsError(
            "Cortical-allocation figure registry exists; pass --overwrite to "
            f"replace it: {registry_path}"
        )
    heatmaps = _render_cortical_heatmaps(slopes, config, overwrite=overwrite)
    fits = _render_cortical_fits(
        model_ids,
        model_rows,
        curves,
        slopes,
        config,
        overwrite=overwrite,
    )
    figures = pd.concat([heatmaps, fits], ignore_index=True, sort=False)
    adjacent_registry_path = (
        config.cortical_allocation_lmm_adjacent_root / "figures.csv"
    )
    if adjacent_registry_path.is_file():
        adjacent = _read_csv(
            adjacent_registry_path,
            {"FigureID", "PublishedRelativePath", "SourcePath", "Status"},
            "Adjacent cortical-allocation figure registry",
        )
        figures = _merge_cortical_figure_registries(figures, adjacent)
    elif figures["FigureID"].duplicated().any():
        raise ValueError("Cortical-allocation figure IDs must be unique.")
    elif figures["PublishedRelativePath"].duplicated().any():
        raise ValueError("Cortical-allocation publication paths must be unique.")
    atomic_csv(figures, registry_path, overwrite=overwrite)
    summary["writes"] = int(len(figures) + 1)
    return summary
