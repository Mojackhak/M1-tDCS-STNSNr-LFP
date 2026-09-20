"""Build and render registered signed-logP heatmaps through visualdf."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import visualdf
from .config import VizConfig
from .data import path_below, read_csv, require_columns
from .plan import VizPlan

RESULT_IDENTITY_COLUMNS = {
    "AnalysisID",
    "ModelID",
    "TestKind",
    "TestID",
    "Engine",
}


def _uses_polar_model_strata(config: VizConfig) -> bool:
    return (
        config["inputs"]["test_kind"] == "emm_pairwise"
        or config.stats_analysis_id == "region-by-phase-lat"
        or config.stats_analysis_id == "laterality-within-polar"
    )


def _uses_polar_output_strata(config: VizConfig) -> bool:
    return _uses_polar_model_strata(config) and (
        config.stats_analysis_id != "laterality-within-polar"
    )


def _uses_fixed_lat_model_strata(config: VizConfig) -> bool:
    return config.stats_analysis_id == "phase-within-lat"


def _conditional_factor(config: VizConfig) -> str:
    factors = [name for name in config["factor_levels"] if name != "Phase"]
    if len(factors) != 1:
        raise ValueError(
            "Heatmap models must contain Phase and one conditional factor."
        )
    return factors[0]


def _matrix_specs(config: VizConfig):
    banded = config["matrices"]["banded"]
    for group_name, group in banded["groups"].items():
        for region in group["regions"]:
            for orientation in config["matrices"]["orientations"]["banded"]:
                yield {
                    "view": "banded",
                    "orientation": orientation,
                    "group": group_name,
                    "domain": group["domain"],
                    "region": region,
                    "rows": banded["rows"],
                    "columns": group["columns"],
                }
    aperiodic = config["matrices"]["aperiodic"]
    for orientation in config["matrices"]["orientations"]["aperiodic"]:
        yield {
            "view": "aperiodic",
            "orientation": orientation,
            "group": "local",
            "domain": aperiodic["domain"],
            "region": aperiodic["display_region"],
            "rows": aperiodic["rows"],
            "columns": aperiodic["columns"],
        }


def _select_model(
    models: pd.DataFrame,
    *,
    config: VizConfig,
    spec: Mapping[str, Any],
    row: Mapping[str, str],
    column: Mapping[str, str],
    polar: str,
    lat: str,
) -> pd.Series:
    source_region = str(column.get("source_region", spec["region"]))
    selected = models.loc[
        models["Domain"].eq(spec["domain"])
        & models["RegionValue"].eq(source_region)
        & models["Metric"].eq(column["metric"])
        & models["FeatureOutput"].eq(column["feature_output"])
        & models["Band"].eq(row["value"])
    ]
    if _uses_polar_model_strata(config):
        selected = selected.loc[selected["Polar"].eq(polar)]
    else:
        selected = selected.loc[selected["Contrast"].eq(config["inputs"]["contrast"])]
    if _uses_fixed_lat_model_strata(config):
        selected = selected.loc[selected["Lat"].eq(lat)]
    if len(selected) != 1:
        raise ValueError(
            "Expected one heatmap model for "
            f"{spec['domain']}/{source_region}/{column['metric']}/"
            f"{column['feature_output']}/{row['value']}/{polar or 'contrast'}/"
            f"{lat or 'joint'}; "
            f"found {len(selected)}."
        )
    return selected.iloc[0]


def _result_table(
    outputs: pd.DataFrame,
    model: pd.Series,
    figure: Mapping[str, Any],
    config: VizConfig,
) -> tuple[Path, pd.DataFrame]:
    inputs = config["inputs"]
    selected = outputs.loc[
        outputs["AnalysisID"].eq(config.stats_analysis_id)
        & outputs["ModelID"].eq(model["ModelID"])
        & outputs["TestKind"].eq(inputs["test_kind"])
        & outputs["TestID"].eq(figure["test_id"])
        & outputs["ResultType"].eq(inputs["result_type"])
    ]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one heatmap result for {model['ModelID']} / "
            f"{figure['test_id']}; found {len(selected)}."
        )
    manifest_row = selected.iloc[0]
    path = path_below(
        Path(str(manifest_row["Path"])), config.stats_root, "Heatmap result path"
    )
    table = read_csv(path, "Heatmap result")
    if len(table) != int(manifest_row["Rows"]):
        raise ValueError(f"Heatmap result row count differs from its manifest: {path}")
    require_columns(table, RESULT_IDENTITY_COLUMNS, f"Heatmap result {path}")
    identity = {
        "AnalysisID": config.stats_analysis_id,
        "ModelID": model["ModelID"],
        "TestKind": inputs["test_kind"],
        "TestID": figure["test_id"],
        "Engine": model["Engine"],
    }
    for column, value in identity.items():
        if not table[column].eq(value).all():
            raise ValueError(
                f"Heatmap result {column} does not match its model: {path}"
            )
    if config.stats_analysis_id == "laterality-within-polar":
        table = table.copy()
        table["Polar"] = str(model["Polar"])
    if config.stats_analysis_id == "pre-by-polar-contact-paired":
        table = table.copy()
        table["Phase"] = "Pre"
    required = {
        inputs["effect_column"],
        inputs["p_column"],
        *inputs["retained_p_columns"],
    }
    if inputs["test_kind"] == "emm_pairwise":
        required.update({"group1", "group2"})
        if figure["test_id"] == "Phase-Lat":
            required.add("Lat")
    else:
        required.update({"Phase", _conditional_factor(config)})
    require_columns(table, required, f"Heatmap result {path}")
    return path, table


def _select_result_row(
    table: pd.DataFrame,
    *,
    config: VizConfig,
    figure: Mapping[str, Any],
    comparison: Mapping[str, str] | None,
    phase: str,
    condition: str,
) -> pd.Series:
    if config["inputs"]["test_kind"] == "emm_pairwise":
        assert comparison is not None
        mask = table["group1"].eq(comparison["group1"]) & table["group2"].eq(
            comparison["group2"]
        )
        if figure["test_id"] == "Phase-Lat":
            mask &= table["Lat"].eq(condition)
    else:
        factor = _conditional_factor(config)
        mask = table["Phase"].eq(phase) & table[factor].eq(condition)
    selected = table.loc[mask]
    if len(selected) != 1:
        detail = (
            comparison["display"] if comparison is not None else f"{phase}/{condition}"
        )
        raise ValueError(
            f"Expected one heatmap inferential row for {detail}; found {len(selected)}."
        )
    return selected.iloc[0]


def _signed_logp(effect: object, p_value: object) -> tuple[float, float]:
    effect_value = float(effect)
    p = float(p_value)
    if not math.isfinite(effect_value):
        raise ValueError("Heatmap effects must be finite.")
    if not math.isfinite(p) or not 0 < p <= 1:
        raise ValueError("Heatmap P values must be finite and satisfy 0 < p <= 1.")
    signed = 0.0 if effect_value == 0 else math.copysign(-math.log10(p), effect_value)
    return effect_value, signed


def _star(p_value: float, config: VizConfig) -> str:
    for threshold in config["style"]["significance"]["thresholds"]:
        if p_value < threshold["p_lt"]:
            return str(threshold["label"])
    return ""


def _panel_coordinates(
    config: VizConfig,
    figure: Mapping[str, Any],
):
    if figure["layout"] == "comparison_rows":
        for row_index, comparison in enumerate(config["labels"]["comparisons"]):
            yield row_index, 0, comparison, "", ""
    elif figure["layout"] == "comparison_rows_lat_columns":
        for row_index, comparison in enumerate(config["labels"]["comparisons"]):
            for column_index, lat in enumerate(config["factor_levels"]["Lat"]):
                yield row_index, column_index, comparison, "", lat
    elif figure["layout"] == "comparison_columns":
        for column_index, comparison in enumerate(config["labels"]["comparisons"]):
            yield 0, column_index, comparison, "", ""
    elif figure["layout"] == "lat_rows_comparison_columns":
        for row_index, lat in enumerate(config["factor_levels"]["Lat"]):
            for column_index, comparison in enumerate(config["labels"]["comparisons"]):
                yield row_index, column_index, comparison, "", lat
    else:
        factor = _conditional_factor(config)
        for row_index, condition in enumerate(config["factor_levels"][factor]):
            for column_index, phase in enumerate(config["factor_levels"]["Phase"]):
                yield row_index, column_index, None, phase, condition


def _top_label(
    config: VizConfig,
    *,
    figure: Mapping[str, Any],
    region: str,
    polar: str,
    phase: str,
    condition: str,
    lat: str,
    comparison: Mapping[str, str] | None,
) -> str:
    if config["inputs"]["test_kind"] == "emmean_vs_null":
        contrast = config["labels"]["contrast"][config["inputs"]["contrast"]]
        if config.stats_analysis_id == "region-by-phase-lat":
            polar_label = config["labels"]["polarity"][polar]
            return f"{phase} | {contrast} | {polar_label}"
        if config.stats_analysis_id == "region-polarity-by-phase-lat":
            region_label = config["labels"]["region"][region]
            return f"{phase} | {region_label} | {contrast}"
        return f"{phase} | {contrast}"
    if figure["layout"] in {
        "comparison_columns",
        "lat_rows_comparison_columns",
    }:
        assert comparison is not None
        return str(comparison["display"])
    region_label = config["labels"]["region"][region]
    polar_label = config["labels"]["polarity"][polar]
    if _uses_fixed_lat_model_strata(config):
        return f"{region_label} | {polar_label} | {lat}"
    if figure["test_id"] == "Phase-Lat":
        return f"{region_label} | {polar_label} | {condition}"
    return f"{region_label} | {polar_label}"


def _right_label(
    config: VizConfig,
    *,
    figure: Mapping[str, Any],
    comparison: Mapping[str, str] | None,
    condition: str,
    region: str,
    polar: str,
    lat: str,
) -> str:
    if config["inputs"]["test_kind"] == "emmean_vs_null":
        if _conditional_factor(config) == "Polar":
            return str(config["labels"]["polarity"][condition])
        return condition
    assert comparison is not None
    if figure["layout"] in {
        "comparison_columns",
        "lat_rows_comparison_columns",
    }:
        region_label = config["labels"]["region"][region]
        polar_label = config["labels"]["polarity"][polar]
        laterality = lat or condition
        if figure["test_id"] == "Phase-Lat":
            return f"{region_label} | {polar_label} | {laterality}"
        return f"{region_label} | {polar_label}"
    return str(comparison["display"])


def _result_values(row: pd.Series, config: VizConfig) -> dict[str, float]:
    values = {"p_raw": math.nan, "p_tukey": math.nan, "p_holm": math.nan}
    for column in config["inputs"]["retained_p_columns"]:
        values[column] = float(row[column])
    return values


def _figure_id(
    config: VizConfig,
    *,
    figure: Mapping[str, Any],
    view: str,
    orientation: str,
    domain: str,
    region: str,
    polar: str,
    lat: str,
) -> str:
    parts = [
        f"analysis-{config.analysis_id}",
        f"test-{figure['id']}",
        f"view-{view}",
        f"orientation-{orientation}",
        f"domain-{domain}",
        f"region-{region}",
    ]
    if polar:
        parts.append(f"polar-{polar}")
    if lat:
        parts.append(f"lat-{lat}")
    return "_".join(parts)


def build_heatmap_plan(
    config: VizConfig, models: pd.DataFrame, outputs: pd.DataFrame
) -> VizPlan:
    """Construct one registered heatmap plan from authoritative stats outputs."""

    result_cache: dict[tuple[str, str], tuple[Path, pd.DataFrame]] = {}
    cell_rows: list[dict[str, Any]] = []
    figure_rows: list[dict[str, Any]] = []
    used_models: set[str] = set()
    polars = (
        list(config["labels"]["polarity"])
        if _uses_polar_output_strata(config)
        else [""]
    )
    lats = (
        list(config["factor_levels"]["Lat"])
        if _uses_fixed_lat_model_strata(config)
        else [""]
    )

    for spec in _matrix_specs(config):
        for polar in polars:
            for lat in lats:
                for figure in config["figures"]:
                    figure_id = _figure_id(
                        config,
                        figure=figure,
                        view=spec["view"],
                        orientation=spec["orientation"],
                        domain=spec["domain"],
                        region=spec["region"],
                        polar=polar,
                        lat=lat,
                    )
                    start = len(cell_rows)
                    for matrix_row_index, matrix_row in enumerate(spec["rows"]):
                        for matrix_column_index, matrix_column in enumerate(
                            spec["columns"]
                        ):
                            base_row_value = matrix_row["value"]
                            base_column_value = str(
                                matrix_column.get(
                                    "source_region",
                                    f"{matrix_column['metric']}/"
                                    f"{matrix_column['feature_output']}",
                                )
                            )
                            base_row_label = matrix_row["label"]
                            base_column_label = matrix_column["label"]
                            transposed = spec["orientation"] in {
                                "parameter-by-band",
                                "region-by-parameter",
                            }
                            if transposed:
                                display_row = matrix_column_index
                                display_column = matrix_row_index
                                display_row_value = base_column_value
                                display_column_value = base_row_value
                                display_row_label = base_column_label
                                display_column_label = base_row_label
                            else:
                                display_row = matrix_row_index
                                display_column = matrix_column_index
                                display_row_value = base_row_value
                                display_column_value = base_column_value
                                display_row_label = base_row_label
                                display_column_label = base_column_label
                            for (
                                panel_row,
                                panel_column,
                                comparison,
                                phase,
                                condition,
                            ) in _panel_coordinates(config, figure):
                                model_polar = (
                                    condition
                                    if config.stats_analysis_id
                                    == "laterality-within-polar"
                                    else polar
                                )
                                model = _select_model(
                                    models,
                                    config=config,
                                    spec=spec,
                                    row=matrix_row,
                                    column=matrix_column,
                                    polar=model_polar,
                                    lat=lat,
                                )
                                model_id = str(model["ModelID"])
                                used_models.add(model_id)
                                cache_key = (model_id, str(figure["test_id"]))
                                if cache_key not in result_cache:
                                    result_cache[cache_key] = _result_table(
                                        outputs, model, figure, config
                                    )
                                result_path, result_table = result_cache[cache_key]
                                result = _select_result_row(
                                    result_table,
                                    config=config,
                                    figure=figure,
                                    comparison=comparison,
                                    phase=phase,
                                    condition=condition,
                                )
                                p_value = float(result[config["inputs"]["p_column"]])
                                effect, signed = _signed_logp(
                                    result[config["inputs"]["effect_column"]], p_value
                                )
                                limit = float(
                                    config["style"]["heatmap"]["value_limits"][1]
                                )
                                retained = _result_values(result, config)
                                conditional_factor = _conditional_factor(config)
                                lat_value = (
                                    lat
                                    if lat
                                    else (
                                        condition if conditional_factor == "Lat" else ""
                                    )
                                )
                                polar_value = (
                                    condition
                                    if conditional_factor == "Polar"
                                    else polar
                                )
                                cell_rows.append(
                                    {
                                        "AnalysisID": config.analysis_id,
                                        "StatsAnalysisID": config.stats_analysis_id,
                                        "FigureID": figure_id,
                                        "TestKind": config["inputs"]["test_kind"],
                                        "TestID": figure["test_id"],
                                        "View": spec["view"],
                                        "Orientation": spec["orientation"],
                                        "Domain": spec["domain"],
                                        "RegionVariable": model["RegionVariable"],
                                        "RegionValue": model["RegionValue"],
                                        "DisplayRegionValue": spec["region"],
                                        "Contrast": model.get("Contrast", ""),
                                        "PanelRow": panel_row,
                                        "PanelColumn": panel_column,
                                        "PanelTopLabel": _top_label(
                                            config,
                                            figure=figure,
                                            region=spec["region"],
                                            polar=polar,
                                            phase=phase,
                                            condition=condition,
                                            lat=lat,
                                            comparison=comparison,
                                        ),
                                        "PanelRightLabel": _right_label(
                                            config,
                                            figure=figure,
                                            comparison=comparison,
                                            condition=condition,
                                            region=spec["region"],
                                            polar=polar,
                                            lat=lat,
                                        ),
                                        "Phase": phase,
                                        "Lat": lat_value,
                                        "Polar": polar_value,
                                        "group1": (
                                            comparison["group1"]
                                            if comparison is not None
                                            else ""
                                        ),
                                        "group2": (
                                            comparison["group2"]
                                            if comparison is not None
                                            else ""
                                        ),
                                        "Comparison": (
                                            comparison["display"]
                                            if comparison is not None
                                            else ""
                                        ),
                                        "MatrixRow": display_row,
                                        "MatrixColumn": display_column,
                                        "MatrixRowValue": display_row_value,
                                        "MatrixColumnValue": display_column_value,
                                        "MatrixRowLabel": display_row_label,
                                        "MatrixColumnLabel": display_column_label,
                                        "Metric": matrix_column["metric"],
                                        "FeatureOutput": matrix_column[
                                            "feature_output"
                                        ],
                                        "Band": matrix_row["value"],
                                        "ModelID": model_id,
                                        "Engine": model["Engine"],
                                        "ModelStatus": model["Status"],
                                        "Singular": bool(model["Singular"]),
                                        "Effect": effect,
                                        "PValue": p_value,
                                        "p_raw": retained["p_raw"],
                                        "p_tukey": retained["p_tukey"],
                                        "p_holm": retained["p_holm"],
                                        "SignedLogP": signed,
                                        "SignedLogPDisplay": float(
                                            np.clip(signed, -limit, limit)
                                        ),
                                        "Star": _star(p_value, config),
                                        "SourceResultPath": str(result_path),
                                    }
                                )
                    figure_cells = cell_rows[start:]
                    frame = pd.DataFrame(figure_cells)
                    if config.analysis_id in {
                        "submission-polar-contact-signed-logp-heatmap",
                        "submission-pre-contact-signed-logp-heatmap",
                        "submission-lat-region-signed-logp-heatmap",
                        "submission-region-signed-logp-heatmap",
                    }:
                        output_parent = config.output_root / "heatmap"
                    else:
                        output_parent = (
                            config.output_root
                            / "heatmap"
                            / config.stats_analysis_id
                            / str(spec["domain"])
                            / str(spec["region"])
                        )
                        if polar:
                            output_parent /= polar
                        if lat:
                            output_parent /= lat
                    filename = config["outputs"]["filename_template"].format(
                        AnalysisID=config.analysis_id,
                        FigureID=figure["id"],
                        TestID=figure["test_id"],
                        View=spec["view"],
                        Orientation=spec["orientation"],
                        Domain=spec["domain"],
                        RegionValue=spec["region"],
                        Polar=polar,
                        Lat=lat,
                    )
                    figure_rows.append(
                        {
                            "AnalysisID": config.analysis_id,
                            "StatsAnalysisID": config.stats_analysis_id,
                            "FigureID": figure_id,
                            "TestKind": config["inputs"]["test_kind"],
                            "TestID": figure["test_id"],
                            "Layout": figure["layout"],
                            "View": spec["view"],
                            "Orientation": spec["orientation"],
                            "Domain": spec["domain"],
                            "RegionValue": spec["region"],
                            "RegionLabel": config["labels"]["region"][spec["region"]],
                            "Polar": polar,
                            "Lat": lat,
                            "Contrast": config["inputs"].get("contrast", ""),
                            "OutputPath": str(output_parent / filename),
                            "n_source_models": int(frame["ModelID"].nunique()),
                            "n_cells": len(frame),
                            "n_significant": int((frame["PValue"] < 0.05).sum()),
                            "n_singular_models": int(
                                frame.loc[frame["Singular"], "ModelID"].nunique()
                            ),
                            "SourceModelStatuses": ";".join(
                                sorted(frame["ModelStatus"].astype(str).unique())
                            ),
                            "Status": "planned",
                            "Message": "",
                        }
                    )

    figures = pd.DataFrame(figure_rows)
    cells = pd.DataFrame(cell_rows)
    if (
        figures["FigureID"].duplicated().any()
        or figures["OutputPath"].duplicated().any()
    ):
        raise ValueError("Heatmap plan contains duplicate figure identities or paths.")
    expected_models = set(models["ModelID"].astype(str))
    if used_models != expected_models:
        missing = sorted(expected_models - used_models)
        raise ValueError(
            f"Heatmap matrices do not cover all configured models: {missing[:5]}"
        )
    return VizPlan(
        figures=figures,
        coverage=cells,
        model_data={},
        result_tables={},
    )


def _panel_matrix(cells: pd.DataFrame, *, value_column: str) -> pd.DataFrame:
    rows = (
        cells[["MatrixRow", "MatrixRowLabel"]]
        .drop_duplicates()
        .sort_values("MatrixRow")["MatrixRowLabel"]
        .tolist()
    )
    columns = (
        cells[["MatrixColumn", "MatrixColumnLabel"]]
        .drop_duplicates()
        .sort_values("MatrixColumn")["MatrixColumnLabel"]
        .tolist()
    )
    if cells.duplicated(["MatrixRow", "MatrixColumn"]).any():
        raise ValueError("Heatmap panel contains duplicate matrix cells.")
    matrix = cells.pivot(
        index="MatrixRowLabel", columns="MatrixColumnLabel", values=value_column
    )
    return matrix.reindex(index=rows, columns=columns)


def render_well_heatmap_figure(
    cells: pd.DataFrame,
    figure: Mapping[str, Any],
    config: VizConfig,
    *,
    value_column: str,
    significance_column: str,
    geometry_key: str,
):
    """Render one canonical heatmap cell table through the visualdf well grid."""

    panel_rows = sorted(cells["PanelRow"].unique())
    panel_columns = sorted(cells["PanelColumn"].unique())
    value_grid: list[list[pd.DataFrame]] = []
    p_grid: list[list[pd.DataFrame]] = []
    for panel_row in panel_rows:
        value_row: list[pd.DataFrame] = []
        p_row: list[pd.DataFrame] = []
        for panel_column in panel_columns:
            panel = cells.loc[
                cells["PanelRow"].eq(panel_row) & cells["PanelColumn"].eq(panel_column)
            ]
            if panel.empty:
                raise ValueError("Heatmap panel grid contains an empty panel.")
            value_row.append(_panel_matrix(panel, value_column=value_column))
            p_row.append(_panel_matrix(panel, value_column=significance_column))
        value_grid.append(value_row)
        p_grid.append(p_row)

    top_labels = []
    for panel_column in panel_columns:
        labels = cells.loc[
            cells["PanelColumn"].eq(panel_column), "PanelTopLabel"
        ].unique()
        if len(labels) != 1:
            raise ValueError("Heatmap panel column has inconsistent top labels.")
        top_labels.append(str(labels[0]))
    right_labels = []
    for panel_row in panel_rows:
        labels = cells.loc[cells["PanelRow"].eq(panel_row), "PanelRightLabel"].unique()
        if len(labels) != 1:
            raise ValueError("Heatmap panel row has inconsistent right labels.")
        right_labels.append(str(labels[0]))

    style = config["style"]
    geometry = style["geometry_mm"]
    strips = style["strips"]
    heatmap = style["heatmap"]
    colorbar_label = figure.get("ColorbarLabel")
    if colorbar_label is None:
        colorbar_label = style["colorbar"]["label"]
    visualdf.set_greek_symbol_font_enabled(False)
    return visualdf.plot_well_heatmap_grid_df(
        df_values=value_grid,
        df_ps=p_grid,
        x_label=None,
        y_label=None,
        xtick_rotation=90,
        ytick_rotation=0,
        title=None,
        label_top=top_labels,
        label_right=right_labels,
        font_family=style["font"]["family"],
        cmap=getattr(visualdf.cm, heatmap["colormap"]),
        na_color=heatmap.get("na_color"),
        vmin=heatmap["value_limits"][0],
        vmax=heatmap["value_limits"][1],
        vmode="sym",
        dpi=style["dpi"],
        cellsize=tuple(geometry["cell_size"][geometry_key]),
        panel_gap_mm=tuple(geometry["panel_gap"][str(figure["Layout"])]),
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
        colorbar_label=str(colorbar_label),
        colorbar_ticks=style["colorbar"]["ticks"],
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


def render_heatmap_figure(
    cells: pd.DataFrame, figure: Mapping[str, Any], config: VizConfig
):
    """Render one planned signed-logP PDF through the shared well-grid adapter."""

    return render_well_heatmap_figure(
        cells,
        figure,
        config,
        value_column="SignedLogPDisplay",
        significance_column="PValue",
        geometry_key=str(figure["View"]),
    )
