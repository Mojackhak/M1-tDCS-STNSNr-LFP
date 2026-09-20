"""Plan and render configured visualization workflows."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lfp_cohort.io import atomic_csv, atomic_text, read_stage_table

from .config import VizConfig
from .correlation_fit import (
    build_correlation_fit_plan,
    render_correlation_fit_figure,
)
from .correlation_heatmap import (
    build_correlation_heatmap_plan,
    render_correlation_heatmap_figure,
)
from .data import path_below as _path_below
from .data import read_csv as _read_csv
from .data import require_columns as _require_columns
from .heatmap import build_heatmap_plan, render_heatmap_figure
from .lat_trajectory import (
    build_lat_trajectory_plan,
    render_lat_trajectory_figure,
)
from .plan import VizPlan, _filename_slug, build_nested_plan
from .programming import (
    build_programming_plan,
    programming_text_outputs,
    render_programming_figure,
)
from .raw import render_raw_figure
from .scalar import render_scalar_figure, top_strip_label, y_axis_label
from .series import render_series_figure, smooth_trace_table
from .submission_scalar import (
    add_output_groups,
    build_submission_statistics,
    filter_contact_pairing,
)

HEATMAP_KINDS = {"signed_logp_heatmap", "correlation_heatmap"}
DETAIL_MANIFEST_BY_KIND = {
    "signed_logp_heatmap": "cells",
    "correlation_heatmap": "cells",
    "correlation_fit": "points",
}


MODEL_COLUMNS = {
    "AnalysisID",
    "ModelID",
    "Engine",
    "Domain",
    "Metric",
    "FeatureOutput",
    "Representation",
    "Band",
    "RegionVariable",
    "RegionValue",
    "SourcePath",
    "n_rows",
    "n_ID",
    "Singular",
    "Status",
}
OUTPUT_COLUMNS = {
    "AnalysisID",
    "ModelID",
    "TestKind",
    "TestID",
    "ResultType",
    "Path",
    "Rows",
    "Status",
}
RESULT_IDENTITY_COLUMNS = {
    "AnalysisID",
    "ModelID",
    "TestKind",
    "TestID",
    "Engine",
}
SUBMISSION_SCALAR_IDS = {
    "submission-phase-contact-scalar",
    "submission-polar-contact-scalar",
    "submission-pre-contact-scalar",
    "submission-lat-region-scalar",
    "submission-region-scalar",
}


def _validate_manifests(
    config: VizConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    models = _read_csv(config.models_manifest, "Models manifest")
    outputs = _read_csv(config.stats_outputs_manifest, "Stats outputs manifest")
    _require_columns(models, MODEL_COLUMNS, "Models manifest")
    _require_columns(outputs, OUTPUT_COLUMNS, "Stats outputs manifest")
    if models["ModelID"].duplicated().any():
        raise ValueError("Models manifest contains duplicate ModelID values.")
    if not models["AnalysisID"].eq(config.stats_analysis_id).all():
        raise ValueError("Models manifest AnalysisID does not match stats_analysis_id.")
    statuses = set(models["Status"].astype(str))
    accepted = set(config["inputs"]["accepted_model_statuses"])
    unsupported = sorted(statuses - accepted)
    if unsupported:
        raise ValueError(
            f"Models manifest contains unsupported statuses: {unsupported}"
        )
    if not models["Representation"].eq("scalar").all():
        raise ValueError("Visualization models must all use scalar representation.")
    stratum_column = (
        "Contrast"
        if config["inputs"]["test_kind"] == "emmean_vs_null"
        and config.stats_analysis_id != "region-by-phase-lat"
        else "Polar"
    )
    _require_columns(models, {stratum_column}, "Models manifest")
    if config.stats_analysis_id == "phase-within-lat":
        _require_columns(models, {"Lat"}, "Models manifest")
    if config.stats_analysis_id == "laterality-within-polar":
        _require_columns(models, {"Polar"}, "Models manifest")
    return models, outputs


def _model_rows(
    table: pd.DataFrame,
    model: pd.Series,
    config: VizConfig,
    source_path: Path,
    pairing: pd.DataFrame | None = None,
) -> pd.DataFrame:
    region_column = str(model["RegionVariable"])
    factor_columns = list(config["factor_levels"])
    required = {
        "ID",
        "Domain",
        "Metric",
        "FeatureOutput",
        "Representation",
        "Band",
        *factor_columns,
        "Value",
        "IncludeAggregate",
        region_column,
    }
    stratum_column = "Contrast" if "Contrast" in model.index else "Polar"
    required.add(stratum_column)
    _require_columns(table, required, f"Visualization source {source_path}")

    include_values = set(table["IncludeAggregate"].dropna().unique())
    if not include_values.issubset({True, False, np.bool_(True), np.bool_(False)}):
        raise TypeError(f"IncludeAggregate must be boolean: {source_path}")
    numeric = pd.to_numeric(table["Value"], errors="coerce")
    invalid_numeric = table["Value"].notna() & numeric.isna()
    if invalid_numeric.any():
        raise TypeError(f"Scalar Value contains non-numeric values: {source_path}")
    finite = np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan))
    eligible = table.loc[
        table["IncludeAggregate"].fillna(False).astype(bool) & finite
    ].copy()
    eligible["Value"] = numeric.loc[eligible.index].astype(float)

    identity = {
        "Domain": model["Domain"],
        "Metric": model["Metric"],
        "FeatureOutput": model["FeatureOutput"],
        "Representation": model["Representation"],
        "Band": model["Band"],
        region_column: model["RegionValue"],
        stratum_column: model[stratum_column],
    }
    if config.stats_analysis_id == "phase-within-lat":
        identity["Lat"] = model["Lat"]
    if config.stats_analysis_id == "laterality-within-polar":
        identity["Polar"] = model["Polar"]
    mask = pd.Series(True, index=eligible.index)
    for column, value in identity.items():
        mask &= eligible[column].eq(value)
    selected = eligible.loc[mask].copy()

    identity_columns = ["ID", *factor_columns]
    contact_column: str | None = None
    if config["inputs"].get("observation_level", "region") == "contact":
        contact_column = (
            "ContactUnitID"
            if "ContactUnitID" in selected
            else "Channel" if str(model["Domain"]) == "local" else "ChannelPair"
        )
        _require_columns(
            selected, {contact_column}, f"Visualization source {source_path}"
        )
        if selected[contact_column].isna().any():
            raise ValueError(
                "Contact-level visualization source contains missing "
                f"{contact_column}: {source_path}"
            )
        identity_columns.insert(1, contact_column)
        if pairing is not None:
            selected = filter_contact_pairing(
                selected,
                model,
                pairing,
                contact_column=contact_column,
            )
            identity_columns.insert(2, "ContactUnitID")
    expected_rows = int(model["n_rows"])
    if len(selected) != expected_rows:
        raise ValueError(
            f"Visualization source row count differs from models.csv for "
            f"{model['ModelID']}: expected {expected_rows}, found {len(selected)}"
        )
    if selected[identity_columns].isna().any().any():
        raise ValueError(
            f"Visualization source contains missing identities: {source_path}"
        )
    duplicate = selected.duplicated(identity_columns, keep=False)
    if duplicate.any():
        unit = "contact-unit/" if contact_column else ""
        raise ValueError(
            f"Visualization source contains duplicate ID/{unit}Phase/Lat rows: "
            f"{source_path}"
        )
    for factor, levels in config["factor_levels"].items():
        unknown = sorted(set(selected[factor].astype(str)) - set(levels))
        if unknown:
            raise ValueError(
                f"Visualization source contains unknown {factor} levels {unknown}: "
                f"{source_path}"
            )
        selected[factor] = pd.Categorical(
            selected[factor], categories=levels, ordered=True
        )
    output_columns = ["ID", *factor_columns, "Value"]
    if contact_column:
        output_columns.insert(1, contact_column)
    if "ContactUnitID" in selected.columns and contact_column != "ContactUnitID":
        output_columns.insert(2, "ContactUnitID")
    return selected[output_columns].reset_index(drop=True)


def _select_result(
    outputs: pd.DataFrame,
    model: pd.Series,
    figure: dict[str, Any],
    result_type: str,
    config: VizConfig,
) -> tuple[Path, pd.DataFrame]:
    test_kind = config["inputs"]["test_kind"]
    selected = outputs.loc[
        outputs["AnalysisID"].eq(config.stats_analysis_id)
        & outputs["ModelID"].eq(model["ModelID"])
        & outputs["TestKind"].eq(test_kind)
        & outputs["TestID"].eq(figure["test_id"])
        & outputs["ResultType"].eq(result_type)
    ]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one {result_type} result for {model['ModelID']} / "
            f"{figure['test_id']}; found {len(selected)}"
        )
    manifest_row = selected.iloc[0]
    path = _path_below(
        Path(str(manifest_row["Path"])), config.stats_root, "Stats result path"
    )
    table = _read_csv(path, f"{result_type.upper()} result")
    if len(table) != int(manifest_row["Rows"]):
        raise ValueError(f"Stats output row count does not match its manifest: {path}")
    _require_columns(table, RESULT_IDENTITY_COLUMNS, f"Stats result {path}")
    expected_identity = {
        "AnalysisID": config.stats_analysis_id,
        "ModelID": model["ModelID"],
        "TestKind": test_kind,
        "TestID": figure["test_id"],
        "Engine": model["Engine"],
    }
    for column, value in expected_identity.items():
        if not table[column].eq(value).all():
            raise ValueError(f"Stats result {column} does not match its model: {path}")
    if config.stats_analysis_id == "laterality-within-polar":
        table = table.copy()
        table["Polar"] = str(model["Polar"])

    x_var = str(figure["x_var"])
    required = {x_var, "emmean", "lower.CL", "upper.CL"}
    if result_type == "tukey":
        required = {"group1", "group2", config["significance"]["p_column"]}
    elif result_type == "null-test":
        required = {x_var, "null", config["significance"]["p_column"]}
    row_var = figure.get("row_var")
    if row_var and figure.get("row_levels_from_model"):
        table = table.copy()
        table[str(row_var)] = model[str(row_var)]
    if row_var:
        required.add(str(row_var))
    _require_columns(table, required, f"Stats result {path}")
    return path, table


def _coverage_rows(
    raw: pd.DataFrame,
    model: pd.Series,
    figure: dict[str, Any],
    config: VizConfig,
    *,
    display_lat: str | None = None,
) -> list[dict[str, Any]]:
    def support(subset: pd.DataFrame) -> dict[str, int]:
        if "ContactUnitID" in subset.columns:
            n_contact_units = len(
                subset[["ID", "ContactUnitID", "Lat"]].drop_duplicates()
            )
        else:
            n_contact_units = 0
        return {
            "n_ID": int(subset["ID"].nunique()),
            "n_contact_units": n_contact_units,
            "n_obs": int(len(subset)),
        }

    rows: list[dict[str, Any]] = []
    x_var = str(figure["x_var"])
    if x_var != "Phase":
        phase_levels = list(config["factor_levels"]["Phase"])
        if len(phase_levels) != 1:
            raise ValueError("A non-Phase scalar x axis requires one Phase level.")
        for x_level in config["factor_levels"][x_var]:
            subset = raw.loc[raw[x_var].eq(x_level)]
            rows.append(
                {
                    "AnalysisID": config.analysis_id,
                    "StatsAnalysisID": config.stats_analysis_id,
                    "ModelID": model["ModelID"],
                    "TestID": figure["test_id"],
                    "Phase": phase_levels[0],
                    "XVariable": x_var,
                    "XValue": x_level,
                    "RowVariable": "",
                    "RowValue": "",
                    "ColumnVariable": "",
                    "ColumnValue": "",
                    **support(subset),
                }
            )
        return rows
    row_var = figure.get("row_var")
    if display_lat is not None:
        row_var = "Lat"
        row_levels = [display_lat]
    elif row_var and figure.get("row_levels_from_model"):
        row_levels = [str(model[str(row_var)])]
    else:
        row_levels = figure.get("row_levels", [None]) if row_var else [None]
    for row_level in row_levels:
        row_data = raw if row_var is None else raw.loc[raw[row_var].eq(row_level)]
        for phase in config["factor_levels"]["Phase"]:
            subset = row_data.loc[row_data["Phase"].eq(phase)]
            rows.append(
                {
                    "AnalysisID": config.analysis_id,
                    "StatsAnalysisID": config.stats_analysis_id,
                    "ModelID": model["ModelID"],
                    "TestID": figure["test_id"],
                    "Phase": phase,
                    "RowVariable": row_var or "",
                    "RowValue": row_level or "",
                    "ColumnVariable": "",
                    "ColumnValue": "",
                    "View": (
                        "split-lat"
                        if figure.get("layout") == "laterality_singletons"
                        else (
                            "faceted-lat"
                            if figure.get("layout") == "laterality_rows"
                            else (
                                "split-polar"
                                if figure.get("layout") == "polarity_singletons"
                                else (
                                    "faceted-polar"
                                    if figure.get("layout") == "polarity_rows"
                                    else str(figure.get("layout", ""))
                                )
                            )
                        )
                    ),
                    "Lat": row_level if row_var == "Lat" else "",
                    "Polar": row_level if row_var == "Polar" else "",
                    **support(subset),
                }
            )
    return rows


def _result_filename(figure: dict[str, Any], config: VizConfig) -> str:
    return config["outputs"]["filename_template"].format(
        AnalysisID=config.analysis_id,
        TestID=figure["test_id"],
    )


def _validate_model_labels(models: pd.DataFrame, config: VizConfig) -> None:
    for _, model in models.iterrows():
        try:
            y_axis_label(model, config)
            top_strip_label(model, config)
        except (KeyError, ValueError) as error:
            detail = error.args[0] if error.args else str(error)
            raise ValueError(
                "Visualization label is not configured: "
                f"ModelID={model['ModelID']!r}, Metric={model['Metric']!r}, "
                f"FeatureOutput={model['FeatureOutput']!r}, Band={model['Band']!r}, "
                f"RegionValue={model['RegionValue']!r}, "
                f"MissingKey={detail!r}"
            ) from error


def _build_submission_lat_scalar_plan(
    config: VizConfig,
    models: pd.DataFrame,
    outputs: pd.DataFrame,
) -> VizPlan:
    """Combine two independently fitted Polar strata only for display."""

    models = add_output_groups(models)
    _validate_model_labels(models, config)
    group_columns = [
        "Domain",
        "Metric",
        "FeatureOutput",
        "Representation",
        "Band",
        "RegionVariable",
        "RegionValue",
        "Contrast",
        "OutputGroup",
    ]
    expected_polars = list(config["factor_levels"]["Polar"])
    source_cache: dict[Path, pd.DataFrame] = {}
    model_data: dict[str, pd.DataFrame] = {}
    result_tables: dict[tuple[str, str, str], pd.DataFrame] = {}
    figure_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []

    for _, grouped in models.groupby(group_columns, sort=False, observed=True):
        if grouped["Polar"].tolist() != expected_polars:
            by_polar = grouped.set_index("Polar", drop=False)
            if (
                set(by_polar.index) != set(expected_polars)
                or by_polar.index.has_duplicates
            ):
                raise ValueError(
                    "Submission Lat figures require one Anodal and one Cathodal "
                    "model per feature identity."
                )
            grouped = by_polar.loc[expected_polars].reset_index(drop=True)
        models_by_polar = {
            str(model["Polar"]): model for _, model in grouped.iterrows()
        }
        data_key = "||".join(
            str(models_by_polar[polar]["ModelID"]) for polar in expected_polars
        )
        raw_frames: list[pd.DataFrame] = []
        emm_frames: list[pd.DataFrame] = []
        null_frames: list[pd.DataFrame] = []
        source_paths: list[Path] = []
        emm_paths: list[Path] = []
        null_paths: list[Path] = []
        result_figure = config["figures"][0]

        for polar in expected_polars:
            model = models_by_polar[polar]
            source_path = _path_below(
                Path(str(model["SourcePath"])),
                config.source_root,
                "Source PKL",
            )
            if not source_path.exists() or source_path.stat().st_size == 0:
                raise FileNotFoundError(
                    f"Source PKL is missing or empty: {source_path}"
                )
            if source_path not in source_cache:
                source_cache[source_path] = read_stage_table(source_path)
            raw_frames.append(
                _model_rows(
                    source_cache[source_path],
                    model,
                    config,
                    source_path,
                )
            )
            emm_path, emm = _select_result(outputs, model, result_figure, "emm", config)
            null_path, null_test = _select_result(
                outputs, model, result_figure, "null-test", config
            )
            if emm_path.parent != null_path.parent:
                raise ValueError(
                    "EMM and null-test results have different parents for "
                    f"{model['ModelID']}."
                )
            source_paths.append(source_path)
            emm_paths.append(emm_path)
            null_paths.append(null_path)
            emm_frames.append(emm)
            null_frames.append(null_test)

        raw = pd.concat(raw_frames, ignore_index=True)
        emm = pd.concat(emm_frames, ignore_index=True)
        null_test = pd.concat(null_frames, ignore_index=True)
        model_data[data_key] = raw
        test_id = str(result_figure["test_id"])
        result_tables[(data_key, test_id, "emm")] = emm
        result_tables[(data_key, test_id, "null-test")] = null_test

        displays: list[tuple[dict[str, Any], str | None]] = []
        for figure in config["figures"]:
            if figure["layout"] == "polarity_singletons":
                displays.extend((figure, polar) for polar in expected_polars)
            else:
                displays.append((figure, None))

        for figure, displayed_polar in displays:
            anchor_polar = displayed_polar or expected_polars[0]
            anchor = models_by_polar[anchor_polar].copy()
            polar_display = displayed_polar or "Anodal-Cathodal"
            anchor["Polar"] = polar_display
            displayed_raw = (
                raw
                if displayed_polar is None
                else raw.loc[raw["Polar"].astype(str).eq(displayed_polar)]
            )
            displayed_null = (
                null_test
                if displayed_polar is None
                else null_test.loc[null_test["Polar"].astype(str).eq(displayed_polar)]
            )
            p_values = pd.to_numeric(
                displayed_null[config["significance"]["p_column"]],
                errors="coerce",
            )
            filename = config["outputs"]["filename_template"].format(
                Band=_filename_slug(anchor["Band"]),
                RegionValue=_filename_slug(anchor["RegionValue"]),
                Contrast=_filename_slug(anchor["Contrast"]),
                PolarDisplay=_filename_slug(polar_display),
            )
            output_path = config.output_root / str(anchor["OutputGroup"]) / filename
            statuses = grouped["Status"].astype(str).tolist()
            figure_rows.append(
                {
                    "AnalysisID": config.analysis_id,
                    "StatsAnalysisID": config.stats_analysis_id,
                    "ModelID": str(anchor["ModelID"]),
                    "SourceModelIDs": ";".join(grouped["ModelID"].astype(str)),
                    "SingularModelIDs": ";".join(
                        grouped.loc[grouped["Singular"].astype(bool), "ModelID"].astype(
                            str
                        )
                    ),
                    "DataKey": data_key,
                    "FigureID": str(figure["id"]),
                    "View": (
                        "split-polar"
                        if displayed_polar is not None
                        else "faceted-polar"
                    ),
                    "TestKind": config["inputs"]["test_kind"],
                    "TestID": str(figure["test_id"]),
                    "Engine": anchor["Engine"],
                    "ModelStatus": ";".join(statuses),
                    "Singular": bool(grouped["Singular"].astype(bool).any()),
                    "Domain": anchor["Domain"],
                    "Metric": anchor["Metric"],
                    "FeatureOutput": anchor["FeatureOutput"],
                    "OutputGroup": anchor["OutputGroup"],
                    "Representation": anchor["Representation"],
                    "Band": anchor["Band"],
                    "RegionVariable": anchor["RegionVariable"],
                    "RegionValue": anchor["RegionValue"],
                    "Polar": polar_display,
                    "Lat": "",
                    "Contrast": anchor["Contrast"],
                    "SourcePath": ";".join(str(path) for path in source_paths),
                    "EMMPath": ";".join(str(path) for path in emm_paths),
                    "TukeyPath": "",
                    "NullTestPath": ";".join(str(path) for path in null_paths),
                    "OutputPath": str(output_path),
                    "n_rows": len(displayed_raw),
                    "n_ID": int(displayed_raw["ID"].nunique()),
                    "n_contact_units": 0,
                    "n_significant": int((p_values < 0.05).sum()),
                    "Status": "planned",
                    "Message": "",
                }
            )
            coverage_figure = dict(figure)
            if displayed_polar is not None:
                coverage_figure["row_var"] = "Polar"
                coverage_figure["row_levels"] = [displayed_polar]
            coverage_rows.extend(_coverage_rows(raw, anchor, coverage_figure, config))

    figures = pd.DataFrame(figure_rows)
    coverage = pd.DataFrame(coverage_rows)
    if figures["OutputPath"].duplicated().any():
        raise ValueError("Figure plan contains duplicate output paths.")
    return VizPlan(
        figures=figures,
        coverage=coverage,
        model_data=model_data,
        result_tables=result_tables,
        submission_tables=build_submission_statistics(config, models, outputs),
    )


def _build_scalar_plan(config: VizConfig) -> VizPlan:
    """Construct the registered scalar figure and coverage plan."""

    models, outputs = _validate_manifests(config)
    if config.analysis_id == "submission-lat-region-scalar":
        return _build_submission_lat_scalar_plan(config, models, outputs)
    submission_scalar = config.analysis_id in SUBMISSION_SCALAR_IDS
    submission_phase_scalar = config.analysis_id == "submission-phase-contact-scalar"
    pairing: pd.DataFrame | None = None
    if submission_scalar:
        models = add_output_groups(models)
    if submission_phase_scalar:
        pairing = _read_csv(config.contact_pairing_manifest, "Contact pairing manifest")
    model_filter = config["inputs"].get("model_filter")
    if model_filter:
        selected = pd.Series(True, index=models.index)
        for column, value in model_filter.items():
            selected &= models[column].eq(value)
        models = models.loc[selected].copy()
        if len(models) != 1:
            raise ValueError(
                "inputs.model_filter must match exactly one model; "
                f"found {len(models)}."
            )
    model_filters = config["inputs"].get("model_filters")
    if model_filters:
        selected_indices: list[int] = []
        for model_filter in model_filters:
            selected = pd.Series(True, index=models.index)
            for column, value in model_filter.items():
                selected &= models[column].eq(value)
            matches = models.index[selected].tolist()
            if len(matches) != 1:
                identity = ", ".join(
                    f"{key}={value}" for key, value in model_filter.items()
                )
                raise ValueError(
                    "Each inputs.model_filters item must match exactly one model; "
                    f"{identity} matched {len(matches)}."
                )
            selected_indices.append(matches[0])
        if len(selected_indices) != len(set(selected_indices)):
            raise ValueError("inputs.model_filters selected a model more than once.")
        models = models.loc[selected_indices].copy()
    _validate_model_labels(models, config)
    source_cache: dict[Path, pd.DataFrame] = {}
    model_data: dict[str, pd.DataFrame] = {}
    result_tables: dict[tuple[str, str, str], pd.DataFrame] = {}
    result_paths: dict[tuple[str, str, str], Path] = {}
    figure_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []

    for _, model in models.iterrows():
        source_path = _path_below(
            Path(str(model["SourcePath"])), config.source_root, "Source PKL"
        )
        if not source_path.exists() or source_path.stat().st_size == 0:
            raise FileNotFoundError(f"Source PKL is missing or empty: {source_path}")
        if source_path not in source_cache:
            source_cache[source_path] = read_stage_table(source_path)
        raw = _model_rows(
            source_cache[source_path],
            model,
            config,
            source_path,
            pairing,
        )
        model_id = str(model["ModelID"])
        model_data[model_id] = raw

        for figure in config["figures"]:
            inference_type = (
                "tukey"
                if config["inputs"]["test_kind"] == "emm_pairwise"
                else "null-test"
            )
            test_id = str(figure["test_id"])
            result_key = (model_id, test_id)
            if (*result_key, "emm") not in result_tables:
                emm_path, emm = _select_result(outputs, model, figure, "emm", config)
                inference_path, inference = _select_result(
                    outputs, model, figure, inference_type, config
                )
                if emm_path.parent != inference_path.parent:
                    raise ValueError(
                        f"EMM and inference results have different parents: {model_id}"
                    )
                result_tables[(*result_key, "emm")] = emm
                result_tables[(*result_key, inference_type)] = inference
                result_paths[(*result_key, "emm")] = emm_path
                result_paths[(*result_key, inference_type)] = inference_path
            else:
                emm = result_tables[(*result_key, "emm")]
                inference = result_tables[(*result_key, inference_type)]
                emm_path = result_paths[(*result_key, "emm")]
                inference_path = result_paths[(*result_key, inference_type)]

            lat_levels: list[str | None] = (
                list(figure["split_levels"])
                if figure.get("layout") == "laterality_singletons"
                else [None]
            )
            for lat in lat_levels:
                displayed_raw = raw if lat is None else raw.loc[raw["Lat"].eq(lat)]
                displayed_inference = (
                    inference
                    if lat is None or "Lat" not in inference.columns
                    else inference.loc[inference["Lat"].eq(lat)]
                )
                p_values = pd.to_numeric(
                    displayed_inference[config["significance"]["p_column"]],
                    errors="coerce",
                )
                significant = int((p_values < 0.05).sum())
                if submission_scalar:
                    lat_display = str(lat) if lat is not None else "Ipsi-Contra"
                    filename_values = {
                        "Band": _filename_slug(model["Band"]),
                        "RegionValue": _filename_slug(model["RegionValue"]),
                        "Polar": _filename_slug(model.get("Polar", "")),
                        "Contrast": _filename_slug(model.get("Contrast", "")),
                        "LatDisplay": _filename_slug(lat_display),
                    }
                    filename = config["outputs"]["filename_template"].format(
                        **filename_values
                    )
                    output_path = (
                        config.output_root / str(model["OutputGroup"]) / filename
                    )
                else:
                    lat_display = str(model.get("Lat", ""))
                    relative_parent = emm_path.parent.relative_to(
                        config.stats_root.resolve()
                    )
                    output_path = (
                        config.output_root
                        / relative_parent
                        / _result_filename(figure, config)
                    )
                if "ContactUnitID" in displayed_raw.columns:
                    n_contact_units = len(
                        displayed_raw[["ID", "ContactUnitID", "Lat"]].drop_duplicates()
                    )
                else:
                    n_contact_units = 0
                figure_rows.append(
                    {
                        "AnalysisID": config.analysis_id,
                        "StatsAnalysisID": config.stats_analysis_id,
                        "ModelID": model_id,
                        "FigureID": str(figure["id"]),
                        "View": (
                            "split-lat"
                            if figure.get("layout") == "laterality_singletons"
                            else (
                                "faceted-lat"
                                if figure.get("layout") == "laterality_rows"
                                else str(figure.get("layout", ""))
                            )
                        ),
                        "TestKind": config["inputs"]["test_kind"],
                        "TestID": test_id,
                        "Engine": model["Engine"],
                        "ModelStatus": model["Status"],
                        "Singular": bool(model["Singular"]),
                        "Domain": model["Domain"],
                        "Metric": model["Metric"],
                        "FeatureOutput": model["FeatureOutput"],
                        "OutputGroup": model.get("OutputGroup", ""),
                        "Representation": model["Representation"],
                        "Band": model["Band"],
                        "RegionVariable": model["RegionVariable"],
                        "RegionValue": model["RegionValue"],
                        "Polar": model.get("Polar", ""),
                        "Lat": lat_display,
                        "Contrast": model.get("Contrast", ""),
                        "SourcePath": str(source_path),
                        "EMMPath": str(emm_path),
                        "TukeyPath": (
                            str(inference_path) if inference_type == "tukey" else ""
                        ),
                        "NullTestPath": (
                            str(inference_path) if inference_type == "null-test" else ""
                        ),
                        "OutputPath": str(output_path),
                        "n_rows": len(displayed_raw),
                        "n_ID": int(displayed_raw["ID"].nunique()),
                        "n_contact_units": n_contact_units,
                        "n_significant": significant,
                        "Status": "planned",
                        "Message": "",
                    }
                )
                coverage_rows.extend(
                    _coverage_rows(
                        raw,
                        model,
                        figure,
                        config,
                        display_lat=lat,
                    )
                )

    figures = pd.DataFrame(figure_rows)
    coverage = pd.DataFrame(coverage_rows)
    if figures["OutputPath"].duplicated().any():
        raise ValueError("Figure plan contains duplicate output paths.")
    submission_tables = (
        build_submission_statistics(config, models, outputs)
        if submission_scalar
        else {}
    )
    return VizPlan(
        figures=figures,
        coverage=coverage,
        model_data=model_data,
        result_tables=result_tables,
        submission_tables=submission_tables,
    )


def build_viz_plan(config: VizConfig) -> VizPlan:
    """Validate sources and construct the configured visualization plan."""

    if config.kind == "scalar":
        return _build_scalar_plan(config)
    if config.kind == "signed_logp_heatmap":
        models, outputs = _validate_manifests(config)
        return build_heatmap_plan(config, models, outputs)
    if config.kind == "correlation_heatmap":
        return build_correlation_heatmap_plan(config)
    if config.kind == "correlation_fit":
        return build_correlation_fit_plan(config)
    if config.kind == "programming_intensity":
        return build_programming_plan(config)
    if config.kind in {"lat_trajectory", "polar_trajectory", "region_trajectory"}:
        return build_lat_trajectory_plan(config)
    return build_nested_plan(config)


def _potential_paths(plan: VizPlan, config: VizConfig) -> list[Path]:
    paths = [Path(path) for path in plan.figures["OutputPath"]]
    detail_manifest = DETAIL_MANIFEST_BY_KIND.get(config.kind, "coverage")
    manifest_keys = ("figures", detail_manifest)
    paths.extend(
        config.manifest_root / config["manifests"][key] for key in manifest_keys
    )
    paths.extend(plan.submission_tables)
    if config.kind == "programming_intensity":
        paths.extend(programming_text_outputs(config))
    return paths


def _trash_existing(paths: list[Path]) -> None:
    trash = Path("/usr/bin/trash")
    if not trash.exists():
        raise FileNotFoundError(f"Trash command is unavailable: {trash}")
    for offset in range(0, len(paths), 200):
        subprocess.run(
            [str(trash), *(str(path) for path in paths[offset : offset + 200])],
            check=True,
        )


def _summary(plan: VizPlan, config: VizConfig) -> dict[str, Any]:
    if config.kind == "programming_intensity":
        return {
            "planned_figures": len(plan.figures),
            "figure_types": plan.figures.PlotKind.value_counts().to_dict(),
            "heatmap_cells": len(plan.coverage),
            "fit_points": len(
                plan.submission_tables[config.output_root / "data/fit_points.csv"]
            ),
            "jitter_rows": len(
                plan.submission_tables[config.output_root / "data/jitter_selected.csv"]
            ),
            "writes": 0,
        }
    if config.kind == "signed_logp_heatmap":
        return {
            "planned_models": int(plan.coverage["ModelID"].nunique()),
            "planned_figures": len(plan.figures),
            "heatmap_cells": len(plan.coverage),
            "singular_models": int(
                plan.coverage.loc[plan.coverage["Singular"], "ModelID"].nunique()
            ),
            "significant_cells": int((plan.coverage["PValue"] < 0.05).sum()),
            "writes": 0,
        }
    if config.kind == "correlation_heatmap":
        p_column = config["inputs"]["p_column"]
        unique_correlations = plan.coverage.drop_duplicates("CorrelationID")
        return {
            "planned_correlations": len(unique_correlations),
            "planned_figures": len(plan.figures),
            "heatmap_cells": len(plan.coverage),
            "estimable_cells": int(unique_correlations["rho"].notna().sum()),
            "significant_cells": int(
                pd.to_numeric(unique_correlations[p_column], errors="coerce")
                .lt(0.05)
                .sum()
            ),
            "writes": 0,
        }
    if config.kind == "correlation_fit":
        return {
            "selected_correlations": len(plan.figures),
            "planned_figures": len(plan.figures),
            "fit_points": len(plan.coverage),
            "writes": 0,
        }
    if config.kind in {"lat_trajectory", "polar_trajectory", "region_trajectory"}:
        return {
            "planned_models": int(plan.figures["ModelID"].nunique()),
            "planned_figures": len(plan.figures),
            "coverage_cells": len(plan.coverage),
            "component_emmeans": len(
                plan.submission_tables[
                    config.output_root / config["manifests"]["emmeans"]
                ]
            ),
            "significant_cells": int(plan.figures["n_significant"].sum()),
            "singular_models": int(
                plan.figures.loc[
                    plan.figures["Singular"].astype(bool), "ModelID"
                ].nunique()
            ),
            "writes": 0,
        }
    if config.analysis_id == "submission-lat-region-scalar":
        planned_model_ids = {
            model_id
            for values in plan.figures["SourceModelIDs"].astype(str)
            for model_id in values.split(";")
            if model_id
        }
        singular_model_ids = {
            model_id
            for values in plan.figures["SingularModelIDs"].astype(str)
            for model_id in values.split(";")
            if model_id
        }
    else:
        planned_model_ids = set(plan.figures["ModelID"].astype(str))
        singular_model_ids = set(
            plan.figures.loc[plan.figures["Singular"], "ModelID"].astype(str)
        )
    summary = {
        "planned_models": len(planned_model_ids),
        "planned_figures": len(plan.figures),
        "coverage_cells": len(plan.coverage),
        "singular_models": len(singular_model_ids),
        "writes": 0,
    }
    if plan.submission_tables:
        statistics_path = config.manifest_root / config["manifests"]["statistics"]
        summary["statistics_tables"] = len(plan.submission_tables[statistics_path])
    significance_key = (
        "significant_cells"
        if set(plan.figures["TestKind"]) == {"emmean_vs_null"}
        else "significant_brackets"
    )
    summary[significance_key] = int(plan.figures["n_significant"].sum())
    return summary


def _render(plan: VizPlan, config: VizConfig) -> dict[str, Any]:
    figures = plan.figures.copy()
    trace_data: dict[str, pd.DataFrame] = {}
    figure_configs = {item["id"]: item for item in config["figures"]}
    for index, row in figures.iterrows():
        test_id = str(row["TestID"])
        output_path = Path(str(row["OutputPath"]))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if config.kind == "programming_intensity":
            fig = render_programming_figure(plan, row, config)
        elif config.kind == "signed_logp_heatmap":
            figure_id = str(row["FigureID"])
            fig = render_heatmap_figure(
                plan.coverage.loc[plan.coverage["FigureID"].eq(figure_id)].copy(),
                row,
                config,
            )
        elif config.kind == "correlation_heatmap":
            figure_id = str(row["FigureID"])
            fig = render_correlation_heatmap_figure(
                plan.coverage.loc[plan.coverage["FigureID"].eq(figure_id)].copy(),
                row,
                config,
            )
        elif config.kind == "correlation_fit":
            figure_id = str(row["FigureID"])
            fig = render_correlation_fit_figure(
                plan.coverage.loc[plan.coverage["FigureID"].eq(figure_id)].copy(),
                row,
                config,
            )
        elif config.kind in {
            "lat_trajectory",
            "polar_trajectory",
            "region_trajectory",
        }:
            model_id = str(row["DataKey"])
            fig = render_lat_trajectory_figure(
                plan.result_tables[(model_id, test_id, "emm")],
                plan.result_tables[(model_id, test_id, "statistics")],
                row,
                config,
                coverage=plan.coverage.loc[plan.coverage["DataKey"].eq(model_id)],
            )
        elif config.kind == "scalar":
            model_id = str(row.get("DataKey", row["ModelID"]))
            figure_id = str(row.get("FigureID", test_id))
            inference_type = (
                "tukey"
                if config["inputs"]["test_kind"] == "emm_pairwise"
                else "null-test"
            )
            fig = render_scalar_figure(
                plan.model_data[model_id],
                plan.result_tables[(model_id, test_id, "emm")],
                plan.result_tables[(model_id, test_id, inference_type)],
                row,
                figure_configs.get(figure_id, figure_configs.get(test_id)),
                config,
            )
        elif config.kind in {"spectral", "trace"}:
            model_id = str(row["ModelID"])
            source = plan.model_data[model_id]
            if config.kind == "trace":
                if model_id not in trace_data:
                    trace_data[model_id] = smooth_trace_table(source, config)
                source = trace_data[model_id]
            fig = render_series_figure(
                source,
                row,
                figure_configs.get(
                    test_id, figure_configs.get(str(row.get("FigureID", "")))
                ),
                config,
                plan.result_tables.get((model_id, test_id, "intervals")),
            )
        else:
            model_id = str(row["ModelID"])
            fig = render_raw_figure(
                plan.model_data[model_id],
                row,
                figure_configs.get(
                    test_id, figure_configs.get(str(row.get("FigureID", "")))
                ),
                config,
            )
        try:
            fig.savefig(
                output_path,
                format="pdf",
                dpi=config["style"]["dpi"],
                bbox_inches="tight",
            )
        finally:
            plt.close(fig)
        figures.at[index, "Status"] = "ok"

    config.manifest_root.mkdir(parents=True, exist_ok=True)
    atomic_csv(
        figures,
        config.manifest_root / config["manifests"]["figures"],
        overwrite=False,
    )
    detail_manifest = DETAIL_MANIFEST_BY_KIND.get(config.kind, "coverage")
    atomic_csv(
        plan.coverage,
        config.manifest_root / config["manifests"][detail_manifest],
        overwrite=False,
    )
    for path, table in plan.submission_tables.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        output = table.copy()
        if config.kind == "scalar" and path == config.manifest_root / config[
            "manifests"
        ].get("statistics", ""):
            output["Status"] = "ok"
        atomic_csv(output, path, overwrite=False)
    text_outputs = (
        programming_text_outputs(config)
        if config.kind == "programming_intensity"
        else {}
    )
    for path, content in text_outputs.items():
        atomic_text(content, path, overwrite=False)
    if config.kind in HEATMAP_KINDS:
        detail_key = "written_heatmap_cells"
    elif config.kind == "correlation_fit":
        detail_key = "written_fit_points"
    else:
        detail_key = "written_coverage_rows"
    return {
        "written_figures": len(figures),
        detail_key: len(plan.coverage),
        "writes": len(figures) + 2 + len(plan.submission_tables) + len(text_outputs),
    }


def run_viz(
    config: VizConfig, *, dry_run: bool = False, overwrite: bool = False
) -> dict[str, Any]:
    """Validate the plan and optionally render every configured PDF."""

    plan = build_viz_plan(config)
    summary = _summary(plan, config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if dry_run:
        return summary

    existing = [path for path in _potential_paths(plan, config) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Planned visualization output exists; pass --overwrite to replace it: "
            f"{existing[0]}"
        )
    if existing:
        _trash_existing(existing)
    execution = _render(plan, config)
    print(json.dumps(execution, indent=2, ensure_ascii=False))
    return {**summary, **execution}
