"""Representation-independent visualization plans."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lfp_cohort.io import read_stage_table

from .config import VizConfig
from .data import (
    nested_has_finite,
    path_below,
    read_csv,
    require_columns,
)


@dataclass(frozen=True)
class VizPlan:
    """Validated figure metadata, coverage, and in-memory plotting inputs."""

    figures: pd.DataFrame
    coverage: pd.DataFrame
    model_data: dict[str, pd.DataFrame]
    result_tables: dict[tuple[str, str, str], pd.DataFrame]
    submission_tables: dict[Path, pd.DataFrame] = field(default_factory=dict)


FEATURE_OUTPUT_COLUMNS = {
    "Stage",
    "SourceStage",
    "Domain",
    "Metric",
    "FeatureOutput",
    "Representation",
    "Path",
    "Rows",
    "AggregationLevel",
}
INTERVAL_COLUMNS = {
    "ModelID",
    "TestID",
    "LineValue",
    "Start",
    "End",
    "Direction",
    "p_value",
}


def _slug(value: Any) -> str:
    return str(value).replace("_", "-").replace(" ", "-")


def _filename_slug(value: Any) -> str:
    return _slug(value).replace("→", "-to-")


def _uses_submission_metric_layout(config: VizConfig) -> bool:
    return config["outputs"].get("directory_fields") == ["Metric"]


def _layout_view(figure: dict[str, Any]) -> str:
    if figure["layout"] == "laterality_singletons":
        return "split-lat"
    if figure["layout"] == "laterality_rows":
        return "faceted-lat"
    return str(figure["layout"]).replace("_", "-")


def _object_id(
    *,
    domain: str,
    metric: str,
    feature_output: str,
    band: str,
    region_variable: str,
    region_value: str,
    polar: str,
) -> str:
    region_key = "region" if region_variable == "Region" else "pair-region"
    return "_".join(
        (
            f"domain-{_slug(domain)}",
            f"metric-{_slug(metric)}",
            f"feature-output-{_slug(feature_output)}",
            f"band-{_slug(band)}",
            f"{region_key}-{_slug(region_value)}",
            f"polar-{_slug(polar)}",
        )
    )


def _match_feature_output(
    manifest: pd.DataFrame,
    *,
    config: VizConfig,
    domain: str,
    metric: str,
    feature_output: str,
) -> pd.Series:
    inputs = config["inputs"]
    selected = manifest.loc[
        manifest["Stage"].eq(inputs["stage"])
        & manifest["SourceStage"].eq(inputs["source_stage"])
        & manifest["AggregationLevel"].eq(inputs["aggregation_level"])
        & manifest["Representation"].eq(inputs["representation"])
        & manifest["Domain"].eq(domain)
        & manifest["Metric"].eq(metric)
        & manifest["FeatureOutput"].eq(feature_output)
    ]
    if len(selected) != 1:
        raise ValueError(
            "Expected one configured feature-output manifest match for "
            f"{domain}/{metric}/{feature_output}; found {len(selected)}."
        )
    return selected.iloc[0]


def _nested_rows(
    table: pd.DataFrame,
    *,
    config: VizConfig,
    source_path: Path,
    domain: str,
    metric: str,
    feature_output: str,
    band: str,
    region_variable: str,
    region_value: str,
    polar: str,
) -> pd.DataFrame:
    required = {
        "ID",
        "Domain",
        "Metric",
        "FeatureOutput",
        "Representation",
        "Polar",
        "Phase",
        "Lat",
        "Value",
        "IncludeAggregate",
        region_variable,
    }
    if config.kind == "trace":
        required.add("Band")
    require_columns(table, required, f"Visualization source {source_path}")
    include_values = set(table["IncludeAggregate"].dropna().unique())
    if not include_values.issubset({True, False, np.bool_(True), np.bool_(False)}):
        raise TypeError(f"IncludeAggregate must be boolean: {source_path}")
    mask = (
        table["IncludeAggregate"].fillna(False).astype(bool)
        & table["Domain"].eq(domain)
        & table["Metric"].eq(metric)
        & table["FeatureOutput"].eq(feature_output)
        & table["Representation"].eq(config.kind)
        & table[region_variable].eq(region_value)
        & table["Polar"].eq(polar)
    )
    if config.kind == "trace":
        mask &= table["Band"].eq(band)
    selected = table.loc[mask].copy()
    selected = selected.loc[selected["Value"].map(nested_has_finite)].copy()
    if selected.empty:
        raise ValueError(
            f"No finite rows remain for {domain}/{metric}/{feature_output}/"
            f"{band}/{region_value}/{polar}."
        )
    expected_type = pd.DataFrame if config.kind == "raw" else pd.Series
    if not selected["Value"].map(lambda value: isinstance(value, expected_type)).all():
        raise TypeError(f"{config.kind} Value has the wrong nested type: {source_path}")
    template = selected.iloc[0]["Value"]
    for value in selected["Value"].iloc[1:]:
        if not value.index.equals(template.index):
            raise ValueError(f"Nested row axes differ: {source_path}")
        if isinstance(template, pd.DataFrame) and not value.columns.equals(
            template.columns
        ):
            raise ValueError(f"Nested column axes differ: {source_path}")
    if config.kind == "spectral":
        unknown = sorted(
            set(selected["Phase"].astype(str)) - set(config["factor_levels"]["Phase"])
        )
        if unknown:
            raise ValueError(f"Spectral source has unknown Phase levels: {unknown}")
        duplicate_keys = ["ID", "Phase", "Lat"]
    else:
        if set(selected["Phase"].astype(str)) != {"All"}:
            raise ValueError(f"{config.kind} source Phase must be All: {source_path}")
        duplicate_keys = ["ID", "Lat"]
    if selected.duplicated(duplicate_keys, keep=False).any():
        raise ValueError(
            f"Visualization source contains duplicate {'/'.join(duplicate_keys)} rows: "
            f"{source_path}"
        )
    selected["Lat"] = pd.Categorical(
        selected["Lat"], categories=config["factor_levels"]["Lat"], ordered=True
    )
    if config.kind == "spectral":
        selected["Phase"] = pd.Categorical(
            selected["Phase"],
            categories=config["factor_levels"]["Phase"],
            ordered=True,
        )
    return selected.reset_index(drop=True)


def _coverage(
    source: pd.DataFrame,
    *,
    config: VizConfig,
    object_id: str,
    figure: dict[str, Any],
    lat: str | None = None,
) -> list[dict[str, Any]]:
    if figure["layout"] == "laterality_singletons":
        if lat is None:
            raise ValueError("Laterality-singleton coverage requires Lat.")
        groups = (
            [
                (
                    phase,
                    lat,
                    source.loc[source["Phase"].eq(phase) & source["Lat"].eq(lat)],
                )
                for phase in config["factor_levels"]["Phase"]
            ]
            if config.kind == "spectral"
            else [("All", lat, source.loc[source["Lat"].eq(lat)])]
        )
    elif figure["layout"] == "marginal":
        groups = (
            [
                (phase, "", source.loc[source["Phase"].eq(phase)])
                for phase in config["factor_levels"]["Phase"]
            ]
            if config.kind == "spectral"
            else [("All", "", source)]
        )
    elif config.kind == "spectral":
        groups = [
            (phase, lat, source.loc[source["Phase"].eq(phase) & source["Lat"].eq(lat)])
            for lat in config["factor_levels"]["Lat"]
            for phase in config["factor_levels"]["Phase"]
        ]
    else:
        groups = [
            ("All", lat, source.loc[source["Lat"].eq(lat)])
            for lat in config["factor_levels"]["Lat"]
        ]
    rows = []
    for phase, lat, subset in groups:
        row = {
            "AnalysisID": config.analysis_id,
            "StatsAnalysisID": "",
            "ModelID": object_id,
            "TestID": figure["id"],
            "Phase": phase,
            "RowVariable": "Lat" if lat else "",
            "RowValue": lat,
            "ColumnVariable": "",
            "ColumnValue": "",
            "n_ID": int(subset["ID"].nunique()),
            "n_obs": int(len(subset)),
        }
        if _uses_submission_metric_layout(config):
            row["View"] = _layout_view(figure)
        rows.append(row)
    return rows


def _load_trace_intervals(config: VizConfig) -> pd.DataFrame | None:
    if config.kind != "trace" or not config["inference"]["enabled"]:
        return None
    intervals = read_csv(
        Path(config["inference"]["interval_manifest"]), "Trace interval manifest"
    )
    if set(intervals.columns) != INTERVAL_COLUMNS:
        raise ValueError(
            f"Trace interval manifest columns must equal {sorted(INTERVAL_COLUMNS)}."
        )
    for column in ("Start", "End", "p_value"):
        numeric = pd.to_numeric(intervals[column], errors="coerce")
        if numeric.isna().any() or not np.isfinite(numeric).all():
            raise TypeError(f"Trace interval {column} must be finite numeric data.")
        intervals[column] = numeric.astype(float)
    if (
        (intervals["Start"] < 0)
        | (intervals["End"] > 240)
        | (intervals["Start"] >= intervals["End"])
    ).any():
        raise ValueError("Trace intervals must satisfy 0 <= Start < End <= 240.")
    if not set(intervals["TestID"]).issubset({"Phase", "Phase-Lat"}):
        raise ValueError("Trace interval TestID must be Phase or Phase-Lat.")
    if not set(intervals["LineValue"]).issubset({"Overall", "Ipsi", "Contra"}):
        raise ValueError("Trace interval LineValue is unsupported.")
    invalid_line = (
        intervals["TestID"].eq("Phase") & ~intervals["LineValue"].eq("Overall")
    ) | (
        intervals["TestID"].eq("Phase-Lat")
        & ~intervals["LineValue"].isin(["Ipsi", "Contra"])
    )
    if invalid_line.any():
        raise ValueError("Trace interval LineValue does not match its TestID.")
    if not set(intervals["Direction"]).issubset({"above_0", "below_0"}):
        raise ValueError("Trace interval Direction is unsupported.")
    if ((intervals["p_value"] < 0) | (intervals["p_value"] > 1)).any():
        raise ValueError("Trace interval p_value must be between zero and one.")
    return intervals


def build_nested_plan(config: VizConfig) -> VizPlan:
    """Build the descriptive spectral, trace, or raw figure plan."""

    manifest = read_csv(config.feature_outputs_manifest, "Feature-output manifest")
    require_columns(manifest, FEATURE_OUTPUT_COLUMNS, "Feature-output manifest")
    model_data: dict[str, pd.DataFrame] = {}
    result_tables: dict[tuple[str, str, str], pd.DataFrame] = {}
    figure_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    intervals = _load_trace_intervals(config)
    used_intervals: set[int] = set()
    for domain in ("local", "connectivity"):
        selector = config["inputs"]["selectors"][domain]
        region_variable = "Region" if domain == "local" else "PairRegion"
        for feature in selector["features"]:
            metric = str(feature["metric"])
            feature_output = str(feature["feature_output"])
            feature_regions = feature.get(
                "canonical_region", selector["canonical_region"]
            )
            manifest_row = _match_feature_output(
                manifest,
                config=config,
                domain=domain,
                metric=metric,
                feature_output=feature_output,
            )
            source_path = path_below(
                Path(str(manifest_row["Path"])), config.table_root, "Source PKL"
            )
            if not source_path.exists() or source_path.stat().st_size == 0:
                raise FileNotFoundError(
                    f"Source PKL is missing or empty: {source_path}"
                )
            source_table = read_stage_table(source_path)
            if len(source_table) != int(manifest_row["Rows"]):
                raise ValueError(
                    f"Source row count differs from its manifest: {source_path}"
                )
            for band in feature["bands"]:
                for region_value in feature_regions:
                    for polar in selector["polar"]:
                        object_id = _object_id(
                            domain=domain,
                            metric=metric,
                            feature_output=feature_output,
                            band=band,
                            region_variable=region_variable,
                            region_value=region_value,
                            polar=polar,
                        )
                        source = _nested_rows(
                            source_table,
                            config=config,
                            source_path=source_path,
                            domain=domain,
                            metric=metric,
                            feature_output=feature_output,
                            band=band,
                            region_variable=region_variable,
                            region_value=region_value,
                            polar=polar,
                        )
                        model_data[object_id] = source
                        relative_parent = source_path.parent.relative_to(
                            config.table_root.resolve()
                        )
                        submission_layout = _uses_submission_metric_layout(config)
                        if submission_layout:
                            output_parent = config.output_root / _filename_slug(metric)
                        else:
                            output_parent = (
                                config.output_root
                                / relative_parent
                                / config.kind
                                / feature_output
                                / band
                                / region_value
                                / polar
                            )
                        for figure in config["figures"]:
                            lat_levels: list[str | None] = (
                                list(figure["split_levels"])
                                if figure["layout"] == "laterality_singletons"
                                else [None]
                            )
                            for lat in lat_levels:
                                displayed_source = (
                                    source.loc[source["Lat"].eq(lat)]
                                    if lat is not None
                                    else source
                                )
                                if displayed_source.empty:
                                    raise ValueError(
                                        f"No rows remain for {object_id}/{lat}."
                                    )
                                significant = 0
                                if intervals is not None:
                                    selected_intervals = intervals.loc[
                                        intervals["ModelID"].eq(object_id)
                                        & intervals["TestID"].eq(figure["id"])
                                    ].copy()
                                    used_intervals.update(selected_intervals.index)
                                    result_tables[
                                        (object_id, str(figure["id"]), "intervals")
                                    ] = selected_intervals.reset_index(drop=True)
                                    significant = int(
                                        (
                                            selected_intervals["p_value"]
                                            < config["inference"]["alpha"]
                                        ).sum()
                                    )
                                lat_display = (
                                    str(lat)
                                    if lat is not None
                                    else "Ipsi-Contra" if submission_layout else ""
                                )
                                output_name = config["outputs"][
                                    "filename_template"
                                ].format(
                                    AnalysisID=config.analysis_id,
                                    TestID=figure["id"],
                                    Lat=lat or "",
                                    LatDisplay=_filename_slug(lat_display),
                                    Metric=_filename_slug(metric),
                                    Band=_filename_slug(band),
                                    RegionValue=_filename_slug(region_value),
                                    Polar=_filename_slug(polar),
                                )
                                output_path_parent = (
                                    output_parent
                                    if submission_layout or lat is None
                                    else output_parent / str(lat)
                                )
                                figure_row = {
                                    "AnalysisID": config.analysis_id,
                                    "StatsAnalysisID": "",
                                    "ModelID": object_id,
                                    "TestKind": "descriptive",
                                    "TestID": figure["id"],
                                    "Engine": "",
                                    "ModelStatus": "not_applicable",
                                    "Singular": False,
                                    "Domain": domain,
                                    "Metric": metric,
                                    "FeatureOutput": feature_output,
                                    "Representation": config.kind,
                                    "Band": band,
                                    "RegionVariable": region_variable,
                                    "RegionValue": region_value,
                                    "Polar": polar,
                                    "SourcePath": str(source_path),
                                    "EMMPath": "",
                                    "TukeyPath": "",
                                    "OutputPath": str(output_path_parent / output_name),
                                    "n_rows": len(displayed_source),
                                    "n_ID": int(displayed_source["ID"].nunique()),
                                    "n_significant": significant,
                                    "Status": "planned",
                                    "Message": "",
                                }
                                if submission_layout:
                                    figure_row["View"] = _layout_view(figure)
                                    figure_row["Lat"] = lat_display
                                elif lat is not None:
                                    figure_row["Lat"] = lat
                                figure_rows.append(figure_row)
                                coverage_rows.extend(
                                    _coverage(
                                        source,
                                        config=config,
                                        object_id=object_id,
                                        figure=figure,
                                        lat=lat,
                                    )
                                )
    figures = pd.DataFrame(figure_rows)
    coverage = pd.DataFrame(coverage_rows)
    if figures.empty:
        raise ValueError("Nested visualization plan is empty.")
    if intervals is not None and used_intervals != set(intervals.index):
        raise ValueError("Trace interval manifest contains unmatched rows.")
    if figures["OutputPath"].duplicated().any():
        raise ValueError("Figure plan contains duplicate output paths.")
    return VizPlan(
        figures=figures,
        coverage=coverage,
        model_data=model_data,
        result_tables=result_tables,
    )
