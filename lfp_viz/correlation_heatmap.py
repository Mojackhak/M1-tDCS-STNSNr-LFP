"""Plan and render patient-level clinical-correlation heatmaps."""

from __future__ import annotations

import copy
from typing import Any, Mapping

import pandas as pd

from .config import (
    SUBMISSION_CLINICAL_CORRELATION_HEATMAP_ID,
    SUBMISSION_CLINICAL_CORRELATION_SCALE_HEATMAP_ID,
    VizConfig,
)
from .data import path_below, read_csv, require_columns
from .heatmap import render_well_heatmap_figure
from .plan import VizPlan


CORRELATION_COLUMNS = {
    "AnalysisID",
    "CorrelationID",
    "EndpointID",
    "EndpointLabel",
    "EndpointOrder",
    "Scale",
    "ScaleLabel",
    "ScaleOrder",
    "PredictorID",
    "Domain",
    "Metric",
    "FeatureOutput",
    "Band",
    "RegionVariable",
    "RegionValue",
    "DisplayRegion",
    "FeatureOrder",
    "FeatureLabel",
    "Polar",
    "Phase",
    "Lat",
    "n",
    "rho",
    "p_raw",
    "p_holm",
    "SignificanceBasis",
    "Status",
    "Message",
}
ADJACENT_PHASE_COLUMNS = {
    "FromPhase",
    "ToPhase",
    "PhaseContrast",
    "PredictorDefinition",
}
ADJACENT_PHASE_MAPPING = {
    ("Early", "Pre", "Early", "Early−Pre", "adjacent_difference"),
    ("Late", "Early", "Late", "Late−Early", "adjacent_difference"),
    ("Post", "Late", "Post", "Post−Late", "adjacent_difference"),
}


def _slug(value: Any) -> str:
    return str(value).replace("_", "-").replace(" ", "-")


def _filename_slug(value: Any) -> str:
    return _slug(value).replace("→", "-to-").replace("/", "-")


def _figure_id(
    config: VizConfig,
    *,
    endpoint: str,
    domain: str,
    region: str,
    polar: str,
    phase: str,
    lat_display: str,
) -> str:
    return "_".join(
        (
            f"analysis-{config.analysis_id}",
            f"endpoint-{_slug(endpoint)}",
            f"domain-{_slug(domain)}",
            f"region-{_slug(region)}",
            f"polar-{_slug(polar)}",
            f"phase-{_slug(phase)}",
            f"lat-{_slug(lat_display)}",
        )
    )


def load_correlation_results(config: VizConfig) -> pd.DataFrame:
    path = path_below(config.correlations_path, config.stats_root, "Correlation result")
    table = read_csv(path, "Correlation result")
    require_columns(table, CORRELATION_COLUMNS, f"Correlation result {path}")
    if not table["AnalysisID"].eq(config.stats_analysis_id).all():
        raise ValueError(
            "Correlation result AnalysisID does not match stats_analysis_id."
        )
    if table["CorrelationID"].duplicated().any():
        raise ValueError("Correlation result contains duplicate CorrelationID values.")
    statuses = set(table["Status"].astype(str))
    unsupported = sorted(statuses - set(config["inputs"]["accepted_statuses"]))
    if unsupported:
        raise ValueError(
            f"Correlation result contains unsupported statuses: {unsupported}"
        )
    if not table["SignificanceBasis"].eq("p_holm").all():
        raise ValueError("Correlation result significance basis is not registered.")
    if config.stats_analysis_id == "clinical-correlation-adjacent-phase":
        require_columns(
            table,
            ADJACENT_PHASE_COLUMNS,
            f"Adjacent-phase correlation result {path}",
        )
        observed_mapping = {
            tuple(row)
            for row in table[
                [
                    "Phase",
                    "FromPhase",
                    "ToPhase",
                    "PhaseContrast",
                    "PredictorDefinition",
                ]
            ]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        }
        if observed_mapping != ADJACENT_PHASE_MAPPING:
            raise ValueError(
                "Adjacent-phase correlation result does not match the registered "
                "Early-Pre, Late-Early, and Post-Late mapping."
            )
    return table


def _scale_figure_id(
    config: VizConfig,
    *,
    scale: str,
    domain: str,
    region: str,
    polar: str,
    lat: str,
) -> str:
    return "_".join(
        (
            f"analysis-{config.analysis_id}",
            f"scale-{_filename_slug(scale)}",
            f"domain-{_filename_slug(domain)}",
            f"region-{_filename_slug(region)}",
            f"polar-{_filename_slug(polar)}",
            f"lat-{_filename_slug(lat)}",
        )
    )


def _scale_composite_figure_id(
    config: VizConfig,
    *,
    scale: str,
    domain: str,
    region: str,
) -> str:
    return "_".join(
        (
            f"analysis-{config.analysis_id}",
            f"scale-{_filename_slug(scale)}",
            f"domain-{_filename_slug(domain)}",
            f"region-{_filename_slug(region)}",
            "layout-phase-rows-composite",
        )
    )


def _build_scale_correlation_heatmap_plan(
    config: VizConfig, correlations: pd.DataFrame
) -> VizPlan:
    """Construct two-scale postoperative Phase-row heatmaps."""

    inputs = config["inputs"]
    scales = [str(value) for value in inputs["selected_scales"]]
    endpoints = [str(value) for value in inputs["selected_endpoints"]]
    selected = correlations.loc[
        correlations["ScaleLabel"].isin(scales)
        & correlations["EndpointID"].isin(endpoints)
    ].copy()
    if set(selected["ScaleLabel"].astype(str)) != set(scales):
        raise ValueError("Scale-centered heatmap does not contain both selected scales.")
    if set(selected["EndpointID"].astype(str)) != set(endpoints):
        raise ValueError(
            "Scale-centered heatmap does not contain both selected endpoints."
        )

    phases = [str(value) for value in config["factor_levels"]["Phase"]]
    polars = [str(value) for value in config["factor_levels"]["Polar"]]
    lateralities = [str(value) for value in config["factor_levels"]["Lat"]]
    region_specs = [
        ("local", "SNr", 44),
        ("local", "STN", 44),
        ("connectivity", "SNr-STN-combined", 35),
    ]
    endpoint_order = {value: index for index, value in enumerate(endpoints)}
    phase_order = {value: index for index, value in enumerate(phases)}

    figure_rows: list[dict[str, Any]] = []
    cell_frames: list[pd.DataFrame] = []
    for scale_label in scales:
        for domain, region, expected_features in region_specs:
            for polar in polars:
                for lat in lateralities:
                    group = selected.loc[
                        selected["ScaleLabel"].eq(scale_label)
                        & selected["Domain"].eq(domain)
                        & selected["DisplayRegion"].eq(region)
                        & selected["Polar"].eq(polar)
                        & selected["Lat"].eq(lat)
                    ].copy()
                    expected_cells = (
                        len(phases) * len(endpoints) * expected_features
                    )
                    if len(group) != expected_cells:
                        raise ValueError(
                            "Scale-centered heatmap group has an unexpected cell count: "
                            f"{scale_label}/{domain}/{region}/{polar}/{lat}; "
                            f"found {len(group)}, expected {expected_cells}."
                        )
                    scale_values = group["Scale"].astype(str).unique()
                    region_variables = group["RegionVariable"].astype(str).unique()
                    if len(scale_values) != 1 or len(region_variables) != 1:
                        raise ValueError(
                            "Scale-centered heatmap group has inconsistent identity fields."
                        )

                    figure_id = _scale_figure_id(
                        config,
                        scale=scale_label,
                        domain=domain,
                        region=region,
                        polar=polar,
                        lat=lat,
                    )
                    cells = group.copy()
                    cells["FigureID"] = figure_id
                    cells["FigureKind"] = "atomic"
                    cells["Layout"] = "phase_rows"
                    cells["PanelRow"] = cells["Phase"].map(phase_order)
                    cells["PanelColumn"] = 0
                    region_label = config["labels"]["region"][region]
                    polar_label = config["labels"]["polarity"][polar]
                    domain_label = config["labels"]["domain"][domain]
                    cells["PanelTopLabel"] = (
                        f"{scale_label} | {polar_label} | {region_label} | "
                        f"{lat} | {domain_label}"
                    )
                    cells["PanelRightLabel"] = cells["Phase"]
                    cells["MatrixRow"] = cells["EndpointID"].map(endpoint_order)
                    cells["MatrixColumn"] = cells["FeatureOrder"].astype(int)
                    cells["MatrixRowValue"] = cells["EndpointID"]
                    cells["MatrixColumnValue"] = (
                        cells["Metric"].astype(str)
                        + "|"
                        + cells["FeatureOutput"].astype(str)
                        + "|"
                        + cells["Band"].astype(str)
                    )
                    cells["MatrixRowLabel"] = cells["EndpointID"].map(
                        config["labels"]["endpoint"]
                    )
                    cells["MatrixColumnLabel"] = config["labels"][
                        "delta_prefix"
                    ] + cells["FeatureLabel"].astype(str)
                    cells["DisplayRegionValue"] = region
                    if cells[["PanelRow", "MatrixRow"]].isna().any().any():
                        raise ValueError(
                            f"Scale-centered heatmap {figure_id} has unsupported levels."
                        )
                    duplicate_keys = [
                        "PanelRow",
                        "PanelColumn",
                        "MatrixRow",
                        "MatrixColumn",
                    ]
                    if cells.duplicated(duplicate_keys).any():
                        raise ValueError(
                            f"Scale-centered heatmap {figure_id} contains duplicate cells."
                        )
                    if cells["FeatureOrder"].nunique() != expected_features:
                        raise ValueError(
                            f"Scale-centered heatmap {figure_id} does not contain "
                            f"{expected_features} features."
                        )

                    scale_slug = _filename_slug(scale_label)
                    output_parent = (
                        config.output_root
                        / "heatmap-by-scale"
                        / scale_slug
                        / domain
                    )
                    filename = config["outputs"]["filename_template"].format(
                        Scale=scale_slug,
                        Domain=_filename_slug(domain),
                        RegionValue=_filename_slug(region),
                        Polar=_filename_slug(polar),
                        Lat=_filename_slug(lat),
                    )
                    p_values = pd.to_numeric(
                        cells[config["inputs"]["p_column"]], errors="coerce"
                    )
                    figure_rows.append(
                        {
                            "AnalysisID": config.analysis_id,
                            "StatsAnalysisID": config.stats_analysis_id,
                            "FigureID": figure_id,
                            "FigureKind": "atomic",
                            "TestKind": "correlation",
                            "TestID": "clinical-correlation",
                            "Layout": "phase_rows",
                            "View": "matrix",
                            "Scale": str(scale_values[0]),
                            "ScaleLabel": scale_label,
                            "Domain": domain,
                            "RegionVariable": str(region_variables[0]),
                            "RegionValue": region,
                            "SourceRegionValues": ";".join(
                                sorted(group["RegionValue"].astype(str).unique())
                            ),
                            "RegionLabel": region_label,
                            "Polar": polar,
                            "Lat": lat,
                            "ColorbarLabel": config["labels"]["colorbar"][
                                scale_label
                            ],
                            "OutputPath": str(output_parent / filename),
                            "n_features": expected_features,
                            "n_cells": len(cells),
                            "n_estimable": int(cells["rho"].notna().sum()),
                            "n_significant": int(p_values.lt(0.05).sum()),
                            "Status": "planned",
                            "Message": "",
                        }
                    )
                    cell_frames.append(cells)

    group_order = {
        (polar, lat): index
        for index, (polar, lat) in enumerate(
            (polar, lat) for polar in polars for lat in lateralities
        )
    }
    panel_order = {
        (polar, lat, phase): group_order[(polar, lat)] * len(phases)
        + phase_order[phase]
        for polar in polars
        for lat in lateralities
        for phase in phases
    }
    for scale_label in scales:
        for domain, region, expected_features in region_specs:
            group = selected.loc[
                selected["ScaleLabel"].eq(scale_label)
                & selected["Domain"].eq(domain)
                & selected["DisplayRegion"].eq(region)
            ].copy()
            expected_cells = (
                len(polars)
                * len(lateralities)
                * len(phases)
                * len(endpoints)
                * expected_features
            )
            if len(group) != expected_cells:
                raise ValueError(
                    "Scale-centered composite has an unexpected cell count: "
                    f"{scale_label}/{domain}/{region}; found {len(group)}, "
                    f"expected {expected_cells}."
                )
            scale_values = group["Scale"].astype(str).unique()
            region_variables = group["RegionVariable"].astype(str).unique()
            if len(scale_values) != 1 or len(region_variables) != 1:
                raise ValueError(
                    "Scale-centered composite has inconsistent identity fields."
                )

            figure_id = _scale_composite_figure_id(
                config,
                scale=scale_label,
                domain=domain,
                region=region,
            )
            cells = group.copy()
            cells["FigureID"] = figure_id
            cells["FigureKind"] = "composite"
            cells["Layout"] = "phase_rows_composite"
            cells["PanelRow"] = [
                panel_order[(str(polar), str(lat), str(phase))]
                for polar, lat, phase in zip(
                    cells["Polar"], cells["Lat"], cells["Phase"]
                )
            ]
            cells["PanelColumn"] = 0
            region_label = config["labels"]["region"][region]
            cells["PanelTopLabel"] = f"{scale_label} | {region_label}"
            cells["PanelRightLabel"] = cells["Phase"]
            cells["PanelRightGroup"] = [
                group_order[(str(polar), str(lat))]
                for polar, lat in zip(cells["Polar"], cells["Lat"])
            ]
            cells["PanelRightGroupLabel"] = [
                f"{config['labels']['polarity'][str(polar)]} | {lat}"
                for polar, lat in zip(cells["Polar"], cells["Lat"])
            ]
            cells["MatrixRow"] = cells["EndpointID"].map(endpoint_order)
            cells["MatrixColumn"] = cells["FeatureOrder"].astype(int)
            cells["MatrixRowValue"] = cells["EndpointID"]
            cells["MatrixColumnValue"] = (
                cells["Metric"].astype(str)
                + "|"
                + cells["FeatureOutput"].astype(str)
                + "|"
                + cells["Band"].astype(str)
            )
            cells["MatrixRowLabel"] = cells["EndpointID"].map(
                config["labels"]["endpoint"]
            )
            cells["MatrixColumnLabel"] = (
                config["labels"]["delta_prefix"]
                + cells["FeatureLabel"].astype(str)
            )
            cells["DisplayRegionValue"] = region
            if cells[["PanelRow", "PanelRightGroup", "MatrixRow"]].isna().any().any():
                raise ValueError(
                    f"Scale-centered composite {figure_id} has unsupported levels."
                )
            duplicate_keys = [
                "PanelRow",
                "PanelColumn",
                "MatrixRow",
                "MatrixColumn",
            ]
            if cells.duplicated(duplicate_keys).any():
                raise ValueError(
                    f"Scale-centered composite {figure_id} contains duplicate cells."
                )
            group_sizes = (
                cells[["PanelRow", "PanelRightGroup"]]
                .drop_duplicates()
                .groupby("PanelRightGroup")["PanelRow"]
                .nunique()
            )
            if len(group_sizes) != 4 or not group_sizes.eq(3).all():
                raise ValueError(
                    f"Scale-centered composite {figure_id} must contain four "
                    "three-phase right-strip groups."
                )
            if cells["FeatureOrder"].nunique() != expected_features:
                raise ValueError(
                    f"Scale-centered composite {figure_id} does not contain "
                    f"{expected_features} features."
                )

            scale_slug = _filename_slug(scale_label)
            output_parent = (
                config.output_root / "heatmap-by-scale" / scale_slug / domain
            )
            filename = config["outputs"]["composite_filename_template"].format(
                Scale=scale_slug,
                Domain=_filename_slug(domain),
                RegionValue=_filename_slug(region),
            )
            p_values = pd.to_numeric(
                cells[config["inputs"]["p_column"]], errors="coerce"
            )
            figure_rows.append(
                {
                    "AnalysisID": config.analysis_id,
                    "StatsAnalysisID": config.stats_analysis_id,
                    "FigureID": figure_id,
                    "FigureKind": "composite",
                    "TestKind": "correlation",
                    "TestID": "clinical-correlation",
                    "Layout": "phase_rows_composite",
                    "View": "matrix",
                    "Scale": str(scale_values[0]),
                    "ScaleLabel": scale_label,
                    "Domain": domain,
                    "RegionVariable": str(region_variables[0]),
                    "RegionValue": region,
                    "SourceRegionValues": ";".join(
                        sorted(group["RegionValue"].astype(str).unique())
                    ),
                    "RegionLabel": region_label,
                    "Polar": "Anodal;Cathodal",
                    "Lat": "Ipsi;Contra",
                    "ColorbarLabel": config["labels"]["colorbar"][scale_label],
                    "OutputPath": str(output_parent / filename),
                    "n_features": expected_features,
                    "n_cells": len(cells),
                    "n_estimable": int(cells["rho"].notna().sum()),
                    "n_significant": int(p_values.lt(0.05).sum()),
                    "Status": "planned",
                    "Message": "",
                }
            )
            cell_frames.append(cells)

    figures = pd.DataFrame(figure_rows)
    cells = pd.concat(cell_frames, ignore_index=True)
    if len(figures) != 30 or len(cells) != 11808:
        raise ValueError(
            "Scale-centered heatmap plan must contain 30 figures and 11,808 cells; "
            f"found {len(figures)} figures and {len(cells)} cells."
        )
    if (
        figures["FigureID"].duplicated().any()
        or figures["OutputPath"].duplicated().any()
    ):
        raise ValueError(
            "Scale-centered heatmap plan contains duplicate figures or paths."
        )
    figure_kind_counts = figures["FigureKind"].value_counts().to_dict()
    if figure_kind_counts != {"atomic": 24, "composite": 6}:
        raise ValueError(
            "Scale-centered heatmap plan must contain 24 atomic and six "
            "composite figures."
        )
    correlation_kind_counts = cells.groupby(["CorrelationID", "FigureKind"]).size()
    if not correlation_kind_counts.eq(1).all():
        raise ValueError(
            "Scale-centered heatmap plan must register each correlation once per "
            "figure kind."
        )
    if cells["CorrelationID"].nunique() != 5904:
        raise ValueError(
            "Scale-centered heatmap plan must contain 5,904 unique correlations."
        )
    return VizPlan(figures=figures, coverage=cells, model_data={}, result_tables={})


def build_correlation_heatmap_plan(config: VizConfig) -> VizPlan:
    """Construct the registered clinical-correlation heatmap layout."""

    correlations = load_correlation_results(config)
    factor_levels = config["factor_levels"]
    expected_levels = {
        "Phase": factor_levels["Phase"],
        "Lat": factor_levels["Lat"],
        "Polar": factor_levels["Polar"],
    }
    for column, expected in expected_levels.items():
        observed = set(correlations[column].astype(str))
        if observed != set(expected):
            raise ValueError(
                f"Correlation result {column} levels must be {expected}; found {sorted(observed)}."
            )

    if config.analysis_id == SUBMISSION_CLINICAL_CORRELATION_SCALE_HEATMAP_ID:
        return _build_scale_correlation_heatmap_plan(config, correlations)

    figure_rows: list[dict[str, Any]] = []
    cell_frames: list[pd.DataFrame] = []
    group_fields = ["EndpointID", "Domain", "DisplayRegion", "Polar", "Phase"]
    for keys, group in correlations.groupby(
        group_fields, observed=True, sort=False, dropna=False
    ):
        endpoint, domain, region, polar, phase = map(str, keys)
        region_label = config["labels"]["region"][region]
        polar_label = config["labels"]["polarity"][polar]
        variants: list[tuple[str, str, list[str], pd.DataFrame]] = []
        for figure_spec in config["figures"]:
            layout = str(figure_spec["layout"])
            if layout == "laterality_singletons":
                for lat in figure_spec["split_levels"]:
                    variants.append(
                        (
                            layout,
                            str(lat),
                            [str(lat)],
                            group.loc[group["Lat"].eq(lat)],
                        )
                    )
            else:
                variants.append(
                    (
                        layout,
                        "Ipsi-Contra",
                        [str(value) for value in figure_spec["row_levels"]],
                        group,
                    )
                )

        for layout, lat_display, row_levels, variant_group in variants:
            figure_id = _figure_id(
                config,
                endpoint=endpoint,
                domain=domain,
                region=region,
                polar=polar,
                phase=phase,
                lat_display=lat_display,
            )
            cells = variant_group.copy()
            cells["FigureID"] = figure_id
            cells["Layout"] = layout
            cells["LatDisplay"] = lat_display
            cells["PanelRow"] = pd.Categorical(
                cells["Lat"], categories=row_levels, ordered=True
            ).codes
            if cells["PanelRow"].lt(0).any():
                raise ValueError(
                    f"Clinical heatmap {figure_id} contains an unsupported Lat value."
                )
            cells["PanelColumn"] = 0
            cells["PanelTopLabel"] = f"{region_label} | {polar_label} | {phase}"
            cells["PanelRightLabel"] = cells["Lat"]
            cells["MatrixRow"] = cells["ScaleOrder"].astype(int)
            cells["MatrixColumn"] = cells["FeatureOrder"].astype(int)
            cells["MatrixRowValue"] = cells["Scale"]
            cells["MatrixColumnValue"] = cells["PredictorID"]
            cells["MatrixRowLabel"] = cells["ScaleLabel"]
            cells["MatrixColumnLabel"] = config["labels"]["delta_prefix"] + cells[
                "FeatureLabel"
            ].astype(str)
            duplicate_keys = [
                "PanelRow",
                "PanelColumn",
                "MatrixRow",
                "MatrixColumn",
            ]
            if cells.duplicated(duplicate_keys).any():
                raise ValueError(
                    f"Clinical heatmap {figure_id} contains duplicate cells."
                )
            if sorted(cells["MatrixRow"].unique()) != list(range(28)):
                raise ValueError(
                    f"Clinical heatmap {figure_id} does not contain 28 scales."
                )
            expected_features = 44 if domain == "local" else 35
            if sorted(cells["MatrixColumn"].unique()) != list(
                range(expected_features)
            ):
                raise ValueError(
                    f"Clinical heatmap {figure_id} does not contain "
                    f"{expected_features} features."
                )
            if sorted(cells["PanelRow"].unique()) != list(range(len(row_levels))):
                raise ValueError(
                    f"Clinical heatmap {figure_id} does not contain its Lat panels."
                )

            output_parent = config.output_root / "heatmap"
            if config.analysis_id != SUBMISSION_CLINICAL_CORRELATION_HEATMAP_ID:
                output_parent /= config.stats_analysis_id
            output_parent = (
                output_parent / endpoint / domain / region / polar / phase
            )
            filename = config["outputs"]["filename_template"].format(
                AnalysisID=config.analysis_id,
                EndpointID=endpoint,
                Domain=domain,
                RegionValue=region,
                Polar=polar,
                Phase=phase,
                LatDisplay=lat_display,
            )
            figure_rows.append(
                {
                    "AnalysisID": config.analysis_id,
                    "StatsAnalysisID": config.stats_analysis_id,
                    "FigureID": figure_id,
                    "TestKind": "correlation",
                    "TestID": "clinical-correlation",
                    "Layout": layout,
                    "LatDisplay": lat_display,
                    "View": "matrix",
                    "Domain": domain,
                    "RegionValue": region,
                    "RegionLabel": region_label,
                    "EndpointID": endpoint,
                    "EndpointLabel": str(cells["EndpointLabel"].iloc[0]),
                    "Polar": polar,
                    "Phase": phase,
                    "ColorbarLabel": config["labels"]["colorbar"][endpoint],
                    "OutputPath": str(output_parent / filename),
                    "n_cells": len(cells),
                    "n_estimable": int(cells["rho"].notna().sum()),
                    "n_significant": int(
                        pd.to_numeric(
                            cells[config["inputs"]["p_column"]], errors="coerce"
                        )
                        .lt(0.05)
                        .sum()
                    ),
                    "Status": "planned",
                    "Message": "",
                }
            )
            cell_frames.append(cells)

    figures = pd.DataFrame(figure_rows)
    cells = pd.concat(cell_frames, ignore_index=True)
    if len(figures) != 216:
        raise ValueError(
            f"Clinical heatmap plan must contain 216 figures; found {len(figures)}."
        )
    if (
        figures["FigureID"].duplicated().any()
        or figures["OutputPath"].duplicated().any()
    ):
        raise ValueError("Clinical heatmap plan contains duplicate figures or paths.")
    if len(cells) != 2 * len(correlations):
        raise ValueError(
            "Clinical split and faceted heatmaps must register each correlation twice."
        )
    return VizPlan(figures=figures, coverage=cells, model_data={}, result_tables={})


def _render_scale_correlation_composite_heatmap(
    cells: pd.DataFrame, figure: Mapping[str, Any], config: VizConfig
):
    require_columns(
        cells,
        {"PanelRightGroup", "PanelRightGroupLabel"},
        "Scale-centered composite cells",
    )
    data = copy.deepcopy(config.data)
    geometry = data["style"]["geometry_mm"]
    phase_strip_width_mm = float(geometry["strip_right_width"])
    group_strip_width_mm = float(geometry["group_strip_right_width"])
    group_strip_pad_mm = float(geometry["group_strip_pad"])
    geometry["strip_right_width"] = (
        phase_strip_width_mm + group_strip_pad_mm + group_strip_width_mm
    )
    composite_config = VizConfig(config.path, data)
    fig = render_well_heatmap_figure(
        cells,
        figure,
        composite_config,
        value_column=config["inputs"]["value_column"],
        significance_column=config["inputs"]["p_column"],
        geometry_key="matrix",
    )

    phase_labels = set(config["factor_levels"]["Phase"])
    phase_strip_axes = [
        axis
        for axis in fig.axes
        if getattr(axis, "_visualdf_strip_axis", False)
        and len(axis.texts) == 1
        and axis.texts[0].get_text() in phase_labels
    ]
    panel_rows = sorted(cells["PanelRow"].astype(int).unique())
    if len(panel_rows) != 12 or len(phase_strip_axes) != len(panel_rows):
        raise ValueError(
            "Scale-centered composite must render exactly 12 Phase strips."
        )
    phase_strip_axes.sort(key=lambda axis: axis.get_position().y0, reverse=True)
    row_axes = dict(zip(panel_rows, phase_strip_axes))

    figure_width_inches = float(fig.get_size_inches()[0])
    phase_width = phase_strip_width_mm / 25.4 / figure_width_inches
    group_width = group_strip_width_mm / 25.4 / figure_width_inches
    group_pad = group_strip_pad_mm / 25.4 / figure_width_inches
    for axis in phase_strip_axes:
        position = axis.get_position()
        axis.set_position([position.x0, position.y0, phase_width, position.height])

    strips = config["style"]["strips"]
    font = config["style"]["font"]
    group_table = cells[
        ["PanelRightGroup", "PanelRightGroupLabel", "PanelRow"]
    ].drop_duplicates()
    for group_index, group in group_table.groupby(
        "PanelRightGroup", sort=True, observed=True
    ):
        rows = sorted(group["PanelRow"].astype(int).unique())
        if len(rows) != 3 or rows != list(range(rows[0], rows[0] + 3)):
            raise ValueError(
                f"Composite right-strip group {group_index} is not three consecutive rows."
            )
        labels = group["PanelRightGroupLabel"].astype(str).unique()
        if len(labels) != 1:
            raise ValueError(
                f"Composite right-strip group {group_index} has inconsistent labels."
            )
        positions = [row_axes[row].get_position() for row in rows]
        y0 = min(position.y0 for position in positions)
        y1 = max(position.y1 for position in positions)
        x0 = positions[0].x0 + phase_width + group_pad
        axis = fig.add_axes([x0, y0, group_width, y1 - y0])
        axis._visualdf_strip_axis = True
        axis.set_facecolor(strips["right_background"])
        axis.patch.set_alpha(1.0)
        axis.text(
            0.5,
            0.5,
            labels[0],
            ha="center",
            va="center",
            rotation=-90,
            fontsize=font["strip_pt"],
            family=font["family"],
            color=strips["text_color"],
            fontweight=strips["font_weight"],
        )
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
    return fig


def render_correlation_heatmap_figure(
    cells: pd.DataFrame, figure: Mapping[str, Any], config: VizConfig
):
    """Render one clinical correlation PDF through the shared well-grid adapter."""

    if figure["Layout"] == "phase_rows_composite":
        return _render_scale_correlation_composite_heatmap(cells, figure, config)

    return render_well_heatmap_figure(
        cells,
        figure,
        config,
        value_column=config["inputs"]["value_column"],
        significance_column=config["inputs"]["p_column"],
        geometry_key="matrix",
    )
