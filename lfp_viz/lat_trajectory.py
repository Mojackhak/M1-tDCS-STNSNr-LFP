"""Build and render paired-component EMM trajectory figures."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lfp_cohort.io import read_stage_table
from lfp_stats.runner import _contact_unit_id

from . import visualdf
from .config import VizConfig
from .data import path_below, read_csv, require_columns
from .plan import VizPlan, _filename_slug
from .scalar import y_axis_label
from .submission_scalar import scalar_output_group

STATISTICS_COLUMNS = {
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
    "Contrast",
    "Phase",
    "emmean",
    "p_holm",
    "ModelStatus",
    "ModelSingular",
    "n_rows",
    "n_ID",
}
PAIRING_COLUMNS = {"ModelID", "SourcePath", "InputPath_x"}


def _line_var(config: VizConfig) -> str:
    return str(config["figures"][0]["line_var"])


def _condition_fields(config: VizConfig) -> list[str]:
    return list(config["inputs"]["figure_condition_fields"])


def _emm_condition_fields(config: VizConfig) -> list[str]:
    line_var = _line_var(config)
    emmeans = config["model"]["emmeans"]
    return [
        str(field)
        for field in (emmeans["panel_var"], emmeans["facet_var"])
        if field is not None and str(field) != line_var
    ]


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "t", "1"}:
            return True
        if normalized in {"false", "f", "0"}:
            return False
        raise ValueError(f"Cannot interpret boolean value: {value!r}")
    return bool(value)


def _component_model_plan(model_ids: list[str], config: VizConfig) -> dict[str, Any]:
    factor_levels = dict(config["factor_levels"])
    factor_levels["Phase"] = list(config["inputs"]["fit_phases"])
    emmeans = config["model"]["emmeans"]
    return {
        "factor_levels": factor_levels,
        "factor_columns": list(config["model"]["factor_columns"]),
        "model": {
            "formula": config["model"]["formula"],
            "reml": config["model"]["reml"],
        },
        "models": [{"model_id": model_id} for model_id in model_ids],
        "tests": [
            {
                "id": "Phase",
                "kind": "emm_pairwise",
                "x_var": emmeans["x_var"],
                "panel_var": emmeans["panel_var"],
                "facet_var": emmeans["facet_var"],
                "weights": "equal",
                "confidence_level": config["model"]["confidence_level"],
                "within_adjust": "tukey",
                "across_method": "none",
                "primary_p": "p_within",
                "contrast_direction": "later_minus_earlier",
            }
        ],
    }


def _run_component_models(
    model_data: pd.DataFrame,
    model_ids: list[str],
    config: VizConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit the registered display-only component model with the shared R runner."""

    rscript = shutil.which("Rscript")
    if rscript is None:
        raise FileNotFoundError("Rscript is unavailable on PATH.")
    repo_root = Path(__file__).resolve().parents[1]
    runner = repo_root / "lfp_stats" / "r" / "run_scalar_models.R"
    runtime_root = Path(config["paths"]["runtime_root"])
    runtime_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{config.analysis_id}-runtime-", dir=runtime_root
    ) as temporary_name:
        temporary = Path(temporary_name)
        data_path = temporary / "model-data.csv"
        plan_path = temporary / "plan.json"
        result_root = temporary / "r-results"
        model_data.to_csv(data_path, index=False)
        plan_path.write_text(
            json.dumps(_component_model_plan(model_ids, config), ensure_ascii=False),
            encoding="utf-8",
        )
        completed = subprocess.run(
            [rscript, str(runner), str(data_path), str(plan_path), str(result_root)],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            message = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"Component trajectory model failed: {message}")
        model_fit = pd.read_csv(result_root / "model_status.csv")
        emmeans = pd.read_csv(result_root / "emm.csv")
    return model_fit, emmeans


def _selected_models(
    statistics: pd.DataFrame, config: VizConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selection = config["inputs"]["selection"]
    p_column = str(selection["column"])
    p_values = pd.to_numeric(statistics[p_column], errors="coerce")
    significant = statistics.loc[p_values < float(selection["value"])].copy()
    accepted = set(config["inputs"]["accepted_model_statuses"])
    significant = significant.loc[
        significant["ModelStatus"].astype(str).isin(accepted)
    ].copy()
    if significant.empty:
        raise ValueError("Trajectory selection contains no significant cells.")
    model_ids = significant["ModelID"].astype(str).drop_duplicates().tolist()
    selected = statistics.loc[statistics["ModelID"].astype(str).isin(model_ids)].copy()
    phases = set(config["inputs"]["fit_phases"])
    group_fields = ["ModelID", *_emm_condition_fields(config)]
    grouped_phases = selected.groupby(group_fields, observed=True)["Phase"].agg(
        lambda values: set(values.astype(str))
    )
    if grouped_phases.ne(phases).any():
        raise ValueError("Selected models do not contain every fitted Phase cell.")
    return significant, selected


def _model_component_rows(
    source: pd.DataFrame,
    model: pd.Series,
    source_path: Path,
    config: VizConfig,
) -> pd.DataFrame:
    region_column = str(model["RegionVariable"])
    line_var = _line_var(config)
    components = dict(config["inputs"]["component_columns"])
    source_strata = list(config["inputs"]["source_strata_fields"])
    model_factors = [
        field for field in config["factor_levels"] if field not in {"Phase", line_var}
    ]
    required = {
        "ID",
        "Phase",
        "Domain",
        "Metric",
        "FeatureOutput",
        "Representation",
        "Band",
        "IncludeAggregate",
        region_column,
        *config["model"]["factor_columns"],
        *source_strata,
        *model_factors,
        *components.values(),
    }
    if config.kind != "region_trajectory":
        required.add("Contrast")
    require_columns(source, required, f"Component trajectory source {source_path}")
    include_values = set(source["IncludeAggregate"].dropna().unique())
    if not include_values.issubset({True, False, np.bool_(True), np.bool_(False)}):
        raise TypeError(f"IncludeAggregate must be boolean: {source_path}")

    mask = (
        source["IncludeAggregate"].fillna(False).astype(bool)
        & source["Domain"].eq(model["Domain"])
        & source["Metric"].eq(model["Metric"])
        & source["FeatureOutput"].eq(model["FeatureOutput"])
        & source["Representation"].eq(model["Representation"])
        & source["Band"].eq(model["Band"])
        & source[region_column].eq(model["RegionValue"])
        & source["Phase"].isin(config["inputs"]["fit_phases"])
    )
    if config.kind != "region_trajectory":
        mask &= source["Contrast"].eq(model["Contrast"])
    for field in source_strata:
        mask &= source[field].eq(model[field])
    selected = source.loc[mask].copy()
    numeric_components: dict[str, pd.Series] = {}
    for level, column in components.items():
        numeric = pd.to_numeric(selected[column], errors="coerce")
        invalid = selected[column].notna() & numeric.isna()
        if invalid.any():
            raise TypeError(f"{column} contains non-numeric values: {source_path}")
        numeric_components[level] = numeric
    finite = np.logical_and.reduce(
        [
            np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))
            for values in numeric_components.values()
        ]
    )
    selected = selected.loc[finite].copy()
    if len(selected) != int(model["n_rows"]):
        raise ValueError(
            "Component trajectory source row count differs from the direct model: "
            f"{model['ModelID']}."
        )
    selected = selected.reset_index(drop=True)
    selected["ComponentPairID"] = [
        f"{model['ModelID']}::{index}" for index in selected.index
    ]

    frames: list[pd.DataFrame] = []
    component_difference = config["inputs"]["component_difference"]
    keep_columns = list(
        dict.fromkeys(
            [
                *config["model"]["factor_columns"],
                "ComponentPairID",
                "Phase",
                *[field for field in model_factors if field in selected.columns],
            ]
        )
    )
    for level in config["factor_levels"][line_var]:
        frame = selected[keep_columns].copy()
        frame[line_var] = level
        frame["ComponentContrast"] = (
            0.5 if level == component_difference["minuend"] else -0.5
        )
        frame["Value"] = pd.to_numeric(
            selected[components[level]], errors="raise"
        ).to_numpy(dtype=float)
        frame["ModelID"] = str(model["ModelID"])
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    result["Phase"] = pd.Categorical(
        result["Phase"],
        categories=config["inputs"]["fit_phases"],
        ordered=True,
    )
    for factor, levels in config["factor_levels"].items():
        if factor == "Phase" or factor not in result.columns:
            continue
        result[factor] = pd.Categorical(
            result[factor].astype(str), categories=levels, ordered=True
        )
    return result


def _pre_source_rows(
    source: pd.DataFrame,
    model: pd.Series,
    source_path: Path,
    config: VizConfig,
    *,
    component_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Return and verify the exact-zero normalized Pre anchors for one model."""

    region_column = str(model["RegionVariable"])
    line_var = _line_var(config)
    region_components = config.kind == "region_trajectory"
    source_strata = list(config["inputs"]["source_strata_fields"])
    condition_fields = _condition_fields(config)
    required = {
        "ID",
        "Phase",
        "Value",
        "Domain",
        "Metric",
        "FeatureOutput",
        "Representation",
        "Band",
        "IncludeAggregate",
        line_var,
        *source_strata,
        *condition_fields,
    }
    if region_components:
        required.update(config["inputs"]["anchor_pair_columns"])
    else:
        required.add(region_column)
    if config.kind == "polar_trajectory":
        contact_column = "Channel" if model["Domain"] == "local" else "ChannelPair"
        required.add(contact_column)
    require_columns(source, required, f"Trajectory Pre source {source_path}")
    mask = (
        source["IncludeAggregate"].fillna(False).astype(bool)
        & source["Domain"].eq(model["Domain"])
        & source["Metric"].eq(model["Metric"])
        & source["FeatureOutput"].eq(model["FeatureOutput"])
        & source["Representation"].eq(model["Representation"])
        & source["Band"].eq(model["Band"])
        & source["Phase"].eq(config["inputs"]["anchor_phase"])
    )
    if region_components:
        mask &= source[line_var].isin(config["factor_levels"][line_var])
    else:
        mask &= source[region_column].eq(model["RegionValue"])
    for field in source_strata:
        mask &= source[field].eq(model[field])
    selected = source.loc[mask].copy()
    if config.kind in {"lat_trajectory", "polar_trajectory"}:
        support_columns = [
            "ID",
            *[field for field in condition_fields if field not in source_strata],
            line_var,
        ]
        if config.kind == "polar_trajectory":
            selected["ContactUnitID"] = selected[contact_column].map(_contact_unit_id)
            support_columns.append("ContactUnitID")
        support = component_rows[support_columns].drop_duplicates()
        selected = selected.merge(
            support, on=support_columns, how="inner", validate="many_to_one"
        )
    if selected.empty:
        raise ValueError(f"Trajectory Pre source is empty: {model['ModelID']}")
    values = pd.to_numeric(selected["Value"], errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float, na_value=np.nan)).all():
        raise ValueError(f"Trajectory Pre source is not finite: {model['ModelID']}")
    if not values.eq(0.0).all():
        maximum = float(values.abs().max())
        raise ValueError(
            "Trajectory Pre source is not exactly zero; "
            f"model={model['ModelID']}, maximum absolute value={maximum:.6g}."
        )
    if region_components:
        pair_columns = list(config["inputs"]["anchor_pair_columns"])
        duplicate = selected.duplicated([*pair_columns, line_var], keep=False)
        if duplicate.any():
            raise ValueError(
                "Trajectory Pre source contains duplicate Region cells: "
                f"{model['ModelID']}."
            )
        wide = selected.pivot(
            index=pair_columns,
            columns=line_var,
            values="Value",
        )
        levels = list(config["factor_levels"][line_var])
        complete_index = wide.dropna(subset=levels).index
        if complete_index.empty:
            raise ValueError(
                f"Trajectory Pre source has no complete Region pair: {model['ModelID']}"
            )
        selected_index = pd.MultiIndex.from_frame(selected[pair_columns])
        selected = selected.loc[selected_index.isin(complete_index)].copy()
    return selected


def _validate_component_emmeans(
    emmeans: pd.DataFrame,
    direct_statistics: pd.DataFrame,
    model_ids: list[str],
    config: VizConfig,
) -> pd.Series:
    line_var = _line_var(config)
    conditions = _emm_condition_fields(config)
    cell_fields = ["ModelID", "Phase", *conditions, line_var]
    require_columns(
        emmeans,
        {
            "ModelID",
            "TestKind",
            "TestID",
            "Phase",
            line_var,
            *conditions,
            "emmean",
            "lower.CL",
            "upper.CL",
        },
        "Component trajectory EMMs",
    )
    expected_per_model = len(config["inputs"]["fit_phases"])
    expected_per_model *= len(config["factor_levels"][line_var])
    for field in conditions:
        expected_per_model *= len(config["factor_levels"][field])
    if len(emmeans) != len(model_ids) * expected_per_model:
        raise ValueError("Component trajectory EMM row count is invalid.")
    if emmeans.duplicated(cell_fields, keep=False).any():
        raise ValueError("Component trajectory EMMs contain duplicate cells.")

    wide = emmeans.pivot(
        index=["ModelID", "Phase", *conditions],
        columns=line_var,
        values="emmean",
    )
    difference = config["inputs"]["component_difference"]
    component_difference = (
        wide[difference["minuend"]].astype(float)
        - wide[difference["subtrahend"]].astype(float)
    ).rename("component_difference")
    direct = direct_statistics.set_index(["ModelID", "Phase", *conditions])[
        "emmean"
    ].astype(float)
    aligned = pd.concat([component_difference, direct.rename("direct")], axis=1)
    absolute_gap = (aligned["component_difference"] - aligned["direct"]).abs()
    tolerance = float(config["inputs"]["difference_tolerance"])
    if aligned.isna().any().any() or not np.allclose(
        aligned["component_difference"],
        aligned["direct"],
        rtol=0.0,
        atol=tolerance,
    ):
        difference = float(np.nanmax(absolute_gap))
        raise ValueError(
            "Component EMM difference does not reproduce the direct EMM; "
            f"maximum absolute difference={difference:.6g}, "
            f"tolerance={tolerance:.6g}."
        )

    phases = set(config["inputs"]["fit_phases"])
    line_levels = set(config["factor_levels"][line_var])
    for model_id in model_ids:
        selected = emmeans.loc[emmeans["ModelID"].astype(str).eq(model_id)]
        if (
            set(selected["Phase"].astype(str)) != phases
            or set(selected[line_var].astype(str)) != line_levels
        ):
            raise ValueError(f"Component EMM levels are incomplete: {model_id}")
    return (
        absolute_gap.groupby(level="ModelID").max().rename("ComponentDifferenceMaxAbs")
    )


def _add_pre_anchors(emmeans: pd.DataFrame, config: VizConfig) -> pd.DataFrame:
    line_var = _line_var(config)
    fitted = emmeans.copy()
    fitted["EstimateType"] = "fitted-emmean"
    fitted["Modeled"] = True
    anchors: list[pd.Series] = []
    for level in config["factor_levels"][line_var]:
        anchor = fitted.loc[fitted[line_var].astype(str).eq(level)].iloc[0].copy()
        anchor["Phase"] = config["inputs"]["anchor_phase"]
        for column in ("emmean", "lower.CL", "upper.CL", "SE"):
            if column in anchor.index:
                anchor[column] = 0.0
        if "df" in anchor.index:
            anchor["df"] = np.nan
        anchor["EstimateType"] = "normalization-anchor"
        anchor["Modeled"] = False
        anchors.append(anchor)
    result = pd.concat([pd.DataFrame(anchors), fitted], ignore_index=True)
    result["Phase"] = pd.Categorical(
        result["Phase"].astype(str),
        categories=config["factor_levels"]["Phase"],
        ordered=True,
    )
    result[line_var] = pd.Categorical(
        result[line_var].astype(str),
        categories=config["factor_levels"][line_var],
        ordered=True,
    )
    return result.sort_values([line_var, "Phase"]).reset_index(drop=True)


def build_lat_trajectory_plan(config: VizConfig) -> VizPlan:
    """Build the current Holm-selected paired-component trajectory figures."""

    statistics_path = Path(config["paths"]["statistics"])
    pairing_path = Path(config["paths"]["pairing"])
    statistics = read_csv(statistics_path, "Submission trajectory statistics")
    pairing = read_csv(pairing_path, "Submission trajectory pairing")
    if config.kind == "region_trajectory" and "Contrast" not in statistics:
        statistics["Contrast"] = statistics["RegionValue"]
    required_statistics = STATISTICS_COLUMNS | set(_condition_fields(config))
    required_statistics |= set(config["inputs"]["source_strata_fields"])
    require_columns(statistics, required_statistics, "Trajectory statistics")
    require_columns(pairing, PAIRING_COLUMNS, "Trajectory pairing")
    if not statistics["AnalysisID"].eq(config.stats_analysis_id).all():
        raise ValueError(
            "Statistics AnalysisID does not match the trajectory configuration."
        )

    significant, selected_statistics = _selected_models(statistics, config)
    model_ids = significant["ModelID"].astype(str).drop_duplicates().tolist()
    model_rows = (
        selected_statistics.sort_values(["ModelID", "Phase"])
        .drop_duplicates("ModelID")
        .set_index("ModelID", drop=False)
        .loc[model_ids]
        .reset_index(drop=True)
    )
    pairing_by_model = pairing.drop_duplicates("ModelID").set_index("ModelID")
    missing_pairing = sorted(set(model_ids) - set(pairing_by_model.index.astype(str)))
    if missing_pairing:
        raise ValueError(f"Trajectory pairing rows are missing: {missing_pairing}")

    source_cache: dict[Path, pd.DataFrame] = {}
    pre_source_cache: dict[Path, pd.DataFrame] = {}
    component_tables: dict[str, pd.DataFrame] = {}
    pre_tables: dict[str, pd.DataFrame] = {}
    source_paths: dict[str, Path] = {}
    for _, model in model_rows.iterrows():
        model_id = str(model["ModelID"])
        pairing_row = pairing_by_model.loc[model_id]
        source_path = path_below(
            Path(str(pairing_row["SourcePath"])),
            config.source_root,
            "Component trajectory source path",
        )
        input_path = Path(str(pairing_row["InputPath_x"])).expanduser().resolve()
        for path, label in (
            (source_path, "component trajectory source"),
            (input_path, "trajectory Pre source"),
        ):
            if not path.exists() or path.stat().st_size == 0:
                raise FileNotFoundError(f"{label} is missing: {path}")
        if source_path not in source_cache:
            source_cache[source_path] = read_stage_table(source_path)
        if input_path not in pre_source_cache:
            pre_source_cache[input_path] = read_stage_table(input_path)
        component_tables[model_id] = _model_component_rows(
            source_cache[source_path], model, source_path, config
        )
        pre_tables[model_id] = _pre_source_rows(
            pre_source_cache[input_path],
            model,
            input_path,
            config,
            component_rows=component_tables[model_id],
        )
        source_paths[model_id] = source_path

    combined = pd.concat(component_tables.values(), ignore_index=True)
    model_fit, fitted_emmeans = _run_component_models(combined, model_ids, config)
    require_columns(
        model_fit, {"ModelID", "Singular", "Status", "Message"}, "Model fit"
    )
    if len(model_fit) != len(model_ids) or model_fit["ModelID"].duplicated().any():
        raise ValueError("Trajectory model-fit results do not match the plan.")
    accepted = set(config["inputs"]["accepted_model_statuses"])
    unsupported = sorted(set(model_fit["Status"].astype(str)) - accepted)
    if unsupported:
        raise ValueError(
            f"Trajectory component models have unsupported statuses: {unsupported}"
        )
    component_gaps = _validate_component_emmeans(
        fitted_emmeans, selected_statistics, model_ids, config
    )

    metadata_columns = [
        "ModelID",
        "Domain",
        "Metric",
        "FeatureOutput",
        "Representation",
        "Band",
        "RegionVariable",
        "RegionValue",
        "Contrast",
        *config["inputs"]["source_strata_fields"],
    ]
    metadata_columns = list(dict.fromkeys(metadata_columns))
    metadata = model_rows[metadata_columns].copy()
    fitted_emmeans = fitted_emmeans.merge(
        metadata, on="ModelID", how="left", validate="many_to_one"
    )
    fitted_emmeans.insert(0, "AnalysisID", config.analysis_id)
    fitted_emmeans.insert(1, "StatsAnalysisID", config.stats_analysis_id)
    model_fit = model_fit.merge(
        metadata, on="ModelID", how="left", validate="one_to_one"
    )
    model_fit["ComponentDifferenceMaxAbs"] = model_fit["ModelID"].map(component_gaps)
    model_fit["DifferenceTolerance"] = float(config["inputs"]["difference_tolerance"])
    model_fit.insert(0, "AnalysisID", config.analysis_id)
    model_fit.insert(1, "StatsAnalysisID", config.stats_analysis_id)

    condition_fields = _condition_fields(config)
    line_var = _line_var(config)
    figure_groups = significant[["ModelID", *condition_fields]].drop_duplicates()
    figures: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    display_emmeans: list[pd.DataFrame] = []
    result_tables: dict[tuple[str, str, str], pd.DataFrame] = {}
    filename_template = str(config["outputs"]["filename_template"])
    fit_by_model = model_fit.set_index("ModelID")
    model_by_id = model_rows.set_index("ModelID", drop=False)
    figure_config = config["figures"][0]

    for _, figure_group in figure_groups.iterrows():
        model_id = str(figure_group["ModelID"])
        model = model_by_id.loc[model_id]
        condition_values = {field: figure_group[field] for field in condition_fields}
        data_key = "__".join(
            [
                model_id,
                *[f"{field}-{condition_values[field]}" for field in condition_fields],
            ]
        )
        group = scalar_output_group(model["Metric"], model["FeatureOutput"])
        format_values = {
            "Band": _filename_slug(model["Band"]),
            "RegionValue": _filename_slug(model["RegionValue"]),
            "Polar": _filename_slug(
                condition_values.get("Polar", model.get("Polar", "Anodal-Cathodal"))
            ),
            "Lat": _filename_slug(condition_values.get("Lat", "Ipsi-Contra")),
        }
        filename = filename_template.format(**format_values)
        output_path = config.output_root / group / filename

        model_emm = fitted_emmeans.loc[
            fitted_emmeans["ModelID"].astype(str).eq(model_id)
        ].copy()
        model_significant = significant.loc[
            significant["ModelID"].astype(str).eq(model_id)
        ].copy()
        for field, value in condition_values.items():
            if field in model_emm.columns:
                model_emm = model_emm.loc[model_emm[field].eq(value)].copy()
            model_significant = model_significant.loc[
                model_significant[field].eq(value)
            ].copy()
        model_emm = _add_pre_anchors(model_emm, config)
        display_emmeans.append(model_emm.assign(DataKey=data_key))

        fit = fit_by_model.loc[model_id]
        direct_singular = _as_bool(model["ModelSingular"])
        component_singular = _as_bool(fit["Singular"])
        polar_value = condition_values.get(
            "Polar", model.get("Polar", "Anodal-Cathodal")
        )
        lat_value = condition_values.get("Lat", "Ipsi-Contra")
        figures.append(
            {
                "AnalysisID": config.analysis_id,
                "StatsAnalysisID": config.stats_analysis_id,
                "ModelID": model_id,
                "DataKey": data_key,
                "FigureID": figure_config["id"],
                "View": figure_config["layout"].replace("_", "-"),
                "TestKind": "emmean_vs_null",
                "TestID": "Phase",
                "Engine": model["Engine"],
                "DirectModelStatus": model["ModelStatus"],
                "DirectSingular": direct_singular,
                "ComponentModelStatus": fit["Status"],
                "ComponentSingular": component_singular,
                "ComponentDifferenceMaxAbs": fit["ComponentDifferenceMaxAbs"],
                "DifferenceTolerance": fit["DifferenceTolerance"],
                "Singular": direct_singular or component_singular,
                "Domain": model["Domain"],
                "Metric": model["Metric"],
                "FeatureOutput": model["FeatureOutput"],
                "OutputGroup": group,
                "Representation": model["Representation"],
                "Band": model["Band"],
                "RegionVariable": model["RegionVariable"],
                "RegionValue": model["RegionValue"],
                "Polar": polar_value,
                "Lat": lat_value,
                "Contrast": model["Contrast"],
                "SourcePath": str(source_paths[model_id]),
                "StatisticsPath": str(statistics_path),
                "EMMPath": str(config.output_root / config["manifests"]["emmeans"]),
                "OutputPath": str(output_path),
                "n_rows": int(model["n_rows"]),
                "n_ID": int(component_tables[model_id]["ID"].nunique()),
                "n_significant": len(model_significant),
                "Status": "planned",
                "Message": "",
            }
        )
        result_tables[(data_key, "Phase", "emm")] = model_emm
        result_tables[(data_key, "Phase", "statistics")] = model_significant

        for phase in config["factor_levels"]["Phase"]:
            for level in config["factor_levels"][line_var]:
                modeled = phase != config["inputs"]["anchor_phase"]
                rows = component_tables[model_id] if modeled else pre_tables[model_id]
                cell = rows.loc[rows[line_var].astype(str).eq(level)].copy()
                if modeled:
                    cell = cell.loc[cell["Phase"].astype(str).eq(phase)]
                for field, value in condition_values.items():
                    if field in cell.columns:
                        cell = cell.loc[cell[field].eq(value)]
                if config.kind == "region_trajectory" and cell.empty:
                    raise ValueError(
                        "Region trajectory display cell has no source support: "
                        f"model={model_id}, phase={phase}, {line_var}={level}."
                    )
                coverage.append(
                    {
                        "AnalysisID": config.analysis_id,
                        "StatsAnalysisID": config.stats_analysis_id,
                        "ModelID": model_id,
                        "DataKey": data_key,
                        "Metric": model["Metric"],
                        "FeatureOutput": model["FeatureOutput"],
                        "Band": model["Band"],
                        "RegionValue": model["RegionValue"],
                        "Polar": polar_value if line_var != "Polar" else level,
                        "Lat": lat_value if line_var != "Lat" else level,
                        "Phase": phase,
                        "LineVariable": line_var,
                        "LineLevel": level,
                        "EstimateType": (
                            "fitted-emmean" if modeled else "normalization-anchor"
                        ),
                        "Modeled": modeled,
                        "n_ID": int(cell["ID"].nunique()),
                        "n_obs": len(cell),
                    }
                )

    figures_table = pd.DataFrame(figures)
    coverage_table = pd.DataFrame(coverage)
    if figures_table["OutputPath"].duplicated().any():
        raise ValueError("Trajectory figure plan contains duplicate output paths.")
    formal_emmeans = pd.concat(display_emmeans, ignore_index=True)
    submission_tables = {
        config.output_root / config["manifests"]["emmeans"]: formal_emmeans,
        config.output_root / config["manifests"]["statistics"]: significant,
        config.output_root / config["manifests"]["model_fit"]: model_fit,
    }
    return VizPlan(
        figures=figures_table,
        coverage=coverage_table,
        model_data=component_tables,
        result_tables=result_tables,
        submission_tables=submission_tables,
    )


def _star_label(p_value: float, config: VizConfig) -> str:
    for threshold in config["significance"]["thresholds"]:
        if p_value < float(threshold["p_lt"]):
            return str(threshold["label"])
    return ""


def render_lat_trajectory_figure(
    emmeans: pd.DataFrame,
    statistics: pd.DataFrame,
    model: pd.Series,
    config: VizConfig,
    *,
    coverage: pd.DataFrame | None = None,
):
    """Render one single-panel paired-component EMM trajectory."""

    style = config["style"]
    font = style["font"]
    widths = style["line_width_pt"]
    geometry = style["geometry_mm"]
    series = style["series"]
    figure_config = config["figures"][0]
    line_var = str(figure_config["line_var"])
    line_levels = list(series["order"])
    sample = config.data.get("sample_size")
    counts = None
    if sample is not None and sample["show"]:
        if coverage is None:
            raise ValueError("Trajectory sample sizes require per-figure coverage.")
        counts = coverage.pivot(index="Phase", columns="LineLevel", values="n_ID")
    line_palette = {level: series[level]["color"] for level in line_levels}
    line_styles = {level: series[level]["line_style"] for level in line_levels}
    line_zorders = {level: series[level]["zorder"] for level in line_levels}

    label_model = model.copy()
    label_model["Contrast"] = np.nan
    y_label = f"{config['labels']['delta_prefix']}{y_axis_label(label_model, config)}"
    strip_parts: list[str] = []
    if line_var != "Region":
        strip_parts.append(config["labels"]["region"][str(model["RegionValue"])])
    for field in _condition_fields(config):
        value = str(model[field])
        strip_parts.append(
            config["labels"]["polarity"][value] if field == "Polar" else value
        )
    strip_label = " | ".join(strip_parts)
    y_limits = visualdf._auto_y_limits_scalar(
        pd.DataFrame({"Value": [float(config["reference_line"]["value"])]}),
        emmeans,
        value_col="Value",
        y_limits=None,
        lower_padding_fraction=style["y_limits"]["lower_padding_fraction"],
        upper_padding_fraction=style["y_limits"]["upper_padding_fraction"],
        reference_values=[config["reference_line"]["value"]],
    )

    visualdf.set_greek_symbol_font_enabled(False)
    figure = visualdf.plot_categorical_emm_trajectory(
        emmeans,
        x_var="Phase",
        line_var=line_var,
        x_levels=list(config["factor_levels"]["Phase"]),
        line_levels=line_levels,
        x_label="Phase",
        y_label=y_label,
        top_strip_label=strip_label,
        font_family=font["latin"],
        line_palette=line_palette,
        line_styles=line_styles,
        line_zorders=line_zorders,
        line_width=widths["emm_line"],
        marker=series["mean_marker"],
        marker_size=series["mean_marker_size_pt"],
        marker_edge_width=series["mean_marker_edge_width_pt"],
        error_bar_linewidth=widths["emm_ci"],
        error_bar_cap=series["ci_cap_pt"],
        y_limits=y_limits,
        reference_value=config["reference_line"]["value"],
        reference_color=config["reference_line"]["color"],
        reference_style=config["reference_line"]["line_style"],
        reference_width=config["reference_line"]["line_width_pt"],
        reference_alpha=config["reference_line"]["alpha"],
        reference_zorder=config["reference_line"]["zorder"],
        grid=style["axes"]["grid"],
        show_top_right_axes=style["axes"]["show_top_right_spines"],
        label_fontsize=font["strip_pt"],
        label_top_bg_color=style["strips"]["top_background"],
        label_text_color=style["strips"]["text_color"],
        label_fontweight=style["strips"]["font_weight"],
        strip_top_height_mm=geometry["strip_top_height"],
        strip_pad_mm=geometry["strip_pad"],
        axis_label_fontsize=font["axis_pt"],
        tick_label_fontsize=font["tick_pt"],
        legend_loc=style["legend"]["location"],
        legend_fontsize=font["legend_pt"],
        legend_frame=style["legend"]["frame"],
        boxsize=(
            geometry["phase_width"] * len(config["factor_levels"]["Phase"]),
            geometry["panel_height"],
        ),
        x_label_offset_mm=geometry["x_label_offset"],
        y_label_offset_mm=geometry["y_label_offset"],
        axis_linewidth=widths["default"],
        dpi=style["dpi"],
        transparent=style["transparent"],
    )

    axis = figure.axes[0]
    ymin, ymax = axis.get_ylim()
    span = ymax - ymin
    x_positions = {
        phase: index for index, phase in enumerate(config["factor_levels"]["Phase"])
    }
    if counts is not None:
        for phase, x in x_positions.items():
            axis.text(
                x,
                sample["y_axes"],
                sample["template"].format(
                    n_1=int(counts.loc[phase, line_levels[0]]),
                    n_2=int(counts.loc[phase, line_levels[1]]),
                ),
                transform=axis.get_xaxis_transform(),
                ha="center",
                va=sample["vertical_alignment"],
                fontsize=font["sample_size_pt"],
                color=sample["color"],
                zorder=7,
            )
    significance = config["significance"]
    for _, result in statistics.sort_values("Phase").iterrows():
        p_value = float(result[significance["p_column"]])
        label = _star_label(p_value, config)
        if not label:
            continue
        phase = str(result["Phase"])
        phase_emm = emmeans.loc[emmeans["Phase"].astype(str).eq(phase)]
        phase_top = float(phase_emm["upper.CL"].max())
        y = phase_top + float(significance["star_offset_fraction"]) * span
        x = float(x_positions[phase])
        axis.text(
            x,
            y,
            label,
            ha="center",
            va="bottom",
            fontsize=font["significance_pt"],
            color="black",
            zorder=21,
            clip_on=False,
        )
    return figure
