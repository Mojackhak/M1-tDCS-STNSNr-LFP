"""Fit the independent adjacent-Phase contact-level mixed-model branch."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .config import StructuralConnectivityConfig
from .contact_lmm import (
    CONTACT_LMM_TABLES,
    FIELD_EXCESS_METRIC,
    OUTCOME_SD_KEY,
    _attach_models_and_support,
    _contact_lmm_input_rows,
    _coverage,
    _figure_specs,
    _heatmap_output_count,
    _plot_heatmaps,
    _plot_slope_figures,
    _planned_models,
    _postprocess,
    _prepare_long_rows,
    _read_csv,
    _render_contact_lmm_figures,
    _run_r_models,
    _standardize_predictors,
    _write_outputs,
    run_contact_lmm_atlas_grid_heatmaps,
    run_contact_lmm_field_excess_atlas_grid_fits,
    run_contact_lmm_field_excess_finder_figures,
)
from .statistics import _read_pickle, _validate_normalized_contract

ADJACENT_PHASE_BASIS = "adjacent_phase"
ADJACENT_ID_SETTINGS_KEY = "contact_lmm_adjacent_id_random_intercept"
ADJACENT_ID_TEMP_KEY = "contact_lmm_adjacent_id_random_intercept_temp_root"
ADJACENT_ID_OUTPUT_KEY = "contact_lmm_adjacent_id_random_intercept_root"
FIELD_EXCESS_LOCAL_OUTCOMES_KEY = "field_excess_local_outcomes"

EMPTY_FIGURE_COLUMNS = (
    "FigureID",
    "FigureType",
    "SourcePath",
    "PublishedRelativePath",
    "Status",
)


def _adjacent_scale(value: str) -> str:
    suffix = "_pre_centered"
    if not value.endswith(suffix):
        raise ValueError(
            f"Adjacent-Phase outcomes require a registered Pre-centered scale: {value}"
        )
    return f"{value.removesuffix(suffix)}_adjacent_phase"


def _adjacent_config(
    config: StructuralConnectivityConfig,
    *,
    settings_key: str = "contact_lmm_adjacent",
    temp_path_key: str = "contact_lmm_adjacent_temp_root",
    output_key: str = "contact_lmm_adjacent_root",
) -> StructuralConnectivityConfig:
    data = deepcopy(config.data)
    settings = deepcopy(data["inference"][settings_key])
    data["analysis"]["id"] = settings["analysis_id"]
    data["inference"]["contact_lmm"] = settings
    data["paths"]["contact_lmm_temp_root"] = data["paths"][temp_path_key]
    data["outputs"]["contact_lmm_root"] = data["outputs"][output_key]
    data["outcomes"]["local_scales"] = {
        metric: _adjacent_scale(str(scale))
        for metric, scale in data["outcomes"]["local_scales"].items()
    }
    data["outcomes"]["connectivity_scales"] = {
        metric: _adjacent_scale(str(scale))
        for metric, scale in data["outcomes"]["connectivity_scales"].items()
    }
    return StructuralConnectivityConfig(
        path=config.path,
        data=data,
        subjects=config.subjects,
        connectomes=config.connectomes,
    )


def _adjacent_id_random_intercept_config(
    config: StructuralConnectivityConfig,
) -> StructuralConnectivityConfig:
    return _adjacent_config(
        config,
        settings_key=ADJACENT_ID_SETTINGS_KEY,
        temp_path_key=ADJACENT_ID_TEMP_KEY,
        output_key=ADJACENT_ID_OUTPUT_KEY,
    )


def derive_adjacent_phase_rows(
    rows: pd.DataFrame,
    *,
    previous_phases: Mapping[str, str],
) -> pd.DataFrame:
    """Replace signed Pre-centered outcomes with signed adjacent-Phase changes."""

    required = {"Phase", "OutcomeScale", "Value"}
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise KeyError(f"Contact rows are missing adjacent-Phase columns: {missing}")
    phase_levels = list(previous_phases)
    observed_phases = set(rows["Phase"].dropna().astype(str))
    unexpected = observed_phases.difference(phase_levels)
    if unexpected:
        raise ValueError(
            f"Adjacent-Phase rows contain unregistered Phases: {sorted(unexpected)}"
        )

    output = rows.copy()
    grouping_columns = [
        column for column in output.columns if column not in {"Phase", "Value"}
    ]
    output["_AdjacentUnit"] = output.groupby(
        grouping_columns,
        sort=False,
        dropna=False,
        observed=True,
    ).ngroup()
    duplicate = output.duplicated(["_AdjacentUnit", "Phase"], keep=False)
    if bool(duplicate.any()):
        row = output.loc[duplicate, ["_AdjacentUnit", "Phase"]].iloc[0]
        raise ValueError(
            "Adjacent-Phase arithmetic requires one row per structural unit and "
            f"Phase: unit={row['_AdjacentUnit']}, Phase={row['Phase']}."
        )

    current = pd.to_numeric(output["Value"], errors="coerce")
    lookup = {
        (int(unit), str(phase)): float(value)
        for unit, phase, value in output.assign(Value=current)[
            ["_AdjacentUnit", "Phase", "Value"]
        ].itertuples(index=False, name=None)
    }
    previous_values: list[float] = []
    previous_labels: list[str] = []
    contrast_labels: list[str] = []
    for unit, phase_value in output[["_AdjacentUnit", "Phase"]].itertuples(
        index=False, name=None
    ):
        phase = str(phase_value)
        previous_phase = str(previous_phases[phase])
        previous_labels.append(previous_phase)
        contrast_labels.append(f"{phase}-{previous_phase}")
        if previous_phase == "Pre":
            previous_values.append(0.0)
        else:
            previous_values.append(lookup.get((int(unit), previous_phase), np.nan))

    previous = pd.Series(previous_values, index=output.index, dtype=float)
    output["CurrentPreCenteredValue"] = current
    output["PreviousPreCenteredValue"] = previous
    output["PreviousPhase"] = previous_labels
    output["ContrastLabel"] = contrast_labels
    output["OutcomeChangeBasis"] = ADJACENT_PHASE_BASIS
    output["Value"] = current - previous
    output["OutcomeScale"] = output["OutcomeScale"].astype(str).map(_adjacent_scale)
    output = output.drop(columns="_AdjacentUnit")
    sort_columns = [
        column
        for column in (
            "OutcomeDomain",
            "ConnectomeSource",
            "LiftMetric",
            "LiftPredictor",
            "ID",
            "ContactUnitID",
            "Phase",
        )
        if column in output.columns
    ]
    return output.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)


def _adjacent_outcome_sd(rows: pd.DataFrame) -> pd.DataFrame:
    base_key = [*OUTCOME_SD_KEY, "ID", "ContactUnitID", "Phase"]
    base = rows[[*base_key, "Value"]].copy()
    base["Value"] = pd.to_numeric(base["Value"], errors="coerce")
    value_counts = base.groupby(base_key, sort=False, dropna=False, observed=True)[
        "Value"
    ].nunique(dropna=True)
    if bool((value_counts > 1).any()):
        bad = value_counts.loc[value_counts > 1].index[0]
        raise ValueError(f"Structural expansion changed an adjacent outcome: {bad}")
    unique = base.drop_duplicates(base_key, keep="first")
    unique = unique.loc[np.isfinite(unique["Value"])].copy()
    output = (
        unique.groupby(
            OUTCOME_SD_KEY,
            sort=True,
            dropna=False,
            observed=True,
        )["Value"]
        .agg(ValueSD=lambda values: values.std(ddof=1), ValueN="count")
        .reset_index()
    )
    output["ValueSDStatus"] = np.where(
        np.isfinite(output["ValueSD"]) & output["ValueSD"].gt(0),
        "ok",
        "nonpositive_or_undefined_sd",
    )
    output["OutcomeChangeBasis"] = ADJACENT_PHASE_BASIS
    return output.sort_values(OUTCOME_SD_KEY, kind="mergesort").reset_index(drop=True)


def _field_excess_local_outcome_entries(
    config: StructuralConnectivityConfig,
) -> list[Mapping[str, Any]]:
    return list(
        config["inference"]["contact_lmm"].get(FIELD_EXCESS_LOCAL_OUTCOMES_KEY, [])
    )


def _load_field_excess_local_outcomes(
    config: StructuralConnectivityConfig,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    phases = set(config["outcomes"]["phases"])
    polarities = set(config["outcomes"]["polarities"])
    laterality = str(config["outcomes"]["laterality"])
    side_map = {
        subject.subject_id: subject.stimulation_side for subject in config.subjects
    }
    for entry in _field_excess_local_outcome_entries(config):
        path = Path(str(entry["path"]))
        metric = str(entry["metric"])
        feature_output = str(entry["feature_output"])
        table = _read_pickle(path)
        _validate_normalized_contract(
            table,
            path,
            domain="local",
            metric=metric,
            transform=str(entry["transform"]),
            transform_policy_status=str(entry["transform_policy_status"]),
        )
        if str(table.attrs.get("feature_output")) != feature_output:
            raise ValueError(
                "Normalized local outcome FeatureOutput does not match the "
                f"registered source: {path}"
            )
        required = {
            "ID",
            "StimSide",
            "Domain",
            "Metric",
            "FeatureOutput",
            "Polar",
            "Phase",
            "Band",
            "Region",
            "Lat",
            "Channel",
            "IncludeAggregate",
            "NormalizationStatus",
            "Value",
        }
        missing = sorted(required.difference(table.columns))
        if missing:
            raise KeyError(
                f"Normalized local outcome is missing columns in {path}: {missing}"
            )
        expected_side = table["ID"].map(side_map)
        selected = table.loc[
            table["ID"].isin(side_map)
            & expected_side.notna()
            & table["StimSide"].eq(expected_side)
            & table["Domain"].eq("local")
            & table["Metric"].eq(metric)
            & table["FeatureOutput"].eq(feature_output)
            & table["Polar"].isin(polarities)
            & table["Phase"].isin(phases)
            & table["Band"].isin(entry["bands"])
            & table["Region"].isin(["STN", "SNr"])
            & table["Lat"].eq(laterality)
            & table["IncludeAggregate"].eq(True)
            & table["NormalizationStatus"].eq("normalized"),
            [
                "ID",
                "StimSide",
                "Polar",
                "Phase",
                "Band",
                "Region",
                "Channel",
                "Metric",
                "FeatureOutput",
                "Value",
            ],
        ].copy()
        key = [
            "ID",
            "StimSide",
            "Polar",
            "Phase",
            "Band",
            "Region",
            "Channel",
            "Metric",
            "FeatureOutput",
        ]
        if selected.duplicated(key).any():
            raise ValueError(
                f"Normalized local outcome has duplicate analysis rows: {path}"
            )
        selected["OutcomeScale"] = str(entry["outcome_scale"])
        frames.append(selected)
    if not frames:
        raise ValueError("No FieldExcess local outcomes are registered.")
    return (
        pd.concat(frames, ignore_index=True, sort=False)
        .sort_values(
            [
                "Metric",
                "FeatureOutput",
                "ID",
                "Polar",
                "Region",
                "Phase",
                "Band",
                "Channel",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def _field_excess_local_model_rows(
    config: StructuralConnectivityConfig,
) -> pd.DataFrame:
    outcomes = _load_field_excess_local_outcomes(config)
    predictors = _read_csv(config.contact_lmm_root / "lift_predictors.csv")
    required = {
        "OutcomeDomain",
        "ConnectomeSource",
        "LiftMetric",
        "LiftPredictor",
        "Region",
        "PairRegion",
        "ID",
        "ContactUnitID",
        "StimSide",
        "Channel",
        "ChannelPair",
        "X_raw",
        "SeedThresholdedMeanPeakE_Vm",
        "RegionThresholdedMeanPeakE_Vm",
        "PredictorStatus",
        "X_center",
        "X_scale",
        "X_z",
        "X_standardization_n",
        "XStandardizationStatus",
    }
    missing = sorted(required.difference(predictors.columns))
    if missing:
        raise KeyError(f"FieldExcess predictors are missing columns: {missing}")
    predictors = predictors.loc[
        predictors["OutcomeDomain"].eq("local")
        & predictors["LiftMetric"].eq(FIELD_EXCESS_METRIC)
        & predictors["LiftPredictor"].eq(predictors["Region"]),
        sorted(required),
    ].copy()
    predictor_key = ["ConnectomeSource", "ID", "StimSide", "Region", "Channel"]
    if predictors.duplicated(predictor_key).any():
        raise ValueError("FieldExcess local predictors contain duplicate units.")
    rows = outcomes.merge(
        predictors,
        on=["ID", "StimSide", "Region", "Channel"],
        how="left",
        validate="many_to_many",
    )
    if rows["ConnectomeSource"].isna().any():
        raise ValueError("A normalized local outcome lacks FieldExcess predictors.")
    for column in ("PairRegion", "ChannelPair"):
        rows[column] = rows[column].fillna("").astype(str)
    adjacent = derive_adjacent_phase_rows(
        rows,
        previous_phases=config["inference"]["contact_lmm"]["previous_phases"],
    )
    status = pd.Series("eligible", index=adjacent.index, dtype="string")
    status.loc[~np.isfinite(pd.to_numeric(adjacent["Value"], errors="coerce"))] = (
        "nonfinite_outcome"
    )
    bad_predictor = ~adjacent["PredictorStatus"].eq("eligible")
    status.loc[bad_predictor] = adjacent.loc[bad_predictor, "PredictorStatus"].astype(
        str
    )
    bad_standardization = adjacent["PredictorStatus"].eq("eligible") & ~adjacent[
        "XStandardizationStatus"
    ].eq("ok")
    status.loc[bad_standardization] = adjacent.loc[
        bad_standardization, "XStandardizationStatus"
    ].astype(str)
    status.loc[
        adjacent["PredictorStatus"].eq("eligible")
        & adjacent["XStandardizationStatus"].eq("ok")
        & ~np.isfinite(pd.to_numeric(adjacent["X_z"], errors="coerce"))
    ] = "nonfinite_standardized_lift"
    adjacent["RowEligibilityStatus"] = status
    return adjacent.sort_values(
        [
            "ConnectomeSource",
            "Metric",
            "FeatureOutput",
            "Polar",
            "Region",
            "Band",
            "ID",
            "ContactUnitID",
            "Phase",
        ],
        kind="mergesort",
    ).reset_index(drop=True)


def _field_excess_local_model_plan(
    config: StructuralConnectivityConfig,
    *,
    first_model_number: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    sources = [source.connectome_id for source in config.connectomes]
    for entry in _field_excess_local_outcome_entries(config):
        for source in sources:
            for polar in config["outcomes"]["polarities"]:
                for region in ("STN", "SNr"):
                    for band in entry["bands"]:
                        records.append(
                            {
                                "OutcomeDomain": "local",
                                "ConnectomeSource": source,
                                "LiftMetric": FIELD_EXCESS_METRIC,
                                "LiftPredictor": region,
                                "Polar": polar,
                                "Metric": str(entry["metric"]),
                                "FeatureOutput": str(entry["feature_output"]),
                                "OutcomeScale": _adjacent_scale(
                                    str(entry["outcome_scale"])
                                ),
                                "Region": region,
                                "PairRegion": "",
                                "Band": str(band),
                            }
                        )
    models = pd.DataFrame(records)
    models.insert(
        0,
        "ModelID",
        [
            f"LMM{number:06d}"
            for number in range(
                first_model_number,
                first_model_number + len(models),
            )
        ],
    )
    models["Formula"] = config["inference"]["contact_lmm"]["formula"]
    models["OutcomeChangeBasis"] = ADJACENT_PHASE_BASIS
    return models


def _field_excess_local_selector(
    table: pd.DataFrame,
    config: StructuralConnectivityConfig,
) -> pd.Series:
    pairs = {
        (str(entry["metric"]), str(entry["feature_output"]))
        for entry in _field_excess_local_outcome_entries(config)
    }
    observed_pairs = pd.Series(
        list(zip(table["Metric"].astype(str), table["FeatureOutput"].astype(str))),
        index=table.index,
    )
    return (
        table["OutcomeDomain"].eq("local")
        & table["LiftMetric"].eq(FIELD_EXCESS_METRIC)
        & observed_pairs.isin(pairs)
    )


def _next_model_number(models: pd.DataFrame) -> int:
    numbers: list[int] = []
    for model_id in models["ModelID"].astype(str):
        match = re.fullmatch(r"LMM(\d{6})", model_id)
        if match is None:
            raise ValueError(f"Unexpected contact-level ModelID: {model_id}")
        numbers.append(int(match.group(1)))
    return max(numbers, default=0) + 1


def _register_basis(
    tables: Iterable[pd.DataFrame],
) -> None:
    for table in tables:
        if "OutcomeChangeBasis" in table.columns:
            observed = set(table["OutcomeChangeBasis"].dropna().astype(str))
            if observed and observed != {ADJACENT_PHASE_BASIS}:
                raise ValueError(
                    f"Unexpected adjacent-Phase basis values: {sorted(observed)}"
                )
            table["OutcomeChangeBasis"] = ADJACENT_PHASE_BASIS
        else:
            table.insert(0, "OutcomeChangeBasis", ADJACENT_PHASE_BASIS)


def _family_references_field_excess_local_models(
    table: pd.DataFrame,
    *,
    model_ids: set[str],
    config: StructuralConnectivityConfig,
) -> pd.Series:
    pairs = {
        (str(entry["metric"]), str(entry["feature_output"]))
        for entry in _field_excess_local_outcome_entries(config)
    }
    selected: list[bool] = []
    for row in table.itertuples(index=False):
        try:
            identity = json.loads(str(row.FamilyKey))
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid contact-level multiplicity FamilyKey: {row.FamilyKey}"
            ) from error
        if str(row.ResultTable) == "phase_slopes":
            selected.append(str(identity.get("ModelID", "")) in model_ids)
            continue
        selected.append(
            str(identity.get("OutcomeDomain", "")) == "local"
            and str(identity.get("LiftMetric", "")) == FIELD_EXCESS_METRIC
            and (
                str(identity.get("Metric", "")),
                str(identity.get("FeatureOutput", "")),
            )
            in pairs
        )
    return pd.Series(selected, index=table.index, dtype=bool)


def _trash_registered_paths(paths: Iterable[Path], *, root: Path) -> int:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return 0
    resolved_root = root.resolve()
    for path in existing:
        try:
            path.resolve().relative_to(resolved_root)
        except ValueError as error:
            raise ValueError(
                "Refusing to move a registered artifact outside the adjacent-Phase "
                f"root to Trash: {path}"
            ) from error
    trash = Path("/usr/bin/trash")
    if not trash.is_file():
        raise FileNotFoundError(f"macOS Trash command is unavailable: {trash}")
    subprocess.run([str(trash), *(str(path) for path in existing)], check=True)
    return len(existing)


def _append_field_excess_local_results(
    existing: Mapping[str, pd.DataFrame],
    processed: Mapping[str, pd.DataFrame],
    *,
    model_rows: pd.DataFrame,
    outcome_sd: pd.DataFrame,
    old_model_ids: set[str],
    config: StructuralConnectivityConfig,
) -> dict[str, pd.DataFrame]:
    model_tables = {
        "model_rows",
        "models",
        "model_fit",
        "fixed_effects",
        "omnibus_tests",
        "phase_slopes",
        "prediction_grid",
    }
    new_values: dict[str, pd.DataFrame] = {
        "model_rows": model_rows,
        **processed,
    }
    combined: dict[str, pd.DataFrame] = {}
    for name in model_tables:
        retained = (
            existing[name]
            .loc[~existing[name]["ModelID"].astype(str).isin(old_model_ids)]
            .copy()
        )
        combined[name] = pd.concat(
            [retained, new_values[name]], ignore_index=True, sort=False
        )
    existing_sd = existing["outcome_sd"]
    sd_pairs = pd.Series(
        list(
            zip(
                existing_sd["Metric"].astype(str),
                existing_sd["FeatureOutput"].astype(str),
            )
        ),
        index=existing_sd.index,
    )
    registered_pairs = {
        (str(entry["metric"]), str(entry["feature_output"]))
        for entry in _field_excess_local_outcome_entries(config)
    }
    retained_sd = existing_sd.loc[
        ~(existing_sd["OutcomeDomain"].eq("local") & sd_pairs.isin(registered_pairs))
    ].copy()
    combined["outcome_sd"] = pd.concat(
        [retained_sd, outcome_sd], ignore_index=True, sort=False
    )
    family_selector = _family_references_field_excess_local_models(
        existing["multiplicity_families"],
        model_ids=old_model_ids,
        config=config,
    )
    combined["multiplicity_families"] = (
        pd.concat(
            [
                existing["multiplicity_families"].loc[~family_selector].copy(),
                processed["multiplicity_families"],
            ],
            ignore_index=True,
            sort=False,
        )
        .sort_values(["ResultTable", "OutcomeDomain", "FamilyKey"], kind="mergesort")
        .reset_index(drop=True)
    )
    combined["lift_predictors"] = existing["lift_predictors"].copy()
    combined["coverage"] = _coverage(combined["models"])
    _register_basis([combined["coverage"]])
    return combined


def _run_contact_lmm_adjacent_branch(
    config: StructuralConnectivityConfig,
    *,
    adjacent: StructuralConnectivityConfig,
    render_figures: bool,
    dry_run: bool,
    overwrite: bool,
) -> dict[str, Any]:
    planned_paths = [adjacent.contact_lmm_root / name for name in CONTACT_LMM_TABLES]
    existing = [path for path in planned_paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Adjacent-Phase contact-level output exists; pass --overwrite to "
            f"replace it: {existing[0]}"
        )

    local, network = _contact_lmm_input_rows(adjacent)
    pre_centered_rows = _prepare_long_rows(local, network)
    settings = adjacent["inference"]["contact_lmm"]
    adjacent_rows = derive_adjacent_phase_rows(
        pre_centered_rows,
        previous_phases=settings["previous_phases"],
    )
    predictors, standardized_rows = _standardize_predictors(adjacent_rows)
    model_plan = _planned_models(adjacent)
    model_plan["OutcomeChangeBasis"] = ADJACENT_PHASE_BASIS
    model_rows, models = _attach_models_and_support(
        standardized_rows,
        model_plan,
        adjacent,
    )
    outcome_sd = _adjacent_outcome_sd(adjacent_rows)
    heatmap_count = (
        _heatmap_output_count(
            _figure_specs(
                models.merge(
                    pd.DataFrame({"Phase": settings["phase_levels"]}),
                    how="cross",
                ),
                adjacent,
            )
        )
        if render_figures
        else 0
    )
    slope_count = len(models) * len(settings["phase_levels"]) if render_figures else 0
    summary: dict[str, Any] = {
        "status": "ready" if dry_run else "complete",
        "outcome_change_basis": ADJACENT_PHASE_BASIS,
        "predictor_metrics": list(settings["predictor_metrics"]),
        "predictor_units": int(len(predictors)),
        "model_rows": int(len(model_rows)),
        "planned_models": int(len(models)),
        "eligible_models": int(models["ModelEligibilityStatus"].eq("eligible").sum()),
        "planned_heatmap_figures": int(heatmap_count),
        "planned_slope_figures": int(slope_count),
        "planned_figures": int(heatmap_count + slope_count),
        "writes": 0,
    }
    if dry_run:
        return summary

    r_outputs = _run_r_models(model_rows, models, adjacent)
    processed = _postprocess(r_outputs, models, outcome_sd, adjacent)
    coverage = _coverage(processed["models"])
    figures = (
        _render_contact_lmm_figures(
            model_rows,
            processed["phase_slopes"],
            processed["prediction_grid"],
            adjacent,
            overwrite=overwrite,
        )
        if render_figures
        else pd.DataFrame(columns=EMPTY_FIGURE_COLUMNS)
    )
    _register_basis(
        [
            predictors,
            model_rows,
            *processed.values(),
            outcome_sd,
            coverage,
            figures,
        ]
    )
    output_tables = {
        "lift_predictors": predictors,
        "model_rows": model_rows,
        "models": processed["models"],
        "model_fit": processed["model_fit"],
        "fixed_effects": processed["fixed_effects"],
        "omnibus_tests": processed["omnibus_tests"],
        "phase_slopes": processed["phase_slopes"],
        "prediction_grid": processed["prediction_grid"],
        "outcome_sd": outcome_sd,
        "coverage": coverage,
        "multiplicity_families": processed["multiplicity_families"],
    }
    _write_outputs(
        output_tables,
        figures,
        adjacent,
        overwrite=overwrite,
        include_visualization=render_figures,
    )
    summary["fit_status_counts"] = (
        processed["models"]["FitStatus"]
        .fillna("missing")
        .value_counts()
        .sort_index()
        .to_dict()
    )
    summary["phase_slope_rows"] = int(len(processed["phase_slopes"]))
    summary["figure_rows"] = int(len(figures))
    summary["writes"] = int(len(CONTACT_LMM_TABLES) + len(figures))
    return summary


def run_contact_lmm_adjacent(
    config: StructuralConnectivityConfig,
    *,
    dry_run: bool,
    overwrite: bool,
) -> dict[str, Any]:
    """Fit S, S/R, and S - R against adjacent-Phase signed outcomes."""

    return _run_contact_lmm_adjacent_branch(
        config,
        adjacent=_adjacent_config(config),
        render_figures=True,
        dry_run=dry_run,
        overwrite=overwrite,
    )


def run_contact_lmm_adjacent_field_excess_heatmaps(
    config: StructuralConnectivityConfig,
    *,
    dry_run: bool,
    overwrite: bool,
) -> dict[str, Any]:
    """Render only the formal adjacent-Phase FieldExcess atlas-grid heatmaps."""

    return run_contact_lmm_atlas_grid_heatmaps(
        _adjacent_config(config),
        dry_run=dry_run,
        overwrite=overwrite,
    )


def run_contact_lmm_adjacent_field_excess_finder_figures(
    config: StructuralConnectivityConfig,
    *,
    dry_run: bool,
    overwrite: bool,
) -> dict[str, Any]:
    """Refresh the approved adjacent FieldExcess Finder figure subset."""

    return run_contact_lmm_field_excess_finder_figures(
        _adjacent_config(config),
        dry_run=dry_run,
        overwrite=overwrite,
    )


def run_contact_lmm_adjacent_field_excess_finder_fits(
    config: StructuralConnectivityConfig,
    *,
    dry_run: bool,
    overwrite: bool,
) -> dict[str, Any]:
    """Refresh only approved adjacent FieldExcess Finder fit PDFs."""

    return run_contact_lmm_field_excess_finder_figures(
        _adjacent_config(config),
        dry_run=dry_run,
        overwrite=overwrite,
        refresh_heatmaps=False,
    )


def run_contact_lmm_adjacent_field_excess_atlas_grid_fits(
    config: StructuralConnectivityConfig,
    *,
    dry_run: bool,
    overwrite: bool,
) -> dict[str, Any]:
    """Render the six approved adjacent-Phase combined-atlas fits."""

    return run_contact_lmm_field_excess_atlas_grid_fits(
        _adjacent_config(config),
        dry_run=dry_run,
        overwrite=overwrite,
    )


def run_contact_lmm_adjacent_field_excess_local(
    config: StructuralConnectivityConfig,
    *,
    dry_run: bool,
    overwrite: bool,
) -> dict[str, Any]:
    """Append registered local outcomes to formal adjacent FieldExcess models."""

    adjacent = _adjacent_config(config)
    root = adjacent.contact_lmm_root
    paths = {name.removesuffix(".csv"): root / name for name in CONTACT_LMM_TABLES}
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "The completed adjacent-Phase branch is required before the "
            f"FieldExcess local expansion: {missing[0]}"
        )
    existing = {name: _read_csv(path) for name, path in paths.items()}
    old_selector = _field_excess_local_selector(existing["models"], adjacent)
    old_model_ids = set(existing["models"].loc[old_selector, "ModelID"].astype(str))
    retained_models = existing["models"].loc[~old_selector].copy()
    pre_model_rows = _field_excess_local_model_rows(adjacent)
    model_plan = _field_excess_local_model_plan(
        adjacent,
        first_model_number=_next_model_number(retained_models),
    )
    model_rows, models = _attach_models_and_support(
        pre_model_rows,
        model_plan,
        adjacent,
    )
    outcome_sd = _adjacent_outcome_sd(pre_model_rows)
    summary: dict[str, Any] = {
        "status": "ready" if dry_run else "complete",
        "outcome_change_basis": ADJACENT_PHASE_BASIS,
        "lift_metric": FIELD_EXCESS_METRIC,
        "additional_outcomes": [
            f"{entry['metric']}:{entry['feature_output']}"
            for entry in _field_excess_local_outcome_entries(adjacent)
        ],
        "replaced_previous_models": int(len(old_model_ids)),
        "planned_models": int(len(models)),
        "eligible_models": int(models["ModelEligibilityStatus"].eq("eligible").sum()),
        "model_rows": int(len(model_rows)),
        "planned_phase_slopes": int(len(models) * 3),
        "planned_slope_figures": int(
            len(models) * len(adjacent["inference"]["contact_lmm"]["phase_levels"])
        ),
        "planned_atlas_grid_heatmaps": 10,
        "refit_existing_models": 0,
        "writes": 0,
        "trashed_artifacts": 0,
    }
    if dry_run:
        return summary
    if not overwrite:
        raise FileExistsError(
            "The FieldExcess local expansion updates the completed adjacent-Phase "
            "branch; pass --overwrite to replace only its dependent artifacts."
        )

    r_outputs = _run_r_models(model_rows, models, adjacent)
    processed = _postprocess(r_outputs, models, outcome_sd, adjacent)
    _register_basis([model_rows, *processed.values(), outcome_sd])
    combined = _append_field_excess_local_results(
        existing,
        processed,
        model_rows=model_rows,
        outcome_sd=outcome_sd,
        old_model_ids=old_model_ids,
        config=adjacent,
    )

    current_figures = existing["figures"]
    old_slope_selector = current_figures["FigureType"].eq("slope") & (
        current_figures["ModelID"].astype(str).isin(old_model_ids)
    )
    heatmap_selector = current_figures["FigureType"].eq("heatmap") & (
        current_figures["LiftMetric"].eq(FIELD_EXCESS_METRIC)
    )
    retained_figures = current_figures.loc[
        ~(old_slope_selector | heatmap_selector)
    ].copy()
    trashed = _trash_registered_paths(
        [
            Path(path)
            for path in current_figures.loc[old_slope_selector, "SourcePath"].dropna()
        ],
        root=root,
    )
    slope_figures = _plot_slope_figures(
        model_rows,
        processed["prediction_grid"],
        processed["phase_slopes"],
        adjacent,
        overwrite=False,
    )
    trashed += _trash_registered_paths(
        [
            Path(path)
            for path in current_figures.loc[heatmap_selector, "SourcePath"].dropna()
        ],
        root=root,
    )
    heatmaps = _plot_heatmaps(
        combined["phase_slopes"],
        adjacent,
        overwrite=False,
        lift_metrics={FIELD_EXCESS_METRIC},
    )
    _register_basis([slope_figures, heatmaps])
    figures = pd.concat(
        [retained_figures, heatmaps, slope_figures],
        ignore_index=True,
        sort=False,
    )
    if figures["FigureID"].duplicated().any():
        raise ValueError("Adjacent-Phase figure IDs must be unique after expansion.")
    if figures["PublishedRelativePath"].duplicated().any():
        raise ValueError(
            "Adjacent-Phase publication paths must be unique after expansion."
        )

    output_tables = {
        "lift_predictors": combined["lift_predictors"],
        "model_rows": combined["model_rows"],
        "models": combined["models"],
        "model_fit": combined["model_fit"],
        "fixed_effects": combined["fixed_effects"],
        "omnibus_tests": combined["omnibus_tests"],
        "phase_slopes": combined["phase_slopes"],
        "prediction_grid": combined["prediction_grid"],
        "outcome_sd": combined["outcome_sd"],
        "coverage": combined["coverage"],
        "multiplicity_families": combined["multiplicity_families"],
    }
    trashed += _trash_registered_paths(paths.values(), root=root)
    _write_outputs(
        output_tables,
        figures,
        adjacent,
        overwrite=False,
        include_visualization=True,
    )
    summary["fit_status_counts"] = (
        processed["models"]["FitStatus"]
        .fillna("missing")
        .value_counts()
        .sort_index()
        .to_dict()
    )
    summary["phase_slope_rows"] = int(len(processed["phase_slopes"]))
    summary["slope_figures"] = int(len(slope_figures))
    summary["atlas_grid_heatmaps"] = int(len(heatmaps))
    summary["total_models"] = int(len(combined["models"]))
    summary["total_phase_slopes"] = int(len(combined["phase_slopes"]))
    summary["total_figures"] = int(len(figures))
    summary["trashed_artifacts"] = int(trashed)
    summary["writes"] = int(
        len(CONTACT_LMM_TABLES) + len(slope_figures) + len(heatmaps)
    )
    return summary


def run_contact_lmm_adjacent_id_random_intercept(
    config: StructuralConnectivityConfig,
    *,
    dry_run: bool,
    overwrite: bool,
) -> dict[str, Any]:
    """Fit the numerical adjacent-Phase branch with only an ID intercept."""

    return _run_contact_lmm_adjacent_branch(
        config,
        adjacent=_adjacent_id_random_intercept_config(config),
        render_figures=False,
        dry_run=dry_run,
        overwrite=overwrite,
    )
