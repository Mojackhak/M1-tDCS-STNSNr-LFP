"""Align normalized LFP changes with separate programmed-intensity endpoints."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml

from lfp_viz.data import require_columns

from .config import (
    ClinicalCorrelationConfig,
    _mapping,
    _string,
    _validate_features,
    _validate_lfp_inputs,
)
from .correlation import _compute_endpoint_correlations, _holm_adjust
from .data import _normalize_ids, _numeric_column, load_lfp_predictors

ANALYSIS_ID = "lfp-programming-intensity"
MINIMUM_N = 4
ZERO_TOLERANCE = 1e-12
ENDPOINTS = (
    ("stn-only-stn", "STN", "STN", 3, 3),
    ("combined-stn", "STN+SNr", "STN", 6, 3),
    ("combined-snr", "STN+SNr", "SNr", 6, 3),
)
MODELS = {"M0": None, "M1": "UPDRSIII_MedOFF", "M2": "PDQ39_Total"}
CELL_GROUP = ["EndpointID", "Polar", "Lat"]
FEATURE_FIELDS = ["Domain", "Metric", "FeatureOutput", "Band", "RegionValue"]
FAMILY_FIELDS = ["Strategy", "EndpointID", "Model", "Polar", *FEATURE_FIELDS]


@dataclass(frozen=True)
class ProgrammingConfig:
    """Parsed inputs and simulation settings for the fixed analysis contract."""

    path: Path
    paths: dict[str, Path]
    lfp: ClinicalCorrelationConfig
    cv: tuple[float, ...]
    iterations: int
    seed: int
    strategy: str = "region"

    @property
    def output_root(self) -> Path:
        return self.paths["output_root"]


@dataclass(frozen=True)
class ProgrammingInputs:
    """Validated sources, adjacent predictors, and one-patient-per-cell joins."""

    programming: pd.DataFrame
    predictors: pd.DataFrame
    predictor_metadata: pd.DataFrame
    model_input: pd.DataFrame
    registry: pd.DataFrame
    manifest: pd.DataFrame
    strategy: str = "region"
    matching_tables: dict[str, pd.DataFrame] = field(default_factory=dict)


def load_programming_config(path: Path | str) -> ProgrammingConfig:
    """Read only configuration sections consumed by this analysis."""

    path = Path(path).resolve()
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict) or set(data) != {
        "lfp_config",
        "strategy",
        "paths",
        "jitter",
    }:
        raise ValueError(
            "Programming config requires lfp_config, strategy, paths, and jitter."
        )
    strategy = _string(data, "strategy")
    if strategy not in {"region", "contact-matched"}:
        raise ValueError("Programming strategy must be region or contact-matched.")
    raw_paths = _mapping(data, "paths")
    expected_paths = {
        "programming_parameters",
        "baseline_covariates",
        "tdcs_execution_status",
        "output_root",
    }
    if strategy == "contact-matched":
        expected_paths.update(
            {"contact_pair_regions", "local_root", "connectivity_root"}
        )
    if set(raw_paths) != expected_paths:
        raise ValueError(
            "Programming input/output path keys do not match the contract."
        )
    paths = {
        key: (path.parent / Path(_string(raw_paths, key)).expanduser()).resolve()
        for key in raw_paths
    }
    lfp_path = (path.parent / _string(data, "lfp_config")).resolve()
    with lfp_path.open(encoding="utf-8") as stream:
        lfp_source = yaml.safe_load(stream)
    source_paths = _mapping(lfp_source, "paths")
    lfp_data = {
        "features": dict(_mapping(lfp_source, "features")),
        "inputs": {"lfp": dict(_mapping(_mapping(lfp_source, "inputs"), "lfp"))},
        "paths": {
            key: str((lfp_path.parent / _string(source_paths, key)).resolve())
            for key in ("feature_manifest", "lfp_root")
        },
    }
    _validate_features(lfp_data)
    _validate_lfp_inputs(lfp_data["inputs"]["lfp"], adjacent_phase=True)
    jitter = _mapping(data, "jitter")
    if set(jitter) != {"cv", "iterations", "seed"}:
        raise ValueError("Jitter requires cv, iterations, and seed.")
    raw_cv = jitter["cv"]
    if not isinstance(raw_cv, list) or not raw_cv:
        raise ValueError("Jitter cv must be a nonempty list.")
    cv = tuple(float(value) for value in raw_cv)
    if any(not np.isfinite(value) or value < 0 for value in cv) or len(set(cv)) != len(
        cv
    ):
        raise ValueError("Jitter CVs must be distinct, finite, nonnegative values.")
    iterations, seed = jitter["iterations"], jitter["seed"]
    if (
        type(iterations) is not int
        or iterations < 1
        or type(seed) is not int
        or seed < 0
    ):
        raise ValueError(
            "Jitter requires positive integer iterations and a nonnegative seed."
        )
    return ProgrammingConfig(
        path,
        paths,
        ClinicalCorrelationConfig(lfp_path, lfp_data),
        cv,
        iterations,
        seed,
        strategy,
    )


def _read_source(
    path: Path, required: set[str], key: list[str], input_type: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    table = pd.read_csv(path, float_precision="round_trip")
    require_columns(table, required, str(path))
    if table["ID"].isna().any():
        raise ValueError(f"Missing patient ID in {path}.")
    table["ID"] = _normalize_ids(table["ID"])
    if table.duplicated(key).any():
        raise ValueError(f"Duplicate {key} in {path}.")
    return table, {
        "InputType": input_type,
        "Path": str(path),
        "Rows": len(table),
        "Columns": len(table.columns),
    }


def _boolean(values: pd.Series, label: str) -> pd.Series:
    parsed = values.astype(str).str.lower().map({"true": True, "false": False})
    if parsed.isna().any():
        raise ValueError(f"{label} must contain only true or false.")
    return parsed.astype(bool)


def _endpoint_table() -> pd.DataFrame:
    return pd.DataFrame(
        ENDPOINTS,
        columns=[
            "EndpointID",
            "Protocol",
            "Target",
            "AssessmentMonthAfterDBSActivation",
            "MonthsOnProtocol",
        ],
    )


def build_registry(metadata: pd.DataFrame, strategy: str = "region") -> pd.DataFrame:
    """Register all planned cells, including those without complete patients."""

    frames = []
    for endpoint in _endpoint_table().to_dict("records"):
        selected = metadata.loc[
            metadata["Domain"].eq("connectivity")
            | metadata["RegionValue"].eq(endpoint["Target"])
        ]
        for model, covariate in MODELS.items():
            frame = selected.copy()
            for key, value in endpoint.items():
                frame[key] = value
            frame["Model"] = model
            frame["Strategy"] = strategy
            frame["Method"] = "spearman" if covariate is None else "partial_spearman"
            frame["Covariate"] = covariate or "none"
            frame["CorrelationID"] = (
                endpoint["EndpointID"] + "__" + model + "__" + frame["PredictorID"]
            )
            frame["PairRegion"] = frame["RegionValue"].where(
                frame["Domain"].eq("connectivity")
            )
            frame["PairDirection"] = np.where(
                frame["Domain"].eq("connectivity"),
                np.where(
                    frame["Metric"].isin(["psi", "trgc"]), "directed", "undirected"
                ),
                "not_applicable",
            )
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_programming_inputs(config: ProgrammingConfig) -> ProgrammingInputs:
    """Parse each source once and validate keys at the source/join boundaries."""

    manifest = []
    program_required = {
        "ID",
        "Protocol",
        "Target",
        "Side",
        "Contact",
        "VoltageV",
        "PulseWidthUs",
        "FrequencyHz",
        "Available",
        "AssessmentMonthAfterDBSActivation",
        "MonthsOnProtocol",
    }
    programming, item = _read_source(
        config.paths["programming_parameters"],
        program_required,
        ["ID", "Protocol", "Target", "Side"],
        "programming_parameters",
    )
    manifest.append(item)
    programming["Available"] = _boolean(programming["Available"], "Available")
    available = programming["Available"]
    for column in ("VoltageV", "PulseWidthUs", "FrequencyHz", "Contact"):
        programming[column] = _numeric_column(
            programming, column, "Programming parameters"
        )
        values = programming.loc[available, column]
        invalid_range = values.lt(0) if column == "Contact" else values.le(0)
        if not np.isfinite(values).all() or invalid_range.any():
            raise ValueError(f"Available programming rows contain invalid {column}.")
    contacts = programming.loc[available, "Contact"]
    if (contacts % 1 != 0).any():
        raise ValueError("Physical contact identifiers must be integers.")
    programming["Contact"] = programming["Contact"].astype("Int64")
    if not programming.loc[available, "Side"].isin(["L", "R"]).all():
        raise ValueError("Available programming sides must be L or R.")
    programming = programming.loc[available].copy()
    endpoint_lookup = _endpoint_table()
    programming = programming.merge(
        endpoint_lookup,
        on=["Protocol", "Target"],
        how="left",
        validate="many_to_one",
        suffixes=("", "Expected"),
    )
    if programming["EndpointID"].isna().any():
        raise ValueError(
            "Programming contains an unregistered protocol/target endpoint."
        )
    for column in ("AssessmentMonthAfterDBSActivation", "MonthsOnProtocol"):
        if not programming[column].eq(programming[column + "Expected"]).all():
            raise ValueError(f"Programming endpoint timing differs in {column}.")
        programming = programming.drop(columns=column + "Expected")
    programming["ProgrammingIntensity"] = (
        programming["VoltageV"] ** 2
        * programming["PulseWidthUs"]
        * programming["FrequencyHz"]
    ).where(programming["Available"])
    programming["PhysicalContact"] = (
        programming["ID"]
        + "|"
        + programming["Side"]
        + "|"
        + programming["Contact"].astype("string")
    ).where(programming["Available"])
    programming["ProgrammingSourcePath"] = str(config.paths["programming_parameters"])

    baseline, item = _read_source(
        config.paths["baseline_covariates"],
        {"ID", "UPDRSIII_MedOFF", "PDQ39_Total"},
        ["ID"],
        "baseline_covariates",
    )
    manifest.append(item)
    baseline = baseline[["ID", "UPDRSIII_MedOFF", "PDQ39_Total"]].copy()
    for column in MODELS.values():
        if column is not None:
            baseline[column] = _numeric_column(baseline, column, "Baseline covariates")
            if np.isinf(baseline[column]).any():
                raise ValueError(f"Infinite baseline value in {column}.")
    baseline["BaselineSourcePath"] = str(config.paths["baseline_covariates"])
    status, item = _read_source(
        config.paths["tdcs_execution_status"],
        {"ID", "StimSide", "AnodeCenteredCompleted", "CathodeCenteredCompleted"},
        ["ID"],
        "tdcs_execution_status",
    )
    manifest.append(item)
    if not status["StimSide"].isin(["L", "R"]).all():
        raise ValueError("tDCS stimulation sides must be L or R.")
    for column in ("AnodeCenteredCompleted", "CathodeCenteredCompleted"):
        status[column] = _boolean(status[column], column)

    matching_tables = {}
    if config.strategy == "contact-matched":
        from .programming_contact import load_contact_predictors

        metadata, predictors, lfp_manifest, matching_tables = load_contact_predictors(
            config, programming
        )
    else:
        metadata, predictors, lfp_manifest = load_lfp_predictors(
            config.lfp, retain_provenance=True
        )
    manifest.extend(lfp_manifest)
    status = status[
        ["ID", "StimSide", "AnodeCenteredCompleted", "CathodeCenteredCompleted"]
    ]
    predictors = predictors.merge(
        status, on="ID", how="left", validate="many_to_one", suffixes=("", "Execution")
    )
    if predictors["StimSideExecution"].isna().any():
        raise ValueError("An LFP patient has no tDCS execution record.")
    if not predictors["StimSide"].eq(predictors["StimSideExecution"]).all():
        raise ValueError("LFP and execution records disagree on stimulation side.")
    completed = np.where(
        predictors["Polar"].eq("Anodal"),
        predictors["AnodeCenteredCompleted"],
        predictors["CathodeCenteredCompleted"],
    )
    predictors = (
        predictors.loc[completed]
        .drop(
            columns=[
                "StimSideExecution",
                "AnodeCenteredCompleted",
                "CathodeCenteredCompleted",
            ]
        )
        .copy()
    )
    predictors["Side"] = np.where(
        predictors["Lat"].eq("Ipsi"),
        predictors["StimSide"],
        predictors["StimSide"].map({"L": "R", "R": "L"}),
    )
    predictors["PairRegion"] = predictors["RegionValue"].where(
        predictors["Domain"].eq("connectivity")
    )
    conn = predictors["Domain"].eq("connectivity")
    expected_direction = np.where(
        predictors["Metric"].isin(["psi", "trgc"]), "directed", "undirected"
    )
    if not predictors.loc[conn, "PairDirection"].eq(expected_direction[conn]).all():
        raise ValueError(
            "Connectivity pair direction differs from the registered direction."
        )
    predictors["PairDirection"] = predictors["PairDirection"].fillna("not_applicable")
    predictors["ExecutionSourcePath"] = str(config.paths["tdcs_execution_status"])
    joined = []
    for endpoint_id, _, target, _, _ in ENDPOINTS:
        doses = programming.loc[
            programming["EndpointID"].eq(endpoint_id) & programming["Available"]
        ]
        if config.strategy == "contact-matched":
            selected = predictors.loc[predictors["EndpointID"].eq(endpoint_id)]
            keys = ["EndpointID", "ID", "Side"]
        else:
            selected = predictors.loc[conn | predictors["RegionValue"].eq(target)]
            keys = ["ID", "Side"]
        frame = selected.merge(doses, on=keys, how="inner", validate="many_to_one")
        joined.append(frame)
    model_input = pd.concat(joined, ignore_index=True).merge(
        baseline, on="ID", how="left", validate="many_to_one"
    )
    if model_input.duplicated(["EndpointID", "PredictorID", "ID"]).any():
        raise ValueError(
            "Model input contains more than one observation per patient cell."
        )
    model_input = model_input.sort_values(
        ["EndpointID", "PredictorID", "ID"]
    ).reset_index(drop=True)
    valid_keys = (
        predictors[["EndpointID", "ID", "PredictorID"]] if matching_tables else None
    )
    for name in ("local_channel_deltas", "connectivity_edge_deltas"):
        if name in matching_tables:
            matching_tables[name] = matching_tables[name].merge(
                valid_keys,
                on=["EndpointID", "ID", "PredictorID"],
                validate="many_to_one",
            )
    manifest_table = pd.DataFrame(manifest)
    for table in (
        programming,
        predictors,
        model_input,
        manifest_table,
        *matching_tables.values(),
    ):
        table["Strategy"] = config.strategy
    return ProgrammingInputs(
        programming.sort_values(["EndpointID", "ID", "Side"]).reset_index(drop=True),
        predictors.sort_values(["PredictorID", "ID"]).reset_index(drop=True),
        metadata,
        model_input,
        build_registry(metadata, config.strategy),
        manifest_table,
        config.strategy,
        matching_tables,
    )


def cell_arrays(
    inputs: ProgrammingInputs, cells: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Select one endpoint/polarity/laterality group of already validated inputs."""

    first = cells.iloc[0]
    selected = inputs.model_input
    mask = np.logical_and.reduce([selected[key].eq(first[key]) for key in CELL_GROUP])
    selected = selected.loc[mask]
    patients = selected.drop_duplicates("ID").set_index("ID").sort_index()
    matrix = selected.pivot(index="ID", columns="PredictorID", values="Value").reindex(
        index=patients.index, columns=cells["PredictorID"]
    )
    return patients.index.to_numpy(), matrix.to_numpy(dtype=float), patients


def fixed_six_holm(values: pd.Series) -> pd.Series:
    """Adjust the six pre-registered positions without shrinking missing families."""

    adjusted = _holm_adjust(values.fillna(1.0))
    return adjusted.where(values.notna())


def compute_programming_correlations(
    inputs: ProgrammingInputs,
    *,
    models: Sequence[str] = tuple(MODELS),
    endpoints: Sequence[str] = tuple(item[0] for item in ENDPOINTS),
) -> pd.DataFrame:
    """Compute requested endpoint/model components; raw p values remain internal."""

    if not set(models).issubset(MODELS) or not set(endpoints).issubset(
        item[0] for item in ENDPOINTS
    ):
        raise ValueError("Unknown programming model or endpoint selection.")
    registry = inputs.registry.loc[
        inputs.registry["Model"].isin(models)
        & inputs.registry["EndpointID"].isin(endpoints)
    ]
    results = []
    for (_, model, _, _), cells in registry.groupby(
        ["EndpointID", "Model", "Polar", "Lat"], sort=False, observed=True
    ):
        cells = cells.reset_index(drop=True)
        ids, x, patients = cell_arrays(inputs, cells)
        y = patients["ProgrammingIntensity"].to_numpy(dtype=float)
        covariate = MODELS[model]
        baseline = (
            np.full(len(y), np.nan)
            if covariate is None
            else patients[covariate].to_numpy(dtype=float)
        )
        estimates, _ = _compute_endpoint_correlations(
            x,
            y,
            baseline,
            ids,
            method="spearman" if covariate is None else "partial_spearman",
            minimum_n=MINIMUM_N,
        )
        if covariate is not None:
            for column in estimates.index[estimates["Status"].eq("constant_input")]:
                complete = (
                    np.isfinite(x[:, column]) & np.isfinite(y) & np.isfinite(baseline)
                )
                if (
                    np.unique(x[complete, column]).size > 1
                    and np.unique(y[complete]).size > 1
                ):
                    estimates.loc[column, "Status"] = "covariate_collinear"
                    estimates.loc[column, "Message"] = (
                        "Ranked predictor or outcome is collinear with the covariate."
                    )
        estimates["rho"] = (
            estimates["rho"]
            .clip(-1.0, 1.0)
            .mask(estimates["rho"].abs() < ZERO_TOLERANCE, 0.0)
        )
        results.append(pd.concat([cells, estimates], axis=1))
    result = pd.concat(results, ignore_index=True)
    families = result.groupby(FAMILY_FIELDS, sort=False, observed=True, dropna=False)
    if not families.size().eq(6).all():
        raise ValueError(
            "Every programming Holm family must retain six planned positions."
        )
    result["p_holm"] = families["p_raw"].transform(fixed_six_holm)
    result = result.drop(columns="p_raw")
    result["CorrectionFamily"] = result[FAMILY_FIELDS].astype(str).agg("|".join, axis=1)
    result["CorrectionFamilySize"] = 6
    result["MinimumN"] = MINIMUM_N
    result["SignificanceBasis"] = "p_holm"
    result["PMethod"] = np.where(
        result["Model"].eq("M0"),
        "exhaustive_patient_permutation",
        "freedman_lane_approximate_exhaustive",
    )
    result.insert(0, "AnalysisID", ANALYSIS_ID)
    return result.sort_values("CorrelationID").reset_index(drop=True)
