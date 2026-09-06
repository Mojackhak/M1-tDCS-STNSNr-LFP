"""Fit the numerical adjacent-Phase cortical-allocation mixed-model branch."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import StructuralConnectivityConfig
from .contact_lmm import _run_r_models
from .cortical_allocation_lmm import (
    CORTICAL_ALLOCATION_OUTPUTS,
    _output_manifest,
    _registered_local_outcomes,
    _write_outputs,
    attach_models_and_support,
    compute_outcome_sd,
    expand_model_rows,
    planned_models,
    postprocess_models,
    prepare_lfp_rows,
    tests_manifest,
)


ADJACENT_PHASE_BASIS = "adjacent_phase"
ADJACENT_SETTINGS_KEY = "cortical_allocation_lmm_adjacent"
ADJACENT_TEMP_KEY = "cortical_allocation_lmm_adjacent_temp_root"
ADJACENT_OUTPUT_KEY = "cortical_allocation_lmm_adjacent_root"

ADJACENT_UNIT_KEY = [
    "ID",
    "StimSide",
    "OutcomeDomain",
    "Metric",
    "FeatureOutput",
    "OutcomeScale",
    "Polar",
    "Band",
    "Lat",
    "Region",
    "PairRegion",
    "ContactUnitID",
]


def _adjacent_scale(value: str) -> str:
    suffix = "_pre_centered"
    if not value.endswith(suffix):
        raise ValueError(
            "Adjacent-Phase outcomes require a registered Pre-centered scale: "
            f"{value}"
        )
    return f"{value.removesuffix(suffix)}_adjacent_phase"


def _adjacent_config(
    config: StructuralConnectivityConfig,
) -> StructuralConnectivityConfig:
    data = deepcopy(config.data)
    settings = deepcopy(data["inference"][ADJACENT_SETTINGS_KEY])
    data["analysis"]["id"] = settings["analysis_id"]
    data["inference"]["cortical_allocation_lmm"] = settings
    data["paths"]["cortical_allocation_lmm_temp_root"] = data["paths"][
        ADJACENT_TEMP_KEY
    ]
    data["outputs"]["cortical_allocation_lmm_root"] = data["outputs"][
        ADJACENT_OUTPUT_KEY
    ]
    data["outcomes"]["local_scales"] = {
        metric: _adjacent_scale(str(scale))
        for metric, scale in data["outcomes"]["local_scales"].items()
    }
    data["outcomes"]["connectivity_scales"] = {
        metric: _adjacent_scale(str(scale))
        for metric, scale in data["outcomes"]["connectivity_scales"].items()
    }
    for entry in data["inference"]["cortical_allocation_lmm"].get(
        "additional_local_outcomes", []
    ):
        entry["outcome_scale"] = _adjacent_scale(str(entry["outcome_scale"]))
    return StructuralConnectivityConfig(
        path=config.path,
        data=data,
        subjects=config.subjects,
        connectomes=config.connectomes,
    )


def regional_allocation_source_path(
    config: StructuralConnectivityConfig,
) -> Path:
    return (
        config.cortical_allocation_lmm_root
        / CORTICAL_ALLOCATION_OUTPUTS[
            "cortical_allocation_regional_field.csv"
        ]
    )


def load_regional_field_allocation(
    config: StructuralConnectivityConfig,
) -> pd.DataFrame:
    """Read and validate the completed Pre-centered regional allocation table."""

    path = regional_allocation_source_path(config)
    if not path.is_file():
        raise FileNotFoundError(
            "The completed cortical regional-allocation table is absent: "
            f"{path}"
        )
    table = pd.read_csv(path, low_memory=False)
    required = {
        "AnalysisID",
        "CandidateSetID",
        "AtlasName",
        "ID",
        "StimSide",
        "CorticalRegion",
        "VoxelVolumeWeightedDose_Vm",
        "CandidateDoseSum_Vm",
        "RelativeAllocation",
        "AllocationStatus",
    }
    missing = sorted(required.difference(table.columns))
    if missing:
        raise KeyError(
            f"Regional field allocation is missing columns: {missing}"
        )

    settings = config["inference"]["cortical_allocation_lmm"]
    if set(table["AnalysisID"].astype(str)) != {settings["analysis_id"]}:
        raise ValueError("Regional allocation has an unexpected source analysis ID.")
    if set(table["CandidateSetID"].astype(str)) != {
        settings["candidate_set_id"]
    }:
        raise ValueError("Regional allocation has an unexpected candidate set.")
    if set(table["AtlasName"].astype(str)) != {settings["atlas_name"]}:
        raise ValueError("Regional allocation has an unexpected atlas name.")

    expected = {
        (subject.subject_id, subject.stimulation_side, region)
        for subject in config.subjects
        for region in settings["regions"]
    }
    observed = set(
        table[["ID", "StimSide", "CorticalRegion"]].itertuples(
            index=False,
            name=None,
        )
    )
    if observed != expected or len(table) != len(expected):
        raise ValueError(
            "Regional allocation does not contain exactly one row per registered "
            "patient, stimulation side, and cortical region."
        )
    if table.duplicated(["ID", "StimSide", "CorticalRegion"]).any():
        raise ValueError("Regional allocation contains duplicate patient-region rows.")

    relative = pd.to_numeric(table["RelativeAllocation"], errors="coerce")
    for keys, group in table.assign(_Relative=relative).groupby(
        ["ID", "StimSide"],
        sort=False,
        observed=True,
    ):
        statuses = set(group["AllocationStatus"].astype(str))
        if statuses == {"ok"}:
            values = group["_Relative"].to_numpy(dtype=float)
            if not np.isfinite(values).all() or not np.isclose(
                values.sum(),
                1.0,
                rtol=0,
                atol=1e-12,
            ):
                raise ValueError(
                    "Finite regional allocations do not sum to one for "
                    f"{keys}."
                )
        elif statuses != {"undefined_zero_candidate_dose"}:
            raise ValueError(
                f"Regional allocation has an unexpected status for {keys}: "
                f"{sorted(statuses)}"
            )
    return table.sort_values(
        ["ID", "CorticalRegion"],
        kind="mergesort",
    ).reset_index(drop=True)


def derive_adjacent_lfp_rows(
    rows: pd.DataFrame,
    *,
    previous_phases: Mapping[str, str],
) -> pd.DataFrame:
    """Derive signed adjacent changes from unique Pre-centered contact rows."""

    required = {*ADJACENT_UNIT_KEY, "Phase", "Value"}
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise KeyError(f"LFP rows are missing adjacent-Phase columns: {missing}")
    phase_levels = list(previous_phases)
    observed_phases = set(rows["Phase"].dropna().astype(str))
    unexpected = observed_phases.difference(phase_levels)
    if unexpected:
        raise ValueError(
            f"Adjacent-Phase rows contain unregistered Phases: {sorted(unexpected)}"
        )
    if rows.duplicated([*ADJACENT_UNIT_KEY, "Phase"]).any():
        raise ValueError(
            "Adjacent-Phase arithmetic requires one row per contact unit and Phase."
        )

    output = rows.copy()
    current = pd.to_numeric(output["Value"], errors="coerce")
    lookup = {
        (*keys, str(phase)): float(value)
        for (*keys, phase, value) in output.assign(_Current=current)[
            [*ADJACENT_UNIT_KEY, "Phase", "_Current"]
        ].itertuples(index=False, name=None)
    }
    previous_values: list[float] = []
    previous_labels: list[str] = []
    contrast_labels: list[str] = []
    for values in output[[*ADJACENT_UNIT_KEY, "Phase"]].itertuples(
        index=False,
        name=None,
    ):
        *keys, phase_value = values
        phase = str(phase_value)
        previous_phase = str(previous_phases[phase])
        previous_labels.append(previous_phase)
        contrast_labels.append(f"{phase}-{previous_phase}")
        previous_values.append(
            0.0
            if previous_phase == "Pre"
            else lookup.get((*keys, previous_phase), np.nan)
        )

    previous = pd.Series(previous_values, index=output.index, dtype=float)
    output["CurrentPreCenteredValue"] = current
    output["PreviousPreCenteredValue"] = previous
    output["PreviousPhase"] = previous_labels
    output["ContrastLabel"] = contrast_labels
    output["OutcomeChangeBasis"] = ADJACENT_PHASE_BASIS
    output["Value"] = current - previous
    output["OutcomeScale"] = output["OutcomeScale"].astype(str).map(
        _adjacent_scale
    )
    return output.sort_values(
        [
            "OutcomeDomain",
            "Metric",
            "ID",
            "ContactUnitID",
            "Phase",
        ],
        kind="mergesort",
    ).reset_index(drop=True)


def adjacent_input_manifest(
    config: StructuralConnectivityConfig,
    adjacent_config: StructuralConnectivityConfig,
) -> pd.DataFrame:
    """Record only the direct numerical inputs of the adjacent branch."""

    analysis_id = adjacent_config["inference"]["cortical_allocation_lmm"][
        "analysis_id"
    ]
    records: list[dict[str, Any]] = []

    def append(input_type: str, path: Path, role: str) -> None:
        records.append(
            {
                "AnalysisID": analysis_id,
                "OutcomeChangeBasis": ADJACENT_PHASE_BASIS,
                "InputType": input_type,
                "InputRole": role,
                "RequiredForExecution": True,
                "ID": "",
                "CorticalRegion": "",
                "Path": str(path),
                "Exists": path.is_file(),
            }
        )

    append("configuration", config.path, "computational_input")
    append(
        "r_runner",
        Path(__file__).parent / "r" / "run_contact_lmm.R",
        "computational_input",
    )
    append(
        "regional_field_allocation",
        regional_allocation_source_path(config),
        "reused_completed_input",
    )
    additional = config["inference"]["cortical_allocation_lmm_adjacent"][
        "additional_local_outcomes"
    ]
    for entry in _registered_local_outcomes(
        config,
        additional_local_outcomes=additional,
    ):
        append(
            f"normalized_local_lfp:{entry['metric']}:{entry['feature_output']}",
            Path(str(entry["path"])),
            "computational_input",
        )
    for metric in config["outcomes"]["connectivity_metrics"]:
        append(
            "normalized_connectivity_lfp",
            config.path_value("connectivity_lfp_root")
            / metric
            / "mean-scalar.pkl",
            "computational_input",
        )
    return pd.DataFrame(records)


def _add_basis(table: pd.DataFrame) -> pd.DataFrame:
    output = table.copy()
    if "OutcomeChangeBasis" in output.columns:
        output["OutcomeChangeBasis"] = ADJACENT_PHASE_BASIS
    else:
        output.insert(0, "OutcomeChangeBasis", ADJACENT_PHASE_BASIS)
    return output


def run_cortical_allocation_lmm_adjacent(
    config: StructuralConnectivityConfig,
    *,
    dry_run: bool,
    overwrite: bool,
) -> dict[str, Any]:
    """Fit and write the isolated adjacent-Phase numerical analysis."""

    adjacent_config = _adjacent_config(config)
    root = adjacent_config.cortical_allocation_lmm_root
    planned_paths = [root / path for path in CORTICAL_ALLOCATION_OUTPUTS.values()]
    existing = [path for path in planned_paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Adjacent-Phase cortical-allocation output exists; pass --overwrite "
            f"to replace it: {existing[0]}"
        )

    allocation = load_regional_field_allocation(config)
    source_lfp = prepare_lfp_rows(
        config,
        additional_local_outcomes=config["inference"][
            "cortical_allocation_lmm_adjacent"
        ]["additional_local_outcomes"],
    )
    settings = adjacent_config["inference"]["cortical_allocation_lmm"]
    lfp_rows = derive_adjacent_lfp_rows(
        source_lfp,
        previous_phases=settings["previous_phases"],
    )
    expanded = expand_model_rows(lfp_rows, allocation)
    model_rows, models, pairing = attach_models_and_support(
        expanded,
        planned_models(adjacent_config),
        adjacent_config,
    )
    models = _add_basis(models)
    pairing = _add_basis(pairing)
    outcome_sd = _add_basis(compute_outcome_sd(lfp_rows))
    summary = {
        "analysis_id": settings["analysis_id"],
        "outcome_change_basis": ADJACENT_PHASE_BASIS,
        "status": "ready" if dry_run else "complete",
        "regional_field_rows": int(len(allocation)),
        "lfp_rows": int(len(lfp_rows)),
        "contact_model_rows": int(len(model_rows)),
        "planned_models": int(len(models)),
        "eligible_models": int(models["ModelEligibilityStatus"].eq("eligible").sum()),
        "writes": 0,
    }
    if dry_run:
        return summary

    r_outputs = _run_r_models(
        model_rows,
        models,
        adjacent_config,
        settings_key="cortical_allocation_lmm",
        predictor_column="RelativeAllocation",
        temp_path_key="cortical_allocation_lmm_temp_root",
    )
    processed = {
        key: _add_basis(table)
        for key, table in postprocess_models(
            r_outputs,
            models,
            outcome_sd,
            adjacent_config,
        ).items()
    }
    tables: dict[str, pd.DataFrame] = {
        "regional_field": allocation,
        "contact_model_rows": model_rows,
        "contact_pairing": pairing,
        "outcome_sd": outcome_sd,
        **processed,
        "inputs": adjacent_input_manifest(config, adjacent_config),
    }
    tables["tests"] = _add_basis(tests_manifest(tables))
    tables["outputs"] = _add_basis(_output_manifest(tables, adjacent_config))
    _write_outputs(tables, adjacent_config, overwrite=overwrite)
    summary["fit_status_counts"] = (
        processed["models"]["FitStatus"]
        .fillna("missing")
        .value_counts()
        .sort_index()
        .to_dict()
    )
    summary["joint_test_rows"] = int(len(processed["joint_tests"]))
    summary["phase_slope_rows"] = int(len(processed["phase_slopes"]))
    summary["prediction_grid_rows"] = int(len(processed["prediction_grid"]))
    summary["writes"] = int(len(CORTICAL_ALLOCATION_OUTPUTS))
    return summary
