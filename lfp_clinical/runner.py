"""Plan and execute the clinical-correlation statistics workflow."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from lfp_cohort.io import atomic_csv

from .config import ClinicalCorrelationConfig
from .correlation import compute_correlation_outputs
from .data import ClinicalCorrelationInputs, load_clinical_correlation_inputs


@dataclass(frozen=True)
class ClinicalCorrelationPlan:
    """Validated input tables and deterministic output identities."""

    inputs: ClinicalCorrelationInputs
    planned_correlations: int


def build_clinical_correlation_plan(
    config: ClinicalCorrelationConfig,
) -> ClinicalCorrelationPlan:
    """Read and validate all inputs without computing permutations or writing."""

    inputs = load_clinical_correlation_inputs(config)
    planned = (
        len(inputs.predictor_metadata)
        * len(config["scales"])
        * len(config["endpoints"])
    )
    return ClinicalCorrelationPlan(inputs=inputs, planned_correlations=planned)


def _summary(plan: ClinicalCorrelationPlan) -> dict[str, Any]:
    clinical_ids = set(plan.inputs.clinical_endpoints["ID"])
    lfp_ids = set(plan.inputs.predictor_values["ID"])
    return {
        "clinical_patients": len(clinical_ids),
        "lfp_patients": len(lfp_ids),
        "matched_patients": len(clinical_ids.intersection(lfp_ids)),
        "scales": int(plan.inputs.clinical_endpoints["Scale"].nunique()),
        "endpoints": 4,
        "predictors": len(plan.inputs.predictor_metadata),
        "planned_correlations": plan.planned_correlations,
        "writes": 0,
    }


def _potential_paths(config: ClinicalCorrelationConfig) -> list[Path]:
    paths = [
        config.clinical_endpoints_path,
        config.fit_points_path,
        config.correlations_path,
        config.manifest_root / config["manifests"]["inputs"],
        config.manifest_root / config["manifests"]["outputs"],
    ]
    if config.is_adjacent_phase:
        paths.insert(1, config.adjacent_phase_predictors_path)
    return paths


def _trash_existing(paths: list[Path]) -> None:
    trash = Path("/usr/bin/trash")
    if not trash.exists():
        raise FileNotFoundError(f"Trash command is unavailable: {trash}")
    subprocess.run([str(trash), *(str(path) for path in paths)], check=True)


def _write_outputs(
    plan: ClinicalCorrelationPlan,
    correlations: pd.DataFrame,
    fit_points: pd.DataFrame,
    config: ClinicalCorrelationConfig,
) -> dict[str, Any]:
    config.clinical_endpoints_path.parent.mkdir(parents=True, exist_ok=True)
    if config.is_adjacent_phase:
        config.adjacent_phase_predictors_path.parent.mkdir(
            parents=True, exist_ok=True
        )
    config.fit_points_path.parent.mkdir(parents=True, exist_ok=True)
    config.correlations_path.parent.mkdir(parents=True, exist_ok=True)
    config.manifest_root.mkdir(parents=True, exist_ok=True)
    atomic_csv(
        plan.inputs.clinical_endpoints,
        config.clinical_endpoints_path,
        overwrite=False,
    )
    if config.is_adjacent_phase:
        atomic_csv(
            plan.inputs.predictor_values,
            config.adjacent_phase_predictors_path,
            overwrite=False,
        )
    atomic_csv(fit_points, config.fit_points_path, overwrite=False)
    atomic_csv(correlations, config.correlations_path, overwrite=False)
    atomic_csv(
        plan.inputs.inputs_manifest,
        config.manifest_root / config["manifests"]["inputs"],
        overwrite=False,
    )
    output_rows = [
        {
            "AnalysisID": config.analysis_id,
            "OutputType": "clinical_endpoints",
            "Path": str(config.clinical_endpoints_path),
            "Rows": len(plan.inputs.clinical_endpoints),
            "Columns": len(plan.inputs.clinical_endpoints.columns),
            "Status": "ok",
        },
        {
            "AnalysisID": config.analysis_id,
            "OutputType": "fit_points",
            "Path": str(config.fit_points_path),
            "Rows": len(fit_points),
            "Columns": len(fit_points.columns),
            "Status": "ok",
        },
        {
            "AnalysisID": config.analysis_id,
            "OutputType": "correlations",
            "Path": str(config.correlations_path),
            "Rows": len(correlations),
            "Columns": len(correlations.columns),
            "Status": "ok",
        },
    ]
    if config.is_adjacent_phase:
        output_rows.insert(
            1,
            {
                "AnalysisID": config.analysis_id,
                "OutputType": "adjacent_phase_predictors",
                "Path": str(config.adjacent_phase_predictors_path),
                "Rows": len(plan.inputs.predictor_values),
                "Columns": len(plan.inputs.predictor_values.columns),
                "Status": "ok",
            },
        )
    outputs = pd.DataFrame(output_rows)
    atomic_csv(
        outputs,
        config.manifest_root / config["manifests"]["outputs"],
        overwrite=False,
    )
    result = {
        "written_clinical_endpoints": len(plan.inputs.clinical_endpoints),
        "written_fit_points": len(fit_points),
        "written_correlations": len(correlations),
        "writes": len(_potential_paths(config)),
        "status_counts": correlations["Status"].value_counts().sort_index().to_dict(),
    }
    if config.is_adjacent_phase:
        result["written_adjacent_phase_predictors"] = len(
            plan.inputs.predictor_values
        )
    return result


def run_clinical_correlations(
    config: ClinicalCorrelationConfig,
    *,
    dry_run: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Validate the plan and optionally compute and persist all correlations."""

    plan = build_clinical_correlation_plan(config)
    summary = _summary(plan)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if dry_run:
        return summary
    existing = [path for path in _potential_paths(config) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Planned clinical-correlation output exists; pass --overwrite to replace it: {existing[0]}"
        )
    if existing:
        _trash_existing(existing)
    correlations, fit_points = compute_correlation_outputs(plan.inputs, config)
    execution = _write_outputs(plan, correlations, fit_points, config)
    print(json.dumps(execution, indent=2, ensure_ascii=False))
    return {**summary, **execution}
