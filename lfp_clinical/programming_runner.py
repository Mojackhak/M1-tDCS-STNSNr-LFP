"""Execute and document the isolated programming-intensity analysis."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from lfp_cohort.io import atomic_csv, atomic_text

from .programming import (
    ANALYSIS_ID,
    ENDPOINTS,
    MODELS,
    ProgrammingConfig,
    ProgrammingInputs,
    compute_programming_correlations,
    load_programming_inputs,
)
from .programming_jitter import compute_programming_jitter
from .runner import _trash_existing

OUTPUTS = {
    "programming_intensity": "derived/programming_intensity.csv",
    "adjacent_phase_predictors": "derived/adjacent_phase_predictors.csv",
    "model_input_long": "derived/model_input_long.csv",
    "correlations": "results/correlations.csv",
    "jitter_summary": "results/jitter_summary.csv",
    "inputs": "manifest/inputs.csv",
    "outputs": "manifest/outputs.csv",
    "run": "manifest/run.json",
    "methods": "README.md",
}
METHODS_PATH = Path(__file__).resolve().parents[1] / "README.md"


def input_summary(inputs: ProgrammingInputs) -> dict[str, Any]:
    dose = inputs.programming.loc[inputs.programming["Available"]]
    coverage = (
        dose.groupby("EndpointID")
        .agg(patients=("ID", "nunique"), sides=("ID", "size"))
        .reset_index()
        .to_dict("records")
    )
    return {
        "analysis_id": ANALYSIS_ID,
        "strategy": inputs.strategy,
        "endpoint_coverage": coverage,
        "features": len(
            inputs.predictor_metadata[
                ["Domain", "Metric", "FeatureOutput", "Band"]
            ].drop_duplicates()
        ),
        "adjacent_predictor_rows": len(inputs.predictors),
        "model_input_rows": len(inputs.model_input),
        "planned_correlations": len(inputs.registry),
        "writes": 0,
    }


def run_programming_intensity(
    config: ProgrammingConfig, *, dry_run: bool = False, overwrite: bool = False
) -> dict[str, Any]:
    """Read once, compute nominal effects and simulations, then write scoped outputs."""

    inputs = load_programming_inputs(config)
    summary = input_summary(inputs)
    print(json.dumps(summary, indent=2), flush=True)
    if dry_run:
        return summary
    paths = {key: config.output_root / relative for key, relative in OUTPUTS.items()}
    paths.update(
        {
            key: config.output_root / "derived" / f"{key}.csv"
            for key in inputs.matching_tables
        }
    )
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Output exists; pass --overwrite to move replaced files to Trash: {existing[0]}"
        )
    print("Computing nominal correlations and fixed-six Holm adjustment.", flush=True)
    correlations = compute_programming_correlations(inputs)
    jitter = compute_programming_jitter(
        inputs, correlations, config, progress=lambda line: print(line, flush=True)
    )
    frames = {
        "programming_intensity": inputs.programming,
        "adjacent_phase_predictors": inputs.predictors,
        "model_input_long": inputs.model_input,
        "correlations": correlations,
        "jitter_summary": jitter,
        "inputs": inputs.manifest,
        **inputs.matching_tables,
    }
    outputs = pd.DataFrame(
        [
            {
                "AnalysisID": ANALYSIS_ID,
                "OutputType": key,
                "Path": str(paths[key]),
                "Rows": len(table),
                "Columns": len(table.columns),
            }
            for key, table in frames.items()
        ]
        + [
            {"AnalysisID": ANALYSIS_ID, "OutputType": key, "Path": str(paths[key])}
            for key in ("run", "methods")
        ]
    )
    outputs["Strategy"] = config.strategy
    counts = (
        correlations.groupby(["Model", "Status"], observed=True)
        .size()
        .rename("rows")
        .reset_index()
        .to_dict("records")
    )
    execution = {
        **summary,
        "writes": len(paths),
        "status_counts": counts,
        "written_correlations": len(correlations),
        "written_jitter_rows": len(jitter),
    }
    lfp_settings = config.lfp.data
    if config.strategy == "contact-matched":
        lfp_settings = {
            "features": config.lfp["features"],
            "paths": {"feature_manifest": str(config.lfp.feature_manifest)},
            "inputs": {
                "lfp": {
                    key: value
                    for key, value in config.lfp["inputs"]["lfp"].items()
                    if key != "manifest_select"
                }
            },
        }
    run_record = {
        **execution,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config.path),
        "lfp_settings_source": str(config.lfp.path),
        "consumed_lfp_settings": lfp_settings,
        "predictor_source_stages": (
            {"local": "aggregate_contact", "connectivity": "split"}
            if config.strategy == "contact-matched"
            else {"local": "aggregate_region", "connectivity": "aggregate_region"}
        ),
        "cross_strategy_comparison": False,
        "endpoints": ENDPOINTS,
        "models": MODELS,
        "minimum_n": 4,
        "alternative": "two-sided",
        "rank_method": "average",
        "correction": "Holm within six adjacent-phase by laterality positions",
        "jitter": {
            "cv": config.cv,
            "iterations": config.iterations,
            "seed": config.seed,
            "generator": "PCG64",
            "stream_key": "seed + CV + ID|Side|Contact",
        },
        "source_paths": {key: str(value) for key, value in config.paths.items()},
        "interpretation": "Exploratory association; proxy is not TEED; nuisance-adjusted inference is approximate; simulation intervals are not sampling confidence intervals.",
    }
    methods = METHODS_PATH.read_text(encoding="utf-8")
    if existing:
        _trash_existing(existing)
    for key, frame in frames.items():
        atomic_csv(frame, paths[key], overwrite=False)
    atomic_csv(outputs, paths["outputs"], overwrite=False)
    atomic_text(
        json.dumps(run_record, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        paths["run"],
        overwrite=False,
    )
    atomic_text(methods, paths["methods"], overwrite=False)
    print(json.dumps(execution, indent=2), flush=True)
    return execution
