"""Run, feature-output, inventory, and QC manifest output."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .feature_outputs import iter_feature_output_specs
from .contracts import FeatureOutputSpec, RecordSpec, RunState
from .io import atomic_csv, atomic_text


def record_feature_output(
    state: RunState,
    stage: str,
    spec: FeatureOutputSpec,
    path: Any,
    table: pd.DataFrame,
    source: str,
) -> None:
    state.feature_output_manifest.append(
        {
            "Stage": stage,
            "SourceStage": source,
            "Domain": spec.domain,
            "Metric": spec.metric,
            "FeatureOutput": spec.feature_output,
            "Representation": spec.representation,
            "Reducer": spec.reducer,
            "Path": str(path),
            "Rows": int(len(table)),
            "Columns": int(len(table.columns)),
            "SourceTransformMode": table.attrs.get("source_transform_mode"),
            "AppliedTransformMode": table.attrs.get("applied_transform_mode"),
            "TransformPolicyStatus": table.attrs.get("transform_policy_status"),
            "NormalizationMode": table.attrs.get("normalization_mode"),
            "AggregationLevel": table.attrs.get("aggregation_level"),
        }
    )


def input_manifest(
    records: Sequence[RecordSpec], config: Mapping[str, Any]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        for spec in iter_feature_output_specs(config):
            path = record.feature_output_paths[spec.key]
            rows.append(
                {
                    "ID": record.subject,
                    "Record": record.record,
                    "Polar": record.polarity,
                    "StimSide": record.stimulation_side,
                    "Domain": spec.domain,
                    "Metric": spec.metric,
                    "FeatureOutput": spec.feature_output,
                    "Path": str(path),
                    "Bytes": path.stat().st_size,
                    "Configured": True,
                }
            )
        for extra in record.extras:
            rows.append(
                {
                    "ID": record.subject,
                    "Record": record.record,
                    "Polar": record.polarity,
                    "StimSide": record.stimulation_side,
                    "Domain": "extra",
                    "Metric": extra.parent.name,
                    "FeatureOutput": extra.stem,
                    "Path": str(extra),
                    "Bytes": extra.stat().st_size,
                    "Configured": False,
                }
            )
    return pd.DataFrame.from_records(rows)


def validate_outputs(state: RunState) -> None:
    """Validate only output paths produced by the current invocation."""
    paths = [Path(row["Path"]) for row in state.feature_output_manifest]
    if len(paths) != len(set(paths)):
        raise ValueError("The output manifest contains duplicate feature-output paths.")
    invalid = [path for path in paths if not path.is_file() or path.stat().st_size == 0]
    if invalid:
        raise FileNotFoundError(
            f"Generated outputs are missing or empty: {[str(path) for path in invalid]}"
        )


def finalize_run(state: RunState, records: Sequence[RecordSpec]) -> None:
    """Write invocation-specific manifests and accumulated QC tables."""
    suffix = state.invocation_stage
    directories = state.config["execution"]["directories"]
    manifest_dir = state.output_root / directories["manifest"]
    qc_dir = state.output_root / directories["qc"]
    atomic_csv(
        input_manifest(records, state.config),
        manifest_dir / f"inputs_{suffix}.csv",
        state.overwrite,
    )
    feature_outputs = pd.DataFrame.from_records(state.feature_output_manifest)
    atomic_csv(
        feature_outputs,
        manifest_dir / f"feature_outputs_{suffix}.csv",
        state.overwrite,
    )
    policy_columns = [
        "Domain",
        "Metric",
        "FeatureOutput",
        "Representation",
        "Reducer",
        "SourceTransformMode",
        "AppliedTransformMode",
        "TransformPolicyStatus",
    ]
    transform_policy = (
        feature_outputs[policy_columns].drop_duplicates().reset_index(drop=True)
    )
    atomic_csv(
        transform_policy,
        manifest_dir / f"transform_policy_{suffix}.csv",
        state.overwrite,
    )
    output_columns = [
        "Stage",
        "SourceStage",
        "Domain",
        "Metric",
        "FeatureOutput",
        "Representation",
        "Reducer",
        "Path",
        "Rows",
        "Columns",
    ]
    atomic_csv(
        feature_outputs[output_columns],
        manifest_dir / f"outputs_{suffix}.csv",
        state.overwrite,
    )
    atomic_text(
        yaml.safe_dump(state.config.data, sort_keys=False, allow_unicode=True),
        manifest_dir / f"config_{suffix}.yaml",
        state.overwrite,
    )
    run_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": state.invocation_stage,
        "config_path": str(state.config.path),
        "input_root": str(state.config.input_root),
        "output_root": str(state.output_root),
        "overwrite": state.overwrite,
        "record_selectors": list(state.record_selectors),
        "n_subjects": len({record.subject for record in records}),
        "n_records": len(records),
        "records": [record.identity for record in records],
        "coordinate_flip": state.flip_provenance,
        "python": os.sys.version.split()[0],
        "pandas": pd.__version__,
    }
    atomic_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=False),
        manifest_dir / f"run_{suffix}.json",
        state.overwrite,
    )
    for name, rows in state.qc.items():
        atomic_csv(
            pd.DataFrame.from_records(rows),
            qc_dir / f"{name}_{suffix}.csv",
            state.overwrite,
        )
