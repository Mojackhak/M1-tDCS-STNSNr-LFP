"""Merge record-level feature outputs into canonical cohort tables."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from ..anatomy import (
    canonicalize_common,
    contact_lookup,
    merge_connectivity_rows,
    merge_local_rows,
)
from ..feature_outputs import (
    configured_transform_mode,
    iter_feature_output_specs,
    source_transform_mode,
    validate_transform_policy,
)
from ..contracts import RecordSpec, RunState
from ..io import atomic_pickle, read_feature_table, stage_path
from ..provenance import record_feature_output
from ..qc import canonicalize_nested_axes, finite_support_qc


def run_merge(
    records: Sequence[RecordSpec], contact_map: pd.DataFrame, state: RunState
) -> None:
    """Merge every configured feature output without downstream decisions."""
    lookup = contact_lookup(contact_map)
    config = state.config
    protocol = int(config["execution"]["pickle_protocol"])
    for spec in iter_feature_output_specs(config):
        parts: list[pd.DataFrame] = []
        source_modes: set[str] = set()
        policy_statuses: set[str] = set()
        exclusions: list[dict[str, Any]] = []
        applied_mode = configured_transform_mode(spec, config)
        for record in records:
            source_path = record.feature_output_paths[spec.key]
            source = read_feature_table(source_path, spec)
            source_mode = source_transform_mode(source, source_path)
            source_modes.add(source_mode)
            policy_statuses.add(
                validate_transform_policy(spec, source_mode, applied_mode, config)
            )
            common = canonicalize_common(source, record, spec, source_path, config)
            if spec.domain == "local":
                accepted, excluded = merge_local_rows(
                    common, record, spec, lookup, config
                )
            else:
                accepted, excluded = merge_connectivity_rows(
                    common, record, spec, lookup, config
                )
            parts.append(accepted)
            exclusions.extend(excluded)
        if len(source_modes) != 1:
            raise ValueError(
                "Source transform modes vary across records for "
                f"{spec.metric}/{spec.feature_output}."
            )
        merged = pd.concat(parts, ignore_index=True)
        if merged.empty or "Value" not in merged:
            raise ValueError(
                "No accepted rows remain after anatomical inclusion for "
                f"{spec.domain}/{spec.metric}/{spec.feature_output}."
            )
        merged, axis_report = canonicalize_nested_axes(
            merged, atol=float(config["aggregation"]["nested_axis_atol"])
        )
        source_mode = next(iter(source_modes))
        policy_status = (
            "configured_override"
            if "configured_override" in policy_statuses
            else "matched"
        )
        merged.attrs.update(
            {
                "source_transform_mode": source_mode,
                "applied_transform_mode": applied_mode,
                "transform_policy_status": policy_status,
                "domain": spec.domain,
                "metric": spec.metric,
                "feature_output": spec.feature_output,
                "representation": spec.representation,
                "reducer": spec.reducer,
                "nested_axis_atol": float(config["aggregation"]["nested_axis_atol"]),
            }
        )
        output = stage_path(state.output_root, "merge", spec, config)
        atomic_pickle(merged, output, state.overwrite, protocol)
        record_feature_output(state, "merge", spec, output, merged, "record_features")
        state.add_qc("anatomy_exclusions", exclusions)
        state.add_qc(
            "nested_axes",
            [
                {
                    "Domain": spec.domain,
                    "Metric": spec.metric,
                    "FeatureOutput": spec.feature_output,
                    **axis_report,
                }
            ],
        )
        state.add_qc("finite_support", finite_support_qc(merged, spec, config))
