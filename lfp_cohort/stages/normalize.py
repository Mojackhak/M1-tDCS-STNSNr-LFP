"""Baseline normalization for split and unsplit feature representations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from lfptensorpipe.stats.preproc.normalize import baseline_normalize

from ..feature_outputs import iter_feature_output_specs
from ..contracts import FeatureOutputSpec, RunState
from ..identity import (
    first_group_labels,
    groupby_casefolded,
    identity_equal,
)
from ..io import atomic_pickle, read_stage_table, stage_path
from ..provenance import record_feature_output
from ..qc import finite_count, nan_like


def normalization_group_columns(
    table: pd.DataFrame, spec: FeatureOutputSpec
) -> list[str]:
    candidates = [
        "ID",
        "Record",
        "Trial",
        "Polar",
        "StimSide",
        "Domain",
        "Metric",
        "FeatureOutput",
        "Representation",
        "Reducer",
        "Channel" if spec.domain == "local" else "ChannelPair",
        "Region" if spec.domain == "local" else "PairRegion",
        "PairDirection",
        "Lat",
        "Band",
    ]
    return [column for column in candidates if column in table]


def _normalize_metadata_group_values(
    group: pd.DataFrame,
    baseline_mask: np.ndarray,
    config: Mapping[str, Any],
) -> dict[Any, Any]:
    """Normalize one metadata group through the supported baseline primitive."""
    policy = config["normalization"]
    baseline_positions = np.flatnonzero(baseline_mask).tolist()
    finite_values = [value for value in group["Value"] if finite_count(value) > 0]
    template = finite_values[0]

    if isinstance(template, pd.DataFrame):
        raise TypeError("Metadata-normalized values must be scalars or Series.")

    if isinstance(template, pd.Series):
        columns: list[pd.Series] = []
        has_data: list[bool] = []
        for value in group["Value"]:
            if finite_count(value) == 0:
                columns.append(pd.Series(np.nan, index=template.index, dtype=float))
                has_data.append(False)
                continue
            if not isinstance(value, pd.Series):
                raise TypeError(
                    "A spectral feature group mixes Series and scalar values."
                )
            if not value.index.equals(template.index):
                raise ValueError("Spectral Series indexes differ during normalization.")
            columns.append(pd.to_numeric(value, errors="coerce"))
            has_data.append(True)
        values = pd.concat(columns, axis=1)
        values.columns = range(len(columns))
        normalized = baseline_normalize(
            values,
            baseline=baseline_positions,
            mode_baseline=policy["mode_baseline"],
            mode=policy["mode"],
        )
        result: dict[Any, Any] = {}
        for position, (index, value) in enumerate(group["Value"].items()):
            if not has_data[position]:
                result[index] = np.nan
                continue
            cell = normalized.iloc[:, position].copy()
            cell.name = value.name
            result[index] = cell
        return result

    if any(
        isinstance(value, (pd.Series, pd.DataFrame))
        for value in group["Value"]
        if finite_count(value) > 0
    ):
        raise TypeError("A scalar feature group mixes scalar and nested values.")
    values = pd.Series(
        [
            float(value) if finite_count(value) > 0 else np.nan
            for value in group["Value"]
        ]
    )
    normalized = baseline_normalize(
        values,
        baseline=baseline_positions,
        mode_baseline=policy["mode_baseline"],
        mode=policy["mode"],
    )
    return {
        index: normalized.iloc[position] for position, index in enumerate(group.index)
    }


def normalize_metadata_baseline(
    table: pd.DataFrame,
    spec: FeatureOutputSpec,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Normalize scalar/spectral rows while preserving missing groups."""
    out = table.copy()
    out["NormalizationStatus"] = "pending"
    excluded_mask = ~out["IncludeAggregate"].astype(bool)
    nonfinite_mask = excluded_mask & out.get(
        "OutlierReason", pd.Series("", index=out.index)
    ).map(lambda value: identity_equal(value, "nonfinite_value"))
    for index in out.index[excluded_mask]:
        out.at[index, "Value"] = nan_like(out.at[index, "Value"])
    out.loc[excluded_mask, "NormalizationStatus"] = "outlier_excluded"
    out.loc[nonfinite_mask, "NormalizationStatus"] = "nonfinite_excluded"

    included = out.loc[~excluded_mask].copy()
    group_columns = normalization_group_columns(included, spec)
    baseline = config["normalization"]["scalar_spectral"]["baseline"]
    missing_indices: list[Any] = []
    qc: list[dict[str, Any]] = []
    for _, group in groupby_casefolded(included, group_columns):
        baseline_mask = np.ones(len(group), dtype=bool)
        for column, value in baseline.items():
            baseline_mask &= (
                group[column]
                .map(lambda item: identity_equal(item, value))
                .to_numpy(dtype=bool)
            )
        baseline_rows = group.loc[baseline_mask]
        has_baseline = any(finite_count(value) > 0 for value in baseline_rows["Value"])
        record = first_group_labels(group, group_columns)
        record.update(
            {
                "Metric": spec.metric,
                "FeatureOutput": spec.feature_output,
                "has_baseline": bool(has_baseline),
                "n_group_rows": int(len(group)),
                "n_baseline_rows": int(len(baseline_rows)),
                "baseline_evaluated": True,
                "NormalizationStatus": (
                    "baseline_available" if has_baseline else "missing_baseline"
                ),
            }
        )
        qc.append(record)
        if has_baseline:
            normalized_values = _normalize_metadata_group_values(
                group,
                baseline_mask,
                config,
            )
            for index, value in normalized_values.items():
                out.at[index, "Value"] = value
            out.loc[group.index, "NormalizationStatus"] = "normalized"
        else:
            missing_indices.extend(group.index.tolist())
    for index in missing_indices:
        out.at[index, "Value"] = nan_like(out.at[index, "Value"])
    out.loc[missing_indices, "NormalizationStatus"] = "missing_baseline"
    return out, qc


def temporal_baseline_has_data(cell: Any, baseline: Sequence[float]) -> bool:
    start, stop = float(baseline[0]), float(baseline[1])
    if isinstance(cell, pd.Series):
        count = len(cell)
        left = int(np.floor(start / 100.0 * count))
        right = int(np.ceil(stop / 100.0 * count))
        return finite_count(cell.iloc[left:right]) > 0
    if isinstance(cell, pd.DataFrame):
        count = cell.shape[1]
        left = int(np.floor(start / 100.0 * count))
        right = int(np.ceil(stop / 100.0 * count))
        return finite_count(cell.iloc[:, left:right]) > 0
    return False


def normalize_temporal_baseline(
    table: pd.DataFrame,
    spec: FeatureOutputSpec,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Normalize each trace/raw cell to its first-quarter mean."""
    out = table.copy()
    out["NormalizationStatus"] = "pending"
    policy = config["normalization"]
    baseline = policy["trace_raw"]["baseline"]
    qc: list[dict[str, Any]] = []
    for index, row in out.iterrows():
        if not bool(row["IncludeAggregate"]):
            out.at[index, "Value"] = nan_like(row["Value"])
            status = (
                "nonfinite_excluded"
                if identity_equal(row.get("OutlierReason", ""), "nonfinite_value")
                else "outlier_excluded"
            )
        elif not temporal_baseline_has_data(row["Value"], baseline):
            out.at[index, "Value"] = nan_like(row["Value"])
            status = "missing_baseline"
        else:
            out.at[index, "Value"] = baseline_normalize(
                row["Value"],
                baseline=baseline,
                mode_baseline=policy["mode_baseline"],
                mode=policy["mode"],
                slice_mode=policy["trace_raw"]["slice_mode"],
            )
            status = "normalized"
        out.at[index, "NormalizationStatus"] = status
        qc.append(
            {
                "ID": row["ID"],
                "Record": row["Record"],
                "Metric": spec.metric,
                "FeatureOutput": spec.feature_output,
                "SourceRow": row["SourceRow"],
                "has_baseline": (
                    np.nan if status.endswith("_excluded") else status == "normalized"
                ),
                "baseline_evaluated": not status.endswith("_excluded"),
                "NormalizationStatus": status,
            }
        )
    return out, qc


def run_normalize(state: RunState) -> None:
    config = state.config
    protocol = int(config["execution"]["pickle_protocol"])
    for spec in iter_feature_output_specs(config):
        source_path = stage_path(state.output_root, "transform", spec, config)
        source = read_stage_table(source_path)
        if spec.representation in {"scalar", "spectral"}:
            normalized, qc = normalize_metadata_baseline(source, spec, config)
        else:
            normalized, qc = normalize_temporal_baseline(source, spec, config)
        normalized.attrs = dict(source.attrs)
        normalized.attrs.update(
            {
                "normalization_mode": config["normalization"]["mode"],
                "normalization_baseline_stat": config["normalization"]["mode_baseline"],
                "normalization_baseline": (
                    config["normalization"]["scalar_spectral"]["baseline"]
                    if spec.representation in {"scalar", "spectral"}
                    else config["normalization"]["trace_raw"]["baseline"]
                ),
            }
        )
        output = stage_path(state.output_root, "normalize", spec, config)
        atomic_pickle(normalized, output, state.overwrite, protocol)
        record_feature_output(state, "normalize", spec, output, normalized, "transform")
        state.add_qc("normalization", qc)
