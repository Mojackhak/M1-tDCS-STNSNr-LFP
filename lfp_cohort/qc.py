"""Nested-axis and finite-support quality-control helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from .contracts import FeatureOutputSpec


def finite_count(cell: Any) -> int:
    if isinstance(cell, pd.Series):
        return int(np.isfinite(cell.to_numpy(dtype=float, copy=False)).sum())
    if isinstance(cell, pd.DataFrame):
        return int(np.isfinite(cell.to_numpy(dtype=float, copy=False)).sum())
    try:
        return int(np.isfinite(float(cell)))
    except (TypeError, ValueError):
        return 0


def value_size(cell: Any) -> int:
    return int(cell.size) if isinstance(cell, (pd.Series, pd.DataFrame)) else 1


def nan_like(cell: Any) -> Any:
    if isinstance(cell, pd.Series):
        return pd.Series(np.nan, index=cell.index, name=cell.name, dtype=float)
    if isinstance(cell, pd.DataFrame):
        return pd.DataFrame(np.nan, index=cell.index, columns=cell.columns, dtype=float)
    return np.nan


def _canonical_axis(
    axis: pd.Index,
    reference: pd.Index,
    *,
    atol: float,
    label: str,
) -> tuple[pd.Index, bool, float]:
    if axis.equals(reference):
        return reference, False, 0.0
    if len(axis) != len(reference):
        raise ValueError(
            f"Nested {label} length differs at merge: {len(axis)} versus "
            f"{len(reference)}."
        )
    try:
        current = axis.to_numpy(dtype=float)
        expected = reference.to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Nested non-numeric {label} labels differ at merge."
        ) from error
    difference = float(np.max(np.abs(current - expected))) if len(current) else 0.0
    if not np.allclose(current, expected, rtol=0.0, atol=atol, equal_nan=True):
        raise ValueError(
            f"Nested {label} labels differ beyond atol={atol}: "
            f"max difference={difference}."
        )
    return reference, True, difference


def canonicalize_nested_axes(
    table: pd.DataFrame, *, atol: float
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Canonicalize floating-point-equivalent labels without interpolation."""
    out = table.copy()
    nested = [
        value for value in out["Value"] if isinstance(value, (pd.Series, pd.DataFrame))
    ]
    report = {
        "n_rows": int(len(out)),
        "n_nested_rows": int(len(nested)),
        "n_canonicalized_rows": 0,
        "max_axis_difference": 0.0,
        "atol": float(atol),
    }
    if not nested:
        return out, report
    template = nested[0]
    for index, value in out["Value"].items():
        if not isinstance(value, (pd.Series, pd.DataFrame)):
            if finite_count(value) > 0:
                raise TypeError(
                    "A nested feature output contains a finite scalar Value."
                )
            continue
        changed = False
        maximum = 0.0
        if isinstance(template, pd.Series):
            if not isinstance(value, pd.Series):
                raise TypeError(
                    "A nested feature output mixes Series and DataFrame values."
                )
            canonical_index, changed, maximum = _canonical_axis(
                value.index, template.index, atol=atol, label="Series index"
            )
            if changed:
                canonical = value.copy()
                canonical.index = canonical_index
                out.at[index, "Value"] = canonical
        else:
            if not isinstance(value, pd.DataFrame):
                raise TypeError(
                    "A nested feature output mixes DataFrame and Series values."
                )
            canonical_index, index_changed, index_difference = _canonical_axis(
                value.index, template.index, atol=atol, label="DataFrame index"
            )
            canonical_columns, columns_changed, columns_difference = _canonical_axis(
                value.columns, template.columns, atol=atol, label="DataFrame columns"
            )
            changed = index_changed or columns_changed
            maximum = max(index_difference, columns_difference)
            if changed:
                canonical = value.copy()
                canonical.index = canonical_index
                canonical.columns = canonical_columns
                out.at[index, "Value"] = canonical
        if changed:
            report["n_canonicalized_rows"] += 1
            report["max_axis_difference"] = max(report["max_axis_difference"], maximum)
    return out, report


def finite_support_qc(
    table: pd.DataFrame, spec: FeatureOutputSpec, config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Summarize finite output support without applying a threshold."""
    records: list[dict[str, Any]] = []
    sample_rate = float(config["validation"]["sample_rate_hz"])
    phase_order = list(config["validation"]["phase_order"])
    label_column = "Channel" if spec.domain == "local" else "ChannelPair"
    for _, row in table.iterrows():
        cell = row["Value"]
        common = {
            "ID": row["ID"],
            "Record": row["Record"],
            "Metric": spec.metric,
            "FeatureOutput": spec.feature_output,
            "Domain": spec.domain,
            "Contact": str(row[label_column]),
            "Band": row.get("Band", np.nan),
        }
        if spec.representation in {"trace", "raw"}:
            if isinstance(cell, pd.Series):
                temporal = np.isfinite(cell.to_numpy(dtype=float, copy=False))
                n_value_expected = len(temporal)
                n_value_finite = int(temporal.sum())
            elif isinstance(cell, pd.DataFrame):
                matrix = cell.to_numpy(dtype=float, copy=False)
                temporal = np.isfinite(matrix).any(axis=0)
                n_value_expected = int(matrix.size)
                n_value_finite = int(np.isfinite(matrix).sum())
            else:
                temporal = np.asarray([], dtype=bool)
                n_value_expected = 1
                n_value_finite = finite_count(cell)
            phase_indices = np.array_split(np.arange(len(temporal)), len(phase_order))
            for phase, indices in zip(phase_order, phase_indices):
                n_expected = int(len(indices))
                n_finite = int(temporal[indices].sum()) if n_expected else 0
                finite_fraction = n_finite / n_expected if n_expected else np.nan
                sparse_burst = spec.metric == "burst" and spec.representation == "raw"
                records.append(
                    {
                        **common,
                        "Phase": phase,
                        "n_expected": n_expected,
                        "n_finite": n_finite,
                        "finite_fraction": finite_fraction,
                        "valid_seconds": np.nan
                        if sparse_burst
                        else n_finite / sample_rate,
                        "valid_fraction": np.nan if sparse_burst else finite_fraction,
                        "support_semantics": (
                            "sparse_burst_feature_coverage"
                            if sparse_burst
                            else "finite_output_time_support"
                        ),
                        "n_value_expected_full": n_value_expected,
                        "n_value_finite_full": n_value_finite,
                    }
                )
        else:
            expected = value_size(cell)
            finite = finite_count(cell)
            records.append(
                {
                    **common,
                    "Phase": row["Phase"],
                    "n_expected": expected,
                    "n_finite": finite,
                    "finite_fraction": finite / expected if expected else np.nan,
                    "valid_seconds": np.nan,
                    "valid_fraction": finite / expected if expected else np.nan,
                    "support_semantics": "finite_output_elements",
                    "n_value_expected_full": expected,
                    "n_value_finite_full": finite,
                }
            )
    return records
