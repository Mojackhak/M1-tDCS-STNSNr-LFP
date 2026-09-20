"""Shared visualization input and nested-value operations."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def read_csv(path: Path, description: str) -> pd.DataFrame:
    """Read one required non-empty CSV table."""

    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{description} is missing or empty: {path}")
    return pd.read_csv(path)


def require_columns(table: pd.DataFrame, required: set[str], description: str) -> None:
    """Require columns once at the external table boundary."""

    missing = sorted(required.difference(table.columns))
    if missing:
        raise KeyError(f"{description} is missing columns: {missing}")


def path_below(path: Path, root: Path, description: str) -> Path:
    """Resolve a configured path and require it below its configured root."""

    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(root.expanduser().resolve())
    except ValueError as error:
        raise ValueError(f"{description} must be below {root}: {resolved}") from error
    return resolved


def nested_has_finite(value: Any) -> bool:
    """Return whether a Series or DataFrame contains a finite element."""

    if not isinstance(value, (pd.Series, pd.DataFrame)):
        return False
    return bool(np.isfinite(value.to_numpy(dtype=float, copy=False)).any())


def strict_nested_mean(values: list[Any]) -> pd.Series | pd.DataFrame:
    """Compute an element-wise NaN-aware mean with unchanged nested axes."""

    if not values:
        raise ValueError("Cannot average an empty nested-value group.")
    template = values[0]
    if not isinstance(template, (pd.Series, pd.DataFrame)):
        raise TypeError("Nested visualization values must be Series or DataFrames.")
    arrays: list[np.ndarray] = []
    for value in values:
        if type(value) is not type(template):
            raise TypeError("Nested visualization values have mixed types.")
        if isinstance(template, pd.Series):
            if not value.index.equals(template.index):
                raise ValueError("Series axes differ during laterality reduction.")
        elif not value.index.equals(template.index) or not value.columns.equals(
            template.columns
        ):
            raise ValueError("DataFrame axes differ during laterality reduction.")
        arrays.append(value.to_numpy(dtype=float, copy=False))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        result = np.nanmean(np.stack(arrays, axis=0), axis=0)
    if isinstance(template, pd.Series):
        return pd.Series(result, index=template.index, name=template.name)
    return pd.DataFrame(result, index=template.index, columns=template.columns)


def marginalize_laterality(
    table: pd.DataFrame, group_columns: list[str]
) -> pd.DataFrame:
    """Reduce Ipsi/Contra within ID before any across-ID visualization mean."""

    rows: list[pd.Series] = []
    for _, group in table.groupby(
        group_columns, observed=True, dropna=False, sort=False
    ):
        row = group.iloc[0].copy()
        row["Value"] = strict_nested_mean(group["Value"].tolist())
        row["Lat"] = "All"
        rows.append(row)
    if not rows:
        return table.iloc[0:0].copy()
    return pd.DataFrame(rows).reset_index(drop=True)
