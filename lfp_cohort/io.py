"""PKL/manifest I/O and output path construction."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contracts import FeatureOutputSpec


def stage_path(
    root: Path, stage: str, spec: FeatureOutputSpec, config: Mapping[str, Any]
) -> Path:
    directory = config["execution"]["directories"][stage]
    return root / directory / spec.domain / spec.metric / f"{spec.feature_output}.pkl"


def aggregate_path(
    root: Path,
    source_stage: str,
    level: str,
    spec: FeatureOutputSpec,
    config: Mapping[str, Any],
) -> Path:
    directory = config["execution"]["directories"]["aggregate"]
    return (
        root
        / directory
        / source_stage
        / level
        / spec.domain
        / spec.metric
        / f"{spec.feature_output}.pkl"
    )


def split_path(
    root: Path,
    source_stage: str,
    spec: FeatureOutputSpec,
    config: Mapping[str, Any],
) -> Path:
    directory = config["execution"]["directories"]["split"]
    return (
        root
        / directory
        / source_stage
        / "connectivity"
        / spec.metric
        / f"{spec.feature_output}.pkl"
    )


def read_table(path: Path) -> pd.DataFrame:
    table = pd.read_pickle(path)
    if not isinstance(table, pd.DataFrame):
        raise TypeError(f"PKL must contain a pandas DataFrame: {path}")
    return table


def read_stage_table(path: Path) -> pd.DataFrame:
    """Read a persisted stage table using the current feature-output schema."""
    table = read_table(path)
    if "Artifact" in table:
        raise KeyError(
            f"Stage table contains legacy Artifact schema; expected FeatureOutput "
            f"only: {path}"
        )
    if "FeatureOutput" not in table:
        raise KeyError(f"Stage table has no FeatureOutput column: {path}")
    return table


def read_feature_table(path: Path, spec: FeatureOutputSpec) -> pd.DataFrame:
    """Validate the external PKL representation once at the file boundary."""
    table = read_table(path)
    if "Value" not in table:
        raise KeyError(f"Feature table has no Value column: {path}")
    nonempty = [value for value in table["Value"] if not _is_missing_scalar(value)]
    expected_type: type[Any] | tuple[type[Any], ...]
    if spec.representation == "scalar":
        expected_type = (int, float, np.number)
    elif spec.representation in {"spectral", "trace"}:
        expected_type = pd.Series
    else:
        expected_type = pd.DataFrame
    invalid = [
        type(value).__name__
        for value in nonempty
        if not isinstance(value, expected_type)
    ]
    if invalid:
        raise TypeError(
            f"Unexpected Value type for {spec.representation} feature output {path}: "
            f"{invalid[0]}"
        )
    return table


def _is_missing_scalar(value: Any) -> bool:
    if isinstance(value, (pd.Series, pd.DataFrame)):
        return False
    missing = pd.isna(value)
    return bool(missing) if np.isscalar(missing) else False


def _ensure_write_target(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace it: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def atomic_pickle(
    table: pd.DataFrame, path: Path, overwrite: bool, protocol: int
) -> None:
    _ensure_write_target(path, overwrite)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temp_path = Path(temp_name)
    try:
        table.to_pickle(temp_path, protocol=protocol)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def atomic_csv(table: pd.DataFrame, path: Path, overwrite: bool) -> None:
    _ensure_write_target(path, overwrite)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temp_path = Path(temp_name)
    try:
        table.to_csv(temp_path, index=False)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def atomic_text(text: str, path: Path, overwrite: bool) -> None:
    _ensure_write_target(path, overwrite)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temp_path = Path(temp_name)
    try:
        temp_path.write_text(text, encoding="utf-8")
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
