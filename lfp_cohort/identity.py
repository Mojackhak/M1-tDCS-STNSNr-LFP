"""Case-insensitive string identity helpers for cohort tables."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd


def casefold_identity(value: Any) -> Any:
    """Case-fold strings recursively while preserving non-string identities."""
    if isinstance(value, str):
        return value.casefold()
    if isinstance(value, (tuple, list)):
        return tuple(casefold_identity(item) for item in value)
    return value


def identity_equal(left: Any, right: Any) -> bool:
    """Compare identity values with case-insensitive string semantics."""
    left_missing = _is_missing_scalar(left)
    right_missing = _is_missing_scalar(right)
    if left_missing or right_missing:
        return left_missing and right_missing
    return bool(casefold_identity(left) == casefold_identity(right))


def groupby_casefolded(table: pd.DataFrame, columns: Sequence[str]) -> Any:
    """Group by case-folded identity keys without changing table values."""
    keys = [table[column].map(casefold_identity).rename(column) for column in columns]
    return table.groupby(keys, observed=True, dropna=False)


def first_group_labels(group: pd.DataFrame, columns: Sequence[str]) -> dict[str, Any]:
    """Return display labels from the first accepted row of a group."""
    first = group.iloc[0]
    return {column: first[column] for column in columns}


def unique_casefolded_strings(values: Iterable[Any]) -> list[str]:
    """Return one first-seen display string per case-insensitive identity."""
    representatives: dict[str, str] = {}
    for value in values:
        if _is_missing_scalar(value):
            continue
        text = str(value)
        representatives.setdefault(text.casefold(), text)
    return [representatives[key] for key in sorted(representatives)]


def canonical_string(value: Any, choices: Sequence[str]) -> str | None:
    """Return the configured spelling matching a string identity, if any."""
    folded = casefold_identity(value)
    for choice in choices:
        if folded == choice.casefold():
            return choice
    return None


def _is_missing_scalar(value: Any) -> bool:
    """Return whether a non-container identity is missing."""
    if isinstance(value, (tuple, list)):
        return False
    missing = pd.isna(value)
    return bool(missing) if np.isscalar(missing) else False
