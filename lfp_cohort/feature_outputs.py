"""Configured feature-output identities and transform-policy metadata."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from .contracts import FeatureOutputSpec


def feature_output_identity(stem: str, config: Mapping[str, Any]) -> tuple[str, str]:
    """Return representation and reducer encoded by a feature-output stem."""
    prefix, separator, representation = stem.rpartition("-")
    metadata = config["metadata"]
    if not separator or representation not in set(metadata["representations"]):
        raise ValueError(f"Unsupported feature-output stem: {stem!r}")
    reducer = (
        prefix if prefix in set(metadata["reducers"]) else metadata["missing_reducer"]
    )
    return representation, reducer


def iter_feature_output_specs(
    config: Mapping[str, Any],
) -> Iterator[FeatureOutputSpec]:
    """Yield the YAML feature-output matrix in deterministic insertion order."""
    for domain, metrics in config["feature_outputs"].items():
        for metric, metric_config in metrics.items():
            if domain == "local":
                feature_outputs = metric_config
                direction = None
            else:
                feature_outputs = metric_config["files"]
                direction = metric_config["direction"]
            for feature_output in feature_outputs:
                representation, reducer = feature_output_identity(
                    feature_output, config
                )
                yield FeatureOutputSpec(
                    domain=domain,
                    metric=metric,
                    feature_output=feature_output,
                    representation=representation,
                    reducer=reducer,
                    direction=direction,
                )


def configured_transform_mode(
    spec: FeatureOutputSpec, config: Mapping[str, Any]
) -> str:
    modes = config["transform"]["modes"]
    metric_modes = modes.get(spec.metric, modes["default"])
    return str(metric_modes.get(spec.reducer, metric_modes.get("default", "none")))


def source_transform_mode(table: pd.DataFrame, path: Path) -> str:
    policy = table.attrs.get("value_transform_policy")
    if not isinstance(policy, Mapping) or "mode" not in policy:
        raise ValueError(f"Missing value_transform_policy.mode in {path}")
    mode = policy["mode"]
    return "none" if mode is None else str(mode)


def validate_transform_policy(
    spec: FeatureOutputSpec,
    source_mode: str,
    applied_mode: str,
    config: Mapping[str, Any],
) -> str:
    """Return matched/override status or reject an unconfigured mismatch."""
    if source_mode == applied_mode:
        return "matched"
    for override in config["transform"].get("allowed_metadata_overrides", []):
        if (
            override["metric"] == spec.metric
            and override["reducer"] == spec.reducer
            and str(override["source"]) == source_mode
            and str(override["applied"]) == applied_mode
        ):
            return "configured_override"
    raise ValueError(
        f"Transform metadata mismatch for {spec.metric}/{spec.feature_output}: "
        f"source={source_mode!r}, configured={applied_mode!r}."
    )
