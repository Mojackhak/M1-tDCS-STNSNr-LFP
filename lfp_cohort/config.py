"""Single-pass YAML configuration loading and boundary validation."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REQUIRED_SECTIONS = (
    "paths",
    "discovery",
    "domains",
    "feature_outputs",
    "metadata",
    "anatomy",
    "coordinates",
    "transform",
    "outlier_filter",
    "normalization",
    "aggregation",
    "connectivity_split",
    "execution",
    "validation",
)


@dataclass(frozen=True)
class CohortConfig(Mapping[str, Any]):
    """Parsed study configuration passed unchanged through every stage."""

    path: Path
    data: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    @property
    def input_root(self) -> Path:
        return Path(self.data["paths"]["input_root"])

    @property
    def output_root(self) -> Path:
        return Path(self.data["paths"]["output_root"])


def _require_mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise TypeError(f"Configuration key {key!r} must contain a mapping.")
    return value


def _validate_config(data: dict[str, Any], path: Path) -> None:
    if "artifacts" in data:
        raise KeyError(
            "Legacy configuration section 'artifacts' is not accepted; use "
            "'feature_outputs'."
        )
    missing = [section for section in REQUIRED_SECTIONS if section not in data]
    if missing:
        raise KeyError(f"Missing configuration sections in {path}: {missing}")
    for section in REQUIRED_SECTIONS:
        _require_mapping(data, section)

    paths = data["paths"]
    for key in ("input_root", "output_root"):
        if not isinstance(paths.get(key), (str, Path)):
            raise TypeError(f"paths.{key} must be a path string.")

    domains = data["domains"]
    expected_kinds = {"local": "node", "connectivity": "pair"}
    for domain, kind in expected_kinds.items():
        configured = _require_mapping(domains, domain).get("kind")
        if configured != kind:
            raise ValueError(f"domains.{domain}.kind must be {kind!r}.")

    feature_outputs = data["feature_outputs"]
    for domain in expected_kinds:
        _require_mapping(feature_outputs, domain)
    for metric, policy in feature_outputs["connectivity"].items():
        if policy.get("direction") not in {"directed", "undirected"}:
            raise ValueError(
                f"feature_outputs.connectivity.{metric}.direction must be directed or "
                "undirected."
            )
        if not isinstance(policy.get("files"), list):
            raise TypeError(
                f"feature_outputs.connectivity.{metric}.files must be a list."
            )

    metadata = data["metadata"]
    representations = set(metadata.get("representations", []))
    if representations != {
        "scalar",
        "spectral",
        "trace",
        "raw",
    }:
        raise ValueError("metadata.representations must define all four contracts.")

    anatomy = data["anatomy"]
    local_keep = set(anatomy.get("local_keep", []))
    local_exclude = set(anatomy.get("local_exclude", []))
    if local_keep & local_exclude or local_keep | local_exclude != {
        "STN",
        "SNr",
        "Mid",
        "EXT",
    }:
        raise ValueError(
            "anatomy.local_keep and anatomy.local_exclude must be disjoint and "
            "cover STN, SNr, Mid, and EXT."
        )

    outlier = data["outlier_filter"]
    outlier_mode = outlier.get("mode")
    if outlier_mode not in {"mcd", "mad"}:
        raise ValueError("outlier_filter.mode must be 'mcd' or 'mad'.")
    enabled = set(outlier.get("enabled_representations", []))
    disabled = set(outlier.get("disabled_representations", []))
    if enabled & disabled or enabled | disabled != representations:
        raise ValueError(
            "outlier_filter enabled and disabled representations must be disjoint "
            "and cover metadata.representations."
        )
    if outlier_mode == "mad":
        if enabled != {"scalar"}:
            raise ValueError(
                "outlier_filter.mode 'mad' requires enabled_representations "
                "to equal ['scalar']."
            )
        if outlier.get("mad_scale") != 1.4826:
            raise ValueError("outlier_filter.mad_scale must equal 1.4826.")
        if outlier.get("robust_z_threshold") != 3.5:
            raise ValueError("outlier_filter.robust_z_threshold must equal 3.5.")

    normalization = data["normalization"]
    if normalization.get("mode") != "mean":
        raise ValueError("normalization.mode must be 'mean'.")
    if normalization.get("mode_baseline") != "mean":
        raise ValueError("normalization.mode_baseline must be 'mean'.")
    if normalization.get("valid_duration_threshold") is not None:
        raise ValueError("normalization.valid_duration_threshold must be null.")

    aggregation = data["aggregation"]
    if aggregation.get("mode") != "mean":
        raise ValueError("aggregation.mode must be 'mean'.")
    if aggregation.get("align") != "strict":
        raise ValueError("aggregation.align must be 'strict'.")
    if aggregation.get("drop_empty") is not False:
        raise ValueError("aggregation.drop_empty must be false.")
    expected_reductions = {
        "contact": {"Epoch"},
        "region": {"Channel", "ChannelPair"},
        "subject": {"Lat"},
    }
    for level, expected in expected_reductions.items():
        configured = _require_mapping(aggregation, level).get("reduce")
        if (
            not isinstance(configured, list)
            or len(configured) != len(expected)
            or set(configured) != expected
        ):
            raise ValueError(
                f"aggregation.{level}.reduce must contain {sorted(expected)}."
            )


def load_config(path: Path | str) -> CohortConfig:
    """Read and validate one YAML file at the configuration boundary."""
    resolved = Path(path).expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise TypeError(f"Configuration must contain a mapping: {resolved}")
    _validate_config(data, resolved)
    return CohortConfig(path=resolved, data=data)
