"""Load and validate the clinical-correlation YAML boundary."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

TOP_LEVEL_SECTIONS = {
    "analysis",
    "paths",
    "inputs",
    "scales",
    "features",
    "endpoints",
    "inference",
    "outputs",
    "manifests",
    "execution",
}
EXPECTED_ENDPOINTS = [
    ("preop", "spearman", "Baseline", [], 3),
    ("stn-3m", "partial_spearman", "STN3m", ["Baseline"], 3),
    ("stn-snr-3m", "partial_spearman", "STNSNr3m", ["Baseline"], 3),
    (
        "incremental-3m",
        "partial_spearman",
        "IncrementalBenefit3m",
        ["Baseline"],
        3,
    ),
]
CLINICAL_CORRELATION_ID = "clinical-correlation"
ADJACENT_PHASE_CORRELATION_ID = "clinical-correlation-adjacent-phase"
ADJACENT_PHASE_DERIVATION = {
    "mode": "adjacent_difference",
    "source_phases": ["Pre", "Early", "Late", "Post"],
    "contrasts": [
        {
            "phase": "Early",
            "from_phase": "Pre",
            "to_phase": "Early",
            "label": "Early−Pre",
        },
        {
            "phase": "Late",
            "from_phase": "Early",
            "to_phase": "Late",
            "label": "Late−Early",
        },
        {
            "phase": "Post",
            "from_phase": "Late",
            "to_phase": "Post",
            "label": "Post−Late",
        },
    ],
}


@dataclass(frozen=True)
class ClinicalCorrelationConfig(Mapping[str, Any]):
    """Validated clinical-correlation configuration."""

    path: Path
    data: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    @property
    def analysis_id(self) -> str:
        return str(self.data["analysis"]["id"])

    @property
    def is_adjacent_phase(self) -> bool:
        return self.analysis_id == ADJACENT_PHASE_CORRELATION_ID

    @property
    def clinical_workbook(self) -> Path:
        return Path(self.data["paths"]["clinical_workbook"])

    @property
    def feature_manifest(self) -> Path:
        return Path(self.data["paths"]["feature_manifest"])

    @property
    def lfp_root(self) -> Path:
        return Path(self.data["paths"]["lfp_root"])

    @property
    def output_root(self) -> Path:
        return Path(self.data["paths"]["output_root"])

    @property
    def manifest_root(self) -> Path:
        return Path(self.data["paths"]["manifest_root"])

    @property
    def clinical_endpoints_path(self) -> Path:
        return self.output_root / self.data["outputs"]["clinical_endpoints"]

    @property
    def correlations_path(self) -> Path:
        return self.output_root / self.data["outputs"]["correlations"]

    @property
    def fit_points_path(self) -> Path:
        return self.output_root / self.data["outputs"]["fit_points"]

    @property
    def adjacent_phase_predictors_path(self) -> Path:
        return self.output_root / self.data["outputs"]["adjacent_phase_predictors"]


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise TypeError(f"Configuration key {key!r} must contain a mapping.")
    return value


def _string(parent: Mapping[str, Any], key: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value:
        raise TypeError(f"Configuration key {key!r} must be a non-empty string.")
    return value


def _string_list(parent: Mapping[str, Any], key: str) -> list[str]:
    value = parent.get(key)
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise TypeError(f"Configuration key {key!r} must be a string list.")
    if len(value) != len(set(value)):
        raise ValueError(f"Configuration key {key!r} contains duplicates.")
    return value


def _validate_paths(data: Mapping[str, Any]) -> None:
    paths = _mapping(data, "paths")
    required = {
        "clinical_workbook",
        "feature_manifest",
        "lfp_root",
        "output_root",
        "manifest_root",
    }
    if set(paths) != required:
        raise ValueError(f"paths must contain exactly {sorted(required)}.")
    resolved = {key: Path(_string(paths, key)).expanduser() for key in required}
    try:
        resolved["manifest_root"].relative_to(resolved["output_root"])
    except ValueError as error:
        raise ValueError(
            "paths.manifest_root must be below paths.output_root."
        ) from error


def _validate_inputs(data: Mapping[str, Any]) -> None:
    analysis_id = str(_mapping(data, "analysis").get("id"))
    inputs = _mapping(data, "inputs")
    clinical = _mapping(inputs, "clinical")
    if set(_mapping(clinical, "columns")) != {
        "id",
        "protocol",
        "phase",
        "scale",
        "value",
        "baseline",
    }:
        raise ValueError(
            "inputs.clinical.columns does not match the workbook contract."
        )
    if clinical.get("phase") != "3m" or dict(_mapping(clinical, "protocols")) != {
        "stn": "STN",
        "stn_snr": "STN+SNr",
    }:
        raise ValueError("The registered 3-month protocol contract is invalid.")
    _string(clinical, "sheet_name")
    lfp = _mapping(inputs, "lfp")
    _validate_lfp_inputs(
        lfp, adjacent_phase=analysis_id == ADJACENT_PHASE_CORRELATION_ID
    )


def _validate_lfp_inputs(lfp: Mapping[str, Any], *, adjacent_phase: bool) -> None:
    """Validate only the LFP selection, independently of clinical endpoints."""

    if dict(_mapping(lfp, "manifest_select")) != {
        "Stage": "aggregate_region",
        "SourceStage": "normalize",
        "AggregationLevel": "region",
        "Representation": "scalar",
    }:
        raise ValueError(
            "inputs.lfp.manifest_select does not select normalized regions."
        )
    expected_levels = {
        "phases": ["Early", "Late", "Post"],
        "laterality": ["Ipsi", "Contra"],
        "polarity": ["Anodal", "Cathodal"],
    }
    for key, expected in expected_levels.items():
        if _string_list(lfp, key) != expected:
            raise ValueError(f"inputs.lfp.{key} must be {expected}.")
    if lfp.get("include_aggregate") is not True or lfp.get("value_column") != "Value":
        raise ValueError("The LFP row filter does not match the scalar table contract.")
    derivation = lfp.get("phase_derivation")
    if adjacent_phase:
        if derivation != ADJACENT_PHASE_DERIVATION:
            raise ValueError("The adjacent-phase derivation contract is invalid.")
    elif derivation is not None:
        raise ValueError("The Phase-minus-Pre workflow cannot derive adjacent phases.")


def _validate_scales(data: Mapping[str, Any]) -> None:
    scales = data.get("scales")
    if not isinstance(scales, list) or len(scales) != 28:
        raise ValueError("scales must register exactly 28 clinical scales.")
    values: list[str] = []
    labels: list[str] = []
    higher = []
    for scale in scales:
        if not isinstance(scale, Mapping):
            raise TypeError("Each scale must be a mapping.")
        values.append(_string(scale, "value"))
        labels.append(_string(scale, "label"))
        direction = scale.get("direction")
        if direction not in {"lower_is_better", "higher_is_better"}:
            raise ValueError(f"Unsupported scale direction: {direction!r}.")
        if direction == "higher_is_better":
            higher.append(scale["value"])
    if len(values) != len(set(values)) or len(labels) != len(set(labels)):
        raise ValueError("Scale values and labels must be unique.")
    if higher != ["SE-ADL score (%)"]:
        raise ValueError("SE-ADL must be the only higher-is-better scale.")


def _validate_features(data: Mapping[str, Any]) -> None:
    features = _mapping(data, "features")
    bands = features.get("bands")
    if not isinstance(bands, list) or [item.get("value") for item in bands] != [
        "delta",
        "theta",
        "alpha",
        "beta_low",
        "beta_high",
        "gamma_low",
        "gamma_high",
    ]:
        raise ValueError("features.bands does not match the canonical band order.")
    local = _mapping(features, "local")
    if local.get("region_column") != "Region" or local.get("regions") != ["STN", "SNr"]:
        raise ValueError("Local regions must be STN and SNr.")
    banded = local.get("banded")
    if not isinstance(banded, list) or len(banded) != 6:
        raise ValueError("Local features must contain six banded metrics.")
    aperiodic = _mapping(local, "aperiodic")
    if [item.get("value") for item in aperiodic.get("parameters", [])] != [
        "exponent",
        "offset",
    ]:
        raise ValueError("Aperiodic features must be Exponent and Offset.")
    connectivity = _mapping(features, "connectivity")
    if (
        connectivity.get("region_column") != "PairRegion"
        or connectivity.get("display_region") != "SNr-STN-combined"
    ):
        raise ValueError("Connectivity region metadata is invalid.")
    metrics = connectivity.get("metrics")
    if not isinstance(metrics, list) or [item.get("metric") for item in metrics] != [
        "ciplv",
        "imcoh_abs",
        "wpli",
        "psi",
        "trgc",
    ]:
        raise ValueError("Connectivity metrics do not match the registered order.")


def _validate_endpoints(data: Mapping[str, Any]) -> None:
    endpoints = data.get("endpoints")
    if not isinstance(endpoints, list) or len(endpoints) != len(EXPECTED_ENDPOINTS):
        raise ValueError("Four clinical endpoints are required.")
    observed = [
        (
            item.get("id"),
            item.get("method"),
            item.get("outcome_column"),
            item.get("covariates"),
            item.get("minimum_n"),
        )
        for item in endpoints
    ]
    if observed != EXPECTED_ENDPOINTS:
        raise ValueError("endpoints do not match the registered analysis contract.")
    for item in endpoints:
        _string(item, "label")


def _validate_inference(data: Mapping[str, Any]) -> None:
    inference = _mapping(data, "inference")
    if {
        key: inference.get(key)
        for key in (
            "alternative",
            "p_method",
            "partial_permutation",
            "rank_method",
            "significance_basis",
        )
    } != {
        "alternative": "two-sided",
        "p_method": "exhaustive_permutation",
        "partial_permutation": "freedman_lane",
        "rank_method": "average",
        "significance_basis": "p_holm",
    }:
        raise ValueError(
            "inference does not match the registered permutation contract."
        )
    if dict(_mapping(inference, "correction")) != {
        "method": "holm",
        "family_fields": [
            "EndpointID",
            "Scale",
            "Domain",
            "RegionValue",
            "Polar",
            "Metric",
            "FeatureOutput",
            "Band",
        ],
    }:
        raise ValueError("The Holm Phase-by-laterality family is invalid.")


def _validate_outputs(data: Mapping[str, Any]) -> None:
    outputs = _mapping(data, "outputs")
    expected_outputs = {
        "clinical_endpoints": "derived/clinical_endpoints.csv",
        "fit_points": "derived/correlation_fit_points.csv",
        "correlations": "results/correlations.csv",
    }
    if _mapping(data, "analysis").get("id") == ADJACENT_PHASE_CORRELATION_ID:
        expected_outputs["adjacent_phase_predictors"] = (
            "derived/adjacent_phase_predictors.csv"
        )
    if dict(outputs) != expected_outputs:
        raise ValueError("outputs do not match the registered paths.")
    if dict(_mapping(data, "manifests")) != {
        "inputs": "inputs.csv",
        "outputs": "outputs.csv",
    }:
        raise ValueError("manifests must be inputs.csv and outputs.csv.")
    if dict(_mapping(data, "execution")) != {
        "on_error": "fail",
        "retry": False,
        "cache": False,
        "resume": False,
    }:
        raise ValueError("execution does not match the failure contract.")


def load_clinical_correlation_config(
    path: Path | str,
) -> ClinicalCorrelationConfig:
    """Parse and validate one clinical-correlation YAML file."""

    resolved = Path(path).expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise TypeError(f"Configuration must contain a mapping: {resolved}")
    if set(data) != TOP_LEVEL_SECTIONS:
        raise ValueError(
            f"Configuration sections must be exactly {sorted(TOP_LEVEL_SECTIONS)}."
        )
    analysis = _mapping(data, "analysis")
    if (
        analysis.get("id")
        not in {
            CLINICAL_CORRELATION_ID,
            ADJACENT_PHASE_CORRELATION_ID,
        }
        or analysis.get("kind") != "clinical_correlation"
    ):
        raise ValueError("analysis identity is invalid.")
    _string(analysis, "question")
    _validate_paths(data)
    _validate_inputs(data)
    _validate_scales(data)
    _validate_features(data)
    _validate_endpoints(data)
    _validate_inference(data)
    _validate_outputs(data)
    return ClinicalCorrelationConfig(path=resolved, data=data)
