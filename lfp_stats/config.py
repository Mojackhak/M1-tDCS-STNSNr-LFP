"""Load and validate statistical-analysis YAML at the file boundary."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REQUIRED_SECTIONS = (
    "analysis",
    "paths",
    "inputs",
    "canonical_region",
    "feature_whitelist",
    "strata",
    "grouping",
    "model_data",
    "model",
    "tests",
    "identity",
    "outputs",
    "manifests",
    "execution",
)
PAIRED_CONTACT_ANALYSES = {
    "phase-by-lat-contact-paired",
    "phase-within-lat-contact-paired",
}
CONTACT_RANDOM_EFFECT_ANALYSES = {
    *PAIRED_CONTACT_ANALYSES,
    "polarity-by-phase-lat-contact-paired",
}
CONTACT_ANALYSES = {
    "phase-by-lat-contact",
    "phase-within-lat-contact",
    "pre-by-polar-contact-paired",
    *CONTACT_RANDOM_EFFECT_ANALYSES,
}
LAT_STRATIFIED_ANALYSES = {
    "phase-within-lat",
    "phase-within-lat-contact",
    "phase-within-lat-contact-paired",
}
POLAR_STRATIFIED_ANALYSES = {"laterality-within-polar"}
PRE_POLAR_ANALYSES = {"pre-by-polar", "pre-by-polar-contact-paired"}


@dataclass(frozen=True)
class StatsConfig(Mapping[str, Any]):
    """Validated study configuration used by the stats runner."""

    path: Path
    data: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    @property
    def table_root(self) -> Path:
        return Path(self.data["paths"]["table_root"])

    @property
    def feature_manifest(self) -> Path:
        return Path(self.data["paths"]["feature_manifest"])

    @property
    def output_root(self) -> Path:
        return Path(self.data["paths"]["output_root"])

    @property
    def manifest_root(self) -> Path:
        return Path(self.data["paths"]["manifest_root"])

    @property
    def derived_root(self) -> Path:
        return Path(self.data["paths"]["derived_root"])

    @property
    def analysis_id(self) -> str:
        return str(self.data["analysis"]["id"])


def _require_mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise TypeError(f"Configuration key {key!r} must contain a mapping.")
    return value


def _require_string(parent: Mapping[str, Any], key: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value:
        raise TypeError(f"Configuration key {key!r} must be a non-empty string.")
    return value


def _require_string_list(parent: Mapping[str, Any], key: str) -> list[str]:
    value = parent.get(key)
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise TypeError(
            f"Configuration key {key!r} must be a non-empty list of strings."
        )
    if len(value) != len(set(value)):
        raise ValueError(f"Configuration key {key!r} contains duplicate values.")
    return value


def _validate_features(
    data: Mapping[str, Any], *, allow_empty_connectivity: bool = False
) -> None:
    canonical = _require_mapping(data, "canonical_region")
    local_regions = _require_string_list(canonical, "local")
    connectivity_regions = _require_mapping(canonical, "connectivity")
    allowed_pair_regions = set(
        _require_string_list(connectivity_regions, "undirected")
        + _require_string_list(connectivity_regions, "directed")
    )

    whitelist = _require_mapping(data, "feature_whitelist")
    local = _require_mapping(whitelist, "local")
    if _require_string(local, "region_column") != "Region":
        raise ValueError("feature_whitelist.local.region_column must be 'Region'.")
    if _require_string_list(local, "region_values") != local_regions:
        raise ValueError("Local region values must match canonical_region.local.")

    connectivity = _require_mapping(whitelist, "connectivity")
    if _require_string(connectivity, "region_column") != "PairRegion":
        raise ValueError(
            "feature_whitelist.connectivity.region_column must be 'PairRegion'."
        )

    for domain_name, domain in (("local", local), ("connectivity", connectivity)):
        metrics = _require_mapping(domain, "metrics")
        if not metrics:
            if domain_name == "connectivity" and allow_empty_connectivity:
                continue
            raise ValueError(f"feature_whitelist.{domain_name}.metrics is empty.")
        for metric, metric_config in metrics.items():
            if not isinstance(metric, str) or not metric:
                raise TypeError("Metric names must be non-empty strings.")
            if not isinstance(metric_config, Mapping):
                raise TypeError(f"Metric configuration must be a mapping: {metric}")
            if domain_name == "connectivity":
                configured_regions = _require_string_list(
                    metric_config, "region_values"
                )
                if not set(configured_regions).issubset(allowed_pair_regions):
                    raise ValueError(
                        f"Connectivity metric has non-canonical regions: {metric}"
                    )
            outputs = _require_mapping(metric_config, "outputs")
            for feature_output, output_config in outputs.items():
                if not isinstance(feature_output, str) or not feature_output:
                    raise TypeError("FeatureOutput names must be non-empty strings.")
                if not isinstance(output_config, Mapping):
                    raise TypeError(
                        f"FeatureOutput configuration must be a mapping: {feature_output}"
                    )
                _require_string_list(output_config, "bands")


def _validate_tests(data: Mapping[str, Any]) -> None:
    tests = data.get("tests")
    if not isinstance(tests, list) or not tests:
        raise TypeError("Configuration key 'tests' must be a non-empty list.")
    identities: set[tuple[str, str]] = set()
    for test in tests:
        if not isinstance(test, Mapping):
            raise TypeError("Each test configuration must be a mapping.")
        test_id = _require_string(test, "id")
        kind = _require_string(test, "kind")
        identity = (kind, test_id)
        if identity in identities:
            raise ValueError(f"Duplicate configured test identity: {identity}")
        identities.add(identity)
        if kind == "joint_term":
            _require_string(test, "term")
        elif kind == "emm_pairwise":
            _require_string(test, "x_var")
            if test.get("primary_p") != "p_tukey":
                raise ValueError("emm_pairwise primary_p must be 'p_tukey'.")
            if test.get("within_adjust") != "tukey":
                raise ValueError("emm_pairwise within_adjust must be 'tukey'.")
            if test.get("across_method") != "none":
                raise ValueError("emm_pairwise across_method must be 'none'.")
            if test.get("contrast_direction") != "later_minus_earlier":
                raise ValueError(
                    "emm_pairwise contrast_direction must be 'later_minus_earlier'."
                )
        elif kind == "emmean_vs_null":
            _require_string(test, "x_var")
            null = test.get("null")
            if not isinstance(null, (int, float)):
                raise TypeError("emmean_vs_null null must be numeric.")
            if test.get("weights") != "equal":
                raise ValueError("emmean_vs_null weights must be 'equal'.")
            if test.get("confidence_level") != 0.95:
                raise ValueError("emmean_vs_null confidence_level must be 0.95.")
            if test.get("adjust_method") != "holm":
                raise ValueError("emmean_vs_null adjust_method must be 'holm'.")
            if test.get("adjust_scope") != "all_cells":
                raise ValueError("emmean_vs_null adjust_scope must be 'all_cells'.")
            if test.get("primary_p") != "p_raw":
                raise ValueError("emmean_vs_null primary_p must be 'p_raw'.")
        else:
            raise ValueError(f"Unsupported test kind: {kind}")


def _validate_derivation(
    data: Mapping[str, Any], paths: Mapping[str, Any]
) -> str | None:
    derivation = data.get("derivation")
    if derivation is None:
        return None
    if not isinstance(derivation, Mapping):
        raise TypeError("Configuration key 'derivation' must contain a mapping.")
    kind = _require_string(derivation, "kind")
    regional_contract = {
        "kind": "region_polarity_difference",
        "region_factor": "Region",
        "region_minuend": "SNr",
        "region_subtrahend": "STN",
        "region_label": "SNr-minus-STN",
        "polar_factor": "Polar",
        "polar_minuend": "Anodal",
        "polar_subtrahend": "Cathodal",
        "polar_label": "Anodal-minus-Cathodal",
        "pair_columns": ["ID", "Phase", "Lat", "Band"],
        "source_columns": ["Record"],
        "exclude_phases": ["Pre"],
        "incomplete": "exclude_and_report",
        "pickle_protocol": 5,
    }
    if kind == "region_polarity_difference":
        if dict(derivation) != regional_contract:
            raise ValueError(
                "region_polarity_difference derivation must equal "
                f"{regional_contract}."
            )
        result = "RegionPolar"
    else:
        factor = _require_string(derivation, "factor")
        result = factor
    contracts = {
        ("polarity-by-phase-lat-contact-paired", "Polar"): {
            "kind": "paired_difference",
            "factor": "Polar",
            "minuend": "Anodal",
            "subtrahend": "Cathodal",
            "label": "Anodal-minus-Cathodal",
            "pair_columns": [
                "ID",
                "ContactUnitID",
                "Phase",
                "Lat",
                "Band",
            ],
            "source_columns": ["Record", "Trial"],
            "exclude_phases": ["Pre"],
            "unpaired": "exclude_and_report",
            "pickle_protocol": 5,
        },
        ("polarity-by-phase-lat", "Polar"): {
            "kind": "paired_difference",
            "factor": "Polar",
            "minuend": "Anodal",
            "subtrahend": "Cathodal",
            "label": "Anodal-minus-Cathodal",
            "pair_columns": ["ID", "Phase", "Lat", "Band"],
            "source_columns": ["Record"],
            "exclude_phases": ["Pre"],
            "unpaired": "exclude_and_report",
            "pickle_protocol": 5,
        },
        ("pre-by-polar", "Polar"): {
            "kind": "paired_difference",
            "factor": "Polar",
            "minuend": "Anodal",
            "subtrahend": "Cathodal",
            "label": "Anodal-minus-Cathodal",
            "pair_columns": ["ID", "Phase", "Lat", "Band"],
            "source_columns": ["Record", "Trial"],
            "include_phases": ["Pre"],
            "unpaired": "exclude_and_report",
            "pickle_protocol": 5,
        },
        ("pre-by-polar-contact-paired", "Polar"): {
            "kind": "paired_difference",
            "factor": "Polar",
            "minuend": "Anodal",
            "subtrahend": "Cathodal",
            "label": "Anodal-minus-Cathodal",
            "pair_columns": [
                "ID",
                "ContactUnitID",
                "Phase",
                "Lat",
                "Band",
            ],
            "source_columns": ["Record", "Trial"],
            "include_phases": ["Pre"],
            "unpaired": "exclude_and_report",
            "pickle_protocol": 5,
        },
        ("laterality-by-phase-polar", "Lat"): {
            "kind": "paired_difference",
            "factor": "Lat",
            "minuend": "Ipsi",
            "subtrahend": "Contra",
            "label": "Ipsi-minus-Contra",
            "pair_columns": [
                "ID",
                "Record",
                "Trial",
                "Polar",
                "Phase",
                "Band",
            ],
            "source_columns": [],
            "exclude_phases": ["Pre"],
            "unpaired": "exclude_and_report",
            "pickle_protocol": 5,
        },
        ("laterality-within-polar", "Lat"): {
            "kind": "paired_difference",
            "factor": "Lat",
            "minuend": "Ipsi",
            "subtrahend": "Contra",
            "label": "Ipsi-minus-Contra",
            "pair_columns": [
                "ID",
                "Record",
                "Trial",
                "Polar",
                "Phase",
                "Band",
            ],
            "source_columns": [],
            "exclude_phases": ["Pre"],
            "unpaired": "exclude_and_report",
            "pickle_protocol": 5,
        },
        ("region-by-phase-lat", "Region"): {
            "kind": "paired_difference",
            "factor": "Region",
            "minuend": "SNr",
            "subtrahend": "STN",
            "label": "SNr-minus-STN",
            "pair_columns": [
                "ID",
                "Record",
                "Trial",
                "Polar",
                "Phase",
                "Lat",
                "Band",
            ],
            "source_columns": [],
            "exclude_phases": ["Pre"],
            "unpaired": "exclude_and_report",
            "pickle_protocol": 5,
        },
    }
    if kind != "region_polarity_difference":
        analysis_id = _require_string(_require_mapping(data, "analysis"), "id")
        expected = contracts.get((analysis_id, factor))
        if expected is None or dict(derivation) != expected:
            raise ValueError(
                f"derivation must equal one supported contract: {contracts}."
            )
    _require_string(paths, "derived_root")
    output_root = Path(str(paths["output_root"]))
    try:
        Path(str(paths["derived_root"])).relative_to(output_root)
    except ValueError as error:
        raise ValueError(
            "paths.derived_root must be below paths.output_root."
        ) from error
    return result


def _validate_config(data: dict[str, Any], path: Path) -> None:
    missing = [section for section in REQUIRED_SECTIONS if section not in data]
    if missing:
        raise KeyError(f"Missing configuration sections in {path}: {missing}")
    for section in REQUIRED_SECTIONS:
        if section == "tests":
            continue
        _require_mapping(data, section)

    analysis = _require_mapping(data, "analysis")
    analysis_id = _require_string(analysis, "id")

    paths = _require_mapping(data, "paths")
    for key in ("table_root", "feature_manifest", "output_root", "manifest_root"):
        _require_string(paths, key)

    derivation_factor = _validate_derivation(data, paths)
    derived_analysis = derivation_factor is not None

    inputs = _require_mapping(data, "inputs")
    manifest_select = _require_mapping(inputs, "manifest_select")
    contact_analysis = analysis_id in CONTACT_ANALYSES
    paired_contact_analysis = analysis_id in PAIRED_CONTACT_ANALYSES
    contact_random_effect_analysis = (
        analysis_id in CONTACT_RANDOM_EFFECT_ANALYSES
    )
    lat_stratified = analysis_id in LAT_STRATIFIED_ANALYSES
    polar_stratified = analysis_id in POLAR_STRATIFIED_ANALYSES
    expected_select = {
        "Stage": "aggregate_contact" if contact_analysis else "aggregate_region",
        "SourceStage": (
            "transform"
            if analysis_id in PRE_POLAR_ANALYSES
            else "normalize" if derived_analysis else "transform"
        ),
        "AggregationLevel": "contact" if contact_analysis else "region",
        "Representation": "scalar",
    }
    if dict(manifest_select) != expected_select:
        raise ValueError(f"inputs.manifest_select must equal {expected_select}.")
    if dict(_require_mapping(inputs, "match_policy")) != {
        "missing_whitelist": "error",
        "duplicate_whitelist": "error",
        "unlisted_candidates": "report",
    }:
        raise ValueError("inputs.match_policy does not match the stats contract.")
    if dict(_require_mapping(inputs, "row_filter")) != {
        "IncludeAggregate": True,
        "finite_columns": ["Value"],
    }:
        raise ValueError("inputs.row_filter does not match the stats contract.")

    model = _require_mapping(data, "model")
    if model.get("engine") != "lmer":
        raise ValueError("model.engine must be 'lmer'.")
    if model.get("reml") is not True:
        raise ValueError("model.reml must be true.")
    if model.get("df_method") != "kenward-roger":
        raise ValueError("model.df_method must be 'kenward-roger'.")
    if model.get("optimizer") != "default":
        raise ValueError("model.optimizer must be 'default'.")
    formula = _require_string(model, "formula")
    if analysis_id in PRE_POLAR_ANALYSES:
        expected_formula = (
            "Value ~ Lat + (1 | ID) + (1 | ID:ContactUnitID)"
            if contact_random_effect_analysis
            else "Value ~ Lat + (1 | ID)"
        )
    elif paired_contact_analysis and lat_stratified:
        expected_formula = (
            "Value ~ Phase + (1 | ID) + (1 | ID:ContactUnitID)"
        )
    elif contact_random_effect_analysis:
        expected_formula = (
            "Value ~ Phase * Lat + (1 | ID) + (1 | ID:ContactUnitID)"
        )
    elif lat_stratified or polar_stratified:
        expected_formula = "Value ~ Phase + (1 | ID)"
    else:
        expected_formula = (
            "Value ~ Phase * Polar + (1 | ID)"
            if derivation_factor == "Lat"
            else "Value ~ Phase * Lat + (1 | ID)"
        )
    if formula != expected_formula:
        raise ValueError(f"model.formula must be {expected_formula!r}.")

    model_data = _require_mapping(data, "model_data")
    factor_levels = _require_mapping(model_data, "factor_levels")
    expected_factor_names = (
        ["Phase"]
        if lat_stratified or polar_stratified
        else ["Phase", "Polar"]
        if derivation_factor == "Lat"
        else ["Phase", "Lat"]
    )
    if list(factor_levels) != expected_factor_names:
        raise ValueError(
            f"model_data.factor_levels must contain {expected_factor_names} in order."
        )
    for factor in expected_factor_names:
        _require_string_list(factor_levels, factor)
    expected_factor_columns = (
        ["ID", "ContactUnitID"]
        if contact_random_effect_analysis
        else ["ID"]
    )
    if model_data.get("factor_columns") != expected_factor_columns:
        raise ValueError(
            "model_data.factor_columns must equal "
            f"{expected_factor_columns}."
        )
    if model_data.get("require_all_factor_levels_per_model") is not True:
        raise ValueError("model_data.require_all_factor_levels_per_model must be true.")
    if model_data.get("require_complete_subjects") is not False:
        raise ValueError("model_data.require_complete_subjects must be false.")
    if model_data.get("minimum_n_ID") is not None:
        raise ValueError("model_data.minimum_n_ID must be null.")

    if paired_contact_analysis:
        contact_pairing = _require_mapping(data, "contact_pairing")
        require_both_laterality = contact_pairing.get(
            "require_both_laterality_per_id"
        )
        incomplete_policy = contact_pairing.get("incomplete")
        if not isinstance(require_both_laterality, bool):
            raise ValueError(
                "contact_pairing.require_both_laterality_per_id must be boolean."
            )
        policy = (require_both_laterality, incomplete_policy)
        allowed_policies = (
            {(True, "exclude_and_report")}
            if analysis_id == "phase-within-lat-contact-paired"
            else {
                (False, "retain_and_report"),
                (True, "exclude_and_report"),
            }
        )
        if policy not in allowed_policies:
            raise ValueError(
                "contact_pairing laterality and incomplete policies do not match "
                "the configured paired-contact analysis."
            )
        expected_contact_pairing = {
            "local_column": "Channel",
            "connectivity_column": "ChannelPair",
            "output_column": "ContactUnitID",
            "complete_factor": "Phase",
            "complete_levels": ["Pre", "Early", "Late", "Post"],
            "laterality_factor": "Lat",
            "laterality_levels": ["Ipsi", "Contra"],
            "require_both_laterality_per_id": require_both_laterality,
            "incomplete": incomplete_policy,
        }
        if dict(contact_pairing) != expected_contact_pairing:
            raise ValueError(
                "contact_pairing must equal "
                f"{expected_contact_pairing}."
            )

    strata = _require_mapping(data, "strata")
    expected_strata = ["Polar", "Lat"] if lat_stratified else ["Polar"]
    if list(strata) != expected_strata:
        raise ValueError(f"strata must contain {expected_strata} in order.")
    if _require_string_list(strata, "Polar") != [
        "Anodal",
        "Cathodal",
    ]:
        raise ValueError("strata.Polar must be [Anodal, Cathodal].")
    if lat_stratified and _require_string_list(strata, "Lat") != [
        "Ipsi",
        "Contra",
    ]:
        raise ValueError("strata.Lat must be [Ipsi, Contra].")

    grouping = _require_mapping(data, "grouping")
    connectivity_metrics = _require_mapping(
        _require_mapping(_require_mapping(data, "feature_whitelist"), "connectivity"),
        "metrics",
    )
    local_only = not connectivity_metrics
    if derivation_factor == "RegionPolar":
        local_grouping = [
            "Metric",
            "FeatureOutput",
            "Band",
            "RegionContrast",
            "Contrast",
        ]
        connectivity_grouping = []
    elif derivation_factor == "Region":
        local_grouping = [
            "Metric",
            "FeatureOutput",
            "Band",
            "RegionContrast",
            "Polar",
        ]
        connectivity_grouping = []
    else:
        local_grouping = ["Metric", "FeatureOutput", "Band", "Region"]
        connectivity_grouping = ["Metric", "FeatureOutput", "Band", "PairRegion"]
    if not derived_analysis:
        local_grouping.append("Polar")
        connectivity_grouping.append("Polar")
    if lat_stratified:
        local_grouping.append("Lat")
        connectivity_grouping.append("Lat")
    if polar_stratified:
        local_grouping.append("Polar")
        connectivity_grouping.append("Polar")
    if local_only:
        connectivity_grouping = []
    if grouping.get("local") != local_grouping:
        raise ValueError("grouping.local does not match the stats contract.")
    if grouping.get("connectivity") != connectivity_grouping:
        raise ValueError("grouping.connectivity does not match the stats contract.")

    _validate_features(
        data,
        allow_empty_connectivity=(
            local_only or derivation_factor in {"Region", "RegionPolar"}
        ),
    )
    _validate_tests(data)

    identity = _require_mapping(data, "identity")
    if identity.get("test_key_fields") != ["ModelID", "TestKind", "TestID"]:
        raise ValueError("identity.test_key_fields does not match the stats contract.")
    if identity.get("field_key_case") != "lower-kebab":
        raise ValueError("identity.field_key_case must be 'lower-kebab'.")

    outputs = _require_mapping(data, "outputs")
    inherited = _require_mapping(outputs, "inherit_source_parent")
    if dict(inherited) != {"path_column": "Path", "relative_to": "table_root"}:
        raise ValueError("outputs.inherit_source_parent does not match the contract.")
    stratum_field = (
        "Contrast"
        if derived_analysis and derivation_factor != "Region"
        else "Polar"
    )
    expected_hierarchy = [
        "source_parent",
        "Representation",
        "FeatureOutput",
        "Band",
        "RegionValue",
        stratum_field,
    ]
    if lat_stratified:
        expected_hierarchy.append("Lat")
    if polar_stratified:
        expected_hierarchy.append("Polar")
    expected_hierarchy.append("EngineDirectory")
    if outputs.get("hierarchy") != expected_hierarchy:
        raise ValueError("outputs.hierarchy does not match the stats contract.")
    if (
        outputs.get("model_fit") is not True
        or outputs.get("diagnostic_plots") is not False
    ):
        raise ValueError("outputs model-fit and diagnostic-plot policies are invalid.")

    execution = _require_mapping(data, "execution")
    if dict(execution) != {
        "on_singular": "continue",
        "on_not_estimable": "record",
        "on_error": "fail",
    }:
        raise ValueError("execution policies do not match the stats contract.")

    manifest_root = Path(paths["manifest_root"])
    output_root = Path(paths["output_root"])
    try:
        manifest_root.relative_to(output_root)
    except ValueError as error:
        raise ValueError(
            "paths.manifest_root must be below paths.output_root."
        ) from error


def load_stats_config(path: Path | str) -> StatsConfig:
    """Parse one stats YAML file and validate it once at the boundary."""

    resolved = Path(path).expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise TypeError(f"Configuration must contain a mapping: {resolved}")
    _validate_config(data, resolved)
    return StatsConfig(path=resolved, data=data)
