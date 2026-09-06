"""Load and validate visualization YAML at the file boundary."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

COMMON_SECTIONS = (
    "visualization",
    "paths",
    "inputs",
    "figures",
    "factor_levels",
    "labels",
    "style",
    "outputs",
    "manifests",
    "execution",
)
CLASSIC_PALETTE = [
    "#F2000E",
    "#0E6AAF",
    "#0CA228",
    "#E87000",
    "#884392",
]
PHASE_LEVELS = ["Pre", "Early", "Late", "Post"]
LAT_LEVELS = ["Ipsi", "Contra"]
REGION_LABELS = {
    "SNr": "SNr",
    "STN": "STN",
    "SNr-STN": "SNr–STN",
    "SNr→STN": "SNr→STN",
}
SUBMISSION_PHASE_SCALAR_ID = "submission-phase-contact-scalar"
SUBMISSION_POLAR_SCALAR_ID = "submission-polar-contact-scalar"
SUBMISSION_PRE_SCALAR_ID = "submission-pre-contact-scalar"
SUBMISSION_LAT_SCALAR_ID = "submission-lat-region-scalar"
SUBMISSION_LAT_TRAJECTORY_ID = "submission-lat-trajectory"
SUBMISSION_POLAR_TRAJECTORY_ID = "submission-polar-trajectory"
SUBMISSION_REGION_TRAJECTORY_ID = "submission-region-trajectory"
SUBMISSION_REGION_SCALAR_ID = "submission-region-scalar"
SUBMISSION_REGION_HEATMAP_ID = "submission-region-signed-logp-heatmap"
SUBMISSION_PRE_HEATMAP_ID = "submission-pre-contact-signed-logp-heatmap"
SUBMISSION_CLINICAL_CORRELATION_HEATMAP_ID = "submission-clinical-correlation-heatmap"
SUBMISSION_CLINICAL_CORRELATION_SCALE_HEATMAP_ID = (
    "submission-clinical-correlation-scale-heatmap"
)
SUBMISSION_CLINICAL_CORRELATION_FIT_ID = "submission-clinical-correlation-fit"
SUBMISSION_CLINICAL_CORRELATION_ADJACENT_PHASE_FIT_ID = (
    "submission-clinical-correlation-adjacent-phase-fit"
)


@dataclass(frozen=True)
class VizConfig(Mapping[str, Any]):
    """Validated study visualization configuration."""

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
        return str(self.data["visualization"]["id"])

    @property
    def kind(self) -> str:
        return str(self.data["visualization"]["kind"])

    @property
    def stats_analysis_id(self) -> str:
        return str(self.data["inputs"].get("stats_analysis_id", ""))

    @property
    def table_root(self) -> Path:
        return Path(self.data["paths"]["table_root"])

    @property
    def source_root(self) -> Path:
        paths = self.data["paths"]
        return Path(paths.get("source_root", paths.get("table_root")))

    @property
    def stats_root(self) -> Path:
        return Path(self.data["paths"]["stats_root"])

    @property
    def models_manifest(self) -> Path:
        return Path(self.data["paths"]["models_manifest"])

    @property
    def stats_outputs_manifest(self) -> Path:
        return Path(self.data["paths"]["stats_outputs_manifest"])

    @property
    def tests_manifest(self) -> Path:
        return Path(self.data["paths"]["tests_manifest"])

    @property
    def contact_pairing_manifest(self) -> Path:
        return Path(self.data["paths"]["contact_pairing_manifest"])

    @property
    def derived_tables_manifest(self) -> Path:
        return Path(self.data["paths"]["derived_tables_manifest"])

    @property
    def correlations_path(self) -> Path:
        return Path(self.data["paths"]["correlations"])

    @property
    def fit_points_path(self) -> Path:
        return Path(self.data["paths"]["fit_points"])

    @property
    def feature_outputs_manifest(self) -> Path:
        return Path(self.data["paths"]["feature_outputs_manifest"])

    @property
    def output_root(self) -> Path:
        return Path(self.data["paths"]["output_root"])

    @property
    def manifest_root(self) -> Path:
        return Path(self.data["paths"]["manifest_root"])


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
        raise TypeError(
            f"Configuration key {key!r} must be a non-empty list of strings."
        )
    if len(value) != len(set(value)):
        raise ValueError(f"Configuration key {key!r} contains duplicate values.")
    return value


def _validate_common(data: dict[str, Any], path: Path) -> str:
    missing = [section for section in COMMON_SECTIONS if section not in data]
    if missing:
        raise KeyError(f"Missing configuration sections in {path}: {missing}")
    for section in COMMON_SECTIONS:
        if section != "figures":
            _mapping(data, section)
    visualization = _mapping(data, "visualization")
    _string(visualization, "id")
    kind = _string(visualization, "kind")
    if kind not in {
        "scalar",
        "spectral",
        "trace",
        "raw",
        "signed_logp_heatmap",
        "correlation_heatmap",
        "correlation_fit",
        "programming_intensity",
        "lat_trajectory",
        "polar_trajectory",
        "region_trajectory",
    }:
        raise ValueError(f"Unsupported visualization kind: {kind}")
    levels = _mapping(data, "factor_levels")
    phase_levels = _string_list(levels, "Phase")
    if kind in {"spectral", "trace", "raw"} and phase_levels != PHASE_LEVELS:
        raise ValueError(f"factor_levels.Phase must be {PHASE_LEVELS}.")
    if (
        kind in {"spectral", "trace", "raw"}
        and _string_list(levels, "Lat") != LAT_LEVELS
    ):
        raise ValueError(f"factor_levels.Lat must be {LAT_LEVELS}.")
    labels = _mapping(data, "labels")
    if dict(_mapping(labels, "polarity")) != {"Anodal": "⊕", "Cathodal": "⊖"}:
        raise ValueError("labels.polarity must map Anodal/Cathodal to ⊕/⊖ symbols.")
    region_labels = dict(_mapping(labels, "region"))
    if any(region_labels.get(key) != value for key, value in REGION_LABELS.items()):
        raise ValueError("labels.region does not match the canonical display labels.")
    paths = _mapping(data, "paths")
    output_root = Path(_string(paths, "output_root"))
    manifest_root = Path(_string(paths, "manifest_root"))
    try:
        manifest_root.relative_to(output_root)
    except ValueError as error:
        raise ValueError(
            "paths.manifest_root must be below paths.output_root."
        ) from error
    outputs = _mapping(data, "outputs")
    if outputs.get("format") != "pdf" or outputs.get("formal_png") is not False:
        raise ValueError("Formal output must contain PDF files only.")
    if outputs.get("hashes") is not False:
        raise ValueError("Visualization hashes must remain disabled.")
    _string(outputs, "filename_template")
    if dict(_mapping(data, "execution")) != {
        "on_error": "fail",
        "retry": False,
        "cache": False,
        "resume": False,
    }:
        raise ValueError("execution does not match the failure contract.")
    return kind


def _validate_scalar(data: dict[str, Any]) -> None:
    for section in ("sample_size", "significance"):
        _mapping(data, section)
    inputs = _mapping(data, "inputs")
    test_kind = inputs.get("test_kind")
    stats_analysis_id = inputs.get("stats_analysis_id")
    fixed_lat_pairwise = (
        test_kind == "emm_pairwise" and stats_analysis_id == "phase-within-lat"
    )
    submission_contact_pairwise = (
        test_kind == "emm_pairwise"
        and stats_analysis_id == "phase-by-lat-contact-paired"
        and _mapping(data, "visualization").get("id") == SUBMISSION_PHASE_SCALAR_ID
    )
    submission_polar_null = (
        test_kind == "emmean_vs_null"
        and stats_analysis_id == "polarity-by-phase-lat-contact-paired"
        and _mapping(data, "visualization").get("id") == SUBMISSION_POLAR_SCALAR_ID
    )
    submission_pre_null = (
        test_kind == "emmean_vs_null"
        and stats_analysis_id == "pre-by-polar-contact-paired"
        and _mapping(data, "visualization").get("id") == SUBMISSION_PRE_SCALAR_ID
    )
    submission_lat_null = (
        test_kind == "emmean_vs_null"
        and stats_analysis_id == "laterality-within-polar"
        and _mapping(data, "visualization").get("id") == SUBMISSION_LAT_SCALAR_ID
    )
    submission_region_null = (
        test_kind == "emmean_vs_null"
        and stats_analysis_id == "region-by-phase-lat"
        and _mapping(data, "visualization").get("id") == SUBMISSION_REGION_SCALAR_ID
    )
    submission_scalar = (
        submission_contact_pairwise
        or submission_polar_null
        or submission_pre_null
        or submission_lat_null
        or submission_region_null
    )
    factor_levels = _mapping(data, "factor_levels")
    if test_kind == "emm_pairwise":
        if list(factor_levels) != ["Phase", "Lat"]:
            raise ValueError("Pairwise scalar factors must be Phase and Lat.")
        if _string_list(factor_levels, "Lat") != LAT_LEVELS:
            raise ValueError(f"factor_levels.Lat must be {LAT_LEVELS}.")
        if submission_contact_pairwise:
            expected_figures = [
                {
                    "id": "Phase-Lat-split",
                    "test_id": "Phase-Lat",
                    "layout": "laterality_singletons",
                    "x_var": "Phase",
                    "split_var": "Lat",
                    "split_levels": LAT_LEVELS,
                },
                {
                    "id": "Phase-Lat-faceted",
                    "test_id": "Phase-Lat",
                    "layout": "laterality_rows",
                    "x_var": "Phase",
                    "row_var": "Lat",
                    "row_levels": LAT_LEVELS,
                },
            ]
        elif fixed_lat_pairwise:
            expected_figures = [
                {
                    "id": "Phase",
                    "test_id": "Phase",
                    "x_var": "Phase",
                    "row_var": "Lat",
                    "row_levels_from_model": True,
                }
            ]
        else:
            expected_figures = [
                {"id": "Phase", "test_id": "Phase", "x_var": "Phase"},
                {
                    "id": "Phase-Lat",
                    "test_id": "Phase-Lat",
                    "x_var": "Phase",
                    "row_var": "Lat",
                    "row_levels": LAT_LEVELS,
                },
            ]
    else:
        pre_polar = inputs.get("stats_analysis_id") in {
            "pre-by-polar",
            "pre-by-polar-contact-paired",
        }
        conditional_factors = [name for name in factor_levels if name != "Phase"]
        if len(conditional_factors) != 1 or conditional_factors[0] not in {
            "Lat",
            "Polar",
        }:
            raise ValueError(
                "Null-test scalar factors must be Phase and one of Lat or Polar."
            )
        conditional = conditional_factors[0]
        expected_levels = LAT_LEVELS if conditional == "Lat" else ["Anodal", "Cathodal"]
        if _string_list(factor_levels, conditional) != expected_levels:
            raise ValueError(f"factor_levels.{conditional} must be {expected_levels}.")
        if submission_polar_null or submission_region_null:
            expected_figures = [
                {
                    "id": "Phase-Lat-split",
                    "test_id": "Phase-Lat",
                    "layout": "laterality_singletons",
                    "x_var": "Phase",
                    "split_var": "Lat",
                    "split_levels": LAT_LEVELS,
                },
                {
                    "id": "Phase-Lat-faceted",
                    "test_id": "Phase-Lat",
                    "layout": "laterality_rows",
                    "x_var": "Phase",
                    "row_var": "Lat",
                    "row_levels": LAT_LEVELS,
                },
            ]
        elif submission_lat_null:
            expected_figures = [
                {
                    "id": "Phase-Polar-split",
                    "test_id": "Phase",
                    "layout": "polarity_singletons",
                    "x_var": "Phase",
                    "split_var": "Polar",
                    "split_levels": ["Anodal", "Cathodal"],
                },
                {
                    "id": "Phase-Polar-faceted",
                    "test_id": "Phase",
                    "layout": "polarity_rows",
                    "x_var": "Phase",
                    "row_var": "Polar",
                    "row_levels": ["Anodal", "Cathodal"],
                },
            ]
        elif pre_polar:
            expected_figures = [{"id": "Lat", "test_id": "Lat", "x_var": "Lat"}]
        else:
            test_id = f"Phase-{conditional}"
            expected_figures = [
                {
                    "id": test_id,
                    "test_id": test_id,
                    "x_var": "Phase",
                    "row_var": conditional,
                    "row_levels": expected_levels,
                }
            ]
    figures = data.get("figures")
    if not isinstance(figures, list) or not figures:
        raise ValueError(
            f"figures must be a non-empty subset of the registered layout: "
            f"{expected_figures}"
        )
    try:
        figure_indices = [expected_figures.index(figure) for figure in figures]
    except ValueError as error:
        raise ValueError(
            f"figures must be a non-empty subset of the registered layout: "
            f"{expected_figures}"
        ) from error
    if figure_indices != sorted(set(figure_indices)):
        raise ValueError(
            f"figures must preserve the registered layout order without duplicates: "
            f"{expected_figures}"
        )
    paths = _mapping(data, "paths")
    required_path_keys = [
        "stats_root",
        "models_manifest",
        "stats_outputs_manifest",
    ]
    if submission_contact_pairwise:
        required_path_keys.extend(["tests_manifest", "contact_pairing_manifest"])
    elif (
        submission_polar_null
        or submission_pre_null
        or submission_lat_null
        or submission_region_null
    ):
        required_path_keys.extend(["tests_manifest", "derived_tables_manifest"])
    for key in required_path_keys:
        _string(paths, key)
    if "source_root" in paths:
        _string(paths, "source_root")
    else:
        _string(paths, "table_root")
    stats_root = Path(paths["stats_root"])
    for key in required_path_keys[1:]:
        try:
            Path(paths[key]).relative_to(stats_root)
        except ValueError as error:
            raise ValueError(f"paths.{key} must be below paths.stats_root.") from error
    _string(inputs, "stats_analysis_id")
    observation_level = inputs.get("observation_level", "region")
    if observation_level not in {"region", "contact"}:
        raise ValueError("inputs.observation_level must be region or contact.")
    if "model_filter" in inputs and "model_filters" in inputs:
        raise ValueError("Use either inputs.model_filter or inputs.model_filters.")
    if "model_filter" in inputs:
        model_filter = _mapping(inputs, "model_filter")
        expected_filter_keys = {
            "Domain",
            "Metric",
            "FeatureOutput",
            "Band",
            "RegionValue",
            "Polar",
        }
        if set(model_filter) != expected_filter_keys:
            raise ValueError(
                "inputs.model_filter must contain exactly Domain, Metric, "
                "FeatureOutput, Band, RegionValue, and Polar."
            )
        for key in expected_filter_keys:
            _string(model_filter, key)
    if "model_filters" in inputs:
        if not fixed_lat_pairwise:
            raise ValueError(
                "inputs.model_filters is registered only for phase-within-lat."
            )
        model_filters = inputs["model_filters"]
        if not isinstance(model_filters, list) or not model_filters:
            raise TypeError("inputs.model_filters must be a non-empty list.")
        expected_filter_keys = {
            "Domain",
            "Metric",
            "FeatureOutput",
            "Band",
            "RegionValue",
            "Polar",
            "Lat",
        }
        for model_filter in model_filters:
            if not isinstance(model_filter, Mapping):
                raise TypeError("Each inputs.model_filters item must be a mapping.")
            if set(model_filter) != expected_filter_keys:
                raise ValueError(
                    "Each inputs.model_filters item must contain exactly Domain, "
                    "Metric, FeatureOutput, Band, RegionValue, Polar, and Lat."
                )
            for key in expected_filter_keys:
                _string(model_filter, key)
        identities = [
            tuple(item[key] for key in sorted(item)) for item in model_filters
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("inputs.model_filters contains duplicate model filters.")
    if inputs.get("accepted_model_statuses") != [
        "ok",
        "singular",
        "convergence_warning",
    ]:
        raise ValueError("inputs.accepted_model_statuses does not match the contract.")
    if test_kind not in {"emm_pairwise", "emmean_vs_null"}:
        raise ValueError("inputs.test_kind must be emm_pairwise or emmean_vs_null.")
    phase_levels = _string_list(factor_levels, "Phase")
    expected_phases = (
        PHASE_LEVELS
        if test_kind == "emm_pairwise"
        else (
            ["Pre"]
            if inputs.get("stats_analysis_id")
            in {"pre-by-polar", "pre-by-polar-contact-paired"}
            else ["Early", "Late", "Post"]
        )
    )
    if phase_levels != expected_phases:
        raise ValueError(f"factor_levels.Phase must be {expected_phases}.")
    if dict(_mapping(inputs, "row_filter")) != {
        "IncludeAggregate": True,
        "finite_columns": ["Value"],
    }:
        raise ValueError("inputs.row_filter does not match the stats boundary.")
    labels = _mapping(data, "labels")
    if dict(_mapping(labels, "bands")) != {
        "delta": "δ",
        "theta": "θ",
        "alpha": "α",
        "beta_low": "low-β",
        "beta_high": "high-β",
        "gamma_low": "low-γ",
        "gamma_high": "high-γ",
    }:
        raise ValueError("labels.bands does not match the configured band labels.")
    y_axis = _mapping(labels, "y_axis")
    for metric in (
        "aperiodic",
        "periodic",
        "raw_power",
        "burst",
        "ciplv",
        "imcoh_abs",
        "wpli",
        "psi",
        "trgc",
    ):
        _mapping(y_axis, metric)
    expected_aperiodic = (
        {
            "exponent": "ΔAperiodic exponent: SNr − STN",
            "offset": "ΔAperiodic offset: SNr − STN",
            "knee": "ΔAperiodic knee: SNr − STN",
        }
        if submission_region_null
        else {
            "exponent": "Aperiodic exponent",
            "offset": "Aperiodic offset",
            "knee": "Aperiodic knee",
        }
    )
    if dict(_mapping(y_axis, "aperiodic")) != expected_aperiodic:
        raise ValueError("labels.y_axis.aperiodic does not match the contract.")
    if test_kind == "emmean_vs_null":
        contrast_labels = dict(_mapping(labels, "contrast"))
        allowed_contrasts = (
            {"Anodal-minus-Cathodal": "⊕−⊖"},
            {"Ipsi-minus-Contra": "Ipsi−Contra"},
            {"SNr-minus-STN": "SNr−STN"},
        )
        if contrast_labels not in allowed_contrasts:
            raise ValueError("labels.contrast does not match a registered contrast.")
        prefix = labels.get("contrast_y_axis_prefix")
        if "Ipsi-minus-Contra" in contrast_labels and prefix != {
            "Ipsi-minus-Contra": "Ipsi−Contra "
        }:
            raise ValueError(
                "labels.contrast_y_axis_prefix must identify Ipsi-minus-Contra."
            )
    style = _mapping(data, "style")
    font = _mapping(style, "font")
    expected_font = "Arial Unicode MS"
    if font.get("latin") != expected_font or font.get("greek") != expected_font:
        raise ValueError(
            f"Visualization Latin and Greek text must use {expected_font}."
        )
    if list(_mapping(style, "phase_palette")) != phase_levels:
        raise ValueError("style.phase_palette must follow the configured Phase order.")
    widths = _mapping(style, "line_width_pt")
    required_widths = {
        "default",
        "emm_line",
        "emm_ci",
        "box_edge",
        "median",
        "raw_mean",
        "whisker",
        "cap",
        "significance_bracket",
    }
    if set(widths) != required_widths:
        raise ValueError("style.line_width_pt has missing or unsupported elements.")
    if not all(
        isinstance(value, (int, float)) and value > 0 for value in widths.values()
    ):
        raise TypeError("All configured line widths must be positive numbers.")
    if widths.get("emm_line") != 0.75 or widths.get("emm_ci") != 1.00:
        raise ValueError("EMM line and CI widths must be 0.75 and 1.00 pt.")
    emm = _mapping(style, "emm")
    if emm.get("mean_marker") != "o" or emm.get("mean_marker_size_pt") != 2:
        raise ValueError("EMM means must use centered 2 pt circular markers.")
    if emm.get("mean_marker_edge_width_pt") != 0:
        raise ValueError("EMM mean-marker edges must be disabled.")
    if emm.get("ci_cap_pt") != 0:
        raise ValueError("EMM confidence-interval caps must be disabled.")
    y_limits = _mapping(style, "y_limits")
    expected_y_limits = {
        "lower_padding_fraction": 0.15,
        "upper_padding_fraction": 0.30,
        "share_within_figure": True,
    }
    if submission_scalar:
        expected_y_limits["share_across_lat_layouts"] = True
    if dict(y_limits) != expected_y_limits:
        raise ValueError("style.y_limits does not match the scalar contract.")
    if style.get("transparent") is not True or style.get("dpi") != 600:
        raise ValueError("Formal visualization must be transparent at 600 dpi.")
    significance = _mapping(data, "significance")
    expected_p = (
        "p_tukey"
        if test_kind == "emm_pairwise"
        else (
            "p_holm"
            if (
                submission_polar_null
                or submission_pre_null
                or submission_lat_null
                or submission_region_null
            )
            else "p_raw"
        )
    )
    if significance.get("p_column") != expected_p:
        raise ValueError(f"significance.p_column must be {expected_p}.")
    if significance.get("hide_non_significant") is not True:
        raise ValueError("Non-significant brackets must be hidden.")
    if significance.get("thresholds") != [
        {"p_lt": 0.001, "label": "***"},
        {"p_lt": 0.01, "label": "**"},
        {"p_lt": 0.05, "label": "*"},
    ]:
        raise ValueError("significance.thresholds must define one to three stars.")
    if test_kind == "emmean_vs_null":
        if significance.get("mode") != "emmean_vs_null":
            raise ValueError("significance.mode must be emmean_vs_null.")
        if significance.get("marker_offset_fraction") != 0.035:
            raise ValueError("significance.marker_offset_fraction must be 0.035.")
        reference = _mapping(data, "reference_line")
        if dict(reference) != {
            "value": 0,
            "color": "#9A9A9A",
            "line_style": ":",
            "line_width_pt": 0.60,
            "alpha": 0.80,
            "zorder": 8,
        }:
            raise ValueError("reference_line does not match the null-test contract.")
    sample = _mapping(data, "sample_size")
    if sample.get("show") is not True or sample.get("id_column") != "ID":
        raise ValueError("sample_size must display unique ID counts.")
    if sample.get("template") != "{n_ID}" or sample.get("y_axes") != 0.00:
        raise ValueError("sample_size placement does not match the contract.")
    if sample.get("vertical_alignment") != "bottom":
        raise ValueError("sample_size.vertical_alignment must be bottom.")
    if submission_scalar:
        expected_observation_level = (
            "region" if submission_lat_null or submission_region_null else "contact"
        )
        if inputs.get("observation_level") != expected_observation_level:
            raise ValueError(
                "Submission scalar inputs.observation_level must match its "
                "registered analysis level."
            )
        outputs = _mapping(data, "outputs")
        if outputs.get("directory_fields") != ["OutputGroup"]:
            raise ValueError(
                "Submission scalar outputs.directory_fields must be " "[OutputGroup]."
            )
        if submission_contact_pairwise:
            expected_filename = "{Band}_{RegionValue}_{Polar}_{LatDisplay}.pdf"
        elif submission_polar_null:
            expected_filename = "{Band}_{RegionValue}_{Contrast}_{LatDisplay}.pdf"
        elif submission_pre_null:
            expected_filename = "{Band}_{RegionValue}_{Contrast}.pdf"
        elif submission_region_null:
            expected_filename = "{Band}_{RegionValue}_{Polar}_{LatDisplay}.pdf"
        else:
            expected_filename = "{Band}_{RegionValue}_{Contrast}_{PolarDisplay}.pdf"
        if outputs.get("filename_template") != expected_filename:
            raise ValueError(
                "Submission scalar filename_template does not match its identity."
            )
        if Path(paths["manifest_root"]) != Path(paths["output_root"]):
            raise ValueError(
                "Submission scalar manifests must be written at the scalar root."
            )
        expected_manifests = {
            "figures": "figures.csv",
            "coverage": "coverage.csv",
            "statistics": (
                "statistics.csv" if submission_contact_pairwise else "tables.csv"
            ),
        }
        if dict(_mapping(data, "manifests")) != expected_manifests:
            raise ValueError("Submission scalar manifests are invalid.")
        statistics = _mapping(data, "statistics")
        if statistics.get("enabled") is not True:
            raise ValueError("Submission scalar statistics must be enabled.")
        expected_p_column = (
            "p_tukey"
            if submission_contact_pairwise
            else "p_holm" if submission_pre_null or submission_region_null else "p_raw"
        )
        if statistics.get("p_column") != expected_p_column:
            raise ValueError(
                f"Submission scalar statistics must use {expected_p_column}."
            )
        if statistics.get("significance_alpha") != 0.05:
            raise ValueError("Submission scalar significance_alpha must equal 0.05.")
        expected_files = (
            {
                "models": "models.csv",
                "tests": "tests.csv",
                "model_fit": "model-fit.csv",
                "omnibus": "omnibus.csv",
                "emmeans": "emmeans.csv",
                "tukey": "tukey.csv",
                "significant_tukey": "significant-tukey.csv",
                "contact_pairing": "contact-pairing.csv",
            }
            if submission_contact_pairwise
            else {
                "models": "models.csv",
                "tests": "tests.csv",
                "model_fit": "model-fit.csv",
                "omnibus": "omnibus.csv",
                "emmeans": "emmeans.csv",
                "null_test": "statistics.csv",
                "significant_raw": "significant-raw.csv",
                "significant_holm": "significant-holm.csv",
                "pairing": "pairing.csv",
            }
        )
        if dict(_mapping(statistics, "files")) != expected_files:
            raise ValueError("Submission scalar statistics files are invalid.")


def _validate_lat_trajectory(data: dict[str, Any]) -> None:
    """Validate a registered paired-component trajectory contract."""

    visualization = _mapping(data, "visualization")
    kind = visualization.get("kind")
    if kind == "lat_trajectory":
        expected_id = SUBMISSION_LAT_TRAJECTORY_ID
        expected_inputs = {
            "stats_analysis_id": "laterality-within-polar",
            "accepted_model_statuses": ["ok", "singular", "convergence_warning"],
            "selection": {"column": "p_holm", "operator": "lt", "value": 0.05},
            "fit_phases": ["Early", "Late", "Post"],
            "anchor_phase": "Pre",
            "component_columns": {"Ipsi": "IpsiValue", "Contra": "ContraValue"},
            "component_difference": {"minuend": "Ipsi", "subtrahend": "Contra"},
            "difference_tolerance": 1.0e-10,
            "source_strata_fields": ["Polar"],
            "figure_condition_fields": ["Polar"],
            "row_filter": {
                "IncludeAggregate": True,
                "finite_columns": ["IpsiValue", "ContraValue"],
            },
        }
        expected_figures = [
            {
                "id": "Phase-Lat-overlay",
                "test_id": "Phase",
                "layout": "laterality_overlay",
                "x_var": "Phase",
                "line_var": "Lat",
                "condition_var": "Polar",
            }
        ]
        expected_levels = {
            "Phase": ["Pre", "Early", "Late", "Post"],
            "Lat": LAT_LEVELS,
        }
        expected_model = {
            "formula": "Value ~ Phase * Lat + (1 | ID)",
            "factor_columns": ["ID"],
            "emmeans": {"x_var": "Phase", "panel_var": "Lat", "facet_var": None},
            "reml": True,
            "df_method": "kenward-roger",
            "confidence_level": 0.95,
        }
        expected_series = {
            "order": LAT_LEVELS,
            "Ipsi": {"color": "#F2000E", "line_style": "-", "zorder": 4},
            "Contra": {"color": "#0E6AAF", "line_style": "-", "zorder": 8},
            "mean_marker": "o",
            "mean_marker_size_pt": 2,
            "mean_marker_edge_width_pt": 0,
            "ci_cap_pt": 2,
        }
        expected_filename = "{Band}_{RegionValue}_{Polar}_Ipsi-Contra.pdf"
    elif kind == "polar_trajectory":
        expected_id = SUBMISSION_POLAR_TRAJECTORY_ID
        expected_inputs = {
            "stats_analysis_id": "polarity-by-phase-lat-contact-paired",
            "accepted_model_statuses": ["ok", "singular", "convergence_warning"],
            "selection": {"column": "p_holm", "operator": "lt", "value": 0.05},
            "fit_phases": ["Early", "Late", "Post"],
            "anchor_phase": "Pre",
            "component_columns": {
                "Anodal": "AnodalValue",
                "Cathodal": "CathodalValue",
            },
            "component_difference": {
                "minuend": "Anodal",
                "subtrahend": "Cathodal",
            },
            "difference_tolerance": 1.0e-5,
            "source_strata_fields": [],
            "figure_condition_fields": ["Lat"],
            "row_filter": {
                "IncludeAggregate": True,
                "finite_columns": ["AnodalValue", "CathodalValue"],
            },
        }
        expected_figures = [
            {
                "id": "Phase-Polar-overlay",
                "test_id": "Phase",
                "layout": "polarity_overlay",
                "x_var": "Phase",
                "line_var": "Polar",
                "condition_var": "Lat",
            }
        ]
        expected_levels = {
            "Phase": ["Pre", "Early", "Late", "Post"],
            "Lat": LAT_LEVELS,
            "Polar": ["Anodal", "Cathodal"],
        }
        expected_model = {
            "formula": (
                "Value ~ Phase * Lat * Polar + (1 | ComponentPairID) + "
                "(1 | ID) + (0 + ComponentContrast | ID) + "
                "(0 + ComponentContrast | ID:ContactUnitID)"
            ),
            "factor_columns": ["ID", "ContactUnitID"],
            "emmeans": {
                "x_var": "Phase",
                "panel_var": "Polar",
                "facet_var": "Lat",
            },
            "reml": True,
            "df_method": "kenward-roger",
            "confidence_level": 0.95,
        }
        expected_series = {
            "order": ["Anodal", "Cathodal"],
            "Anodal": {"color": "#F2000E", "line_style": "-", "zorder": 4},
            "Cathodal": {"color": "#0E6AAF", "line_style": "-", "zorder": 8},
            "mean_marker": "o",
            "mean_marker_size_pt": 2,
            "mean_marker_edge_width_pt": 0,
            "ci_cap_pt": 2,
        }
        expected_filename = "{Band}_{RegionValue}_{Lat}_Anodal-Cathodal.pdf"
    elif kind == "region_trajectory":
        expected_id = SUBMISSION_REGION_TRAJECTORY_ID
        expected_inputs = {
            "stats_analysis_id": "region-by-phase-lat",
            "accepted_model_statuses": ["ok", "singular", "convergence_warning"],
            "selection": {"column": "p_holm", "operator": "lt", "value": 0.05},
            "fit_phases": ["Early", "Late", "Post"],
            "anchor_phase": "Pre",
            "anchor_pair_columns": [
                "ID",
                "Record",
                "Trial",
                "Polar",
                "Phase",
                "Lat",
                "Band",
            ],
            "component_columns": {"SNr": "SNrValue", "STN": "STNValue"},
            "component_difference": {"minuend": "SNr", "subtrahend": "STN"},
            "difference_tolerance": 1.0e-5,
            "source_strata_fields": ["Polar"],
            "figure_condition_fields": ["Polar", "Lat"],
            "row_filter": {
                "IncludeAggregate": True,
                "finite_columns": ["SNrValue", "STNValue"],
            },
        }
        expected_figures = [
            {
                "id": "Phase-Region-overlay",
                "test_id": "Phase",
                "layout": "region_overlay",
                "x_var": "Phase",
                "line_var": "Region",
                "condition_vars": ["Polar", "Lat"],
            }
        ]
        expected_levels = {
            "Phase": ["Pre", "Early", "Late", "Post"],
            "Lat": LAT_LEVELS,
            "Region": ["SNr", "STN"],
        }
        expected_model = {
            "formula": (
                "Value ~ Phase * Lat * Region + (1 | ComponentPairID) + "
                "(1 | ID) + (0 + ComponentContrast | ID)"
            ),
            "factor_columns": ["ID"],
            "emmeans": {
                "x_var": "Phase",
                "panel_var": "Region",
                "facet_var": "Lat",
            },
            "reml": True,
            "df_method": "kenward-roger",
            "confidence_level": 0.95,
        }
        expected_series = {
            "order": ["SNr", "STN"],
            "SNr": {"color": "#F2000E", "line_style": "-", "zorder": 4},
            "STN": {"color": "#0E6AAF", "line_style": "-", "zorder": 8},
            "mean_marker": "o",
            "mean_marker_size_pt": 2,
            "mean_marker_edge_width_pt": 0,
            "ci_cap_pt": 2,
        }
        expected_filename = "{Band}_{Polar}_{Lat}_SNr-STN.pdf"
    else:
        raise ValueError("Unsupported paired-component trajectory kind.")
    if visualization.get("id") != expected_id:
        raise ValueError("Paired-component trajectory visualization id is invalid.")

    paths = _mapping(data, "paths")
    required_paths = {
        "source_root",
        "stats_root",
        "statistics",
        "pairing",
        "runtime_root",
        "output_root",
        "manifest_root",
    }
    if set(paths) != required_paths:
        raise ValueError("Lat trajectory paths do not match the registered contract.")
    for key in required_paths:
        _string(paths, key)
    stats_root = Path(paths["stats_root"])
    for key in ("statistics", "pairing"):
        try:
            Path(paths[key]).relative_to(stats_root)
        except ValueError as error:
            raise ValueError(f"paths.{key} must be below paths.stats_root.") from error
    if Path(paths["manifest_root"]) != Path(paths["output_root"]):
        raise ValueError("Lat trajectory manifests must be written at the output root.")

    inputs = _mapping(data, "inputs")
    if dict(inputs) != expected_inputs:
        raise ValueError("Trajectory inputs do not match the paired stats contract.")

    if data.get("figures") != expected_figures:
        raise ValueError("Trajectory figure layout is invalid.")
    if dict(_mapping(data, "factor_levels")) != expected_levels:
        raise ValueError("Trajectory factor levels are invalid.")

    model = _mapping(data, "model")
    if dict(model) != expected_model:
        raise ValueError("Trajectory component model is invalid.")

    labels = _mapping(data, "labels")
    if dict(_mapping(labels, "bands")) != {
        "delta": "δ",
        "theta": "θ",
        "alpha": "α",
        "beta_low": "low-β",
        "beta_high": "high-β",
        "gamma_low": "low-γ",
        "gamma_high": "high-γ",
    }:
        raise ValueError("Lat trajectory band labels are invalid.")
    y_axis = _mapping(labels, "y_axis")
    for metric in (
        "aperiodic",
        "periodic",
        "raw_power",
        "burst",
        "ciplv",
        "imcoh_abs",
        "wpli",
        "psi",
        "trgc",
    ):
        _mapping(y_axis, metric)
    if labels.get("delta_prefix") != "Δ ":
        raise ValueError("Lat trajectory y-axis delta prefix must be 'Δ '.")

    style = _mapping(data, "style")
    expected_font = {
        "latin": "Arial Unicode MS",
        "greek": "Arial Unicode MS",
        "axis_pt": 7,
        "strip_pt": 7,
        "legend_pt": 6,
        "significance_pt": 7,
        "tick_pt": 6,
    }
    if kind in {"lat_trajectory", "polar_trajectory"}:
        expected_font["sample_size_pt"] = 6
        if dict(_mapping(data, "sample_size")) != {
            "show": True,
            "template": "{n_1}/{n_2}",
            "color": "#2C2C2C",
            "y_axes": 0.00,
            "vertical_alignment": "bottom",
        }:
            raise ValueError("Trajectory sample-size style is invalid.")
    if dict(_mapping(style, "font")) != expected_font:
        raise ValueError("Lat trajectory font style is invalid.")
    if dict(_mapping(style, "line_width_pt")) != {
        "default": 0.60,
        "emm_line": 0.75,
        "emm_ci": 1.00,
    }:
        raise ValueError("Lat trajectory line widths are invalid.")
    if dict(_mapping(style, "series")) != expected_series:
        raise ValueError("Trajectory series style is invalid.")
    if dict(_mapping(style, "geometry_mm")) != {
        "phase_width": 6,
        "panel_height": 25,
        "strip_top_height": 4.5,
        "strip_pad": 2,
        "x_label_offset": 5,
        "y_label_offset": 7,
    }:
        raise ValueError("Lat trajectory geometry is invalid.")
    if dict(_mapping(style, "strips")) != {
        "top_background": "#D7E3E0",
        "text_color": "black",
        "font_weight": "bold",
    }:
        raise ValueError("Lat trajectory strip style is invalid.")
    if dict(_mapping(style, "y_limits")) != {
        "lower_padding_fraction": 0.15,
        "upper_padding_fraction": 0.30,
    }:
        raise ValueError("Lat trajectory y-limit style is invalid.")
    if dict(_mapping(style, "axes")) != {
        "grid": False,
        "show_top_right_spines": False,
    }:
        raise ValueError("Lat trajectory axes style is invalid.")
    if dict(_mapping(style, "legend")) != {
        "location": "outside_right",
        "frame": False,
        "order": expected_series["order"],
    }:
        raise ValueError("Trajectory legend style is invalid.")
    if style.get("transparent") is not True or style.get("dpi") != 600:
        raise ValueError("Lat trajectory output must be transparent at 600 dpi.")

    significance = _mapping(data, "significance")
    if dict(significance) != {
        "p_column": "p_holm",
        "hide_non_significant": True,
        "thresholds": [
            {"p_lt": 0.001, "label": "***"},
            {"p_lt": 0.01, "label": "**"},
            {"p_lt": 0.05, "label": "*"},
        ],
        "star_offset_fraction": 0.07,
    }:
        raise ValueError("Lat trajectory significance style is invalid.")
    if dict(_mapping(data, "reference_line")) != {
        "value": 0,
        "color": "#9A9A9A",
        "line_style": ":",
        "line_width_pt": 0.60,
        "alpha": 0.80,
        "zorder": 2,
    }:
        raise ValueError("Lat trajectory reference line is invalid.")

    outputs = _mapping(data, "outputs")
    if outputs.get("directory_fields") != ["OutputGroup"]:
        raise ValueError("Lat trajectory output directory identity is invalid.")
    if outputs.get("filename_template") != expected_filename:
        raise ValueError("Trajectory filename template is invalid.")
    if dict(_mapping(data, "manifests")) != {
        "figures": "figures.csv",
        "coverage": "coverage.csv",
        "emmeans": "emmeans.csv",
        "statistics": "statistics.csv",
        "model_fit": "model-fit.csv",
    }:
        raise ValueError("Lat trajectory manifests are invalid.")


def _validate_nested(data: dict[str, Any], kind: str) -> None:
    paths = _mapping(data, "paths")
    table_root = Path(_string(paths, "table_root"))
    manifest = Path(_string(paths, "feature_outputs_manifest"))
    try:
        manifest.relative_to(table_root)
    except ValueError as error:
        raise ValueError(
            "paths.feature_outputs_manifest must be below paths.table_root."
        ) from error
    inputs = _mapping(data, "inputs")
    expected_source = "transform" if kind == "spectral" else "normalize"
    expected = {
        "stage": "aggregate_region",
        "source_stage": expected_source,
        "aggregation_level": "region",
        "representation": kind,
    }
    for key, value in expected.items():
        if inputs.get(key) != value:
            raise ValueError(f"inputs.{key} must be {value!r} for {kind}.")
    if dict(_mapping(inputs, "row_filter")) != {
        "IncludeAggregate": True,
        "finite_nested_value": True,
    }:
        raise ValueError("inputs.row_filter does not match the nested-value boundary.")
    selectors = _mapping(inputs, "selectors")
    identities: list[tuple[str, str, str]] = []
    for domain in ("local", "connectivity"):
        domain_cfg = _mapping(selectors, domain)
        regions = _string_list(domain_cfg, "canonical_region")
        allowed_regions = (
            {"SNr", "STN"} if domain == "local" else {"SNr-STN", "SNr→STN"}
        )
        if not set(regions).issubset(allowed_regions):
            raise ValueError(f"Unsupported {domain} canonical_region value.")
        polar = _string_list(domain_cfg, "polar")
        if not set(polar).issubset({"Anodal", "Cathodal"}):
            raise ValueError(f"Unsupported {domain} polar value.")
        features = domain_cfg.get("features")
        if not isinstance(features, list) or not features:
            raise TypeError(f"inputs.selectors.{domain}.features must be non-empty.")
        for feature in features:
            if not isinstance(feature, Mapping):
                raise TypeError("Each feature selector must be a mapping.")
            metric = _string(feature, "metric")
            feature_output = _string(feature, "feature_output")
            _string_list(feature, "bands")
            if "canonical_region" in feature:
                feature_regions = _string_list(feature, "canonical_region")
                if not set(feature_regions).issubset(allowed_regions):
                    raise ValueError(
                        f"Unsupported {domain} feature canonical_region value."
                    )
            identities.append((domain, metric, feature_output))
    if len(identities) != len(set(identities)):
        raise ValueError("Nested feature selectors contain duplicate identities.")
    expected_figures = {
        "spectral": [
            {"id": "Phase", "layout": "marginal", "line_var": "Phase"},
            {
                "id": "Phase-Lat",
                "layout": "laterality_rows",
                "line_var": "Phase",
                "row_var": "Lat",
                "row_levels": LAT_LEVELS,
            },
        ],
        "trace": [
            {"id": "Phase", "layout": "marginal", "line_label": "Overall"},
            {
                "id": "Phase-Lat",
                "layout": "laterality_lines",
                "line_var": "Lat",
                "line_levels": LAT_LEVELS,
            },
        ],
        "raw": [
            {"id": "Phase", "layout": "marginal"},
            {
                "id": "Phase-Lat",
                "layout": "laterality_rows",
                "row_var": "Lat",
                "row_levels": LAT_LEVELS,
            },
        ],
    }
    singleton_figures = {
        "spectral": [
            {
                "id": "Phase-within-Lat",
                "layout": "laterality_singletons",
                "line_var": "Phase",
                "split_var": "Lat",
                "split_levels": LAT_LEVELS,
            }
        ],
        "trace": [
            {
                "id": "Phase-within-Lat",
                "layout": "laterality_singletons",
                "line_var": "Lat",
                "split_var": "Lat",
                "split_levels": LAT_LEVELS,
            }
        ],
        "raw": [
            {
                "id": "Phase-within-Lat",
                "layout": "laterality_singletons",
                "split_var": "Lat",
                "split_levels": LAT_LEVELS,
            }
        ],
    }
    submission_figures = {
        "spectral": [
            {
                "id": "Phase-Lat-split",
                "layout": "laterality_singletons",
                "line_var": "Phase",
                "split_var": "Lat",
                "split_levels": LAT_LEVELS,
            },
            {
                "id": "Phase-Lat-faceted",
                "layout": "laterality_rows",
                "line_var": "Phase",
                "row_var": "Lat",
                "row_levels": LAT_LEVELS,
            },
        ],
        "trace": [
            {
                "id": "Phase-Lat-split",
                "layout": "laterality_singletons",
                "line_var": "Lat",
                "split_var": "Lat",
                "split_levels": LAT_LEVELS,
            },
            {
                "id": "Phase-Lat-faceted",
                "layout": "laterality_rows",
                "line_var": "Lat",
                "row_var": "Lat",
                "row_levels": LAT_LEVELS,
            },
        ],
        "raw": [
            {
                "id": "Phase-Lat-split",
                "layout": "laterality_singletons",
                "split_var": "Lat",
                "split_levels": LAT_LEVELS,
            },
            {
                "id": "Phase-Lat-faceted",
                "layout": "laterality_rows",
                "row_var": "Lat",
                "row_levels": LAT_LEVELS,
            },
        ],
    }
    figures = data.get("figures")
    is_laterality_singletons = figures == singleton_figures[kind]
    is_submission_layout = figures == submission_figures[kind]
    if (
        figures != expected_figures[kind]
        and not is_laterality_singletons
        and not is_submission_layout
    ):
        raise ValueError(f"figures do not match the registered {kind} layout.")
    outputs = _mapping(data, "outputs")
    if is_laterality_singletons and "{Lat}" not in _string(
        outputs, "filename_template"
    ):
        raise ValueError("Laterality-singleton filename_template must include {Lat}.")
    if is_submission_layout:
        if outputs.get("directory_fields") != ["Metric"]:
            raise ValueError("Submission outputs.directory_fields must be [Metric].")
        if outputs.get("filename_template") != (
            "{Band}_{RegionValue}_{Polar}_{LatDisplay}.pdf"
        ):
            raise ValueError(
                "Submission filename_template must contain only value fields."
            )
        if Path(paths["manifest_root"]) != Path(paths["output_root"]):
            raise ValueError(
                "Submission manifests must be written at the representation root."
            )
    style = _mapping(data, "style")
    font = _mapping(style, "font")
    if font.get("family") != "Arial Unicode MS":
        raise ValueError("Nested visualization text must use Arial Unicode MS.")
    if style.get("transparent") is not True or style.get("dpi") != 600:
        raise ValueError("Formal visualization must be transparent at 600 dpi.")
    if kind in {"spectral", "trace"} and style.get("palette") != CLASSIC_PALETTE:
        raise ValueError("style.palette must equal the classic five-color palette.")
    axes = _mapping(style, "axes")
    expected_top_right = kind == "raw"
    if axes.get("show_top_right_spines") is not expected_top_right:
        raise ValueError(
            f"{kind} style.axes.show_top_right_spines must be {expected_top_right}."
        )
    legend = _mapping(style, "legend")
    if kind in {"spectral", "trace"}:
        expected_legend = {
            "show": True,
            "location": "outside_right",
            "frame": True,
            "fancybox": True,
            "frame_alpha": 0.90,
            "face_color": "white",
            "edge_color": "#CCCCCC",
            "edge_width": 0.60,
        }
        if dict(legend) != expected_legend:
            raise ValueError("Series legend style does not match the pain contract.")
    if kind in {"trace", "raw"} and style.get("temporal_boundaries") != [60, 120, 180]:
        raise ValueError("style.temporal_boundaries must be [60, 120, 180].")
    if kind == "trace":
        expected_smoothing = {
            "enabled": True,
            "method": "lowess",
            "fraction": 0.05,
            "iterations": 3,
            "delta_seconds": 0.2,
            "scope": "each_input_series",
            "persist_smoothed_values": False,
        }
        if dict(_mapping(style, "smoothing")) != expected_smoothing:
            raise ValueError(
                "Trace smoothing does not match the rendering-only LOWESS contract."
            )
        if dict(_mapping(style, "horizontal_reference")) != {
            "values": [0],
            "color": "#9A9A9A",
            "style": ":",
            "alpha": 0.80,
            "zorder": 8,
        }:
            raise ValueError(
                "Trace horizontal_reference does not match the shared contract."
            )
        if _mapping(style, "line_width_pt").get("reference") != 0.60:
            raise ValueError("Trace reference line width must be 0.60 pt.")
        inference = _mapping(data, "inference")
        enabled = inference.get("enabled")
        annotations = inference.get("significance_annotations")
        if not isinstance(enabled, bool) or not isinstance(annotations, bool):
            raise TypeError("Trace inference flags must be boolean.")
        if enabled:
            if is_laterality_singletons or is_submission_layout:
                raise ValueError(
                    "Submission and laterality-singleton trace inference must remain "
                    "disabled."
                )
            _string(inference, "interval_manifest")
            if annotations is not True:
                raise ValueError("Enabled trace inference must draw annotations.")
        elif inference.get("interval_manifest") is not None or annotations is not False:
            raise ValueError(
                "Disabled trace inference must not read or draw intervals."
            )
        if inference.get("alpha") != 0.05:
            raise ValueError("Trace interval alpha must be 0.05.")
        if inference.get("interval_shadow_alpha") != 0.15:
            raise ValueError("Trace interval-shadow alpha must be 0.15.")
        if inference.get("interval_star_pt") != 7:
            raise ValueError("Trace interval stars must use 7 pt text.")
    if kind == "raw":
        heatmap = _mapping(style, "heatmap")
        if heatmap.get("colormap") != "vik" or heatmap.get("value_mode") != "symmetric":
            raise ValueError("Raw heatmaps must use symmetric cm.vik colors.")
        if legend.get("show") is not False:
            raise ValueError("Raw heatmaps must not display a legend.")


def _validate_heatmap_matrix_item(
    item: Any, *, description: str, source_region: bool = False
) -> tuple[str, str]:
    if not isinstance(item, Mapping):
        raise TypeError(f"{description} must contain mappings.")
    expected_keys = {"metric", "feature_output", "label"}
    if source_region:
        expected_keys.add("source_region")
    if set(item) != expected_keys:
        raise ValueError(
            f"{description} entries must contain exactly {sorted(expected_keys)}."
        )
    if source_region:
        _string(item, "source_region")
    return _string(item, "metric"), _string(item, "feature_output")


def _validate_heatmap_rows(rows: Any, *, expected_values: list[str]) -> None:
    if not isinstance(rows, list) or not rows:
        raise TypeError("Heatmap matrix rows must be a non-empty list.")
    values: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"value", "label"}:
            raise ValueError("Heatmap rows must contain value and label.")
        values.append(_string(row, "value"))
        _string(row, "label")
    if values != expected_values:
        raise ValueError(f"Heatmap row order must be {expected_values}.")


def _validate_heatmap_matrices(data: dict[str, Any]) -> None:
    matrices = _mapping(data, "matrices")
    if set(matrices) != {"orientations", "banded", "aperiodic"}:
        raise ValueError(
            "matrices must contain orientations, banded, and aperiodic views."
        )
    if dict(_mapping(matrices, "orientations")) != {
        "banded": ["band-by-parameter", "parameter-by-band"],
        "aperiodic": ["parameter-by-region", "region-by-parameter"],
    }:
        raise ValueError("Heatmap orientations do not match the registered contract.")
    banded = _mapping(matrices, "banded")
    _validate_heatmap_rows(
        banded.get("rows"),
        expected_values=[
            "delta",
            "theta",
            "alpha",
            "beta_low",
            "beta_high",
            "gamma_low",
            "gamma_high",
        ],
    )
    groups = _mapping(banded, "groups")
    expected_groups = {
        "local": ("local", ["SNr", "STN"]),
        "connectivity": ("connectivity", ["SNr-STN-combined"]),
    }
    if set(groups) != set(expected_groups):
        raise ValueError("Banded heatmap groups do not match the canonical groups.")
    identities: list[tuple[str, str, str]] = []
    for group_name, (domain, regions) in expected_groups.items():
        group = _mapping(groups, group_name)
        if group.get("domain") != domain or group.get("regions") != regions:
            raise ValueError(f"Heatmap group {group_name!r} has invalid anatomy.")
        columns = group.get("columns")
        if not isinstance(columns, list) or not columns:
            raise TypeError(f"Heatmap group {group_name!r} columns must be non-empty.")
        for item in columns:
            metric, feature_output = _validate_heatmap_matrix_item(
                item,
                description=f"Heatmap group {group_name!r} columns",
                source_region=group_name == "connectivity",
            )
            identities.append((group_name, metric, feature_output))
    if len(identities) != len(set(identities)):
        raise ValueError("Heatmap matrix columns contain duplicate feature identities.")
    connectivity_columns = groups["connectivity"]["columns"]
    expected_connectivity = [
        ("ciplv", "SNr-STN"),
        ("imcoh_abs", "SNr-STN"),
        ("wpli", "SNr-STN"),
        ("psi", "SNr→STN"),
        ("trgc", "SNr→STN"),
    ]
    if [
        (item["metric"], item["source_region"]) for item in connectivity_columns
    ] != expected_connectivity:
        raise ValueError(
            "The combined connectivity heatmap must contain the registered "
            "undirected and directed parameters in order."
        )

    aperiodic = _mapping(matrices, "aperiodic")
    if (
        aperiodic.get("domain") != "local"
        or aperiodic.get("display_region") != "Params"
    ):
        raise ValueError("Aperiodic heatmaps must use the shared local Params view.")
    _validate_heatmap_rows(
        aperiodic.get("rows"), expected_values=["exponent", "offset"]
    )
    columns = aperiodic.get("columns")
    if not isinstance(columns, list) or len(columns) != 2:
        raise ValueError("Aperiodic heatmaps must contain SNr and STN columns.")
    identities = [
        (
            *_validate_heatmap_matrix_item(
                item,
                description="Aperiodic heatmap columns",
                source_region=True,
            ),
            item["source_region"],
            item["label"],
        )
        for item in columns
    ]
    if identities != [
        ("aperiodic", "mean-scalar", "SNr", "SNr"),
        ("aperiodic", "mean-scalar", "STN", "STN"),
    ]:
        raise ValueError(
            "Aperiodic heatmap columns must be SNr and STN aperiodic/mean-scalar."
        )


def _validate_regional_contrast_heatmap_matrices(data: dict[str, Any]) -> None:
    matrices = _mapping(data, "matrices")
    if set(matrices) != {"orientations", "banded", "aperiodic"}:
        raise ValueError(
            "matrices must contain orientations, banded, and aperiodic views."
        )
    if dict(_mapping(matrices, "orientations")) != {
        "banded": ["band-by-parameter", "parameter-by-band"],
        "aperiodic": ["parameter-by-region", "region-by-parameter"],
    }:
        raise ValueError("Heatmap orientations do not match the registered contract.")
    banded = _mapping(matrices, "banded")
    _validate_heatmap_rows(
        banded.get("rows"),
        expected_values=[
            "delta",
            "theta",
            "alpha",
            "beta_low",
            "beta_high",
            "gamma_low",
            "gamma_high",
        ],
    )
    groups = _mapping(banded, "groups")
    if set(groups) != {"local"}:
        raise ValueError("Regional contrast heatmaps must contain only local data.")
    local = _mapping(groups, "local")
    if local.get("domain") != "local" or local.get("regions") != ["SNr-minus-STN"]:
        raise ValueError("Regional contrast heatmaps require SNr-minus-STN.")
    columns = local.get("columns")
    if not isinstance(columns, list):
        raise TypeError("Regional contrast banded columns must be a list.")
    identities = [
        _validate_heatmap_matrix_item(
            item, description="Regional contrast banded columns"
        )
        for item in columns
    ]
    if identities != [
        ("periodic", "mean-scalar"),
        ("raw_power", "mean-scalar"),
        ("burst", "mean-scalar"),
        ("burst", "duration-scalar"),
        ("burst", "rate-scalar"),
        ("burst", "occupancy-scalar"),
    ]:
        raise ValueError("Regional contrast banded parameters are invalid.")

    aperiodic = _mapping(matrices, "aperiodic")
    if (
        aperiodic.get("domain") != "local"
        or aperiodic.get("display_region") != "SNr-minus-STN"
    ):
        raise ValueError("Regional contrast aperiodic anatomy is invalid.")
    _validate_heatmap_rows(
        aperiodic.get("rows"), expected_values=["exponent", "offset"]
    )
    columns = aperiodic.get("columns")
    if not isinstance(columns, list) or len(columns) != 1:
        raise ValueError(
            "Regional contrast aperiodic heatmaps require one contrast column."
        )
    item = columns[0]
    identity = (
        *_validate_heatmap_matrix_item(
            item,
            description="Regional contrast aperiodic column",
            source_region=True,
        ),
        item["source_region"],
        item["label"],
    )
    if identity != (
        "aperiodic",
        "mean-scalar",
        "SNr-minus-STN",
        "SNr−STN",
    ):
        raise ValueError("Regional contrast aperiodic parameter is invalid.")


def _validate_signed_logp_heatmap(data: dict[str, Any]) -> None:
    for section in ("matrices",):
        _mapping(data, section)
    paths = _mapping(data, "paths")
    for key in ("stats_root", "models_manifest", "stats_outputs_manifest"):
        _string(paths, key)
    stats_root = Path(paths["stats_root"])
    for key in ("models_manifest", "stats_outputs_manifest"):
        try:
            Path(paths[key]).relative_to(stats_root)
        except ValueError as error:
            raise ValueError(f"paths.{key} must be below paths.stats_root.") from error

    inputs = _mapping(data, "inputs")
    stats_analysis_id = _string(inputs, "stats_analysis_id")
    accepted = ["ok", "singular", "convergence_warning"]
    if inputs.get("accepted_model_statuses") != accepted:
        raise ValueError("inputs.accepted_model_statuses does not match the contract.")
    test_kind = inputs.get("test_kind")
    if test_kind == "emm_pairwise":
        if stats_analysis_id in {
            "phase-by-lat",
            "phase-by-lat-contact-paired",
        }:
            expected_analysis_id = stats_analysis_id
            expected_figures = [
                {"id": "Phase", "test_id": "Phase", "layout": "comparison_rows"},
                {
                    "id": "Phase-Lat",
                    "test_id": "Phase-Lat",
                    "layout": "comparison_rows_lat_columns",
                },
            ]
            if stats_analysis_id == "phase-by-lat-contact-paired":
                expected_figures.extend(
                    [
                        {
                            "id": "Phase-comparison-columns",
                            "test_id": "Phase",
                            "layout": "comparison_columns",
                        },
                        {
                            "id": "Phase-Lat-comparison-columns",
                            "test_id": "Phase-Lat",
                            "layout": "lat_rows_comparison_columns",
                        },
                    ]
                )
        elif stats_analysis_id == "phase-within-lat":
            expected_analysis_id = "phase-within-lat"
            expected_figures = [
                {"id": "Phase", "test_id": "Phase", "layout": "comparison_rows"}
            ]
        else:
            raise ValueError("Pairwise heatmap analysis is not registered.")
        expected_inputs = {
            "stats_analysis_id": expected_analysis_id,
            "accepted_model_statuses": accepted,
            "test_kind": "emm_pairwise",
            "result_type": "tukey",
            "effect_column": "estimate",
            "p_column": "p_tukey",
            "retained_p_columns": ["p_raw", "p_tukey"],
        }
        expected_phases = PHASE_LEVELS
        factor_levels = _mapping(data, "factor_levels")
        if list(factor_levels) != ["Phase", "Lat"]:
            raise ValueError("Pairwise heatmap factors must be Phase and Lat.")
        if _string_list(factor_levels, "Lat") != LAT_LEVELS:
            raise ValueError(f"factor_levels.Lat must be {LAT_LEVELS}.")
        comparisons = _mapping(data, "labels").get("comparisons")
        expected_comparisons = [
            {"group1": "Early", "group2": "Pre", "display": "Pre vs Early"},
            {"group1": "Late", "group2": "Pre", "display": "Pre vs Late"},
            {"group1": "Post", "group2": "Pre", "display": "Pre vs Post"},
            {"group1": "Late", "group2": "Early", "display": "Early vs Late"},
            {"group1": "Post", "group2": "Early", "display": "Early vs Post"},
            {"group1": "Post", "group2": "Late", "display": "Late vs Post"},
        ]
        if comparisons != expected_comparisons:
            raise ValueError("labels.comparisons does not match the contrast contract.")
    elif test_kind == "emmean_vs_null":
        contrast = _string(inputs, "contrast")
        null_contracts = {
            (
                "polarity-by-phase-lat-contact-paired",
                "Anodal-minus-Cathodal",
            ): {
                "stats_analysis_id": "polarity-by-phase-lat-contact-paired",
                "conditional": "Lat",
                "levels": LAT_LEVELS,
                "test_id": "Phase-Lat",
                "layout": "lat_rows_phase_columns",
                "label": "⊕−⊖",
            },
            (
                "pre-by-polar-contact-paired",
                "Anodal-minus-Cathodal",
            ): {
                "stats_analysis_id": "pre-by-polar-contact-paired",
                "conditional": "Lat",
                "levels": LAT_LEVELS,
                "test_id": "Lat",
                "layout": "lat_rows_phase_columns",
                "label": "⊕−⊖",
            },
            ("polarity-by-phase-lat", "Anodal-minus-Cathodal"): {
                "stats_analysis_id": "polarity-by-phase-lat",
                "conditional": "Lat",
                "levels": LAT_LEVELS,
                "test_id": "Phase-Lat",
                "layout": "lat_rows_phase_columns",
                "label": "⊕−⊖",
            },
            ("laterality-by-phase-polar", "Ipsi-minus-Contra"): {
                "stats_analysis_id": "laterality-by-phase-polar",
                "conditional": "Polar",
                "levels": ["Anodal", "Cathodal"],
                "test_id": "Phase-Polar",
                "layout": "polar_rows_phase_columns",
                "label": "Ipsi−Contra",
            },
            ("laterality-within-polar", "Ipsi-minus-Contra"): {
                "stats_analysis_id": "laterality-within-polar",
                "conditional": "Polar",
                "levels": ["Anodal", "Cathodal"],
                "test_id": "Phase",
                "layout": "polar_rows_phase_columns",
                "label": "Ipsi−Contra",
            },
            (
                "region-polarity-by-phase-lat",
                "Anodal-minus-Cathodal",
            ): {
                "stats_analysis_id": "region-polarity-by-phase-lat",
                "conditional": "Lat",
                "levels": LAT_LEVELS,
                "test_id": "Phase-Lat",
                "layout": "lat_rows_phase_columns",
                "label": "⊕−⊖",
            },
            ("region-by-phase-lat", "SNr-minus-STN"): {
                "stats_analysis_id": "region-by-phase-lat",
                "conditional": "Lat",
                "levels": LAT_LEVELS,
                "test_id": "Phase-Lat",
                "layout": "lat_rows_phase_columns",
                "label": "SNr−STN",
            },
        }
        contract = null_contracts.get((stats_analysis_id, contrast))
        if contract is None:
            raise ValueError("Heatmap contrast is not registered.")
        visualization_id = _mapping(data, "visualization").get("id")
        submission_holm_heatmap = (
            stats_analysis_id,
            visualization_id,
        ) in {
            (
                "polarity-by-phase-lat-contact-paired",
                "submission-polar-contact-signed-logp-heatmap",
            ),
            (
                "pre-by-polar-contact-paired",
                SUBMISSION_PRE_HEATMAP_ID,
            ),
            (
                "laterality-within-polar",
                "submission-lat-region-signed-logp-heatmap",
            ),
            ("region-by-phase-lat", SUBMISSION_REGION_HEATMAP_ID),
        }
        expected_inputs = {
            "stats_analysis_id": contract["stats_analysis_id"],
            "accepted_model_statuses": accepted,
            "test_kind": "emmean_vs_null",
            "result_type": "null-test",
            "effect_column": "emmean",
            "p_column": "p_holm" if submission_holm_heatmap else "p_raw",
            "retained_p_columns": ["p_raw", "p_holm"],
            "contrast": contrast,
        }
        expected_figures = [
            {
                "id": contract["test_id"],
                "test_id": contract["test_id"],
                "layout": contract["layout"],
            }
        ]
        expected_phases = (
            ["Pre"]
            if stats_analysis_id == "pre-by-polar-contact-paired"
            else ["Early", "Late", "Post"]
        )
        if dict(_mapping(_mapping(data, "labels"), "contrast")) != {
            contrast: contract["label"]
        }:
            raise ValueError("labels.contrast does not match the null-test contrast.")
        factor_levels = _mapping(data, "factor_levels")
        if list(factor_levels) != ["Phase", contract["conditional"]]:
            raise ValueError("Heatmap factor levels do not match the null-test model.")
        if _string_list(factor_levels, contract["conditional"]) != contract["levels"]:
            raise ValueError("Heatmap conditional factor levels are invalid.")
    else:
        raise ValueError("Heatmap inputs.test_kind is unsupported.")
    if (
        dict(inputs) != expected_inputs
        or stats_analysis_id != expected_inputs["stats_analysis_id"]
    ):
        raise ValueError("Heatmap inputs do not match the registered analysis.")
    if data.get("figures") != expected_figures:
        raise ValueError("Heatmap figures do not match the registered layouts.")
    if _string_list(_mapping(data, "factor_levels"), "Phase") != expected_phases:
        raise ValueError(f"Heatmap Phase levels must be {expected_phases}.")

    regional_contrast = stats_analysis_id in {
        "region-by-phase-lat",
        "region-polarity-by-phase-lat",
    }
    if regional_contrast:
        _validate_regional_contrast_heatmap_matrices(data)
        if (
            dict(_mapping(_mapping(data, "labels"), "region")).get("SNr-minus-STN")
            != "SNr−STN"
        ):
            raise ValueError("The regional contrast display label must be SNr−STN.")
    else:
        _validate_heatmap_matrices(data)
        if (
            dict(_mapping(_mapping(data, "labels"), "region")).get("SNr-STN-combined")
            != "SNr→/-STN"
        ):
            raise ValueError("The combined connectivity strip label must be SNr→/-STN.")
        if dict(_mapping(_mapping(data, "labels"), "region")).get("Params") != "Params":
            raise ValueError("The shared aperiodic strip label must be Params.")
    style = _mapping(data, "style")
    if dict(_mapping(style, "font")) != {
        "family": "Arial Unicode MS",
        "strip_pt": 7,
        "tick_pt": 6,
        "colorbar_pt": 6,
        "star_pt": 7,
    }:
        raise ValueError("Heatmap font style does not match the contract.")
    geometry = _mapping(style, "geometry_mm")
    if geometry.get("cell_size") != {
        "banded": [3.0, 3.0],
        "aperiodic": [3.0, 3.0],
    }:
        raise ValueError("Heatmap cell geometry does not match the view contract.")
    expected_panel_gap = {
        "comparison_rows": [0.0, 3.0],
        "comparison_rows_lat_columns": [3.0, 3.0],
        "lat_rows_phase_columns": [3.0, 3.0],
    }
    if stats_analysis_id == "phase-by-lat-contact-paired":
        expected_panel_gap.update(
            {
                "comparison_columns": [3.0, 3.0],
                "lat_rows_comparison_columns": [3.0, 3.0],
            }
        )
    if test_kind == "emmean_vs_null" and inputs["contrast"] == "Ipsi-minus-Contra":
        expected_panel_gap["polar_rows_phase_columns"] = [3.0, 3.0]
    if geometry.get("panel_gap") != expected_panel_gap:
        raise ValueError("Heatmap panel gaps do not match the pain layout contract.")
    if geometry.get("strip_pad") != 2.0:
        raise ValueError("Heatmap strip padding must match the pain 2 mm contract.")
    heatmap = _mapping(style, "heatmap")
    expected_heatmap = {
        "colormap": "vik",
        "value_mode": "symmetric",
        "value_limits": [-3.0, 3.0],
        "clip_display": True,
    }
    if data["visualization"]["id"] == "submission-phase-contact-signed-logp-heatmap":
        expected_heatmap["na_color"] = "#D9D9D9"
    if dict(heatmap) != expected_heatmap:
        raise ValueError("Heatmap color scale does not match the contract.")
    if dict(_mapping(style, "grid")) != {
        "color": "black",
        "width_pt": 0.60,
        "alpha": 0.90,
    }:
        raise ValueError("Heatmap grid style does not match the contract.")
    significance = _mapping(style, "significance")
    if significance.get("thresholds") != [
        {"p_lt": 0.001, "label": "***"},
        {"p_lt": 0.01, "label": "**"},
        {"p_lt": 0.05, "label": "*"},
    ]:
        raise ValueError("Heatmap significance thresholds are invalid.")
    if style.get("transparent") is not True or style.get("dpi") != 600:
        raise ValueError("Heatmaps must be transparent at 600 dpi.")
    if dict(_mapping(data, "manifests")) != {
        "figures": "figures.csv",
        "cells": "cells.csv",
    }:
        raise ValueError("Heatmap manifests must be figures.csv and cells.csv.")
    if "{Orientation}" not in _string(_mapping(data, "outputs"), "filename_template"):
        raise ValueError("Heatmap filename_template must include Orientation.")
    if (
        stats_analysis_id == "phase-by-lat-contact-paired"
        and "{FigureID}" not in data["outputs"]["filename_template"]
    ):
        raise ValueError(
            "Submission Phase heatmap filename_template must include FigureID."
        )
    if (
        stats_analysis_id == "phase-within-lat"
        and "{Lat}" not in data["outputs"]["filename_template"]
    ):
        raise ValueError("Fixed-Lat heatmap filename_template must include Lat.")


def _validate_correlation_heatmap(data: dict[str, Any]) -> None:
    p_column = "p_holm"
    scale_centered = (
        _mapping(data, "visualization").get("id")
        == SUBMISSION_CLINICAL_CORRELATION_SCALE_HEATMAP_ID
    )
    paths = _mapping(data, "paths")
    if set(paths) != {"stats_root", "correlations", "output_root", "manifest_root"}:
        raise ValueError(
            "Correlation heatmap paths do not match the registered contract."
        )
    stats_root = Path(_string(paths, "stats_root"))
    try:
        Path(_string(paths, "correlations")).relative_to(stats_root)
    except ValueError as error:
        raise ValueError(
            "paths.correlations must be below paths.stats_root."
        ) from error

    inputs = _mapping(data, "inputs")
    expected_inputs: dict[str, Any] = {
        "stats_analysis_id": (
            "clinical-correlation-adjacent-phase"
            if scale_centered
            else "clinical-correlation"
        ),
        "accepted_statuses": ["ok", "insufficient_n", "constant_input"],
        "value_column": "rho",
        "p_column": p_column,
        "retained_p_columns": ["p_raw", "p_holm"],
    }
    if scale_centered:
        expected_inputs.update(
            {
                "selected_scales": ["UPDRS-III", "PDQ-39"],
                "selected_endpoints": ["stn-3m", "stn-snr-3m"],
            }
        )
    if dict(inputs) != expected_inputs:
        raise ValueError("Correlation heatmap inputs do not match the stats contract.")
    expected_figures = (
        [
            {
                "id": "clinical-correlation-scale",
                "test_id": "clinical-correlation",
                "layout": "phase_rows",
            },
            {
                "id": "clinical-correlation-scale-composite",
                "test_id": "clinical-correlation",
                "layout": "phase_rows_composite",
            },
        ]
        if scale_centered
        else [
            {
                "id": "clinical-correlation-lat-split",
                "test_id": "clinical-correlation",
                "layout": "laterality_singletons",
                "split_var": "Lat",
                "split_levels": ["Ipsi", "Contra"],
            },
            {
                "id": "clinical-correlation-lat-faceted",
                "test_id": "clinical-correlation",
                "layout": "laterality_rows",
                "row_var": "Lat",
                "row_levels": ["Ipsi", "Contra"],
            },
        ]
    )
    if data.get("figures") != expected_figures:
        raise ValueError("Correlation heatmap layouts are invalid.")
    factor_levels = _mapping(data, "factor_levels")
    if dict(factor_levels) != {
        "Phase": ["Early", "Late", "Post"],
        "Lat": ["Ipsi", "Contra"],
        "Polar": ["Anodal", "Cathodal"],
    }:
        raise ValueError("Correlation heatmap factor levels are invalid.")
    labels = _mapping(data, "labels")
    if dict(_mapping(labels, "region")).get("SNr-STN-combined") != "SNr→/-STN":
        raise ValueError("The combined connectivity strip label must be SNr→/-STN.")
    if _string(labels, "delta_prefix") != "Δ":
        raise ValueError("The correlation heatmap delta prefix must be Δ.")
    expected_colorbar_labels = (
        {
            "UPDRS-III": "Partial Spearman's ρ",
            "PDQ-39": "Partial Spearman's ρ",
        }
        if scale_centered
        else {
            "preop": "Spearman's ρ (Preop)",
            "stn-3m": "Partial Spearman's ρ (STN, 3m)",
            "stn-snr-3m": "Partial Spearman's ρ (STN+SNr, 3m)",
            "incremental-3m": ("Partial Spearman's ρ (STN+SNr vs STN increment, 3m)"),
        }
    )
    if dict(_mapping(labels, "colorbar")) != expected_colorbar_labels:
        raise ValueError("Correlation heatmap colorbar labels are invalid.")
    if scale_centered and dict(_mapping(labels, "domain")) != {
        "local": "Local",
        "connectivity": "Connectivity",
    }:
        raise ValueError("Scale-centered heatmap domain labels are invalid.")
    if scale_centered and dict(_mapping(labels, "endpoint")) != {
        "stn-3m": "STN-DBS",
        "stn-snr-3m": "STN+SNr-DBS",
    }:
        raise ValueError("Scale-centered heatmap endpoint labels are invalid.")

    style = _mapping(data, "style")
    if dict(_mapping(style, "font")) != {
        "family": "Arial Unicode MS",
        "strip_pt": 7,
        "tick_pt": 6,
        "colorbar_pt": 6,
        "star_pt": 7,
    }:
        raise ValueError("Correlation heatmap font style is invalid.")
    expected_panel_gap = (
        {
            "phase_rows": [0.0, 3.0],
            "phase_rows_composite": [0.0, 3.0],
        }
        if scale_centered
        else {
            "laterality_singletons": [0.0, 0.0],
            "laterality_rows": [0.0, 3.0],
        }
    )
    expected_geometry = {
        "cell_size": {"matrix": [3.0, 3.0] if scale_centered else [2.5, 2.5]},
        "panel_gap": expected_panel_gap,
        "strip_top_height": 4.5,
        "strip_right_width": 4.5,
        "strip_pad": 2.0,
        "colorbar_width": 3.0,
        "colorbar_pad": 2.0,
        "colorbar_label_offset": 5.0,
    }
    if scale_centered:
        expected_geometry.update(
            {
                "group_strip_right_width": 4.5,
                "group_strip_pad": 2.0,
            }
        )
    if dict(_mapping(style, "geometry_mm")) != expected_geometry:
        raise ValueError("Correlation heatmap geometry is invalid.")
    if dict(_mapping(style, "strips")) != {
        "top_background": "#D7E3E0",
        "right_background": "#E3DCCF",
        "text_color": "black",
        "font_weight": "bold",
    }:
        raise ValueError("Correlation heatmap strip style is invalid.")
    expected_heatmap = {
        "colormap": "vik",
        "value_mode": "symmetric",
        "value_limits": [-1.0, 1.0],
        "clip_display": False,
    }
    if scale_centered:
        expected_heatmap["na_color"] = "#D9D9D9"
    if dict(_mapping(style, "heatmap")) != expected_heatmap:
        raise ValueError("Correlation heatmap color scale is invalid.")
    if dict(_mapping(style, "grid")) != {
        "color": "black",
        "width_pt": 0.60,
        "alpha": 0.90,
    }:
        raise ValueError("Correlation heatmap grid style is invalid.")
    if dict(_mapping(style, "significance")) != {
        "basis": p_column,
        "hide_non_significant": True,
        "text_color": "white",
        "thresholds": [
            {"p_lt": 0.001, "label": "***"},
            {"p_lt": 0.01, "label": "**"},
            {"p_lt": 0.05, "label": "*"},
        ],
    }:
        raise ValueError("Correlation heatmap significance style is invalid.")
    expected_default_colorbar = (
        "Partial Spearman's ρ" if scale_centered else "Spearman's ρ"
    )
    if dict(_mapping(style, "colorbar")) != {
        "label": expected_default_colorbar,
        "ticks": [-1.0, 0.0, 1.0],
    }:
        raise ValueError("Correlation heatmap colorbar is invalid.")
    if style.get("transparent") is not True or style.get("dpi") != 600:
        raise ValueError("Correlation heatmaps must be transparent at 600 dpi.")
    if dict(_mapping(data, "manifests")) != {
        "figures": "figures.csv",
        "cells": "cells.csv",
    }:
        raise ValueError(
            "Correlation heatmap manifests must be figures.csv and cells.csv."
        )
    filename = _string(_mapping(data, "outputs"), "filename_template")
    required_fields = (
        {"{Scale}", "{Domain}", "{RegionValue}", "{Polar}", "{Lat}"}
        if scale_centered
        else {
            "{AnalysisID}",
            "{EndpointID}",
            "{Domain}",
            "{RegionValue}",
            "{Polar}",
            "{Phase}",
            "{LatDisplay}",
        }
    )
    if any(field not in filename for field in required_fields):
        raise ValueError("Correlation heatmap filename_template omits identity fields.")
    if scale_centered:
        composite_filename = _string(
            _mapping(data, "outputs"), "composite_filename_template"
        )
        composite_fields = {"{Scale}", "{Domain}", "{RegionValue}"}
        if any(field not in composite_filename for field in composite_fields):
            raise ValueError(
                "Correlation heatmap composite_filename_template omits identity fields."
            )


def _validate_correlation_fit(data: dict[str, Any]) -> None:
    p_column = "p_holm"
    paths = _mapping(data, "paths")
    if set(paths) != {
        "stats_root",
        "correlations",
        "fit_points",
        "output_root",
        "manifest_root",
    }:
        raise ValueError("Correlation fit paths do not match the registered contract.")
    stats_root = Path(_string(paths, "stats_root"))
    for key in ("correlations", "fit_points"):
        try:
            Path(_string(paths, key)).relative_to(stats_root)
        except ValueError as error:
            raise ValueError(f"paths.{key} must be below paths.stats_root.") from error

    inputs = _mapping(data, "inputs")
    stats_analysis_id = _string(inputs, "stats_analysis_id")
    adjacent_phase = stats_analysis_id == "clinical-correlation-adjacent-phase"
    expected_stats_analysis_id = (
        "clinical-correlation-adjacent-phase"
        if adjacent_phase
        else "clinical-correlation"
    )
    expected_visualization_id = (
        SUBMISSION_CLINICAL_CORRELATION_ADJACENT_PHASE_FIT_ID
        if adjacent_phase
        else (
            SUBMISSION_CLINICAL_CORRELATION_FIT_ID
            if _mapping(data, "visualization").get("id")
            == SUBMISSION_CLINICAL_CORRELATION_FIT_ID
            else "clinical-correlation-fit"
        )
    )
    if _mapping(data, "visualization").get("id") != expected_visualization_id:
        raise ValueError("Correlation fit visualization id is invalid.")
    if dict(inputs) != {
        "stats_analysis_id": expected_stats_analysis_id,
        "accepted_statuses": ["ok", "insufficient_n", "constant_input"],
        "selection": {"column": "p_holm", "operator": "lt", "value": 0.05},
        "x_column": "XRaw",
        "y_column": "YRaw",
        "p_column": p_column,
    }:
        raise ValueError("Correlation fit inputs do not match the stats contract.")
    expected_figure_id = (
        "clinical-correlation-adjacent-phase-fit"
        if adjacent_phase
        else "clinical-correlation-fit"
    )
    expected_test_id = (
        "clinical-correlation-adjacent-phase"
        if adjacent_phase
        else "clinical-correlation"
    )
    if data.get("figures") != [
        {
            "id": expected_figure_id,
            "test_id": expected_test_id,
            "layout": "stratum_single",
        }
    ]:
        raise ValueError("Correlation fit figure layout must be stratum_single.")
    if dict(_mapping(data, "factor_levels")) != {
        "Phase": ["Early", "Late", "Post"],
        "Lat": ["Ipsi", "Contra"],
        "Polar": ["Anodal", "Cathodal"],
    }:
        raise ValueError("Correlation fit factor levels are invalid.")

    labels = _mapping(data, "labels")
    if _string(labels, "delta_prefix") != "Δ":
        raise ValueError("The correlation fit delta prefix must be Δ.")
    if dict(_mapping(labels, "endpoint")) != {
        "stn-3m": "STN-DBS",
        "stn-snr-3m": "STN+SNr-DBS",
    }:
        raise ValueError("Correlation fit endpoint labels are invalid.")
    phase_contrast_labels = labels.get("phase_contrast")
    if adjacent_phase and phase_contrast_labels != {
        "Early": "Early−Pre",
        "Late": "Late−Early",
        "Post": "Post−Late",
    }:
        raise ValueError("Adjacent-phase fit labels are invalid.")
    if not adjacent_phase and phase_contrast_labels is not None:
        raise ValueError("Phase-minus-Pre fit cannot define adjacent-phase labels.")
    style = _mapping(data, "style")
    if dict(_mapping(style, "font")) != {
        "family": "Arial Unicode MS",
        "strip_pt": 7,
        "axis_pt": 7,
        "tick_pt": 6,
    }:
        raise ValueError("Correlation fit font style is invalid.")
    if dict(_mapping(style, "geometry_mm")) != {
        "boxsize": [30.0, 25.0],
        "panel_gap": [0.0, 0.0],
        "strip_top_height": 4.5,
        "strip_right_width": 4.5,
        "strip_pad": 2.0,
        "x_label_offset": 5.0,
        "y_label_offset": 5.0,
    }:
        raise ValueError("Correlation fit geometry is invalid.")
    if dict(_mapping(style, "strips")) != {
        "top_background": "#D7E3E0",
        "right_background": "#E3DCCF",
        "text_color": "black",
        "font_weight": "bold",
    }:
        raise ValueError("Correlation fit strip style is invalid.")
    if dict(_mapping(style, "points")) != {
        "color_var": "ID",
        "palette": "viridis",
        "alpha": 1.0,
        "size": 10.0,
    }:
        raise ValueError("Correlation fit point style is invalid.")
    if dict(_mapping(style, "fit")) != {
        "ordinary_formula": "YRaw ~ XRaw",
        "partial_formula": "YRaw ~ XRaw + Baseline",
        "baseline_reference": "median",
        "confidence_level": 0.95,
        "interval_type": "mean",
        "line_color": "black",
        "line_width": 1.0,
        "show_ribbon": True,
        "ribbon_alpha": 0.30,
    }:
        raise ValueError("Correlation fit line style is invalid.")
    expected_annotation = {
        "p_column": p_column,
        "text_format": ("ρ = {beta:.2f}\n$p_{{\\mathrm{{Holm}}}}$ = {p:.3f} ({stars})"),
        "small_p_threshold": 0.001,
        "small_p_text_format": (
            "ρ = {beta:.2f}\n$p_{{\\mathrm{{Holm}}}}$ < 0.001 (***)"
        ),
        "location": "auto_by_fit_slope",
        "box_alpha": 0.0,
    }
    if dict(_mapping(style, "annotation")) != expected_annotation:
        raise ValueError("Correlation fit annotation style is invalid.")
    if dict(_mapping(style, "axes")) != {
        "grid": False,
        "show_top_right": False,
    }:
        raise ValueError("Correlation fit axes style is invalid.")
    if dict(_mapping(style, "legend")) != {"location": "none"}:
        raise ValueError("Correlation fit legend style is invalid.")
    if style.get("transparent") is not True or style.get("dpi") != 600:
        raise ValueError("Correlation fits must be transparent at 600 dpi.")
    if dict(_mapping(data, "manifests")) != {
        "figures": "figures.csv",
        "points": "points.csv",
    }:
        raise ValueError(
            "Correlation fit manifests must be figures.csv and points.csv."
        )
    filename = _string(_mapping(data, "outputs"), "filename_template")
    if any(field not in filename for field in ("{AnalysisID}", "{CorrelationID}")):
        raise ValueError("Correlation fit filename_template omits identity fields.")


def load_viz_config(path: Path | str) -> VizConfig:
    """Parse and validate one visualization YAML file."""

    resolved = Path(path).expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise TypeError(f"Configuration must contain a mapping: {resolved}")
    kind = _validate_common(data, resolved)
    if kind == "scalar":
        _validate_scalar(data)
    elif kind in {"lat_trajectory", "polar_trajectory", "region_trajectory"}:
        _validate_lat_trajectory(data)
    elif kind == "signed_logp_heatmap":
        _validate_signed_logp_heatmap(data)
    elif kind == "correlation_heatmap":
        _validate_correlation_heatmap(data)
    elif kind == "correlation_fit":
        _validate_correlation_fit(data)
    elif kind == "programming_intensity":
        from .programming import validate_programming_config

        validate_programming_config(data)
        fit_reference = load_viz_config(
            resolved.parent / data["inputs"]["fit_style_config"]
        )
        data["style"]["fit_reference"] = fit_reference["style"]
    else:
        _validate_nested(data, kind)
    return VizConfig(path=resolved, data=data)
