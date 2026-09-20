"""Plan and execute configured scalar mixed-model analyses."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lfp_cohort.io import atomic_csv, atomic_pickle, read_stage_table

from .config import PAIRED_CONTACT_ANALYSES, StatsConfig


FIELD_KEYS = {
    "Domain": "domain",
    "Metric": "metric",
    "FeatureOutput": "feature-output",
    "Band": "band",
    "Region": "region",
    "RegionContrast": "region-contrast",
    "PairRegion": "pair-region",
    "Polar": "polar",
    "Lat": "lat",
    "Contrast": "contrast",
}


@dataclass(frozen=True)
class StatsPlan:
    """Resolved inputs, model rows, test rows, and scalar model data."""

    models: pd.DataFrame
    tests: pd.DataFrame
    model_data: pd.DataFrame
    selected_inputs: tuple[str, ...]
    unlisted_inputs: tuple[str, ...]
    derived_tables: dict[Path, pd.DataFrame]
    derived_manifest: pd.DataFrame
    contact_pairing: pd.DataFrame

    @property
    def maximum_result_files(self) -> int:
        return len(self.models) + sum(
            len(_test_result_types(kind)) for kind in self.tests["TestKind"]
        )


def _render_value(value: Any, replacements: dict[str, str]) -> str:
    rendered = str(value)
    for old, new in replacements.items():
        rendered = rendered.replace(old, new)
    return rendered


def _model_id(values: dict[str, Any], fields: list[str], config: StatsConfig) -> str:
    identity = config["identity"]
    key_separator = identity["key_value_separator"]
    element_separator = identity["element_separator"]
    replacements = dict(identity["value_character_map"])
    elements = []
    for field in fields:
        key = FIELD_KEYS[field]
        value = _render_value(values[field], replacements)
        elements.append(f"{key}{key_separator}{value}")
    return element_separator.join(elements)


def _result_filename(config: StatsConfig, values: dict[str, str], kind: str) -> str:
    output = config["outputs"]
    fields = output["filename_fields"][kind]
    key_map = output["filename_key_map"]
    replacements = dict(output["value_character_map"])
    elements = []
    for field in fields:
        key = key_map[field]
        value = _render_value(values[field], replacements)
        elements.append(f"{key}{output['key_value_separator']}{value}")
    return output["element_separator"].join(elements) + ".csv"


def _test_result_types(test_kind: str) -> tuple[str, ...]:
    if test_kind == "joint_term":
        return ("omnibus",)
    if test_kind == "emm_pairwise":
        return ("emm", "tukey")
    if test_kind == "emmean_vs_null":
        return ("emm", "null-test")
    raise ValueError(f"Unsupported test kind: {test_kind}")


def _uses_paired_difference(config: StatsConfig) -> bool:
    return config.data.get("derivation", {}).get("kind") == "paired_difference"


def _uses_region_polarity_difference(config: StatsConfig) -> bool:
    return (
        config.data.get("derivation", {}).get("kind")
        == "region_polarity_difference"
    )


def _uses_derivation(config: StatsConfig) -> bool:
    return _uses_paired_difference(config) or _uses_region_polarity_difference(config)


def _uses_contact_pairing(config: StatsConfig) -> bool:
    return config.analysis_id in PAIRED_CONTACT_ANALYSES


def _model_factors(config: StatsConfig) -> list[str]:
    return list(config["model_data"]["factor_levels"])


def _conditional_factor(config: StatsConfig) -> str | None:
    factors = [factor for factor in _model_factors(config) if factor != "Phase"]
    if len(factors) > 1:
        raise ValueError("Stats models may contain at most one conditional factor.")
    return factors[0] if factors else None


def _model_fit_path(model: pd.Series, config: StatsConfig) -> Path:
    name = _result_filename(
        config,
        {"AnalysisID": config.analysis_id, "ResultType": "model-fit"},
        "model_fit",
    )
    return Path(model["_OutputDir"]) / name


def _test_result_path(
    model: pd.Series,
    test_kind: str,
    test_id: str,
    result_type: str,
    config: StatsConfig,
) -> Path:
    name = _result_filename(
        config,
        {
            "AnalysisID": config.analysis_id,
            "TestID": test_id,
            "TestKind": test_kind,
            "ResultType": result_type,
        },
        "test_result",
    )
    return Path(model["_OutputDir"]) / name


def _configured_inputs(config: StatsConfig) -> list[dict[str, Any]]:
    configured: list[dict[str, Any]] = []
    for domain, domain_config in config["feature_whitelist"].items():
        domain_regions = domain_config.get("region_values")
        for metric, metric_config in domain_config["metrics"].items():
            regions = metric_config.get("region_values", domain_regions)
            for feature_output, output_config in metric_config["outputs"].items():
                configured.append(
                    {
                        "Domain": domain,
                        "Metric": metric,
                        "FeatureOutput": feature_output,
                        "Representation": "scalar",
                        "RegionColumn": domain_config["region_column"],
                        "RegionValues": list(regions),
                        "Bands": list(output_config["bands"]),
                    }
                )
    return configured


def _select_manifest_inputs(
    config: StatsConfig,
) -> tuple[list[tuple[dict[str, Any], pd.Series]], tuple[str, ...]]:
    manifest = pd.read_csv(config.feature_manifest)
    required = {
        "Stage",
        "SourceStage",
        "AggregationLevel",
        "Representation",
        "Domain",
        "Metric",
        "FeatureOutput",
        "Path",
    }
    missing_columns = sorted(required.difference(manifest.columns))
    if missing_columns:
        raise KeyError(f"Feature manifest is missing columns: {missing_columns}")

    candidate_mask = pd.Series(True, index=manifest.index)
    for column, value in config["inputs"]["manifest_select"].items():
        candidate_mask &= manifest[column].eq(value)
    candidates = manifest.loc[candidate_mask].copy()

    selected: list[tuple[dict[str, Any], pd.Series]] = []
    selected_indices: set[int] = set()
    problems: list[str] = []
    for specification in _configured_inputs(config):
        matches = candidates.loc[
            candidates["Domain"].eq(specification["Domain"])
            & candidates["Metric"].eq(specification["Metric"])
            & candidates["FeatureOutput"].eq(specification["FeatureOutput"])
            & candidates["Representation"].eq(specification["Representation"])
        ]
        label = "/".join(
            str(specification[key]) for key in ("Domain", "Metric", "FeatureOutput")
        )
        if len(matches) == 0:
            problems.append(f"missing configured input {label}")
        elif len(matches) > 1:
            problems.append(f"duplicate configured input {label}: {len(matches)} rows")
        else:
            index = int(matches.index[0])
            selected_indices.add(index)
            selected.append((specification, matches.iloc[0]))
    if problems:
        raise ValueError("; ".join(problems))

    unlisted = tuple(
        str(path)
        for path in candidates.loc[~candidates.index.isin(selected_indices), "Path"]
    )
    return selected, unlisted


def _validate_table_identity(
    table: pd.DataFrame, specification: dict[str, Any], path: Path
) -> None:
    required = {
        "ID",
        "Domain",
        "Metric",
        "FeatureOutput",
        "Representation",
        "Band",
        "Polar",
        "Phase",
        "Lat",
        "Value",
        "IncludeAggregate",
        specification["RegionColumn"],
    }
    missing = sorted(required.difference(table.columns))
    if missing:
        raise KeyError(f"Stats input is missing columns {missing}: {path}")
    for column in ("Domain", "Metric", "FeatureOutput", "Representation"):
        values = set(table[column].dropna().astype(str).unique())
        expected = {str(specification[column])}
        if values != expected:
            raise ValueError(
                f"Stats input {column} values {sorted(values)} do not equal "
                f"{sorted(expected)}: {path}"
            )
    if not pd.api.types.is_bool_dtype(table["IncludeAggregate"].dtype):
        values = set(table["IncludeAggregate"].dropna().unique())
        if not values.issubset({True, False, np.bool_(True), np.bool_(False)}):
            raise TypeError(f"IncludeAggregate must be boolean: {path}")


def _eligible_rows(
    table: pd.DataFrame, specification: dict[str, Any], config: StatsConfig, path: Path
) -> pd.DataFrame:
    _validate_table_identity(table, specification, path)
    numeric = pd.to_numeric(table["Value"], errors="coerce")
    invalid_numeric = table["Value"].notna() & numeric.isna()
    if invalid_numeric.any():
        raise TypeError(f"Scalar Value contains non-numeric values: {path}")
    include = table["IncludeAggregate"].fillna(False).astype(bool)
    finite = np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan))
    eligible = table.loc[include & finite].copy()
    eligible["Value"] = numeric.loc[eligible.index].astype(float)
    eligible = eligible.loc[
        eligible["Band"].isin(specification["Bands"])
        & eligible[specification["RegionColumn"]].isin(specification["RegionValues"])
        & eligible["Polar"].isin(config["strata"]["Polar"])
    ].copy()
    configured_phases = list(config["model_data"]["factor_levels"]["Phase"])
    derivation = config.data.get("derivation", {})
    included_phases = list(derivation.get("include_phases", []))
    excluded_phases = list(derivation.get("exclude_phases", []))
    if included_phases:
        eligible = eligible.loc[eligible["Phase"].isin(included_phases)].copy()
    observed_phases = set(eligible["Phase"].dropna().astype(str))
    unknown_phases = sorted(observed_phases - set(configured_phases + excluded_phases))
    if unknown_phases:
        raise ValueError(
            f"Eligible stats rows contain unknown Phase levels {unknown_phases}: {path}"
        )
    if excluded_phases:
        eligible = eligible.loc[eligible["Phase"].isin(configured_phases)].copy()
    identity_columns = ["ID", "Phase", "Lat", specification["RegionColumn"]]
    if eligible[identity_columns].isna().any().any():
        raise ValueError(f"Eligible stats rows contain missing identity values: {path}")
    for factor, levels in config["model_data"]["factor_levels"].items():
        observed = set(eligible[factor].dropna().astype(str))
        unknown = sorted(observed - set(levels))
        if unknown:
            raise ValueError(
                f"Eligible stats rows contain unknown {factor} levels {unknown}: {path}"
            )
    if "Lat" in config["strata"]:
        observed_lat = set(eligible["Lat"].dropna().astype(str))
        unknown_lat = sorted(observed_lat - set(config["strata"]["Lat"]))
        if unknown_lat:
            raise ValueError(
                f"Eligible stats rows contain unknown Lat levels {unknown_lat}: "
                f"{path}"
            )
    return eligible


def _coverage(data: pd.DataFrame, config: StatsConfig) -> dict[str, int]:
    factor_levels = config["model_data"]["factor_levels"]
    conditional = _conditional_factor(config)
    expected_cells = int(np.prod([len(levels) for levels in factor_levels.values()]))
    complete_key = f"n_ID_complete{expected_cells}"
    both_key = f"n_ID_both_{conditional.lower()}" if conditional else None
    if data.empty:
        result = {
            "n_rows": 0,
            "n_ID": 0,
            complete_key: 0,
        }
        if both_key:
            result[both_key] = 0
        return result
    by_id = data.groupby("ID", observed=True)
    complete = by_id.apply(
        lambda rows: (
            len(rows[_model_factors(config)].drop_duplicates()) == expected_cells
        ),
        include_groups=False,
    )
    result = {
        "n_rows": int(len(data)),
        "n_ID": int(data["ID"].nunique()),
        complete_key: int(complete.sum()),
    }
    if both_key and conditional:
        both = by_id[conditional].nunique().eq(len(factor_levels[conditional]))
        result[both_key] = int(both.sum())
    return result


def _contact_unit_id(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value), ensure_ascii=False, separators=(",", ":"))
    return str(value)


def _pair_contact_rows(
    data: pd.DataFrame,
    contact_column: str,
    config: StatsConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply configured Phase-coverage and laterality policies to contacts."""

    pairing = config["contact_pairing"]
    output_column = str(pairing["output_column"])
    phase_column = str(pairing["complete_factor"])
    lat_column = str(pairing["laterality_factor"])
    phase_levels = list(pairing["complete_levels"])
    lat_levels = list(pairing["laterality_levels"])
    paired = data.copy()
    paired[output_column] = paired[contact_column].map(_contact_unit_id)

    contact_lat_counts = paired.groupby(
        ["ID", output_column], observed=True
    )[lat_column].nunique()
    if contact_lat_counts.gt(1).any():
        raise ValueError(
            "ContactUnitID maps to multiple Lat values within one ID and model."
        )

    qc_rows: list[dict[str, Any]] = []
    group_columns = ["ID", output_column, lat_column]
    for (subject, contact, lat), rows in paired.groupby(
        group_columns, observed=True, sort=False
    ):
        observed_set = set(rows[phase_column].astype(str))
        observed = [level for level in phase_levels if level in observed_set]
        missing = [level for level in phase_levels if level not in observed_set]
        qc_rows.append(
            {
                "ID": subject,
                "SourceContactColumn": contact_column,
                output_column: contact,
                lat_column: lat,
                "ObservedPhases": "|".join(observed),
                "MissingPhases": "|".join(missing),
                "CompletePhase": not missing,
            }
        )

    qc_columns = [
        "ID",
        "SourceContactColumn",
        output_column,
        lat_column,
        "ObservedPhases",
        "MissingPhases",
        "CompletePhase",
        "IDHasBothLat",
        "Included",
        "ExclusionReason",
    ]
    if not qc_rows:
        return paired.iloc[0:0].copy(), pd.DataFrame(columns=qc_columns)

    qc = pd.DataFrame(qc_rows)
    retain_incomplete = pairing["incomplete"] == "retain_and_report"
    laterality_basis = qc if retain_incomplete else qc.loc[qc["CompletePhase"]]
    id_has_both = laterality_basis.groupby("ID", observed=True)[lat_column].apply(
        lambda values: set(values.astype(str)) == set(lat_levels),
        include_groups=False,
    )
    qc["IDHasBothLat"] = qc["ID"].map(id_has_both).fillna(False).astype(bool)
    require_both_laterality = bool(pairing["require_both_laterality_per_id"])
    laterality_eligible = (
        qc["IDHasBothLat"]
        if require_both_laterality
        else pd.Series(True, index=qc.index)
    )
    phase_eligible = (
        pd.Series(True, index=qc.index)
        if retain_incomplete
        else qc["CompletePhase"]
    )
    qc["Included"] = phase_eligible & laterality_eligible
    if not retain_incomplete and require_both_laterality:
        qc["ExclusionReason"] = np.select(
            [~qc["CompletePhase"], ~qc["IDHasBothLat"]],
            ["missing_phase", "missing_laterality_after_contact_filter"],
            default="",
        )
    elif not retain_incomplete:
        qc["ExclusionReason"] = np.where(
            qc["CompletePhase"], "", "missing_phase"
        )
    else:
        qc["ExclusionReason"] = ""

    included_keys = set(
        map(
            tuple,
            qc.loc[qc["Included"], group_columns].itertuples(
                index=False, name=None
            ),
        )
    )
    keep = paired[group_columns].apply(tuple, axis=1).isin(included_keys)
    return paired.loc[keep].copy(), qc[qc_columns]


def _contact_coverage(data: pd.DataFrame) -> dict[str, int]:
    if data.empty:
        return {
            "n_contact_units": 0,
            "n_contact_units_ipsi": 0,
            "n_contact_units_contra": 0,
        }
    contacts = data[["ID", "ContactUnitID", "Lat"]].drop_duplicates()
    return {
        "n_contact_units": int(len(contacts)),
        "n_contact_units_ipsi": int(contacts["Lat"].eq("Ipsi").sum()),
        "n_contact_units_contra": int(contacts["Lat"].eq("Contra").sum()),
    }


def _derive_paired_difference(
    eligible: pd.DataFrame,
    specification: dict[str, Any],
    source_path: Path,
    config: StatsConfig,
) -> tuple[Path, pd.DataFrame, dict[str, Any]]:
    """Pair the configured factor levels and calculate their scalar difference."""

    region_column = specification["RegionColumn"]
    derivation = config["derivation"]
    factor = str(derivation["factor"])
    pair_keys = list(derivation["pair_columns"])
    if factor != region_column:
        pair_keys.append(region_column)
    source_columns = list(derivation["source_columns"])
    source = eligible.copy()
    if "ContactUnitID" in pair_keys and "ContactUnitID" not in source:
        contact_column = (
            "Channel" if specification["Domain"] == "local" else "ChannelPair"
        )
        if contact_column not in source:
            raise KeyError(
                "Contact-paired difference source is missing "
                f"{contact_column}: {source_path}"
            )
        if source[contact_column].isna().any():
            raise ValueError(
                "Contact-paired difference source contains missing "
                f"{contact_column}: {source_path}"
            )
        source["ContactUnitID"] = source[contact_column].map(_contact_unit_id)
    required = [*pair_keys, factor, *source_columns]
    missing = sorted(set(required).difference(source.columns))
    if missing:
        raise KeyError(
            f"Paired-difference source is missing columns {missing}: {source_path}"
        )
    duplicate = source.duplicated([*pair_keys, factor], keep=False)
    if duplicate.any():
        raise ValueError(
            f"Paired-difference source contains duplicate {factor} rows for "
            f"{'/'.join(pair_keys)}: {source_path}"
        )

    minuend = str(derivation["minuend"])
    subtrahend = str(derivation["subtrahend"])
    minuend_value = f"{minuend}Value"
    subtrahend_value = f"{subtrahend}Value"
    left_rename = {"Value": minuend_value}
    right_rename = {"Value": subtrahend_value}
    left_rename.update({column: f"{minuend}{column}" for column in source_columns})
    right_rename.update({column: f"{subtrahend}{column}" for column in source_columns})
    left = source.loc[
        source[factor].eq(minuend), [*pair_keys, "Value", *source_columns]
    ].rename(columns=left_rename)
    right = source.loc[
        source[factor].eq(subtrahend), [*pair_keys, "Value", *source_columns]
    ].rename(columns=right_rename)
    paired = left.merge(
        right,
        on=pair_keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    counts = paired["_merge"].value_counts()
    matched = paired.loc[paired["_merge"].eq("both")].drop(columns="_merge").copy()
    matched["Value"] = matched[minuend_value] - matched[subtrahend_value]
    matched["Domain"] = specification["Domain"]
    matched["Metric"] = specification["Metric"]
    matched["FeatureOutput"] = specification["FeatureOutput"]
    matched["Representation"] = specification["Representation"]
    contrast_column = "RegionContrast" if factor == region_column else "Contrast"
    matched[contrast_column] = derivation["label"]
    matched["IncludeAggregate"] = True
    identity_order = [
        column
        for column in (
            "ID",
            "ContactUnitID",
            "Record",
            "Trial",
            "Phase",
            "Lat",
            "Polar",
        )
        if column in matched.columns
    ]
    region_identity = [] if factor == region_column else [region_column]
    ordered = [
        *identity_order,
        "Domain",
        "Metric",
        "FeatureOutput",
        "Representation",
        "Band",
        *region_identity,
        contrast_column,
        "Value",
        minuend_value,
        subtrahend_value,
        *(f"{minuend}{column}" for column in source_columns),
        *(f"{subtrahend}{column}" for column in source_columns),
        "IncludeAggregate",
    ]
    matched = matched[ordered].reset_index(drop=True)
    relative_path = source_path.relative_to(config.table_root)
    derived_path = config.derived_root / relative_path
    manifest_row = {
        "AnalysisID": config.analysis_id,
        "Domain": specification["Domain"],
        "Metric": specification["Metric"],
        "FeatureOutput": specification["FeatureOutput"],
        "InputPath": str(source_path),
        "DerivedPath": str(derived_path),
        f"{minuend}Rows": int(len(left)),
        f"{subtrahend}Rows": int(len(right)),
        "PairedRows": int(len(matched)),
        f"Unpaired{minuend}Rows": int(counts.get("left_only", 0)),
        f"Unpaired{subtrahend}Rows": int(counts.get("right_only", 0)),
    }
    return derived_path, matched, manifest_row


def _derive_region_polarity_difference(
    eligible: pd.DataFrame,
    specification: dict[str, Any],
    source_path: Path,
    config: StatsConfig,
) -> tuple[Path, pd.DataFrame, dict[str, Any]]:
    """Calculate the SNr-minus-STN difference in the polarity effect."""

    if specification["Domain"] != "local" or specification["RegionColumn"] != "Region":
        raise ValueError(
            "Regional polarity differences require local inputs with Region labels."
        )
    derivation = config["derivation"]
    region_factor = str(derivation["region_factor"])
    polar_factor = str(derivation["polar_factor"])
    pair_keys = list(derivation["pair_columns"])
    source_columns = list(derivation["source_columns"])
    required = [*pair_keys, region_factor, polar_factor, *source_columns]
    missing = sorted(set(required).difference(eligible.columns))
    if missing:
        raise KeyError(
            "Regional polarity source is missing columns "
            f"{missing}: {source_path}"
        )
    duplicate = eligible.duplicated(
        [*pair_keys, region_factor, polar_factor], keep=False
    )
    if duplicate.any():
        raise ValueError(
            "Regional polarity source contains duplicate Region-by-Polar rows for "
            f"{'/'.join(pair_keys)}: {source_path}"
        )

    regions = [
        str(derivation["region_minuend"]),
        str(derivation["region_subtrahend"]),
    ]
    polars = [
        str(derivation["polar_minuend"]),
        str(derivation["polar_subtrahend"]),
    ]
    combined: pd.DataFrame | None = None
    value_columns: list[str] = []
    cell_counts: dict[str, int] = {}
    for region in regions:
        for polar in polars:
            prefix = f"{region}{polar}"
            value_column = f"{prefix}Value"
            value_columns.append(value_column)
            rename = {"Value": value_column}
            rename.update(
                {column: f"{prefix}{column}" for column in source_columns}
            )
            cell = eligible.loc[
                eligible[region_factor].eq(region)
                & eligible[polar_factor].eq(polar),
                [*pair_keys, "Value", *source_columns],
            ].rename(columns=rename)
            cell_counts[f"{prefix}Rows"] = int(len(cell))
            combined = (
                cell
                if combined is None
                else combined.merge(
                    cell,
                    on=pair_keys,
                    how="outer",
                    validate="one_to_one",
                )
            )

    if combined is None:
        raise ValueError(f"Regional polarity source contains no configured cells: {source_path}")
    candidate_rows = int(len(combined))
    missing_counts = {
        f"Missing{column.removesuffix('Value')}Rows": int(combined[column].isna().sum())
        for column in value_columns
    }
    matched = combined.dropna(subset=value_columns).copy()
    region_minuend, region_subtrahend = regions
    polar_minuend, polar_subtrahend = polars
    region_minuend_difference = f"{region_minuend}PolarityDifference"
    region_subtrahend_difference = f"{region_subtrahend}PolarityDifference"
    matched[region_minuend_difference] = (
        matched[f"{region_minuend}{polar_minuend}Value"]
        - matched[f"{region_minuend}{polar_subtrahend}Value"]
    )
    matched[region_subtrahend_difference] = (
        matched[f"{region_subtrahend}{polar_minuend}Value"]
        - matched[f"{region_subtrahend}{polar_subtrahend}Value"]
    )
    matched["Value"] = (
        matched[region_minuend_difference]
        - matched[region_subtrahend_difference]
    )
    matched["Domain"] = specification["Domain"]
    matched["Metric"] = specification["Metric"]
    matched["FeatureOutput"] = specification["FeatureOutput"]
    matched["Representation"] = specification["Representation"]
    matched["RegionContrast"] = derivation["region_label"]
    matched["Contrast"] = derivation["polar_label"]
    matched["IncludeAggregate"] = True
    source_value_columns = [
        f"{region}{polar}Value" for region in regions for polar in polars
    ]
    source_provenance_columns = [
        f"{region}{polar}{column}"
        for region in regions
        for polar in polars
        for column in source_columns
    ]
    ordered = [
        *pair_keys,
        "Domain",
        "Metric",
        "FeatureOutput",
        "Representation",
        "RegionContrast",
        "Contrast",
        "Value",
        region_minuend_difference,
        region_subtrahend_difference,
        *source_value_columns,
        *source_provenance_columns,
        "IncludeAggregate",
    ]
    matched = matched[ordered].reset_index(drop=True)
    derived_path = config.derived_root / source_path.relative_to(config.table_root)
    manifest_row = {
        "AnalysisID": config.analysis_id,
        "Domain": specification["Domain"],
        "Metric": specification["Metric"],
        "FeatureOutput": specification["FeatureOutput"],
        "InputPath": str(source_path),
        "DerivedPath": str(derived_path),
        **cell_counts,
        "CandidateRows": candidate_rows,
        "PairedRows": int(len(matched)),
        "IncompleteRows": candidate_rows - int(len(matched)),
        **missing_counts,
    }
    return derived_path, matched, manifest_row


def _test_rows(model_ids: list[str], config: StatsConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_id in model_ids:
        for test in config["tests"]:
            rows.append(
                {
                    "AnalysisID": config.analysis_id,
                    "ModelID": model_id,
                    "TestKind": test["kind"],
                    "TestID": test["id"],
                    "Term": test.get("term", ""),
                    "XVar": test.get("x_var", ""),
                    "PanelVar": test.get("panel_var", ""),
                    "FacetVar": test.get("facet_var", ""),
                    "Weights": test.get("weights", ""),
                    "ConfidenceLevel": test.get("confidence_level", np.nan),
                    "WithinAdjust": test.get("within_adjust", ""),
                    "AcrossMethod": test.get("across_method", ""),
                    "AdjustMethod": test.get("adjust_method", ""),
                    "AdjustScope": test.get("adjust_scope", ""),
                    "Null": test.get("null", np.nan),
                    "PrimaryP": test.get("primary_p", ""),
                    "ContrastDirection": test.get("contrast_direction", ""),
                }
            )
    return pd.DataFrame(rows)


def build_stats_plan(config: StatsConfig) -> StatsPlan:
    """Resolve manifest inputs and construct every configured model without writing."""

    selected, unlisted = _select_manifest_inputs(config)
    model_rows: list[dict[str, Any]] = []
    data_rows: list[pd.DataFrame] = []
    selected_paths: list[str] = []
    derived_tables: dict[Path, pd.DataFrame] = {}
    derived_manifest_rows: list[dict[str, Any]] = []
    contact_pairing_rows: list[pd.DataFrame] = []
    paired_difference = _uses_paired_difference(config)
    regional_polarity_difference = _uses_region_polarity_difference(config)
    paired_contacts = _uses_contact_pairing(config)
    contact_level = (
        config["inputs"]["manifest_select"]["AggregationLevel"] == "contact"
    )
    engine_directory = config["outputs"]["engine_directories"][
        config["model"]["engine"]
    ]

    for specification, manifest_row in selected:
        source_path = Path(str(manifest_row["Path"]))
        selected_paths.append(str(source_path))
        table = read_stage_table(source_path)
        eligible = _eligible_rows(table, specification, config, source_path)
        region_column = specification["RegionColumn"]
        source_parent = source_path.parent.relative_to(config.table_root)
        fields = list(config["identity"]["model_id_fields"][specification["Domain"]])

        model_source_path = source_path
        model_table = eligible
        region_values = list(specification["RegionValues"])
        stratum_column = "Polar"
        stratum_values = list(config["strata"]["Polar"])
        lat_values: list[str | None] = list(config["strata"].get("Lat", [None]))
        polar_values: list[str | None] = (
            list(config["strata"]["Polar"])
            if config.analysis_id == "laterality-within-polar"
            else [None]
        )
        stratified_values = [
            (lat_value, polar_value)
            for lat_value in lat_values
            for polar_value in polar_values
        ]
        if paired_difference:
            model_source_path, model_table, manifest_row = _derive_paired_difference(
                eligible, specification, source_path, config
            )
            derived_tables[model_source_path] = model_table
            derived_manifest_rows.append(manifest_row)
            if str(config["derivation"]["factor"]) == region_column:
                region_column = "RegionContrast"
                region_values = [str(config["derivation"]["label"])]
            else:
                stratum_column = "Contrast"
                stratum_values = [str(config["derivation"]["label"])]
        elif regional_polarity_difference:
            model_source_path, model_table, manifest_row = (
                _derive_region_polarity_difference(
                    eligible, specification, source_path, config
                )
            )
            derived_tables[model_source_path] = model_table
            derived_manifest_rows.append(manifest_row)
            region_column = "RegionContrast"
            region_values = [str(config["derivation"]["region_label"])]
            stratum_column = "Contrast"
            stratum_values = [str(config["derivation"]["polar_label"])]

        for band in specification["Bands"]:
            for region_value in region_values:
                for stratum_value in stratum_values:
                    for lat_value, polar_value in stratified_values:
                        mask = (
                            model_table["Band"].eq(band)
                            & model_table[region_column].eq(region_value)
                            & model_table[stratum_column].eq(stratum_value)
                        )
                        if lat_value is not None and not paired_contacts:
                            mask &= model_table["Lat"].eq(lat_value)
                        if polar_value is not None:
                            mask &= model_table["Polar"].eq(polar_value)
                        subset = model_table.loc[mask].copy()
                        model_factors = _model_factors(config)
                        contact_column: str | None = None
                        if contact_level:
                            contact_column = (
                                "ContactUnitID"
                                if "ContactUnitID" in subset
                                else "Channel"
                                if specification["Domain"] == "local"
                                else "ChannelPair"
                            )
                            if contact_column not in subset:
                                raise KeyError(
                                    "Contact-level stats input is missing "
                                    f"{contact_column}: {model_source_path}"
                                )
                            if subset[contact_column].isna().any():
                                raise ValueError(
                                    "Contact-level stats input contains missing "
                                    f"{contact_column}: {model_source_path}"
                                )
                        identity_values = {
                            "Domain": specification["Domain"],
                            "Metric": specification["Metric"],
                            "FeatureOutput": specification["FeatureOutput"],
                            "Band": band,
                            region_column: region_value,
                            stratum_column: stratum_value,
                        }
                        if lat_value is not None:
                            identity_values["Lat"] = lat_value
                        if polar_value is not None:
                            identity_values["Polar"] = polar_value
                        model_id = _model_id(identity_values, fields, config)
                        if paired_contacts:
                            if contact_column is None:
                                raise ValueError(
                                    "Paired-contact analysis requires contact input."
                                )
                            subset, pairing_qc = _pair_contact_rows(
                                subset, contact_column, config
                            )
                            if lat_value is not None:
                                subset = subset.loc[subset["Lat"].eq(lat_value)].copy()
                            pairing_qc = pairing_qc.assign(
                                AnalysisID=config.analysis_id,
                                ModelID=model_id,
                                Domain=specification["Domain"],
                                Metric=specification["Metric"],
                                FeatureOutput=specification["FeatureOutput"],
                                Band=band,
                                RegionVariable=region_column,
                                RegionValue=region_value,
                                Polar=stratum_value,
                                ModelLat=lat_value or "",
                            )
                            contact_pairing_rows.append(pairing_qc)
                        identity_columns = ["ID", *model_factors]
                        if contact_column is not None:
                            identity_columns.insert(1, contact_column)
                        duplicate = subset.duplicated(identity_columns, keep=False)
                        if duplicate.any():
                            unit = "contact-unit/" if contact_level else ""
                            raise ValueError(
                                f"Stats model contains duplicate ID/{unit}factor "
                                f"rows: {model_source_path}, {band}, {region_value}, "
                                f"{stratum_value}, {lat_value or 'all Lat'}"
                            )
                        output_dir = (
                            config.output_root
                            / source_parent
                            / specification["Representation"]
                            / specification["FeatureOutput"]
                            / band
                            / region_value
                            / stratum_value
                        )
                        if lat_value is not None:
                            output_dir /= lat_value
                        if polar_value is not None:
                            output_dir /= polar_value
                        output_dir /= engine_directory
                        coverage = _coverage(subset, config)
                        if paired_contacts:
                            coverage.update(_contact_coverage(subset))
                        elif "ContactUnitID" in subset:
                            coverage.update(_contact_coverage(subset))
                        model_row = {
                            "AnalysisID": config.analysis_id,
                            "ModelID": model_id,
                            "Engine": config["model"]["engine"],
                            "Formula": config["model"]["formula"],
                            "Domain": specification["Domain"],
                            "Metric": specification["Metric"],
                            "FeatureOutput": specification["FeatureOutput"],
                            "Representation": specification["Representation"],
                            "Band": band,
                            "RegionVariable": region_column,
                            "RegionValue": region_value,
                            stratum_column: stratum_value,
                            "SourcePath": str(model_source_path),
                            "InputPath": str(source_path),
                            "IncludeAggregateRequired": True,
                            "FiniteValueRequired": True,
                            **coverage,
                            "_OutputDir": str(output_dir),
                        }
                        if lat_value is not None:
                            model_row["Lat"] = lat_value
                        if polar_value is not None:
                            model_row["Polar"] = polar_value
                        model_rows.append(model_row)
                        model_columns = list(
                            config["model_data"]["factor_columns"]
                        )
                        if (
                            config.analysis_id
                            == "pre-by-polar-contact-paired"
                            and contact_column not in model_columns
                        ):
                            model_columns.append(contact_column)
                        model_columns.extend([*model_factors, "Value"])
                        model_data = subset[model_columns].copy()
                        model_data.insert(0, "ModelID", model_id)
                        data_rows.append(model_data)

    models = pd.DataFrame(model_rows)
    if models["ModelID"].duplicated().any():
        duplicate_ids = models.loc[models["ModelID"].duplicated(), "ModelID"].tolist()
        raise ValueError(f"Generated duplicate ModelID values: {duplicate_ids[:3]}")
    tests = _test_rows(models["ModelID"].tolist(), config)
    model_data = (
        pd.concat(data_rows, ignore_index=True)
        if data_rows
        else pd.DataFrame(
            columns=[
                "ModelID",
                *config["model_data"]["factor_columns"],
                *(
                    ["ContactUnitID"]
                    if config.analysis_id == "pre-by-polar-contact-paired"
                    else []
                ),
                *_model_factors(config),
                "Value",
            ]
        )
    )
    return StatsPlan(
        models=models,
        tests=tests,
        model_data=model_data,
        selected_inputs=tuple(selected_paths),
        unlisted_inputs=unlisted,
        derived_tables=derived_tables,
        derived_manifest=pd.DataFrame(derived_manifest_rows),
        contact_pairing=(
            pd.concat(contact_pairing_rows, ignore_index=True)
            if contact_pairing_rows
            else pd.DataFrame()
        ),
    )


def _summary(plan: StatsPlan, config: StatsConfig) -> dict[str, Any]:
    summary = {
        "analysis_id": config.analysis_id,
        "selected_inputs": len(plan.selected_inputs),
        "missing_configured_inputs": 0,
        "duplicate_configured_inputs": 0,
        "unlisted_inputs": list(plan.unlisted_inputs),
        "planned_models": len(plan.models),
        "registered_tests": len(plan.tests),
        "maximum_result_files": plan.maximum_result_files,
        "derived_tables": len(plan.derived_tables),
        "paired_rows": int(
            plan.derived_manifest.get("PairedRows", pd.Series(dtype=int)).sum()
        ),
    }
    if _uses_contact_pairing(config):
        summary.update(
            {
                "contact_units_evaluated": int(len(plan.contact_pairing)),
                "contact_units_included": int(
                    plan.contact_pairing.get(
                        "Included", pd.Series(dtype=bool)
                    ).sum()
                ),
                "contact_units_missing_phase": int(
                    plan.contact_pairing.get(
                        "CompletePhase", pd.Series(dtype=bool)
                    ).eq(False).sum()
                ),
                "contact_units_missing_laterality": int(
                    plan.contact_pairing.get(
                        "ExclusionReason", pd.Series(dtype=str)
                    ).eq("missing_laterality_after_contact_filter").sum()
                ),
            }
        )
    if _uses_paired_difference(config):
        for level in (
            config["derivation"]["minuend"],
            config["derivation"]["subtrahend"],
        ):
            column = f"Unpaired{level}Rows"
            summary[f"unpaired_{str(level).lower()}_rows"] = int(
                plan.derived_manifest.get(column, pd.Series(dtype=int)).sum()
            )
    if _uses_region_polarity_difference(config):
        summary["incomplete_rows"] = int(
            plan.derived_manifest.get(
                "IncompleteRows", pd.Series(dtype=int)
            ).sum()
        )
    return summary


def _potential_paths(plan: StatsPlan, config: StatsConfig) -> list[Path]:
    paths: list[Path] = []
    paths.extend(plan.derived_tables)
    models = plan.models.set_index("ModelID", drop=False)
    for _, model in plan.models.iterrows():
        paths.append(_model_fit_path(model, config))
    for _, test in plan.tests.iterrows():
        model = models.loc[test["ModelID"]]
        for result_type in _test_result_types(test["TestKind"]):
            paths.append(
                _test_result_path(
                    model,
                    test["TestKind"],
                    test["TestID"],
                    result_type,
                    config,
                )
            )
    for manifest_config in config["manifests"].values():
        paths.append(config.manifest_root / manifest_config["filename"])
    return paths


def _trash_existing(paths: list[Path]) -> None:
    trash = Path("/usr/bin/trash")
    if not trash.exists():
        raise FileNotFoundError(f"Trash command is unavailable: {trash}")
    for offset in range(0, len(paths), 200):
        subprocess.run(
            [str(trash), *(str(path) for path in paths[offset : offset + 200])],
            check=True,
        )


def _r_plan(plan: StatsPlan, config: StatsConfig) -> dict[str, Any]:
    tests = []
    for configured in config["tests"]:
        test = dict(configured)
        test.setdefault("panel_var", None)
        test.setdefault("facet_var", None)
        tests.append(test)
    return {
        "factor_levels": dict(config["model_data"]["factor_levels"]),
        "factor_columns": list(config["model_data"]["factor_columns"]),
        "model": {
            "formula": config["model"]["formula"],
            "reml": config["model"]["reml"],
        },
        "models": [
            {"model_id": model_id} for model_id in plan.models["ModelID"].tolist()
        ],
        "tests": tests,
    }


def _read_r_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"R runner did not produce expected table: {path}")
    return pd.read_csv(path)


def _run_r(plan: StatsPlan, config: StatsConfig) -> dict[str, pd.DataFrame]:
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise FileNotFoundError("Rscript is unavailable on PATH.")
    repo_root = Path(__file__).resolve().parents[1]
    r_runner = repo_root / "lfp_stats" / "r" / "run_scalar_models.R"
    config.output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{config.analysis_id}-runtime-", dir=config.output_root
    ) as temporary_name:
        temporary = Path(temporary_name)
        data_path = temporary / "model_data.csv"
        plan_path = temporary / "plan.json"
        result_root = temporary / "r_results"
        plan.model_data.to_csv(data_path, index=False)
        plan_path.write_text(
            json.dumps(_r_plan(plan, config), ensure_ascii=False),
            encoding="utf-8",
        )
        completed = subprocess.run(
            [rscript, str(r_runner), str(data_path), str(plan_path), str(result_root)],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            message = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"R stats runner failed: {message}")
        return {
            "model_status": _read_r_table(result_root / "model_status.csv"),
            "test_status": _read_r_table(result_root / "test_status.csv"),
            "omnibus": _read_r_table(result_root / "joint.csv"),
            "emm": _read_r_table(result_root / "emm.csv"),
            "tukey": _read_r_table(result_root / "tukey.csv"),
            "null-test": _read_r_table(result_root / "null_test.csv"),
        }


def _validate_r_status(
    plan: StatsPlan, results: dict[str, pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_status = results["model_status"]
    test_status = results["test_status"]
    if (
        len(model_status) != len(plan.models)
        or model_status["ModelID"].duplicated().any()
    ):
        raise ValueError("R model-status table does not match the planned models.")
    test_key = ["ModelID", "TestKind", "TestID"]
    if len(test_status) != len(plan.tests) or test_status.duplicated(test_key).any():
        raise ValueError("R test-status table does not match the registered tests.")
    if set(model_status["ModelID"]) != set(plan.models["ModelID"]):
        raise ValueError("R model-status identities do not match the model plan.")
    planned_test_keys = set(map(tuple, plan.tests[test_key].to_numpy()))
    returned_test_keys = set(map(tuple, test_status[test_key].to_numpy()))
    if returned_test_keys != planned_test_keys:
        raise ValueError("R test-status identities do not match the test plan.")
    return model_status, test_status


def _append_result_identity(
    table: pd.DataFrame,
    model_id: str,
    test_kind: str,
    test_id: str,
    config: StatsConfig,
) -> pd.DataFrame:
    identity_columns = ["ModelID", "TestKind", "TestID"]
    result = table.drop(columns=identity_columns, errors="ignore").copy()
    identity_values = {
        "AnalysisID": config.analysis_id,
        "ModelID": model_id,
        "TestKind": test_kind,
        "TestID": test_id,
        "Engine": config["model"]["engine"],
    }
    for column in config["outputs"]["result_identity_fields"]:
        result[column] = identity_values[column]
    return result


def _write_outputs(
    plan: StatsPlan,
    config: StatsConfig,
    results: dict[str, pd.DataFrame],
    overwrite: bool,
) -> dict[str, Any]:
    model_status, test_status = _validate_r_status(plan, results)
    potential = _potential_paths(plan, config)
    existing = [path for path in potential if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Planned stats output exists; pass --overwrite to replace it: {existing[0]}"
        )
    if existing:
        _trash_existing(existing)

    if plan.derived_tables:
        protocol = int(config["derivation"]["pickle_protocol"])
        for path, table in plan.derived_tables.items():
            atomic_pickle(table, path, overwrite=False, protocol=protocol)

    models = plan.models.merge(
        model_status,
        on="ModelID",
        how="left",
        validate="one_to_one",
    )
    tests = plan.tests.merge(
        test_status,
        on=["ModelID", "TestKind", "TestID"],
        how="left",
        validate="one_to_one",
    )
    models_by_id = models.set_index("ModelID", drop=False)
    output_rows: list[dict[str, Any]] = []

    for _, model in models.iterrows():
        fit_path = _model_fit_path(model, config)
        complete_columns = [
            column for column in model.index if str(column).startswith("n_ID_complete")
        ]
        if len(complete_columns) != 1:
            raise ValueError(
                f"Model must contain one complete-ID count: {model['ModelID']}"
            )
        both_columns = [
            column for column in model.index if str(column).startswith("n_ID_both_")
        ]
        if len(both_columns) > 1:
            raise ValueError(
                f"Model contains multiple conditional support counts: "
                f"{model['ModelID']}"
            )
        fit_columns = [
            "AnalysisID",
            "ModelID",
            "Engine",
            "Formula",
            "n_rows",
            "n_ID",
            *[
                column
                for column in model.index
                if str(column).startswith("n_contact_units")
            ],
            *both_columns,
            complete_columns[0],
            "Singular",
            "Status",
            "Message",
        ]
        fit_table = pd.DataFrame([{column: model[column] for column in fit_columns}])
        atomic_csv(fit_table, fit_path, overwrite=False)
        output_rows.append(
            {
                "AnalysisID": config.analysis_id,
                "ModelID": model["ModelID"],
                "TestKind": "",
                "TestID": "",
                "ResultType": "model-fit",
                "Path": str(fit_path),
                "Rows": 1,
                "Columns": len(fit_table.columns),
                "Status": model["Status"],
            }
        )

    result_tables = {
        "omnibus": results["omnibus"],
        "emm": results["emm"],
        "tukey": results["tukey"],
        "null-test": results["null-test"],
    }
    for _, test in tests.iterrows():
        if test["Status"] == "not_estimable":
            continue
        model = models_by_id.loc[test["ModelID"]]
        for result_type in _test_result_types(test["TestKind"]):
            combined = result_tables[result_type]
            mask = (
                combined["ModelID"].eq(test["ModelID"])
                & combined["TestKind"].eq(test["TestKind"])
                & combined["TestID"].eq(test["TestID"])
            )
            selected = combined.loc[mask].copy()
            if selected.empty:
                raise ValueError(
                    "R result is missing for an estimable registered test: "
                    f"{test['ModelID']}, {test['TestKind']}, {test['TestID']}, "
                    f"{result_type}"
                )
            if test["TestKind"] in {"emm_pairwise", "emmean_vs_null"}:
                active_axes = {
                    test[column]
                    for column in ("XVar", "PanelVar", "FacetVar")
                    if test[column]
                }
                for axis in config["model_data"]["factor_levels"]:
                    if axis not in active_axes:
                        selected = selected.drop(columns=axis, errors="ignore")
            table = _append_result_identity(
                selected,
                test["ModelID"],
                test["TestKind"],
                test["TestID"],
                config,
            )
            path = _test_result_path(
                model,
                test["TestKind"],
                test["TestID"],
                result_type,
                config,
            )
            atomic_csv(table, path, overwrite=False)
            output_rows.append(
                {
                    "AnalysisID": config.analysis_id,
                    "ModelID": test["ModelID"],
                    "TestKind": test["TestKind"],
                    "TestID": test["TestID"],
                    "ResultType": result_type,
                    "Path": str(path),
                    "Rows": len(table),
                    "Columns": len(table.columns),
                    "Status": test["Status"],
                }
            )

    model_fields = config["manifests"]["models"]["fields"]
    test_fields = config["manifests"]["tests"]["fields"]
    output_fields = config["manifests"]["outputs"]["fields"]
    outputs = pd.DataFrame(output_rows, columns=output_fields)
    config.manifest_root.mkdir(parents=True, exist_ok=True)
    atomic_csv(
        models[model_fields],
        config.manifest_root / config["manifests"]["models"]["filename"],
        overwrite=False,
    )
    atomic_csv(
        tests[test_fields],
        config.manifest_root / config["manifests"]["tests"]["filename"],
        overwrite=False,
    )
    atomic_csv(
        outputs,
        config.manifest_root / config["manifests"]["outputs"]["filename"],
        overwrite=False,
    )
    if plan.derived_tables:
        derived_config = config["manifests"]["derived_tables"]
        atomic_csv(
            plan.derived_manifest[derived_config["fields"]],
            config.manifest_root / derived_config["filename"],
            overwrite=False,
        )
    if _uses_contact_pairing(config):
        pairing_config = config["manifests"]["contact_pairing"]
        atomic_csv(
            plan.contact_pairing.reindex(columns=pairing_config["fields"]),
            config.manifest_root / pairing_config["filename"],
            overwrite=False,
        )
    return {
        "models": len(models),
        "tests": len(tests),
        "outputs": len(outputs),
        "derived_tables": len(plan.derived_tables),
        "contact_pairing_rows": len(plan.contact_pairing),
        "model_status": models["Status"].value_counts().sort_index().to_dict(),
    }


def run_stats(
    config: StatsConfig, *, dry_run: bool = False, overwrite: bool = False
) -> dict[str, Any]:
    """Plan the configured analysis and optionally execute every registered model."""

    plan = build_stats_plan(config)
    summary = _summary(plan, config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if dry_run:
        return summary

    existing = [path for path in _potential_paths(plan, config) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Planned stats output exists; pass --overwrite to replace it: {existing[0]}"
        )
    results = _run_r(plan, config)
    execution = _write_outputs(plan, config, results, overwrite)
    print(json.dumps(execution, indent=2, ensure_ascii=False))
    return {**summary, **execution}
