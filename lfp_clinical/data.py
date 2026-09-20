"""Load and align clinical endpoints with normalized region LFP predictors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lfp_cohort.io import read_stage_table
from lfp_viz.data import path_below, require_columns

from .config import ClinicalCorrelationConfig

MANIFEST_COLUMNS = {
    "Stage",
    "SourceStage",
    "Domain",
    "Metric",
    "FeatureOutput",
    "Representation",
    "Path",
    "Rows",
    "AggregationLevel",
}
ADJACENT_IDENTITY_COLUMNS = [
    "ID",
    "Domain",
    "Metric",
    "FeatureOutput",
    "Band",
    "RegionVariable",
    "RegionValue",
    "DisplayRegion",
    "FeatureOrder",
    "FeatureLabel",
    "Polar",
    "Lat",
]


@dataclass(frozen=True)
class ClinicalCorrelationInputs:
    """Validated patient endpoints and aligned LFP predictor tables."""

    clinical_endpoints: pd.DataFrame
    predictor_metadata: pd.DataFrame
    predictor_values: pd.DataFrame
    inputs_manifest: pd.DataFrame


def _normalize_ids(values: pd.Series) -> pd.Series:
    normalized = values.astype(str).str.strip().str.replace(r"^sub-", "", regex=True)
    if normalized.eq("").any():
        raise ValueError("Subject IDs must not be empty after normalization.")
    return normalized


def _numeric_column(table: pd.DataFrame, column: str, description: str) -> pd.Series:
    numeric = pd.to_numeric(table[column], errors="coerce")
    invalid = table[column].notna() & numeric.isna()
    if invalid.any():
        raise TypeError(f"{description} contains non-numeric values in {column}.")
    return numeric.astype(float)


def load_clinical_endpoints(
    config: ClinicalCorrelationConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read the source workbook and derive the registered patient endpoints."""

    clinical_config = config["inputs"]["clinical"]
    columns = clinical_config["columns"]
    source = pd.read_excel(
        config.clinical_workbook,
        sheet_name=clinical_config["sheet_name"],
    )
    required = set(columns.values())
    require_columns(source, required, f"Clinical workbook {config.clinical_workbook}")
    table = source.loc[:, list(columns.values())].rename(
        columns={value: key.capitalize() for key, value in columns.items()}
    )
    table = table.rename(
        columns={
            "Id": "ID",
            "Protocol": "Protocol",
            "Phase": "Phase",
            "Scale": "Scale",
            "Value": "Value",
            "Baseline": "Baseline",
        }
    )
    table["ID"] = _normalize_ids(table["ID"])
    table["Scale"] = table["Scale"].astype(str)
    table["Protocol"] = table["Protocol"].astype(str)
    table["Phase"] = table["Phase"].astype(str)
    table["Value"] = _numeric_column(table, "Value", "Clinical workbook")
    table["Baseline"] = _numeric_column(table, "Baseline", "Clinical workbook")

    duplicate_keys = ["ID", "Scale", "Protocol", "Phase"]
    if table.duplicated(duplicate_keys, keep=False).any():
        raise ValueError("Clinical workbook contains duplicate endpoint rows.")
    baseline_counts = table.groupby(["ID", "Scale"], observed=True)["Baseline"].nunique(
        dropna=False
    )
    if baseline_counts.gt(1).any():
        raise ValueError("Baseline differs within an ID and Scale.")

    phase = str(clinical_config["phase"])
    protocols = clinical_config["protocols"]
    selected = table.loc[table["Phase"].eq(phase)].copy()
    configured_scales = [item["value"] for item in config["scales"]]
    observed_scales = set(selected["Scale"])
    if observed_scales != set(configured_scales):
        missing = sorted(set(configured_scales) - observed_scales)
        unexpected = sorted(observed_scales - set(configured_scales))
        raise ValueError(
            f"Clinical scale registry mismatch; missing={missing}, unexpected={unexpected}."
        )
    observed_protocols = set(selected["Protocol"])
    expected_protocols = {protocols["stn"], protocols["stn_snr"]}
    if observed_protocols != expected_protocols:
        raise ValueError(
            f"Clinical protocols must be {sorted(expected_protocols)}; found {sorted(observed_protocols)}."
        )

    values = selected.set_index(["ID", "Scale", "Protocol"])["Value"].unstack(
        "Protocol"
    )
    values = values.rename(
        columns={protocols["stn"]: "STN3m", protocols["stn_snr"]: "STNSNr3m"}
    )
    baseline = (
        selected.groupby(["ID", "Scale"], observed=True)["Baseline"]
        .first()
        .rename("Baseline")
    )
    endpoints = baseline.to_frame().join(values, how="outer").reset_index()
    scale_metadata = pd.DataFrame(
        [
            {
                "Scale": item["value"],
                "ScaleLabel": item["label"],
                "ScaleDirection": item["direction"],
                "ScaleOrder": order,
            }
            for order, item in enumerate(config["scales"])
        ]
    )
    endpoints = endpoints.merge(scale_metadata, on="Scale", validate="many_to_one")
    endpoints["IncrementalBenefit3m"] = np.where(
        endpoints["ScaleDirection"].eq("higher_is_better"),
        endpoints["STNSNr3m"] - endpoints["STN3m"],
        endpoints["STN3m"] - endpoints["STNSNr3m"],
    )
    endpoints = endpoints.sort_values(["ScaleOrder", "ID"]).reset_index(drop=True)
    output_columns = [
        "ID",
        "Scale",
        "ScaleLabel",
        "ScaleDirection",
        "ScaleOrder",
        "Baseline",
        "STN3m",
        "STNSNr3m",
        "IncrementalBenefit3m",
    ]
    endpoints = endpoints.loc[:, output_columns]
    return endpoints, {
        "InputType": "clinical_workbook",
        "Domain": "clinical",
        "Metric": "scale_score",
        "FeatureOutput": "3m",
        "Path": str(config.clinical_workbook),
        "Rows": len(source),
        "Columns": len(source.columns),
    }


def build_feature_specifications(config: ClinicalCorrelationConfig) -> pd.DataFrame:
    """Build the ordered local and connectivity feature registry."""

    features = config["features"]
    bands = features["bands"]
    rows: list[dict[str, Any]] = []
    local = features["local"]
    for region in local["regions"]:
        feature_order = 0
        for band in bands:
            for item in local["banded"]:
                rows.append(
                    {
                        "Domain": "local",
                        "Metric": item["metric"],
                        "FeatureOutput": item["feature_output"],
                        "Band": band["value"],
                        "RegionVariable": local["region_column"],
                        "RegionValue": region,
                        "DisplayRegion": region,
                        "FeatureOrder": feature_order,
                        "FeatureLabel": f"{band['label']} {item['label']}",
                    }
                )
                feature_order += 1
        aperiodic = local["aperiodic"]
        for parameter in aperiodic["parameters"]:
            rows.append(
                {
                    "Domain": "local",
                    "Metric": aperiodic["metric"],
                    "FeatureOutput": aperiodic["feature_output"],
                    "Band": parameter["value"],
                    "RegionVariable": local["region_column"],
                    "RegionValue": region,
                    "DisplayRegion": region,
                    "FeatureOrder": feature_order,
                    "FeatureLabel": parameter["label"],
                }
            )
            feature_order += 1

    connectivity = features["connectivity"]
    feature_order = 0
    for band in bands:
        for item in connectivity["metrics"]:
            rows.append(
                {
                    "Domain": "connectivity",
                    "Metric": item["metric"],
                    "FeatureOutput": item["feature_output"],
                    "Band": band["value"],
                    "RegionVariable": connectivity["region_column"],
                    "RegionValue": item["source_region"],
                    "DisplayRegion": connectivity["display_region"],
                    "FeatureOrder": feature_order,
                    "FeatureLabel": f"{band['label']} {item['label']}",
                }
            )
            feature_order += 1
    specifications = pd.DataFrame(rows)
    key = [
        "Domain",
        "Metric",
        "FeatureOutput",
        "Band",
        "RegionValue",
    ]
    if specifications.duplicated(key).any():
        raise ValueError("Feature registry contains duplicate source identities.")
    return specifications


def _predictor_id(row: pd.Series) -> str:
    parts = {
        "domain": row["Domain"],
        "metric": row["Metric"],
        "feature-output": row["FeatureOutput"],
        "band": row["Band"],
        "region": row["RegionValue"],
        "polar": row["Polar"],
        "phase": row["Phase"],
        "lat": row["Lat"],
    }
    return "_".join(
        f"{key}-{str(value).replace('_', '-').replace(' ', '-')}"
        for key, value in parts.items()
    )


def _predictor_registry(
    specifications: pd.DataFrame, config: ClinicalCorrelationConfig
) -> pd.DataFrame:
    levels = config["inputs"]["lfp"]
    derivation = levels.get("phase_derivation")
    phase_metadata = (
        [
            {
                "Phase": item["phase"],
                "FromPhase": item["from_phase"],
                "ToPhase": item["to_phase"],
                "PhaseContrast": item["label"],
                "PredictorDefinition": derivation["mode"],
            }
            for item in derivation["contrasts"]
        ]
        if derivation is not None
        else [{"Phase": phase} for phase in levels["phases"]]
    )
    rows = []
    for _, spec in specifications.iterrows():
        for polar in levels["polarity"]:
            for phase_fields in phase_metadata:
                for lat in levels["laterality"]:
                    rows.append(
                        {
                            **spec.to_dict(),
                            "Polar": polar,
                            **phase_fields,
                            "Lat": lat,
                        }
                    )
    registry = pd.DataFrame(rows)
    registry["PredictorID"] = registry.apply(_predictor_id, axis=1)
    if registry["PredictorID"].duplicated().any():
        raise ValueError("Predictor registry contains duplicate identities.")
    return registry


def derive_adjacent_phase_values(
    source_rows: pd.DataFrame, phase_derivation: dict[str, Any]
) -> pd.DataFrame:
    """Pair normalized phase rows and subtract each immediately preceding phase."""

    required = set(ADJACENT_IDENTITY_COLUMNS) | {"Phase", "Value"}
    require_columns(source_rows, required, "Adjacent-phase LFP source")
    if source_rows.duplicated([*ADJACENT_IDENTITY_COLUMNS, "Phase"]).any():
        raise ValueError("Adjacent-phase LFP source contains duplicate phase rows.")

    frames: list[pd.DataFrame] = []
    for contrast in phase_derivation["contrasts"]:
        from_phase = str(contrast["from_phase"])
        to_phase = str(contrast["to_phase"])
        from_rows = source_rows.loc[
            source_rows["Phase"].eq(from_phase),
            [*ADJACENT_IDENTITY_COLUMNS, "Value"],
        ].rename(columns={"Value": "FromValue"})
        to_rows = source_rows.loc[
            source_rows["Phase"].eq(to_phase),
            [*ADJACENT_IDENTITY_COLUMNS, "Value"],
        ].rename(columns={"Value": "ToValue"})
        paired = from_rows.merge(
            to_rows,
            on=ADJACENT_IDENTITY_COLUMNS,
            how="outer",
            validate="one_to_one",
        )
        finite = np.isfinite(paired["FromValue"]) & np.isfinite(paired["ToValue"])
        paired["Value"] = np.where(
            finite,
            paired["ToValue"] - paired["FromValue"],
            np.nan,
        )
        paired["Phase"] = str(contrast["phase"])
        paired["FromPhase"] = from_phase
        paired["ToPhase"] = to_phase
        paired["PhaseContrast"] = str(contrast["label"])
        paired["PredictorDefinition"] = str(phase_derivation["mode"])
        paired["PredictorID"] = paired.apply(_predictor_id, axis=1)
        frames.append(paired)

    derived = pd.concat(frames, ignore_index=True)
    if derived.duplicated(["ID", "PredictorID"]).any():
        raise ValueError("Adjacent-phase predictors contain duplicate patient rows.")
    return derived.sort_values(["PredictorID", "ID"]).reset_index(drop=True)


def load_lfp_predictors(
    config: ClinicalCorrelationConfig,
    *,
    retain_provenance: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Select manifest-backed normalized region scalars and build predictor rows."""

    manifest = pd.read_csv(config.feature_manifest)
    require_columns(
        manifest, MANIFEST_COLUMNS, f"Feature manifest {config.feature_manifest}"
    )
    select = pd.Series(True, index=manifest.index)
    for column, value in config["inputs"]["lfp"]["manifest_select"].items():
        select &= manifest[column].eq(value)
    candidates = manifest.loc[select].copy()
    specifications = build_feature_specifications(config)
    source_specs = specifications[
        ["Domain", "Metric", "FeatureOutput"]
    ].drop_duplicates()
    selected_manifest = candidates.merge(
        source_specs,
        on=["Domain", "Metric", "FeatureOutput"],
        how="inner",
        validate="one_to_one",
    )
    if len(selected_manifest) != len(source_specs):
        raise ValueError(
            f"Expected {len(source_specs)} normalized region scalar inputs; found {len(selected_manifest)}."
        )

    input_rows: list[dict[str, Any]] = [
        {
            "InputType": "feature_manifest",
            "Domain": "manifest",
            "Metric": "feature_outputs",
            "FeatureOutput": "csv",
            "Path": str(config.feature_manifest),
            "Rows": len(manifest),
            "Columns": len(manifest.columns),
        }
    ]
    value_rows: list[pd.DataFrame] = []
    levels = config["inputs"]["lfp"]
    for _, manifest_row in selected_manifest.sort_values(
        ["Domain", "Metric", "FeatureOutput"]
    ).iterrows():
        source_path = path_below(
            Path(str(manifest_row["Path"])), config.lfp_root, "LFP source"
        )
        table = read_stage_table(source_path)
        if len(table) != int(manifest_row["Rows"]):
            raise ValueError(f"LFP row count differs from its manifest: {source_path}")
        domain = str(manifest_row["Domain"])
        region_column = "Region" if domain == "local" else "PairRegion"
        required = {
            "ID",
            "Domain",
            "Metric",
            "FeatureOutput",
            "Representation",
            "Band",
            region_column,
            "Polar",
            "Phase",
            "Lat",
            "Value",
            "IncludeAggregate",
        }
        require_columns(table, required, f"LFP source {source_path}")
        include_values = set(table["IncludeAggregate"].dropna().unique())
        if not include_values.issubset({True, False, np.bool_(True), np.bool_(False)}):
            raise TypeError(f"IncludeAggregate must be boolean: {source_path}")
        numeric = _numeric_column(table, "Value", f"LFP source {source_path}")
        source_specs_subset = specifications.loc[
            specifications["Domain"].eq(domain)
            & specifications["Metric"].eq(manifest_row["Metric"])
            & specifications["FeatureOutput"].eq(manifest_row["FeatureOutput"])
        ].copy()
        source_rows = table.copy()
        source_rows["Value"] = numeric
        source_phases = (
            levels["phase_derivation"]["source_phases"]
            if levels.get("phase_derivation") is not None
            else levels["phases"]
        )
        source_rows = source_rows.loc[
            source_rows["IncludeAggregate"].fillna(False).astype(bool)
            & source_rows["Polar"].isin(levels["polarity"])
            & source_rows["Phase"].isin(source_phases)
            & source_rows["Lat"].isin(levels["laterality"])
        ].copy()
        source_rows = source_rows.rename(columns={region_column: "RegionValue"})
        source_rows["ID"] = _normalize_ids(source_rows["ID"])
        selected_rows = source_rows.merge(
            source_specs_subset,
            on=["Domain", "Metric", "FeatureOutput", "Band", "RegionValue"],
            how="inner",
            validate="many_to_one",
        )
        if retain_provenance:
            provenance_columns = ["Record", "Trial", "StimSide", "space", "atlas"]
            require_columns(selected_rows, set(provenance_columns), str(source_path))
            if domain == "connectivity":
                require_columns(selected_rows, {"PairDirection"}, str(source_path))
            else:
                selected_rows["PairDirection"] = pd.NA
            selected_rows["SourcePath"] = str(source_path)
            provenance_columns += ["PairDirection", "SourcePath"]
            provenance = selected_rows[
                [*ADJACENT_IDENTITY_COLUMNS, *provenance_columns]
            ].drop_duplicates()
        phase_derivation = levels.get("phase_derivation")
        if phase_derivation is None:
            selected_rows["PredictorID"] = selected_rows.apply(_predictor_id, axis=1)
            if selected_rows.duplicated(["ID", "PredictorID"]).any():
                raise ValueError(
                    f"LFP source contains duplicate patient predictors: {source_path}"
                )
            value_rows.append(selected_rows[["ID", "PredictorID", "Value"]])
        else:
            derived = derive_adjacent_phase_values(selected_rows, phase_derivation)
            if retain_provenance:
                derived = derived.merge(
                    provenance,
                    on=ADJACENT_IDENTITY_COLUMNS,
                    how="left",
                    validate="many_to_one",
                )
            value_rows.append(derived)
        input_rows.append(
            {
                "InputType": "lfp_table",
                "Domain": domain,
                "Metric": manifest_row["Metric"],
                "FeatureOutput": manifest_row["FeatureOutput"],
                "Path": str(source_path),
                "Rows": len(table),
                "Columns": len(table.columns),
            }
        )

    predictor_metadata = _predictor_registry(specifications, config)
    predictor_values = pd.concat(value_rows, ignore_index=True)
    unknown = sorted(
        set(predictor_values["PredictorID"]) - set(predictor_metadata["PredictorID"])
    )
    if unknown:
        raise ValueError(f"LFP rows contain unregistered predictors: {unknown[:5]}")
    return predictor_metadata, predictor_values, input_rows


def load_clinical_correlation_inputs(
    config: ClinicalCorrelationConfig,
) -> ClinicalCorrelationInputs:
    """Load all registered external inputs once and return aligned tables."""

    endpoints, clinical_manifest = load_clinical_endpoints(config)
    predictor_metadata, predictor_values, lfp_manifest = load_lfp_predictors(config)
    clinical_ids = set(endpoints["ID"])
    lfp_ids = set(predictor_values["ID"])
    if not clinical_ids.intersection(lfp_ids):
        raise ValueError("Clinical and LFP inputs have no shared patient IDs.")
    inputs_manifest = pd.DataFrame([clinical_manifest, *lfp_manifest])
    return ClinicalCorrelationInputs(
        clinical_endpoints=endpoints,
        predictor_metadata=predictor_metadata,
        predictor_values=predictor_values,
        inputs_manifest=inputs_manifest,
    )
