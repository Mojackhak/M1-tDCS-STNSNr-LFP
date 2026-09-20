"""Fit and publish the appended contact-level structural-connectivity models."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any

import numpy as np
import pandas as pd
from pypdf import PdfReader

from lfp_clinical.correlation import _benjamini_hochberg
from lfp_viz import visualdf
from lfp_viz.config import VizConfig, load_viz_config
from lfp_viz.scalar import y_axis_label

from .artifacts import atomic_csv, prepare_output
from .config import StructuralConnectivityConfig
from .statistics import _eligibility_status, _read_lift

CONTACT_LMM_TABLES = (
    "lift_predictors.csv",
    "model_rows.csv",
    "models.csv",
    "model_fit.csv",
    "fixed_effects.csv",
    "omnibus_tests.csv",
    "phase_slopes.csv",
    "prediction_grid.csv",
    "outcome_sd.csv",
    "coverage.csv",
    "multiplicity_families.csv",
    "inputs.csv",
    "outputs.csv",
    "figures.csv",
    "tables.csv",
)

MODEL_KEY = [
    "OutcomeDomain",
    "ConnectomeSource",
    "LiftMetric",
    "LiftPredictor",
    "Polar",
    "Metric",
    "FeatureOutput",
    "Region",
    "PairRegion",
    "Band",
]
STANDARDIZATION_KEY = [
    "OutcomeDomain",
    "ConnectomeSource",
    "LiftMetric",
    "LiftPredictor",
    "Region",
    "PairRegion",
]
UNIT_KEY = [*STANDARDIZATION_KEY, "ID", "ContactUnitID"]
OUTCOME_SD_KEY = [
    "OutcomeDomain",
    "Metric",
    "FeatureOutput",
    "OutcomeScale",
    "Polar",
    "Band",
    "Region",
    "PairRegion",
]
OMNIBUS_TERMS = ("Phase", "X_z", "Phase:X_z")
HEATMAP_CONTEXTS = ("STN", "SNr", "RegionMean")
HEATMAP_ORIENTATIONS = ("band-by-metric", "metric-by-band")
SINGLE_ATLAS_LAYOUT = "single-atlas"
ATLAS_GRID_LAYOUT = "atlas-grid"
FIELD_EXCESS_METRIC = "FieldExcess"
FIELD_EXCESS_FINDER_HEATMAP_COUNT = 6
FIELD_EXCESS_FINDER_RENDERED_FIT_COUNT = 94
FIELD_EXCESS_FINDER_FIXED_FIT_COUNT = 24
FIELD_EXCESS_FINDER_ATLAS_GRID_FIT_COUNT = 8
FIELD_EXCESS_FINDER_CONNECTOME_SOURCES = (
    "individualized_dti",
    "mgh_usc_hcp_32_horn_2017",
    "ppmi_85_ewert_2017",
    "dtor_985_full_elias_2024",
)
FIELD_EXCESS_FINDER_PREDICTORS = ("STN", "SNr")
FIELD_EXCESS_FINDER_POLAR = "Cathodal"
FIELD_EXCESS_FINDER_PHASE = "Late"
FIELD_EXCESS_FINDER_BAND = "delta"
FIELD_EXCESS_FINDER_METRICS = ("ciplv", "imcoh_abs", "wpli")
FIELD_EXCESS_FINDER_LOCAL_PHASES = ("Late", "Post")
FIELD_EXCESS_FINDER_LOCAL_METRIC = "burst"
FIELD_EXCESS_FINDER_LOCAL_FEATURE_OUTPUT = "mean-scalar"
FIELD_EXCESS_FINDER_LOCAL_REGION = "SNr"
FIELD_EXCESS_ATLAS_GRID_FIGURE_TYPE = "slope_atlas_grid"
SEED_FIELD_EXPOSURE_METRIC = "SeedFieldExposure"
SEED_COMPONENT = "SeedThresholdedMeanPeakE_Vm"
REGION_COMPONENT = "RegionThresholdedMeanPeakE_Vm"
COMPONENT_COLUMNS = (SEED_COMPONENT, REGION_COMPONENT)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(
            f"Required structural-connectivity table is absent: {path}"
        )
    return pd.read_csv(path, low_memory=False)


def _field_excess_rows(
    config: StructuralConnectivityConfig,
    local_rows: pd.DataFrame,
    network_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    enriched = _read_lift(config)
    components = enriched[
        [
            "ID",
            "StimSide",
            "ConnectomeSource",
            "Channel",
            "Region",
            *COMPONENT_COLUMNS,
            "FieldExcessE_Vm",
            "FieldExcessStatus",
        ]
    ].copy()

    local = local_rows.loc[local_rows["LiftMetric"].eq("FieldWeightedLift")].copy()
    local["LiftMetric"] = FIELD_EXCESS_METRIC
    local = local.merge(
        components,
        on=["ID", "StimSide", "ConnectomeSource", "Channel", "Region"],
        how="left",
        validate="many_to_one",
    )
    local["Lift"] = pd.to_numeric(local.pop("FieldExcessE_Vm"), errors="coerce")
    local["LiftStatus"] = local.pop("FieldExcessStatus")
    local["EligibilityStatus"] = _eligibility_status(
        local["Value"], local["Lift"], local["LiftStatus"]
    )

    network = network_rows.loc[
        network_rows["LiftMetric"].eq("FieldWeightedLift")
    ].copy()
    network["LiftMetric"] = FIELD_EXCESS_METRIC
    for region in ("STN", "SNr"):
        endpoint = components.loc[components["Region"].eq(region)].drop(
            columns="Region"
        )
        endpoint = endpoint.rename(
            columns={
                "Channel": f"{region}Channel",
                SEED_COMPONENT: f"{region}{SEED_COMPONENT}",
                REGION_COMPONENT: f"{region}{REGION_COMPONENT}",
                "FieldExcessE_Vm": f"{region}FieldExcessE_Vm",
                "FieldExcessStatus": f"{region}FieldExcessStatus",
            }
        )
        network = network.merge(
            endpoint,
            on=["ID", "StimSide", "ConnectomeSource", f"{region}Channel"],
            how="left",
            validate="many_to_one",
        )
        network[f"{region}Lift"] = pd.to_numeric(
            network.pop(f"{region}FieldExcessE_Vm"), errors="coerce"
        )
        network[f"{region}LiftStatus"] = network.pop(f"{region}FieldExcessStatus")
    network["RegionMeanLift"] = (
        pd.to_numeric(network["STNLift"], errors="coerce")
        + pd.to_numeric(network["SNrLift"], errors="coerce")
    ) / 2.0
    outcome_finite = np.isfinite(pd.to_numeric(network["Value"], errors="coerce"))
    stn_ok = network["STNLiftStatus"].eq("ok") & np.isfinite(
        pd.to_numeric(network["STNLift"], errors="coerce")
    )
    snr_ok = network["SNrLiftStatus"].eq("ok") & np.isfinite(
        pd.to_numeric(network["SNrLift"], errors="coerce")
    )
    status = pd.Series("eligible", index=network.index, dtype="string")
    status.loc[~outcome_finite] = "nonfinite_outcome"
    status.loc[outcome_finite & ~stn_ok] = "missing_or_nonfinite_stn_lift"
    status.loc[outcome_finite & stn_ok & ~snr_ok] = "missing_or_nonfinite_snr_lift"
    network["EligibilityStatus"] = status
    return local, network


def _seed_field_exposure_status(table: pd.DataFrame) -> pd.Series:
    seed_count = pd.to_numeric(table["NSeed"], errors="coerce")
    seed_exposure = pd.to_numeric(table[SEED_COMPONENT], errors="coerce")
    status = pd.Series("ok", index=table.index, dtype="string")
    valid_count = np.isfinite(seed_count) & seed_count.gt(0)
    status.loc[~valid_count] = "missing_no_seed_streamlines"
    status.loc[valid_count & ~np.isfinite(seed_exposure)] = (
        "nonfinite_seed_field_exposure"
    )
    return status


def _seed_field_exposure_rows(
    config: StructuralConnectivityConfig,
    local_rows: pd.DataFrame,
    network_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    enriched = _read_lift(config)
    components = enriched[
        [
            "ID",
            "StimSide",
            "ConnectomeSource",
            "Channel",
            "Region",
            "NSeed",
            *COMPONENT_COLUMNS,
        ]
    ].copy()
    components["SeedFieldExposureStatus"] = _seed_field_exposure_status(components)
    components = components.drop(columns="NSeed")

    local = local_rows.loc[local_rows["LiftMetric"].eq("FieldWeightedLift")].copy()
    local["LiftMetric"] = SEED_FIELD_EXPOSURE_METRIC
    local = local.merge(
        components,
        on=["ID", "StimSide", "ConnectomeSource", "Channel", "Region"],
        how="left",
        validate="many_to_one",
    )
    local["Lift"] = pd.to_numeric(local[SEED_COMPONENT], errors="coerce")
    local["LiftStatus"] = local.pop("SeedFieldExposureStatus")
    local["EligibilityStatus"] = _eligibility_status(
        local["Value"], local["Lift"], local["LiftStatus"]
    )

    network = network_rows.loc[
        network_rows["LiftMetric"].eq("FieldWeightedLift")
    ].copy()
    network["LiftMetric"] = SEED_FIELD_EXPOSURE_METRIC
    for region in ("STN", "SNr"):
        endpoint = components.loc[components["Region"].eq(region)].drop(
            columns="Region"
        )
        endpoint = endpoint.rename(
            columns={
                "Channel": f"{region}Channel",
                SEED_COMPONENT: f"{region}{SEED_COMPONENT}",
                REGION_COMPONENT: f"{region}{REGION_COMPONENT}",
                "SeedFieldExposureStatus": f"{region}SeedFieldExposureStatus",
            }
        )
        network = network.merge(
            endpoint,
            on=["ID", "StimSide", "ConnectomeSource", f"{region}Channel"],
            how="left",
            validate="many_to_one",
        )
        network[f"{region}Lift"] = pd.to_numeric(
            network[f"{region}{SEED_COMPONENT}"], errors="coerce"
        )
        network[f"{region}LiftStatus"] = network.pop(f"{region}SeedFieldExposureStatus")
    network["RegionMeanLift"] = (
        pd.to_numeric(network["STNLift"], errors="coerce")
        + pd.to_numeric(network["SNrLift"], errors="coerce")
    ) / 2.0
    outcome_finite = np.isfinite(pd.to_numeric(network["Value"], errors="coerce"))
    stn_ok = network["STNLiftStatus"].eq("ok") & np.isfinite(
        pd.to_numeric(network["STNLift"], errors="coerce")
    )
    snr_ok = network["SNrLiftStatus"].eq("ok") & np.isfinite(
        pd.to_numeric(network["SNrLift"], errors="coerce")
    )
    status = pd.Series("eligible", index=network.index, dtype="string")
    status.loc[~outcome_finite] = "nonfinite_outcome"
    status.loc[outcome_finite & ~stn_ok] = "missing_or_nonfinite_stn_lift"
    status.loc[outcome_finite & stn_ok & ~snr_ok] = "missing_or_nonfinite_snr_lift"
    network["EligibilityStatus"] = status
    return local, network


def _contact_lmm_input_rows(
    config: StructuralConnectivityConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    local = _read_csv(config.output_path("local_rows"))
    network = _read_csv(config.output_path("connectivity_rows"))
    expected_existing = set(config["inference"]["lift_metrics"])
    for name, table in (("local_rows.csv", local), ("connectivity_rows.csv", network)):
        if "LiftMetric" not in table:
            raise KeyError(f"{name} is missing LiftMetric.")
        observed = set(table["LiftMetric"].dropna().astype(str))
        if observed != expected_existing:
            raise ValueError(
                f"{name} does not contain the registered Spearman metrics: "
                f"{sorted(observed)}"
            )

    registered = list(config["inference"]["contact_lmm"]["predictor_metrics"])
    supported = {
        *expected_existing,
        FIELD_EXCESS_METRIC,
        SEED_FIELD_EXPOSURE_METRIC,
    }
    unsupported = set(registered).difference(supported)
    if unsupported:
        raise ValueError(
            f"Unsupported contact-level predictor metrics: {sorted(unsupported)}"
        )
    local_frames = [local.loc[local["LiftMetric"].isin(registered)].copy()]
    network_frames = [network.loc[network["LiftMetric"].isin(registered)].copy()]
    if FIELD_EXCESS_METRIC in registered:
        local_excess, network_excess = _field_excess_rows(config, local, network)
        local_frames.append(local_excess)
        network_frames.append(network_excess)
    if SEED_FIELD_EXPOSURE_METRIC in registered:
        local_seed, network_seed = _seed_field_exposure_rows(config, local, network)
        local_frames.append(local_seed)
        network_frames.append(network_seed)
    local_output = pd.concat(local_frames, ignore_index=True, sort=False)
    network_output = pd.concat(network_frames, ignore_index=True, sort=False)
    for column in COMPONENT_COLUMNS:
        if column not in local_output:
            local_output[column] = np.nan
    for region in ("STN", "SNr"):
        for component in COMPONENT_COLUMNS:
            column = f"{region}{component}"
            if column not in network_output:
                network_output[column] = np.nan
    return local_output, network_output


def _require_columns(table: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = sorted(set(required).difference(table.columns))
    if missing:
        raise KeyError(f"{name} is missing columns: {missing}")


def _predictor_status(status: pd.Series, values: pd.Series) -> pd.Series:
    output = status.fillna("missing_lift_row").astype(str)
    finite = np.isfinite(pd.to_numeric(values, errors="coerce"))
    output = output.where(~status.eq("ok"), "eligible")
    output = output.where(~(status.eq("ok") & ~finite), "nonfinite_lift")
    return output


def _prepare_long_rows(
    local_rows: pd.DataFrame,
    connectivity_rows: pd.DataFrame,
) -> pd.DataFrame:
    common = {
        "LiftMetric",
        "ID",
        "StimSide",
        "ConnectomeSource",
        "Metric",
        "FeatureOutput",
        "OutcomeScale",
        "Polar",
        "Phase",
        "Band",
        "Value",
    }
    _require_columns(
        local_rows,
        common
        | {
            "Region",
            "Channel",
            "Lift",
            "LiftStatus",
            *COMPONENT_COLUMNS,
        },
        "local_rows.csv",
    )
    _require_columns(
        connectivity_rows,
        common
        | {
            "PairRegion",
            "ChannelPair",
            "channel_a",
            "channel_b",
            "STNLift",
            "SNrLift",
            "STNLiftStatus",
            "SNrLiftStatus",
            *(
                f"{region}{component}"
                for region in ("STN", "SNr")
                for component in COMPONENT_COLUMNS
            ),
        },
        "connectivity_rows.csv",
    )

    local = local_rows[
        [
            "LiftMetric",
            "ID",
            "StimSide",
            "ConnectomeSource",
            "Metric",
            "FeatureOutput",
            "OutcomeScale",
            "Polar",
            "Region",
            "Phase",
            "Band",
            "Channel",
            "Lift",
            "LiftStatus",
            *COMPONENT_COLUMNS,
            "Value",
        ]
    ].copy()
    local["OutcomeDomain"] = "local"
    local["PairRegion"] = ""
    local["LiftPredictor"] = local["Region"].astype(str)
    local["ContactUnitID"] = local["Channel"].astype(str)
    local["ChannelPair"] = ""
    local["X_raw"] = pd.to_numeric(local["Lift"], errors="coerce")
    local["PredictorStatus"] = _predictor_status(local["LiftStatus"], local["X_raw"])

    network_frames: list[pd.DataFrame] = []
    network_base_columns = [
        "LiftMetric",
        "ID",
        "StimSide",
        "ConnectomeSource",
        "Metric",
        "FeatureOutput",
        "OutcomeScale",
        "Polar",
        "PairRegion",
        "Phase",
        "Band",
        "ChannelPair",
        "channel_a",
        "channel_b",
        "STNLift",
        "SNrLift",
        "STNLiftStatus",
        "SNrLiftStatus",
        *(
            f"{region}{component}"
            for region in ("STN", "SNr")
            for component in COMPONENT_COLUMNS
        ),
        "Value",
    ]
    for predictor in ("STN", "SNr", "RegionMean"):
        network = connectivity_rows[network_base_columns].copy()
        network["OutcomeDomain"] = "connectivity"
        network["Region"] = ""
        network["Channel"] = ""
        network["LiftPredictor"] = predictor
        network["ContactUnitID"] = (
            network["channel_a"].astype(str) + "|" + network["channel_b"].astype(str)
        )
        stn = pd.to_numeric(network["STNLift"], errors="coerce")
        snr = pd.to_numeric(network["SNrLift"], errors="coerce")
        if predictor == "STN":
            network["X_raw"] = stn
            network[SEED_COMPONENT] = network[f"STN{SEED_COMPONENT}"]
            network[REGION_COMPONENT] = network[f"STN{REGION_COMPONENT}"]
            network["PredictorStatus"] = _predictor_status(
                network["STNLiftStatus"], stn
            )
        elif predictor == "SNr":
            network["X_raw"] = snr
            network[SEED_COMPONENT] = network[f"SNr{SEED_COMPONENT}"]
            network[REGION_COMPONENT] = network[f"SNr{REGION_COMPONENT}"]
            network["PredictorStatus"] = _predictor_status(
                network["SNrLiftStatus"], snr
            )
        else:
            network["X_raw"] = (stn + snr) / 2.0
            network[SEED_COMPONENT] = (
                pd.to_numeric(network[f"STN{SEED_COMPONENT}"], errors="coerce")
                + pd.to_numeric(network[f"SNr{SEED_COMPONENT}"], errors="coerce")
            ) / 2.0
            network[REGION_COMPONENT] = (
                pd.to_numeric(network[f"STN{REGION_COMPONENT}"], errors="coerce")
                + pd.to_numeric(network[f"SNr{REGION_COMPONENT}"], errors="coerce")
            ) / 2.0
            stn_ok = network["STNLiftStatus"].eq("ok") & np.isfinite(stn)
            snr_ok = network["SNrLiftStatus"].eq("ok") & np.isfinite(snr)
            status = pd.Series("eligible", index=network.index, dtype="string")
            status.loc[~stn_ok] = "missing_or_nonfinite_stn_lift"
            status.loc[stn_ok & ~snr_ok] = "missing_or_nonfinite_snr_lift"
            network["PredictorStatus"] = status
        network_frames.append(network)

    output = pd.concat([local, *network_frames], ignore_index=True, sort=False)
    output["Value"] = pd.to_numeric(output["Value"], errors="coerce")
    for column in ("Region", "PairRegion", "Channel", "ChannelPair"):
        output[column] = output[column].fillna("").astype(str)
    columns = [
        "OutcomeDomain",
        "ConnectomeSource",
        "LiftMetric",
        "LiftPredictor",
        "ID",
        "StimSide",
        "ContactUnitID",
        "Channel",
        "ChannelPair",
        "Metric",
        "FeatureOutput",
        "OutcomeScale",
        "Polar",
        "Region",
        "PairRegion",
        "Phase",
        "Band",
        "X_raw",
        *COMPONENT_COLUMNS,
        "PredictorStatus",
        "Value",
    ]
    prepared = output[columns]
    excess = prepared["LiftMetric"].eq(FIELD_EXCESS_METRIC)
    if bool(excess.any()):
        expected = pd.to_numeric(
            prepared.loc[excess, SEED_COMPONENT], errors="coerce"
        ) - pd.to_numeric(prepared.loc[excess, REGION_COMPONENT], errors="coerce")
        observed = pd.to_numeric(prepared.loc[excess, "X_raw"], errors="coerce")
        finite = np.isfinite(expected) & np.isfinite(observed)
        if not bool(finite.all()) or not np.allclose(
            expected[finite], observed[finite], atol=1e-15, rtol=1e-12
        ):
            raise ValueError(
                "FieldExcess rows do not equal seed minus Region exposure."
            )
    seed_exposure = prepared["LiftMetric"].eq(SEED_FIELD_EXPOSURE_METRIC)
    if bool(seed_exposure.any()):
        expected = pd.to_numeric(
            prepared.loc[seed_exposure, SEED_COMPONENT], errors="coerce"
        )
        observed = pd.to_numeric(prepared.loc[seed_exposure, "X_raw"], errors="coerce")
        finite = np.isfinite(expected) & np.isfinite(observed)
        if not bool(finite.all()) or not np.allclose(
            expected[finite], observed[finite], atol=1e-15, rtol=1e-12
        ):
            raise ValueError("SeedFieldExposure rows do not equal seed field exposure.")
    return prepared.sort_values(
        [*MODEL_KEY, "ID", "ContactUnitID", "Phase"], kind="mergesort"
    ).reset_index(drop=True)


def _standardize_predictors(
    rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouped = rows.groupby(UNIT_KEY, sort=False, dropna=False, observed=True)
    x_counts = grouped["X_raw"].nunique(dropna=True)
    if bool((x_counts > 1).any()):
        bad = x_counts.loc[x_counts > 1].index[0]
        raise ValueError(
            f"Repeated contact rows carry different raw Lift values: {bad}"
        )
    status_counts = grouped["PredictorStatus"].nunique(dropna=False)
    if bool((status_counts > 1).any()):
        bad = status_counts.loc[status_counts > 1].index[0]
        raise ValueError(f"Repeated contact rows carry different Lift statuses: {bad}")
    for component in COMPONENT_COLUMNS:
        component_counts = grouped[component].nunique(dropna=True)
        if bool((component_counts > 1).any()):
            bad = component_counts.loc[component_counts > 1].index[0]
            raise ValueError(
                f"Repeated contact rows carry different {component} values: {bad}"
            )

    unit_columns = [
        *UNIT_KEY,
        "StimSide",
        "Channel",
        "ChannelPair",
        "X_raw",
        *COMPONENT_COLUMNS,
        "PredictorStatus",
    ]
    units = rows[unit_columns].drop_duplicates(UNIT_KEY, keep="first").copy()
    units["X_center"] = np.nan
    units["X_scale"] = np.nan
    units["X_z"] = np.nan
    units["X_standardization_n"] = 0
    units["XStandardizationStatus"] = "insufficient_finite_units"

    for _, group in units.groupby(
        STANDARDIZATION_KEY, sort=True, dropna=False, observed=True
    ):
        finite = group["PredictorStatus"].eq("eligible") & np.isfinite(
            pd.to_numeric(group["X_raw"], errors="coerce")
        )
        index = group.index
        values = group.loc[finite, "X_raw"].to_numpy(dtype=float)
        count = int(values.size)
        units.loc[index, "X_standardization_n"] = count
        if count == 0:
            continue
        center = float(np.mean(values))
        scale = float(np.std(values, ddof=1)) if count >= 2 else np.nan
        units.loc[index, "X_center"] = center
        units.loc[index, "X_scale"] = scale
        if count < 2 or not np.isfinite(scale) or scale <= 0:
            units.loc[index, "XStandardizationStatus"] = "constant_or_insufficient_x"
            continue
        units.loc[index, "XStandardizationStatus"] = "ok"
        finite_index = group.index[finite]
        units.loc[finite_index, "X_z"] = (
            units.loc[finite_index, "X_raw"] - center
        ) / scale

    merge_columns = [
        *UNIT_KEY,
        "X_center",
        "X_scale",
        "X_z",
        "X_standardization_n",
        "XStandardizationStatus",
    ]
    expanded = rows.merge(
        units[merge_columns],
        on=UNIT_KEY,
        how="left",
        validate="many_to_one",
    )
    status = pd.Series("eligible", index=expanded.index, dtype="string")
    status.loc[~np.isfinite(expanded["Value"])] = "nonfinite_outcome"
    bad_predictor = ~expanded["PredictorStatus"].eq("eligible")
    status.loc[bad_predictor] = expanded.loc[bad_predictor, "PredictorStatus"]
    bad_standardization = expanded["PredictorStatus"].eq("eligible") & ~expanded[
        "XStandardizationStatus"
    ].eq("ok")
    status.loc[bad_standardization] = expanded.loc[
        bad_standardization, "XStandardizationStatus"
    ]
    status.loc[
        expanded["PredictorStatus"].eq("eligible")
        & expanded["XStandardizationStatus"].eq("ok")
        & ~np.isfinite(expanded["X_z"])
    ] = "nonfinite_standardized_lift"
    expanded["RowEligibilityStatus"] = status
    return (
        units.sort_values(UNIT_KEY, kind="mergesort").reset_index(drop=True),
        expanded,
    )


def _planned_models(config: StructuralConnectivityConfig) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    sources = [value.connectome_id for value in config.connectomes]
    registered_metrics = list(config["inference"]["contact_lmm"]["predictor_metrics"])
    existing_metrics = [
        metric
        for metric in config["inference"]["lift_metrics"]
        if metric in registered_metrics
    ]
    appended_metrics = [
        metric for metric in registered_metrics if metric not in existing_metrics
    ]
    polarities = list(config["outcomes"]["polarities"])
    bands = list(config["outcomes"]["bands"])
    metric_groups = [existing_metrics, *([metric] for metric in appended_metrics)]
    for metric_group in metric_groups:
        for source in sources:
            for lift_metric in metric_group:
                for polar in polarities:
                    for region in ("STN", "SNr"):
                        for band in bands:
                            records.append(
                                {
                                    "OutcomeDomain": "local",
                                    "ConnectomeSource": source,
                                    "LiftMetric": lift_metric,
                                    "LiftPredictor": region,
                                    "Polar": polar,
                                    "Metric": "periodic",
                                    "FeatureOutput": "mean-scalar",
                                    "OutcomeScale": config["outcomes"]["local_scales"][
                                        "periodic"
                                    ],
                                    "Region": region,
                                    "PairRegion": "",
                                    "Band": band,
                                }
                            )
                    for metric in config["outcomes"]["connectivity_metrics"]:
                        for predictor in config["inference"][
                            "connectivity_lift_predictors"
                        ]:
                            for band in bands:
                                records.append(
                                    {
                                        "OutcomeDomain": "connectivity",
                                        "ConnectomeSource": source,
                                        "LiftMetric": lift_metric,
                                        "LiftPredictor": predictor,
                                        "Polar": polar,
                                        "Metric": metric,
                                        "FeatureOutput": "mean-scalar",
                                        "OutcomeScale": config["outcomes"][
                                            "connectivity_scales"
                                        ][metric],
                                        "Region": "",
                                        "PairRegion": config["outcomes"][
                                            "connectivity_pair_regions"
                                        ][metric],
                                        "Band": band,
                                    }
                                )
    models = pd.DataFrame(records)
    models.insert(
        0,
        "ModelID",
        [f"LMM{index:06d}" for index in range(1, len(models) + 1)],
    )
    models["Formula"] = config["inference"]["contact_lmm"]["formula"]
    return models


def _attach_models_and_support(
    rows: pd.DataFrame,
    models: pd.DataFrame,
    config: StructuralConnectivityConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_rows = rows.merge(
        models[["ModelID", *MODEL_KEY]],
        on=MODEL_KEY,
        how="left",
        validate="many_to_one",
    )
    if model_rows["ModelID"].isna().any():
        raise ValueError("Prepared contact rows contain an unregistered model cell.")
    minimum_n = int(config["inference"]["contact_lmm"]["minimum_n"])
    phases = tuple(config["inference"]["contact_lmm"]["phase_levels"])
    support_rows: list[dict[str, Any]] = []
    for model_id, group in model_rows.groupby("ModelID", sort=False, observed=True):
        finite = (
            group["PredictorStatus"].eq("eligible")
            & np.isfinite(group["X_raw"])
            & np.isfinite(group["Value"])
        )
        cell = group.loc[finite]
        phase_values = tuple(
            phase for phase in phases if phase in set(cell["Phase"].astype(str))
        )
        n_ids = int(cell["ID"].nunique())
        n_x = int(cell["X_raw"].nunique())
        standardization_ok = bool(cell["XStandardizationStatus"].eq("ok").any())
        if n_ids < minimum_n:
            status = "insufficient_n"
        elif phase_values != phases:
            status = "missing_phase"
        elif n_x < 2 or not standardization_ok:
            status = "constant_predictor"
        else:
            status = "eligible"
        support_rows.append(
            {
                "ModelID": model_id,
                "ModelEligibilityStatus": status,
                "n_rows": int(len(cell)),
                "n_ID": n_ids,
                "n_ContactUnitID": int(cell["ContactUnitID"].nunique()),
                "n_unique_X": n_x,
                "PhaseMembership": ";".join(phase_values),
                "PatientIDs": ";".join(sorted(cell["ID"].astype(str).unique())),
                "ContactUnitMembership": ";".join(
                    sorted(
                        f"{row.ID}:{row.ContactUnitID}"
                        for row in cell[["ID", "ContactUnitID"]]
                        .drop_duplicates()
                        .itertuples(index=False)
                    )
                ),
            }
        )
    support = pd.DataFrame(support_rows)
    models = models.merge(support, on="ModelID", how="left", validate="one_to_one")
    missing = models["ModelEligibilityStatus"].isna()
    models.loc[missing, "ModelEligibilityStatus"] = "insufficient_n"
    for column in ("n_rows", "n_ID", "n_ContactUnitID", "n_unique_X"):
        models[column] = models[column].fillna(0).astype(int)
    for column in ("PhaseMembership", "PatientIDs", "ContactUnitMembership"):
        models[column] = models[column].fillna("")
    model_rows = model_rows.sort_values(
        ["ModelID", "ID", "ContactUnitID", "Phase"], kind="mergesort"
    ).reset_index(drop=True)
    return model_rows, models


def _outcome_sd(
    local_rows: pd.DataFrame,
    connectivity_rows: pd.DataFrame,
) -> pd.DataFrame:
    local = local_rows[
        [
            "ID",
            "Metric",
            "FeatureOutput",
            "OutcomeScale",
            "Polar",
            "Band",
            "Region",
            "Phase",
            "Channel",
            "Value",
        ]
    ].copy()
    local["OutcomeDomain"] = "local"
    local["PairRegion"] = ""
    local["ContactUnitID"] = local["Channel"].astype(str)
    network = connectivity_rows[
        [
            "ID",
            "Metric",
            "FeatureOutput",
            "OutcomeScale",
            "Polar",
            "Band",
            "PairRegion",
            "Phase",
            "channel_a",
            "channel_b",
            "Value",
        ]
    ].copy()
    network["OutcomeDomain"] = "connectivity"
    network["Region"] = ""
    network["ContactUnitID"] = (
        network["channel_a"].astype(str) + "|" + network["channel_b"].astype(str)
    )
    base = pd.concat([local, network], ignore_index=True, sort=False)
    base["Value"] = pd.to_numeric(base["Value"], errors="coerce")
    base_key = [*OUTCOME_SD_KEY, "ID", "ContactUnitID", "Phase"]
    value_counts = base.groupby(base_key, sort=False, dropna=False, observed=True)[
        "Value"
    ].nunique(dropna=True)
    if bool((value_counts > 1).any()):
        bad = value_counts.loc[value_counts > 1].index[0]
        raise ValueError(f"Structural expansion changed an outcome value: {bad}")
    unique = base.drop_duplicates(base_key, keep="first")
    unique = unique.loc[np.isfinite(unique["Value"])].copy()
    output = (
        unique.groupby(OUTCOME_SD_KEY, sort=True, dropna=False, observed=True)["Value"]
        .agg(ValueSD=lambda values: values.std(ddof=1), ValueN="count")
        .reset_index()
    )
    output["ValueSDStatus"] = np.where(
        np.isfinite(output["ValueSD"]) & output["ValueSD"].gt(0),
        "ok",
        "nonpositive_or_undefined_sd",
    )
    return output.sort_values(OUTCOME_SD_KEY, kind="mergesort").reset_index(drop=True)


def _run_r_models(
    model_rows: pd.DataFrame,
    models: pd.DataFrame,
    config: StructuralConnectivityConfig,
    *,
    settings_key: str = "contact_lmm",
    predictor_column: str = "X_z",
    temp_path_key: str = "contact_lmm_temp_root",
) -> dict[str, pd.DataFrame]:
    eligible_ids = set(
        models.loc[models["ModelEligibilityStatus"].eq("eligible"), "ModelID"].astype(
            str
        )
    )
    r_rows = model_rows.loc[
        model_rows["ModelID"].isin(eligible_ids)
        & model_rows["RowEligibilityStatus"].eq("eligible"),
        [
            "ModelID",
            "Value",
            "Phase",
            predictor_column,
            "ID",
            "ContactUnitID",
        ],
    ].copy()
    if not eligible_ids:
        raise ValueError(
            "No contact-level mixed model satisfies the execution contract."
        )
    missing_ids = eligible_ids.difference(r_rows["ModelID"].astype(str))
    if missing_ids:
        raise ValueError(
            f"Eligible mixed models lack executable rows: {sorted(missing_ids)[:3]}"
        )

    script = Path(__file__).parent / "r" / "run_contact_lmm.R"
    if not script.is_file():
        raise FileNotFoundError(f"Contact-level R runner is absent: {script}")
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise FileNotFoundError("Rscript is unavailable in the active environment.")
    temporary_root = config.path_value(temp_path_key)
    temporary_root.mkdir(parents=True, exist_ok=True)
    settings = dict(config["inference"][settings_key])
    settings["predictor_column"] = predictor_column
    with tempfile.TemporaryDirectory(prefix="run-", dir=temporary_root) as name:
        run_root = Path(name)
        data_path = run_root / "model_rows.csv"
        settings_path = run_root / "settings.json"
        output_root = run_root / "r_outputs"
        r_rows.to_csv(data_path, index=False)
        settings_path.write_text(
            json.dumps(settings, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        result = subprocess.run(
            [
                rscript,
                str(script),
                str(data_path),
                str(settings_path),
                str(output_root),
            ],
            check=False,
            text=True,
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Contact-level R runner failed.\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
        names = {
            "model_fit": "model_fit.csv",
            "fixed_effects": "fixed_effects.csv",
            "omnibus_tests": "omnibus_tests.csv",
            "phase_slopes": "phase_slopes.csv",
            "phase_slope_contrasts": "phase_slope_contrasts.csv",
            "prediction_grid": "prediction_grid.csv",
        }
        return {
            key: pd.read_csv(output_root / filename, low_memory=False)
            for key, filename in names.items()
        }


def _metadata_merge(
    table: pd.DataFrame,
    models: pd.DataFrame,
) -> pd.DataFrame:
    metadata = [
        "ModelID",
        *MODEL_KEY,
        "OutcomeScale",
        "Formula",
        "ModelEligibilityStatus",
        "n_rows",
        "n_ID",
        "n_ContactUnitID",
        "n_unique_X",
        "PhaseMembership",
        "PatientIDs",
        "ContactUnitMembership",
    ]
    metadata.extend(
        column for column in ("OutcomeChangeBasis",) if column in models.columns
    )
    return models[metadata].merge(
        table, on="ModelID", how="left", validate="one_to_many"
    )


def _planned_inference_table(
    models: pd.DataFrame,
    observed: pd.DataFrame,
    *,
    dimension: str,
    levels: Iterable[str],
) -> pd.DataFrame:
    plan = models[["ModelID"]].merge(
        pd.DataFrame({dimension: list(levels)}), how="cross"
    )
    merged = plan.merge(
        observed,
        on=["ModelID", dimension],
        how="left",
        validate="one_to_one",
    )
    return _metadata_merge(merged, models)


def _result_status(table: pd.DataFrame) -> pd.Series:
    status = table["ModelEligibilityStatus"].astype(str).copy()
    eligible = table["ModelEligibilityStatus"].eq("eligible")
    if "FitStatus" in table:
        status.loc[eligible] = table.loc[eligible, "FitStatus"].fillna("fit_error")
    finite = np.isfinite(pd.to_numeric(table.get("P"), errors="coerce"))
    fitted = status.isin(["ok", "singular", "convergence_warning"])
    status.loc[fitted & ~finite] = "not_estimable"
    return status


def _apply_bh(
    table: pd.DataFrame,
    *,
    result_table: str,
    planned: int,
    planned_overrides: Mapping[tuple[str, str, str], int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = table.copy()
    output["Q"] = np.nan
    output["m_planned"] = planned
    output["m_tested"] = 0
    output["FamilyKey"] = ""
    family_records: list[dict[str, Any]] = []
    local_family = [
        "LiftMetric",
        "ConnectomeSource",
        "Polar",
        "OutcomeDomain",
        "Metric",
        "FeatureOutput",
        "Region",
        "LiftPredictor",
    ]
    network_family = [
        "LiftMetric",
        "ConnectomeSource",
        "Polar",
        "OutcomeDomain",
        "Metric",
        "FeatureOutput",
        "PairRegion",
        "LiftPredictor",
    ]
    if "OutcomeChangeBasis" in output.columns:
        local_family.insert(0, "OutcomeChangeBasis")
        network_family.insert(0, "OutcomeChangeBasis")
    if result_table == "omnibus_tests":
        local_family.append("Term")
        network_family.append("Term")
    for domain, columns in (
        ("local", local_family),
        ("connectivity", network_family),
    ):
        subset = output.loc[output["OutcomeDomain"].eq(domain)]
        for keys, group in subset.groupby(
            columns, sort=True, dropna=False, observed=True
        ):
            metric = str(group["Metric"].iloc[0])
            feature_output = str(group["FeatureOutput"].iloc[0])
            family_planned = (planned_overrides or {}).get(
                (domain, metric, feature_output), planned
            )
            if len(group) != family_planned:
                raise ValueError(
                    f"{result_table} BH family has {len(group)} rows instead of "
                    f"{family_planned}: {keys}"
                )
            finite = np.isfinite(pd.to_numeric(group["P"], errors="coerce"))
            tested = int(finite.sum())
            family_key = json.dumps(
                dict(zip(columns, keys, strict=True)),
                ensure_ascii=False,
                sort_keys=True,
            )
            output.loc[group.index, "Q"] = _benjamini_hochberg(group["P"])
            output.loc[group.index, "m_planned"] = family_planned
            output.loc[group.index, "m_tested"] = tested
            output.loc[group.index, "FamilyKey"] = family_key
            family_records.append(
                {
                    "ResultTable": result_table,
                    "FamilyKey": family_key,
                    "OutcomeDomain": domain,
                    "CorrectionMethod": "BH",
                    "m_planned": family_planned,
                    "m_tested": tested,
                }
            )
    alpha = 0.05
    output["P_lt_0_05"] = pd.array(
        [value < alpha if np.isfinite(value) else pd.NA for value in output["P"]],
        dtype="boolean",
    )
    output["Q_lt_0_05"] = pd.array(
        [value < alpha if np.isfinite(value) else pd.NA for value in output["Q"]],
        dtype="boolean",
    )
    return output, pd.DataFrame(family_records)


def _holm_adjust(values: Iterable[object], *, planned: int) -> np.ndarray:
    numeric = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(
        dtype=float
    )
    adjusted = np.full(len(numeric), np.nan, dtype=float)
    finite_positions = np.flatnonzero(np.isfinite(numeric))
    if len(finite_positions) == 0:
        return adjusted
    order = np.argsort(numeric[finite_positions], kind="mergesort")
    ordered_positions = finite_positions[order]
    ranked = numeric[ordered_positions]
    multipliers = planned - np.arange(len(ranked))
    adjusted_ranked = np.minimum(
        1.0,
        np.maximum.accumulate(ranked * multipliers),
    )
    adjusted[ordered_positions] = adjusted_ranked
    return adjusted


def _register_within_model_holm(
    table: pd.DataFrame,
    *,
    phases: Iterable[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = table.copy()
    phase_levels = list(phases)
    registered = (
        pd.to_numeric(output["P_holm"], errors="coerce")
        if "P_holm" in output.columns
        else None
    )
    output = output.drop(
        columns=[
            "Q",
            "Q_lt_0_05",
            "P_holm",
            "P_holm_lt_0_05",
            "P_lt_0_05",
            "m_planned",
            "m_tested",
            "FamilyKey",
            "CorrectionMethod",
        ],
        errors="ignore",
    )
    output.insert(output.columns.get_loc("P") + 1, "P_holm", np.nan)
    output["m_planned"] = len(phase_levels)
    output["m_tested"] = 0
    output["FamilyKey"] = ""
    output["CorrectionMethod"] = "holm"
    family_records: list[dict[str, Any]] = []
    for model_id, group in output.groupby("ModelID", sort=True, observed=True):
        if len(group) != len(phase_levels) or set(group["Phase"]) != set(phase_levels):
            raise ValueError(
                "Each Phase-slope Holm family must contain one row per Phase: "
                f"{model_id} has {len(group)} rows."
            )
        finite = np.isfinite(pd.to_numeric(group["P"], errors="coerce"))
        tested = int(finite.sum())
        adjusted = _holm_adjust(group["P"], planned=len(phase_levels))
        if registered is not None:
            registered_group = registered.loc[group.index].to_numpy(dtype=float)
            registered_finite = np.isfinite(registered_group)
            if not np.array_equal(registered_finite, np.isfinite(adjusted)) or (
                registered_finite.any()
                and not np.allclose(
                    registered_group[registered_finite],
                    adjusted[registered_finite],
                    rtol=0.0,
                    atol=1e-12,
                )
            ):
                raise ValueError(
                    "R-computed P_holm does not match the registered three-Phase "
                    f"Holm adjustment for {model_id}."
                )
        family_identity = {"ModelID": str(model_id)}
        if "OutcomeChangeBasis" in group.columns:
            family_identity["OutcomeChangeBasis"] = str(
                group["OutcomeChangeBasis"].iloc[0]
            )
        family_key = json.dumps(family_identity, ensure_ascii=False, sort_keys=True)
        output.loc[group.index, "P_holm"] = adjusted
        output.loc[group.index, "m_tested"] = tested
        output.loc[group.index, "FamilyKey"] = family_key
        family_record = {
            "ResultTable": "phase_slopes",
            "FamilyKey": family_key,
            "OutcomeDomain": str(group["OutcomeDomain"].iloc[0]),
            "CorrectionMethod": "holm",
            "m_planned": len(phase_levels),
            "m_tested": tested,
        }
        if "OutcomeChangeBasis" in group.columns:
            family_record["OutcomeChangeBasis"] = str(
                group["OutcomeChangeBasis"].iloc[0]
            )
        family_records.append(family_record)
    alpha = 0.05
    output["P_lt_0_05"] = pd.array(
        [value < alpha if np.isfinite(value) else pd.NA for value in output["P"]],
        dtype="boolean",
    )
    output["P_holm_lt_0_05"] = pd.array(
        [value < alpha if np.isfinite(value) else pd.NA for value in output["P_holm"]],
        dtype="boolean",
    )
    return output, pd.DataFrame(family_records)


def _standardize_slopes(
    slopes: pd.DataFrame,
    outcome_sd: pd.DataFrame,
    display_limit: float,
) -> pd.DataFrame:
    output = slopes.merge(
        outcome_sd,
        on=OUTCOME_SD_KEY,
        how="left",
        validate="many_to_one",
    )
    valid = np.isfinite(output["ValueSD"]) & output["ValueSD"].gt(0)
    for source, target in (
        ("Slope", "StandardizedSlope"),
        ("Lower", "StandardizedLower"),
        ("Upper", "StandardizedUpper"),
    ):
        output[target] = np.where(valid, output[source] / output["ValueSD"], np.nan)
    output["StandardizedSlopeDisplay"] = output["StandardizedSlope"].clip(
        -display_limit, display_limit
    )
    output["DisplayClipped"] = pd.array(
        np.where(
            np.isfinite(output["StandardizedSlope"]),
            output["StandardizedSlope"].abs().gt(display_limit),
            pd.NA,
        ),
        dtype="boolean",
    )
    output["Star"] = output["P_holm"].map(_significance_star)
    return output


def _significance_star(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    if number < 0.001:
        return "***"
    if number < 0.01:
        return "**"
    if number < 0.05:
        return "*"
    return ""


def _sanitize(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")


def _visual_style_paths(
    config: StructuralConnectivityConfig,
) -> dict[str, Path]:
    settings = config["inference"]["contact_lmm"]["visualization"]
    paths: dict[str, Path] = {}
    for output_key, setting_key in (
        ("phase_scalar_style", "phase_scalar_style_config"),
        ("phase_heatmap_style", "phase_heatmap_style_config"),
    ):
        path = Path(str(settings[setting_key])).expanduser()
        if not path.is_absolute():
            path = config.path.parent.parent / path
        paths[output_key] = path.resolve()
    return paths


def _visual_styles(
    config: StructuralConnectivityConfig,
) -> tuple[VizConfig, VizConfig]:
    paths = _visual_style_paths(config)
    return (
        load_viz_config(paths["phase_scalar_style"]),
        load_viz_config(paths["phase_heatmap_style"]),
    )


def _atomic_pdf(
    figure: Any,
    path: Path,
    *,
    overwrite: bool,
    dpi: int,
) -> None:
    prepare_output(path, overwrite=overwrite)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".pdf", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        figure.savefig(
            temporary,
            format="pdf",
            dpi=dpi,
            bbox_inches="tight",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _figure_specs(
    slopes: pd.DataFrame,
    config: StructuralConnectivityConfig,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    sources = sorted(slopes["ConnectomeSource"].dropna().astype(str).unique())
    polarities = sorted(slopes["Polar"].dropna().astype(str).unique())
    registered_metrics = set(config["inference"]["contact_lmm"]["predictor_metrics"])
    existing_metrics = sorted(
        metric
        for metric in config["inference"]["lift_metrics"]
        if metric in registered_metrics
    )
    appended_metrics = sorted(registered_metrics.difference(existing_metrics))
    metric_groups = [existing_metrics, *([metric] for metric in appended_metrics)]
    for metric_group in metric_groups:
        for source in sources:
            for lift_metric in metric_group:
                for polar in polarities:
                    group = slopes.loc[
                        slopes["ConnectomeSource"].eq(source)
                        & slopes["LiftMetric"].eq(lift_metric)
                        & slopes["Polar"].eq(polar)
                    ]
                    if group.empty:
                        continue
                    local_contexts = set(
                        group.loc[group["OutcomeDomain"].eq("local"), "Region"].astype(
                            str
                        )
                    )
                    network_contexts = set(
                        group.loc[
                            group["OutcomeDomain"].eq("connectivity"),
                            "LiftPredictor",
                        ].astype(str)
                    )
                    for context in HEATMAP_CONTEXTS:
                        if (
                            context not in local_contexts
                            and context not in network_contexts
                        ):
                            continue
                        for orientation in HEATMAP_ORIENTATIONS:
                            records.append(
                                {
                                    "OutcomeDomain": (
                                        "combined"
                                        if context in {"STN", "SNr"}
                                        else "connectivity"
                                    ),
                                    "ConnectomeSource": source,
                                    "LiftMetric": lift_metric,
                                    "LiftPredictor": context,
                                    "DisplayContext": context,
                                    "Polar": polar,
                                    "Region": (
                                        context if context in {"STN", "SNr"} else ""
                                    ),
                                    "Orientation": orientation,
                                    "Layout": SINGLE_ATLAS_LAYOUT,
                                    "ConnectomeSources": source,
                                }
                            )
    for index, record in enumerate(records, start=1):
        bits = [
            f"connectome-{record['ConnectomeSource']}",
            f"lift-{record['LiftMetric']}",
            f"context-{record['DisplayContext']}",
            f"polar-{record['Polar']}",
            f"orientation-{record['Orientation']}",
        ]
        filename = "_".join(_sanitize(value) for value in bits) + ".pdf"
        record["FigureID"] = f"LMMHEAT{index:03d}"
        record["FigureType"] = "heatmap"
        record["Filename"] = filename

    atlas_grid = config["inference"]["contact_lmm"]["visualization"].get(
        "atlas_grid_heatmap"
    )
    if atlas_grid is None:
        return records

    lift_metric = str(atlas_grid["lift_metric"])
    connectome_order = [str(value) for value in atlas_grid["connectome_order"]]
    atlas_records: list[dict[str, Any]] = []
    metric_rows = slopes.loc[
        slopes["LiftMetric"].eq(lift_metric)
        & slopes["ConnectomeSource"].isin(connectome_order)
    ]
    for polar in polarities:
        polar_rows = metric_rows.loc[metric_rows["Polar"].eq(polar)]
        if polar_rows.empty:
            continue
        local_contexts = set(
            polar_rows.loc[polar_rows["OutcomeDomain"].eq("local"), "Region"].astype(
                str
            )
        )
        network_contexts = set(
            polar_rows.loc[
                polar_rows["OutcomeDomain"].eq("connectivity"), "LiftPredictor"
            ].astype(str)
        )
        for context in HEATMAP_CONTEXTS:
            if context not in local_contexts and context not in network_contexts:
                continue
            atlas_records.append(
                {
                    "OutcomeDomain": (
                        "combined" if context in {"STN", "SNr"} else "connectivity"
                    ),
                    "ConnectomeSource": "",
                    "ConnectomeSources": ";".join(connectome_order),
                    "LiftMetric": lift_metric,
                    "LiftPredictor": context,
                    "DisplayContext": context,
                    "Polar": polar,
                    "Region": context if context in {"STN", "SNr"} else "",
                    "Orientation": str(atlas_grid["orientation"]),
                    "Layout": ATLAS_GRID_LAYOUT,
                }
            )

    for index, record in enumerate(atlas_records, start=1):
        bits = [
            "atlas-grid",
            f"lift-{record['LiftMetric']}",
            f"context-{record['DisplayContext']}",
            f"polar-{record['Polar']}",
            f"orientation-{record['Orientation']}",
        ]
        record["FigureID"] = f"LMMATLAS{index:03d}"
        record["FigureType"] = "heatmap"
        record["Filename"] = "_".join(_sanitize(value) for value in bits) + ".pdf"

    return [
        record for record in records if record["LiftMetric"] != lift_metric
    ] + atlas_records


def _heatmap_output_count(specs: Iterable[Mapping[str, Any]]) -> int:
    return sum(1 for _ in specs)


def _combined_heatmap_filename(spec: Mapping[str, Any]) -> str:
    filename = str(spec["Filename"])
    orientation = _sanitize(str(spec["Orientation"]))
    marker = f"orientation-{orientation}.pdf"
    if not filename.endswith(marker):
        raise ValueError(
            f"Atlas-grid heatmap filename lacks its orientation: {filename}"
        )
    return (
        filename.removesuffix(marker) + "orientation-metric-by-band-plus-aperiodic.pdf"
    )


def _display_metric_key(metric: object, feature_output: object) -> str:
    return f"{metric}/{feature_output}"


def _banded_metric_contract(
    style_config: VizConfig,
    *,
    context: str,
    expanded_local: bool,
) -> tuple[list[str], dict[str, str]]:
    groups = style_config["matrices"]["banded"]["groups"]
    columns: list[Mapping[str, Any]] = []
    if context in {"STN", "SNr"}:
        local_columns = list(groups["local"]["columns"])
        if not expanded_local:
            local_columns = [
                column
                for column in local_columns
                if str(column["metric"]) == "periodic"
                and str(column["feature_output"]) == "mean-scalar"
            ]
        columns.extend(local_columns)
    columns.extend(groups["connectivity"]["columns"])
    order = [
        _display_metric_key(column["metric"], column["feature_output"])
        for column in columns
    ]
    labels = {
        _display_metric_key(column["metric"], column["feature_output"]): str(
            column["label"]
        )
        for column in columns
    }
    return order, labels


def _aperiodic_parameter_contract(
    style_config: VizConfig,
) -> tuple[list[str], dict[str, str]]:
    rows = style_config["matrices"]["aperiodic"]["rows"]
    return (
        [str(row["value"]) for row in rows],
        {str(row["value"]): str(row["label"]) for row in rows},
    )


def _metric_labels(style_config: VizConfig) -> dict[str, str]:
    labels: dict[str, str] = {}
    for group in style_config["matrices"]["banded"]["groups"].values():
        for column in group["columns"]:
            labels[str(column["metric"])] = str(column["label"])
    return labels


def _heatmap_metric_order(
    config: StructuralConnectivityConfig,
    style_config: VizConfig,
) -> list[str]:
    registered = set(config["outcomes"]["connectivity_metrics"])
    connectivity = [
        str(column["metric"])
        for column in style_config["matrices"]["banded"]["groups"]["connectivity"][
            "columns"
        ]
        if str(column["metric"]) in registered
    ]
    return ["periodic", *connectivity]


def _heatmap_top_label(
    spec: Mapping[str, Any],
    phase: str,
    style_config: VizConfig,
    config: StructuralConnectivityConfig,
) -> str:
    context = str(spec["DisplayContext"])
    context_label = config["inference"]["contact_lmm"]["visualization"][
        "lift_predictor_labels"
    ][context]
    if context == "RegionMean":
        context_label = "Mean"
    polar = style_config["labels"]["polarity"][str(spec["Polar"])]
    phase_label = (
        config["inference"]["contact_lmm"]["visualization"]
        .get("phase_display_labels", {})
        .get(phase, phase)
    )
    return f"{context_label} | {polar} | {phase_label}"


def _heatmap_context_rows(
    slopes: pd.DataFrame,
    spec: Mapping[str, Any],
    *,
    source: str,
) -> pd.DataFrame:
    base_selector = (
        slopes["ConnectomeSource"].eq(source)
        & slopes["LiftMetric"].eq(spec["LiftMetric"])
        & slopes["Polar"].eq(spec["Polar"])
    )
    context = str(spec["DisplayContext"])
    network_selector = (
        base_selector
        & slopes["OutcomeDomain"].eq("connectivity")
        & slopes["LiftPredictor"].eq(context)
    )
    cells = [slopes.loc[network_selector].copy()]
    if context in {"STN", "SNr"}:
        local_selector = (
            base_selector
            & slopes["OutcomeDomain"].eq("local")
            & slopes["Region"].eq(context)
            & slopes["LiftPredictor"].eq(context)
        )
        cells.append(slopes.loc[local_selector].copy())
    output = pd.concat(cells, ignore_index=True, sort=False)
    if "FeatureOutput" not in output:
        output["FeatureOutput"] = "mean-scalar"
    output["DisplayMetric"] = [
        _display_metric_key(metric, feature_output)
        for metric, feature_output in output[["Metric", "FeatureOutput"]].itertuples(
            index=False, name=None
        )
    ]
    return output


def _heatmap_panel(
    cell: pd.DataFrame,
    *,
    phase: str,
    bands: list[str],
    metric_order: list[str],
    band_labels: Mapping[str, str],
    metric_labels: Mapping[str, str],
    orientation: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    phase_rows = cell.loc[cell["Phase"].eq(phase) & cell["Band"].isin(bands)]
    value = phase_rows.pivot(
        index="Band", columns="DisplayMetric", values="StandardizedSlopeDisplay"
    ).reindex(index=bands, columns=metric_order)
    adjusted_p = phase_rows.pivot(
        index="Band", columns="DisplayMetric", values="P_holm"
    ).reindex(index=bands, columns=metric_order)
    value.index = [band_labels[str(band)] for band in value.index]
    value.columns = [metric_labels[str(metric)] for metric in value.columns]
    adjusted_p.index = value.index
    adjusted_p.columns = value.columns
    if orientation == "metric-by-band":
        value = value.T
        adjusted_p = adjusted_p.T
    return value, adjusted_p


def _aperiodic_heatmap_panel(
    cell: pd.DataFrame,
    *,
    phase: str,
    parameters: list[str],
    parameter_labels: Mapping[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = cell.loc[
        cell["Phase"].eq(phase)
        & cell["Metric"].eq("aperiodic")
        & cell["FeatureOutput"].eq("mean-scalar")
        & cell["Band"].isin(parameters)
    ].copy()
    if rows.duplicated("Band").any():
        raise ValueError("Aperiodic heatmap cells must be unique within a panel.")
    values = rows.set_index("Band")["StandardizedSlopeDisplay"].reindex(parameters)
    adjusted = rows.set_index("Band")["P_holm"].reindex(parameters)
    index = [parameter_labels[parameter] for parameter in parameters]
    return (
        pd.DataFrame({"Aperiodic": values.to_numpy()}, index=index),
        pd.DataFrame({"Aperiodic": adjusted.to_numpy()}, index=index),
    )


def _pad_aperiodic_heatmap_panel(
    values: pd.DataFrame,
    adjusted_p: pd.DataFrame,
    *,
    band_columns: Iterable[object],
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Center one aperiodic column in a seven-column atlas panel."""

    columns = [str(value) for value in band_columns]
    if values.shape[1] != 1 or adjusted_p.shape[1] != 1:
        raise ValueError("Aperiodic atlas panels must contain exactly one column.")
    if not columns:
        raise ValueError("The combined atlas panel requires band columns.")
    center = len(columns) // 2
    padded_values = pd.DataFrame(np.nan, index=values.index, columns=columns)
    padded_adjusted = pd.DataFrame(np.nan, index=values.index, columns=columns)
    padded_values.iloc[:, center] = values.iloc[:, 0].to_numpy()
    padded_adjusted.iloc[:, center] = adjusted_p.iloc[:, 0].to_numpy()
    return padded_values, padded_adjusted, center


def _format_combined_atlas_grid_heatmap(
    figure: Any,
    *,
    phase_labels: list[str],
    atlas_count: int,
    aperiodic_column: int,
    cell_width_mm: float,
    label_right_bg_color: str,
    label_text_color: str,
    label_fontsize: float,
    label_fontweight: str,
) -> None:
    """Format alternating banded/aperiodic rows as three Phase groups."""

    grid_rows = len(phase_labels) * 2
    panel_count = grid_rows * atlas_count
    panel_axes = np.asarray(figure.axes[:panel_count], dtype=object).reshape(
        grid_rows, atlas_count
    )
    figure_width_in, figure_height_in = figure.get_size_inches()
    mm_to_x = 1.0 / (float(figure_width_in) * 25.4)
    mm_to_y = 1.0 / (float(figure_height_in) * 25.4)

    # The five gaps sum to the 20 mm reserved by the renderer. The final
    # 8 mm gap leaves room for the bottom band labels before the last
    # aperiodic block without changing the page height.
    row_gaps_mm = [2.0, 4.0, 2.0, 4.0, 8.0]
    if len(row_gaps_mm) != grid_rows - 1:
        raise ValueError("The combined atlas row-gap contract is inconsistent.")
    row_heights = [panel_axes[row, 0].get_position().height for row in range(grid_rows)]
    row_y = [0.0] * grid_rows
    cursor = panel_axes[-1, 0].get_position().y0
    for row in range(grid_rows - 1, -1, -1):
        row_y[row] = cursor
        cursor += row_heights[row]
        if row > 0:
            cursor += row_gaps_mm[row - 1] * mm_to_y

    for row in range(grid_rows):
        is_aperiodic = row % 2 == 1
        for column in range(atlas_count):
            axis = panel_axes[row, column]
            position = axis.get_position()
            x0 = position.x0
            width = position.width
            if is_aperiodic:
                width = cell_width_mm * mm_to_x
                x0 = position.x0 + (position.width - width) / 2.0
                axis.set_xlim(aperiodic_column, aperiodic_column + 1)
                axis.set_xticks([])
                for text in list(axis.texts):
                    if text.get_text() == "NA" and not (
                        aperiodic_column
                        <= text.get_position()[0]
                        < aperiodic_column + 1
                    ):
                        text.remove()
            elif row != grid_rows - 2:
                axis.tick_params(axis="x", labelbottom=False)
            axis.set_position([x0, row_y[row], width, row_heights[row]])

    strip_axes = [
        axis
        for axis in list(figure.axes)
        if getattr(axis, "_visualdf_strip_axis", False)
    ]
    right_strips = strip_axes[atlas_count:]
    if len(right_strips) != grid_rows:
        raise ValueError(
            "The combined atlas heatmap did not create the expected right strips."
        )
    strip_position = right_strips[0].get_position()
    strip_x0 = strip_position.x0
    strip_width = strip_position.width
    for axis in right_strips:
        axis.remove()

    for phase_index, phase_label in enumerate(phase_labels):
        banded_axis = panel_axes[phase_index * 2, -1]
        aperiodic_axis = panel_axes[phase_index * 2 + 1, -1]
        y0 = aperiodic_axis.get_position().y0
        y1 = banded_axis.get_position().y1
        strip = figure.add_axes([strip_x0, y0, strip_width, y1 - y0])
        strip._visualdf_strip_axis = True
        strip.set_facecolor(label_right_bg_color)
        strip.text(
            0.5,
            0.5,
            phase_label,
            ha="center",
            va="center",
            rotation=-90,
            fontsize=label_fontsize,
            color=label_text_color,
            fontweight=label_fontweight,
        )
        strip.set_xticks([])
        strip.set_yticks([])
        for spine in strip.spines.values():
            spine.set_visible(False)
    visualdf._ensure_strip_background_opaque(figure)


def _pdf_page_dimensions_mm(path: Path) -> tuple[float, float]:
    reader = PdfReader(str(path))
    if len(reader.pages) != 1:
        raise ValueError(f"Heatmap PDF must contain exactly one page: {path}")
    box = reader.pages[0].mediabox
    points_to_mm = 25.4 / 72.0
    return float(box.width) * points_to_mm, float(box.height) * points_to_mm


def _plot_heatmaps(
    slopes: pd.DataFrame,
    config: StructuralConnectivityConfig,
    *,
    overwrite: bool,
    lift_metrics: set[str] | None = None,
) -> pd.DataFrame:
    phases = list(config["inference"]["contact_lmm"]["phase_levels"])
    bands = list(config["outcomes"]["bands"])
    settings = config["inference"]["contact_lmm"]
    _, style_config = _visual_styles(config)
    style = style_config["style"]
    geometry = style["geometry_mm"]
    strips = style["strips"]
    heatmap = style["heatmap"]
    band_labels = {
        str(row["value"]): str(row["label"])
        for row in style_config["matrices"]["banded"]["rows"]
    }
    aperiodic_parameters, aperiodic_labels = _aperiodic_parameter_contract(style_config)
    limit = float(settings["display_limit"])
    cmap = getattr(visualdf.cm, heatmap["colormap"]).copy()
    cmap.set_bad("#d9d9d9")
    figure_root = config.contact_lmm_root / "figures" / "heatmap"
    records: list[dict[str, Any]] = []
    visualdf.set_greek_symbol_font_enabled(False)
    specs = _figure_specs(slopes, config)
    if lift_metrics is not None:
        specs = [spec for spec in specs if spec["LiftMetric"] in lift_metrics]
    for spec in specs:
        layout = str(spec["Layout"])
        context = str(spec["DisplayContext"])
        metric_order, metric_labels = _banded_metric_contract(
            style_config,
            context=context,
            expanded_local=layout == ATLAS_GRID_LAYOUT,
        )
        has_aperiodic_block = layout == ATLAS_GRID_LAYOUT and context in {
            "STN",
            "SNr",
        }
        colorbar_label = "Standardized slope"
        if layout == ATLAS_GRID_LAYOUT:
            atlas_grid = settings["visualization"]["atlas_grid_heatmap"]
            sources = [str(value) for value in atlas_grid["connectome_order"]]
            value_grid = []
            adjusted_p_grid = []
            aperiodic_value_grid: list[list[pd.DataFrame]] = []
            aperiodic_adjusted_p_grid: list[list[pd.DataFrame]] = []
            for phase in phases:
                value_row: list[pd.DataFrame] = []
                adjusted_p_row: list[pd.DataFrame] = []
                aperiodic_value_row: list[pd.DataFrame] = []
                aperiodic_adjusted_p_row: list[pd.DataFrame] = []
                for source in sources:
                    cell = _heatmap_context_rows(slopes, spec, source=source)
                    value, adjusted_p = _heatmap_panel(
                        cell,
                        phase=phase,
                        bands=bands,
                        metric_order=metric_order,
                        band_labels=band_labels,
                        metric_labels=metric_labels,
                        orientation=str(spec["Orientation"]),
                    )
                    value_row.append(value)
                    adjusted_p_row.append(adjusted_p)
                    if has_aperiodic_block:
                        aperiodic_value, aperiodic_adjusted_p = (
                            _aperiodic_heatmap_panel(
                                cell,
                                phase=phase,
                                parameters=aperiodic_parameters,
                                parameter_labels=aperiodic_labels,
                            )
                        )
                        aperiodic_value_row.append(aperiodic_value)
                        aperiodic_adjusted_p_row.append(aperiodic_adjusted_p)
                value_grid.append(value_row)
                adjusted_p_grid.append(adjusted_p_row)
                if has_aperiodic_block:
                    aperiodic_value_grid.append(aperiodic_value_row)
                    aperiodic_adjusted_p_grid.append(aperiodic_adjusted_p_row)
            connectome_label = ""
            label_top = [
                settings["visualization"]["connectome_labels"][source]
                for source in sources
            ]
            label_right = [
                settings["visualization"]["phase_display_labels"].get(phase, phase)
                for phase in phases
            ]
            predictor_label = {
                "STN": "STN",
                "SNr": "SNr",
                "RegionMean": "Mean STN/SNr",
            }[context]
            polarity_label = {
                "Anodal": r"$\oplus$",
                "Cathodal": r"$\ominus$",
            }[str(spec["Polar"])]
            colorbar_label = (
                "Standardized slope for streamline field-exposure excess, "
                rf"$\beta_{{\mathrm{{std}}}}$ | {predictor_label} | "
                f"{polarity_label}"
            )
        else:
            source = str(spec["ConnectomeSource"])
            cell = _heatmap_context_rows(slopes, spec, source=source)
            value_row = []
            adjusted_p_row = []
            for phase in phases:
                value, adjusted_p = _heatmap_panel(
                    cell,
                    phase=phase,
                    bands=bands,
                    metric_order=metric_order,
                    band_labels=band_labels,
                    metric_labels=metric_labels,
                    orientation=str(spec["Orientation"]),
                )
                value_row.append(value)
                adjusted_p_row.append(adjusted_p)
            value_grid = [value_row]
            adjusted_p_grid = [adjusted_p_row]
            connectome_label = settings["visualization"]["connectome_labels"][source]
            label_top = [
                _heatmap_top_label(spec, phase, style_config, config)
                for phase in phases
            ]
            label_right = None
        plot_values = value_grid
        plot_adjusted_p = adjusted_p_grid
        plot_label_right = label_right
        plot_panel_gap = tuple(geometry["panel_gap"]["lat_rows_phase_columns"])
        plot_show_x_ticklabels = "bottom"
        aperiodic_column = -1
        if has_aperiodic_block:
            plot_values = []
            plot_adjusted_p = []
            for phase_index, (banded_row, banded_p_row) in enumerate(
                zip(value_grid, adjusted_p_grid, strict=True)
            ):
                plot_values.append(banded_row)
                plot_adjusted_p.append(banded_p_row)
                padded_value_row: list[pd.DataFrame] = []
                padded_p_row: list[pd.DataFrame] = []
                for atlas_index, (aperiodic_value, aperiodic_p) in enumerate(
                    zip(
                        aperiodic_value_grid[phase_index],
                        aperiodic_adjusted_p_grid[phase_index],
                        strict=True,
                    )
                ):
                    padded_value, padded_p, center = _pad_aperiodic_heatmap_panel(
                        aperiodic_value,
                        aperiodic_p,
                        band_columns=banded_row[atlas_index].columns,
                    )
                    if aperiodic_column not in {-1, center}:
                        raise ValueError(
                            "Combined aperiodic panels have inconsistent centers."
                        )
                    aperiodic_column = center
                    padded_value_row.append(padded_value)
                    padded_p_row.append(padded_p)
                plot_values.append(padded_value_row)
                plot_adjusted_p.append(padded_p_row)
            plot_label_right = [""] * len(plot_values)
            plot_panel_gap = (
                float(geometry["panel_gap"]["lat_rows_phase_columns"][0]),
                4.0,
            )
            plot_show_x_ticklabels = "all"

        cell_size = tuple(float(value) for value in geometry["cell_size"]["banded"])
        figure = visualdf.plot_well_heatmap_grid_df(
            df_values=plot_values,
            df_ps=plot_adjusted_p,
            x_label=None,
            y_label=connectome_label,
            xtick_rotation=90,
            ytick_rotation=0,
            title=None,
            label_top=label_top,
            label_right=plot_label_right,
            font_family=style["font"]["family"],
            cmap=cmap,
            na_color=heatmap.get("na_color"),
            vmin=-limit,
            vmax=limit,
            vmode="sym",
            dpi=style["dpi"],
            cellsize=cell_size,
            panel_gap_mm=plot_panel_gap,
            label_top_height_mm=geometry["strip_top_height"],
            label_right_width_mm=geometry["strip_right_width"],
            strip_pad_mm=geometry["strip_pad"],
            label_top_bg_color=strips["top_background"],
            label_right_bg_color=strips["right_background"],
            label_text_color=strips["text_color"],
            label_fontsize=style["font"]["strip_pt"],
            label_fontweight=strips["font_weight"],
            include_global_label_margins=False,
            cbar_label_offset_mm=geometry["colorbar_label_offset"],
            grid_lines=None,
            gline_color=style["grid"]["color"],
            gline_width=style["grid"]["width_pt"],
            gline_alpha=style["grid"]["alpha"],
            axis_label_fontsize=style["font"]["strip_pt"],
            tick_label_fontsize=style["font"]["tick_pt"],
            colorbar_label=colorbar_label,
            colorbar_ticks=[-limit, 0.0, limit],
            colorbar_width_mm=geometry["colorbar_width"],
            colorbar_pad_mm=geometry["colorbar_pad"],
            show_p=True,
            hide_ns=style["significance"]["hide_non_significant"],
            p_text_color=style["significance"]["text_color"],
            p_text_size=style["font"]["star_pt"],
            show_x_ticklabels=plot_show_x_ticklabels,
            show_y_ticklabels="left",
            transparent=style["transparent"],
        )
        if has_aperiodic_block:
            _format_combined_atlas_grid_heatmap(
                figure,
                phase_labels=label_right,
                atlas_count=len(sources),
                aperiodic_column=aperiodic_column,
                cell_width_mm=cell_size[0],
                label_right_bg_color=strips["right_background"],
                label_text_color=strips["text_color"],
                label_fontsize=style["font"]["strip_pt"],
                label_fontweight=strips["font_weight"],
            )
        output_filename = (
            _combined_heatmap_filename(spec)
            if has_aperiodic_block
            else str(spec["Filename"])
        )
        source_path = figure_root / output_filename
        try:
            _atomic_pdf(
                figure,
                source_path,
                overwrite=overwrite,
                dpi=style["dpi"],
            )
        finally:
            visualdf.plt.close(figure)
        width_mm, height_mm = _pdf_page_dimensions_mm(source_path)
        if layout == ATLAS_GRID_LAYOUT:
            max_width = float(atlas_grid["max_width_mm"])
            max_height = float(atlas_grid["max_height_mm"])
            if width_mm > max_width or height_mm > max_height:
                source_path.unlink(missing_ok=True)
                raise ValueError(
                    "Cross-atlas FieldExcess heatmap exceeds its registered "
                    f"PDF size: {width_mm:.2f} x {height_mm:.2f} mm; maximum "
                    f"{max_width:.2f} x {max_height:.2f} mm."
                )

        displayed_metrics = list(metric_order)
        metric_blocks = "banded"
        aperiodic_parameter_text = ""
        orientation = str(spec["Orientation"])
        if has_aperiodic_block:
            displayed_metrics.append("aperiodic/mean-scalar")
            metric_blocks = "banded;aperiodic"
            aperiodic_parameter_text = ";".join(aperiodic_parameters)
            orientation = "metric-by-band-plus-aperiodic"
        base_record = {
            **spec,
            "Filename": output_filename,
            "SourcePath": str(source_path),
            "PublishedRelativePath": f"heatmap/{output_filename}",
            "Orientation": orientation,
            "SourceSlopeTable": str(config.contact_lmm_root / "phase_slopes.csv"),
            "PhasePanels": ";".join(phases),
            "Bands": ";".join(bands),
            "Metrics": ";".join(displayed_metrics),
            "MetricBlocks": metric_blocks,
            "AperiodicParameters": aperiodic_parameter_text,
            "ValueColumn": "StandardizedSlopeDisplay",
            "SignificanceColumn": "P_holm",
            "YAxisLabel": connectome_label,
            "HeaderLabel": "",
            "ColorbarLabel": colorbar_label,
            "AtlasColumns": (
                spec["ConnectomeSources"] if layout == ATLAS_GRID_LAYOUT else ""
            ),
            "PhaseRows": ";".join(phases) if layout == ATLAS_GRID_LAYOUT else "",
            "Renderer": "lfp_viz.visualdf.plot_well_heatmap_grid_df",
            "StyleConfig": str(style_config.path),
            "ColorMap": heatmap["colormap"],
            "DisplayMin": -limit,
            "DisplayMax": limit,
            "WidthMm": width_mm,
            "HeightMm": height_mm,
            "Status": "complete",
        }
        records.append(base_record)
    return pd.DataFrame(records)


def _slope_figure_specs(slopes: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    metadata = slopes.drop_duplicates(["ModelID", "Phase"], keep="first").sort_values(
        "ModelID", kind="mergesort"
    )
    for row in metadata.itertuples(index=False):
        region = "" if pd.isna(row.Region) else str(row.Region)
        pair_region = "" if pd.isna(row.PairRegion) else str(row.PairRegion)
        bits = [
            f"model-{row.ModelID}",
            str(row.OutcomeDomain),
            f"connectome-{row.ConnectomeSource}",
            f"lift-{row.LiftMetric}",
            f"predictor-{row.LiftPredictor}",
            f"polar-{row.Polar}",
            f"metric-{row.Metric}",
            f"band-{row.Band}",
            f"phase-{row.Phase}",
        ]
        if str(row.Metric) == "burst":
            bits.append(f"feature-output-{row.FeatureOutput}")
        if region:
            bits.append(f"region-{region}")
        if pair_region:
            bits.append(f"pair-{pair_region}")
        records.append(
            {
                "FigureID": (
                    str(row.ModelID).replace("LMM", "LMMSLOPE", 1)
                    + _sanitize(str(row.Phase)).upper()
                ),
                "FigureType": "slope",
                "ModelID": row.ModelID,
                "Phase": row.Phase,
                "OutcomeDomain": row.OutcomeDomain,
                "ConnectomeSource": row.ConnectomeSource,
                "LiftMetric": row.LiftMetric,
                "LiftPredictor": row.LiftPredictor,
                "Polar": row.Polar,
                "Metric": row.Metric,
                "FeatureOutput": row.FeatureOutput,
                "Region": region,
                "PairRegion": pair_region,
                "Band": row.Band,
                "Filename": "_".join(_sanitize(value) for value in bits) + ".pdf",
            }
        )
    return records


def _slope_region_label(spec: Mapping[str, Any], style_config: VizConfig) -> str:
    key = (
        str(spec["Region"])
        if spec["OutcomeDomain"] == "local"
        else str(spec["PairRegion"])
    )
    return str(style_config["labels"]["region"][key])


def _plot_slope_figures(
    model_rows: pd.DataFrame,
    curves: pd.DataFrame,
    slopes: pd.DataFrame,
    config: StructuralConnectivityConfig,
    *,
    overwrite: bool,
) -> pd.DataFrame:
    scalar_style, _ = _visual_styles(config)
    style = scalar_style["style"]
    font = style["font"]
    widths = style["line_width_pt"]
    geometry = style["geometry_mm"]
    strips = style["strips"]
    visualization = config["inference"]["contact_lmm"]["visualization"]
    registered_phases = list(config["inference"]["contact_lmm"]["phase_levels"])
    figure_root = config.contact_lmm_root / "figures" / "slope"
    records: list[dict[str, Any]] = []
    visualdf.set_greek_symbol_font_enabled(False)

    for spec in _slope_figure_specs(slopes):
        model_id = str(spec["ModelID"])
        phase = str(spec["Phase"])
        if phase not in registered_phases:
            raise ValueError(f"Slope figure has an unregistered Phase: {phase}")
        raw = model_rows.loc[
            model_rows["ModelID"].eq(model_id)
            & model_rows["RowEligibilityStatus"].eq("eligible")
            & model_rows["Phase"].eq(phase)
        ].copy()
        curve = curves.loc[
            curves["ModelID"].eq(model_id) & curves["Phase"].eq(phase)
        ].copy()
        slope = slopes.loc[
            slopes["ModelID"].eq(model_id) & slopes["Phase"].eq(phase)
        ].copy()
        if raw.empty or curve.empty or len(slope) != 1:
            raise ValueError(
                f"Slope figure inputs are incomplete for {model_id}: "
                f"Phase={phase}, raw={len(raw)}, curve={len(curve)}, "
                f"slope={len(slope)}."
            )

        region_label = _slope_region_label(spec, scalar_style)
        polar_label = scalar_style["labels"]["polarity"][str(spec["Polar"])]
        phase_labels = visualization.get("phase_display_labels", {})
        panel_levels = [
            f"{region_label} | {polar_label} | {phase_labels.get(phase, phase)}"
        ]
        panel_map = {phase: panel_levels[0]}
        connectome_label = visualization["connectome_labels"][
            str(spec["ConnectomeSource"])
        ]
        for table in (raw, curve, slope):
            table["PhasePanel"] = table["Phase"].map(panel_map)
            table["ConnectomeLabel"] = connectome_label
        curve = curve.rename(
            columns={
                "PredictedValue": "emmean",
                "Lower": "lower.CL",
                "Upper": "upper.CL",
            }
        )
        predictor_label = visualization["lift_predictor_labels"][
            str(spec["LiftPredictor"])
        ]
        lift_label = visualization["lift_metric_labels"][str(spec["LiftMetric"])]
        x_label = f"{predictor_label} {lift_label}, z-score"
        adjacent_phase = (
            config["inference"]["contact_lmm"].get("outcome_change_basis")
            == "adjacent_phase"
        )
        if adjacent_phase and str(spec["LiftMetric"]) == FIELD_EXCESS_METRIC:
            x_label = {
                "STN": "STN streamline field-exposure excess",
                "SNr": "SNr streamline field-exposure excess",
                "RegionMean": "Mean STN/SNr streamline field-exposure excess",
            }[str(spec["LiftPredictor"])]
        outcome_label = y_axis_label(spec, scalar_style)
        y_label = (
            rf"$\Delta$ {outcome_label}"
            if adjacent_phase
            else f"{outcome_label}\n{connectome_label}"
        )
        standardized_slope = float(slope["StandardizedSlope"].iloc[0])
        slope_text_loc = "upper left" if standardized_slope > 0.0 else "lower left"

        figure = visualdf.plot_triple_interaction_fit(
            df=raw,
            curve=curve,
            slope=slope,
            value_col="Value",
            x_var="X_z",
            panel_var="PhasePanel",
            facet_var="ConnectomeLabel",
            color_var="Phase",
            panel_levels=panel_levels,
            facet_levels=[connectome_label],
            x_label=x_label,
            y_label=y_label,
            title=None,
            font_family=font["latin"],
            palette=style["phase_palette"],
            jitter_alpha=style["jitter"]["alpha"],
            jitter_size=style["jitter"]["size"],
            curve_line_width=widths["emm_line"],
            curve_line_color=style["emm"]["color"],
            ribbon_alpha=style["box"]["fill_alpha"],
            show_ribbon=True,
            grid=style["axes"]["grid"],
            dpi=style["dpi"],
            seed=style["jitter"]["seed"],
            show_top_right_axes=style["axes"]["show_top_right_spines"],
            label_fontsize=font["strip_pt"],
            label_top_bg_color=strips["top_background"],
            label_right_bg_color=strips["right_background"],
            label_text_color=strips["text_color"],
            label_fontweight=strips["font_weight"],
            strip_top_height_mm=geometry["strip_top_height"],
            strip_right_width_mm=geometry["strip_right_width"],
            strip_pad_mm=geometry["strip_pad"],
            title_fontsize=font["axis_pt"],
            axis_label_fontsize=font["axis_pt"],
            tick_label_fontsize=font["tick_pt"],
            legend_loc="none",
            boxsize=(
                float(visualization["slope_panel_width_mm"]),
                float(geometry["panel_height"]),
            ),
            panel_gap=tuple(geometry["panel_gap"]),
            x_label_offset_mm=geometry["x_label_offset"],
            y_label_offset_mm=geometry["y_label_offset"],
            slope_beta_col="StandardizedSlope",
            slope_lower_col="StandardizedLower",
            slope_upper_col="StandardizedUpper",
            slope_p_col="P_holm",
            slope_text_fmt=(
                r"$\beta_{{\mathrm{{std}}}} = {beta:.2f}$"
                "\n"
                r"$p_{{\mathrm{{Holm}}}} = {p:.3g}$ ({stars})"
            ),
            slope_text_loc=slope_text_loc,
            slope_text_box_alpha=0.0,
            transparent=style["transparent"],
        )
        source_path = figure_root / str(spec["Filename"])
        try:
            _atomic_pdf(
                figure,
                source_path,
                overwrite=overwrite,
                dpi=style["dpi"],
            )
        finally:
            visualdf.plt.close(figure)
        records.append(
            {
                **spec,
                "SourcePath": str(source_path),
                "PublishedRelativePath": f"slope/{spec['Filename']}",
                "SourcePointTable": str(config.contact_lmm_root / "model_rows.csv"),
                "SourceCurveTable": str(
                    config.contact_lmm_root / "prediction_grid.csv"
                ),
                "SourceSlopeTable": str(config.contact_lmm_root / "phase_slopes.csv"),
                "PhasePanels": phase,
                "ValueColumn": "Value",
                "PredictorColumn": "X_z",
                "SlopeColumn": "StandardizedSlope",
                "SignificanceColumn": "P_holm",
                "YAxisLabel": y_label,
                "Renderer": "lfp_viz.visualdf.plot_triple_interaction_fit",
                "StyleConfig": str(scalar_style.path),
                "Status": "complete",
            }
        )
    return pd.DataFrame(records)


def _postprocess(
    r_outputs: Mapping[str, pd.DataFrame],
    models: pd.DataFrame,
    outcome_sd: pd.DataFrame,
    config: StructuralConnectivityConfig,
) -> dict[str, pd.DataFrame]:
    model_fit_observed = r_outputs["model_fit"]
    model_fit = _metadata_merge(model_fit_observed, models)
    noneligible = ~model_fit["ModelEligibilityStatus"].eq("eligible")
    model_fit.loc[noneligible, "FitStatus"] = model_fit.loc[
        noneligible, "ModelEligibilityStatus"
    ]
    model_fit["FitStatus"] = model_fit["FitStatus"].fillna("fit_error")

    fitted_status = model_fit[["ModelID", "FitStatus", "InferenceStatus"]].copy()
    fixed = _metadata_merge(r_outputs["fixed_effects"], models).merge(
        fitted_status, on="ModelID", how="left", validate="many_to_one"
    )

    phases = config["inference"]["contact_lmm"]["phase_levels"]
    slopes = _planned_inference_table(
        models,
        r_outputs["phase_slopes"],
        dimension="Phase",
        levels=phases,
    ).merge(fitted_status, on="ModelID", how="left", validate="many_to_one")
    slopes["Status"] = _result_status(slopes)
    slopes, slope_families = _register_within_model_holm(slopes, phases=phases)
    slopes = _standardize_slopes(
        slopes,
        outcome_sd,
        float(config["inference"]["contact_lmm"]["display_limit"]),
    )

    omnibus = _planned_inference_table(
        models,
        r_outputs["omnibus_tests"],
        dimension="Term",
        levels=OMNIBUS_TERMS,
    ).merge(fitted_status, on="ModelID", how="left", validate="many_to_one")
    omnibus["Status"] = _result_status(omnibus)
    planned_overrides = {
        ("local", str(row["metric"]), str(row["feature_output"])): len(row["bands"])
        for row in config["inference"]["contact_lmm"].get(
            "field_excess_local_outcomes", []
        )
    }
    omnibus, omnibus_families = _apply_bh(
        omnibus,
        result_table="omnibus_tests",
        planned=7,
        planned_overrides=planned_overrides,
    )

    predictions = _metadata_merge(r_outputs["prediction_grid"], models).merge(
        fitted_status, on="ModelID", how="left", validate="many_to_one"
    )
    families = pd.concat(
        [slope_families, omnibus_families], ignore_index=True
    ).sort_values(["ResultTable", "OutcomeDomain", "FamilyKey"], kind="mergesort")
    models = models.merge(
        fitted_status, on="ModelID", how="left", validate="one_to_one"
    )
    models.loc[~models["ModelEligibilityStatus"].eq("eligible"), "FitStatus"] = (
        models.loc[
            ~models["ModelEligibilityStatus"].eq("eligible"),
            "ModelEligibilityStatus",
        ]
    )
    return {
        "models": models,
        "model_fit": model_fit,
        "fixed_effects": fixed,
        "omnibus_tests": omnibus,
        "phase_slopes": slopes,
        "prediction_grid": predictions,
        "multiplicity_families": families.reset_index(drop=True),
    }


def _coverage(models: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "OutcomeDomain",
        "ConnectomeSource",
        "LiftMetric",
        "LiftPredictor",
        "Polar",
        "Metric",
        "FeatureOutput",
        "Region",
        "PairRegion",
    ]
    records: list[dict[str, Any]] = []
    for keys, group in models.groupby(
        group_columns, sort=True, dropna=False, observed=True
    ):
        records.append(
            {
                **dict(zip(group_columns, keys, strict=True)),
                "PlannedModels": int(len(group)),
                "EligibleModels": int(
                    group["ModelEligibilityStatus"].eq("eligible").sum()
                ),
                "EligibilityStatusCounts": json.dumps(
                    group["ModelEligibilityStatus"]
                    .value_counts(dropna=False)
                    .sort_index()
                    .to_dict(),
                    sort_keys=True,
                ),
                "FitStatusCounts": json.dumps(
                    group["FitStatus"]
                    .fillna("missing")
                    .value_counts()
                    .sort_index()
                    .to_dict(),
                    sort_keys=True,
                ),
                "n_ID_min": int(group["n_ID"].min()),
                "n_ID_max": int(group["n_ID"].max()),
                "n_ContactUnitID_min": int(group["n_ContactUnitID"].min()),
                "n_ContactUnitID_max": int(group["n_ContactUnitID"].max()),
            }
        )
    return pd.DataFrame(records)


def _input_manifest(
    config: StructuralConnectivityConfig,
    *,
    include_visualization: bool = True,
) -> pd.DataFrame:
    inputs = [
        ("configuration", config.path),
        ("local_rows", config.output_path("local_rows")),
        ("connectivity_rows", config.output_path("connectivity_rows")),
        (
            "channel_field_weighted_lift",
            config.output_path("channel_field_weighted_lift"),
        ),
        ("r_runner", Path(__file__).parent / "r" / "run_contact_lmm.R"),
    ]
    inputs.extend(
        (
            f"field_excess_local_outcome:{row['metric']}:{row['feature_output']}",
            Path(str(row["path"])),
        )
        for row in config["inference"]["contact_lmm"].get(
            "field_excess_local_outcomes", []
        )
    )
    if include_visualization:
        style_paths = _visual_style_paths(config)
        inputs.extend(
            [
                ("phase_scalar_style", style_paths["phase_scalar_style"]),
                ("phase_heatmap_style", style_paths["phase_heatmap_style"]),
            ]
        )
    return pd.DataFrame(
        [
            {
                "AnalysisID": config.analysis_id,
                "InputType": input_type,
                "Path": str(path),
                "Exists": path.is_file(),
            }
            for input_type, path in inputs
        ]
    )


def _registered_output_tables(
    tables: Mapping[str, pd.DataFrame],
    figures: pd.DataFrame,
    config: StructuralConnectivityConfig,
    *,
    include_visualization: bool = True,
) -> dict[str, pd.DataFrame]:
    root = config.contact_lmm_root
    named_tables = dict(tables)
    named_tables["inputs"] = _input_manifest(
        config,
        include_visualization=include_visualization,
    )
    named_tables["figures"] = figures
    output_records = [
        {
            "AnalysisID": config.analysis_id,
            "OutputType": name,
            "Path": str(root / f"{name}.csv"),
            "Rows": int(len(table)),
            "Columns": int(len(table.columns)),
            "Status": "complete",
        }
        for name, table in named_tables.items()
    ]
    output_records.extend(
        {
            "AnalysisID": config.analysis_id,
            "OutputType": f"{row.FigureType}_pdf",
            "Path": row.SourcePath,
            "Rows": pd.NA,
            "Columns": pd.NA,
            "Status": row.Status,
        }
        for row in figures.itertuples(index=False)
    )
    outputs = pd.DataFrame(output_records)
    named_tables["outputs"] = outputs
    table_registry = pd.DataFrame(
        [
            {
                "Table": f"{name}.csv",
                "Path": str(root / f"{name}.csv"),
                "Rows": int(len(table)),
                "Columns": int(len(table.columns)),
            }
            for name, table in named_tables.items()
        ]
    )
    named_tables["tables"] = table_registry
    expected = {path.removesuffix(".csv") for path in CONTACT_LMM_TABLES}
    if set(named_tables) != expected:
        raise ValueError(
            f"Contact-level output table contract mismatch: {sorted(set(named_tables) ^ expected)}"
        )
    return named_tables


def _write_outputs(
    tables: Mapping[str, pd.DataFrame],
    figures: pd.DataFrame,
    config: StructuralConnectivityConfig,
    *,
    overwrite: bool,
    include_visualization: bool = True,
) -> None:
    root = config.contact_lmm_root
    named_tables = _registered_output_tables(
        tables,
        figures,
        config,
        include_visualization=include_visualization,
    )
    for name in named_tables:
        atomic_csv(named_tables[name], root / f"{name}.csv", overwrite=overwrite)


def _existing_computation_tables(
    config: StructuralConnectivityConfig,
) -> dict[str, pd.DataFrame]:
    registry_names = {"inputs", "outputs", "figures", "tables"}
    names = {path.removesuffix(".csv") for path in CONTACT_LMM_TABLES}.difference(
        registry_names
    )
    return {
        name: _read_csv(config.contact_lmm_root / f"{name}.csv")
        for name in sorted(names)
    }


def _write_figure_registries(
    figures: pd.DataFrame,
    config: StructuralConnectivityConfig,
    *,
    overwrite: bool,
) -> None:
    root = config.contact_lmm_root
    named_tables = _registered_output_tables(
        _existing_computation_tables(config), figures, config
    )
    for name in ("inputs", "figures", "outputs", "tables"):
        atomic_csv(named_tables[name], root / f"{name}.csv", overwrite=overwrite)


def _render_contact_lmm_figures(
    model_rows: pd.DataFrame,
    slopes: pd.DataFrame,
    curves: pd.DataFrame,
    config: StructuralConnectivityConfig,
    *,
    overwrite: bool,
) -> pd.DataFrame:
    heatmaps = _plot_heatmaps(slopes, config, overwrite=overwrite)
    slope_figures = _plot_slope_figures(
        model_rows,
        curves,
        slopes,
        config,
        overwrite=overwrite,
    )
    figures = pd.concat([heatmaps, slope_figures], ignore_index=True, sort=False)
    if figures["FigureID"].duplicated().any():
        raise ValueError("Contact-level figure IDs must be unique.")
    if figures["PublishedRelativePath"].duplicated().any():
        raise ValueError("Contact-level publication paths must be unique.")
    return figures


def run_contact_lmm_figures(
    config: StructuralConnectivityConfig,
    *,
    dry_run: bool,
    overwrite: bool,
) -> dict[str, Any]:
    """Render registered figures from existing mixed-model result tables."""

    model_rows = _read_csv(config.contact_lmm_root / "model_rows.csv")
    slopes = _read_csv(config.contact_lmm_root / "phase_slopes.csv")
    curves = _read_csv(config.contact_lmm_root / "prediction_grid.csv")
    _visual_styles(config)
    heatmap_count = _heatmap_output_count(_figure_specs(slopes, config))
    slope_count = len(_slope_figure_specs(slopes))
    summary = {
        "status": "ready" if dry_run else "complete",
        "heatmap_figures": int(heatmap_count),
        "slope_figures": int(slope_count),
        "figure_rows": int(heatmap_count + slope_count),
        "mixed_model_refit": False,
        "writes": 0,
    }
    if dry_run:
        return summary
    registry_paths = [
        config.contact_lmm_root / f"{name}.csv"
        for name in ("inputs", "figures", "outputs", "tables")
    ]
    existing = [path for path in registry_paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Contact-level figure registry exists; pass --overwrite to replace "
            f"it: {existing[0]}"
        )
    figures = _render_contact_lmm_figures(
        model_rows,
        slopes,
        curves,
        config,
        overwrite=overwrite,
    )
    _write_figure_registries(figures, config, overwrite=overwrite)
    summary["writes"] = int(len(figures) + 4)
    return summary


def _trash_generated_files(paths: list[Path], *, root: Path) -> None:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return
    resolved_root = root.resolve()
    for path in existing:
        try:
            path.resolve().relative_to(resolved_root)
        except ValueError as error:
            raise ValueError(
                "Refusing to remove a generated file outside the registered output "
                f"directory: {path}"
            ) from error
    trash = Path("/usr/bin/trash")
    if not trash.is_file():
        raise FileNotFoundError(f"macOS Trash command is unavailable: {trash}")
    subprocess.run([str(trash), *(str(path) for path in existing)], check=True)


def _trash_generated_heatmaps(paths: list[Path], *, heatmap_root: Path) -> None:
    _trash_generated_files(paths, root=heatmap_root)


def run_contact_lmm_atlas_grid_heatmaps(
    config: StructuralConnectivityConfig,
    *,
    dry_run: bool,
    overwrite: bool,
) -> dict[str, Any]:
    """Replace only the registered atlas-grid metric heatmaps from saved slopes."""

    visualization = config["inference"]["contact_lmm"]["visualization"]
    atlas_grid = visualization.get("atlas_grid_heatmap")
    if atlas_grid is None:
        raise ValueError("The contact-level atlas-grid heatmap contract is absent.")
    lift_metric = str(atlas_grid["lift_metric"])
    slopes = _read_csv(config.contact_lmm_root / "phase_slopes.csv")
    _visual_styles(config)
    planned = [
        spec
        for spec in _figure_specs(slopes, config)
        if spec["LiftMetric"] == lift_metric
    ]
    summary: dict[str, Any] = {
        "status": "ready" if dry_run else "complete",
        "lift_metric": lift_metric,
        "heatmap_figures": int(_heatmap_output_count(planned)),
        "mixed_model_refit": False,
        "slope_rewrite": False,
        "multiplicity_rewrite": False,
        "writes": 0,
        "stale_heatmaps_trashed": 0,
    }
    if dry_run:
        return summary

    registry_path = config.contact_lmm_root / "figures.csv"
    if registry_path.exists() and not overwrite:
        raise FileExistsError(
            "Contact-level figure registry exists; pass --overwrite to replace "
            f"the registered atlas-grid metric heatmaps: {registry_path}"
        )
    existing_figures = _read_csv(registry_path)
    required = {"FigureID", "FigureType", "LiftMetric", "SourcePath"}
    missing = sorted(required.difference(existing_figures.columns))
    if missing:
        raise KeyError(f"Contact-level figure registry is missing columns: {missing}")
    replacement = existing_figures["FigureType"].eq("heatmap") & existing_figures[
        "LiftMetric"
    ].eq(lift_metric)
    replaced_figures = existing_figures.loc[replacement].copy()
    retained_figures = existing_figures.loc[~replacement].copy()

    replaced_paths = [
        Path(path) for path in replaced_figures["SourcePath"].dropna().astype(str)
    ]
    replaced_existing = [path for path in replaced_paths if path.exists()]
    _trash_generated_heatmaps(
        replaced_paths,
        heatmap_root=config.contact_lmm_root / "figures" / "heatmap",
    )
    atlas_figures = _plot_heatmaps(
        slopes,
        config,
        overwrite=False,
        lift_metrics={lift_metric},
    )
    figures = pd.concat(
        [retained_figures, atlas_figures], ignore_index=True, sort=False
    )
    if figures["FigureID"].duplicated().any():
        raise ValueError("Contact-level figure IDs must be unique.")
    if figures["PublishedRelativePath"].duplicated().any():
        raise ValueError("Contact-level publication paths must be unique.")
    registry_paths = [
        config.contact_lmm_root / f"{name}.csv"
        for name in ("inputs", "figures", "outputs", "tables")
    ]
    _trash_generated_files(registry_paths, root=config.contact_lmm_root)
    _write_figure_registries(figures, config, overwrite=False)

    summary["writes"] = int(len(atlas_figures) + 4)
    summary["stale_heatmaps_trashed"] = int(len(replaced_existing))
    return summary


def _field_excess_significant_phase_slopes(
    slopes: pd.DataFrame,
) -> pd.DataFrame:
    required = {"ModelID", "Phase", "LiftMetric", "P_holm"}
    missing = sorted(required.difference(slopes.columns))
    if missing:
        raise KeyError(f"Phase slopes are missing Finder-selection columns: {missing}")
    adjusted_p = pd.to_numeric(slopes["P_holm"], errors="coerce")
    selected = slopes.loc[
        slopes["LiftMetric"].eq(FIELD_EXCESS_METRIC)
        & adjusted_p.notna()
        & np.isfinite(adjusted_p)
        & adjusted_p.lt(0.05)
    ].copy()
    if selected[["ModelID", "Phase"]].duplicated().any():
        raise ValueError(
            "FieldExcess Finder fit selection contains duplicate ModelID-Phase rows."
        )
    return selected


def _field_excess_fixed_phase_slopes(slopes: pd.DataFrame) -> pd.DataFrame:
    required = {
        "ModelID",
        "Phase",
        "OutcomeDomain",
        "ConnectomeSource",
        "LiftMetric",
        "LiftPredictor",
        "Polar",
        "Metric",
        "Band",
    }
    missing = sorted(required.difference(slopes.columns))
    if missing:
        raise KeyError(f"Phase slopes are missing fixed-fit columns: {missing}")
    selected = slopes.loc[
        slopes["LiftMetric"].eq(FIELD_EXCESS_METRIC)
        & slopes["OutcomeDomain"].eq("connectivity")
        & slopes["ConnectomeSource"].isin(FIELD_EXCESS_FINDER_CONNECTOME_SOURCES)
        & slopes["LiftPredictor"].isin(FIELD_EXCESS_FINDER_PREDICTORS)
        & slopes["Polar"].eq(FIELD_EXCESS_FINDER_POLAR)
        & slopes["Phase"].eq(FIELD_EXCESS_FINDER_PHASE)
        & slopes["Band"].eq(FIELD_EXCESS_FINDER_BAND)
        & slopes["Metric"].isin(FIELD_EXCESS_FINDER_METRICS)
    ].copy()
    if selected[["ModelID", "Phase"]].duplicated().any():
        raise ValueError("Fixed FieldExcess fit selection contains duplicate rows.")
    cells = selected.groupby(
        ["ConnectomeSource", "LiftPredictor", "Metric"], dropna=False
    ).size()
    if len(selected) != FIELD_EXCESS_FINDER_FIXED_FIT_COUNT or not cells.eq(1).all():
        raise ValueError(
            "Fixed FieldExcess fit selection differs from the approved "
            f"2 x 3 x 4 contract: expected {FIELD_EXCESS_FINDER_FIXED_FIT_COUNT}, "
            f"got {len(selected)}."
        )
    return selected


def _field_excess_atlas_grid_fit_specs(
    slopes: pd.DataFrame,
) -> list[dict[str, Any]]:
    fixed = _field_excess_fixed_phase_slopes(slopes)
    records: list[dict[str, Any]] = []
    index = 0
    for predictor in FIELD_EXCESS_FINDER_PREDICTORS:
        for metric in FIELD_EXCESS_FINDER_METRICS:
            index += 1
            cell = fixed.loc[
                fixed["LiftPredictor"].eq(predictor) & fixed["Metric"].eq(metric)
            ].copy()
            observed_sources = tuple(
                source
                for source in FIELD_EXCESS_FINDER_CONNECTOME_SOURCES
                if source in set(cell["ConnectomeSource"].astype(str))
            )
            if (
                len(cell) != len(FIELD_EXCESS_FINDER_CONNECTOME_SOURCES)
                or observed_sources != FIELD_EXCESS_FINDER_CONNECTOME_SOURCES
            ):
                raise ValueError(
                    "Combined-atlas FieldExcess fit requires one row from each "
                    f"registered connectome: predictor={predictor}, metric={metric}."
                )
            if not cell["PairRegion"].eq("SNr-STN").all():
                raise ValueError(
                    "Combined-atlas FieldExcess fits require PairRegion=SNr-STN."
                )
            filename = (
                f"field-excess_predictor-{_sanitize(predictor)}_"
                f"polar-{FIELD_EXCESS_FINDER_POLAR}_metric-{_sanitize(metric)}_"
                f"band-{FIELD_EXCESS_FINDER_BAND}_"
                f"phase-{FIELD_EXCESS_FINDER_PHASE}_atlas-grid.pdf"
            )
            records.append(
                {
                    "FigureID": f"LMMATLASGRIDFIT{index:03d}",
                    "FigureType": FIELD_EXCESS_ATLAS_GRID_FIGURE_TYPE,
                    "OutcomeDomain": "connectivity",
                    "ConnectomeSource": pd.NA,
                    "ConnectomeSources": ";".join(
                        FIELD_EXCESS_FINDER_CONNECTOME_SOURCES
                    ),
                    "LiftMetric": FIELD_EXCESS_METRIC,
                    "LiftPredictor": predictor,
                    "Polar": FIELD_EXCESS_FINDER_POLAR,
                    "Metric": metric,
                    "FeatureOutput": "mean-scalar",
                    "Region": "",
                    "PairRegion": "SNr-STN",
                    "Band": FIELD_EXCESS_FINDER_BAND,
                    "Phase": FIELD_EXCESS_FINDER_PHASE,
                    "Filename": filename,
                    "ModelIDs": ";".join(
                        cell.set_index("ConnectomeSource")
                        .loc[
                            list(FIELD_EXCESS_FINDER_CONNECTOME_SOURCES),
                            "ModelID",
                        ]
                        .astype(str)
                    ),
                }
            )
    required = {
        "ModelID",
        "Phase",
        "OutcomeDomain",
        "ConnectomeSource",
        "LiftMetric",
        "LiftPredictor",
        "Polar",
        "Metric",
        "FeatureOutput",
        "Region",
        "Band",
    }
    missing = sorted(required.difference(slopes.columns))
    if missing:
        raise KeyError(f"Phase slopes are missing local atlas-grid columns: {missing}")
    for phase in FIELD_EXCESS_FINDER_LOCAL_PHASES:
        index += 1
        cell = slopes.loc[
            slopes["LiftMetric"].eq(FIELD_EXCESS_METRIC)
            & slopes["OutcomeDomain"].eq("local")
            & slopes["ConnectomeSource"].isin(FIELD_EXCESS_FINDER_CONNECTOME_SOURCES)
            & slopes["LiftPredictor"].eq(FIELD_EXCESS_FINDER_LOCAL_REGION)
            & slopes["Polar"].eq(FIELD_EXCESS_FINDER_POLAR)
            & slopes["Phase"].eq(phase)
            & slopes["Band"].eq(FIELD_EXCESS_FINDER_BAND)
            & slopes["Metric"].eq(FIELD_EXCESS_FINDER_LOCAL_METRIC)
            & slopes["FeatureOutput"].eq(FIELD_EXCESS_FINDER_LOCAL_FEATURE_OUTPUT)
            & slopes["Region"].eq(FIELD_EXCESS_FINDER_LOCAL_REGION)
        ].copy()
        observed_sources = tuple(
            source
            for source in FIELD_EXCESS_FINDER_CONNECTOME_SOURCES
            if source in set(cell["ConnectomeSource"].astype(str))
        )
        if (
            len(cell) != len(FIELD_EXCESS_FINDER_CONNECTOME_SOURCES)
            or observed_sources != FIELD_EXCESS_FINDER_CONNECTOME_SOURCES
        ):
            raise ValueError(
                "Combined-atlas local FieldExcess fit requires one row from each "
                f"registered connectome: phase={phase}."
            )
        filename = (
            f"field-excess_predictor-{FIELD_EXCESS_FINDER_LOCAL_REGION}_"
            f"polar-{FIELD_EXCESS_FINDER_POLAR}_"
            f"metric-{FIELD_EXCESS_FINDER_LOCAL_METRIC}_"
            f"band-{FIELD_EXCESS_FINDER_BAND}_phase-{phase}_"
            f"feature-output-{FIELD_EXCESS_FINDER_LOCAL_FEATURE_OUTPUT}_"
            f"region-{FIELD_EXCESS_FINDER_LOCAL_REGION}_atlas-grid.pdf"
        )
        records.append(
            {
                "FigureID": f"LMMATLASGRIDFIT{index:03d}",
                "FigureType": FIELD_EXCESS_ATLAS_GRID_FIGURE_TYPE,
                "OutcomeDomain": "local",
                "ConnectomeSource": pd.NA,
                "ConnectomeSources": ";".join(FIELD_EXCESS_FINDER_CONNECTOME_SOURCES),
                "LiftMetric": FIELD_EXCESS_METRIC,
                "LiftPredictor": FIELD_EXCESS_FINDER_LOCAL_REGION,
                "Polar": FIELD_EXCESS_FINDER_POLAR,
                "Metric": FIELD_EXCESS_FINDER_LOCAL_METRIC,
                "FeatureOutput": FIELD_EXCESS_FINDER_LOCAL_FEATURE_OUTPUT,
                "Region": FIELD_EXCESS_FINDER_LOCAL_REGION,
                "PairRegion": "",
                "Band": FIELD_EXCESS_FINDER_BAND,
                "Phase": phase,
                "Filename": filename,
                "ModelIDs": ";".join(
                    cell.set_index("ConnectomeSource")
                    .loc[
                        list(FIELD_EXCESS_FINDER_CONNECTOME_SOURCES),
                        "ModelID",
                    ]
                    .astype(str)
                ),
            }
        )
    if len(records) != FIELD_EXCESS_FINDER_ATLAS_GRID_FIT_COUNT:
        raise ValueError(
            "Combined-atlas FieldExcess fit count differs from the approved "
            f"contract: expected {FIELD_EXCESS_FINDER_ATLAS_GRID_FIT_COUNT}, "
            f"got {len(records)}."
        )
    return records


def _atlas_specific_x_limits(
    raw: pd.DataFrame,
    curve: pd.DataFrame,
    *,
    source: str,
) -> tuple[float, float]:
    values = pd.concat(
        [
            raw.loc[raw["ConnectomeSource"].eq(source), "X_z"],
            curve.loc[curve["ConnectomeSource"].eq(source), "X_z"],
        ],
        ignore_index=True,
    )
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise ValueError(f"No finite X_z values are available for {source}.")
    lower = float(finite.min())
    upper = float(finite.max())
    span = upper - lower
    padding = 0.05 * span if span > 0.0 else 0.05
    return lower - padding, upper + padding


def _format_atlas_grid_p_holm(value: float) -> str:
    """Format a registered Holm P value for the curated atlas-grid fits."""

    return "<0.001" if value < 0.001 else f"{value:.3f}"


def _plot_field_excess_atlas_grid_fits(
    model_rows: pd.DataFrame,
    curves: pd.DataFrame,
    slopes: pd.DataFrame,
    config: StructuralConnectivityConfig,
    *,
    overwrite: bool,
) -> pd.DataFrame:
    scalar_style, _ = _visual_styles(config)
    style = scalar_style["style"]
    font = style["font"]
    widths = style["line_width_pt"]
    geometry = style["geometry_mm"]
    strips = style["strips"]
    visualization = config["inference"]["contact_lmm"]["visualization"]
    atlas_labels = visualization["connectome_labels"]
    atlas_order = [
        atlas_labels[source] for source in FIELD_EXCESS_FINDER_CONNECTOME_SOURCES
    ]
    figure_root = config.contact_lmm_root / "figures" / "slope_atlas_grid"
    records: list[dict[str, Any]] = []
    visualdf.set_greek_symbol_font_enabled(False)

    for spec in _field_excess_atlas_grid_fit_specs(slopes):
        phase = str(spec["Phase"])
        model_ids = str(spec["ModelIDs"]).split(";")
        raw = model_rows.loc[
            model_rows["ModelID"].isin(model_ids)
            & model_rows["RowEligibilityStatus"].eq("eligible")
            & model_rows["Phase"].eq(phase)
        ].copy()
        curve = curves.loc[
            curves["ModelID"].isin(model_ids) & curves["Phase"].eq(phase)
        ].copy()
        slope = slopes.loc[
            slopes["ModelID"].isin(model_ids) & slopes["Phase"].eq(phase)
        ].copy()
        for label, table in (("raw", raw), ("curve", curve), ("slope", slope)):
            observed = set(table["ConnectomeSource"].dropna().astype(str))
            if observed != set(FIELD_EXCESS_FINDER_CONNECTOME_SOURCES):
                raise ValueError(
                    f"Combined-atlas {label} rows do not cover all connectomes: "
                    f"predictor={spec['LiftPredictor']}, metric={spec['Metric']}."
                )
        if slope.groupby("ConnectomeSource").size().ne(1).any():
            raise ValueError(
                "Combined-atlas fit requires exactly one slope row per connectome."
            )

        region_label = _slope_region_label(spec, scalar_style)
        polar_label = scalar_style["labels"]["polarity"][str(spec["Polar"])]
        phase_label = visualization["phase_display_labels"][phase]
        context = f"{region_label} | {polar_label} | {phase_label}"
        for table in (raw, curve, slope):
            table["AtlasPanel"] = table["ConnectomeSource"].map(atlas_labels)
            table["ContextPanel"] = context
        curve = curve.rename(
            columns={
                "PredictedValue": "emmean",
                "Lower": "lower.CL",
                "Upper": "upper.CL",
            }
        )
        predictor_label = {
            "STN": "STN streamline field-exposure excess",
            "SNr": "SNr streamline field-exposure excess",
        }[str(spec["LiftPredictor"])]
        y_label = rf"$\Delta$ {y_axis_label(spec, scalar_style)}"
        figure = visualdf.plot_triple_interaction_fit(
            df=raw,
            curve=curve,
            slope=slope,
            value_col="Value",
            x_var="X_z",
            panel_var="AtlasPanel",
            facet_var="ContextPanel",
            color_var="ID",
            panel_levels=atlas_order,
            facet_levels=[context],
            x_label=predictor_label,
            y_label=y_label,
            title=None,
            font_family=font["latin"],
            palette="viridis",
            jitter_alpha=1.0,
            jitter_size=10.0,
            curve_line_width=widths["emm_line"],
            curve_line_color=style["emm"]["color"],
            ribbon_alpha=style["box"]["fill_alpha"],
            show_ribbon=True,
            grid=style["axes"]["grid"],
            dpi=style["dpi"],
            seed=style["jitter"]["seed"],
            show_top_right_axes=style["axes"]["show_top_right_spines"],
            label_fontsize=font["strip_pt"],
            label_top_bg_color=strips["top_background"],
            label_right_bg_color=strips["right_background"],
            label_text_color=strips["text_color"],
            label_fontweight=strips["font_weight"],
            strip_top_height_mm=geometry["strip_top_height"],
            strip_right_width_mm=geometry["strip_right_width"],
            strip_pad_mm=geometry["strip_pad"],
            title_fontsize=font["axis_pt"],
            axis_label_fontsize=font["axis_pt"],
            tick_label_fontsize=font["tick_pt"],
            legend_loc="none",
            boxsize=(30.0, 32.0),
            panel_gap=(2.0, 3.0),
            x_label_offset_mm=geometry["x_label_offset"],
            y_label_offset_mm=geometry["y_label_offset"],
            slope_beta_col="StandardizedSlope",
            slope_lower_col="StandardizedLower",
            slope_upper_col="StandardizedUpper",
            slope_p_col="P_holm",
            slope_text_fmt=(
                r"$\beta_{{\mathrm{{std}}}} = {beta:.2f}$"
                "\n"
                r"$p_{{\mathrm{{Holm}}}} = {p:.3g}$ ({stars})"
            ),
            slope_text_loc="upper left",
            slope_text_box_alpha=0.0,
            share_x_axes=False,
            transparent=style["transparent"],
        )
        plot_axes = figure.axes[: len(FIELD_EXCESS_FINDER_CONNECTOME_SOURCES)]
        if len(plot_axes) != len(FIELD_EXCESS_FINDER_CONNECTOME_SOURCES):
            visualdf.plt.close(figure)
            raise ValueError("Combined-atlas fit did not create four plot panels.")
        for axis, source in zip(
            plot_axes, FIELD_EXCESS_FINDER_CONNECTOME_SOURCES, strict=True
        ):
            axis.set_xlim(*_atlas_specific_x_limits(raw, curve, source=source))
            slope_row = slope.loc[slope["ConnectomeSource"].eq(source)].iloc[0]
            standardized_slope = float(slope_row["StandardizedSlope"])
            p_holm = float(slope_row["P_holm"])
            if len(axis.texts) != 1:
                visualdf.plt.close(figure)
                raise ValueError(
                    "Combined-atlas fit requires one slope annotation per panel."
                )
            annotation = axis.texts[0]
            p_display = _format_atlas_grid_p_holm(p_holm)
            p_relation = "<" if p_display.startswith("<") else "="
            p_value_display = p_display.removeprefix("<")
            stars = visualdf.p_to_stars(p_holm)
            annotation.set_text(
                rf"$\beta_{{\mathrm{{std}}}} = {standardized_slope:.2f}$"
                "\n"
                rf"$p_{{\mathrm{{Holm}}}} {p_relation} {p_value_display}$ "
                f"({stars})"
            )
            annotation_at_top = (
                spec["OutcomeDomain"] == "connectivity" or standardized_slope > 0.0
            )
            annotation.set_position((0.05, 0.95 if annotation_at_top else 0.05))
            annotation.set_verticalalignment("top" if annotation_at_top else "bottom")

        source_path = figure_root / str(spec["Filename"])
        try:
            _atomic_pdf(
                figure,
                source_path,
                overwrite=overwrite,
                dpi=style["dpi"],
            )
        finally:
            visualdf.plt.close(figure)
        width_mm, height_mm = _pdf_page_dimensions_mm(source_path)
        if width_mm > 150.0 or height_mm > 210.0:
            _trash_generated_files([source_path], root=config.contact_lmm_root)
            raise ValueError(
                "Combined-atlas fit exceeds the registered PDF size: "
                f"{width_mm:.2f} x {height_mm:.2f} mm."
            )
        records.append(
            {
                **spec,
                "OutcomeChangeBasis": "adjacent_phase",
                "DisplayContext": str(spec["LiftPredictor"]),
                "Orientation": "atlas-grid",
                "SourcePath": str(source_path),
                "PublishedRelativePath": (f"slope_atlas_grid/{spec['Filename']}"),
                "SourcePointTable": str(config.contact_lmm_root / "model_rows.csv"),
                "SourceCurveTable": str(
                    config.contact_lmm_root / "prediction_grid.csv"
                ),
                "SourceSlopeTable": str(config.contact_lmm_root / "phase_slopes.csv"),
                "PhasePanels": phase,
                "ValueColumn": "Value",
                "PredictorColumn": "X_z",
                "SlopeColumn": "StandardizedSlope",
                "SignificanceColumn": "P_holm",
                "YAxisLabel": y_label,
                "Renderer": "lfp_viz.visualdf.plot_triple_interaction_fit",
                "StyleConfig": str(scalar_style.path),
                "Layout": "atlas-grid",
                "AtlasColumns": ";".join(FIELD_EXCESS_FINDER_CONNECTOME_SOURCES),
                "WidthMm": width_mm,
                "HeightMm": height_mm,
                "Status": "complete",
            }
        )
    return pd.DataFrame(records)


def _field_excess_finder_phase_slopes(slopes: pd.DataFrame) -> pd.DataFrame:
    significant = _field_excess_significant_phase_slopes(slopes)
    fixed = _field_excess_fixed_phase_slopes(slopes)
    selected = (
        pd.concat([significant, fixed], ignore_index=True, sort=False)
        .drop_duplicates(["ModelID", "Phase"], keep="first")
        .sort_values(["ModelID", "Phase"], kind="mergesort")
        .reset_index(drop=True)
    )
    if len(selected) != FIELD_EXCESS_FINDER_RENDERED_FIT_COUNT:
        raise ValueError(
            "Rendered FieldExcess fit selection differs from the approved union: "
            f"expected {FIELD_EXCESS_FINDER_RENDERED_FIT_COUNT}, got "
            f"{len(selected)}."
        )
    return selected


def run_contact_lmm_field_excess_finder_figures(
    config: StructuralConnectivityConfig,
    *,
    dry_run: bool,
    overwrite: bool,
    refresh_heatmaps: bool = True,
) -> dict[str, Any]:
    """Refresh approved FieldExcess Finder figures at the requested scope."""

    model_rows = _read_csv(config.contact_lmm_root / "model_rows.csv")
    slopes = _read_csv(config.contact_lmm_root / "phase_slopes.csv")
    curves = _read_csv(config.contact_lmm_root / "prediction_grid.csv")
    _visual_styles(config)
    planned_heatmaps = (
        [
            spec
            for spec in _figure_specs(slopes, config)
            if spec["LiftMetric"] == FIELD_EXCESS_METRIC
        ]
        if refresh_heatmaps
        else []
    )
    heatmap_count = _heatmap_output_count(planned_heatmaps)
    selected_slopes = _field_excess_finder_phase_slopes(slopes)
    fixed_fit_count = len(_slope_figure_specs(_field_excess_fixed_phase_slopes(slopes)))
    fit_count = len(_slope_figure_specs(selected_slopes))
    if refresh_heatmaps and heatmap_count != FIELD_EXCESS_FINDER_HEATMAP_COUNT:
        raise ValueError(
            "FieldExcess Finder heatmap count differs from the approved contract: "
            f"expected {FIELD_EXCESS_FINDER_HEATMAP_COUNT}, got {heatmap_count}."
        )
    if fit_count != FIELD_EXCESS_FINDER_RENDERED_FIT_COUNT:
        raise ValueError(
            "FieldExcess Finder fit count differs from the approved contract: "
            f"expected {FIELD_EXCESS_FINDER_RENDERED_FIT_COUNT}, got {fit_count}."
        )
    summary: dict[str, Any] = {
        "status": "ready" if dry_run else "complete",
        "lift_metric": FIELD_EXCESS_METRIC,
        "heatmap_figures": int(heatmap_count),
        "selected_fit_figures": int(fit_count),
        "fixed_fit_figures": int(fixed_fit_count),
        "mixed_model_refit": False,
        "slope_rewrite": False,
        "multiplicity_rewrite": False,
        "writes": 0,
        "replaced_figures_trashed": 0,
    }
    if dry_run:
        return summary

    registry_path = config.contact_lmm_root / "figures.csv"
    if registry_path.exists() and not overwrite:
        raise FileExistsError(
            "Contact-level figure registry exists; pass --overwrite to replace "
            f"the registered Finder figure subset: {registry_path}"
        )
    existing_figures = _read_csv(registry_path)
    required = {
        "FigureID",
        "FigureType",
        "LiftMetric",
        "SourcePath",
        "PublishedRelativePath",
    }
    missing = sorted(required.difference(existing_figures.columns))
    if missing:
        raise KeyError(f"Contact-level figure registry is missing columns: {missing}")
    phase = existing_figures.get(
        "Phase", pd.Series(pd.NA, index=existing_figures.index, dtype="object")
    )
    phase_specific = phase.notna() & phase.astype(str).str.len().gt(0)
    replacement = existing_figures["LiftMetric"].eq(FIELD_EXCESS_METRIC) & (
        (refresh_heatmaps & existing_figures["FigureType"].eq("heatmap"))
        | (existing_figures["FigureType"].eq("slope") & phase_specific)
    )
    replaced_figures = existing_figures.loc[replacement].copy()
    retained_figures = existing_figures.loc[~replacement].copy()
    replaced_paths = [
        Path(path) for path in replaced_figures["SourcePath"].dropna().astype(str)
    ]
    replaced_existing = [path for path in replaced_paths if path.exists()]
    _trash_generated_files(replaced_paths, root=config.contact_lmm_root)

    heatmaps = (
        _plot_heatmaps(
            slopes,
            config,
            overwrite=False,
            lift_metrics={FIELD_EXCESS_METRIC},
        )
        if refresh_heatmaps
        else existing_figures.iloc[0:0].copy()
    )
    fit_figures = _plot_slope_figures(
        model_rows,
        curves,
        selected_slopes,
        config,
        overwrite=False,
    )
    if refresh_heatmaps and len(heatmaps) != FIELD_EXCESS_FINDER_HEATMAP_COUNT:
        raise ValueError("Rendered FieldExcess heatmap count changed during refresh.")
    if len(fit_figures) != FIELD_EXCESS_FINDER_RENDERED_FIT_COUNT:
        raise ValueError("Rendered FieldExcess fit count changed during refresh.")
    figures = pd.concat(
        [retained_figures, heatmaps, fit_figures], ignore_index=True, sort=False
    )
    if figures["FigureID"].duplicated().any():
        raise ValueError("Contact-level figure IDs must be unique.")
    if figures["PublishedRelativePath"].duplicated().any():
        raise ValueError("Contact-level publication paths must be unique.")
    registry_paths = [
        config.contact_lmm_root / f"{name}.csv"
        for name in ("inputs", "figures", "outputs", "tables")
    ]
    _trash_generated_files(registry_paths, root=config.contact_lmm_root)
    _write_figure_registries(figures, config, overwrite=False)

    summary["writes"] = int(len(heatmaps) + len(fit_figures) + 4)
    summary["replaced_figures_trashed"] = int(len(replaced_existing))
    return summary


def run_contact_lmm_field_excess_atlas_grid_fits(
    config: StructuralConnectivityConfig,
    *,
    dry_run: bool,
    overwrite: bool,
) -> dict[str, Any]:
    """Render and register the eight approved combined-atlas FieldExcess fits."""

    model_rows = _read_csv(config.contact_lmm_root / "model_rows.csv")
    slopes = _read_csv(config.contact_lmm_root / "phase_slopes.csv")
    curves = _read_csv(config.contact_lmm_root / "prediction_grid.csv")
    planned_count = len(_field_excess_atlas_grid_fit_specs(slopes))
    if planned_count != FIELD_EXCESS_FINDER_ATLAS_GRID_FIT_COUNT:
        raise ValueError(
            "Combined-atlas FieldExcess fit count differs from the approved "
            f"contract: expected {FIELD_EXCESS_FINDER_ATLAS_GRID_FIT_COUNT}, "
            f"got {planned_count}."
        )
    summary: dict[str, Any] = {
        "status": "ready" if dry_run else "complete",
        "lift_metric": FIELD_EXCESS_METRIC,
        "atlas_grid_fit_figures": int(planned_count),
        "mixed_model_refit": False,
        "slope_rewrite": False,
        "multiplicity_rewrite": False,
        "single_atlas_fit_rewrite": False,
        "writes": 0,
        "replaced_figures_trashed": 0,
    }
    if dry_run:
        return summary

    registry_path = config.contact_lmm_root / "figures.csv"
    if registry_path.exists() and not overwrite:
        raise FileExistsError(
            "Contact-level figure registry exists; pass --overwrite to register "
            f"the combined-atlas fits: {registry_path}"
        )
    existing_figures = _read_csv(registry_path)
    required = {"FigureID", "FigureType", "SourcePath", "PublishedRelativePath"}
    missing = sorted(required.difference(existing_figures.columns))
    if missing:
        raise KeyError(f"Contact-level figure registry is missing columns: {missing}")
    replacement = existing_figures["FigureType"].eq(FIELD_EXCESS_ATLAS_GRID_FIGURE_TYPE)
    replaced_figures = existing_figures.loc[replacement].copy()
    retained_figures = existing_figures.loc[~replacement].copy()
    replaced_paths = [
        Path(path) for path in replaced_figures["SourcePath"].dropna().astype(str)
    ]
    replaced_existing = [path for path in replaced_paths if path.exists()]
    _trash_generated_files(replaced_paths, root=config.contact_lmm_root)

    atlas_grid_fits = _plot_field_excess_atlas_grid_fits(
        model_rows,
        curves,
        slopes,
        config,
        overwrite=False,
    )
    if len(atlas_grid_fits) != FIELD_EXCESS_FINDER_ATLAS_GRID_FIT_COUNT:
        raise ValueError("Rendered combined-atlas FieldExcess fit count changed.")
    figures = pd.concat(
        [retained_figures, atlas_grid_fits], ignore_index=True, sort=False
    )
    if figures["FigureID"].duplicated().any():
        raise ValueError("Contact-level figure IDs must be unique.")
    if figures["PublishedRelativePath"].duplicated().any():
        raise ValueError("Contact-level publication paths must be unique.")
    registry_paths = [
        config.contact_lmm_root / f"{name}.csv"
        for name in ("inputs", "figures", "outputs", "tables")
    ]
    _trash_generated_files(registry_paths, root=config.contact_lmm_root)
    _write_figure_registries(figures, config, overwrite=False)

    summary["writes"] = int(len(atlas_grid_fits) + 4)
    summary["replaced_figures_trashed"] = int(len(replaced_existing))
    return summary


def run_contact_lmm_slope_adjustment(
    config: StructuralConnectivityConfig,
    *,
    dry_run: bool,
    overwrite: bool,
) -> dict[str, Any]:
    """Refresh within-model Holm values and their dependent figures."""

    settings = config["inference"]["contact_lmm"]
    if settings["slope_p_adjust"] != "holm":
        raise ValueError(
            "The contact-level slope-adjustment stage requires slope_p_adjust=holm."
        )
    slopes = _read_csv(config.contact_lmm_root / "phase_slopes.csv")
    families = _read_csv(config.contact_lmm_root / "multiplicity_families.csv")
    model_rows = _read_csv(config.contact_lmm_root / "model_rows.csv")
    curves = _read_csv(config.contact_lmm_root / "prediction_grid.csv")
    adjusted_slopes, slope_families = _register_within_model_holm(
        slopes,
        phases=settings["phase_levels"],
    )
    adjusted_slopes["Star"] = adjusted_slopes["P_holm"].map(_significance_star)
    retained_families = families.loc[~families["ResultTable"].eq("phase_slopes")].copy()
    if "CorrectionMethod" not in retained_families.columns:
        retained_families["CorrectionMethod"] = "BH"
    else:
        retained_families["CorrectionMethod"] = retained_families[
            "CorrectionMethod"
        ].fillna("BH")
    adjusted_families = pd.concat(
        [retained_families, slope_families], ignore_index=True, sort=False
    ).sort_values(["ResultTable", "OutcomeDomain", "FamilyKey"], kind="mergesort")
    _visual_styles(config)
    heatmap_count = _heatmap_output_count(_figure_specs(adjusted_slopes, config))
    slope_count = len(_slope_figure_specs(adjusted_slopes))
    summary = {
        "status": "ready" if dry_run else "complete",
        "adjustment_method": "holm",
        "phase_slope_rows": int(len(adjusted_slopes)),
        "phase_slope_families": int(len(slope_families)),
        "heatmap_figures": int(heatmap_count),
        "slope_figures": int(slope_count),
        "figure_rows": int(heatmap_count + slope_count),
        "mixed_model_refit": False,
        "omnibus_rewrite": False,
        "prediction_rewrite": False,
        "writes": 0,
    }
    if dry_run:
        return summary
    replaced_tables = [
        config.contact_lmm_root / "phase_slopes.csv",
        config.contact_lmm_root / "multiplicity_families.csv",
        *(
            config.contact_lmm_root / f"{name}.csv"
            for name in ("inputs", "figures", "outputs", "tables")
        ),
    ]
    existing = [path for path in replaced_tables if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Contact-level slope-adjustment output exists; pass --overwrite to "
            f"replace it: {existing[0]}"
        )
    figures = _render_contact_lmm_figures(
        model_rows,
        adjusted_slopes,
        curves,
        config,
        overwrite=overwrite,
    )
    atomic_csv(
        adjusted_slopes,
        config.contact_lmm_root / "phase_slopes.csv",
        overwrite=overwrite,
    )
    atomic_csv(
        adjusted_families.reset_index(drop=True),
        config.contact_lmm_root / "multiplicity_families.csv",
        overwrite=overwrite,
    )
    _write_figure_registries(figures, config, overwrite=overwrite)
    summary["writes"] = int(len(figures) + 6)
    return summary


def run_contact_lmm(
    config: StructuralConnectivityConfig,
    *,
    dry_run: bool,
    overwrite: bool,
) -> dict[str, Any]:
    """Prepare contact rows, fit every eligible model, and write neutral outputs."""

    planned_paths = [config.contact_lmm_root / name for name in CONTACT_LMM_TABLES]
    existing = [path for path in planned_paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Contact-level output exists; pass --overwrite to replace it: {existing[0]}"
        )
    local, network = _contact_lmm_input_rows(config)
    long_rows = _prepare_long_rows(local, network)
    predictors, standardized_rows = _standardize_predictors(long_rows)
    model_plan = _planned_models(config)
    model_rows, models = _attach_models_and_support(
        standardized_rows, model_plan, config
    )
    outcome_sd = _outcome_sd(local, network)
    heatmap_count = _heatmap_output_count(
        _figure_specs(
            models[[*MODEL_KEY]].merge(
                pd.DataFrame({"Phase": config["outcomes"]["phases"]}), how="cross"
            ),
            config,
        )
    )
    slope_count = len(models) * len(config["inference"]["contact_lmm"]["phase_levels"])
    summary = {
        "status": "ready" if dry_run else "complete",
        "predictor_units": int(len(predictors)),
        "model_rows": int(len(model_rows)),
        "planned_models": int(len(models)),
        "eligible_models": int(models["ModelEligibilityStatus"].eq("eligible").sum()),
        "planned_heatmap_figures": int(heatmap_count),
        "planned_slope_figures": int(slope_count),
        "planned_figures": int(heatmap_count + slope_count),
        "writes": 0,
    }
    if dry_run:
        return summary

    r_outputs = _run_r_models(model_rows, models, config)
    processed = _postprocess(r_outputs, models, outcome_sd, config)
    coverage = _coverage(processed["models"])
    figures = _render_contact_lmm_figures(
        model_rows,
        processed["phase_slopes"],
        processed["prediction_grid"],
        config,
        overwrite=overwrite,
    )
    output_tables = {
        "lift_predictors": predictors,
        "model_rows": model_rows,
        "models": processed["models"],
        "model_fit": processed["model_fit"],
        "fixed_effects": processed["fixed_effects"],
        "omnibus_tests": processed["omnibus_tests"],
        "phase_slopes": processed["phase_slopes"],
        "prediction_grid": processed["prediction_grid"],
        "outcome_sd": outcome_sd,
        "coverage": coverage,
        "multiplicity_families": processed["multiplicity_families"],
    }
    _write_outputs(
        output_tables,
        figures,
        config,
        overwrite=overwrite,
    )
    summary["fit_status_counts"] = (
        processed["models"]["FitStatus"]
        .fillna("missing")
        .value_counts()
        .sort_index()
        .to_dict()
    )
    summary["phase_slope_rows"] = int(len(processed["phase_slopes"]))
    summary["figure_rows"] = int(len(figures))
    summary["writes"] = int(len(CONTACT_LMM_TABLES) + len(figures))
    return summary
