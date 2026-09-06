"""Prepare source-specific LFP/Lift rows and the frozen Spearman analyses."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from lfp_clinical.correlation import _benjamini_hochberg, _ordinary_group

from .artifacts import atomic_csv
from .config import StructuralConnectivityConfig


LIFT_METRIC_COLUMNS = {
    "BinaryLift": ("Lift", "LiftStatus"),
    "FieldWeightedLift": ("FieldWeightedLift", "FieldWeightedLiftStatus"),
    "FieldExcess": ("FieldExcessE_Vm", "FieldExcessStatus"),
}

FIELD_EXCESS_COMPONENT_COLUMNS = (
    "SeedThresholdedMeanPeakE_Vm",
    "RegionThresholdedMeanPeakE_Vm",
)


def derive_field_excess(table: pd.DataFrame) -> pd.DataFrame:
    """Derive the registered absolute field contrast from existing Lift support."""

    required = {*FIELD_EXCESS_COMPONENT_COLUMNS, "FieldWeightedLiftStatus"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise KeyError(f"Field-exposure table is missing columns: {missing}")
    output = table.copy()
    seed = pd.to_numeric(output["SeedThresholdedMeanPeakE_Vm"], errors="coerce")
    region = pd.to_numeric(output["RegionThresholdedMeanPeakE_Vm"], errors="coerce")
    finite = np.isfinite(seed) & np.isfinite(region)
    source_status = output["FieldWeightedLiftStatus"].astype("string")
    source_ok = source_status.fillna("").eq("ok")
    status = source_status.fillna("missing_field_weighted_lift_status")
    status = status.where(~source_ok, "ok")
    status = status.where(
        ~(source_ok & ~finite),
        "nonfinite_field_excess_component",
    )
    output["FieldExcessE_Vm"] = (seed - region).where(finite)
    output["FieldExcessStatus"] = status
    return output


def _read_pickle(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Normalized LFP table does not exist: {path}")
    value = pd.read_pickle(path)
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"Normalized LFP input is not a DataFrame: {path}")
    return value


def _validate_normalized_contract(
    table: pd.DataFrame,
    path: Path,
    *,
    domain: str,
    metric: str,
    transform: str,
    transform_policy_status: str = "matched",
) -> None:
    expected_attrs = {
        "domain": domain,
        "metric": metric,
        "applied_transform_mode": transform,
        "transform_policy_status": transform_policy_status,
        "normalization_mode": "mean",
        "normalization_baseline_stat": "mean",
        "normalization_baseline": {"Phase": "Pre"},
        "aggregation_level": "contact",
    }
    mismatches = {
        key: (table.attrs.get(key), expected)
        for key, expected in expected_attrs.items()
        if table.attrs.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Normalized LFP metadata contract mismatch in {path}: {mismatches}")
    if "Phase" not in table or "Value" not in table:
        raise KeyError(f"Normalized LFP table lacks Phase or Value: {path}")
    pre = pd.to_numeric(table.loc[table["Phase"].eq("Pre"), "Value"], errors="coerce")
    finite_pre = pre[np.isfinite(pre)]
    if len(finite_pre) and not np.allclose(finite_pre, 0.0, atol=1e-12, rtol=0):
        raise ValueError(f"Finite normalized Pre values are not zero in {path}.")


def _read_lift(config: StructuralConnectivityConfig) -> pd.DataFrame:
    path = config.output_path("channel_field_weighted_lift")
    if not path.is_file():
        raise FileNotFoundError(f"Enriched channel Lift table does not exist: {path}")
    table = pd.read_csv(path)
    required = {
        "ID",
        "StimSide",
        "ConnectomeSource",
        "Channel",
        "Region",
        "Lift",
        "LiftStatus",
        "FieldWeightedLift",
        "FieldWeightedLiftStatus",
        *FIELD_EXCESS_COMPONENT_COLUMNS,
        "NAll",
        "NSeed",
        "NTarget",
        "NSeedTarget",
    }
    missing = sorted(required.difference(table.columns))
    if missing:
        raise KeyError(f"Enriched channel Lift table is missing columns: {missing}")
    key = ["ConnectomeSource", "ID", "StimSide", "Channel", "Region"]
    if table.duplicated(key).any():
        raise ValueError(
            "Channel Lift table has duplicate source/patient/channel/Region rows."
        )
    return derive_field_excess(table)


def _lift_metric_view(table: pd.DataFrame, lift_metric: str) -> pd.DataFrame:
    try:
        value_column, status_column = LIFT_METRIC_COLUMNS[lift_metric]
    except KeyError as error:
        raise ValueError(f"Unsupported Lift metric: {lift_metric}") from error
    view = table.copy()
    view["Lift"] = pd.to_numeric(view[value_column], errors="coerce")
    view["LiftStatus"] = view[status_column]
    return view


def _subject_side_map(config: StructuralConnectivityConfig) -> dict[str, str]:
    return {
        subject.subject_id: subject.stimulation_side for subject in config.subjects
    }


def _base_lfp_filter(
    table: pd.DataFrame,
    config: StructuralConnectivityConfig,
    *,
    bands: Iterable[str] | None = None,
) -> pd.Series:
    required = {
        "ID",
        "StimSide",
        "Polar",
        "Phase",
        "Band",
        "Lat",
        "IncludeAggregate",
        "NormalizationStatus",
        "Value",
        "ValueN",
    }
    missing = sorted(required.difference(table.columns))
    if missing:
        raise KeyError(f"Normalized LFP table is missing columns: {missing}")
    side_map = _subject_side_map(config)
    expected_side = table["ID"].map(side_map)
    outcomes = config["outcomes"]
    selected_bands = outcomes["bands"] if bands is None else list(bands)
    return (
        table["ID"].isin(side_map)
        & expected_side.notna()
        & table["StimSide"].eq(expected_side)
        & table["Polar"].isin(outcomes["polarities"])
        & table["Phase"].isin(outcomes["phases"])
        & table["Band"].isin(selected_bands)
        & table["Lat"].eq(outcomes["laterality"])
        & table["IncludeAggregate"].eq(True)
        & table["NormalizationStatus"].eq("normalized")
    )


def _eligibility_status(
    value: pd.Series,
    lift: pd.Series,
    lift_status: pd.Series,
) -> pd.Series:
    status = pd.Series("eligible", index=value.index, dtype="string")
    status.loc[~np.isfinite(pd.to_numeric(value, errors="coerce"))] = "nonfinite_outcome"
    status.loc[lift_status.isna()] = "missing_lift_row"
    not_ok = lift_status.notna() & ~lift_status.eq("ok")
    status.loc[not_ok] = lift_status.loc[not_ok].astype(str)
    nonfinite_lift = lift_status.eq("ok") & ~np.isfinite(
        pd.to_numeric(lift, errors="coerce")
    )
    status.loc[nonfinite_lift] = "nonfinite_lift"
    return status


def prepare_local_rows(
    config: StructuralConnectivityConfig,
    lift: pd.DataFrame,
) -> pd.DataFrame:
    """Attach every source-specific assigned-Region Lift to local outcomes."""

    path = config.path_value("local_lfp")
    table = _read_pickle(path)
    _validate_normalized_contract(
        table,
        path,
        domain="local",
        metric="periodic",
        transform="dB",
    )
    required = {"Domain", "Metric", "Region", "Channel"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise KeyError(f"Local LFP table is missing columns: {missing}")
    selected = table.loc[
        _base_lfp_filter(table, config)
        & table["Domain"].eq("local")
        & table["Metric"].eq("periodic")
        & table["Region"].isin(["STN", "SNr"])
    ].copy()
    key = ["ID", "StimSide", "Polar", "Phase", "Band", "Region", "Channel"]
    if selected.duplicated(key).any():
        raise ValueError("Local normalized table has duplicate analysis rows.")
    lift_columns = [
        "ID",
        "StimSide",
        "ConnectomeSource",
        "Channel",
        "Region",
        "Lift",
        "LiftStatus",
        "NAll",
        "NSeed",
        "NTarget",
        "NSeedTarget",
        "MembershipPath",
    ]
    merged = selected.merge(
        lift[lift_columns],
        on=["ID", "StimSide", "Channel", "Region"],
        how="left",
        validate="many_to_many",
    )
    expanded_key = ["ConnectomeSource", *key]
    if merged.duplicated(expanded_key).any():
        raise ValueError("Local source-expanded table has duplicate analysis rows.")
    merged["OutcomeScale"] = config["outcomes"]["local_scales"]["periodic"]
    merged["EligibilityStatus"] = _eligibility_status(
        merged["Value"], merged["Lift"], merged["LiftStatus"]
    )
    return merged.sort_values(
        ["ConnectomeSource", "ID", "Polar", "Region", "Phase", "Band", "Channel"],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)


def _endpoint_lift_columns(prefix: str) -> dict[str, str]:
    return {
        "Channel": f"{prefix}Channel",
        "Region": f"{prefix}Region",
        "Lift": f"{prefix}Lift",
        "LiftStatus": f"{prefix}LiftStatus",
        "NAll": f"{prefix}NAll",
        "NSeed": f"{prefix}NSeed",
        "NTarget": f"{prefix}NTarget",
        "NSeedTarget": f"{prefix}NSeedTarget",
        "MembershipPath": f"{prefix}MembershipPath",
    }


def prepare_connectivity_rows(
    config: StructuralConnectivityConfig,
    lift: pd.DataFrame,
) -> pd.DataFrame:
    """Attach common-pair STN and SNr endpoint Lift to all network outcomes."""

    frames: list[pd.DataFrame] = []
    pair_regions = config["outcomes"]["connectivity_pair_regions"]
    lift_base = lift[
        [
            "ID",
            "StimSide",
            "ConnectomeSource",
            "Channel",
            "Region",
            "Lift",
            "LiftStatus",
            "NAll",
            "NSeed",
            "NTarget",
            "NSeedTarget",
            "MembershipPath",
        ]
    ]
    for metric in config["outcomes"]["connectivity_metrics"]:
        path = config.path_value("connectivity_lfp_root") / metric / "mean-scalar.pkl"
        table = _read_pickle(path)
        transform = {
            "ciplv": "logit",
            "imcoh_abs": "fisherz",
            "psi": "none",
            "trgc": "none",
            "wpli": "logit",
        }[metric]
        _validate_normalized_contract(
            table,
            path,
            domain="connectivity",
            metric=metric,
            transform=transform,
        )
        required = {
            "Domain",
            "Metric",
            "PairRegion",
            "ChannelPair",
            "channel_a",
            "channel_b",
            "region_a",
            "region_b",
        }
        missing = sorted(required.difference(table.columns))
        if missing:
            raise KeyError(f"Connectivity LFP table is missing columns: {missing}")
        selected = table.loc[
            _base_lfp_filter(table, config)
            & table["Domain"].eq("connectivity")
            & table["Metric"].eq(metric)
            & table["PairRegion"].eq(pair_regions[metric])
        ].copy()
        key = [
            "ID",
            "StimSide",
            "Polar",
            "Metric",
            "PairRegion",
            "Phase",
            "Band",
            "channel_a",
            "channel_b",
        ]
        if selected.duplicated(key).any():
            raise ValueError(
                f"Connectivity normalized table has duplicate analysis rows: {metric}"
            )
        first = lift_base.rename(columns=_endpoint_lift_columns("A"))
        merged = selected.merge(
            first,
            left_on=["ID", "StimSide", "channel_a", "region_a"],
            right_on=["ID", "StimSide", "AChannel", "ARegion"],
            how="left",
            validate="many_to_many",
        )
        second = lift_base.rename(columns=_endpoint_lift_columns("B"))
        merged = merged.merge(
            second,
            left_on=[
                "ID",
                "StimSide",
                "ConnectomeSource",
                "channel_b",
                "region_b",
            ],
            right_on=[
                "ID",
                "StimSide",
                "ConnectomeSource",
                "BChannel",
                "BRegion",
            ],
            how="left",
            validate="many_to_one",
        )
        expanded_key = ["ConnectomeSource", *key]
        if merged.duplicated(expanded_key).any():
            raise ValueError(
                f"Connectivity source-expanded table has duplicate analysis rows: {metric}"
            )
        valid_regions = (
            merged[["region_a", "region_b"]]
            .apply(lambda row: set(row.astype(str)) == {"STN", "SNr"}, axis=1)
        )
        if not bool(valid_regions.all()):
            raise ValueError(f"Connectivity rows do not contain one STN and one SNr: {metric}")
        a_is_stn = merged["region_a"].eq("STN")
        for suffix in (
            "Lift",
            "LiftStatus",
            "NAll",
            "NSeed",
            "NTarget",
            "NSeedTarget",
            "MembershipPath",
            "Channel",
        ):
            merged[f"STN{suffix}"] = merged[f"A{suffix}"].where(
                a_is_stn, merged[f"B{suffix}"]
            )
            merged[f"SNr{suffix}"] = merged[f"B{suffix}"].where(
                a_is_stn, merged[f"A{suffix}"]
            )
        merged["RegionMeanLift"] = (
            pd.to_numeric(merged["STNLift"], errors="coerce")
            + pd.to_numeric(merged["SNrLift"], errors="coerce")
        ) / 2.0
        outcome_finite = np.isfinite(pd.to_numeric(merged["Value"], errors="coerce"))
        stn_ok = merged["STNLiftStatus"].eq("ok") & np.isfinite(
            pd.to_numeric(merged["STNLift"], errors="coerce")
        )
        snr_ok = merged["SNrLiftStatus"].eq("ok") & np.isfinite(
            pd.to_numeric(merged["SNrLift"], errors="coerce")
        )
        status = pd.Series("eligible", index=merged.index, dtype="string")
        status.loc[~outcome_finite] = "nonfinite_outcome"
        status.loc[outcome_finite & ~stn_ok] = "missing_or_nonfinite_stn_lift"
        status.loc[outcome_finite & stn_ok & ~snr_ok] = "missing_or_nonfinite_snr_lift"
        merged["EligibilityStatus"] = status
        merged["OutcomeScale"] = config["outcomes"]["connectivity_scales"][metric]
        frames.append(merged)
    output = pd.concat(frames, ignore_index=True)
    return output.sort_values(
        [
            "ConnectomeSource",
            "ID",
            "Polar",
            "Metric",
            "PairRegion",
            "Phase",
            "Band",
            "channel_a",
            "channel_b",
        ],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)


def _membership(values: Iterable[str]) -> str:
    return ";".join(sorted(str(value) for value in values))


def _counter_json(values: Iterable[str]) -> str:
    return json.dumps(dict(sorted(Counter(str(value) for value in values).items())))


def build_analysis_points(
    local_rows: pd.DataFrame,
    connectivity_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return long subject-aggregated and row-unaggregated X/Y tables."""

    subject_records: list[dict[str, Any]] = []
    row_records: list[dict[str, Any]] = []
    local = local_rows.loc[local_rows["EligibilityStatus"].eq("eligible")].copy()
    local_group = [
        "ConnectomeSource",
        "Polar",
        "Region",
        "Phase",
        "Band",
        "ID",
    ]
    for keys, group in local.groupby(local_group, sort=True, observed=True):
        source, polar, region, phase, band, subject_id = keys
        subject_records.append(
            {
                "AnalysisLevel": "subject_aggregated",
                "OutcomeDomain": "local",
                "Metric": "periodic",
                "OutcomeScale": group["OutcomeScale"].iloc[0],
                "ConnectomeSource": source,
                "Polar": polar,
                "Region": region,
                "PairRegion": "",
                "Phase": phase,
                "Band": band,
                "LiftPredictor": region,
                "ID": subject_id,
                "X": float(group["Lift"].mean()),
                "Y": float(group["Value"].mean()),
                "UnitCount": int(len(group)),
                "UnitMembership": _membership(group["Channel"]),
                "EndpointMultiplicity": "",
            }
        )
    for row in local.itertuples(index=False):
        row_records.append(
            {
                "AnalysisLevel": "row_unaggregated",
                "OutcomeDomain": "local",
                "Metric": "periodic",
                "OutcomeScale": row.OutcomeScale,
                "ConnectomeSource": row.ConnectomeSource,
                "Polar": row.Polar,
                "Region": row.Region,
                "PairRegion": "",
                "Phase": row.Phase,
                "Band": row.Band,
                "LiftPredictor": row.Region,
                "ID": row.ID,
                "RowID": str(row.Channel),
                "X": float(row.Lift),
                "Y": float(row.Value),
                "UnitMembership": str(row.Channel),
            }
        )

    network = connectivity_rows.loc[
        connectivity_rows["EligibilityStatus"].eq("eligible")
    ].copy()
    network_group = [
        "ConnectomeSource",
        "Polar",
        "Metric",
        "PairRegion",
        "Phase",
        "Band",
        "ID",
    ]
    for keys, group in network.groupby(network_group, sort=True, observed=True):
        source, polar, metric, pair_region, phase, band, subject_id = keys
        stn_mean = float(group["STNLift"].mean())
        snr_mean = float(group["SNrLift"].mean())
        y_mean = float(group["Value"].mean())
        pair_membership = _membership(
            f"{a}|{b}" for a, b in zip(group["channel_a"], group["channel_b"], strict=True)
        )
        multiplicity = json.dumps(
            {
                "STN": json.loads(_counter_json(group["STNChannel"])),
                "SNr": json.loads(_counter_json(group["SNrChannel"])),
            },
            sort_keys=True,
        )
        for predictor, x_value in (
            ("STN", stn_mean),
            ("SNr", snr_mean),
            ("RegionMean", (stn_mean + snr_mean) / 2.0),
        ):
            subject_records.append(
                {
                    "AnalysisLevel": "subject_aggregated",
                    "OutcomeDomain": "connectivity",
                    "Metric": metric,
                    "OutcomeScale": group["OutcomeScale"].iloc[0],
                    "ConnectomeSource": source,
                    "Polar": polar,
                    "Region": "",
                    "PairRegion": pair_region,
                    "Phase": phase,
                    "Band": band,
                    "LiftPredictor": predictor,
                    "ID": subject_id,
                    "X": x_value,
                    "Y": y_mean,
                    "UnitCount": int(len(group)),
                    "UnitMembership": pair_membership,
                    "EndpointMultiplicity": multiplicity,
                }
            )
    for row in network.itertuples(index=False):
        row_id = f"{row.channel_a}|{row.channel_b}"
        for predictor, x_value in (
            ("STN", row.STNLift),
            ("SNr", row.SNrLift),
            ("RegionMean", row.RegionMeanLift),
        ):
            row_records.append(
                {
                    "AnalysisLevel": "row_unaggregated",
                    "OutcomeDomain": "connectivity",
                    "Metric": row.Metric,
                    "OutcomeScale": row.OutcomeScale,
                    "ConnectomeSource": row.ConnectomeSource,
                    "Polar": row.Polar,
                    "Region": "",
                    "PairRegion": row.PairRegion,
                    "Phase": row.Phase,
                    "Band": row.Band,
                    "LiftPredictor": predictor,
                    "ID": row.ID,
                    "RowID": row_id,
                    "X": float(x_value),
                    "Y": float(row.Value),
                    "UnitMembership": row_id,
                }
            )
    subject = pd.DataFrame(subject_records)
    unaggregated = pd.DataFrame(row_records)
    return subject, unaggregated


def _result_row(
    points: pd.DataFrame,
    *,
    lift_metric: str,
    analysis_level: str,
    outcome_domain: str,
    metric: str,
    outcome_scale: str,
    source: str,
    polar: str,
    region: str,
    pair_region: str,
    phase: str,
    band: str,
    predictor: str,
    minimum_n: int,
) -> dict[str, Any]:
    selector = (
        points["AnalysisLevel"].eq(analysis_level)
        & points["OutcomeDomain"].eq(outcome_domain)
        & points["Metric"].eq(metric)
        & points["ConnectomeSource"].eq(source)
        & points["Polar"].eq(polar)
        & points["Region"].fillna("").eq(region)
        & points["PairRegion"].fillna("").eq(pair_region)
        & points["Phase"].eq(phase)
        & points["Band"].eq(band)
        & points["LiftPredictor"].eq(predictor)
    )
    cell = points.loc[selector].sort_values(
        ["ID"] + (["RowID"] if analysis_level == "row_unaggregated" else []),
        kind="mergesort",
    )
    finite = np.isfinite(pd.to_numeric(cell["X"], errors="coerce")) & np.isfinite(
        pd.to_numeric(cell["Y"], errors="coerce")
    )
    cell = cell.loc[finite]
    n_rows = int(len(cell))
    n_unique_subjects = int(cell["ID"].nunique()) if n_rows else 0
    n_subjects = n_rows if analysis_level == "subject_aggregated" else n_unique_subjects
    execution_n = n_subjects if analysis_level == "subject_aggregated" else n_rows
    status = "insufficient_n"
    rho = np.nan
    p_value = np.nan
    method = (
        "exhaustive_exact_spearman_two_sided"
        if analysis_level == "subject_aggregated"
        else "asymptotic_spearman_two_sided"
    )
    if execution_n >= minimum_n:
        x = cell["X"].to_numpy(dtype=float)
        y = cell["Y"].to_numpy(dtype=float)
        if np.unique(x).size < 2 or np.unique(y).size < 2:
            status = "undefined_constant"
        elif analysis_level == "subject_aggregated":
            rho_values, p_values, valid = _ordinary_group(x[:, None], y)
            if bool(valid[0]) and np.isfinite(rho_values[0]) and np.isfinite(p_values[0]):
                rho = float(rho_values[0])
                p_value = float(p_values[0])
                status = "ok"
            else:
                status = "undefined_constant"
        else:
            result = spearmanr(x, y, alternative="two-sided")
            if np.isfinite(result.statistic) and np.isfinite(result.pvalue):
                rho = float(result.statistic)
                p_value = float(result.pvalue)
                status = "ok"
            else:
                status = "undefined_constant"
    patient_ids = _membership(cell["ID"].drop_duplicates()) if n_rows else ""
    unit_membership = (
        ";".join(
            f"{row.ID}:{row.UnitMembership}"
            for row in cell.itertuples(index=False)
        )
        if n_rows
        else ""
    )
    return {
        "LiftMetric": lift_metric,
        "AnalysisLevel": analysis_level,
        "OutcomeDomain": outcome_domain,
        "Metric": metric,
        "OutcomeScale": outcome_scale,
        "ConnectomeSource": source,
        "Polar": polar,
        "Region": region,
        "PairRegion": pair_region,
        "Phase": phase,
        "Band": band,
        "LiftPredictor": predictor,
        "Method": method,
        "rho": rho,
        "P": p_value,
        "Status": status,
        "n_subjects": n_subjects,
        "n_rows": n_rows,
        "n_unique_subjects": n_unique_subjects,
        "PatientIDs": patient_ids,
        "UnitMembership": unit_membership,
    }


def compute_correlations(
    subject_points: pd.DataFrame,
    row_points: pd.DataFrame,
    config: StructuralConnectivityConfig,
    *,
    lift_metric: str = "BinaryLift",
) -> pd.DataFrame:
    """Calculate every planned cell and separate 21-test BH families."""

    results: list[dict[str, Any]] = []
    levels = config["inference"]["analysis_levels"]
    sources = [source.connectome_id for source in config.connectomes]
    polarities = config["outcomes"]["polarities"]
    phases = config["outcomes"]["phases"]
    bands = config["outcomes"]["bands"]
    minimum_n = int(config["inference"]["minimum_n"])
    points_by_level = {
        "subject_aggregated": subject_points,
        "row_unaggregated": row_points,
    }
    for level in levels:
        points = points_by_level[level]
        for source in sources:
            for polar in polarities:
                for region in ("STN", "SNr"):
                    for phase in phases:
                        for band in bands:
                            results.append(
                                _result_row(
                                    points,
                                    lift_metric=lift_metric,
                                    analysis_level=level,
                                    outcome_domain="local",
                                    metric="periodic",
                                    outcome_scale=config["outcomes"]["local_scales"][
                                        "periodic"
                                    ],
                                    source=source,
                                    polar=polar,
                                    region=region,
                                    pair_region="",
                                    phase=phase,
                                    band=band,
                                    predictor=region,
                                    minimum_n=minimum_n,
                                )
                            )
                for metric in config["outcomes"]["connectivity_metrics"]:
                    pair_region = config["outcomes"]["connectivity_pair_regions"][
                        metric
                    ]
                    scale = config["outcomes"]["connectivity_scales"][metric]
                    for predictor in config["inference"]["connectivity_lift_predictors"]:
                        for phase in phases:
                            for band in bands:
                                results.append(
                                    _result_row(
                                        points,
                                        lift_metric=lift_metric,
                                        analysis_level=level,
                                        outcome_domain="connectivity",
                                        metric=metric,
                                        outcome_scale=scale,
                                        source=source,
                                        polar=polar,
                                        region="",
                                        pair_region=pair_region,
                                        phase=phase,
                                        band=band,
                                        predictor=predictor,
                                        minimum_n=minimum_n,
                                    )
                                )
    output = pd.DataFrame(results)
    output["Q"] = np.nan
    output["m_planned"] = int(config["inference"]["bh_planned_tests"])
    output["m_tested"] = 0
    output["FamilyKey"] = ""
    local_family = [
        "AnalysisLevel",
        "LiftMetric",
        "ConnectomeSource",
        "Polar",
        "OutcomeDomain",
        "Region",
    ]
    network_family = [
        "AnalysisLevel",
        "LiftMetric",
        "ConnectomeSource",
        "Polar",
        "OutcomeDomain",
        "Metric",
        "PairRegion",
        "LiftPredictor",
    ]
    for domain, family_columns in (
        ("local", local_family),
        ("connectivity", network_family),
    ):
        subset = output.loc[output["OutcomeDomain"].eq(domain)]
        for keys, group in subset.groupby(family_columns, sort=True, observed=True):
            if len(group) != 21:
                raise ValueError(
                    f"BH family does not contain 21 planned rows: {keys} has {len(group)}"
                )
            finite = group["P"].notna() & np.isfinite(group["P"])
            tested = int(finite.sum())
            q_values = _benjamini_hochberg(group["P"])
            family_key = json.dumps(
                dict(zip(family_columns, keys, strict=True)),
                ensure_ascii=False,
                sort_keys=True,
            )
            output.loc[group.index, "Q"] = q_values
            output.loc[group.index, "m_tested"] = tested
            output.loc[group.index, "FamilyKey"] = family_key
    subject = output["AnalysisLevel"].eq("subject_aggregated")
    output["p_exact"] = output["P"].where(subject)
    output["p_rowwise"] = output["P"].where(~subject)
    output["q_exact"] = output["Q"].where(subject)
    output["q_rowwise"] = output["Q"].where(~subject)
    alpha = float(config["inference"]["alpha"])
    output["p_lt_0_05"] = pd.array(
        [value < alpha if np.isfinite(value) else pd.NA for value in output["P"]],
        dtype="boolean",
    )
    output["q_lt_0_05"] = pd.array(
        [value < alpha if np.isfinite(value) else pd.NA for value in output["Q"]],
        dtype="boolean",
    )
    return output.sort_values(
        [
            "AnalysisLevel",
            "LiftMetric",
            "OutcomeDomain",
            "ConnectomeSource",
            "Polar",
            "Metric",
            "Region",
            "PairRegion",
            "LiftPredictor",
            "Phase",
            "Band",
        ],
        kind="mergesort",
    ).reset_index(drop=True)


def _planned_outputs(config: StructuralConnectivityConfig) -> list[Path]:
    return [
        config.output_path("local_rows"),
        config.output_path("connectivity_rows"),
        config.output_path("subject_aggregates"),
        config.output_path("correlations"),
        config.output_path("inputs_manifest"),
        config.output_path("outputs_manifest"),
    ]


def run_statistics(
    config: StructuralConnectivityConfig,
    *,
    dry_run: bool,
    overwrite: bool,
) -> dict[str, Any]:
    """Prepare rows, compute both analysis levels, and write neutral outputs."""

    existing = [path for path in _planned_outputs(config) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Statistics output exists; pass --overwrite to replace it: {existing[0]}"
        )
    enriched_lift = _read_lift(config)
    lift_metrics = list(config["inference"]["lift_metrics"])
    local_frames: list[pd.DataFrame] = []
    network_frames: list[pd.DataFrame] = []
    subject_frames: list[pd.DataFrame] = []
    row_frames: list[pd.DataFrame] = []
    for lift_metric in lift_metrics:
        lift = _lift_metric_view(enriched_lift, lift_metric)
        local_metric = prepare_local_rows(config, lift)
        network_metric = prepare_connectivity_rows(config, lift)
        local_metric.insert(0, "LiftMetric", lift_metric)
        network_metric.insert(0, "LiftMetric", lift_metric)
        subject_metric, row_metric = build_analysis_points(
            local_metric,
            network_metric,
        )
        subject_metric.insert(1, "LiftMetric", lift_metric)
        row_metric.insert(1, "LiftMetric", lift_metric)
        local_frames.append(local_metric)
        network_frames.append(network_metric)
        subject_frames.append(subject_metric)
        row_frames.append(row_metric)
    local = pd.concat(local_frames, ignore_index=True)
    network = pd.concat(network_frames, ignore_index=True)
    subject_points = pd.concat(subject_frames, ignore_index=True)
    row_points = pd.concat(row_frames, ignore_index=True)
    summary = {
        "status": "ready" if dry_run else "complete",
        "channel_lift_rows": int(len(enriched_lift)),
        "lift_metrics": lift_metrics,
        "local_source_rows": int(len(local)),
        "local_eligible_rows": int(local["EligibilityStatus"].eq("eligible").sum()),
        "connectivity_source_rows": int(len(network)),
        "connectivity_eligible_rows": int(
            network["EligibilityStatus"].eq("eligible").sum()
        ),
        "subject_aggregate_rows": int(len(subject_points)),
        "row_analysis_points": int(len(row_points)),
        "writes": 0 if dry_run else 6,
    }
    if dry_run:
        return summary
    correlations = pd.concat(
        [
            compute_correlations(
                subject_points.loc[subject_points["LiftMetric"].eq(lift_metric)],
                row_points.loc[row_points["LiftMetric"].eq(lift_metric)],
                config,
                lift_metric=lift_metric,
            )
            for lift_metric in lift_metrics
        ],
        ignore_index=True,
    )
    atomic_csv(local, config.output_path("local_rows"), overwrite=overwrite)
    atomic_csv(network, config.output_path("connectivity_rows"), overwrite=overwrite)
    atomic_csv(
        subject_points,
        config.output_path("subject_aggregates"),
        overwrite=overwrite,
    )
    atomic_csv(correlations, config.output_path("correlations"), overwrite=overwrite)
    input_paths = [
        config.output_path("channel_field_weighted_lift"),
        config.path_value("local_lfp"),
        *[
            config.path_value("connectivity_lfp_root") / metric / "mean-scalar.pkl"
            for metric in config["outcomes"]["connectivity_metrics"]
        ],
    ]
    inputs_manifest = pd.DataFrame(
        {
            "AnalysisID": config.analysis_id,
            "Path": [str(path) for path in input_paths],
            "Exists": [path.is_file() for path in input_paths],
        }
    )
    atomic_csv(
        inputs_manifest,
        config.output_path("inputs_manifest"),
        overwrite=overwrite,
    )
    output_tables = [
        ("local_rows", config.output_path("local_rows"), local),
        ("connectivity_rows", config.output_path("connectivity_rows"), network),
        ("subject_aggregates", config.output_path("subject_aggregates"), subject_points),
        ("correlations", config.output_path("correlations"), correlations),
        ("inputs_manifest", config.output_path("inputs_manifest"), inputs_manifest),
    ]
    outputs_manifest = pd.DataFrame(
        [
            {
                "AnalysisID": config.analysis_id,
                "OutputType": output_type,
                "Path": str(path),
                "Rows": len(table),
                "Columns": len(table.columns),
                "Status": "complete",
            }
            for output_type, path, table in output_tables
        ]
    )
    atomic_csv(
        outputs_manifest,
        config.output_path("outputs_manifest"),
        overwrite=overwrite,
    )
    summary["correlation_rows"] = int(len(correlations))
    summary["estimable_correlations"] = int(correlations["Status"].eq("ok").sum())
    return summary
