"""Compute ordinary and baseline-adjusted rank correlations."""

from __future__ import annotations

from functools import lru_cache
from itertools import permutations
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .config import ClinicalCorrelationConfig
from .data import ClinicalCorrelationInputs


@lru_cache(maxsize=None)
def _permutation_indices(n: int) -> np.ndarray:
    return np.asarray(list(permutations(range(n))), dtype=np.int16)


def _mask_key(mask: np.ndarray) -> bytes:
    return np.packbits(mask.astype(np.uint8)).tobytes()


def _standardize_columns(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = values - np.mean(values, axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=0)
    valid = norms > np.finfo(float).eps
    standardized = np.zeros_like(centered, dtype=float)
    standardized[:, valid] = centered[:, valid] / norms[valid]
    return standardized, valid


def _residual_tolerance(values: np.ndarray) -> np.ndarray:
    """Return a scale-aware zero tolerance for projected ranked values."""

    return np.sqrt(np.finfo(float).eps) * np.linalg.norm(values, axis=0)


def rank_effect_coordinates(
    values: np.ndarray, residualizer: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return unit-length rank columns, optionally covariate-adjusted, without p values."""

    ranked = np.asarray(rankdata(values, axis=0, method="average"), dtype=float)
    if residualizer is None:
        return _standardize_columns(ranked)
    residuals = residualizer @ ranked
    norms = np.linalg.norm(residuals, axis=0)
    valid = norms > _residual_tolerance(ranked)
    standardized = np.zeros_like(residuals)
    standardized[:, valid] = residuals[:, valid] / norms[valid]
    return standardized, valid


def _ordinary_group_details(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y_rank = rankdata(y, method="average").astype(float)
    x_standard, x_valid = rank_effect_coordinates(x)
    x_plot = x_standard * np.sqrt(len(y) - 1)
    y_centered = y_rank - np.mean(y_rank)
    y_norm = np.linalg.norm(y_centered)
    if y_norm <= np.finfo(float).eps:
        return (
            np.full(x.shape[1], np.nan),
            np.full(x.shape[1], np.nan),
            np.zeros(x.shape[1], dtype=bool),
            x_plot,
            np.full(len(y), np.nan),
        )
    y_standard = y_centered / y_norm
    y_plot = y_standard * np.sqrt(len(y) - 1)
    observed = x_standard.T @ y_standard
    permuted_y = y_standard[_permutation_indices(len(y))]
    permuted = x_standard.T @ permuted_y.T
    p_values = np.mean(
        np.abs(permuted) >= np.abs(observed[:, None]) - 1e-12,
        axis=1,
    )
    observed[~x_valid] = np.nan
    p_values[~x_valid] = np.nan
    return observed, p_values, x_valid, x_plot, y_plot


def _ordinary_group(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rho, p_values, valid, _, _ = _ordinary_group_details(x, y)
    return rho, p_values, valid


def _partial_group_details(
    x: np.ndarray, y: np.ndarray, baseline: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    baseline_rank = rankdata(baseline, method="average").astype(float)
    design = np.column_stack((np.ones(len(baseline_rank)), baseline_rank))
    residualizer = np.eye(len(baseline_rank)) - design @ np.linalg.pinv(design)

    x_standard, x_valid = rank_effect_coordinates(x, residualizer)
    x_plot = x_standard * np.sqrt(len(y) - 1)

    y_rank = rankdata(y, method="average").astype(float)
    y_residual = residualizer @ y_rank
    y_norm = np.linalg.norm(y_residual)
    if y_norm <= float(_residual_tolerance(y_rank)):
        return (
            np.full(x.shape[1], np.nan),
            np.full(x.shape[1], np.nan),
            np.zeros(x.shape[1], dtype=bool),
            x_plot,
            np.full(len(y), np.nan),
        )
    y_standard = y_residual / y_norm
    y_plot = y_standard * np.sqrt(len(y) - 1)
    observed = x_standard.T @ y_standard

    permuted_residuals = y_residual[_permutation_indices(len(y))] @ residualizer
    permutation_norms = np.linalg.norm(permuted_residuals, axis=1)
    permutation_tolerance = np.sqrt(np.finfo(float).eps) * y_norm
    valid_permutations = permutation_norms > permutation_tolerance
    permuted_standard = np.zeros_like(permuted_residuals)
    permuted_standard[valid_permutations] = (
        permuted_residuals[valid_permutations]
        / permutation_norms[valid_permutations, None]
    )
    permuted = x_standard.T @ permuted_standard.T
    p_values = np.mean(
        np.abs(permuted) >= np.abs(observed[:, None]) - 1e-12,
        axis=1,
    )
    observed[~x_valid] = np.nan
    p_values[~x_valid] = np.nan
    return observed, p_values, x_valid, x_plot, y_plot


def _partial_group(
    x: np.ndarray, y: np.ndarray, baseline: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rho, p_values, valid, _, _ = _partial_group_details(x, y, baseline)
    return rho, p_values, valid


def _compute_endpoint_correlations(
    predictors: np.ndarray,
    outcome: np.ndarray,
    baseline: np.ndarray,
    subject_ids: np.ndarray,
    *,
    method: str,
    minimum_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_predictors = predictors.shape[1]
    result = pd.DataFrame(
        {
            "n": np.zeros(n_predictors, dtype=int),
            "SubjectIDs": [""] * n_predictors,
            "rho": np.full(n_predictors, np.nan),
            "p_raw": np.full(n_predictors, np.nan),
            "Status": ["insufficient_n"] * n_predictors,
            "Message": [f"Fewer than {minimum_n} complete patients."] * n_predictors,
        }
    )
    base_valid = np.isfinite(outcome)
    if method == "partial_spearman":
        base_valid &= np.isfinite(baseline)

    grouped: dict[bytes, tuple[np.ndarray, list[int]]] = {}
    point_frames: list[pd.DataFrame] = []
    for column in range(n_predictors):
        mask = base_valid & np.isfinite(predictors[:, column])
        key = _mask_key(mask)
        if key not in grouped:
            grouped[key] = (mask, [])
        grouped[key][1].append(column)

    for mask, columns in grouped.values():
        column_array = np.asarray(columns, dtype=int)
        n = int(mask.sum())
        joined_ids = ";".join(subject_ids[mask].astype(str))
        result.loc[column_array, "n"] = n
        result.loc[column_array, "SubjectIDs"] = joined_ids
        if n < minimum_n:
            continue
        x = predictors[np.ix_(mask, column_array)]
        y = outcome[mask]
        if method == "spearman":
            rho, p_values, valid, x_plot, y_plot = _ordinary_group_details(x, y)
            transform = "rank_z"
        else:
            rho, p_values, valid, x_plot, y_plot = _partial_group_details(
                x, y, baseline[mask]
            )
            transform = "baseline_adjusted_rank_residual_z"
        result.loc[column_array, "rho"] = rho
        result.loc[column_array, "p_raw"] = p_values
        valid_columns = column_array[valid]
        constant_columns = column_array[~valid]
        result.loc[valid_columns, "Status"] = "ok"
        result.loc[valid_columns, "Message"] = ""
        result.loc[constant_columns, "Status"] = "constant_input"
        result.loc[constant_columns, "Message"] = (
            "Ranked input or baseline-adjusted residual is constant."
        )
        valid_local = np.flatnonzero(valid)
        if len(valid_local) == 0:
            continue
        valid_columns = column_array[valid_local]
        repeated = len(valid_columns)
        point_frames.append(
            pd.DataFrame(
                {
                    "PredictorIndex": np.repeat(valid_columns, n),
                    "ID": np.tile(subject_ids[mask].astype(str), repeated),
                    "XRaw": x[:, valid_local].T.reshape(-1),
                    "YRaw": np.tile(y, repeated),
                    "Baseline": np.tile(baseline[mask], repeated),
                    "XPlot": x_plot[:, valid_local].T.reshape(-1),
                    "YPlot": np.tile(y_plot, repeated),
                    "Transform": transform,
                }
            )
        )
    point_columns = [
        "PredictorIndex",
        "ID",
        "XRaw",
        "YRaw",
        "Baseline",
        "XPlot",
        "YPlot",
        "Transform",
    ]
    points = (
        pd.concat(point_frames, ignore_index=True)
        if point_frames
        else pd.DataFrame(columns=point_columns)
    )
    return result, points


def _holm_adjust(values: pd.Series) -> pd.Series:
    adjusted = pd.Series(np.nan, index=values.index, dtype=float)
    finite = values.notna() & np.isfinite(values)
    if not finite.any():
        return adjusted
    p_values = values.loc[finite].to_numpy(dtype=float)
    order = np.argsort(p_values, kind="mergesort")
    ranked = p_values[order]
    count = len(ranked)
    adjusted_ranked = ranked * np.arange(count, 0, -1)
    adjusted_ranked = np.maximum.accumulate(adjusted_ranked)
    adjusted_ranked = np.clip(adjusted_ranked, 0.0, 1.0)
    adjusted_values = np.empty(count, dtype=float)
    adjusted_values[order] = adjusted_ranked
    adjusted.loc[finite] = adjusted_values
    return adjusted


def _benjamini_hochberg(values: pd.Series) -> pd.Series:
    """Return BH-adjusted values while retaining missing planned tests."""

    adjusted = pd.Series(np.nan, index=values.index, dtype=float)
    numeric = pd.to_numeric(values, errors="coerce")
    finite = np.isfinite(numeric)
    if not finite.any():
        return adjusted
    p_values = numeric.loc[finite].to_numpy(dtype=float)
    order = np.argsort(p_values, kind="mergesort")
    ranked = p_values[order]
    scaled = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted_ranked = np.minimum.accumulate(scaled[::-1])[::-1]
    adjusted_values = np.empty(len(ranked), dtype=float)
    adjusted_values[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    adjusted.loc[finite] = adjusted_values
    return adjusted


def _family_id(row: pd.Series, fields: list[str]) -> str:
    return "_".join(
        f"{field.lower()}-{str(row[field]).replace('_', '-').replace(' ', '-')}"
        for field in fields
    )


def compute_correlation_outputs(
    inputs: ClinicalCorrelationInputs,
    config: ClinicalCorrelationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute registered correlations and their patient-level plot coordinates."""

    metadata = inputs.predictor_metadata.reset_index(drop=True).copy()
    predictor_values = inputs.predictor_values.pivot(
        index="ID", columns="PredictorID", values="Value"
    ).reindex(columns=metadata["PredictorID"])
    clinical_ids = set(inputs.clinical_endpoints["ID"])
    subject_ids = np.asarray(sorted(clinical_ids.intersection(predictor_values.index)))
    if len(subject_ids) == 0:
        raise ValueError("No patients remain after clinical and LFP alignment.")
    predictor_matrix = predictor_values.reindex(subject_ids).to_numpy(dtype=float)

    endpoint_specs = [
        {
            **item,
            "EndpointOrder": order,
        }
        for order, item in enumerate(config["endpoints"])
    ]
    rows: list[pd.DataFrame] = []
    point_rows: list[pd.DataFrame] = []
    for scale in config["scales"]:
        scale_rows = inputs.clinical_endpoints.loc[
            inputs.clinical_endpoints["Scale"].eq(scale["value"])
        ].set_index("ID")
        scale_rows = scale_rows.reindex(subject_ids)
        baseline = scale_rows["Baseline"].to_numpy(dtype=float)
        scale_order = int(scale_rows["ScaleOrder"].dropna().iloc[0])
        for endpoint in endpoint_specs:
            outcome = scale_rows[endpoint["outcome_column"]].to_numpy(dtype=float)
            estimates, endpoint_points = _compute_endpoint_correlations(
                predictor_matrix,
                outcome,
                baseline,
                subject_ids,
                method=endpoint["method"],
                minimum_n=int(endpoint["minimum_n"]),
            )
            frame = pd.concat([metadata.reset_index(drop=True), estimates], axis=1)
            frame.insert(0, "AnalysisID", config.analysis_id)
            frame.insert(1, "EndpointID", endpoint["id"])
            frame.insert(2, "EndpointLabel", endpoint["label"])
            frame.insert(3, "EndpointOrder", endpoint["EndpointOrder"])
            frame.insert(4, "Scale", scale["value"])
            frame.insert(5, "ScaleLabel", scale["label"])
            frame.insert(6, "ScaleDirection", scale["direction"])
            frame.insert(7, "ScaleOrder", scale_order)
            frame["Method"] = endpoint["method"]
            frame["Covariates"] = ";".join(endpoint["covariates"]) or "none"
            frame["MinimumN"] = int(endpoint["minimum_n"])
            rows.append(frame)
            if not endpoint_points.empty:
                point_frame = endpoint_points.copy()
                predictor_index = point_frame.pop("PredictorIndex").astype(int)
                point_frame.insert(
                    0,
                    "PredictorID",
                    metadata.loc[predictor_index, "PredictorID"].to_numpy(),
                )
                point_frame.insert(0, "Method", endpoint["method"])
                point_frame.insert(0, "Scale", scale["value"])
                point_frame.insert(0, "EndpointID", endpoint["id"])
                point_frame.insert(0, "AnalysisID", config.analysis_id)
                point_frame.insert(
                    1,
                    "CorrelationID",
                    endpoint["id"]
                    + "_scale-"
                    + str(scale_order).zfill(2)
                    + "_"
                    + point_frame["PredictorID"].astype(str),
                )
                point_rows.append(point_frame)
    result = pd.concat(rows, ignore_index=True)
    family_fields = list(config["inference"]["correction"]["family_fields"])
    result["CorrectionFamily"] = result.apply(_family_id, axis=1, fields=family_fields)
    grouped = result.groupby(family_fields, observed=True, sort=False, dropna=False)
    result["p_holm"] = grouped["p_raw"].transform(_holm_adjust)
    result["CorrectionFamilySize"] = grouped["p_raw"].transform(
        lambda values: int(values.notna().sum())
    )
    result["SignificanceBasis"] = config["inference"]["significance_basis"]
    result["CorrelationID"] = (
        result["EndpointID"].astype(str)
        + "_scale-"
        + result["ScaleOrder"].astype(int).astype(str).str.zfill(2)
        + "_"
        + result["PredictorID"].astype(str)
    )
    if result["CorrelationID"].duplicated().any():
        raise ValueError("Correlation result contains duplicate identities.")
    predictor_definition_columns = [
        "FromPhase",
        "ToPhase",
        "PhaseContrast",
        "PredictorDefinition",
    ]
    columns = [
        "AnalysisID",
        "CorrelationID",
        "EndpointID",
        "EndpointLabel",
        "EndpointOrder",
        "Scale",
        "ScaleLabel",
        "ScaleDirection",
        "ScaleOrder",
        "Method",
        "Covariates",
        "MinimumN",
        "PredictorID",
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
        "Phase",
        *[
            column
            for column in predictor_definition_columns
            if column in result.columns
        ],
        "Lat",
        "n",
        "SubjectIDs",
        "rho",
        "p_raw",
        "p_holm",
        "CorrectionFamily",
        "CorrectionFamilySize",
        "SignificanceBasis",
        "Status",
        "Message",
    ]
    correlations = (
        result.loc[:, columns]
        .sort_values(
            [
                "EndpointOrder",
                "Domain",
                "DisplayRegion",
                "Polar",
                "Phase",
                "Lat",
                "ScaleOrder",
                "FeatureOrder",
            ]
        )
        .reset_index(drop=True)
    )
    fit_points = pd.concat(point_rows, ignore_index=True)
    fit_points = fit_points.loc[
        :,
        [
            "AnalysisID",
            "CorrelationID",
            "EndpointID",
            "Scale",
            "PredictorID",
            "ID",
            "Method",
            "Transform",
            "XRaw",
            "YRaw",
            "Baseline",
            "XPlot",
            "YPlot",
        ],
    ].sort_values(["CorrelationID", "ID"])
    if fit_points.duplicated(["CorrelationID", "ID"]).any():
        raise ValueError("Fit-point output contains duplicate patient coordinates.")
    expected_counts = correlations.loc[
        correlations["Status"].eq("ok"), ["CorrelationID", "n"]
    ].set_index("CorrelationID")["n"]
    observed_counts = fit_points.groupby("CorrelationID", observed=True).size()
    if not observed_counts.equals(expected_counts.reindex(observed_counts.index)):
        raise ValueError("Fit-point patient counts do not match correlation results.")
    if set(observed_counts.index) != set(expected_counts.index):
        raise ValueError(
            "Fit-point identities do not cover all estimable correlations."
        )
    return correlations, fit_points.reset_index(drop=True)


def compute_correlation_table(
    inputs: ClinicalCorrelationInputs,
    config: ClinicalCorrelationConfig,
) -> pd.DataFrame:
    """Compute every registered correlation and Holm-adjusted P value."""

    correlations, _ = compute_correlation_outputs(inputs, config)
    return correlations
