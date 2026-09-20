"""PCA-assisted robust MCD outlier decisions."""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from lfptensorpipe.stats.preproc.transform import transform_df
from scipy.stats import chi2
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA

from ..feature_outputs import configured_transform_mode, iter_feature_output_specs
from ..contracts import FeatureOutputSpec, RunState
from ..identity import (
    casefold_identity,
    first_group_labels,
    groupby_casefolded,
    identity_equal,
)
from ..io import atomic_pickle, read_stage_table, stage_path
from ..provenance import record_feature_output
from ..qc import finite_count


TOL_TOTAL_VAR = 1e-12
TOL_COV_REG = 1e-6


def strict_feature_matrix(values: Sequence[Any]) -> np.ndarray:
    """Flatten identically labelled nested values for MCD detection."""
    non_scalar = [
        value for value in values if isinstance(value, (pd.Series, pd.DataFrame))
    ]
    if not non_scalar:
        return np.asarray(
            [float(value) if finite_count(value) else np.nan for value in values],
            dtype=float,
        ).reshape(-1, 1)
    template = non_scalar[0]
    rows: list[np.ndarray] = []
    for value in values:
        if isinstance(template, pd.Series):
            if isinstance(value, pd.Series):
                if not value.index.equals(template.index):
                    raise ValueError("Series axes differ within an outlier group.")
                rows.append(value.to_numpy(dtype=float, copy=False))
            elif finite_count(value) == 0:
                rows.append(np.full(template.size, np.nan))
            else:
                raise TypeError("Mixed scalar and Series values in an outlier group.")
        elif isinstance(value, pd.DataFrame):
            if not value.index.equals(template.index) or not value.columns.equals(
                template.columns
            ):
                raise ValueError("DataFrame axes differ within an outlier group.")
            rows.append(value.to_numpy(dtype=float, copy=False).ravel())
        elif finite_count(value) == 0:
            rows.append(np.full(template.size, np.nan))
        else:
            raise TypeError("Mixed scalar and DataFrame values in an outlier group.")
    return np.vstack(rows)


def _impute_and_scale(matrix: np.ndarray, robust_scale: bool) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanmedian(matrix, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    output = matrix.copy()
    missing = np.where(~np.isfinite(output))
    if missing[0].size:
        output[missing] = np.take(median, missing[1])
    if robust_scale:
        q1 = np.nanpercentile(output, 25, axis=0)
        q3 = np.nanpercentile(output, 75, axis=0)
        scale = np.where((q3 - q1) == 0, 1.0, q3 - q1)
        output = (output - np.nanmedian(output, axis=0)) / scale
    return output


def detect_mcd(
    matrix: np.ndarray, policy: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, str]:
    """Run the configured MyLFP-style PCA-assisted MCD detector."""
    n_samples, n_features = matrix.shape
    keep = np.ones(n_samples, dtype=bool)
    scores = np.full(n_samples, np.nan)
    if n_samples == 0 or n_features == 0 or not np.isfinite(matrix).any():
        return keep, scores, "nonfinite"
    if n_samples < 2:
        return keep, scores, "insufficient_samples"
    scaled = _impute_and_scale(matrix, bool(policy["robust_scale"]))
    total_var = float(np.nanvar(scaled, axis=0, ddof=1).sum())
    if not np.isfinite(total_var) or total_var <= TOL_TOTAL_VAR:
        return keep, scores, "constant"

    use_pca = bool(
        policy["pca"]
        and (
            n_samples < int(policy["min_n"])
            or n_samples < float(policy["min_n_over_p"]) * n_features
        )
    )
    if use_pca:
        component_cap = max(
            1, min(n_samples - 1, n_features, int(policy["pca_max_components"]))
        )
        pca = PCA(
            n_components=component_cap,
            svd_solver="full",
            whiten=True,
            random_state=int(policy["random_state"]),
        )
        transformed = pca.fit_transform(scaled)
        explained_target = policy.get("pca_explained_var")
        if explained_target is not None and 0 < float(explained_target) < 1:
            cumulative = np.cumsum(pca.explained_variance_ratio_)
            components = int(np.searchsorted(cumulative, float(explained_target)) + 1)
            components = max(1, min(components, component_cap))
            reduced = transformed[:, :components]
        else:
            reduced = transformed
            components = reduced.shape[1]
    else:
        reduced = scaled
        components = reduced.shape[1]
    if n_samples <= 2 or n_samples <= components:
        return keep, scores, "insufficient_rank"

    h_min = int(math.ceil((n_samples + components + 1) / 2.0))
    minimum_support = min(1.0, max(0.5, h_min / float(n_samples)))
    support = min(1.0, max(float(policy["support_fraction"]), minimum_support))

    def fit_mcd(support_fraction: float) -> tuple[np.ndarray, bool]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", category=RuntimeWarning)
            model = MinCovDet(
                support_fraction=support_fraction,
                random_state=int(policy["random_state"]),
            ).fit(reduced)
            distances = model.mahalanobis(reduced)
            determinant_warning = any(
                "Determinant has increased" in str(item.message) for item in caught
            )
        return distances, determinant_warning

    distances: np.ndarray | None = None
    warned = False
    try:
        distances, warned = fit_mcd(support)
    except (ValueError, np.linalg.LinAlgError):
        warned = True
    if distances is None or warned:
        try:
            distances, _ = fit_mcd(min(1.0, max(support, 0.99)))
        except (ValueError, np.linalg.LinAlgError):
            distances = None
    method = "mcd"
    if distances is None:
        center = np.nanmean(reduced, axis=0)
        differences = reduced - center
        covariance = np.atleast_2d(np.cov(reduced, rowvar=False))
        trace = float(np.trace(covariance))
        ridge = TOL_COV_REG * (
            trace / max(1, covariance.shape[0])
            if np.isfinite(trace) and trace > 0
            else 1.0
        )
        inverse = np.linalg.pinv(covariance + ridge * np.eye(covariance.shape[0]))
        distances = np.einsum("ij,jk,ik->i", differences, inverse, differences)
        method = "empirical_covariance"

    threshold = float(chi2.ppf(float(policy["alpha"]), df=components))
    keep = distances <= threshold
    if bool(policy["relax_if_all_rejected"]) and not keep.any():
        relaxed = float(chi2.ppf(float(policy["relax_alpha"]), df=components))
        keep = distances <= relaxed
        method = f"{method}_relaxed"
    return keep, distances, method


def detect_mad(
    matrix: np.ndarray, policy: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, str]:
    """Apply the configured absolute robust-z rule to one scalar group."""
    n_samples, n_features = matrix.shape
    if n_features != 1:
        raise ValueError("MAD outlier detection requires scalar feature values.")
    keep = np.ones(n_samples, dtype=bool)
    scores = np.full(n_samples, np.nan)
    if n_samples == 0:
        return keep, scores, "nonfinite"
    values = matrix[:, 0]
    if not np.isfinite(values).all():
        raise ValueError("MAD outlier detection received nonfinite scalar values.")
    center = float(np.median(values))
    absolute_deviation = np.abs(values - center)
    mad = float(np.median(absolute_deviation))
    if mad == 0.0:
        scores = np.where(absolute_deviation == 0.0, 0.0, np.inf)
    else:
        scores = absolute_deviation / (float(policy["mad_scale"]) * mad)
    keep = scores <= float(policy["robust_z_threshold"])
    return keep, scores, "mad"


def mark_outliers(
    merged: pd.DataFrame,
    transformed_preview: pd.DataFrame,
    spec: FeatureOutputSpec,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    """Mark source rows while preserving every merged row."""
    out = merged.copy()
    out["IsOutlier"] = False
    out["IncludeAggregate"] = True
    out["OutlierReason"] = ""
    detail: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    policy = config["outlier_filter"]
    if spec.representation not in policy["enabled_representations"]:
        for source_row in out["SourceRow"]:
            detail.append(
                {
                    "Metric": spec.metric,
                    "FeatureOutput": spec.feature_output,
                    "SourceRow": source_row,
                    "IsOutlier": False,
                    "IncludeAggregate": True,
                    "OutlierReason": "",
                    "Score": np.nan,
                    "Method": "disabled",
                }
            )
        return out, detail, groups

    group_columns = [
        column for column in policy["group_candidates"] if column in transformed_preview
    ]
    grouped = (
        groupby_casefolded(transformed_preview, group_columns)
        if group_columns
        else [((), transformed_preview)]
    )
    for _, group in grouped:
        matrix = strict_feature_matrix(group["Value"].tolist())
        finite_rows = np.isfinite(matrix).any(axis=1)
        keep = np.ones(len(group), dtype=bool)
        scores = np.full(len(group), np.nan)
        if finite_rows.any():
            detector = detect_mad if policy["mode"] == "mad" else detect_mcd
            finite_keep, finite_scores, method = detector(matrix[finite_rows], policy)
            keep[finite_rows] = finite_keep
            scores[finite_rows] = finite_scores
        else:
            method = "nonfinite"
        indices = group.index.to_numpy()
        outlier_positions = finite_rows & ~keep
        nonfinite_positions = ~finite_rows
        outliers = indices[outlier_positions]
        nonfinite = indices[nonfinite_positions]
        outlier_reason = "mad_robust_z" if policy["mode"] == "mad" else "mcd_chi2"
        out.loc[outliers, "IsOutlier"] = True
        out.loc[outliers, "IncludeAggregate"] = False
        out.loc[outliers, "OutlierReason"] = outlier_reason
        out.loc[nonfinite, "IncludeAggregate"] = False
        out.loc[nonfinite, "OutlierReason"] = "nonfinite_value"
        group_labels = first_group_labels(group, group_columns)
        group_record = {
            **group_labels,
            "Metric": spec.metric,
            "FeatureOutput": spec.feature_output,
            "n_rows": int(len(group)),
            "n_outliers": int(outlier_positions.sum()),
            "n_nonfinite_excluded": int(nonfinite_positions.sum()),
            "Method": method,
        }
        groups.append(group_record)
        for position, index in enumerate(indices):
            is_outlier = bool(outlier_positions[position])
            is_nonfinite = bool(nonfinite_positions[position])
            reason = (
                "nonfinite_value"
                if is_nonfinite
                else outlier_reason
                if is_outlier
                else ""
            )
            detail.append(
                {
                    **group_labels,
                    "Metric": spec.metric,
                    "FeatureOutput": spec.feature_output,
                    "SourceRow": out.at[index, "SourceRow"],
                    "IsOutlier": is_outlier,
                    "IncludeAggregate": not (is_outlier or is_nonfinite),
                    "OutlierReason": reason,
                    "Score": (
                        float(scores[position])
                        if np.isfinite(scores[position])
                        or (policy["mode"] == "mad" and np.isinf(scores[position]))
                        else np.nan
                    ),
                    "Method": "nonfinite_excluded" if is_nonfinite else method,
                }
            )
    return out, detail, groups


def _update_manifest_columns(
    state: RunState, spec: FeatureOutputSpec, stage: str, columns: int
) -> bool:
    found = False
    for row in state.feature_output_manifest:
        if (
            identity_equal(row["Stage"], stage)
            and identity_equal(row["Domain"], spec.domain)
            and identity_equal(row["Metric"], spec.metric)
            and identity_equal(row["FeatureOutput"], spec.feature_output)
        ):
            row["Columns"] = columns
            found = True
    return found


def run_outliers(state: RunState, *, update_transform: bool) -> None:
    """Attach one transformed-scale decision to merge and transform branches."""
    config = state.config
    protocol = int(config["execution"]["pickle_protocol"])
    enabled = set(config["outlier_filter"]["enabled_representations"])
    for spec in iter_feature_output_specs(config):
        merge_path = stage_path(state.output_root, "merge", spec, config)
        merged = read_stage_table(merge_path)
        if update_transform:
            transform_path = stage_path(state.output_root, "transform", spec, config)
            transformed = read_stage_table(transform_path)
            if [casefold_identity(value) for value in merged["SourceRow"]] != [
                casefold_identity(value) for value in transformed["SourceRow"]
            ]:
                raise ValueError(
                    "Merge and transform SourceRow order differs for "
                    f"{spec.metric}/{spec.feature_output}."
                )
            preview = transformed
        else:
            transformed = None
            preview = (
                transform_df(
                    merged,
                    value_col="Value",
                    mode=configured_transform_mode(spec, config),
                    drop_empty=False,
                )
                if spec.representation in enabled
                else merged
            )
        marked, detail, groups = mark_outliers(merged, preview, spec, config)
        atomic_pickle(marked, merge_path, True, protocol)
        merge_recorded = _update_manifest_columns(
            state, spec, "merge", len(marked.columns)
        )
        if not merge_recorded:
            record_feature_output(state, "outliers", spec, merge_path, marked, "merge")
        if transformed is not None:
            for column in ("IsOutlier", "IncludeAggregate", "OutlierReason"):
                transformed[column] = marked[column].to_numpy()
            atomic_pickle(transformed, transform_path, True, protocol)
            _update_manifest_columns(state, spec, "transform", len(transformed.columns))
        state.add_qc("outliers", detail)
        state.add_qc("outlier_groups", groups)
