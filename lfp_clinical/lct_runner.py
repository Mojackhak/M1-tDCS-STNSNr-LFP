"""Compute LCT and LCT-adjusted adjacent-phase associations without figures."""

from __future__ import annotations

import argparse
from math import factorial
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import yaml

from lfp_cohort.io import atomic_csv
from .correlation import _holm_adjust, _ordinary_group, _partial_group
from .runner import _trash_existing


FAMILY = [
    "ScaleLabel", "EndpointID", "Domain", "RegionValue", "Polar", "Metric",
    "FeatureOutput", "Band",
]


def estimate(x: np.ndarray, y: np.ndarray, covariates: np.ndarray) -> pd.DataFrame:
    """Estimate complete-case columns, retaining non-estimable identities."""
    n, count = x.shape
    design = np.column_stack((np.ones(n), rankdata(covariates, axis=0)))
    design_rank = int(np.linalg.matrix_rank(design)) if n else 0
    df = n - design_rank - 1
    result = pd.DataFrame({
        "n": [n] * count, "DesignRank": design_rank, "ReferenceDF": df,
        "rho": np.nan, "p_raw": np.nan, "Status": "insufficient_n",
        "EnumeratedPermutations": 0,
    })
    if n < 5 or df <= 0:
        return result
    rho, p, valid = _partial_group(x, y, covariates)
    result["rho"], result["p_raw"] = rho, p
    result["Status"] = np.where(valid, "ok", "constant_rank_residual")
    result["EnumeratedPermutations"] = factorial(n)
    return result


def compute(lct: pd.DataFrame, endpoints: pd.DataFrame, predictors: pd.DataFrame):
    """Return endpoint results, predictor results and patient-level coverage."""
    if lct.ID.duplicated().any() or endpoints.duplicated(["ID", "ScaleLabel"]).any():
        raise ValueError("Clinical input patient keys must be unique.")
    if predictors.duplicated(["ID", "PredictorID"]).any():
        raise ValueError("Predictor patient keys must be unique.")
    ids = lct.ID.to_numpy()
    lct_values = lct.LCT_ImprovementPercent.to_numpy(float)
    varying = ["ID", "Value", "FromValue", "ToValue"]
    metadata = predictors.drop(columns=varying).drop_duplicates()
    if metadata.PredictorID.duplicated().any():
        raise ValueError("Predictor metadata must be invariant across patients.")
    metadata = metadata.reset_index(drop=True)
    matrix = predictors.pivot(index="ID", columns="PredictorID", values="Value")
    matrix = matrix.reindex(index=ids, columns=metadata.PredictorID).to_numpy(float)
    clinical_results, tdcs_results, coverage = [], [], []
    for scale in ["UPDRS-III", "PDQ-39"]:
        clinical = endpoints[endpoints.ScaleLabel.eq(scale)].set_index("ID").reindex(ids)
        baseline = clinical.Baseline.to_numpy(float)
        for endpoint, column in [("stn-3m", "STN3m"), ("stn-snr-3m", "STNSNr3m")]:
            outcome = clinical[column].to_numpy(float)
            base_mask = np.isfinite(baseline) & np.isfinite(outcome) & np.isfinite(lct_values)
            x, y, b = lct_values[base_mask, None], outcome[base_mask], baseline[base_mask]
            row = estimate(x, y, b).iloc[0].to_dict()
            ordinary_rho, ordinary_p, _ = _ordinary_group(x, y) if len(y) >= 3 else (
                [np.nan], [np.nan], [False]
            )
            row.update(ScaleLabel=scale, EndpointID=endpoint,
                       SubjectIDs=";".join(ids[base_mask]),
                       rho_ordinary=ordinary_rho[0], p_raw_ordinary=ordinary_p[0])
            clinical_results.append(row)

            def record_coverage(predictor_id, values, mask):
                frame = pd.DataFrame({
                    "ScaleLabel": scale, "EndpointID": endpoint,
                    "PredictorID": predictor_id, "ID": ids,
                    "Baseline": baseline, "Outcome": outcome, "LCT": lct_values,
                    "PredictorValue": values, "Included": mask,
                })
                reasons = []
                for index in range(len(ids)):
                    reasons.append(";".join(name for name, present in [
                        ("missing_baseline", np.isfinite(baseline[index])),
                        ("missing_outcome", np.isfinite(outcome[index])),
                        ("missing_lct", np.isfinite(lct_values[index])),
                        ("missing_predictor", np.isfinite(values[index])),
                    ] if not present))
                frame["MissingReason"] = reasons
                coverage.append(frame)

            record_coverage("LCT", lct_values, base_mask)
            groups = {}
            for j in range(matrix.shape[1]):
                mask = base_mask & np.isfinite(matrix[:, j])
                groups.setdefault(tuple(mask), []).append(j)
                record_coverage(metadata.PredictorID.iloc[j], matrix[:, j], mask)
            for mask_key, columns in groups.items():
                mask = np.asarray(mask_key)
                x = matrix[np.ix_(mask, columns)]
                y, b, l = outcome[mask], baseline[mask], lct_values[mask]
                estimates = estimate(x, y, np.column_stack((b, l)))
                estimates["SubjectIDs"] = ";".join(ids[mask])
                estimates["SampleSizeBand"] = "n>=6" if len(y) >= 6 else f"n={len(y)}"
                estimates["rho_baseline_only"] = np.nan
                estimates["p_raw_baseline_only"] = np.nan
                estimates["rho_lct_matched"] = np.nan
                estimates["p_raw_lct_matched"] = np.nan
                if len(y) >= 5:
                    rho, p, _ = _partial_group(x, y, b)
                    estimates["rho_baseline_only"] = rho
                    estimates["p_raw_baseline_only"] = p
                    rho, p, _ = _partial_group(l[:, None], y, b)
                    estimates["rho_lct_matched"] = rho[0]
                    estimates["p_raw_lct_matched"] = p[0]
                frame = pd.concat([metadata.iloc[columns].reset_index(drop=True), estimates], axis=1)
                frame["ScaleLabel"], frame["EndpointID"] = scale, endpoint
                tdcs_results.append(frame)
    clinical = pd.DataFrame(clinical_results)
    clinical["Covariates"] = "Baseline"
    tdcs = pd.concat(tdcs_results, ignore_index=True)
    grouped = tdcs.groupby(FAMILY, dropna=False, sort=False)
    tdcs["p_holm"] = grouped.p_raw.transform(_holm_adjust)
    tdcs["p_holm_baseline_only"] = grouped.p_raw_baseline_only.transform(_holm_adjust)
    tdcs["CorrectionFamilySize"] = grouped.p_raw.transform("count")
    tdcs["CorrectionFamily"] = tdcs[FAMILY].astype(str).agg("|".join, axis=1)
    tdcs["Covariates"] = "Baseline;LCT_ImprovementPercent"
    return clinical, tdcs.sort_values(FAMILY + ["Phase", "Lat"]), pd.concat(coverage, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    paths = yaml.safe_load(args.config.read_text())["paths"]
    tables = {name: pd.read_csv(paths[name]) for name in ["lct", "endpoints", "predictors"]}
    clinical, tdcs, coverage = compute(**tables)
    outputs = {
        "lct-outcome/results.csv": clinical,
        "tdcs-adjusted/results.csv": tdcs,
        "tdcs-adjusted/significant-holm.csv": tdcs[tdcs.p_holm.lt(0.05)],
        "coverage.csv": coverage,
    }
    root = Path(paths["output_root"])
    manifest = [{"Role": "input", "Path": str(Path(paths[k]).resolve()), "Rows": len(v)}
                for k, v in tables.items()]
    manifest += [{"Role": "config", "Path": str(args.config.resolve()), "Rows": np.nan}]
    manifest += [{"Role": "output", "Path": str(root / k), "Rows": len(v)}
                 for k, v in outputs.items()]
    outputs["manifest.csv"] = pd.DataFrame(manifest)
    existing = [root / name for name in outputs if (root / name).exists()]
    if existing:
        _trash_existing(existing)
    for name, table in outputs.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_csv(table, target, overwrite=False)
    print(clinical.to_string(index=False))
    print(tdcs.groupby(["SampleSizeBand", "Status"]).size().to_string())
    print(f"Holm-significant adjusted associations: {tdcs.p_holm.lt(0.05).sum()}")
    print(f"Output: {root}")


if __name__ == "__main__":
    main()
