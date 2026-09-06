"""Physical-contact jitter with effect-only, fixed-patient rank calculations."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .correlation import rank_effect_coordinates
from .programming import (
    CELL_GROUP,
    MODELS,
    ZERO_TOLERANCE,
    ProgrammingConfig,
    ProgrammingInputs,
    cell_arrays,
)

SUMMARY_COLUMNS = [
    "rho_median",
    "rho_p025",
    "rho_p975",
    "PrRhoNegative",
    "NominalSignAgreement",
    "DoseRankRhoMedian",
    "DoseRankRhoP025",
    "DoseRankRhoP975",
]


def draw_contact_jitter(
    contacts: list[str], cv: float, iterations: int, seed: int
) -> pd.DataFrame:
    """Use stable contact-keyed streams, independent of input or cohort ordering."""

    keys = sorted(set(contacts))
    values = np.ones((iterations, len(keys)), dtype=float)
    if cv > 0:
        variance = np.log1p(cv * cv)
        for column, key in enumerate(keys):
            entropy = [seed, *float(cv).hex().encode("ascii"), 0, *key.encode("utf-8")]
            generator = np.random.Generator(
                np.random.PCG64(np.random.SeedSequence(entropy))
            )
            values[:, column] = generator.lognormal(
                -variance / 2, np.sqrt(variance), iterations
            )
    return pd.DataFrame(values, columns=keys)


def summarize_jitter_effects(
    effects: np.ndarray, dose_ranks: np.ndarray, nominal: np.ndarray
) -> pd.DataFrame:
    """Summarize feature-by-iteration effects using only estimable iterations."""

    effects = np.clip(effects, -1, 1)
    effects = np.where(np.abs(effects) < ZERO_TOLERANCE, 0.0, effects)
    finite = np.isfinite(effects)
    counts = finite.sum(axis=1)
    result = pd.DataFrame(
        np.nan, index=np.arange(len(nominal)), columns=SUMMARY_COLUMNS
    )
    result["ValidIterations"] = counts
    result["ValidFraction"] = counts / effects.shape[1]
    result["JitterStatus"] = np.where(counts > 0, "ok", "no_valid_iterations")
    for column in np.flatnonzero(counts):
        valid = finite[column]
        values = effects[column, valid]
        result.loc[column, ["rho_p025", "rho_median", "rho_p975"]] = np.quantile(
            values, [0.025, 0.5, 0.975]
        )
        result.loc[column, "PrRhoNegative"] = np.mean(values < 0)
        if abs(nominal[column]) >= ZERO_TOLERANCE:
            result.loc[column, "NominalSignAgreement"] = np.mean(
                np.sign(values) == np.sign(nominal[column])
            )
        rank_values = dose_ranks[valid]
        result.loc[
            column, ["DoseRankRhoP025", "DoseRankRhoMedian", "DoseRankRhoP975"]
        ] = np.quantile(rank_values, [0.025, 0.5, 0.975])
    return result


def compute_programming_jitter(
    inputs: ProgrammingInputs,
    correlations: pd.DataFrame,
    config: ProgrammingConfig,
    *,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Perturb only dose for the supplied nominal endpoint/model components."""

    contacts = inputs.programming.loc[
        inputs.programming["Available"], "PhysicalContact"
    ].tolist()
    all_results = []
    for cv in config.cv:
        draws = draw_contact_jitter(contacts, cv, config.iterations, config.seed)
        # Nominal p values are deliberately absent from the simulation output.
        frame = correlations.drop(columns="p_holm").copy()
        frame["JitterCV"] = cv
        frame["Iterations"] = config.iterations
        frame["Seed"] = config.seed
        for column in SUMMARY_COLUMNS:
            frame[column] = np.nan
        frame["ValidIterations"] = 0
        frame["ValidFraction"] = 0.0
        frame["JitterStatus"] = "nominal_not_estimable"
        valid = frame.loc[frame["Status"].eq("ok")]
        for _, cells in valid.groupby(
            [*CELL_GROUP, "Model", "SubjectIDs"], sort=False, observed=True
        ):
            ids, x_all, patient_rows = cell_arrays(inputs, cells)
            fixed_ids = cells.iloc[0]["SubjectIDs"].split(";")
            selected = pd.Index(ids).get_indexer(fixed_ids)
            x = x_all[selected]
            patients = patient_rows.loc[fixed_ids]
            y = patients["ProgrammingIntensity"].to_numpy(dtype=float)
            simulated = (
                y[:, None] * draws[patients["PhysicalContact"].tolist()].to_numpy().T
            )
            covariate = MODELS[cells.iloc[0]["Model"]]
            residualizer = None
            if covariate is not None:
                cov_rank = rankdata(
                    patients[covariate].to_numpy(dtype=float), method="average"
                )
                design = np.column_stack((np.ones(len(y)), cov_rank))
                residualizer = np.eye(len(y)) - design @ np.linalg.pinv(design)
            x_standard, x_valid = rank_effect_coordinates(x, residualizer)
            y_standard, y_valid = rank_effect_coordinates(simulated, residualizer)
            effects = x_standard.T @ y_standard
            effects[~x_valid, :] = np.nan
            effects[:, ~y_valid] = np.nan
            original_rank, _ = rank_effect_coordinates(y[:, None])
            simulated_rank, rank_valid = rank_effect_coordinates(simulated)
            dose_ranks = np.clip((original_rank.T @ simulated_rank).ravel(), -1, 1)
            dose_ranks[~rank_valid] = np.nan
            summary = summarize_jitter_effects(
                effects, dose_ranks, cells["rho"].to_numpy()
            )
            frame.loc[cells.index, summary.columns] = summary.to_numpy()
        frame["ValidIterations"] = frame["ValidIterations"].astype(int)
        all_results.append(frame)
        if progress is not None:
            progress(
                f"Jitter CV={cv:g}: {len(frame)} nominal cells, {config.iterations} iterations."
            )
    return (
        pd.concat(all_results, ignore_index=True)
        .sort_values(["CorrelationID", "JitterCV"])
        .reset_index(drop=True)
    )
