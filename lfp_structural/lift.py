"""Extract any-segment mask membership and Region-conditioned Lift."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .field_exposure import (
    FiberExposure,
    FieldSpec,
    FieldSummary,
    build_sparse_field_lookup,
    maximum_field_exposure,
)
from .lead_runtime import activate_sextant
from .masks import MaskSpec, MaskSummary, resolve_mask


@dataclass(frozen=True)
class MembershipExtraction:
    """Unique one-based fiber IDs for every mask in one connectome traversal."""

    n_source_fibers: int
    memberships: Mapping[str, np.ndarray]
    masks: tuple[MaskSummary, ...]


@dataclass(frozen=True)
class FieldExposureExtraction:
    """Positive continuous-field exposure from one connectome traversal."""

    n_source_fibers: int
    exposures: Mapping[str, FiberExposure]
    fields: tuple[FieldSummary, ...]


@dataclass(frozen=True)
class LiftStatistic:
    """One channel-by-Region Region-conditioned enrichment result."""

    n_all: int
    n_seed: int
    n_target: int
    n_seed_target: int
    seed_target_fraction: float | None
    target_background_fraction: float | None
    lift: float | None
    status: str


@dataclass(frozen=True)
class FieldWeightedLiftStatistic:
    """Continuous cortical-field exposure within one Region-conditioned universe."""

    field_threshold_v_per_m: float
    n_field_exposed_all: int
    n_field_exposed_seed: int
    n_field_above_threshold_all: int
    n_field_above_threshold_seed: int
    region_sum_peak_e_v_per_m: float
    seed_sum_peak_e_v_per_m: float
    region_mean_peak_e_v_per_m: float | None
    seed_mean_peak_e_v_per_m: float | None
    region_thresholded_sum_peak_e_v_per_m: float
    seed_thresholded_sum_peak_e_v_per_m: float
    region_thresholded_mean_peak_e_v_per_m: float | None
    seed_thresholded_mean_peak_e_v_per_m: float | None
    field_weighted_lift: float | None
    status: str


def extract_memberships(
    connectome: Any,
    mask_specs: Iterable[MaskSpec],
    *,
    sextant_python_root: Path | str,
    chunk_size: int,
) -> MembershipExtraction:
    """Traverse one source once and collect unique fiber IDs for every mask."""

    activate_sextant(sextant_python_root)
    from sextant.shared.fiber_membership import (
        build_sparse_lookup,
        optimized_membership,
    )

    specs = tuple(mask_specs)
    if not specs:
        raise ValueError("At least one mask is required for membership extraction.")
    identifiers = [spec.mask_id for spec in specs]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Mask identifiers must be unique within one traversal.")
    resolved_and_summary = [resolve_mask(spec, sextant_python_root) for spec in specs]
    resolved = [item[0] for item in resolved_and_summary]
    summaries = tuple(item[1] for item in resolved_and_summary)
    lookup = build_sparse_lookup(resolved)
    selected: list[list[np.ndarray]] = [[] for _ in specs]
    observed = 0
    previous_id = 0
    for chunk in connectome.iter_chunks(chunk_size):
        fiber_ids = np.asarray(chunk.fiber_ids, dtype=np.int64)
        if fiber_ids.size and int(fiber_ids[0]) != previous_id + 1:
            raise ValueError(
                "Connectome chunks must expose consecutive one-based fiber IDs."
            )
        membership = optimized_membership(chunk, lookup)
        if membership.shape != (fiber_ids.size, len(specs)):
            raise ValueError("Segment-aware membership returned an unexpected shape.")
        for mask_index in range(len(specs)):
            hits = fiber_ids[membership[:, mask_index]]
            if hits.size:
                selected[mask_index].append(hits)
        if fiber_ids.size:
            previous_id = int(fiber_ids[-1])
        observed += int(fiber_ids.size)
    if observed != int(connectome.n_fibers):
        raise ValueError(
            f"Connectome traversal count mismatch: expected {connectome.n_fibers}, "
            f"observed {observed}"
        )
    memberships: dict[str, np.ndarray] = {}
    for spec, chunks in zip(specs, selected, strict=True):
        values = (
            np.concatenate(chunks).astype(np.int64, copy=False)
            if chunks
            else np.empty(0, dtype=np.int64)
        )
        values.setflags(write=False)
        memberships[spec.mask_id] = values
    return MembershipExtraction(
        n_source_fibers=observed,
        memberships=memberships,
        masks=summaries,
    )


def extract_field_exposures(
    connectome: Any,
    field_specs: Iterable[FieldSpec],
    *,
    chunk_size: int,
) -> FieldExposureExtraction:
    """Traverse one source once and retain only positive per-fiber peak exposure."""

    specs = tuple(field_specs)
    lookup, summaries = build_sparse_field_lookup(specs)
    exposed_ids: list[list[np.ndarray]] = [[] for _ in specs]
    exposed_values: list[list[np.ndarray]] = [[] for _ in specs]
    observed = 0
    previous_id = 0
    for chunk in connectome.iter_chunks(chunk_size):
        fiber_ids = np.asarray(chunk.fiber_ids, dtype=np.int64)
        if fiber_ids.size and int(fiber_ids[0]) != previous_id + 1:
            raise ValueError(
                "Connectome chunks must expose consecutive one-based fiber IDs."
            )
        exposure = maximum_field_exposure(chunk, lookup)
        if exposure.shape != (fiber_ids.size, len(specs)):
            raise ValueError("Continuous field traversal returned an unexpected shape.")
        for field_index in range(len(specs)):
            positive = exposure[:, field_index] > 0
            if np.any(positive):
                exposed_ids[field_index].append(fiber_ids[positive])
                exposed_values[field_index].append(exposure[positive, field_index])
        if fiber_ids.size:
            previous_id = int(fiber_ids[-1])
        observed += int(fiber_ids.size)
    if observed != int(connectome.n_fibers):
        raise ValueError(
            f"Connectome traversal count mismatch: expected {connectome.n_fibers}, "
            f"observed {observed}"
        )
    exposures: dict[str, FiberExposure] = {}
    for spec, id_chunks, value_chunks in zip(
        specs,
        exposed_ids,
        exposed_values,
        strict=True,
    ):
        fiber_ids = (
            np.concatenate(id_chunks).astype(np.int64, copy=False)
            if id_chunks
            else np.empty(0, dtype=np.int64)
        )
        peak_values = (
            np.concatenate(value_chunks).astype(np.float32, copy=False)
            if value_chunks
            else np.empty(0, dtype=np.float32)
        )
        fiber_ids.setflags(write=False)
        peak_values.setflags(write=False)
        exposures[spec.field_id] = FiberExposure(
            fiber_ids=fiber_ids,
            peak_values=peak_values,
        )
    return FieldExposureExtraction(
        n_source_fibers=observed,
        exposures=exposures,
        fields=summaries,
    )


def _intersect(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.intersect1d(left, right, assume_unique=True)


def compute_region_conditioned_lift(
    region_fiber_ids: np.ndarray,
    seed_fiber_ids: np.ndarray,
    target_fiber_ids: np.ndarray,
) -> LiftStatistic:
    """Calculate the frozen Region-conditioned Lift without pseudocounts."""

    region = np.asarray(region_fiber_ids, dtype=np.int64)
    seed = np.asarray(seed_fiber_ids, dtype=np.int64)
    target = np.asarray(target_fiber_ids, dtype=np.int64)
    region_seed = _intersect(region, seed)
    region_target = _intersect(region, target)
    region_seed_target = _intersect(region_seed, region_target)
    n_all = int(region.size)
    n_seed = int(region_seed.size)
    n_target = int(region_target.size)
    n_seed_target = int(region_seed_target.size)
    if n_all == 0:
        return LiftStatistic(
            n_all=n_all,
            n_seed=n_seed,
            n_target=n_target,
            n_seed_target=n_seed_target,
            seed_target_fraction=None,
            target_background_fraction=None,
            lift=None,
            status="missing_empty_region_universe",
        )
    background = n_target / n_all
    if n_seed == 0:
        return LiftStatistic(
            n_all=n_all,
            n_seed=n_seed,
            n_target=n_target,
            n_seed_target=n_seed_target,
            seed_target_fraction=None,
            target_background_fraction=background,
            lift=None,
            status="missing_no_seed_support",
        )
    conditional = n_seed_target / n_seed
    if n_target == 0:
        return LiftStatistic(
            n_all=n_all,
            n_seed=n_seed,
            n_target=n_target,
            n_seed_target=n_seed_target,
            seed_target_fraction=conditional,
            target_background_fraction=background,
            lift=None,
            status="missing_no_target_support",
        )
    return LiftStatistic(
        n_all=n_all,
        n_seed=n_seed,
        n_target=n_target,
        n_seed_target=n_seed_target,
        seed_target_fraction=conditional,
        target_background_fraction=background,
        lift=conditional / background,
        status="ok",
    )


def _sum_exposure(
    selected_fiber_ids: np.ndarray,
    exposure_fiber_ids: np.ndarray,
    exposure_values: np.ndarray,
) -> tuple[int, float]:
    common, _, exposure_indexes = np.intersect1d(
        np.asarray(selected_fiber_ids, dtype=np.int64),
        np.asarray(exposure_fiber_ids, dtype=np.int64),
        assume_unique=True,
        return_indices=True,
    )
    values = np.asarray(exposure_values, dtype=np.float64)
    if values.ndim != 1 or values.size != np.asarray(exposure_fiber_ids).size:
        raise ValueError("Exposure fiber IDs and values must be aligned vectors.")
    return int(common.size), float(np.sum(values[exposure_indexes], dtype=np.float64))


def compute_region_conditioned_field_weighted_lift(
    region_fiber_ids: np.ndarray,
    seed_fiber_ids: np.ndarray,
    exposure_fiber_ids: np.ndarray,
    exposure_values: np.ndarray,
    *,
    field_threshold_v_per_m: float,
) -> FieldWeightedLiftStatistic:
    """Calculate threshold-gated peak-field enrichment with fixed fiber counts."""

    region = np.asarray(region_fiber_ids, dtype=np.int64)
    seed = np.asarray(seed_fiber_ids, dtype=np.int64)
    region_seed = _intersect(region, seed)
    n_all = int(region.size)
    n_seed = int(region_seed.size)
    n_exposed_all, region_sum = _sum_exposure(
        region,
        exposure_fiber_ids,
        exposure_values,
    )
    n_exposed_seed, seed_sum = _sum_exposure(
        region_seed,
        exposure_fiber_ids,
        exposure_values,
    )
    values = np.asarray(exposure_values, dtype=np.float64)
    above_threshold = values >= field_threshold_v_per_m
    thresholded_ids = np.asarray(exposure_fiber_ids, dtype=np.int64)[above_threshold]
    thresholded_values = values[above_threshold]
    n_above_all, region_thresholded_sum = _sum_exposure(
        region,
        thresholded_ids,
        thresholded_values,
    )
    n_above_seed, seed_thresholded_sum = _sum_exposure(
        region_seed,
        thresholded_ids,
        thresholded_values,
    )
    if n_all == 0:
        return FieldWeightedLiftStatistic(
            field_threshold_v_per_m=field_threshold_v_per_m,
            n_field_exposed_all=n_exposed_all,
            n_field_exposed_seed=n_exposed_seed,
            n_field_above_threshold_all=n_above_all,
            n_field_above_threshold_seed=n_above_seed,
            region_sum_peak_e_v_per_m=region_sum,
            seed_sum_peak_e_v_per_m=seed_sum,
            region_mean_peak_e_v_per_m=None,
            seed_mean_peak_e_v_per_m=None,
            region_thresholded_sum_peak_e_v_per_m=region_thresholded_sum,
            seed_thresholded_sum_peak_e_v_per_m=seed_thresholded_sum,
            region_thresholded_mean_peak_e_v_per_m=None,
            seed_thresholded_mean_peak_e_v_per_m=None,
            field_weighted_lift=None,
            status="missing_empty_region_universe",
        )
    region_mean = region_sum / n_all
    region_thresholded_mean = region_thresholded_sum / n_all
    if n_seed == 0:
        return FieldWeightedLiftStatistic(
            field_threshold_v_per_m=field_threshold_v_per_m,
            n_field_exposed_all=n_exposed_all,
            n_field_exposed_seed=n_exposed_seed,
            n_field_above_threshold_all=n_above_all,
            n_field_above_threshold_seed=n_above_seed,
            region_sum_peak_e_v_per_m=region_sum,
            seed_sum_peak_e_v_per_m=seed_sum,
            region_mean_peak_e_v_per_m=region_mean,
            seed_mean_peak_e_v_per_m=None,
            region_thresholded_sum_peak_e_v_per_m=region_thresholded_sum,
            seed_thresholded_sum_peak_e_v_per_m=seed_thresholded_sum,
            region_thresholded_mean_peak_e_v_per_m=region_thresholded_mean,
            seed_thresholded_mean_peak_e_v_per_m=None,
            field_weighted_lift=None,
            status="missing_no_seed_support",
        )
    seed_mean = seed_sum / n_seed
    seed_thresholded_mean = seed_thresholded_sum / n_seed
    if region_thresholded_sum == 0.0:
        return FieldWeightedLiftStatistic(
            field_threshold_v_per_m=field_threshold_v_per_m,
            n_field_exposed_all=n_exposed_all,
            n_field_exposed_seed=n_exposed_seed,
            n_field_above_threshold_all=n_above_all,
            n_field_above_threshold_seed=n_above_seed,
            region_sum_peak_e_v_per_m=region_sum,
            seed_sum_peak_e_v_per_m=seed_sum,
            region_mean_peak_e_v_per_m=region_mean,
            seed_mean_peak_e_v_per_m=seed_mean,
            region_thresholded_sum_peak_e_v_per_m=region_thresholded_sum,
            seed_thresholded_sum_peak_e_v_per_m=seed_thresholded_sum,
            region_thresholded_mean_peak_e_v_per_m=region_thresholded_mean,
            seed_thresholded_mean_peak_e_v_per_m=seed_thresholded_mean,
            field_weighted_lift=None,
            status="missing_no_region_suprathreshold_exposure",
        )
    return FieldWeightedLiftStatistic(
        field_threshold_v_per_m=field_threshold_v_per_m,
        n_field_exposed_all=n_exposed_all,
        n_field_exposed_seed=n_exposed_seed,
        n_field_above_threshold_all=n_above_all,
        n_field_above_threshold_seed=n_above_seed,
        region_sum_peak_e_v_per_m=region_sum,
        seed_sum_peak_e_v_per_m=seed_sum,
        region_mean_peak_e_v_per_m=region_mean,
        seed_mean_peak_e_v_per_m=seed_mean,
        region_thresholded_sum_peak_e_v_per_m=region_thresholded_sum,
        seed_thresholded_sum_peak_e_v_per_m=seed_thresholded_sum,
        region_thresholded_mean_peak_e_v_per_m=region_thresholded_mean,
        seed_thresholded_mean_peak_e_v_per_m=seed_thresholded_mean,
        field_weighted_lift=seed_thresholded_mean / region_thresholded_mean,
        status="ok",
    )
