"""Compute cortical field allocation and fit contact-level mixed models."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd

from lfp_clinical.correlation import _benjamini_hochberg

from .artifacts import atomic_csv
from .config import StructuralConnectivityConfig
from .contact_lmm import _run_r_models
from .statistics import (
    _base_lfp_filter,
    _read_pickle,
    _validate_normalized_contract,
)


CORTICAL_ALLOCATION_OUTPUTS = {
    "cortical_allocation_regional_field.csv": (
        "tables/regional_field_allocation.csv"
    ),
    "cortical_allocation_contact_model_rows.csv": (
        "tables/contact_model_rows.csv"
    ),
    "cortical_allocation_contact_pairing.csv": "tables/contact_pairing.csv",
    "cortical_allocation_outcome_sd.csv": "tables/outcome_sd.csv",
    "cortical_allocation_models.csv": "results/models.csv",
    "cortical_allocation_fixed_effects.csv": "results/fixed_effects.csv",
    "cortical_allocation_joint_tests.csv": "results/joint_tests.csv",
    "cortical_allocation_phase_slopes.csv": "results/phase_slopes.csv",
    "cortical_allocation_phase_slope_contrasts.csv": (
        "results/phase_slope_contrasts.csv"
    ),
    "cortical_allocation_prediction_grid.csv": "results/prediction_grid.csv",
    "cortical_allocation_inputs.csv": "manifest/inputs.csv",
    "cortical_allocation_tests.csv": "manifest/tests.csv",
    "cortical_allocation_outputs.csv": "manifest/outputs.csv",
}
CORTICAL_ALLOCATION_TABLES = tuple(CORTICAL_ALLOCATION_OUTPUTS)

MODEL_KEY = [
    "CorticalRegion",
    "Polar",
    "OutcomeDomain",
    "Metric",
    "FeatureOutput",
    "OutcomeScale",
    "Region",
    "PairRegion",
    "Band",
    "Lat",
]
OUTCOME_SD_KEY = [
    "OutcomeDomain",
    "Metric",
    "FeatureOutput",
    "OutcomeScale",
    "Polar",
    "Band",
    "Region",
    "PairRegion",
    "Lat",
]
PHASE_CONTRASTS = ("Early - Late", "Early - Post", "Late - Post")


def _require_columns(table: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = sorted(set(required).difference(table.columns))
    if missing:
        raise KeyError(f"{name} is missing columns: {missing}")


def _load_image(path: Path, label: str) -> nib.spatialimages.SpatialImage:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    image = nib.load(path)
    if len(image.shape) != 3:
        raise ValueError(f"{label} must be a three-dimensional image: {path}")
    return image


def _check_same_grid(
    reference: nib.spatialimages.SpatialImage,
    candidate: nib.spatialimages.SpatialImage,
    label: str,
) -> None:
    if candidate.shape != reference.shape or not np.allclose(
        candidate.affine,
        reference.affine,
        rtol=0,
        atol=1e-5,
    ):
        raise ValueError(f"{label} is not on the registered MNI field grid.")


def _binary_mask(data: np.ndarray, label: str) -> np.ndarray:
    if not np.isfinite(data).all():
        raise ValueError(f"{label} contains nonfinite values.")
    if bool(np.any((data != 0) & (data != 1))):
        raise ValueError(f"{label} is not binary.")
    return data.astype(bool, copy=False)


def compute_regional_field_allocation(
    config: StructuralConnectivityConfig,
    *,
    slice_chunk: int | None = None,
) -> pd.DataFrame:
    """Integrate thresholded MNI field magnitude over four actual-side ROIs."""

    settings = config["inference"]["cortical_allocation_lmm"]
    regions = list(settings["regions"])
    threshold = float(config["masks"]["field_threshold_v_per_m"])
    records: list[dict[str, Any]] = []
    for subject in config.subjects:
        subject_id = subject.subject_id
        side = subject.stimulation_side
        hemisphere = "lh" if side == "L" else "rh"
        field_path = config.mni_cortical_magnitude(subject_id)
        target_path = config.mni_target(subject_id)
        field_image = _load_image(field_path, "MNI cortical magnitude")
        target_image = _load_image(target_path, "MNI Tstim mask")
        _check_same_grid(field_image, target_image, "MNI Tstim mask")
        region_paths = {
            region: config.cortical_region_mask(subject_id, region)
            for region in regions
        }
        region_images = {
            region: _load_image(path, f"{region} cortical ROI")
            for region, path in region_paths.items()
        }
        for region, image in region_images.items():
            _check_same_grid(field_image, image, f"{region} cortical ROI")

        voxel_volume = float(abs(np.linalg.det(field_image.affine[:3, :3])))
        if not np.isfinite(voxel_volume) or voxel_volume <= 0:
            raise ValueError(f"Invalid MNI voxel volume for {subject_id}.")
        chunk_size = slice_chunk or field_image.shape[2]
        if chunk_size <= 0:
            raise ValueError("slice_chunk must be positive when specified.")
        accumulators = {
            region: {
                "region_voxels": 0,
                "suprathreshold_voxels": 0,
                "field_sum": 0.0,
            }
            for region in regions
        }
        for start in range(0, field_image.shape[2], chunk_size):
            stop = min(start + chunk_size, field_image.shape[2])
            index = (slice(None), slice(None), slice(start, stop))
            field = np.asarray(field_image.dataobj[index], dtype=np.float64)
            if not np.isfinite(field).all() or bool(np.any(field < 0)):
                raise ValueError(
                    f"MNI cortical magnitude contains invalid values: {field_path}"
                )
            target = _binary_mask(
                np.asarray(target_image.dataobj[index]),
                f"MNI Tstim mask for {subject_id}",
            )
            occupied = np.zeros(target.shape, dtype=bool)
            for region in regions:
                roi = _binary_mask(
                    np.asarray(region_images[region].dataobj[index]),
                    f"{region} cortical ROI for {subject_id}",
                )
                if bool(np.any(occupied & roi)):
                    raise ValueError(
                        f"Candidate cortical ROIs overlap for {subject_id}: {region}"
                    )
                occupied |= roi
                selected = roi & target
                accumulator = accumulators[region]
                accumulator["region_voxels"] += int(np.count_nonzero(roi))
                accumulator["suprathreshold_voxels"] += int(
                    np.count_nonzero(selected)
                )
                accumulator["field_sum"] += float(field[selected].sum(dtype=np.float64))

        doses: dict[str, float] = {}
        for region, values in accumulators.items():
            count = int(values["region_voxels"])
            if count == 0:
                raise ValueError(f"Candidate cortical ROI is empty: {region_paths[region]}")
            doses[region] = float(values["field_sum"]) / count
        candidate_sum = float(sum(doses.values()))
        allocation_status = (
            "ok"
            if np.isfinite(candidate_sum) and candidate_sum > 0
            else "undefined_zero_candidate_dose"
        )
        for region in regions:
            values = accumulators[region]
            region_voxels = int(values["region_voxels"])
            active_voxels = int(values["suprathreshold_voxels"])
            field_sum = float(values["field_sum"])
            records.append(
                {
                    "AnalysisID": settings["analysis_id"],
                    "CandidateSetID": settings["candidate_set_id"],
                    "AtlasName": settings["atlas_name"],
                    "ID": subject_id,
                    "StimSide": side,
                    "Hemisphere": hemisphere,
                    "Space": "MNI152NLin2009bAsym",
                    "CorticalRegion": region,
                    "FieldThreshold_Vm": threshold,
                    "VoxelVolume_mm3": voxel_volume,
                    "RegionVoxelCount": region_voxels,
                    "SuprathresholdVoxelCount": active_voxels,
                    "RegionVolume_mm3": region_voxels * voxel_volume,
                    "SuprathresholdVolume_mm3": active_voxels * voxel_volume,
                    "CoverageFraction": active_voxels / region_voxels,
                    "MagnEVolumeIntegral_Vm_mm3": field_sum * voxel_volume,
                    "SuprathresholdMeanMagnE_Vm": (
                        field_sum / active_voxels if active_voxels else np.nan
                    ),
                    "VoxelVolumeWeightedDose_Vm": doses[region],
                    "CandidateDoseSum_Vm": candidate_sum,
                    "RelativeAllocation": (
                        doses[region] / candidate_sum
                        if allocation_status == "ok"
                        else np.nan
                    ),
                    "AllocationStatus": allocation_status,
                    "FieldPath": str(field_path),
                    "TargetPath": str(target_path),
                    "RegionMaskPath": str(region_paths[region]),
                }
            )
    output = pd.DataFrame(records)
    expected = len(config.subjects) * len(regions)
    if len(output) != expected:
        raise ValueError(
            f"Regional field allocation has {len(output)} rows instead of {expected}."
        )
    finite_groups = output.loc[output["AllocationStatus"].eq("ok")].groupby(
        ["ID", "StimSide"], sort=False, observed=True
    )["RelativeAllocation"]
    sums = finite_groups.sum()
    if len(sums) and not np.allclose(sums, 1.0, rtol=0, atol=1e-12):
        raise ValueError("Finite cortical relative allocations do not sum to one.")
    return output.sort_values(
        ["ID", "CorticalRegion"], kind="mergesort"
    ).reset_index(drop=True)


def _registered_local_outcomes(
    config: StructuralConnectivityConfig,
    *,
    additional_local_outcomes: Iterable[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    settings = config["inference"]["cortical_allocation_lmm"]
    additional = (
        list(settings.get("additional_local_outcomes", []))
        if additional_local_outcomes is None
        else list(additional_local_outcomes)
    )
    entries = [
        {
            "metric": "periodic",
            "feature_output": "mean-scalar",
            "path": str(config.path_value("local_lfp")),
            "transform": "dB",
            "transform_policy_status": "matched",
            "outcome_scale": config["outcomes"]["local_scales"]["periodic"],
            "bands": list(config["outcomes"]["bands"]),
        },
        *(dict(entry) for entry in additional),
    ]
    identities = [
        (str(entry["metric"]), str(entry["feature_output"])) for entry in entries
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("Registered cortical-allocation local outcomes are duplicated.")
    return entries


def _local_lfp_rows(
    config: StructuralConnectivityConfig,
    *,
    additional_local_outcomes: Iterable[Mapping[str, Any]] | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for entry in _registered_local_outcomes(
        config,
        additional_local_outcomes=additional_local_outcomes,
    ):
        path = Path(str(entry["path"]))
        metric = str(entry["metric"])
        feature_output = str(entry["feature_output"])
        table = _read_pickle(path)
        _validate_normalized_contract(
            table,
            path,
            domain="local",
            metric=metric,
            transform=str(entry["transform"]),
            transform_policy_status=str(entry["transform_policy_status"]),
        )
        if str(table.attrs.get("feature_output")) != feature_output:
            raise ValueError(
                "Normalized local outcome FeatureOutput does not match the "
                f"registered source: {path}"
            )
        _require_columns(
            table,
            {"Domain", "Metric", "FeatureOutput", "Region", "Channel"},
            path.name,
        )
        selected = table.loc[
            _base_lfp_filter(table, config, bands=entry["bands"])
            & table["Domain"].eq("local")
            & table["Metric"].eq(metric)
            & table["FeatureOutput"].eq(feature_output)
            & table["Region"].isin(["STN", "SNr"])
        ].copy()
        key = [
            "ID",
            "StimSide",
            "Polar",
            "Phase",
            "Band",
            "Region",
            "Channel",
            "Metric",
            "FeatureOutput",
        ]
        if selected.duplicated(key).any():
            raise ValueError(
                f"Local normalized LFP table has duplicate model rows: {path}"
            )
        selected["OutcomeDomain"] = "local"
        selected["OutcomeScale"] = str(entry["outcome_scale"])
        selected["PairRegion"] = ""
        selected["ChannelPair"] = ""
        selected["channel_a"] = ""
        selected["channel_b"] = ""
        selected["ContactUnitID"] = selected["Channel"].astype(str)
        selected["LFPSourcePath"] = str(path)
        frames.append(selected)
    return pd.concat(frames, ignore_index=True, sort=False)


def _connectivity_lfp_rows(config: StructuralConnectivityConfig) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    transforms = {
        "ciplv": "logit",
        "imcoh_abs": "fisherz",
        "psi": "none",
        "trgc": "none",
        "wpli": "logit",
    }
    for metric in config["outcomes"]["connectivity_metrics"]:
        path = config.path_value("connectivity_lfp_root") / metric / "mean-scalar.pkl"
        table = _read_pickle(path)
        _validate_normalized_contract(
            table,
            path,
            domain="connectivity",
            metric=metric,
            transform=transforms[metric],
        )
        _require_columns(
            table,
            {
                "Domain",
                "Metric",
                "PairRegion",
                "ChannelPair",
                "channel_a",
                "channel_b",
            },
            path.name,
        )
        pair_region = config["outcomes"]["connectivity_pair_regions"][metric]
        selected = table.loc[
            _base_lfp_filter(table, config)
            & table["Domain"].eq("connectivity")
            & table["Metric"].eq(metric)
            & table["PairRegion"].eq(pair_region)
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
                f"Connectivity normalized LFP table has duplicate model rows: {metric}"
            )
        selected["OutcomeDomain"] = "connectivity"
        selected["OutcomeScale"] = config["outcomes"]["connectivity_scales"][
            metric
        ]
        selected["Region"] = ""
        selected["Channel"] = ""
        selected["ContactUnitID"] = (
            selected["channel_a"].astype(str)
            + "|"
            + selected["channel_b"].astype(str)
        )
        selected["LFPSourcePath"] = str(path)
        frames.append(selected)
    return pd.concat(frames, ignore_index=True, sort=False)


def prepare_lfp_rows(
    config: StructuralConnectivityConfig,
    *,
    additional_local_outcomes: Iterable[Mapping[str, Any]] | None = None,
) -> pd.DataFrame:
    """Read the signed Pre-centered normalized contact outcomes once."""

    output = pd.concat(
        [
            _local_lfp_rows(
                config,
                additional_local_outcomes=additional_local_outcomes,
            ),
            _connectivity_lfp_rows(config),
        ],
        ignore_index=True,
        sort=False,
    )
    for column in (
        "Record",
        "Trial",
        "Region",
        "PairRegion",
        "Channel",
        "ChannelPair",
        "channel_a",
        "channel_b",
    ):
        output[column] = output[column].fillna("").astype(str)
    output["Value"] = pd.to_numeric(output["Value"], errors="coerce")
    columns = [
        "ID",
        "Record",
        "Trial",
        "StimSide",
        "OutcomeDomain",
        "Metric",
        "FeatureOutput",
        "OutcomeScale",
        "Polar",
        "Phase",
        "Band",
        "Lat",
        "Region",
        "PairRegion",
        "ContactUnitID",
        "Channel",
        "ChannelPair",
        "channel_a",
        "channel_b",
        "Value",
        "ValueN",
        "LFPSourcePath",
    ]
    output = output[columns]
    key = [
        "ID",
        "StimSide",
        "OutcomeDomain",
        "Metric",
        "FeatureOutput",
        "Polar",
        "Phase",
        "Band",
        "Region",
        "PairRegion",
        "ContactUnitID",
    ]
    if output.duplicated(key).any():
        raise ValueError("Normalized LFP inputs contain duplicate contact model rows.")
    return output.sort_values(key, kind="mergesort").reset_index(drop=True)


def expand_model_rows(
    lfp_rows: pd.DataFrame,
    allocation: pd.DataFrame,
) -> pd.DataFrame:
    """Cross each normalized contact outcome with the four subject allocations."""

    allocation_columns = [
        "ID",
        "StimSide",
        "CorticalRegion",
        "VoxelVolumeWeightedDose_Vm",
        "CandidateDoseSum_Vm",
        "RelativeAllocation",
        "AllocationStatus",
    ]
    expanded = lfp_rows.merge(
        allocation[allocation_columns],
        on=["ID", "StimSide"],
        how="left",
        validate="many_to_many",
    )
    expected_multiplier = allocation["CorticalRegion"].nunique()
    if len(expanded) != len(lfp_rows) * expected_multiplier:
        raise ValueError("Each LFP row was not expanded to every cortical region.")
    status = pd.Series("eligible", index=expanded.index, dtype="string")
    missing = expanded["CorticalRegion"].isna()
    status.loc[missing] = "missing_regional_allocation"
    bad_allocation = ~missing & ~expanded["AllocationStatus"].eq("ok")
    status.loc[bad_allocation] = expanded.loc[
        bad_allocation, "AllocationStatus"
    ].astype(str)
    nonfinite_x = (
        expanded["AllocationStatus"].eq("ok")
        & ~np.isfinite(pd.to_numeric(expanded["RelativeAllocation"], errors="coerce"))
    )
    status.loc[nonfinite_x] = "nonfinite_relative_allocation"
    nonfinite_y = ~np.isfinite(pd.to_numeric(expanded["Value"], errors="coerce"))
    status.loc[nonfinite_y] = "nonfinite_outcome"
    expanded["RowEligibilityStatus"] = status
    return expanded.sort_values(
        [*MODEL_KEY, "ID", "ContactUnitID", "Phase"], kind="mergesort"
    ).reset_index(drop=True)


def planned_models(config: StructuralConnectivityConfig) -> pd.DataFrame:
    """Register every cortical-region by outcome model cell."""

    settings = config["inference"]["cortical_allocation_lmm"]
    records: list[dict[str, Any]] = []
    periodic = _registered_local_outcomes(
        config,
        additional_local_outcomes=[],
    )[0]
    for cortical_region in settings["regions"]:
        for polar in config["outcomes"]["polarities"]:
            for region in ("STN", "SNr"):
                for band in periodic["bands"]:
                    records.append(
                        {
                            "CorticalRegion": cortical_region,
                            "Polar": polar,
                            "OutcomeDomain": "local",
                            "Metric": periodic["metric"],
                            "FeatureOutput": periodic["feature_output"],
                            "OutcomeScale": periodic["outcome_scale"],
                            "Region": region,
                            "PairRegion": "",
                            "Band": band,
                            "Lat": config["outcomes"]["laterality"],
                        }
                    )
            for metric in config["outcomes"]["connectivity_metrics"]:
                for band in config["outcomes"]["bands"]:
                    records.append(
                        {
                            "CorticalRegion": cortical_region,
                            "Polar": polar,
                            "OutcomeDomain": "connectivity",
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
                            "Lat": config["outcomes"]["laterality"],
                        }
                    )
    for entry in settings.get("additional_local_outcomes", []):
        for cortical_region in settings["regions"]:
            for polar in config["outcomes"]["polarities"]:
                for region in ("STN", "SNr"):
                    for band in entry["bands"]:
                        records.append(
                            {
                                "CorticalRegion": cortical_region,
                                "Polar": polar,
                                "OutcomeDomain": "local",
                                "Metric": str(entry["metric"]),
                                "FeatureOutput": str(entry["feature_output"]),
                                "OutcomeScale": str(entry["outcome_scale"]),
                                "Region": region,
                                "PairRegion": "",
                                "Band": str(band),
                                "Lat": config["outcomes"]["laterality"],
                            }
                        )
    models = pd.DataFrame(records)
    models.insert(
        0,
        "ModelID",
        [f"CALMM{index:06d}" for index in range(1, len(models) + 1)],
    )
    models["Formula"] = settings["formula"]
    models["PredictorColumn"] = settings["predictor_column"]
    return models


def _membership(values: Iterable[object]) -> str:
    return ";".join(sorted({str(value) for value in values}))


def attach_models_and_support(
    rows: pd.DataFrame,
    models: pd.DataFrame,
    config: StructuralConnectivityConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Attach model identifiers and record patient/contact/Phase support."""

    model_rows = rows.merge(
        models[["ModelID", *MODEL_KEY]],
        on=MODEL_KEY,
        how="left",
        validate="many_to_one",
    )
    if model_rows["ModelID"].isna().any():
        raise ValueError("Prepared cortical-allocation rows contain an unknown cell.")
    phases = tuple(config["inference"]["cortical_allocation_lmm"]["phase_levels"])
    minimum_n = int(config["inference"]["cortical_allocation_lmm"]["minimum_n"])
    pairing_records: list[dict[str, Any]] = []
    for keys, group in model_rows.groupby(
        ["ModelID", "ID", "ContactUnitID"],
        sort=True,
        dropna=False,
        observed=True,
    ):
        model_id, subject_id, contact_unit = keys
        eligible = group.loc[group["RowEligibilityStatus"].eq("eligible")]
        observed = tuple(
            phase for phase in phases if phase in set(eligible["Phase"].astype(str))
        )
        pairing_records.append(
            {
                "ModelID": model_id,
                "ID": subject_id,
                "ContactUnitID": contact_unit,
                "ObservedPhases": ";".join(observed),
                "MissingPhases": ";".join(
                    phase for phase in phases if phase not in observed
                ),
                "NObservedPhases": len(observed),
                "CompletePhase": observed == phases,
                "Included": bool(observed),
                "ExclusionReason": (
                    ""
                    if observed
                    else _membership(group["RowEligibilityStatus"].astype(str))
                ),
            }
        )
    pairing = pd.DataFrame(pairing_records)

    support_records: list[dict[str, Any]] = []
    for model_id, group in model_rows.groupby("ModelID", sort=False, observed=True):
        cell = group.loc[group["RowEligibilityStatus"].eq("eligible")]
        phase_membership = tuple(
            phase for phase in phases if phase in set(cell["Phase"].astype(str))
        )
        n_ids = int(cell["ID"].nunique())
        n_x = int(cell["RelativeAllocation"].nunique(dropna=True))
        if n_ids < minimum_n:
            status = "insufficient_n"
        elif phase_membership != phases:
            status = "missing_phase"
        elif n_x < 2:
            status = "constant_predictor"
        else:
            status = "eligible"
        contact_rows = pairing.loc[pairing["ModelID"].eq(model_id)]
        support_records.append(
            {
                "ModelID": model_id,
                "ModelEligibilityStatus": status,
                "n_rows": int(len(cell)),
                "n_ID": n_ids,
                "n_ContactUnitID": int(
                    cell[["ID", "ContactUnitID"]].drop_duplicates().shape[0]
                ),
                "n_complete_ContactUnitID": int(contact_rows["CompletePhase"].sum()),
                "n_incomplete_ContactUnitID": int(
                    contact_rows["Included"].sum() - contact_rows["CompletePhase"].sum()
                ),
                "n_unique_RelativeAllocation": n_x,
                "RelativeAllocationMin": (
                    float(cell["RelativeAllocation"].min()) if len(cell) else np.nan
                ),
                "RelativeAllocationMax": (
                    float(cell["RelativeAllocation"].max()) if len(cell) else np.nan
                ),
                "PhaseMembership": ";".join(phase_membership),
                "PatientIDs": _membership(cell["ID"]),
                "ContactUnitMembership": _membership(
                    f"{row.ID}:{row.ContactUnitID}"
                    for row in cell[["ID", "ContactUnitID"]]
                    .drop_duplicates()
                    .itertuples(index=False)
                ),
            }
        )
    support = pd.DataFrame(support_records)
    models = models.merge(support, on="ModelID", how="left", validate="one_to_one")
    missing = models["ModelEligibilityStatus"].isna()
    models.loc[missing, "ModelEligibilityStatus"] = "insufficient_n"
    integer_columns = [
        "n_rows",
        "n_ID",
        "n_ContactUnitID",
        "n_complete_ContactUnitID",
        "n_incomplete_ContactUnitID",
        "n_unique_RelativeAllocation",
    ]
    for column in integer_columns:
        models[column] = models[column].fillna(0).astype(int)
    for column in ("PhaseMembership", "PatientIDs", "ContactUnitMembership"):
        models[column] = models[column].fillna("")
    model_rows = model_rows.sort_values(
        ["ModelID", "ID", "ContactUnitID", "Phase"], kind="mergesort"
    ).reset_index(drop=True)
    pairing = pairing.sort_values(
        ["ModelID", "ID", "ContactUnitID"], kind="mergesort"
    ).reset_index(drop=True)
    return model_rows, models, pairing


def compute_outcome_sd(lfp_rows: pd.DataFrame) -> pd.DataFrame:
    """Compute one audit SD from unexpanded signed normalized outcomes."""

    base_key = [*OUTCOME_SD_KEY, "ID", "ContactUnitID", "Phase"]
    counts = lfp_rows.groupby(
        base_key, sort=False, dropna=False, observed=True
    )["Value"].nunique(dropna=True)
    if bool((counts > 1).any()):
        bad = counts.loc[counts > 1].index[0]
        raise ValueError(f"Normalized contact outcome is not unique: {bad}")
    unique = lfp_rows.drop_duplicates(base_key, keep="first")
    unique = unique.loc[np.isfinite(unique["Value"])].copy()
    output = (
        unique.groupby(OUTCOME_SD_KEY, sort=True, dropna=False, observed=True)[
            "Value"
        ]
        .agg(ValueSD=lambda values: values.std(ddof=1), ValueN="count")
        .reset_index()
    )
    output["ValueSDStatus"] = np.where(
        np.isfinite(output["ValueSD"]) & output["ValueSD"].gt(0),
        "ok",
        "nonpositive_or_undefined_sd",
    )
    return output.sort_values(OUTCOME_SD_KEY, kind="mergesort").reset_index(
        drop=True
    )


def _model_metadata(table: pd.DataFrame, models: pd.DataFrame) -> pd.DataFrame:
    metadata = [
        "ModelID",
        *MODEL_KEY,
        "Formula",
        "PredictorColumn",
        "ModelEligibilityStatus",
        "n_rows",
        "n_ID",
        "n_ContactUnitID",
        "n_complete_ContactUnitID",
        "n_incomplete_ContactUnitID",
        "n_unique_RelativeAllocation",
        "RelativeAllocationMin",
        "RelativeAllocationMax",
        "PhaseMembership",
        "PatientIDs",
        "ContactUnitMembership",
    ]
    return models[metadata].merge(table, on="ModelID", how="left")


def _result_status(table: pd.DataFrame) -> pd.Series:
    status = table["ModelEligibilityStatus"].astype(str).copy()
    eligible = table["ModelEligibilityStatus"].eq("eligible")
    status.loc[eligible] = table.loc[eligible, "FitStatus"].fillna("fit_error")
    finite = np.isfinite(pd.to_numeric(table.get("P"), errors="coerce"))
    fitted = status.isin(["ok", "singular", "convergence_warning"])
    status.loc[fitted & ~finite] = "not_estimable"
    return status


def _p_flag(values: pd.Series) -> pd.Series:
    return pd.array(
        [value < 0.05 if np.isfinite(value) else pd.NA for value in values],
        dtype="boolean",
    )


def _attach_outcome_scaled_effects(
    table: pd.DataFrame,
    outcome_sd: pd.DataFrame,
    *,
    estimate_column: str,
) -> pd.DataFrame:
    output = table.merge(
        outcome_sd,
        on=OUTCOME_SD_KEY,
        how="left",
        validate="many_to_one",
    )
    valid_sd = np.isfinite(output["ValueSD"]) & output["ValueSD"].gt(0)
    output["EstimatePer10PctAllocation"] = output[estimate_column] * 0.10
    output["SEPer10PctAllocation"] = output["SE"] * 0.10
    output["LowerPer10PctAllocation"] = output["Lower"] * 0.10
    output["UpperPer10PctAllocation"] = output["Upper"] * 0.10
    output["OutcomeSDEffectPerUnitAllocation"] = np.where(
        valid_sd,
        output[estimate_column] / output["ValueSD"],
        np.nan,
    )
    output["OutcomeSDEffectPer10PctAllocation"] = np.where(
        valid_sd,
        output["EstimatePer10PctAllocation"] / output["ValueSD"],
        np.nan,
    )
    return output


def _apply_joint_bh(
    joint: pd.DataFrame,
    config: StructuralConnectivityConfig,
) -> pd.DataFrame:
    output = joint.copy()
    default_planned = int(
        config["inference"]["cortical_allocation_lmm"]["bh_joint_planned_tests"]
    )
    cortical_region_count = len(
        config["inference"]["cortical_allocation_lmm"]["regions"]
    )
    local_planned = {
        (str(entry["metric"]), str(entry["feature_output"])): (
            cortical_region_count * len(entry["bands"])
        )
        for entry in _registered_local_outcomes(config)
    }
    family_columns = [
        "Polar",
        "OutcomeDomain",
        "Metric",
        "FeatureOutput",
        "OutcomeScale",
        "Region",
        "PairRegion",
        "Lat",
    ]
    output["Q"] = np.nan
    output["m_planned"] = 0
    output["m_tested"] = 0
    output["FamilyKey"] = ""
    for keys, group in output.groupby(
        family_columns, sort=True, dropna=False, observed=True
    ):
        identity = dict(zip(family_columns, keys, strict=True))
        planned = (
            local_planned[(str(identity["Metric"]), str(identity["FeatureOutput"]))]
            if str(identity["OutcomeDomain"]) == "local"
            else default_planned
        )
        if len(group) != planned:
            raise ValueError(
                f"Joint-test BH family has {len(group)} rows instead of {planned}: {keys}"
            )
        family_key = json.dumps(
            identity,
            ensure_ascii=False,
            sort_keys=True,
        )
        output.loc[group.index, "Q"] = _benjamini_hochberg(group["P"])
        output.loc[group.index, "m_planned"] = planned
        output.loc[group.index, "m_tested"] = int(
            np.isfinite(pd.to_numeric(group["P"], errors="coerce")).sum()
        )
        output.loc[group.index, "FamilyKey"] = family_key
    output["P_lt_0_05"] = _p_flag(output["P"])
    output["Q_lt_0_05"] = _p_flag(output["Q"])
    return output


def postprocess_models(
    r_outputs: Mapping[str, pd.DataFrame],
    models: pd.DataFrame,
    outcome_sd: pd.DataFrame,
    config: StructuralConnectivityConfig,
) -> dict[str, pd.DataFrame]:
    """Create planned, factual model and inference result tables."""

    fit = r_outputs["model_fit"]
    model_output = models.merge(fit, on="ModelID", how="left", validate="one_to_one")
    noneligible = ~model_output["ModelEligibilityStatus"].eq("eligible")
    model_output.loc[noneligible, "FitStatus"] = model_output.loc[
        noneligible, "ModelEligibilityStatus"
    ]
    model_output.loc[
        model_output["ModelEligibilityStatus"].eq("eligible")
        & model_output["FitStatus"].isna(),
        "FitStatus",
    ] = "fit_error"
    fit_status = model_output[
        ["ModelID", "FitStatus", "InferenceStatus"]
    ].copy()

    fixed = _model_metadata(r_outputs["fixed_effects"], models).merge(
        fit_status, on="ModelID", how="left", validate="many_to_one"
    )
    fixed["Status"] = _result_status(fixed)
    fixed["P_lt_0_05"] = _p_flag(fixed["P"])

    joint_term = config["inference"]["cortical_allocation_lmm"]["joint_term"]
    observed_joint = r_outputs["omnibus_tests"].loc[
        r_outputs["omnibus_tests"]["Term"].eq(joint_term)
    ].copy()
    if observed_joint.duplicated("ModelID").any():
        raise ValueError("R output contains duplicate joint interaction tests.")
    joint_plan = models[["ModelID"]].assign(Term=joint_term)
    joint = joint_plan.merge(
        observed_joint,
        on=["ModelID", "Term"],
        how="left",
        validate="one_to_one",
    )
    joint = _model_metadata(joint, models).merge(
        fit_status, on="ModelID", how="left", validate="many_to_one"
    )
    joint["Status"] = _result_status(joint)
    joint = _apply_joint_bh(joint, config)

    phase_plan = models[["ModelID"]].merge(
        pd.DataFrame(
            {
                "Phase": config["inference"]["cortical_allocation_lmm"][
                    "phase_levels"
                ]
            }
        ),
        how="cross",
    )
    slopes = phase_plan.merge(
        r_outputs["phase_slopes"],
        on=["ModelID", "Phase"],
        how="left",
        validate="one_to_one",
    )
    slopes = _model_metadata(slopes, models).merge(
        fit_status, on="ModelID", how="left", validate="many_to_one"
    )
    slopes["Status"] = _result_status(slopes)
    slopes["P_lt_0_05"] = _p_flag(slopes["P"])
    slopes["P_holm_lt_0_05"] = _p_flag(slopes["P_holm"])
    slopes = _attach_outcome_scaled_effects(
        slopes,
        outcome_sd,
        estimate_column="Slope",
    )
    slopes["slope"] = slopes["Slope"]
    slopes["lower"] = slopes["Lower"]
    slopes["upper"] = slopes["Upper"]
    slopes["p"] = slopes["P"]
    slopes["p_holm"] = slopes["P_holm"]
    slopes["eval_at"] = slopes["EvalAt"]

    contrast_plan = models[["ModelID"]].merge(
        pd.DataFrame({"Contrast": PHASE_CONTRASTS}), how="cross"
    )
    contrasts = contrast_plan.merge(
        r_outputs["phase_slope_contrasts"],
        on=["ModelID", "Contrast"],
        how="left",
        validate="one_to_one",
    )
    contrasts = _model_metadata(contrasts, models).merge(
        fit_status, on="ModelID", how="left", validate="many_to_one"
    )
    contrasts["Status"] = _result_status(contrasts)
    contrasts["P_lt_0_05"] = _p_flag(contrasts["P"])
    contrasts = _attach_outcome_scaled_effects(
        contrasts,
        outcome_sd,
        estimate_column="Estimate",
    )

    predictions = _model_metadata(r_outputs["prediction_grid"], models).merge(
        fit_status, on="ModelID", how="left", validate="many_to_one"
    )
    predictions["Status"] = predictions["ModelEligibilityStatus"].astype(str)
    prediction_fitted = predictions["ModelEligibilityStatus"].eq("eligible")
    predictions.loc[prediction_fitted, "Status"] = predictions.loc[
        prediction_fitted, "FitStatus"
    ].fillna("fit_error")
    predictions["emmean"] = predictions["PredictedValue"]
    predictions["lower"] = predictions["Lower"]
    predictions["upper"] = predictions["Upper"]
    return {
        "models": model_output,
        "fixed_effects": fixed,
        "joint_tests": joint,
        "phase_slopes": slopes,
        "phase_slope_contrasts": contrasts,
        "prediction_grid": predictions,
    }


def _candidate_tck(
    config: StructuralConnectivityConfig,
    subject_id: str,
    region: str,
) -> Path:
    subject = config.subject(subject_id)
    hemisphere = "lh" if subject.stimulation_side == "L" else "rh"
    return (
        config.subject_leaddbs_root(subject_id)
        / "connectomics"
        / "dMRI"
        / "mrtrix_seed_target"
        / "tractograms"
        / "MNI152NLin2009bAsym"
        / hemisphere
        / "STNSNrplus"
        / "targets"
        / hemisphere
        / f"{region}.tck"
    )


def input_manifest(config: StructuralConnectivityConfig) -> pd.DataFrame:
    settings = config["inference"]["cortical_allocation_lmm"]
    records: list[dict[str, Any]] = []

    def append(
        input_type: str,
        path: Path,
        *,
        role: str,
        required: bool,
        subject_id: str = "",
        region: str = "",
    ) -> None:
        records.append(
            {
                "AnalysisID": settings["analysis_id"],
                "InputType": input_type,
                "InputRole": role,
                "RequiredForExecution": required,
                "ID": subject_id,
                "CorticalRegion": region,
                "Path": str(path),
                "Exists": path.is_file(),
            }
        )

    append("configuration", config.path, role="computational_input", required=True)
    append(
        "r_runner",
        Path(__file__).parent / "r" / "run_contact_lmm.R",
        role="computational_input",
        required=True,
    )
    for entry in _registered_local_outcomes(config):
        append(
            (
                "normalized_local_lfp:"
                f"{entry['metric']}:{entry['feature_output']}"
            ),
            Path(str(entry["path"])),
            role="computational_input",
            required=True,
        )
    for metric in config["outcomes"]["connectivity_metrics"]:
        append(
            "normalized_connectivity_lfp",
            config.path_value("connectivity_lfp_root") / metric / "mean-scalar.pkl",
            role="computational_input",
            required=True,
        )
    for subject in config.subjects:
        append(
            "mni_cortical_magnitude",
            config.mni_cortical_magnitude(subject.subject_id),
            role="computational_input",
            required=True,
            subject_id=subject.subject_id,
        )
        append(
            "mni_tstim",
            config.mni_target(subject.subject_id),
            role="computational_input",
            required=True,
            subject_id=subject.subject_id,
        )
        for region in settings["regions"]:
            append(
                "cortical_region_mask",
                config.cortical_region_mask(subject.subject_id, region),
                role="computational_input",
                required=True,
                subject_id=subject.subject_id,
                region=region,
            )
            append(
                "candidate_region_tractography",
                _candidate_tck(config, subject.subject_id, region),
                role="candidate_selection_provenance",
                required=False,
                subject_id=subject.subject_id,
                region=region,
            )
    return pd.DataFrame(records)


def tests_manifest(outputs: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    joint_columns = [
        "ModelID",
        *MODEL_KEY,
        "Term",
        "F",
        "P",
        "Q",
        "m_planned",
        "m_tested",
        "FamilyKey",
        "Status",
    ]
    for row in outputs["joint_tests"][joint_columns].itertuples(index=False):
        records.append(
            {
                **row._asdict(),
                "ResultTable": "joint_tests",
                "TestType": "phase_by_relative_allocation_joint",
                "TestLevel": row.Term,
                "Estimate": np.nan,
                "Statistic": row.F,
                "CorrectionMethod": "BH",
            }
        )
    slope_p = pd.to_numeric(outputs["phase_slopes"]["P"], errors="coerce")
    slope_tested = (
        pd.Series(
            np.isfinite(slope_p),
            index=outputs["phase_slopes"].index,
        )
        .groupby(outputs["phase_slopes"]["ModelID"], sort=False)
        .sum()
        .astype(int)
    )
    for table_name, test_type, level_column, estimate_column in (
        ("phase_slopes", "phase_specific_slope", "Phase", "Slope"),
        (
            "phase_slope_contrasts",
            "phase_slope_contrast",
            "Contrast",
            "Estimate",
        ),
    ):
        is_slope = table_name == "phase_slopes"
        columns = [
            "ModelID",
            *MODEL_KEY,
            level_column,
            estimate_column,
            "T",
            "P",
            "Status",
        ]
        if is_slope:
            columns.append("P_holm")
        for row in outputs[table_name][columns].itertuples(index=False):
            values = row._asdict()
            model_id = values["ModelID"]
            records.append(
                {
                    **{key: values[key] for key in ["ModelID", *MODEL_KEY]},
                    "ResultTable": table_name,
                    "TestType": test_type,
                    "TestLevel": values[level_column],
                    "Estimate": values[estimate_column],
                    "Statistic": values["T"],
                    "P": values["P"],
                    "Q": np.nan,
                    "P_holm": values["P_holm"] if is_slope else np.nan,
                    "m_planned": 3 if is_slope else pd.NA,
                    "m_tested": (
                        int(slope_tested.get(model_id, 0))
                        if is_slope
                        else pd.NA
                    ),
                    "FamilyKey": (
                        json.dumps({"ModelID": model_id}, sort_keys=True)
                        if is_slope
                        else ""
                    ),
                    "CorrectionMethod": "holm" if is_slope else "none",
                    "Status": values["Status"],
                }
            )
    table = pd.DataFrame(records)
    table.insert(
        0,
        "TestID",
        [f"CATEST{index:06d}" for index in range(1, len(table) + 1)],
    )
    return table


def _output_manifest(
    tables: Mapping[str, pd.DataFrame],
    config: StructuralConnectivityConfig,
) -> pd.DataFrame:
    root = config.cortical_allocation_lmm_root
    records = []
    for published_name, relative_path in CORTICAL_ALLOCATION_OUTPUTS.items():
        key = published_name.removeprefix("cortical_allocation_").removesuffix(".csv")
        table = tables.get(key)
        records.append(
            {
                "AnalysisID": config["inference"]["cortical_allocation_lmm"][
                    "analysis_id"
                ],
                "OutputType": key,
                "RelativePath": relative_path,
                "Path": str(root / relative_path),
                "Rows": int(len(table)) if table is not None else len(CORTICAL_ALLOCATION_OUTPUTS),
                "Columns": int(len(table.columns)) if table is not None else 7,
                "Status": "complete",
            }
        )
    return pd.DataFrame(records)


def _write_outputs(
    tables: Mapping[str, pd.DataFrame],
    config: StructuralConnectivityConfig,
    *,
    overwrite: bool,
) -> None:
    root = config.cortical_allocation_lmm_root
    for published_name, relative_path in CORTICAL_ALLOCATION_OUTPUTS.items():
        key = published_name.removeprefix("cortical_allocation_").removesuffix(".csv")
        atomic_csv(tables[key], root / relative_path, overwrite=overwrite)


def run_cortical_allocation_lmm(
    config: StructuralConnectivityConfig,
    *,
    dry_run: bool,
    overwrite: bool,
) -> dict[str, Any]:
    """Compute four-region allocation, fit models, and write registered tables."""

    root = config.cortical_allocation_lmm_root
    planned_paths = [root / path for path in CORTICAL_ALLOCATION_OUTPUTS.values()]
    existing = [path for path in planned_paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Cortical-allocation output exists; pass --overwrite to replace it: "
            f"{existing[0]}"
        )
    allocation = compute_regional_field_allocation(config)
    lfp_rows = prepare_lfp_rows(config)
    expanded = expand_model_rows(lfp_rows, allocation)
    model_rows, models, pairing = attach_models_and_support(
        expanded, planned_models(config), config
    )
    outcome_sd = compute_outcome_sd(lfp_rows)
    summary = {
        "status": "ready" if dry_run else "complete",
        "regional_field_rows": int(len(allocation)),
        "lfp_rows": int(len(lfp_rows)),
        "contact_model_rows": int(len(model_rows)),
        "planned_models": int(len(models)),
        "eligible_models": int(models["ModelEligibilityStatus"].eq("eligible").sum()),
        "writes": 0,
    }
    if dry_run:
        return summary

    r_outputs = _run_r_models(
        model_rows,
        models,
        config,
        settings_key="cortical_allocation_lmm",
        predictor_column="RelativeAllocation",
        temp_path_key="cortical_allocation_lmm_temp_root",
    )
    processed = postprocess_models(r_outputs, models, outcome_sd, config)
    tables: dict[str, pd.DataFrame] = {
        "regional_field": allocation,
        "contact_model_rows": model_rows,
        "contact_pairing": pairing,
        "outcome_sd": outcome_sd,
        **processed,
        "inputs": input_manifest(config),
    }
    tables["tests"] = tests_manifest(tables)
    tables["outputs"] = _output_manifest(tables, config)
    _write_outputs(tables, config, overwrite=overwrite)
    summary["fit_status_counts"] = (
        processed["models"]["FitStatus"]
        .fillna("missing")
        .value_counts()
        .sort_index()
        .to_dict()
    )
    summary["joint_test_rows"] = int(len(processed["joint_tests"]))
    summary["phase_slope_rows"] = int(len(processed["phase_slopes"]))
    summary["prediction_grid_rows"] = int(len(processed["prediction_grid"]))
    summary["writes"] = int(len(CORTICAL_ALLOCATION_OUTPUTS))
    return summary
