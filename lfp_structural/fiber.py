"""Orchestrate four-source Region-conditioned streamline Lift extraction."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import tempfile
from typing import Any, Mapping

import nibabel as nib
import numpy as np
import pandas as pd

from .artifacts import atomic_csv, atomic_json, atomic_npz
from .config import ConnectomeConfig, StructuralConnectivityConfig
from .connectomes import ConnectomeSummary, open_connectome
from .field_exposure import FieldSpec
from .geometry import forward_warp_scalar
from .lift import (
    FieldExposureExtraction,
    MembershipExtraction,
    compute_region_conditioned_field_weighted_lift,
    compute_region_conditioned_lift,
    extract_field_exposures,
    extract_memberships,
)
from .masks import MaskSpec


def _mask_id(subject_id: str, kind: str, name: str = "") -> str:
    suffix = f"__{name}" if name else ""
    return f"{subject_id}__{kind}{suffix}"


def _field_id(subject_id: str) -> str:
    return f"{subject_id}__field__corticalMagnE"


def ribbon_masked_magnitude(
    magnitude: np.ndarray,
    cortical_ribbon: np.ndarray,
) -> np.ndarray:
    """Retain finite nonnegative magnitude only inside the cortical ribbon."""

    values = np.asarray(magnitude, dtype=np.float32)
    ribbon = np.asarray(cortical_ribbon, dtype=bool)
    if values.shape != ribbon.shape:
        raise ValueError("Magnitude and cortical ribbon must share one grid.")
    if not np.all(np.isfinite(values[ribbon])):
        raise ValueError("Cortical ribbon contains nonfinite magnitude values.")
    if np.any(values[ribbon] < 0):
        raise ValueError("Cortical ribbon contains negative magnitude values.")
    return np.where(ribbon, values, 0.0).astype(np.float32, copy=False)


def _prepare_mni_cortical_magnitude(
    config: StructuralConnectivityConfig,
    subject_id: str,
    *,
    overwrite: bool,
) -> Path:
    """Derive one MNI scalar map from existing field and ribbon artifacts."""

    magnitude_path = config.native_magnitude(subject_id)
    ribbon_path = config.native_cortical_ribbon(subject_id)
    magnitude_image = nib.load(magnitude_path)
    ribbon_image = nib.load(ribbon_path)
    if magnitude_image.shape != ribbon_image.shape or not np.allclose(
        magnitude_image.affine,
        ribbon_image.affine,
        atol=1e-7,
        rtol=0,
    ):
        raise ValueError("Native magnitude and cortical ribbon do not share one grid.")
    masked = ribbon_masked_magnitude(
        np.asanyarray(magnitude_image.dataobj),
        np.asanyarray(ribbon_image.dataobj) > 0,
    )
    output_path = config.mni_cortical_magnitude(subject_id)
    with tempfile.TemporaryDirectory(prefix="structural-native-field-") as directory:
        temporary = Path(directory) / "cortical_magnE.nii.gz"
        header = magnitude_image.header.copy()
        header.set_data_dtype(np.float32)
        nib.save(
            nib.Nifti1Image(masked, magnitude_image.affine, header),
            temporary,
        )
        positive_voxels, maximum_value = forward_warp_scalar(
            temporary,
            output_path,
            deformation_path=config.forward_deformation(subject_id),
            reference_path=config.mni_reference,
            leaddbs_root=config.leaddbs_root,
            overwrite=overwrite,
        )
    manifest_path = config.field_exposure_manifest(subject_id)
    atomic_json(
        {
            "status": "complete",
            "subject_id": subject_id,
            "stimulation_side": config.subject(subject_id).stimulation_side,
            "source_magnitude_path": str(magnitude_path),
            "source_cortical_ribbon_path": str(ribbon_path),
            "forward_deformation_path": str(config.forward_deformation(subject_id)),
            "mni_reference_path": str(config.mni_reference),
            "mni_cortical_magnitude_path": str(output_path),
            "interpolation_to_mni": "Linear",
            "outside_native_cortical_ribbon": 0.0,
            "units": "V/m",
            "fiber_aggregation": "maximum_traversed_voxel",
            "positive_mni_voxels": positive_voxels,
            "maximum_mni_value_v_per_m": maximum_value,
            "upstream_recomputed": False,
        },
        manifest_path,
        overwrite=overwrite,
    )
    return output_path


def _subject_field_spec(
    config: StructuralConnectivityConfig,
    subject_id: str,
) -> FieldSpec:
    return FieldSpec(
        field_id=_field_id(subject_id),
        path=config.mni_cortical_magnitude(subject_id),
    )


def _subject_mask_specs(
    config: StructuralConnectivityConfig,
    subject_id: str,
    subject_channels: pd.DataFrame,
) -> list[MaskSpec]:
    threshold = float(config["masks"]["region_probability_threshold"])
    specs = [
        MaskSpec(
            _mask_id(subject_id, "region", "STN"),
            config.region_mask(subject_id, "STN"),
            threshold,
            "region",
        ),
        MaskSpec(
            _mask_id(subject_id, "region", "SNr"),
            config.region_mask(subject_id, "SNr"),
            threshold,
            "region",
        ),
        MaskSpec(
            _mask_id(subject_id, "target"),
            config.mni_target(subject_id),
            None,
            "stimulation_target",
            allow_empty=True,
        ),
    ]
    for channel in subject_channels["Channel"].astype(str):
        specs.append(
            MaskSpec(
                _mask_id(subject_id, "seed", channel),
                config.mni_contact_seed(subject_id, channel),
                None,
                "contact_seed",
            )
        )
    return specs


def _individual_source(
    source: ConnectomeConfig,
    config: StructuralConnectivityConfig,
    subject_id: str,
) -> tuple[Path, str]:
    if source.connectome_id != "individualized_dti" or source.representation != "tck":
        raise ValueError("The individualized source registration is invalid.")
    return config.individualized_tck(subject_id), "tck"


def _archive_keys(
    subject_id: str,
    channels: pd.DataFrame,
) -> dict[str, str]:
    keys = {
        _mask_id(subject_id, "region", "STN"): "region_STN",
        _mask_id(subject_id, "region", "SNr"): "region_SNr",
        _mask_id(subject_id, "target"): "target",
    }
    for channel in channels["Channel"].astype(str):
        keys[_mask_id(subject_id, "seed", channel)] = f"seed_{channel}"
    return keys


def _field_archive_keys() -> dict[str, str]:
    return {
        "fiber_ids": "field_exposed_fiber_ids",
        "peak_values": "field_peak_e_v_per_m",
    }


def compute_subject_lift_rows(
    *,
    subject_id: str,
    stimulation_side: str,
    normalization_approval: float,
    source_summary: ConnectomeSummary,
    memberships: Mapping[str, np.ndarray],
    channels: pd.DataFrame,
    membership_path: Path,
) -> list[dict[str, Any]]:
    """Compute assigned-Region channel rows from one extracted membership set."""

    target_key = _mask_id(subject_id, "target")
    target_ids = memberships[target_key]
    stn_ids = memberships[_mask_id(subject_id, "region", "STN")]
    snr_ids = memberships[_mask_id(subject_id, "region", "SNr")]
    overlap_count = int(np.intersect1d(stn_ids, snr_ids, assume_unique=True).size)
    archive_keys = _archive_keys(subject_id, channels)
    rows: list[dict[str, Any]] = []
    for channel_row in channels.itertuples(index=False):
        channel = str(channel_row.Channel)
        region = str(channel_row.Region)
        region_key = _mask_id(subject_id, "region", region)
        seed_key = _mask_id(subject_id, "seed", channel)
        statistic = compute_region_conditioned_lift(
            memberships[region_key],
            memberships[seed_key],
            target_ids,
        )
        rows.append(
            {
                "ID": subject_id,
                "StimSide": stimulation_side,
                "NormalizationApproval": normalization_approval,
                "ConnectomeSource": source_summary.connectome_id,
                "ConnectomeRepresentation": source_summary.representation,
                "ConnectomePath": source_summary.path,
                "NSourceFibers": source_summary.n_fibers,
                "FiberWeight": 1,
                "Channel": channel,
                "Region": region,
                "CatalogSources": channel_row.CatalogSources,
                "NAll": statistic.n_all,
                "NSeed": statistic.n_seed,
                "NTarget": statistic.n_target,
                "NSeedTarget": statistic.n_seed_target,
                "SeedTargetFraction": statistic.seed_target_fraction,
                "TargetBackgroundFraction": statistic.target_background_fraction,
                "Lift": statistic.lift,
                "LiftStatus": statistic.status,
                "RegionOverlapCount": overlap_count,
                "MembershipPath": str(membership_path),
                "RegionMembershipKey": archive_keys[region_key],
                "SeedMembershipKey": archive_keys[seed_key],
                "TargetMembershipKey": archive_keys[target_key],
                "Intersection": "any_segment_occupied_voxel",
            }
        )
    return rows


def compute_subject_field_weighted_rows(
    *,
    subject_id: str,
    stimulation_side: str,
    source_summary: ConnectomeSummary,
    memberships: Mapping[str, np.ndarray],
    field_exposure_ids: np.ndarray,
    field_exposure_values: np.ndarray,
    field_path: Path,
    field_manifest_path: Path,
    field_exposure_path: Path,
    channels: pd.DataFrame,
    field_threshold_v_per_m: float,
) -> list[dict[str, Any]]:
    """Compute threshold-gated field exposure for existing memberships."""

    field_archive_keys = _field_archive_keys()
    rows: list[dict[str, Any]] = []
    for channel_row in channels.itertuples(index=False):
        channel = str(channel_row.Channel)
        region = str(channel_row.Region)
        region_key = _mask_id(subject_id, "region", region)
        seed_key = _mask_id(subject_id, "seed", channel)
        weighted = compute_region_conditioned_field_weighted_lift(
            memberships[region_key],
            memberships[seed_key],
            field_exposure_ids,
            field_exposure_values,
            field_threshold_v_per_m=field_threshold_v_per_m,
        )
        rows.append(
            {
                "ID": subject_id,
                "StimSide": stimulation_side,
                "ConnectomeSource": source_summary.connectome_id,
                "Channel": channel,
                "Region": region,
                "NFieldExposedAll": weighted.n_field_exposed_all,
                "NFieldExposedSeed": weighted.n_field_exposed_seed,
                "FieldThreshold_Vm": weighted.field_threshold_v_per_m,
                "NFieldAboveThresholdAll": weighted.n_field_above_threshold_all,
                "NFieldAboveThresholdSeed": weighted.n_field_above_threshold_seed,
                "RegionSumPeakE_Vm": weighted.region_sum_peak_e_v_per_m,
                "SeedSumPeakE_Vm": weighted.seed_sum_peak_e_v_per_m,
                "RegionMeanPeakE_Vm": weighted.region_mean_peak_e_v_per_m,
                "SeedMeanPeakE_Vm": weighted.seed_mean_peak_e_v_per_m,
                "RegionThresholdedSumPeakE_Vm": (
                    weighted.region_thresholded_sum_peak_e_v_per_m
                ),
                "SeedThresholdedSumPeakE_Vm": (
                    weighted.seed_thresholded_sum_peak_e_v_per_m
                ),
                "RegionThresholdedMeanPeakE_Vm": (
                    weighted.region_thresholded_mean_peak_e_v_per_m
                ),
                "SeedThresholdedMeanPeakE_Vm": (
                    weighted.seed_thresholded_mean_peak_e_v_per_m
                ),
                "FieldWeightedLift": weighted.field_weighted_lift,
                "FieldWeightedLiftStatus": weighted.status,
                "FieldPath": str(field_path),
                "FieldManifestPath": str(field_manifest_path),
                "FieldExposurePath": str(field_exposure_path),
                "FieldExposureFiberIdsKey": field_archive_keys["fiber_ids"],
                "FieldPeakEKey": field_archive_keys["peak_values"],
                "FieldUnits": "V/m",
                "FieldAggregation": "maximum_traversed_voxel",
            }
        )
    return rows


def _write_subject_membership(
    *,
    config: StructuralConnectivityConfig,
    subject_id: str,
    source_summary: ConnectomeSummary,
    extraction: MembershipExtraction,
    channels: pd.DataFrame,
    overwrite: bool,
) -> Path:
    root = (
        config.patient_derivative_root(subject_id)
        / "fibers"
        / source_summary.connectome_id
    )
    membership_path = root / "memberships.npz"
    manifest_path = root / "membership_manifest.json"
    archive_keys = _archive_keys(subject_id, channels)
    arrays = {
        archive_key: extraction.memberships[mask_id]
        for mask_id, archive_key in archive_keys.items()
    }
    atomic_npz(arrays, membership_path, overwrite=overwrite)
    masks = [
        asdict(summary)
        for summary in extraction.masks
        if summary.mask_id in archive_keys
    ]
    atomic_json(
        {
            "status": "complete",
            "subject_id": subject_id,
            "stimulation_side": config.subject(subject_id).stimulation_side,
            "connectome": asdict(source_summary),
            "intersection": "any_segment_occupied_voxel",
            "fiber_weight": 1,
            "archive": str(membership_path),
            "archive_keys": archive_keys,
            "masks": masks,
        },
        manifest_path,
        overwrite=overwrite,
    )
    return membership_path


def _subject_rows(mapping: pd.DataFrame, subject_id: str) -> pd.DataFrame:
    rows = mapping.loc[
        mapping["ID"].astype(str).eq(subject_id),
        ["ID", "StimSide", "Channel", "Region", "CatalogSources"],
    ].drop_duplicates()
    if rows.empty:
        raise ValueError(f"Contact mapping has no channels for {subject_id}.")
    return rows.sort_values("Channel", kind="mergesort").reset_index(drop=True)


def _load_contact_mapping(config: StructuralConnectivityConfig) -> pd.DataFrame:
    path = config.output_path("contact_mapping")
    if not path.is_file():
        raise FileNotFoundError(f"Contact mapping has not been generated: {path}")
    mapping = pd.read_csv(path)
    required = {"ID", "StimSide", "Channel", "Region", "CatalogSources", "Status"}
    missing = sorted(required.difference(mapping.columns))
    if missing:
        raise KeyError(f"Contact mapping is missing columns: {missing}")
    if not mapping["Status"].eq("complete").all():
        raise ValueError("Contact mapping contains incomplete rows.")
    return mapping


def _planned_membership_paths(
    config: StructuralConnectivityConfig,
) -> list[Path]:
    return [
        config.membership_archive(subject.subject_id, source.connectome_id)
        for source in config.connectomes
        for subject in config.subjects
    ]


def _load_subject_memberships(
    *,
    config: StructuralConnectivityConfig,
    subject_id: str,
    source_summary: ConnectomeSummary,
    channels: pd.DataFrame,
) -> Mapping[str, np.ndarray]:
    """Load the existing binary archive at the file-parsing boundary."""

    path = config.membership_archive(subject_id, source_summary.connectome_id)
    archive_keys = _archive_keys(subject_id, channels)
    memberships: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(set(archive_keys.values()).difference(archive.files))
        if missing:
            raise KeyError(f"Membership archive is missing arrays: {path}: {missing}")
        for mask_id, archive_key in archive_keys.items():
            values = np.asarray(archive[archive_key], dtype=np.int64)
            if values.ndim != 1:
                raise ValueError(f"Membership array is not one-dimensional: {path}")
            if values.size and (
                values[0] < 1
                or values[-1] > source_summary.n_fibers
                or np.any(np.diff(values) <= 0)
            ):
                raise ValueError(
                    f"Membership fiber IDs are invalid: {path}:{archive_key}"
                )
            values.setflags(write=False)
            memberships[mask_id] = values
    return memberships


def _write_subject_field_exposure(
    *,
    config: StructuralConnectivityConfig,
    subject_id: str,
    source_summary: ConnectomeSummary,
    extraction: FieldExposureExtraction,
    overwrite: bool,
) -> Path:
    """Write positive continuous exposure without changing binary memberships."""

    field_keys = _field_archive_keys()
    field_id = _field_id(subject_id)
    exposure = extraction.exposures[field_id]
    summaries = [
        summary for summary in extraction.fields if summary.field_id == field_id
    ]
    if len(summaries) != 1:
        raise ValueError(f"Expected one continuous field for {subject_id}.")
    exposure_path = config.field_exposure_archive(
        subject_id, source_summary.connectome_id
    )
    atomic_npz(
        {
            field_keys["fiber_ids"]: exposure.fiber_ids,
            field_keys["peak_values"]: exposure.peak_values,
        },
        exposure_path,
        overwrite=overwrite,
    )
    atomic_json(
        {
            "status": "complete",
            "subject_id": subject_id,
            "stimulation_side": config.subject(subject_id).stimulation_side,
            "connectome": asdict(source_summary),
            "continuous_field": asdict(summaries[0]),
            "field_units": "V/m",
            "field_aggregation": "maximum_traversed_voxel",
            "intersection": "any_segment_occupied_voxel",
            "archive": str(exposure_path),
            "archive_keys": field_keys,
            "positive_exposure_fibers": int(exposure.fiber_ids.size),
            "binary_membership_archive": str(
                config.membership_archive(subject_id, source_summary.connectome_id)
            ),
            "binary_membership_modified": False,
        },
        config.field_exposure_archive_manifest(
            subject_id, source_summary.connectome_id
        ),
        overwrite=overwrite,
    )
    return exposure_path


def _load_subject_field_exposure(
    path: Path,
    *,
    n_source_fibers: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one raw positive-exposure archive at the file boundary."""

    keys = _field_archive_keys()
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(set(keys.values()).difference(archive.files))
        if missing:
            raise KeyError(f"Field exposure archive is missing arrays: {path}: {missing}")
        fiber_ids = np.asarray(archive[keys["fiber_ids"]], dtype=np.int64)
        peak_values = np.asarray(archive[keys["peak_values"]], dtype=np.float64)
    if fiber_ids.ndim != 1 or peak_values.ndim != 1 or fiber_ids.size != peak_values.size:
        raise ValueError(f"Field exposure arrays are not aligned vectors: {path}")
    if fiber_ids.size and (
        fiber_ids[0] < 1
        or fiber_ids[-1] > n_source_fibers
        or np.any(np.diff(fiber_ids) <= 0)
    ):
        raise ValueError(f"Field exposure fiber IDs are invalid: {path}")
    if not np.all(np.isfinite(peak_values)) or np.any(peak_values <= 0):
        raise ValueError(f"Field exposure values must be finite and positive: {path}")
    fiber_ids.setflags(write=False)
    peak_values.setflags(write=False)
    return fiber_ids, peak_values


def _source_summary_from_channel_rows(
    rows: pd.DataFrame,
    *,
    source_id: str,
    subject_id: str,
) -> ConnectomeSummary:
    """Recover source metadata from the registered binary channel rows."""

    selected = rows.loc[
        rows["ConnectomeSource"].eq(source_id) & rows["ID"].eq(subject_id)
    ]
    if selected.empty:
        raise ValueError(f"Binary Lift has no rows for {subject_id}: {source_id}")
    values: dict[str, Any] = {}
    for column in ("ConnectomeRepresentation", "ConnectomePath", "NSourceFibers"):
        unique = selected[column].drop_duplicates().tolist()
        if len(unique) != 1:
            raise ValueError(
                f"Binary Lift metadata is not unique for {subject_id}: "
                f"{source_id}: {column}"
            )
        values[column] = unique[0]
    return ConnectomeSummary(
        connectome_id=source_id,
        representation=str(values["ConnectomeRepresentation"]),
        path=str(values["ConnectomePath"]),
        n_fibers=int(values["NSourceFibers"]),
        fiber_id_base=1,
        weighted=False,
    )


def _finalize_field_weighted_table(
    *,
    binary: pd.DataFrame,
    weighted_rows: list[dict[str, Any]],
    config: StructuralConnectivityConfig,
    overwrite: bool,
) -> pd.DataFrame:
    """Merge one-to-one thresholded rows into the unchanged binary table."""

    keys = ["ID", "StimSide", "ConnectomeSource", "Channel", "Region"]
    weighted = pd.DataFrame(weighted_rows)
    missing_columns = sorted(set(keys).difference(weighted.columns))
    if missing_columns:
        raise KeyError(
            f"Field-weighted Lift table is missing columns: {missing_columns}"
        )
    if weighted.duplicated(keys, keep=False).any():
        raise ValueError("Field-weighted Lift table has duplicate channel keys.")
    key_check = binary[keys].merge(
        weighted[keys],
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not key_check["_merge"].eq("both").all():
        raise ValueError("Binary and field-weighted channel keys do not match.")
    table = binary.merge(
        weighted,
        on=keys,
        how="left",
        validate="one_to_one",
    )
    table = table.sort_values(
        ["ConnectomeSource", "ID", "Channel"],
        kind="mergesort",
    ).reset_index(drop=True)
    atomic_csv(
        table,
        config.output_path("channel_field_weighted_lift"),
        overwrite=overwrite,
    )
    return table


def run_field_weighted_lift_reaggregation(
    config: StructuralConnectivityConfig,
    *,
    dry_run: bool,
    overwrite: bool,
) -> pd.DataFrame:
    """Reaggregate thresholded Lift from immutable exposure archives."""

    mapping = _load_contact_mapping(config)
    binary_path = config.output_path("channel_lift")
    output_path = config.output_path("channel_field_weighted_lift")
    required_paths = [binary_path]
    for source in config.connectomes:
        for subject in config.subjects:
            subject_id = subject.subject_id
            required_paths.extend(
                [
                    config.membership_archive(subject_id, source.connectome_id),
                    config.field_exposure_archive(subject_id, source.connectome_id),
                    config.mni_cortical_magnitude(subject_id),
                    config.field_exposure_manifest(subject_id),
                ]
            )
    missing = [path for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Field-weighted reaggregation input does not exist: {missing[0]}"
        )
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            "Field-weighted Lift output exists; pass --overwrite to replace it: "
            f"{output_path}"
        )
    if dry_run:
        return pd.DataFrame(
            {
                "ConnectomeSource": [
                    source.connectome_id for source in config.connectomes
                ],
                "Subjects": [len(config.subjects)] * len(config.connectomes),
                "FieldThreshold_Vm": [
                    float(config["masks"]["field_threshold_v_per_m"])
                ]
                * len(config.connectomes),
                "Status": ["ready"] * len(config.connectomes),
            }
        )

    binary = pd.read_csv(binary_path)
    weighted_rows: list[dict[str, Any]] = []
    threshold = float(config["masks"]["field_threshold_v_per_m"])
    for source in config.connectomes:
        for subject in config.subjects:
            subject_id = subject.subject_id
            channels = _subject_rows(mapping, subject_id)
            source_summary = _source_summary_from_channel_rows(
                binary,
                source_id=source.connectome_id,
                subject_id=subject_id,
            )
            memberships = _load_subject_memberships(
                config=config,
                subject_id=subject_id,
                source_summary=source_summary,
                channels=channels,
            )
            exposure_path = config.field_exposure_archive(
                subject_id, source.connectome_id
            )
            exposure_ids, exposure_values = _load_subject_field_exposure(
                exposure_path,
                n_source_fibers=source_summary.n_fibers,
            )
            weighted_rows.extend(
                compute_subject_field_weighted_rows(
                    subject_id=subject_id,
                    stimulation_side=subject.stimulation_side,
                    source_summary=source_summary,
                    memberships=memberships,
                    field_exposure_ids=exposure_ids,
                    field_exposure_values=exposure_values,
                    field_path=config.mni_cortical_magnitude(subject_id),
                    field_manifest_path=config.field_exposure_manifest(subject_id),
                    field_exposure_path=exposure_path,
                    channels=channels,
                    field_threshold_v_per_m=threshold,
                )
            )
    return _finalize_field_weighted_table(
        binary=binary,
        weighted_rows=weighted_rows,
        config=config,
        overwrite=overwrite,
    )


def run_lift_extraction(
    config: StructuralConnectivityConfig,
    *,
    dry_run: bool,
    overwrite: bool,
) -> pd.DataFrame:
    """Run each individualized source once and each normative source once."""

    mapping = _load_contact_mapping(config)
    required_paths: list[Path] = [
        config.sextant_python_root / "sextant" / "shared" / "fiber_membership.py"
    ]
    for subject in config.subjects:
        subject_id = subject.subject_id
        required_paths.extend(
            [
                config.mni_target(subject_id),
                config.region_mask(subject_id, "STN"),
                config.region_mask(subject_id, "SNr"),
            ]
        )
        channels = _subject_rows(mapping, subject_id)
        required_paths.extend(
            config.mni_contact_seed(subject_id, channel)
            for channel in channels["Channel"].astype(str)
        )
        required_paths.append(config.individualized_tck(subject_id))
    required_paths.extend(
        source.path for source in config.connectomes if source.path is not None
    )
    missing = [path for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Lift input does not exist: {missing[0]}")
    planned_outputs = _planned_membership_paths(config) + [
        config.output_path("channel_lift")
    ]
    existing = [path for path in planned_outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Lift output exists; pass --overwrite to replace it: {existing[0]}"
        )
    if dry_run:
        return pd.DataFrame(
            {
                "ConnectomeSource": [
                    source.connectome_id for source in config.connectomes
                ],
                "Representation": [
                    source.representation for source in config.connectomes
                ],
                "Subjects": [len(config.subjects)] * len(config.connectomes),
                "Status": ["ready"] * len(config.connectomes),
            }
        )

    all_rows: list[dict[str, Any]] = []
    for source in config.connectomes:
        if source.representation == "tck":
            for subject in config.subjects:
                subject_id = subject.subject_id
                channels = _subject_rows(mapping, subject_id)
                path, representation = _individual_source(source, config, subject_id)
                connectome = open_connectome(
                    path,
                    connectome_id=source.connectome_id,
                    representation=representation,
                    sextant_python_root=config.sextant_python_root,
                )
                extraction = extract_memberships(
                    connectome,
                    _subject_mask_specs(config, subject_id, channels),
                    sextant_python_root=config.sextant_python_root,
                    chunk_size=int(config["execution"]["tck_chunk_size"]),
                )
                membership_path = _write_subject_membership(
                    config=config,
                    subject_id=subject_id,
                    source_summary=connectome.summary,
                    extraction=extraction,
                    channels=channels,
                    overwrite=overwrite,
                )
                all_rows.extend(
                    compute_subject_lift_rows(
                        subject_id=subject_id,
                        stimulation_side=subject.stimulation_side,
                        normalization_approval=subject.normalization_approval,
                        source_summary=connectome.summary,
                        memberships=extraction.memberships,
                        channels=channels,
                        membership_path=membership_path,
                    )
                )
            continue

        if source.path is None:
            raise ValueError(f"Normative source has no path: {source.connectome_id}")
        combined_specs: list[MaskSpec] = []
        channels_by_subject: dict[str, pd.DataFrame] = {}
        for subject in config.subjects:
            channels = _subject_rows(mapping, subject.subject_id)
            channels_by_subject[subject.subject_id] = channels
            combined_specs.extend(
                _subject_mask_specs(config, subject.subject_id, channels)
            )
        connectome = open_connectome(
            source.path,
            connectome_id=source.connectome_id,
            representation=source.representation,
            sextant_python_root=config.sextant_python_root,
        )
        extraction = extract_memberships(
            connectome,
            combined_specs,
            sextant_python_root=config.sextant_python_root,
            chunk_size=int(config["execution"]["hdf5_chunk_size"]),
        )
        for subject in config.subjects:
            subject_id = subject.subject_id
            channels = channels_by_subject[subject_id]
            membership_path = _write_subject_membership(
                config=config,
                subject_id=subject_id,
                source_summary=connectome.summary,
                extraction=extraction,
                channels=channels,
                overwrite=overwrite,
            )
            all_rows.extend(
                compute_subject_lift_rows(
                    subject_id=subject_id,
                    stimulation_side=subject.stimulation_side,
                    normalization_approval=subject.normalization_approval,
                    source_summary=connectome.summary,
                    memberships=extraction.memberships,
                    channels=channels,
                    membership_path=membership_path,
                )
            )

    table = (
        pd.DataFrame(all_rows)
        .sort_values(["ConnectomeSource", "ID", "Channel"], kind="mergesort")
        .reset_index(drop=True)
    )
    atomic_csv(table, config.output_path("channel_lift"), overwrite=overwrite)
    return table


def run_field_weighted_lift_extraction(
    config: StructuralConnectivityConfig,
    *,
    dry_run: bool,
    overwrite: bool,
) -> pd.DataFrame:
    """Add continuous field exposure while preserving all binary Lift artifacts."""

    mapping = _load_contact_mapping(config)
    binary_lift_path = config.output_path("channel_lift")
    required_paths: list[Path] = [
        binary_lift_path,
        config.mni_reference,
        config.sextant_python_root / "sextant" / "tractography" / "dwi_seed_target",
    ]
    for subject in config.subjects:
        subject_id = subject.subject_id
        _subject_rows(mapping, subject_id)
        required_paths.extend(
            [
                config.native_magnitude(subject_id),
                config.native_cortical_ribbon(subject_id),
                config.forward_deformation(subject_id),
                config.individualized_tck(subject_id),
            ]
        )
        required_paths.extend(
            config.membership_archive(subject_id, source.connectome_id)
            for source in config.connectomes
        )
    required_paths.extend(
        source.path for source in config.connectomes if source.path is not None
    )
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Field-weighted Lift input does not exist: {missing[0]}"
        )
    planned_outputs = [
        config.mni_cortical_magnitude(subject.subject_id) for subject in config.subjects
    ] + [
        config.field_exposure_manifest(subject.subject_id)
        for subject in config.subjects
    ]
    for source in config.connectomes:
        for subject in config.subjects:
            planned_outputs.extend(
                [
                    config.field_exposure_archive(
                        subject.subject_id,
                        source.connectome_id,
                    ),
                    config.field_exposure_archive_manifest(
                        subject.subject_id,
                        source.connectome_id,
                    ),
                ]
            )
    planned_outputs.append(config.output_path("channel_field_weighted_lift"))
    existing = [path for path in planned_outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Field-weighted Lift output exists; pass --overwrite to replace it: "
            f"{existing[0]}"
        )
    if dry_run:
        return pd.DataFrame(
            {
                "ConnectomeSource": [
                    source.connectome_id for source in config.connectomes
                ],
                "Representation": [
                    source.representation for source in config.connectomes
                ],
                "Subjects": [len(config.subjects)] * len(config.connectomes),
                "Status": ["ready"] * len(config.connectomes),
            }
        )

    keys = ["ID", "StimSide", "ConnectomeSource", "Channel", "Region"]
    binary = pd.read_csv(binary_lift_path)
    missing_columns = sorted(set(keys).difference(binary.columns))
    if missing_columns:
        raise KeyError(f"Binary Lift table is missing columns: {missing_columns}")
    if binary.duplicated(keys, keep=False).any():
        raise ValueError("Binary Lift table has duplicate channel keys.")

    field_paths = {
        subject.subject_id: _prepare_mni_cortical_magnitude(
            config,
            subject.subject_id,
            overwrite=overwrite,
        )
        for subject in config.subjects
    }
    weighted_rows: list[dict[str, Any]] = []
    for source in config.connectomes:
        if source.representation == "tck":
            for subject in config.subjects:
                subject_id = subject.subject_id
                channels = _subject_rows(mapping, subject_id)
                path, representation = _individual_source(source, config, subject_id)
                connectome = open_connectome(
                    path,
                    connectome_id=source.connectome_id,
                    representation=representation,
                    sextant_python_root=config.sextant_python_root,
                )
                memberships = _load_subject_memberships(
                    config=config,
                    subject_id=subject_id,
                    source_summary=connectome.summary,
                    channels=channels,
                )
                extraction = extract_field_exposures(
                    connectome,
                    [_subject_field_spec(config, subject_id)],
                    chunk_size=int(config["execution"]["tck_chunk_size"]),
                )
                exposure_path = _write_subject_field_exposure(
                    config=config,
                    subject_id=subject_id,
                    source_summary=connectome.summary,
                    extraction=extraction,
                    overwrite=overwrite,
                )
                exposure = extraction.exposures[_field_id(subject_id)]
                weighted_rows.extend(
                    compute_subject_field_weighted_rows(
                        subject_id=subject_id,
                        stimulation_side=subject.stimulation_side,
                        source_summary=connectome.summary,
                        memberships=memberships,
                        field_exposure_ids=exposure.fiber_ids,
                        field_exposure_values=exposure.peak_values,
                        field_path=field_paths[subject_id],
                        field_manifest_path=config.field_exposure_manifest(subject_id),
                        field_exposure_path=exposure_path,
                        channels=channels,
                        field_threshold_v_per_m=float(
                            config["masks"]["field_threshold_v_per_m"]
                        ),
                    )
                )
            continue

        if source.path is None:
            raise ValueError(f"Normative source has no path: {source.connectome_id}")
        connectome = open_connectome(
            source.path,
            connectome_id=source.connectome_id,
            representation=source.representation,
            sextant_python_root=config.sextant_python_root,
        )
        memberships_by_subject: dict[str, Mapping[str, np.ndarray]] = {}
        for subject in config.subjects:
            subject_id = subject.subject_id
            memberships_by_subject[subject_id] = _load_subject_memberships(
                config=config,
                subject_id=subject_id,
                source_summary=connectome.summary,
                channels=_subject_rows(mapping, subject_id),
            )
        extraction = extract_field_exposures(
            connectome,
            [
                _subject_field_spec(config, subject.subject_id)
                for subject in config.subjects
            ],
            chunk_size=int(config["execution"]["hdf5_chunk_size"]),
        )
        for subject in config.subjects:
            subject_id = subject.subject_id
            channels = _subject_rows(mapping, subject_id)
            exposure_path = _write_subject_field_exposure(
                config=config,
                subject_id=subject_id,
                source_summary=connectome.summary,
                extraction=extraction,
                overwrite=overwrite,
            )
            exposure = extraction.exposures[_field_id(subject_id)]
            weighted_rows.extend(
                compute_subject_field_weighted_rows(
                    subject_id=subject_id,
                    stimulation_side=subject.stimulation_side,
                    source_summary=connectome.summary,
                    memberships=memberships_by_subject[subject_id],
                    field_exposure_ids=exposure.fiber_ids,
                    field_exposure_values=exposure.peak_values,
                    field_path=field_paths[subject_id],
                    field_manifest_path=config.field_exposure_manifest(subject_id),
                    field_exposure_path=exposure_path,
                    channels=channels,
                    field_threshold_v_per_m=float(
                        config["masks"]["field_threshold_v_per_m"]
                    ),
                )
            )

    return _finalize_field_weighted_table(
        binary=binary,
        weighted_rows=weighted_rows,
        config=config,
        overwrite=overwrite,
    )
