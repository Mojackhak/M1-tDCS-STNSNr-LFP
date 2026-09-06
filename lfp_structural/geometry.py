"""Create SCRF bipolar midpoint spheres and forward-warped MNI labels."""

from __future__ import annotations

from pathlib import Path
import platform
import subprocess
import tempfile

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.io import loadmat

from .artifacts import atomic_csv, atomic_nifti, prepare_output
from .channels import build_channel_catalog
from .config import StructuralConnectivityConfig


def _parse_channel(channel: str) -> tuple[int, int]:
    parts = str(channel).split("_")
    if len(parts) != 2:
        raise ValueError(f"Bipolar channel must contain two contact labels: {channel}")
    try:
        endpoints = (int(parts[0]), int(parts[1]))
    except ValueError as error:
        raise ValueError(
            f"Bipolar channel has non-integer contacts: {channel}"
        ) from error
    if endpoints[0] == endpoints[1]:
        raise ValueError(f"Bipolar channel repeats one contact: {channel}")
    return endpoints


def _contact_reference(global_contact: int) -> tuple[int, int, str]:
    if 0 <= global_contact <= 7:
        return 1, global_contact, "L"
    if 8 <= global_contact <= 15:
        return 0, global_contact - 8, "R"
    raise ValueError(f"Global SceneRay contact is outside 0-15: {global_contact}")


def _scrf_coordinates(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Lead-DBS reconstruction does not exist: {path}")
    value = loadmat(path, simplify_cells=True)
    try:
        raw = value["reco"]["scrf"]["coords_mm"]
    except (KeyError, TypeError) as error:
        raise KeyError(f"Reconstruction lacks reco.scrf.coords_mm: {path}") from error
    cells = tuple(
        np.asarray(cell, dtype=float) for cell in np.asarray(raw, dtype=object)
    )
    if len(cells) != 2 or any(cell.shape != (8, 3) for cell in cells):
        raise ValueError(
            f"Expected right/left SCRF coordinate cells with shape 8x3: {path}"
        )
    if not all(np.all(np.isfinite(cell)) for cell in cells):
        raise ValueError(f"SCRF coordinates contain nonfinite values: {path}")
    return cells[0], cells[1]


def _sphere_image(
    reference_path: Path, center_mm: np.ndarray, radius_mm: float
) -> nib.Nifti1Image:
    reference = nib.load(reference_path)
    if len(reference.shape) != 3:
        raise ValueError(
            f"anchorNative T1w must be three-dimensional: {reference_path}"
        )
    affine = np.asarray(reference.affine, dtype=float)
    inverse = np.linalg.inv(affine)
    center_voxel = nib.affines.apply_affine(inverse, center_mm)
    smallest_scale = float(np.min(np.linalg.svd(affine[:3, :3], compute_uv=False)))
    reach = int(np.ceil(radius_mm / smallest_scale)) + 1
    lower = np.maximum(np.floor(center_voxel).astype(int) - reach, 0)
    upper = np.minimum(
        np.ceil(center_voxel).astype(int) + reach + 1,
        np.asarray(reference.shape, dtype=int),
    )
    grid = np.stack(
        np.meshgrid(
            np.arange(lower[0], upper[0]),
            np.arange(lower[1], upper[1]),
            np.arange(lower[2], upper[2]),
            indexing="ij",
        ),
        axis=-1,
    )
    world = nib.affines.apply_affine(affine, grid.reshape(-1, 3))
    within = np.linalg.norm(world - center_mm[None, :], axis=1) <= radius_mm
    data = np.zeros(reference.shape, dtype=np.uint8)
    local = within.reshape(grid.shape[:3])
    data[
        lower[0] : upper[0],
        lower[1] : upper[1],
        lower[2] : upper[2],
    ] = local
    if not np.any(data):
        raise ValueError(
            f"The {radius_mm:g} mm sphere contains no anchorNative voxel centers."
        )
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    return nib.Nifti1Image(data, affine, header)


def _ants_executable(leaddbs_root: Path) -> Path:
    machine = platform.machine().lower()
    suffix = "maca64" if machine == "arm64" else "maci64"
    executable = leaddbs_root / "ext_libs" / "ANTs" / f"antsApplyTransforms.{suffix}"
    if not executable.is_file():
        raise FileNotFoundError(
            f"Lead-DBS ANTs executable does not exist: {executable}"
        )
    return executable


def forward_warp_label(
    input_path: Path,
    output_path: Path,
    *,
    deformation_path: Path,
    reference_path: Path,
    leaddbs_root: Path,
    overwrite: bool,
    allow_empty: bool = False,
) -> tuple[int, float]:
    """Apply the Lead-DBS forward deformation with GenericLabel interpolation."""

    for path, label in (
        (input_path, "input label"),
        (deformation_path, "forward deformation"),
        (reference_path, "MNI reference"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    prepare_output(output_path, overwrite=overwrite)
    with tempfile.TemporaryDirectory(prefix="structural-warp-") as directory:
        temporary = Path(directory) / "warped.nii.gz"
        command = [
            str(_ants_executable(leaddbs_root)),
            "--verbose",
            "1",
            "--dimensionality",
            "3",
            "--input-image-type",
            "0",
            "--float",
            "1",
            "--input",
            str(input_path),
            "--output",
            str(temporary),
            "--reference-image",
            str(reference_path),
            "--transform",
            f"[{deformation_path},0]",
            "--interpolation",
            "GenericLabel",
        ]
        subprocess.run(command, check=True)
        warped = nib.load(temporary)
        reference = nib.load(reference_path)
        if warped.shape != reference.shape or not np.allclose(
            warped.affine, reference.affine, atol=1e-7, rtol=0
        ):
            raise ValueError(
                "Warped label does not match the configured MNI reference grid."
            )
        data = (np.asanyarray(warped.dataobj) != 0).astype(np.uint8)
        if not np.any(data) and not allow_empty:
            raise ValueError(f"Forward-warped label is empty: {input_path}")
        header = reference.header.copy()
        header.set_data_dtype(np.uint8)
        image = nib.Nifti1Image(data, reference.affine, header)
        nib.save(image, output_path)
    voxel_count = int(np.count_nonzero(data))
    voxel_volume = float(abs(np.linalg.det(reference.affine[:3, :3])))
    return voxel_count, voxel_count * voxel_volume


def forward_warp_scalar(
    input_path: Path,
    output_path: Path,
    *,
    deformation_path: Path,
    reference_path: Path,
    leaddbs_root: Path,
    overwrite: bool,
) -> tuple[int, float]:
    """Apply the Lead-DBS forward deformation with linear scalar interpolation."""

    for path, label in (
        (input_path, "input scalar"),
        (deformation_path, "forward deformation"),
        (reference_path, "MNI reference"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output exists; pass --overwrite to replace it: {output_path}"
        )
    with tempfile.TemporaryDirectory(prefix="structural-scalar-warp-") as directory:
        temporary = Path(directory) / "warped.nii.gz"
        command = [
            str(_ants_executable(leaddbs_root)),
            "--verbose",
            "1",
            "--dimensionality",
            "3",
            "--input-image-type",
            "0",
            "--float",
            "1",
            "--input",
            str(input_path),
            "--output",
            str(temporary),
            "--reference-image",
            str(reference_path),
            "--transform",
            f"[{deformation_path},0]",
            "--interpolation",
            "Linear",
        ]
        subprocess.run(command, check=True)
        warped = nib.load(temporary)
        reference = nib.load(reference_path)
        if warped.shape != reference.shape or not np.allclose(
            warped.affine, reference.affine, atol=1e-7, rtol=0
        ):
            raise ValueError(
                "Warped scalar does not match the configured MNI reference grid."
            )
        data = np.asanyarray(warped.dataobj, dtype=np.float32)
        if not np.all(np.isfinite(data)):
            raise ValueError(
                f"Forward-warped scalar contains nonfinite values: {input_path}"
            )
        header = reference.header.copy()
        header.set_data_dtype(np.float32)
        atomic_nifti(
            nib.Nifti1Image(data, reference.affine, header),
            output_path,
            overwrite=overwrite,
        )
    return int(np.count_nonzero(data > 0)), float(np.max(data))


def _localization_row(
    config: StructuralConnectivityConfig,
    subject_id: str,
    channel: str,
    region: str,
) -> pd.Series:
    path = config.localization_table(subject_id)
    if not path.is_file():
        raise FileNotFoundError(f"Localization table does not exist: {path}")
    table = pd.read_csv(path)
    matches = table.loc[table["channel"].astype(str).eq(channel)]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one localization row for {subject_id}:{channel}, found {len(matches)}"
        )
    row = matches.iloc[0]
    expected_stn = region == "STN"
    expected_snr = region == "SNr"
    if bool(row["STN_in"]) != expected_stn or bool(row["SNr_in"]) != expected_snr:
        raise ValueError(
            f"Localization assignment disagrees for {subject_id}:{channel}:{region}"
        )
    return row


def build_contact_mapping(config: StructuralConnectivityConfig) -> pd.DataFrame:
    """Resolve every selected bipolar endpoint and SCRF midpoint without writing."""

    catalog = build_channel_catalog(config)
    rows: list[dict[str, object]] = []
    for subject_id, subject_rows in catalog.groupby("ID", sort=True, observed=True):
        subject = config.subject(str(subject_id))
        right, left = _scrf_coordinates(config.reconstruction(str(subject_id)))
        cells = (right, left)
        for row in subject_rows.itertuples(index=False):
            channel = str(row.Channel)
            region = str(row.Region)
            localization = _localization_row(config, str(subject_id), channel, region)
            endpoint_a, endpoint_b = _parse_channel(channel)
            cell_a, row_a, side_a = _contact_reference(endpoint_a)
            cell_b, row_b, side_b = _contact_reference(endpoint_b)
            if side_a != side_b or side_a != subject.stimulation_side:
                raise ValueError(
                    f"Channel is not ipsilateral to the registered montage: "
                    f"{subject_id}:{channel}"
                )
            coordinate_a = cells[cell_a][row_a]
            coordinate_b = cells[cell_b][row_b]
            midpoint = (coordinate_a + coordinate_b) / 2.0
            rows.append(
                {
                    "ID": subject_id,
                    "StimSide": subject.stimulation_side,
                    "NormalizationApproval": subject.normalization_approval,
                    "Channel": channel,
                    "Region": region,
                    "CatalogSources": row.CatalogSources,
                    "EndpointA": endpoint_a,
                    "EndpointB": endpoint_b,
                    "EndpointACoordsCellMatlab": cell_a + 1,
                    "EndpointBCoordsCellMatlab": cell_b + 1,
                    "EndpointACoordsRowMatlab": row_a + 1,
                    "EndpointBCoordsRowMatlab": row_b + 1,
                    "EndpointAX": coordinate_a[0],
                    "EndpointAY": coordinate_a[1],
                    "EndpointAZ": coordinate_a[2],
                    "EndpointBX": coordinate_b[0],
                    "EndpointBY": coordinate_b[1],
                    "EndpointBZ": coordinate_b[2],
                    "MidpointSCRFX": midpoint[0],
                    "MidpointSCRFY": midpoint[1],
                    "MidpointSCRFZ": midpoint[2],
                    "LocalizationMniX": float(localization["mni_x"]),
                    "LocalizationMniY": float(localization["mni_y"]),
                    "LocalizationMniZ": float(localization["mni_z"]),
                    "ReconstructionPath": str(config.reconstruction(str(subject_id))),
                    "LocalizationPath": str(config.localization_table(str(subject_id))),
                    "AnchorReferencePath": str(config.anchor_t1(str(subject_id))),
                    "ForwardDeformationPath": str(
                        config.forward_deformation(str(subject_id))
                    ),
                    "MniReferencePath": str(config.mni_reference),
                    "RadiusMm": float(config["masks"]["contact_radius_mm"]),
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(["ID", "Channel"], kind="mergesort")
        .reset_index(drop=True)
    )


def generate_contact_seeds(
    config: StructuralConnectivityConfig,
    *,
    dry_run: bool,
    overwrite: bool,
) -> pd.DataFrame:
    """Generate all native spheres, MNI labels, and the mapping table."""

    mapping = build_contact_mapping(config)
    output_rows: list[dict[str, object]] = []
    for row in mapping.to_dict(orient="records"):
        subject_id = str(row["ID"])
        channel = str(row["Channel"])
        native_path = config.native_contact_seed(subject_id, channel)
        mni_path = config.mni_contact_seed(subject_id, channel)
        center = np.asarray(
            [row["MidpointSCRFX"], row["MidpointSCRFY"], row["MidpointSCRFZ"]],
            dtype=float,
        )
        sphere = _sphere_image(
            config.anchor_t1(subject_id),
            center,
            float(config["masks"]["contact_radius_mm"]),
        )
        native_data = np.asanyarray(sphere.dataobj)
        native_count = int(np.count_nonzero(native_data))
        native_volume = native_count * float(abs(np.linalg.det(sphere.affine[:3, :3])))
        mni_count: int | None = None
        mni_volume: float | None = None
        centroid: np.ndarray | None = None
        localization_distance: float | None = None
        if not dry_run:
            atomic_nifti(sphere, native_path, overwrite=overwrite)
            mni_count, mni_volume = forward_warp_label(
                native_path,
                mni_path,
                deformation_path=config.forward_deformation(subject_id),
                reference_path=config.mni_reference,
                leaddbs_root=config.leaddbs_root,
                overwrite=overwrite,
            )
            mni_image = nib.load(mni_path)
            occupied = np.argwhere(np.asanyarray(mni_image.dataobj) > 0)
            centroid = nib.affines.apply_affine(mni_image.affine, occupied).mean(axis=0)
            localization_coordinate = np.asarray(
                [
                    row["LocalizationMniX"],
                    row["LocalizationMniY"],
                    row["LocalizationMniZ"],
                ],
                dtype=float,
            )
            localization_distance = float(
                np.linalg.norm(centroid - localization_coordinate)
            )
        output_rows.append(
            {
                **row,
                "NativeSeedPath": str(native_path),
                "MniSeedPath": str(mni_path),
                "NativeVoxelCount": native_count,
                "NativeVolumeMm3": native_volume,
                "MniVoxelCount": mni_count,
                "MniVolumeMm3": mni_volume,
                "MniCentroidX": centroid[0] if centroid is not None else None,
                "MniCentroidY": centroid[1] if centroid is not None else None,
                "MniCentroidZ": centroid[2] if centroid is not None else None,
                "MniCentroidToLocalizationMm": localization_distance,
                "MniGridMatchesReference": True if centroid is not None else None,
                "Interpolation": "GenericLabel",
                "Status": "planned" if dry_run else "complete",
            }
        )
    output = pd.DataFrame(output_rows)
    if not dry_run:
        atomic_csv(
            output,
            config.output_path("contact_mapping"),
            overwrite=overwrite,
        )
    return output
