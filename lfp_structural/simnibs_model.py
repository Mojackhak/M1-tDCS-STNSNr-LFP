"""Build CHARM head models, HD-tDCS fields, and cortical Tstim masks."""

from __future__ import annotations

import csv
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

from .artifacts import atomic_csv, atomic_json, atomic_nifti, prepare_output
from .config import StructuralConnectivityConfig
from .geometry import forward_warp_label


def _require_simnibs() -> Any:
    try:
        import simnibs
    except ImportError as error:
        raise RuntimeError(
            "This stage must run in the configured simnibs Conda environment."
        ) from error
    if str(simnibs.__version__) != "4.6.0":
        raise RuntimeError(
            f"The frozen SimNIBS release is 4.6.0, found {simnibs.__version__}."
        )
    return simnibs


def simnibs_root(config: StructuralConnectivityConfig, subject_id: str) -> Path:
    return config.patient_derivative_root(subject_id) / "simnibs"


def m2m_path(config: StructuralConnectivityConfig, subject_id: str) -> Path:
    return simnibs_root(config, subject_id) / f"m2m_{subject_id}"


def head_mesh_path(config: StructuralConnectivityConfig, subject_id: str) -> Path:
    return m2m_path(config, subject_id) / f"{subject_id}.msh"


def field_root(config: StructuralConnectivityConfig, subject_id: str) -> Path:
    return config.patient_derivative_root(subject_id) / "fields"


def field_volume_path(
    config: StructuralConnectivityConfig,
    subject_id: str,
    polarity: str,
    field: str,
) -> Path:
    side = config.subject(subject_id).stimulation_side
    return (
        field_root(config, subject_id)
        / "anchorNative"
        / (
            f"{subject_id}_side-{side}_polarity-{polarity}_"
            f"space-anchorNative_desc-{field}.nii.gz"
        )
    )


def _head_model_outputs(
    config: StructuralConnectivityConfig,
    subject_id: str,
) -> list[Path]:
    root = m2m_path(config, subject_id)
    return [
        head_mesh_path(config, subject_id),
        root / "final_tissues.nii.gz",
        root / "surfaces" / "lh.pial.gii",
        root / "surfaces" / "rh.pial.gii",
        root / "surfaces" / "lh.white.gii",
        root / "surfaces" / "rh.white.gii",
        root / "eeg_positions" / "EEG10-10_Neuroelectrics.csv",
    ]


def run_head_model(
    config: StructuralConnectivityConfig,
    subject_id: str,
    *,
    dry_run: bool,
    overwrite: bool,
) -> dict[str, Any]:
    """Run CHARM with the already coregistered T2 copied without registration."""

    _require_simnibs()
    t1 = config.anchor_t1(subject_id)
    t2 = config.anchor_t2(subject_id)
    for path, label in ((t1, "T1w"), (t2, "T2w")):
        if not path.is_file():
            raise FileNotFoundError(f"Subject {label} does not exist: {path}")
    root = simnibs_root(config, subject_id)
    m2m = m2m_path(config, subject_id)
    mesh = head_mesh_path(config, subject_id)
    marker = root / "head_model_manifest.json"
    existing = [path for path in (m2m, marker) if path.exists()]
    charm = Path(sys.executable).parent / "charm"
    if not charm.is_file():
        raise FileNotFoundError(f"CHARM executable does not exist: {charm}")
    command = [
        str(charm),
        subject_id,
        str(t1),
        str(t2),
        "--skipregisterT2",
    ]
    result = {
        "status": "ready" if dry_run else "complete",
        "subject_id": subject_id,
        "t1": str(t1),
        "t2": str(t2),
        "register_t2": False,
        "command": command,
        "m2m_path": str(m2m),
        "mesh_path": str(mesh),
    }
    required_outputs = _head_model_outputs(config, subject_id)
    complete_existing = all(path.is_file() for path in required_outputs)
    if complete_existing and not marker.exists() and not overwrite:
        result["execution"] = (
            "would_finalize_existing_outputs"
            if dry_run
            else "finalized_existing_outputs"
        )
        if not dry_run:
            atomic_json(result, marker, overwrite=False)
        return result
    if existing and not overwrite:
        raise FileExistsError(
            f"Head-model output exists; pass --overwrite to replace it: {existing[0]}"
        )
    if dry_run:
        result["execution"] = "planned"
        return result
    for path in existing:
        prepare_output(path, overwrite=True)
        if path.is_dir():
            path.rmdir()
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, cwd=root, check=True)
    missing = [path for path in required_outputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"CHARM did not produce required output: {missing[0]}")
    result["execution"] = "executed"
    atomic_json(result, marker, overwrite=False)
    return result


def _read_cap_positions(path: Path, labels: list[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Warped Neuroelectrics cap does not exist: {path}")
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in csv.reader(handle):
            if len(raw) >= 5 and raw[0] == "Electrode" and raw[4] in labels:
                rows.append(
                    {
                        "Electrode": raw[4],
                        "X": float(raw[1]),
                        "Y": float(raw[2]),
                        "Z": float(raw[3]),
                    }
                )
    table = pd.DataFrame(rows)
    if set(table.get("Electrode", [])) != set(labels) or len(table) != len(labels):
        raise ValueError(
            f"Warped cap does not contain each montage electrode exactly once: {path}"
        )
    order = {label: index for index, label in enumerate(labels)}
    table["Order"] = table["Electrode"].map(order)
    return table.sort_values("Order").drop(columns="Order").reset_index(drop=True)


def _add_montage(
    session: Any,
    *,
    name: str,
    currents: list[float],
    electrode_labels: list[str],
    electrode_definition: str,
    electrode_shape: str,
    dimensions_mm: list[float],
    thickness_mm: float,
    gel_conductivity: float,
) -> None:
    tdcs = session.add_tdcslist()
    tdcs.name = name
    tdcs.currents = currents
    tdcs.cond[499].value = gel_conductivity
    for channel, label in enumerate(electrode_labels, start=1):
        electrode = tdcs.add_electrode()
        electrode.name = label
        electrode.channelnr = channel
        electrode.centre = label
        electrode.definition = electrode_definition
        electrode.shape = electrode_shape
        electrode.dimensions = dimensions_mm
        electrode.thickness = thickness_mm


def _move_generated_volume(generated: Path, destination: Path) -> None:
    if not generated.is_file():
        raise FileNotFoundError(f"SimNIBS did not generate field volume: {generated}")
    prepare_output(destination, overwrite=False)
    os.replace(generated, destination)


def _current_calibration_qc(
    log_path: Path,
    polarity_names: list[str],
    *,
    expected_pairs_per_polarity: int,
) -> dict[str, Any]:
    """Read the SimNIBS pairwise current-calibration estimates from its log."""

    if not log_path.is_file():
        raise FileNotFoundError(f"SimNIBS simulation log does not exist: {log_path}")
    values: dict[str, list[float]] = {name: [] for name in polarity_names}
    current: str | None = None
    poslist_pattern = re.compile(r"Running Poslist Number:\s*(\d+)")
    error_patterns = (
        re.compile(r"Estimated current calibration error:\s*([0-9.]+)%"),
        re.compile(r"Estimated error value:\s*([0-9.]+)%"),
    )
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        poslist = poslist_pattern.search(line)
        if poslist:
            index = int(poslist.group(1)) - 1
            current = polarity_names[index] if 0 <= index < len(polarity_names) else None
            continue
        for pattern in error_patterns:
            match = pattern.search(line)
            if match and current is not None:
                values[current].append(float(match.group(1)))
                break
    invalid = {
        polarity: len(errors)
        for polarity, errors in values.items()
        if len(errors) != expected_pairs_per_polarity
    }
    if invalid:
        raise ValueError(
            f"Unexpected SimNIBS calibration-error count in {log_path}: {invalid}"
        )
    flattened = [value for errors in values.values() for value in errors]
    warning_threshold = 10.0
    return {
        "source_log": str(log_path),
        "errors_percent_by_polarity": values,
        "warning_threshold_percent": warning_threshold,
        "warnings_over_threshold": int(
            sum(value > warning_threshold for value in flattened)
        ),
        "maximum_error_percent": float(max(flattened)),
    }


def run_field_model(
    config: StructuralConnectivityConfig,
    subject_id: str,
    *,
    dry_run: bool,
    overwrite: bool,
    cpus: int,
) -> dict[str, Any]:
    """Solve both exact polarity reversals and interpolate E to anchorNative."""

    if cpus != 1:
        raise ValueError(
            "The field stage requires --cpus 1 so every pairwise current-"
            "calibration estimate is retained in the SimNIBS session log."
        )
    simnibs = _require_simnibs()
    from simnibs import sim_struct
    from simnibs.utils import transformations
    from simnibs.utils.file_finder import SubjectFiles

    m2m = m2m_path(config, subject_id)
    mesh = head_mesh_path(config, subject_id)
    if not m2m.is_dir() or not mesh.is_file():
        raise FileNotFoundError(f"Accepted CHARM head model is missing for {subject_id}.")
    output_root = field_root(config, subject_id)
    marker = output_root / "field_manifest.json"
    if output_root.exists() and not overwrite:
        raise FileExistsError(
            f"Field output exists; pass --overwrite to replace it: {output_root}"
        )
    side = config.subject(subject_id).stimulation_side
    electrode_labels = list(config["montages"][side]["electrodes"])
    cap_path = Path(
        SubjectFiles(subpath=str(m2m)).get_eeg_cap("EEG10-10_Neuroelectrics.csv")
    )
    positions = _read_cap_positions(cap_path, electrode_labels)
    settings = config["montages"]["electrode"]
    anodal = [float(value) for value in config["montages"]["anode_centered_currents_a"]]
    cathodal = [float(value) for value in config["montages"]["cathode_centered_currents_a"]]
    if not np.isclose(sum(anodal), 0.0) or not np.isclose(sum(cathodal), 0.0):
        raise ValueError("Each configured five-channel current vector must sum to zero.")
    plan = {
        "status": "ready" if dry_run else "complete",
        "subject_id": subject_id,
        "stimulation_side": side,
        "m2m_path": str(m2m),
        "mesh_path": str(mesh),
        "cap_path": str(cap_path),
        "electrodes": electrode_labels,
        "anode_centered_currents_a": anodal,
        "cathode_centered_currents_a": cathodal,
        "electrode_definition": settings["definition"],
        "electrode_shape": settings["shape"],
        "electrode_dimensions_mm": settings["dimensions_mm"],
        "electrode_thickness_mm": settings["thickness_mm"],
        "gel_conductivity_s_per_m": settings["conductivity_s_per_m"],
        "fields": "eE",
        "surface_quantities": ["magnitude", "normal", "tangent", "angle"],
        "volume_reference": str(config.anchor_t1(subject_id)),
        "volume_tissues": [2],
        "cpus": cpus,
        "simnibs_release": str(simnibs.__version__),
    }
    if dry_run:
        return plan
    prepare_output(output_root, overwrite=overwrite)
    if output_root.exists():
        output_root.rmdir()
    session = sim_struct.SESSION()
    session.subpath = str(m2m)
    session.pathfem = str(output_root)
    session.fields = "eE"
    session.map_to_surf = True
    session.map_to_fsavg = False
    session.map_to_vol = False
    session.map_to_MNI = False
    session.open_in_gmsh = False
    session.eeg_cap = str(cap_path)
    common = {
        "electrode_labels": electrode_labels,
        "electrode_definition": str(settings["definition"]),
        "electrode_shape": str(settings["shape"]),
        "dimensions_mm": [float(value) for value in settings["dimensions_mm"]],
        "thickness_mm": float(settings["thickness_mm"]),
        "gel_conductivity": float(settings["conductivity_s_per_m"]),
    }
    names = {
        "Anodal": f"{subject_id}_side-{side}_polarity-Anodal",
        "Cathodal": f"{subject_id}_side-{side}_polarity-Cathodal",
    }
    _add_montage(session, name=names["Anodal"], currents=anodal, **common)
    _add_montage(session, name=names["Cathodal"], currents=cathodal, **common)
    simnibs.run_simnibs(session, cpus=cpus)
    logs = sorted(output_root.glob("simnibs_simulation_*.log"))
    if len(logs) != 1:
        raise ValueError(
            f"Expected one SimNIBS simulation log in {output_root}, found {len(logs)}."
        )
    plan["current_calibration_qc"] = _current_calibration_qc(
        logs[0],
        list(names),
        expected_pairs_per_polarity=len(electrode_labels) - 1,
    )

    volume_root = output_root / "anchorNative"
    volume_root.mkdir(parents=True, exist_ok=True)
    volume_paths: dict[str, dict[str, str]] = {}
    for polarity, name in names.items():
        result_mesh = output_root / f"{name}_scalar.msh"
        if not result_mesh.is_file():
            raise FileNotFoundError(f"SimNIBS result mesh does not exist: {result_mesh}")
        base = volume_root / f".{name}_field.nii.gz"
        transformations.interpolate_to_volume(
            str(result_mesh),
            str(config.anchor_t1(subject_id)),
            str(base),
            keep_tissues=[2],
            method="linear",
        )
        generated_magnitude = volume_root / f".{name}_field_magnE.nii.gz"
        generated_vector = volume_root / f".{name}_field_E.nii.gz"
        magnitude_path = field_volume_path(
            config, subject_id, polarity, "magnE"
        )
        vector_path = field_volume_path(config, subject_id, polarity, "E")
        _move_generated_volume(generated_magnitude, magnitude_path)
        _move_generated_volume(generated_vector, vector_path)
        volume_paths[polarity] = {
            "magnE": str(magnitude_path),
            "E": str(vector_path),
            "mesh": str(result_mesh),
        }

    anodal_magnitude = np.asanyarray(
        nib.load(volume_paths["Anodal"]["magnE"]).dataobj, dtype=float
    )
    cathodal_magnitude = np.asanyarray(
        nib.load(volume_paths["Cathodal"]["magnE"]).dataobj, dtype=float
    )
    anodal_vector = np.asanyarray(
        nib.load(volume_paths["Anodal"]["E"]).dataobj, dtype=float
    )
    cathodal_vector = np.asanyarray(
        nib.load(volume_paths["Cathodal"]["E"]).dataobj, dtype=float
    )
    plan["polarity_reversal_qc"] = {
        "max_abs_magnitude_difference_v_per_m": float(
            np.nanmax(np.abs(anodal_magnitude - cathodal_magnitude))
        ),
        "max_abs_vector_sum_v_per_m": float(
            np.nanmax(np.abs(anodal_vector + cathodal_vector))
        ),
    }
    plan["volume_outputs"] = volume_paths
    positions["StimSide"] = side
    positions["AnodalCurrentA"] = anodal
    positions["CathodalCurrentA"] = cathodal
    atomic_csv(
        positions,
        output_root / "electrode_positions.csv",
        overwrite=False,
    )
    atomic_json(plan, marker, overwrite=False)
    return plan


def _cortical_ribbon(
    m2m: Path,
    reference: nib.spatialimages.SpatialImage,
) -> np.ndarray:
    from simnibs.mesh_tools import mesh_io
    from simnibs.segmentation.brain_surface import mask_from_surface
    from simnibs.utils.file_finder import SubjectFiles

    files = SubjectFiles(subpath=str(m2m))
    white = mesh_io.load_subject_surfaces(files, "white")
    white_mesh = white["lh"].join_mesh(white["rh"])
    pial = mesh_io.load_subject_surfaces(files, "pial")
    pial_mesh = pial["lh"].join_mesh(pial["rh"])
    inside_white = mask_from_surface(
        white_mesh.nodes[:],
        white_mesh.elm[:, :3] - 1,
        reference.affine,
        reference.shape,
    )
    inside_pial = mask_from_surface(
        pial_mesh.nodes[:],
        pial_mesh.elm[:, :3] - 1,
        reference.affine,
        reference.shape,
    )
    ribbon = inside_pial & ~inside_white
    if not np.any(ribbon):
        raise ValueError("Pial-minus-white cortical ribbon is empty.")
    return ribbon


def threshold_target_arrays(
    magnitude: np.ndarray,
    cortical_ribbon: np.ndarray,
    *,
    field_threshold: float,
    ahm_quantile: float,
    ahm_fraction: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Create fixed-threshold Tstim and descriptive AHM masks."""

    values = np.asarray(magnitude, dtype=float)
    ribbon = np.asarray(cortical_ribbon, dtype=bool)
    if values.shape != ribbon.shape:
        raise ValueError("Magnitude and cortical-ribbon arrays must share one grid.")
    finite_ribbon = ribbon & np.isfinite(values)
    if not np.any(finite_ribbon):
        raise ValueError("Cortical ribbon contains no finite magnitude values.")
    robust_peak = float(np.quantile(values[finite_ribbon], ahm_quantile))
    ahm_threshold = ahm_fraction * robust_peak
    target = finite_ribbon & (values >= field_threshold)
    ahm = finite_ribbon & (values >= ahm_threshold)
    return target, ahm, robust_peak, ahm_threshold


def build_tstim_target(
    config: StructuralConnectivityConfig,
    subject_id: str,
    *,
    dry_run: bool,
    overwrite: bool,
) -> dict[str, Any]:
    """Threshold the complete montage field in the pial-minus-white ribbon."""

    _require_simnibs()
    magnitude_path = field_volume_path(config, subject_id, "Anodal", "magnE")
    if not magnitude_path.is_file():
        raise FileNotFoundError(f"Anodal magnitude field does not exist: {magnitude_path}")
    reference_path = config.anchor_t1(subject_id)
    reference = nib.load(reference_path)
    magnitude_image = nib.load(magnitude_path)
    if magnitude_image.shape != reference.shape or not np.allclose(
        magnitude_image.affine, reference.affine, atol=1e-7, rtol=0
    ):
        raise ValueError("Magnitude field does not match the exact anchorNative T1w grid.")
    native_target = config.native_target(subject_id)
    mni_target = config.mni_target(subject_id)
    side = config.subject(subject_id).stimulation_side
    ribbon_path = (
        config.patient_derivative_root(subject_id)
        / "masks"
        / "anchorNative"
        / f"{subject_id}_space-anchorNative_desc-corticalRibbon.nii.gz"
    )
    ahm_path = (
        config.patient_derivative_root(subject_id)
        / "masks"
        / "anchorNative"
        / f"{subject_id}_side-{side}_space-anchorNative_desc-AHM.nii.gz"
    )
    marker = (
        config.patient_derivative_root(subject_id)
        / "masks"
        / "target_manifest.json"
    )
    outputs = [native_target, mni_target, ribbon_path, ahm_path, marker]
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Target output exists; pass --overwrite to replace it: {existing[0]}"
        )
    ribbon = _cortical_ribbon(m2m_path(config, subject_id), reference)
    magnitude = np.asanyarray(magnitude_image.dataobj, dtype=float)
    target, ahm, robust_peak, ahm_threshold = threshold_target_arrays(
        magnitude,
        ribbon,
        field_threshold=float(config["masks"]["field_threshold_v_per_m"]),
        ahm_quantile=float(config["masks"]["descriptive_ahm_quantile"]),
        ahm_fraction=float(config["masks"]["descriptive_ahm_fraction"]),
    )
    voxel_volume = float(abs(np.linalg.det(reference.affine[:3, :3])))
    components, component_count = ndimage.label(target)
    component_sizes = (
        np.bincount(components[target])[1:]
        if component_count
        else np.asarray([], dtype=int)
    )
    result: dict[str, Any] = {
        "status": "ready" if dry_run else "complete",
        "subject_id": subject_id,
        "stimulation_side": side,
        "magnitude_path": str(magnitude_path),
        "reference_path": str(reference_path),
        "cortical_ribbon_definition": "inside_pial_and_outside_white",
        "field_threshold_v_per_m": float(
            config["masks"]["field_threshold_v_per_m"]
        ),
        "robust_peak_quantile": float(
            config["masks"]["descriptive_ahm_quantile"]
        ),
        "robust_peak_v_per_m": robust_peak,
        "ahm_fraction": float(config["masks"]["descriptive_ahm_fraction"]),
        "ahm_threshold_v_per_m": ahm_threshold,
        "cortical_ribbon_voxels": int(np.count_nonzero(ribbon)),
        "cortical_ribbon_volume_mm3": float(np.count_nonzero(ribbon)) * voxel_volume,
        "target_voxels": int(np.count_nonzero(target)),
        "target_volume_mm3": float(np.count_nonzero(target)) * voxel_volume,
        "target_status": (
            "nonempty" if np.any(target) else "empty_fixed_threshold"
        ),
        "target_components": int(component_count),
        "target_component_sizes_voxels": component_sizes.astype(int).tolist(),
        "ahm_voxels": int(np.count_nonzero(ahm)),
        "ahm_volume_mm3": float(np.count_nonzero(ahm)) * voxel_volume,
        "native_target_path": str(native_target),
        "mni_target_path": str(mni_target),
        "ribbon_path": str(ribbon_path),
        "ahm_path": str(ahm_path),
        "interpolation_to_mni": "GenericLabel",
        "all_target_components_retained": True,
    }
    if dry_run:
        return result
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    atomic_nifti(
        nib.Nifti1Image(ribbon.astype(np.uint8), reference.affine, header),
        ribbon_path,
        overwrite=overwrite,
    )
    atomic_nifti(
        nib.Nifti1Image(target.astype(np.uint8), reference.affine, header),
        native_target,
        overwrite=overwrite,
    )
    atomic_nifti(
        nib.Nifti1Image(ahm.astype(np.uint8), reference.affine, header),
        ahm_path,
        overwrite=overwrite,
    )
    mni_voxels, mni_volume = forward_warp_label(
        native_target,
        mni_target,
        deformation_path=config.forward_deformation(subject_id),
        reference_path=config.mni_reference,
        leaddbs_root=config.leaddbs_root,
        overwrite=overwrite,
        allow_empty=not np.any(target),
    )
    result["mni_target_voxels"] = mni_voxels
    result["mni_target_volume_mm3"] = mni_volume
    atomic_json(result, marker, overwrite=overwrite)
    return result
