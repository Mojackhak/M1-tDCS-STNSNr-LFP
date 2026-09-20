"""Resolve NIfTI masks for Lead-DBS segment-aware streamline traversal."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from .lead_runtime import activate_sextant


@dataclass(frozen=True)
class MaskSpec:
    """One named binary or probability-thresholded mask."""

    mask_id: str
    path: Path
    probability_threshold: float | None
    role: str
    allow_empty: bool = False


@dataclass(frozen=True)
class MaskSummary:
    """Compact resolved-mask provenance for tabular output."""

    mask_id: str
    path: str
    role: str
    threshold: float | None
    voxel_count: int
    volume_mm3: float
    shape: str


def resolve_mask(
    spec: MaskSpec, sextant_python_root: Path | str
) -> tuple[Any, MaskSummary]:
    """Load and threshold one NIfTI mask at the external-file boundary."""

    activate_sextant(sextant_python_root)
    from sextant.shared.connectome_types import ResolvedMask

    path = Path(spec.path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Mask does not exist: {path}")
    image = nib.load(path)
    if len(image.shape) != 3:
        raise ValueError(f"Mask must be three-dimensional: {path}")
    values = np.asanyarray(image.dataobj)
    if spec.probability_threshold is None:
        occupied = np.isfinite(values) & (values > 0)
        source_value_type = "binary"
        threshold_source = "nonzero"
    else:
        occupied = np.isfinite(values) & (values >= spec.probability_threshold)
        source_value_type = "probability"
        threshold_source = "configured_probability_threshold"
    indices = np.flatnonzero(occupied.reshape(-1, order="C")).astype(np.int64)
    if indices.size == 0 and not spec.allow_empty:
        raise ValueError(f"Mask is empty after thresholding: {path}")
    indices.setflags(write=False)
    affine = np.asarray(image.affine, dtype=np.float64)
    affine.setflags(write=False)
    voxel_volume = float(abs(np.linalg.det(affine[:3, :3])))
    shape = tuple(int(value) for value in image.shape)
    resolved = ResolvedMask(
        roi_id=spec.mask_id,
        role=spec.role,
        source_path=path,
        relative_path=path.name,
        target_group="structural_connectivity",
        source_value_type=source_value_type,
        probability_threshold=spec.probability_threshold,
        threshold_source=threshold_source,
        source_artifact_id=f"structural_connectivity:{spec.mask_id}",
        source_revision=1,
        voxel_count=int(indices.size),
        physical_volume_mm3=float(indices.size) * voxel_volume,
        status="valid" if indices.size else "empty",
        shape=shape,
        affine=affine,
        flat_voxel_indices=indices,
    )
    summary = MaskSummary(
        mask_id=spec.mask_id,
        path=str(path),
        role=spec.role,
        threshold=spec.probability_threshold,
        voxel_count=int(indices.size),
        volume_mm3=float(indices.size) * voxel_volume,
        shape="x".join(str(value) for value in shape),
    )
    return resolved, summary
