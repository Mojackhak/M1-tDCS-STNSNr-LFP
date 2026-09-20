"""Sample maximum continuous cortical electric-field exposure along streamlines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import nibabel as nib
from numba import njit
import numpy as np


@dataclass(frozen=True)
class FieldSpec:
    """One continuous scalar field registered to streamline world coordinates."""

    field_id: str
    path: Path
    role: str = "cortical_magnE"
    units: str = "V/m"
    aggregation: str = "maximum_traversed_voxel"


@dataclass(frozen=True)
class FieldSummary:
    """Compact continuous-field provenance for persisted outputs."""

    field_id: str
    path: str
    role: str
    units: str
    aggregation: str
    positive_voxel_count: int
    maximum_value: float
    shape: str


@dataclass(frozen=True)
class FiberExposure:
    """Positive-exposure one-based fiber IDs and aligned maximum field values."""

    fiber_ids: np.ndarray
    peak_values: np.ndarray


@dataclass(frozen=True)
class SparseFieldLookup:
    """Sparse continuous fields sharing one exact voxel grid."""

    field_ids: tuple[str, ...]
    shape: np.ndarray
    inverse_affine: np.ndarray
    voxel_keys: np.ndarray
    voxel_values: np.ndarray
    lower_bound: np.ndarray
    upper_bound: np.ndarray


def build_sparse_field_lookup(
    specs: Sequence[FieldSpec],
) -> tuple[SparseFieldLookup, tuple[FieldSummary, ...]]:
    """Load nonnegative scalar fields and combine their positive voxels."""

    if not specs:
        raise ValueError("At least one scalar field is required.")
    identifiers = [spec.field_id for spec in specs]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Continuous-field identifiers must be unique.")

    reference_shape: tuple[int, int, int] | None = None
    reference_affine: np.ndarray | None = None
    field_keys: list[np.ndarray] = []
    field_values: list[np.ndarray] = []
    summaries: list[FieldSummary] = []
    for spec in specs:
        path = Path(spec.path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Continuous scalar field does not exist: {path}")
        image = nib.load(path)
        if len(image.shape) != 3:
            raise ValueError(
                f"Continuous scalar field must be three-dimensional: {path}"
            )
        shape = tuple(int(value) for value in image.shape)
        affine = np.asarray(image.affine, dtype=np.float64)
        if reference_shape is None:
            reference_shape = shape
            reference_affine = affine
        elif shape != reference_shape or not np.allclose(
            affine, reference_affine, atol=1e-7, rtol=0
        ):
            raise ValueError("Continuous scalar fields must share one exact grid.")

        values = np.asanyarray(image.dataobj, dtype=np.float32)
        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"Continuous scalar field contains nonfinite values: {path}"
            )
        if np.any(values < 0):
            raise ValueError(
                f"Continuous magnitude field contains negative values: {path}"
            )
        flat = values.reshape(-1, order="C")
        keys = np.flatnonzero(flat > 0).astype(np.int64)
        positive = np.asarray(flat[keys], dtype=np.float32)
        keys.setflags(write=False)
        positive.setflags(write=False)
        field_keys.append(keys)
        field_values.append(positive)
        summaries.append(
            FieldSummary(
                field_id=spec.field_id,
                path=str(path),
                role=spec.role,
                units=spec.units,
                aggregation=spec.aggregation,
                positive_voxel_count=int(keys.size),
                maximum_value=float(np.max(positive)) if positive.size else 0.0,
                shape="x".join(str(value) for value in shape),
            )
        )

    assert reference_shape is not None and reference_affine is not None
    nonempty = [keys for keys in field_keys if keys.size]
    voxel_keys = (
        np.unique(np.concatenate(nonempty)) if nonempty else np.empty(0, dtype=np.int64)
    )
    voxel_values = np.zeros((voxel_keys.size, len(specs)), dtype=np.float32)
    for field_index, (keys, values) in enumerate(
        zip(field_keys, field_values, strict=True)
    ):
        if keys.size:
            positions = np.searchsorted(voxel_keys, keys)
            voxel_values[positions, field_index] = values
    shape_array = np.asarray(reference_shape, dtype=np.int64)
    if voxel_keys.size:
        coordinates = np.asarray(
            np.unravel_index(voxel_keys, reference_shape, order="C"),
            dtype=np.int64,
        )
        lower_bound = np.min(coordinates, axis=1)
        upper_bound = np.max(coordinates, axis=1) + 1
    else:
        lower_bound = np.zeros(3, dtype=np.int64)
        upper_bound = np.zeros(3, dtype=np.int64)
    try:
        inverse_affine = np.linalg.inv(reference_affine)
    except np.linalg.LinAlgError as error:
        raise ValueError("Continuous scalar-field affine is singular.") from error
    for array in (
        shape_array,
        inverse_affine,
        voxel_keys,
        voxel_values,
        lower_bound,
        upper_bound,
    ):
        array.setflags(write=False)
    return (
        SparseFieldLookup(
            field_ids=tuple(identifiers),
            shape=shape_array,
            inverse_affine=inverse_affine,
            voxel_keys=voxel_keys,
            voxel_values=voxel_values,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        ),
        tuple(summaries),
    )


@njit(cache=True)
def _binary_search(keys: np.ndarray, value: int) -> int:
    left = 0
    right = keys.size
    while left < right:
        middle = (left + right) // 2
        candidate = keys[middle]
        if candidate < value:
            left = middle + 1
        else:
            right = middle
    if left < keys.size and keys[left] == value:
        return left
    return -1


@njit(cache=True)
def _transform_point(point: np.ndarray, inverse_affine: np.ndarray) -> np.ndarray:
    transformed = np.empty(3, dtype=np.float64)
    for axis in range(3):
        transformed[axis] = (
            inverse_affine[axis, 0] * point[0]
            + inverse_affine[axis, 1] * point[1]
            + inverse_affine[axis, 2] * point[2]
            + inverse_affine[axis, 3]
            + 0.5
        )
    return transformed


@njit(cache=True)
def _clip_segment_to_grid(
    start: np.ndarray,
    stop: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
) -> tuple[bool, float, float]:
    enter = 0.0
    leave = 1.0
    for axis in range(3):
        direction = stop[axis] - start[axis]
        lower = float(lower_bound[axis])
        upper = np.nextafter(float(upper_bound[axis]), -np.inf)
        if direction == 0.0:
            if start[axis] < lower or start[axis] > upper:
                return False, 0.0, 0.0
            continue
        first = (lower - start[axis]) / direction
        second = (upper - start[axis]) / direction
        if first > second:
            first, second = second, first
        if first > enter:
            enter = first
        if second < leave:
            leave = second
        if enter > leave:
            return False, 0.0, 0.0
    if leave < 0.0 or enter > 1.0:
        return False, 0.0, 0.0
    return True, max(0.0, enter), min(1.0, leave)


@njit(cache=True)
def _traverse_field(
    points: np.ndarray,
    point_offsets: np.ndarray,
    inverse_affine: np.ndarray,
    shape: np.ndarray,
    voxel_keys: np.ndarray,
    voxel_values: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    output: np.ndarray,
) -> None:
    for fiber_index in range(point_offsets.size - 1):
        point_start = point_offsets[fiber_index]
        point_stop = point_offsets[fiber_index + 1]
        for point_index in range(point_start, point_stop - 1):
            start = _transform_point(points[point_index], inverse_affine)
            stop = _transform_point(points[point_index + 1], inverse_affine)
            intersects, enter, leave = _clip_segment_to_grid(
                start,
                stop,
                lower_bound,
                upper_bound,
            )
            if not intersects:
                continue
            original_direction = stop - start
            clipped_start = start + original_direction * enter
            clipped_stop = start + original_direction * leave
            direction = clipped_stop - clipped_start
            cell = np.empty(3, dtype=np.int64)
            end_cell = np.empty(3, dtype=np.int64)
            step = np.zeros(3, dtype=np.int64)
            next_boundary_t = np.empty(3, dtype=np.float64)
            delta_t = np.empty(3, dtype=np.float64)
            for axis in range(3):
                cell[axis] = int(np.floor(clipped_start[axis]))
                end_cell[axis] = int(np.floor(clipped_stop[axis]))
                if cell[axis] < lower_bound[axis]:
                    cell[axis] = lower_bound[axis]
                elif cell[axis] >= upper_bound[axis]:
                    cell[axis] = upper_bound[axis] - 1
                if end_cell[axis] < lower_bound[axis]:
                    end_cell[axis] = lower_bound[axis]
                elif end_cell[axis] >= upper_bound[axis]:
                    end_cell[axis] = upper_bound[axis] - 1
                if direction[axis] > 0.0:
                    step[axis] = 1
                    next_boundary_t[axis] = (
                        float(cell[axis] + 1) - clipped_start[axis]
                    ) / direction[axis]
                    delta_t[axis] = 1.0 / direction[axis]
                elif direction[axis] < 0.0:
                    step[axis] = -1
                    next_boundary_t[axis] = (
                        float(cell[axis]) - clipped_start[axis]
                    ) / direction[axis]
                    delta_t[axis] = -1.0 / direction[axis]
                else:
                    next_boundary_t[axis] = np.inf
                    delta_t[axis] = np.inf

            maximum_steps = int(
                (upper_bound[0] - lower_bound[0])
                + (upper_bound[1] - lower_bound[1])
                + (upper_bound[2] - lower_bound[2])
                + 3
            )
            for _ in range(maximum_steps):
                flat_key = (cell[0] * shape[1] + cell[1]) * shape[2] + cell[2]
                key_index = _binary_search(voxel_keys, flat_key)
                if key_index >= 0:
                    for field_index in range(voxel_values.shape[1]):
                        value = voxel_values[key_index, field_index]
                        if value > output[fiber_index, field_index]:
                            output[fiber_index, field_index] = value
                if (
                    cell[0] == end_cell[0]
                    and cell[1] == end_cell[1]
                    and cell[2] == end_cell[2]
                ):
                    break
                minimum_t = min(
                    next_boundary_t[0],
                    next_boundary_t[1],
                    next_boundary_t[2],
                )
                tolerance = 1e-12 * max(1.0, abs(minimum_t))
                for axis in range(3):
                    if next_boundary_t[axis] <= minimum_t + tolerance:
                        cell[axis] += step[axis]
                        next_boundary_t[axis] += delta_t[axis]


def maximum_field_exposure(chunk: Any, lookup: SparseFieldLookup) -> np.ndarray:
    """Return one maximum traversed voxel value per fiber and scalar field."""

    points = np.asarray(chunk.points, dtype=np.float32)
    offsets = np.asarray(chunk.point_offsets, dtype=np.int64)
    fiber_ids = np.asarray(chunk.fiber_ids, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Chunk points must have shape (n_points, 3).")
    if offsets.ndim != 1 or offsets.size != fiber_ids.size + 1:
        raise ValueError("Chunk point offsets must have one boundary per fiber.")
    if (
        offsets[0] != 0
        or offsets[-1] != points.shape[0]
        or np.any(offsets[1:] - offsets[:-1] < 2)
    ):
        raise ValueError("Chunk point offsets do not define valid streamlines.")
    output = np.zeros((fiber_ids.size, len(lookup.field_ids)), dtype=np.float32)
    if lookup.voxel_keys.size:
        _traverse_field(
            points,
            offsets,
            lookup.inverse_affine,
            lookup.shape,
            lookup.voxel_keys,
            lookup.voxel_values,
            lookup.lower_bound,
            lookup.upper_bound,
            output,
        )
    return output
