"""Unified streaming adapters for TCK and Lead-DBS HDF5 connectomes."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Iterator

import nibabel as nib
import numpy as np

from .lead_runtime import activate_sextant


@dataclass(frozen=True)
class ConnectomeSummary:
    """Source metadata required by the structural-connectivity outputs."""

    connectome_id: str
    representation: str
    path: str
    n_fibers: int
    fiber_id_base: int
    weighted: bool


class TCKConnectome:
    """Stream a TCK as globally one-based Lead-DBS-compatible fiber chunks."""

    def __init__(
        self,
        path: Path | str,
        *,
        connectome_id: str,
        sextant_python_root: Path | str,
    ) -> None:
        activate_sextant(sextant_python_root)
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"TCK connectome does not exist: {source}")
        try:
            loaded = nib.streamlines.load(source, lazy_load=True)
            count = int(loaded.header["count"])
        except Exception as error:
            raise ValueError(
                f"Cannot inspect TCK connectome {source}: {error}"
            ) from error
        if count <= 0:
            raise ValueError(f"TCK connectome has no streamlines: {source}")
        self._path = source
        self._connectome_id = connectome_id
        self._n_fibers = count
        self._sextant_python_root = Path(sextant_python_root)

    @property
    def n_fibers(self) -> int:
        return self._n_fibers

    @property
    def summary(self) -> ConnectomeSummary:
        return ConnectomeSummary(
            connectome_id=self._connectome_id,
            representation="tck",
            path=str(self._path),
            n_fibers=self._n_fibers,
            fiber_id_base=1,
            weighted=False,
        )

    def iter_chunks(self, chunk_size: int) -> Iterator[Any]:
        """Yield bounded chunks and verify the header count at end of stream."""

        if (
            not isinstance(chunk_size, int)
            or isinstance(chunk_size, bool)
            or chunk_size < 1
        ):
            raise ValueError("TCK chunk size must be a positive integer.")
        activate_sextant(self._sextant_python_root)
        from sextant.tractography.dwi_seed_target.classification import (
            streamlines_to_fiber_chunk,
        )
        from sextant.tractography.dwi_seed_target.tck import iter_tck
        from sextant.shared.connectome_types import FiberChunk

        stream = iter(iter_tck(self._path))
        first_id = 1
        while True:
            batch = list(islice(stream, chunk_size))
            if not batch:
                break
            packed = streamlines_to_fiber_chunk(batch)
            fiber_ids = np.arange(
                first_id,
                first_id + len(batch),
                dtype=np.int64,
            )
            fiber_ids.setflags(write=False)
            yield FiberChunk(
                fiber_ids=fiber_ids,
                point_offsets=packed.point_offsets,
                points=packed.points,
            )
            first_id += len(batch)
        observed = first_id - 1
        if observed != self._n_fibers:
            raise ValueError(
                f"TCK header count mismatch for {self._path}: "
                f"header={self._n_fibers}, observed={observed}"
            )


class HDF5Connectome:
    """Expose the tested Lead-DBS MATLAB-v7.3 adapter under the same interface."""

    def __init__(
        self,
        path: Path | str,
        *,
        connectome_id: str,
        sextant_python_root: Path | str,
    ) -> None:
        activate_sextant(sextant_python_root)
        from sextant.shared.connectome import LeadDBSHDF5Connectome

        self._adapter = LeadDBSHDF5Connectome(
            path,
            resource_id=connectome_id,
            resource_revision=1,
        )
        self._connectome_id = connectome_id

    @property
    def n_fibers(self) -> int:
        return int(self._adapter.metadata.n_fibers)

    @property
    def summary(self) -> ConnectomeSummary:
        metadata = self._adapter.metadata
        return ConnectomeSummary(
            connectome_id=self._connectome_id,
            representation="leaddbs_hdf5",
            path=str(metadata.source_path),
            n_fibers=int(metadata.n_fibers),
            fiber_id_base=1,
            weighted=False,
        )

    def iter_chunks(self, chunk_size: int) -> Iterator[Any]:
        yield from self._adapter.iter_chunks(chunk_size)


def open_connectome(
    path: Path | str,
    *,
    connectome_id: str,
    representation: str,
    sextant_python_root: Path | str,
) -> TCKConnectome | HDF5Connectome:
    """Open one registered unweighted streamline source."""

    if representation == "tck":
        return TCKConnectome(
            path,
            connectome_id=connectome_id,
            sextant_python_root=sextant_python_root,
        )
    if representation == "leaddbs_hdf5":
        return HDF5Connectome(
            path,
            connectome_id=connectome_id,
            sextant_python_root=sextant_python_root,
        )
    raise ValueError(f"Unsupported connectome representation: {representation}")
