"""Small atomic writers and recoverable replacement for generated artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd


def prepare_output(path: Path, *, overwrite: bool) -> None:
    """Prepare one generated path, moving an approved replacement to Trash."""

    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output exists; pass --overwrite to replace it: {path}"
            )
        trash = Path("/usr/bin/trash")
        if not trash.is_file():
            raise FileNotFoundError(f"macOS Trash command is unavailable: {trash}")
        subprocess.run([str(trash), str(path)], check=True)
    path.parent.mkdir(parents=True, exist_ok=True)


def atomic_csv(table: pd.DataFrame, path: Path, *, overwrite: bool) -> None:
    """Write one CSV atomically after recoverable replacement handling."""

    prepare_output(path, overwrite=overwrite)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temp_name)
    try:
        table.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(value: Any, path: Path, *, overwrite: bool) -> None:
    """Write one JSON document atomically."""

    prepare_output(path, overwrite=overwrite)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temp_name)
    try:
        temporary.write_text(
            json.dumps(value, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_npz(arrays: dict[str, np.ndarray], path: Path, *, overwrite: bool) -> None:
    """Write one compressed NumPy archive atomically."""

    prepare_output(path, overwrite=overwrite)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temp_name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_nifti(image: nib.spatialimages.SpatialImage, path: Path, *, overwrite: bool) -> None:
    """Write one NIfTI image atomically."""

    prepare_output(path, overwrite=overwrite)
    suffix = ".nii.gz" if path.name.endswith(".nii.gz") else path.suffix
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=suffix, dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temp_name)
    try:
        nib.save(image, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()

