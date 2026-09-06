"""Localization loading and Lead-DBS nonlinear coordinate flipping."""

from __future__ import annotations

import os
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from .anatomy import classify_region, parse_channel_side
from .contracts import RecordSpec
from .identity import casefold_identity, identity_equal
from .io import read_table


def _matlab_quote(value: str | Path) -> str:
    return str(value).replace("'", "''")


def _validate_leaddbs_provenance(
    flip_config: Mapping[str, Any],
) -> tuple[Path, str]:
    root = Path(flip_config["leaddbs_root"])
    configured_transform = Path(flip_config["transform_path"]).resolve()
    active_transform = (
        root
        / "templates"
        / "space"
        / str(flip_config["space"])
        / "fliplr"
        / "InverseComposite.nii.gz"
    ).resolve()
    if configured_transform != active_transform:
        raise ValueError(
            "coordinates.transform_path does not match the configured Lead-DBS "
            f"space: {configured_transform} != {active_transform}."
        )
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Could not read the configured Lead-DBS Git commit.\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    actual_commit = completed.stdout.strip()
    configured_commit = str(flip_config["leaddbs_commit"]).strip()
    if not actual_commit.startswith(configured_commit):
        raise ValueError(
            "coordinates.leaddbs_commit does not match the configured checkout: "
            f"expected {configured_commit}, found {actual_commit}."
        )
    return active_transform, actual_commit


def apply_coordinate_flip(
    contact_map: pd.DataFrame,
    config: Mapping[str, Any],
    temporary_parent: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add right-normalized coordinate columns without changing originals."""
    flip_config = config["coordinates"]
    out = contact_map.copy()
    provenance = dict(flip_config)
    provenance["n_coordinates"] = int(len(out))
    if not flip_config["enabled"]:
        for axis in ("x", "y", "z"):
            out[f"mni_{axis}_flip"] = out[f"mni_{axis}"].astype(float)
        provenance["status"] = "disabled_copy"
        return out, provenance
    if flip_config["backend"] != "matlab_leaddbs":
        raise ValueError(
            f"Unsupported coordinate flip backend: {flip_config['backend']!r}"
        )
    required_paths = [
        Path(flip_config["matlab_executable"]),
        Path(flip_config["helper_dir"]),
        Path(flip_config["leaddbs_root"]),
        Path(flip_config["transform_path"]),
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Coordinate flip dependency is missing: {missing}")
    active_transform, actual_commit = _validate_leaddbs_provenance(flip_config)
    provenance["leaddbs_commit_actual"] = actual_commit
    provenance["transform_path_actual"] = str(active_transform)

    temporary_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="coord-flip-", dir=temporary_parent
    ) as temp_name:
        temp_dir = Path(temp_name)
        input_csv = temp_dir / "coordinates.csv"
        output_csv = temp_dir / "coordinates_flipped.csv"
        matlab_table = out[
            ["ID", "Record", "Channel", "mni_x", "mni_y", "mni_z"]
        ].rename(columns={"mni_x": "MNI_x", "mni_y": "MNI_y", "mni_z": "MNI_z"})
        matlab_table.to_csv(input_csv, index=False)
        command = (
            f"addpath(genpath('{_matlab_quote(flip_config['leaddbs_root'])}'));"
            f"addpath(genpath('{_matlab_quote(flip_config['helper_dir'])}'),'-begin');"
            "active_space=ea_getspace;"
            f"if ~strcmp(active_space,'{_matlab_quote(flip_config['space'])}');"
            "error('Configured Lead-DBS space is not active.');end;"
            "active_transform=fullfile(ea_space,'fliplr',"
            "'InverseComposite.nii.gz');"
            f"if ~strcmp(active_transform,'{_matlab_quote(active_transform)}');"
            "error('Configured Lead-DBS flip transform is not active.');end;"
            "flip_coords_lr_nonlinear("
            f"'{_matlab_quote(input_csv)}','{_matlab_quote(flip_config['direction'])}',"
            f"'OutCsv','{_matlab_quote(output_csv)}','WriteNewColumns',true);"
        )
        environment = os.environ.copy()
        environment["LEADDBS_SPACE_OVERRIDE"] = str(flip_config["space"])
        completed = subprocess.run(
            [str(flip_config["matlab_executable"]), "-batch", command],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Lead-DBS coordinate flip failed with return code "
                f"{completed.returncode}.\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        flipped = pd.read_csv(output_csv)
        if len(flipped) != len(out):
            raise ValueError("Coordinate flip changed the number of coordinate rows.")
        for axis in ("x", "y", "z"):
            out[f"mni_{axis}_flip"] = pd.to_numeric(
                flipped[f"MNI_{axis}_flip"], errors="raise"
            ).to_numpy(dtype=float)
    provenance["status"] = "completed"
    return out, provenance


def build_contact_map(
    records: Sequence[RecordSpec],
    config: Mapping[str, Any],
    temporary_parent: Path,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    """Load localization and establish region, side, and flipped coordinates."""
    parts: list[pd.DataFrame] = []
    checks: list[dict[str, Any]] = []
    tolerance = float(config["anatomy"].get("coordinate_sign_tolerance", 0.0))
    required = {"channel", "mni_x", "mni_y", "mni_z", "SNr_in", "STN_in"}
    for record in records:
        path = record.path / "localize" / "channel_representative_coords.pkl"
        table = read_table(path)
        missing = required - set(table.columns)
        if missing:
            raise KeyError(
                f"Localization table {path} is missing columns: {sorted(missing)}"
            )
        table = table.copy()
        table["ID"] = record.subject
        table["Record"] = record.record
        table["Channel"] = table["channel"].astype(str)
        table["Region"] = [
            classify_region(stn, snr)
            for stn, snr in zip(table["STN_in"], table["SNr_in"])
        ]
        table["Side"] = [
            parse_channel_side(channel, config) for channel in table["Channel"]
        ]
        for row in table.itertuples(index=False):
            x = float(row.mni_x)
            sign_ok = (identity_equal(row.Side, "L") and x < -tolerance) or (
                identity_equal(row.Side, "R") and x > tolerance
            )
            checks.append(
                {
                    "ID": record.subject,
                    "Record": record.record,
                    "Channel": row.Channel,
                    "Side": row.Side,
                    "mni_x": x,
                    "coordinate_sign_ok": bool(sign_ok),
                }
            )
            if not sign_ok:
                raise ValueError(
                    "Coordinate sign disagrees with channel side for "
                    f"{record.subject}/{record.record}/{row.Channel}: x={x}."
                )
        optional = [
            column
            for column in ("space", "atlas", "anode", "cathode", "rep_coord")
            if column in table
        ]
        parts.append(
            table[
                [
                    "ID",
                    "Record",
                    "Channel",
                    "Region",
                    "Side",
                    "mni_x",
                    "mni_y",
                    "mni_z",
                    "SNr_in",
                    "STN_in",
                    *optional,
                ]
            ]
        )
    contact_map = pd.concat(parts, ignore_index=True)
    duplicate_keys = pd.DataFrame(
        {
            column: contact_map[column].map(casefold_identity)
            for column in ("ID", "Record", "Channel")
        }
    )
    if duplicate_keys.duplicated().any():
        raise ValueError("Localization tables contain duplicate record/channel rows.")
    contact_map, provenance = apply_coordinate_flip(
        contact_map, config, temporary_parent
    )
    return contact_map, checks, provenance
