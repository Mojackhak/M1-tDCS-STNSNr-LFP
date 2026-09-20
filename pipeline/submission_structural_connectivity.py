"""Publish authoritative structural-connectivity CSV tables for submission."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import filecmp
import json
from pathlib import Path
import plistlib
import shutil
import subprocess
from typing import Any

import pandas as pd
import yaml

from lfp_structural.artifacts import atomic_csv
from lfp_structural.config import StructuralConnectivityConfig, load_config
from lfp_structural.contact_lmm import (
    CONTACT_LMM_TABLES,
    FIELD_EXCESS_ATLAS_GRID_FIGURE_TYPE,
    FIELD_EXCESS_FINDER_ATLAS_GRID_FIT_COUNT,
    FIELD_EXCESS_FINDER_BAND,
    FIELD_EXCESS_FINDER_CONNECTOME_SOURCES,
    FIELD_EXCESS_FINDER_HEATMAP_COUNT,
    FIELD_EXCESS_FINDER_LOCAL_FEATURE_OUTPUT,
    FIELD_EXCESS_FINDER_LOCAL_METRIC,
    FIELD_EXCESS_FINDER_LOCAL_PHASES,
    FIELD_EXCESS_FINDER_LOCAL_REGION,
    FIELD_EXCESS_FINDER_METRICS,
    FIELD_EXCESS_FINDER_PHASE,
    FIELD_EXCESS_FINDER_POLAR,
    FIELD_EXCESS_FINDER_PREDICTORS,
    FIELD_EXCESS_METRIC,
)
from lfp_structural.cortical_allocation_lmm import (
    CORTICAL_ALLOCATION_OUTPUTS,
    CORTICAL_ALLOCATION_TABLES,
)


STAGES = (
    "publish",
    "contact-lmm",
    "contact-lmm-adjacent",
    "contact-lmm-adjacent-field-excess-atlas-grid-fits",
    "cortical-allocation-adjacent-figures",
    "finder-tags",
    "finder-contact-lmm",
)
SOURCE_TABLES = {
    "channel_lift.csv": "channel_lift",
    "channel_field_weighted_lift.csv": "channel_field_weighted_lift",
    "local_rows.csv": "local_rows",
    "connectivity_rows.csv": "connectivity_rows",
    "subject_aggregates.csv": "subject_aggregates",
    "correlations.csv": "correlations",
    "inputs.csv": "inputs_manifest",
    "outputs.csv": "outputs_manifest",
}
ROOT_TABLES = (*SOURCE_TABLES, *CORTICAL_ALLOCATION_TABLES, "tables.csv")
ATLAS_GRID_FIT_DEPENDENT_TABLES = ("outputs.csv", "tables.csv", "figures.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES, default="publish")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--figure-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _load_submission(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise TypeError(f"Submission configuration must contain a mapping: {resolved}")
    if data.get("submission") != {"id": "structural-connectivity"}:
        raise ValueError("submission must equal {id: structural-connectivity}.")
    if data.get("configs") != {"structural": "configs/structural_connectivity.yaml"}:
        raise ValueError("configs.structural does not match the registered contract.")
    paths = data.get("paths")
    if not isinstance(paths, dict) or set(paths) != {"output_root", "figure_root"}:
        raise ValueError("paths must contain output_root and figure_root only.")
    for key in ("output_root", "figure_root"):
        if not isinstance(paths[key], str) or not paths[key]:
            raise TypeError(f"paths.{key} must be a non-empty string.")
    expected_publish = {
        "root_tables": list(ROOT_TABLES),
        "contact_lmm_tables": list(CONTACT_LMM_TABLES),
        "contact_lmm_adjacent_tables": list(CONTACT_LMM_TABLES),
    }
    if data.get("publish") != expected_publish:
        raise ValueError("publish does not match the submission contract.")
    if data.get("execution") != {
        "on_error": "fail",
        "retry": False,
        "cache": False,
        "resume": False,
    }:
        raise ValueError("execution does not match the submission contract.")
    return data


def _resolve_config_path(submission_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (submission_path.parent.parent / path).resolve()


def _contact_figure_root(output_root: Path, figure_root: Path) -> Path:
    expected = output_root / "contact_lmm"
    if figure_root.resolve() != expected.resolve():
        raise ValueError(
            "figure_root must equal output_root/contact_lmm; "
            f"expected {expected}, received {figure_root}."
        )
    return expected


def _trash_existing(paths: list[Path]) -> None:
    if not paths:
        return
    trash = Path("/usr/bin/trash")
    if not trash.is_file():
        raise FileNotFoundError(f"Trash command is unavailable: {trash}")
    subprocess.run([str(trash), *(str(path) for path in paths)], check=True)


def _copy_registered_file(
    source: Path,
    target: Path,
    *,
    overwrite: bool,
) -> bool:
    if target.exists():
        if target.is_file() and filecmp.cmp(source, target, shallow=False):
            return False
        if not overwrite:
            raise FileExistsError(
                f"Submission file differs; pass --overwrite to replace it: {target}"
            )
        _trash_existing([target])
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return True


def _source_inventory(config: StructuralConnectivityConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for published_name, output_key in SOURCE_TABLES.items():
        source = config.output_path(output_key)
        if not source.is_file() or source.stat().st_size == 0:
            raise FileNotFoundError(
                f"Authoritative structural-connectivity table is missing: {source}"
            )
        table = pd.read_csv(source, low_memory=False)
        rows.append(
            {
                "Table": published_name,
                "SourcePath": str(source),
                "Rows": int(len(table)),
                "Columns": int(len(table.columns)),
            }
        )
    return pd.DataFrame(rows)


def _contact_lmm_inventory(config: StructuralConnectivityConfig) -> pd.DataFrame:
    return _contact_lmm_root_inventory(config.contact_lmm_root, "contact-level")


def _contact_lmm_root_inventory(root: Path, label: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name in CONTACT_LMM_TABLES:
        source = root / name
        if not source.is_file() or source.stat().st_size == 0:
            raise FileNotFoundError(
                f"Authoritative {label} mixed-model table is missing: {source}"
            )
        table = pd.read_csv(source, low_memory=False)
        rows.append(
            {
                "Table": name,
                "SourcePath": str(source),
                "Rows": int(len(table)),
                "Columns": int(len(table.columns)),
            }
        )
    return pd.DataFrame(rows)


def _cortical_allocation_inventory(
    config: StructuralConnectivityConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for published_name, relative_path in CORTICAL_ALLOCATION_OUTPUTS.items():
        source = config.cortical_allocation_lmm_root / relative_path
        if not source.is_file() or source.stat().st_size == 0:
            raise FileNotFoundError(
                f"Authoritative cortical-allocation table is missing: {source}"
            )
        table = pd.read_csv(source, low_memory=False)
        rows.append(
            {
                "Table": published_name,
                "SourcePath": str(source),
                "Rows": int(len(table)),
                "Columns": int(len(table.columns)),
            }
        )
    return pd.DataFrame(rows)


def _figure_inventory(config: StructuralConnectivityConfig) -> pd.DataFrame:
    return _registered_figure_inventory(
        config.contact_lmm_root / "figures.csv",
        "Contact-level",
    )


def _adjacent_figure_inventory(
    config: StructuralConnectivityConfig,
) -> pd.DataFrame:
    return _registered_figure_inventory(
        config.contact_lmm_adjacent_root / "figures.csv",
        "Adjacent-Phase contact-level",
    )


def _registered_figure_inventory(
    registry_path: Path,
    label: str,
) -> pd.DataFrame:
    if not registry_path.is_file() or registry_path.stat().st_size == 0:
        raise FileNotFoundError(f"{label} figure registry is missing: {registry_path}")
    registry = pd.read_csv(registry_path, low_memory=False)
    required = {"FigureID", "SourcePath", "PublishedRelativePath", "Status"}
    missing = sorted(required.difference(registry.columns))
    if missing:
        raise KeyError(f"{label} figure registry is missing columns: {missing}")
    if registry["PublishedRelativePath"].duplicated().any():
        raise ValueError(f"{label} figure registry has duplicate publication paths.")
    for row in registry.itertuples(index=False):
        relative = Path(str(row.PublishedRelativePath))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe published figure path: {relative}")
        source = Path(str(row.SourcePath))
        if not source.is_file() or source.suffix.lower() != ".pdf":
            raise FileNotFoundError(f"Registered {label} figure is missing: {source}")
    return registry


def _additional_figure_inventories(
    config: StructuralConnectivityConfig,
) -> dict[str, tuple[Path, pd.DataFrame]]:
    registries = {
        "spearman": config.summary_root / "spearman" / "figures.csv",
        "cortical_allocation": (config.cortical_allocation_lmm_root / "figures.csv"),
    }
    return {
        method: (
            path,
            _registered_figure_inventory(path, method.replace("_", " ").title()),
        )
        for method, path in registries.items()
    }


def publish_cortical_allocation_adjacent_figures(
    config: StructuralConnectivityConfig,
    output_root: Path,
    figure_root: Path,
    *,
    dry_run: bool,
    overwrite: bool,
) -> dict[str, Any]:
    """Publish only the six adjacent cortical-allocation heatmaps."""

    _contact_figure_root(output_root, figure_root)
    adjacent = _registered_figure_inventory(
        config.cortical_allocation_lmm_adjacent_root / "figures.csv",
        "Adjacent cortical-allocation",
    )
    if (
        len(adjacent) != 6
        or not adjacent["PublishedRelativePath"]
        .astype(str)
        .str.startswith("adjacent_phase/heatmap/")
        .all()
    ):
        raise ValueError(
            "The adjacent cortical-allocation registry must contain exactly six "
            "heatmaps under adjacent_phase/heatmap."
        )
    combined_registry_path = config.cortical_allocation_lmm_root / "figures.csv"
    combined = _registered_figure_inventory(
        combined_registry_path,
        "Combined cortical-allocation",
    )
    adjacent_keys = set(
        adjacent[["FigureID", "PublishedRelativePath", "SourcePath"]].itertuples(
            index=False,
            name=None,
        )
    )
    combined_keys = set(
        combined[["FigureID", "PublishedRelativePath", "SourcePath"]].itertuples(
            index=False,
            name=None,
        )
    )
    if not adjacent_keys.issubset(combined_keys):
        raise ValueError(
            "The combined cortical-allocation registry does not contain the "
            "adjacent heatmap registry."
        )

    publication_root = output_root / "cortical_allocation"
    copy_specs = [
        *[
            (
                Path(str(row.SourcePath)),
                publication_root / str(row.PublishedRelativePath),
            )
            for row in adjacent.itertuples(index=False)
        ],
        (combined_registry_path, publication_root / "figures.csv"),
    ]
    differing = [
        target
        for source, target in copy_specs
        if target.exists()
        and (not target.is_file() or not filecmp.cmp(source, target, shallow=False))
    ]
    if differing and not overwrite:
        raise FileExistsError(
            "Submission artifact differs; pass --overwrite to replace it: "
            f"{differing[0]}"
        )
    if dry_run:
        return {
            "figures": int(len(adjacent)),
            "combined_registry_rows": int(len(combined)),
            "writes": 0,
        }
    writes = sum(
        int(_copy_registered_file(source, target, overwrite=overwrite))
        for source, target in copy_specs
    )
    return {
        "figures": int(len(adjacent)),
        "combined_registry_rows": int(len(combined)),
        "writes": int(writes),
    }


def publish_contact_lmm(
    config: StructuralConnectivityConfig,
    output_root: Path,
    figure_root: Path,
    *,
    dry_run: bool,
    overwrite: bool,
) -> dict[str, Any]:
    figure_root = _contact_figure_root(output_root, figure_root)
    contact_inventory = _contact_lmm_inventory(config)
    figures = _figure_inventory(config)
    contact_output_root = output_root / "contact_lmm"
    contact_inventory["PublishedPath"] = contact_inventory["Table"].map(
        lambda name: str(contact_output_root / name)
    )
    figures["PublishedPath"] = figures["PublishedRelativePath"].map(
        lambda value: str(figure_root / str(value))
    )
    copy_specs = [
        *[
            (config.contact_lmm_root / name, contact_output_root / name)
            for name in CONTACT_LMM_TABLES
        ],
        *[
            (Path(str(row.SourcePath)), Path(str(row.PublishedPath)))
            for row in figures.itertuples(index=False)
        ],
    ]
    differing = [
        target
        for source, target in copy_specs
        if target.exists()
        and (not target.is_file() or not filecmp.cmp(source, target, shallow=False))
    ]
    if differing and not overwrite:
        raise FileExistsError(
            "Submission artifact differs; pass --overwrite to replace it: "
            f"{differing[0]}"
        )
    if dry_run:
        return {
            "contact_lmm_tables": int(len(CONTACT_LMM_TABLES)),
            "contact_lmm_source_rows": int(contact_inventory["Rows"].sum()),
            "figures": int(len(figures)),
            "writes": 0,
        }
    contact_output_root.mkdir(parents=True, exist_ok=True)
    writes = sum(
        int(_copy_registered_file(source, target, overwrite=overwrite))
        for source, target in copy_specs
    )
    return {
        "contact_lmm_tables": int(len(CONTACT_LMM_TABLES)),
        "contact_lmm_source_rows": int(contact_inventory["Rows"].sum()),
        "figures": int(len(figures)),
        "writes": int(writes),
    }


def publish_contact_lmm_adjacent(
    config: StructuralConnectivityConfig,
    output_root: Path,
    figure_root: Path,
    *,
    dry_run: bool,
    overwrite: bool,
) -> dict[str, Any]:
    figure_root = _contact_figure_root(output_root, figure_root) / "adjacent_phase"
    source_root = config.contact_lmm_adjacent_root
    inventory = _contact_lmm_root_inventory(
        source_root,
        "adjacent-Phase contact-level",
    )
    figures = _adjacent_figure_inventory(config)
    output_branch = output_root / "contact_lmm" / "adjacent_phase"
    inventory["PublishedPath"] = inventory["Table"].map(
        lambda name: str(output_branch / name)
    )
    figures["PublishedPath"] = figures["PublishedRelativePath"].map(
        lambda value: str(figure_root / str(value))
    )
    copy_specs = [
        *[(source_root / name, output_branch / name) for name in CONTACT_LMM_TABLES],
        *[
            (Path(str(row.SourcePath)), Path(str(row.PublishedPath)))
            for row in figures.itertuples(index=False)
        ],
    ]
    registered_targets = {
        Path(str(row.PublishedPath)).resolve()
        for row in figures.itertuples(index=False)
    }
    stale_field_excess = [
        path
        for path in (output_branch / "heatmap").glob("*lift-FieldExcess_*.pdf")
        if not path.name.startswith("._")
        if path.resolve() not in registered_targets
    ]
    differing = [
        target
        for source, target in copy_specs
        if target.exists()
        and (not target.is_file() or not filecmp.cmp(source, target, shallow=False))
    ]
    differing.extend(stale_field_excess)
    if differing and not overwrite:
        raise FileExistsError(
            "Submission artifact differs; pass --overwrite to replace it: "
            f"{differing[0]}"
        )
    if dry_run:
        return {
            "contact_lmm_adjacent_tables": int(len(CONTACT_LMM_TABLES)),
            "contact_lmm_adjacent_source_rows": int(inventory["Rows"].sum()),
            "figures": int(len(figures)),
            "writes": 0,
        }
    output_branch.mkdir(parents=True, exist_ok=True)
    writes = sum(
        int(_copy_registered_file(source, target, overwrite=overwrite))
        for source, target in copy_specs
    )
    _trash_existing(stale_field_excess)
    writes += len(stale_field_excess)
    return {
        "contact_lmm_adjacent_tables": int(len(CONTACT_LMM_TABLES)),
        "contact_lmm_adjacent_source_rows": int(inventory["Rows"].sum()),
        "figures": int(len(figures)),
        "writes": int(writes),
    }


def publish_contact_lmm_adjacent_field_excess_atlas_grid_fits(
    config: StructuralConnectivityConfig,
    output_root: Path,
    figure_root: Path,
    *,
    dry_run: bool,
    overwrite: bool,
) -> dict[str, Any]:
    publication_root = _contact_figure_root(output_root, figure_root) / "adjacent_phase"
    source_root = config.contact_lmm_adjacent_root
    figures = _adjacent_figure_inventory(config)
    selected = _field_excess_finder_atlas_grid_rows(figures).copy()
    selected["PublishedPath"] = selected["PublishedRelativePath"].map(
        lambda value: str(publication_root / str(value))
    )

    registry_specs = []
    for name in ATLAS_GRID_FIT_DEPENDENT_TABLES:
        source = source_root / name
        if not source.is_file() or source.stat().st_size == 0:
            raise FileNotFoundError(
                f"Atlas-grid dependent registry is missing: {source}"
            )
        registry_specs.append((source, publication_root / name))
    figure_specs = [
        (Path(str(row.SourcePath)), Path(str(row.PublishedPath)))
        for row in selected.itertuples(index=False)
    ]
    copy_specs = [*figure_specs, *registry_specs]
    differing = [
        target
        for source, target in copy_specs
        if target.exists()
        and (not target.is_file() or not filecmp.cmp(source, target, shallow=False))
    ]
    if differing and not overwrite:
        raise FileExistsError(
            "Submission artifact differs; pass --overwrite to replace it: "
            f"{differing[0]}"
        )
    if dry_run:
        return {
            "atlas_grid_fit_figures": int(len(selected)),
            "dependent_registry_tables": int(len(registry_specs)),
            "differing_files": int(len(differing)),
            "writes": 0,
        }

    publication_root.mkdir(parents=True, exist_ok=True)
    writes = sum(
        int(_copy_registered_file(source, target, overwrite=overwrite))
        for source, target in copy_specs
    )
    return {
        "atlas_grid_fit_figures": int(len(selected)),
        "dependent_registry_tables": int(len(registry_specs)),
        "differing_files": int(len(differing)),
        "writes": int(writes),
    }


def publish_tables(
    config: StructuralConnectivityConfig,
    output_root: Path,
    figure_root: Path,
    *,
    dry_run: bool,
    overwrite: bool,
) -> dict[str, Any]:
    figure_root = _contact_figure_root(output_root, figure_root)
    inventory = _source_inventory(config)
    cortical_inventory = _cortical_allocation_inventory(config)
    contact_inventory = _contact_lmm_inventory(config)
    adjacent_contact_inventory = _contact_lmm_root_inventory(
        config.contact_lmm_adjacent_root,
        "adjacent-Phase contact-level",
    )
    figures = _figure_inventory(config)
    adjacent_figures = _adjacent_figure_inventory(config)
    additional_figures = _additional_figure_inventories(config)
    inventory = pd.concat([inventory, cortical_inventory], ignore_index=True)
    inventory["PublishedPath"] = inventory["Table"].map(
        lambda name: str(output_root / name)
    )
    contact_output_root = output_root / "contact_lmm"
    contact_inventory["PublishedPath"] = contact_inventory["Table"].map(
        lambda name: str(contact_output_root / name)
    )
    figures["PublishedPath"] = figures["PublishedRelativePath"].map(
        lambda value: str(figure_root / str(value))
    )
    adjacent_contact_output_root = contact_output_root / "adjacent_phase"
    adjacent_contact_inventory["PublishedPath"] = adjacent_contact_inventory[
        "Table"
    ].map(lambda name: str(adjacent_contact_output_root / name))
    adjacent_figures["PublishedPath"] = adjacent_figures["PublishedRelativePath"].map(
        lambda value: str(adjacent_contact_output_root / str(value))
    )
    for method, (_, registry) in additional_figures.items():
        method_root = output_root / method
        registry["PublishedPath"] = registry["PublishedRelativePath"].map(
            lambda value, root=method_root: str(root / str(value))
        )
    copy_specs = [
        *[
            (config.output_path(output_key), output_root / published_name)
            for published_name, output_key in SOURCE_TABLES.items()
        ],
        *[
            (
                config.cortical_allocation_lmm_root / relative_path,
                output_root / published_name,
            )
            for published_name, relative_path in CORTICAL_ALLOCATION_OUTPUTS.items()
        ],
        *[
            (config.contact_lmm_root / name, contact_output_root / name)
            for name in CONTACT_LMM_TABLES
        ],
        *[
            (Path(str(row.SourcePath)), Path(str(row.PublishedPath)))
            for row in figures.itertuples(index=False)
        ],
        *[
            (
                config.contact_lmm_adjacent_root / name,
                adjacent_contact_output_root / name,
            )
            for name in CONTACT_LMM_TABLES
        ],
        *[
            (Path(str(row.SourcePath)), Path(str(row.PublishedPath)))
            for row in adjacent_figures.itertuples(index=False)
        ],
        *[
            (registry_path, output_root / method / "figures.csv")
            for method, (registry_path, _) in additional_figures.items()
        ],
        *[
            (Path(str(row.SourcePath)), Path(str(row.PublishedPath)))
            for _, registry in additional_figures.values()
            for row in registry.itertuples(index=False)
        ],
    ]
    differing = [
        target
        for source, target in copy_specs
        if target.exists()
        and (not target.is_file() or not filecmp.cmp(source, target, shallow=False))
    ]
    table_registry_path = output_root / "tables.csv"
    registry_matches = table_registry_path.is_file() and (
        table_registry_path.read_bytes()
        == inventory.to_csv(index=False).encode("utf-8")
    )
    if table_registry_path.exists() and not registry_matches:
        differing.append(table_registry_path)
    if differing and not overwrite:
        raise FileExistsError(
            "Submission output differs or its registry must be regenerated; pass "
            f"--overwrite to replace it: {differing[0]}"
        )
    if dry_run:
        additional_count = sum(
            len(registry) for _, registry in additional_figures.values()
        )
        return {
            "root_tables": int(len(ROOT_TABLES)),
            "root_source_rows": int(inventory["Rows"].sum()),
            "cortical_allocation_tables": int(len(CORTICAL_ALLOCATION_TABLES)),
            "cortical_allocation_source_rows": int(cortical_inventory["Rows"].sum()),
            "contact_lmm_tables": int(len(CONTACT_LMM_TABLES)),
            "contact_lmm_source_rows": int(contact_inventory["Rows"].sum()),
            "contact_lmm_figures": int(len(figures)),
            "contact_lmm_adjacent_tables": int(len(CONTACT_LMM_TABLES)),
            "contact_lmm_adjacent_source_rows": int(
                adjacent_contact_inventory["Rows"].sum()
            ),
            "contact_lmm_adjacent_figures": int(len(adjacent_figures)),
            "spearman_figures": int(len(additional_figures["spearman"][1])),
            "cortical_allocation_figures": int(
                len(additional_figures["cortical_allocation"][1])
            ),
            "figures": int(len(figures) + len(adjacent_figures) + additional_count),
            "writes": 0,
        }
    output_root.mkdir(parents=True, exist_ok=True)
    writes = 0
    for published_name, output_key in SOURCE_TABLES.items():
        writes += int(
            _copy_registered_file(
                config.output_path(output_key),
                output_root / published_name,
                overwrite=overwrite,
            )
        )
    for published_name, relative_path in CORTICAL_ALLOCATION_OUTPUTS.items():
        writes += int(
            _copy_registered_file(
                config.cortical_allocation_lmm_root / relative_path,
                output_root / published_name,
                overwrite=overwrite,
            )
        )
    if not registry_matches:
        atomic_csv(inventory, table_registry_path, overwrite=overwrite)
        writes += 1
    contact_output_root.mkdir(parents=True, exist_ok=True)
    for name in CONTACT_LMM_TABLES:
        writes += int(
            _copy_registered_file(
                config.contact_lmm_root / name,
                contact_output_root / name,
                overwrite=overwrite,
            )
        )
    for row in figures.itertuples(index=False):
        target = Path(str(row.PublishedPath))
        writes += int(
            _copy_registered_file(
                Path(str(row.SourcePath)),
                target,
                overwrite=overwrite,
            )
        )
    adjacent_contact_output_root.mkdir(parents=True, exist_ok=True)
    for name in CONTACT_LMM_TABLES:
        writes += int(
            _copy_registered_file(
                config.contact_lmm_adjacent_root / name,
                adjacent_contact_output_root / name,
                overwrite=overwrite,
            )
        )
    for row in adjacent_figures.itertuples(index=False):
        writes += int(
            _copy_registered_file(
                Path(str(row.SourcePath)),
                Path(str(row.PublishedPath)),
                overwrite=overwrite,
            )
        )
    for method, (registry_path, registry) in additional_figures.items():
        method_root = output_root / method
        writes += int(
            _copy_registered_file(
                registry_path,
                method_root / "figures.csv",
                overwrite=overwrite,
            )
        )
        for row in registry.itertuples(index=False):
            writes += int(
                _copy_registered_file(
                    Path(str(row.SourcePath)),
                    Path(str(row.PublishedPath)),
                    overwrite=overwrite,
                )
            )
    additional_count = sum(len(registry) for _, registry in additional_figures.values())
    return {
        "root_tables": int(len(ROOT_TABLES)),
        "root_source_rows": int(inventory["Rows"].sum()),
        "cortical_allocation_tables": int(len(CORTICAL_ALLOCATION_TABLES)),
        "cortical_allocation_source_rows": int(cortical_inventory["Rows"].sum()),
        "contact_lmm_tables": int(len(CONTACT_LMM_TABLES)),
        "contact_lmm_source_rows": int(contact_inventory["Rows"].sum()),
        "contact_lmm_figures": int(len(figures)),
        "contact_lmm_adjacent_tables": int(len(CONTACT_LMM_TABLES)),
        "contact_lmm_adjacent_source_rows": int(
            adjacent_contact_inventory["Rows"].sum()
        ),
        "contact_lmm_adjacent_figures": int(len(adjacent_figures)),
        "spearman_figures": int(len(additional_figures["spearman"][1])),
        "cortical_allocation_figures": int(
            len(additional_figures["cortical_allocation"][1])
        ),
        "figures": int(len(figures) + len(adjacent_figures) + additional_count),
        "writes": int(writes),
    }


FINDER_TAG_ATTRIBUTE = "com.apple.metadata:_kMDItemUserTags"
FINDER_TAGS = {
    "spearman": "Spearman",
    "contact_lmm": "Contact-LMM",
    "cortical_allocation": "Cortical-allocation",
}


def _published_registry(root: Path, method: str) -> pd.DataFrame:
    path = root / method / "figures.csv"
    label = f"Published {method.replace('_', ' ')} figure registry"
    registry = _read_published_csv(
        path,
        {"FigureID", "PublishedRelativePath", "Status"},
        label,
    )
    if registry["PublishedRelativePath"].duplicated().any():
        raise ValueError(f"{label} has duplicate publication paths.")
    for value in registry["PublishedRelativePath"]:
        relative = Path(str(value))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe published figure path: {relative}")
    return registry


def _field_excess_finder_atlas_grid_rows(
    registry: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "FigureType",
        "Phase",
        "LiftMetric",
        "LiftPredictor",
        "OutcomeDomain",
        "ConnectomeSource",
        "Polar",
        "Metric",
        "FeatureOutput",
        "Region",
        "PairRegion",
        "Band",
        "ConnectomeSources",
        "Layout",
    }
    if not required.issubset(registry.columns):
        raise KeyError(
            "Contact-LMM figure registry is missing columns: "
            f"{sorted(required.difference(registry.columns))}"
        )
    common_fit_mask = (
        registry["FigureType"].eq(FIELD_EXCESS_ATLAS_GRID_FIGURE_TYPE)
        & registry["LiftMetric"].eq(FIELD_EXCESS_METRIC)
        & registry["Polar"].eq(FIELD_EXCESS_FINDER_POLAR)
        & registry["Band"].eq(FIELD_EXCESS_FINDER_BAND)
        & registry["Layout"].eq("atlas-grid")
    )
    connectivity_fit_mask = (
        common_fit_mask
        & registry["OutcomeDomain"].eq("connectivity")
        & registry["LiftPredictor"].isin(FIELD_EXCESS_FINDER_PREDICTORS)
        & registry["Phase"].eq(FIELD_EXCESS_FINDER_PHASE)
        & registry["Metric"].isin(FIELD_EXCESS_FINDER_METRICS)
        & registry["PairRegion"].eq("SNr-STN")
    )
    local_fit_mask = (
        common_fit_mask
        & registry["OutcomeDomain"].eq("local")
        & registry["LiftPredictor"].eq(FIELD_EXCESS_FINDER_LOCAL_REGION)
        & registry["Phase"].isin(FIELD_EXCESS_FINDER_LOCAL_PHASES)
        & registry["Metric"].eq(FIELD_EXCESS_FINDER_LOCAL_METRIC)
        & registry["FeatureOutput"].eq(FIELD_EXCESS_FINDER_LOCAL_FEATURE_OUTPUT)
        & registry["Region"].eq(FIELD_EXCESS_FINDER_LOCAL_REGION)
    )
    fit_mask = connectivity_fit_mask | local_fit_mask
    selected = registry.loc[fit_mask].copy()
    fit_count = int(len(selected))
    if fit_count != FIELD_EXCESS_FINDER_ATLAS_GRID_FIT_COUNT:
        raise ValueError(
            "Published Contact-LMM fit selection differs from the approved count: "
            f"expected {FIELD_EXCESS_FINDER_ATLAS_GRID_FIT_COUNT}, got {fit_count}."
        )
    connectivity_cells = (
        registry.loc[connectivity_fit_mask]
        .groupby(["LiftPredictor", "Metric"], dropna=False)
        .size()
    )
    if len(connectivity_cells) != 6 or not connectivity_cells.eq(1).all():
        raise ValueError(
            "Published Contact-LMM atlas-grid connectivity selection must contain "
            "one PDF per predictor and metric cell."
        )
    local_cells = registry.loc[local_fit_mask].groupby("Phase", dropna=False).size()
    if (
        tuple(sorted(local_cells.index.astype(str)))
        != tuple(sorted(FIELD_EXCESS_FINDER_LOCAL_PHASES))
        or not local_cells.eq(1).all()
    ):
        raise ValueError(
            "Published Contact-LMM local atlas-grid selection must contain the "
            "Late and Post SNr burst-amplitude PDFs."
        )
    expected_sources = ";".join(FIELD_EXCESS_FINDER_CONNECTOME_SOURCES)
    if not selected["ConnectomeSources"].eq(expected_sources).all():
        raise ValueError(
            "Published Contact-LMM atlas-grid fits must contain all four "
            "connectome sources in the registered order."
        )
    return selected


def _contact_finder_targets(root: Path) -> list[Path]:
    registry = _published_registry(root.parent, root.name)
    fits = _field_excess_finder_atlas_grid_rows(registry)
    heatmaps = registry.loc[
        registry["FigureType"].eq("heatmap")
        & registry["LiftMetric"].eq(FIELD_EXCESS_METRIC)
    ]
    heatmap_count = int(len(heatmaps))
    if heatmap_count != FIELD_EXCESS_FINDER_HEATMAP_COUNT:
        raise ValueError(
            "Published Contact-LMM heatmap selection differs from the approved "
            f"count: expected {FIELD_EXCESS_FINDER_HEATMAP_COUNT}, got "
            f"{heatmap_count}."
        )
    selected = pd.concat([heatmaps, fits], ignore_index=True)
    return [root / str(relative) for relative in selected["PublishedRelativePath"]]


def _finder_tag_targets(output_root: Path) -> dict[str, list[Path]]:
    targets: dict[str, list[Path]] = {}
    for method in ("spearman", "cortical_allocation"):
        registry = _published_registry(output_root, method)
        required_roles = {
            "all_heatmap",
            "corrected_significant_connectivity_fit",
        }
        if "SelectionRole" not in registry.columns:
            raise KeyError(f"Published {method} registry is missing SelectionRole.")
        unsupported = sorted(set(registry["SelectionRole"]) - required_roles)
        if unsupported:
            raise ValueError(
                f"Published {method} registry has unsupported roles: {unsupported}"
            )
        targets[method] = [
            output_root / method / str(relative)
            for relative in registry["PublishedRelativePath"]
        ]

    contact_root = output_root / "contact_lmm" / "adjacent_phase"
    targets["contact_lmm"] = _contact_finder_targets(contact_root)
    all_paths = [path for paths in targets.values() for path in paths]
    if len(all_paths) != len(set(all_paths)):
        raise ValueError("A published PDF was selected for multiple method tags.")
    missing = [path for path in all_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Published figure selected for tagging is missing: {missing[0]}"
        )
    return targets


def _read_published_csv(
    path: Path,
    required: set[str],
    label: str,
) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{label} is missing: {path}")
    table = pd.read_csv(path, low_memory=False)
    missing = sorted(required.difference(table.columns))
    if missing:
        raise KeyError(f"{label} is missing columns: {missing}")
    return table


def _read_finder_tags(path: Path) -> list[str]:
    command = ["/usr/bin/xattr", "-px", FINDER_TAG_ATTRIBUTE, str(path)]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        if "No such xattr" in result.stderr:
            return []
        raise subprocess.CalledProcessError(
            result.returncode,
            command,
            output=result.stdout,
            stderr=result.stderr,
        )
    payload = bytes.fromhex("".join(result.stdout.split()))
    values = plistlib.loads(payload)
    if not isinstance(values, list) or not all(
        isinstance(value, str) for value in values
    ):
        raise TypeError(f"Finder tag attribute is not a string array: {path}")
    return values


def _finder_tag_label(value: str) -> str:
    return value.split("\n", maxsplit=1)[0]


def _add_finder_tag(path: Path, tag: str) -> bool:
    values = _read_finder_tags(path)
    method_labels = set(FINDER_TAGS.values())
    updated: list[str] = []
    expected_seen = False
    for value in values:
        label = _finder_tag_label(value)
        if label not in method_labels:
            updated.append(value)
        elif label == tag and not expected_seen:
            updated.append(value)
            expected_seen = True
    if not expected_seen:
        updated.append(f"{tag}\n0")
    if updated == values:
        return False
    payload = plistlib.dumps(updated, fmt=plistlib.FMT_BINARY)
    subprocess.run(
        [
            "/usr/bin/xattr",
            "-wx",
            FINDER_TAG_ATTRIBUTE,
            payload.hex(),
            str(path),
        ],
        check=True,
    )
    written_method_tags = [
        _finder_tag_label(value)
        for value in _read_finder_tags(path)
        if _finder_tag_label(value) in method_labels
    ]
    if written_method_tags != [tag]:
        raise OSError(f"Finder tag verification failed for {path}: {tag}")
    return True


def _remove_finder_tag(path: Path, tag: str) -> bool:
    values = _read_finder_tags(path)
    updated = [value for value in values if _finder_tag_label(value) != tag]
    if updated == values:
        return False
    if updated:
        payload = plistlib.dumps(updated, fmt=plistlib.FMT_BINARY)
        subprocess.run(
            [
                "/usr/bin/xattr",
                "-wx",
                FINDER_TAG_ATTRIBUTE,
                payload.hex(),
                str(path),
            ],
            check=True,
        )
    else:
        subprocess.run(
            ["/usr/bin/xattr", "-d", FINDER_TAG_ATTRIBUTE, str(path)],
            check=True,
        )
    if tag in {_finder_tag_label(value) for value in _read_finder_tags(path)}:
        raise OSError(f"Finder tag removal verification failed for {path}: {tag}")
    return True


def _tagged_pdf_paths(output_root: Path, tag: str) -> list[Path]:
    tagged: list[Path] = []
    for path in sorted(output_root.rglob("*.pdf")):
        if path.name.startswith("._") or not path.is_file():
            continue
        labels = {_finder_tag_label(value) for value in _read_finder_tags(path)}
        if tag in labels:
            tagged.append(path)
    return tagged


def refresh_contact_lmm_finder_tag(
    output_root: Path,
    *,
    dry_run: bool,
) -> dict[str, Any]:
    """Replace the live Contact-LMM Finder-tag set with the approved 12 PDFs."""

    tag = FINDER_TAGS["contact_lmm"]
    targets = _contact_finder_targets(output_root / "contact_lmm" / "adjacent_phase")
    current = _tagged_pdf_paths(output_root, tag)
    summary: dict[str, Any] = {
        "status": "ready" if dry_run else "complete",
        "tag": tag,
        "currently_tagged": int(len(current)),
        "selected": int(len(targets)),
        "removed": 0,
        "added": 0,
        "writes": 0,
    }
    if dry_run:
        return summary
    removed = sum(int(_remove_finder_tag(path, tag)) for path in current)
    added = sum(int(_add_finder_tag(path, tag)) for path in targets)
    final = _tagged_pdf_paths(output_root, tag)
    if set(final) != set(targets):
        raise OSError(
            "Final Contact-LMM Finder-tag set does not match the approved targets."
        )
    summary["removed"] = int(removed)
    summary["added"] = int(added)
    summary["writes"] = int(removed + added)
    return summary


def apply_finder_tags(output_root: Path, *, dry_run: bool) -> dict[str, Any]:
    """Apply the three method tags to the registered published PDF subsets."""

    targets = _finder_tag_targets(output_root)
    summary: dict[str, Any] = {
        "status": "ready" if dry_run else "complete",
        "selected": {
            FINDER_TAGS[method]: len(paths) for method, paths in targets.items()
        },
        "writes": 0,
    }
    if dry_run:
        return summary
    contact_refresh = refresh_contact_lmm_finder_tag(output_root, dry_run=False)
    writes = int(contact_refresh["writes"])
    for method, paths in targets.items():
        if method == "contact_lmm":
            continue
        tag = FINDER_TAGS[method]
        for path in paths:
            writes += int(_add_finder_tag(path, tag))
    summary["writes"] = writes
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    submission_path = args.config.expanduser().resolve()
    data = _load_submission(submission_path)
    structural_path = _resolve_config_path(
        submission_path,
        data["configs"]["structural"],
    )
    structural = load_config(structural_path)
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root
        else Path(data["paths"]["output_root"]).expanduser().resolve()
    )
    figure_root = (
        args.figure_root.expanduser().resolve()
        if args.figure_root
        else Path(data["paths"]["figure_root"]).expanduser().resolve()
    )
    if args.stage == "finder-tags":
        result = apply_finder_tags(output_root, dry_run=args.dry_run)
    elif args.stage == "finder-contact-lmm":
        result = refresh_contact_lmm_finder_tag(
            output_root,
            dry_run=args.dry_run,
        )
    elif args.stage == "cortical-allocation-adjacent-figures":
        result = publish_cortical_allocation_adjacent_figures(
            structural,
            output_root,
            figure_root,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
    else:
        publishers = {
            "publish": publish_tables,
            "contact-lmm": publish_contact_lmm,
            "contact-lmm-adjacent": publish_contact_lmm_adjacent,
            "contact-lmm-adjacent-field-excess-atlas-grid-fits": (
                publish_contact_lmm_adjacent_field_excess_atlas_grid_fits
            ),
        }
        publisher = publishers[args.stage]
        result = publisher(
            structural,
            output_root,
            figure_root,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
    summary = {
        "stage": args.stage,
        "output_root": str(output_root),
        "publish": result,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
