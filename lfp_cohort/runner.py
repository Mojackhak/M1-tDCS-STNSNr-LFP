"""Pipeline stage orchestration without scientific processing logic."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .feature_outputs import iter_feature_output_specs
from .config import CohortConfig
from .contracts import RunState, SOURCE_STAGES
from .coordinates import build_contact_map
from .discovery import discover_records, dry_run_summary
from .io import aggregate_path, stage_path
from .provenance import finalize_run, validate_outputs
from .stages.aggregate import run_aggregate
from .stages.merge import run_merge
from .stages.normalize import run_normalize
from .stages.outliers import run_outliers
from .stages.split import run_split
from .stages.transform import run_transform


def _standalone_prerequisites(config: CohortConfig, stage: str) -> list[Path]:
    specs = list(iter_feature_output_specs(config))
    root = config.output_root
    if stage == "transform":
        return [stage_path(root, "merge", spec, config) for spec in specs]
    if stage == "normalize":
        return [stage_path(root, "transform", spec, config) for spec in specs]
    return []


def _source_branch_paths(
    config: CohortConfig, stage: str, source_stage: str
) -> list[Path]:
    specs = list(iter_feature_output_specs(config))
    if stage == "aggregate":
        return [
            stage_path(config.output_root, source_stage, spec, config) for spec in specs
        ]
    return [
        aggregate_path(config.output_root, source_stage, "contact", spec, config)
        for spec in specs
        if spec.domain == "connectivity"
    ]


def _available_source_branches(config: CohortConfig, stage: str) -> tuple[str, ...]:
    available: list[str] = []
    for source_stage in SOURCE_STAGES:
        paths = _source_branch_paths(config, stage, source_stage)
        present = [path.is_file() for path in paths]
        if not any(present):
            continue
        if not all(present):
            missing = [path for path, exists in zip(paths, present) if not exists]
            preview = ", ".join(str(path) for path in missing[:3])
            raise FileNotFoundError(
                f"Stage {stage!r} found a partial {source_stage!r} branch with "
                f"{len(missing)} missing files: {preview}"
            )
        available.append(source_stage)
    if not available:
        raise FileNotFoundError(
            f"Stage {stage!r} found no complete available source branch."
        )
    return tuple(available)


def _check_prerequisites(config: CohortConfig, stage: str) -> tuple[str, ...]:
    if stage in {"aggregate", "split"}:
        return _available_source_branches(config, stage)
    missing = [
        path for path in _standalone_prerequisites(config, stage) if not path.is_file()
    ]
    if missing:
        preview = ", ".join(str(path) for path in missing[:3])
        raise FileNotFoundError(
            f"Stage {stage!r} is missing {len(missing)} direct prerequisite files: "
            f"{preview}"
        )
    return SOURCE_STAGES


def run_pipeline(
    config: CohortConfig,
    *,
    stage: str = "all",
    dry_run: bool = False,
    overwrite: bool = False,
    record_selectors: Sequence[str] = (),
) -> dict[str, Any] | None:
    """Discover once, then orchestrate only the selected stage chain."""
    if record_selectors and stage not in {"merge", "all"}:
        raise ValueError("--record is supported only with --stage merge or all.")
    records = discover_records(config, tuple(record_selectors))
    if dry_run:
        summary = dry_run_summary(records, config)
        print(json.dumps(summary, indent=2))
        return summary
    source_stages = SOURCE_STAGES
    if stage != "all":
        source_stages = _check_prerequisites(config, stage)

    effective_overwrite = bool(overwrite or config["execution"].get("overwrite", False))
    if stage == "transform" and not effective_overwrite:
        raise FileExistsError(
            "Standalone transform refreshes existing merge outlier decisions; pass "
            "--overwrite to authorize both branch updates."
        )

    state = RunState(
        config=config,
        overwrite=effective_overwrite,
        invocation_stage=stage,
        record_selectors=tuple(record_selectors),
    )
    if stage in {"merge", "all"}:
        manifest_dir = (
            state.output_root / config["execution"]["directories"]["manifest"]
        )
        contact_map, coordinate_checks, provenance = build_contact_map(
            records, config, manifest_dir
        )
        state.flip_provenance = provenance
        state.add_qc("coordinate_checks", coordinate_checks)
        run_merge(records, contact_map, state)
    if stage == "merge":
        run_outliers(state, update_transform=False)
    if stage in {"transform", "all"}:
        run_transform(state)
        run_outliers(state, update_transform=True)
    if stage in {"normalize", "all"}:
        run_normalize(state)
    if stage in {"aggregate", "all"}:
        run_aggregate(state, source_stages)
    if stage in {"split", "all"}:
        run_split(state, source_stages)
    validate_outputs(state)
    finalize_run(state, records)
    return None
