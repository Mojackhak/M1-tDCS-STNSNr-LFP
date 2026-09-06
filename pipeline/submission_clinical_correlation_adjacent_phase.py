"""Publish adjacent-phase clinical-correlation fits for submission."""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from lfp_cohort.io import atomic_csv
from lfp_viz.config import VizConfig, load_viz_config
from lfp_viz.runner import run_viz
from pipeline.submission_clinical_correlation import (
    ROOT_TABLES,
    _override_output,
    _read_figure_manifest,
    _resolve_config_path,
    _significant_rows,
    _trash_existing,
)


STAGES = ("all", "fit", "publish")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _load_submission(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise TypeError(f"Submission configuration must contain a mapping: {resolved}")
    if data.get("submission") != {"id": "clinical-correlation-adjacent-phase"}:
        raise ValueError(
            "submission must equal {id: clinical-correlation-adjacent-phase}."
        )
    if data.get("configs") != {
        "fit": "configs/viz/submission_clinical_correlation_adjacent_phase_fit.yaml"
    }:
        raise ValueError("configs.fit does not match the adjacent-phase fit contract.")
    paths = data.get("paths")
    if not isinstance(paths, dict) or set(paths) != {"output_root"}:
        raise ValueError("paths must contain output_root only.")
    if not isinstance(paths["output_root"], str) or not paths["output_root"]:
        raise TypeError("paths.output_root must be a non-empty string.")
    if data.get("publish") != {"root_tables": list(ROOT_TABLES)}:
        raise ValueError("publish.root_tables does not match the submission contract.")
    if data.get("execution") != {
        "on_error": "fail",
        "retry": False,
        "cache": False,
        "resume": False,
    }:
        raise ValueError("execution does not match the failure contract.")
    return data


def _load_fit(
    submission_path: Path, data: dict[str, Any], output_root: Path
) -> VizConfig:
    fit_path = _resolve_config_path(submission_path, data["configs"]["fit"])
    fit = _override_output(load_viz_config(fit_path), output_root, "fit")
    if fit.stats_analysis_id != "clinical-correlation-adjacent-phase":
        raise ValueError("Adjacent-phase fit points to the wrong statistics analysis.")
    return fit


def _publication_summary(fit: VizConfig) -> dict[str, int]:
    correlations = pd.read_csv(fit.correlations_path)
    significant = _significant_rows(correlations)
    return {
        "correlations": len(correlations),
        "significant_p_holm": len(significant),
        "planned_fits": len(significant),
        "planned_figures": len(significant),
        "planned_root_tables": len(ROOT_TABLES),
    }


def _publish_root_tables(
    fit: VizConfig, output_root: Path, *, overwrite: bool
) -> dict[str, int]:
    correlations = pd.read_csv(fit.correlations_path)
    significant = _significant_rows(correlations)
    figures = _read_figure_manifest(
        fit.manifest_root / fit["manifests"]["figures"],
        output_root,
        "adjacent-phase-fit",
    )
    if len(figures) != len(significant):
        raise ValueError(
            "Adjacent-phase fit count differs from corrected correlations: "
            f"{len(figures)} != {len(significant)}."
        )
    if set(figures["CorrelationID"]) != set(significant["CorrelationID"]):
        raise ValueError(
            "Adjacent-phase fit identities differ from corrected correlations."
        )

    targets = {name: output_root / name for name in ROOT_TABLES}
    existing = [path for path in targets.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Adjacent-phase root table exists; pass --overwrite to replace it: "
            f"{existing[0]}"
        )
    if existing:
        _trash_existing(existing)
    output_root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(fit.correlations_path, targets["correlations.csv"])
    atomic_csv(significant, targets["significant-p-holm.csv"], overwrite=False)
    atomic_csv(figures, targets["figures.csv"], overwrite=False)
    return {
        "written_correlations": len(correlations),
        "written_significant_p_holm": len(significant),
        "written_figures": len(figures),
        "writes": len(ROOT_TABLES),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    submission_path = args.config.expanduser().resolve()
    data = _load_submission(submission_path)
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root
        else Path(data["paths"]["output_root"]).expanduser().resolve()
    )
    fit = _load_fit(submission_path, data, output_root)

    summary: dict[str, Any] = {
        "stage": args.stage,
        "output_root": str(output_root),
    }
    if args.stage in {"all", "fit"}:
        summary["fit"] = run_viz(
            fit, dry_run=args.dry_run, overwrite=args.overwrite
        )
    if args.stage in {"all", "publish"}:
        summary["publish"] = (
            _publication_summary(fit)
            if args.dry_run
            else _publish_root_tables(fit, output_root, overwrite=args.overwrite)
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
