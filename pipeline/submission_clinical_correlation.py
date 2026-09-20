"""Publish corrected clinical-correlation tables and figures for submission."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from lfp_cohort.io import atomic_csv
from lfp_viz.config import VizConfig, load_viz_config
from lfp_viz.runner import run_viz


STAGES = ("all", "heatmap", "scale-heatmap", "fit", "publish")
ROOT_TABLES = ("correlations.csv", "significant-p-holm.csv", "figures.csv")


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
    if data.get("submission") != {"id": "clinical-correlation"}:
        raise ValueError("submission must equal {id: clinical-correlation}.")
    configs = data.get("configs")
    if not isinstance(configs, dict) or set(configs) != {
        "heatmap",
        "scale_heatmap",
        "fit",
    }:
        raise ValueError("configs must contain heatmap, scale_heatmap, and fit paths.")
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
        raise ValueError("execution does not match the submission contract.")
    return data


def _resolve_config_path(submission_path: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise TypeError("Referenced configuration paths must be non-empty strings.")
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (submission_path.parent.parent / path).resolve()


def _override_output(config: VizConfig, output_root: Path, kind: str) -> VizConfig:
    data = copy.deepcopy(config.data)
    data["paths"].update(
        {
            "output_root": str(output_root),
            "manifest_root": str(output_root / kind),
        }
    )
    return VizConfig(config.path, data)


def _load_components(
    submission_path: Path,
    data: dict[str, Any],
    output_root: Path,
) -> tuple[VizConfig, VizConfig, VizConfig]:
    config_paths = {
        key: _resolve_config_path(submission_path, value)
        for key, value in data["configs"].items()
    }
    heatmap = _override_output(
        load_viz_config(config_paths["heatmap"]), output_root, "heatmap"
    )
    scale_heatmap = _override_output(
        load_viz_config(config_paths["scale_heatmap"]),
        output_root,
        "heatmap-by-scale",
    )
    fit = _override_output(load_viz_config(config_paths["fit"]), output_root, "fit")
    if heatmap.stats_root != fit.stats_root:
        raise ValueError(
            "Endpoint heatmap and fit configurations must use the same stats root."
        )
    if heatmap.correlations_path != fit.correlations_path:
        raise ValueError(
            "Endpoint heatmap and fit configurations must use the same correlation table."
        )
    if scale_heatmap.stats_analysis_id != "clinical-correlation-adjacent-phase":
        raise ValueError(
            "Scale-centered heatmaps must use clinical-correlation-adjacent-phase."
        )
    return heatmap, scale_heatmap, fit


def _significant_rows(correlations: pd.DataFrame) -> pd.DataFrame:
    required = {"CorrelationID", "Status", "p_holm", "p_raw", "rho"}
    missing = sorted(required - set(correlations.columns))
    if missing:
        raise KeyError(f"Correlation table is missing columns: {missing}")
    adjusted_values = pd.to_numeric(correlations["p_holm"], errors="coerce")
    return correlations.loc[
        correlations["Status"].eq("ok")
        & np.isfinite(adjusted_values.to_numpy(dtype=float, na_value=np.nan))
        & adjusted_values.lt(0.05)
    ].copy()


def _publication_summary(
    heatmap: VizConfig, scale_heatmap: VizConfig, fit: VizConfig
) -> dict[str, Any]:
    correlations = pd.read_csv(heatmap.correlations_path)
    significant = _significant_rows(correlations)
    return {
        "correlations": len(correlations),
        "significant_p_holm": len(significant),
        "planned_heatmaps": 216,
        "planned_scale_heatmaps": 30,
        "planned_fits": len(significant),
        "planned_figures": 216 + 30 + len(significant),
        "planned_root_tables": len(ROOT_TABLES),
    }


def _trash_existing(paths: list[Path]) -> None:
    if not paths:
        return
    trash = Path("/usr/bin/trash")
    if not trash.exists():
        raise FileNotFoundError(f"Trash command is unavailable: {trash}")
    subprocess.run([str(trash), *(str(path) for path in paths)], check=True)


def _read_figure_manifest(path: Path, output_root: Path, kind: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Submission {kind} figure manifest is missing: {path}")
    figures = pd.read_csv(path)
    required = {"FigureID", "OutputPath", "Status"}
    missing = sorted(required - set(figures.columns))
    if missing:
        raise KeyError(f"Submission {kind} figure manifest is missing: {missing}")
    if not figures["Status"].eq("ok").all():
        raise ValueError(f"Submission {kind} figure manifest is not complete.")
    output_paths = [Path(value) for value in figures["OutputPath"]]
    for figure_path in output_paths:
        try:
            figure_path.relative_to(output_root)
        except ValueError as error:
            raise ValueError(
                f"Submission {kind} figure is outside the output root: {figure_path}"
            ) from error
        if not figure_path.is_file() or figure_path.stat().st_size == 0:
            raise FileNotFoundError(
                f"Submission {kind} figure is missing or empty: {figure_path}"
            )
    figures.insert(0, "FigureType", kind)
    return figures


def _publish_root_tables(
    heatmap: VizConfig,
    scale_heatmap: VizConfig,
    fit: VizConfig,
    output_root: Path,
    *,
    overwrite: bool,
) -> dict[str, int]:
    correlations = pd.read_csv(heatmap.correlations_path)
    significant = _significant_rows(correlations)
    heatmap_figures = _read_figure_manifest(
        heatmap.manifest_root / heatmap["manifests"]["figures"],
        output_root,
        "heatmap",
    )
    scale_heatmap_figures = _read_figure_manifest(
        scale_heatmap.manifest_root
        / scale_heatmap["manifests"]["figures"],
        output_root,
        "scale-heatmap",
    )
    fit_figures = _read_figure_manifest(
        fit.manifest_root / fit["manifests"]["figures"], output_root, "fit"
    )
    if len(heatmap_figures) != 216:
        raise ValueError(
            "Submission heatmap manifest must contain 216 rows; found "
            f"{len(heatmap_figures)}."
        )
    if len(scale_heatmap_figures) != 30:
        raise ValueError(
            "Submission scale heatmap manifest must contain 30 rows; found "
            f"{len(scale_heatmap_figures)}."
        )
    if len(fit_figures) != len(significant):
        raise ValueError(
            "Submission fit manifest count differs from corrected correlations: "
            f"{len(fit_figures)} != {len(significant)}."
        )
    if set(fit_figures["CorrelationID"]) != set(significant["CorrelationID"]):
        raise ValueError(
            "Submission fit identities differ from the corrected correlation table."
        )
    figures = pd.concat(
        [heatmap_figures, scale_heatmap_figures, fit_figures],
        ignore_index=True,
        sort=False,
    )

    targets = {name: output_root / name for name in ROOT_TABLES}
    existing = [path for path in targets.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Submission root table exists; pass --overwrite to replace it: "
            f"{existing[0]}"
        )
    if existing:
        _trash_existing(existing)
    output_root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(heatmap.correlations_path, targets["correlations.csv"])
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
    heatmap, scale_heatmap, fit = _load_components(
        submission_path, data, output_root
    )

    summary: dict[str, Any] = {
        "stage": args.stage,
        "output_root": str(output_root),
    }
    if args.stage in {"all", "heatmap"}:
        summary["heatmap"] = run_viz(
            heatmap, dry_run=args.dry_run, overwrite=args.overwrite
        )
    if args.stage in {"all", "scale-heatmap"}:
        summary["scale_heatmap"] = run_viz(
            scale_heatmap, dry_run=args.dry_run, overwrite=args.overwrite
        )
    if args.stage in {"all", "fit"}:
        summary["fit"] = run_viz(
            fit, dry_run=args.dry_run, overwrite=args.overwrite
        )
    if args.stage in {"all", "publish"}:
        summary["publish"] = (
            _publication_summary(heatmap, scale_heatmap, fit)
            if args.dry_run
            else _publish_root_tables(
                heatmap,
                scale_heatmap,
                fit,
                output_root,
                overwrite=args.overwrite,
            )
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
