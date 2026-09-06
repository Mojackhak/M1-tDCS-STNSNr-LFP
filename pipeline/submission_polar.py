"""Run and publish one configured submission difference analysis."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from lfp_cohort.io import atomic_csv
from lfp_stats.config import StatsConfig, load_stats_config
from lfp_stats.runner import run_stats
from lfp_viz.config import VizConfig, load_viz_config
from lfp_viz.runner import run_viz


STAGES = ("all", "stats", "scalar", "heatmap")
ROOT_TABLES = (
    "models.csv",
    "model-fit.csv",
    "statistics.csv",
    "coverage.csv",
    "pairing.csv",
    "figures.csv",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _load_submission(path: Path, expected_id: str = "polar") -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise TypeError(f"Submission configuration must contain a mapping: {resolved}")
    if data.get("submission") != {"id": expected_id}:
        raise ValueError(f"submission must equal {{id: {expected_id}}}.")
    configs = data.get("configs")
    if not isinstance(configs, dict) or set(configs) != {
        "stats",
        "scalar",
        "heatmap",
    }:
        raise ValueError("configs must contain stats, scalar, and heatmap paths.")
    paths = data.get("paths")
    if not isinstance(paths, dict) or set(paths) != {"output_root"}:
        raise ValueError("paths must contain output_root only.")
    if not isinstance(paths["output_root"], str) or not paths["output_root"]:
        raise TypeError("paths.output_root must be a non-empty string.")
    publish = data.get("publish")
    if publish != {"root_tables": list(ROOT_TABLES)}:
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


def _override_roots(
    stats: StatsConfig,
    scalar: VizConfig,
    heatmap: VizConfig,
    output_root: Path,
) -> tuple[StatsConfig, VizConfig, VizConfig]:
    stats_data = copy.deepcopy(stats.data)
    scalar_data = copy.deepcopy(scalar.data)
    heatmap_data = copy.deepcopy(heatmap.data)
    work_stats = output_root / "work" / "stats"
    stats_analysis_id = stats.analysis_id
    stats_data["paths"].update(
        {
            "output_root": str(work_stats),
            "derived_root": str(
                work_stats / "derived" / stats_analysis_id
            ),
            "manifest_root": str(
                work_stats / "manifest" / stats_analysis_id
            ),
        }
    )
    stats_manifest = Path(stats_data["paths"]["manifest_root"])
    scalar_data["paths"].update(
        {
            "source_root": stats_data["paths"]["derived_root"],
            "stats_root": str(work_stats),
            "models_manifest": str(stats_manifest / "models.csv"),
            "tests_manifest": str(stats_manifest / "tests.csv"),
            "stats_outputs_manifest": str(stats_manifest / "outputs.csv"),
            "derived_tables_manifest": str(stats_manifest / "derived_tables.csv"),
            "output_root": str(output_root / "scalar"),
            "manifest_root": str(output_root / "scalar"),
        }
    )
    heatmap_data["paths"].update(
        {
            "stats_root": str(work_stats),
            "models_manifest": str(stats_manifest / "models.csv"),
            "stats_outputs_manifest": str(stats_manifest / "outputs.csv"),
            "output_root": str(output_root),
            "manifest_root": str(output_root / "heatmap"),
        }
    )
    return (
        StatsConfig(stats.path, stats_data),
        VizConfig(scalar.path, scalar_data),
        VizConfig(heatmap.path, heatmap_data),
    )


def _override_visualization_outputs(
    scalar: VizConfig,
    heatmap: VizConfig,
    output_root: Path,
) -> tuple[VizConfig, VizConfig]:
    """Redirect dry-run figure plans while retaining existing stats manifests."""

    scalar_data = copy.deepcopy(scalar.data)
    heatmap_data = copy.deepcopy(heatmap.data)
    scalar_data["paths"].update(
        {
            "output_root": str(output_root / "scalar"),
            "manifest_root": str(output_root / "scalar"),
        }
    )
    heatmap_data["paths"].update(
        {
            "output_root": str(output_root),
            "manifest_root": str(output_root / "heatmap"),
        }
    )
    return VizConfig(scalar.path, scalar_data), VizConfig(heatmap.path, heatmap_data)


def _trash(paths: list[Path]) -> None:
    if not paths:
        return
    trash = Path("/usr/bin/trash")
    if not trash.exists():
        raise FileNotFoundError(f"Trash command is unavailable: {trash}")
    subprocess.run([str(trash), *(str(path) for path in paths)], check=True)


def _publish_root_tables(
    scalar_root: Path, output_root: Path, *, overwrite: bool
) -> int:
    sources = {name: scalar_root / name for name in ROOT_TABLES}
    missing = [path for path in sources.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Scalar submission table is missing: {missing[0]}")
    targets = {name: output_root / name for name in ROOT_TABLES}
    existing = [path for path in targets.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Submission root table exists; pass --overwrite to replace it: "
            f"{existing[0]}"
        )
    if existing:
        _trash(existing)
    for name, source in sources.items():
        table = pd.read_csv(source)
        atomic_csv(table, targets[name], overwrite=False)
    return len(targets)


def main(submission_id: str = "polar") -> None:
    args = build_parser().parse_args()
    submission_path = args.config.expanduser().resolve()
    data = _load_submission(submission_path, submission_id)
    config_paths = {
        key: _resolve_config_path(submission_path, value)
        for key, value in data["configs"].items()
    }
    stats = load_stats_config(config_paths["stats"])
    scalar = load_viz_config(config_paths["scalar"])
    heatmap = load_viz_config(config_paths["heatmap"])
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root
        else Path(data["paths"]["output_root"]).expanduser().resolve()
    )
    configured_output = Path(data["paths"]["output_root"]).expanduser().resolve()
    if output_root != configured_output:
        test_stats, test_scalar, test_heatmap = _override_roots(
            stats, scalar, heatmap, output_root
        )
        stats = test_stats
        if args.dry_run:
            scalar, heatmap = _override_visualization_outputs(
                scalar, heatmap, output_root
            )
        else:
            scalar, heatmap = test_scalar, test_heatmap

    summary: dict[str, Any] = {"stage": args.stage, "output_root": str(output_root)}
    if args.stage in {"all", "stats"}:
        summary["stats"] = run_stats(
            stats, dry_run=args.dry_run, overwrite=args.overwrite
        )
    if args.stage in {"all", "scalar"}:
        summary["scalar"] = run_viz(
            scalar, dry_run=args.dry_run, overwrite=args.overwrite
        )
    if args.stage in {"all", "heatmap"}:
        summary["heatmap"] = run_viz(
            heatmap, dry_run=args.dry_run, overwrite=args.overwrite
        )
    if not args.dry_run and args.stage in {"all", "scalar"}:
        summary["published_root_tables"] = _publish_root_tables(
            scalar.output_root, output_root, overwrite=args.overwrite
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
