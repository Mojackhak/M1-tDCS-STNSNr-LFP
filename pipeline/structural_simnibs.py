#!/usr/bin/env python3
"""Run CHARM, HD-tDCS field, or Tstim stages for registered subjects."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path

from lfp_structural.config import load_config
from lfp_structural.simnibs_model import (
    build_tstim_target,
    run_field_model,
    run_head_model,
)


STAGES = ("head-model", "field", "target")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--subject", action="append", required=True)
    parser.add_argument("--cpus", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cpus < 1:
        raise ValueError("--cpus must be a positive integer.")
    config = load_config(args.config)
    registered = {subject.subject_id for subject in config.subjects}
    unknown = sorted(set(args.subject).difference(registered))
    if unknown:
        raise ValueError(f"Unknown registered subject: {unknown[0]}")
    results = []
    for subject_id in args.subject:
        if args.stage == "head-model":
            result = run_head_model(
                config,
                subject_id,
                dry_run=args.dry_run,
                overwrite=args.overwrite,
            )
        elif args.stage == "field":
            result = run_field_model(
                config,
                subject_id,
                dry_run=args.dry_run,
                overwrite=args.overwrite,
                cpus=args.cpus,
            )
        else:
            result = build_tstim_target(
                config,
                subject_id,
                dry_run=args.dry_run,
                overwrite=args.overwrite,
            )
        results.append(result)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

