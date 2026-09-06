#!/usr/bin/env python3
"""Run configured cohort preprocessing for LFP-TensorPipe feature tables."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from lfp_cohort.config import load_config
from lfp_cohort.contracts import STAGES
from lfp_cohort.runner import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--record", action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    run_pipeline(
        config,
        stage=args.stage,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        record_selectors=args.record,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
