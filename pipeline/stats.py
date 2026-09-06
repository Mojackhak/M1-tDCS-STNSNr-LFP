#!/usr/bin/env python3
"""Run a configured statistical analysis independently of visualization."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from lfp_stats.config import load_stats_config
from lfp_stats.runner import run_stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_stats_config(args.config)
    run_stats(config, dry_run=args.dry_run, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
