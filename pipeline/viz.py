#!/usr/bin/env python3
"""Render a configured visualization independently of statistical fitting."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from lfp_viz.config import load_viz_config
from lfp_viz.runner import run_viz


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_viz_config(args.config)
    run_viz(config, dry_run=args.dry_run, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
