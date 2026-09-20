#!/usr/bin/env python3
"""Run the configured patient-level clinical-correlation analysis."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from lfp_clinical.config import load_clinical_correlation_config
from lfp_clinical.runner import run_clinical_correlations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_clinical_correlation_config(args.config)
    run_clinical_correlations(
        config,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
