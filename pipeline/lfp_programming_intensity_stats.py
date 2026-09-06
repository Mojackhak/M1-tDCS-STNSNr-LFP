#!/usr/bin/env python3
"""Run signed adjacent-phase LFP versus programmed-intensity associations."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from lfp_clinical.programming import load_programming_config
from lfp_clinical.programming_runner import run_programming_intensity


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    run_programming_intensity(
        load_programming_config(args.config),
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
