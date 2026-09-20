#!/usr/bin/env python3
"""Run registered structural-connectivity computation stages."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path

from lfp_structural.config import load_config
from lfp_structural.contact_lmm import (
    run_contact_lmm,
    run_contact_lmm_figures,
    run_contact_lmm_slope_adjustment,
)
from lfp_structural.contact_lmm_adjacent import (
    run_contact_lmm_adjacent,
    run_contact_lmm_adjacent_field_excess_atlas_grid_fits,
    run_contact_lmm_adjacent_field_excess_finder_fits,
    run_contact_lmm_adjacent_field_excess_finder_figures,
    run_contact_lmm_adjacent_field_excess_heatmaps,
    run_contact_lmm_adjacent_field_excess_local,
    run_contact_lmm_adjacent_id_random_intercept,
)
from lfp_structural.cortical_allocation_lmm import run_cortical_allocation_lmm
from lfp_structural.cortical_allocation_lmm_adjacent import (
    run_cortical_allocation_lmm_adjacent,
)
from lfp_structural.fiber import (
    run_field_weighted_lift_extraction,
    run_field_weighted_lift_reaggregation,
    run_lift_extraction,
)
from lfp_structural.geometry import generate_contact_seeds
from lfp_structural.inventory import run_inventory
from lfp_structural.result_figures import (
    run_cortical_allocation_adjacent_figures,
    run_cortical_allocation_figures,
    run_spearman_figures,
)
from lfp_structural.statistics import run_statistics


STAGES = (
    "inventory",
    "contact-seeds",
    "lift",
    "field-weighted-lift",
    "field-weighted-reaggregate",
    "statistics",
    "contact-lmm",
    "contact-lmm-adjacent",
    "contact-lmm-adjacent-field-excess-atlas-grid-fits",
    "contact-lmm-adjacent-field-excess-finder-fits",
    "contact-lmm-adjacent-field-excess-finder-figures",
    "contact-lmm-adjacent-field-excess-local",
    "contact-lmm-adjacent-field-excess-heatmaps",
    "contact-lmm-adjacent-id-random-intercept",
    "contact-lmm-slope-adjustment",
    "contact-lmm-figures",
    "cortical-allocation-lmm",
    "cortical-allocation-lmm-adjacent",
    "cortical-allocation-adjacent-figures",
    "spearman-figures",
    "cortical-allocation-figures",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.stage == "inventory":
        table = run_inventory(
            config,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
        summary = {
            "rows": int(len(table)),
            "present": int(table["Exists"].sum()),
            "missing_inputs": int(table["Status"].eq("missing_input").sum()),
            "missing_expected_outputs": int(
                table["Status"].eq("missing_expected_output").sum()
            ),
            "writes": 0 if args.dry_run else 1,
        }
    elif args.stage == "contact-seeds":
        table = generate_contact_seeds(
            config,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
        summary = {
            "rows": int(len(table)),
            "subjects": int(table["ID"].nunique()),
            "native_voxels": int(table["NativeVoxelCount"].sum()),
            "mni_voxels": (
                int(table["MniVoxelCount"].sum())
                if table["MniVoxelCount"].notna().all()
                else None
            ),
            "writes": 0 if args.dry_run else int(len(table) * 2 + 1),
        }
    elif args.stage == "lift":
        table = run_lift_extraction(
            config,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
        summary = {
            "rows": int(len(table)),
            "status_counts": table.iloc[:, -1].value_counts().sort_index().to_dict()
            if args.dry_run
            else table["LiftStatus"].value_counts().sort_index().to_dict(),
            "writes": (
                0
                if args.dry_run
                else int(len(config.subjects) * len(config.connectomes) * 2 + 1)
            ),
        }
    elif args.stage == "field-weighted-lift":
        table = run_field_weighted_lift_extraction(
            config,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
        summary = {
            "rows": int(len(table)),
            "status_counts": table.iloc[:, -1].value_counts().sort_index().to_dict()
            if args.dry_run
            else table["FieldWeightedLiftStatus"].value_counts().sort_index().to_dict(),
            "writes": (
                0
                if args.dry_run
                else int(len(config.subjects) * (len(config.connectomes) * 2 + 2) + 1)
            ),
        }
    elif args.stage == "field-weighted-reaggregate":
        table = run_field_weighted_lift_reaggregation(
            config,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
        summary = {
            "rows": int(len(table)),
            "status_counts": (
                table["Status"].value_counts().sort_index().to_dict()
                if args.dry_run
                else table["FieldWeightedLiftStatus"]
                .value_counts()
                .sort_index()
                .to_dict()
            ),
            "writes": 0 if args.dry_run else 1,
        }
    elif args.stage == "statistics":
        summary = run_statistics(
            config,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
    elif args.stage == "contact-lmm":
        summary = run_contact_lmm(
            config,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
    elif args.stage == "contact-lmm-adjacent":
        summary = run_contact_lmm_adjacent(
            config,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
    elif args.stage == "contact-lmm-adjacent-field-excess-finder-figures":
        summary = run_contact_lmm_adjacent_field_excess_finder_figures(
            config,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
    elif args.stage == "contact-lmm-adjacent-field-excess-atlas-grid-fits":
        summary = run_contact_lmm_adjacent_field_excess_atlas_grid_fits(
            config,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
    elif args.stage == "contact-lmm-adjacent-field-excess-finder-fits":
        summary = run_contact_lmm_adjacent_field_excess_finder_fits(
            config,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
    elif args.stage == "contact-lmm-adjacent-field-excess-local":
        summary = run_contact_lmm_adjacent_field_excess_local(
            config,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
    elif args.stage == "contact-lmm-adjacent-field-excess-heatmaps":
        summary = run_contact_lmm_adjacent_field_excess_heatmaps(
            config,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
    elif args.stage == "contact-lmm-adjacent-id-random-intercept":
        summary = run_contact_lmm_adjacent_id_random_intercept(
            config,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
    elif args.stage == "contact-lmm-slope-adjustment":
        summary = run_contact_lmm_slope_adjustment(
            config,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
    elif args.stage == "contact-lmm-figures":
        summary = run_contact_lmm_figures(
            config,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
    elif args.stage == "cortical-allocation-lmm":
        summary = run_cortical_allocation_lmm(
            config,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
    elif args.stage == "cortical-allocation-lmm-adjacent":
        summary = run_cortical_allocation_lmm_adjacent(
            config,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
    elif args.stage == "cortical-allocation-adjacent-figures":
        summary = run_cortical_allocation_adjacent_figures(
            config,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
    elif args.stage == "spearman-figures":
        summary = run_spearman_figures(
            config,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
    else:
        summary = run_cortical_allocation_figures(
            config,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
