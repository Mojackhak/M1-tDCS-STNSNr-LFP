"""Create a lightweight current-state inventory for registered stage inputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .artifacts import atomic_csv
from .channels import build_channel_catalog
from .config import StructuralConnectivityConfig


def build_inventory(config: StructuralConnectivityConfig) -> pd.DataFrame:
    """List registered files without traversing connectomes or running models."""

    rows: list[dict[str, object]] = []

    def add(
        *,
        subject_id: str,
        category: str,
        path: Path,
        required_stage: str,
        expected_output: bool = False,
    ) -> None:
        exists = path.is_file() or path.is_dir()
        rows.append(
            {
                "ID": subject_id,
                "Category": category,
                "Path": str(path),
                "RequiredStage": required_stage,
                "ExpectedOutput": expected_output,
                "Exists": exists,
                "Status": (
                    "present"
                    if exists
                    else "missing_expected_output"
                    if expected_output
                    else "missing_input"
                ),
            }
        )

    channels = build_channel_catalog(config)
    for subject in config.subjects:
        subject_id = subject.subject_id
        add(
            subject_id=subject_id,
            category="anchor_t1",
            path=config.anchor_t1(subject_id),
            required_stage="head-model/contact-seeds",
        )
        add(
            subject_id=subject_id,
            category="anchor_t2",
            path=config.anchor_t2(subject_id),
            required_stage="head-model",
        )
        add(
            subject_id=subject_id,
            category="reconstruction",
            path=config.reconstruction(subject_id),
            required_stage="contact-seeds",
        )
        add(
            subject_id=subject_id,
            category="forward_deformation",
            path=config.forward_deformation(subject_id),
            required_stage="contact-seeds/target/field-weighted-lift",
        )
        add(
            subject_id=subject_id,
            category="localization_table",
            path=config.localization_table(subject_id),
            required_stage="contact-seeds",
        )
        add(
            subject_id=subject_id,
            category="individualized_tck",
            path=config.individualized_tck(subject_id),
            required_stage="lift/field-weighted-lift",
        )
        add(
            subject_id=subject_id,
            category="stn_region_mask",
            path=config.region_mask(subject_id, "STN"),
            required_stage="lift",
        )
        add(
            subject_id=subject_id,
            category="snr_region_mask",
            path=config.region_mask(subject_id, "SNr"),
            required_stage="lift",
        )
        add(
            subject_id=subject_id,
            category="mni_tstim",
            path=config.mni_target(subject_id),
            required_stage="lift",
            expected_output=True,
        )
        add(
            subject_id=subject_id,
            category="native_magnitude",
            path=config.native_magnitude(subject_id),
            required_stage="field-weighted-lift",
            expected_output=True,
        )
        add(
            subject_id=subject_id,
            category="native_cortical_ribbon",
            path=config.native_cortical_ribbon(subject_id),
            required_stage="field-weighted-lift",
            expected_output=True,
        )
        add(
            subject_id=subject_id,
            category="mni_cortical_magnitude",
            path=config.mni_cortical_magnitude(subject_id),
            required_stage="field-weighted-lift",
            expected_output=True,
        )
        add(
            subject_id=subject_id,
            category="field_exposure_manifest",
            path=config.field_exposure_manifest(subject_id),
            required_stage="field-weighted-lift",
            expected_output=True,
        )
        subject_channels = channels.loc[channels["ID"].eq(subject_id)]
        for channel in subject_channels["Channel"].astype(str):
            add(
                subject_id=subject_id,
                category=f"mni_contact_seed:{channel}",
                path=config.mni_contact_seed(subject_id, channel),
                required_stage="lift",
                expected_output=True,
            )
        for source in config.connectomes:
            connectome_id = source.connectome_id
            add(
                subject_id=subject_id,
                category=f"binary_membership:{connectome_id}",
                path=config.membership_archive(subject_id, connectome_id),
                required_stage="lift/field-weighted-lift",
                expected_output=True,
            )
            add(
                subject_id=subject_id,
                category=f"field_exposure_archive:{connectome_id}",
                path=config.field_exposure_archive(subject_id, connectome_id),
                required_stage="field-weighted-lift",
                expected_output=True,
            )
            add(
                subject_id=subject_id,
                category=f"field_exposure_archive_manifest:{connectome_id}",
                path=config.field_exposure_archive_manifest(
                    subject_id,
                    connectome_id,
                ),
                required_stage="field-weighted-lift",
                expected_output=True,
            )
    shared = [
        (
            "mni_reference",
            config.mni_reference,
            "contact-seeds/target/field-weighted-lift",
        ),
        (
            "sextant_python_root",
            config.sextant_python_root,
            "lift/field-weighted-lift",
        ),
        ("local_lfp", config.path_value("local_lfp"), "statistics"),
        (
            "binary_channel_lift",
            config.output_path("channel_lift"),
            "lift/field-weighted-lift/statistics",
        ),
    ]
    shared.extend(
        (
            f"connectivity_lfp:{metric}",
            config.path_value("connectivity_lfp_root") / metric / "mean-scalar.pkl",
            "statistics",
        )
        for metric in config["outcomes"]["connectivity_metrics"]
    )
    shared.extend(
        (
            f"connectome:{source.connectome_id}",
            source.path,
            "lift/field-weighted-lift",
        )
        for source in config.connectomes
        if source.path is not None
    )
    for category, path, stage in shared:
        add(
            subject_id="shared",
            category=category,
            path=Path(path),
            required_stage=stage,
        )
    add(
        subject_id="shared",
        category="field_weighted_channel_lift",
        path=config.output_path("channel_field_weighted_lift"),
        required_stage="field-weighted-lift",
        expected_output=True,
    )
    return (
        pd.DataFrame(rows)
        .sort_values(["ID", "Category"], kind="mergesort")
        .reset_index(drop=True)
    )


def run_inventory(
    config: StructuralConnectivityConfig,
    *,
    dry_run: bool,
    overwrite: bool,
) -> pd.DataFrame:
    """Build and optionally persist the current inventory."""

    inventory = build_inventory(config)
    if not dry_run:
        atomic_csv(
            inventory,
            config.output_path("inventory"),
            overwrite=overwrite,
        )
    return inventory
