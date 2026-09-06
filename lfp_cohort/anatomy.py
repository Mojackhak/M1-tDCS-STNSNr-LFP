"""Channel anatomy, laterality, and connectivity inclusion rules."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from .contracts import FeatureOutputSpec, RecordSpec
from .identity import canonical_string, casefold_identity, identity_equal


def parse_channel_side(channel: str, config: Mapping[str, Any]) -> str:
    """Determine side from both contacts in a bipolar channel label."""
    match = re.fullmatch(r"(\d+)_(\d+)", str(channel))
    if match is None:
        raise ValueError(f"Unsupported bipolar channel label: {channel!r}")
    contacts = [int(match.group(1)), int(match.group(2))]
    side_config = config["anatomy"]["channel_side"]

    def contact_side(contact: int) -> str:
        if side_config["left_min"] <= contact <= side_config["left_max"]:
            return "L"
        if side_config["right_min"] <= contact <= side_config["right_max"]:
            return "R"
        raise ValueError(f"Contact {contact} is outside the configured side ranges.")

    sides = {contact_side(contact) for contact in contacts}
    if len(sides) != 1:
        raise ValueError(f"Bipolar channel {channel!r} crosses the configured midline.")
    return sides.pop()


def classify_region(stn_in: Any, snr_in: Any) -> str:
    """Map STN/SNr flags to exclusive or boundary classes."""
    stn = bool(stn_in)
    snr = bool(snr_in)
    if stn and not snr:
        return "STN"
    if snr and not stn:
        return "SNr"
    if stn and snr:
        return "Mid"
    return "EXT"


def canonicalize_common(
    table: pd.DataFrame,
    record: RecordSpec,
    spec: FeatureOutputSpec,
    source_path: Path,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Add stable cohort identity columns without altering nested values."""
    for column, expected in (
        ("Subject", record.subject),
        ("subject", record.subject),
        ("Record", record.record),
        ("record", record.record),
    ):
        if (
            column in table
            and not all(
                identity_equal(value, expected) for value in table[column].dropna()
            )
        ):
            raise ValueError(
                f"Column {column} disagrees with path identity in {source_path}"
            )
    out = table.copy()
    out = out.drop(
        columns=[
            column
            for column in ("Subject", "subject", "record", "channel")
            if column in out
        ]
    )
    out["ID"] = record.subject
    out["Record"] = record.record
    out["Polar"] = record.polarity
    out["StimSide"] = record.stimulation_side
    out["Domain"] = spec.domain
    out["Metric"] = spec.metric
    out["FeatureOutput"] = spec.feature_output
    out["Representation"] = spec.representation
    out["Reducer"] = spec.reducer
    unsplit_phase = config["metadata"]["phase_for_unsplit"]
    if spec.representation in {"trace", "raw"}:
        if "Phase" in out:
            unexpected = {
                str(value)
                for value in out["Phase"].dropna()
                if not identity_equal(value, unsplit_phase)
            }
            if unexpected:
                raise ValueError(
                    f"Unsplit feature output has phase labels in {source_path}: "
                    f"{sorted(unexpected)}"
                )
        out["Phase"] = unsplit_phase
    else:
        if "Phase" not in out:
            raise KeyError(f"Split feature output has no Phase column: {source_path}")
        phase_order = list(config["validation"]["phase_order"])
        canonical_phases: list[Any] = []
        unexpected: set[str] = set()
        for value in out["Phase"]:
            if pd.isna(value):
                canonical_phases.append(value)
                continue
            canonical = canonical_string(value, phase_order)
            if canonical is None:
                unexpected.add(str(value))
                canonical_phases.append(value)
            else:
                canonical_phases.append(canonical)
        if unexpected:
            raise ValueError(
                f"Split feature output has unsupported phases in {source_path}: "
                f"{sorted(unexpected)}"
            )
        out["Phase"] = canonical_phases
    out["SourcePath"] = str(source_path)
    out["SourceRow"] = [
        f"{record.subject}/{record.record}/{spec.metric}/{spec.feature_output}/{position}"
        for position in range(len(out))
    ]
    return out


def laterality(side: str, stimulation_side: str, config: Mapping[str, Any]) -> str:
    labels = config["anatomy"]["laterality_labels"]
    return (
        labels["ipsilateral"]
        if identity_equal(side, stimulation_side)
        else labels["contralateral"]
    )


def contact_lookup(
    contact_map: pd.DataFrame,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {
        tuple(
            casefold_identity(row[column]) for column in ("ID", "Record", "Channel")
        ): row.to_dict()
        for _, row in contact_map.iterrows()
    }


def _casefold_contact_lookup(
    lookup: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    """Normalize externally supplied contact lookup keys once per merge call."""
    return {
        tuple(casefold_identity(value) for value in key): contact
        for key, contact in lookup.items()
    }


def merge_local_rows(
    table: pd.DataFrame,
    record: RecordSpec,
    spec: FeatureOutputSpec,
    lookup: Mapping[tuple[str, str, str], Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Classify and retain exclusive local STN/SNr rows."""
    if "Channel" not in table:
        raise KeyError(
            f"Local table has no Channel column: {spec.metric}/{spec.feature_output}"
        )
    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    lookup_by_identity = _casefold_contact_lookup(lookup)
    for _, row in table.iterrows():
        channel = str(row["Channel"])
        key = tuple(
            casefold_identity(value) for value in (record.subject, record.record, channel)
        )
        if key not in lookup_by_identity:
            raise KeyError(f"Channel {key} is missing from localization.")
        contact = lookup_by_identity[key]
        if "STN_in" in row and "SNr_in" in row:
            feature_region = classify_region(row["STN_in"], row["SNr_in"])
            if not identity_equal(feature_region, contact["Region"]):
                raise ValueError(f"Feature/localization region mismatch for {key}.")
        item = row.to_dict()
        item["Channel"] = channel
        item["Region"] = contact["Region"]
        item["Side"] = contact["Side"]
        item["Lat"] = laterality(contact["Side"], record.stimulation_side, config)
        for column in (
            "mni_x",
            "mni_y",
            "mni_z",
            "mni_x_flip",
            "mni_y_flip",
            "mni_z_flip",
            "SNr_in",
            "STN_in",
            "space",
            "atlas",
            "anode",
            "cathode",
            "rep_coord",
        ):
            if column in contact:
                item[column] = contact[column]
        if contact["Region"] in config["anatomy"]["local_keep"]:
            rows.append(item)
        else:
            exclusions.append(
                {
                    "ID": record.subject,
                    "Record": record.record,
                    "Domain": spec.domain,
                    "Metric": spec.metric,
                    "FeatureOutput": spec.feature_output,
                    "Channel": channel,
                    "Reason": contact["Region"],
                    "SourceRow": row["SourceRow"],
                }
            )
    return pd.DataFrame.from_records(rows), exclusions


def merge_connectivity_rows(
    table: pd.DataFrame,
    record: RecordSpec,
    spec: FeatureOutputSpec,
    lookup: Mapping[tuple[str, str, str], Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Apply exclusive endpoint and direction rules to connectivity rows."""
    missing = {"channel_a", "channel_b"} - set(table.columns)
    if missing:
        raise KeyError(
            f"Connectivity table is missing endpoint columns: {sorted(missing)}"
        )
    kept: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    lookup_by_identity = _casefold_contact_lookup(lookup)
    for _, row in table.iterrows():
        source_a = str(row["channel_a"])
        source_b = str(row["channel_b"])
        key_a = tuple(
            casefold_identity(value)
            for value in (record.subject, record.record, source_a)
        )
        key_b = tuple(
            casefold_identity(value)
            for value in (record.subject, record.record, source_b)
        )
        if key_a not in lookup_by_identity or key_b not in lookup_by_identity:
            raise KeyError(
                f"Connectivity endpoint is missing from localization: {key_a}, {key_b}"
            )
        contact_a = lookup_by_identity[key_a]
        contact_b = lookup_by_identity[key_b]
        region_a = contact_a["Region"]
        region_b = contact_b["Region"]
        reason = ""
        if region_a in {"Mid", "EXT"} or region_b in {"Mid", "EXT"}:
            reason = f"endpoint_{region_a}_{region_b}"
        elif spec.direction == "undirected" and {region_a, region_b} != {"SNr", "STN"}:
            reason = "not_exclusive_snr_stn"
        elif spec.direction == "directed" and not (
            region_a == "SNr" and region_b == "STN"
        ):
            reason = (
                "reverse_direction"
                if region_a == "STN" and region_b == "SNr"
                else "not_snr_to_stn"
            )
        if not identity_equal(contact_a["Side"], contact_b["Side"]):
            reason = reason or "cross_side_pair"
        if reason:
            exclusions.append(
                {
                    "ID": record.subject,
                    "Record": record.record,
                    "Domain": spec.domain,
                    "Metric": spec.metric,
                    "FeatureOutput": spec.feature_output,
                    "ChannelPair": (source_a, source_b),
                    "Reason": reason,
                    "SourceRow": row["SourceRow"],
                }
            )
            continue
        if spec.direction == "undirected" and region_a == "STN":
            channel_a, channel_b = source_b, source_a
            endpoint_a, endpoint_b = contact_b, contact_a
        else:
            channel_a, channel_b = source_a, source_b
            endpoint_a, endpoint_b = contact_a, contact_b
        item = row.to_dict()
        item.update(
            {
                "channel_a": channel_a,
                "channel_b": channel_b,
                "region_a": endpoint_a["Region"],
                "region_b": endpoint_b["Region"],
                "ChannelPair": (channel_a, channel_b),
                "Channel": (channel_a, channel_b),
                "PairDirection": spec.direction,
                "PairRegion": config["anatomy"][
                    "undirected_pair_keep"
                    if spec.direction == "undirected"
                    else "directed_pair_keep"
                ],
                "PairSide": endpoint_a["Side"],
            }
        )
        item["Lat"] = laterality(endpoint_a["Side"], record.stimulation_side, config)
        item["LatPair"] = item["Lat"]
        item["pair_key_ordered"] = json.dumps([channel_a, channel_b])
        item["pair_key_undirected"] = json.dumps(sorted([channel_a, channel_b]))
        item["pair_key"] = item[
            "pair_key_undirected"
            if spec.direction == "undirected"
            else "pair_key_ordered"
        ]
        for axis in ("x", "y", "z"):
            item[f"mni_{axis}"] = (
                endpoint_a[f"mni_{axis}"],
                endpoint_b[f"mni_{axis}"],
            )
            item[f"mni_{axis}_flip"] = (
                endpoint_a[f"mni_{axis}_flip"],
                endpoint_b[f"mni_{axis}_flip"],
            )
        kept.append(item)
    return pd.DataFrame.from_records(kept), exclusions
