"""Construct dose-contact-matched bipolar and equal-connection predictors."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lfp_cohort.io import read_stage_table
from lfp_viz.data import path_below, require_columns

from .data import (
    ADJACENT_IDENTITY_COLUMNS,
    MANIFEST_COLUMNS,
    _normalize_ids,
    _numeric_column,
    _predictor_registry,
    build_feature_specifications,
    derive_adjacent_phase_values,
)


def build_contact_matches(
    programming: pd.DataFrame, pairs: pd.DataFrame
) -> pd.DataFrame:
    """Retain exact-contact candidates, including explicit reasons for exclusion."""
    pairs = pairs.rename(columns={"ContactPair": "Channel", "LeadSide": "Side"}).copy()
    tokens = pairs.Channel.str.extract(r"^(\d+)_(\d+)$")
    if tokens.isna().any().any():
        raise ValueError("ContactPair must contain two integer contact tokens.")
    numbers = tokens.astype(int)
    if (
        not numbers[1].sub(numbers[0]).eq(1).all()
        or not pairs.Side.isin(["L", "R"]).all()
    ):
        raise ValueError(
            "Contact-pair anatomy requires adjacent contacts and physical L/R sides."
        )
    pairs["Contact"] = numbers.to_numpy().tolist()
    pairs = pairs[["ID", "Side", "Channel", "RegionClass", "Contact"]].explode(
        "Contact"
    )
    keys = ["EndpointID", "ID", "Target", "Side", "Contact", "PhysicalContact"]
    matches = programming[keys].merge(pairs, on=["ID", "Side", "Contact"], how="left")
    matches["Eligible"] = matches.Channel.notna() & matches.RegionClass.eq(
        matches.Target
    )
    matches["MatchStatus"] = np.select(
        [matches.Channel.isna(), matches.Eligible],
        ["no_recording_pair", "matched"],
        default="different_region",
    )
    return matches.sort_values(["EndpointID", "ID", "Side", "Channel"]).reset_index(
        drop=True
    )


def derive_member_deltas(rows: pd.DataFrame, derivation: dict) -> pd.DataFrame:
    """Reuse adjacent subtraction separately for each original recording member."""
    identity = ["EndpointID", "MemberID", *ADJACENT_IDENTITY_COLUMNS]
    phase_key = [*identity, "Phase"]
    provenance = [
        "Channel",
        "Region",
        "Side",
        "StimSide",
        "Record",
        "Trial",
        "space",
        "atlas",
        "SourcePath",
        "PairRegion",
        "PairDirection",
        "Target",
        "Contact",
        "PhysicalContact",
        "ContactRegionSourcePath",
        "AggregationUnit",
    ]
    if rows.Domain.eq("connectivity").any():
        provenance += [
            "ConnectionID",
            "ChannelPair",
            "channel_a",
            "channel_b",
            "EndpointRole",
        ]
    # Split copies are one measurement. Conflicting phase values or source
    # provenance are parsing errors, not extra observations to average.
    counts = rows.groupby(phase_key, dropna=False, observed=True)[
        ["Value", "Record", "Trial", "StimSide", "space", "atlas", "Side"]
    ].nunique(dropna=False)
    if counts.gt(1).any().any():
        raise ValueError(
            "Conflicting values or recording provenance for a matched member/phase."
        )
    rows = rows.drop_duplicates(phase_key)
    member_metadata = rows[[*identity, *provenance]].drop_duplicates()
    frames = []
    for (endpoint, member), group in rows.groupby(
        ["EndpointID", "MemberID"], sort=True
    ):
        delta = derive_adjacent_phase_values(group, derivation)
        delta["EndpointID"], delta["MemberID"] = endpoint, member
        frames.append(delta)
    if not frames:
        return pd.DataFrame(
            columns=[
                *identity,
                "Phase",
                "FromPhase",
                "ToPhase",
                "PhaseContrast",
                "PredictorDefinition",
                "PredictorID",
                "FromValue",
                "ToValue",
                "Value",
                *provenance,
                "UsedInAverage",
            ]
        )
    result = pd.concat(frames, ignore_index=True).merge(
        member_metadata, on=identity, how="left", validate="many_to_one"
    )
    result["UsedInAverage"] = np.isfinite(result.Value)
    return result.sort_values(
        ["EndpointID", "PredictorID", "ID", "MemberID"]
    ).reset_index(drop=True)


def average_member_deltas(
    members: pd.DataFrame, metadata: pd.DataFrame
) -> pd.DataFrame:
    """Average unique finite member changes directly, never endpoint means."""
    fields = [
        *metadata.columns,
        "ID",
        "EndpointID",
        "Side",
        "StimSide",
        "Record",
        "Trial",
        "space",
        "atlas",
        "SourcePath",
        "PairRegion",
        "PairDirection",
        "ContactRegionSourcePath",
        "AggregationUnit",
    ]
    rows = []
    for _, group in members.groupby(["EndpointID", "PredictorID", "ID"], sort=True):
        valid = group.loc[group.UsedInAverage]
        row = group.iloc[0][fields].to_dict()
        row.update(
            {
                "FromValue": valid.FromValue.mean(),
                "ToValue": valid.ToValue.mean(),
                "Value": valid.Value.mean(),
                "NAvailableMembers": len(group),
                "NUsedMembers": len(valid),
                "AvailableMembers": json.dumps(sorted(group.MemberID.tolist())),
                "UsedMembers": json.dumps(sorted(valid.MemberID.tolist())),
                "UsedChannels": ";".join(sorted(set(valid.Channel))),
            }
        )
        rows.append(row)
    return pd.DataFrame(
        rows,
        columns=[
            *fields,
            "FromValue",
            "ToValue",
            "Value",
            "NAvailableMembers",
            "NUsedMembers",
            "AvailableMembers",
            "UsedMembers",
            "UsedChannels",
        ],
    )


def load_contact_predictors(config, programming):
    """Read normalized contact scalars and split edges without region aggregation."""
    from .programming import _read_source

    pairs, anatomy_manifest = _read_source(
        config.paths["contact_pair_regions"],
        {"ID", "ContactPair", "LeadSide", "RegionClass"},
        ["ID", "LeadSide", "ContactPair"],
        "contact_pair_regions",
    )
    matches = build_contact_matches(programming, pairs)
    matches["ContactRegionSourcePath"] = str(config.paths["contact_pair_regions"])
    eligible = matches.loc[matches.Eligible].drop(
        columns=["RegionClass", "Eligible", "MatchStatus"]
    )
    anatomy = pairs.rename(columns={"ContactPair": "Channel", "LeadSide": "Side"})[
        ["ID", "Channel", "Side", "RegionClass"]
    ]
    manifest = pd.read_csv(config.lfp.feature_manifest)
    require_columns(manifest, MANIFEST_COLUMNS, str(config.lfp.feature_manifest))
    source_mask = (
        manifest.Domain.eq("local") & manifest.Stage.eq("aggregate_contact")
    ) | (manifest.Domain.eq("connectivity") & manifest.Stage.eq("split"))
    candidates = manifest.loc[
        source_mask
        & manifest.SourceStage.eq("normalize")
        & manifest.AggregationLevel.eq("contact")
        & manifest.Representation.eq("scalar")
    ]
    specifications = build_feature_specifications(config.lfp)
    metadata = _predictor_registry(specifications, config.lfp)
    source_keys = ["Domain", "Metric", "FeatureOutput"]
    sources = specifications[source_keys].drop_duplicates()
    selected = candidates.merge(sources, on=source_keys, validate="one_to_one")
    if len(selected) != len(sources):
        raise ValueError(
            "Contact strategy requires every registered local/split scalar source."
        )
    input_manifest = [
        anatomy_manifest,
        {
            "InputType": "feature_manifest",
            "Path": str(config.lfp.feature_manifest),
            "Rows": len(manifest),
            "Columns": len(manifest.columns),
        },
    ]
    levels = config.lfp["inputs"]["lfp"]
    all_members = []
    for source in selected.sort_values(source_keys).itertuples(index=False):
        path = path_below(
            Path(source.Path),
            config.paths[source.Domain + "_root"],
            "Contact LFP source",
        )
        table = read_stage_table(path)
        required = {
            *source_keys,
            "ID",
            "Channel",
            "Region",
            "Polar",
            "Phase",
            "Lat",
            "Band",
            "Value",
            "IncludeAggregate",
            "StimSide",
            "Record",
            "Trial",
            "space",
            "atlas",
            "Side" if source.Domain == "local" else "PairSide",
        }
        if source.Domain == "connectivity":
            required.update(
                {
                    "PairRegion",
                    "PairDirection",
                    "ChannelPair",
                    "channel_a",
                    "channel_b",
                    "EndpointRole",
                }
            )
        require_columns(table, required, str(path))
        if len(table) != int(source.Rows):
            raise ValueError(f"Contact scalar row count differs from manifest: {path}")
        if not set(table.IncludeAggregate.dropna().unique()).issubset({True, False}):
            raise TypeError(f"IncludeAggregate must be boolean: {path}")
        table = table.copy()
        table["Value"] = _numeric_column(table, "Value", str(path))
        rows = table.loc[
            table.IncludeAggregate.fillna(False).astype(bool)
            & table.Polar.isin(levels["polarity"])
            & table.Lat.isin(levels["laterality"])
            & table.Phase.isin(levels["phase_derivation"]["source_phases"])
        ].copy()
        rows["ID"] = _normalize_ids(rows.ID)
        if source.Domain == "connectivity":
            rows["Side"] = rows.PairSide
            directed = source.Metric in {"psi", "trgc"}
            expected_region = "SNr→STN" if directed else "SNr-STN"
            if (
                not rows.PairDirection.eq(
                    "directed" if directed else "undirected"
                ).all()
                or not rows.PairRegion.eq(expected_region).all()
            ):
                raise ValueError(f"Unexpected connectivity region/direction: {path}")
            rows["ConnectionID"] = [
                json.dumps([a, b] if directed else sorted([a, b]))
                for a, b in zip(rows.channel_a, rows.channel_b)
            ]
            rows["MemberID"], rows["RegionValue"] = rows.ConnectionID, rows.PairRegion
        else:
            rows["MemberID"], rows["RegionValue"] = rows.Channel, rows.Region
            rows["PairRegion"], rows["PairDirection"] = pd.NA, "not_applicable"
        expected_side = np.where(
            rows.Lat.eq("Ipsi"), rows.StimSide, rows.StimSide.map({"L": "R", "R": "L"})
        )
        if not rows.Side.eq(expected_side).all():
            raise ValueError(f"Physical side and tDCS laterality disagree: {path}")
        rows = rows.merge(
            anatomy, on=["ID", "Side", "Channel"], how="left", validate="many_to_one"
        )
        if not rows.Region.eq(rows.RegionClass).all():
            raise ValueError(
                f"LFP channel region disagrees with contact-pair anatomy: {path}"
            )
        rows = rows.merge(
            specifications,
            on=[*source_keys, "Band", "RegionValue"],
            validate="many_to_one",
        )
        rows = rows.merge(
            eligible, on=["ID", "Side", "Channel"], how="inner", validate="many_to_many"
        )
        rows["SourcePath"] = str(path)
        rows["AggregationUnit"] = (
            "connection" if source.Domain == "connectivity" else "channel"
        )
        all_members.append(derive_member_deltas(rows, levels["phase_derivation"]))
        input_manifest.append(
            {
                "InputType": "lfp_table",
                "Domain": source.Domain,
                "Metric": source.Metric,
                "FeatureOutput": source.FeatureOutput,
                "Stage": source.Stage,
                "Path": str(path),
                "Rows": len(table),
                "Columns": len(table.columns),
            }
        )
    members = pd.concat(all_members, ignore_index=True)
    predictors = average_member_deltas(members, metadata)
    audits = {
        "contact_matching": matches,
        "local_channel_deltas": members.loc[members.Domain.eq("local")].reset_index(
            drop=True
        ),
        "connectivity_edge_deltas": members.loc[
            members.Domain.eq("connectivity")
        ].reset_index(drop=True),
    }
    return metadata, predictors, input_manifest, audits
