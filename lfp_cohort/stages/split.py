"""Connectivity contact-pair endpoint splitting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from ..feature_outputs import iter_feature_output_specs
from ..contracts import FeatureOutputSpec, RunState, SOURCE_STAGES
from ..io import aggregate_path, atomic_pickle, read_stage_table, split_path
from ..provenance import record_feature_output


def split_connectivity_table(
    table: pd.DataFrame, spec: FeatureOutputSpec, config: Mapping[str, Any]
) -> pd.DataFrame:
    """Split each accepted pair into two configured endpoint rows."""
    policy = config["connectivity_split"]
    roles = (
        policy["undirected_roles"]
        if spec.direction == "undirected"
        else policy["directed_roles"]
    )
    rows: list[dict[str, Any]] = []
    for _, row in table.iterrows():
        for endpoint_index in (0, 1):
            item = row.to_dict()
            item[policy["channel_pair_backup"]] = row["ChannelPair"]
            item[policy["region_backup"]] = row["PairRegion"]
            item["Channel"] = row["channel_a" if endpoint_index == 0 else "channel_b"]
            item["Region"] = row["region_a" if endpoint_index == 0 else "region_b"]
            item["EndpointRole"] = roles[endpoint_index]
            for axis in ("x", "y", "z"):
                item[f"mni_{axis}"] = row[f"mni_{axis}"][endpoint_index]
                item[f"mni_{axis}_flip"] = row[f"mni_{axis}_flip"][endpoint_index]
            rows.append(item)
    output = pd.DataFrame.from_records(rows)
    output.attrs = dict(table.attrs)
    output.attrs["connectivity_split"] = True
    return output


def run_split(state: RunState, source_stages: Sequence[str] = SOURCE_STAGES) -> None:
    config = state.config
    protocol = int(config["execution"]["pickle_protocol"])
    for source_stage in source_stages:
        for spec in iter_feature_output_specs(config):
            if spec.domain != "connectivity":
                continue
            source = read_stage_table(
                aggregate_path(state.output_root, source_stage, "contact", spec, config)
            )
            split = split_connectivity_table(source, spec, config)
            output = split_path(state.output_root, source_stage, spec, config)
            atomic_pickle(split, output, state.overwrite, protocol)
            record_feature_output(state, "split", spec, output, split, source_stage)
