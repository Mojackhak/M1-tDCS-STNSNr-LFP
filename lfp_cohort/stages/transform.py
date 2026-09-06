"""Configured element-wise feature transforms."""

from __future__ import annotations

import pandas as pd
from lfptensorpipe.stats.preproc.transform import transform_df

from ..feature_outputs import configured_transform_mode, iter_feature_output_specs
from ..contracts import RunState
from ..io import atomic_pickle, read_stage_table, stage_path
from ..provenance import record_feature_output
from ..qc import finite_count, nan_like


def transform_feature_table(source: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Transform values while retaining empty nested cell types and axes."""
    transformed = transform_df(source, value_col="Value", mode=mode, drop_empty=False)
    empty_nested = [
        (position, value)
        for position, value in enumerate(source["Value"])
        if isinstance(value, (pd.Series, pd.DataFrame)) and finite_count(value) == 0
    ]
    if empty_nested:
        transformed["Value"] = transformed["Value"].astype(object)
        value_column = transformed.columns.get_loc("Value")
        for position, value in empty_nested:
            transformed.iat[position, value_column] = nan_like(value)
    return transformed


def run_transform(state: RunState) -> None:
    config = state.config
    protocol = int(config["execution"]["pickle_protocol"])
    for spec in iter_feature_output_specs(config):
        source_path = stage_path(state.output_root, "merge", spec, config)
        source = read_stage_table(source_path)
        mode = configured_transform_mode(spec, config)
        before_finite = sum(finite_count(value) for value in source["Value"])
        transformed = transform_feature_table(source, mode)
        after_finite = sum(finite_count(value) for value in transformed["Value"])
        transformed.attrs = dict(source.attrs)
        transformed.attrs["applied_transform_mode"] = mode
        output = stage_path(state.output_root, "transform", spec, config)
        atomic_pickle(transformed, output, state.overwrite, protocol)
        record_feature_output(state, "transform", spec, output, transformed, "merge")
        state.add_qc(
            "transform_invalid",
            [
                {
                    "Domain": spec.domain,
                    "Metric": spec.metric,
                    "FeatureOutput": spec.feature_output,
                    "Mode": mode,
                    "n_finite_before": before_finite,
                    "n_finite_after": after_finite,
                    "n_new_invalid": before_finite - after_finite,
                }
            ],
        )
