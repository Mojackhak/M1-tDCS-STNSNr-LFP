"""Equal-weight contact, region, and subject aggregation."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..feature_outputs import iter_feature_output_specs
from ..contracts import FeatureOutputSpec, RunState, SOURCE_STAGES
from ..identity import (
    first_group_labels,
    groupby_casefolded,
    unique_casefolded_strings,
)
from ..io import aggregate_path, atomic_pickle, read_stage_table, stage_path
from ..provenance import record_feature_output
from ..qc import finite_count


def strict_mean_cells(values: Sequence[Any]) -> Any:
    """Compute an equal-weight NaN-aware mean with strict nested axes."""
    nested = [value for value in values if isinstance(value, (pd.Series, pd.DataFrame))]
    if not nested:
        numeric = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(
            dtype=float
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return float(np.nanmean(numeric)) if np.isfinite(numeric).any() else np.nan
    template = nested[0]
    arrays: list[np.ndarray] = []
    for value in values:
        if isinstance(template, pd.Series):
            if isinstance(value, pd.Series):
                if not value.index.equals(template.index):
                    raise ValueError("Series axes differ during strict aggregation.")
                arrays.append(value.to_numpy(dtype=float, copy=False))
            elif finite_count(value) == 0:
                arrays.append(np.full(template.size, np.nan))
            else:
                raise TypeError("Mixed scalar and Series values during aggregation.")
        elif isinstance(value, pd.DataFrame):
            if not value.index.equals(template.index) or not value.columns.equals(
                template.columns
            ):
                raise ValueError("DataFrame axes differ during strict aggregation.")
            arrays.append(value.to_numpy(dtype=float, copy=False))
        elif finite_count(value) == 0:
            arrays.append(np.full(template.shape, np.nan))
        else:
            raise TypeError("Mixed scalar and DataFrame values during aggregation.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(np.stack(arrays, axis=0), axis=0)
    if isinstance(template, pd.Series):
        return pd.Series(mean, index=template.index, name=template.name)
    return pd.DataFrame(mean, index=template.index, columns=template.columns)


def finite_contributor_counts(values: Sequence[Any], aggregate: Any) -> Any:
    """Count finite direct contributors with the aggregate value's axes."""
    if isinstance(aggregate, pd.Series):
        counts = np.zeros(aggregate.size, dtype=int)
        for value in values:
            if isinstance(value, pd.Series):
                counts += np.isfinite(value.to_numpy(dtype=float, copy=False))
        return pd.Series(counts, index=aggregate.index, name=aggregate.name)
    if isinstance(aggregate, pd.DataFrame):
        counts = np.zeros(aggregate.shape, dtype=int)
        for value in values:
            if isinstance(value, pd.DataFrame):
                counts += np.isfinite(value.to_numpy(dtype=float, copy=False))
        return pd.DataFrame(counts, index=aggregate.index, columns=aggregate.columns)
    return int(sum(finite_count(value) > 0 for value in values))


def _present(table: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    return [column for column in columns if column in table]


def aggregation_group_columns(
    table: pd.DataFrame, spec: FeatureOutputSpec, level: str
) -> list[str]:
    common = _present(
        table,
        [
            "ID",
            "Record",
            "Trial",
            "Polar",
            "StimSide",
            "Domain",
            "Metric",
            "FeatureOutput",
            "Representation",
            "Reducer",
            "Phase",
            "Band",
            "space",
            "atlas",
        ],
    )
    if spec.domain == "local":
        region = ["Region"] + ([] if level == "subject" else ["Lat"])
        if level == "contact":
            region += [
                "Channel",
                "Side",
                "mni_x",
                "mni_y",
                "mni_z",
                "mni_x_flip",
                "mni_y_flip",
                "mni_z_flip",
            ]
    else:
        region = ["PairRegion", "PairDirection"] + (
            [] if level == "subject" else ["Lat", "LatPair"]
        )
        if level == "contact":
            region += [
                "ChannelPair",
                "PairSide",
                "channel_a",
                "channel_b",
                "region_a",
                "region_b",
                "pair_key",
                "pair_key_ordered",
                "pair_key_undirected",
                "mni_x",
                "mni_y",
                "mni_z",
                "mni_x_flip",
                "mni_y_flip",
                "mni_z_flip",
            ]
    return common + _present(table, region)


def aggregate_table(
    table: pd.DataFrame,
    spec: FeatureOutputSpec,
    level: str,
    member_separator: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Aggregate one table to the requested configured level."""
    included = table.loc[table["IncludeAggregate"].astype(bool)].copy()
    if included.empty:
        raise ValueError(
            "No rows with IncludeAggregate=true remain for "
            f"{spec.domain}/{spec.metric}/{spec.feature_output} at {level} level."
        )
    group_columns = aggregation_group_columns(included, spec, level)
    rows: list[dict[str, Any]] = []
    qc: list[dict[str, Any]] = []
    for _, group in groupby_casefolded(included, group_columns):
        item = first_group_labels(group, group_columns)
        item.update(
            {
                "Domain": spec.domain,
                "Metric": spec.metric,
                "FeatureOutput": spec.feature_output,
                "Representation": spec.representation,
                "Reducer": spec.reducer,
            }
        )
        values = group["Value"].tolist()
        finite_rows = group["Value"].map(finite_count).gt(0)
        item["Value"] = strict_mean_cells(values)
        item["ValueN"] = finite_contributor_counts(values, item["Value"])
        item["n_rows"] = int(len(group))
        item["n_rows_finite"] = int(finite_rows.sum())
        item["IsOutlier"] = False
        item["IncludeAggregate"] = True
        if "NormalizationStatus" in group:
            statuses = unique_casefolded_strings(group["NormalizationStatus"])
            item["NormalizationStatus"] = statuses[0] if len(statuses) == 1 else "mixed"
        if level == "contact":
            members = (
                unique_casefolded_strings(group["Epoch"])
                if "Epoch" in group
                else []
            )
            finite_members = (
                unique_casefolded_strings(group.loc[finite_rows, "Epoch"])
                if "Epoch" in group
                else []
            )
            item["n_epochs"] = len(members)
            item["n_epochs_finite"] = len(finite_members)
            item["epoch_members"] = member_separator.join(members)
        elif level == "region":
            member_column = "Channel" if spec.domain == "local" else "ChannelPair"
            members = unique_casefolded_strings(group[member_column])
            finite_members = unique_casefolded_strings(
                group.loc[finite_rows, member_column]
            )
            item["n_contacts"] = len(members)
            item["n_contacts_finite"] = len(finite_members)
            item["contact_members"] = member_separator.join(members)
            if spec.domain == "local":
                item["n_channels"] = len(members)
                item["n_channels_finite"] = len(finite_members)
                item["channel_members"] = member_separator.join(members)
            else:
                item["n_pairs"] = len(members)
                item["n_pairs_finite"] = len(finite_members)
                item["pair_members"] = member_separator.join(members)
        else:
            members = unique_casefolded_strings(group["Lat"])
            finite_members = unique_casefolded_strings(group.loc[finite_rows, "Lat"])
            item["n_lat"] = len(members)
            item["n_lat_finite"] = len(finite_members)
            item["lat_members"] = member_separator.join(members)
        rows.append(item)
        qc.append(
            {
                "Level": level,
                "Domain": spec.domain,
                "Metric": spec.metric,
                "FeatureOutput": spec.feature_output,
                **{column: item[column] for column in group_columns},
                **{
                    key: value
                    for key, value in item.items()
                    if key.startswith("n_") or key.endswith("_members")
                },
            }
        )
    output = pd.DataFrame.from_records(rows)
    output.attrs = dict(table.attrs)
    output.attrs.update(
        {
            "aggregation_level": level,
            "aggregation_mode": "mean",
            "aggregation_align": "strict",
        }
    )
    return output, qc


def run_aggregate(
    state: RunState, source_stages: Sequence[str] = SOURCE_STAGES
) -> None:
    config = state.config
    protocol = int(config["execution"]["pickle_protocol"])
    separator = config["aggregation"]["member_separator"]
    for source_stage in source_stages:
        for spec in iter_feature_output_specs(config):
            source = read_stage_table(
                stage_path(state.output_root, source_stage, spec, config)
            )
            current = source
            for level in ("contact", "region", "subject"):
                aggregated, qc = aggregate_table(current, spec, level, separator)
                output = aggregate_path(
                    state.output_root, source_stage, level, spec, config
                )
                atomic_pickle(aggregated, output, state.overwrite, protocol)
                record_feature_output(
                    state,
                    f"aggregate_{level}",
                    spec,
                    output,
                    aggregated,
                    source_stage,
                )
                state.add_qc(
                    "aggregation_members",
                    [{"SourceStage": source_stage, **row} for row in qc],
                )
                current = aggregated
