"""Build the registered ipsilateral channel catalog from normalized LFP tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import StructuralConnectivityConfig


def _read_frame(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Normalized LFP table does not exist: {path}")
    frame = pd.read_pickle(path)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Normalized LFP input is not a DataFrame: {path}")
    return frame


def _base_filter(frame: pd.DataFrame, config: StructuralConnectivityConfig) -> pd.Series:
    required = {
        "ID",
        "StimSide",
        "Lat",
        "IncludeAggregate",
        "NormalizationStatus",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Normalized LFP table is missing columns: {missing}")
    subject_sides = {
        subject.subject_id: subject.stimulation_side for subject in config.subjects
    }
    expected_side = frame["ID"].map(subject_sides)
    return (
        frame["ID"].isin(subject_sides)
        & expected_side.notna()
        & frame["StimSide"].eq(expected_side)
        & frame["Lat"].eq(config["outcomes"]["laterality"])
        & frame["IncludeAggregate"].eq(True)
        & frame["NormalizationStatus"].eq("normalized")
    )


def build_channel_catalog(config: StructuralConnectivityConfig) -> pd.DataFrame:
    """Return one row per analysis endpoint channel and anatomical Region."""

    rows: list[pd.DataFrame] = []
    local_path = config.path_value("local_lfp")
    local = _read_frame(local_path)
    local_required = {"Channel", "Region"}
    if not local_required.issubset(local.columns):
        raise KeyError(f"Local LFP table lacks {sorted(local_required)}: {local_path}")
    local = local.loc[
        _base_filter(local, config)
        & local["Region"].isin(["STN", "SNr"]),
        ["ID", "StimSide", "Channel", "Region"],
    ].drop_duplicates()
    local["CatalogSource"] = "local"
    rows.append(local)

    for metric in config["outcomes"]["connectivity_metrics"]:
        path = config.path_value("connectivity_lfp_root") / metric / "mean-scalar.pkl"
        frame = _read_frame(path)
        required = {"channel_a", "channel_b", "region_a", "region_b"}
        if not required.issubset(frame.columns):
            raise KeyError(f"Connectivity LFP table lacks {sorted(required)}: {path}")
        filtered = frame.loc[_base_filter(frame, config)]
        for endpoint in ("a", "b"):
            endpoint_rows = filtered[
                ["ID", "StimSide", f"channel_{endpoint}", f"region_{endpoint}"]
            ].rename(
                columns={
                    f"channel_{endpoint}": "Channel",
                    f"region_{endpoint}": "Region",
                }
            )
            endpoint_rows = endpoint_rows.loc[
                endpoint_rows["Region"].isin(["STN", "SNr"])
            ].drop_duplicates()
            endpoint_rows["CatalogSource"] = f"connectivity:{metric}"
            rows.append(endpoint_rows)

    catalog = pd.concat(rows, ignore_index=True)
    if catalog.empty:
        raise ValueError("No ipsilateral STN/SNr endpoint channels were discovered.")
    grouped = catalog.groupby(["ID", "Channel"], sort=True, observed=True)
    conflicts = grouped["Region"].nunique().loc[lambda values: values > 1]
    if not conflicts.empty:
        raise ValueError(
            "Channels have inconsistent Region assignments: "
            + ", ".join(f"{key[0]}:{key[1]}" for key in conflicts.index)
        )
    output = grouped.agg(
        StimSide=("StimSide", "first"),
        Region=("Region", "first"),
        CatalogSources=(
            "CatalogSource",
            lambda values: ";".join(sorted(set(str(value) for value in values))),
        ),
    ).reset_index()
    for subject in config.subjects:
        if not np.any(output["ID"].eq(subject.subject_id)):
            raise ValueError(
                f"Registered subject has no eligible ipsilateral endpoint channel: "
                f"{subject.subject_id}"
            )
    return output.sort_values(["ID", "Channel"], kind="mergesort").reset_index(drop=True)

