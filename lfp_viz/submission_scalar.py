"""Prepare the manuscript-facing Phase scalar tables and output identities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import VizConfig
from .data import path_below, read_csv, require_columns


BURST_OUTPUT_GROUPS = {
    "mean-scalar": "burst-mean",
    "rate-scalar": "burst-rate",
    "duration-scalar": "burst-duration",
    "occupancy-scalar": "burst-occupancy",
}
OUTPUT_GROUP_ORDER = [
    "aperiodic",
    "burst-mean",
    "burst-rate",
    "burst-duration",
    "burst-occupancy",
    "periodic",
    "raw-power",
    "ciplv",
    "imcoh-abs",
    "wpli",
    "psi",
    "trgc",
]
PAIRING_COLUMNS = {"ModelID", "ID", "ContactUnitID", "Lat", "Included"}


def _contact_unit_id(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value), ensure_ascii=False, separators=(",", ":"))
    return str(value)


def scalar_output_group(metric: object, feature_output: object) -> str:
    """Return the registered submission directory for one scalar model."""

    metric_value = str(metric)
    if metric_value == "burst":
        try:
            return BURST_OUTPUT_GROUPS[str(feature_output)]
        except KeyError as error:
            raise ValueError(
                f"Unsupported burst FeatureOutput: {feature_output!r}"
            ) from error
    group = metric_value.replace("_", "-")
    if group not in OUTPUT_GROUP_ORDER:
        raise ValueError(f"Unsupported submission scalar Metric: {metric_value!r}")
    return group


def add_output_groups(models: pd.DataFrame) -> pd.DataFrame:
    """Attach the output directory identity without changing model identity."""

    require_columns(models, {"Metric", "FeatureOutput"}, "Models manifest")
    result = models.copy()
    result["OutputGroup"] = [
        scalar_output_group(metric, feature_output)
        for metric, feature_output in zip(
            result["Metric"], result["FeatureOutput"], strict=True
        )
    ]
    return result


def filter_contact_pairing(
    selected: pd.DataFrame,
    model: pd.Series,
    pairing: pd.DataFrame,
    *,
    contact_column: str,
) -> pd.DataFrame:
    """Apply the exact contact inclusion decisions used by one fitted model."""

    require_columns(pairing, PAIRING_COLUMNS, "Contact pairing manifest")
    model_pairing = pairing.loc[pairing["ModelID"].eq(model["ModelID"])].copy()
    if model_pairing.empty:
        raise ValueError(
            f"Contact pairing manifest has no rows for {model['ModelID']}."
        )
    duplicate = model_pairing.duplicated(
        ["ID", "ContactUnitID", "Lat"], keep=False
    )
    if duplicate.any():
        raise ValueError(
            f"Contact pairing manifest contains duplicate units for {model['ModelID']}."
        )
    include_values = set(model_pairing["Included"].dropna().unique())
    if not include_values.issubset({True, False}):
        raise TypeError("Contact pairing Included must be boolean.")

    source = selected.copy()
    source["ContactUnitID"] = source[contact_column].map(_contact_unit_id)
    keys = ["ID", "ContactUnitID", "Lat"]
    for frame in (source, model_pairing):
        for column in keys:
            frame[column] = frame[column].astype(str)

    source_units = source[keys].drop_duplicates()
    registered_units = model_pairing[keys].drop_duplicates()
    unknown = source_units.merge(
        registered_units.assign(_registered=True), on=keys, how="left"
    )
    if unknown["_registered"].isna().any():
        raise ValueError(
            f"Visualization source contains unregistered contacts for {model['ModelID']}."
        )

    included = model_pairing.loc[model_pairing["Included"].astype(bool), keys]
    retained = source.merge(included.assign(_included=True), on=keys, how="inner")
    return retained.drop(columns="_included")


def _read_registered_results(
    outputs: pd.DataFrame,
    *,
    config: VizConfig,
    result_type: str,
) -> pd.DataFrame:
    selected = outputs.loc[outputs["ResultType"].eq(result_type)]
    frames: list[pd.DataFrame] = []
    for row in selected.itertuples(index=False):
        path = path_below(Path(str(row.Path)), config.stats_root, "Stats result path")
        table = read_csv(path, f"{result_type} result")
        if len(table) != int(row.Rows):
            raise ValueError(f"Stats output row count differs from its manifest: {path}")
        frames.append(table)
    if not frames:
        raise ValueError(f"No registered {result_type} results were selected.")
    return pd.concat(frames, ignore_index=True)


def _attach_model_metadata(table: pd.DataFrame, models: pd.DataFrame) -> pd.DataFrame:
    requested = [
        "ModelID",
        "Domain",
        "Metric",
        "FeatureOutput",
        "Representation",
        "Band",
        "RegionVariable",
        "RegionValue",
        "Polar",
        "Contrast",
        "OutputGroup",
        "Status",
        "Singular",
        "n_rows",
        "n_ID",
        "n_contact_units",
        "n_contact_units_ipsi",
        "n_contact_units_contra",
    ]
    metadata = models[[column for column in requested if column in models]].rename(
        columns={"Status": "ModelStatus", "Singular": "ModelSingular"}
    )
    missing = [column for column in metadata.columns if column not in table.columns]
    return table.merge(
        metadata[["ModelID", *[column for column in missing if column != "ModelID"]]],
        on="ModelID",
        how="left",
        validate="many_to_one",
    )


def _difference_pairing_summary(
    config: VizConfig, models: pd.DataFrame
) -> pd.DataFrame:
    """Attach source-table paired-difference counts to every submission model."""

    derived = read_csv(config.derived_tables_manifest, "Derived tables manifest")
    keys = ["Domain", "Metric", "FeatureOutput"]
    require_columns(
        derived,
        {
            *keys,
            "InputPath",
            "DerivedPath",
            "PairedRows",
        },
        "Derived tables manifest",
    )
    if derived.duplicated(keys, keep=False).any():
        raise ValueError("Derived tables manifest contains duplicate feature rows.")
    return models.merge(
        derived.drop(columns="AnalysisID", errors="ignore"),
        on=keys,
        how="left",
        validate="many_to_one",
    )


def build_submission_statistics(
    config: VizConfig,
    models: pd.DataFrame,
    outputs: pd.DataFrame,
) -> dict[Path, pd.DataFrame]:
    """Build all co-located result tables and their root manifest."""

    model_ids = set(models["ModelID"].astype(str))
    tests = read_csv(config.tests_manifest, "Tests manifest")
    null_submission = config["inputs"]["test_kind"] == "emmean_vs_null"
    pairing = (
        _difference_pairing_summary(config, models)
        if null_submission
        else read_csv(config.contact_pairing_manifest, "Contact pairing manifest")
    )
    require_columns(tests, {"ModelID"}, "Tests manifest")
    if not null_submission:
        require_columns(pairing, PAIRING_COLUMNS, "Contact pairing manifest")
    tests = tests.loc[tests["ModelID"].astype(str).isin(model_ids)].copy()
    pairing = pairing.loc[pairing["ModelID"].astype(str).isin(model_ids)].copy()
    outputs = outputs.loc[outputs["ModelID"].astype(str).isin(model_ids)].copy()

    model_fit = _attach_model_metadata(
        _read_registered_results(outputs, config=config, result_type="model-fit"),
        models,
    )
    omnibus = _attach_model_metadata(
        _read_registered_results(outputs, config=config, result_type="omnibus"),
        models,
    )
    emmeans = _attach_model_metadata(
        _read_registered_results(outputs, config=config, result_type="emm"), models
    )
    p_column = str(config["statistics"]["p_column"])
    if null_submission:
        null_test = _attach_model_metadata(
            _read_registered_results(
                outputs, config=config, result_type="null-test"
            ),
            models,
        )
        require_columns(
            null_test,
            {p_column, "p_raw", "p_holm"},
            "Consolidated null-test results",
        )
        alpha = float(config["statistics"]["significance_alpha"])
        significant_raw = null_test.loc[
            pd.to_numeric(null_test["p_raw"], errors="coerce") < alpha
        ].copy()
        significant_holm = null_test.loc[
            pd.to_numeric(null_test["p_holm"], errors="coerce") < alpha
        ].copy()
    else:
        tukey = _attach_model_metadata(
            _read_registered_results(outputs, config=config, result_type="tukey"),
            models,
        )
        require_columns(tukey, {p_column}, "Consolidated Tukey results")
        p_values = pd.to_numeric(tukey[p_column], errors="coerce")
        significant = tukey.loc[
            p_values < float(config["statistics"]["significance_alpha"])
        ].copy()

    tests = _attach_model_metadata(tests, models)
    pairing = _attach_model_metadata(pairing, models)
    table_sources: dict[str, pd.DataFrame] = {
        "models": models,
        "tests": tests,
        "model_fit": model_fit,
        "omnibus": omnibus,
        "emmeans": emmeans,
    }
    if null_submission:
        table_sources.update(
            {
                "null_test": null_test,
                "significant_raw": significant_raw,
                "significant_holm": significant_holm,
                "pairing": pairing,
            }
        )
    else:
        table_sources.update(
            {
                "tukey": tukey,
                "significant_tukey": significant,
                "contact_pairing": pairing,
            }
        )
    filenames = config["statistics"]["files"]
    groups = [group for group in OUTPUT_GROUP_ORDER if group in set(models["OutputGroup"])]
    tables: dict[Path, pd.DataFrame] = {}
    manifest_rows: list[dict[str, Any]] = []
    for group in groups:
        group_ids = set(
            models.loc[models["OutputGroup"].eq(group), "ModelID"].astype(str)
        )
        for table_type, source in table_sources.items():
            table = source.loc[source["ModelID"].astype(str).isin(group_ids)].copy()
            path = config.output_root / group / str(filenames[table_type])
            tables[path] = table
            manifest_rows.append(
                {
                    "AnalysisID": config.analysis_id,
                    "StatsAnalysisID": config.stats_analysis_id,
                    "OutputGroup": group,
                    "TableType": table_type,
                    "Path": str(path.relative_to(config.output_root)),
                    "Rows": len(table),
                    "Columns": len(table.columns),
                    "Status": "planned",
                }
            )
    if null_submission:
        root_sources = {
            "models": models,
            "model_fit": model_fit,
            "null_test": null_test,
            "pairing": pairing,
        }
        for table_type, source in root_sources.items():
            path = config.output_root / str(filenames[table_type])
            tables[path] = source.copy()
            manifest_rows.append(
                {
                    "AnalysisID": config.analysis_id,
                    "StatsAnalysisID": config.stats_analysis_id,
                    "OutputGroup": "all",
                    "TableType": table_type,
                    "Path": str(path.relative_to(config.output_root)),
                    "Rows": len(source),
                    "Columns": len(source.columns),
                    "Status": "planned",
                }
            )
    statistics = pd.DataFrame(manifest_rows)
    tables[config.manifest_root / config["manifests"]["statistics"]] = statistics
    return tables
