"""Export five supplementary cohort tables from private structured CSV inputs."""

import argparse
from pathlib import Path

import pandas as pd


def build_tables(cohort_root: Path) -> dict[str, pd.DataFrame]:
    """Select manuscript fields without changing scores or programming units."""
    baseline = pd.read_csv(cohort_root / "baseline_covariates.csv", dtype={"ID": str})
    execution = pd.read_csv(
        cohort_root / "tdcs_execution_status.csv", dtype={"ID": str}
    )
    contacts = pd.read_csv(cohort_root / "contact_pair_regions.csv", dtype={"ID": str})
    programming = pd.read_csv(
        cohort_root / "programming_parameters.csv", dtype={"ID": str}
    )
    baseline_labels = {
        "ID": "Subject",
        "AgeYears": "Age (years)",
        "Sex": "Sex",
        "DiseaseDurationYears": "Disease duration (years)",
        "UPDRSIII_MedOFF": "MDS-UPDRS-III (Med-OFF) scores",
        "PDQ39_Total": "PDQ-39 scores",
    }
    execution_labels = {
        "ID": "Subject",
        "StimSide": "Side",
        "AnodeCenteredCompleted": "Anode-centered HD-tDCS",
        "CathodeCenteredCompleted": "Cathode-centered HD-tDCS",
    }
    execution = execution[list(execution_labels)].copy()
    for column in ["AnodeCenteredCompleted", "CathodeCenteredCompleted"]:
        values = execution[column].astype(str)
        if not values.isin(["True", "False"]).all():
            raise ValueError(f"Expected True or False in {column}.")
        execution[column] = values.map({"True": "Completed", "False": "Not completed"})
    pairs = ["0_1", "1_2", "2_3", "8_9", "9_10", "10_11"]
    regions = contacts.pivot(index="ID", columns="ContactPair", values="RegionClass")
    regions = regions.reindex(index=contacts.ID.drop_duplicates(), columns=pairs)
    regions.columns = [
        f"{pair} ({side})" for pair, side in zip(pairs, ["L"] * 3 + ["R"] * 3)
    ]
    regions = regions.rename_axis("Subject").reset_index()
    programming_labels = {
        "ID": "Subject",
        "Contact": "Contact",
        "Target": "Target",
        "Side": "Side",
        "VoltageV": "Voltage (V)",
        "PulseWidthUs": "Pulsewidth (µs)",
        "FrequencyHz": "Frequency (Hz)",
    }
    tables = {
        "Supplementary_Table_01": baseline[list(baseline_labels)].rename(
            columns=baseline_labels
        ),
        "Supplementary_Table_02": execution.rename(columns=execution_labels),
        "Supplementary_Table_03": regions,
    }
    for number, protocol in [(4, "STN"), (5, "STN+SNr")]:
        tables[f"Supplementary_Table_{number:02d}"] = (
            programming.loc[programming.Protocol.eq(protocol), list(programming_labels)]
            .rename(columns=programming_labels)
            .reset_index(drop=True)
        )
    return tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    tables = build_tables(args.cohort_root)
    targets = [args.output_root / f"{name}.csv" for name in tables]
    workbook = args.output_root / "Supplementary_tables.xlsx"
    for path in [*targets, workbook]:
        if path.exists():
            raise FileExistsError(
                f"Use a fresh output directory; output exists: {path}"
            )
    args.output_root.mkdir(parents=True, exist_ok=True)
    for target, table in zip(targets, tables.values()):
        table.to_csv(target, index=False, na_rep="NA")
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        for name, table in tables.items():
            table.to_excel(writer, sheet_name=name, index=False, na_rep="NA")


if __name__ == "__main__":
    main()
