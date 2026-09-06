"""Record and feature-output discovery."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .feature_outputs import iter_feature_output_specs
from .config import CohortConfig
from .contracts import RecordSpec
from .identity import casefold_identity


def parse_record_policy(record: str, config: Mapping[str, Any]) -> tuple[str, str]:
    metadata = config["metadata"]
    matches = [
        label
        for token, label in metadata["polarity_tokens"].items()
        if re.search(
            rf"(?:^|_){re.escape(token)}(?:_|$)", record, flags=re.IGNORECASE
        )
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Record {record!r} does not identify exactly one configured polarity."
        )
    side_match = re.search(
        metadata["stimulation_side_regex"], record, flags=re.IGNORECASE
    )
    if side_match is None:
        raise ValueError(f"Record {record!r} does not identify a stimulation side.")
    return matches[0], side_match.group(1).upper()


def _record_selected(subject: str, record: str, selectors: Sequence[str]) -> bool:
    if not selectors:
        return True
    identity = f"{subject}/{record}"
    candidates = {casefold_identity(record), casefold_identity(identity)}
    return any(casefold_identity(selector) in candidates for selector in selectors)


def discover_records(
    config: CohortConfig,
    selectors: Sequence[str] = (),
    input_root: Path | None = None,
) -> list[RecordSpec]:
    """Discover selected records and enforce the feature-output matrix."""
    root = Path(input_root) if input_root is not None else config.input_root
    discovery = config["discovery"]
    prefixes = tuple(discovery.get("ignore_basename_prefixes", []))
    specs = list(iter_feature_output_specs(config))
    records: list[RecordSpec] = []

    for subject_dir in sorted(root.glob(discovery["subject_glob"])):
        if not subject_dir.is_dir() or subject_dir.name.startswith(prefixes):
            continue
        for record_dir in sorted(subject_dir.glob(discovery["record_glob"])):
            if not record_dir.is_dir() or record_dir.name.startswith(prefixes):
                continue
            if not _record_selected(subject_dir.name, record_dir.name, selectors):
                continue
            feature_root = record_dir / discovery["features_dir"]
            if not feature_root.is_dir():
                continue
            polarity, stimulation_side = parse_record_policy(record_dir.name, config)
            feature_output_paths: dict[tuple[str, str, str], Path] = {}
            configured_paths: set[Path] = set()
            for spec in specs:
                candidates = [
                    path
                    for path in feature_root.glob(
                        f"*/{spec.metric}/{spec.feature_output}.pkl"
                    )
                    if not path.name.startswith(prefixes)
                ]
                if len(candidates) != 1:
                    raise ValueError(
                        f"Expected one {spec.metric}/{spec.feature_output}.pkl in "
                        f"{record_dir}; found {len(candidates)}."
                    )
                path = candidates[0]
                feature_output_paths[spec.key] = path
                configured_paths.add(path)
            all_pkls = {
                path
                for path in feature_root.rglob("*.pkl")
                if not path.name.startswith(prefixes)
            }
            records.append(
                RecordSpec(
                    subject=subject_dir.name,
                    record=record_dir.name,
                    path=record_dir,
                    polarity=polarity,
                    stimulation_side=stimulation_side,
                    feature_output_paths=feature_output_paths,
                    extras=tuple(sorted(all_pkls - configured_paths)),
                )
            )

    seen_identities: dict[str, str] = {}
    duplicate_identities: list[tuple[str, str]] = []
    for record in records:
        folded = casefold_identity(record.identity)
        if folded in seen_identities:
            duplicate_identities.append((seen_identities[folded], record.identity))
        else:
            seen_identities[folded] = record.identity
    if duplicate_identities:
        raise ValueError(
            "Case-insensitive duplicate subject/record identities were discovered: "
            f"{duplicate_identities}"
        )

    if not records:
        selection = f" for selectors {list(selectors)}" if selectors else ""
        raise FileNotFoundError(f"No records were discovered under {root}{selection}")
    unmatched = [
        selector
        for selector in selectors
        if not any(
            casefold_identity(selector)
            in {
                casefold_identity(record.record),
                casefold_identity(record.identity),
            }
            for record in records
        )
    ]
    if unmatched:
        raise ValueError(
            f"Record selectors did not match discovered records: {unmatched}"
        )
    return records


def dry_run_summary(
    records: Sequence[RecordSpec], config: CohortConfig
) -> dict[str, Any]:
    specs = list(iter_feature_output_specs(config))
    return {
        "input_root": str(config.input_root),
        "output_root": str(config.output_root),
        "subjects": sorted({record.subject for record in records}),
        "n_subjects": len({record.subject for record in records}),
        "n_records": len(records),
        "n_feature_outputs_per_record": len(specs),
        "n_configured_inputs": len(records) * len(specs),
        "n_extra_pkls": sum(len(record.extras) for record in records),
        "records": [record.identity for record in records],
    }
