"""Stable data identities shared by the cohort processing modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .config import CohortConfig


DOMAINS = ("local", "connectivity")
REPRESENTATIONS = ("scalar", "spectral", "trace", "raw")
AGGREGATION_LEVELS = ("contact", "region", "subject")
SOURCE_STAGES = ("merge", "transform", "normalize")
STAGES = (*SOURCE_STAGES, "aggregate", "split", "all")


@dataclass(frozen=True)
class FeatureOutputSpec:
    domain: str
    metric: str
    feature_output: str
    representation: str
    reducer: str
    direction: str | None = None

    @property
    def key(self) -> tuple[str, str, str]:
        return self.domain, self.metric, self.feature_output


@dataclass(frozen=True)
class RecordSpec:
    subject: str
    record: str
    path: Path
    polarity: str
    stimulation_side: str
    feature_output_paths: Mapping[tuple[str, str, str], Path]
    extras: tuple[Path, ...]

    @property
    def identity(self) -> str:
        return f"{self.subject}/{self.record}"


@dataclass
class RunState:
    config: CohortConfig
    overwrite: bool
    invocation_stage: str
    record_selectors: tuple[str, ...] = ()
    feature_output_manifest: list[dict[str, Any]] = field(default_factory=list)
    qc: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    flip_provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def output_root(self) -> Path:
        return self.config.output_root

    def add_qc(self, name: str, rows: Iterable[dict[str, Any]]) -> None:
        self.qc.setdefault(name, []).extend(rows)
