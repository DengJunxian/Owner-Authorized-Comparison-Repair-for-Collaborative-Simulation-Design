"""Strict loading for frozen empirical-evaluation configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from policytwin.empirical.schema import EmpiricalEvaluationConfig
from policytwin.provenance import SourceArtifact, read_json_object, sha256_file


@dataclass(frozen=True, slots=True)
class LoadedEmpiricalConfig:
    """Immutable config plus byte-level and semantic source commitments."""

    config: EmpiricalEvaluationConfig
    source: SourceArtifact


def load_empirical_config(path: Path) -> LoadedEmpiricalConfig:
    """Load strict JSON; unknown auto-selection fields fail closed."""
    mapping = read_json_object(path)
    config = EmpiricalEvaluationConfig.from_mapping(mapping)
    return LoadedEmpiricalConfig(
        config=config,
        source=SourceArtifact(
            path=str(path),
            source_sha256=sha256_file(path),
            semantic_sha256=config.sha256,
        ),
    )


__all__ = ["LoadedEmpiricalConfig", "load_empirical_config"]
