"""Typed, deterministic derivation of paper statistics from audited draws.

The derivation artifact is a frozen recipe and commitment, never a container for
reported values.  Every displayed number is recomputed from completed
``ExperimentResult`` distribution draws by the assembler in this module.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import NoReturn, Self, cast

from policytwin.evaluation import (
    SelectionRule,
    SelectivePredictionError,
    SelectivePredictionEvaluation,
    SelectivePredictionRecord,
    ValidityCertificate,
    ValidityEvidenceAuthority,
    evaluate_selective_predictions,
)
from policytwin.experiments import (
    DISTRIBUTION_RESULT_SCHEMA_VERSION,
    DistributionMCSEEstimate,
    DistributionMCSESpec,
    ExperimentMatrix,
    ExperimentResult,
    MCSEStatistic,
)
from policytwin.provenance import canonical_json_bytes, sha256_bytes, sha256_file

STATISTICS_DERIVATION_SCHEMA = "policytwin.reporting-statistics-derivation.v3"
STATISTICS_ASSEMBLER_ID = "policytwin.reporting.typed_draw_assembler"
STATISTICS_ASSEMBLER_VERSION = "3.0.0"
TYPED_REPORTING_RUNNER_KIND = "typed_reporting_source_v1"
REPORT_SECTIONS: tuple[str, ...] = (
    "paired_distribution",
    "predicted_vs_empirical",
    "coverage_width",
    "group_benefit_harm",
    "ablation_table",
    "calibration_diagnostics",
    "risk_coverage",
    "certificate_heatmap",
    "compute_scaling_table",
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_CERTIFICATE_COLUMNS = ("pass", "warn", "fail", "unknown")
_CERTIFICATE_CHECK_IDS = (
    "temporal_leakage",
    "pre_policy_posterior_predictive",
    "ood_policy_distance",
    "model_disagreement",
    "seed_instability",
    "prompt_instability",
    "parameter_support_failure",
    "empirical_coverage_history",
    "unsupported_group_labels",
    "missing_required_mechanisms",
)

Scalar = str | int | float | None
FrozenRow = tuple[tuple[str, Scalar], ...]
FrozenSection = tuple[str, tuple[FrozenRow, ...]]


class StatisticsDerivationError(RuntimeError):
    """Base class for typed-statistics derivation failures."""


class StatisticsDerivationSchemaError(StatisticsDerivationError, ValueError):
    """Raised for a malformed or unsupported derivation artifact."""


class StatisticsDerivationIntegrityError(StatisticsDerivationError):
    """Raised when an artifact disagrees with audited inputs or recomputation."""


def _schema_fail(message: str) -> NoReturn:
    raise StatisticsDerivationSchemaError(message)


def _integrity_fail(message: str) -> NoReturn:
    raise StatisticsDerivationIntegrityError(message)


def _strict_fields(
    value: Mapping[str, object], expected: frozenset[str], context: str
) -> None:
    actual = frozenset(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        _schema_fail(f"{context} is missing fields: {', '.join(missing)}")
    if extra:
        _schema_fail(f"{context} has unknown fields: {', '.join(extra)}")


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        _schema_fail(f"{field} must be an object with string keys")
    return cast(Mapping[str, object], value)


def _array(value: object, field: str, *, allow_empty: bool = False) -> Sequence[object]:
    if not isinstance(value, list) or (not value and not allow_empty):
        requirement = "an array" if allow_empty else "a non-empty array"
        _schema_fail(f"{field} must be {requirement}")
    return cast(Sequence[object], value)


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        _schema_fail(f"{field} must be a stable identifier")
    return value


def _text(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 256
        or "\x00" in value
    ):
        _schema_fail(f"{field} must be non-empty text of at most 256 characters")
    return value


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _schema_fail(f"{field} must be a lowercase SHA-256 digest")
    return value


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        _schema_fail(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        _schema_fail(f"{field} must be finite")
    return result


def _fraction(value: object, field: str, *, open_interval: bool = False) -> float:
    result = _finite(value, field)
    if not 0.0 <= result <= 1.0 or (open_interval and result in {0.0, 1.0}):
        interval = "(0, 1)" if open_interval else "[0, 1]"
        _schema_fail(f"{field} must lie in {interval}")
    return result


def _selection_rule_from_mapping(value: object) -> SelectionRule:
    try:
        return SelectionRule.from_mapping(_mapping(value, "selection_rule"))
    except SelectivePredictionError as error:
        raise StatisticsDerivationSchemaError(
            "statistics derivation selection rule is invalid"
        ) from error


class ReportSection(StrEnum):
    """The exact nine sections allowed in a formal paper bundle."""

    PAIRED_DISTRIBUTION = "paired_distribution"
    PREDICTED_VS_EMPIRICAL = "predicted_vs_empirical"
    COVERAGE_WIDTH = "coverage_width"
    GROUP_BENEFIT_HARM = "group_benefit_harm"
    ABLATION_TABLE = "ablation_table"
    CALIBRATION_DIAGNOSTICS = "calibration_diagnostics"
    RISK_COVERAGE = "risk_coverage"
    CERTIFICATE_HEATMAP = "certificate_heatmap"
    COMPUTE_SCALING_TABLE = "compute_scaling_table"


class ReportingSourceKind(StrEnum):
    """Typed meanings that a completed distribution-result row may carry."""

    PAIRED_EFFECT_DRAWS = "paired_effect_draws"
    PREDICTED_DRAWS = "predicted_draws"
    EMPIRICAL_DRAWS = "empirical_draws"
    INTERVAL_COVERAGE_INDICATORS = "interval_coverage_indicators"
    INTERVAL_WIDTH_DRAWS = "interval_width_draws"
    GROUP_EFFECT_DRAWS = "group_effect_draws"
    ABLATION_EFFECT_DRAWS = "ablation_effect_draws"
    CALIBRATION_OBSERVED_DRAWS = "calibration_observed_draws"
    CALIBRATION_PREDICTED_DRAWS = "calibration_predicted_draws"
    RISK_UNCERTAINTY_DRAWS = "risk_uncertainty_draws"
    RISK_LOSS_DRAWS = "risk_loss_draws"
    COMPUTE_WALL_SECONDS_DRAWS = "compute_wall_seconds_draws"


_SOURCE_KINDS: dict[ReportSection, tuple[ReportingSourceKind, ...]] = {
    ReportSection.PAIRED_DISTRIBUTION: (ReportingSourceKind.PAIRED_EFFECT_DRAWS,),
    ReportSection.PREDICTED_VS_EMPIRICAL: (
        ReportingSourceKind.PREDICTED_DRAWS,
        ReportingSourceKind.EMPIRICAL_DRAWS,
    ),
    ReportSection.COVERAGE_WIDTH: (
        ReportingSourceKind.INTERVAL_COVERAGE_INDICATORS,
        ReportingSourceKind.INTERVAL_WIDTH_DRAWS,
    ),
    ReportSection.GROUP_BENEFIT_HARM: (ReportingSourceKind.GROUP_EFFECT_DRAWS,),
    ReportSection.ABLATION_TABLE: (ReportingSourceKind.ABLATION_EFFECT_DRAWS,),
    ReportSection.CALIBRATION_DIAGNOSTICS: (
        ReportingSourceKind.CALIBRATION_OBSERVED_DRAWS,
        ReportingSourceKind.CALIBRATION_PREDICTED_DRAWS,
    ),
    ReportSection.RISK_COVERAGE: (
        ReportingSourceKind.RISK_UNCERTAINTY_DRAWS,
        ReportingSourceKind.RISK_LOSS_DRAWS,
    ),
    ReportSection.CERTIFICATE_HEATMAP: (),
    ReportSection.COMPUTE_SCALING_TABLE: (
        ReportingSourceKind.COMPUTE_WALL_SECONDS_DRAWS,
    ),
}

_ARGUMENT_FIELDS: dict[ReportSection, tuple[str, ...]] = {
    ReportSection.PAIRED_DISTRIBUTION: ("difference",),
    ReportSection.PREDICTED_VS_EMPIRICAL: ("label",),
    ReportSection.COVERAGE_WIDTH: (
        "label",
        "registered_output_id",
        "metric_id",
        "cohort_id",
        "horizon_id",
        "interval_kind",
        "interval_method_id",
        "estimand_id",
        "sampling_unit",
        "level",
    ),
    ReportSection.GROUP_BENEFIT_HARM: (
        "group",
        "cohort_basis",
        "cohort_definition_sha256",
    ),
    ReportSection.ABLATION_TABLE: (
        "mechanism",
        "uncertainty_kind",
        "method_id",
        "mcse_spec_id",
        "level",
        "sampling_unit",
        "estimand_id",
    ),
    ReportSection.CALIBRATION_DIAGNOSTICS: ("statistic",),
    ReportSection.RISK_COVERAGE: ("uncertainty_threshold",),
    ReportSection.CERTIFICATE_HEATMAP: ("row", "column"),
    ReportSection.COMPUTE_SCALING_TABLE: (),
}


def statistics_assembler_source_sha256() -> str:
    """Hash the exact assembler source file executing the derivation."""

    return sha256_file(Path(__file__).resolve())


def _normalized_arguments(
    section: ReportSection, raw: Mapping[str, object]
) -> tuple[tuple[str, Scalar], ...]:
    fields = _ARGUMENT_FIELDS[section]
    _strict_fields(raw, frozenset(fields), f"{section.value} cell arguments")
    normalized: list[tuple[str, Scalar]] = []
    for field in fields:
        value = raw[field]
        context = f"{section.value}.{field}"
        if field in {
            "registered_output_id",
            "metric_id",
            "cohort_id",
            "horizon_id",
            "interval_kind",
            "interval_method_id",
            "estimand_id",
            "sampling_unit",
            "group",
            "cohort_basis",
            "mechanism",
            "uncertainty_kind",
            "method_id",
            "mcse_spec_id",
            "statistic",
            "row",
            "column",
        }:
            parsed: Scalar = _identifier(value, context)
        elif field == "label":
            parsed = _text(value, context)
        elif field == "cohort_definition_sha256":
            parsed = _sha256(value, context)
        elif field in {"difference", "uncertainty_threshold"}:
            parsed = _finite(value, context)
            if field == "uncertainty_threshold" and parsed < 0.0:
                _schema_fail("risk_coverage.uncertainty_threshold must be non-negative")
        elif field == "level":
            parsed = (
                None if value is None else _fraction(value, context, open_interval=True)
            )
        else:  # pragma: no cover - guarded by the frozen field registry
            raise AssertionError(f"unhandled derivation argument: {field}")
        normalized.append((field, parsed))
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class DerivationSourceRow:
    """Exact binding from one audited completed row to a typed source meaning."""

    row_id: str
    config_sha256: str
    result_sha256: str
    source_kind: ReportingSourceKind

    def __post_init__(self) -> None:
        _identifier(self.row_id, "derivation source row_id")
        _sha256(self.config_sha256, "derivation source config_sha256")
        _sha256(self.result_sha256, "derivation source result_sha256")
        if not isinstance(self.source_kind, ReportingSourceKind):
            _schema_fail("derivation source_kind is invalid")

    def to_mapping(self) -> dict[str, object]:
        return {
            "config_sha256": self.config_sha256,
            "result_sha256": self.result_sha256,
            "row_id": self.row_id,
            "source_kind": self.source_kind.value,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(
            value,
            frozenset({"config_sha256", "result_sha256", "row_id", "source_kind"}),
            "derivation source row",
        )
        raw_kind = value["source_kind"]
        if not isinstance(raw_kind, str):
            _schema_fail("derivation source_kind must be a string")
        try:
            kind = ReportingSourceKind(raw_kind)
        except (TypeError, ValueError) as error:
            raise StatisticsDerivationSchemaError(
                "derivation source_kind is unsupported"
            ) from error
        return cls(
            row_id=_identifier(value["row_id"], "derivation source row_id"),
            config_sha256=_sha256(
                value["config_sha256"], "derivation source config_sha256"
            ),
            result_sha256=_sha256(
                value["result_sha256"], "derivation source result_sha256"
            ),
            source_kind=kind,
        )


@dataclass(frozen=True, slots=True)
class StatisticCellRecipe:
    """One output cell with a strict section-specific source signature."""

    section: ReportSection
    cell_id: str
    source_row_ids: tuple[str, ...]
    arguments: tuple[tuple[str, Scalar], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.section, ReportSection):
            _schema_fail("derivation cell section is invalid")
        _identifier(self.cell_id, "derivation cell_id")
        if not isinstance(self.source_row_ids, tuple):
            _schema_fail("derivation source_row_ids must be immutable")
        normalized_rows = tuple(
            _identifier(item, "derivation cell source row_id")
            for item in self.source_row_ids
        )
        if len(normalized_rows) != len(_SOURCE_KINDS[self.section]):
            _schema_fail(
                f"{self.section.value} cell requires exactly "
                f"{len(_SOURCE_KINDS[self.section])} source rows"
            )
        if len(set(normalized_rows)) != len(normalized_rows):
            _schema_fail("a derivation cell cannot repeat a source row")
        object.__setattr__(self, "source_row_ids", normalized_rows)
        if not isinstance(self.arguments, tuple):
            _schema_fail("derivation cell arguments must be immutable")
        raw = {key: value for key, value in self.arguments}
        if len(raw) != len(self.arguments):
            _schema_fail("derivation cell arguments contain duplicate fields")
        object.__setattr__(self, "arguments", _normalized_arguments(self.section, raw))

    @property
    def argument_mapping(self) -> dict[str, Scalar]:
        return dict(self.arguments)

    def to_mapping(self) -> dict[str, object]:
        return {
            "arguments": self.argument_mapping,
            "cell_id": self.cell_id,
            "section": self.section.value,
            "source_row_ids": list(self.source_row_ids),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(
            value,
            frozenset({"arguments", "cell_id", "section", "source_row_ids"}),
            "derivation cell",
        )
        raw_section = value["section"]
        if not isinstance(raw_section, str):
            _schema_fail("derivation cell section must be a string")
        try:
            section = ReportSection(raw_section)
        except (TypeError, ValueError) as error:
            raise StatisticsDerivationSchemaError(
                "derivation cell section is unsupported"
            ) from error
        raw_rows = _array(
            value["source_row_ids"],
            "derivation source_row_ids",
            allow_empty=section is ReportSection.CERTIFICATE_HEATMAP,
        )
        arguments = _mapping(value["arguments"], "derivation cell arguments")
        return cls(
            section=section,
            cell_id=_identifier(value["cell_id"], "derivation cell_id"),
            source_row_ids=tuple(
                _identifier(item, "derivation source row_id") for item in raw_rows
            ),
            arguments=tuple(_normalized_arguments(section, arguments)),
        )


@dataclass(frozen=True, slots=True)
class StatisticsDerivationArtifact:
    """Externally anchored recipe for all formal reporting cells."""

    artifact_id: str
    matrix_sha256: str
    matrix_run_manifest_sha256: str
    assembler_source_sha256: str
    source_rows: tuple[DerivationSourceRow, ...]
    cells: tuple[StatisticCellRecipe, ...]
    selection_rule: SelectionRule
    validity_certificate_sha256: str
    statistics_sha256: str
    assembler_id: str = STATISTICS_ASSEMBLER_ID
    assembler_version: str = STATISTICS_ASSEMBLER_VERSION
    schema_version: str = STATISTICS_DERIVATION_SCHEMA

    def __post_init__(self) -> None:
        _identifier(self.artifact_id, "statistics derivation artifact_id")
        _sha256(self.matrix_sha256, "derivation matrix_sha256")
        _sha256(
            self.matrix_run_manifest_sha256,
            "derivation matrix_run_manifest_sha256",
        )
        _sha256(self.assembler_source_sha256, "assembler_source_sha256")
        _sha256(
            self.validity_certificate_sha256,
            "derivation validity_certificate_sha256",
        )
        _sha256(self.statistics_sha256, "derivation statistics_sha256")
        if self.assembler_id != STATISTICS_ASSEMBLER_ID:
            _schema_fail("statistics derivation assembler_id is unsupported")
        if self.assembler_version != STATISTICS_ASSEMBLER_VERSION:
            _schema_fail("statistics derivation assembler_version is unsupported")
        if self.schema_version != STATISTICS_DERIVATION_SCHEMA:
            _schema_fail("statistics derivation schema_version is unsupported")
        if self.assembler_source_sha256 != statistics_assembler_source_sha256():
            _integrity_fail(
                "statistics derivation assembler source hash does not match live code"
            )
        if type(self.selection_rule) is not SelectionRule:
            _schema_fail("statistics derivation must retain a typed SelectionRule")
        try:
            validated_rule = SelectionRule.from_mapping(
                self.selection_rule.to_mapping()
            )
        except SelectivePredictionError as error:
            raise StatisticsDerivationSchemaError(
                "statistics derivation selection rule is invalid"
            ) from error
        if validated_rule.synthetic_only is not True:
            _schema_fail("formal derivation currently supports synthetic inputs only")
        object.__setattr__(self, "selection_rule", validated_rule)
        if not isinstance(self.source_rows, tuple) or not self.source_rows:
            _schema_fail("statistics derivation source_rows must be non-empty")
        if not all(isinstance(item, DerivationSourceRow) for item in self.source_rows):
            _schema_fail("statistics derivation contains an invalid source row")
        source_ids = tuple(item.row_id for item in self.source_rows)
        if len(set(source_ids)) != len(source_ids):
            _schema_fail("statistics derivation source row IDs must be unique")
        if not isinstance(self.cells, tuple) or not self.cells:
            _schema_fail("statistics derivation cells must be non-empty")
        if not all(isinstance(item, StatisticCellRecipe) for item in self.cells):
            _schema_fail("statistics derivation contains an invalid cell")
        cell_ids = tuple(item.cell_id for item in self.cells)
        if len(set(cell_ids)) != len(cell_ids):
            _schema_fail("statistics derivation cell IDs must be globally unique")
        section_order = {section: index for index, section in enumerate(ReportSection)}
        observed_order = tuple(
            (section_order[item.section], item.cell_id) for item in self.cells
        )
        if observed_order != tuple(sorted(observed_order)):
            _schema_fail(
                "statistics derivation cells must be sorted by section and cell_id"
            )
        observed_sections = tuple(
            dict.fromkeys(item.section.value for item in self.cells)
        )
        if observed_sections != REPORT_SECTIONS:
            _schema_fail(
                "statistics derivation must contain all nine sections in "
                "canonical order"
            )
        risk_cells = tuple(
            item for item in self.cells if item.section is ReportSection.RISK_COVERAGE
        )
        risk_thresholds = tuple(
            cast(float, item.argument_mapping["uncertainty_threshold"])
            for item in risk_cells
        )
        if risk_thresholds != validated_rule.uncertainty_thresholds:
            _schema_fail(
                "risk-coverage cells must cover every frozen selection-rule "
                "threshold exactly once and in increasing order"
            )
        risk_sources = {item.source_row_ids for item in risk_cells}
        if len(risk_sources) != 1:
            _schema_fail("all risk-coverage thresholds must use one paired draw source")
        heatmap_cells = tuple(
            item
            for item in self.cells
            if item.section is ReportSection.CERTIFICATE_HEATMAP
        )
        heatmap_keys = {
            (
                cast(str, item.argument_mapping["row"]),
                cast(str, item.argument_mapping["column"]),
            )
            for item in heatmap_cells
        }
        expected_heatmap_keys = {
            (check_id, column)
            for check_id in _CERTIFICATE_CHECK_IDS
            for column in _CERTIFICATE_COLUMNS
        }
        if heatmap_keys != expected_heatmap_keys or len(heatmap_cells) != len(
            expected_heatmap_keys
        ):
            _schema_fail(
                "certificate heatmap must contain the exact canonical 10x4 status grid"
            )

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def to_mapping(self) -> dict[str, object]:
        return {
            "artifact_id": self.artifact_id,
            "assembler_id": self.assembler_id,
            "assembler_source_sha256": self.assembler_source_sha256,
            "assembler_version": self.assembler_version,
            "cells": [item.to_mapping() for item in self.cells],
            "matrix_run_manifest_sha256": self.matrix_run_manifest_sha256,
            "matrix_sha256": self.matrix_sha256,
            "schema_version": self.schema_version,
            "selection_rule": self.selection_rule.to_mapping(),
            "source_rows": [item.to_mapping() for item in self.source_rows],
            "statistics_sha256": self.statistics_sha256,
            "validity_certificate_sha256": self.validity_certificate_sha256,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(
            value,
            frozenset(
                {
                    "artifact_id",
                    "assembler_id",
                    "assembler_source_sha256",
                    "assembler_version",
                    "cells",
                    "matrix_run_manifest_sha256",
                    "matrix_sha256",
                    "schema_version",
                    "selection_rule",
                    "source_rows",
                    "statistics_sha256",
                    "validity_certificate_sha256",
                }
            ),
            "statistics derivation artifact",
        )
        raw_sources = _array(value["source_rows"], "derivation source_rows")
        raw_cells = _array(value["cells"], "derivation cells")
        return cls(
            artifact_id=_identifier(value["artifact_id"], "derivation artifact_id"),
            matrix_sha256=_sha256(value["matrix_sha256"], "matrix_sha256"),
            matrix_run_manifest_sha256=_sha256(
                value["matrix_run_manifest_sha256"],
                "matrix_run_manifest_sha256",
            ),
            assembler_source_sha256=_sha256(
                value["assembler_source_sha256"], "assembler_source_sha256"
            ),
            source_rows=tuple(
                DerivationSourceRow.from_mapping(_mapping(item, "derivation source"))
                for item in raw_sources
            ),
            cells=tuple(
                StatisticCellRecipe.from_mapping(_mapping(item, "derivation cell"))
                for item in raw_cells
            ),
            selection_rule=_selection_rule_from_mapping(value["selection_rule"]),
            validity_certificate_sha256=_sha256(
                value["validity_certificate_sha256"],
                "validity_certificate_sha256",
            ),
            statistics_sha256=_sha256(value["statistics_sha256"], "statistics_sha256"),
            assembler_id=_identifier(value["assembler_id"], "assembler_id"),
            assembler_version=_text(value["assembler_version"], "assembler_version"),
            schema_version=_text(value["schema_version"], "schema_version"),
        )


@dataclass(frozen=True, slots=True)
class DerivedReportingStatistics:
    """Deeply immutable recomputed rows for the nine paper sections."""

    sections: tuple[FrozenSection, ...]

    def __post_init__(self) -> None:
        if tuple(section for section, _ in self.sections) != REPORT_SECTIONS:
            _schema_fail("derived statistics sections are incomplete or out of order")
        if any(not rows for _, rows in self.sections):
            _schema_fail("every derived statistics section must contain a row")

    def to_mapping(self) -> dict[str, object]:
        return {section: [dict(row) for row in rows] for section, rows in self.sections}

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))


def _freeze_rows(rows: list[dict[str, Scalar]]) -> tuple[FrozenRow, ...]:
    return tuple(tuple(row.items()) for row in rows)


def _mean(values: tuple[float, ...]) -> float:
    return math.fsum(values) / len(values)


def _validated_sources(
    artifact: StatisticsDerivationArtifact,
    completed_results: tuple[ExperimentResult, ...],
    matrix: ExperimentMatrix,
    matrix_run_manifest_sha256: str,
) -> tuple[dict[str, ExperimentResult], dict[str, ReportingSourceKind]]:
    if artifact.matrix_sha256 != matrix.sha256:
        _integrity_fail("statistics derivation matrix hash mismatch")
    if artifact.matrix_run_manifest_sha256 != matrix_run_manifest_sha256:
        _integrity_fail("statistics derivation run-manifest hash mismatch")
    expected = tuple(
        (result.row_id, result.config_sha256, result.sha256)
        for result in completed_results
    )
    observed = tuple(
        (source.row_id, source.config_sha256, source.result_sha256)
        for source in artifact.source_rows
    )
    if observed != expected:
        _integrity_fail(
            "statistics derivation source rows omit, add, reorder, or alter "
            "audited completed rows"
        )
    results = {result.row_id: result for result in completed_results}
    kinds = {source.row_id: source.source_kind for source in artifact.source_rows}
    for source in artifact.source_rows:
        result = results[source.row_id]
        if result.schema_version != DISTRIBUTION_RESULT_SCHEMA_VERSION:
            _integrity_fail(
                f"typed reporting source {source.row_id!r} uses an unsupported "
                "generic scalar result"
            )
        if result.runner_kind != TYPED_REPORTING_RUNNER_KIND:
            _integrity_fail(
                f"typed reporting source {source.row_id!r} has an unsupported runner"
            )
        if result.metric_name != source.source_kind.value:
            _integrity_fail(
                f"typed reporting source {source.row_id!r} metric/source kind mismatch"
            )
        if result.mechanism_execution is not True:
            _integrity_fail(
                f"typed reporting source {source.row_id!r} did not execute "
                "its mechanism"
            )
        matrix_row = matrix.row(source.row_id)
        if matrix_row.config_sha256 != source.config_sha256:
            _integrity_fail(
                f"typed reporting source {source.row_id!r} changed matrix config"
            )
        if result.frozen_inputs_sha256 != matrix_row.frozen_inputs.sha256:
            _integrity_fail(
                f"typed reporting source {source.row_id!r} changed frozen inputs"
            )
        if (
            result.draw_identity_sha256 != matrix_row.draw_identity_sha256
            or result.shared_draw_identity is not matrix_row.has_shared_draw_identity
        ):
            _integrity_fail(
                f"typed reporting source {source.row_id!r} changed draw identity"
            )
    return results, kinds


def _source_results(
    cell: StatisticCellRecipe,
    results: Mapping[str, ExperimentResult],
    kinds: Mapping[str, ReportingSourceKind],
) -> tuple[ExperimentResult, ...]:
    resolved: list[ExperimentResult] = []
    for row_id, expected_kind in zip(
        cell.source_row_ids, _SOURCE_KINDS[cell.section], strict=True
    ):
        result = results.get(row_id)
        if result is None:
            _integrity_fail(
                f"derivation cell {cell.cell_id!r} references an unaudited row"
            )
        if kinds[row_id] is not expected_kind:
            _integrity_fail(
                f"derivation cell {cell.cell_id!r} has the wrong typed source order"
            )
        resolved.append(result)
    return tuple(resolved)


def _require_values(
    values: tuple[float, ...],
    *,
    field: str,
    minimum: float | None = None,
    maximum: float | None = None,
    binary: bool = False,
) -> None:
    if binary and any(value not in {0.0, 1.0} for value in values):
        _integrity_fail(f"{field} must contain only exact 0/1 indicators")
    if minimum is not None and any(value < minimum for value in values):
        _integrity_fail(f"{field} contains a value below {minimum}")
    if maximum is not None and any(value > maximum for value in values):
        _integrity_fail(f"{field} contains a value above {maximum}")


def _require_paired_draw_identity(
    left: ExperimentResult,
    right: ExperimentResult,
    *,
    section: str,
) -> None:
    if left.frozen_inputs_sha256 != right.frozen_inputs_sha256:
        _integrity_fail(f"{section} sources use different frozen inputs")
    if not left.shared_draw_identity or not right.shared_draw_identity:
        _integrity_fail(
            f"{section} sources lack an explicit shared draw identity contract"
        )
    if left.draw_identity_sha256 != right.draw_identity_sha256:
        _integrity_fail(f"{section} sources use different draw identity contracts")
    if left.event_keys != right.event_keys:
        _integrity_fail(f"{section} sources have mismatched draw event keys")


def _revalidated_validity_certificate(
    certificate: object,
) -> ValidityCertificate:
    if type(certificate) is not ValidityCertificate:
        _schema_fail("validity_certificate must be a typed ValidityCertificate")
    assert isinstance(certificate, ValidityCertificate)
    try:
        validated = ValidityCertificate(
            schema_version=certificate.schema_version,
            level=certificate.level,
            diagnostic_level=certificate.diagnostic_level,
            evidence_authority=certificate.evidence_authority,
            thresholds=certificate.thresholds,
            evidence=certificate.evidence,
            thresholds_sha256=certificate.thresholds_sha256,
            evidence_sha256=certificate.evidence_sha256,
            checks=certificate.checks,
            synthetic_only=certificate.synthetic_only,
        )
    except (RuntimeError, ValueError) as error:
        raise StatisticsDerivationIntegrityError(
            "validity certificate failed semantic revalidation"
        ) from error
    if validated != certificate:
        _integrity_fail("validity certificate is not canonical")
    if validated.synthetic_only is not True:
        _integrity_fail("formal derivation currently supports synthetic inputs only")
    if (
        validated.evidence_authority
        is not ValidityEvidenceAuthority.SELF_REPORTED_SYNTHETIC_FIXTURE
    ):
        _integrity_fail("validity evidence authority is unsupported")
    return validated


def _risk_evaluation(
    uncertainty_result: ExperimentResult,
    loss_result: ExperimentResult,
    rule: SelectionRule,
) -> SelectivePredictionEvaluation:
    if len(uncertainty_result.samples) != len(loss_result.samples):
        _integrity_fail("risk uncertainty and loss sources have different counts")
    _require_paired_draw_identity(
        uncertainty_result,
        loss_result,
        section="risk-coverage",
    )
    _require_values(
        uncertainty_result.samples,
        field="risk uncertainty draws",
        minimum=0.0,
    )
    _require_values(loss_result.samples, field="risk losses", minimum=0.0)
    records = tuple(
        SelectivePredictionRecord(
            prediction_id=event_key,
            uncertainty_score=uncertainty,
            loss=loss,
        )
        for event_key, uncertainty, loss in zip(
            uncertainty_result.event_keys,
            uncertainty_result.samples,
            loss_result.samples,
            strict=True,
        )
    )
    try:
        return evaluate_selective_predictions(records, rule)
    except SelectivePredictionError as error:
        raise StatisticsDerivationIntegrityError(
            "risk-coverage sources do not reproduce the frozen selection rule"
        ) from error


def _reported_mean_distribution_estimate(
    result: ExperimentResult,
    *,
    mcse_spec_id: str,
) -> DistributionMCSEEstimate:
    """Resolve the uniquely frozen event-keyed MCSE record for the reported mean."""

    if len(result.distribution_mcse_specs) != len(result.distribution_estimates):
        _integrity_fail(
            f"ablation source {result.row_id!r} has unpaired distribution MCSE "
            "specs and estimates"
        )

    matches: list[tuple[DistributionMCSESpec, DistributionMCSEEstimate]] = []
    for spec, estimate in zip(
        result.distribution_mcse_specs,
        result.distribution_estimates,
        strict=True,
    ):
        if spec.spec_id == mcse_spec_id:
            matches.append((spec, estimate))

    if not matches:
        _integrity_fail(
            f"ablation source {result.row_id!r} is missing frozen MCSE spec "
            f"{mcse_spec_id!r}"
        )
    if len(matches) != 1:
        _integrity_fail(
            f"ablation source {result.row_id!r} has ambiguous frozen MCSE spec "
            f"{mcse_spec_id!r}"
        )
    spec, match = matches[0]
    if spec.statistic is not MCSEStatistic.MEAN:
        _integrity_fail(
            f"ablation source {result.row_id!r} frozen MCSE spec "
            f"{mcse_spec_id!r} does not estimate the reported mean"
        )
    if match.statistic is not spec.statistic:
        _integrity_fail(
            f"ablation source {result.row_id!r} frozen MCSE spec/estimate "
            "statistic mismatch"
        )
    if match.spec_sha256 != spec.sha256:
        _integrity_fail(
            f"ablation source {result.row_id!r} frozen MCSE spec commitment mismatch"
        )
    if match.sample_count != len(result.samples):
        _integrity_fail(
            f"ablation source {result.row_id!r} frozen MCSE sample-count mismatch"
        )
    if not math.isclose(
        match.estimate,
        result.mean,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        _integrity_fail(
            f"ablation source {result.row_id!r} mean DistributionMCSEEstimate "
            "does not match the reported estimate"
        )
    return match


def derive_reporting_statistics(
    artifact: StatisticsDerivationArtifact,
    completed_results: tuple[ExperimentResult, ...],
    matrix: ExperimentMatrix,
    validity_certificate: ValidityCertificate,
    *,
    matrix_run_manifest_sha256: str,
) -> DerivedReportingStatistics:
    """Recompute all report cells and verify the artifact's output commitment."""

    if not isinstance(artifact, StatisticsDerivationArtifact):
        _schema_fail("artifact must be a StatisticsDerivationArtifact")
    if not isinstance(completed_results, tuple) or not completed_results:
        _schema_fail("completed_results must be a non-empty immutable tuple")
    _sha256(matrix_run_manifest_sha256, "matrix_run_manifest_sha256")
    results, kinds = _validated_sources(
        artifact, completed_results, matrix, matrix_run_manifest_sha256
    )
    certificate = _revalidated_validity_certificate(validity_certificate)
    if artifact.validity_certificate_sha256 != certificate.sha256:
        _integrity_fail("statistics derivation validity certificate hash mismatch")
    certificate_checks = {check.check_id: check for check in certificate.checks}
    risk_cells = tuple(
        cell for cell in artifact.cells if cell.section is ReportSection.RISK_COVERAGE
    )
    risk_source_results = _source_results(risk_cells[0], results, kinds)
    selective_evaluation = _risk_evaluation(
        risk_source_results[0],
        risk_source_results[1],
        artifact.selection_rule,
    )
    risk_points = {
        point.uncertainty_threshold: point for point in selective_evaluation.points
    }
    ablation_sources = tuple(
        results[source.row_id]
        for source in artifact.source_rows
        if source.source_kind is ReportingSourceKind.ABLATION_EFFECT_DRAWS
    )
    if len(ablation_sources) < 2:
        _integrity_fail(
            "ablation reporting requires a baseline and at least one comparable row"
        )
    for ablation_result in ablation_sources[1:]:
        _require_paired_draw_identity(
            ablation_sources[0],
            ablation_result,
            section="ablation-baseline",
        )
    rows_by_section: dict[ReportSection, list[dict[str, Scalar]]] = {
        section: [] for section in ReportSection
    }
    reference_counts: Counter[str] = Counter()
    semantic_keys: dict[ReportSection, set[tuple[Scalar, ...]]] = {
        section: set() for section in ReportSection
    }
    paired_expected: dict[str, set[float]] = {}

    for cell in artifact.cells:
        source_results = _source_results(cell, results, kinds)
        arguments = cell.argument_mapping
        reference_counts.update(cell.source_row_ids)
        output: dict[str, Scalar]
        semantic_key: tuple[Scalar, ...]

        if cell.section is ReportSection.PAIRED_DISTRIBUTION:
            difference = cast(float, arguments["difference"])
            counts = Counter(source_results[0].samples)
            paired_expected.setdefault(source_results[0].row_id, set()).add(difference)
            output = {
                "difference": difference,
                "density": counts[difference] / len(source_results[0].samples),
            }
            semantic_key = (difference,)
        elif cell.section is ReportSection.PREDICTED_VS_EMPIRICAL:
            output = {
                "label": cast(str, arguments["label"]),
                "predicted": source_results[0].mean,
                "empirical": source_results[1].mean,
            }
            semantic_key = (cast(str, arguments["label"]),)
        elif cell.section is ReportSection.COVERAGE_WIDTH:
            if len(source_results[0].samples) != len(source_results[1].samples):
                _integrity_fail("coverage and width sources have different draw counts")
            _require_paired_draw_identity(
                source_results[0],
                source_results[1],
                section="coverage-width",
            )
            _require_values(
                source_results[0].samples,
                field="coverage indicators",
                binary=True,
            )
            _require_values(
                source_results[1].samples, field="interval widths", minimum=0.0
            )
            output = {
                "label": cast(str, arguments["label"]),
                "registered_output_id": cast(str, arguments["registered_output_id"]),
                "metric_id": cast(str, arguments["metric_id"]),
                "cohort_id": cast(str, arguments["cohort_id"]),
                "horizon_id": cast(str, arguments["horizon_id"]),
                "interval_kind": cast(str, arguments["interval_kind"]),
                "interval_method_id": cast(str, arguments["interval_method_id"]),
                "estimand_id": cast(str, arguments["estimand_id"]),
                "sampling_unit": cast(str, arguments["sampling_unit"]),
                "valid_draw_count": len(source_results[0].samples),
                "level": cast(float, arguments["level"]),
                "coverage": _mean(source_results[0].samples),
                "width": _mean(source_results[1].samples),
            }
            semantic_key = tuple(
                arguments[field]
                for field in (
                    "registered_output_id",
                    "metric_id",
                    "cohort_id",
                    "horizon_id",
                    "interval_kind",
                    "interval_method_id",
                    "estimand_id",
                    "sampling_unit",
                    "level",
                )
            )
        elif cell.section is ReportSection.GROUP_BENEFIT_HARM:
            count = len(source_results[0].samples)
            output = {
                "group": cast(str, arguments["group"]),
                "cohort_basis": cast(str, arguments["cohort_basis"]),
                "cohort_definition_sha256": cast(
                    str, arguments["cohort_definition_sha256"]
                ),
                "benefit_probability": sum(
                    value > 0.0 for value in source_results[0].samples
                )
                / count,
                "harm_probability": sum(
                    value < 0.0 for value in source_results[0].samples
                )
                / count,
            }
            semantic_key = (cast(str, arguments["group"]),)
        elif cell.section is ReportSection.ABLATION_TABLE:
            if (
                arguments["method_id"] != "event_keyed_bootstrap_mcse"
                or arguments["uncertainty_kind"] != "monte_carlo_standard_error"
                or arguments["level"] is not None
            ):
                _schema_fail(
                    "typed assembler currently supports only event-keyed ablation MCSE"
                )
            distribution_estimate = _reported_mean_distribution_estimate(
                source_results[0],
                mcse_spec_id=cast(str, arguments["mcse_spec_id"]),
            )
            output = {
                "mechanism": cast(str, arguments["mechanism"]),
                "estimate": distribution_estimate.estimate,
                "uncertainty": distribution_estimate.mcse,
                "uncertainty_kind": cast(str, arguments["uncertainty_kind"]),
                "method_id": cast(str, arguments["method_id"]),
                "mcse_spec_id": cast(str, arguments["mcse_spec_id"]),
                "level": None,
                "sampling_unit": cast(str, arguments["sampling_unit"]),
                "valid_draw_count": len(source_results[0].samples),
                "estimand_id": cast(str, arguments["estimand_id"]),
            }
            semantic_key = (cast(str, arguments["mechanism"]),)
        elif cell.section is ReportSection.CALIBRATION_DIAGNOSTICS:
            output = {
                "statistic": cast(str, arguments["statistic"]),
                "observed": source_results[0].mean,
                "predicted": source_results[1].mean,
            }
            semantic_key = (cast(str, arguments["statistic"]),)
        elif cell.section is ReportSection.RISK_COVERAGE:
            threshold = cast(float, arguments["uncertainty_threshold"])
            point = risk_points.get(threshold)
            if point is None:  # Defensive: the artifact validates exact coverage.
                _integrity_fail("risk-coverage threshold is absent from selection rule")
            output = {
                "coverage": point.coverage,
                "risk": point.mean_loss,
            }
            semantic_key = (threshold,)
        elif cell.section is ReportSection.CERTIFICATE_HEATMAP:
            check_id = cast(str, arguments["row"])
            column = cast(str, arguments["column"])
            check = certificate_checks.get(check_id)
            if check is None or column not in _CERTIFICATE_COLUMNS:
                _integrity_fail("certificate heatmap cell is not canonical")
            output = {
                "row": check_id,
                "column": column,
                "value": 1.0 if check.status.value == column else 0.0,
            }
            semantic_key = (check_id, column)
        elif cell.section is ReportSection.COMPUTE_SCALING_TABLE:
            _require_values(
                source_results[0].samples,
                field="compute wall seconds",
                minimum=0.0,
            )
            row = matrix.row(source_results[0].row_id)
            output = {
                "workers": row.resources.cpu_cores,
                "events": len(source_results[0].event_keys),
                "wall_seconds": source_results[0].mean,
            }
            semantic_key = (
                row.resources.cpu_cores,
                len(source_results[0].event_keys),
            )
        else:  # pragma: no cover - exhaustive enum branch
            raise AssertionError(f"unsupported report section: {cell.section}")

        if semantic_key in semantic_keys[cell.section]:
            _schema_fail(f"{cell.section.value} contains a duplicate semantic cell")
        semantic_keys[cell.section].add(semantic_key)
        rows_by_section[cell.section].append(output)

    for source in artifact.source_rows:
        count = reference_counts[source.row_id]
        if source.source_kind is ReportingSourceKind.PAIRED_EFFECT_DRAWS:
            expected_differences = set(results[source.row_id].samples)
            if paired_expected.get(source.row_id, set()) != expected_differences:
                _integrity_fail(
                    f"paired distribution cells do not exhaust row {source.row_id!r}"
                )
            if count != len(expected_differences):
                _integrity_fail(
                    f"paired distribution row {source.row_id!r} has extra cells"
                )
        elif source.source_kind in {
            ReportingSourceKind.RISK_UNCERTAINTY_DRAWS,
            ReportingSourceKind.RISK_LOSS_DRAWS,
        }:
            if count != len(artifact.selection_rule.uncertainty_thresholds):
                _integrity_fail(
                    f"risk source row {source.row_id!r} must feed every frozen "
                    "threshold"
                )
        elif count != 1:
            _integrity_fail(
                f"typed source row {source.row_id!r} must feed exactly one cell"
            )

    derived = DerivedReportingStatistics(
        sections=tuple(
            (section.value, _freeze_rows(rows_by_section[section]))
            for section in ReportSection
        )
    )
    if derived.sha256 != artifact.statistics_sha256:
        _integrity_fail(
            "recomputed reporting statistics do not match the derivation commitment"
        )
    return derived


__all__ = [
    "DerivedReportingStatistics",
    "DerivationSourceRow",
    "REPORT_SECTIONS",
    "ReportSection",
    "ReportingSourceKind",
    "STATISTICS_ASSEMBLER_ID",
    "STATISTICS_ASSEMBLER_VERSION",
    "STATISTICS_DERIVATION_SCHEMA",
    "StatisticCellRecipe",
    "StatisticsDerivationArtifact",
    "StatisticsDerivationError",
    "StatisticsDerivationIntegrityError",
    "StatisticsDerivationSchemaError",
    "TYPED_REPORTING_RUNNER_KIND",
    "derive_reporting_statistics",
    "statistics_assembler_source_sha256",
]
