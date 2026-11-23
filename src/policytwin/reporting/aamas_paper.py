"""One-way paper artifacts from an audited synthetic AAMAS publication bundle.

This module is deliberately a renderer, not an analysis package.  It first
invokes :func:`audit_aamas_publication` with caller-retained commitments, then
loads only the frozen reporting artifacts, the misspecification artifact, and
the root timing/status records.  It never runs a simulator, fits a model,
chooses a specification, or substitutes a value for unavailable evidence.

The SVG and LaTeX outputs use only the Python standard library.  Every output
is marked ``SYNTHETIC`` and carries the publication plan SHA-256.  The absent
empirical comparison is retained as an explicit blocked manifest entry and no
predicted-versus-empirical figure is created.
"""

from __future__ import annotations

import html
import math
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NoReturn, Self, cast

from policytwin.experiments.aamas_misspecification import (
    MISSPECIFICATION_CLAIM_SCOPE,
    MISSPECIFICATION_SCHEMA_VERSION,
    MisspecificationCell,
    NegativeControlDecision,
)
from policytwin.experiments.aamas_publication import (
    AAMASPublicationAttempt,
    AAMASPublicationAttemptStatus,
    AAMASPublicationAuditReport,
    AAMASPublicationError,
    AAMASPublicationPlan,
)
from policytwin.experiments.aamas_publication import (
    audit_aamas_publication as _audit_aamas_publication,
)
from policytwin.provenance import (
    canonical_json_bytes,
    read_json_bytes_object,
    read_jsonl_bytes_objects,
    sha256_bytes,
)
from policytwin.reporting.aamas_integrated_branch import (
    AAMAS_INTEGRATED_BRANCH_REPORT_FILENAME,
    AAMASIntegratedBranchReportArtifact,
    IntegratedBranchDifferenceSummary,
)
from policytwin.reporting.aamas_submission import (
    AAMAS_SUBMISSION_FILENAME,
    AAMASSubmissionArtifact,
    ReportedScalar,
    RQ1RNGRow,
    RQ2ParameterSummary,
    RQ3RiskCoveragePoint,
    ScalarStatus,
    TypedInterval,
)

AAMAS_PAPER_SPEC_SCHEMA_VERSION = "policytwin.aamas-paper-spec.v2"
AAMAS_PAPER_MANIFEST_SCHEMA_VERSION = "policytwin.aamas-paper-manifest.v2"
AAMAS_PAPER_SEAL_SCHEMA_VERSION = "policytwin.aamas-paper-seal.v2"
AAMAS_PAPER_ARTIFACT_LABEL: Literal["SYNTHETIC"] = "SYNTHETIC"

_CALIBRATION_PRIOR_SUPPORTS = (
    ("liquidity", 0.5, 2.0),
    ("signal_strength", -1.0, 1.0),
)
_INTEGRATED_PRIMARY_METRICS = ("quoted_spread",)
_INTEGRATED_METRIC_PRESENTATION = (
    ("cash", "cash_units", "positive_means_more_group_cash_not_welfare"),
    (
        "filled_quantity",
        "asset_quantity",
        "positive_means_more_executed_quantity",
    ),
    ("inventory", "asset_quantity", "signed_inventory_change_no_normative_order"),
    (
        "market_trade_volume",
        "asset_quantity",
        "positive_means_more_market_volume",
    ),
    (
        "mean_buy_probability",
        "probability",
        "positive_means_higher_mean_buy_probability",
    ),
    (
        "mean_turnover_probability",
        "probability",
        "positive_means_higher_mean_turnover_probability",
    ),
    ("memory_entries", "entry_count", "positive_means_more_memory_entries"),
    (
        "net_worth",
        "cash_units",
        "positive_means_higher_group_net_worth_not_utility",
    ),
    (
        "quoted_spread",
        "price_units",
        "negative_means_narrower_candidate_quoted_spread",
    ),
    (
        "social_belief",
        "belief_score",
        "positive_means_higher_synthetic_belief_score",
    ),
    ("submitted_orders", "order_count", "positive_means_more_submitted_orders"),
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FIGURE_FILES = (
    "rng-variance-mcse.svg",
    "calibration-recovery-coverage.svg",
    "interval-coverage-width.svg",
    "risk-coverage.svg",
    "integrated-branch-effect-distribution.svg",
    "misspec-ppc-failure.svg",
    "failure-certificate-heatmap.svg",
)
_TABLE_FILES = (
    "main-results.tex",
    "ablation.tex",
    "evidence-status.tex",
    "compute.tex",
    "results-macros.tex",
)
_CONTROL_FILES = ("paper-spec.json", "artifact_manifest.json", "seal.json")
_EXPECTED_FILES = frozenset((*_FIGURE_FILES, *_TABLE_FILES, *_CONTROL_FILES))

_SCIENTIFIC_OUTPUT_REGISTRY = (
    ("paired-effect-distributions", "generated_synthetic"),
    ("predicted-vs-empirical-effect-plot", "blocked_no_empirical_artifact"),
    ("interval-coverage-width", "generated_synthetic"),
    (
        "group-benefit-harm-distributions",
        "blocked_no_utility_equivalent_group_artifact",
    ),
    ("ablation-table", "generated_synthetic"),
    ("calibration-diagnostics", "generated_synthetic"),
    ("risk-coverage", "generated_synthetic"),
    ("failure-certificate-heatmap", "generated_synthetic"),
    ("compute-scaling-table", "blocked_no_multi_scale_compute_artifact"),
)
_BLOCKED_OUTPUT_REASONS = {
    "predicted-vs-empirical-effect-plot": (
        "No validated empirical effect artifact is present in the audited publication "
        "bundle."
    ),
    "group-benefit-harm-distributions": (
        "The audited bundle has no utility-equivalent, welfare-authorized group "
        "benefit/harm artifact; latent-strategy outcome cells cannot substitute for it."
    ),
    "compute-scaling-table": (
        "The audited bundle contains one observed component-runtime record per "
        "component, not a frozen multi-scale compute experiment."
    ),
}

_SOURCE_PATHS = {
    "core-reporting": f"core-reporting/{AAMAS_SUBMISSION_FILENAME}",
    "integrated-branch-reporting": (
        f"integrated-branch-reporting/{AAMAS_INTEGRATED_BRANCH_REPORT_FILENAME}"
    ),
    "misspecification": "misspecification/artifact.json",
    "root-result": "result.json",
    "root-attempt-registry": "attempt-registry.json",
    "root-output-status": "output-registry.json",
    "root-timing": "timing.jsonl",
    "validity-certificate": "validity-certificate.json",
}
_SOURCE_ROLES = {
    "core-reporting": "validated_frozen_core_reporting",
    "integrated-branch-reporting": "validated_frozen_integrated_branch_reporting",
    "misspecification": "validated_structural_misspecification_negative_control",
    "root-result": "validated_root_scientific_status",
    "root-attempt-registry": "validated_root_terminal_attempt_registry",
    "root-output-status": "validated_root_output_status_registry",
    "root-timing": "validated_root_operational_timing",
    "validity-certificate": "validated_self_reported_synthetic_validity_certificate",
}


class AAMASPaperError(RuntimeError):
    """Base error for an invalid paper spec or rendering request."""


class AAMASPaperIntegrityError(AAMASPaperError):
    """Raised when an input or rendered byte differs from its commitment."""


def _fail(message: str) -> NoReturn:
    raise AAMASPaperError(message)


def _integrity_fail(message: str) -> NoReturn:
    raise AAMASPaperIntegrityError(message)


def _strict_fields(
    value: Mapping[str, object], expected: frozenset[str], context: str
) -> None:
    observed = frozenset(value)
    if observed != expected:
        _integrity_fail(
            f"{context} fields differ: missing={sorted(expected - observed)!r}, "
            f"extra={sorted(observed - expected)!r}"
        )


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        _integrity_fail(f"{field} must be an object with string keys")
    return cast(Mapping[str, object], value)


def _sequence(value: object, field: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        _integrity_fail(f"{field} must be an array")
    return cast(Sequence[object], value)


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        _integrity_fail(f"{field} must be a non-empty string")
    return value


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _integrity_fail(f"{field} must be a lowercase SHA-256")
    return value


def _integer(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _integrity_fail(f"{field} must be an integer >= {minimum}")
    return value


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        _integrity_fail(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        _integrity_fail(f"{field} must be finite")
    return result


def _optional_fraction(value: object, field: str) -> float | None:
    if value is None:
        return None
    result = _finite(value, field)
    if not 0.0 <= result <= 1.0:
        _integrity_fail(f"{field} must lie in [0, 1]")
    return result


def _canonical_file_bytes(value: object) -> bytes:
    return canonical_json_bytes(value) + b"\n"


@dataclass(frozen=True, slots=True)
class FrozenAAMASPaperSpec:
    """Immutable paper rendering definitions fixed independently of results."""

    renderer_id: str = "policytwin-stdlib-svg-latex-v2"
    numeric_format: str = ".6g"
    svg_width: int = 960
    svg_minimum_height: int = 360
    difference_definition: str = "candidate_minus_baseline"
    rng_panel_semantics: str = (
        "variance_in_paired_outcome_units_squared_and_mcse_in_paired_outcome_units_"
        "on_separate_independently_scaled_panels"
    )
    rng_macro_aggregation: str = (
        "arithmetic_mean_over_all_complete_frozen_stress_cells_by_rng_mode"
    )
    calibration_error_unit: str = "fraction_of_frozen_prior_support_width"
    calibration_prior_supports: tuple[tuple[str, float, float], ...] = (
        _CALIBRATION_PRIOR_SUPPORTS
    )
    integrated_primary_metrics: tuple[str, ...] = _INTEGRATED_PRIMARY_METRICS
    integrated_metric_presentation: tuple[tuple[str, str, str], ...] = (
        _INTEGRATED_METRIC_PRESENTATION
    )
    scientific_output_registry: tuple[tuple[str, str], ...] = (
        _SCIENTIFIC_OUTPUT_REGISTRY
    )
    integrated_quantile_interpretation: str = (
        "descriptive_finite_synthetic_replicate_quantiles_not_confidence_intervals"
    )
    recovery_interval_interpretation: str = "binomial_wilson_interval"
    risk_interval_interpretation: str = "synthetic_target_resampling_interval"
    empirical_comparison_status: str = "blocked_no_empirical_artifact"
    artifact_label: Literal["SYNTHETIC"] = AAMAS_PAPER_ARTIFACT_LABEL
    schema_version: str = AAMAS_PAPER_SPEC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        expected: tuple[tuple[object, object], ...] = (
            (self.renderer_id, "policytwin-stdlib-svg-latex-v2"),
            (self.numeric_format, ".6g"),
            (self.svg_width, 960),
            (self.svg_minimum_height, 360),
            (self.difference_definition, "candidate_minus_baseline"),
            (
                self.rng_panel_semantics,
                "variance_in_paired_outcome_units_squared_and_mcse_in_paired_"
                "outcome_units_on_separate_independently_scaled_panels",
            ),
            (
                self.rng_macro_aggregation,
                "arithmetic_mean_over_all_complete_frozen_stress_cells_by_rng_mode",
            ),
            (
                self.calibration_error_unit,
                "fraction_of_frozen_prior_support_width",
            ),
            (self.calibration_prior_supports, _CALIBRATION_PRIOR_SUPPORTS),
            (self.integrated_primary_metrics, _INTEGRATED_PRIMARY_METRICS),
            (
                self.integrated_metric_presentation,
                _INTEGRATED_METRIC_PRESENTATION,
            ),
            (self.scientific_output_registry, _SCIENTIFIC_OUTPUT_REGISTRY),
            (
                self.integrated_quantile_interpretation,
                "descriptive_finite_synthetic_replicate_quantiles_"
                "not_confidence_intervals",
            ),
            (self.recovery_interval_interpretation, "binomial_wilson_interval"),
            (
                self.risk_interval_interpretation,
                "synthetic_target_resampling_interval",
            ),
            (
                self.empirical_comparison_status,
                "blocked_no_empirical_artifact",
            ),
            (self.artifact_label, AAMAS_PAPER_ARTIFACT_LABEL),
            (self.schema_version, AAMAS_PAPER_SPEC_SCHEMA_VERSION),
        )
        if any(observed != required for observed, required in expected):
            _fail("AAMAS paper rendering definitions are frozen by schema v2")

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def to_mapping(self) -> dict[str, object]:
        return {
            "artifact_label": self.artifact_label,
            "calibration_error_unit": self.calibration_error_unit,
            "calibration_prior_supports": [
                {"lower": lower, "name": name, "upper": upper}
                for name, lower, upper in self.calibration_prior_supports
            ],
            "difference_definition": self.difference_definition,
            "empirical_comparison_status": self.empirical_comparison_status,
            "figure_registry": list(_FIGURE_FILES),
            "integrated_metric_presentation": [
                {"direction": direction, "metric": metric, "unit": unit}
                for metric, unit, direction in self.integrated_metric_presentation
            ],
            "integrated_primary_metrics": list(self.integrated_primary_metrics),
            "integrated_quantile_interpretation": (
                self.integrated_quantile_interpretation
            ),
            "numeric_format": self.numeric_format,
            "recovery_interval_interpretation": (self.recovery_interval_interpretation),
            "renderer_id": self.renderer_id,
            "rng_macro_aggregation": self.rng_macro_aggregation,
            "rng_panel_semantics": self.rng_panel_semantics,
            "risk_interval_interpretation": self.risk_interval_interpretation,
            "scientific_output_registry": [
                {"output_id": output_id, "status": status}
                for output_id, status in self.scientific_output_registry
            ],
            "schema_version": self.schema_version,
            "svg_minimum_height": self.svg_minimum_height,
            "svg_width": self.svg_width,
            "table_registry": list(_TABLE_FILES),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(
            value,
            frozenset(
                {
                    "artifact_label",
                    "calibration_error_unit",
                    "calibration_prior_supports",
                    "difference_definition",
                    "empirical_comparison_status",
                    "figure_registry",
                    "integrated_metric_presentation",
                    "integrated_primary_metrics",
                    "integrated_quantile_interpretation",
                    "numeric_format",
                    "recovery_interval_interpretation",
                    "renderer_id",
                    "rng_macro_aggregation",
                    "rng_panel_semantics",
                    "risk_interval_interpretation",
                    "scientific_output_registry",
                    "schema_version",
                    "svg_minimum_height",
                    "svg_width",
                    "table_registry",
                }
            ),
            "paper spec",
        )
        if value["figure_registry"] != list(_FIGURE_FILES) or value[
            "table_registry"
        ] != list(_TABLE_FILES):
            _integrity_fail("paper output registry changed")
        supports: list[tuple[str, float, float]] = []
        for raw in _sequence(
            value["calibration_prior_supports"], "calibration_prior_supports"
        ):
            row = _mapping(raw, "calibration prior support")
            _strict_fields(
                row,
                frozenset({"lower", "name", "upper"}),
                "calibration prior support",
            )
            supports.append(
                (
                    _string(row["name"], "calibration prior name"),
                    _finite(row["lower"], "calibration prior lower"),
                    _finite(row["upper"], "calibration prior upper"),
                )
            )
        metric_presentation: list[tuple[str, str, str]] = []
        for raw in _sequence(
            value["integrated_metric_presentation"],
            "integrated_metric_presentation",
        ):
            row = _mapping(raw, "integrated metric presentation")
            _strict_fields(
                row,
                frozenset({"direction", "metric", "unit"}),
                "integrated metric presentation",
            )
            metric_presentation.append(
                (
                    _string(row["metric"], "integrated metric"),
                    _string(row["unit"], "integrated metric unit"),
                    _string(row["direction"], "integrated metric direction"),
                )
            )
        scientific_outputs: list[tuple[str, str]] = []
        for raw in _sequence(
            value["scientific_output_registry"], "scientific_output_registry"
        ):
            row = _mapping(raw, "scientific output registry row")
            _strict_fields(
                row,
                frozenset({"output_id", "status"}),
                "scientific output registry row",
            )
            scientific_outputs.append(
                (
                    _string(row["output_id"], "scientific output_id"),
                    _string(row["status"], "scientific output status"),
                )
            )
        result = cls(
            renderer_id=_string(value["renderer_id"], "renderer_id"),
            numeric_format=_string(value["numeric_format"], "numeric_format"),
            svg_width=_integer(value["svg_width"], "svg_width", minimum=1),
            svg_minimum_height=_integer(
                value["svg_minimum_height"], "svg_minimum_height", minimum=1
            ),
            difference_definition=_string(
                value["difference_definition"], "difference_definition"
            ),
            rng_panel_semantics=_string(
                value["rng_panel_semantics"], "rng_panel_semantics"
            ),
            rng_macro_aggregation=_string(
                value["rng_macro_aggregation"], "rng_macro_aggregation"
            ),
            calibration_error_unit=_string(
                value["calibration_error_unit"], "calibration_error_unit"
            ),
            calibration_prior_supports=tuple(supports),
            integrated_primary_metrics=tuple(
                _string(item, "integrated primary metric")
                for item in _sequence(
                    value["integrated_primary_metrics"],
                    "integrated_primary_metrics",
                )
            ),
            integrated_metric_presentation=tuple(metric_presentation),
            scientific_output_registry=tuple(scientific_outputs),
            integrated_quantile_interpretation=_string(
                value["integrated_quantile_interpretation"],
                "integrated_quantile_interpretation",
            ),
            recovery_interval_interpretation=_string(
                value["recovery_interval_interpretation"],
                "recovery_interval_interpretation",
            ),
            risk_interval_interpretation=_string(
                value["risk_interval_interpretation"],
                "risk_interval_interpretation",
            ),
            empirical_comparison_status=_string(
                value["empirical_comparison_status"],
                "empirical_comparison_status",
            ),
            artifact_label=cast(Literal["SYNTHETIC"], value["artifact_label"]),
            schema_version=_string(value["schema_version"], "schema_version"),
        )
        if result.to_mapping() != dict(value):
            _integrity_fail("paper spec is not in canonical form")
        return result


@dataclass(frozen=True, slots=True)
class _MisspecificationAggregate:
    cell: MisspecificationCell
    expected_target_attempts: int
    completed_target_attempts: int
    failed_target_attempt_ids: tuple[str, ...]
    held_out_statistic_count: int
    held_out_ppc_failures: int | None
    held_out_ppc_failure_rate: float | None
    normalized_recovery_loss: float | None
    boundary_rate: float | None
    multimodality_rate: float | None

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(value, frozenset(cls.__annotations__), "misspec aggregate")
        try:
            cell = MisspecificationCell(_string(value["cell"], "cell"))
        except ValueError as error:
            raise AAMASPaperIntegrityError("unknown misspecification cell") from error
        failed = tuple(
            _string(item, "failed_target_attempt_id")
            for item in _sequence(
                value["failed_target_attempt_ids"], "failed_target_attempt_ids"
            )
        )
        raw_failures = value["held_out_ppc_failures"]
        failures = (
            None
            if raw_failures is None
            else _integer(raw_failures, "held_out_ppc_failures")
        )
        result = cls(
            cell=cell,
            expected_target_attempts=_integer(
                value["expected_target_attempts"],
                "expected_target_attempts",
                minimum=1,
            ),
            completed_target_attempts=_integer(
                value["completed_target_attempts"], "completed_target_attempts"
            ),
            failed_target_attempt_ids=failed,
            held_out_statistic_count=_integer(
                value["held_out_statistic_count"],
                "held_out_statistic_count",
                minimum=1,
            ),
            held_out_ppc_failures=failures,
            held_out_ppc_failure_rate=_optional_fraction(
                value["held_out_ppc_failure_rate"], "held_out_ppc_failure_rate"
            ),
            normalized_recovery_loss=_optional_fraction(
                value["normalized_recovery_loss"], "normalized_recovery_loss"
            ),
            boundary_rate=_optional_fraction(value["boundary_rate"], "boundary_rate"),
            multimodality_rate=_optional_fraction(
                value["multimodality_rate"], "multimodality_rate"
            ),
        )
        if result.completed_target_attempts > result.expected_target_attempts:
            _integrity_fail("misspecification completion exceeds frozen denominator")
        metrics = (
            result.held_out_ppc_failures,
            result.held_out_ppc_failure_rate,
            result.normalized_recovery_loss,
            result.boundary_rate,
            result.multimodality_rate,
        )
        if failed:
            if any(item is not None for item in metrics):
                _integrity_fail("failed misspecification aggregate retained metrics")
        elif result.completed_target_attempts != result.expected_target_attempts or any(
            item is None for item in metrics
        ):
            _integrity_fail("complete misspecification aggregate omitted metrics")
        return result


@dataclass(frozen=True, slots=True)
class _MisspecificationView:
    plan_sha256: str
    completed_count: int
    failed_count: int
    decision: NegativeControlDecision
    held_out_ppc_failure_gap: float | None
    target_aggregates: tuple[_MisspecificationAggregate, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(
            value,
            frozenset(
                {
                    "assessment",
                    "attempts",
                    "claim_scope",
                    "completed_count",
                    "failed_count",
                    "plan_sha256",
                    "schema_version",
                    "synthetic_only",
                    "target_aggregates",
                    "validity_certificate",
                }
            ),
            "misspecification artifact",
        )
        if (
            value["claim_scope"] != MISSPECIFICATION_CLAIM_SCOPE
            or value["schema_version"] != MISSPECIFICATION_SCHEMA_VERSION
            or value["synthetic_only"] is not True
        ):
            _integrity_fail("misspecification artifact authority changed")
        assessment = _mapping(value["assessment"], "misspecification assessment")
        _strict_fields(
            assessment,
            frozenset(
                {
                    "boundary_threshold_exceeded",
                    "decision",
                    "held_out_ppc_failure_gap",
                    "multimodality_threshold_exceeded",
                    "ppc_detection",
                    "recovery_threshold_exceeded",
                }
            ),
            "misspecification assessment",
        )
        try:
            decision = NegativeControlDecision(
                _string(assessment["decision"], "negative-control decision")
            )
        except ValueError as error:
            raise AAMASPaperIntegrityError(
                "unknown negative-control decision"
            ) from error
        gap = assessment["held_out_ppc_failure_gap"]
        aggregates = tuple(
            _MisspecificationAggregate.from_mapping(
                _mapping(item, "misspecification aggregate")
            )
            for item in _sequence(value["target_aggregates"], "target_aggregates")
        )
        if tuple(item.cell for item in aggregates) != tuple(MisspecificationCell):
            _integrity_fail("misspecification cells are missing or reordered")
        attempt_count = len(_sequence(value["attempts"], "attempts"))
        completed = _integer(value["completed_count"], "completed_count")
        failed = _integer(value["failed_count"], "failed_count")
        if completed + failed != attempt_count:
            _integrity_fail("misspecification terminal counts are inconsistent")
        return cls(
            plan_sha256=_sha256(value["plan_sha256"], "misspecification plan_sha256"),
            completed_count=completed,
            failed_count=failed,
            decision=decision,
            held_out_ppc_failure_gap=(
                None if gap is None else _finite(gap, "held_out_ppc_failure_gap")
            ),
            target_aggregates=aggregates,
        )


@dataclass(frozen=True, slots=True)
class _RootStatus:
    submission_evidence_complete: bool
    scientifically_complete: bool
    publication_eligible: bool
    validity_level: str
    output_statuses: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class _ValidityView:
    level: Literal["abstain"]
    diagnostic_level: str
    checks: tuple[tuple[str, str], ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(
            value,
            frozenset(
                {
                    "checks",
                    "diagnostic_level",
                    "evidence",
                    "evidence_authority",
                    "evidence_sha256",
                    "level",
                    "schema_version",
                    "synthetic_only",
                    "thresholds",
                    "thresholds_sha256",
                }
            ),
            "validity certificate",
        )
        if (
            value["level"] != "abstain"
            or value["synthetic_only"] is not True
            or value["schema_version"] != 3
        ):
            _integrity_fail("validity certificate broadened synthetic authority")
        checks: list[tuple[str, str]] = []
        for raw in _sequence(value["checks"], "validity checks"):
            row = _mapping(raw, "validity check")
            _strict_fields(
                row,
                frozenset({"check_id", "observed", "reason", "status"}),
                "validity check",
            )
            check_id = _string(row["check_id"], "validity check_id")
            status_value = _string(row["status"], "validity check status")
            if status_value not in {"pass", "warn", "fail", "unknown"}:
                _integrity_fail("validity check has an unknown status")
            checks.append((check_id, status_value))
        if not checks or len(checks) != len({item[0] for item in checks}):
            _integrity_fail("validity check registry is empty or duplicated")
        return cls(
            level="abstain",
            diagnostic_level=_string(
                value["diagnostic_level"], "validity diagnostic_level"
            ),
            checks=tuple(checks),
        )


@dataclass(frozen=True, slots=True)
class _TimingRow:
    component_id: str
    wall_seconds: float


@dataclass(frozen=True, slots=True)
class AAMASPaperSourceRecord:
    source_id: str
    path: str
    role: str
    sha256: str
    size_bytes: int
    status: Literal["validated_by_audit_aamas_publication"] = (
        "validated_by_audit_aamas_publication"
    )

    def __post_init__(self) -> None:
        if self.source_id not in _SOURCE_PATHS:
            _fail("paper source identifier is not registered")
        if self.path != _SOURCE_PATHS[self.source_id]:
            _fail("paper source path differs from its fixed registry")
        if self.role != _SOURCE_ROLES[self.source_id]:
            _fail("paper source role differs from its fixed registry")
        _sha256(self.sha256, "paper source sha256")
        _integer(self.size_bytes, "paper source size_bytes")
        if self.status != "validated_by_audit_aamas_publication":
            _fail("paper source status cannot bypass the publication audit")

    def to_mapping(self) -> dict[str, object]:
        return {
            "path": self.path,
            "role": self.role,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "source_id": self.source_id,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class _LoadedSources:
    publication_audit: AAMASPublicationAuditReport
    core: AAMASSubmissionArtifact
    branch: AAMASIntegratedBranchReportArtifact
    misspecification: _MisspecificationView
    root_status: _RootStatus
    attempts: tuple[AAMASPublicationAttempt, ...]
    validity: _ValidityView
    timing: tuple[_TimingRow, ...]
    records: tuple[AAMASPaperSourceRecord, ...]


def _stable_regular_bytes(path: Path) -> bytes:
    try:
        before = path.lstat()
    except OSError as error:
        raise AAMASPaperIntegrityError(f"cannot stat source {path}: {error}") from error
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_nlink != 1
    ):
        _integrity_fail(f"source is not a singly linked regular file: {path}")
    try:
        payload = path.read_bytes()
        after = path.lstat()
    except OSError as error:
        raise AAMASPaperIntegrityError(f"cannot read source {path}: {error}") from error

    def identity(item: os.stat_result) -> tuple[int, int, int, int, int, int]:
        return (
            item.st_dev,
            item.st_ino,
            item.st_mode,
            item.st_nlink,
            item.st_size,
            item.st_mtime_ns,
        )

    if identity(before) != identity(after) or len(payload) != after.st_size:
        _integrity_fail(f"source changed while being read: {path}")
    return payload


def _parse_root_status(
    result: Mapping[str, object],
    output_registry: Mapping[str, object],
    *,
    plan_sha256: str,
) -> _RootStatus:
    _strict_fields(
        result,
        frozenset(
            {
                "claim_scope",
                "components",
                "failed_or_blocked_attempts",
                "failure_registry_sha256",
                "integrated_branch_reporting_status",
                "output_registry_sha256",
                "plan_sha256",
                "publication_eligible",
                "schema_version",
                "scientifically_complete",
                "submission_evidence_complete",
                "synthetic_only",
                "validity_level",
            }
        ),
        "publication result",
    )
    if (
        result["plan_sha256"] != plan_sha256
        or result["synthetic_only"] is not True
        or result["publication_eligible"] is not False
        or result["validity_level"] != "abstain"
    ):
        _integrity_fail("root publication status broadened its authority")
    _strict_fields(
        output_registry,
        frozenset({"entries", "failure_propagation", "plan_sha256", "schema_version"}),
        "root output registry",
    )
    if (
        output_registry["plan_sha256"] != plan_sha256
        or output_registry["failure_propagation"]
        != "fail_closed_no_complete_case_or_missing_output"
    ):
        _integrity_fail("root output registry changed authority")
    rows: list[tuple[str, str]] = []
    for raw in _sequence(output_registry["entries"], "root output entries"):
        row = _mapping(raw, "root output row")
        _strict_fields(
            row,
            frozenset({"artifact_sha256", "definition", "output_id", "status"}),
            "root output row",
        )
        output_id = _string(row["output_id"], "output_id")
        status_value = _string(row["status"], "output status")
        digest = row["artifact_sha256"]
        if digest is not None:
            _sha256(digest, "output artifact_sha256")
        rows.append((output_id, status_value))
    if len(rows) != len({output_id for output_id, _status in rows}):
        _integrity_fail("root output registry contains duplicate output identifiers")
    submission_complete = result["submission_evidence_complete"]
    scientific_complete = result["scientifically_complete"]
    if not isinstance(submission_complete, bool) or not isinstance(
        scientific_complete, bool
    ):
        _integrity_fail("root completion flags must be boolean")
    return _RootStatus(
        submission_evidence_complete=submission_complete,
        scientifically_complete=scientific_complete,
        publication_eligible=False,
        validity_level="abstain",
        output_statuses=tuple(rows),
    )


def _parse_timing(
    rows: Sequence[Mapping[str, object]], *, plan_sha256: str
) -> tuple[_TimingRow, ...]:
    result: list[_TimingRow] = []
    for row in rows:
        _strict_fields(
            row,
            frozenset(
                {
                    "component_id",
                    "event",
                    "event_key",
                    "measurement_scope",
                    "schema_version",
                    "wall_seconds",
                }
            ),
            "root timing row",
        )
        component_id = _string(row["component_id"], "timing component_id")
        if (
            row["event"] != "observed"
            or row["measurement_scope"] != "operational_non_scientific_process"
        ):
            _integrity_fail("timing semantics changed")
        expected_key = sha256_bytes(
            canonical_json_bytes(
                {
                    "component_id": component_id,
                    "plan_sha256": plan_sha256,
                    "purpose": "operational_wall_time",
                }
            )
        )
        if row["event_key"] != expected_key:
            _integrity_fail("timing EventKey changed")
        wall = _finite(row["wall_seconds"], "wall_seconds")
        if wall < 0.0:
            _integrity_fail("timing duration cannot be negative")
        result.append(_TimingRow(component_id, wall))
    if not result or len({item.component_id for item in result}) != len(result):
        _integrity_fail("timing registry is empty or duplicated")
    return tuple(result)


def _parse_attempt_registry(
    value: Mapping[str, object], *, plan: AAMASPublicationPlan
) -> tuple[AAMASPublicationAttempt, ...]:
    _strict_fields(
        value,
        frozenset(
            {
                "attempts",
                "definitions",
                "failure_propagation",
                "plan_sha256",
                "schema_version",
            }
        ),
        "root attempt registry",
    )
    frozen_definitions = plan.to_mapping()["attempt_registry"]
    if (
        value["definitions"] != frozen_definitions
        or value["failure_propagation"]
        != "every_expected_attempt_has_one_terminal_state"
        or value["plan_sha256"] != plan.sha256
        or value["schema_version"] != 1
    ):
        _integrity_fail("root attempt registry changed its frozen authority")
    attempts = tuple(
        AAMASPublicationAttempt.from_mapping(_mapping(raw, "root attempt"))
        for raw in _sequence(value["attempts"], "root attempts")
    )
    definition_rows = tuple(
        _mapping(raw, "root attempt definition")
        for raw in _sequence(frozen_definitions, "frozen attempt definitions")
    )
    expected_ids = tuple(
        _string(row["attempt_id"], "frozen attempt_id") for row in definition_rows
    )
    if tuple(item.attempt_id for item in attempts) != expected_ids:
        _integrity_fail("root attempt rows are missing, extra, or reordered")
    if [item.to_mapping() for item in attempts] != value["attempts"]:
        _integrity_fail("root attempt rows are not canonical")
    return attempts


def _load_sources(
    publication_directory: Path,
    *,
    expected_plan: AAMASPublicationPlan,
    expected_plan_sha256: str,
    expected_study_commitment_sha256: str,
    expected_root_seal_sha256: str,
    project_root: Path | None,
    allow_formal_scale: bool,
) -> _LoadedSources:
    def audit_source(directory: Path) -> AAMASPublicationAuditReport:
        try:
            return _audit_aamas_publication(
                directory,
                expected_plan=expected_plan,
                expected_plan_sha256=expected_plan_sha256,
                expected_study_commitment_sha256=expected_study_commitment_sha256,
                expected_root_seal_sha256=expected_root_seal_sha256,
                project_root=project_root,
                allow_formal_scale=allow_formal_scale,
            )
        except AAMASPublicationError as error:
            raise AAMASPaperIntegrityError(
                f"publication source audit failed: {error}"
            ) from error

    live_audit = audit_source(publication_directory)
    if live_audit.plan_sha256 != expected_plan.sha256:
        _integrity_fail("publication audit returned an unexpected plan")

    # Never re-read the audited live tree.  A private copy is independently
    # re-audited against the same three caller-held commitments before and
    # after the allowed bytes are parsed.  Preserving symlinks makes an
    # adversarial type change visible to the fixed-layout publication audit.
    with tempfile.TemporaryDirectory(prefix="policytwin-aamas-paper-source-") as tmp:
        snapshot = Path(tmp) / "publication"
        try:
            shutil.copytree(publication_directory, snapshot, symlinks=True)
        except OSError as error:
            raise AAMASPaperIntegrityError(
                f"cannot create private publication snapshot: {error}"
            ) from error
        snapshot_audit_before = audit_source(snapshot)
        if snapshot_audit_before != live_audit:
            _integrity_fail(
                "private publication snapshot audit differs from live audit receipt"
            )
        payloads: dict[str, bytes] = {}
        records: list[AAMASPaperSourceRecord] = []
        for source_id, relative in _SOURCE_PATHS.items():
            payload = _stable_regular_bytes(snapshot / relative)
            payloads[source_id] = payload
            records.append(
                AAMASPaperSourceRecord(
                    source_id=source_id,
                    path=relative,
                    role=_SOURCE_ROLES[source_id],
                    sha256=sha256_bytes(payload),
                    size_bytes=len(payload),
                )
            )
        core = AAMASSubmissionArtifact.from_mapping(
            read_json_bytes_object(payloads["core-reporting"], source="core reporting")
        )
        branch = AAMASIntegratedBranchReportArtifact.from_mapping(
            read_json_bytes_object(
                payloads["integrated-branch-reporting"],
                source="integrated branch reporting",
            )
        )
        misspecification = _MisspecificationView.from_mapping(
            read_json_bytes_object(
                payloads["misspecification"], source="misspecification artifact"
            )
        )
        root_result = read_json_bytes_object(
            payloads["root-result"], source="root result"
        )
        root_output = read_json_bytes_object(
            payloads["root-output-status"], source="root output status"
        )
        root_status = _parse_root_status(
            root_result, root_output, plan_sha256=expected_plan.sha256
        )
        attempts = _parse_attempt_registry(
            read_json_bytes_object(
                payloads["root-attempt-registry"], source="root attempt registry"
            ),
            plan=expected_plan,
        )
        validity = _ValidityView.from_mapping(
            read_json_bytes_object(
                payloads["validity-certificate"], source="validity certificate"
            )
        )
        timing = _parse_timing(
            read_jsonl_bytes_objects(payloads["root-timing"], source="root timing"),
            plan_sha256=expected_plan.sha256,
        )
        if core.reporting_spec != expected_plan.reporting_spec:
            _integrity_fail("core reporting spec differs from publication plan")
        if branch.reporting_spec != expected_plan.integrated_branch_reporting_spec:
            _integrity_fail("branch reporting spec differs from publication plan")
        snapshot_audit_after = audit_source(snapshot)
        if snapshot_audit_after != snapshot_audit_before:
            _integrity_fail("private publication snapshot changed after source reads")
    return _LoadedSources(
        publication_audit=snapshot_audit_after,
        core=core,
        branch=branch,
        misspecification=misspecification,
        root_status=root_status,
        attempts=attempts,
        validity=validity,
        timing=timing,
        records=tuple(records),
    )


def _number(value: int | float, spec: FrozenAAMASPaperSpec) -> str:
    result = format(float(value), spec.numeric_format)
    return "0" if result in {"-0", "-0.0"} else result


def _scalar(value: ReportedScalar, spec: FrozenAAMASPaperSpec) -> str:
    if value.status is ScalarStatus.DEFINED:
        assert value.value is not None
        return _number(value.value, spec)
    return f"UNAVAILABLE[{value.status.value}]"


def _interval(value: TypedInterval, spec: FrozenAAMASPaperSpec) -> str:
    if value.status is ScalarStatus.DEFINED:
        assert value.lower is not None and value.upper is not None
        return f"[{_number(value.lower, spec)}, {_number(value.upper, spec)}]"
    return f"UNAVAILABLE[{value.status.value}]"


def _latex(value: object) -> str:
    result = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in result)


def _svg_document(
    *,
    title: str,
    subtitle: str,
    plan_sha256: str,
    source_ids: Sequence[str],
    height: int,
    body: Sequence[str],
    spec: FrozenAAMASPaperSpec,
) -> bytes:
    width = spec.svg_width
    safe_title = html.escape(f"SYNTHETIC — {title} — plan {plan_sha256}", quote=True)
    safe_subtitle = html.escape(subtitle, quote=True)
    metadata = html.escape(
        canonical_json_bytes(
            {
                "artifact_label": AAMAS_PAPER_ARTIFACT_LABEL,
                "plan_sha256": plan_sha256,
                "source_ids": list(source_ids),
            }
        ).decode("utf-8"),
        quote=False,
    )
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" '
            'role="img">'
        ),
        f"  <title>{safe_title}</title>",
        f"  <desc>{safe_subtitle}</desc>",
        f"  <metadata>{metadata}</metadata>",
        '  <rect width="100%" height="100%" fill="#ffffff"/>',
        '  <g font-family="Arial, Helvetica, sans-serif" fill="#17212b">',
        f'    <text x="24" y="30" font-size="17" font-weight="700">{safe_title}</text>',
        f'    <text x="24" y="51" font-size="12">{safe_subtitle}</text>',
        *body,
        (
            f'    <text x="24" y="{height - 15}" font-size="10" '
            'fill="#5c6773">SYNTHETIC; source-audited paper rendering; '
            f"plan {plan_sha256}</text>"
        ),
        "  </g>",
        "</svg>",
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def _defined_float(value: ReportedScalar) -> float | None:
    if value.status is not ScalarStatus.DEFINED:
        return None
    assert value.value is not None
    return float(value.value)


def _prior_support(
    parameter: str, spec: FrozenAAMASPaperSpec
) -> tuple[float, float, float]:
    matches = tuple(
        (lower, upper)
        for name, lower, upper in spec.calibration_prior_supports
        if name == parameter
    )
    if len(matches) != 1:
        _integrity_fail(f"parameter lacks one frozen prior support: {parameter}")
    lower, upper = matches[0]
    width = upper - lower
    if width <= 0.0:
        _integrity_fail(f"parameter prior support is not ordered: {parameter}")
    return lower, upper, width


def _normalized_mae(
    row: RQ2ParameterSummary, spec: FrozenAAMASPaperSpec
) -> tuple[float | None, str, float]:
    _lower, _upper, width = _prior_support(row.parameter, spec)
    raw = _defined_float(row.mae)
    if raw is None:
        return None, _scalar(row.mae, spec), width
    normalized = raw / width
    return normalized, _number(normalized, spec), width


def _normalized_parameter_scalar(
    value: ReportedScalar, *, parameter: str, spec: FrozenAAMASPaperSpec
) -> tuple[float | None, str, float]:
    _lower, _upper, width = _prior_support(parameter, spec)
    raw = _defined_float(value)
    if raw is None:
        return None, _scalar(value, spec), width
    normalized = raw / width
    return normalized, _number(normalized, spec), width


def _metric_presentation(
    metric_name: str, spec: FrozenAAMASPaperSpec
) -> tuple[str, str]:
    matches = tuple(
        (unit, direction)
        for metric, unit, direction in spec.integrated_metric_presentation
        if metric == metric_name
    )
    if len(matches) != 1:
        _integrity_fail(
            f"integrated metric lacks one frozen unit/direction: {metric_name}"
        )
    return matches[0]


def _rng_svg(
    rows: Sequence[RQ1RNGRow],
    *,
    plan_sha256: str,
    spec: FrozenAAMASPaperSpec,
) -> bytes:
    if not rows:
        _integrity_fail("core reporting omitted RNG rows")
    variance_values = tuple(
        item
        for row in rows
        if (item := _defined_float(row.sample_variance)) is not None
    )
    mcse_values = tuple(
        item for row in rows if (item := _defined_float(row.mcse)) is not None
    )
    variance_scale = max(variance_values, default=1.0) or 1.0
    mcse_scale = max(mcse_values, default=1.0) or 1.0
    height = max(spec.svg_minimum_height, 105 + 30 * len(rows))
    body = [
        (
            '    <g data-layout="separate-panels" '
            'data-variance-unit="paired_outcome_units_squared" '
            'data-mcse-unit="paired_outcome_units" '
            f'data-variance-scale-max="{_number(variance_scale, spec)}" '
            f'data-mcse-scale-max="{_number(mcse_scale, spec)}">'
        ),
        (
            '      <text x="300" y="82" font-size="11" font-weight="700">'
            'Panel A: variance (paired-outcome units²; ddof=1)</text>'
        ),
        (
            '      <text x="665" y="82" font-size="11" font-weight="700">'
            'Panel B: MCSE (paired-outcome units)</text>'
        ),
        '      <line x1="300" x2="545" y1="91" y2="91" stroke="#8c98a4"/>',
        '      <line x1="665" x2="910" y1="91" y2="91" stroke="#8c98a4"/>',
        "    </g>",
    ]
    for index, row in enumerate(rows):
        y = 118 + index * 30
        variance = _defined_float(row.sample_variance)
        mcse = _defined_float(row.mcse)
        variance_text = _scalar(row.sample_variance, spec)
        mcse_text = _scalar(row.mcse, spec)
        label = html.escape(f"{row.cell_id} / {row.mode}", quote=True)
        body.append(
            f'    <g data-cell-id="{html.escape(row.cell_id, quote=True)}" '
            f'data-mode="{html.escape(row.mode, quote=True)}" '
            f'data-variance="{html.escape(variance_text, quote=True)}" '
            f'data-mcse="{html.escape(mcse_text, quote=True)}">'
        )
        body.append(f'      <text x="24" y="{y + 11}" font-size="10">{label}</text>')
        body.append(
            f'      <line x1="300" x2="545" y1="{y + 8}" y2="{y + 8}" '
            'stroke="#d9dee5"/>'
        )
        body.append(
            f'      <line x1="665" x2="910" y1="{y + 8}" y2="{y + 8}" '
            'stroke="#d9dee5"/>'
        )
        if variance is not None:
            width = 225.0 * variance / variance_scale
            body.append(
                f'      <rect x="300" y="{y}" width="{width:.3f}" height="7" '
                'fill="#315efb"/>'
            )
        if mcse is not None:
            width = 225.0 * mcse / mcse_scale
            body.append(
                f'      <rect x="665" y="{y}" width="{width:.3f}" '
                'height="7" fill="#f08c46"/>'
            )
        body.append(
            f'      <text x="548" y="{y + 8}" font-size="8">'
            f"{html.escape(variance_text)}</text>"
        )
        body.append(
            f'      <text x="913" y="{y + 8}" font-size="8" text-anchor="end">'
            f"{html.escape(mcse_text)}</text>"
        )
        body.append("    </g>")
    return _svg_document(
        title=(
            "RNG variance (outcome units²) and MCSE (outcome units): "
            "separate panels"
        ),
        subtitle=(
            "Panels use independent scales because variance and MCSE have different "
            "units; MCSE is sqrt(variance / completed replicates)."
        ),
        plan_sha256=plan_sha256,
        source_ids=("core-reporting",),
        height=height,
        body=body,
        spec=spec,
    )


def _calibration_svg(
    rows: Sequence[RQ2ParameterSummary],
    *,
    plan_sha256: str,
    spec: FrozenAAMASPaperSpec,
) -> bytes:
    selected = tuple(row for row in rows if row.role == "target")
    if not selected:
        _integrity_fail("core reporting omitted target recovery summaries")
    mae_values = [
        item
        for row in selected
        if (item := _normalized_mae(row, spec)[0]) is not None
    ]
    mae_scale = max(mae_values, default=1.0)
    if mae_scale == 0.0:
        mae_scale = 1.0
    height = max(spec.svg_minimum_height, 120 + 46 * len(selected))
    body = [
        (
            '    <text x="300" y="82" font-size="11">Normalized MAE '
            '(error / frozen prior width)</text>'
        ),
        (
            '    <text x="655" y="82" font-size="11">Empirical 90% coverage '
            "and 95% Wilson interval</text>"
        ),
        '    <line x1="655" x2="915" y1="91" y2="91" stroke="#8c98a4"/>',
    ]
    for index, row in enumerate(selected):
        y = 108 + index * 46
        mae, mae_text, support_width = _normalized_mae(row, spec)
        coverage = _defined_float(row.empirical_90_coverage)
        interval = row.binomial_wilson_interval
        raw_mae_text = _scalar(row.mae, spec)
        coverage_text = _scalar(row.empirical_90_coverage, spec)
        interval_text = _interval(interval, spec)
        body.append(
            f'    <g data-role="target" '
            f'data-parameter="{html.escape(row.parameter, quote=True)}" '
            f'data-mae-raw="{html.escape(raw_mae_text, quote=True)}" '
            f'data-mae-normalized="{html.escape(mae_text, quote=True)}" '
            f'data-prior-support-width="{_number(support_width, spec)}" '
            'data-mae-unit="fraction_of_frozen_prior_support_width" '
            f'data-coverage="{html.escape(coverage_text, quote=True)}" '
            f'data-wilson-interval="{html.escape(interval_text, quote=True)}">'
        )
        body.append(
            f'      <text x="24" y="{y + 12}" font-size="11">'
            f"{html.escape(row.parameter)}</text>"
        )
        body.append(
            f'      <line x1="300" x2="570" y1="{y + 8}" y2="{y + 8}" '
            'stroke="#d9dee5"/>'
        )
        if mae is not None:
            body.append(
                f'      <rect x="300" y="{y + 2}" '
                f'width="{250.0 * mae / mae_scale:.3f}" height="12" '
                'fill="#7a5af8"/>'
            )
        if coverage is not None:
            x = 655.0 + 260.0 * coverage
            body.append(
                f'      <circle cx="{x:.3f}" cy="{y + 8}" r="5" fill="#167d5a"/>'
            )
        if interval.status is ScalarStatus.DEFINED:
            assert interval.lower is not None and interval.upper is not None
            left = 655.0 + 260.0 * interval.lower
            right = 655.0 + 260.0 * interval.upper
            body.append(
                f'      <line x1="{left:.3f}" x2="{right:.3f}" '
                f'y1="{y + 8}" y2="{y + 8}" stroke="#167d5a" stroke-width="2"/>'
            )
        body.append(
            f'      <text x="300" y="{y + 29}" font-size="9">MAE '
            f"{html.escape(mae_text)} prior widths; coverage "
            f"{html.escape(coverage_text)}; "
            f"Wilson {html.escape(interval_text)}</text>"
        )
        body.append("    </g>")
    return _svg_document(
        title="Synthetic calibration recovery and coverage",
        subtitle=(
            "MAE is normalized by each frozen prior-support width; coverage is "
            "empirical 90% interval coverage and whiskers are 95% binomial Wilson "
            "intervals, not posterior intervals."
        ),
        plan_sha256=plan_sha256,
        source_ids=("core-reporting",),
        height=height,
        body=body,
        spec=spec,
    )


def _interval_coverage_width_svg(
    rows: Sequence[RQ2ParameterSummary],
    *,
    plan_sha256: str,
    spec: FrozenAAMASPaperSpec,
) -> bytes:
    selected = tuple(row for row in rows if row.role == "target")
    if not selected:
        _integrity_fail("interval coverage/width figure lacks target summaries")
    widths = tuple(
        item
        for row in selected
        if (
            item := _normalized_parameter_scalar(
                row.mean_interval_width,
                parameter=row.parameter,
                spec=spec,
            )[0]
        )
        is not None
    )
    width_scale = max(widths, default=1.0) or 1.0
    height = max(spec.svg_minimum_height, 120 + 42 * len(selected))
    body = [
        (
            '    <g data-layout="separate-panels" '
            'data-width-unit="fraction_of_frozen_prior_support_width" '
            'data-coverage-unit="probability" '
            f'data-width-scale-max="{_number(width_scale, spec)}">'
        ),
        (
            '      <text x="290" y="82" font-size="11" font-weight="700">'
            'Panel A: mean 90% interval width / prior-support width</text>'
        ),
        (
            '      <text x="665" y="82" font-size="11" font-weight="700">'
            'Panel B: empirical 90% coverage (probability)</text>'
        ),
        '      <line x1="290" x2="555" y1="91" y2="91" stroke="#8c98a4"/>',
        '      <line x1="665" x2="915" y1="91" y2="91" stroke="#8c98a4"/>',
        "    </g>",
    ]
    for index, row in enumerate(selected):
        y = 108 + 42 * index
        normalized_width, width_text, support_width = _normalized_parameter_scalar(
            row.mean_interval_width,
            parameter=row.parameter,
            spec=spec,
        )
        coverage = _defined_float(row.empirical_90_coverage)
        coverage_text = _scalar(row.empirical_90_coverage, spec)
        body.append(
            f'    <g data-parameter="{html.escape(row.parameter, quote=True)}" '
            'data-normalized-mean-interval-width="'
            f'{html.escape(width_text, quote=True)}" '
            f'data-prior-support-width="{_number(support_width, spec)}" '
            f'data-empirical-coverage="{html.escape(coverage_text, quote=True)}">'
        )
        body.append(
            f'      <text x="24" y="{y + 10}" font-size="11">'
            f"{html.escape(row.parameter)}</text>"
        )
        if normalized_width is not None:
            body.append(
                f'      <rect x="290" y="{y}" '
                f'width="{245.0 * normalized_width / width_scale:.3f}" height="10" '
                'fill="#7a5af8"/>'
            )
        if coverage is not None:
            x = 665.0 + 250.0 * coverage
            body.append(
                f'      <circle cx="{x:.3f}" cy="{y + 5}" r="5" fill="#167d5a"/>'
            )
        body.append(
            f'      <text x="290" y="{y + 27}" font-size="9">width '
            f"{html.escape(width_text)} prior widths; coverage "
            f"{html.escape(coverage_text)}</text>"
        )
        body.append("    </g>")
    return _svg_document(
        title="Interval coverage and width on unit-consistent separate panels",
        subtitle=(
            "Widths are normalized by frozen prior-support width; coverage is a "
            "probability. Independent axes prevent mixing parameter and probability "
            "units."
        ),
        plan_sha256=plan_sha256,
        source_ids=("core-reporting",),
        height=height,
        body=body,
        spec=spec,
    )


def _risk_svg(
    rows: Sequence[RQ3RiskCoveragePoint],
    *,
    plan_sha256: str,
    spec: FrozenAAMASPaperSpec,
) -> bytes:
    if not rows:
        _integrity_fail("core reporting omitted the frozen risk-coverage grid")
    defined = tuple(
        row
        for row in rows
        if row.selection_coverage.status is ScalarStatus.DEFINED
        and row.mean_loss.status is ScalarStatus.DEFINED
    )
    risks = [cast(float, row.mean_loss.value) for row in defined]
    risk_max = max(risks, default=1.0)
    if risk_max == 0.0:
        risk_max = 1.0
    body = [
        '    <line x1="105" x2="900" y1="300" y2="300" stroke="#17212b"/>',
        '    <line x1="105" x2="105" y1="80" y2="300" stroke="#17212b"/>',
        '    <text x="430" y="332" font-size="12">Selection coverage</text>',
        (
            '    <text x="20" y="190" font-size="12" '
            'transform="rotate(-90 20 190)">Mean normalized target loss</text>'
        ),
    ]
    for row in rows:
        coverage = _defined_float(row.selection_coverage)
        risk = _defined_float(row.mean_loss)
        interval = row.resampling_interval
        coverage_attribute = html.escape(
            _scalar(row.selection_coverage, spec), quote=True
        )
        risk_attribute = html.escape(_scalar(row.mean_loss, spec), quote=True)
        interval_attribute = html.escape(_interval(interval, spec), quote=True)
        body.append(
            f'    <g data-threshold="{_number(row.threshold, spec)}" '
            f'data-selection-coverage="{coverage_attribute}" '
            f'data-mean-loss="{risk_attribute}" '
            f'data-resampling-interval="{interval_attribute}">'
        )
        if coverage is not None and risk is not None:
            x = 105.0 + 795.0 * coverage
            y = 300.0 - 210.0 * risk / risk_max
            if interval.status is ScalarStatus.DEFINED:
                assert interval.lower is not None and interval.upper is not None
                low_y = 300.0 - 210.0 * interval.lower / risk_max
                high_y = 300.0 - 210.0 * interval.upper / risk_max
                body.append(
                    f'      <line x1="{x:.3f}" x2="{x:.3f}" '
                    f'y1="{low_y:.3f}" y2="{high_y:.3f}" '
                    'stroke="#b85c00" stroke-width="2"/>'
                )
            body.append(
                f'      <circle cx="{x:.3f}" cy="{y:.3f}" r="5" fill="#b85c00"/>'
            )
            body.append(
                f'      <text x="{x + 7:.3f}" y="{y - 5:.3f}" font-size="9">'
                f"t={_number(row.threshold, spec)}</text>"
            )
        else:
            body.append(
                f'      <text x="115" y="{88 + 14 * rows.index(row)}" '
                f'font-size="9">threshold {_number(row.threshold, spec)}: '
                f"{html.escape(_scalar(row.mean_loss, spec))}</text>"
            )
        body.append("    </g>")
    return _svg_document(
        title="Target-only selective risk versus coverage",
        subtitle=(
            "Whiskers are synthetic target resampling intervals, not confidence "
            "or posterior intervals; undefined frozen thresholds remain explicit."
        ),
        plan_sha256=plan_sha256,
        source_ids=("core-reporting",),
        height=spec.svg_minimum_height,
        body=body,
        spec=spec,
    )


def _branch_svg(
    rows: Sequence[IntegratedBranchDifferenceSummary],
    *,
    plan_sha256: str,
    spec: FrozenAAMASPaperSpec,
) -> bytes:
    if not rows:
        _integrity_fail("branch reporting omitted effect summaries")
    registered_metrics = {
        metric for metric, _unit, _direction in spec.integrated_metric_presentation
    }
    observed_metrics = {row.metric_name for row in rows}
    if observed_metrics != registered_metrics:
        _integrity_fail(
            "integrated branch metric registry is missing, extra, or changed"
        )
    selected = tuple(
        row for row in rows if row.metric_name in spec.integrated_primary_metrics
    )
    if not selected or {row.metric_name for row in selected} != set(
        spec.integrated_primary_metrics
    ):
        _integrity_fail("integrated branch primary metric panel is incomplete")
    selected_presentations = {
        _metric_presentation(row.metric_name, spec) for row in selected
    }
    selected_units = {unit for unit, _direction in selected_presentations}
    if len(selected_units) != 1:
        _integrity_fail("integrated branch primary panel mixes metric units")
    primary_unit = next(iter(selected_units))
    values = [value for row in selected for value in row.draws]
    minimum = min((*values, 0.0))
    maximum = max((*values, 0.0))
    if minimum == maximum:
        minimum -= 1.0
        maximum += 1.0
    span = maximum - minimum
    axis_left = 345.0
    axis_width = 570.0

    def x(value: float) -> float:
        return axis_left + axis_width * (value - minimum) / span

    height = max(spec.svg_minimum_height, 115 + 38 * len(selected))
    zero_x = x(0.0)
    body = [
        f'    <line x1="{zero_x:.3f}" x2="{zero_x:.3f}" y1="72" '
        f'y2="{height - 40}" stroke="#8c98a4" stroke-dasharray="3 3"/>',
        (
            f'    <text x="345" y="75" font-size="10">Primary metric panel: '
            f"{html.escape(', '.join(spec.integrated_primary_metrics))}; unit "
            f"{html.escape(primary_unit)}; q05–q95 are descriptive, not CIs</text>"
        ),
    ]
    for row_index, row in enumerate(selected):
        y = 103 + row_index * 38
        label = f"{row.horizon_id}/{row.group_id}/{row.metric_name}"
        unit, direction = _metric_presentation(row.metric_name, spec)
        q = row.quantiles
        mcse = row.mcse.value if row.mcse.status == "defined" else None
        mcse_attribute = html.escape(
            "UNAVAILABLE" if mcse is None else _number(mcse, spec), quote=True
        )
        body.append(
            f'    <g data-cell="{html.escape(label, quote=True)}" '
            f'data-mean="{_number(row.mean, spec)}" '
            f'data-q05="{_number(q.q05, spec)}" '
            f'data-q50="{_number(q.q50, spec)}" '
            f'data-q95="{_number(q.q95, spec)}" '
            f'data-mcse="{mcse_attribute}" '
            f'data-unit="{html.escape(unit, quote=True)}" '
            f'data-direction="{html.escape(direction, quote=True)}">'
        )
        body.append(
            f'      <text x="24" y="{y + 4}" font-size="9">{html.escape(label)}</text>'
        )
        for draw_index, draw in enumerate(row.draws):
            jitter = ((draw_index % 5) - 2) * 1.4
            body.append(
                f'      <circle cx="{x(draw):.3f}" cy="{y + jitter:.3f}" r="1.6" '
                'fill="#92a5bd" opacity="0.55"/>'
            )
        body.append(
            f'      <line x1="{x(q.q05):.3f}" x2="{x(q.q95):.3f}" '
            f'y1="{y}" y2="{y}" stroke="#315efb" stroke-width="3"/>'
        )
        body.append(
            f'      <circle cx="{x(q.q50):.3f}" cy="{y:.3f}" r="4" fill="#17212b"/>'
        )
        mean_x = x(row.mean)
        body.append(
            f'      <path d="M {mean_x:.3f} {y - 5:.3f} L {mean_x + 5:.3f} '
            f"{y:.3f} L {mean_x:.3f} {y + 5:.3f} L {mean_x - 5:.3f} "
            f'{y:.3f} Z" fill="#f08c46"/>'
        )
        body.append("    </g>")
    return _svg_document(
        title="Integrated CLOB quoted-spread paired effects (price units)",
        subtitle=(
            "The frozen primary panel contains only quoted_spread in price_units; "
            "all other registered metric/unit cells remain in the generated table. "
            "Differences are candidate-minus-baseline; quantiles are descriptive, "
            "not confidence intervals."
        ),
        plan_sha256=plan_sha256,
        source_ids=("integrated-branch-reporting",),
        height=height,
        body=body,
        spec=spec,
    )


def _misspec_svg(
    view: _MisspecificationView,
    *,
    plan_sha256: str,
    spec: FrozenAAMASPaperSpec,
) -> bytes:
    body = [
        '    <line x1="130" x2="900" y1="300" y2="300" stroke="#17212b"/>',
        '    <line x1="130" x2="130" y1="80" y2="300" stroke="#17212b"/>',
        (
            '    <text x="22" y="205" font-size="12" '
            'transform="rotate(-90 22 205)">Held-out PPC failure rate</text>'
        ),
    ]
    for index, row in enumerate(view.target_aggregates):
        x = 240 + index * 360
        rate = row.held_out_ppc_failure_rate
        rate_text = "UNAVAILABLE" if rate is None else _number(rate, spec)
        body.append(
            f'    <g data-cell="{html.escape(row.cell.value, quote=True)}" '
            f'data-held-out-ppc-failure-rate="{html.escape(rate_text, quote=True)}">'
        )
        if rate is not None:
            height_value = 200.0 * rate
            body.append(
                f'      <rect x="{x}" y="{300 - height_value:.3f}" width="150" '
                f'height="{height_value:.3f}" fill="'
                f'{"#315efb" if index == 0 else "#c23b53"}"/>'
            )
            body.append(
                f'      <text x="{x + 53}" y="{290 - height_value:.3f}" '
                f'font-size="12">{rate_text}</text>'
            )
        else:
            body.append(
                f'      <text x="{x}" y="270" font-size="11">UNAVAILABLE</text>'
            )
        body.append(
            f'      <text x="{x}" y="322" font-size="11">'
            f"{html.escape(row.cell.value)}</text>"
        )
        body.append("    </g>")
    gap = (
        "UNAVAILABLE"
        if view.held_out_ppc_failure_gap is None
        else _number(view.held_out_ppc_failure_gap, spec)
    )
    body.append(
        f'    <text x="610" y="62" font-size="11">Decision: '
        f"{html.escape(view.decision.value)}; gap: {html.escape(gap)}</text>"
    )
    return _svg_document(
        title="Structural misspecification held-out PPC failures",
        subtitle=(
            "Frozen synthetic negative control; rates use every registered target "
            "attempt and do not authorize real-market validity."
        ),
        plan_sha256=plan_sha256,
        source_ids=("misspecification",),
        height=spec.svg_minimum_height,
        body=body,
        spec=spec,
    )


def _validity_heatmap_svg(
    view: _ValidityView,
    *,
    plan_sha256: str,
    spec: FrozenAAMASPaperSpec,
) -> bytes:
    colors = {
        "pass": "#167d5a",
        "warn": "#d89b00",
        "fail": "#c23b53",
        "unknown": "#8c98a4",
    }
    height = max(spec.svg_minimum_height, 115 + 30 * len(view.checks))
    body = [
        (
            '    <text x="24" y="78" font-size="11">Self-reported synthetic '
            f"certificate: level {html.escape(view.level)}; diagnostic "
            f"{html.escape(view.diagnostic_level)}</text>"
        )
    ]
    for index, (check_id, status_value) in enumerate(view.checks):
        y = 98 + 30 * index
        body.extend(
            (
                f'    <g data-check-id="{html.escape(check_id, quote=True)}" '
                f'data-status="{html.escape(status_value, quote=True)}">',
                f'      <text x="24" y="{y + 16}" font-size="11">'
                f"{html.escape(check_id)}</text>",
                f'      <rect x="360" y="{y}" width="170" height="22" '
                f'fill="{colors[status_value]}"/>',
                f'      <text x="445" y="{y + 15}" text-anchor="middle" '
                f'font-size="11" fill="#ffffff">{html.escape(status_value)}</text>',
                "    </g>",
            )
        )
    return _svg_document(
        title="Failure-certificate status heatmap (self-reported synthetic evidence)",
        subtitle=(
            "Every frozen validity check is retained; UNKNOWN and failure states are "
            "not zero-filled. The certificate remains ABSTAIN and is not external "
            "validation."
        ),
        plan_sha256=plan_sha256,
        source_ids=("validity-certificate",),
        height=height,
        body=body,
        spec=spec,
    )


def _table_header(plan_sha256: str, description: str) -> list[str]:
    return [
        "% SYNTHETIC: generated only from audited PolicyTwin result artifacts.",
        f"% Plan SHA-256: {plan_sha256}",
        f"% {description}",
        "",
    ]


def _main_results_tex(
    loaded: _LoadedSources,
    *,
    plan_sha256: str,
    spec: FrozenAAMASPaperSpec,
) -> bytes:
    lines = _table_header(
        plan_sha256,
        "Descriptive synthetic results; no empirical or real-causal authority.",
    )
    lines.extend(
        [
            r"\begin{table}[t]",
            r"\centering\small",
            r"\begin{tabular}{llllrrr}",
            (
                r"Cell & RNG & Status & $n$ & Mean & Variance "
                r"(outcome units$^2$) & MCSE (outcome units) \\"
            ),
            r"\hline",
        ]
    )
    for rng_row in loaded.core.rq1_rng_rows:
        status_value = "complete" if rng_row.failed == 0 else "failed_registered"
        lines.append(
            " & ".join(
                (
                    _latex(rng_row.cell_id),
                    _latex(rng_row.mode),
                    _latex(status_value),
                    str(rng_row.completed),
                    _latex(_scalar(rng_row.mean_paired_difference, spec)),
                    _latex(_scalar(rng_row.sample_variance, spec)),
                    _latex(_scalar(rng_row.mcse, spec)),
                )
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\end{tabular}",
            (
                r"\caption{SYNTHETIC RNG paired-difference results. Variance uses "
                r"ddof=1 and MCSE is derived from the retained replicate count. "
                rf"Plan hash: \texttt{{{plan_sha256}}}.}}"
            ),
            r"\end{table}",
            "",
            r"\begin{table}[t]",
            r"\centering\small",
            r"\begin{tabular}{llrrrr}",
            (
                r"Role & Parameter & $n$ & Normalized MAE & Empirical 90\% coverage & "
                r"95\% Wilson interval \\"
            ),
            r"\hline",
        ]
    )
    for parameter_row in loaded.core.rq2_parameter_summaries:
        _normalized_value, normalized_text, _support_width = _normalized_mae(
            parameter_row, spec
        )
        lines.append(
            " & ".join(
                (
                    _latex(parameter_row.role),
                    _latex(parameter_row.parameter),
                    str(parameter_row.n),
                    _latex(normalized_text),
                    _latex(_scalar(parameter_row.empirical_90_coverage, spec)),
                    _latex(_interval(parameter_row.binomial_wilson_interval, spec)),
                )
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\end{tabular}",
            (
                r"\caption{SYNTHETIC calibration recovery. MAE is divided by the "
                r"parameter's frozen prior-support width. The final column is a "
                r"binomial Wilson interval for the empirical coverage proportion, "
                rf"not a parameter interval. Plan hash: \texttt{{{plan_sha256}}}.}}"
            ),
            r"\end{table}",
            "",
            r"\begin{table}[t]",
            r"\centering\scriptsize",
            r"\begin{tabular}{lllllrrrr}",
            (
                r"Horizon & Group & Metric & Unit & Direction & Mean & MCSE & "
                r"q05 & q50 / q95 \\"
            ),
            r"\hline",
        ]
    )
    for branch_row in loaded.branch.restriction_summaries:
        q = branch_row.quantiles
        unit, direction = _metric_presentation(branch_row.metric_name, spec)
        mcse = (
            "UNAVAILABLE[undefined_zero_denominator]"
            if branch_row.mcse.value is None
            else _number(branch_row.mcse.value, spec)
        )
        lines.append(
            " & ".join(
                (
                    _latex(branch_row.horizon_id),
                    _latex(branch_row.group_id),
                    _latex(branch_row.metric_name),
                    _latex(unit),
                    _latex(direction),
                    _number(branch_row.mean, spec),
                    _latex(mcse),
                    _number(q.q05, spec),
                    f"{_number(q.q50, spec)} / {_number(q.q95, spec)}",
                )
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\end{tabular}",
            (
                r"\caption{SYNTHETIC integrated CLOB candidate-minus-baseline "
                r"effects. Unit and directional interpretation are frozen before "
                r"rendering; direction is descriptive and not a welfare ordering. "
                r"q05, q50, and q95 are descriptive finite-replicate quantiles, "
                r"not confidence intervals. "
                rf"Plan hash: \texttt{{{plan_sha256}}}.}}"
            ),
            r"\end{table}",
            "",
            r"\begin{table}[t]",
            r"\centering\small",
            r"\begin{tabular}{lrrrrr}",
            (
                r"Negative-control cell & Expected & Completed & Held-out PPC "
                r"failure & Recovery loss & Boundary / multimodality \\"
            ),
            r"\hline",
        ]
    )

    def optional_misspec(value: float | None) -> str:
        return "UNAVAILABLE" if value is None else _number(value, spec)

    for misspec_row in loaded.misspecification.target_aggregates:
        lines.append(
            " & ".join(
                (
                    _latex(misspec_row.cell.value),
                    str(misspec_row.expected_target_attempts),
                    str(misspec_row.completed_target_attempts),
                    optional_misspec(misspec_row.held_out_ppc_failure_rate),
                    optional_misspec(misspec_row.normalized_recovery_loss),
                    f"{optional_misspec(misspec_row.boundary_rate)} / "
                    f"{optional_misspec(misspec_row.multimodality_rate)}",
                )
            )
            + r" \\"
        )
    gap = (
        "UNAVAILABLE"
        if loaded.misspecification.held_out_ppc_failure_gap is None
        else _number(loaded.misspecification.held_out_ppc_failure_gap, spec)
    )
    lines.extend(
        [
            r"\end{tabular}",
            (
                r"\caption{SYNTHETIC structural misspecification negative control. "
                f"Frozen decision: {_latex(loaded.misspecification.decision.value)}; "
                f"held-out PPC failure-rate gap: {_latex(gap)}. "
                rf"Plan hash: \texttt{{{plan_sha256}}}.}}"
            ),
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def _ablation_tex(
    artifact: AAMASSubmissionArtifact,
    *,
    plan_sha256: str,
    spec: FrozenAAMASPaperSpec,
) -> bytes:
    if not artifact.rq1_mechanism_summaries:
        _integrity_fail("core reporting omitted preregistered ablation summaries")
    lines = _table_header(
        plan_sha256,
        "Complete frozen mechanism-ablation registry, including failure statuses.",
    )
    lines.extend(
        [
            r"\begin{table}[t]",
            r"\centering\small",
            r"\begin{tabular}{lrrrrrr}",
            r"Variant & Expected & Completed & Failed & Mean & q05 & q50 / q95 \\",
            r"\hline",
        ]
    )
    for row in artifact.rq1_mechanism_summaries:
        lines.append(
            " & ".join(
                (
                    _latex(row.variant),
                    str(row.expected),
                    str(row.completed),
                    str(row.failed),
                    _latex(_scalar(row.mean, spec)),
                    _latex(_scalar(row.q05, spec)),
                    f"{_latex(_scalar(row.q50, spec))} / "
                    f"{_latex(_scalar(row.q95, spec))}",
                )
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\end{tabular}",
            (
                r"\caption{SYNTHETIC frozen mechanism ablation registry; failed "
                r"runs are retained and block their aggregate rather than being "
                rf"dropped. Plan hash: \texttt{{{plan_sha256}}}.}}"
            ),
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def _evidence_status_tex(
    loaded: _LoadedSources,
    *,
    plan_sha256: str,
) -> bytes:
    lines = _table_header(
        plan_sha256,
        "Authority and failure status; unavailable evidence is never zero-filled.",
    )
    lines.extend(
        [
            r"\begin{table}[t]",
            r"\centering\small",
            r"\begin{tabular}{ll}",
            r"Registered output & Status \\",
            r"\hline",
        ]
    )
    for output_id, status_value in loaded.root_status.output_statuses:
        lines.append(f"{_latex(output_id)} & {_latex(status_value)} \\\\")
    lines.append(
        r"predicted\_vs\_empirical\_effect\_plot & "
        r"blocked\_no\_empirical\_artifact \\"
    )
    lines.extend(
        [
            r"\hline",
            r"Claim authority & synthetic-only; validity ABSTAIN \\",
            r"Publication eligible & false \\",
            r"\end{tabular}",
            (
                r"\caption{SYNTHETIC evidence status. No empirical artifact was "
                r"available, so no predicted-versus-empirical figure was generated. "
                rf"Plan hash: \texttt{{{plan_sha256}}}.}}"
            ),
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def _compute_tex(
    timing: Sequence[_TimingRow],
    *,
    plan_sha256: str,
    spec: FrozenAAMASPaperSpec,
) -> bytes:
    lines = _table_header(
        plan_sha256,
        "Single-run observed wall time; this is not compute-scaling evidence.",
    )
    lines.extend(
        [
            r"\begin{table}[t]",
            r"\centering\small",
            r"\begin{tabular}{lr}",
            r"Component & Wall time (seconds) \\",
            r"\hline",
        ]
    )
    for row in timing:
        lines.append(
            f"{_latex(row.component_id)} & {_number(row.wall_seconds, spec)} \\\\"
        )
    total = sum(row.wall_seconds for row in timing)
    lines.extend(
        [
            r"\hline",
            f"total & {_number(total, spec)} \\\\",
            r"\end{tabular}",
            (
                r"\caption{SYNTHETIC single-run observed component runtime. "
                r"Durations are non-scientific process measurements and do not "
                r"constitute a compute-scaling experiment. "
                rf"Plan hash: \texttt{{{plan_sha256}}}.}}"
            ),
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def _macros_tex(
    loaded: _LoadedSources,
    *,
    plan_sha256: str,
    spec: FrozenAAMASPaperSpec,
) -> bytes:
    target_parameters = tuple(
        row for row in loaded.core.rq2_parameter_summaries if row.role == "target"
    )
    expected_parameters = tuple(
        name for name, _lower, _upper in spec.calibration_prior_supports
    )
    if tuple(row.parameter for row in target_parameters) != expected_parameters:
        _integrity_fail("target parameter macro registry is missing or reordered")
    misspec_by_cell = {
        row.cell: row for row in loaded.misspecification.target_aggregates
    }
    well = misspec_by_cell[MisspecificationCell.WELL_SPECIFIED]
    misspecified = misspec_by_cell[MisspecificationCell.MISSPECIFIED]
    total_wall = sum(row.wall_seconds for row in loaded.timing)

    def optional(value: float | None) -> str:
        return "UNAVAILABLE" if value is None else _number(value, spec)

    def complete_mode_mean(mode: str, field: str) -> str:
        selected = tuple(row for row in loaded.core.rq1_rng_rows if row.mode == mode)
        if not selected:
            _integrity_fail(f"RNG macro mode is absent: {mode}")
        values: list[float] = []
        for row in selected:
            scalar = cast(ReportedScalar, getattr(row, field))
            value = _defined_float(scalar)
            if value is None:
                return "UNAVAILABLE[aggregate_contains_undefined_source]"
            values.append(value)
        return _number(math.fsum(values) / len(values), spec)

    def interval_bound(value: TypedInterval, bound: Literal["lower", "upper"]) -> str:
        if value.status is not ScalarStatus.DEFINED:
            return f"UNAVAILABLE[{value.status.value}]"
        result = value.lower if bound == "lower" else value.upper
        assert result is not None
        return _number(result, spec)

    def macro(name: str, value: object) -> str:
        return rf"\providecommand{{\{name}}}{{{_latex(value)}}}"

    lines = _table_header(
        plan_sha256,
        "Machine-generated paper macros; all values are synthetic.",
    )
    lines.extend(
        [
            macro("PolicyTwinArtifactLabel", "SYNTHETIC"),
            macro("PolicyTwinPlanHash", plan_sha256),
            macro("PolicyTwinValidityLevel", "ABSTAIN"),
            macro("PolicyTwinRNGCellCount", len(loaded.core.rq1_rng_rows)),
            macro("PolicyTwinTargetParameterCount", len(target_parameters)),
            macro(
                "PolicyTwinBranchActivationRate",
                _number(loaded.branch.branch_activation_rate, spec),
            ),
            macro(
                "PolicyTwinWellSpecifiedPPCFailureRate",
                optional(well.held_out_ppc_failure_rate),
            ),
            macro(
                "PolicyTwinMisspecifiedPPCFailureRate",
                optional(misspecified.held_out_ppc_failure_rate),
            ),
            macro(
                "PolicyTwinMisspecificationDecision",
                loaded.misspecification.decision.value,
            ),
            macro(
                "PolicyTwinOperationalWallSeconds", _number(total_wall, spec)
            ),
        ]
    )
    mode_macros = (
        ("independent", "Independent"),
        ("same_sequential", "SameSequential"),
        ("event_keyed", "EventKeyed"),
    )
    for mode, stem in mode_macros:
        lines.extend(
            (
                macro(
                    f"PolicyTwin{stem}ExactEffectRate",
                    complete_mode_mean(mode, "exact_effect_rate"),
                ),
                macro(
                    f"PolicyTwin{stem}PostBranchShockMatchRate",
                    complete_mode_mean(mode, "post_branch_shock_match_rate"),
                ),
                macro(
                    f"PolicyTwin{stem}VarianceRatioToIndependent",
                    complete_mode_mean(mode, "variance_ratio_vs_independent"),
                ),
            )
        )
    invariant_rates = loaded.branch.invariant_rates
    lines.extend(
        (
            macro(
                "PolicyTwinCLOBNoOpExactZeroRate",
                _number(invariant_rates.no_op_exact_zero_rate, spec),
            ),
            macro(
                "PolicyTwinCLOBBaselineInvarianceRate",
                _number(invariant_rates.baseline_invariance_rate, spec),
            ),
            macro(
                "PolicyTwinCLOBShockAlignmentRate",
                _number(invariant_rates.event_keyed_shock_alignment_rate, spec),
            ),
            macro(
                "PolicyTwinCLOBReplayIdentityRate",
                _number(invariant_rates.replay_byte_identity_rate, spec),
            ),
        )
    )
    completed_attempts = sum(
        item.status is AAMASPublicationAttemptStatus.COMPLETED
        for item in loaded.attempts
    )
    failed_attempts = sum(
        item.status is AAMASPublicationAttemptStatus.FAILED for item in loaded.attempts
    )
    blocked_attempts = sum(
        item.status is AAMASPublicationAttemptStatus.BLOCKED_UPSTREAM
        for item in loaded.attempts
    )
    lines.extend(
        (
            macro("PolicyTwinAttemptCompletedCount", completed_attempts),
            macro("PolicyTwinAttemptFailedCount", failed_attempts),
            macro("PolicyTwinAttemptBlockedCount", blocked_attempts),
        )
    )
    parameter_macro_stems = {
        "liquidity": "Liquidity",
        "signal_strength": "SignalStrength",
    }
    for row in target_parameters:
        stem = parameter_macro_stems[row.parameter]
        _normalized, normalized_text, _width = _normalized_mae(row, spec)
        lines.extend(
            (
                macro(f"PolicyTwin{stem}NormalizedMAE", normalized_text),
                macro(
                    f"PolicyTwin{stem}EmpiricalNinetyCoverage",
                    _scalar(row.empirical_90_coverage, spec),
                ),
                macro(
                    f"PolicyTwin{stem}CoverageWilsonLower",
                    interval_bound(row.binomial_wilson_interval, "lower"),
                ),
                macro(
                    f"PolicyTwin{stem}CoverageWilsonUpper",
                    interval_bound(row.binomial_wilson_interval, "upper"),
                ),
            )
        )
    risk_rows = tuple(
        sorted(loaded.core.rq3_risk_coverage, key=lambda row: row.threshold)
    )
    if not risk_rows:
        _integrity_fail("risk-coverage endpoint macros lack frozen rows")
    for risk_row, stem in ((risk_rows[0], "Lowest"), (risk_rows[-1], "Highest")):
        lines.extend(
            (
                macro(
                    f"PolicyTwinRiskCoverage{stem}Threshold",
                    _number(risk_row.threshold, spec),
                ),
                macro(
                    f"PolicyTwinRiskCoverage{stem}SelectionCoverage",
                    _scalar(risk_row.selection_coverage, spec),
                ),
                macro(
                    f"PolicyTwinRiskCoverage{stem}MeanLoss",
                    _scalar(risk_row.mean_loss, spec),
                ),
            )
        )
    lines.append("")
    return "\n".join(lines).encode("utf-8")


def _render_payloads(
    loaded: _LoadedSources,
    *,
    plan_sha256: str,
    spec: FrozenAAMASPaperSpec,
) -> dict[str, bytes]:
    return {
        "rng-variance-mcse.svg": _rng_svg(
            loaded.core.rq1_rng_rows, plan_sha256=plan_sha256, spec=spec
        ),
        "calibration-recovery-coverage.svg": _calibration_svg(
            loaded.core.rq2_parameter_summaries,
            plan_sha256=plan_sha256,
            spec=spec,
        ),
        "interval-coverage-width.svg": _interval_coverage_width_svg(
            loaded.core.rq2_parameter_summaries,
            plan_sha256=plan_sha256,
            spec=spec,
        ),
        "risk-coverage.svg": _risk_svg(
            loaded.core.rq3_risk_coverage, plan_sha256=plan_sha256, spec=spec
        ),
        "integrated-branch-effect-distribution.svg": _branch_svg(
            loaded.branch.restriction_summaries,
            plan_sha256=plan_sha256,
            spec=spec,
        ),
        "misspec-ppc-failure.svg": _misspec_svg(
            loaded.misspecification, plan_sha256=plan_sha256, spec=spec
        ),
        "failure-certificate-heatmap.svg": _validity_heatmap_svg(
            loaded.validity, plan_sha256=plan_sha256, spec=spec
        ),
        "main-results.tex": _main_results_tex(
            loaded, plan_sha256=plan_sha256, spec=spec
        ),
        "ablation.tex": _ablation_tex(loaded.core, plan_sha256=plan_sha256, spec=spec),
        "evidence-status.tex": _evidence_status_tex(loaded, plan_sha256=plan_sha256),
        "compute.tex": _compute_tex(loaded.timing, plan_sha256=plan_sha256, spec=spec),
        "results-macros.tex": _macros_tex(loaded, plan_sha256=plan_sha256, spec=spec),
    }


def _output_sources(filename: str) -> tuple[str, ...]:
    mapping = {
        "rng-variance-mcse.svg": ("core-reporting",),
        "calibration-recovery-coverage.svg": ("core-reporting",),
        "interval-coverage-width.svg": ("core-reporting",),
        "risk-coverage.svg": ("core-reporting",),
        "integrated-branch-effect-distribution.svg": ("integrated-branch-reporting",),
        "misspec-ppc-failure.svg": ("misspecification",),
        "failure-certificate-heatmap.svg": ("validity-certificate",),
        "main-results.tex": (
            "core-reporting",
            "integrated-branch-reporting",
            "misspecification",
        ),
        "ablation.tex": ("core-reporting",),
        "evidence-status.tex": (
            "root-result",
            "root-output-status",
        ),
        "compute.tex": ("root-timing",),
        "results-macros.tex": (
            "core-reporting",
            "integrated-branch-reporting",
            "misspecification",
            "root-timing",
            "root-result",
            "root-attempt-registry",
        ),
    }
    try:
        return mapping[filename]
    except KeyError as error:
        raise AAMASPaperIntegrityError(
            f"unregistered paper output: {filename}"
        ) from error


def _output_id(filename: str) -> str:
    overrides = {
        "calibration-recovery-coverage.svg": "calibration-diagnostics",
        "interval-coverage-width.svg": "interval-coverage-width",
        "risk-coverage.svg": "risk-coverage",
        "integrated-branch-effect-distribution.svg": (
            "paired-effect-distributions"
        ),
        "failure-certificate-heatmap.svg": "failure-certificate-heatmap",
        "ablation.tex": "ablation-table",
        "compute.tex": "observed-component-runtime-table",
    }
    return overrides.get(filename, filename.removesuffix(".svg").removesuffix(".tex"))


def _output_unit_semantics(
    filename: str, spec: FrozenAAMASPaperSpec
) -> dict[str, object]:
    semantics: dict[str, dict[str, object]] = {
        "rng-variance-mcse.svg": {
            "layout": "separate_independently_scaled_panels",
            "mcse_unit": "paired_outcome_units",
            "variance_unit": "paired_outcome_units_squared",
        },
        "calibration-recovery-coverage.svg": {
            "coverage_unit": "probability",
            "mae_unit": spec.calibration_error_unit,
        },
        "interval-coverage-width.svg": {
            "coverage_unit": "probability",
            "layout": "separate_independently_scaled_panels",
            "width_unit": "fraction_of_frozen_prior_support_width",
        },
        "risk-coverage.svg": {
            "coverage_unit": "probability",
            "loss_unit": "normalized_target_loss",
        },
        "integrated-branch-effect-distribution.svg": {
            "metric_names": list(spec.integrated_primary_metrics),
            "unit": "price_units",
        },
        "misspec-ppc-failure.svg": {"failure_rate_unit": "probability"},
        "failure-certificate-heatmap.svg": {
            "cell_value": "categorical_validity_status"
        },
        "main-results.tex": {
            "unit_policy": "each_column_or_metric_row_declares_its_unit"
        },
        "ablation.tex": {"unit_policy": "frozen_quoted_spread_effect_units"},
        "evidence-status.tex": {"unit_policy": "categorical_status_only"},
        "compute.tex": {
            "evidence_scope": "single_observed_run_not_scaling",
            "unit": "seconds",
        },
        "results-macros.tex": {
            "unit_policy": "macro_names_and_frozen_derivations_define_units"
        },
    }
    try:
        return semantics[filename]
    except KeyError as error:
        raise AAMASPaperIntegrityError(
            f"output lacks frozen unit semantics: {filename}"
        ) from error


def _manifest_mapping(
    payloads: Mapping[str, bytes],
    loaded: _LoadedSources,
    *,
    plan_sha256: str,
    paper_spec_sha256: str,
    publication_root_seal_sha256: str,
    spec: FrozenAAMASPaperSpec,
) -> dict[str, object]:
    if tuple(payloads) != (*_FIGURE_FILES, *_TABLE_FILES):
        _integrity_fail("rendered payload membership or order changed")
    source_by_id = {item.source_id: item for item in loaded.records}
    if frozenset(source_by_id) != frozenset(_SOURCE_PATHS):
        _integrity_fail("validated paper source registry changed")

    def output_source_rows(filename: str) -> list[dict[str, object]]:
        return [
            {
                "role": source_by_id[source_id].role,
                "sha256": source_by_id[source_id].sha256,
                "source_id": source_id,
                "status": source_by_id[source_id].status,
            }
            for source_id in _output_sources(filename)
        ]

    outputs: list[dict[str, object]] = [
        {
            "artifact_label": AAMAS_PAPER_ARTIFACT_LABEL,
            "media_type": (
                "image/svg+xml" if filename.endswith(".svg") else "text/x-tex"
            ),
            "output_id": _output_id(filename),
            "path": filename,
            "plan_sha256": plan_sha256,
            "sha256": sha256_bytes(payload),
            "size_bytes": len(payload),
            "source_ids": list(_output_sources(filename)),
            "sources": output_source_rows(filename),
            "status": "generated_synthetic",
            "unit_semantics": _output_unit_semantics(filename, spec),
        }
        for filename, payload in payloads.items()
    ]
    generated_ids = {cast(str, row["output_id"]) for row in outputs}
    for output_id, status_value in spec.scientific_output_registry:
        if status_value == "generated_synthetic":
            if output_id not in generated_ids:
                _integrity_fail(
                    f"registered generated scientific output is absent: {output_id}"
                )
            continue
        reason = _BLOCKED_OUTPUT_REASONS.get(output_id)
        if reason is None:
            _integrity_fail(f"blocked output lacks a frozen reason: {output_id}")
        outputs.append(
            {
                "artifact_label": AAMAS_PAPER_ARTIFACT_LABEL,
                "media_type": None,
                "output_id": output_id,
                "path": None,
                "plan_sha256": plan_sha256,
                "reason": reason,
                "sha256": None,
                "size_bytes": None,
                "source_ids": [],
                "sources": [],
                "status": status_value,
                "unit_semantics": None,
            }
        )
    return {
        "artifact_label": AAMAS_PAPER_ARTIFACT_LABEL,
        "empirical_input": {
            "input_id": "validated_empirical_effect_artifact",
            "status": "blocked_no_empirical_artifact",
        },
        "failure_policy": "fail_closed_for_every_unregistered_or_missing_source",
        "macro_derivations": {
            "attempt_counts": "all_eight_terminal_outer_attempt_registry_rows",
            "clob_invariant_rates": "complete_integrated_branch_replicate_registry",
            "parameter_recovery": (
                "target_mae_divided_by_frozen_prior_support_width_with_empirical_"
                "coverage_and_wilson_endpoints"
            ),
            "risk_coverage_endpoints": (
                "lowest_and_highest_prefrozen_threshold_rows_no_result_selection"
            ),
            "rng_mode_statistics": spec.rng_macro_aggregation,
        },
        "outputs": outputs,
        "paper_spec_sha256": paper_spec_sha256,
        "plan_sha256": plan_sha256,
        "publication_root_seal_sha256": publication_root_seal_sha256,
        "quantile_semantics": (
            "integrated_q05_q50_q95_are_descriptive_quantiles_not_confidence_intervals"
        ),
        "recovery_coverage_interval_semantics": (
            "95_percent_binomial_wilson_interval_for_empirical_90_percent_coverage"
        ),
        "risk_interval_semantics": (
            "synthetic_target_resampling_interval_not_confidence_interval"
        ),
        "scientific_output_registry": [
            {"output_id": output_id, "status": status}
            for output_id, status in spec.scientific_output_registry
        ],
        "schema_version": AAMAS_PAPER_MANIFEST_SCHEMA_VERSION,
        "sources": [item.to_mapping() for item in loaded.records],
        "synthetic_only": True,
    }


def _seal_mapping(
    payloads: Mapping[str, bytes],
    *,
    manifest_bytes: bytes,
    spec_bytes: bytes,
    plan_sha256: str,
    paper_spec_sha256: str,
    publication_root_seal_sha256: str,
) -> dict[str, object]:
    output_hashes = [
        {"path": path, "sha256": sha256_bytes(payload)}
        for path, payload in payloads.items()
    ]
    return {
        "artifact_label": AAMAS_PAPER_ARTIFACT_LABEL,
        "caller_retained_digest_required": True,
        "manifest_raw_sha256": sha256_bytes(manifest_bytes),
        "output_hash_registry_sha256": sha256_bytes(
            canonical_json_bytes(output_hashes)
        ),
        "paper_spec_raw_sha256": sha256_bytes(spec_bytes),
        "paper_spec_sha256": paper_spec_sha256,
        "plan_sha256": plan_sha256,
        "publication_root_seal_sha256": publication_root_seal_sha256,
        "schema_version": AAMAS_PAPER_SEAL_SCHEMA_VERSION,
        "synthetic_only": True,
    }


def _all_expected_bytes(
    loaded: _LoadedSources,
    *,
    plan_sha256: str,
    publication_root_seal_sha256: str,
    spec: FrozenAAMASPaperSpec,
) -> tuple[dict[str, bytes], dict[str, object], dict[str, object]]:
    payloads = _render_payloads(loaded, plan_sha256=plan_sha256, spec=spec)
    spec_bytes = _canonical_file_bytes(spec.to_mapping())
    manifest = _manifest_mapping(
        payloads,
        loaded,
        plan_sha256=plan_sha256,
        paper_spec_sha256=spec.sha256,
        publication_root_seal_sha256=publication_root_seal_sha256,
        spec=spec,
    )
    manifest_bytes = _canonical_file_bytes(manifest)
    seal = _seal_mapping(
        payloads,
        manifest_bytes=manifest_bytes,
        spec_bytes=spec_bytes,
        plan_sha256=plan_sha256,
        paper_spec_sha256=spec.sha256,
        publication_root_seal_sha256=publication_root_seal_sha256,
    )
    complete = dict(payloads)
    complete["paper-spec.json"] = spec_bytes
    complete["artifact_manifest.json"] = manifest_bytes
    complete["seal.json"] = _canonical_file_bytes(seal)
    return complete, manifest, seal


def _write_new(path: Path, payload: bytes) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                _integrity_fail(f"short write while creating {path.name}")
            remaining = remaining[written:]
        os.fsync(descriptor)
    except OSError as error:
        raise AAMASPaperIntegrityError(
            f"cannot exclusively create {path.name}: {error}"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _verify_membership(directory: Path) -> None:
    try:
        root = directory.lstat()
        entries = {item.name: item for item in directory.iterdir()}
    except OSError as error:
        raise AAMASPaperIntegrityError(
            f"cannot inspect paper artifact directory: {error}"
        ) from error
    if not stat.S_ISDIR(root.st_mode) or stat.S_ISLNK(root.st_mode):
        _integrity_fail("paper artifact root must be a real directory")
    if frozenset(entries) != _EXPECTED_FILES:
        _integrity_fail("paper artifact fixed membership changed")
    for name, path in entries.items():
        details = path.lstat()
        if (
            not stat.S_ISREG(details.st_mode)
            or stat.S_ISLNK(details.st_mode)
            or details.st_nlink != 1
        ):
            _integrity_fail(f"paper artifact is not singly linked: {name}")


def _audit_against_loaded(
    directory: Path,
    loaded: _LoadedSources,
    *,
    expected_plan_sha256: str,
    expected_publication_root_seal_sha256: str,
    expected_paper_spec: FrozenAAMASPaperSpec,
    expected_paper_spec_sha256: str,
    expected_paper_seal_sha256: str,
) -> tuple[dict[str, object], dict[str, object]]:
    if expected_paper_spec.sha256 != _sha256(
        expected_paper_spec_sha256, "expected_paper_spec_sha256"
    ):
        _integrity_fail("external paper spec digest differs from typed spec")
    _verify_membership(directory)
    expected, manifest, seal = _all_expected_bytes(
        loaded,
        plan_sha256=expected_plan_sha256,
        publication_root_seal_sha256=expected_publication_root_seal_sha256,
        spec=expected_paper_spec,
    )
    if sha256_bytes(_stable_regular_bytes(directory / "seal.json")) != _sha256(
        expected_paper_seal_sha256, "expected_paper_seal_sha256"
    ):
        _integrity_fail("paper seal differs from caller-retained digest")
    for name, expected_bytes in expected.items():
        observed = _stable_regular_bytes(directory / name)
        if observed != expected_bytes:
            _integrity_fail(f"paper artifact byte mismatch after re-render: {name}")
        if AAMAS_PAPER_ARTIFACT_LABEL.encode("ascii") not in observed:
            _integrity_fail(f"paper artifact lacks SYNTHETIC label: {name}")
    parsed_spec = FrozenAAMASPaperSpec.from_mapping(
        read_json_bytes_object(
            expected["paper-spec.json"], source="rendered paper spec"
        )
    )
    if parsed_spec != expected_paper_spec:
        _integrity_fail("stored paper spec differs from caller-held typed spec")
    if (
        read_json_bytes_object(
            expected["artifact_manifest.json"], source="artifact manifest"
        )
        != manifest
    ):
        _integrity_fail("artifact manifest is not canonical")
    if read_json_bytes_object(expected["seal.json"], source="paper seal") != seal:
        _integrity_fail("paper seal is not canonical")
    return manifest, seal


def _audit_paper_directory_snapshot(
    directory: Path,
    loaded: _LoadedSources,
    *,
    expected_plan_sha256: str,
    expected_publication_root_seal_sha256: str,
    expected_paper_spec: FrozenAAMASPaperSpec,
    expected_paper_spec_sha256: str,
    expected_paper_seal_sha256: str,
) -> tuple[dict[str, object], dict[str, object]]:
    def audit(target: Path) -> tuple[dict[str, object], dict[str, object]]:
        return _audit_against_loaded(
            target,
            loaded,
            expected_plan_sha256=expected_plan_sha256,
            expected_publication_root_seal_sha256=(
                expected_publication_root_seal_sha256
            ),
            expected_paper_spec=expected_paper_spec,
            expected_paper_spec_sha256=expected_paper_spec_sha256,
            expected_paper_seal_sha256=expected_paper_seal_sha256,
        )

    live_before = audit(directory)
    with tempfile.TemporaryDirectory(prefix="policytwin-aamas-paper-audit-") as tmp:
        snapshot = Path(tmp) / "paper"
        try:
            shutil.copytree(directory, snapshot, symlinks=True)
        except OSError as error:
            raise AAMASPaperIntegrityError(
                f"cannot create private paper snapshot: {error}"
            ) from error
        snapshot_before = audit(snapshot)
        if snapshot_before != live_before:
            _integrity_fail("private paper snapshot differs from live audit receipt")
        snapshot_after = audit(snapshot)
        if snapshot_after != snapshot_before:
            _integrity_fail("private paper snapshot changed during audit")
    live_after = audit(directory)
    if live_after != live_before:
        _integrity_fail("paper directory changed during private-snapshot audit")
    return live_after


@dataclass(frozen=True, slots=True)
class AAMASPaperWriteReceipt:
    directory: Path
    plan_sha256: str
    publication_root_seal_sha256: str
    paper_spec_sha256: str
    artifact_manifest_sha256: str
    paper_seal_sha256: str
    generated_output_count: int
    blocked_output_count: int
    synthetic_only: Literal[True] = True

    def __post_init__(self) -> None:
        if not isinstance(self.directory, Path):
            _fail("paper write receipt directory must be a Path")
        for field in (
            "plan_sha256",
            "publication_root_seal_sha256",
            "paper_spec_sha256",
            "artifact_manifest_sha256",
            "paper_seal_sha256",
        ):
            _sha256(getattr(self, field), f"paper receipt {field}")
        _integer(self.generated_output_count, "generated_output_count")
        _integer(self.blocked_output_count, "blocked_output_count")
        if self.synthetic_only is not True:
            _fail("paper write receipt cannot broaden synthetic-only authority")


@dataclass(frozen=True, slots=True)
class AAMASPaperAuditReport:
    integrity_valid: Literal[True]
    plan_sha256: str
    publication_root_seal_sha256: str
    paper_spec_sha256: str
    paper_seal_sha256: str
    generated_output_count: int
    blocked_outputs: tuple[str, ...]
    source_records: tuple[AAMASPaperSourceRecord, ...]
    publication_validity_level: Literal["abstain"]
    synthetic_only: Literal[True] = True

    def __post_init__(self) -> None:
        if self.integrity_valid is not True:
            _fail("paper audit report can only represent a passing audit")
        for field in (
            "plan_sha256",
            "publication_root_seal_sha256",
            "paper_spec_sha256",
            "paper_seal_sha256",
        ):
            _sha256(getattr(self, field), f"paper audit {field}")
        _integer(self.generated_output_count, "generated_output_count")
        if not all(
            isinstance(item, AAMASPaperSourceRecord) for item in self.source_records
        ):
            _fail("paper audit source records must be typed")
        if (
            self.publication_validity_level != "abstain"
            or self.synthetic_only is not True
        ):
            _fail("paper audit cannot authorize publication or real-data claims")


def write_aamas_paper_artifacts(
    publication_directory: Path,
    output_directory: Path,
    *,
    expected_plan: AAMASPublicationPlan,
    expected_plan_sha256: str,
    expected_study_commitment_sha256: str,
    expected_root_seal_sha256: str,
    expected_paper_spec: FrozenAAMASPaperSpec,
    expected_paper_spec_sha256: str,
    project_root: Path | None = None,
    allow_formal_scale: bool = False,
) -> AAMASPaperWriteReceipt:
    """Audit sources and atomically publish deterministic paper artifacts."""

    if not isinstance(publication_directory, Path) or not isinstance(
        output_directory, Path
    ):
        _fail("paper writer requires Path inputs")
    if not isinstance(expected_plan, AAMASPublicationPlan) or not isinstance(
        expected_paper_spec, FrozenAAMASPaperSpec
    ):
        _fail("paper writer requires typed frozen plan and paper spec")
    if expected_plan.sha256 != _sha256(expected_plan_sha256, "expected_plan_sha256"):
        _integrity_fail("external plan digest differs from typed publication plan")
    if expected_paper_spec.sha256 != _sha256(
        expected_paper_spec_sha256, "expected_paper_spec_sha256"
    ):
        _integrity_fail("external paper spec digest differs from typed spec")
    if os.path.lexists(output_directory):
        _integrity_fail("paper output directory already exists")
    loaded = _load_sources(
        publication_directory,
        expected_plan=expected_plan,
        expected_plan_sha256=expected_plan_sha256,
        expected_study_commitment_sha256=expected_study_commitment_sha256,
        expected_root_seal_sha256=expected_root_seal_sha256,
        project_root=project_root,
        allow_formal_scale=allow_formal_scale,
    )
    expected, manifest, _seal = _all_expected_bytes(
        loaded,
        plan_sha256=expected_plan_sha256,
        publication_root_seal_sha256=expected_root_seal_sha256,
        spec=expected_paper_spec,
    )
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_directory.name}.staging-",
            dir=output_directory.parent,
        )
    )
    paper_seal_sha256 = sha256_bytes(expected["seal.json"])
    published_identity: tuple[int, int] | None = None

    def audit_target(target: Path) -> tuple[dict[str, object], dict[str, object]]:
        return _audit_against_loaded(
            target,
            loaded,
            expected_plan_sha256=expected_plan_sha256,
            expected_publication_root_seal_sha256=expected_root_seal_sha256,
            expected_paper_spec=expected_paper_spec,
            expected_paper_spec_sha256=expected_paper_spec_sha256,
            expected_paper_seal_sha256=paper_seal_sha256,
        )

    try:
        for name in (*_FIGURE_FILES, *_TABLE_FILES, *_CONTROL_FILES):
            _write_new(staging / name, expected[name])
        audit_target(staging)
        # A second exact read closes the staging audit-to-rename window.
        audit_target(staging)
        os.replace(staging, output_directory)
        published = output_directory.lstat()
        published_identity = (published.st_dev, published.st_ino)
        _audit_paper_directory_snapshot(
            output_directory,
            loaded,
            expected_plan_sha256=expected_plan_sha256,
            expected_publication_root_seal_sha256=expected_root_seal_sha256,
            expected_paper_spec=expected_paper_spec,
            expected_paper_spec_sha256=expected_paper_spec_sha256,
            expected_paper_seal_sha256=paper_seal_sha256,
        )
    except BaseException:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
        if published_identity is not None and os.path.lexists(output_directory):
            current = output_directory.lstat()
            if (current.st_dev, current.st_ino) == published_identity:
                if stat.S_ISDIR(current.st_mode) and not stat.S_ISLNK(current.st_mode):
                    shutil.rmtree(output_directory)
                else:
                    output_directory.unlink()
        raise
    generated = cast(list[Mapping[str, object]], manifest["outputs"])
    return AAMASPaperWriteReceipt(
        directory=output_directory,
        plan_sha256=expected_plan_sha256,
        publication_root_seal_sha256=expected_root_seal_sha256,
        paper_spec_sha256=expected_paper_spec_sha256,
        artifact_manifest_sha256=sha256_bytes(expected["artifact_manifest.json"]),
        paper_seal_sha256=paper_seal_sha256,
        generated_output_count=sum(
            item["status"] == "generated_synthetic" for item in generated
        ),
        blocked_output_count=sum(
            item["status"] != "generated_synthetic" for item in generated
        ),
        synthetic_only=True,
    )


def audit_aamas_paper_artifacts(
    publication_directory: Path,
    paper_directory: Path,
    *,
    expected_plan: AAMASPublicationPlan,
    expected_plan_sha256: str,
    expected_study_commitment_sha256: str,
    expected_root_seal_sha256: str,
    expected_paper_spec: FrozenAAMASPaperSpec,
    expected_paper_spec_sha256: str,
    expected_paper_seal_sha256: str,
    project_root: Path | None = None,
    allow_formal_scale: bool = False,
) -> AAMASPaperAuditReport:
    """Re-audit sources, re-render every output, and compare exact bytes."""

    if not isinstance(publication_directory, Path) or not isinstance(
        paper_directory, Path
    ):
        _fail("paper audit requires Path inputs")
    if not isinstance(expected_plan, AAMASPublicationPlan) or not isinstance(
        expected_paper_spec, FrozenAAMASPaperSpec
    ):
        _fail("paper audit requires typed frozen plan and paper spec")
    if expected_plan.sha256 != _sha256(expected_plan_sha256, "expected_plan_sha256"):
        _integrity_fail("external plan digest differs from typed publication plan")
    loaded = _load_sources(
        publication_directory,
        expected_plan=expected_plan,
        expected_plan_sha256=expected_plan_sha256,
        expected_study_commitment_sha256=expected_study_commitment_sha256,
        expected_root_seal_sha256=expected_root_seal_sha256,
        project_root=project_root,
        allow_formal_scale=allow_formal_scale,
    )
    manifest, _seal = _audit_paper_directory_snapshot(
        paper_directory,
        loaded,
        expected_plan_sha256=expected_plan_sha256,
        expected_publication_root_seal_sha256=expected_root_seal_sha256,
        expected_paper_spec=expected_paper_spec,
        expected_paper_spec_sha256=expected_paper_spec_sha256,
        expected_paper_seal_sha256=expected_paper_seal_sha256,
    )
    outputs = cast(list[Mapping[str, object]], manifest["outputs"])
    blocked = tuple(
        cast(str, item["output_id"])
        for item in outputs
        if item["status"] != "generated_synthetic"
    )
    return AAMASPaperAuditReport(
        integrity_valid=True,
        plan_sha256=expected_plan_sha256,
        publication_root_seal_sha256=expected_root_seal_sha256,
        paper_spec_sha256=expected_paper_spec_sha256,
        paper_seal_sha256=expected_paper_seal_sha256,
        generated_output_count=sum(
            item["status"] == "generated_synthetic" for item in outputs
        ),
        blocked_outputs=blocked,
        source_records=loaded.records,
        publication_validity_level="abstain",
        synthetic_only=True,
    )


__all__ = [
    "AAMAS_PAPER_ARTIFACT_LABEL",
    "AAMAS_PAPER_MANIFEST_SCHEMA_VERSION",
    "AAMAS_PAPER_SEAL_SCHEMA_VERSION",
    "AAMAS_PAPER_SPEC_SCHEMA_VERSION",
    "AAMASPaperAuditReport",
    "AAMASPaperError",
    "AAMASPaperIntegrityError",
    "AAMASPaperSourceRecord",
    "AAMASPaperWriteReceipt",
    "FrozenAAMASPaperSpec",
    "audit_aamas_paper_artifacts",
    "write_aamas_paper_artifacts",
]
