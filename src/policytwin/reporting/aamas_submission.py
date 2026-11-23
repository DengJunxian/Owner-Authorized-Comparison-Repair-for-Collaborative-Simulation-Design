"""Frozen, source-audited derivation for the synthetic AAMAS submission.

The functions in this module never estimate a model.  They first invoke the
verifier-owned audits for the RQ1 benchmark and the RQ2/RQ3 calibration
collection, then derive reporting rows from those sealed source bytes.  The
result remains synthetic-only and deliberately exposes an ABSTAIN hand-off for
the umbrella validity certificate rather than inventing missing evidence.
"""

from __future__ import annotations

import math
import os
import shutil
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from statistics import fmean
from typing import Literal, NoReturn, Self, cast

from policytwin.counterfactual import EventType, HashBasedRNG, SemanticEventKey
from policytwin.evaluation import ValidityEvidence, ValidityLevel
from policytwin.experiments.aamas_benchmark import (
    AAMASAttemptKind,
    AAMASAttemptStatus,
    AAMASBenchmarkArtifact,
    AAMASBenchmarkPlan,
    AAMASSourceKind,
    audit_aamas_benchmark,
)
from policytwin.experiments.aamas_calibration import (
    AAMASCalibrationPlan,
    AAMASCalibrationSpec,
    AttemptStatus,
    TruthRole,
    audit_aamas_calibration_collection,
    build_aamas_calibration_plan,
)
from policytwin.experiments.mechanism_ablations import (
    MechanismAblationStatus,
    MechanismAblationVariant,
)
from policytwin.experiments.mechanism_smoke import (
    MechanismSmokeArtifact,
    RNGCouplingMode,
)
from policytwin.provenance import (
    MetadataValidationError,
    canonical_json_bytes,
    read_json_object,
    sha256_bytes,
    sha256_file,
    write_json_file,
    write_text_file,
)

AAMAS_REPORTING_SPEC_SCHEMA_VERSION = "policytwin.aamas-reporting-spec.v1"
AAMAS_SUBMISSION_ARTIFACT_SCHEMA_VERSION = "policytwin.aamas-submission.v1"
AAMAS_SUBMISSION_CLAIM_SCOPE = "synthetic_aamas_submission_diagnostics_only"
AAMAS_SUBMISSION_FILENAME = "aamas-submission.json"
AAMAS_SUBMISSION_SEAL_FILENAME = "aamas-submission.sha256"

_SHA256_LENGTH = 64
_WILSON_Z_HEX = "0x1.f5c28f5c28f5cp+0"
_WILSON_Z = float.fromhex(_WILSON_Z_HEX)


class AAMASReportingError(RuntimeError):
    """Base class for submission-derivation failures."""


class AAMASReportingSchemaError(AAMASReportingError, ValueError):
    """Raised when a reporting spec or artifact has an invalid schema."""


class AAMASReportingIntegrityError(AAMASReportingError):
    """Raised when audited sources or deterministic re-derivation disagree."""


def _fail(message: str) -> NoReturn:
    raise AAMASReportingSchemaError(message)


def _integrity_fail(message: str) -> NoReturn:
    raise AAMASReportingIntegrityError(message)


def _strict_fields(
    value: Mapping[str, object], expected: frozenset[str], context: str
) -> None:
    observed = frozenset(value)
    if observed != expected:
        _fail(
            f"{context} fields differ: missing={sorted(expected - observed)!r}, "
            f"extra={sorted(observed - expected)!r}"
        )


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        _fail(f"{field} must be an object with string keys")
    return cast(Mapping[str, object], value)


def _array(value: object, field: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        _fail(f"{field} must be an array")
    return cast(Sequence[object], value)


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{field} must be a non-empty string")
    return value


def _optional_string(value: object, field: str) -> str | None:
    if value is None:
        return None
    return _string(value, field)


def _integer(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"{field} must be an integer >= {minimum}")
    return value


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        _fail(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{field} must be finite")
    return result


def _optional_finite(value: object, field: str) -> float | None:
    return None if value is None else _finite(value, field)


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        _fail(f"{field} must be boolean")
    return value


def _sha256(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{field} must be a lowercase SHA-256 digest")
    return value


def _optional_sha256(value: object, field: str) -> str | None:
    return None if value is None else _sha256(value, field)


def _fraction(value: object, field: str) -> float:
    result = _finite(value, field)
    if not 0.0 <= result <= 1.0:
        _fail(f"{field} must lie in [0, 1]")
    return result


def _quantile(values: Sequence[float], probability: float) -> float:
    """Frozen linear interpolation at ``p * (n - 1)`` (R type 7)."""

    if not values:
        _integrity_fail("a quantile requires at least one source value")
    ordered = sorted(float(item) for item in values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    interpolated = ordered[lower] + (ordered[upper] - ordered[lower]) * fraction
    return min(ordered[upper], max(ordered[lower], interpolated))


def _sample_variance(values: Sequence[float]) -> float:
    if len(values) < 2:
        _integrity_fail("sample variance with ddof=1 requires at least two values")
    mean = fmean(values)
    return sum((value - mean) ** 2 for value in values) / (len(values) - 1)


def _semantic_hash(value: object) -> str:
    return sha256_bytes(canonical_json_bytes(value))


class ScalarStatus(StrEnum):
    """Why a registered scalar is or is not available."""

    DEFINED = "defined"
    BLOCKED_SOURCE_FAILURE = "blocked_source_failure"
    UNDEFINED_ZERO_DENOMINATOR = "undefined_zero_denominator"
    UNDEFINED_MINIMUM_RETAINED = "undefined_minimum_retained"
    UNDEFINED_INSUFFICIENT_SAMPLE = "undefined_insufficient_sample"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True, slots=True)
class ReportedScalar:
    """A finite number or an explicit typed reason for its absence."""

    status: ScalarStatus
    value: int | float | None
    reason: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.status, ScalarStatus):
            _fail("reported scalar status is invalid")
        if self.status is ScalarStatus.DEFINED:
            if isinstance(self.value, bool) or not isinstance(self.value, int | float):
                _fail("defined scalar must contain a number")
            if not math.isfinite(float(self.value)) or self.reason is not None:
                _fail("defined scalar must be finite and have no reason")
        elif (
            self.value is not None
            or not isinstance(self.reason, str)
            or not self.reason
        ):
            _fail("undefined scalar must contain null plus a non-empty reason")

    @classmethod
    def defined(cls, value: int | float) -> ReportedScalar:
        return cls(ScalarStatus.DEFINED, value, None)

    @classmethod
    def unavailable(cls, status: ScalarStatus, reason: str) -> ReportedScalar:
        if status is ScalarStatus.DEFINED:
            _fail("unavailable scalar cannot use defined status")
        return cls(status, None, reason)

    def to_mapping(self) -> dict[str, object]:
        return {"reason": self.reason, "status": self.status.value, "value": self.value}

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(value, frozenset({"reason", "status", "value"}), "scalar")
        try:
            status = ScalarStatus(_string(value["status"], "scalar status"))
        except ValueError as error:
            raise AAMASReportingSchemaError("unsupported scalar status") from error
        raw_value = value["value"]
        if raw_value is not None and (
            isinstance(raw_value, bool) or not isinstance(raw_value, int | float)
        ):
            _fail("scalar value must be numeric or null")
        return cls(
            status=status,
            value=raw_value,
            reason=_optional_string(value["reason"], "scalar reason"),
        )


@dataclass(frozen=True, slots=True)
class TypedInterval:
    """Named interval with semantics separate from posterior/confidence claims."""

    status: ScalarStatus
    interval_kind: str
    nominal_probability: float
    lower: float | None
    upper: float | None
    reason: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.status, ScalarStatus):
            _fail("typed interval status is invalid")
        _string(self.interval_kind, "interval_kind")
        probability = _fraction(self.nominal_probability, "nominal_probability")
        object.__setattr__(self, "nominal_probability", probability)
        if self.status is ScalarStatus.DEFINED:
            if self.lower is None or self.upper is None:
                _fail("defined interval requires both endpoints")
            lower = _finite(self.lower, "interval lower")
            upper = _finite(self.upper, "interval upper")
            if lower > upper or self.reason is not None:
                _fail("defined interval endpoints or reason are inconsistent")
            object.__setattr__(self, "lower", lower)
            object.__setattr__(self, "upper", upper)
        elif (
            self.lower is not None
            or self.upper is not None
            or not isinstance(self.reason, str)
            or not self.reason
        ):
            _fail("undefined interval must retain only a typed reason")

    @classmethod
    def defined(
        cls, kind: str, nominal: float, lower: float, upper: float
    ) -> TypedInterval:
        return cls(ScalarStatus.DEFINED, kind, nominal, lower, upper, None)

    @classmethod
    def unavailable(
        cls, status: ScalarStatus, kind: str, nominal: float, reason: str
    ) -> TypedInterval:
        return cls(status, kind, nominal, None, None, reason)

    def to_mapping(self) -> dict[str, object]:
        return {
            "interval_kind": self.interval_kind,
            "lower": self.lower,
            "nominal_probability": self.nominal_probability,
            "reason": self.reason,
            "status": self.status.value,
            "upper": self.upper,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(
            value,
            frozenset(
                {
                    "interval_kind",
                    "lower",
                    "nominal_probability",
                    "reason",
                    "status",
                    "upper",
                }
            ),
            "typed interval",
        )
        try:
            status = ScalarStatus(_string(value["status"], "interval status"))
        except ValueError as error:
            raise AAMASReportingSchemaError("unsupported interval status") from error
        return cls(
            status=status,
            interval_kind=_string(value["interval_kind"], "interval kind"),
            nominal_probability=_fraction(
                value["nominal_probability"], "nominal probability"
            ),
            lower=_optional_finite(value["lower"], "interval lower"),
            upper=_optional_finite(value["upper"], "interval upper"),
            reason=_optional_string(value["reason"], "interval reason"),
        )


@dataclass(frozen=True, slots=True)
class FrozenAAMASReportingSpec:
    """Complete immutable definitions for every submission statistic."""

    derivation_id: str = "policytwin.aamas.synthetic-submission.v1"
    rng_variance_ddof: int = 1
    rng_mcse_formula: str = "sqrt(sample_variance_ddof_1/n)"
    exact_effect_equality: str = "exact_binary64_equality_to_frozen_policy_effect"
    shock_match_unit: str = "exogenous_step_draw"
    mechanism_horizon_id: str = "terminal"
    mechanism_group_id: str = "all-latent-strategies"
    mechanism_metric_name: str = "quoted_spread"
    quantile_probabilities: tuple[float, float, float] = (0.05, 0.5, 0.95)
    quantile_method: str = "linear_interpolation_at_p_times_n_minus_1"
    calibration_roles: tuple[str, str] = ("design", "target")
    calibration_parameters: tuple[str, str] = ("liquidity", "signal_strength")
    held_out_statistics: tuple[str, str, str] = (
        "macro_volatility",
        "meso_spread",
        "micro_order_size",
    )
    aggregate_failure_rule: str = "any_expected_attempt_failure_blocks_all_aggregates"
    formal_design_truth_count: int = 32
    formal_target_truth_count: int = 256
    formal_minimum_retained: int = 64
    coverage_interval_kind: str = "binomial_wilson_interval"
    coverage_interval_nominal_probability: float = 0.95
    coverage_interval_z_hex: str = _WILSON_Z_HEX
    sbc_reporting_rule: str = (
        "complete_rank_counts_and_interval_coverage_only_no_post_hoc_significance_tests"
    )
    bootstrap_resample_count: int = 1000
    bootstrap_seed: str = "policytwin-aamas-target-bootstrap-v1"
    bootstrap_domain: str = "policytwin.reporting.aamas.target_resample.v1"
    bootstrap_interval_kind: str = "synthetic_target_resampling_interval"
    bootstrap_quantiles: tuple[float, float] = (0.05, 0.95)
    bootstrap_event_timestamp: datetime = datetime(2024, 12, 1, tzinfo=UTC)
    risk_record_role: str = "target"
    validity_entry_rule: str = (
        "source_derived_evidence_only_missing_authority_forces_abstain"
    )
    synthetic_only: Literal[True] = True
    schema_version: str = AAMAS_REPORTING_SPEC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.derivation_id != "policytwin.aamas.synthetic-submission.v1":
            _fail("AAMAS reporting derivation identity changed")
        if self.rng_variance_ddof != 1:
            _fail("AAMAS reporting freezes sample variance at ddof=1")
        if self.rng_mcse_formula != "sqrt(sample_variance_ddof_1/n)":
            _fail("AAMAS reporting MCSE formula changed")
        if (
            self.exact_effect_equality
            != "exact_binary64_equality_to_frozen_policy_effect"
            or self.shock_match_unit != "exogenous_step_draw"
        ):
            _fail("RNG equality or shock-match definition changed")
        if (
            self.mechanism_horizon_id,
            self.mechanism_group_id,
            self.mechanism_metric_name,
        ) != ("terminal", "all-latent-strategies", "quoted_spread"):
            _fail("registered mechanism output cell changed")
        if self.quantile_probabilities != (0.05, 0.5, 0.95):
            _fail("AAMAS reporting quantiles are frozen at 0.05, 0.5, and 0.95")
        if self.quantile_method != "linear_interpolation_at_p_times_n_minus_1":
            _fail("AAMAS reporting quantile method changed")
        if self.bootstrap_quantiles != (0.05, 0.95):
            _fail("AAMAS bootstrap quantiles are frozen at 0.05 and 0.95")
        if self.calibration_roles != ("design", "target"):
            _fail("calibration roles must remain design and target")
        if self.calibration_parameters != ("liquidity", "signal_strength"):
            _fail("calibration parameter registry changed")
        if self.held_out_statistics != (
            "macro_volatility",
            "meso_spread",
            "micro_order_size",
        ):
            _fail("held-out statistic registry changed")
        if (
            self.aggregate_failure_rule
            != "any_expected_attempt_failure_blocks_all_aggregates"
        ):
            _fail("aggregate failure propagation rule changed")
        for field in (
            "formal_design_truth_count",
            "formal_target_truth_count",
            "formal_minimum_retained",
            "bootstrap_resample_count",
        ):
            _integer(getattr(self, field), field, minimum=1)
        if self.formal_minimum_retained > self.formal_target_truth_count:
            _fail("formal minimum_retained exceeds target count")
        if (
            self.formal_design_truth_count,
            self.formal_target_truth_count,
            self.formal_minimum_retained,
        ) != (32, 256, 64):
            _fail("formal calibration population or retention boundary changed")
        if self.coverage_interval_kind != "binomial_wilson_interval":
            _fail("recovery coverage interval kind changed")
        if self.coverage_interval_nominal_probability != 0.95:
            _fail("Wilson score nominal probability changed")
        if self.coverage_interval_z_hex != _WILSON_Z_HEX:
            _fail("Wilson score z constant changed")
        if self.sbc_reporting_rule != (
            "complete_rank_counts_and_interval_coverage_only_"
            "no_post_hoc_significance_tests"
        ):
            _fail("SBC reporting rule changed")
        if (
            self.bootstrap_resample_count != 1000
            or self.bootstrap_seed != "policytwin-aamas-target-bootstrap-v1"
            or self.bootstrap_domain != "policytwin.reporting.aamas.target_resample.v1"
        ):
            _fail("bootstrap count, seed, or domain changed")
        if self.bootstrap_interval_kind != "synthetic_target_resampling_interval":
            _fail("risk interval must remain a synthetic resampling interval")
        if self.risk_record_role != TruthRole.TARGET.value:
            _fail("risk records must remain target-only")
        timestamp = self.bootstrap_event_timestamp
        if not isinstance(timestamp, datetime) or timestamp.tzinfo is None:
            _fail("bootstrap_event_timestamp must be timezone-aware")
        object.__setattr__(self, "bootstrap_event_timestamp", timestamp.astimezone(UTC))
        if self.bootstrap_event_timestamp != datetime(2024, 12, 1, tzinfo=UTC):
            _fail("bootstrap EventKey timestamp changed")
        if (
            self.validity_entry_rule
            != "source_derived_evidence_only_missing_authority_forces_abstain"
        ):
            _fail("validity ABSTAIN hand-off rule changed")
        if self.synthetic_only is not True:
            _fail("AAMAS submission reporting is synthetic-only")
        if self.schema_version != AAMAS_REPORTING_SPEC_SCHEMA_VERSION:
            _fail("unsupported AAMAS reporting spec schema")

    @property
    def sha256(self) -> str:
        return _semantic_hash(self.to_mapping())

    def to_mapping(self) -> dict[str, object]:
        return {
            "aggregate_failure_rule": self.aggregate_failure_rule,
            "bootstrap_domain": self.bootstrap_domain,
            "bootstrap_event_timestamp": self.bootstrap_event_timestamp.isoformat(),
            "bootstrap_interval_kind": self.bootstrap_interval_kind,
            "bootstrap_quantiles": list(self.bootstrap_quantiles),
            "bootstrap_resample_count": self.bootstrap_resample_count,
            "bootstrap_seed": self.bootstrap_seed,
            "calibration_parameters": list(self.calibration_parameters),
            "calibration_roles": list(self.calibration_roles),
            "coverage_interval_kind": self.coverage_interval_kind,
            "coverage_interval_nominal_probability": (
                self.coverage_interval_nominal_probability
            ),
            "coverage_interval_z_hex": self.coverage_interval_z_hex,
            "derivation_id": self.derivation_id,
            "exact_effect_equality": self.exact_effect_equality,
            "formal_design_truth_count": self.formal_design_truth_count,
            "formal_minimum_retained": self.formal_minimum_retained,
            "formal_target_truth_count": self.formal_target_truth_count,
            "held_out_statistics": list(self.held_out_statistics),
            "mechanism_group_id": self.mechanism_group_id,
            "mechanism_horizon_id": self.mechanism_horizon_id,
            "mechanism_metric_name": self.mechanism_metric_name,
            "quantile_method": self.quantile_method,
            "quantile_probabilities": list(self.quantile_probabilities),
            "risk_record_role": self.risk_record_role,
            "rng_mcse_formula": self.rng_mcse_formula,
            "rng_variance_ddof": self.rng_variance_ddof,
            "sbc_reporting_rule": self.sbc_reporting_rule,
            "schema_version": self.schema_version,
            "shock_match_unit": self.shock_match_unit,
            "synthetic_only": self.synthetic_only,
            "validity_entry_rule": self.validity_entry_rule,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        expected = frozenset(cls().to_mapping())
        _strict_fields(value, expected, "AAMAS reporting spec")
        if value["synthetic_only"] is not True:
            _fail("reporting spec must remain synthetic-only")

        def strings(field: str) -> tuple[str, ...]:
            return tuple(
                _string(item, f"{field} item") for item in _array(value[field], field)
            )

        try:
            timestamp = datetime.fromisoformat(
                _string(value["bootstrap_event_timestamp"], "bootstrap timestamp")
            )
        except ValueError as error:
            raise AAMASReportingSchemaError("invalid bootstrap timestamp") from error
        result = cls(
            derivation_id=_string(value["derivation_id"], "derivation_id"),
            rng_variance_ddof=_integer(value["rng_variance_ddof"], "rng_variance_ddof"),
            rng_mcse_formula=_string(value["rng_mcse_formula"], "rng_mcse_formula"),
            exact_effect_equality=_string(
                value["exact_effect_equality"], "exact_effect_equality"
            ),
            shock_match_unit=_string(value["shock_match_unit"], "shock_match_unit"),
            mechanism_horizon_id=_string(
                value["mechanism_horizon_id"], "mechanism_horizon_id"
            ),
            mechanism_group_id=_string(
                value["mechanism_group_id"], "mechanism_group_id"
            ),
            mechanism_metric_name=_string(
                value["mechanism_metric_name"], "mechanism_metric_name"
            ),
            quantile_probabilities=cast(
                tuple[float, float, float],
                tuple(
                    _fraction(item, "quantile probability")
                    for item in _array(
                        value["quantile_probabilities"], "quantile_probabilities"
                    )
                ),
            ),
            quantile_method=_string(value["quantile_method"], "quantile_method"),
            calibration_roles=cast(tuple[str, str], strings("calibration_roles")),
            calibration_parameters=cast(
                tuple[str, str], strings("calibration_parameters")
            ),
            held_out_statistics=cast(
                tuple[str, str, str], strings("held_out_statistics")
            ),
            aggregate_failure_rule=_string(
                value["aggregate_failure_rule"], "aggregate_failure_rule"
            ),
            formal_design_truth_count=_integer(
                value["formal_design_truth_count"],
                "formal_design_truth_count",
                minimum=1,
            ),
            formal_target_truth_count=_integer(
                value["formal_target_truth_count"],
                "formal_target_truth_count",
                minimum=1,
            ),
            formal_minimum_retained=_integer(
                value["formal_minimum_retained"], "formal_minimum_retained", minimum=1
            ),
            coverage_interval_kind=_string(
                value["coverage_interval_kind"], "coverage_interval_kind"
            ),
            coverage_interval_nominal_probability=_fraction(
                value["coverage_interval_nominal_probability"],
                "coverage_interval_nominal_probability",
            ),
            coverage_interval_z_hex=_string(
                value["coverage_interval_z_hex"], "coverage_interval_z_hex"
            ),
            sbc_reporting_rule=_string(
                value["sbc_reporting_rule"], "sbc_reporting_rule"
            ),
            bootstrap_resample_count=_integer(
                value["bootstrap_resample_count"], "bootstrap_resample_count", minimum=1
            ),
            bootstrap_seed=_string(value["bootstrap_seed"], "bootstrap_seed"),
            bootstrap_domain=_string(value["bootstrap_domain"], "bootstrap_domain"),
            bootstrap_interval_kind=_string(
                value["bootstrap_interval_kind"], "bootstrap_interval_kind"
            ),
            bootstrap_quantiles=cast(
                tuple[float, float],
                tuple(
                    _fraction(item, "bootstrap quantile")
                    for item in _array(
                        value["bootstrap_quantiles"], "bootstrap_quantiles"
                    )
                ),
            ),
            bootstrap_event_timestamp=timestamp,
            risk_record_role=_string(value["risk_record_role"], "risk_record_role"),
            validity_entry_rule=_string(
                value["validity_entry_rule"], "validity_entry_rule"
            ),
            synthetic_only=True,
            schema_version=_string(value["schema_version"], "schema_version"),
        )
        if result.to_mapping() != dict(value):
            _fail("reporting spec is not in canonical serialized form")
        return result


@dataclass(frozen=True, slots=True)
class AAMASReportingSources:
    """Caller-independent commitments returned by the two source audits."""

    rq1_artifact_sha256: str
    rq1_plan_sha256: str
    rq1_audited_source_count: int
    calibration_seal_raw_sha256: str
    calibration_spec_sha256: str
    calibration_plan_sha256: str
    calibration_results_sha256: str
    calibration_expected_attempt_count: int
    calibration_completed_attempt_count: int

    def __post_init__(self) -> None:
        for field in (
            "rq1_artifact_sha256",
            "rq1_plan_sha256",
            "calibration_seal_raw_sha256",
            "calibration_spec_sha256",
            "calibration_plan_sha256",
            "calibration_results_sha256",
        ):
            _sha256(getattr(self, field), field)
        for field in (
            "rq1_audited_source_count",
            "calibration_expected_attempt_count",
            "calibration_completed_attempt_count",
        ):
            _integer(getattr(self, field), field)
        if (
            self.calibration_completed_attempt_count
            > self.calibration_expected_attempt_count
        ):
            _fail("completed calibration attempts exceed expected attempts")

    def to_mapping(self) -> dict[str, object]:
        return {
            "calibration_completed_attempt_count": (
                self.calibration_completed_attempt_count
            ),
            "calibration_expected_attempt_count": (
                self.calibration_expected_attempt_count
            ),
            "calibration_plan_sha256": self.calibration_plan_sha256,
            "calibration_results_sha256": self.calibration_results_sha256,
            "calibration_seal_raw_sha256": self.calibration_seal_raw_sha256,
            "calibration_spec_sha256": self.calibration_spec_sha256,
            "rq1_artifact_sha256": self.rq1_artifact_sha256,
            "rq1_audited_source_count": self.rq1_audited_source_count,
            "rq1_plan_sha256": self.rq1_plan_sha256,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(value, frozenset(cls.__annotations__), "reporting sources")
        return cls(
            rq1_artifact_sha256=_sha256(
                value["rq1_artifact_sha256"], "rq1_artifact_sha256"
            ),
            rq1_plan_sha256=_sha256(value["rq1_plan_sha256"], "rq1_plan_sha256"),
            rq1_audited_source_count=_integer(
                value["rq1_audited_source_count"], "rq1_audited_source_count"
            ),
            calibration_seal_raw_sha256=_sha256(
                value["calibration_seal_raw_sha256"],
                "calibration_seal_raw_sha256",
            ),
            calibration_spec_sha256=_sha256(
                value["calibration_spec_sha256"], "calibration_spec_sha256"
            ),
            calibration_plan_sha256=_sha256(
                value["calibration_plan_sha256"], "calibration_plan_sha256"
            ),
            calibration_results_sha256=_sha256(
                value["calibration_results_sha256"], "calibration_results_sha256"
            ),
            calibration_expected_attempt_count=_integer(
                value["calibration_expected_attempt_count"],
                "calibration_expected_attempt_count",
            ),
            calibration_completed_attempt_count=_integer(
                value["calibration_completed_attempt_count"],
                "calibration_completed_attempt_count",
            ),
        )


@dataclass(frozen=True, slots=True)
class RQ1RNGRow:
    """One preregistered stress-cell by RNG-mode reporting row."""

    cell_id: str
    mode: str
    branch_step: int
    branch_draw_count: int
    expected: int
    completed: int
    failed: int
    paired_differences: tuple[float, ...]
    mean_paired_difference: ReportedScalar
    sample_variance: ReportedScalar
    mcse: ReportedScalar
    bias_from_policy_effect: ReportedScalar
    exact_effect_rate: ReportedScalar
    all_shock_match_rate: ReportedScalar
    post_branch_shock_match_rate: ReportedScalar
    variance_ratio_vs_independent: ReportedScalar
    baseline_draws_sha256: str | None
    event_keys_sha256: str | None
    source_artifact_sha256: str | None

    def __post_init__(self) -> None:
        _string(self.cell_id, "RNG cell_id")
        try:
            RNGCouplingMode(self.mode)
        except ValueError as error:
            raise AAMASReportingSchemaError("unknown RNG mode") from error
        _integer(self.branch_step, "branch_step")
        _integer(self.branch_draw_count, "branch_draw_count", minimum=1)
        for field in ("expected", "completed", "failed"):
            _integer(getattr(self, field), field)
        if self.completed + self.failed != self.expected:
            _fail("RNG terminal counts do not equal expected replicates")
        if len(self.paired_differences) != self.completed:
            _fail("RNG paired draws do not equal completed replicate count")
        if any(not math.isfinite(item) for item in self.paired_differences):
            _fail("RNG paired differences must be finite")
        for field in (
            "mean_paired_difference",
            "sample_variance",
            "mcse",
            "bias_from_policy_effect",
            "exact_effect_rate",
            "all_shock_match_rate",
            "post_branch_shock_match_rate",
            "variance_ratio_vs_independent",
        ):
            if not isinstance(getattr(self, field), ReportedScalar):
                _fail(f"{field} must be a ReportedScalar")
        hashes = (
            self.baseline_draws_sha256,
            self.event_keys_sha256,
            self.source_artifact_sha256,
        )
        if self.completed:
            if self.failed or any(item is None for item in hashes):
                _fail("completed RNG row has inconsistent failure or source hashes")
            for digest in hashes:
                _sha256(digest, "RNG source digest")
        elif any(item is not None for item in hashes):
            _fail("failed RNG row cannot claim source hashes")

    def to_mapping(self) -> dict[str, object]:
        return {
            "all_shock_match_rate": self.all_shock_match_rate.to_mapping(),
            "baseline_draws_sha256": self.baseline_draws_sha256,
            "bias_from_policy_effect": self.bias_from_policy_effect.to_mapping(),
            "branch_draw_count": self.branch_draw_count,
            "branch_step": self.branch_step,
            "cell_id": self.cell_id,
            "completed": self.completed,
            "event_keys_sha256": self.event_keys_sha256,
            "exact_effect_rate": self.exact_effect_rate.to_mapping(),
            "expected": self.expected,
            "failed": self.failed,
            "mcse": self.mcse.to_mapping(),
            "mean_paired_difference": self.mean_paired_difference.to_mapping(),
            "mode": self.mode,
            "paired_differences": list(self.paired_differences),
            "post_branch_shock_match_rate": (
                self.post_branch_shock_match_rate.to_mapping()
            ),
            "sample_variance": self.sample_variance.to_mapping(),
            "source_artifact_sha256": self.source_artifact_sha256,
            "variance_ratio_vs_independent": (
                self.variance_ratio_vs_independent.to_mapping()
            ),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(value, frozenset(cls.__annotations__), "RQ1 RNG row")
        return cls(
            cell_id=_string(value["cell_id"], "cell_id"),
            mode=_string(value["mode"], "mode"),
            branch_step=_integer(value["branch_step"], "branch_step"),
            branch_draw_count=_integer(
                value["branch_draw_count"], "branch_draw_count", minimum=1
            ),
            expected=_integer(value["expected"], "expected"),
            completed=_integer(value["completed"], "completed"),
            failed=_integer(value["failed"], "failed"),
            paired_differences=tuple(
                _finite(item, "paired difference")
                for item in _array(value["paired_differences"], "paired_differences")
            ),
            mean_paired_difference=ReportedScalar.from_mapping(
                _mapping(value["mean_paired_difference"], "mean_paired_difference")
            ),
            sample_variance=ReportedScalar.from_mapping(
                _mapping(value["sample_variance"], "sample_variance")
            ),
            mcse=ReportedScalar.from_mapping(_mapping(value["mcse"], "mcse")),
            bias_from_policy_effect=ReportedScalar.from_mapping(
                _mapping(value["bias_from_policy_effect"], "bias_from_policy_effect")
            ),
            exact_effect_rate=ReportedScalar.from_mapping(
                _mapping(value["exact_effect_rate"], "exact_effect_rate")
            ),
            all_shock_match_rate=ReportedScalar.from_mapping(
                _mapping(value["all_shock_match_rate"], "all_shock_match_rate")
            ),
            post_branch_shock_match_rate=ReportedScalar.from_mapping(
                _mapping(
                    value["post_branch_shock_match_rate"],
                    "post_branch_shock_match_rate",
                )
            ),
            variance_ratio_vs_independent=ReportedScalar.from_mapping(
                _mapping(
                    value["variance_ratio_vs_independent"],
                    "variance_ratio_vs_independent",
                )
            ),
            baseline_draws_sha256=_optional_sha256(
                value["baseline_draws_sha256"], "baseline_draws_sha256"
            ),
            event_keys_sha256=_optional_sha256(
                value["event_keys_sha256"], "event_keys_sha256"
            ),
            source_artifact_sha256=_optional_sha256(
                value["source_artifact_sha256"], "source_artifact_sha256"
            ),
        )


@dataclass(frozen=True, slots=True)
class RQ1BaselineEqualityInvariant:
    """Cross-cell equality of baseline exogenous draws for one RNG mode."""

    mode: str
    cell_ids: tuple[str, ...]
    baseline_draws_sha256: str | None
    event_keys_sha256: str | None
    passed: bool | None
    reason: str | None

    def __post_init__(self) -> None:
        try:
            RNGCouplingMode(self.mode)
        except ValueError as error:
            raise AAMASReportingSchemaError("unknown invariant RNG mode") from error
        if not self.cell_ids or len(set(self.cell_ids)) != len(self.cell_ids):
            _fail("baseline invariant cell registry is invalid")
        if self.passed is True:
            _sha256(self.baseline_draws_sha256, "baseline invariant draw digest")
            _sha256(self.event_keys_sha256, "baseline invariant key digest")
            if self.reason is not None:
                _fail("passing baseline invariant cannot retain a reason")
        elif self.passed is None:
            if (
                self.baseline_draws_sha256 is not None
                or self.event_keys_sha256 is not None
                or not isinstance(self.reason, str)
                or not self.reason
            ):
                _fail("blocked baseline invariant must retain only a reason")
        else:
            _fail("a failed cross-cell equality check cannot be published")

    def to_mapping(self) -> dict[str, object]:
        return {
            "baseline_draws_sha256": self.baseline_draws_sha256,
            "cell_ids": list(self.cell_ids),
            "event_keys_sha256": self.event_keys_sha256,
            "mode": self.mode,
            "passed": self.passed,
            "reason": self.reason,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(value, frozenset(cls.__annotations__), "baseline invariant")
        raw_passed = value["passed"]
        if raw_passed is not None and not isinstance(raw_passed, bool):
            _fail("baseline invariant passed must be boolean or null")
        return cls(
            mode=_string(value["mode"], "mode"),
            cell_ids=tuple(
                _string(item, "cell_id")
                for item in _array(value["cell_ids"], "cell_ids")
            ),
            baseline_draws_sha256=_optional_sha256(
                value["baseline_draws_sha256"], "baseline_draws_sha256"
            ),
            event_keys_sha256=_optional_sha256(
                value["event_keys_sha256"], "event_keys_sha256"
            ),
            passed=raw_passed,
            reason=_optional_string(value["reason"], "reason"),
        )


@dataclass(frozen=True, slots=True)
class RQ1MechanismRow:
    """One expected mechanism variant/replicate row, including failures."""

    variant: str
    replicate_index: int
    status: str
    event_key: str
    metric_value: float | None
    failure_stage: str | None
    failure_type: str | None
    failure_message: str | None
    paired_outcome_sha256: str | None

    def __post_init__(self) -> None:
        try:
            MechanismAblationVariant(self.variant)
            status = MechanismAblationStatus(self.status)
        except ValueError as error:
            raise AAMASReportingSchemaError("unknown mechanism row enum") from error
        _integer(self.replicate_index, "replicate_index")
        if not self.event_key.startswith("ptk1:"):
            _fail("mechanism row event key is invalid")
        if status is MechanismAblationStatus.COMPLETED:
            _finite(self.metric_value, "mechanism metric value")
            if any(
                item is not None
                for item in (
                    self.failure_stage,
                    self.failure_type,
                    self.failure_message,
                )
            ):
                _fail("completed mechanism row cannot retain a failure")
            _sha256(self.paired_outcome_sha256, "paired_outcome_sha256")
        else:
            if self.metric_value is not None or self.paired_outcome_sha256 is not None:
                _fail("failed mechanism row cannot claim an output")
            for item in (self.failure_stage, self.failure_type, self.failure_message):
                _string(item, "mechanism failure field")

    def to_mapping(self) -> dict[str, object]:
        return {
            "event_key": self.event_key,
            "failure_message": self.failure_message,
            "failure_stage": self.failure_stage,
            "failure_type": self.failure_type,
            "metric_value": self.metric_value,
            "paired_outcome_sha256": self.paired_outcome_sha256,
            "replicate_index": self.replicate_index,
            "status": self.status,
            "variant": self.variant,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(value, frozenset(cls.__annotations__), "mechanism row")
        return cls(
            variant=_string(value["variant"], "variant"),
            replicate_index=_integer(value["replicate_index"], "replicate_index"),
            status=_string(value["status"], "status"),
            event_key=_string(value["event_key"], "event_key"),
            metric_value=_optional_finite(value["metric_value"], "metric_value"),
            failure_stage=_optional_string(value["failure_stage"], "failure_stage"),
            failure_type=_optional_string(value["failure_type"], "failure_type"),
            failure_message=_optional_string(
                value["failure_message"], "failure_message"
            ),
            paired_outcome_sha256=_optional_sha256(
                value["paired_outcome_sha256"], "paired_outcome_sha256"
            ),
        )


@dataclass(frozen=True, slots=True)
class RQ1MechanismSummary:
    """Aligned full draws and failure-safe summaries for one variant."""

    variant: str
    expected: int
    completed: int
    failed: int
    full_draws: tuple[float | None, ...]
    mean: ReportedScalar
    q05: ReportedScalar
    q50: ReportedScalar
    q95: ReportedScalar

    def __post_init__(self) -> None:
        try:
            MechanismAblationVariant(self.variant)
        except ValueError as error:
            raise AAMASReportingSchemaError(
                "unknown mechanism summary variant"
            ) from error
        for field in ("expected", "completed", "failed"):
            _integer(getattr(self, field), field)
        if (
            self.completed + self.failed != self.expected
            or len(self.full_draws) != self.expected
        ):
            _fail("mechanism summary terminal counts are inconsistent")
        if sum(item is not None for item in self.full_draws) != self.completed:
            _fail("mechanism summary draws are not aligned to terminal rows")
        for item in self.full_draws:
            if item is not None:
                _finite(item, "mechanism full draw")
        for field in ("mean", "q05", "q50", "q95"):
            if not isinstance(getattr(self, field), ReportedScalar):
                _fail(f"mechanism {field} must be typed")

    def to_mapping(self) -> dict[str, object]:
        return {
            "completed": self.completed,
            "expected": self.expected,
            "failed": self.failed,
            "full_draws": list(self.full_draws),
            "mean": self.mean.to_mapping(),
            "q05": self.q05.to_mapping(),
            "q50": self.q50.to_mapping(),
            "q95": self.q95.to_mapping(),
            "variant": self.variant,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(value, frozenset(cls.__annotations__), "mechanism summary")
        return cls(
            variant=_string(value["variant"], "variant"),
            expected=_integer(value["expected"], "expected"),
            completed=_integer(value["completed"], "completed"),
            failed=_integer(value["failed"], "failed"),
            full_draws=tuple(
                _optional_finite(item, "mechanism full draw")
                for item in _array(value["full_draws"], "full_draws")
            ),
            mean=ReportedScalar.from_mapping(_mapping(value["mean"], "mean")),
            q05=ReportedScalar.from_mapping(_mapping(value["q05"], "q05")),
            q50=ReportedScalar.from_mapping(_mapping(value["q50"], "q50")),
            q95=ReportedScalar.from_mapping(_mapping(value["q95"], "q95")),
        )


@dataclass(frozen=True, slots=True)
class RQ2ParameterRow:
    """One expected truth/parameter recovery and identifiability row."""

    truth_id: str
    role: str
    parameter: str
    attempt_status: str
    truth: float
    posterior_median: float | None
    signed_error: float | None
    absolute_error: float | None
    lower_90: float | None
    upper_90: float | None
    covered_90: bool | None
    boundary_pileup: bool | None
    multimodal: bool | None
    nonidentifiable: bool | None
    report_sha256: str | None
    failure_type: str | None
    failure_message: str | None

    def __post_init__(self) -> None:
        _string(self.truth_id, "truth_id")
        try:
            TruthRole(self.role)
            status = AttemptStatus(self.attempt_status)
        except ValueError as error:
            raise AAMASReportingSchemaError("unknown RQ2 row role or status") from error
        _string(self.parameter, "parameter")
        truth = _finite(self.truth, "truth")
        object.__setattr__(self, "truth", truth)
        numeric = (
            self.posterior_median,
            self.signed_error,
            self.absolute_error,
            self.lower_90,
            self.upper_90,
        )
        flags = (
            self.covered_90,
            self.boundary_pileup,
            self.multimodal,
            self.nonidentifiable,
        )
        if status is AttemptStatus.COMPLETED:
            if any(item is None for item in numeric) or any(
                item is None for item in flags
            ):
                _fail("completed RQ2 row lacks recovery or diagnostic values")
            values = tuple(_finite(item, "RQ2 completed value") for item in numeric)
            median, signed, absolute, lower, upper = values
            if not math.isclose(signed, median - truth, rel_tol=1e-12, abs_tol=1e-12):
                _fail("RQ2 signed error is inconsistent")
            if not math.isclose(absolute, abs(signed), rel_tol=1e-12, abs_tol=1e-12):
                _fail("RQ2 absolute error is inconsistent")
            if lower > upper or self.covered_90 is not (lower <= truth <= upper):
                _fail("RQ2 recovery interval is inconsistent")
            for flag in flags:
                _boolean(flag, "RQ2 diagnostic flag")
            _sha256(self.report_sha256, "RQ2 report_sha256")
            if self.failure_type is not None or self.failure_message is not None:
                _fail("completed RQ2 row cannot retain failure fields")
        else:
            if any(item is not None for item in (*numeric, *flags, self.report_sha256)):
                _fail("failed RQ2 row cannot claim completed diagnostics")
            _string(self.failure_type, "RQ2 failure_type")
            _string(self.failure_message, "RQ2 failure_message")

    def to_mapping(self) -> dict[str, object]:
        return {
            "absolute_error": self.absolute_error,
            "attempt_status": self.attempt_status,
            "boundary_pileup": self.boundary_pileup,
            "covered_90": self.covered_90,
            "failure_message": self.failure_message,
            "failure_type": self.failure_type,
            "lower_90": self.lower_90,
            "multimodal": self.multimodal,
            "nonidentifiable": self.nonidentifiable,
            "parameter": self.parameter,
            "posterior_median": self.posterior_median,
            "report_sha256": self.report_sha256,
            "role": self.role,
            "signed_error": self.signed_error,
            "truth": self.truth,
            "truth_id": self.truth_id,
            "upper_90": self.upper_90,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(value, frozenset(cls.__annotations__), "RQ2 parameter row")
        return cls(
            truth_id=_string(value["truth_id"], "truth_id"),
            role=_string(value["role"], "role"),
            parameter=_string(value["parameter"], "parameter"),
            attempt_status=_string(value["attempt_status"], "attempt_status"),
            truth=_finite(value["truth"], "truth"),
            posterior_median=_optional_finite(
                value["posterior_median"], "posterior_median"
            ),
            signed_error=_optional_finite(value["signed_error"], "signed_error"),
            absolute_error=_optional_finite(value["absolute_error"], "absolute_error"),
            lower_90=_optional_finite(value["lower_90"], "lower_90"),
            upper_90=_optional_finite(value["upper_90"], "upper_90"),
            covered_90=(
                None
                if value["covered_90"] is None
                else _boolean(value["covered_90"], "covered_90")
            ),
            boundary_pileup=(
                None
                if value["boundary_pileup"] is None
                else _boolean(value["boundary_pileup"], "boundary_pileup")
            ),
            multimodal=(
                None
                if value["multimodal"] is None
                else _boolean(value["multimodal"], "multimodal")
            ),
            nonidentifiable=(
                None
                if value["nonidentifiable"] is None
                else _boolean(value["nonidentifiable"], "nonidentifiable")
            ),
            report_sha256=_optional_sha256(value["report_sha256"], "report_sha256"),
            failure_type=_optional_string(value["failure_type"], "failure_type"),
            failure_message=_optional_string(
                value["failure_message"], "failure_message"
            ),
        )


@dataclass(frozen=True, slots=True)
class RQ2ParameterSummary:
    """Failure-safe role/parameter aggregate over all expected truths."""

    role: str
    parameter: str
    n: int
    completed: int
    failed: int
    signed_bias: ReportedScalar
    mae: ReportedScalar
    rmse: ReportedScalar
    empirical_90_coverage: ReportedScalar
    mean_interval_width: ReportedScalar
    binomial_wilson_interval: TypedInterval
    boundary_pileup_count: ReportedScalar
    boundary_pileup_rate: ReportedScalar
    multimodality_count: ReportedScalar
    multimodality_rate: ReportedScalar
    nonidentifiability_count: ReportedScalar
    nonidentifiability_rate: ReportedScalar

    def __post_init__(self) -> None:
        try:
            TruthRole(self.role)
        except ValueError as error:
            raise AAMASReportingSchemaError("unknown RQ2 summary role") from error
        _string(self.parameter, "parameter")
        for field in ("n", "completed", "failed"):
            _integer(getattr(self, field), field)
        if self.completed + self.failed != self.n:
            _fail("RQ2 summary counts do not cover expected truths")
        for field in (
            "signed_bias",
            "mae",
            "rmse",
            "empirical_90_coverage",
            "mean_interval_width",
            "boundary_pileup_count",
            "boundary_pileup_rate",
            "multimodality_count",
            "multimodality_rate",
            "nonidentifiability_count",
            "nonidentifiability_rate",
        ):
            if not isinstance(getattr(self, field), ReportedScalar):
                _fail(f"RQ2 summary {field} must be typed")
        if not isinstance(self.binomial_wilson_interval, TypedInterval):
            _fail("RQ2 coverage interval must be typed")

    def to_mapping(self) -> dict[str, object]:
        return {
            "binomial_wilson_interval": self.binomial_wilson_interval.to_mapping(),
            "boundary_pileup_count": self.boundary_pileup_count.to_mapping(),
            "boundary_pileup_rate": self.boundary_pileup_rate.to_mapping(),
            "completed": self.completed,
            "empirical_90_coverage": self.empirical_90_coverage.to_mapping(),
            "failed": self.failed,
            "mae": self.mae.to_mapping(),
            "mean_interval_width": self.mean_interval_width.to_mapping(),
            "multimodality_count": self.multimodality_count.to_mapping(),
            "multimodality_rate": self.multimodality_rate.to_mapping(),
            "n": self.n,
            "nonidentifiability_count": self.nonidentifiability_count.to_mapping(),
            "nonidentifiability_rate": self.nonidentifiability_rate.to_mapping(),
            "parameter": self.parameter,
            "rmse": self.rmse.to_mapping(),
            "role": self.role,
            "signed_bias": self.signed_bias.to_mapping(),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(value, frozenset(cls.__annotations__), "RQ2 parameter summary")

        def scalar(field: str) -> ReportedScalar:
            return ReportedScalar.from_mapping(_mapping(value[field], field))

        return cls(
            role=_string(value["role"], "role"),
            parameter=_string(value["parameter"], "parameter"),
            n=_integer(value["n"], "n"),
            completed=_integer(value["completed"], "completed"),
            failed=_integer(value["failed"], "failed"),
            signed_bias=scalar("signed_bias"),
            mae=scalar("mae"),
            rmse=scalar("rmse"),
            empirical_90_coverage=scalar("empirical_90_coverage"),
            mean_interval_width=scalar("mean_interval_width"),
            binomial_wilson_interval=TypedInterval.from_mapping(
                _mapping(value["binomial_wilson_interval"], "binomial_wilson_interval")
            ),
            boundary_pileup_count=scalar("boundary_pileup_count"),
            boundary_pileup_rate=scalar("boundary_pileup_rate"),
            multimodality_count=scalar("multimodality_count"),
            multimodality_rate=scalar("multimodality_rate"),
            nonidentifiability_count=scalar("nonidentifiability_count"),
            nonidentifiability_rate=scalar("nonidentifiability_rate"),
        )


@dataclass(frozen=True, slots=True)
class RQ2PPCRow:
    """One expected held-out PPC coverage observation."""

    truth_id: str
    role: str
    statistic: str
    attempt_status: str
    covered: bool | None
    report_sha256: str | None

    def __post_init__(self) -> None:
        _string(self.truth_id, "PPC truth_id")
        try:
            TruthRole(self.role)
            status = AttemptStatus(self.attempt_status)
        except ValueError as error:
            raise AAMASReportingSchemaError("unknown PPC row role or status") from error
        _string(self.statistic, "PPC statistic")
        if status is AttemptStatus.COMPLETED:
            _boolean(self.covered, "PPC covered")
            _sha256(self.report_sha256, "PPC report_sha256")
        elif self.covered is not None or self.report_sha256 is not None:
            _fail("failed PPC row cannot claim coverage")

    def to_mapping(self) -> dict[str, object]:
        return {
            "attempt_status": self.attempt_status,
            "covered": self.covered,
            "report_sha256": self.report_sha256,
            "role": self.role,
            "statistic": self.statistic,
            "truth_id": self.truth_id,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(value, frozenset(cls.__annotations__), "RQ2 PPC row")
        return cls(
            truth_id=_string(value["truth_id"], "truth_id"),
            role=_string(value["role"], "role"),
            statistic=_string(value["statistic"], "statistic"),
            attempt_status=_string(value["attempt_status"], "attempt_status"),
            covered=(
                None
                if value["covered"] is None
                else _boolean(value["covered"], "covered")
            ),
            report_sha256=_optional_sha256(value["report_sha256"], "report_sha256"),
        )


@dataclass(frozen=True, slots=True)
class RQ2PPCSummary:
    role: str
    statistic: str
    n: int
    completed: int
    failed: int
    held_out_coverage: ReportedScalar

    def __post_init__(self) -> None:
        try:
            TruthRole(self.role)
        except ValueError as error:
            raise AAMASReportingSchemaError("unknown PPC summary role") from error
        _string(self.statistic, "PPC summary statistic")
        for field in ("n", "completed", "failed"):
            _integer(getattr(self, field), field)
        if self.completed + self.failed != self.n:
            _fail("PPC summary counts are inconsistent")
        if not isinstance(self.held_out_coverage, ReportedScalar):
            _fail("PPC coverage must be a typed scalar")

    def to_mapping(self) -> dict[str, object]:
        return {
            "completed": self.completed,
            "failed": self.failed,
            "held_out_coverage": self.held_out_coverage.to_mapping(),
            "n": self.n,
            "role": self.role,
            "statistic": self.statistic,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(value, frozenset(cls.__annotations__), "RQ2 PPC summary")
        return cls(
            role=_string(value["role"], "role"),
            statistic=_string(value["statistic"], "statistic"),
            n=_integer(value["n"], "n"),
            completed=_integer(value["completed"], "completed"),
            failed=_integer(value["failed"], "failed"),
            held_out_coverage=ReportedScalar.from_mapping(
                _mapping(value["held_out_coverage"], "held_out_coverage")
            ),
        )


@dataclass(frozen=True, slots=True)
class RQ2SBCRankSummary:
    """Complete rank counts and interval coverage, without significance tests."""

    role: str
    parameter: str
    expected_truths: int
    completed_truths: int
    failed_truths: int
    posterior_draws_per_replicate: int | None
    ranks: tuple[int, ...]
    histogram: tuple[int, ...]
    interval_coverage_90: ReportedScalar
    significance_test: Literal[None] = None

    def __post_init__(self) -> None:
        try:
            TruthRole(self.role)
        except ValueError as error:
            raise AAMASReportingSchemaError("unknown SBC summary role") from error
        _string(self.parameter, "SBC parameter")
        for field in ("expected_truths", "completed_truths", "failed_truths"):
            _integer(getattr(self, field), field)
        if self.completed_truths + self.failed_truths != self.expected_truths:
            _fail("SBC terminal counts are inconsistent")
        if self.posterior_draws_per_replicate is None:
            if self.ranks or self.histogram:
                _fail("blocked SBC summary cannot retain complete-case ranks")
        else:
            draws = _integer(
                self.posterior_draws_per_replicate,
                "posterior_draws_per_replicate",
                minimum=2,
            )
            if len(self.histogram) != draws + 1 or sum(self.histogram) != len(
                self.ranks
            ):
                _fail("SBC rank histogram is inconsistent")
            if any(rank < 0 or rank > draws for rank in self.ranks):
                _fail("SBC rank is outside support")
        if not isinstance(self.interval_coverage_90, ReportedScalar):
            _fail("SBC coverage must be typed")
        if self.significance_test is not None:
            _fail("AAMAS SBC reporting forbids post-hoc significance tests")

    def to_mapping(self) -> dict[str, object]:
        return {
            "completed_truths": self.completed_truths,
            "expected_truths": self.expected_truths,
            "failed_truths": self.failed_truths,
            "histogram": list(self.histogram),
            "interval_coverage_90": self.interval_coverage_90.to_mapping(),
            "parameter": self.parameter,
            "posterior_draws_per_replicate": self.posterior_draws_per_replicate,
            "ranks": list(self.ranks),
            "role": self.role,
            "significance_test": self.significance_test,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(value, frozenset(cls.__annotations__), "RQ2 SBC summary")
        raw_draws = value["posterior_draws_per_replicate"]
        if raw_draws is not None:
            raw_draws = _integer(raw_draws, "posterior_draws_per_replicate", minimum=2)
        if value["significance_test"] is not None:
            _fail("SBC significance_test must remain null")
        return cls(
            role=_string(value["role"], "role"),
            parameter=_string(value["parameter"], "parameter"),
            expected_truths=_integer(value["expected_truths"], "expected_truths"),
            completed_truths=_integer(value["completed_truths"], "completed_truths"),
            failed_truths=_integer(value["failed_truths"], "failed_truths"),
            posterior_draws_per_replicate=raw_draws,
            ranks=tuple(
                _integer(item, "SBC rank") for item in _array(value["ranks"], "ranks")
            ),
            histogram=tuple(
                _integer(item, "SBC histogram count")
                for item in _array(value["histogram"], "histogram")
            ),
            interval_coverage_90=ReportedScalar.from_mapping(
                _mapping(value["interval_coverage_90"], "interval_coverage_90")
            ),
            significance_test=None,
        )


@dataclass(frozen=True, slots=True)
class RQ3TargetRecord:
    """One sealed target-only uncertainty/loss pair."""

    truth_id: str
    uncertainty_score: float
    loss: float
    role: Literal["target"] = "target"

    def __post_init__(self) -> None:
        _string(self.truth_id, "RQ3 truth_id")
        uncertainty = _finite(self.uncertainty_score, "uncertainty_score")
        loss = _finite(self.loss, "loss")
        if uncertainty < 0.0 or loss < 0.0:
            _fail("RQ3 uncertainty and loss must be non-negative")
        if self.role != TruthRole.TARGET.value:
            _fail("RQ3 records must be target-only")
        object.__setattr__(self, "uncertainty_score", uncertainty)
        object.__setattr__(self, "loss", loss)

    def to_mapping(self) -> dict[str, object]:
        return {
            "loss": self.loss,
            "role": self.role,
            "truth_id": self.truth_id,
            "uncertainty_score": self.uncertainty_score,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(value, frozenset(cls.__annotations__), "RQ3 target record")
        if value["role"] != TruthRole.TARGET.value:
            _fail("RQ3 row role must remain target")
        return cls(
            truth_id=_string(value["truth_id"], "truth_id"),
            uncertainty_score=_finite(value["uncertainty_score"], "uncertainty_score"),
            loss=_finite(value["loss"], "loss"),
            role="target",
        )


@dataclass(frozen=True, slots=True)
class RQ3RiskCoveragePoint:
    """One frozen threshold with target-only risk and deterministic resampling."""

    threshold: float
    expected_target_n: int
    source_record_n: int
    retained_n: int | None
    minimum_retained: int
    selection_coverage: ReportedScalar
    mean_loss: ReportedScalar
    resampling_interval: TypedInterval
    bootstrap_draws: tuple[float, ...]
    bootstrap_event_keys: tuple[str, ...]
    bootstrap_event_keys_sha256: str | None

    def __post_init__(self) -> None:
        threshold = _finite(self.threshold, "risk threshold")
        if threshold < 0.0:
            _fail("risk threshold must be non-negative")
        object.__setattr__(self, "threshold", threshold)
        for field in ("expected_target_n", "source_record_n", "minimum_retained"):
            _integer(getattr(self, field), field)
        if self.retained_n is not None:
            _integer(self.retained_n, "retained_n")
        if self.source_record_n > self.expected_target_n or (
            self.retained_n is not None and self.retained_n > self.source_record_n
        ):
            _fail("risk-coverage counts are inconsistent")
        if not isinstance(self.selection_coverage, ReportedScalar) or not isinstance(
            self.mean_loss, ReportedScalar
        ):
            _fail("risk coverage and loss must be typed scalars")
        if not isinstance(self.resampling_interval, TypedInterval):
            _fail("risk resampling interval must be typed")
        if (
            self.resampling_interval.interval_kind
            != "synthetic_target_resampling_interval"
        ):
            _fail("risk interval semantics changed")
        if self.bootstrap_draws:
            if len(self.bootstrap_draws) != len(self.bootstrap_event_keys):
                _fail("bootstrap draws and EventKeys are not aligned")
            if any(not math.isfinite(item) for item in self.bootstrap_draws):
                _fail("bootstrap draws must be finite")
            if any(not key.startswith("pse1:") for key in self.bootstrap_event_keys):
                _fail("bootstrap EventKey is invalid")
            expected_digest = _semantic_hash(list(self.bootstrap_event_keys))
            if self.bootstrap_event_keys_sha256 != expected_digest:
                _fail("bootstrap EventKey commitment is inconsistent")
            if self.mean_loss.status is not ScalarStatus.DEFINED:
                _fail("bootstrap draws require a defined target mean loss")
        elif self.bootstrap_event_keys or self.bootstrap_event_keys_sha256 is not None:
            _fail("undefined bootstrap cannot retain keys or a commitment")
        elif (
            self.mean_loss.status is ScalarStatus.DEFINED
            or self.resampling_interval.status is ScalarStatus.DEFINED
        ):
            _fail("defined target risk requires retained bootstrap evidence")
        if self.retained_n is None and (
            self.selection_coverage.status is ScalarStatus.DEFINED
            or self.mean_loss.status is ScalarStatus.DEFINED
        ):
            _fail("undefined retained count cannot support target risk metrics")

    def to_mapping(self) -> dict[str, object]:
        return {
            "bootstrap_draws": list(self.bootstrap_draws),
            "bootstrap_event_keys": list(self.bootstrap_event_keys),
            "bootstrap_event_keys_sha256": self.bootstrap_event_keys_sha256,
            "expected_target_n": self.expected_target_n,
            "mean_loss": self.mean_loss.to_mapping(),
            "minimum_retained": self.minimum_retained,
            "resampling_interval": self.resampling_interval.to_mapping(),
            "retained_n": self.retained_n,
            "selection_coverage": self.selection_coverage.to_mapping(),
            "source_record_n": self.source_record_n,
            "threshold": self.threshold,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(value, frozenset(cls.__annotations__), "RQ3 risk point")
        return cls(
            threshold=_finite(value["threshold"], "threshold"),
            expected_target_n=_integer(value["expected_target_n"], "expected_target_n"),
            source_record_n=_integer(value["source_record_n"], "source_record_n"),
            retained_n=(
                None
                if value["retained_n"] is None
                else _integer(value["retained_n"], "retained_n")
            ),
            minimum_retained=_integer(value["minimum_retained"], "minimum_retained"),
            selection_coverage=ReportedScalar.from_mapping(
                _mapping(value["selection_coverage"], "selection_coverage")
            ),
            mean_loss=ReportedScalar.from_mapping(
                _mapping(value["mean_loss"], "mean_loss")
            ),
            resampling_interval=TypedInterval.from_mapping(
                _mapping(value["resampling_interval"], "resampling_interval")
            ),
            bootstrap_draws=tuple(
                _finite(item, "bootstrap draw")
                for item in _array(value["bootstrap_draws"], "bootstrap_draws")
            ),
            bootstrap_event_keys=tuple(
                _string(item, "bootstrap EventKey")
                for item in _array(
                    value["bootstrap_event_keys"], "bootstrap_event_keys"
                )
            ),
            bootstrap_event_keys_sha256=_optional_sha256(
                value["bootstrap_event_keys_sha256"], "bootstrap_event_keys_sha256"
            ),
        )


@dataclass(frozen=True, slots=True)
class ValidityCertificateEntry:
    """Fail-closed hand-off; the umbrella layer must supply independent evidence."""

    level: Literal["abstain"]
    certificate_mapping: Literal[None]
    source_derived_evidence: tuple[tuple[str, object], ...]
    missing_evidence_fields: tuple[str, ...]
    reason: str

    def __post_init__(self) -> None:
        if (
            self.level != ValidityLevel.ABSTAIN.value
            or self.certificate_mapping is not None
        ):
            _fail("reporting validity entry must remain an unissued ABSTAIN hand-off")
        if tuple(name for name, _ in self.source_derived_evidence) != tuple(
            sorted(name for name, _ in self.source_derived_evidence)
        ):
            _fail("validity evidence mapping must use canonical field order")
        evidence = dict(self.source_derived_evidence)
        expected = frozenset(ValidityEvidence.__annotations__)
        if frozenset(evidence) != expected:
            _fail("validity hand-off must retain the exact evidence schema")
        # Reconstructing the value object validates every source-derived value.
        ValidityEvidence.from_mapping(evidence)
        if self.missing_evidence_fields != tuple(sorted(self.missing_evidence_fields)):
            _fail("missing validity evidence fields must be sorted")
        if not self.missing_evidence_fields:
            _fail("reporting sources cannot independently authorize validity")
        _string(self.reason, "validity abstention reason")

    def to_mapping(self) -> dict[str, object]:
        return {
            "certificate_mapping": self.certificate_mapping,
            "level": self.level,
            "missing_evidence_fields": list(self.missing_evidence_fields),
            "reason": self.reason,
            "source_derived_evidence": dict(self.source_derived_evidence),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(value, frozenset(cls.__annotations__), "validity entry")
        if value["level"] != "abstain" or value["certificate_mapping"] is not None:
            _fail("validity entry authority changed")
        evidence = _mapping(value["source_derived_evidence"], "source_derived_evidence")
        return cls(
            level="abstain",
            certificate_mapping=None,
            source_derived_evidence=tuple(sorted(evidence.items())),
            missing_evidence_fields=tuple(
                _string(item, "missing evidence field")
                for item in _array(
                    value["missing_evidence_fields"], "missing_evidence_fields"
                )
            ),
            reason=_string(value["reason"], "reason"),
        )


@dataclass(frozen=True, slots=True)
class OutputRegistryEntry:
    output_id: str
    row_schema: str
    row_count: int
    status: str

    def __post_init__(self) -> None:
        for field in ("output_id", "row_schema", "status"):
            _string(getattr(self, field), field)
        _integer(self.row_count, "row_count")

    def to_mapping(self) -> dict[str, object]:
        return {
            "output_id": self.output_id,
            "row_count": self.row_count,
            "row_schema": self.row_schema,
            "status": self.status,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(value, frozenset(cls.__annotations__), "output registry entry")
        return cls(
            output_id=_string(value["output_id"], "output_id"),
            row_schema=_string(value["row_schema"], "row_schema"),
            row_count=_integer(value["row_count"], "row_count"),
            status=_string(value["status"], "status"),
        )


@dataclass(frozen=True, slots=True)
class AAMASSubmissionArtifact:
    """Immutable reporting artifact derived only from two audited source bundles."""

    reporting_spec: FrozenAAMASReportingSpec
    reporting_spec_sha256: str
    sources: AAMASReportingSources
    output_registry: tuple[OutputRegistryEntry, ...]
    rq1_rng_rows: tuple[RQ1RNGRow, ...]
    rq1_baseline_invariants: tuple[RQ1BaselineEqualityInvariant, ...]
    rq1_mechanism_rows: tuple[RQ1MechanismRow, ...]
    rq1_mechanism_summaries: tuple[RQ1MechanismSummary, ...]
    rq2_parameter_rows: tuple[RQ2ParameterRow, ...]
    rq2_parameter_summaries: tuple[RQ2ParameterSummary, ...]
    rq2_ppc_rows: tuple[RQ2PPCRow, ...]
    rq2_ppc_summaries: tuple[RQ2PPCSummary, ...]
    rq2_sbc_summaries: tuple[RQ2SBCRankSummary, ...]
    rq3_target_records: tuple[RQ3TargetRecord, ...]
    rq3_risk_coverage: tuple[RQ3RiskCoveragePoint, ...]
    validity_certificate_entry: ValidityCertificateEntry
    formal_source_scale: bool
    claim_scope: str = AAMAS_SUBMISSION_CLAIM_SCOPE
    synthetic_only: Literal[True] = True
    schema_version: str = AAMAS_SUBMISSION_ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.reporting_spec, FrozenAAMASReportingSpec):
            _fail("submission artifact requires a frozen reporting spec")
        if self.reporting_spec_sha256 != self.reporting_spec.sha256:
            _fail("reporting spec hash is inconsistent")
        if not isinstance(self.sources, AAMASReportingSources):
            _fail("submission sources have the wrong type")
        typed_collections = (
            (self.output_registry, OutputRegistryEntry),
            (self.rq1_rng_rows, RQ1RNGRow),
            (self.rq1_baseline_invariants, RQ1BaselineEqualityInvariant),
            (self.rq1_mechanism_rows, RQ1MechanismRow),
            (self.rq1_mechanism_summaries, RQ1MechanismSummary),
            (self.rq2_parameter_rows, RQ2ParameterRow),
            (self.rq2_parameter_summaries, RQ2ParameterSummary),
            (self.rq2_ppc_rows, RQ2PPCRow),
            (self.rq2_ppc_summaries, RQ2PPCSummary),
            (self.rq2_sbc_summaries, RQ2SBCRankSummary),
            (self.rq3_target_records, RQ3TargetRecord),
            (self.rq3_risk_coverage, RQ3RiskCoveragePoint),
        )
        for collection, expected_type in typed_collections:
            if not isinstance(collection, tuple) or not all(
                isinstance(item, expected_type) for item in collection
            ):
                _fail("submission row collection is not immutable and typed")
        if tuple(item.mode for item in self.rq1_baseline_invariants) != tuple(
            sorted(mode.value for mode in RNGCouplingMode)
        ):
            _fail("cross-cell invariant registry must cover every RNG mode")
        if not isinstance(self.validity_certificate_entry, ValidityCertificateEntry):
            _fail("validity hand-off has the wrong type")
        if not isinstance(self.formal_source_scale, bool):
            _fail("formal_source_scale must be boolean")
        if (
            self.claim_scope != AAMAS_SUBMISSION_CLAIM_SCOPE
            or self.synthetic_only is not True
        ):
            _fail("submission artifact authority boundary changed")
        if self.schema_version != AAMAS_SUBMISSION_ARTIFACT_SCHEMA_VERSION:
            _fail("unsupported submission artifact schema")
        expected_registry = _output_registry_for(self)
        if self.output_registry != expected_registry:
            _fail("output registry does not match retained rows and statuses")

    @property
    def sha256(self) -> str:
        return _semantic_hash(self.to_mapping())

    def to_mapping(self) -> dict[str, object]:
        return {
            "claim_scope": self.claim_scope,
            "formal_source_scale": self.formal_source_scale,
            "output_registry": [item.to_mapping() for item in self.output_registry],
            "reporting_spec": self.reporting_spec.to_mapping(),
            "reporting_spec_sha256": self.reporting_spec_sha256,
            "rq1_baseline_invariants": [
                item.to_mapping() for item in self.rq1_baseline_invariants
            ],
            "rq1_mechanism_rows": [
                item.to_mapping() for item in self.rq1_mechanism_rows
            ],
            "rq1_mechanism_summaries": [
                item.to_mapping() for item in self.rq1_mechanism_summaries
            ],
            "rq1_rng_rows": [item.to_mapping() for item in self.rq1_rng_rows],
            "rq2_parameter_rows": [
                item.to_mapping() for item in self.rq2_parameter_rows
            ],
            "rq2_parameter_summaries": [
                item.to_mapping() for item in self.rq2_parameter_summaries
            ],
            "rq2_ppc_rows": [item.to_mapping() for item in self.rq2_ppc_rows],
            "rq2_ppc_summaries": [item.to_mapping() for item in self.rq2_ppc_summaries],
            "rq2_sbc_summaries": [item.to_mapping() for item in self.rq2_sbc_summaries],
            "rq3_risk_coverage": [item.to_mapping() for item in self.rq3_risk_coverage],
            "rq3_target_records": [
                item.to_mapping() for item in self.rq3_target_records
            ],
            "schema_version": self.schema_version,
            "sources": self.sources.to_mapping(),
            "synthetic_only": self.synthetic_only,
            "validity_certificate_entry": self.validity_certificate_entry.to_mapping(),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        expected = frozenset(
            {
                "claim_scope",
                "formal_source_scale",
                "output_registry",
                "reporting_spec",
                "reporting_spec_sha256",
                "rq1_baseline_invariants",
                "rq1_mechanism_rows",
                "rq1_mechanism_summaries",
                "rq1_rng_rows",
                "rq2_parameter_rows",
                "rq2_parameter_summaries",
                "rq2_ppc_rows",
                "rq2_ppc_summaries",
                "rq2_sbc_summaries",
                "rq3_risk_coverage",
                "rq3_target_records",
                "schema_version",
                "sources",
                "synthetic_only",
                "validity_certificate_entry",
            }
        )
        _strict_fields(value, expected, "AAMAS submission artifact")
        if value["synthetic_only"] is not True:
            _fail("submission artifact must remain synthetic-only")

        def rows(field: str, loader: object) -> tuple[object, ...]:
            if not callable(loader):
                _fail("row loader must be callable")
            return tuple(
                loader(_mapping(item, f"{field} item"))
                for item in _array(value[field], field)
            )

        artifact = cls(
            reporting_spec=FrozenAAMASReportingSpec.from_mapping(
                _mapping(value["reporting_spec"], "reporting_spec")
            ),
            reporting_spec_sha256=_sha256(
                value["reporting_spec_sha256"], "reporting_spec_sha256"
            ),
            sources=AAMASReportingSources.from_mapping(
                _mapping(value["sources"], "sources")
            ),
            output_registry=cast(
                tuple[OutputRegistryEntry, ...],
                rows("output_registry", OutputRegistryEntry.from_mapping),
            ),
            rq1_rng_rows=cast(
                tuple[RQ1RNGRow, ...], rows("rq1_rng_rows", RQ1RNGRow.from_mapping)
            ),
            rq1_baseline_invariants=cast(
                tuple[RQ1BaselineEqualityInvariant, ...],
                rows(
                    "rq1_baseline_invariants",
                    RQ1BaselineEqualityInvariant.from_mapping,
                ),
            ),
            rq1_mechanism_rows=cast(
                tuple[RQ1MechanismRow, ...],
                rows("rq1_mechanism_rows", RQ1MechanismRow.from_mapping),
            ),
            rq1_mechanism_summaries=cast(
                tuple[RQ1MechanismSummary, ...],
                rows("rq1_mechanism_summaries", RQ1MechanismSummary.from_mapping),
            ),
            rq2_parameter_rows=cast(
                tuple[RQ2ParameterRow, ...],
                rows("rq2_parameter_rows", RQ2ParameterRow.from_mapping),
            ),
            rq2_parameter_summaries=cast(
                tuple[RQ2ParameterSummary, ...],
                rows("rq2_parameter_summaries", RQ2ParameterSummary.from_mapping),
            ),
            rq2_ppc_rows=cast(
                tuple[RQ2PPCRow, ...], rows("rq2_ppc_rows", RQ2PPCRow.from_mapping)
            ),
            rq2_ppc_summaries=cast(
                tuple[RQ2PPCSummary, ...],
                rows("rq2_ppc_summaries", RQ2PPCSummary.from_mapping),
            ),
            rq2_sbc_summaries=cast(
                tuple[RQ2SBCRankSummary, ...],
                rows("rq2_sbc_summaries", RQ2SBCRankSummary.from_mapping),
            ),
            rq3_target_records=cast(
                tuple[RQ3TargetRecord, ...],
                rows("rq3_target_records", RQ3TargetRecord.from_mapping),
            ),
            rq3_risk_coverage=cast(
                tuple[RQ3RiskCoveragePoint, ...],
                rows("rq3_risk_coverage", RQ3RiskCoveragePoint.from_mapping),
            ),
            validity_certificate_entry=ValidityCertificateEntry.from_mapping(
                _mapping(
                    value["validity_certificate_entry"], "validity_certificate_entry"
                )
            ),
            formal_source_scale=_boolean(
                value["formal_source_scale"], "formal_source_scale"
            ),
            claim_scope=_string(value["claim_scope"], "claim_scope"),
            synthetic_only=True,
            schema_version=_string(value["schema_version"], "schema_version"),
        )
        if artifact.to_mapping() != dict(value):
            _fail("submission artifact is not in canonical serialized form")
        return artifact


def _output_registry_values(
    *,
    rng_rows: tuple[RQ1RNGRow, ...],
    invariants: tuple[RQ1BaselineEqualityInvariant, ...],
    mechanism_rows: tuple[RQ1MechanismRow, ...],
    mechanism_summaries: tuple[RQ1MechanismSummary, ...],
    parameter_rows: tuple[RQ2ParameterRow, ...],
    parameter_summaries: tuple[RQ2ParameterSummary, ...],
    ppc_rows: tuple[RQ2PPCRow, ...],
    ppc_summaries: tuple[RQ2PPCSummary, ...],
    sbc_summaries: tuple[RQ2SBCRankSummary, ...],
    target_records: tuple[RQ3TargetRecord, ...],
    risk_points: tuple[RQ3RiskCoveragePoint, ...],
) -> tuple[OutputRegistryEntry, ...]:
    rng_status = "complete" if all(item.failed == 0 for item in rng_rows) else "blocked"
    invariant_status = (
        "complete" if all(item.passed is True for item in invariants) else "blocked"
    )
    mechanism_status = (
        "complete"
        if all(item.status == "completed" for item in mechanism_rows)
        else "blocked"
    )
    calibration_status = (
        "complete"
        if parameter_rows
        and all(item.attempt_status == "completed" for item in parameter_rows)
        else "blocked"
    )
    risk_status = (
        "blocked"
        if not target_records
        else (
            "complete"
            if all(
                item.mean_loss.status is ScalarStatus.DEFINED for item in risk_points
            )
            else "complete_with_registered_undefined_points"
        )
    )
    return (
        OutputRegistryEntry("rq1_rng", "RQ1RNGRow.v1", len(rng_rows), rng_status),
        OutputRegistryEntry(
            "rq1_cross_cell_baseline_invariants",
            "RQ1BaselineEqualityInvariant.v1",
            len(invariants),
            invariant_status,
        ),
        OutputRegistryEntry(
            "rq1_mechanism_rows",
            "RQ1MechanismRow.v1",
            len(mechanism_rows),
            mechanism_status,
        ),
        OutputRegistryEntry(
            "rq1_mechanism_summaries",
            "RQ1MechanismSummary.v1",
            len(mechanism_summaries),
            mechanism_status,
        ),
        OutputRegistryEntry(
            "rq2_parameter_rows",
            "RQ2ParameterRow.v1",
            len(parameter_rows),
            calibration_status,
        ),
        OutputRegistryEntry(
            "rq2_parameter_summaries",
            "RQ2ParameterSummary.v1",
            len(parameter_summaries),
            calibration_status,
        ),
        OutputRegistryEntry(
            "rq2_held_out_ppc_rows",
            "RQ2PPCRow.v1",
            len(ppc_rows),
            calibration_status,
        ),
        OutputRegistryEntry(
            "rq2_held_out_ppc_summaries",
            "RQ2PPCSummary.v1",
            len(ppc_summaries),
            calibration_status,
        ),
        OutputRegistryEntry(
            "rq2_sbc_rank_summaries",
            "RQ2SBCRankSummary.v1",
            len(sbc_summaries),
            calibration_status,
        ),
        OutputRegistryEntry(
            "rq3_target_records",
            "RQ3TargetRecord.v1",
            len(target_records),
            risk_status,
        ),
        OutputRegistryEntry(
            "rq3_risk_coverage",
            "RQ3RiskCoveragePoint.v1",
            len(risk_points),
            risk_status,
        ),
        OutputRegistryEntry(
            "validity_certificate_entry",
            "ValidityCertificateEntry.v1",
            1,
            "abstain",
        ),
    )


def _output_registry_for(
    artifact: AAMASSubmissionArtifact,
) -> tuple[OutputRegistryEntry, ...]:
    return _output_registry_values(
        rng_rows=artifact.rq1_rng_rows,
        invariants=artifact.rq1_baseline_invariants,
        mechanism_rows=artifact.rq1_mechanism_rows,
        mechanism_summaries=artifact.rq1_mechanism_summaries,
        parameter_rows=artifact.rq2_parameter_rows,
        parameter_summaries=artifact.rq2_parameter_summaries,
        ppc_rows=artifact.rq2_ppc_rows,
        ppc_summaries=artifact.rq2_ppc_summaries,
        sbc_summaries=artifact.rq2_sbc_summaries,
        target_records=artifact.rq3_target_records,
        risk_points=artifact.rq3_risk_coverage,
    )


def _blocked_scalars(reason: str, count: int) -> tuple[ReportedScalar, ...]:
    return tuple(
        ReportedScalar.unavailable(ScalarStatus.BLOCKED_SOURCE_FAILURE, reason)
        for _ in range(count)
    )


def _derive_rng(
    directory: Path,
    plan: AAMASBenchmarkPlan,
    artifact: AAMASBenchmarkArtifact,
) -> tuple[tuple[RQ1RNGRow, ...], tuple[RQ1BaselineEqualityInvariant, ...]]:
    attempts = {item.attempt_id: item for item in artifact.attempts}
    sources = {item.attempt_id: item for item in artifact.source_artifacts}
    raw: dict[tuple[str, str], dict[str, object]] = {}
    for cell, expected in zip(
        plan.rng_cells,
        (
            item
            for item in plan.expected_attempts
            if item.kind is AAMASAttemptKind.RNG_MECHANISM_SMOKE
        ),
        strict=True,
    ):
        attempt = attempts[expected.attempt_id]
        if attempt.status is not AAMASAttemptStatus.COMPLETED:
            reason = (
                f"source attempt {attempt.attempt_id} ended {attempt.status.value}: "
                f"{attempt.failure_code}: {attempt.failure_message}"
            )
            for mode in RNGCouplingMode:
                raw[(cell.cell_id, mode.value)] = {
                    "cell": cell,
                    "mode": mode,
                    "differences": (),
                    "baseline": (),
                    "candidate": (),
                    "event_keys": (),
                    "source_sha256": None,
                    "reason": reason,
                }
            continue
        source = sources.get(attempt.attempt_id)
        if source is None or source.kind is not AAMASSourceKind.RNG_MECHANISM_SMOKE:
            _integrity_fail("completed RNG attempt lacks its audited source")
        try:
            nested = MechanismSmokeArtifact.from_mapping(
                read_json_object(directory / source.relative_path)
            )
        except (OSError, ValueError, MetadataValidationError) as error:
            raise AAMASReportingIntegrityError(
                "cannot reconstruct an already-audited RNG source"
            ) from error
        if nested.sha256 != source.semantic_sha256:
            _integrity_fail("RNG source changed after its benchmark audit")
        for receipt in nested.receipts:
            raw[(cell.cell_id, receipt.mode.value)] = {
                "cell": cell,
                "mode": receipt.mode,
                "differences": tuple(float(item) for item in receipt.differences),
                "baseline": receipt.baseline_draws,
                "candidate": receipt.candidate_draws,
                "event_keys": receipt.event_keys,
                "source_sha256": nested.sha256,
                "reason": None,
            }

    variance_by_cell: dict[str, float | None] = {}
    for cell in plan.rng_cells:
        item = raw[(cell.cell_id, RNGCouplingMode.INDEPENDENT.value)]
        differences = cast(tuple[float, ...], item["differences"])
        variance_by_cell[cell.cell_id] = (
            None if not differences else _sample_variance(differences)
        )

    rows: list[RQ1RNGRow] = []
    for cell in plan.rng_cells:
        for mode in sorted(RNGCouplingMode, key=lambda item: item.value):
            item = raw[(cell.cell_id, mode.value)]
            differences = cast(tuple[float, ...], item["differences"])
            source_reason = cast(str | None, item["reason"])
            if source_reason is not None:
                blocked = _blocked_scalars(source_reason, 7)
                ratio = ReportedScalar.unavailable(
                    ScalarStatus.BLOCKED_SOURCE_FAILURE, source_reason
                )
                rows.append(
                    RQ1RNGRow(
                        cell_id=cell.cell_id,
                        mode=mode.value,
                        branch_step=cell.branch_step,
                        branch_draw_count=cell.branch_draw_count,
                        expected=plan.rng_replicate_count,
                        completed=0,
                        failed=plan.rng_replicate_count,
                        paired_differences=(),
                        mean_paired_difference=blocked[0],
                        sample_variance=blocked[1],
                        mcse=blocked[2],
                        bias_from_policy_effect=blocked[3],
                        exact_effect_rate=blocked[4],
                        all_shock_match_rate=blocked[5],
                        post_branch_shock_match_rate=blocked[6],
                        variance_ratio_vs_independent=ratio,
                        baseline_draws_sha256=None,
                        event_keys_sha256=None,
                        source_artifact_sha256=None,
                    )
                )
                continue
            baseline = cast(tuple[int, ...], item["baseline"])
            candidate = cast(tuple[int, ...], item["candidate"])
            event_keys = cast(tuple[str, ...], item["event_keys"])
            expected_draws = plan.rng_replicate_count * plan.rng_step_count
            if len(baseline) != expected_draws or len(candidate) != expected_draws:
                _integrity_fail("RNG receipt draw count changed after audit")
            mean = fmean(differences)
            variance = _sample_variance(differences)
            post_pairs: list[tuple[int, int]] = []
            for replicate in range(plan.rng_replicate_count):
                start = replicate * plan.rng_step_count + cell.branch_step + 1
                end = (replicate + 1) * plan.rng_step_count
                post_pairs.extend(
                    zip(baseline[start:end], candidate[start:end], strict=True)
                )
            if not post_pairs:
                _integrity_fail("RNG branch leaves no post-branch exogenous draws")
            denominator = variance_by_cell[cell.cell_id]
            if denominator is None:
                _integrity_fail("completed cell lacks independent variance")
            ratio = (
                ReportedScalar.unavailable(
                    ScalarStatus.UNDEFINED_ZERO_DENOMINATOR,
                    "independent-mode sample variance equals zero",
                )
                if denominator == 0.0
                else ReportedScalar.defined(variance / denominator)
            )
            rows.append(
                RQ1RNGRow(
                    cell_id=cell.cell_id,
                    mode=mode.value,
                    branch_step=cell.branch_step,
                    branch_draw_count=cell.branch_draw_count,
                    expected=plan.rng_replicate_count,
                    completed=plan.rng_replicate_count,
                    failed=0,
                    paired_differences=differences,
                    mean_paired_difference=ReportedScalar.defined(mean),
                    sample_variance=ReportedScalar.defined(variance),
                    mcse=ReportedScalar.defined(
                        math.sqrt(variance / plan.rng_replicate_count)
                    ),
                    bias_from_policy_effect=ReportedScalar.defined(
                        mean - plan.rng_policy_effect
                    ),
                    exact_effect_rate=ReportedScalar.defined(
                        sum(item == plan.rng_policy_effect for item in differences)
                        / len(differences)
                    ),
                    all_shock_match_rate=ReportedScalar.defined(
                        sum(
                            left == right
                            for left, right in zip(baseline, candidate, strict=True)
                        )
                        / len(baseline)
                    ),
                    post_branch_shock_match_rate=ReportedScalar.defined(
                        sum(left == right for left, right in post_pairs)
                        / len(post_pairs)
                    ),
                    variance_ratio_vs_independent=ratio,
                    baseline_draws_sha256=_semantic_hash(list(baseline)),
                    event_keys_sha256=_semantic_hash(list(event_keys)),
                    source_artifact_sha256=cast(str, item["source_sha256"]),
                )
            )

    invariants: list[RQ1BaselineEqualityInvariant] = []
    cell_ids = tuple(cell.cell_id for cell in plan.rng_cells)
    for mode in sorted(RNGCouplingMode, key=lambda item: item.value):
        mode_rows = tuple(item for item in rows if item.mode == mode.value)
        if any(item.completed == 0 for item in mode_rows):
            invariants.append(
                RQ1BaselineEqualityInvariant(
                    mode=mode.value,
                    cell_ids=cell_ids,
                    baseline_draws_sha256=None,
                    event_keys_sha256=None,
                    passed=None,
                    reason="one or more preregistered stress cells failed",
                )
            )
            continue
        draw_hashes = {item.baseline_draws_sha256 for item in mode_rows}
        key_hashes = {item.event_keys_sha256 for item in mode_rows}
        if len(draw_hashes) != 1 or len(key_hashes) != 1:
            _integrity_fail("cross-cell baseline exogenous equality invariant failed")
        draw_digest = next(iter(draw_hashes))
        key_digest = next(iter(key_hashes))
        if draw_digest is None or key_digest is None:
            _integrity_fail("passing baseline invariant lacks source commitments")
        invariants.append(
            RQ1BaselineEqualityInvariant(
                mode=mode.value,
                cell_ids=cell_ids,
                baseline_draws_sha256=draw_digest,
                event_keys_sha256=key_digest,
                passed=True,
                reason=None,
            )
        )
    return tuple(rows), tuple(invariants)


def _derive_mechanism(
    directory: Path,
    plan: AAMASBenchmarkPlan,
    artifact: AAMASBenchmarkArtifact,
    spec: FrozenAAMASReportingSpec,
) -> tuple[tuple[RQ1MechanismRow, ...], tuple[RQ1MechanismSummary, ...]]:
    expected = next(
        item
        for item in plan.expected_attempts
        if item.kind is AAMASAttemptKind.MECHANISM_ABLATION
    )
    attempt = next(
        item for item in artifact.attempts if item.attempt_id == expected.attempt_id
    )
    source = next(
        (
            item
            for item in artifact.source_artifacts
            if item.attempt_id == expected.attempt_id
        ),
        None,
    )
    variants = plan.mechanism_ablation_spec.variants
    rows: list[RQ1MechanismRow] = []
    if source is None:
        reason = attempt.failure_message or "mechanism-ablation source attempt failed"
        failure_type = attempt.failure_code or "missing_source_artifact"
        for variant in variants:
            for replicate in range(plan.mechanism_replicate_count):
                rows.append(
                    RQ1MechanismRow(
                        variant=variant.value,
                        replicate_index=replicate,
                        status=MechanismAblationStatus.FAILED.value,
                        event_key=plan.mechanism_ablation_spec.event_key_for(
                            replicate
                        ).value,
                        metric_value=None,
                        failure_stage="aamas_benchmark_attempt",
                        failure_type=failure_type,
                        failure_message=reason,
                        paired_outcome_sha256=None,
                    )
                )
    else:
        if source.kind is not AAMASSourceKind.MECHANISM_ABLATION:
            _integrity_fail("mechanism attempt points to the wrong audited source kind")
        try:
            stored = read_json_object(directory / source.relative_path)
        except (OSError, MetadataValidationError) as error:
            raise AAMASReportingIntegrityError(
                "cannot read the already-audited mechanism source"
            ) from error
        if _semantic_hash(stored) != source.semantic_sha256:
            _integrity_fail("mechanism source changed after its benchmark audit")
        raw_results = _array(stored.get("results"), "mechanism results")
        expected_identities = tuple(
            (variant.value, replicate)
            for variant in variants
            for replicate in range(plan.mechanism_replicate_count)
        )
        observed_identities: list[tuple[str, int]] = []
        for raw in raw_results:
            result = _mapping(raw, "mechanism result")
            raw_variant = _string(result.get("variant"), "mechanism variant")
            replicate = _integer(result.get("replicate_index"), "replicate_index")
            observed_identities.append((raw_variant, replicate))
            event_key_mapping = _mapping(result.get("event_key"), "mechanism event_key")
            event_key = _string(
                event_key_mapping.get("value"), "mechanism event key value"
            )
            status = _string(result.get("status"), "mechanism status")
            try:
                typed_status = MechanismAblationStatus(status)
            except ValueError as error:
                raise AAMASReportingIntegrityError(
                    "audited mechanism source contains an unknown status"
                ) from error
            if typed_status is MechanismAblationStatus.COMPLETED:
                receipt = _mapping(result.get("receipt"), "mechanism receipt")
                cells = _array(receipt.get("metric_cells"), "metric_cells")
                matches: list[float] = []
                for raw_cell in cells:
                    cell = _mapping(raw_cell, "metric cell")
                    if (
                        cell.get("horizon_id") == spec.mechanism_horizon_id
                        and cell.get("group_id") == spec.mechanism_group_id
                        and cell.get("metric_name") == spec.mechanism_metric_name
                    ):
                        matches.append(
                            _finite(cell.get("value"), "mechanism metric value")
                        )
                if len(matches) != 1:
                    _integrity_fail(
                        "audited mechanism source lacks exactly one registered "
                        "main cell"
                    )
                rows.append(
                    RQ1MechanismRow(
                        variant=raw_variant,
                        replicate_index=replicate,
                        status=status,
                        event_key=event_key,
                        metric_value=matches[0],
                        failure_stage=None,
                        failure_type=None,
                        failure_message=None,
                        paired_outcome_sha256=_sha256(
                            receipt.get("paired_outcome_sha256"),
                            "paired_outcome_sha256",
                        ),
                    )
                )
            else:
                failure = _mapping(result.get("failure"), "mechanism failure")
                rows.append(
                    RQ1MechanismRow(
                        variant=raw_variant,
                        replicate_index=replicate,
                        status=status,
                        event_key=event_key,
                        metric_value=None,
                        failure_stage=_string(failure.get("stage"), "failure stage"),
                        failure_type=_string(
                            failure.get("exception_type"), "failure exception_type"
                        ),
                        failure_message=_string(
                            failure.get("message"), "failure message"
                        ),
                        paired_outcome_sha256=None,
                    )
                )
        if tuple(observed_identities) != expected_identities:
            _integrity_fail("mechanism source row registry changed after audit")

    row_tuple = tuple(rows)
    grid_failed = any(
        item.status != MechanismAblationStatus.COMPLETED.value for item in row_tuple
    )
    summaries: list[RQ1MechanismSummary] = []
    for variant in variants:
        variant_rows = tuple(
            item for item in row_tuple if item.variant == variant.value
        )
        full_draws = tuple(item.metric_value for item in variant_rows)
        completed = sum(item.metric_value is not None for item in variant_rows)
        failed = len(variant_rows) - completed
        if grid_failed:
            blocked = _blocked_scalars(
                "the preregistered mechanism grid contains at least one failed row; "
                "complete-case aggregation is forbidden",
                4,
            )
            summary_values = blocked
        else:
            values = tuple(cast(float, item) for item in full_draws)
            summary_values = (
                ReportedScalar.defined(fmean(values)),
                ReportedScalar.defined(
                    _quantile(values, spec.quantile_probabilities[0])
                ),
                ReportedScalar.defined(
                    _quantile(values, spec.quantile_probabilities[1])
                ),
                ReportedScalar.defined(
                    _quantile(values, spec.quantile_probabilities[2])
                ),
            )
        summaries.append(
            RQ1MechanismSummary(
                variant=variant.value,
                expected=len(variant_rows),
                completed=completed,
                failed=failed,
                full_draws=full_draws,
                mean=summary_values[0],
                q05=summary_values[1],
                q50=summary_values[2],
                q95=summary_values[3],
            )
        )
    return row_tuple, tuple(summaries)


@dataclass(frozen=True, slots=True)
class _CalibrationSources:
    spec: AAMASCalibrationSpec
    attempt_rows: Mapping[str, Mapping[str, object]]
    truth_results: Mapping[str, Mapping[str, object]]
    risk_mapping: Mapping[str, object]


def _load_calibration_sources(directory: Path) -> _CalibrationSources:
    try:
        calibration_spec = AAMASCalibrationSpec.from_mapping(
            read_json_object(directory / "spec.json")
        )
        attempt_registry = read_json_object(directory / "attempt_registry.json")
        results = read_json_object(directory / "results.json")
    except (OSError, ValueError, MetadataValidationError) as error:
        raise AAMASReportingIntegrityError(
            "cannot reconstruct already-audited calibration sources"
        ) from error
    raw_attempts = _array(attempt_registry.get("attempts"), "attempt registry")
    attempts: dict[str, Mapping[str, object]] = {}
    for raw in raw_attempts:
        row = _mapping(raw, "attempt registry row")
        attempt_id = _string(row.get("attempt_id"), "attempt_id")
        if attempt_id in attempts:
            _integrity_fail("calibration source contains a duplicate attempt row")
        attempts[attempt_id] = row
    raw_truth_results = _array(results.get("truth_results"), "truth_results")
    truth_results: dict[str, Mapping[str, object]] = {}
    for raw in raw_truth_results:
        row = _mapping(raw, "truth result")
        truth_id = _string(row.get("truth_id"), "truth result truth_id")
        if truth_id in truth_results:
            _integrity_fail("calibration results contain a duplicate truth row")
        truth_results[truth_id] = row
    return _CalibrationSources(
        spec=calibration_spec,
        attempt_rows=attempts,
        truth_results=truth_results,
        risk_mapping=_mapping(results.get("risk_coverage"), "risk_coverage"),
    )


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0 or not 0 <= successes <= total:
        _integrity_fail("Wilson interval requires 0 <= successes <= total")
    proportion = successes / total
    z_squared = _WILSON_Z * _WILSON_Z
    denominator = 1.0 + z_squared / total
    center = (proportion + z_squared / (2.0 * total)) / denominator
    half_width = (
        _WILSON_Z
        * math.sqrt(
            proportion * (1.0 - proportion) / total + z_squared / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def _derive_calibration(
    plan: AAMASCalibrationPlan,
    sources: _CalibrationSources,
    spec: FrozenAAMASReportingSpec,
) -> tuple[
    tuple[RQ2ParameterRow, ...],
    tuple[RQ2ParameterSummary, ...],
    tuple[RQ2PPCRow, ...],
    tuple[RQ2PPCSummary, ...],
    tuple[RQ2SBCRankSummary, ...],
]:
    truths = plan.truths
    expected_attempts = plan.expected_attempts
    truth_by_id = {item.truth_id: item for item in truths}
    expected_by_truth = {item.truth_id: item for item in expected_attempts}
    if set(sources.attempt_rows) != {item.attempt_id for item in expected_attempts}:
        _integrity_fail("calibration attempt rows differ from the rebuilt plan")
    all_completed = True
    parameter_rows: list[RQ2ParameterRow] = []
    ppc_rows: list[RQ2PPCRow] = []
    sbc_by_truth_parameter: dict[
        tuple[str, str], tuple[int, tuple[int, ...], tuple[int, ...], float]
    ] = {}

    for truth_id, truth in truth_by_id.items():
        expected = expected_by_truth[truth_id]
        attempt_id = expected.attempt_id
        attempt = sources.attempt_rows[attempt_id]
        raw_status = _string(attempt.get("status"), "calibration attempt status")
        try:
            status = AttemptStatus(raw_status)
        except ValueError as error:
            raise AAMASReportingIntegrityError(
                "calibration source has an unknown attempt status"
            ) from error
        role = truth.role
        parameters = truth.parameters
        truth_parameters = dict(parameters)
        if tuple(sorted(truth_parameters)) != spec.calibration_parameters:
            _integrity_fail("calibration truth parameter registry changed")
        result = sources.truth_results.get(truth_id)
        if status is AttemptStatus.FAILED:
            all_completed = False
            if result is not None:
                _integrity_fail("failed calibration attempt claims a truth result")
            error_type = _string(attempt.get("error_type"), "calibration error_type")
            error_message = _string(
                attempt.get("error_message"), "calibration error_message"
            )
            for parameter in spec.calibration_parameters:
                parameter_rows.append(
                    RQ2ParameterRow(
                        truth_id=truth_id,
                        role=role.value,
                        parameter=parameter,
                        attempt_status=status.value,
                        truth=truth_parameters[parameter],
                        posterior_median=None,
                        signed_error=None,
                        absolute_error=None,
                        lower_90=None,
                        upper_90=None,
                        covered_90=None,
                        boundary_pileup=None,
                        multimodal=None,
                        nonidentifiable=None,
                        report_sha256=None,
                        failure_type=error_type,
                        failure_message=error_message,
                    )
                )
            for statistic in spec.held_out_statistics:
                ppc_rows.append(
                    RQ2PPCRow(
                        truth_id=truth_id,
                        role=role.value,
                        statistic=statistic,
                        attempt_status=status.value,
                        covered=None,
                        report_sha256=None,
                    )
                )
            continue
        if result is None:
            _integrity_fail("completed calibration attempt lacks a sealed truth result")
        report_sha256 = _sha256(result.get("report_sha256"), "report_sha256")
        recovery_rows = {
            _string(row.get("parameter"), "recovery parameter"): row
            for raw in _array(result.get("recovery"), "recovery")
            for row in (_mapping(raw, "recovery row"),)
        }
        diagnostic_rows = {
            _string(row.get("parameter"), "identifiability parameter"): row
            for raw in _array(result.get("identifiability"), "identifiability")
            for row in (_mapping(raw, "identifiability row"),)
        }
        if set(recovery_rows) != set(spec.calibration_parameters) or set(
            diagnostic_rows
        ) != set(spec.calibration_parameters):
            _integrity_fail("calibration recovery or identifiability registry changed")
        for parameter in spec.calibration_parameters:
            recovery = recovery_rows[parameter]
            diagnostic = diagnostic_rows[parameter]
            median = _finite(recovery.get("posterior_median"), "posterior_median")
            truth_value = _finite(recovery.get("truth"), "recovery truth")
            if truth_value != truth_parameters[parameter]:
                _integrity_fail("recovery truth differs from the sealed truth manifest")
            parameter_rows.append(
                RQ2ParameterRow(
                    truth_id=truth_id,
                    role=role.value,
                    parameter=parameter,
                    attempt_status=status.value,
                    truth=truth_value,
                    posterior_median=median,
                    signed_error=median - truth_value,
                    absolute_error=_finite(
                        recovery.get("absolute_error"), "absolute_error"
                    ),
                    lower_90=_finite(recovery.get("lower_90"), "lower_90"),
                    upper_90=_finite(recovery.get("upper_90"), "upper_90"),
                    covered_90=_boolean(recovery.get("covered"), "covered"),
                    boundary_pileup=_boolean(
                        diagnostic.get("boundary_pileup"), "boundary_pileup"
                    ),
                    multimodal=_boolean(diagnostic.get("multimodal"), "multimodal"),
                    nonidentifiable=not _boolean(
                        diagnostic.get("estimable_for_reporting"),
                        "estimable_for_reporting",
                    ),
                    report_sha256=report_sha256,
                    failure_type=None,
                    failure_message=None,
                )
            )

        ppc_by_name: dict[str, Mapping[str, object]] = {}
        for raw in _array(result.get("ppc"), "PPC"):
            row = _mapping(raw, "PPC row")
            if row.get("role") != "held_out_validation":
                continue
            name = _string(row.get("name"), "PPC name")
            if name in ppc_by_name:
                _integrity_fail("held-out PPC source contains a duplicate statistic")
            ppc_by_name[name] = row
        if set(ppc_by_name) != set(spec.held_out_statistics):
            _integrity_fail("held-out PPC statistic registry changed")
        for statistic in spec.held_out_statistics:
            ppc_rows.append(
                RQ2PPCRow(
                    truth_id=truth_id,
                    role=role.value,
                    statistic=statistic,
                    attempt_status=status.value,
                    covered=_boolean(
                        ppc_by_name[statistic].get("covered"), "PPC covered"
                    ),
                    report_sha256=report_sha256,
                )
            )

        sbc = _mapping(result.get("sbc"), "SBC")
        sbc_parameters = {
            _string(row.get("parameter"), "SBC parameter"): row
            for raw in _array(sbc.get("parameters"), "SBC parameters")
            for row in (_mapping(raw, "SBC parameter row"),)
        }
        if set(sbc_parameters) != set(spec.calibration_parameters):
            _integrity_fail("SBC parameter registry changed")
        for parameter in spec.calibration_parameters:
            row = sbc_parameters[parameter]
            draws = _integer(
                row.get("posterior_draws_per_replicate"),
                "SBC posterior_draws_per_replicate",
                minimum=2,
            )
            ranks = tuple(
                _integer(item, "SBC rank")
                for item in _array(row.get("ranks"), "SBC ranks")
            )
            histogram = tuple(
                _integer(item, "SBC histogram count")
                for item in _array(row.get("histogram"), "SBC histogram")
            )
            coverage = _fraction(
                row.get("interval_coverage_90"), "SBC interval_coverage_90"
            )
            sbc_by_truth_parameter[(truth_id, parameter)] = (
                draws,
                ranks,
                histogram,
                coverage,
            )

    parameter_tuple = tuple(parameter_rows)
    ppc_tuple = tuple(ppc_rows)
    summaries: list[RQ2ParameterSummary] = []
    ppc_summaries: list[RQ2PPCSummary] = []
    sbc_summaries: list[RQ2SBCRankSummary] = []
    blocked_reason = (
        "at least one expected calibration attempt failed; complete-case aggregation "
        "is forbidden"
    )
    for role_value in spec.calibration_roles:
        for parameter in spec.calibration_parameters:
            selected = tuple(
                item
                for item in parameter_tuple
                if item.role == role_value and item.parameter == parameter
            )
            completed = sum(item.attempt_status == "completed" for item in selected)
            failed = len(selected) - completed
            if not all_completed:
                scalars = _blocked_scalars(blocked_reason, 11)
                interval = TypedInterval.unavailable(
                    ScalarStatus.BLOCKED_SOURCE_FAILURE,
                    spec.coverage_interval_kind,
                    spec.coverage_interval_nominal_probability,
                    blocked_reason,
                )
            else:
                signed = tuple(cast(float, item.signed_error) for item in selected)
                absolute = tuple(cast(float, item.absolute_error) for item in selected)
                widths = tuple(
                    cast(float, item.upper_90) - cast(float, item.lower_90)
                    for item in selected
                )
                covered_count = sum(item.covered_90 is True for item in selected)
                boundary_count = sum(item.boundary_pileup is True for item in selected)
                multimodal_count = sum(item.multimodal is True for item in selected)
                nonidentifiable_count = sum(
                    item.nonidentifiable is True for item in selected
                )
                scalars = (
                    ReportedScalar.defined(fmean(signed)),
                    ReportedScalar.defined(fmean(absolute)),
                    ReportedScalar.defined(
                        math.sqrt(fmean(tuple(item * item for item in signed)))
                    ),
                    ReportedScalar.defined(covered_count / len(selected)),
                    ReportedScalar.defined(fmean(widths)),
                    ReportedScalar.defined(boundary_count),
                    ReportedScalar.defined(boundary_count / len(selected)),
                    ReportedScalar.defined(multimodal_count),
                    ReportedScalar.defined(multimodal_count / len(selected)),
                    ReportedScalar.defined(nonidentifiable_count),
                    ReportedScalar.defined(nonidentifiable_count / len(selected)),
                )
                if role_value == TruthRole.TARGET.value:
                    lower, upper = _wilson_interval(covered_count, len(selected))
                    interval = TypedInterval.defined(
                        spec.coverage_interval_kind,
                        spec.coverage_interval_nominal_probability,
                        lower,
                        upper,
                    )
                else:
                    interval = TypedInterval.unavailable(
                        ScalarStatus.NOT_APPLICABLE,
                        spec.coverage_interval_kind,
                        spec.coverage_interval_nominal_probability,
                        "the frozen Wilson interval is registered for target "
                        "recovery only",
                    )
            summaries.append(
                RQ2ParameterSummary(
                    role=role_value,
                    parameter=parameter,
                    n=len(selected),
                    completed=completed,
                    failed=failed,
                    signed_bias=scalars[0],
                    mae=scalars[1],
                    rmse=scalars[2],
                    empirical_90_coverage=scalars[3],
                    mean_interval_width=scalars[4],
                    binomial_wilson_interval=interval,
                    boundary_pileup_count=scalars[5],
                    boundary_pileup_rate=scalars[6],
                    multimodality_count=scalars[7],
                    multimodality_rate=scalars[8],
                    nonidentifiability_count=scalars[9],
                    nonidentifiability_rate=scalars[10],
                )
            )

            role_truth_ids = tuple(
                item.truth_id for item in truths if item.role.value == role_value
            )
            if not all_completed:
                sbc_summaries.append(
                    RQ2SBCRankSummary(
                        role=role_value,
                        parameter=parameter,
                        expected_truths=len(role_truth_ids),
                        completed_truths=sum(
                            (truth_id, parameter) in sbc_by_truth_parameter
                            for truth_id in role_truth_ids
                        ),
                        failed_truths=sum(
                            (truth_id, parameter) not in sbc_by_truth_parameter
                            for truth_id in role_truth_ids
                        ),
                        posterior_draws_per_replicate=None,
                        ranks=(),
                        histogram=(),
                        interval_coverage_90=ReportedScalar.unavailable(
                            ScalarStatus.BLOCKED_SOURCE_FAILURE,
                            blocked_reason,
                        ),
                    )
                )
            else:
                source_sbc = tuple(
                    sbc_by_truth_parameter[(truth_id, parameter)]
                    for truth_id in role_truth_ids
                )
                draw_counts = {item[0] for item in source_sbc}
                if len(draw_counts) != 1:
                    _integrity_fail("SBC posterior draw count differs across truths")
                draws = next(iter(draw_counts))
                ranks = tuple(rank for item in source_sbc for rank in item[1])
                histogram = tuple(
                    sum(item[2][index] for item in source_sbc)
                    for index in range(draws + 1)
                )
                total_rank_count = sum(len(item[1]) for item in source_sbc)
                sbc_coverage = (
                    sum(item[3] * len(item[1]) for item in source_sbc)
                    / total_rank_count
                )
                sbc_summaries.append(
                    RQ2SBCRankSummary(
                        role=role_value,
                        parameter=parameter,
                        expected_truths=len(role_truth_ids),
                        completed_truths=len(role_truth_ids),
                        failed_truths=0,
                        posterior_draws_per_replicate=draws,
                        ranks=ranks,
                        histogram=histogram,
                        interval_coverage_90=ReportedScalar.defined(sbc_coverage),
                    )
                )

        for statistic in spec.held_out_statistics:
            selected_ppc = tuple(
                item
                for item in ppc_tuple
                if item.role == role_value and item.statistic == statistic
            )
            completed = sum(item.attempt_status == "completed" for item in selected_ppc)
            failed = len(selected_ppc) - completed
            ppc_coverage = (
                ReportedScalar.unavailable(
                    ScalarStatus.BLOCKED_SOURCE_FAILURE, blocked_reason
                )
                if not all_completed
                else ReportedScalar.defined(
                    sum(item.covered is True for item in selected_ppc)
                    / len(selected_ppc)
                )
            )
            ppc_summaries.append(
                RQ2PPCSummary(
                    role=role_value,
                    statistic=statistic,
                    n=len(selected_ppc),
                    completed=completed,
                    failed=failed,
                    held_out_coverage=ppc_coverage,
                )
            )
    return (
        parameter_tuple,
        tuple(summaries),
        ppc_tuple,
        tuple(ppc_summaries),
        tuple(sbc_summaries),
    )


def _bootstrap_target_loss(
    records: tuple[RQ3TargetRecord, ...],
    *,
    threshold_index: int,
    reporting_spec: FrozenAAMASReportingSpec,
) -> tuple[tuple[float, ...], tuple[str, ...]]:
    rng = HashBasedRNG(reporting_spec.bootstrap_seed)
    draws: list[float] = []
    event_keys: list[str] = []
    for bootstrap_index in range(reporting_spec.bootstrap_resample_count):
        event_key = SemanticEventKey(
            scenario_id="aamas.synthetic.reporting",
            event_type=EventType.EVALUATION_ORDER_OUTCOME,
            entity_id=f"risk_t{threshold_index:02d}_b{bootstrap_index:04d}",
            timestamp=reporting_spec.bootstrap_event_timestamp,
        )
        event_keys.append(event_key.value)
        resample = tuple(
            records[
                rng.integer(
                    event_key,
                    0,
                    len(records) - 1,
                    domain=reporting_spec.bootstrap_domain,
                    draw_index=position,
                )
            ].loss
            for position in range(len(records))
        )
        draws.append(fmean(resample))
    return tuple(draws), tuple(event_keys)


def _derive_risk(
    calibration_plan: AAMASCalibrationPlan,
    calibration_sources: _CalibrationSources,
    parameter_rows: tuple[RQ2ParameterRow, ...],
    reporting_spec: FrozenAAMASReportingSpec,
) -> tuple[tuple[RQ3TargetRecord, ...], tuple[RQ3RiskCoveragePoint, ...]]:
    truths = calibration_plan.truths
    target_ids = tuple(
        item.truth_id for item in truths if item.role is TruthRole.TARGET
    )
    risk = calibration_sources.risk_mapping
    if risk.get("role") != TruthRole.TARGET.value:
        _integrity_fail("risk source role is not target")
    raw_records = _array(risk.get("records"), "risk records")
    by_id: dict[str, RQ3TargetRecord] = {}
    for raw in raw_records:
        row = _mapping(raw, "risk record")
        prediction_id = _string(row.get("prediction_id"), "prediction_id")
        if prediction_id in by_id:
            _integrity_fail("risk source contains a duplicate prediction")
        by_id[prediction_id] = RQ3TargetRecord(
            truth_id=prediction_id,
            uncertainty_score=_finite(
                row.get("uncertainty_score"), "risk uncertainty_score"
            ),
            loss=_finite(row.get("loss"), "risk loss"),
        )
    if not set(by_id).issubset(set(target_ids)):
        _integrity_fail("design truth or unknown truth leaked into target risk records")
    all_attempts_completed = all(
        item.attempt_status == AttemptStatus.COMPLETED.value for item in parameter_rows
    )
    if all_attempts_completed and set(by_id) != set(target_ids):
        _integrity_fail("complete calibration source omitted target risk records")
    records = tuple(by_id[item] for item in target_ids if item in by_id)
    source_complete = len(records) == len(target_ids)
    points: list[RQ3RiskCoveragePoint] = []
    for threshold_index, threshold in enumerate(
        calibration_sources.spec.uncertainty_thresholds
    ):
        if not source_complete:
            reason = (
                "target risk records are blocked because at least one expected "
                "calibration attempt failed"
            )
            points.append(
                RQ3RiskCoveragePoint(
                    threshold=threshold,
                    expected_target_n=len(target_ids),
                    source_record_n=len(records),
                    retained_n=None,
                    minimum_retained=calibration_sources.spec.minimum_retained,
                    selection_coverage=ReportedScalar.unavailable(
                        ScalarStatus.BLOCKED_SOURCE_FAILURE, reason
                    ),
                    mean_loss=ReportedScalar.unavailable(
                        ScalarStatus.BLOCKED_SOURCE_FAILURE, reason
                    ),
                    resampling_interval=TypedInterval.unavailable(
                        ScalarStatus.BLOCKED_SOURCE_FAILURE,
                        reporting_spec.bootstrap_interval_kind,
                        reporting_spec.bootstrap_quantiles[1]
                        - reporting_spec.bootstrap_quantiles[0],
                        reason,
                    ),
                    bootstrap_draws=(),
                    bootstrap_event_keys=(),
                    bootstrap_event_keys_sha256=None,
                )
            )
            continue
        selected = tuple(
            item for item in records if item.uncertainty_score <= threshold
        )
        coverage = ReportedScalar.defined(len(selected) / len(records))
        if len(selected) < calibration_sources.spec.minimum_retained:
            reason = (
                f"retained {len(selected)} targets, below frozen minimum "
                f"{calibration_sources.spec.minimum_retained}"
            )
            points.append(
                RQ3RiskCoveragePoint(
                    threshold=threshold,
                    expected_target_n=len(target_ids),
                    source_record_n=len(records),
                    retained_n=len(selected),
                    minimum_retained=calibration_sources.spec.minimum_retained,
                    selection_coverage=coverage,
                    mean_loss=ReportedScalar.unavailable(
                        ScalarStatus.UNDEFINED_MINIMUM_RETAINED, reason
                    ),
                    resampling_interval=TypedInterval.unavailable(
                        ScalarStatus.UNDEFINED_MINIMUM_RETAINED,
                        reporting_spec.bootstrap_interval_kind,
                        reporting_spec.bootstrap_quantiles[1]
                        - reporting_spec.bootstrap_quantiles[0],
                        reason,
                    ),
                    bootstrap_draws=(),
                    bootstrap_event_keys=(),
                    bootstrap_event_keys_sha256=None,
                )
            )
            continue
        bootstrap_draws, event_keys = _bootstrap_target_loss(
            selected,
            threshold_index=threshold_index,
            reporting_spec=reporting_spec,
        )
        points.append(
            RQ3RiskCoveragePoint(
                threshold=threshold,
                expected_target_n=len(target_ids),
                source_record_n=len(records),
                retained_n=len(selected),
                minimum_retained=calibration_sources.spec.minimum_retained,
                selection_coverage=coverage,
                mean_loss=ReportedScalar.defined(
                    fmean(tuple(item.loss for item in selected))
                ),
                resampling_interval=TypedInterval.defined(
                    reporting_spec.bootstrap_interval_kind,
                    reporting_spec.bootstrap_quantiles[1]
                    - reporting_spec.bootstrap_quantiles[0],
                    _quantile(bootstrap_draws, reporting_spec.bootstrap_quantiles[0]),
                    _quantile(bootstrap_draws, reporting_spec.bootstrap_quantiles[1]),
                ),
                bootstrap_draws=bootstrap_draws,
                bootstrap_event_keys=event_keys,
                bootstrap_event_keys_sha256=_semantic_hash(list(event_keys)),
            )
        )
    return records, tuple(points)


def _validity_entry(
    ppc_rows: tuple[RQ2PPCRow, ...],
) -> ValidityCertificateEntry:
    target_ppc = tuple(item for item in ppc_rows if item.role == TruthRole.TARGET.value)
    ppc_failure: float | None = None
    if target_ppc and all(item.covered is not None for item in target_ppc):
        ppc_failure = 1.0 - sum(item.covered is True for item in target_ppc) / len(
            target_ppc
        )
    evidence = ValidityEvidence(
        temporal_leakage_detected=None,
        pre_policy_ppc_failure_rate=ppc_failure,
        ood_policy_distance=None,
        model_disagreement=None,
        seed_instability=None,
        empirical_coverage_history=None,
        observed_group_labels=None,
        active_mechanisms=None,
        prompt_instability=None,
        parameter_support_failure_rate=None,
    )
    evidence_mapping = evidence.to_mapping()
    missing = tuple(
        sorted(key for key, value in evidence_mapping.items() if value is None)
    )
    return ValidityCertificateEntry(
        level="abstain",
        certificate_mapping=None,
        source_derived_evidence=tuple(sorted(evidence_mapping.items())),
        missing_evidence_fields=missing,
        reason=(
            "reporting sources provide no independent temporal, OOD, ensemble, seed, "
            "coverage-history, group-label, or required-mechanism authority; umbrella "
            "ValidityCertificate thresholds and typed upstream receipts are required"
        ),
    )


def derive_aamas_submission_artifact(
    *,
    rq1_directory: Path,
    rq1_plan: AAMASBenchmarkPlan,
    calibration_directory: Path,
    calibration_seal_sha256: str,
    reporting_spec: FrozenAAMASReportingSpec | None = None,
) -> AAMASSubmissionArtifact:
    """Audit immutable source snapshots, then derive frozen synthetic outputs.

    The first audit establishes the caller-visible source identity.  Reporting
    then copies both bundles to private temporary directories, audits those
    copies against the same external commitments, derives exclusively from the
    copied bytes, and audits them once more before returning.  This prevents an
    audit-then-reread race from pairing old source digests with changed values.
    """

    if not isinstance(rq1_directory, Path) or not isinstance(
        calibration_directory, Path
    ):
        _fail("source directories must be pathlib.Path values")
    if not isinstance(rq1_plan, AAMASBenchmarkPlan):
        _fail("rq1_plan must be an AAMASBenchmarkPlan")
    seal = _sha256(calibration_seal_sha256, "calibration_seal_sha256")
    reporting_spec = (
        FrozenAAMASReportingSpec() if reporting_spec is None else reporting_spec
    )
    if not isinstance(reporting_spec, FrozenAAMASReportingSpec):
        _fail("reporting_spec must be a FrozenAAMASReportingSpec")

    # These are verifier-owned replay audits, not shape checks.
    original_rq1_audit = audit_aamas_benchmark(rq1_directory, rq1_plan)
    original_calibration_audit = audit_aamas_calibration_collection(
        calibration_directory,
        expected_seal_sha256=seal,
    )

    with tempfile.TemporaryDirectory(prefix="policytwin-aamas-reporting-") as raw:
        snapshot_root = Path(raw)
        snapshot_rq1 = snapshot_root / "rq1"
        snapshot_calibration = snapshot_root / "calibration"
        try:
            shutil.copytree(rq1_directory, snapshot_rq1, symlinks=True)
            shutil.copytree(
                calibration_directory,
                snapshot_calibration,
                symlinks=True,
            )
        except OSError as error:
            raise AAMASReportingIntegrityError(
                "cannot create private reporting source snapshots"
            ) from error

        rq1_audit = audit_aamas_benchmark(snapshot_rq1, rq1_plan)
        calibration_audit = audit_aamas_calibration_collection(
            snapshot_calibration,
            expected_seal_sha256=seal,
        )
        if (
            rq1_audit != original_rq1_audit
            or calibration_audit != original_calibration_audit
        ):
            _integrity_fail(
                "reporting source identity changed while creating private snapshots"
            )

        try:
            rq1_artifact = AAMASBenchmarkArtifact.from_mapping(
                read_json_object(snapshot_rq1 / "result.json")
            )
        except (OSError, ValueError, MetadataValidationError) as error:
            raise AAMASReportingIntegrityError(
                "cannot load RQ1 artifact from its audited private snapshot"
            ) from error
        if rq1_artifact.sha256 != rq1_audit.artifact_sha256:
            _integrity_fail("RQ1 snapshot artifact differs from its audit receipt")

        calibration_sources = _load_calibration_sources(snapshot_calibration)
        calibration_plan = build_aamas_calibration_plan(calibration_sources.spec)
        if (
            calibration_audit.spec_sha256 != calibration_sources.spec.sha256
            or calibration_audit.plan_sha256 != calibration_plan.sha256
            or calibration_audit.results_sha256
            != sha256_file(snapshot_calibration / "results.json")
        ):
            _integrity_fail("calibration snapshot differs from its audit receipt")

        rng_rows, invariants = _derive_rng(snapshot_rq1, rq1_plan, rq1_artifact)
        mechanism_rows, mechanism_summaries = _derive_mechanism(
            snapshot_rq1, rq1_plan, rq1_artifact, reporting_spec
        )
        (
            parameter_rows,
            parameter_summaries,
            ppc_rows,
            ppc_summaries,
            sbc_summaries,
        ) = _derive_calibration(calibration_plan, calibration_sources, reporting_spec)
        target_records, risk_points = _derive_risk(
            calibration_plan,
            calibration_sources,
            parameter_rows,
            reporting_spec,
        )
        validity = _validity_entry(ppc_rows)
        registry = _output_registry_values(
            rng_rows=rng_rows,
            invariants=invariants,
            mechanism_rows=mechanism_rows,
            mechanism_summaries=mechanism_summaries,
            parameter_rows=parameter_rows,
            parameter_summaries=parameter_summaries,
            ppc_rows=ppc_rows,
            ppc_summaries=ppc_summaries,
            sbc_summaries=sbc_summaries,
            target_records=target_records,
            risk_points=risk_points,
        )
        source_commitments = AAMASReportingSources(
            rq1_artifact_sha256=rq1_audit.artifact_sha256,
            rq1_plan_sha256=rq1_audit.plan_sha256,
            rq1_audited_source_count=rq1_audit.audited_source_count,
            calibration_seal_raw_sha256=seal,
            calibration_spec_sha256=calibration_audit.spec_sha256,
            calibration_plan_sha256=calibration_audit.plan_sha256,
            calibration_results_sha256=calibration_audit.results_sha256,
            calibration_expected_attempt_count=(
                calibration_audit.expected_attempt_count
            ),
            calibration_completed_attempt_count=(
                calibration_audit.completed_attempt_count
            ),
        )
        artifact = AAMASSubmissionArtifact(
            reporting_spec=reporting_spec,
            reporting_spec_sha256=reporting_spec.sha256,
            sources=source_commitments,
            output_registry=registry,
            rq1_rng_rows=rng_rows,
            rq1_baseline_invariants=invariants,
            rq1_mechanism_rows=mechanism_rows,
            rq1_mechanism_summaries=mechanism_summaries,
            rq2_parameter_rows=parameter_rows,
            rq2_parameter_summaries=parameter_summaries,
            rq2_ppc_rows=ppc_rows,
            rq2_ppc_summaries=ppc_summaries,
            rq2_sbc_summaries=sbc_summaries,
            rq3_target_records=target_records,
            rq3_risk_coverage=risk_points,
            validity_certificate_entry=validity,
            formal_source_scale=(
                rq1_plan.is_formal_scale and calibration_sources.spec.is_formal_scale
            ),
        )

        if (
            audit_aamas_benchmark(snapshot_rq1, rq1_plan) != rq1_audit
            or audit_aamas_calibration_collection(
                snapshot_calibration,
                expected_seal_sha256=seal,
            )
            != calibration_audit
        ):
            _integrity_fail("private reporting sources changed during derivation")
        return artifact


def write_aamas_submission_artifact(
    directory: Path,
    artifact: AAMASSubmissionArtifact,
) -> tuple[Path, Path]:
    """Atomically publish one immutable artifact and exact-byte seal."""

    if not isinstance(directory, Path) or not isinstance(
        artifact, AAMASSubmissionArtifact
    ):
        _fail("writer requires a Path and AAMASSubmissionArtifact")
    if directory.exists() or directory.is_symlink():
        _integrity_fail("AAMAS submission artifact directory already exists")
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = directory.with_name(
        f".{directory.name}.{artifact.sha256[:16]}.staging"
    )
    if staging.exists() or staging.is_symlink():
        _integrity_fail("AAMAS submission artifact staging path already exists")
    try:
        staging.mkdir()
        artifact_path = staging / AAMAS_SUBMISSION_FILENAME
        seal_path = staging / AAMAS_SUBMISSION_SEAL_FILENAME
        write_json_file(artifact_path, artifact.to_mapping())
        write_text_file(seal_path, f"{sha256_file(artifact_path)}\n")
        _audit_submission_artifact_membership(staging)
        os.replace(staging, directory)
        _audit_submission_artifact_membership(directory)
    except Exception:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise
    return (
        directory / AAMAS_SUBMISSION_FILENAME,
        directory / AAMAS_SUBMISSION_SEAL_FILENAME,
    )


def _audit_submission_artifact_membership(directory: Path) -> None:
    if not directory.is_dir() or directory.is_symlink():
        _integrity_fail("submission artifact root must be a real directory")
    expected = frozenset(
        {AAMAS_SUBMISSION_FILENAME, AAMAS_SUBMISSION_SEAL_FILENAME}
    )
    try:
        members = frozenset(item.name for item in directory.iterdir())
    except OSError as error:
        raise AAMASReportingIntegrityError(
            "cannot enumerate stored AAMAS submission artifact"
        ) from error
    if members != expected:
        _integrity_fail("submission artifact directory membership changed")
    for name in expected:
        path = directory / name
        try:
            info = path.lstat()
        except OSError as error:
            raise AAMASReportingIntegrityError(
                "cannot inspect stored AAMAS submission artifact"
            ) from error
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            _integrity_fail(
                "submission artifact entries must be singly linked regular files"
            )


@dataclass(frozen=True, slots=True)
class AAMASSubmissionAudit:
    artifact_sha256: str
    reporting_spec_sha256: str
    rq1_artifact_sha256: str
    calibration_results_sha256: str
    output_count: int
    valid: Literal[True] = True

    def __post_init__(self) -> None:
        for field in (
            "artifact_sha256",
            "reporting_spec_sha256",
            "rq1_artifact_sha256",
            "calibration_results_sha256",
        ):
            _sha256(getattr(self, field), field)
        _integer(self.output_count, "output_count", minimum=1)
        if self.valid is not True:
            _fail("only a positive submission audit receipt may be constructed")


def audit_aamas_submission_artifact(
    *,
    directory: Path,
    rq1_directory: Path,
    rq1_plan: AAMASBenchmarkPlan,
    calibration_directory: Path,
    calibration_seal_sha256: str,
    expected_reporting_spec: FrozenAAMASReportingSpec,
) -> AAMASSubmissionAudit:
    """Re-audit sources, re-derive rows, and compare stored semantics."""

    if not isinstance(expected_reporting_spec, FrozenAAMASReportingSpec):
        _fail("expected_reporting_spec must be frozen and typed")
    _audit_submission_artifact_membership(directory)
    try:
        stored = AAMASSubmissionArtifact.from_mapping(
            read_json_object(directory / AAMAS_SUBMISSION_FILENAME)
        )
        seal_text = (directory / AAMAS_SUBMISSION_SEAL_FILENAME).read_text(
            encoding="ascii"
        )
    except (OSError, UnicodeError, ValueError, MetadataValidationError) as error:
        raise AAMASReportingIntegrityError(
            "cannot parse stored AAMAS submission artifact"
        ) from error
    if not seal_text.endswith("\n") or seal_text.count("\n") != 1:
        _integrity_fail("submission artifact seal is not canonical")
    try:
        seal = _sha256(seal_text[:-1], "submission artifact seal")
    except AAMASReportingSchemaError as error:
        raise AAMASReportingIntegrityError(
            "submission artifact seal is invalid"
        ) from error
    if stored.reporting_spec != expected_reporting_spec:
        _integrity_fail("stored reporting spec differs from its external commitment")
    try:
        raw_sha256 = sha256_file(directory / AAMAS_SUBMISSION_FILENAME)
    except OSError as error:
        raise AAMASReportingIntegrityError(
            "cannot hash stored AAMAS submission artifact"
        ) from error
    if seal != raw_sha256:
        _integrity_fail("submission artifact exact-byte seal does not match")
    replay = derive_aamas_submission_artifact(
        rq1_directory=rq1_directory,
        rq1_plan=rq1_plan,
        calibration_directory=calibration_directory,
        calibration_seal_sha256=calibration_seal_sha256,
        reporting_spec=expected_reporting_spec,
    )
    if stored.to_mapping() != replay.to_mapping():
        _integrity_fail("submission artifact differs from audited source re-derivation")
    return AAMASSubmissionAudit(
        artifact_sha256=stored.sha256,
        reporting_spec_sha256=stored.reporting_spec_sha256,
        rq1_artifact_sha256=stored.sources.rq1_artifact_sha256,
        calibration_results_sha256=stored.sources.calibration_results_sha256,
        output_count=len(stored.output_registry),
    )


__all__ = [
    "AAMAS_REPORTING_SPEC_SCHEMA_VERSION",
    "AAMAS_SUBMISSION_ARTIFACT_SCHEMA_VERSION",
    "AAMAS_SUBMISSION_CLAIM_SCOPE",
    "AAMASReportingError",
    "AAMASReportingIntegrityError",
    "AAMASReportingSchemaError",
    "AAMASReportingSources",
    "AAMASSubmissionArtifact",
    "AAMASSubmissionAudit",
    "FrozenAAMASReportingSpec",
    "OutputRegistryEntry",
    "RQ1BaselineEqualityInvariant",
    "RQ1MechanismRow",
    "RQ1MechanismSummary",
    "RQ1RNGRow",
    "RQ2PPCRow",
    "RQ2PPCSummary",
    "RQ2ParameterRow",
    "RQ2ParameterSummary",
    "RQ2SBCRankSummary",
    "RQ3RiskCoveragePoint",
    "RQ3TargetRecord",
    "ReportedScalar",
    "ScalarStatus",
    "TypedInterval",
    "ValidityCertificateEntry",
    "audit_aamas_submission_artifact",
    "derive_aamas_submission_artifact",
    "write_aamas_submission_artifact",
]
