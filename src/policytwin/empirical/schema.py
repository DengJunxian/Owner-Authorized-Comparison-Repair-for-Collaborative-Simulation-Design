"""Immutable schemas for post-freeze empirical evaluation.

The empirical package deliberately uses relative event time and a common policy
start.  Staggered adoption is outside the current estimand and is rejected rather
than being silently approximated by a two-way fixed-effects regression.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import NoReturn, Self, cast

from policytwin.data.temporal_firewall import TemporalRecord
from policytwin.data.transformation_receipts import (
    TransformationReceipt,
    TransformationReceiptError,
)
from policytwin.provenance import canonical_json_bytes, sha256_bytes

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class EmpiricalEvaluationError(RuntimeError):
    """Base class for fail-closed empirical-evaluation failures."""


class EmpiricalSchemaError(EmpiricalEvaluationError, ValueError):
    """Raised when an empirical schema is incomplete or internally inconsistent."""


class FrozenInputError(EmpiricalEvaluationError):
    """Raised when model, prediction, config, or data freeze proof is invalid."""


class FeedbackProhibitedError(EmpiricalEvaluationError):
    """Raised when an empirical result is requested for simulator calibration."""


def _fail(message: str) -> NoReturn:
    raise EmpiricalSchemaError(message)


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        _fail(f"{field} must be a stable identifier")
    return value


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 512:
        _fail(f"{field} must be a non-empty string of at most 512 characters")
    return value


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        _fail(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{field} must be finite")
    return result


def _fraction(value: object, field: str, *, open_interval: bool = False) -> float:
    result = _finite(value, field)
    valid = 0.0 < result < 1.0 if open_interval else 0.0 <= result <= 1.0
    if not valid:
        interval = "(0, 1)" if open_interval else "[0, 1]"
        _fail(f"{field} must lie in {interval}")
    return result


def _timestamp(value: object, field: str) -> datetime:
    if not isinstance(value, datetime):
        _fail(f"{field} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        _fail(f"{field} must include an explicit timezone")
    return value.astimezone(UTC)


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail(f"{field} must be a lowercase SHA-256 digest")
    return value


def _fields(
    mapping: Mapping[str, object], expected: frozenset[str], context: str
) -> None:
    actual = frozenset(mapping)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        _fail(f"{context} missing fields: {', '.join(missing)}")
    if unknown:
        _fail(f"{context} has unknown fields: {', '.join(unknown)}")


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        _fail(f"{field} must be an object with string keys")
    return cast(Mapping[str, object], value)


def _array(value: object, field: str) -> list[object]:
    if not isinstance(value, list):
        _fail(f"{field} must be an array")
    return cast(list[object], value)


def _parse_timestamp(value: object, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        _fail(f"{field} must be an ISO 8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise EmpiricalSchemaError(f"{field} must be ISO 8601") from error
    return _timestamp(parsed, field)


def _optional_float(value: object, field: str) -> float | None:
    return None if value is None else _finite(value, field)


def _unique_ints(values: object, field: str) -> tuple[int, ...]:
    if not isinstance(values, tuple) or not values:
        _fail(f"{field} must be a non-empty immutable tuple")
    if not all(
        isinstance(value, int) and not isinstance(value, bool) for value in values
    ):
        _fail(f"{field} must contain only integers")
    if len(set(values)) != len(values):
        _fail(f"{field} must not contain duplicates")
    return values


def _unique_identifiers(values: object, field: str) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        _fail(f"{field} must be an immutable tuple")
    normalized = tuple(_identifier(value, f"{field} item") for value in values)
    if len(set(normalized)) != len(normalized):
        _fail(f"{field} must not contain duplicates")
    return normalized


class ObservationStatus(StrEnum):
    """Explicit state for every cell in the frozen unit-by-period universe."""

    OBSERVED = "observed"
    MISSING = "missing"
    EXITED = "exited"
    NOT_IN_SAMPLE = "not_in_sample"


class MissingDataPolicy(StrEnum):
    """Frozen handling rule; neither mode silently changes the analysis window."""

    ERROR = "error"
    COMPLETE_CASE = "complete_case"


class EstimateStatus(StrEnum):
    """Machine-readable status retained for every frozen specification."""

    ESTIMATED = "estimated"
    INSUFFICIENT_DATA = "insufficient_data"
    SINGULAR_DESIGN = "singular_design"


class DiagnosticStatus(StrEnum):
    """Outcome of a diagnostic with a threshold fixed before evaluation."""

    PASS = "pass"
    FAIL = "fail"
    INSUFFICIENT_DATA = "insufficient_data"


class FreezeMode(StrEnum):
    """How the immutable evaluation inputs were verified."""

    VERIFIED_TEMPORAL_SNAPSHOT = "verified_temporal_snapshot"
    SYNTHETIC_TEST_FIXTURE = "synthetic_test_fixture"


class DerivationEvidenceClass(StrEnum):
    """Machine-readable authority boundary for empirical panel derivation."""

    UNVERIFIED_SYNTHETIC_FIXTURE = "unverified_synthetic_fixture"
    SYNTHETIC_TABLE_REPLAY_ONLY = "synthetic_table_replay_only"


class ArtifactConsumer(StrEnum):
    """Destinations recognized by the empirical artifact boundary."""

    REPORTING = "reporting"
    REPRODUCIBILITY_AUDIT = "reproducibility_audit"
    SIMULATOR_CALIBRATION = "simulator_calibration"
    SIMULATOR_TUNING = "simulator_tuning"


class RandomizationDesign(StrEnum):
    """Frozen assignment mechanism supported by randomization inference."""

    UNIT_COMPLETE = "unit_complete"


class CausalReportingStatus(StrEnum):
    """Internal diagnostics only; never an authorization for a causal claim."""

    VALID = "passes_internal_causal_diagnostics"
    INVALID = "fails_internal_causal_diagnostics"


@dataclass(frozen=True, slots=True)
class PeriodTimeMapping:
    """Frozen mapping from a relative-period label to one UTC event time.

    When the config supplies these entries, elapsed UTC days rather than the
    integer labels are used as the pretrend time coordinate.
    """

    relative_period: int
    event_time: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.relative_period, int) or isinstance(
            self.relative_period, bool
        ):
            _fail("period mapping relative_period must be an integer")
        object.__setattr__(
            self,
            "event_time",
            _timestamp(self.event_time, "period mapping event_time"),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "event_time": self.event_time.isoformat(),
            "relative_period": self.relative_period,
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> Self:
        """Strictly reconstruct one frozen relative-period mapping."""
        _fields(
            mapping,
            frozenset({"event_time", "relative_period"}),
            "period time mapping",
        )
        period = mapping["relative_period"]
        if not isinstance(period, int) or isinstance(period, bool):
            _fail("period mapping relative_period must be an integer")
        return cls(
            relative_period=period,
            event_time=_parse_timestamp(mapping["event_time"], "period event_time"),
        )


@dataclass(frozen=True, slots=True)
class TreatmentDefinition:
    """Frozen common-timing treatment assignment and policy interval."""

    policy_id: str
    policy_start: datetime
    policy_end: datetime
    randomization_design: RandomizationDesign
    treated_unit_ids: tuple[str, ...]
    control_unit_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _identifier(self.policy_id, "policy_id")
        object.__setattr__(
            self, "policy_start", _timestamp(self.policy_start, "policy_start")
        )
        object.__setattr__(
            self, "policy_end", _timestamp(self.policy_end, "policy_end")
        )
        if self.policy_end <= self.policy_start:
            _fail("policy_end must be after policy_start")
        if not isinstance(self.randomization_design, RandomizationDesign):
            _fail("randomization_design must be explicitly configured")
        treated = _unique_identifiers(self.treated_unit_ids, "treated_unit_ids")
        control = _unique_identifiers(self.control_unit_ids, "control_unit_ids")
        if not treated or not control:
            _fail("treated and control cohorts must both be non-empty")
        if set(treated) & set(control):
            _fail("treated and control cohorts must be disjoint")

    @property
    def unit_ids(self) -> tuple[str, ...]:
        """Return the complete, deterministic frozen cohort."""
        return tuple(sorted((*self.treated_unit_ids, *self.control_unit_ids)))

    def is_treated(self, unit_id: str) -> bool:
        """Return the assignment fixed in this treatment definition."""
        if unit_id not in self.unit_ids:
            raise EmpiricalSchemaError(f"unit {unit_id!r} is outside the frozen cohort")
        return unit_id in self.treated_unit_ids

    def to_mapping(self) -> dict[str, object]:
        return {
            "control_unit_ids": list(self.control_unit_ids),
            "policy_end": self.policy_end.isoformat(),
            "policy_id": self.policy_id,
            "policy_start": self.policy_start.isoformat(),
            "randomization_design": self.randomization_design.value,
            "treated_unit_ids": list(self.treated_unit_ids),
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> Self:
        """Strictly load a frozen treatment definition."""
        _fields(
            mapping,
            frozenset(
                {
                    "policy_id",
                    "policy_start",
                    "policy_end",
                    "randomization_design",
                    "treated_unit_ids",
                    "control_unit_ids",
                }
            ),
            "treatment definition",
        )
        design = mapping["randomization_design"]
        if not isinstance(design, str):
            _fail("randomization_design must be a string")
        try:
            parsed_design = RandomizationDesign(design)
        except ValueError as error:
            raise EmpiricalSchemaError("randomization_design is unsupported") from error
        return cls(
            policy_id=_identifier(mapping["policy_id"], "policy_id"),
            policy_start=_parse_timestamp(mapping["policy_start"], "policy_start"),
            policy_end=_parse_timestamp(mapping["policy_end"], "policy_end"),
            randomization_design=parsed_design,
            treated_unit_ids=tuple(
                _identifier(item, "treated_unit_ids item")
                for item in _array(mapping["treated_unit_ids"], "treated_unit_ids")
            ),
            control_unit_ids=tuple(
                _identifier(item, "control_unit_ids item")
                for item in _array(mapping["control_unit_ids"], "control_unit_ids")
            ),
        )


@dataclass(frozen=True, slots=True)
class SubgroupDefinition:
    """A subgroup defined only by a frozen pre-treatment numeric characteristic."""

    subgroup_id: str
    characteristic_name: str
    lower_inclusive: float | None = None
    upper_exclusive: float | None = None

    def __post_init__(self) -> None:
        _identifier(self.subgroup_id, "subgroup_id")
        if self.subgroup_id == "all":
            _fail("subgroup_id 'all' is reserved for the full cohort")
        _identifier(self.characteristic_name, "characteristic_name")
        if self.lower_inclusive is not None:
            object.__setattr__(
                self,
                "lower_inclusive",
                _finite(self.lower_inclusive, "lower_inclusive"),
            )
        if self.upper_exclusive is not None:
            object.__setattr__(
                self,
                "upper_exclusive",
                _finite(self.upper_exclusive, "upper_exclusive"),
            )
        if (
            self.lower_inclusive is not None
            and self.upper_exclusive is not None
            and self.lower_inclusive >= self.upper_exclusive
        ):
            _fail("subgroup lower bound must be below its upper bound")

    def contains(self, value: float) -> bool:
        """Apply the pre-registered interval without data-dependent changes."""
        return (self.lower_inclusive is None or value >= self.lower_inclusive) and (
            self.upper_exclusive is None or value < self.upper_exclusive
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "characteristic_name": self.characteristic_name,
            "lower_inclusive": self.lower_inclusive,
            "subgroup_id": self.subgroup_id,
            "upper_exclusive": self.upper_exclusive,
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> Self:
        """Strictly load a pre-treatment subgroup definition."""
        _fields(
            mapping,
            frozenset(
                {
                    "subgroup_id",
                    "characteristic_name",
                    "lower_inclusive",
                    "upper_exclusive",
                }
            ),
            "subgroup definition",
        )
        return cls(
            subgroup_id=_identifier(mapping["subgroup_id"], "subgroup_id"),
            characteristic_name=_identifier(
                mapping["characteristic_name"], "characteristic_name"
            ),
            lower_inclusive=_optional_float(
                mapping["lower_inclusive"], "lower_inclusive"
            ),
            upper_exclusive=_optional_float(
                mapping["upper_exclusive"], "upper_exclusive"
            ),
        )


@dataclass(frozen=True, slots=True)
class EmpiricalSpecification:
    """One immutable member of the complete specification curve."""

    specification_id: str
    pre_periods: tuple[int, ...]
    post_periods: tuple[int, ...]
    event_periods: tuple[int, ...]
    reference_period: int
    max_absolute_lead_tolerance: float
    max_slope_gap_per_day: float
    covariates: tuple[str, ...] = ()
    quantiles: tuple[float, ...] = (0.25, 0.5, 0.75)
    subgroup_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.specification_id, "specification_id")
        pre = _unique_ints(self.pre_periods, "pre_periods")
        post = _unique_ints(self.post_periods, "post_periods")
        event = _unique_ints(self.event_periods, "event_periods")
        if any(period >= 0 for period in pre):
            _fail("pre_periods must be strictly before relative period zero")
        if any(period < 0 for period in post):
            _fail("post_periods must be at or after relative period zero")
        if not isinstance(self.reference_period, int) or isinstance(
            self.reference_period, bool
        ):
            _fail("reference_period must be an integer")
        if self.reference_period >= 0 or self.reference_period not in event:
            _fail("reference_period must be a negative member of event_periods")
        if not any(period != self.reference_period for period in event):
            _fail("event_periods must contain a non-reference period")
        covariates = _unique_identifiers(self.covariates, "covariates")
        subgroups = _unique_identifiers(self.subgroup_ids, "subgroup_ids")
        if not isinstance(self.quantiles, tuple) or not self.quantiles:
            _fail("quantiles must be a non-empty immutable tuple")
        quantiles = tuple(
            _fraction(value, "quantile", open_interval=True) for value in self.quantiles
        )
        if len(set(quantiles)) != len(quantiles):
            _fail("quantiles must be unique")
        object.__setattr__(self, "quantiles", quantiles)
        if covariates != self.covariates or subgroups != self.subgroup_ids:
            _fail("covariates and subgroup_ids must contain stable identifiers")
        for field in (
            "max_absolute_lead_tolerance",
            "max_slope_gap_per_day",
        ):
            tolerance = _finite(getattr(self, field), field)
            if tolerance < 0:
                _fail(f"{field} must not be negative")
            object.__setattr__(self, field, tolerance)

    @property
    def required_periods(self) -> tuple[int, ...]:
        """Return every period this specification commits to inspect."""
        return tuple(
            sorted(
                set(self.pre_periods) | set(self.post_periods) | set(self.event_periods)
            )
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "covariates": list(self.covariates),
            "event_periods": list(self.event_periods),
            "max_absolute_lead_tolerance": self.max_absolute_lead_tolerance,
            "max_slope_gap_per_day": self.max_slope_gap_per_day,
            "post_periods": list(self.post_periods),
            "pre_periods": list(self.pre_periods),
            "quantiles": list(self.quantiles),
            "reference_period": self.reference_period,
            "specification_id": self.specification_id,
            "subgroup_ids": list(self.subgroup_ids),
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> Self:
        """Strictly load one pre-registered specification."""
        _fields(
            mapping,
            frozenset(
                {
                    "specification_id",
                    "pre_periods",
                    "post_periods",
                    "event_periods",
                    "reference_period",
                    "covariates",
                    "quantiles",
                    "subgroup_ids",
                    "max_absolute_lead_tolerance",
                    "max_slope_gap_per_day",
                }
            ),
            "empirical specification",
        )

        def integer_tuple(field: str) -> tuple[int, ...]:
            values = _array(mapping[field], field)
            if not all(
                isinstance(item, int) and not isinstance(item, bool) for item in values
            ):
                _fail(f"{field} must contain integers")
            return tuple(cast(list[int], values))

        def identifier_tuple(field: str) -> tuple[str, ...]:
            return tuple(
                _identifier(item, f"{field} item")
                for item in _array(mapping[field], field)
            )

        reference = mapping["reference_period"]
        if not isinstance(reference, int) or isinstance(reference, bool):
            _fail("reference_period must be an integer")
        return cls(
            specification_id=_identifier(
                mapping["specification_id"], "specification_id"
            ),
            pre_periods=integer_tuple("pre_periods"),
            post_periods=integer_tuple("post_periods"),
            event_periods=integer_tuple("event_periods"),
            reference_period=reference,
            max_absolute_lead_tolerance=_finite(
                mapping["max_absolute_lead_tolerance"],
                "max_absolute_lead_tolerance",
            ),
            max_slope_gap_per_day=_finite(
                mapping["max_slope_gap_per_day"], "max_slope_gap_per_day"
            ),
            covariates=identifier_tuple("covariates"),
            quantiles=tuple(
                _finite(item, "quantile")
                for item in _array(mapping["quantiles"], "quantiles")
            ),
            subgroup_ids=identifier_tuple("subgroup_ids"),
        )


@dataclass(frozen=True, slots=True)
class EmpiricalEvaluationConfig:
    """Complete frozen configuration; it has no best-specification selector."""

    schema_version: int
    study_id: str
    treatment: TreatmentDefinition
    specifications: tuple[EmpiricalSpecification, ...]
    subgroups: tuple[SubgroupDefinition, ...]
    bootstrap_repetitions: int
    randomization_repetitions: int
    confidence_level: float
    random_seed: str
    missing_data_policy: MissingDataPolicy
    max_missing_fraction: float
    max_attrition_fraction: float
    max_composition_shift: float
    synthetic_test_only: bool = False
    period_time_mapping: tuple[PeriodTimeMapping, ...] = ()
    minimum_valid_bootstrap_draws: int = 1
    minimum_valid_randomization_draws: int = 1
    minimum_total_clusters: int = 4
    minimum_treated_clusters: int = 2
    minimum_control_clusters: int = 2
    maximum_invalid_bootstrap_fraction: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 4:
            _fail("empirical config schema_version must be 4")
        _identifier(self.study_id, "study_id")
        if not isinstance(self.treatment, TreatmentDefinition):
            _fail("treatment must be a TreatmentDefinition")
        if not isinstance(self.specifications, tuple) or not self.specifications:
            _fail("specifications must be a non-empty immutable tuple")
        if not all(
            isinstance(specification, EmpiricalSpecification)
            for specification in self.specifications
        ):
            _fail("specifications contain an invalid value")
        spec_ids = tuple(item.specification_id for item in self.specifications)
        if len(set(spec_ids)) != len(spec_ids):
            _fail("specification IDs must be unique")
        if not isinstance(self.subgroups, tuple) or not all(
            isinstance(subgroup, SubgroupDefinition) for subgroup in self.subgroups
        ):
            _fail("subgroups must be an immutable tuple of SubgroupDefinition")
        subgroup_ids = tuple(item.subgroup_id for item in self.subgroups)
        if len(set(subgroup_ids)) != len(subgroup_ids):
            _fail("subgroup IDs must be unique")
        unknown = sorted(
            {
                subgroup_id
                for spec in self.specifications
                for subgroup_id in spec.subgroup_ids
                if subgroup_id not in subgroup_ids
            }
        )
        if unknown:
            _fail(f"specifications reference unknown subgroups: {', '.join(unknown)}")
        for field in ("bootstrap_repetitions", "randomization_repetitions"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                _fail(f"{field} must be a positive integer")
        for minimum_field, repetition_field in (
            ("minimum_valid_bootstrap_draws", "bootstrap_repetitions"),
            ("minimum_valid_randomization_draws", "randomization_repetitions"),
        ):
            minimum = getattr(self, minimum_field)
            repetitions = getattr(self, repetition_field)
            if (
                isinstance(minimum, bool)
                or not isinstance(minimum, int)
                or minimum < 1
                or minimum > repetitions
            ):
                _fail(
                    f"{minimum_field} must be a positive integer no greater than "
                    f"{repetition_field}"
                )
        for field in (
            "minimum_total_clusters",
            "minimum_treated_clusters",
            "minimum_control_clusters",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                _fail(f"{field} must be a positive integer")
        if self.minimum_treated_clusters < 2 or self.minimum_control_clusters < 2:
            _fail("cluster-bootstrap inference requires at least two clusters per arm")
        if self.minimum_total_clusters < (
            self.minimum_treated_clusters + self.minimum_control_clusters
        ):
            _fail(
                "minimum_total_clusters must be at least the sum of the frozen "
                "arm-specific cluster minima"
            )
        object.__setattr__(
            self,
            "maximum_invalid_bootstrap_fraction",
            _fraction(
                self.maximum_invalid_bootstrap_fraction,
                "maximum_invalid_bootstrap_fraction",
            ),
        )
        object.__setattr__(
            self,
            "confidence_level",
            _fraction(self.confidence_level, "confidence_level", open_interval=True),
        )
        _text(self.random_seed, "random_seed")
        if not isinstance(self.missing_data_policy, MissingDataPolicy):
            _fail("missing_data_policy must be a MissingDataPolicy")
        for field in ("max_missing_fraction", "max_attrition_fraction"):
            object.__setattr__(self, field, _fraction(getattr(self, field), field))
        shift = _finite(self.max_composition_shift, "max_composition_shift")
        if shift < 0:
            _fail("max_composition_shift must not be negative")
        object.__setattr__(self, "max_composition_shift", shift)
        if not isinstance(self.synthetic_test_only, bool):
            _fail("synthetic_test_only must be a boolean")
        if not isinstance(self.period_time_mapping, tuple) or not all(
            isinstance(item, PeriodTimeMapping) for item in self.period_time_mapping
        ):
            _fail("period_time_mapping must be an immutable PeriodTimeMapping tuple")
        mapping = tuple(
            sorted(self.period_time_mapping, key=lambda item: item.relative_period)
        )
        if mapping:
            periods = tuple(item.relative_period for item in mapping)
            if len(set(periods)) != len(periods):
                _fail("period_time_mapping relative periods must be unique")
            if periods != self.required_periods:
                _fail(
                    "period_time_mapping must cover the frozen specification-period "
                    "union exactly"
                )
            times = tuple(item.event_time for item in mapping)
            if any(
                current <= previous
                for previous, current in zip(times, times[1:], strict=False)
            ):
                _fail("period_time_mapping event times must be strictly increasing")
            policy_start = self.treatment.policy_start
            for item in mapping:
                if item.relative_period < 0 and item.event_time >= policy_start:
                    _fail("negative mapped periods must occur before policy_start")
                if item.relative_period >= 0 and item.event_time < policy_start:
                    _fail(
                        "non-negative mapped periods must not occur before policy_start"
                    )
        object.__setattr__(self, "period_time_mapping", mapping)
        if not self.synthetic_test_only:
            if not mapping:
                _fail(
                    "non-synthetic empirical evaluation requires a frozen "
                    "period_time_mapping; integer period labels are synthetic-only"
                )
            formal_draw_floor = 999
            for repetition_field, minimum_field in (
                ("bootstrap_repetitions", "minimum_valid_bootstrap_draws"),
                ("randomization_repetitions", "minimum_valid_randomization_draws"),
            ):
                if (
                    getattr(self, repetition_field) < formal_draw_floor
                    or getattr(self, minimum_field) < formal_draw_floor
                ):
                    _fail(
                        "non-synthetic empirical evaluation requires at least "
                        f"{formal_draw_floor} requested and valid draws for "
                        f"{repetition_field}"
                    )

    @property
    def required_periods(self) -> tuple[int, ...]:
        """Return the frozen union of periods across every specification."""
        return tuple(
            sorted(
                {
                    period
                    for specification in self.specifications
                    for period in specification.required_periods
                }
            )
        )

    def event_time_for_period(self, relative_period: int) -> datetime | None:
        """Return the frozen calendar event time, or None in integer-label mode."""
        if relative_period not in self.required_periods:
            raise EmpiricalSchemaError(
                f"relative period {relative_period} is outside the frozen config"
            )
        if not self.period_time_mapping:
            return None
        return next(
            item.event_time
            for item in self.period_time_mapping
            if item.relative_period == relative_period
        )

    def pretrend_coordinate(self, relative_period: int) -> float:
        """Return elapsed UTC days, with one period as one synthetic test day."""
        event_time = self.event_time_for_period(relative_period)
        if event_time is None:
            return float(relative_period)
        return (event_time - self.treatment.policy_start) / timedelta(days=1)

    @property
    def causal_reporting_criteria_sha256(self) -> str:
        """Commit to support diagnostics without using target-effect magnitude."""
        criteria = {
            "all_frozen_estimates_must_be_estimable": True,
            "all_randomization_draws_must_be_defined": True,
            "all_sample_audit_flags_invalidate": True,
            "complete_case_post_policy_observation_exclusion_invalidates": True,
            "bootstrap_cluster_support_must_pass": True,
            "bootstrap_invalid_fraction_must_pass": True,
            "criteria_schema_version": "policytwin.empirical-causal-reporting.v4",
            "max_attrition_fraction": self.max_attrition_fraction,
            "max_composition_shift": self.max_composition_shift,
            "max_missing_fraction": self.max_missing_fraction,
            "missing_data_policy": self.missing_data_policy.value,
            "minimum_valid_bootstrap_draws": self.minimum_valid_bootstrap_draws,
            "minimum_valid_randomization_draws": (
                self.minimum_valid_randomization_draws
            ),
            "period_time_mapping": [
                item.to_mapping() for item in self.period_time_mapping
            ],
            "minimum_total_clusters": self.minimum_total_clusters,
            "minimum_treated_clusters": self.minimum_treated_clusters,
            "minimum_control_clusters": self.minimum_control_clusters,
            "maximum_invalid_bootstrap_fraction": (
                self.maximum_invalid_bootstrap_fraction
            ),
            "pre_policy_leads_must_pass_frozen_tolerance": True,
            "registered_covariate_exclusion_invalidates": True,
            "pretrend_thresholds": [
                {
                    "max_absolute_lead_tolerance": (item.max_absolute_lead_tolerance),
                    "max_slope_gap_per_day": item.max_slope_gap_per_day,
                    "specification_id": item.specification_id,
                }
                for item in self.specifications
            ],
            "pretrend_slope_gap_must_pass_frozen_tolerance": True,
        }
        return sha256_bytes(canonical_json_bytes(criteria))

    @property
    def sha256(self) -> str:
        """Return the semantic commitment to every estimand and threshold."""
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def to_mapping(self) -> dict[str, object]:
        return {
            "bootstrap_repetitions": self.bootstrap_repetitions,
            "confidence_level": self.confidence_level,
            "max_attrition_fraction": self.max_attrition_fraction,
            "max_composition_shift": self.max_composition_shift,
            "max_missing_fraction": self.max_missing_fraction,
            "maximum_invalid_bootstrap_fraction": (
                self.maximum_invalid_bootstrap_fraction
            ),
            "minimum_control_clusters": self.minimum_control_clusters,
            "minimum_total_clusters": self.minimum_total_clusters,
            "minimum_treated_clusters": self.minimum_treated_clusters,
            "minimum_valid_bootstrap_draws": self.minimum_valid_bootstrap_draws,
            "minimum_valid_randomization_draws": (
                self.minimum_valid_randomization_draws
            ),
            "missing_data_policy": self.missing_data_policy.value,
            "period_time_mapping": [
                item.to_mapping() for item in self.period_time_mapping
            ],
            "random_seed": self.random_seed,
            "randomization_repetitions": self.randomization_repetitions,
            "schema_version": self.schema_version,
            "specifications": [item.to_mapping() for item in self.specifications],
            "study_id": self.study_id,
            "subgroups": [item.to_mapping() for item in self.subgroups],
            "synthetic_test_only": self.synthetic_test_only,
            "treatment": self.treatment.to_mapping(),
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> Self:
        """Strictly load the full config, rejecting auto-selection fields."""
        _fields(
            mapping,
            frozenset(
                {
                    "schema_version",
                    "study_id",
                    "treatment",
                    "specifications",
                    "subgroups",
                    "bootstrap_repetitions",
                    "randomization_repetitions",
                    "confidence_level",
                    "random_seed",
                    "missing_data_policy",
                    "max_missing_fraction",
                    "max_attrition_fraction",
                    "max_composition_shift",
                    "synthetic_test_only",
                    "period_time_mapping",
                    "minimum_valid_bootstrap_draws",
                    "minimum_valid_randomization_draws",
                    "minimum_total_clusters",
                    "minimum_treated_clusters",
                    "minimum_control_clusters",
                    "maximum_invalid_bootstrap_fraction",
                }
            ),
            "empirical evaluation config",
        )
        schema_version = mapping["schema_version"]
        bootstrap = mapping["bootstrap_repetitions"]
        randomization = mapping["randomization_repetitions"]
        minimum_bootstrap = mapping["minimum_valid_bootstrap_draws"]
        minimum_randomization = mapping["minimum_valid_randomization_draws"]
        minimum_total_clusters = mapping["minimum_total_clusters"]
        minimum_treated_clusters = mapping["minimum_treated_clusters"]
        minimum_control_clusters = mapping["minimum_control_clusters"]
        if not isinstance(schema_version, int) or isinstance(schema_version, bool):
            _fail("schema_version must be an integer")
        if not isinstance(bootstrap, int) or isinstance(bootstrap, bool):
            _fail("bootstrap_repetitions must be an integer")
        if not isinstance(randomization, int) or isinstance(randomization, bool):
            _fail("randomization_repetitions must be an integer")
        if not isinstance(minimum_bootstrap, int) or isinstance(
            minimum_bootstrap, bool
        ):
            _fail("minimum_valid_bootstrap_draws must be an integer")
        if not isinstance(minimum_randomization, int) or isinstance(
            minimum_randomization, bool
        ):
            _fail("minimum_valid_randomization_draws must be an integer")
        for field, value in (
            ("minimum_total_clusters", minimum_total_clusters),
            ("minimum_treated_clusters", minimum_treated_clusters),
            ("minimum_control_clusters", minimum_control_clusters),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                _fail(f"{field} must be an integer")
        missing_policy = mapping["missing_data_policy"]
        if not isinstance(missing_policy, str):
            _fail("missing_data_policy must be a string")
        try:
            parsed_missing_policy = MissingDataPolicy(missing_policy)
        except ValueError as error:
            raise EmpiricalSchemaError("missing_data_policy is invalid") from error
        synthetic = mapping["synthetic_test_only"]
        if not isinstance(synthetic, bool):
            _fail("synthetic_test_only must be a boolean")
        return cls(
            schema_version=schema_version,
            study_id=_identifier(mapping["study_id"], "study_id"),
            treatment=TreatmentDefinition.from_mapping(
                _mapping(mapping["treatment"], "treatment")
            ),
            specifications=tuple(
                EmpiricalSpecification.from_mapping(
                    _mapping(item, "specifications item")
                )
                for item in _array(mapping["specifications"], "specifications")
            ),
            subgroups=tuple(
                SubgroupDefinition.from_mapping(_mapping(item, "subgroups item"))
                for item in _array(mapping["subgroups"], "subgroups")
            ),
            bootstrap_repetitions=bootstrap,
            randomization_repetitions=randomization,
            confidence_level=_finite(mapping["confidence_level"], "confidence_level"),
            random_seed=_text(mapping["random_seed"], "random_seed"),
            missing_data_policy=parsed_missing_policy,
            max_missing_fraction=_finite(
                mapping["max_missing_fraction"], "max_missing_fraction"
            ),
            max_attrition_fraction=_finite(
                mapping["max_attrition_fraction"], "max_attrition_fraction"
            ),
            max_composition_shift=_finite(
                mapping["max_composition_shift"], "max_composition_shift"
            ),
            synthetic_test_only=synthetic,
            period_time_mapping=tuple(
                PeriodTimeMapping.from_mapping(
                    _mapping(item, "period_time_mapping item")
                )
                for item in _array(
                    mapping["period_time_mapping"], "period_time_mapping"
                )
            ),
            minimum_valid_bootstrap_draws=minimum_bootstrap,
            minimum_valid_randomization_draws=minimum_randomization,
            minimum_total_clusters=cast(int, minimum_total_clusters),
            minimum_treated_clusters=cast(int, minimum_treated_clusters),
            minimum_control_clusters=cast(int, minimum_control_clusters),
            maximum_invalid_bootstrap_fraction=_finite(
                mapping["maximum_invalid_bootstrap_fraction"],
                "maximum_invalid_bootstrap_fraction",
            ),
        )


@dataclass(frozen=True, slots=True)
class PreTreatmentCharacteristic:
    """A numeric characteristic with record-level temporal provenance."""

    unit_id: str
    name: str
    value: float
    measured_relative_period: int
    record: TemporalRecord
    derivation_receipt: TransformationReceipt | None = None

    def __post_init__(self) -> None:
        _identifier(self.unit_id, "characteristic unit_id")
        _identifier(self.name, "characteristic name")
        object.__setattr__(self, "value", _finite(self.value, "characteristic value"))
        if (
            not isinstance(self.measured_relative_period, int)
            or isinstance(self.measured_relative_period, bool)
            or self.measured_relative_period >= 0
        ):
            _fail("characteristics must be measured strictly before treatment")
        if not isinstance(self.record, TemporalRecord):
            _fail("characteristic record must be a TemporalRecord")
        if self.record.content_hash != self.payload_sha256:
            _fail("characteristic record content_hash does not match its payload")
        if self.derivation_receipt is not None:
            if not isinstance(self.derivation_receipt, TransformationReceipt):
                _fail("characteristic derivation_receipt has an invalid schema")
            if self.derivation_receipt.output_mapping != self.payload_mapping:
                _fail("characteristic derivation receipt output changed")
            if self.derivation_receipt.output_record != self.record:
                _fail("characteristic record differs from its derivation receipt")

    @property
    def payload_mapping(self) -> dict[str, object]:
        """Return the complete non-provenance payload committed by the record."""
        return {
            "measured_relative_period": self.measured_relative_period,
            "name": self.name,
            "unit_id": self.unit_id,
            "value": self.value,
        }

    @property
    def payload_sha256(self) -> str:
        """Return the commitment to the characteristic value, excluding metadata."""
        return sha256_bytes(canonical_json_bytes(self.payload_mapping))

    def to_mapping(self) -> dict[str, object]:
        mapping: dict[str, object] = {
            "measured_relative_period": self.measured_relative_period,
            "name": self.name,
            "record": self.record.to_mapping(),
            "unit_id": self.unit_id,
            "value": self.value,
        }
        if self.derivation_receipt is not None:
            mapping["derivation_receipt"] = self.derivation_receipt.to_mapping()
        return mapping

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> Self:
        """Strictly reconstruct one record-backed pre-treatment value."""
        expected = {
            "measured_relative_period",
            "name",
            "record",
            "unit_id",
            "value",
        }
        if "derivation_receipt" in mapping:
            expected.add("derivation_receipt")
        _fields(mapping, frozenset(expected), "pre-treatment characteristic")
        measured_period = mapping["measured_relative_period"]
        if not isinstance(measured_period, int) or isinstance(measured_period, bool):
            _fail("measured_relative_period must be an integer")
        try:
            receipt = (
                None
                if "derivation_receipt" not in mapping
                else TransformationReceipt.from_mapping(
                    _mapping(mapping["derivation_receipt"], "derivation_receipt")
                )
            )
            return cls(
                unit_id=_identifier(mapping["unit_id"], "characteristic unit_id"),
                name=_identifier(mapping["name"], "characteristic name"),
                value=_finite(mapping["value"], "characteristic value"),
                measured_relative_period=measured_period,
                record=TemporalRecord.from_mapping(
                    _mapping(mapping["record"], "characteristic record")
                ),
                derivation_receipt=receipt,
            )
        except TransformationReceiptError as error:
            raise EmpiricalSchemaError(
                f"invalid characteristic derivation receipt: {error}"
            ) from error


@dataclass(frozen=True, slots=True)
class PanelUnit:
    """One member of the frozen cohort and its cluster assignment."""

    unit_id: str
    cluster_id: str
    characteristics: tuple[PreTreatmentCharacteristic, ...]

    def __post_init__(self) -> None:
        _identifier(self.unit_id, "unit_id")
        _identifier(self.cluster_id, "cluster_id")
        if not isinstance(self.characteristics, tuple) or not all(
            isinstance(item, PreTreatmentCharacteristic)
            for item in self.characteristics
        ):
            _fail("characteristics must be an immutable tuple")
        if any(item.unit_id != self.unit_id for item in self.characteristics):
            _fail("characteristic unit_id does not match its PanelUnit")
        names = tuple(item.name for item in self.characteristics)
        if len(set(names)) != len(names):
            _fail("characteristic names must be unique within a unit")

    def characteristic(self, name: str) -> float | None:
        """Return a frozen pre-treatment value, or None when explicitly unavailable."""
        for item in self.characteristics:
            if item.name == name:
                return item.value
        return None

    def to_mapping(self) -> dict[str, object]:
        return {
            "characteristics": [item.to_mapping() for item in self.characteristics],
            "cluster_id": self.cluster_id,
            "unit_id": self.unit_id,
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> Self:
        """Strictly reconstruct one member of the frozen cohort."""
        _fields(
            mapping,
            frozenset({"characteristics", "cluster_id", "unit_id"}),
            "panel unit",
        )
        return cls(
            unit_id=_identifier(mapping["unit_id"], "unit_id"),
            cluster_id=_identifier(mapping["cluster_id"], "cluster_id"),
            characteristics=tuple(
                PreTreatmentCharacteristic.from_mapping(
                    _mapping(item, "characteristics item")
                )
                for item in _array(mapping["characteristics"], "characteristics")
            ),
        )


@dataclass(frozen=True, slots=True)
class PostTreatmentGroup:
    """A post-treatment label retained only for descriptive composition audit."""

    name: str
    value: str

    def __post_init__(self) -> None:
        _identifier(self.name, "post-treatment group name")
        _text(self.value, "post-treatment group value")

    def to_mapping(self) -> dict[str, object]:
        return {"name": self.name, "value": self.value}

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> Self:
        """Strictly reconstruct one descriptive post-treatment label."""
        _fields(mapping, frozenset({"name", "value"}), "post-treatment group")
        return cls(
            name=_identifier(mapping["name"], "post-treatment group name"),
            value=_text(mapping["value"], "post-treatment group value"),
        )


@dataclass(frozen=True, slots=True)
class PanelObservation:
    """One explicit cell of the frozen unit-by-period observation universe."""

    unit_id: str
    relative_period: int
    status: ObservationStatus
    outcome: float | None
    record: TemporalRecord
    post_treatment_group: PostTreatmentGroup | None = None
    derivation_receipt: TransformationReceipt | None = None

    def __post_init__(self) -> None:
        _identifier(self.unit_id, "observation unit_id")
        if not isinstance(self.relative_period, int) or isinstance(
            self.relative_period, bool
        ):
            _fail("relative_period must be an integer")
        if not isinstance(self.status, ObservationStatus):
            _fail("status must be an ObservationStatus")
        if self.status is ObservationStatus.OBSERVED:
            if self.outcome is None:
                _fail("observed cells require an outcome")
            object.__setattr__(self, "outcome", _finite(self.outcome, "outcome"))
        elif self.outcome is not None:
            _fail("non-observed cells must not contain an outcome")
        if not isinstance(self.record, TemporalRecord):
            _fail("observation record must be a TemporalRecord")
        if self.post_treatment_group is not None and not isinstance(
            self.post_treatment_group, PostTreatmentGroup
        ):
            _fail("post_treatment_group has an invalid schema")
        if self.relative_period < 0 and self.post_treatment_group is not None:
            _fail("post-treatment group labels are forbidden before treatment")
        if self.record.content_hash != self.payload_sha256:
            _fail("observation record content_hash does not match its payload")
        if self.derivation_receipt is not None:
            if not isinstance(self.derivation_receipt, TransformationReceipt):
                _fail("observation derivation_receipt has an invalid schema")
            if self.derivation_receipt.output_mapping != self.payload_mapping:
                _fail("observation derivation receipt output changed")
            if self.derivation_receipt.output_record != self.record:
                _fail("observation record differs from its derivation receipt")

    @property
    def payload_mapping(self) -> dict[str, object]:
        """Return the complete non-provenance payload committed by the record."""
        return {
            "outcome": self.outcome,
            "post_treatment_group": (
                self.post_treatment_group.to_mapping()
                if self.post_treatment_group is not None
                else None
            ),
            "relative_period": self.relative_period,
            "status": self.status.value,
            "unit_id": self.unit_id,
        }

    @property
    def payload_sha256(self) -> str:
        """Return the commitment to this cell, excluding temporal metadata."""
        return sha256_bytes(canonical_json_bytes(self.payload_mapping))

    def to_mapping(self) -> dict[str, object]:
        mapping: dict[str, object] = {
            "outcome": self.outcome,
            "post_treatment_group": (
                self.post_treatment_group.to_mapping()
                if self.post_treatment_group is not None
                else None
            ),
            "record": self.record.to_mapping(),
            "relative_period": self.relative_period,
            "status": self.status.value,
            "unit_id": self.unit_id,
        }
        if self.derivation_receipt is not None:
            mapping["derivation_receipt"] = self.derivation_receipt.to_mapping()
        return mapping

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> Self:
        """Strictly reconstruct one explicit unit-period panel cell."""
        expected = {
            "outcome",
            "post_treatment_group",
            "record",
            "relative_period",
            "status",
            "unit_id",
        }
        if "derivation_receipt" in mapping:
            expected.add("derivation_receipt")
        _fields(mapping, frozenset(expected), "panel observation")
        relative_period = mapping["relative_period"]
        if not isinstance(relative_period, int) or isinstance(relative_period, bool):
            _fail("relative_period must be an integer")
        status = mapping["status"]
        if not isinstance(status, str):
            _fail("observation status must be a string")
        try:
            parsed_status = ObservationStatus(status)
        except ValueError as error:
            raise EmpiricalSchemaError("observation status is invalid") from error
        group_mapping = mapping["post_treatment_group"]
        try:
            receipt = (
                None
                if "derivation_receipt" not in mapping
                else TransformationReceipt.from_mapping(
                    _mapping(mapping["derivation_receipt"], "derivation_receipt")
                )
            )
            return cls(
                unit_id=_identifier(mapping["unit_id"], "observation unit_id"),
                relative_period=relative_period,
                status=parsed_status,
                outcome=_optional_float(mapping["outcome"], "outcome"),
                record=TemporalRecord.from_mapping(
                    _mapping(mapping["record"], "observation record")
                ),
                post_treatment_group=(
                    None
                    if group_mapping is None
                    else PostTreatmentGroup.from_mapping(
                        _mapping(group_mapping, "post_treatment_group")
                    )
                ),
                derivation_receipt=receipt,
            )
        except TransformationReceiptError as error:
            raise EmpiricalSchemaError(
                f"invalid observation derivation receipt: {error}"
            ) from error


@dataclass(frozen=True, slots=True)
class FrozenPanel:
    """Frozen expected cohort, period grid, and explicit observation cells."""

    schema_version: int
    panel_id: str
    units: tuple[PanelUnit, ...]
    expected_periods: tuple[int, ...]
    observations: tuple[PanelObservation, ...]
    synthetic_only: bool

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version not in {1, 2}:
            _fail("panel schema_version must be 1 or 2")
        _identifier(self.panel_id, "panel_id")
        if (
            not isinstance(self.units, tuple)
            or not self.units
            or not all(isinstance(unit, PanelUnit) for unit in self.units)
        ):
            _fail("units must be a non-empty immutable tuple of PanelUnit")
        unit_ids = tuple(unit.unit_id for unit in self.units)
        if len(set(unit_ids)) != len(unit_ids):
            _fail("panel unit IDs must be unique")
        periods = _unique_ints(self.expected_periods, "expected_periods")
        if periods != tuple(sorted(periods)):
            _fail("expected_periods must be in increasing order")
        if not isinstance(self.observations, tuple) or not all(
            isinstance(item, PanelObservation) for item in self.observations
        ):
            _fail("observations must be an immutable tuple of PanelObservation")
        actual = tuple(
            (item.unit_id, item.relative_period) for item in self.observations
        )
        if len(set(actual)) != len(actual):
            _fail("panel contains duplicate unit-period cells")
        expected = {(unit_id, period) for unit_id in unit_ids for period in periods}
        actual_set = set(actual)
        if actual_set != expected:
            missing = sorted(expected - actual_set)
            extra = sorted(actual_set - expected)
            _fail(
                "panel must explicitly represent the complete frozen unit-period "
                f"universe; missing={missing[:3]!r}, extra={extra[:3]!r}"
            )
        record_ids = tuple(item.record.record_id for item in self.observations)
        record_ids += tuple(
            characteristic.record.record_id
            for unit in self.units
            for characteristic in unit.characteristics
        )
        if len(set(record_ids)) != len(record_ids):
            _fail("panel temporal record IDs must be globally unique")
        if not isinstance(self.synthetic_only, bool):
            _fail("synthetic_only must be a boolean")
        receipts = self.derivation_receipts
        if self.schema_version == 1:
            if receipts:
                _fail("panel schema_version 1 forbids derivation receipts")
        else:
            expected_receipt_count = len(self.observations) + sum(
                len(unit.characteristics) for unit in self.units
            )
            if len(receipts) != expected_receipt_count:
                _fail(
                    "panel schema_version 2 requires one derivation receipt for "
                    "every observation and characteristic"
                )
            if not self.synthetic_only:
                _fail(
                    "panel schema_version 2 receipts currently support only "
                    "synthetic table replay"
                )
            if any(
                observation.post_treatment_group is not None
                for observation in self.observations
            ):
                _fail(
                    "panel schema_version 2 does not yet support derived "
                    "post-treatment group labels"
                )
            receipt_ids = tuple(receipt.sha256 for receipt in receipts)
            if len(set(receipt_ids)) != len(receipt_ids):
                _fail("panel derivation receipts must be unique")
            contracts = {
                receipt.plan.construction_contract_sha256 for receipt in receipts
            }
            if len(contracts) != 1:
                _fail("panel derivation receipts must share one construction contract")

    @property
    def derivation_receipts(self) -> tuple[TransformationReceipt, ...]:
        """Return all cell receipts in deterministic semantic-key order."""
        observation_receipts = tuple(
            observation.derivation_receipt
            for observation in sorted(
                self.observations,
                key=lambda item: (item.unit_id, item.relative_period),
            )
            if observation.derivation_receipt is not None
        )
        characteristic_receipts = tuple(
            characteristic.derivation_receipt
            for unit in sorted(self.units, key=lambda item: item.unit_id)
            for characteristic in sorted(
                unit.characteristics, key=lambda item: item.name
            )
            if characteristic.derivation_receipt is not None
        )
        return (*observation_receipts, *characteristic_receipts)

    @property
    def derivation_registry_sha256(self) -> str:
        """Commit to every receipt and its exact empirical cell identity."""
        entries: list[dict[str, object]] = []
        for observation in sorted(
            self.observations,
            key=lambda item: (item.unit_id, item.relative_period),
        ):
            if observation.derivation_receipt is not None:
                entries.append(
                    {
                        "cell_kind": "observation",
                        "receipt_sha256": observation.derivation_receipt.sha256,
                        "relative_period": observation.relative_period,
                        "unit_id": observation.unit_id,
                    }
                )
        for unit in sorted(self.units, key=lambda item: item.unit_id):
            for characteristic in sorted(
                unit.characteristics, key=lambda item: item.name
            ):
                if characteristic.derivation_receipt is not None:
                    entries.append(
                        {
                            "cell_kind": "characteristic",
                            "name": characteristic.name,
                            "receipt_sha256": characteristic.derivation_receipt.sha256,
                            "unit_id": characteristic.unit_id,
                        }
                    )
        return sha256_bytes(
            canonical_json_bytes(
                {
                    "entries": entries,
                    "panel_id": self.panel_id,
                    "schema_version": self.schema_version,
                }
            )
        )

    @property
    def sha256(self) -> str:
        """Return the order-independent semantic panel commitment."""
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def observation(self, unit_id: str, period: int) -> PanelObservation:
        """Return an explicitly represented cell from the frozen universe."""
        for item in self.observations:
            if item.unit_id == unit_id and item.relative_period == period:
                return item
        raise EmpiricalSchemaError("unit-period cell is outside the frozen panel")

    def unit(self, unit_id: str) -> PanelUnit:
        """Return one frozen cohort member."""
        for unit in self.units:
            if unit.unit_id == unit_id:
                return unit
        raise EmpiricalSchemaError(f"unit {unit_id!r} is outside the frozen panel")

    def to_mapping(self) -> dict[str, object]:
        return {
            "expected_periods": list(self.expected_periods),
            "observations": [
                item.to_mapping()
                for item in sorted(
                    self.observations,
                    key=lambda value: (value.unit_id, value.relative_period),
                )
            ],
            "panel_id": self.panel_id,
            "schema_version": self.schema_version,
            "synthetic_only": self.synthetic_only,
            "units": [
                item.to_mapping()
                for item in sorted(self.units, key=lambda value: value.unit_id)
            ],
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> Self:
        """Strictly parse the complete panel from frozen snapshot bytes."""
        _fields(
            mapping,
            frozenset(
                {
                    "expected_periods",
                    "observations",
                    "panel_id",
                    "schema_version",
                    "synthetic_only",
                    "units",
                }
            ),
            "frozen panel",
        )
        schema_version = mapping["schema_version"]
        if not isinstance(schema_version, int) or isinstance(schema_version, bool):
            _fail("panel schema_version must be an integer")
        synthetic_only = mapping["synthetic_only"]
        if not isinstance(synthetic_only, bool):
            _fail("synthetic_only must be a boolean")
        expected_periods: list[int] = []
        for item in _array(mapping["expected_periods"], "expected_periods"):
            if not isinstance(item, int) or isinstance(item, bool):
                _fail("expected_periods must contain only integers")
            expected_periods.append(item)
        return cls(
            schema_version=schema_version,
            panel_id=_identifier(mapping["panel_id"], "panel_id"),
            units=tuple(
                PanelUnit.from_mapping(_mapping(item, "units item"))
                for item in _array(mapping["units"], "units")
            ),
            expected_periods=tuple(expected_periods),
            observations=tuple(
                PanelObservation.from_mapping(_mapping(item, "observations item"))
                for item in _array(mapping["observations"], "observations")
            ),
            synthetic_only=synthetic_only,
        )


@dataclass(frozen=True, slots=True)
class FrozenEvaluationInputs:
    """Serializable evidence emitted by a frozen evaluation.

    This object is deliberately *not* an execution credential.  A real
    evaluation additionally requires an opaque authorization issued by the
    temporal snapshot bridge.
    """

    schema_version: int
    mode: FreezeMode
    config_sha256: str
    panel_sha256: str
    data_manifest_sha256: str
    model_checkpoint_sha256: str
    prediction_artifact_sha256: str
    panel_artifact_sha256: str
    panel_lineage_sha256: str
    panel_derivation_registry_sha256: str
    panel_derivation_receipt_count: int
    transformation_implementation_sha256: str
    derivation_evidence_class: DerivationEvidenceClass
    panel_derivation_replay_verified: bool
    source_authority_verified: bool
    checkpoint_derivation_replay_verified: bool
    prediction_derivation_replay_verified: bool
    cohort_derivation_replay_verified: bool
    snapshot_id: str
    model_frozen_at: datetime
    predictions_frozen_at: datetime
    post_policy_data_first_available_at: datetime
    snapshot_frozen_at: datetime
    evaluation_started_at: datetime
    policy_start: datetime
    knowledge_cutoff: datetime

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 3:
            _fail("frozen-input receipt schema_version must be 3")
        if not isinstance(self.mode, FreezeMode):
            _fail("mode must be a FreezeMode")
        for field in (
            "config_sha256",
            "panel_sha256",
            "data_manifest_sha256",
            "model_checkpoint_sha256",
            "prediction_artifact_sha256",
            "panel_artifact_sha256",
            "panel_lineage_sha256",
            "panel_derivation_registry_sha256",
            "transformation_implementation_sha256",
        ):
            _sha256(getattr(self, field), field)
        if (
            isinstance(self.panel_derivation_receipt_count, bool)
            or not isinstance(self.panel_derivation_receipt_count, int)
            or self.panel_derivation_receipt_count < 0
        ):
            _fail("panel_derivation_receipt_count must be a non-negative integer")
        if not isinstance(self.derivation_evidence_class, DerivationEvidenceClass):
            _fail("derivation_evidence_class has an invalid schema")
        for field in (
            "panel_derivation_replay_verified",
            "source_authority_verified",
            "checkpoint_derivation_replay_verified",
            "prediction_derivation_replay_verified",
            "cohort_derivation_replay_verified",
        ):
            if not isinstance(getattr(self, field), bool):
                _fail(f"{field} must be a boolean")
        if (
            self.source_authority_verified
            or self.checkpoint_derivation_replay_verified
            or self.prediction_derivation_replay_verified
            or self.cohort_derivation_replay_verified
        ):
            _fail(
                "empirical derivation schema version 3 cannot claim source, "
                "checkpoint, prediction, or cohort authority"
            )
        if self.mode is FreezeMode.SYNTHETIC_TEST_FIXTURE:
            if (
                self.derivation_evidence_class
                is not DerivationEvidenceClass.UNVERIFIED_SYNTHETIC_FIXTURE
                or self.panel_derivation_replay_verified
                or self.panel_derivation_receipt_count != 0
            ):
                _fail("direct synthetic fixtures must remain explicitly unverified")
        elif (
            self.derivation_evidence_class
            is not DerivationEvidenceClass.SYNTHETIC_TABLE_REPLAY_ONLY
            or not self.panel_derivation_replay_verified
            or self.panel_derivation_receipt_count <= 0
        ):
            _fail(
                "verified snapshot mode requires a non-empty synthetic table "
                "derivation replay"
            )
        _identifier(self.snapshot_id, "snapshot_id")
        for field in (
            "model_frozen_at",
            "predictions_frozen_at",
            "post_policy_data_first_available_at",
            "snapshot_frozen_at",
            "evaluation_started_at",
            "policy_start",
            "knowledge_cutoff",
        ):
            object.__setattr__(self, field, _timestamp(getattr(self, field), field))
        if self.model_frozen_at > self.predictions_frozen_at:
            _fail("model must be frozen no later than predictions")
        if self.predictions_frozen_at >= self.post_policy_data_first_available_at:
            _fail("predictions must be frozen before policy-post data are available")
        if self.model_frozen_at >= self.post_policy_data_first_available_at:
            _fail("model must be frozen before policy-post data are available")
        if self.predictions_frozen_at > self.policy_start:
            _fail("predictions must be frozen no later than policy start")
        if self.policy_start > self.post_policy_data_first_available_at:
            _fail("policy-post data cannot predate policy start")
        if self.post_policy_data_first_available_at > self.snapshot_frozen_at:
            _fail("snapshot was frozen before its policy-post data were available")
        if self.snapshot_frozen_at > self.evaluation_started_at:
            _fail("snapshot must be frozen before empirical evaluation starts")
        if self.knowledge_cutoff > self.evaluation_started_at:
            _fail("knowledge cutoff must not follow evaluation start")
        if self.knowledge_cutoff < self.post_policy_data_first_available_at:
            _fail("knowledge cutoff precedes the first policy-post observation")

    @classmethod
    def for_synthetic_fixture(
        cls,
        config: EmpiricalEvaluationConfig,
        panel: FrozenPanel,
        *,
        frozen_at: datetime,
        evaluation_started_at: datetime,
        knowledge_cutoff: datetime,
    ) -> FrozenEvaluationInputs:
        """Build an explicit receipt for tests; never impersonate real data."""
        if not config.synthetic_test_only or not panel.synthetic_only:
            raise FrozenInputError(
                "synthetic receipts require config and panel to be marked "
                "synthetic_only"
            )
        if panel.schema_version != 1 or panel.derivation_receipts:
            raise FrozenInputError(
                "direct synthetic fixtures require an unverified schema-version-1 "
                "panel; receipt-backed panels must use the snapshot replay bridge"
            )
        panel_sha256 = panel.sha256
        post_policy_records = tuple(
            observation.record
            for observation in panel.observations
            if observation.relative_period >= 0
        )
        if not post_policy_records:
            raise FrozenInputError(
                "synthetic empirical panel requires policy-post observations"
            )
        first_post_available_at = min(
            record.available_time for record in post_policy_records
        )
        pre_reveal_freeze = min(
            frozen_at,
            config.treatment.policy_start - timedelta(microseconds=1),
            first_post_available_at - timedelta(microseconds=1),
        )
        return cls(
            schema_version=3,
            mode=FreezeMode.SYNTHETIC_TEST_FIXTURE,
            config_sha256=config.sha256,
            panel_sha256=panel_sha256,
            data_manifest_sha256=sha256_bytes(
                canonical_json_bytes(
                    {"panel_sha256": panel_sha256, "scope": "synthetic_fixture"}
                )
            ),
            model_checkpoint_sha256=sha256_bytes(
                canonical_json_bytes(
                    {"scope": "synthetic_dgp", "study_id": config.study_id}
                )
            ),
            prediction_artifact_sha256=sha256_bytes(
                canonical_json_bytes(
                    {
                        "scope": "synthetic_prediction_fixture",
                        "study_id": config.study_id,
                    }
                )
            ),
            panel_artifact_sha256=panel_sha256,
            panel_lineage_sha256=sha256_bytes(
                canonical_json_bytes(
                    {
                        "record_ids": sorted(
                            [
                                observation.record.record_id
                                for observation in panel.observations
                            ]
                            + [
                                characteristic.record.record_id
                                for unit in panel.units
                                for characteristic in unit.characteristics
                            ]
                        ),
                        "scope": "synthetic_panel_lineage",
                    }
                )
            ),
            panel_derivation_registry_sha256=panel.derivation_registry_sha256,
            panel_derivation_receipt_count=0,
            transformation_implementation_sha256=sha256_bytes(
                canonical_json_bytes({"scope": "not_applicable_unverified_fixture"})
            ),
            derivation_evidence_class=(
                DerivationEvidenceClass.UNVERIFIED_SYNTHETIC_FIXTURE
            ),
            panel_derivation_replay_verified=False,
            source_authority_verified=False,
            checkpoint_derivation_replay_verified=False,
            prediction_derivation_replay_verified=False,
            cohort_derivation_replay_verified=False,
            snapshot_id=f"synthetic:{panel.panel_id}",
            model_frozen_at=pre_reveal_freeze,
            predictions_frozen_at=pre_reveal_freeze,
            post_policy_data_first_available_at=first_post_available_at,
            snapshot_frozen_at=frozen_at,
            evaluation_started_at=evaluation_started_at,
            policy_start=config.treatment.policy_start,
            knowledge_cutoff=knowledge_cutoff,
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "checkpoint_derivation_replay_verified": (
                self.checkpoint_derivation_replay_verified
            ),
            "cohort_derivation_replay_verified": (
                self.cohort_derivation_replay_verified
            ),
            "config_sha256": self.config_sha256,
            "data_manifest_sha256": self.data_manifest_sha256,
            "evaluation_started_at": self.evaluation_started_at.isoformat(),
            "knowledge_cutoff": self.knowledge_cutoff.isoformat(),
            "mode": self.mode.value,
            "model_checkpoint_sha256": self.model_checkpoint_sha256,
            "model_frozen_at": self.model_frozen_at.isoformat(),
            "panel_sha256": self.panel_sha256,
            "panel_artifact_sha256": self.panel_artifact_sha256,
            "panel_derivation_registry_sha256": (self.panel_derivation_registry_sha256),
            "panel_derivation_receipt_count": self.panel_derivation_receipt_count,
            "panel_derivation_replay_verified": (self.panel_derivation_replay_verified),
            "panel_lineage_sha256": self.panel_lineage_sha256,
            "policy_start": self.policy_start.isoformat(),
            "post_policy_data_first_available_at": (
                self.post_policy_data_first_available_at.isoformat()
            ),
            "prediction_artifact_sha256": self.prediction_artifact_sha256,
            "prediction_derivation_replay_verified": (
                self.prediction_derivation_replay_verified
            ),
            "predictions_frozen_at": self.predictions_frozen_at.isoformat(),
            "schema_version": self.schema_version,
            "snapshot_frozen_at": self.snapshot_frozen_at.isoformat(),
            "snapshot_id": self.snapshot_id,
            "source_authority_verified": self.source_authority_verified,
            "transformation_implementation_sha256": (
                self.transformation_implementation_sha256
            ),
            "derivation_evidence_class": self.derivation_evidence_class.value,
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> Self:
        """Strictly reconstruct serialized freeze evidence.

        A reconstructed real-data receipt remains evidence rather than execution
        authority.  Callers must not use this parser to bypass the opaque temporal
        firewall authorization required by :class:`EmpiricalEvaluator`.
        """
        expected = frozenset(
            {
                "checkpoint_derivation_replay_verified",
                "cohort_derivation_replay_verified",
                "config_sha256",
                "data_manifest_sha256",
                "derivation_evidence_class",
                "evaluation_started_at",
                "knowledge_cutoff",
                "mode",
                "model_checkpoint_sha256",
                "model_frozen_at",
                "panel_artifact_sha256",
                "panel_derivation_registry_sha256",
                "panel_derivation_receipt_count",
                "panel_derivation_replay_verified",
                "panel_lineage_sha256",
                "panel_sha256",
                "policy_start",
                "post_policy_data_first_available_at",
                "prediction_artifact_sha256",
                "prediction_derivation_replay_verified",
                "predictions_frozen_at",
                "schema_version",
                "snapshot_frozen_at",
                "snapshot_id",
                "source_authority_verified",
                "transformation_implementation_sha256",
            }
        )
        _fields(mapping, expected, "frozen evaluation inputs")
        schema_version = mapping["schema_version"]
        if not isinstance(schema_version, int) or isinstance(schema_version, bool):
            _fail("frozen-input receipt schema_version must be an integer")
        raw_mode = mapping["mode"]
        if not isinstance(raw_mode, str):
            _fail("frozen-input mode must be a string")
        try:
            mode = FreezeMode(raw_mode)
        except ValueError as error:
            raise EmpiricalSchemaError("frozen-input mode is invalid") from error
        raw_evidence_class = mapping["derivation_evidence_class"]
        if not isinstance(raw_evidence_class, str):
            _fail("derivation_evidence_class must be a string")
        try:
            evidence_class = DerivationEvidenceClass(raw_evidence_class)
        except ValueError as error:
            raise EmpiricalSchemaError(
                "derivation_evidence_class is invalid"
            ) from error
        receipt_count = mapping["panel_derivation_receipt_count"]
        if not isinstance(receipt_count, int) or isinstance(receipt_count, bool):
            _fail("panel_derivation_receipt_count must be an integer")
        boolean_fields = (
            "panel_derivation_replay_verified",
            "source_authority_verified",
            "checkpoint_derivation_replay_verified",
            "prediction_derivation_replay_verified",
            "cohort_derivation_replay_verified",
        )
        if not all(isinstance(mapping[field], bool) for field in boolean_fields):
            _fail("derivation verification fields must be booleans")
        return cls(
            schema_version=schema_version,
            mode=mode,
            config_sha256=_sha256(mapping["config_sha256"], "config_sha256"),
            panel_sha256=_sha256(mapping["panel_sha256"], "panel_sha256"),
            data_manifest_sha256=_sha256(
                mapping["data_manifest_sha256"], "data_manifest_sha256"
            ),
            model_checkpoint_sha256=_sha256(
                mapping["model_checkpoint_sha256"], "model_checkpoint_sha256"
            ),
            prediction_artifact_sha256=_sha256(
                mapping["prediction_artifact_sha256"],
                "prediction_artifact_sha256",
            ),
            panel_artifact_sha256=_sha256(
                mapping["panel_artifact_sha256"], "panel_artifact_sha256"
            ),
            panel_lineage_sha256=_sha256(
                mapping["panel_lineage_sha256"], "panel_lineage_sha256"
            ),
            panel_derivation_registry_sha256=_sha256(
                mapping["panel_derivation_registry_sha256"],
                "panel_derivation_registry_sha256",
            ),
            panel_derivation_receipt_count=receipt_count,
            transformation_implementation_sha256=_sha256(
                mapping["transformation_implementation_sha256"],
                "transformation_implementation_sha256",
            ),
            derivation_evidence_class=evidence_class,
            panel_derivation_replay_verified=cast(
                bool, mapping["panel_derivation_replay_verified"]
            ),
            source_authority_verified=cast(bool, mapping["source_authority_verified"]),
            checkpoint_derivation_replay_verified=cast(
                bool, mapping["checkpoint_derivation_replay_verified"]
            ),
            prediction_derivation_replay_verified=cast(
                bool, mapping["prediction_derivation_replay_verified"]
            ),
            cohort_derivation_replay_verified=cast(
                bool, mapping["cohort_derivation_replay_verified"]
            ),
            snapshot_id=_identifier(mapping["snapshot_id"], "snapshot_id"),
            model_frozen_at=_parse_timestamp(
                mapping["model_frozen_at"], "model_frozen_at"
            ),
            predictions_frozen_at=_parse_timestamp(
                mapping["predictions_frozen_at"], "predictions_frozen_at"
            ),
            post_policy_data_first_available_at=_parse_timestamp(
                mapping["post_policy_data_first_available_at"],
                "post_policy_data_first_available_at",
            ),
            snapshot_frozen_at=_parse_timestamp(
                mapping["snapshot_frozen_at"], "snapshot_frozen_at"
            ),
            evaluation_started_at=_parse_timestamp(
                mapping["evaluation_started_at"], "evaluation_started_at"
            ),
            policy_start=_parse_timestamp(mapping["policy_start"], "policy_start"),
            knowledge_cutoff=_parse_timestamp(
                mapping["knowledge_cutoff"], "knowledge_cutoff"
            ),
        )


@dataclass(frozen=True, slots=True)
class NamedEstimate:
    """One point estimate retained even when the frozen sample is not estimable."""

    name: str
    point: float | None
    status: EstimateStatus
    n_treated: int
    n_control: int
    reason: str | None = None

    def __post_init__(self) -> None:
        _text(self.name, "estimate name")
        if not isinstance(self.status, EstimateStatus):
            _fail("estimate status is invalid")
        if self.point is not None:
            object.__setattr__(self, "point", _finite(self.point, "estimate point"))
        if self.status is EstimateStatus.ESTIMATED and self.point is None:
            _fail("estimated result requires a point value")
        if self.status is not EstimateStatus.ESTIMATED and self.point is not None:
            _fail("non-estimated result must not claim a point value")
        for field in ("n_treated", "n_control"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                _fail(f"{field} must be a non-negative integer")
        if self.reason is not None:
            _text(self.reason, "estimate reason")

    def to_mapping(self) -> dict[str, object]:
        return {
            "n_control": self.n_control,
            "n_treated": self.n_treated,
            "name": self.name,
            "point": self.point,
            "reason": self.reason,
            "status": self.status.value,
        }


@dataclass(frozen=True, slots=True)
class DrawValue:
    """One named value in a resampling draw; None records failed support."""

    name: str
    value: float | None

    def __post_init__(self) -> None:
        _text(self.name, "draw value name")
        if self.value is not None:
            object.__setattr__(self, "value", _finite(self.value, "draw value"))

    def to_mapping(self) -> dict[str, object]:
        return {"name": self.name, "value": self.value}


@dataclass(frozen=True, slots=True)
class BootstrapDraw:
    """One full cluster-bootstrap draw with an explicit stateless key."""

    replicate: int
    event_key: str
    values: tuple[DrawValue, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.replicate, bool)
            or not isinstance(self.replicate, int)
            or self.replicate < 0
        ):
            _fail("bootstrap replicate must be a non-negative integer")
        if (
            not isinstance(self.event_key, str)
            or re.fullmatch(r"ptk1:[0-9a-f]{64}", self.event_key) is None
        ):
            _fail("bootstrap event_key must be a PolicyTwin event key")
        if not isinstance(self.values, tuple) or not all(
            isinstance(value, DrawValue) for value in self.values
        ):
            _fail("bootstrap values must be an immutable tuple")
        names = tuple(value.name for value in self.values)
        if len(set(names)) != len(names):
            _fail("bootstrap draw value names must be unique")

    def to_mapping(self) -> dict[str, object]:
        return {
            "event_key": self.event_key,
            "replicate": self.replicate,
            "values": [item.to_mapping() for item in self.values],
        }


@dataclass(frozen=True, slots=True)
class ClusterBootstrapAudit:
    """Frozen support and draw-accounting gate for cluster inference.

    A draw is globally valid only when every registered estimate in that draw
    is defined.  This conservative gate prevents an apparently healthy primary
    interval from hiding unsupported subgroup or specification-curve rows.
    """

    total_clusters: int
    treated_clusters: int
    control_clusters: int
    mixed_cluster_ids: tuple[str, ...]
    minimum_total_clusters: int
    minimum_treated_clusters: int
    minimum_control_clusters: int
    requested_draws: int
    valid_draws: int
    invalid_draws: int
    minimum_valid_draws: int
    maximum_invalid_fraction: float

    def __post_init__(self) -> None:
        for field in (
            "total_clusters",
            "treated_clusters",
            "control_clusters",
            "requested_draws",
            "valid_draws",
            "invalid_draws",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                _fail(f"cluster bootstrap {field} must be a non-negative integer")
        for field in (
            "minimum_total_clusters",
            "minimum_treated_clusters",
            "minimum_control_clusters",
            "minimum_valid_draws",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                _fail(f"cluster bootstrap {field} must be a positive integer")
        if self.minimum_treated_clusters < 2 or self.minimum_control_clusters < 2:
            _fail("cluster bootstrap requires at least two frozen clusters per arm")
        if self.minimum_total_clusters < (
            self.minimum_treated_clusters + self.minimum_control_clusters
        ):
            _fail("cluster bootstrap total-cluster floor is inconsistent")
        if self.requested_draws < 1:
            _fail("cluster bootstrap requested_draws must be positive")
        if self.valid_draws + self.invalid_draws != self.requested_draws:
            _fail("cluster bootstrap draw accounting is inconsistent")
        if self.minimum_valid_draws > self.requested_draws:
            _fail("cluster bootstrap valid-draw floor exceeds requested draws")
        object.__setattr__(
            self,
            "maximum_invalid_fraction",
            _fraction(
                self.maximum_invalid_fraction,
                "cluster bootstrap maximum_invalid_fraction",
            ),
        )
        mixed = _unique_identifiers(
            self.mixed_cluster_ids, "cluster bootstrap mixed_cluster_ids"
        )
        if tuple(sorted(mixed)) != mixed:
            _fail("cluster bootstrap mixed_cluster_ids must be sorted")
        if self.total_clusters != (
            self.treated_clusters + self.control_clusters + len(mixed)
        ):
            _fail("cluster bootstrap cluster counts are inconsistent")

    @property
    def invalid_fraction(self) -> float:
        """Return the retained invalid-draw fraction."""
        return self.invalid_draws / self.requested_draws

    @property
    def support_gate_passed(self) -> bool:
        """Require disjoint cluster assignment and every frozen support floor."""
        return (
            not self.mixed_cluster_ids
            and self.total_clusters >= self.minimum_total_clusters
            and self.treated_clusters >= self.minimum_treated_clusters
            and self.control_clusters >= self.minimum_control_clusters
        )

    @property
    def valid_draw_gate_passed(self) -> bool:
        """Require support, the draw floor, and the frozen invalid-fraction cap."""
        return (
            self.support_gate_passed
            and self.valid_draws >= self.minimum_valid_draws
            and self.invalid_fraction <= self.maximum_invalid_fraction
        )

    @property
    def reasons(self) -> tuple[str, ...]:
        """Return deterministic reason codes without inspecting effect magnitude."""
        reasons: list[str] = []
        if self.mixed_cluster_ids:
            reasons.append("mixed_treatment_control_clusters")
        if self.total_clusters < self.minimum_total_clusters:
            reasons.append("insufficient_total_clusters")
        if self.treated_clusters < self.minimum_treated_clusters:
            reasons.append("insufficient_treated_clusters")
        if self.control_clusters < self.minimum_control_clusters:
            reasons.append("insufficient_control_clusters")
        if self.valid_draws < self.minimum_valid_draws:
            reasons.append("insufficient_valid_bootstrap_draws")
        if self.invalid_fraction > self.maximum_invalid_fraction:
            reasons.append("bootstrap_invalid_fraction_exceeded")
        return tuple(reasons)

    def to_mapping(self) -> dict[str, object]:
        return {
            "control_clusters": self.control_clusters,
            "invalid_draws": self.invalid_draws,
            "invalid_fraction": self.invalid_fraction,
            "maximum_invalid_fraction": self.maximum_invalid_fraction,
            "minimum_control_clusters": self.minimum_control_clusters,
            "minimum_total_clusters": self.minimum_total_clusters,
            "minimum_treated_clusters": self.minimum_treated_clusters,
            "minimum_valid_draws": self.minimum_valid_draws,
            "mixed_cluster_ids": list(self.mixed_cluster_ids),
            "reasons": list(self.reasons),
            "requested_draws": self.requested_draws,
            "support_gate_passed": self.support_gate_passed,
            "total_clusters": self.total_clusters,
            "treated_clusters": self.treated_clusters,
            "valid_draw_gate_passed": self.valid_draw_gate_passed,
            "valid_draws": self.valid_draws,
        }


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    """Percentile interval derived from the retained bootstrap distribution."""

    name: str
    confidence_level: float
    lower: float | None
    upper: float | None
    valid_draws: int
    minimum_valid_draws: int = 1
    inference_gate_passed: bool = True

    def __post_init__(self) -> None:
        _text(self.name, "confidence interval name")
        object.__setattr__(
            self,
            "confidence_level",
            _fraction(self.confidence_level, "confidence_level", open_interval=True),
        )
        if (self.lower is None) != (self.upper is None):
            _fail("confidence interval endpoints must be present together")
        if self.lower is not None and self.upper is not None:
            object.__setattr__(self, "lower", _finite(self.lower, "interval lower"))
            object.__setattr__(self, "upper", _finite(self.upper, "interval upper"))
            if self.lower > self.upper:
                _fail("confidence interval lower endpoint exceeds upper endpoint")
        if (
            isinstance(self.valid_draws, bool)
            or not isinstance(self.valid_draws, int)
            or self.valid_draws < 0
        ):
            _fail("valid_draws must be a non-negative integer")
        if (
            isinstance(self.minimum_valid_draws, bool)
            or not isinstance(self.minimum_valid_draws, int)
            or self.minimum_valid_draws < 1
        ):
            _fail("minimum_valid_draws must be a positive integer")
        if not isinstance(self.inference_gate_passed, bool):
            _fail("inference_gate_passed must be a boolean")
        if self.lower is not None and not self.valid_draw_gate_passed:
            _fail("confidence interval endpoints require the frozen valid-draw gate")

    @property
    def valid_draw_gate_passed(self) -> bool:
        """Return whether the frozen valid bootstrap-draw floor was met."""
        return (
            self.inference_gate_passed and self.valid_draws >= self.minimum_valid_draws
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "confidence_level": self.confidence_level,
            "lower": self.lower,
            "name": self.name,
            "inference_gate_passed": self.inference_gate_passed,
            "minimum_valid_draws": self.minimum_valid_draws,
            "upper": self.upper,
            "valid_draws": self.valid_draws,
            "valid_draw_gate_passed": self.valid_draw_gate_passed,
        }


@dataclass(frozen=True, slots=True)
class RandomizationDraw:
    """One complete-randomization placebo assignment and statistic."""

    replicate: int
    event_key: str
    statistic: float | None

    def __post_init__(self) -> None:
        if (
            isinstance(self.replicate, bool)
            or not isinstance(self.replicate, int)
            or self.replicate < 0
        ):
            _fail("randomization replicate must be a non-negative integer")
        if (
            not isinstance(self.event_key, str)
            or re.fullmatch(r"ptk1:[0-9a-f]{64}", self.event_key) is None
        ):
            _fail("randomization event_key must be a PolicyTwin event key")
        if self.statistic is not None:
            object.__setattr__(
                self, "statistic", _finite(self.statistic, "randomization statistic")
            )

    def to_mapping(self) -> dict[str, object]:
        return {
            "event_key": self.event_key,
            "replicate": self.replicate,
            "statistic": self.statistic,
        }


@dataclass(frozen=True, slots=True)
class RandomizationInferenceResult:
    """Observed DiD, complete null distribution, and finite-sample p-value."""

    observed_statistic: float | None
    p_value: float | None
    draws: tuple[RandomizationDraw, ...]
    requested_draws: int
    minimum_valid_draws: int = 1

    def __post_init__(self) -> None:
        if self.observed_statistic is not None:
            object.__setattr__(
                self,
                "observed_statistic",
                _finite(self.observed_statistic, "observed statistic"),
            )
        if self.p_value is not None:
            object.__setattr__(self, "p_value", _fraction(self.p_value, "p_value"))
        if not isinstance(self.draws, tuple) or not all(
            isinstance(draw, RandomizationDraw) for draw in self.draws
        ):
            _fail("randomization draws must be an immutable tuple")
        replicates = tuple(draw.replicate for draw in self.draws)
        if replicates != tuple(range(len(self.draws))):
            _fail("randomization draw replicates must be contiguous from zero")
        keys = tuple(draw.event_key for draw in self.draws)
        if len(set(keys)) != len(keys):
            _fail("randomization event keys must be unique")
        if (
            isinstance(self.requested_draws, bool)
            or not isinstance(self.requested_draws, int)
            or self.requested_draws < 1
        ):
            _fail("randomization requested_draws must be a positive integer")
        if len(self.draws) != self.requested_draws:
            _fail("retained randomization draws must equal requested_draws")
        if (
            isinstance(self.minimum_valid_draws, bool)
            or not isinstance(self.minimum_valid_draws, int)
            or self.minimum_valid_draws < 1
        ):
            _fail("randomization minimum_valid_draws must be a positive integer")
        if self.p_value is not None and (
            self.observed_statistic is None or not self.valid_draw_gate_passed
        ):
            _fail(
                "randomization p_value requires an observed statistic and the "
                "valid-draw gate"
            )

    @property
    def valid_draws(self) -> int:
        """Return the retained finite placebo-statistic count."""
        return sum(draw.statistic is not None for draw in self.draws)

    @property
    def invalid_draws(self) -> int:
        """Return the retained undefined placebo-statistic count."""
        return self.requested_draws - self.valid_draws

    @property
    def valid_draw_gate_passed(self) -> bool:
        """Require the frozen draw floor and a fully defined randomization support."""
        return self.invalid_draws == 0 and self.valid_draws >= self.minimum_valid_draws

    def to_mapping(self) -> dict[str, object]:
        return {
            "draws": [item.to_mapping() for item in self.draws],
            "invalid_draws": self.invalid_draws,
            "minimum_valid_draws": self.minimum_valid_draws,
            "observed_statistic": self.observed_statistic,
            "p_value": self.p_value,
            "requested_draws": self.requested_draws,
            "valid_draw_gate_passed": self.valid_draw_gate_passed,
            "valid_draws": self.valid_draws,
        }


@dataclass(frozen=True, slots=True)
class PreTrendDiagnostic:
    """Frozen-threshold parallel-trend and pre-treatment lead diagnostics."""

    status: DiagnosticStatus
    lead_estimates: tuple[DrawValue, ...]
    max_absolute_lead: float | None
    treated_control_slope_gap: float | None
    max_absolute_lead_tolerance: float
    max_slope_gap_per_day: float
    reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, DiagnosticStatus):
            _fail("pretrend status is invalid")
        if not isinstance(self.lead_estimates, tuple) or not all(
            isinstance(item, DrawValue) for item in self.lead_estimates
        ):
            _fail("lead_estimates must be an immutable tuple")
        for field in ("max_absolute_lead", "treated_control_slope_gap"):
            value = getattr(self, field)
            if value is not None:
                object.__setattr__(self, field, _finite(value, field))
        for field in (
            "max_absolute_lead_tolerance",
            "max_slope_gap_per_day",
        ):
            tolerance = _finite(getattr(self, field), field)
            if tolerance < 0:
                _fail(f"{field} must not be negative")
            object.__setattr__(self, field, tolerance)
        if self.reason is not None:
            _text(self.reason, "pretrend reason")

    def to_mapping(self) -> dict[str, object]:
        return {
            "lead_estimates": [item.to_mapping() for item in self.lead_estimates],
            "max_absolute_lead": self.max_absolute_lead,
            "max_absolute_lead_tolerance": self.max_absolute_lead_tolerance,
            "max_slope_gap_per_day": self.max_slope_gap_per_day,
            "reason": self.reason,
            "status": self.status.value,
            "treated_control_slope_gap": self.treated_control_slope_gap,
        }


@dataclass(frozen=True, slots=True)
class PeriodCompositionAudit:
    """Exact arm-by-period accounting against the frozen cohort universe."""

    relative_period: int
    arm: str
    expected: int
    observed: int
    missing: int
    exited: int
    not_in_sample: int
    entrants: int
    exits: int
    reentries: int
    observed_unit_ids_sha256: str
    characteristic_means: tuple[tuple[str, float], ...]
    post_treatment_group_counts: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.relative_period, int) or isinstance(
            self.relative_period, bool
        ):
            _fail("composition relative_period must be an integer")
        if self.arm not in {"treated", "control"}:
            _fail("composition arm must be treated or control")
        for field in (
            "expected",
            "observed",
            "missing",
            "exited",
            "not_in_sample",
            "entrants",
            "exits",
            "reentries",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                _fail(f"composition {field} must be a non-negative integer")
        if (
            self.observed + self.missing + self.exited + self.not_in_sample
            != self.expected
        ):
            _fail("composition status counts do not close to the expected cohort")
        _sha256(self.observed_unit_ids_sha256, "observed_unit_ids_sha256")
        if not isinstance(self.characteristic_means, tuple):
            _fail("characteristic_means must be an immutable tuple")
        characteristic_names: list[str] = []
        for name, value in self.characteristic_means:
            characteristic_names.append(_identifier(name, "characteristic mean name"))
            _finite(value, "characteristic mean")
        if len(set(characteristic_names)) != len(characteristic_names):
            _fail("characteristic mean names must be unique")
        if not isinstance(self.post_treatment_group_counts, tuple):
            _fail("post-treatment group counts must be an immutable tuple")
        group_names: list[str] = []
        for name, count in self.post_treatment_group_counts:
            group_names.append(_text(name, "post-treatment group audit key"))
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                _fail("post-treatment group counts must be non-negative integers")
        if len(set(group_names)) != len(group_names):
            _fail("post-treatment group audit keys must be unique")

    def to_mapping(self) -> dict[str, object]:
        return {
            "arm": self.arm,
            "characteristic_means": dict(self.characteristic_means),
            "entrants": self.entrants,
            "exited": self.exited,
            "exits": self.exits,
            "expected": self.expected,
            "missing": self.missing,
            "not_in_sample": self.not_in_sample,
            "observed": self.observed,
            "observed_unit_ids_sha256": self.observed_unit_ids_sha256,
            "post_treatment_group_counts": dict(self.post_treatment_group_counts),
            "reentries": self.reentries,
            "relative_period": self.relative_period,
        }


@dataclass(frozen=True, slots=True)
class SampleAudit:
    """Complete missingness, attrition, and composition audit for one spec."""

    period_audits: tuple[PeriodCompositionAudit, ...]
    complete_case_unit_ids: tuple[str, ...]
    excluded_units: tuple[tuple[str, str], ...]
    post_policy_observation_exclusion_unit_ids: tuple[str, ...]
    registered_covariate_exclusion_unit_ids: tuple[str, ...]
    max_missing_fraction_observed: float
    max_attrition_fraction_observed: float
    max_composition_shift_observed: float
    flags: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.period_audits, tuple) or not all(
            isinstance(item, PeriodCompositionAudit) for item in self.period_audits
        ):
            _fail("period_audits must be an immutable tuple")
        _unique_identifiers(self.complete_case_unit_ids, "complete_case_unit_ids")
        if not isinstance(self.excluded_units, tuple):
            _fail("excluded_units must be an immutable tuple")
        for unit_id, reason in self.excluded_units:
            _identifier(unit_id, "excluded unit_id")
            _text(reason, "excluded unit reason")
        post_policy_exclusions = _unique_identifiers(
            self.post_policy_observation_exclusion_unit_ids,
            "post_policy_observation_exclusion_unit_ids",
        )
        excluded_unit_ids = {unit_id for unit_id, _ in self.excluded_units}
        if not set(post_policy_exclusions) <= excluded_unit_ids:
            _fail("post-policy exclusion units must also appear in excluded_units")
        covariate_exclusions = _unique_identifiers(
            self.registered_covariate_exclusion_unit_ids,
            "registered_covariate_exclusion_unit_ids",
        )
        if not set(covariate_exclusions) <= excluded_unit_ids:
            _fail("registered-covariate exclusion units must appear in excluded_units")
        object.__setattr__(
            self,
            "max_missing_fraction_observed",
            _fraction(
                self.max_missing_fraction_observed,
                "max_missing_fraction_observed",
            ),
        )
        object.__setattr__(
            self,
            "max_attrition_fraction_observed",
            _fraction(
                self.max_attrition_fraction_observed,
                "max_attrition_fraction_observed",
            ),
        )
        shift = _finite(
            self.max_composition_shift_observed,
            "max_composition_shift_observed",
        )
        if shift < 0:
            _fail("max_composition_shift_observed must not be negative")
        object.__setattr__(self, "max_composition_shift_observed", shift)
        if not isinstance(self.flags, tuple) or not all(
            isinstance(flag, str) and flag for flag in self.flags
        ):
            _fail("sample audit flags must be immutable non-empty strings")
        hard_flag = "post_policy_observation_exclusion_under_complete_case"
        if bool(post_policy_exclusions) != (hard_flag in self.flags):
            _fail("post-policy exclusion units and hard audit flag must agree")
        reason_covariate_exclusions = tuple(
            unit_id
            for unit_id, reason in self.excluded_units
            if any(item.startswith("missing_covariates:") for item in reason.split(";"))
        )
        if set(covariate_exclusions) != set(reason_covariate_exclusions):
            _fail(
                "registered-covariate exclusion units and excluded reasons must agree"
            )
        covariate_flag = "registered_covariate_exclusion_under_complete_case"
        if bool(covariate_exclusions) != (covariate_flag in self.flags):
            _fail("registered-covariate exclusions and hard audit flag must agree")

    @property
    def registered_covariate_exclusion_count(self) -> int:
        """Return the frozen-cohort count excluded for registered covariates."""
        return len(self.registered_covariate_exclusion_unit_ids)

    def to_mapping(self) -> dict[str, object]:
        return {
            "complete_case_unit_ids": list(self.complete_case_unit_ids),
            "excluded_units": [
                {"reason": reason, "unit_id": unit_id}
                for unit_id, reason in self.excluded_units
            ],
            "flags": list(self.flags),
            "max_attrition_fraction_observed": self.max_attrition_fraction_observed,
            "max_composition_shift_observed": self.max_composition_shift_observed,
            "max_missing_fraction_observed": self.max_missing_fraction_observed,
            "period_audits": [item.to_mapping() for item in self.period_audits],
            "post_policy_observation_exclusion_unit_ids": list(
                self.post_policy_observation_exclusion_unit_ids
            ),
            "registered_covariate_exclusion_count": (
                self.registered_covariate_exclusion_count
            ),
            "registered_covariate_exclusion_unit_ids": list(
                self.registered_covariate_exclusion_unit_ids
            ),
        }


@dataclass(frozen=True, slots=True)
class CausalReportingDecision:
    """Frozen-criteria eligibility decision for one specification row.

    Reasons are diagnostic/support codes only.  The sign, magnitude, and
    significance of the target treatment estimate are deliberately absent.
    """

    status: CausalReportingStatus
    criteria_sha256: str
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.status, CausalReportingStatus):
            _fail("causal reporting status is invalid")
        _sha256(self.criteria_sha256, "causal reporting criteria_sha256")
        if not isinstance(self.reasons, tuple):
            _fail("causal reporting reasons must be an immutable tuple")
        reasons = tuple(
            sorted(
                _identifier(item, "causal reporting reason") for item in self.reasons
            )
        )
        if len(set(reasons)) != len(reasons):
            _fail("causal reporting reasons must be unique")
        object.__setattr__(self, "reasons", reasons)
        if self.status is CausalReportingStatus.VALID and reasons:
            _fail("valid causal reporting status must not carry invalidity reasons")
        if self.status is CausalReportingStatus.INVALID and not reasons:
            _fail("invalid causal reporting status requires at least one reason")

    def to_mapping(self) -> dict[str, object]:
        return {
            "criteria_sha256": self.criteria_sha256,
            "reasons": list(self.reasons),
            "status": self.status.value,
        }


@dataclass(frozen=True, slots=True)
class SpecificationResult:
    """All outputs for one frozen specification, including failed diagnostics."""

    specification_id: str
    estimates: tuple[NamedEstimate, ...]
    confidence_intervals: tuple[ConfidenceInterval, ...]
    bootstrap_draws: tuple[BootstrapDraw, ...]
    bootstrap_inference: ClusterBootstrapAudit
    randomization_inference: RandomizationInferenceResult
    pretrend_diagnostic: PreTrendDiagnostic
    sample_audit: SampleAudit
    causal_reporting: CausalReportingDecision

    def __post_init__(self) -> None:
        _identifier(self.specification_id, "result specification_id")
        if not isinstance(self.estimates, tuple) or not all(
            isinstance(item, NamedEstimate) for item in self.estimates
        ):
            _fail("estimates must be an immutable tuple")
        names = tuple(item.name for item in self.estimates)
        if len(set(names)) != len(names):
            _fail("estimate names must be unique within a specification")
        if not isinstance(self.confidence_intervals, tuple) or not all(
            isinstance(item, ConfidenceInterval) for item in self.confidence_intervals
        ):
            _fail("confidence_intervals must be an immutable tuple")
        interval_names = tuple(item.name for item in self.confidence_intervals)
        if interval_names != names:
            _fail("confidence intervals must exactly match the estimate set")
        if not isinstance(self.bootstrap_draws, tuple) or not all(
            isinstance(draw, BootstrapDraw) for draw in self.bootstrap_draws
        ):
            _fail("bootstrap_draws must be an immutable tuple")
        replicates = tuple(draw.replicate for draw in self.bootstrap_draws)
        if replicates != tuple(range(len(self.bootstrap_draws))):
            _fail("bootstrap draw replicates must be contiguous from zero")
        event_keys = tuple(draw.event_key for draw in self.bootstrap_draws)
        if len(set(event_keys)) != len(event_keys):
            _fail("bootstrap event keys must be unique")
        for draw in self.bootstrap_draws:
            if tuple(value.name for value in draw.values) != names:
                _fail("bootstrap draw values must exactly match the estimate set")
        if not isinstance(self.bootstrap_inference, ClusterBootstrapAudit):
            _fail("bootstrap_inference has an invalid schema")
        if self.bootstrap_inference.requested_draws != len(self.bootstrap_draws):
            _fail("bootstrap audit requested draws differ from retained draws")
        globally_valid = sum(
            bool(draw.values) and all(value.value is not None for value in draw.values)
            for draw in self.bootstrap_draws
        )
        if (
            self.bootstrap_inference.valid_draws != globally_valid
            or self.bootstrap_inference.invalid_draws
            != len(self.bootstrap_draws) - globally_valid
        ):
            _fail("bootstrap audit draw accounting differs from retained draws")
        if any(
            interval.inference_gate_passed
            is not self.bootstrap_inference.valid_draw_gate_passed
            for interval in self.confidence_intervals
        ):
            _fail("confidence intervals do not retain the global bootstrap gate")
        if not isinstance(self.randomization_inference, RandomizationInferenceResult):
            _fail("randomization_inference has an invalid schema")
        if not isinstance(self.pretrend_diagnostic, PreTrendDiagnostic):
            _fail("pretrend_diagnostic has an invalid schema")
        if not isinstance(self.sample_audit, SampleAudit):
            _fail("sample_audit has an invalid schema")
        if not isinstance(self.causal_reporting, CausalReportingDecision):
            _fail("causal_reporting has an invalid schema")
        if self.causal_reporting.status is CausalReportingStatus.VALID and (
            self.pretrend_diagnostic.status is not DiagnosticStatus.PASS
            or self.sample_audit.flags
            or any(
                item.status is not EstimateStatus.ESTIMATED for item in self.estimates
            )
            or any(
                not item.valid_draw_gate_passed for item in self.confidence_intervals
            )
            or not self.bootstrap_inference.valid_draw_gate_passed
            or not self.randomization_inference.valid_draw_gate_passed
        ):
            _fail("causal-valid specification contradicts retained diagnostics")

    def to_mapping(self) -> dict[str, object]:
        return {
            "bootstrap_draws": [item.to_mapping() for item in self.bootstrap_draws],
            "bootstrap_inference": self.bootstrap_inference.to_mapping(),
            "causal_reporting": self.causal_reporting.to_mapping(),
            "confidence_intervals": [
                item.to_mapping() for item in self.confidence_intervals
            ],
            "estimates": [item.to_mapping() for item in self.estimates],
            "pretrend_diagnostic": self.pretrend_diagnostic.to_mapping(),
            "randomization_inference": self.randomization_inference.to_mapping(),
            "sample_audit": self.sample_audit.to_mapping(),
            "specification_id": self.specification_id,
        }


@dataclass(frozen=True, slots=True)
class EmpiricalEvaluationArtifact:
    """Versioned full specification curve with no selected/best specification."""

    schema_version: int
    study_id: str
    synthetic_only: bool
    config_sha256: str
    panel_sha256: str
    causal_reporting_criteria_sha256: str
    registered_specification_ids: tuple[str, ...]
    frozen_inputs: FrozenEvaluationInputs
    specification_results: tuple[SpecificationResult, ...]
    selection_policy: str = "all_frozen_specifications_reported"

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 7:
            _fail("empirical artifact schema_version must be 7")
        _identifier(self.study_id, "artifact study_id")
        if not isinstance(self.synthetic_only, bool):
            _fail("artifact synthetic_only must be a boolean")
        _sha256(self.config_sha256, "artifact config_sha256")
        _sha256(self.panel_sha256, "artifact panel_sha256")
        _sha256(
            self.causal_reporting_criteria_sha256,
            "artifact causal_reporting_criteria_sha256",
        )
        if not isinstance(self.frozen_inputs, FrozenEvaluationInputs):
            _fail("artifact frozen_inputs receipt is invalid")
        if (
            self.frozen_inputs.mode is FreezeMode.SYNTHETIC_TEST_FIXTURE
            and not self.synthetic_only
        ):
            _fail("artifact synthetic_only marker contradicts frozen-input authority")
        if (
            self.config_sha256 != self.frozen_inputs.config_sha256
            or self.panel_sha256 != self.frozen_inputs.panel_sha256
        ):
            _fail("artifact commitments contradict the frozen-input receipt")
        registered_ids = _unique_identifiers(
            self.registered_specification_ids,
            "registered_specification_ids",
        )
        if not registered_ids:
            _fail("registered_specification_ids must not be empty")
        if not isinstance(self.specification_results, tuple) or not all(
            isinstance(item, SpecificationResult) for item in self.specification_results
        ):
            _fail("specification_results must be an immutable tuple")
        result_ids = tuple(item.specification_id for item in self.specification_results)
        if len(set(result_ids)) != len(result_ids):
            _fail("artifact specification IDs must be unique")
        if result_ids != registered_ids:
            _fail(
                "artifact results must retain every registered specification in order"
            )
        if self.selection_policy != "all_frozen_specifications_reported":
            _fail("empirical artifacts may not select a preferred specification")
        if any(
            item.causal_reporting.criteria_sha256
            != self.causal_reporting_criteria_sha256
            for item in self.specification_results
        ):
            _fail("specification causal-reporting criteria commitments differ")

    @property
    def sha256(self) -> str:
        """Return a deterministic digest of the complete scientific artifact."""
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def export_for(self, consumer: ArtifactConsumer) -> dict[str, object]:
        """Export only to reporting/audit; reject simulator feedback by construction."""
        if consumer in {
            ArtifactConsumer.SIMULATOR_CALIBRATION,
            ArtifactConsumer.SIMULATOR_TUNING,
        }:
            raise FeedbackProhibitedError(
                "post-hoc empirical estimates must not enter simulator calibration "
                "or tuning"
            )
        if consumer not in {
            ArtifactConsumer.REPORTING,
            ArtifactConsumer.REPRODUCIBILITY_AUDIT,
        }:
            raise FeedbackProhibitedError("unrecognized empirical artifact consumer")
        return self.to_mapping()

    def to_mapping(self) -> dict[str, object]:
        return {
            "config_sha256": self.config_sha256,
            "causal_reporting_criteria_sha256": (self.causal_reporting_criteria_sha256),
            "frozen_inputs": self.frozen_inputs.to_mapping(),
            "panel_sha256": self.panel_sha256,
            "registered_specification_ids": list(self.registered_specification_ids),
            "schema_version": self.schema_version,
            "selection_policy": self.selection_policy,
            "specification_results": [
                item.to_mapping() for item in self.specification_results
            ],
            "study_id": self.study_id,
            "synthetic_only": self.synthetic_only,
        }

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, object],
        *,
        config: EmpiricalEvaluationConfig,
        panel: FrozenPanel,
        frozen_inputs: FrozenEvaluationInputs,
    ) -> Self:
        """Reconstruct only by re-running the complete frozen estimator.

        Result fields are intentionally not trusted or parsed independently.  The
        supplied typed config, complete frozen panel, and freeze receipt reproduce
        every point estimate, draw, interval, diagnostic, sample audit, subgroup,
        and specification-curve entry.  Exact canonical equality is required.
        For a retained receipt-backed synthetic panel, this proves estimator
        reproducibility and internal cross-binding only: it does not reopen or
        revalidate the source TABLE snapshot and does not issue execution
        authority.  Serialized real-data receipts cannot authorize this replay
        and remain categorically blocked outside the temporal-firewall bridge.
        """
        if not isinstance(config, EmpiricalEvaluationConfig):
            _fail("artifact verifier config has an invalid schema")
        if not isinstance(panel, FrozenPanel):
            _fail("artifact verifier panel has an invalid schema")
        if not isinstance(frozen_inputs, FrozenEvaluationInputs):
            _fail("artifact verifier frozen inputs have an invalid schema")
        from policytwin.empirical.evaluator import (
            EmpiricalEvaluator,
            _replay_verified_synthetic_artifact,
        )

        if frozen_inputs.mode is FreezeMode.VERIFIED_TEMPORAL_SNAPSHOT:
            if not config.synthetic_test_only or not panel.synthetic_only:
                raise FrozenInputError(
                    "serialized real-data receipt is not replay authority; an "
                    "external temporal authorization is required"
                )
            expected = _replay_verified_synthetic_artifact(
                config=config,
                panel=panel,
                frozen_inputs=frozen_inputs,
            )
        else:
            expected = EmpiricalEvaluator().evaluate(
                config=config,
                panel=panel,
                frozen_inputs=frozen_inputs,
            )
        if canonical_json_bytes(mapping) != canonical_json_bytes(expected.to_mapping()):
            _fail(
                "serialized empirical results do not reproduce from the complete "
                "frozen config and panel"
            )
        return cast(Self, expected)


__all__ = [
    "ArtifactConsumer",
    "BootstrapDraw",
    "CausalReportingDecision",
    "CausalReportingStatus",
    "ClusterBootstrapAudit",
    "ConfidenceInterval",
    "DiagnosticStatus",
    "DerivationEvidenceClass",
    "DrawValue",
    "EmpiricalEvaluationArtifact",
    "EmpiricalEvaluationConfig",
    "EmpiricalEvaluationError",
    "EmpiricalSchemaError",
    "EmpiricalSpecification",
    "EstimateStatus",
    "FeedbackProhibitedError",
    "FreezeMode",
    "FrozenEvaluationInputs",
    "FrozenInputError",
    "FrozenPanel",
    "MissingDataPolicy",
    "NamedEstimate",
    "ObservationStatus",
    "PanelObservation",
    "PanelUnit",
    "PeriodTimeMapping",
    "PeriodCompositionAudit",
    "PostTreatmentGroup",
    "PreTreatmentCharacteristic",
    "PreTrendDiagnostic",
    "RandomizationDraw",
    "RandomizationDesign",
    "RandomizationInferenceResult",
    "SampleAudit",
    "SpecificationResult",
    "SubgroupDefinition",
    "TreatmentDefinition",
]
