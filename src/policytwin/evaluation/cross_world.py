"""Synthetic-only sensitivity analysis for cross-world outcome couplings.

Marginal outcome distributions do not identify a joint distribution across
policy worlds.  This module makes that ambiguity explicit for finite,
equal-weight synthetic empirical measures.  It does not authorize a real-unit
coupling or a real-world causal interpretation.
"""

from __future__ import annotations

import math
import os
import re
import stat
import sys
from bisect import bisect_left, bisect_right
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from pathlib import Path
from typing import Literal, NoReturn, Self, cast

from policytwin.counterfactual.events import SemanticEventKey
from policytwin.counterfactual.paired import WorldRole
from policytwin.protocol import CohortBasis, FrozenCohort
from policytwin.provenance import (
    canonical_json_bytes,
    read_json_bytes_object,
    sha256_bytes,
)

CROSS_WORLD_SCHEMA_VERSION = "policytwin.cross-world-coupling.v2"
CROSS_WORLD_CLAIM_SCOPE: Literal["synthetic_cross_world_coupling_sensitivity_only"] = (
    "synthetic_cross_world_coupling_sensitivity_only"
)
CROSS_WORLD_SOURCE_AUTHORITY: Literal[
    "unverified_caller_authored_synthetic_marginals"
] = "unverified_caller_authored_synthetic_marginals"

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CANONICAL_POSITIVE_DECIMAL = re.compile(
    r"^(?:0\.[0-9]*[1-9]|[1-9][0-9]*(?:\.[0-9]*[1-9])?)$"
)
_MAX_SAMPLE_SIZE = 100_000
_MAX_ARTIFACT_BYTES = 256 * 1024 * 1024
_MAX_SAFE_MAGNITUDE = math.sqrt(sys.float_info.max) / 16.0
_MIN_SAFE_QUANTUM = math.sqrt(sys.float_info.min * _MAX_SAMPLE_SIZE) * 16.0


class CrossWorldCouplingError(ValueError):
    """Raised when cross-world sensitivity inputs fail closed."""


def _fail(message: str) -> NoReturn:
    raise CrossWorldCouplingError(message)


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        _fail(f"{field} must be a stable ASCII identifier")
    return value


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail(f"{field} must be a lowercase SHA-256 digest")
    return value


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        _fail(f"{field} must be an object with string keys")
    return cast(Mapping[str, object], value)


def _fields(value: Mapping[str, object], expected: frozenset[str], field: str) -> None:
    if frozenset(value) != expected:
        _fail(f"{field} has missing or unknown fields")


def _string(value: object, field: str) -> str:
    if not isinstance(value, str):
        _fail(f"{field} must be a string")
    return value


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{field} must be an integer")
    return value


def _positive_decimal(value: object, field: str) -> str:
    text = _string(value, field)
    if len(text) > 512:
        _fail(f"{field} exceeds the decimal descriptor length limit")
    if _CANONICAL_POSITIVE_DECIMAL.fullmatch(text) is None:
        _fail(f"{field} must be a canonical positive decimal string")
    try:
        parsed = Decimal(text)
    except InvalidOperation as error:
        raise CrossWorldCouplingError(f"{field} is invalid") from error
    if not parsed.is_finite() or parsed <= 0:
        _fail(f"{field} must be positive and finite")
    if parsed > Decimal(str(_MAX_SAFE_MAGNITUDE)):
        _fail(f"{field} exceeds the numerically safe magnitude")
    if parsed < Decimal(str(_MIN_SAFE_QUANTUM)):
        _fail(f"{field} is below the binary64 moment-safety floor")
    return text


def _array(value: object, field: str) -> list[object]:
    if not isinstance(value, list):
        _fail(f"{field} must be an array")
    return cast(list[object], value)


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        _fail(f"{field} must be a boolean")
    return value


def _enum[T](enum_type: type[T], value: object, field: str) -> T:
    text = _string(value, field)
    try:
        return enum_type(text)  # type: ignore[call-arg]
    except ValueError as error:
        raise CrossWorldCouplingError(f"{field} is unsupported") from error


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        _fail(f"{field} must be a finite real number")
    try:
        normalized = float(value)
    except OverflowError as error:
        raise CrossWorldCouplingError(
            f"{field} is outside the supported range"
        ) from error
    if not math.isfinite(normalized):
        _fail(f"{field} must be finite")
    if abs(normalized) > _MAX_SAFE_MAGNITUDE:
        _fail(f"{field} exceeds the numerically safe magnitude")
    if normalized == 0.0:
        return 0.0
    return normalized


def _probability(value: object, field: str) -> float:
    normalized = _finite(value, field)
    if not 0.0 <= normalized <= 1.0:
        _fail(f"{field} must be in [0, 1]")
    return normalized


class CrossWorldSamplingUnit(StrEnum):
    """Closed set of attempted sampling units.

    The real-unit member is intentionally present so attempts to cross the
    current evidence boundary fail with a stable, explicit error.
    """

    SYNTHETIC_MODEL_DRAW = "synthetic_model_draw"
    FROZEN_REAL_UNIT = "frozen_real_unit"


class BenefitDirection(StrEnum):
    """Frozen direction for the strict benefit event."""

    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


class JointCouplingConstraint(StrEnum):
    """Admissible class of cross-world couplings."""

    EMPIRICAL_MARGINALS_ONLY = "empirical_marginals_only"
    IDENTICAL_POLICY_IDENTITY_ONLY = "identical_policy_identity_only"


class CouplingKind(StrEnum):
    """Implemented cross-world coupling constructions."""

    CALLER_DECLARED_NATIVE_ALIGNMENT = "caller_declared_native_event_key_alignment"
    MARGINAL_INDEPENDENCE_REFERENCE = "marginal_independence_reference"


class CohortInterpretation(StrEnum):
    """Non-promotable interpretation of the synthetic cohort label."""

    SYNTHETIC_ANALYTICAL_COHORT = "synthetic_analytical_cohort"
    LATENT_MODEL_STRATEGY_GROUP = "latent_model_strategy_group"


@dataclass(frozen=True, slots=True)
class CrossWorldScientificCell:
    """Scientific identity that both marginal samples must represent.

    The hashes bind the diagnostic to one policy comparison and one paired
    execution lineage.  They do not upgrade the caller-authored synthetic
    source to independent evidence.
    """

    study_protocol_sha256: str
    data_manifest_sha256: str
    baseline_policy_sha256: str
    candidate_policy_sha256: str
    model_config_sha256: str
    model_checkpoint_sha256: str
    posterior_draw_manifest_sha256: str
    paired_execution_receipt_sha256: str
    pair_integrity_audit_sha256: str
    baseline_source_artifact_sha256: str
    candidate_source_artifact_sha256: str
    cohort_definition_sha256: str
    metric_definition_sha256: str
    outcome_id: str
    outcome_unit: str
    value_quantum: str
    horizon_id: str
    benefit_direction: BenefitDirection
    estimand_id: Literal["candidate_minus_baseline"] = "candidate_minus_baseline"

    def __post_init__(self) -> None:
        for field in (
            "study_protocol_sha256",
            "data_manifest_sha256",
            "baseline_policy_sha256",
            "candidate_policy_sha256",
            "model_config_sha256",
            "model_checkpoint_sha256",
            "posterior_draw_manifest_sha256",
            "paired_execution_receipt_sha256",
            "pair_integrity_audit_sha256",
            "baseline_source_artifact_sha256",
            "candidate_source_artifact_sha256",
            "cohort_definition_sha256",
            "metric_definition_sha256",
        ):
            _sha256(getattr(self, field), field)
        for field in ("outcome_id", "outcome_unit", "horizon_id"):
            _identifier(getattr(self, field), field)
        _positive_decimal(self.value_quantum, "value_quantum")
        if not isinstance(self.benefit_direction, BenefitDirection):
            _fail("benefit_direction must be a BenefitDirection")
        if self.estimand_id != "candidate_minus_baseline":
            _fail("cross-world estimand must be candidate_minus_baseline")

    def to_mapping(self) -> dict[str, object]:
        """Return the complete scientific cell."""
        return {
            field: (
                getattr(self, field).value
                if field == "benefit_direction"
                else getattr(self, field)
            )
            for field in self.__dataclass_fields__
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        """Parse an exact scientific-cell mapping."""
        mapping = _mapping(value, "scientific cell")
        _fields(mapping, frozenset(cls.__dataclass_fields__), "scientific cell")
        return cls(
            study_protocol_sha256=_sha256(
                mapping["study_protocol_sha256"], "study_protocol_sha256"
            ),
            data_manifest_sha256=_sha256(
                mapping["data_manifest_sha256"], "data_manifest_sha256"
            ),
            baseline_policy_sha256=_sha256(
                mapping["baseline_policy_sha256"], "baseline_policy_sha256"
            ),
            candidate_policy_sha256=_sha256(
                mapping["candidate_policy_sha256"], "candidate_policy_sha256"
            ),
            model_config_sha256=_sha256(
                mapping["model_config_sha256"], "model_config_sha256"
            ),
            model_checkpoint_sha256=_sha256(
                mapping["model_checkpoint_sha256"], "model_checkpoint_sha256"
            ),
            posterior_draw_manifest_sha256=_sha256(
                mapping["posterior_draw_manifest_sha256"],
                "posterior_draw_manifest_sha256",
            ),
            paired_execution_receipt_sha256=_sha256(
                mapping["paired_execution_receipt_sha256"],
                "paired_execution_receipt_sha256",
            ),
            pair_integrity_audit_sha256=_sha256(
                mapping["pair_integrity_audit_sha256"],
                "pair_integrity_audit_sha256",
            ),
            baseline_source_artifact_sha256=_sha256(
                mapping["baseline_source_artifact_sha256"],
                "baseline_source_artifact_sha256",
            ),
            candidate_source_artifact_sha256=_sha256(
                mapping["candidate_source_artifact_sha256"],
                "candidate_source_artifact_sha256",
            ),
            cohort_definition_sha256=_sha256(
                mapping["cohort_definition_sha256"], "cohort_definition_sha256"
            ),
            metric_definition_sha256=_sha256(
                mapping["metric_definition_sha256"], "metric_definition_sha256"
            ),
            outcome_id=_identifier(mapping["outcome_id"], "outcome_id"),
            outcome_unit=_identifier(mapping["outcome_unit"], "outcome_unit"),
            value_quantum=_positive_decimal(mapping["value_quantum"], "value_quantum"),
            horizon_id=_identifier(mapping["horizon_id"], "horizon_id"),
            benefit_direction=_enum(
                BenefitDirection,
                mapping["benefit_direction"],
                "benefit_direction",
            ),
            estimand_id=cast(
                Literal["candidate_minus_baseline"],
                _string(mapping["estimand_id"], "estimand_id"),
            ),
        )


@dataclass(frozen=True, slots=True)
class CrossWorldMarginalObservation:
    """One synthetic marginal outcome bound to an event-keyed draw identity."""

    draw_id: str
    event_key: SemanticEventKey
    value_units: int

    def __post_init__(self) -> None:
        _identifier(self.draw_id, "draw_id")
        if not isinstance(self.event_key, SemanticEventKey):
            _fail("event_key must be a SemanticEventKey")
        _integer(self.value_units, "value_units")

    def to_mapping(self) -> dict[str, object]:
        """Return the complete event-addressed observation."""
        return {
            "draw_id": self.draw_id,
            "event_key": self.event_key.to_mapping(),
            "value_units": self.value_units,
        }


@dataclass(frozen=True, slots=True)
class CrossWorldMarginalSample:
    """One equal-weight finite empirical marginal for a policy world."""

    role: WorldRole
    observations: tuple[CrossWorldMarginalObservation, ...]
    source_artifact_sha256: str
    synthetic_only: Literal[True] = True
    schema_version: str = CROSS_WORLD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.role, WorldRole):
            _fail("role must be a WorldRole")
        if self.synthetic_only is not True:
            _fail("cross-world marginal samples must remain synthetic-only")
        if self.schema_version != CROSS_WORLD_SCHEMA_VERSION:
            _fail("unsupported cross-world marginal schema")
        _sha256(self.source_artifact_sha256, "source_artifact_sha256")
        if not isinstance(self.observations, tuple) or len(self.observations) < 2:
            _fail("observations must contain at least two immutable draws")
        if len(self.observations) > _MAX_SAMPLE_SIZE:
            _fail("observations exceed the supported sample-size limit")
        if not all(
            isinstance(item, CrossWorldMarginalObservation)
            for item in self.observations
        ):
            _fail("observations must contain CrossWorldMarginalObservation items")
        ordered = tuple(
            sorted(self.observations, key=lambda item: item.event_key.value)
        )
        keys = tuple(item.event_key.value for item in ordered)
        draws = tuple(item.draw_id for item in ordered)
        if len(set(keys)) != len(keys):
            _fail("event keys must be unique within a marginal sample")
        if len(set(draws)) != len(draws):
            _fail("draw IDs must be unique within a marginal sample")
        object.__setattr__(self, "observations", ordered)

    def to_mapping(self) -> dict[str, object]:
        """Return the canonical finite empirical sample."""
        return {
            "observations": [item.to_mapping() for item in self.observations],
            "role": self.role.value,
            "schema_version": self.schema_version,
            "source_artifact_sha256": self.source_artifact_sha256,
            "synthetic_only": self.synthetic_only,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        """Parse an exact marginal sample mapping."""
        mapping = _mapping(value, "marginal sample")
        _fields(mapping, frozenset(cls.__dataclass_fields__), "marginal sample")
        observations: list[CrossWorldMarginalObservation] = []
        for raw in _array(mapping["observations"], "observations"):
            item = _mapping(raw, "marginal observation")
            _fields(
                item,
                frozenset(CrossWorldMarginalObservation.__dataclass_fields__),
                "marginal observation",
            )
            try:
                event_key = SemanticEventKey.from_mapping(
                    _mapping(item["event_key"], "event_key")
                )
            except ValueError as error:
                raise CrossWorldCouplingError("event_key is invalid") from error
            observations.append(
                CrossWorldMarginalObservation(
                    draw_id=_identifier(item["draw_id"], "draw_id"),
                    event_key=event_key,
                    value_units=_integer(item["value_units"], "value_units"),
                )
            )
        return cls(
            role=_enum(WorldRole, mapping["role"], "role"),
            observations=tuple(observations),
            source_artifact_sha256=_sha256(
                mapping["source_artifact_sha256"], "source_artifact_sha256"
            ),
            synthetic_only=cast(
                Literal[True], _boolean(mapping["synthetic_only"], "synthetic_only")
            ),
            schema_version=_string(mapping["schema_version"], "schema_version"),
        )

    @property
    def sha256(self) -> str:
        """Return the sample commitment."""
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))


@dataclass(frozen=True, slots=True)
class CrossWorldCouplingInput:
    """Two aligned synthetic marginals and their frozen cohort semantics."""

    baseline: CrossWorldMarginalSample
    candidate: CrossWorldMarginalSample
    cohort: FrozenCohort
    scientific_cell: CrossWorldScientificCell
    sampling_unit: CrossWorldSamplingUnit = CrossWorldSamplingUnit.SYNTHETIC_MODEL_DRAW
    schema_version: str = CROSS_WORLD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.baseline, CrossWorldMarginalSample) or not isinstance(
            self.candidate, CrossWorldMarginalSample
        ):
            _fail("baseline and candidate must be typed marginal samples")
        if self.baseline.role is not WorldRole.BASELINE:
            _fail("baseline sample must have the baseline role")
        if self.candidate.role is not WorldRole.CANDIDATE:
            _fail("candidate sample must have the candidate role")
        if not isinstance(self.sampling_unit, CrossWorldSamplingUnit):
            _fail("sampling_unit must be a CrossWorldSamplingUnit")
        if self.sampling_unit is CrossWorldSamplingUnit.FROZEN_REAL_UNIT:
            _fail(
                "FROZEN_REAL_UNIT is unsupported without an independent typed "
                "identity and temporal-lineage authority"
            )
        if not isinstance(self.cohort, FrozenCohort):
            _fail("cohort must be a FrozenCohort")
        if not isinstance(self.scientific_cell, CrossWorldScientificCell):
            _fail("scientific_cell must be a CrossWorldScientificCell")
        if self.schema_version != CROSS_WORLD_SCHEMA_VERSION:
            _fail("unsupported cross-world input schema")
        if (
            self.scientific_cell.cohort_definition_sha256
            != self.cohort.definition_sha256
        ):
            _fail("scientific cell does not bind the frozen cohort definition")
        if (
            self.scientific_cell.baseline_source_artifact_sha256
            != self.baseline.source_artifact_sha256
            or self.scientific_cell.candidate_source_artifact_sha256
            != self.candidate.source_artifact_sha256
        ):
            _fail("scientific cell does not bind both marginal source artifacts")
        quantum = Decimal(self.scientific_cell.value_quantum)
        for sample in (self.baseline, self.candidate):
            for observation in sample.observations:
                physical_value = Decimal(observation.value_units) * quantum
                try:
                    _finite(float(physical_value), "quantized outcome value")
                except (OverflowError, ValueError) as error:
                    raise CrossWorldCouplingError(
                        "quantized outcome value exceeds the supported range"
                    ) from error
        if self.cohort.basis is CohortBasis.POST_TREATMENT_DESCRIPTIVE:
            _fail("post-treatment descriptive cohorts are not admissible")

        baseline_by_key = {
            item.event_key.value: item for item in self.baseline.observations
        }
        candidate_by_key = {
            item.event_key.value: item for item in self.candidate.observations
        }
        if baseline_by_key.keys() != candidate_by_key.keys():
            _fail("baseline and candidate must have identical event-key sets")
        for event_key, baseline in baseline_by_key.items():
            candidate = candidate_by_key[event_key]
            if baseline.event_key != candidate.event_key:
                _fail("paired semantic event identities must match exactly")
            if baseline.draw_id != candidate.draw_id:
                _fail("paired event keys must bind the same draw ID")

    def to_mapping(self) -> dict[str, object]:
        """Return the complete synthetic input commitment."""
        return {
            "baseline": self.baseline.to_mapping(),
            "candidate": self.candidate.to_mapping(),
            "cohort": self.cohort.to_mapping(),
            "sampling_unit": self.sampling_unit.value,
            "schema_version": self.schema_version,
            "scientific_cell": self.scientific_cell.to_mapping(),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        """Parse an exact input mapping and revalidate cross-world identity."""
        mapping = _mapping(value, "cross-world input")
        _fields(mapping, frozenset(cls.__dataclass_fields__), "cross-world input")
        try:
            cohort = FrozenCohort.from_mapping(_mapping(mapping["cohort"], "cohort"))
        except ValueError as error:
            raise CrossWorldCouplingError("cohort is invalid") from error
        return cls(
            baseline=CrossWorldMarginalSample.from_mapping(
                _mapping(mapping["baseline"], "baseline")
            ),
            candidate=CrossWorldMarginalSample.from_mapping(
                _mapping(mapping["candidate"], "candidate")
            ),
            cohort=cohort,
            scientific_cell=CrossWorldScientificCell.from_mapping(
                _mapping(mapping["scientific_cell"], "scientific_cell")
            ),
            sampling_unit=_enum(
                CrossWorldSamplingUnit,
                mapping["sampling_unit"],
                "sampling_unit",
            ),
            schema_version=_string(mapping["schema_version"], "schema_version"),
        )

    @property
    def sha256(self) -> str:
        """Return the complete input commitment."""
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))


@dataclass(frozen=True, slots=True)
class CrossWorldCouplingConfig:
    """Caller-declared immutable sensitivity grid and structural constraint."""

    benefit_direction: BenefitDirection
    threshold_unit: str
    analysis_plan_sha256: str
    minimum_improvement_units: int = 0
    native_weights: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    joint_constraint: JointCouplingConstraint = (
        JointCouplingConstraint.EMPIRICAL_MARGINALS_ONLY
    )
    schema_version: str = CROSS_WORLD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.benefit_direction, BenefitDirection):
            _fail("benefit_direction must be a BenefitDirection")
        if not isinstance(self.joint_constraint, JointCouplingConstraint):
            _fail("joint_constraint must be a JointCouplingConstraint")
        _identifier(self.threshold_unit, "threshold_unit")
        _sha256(self.analysis_plan_sha256, "analysis_plan_sha256")
        if self.schema_version != CROSS_WORLD_SCHEMA_VERSION:
            _fail("unsupported cross-world config schema")
        minimum_improvement_units = _integer(
            self.minimum_improvement_units,
            "minimum_improvement_units",
        )
        if minimum_improvement_units < 0:
            _fail("minimum_improvement_units must be non-negative")
        if not isinstance(self.native_weights, tuple) or not self.native_weights:
            _fail("native_weights must be a non-empty immutable tuple")
        weights = tuple(
            _probability(item, "native weight") for item in self.native_weights
        )
        if tuple(sorted(set(weights))) != weights:
            _fail("native_weights must be unique and strictly increasing")
        if self.joint_constraint is JointCouplingConstraint.EMPIRICAL_MARGINALS_ONLY:
            if weights[0] != 0.0 or weights[-1] != 1.0:
                _fail("the unrestricted path must include product and native endpoints")
        elif weights != (1.0,):
            _fail(
                "identical-policy identity analysis forbids unrestricted marginal "
                "rearrangements; native_weights must be (1.0,)"
            )
        object.__setattr__(self, "native_weights", weights)

    def to_mapping(self) -> dict[str, object]:
        """Return the frozen analysis configuration."""
        return {
            "benefit_direction": self.benefit_direction.value,
            "analysis_plan_sha256": self.analysis_plan_sha256,
            "joint_constraint": self.joint_constraint.value,
            "minimum_improvement_units": self.minimum_improvement_units,
            "native_weights": list(self.native_weights),
            "schema_version": self.schema_version,
            "threshold_unit": self.threshold_unit,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        """Parse an exact declared sensitivity configuration."""
        mapping = _mapping(value, "cross-world config")
        _fields(mapping, frozenset(cls.__dataclass_fields__), "cross-world config")
        return cls(
            benefit_direction=_enum(
                BenefitDirection,
                mapping["benefit_direction"],
                "benefit_direction",
            ),
            threshold_unit=_identifier(mapping["threshold_unit"], "threshold_unit"),
            analysis_plan_sha256=_sha256(
                mapping["analysis_plan_sha256"], "analysis_plan_sha256"
            ),
            minimum_improvement_units=_integer(
                mapping["minimum_improvement_units"], "minimum_improvement_units"
            ),
            native_weights=tuple(
                _finite(item, "native weight")
                for item in _array(mapping["native_weights"], "native_weights")
            ),
            joint_constraint=_enum(
                JointCouplingConstraint,
                mapping["joint_constraint"],
                "joint_constraint",
            ),
            schema_version=_string(mapping["schema_version"], "schema_version"),
        )

    @property
    def sha256(self) -> str:
        """Return the configuration commitment."""
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))


@dataclass(frozen=True, slots=True)
class CouplingMetrics:
    """Equal-weight effect summaries under one explicit joint law."""

    coupling: CouplingKind
    mean_effect: float
    strict_benefit_probability: float
    effect_population_variance: float
    weighted_pair_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.coupling, CouplingKind):
            _fail("coupling must be a CouplingKind")
        object.__setattr__(
            self, "mean_effect", _finite(self.mean_effect, "mean_effect")
        )
        object.__setattr__(
            self,
            "strict_benefit_probability",
            _probability(self.strict_benefit_probability, "strict_benefit_probability"),
        )
        variance = _finite(
            self.effect_population_variance, "effect_population_variance"
        )
        if variance < 0.0:
            _fail("effect_population_variance must be non-negative")
        object.__setattr__(self, "effect_population_variance", variance)
        if (
            isinstance(self.weighted_pair_count, bool)
            or not isinstance(self.weighted_pair_count, int)
            or self.weighted_pair_count <= 0
        ):
            _fail("weighted_pair_count must be a positive integer")

    def to_mapping(self) -> dict[str, object]:
        """Return one coupling summary."""
        return {
            "coupling": self.coupling.value,
            "effect_population_variance": self.effect_population_variance,
            "mean_effect": self.mean_effect,
            "strict_benefit_probability": self.strict_benefit_probability,
            "weighted_pair_count": self.weighted_pair_count,
        }


@dataclass(frozen=True, slots=True)
class CouplingPathPoint:
    """A mixture of the native joint law and empirical product law."""

    native_weight: float
    mean_effect: float
    strict_benefit_probability: float
    effect_population_variance: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "native_weight", _probability(self.native_weight, "native_weight")
        )
        object.__setattr__(
            self, "mean_effect", _finite(self.mean_effect, "mean_effect")
        )
        object.__setattr__(
            self,
            "strict_benefit_probability",
            _probability(self.strict_benefit_probability, "strict_benefit_probability"),
        )
        variance = _finite(
            self.effect_population_variance, "effect_population_variance"
        )
        if variance < 0.0:
            _fail("effect_population_variance must be non-negative")
        object.__setattr__(self, "effect_population_variance", variance)

    def to_mapping(self) -> dict[str, object]:
        """Return one frozen path point."""
        return {
            "effect_population_variance": self.effect_population_variance,
            "mean_effect": self.mean_effect,
            "native_weight": self.native_weight,
            "strict_benefit_probability": self.strict_benefit_probability,
        }


@dataclass(frozen=True, slots=True)
class SharpCouplingBounds:
    """Sharp extrema over the declared equal-weight coupling class."""

    coupling_class: JointCouplingConstraint
    strict_benefit_probability_lower: float
    strict_benefit_probability_upper: float
    effect_population_variance_lower: float
    effect_population_variance_upper: float
    componentwise_sharp_only: Literal[True] = True
    jointly_sharp_rectangle: Literal[False] = False
    single_threshold_pointwise: Literal[True] = True
    sampling_uncertainty_included: Literal[False] = False

    def __post_init__(self) -> None:
        if not isinstance(self.coupling_class, JointCouplingConstraint):
            _fail("coupling_class must be a JointCouplingConstraint")
        benefit_lower = _probability(
            self.strict_benefit_probability_lower,
            "strict_benefit_probability_lower",
        )
        benefit_upper = _probability(
            self.strict_benefit_probability_upper,
            "strict_benefit_probability_upper",
        )
        if benefit_lower > benefit_upper:
            _fail("strict benefit bounds must be ordered")
        variance_lower = _finite(
            self.effect_population_variance_lower,
            "effect_population_variance_lower",
        )
        variance_upper = _finite(
            self.effect_population_variance_upper,
            "effect_population_variance_upper",
        )
        if variance_lower < 0.0 or variance_lower > variance_upper:
            _fail("effect variance bounds must be non-negative and ordered")
        if (
            self.componentwise_sharp_only is not True
            or self.jointly_sharp_rectangle is not False
            or self.single_threshold_pointwise is not True
            or self.sampling_uncertainty_included is not False
        ):
            _fail("sharp-bound interpretation flags cannot be promoted")

    def to_mapping(self) -> dict[str, object]:
        """Return exact bound semantics and values."""
        return {
            "coupling_class": self.coupling_class.value,
            "componentwise_sharp_only": self.componentwise_sharp_only,
            "effect_population_variance_lower": (self.effect_population_variance_lower),
            "effect_population_variance_upper": (self.effect_population_variance_upper),
            "jointly_sharp_rectangle": self.jointly_sharp_rectangle,
            "sampling_uncertainty_included": self.sampling_uncertainty_included,
            "single_threshold_pointwise": self.single_threshold_pointwise,
            "strict_benefit_probability_lower": (self.strict_benefit_probability_lower),
            "strict_benefit_probability_upper": (self.strict_benefit_probability_upper),
        }


@dataclass(frozen=True, slots=True)
class _DerivedCrossWorldAnalysis:
    native: CouplingMetrics
    marginal_independence_reference: CouplingMetrics | None
    path: tuple[CouplingPathPoint, ...]
    sharp_bounds: SharpCouplingBounds


def _decimal_to_float(value: Decimal, field: str) -> float:
    try:
        normalized = float(value)
    except (OverflowError, ValueError) as error:
        raise CrossWorldCouplingError(f"{field} exceeds the supported range") from error
    if not math.isfinite(normalized) or abs(normalized) > _MAX_SAFE_MAGNITUDE:
        _fail(f"{field} exceeds the supported range")
    if value != 0 and normalized == 0.0:
        _fail(f"{field} underflows the supported binary64 domain")
    return 0.0 if normalized == 0.0 else normalized


def _physical_value(value_units: int, quantum: Decimal) -> float:
    return _decimal_to_float(
        Decimal(value_units) * quantum,
        "quantized outcome value",
    )


def _scaled_mean(value_units: tuple[int, ...], quantum: Decimal) -> float:
    return _decimal_to_float(
        Decimal(sum(value_units)) * quantum / Decimal(len(value_units)),
        "quantized mean",
    )


def _variance_numerator(value_units: tuple[int, ...]) -> int:
    sample_size = len(value_units)
    total = sum(value_units)
    numerator = (
        sample_size * sum(value * value for value in value_units) - total * total
    )
    if numerator < 0:
        _fail("exact population variance numerator cannot be negative")
    return numerator


def _scaled_population_variance(
    value_units: tuple[int, ...],
    quantum: Decimal,
) -> float:
    numerator = _variance_numerator(value_units)
    if numerator == 0:
        return 0.0
    sample_size = len(value_units)
    return _decimal_to_float(
        Decimal(numerator) * quantum * quantum / Decimal(sample_size * sample_size),
        "quantized effect population variance",
    )


def _scaled_independence_variance(
    baseline: tuple[int, ...],
    candidate: tuple[int, ...],
    quantum: Decimal,
) -> float:
    sample_size = len(baseline)
    numerator = _variance_numerator(baseline) + _variance_numerator(candidate)
    if numerator == 0:
        return 0.0
    return _decimal_to_float(
        Decimal(numerator) * quantum * quantum / Decimal(sample_size * sample_size),
        "marginal independence effect population variance",
    )


def _is_benefit(
    effect_units: int,
    direction: BenefitDirection,
    minimum_improvement_units: int,
) -> bool:
    direction_adjusted_effect = (
        effect_units
        if direction is BenefitDirection.HIGHER_IS_BETTER
        else -effect_units
    )
    return direction_adjusted_effect > minimum_improvement_units


def _metrics(
    kind: CouplingKind,
    effect_units: tuple[int, ...],
    quantum: Decimal,
    direction: BenefitDirection,
    minimum_improvement_units: int,
) -> CouplingMetrics:
    return CouplingMetrics(
        coupling=kind,
        mean_effect=_scaled_mean(effect_units, quantum),
        strict_benefit_probability=(
            sum(
                _is_benefit(value, direction, minimum_improvement_units)
                for value in effect_units
            )
            / len(effect_units)
        ),
        effect_population_variance=_scaled_population_variance(effect_units, quantum),
        weighted_pair_count=len(effect_units),
    )


def _maximum_benefit_count(
    baseline: tuple[int, ...],
    candidate: tuple[int, ...],
    direction: BenefitDirection,
    minimum_improvement_units: int,
) -> int:
    """Return the threshold-graph matching maximum in O(n log n)."""
    lower_side, higher_side = (
        (tuple(sorted(baseline)), tuple(sorted(candidate)))
        if direction is BenefitDirection.HIGHER_IS_BETTER
        else (tuple(sorted(candidate)), tuple(sorted(baseline)))
    )
    lower_index = 0
    matches = 0
    for higher_value in higher_side:
        if lower_index == len(lower_side):
            break
        if higher_value > lower_side[lower_index] + minimum_improvement_units:
            lower_index += 1
            matches += 1
    return matches


def _maximum_nonbenefit_count(
    baseline: tuple[int, ...],
    candidate: tuple[int, ...],
    direction: BenefitDirection,
    minimum_improvement_units: int,
) -> int:
    """Return the complementary threshold-graph matching maximum."""
    lower_side, higher_side = (
        (tuple(sorted(candidate)), tuple(sorted(baseline)))
        if direction is BenefitDirection.HIGHER_IS_BETTER
        else (tuple(sorted(baseline)), tuple(sorted(candidate)))
    )
    lower_index = 0
    matches = 0
    for higher_value in higher_side:
        if lower_index == len(lower_side):
            break
        if lower_side[lower_index] <= higher_value + minimum_improvement_units:
            lower_index += 1
            matches += 1
    return matches


def _marginal_independence_metrics(
    baseline: tuple[int, ...],
    candidate: tuple[int, ...],
    quantum: Decimal,
    direction: BenefitDirection,
    minimum_improvement_units: int,
) -> CouplingMetrics:
    """Compute the empirical product reference without materializing n² effects."""
    candidate_sorted = tuple(sorted(candidate))
    if direction is BenefitDirection.HIGHER_IS_BETTER:
        benefit_count = sum(
            len(candidate_sorted)
            - bisect_right(
                candidate_sorted,
                baseline_value + minimum_improvement_units,
            )
            for baseline_value in baseline
        )
    else:
        benefit_count = sum(
            bisect_left(
                candidate_sorted,
                baseline_value - minimum_improvement_units,
            )
            for baseline_value in baseline
        )
    pair_count = len(baseline) * len(candidate)
    paired_effect_units = tuple(
        candidate_value - baseline_value
        for baseline_value, candidate_value in zip(baseline, candidate, strict=True)
    )
    mean_effect = _scaled_mean(paired_effect_units, quantum)
    variance = _scaled_independence_variance(baseline, candidate, quantum)
    return CouplingMetrics(
        coupling=CouplingKind.MARGINAL_INDEPENDENCE_REFERENCE,
        mean_effect=mean_effect,
        strict_benefit_probability=benefit_count / pair_count,
        effect_population_variance=variance,
        weighted_pair_count=pair_count,
    )


def _sharp_bounds(
    baseline: tuple[int, ...],
    candidate: tuple[int, ...],
    quantum: Decimal,
    direction: BenefitDirection,
    minimum_improvement_units: int,
) -> SharpCouplingBounds:
    maximum_benefits = _maximum_benefit_count(
        baseline,
        candidate,
        direction,
        minimum_improvement_units,
    )
    maximum_nonbenefits = _maximum_nonbenefit_count(
        baseline,
        candidate,
        direction,
        minimum_improvement_units,
    )
    sample_size = len(baseline)

    baseline_sorted = tuple(sorted(baseline))
    candidate_sorted = tuple(sorted(candidate))
    comonotone_effect_units = tuple(
        candidate_value - baseline_value
        for baseline_value, candidate_value in zip(
            baseline_sorted, candidate_sorted, strict=True
        )
    )
    countermonotone_effect_units = tuple(
        candidate_value - baseline_value
        for baseline_value, candidate_value in zip(
            baseline_sorted, reversed(candidate_sorted), strict=True
        )
    )
    variance_extrema = (
        _scaled_population_variance(comonotone_effect_units, quantum),
        _scaled_population_variance(countermonotone_effect_units, quantum),
    )
    return SharpCouplingBounds(
        coupling_class=JointCouplingConstraint.EMPIRICAL_MARGINALS_ONLY,
        strict_benefit_probability_lower=(sample_size - maximum_nonbenefits)
        / sample_size,
        strict_benefit_probability_upper=maximum_benefits / sample_size,
        effect_population_variance_lower=min(variance_extrema),
        effect_population_variance_upper=max(variance_extrema),
    )


def _aligned_values(
    analysis_input: CrossWorldCouplingInput,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    baseline_by_key = {
        item.event_key.value: item.value_units
        for item in analysis_input.baseline.observations
    }
    candidate_by_key = {
        item.event_key.value: item.value_units
        for item in analysis_input.candidate.observations
    }
    keys = tuple(sorted(baseline_by_key))
    return (
        tuple(baseline_by_key[key] for key in keys),
        tuple(candidate_by_key[key] for key in keys),
    )


def _derive_analysis(
    analysis_input: CrossWorldCouplingInput,
    config: CrossWorldCouplingConfig,
) -> _DerivedCrossWorldAnalysis:
    baseline, candidate = _aligned_values(analysis_input)
    quantum = Decimal(analysis_input.scientific_cell.value_quantum)
    if config.threshold_unit != analysis_input.scientific_cell.outcome_unit:
        _fail("threshold_unit does not match the scientific outcome unit")
    if config.benefit_direction is not analysis_input.scientific_cell.benefit_direction:
        _fail("benefit_direction does not match the frozen metric definition")
    native_effect_units = tuple(
        candidate_value - baseline_value
        for baseline_value, candidate_value in zip(baseline, candidate, strict=True)
    )
    native = _metrics(
        CouplingKind.CALLER_DECLARED_NATIVE_ALIGNMENT,
        native_effect_units,
        quantum,
        config.benefit_direction,
        config.minimum_improvement_units,
    )

    if (
        config.joint_constraint
        is JointCouplingConstraint.IDENTICAL_POLICY_IDENTITY_ONLY
    ):
        cell = analysis_input.scientific_cell
        if cell.baseline_policy_sha256 != cell.candidate_policy_sha256:
            _fail(
                "identity-only analysis requires identical baseline and candidate "
                "policy hashes"
            )
        if any(effect != 0 for effect in native_effect_units):
            _fail("identical-policy outcomes must match at every event key")
        zero_bounds = SharpCouplingBounds(
            coupling_class=JointCouplingConstraint.IDENTICAL_POLICY_IDENTITY_ONLY,
            strict_benefit_probability_lower=0.0,
            strict_benefit_probability_upper=0.0,
            effect_population_variance_lower=0.0,
            effect_population_variance_upper=0.0,
        )
        return _DerivedCrossWorldAnalysis(
            native=native,
            marginal_independence_reference=None,
            path=(
                CouplingPathPoint(
                    native_weight=1.0,
                    mean_effect=0.0,
                    strict_benefit_probability=0.0,
                    effect_population_variance=0.0,
                ),
            ),
            sharp_bounds=zero_bounds,
        )

    independence = _marginal_independence_metrics(
        baseline,
        candidate,
        quantum,
        config.benefit_direction,
        config.minimum_improvement_units,
    )
    path: list[CouplingPathPoint] = []
    for native_weight in config.native_weights:
        product_weight = 1.0 - native_weight
        if native.mean_effect != independence.mean_effect:
            _fail("fixed marginals must imply one invariant mean effect")
        mean_effect = native.mean_effect
        variance = _finite(
            native_weight * native.effect_population_variance
            + product_weight * independence.effect_population_variance,
            "coupling-path effect population variance",
        )
        path.append(
            CouplingPathPoint(
                native_weight=native_weight,
                mean_effect=mean_effect,
                strict_benefit_probability=(
                    native_weight * native.strict_benefit_probability
                    + product_weight * independence.strict_benefit_probability
                ),
                effect_population_variance=variance,
            )
        )
    return _DerivedCrossWorldAnalysis(
        native=native,
        marginal_independence_reference=independence,
        path=tuple(path),
        sharp_bounds=_sharp_bounds(
            baseline,
            candidate,
            quantum,
            config.benefit_direction,
            config.minimum_improvement_units,
        ),
    )


@dataclass(frozen=True, slots=True)
class CrossWorldCouplingResult:
    """Auditable synthetic coupling analysis with non-promotable claim flags."""

    analysis_input: CrossWorldCouplingInput
    config: CrossWorldCouplingConfig
    native: CouplingMetrics
    marginal_independence_reference: CouplingMetrics | None
    path: tuple[CouplingPathPoint, ...]
    sharp_bounds: SharpCouplingBounds
    cohort_interpretation: CohortInterpretation
    claim_scope: Literal["synthetic_cross_world_coupling_sensitivity_only"] = (
        CROSS_WORLD_CLAIM_SCOPE
    )
    source_authority: Literal["unverified_caller_authored_synthetic_marginals"] = (
        CROSS_WORLD_SOURCE_AUTHORITY
    )
    joint_law_identified_by_marginals: Literal[False] = False
    real_causal_claim: Literal[False] = False
    observed_investor_identity: Literal[False] = False
    externally_preregistered: Literal[False] = False
    source_artifacts_independently_verified: Literal[False] = False
    sensitivity_path_exhausts_coupling_class: Literal[False] = False
    formal_evidence: Literal[False] = False
    synthetic_only: Literal[True] = True
    schema_version: str = CROSS_WORLD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.analysis_input, CrossWorldCouplingInput):
            _fail("result requires typed cross-world input")
        if not isinstance(self.config, CrossWorldCouplingConfig):
            _fail("result requires typed coupling config")
        if self.schema_version != CROSS_WORLD_SCHEMA_VERSION:
            _fail("unsupported cross-world coupling schema")
        if self.claim_scope != CROSS_WORLD_CLAIM_SCOPE:
            _fail("cross-world claim scope cannot be caller-promoted")
        if self.source_authority != CROSS_WORLD_SOURCE_AUTHORITY:
            _fail("cross-world source authority cannot be caller-promoted")
        if self.joint_law_identified_by_marginals is not False:
            _fail("marginal validation cannot identify a cross-world joint law")
        if self.real_causal_claim is not False:
            _fail("synthetic coupling analysis cannot authorize a real causal claim")
        if self.observed_investor_identity is not False:
            _fail("synthetic or latent cohorts are not observed investor identities")
        if self.externally_preregistered is not False:
            _fail("caller-authored analysis plans are not external preregistrations")
        if self.source_artifacts_independently_verified is not False:
            _fail("caller-authored source hashes are not independent verification")
        if self.sensitivity_path_exhausts_coupling_class is not False:
            _fail("the native-to-independence path does not exhaust all couplings")
        if self.formal_evidence is not False or self.synthetic_only is not True:
            _fail("cross-world sensitivity must remain non-formal and synthetic-only")

        expected_interpretation = (
            CohortInterpretation.LATENT_MODEL_STRATEGY_GROUP
            if self.analysis_input.cohort.basis is CohortBasis.LATENT_STRATEGY
            else CohortInterpretation.SYNTHETIC_ANALYTICAL_COHORT
        )
        if self.cohort_interpretation is not expected_interpretation:
            _fail("cohort interpretation does not match its frozen basis")

        expected = _derive_analysis(self.analysis_input, self.config)
        if (
            self.native != expected.native
            or self.marginal_independence_reference
            != expected.marginal_independence_reference
            or self.path != expected.path
            or self.sharp_bounds != expected.sharp_bounds
        ):
            _fail("cross-world result does not reproduce from typed inputs")

    def to_mapping(self) -> dict[str, object]:
        """Return the complete analysis and its scientific boundaries."""
        return {
            "analysis_input": self.analysis_input.to_mapping(),
            "claim_scope": self.claim_scope,
            "cohort_interpretation": self.cohort_interpretation.value,
            "config": self.config.to_mapping(),
            "config_sha256": self.config.sha256,
            "effect_definition": "candidate_minus_baseline",
            "externally_preregistered": self.externally_preregistered,
            "marginal_independence_reference": (
                None
                if self.marginal_independence_reference is None
                else self.marginal_independence_reference.to_mapping()
            ),
            "formal_evidence": self.formal_evidence,
            "input_sha256": self.analysis_input.sha256,
            "joint_law_identified_by_marginals": (
                self.joint_law_identified_by_marginals
            ),
            "native": self.native.to_mapping(),
            "observed_investor_identity": self.observed_investor_identity,
            "path": [item.to_mapping() for item in self.path],
            "real_causal_claim": self.real_causal_claim,
            "schema_version": self.schema_version,
            "sensitivity_path_exhausts_coupling_class": (
                self.sensitivity_path_exhausts_coupling_class
            ),
            "sharp_bounds": self.sharp_bounds.to_mapping(),
            "source_authority": self.source_authority,
            "source_artifacts_independently_verified": (
                self.source_artifacts_independently_verified
            ),
            "strict_benefit_rule": {
                "comparison": (
                    "direction_adjusted_effect_strictly_greater_than_threshold"
                ),
                "direction": self.config.benefit_direction.value,
                "effect_definition": "candidate_minus_baseline",
                "minimum_improvement_units": (self.config.minimum_improvement_units),
                "outcome_unit": self.analysis_input.scientific_cell.outcome_unit,
                "value_quantum": self.analysis_input.scientific_cell.value_quantum,
                "threshold_ties_are_benefits": False,
            },
            "synthetic_only": self.synthetic_only,
            "variance_definition": {
                "base_outcome_unit": (self.analysis_input.scientific_cell.outcome_unit),
                "estimator": "equal_weight_empirical_population_variance",
                "unit_power": 2,
                "value_quantum": (self.analysis_input.scientific_cell.value_quantum),
                "value_quantum_power": 2,
            },
        }

    @property
    def sha256(self) -> str:
        """Return the complete result commitment."""
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        """Parse a complete result and independently rederive every statistic."""
        mapping = _mapping(value, "cross-world result")
        expected_fields = frozenset(
            {
                "analysis_input",
                "claim_scope",
                "cohort_interpretation",
                "config",
                "config_sha256",
                "effect_definition",
                "externally_preregistered",
                "formal_evidence",
                "input_sha256",
                "joint_law_identified_by_marginals",
                "marginal_independence_reference",
                "native",
                "observed_investor_identity",
                "path",
                "real_causal_claim",
                "schema_version",
                "sensitivity_path_exhausts_coupling_class",
                "sharp_bounds",
                "source_artifacts_independently_verified",
                "source_authority",
                "strict_benefit_rule",
                "synthetic_only",
                "variance_definition",
            }
        )
        _fields(mapping, expected_fields, "cross-world result")
        independence_raw = mapping["marginal_independence_reference"]
        result = cls(
            analysis_input=CrossWorldCouplingInput.from_mapping(
                _mapping(mapping["analysis_input"], "analysis_input")
            ),
            config=CrossWorldCouplingConfig.from_mapping(
                _mapping(mapping["config"], "config")
            ),
            native=_coupling_metrics_from_mapping(mapping["native"]),
            marginal_independence_reference=(
                None
                if independence_raw is None
                else _coupling_metrics_from_mapping(independence_raw)
            ),
            path=tuple(
                _path_point_from_mapping(item)
                for item in _array(mapping["path"], "path")
            ),
            sharp_bounds=_sharp_bounds_from_mapping(mapping["sharp_bounds"]),
            cohort_interpretation=_enum(
                CohortInterpretation,
                mapping["cohort_interpretation"],
                "cohort_interpretation",
            ),
            claim_scope=cast(
                Literal["synthetic_cross_world_coupling_sensitivity_only"],
                _string(mapping["claim_scope"], "claim_scope"),
            ),
            source_authority=cast(
                Literal["unverified_caller_authored_synthetic_marginals"],
                _string(mapping["source_authority"], "source_authority"),
            ),
            joint_law_identified_by_marginals=cast(
                Literal[False],
                _boolean(
                    mapping["joint_law_identified_by_marginals"],
                    "joint_law_identified_by_marginals",
                ),
            ),
            real_causal_claim=cast(
                Literal[False],
                _boolean(mapping["real_causal_claim"], "real_causal_claim"),
            ),
            observed_investor_identity=cast(
                Literal[False],
                _boolean(
                    mapping["observed_investor_identity"],
                    "observed_investor_identity",
                ),
            ),
            externally_preregistered=cast(
                Literal[False],
                _boolean(
                    mapping["externally_preregistered"],
                    "externally_preregistered",
                ),
            ),
            source_artifacts_independently_verified=cast(
                Literal[False],
                _boolean(
                    mapping["source_artifacts_independently_verified"],
                    "source_artifacts_independently_verified",
                ),
            ),
            sensitivity_path_exhausts_coupling_class=cast(
                Literal[False],
                _boolean(
                    mapping["sensitivity_path_exhausts_coupling_class"],
                    "sensitivity_path_exhausts_coupling_class",
                ),
            ),
            formal_evidence=cast(
                Literal[False],
                _boolean(mapping["formal_evidence"], "formal_evidence"),
            ),
            synthetic_only=cast(
                Literal[True],
                _boolean(mapping["synthetic_only"], "synthetic_only"),
            ),
            schema_version=_string(mapping["schema_version"], "schema_version"),
        )
        if canonical_json_bytes(result.to_mapping()) != canonical_json_bytes(mapping):
            _fail("cross-world result contains inconsistent derived fields")
        return result


def _coupling_metrics_from_mapping(value: object) -> CouplingMetrics:
    mapping = _mapping(value, "coupling metrics")
    _fields(
        mapping, frozenset(CouplingMetrics.__dataclass_fields__), "coupling metrics"
    )
    return CouplingMetrics(
        coupling=_enum(CouplingKind, mapping["coupling"], "coupling"),
        mean_effect=_finite(mapping["mean_effect"], "mean_effect"),
        strict_benefit_probability=_probability(
            mapping["strict_benefit_probability"],
            "strict_benefit_probability",
        ),
        effect_population_variance=_finite(
            mapping["effect_population_variance"],
            "effect_population_variance",
        ),
        weighted_pair_count=_integer(
            mapping["weighted_pair_count"],
            "weighted_pair_count",
        ),
    )


def _path_point_from_mapping(value: object) -> CouplingPathPoint:
    mapping = _mapping(value, "coupling path point")
    _fields(
        mapping,
        frozenset(CouplingPathPoint.__dataclass_fields__),
        "coupling path point",
    )
    return CouplingPathPoint(
        native_weight=_probability(mapping["native_weight"], "native_weight"),
        mean_effect=_finite(mapping["mean_effect"], "mean_effect"),
        strict_benefit_probability=_probability(
            mapping["strict_benefit_probability"],
            "strict_benefit_probability",
        ),
        effect_population_variance=_finite(
            mapping["effect_population_variance"],
            "effect_population_variance",
        ),
    )


def _sharp_bounds_from_mapping(value: object) -> SharpCouplingBounds:
    mapping = _mapping(value, "sharp coupling bounds")
    _fields(
        mapping,
        frozenset(SharpCouplingBounds.__dataclass_fields__),
        "sharp coupling bounds",
    )
    return SharpCouplingBounds(
        coupling_class=_enum(
            JointCouplingConstraint,
            mapping["coupling_class"],
            "coupling_class",
        ),
        strict_benefit_probability_lower=_probability(
            mapping["strict_benefit_probability_lower"],
            "strict_benefit_probability_lower",
        ),
        strict_benefit_probability_upper=_probability(
            mapping["strict_benefit_probability_upper"],
            "strict_benefit_probability_upper",
        ),
        effect_population_variance_lower=_finite(
            mapping["effect_population_variance_lower"],
            "effect_population_variance_lower",
        ),
        effect_population_variance_upper=_finite(
            mapping["effect_population_variance_upper"],
            "effect_population_variance_upper",
        ),
        componentwise_sharp_only=cast(
            Literal[True],
            _boolean(
                mapping["componentwise_sharp_only"],
                "componentwise_sharp_only",
            ),
        ),
        jointly_sharp_rectangle=cast(
            Literal[False],
            _boolean(
                mapping["jointly_sharp_rectangle"],
                "jointly_sharp_rectangle",
            ),
        ),
        single_threshold_pointwise=cast(
            Literal[True],
            _boolean(
                mapping["single_threshold_pointwise"],
                "single_threshold_pointwise",
            ),
        ),
        sampling_uncertainty_included=cast(
            Literal[False],
            _boolean(
                mapping["sampling_uncertainty_included"],
                "sampling_uncertainty_included",
            ),
        ),
    )


def _stat_fingerprint(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
    )


def load_cross_world_coupling_result(
    path: Path,
    *,
    expected_sha256: str,
) -> CrossWorldCouplingResult:
    """Load bounded canonical JSON under an external raw-byte commitment."""
    if not isinstance(path, Path):
        _fail("cross-world result path must be a Path")
    expected = _sha256(expected_sha256, "expected_sha256")
    candidate = Path(os.path.abspath(path.expanduser()))
    current = Path(candidate.anchor)
    try:
        for part in candidate.parts[1:]:
            current /= part
            if current.is_symlink():
                _fail("cross-world result path contains a symbolic link")
        descriptor = os.open(
            candidate,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
        )
    except OSError as error:
        raise CrossWorldCouplingError("cannot open cross-world result") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > _MAX_ARTIFACT_BYTES
        ):
            _fail("cross-world result is not a bounded regular file")
        chunks: list[bytes] = []
        total = 0
        while total <= _MAX_ARTIFACT_BYTES:
            chunk = os.read(
                descriptor,
                min(1024 * 1024, _MAX_ARTIFACT_BYTES + 1 - total),
            )
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        if total > _MAX_ARTIFACT_BYTES:
            _fail("cross-world result exceeds its byte limit")
        after = os.fstat(descriptor)
        content = b"".join(chunks)
    finally:
        os.close(descriptor)
    if (
        _stat_fingerprint(before) != _stat_fingerprint(after)
        or len(content) != before.st_size
    ):
        _fail("cross-world result changed while it was read")
    try:
        linked = candidate.stat(follow_symlinks=False)
    except OSError as error:
        raise CrossWorldCouplingError("cannot restat cross-world result") from error
    if not stat.S_ISREG(linked.st_mode) or _stat_fingerprint(
        linked
    ) != _stat_fingerprint(after):
        _fail("cross-world result path identity changed while it was read")
    if sha256_bytes(content) != expected:
        _fail("cross-world result differs from its external commitment")
    try:
        parsed = read_json_bytes_object(content, source=str(candidate))
    except Exception as error:
        raise CrossWorldCouplingError("cannot parse cross-world result") from error
    if canonical_json_bytes(parsed) != content:
        _fail("cross-world result must use canonical JSON encoding")
    result = CrossWorldCouplingResult.from_mapping(parsed)
    if result.sha256 != expected:
        _fail("cross-world semantic replay differs from its raw commitment")
    return result


def evaluate_cross_world_coupling(
    analysis_input: CrossWorldCouplingInput,
    config: CrossWorldCouplingConfig,
) -> CrossWorldCouplingResult:
    """Evaluate caller-declared synthetic cross-world coupling sensitivity."""
    if not isinstance(analysis_input, CrossWorldCouplingInput):
        _fail("analysis_input must be a CrossWorldCouplingInput")
    if not isinstance(config, CrossWorldCouplingConfig):
        _fail("config must be a CrossWorldCouplingConfig")
    derived = _derive_analysis(analysis_input, config)
    interpretation = (
        CohortInterpretation.LATENT_MODEL_STRATEGY_GROUP
        if analysis_input.cohort.basis is CohortBasis.LATENT_STRATEGY
        else CohortInterpretation.SYNTHETIC_ANALYTICAL_COHORT
    )
    return CrossWorldCouplingResult(
        analysis_input=analysis_input,
        config=config,
        native=derived.native,
        marginal_independence_reference=derived.marginal_independence_reference,
        path=derived.path,
        sharp_bounds=derived.sharp_bounds,
        cohort_interpretation=interpretation,
    )


__all__ = [
    "CROSS_WORLD_CLAIM_SCOPE",
    "CROSS_WORLD_SCHEMA_VERSION",
    "CROSS_WORLD_SOURCE_AUTHORITY",
    "BenefitDirection",
    "CohortInterpretation",
    "CouplingKind",
    "CouplingMetrics",
    "CouplingPathPoint",
    "CrossWorldCouplingConfig",
    "CrossWorldCouplingError",
    "CrossWorldCouplingInput",
    "CrossWorldCouplingResult",
    "CrossWorldMarginalObservation",
    "CrossWorldMarginalSample",
    "CrossWorldSamplingUnit",
    "CrossWorldScientificCell",
    "JointCouplingConstraint",
    "SharpCouplingBounds",
    "evaluate_cross_world_coupling",
    "load_cross_world_coupling_result",
]
