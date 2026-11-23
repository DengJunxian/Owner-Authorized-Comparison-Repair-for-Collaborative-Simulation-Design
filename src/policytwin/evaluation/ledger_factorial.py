"""Exact fixed-grid posterior-by-scenario decomposition for synthetic ledgers.

This module is deliberately a pure analysis boundary.  It does not execute a
simulator, establish source custody, prove external preregistration, or support
real-world causal claims.  It consumes a complete, typed ``M x R`` grid whose
posterior and scenario commitments were fixed elsewhere and rederives every
reported byte from that grid and a pre-outcome analysis specification.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from fractions import Fraction
from typing import Literal, NoReturn, Self

from policytwin.counterfactual.randomness import EventKey, deterministic_int
from policytwin.evaluation.ledger_derived import (
    ExactMetric,
    ExactMetricState,
    ExactRational,
)
from policytwin.provenance import (
    MetadataValidationError,
    canonical_json_bytes,
    read_json_bytes_object,
    sha256_bytes,
)

LEDGER_FACTORIAL_SPEC_SCHEMA_VERSION = "policytwin.ledger-factorial-spec.v1"
LEDGER_FACTORIAL_RESULT_SCHEMA_VERSION = "policytwin.ledger-factorial-result.v1"
LEDGER_FACTORIAL_BOOTSTRAP_SCHEMA_VERSION = (
    "policytwin.ledger-factorial-crossed-bootstrap.v1"
)
LEDGER_FACTORIAL_CLAIM_SCOPE = (
    "synthetic_fixed_grid_posterior_by_scenario_decomposition_only"
)

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EVENT_KEY = re.compile(r"^ptk1:[0-9a-f]{64}$")
_MAX_AXIS_COUNT = 4096
_MAX_GRID_CELLS = 1_000_000
_MAX_BOOTSTRAP_REPLICATES = 100_000
_MAX_RESULT_BYTES = 256 * 1024 * 1024


class LedgerFactorialError(RuntimeError):
    """Raised when the fixed factorial evidence contract cannot be proved."""


def _fail(message: str) -> NoReturn:
    raise LedgerFactorialError(message)


class FactorialUndefinedPolicy(StrEnum):
    """Frozen handling of undefined cells for one registered output."""

    FAIL_CLOSED = "fail_closed"
    RETAIN_STATE_TRANSITIONS = "retain_state_transitions_without_numeric_summary"


class FactorialOutputRole(StrEnum):
    """Frozen registration role; secondary and exploratory remain distinct."""

    PRIMARY = "primary"
    SECONDARY = "secondary"
    EXPLORATORY = "exploratory"


class FactorialBenefitDirection(StrEnum):
    """Frozen sign interpretation copied from the complete source definition."""

    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"
    DESCRIPTIVE = "descriptive_not_ordered"


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        _fail(f"{field} must be a stable ASCII identifier")
    return value


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail(f"{field} must be a lowercase SHA-256 digest")
    return value


def _axis_count(value: object, field: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 2
        or value > _MAX_AXIS_COUNT
    ):
        _fail(f"{field} must be an integer from 2 through {_MAX_AXIS_COUNT}")
    return value


def _ordinal(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(f"{field} must be a non-negative integer")
    return value


def _aware_utc(value: object, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        _fail(f"{field} must be a timezone-aware datetime")
    return value.astimezone(UTC)


def _fraction(value: ExactRational) -> Fraction:
    if not isinstance(value, ExactRational):
        raise TypeError("exact values must be ExactRational instances")
    return Fraction(value.numerator, value.denominator)


def _exact(value: Fraction | int) -> ExactRational:
    material = Fraction(value)
    return ExactRational.from_ratio(material.numerator, material.denominator)


def _is_non_negative(value: ExactRational) -> bool:
    return value.numerator >= 0


def _subtract_factorial_metrics(
    candidate: ExactMetric,
    baseline: ExactMetric,
) -> ExactMetric:
    if type(candidate) is not ExactMetric or type(baseline) is not ExactMetric:
        _fail("factorial transition requires exact baseline and candidate metrics")
    if (
        candidate.state is not ExactMetricState.DEFINED
        or baseline.state is not ExactMetricState.DEFINED
    ):
        return ExactMetric(ExactMetricState.UNDEFINED_WORLD_METRIC, None)
    assert candidate.value is not None
    assert baseline.value is not None
    return ExactMetric(
        ExactMetricState.DEFINED,
        candidate.value.minus(baseline.value),
    )


def _validated_metric_copy(metric: ExactMetric) -> ExactMetric:
    if type(metric) is not ExactMetric:
        _fail("factorial value must use the exact metric type")
    value = metric.value
    validated_value = (
        None if value is None else ExactRational(value.numerator, value.denominator)
    )
    return ExactMetric(metric.state, validated_value)


@dataclass(frozen=True, slots=True)
class FactorialOutputSpec:
    """One output fixed before target outcomes become available."""

    output_id: str
    role: FactorialOutputRole
    undefined_policy: FactorialUndefinedPolicy
    benefit_direction: FactorialBenefitDirection
    source_definition_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.output_id, "output_id")
        if not isinstance(self.role, FactorialOutputRole):
            _fail("factorial output role must use the closed vocabulary")
        if not isinstance(self.undefined_policy, FactorialUndefinedPolicy):
            _fail("undefined_policy must use the closed factorial vocabulary")
        if (
            self.role is FactorialOutputRole.PRIMARY
            and self.undefined_policy is not FactorialUndefinedPolicy.FAIL_CLOSED
        ):
            _fail("the primary factorial output must fail closed when undefined")
        if not isinstance(self.benefit_direction, FactorialBenefitDirection):
            _fail("benefit direction must use the closed factorial vocabulary")
        _sha256(self.source_definition_sha256, "source_definition_sha256")

    @property
    def primary(self) -> bool:
        return self.role is FactorialOutputRole.PRIMARY

    def to_mapping(self) -> dict[str, object]:
        return {
            "benefit_direction": self.benefit_direction.value,
            "output_id": self.output_id,
            "primary": self.primary,
            "role": self.role.value,
            "source_definition_sha256": self.source_definition_sha256,
            "undefined_policy": self.undefined_policy.value,
        }


@dataclass(frozen=True, slots=True)
class FactorialPrecisionTarget:
    """Frozen exact threshold checked only after every bootstrap replicate.

    The timestamps are caller-provided chronology commitments.  They do not by
    themselves constitute independent custody or external preregistration.
    """

    output_id: str
    maximum_bootstrap_mean_variance: ExactRational
    frozen_at: datetime
    outcomes_available_at: datetime
    policy_post_data_used: Literal[False] = False

    def __post_init__(self) -> None:
        _identifier(self.output_id, "precision target output_id")
        if not isinstance(self.maximum_bootstrap_mean_variance, ExactRational):
            raise TypeError("maximum_bootstrap_mean_variance must be ExactRational")
        if not _is_non_negative(self.maximum_bootstrap_mean_variance):
            _fail("maximum bootstrap mean variance cannot be negative")
        frozen = _aware_utc(self.frozen_at, "precision target frozen_at")
        available = _aware_utc(
            self.outcomes_available_at, "precision target outcomes_available_at"
        )
        object.__setattr__(self, "frozen_at", frozen)
        object.__setattr__(self, "outcomes_available_at", available)
        if frozen >= available:
            _fail(
                "precision target must be frozen before target outcomes are available"
            )
        if self.policy_post_data_used is not False:
            _fail("policy-post data cannot be used to set the precision target")

    def to_mapping(self) -> dict[str, object]:
        return {
            "frozen_at": self.frozen_at.isoformat(),
            "maximum_bootstrap_mean_variance": (
                self.maximum_bootstrap_mean_variance.to_mapping()
            ),
            "outcomes_available_at": self.outcomes_available_at.isoformat(),
            "output_id": self.output_id,
            "policy_post_data_used": self.policy_post_data_used,
        }


@dataclass(frozen=True, slots=True)
class LedgerFactorialSpec:
    """Immutable balanced-grid and crossed-bootstrap analysis contract."""

    spec_id: str
    posterior_draw_count: int
    scenario_replicate_count: int
    outputs: tuple[FactorialOutputSpec, ...]
    bootstrap_replicates: int
    bootstrap_seed: str
    precision_target: FactorialPrecisionTarget
    frozen_at: datetime
    outcomes_available_at: datetime
    synthetic_only: Literal[True] = True
    externally_preregistered: Literal[False] = False
    schema_version: str = LEDGER_FACTORIAL_SPEC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _identifier(self.spec_id, "spec_id")
        posterior_count = _axis_count(self.posterior_draw_count, "posterior_draw_count")
        scenario_count = _axis_count(
            self.scenario_replicate_count, "scenario_replicate_count"
        )
        if posterior_count * scenario_count > _MAX_GRID_CELLS:
            _fail("factorial grid exceeds its fixed cell limit")
        if not isinstance(self.outputs, tuple) or not self.outputs:
            _fail("outputs must be a non-empty immutable tuple")
        if not all(type(item) is FactorialOutputSpec for item in self.outputs):
            _fail("outputs must contain exact FactorialOutputSpec values")
        output_ids = tuple(item.output_id for item in self.outputs)
        if output_ids != tuple(sorted(output_ids)) or len(set(output_ids)) != len(
            output_ids
        ):
            _fail("outputs must be unique and sorted by output_id")
        primary_ids = tuple(item.output_id for item in self.outputs if item.primary)
        if len(primary_ids) != 1:
            _fail("exactly one registered output must be primary")
        if (
            isinstance(self.bootstrap_replicates, bool)
            or not isinstance(self.bootstrap_replicates, int)
            or self.bootstrap_replicates < 2
            or self.bootstrap_replicates > _MAX_BOOTSTRAP_REPLICATES
        ):
            _fail(
                "bootstrap_replicates must be an integer from 2 through "
                f"{_MAX_BOOTSTRAP_REPLICATES}"
            )
        _identifier(self.bootstrap_seed, "bootstrap_seed")
        if type(self.precision_target) is not FactorialPrecisionTarget:
            _fail("precision_target must have the exact frozen schema type")
        frozen = _aware_utc(self.frozen_at, "frozen_at")
        available = _aware_utc(self.outcomes_available_at, "outcomes_available_at")
        object.__setattr__(self, "frozen_at", frozen)
        object.__setattr__(self, "outcomes_available_at", available)
        if frozen >= available:
            _fail("analysis specification must be frozen before target outcomes")
        if (
            self.precision_target.output_id != primary_ids[0]
            or self.precision_target.frozen_at != frozen
            or self.precision_target.outcomes_available_at != available
        ):
            _fail("precision target must bind the primary output and spec chronology")
        if self.synthetic_only is not True:
            _fail("this analysis core must remain synthetic-only")
        if self.externally_preregistered is not False:
            _fail("local timestamps cannot claim external preregistration")
        if self.schema_version != LEDGER_FACTORIAL_SPEC_SCHEMA_VERSION:
            _fail("factorial specification schema version changed")

    @property
    def primary_output_id(self) -> str:
        return next(item.output_id for item in self.outputs if item.primary)

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def to_mapping(self) -> dict[str, object]:
        return {
            "bootstrap_replicates": self.bootstrap_replicates,
            "bootstrap_seed": self.bootstrap_seed,
            "externally_preregistered": self.externally_preregistered,
            "frozen_at": self.frozen_at.isoformat(),
            "outputs": [item.to_mapping() for item in self.outputs],
            "outcomes_available_at": self.outcomes_available_at.isoformat(),
            "posterior_draw_count": self.posterior_draw_count,
            "precision_target": self.precision_target.to_mapping(),
            "scenario_replicate_count": self.scenario_replicate_count,
            "schema_version": self.schema_version,
            "spec_id": self.spec_id,
            "synthetic_only": self.synthetic_only,
        }


@dataclass(frozen=True, slots=True)
class FactorialOutcome:
    """One exact value or explicit undefined state in a grid cell."""

    output_id: str
    baseline: ExactMetric
    candidate: ExactMetric
    metric: ExactMetric

    def __post_init__(self) -> None:
        _identifier(self.output_id, "outcome output_id")
        if not all(
            type(value) is ExactMetric
            for value in (self.baseline, self.candidate, self.metric)
        ):
            raise TypeError("factorial transition values must be exact metrics")
        if self.metric != _subtract_factorial_metrics(self.candidate, self.baseline):
            _fail("factorial metric is not exact candidate minus baseline")

    @classmethod
    def defined(cls, output_id: str, value: ExactRational) -> Self:
        """Construct a defined exact outcome without floating-point conversion."""
        return cls(
            output_id=output_id,
            baseline=ExactMetric.integer(0),
            candidate=ExactMetric(ExactMetricState.DEFINED, value),
            metric=ExactMetric(state=ExactMetricState.DEFINED, value=value),
        )

    @classmethod
    def from_transition(
        cls,
        output_id: str,
        *,
        baseline: ExactMetric,
        candidate: ExactMetric,
    ) -> Self:
        """Retain both world states and derive their exact paired difference."""
        return cls(
            output_id=output_id,
            baseline=baseline,
            candidate=candidate,
            metric=_subtract_factorial_metrics(candidate, baseline),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "baseline": self.baseline.to_mapping(),
            "candidate": self.candidate.to_mapping(),
            "metric": self.metric.to_mapping(),
            "output_id": self.output_id,
        }


@dataclass(frozen=True, slots=True)
class FactorialGridCell:
    """One immutable posterior-row by scenario-column outcome cell."""

    posterior_index: int
    scenario_index: int
    ordinal: int
    posterior_commitment_sha256: str
    scenario_commitment_sha256: str
    outcomes: tuple[FactorialOutcome, ...]

    def __post_init__(self) -> None:
        _ordinal(self.posterior_index, "posterior_index")
        _ordinal(self.scenario_index, "scenario_index")
        _ordinal(self.ordinal, "ordinal")
        _sha256(self.posterior_commitment_sha256, "posterior commitment")
        _sha256(self.scenario_commitment_sha256, "scenario commitment")
        if not isinstance(self.outcomes, tuple) or not self.outcomes:
            _fail("cell outcomes must be a non-empty immutable tuple")
        if not all(type(item) is FactorialOutcome for item in self.outcomes):
            _fail("cell outcomes must use the exact FactorialOutcome schema")
        output_ids = tuple(item.output_id for item in self.outcomes)
        if output_ids != tuple(sorted(output_ids)) or len(set(output_ids)) != len(
            output_ids
        ):
            _fail("cell outcomes must be unique and sorted by output_id")

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def to_mapping(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "outcomes": [item.to_mapping() for item in self.outcomes],
            "posterior_commitment_sha256": self.posterior_commitment_sha256,
            "posterior_index": self.posterior_index,
            "scenario_commitment_sha256": self.scenario_commitment_sha256,
            "scenario_index": self.scenario_index,
        }


@dataclass(frozen=True, slots=True)
class ExactPopulationAnova:
    """Exact population decomposition over the complete balanced grid."""

    grand_mean: ExactRational
    posterior: ExactRational
    scenario: ExactRational
    interaction: ExactRational
    total: ExactRational

    def __post_init__(self) -> None:
        for field in ("grand_mean", "posterior", "scenario", "interaction", "total"):
            if not isinstance(getattr(self, field), ExactRational):
                raise TypeError(f"{field} must be ExactRational")
        for field in ("posterior", "scenario", "interaction", "total"):
            if not _is_non_negative(getattr(self, field)):
                _fail(f"ANOVA {field} component cannot be negative")
        recomposed = self.posterior.plus(self.scenario).plus(self.interaction)
        if recomposed != self.total:
            _fail(
                "exact ANOVA identity posterior + scenario + interaction = total failed"
            )

    def to_mapping(self) -> dict[str, object]:
        return {
            "grand_mean": self.grand_mean.to_mapping(),
            "interaction": self.interaction.to_mapping(),
            "posterior": self.posterior.to_mapping(),
            "scenario": self.scenario.to_mapping(),
            "total": self.total.to_mapping(),
        }


@dataclass(frozen=True, slots=True)
class CrossedBootstrapReplicate:
    """One row-and-column resample; cells are never sampled flat iid."""

    ordinal: int
    posterior_indices: tuple[int, ...]
    scenario_indices: tuple[int, ...]
    posterior_event_keys: tuple[str, ...]
    scenario_event_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        _ordinal(self.ordinal, "bootstrap ordinal")
        if not self.posterior_indices or not self.scenario_indices:
            _fail("crossed bootstrap axes cannot be empty")
        if len(self.posterior_indices) != len(self.posterior_event_keys) or len(
            self.scenario_indices
        ) != len(self.scenario_event_keys):
            _fail("every bootstrap axis selection requires one event key")
        for value in (*self.posterior_indices, *self.scenario_indices):
            _ordinal(value, "bootstrap selected index")
        for key in (*self.posterior_event_keys, *self.scenario_event_keys):
            if not isinstance(key, str) or _EVENT_KEY.fullmatch(key) is None:
                _fail("bootstrap event key has an invalid format")
        if len(set(self.posterior_event_keys)) != len(self.posterior_event_keys) or len(
            set(self.scenario_event_keys)
        ) != len(self.scenario_event_keys):
            _fail("bootstrap event keys must be unique within each axis draw")
        if set(self.posterior_event_keys) & set(self.scenario_event_keys):
            _fail("posterior and scenario bootstrap key domains must be disjoint")

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def to_mapping(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "posterior_event_keys": list(self.posterior_event_keys),
            "posterior_indices": list(self.posterior_indices),
            "scenario_event_keys": list(self.scenario_event_keys),
            "scenario_indices": list(self.scenario_indices),
        }


@dataclass(frozen=True, slots=True)
class CrossedBootstrapManifest:
    """Outcome-independent deterministic resampling manifest shared by outputs."""

    analysis_spec_sha256: str
    posterior_draw_count: int
    scenario_replicate_count: int
    replicate_count: int
    replicates: tuple[CrossedBootstrapReplicate, ...]
    flat_iid_cell_resampling: Literal[False] = False
    schema_version: str = LEDGER_FACTORIAL_BOOTSTRAP_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _sha256(self.analysis_spec_sha256, "analysis_spec_sha256")
        posterior_count = _axis_count(
            self.posterior_draw_count, "bootstrap posterior_draw_count"
        )
        scenario_count = _axis_count(
            self.scenario_replicate_count, "bootstrap scenario_replicate_count"
        )
        if (
            isinstance(self.replicate_count, bool)
            or not isinstance(self.replicate_count, int)
            or self.replicate_count < 2
        ):
            _fail("bootstrap replicate_count must be at least two")
        if not isinstance(self.replicates, tuple) or len(self.replicates) != (
            self.replicate_count
        ):
            _fail("bootstrap manifest must retain every fixed replicate")
        for ordinal, replicate in enumerate(self.replicates):
            if type(replicate) is not CrossedBootstrapReplicate:
                _fail("bootstrap manifest has the wrong replicate type")
            if (
                replicate.ordinal != ordinal
                or len(replicate.posterior_indices) != posterior_count
                or len(replicate.scenario_indices) != scenario_count
                or any(
                    index >= posterior_count for index in replicate.posterior_indices
                )
                or any(index >= scenario_count for index in replicate.scenario_indices)
            ):
                _fail("bootstrap replicate differs from the fixed balanced axes")
        all_keys = tuple(
            key
            for replicate in self.replicates
            for key in (
                *replicate.posterior_event_keys,
                *replicate.scenario_event_keys,
            )
        )
        if len(set(all_keys)) != len(all_keys):
            _fail("bootstrap event keys must be unique across the fixed manifest")
        if self.flat_iid_cell_resampling is not False:
            _fail("flat iid cell bootstrap is forbidden for the crossed design")
        if self.schema_version != LEDGER_FACTORIAL_BOOTSTRAP_SCHEMA_VERSION:
            _fail("crossed bootstrap schema version changed")

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def to_mapping(self) -> dict[str, object]:
        return {
            "analysis_spec_sha256": self.analysis_spec_sha256,
            "flat_iid_cell_resampling": self.flat_iid_cell_resampling,
            "posterior_draw_count": self.posterior_draw_count,
            "replicate_count": self.replicate_count,
            "replicates": [item.to_mapping() for item in self.replicates],
            "scenario_replicate_count": self.scenario_replicate_count,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True)
class FactorialOutputObservation:
    """One retained grid observation, including an explicit undefined state."""

    ordinal: int
    baseline: ExactMetric
    candidate: ExactMetric
    metric: ExactMetric

    def __post_init__(self) -> None:
        _ordinal(self.ordinal, "observation ordinal")
        if not all(
            type(value) is ExactMetric
            for value in (self.baseline, self.candidate, self.metric)
        ):
            raise TypeError("observation transition values must be ExactMetric")
        if self.metric != _subtract_factorial_metrics(self.candidate, self.baseline):
            _fail("observation metric is not exact candidate minus baseline")

    def to_mapping(self) -> dict[str, object]:
        return {
            "baseline": self.baseline.to_mapping(),
            "candidate": self.candidate.to_mapping(),
            "metric": self.metric.to_mapping(),
            "ordinal": self.ordinal,
        }


@dataclass(frozen=True, slots=True)
class CrossedBootstrapMean:
    """Descriptive resampled mean; it is not labelled as an interval or MCSE."""

    replicate_ordinal: int
    replicate_manifest_sha256: str
    value: ExactRational

    def __post_init__(self) -> None:
        _ordinal(self.replicate_ordinal, "replicate_ordinal")
        _sha256(self.replicate_manifest_sha256, "replicate_manifest_sha256")
        if not isinstance(self.value, ExactRational):
            raise TypeError("bootstrap mean must be ExactRational")

    def to_mapping(self) -> dict[str, object]:
        return {
            "replicate_manifest_sha256": self.replicate_manifest_sha256,
            "replicate_ordinal": self.replicate_ordinal,
            "value": self.value.to_mapping(),
        }


class FactorialOutputState(StrEnum):
    """Whether the full registered output grid admits numeric analysis."""

    DEFINED_COMPLETE_GRID = "defined_complete_grid"
    UNDEFINED_RETAINED_NO_COMPLETE_CASE = "undefined_retained_no_complete_case"


@dataclass(frozen=True, slots=True)
class FactorialOutputAnalysis:
    """Complete-grid output analysis bound to the one shared manifest."""

    output_id: str
    primary: bool
    state: FactorialOutputState
    observations: tuple[FactorialOutputObservation, ...]
    undefined_ordinals: tuple[int, ...]
    anova: ExactPopulationAnova | None
    bootstrap_means: tuple[CrossedBootstrapMean, ...]
    bootstrap_manifest_sha256: str
    complete_case_analysis_used: Literal[False] = False

    def __post_init__(self) -> None:
        _identifier(self.output_id, "analysis output_id")
        if not isinstance(self.primary, bool):
            _fail("analysis primary flag must be boolean")
        if not isinstance(self.state, FactorialOutputState):
            _fail("factorial output state has the wrong type")
        if not isinstance(self.observations, tuple) or not self.observations:
            _fail("output analysis must retain every grid observation")
        for ordinal, observation in enumerate(self.observations):
            if (
                type(observation) is not FactorialOutputObservation
                or observation.ordinal != ordinal
            ):
                _fail("output observations must cover contiguous grid ordinals")
        expected_undefined = tuple(
            item.ordinal
            for item in self.observations
            if item.metric.state is not ExactMetricState.DEFINED
        )
        if self.undefined_ordinals != expected_undefined:
            _fail("undefined ordinal audit differs from retained observations")
        _sha256(self.bootstrap_manifest_sha256, "bootstrap_manifest_sha256")
        if self.state is FactorialOutputState.DEFINED_COMPLETE_GRID:
            if expected_undefined or not isinstance(self.anova, ExactPopulationAnova):
                _fail("defined output requires a complete exact ANOVA")
            if not self.bootstrap_means:
                _fail("defined output requires every crossed bootstrap mean")
            for ordinal, draw in enumerate(self.bootstrap_means):
                if (
                    type(draw) is not CrossedBootstrapMean
                    or draw.replicate_ordinal != ordinal
                ):
                    _fail("bootstrap means must cover contiguous fixed replicates")
        else:
            if self.primary:
                _fail("the primary output cannot be retained as undefined")
            if not expected_undefined or self.anova is not None or self.bootstrap_means:
                _fail("undefined output cannot use complete-case numeric summaries")
        if self.complete_case_analysis_used is not False:
            _fail("complete-case analysis is forbidden")

    def to_mapping(self) -> dict[str, object]:
        return {
            "anova": None if self.anova is None else self.anova.to_mapping(),
            "bootstrap_manifest_sha256": self.bootstrap_manifest_sha256,
            "bootstrap_means": [item.to_mapping() for item in self.bootstrap_means],
            "complete_case_analysis_used": self.complete_case_analysis_used,
            "observations": [item.to_mapping() for item in self.observations],
            "output_id": self.output_id,
            "primary": self.primary,
            "state": self.state.value,
            "undefined_ordinals": list(self.undefined_ordinals),
        }


@dataclass(frozen=True, slots=True)
class FactorialPrecisionAudit:
    """Exact fixed-``B`` precision result; a failure remains in the artifact.

    The observed quantity is the population variance (denominator ``B``) of
    the complete frozen set of crossed-bootstrap means.  It is not a sample
    variance estimate, an iid MCSE, or an interval.
    """

    output_id: str
    threshold: ExactRational
    observed_bootstrap_mean_variance: ExactRational
    planned_replicates: int
    completed_replicates: int
    passed: bool
    failure_retained: bool
    evaluated_after_complete_fixed_grid: Literal[True] = True
    evaluated_after_all_bootstrap_replicates: Literal[True] = True
    early_stopping_used: Literal[False] = False
    variance_definition: Literal[
        "population_variance_over_all_fixed_crossed_bootstrap_means"
    ] = "population_variance_over_all_fixed_crossed_bootstrap_means"

    def __post_init__(self) -> None:
        _identifier(self.output_id, "precision audit output_id")
        for field in ("threshold", "observed_bootstrap_mean_variance"):
            value = getattr(self, field)
            if not isinstance(value, ExactRational) or not _is_non_negative(value):
                _fail(f"precision {field} must be a non-negative ExactRational")
        if (
            isinstance(self.planned_replicates, bool)
            or not isinstance(self.planned_replicates, int)
            or self.planned_replicates < 2
            or self.completed_replicates != self.planned_replicates
        ):
            _fail("precision audit requires the complete fixed bootstrap plan")
        expected_pass = _fraction(self.observed_bootstrap_mean_variance) <= _fraction(
            self.threshold
        )
        if self.passed is not expected_pass or self.failure_retained is not (
            not expected_pass
        ):
            _fail("precision decision differs from its exact frozen threshold")
        if (
            self.evaluated_after_complete_fixed_grid is not True
            or self.evaluated_after_all_bootstrap_replicates is not True
            or self.early_stopping_used is not False
            or self.variance_definition
            != "population_variance_over_all_fixed_crossed_bootstrap_means"
        ):
            _fail("precision audit chronology or stopping semantics changed")

    def to_mapping(self) -> dict[str, object]:
        return {
            "completed_replicates": self.completed_replicates,
            "early_stopping_used": self.early_stopping_used,
            "evaluated_after_all_bootstrap_replicates": (
                self.evaluated_after_all_bootstrap_replicates
            ),
            "evaluated_after_complete_fixed_grid": (
                self.evaluated_after_complete_fixed_grid
            ),
            "failure_retained": self.failure_retained,
            "observed_bootstrap_mean_variance": (
                self.observed_bootstrap_mean_variance.to_mapping()
            ),
            "output_id": self.output_id,
            "passed": self.passed,
            "planned_replicates": self.planned_replicates,
            "threshold": self.threshold.to_mapping(),
            "variance_definition": self.variance_definition,
        }


@dataclass(frozen=True, slots=True)
class LedgerFactorialClaimBoundary:
    """Conservative claim flags that this pure synthetic core cannot widen."""

    synthetic_only: Literal[True] = True
    real_world_causal_claim: Literal[False] = False
    credible_interval_claim: Literal[False] = False
    confidence_interval_claim: Literal[False] = False
    publication_eligible: Literal[False] = False
    iid_mcse_reported: Literal[False] = False
    externally_preregistered: Literal[False] = False
    analysis_chronology_externally_verified: Literal[False] = False

    def __post_init__(self) -> None:
        if self.synthetic_only is not True or any(
            value is not False
            for value in (
                self.real_world_causal_claim,
                self.credible_interval_claim,
                self.confidence_interval_claim,
                self.publication_eligible,
                self.iid_mcse_reported,
                self.externally_preregistered,
                self.analysis_chronology_externally_verified,
            )
        ):
            _fail("factorial analysis claim boundary was widened")

    def to_mapping(self) -> dict[str, object]:
        return {
            "analysis_chronology_externally_verified": (
                self.analysis_chronology_externally_verified
            ),
            "confidence_interval_claim": self.confidence_interval_claim,
            "credible_interval_claim": self.credible_interval_claim,
            "externally_preregistered": self.externally_preregistered,
            "iid_mcse_reported": self.iid_mcse_reported,
            "publication_eligible": self.publication_eligible,
            "real_world_causal_claim": self.real_world_causal_claim,
            "synthetic_only": self.synthetic_only,
        }


@dataclass(frozen=True, slots=True, init=False)
class LedgerFactorialEvaluation:
    """Canonical exact result over every frozen posterior/scenario cell."""

    spec: LedgerFactorialSpec
    analysis_spec_sha256: str
    grid_manifest_sha256: str
    posterior_commitment_manifest_sha256: str
    scenario_commitment_manifest_sha256: str
    cell_count: int
    bootstrap_manifest: CrossedBootstrapManifest
    outputs: tuple[FactorialOutputAnalysis, ...]
    precision_audit: FactorialPrecisionAudit
    claims: LedgerFactorialClaimBoundary
    claim_scope: str = LEDGER_FACTORIAL_CLAIM_SCOPE
    all_grid_cells_accounted: Literal[True] = True
    failed_cells_silently_dropped: Literal[False] = False
    schema_version: str = LEDGER_FACTORIAL_RESULT_SCHEMA_VERSION

    def __init__(self) -> None:
        raise TypeError(
            "LedgerFactorialEvaluation is derivation-produced; use "
            "derive_ledger_factorial_evaluation or the public bytes verifier"
        )

    def __copy__(self) -> NoReturn:
        raise TypeError("factorial evaluations cannot be copied")

    def __deepcopy__(self, memo: object) -> NoReturn:
        del memo
        raise TypeError("factorial evaluations cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("factorial evaluations cannot be serialized")

    def _fresh_validate(self) -> None:
        if type(self.spec) is not LedgerFactorialSpec:
            _fail("result spec has the wrong exact schema type")
        validated_outputs = tuple(
            FactorialOutputSpec(
                output_id=item.output_id,
                role=item.role,
                undefined_policy=item.undefined_policy,
                benefit_direction=item.benefit_direction,
                source_definition_sha256=item.source_definition_sha256,
            )
            for item in self.spec.outputs
        )
        target = self.spec.precision_target
        validated_target = FactorialPrecisionTarget(
            output_id=target.output_id,
            maximum_bootstrap_mean_variance=ExactRational(
                target.maximum_bootstrap_mean_variance.numerator,
                target.maximum_bootstrap_mean_variance.denominator,
            ),
            frozen_at=target.frozen_at,
            outcomes_available_at=target.outcomes_available_at,
            policy_post_data_used=target.policy_post_data_used,
        )
        validated_spec = LedgerFactorialSpec(
            spec_id=self.spec.spec_id,
            posterior_draw_count=self.spec.posterior_draw_count,
            scenario_replicate_count=self.spec.scenario_replicate_count,
            outputs=validated_outputs,
            bootstrap_replicates=self.spec.bootstrap_replicates,
            bootstrap_seed=self.spec.bootstrap_seed,
            precision_target=validated_target,
            frozen_at=self.spec.frozen_at,
            outcomes_available_at=self.spec.outcomes_available_at,
            synthetic_only=self.spec.synthetic_only,
            externally_preregistered=self.spec.externally_preregistered,
            schema_version=self.spec.schema_version,
        )
        if validated_spec != self.spec:
            _fail("result analysis specification changed after validation")
        if self.analysis_spec_sha256 != self.spec.sha256:
            _fail("result analysis spec commitment does not reproduce")
        for field in (
            "grid_manifest_sha256",
            "posterior_commitment_manifest_sha256",
            "scenario_commitment_manifest_sha256",
        ):
            _sha256(getattr(self, field), field)
        if self.cell_count != (
            self.spec.posterior_draw_count * self.spec.scenario_replicate_count
        ):
            _fail("result cell count does not cover the complete fixed grid")
        expected_bootstrap_manifest = _bootstrap_manifest(self.spec)
        if self.bootstrap_manifest != expected_bootstrap_manifest:
            _fail("result bootstrap manifest differs from the frozen specification")
        if not isinstance(self.outputs, tuple) or len(self.outputs) != len(
            self.spec.outputs
        ):
            _fail("result must retain every registered output")
        manifest_sha256 = self.bootstrap_manifest.sha256
        for expected, actual in zip(self.spec.outputs, self.outputs, strict=True):
            if (
                type(actual) is not FactorialOutputAnalysis
                or actual.output_id != expected.output_id
                or actual.primary is not expected.primary
                or len(actual.observations) != self.cell_count
                or actual.bootstrap_manifest_sha256 != manifest_sha256
            ):
                _fail("output analysis differs from the frozen output manifest")
            if actual.bootstrap_means and len(actual.bootstrap_means) != (
                self.spec.bootstrap_replicates
            ):
                _fail("defined output lacks a complete shared-manifest bootstrap")
            reconstructed_observations: list[FactorialOutputObservation] = []
            for observation in actual.observations:
                if type(observation) is not FactorialOutputObservation:
                    _fail("factorial output contains an invalid observation type")
                reconstructed_observations.append(
                    FactorialOutputObservation(
                        ordinal=observation.ordinal,
                        baseline=_validated_metric_copy(observation.baseline),
                        candidate=_validated_metric_copy(observation.candidate),
                        metric=_validated_metric_copy(observation.metric),
                    )
                )
            observations = tuple(reconstructed_observations)
            undefined_ordinals = tuple(
                item.ordinal
                for item in observations
                if item.metric.state is not ExactMetricState.DEFINED
            )
            if undefined_ordinals:
                if expected.undefined_policy is FactorialUndefinedPolicy.FAIL_CLOSED:
                    _fail("a fail-closed factorial output became undefined")
                expected_analysis = FactorialOutputAnalysis(
                    output_id=expected.output_id,
                    primary=expected.primary,
                    state=FactorialOutputState.UNDEFINED_RETAINED_NO_COMPLETE_CASE,
                    observations=observations,
                    undefined_ordinals=undefined_ordinals,
                    anova=None,
                    bootstrap_means=(),
                    bootstrap_manifest_sha256=manifest_sha256,
                )
            else:
                values = tuple(item.metric.value for item in observations)
                if any(value is None for value in values):
                    _fail("defined factorial observation lost its exact value")
                exact_values = tuple(value for value in values if value is not None)
                matrix = tuple(
                    exact_values[
                        posterior_index * self.spec.scenario_replicate_count : (
                            posterior_index + 1
                        )
                        * self.spec.scenario_replicate_count
                    ]
                    for posterior_index in range(self.spec.posterior_draw_count)
                )
                expected_analysis = FactorialOutputAnalysis(
                    output_id=expected.output_id,
                    primary=expected.primary,
                    state=FactorialOutputState.DEFINED_COMPLETE_GRID,
                    observations=observations,
                    undefined_ordinals=(),
                    anova=_anova(matrix),
                    bootstrap_means=_bootstrap_means(
                        matrix, expected_bootstrap_manifest
                    ),
                    bootstrap_manifest_sha256=manifest_sha256,
                )
            if actual != expected_analysis:
                _fail("factorial output arithmetic differs from retained observations")
        primary = next(item for item in self.outputs if item.primary)
        observed_variance = _population_variance(
            tuple(item.value for item in primary.bootstrap_means)
        )
        threshold = self.spec.precision_target.maximum_bootstrap_mean_variance
        expected_precision_audit = FactorialPrecisionAudit(
            output_id=self.spec.primary_output_id,
            threshold=threshold,
            observed_bootstrap_mean_variance=observed_variance,
            planned_replicates=self.spec.bootstrap_replicates,
            completed_replicates=len(primary.bootstrap_means),
            passed=_fraction(observed_variance) <= _fraction(threshold),
            failure_retained=_fraction(observed_variance) > _fraction(threshold),
        )
        if self.precision_audit != expected_precision_audit:
            _fail("precision audit differs from its frozen target")
        if self.claims != LedgerFactorialClaimBoundary():
            _fail("factorial result claim boundary changed")
        if self.claim_scope != LEDGER_FACTORIAL_CLAIM_SCOPE:
            _fail("factorial claim scope changed")
        if (
            self.all_grid_cells_accounted is not True
            or self.failed_cells_silently_dropped is not False
            or self.schema_version != LEDGER_FACTORIAL_RESULT_SCHEMA_VERSION
        ):
            _fail("factorial result completeness or schema semantics changed")

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_mapping())

    @property
    def sha256(self) -> str:
        return sha256_bytes(self.canonical_bytes)

    def to_mapping(self) -> dict[str, object]:
        return {
            "all_grid_cells_accounted": self.all_grid_cells_accounted,
            "analysis_spec_sha256": self.analysis_spec_sha256,
            "bootstrap_manifest": self.bootstrap_manifest.to_mapping(),
            "cell_count": self.cell_count,
            "claim_scope": self.claim_scope,
            "claims": self.claims.to_mapping(),
            "failed_cells_silently_dropped": self.failed_cells_silently_dropped,
            "grid_manifest_sha256": self.grid_manifest_sha256,
            "outputs": [item.to_mapping() for item in self.outputs],
            "posterior_commitment_manifest_sha256": (
                self.posterior_commitment_manifest_sha256
            ),
            "precision_audit": self.precision_audit.to_mapping(),
            "scenario_commitment_manifest_sha256": (
                self.scenario_commitment_manifest_sha256
            ),
            "schema_version": self.schema_version,
            "spec": self.spec.to_mapping(),
        }


def _ordered_complete_grid(
    spec: LedgerFactorialSpec,
    grid: tuple[FactorialGridCell, ...],
) -> tuple[FactorialGridCell, ...]:
    expected_count = spec.posterior_draw_count * spec.scenario_replicate_count
    if len(grid) != expected_count:
        _fail("factorial grid is missing or has extra cells")
    by_position: dict[tuple[int, int], FactorialGridCell] = {}
    expected_outputs = tuple(item.output_id for item in spec.outputs)
    for cell in grid:
        if type(cell) is not FactorialGridCell:
            _fail("grid must contain exact FactorialGridCell values")
        position = (cell.posterior_index, cell.scenario_index)
        if position in by_position:
            _fail("factorial grid contains a duplicate posterior/scenario cell")
        if (
            cell.posterior_index >= spec.posterior_draw_count
            or cell.scenario_index >= spec.scenario_replicate_count
        ):
            _fail("factorial grid cell lies outside the frozen axes")
        expected_ordinal = (
            cell.posterior_index * spec.scenario_replicate_count + cell.scenario_index
        )
        if cell.ordinal != expected_ordinal:
            _fail("factorial cell ordinal must equal m * R + r")
        if tuple(item.output_id for item in cell.outcomes) != expected_outputs:
            _fail("factorial cell output manifest differs from the frozen spec")
        by_position[position] = cell
    expected_positions = tuple(
        (posterior_index, scenario_index)
        for posterior_index in range(spec.posterior_draw_count)
        for scenario_index in range(spec.scenario_replicate_count)
    )
    if set(by_position) != set(expected_positions):
        _fail("factorial grid is not the complete balanced M x R design")
    ordered = tuple(by_position[position] for position in expected_positions)

    posterior_commitments: list[str] = []
    for posterior_index in range(spec.posterior_draw_count):
        commitments = {
            ordered[
                posterior_index * spec.scenario_replicate_count + scenario_index
            ].posterior_commitment_sha256
            for scenario_index in range(spec.scenario_replicate_count)
        }
        if len(commitments) != 1:
            _fail("one posterior row has inconsistent posterior commitments")
        posterior_commitments.append(next(iter(commitments)))
    if len(set(posterior_commitments)) != len(posterior_commitments):
        _fail("posterior commitments must be unique across posterior rows")

    scenario_commitments: list[str] = []
    for scenario_index in range(spec.scenario_replicate_count):
        commitments = {
            ordered[
                posterior_index * spec.scenario_replicate_count + scenario_index
            ].scenario_commitment_sha256
            for posterior_index in range(spec.posterior_draw_count)
        }
        if len(commitments) != 1:
            _fail("one scenario column has inconsistent scenario commitments")
        scenario_commitments.append(next(iter(commitments)))
    if len(set(scenario_commitments)) != len(scenario_commitments):
        _fail("scenario commitments must be unique across scenario columns")

    primary_index = expected_outputs.index(spec.primary_output_id)
    for cell in ordered:
        primary = cell.outcomes[primary_index].metric
        if primary.state is not ExactMetricState.DEFINED:
            _fail(
                "primary output is undefined; complete-case or silent filtering is "
                f"forbidden at ordinal {cell.ordinal}"
            )
    return ordered


def _bootstrap_manifest(spec: LedgerFactorialSpec) -> CrossedBootstrapManifest:
    replicates: list[CrossedBootstrapReplicate] = []
    seed = f"{spec.bootstrap_seed}:{spec.sha256}"
    for replicate_ordinal in range(spec.bootstrap_replicates):
        posterior_keys: list[str] = []
        posterior_indices: list[int] = []
        for position in range(spec.posterior_draw_count):
            key = EventKey(
                experiment_id=spec.spec_id,
                seed=seed,
                replicate_id=replicate_ordinal,
                simulation_step=position,
                purpose="ledger-factorial-bootstrap-posterior-axis",
            )
            posterior_keys.append(key.value)
            posterior_indices.append(
                deterministic_int(key, 0, spec.posterior_draw_count - 1)
            )
        scenario_keys: list[str] = []
        scenario_indices: list[int] = []
        for position in range(spec.scenario_replicate_count):
            key = EventKey(
                experiment_id=spec.spec_id,
                seed=seed,
                replicate_id=replicate_ordinal,
                simulation_step=position,
                purpose="ledger-factorial-bootstrap-scenario-axis",
            )
            scenario_keys.append(key.value)
            scenario_indices.append(
                deterministic_int(key, 0, spec.scenario_replicate_count - 1)
            )
        replicates.append(
            CrossedBootstrapReplicate(
                ordinal=replicate_ordinal,
                posterior_indices=tuple(posterior_indices),
                scenario_indices=tuple(scenario_indices),
                posterior_event_keys=tuple(posterior_keys),
                scenario_event_keys=tuple(scenario_keys),
            )
        )
    return CrossedBootstrapManifest(
        analysis_spec_sha256=spec.sha256,
        posterior_draw_count=spec.posterior_draw_count,
        scenario_replicate_count=spec.scenario_replicate_count,
        replicate_count=spec.bootstrap_replicates,
        replicates=tuple(replicates),
    )


def _anova(
    matrix: tuple[tuple[ExactRational, ...], ...],
) -> ExactPopulationAnova:
    posterior_count = len(matrix)
    scenario_count = len(matrix[0])
    values = tuple(tuple(_fraction(value) for value in row) for row in matrix)
    grand_mean = sum((sum(row) for row in values), Fraction()) / Fraction(
        posterior_count * scenario_count
    )
    posterior_means = tuple(sum(row) / Fraction(scenario_count) for row in values)
    scenario_means = tuple(
        sum(values[posterior][scenario] for posterior in range(posterior_count))
        / Fraction(posterior_count)
        for scenario in range(scenario_count)
    )
    posterior_deviations = tuple(value - grand_mean for value in posterior_means)
    posterior = sum(
        (value * value for value in posterior_deviations), Fraction()
    ) / Fraction(posterior_count)
    scenario_deviations = tuple(value - grand_mean for value in scenario_means)
    scenario = sum(
        (value * value for value in scenario_deviations), Fraction()
    ) / Fraction(scenario_count)
    interaction_residuals = tuple(
        values[posterior_index][scenario_index]
        - posterior_means[posterior_index]
        - scenario_means[scenario_index]
        + grand_mean
        for posterior_index in range(posterior_count)
        for scenario_index in range(scenario_count)
    )
    interaction = sum(
        (value * value for value in interaction_residuals),
        Fraction(),
    ) / Fraction(posterior_count * scenario_count)
    total_deviations = tuple(value - grand_mean for row in values for value in row)
    total = sum(
        (value * value for value in total_deviations),
        Fraction(),
    ) / Fraction(posterior_count * scenario_count)
    if posterior + scenario + interaction != total:
        _fail("internal exact population ANOVA recomposition failed")
    return ExactPopulationAnova(
        grand_mean=_exact(grand_mean),
        posterior=_exact(posterior),
        scenario=_exact(scenario),
        interaction=_exact(interaction),
        total=_exact(total),
    )


def _bootstrap_means(
    matrix: tuple[tuple[ExactRational, ...], ...],
    manifest: CrossedBootstrapManifest,
) -> tuple[CrossedBootstrapMean, ...]:
    denominator = manifest.posterior_draw_count * manifest.scenario_replicate_count
    draws: list[CrossedBootstrapMean] = []
    for replicate in manifest.replicates:
        total = sum(
            (
                _fraction(matrix[posterior_index][scenario_index])
                for posterior_index in replicate.posterior_indices
                for scenario_index in replicate.scenario_indices
            ),
            Fraction(),
        )
        draws.append(
            CrossedBootstrapMean(
                replicate_ordinal=replicate.ordinal,
                replicate_manifest_sha256=replicate.sha256,
                value=_exact(total / Fraction(denominator)),
            )
        )
    return tuple(draws)


def _population_variance(values: tuple[ExactRational, ...]) -> ExactRational:
    if len(values) < 2:
        _fail("population variance requires the complete fixed bootstrap plan")
    fractions = tuple(_fraction(value) for value in values)
    mean = sum(fractions, Fraction()) / Fraction(len(fractions))
    deviations = tuple(value - mean for value in fractions)
    variance = sum((value * value for value in deviations), Fraction()) / Fraction(
        len(fractions)
    )
    return _exact(variance)


def derive_ledger_factorial_evaluation(
    *,
    spec: LedgerFactorialSpec,
    grid: tuple[FactorialGridCell, ...],
) -> LedgerFactorialEvaluation:
    """Derive exact results from every cell of a frozen balanced grid.

    Input order is irrelevant, but missing, duplicated, undefined-primary, or
    commitment-inconsistent cells fail closed.  The function never changes the
    fixed grid or bootstrap count in response to observed precision.
    """
    if type(spec) is not LedgerFactorialSpec:
        raise TypeError("spec must be an exact LedgerFactorialSpec")
    if not isinstance(grid, tuple):
        raise TypeError("grid must be an immutable tuple")
    ordered = _ordered_complete_grid(spec, grid)
    bootstrap_manifest = _bootstrap_manifest(spec)
    output_analyses: list[FactorialOutputAnalysis] = []
    for output_index, output_spec in enumerate(spec.outputs):
        observations = tuple(
            FactorialOutputObservation(
                ordinal=cell.ordinal,
                baseline=cell.outcomes[output_index].baseline,
                candidate=cell.outcomes[output_index].candidate,
                metric=cell.outcomes[output_index].metric,
            )
            for cell in ordered
        )
        undefined_ordinals = tuple(
            item.ordinal
            for item in observations
            if item.metric.state is not ExactMetricState.DEFINED
        )
        if undefined_ordinals:
            if output_spec.primary:
                _fail("primary output cannot be undefined")
            if output_spec.undefined_policy is FactorialUndefinedPolicy.FAIL_CLOSED:
                _fail(
                    "registered fail-closed output is undefined at ordinals "
                    + ",".join(str(value) for value in undefined_ordinals)
                )
            output_analyses.append(
                FactorialOutputAnalysis(
                    output_id=output_spec.output_id,
                    primary=False,
                    state=FactorialOutputState.UNDEFINED_RETAINED_NO_COMPLETE_CASE,
                    observations=observations,
                    undefined_ordinals=undefined_ordinals,
                    anova=None,
                    bootstrap_means=(),
                    bootstrap_manifest_sha256=bootstrap_manifest.sha256,
                )
            )
            continue
        matrix = tuple(
            tuple(
                observations[
                    posterior_index * spec.scenario_replicate_count + scenario_index
                ].metric.value
                for scenario_index in range(spec.scenario_replicate_count)
            )
            for posterior_index in range(spec.posterior_draw_count)
        )
        if any(value is None for row in matrix for value in row):
            _fail("defined outcome unexpectedly lacks an exact value")
        exact_matrix = tuple(
            tuple(value for value in row if value is not None) for row in matrix
        )
        if any(len(row) != spec.scenario_replicate_count for row in exact_matrix):
            _fail("numeric factorial matrix changed while it was derived")
        output_analyses.append(
            FactorialOutputAnalysis(
                output_id=output_spec.output_id,
                primary=output_spec.primary,
                state=FactorialOutputState.DEFINED_COMPLETE_GRID,
                observations=observations,
                undefined_ordinals=(),
                anova=_anova(exact_matrix),
                bootstrap_means=_bootstrap_means(exact_matrix, bootstrap_manifest),
                bootstrap_manifest_sha256=bootstrap_manifest.sha256,
            )
        )
    outputs = tuple(output_analyses)
    primary = next(item for item in outputs if item.primary)
    if not primary.bootstrap_means:
        _fail("primary output lacks its complete fixed bootstrap")
    observed_variance = _population_variance(
        tuple(item.value for item in primary.bootstrap_means)
    )
    threshold = spec.precision_target.maximum_bootstrap_mean_variance
    passed = _fraction(observed_variance) <= _fraction(threshold)
    posterior_commitments = tuple(
        ordered[index * spec.scenario_replicate_count].posterior_commitment_sha256
        for index in range(spec.posterior_draw_count)
    )
    scenario_commitments = tuple(
        ordered[index].scenario_commitment_sha256
        for index in range(spec.scenario_replicate_count)
    )
    precision_audit = FactorialPrecisionAudit(
        output_id=spec.primary_output_id,
        threshold=threshold,
        observed_bootstrap_mean_variance=observed_variance,
        planned_replicates=spec.bootstrap_replicates,
        completed_replicates=len(primary.bootstrap_means),
        passed=passed,
        failure_retained=not passed,
    )
    result = object.__new__(LedgerFactorialEvaluation)
    fields: tuple[tuple[str, object], ...] = (
        ("spec", spec),
        ("analysis_spec_sha256", spec.sha256),
        (
            "grid_manifest_sha256",
            sha256_bytes(canonical_json_bytes([item.to_mapping() for item in ordered])),
        ),
        (
            "posterior_commitment_manifest_sha256",
            sha256_bytes(canonical_json_bytes(list(posterior_commitments))),
        ),
        (
            "scenario_commitment_manifest_sha256",
            sha256_bytes(canonical_json_bytes(list(scenario_commitments))),
        ),
        ("cell_count", len(ordered)),
        ("bootstrap_manifest", bootstrap_manifest),
        ("outputs", outputs),
        ("precision_audit", precision_audit),
        ("claims", LedgerFactorialClaimBoundary()),
        ("claim_scope", LEDGER_FACTORIAL_CLAIM_SCOPE),
        ("all_grid_cells_accounted", True),
        ("failed_cells_silently_dropped", False),
        ("schema_version", LEDGER_FACTORIAL_RESULT_SCHEMA_VERSION),
    )
    for field, value in fields:
        object.__setattr__(result, field, value)
    result._fresh_validate()
    return result


def verify_ledger_factorial_evaluation_bytes(
    content: bytes,
    *,
    expected_sha256: str,
    spec: LedgerFactorialSpec,
    grid: tuple[FactorialGridCell, ...],
) -> LedgerFactorialEvaluation:
    """Verify canonical bytes by fully rederiving them from typed grid inputs.

    A caller-supplied digest only commits to the supplied bytes.  Acceptance
    here does not establish external custody, historical chronology, or source
    authenticity, and it cannot widen the conservative claim flags.
    """
    if type(content) is not bytes:
        raise TypeError("content must be exact bytes")
    expected = _sha256(expected_sha256, "expected_sha256")
    if not content or len(content) > _MAX_RESULT_BYTES:
        _fail("factorial result bytes are empty or exceed their byte limit")
    if sha256_bytes(content) != expected:
        _fail("factorial result hash differs from its external commitment")
    try:
        mapping = read_json_bytes_object(
            content, source="ledger factorial evaluation bytes"
        )
    except MetadataValidationError as error:
        raise LedgerFactorialError(str(error)) from error
    if canonical_json_bytes(mapping) != content:
        _fail("factorial result must use canonical duplicate-free JSON bytes")
    rederived = derive_ledger_factorial_evaluation(spec=spec, grid=grid)
    if content != rederived.canonical_bytes:
        _fail("factorial result differs from complete typed-grid rederivation")
    return rederived


# Short aliases keep the public pure-core vocabulary discoverable without
# pretending that this module is a simulator or a protocol executor.
FactorialAnalysisSpec = LedgerFactorialSpec
FactorialAnalysisResult = LedgerFactorialEvaluation
analyze_ledger_factorial = derive_ledger_factorial_evaluation


__all__ = [
    "LEDGER_FACTORIAL_BOOTSTRAP_SCHEMA_VERSION",
    "LEDGER_FACTORIAL_CLAIM_SCOPE",
    "LEDGER_FACTORIAL_RESULT_SCHEMA_VERSION",
    "LEDGER_FACTORIAL_SPEC_SCHEMA_VERSION",
    "CrossedBootstrapManifest",
    "CrossedBootstrapMean",
    "CrossedBootstrapReplicate",
    "ExactPopulationAnova",
    "FactorialAnalysisResult",
    "FactorialAnalysisSpec",
    "FactorialBenefitDirection",
    "FactorialGridCell",
    "FactorialOutcome",
    "FactorialOutputAnalysis",
    "FactorialOutputRole",
    "FactorialOutputSpec",
    "FactorialOutputState",
    "FactorialUndefinedPolicy",
    "FactorialPrecisionAudit",
    "FactorialPrecisionTarget",
    "LedgerFactorialClaimBoundary",
    "LedgerFactorialError",
    "LedgerFactorialEvaluation",
    "LedgerFactorialSpec",
    "analyze_ledger_factorial",
    "derive_ledger_factorial_evaluation",
    "verify_ledger_factorial_evaluation_bytes",
]
