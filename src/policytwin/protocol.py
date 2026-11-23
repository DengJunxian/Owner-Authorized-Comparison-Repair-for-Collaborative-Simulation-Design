"""Study-wide preregistration and deterministic release-eligibility decisions.

The protocol freezes scientific choices before policy-post evaluation data are
available.  The release decision is deterministic evidence for downstream
audits; it is not a signature and it does not make caller declarations true.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from threading import Lock
from typing import NoReturn, cast
from weakref import WeakKeyDictionary

from policytwin.provenance import (
    MetadataValidationError,
    canonical_json_bytes,
    read_json_bytes_object,
    sha256_bytes,
)

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


class ProtocolValidationError(ValueError):
    """Raised when a study protocol or release-evidence object is unsafe."""


def _fail(message: str) -> NoReturn:
    raise ProtocolValidationError(message)


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        _fail(f"{field} must be a stable identifier")
    return value


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{field} must be a non-empty string")
    return value


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail(f"{field} must be a lowercase SHA-256 digest")
    return value


def _optional_sha256(value: object, field: str) -> str | None:
    if value is None:
        return None
    return _sha256(value, field)


def _aware(value: object, field: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as error:
            raise ProtocolValidationError(f"{field} must be ISO 8601") from error
    else:
        _fail(f"{field} must be an aware datetime")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        _fail(f"{field} must include a timezone")
    return parsed.astimezone(UTC)


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


def _sequence(value: object, field: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        _fail(f"{field} must be an array")
    return cast(Sequence[object], value)


def _enum(enum_type: type[StrEnum], value: object, field: str) -> StrEnum:
    if not isinstance(value, str):
        _fail(f"{field} must be a string")
    try:
        return enum_type(value)
    except ValueError as error:
        allowed = ", ".join(member.value for member in enum_type)
        raise ProtocolValidationError(f"{field} must be one of: {allowed}") from error


def _unique(values: tuple[str, ...], field: str, *, nonempty: bool = True) -> None:
    if nonempty and not values:
        _fail(f"{field} must not be empty")
    if len(set(values)) != len(values):
        _fail(f"{field} must not contain duplicates")


class StudyMode(StrEnum):
    """Requested design chronology; the outcome-knowledge audit may downgrade it."""

    SYNTHETIC = "synthetic_engineering"
    PSEUDO_PROSPECTIVE = "pseudo_prospective_real_data"
    RETROSPECTIVE_OUTCOME_KNOWN = "retrospective_outcome_known_design"


class ClaimScope(StrEnum):
    """Maximum scientific interpretation allowed by the registered protocol."""

    SYNTHETIC_ONLY = "synthetic_mechanics_only"
    MODEL_IMPLIED = "model_implied_counterfactual"
    REAL_CAUSAL = "real_world_causal"


class RandomnessProtocol(StrEnum):
    """Frozen coupling rule for the main counterfactual comparison."""

    EVENT_KEYED_PAIRED = "event_keyed_paired"
    INDEPENDENT_DIAGNOSTIC = "independent_rng_diagnostic_only"
    SEQUENTIAL_DIAGNOSTIC = "same_sequential_seed_diagnostic_only"


class OutputRole(StrEnum):
    """Preregistered status of one output cell."""

    PRIMARY = "primary"
    SECONDARY = "secondary"
    EXPLORATORY = "exploratory"


class CohortBasis(StrEnum):
    """Provenance of a group definition."""

    PRE_TREATMENT_OBSERVED = "pre_treatment_observed"
    LATENT_STRATEGY = "latent_strategy"
    POST_TREATMENT_DESCRIPTIVE = "post_treatment_descriptive"


class ProtocolExecutionConsumer(StrEnum):
    """Production boundary allowed to consume a protocol authorization."""

    CALIBRATION_WORKFLOW = "calibration_workflow"
    INTEGRATED_SYNTHETIC_SIMULATION = "integrated_synthetic_simulation"
    MECHANISM_ABLATION_WORKFLOW = "mechanism_ablation_workflow"
    EXTENDED_ABLATION_WORKFLOW = "extended_ablation_workflow"
    LEDGER_POSTERIOR_EVALUATION = "ledger_posterior_evaluation"
    LEDGER_FACTORIAL_EVALUATION = "ledger_factorial_evaluation"
    POLICY_FUNCTIONAL_META_EVALUATION = "policy_functional_meta_evaluation"
    EMPIRICAL_FROZEN_EVALUATION = "empirical_frozen_evaluation"


class ProtocolExecutionPurpose(StrEnum):
    """Scientific purpose bound to one execution authorization."""

    PRE_POLICY_CALIBRATION = "pre_policy_calibration"
    PAIRED_SYNTHETIC_POLICY_SIMULATION = "paired_synthetic_policy_simulation"
    SYNTHETIC_MECHANISM_ABLATION = "synthetic_mechanism_ablation"
    SYNTHETIC_EXTENDED_ABLATION = "synthetic_extended_ablation"
    SYNTHETIC_LEDGER_POSTERIOR_EVALUATION = "synthetic_ledger_posterior_evaluation"
    SYNTHETIC_LEDGER_FACTORIAL_EVALUATION = "synthetic_ledger_factorial_evaluation"
    SYNTHETIC_POLICY_FUNCTIONAL_META_EVALUATION = (
        "synthetic_policy_functional_meta_evaluation"
    )
    POLICY_POST_EMPIRICAL_EVALUATION = "policy_post_empirical_evaluation"


@dataclass(frozen=True, slots=True)
class FrozenDefinition:
    """Human-readable label bound to an externally reviewed semantic object."""

    definition_id: str
    description: str
    definition_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.definition_id, "definition_id")
        _text(self.description, "description")
        _sha256(self.definition_sha256, "definition_sha256")

    def to_mapping(self) -> dict[str, object]:
        """Return the canonical representation."""
        return {
            "definition_id": self.definition_id,
            "definition_sha256": self.definition_sha256,
            "description": self.description,
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> FrozenDefinition:
        """Parse an exact-schema definition."""
        _fields(
            mapping,
            frozenset({"definition_id", "definition_sha256", "description"}),
            "frozen definition",
        )
        return cls(
            definition_id=_identifier(mapping["definition_id"], "definition_id"),
            description=_text(mapping["description"], "description"),
            definition_sha256=_sha256(
                mapping["definition_sha256"], "definition_sha256"
            ),
        )


@dataclass(frozen=True, slots=True)
class FrozenCohort:
    """One immutable cohort with an explicit interpretation boundary."""

    cohort_id: str
    description: str
    definition_sha256: str
    basis: CohortBasis

    def __post_init__(self) -> None:
        _identifier(self.cohort_id, "cohort_id")
        _text(self.description, "cohort description")
        _sha256(self.definition_sha256, "cohort definition_sha256")
        if not isinstance(self.basis, CohortBasis):
            _fail("cohort basis must be a CohortBasis")

    def to_mapping(self) -> dict[str, object]:
        """Return the canonical representation."""
        return {
            "basis": self.basis.value,
            "cohort_id": self.cohort_id,
            "definition_sha256": self.definition_sha256,
            "description": self.description,
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> FrozenCohort:
        """Parse one exact-schema cohort."""
        _fields(
            mapping,
            frozenset({"basis", "cohort_id", "definition_sha256", "description"}),
            "frozen cohort",
        )
        return cls(
            cohort_id=_identifier(mapping["cohort_id"], "cohort_id"),
            description=_text(mapping["description"], "cohort description"),
            definition_sha256=_sha256(
                mapping["definition_sha256"], "cohort definition_sha256"
            ),
            basis=cast(CohortBasis, _enum(CohortBasis, mapping["basis"], "basis")),
        )


@dataclass(frozen=True, slots=True)
class RegisteredOutput:
    """One metric/cohort/horizon cell with fixed uncertainty semantics."""

    output_id: str
    metric_id: str
    cohort_id: str
    horizon_id: str
    role: OutputRole
    interval_method: str
    interval_levels: tuple[float, ...]

    def __post_init__(self) -> None:
        for field in ("output_id", "metric_id", "cohort_id", "horizon_id"):
            _identifier(getattr(self, field), field)
        if not isinstance(self.role, OutputRole):
            _fail("output role must be an OutputRole")
        _identifier(self.interval_method, "interval_method")
        if not isinstance(self.interval_levels, tuple):
            _fail("interval_levels must be an immutable tuple")
        if len(set(self.interval_levels)) != len(self.interval_levels):
            _fail("interval_levels must not contain duplicates")
        previous = 0.0
        for level in self.interval_levels:
            if (
                isinstance(level, bool)
                or not isinstance(level, int | float)
                or not 0.0 < float(level) < 1.0
                or float(level) <= previous
            ):
                _fail("interval_levels must be strictly increasing probabilities")
            previous = float(level)
        if self.interval_method == "none" and self.interval_levels:
            _fail("interval_method none cannot declare interval levels")
        if self.interval_method != "none" and not self.interval_levels:
            _fail("an interval method requires at least one interval level")

    def to_mapping(self) -> dict[str, object]:
        """Return the canonical representation."""
        return {
            "cohort_id": self.cohort_id,
            "horizon_id": self.horizon_id,
            "interval_levels": list(self.interval_levels),
            "interval_method": self.interval_method,
            "metric_id": self.metric_id,
            "output_id": self.output_id,
            "role": self.role.value,
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> RegisteredOutput:
        """Parse one exact-schema output specification."""
        _fields(
            mapping,
            frozenset(
                {
                    "cohort_id",
                    "horizon_id",
                    "interval_levels",
                    "interval_method",
                    "metric_id",
                    "output_id",
                    "role",
                }
            ),
            "registered output",
        )
        raw_levels = _sequence(mapping["interval_levels"], "interval_levels")
        levels: list[float] = []
        for value in raw_levels:
            if isinstance(value, bool) or not isinstance(value, int | float):
                _fail("interval_levels must contain only numbers")
            levels.append(float(value))
        return cls(
            output_id=_identifier(mapping["output_id"], "output_id"),
            metric_id=_identifier(mapping["metric_id"], "metric_id"),
            cohort_id=_identifier(mapping["cohort_id"], "cohort_id"),
            horizon_id=_identifier(mapping["horizon_id"], "horizon_id"),
            role=cast(OutputRole, _enum(OutputRole, mapping["role"], "role")),
            interval_method=_identifier(mapping["interval_method"], "interval_method"),
            interval_levels=tuple(levels),
        )


@dataclass(frozen=True, slots=True)
class FailureThreshold:
    """A validity threshold frozen independently of target outcomes."""

    check_id: str
    comparator: str
    yellow_value: float
    red_value: float
    abstain_at_red: bool

    def __post_init__(self) -> None:
        _identifier(self.check_id, "check_id")
        if self.comparator not in {"higher_is_worse", "lower_is_worse"}:
            _fail("comparator must be higher_is_worse or lower_is_worse")
        for field in ("yellow_value", "red_value"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int | float):
                _fail(f"{field} must be numeric")
            if not float("-inf") < float(value) < float("inf"):
                _fail(f"{field} must be finite")
        if self.comparator == "higher_is_worse" and self.yellow_value > self.red_value:
            _fail("higher-is-worse red threshold must be at least yellow")
        if self.comparator == "lower_is_worse" and self.yellow_value < self.red_value:
            _fail("lower-is-worse red threshold must be at most yellow")
        if not isinstance(self.abstain_at_red, bool):
            _fail("abstain_at_red must be boolean")

    def to_mapping(self) -> dict[str, object]:
        """Return the canonical representation."""
        return {
            "abstain_at_red": self.abstain_at_red,
            "check_id": self.check_id,
            "comparator": self.comparator,
            "red_value": float(self.red_value),
            "yellow_value": float(self.yellow_value),
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> FailureThreshold:
        """Parse one exact-schema threshold."""
        _fields(
            mapping,
            frozenset(
                {
                    "abstain_at_red",
                    "check_id",
                    "comparator",
                    "red_value",
                    "yellow_value",
                }
            ),
            "failure threshold",
        )
        abstain = mapping["abstain_at_red"]
        if not isinstance(abstain, bool):
            _fail("abstain_at_red must be boolean")
        comparator = mapping["comparator"]
        if not isinstance(comparator, str):
            _fail("comparator must be a string")
        yellow = mapping["yellow_value"]
        red = mapping["red_value"]
        if isinstance(yellow, bool) or not isinstance(yellow, int | float):
            _fail("yellow_value must be numeric")
        if isinstance(red, bool) or not isinstance(red, int | float):
            _fail("red_value must be numeric")
        return cls(
            check_id=_identifier(mapping["check_id"], "check_id"),
            comparator=comparator,
            yellow_value=float(yellow),
            red_value=float(red),
            abstain_at_red=abstain,
        )


@dataclass(frozen=True, slots=True)
class StudyProtocol:
    """Complete immutable commitment to one study's scientific choices."""

    schema_version: int
    study_id: str
    mode: StudyMode
    claim_scope: ClaimScope
    registered_at: datetime
    knowledge_cutoff: datetime
    model_frozen_at: datetime
    policy_start: datetime
    policy_end: datetime
    evaluation_reveal_at: datetime
    estimand: FrozenDefinition
    baseline_policy: FrozenDefinition
    candidate_policy: FrozenDefinition
    cohorts: tuple[FrozenCohort, ...]
    horizons: tuple[FrozenDefinition, ...]
    metrics: tuple[FrozenDefinition, ...]
    baselines: tuple[FrozenDefinition, ...]
    outputs: tuple[RegisteredOutput, ...]
    calibration_statistics: tuple[str, ...]
    heldout_validation_statistics: tuple[str, ...]
    policy_evaluation_outcomes: tuple[str, ...]
    failure_thresholds: tuple[FailureThreshold, ...]
    abstention_rule: FrozenDefinition
    randomness_protocol: RandomnessProtocol
    evidence_plans: tuple[FrozenDefinition, ...] = ()
    target_event_id: str | None = None
    outcome_knowledge_manifest_source_sha256: str | None = None
    outcome_knowledge_manifest_semantic_sha256: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 2:
            _fail("study protocol schema_version must equal 2")
        _identifier(self.study_id, "study_id")
        if not isinstance(self.mode, StudyMode):
            _fail("mode must be a StudyMode")
        if not isinstance(self.claim_scope, ClaimScope):
            _fail("claim_scope must be a ClaimScope")
        if (
            self.mode is StudyMode.SYNTHETIC
            and self.claim_scope is not ClaimScope.SYNTHETIC_ONLY
        ):
            _fail("synthetic protocols may claim only synthetic mechanics")
        if self.mode is not StudyMode.SYNTHETIC and (
            self.claim_scope is ClaimScope.SYNTHETIC_ONLY
        ):
            _fail("real-data protocols must state a non-synthetic claim boundary")
        outcome_commitments = (
            self.target_event_id,
            self.outcome_knowledge_manifest_source_sha256,
            self.outcome_knowledge_manifest_semantic_sha256,
        )
        if self.mode is StudyMode.SYNTHETIC:
            if any(value is not None for value in outcome_commitments):
                _fail("synthetic protocols must not bind a real-event outcome ledger")
        else:
            if any(value is None for value in outcome_commitments):
                _fail(
                    "real-data protocols require a target event and exact outcome-"
                    "knowledge manifest commitments"
                )
            event_id = _identifier(self.target_event_id, "target_event_id")
            if event_id != self.candidate_policy.definition_id:
                _fail("target_event_id must equal the candidate policy definition ID")
            _sha256(
                self.outcome_knowledge_manifest_source_sha256,
                "outcome_knowledge_manifest_source_sha256",
            )
            _sha256(
                self.outcome_knowledge_manifest_semantic_sha256,
                "outcome_knowledge_manifest_semantic_sha256",
            )
        for field in (
            "registered_at",
            "knowledge_cutoff",
            "model_frozen_at",
            "policy_start",
            "policy_end",
            "evaluation_reveal_at",
        ):
            object.__setattr__(self, field, _aware(getattr(self, field), field))
        if self.mode is StudyMode.RETROSPECTIVE_OUTCOME_KNOWN:
            if not (
                self.knowledge_cutoff
                < self.policy_start
                <= self.policy_end
                < self.model_frozen_at
                <= self.registered_at
                < self.evaluation_reveal_at
            ):
                _fail(
                    "retrospective dates must satisfy cutoff < policy start <= "
                    "policy end < model freeze <= registration < evaluation reveal"
                )
        elif not (
            self.knowledge_cutoff
            <= self.model_frozen_at
            <= self.registered_at
            < self.policy_start
            <= self.policy_end
            < self.evaluation_reveal_at
        ):
            _fail(
                "dates must satisfy cutoff <= model freeze <= registration < "
                "policy start <= policy end < evaluation reveal"
            )
        if self.randomness_protocol is not RandomnessProtocol.EVENT_KEYED_PAIRED:
            _fail("the main study must use event-keyed paired randomness")
        for field in (
            "cohorts",
            "horizons",
            "metrics",
            "baselines",
            "outputs",
            "calibration_statistics",
            "heldout_validation_statistics",
            "policy_evaluation_outcomes",
            "failure_thresholds",
            "evidence_plans",
        ):
            if not isinstance(getattr(self, field), tuple):
                _fail(f"{field} must be an immutable tuple")
        cohort_ids = tuple(item.cohort_id for item in self.cohorts)
        horizon_ids = tuple(item.definition_id for item in self.horizons)
        metric_ids = tuple(item.definition_id for item in self.metrics)
        baseline_ids = tuple(item.definition_id for item in self.baselines)
        output_ids = tuple(item.output_id for item in self.outputs)
        threshold_ids = tuple(item.check_id for item in self.failure_thresholds)
        if not all(isinstance(item, FrozenDefinition) for item in self.evidence_plans):
            _fail("evidence_plans must contain only FrozenDefinition values")
        evidence_plan_ids = tuple(item.definition_id for item in self.evidence_plans)
        for values, field in (
            (cohort_ids, "cohorts"),
            (horizon_ids, "horizons"),
            (metric_ids, "metrics"),
            (baseline_ids, "baselines"),
            (output_ids, "outputs"),
            (threshold_ids, "failure_thresholds"),
        ):
            _unique(values, field)
        _unique(evidence_plan_ids, "evidence_plans", nonempty=False)
        primary_count = 0
        cohort_by_id = {item.cohort_id: item for item in self.cohorts}
        for output in self.outputs:
            if output.metric_id not in metric_ids:
                _fail(f"output {output.output_id} references an unknown metric")
            if output.cohort_id not in cohort_ids:
                _fail(f"output {output.output_id} references an unknown cohort")
            if output.horizon_id not in horizon_ids:
                _fail(f"output {output.output_id} references an unknown horizon")
            if output.role is OutputRole.PRIMARY:
                primary_count += 1
                if (
                    cohort_by_id[output.cohort_id].basis
                    is CohortBasis.POST_TREATMENT_DESCRIPTIVE
                ):
                    _fail("post-treatment descriptive cohorts cannot be primary")
        if primary_count == 0:
            _fail("at least one primary output is required")
        statistic_sets: list[set[str]] = []
        for values, field in (
            (self.calibration_statistics, "calibration_statistics"),
            (self.heldout_validation_statistics, "heldout_validation_statistics"),
            (self.policy_evaluation_outcomes, "policy_evaluation_outcomes"),
        ):
            for value in values:
                _identifier(value, f"{field} item")
            _unique(values, field)
            statistic_sets.append(set(values))
        if any(
            left & right
            for index, left in enumerate(statistic_sets)
            for right in statistic_sets[index + 1 :]
        ):
            _fail(
                "calibration, held-out validation, and policy evaluation names "
                "must be pairwise disjoint"
            )

    @property
    def sha256(self) -> str:
        """Return the semantic commitment for every registered choice."""
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def to_mapping(self) -> dict[str, object]:
        """Return a deterministic, machine-readable representation."""
        return {
            "abstention_rule": self.abstention_rule.to_mapping(),
            "baseline_policy": self.baseline_policy.to_mapping(),
            "baselines": [item.to_mapping() for item in self.baselines],
            "calibration_statistics": list(self.calibration_statistics),
            "candidate_policy": self.candidate_policy.to_mapping(),
            "claim_scope": self.claim_scope.value,
            "cohorts": [item.to_mapping() for item in self.cohorts],
            "estimand": self.estimand.to_mapping(),
            "evidence_plans": [item.to_mapping() for item in self.evidence_plans],
            "evaluation_reveal_at": self.evaluation_reveal_at.isoformat(),
            "failure_thresholds": [
                item.to_mapping() for item in self.failure_thresholds
            ],
            "heldout_validation_statistics": list(self.heldout_validation_statistics),
            "horizons": [item.to_mapping() for item in self.horizons],
            "knowledge_cutoff": self.knowledge_cutoff.isoformat(),
            "metrics": [item.to_mapping() for item in self.metrics],
            "mode": self.mode.value,
            "model_frozen_at": self.model_frozen_at.isoformat(),
            "outputs": [item.to_mapping() for item in self.outputs],
            "policy_end": self.policy_end.isoformat(),
            "policy_evaluation_outcomes": list(self.policy_evaluation_outcomes),
            "policy_start": self.policy_start.isoformat(),
            "randomness_protocol": self.randomness_protocol.value,
            "registered_at": self.registered_at.isoformat(),
            "schema_version": self.schema_version,
            "study_id": self.study_id,
            "target_event_id": self.target_event_id,
            "outcome_knowledge_manifest_source_sha256": (
                self.outcome_knowledge_manifest_source_sha256
            ),
            "outcome_knowledge_manifest_semantic_sha256": (
                self.outcome_knowledge_manifest_semantic_sha256
            ),
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> StudyProtocol:
        """Parse a strict protocol without defaults or unknown fields."""
        expected = frozenset(
            {
                "abstention_rule",
                "baseline_policy",
                "baselines",
                "calibration_statistics",
                "candidate_policy",
                "claim_scope",
                "cohorts",
                "estimand",
                "evidence_plans",
                "evaluation_reveal_at",
                "failure_thresholds",
                "heldout_validation_statistics",
                "horizons",
                "knowledge_cutoff",
                "metrics",
                "mode",
                "model_frozen_at",
                "outputs",
                "policy_end",
                "policy_evaluation_outcomes",
                "policy_start",
                "randomness_protocol",
                "registered_at",
                "schema_version",
                "study_id",
                "target_event_id",
                "outcome_knowledge_manifest_source_sha256",
                "outcome_knowledge_manifest_semantic_sha256",
            }
        )
        _fields(mapping, expected, "study protocol")
        version = mapping["schema_version"]
        if isinstance(version, bool) or not isinstance(version, int):
            _fail("schema_version must be an integer")

        def definitions(field: str) -> tuple[FrozenDefinition, ...]:
            return tuple(
                FrozenDefinition.from_mapping(_mapping(item, f"{field} item"))
                for item in _sequence(mapping[field], field)
            )

        def identifiers(field: str) -> tuple[str, ...]:
            return tuple(
                _identifier(item, f"{field} item")
                for item in _sequence(mapping[field], field)
            )

        return cls(
            schema_version=version,
            study_id=_identifier(mapping["study_id"], "study_id"),
            mode=cast(StudyMode, _enum(StudyMode, mapping["mode"], "mode")),
            claim_scope=cast(
                ClaimScope,
                _enum(ClaimScope, mapping["claim_scope"], "claim_scope"),
            ),
            registered_at=_aware(mapping["registered_at"], "registered_at"),
            knowledge_cutoff=_aware(mapping["knowledge_cutoff"], "knowledge_cutoff"),
            model_frozen_at=_aware(mapping["model_frozen_at"], "model_frozen_at"),
            policy_start=_aware(mapping["policy_start"], "policy_start"),
            policy_end=_aware(mapping["policy_end"], "policy_end"),
            evaluation_reveal_at=_aware(
                mapping["evaluation_reveal_at"], "evaluation_reveal_at"
            ),
            estimand=FrozenDefinition.from_mapping(
                _mapping(mapping["estimand"], "estimand")
            ),
            evidence_plans=definitions("evidence_plans"),
            baseline_policy=FrozenDefinition.from_mapping(
                _mapping(mapping["baseline_policy"], "baseline_policy")
            ),
            candidate_policy=FrozenDefinition.from_mapping(
                _mapping(mapping["candidate_policy"], "candidate_policy")
            ),
            cohorts=tuple(
                FrozenCohort.from_mapping(_mapping(item, "cohorts item"))
                for item in _sequence(mapping["cohorts"], "cohorts")
            ),
            horizons=definitions("horizons"),
            metrics=definitions("metrics"),
            baselines=definitions("baselines"),
            outputs=tuple(
                RegisteredOutput.from_mapping(_mapping(item, "outputs item"))
                for item in _sequence(mapping["outputs"], "outputs")
            ),
            calibration_statistics=identifiers("calibration_statistics"),
            heldout_validation_statistics=identifiers("heldout_validation_statistics"),
            policy_evaluation_outcomes=identifiers("policy_evaluation_outcomes"),
            failure_thresholds=tuple(
                FailureThreshold.from_mapping(_mapping(item, "failure_thresholds item"))
                for item in _sequence(
                    mapping["failure_thresholds"], "failure_thresholds"
                )
            ),
            abstention_rule=FrozenDefinition.from_mapping(
                _mapping(mapping["abstention_rule"], "abstention_rule")
            ),
            randomness_protocol=cast(
                RandomnessProtocol,
                _enum(
                    RandomnessProtocol,
                    mapping["randomness_protocol"],
                    "randomness_protocol",
                ),
            ),
            target_event_id=(
                None
                if mapping["target_event_id"] is None
                else _identifier(mapping["target_event_id"], "target_event_id")
            ),
            outcome_knowledge_manifest_source_sha256=_optional_sha256(
                mapping["outcome_knowledge_manifest_source_sha256"],
                "outcome_knowledge_manifest_source_sha256",
            ),
            outcome_knowledge_manifest_semantic_sha256=_optional_sha256(
                mapping["outcome_knowledge_manifest_semantic_sha256"],
                "outcome_knowledge_manifest_semantic_sha256",
            ),
        )


@dataclass(frozen=True, slots=True)
class LoadedStudyProtocol:
    """Protocol plus exact source and semantic commitments."""

    protocol: StudyProtocol
    source_sha256: str
    semantic_sha256: str

    def __post_init__(self) -> None:
        _sha256(self.source_sha256, "source_sha256")
        _sha256(self.semantic_sha256, "semantic_sha256")
        if self.protocol.sha256 != self.semantic_sha256:
            _fail("loaded protocol semantic hash does not match the protocol")


def load_study_protocol(
    path: Path, *, expected_source_sha256: str
) -> LoadedStudyProtocol:
    """Load exact bytes only when an external source commitment matches."""
    expected = _sha256(expected_source_sha256, "expected_source_sha256")
    try:
        content = path.read_bytes()
    except OSError as error:
        raise ProtocolValidationError(f"cannot read study protocol: {error}") from error
    observed = sha256_bytes(content)
    if observed != expected:
        _fail("study protocol source hash does not match its external commitment")
    try:
        mapping = read_json_bytes_object(content, source=str(path))
    except MetadataValidationError as error:
        raise ProtocolValidationError(str(error)) from error
    protocol = StudyProtocol.from_mapping(mapping)
    return LoadedStudyProtocol(
        protocol=protocol,
        source_sha256=observed,
        semantic_sha256=protocol.sha256,
    )


_PURPOSE_BY_CONSUMER = {
    ProtocolExecutionConsumer.CALIBRATION_WORKFLOW: (
        ProtocolExecutionPurpose.PRE_POLICY_CALIBRATION
    ),
    ProtocolExecutionConsumer.INTEGRATED_SYNTHETIC_SIMULATION: (
        ProtocolExecutionPurpose.PAIRED_SYNTHETIC_POLICY_SIMULATION
    ),
    ProtocolExecutionConsumer.MECHANISM_ABLATION_WORKFLOW: (
        ProtocolExecutionPurpose.SYNTHETIC_MECHANISM_ABLATION
    ),
    ProtocolExecutionConsumer.EXTENDED_ABLATION_WORKFLOW: (
        ProtocolExecutionPurpose.SYNTHETIC_EXTENDED_ABLATION
    ),
    ProtocolExecutionConsumer.LEDGER_POSTERIOR_EVALUATION: (
        ProtocolExecutionPurpose.SYNTHETIC_LEDGER_POSTERIOR_EVALUATION
    ),
    ProtocolExecutionConsumer.LEDGER_FACTORIAL_EVALUATION: (
        ProtocolExecutionPurpose.SYNTHETIC_LEDGER_FACTORIAL_EVALUATION
    ),
    ProtocolExecutionConsumer.POLICY_FUNCTIONAL_META_EVALUATION: (
        ProtocolExecutionPurpose.SYNTHETIC_POLICY_FUNCTIONAL_META_EVALUATION
    ),
    ProtocolExecutionConsumer.EMPIRICAL_FROZEN_EVALUATION: (
        ProtocolExecutionPurpose.POLICY_POST_EMPIRICAL_EVALUATION
    ),
}


@dataclass(frozen=True, slots=True)
class ProtocolExecutionReceipt:
    """Serializable evidence for one authorized protocol-bound execution.

    A receipt is audit evidence only.  It cannot authorize an execution; the
    corresponding opaque :class:`ProtocolExecutionAuthorization` must still be
    present in the private issuance registry and is consumed exactly once.
    """

    schema_version: int
    execution_id: str
    issuer_kind: str
    protocol_sha256: str
    study_id: str
    study_mode: StudyMode
    claim_scope: ClaimScope
    knowledge_cutoff: datetime
    model_frozen_at: datetime
    registered_at: datetime
    policy_start: datetime
    policy_end: datetime
    evaluation_reveal_at: datetime
    consumer: ProtocolExecutionConsumer
    purpose: ProtocolExecutionPurpose
    config_sha256: str
    input_sha256: str
    execution_started_at: datetime

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            _fail("protocol execution receipt schema_version must equal 1")
        _identifier(self.execution_id, "execution_id")
        if self.issuer_kind != "synthetic_fixture_issuer":
            _fail("unsupported protocol execution receipt issuer")
        _sha256(self.protocol_sha256, "protocol_sha256")
        _identifier(self.study_id, "study_id")
        if self.study_mode is not StudyMode.SYNTHETIC:
            _fail("synthetic fixture receipts cannot claim a real study mode")
        if self.claim_scope is not ClaimScope.SYNTHETIC_ONLY:
            _fail("synthetic fixture receipts cannot authorize a non-synthetic claim")
        for field in (
            "knowledge_cutoff",
            "model_frozen_at",
            "registered_at",
            "policy_start",
            "policy_end",
            "evaluation_reveal_at",
            "execution_started_at",
        ):
            object.__setattr__(self, field, _aware(getattr(self, field), field))
        if not (
            self.knowledge_cutoff
            <= self.model_frozen_at
            <= self.registered_at
            < self.policy_start
            <= self.policy_end
            < self.evaluation_reveal_at
        ):
            _fail("protocol execution receipt contains invalid study chronology")
        if not isinstance(self.consumer, ProtocolExecutionConsumer):
            _fail("consumer must be a ProtocolExecutionConsumer")
        if not isinstance(self.purpose, ProtocolExecutionPurpose):
            _fail("purpose must be a ProtocolExecutionPurpose")
        if _PURPOSE_BY_CONSUMER.get(self.consumer) is not self.purpose:
            _fail("protocol execution consumer and purpose are incompatible")
        _sha256(self.config_sha256, "config_sha256")
        _sha256(self.input_sha256, "input_sha256")
        _validate_protocol_execution_time(
            consumer=self.consumer,
            execution_started_at=self.execution_started_at,
            model_frozen_at=self.model_frozen_at,
            registered_at=self.registered_at,
            evaluation_reveal_at=self.evaluation_reveal_at,
        )

    @property
    def sha256(self) -> str:
        """Commit to every authorization binding and chronology field."""
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic audit evidence; this is never authority."""
        return {
            "claim_scope": self.claim_scope.value,
            "config_sha256": self.config_sha256,
            "consumer": self.consumer.value,
            "evaluation_reveal_at": self.evaluation_reveal_at.isoformat(),
            "execution_id": self.execution_id,
            "execution_started_at": self.execution_started_at.isoformat(),
            "input_sha256": self.input_sha256,
            "issuer_kind": self.issuer_kind,
            "knowledge_cutoff": self.knowledge_cutoff.isoformat(),
            "model_frozen_at": self.model_frozen_at.isoformat(),
            "policy_end": self.policy_end.isoformat(),
            "policy_start": self.policy_start.isoformat(),
            "protocol_sha256": self.protocol_sha256,
            "purpose": self.purpose.value,
            "registered_at": self.registered_at.isoformat(),
            "schema_version": self.schema_version,
            "study_id": self.study_id,
            "study_mode": self.study_mode.value,
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> ProtocolExecutionReceipt:
        """Strictly reconstruct audit evidence without minting authority."""
        _fields(
            mapping,
            frozenset(
                {
                    "claim_scope",
                    "config_sha256",
                    "consumer",
                    "evaluation_reveal_at",
                    "execution_id",
                    "execution_started_at",
                    "input_sha256",
                    "issuer_kind",
                    "knowledge_cutoff",
                    "model_frozen_at",
                    "policy_end",
                    "policy_start",
                    "protocol_sha256",
                    "purpose",
                    "registered_at",
                    "schema_version",
                    "study_id",
                    "study_mode",
                }
            ),
            "protocol execution receipt",
        )
        version = mapping["schema_version"]
        if isinstance(version, bool) or not isinstance(version, int):
            _fail("protocol execution receipt schema_version must be an integer")
        issuer = mapping["issuer_kind"]
        if not isinstance(issuer, str):
            _fail("issuer_kind must be a string")
        return cls(
            schema_version=version,
            execution_id=_identifier(mapping["execution_id"], "execution_id"),
            issuer_kind=issuer,
            protocol_sha256=_sha256(mapping["protocol_sha256"], "protocol_sha256"),
            study_id=_identifier(mapping["study_id"], "study_id"),
            study_mode=cast(
                StudyMode, _enum(StudyMode, mapping["study_mode"], "study_mode")
            ),
            claim_scope=cast(
                ClaimScope,
                _enum(ClaimScope, mapping["claim_scope"], "claim_scope"),
            ),
            knowledge_cutoff=_aware(mapping["knowledge_cutoff"], "knowledge_cutoff"),
            model_frozen_at=_aware(mapping["model_frozen_at"], "model_frozen_at"),
            registered_at=_aware(mapping["registered_at"], "registered_at"),
            policy_start=_aware(mapping["policy_start"], "policy_start"),
            policy_end=_aware(mapping["policy_end"], "policy_end"),
            evaluation_reveal_at=_aware(
                mapping["evaluation_reveal_at"], "evaluation_reveal_at"
            ),
            consumer=cast(
                ProtocolExecutionConsumer,
                _enum(
                    ProtocolExecutionConsumer,
                    mapping["consumer"],
                    "consumer",
                ),
            ),
            purpose=cast(
                ProtocolExecutionPurpose,
                _enum(
                    ProtocolExecutionPurpose,
                    mapping["purpose"],
                    "purpose",
                ),
            ),
            config_sha256=_sha256(mapping["config_sha256"], "config_sha256"),
            input_sha256=_sha256(mapping["input_sha256"], "input_sha256"),
            execution_started_at=_aware(
                mapping["execution_started_at"], "execution_started_at"
            ),
        )


def _validate_protocol_execution_time(
    *,
    consumer: ProtocolExecutionConsumer,
    execution_started_at: datetime,
    model_frozen_at: datetime,
    registered_at: datetime,
    evaluation_reveal_at: datetime,
) -> None:
    if consumer is ProtocolExecutionConsumer.CALIBRATION_WORKFLOW:
        if execution_started_at > model_frozen_at:
            _fail("calibration authorization cannot be used after model freeze")
        return
    if consumer in {
        ProtocolExecutionConsumer.INTEGRATED_SYNTHETIC_SIMULATION,
        ProtocolExecutionConsumer.LEDGER_POSTERIOR_EVALUATION,
        ProtocolExecutionConsumer.LEDGER_FACTORIAL_EVALUATION,
        ProtocolExecutionConsumer.POLICY_FUNCTIONAL_META_EVALUATION,
    }:
        if execution_started_at < registered_at:
            _fail("synthetic evaluation cannot start before registration")
        if execution_started_at >= evaluation_reveal_at:
            _fail("synthetic evaluation must start before the registered reveal")
        return
    if consumer in {
        ProtocolExecutionConsumer.MECHANISM_ABLATION_WORKFLOW,
        ProtocolExecutionConsumer.EXTENDED_ABLATION_WORKFLOW,
    }:
        if execution_started_at < registered_at:
            _fail("synthetic ablation cannot start before registration")
        if execution_started_at >= evaluation_reveal_at:
            _fail("synthetic ablation must start before the registered reveal")
        return
    if consumer is ProtocolExecutionConsumer.EMPIRICAL_FROZEN_EVALUATION:
        if execution_started_at < evaluation_reveal_at:
            _fail("empirical evaluation cannot start before the registered reveal")
        return
    _fail("unsupported protocol execution consumer")


def validate_protocol_execution_receipt_binding(
    protocol: StudyProtocol,
    receipt: ProtocolExecutionReceipt,
) -> None:
    """Validate stored receipt evidence against every duplicated protocol field."""
    if not isinstance(protocol, StudyProtocol):
        _fail("protocol receipt binding requires a typed StudyProtocol")
    if not isinstance(receipt, ProtocolExecutionReceipt):
        _fail("protocol receipt binding requires a typed execution receipt")
    if protocol.sha256 != receipt.protocol_sha256:
        _fail("protocol execution receipt is bound to a different protocol")
    if (
        protocol.study_id != receipt.study_id
        or protocol.mode is not receipt.study_mode
        or protocol.claim_scope is not receipt.claim_scope
        or protocol.knowledge_cutoff != receipt.knowledge_cutoff
        or protocol.model_frozen_at != receipt.model_frozen_at
        or protocol.registered_at != receipt.registered_at
        or protocol.policy_start != receipt.policy_start
        or protocol.policy_end != receipt.policy_end
        or protocol.evaluation_reveal_at != receipt.evaluation_reveal_at
    ):
        _fail("protocol execution receipt chronology or claim scope is inconsistent")
    _validate_protocol_execution_time(
        consumer=receipt.consumer,
        execution_started_at=receipt.execution_started_at,
        model_frozen_at=protocol.model_frozen_at,
        registered_at=protocol.registered_at,
        evaluation_reveal_at=protocol.evaluation_reveal_at,
    )


@dataclass(frozen=True, slots=True)
class _ProtocolExecutionAuthorizationMint:
    """Private canonical binding retained independently of the capability."""

    protocol_source: bytes
    receipt_source: bytes
    seal_sha256: str


def _protocol_authorization_mint_seal(
    *, protocol_source: bytes, receipt_source: bytes
) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "protocol_source_sha256": sha256_bytes(protocol_source),
                "receipt_source_sha256": sha256_bytes(receipt_source),
                "schema_version": 1,
            }
        )
    )


def _make_protocol_authorization_mint(
    protocol: StudyProtocol,
    receipt: ProtocolExecutionReceipt,
) -> _ProtocolExecutionAuthorizationMint:
    protocol_source = canonical_json_bytes(protocol.to_mapping())
    receipt_source = canonical_json_bytes(receipt.to_mapping())
    return _ProtocolExecutionAuthorizationMint(
        protocol_source=protocol_source,
        receipt_source=receipt_source,
        seal_sha256=_protocol_authorization_mint_seal(
            protocol_source=protocol_source,
            receipt_source=receipt_source,
        ),
    )


def _decode_protocol_authorization_mint(
    mint: _ProtocolExecutionAuthorizationMint,
) -> tuple[StudyProtocol, ProtocolExecutionReceipt]:
    expected_seal = _protocol_authorization_mint_seal(
        protocol_source=mint.protocol_source,
        receipt_source=mint.receipt_source,
    )
    if mint.seal_sha256 != expected_seal:
        _fail("protocol execution authorization mint seal does not match")
    try:
        protocol_mapping = read_json_bytes_object(
            mint.protocol_source,
            source="protocol execution authorization mint protocol",
        )
        receipt_mapping = read_json_bytes_object(
            mint.receipt_source,
            source="protocol execution authorization mint receipt",
        )
    except MetadataValidationError as error:
        raise ProtocolValidationError(
            f"protocol execution authorization mint is invalid: {error}"
        ) from error
    if (
        canonical_json_bytes(protocol_mapping) != mint.protocol_source
        or canonical_json_bytes(receipt_mapping) != mint.receipt_source
    ):
        _fail("protocol execution authorization mint is not canonical")
    protocol = StudyProtocol.from_mapping(protocol_mapping)
    receipt = ProtocolExecutionReceipt.from_mapping(receipt_mapping)
    validate_protocol_execution_receipt_binding(protocol, receipt)
    return protocol, receipt


# Active membership is the minting proof.  The immutable mint record lives in a
# separate weak-key registry, so mutating a receipt copy or the capability object
# cannot redirect its binding.  The lock makes issue/consume registry transitions
# atomic.  This is an API trust boundary, not a same-process sandbox.
_PROTOCOL_AUTHORIZATION_REGISTRY_LOCK = Lock()
_ISSUED_PROTOCOL_AUTHORIZATIONS: set[ProtocolExecutionAuthorization] = set()
_MINTED_PROTOCOL_AUTHORIZATIONS: WeakKeyDictionary[
    ProtocolExecutionAuthorization, _ProtocolExecutionAuthorizationMint
] = WeakKeyDictionary()


def _reject_protocol_authorization_construction() -> NoReturn:
    _fail(
        "protocol execution authorization is opaque and may only be issued by "
        "a trusted protocol execution issuer"
    )


class ProtocolExecutionAuthorization:
    """Opaque, single-use authority for one exact protocol/config/input tuple."""

    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        _reject_protocol_authorization_construction()

    def __copy__(self) -> NoReturn:
        _reject_protocol_authorization_construction()

    def __deepcopy__(self, memo: object) -> NoReturn:
        del memo
        _reject_protocol_authorization_construction()

    def __reduce__(self) -> NoReturn:
        raise TypeError("ProtocolExecutionAuthorization cannot be serialized")

    @property
    def receipt(self) -> ProtocolExecutionReceipt:
        """Return strictly reconstructed evidence, never the private mint binding."""
        with _PROTOCOL_AUTHORIZATION_REGISTRY_LOCK:
            mint = _MINTED_PROTOCOL_AUTHORIZATIONS.get(self)
            if mint is None:
                _fail("protocol execution authorization was not issued")
            _, receipt = _decode_protocol_authorization_mint(mint)
        return receipt


class SyntheticProtocolExecutionIssuer:
    """Explicit fixture-only issuer; real protocols always fail closed here.

    A real issuer must verify an externally anchored, pre-reveal registration
    and re-load the exact committed protocol bytes.  That trust root is not
    available in this repository, so this class categorically rejects real
    study modes instead of accepting caller booleans or receipt hashes.
    """

    __slots__ = ("_execution_ids", "_lock", "_protocol")

    def __init__(self, protocol: StudyProtocol) -> None:
        if not isinstance(protocol, StudyProtocol):
            _fail("synthetic protocol issuer requires a typed StudyProtocol")
        if protocol.mode is not StudyMode.SYNTHETIC:
            _fail(
                "real protocol execution requires an external preregistration "
                "verifier, which is unavailable"
            )
        if protocol.claim_scope is not ClaimScope.SYNTHETIC_ONLY:
            _fail("synthetic protocol issuer cannot authorize real claims")
        self._protocol = protocol
        self._execution_ids: set[str] = set()
        self._lock = Lock()

    def issue(
        self,
        *,
        execution_id: str,
        consumer: ProtocolExecutionConsumer,
        purpose: ProtocolExecutionPurpose,
        config_sha256: str,
        input_sha256: str,
        execution_started_at: datetime,
    ) -> ProtocolExecutionAuthorization:
        """Mint one non-serializable authority for an exact execution tuple."""
        identifier = _identifier(execution_id, "execution_id")
        if not isinstance(consumer, ProtocolExecutionConsumer):
            _fail("consumer must be a ProtocolExecutionConsumer")
        if not isinstance(purpose, ProtocolExecutionPurpose):
            _fail("purpose must be a ProtocolExecutionPurpose")
        if _PURPOSE_BY_CONSUMER.get(consumer) is not purpose:
            _fail("protocol execution consumer and purpose are incompatible")
        execution_time = _aware(execution_started_at, "execution_started_at")
        receipt = ProtocolExecutionReceipt(
            schema_version=1,
            execution_id=identifier,
            issuer_kind="synthetic_fixture_issuer",
            protocol_sha256=self._protocol.sha256,
            study_id=self._protocol.study_id,
            study_mode=self._protocol.mode,
            claim_scope=self._protocol.claim_scope,
            knowledge_cutoff=self._protocol.knowledge_cutoff,
            model_frozen_at=self._protocol.model_frozen_at,
            registered_at=self._protocol.registered_at,
            policy_start=self._protocol.policy_start,
            policy_end=self._protocol.policy_end,
            evaluation_reveal_at=self._protocol.evaluation_reveal_at,
            consumer=consumer,
            purpose=purpose,
            config_sha256=_sha256(config_sha256, "config_sha256"),
            input_sha256=_sha256(input_sha256, "input_sha256"),
            execution_started_at=execution_time,
        )
        mint = _make_protocol_authorization_mint(self._protocol, receipt)
        authorization = object.__new__(ProtocolExecutionAuthorization)
        with self._lock:
            if identifier in self._execution_ids:
                _fail("protocol execution_id has already been issued by this issuer")
            with _PROTOCOL_AUTHORIZATION_REGISTRY_LOCK:
                _MINTED_PROTOCOL_AUTHORIZATIONS[authorization] = mint
                _ISSUED_PROTOCOL_AUTHORIZATIONS.add(authorization)
            self._execution_ids.add(identifier)
        return authorization


def consume_protocol_execution_authorization(
    authorization: ProtocolExecutionAuthorization,
    *,
    protocol: StudyProtocol,
    consumer: ProtocolExecutionConsumer,
    purpose: ProtocolExecutionPurpose,
    config_sha256: str,
    input_sha256: str,
    execution_started_at: datetime,
) -> ProtocolExecutionReceipt:
    """Consume and revalidate one exact execution authority.

    The registry entry is removed before any caller-supplied binding is checked.
    A wrong protocol, config, input, time, consumer, or purpose therefore burns
    the capability and cannot be corrected after observing a rejection.
    """
    if type(authorization) is not ProtocolExecutionAuthorization:
        _fail("protocol execution authorization has an invalid type")
    with _PROTOCOL_AUTHORIZATION_REGISTRY_LOCK:
        if authorization not in _ISSUED_PROTOCOL_AUTHORIZATIONS:
            _fail(
                "protocol execution authorization was not issued by the trusted "
                "issuer or has already been consumed"
            )
        try:
            mint = _MINTED_PROTOCOL_AUTHORIZATIONS.get(authorization)
            if mint is None:
                _fail("protocol execution authorization mint binding is unavailable")
            minted_protocol, receipt = _decode_protocol_authorization_mint(mint)
        finally:
            # Burn while holding the same lock used for the membership check.
            # Seal failures and all later caller-binding failures remain single-use.
            _ISSUED_PROTOCOL_AUTHORIZATIONS.discard(authorization)
    if not isinstance(protocol, StudyProtocol):
        _fail("protocol execution requires a typed StudyProtocol")
    if (
        protocol.sha256 != receipt.protocol_sha256
        or protocol.to_mapping() != minted_protocol.to_mapping()
    ):
        _fail("protocol execution authorization is bound to a different protocol")
    validate_protocol_execution_receipt_binding(protocol, receipt)
    if consumer is not receipt.consumer or purpose is not receipt.purpose:
        _fail("protocol execution authorization cannot be replayed across consumers")
    if _sha256(config_sha256, "config_sha256") != receipt.config_sha256:
        _fail("protocol execution config differs from the authorization")
    if _sha256(input_sha256, "input_sha256") != receipt.input_sha256:
        _fail("protocol execution input differs from the authorization")
    started_at = _aware(execution_started_at, "execution_started_at")
    if started_at != receipt.execution_started_at:
        _fail("protocol execution time differs from the authorization")
    return receipt


@dataclass(frozen=True, slots=True)
class ReleaseEvidence:
    """Declared commitments and verified audit outcomes for one sealed run."""

    protocol_sha256: str
    git_commit: str | None
    git_dirty: bool
    environment_lock_sha256: str | None
    run_config_sha256: str | None
    data_manifest_sha256: str | None
    model_checkpoint_sha256: str | None
    event_key_manifest_sha256: str | None
    run_manifest_sha256: str | None
    result_artifact_sha256: str | None
    temporal_audit_passed: bool
    validity_audit_passed: bool
    registered_rows_total: int
    registered_rows_accounted: int
    failed_rows_reported: bool
    primary_policy_sources_verified: bool
    data_licenses_verified: bool
    post_policy_inputs_frozen: bool
    external_preregistration_receipt_sha256: str | None
    preregistration_timestamp_verified: bool
    empirical_identification_audit_sha256: str | None
    empirical_identification_passed: bool

    def __post_init__(self) -> None:
        _sha256(self.protocol_sha256, "protocol_sha256")
        if (
            self.git_commit is not None
            and _GIT_COMMIT.fullmatch(self.git_commit) is None
        ):
            _fail("git_commit must be null or a full lowercase Git object ID")
        for field in (
            "git_dirty",
            "temporal_audit_passed",
            "validity_audit_passed",
            "failed_rows_reported",
            "primary_policy_sources_verified",
            "data_licenses_verified",
            "post_policy_inputs_frozen",
            "preregistration_timestamp_verified",
            "empirical_identification_passed",
        ):
            if not isinstance(getattr(self, field), bool):
                _fail(f"{field} must be boolean")
        for field in (
            "environment_lock_sha256",
            "run_config_sha256",
            "data_manifest_sha256",
            "model_checkpoint_sha256",
            "event_key_manifest_sha256",
            "run_manifest_sha256",
            "result_artifact_sha256",
            "external_preregistration_receipt_sha256",
            "empirical_identification_audit_sha256",
        ):
            _optional_sha256(getattr(self, field), field)
        for field in ("registered_rows_total", "registered_rows_accounted"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                _fail(f"{field} must be a non-negative integer")
        if self.registered_rows_accounted > self.registered_rows_total:
            _fail("accounted rows cannot exceed registered rows")


@dataclass(frozen=True, slots=True)
class ReleaseEligibility:
    """Deterministic decision separating integrity from claim eligibility."""

    protocol_sha256: str
    integrity_valid: bool
    formal_reporting_eligible: bool
    allowed_claim_scope: ClaimScope | None
    integrity_errors: tuple[str, ...]
    eligibility_errors: tuple[str, ...]
    outcome_knowledge_manifest_source_sha256: str | None
    outcome_knowledge_manifest_semantic_sha256: str | None
    design_temporality_decision_sha256: str | None
    derived_design_temporality: str | None

    def __post_init__(self) -> None:
        _sha256(self.protocol_sha256, "protocol_sha256")
        if self.formal_reporting_eligible and not self.integrity_valid:
            _fail("formal reporting eligibility requires valid integrity")
        if self.formal_reporting_eligible != (self.allowed_claim_scope is not None):
            _fail("allowed claim scope must exist exactly for eligible releases")
        commitments = (
            self.outcome_knowledge_manifest_source_sha256,
            self.outcome_knowledge_manifest_semantic_sha256,
            self.design_temporality_decision_sha256,
            self.derived_design_temporality,
        )
        if any(value is None for value in commitments) and any(
            value is not None for value in commitments
        ):
            _fail("outcome-knowledge release commitments must be present together")
        for field in (
            "outcome_knowledge_manifest_source_sha256",
            "outcome_knowledge_manifest_semantic_sha256",
            "design_temporality_decision_sha256",
        ):
            _optional_sha256(getattr(self, field), field)
        if self.derived_design_temporality is not None:
            _identifier(
                self.derived_design_temporality,
                "derived_design_temporality",
            )

    def to_mapping(self) -> dict[str, object]:
        """Return a machine-readable decision for audit artifacts."""
        return {
            "allowed_claim_scope": (
                None
                if self.allowed_claim_scope is None
                else self.allowed_claim_scope.value
            ),
            "eligibility_errors": list(self.eligibility_errors),
            "formal_reporting_eligible": self.formal_reporting_eligible,
            "integrity_errors": list(self.integrity_errors),
            "integrity_valid": self.integrity_valid,
            "outcome_knowledge_manifest_semantic_sha256": (
                self.outcome_knowledge_manifest_semantic_sha256
            ),
            "outcome_knowledge_manifest_source_sha256": (
                self.outcome_knowledge_manifest_source_sha256
            ),
            "design_temporality_decision_sha256": (
                self.design_temporality_decision_sha256
            ),
            "derived_design_temporality": self.derived_design_temporality,
            "protocol_sha256": self.protocol_sha256,
            "schema_version": 2,
        }


def evaluate_release_eligibility(
    protocol: StudyProtocol,
    evidence: ReleaseEvidence,
    *,
    outcome_knowledge_manifest_path: Path | None = None,
) -> ReleaseEligibility:
    """Fail closed unless complete run evidence supports the registered claim.

    A caller-supplied :class:`DesignTemporalityDecision` is intentionally not
    accepted.  For a real study this boundary rereads the exact manifest bytes
    committed by the externally anchored protocol and derives the decision
    itself.  The serialized decision remains non-authorizing audit evidence.
    """
    integrity: list[str] = []
    eligibility: list[str] = []
    outcome_source_sha256: str | None = None
    outcome_semantic_sha256: str | None = None
    temporality_decision_sha256: str | None = None
    derived_temporality: str | None = None
    if evidence.protocol_sha256 != protocol.sha256:
        integrity.append("run evidence is bound to a different study protocol")
    for field in (
        "environment_lock_sha256",
        "run_config_sha256",
        "data_manifest_sha256",
        "model_checkpoint_sha256",
        "event_key_manifest_sha256",
        "run_manifest_sha256",
        "result_artifact_sha256",
    ):
        if getattr(evidence, field) is None:
            integrity.append(f"missing required commitment: {field}")
    if not evidence.temporal_audit_passed:
        integrity.append("temporal leakage audit did not pass")
    if not evidence.validity_audit_passed:
        integrity.append("validity audit did not pass")
    if evidence.registered_rows_total == 0:
        integrity.append("no experiment rows were registered")
    if evidence.registered_rows_accounted != evidence.registered_rows_total:
        integrity.append("not every registered experiment row is accounted for")
    if not evidence.failed_rows_reported:
        integrity.append("failed experiment rows were not retained")
    if evidence.git_commit is None:
        eligibility.append("release is not traceable to a Git commit")
    if evidence.git_dirty:
        eligibility.append("release was produced from a dirty worktree")
    if protocol.mode is StudyMode.SYNTHETIC:
        if outcome_knowledge_manifest_path is not None:
            integrity.append(
                "synthetic release must not consume a real-event outcome ledger"
            )
    else:
        from policytwin.outcome_knowledge import (
            DesignTemporality,
            OutcomeKnowledgeError,
            assess_design_temporality,
            load_outcome_knowledge_manifest,
        )

        eligibility.append(
            "real-study formal release requires typed verifier receipts; "
            "declarative ReleaseEvidence cannot authorize it"
        )
        decision = None
        if outcome_knowledge_manifest_path is None:
            eligibility.append("outcome-knowledge temporality audit is missing")
        else:
            try:
                expected_source = _sha256(
                    protocol.outcome_knowledge_manifest_source_sha256,
                    "outcome_knowledge_manifest_source_sha256",
                )
                expected_semantic = _sha256(
                    protocol.outcome_knowledge_manifest_semantic_sha256,
                    "outcome_knowledge_manifest_semantic_sha256",
                )
                event_id = _identifier(protocol.target_event_id, "target_event_id")
                loaded_manifest = load_outcome_knowledge_manifest(
                    outcome_knowledge_manifest_path,
                    expected_source_sha256=expected_source,
                )
                if loaded_manifest.semantic_sha256 != expected_semantic:
                    integrity.append(
                        "outcome-knowledge manifest semantic commitment mismatch"
                    )
                else:
                    decision = assess_design_temporality(
                        protocol,
                        target_event_id=event_id,
                        manifest=loaded_manifest,
                    )
                    outcome_source_sha256 = loaded_manifest.source_sha256
                    outcome_semantic_sha256 = loaded_manifest.semantic_sha256
                    temporality_decision_sha256 = decision.sha256
                    derived_temporality = decision.derived_temporality.value
            except OutcomeKnowledgeError as error:
                integrity.append(f"outcome-knowledge temporality audit failed: {error}")
        if decision is not None:
            if protocol.mode is StudyMode.PSEUDO_PROSPECTIVE and (
                decision.derived_temporality
                is not DesignTemporality.PSEUDO_PROSPECTIVE_BLINDED
                or not decision.pseudo_prospective_eligible
            ):
                eligibility.append(
                    "pseudo-prospective design lacks externally verified blinding"
                )
            if protocol.mode is StudyMode.RETROSPECTIVE_OUTCOME_KNOWN and (
                decision.derived_temporality
                is not DesignTemporality.RETROSPECTIVE_OUTCOME_KNOWN
            ):
                eligibility.append(
                    "retrospective design lacks a matching outcome-exposure audit"
                )
        if evidence.external_preregistration_receipt_sha256 is None:
            eligibility.append("external preregistration commitment is missing")
        if not evidence.preregistration_timestamp_verified:
            eligibility.append(
                "pre-reveal registration time is not externally verified"
            )
        if not evidence.primary_policy_sources_verified:
            eligibility.append("primary policy sources are not verified")
        if not evidence.data_licenses_verified:
            eligibility.append("data licenses are not verified")
        if not evidence.post_policy_inputs_frozen:
            eligibility.append("policy-post evaluation inputs are not frozen")
        if protocol.claim_scope is ClaimScope.REAL_CAUSAL:
            if evidence.empirical_identification_audit_sha256 is None:
                eligibility.append("empirical identification audit is missing")
            if not evidence.empirical_identification_passed:
                eligibility.append("empirical identification audit did not pass")
    integrity_valid = not integrity
    formal = integrity_valid and not eligibility
    return ReleaseEligibility(
        protocol_sha256=protocol.sha256,
        integrity_valid=integrity_valid,
        formal_reporting_eligible=formal,
        allowed_claim_scope=protocol.claim_scope if formal else None,
        integrity_errors=tuple(integrity),
        eligibility_errors=tuple(eligibility),
        outcome_knowledge_manifest_source_sha256=outcome_source_sha256,
        outcome_knowledge_manifest_semantic_sha256=outcome_semantic_sha256,
        design_temporality_decision_sha256=temporality_decision_sha256,
        derived_design_temporality=derived_temporality,
    )


__all__ = [
    "ClaimScope",
    "CohortBasis",
    "FailureThreshold",
    "FrozenCohort",
    "FrozenDefinition",
    "LoadedStudyProtocol",
    "OutputRole",
    "ProtocolExecutionAuthorization",
    "ProtocolExecutionConsumer",
    "ProtocolExecutionPurpose",
    "ProtocolExecutionReceipt",
    "ProtocolValidationError",
    "RandomnessProtocol",
    "RegisteredOutput",
    "ReleaseEligibility",
    "ReleaseEvidence",
    "StudyMode",
    "StudyProtocol",
    "SyntheticProtocolExecutionIssuer",
    "consume_protocol_execution_authorization",
    "evaluate_release_eligibility",
    "load_study_protocol",
    "validate_protocol_execution_receipt_binding",
]
