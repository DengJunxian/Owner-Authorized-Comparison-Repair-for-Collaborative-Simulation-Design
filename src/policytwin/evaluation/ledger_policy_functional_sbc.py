"""Exact synthetic-only core for paired policy-functional SBC.

This module deliberately starts *after* evidence production.  It accepts typed
exact scenario values and commitments, derives randomized ranks, and audits a
fixed set of controlled canaries.  It does not run an oracle, verify a rooted
evidence bundle, establish external registration, or support real-world
coverage or causal claims.

Every ``sha256`` exposed here is a content identifier for the caller-presented
value at hashing time.  It is not an issuance-time seal, custody receipt, or
tamper-evident identity for a previously issued result.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from fractions import Fraction
from typing import Literal, NoReturn

from policytwin.counterfactual.randomness import EventKey, deterministic_int
from policytwin.evaluation.ledger_derived import (
    ExactRational,
    LedgerDerivedEvidenceError,
)
from policytwin.provenance import canonical_json_bytes, sha256_bytes

POLICY_FUNCTIONAL_SBC_SCHEMA_VERSION = "policytwin.policy-functional-sbc.v3"
POLICY_FUNCTIONAL_SBC_CLAIM_SCOPE: Literal[
    "synthetic_policy_functional_sbc_core_arithmetic_only"
] = "synthetic_policy_functional_sbc_core_arithmetic_only"
POLICY_FUNCTIONAL_SBC_SCENARIO_FUNCTIONAL: Literal[
    "exact_arithmetic_mean_over_caller_committed_scenario_replicates"
] = "exact_arithmetic_mean_over_caller_committed_scenario_replicates"
POLICY_FUNCTIONAL_SBC_TIE_RULE: Literal["uniform_integer_over_exact_tie_positions"] = (
    "uniform_integer_over_exact_tie_positions"
)
POLICY_FUNCTIONAL_SBC_ESTIMAND_SCOPE: Literal[
    "caller_committed_finite_tape_g_omega_theta_shared_commitment_equality_only_consumption_unverified"
] = (
    "caller_committed_finite_tape_g_omega_theta_"
    "shared_commitment_equality_only_consumption_unverified"
)
POLICY_FUNCTIONAL_SBC_TIE_DOMAIN: Literal[
    "policy-functional-sbc-randomized-exact-tie-v1"
] = "policy-functional-sbc-randomized-exact-tie-v1"

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EVENT_KEY = re.compile(r"^ptk1:[0-9a-f]{64}$")
_MAX_CELLS = 16_384


class PolicyFunctionalSBCError(ValueError):
    """Raised when the pure policy-functional SBC contract is violated."""


def _fail(message: str) -> NoReturn:
    raise PolicyFunctionalSBCError(message)


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        _fail(f"{field} must be a stable ASCII identifier")
    return value


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail(f"{field} must be a lowercase SHA-256 digest")
    return value


def _positive_int(value: object, field: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"{field} must be an integer of at least {minimum}")
    return value


def _ordinal(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(f"{field} must be a non-negative integer")
    return value


def _reason(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 1024:
        _fail(f"{field} must be bounded non-empty text")
    return value


def _aware_utc(value: object, field: str) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        _fail(f"{field} must be a timezone-aware datetime")
    try:
        return value.astimezone(UTC)
    except (OverflowError, ValueError) as error:
        raise PolicyFunctionalSBCError(f"{field} is outside the UTC range") from error


def _event_keys(values: object, field: str) -> tuple[str, ...]:
    if not isinstance(values, tuple) or not values:
        _fail(f"{field} must be a non-empty immutable tuple")
    if not all(
        isinstance(value, str) and _EVENT_KEY.fullmatch(value) for value in values
    ):
        _fail(f"{field} contains an invalid EventKey commitment")
    normalized = tuple(sorted(values))
    if len(set(normalized)) != len(normalized):
        _fail(f"{field} must not reuse an EventKey")
    return normalized


def _exact(value: object, field: str) -> ExactRational:
    if type(value) is not ExactRational:
        raise TypeError(f"{field} must be an exact ExactRational")
    # ``frozen=True`` is not a security boundary: ``object.__setattr__`` can
    # still corrupt a nested instance.  Reconstructing here gives every fresh
    # validation path the canonical ExactRational checks as well.
    try:
        fresh = ExactRational(value.numerator, value.denominator)
    except (LedgerDerivedEvidenceError, TypeError, ValueError) as error:
        raise PolicyFunctionalSBCError(f"{field} is not canonical") from error
    if fresh != value:
        _fail(f"{field} is not canonical")
    return value


def _fraction(value: ExactRational) -> Fraction:
    _exact(value, "exact value")
    return Fraction(value.numerator, value.denominator)


def _from_fraction(value: Fraction) -> ExactRational:
    return ExactRational.from_ratio(value.numerator, value.denominator)


def _mean(values: tuple[ExactRational, ...]) -> ExactRational:
    if not values:
        _fail("scenario mean requires at least one exact value")
    return _from_fraction(
        sum((_fraction(value) for value in values), Fraction()) / len(values)
    )


class MetaReplicateStatus(StrEnum):
    """Complete accounting state for one caller-planned reference replicate."""

    COMPLETED = "completed"
    FAILED = "failed"
    ABSTAINED = "abstained"


class MetaReplicateFailureStage(StrEnum):
    """Closed stage at which a reference replicate failed or abstained."""

    PRE_POLICY_CALIBRATION = "pre_policy_calibration"
    POSTERIOR_CONSTRUCTION = "posterior_construction"
    ORACLE_FUNCTIONAL_INPUT = "oracle_functional_input"
    POSTERIOR_FUNCTIONAL_INPUT = "posterior_functional_input"
    SCENARIO_TAPE_INTEGRITY = "scenario_tape_integrity"
    RANK_DERIVATION = "rank_derivation"
    CONTROLLED_ABSTENTION = "controlled_abstention"


class CanaryKind(StrEnum):
    """Closed controlled-misspecification canary registry."""

    POSTERIOR_SHIFT = "posterior_shift"
    IGNORED_DATA = "ignored_data"
    RNG_MISMATCH = "rng_mismatch"
    OMITTED_MECHANISM = "omitted_mechanism"


class CanaryDetection(StrEnum):
    """Whether the registered guard detected the controlled violation."""

    DETECTED_FAIL = "detected_fail"
    DETECTED_ABSTAIN = "detected_abstain"
    NOT_DETECTED = "not_detected"

    @property
    def detected(self) -> bool:
        return self in {
            CanaryDetection.DETECTED_FAIL,
            CanaryDetection.DETECTED_ABSTAIN,
        }


class PolicyFunctionalSBCDecision(StrEnum):
    """Synthetic method-check decision, never a publication validity color."""

    SYNTHETIC_CORE_ARITHMETIC_PASS_UNVERIFIED = (
        "synthetic_core_arithmetic_pass_unverified"
    )
    FAIL = "fail"
    ABSTAIN = "abstain"


@dataclass(frozen=True, slots=True)
class CallerDeclaredRankThreshold:
    """Exact threshold with caller-declared, not externally verified, chronology."""

    threshold_id: str
    maximum_rank_cdf_discrepancy: ExactRational
    threshold_design_sha256: str
    generation_event_keys: tuple[str, ...]
    declared_frozen_at: datetime
    declared_audit_outcomes_available_at: datetime
    caller_declared_policy_post_data_used: Literal[False] = False
    externally_registered: Literal[False] = False
    synthetic_only: Literal[True] = True

    def __post_init__(self) -> None:
        _identifier(self.threshold_id, "threshold_id")
        threshold = _exact(
            self.maximum_rank_cdf_discrepancy,
            "maximum_rank_cdf_discrepancy",
        )
        if not Fraction(0) <= _fraction(threshold) <= Fraction(1):
            _fail("rank-CDF discrepancy threshold must lie in [0, 1]")
        _sha256(self.threshold_design_sha256, "threshold_design_sha256")
        object.__setattr__(
            self,
            "generation_event_keys",
            _event_keys(self.generation_event_keys, "threshold-design EventKeys"),
        )
        declared_frozen = _aware_utc(
            self.declared_frozen_at,
            "threshold declared_frozen_at",
        )
        available = _aware_utc(
            self.declared_audit_outcomes_available_at,
            "threshold declared_audit_outcomes_available_at",
        )
        object.__setattr__(self, "declared_frozen_at", declared_frozen)
        object.__setattr__(
            self,
            "declared_audit_outcomes_available_at",
            available,
        )
        if declared_frozen >= available:
            _fail(
                "caller-declared rank-threshold time must precede the "
                "caller-declared audit-outcome availability time"
            )
        if (
            self.caller_declared_policy_post_data_used is not False
            or self.externally_registered is not False
            or self.synthetic_only is not True
        ):
            _fail("threshold scientific boundary was widened")

    def _fresh_validate(self) -> None:
        fresh = CallerDeclaredRankThreshold(
            threshold_id=self.threshold_id,
            maximum_rank_cdf_discrepancy=self.maximum_rank_cdf_discrepancy,
            threshold_design_sha256=self.threshold_design_sha256,
            generation_event_keys=self.generation_event_keys,
            declared_frozen_at=self.declared_frozen_at,
            declared_audit_outcomes_available_at=(
                self.declared_audit_outcomes_available_at
            ),
            caller_declared_policy_post_data_used=(
                self.caller_declared_policy_post_data_used
            ),
            externally_registered=self.externally_registered,
            synthetic_only=self.synthetic_only,
        )
        if fresh != self:
            _fail("rank threshold was mutated outside validated construction")

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def to_mapping(self) -> dict[str, object]:
        self._fresh_validate()
        return {
            "declared_audit_outcomes_available_at": (
                self.declared_audit_outcomes_available_at.isoformat()
            ),
            "externally_registered": self.externally_registered,
            "declared_frozen_at": self.declared_frozen_at.isoformat(),
            "generation_event_keys": list(self.generation_event_keys),
            "maximum_rank_cdf_discrepancy": (
                self.maximum_rank_cdf_discrepancy.to_mapping()
            ),
            "caller_declared_policy_post_data_used": (
                self.caller_declared_policy_post_data_used
            ),
            "synthetic_only": self.synthetic_only,
            "threshold_design_sha256": self.threshold_design_sha256,
            "threshold_id": self.threshold_id,
        }


@dataclass(frozen=True, slots=True)
class PolicyFunctionalSBCClaimBoundary:
    """Claims that this caller-fed exact-arithmetic core cannot establish."""

    claim_scope: Literal["synthetic_policy_functional_sbc_core_arithmetic_only"] = (
        POLICY_FUNCTIONAL_SBC_CLAIM_SCOPE
    )
    synthetic_only: Literal[True] = True
    real_world_coverage_claim: Literal[False] = False
    distribution_free_validity_claim: Literal[False] = False
    real_world_causal_claim: Literal[False] = False
    publication_eligible: Literal[False] = False
    externally_registered: Literal[False] = False
    independent_reproduction: Literal[False] = False
    oracle_runner_verified: Literal[False] = False
    rooted_evidence_bundle_verified: Literal[False] = False
    rng_consumption_verified: Literal[False] = False
    threshold_chronology_externally_verified: Literal[False] = False
    caller_authored_core_diagnostic: Literal[True] = True
    validity_authorization: Literal[False] = False
    posterior_predictive_coverage_claim: Literal[False] = False
    finite_tape_exchangeability_verified: Literal[False] = False
    oracle_truth_replay_verified: Literal[False] = False
    pre_policy_data_use_verified: Literal[False] = False
    mechanism_alignment_verified: Literal[False] = False
    canary_execution_verified: Literal[False] = False
    full_bmr_cell_accounting_verified: Literal[False] = False
    tie_seed_chronology_verified: Literal[False] = False
    rank_test_power_verified: Literal[False] = False

    def __post_init__(self) -> None:
        if self.claim_scope != POLICY_FUNCTIONAL_SBC_CLAIM_SCOPE:
            _fail("policy-functional SBC claim scope changed")
        if (
            self.synthetic_only is not True
            or any(
                value is not False
                for value in (
                    self.real_world_coverage_claim,
                    self.distribution_free_validity_claim,
                    self.real_world_causal_claim,
                    self.publication_eligible,
                    self.externally_registered,
                    self.independent_reproduction,
                    self.oracle_runner_verified,
                    self.rooted_evidence_bundle_verified,
                    self.rng_consumption_verified,
                    self.threshold_chronology_externally_verified,
                    self.validity_authorization,
                    self.posterior_predictive_coverage_claim,
                    self.finite_tape_exchangeability_verified,
                    self.oracle_truth_replay_verified,
                    self.pre_policy_data_use_verified,
                    self.mechanism_alignment_verified,
                    self.canary_execution_verified,
                    self.full_bmr_cell_accounting_verified,
                    self.tie_seed_chronology_verified,
                    self.rank_test_power_verified,
                )
            )
            or self.caller_authored_core_diagnostic is not True
        ):
            _fail("policy-functional SBC claim boundary was widened")

    def _fresh_validate(self) -> None:
        fresh = PolicyFunctionalSBCClaimBoundary(
            claim_scope=self.claim_scope,
            synthetic_only=self.synthetic_only,
            real_world_coverage_claim=self.real_world_coverage_claim,
            distribution_free_validity_claim=self.distribution_free_validity_claim,
            real_world_causal_claim=self.real_world_causal_claim,
            publication_eligible=self.publication_eligible,
            externally_registered=self.externally_registered,
            independent_reproduction=self.independent_reproduction,
            oracle_runner_verified=self.oracle_runner_verified,
            rooted_evidence_bundle_verified=self.rooted_evidence_bundle_verified,
            rng_consumption_verified=self.rng_consumption_verified,
            threshold_chronology_externally_verified=(
                self.threshold_chronology_externally_verified
            ),
            caller_authored_core_diagnostic=self.caller_authored_core_diagnostic,
            validity_authorization=self.validity_authorization,
            posterior_predictive_coverage_claim=(
                self.posterior_predictive_coverage_claim
            ),
            finite_tape_exchangeability_verified=(
                self.finite_tape_exchangeability_verified
            ),
            oracle_truth_replay_verified=self.oracle_truth_replay_verified,
            pre_policy_data_use_verified=self.pre_policy_data_use_verified,
            mechanism_alignment_verified=self.mechanism_alignment_verified,
            canary_execution_verified=self.canary_execution_verified,
            full_bmr_cell_accounting_verified=(self.full_bmr_cell_accounting_verified),
            tie_seed_chronology_verified=self.tie_seed_chronology_verified,
            rank_test_power_verified=self.rank_test_power_verified,
        )
        if fresh != self:
            _fail("claim boundary was mutated outside validated construction")

    def to_mapping(self) -> dict[str, object]:
        self._fresh_validate()
        return {
            "claim_scope": self.claim_scope,
            "distribution_free_validity_claim": self.distribution_free_validity_claim,
            "caller_authored_core_diagnostic": self.caller_authored_core_diagnostic,
            "canary_execution_verified": self.canary_execution_verified,
            "externally_registered": self.externally_registered,
            "finite_tape_exchangeability_verified": (
                self.finite_tape_exchangeability_verified
            ),
            "full_bmr_cell_accounting_verified": (
                self.full_bmr_cell_accounting_verified
            ),
            "independent_reproduction": self.independent_reproduction,
            "oracle_runner_verified": self.oracle_runner_verified,
            "oracle_truth_replay_verified": self.oracle_truth_replay_verified,
            "publication_eligible": self.publication_eligible,
            "posterior_predictive_coverage_claim": (
                self.posterior_predictive_coverage_claim
            ),
            "real_world_causal_claim": self.real_world_causal_claim,
            "real_world_coverage_claim": self.real_world_coverage_claim,
            "rank_test_power_verified": self.rank_test_power_verified,
            "pre_policy_data_use_verified": self.pre_policy_data_use_verified,
            "rooted_evidence_bundle_verified": self.rooted_evidence_bundle_verified,
            "rng_consumption_verified": self.rng_consumption_verified,
            "mechanism_alignment_verified": self.mechanism_alignment_verified,
            "synthetic_only": self.synthetic_only,
            "threshold_chronology_externally_verified": (
                self.threshold_chronology_externally_verified
            ),
            "tie_seed_chronology_verified": self.tie_seed_chronology_verified,
            "validity_authorization": self.validity_authorization,
        }


@dataclass(frozen=True, slots=True)
class PolicyFunctionalSBCSpec:
    """Caller-fixed dimensions, output, threshold, and RNG names for phase one."""

    spec_id: str
    meta_replicate_count: int
    posterior_draw_count: int
    scenario_replicate_count: int
    primary_output_id: str
    primary_output_definition_sha256: str
    threshold: CallerDeclaredRankThreshold
    oracle_audit_generation_event_keys: tuple[str, ...]
    parameter_sbc_sha256: str
    randomized_tie_seed: str
    parameter_sbc_used_for_policy_functional_decision: Literal[False] = False
    scenario_functional: Literal[
        "exact_arithmetic_mean_over_caller_committed_scenario_replicates"
    ] = POLICY_FUNCTIONAL_SBC_SCENARIO_FUNCTIONAL
    estimand_scope: Literal[
        "caller_committed_finite_tape_g_omega_theta_shared_commitment_equality_only_consumption_unverified"
    ] = POLICY_FUNCTIONAL_SBC_ESTIMAND_SCOPE
    randomized_tie_rule: Literal["uniform_integer_over_exact_tie_positions"] = (
        POLICY_FUNCTIONAL_SBC_TIE_RULE
    )
    synthetic_only: Literal[True] = True
    schema_version: str = POLICY_FUNCTIONAL_SBC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _identifier(self.spec_id, "spec_id")
        b_count = _positive_int(self.meta_replicate_count, "meta_replicate_count")
        m_count = _positive_int(
            self.posterior_draw_count, "posterior_draw_count", minimum=2
        )
        r_count = _positive_int(
            self.scenario_replicate_count, "scenario_replicate_count"
        )
        if b_count * (m_count + 1) * r_count > _MAX_CELLS:
            _fail("policy-functional SBC plan exceeds its fixed exact-cell limit")
        _identifier(self.primary_output_id, "primary_output_id")
        _sha256(
            self.primary_output_definition_sha256,
            "primary_output_definition_sha256",
        )
        if type(self.threshold) is not CallerDeclaredRankThreshold:
            raise TypeError("threshold must be an exact CallerDeclaredRankThreshold")
        self.threshold._fresh_validate()
        object.__setattr__(
            self,
            "oracle_audit_generation_event_keys",
            _event_keys(
                self.oracle_audit_generation_event_keys,
                "oracle-audit EventKeys",
            ),
        )
        _sha256(self.parameter_sbc_sha256, "parameter_sbc_sha256")
        _identifier(self.randomized_tie_seed, "randomized_tie_seed")
        if self.parameter_sbc_used_for_policy_functional_decision is not False:
            _fail("parameter SBC cannot drive the policy-functional decision")
        if (
            self.scenario_functional != POLICY_FUNCTIONAL_SBC_SCENARIO_FUNCTIONAL
            or self.estimand_scope != POLICY_FUNCTIONAL_SBC_ESTIMAND_SCOPE
            or self.randomized_tie_rule != POLICY_FUNCTIONAL_SBC_TIE_RULE
            or self.synthetic_only is not True
            or self.schema_version != POLICY_FUNCTIONAL_SBC_SCHEMA_VERSION
        ):
            _fail("policy-functional SBC schema or scientific semantics changed")
        design = set(self.threshold.generation_event_keys)
        audit = set(self.oracle_audit_generation_event_keys)
        if design & audit:
            _fail("threshold-design and oracle-audit EventKeys must be disjoint")
        planned_ties = set(self.planned_tie_event_key_values)
        if len(planned_ties) != self.meta_replicate_count:
            _fail("planned randomized-tie EventKeys must be unique")
        if planned_ties & design or planned_ties & audit:
            _fail("randomized-tie EventKeys overlap another generation namespace")

    def _fresh_validate(self) -> None:
        self.threshold._fresh_validate()
        fresh = PolicyFunctionalSBCSpec(
            spec_id=self.spec_id,
            meta_replicate_count=self.meta_replicate_count,
            posterior_draw_count=self.posterior_draw_count,
            scenario_replicate_count=self.scenario_replicate_count,
            primary_output_id=self.primary_output_id,
            primary_output_definition_sha256=(self.primary_output_definition_sha256),
            threshold=self.threshold,
            oracle_audit_generation_event_keys=(
                self.oracle_audit_generation_event_keys
            ),
            parameter_sbc_sha256=self.parameter_sbc_sha256,
            randomized_tie_seed=self.randomized_tie_seed,
            parameter_sbc_used_for_policy_functional_decision=(
                self.parameter_sbc_used_for_policy_functional_decision
            ),
            scenario_functional=self.scenario_functional,
            estimand_scope=self.estimand_scope,
            randomized_tie_rule=self.randomized_tie_rule,
            synthetic_only=self.synthetic_only,
            schema_version=self.schema_version,
        )
        if fresh != self:
            _fail("policy-functional SBC spec was mutated outside construction")

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    @property
    def tie_design_sha256(self) -> str:
        """Commit only to the inputs that define randomized-rank identities."""
        return sha256_bytes(canonical_json_bytes(self.tie_design_mapping()))

    def tie_design_mapping(self) -> dict[str, object]:
        return {
            "meta_replicate_count": self.meta_replicate_count,
            "posterior_draw_count": self.posterior_draw_count,
            "primary_output_definition_sha256": (self.primary_output_definition_sha256),
            "primary_output_id": self.primary_output_id,
            "randomized_tie_rule": self.randomized_tie_rule,
            "scenario_replicate_count": self.scenario_replicate_count,
            "spec_id": self.spec_id,
        }

    @property
    def planned_tie_event_key_values(self) -> tuple[str, ...]:
        return tuple(
            policy_functional_tie_event_key(self, replicate).value
            for replicate in range(self.meta_replicate_count)
        )

    @property
    def complete_oracle_audit_generation_event_keys(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                self.oracle_audit_generation_event_keys
                + self.planned_tie_event_key_values
            )
        )

    def to_mapping(self) -> dict[str, object]:
        self._fresh_validate()
        # Tie keys use the explicit tie seed plus the narrow tie-design hash;
        # neither the whole spec hash nor audit outcomes influence tie draws.
        return {
            "meta_replicate_count": self.meta_replicate_count,
            "estimand_scope": self.estimand_scope,
            "oracle_audit_generation_event_keys": list(
                self.oracle_audit_generation_event_keys
            ),
            "parameter_sbc_sha256": self.parameter_sbc_sha256,
            "parameter_sbc_used_for_policy_functional_decision": (
                self.parameter_sbc_used_for_policy_functional_decision
            ),
            "posterior_draw_count": self.posterior_draw_count,
            "primary_output_definition_sha256": (self.primary_output_definition_sha256),
            "primary_output_id": self.primary_output_id,
            "randomized_tie_seed": self.randomized_tie_seed,
            "randomized_tie_rule": self.randomized_tie_rule,
            "scenario_functional": self.scenario_functional,
            "scenario_replicate_count": self.scenario_replicate_count,
            "schema_version": self.schema_version,
            "spec_id": self.spec_id,
            "synthetic_only": self.synthetic_only,
            "threshold": self.threshold.to_mapping(),
            "threshold_sha256": self.threshold.sha256,
            "tie_design_sha256": self.tie_design_sha256,
        }


def policy_functional_tie_event_key(
    spec: PolicyFunctionalSBCSpec,
    replicate_index: int,
) -> EventKey:
    """Derive the sole randomized-tie key from the spec and replicate identity."""
    if type(spec) is not PolicyFunctionalSBCSpec:
        raise TypeError("spec must be an exact PolicyFunctionalSBCSpec")
    index = _ordinal(replicate_index, "replicate_index")
    if index >= spec.meta_replicate_count:
        _fail("tie replicate_index lies outside the frozen meta-replicate plan")
    return EventKey(
        experiment_id=spec.spec_id,
        seed=spec.randomized_tie_seed,
        replicate_id=index,
        simulation_step=0,
        purpose=f"{POLICY_FUNCTIONAL_SBC_TIE_DOMAIN}.{spec.tie_design_sha256}",
    )


@dataclass(frozen=True, slots=True)
class PolicyFunctionalOracleRow:
    """Caller-supplied exact oracle scenario row; no runner authority is implied."""

    truth_parameter_sha256: str
    scenario_values: tuple[ExactRational, ...]
    scenario_commitments: tuple[str, ...]
    source_evidence_sha256: str

    def __post_init__(self) -> None:
        _sha256(self.truth_parameter_sha256, "truth_parameter_sha256")
        _sha256(self.source_evidence_sha256, "source_evidence_sha256")
        self._canonicalize_scenarios()

    def _canonicalize_scenarios(self) -> None:
        if (
            not isinstance(self.scenario_values, tuple)
            or not self.scenario_values
            or not isinstance(self.scenario_commitments, tuple)
            or len(self.scenario_values) != len(self.scenario_commitments)
        ):
            _fail("oracle scenario values and commitments must align and be non-empty")
        pairs: list[tuple[str, ExactRational]] = []
        for commitment, value in zip(
            self.scenario_commitments, self.scenario_values, strict=True
        ):
            pairs.append(
                (
                    _sha256(commitment, "oracle scenario commitment"),
                    _exact(value, "oracle scenario value"),
                )
            )
        pairs.sort(key=lambda item: item[0])
        if len({item[0] for item in pairs}) != len(pairs):
            _fail("oracle scenario commitments must be unique")
        object.__setattr__(
            self, "scenario_commitments", tuple(item[0] for item in pairs)
        )
        object.__setattr__(self, "scenario_values", tuple(item[1] for item in pairs))

    @property
    def scenario_mean(self) -> ExactRational:
        self._fresh_validate()
        return _mean(self.scenario_values)

    def _fresh_validate(self) -> None:
        fresh = PolicyFunctionalOracleRow(
            truth_parameter_sha256=self.truth_parameter_sha256,
            scenario_values=self.scenario_values,
            scenario_commitments=self.scenario_commitments,
            source_evidence_sha256=self.source_evidence_sha256,
        )
        if fresh != self:
            _fail("oracle row was mutated outside validated construction")

    def to_mapping(self) -> dict[str, object]:
        self._fresh_validate()
        return {
            "scenario_commitments": list(self.scenario_commitments),
            "scenario_mean": self.scenario_mean.to_mapping(),
            "scenario_values": [value.to_mapping() for value in self.scenario_values],
            "source_evidence_sha256": self.source_evidence_sha256,
            "truth_parameter_sha256": self.truth_parameter_sha256,
        }


@dataclass(frozen=True, slots=True)
class PolicyFunctionalPosteriorRow:
    """One exact posterior functional row over the same frozen scenario columns."""

    posterior_index: int
    posterior_draw_sha256: str
    parameter_sha256: str
    scenario_values: tuple[ExactRational, ...]
    scenario_commitments: tuple[str, ...]
    source_evidence_sha256: str

    def __post_init__(self) -> None:
        _ordinal(self.posterior_index, "posterior_index")
        for field in (
            "posterior_draw_sha256",
            "parameter_sha256",
            "source_evidence_sha256",
        ):
            _sha256(getattr(self, field), field)
        if (
            not isinstance(self.scenario_values, tuple)
            or not self.scenario_values
            or not isinstance(self.scenario_commitments, tuple)
            or len(self.scenario_values) != len(self.scenario_commitments)
        ):
            _fail(
                "posterior scenario values and commitments must align and be non-empty"
            )
        pairs: list[tuple[str, ExactRational]] = []
        for commitment, value in zip(
            self.scenario_commitments, self.scenario_values, strict=True
        ):
            pairs.append(
                (
                    _sha256(commitment, "posterior scenario commitment"),
                    _exact(value, "posterior scenario value"),
                )
            )
        pairs.sort(key=lambda item: item[0])
        if len({item[0] for item in pairs}) != len(pairs):
            _fail("posterior scenario commitments must be unique")
        object.__setattr__(
            self, "scenario_commitments", tuple(item[0] for item in pairs)
        )
        object.__setattr__(self, "scenario_values", tuple(item[1] for item in pairs))

    @property
    def scenario_mean(self) -> ExactRational:
        self._fresh_validate()
        return _mean(self.scenario_values)

    def _fresh_validate(self) -> None:
        fresh = PolicyFunctionalPosteriorRow(
            posterior_index=self.posterior_index,
            posterior_draw_sha256=self.posterior_draw_sha256,
            parameter_sha256=self.parameter_sha256,
            scenario_values=self.scenario_values,
            scenario_commitments=self.scenario_commitments,
            source_evidence_sha256=self.source_evidence_sha256,
        )
        if fresh != self:
            _fail("posterior row was mutated outside validated construction")

    def to_mapping(self) -> dict[str, object]:
        self._fresh_validate()
        return {
            "parameter_sha256": self.parameter_sha256,
            "posterior_draw_sha256": self.posterior_draw_sha256,
            "posterior_index": self.posterior_index,
            "scenario_commitments": list(self.scenario_commitments),
            "scenario_mean": self.scenario_mean.to_mapping(),
            "scenario_values": [value.to_mapping() for value in self.scenario_values],
            "source_evidence_sha256": self.source_evidence_sha256,
        }


@dataclass(frozen=True, slots=True)
class PolicyFunctionalMetaReplicate:
    """One retained reference replicate, including explicit failure/abstention."""

    replicate_index: int
    status: MetaReplicateStatus
    oracle: PolicyFunctionalOracleRow | None
    posterior_rows: tuple[PolicyFunctionalPosteriorRow, ...]
    status_reason: str | None = None
    status_evidence_sha256: str | None = None
    failure_stage: MetaReplicateFailureStage | None = None

    def __post_init__(self) -> None:
        _ordinal(self.replicate_index, "replicate_index")
        if not isinstance(self.status, MetaReplicateStatus):
            _fail("meta-replicate status must use the closed vocabulary")
        if not isinstance(self.posterior_rows, tuple) or not all(
            type(row) is PolicyFunctionalPosteriorRow for row in self.posterior_rows
        ):
            _fail("posterior_rows must be an immutable typed tuple")
        ordered = tuple(
            sorted(self.posterior_rows, key=lambda row: row.posterior_index)
        )
        indices = tuple(row.posterior_index for row in ordered)
        if len(set(indices)) != len(indices):
            _fail("posterior row indices must be unique within a replicate")
        object.__setattr__(self, "posterior_rows", ordered)
        if self.status is MetaReplicateStatus.COMPLETED:
            if type(self.oracle) is not PolicyFunctionalOracleRow:
                _fail("completed meta-replicate requires an exact oracle row")
            if (
                not ordered
                or self.status_reason is not None
                or self.status_evidence_sha256 is not None
                or self.failure_stage is not None
            ):
                _fail("completed meta-replicate requires rows and no failure evidence")
        elif (
            self.oracle is not None
            or ordered
            or self.status_reason is None
            or self.status_evidence_sha256 is None
            or not isinstance(self.failure_stage, MetaReplicateFailureStage)
        ):
            _fail(
                "failed/abstained meta-replicate requires reason, evidence, and "
                "a closed failure stage"
            )
        if self.status_reason is not None and self.status_evidence_sha256 is not None:
            _reason(self.status_reason, "status_reason")
            _sha256(self.status_evidence_sha256, "status_evidence_sha256")

    def _fresh_validate(self) -> None:
        if self.oracle is not None:
            self.oracle._fresh_validate()
        for row in self.posterior_rows:
            row._fresh_validate()
        fresh = PolicyFunctionalMetaReplicate(
            replicate_index=self.replicate_index,
            status=self.status,
            oracle=self.oracle,
            posterior_rows=self.posterior_rows,
            status_reason=self.status_reason,
            status_evidence_sha256=self.status_evidence_sha256,
            failure_stage=self.failure_stage,
        )
        if fresh != self:
            _fail("meta-replicate was mutated outside validated construction")

    def to_mapping(self) -> dict[str, object]:
        self._fresh_validate()
        return {
            "failure_stage": (
                None if self.failure_stage is None else self.failure_stage.value
            ),
            "oracle": None if self.oracle is None else self.oracle.to_mapping(),
            "posterior_rows": [row.to_mapping() for row in self.posterior_rows],
            "replicate_index": self.replicate_index,
            "status": self.status.value,
            "status_evidence_sha256": self.status_evidence_sha256,
            "status_reason": self.status_reason,
        }


@dataclass(frozen=True, slots=True)
class PolicyFunctionalCanaryResult:
    """Retained verdict from one separately implemented controlled canary guard."""

    kind: CanaryKind
    detection: CanaryDetection
    evidence_sha256: str
    spec_sha256: str
    primary_output_definition_sha256: str
    reason: str
    caller_authored: Literal[True] = True
    execution_verified: Literal[False] = False

    def __post_init__(self) -> None:
        if not isinstance(self.kind, CanaryKind) or not isinstance(
            self.detection, CanaryDetection
        ):
            _fail("canary kind and detection must use the closed vocabulary")
        _sha256(self.evidence_sha256, "canary evidence_sha256")
        _sha256(self.spec_sha256, "canary spec_sha256")
        _sha256(
            self.primary_output_definition_sha256,
            "canary primary_output_definition_sha256",
        )
        _reason(self.reason, "canary reason")
        if self.caller_authored is not True or self.execution_verified is not False:
            _fail("caller-authored canary authority boundary was widened")

    def _fresh_validate(self) -> None:
        fresh = PolicyFunctionalCanaryResult(
            kind=self.kind,
            detection=self.detection,
            evidence_sha256=self.evidence_sha256,
            spec_sha256=self.spec_sha256,
            primary_output_definition_sha256=(self.primary_output_definition_sha256),
            reason=self.reason,
            caller_authored=self.caller_authored,
            execution_verified=self.execution_verified,
        )
        if fresh != self:
            _fail("canary verdict was mutated outside validated construction")

    def to_mapping(self) -> dict[str, object]:
        self._fresh_validate()
        return {
            "caller_authored": self.caller_authored,
            "detection": self.detection.value,
            "evidence_sha256": self.evidence_sha256,
            "execution_verified": self.execution_verified,
            "kind": self.kind.value,
            "primary_output_definition_sha256": (self.primary_output_definition_sha256),
            "reason": self.reason,
            "spec_sha256": self.spec_sha256,
        }


@dataclass(frozen=True, slots=True)
class PolicyFunctionalRankRecord:
    """One exact randomized policy-functional rank and all tie ingredients."""

    replicate_index: int
    output_id: str
    oracle_scenario_mean: ExactRational
    posterior_scenario_means: tuple[ExactRational, ...]
    less_count: int
    equal_count: int
    tie_event_key: str
    tie_domain: Literal["policy-functional-sbc-randomized-exact-tie-v1"]
    tie_draw: int
    rank: int

    def __post_init__(self) -> None:
        _ordinal(self.replicate_index, "rank replicate_index")
        _identifier(self.output_id, "rank output_id")
        oracle = _exact(self.oracle_scenario_mean, "oracle_scenario_mean")
        if (
            not isinstance(self.posterior_scenario_means, tuple)
            or not self.posterior_scenario_means
            or not all(
                type(value) is ExactRational for value in self.posterior_scenario_means
            )
        ):
            _fail("posterior_scenario_means must be an immutable exact tuple")
        for index, value in enumerate(self.posterior_scenario_means):
            _exact(value, f"posterior_scenario_means[{index}]")
        less = sum(
            _fraction(value) < _fraction(oracle)
            for value in self.posterior_scenario_means
        )
        equal = sum(
            _fraction(value) == _fraction(oracle)
            for value in self.posterior_scenario_means
        )
        if self.less_count != less or self.equal_count != equal:
            _fail("rank comparison counts do not reproduce from exact means")
        if (
            not isinstance(self.tie_event_key, str)
            or _EVENT_KEY.fullmatch(self.tie_event_key) is None
        ):
            _fail("rank tie_event_key is invalid")
        if self.tie_domain != POLICY_FUNCTIONAL_SBC_TIE_DOMAIN:
            _fail("randomized rank tie domain changed")
        if (
            isinstance(self.tie_draw, bool)
            or not isinstance(self.tie_draw, int)
            or not 0 <= self.tie_draw <= equal
            or self.rank != less + self.tie_draw
        ):
            _fail("randomized rank does not follow the frozen exact-tie rule")

    def _fresh_validate(self) -> None:
        fresh = PolicyFunctionalRankRecord(
            replicate_index=self.replicate_index,
            output_id=self.output_id,
            oracle_scenario_mean=self.oracle_scenario_mean,
            posterior_scenario_means=self.posterior_scenario_means,
            less_count=self.less_count,
            equal_count=self.equal_count,
            tie_event_key=self.tie_event_key,
            tie_domain=self.tie_domain,
            tie_draw=self.tie_draw,
            rank=self.rank,
        )
        if fresh != self:
            _fail("rank record was mutated outside validated construction")

    def to_mapping(self) -> dict[str, object]:
        self._fresh_validate()
        return {
            "equal_count": self.equal_count,
            "less_count": self.less_count,
            "oracle_scenario_mean": self.oracle_scenario_mean.to_mapping(),
            "output_id": self.output_id,
            "posterior_scenario_means": [
                value.to_mapping() for value in self.posterior_scenario_means
            ],
            "rank": self.rank,
            "replicate_index": self.replicate_index,
            "tie_draw": self.tie_draw,
            "tie_domain": self.tie_domain,
            "tie_event_key": self.tie_event_key,
        }


def _rank_record(
    spec: PolicyFunctionalSBCSpec,
    replicate: PolicyFunctionalMetaReplicate,
) -> PolicyFunctionalRankRecord:
    if (
        replicate.status is not MetaReplicateStatus.COMPLETED
        or replicate.oracle is None
    ):
        _fail("only a completed replicate can produce a policy-functional rank")
    posterior_means = tuple(row.scenario_mean for row in replicate.posterior_rows)
    oracle_mean = replicate.oracle.scenario_mean
    less = sum(_fraction(value) < _fraction(oracle_mean) for value in posterior_means)
    equal = sum(_fraction(value) == _fraction(oracle_mean) for value in posterior_means)
    tie_key = policy_functional_tie_event_key(spec, replicate.replicate_index)
    tie_draw = deterministic_int(tie_key, 0, equal)
    return PolicyFunctionalRankRecord(
        replicate_index=replicate.replicate_index,
        output_id=spec.primary_output_id,
        oracle_scenario_mean=oracle_mean,
        posterior_scenario_means=posterior_means,
        less_count=less,
        equal_count=equal,
        tie_event_key=tie_key.value,
        tie_domain=POLICY_FUNCTIONAL_SBC_TIE_DOMAIN,
        tie_draw=tie_draw,
        rank=less + tie_draw,
    )


def exact_rank_cdf_discrepancy(
    histogram: tuple[int, ...],
    *,
    posterior_draw_count: int,
) -> ExactRational:
    """Return exact max distance from the discrete-uniform rank CDF."""
    draws = _positive_int(posterior_draw_count, "posterior_draw_count", minimum=2)
    if (
        not isinstance(histogram, tuple)
        or len(histogram) != draws + 1
        or any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0
            for count in histogram
        )
        or sum(histogram) <= 0
    ):
        _fail("rank histogram is invalid or empty")
    total = sum(histogram)
    cumulative = 0
    maximum = Fraction()
    for rank, count in enumerate(histogram):
        cumulative += count
        observed = Fraction(cumulative, total)
        expected = Fraction(rank + 1, draws + 1)
        maximum = max(maximum, abs(observed - expected))
    return _from_fraction(maximum)


@dataclass(frozen=True, slots=True)
class PolicyFunctionalSBCResult:
    """Fully rederived phase-one result over caller-fixed reference rows."""

    spec: PolicyFunctionalSBCSpec
    meta_replicates: tuple[PolicyFunctionalMetaReplicate, ...]
    canaries: tuple[PolicyFunctionalCanaryResult, ...]
    rank_records: tuple[PolicyFunctionalRankRecord, ...]
    rank_histogram: tuple[int, ...]
    rank_cdf_discrepancy: ExactRational | None
    degenerate_all_tie_functional: bool
    decision: PolicyFunctionalSBCDecision
    decision_reasons: tuple[str, ...]
    completed_reference_replicates: int
    failed_reference_replicates: int
    abstained_reference_replicates: int
    threshold_design_generation_event_keys: tuple[str, ...]
    oracle_audit_generation_event_keys: tuple[str, ...]
    parameter_sbc_sha256: str
    parameter_sbc_used_for_policy_functional_decision: Literal[False]
    claims: PolicyFunctionalSBCClaimBoundary
    rng_consumption_verified: Literal[False] = False
    threshold_chronology_externally_verified: Literal[False] = False
    finite_tape_exchangeability_verified: Literal[False] = False
    oracle_truth_replay_verified: Literal[False] = False
    pre_policy_data_use_verified: Literal[False] = False
    mechanism_alignment_verified: Literal[False] = False
    canary_execution_verified: Literal[False] = False
    full_bmr_cell_accounting_verified: Literal[False] = False
    tie_seed_chronology_verified: Literal[False] = False
    rank_test_power_verified: Literal[False] = False
    caller_authored_core_diagnostic: Literal[True] = True
    validity_authorization: Literal[False] = False
    all_planned_meta_replicate_statuses_accounted: Literal[True] = True
    all_planned_bmr_cell_statuses_accounted: Literal[False] = False
    complete_case_histogram_used: Literal[False] = False
    failed_or_abstained_replicates_retained: Literal[True] = True
    schema_version: str = POLICY_FUNCTIONAL_SBC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        self._fresh_validate()

    def _fresh_validate(self) -> None:
        if type(self.spec) is not PolicyFunctionalSBCSpec:
            raise TypeError("result spec must be an exact PolicyFunctionalSBCSpec")
        self.spec._fresh_validate()
        if not isinstance(self.meta_replicates, tuple):
            _fail("result meta_replicates must be an immutable tuple")
        for replicate in self.meta_replicates:
            if type(replicate) is not PolicyFunctionalMetaReplicate:
                _fail("result contains an untyped meta-replicate")
            replicate._fresh_validate()
        if not isinstance(self.canaries, tuple):
            _fail("result canaries must be an immutable tuple")
        for canary in self.canaries:
            if type(canary) is not PolicyFunctionalCanaryResult:
                _fail("result contains an untyped canary")
            canary._fresh_validate()
        if not isinstance(self.rank_records, tuple):
            _fail("result rank_records must be an immutable tuple")
        for record in self.rank_records:
            if type(record) is not PolicyFunctionalRankRecord:
                _fail("result contains an untyped rank record")
            record._fresh_validate()
        expected = _derive_components(self.spec, self.meta_replicates, self.canaries)
        actual = (
            self.meta_replicates,
            self.canaries,
            self.rank_records,
            self.rank_histogram,
            self.rank_cdf_discrepancy,
            self.decision,
            self.decision_reasons,
            self.completed_reference_replicates,
            self.failed_reference_replicates,
            self.abstained_reference_replicates,
        )
        if actual != expected:
            _fail("policy-functional SBC result fields do not rederive from inputs")
        expected_degenerate = _all_tie_degenerate(
            self.spec,
            self.rank_records,
            completed=self.completed_reference_replicates,
        )
        if (
            not isinstance(self.degenerate_all_tie_functional, bool)
            or self.degenerate_all_tie_functional is not expected_degenerate
        ):
            _fail("all-tie finite-tape degeneracy flag does not rederive")
        if self.threshold_design_generation_event_keys != (
            self.spec.threshold.generation_event_keys
        ):
            _fail("result threshold-design EventKey manifest changed")
        if self.oracle_audit_generation_event_keys != (
            self.spec.complete_oracle_audit_generation_event_keys
        ):
            _fail("result oracle-audit EventKey manifest changed")
        if set(self.threshold_design_generation_event_keys) & set(
            self.oracle_audit_generation_event_keys
        ):
            _fail("result threshold-design and oracle-audit EventKeys overlap")
        if (
            self.parameter_sbc_sha256 != self.spec.parameter_sbc_sha256
            or self.parameter_sbc_used_for_policy_functional_decision is not False
        ):
            _fail("parameter SBC was relabeled as policy-functional evidence")
        if type(self.claims) is not PolicyFunctionalSBCClaimBoundary:
            raise TypeError("claims must use the exact conservative boundary")
        self.claims._fresh_validate()
        if self.claims != PolicyFunctionalSBCClaimBoundary():
            _fail("result claims differ from the conservative claim boundary")
        if (
            self.all_planned_meta_replicate_statuses_accounted is not True
            or self.all_planned_bmr_cell_statuses_accounted is not False
            or self.complete_case_histogram_used is not False
            or self.failed_or_abstained_replicates_retained is not True
            or self.schema_version != POLICY_FUNCTIONAL_SBC_SCHEMA_VERSION
            or self.rng_consumption_verified is not False
            or self.threshold_chronology_externally_verified is not False
            or self.finite_tape_exchangeability_verified is not False
            or self.oracle_truth_replay_verified is not False
            or self.pre_policy_data_use_verified is not False
            or self.mechanism_alignment_verified is not False
            or self.canary_execution_verified is not False
            or self.full_bmr_cell_accounting_verified is not False
            or self.tie_seed_chronology_verified is not False
            or self.rank_test_power_verified is not False
            or self.caller_authored_core_diagnostic is not True
            or self.validity_authorization is not False
        ):
            _fail("result accounting or schema boundary changed")

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def to_mapping(self) -> dict[str, object]:
        self._fresh_validate()
        return {
            "abstained_reference_replicates": self.abstained_reference_replicates,
            "all_planned_bmr_cell_statuses_accounted": (
                self.all_planned_bmr_cell_statuses_accounted
            ),
            "all_planned_meta_replicate_statuses_accounted": (
                self.all_planned_meta_replicate_statuses_accounted
            ),
            "canaries": [canary.to_mapping() for canary in self.canaries],
            "caller_authored_core_diagnostic": self.caller_authored_core_diagnostic,
            "canary_execution_verified": self.canary_execution_verified,
            "claims": self.claims.to_mapping(),
            "complete_case_histogram_used": self.complete_case_histogram_used,
            "completed_reference_replicates": self.completed_reference_replicates,
            "decision": self.decision.value,
            "decision_reasons": list(self.decision_reasons),
            "degenerate_all_tie_functional": self.degenerate_all_tie_functional,
            "failed_or_abstained_replicates_retained": (
                self.failed_or_abstained_replicates_retained
            ),
            "failed_reference_replicates": self.failed_reference_replicates,
            "finite_tape_exchangeability_verified": (
                self.finite_tape_exchangeability_verified
            ),
            "full_bmr_cell_accounting_verified": (
                self.full_bmr_cell_accounting_verified
            ),
            "mechanism_alignment_verified": self.mechanism_alignment_verified,
            "meta_replicates": [item.to_mapping() for item in self.meta_replicates],
            "oracle_audit_generation_event_keys": list(
                self.oracle_audit_generation_event_keys
            ),
            "parameter_sbc_sha256": self.parameter_sbc_sha256,
            "parameter_sbc_used_for_policy_functional_decision": (
                self.parameter_sbc_used_for_policy_functional_decision
            ),
            "oracle_truth_replay_verified": self.oracle_truth_replay_verified,
            "pre_policy_data_use_verified": self.pre_policy_data_use_verified,
            "rank_cdf_discrepancy": (
                None
                if self.rank_cdf_discrepancy is None
                else self.rank_cdf_discrepancy.to_mapping()
            ),
            "rank_histogram": list(self.rank_histogram),
            "rank_records": [record.to_mapping() for record in self.rank_records],
            "rank_test_power_verified": self.rank_test_power_verified,
            "rng_consumption_verified": self.rng_consumption_verified,
            "schema_version": self.schema_version,
            "spec": self.spec.to_mapping(),
            "spec_sha256": self.spec.sha256,
            "threshold_design_generation_event_keys": list(
                self.threshold_design_generation_event_keys
            ),
            "threshold_chronology_externally_verified": (
                self.threshold_chronology_externally_verified
            ),
            "tie_seed_chronology_verified": self.tie_seed_chronology_verified,
            "validity_authorization": self.validity_authorization,
        }


def _validate_completed_dimensions(
    spec: PolicyFunctionalSBCSpec,
    replicate: PolicyFunctionalMetaReplicate,
) -> None:
    if replicate.status is not MetaReplicateStatus.COMPLETED:
        return
    oracle = replicate.oracle
    if oracle is None:
        _fail("completed replicate unexpectedly lacks an oracle")
    if len(oracle.scenario_values) != spec.scenario_replicate_count:
        _fail("oracle row does not cover the frozen scenario count")
    if len(replicate.posterior_rows) != spec.posterior_draw_count:
        _fail("replicate does not cover the frozen posterior count")
    if tuple(row.posterior_index for row in replicate.posterior_rows) != tuple(
        range(spec.posterior_draw_count)
    ):
        _fail("posterior row indices do not cover the frozen posterior axis")
    posterior_draws = tuple(
        row.posterior_draw_sha256 for row in replicate.posterior_rows
    )
    if len(set(posterior_draws)) != len(posterior_draws):
        _fail("posterior row commitments must be unique")
    for row in replicate.posterior_rows:
        if len(row.scenario_values) != spec.scenario_replicate_count:
            _fail("posterior row does not cover the frozen scenario count")
        if row.scenario_commitments != oracle.scenario_commitments:
            _fail("oracle and posterior rows do not share the exact scenario tape")


def _canonical_canaries(
    spec: PolicyFunctionalSBCSpec,
    canaries: tuple[PolicyFunctionalCanaryResult, ...],
) -> tuple[PolicyFunctionalCanaryResult, ...]:
    if not isinstance(canaries, tuple) or not all(
        type(canary) is PolicyFunctionalCanaryResult for canary in canaries
    ):
        _fail("canaries must be an immutable typed tuple")
    for canary in canaries:
        canary._fresh_validate()
    ordered = tuple(sorted(canaries, key=lambda canary: canary.kind.value))
    kinds = tuple(canary.kind for canary in ordered)
    if set(kinds) != set(CanaryKind) or len(kinds) != len(CanaryKind):
        _fail("result must retain each required canary exactly once")
    evidence = tuple(canary.evidence_sha256 for canary in ordered)
    if len(set(evidence)) != len(evidence):
        _fail("required canaries must have unique evidence commitments")
    if any(
        canary.spec_sha256 != spec.sha256
        or canary.primary_output_definition_sha256
        != spec.primary_output_definition_sha256
        or canary.caller_authored is not True
        or canary.execution_verified is not False
        for canary in ordered
    ):
        _fail("canary verdict does not bind the current spec and primary output")
    return ordered


def _derive_components(
    spec: PolicyFunctionalSBCSpec,
    meta_replicates: tuple[PolicyFunctionalMetaReplicate, ...],
    canaries: tuple[PolicyFunctionalCanaryResult, ...],
) -> tuple[
    tuple[PolicyFunctionalMetaReplicate, ...],
    tuple[PolicyFunctionalCanaryResult, ...],
    tuple[PolicyFunctionalRankRecord, ...],
    tuple[int, ...],
    ExactRational | None,
    PolicyFunctionalSBCDecision,
    tuple[str, ...],
    int,
    int,
    int,
]:
    spec._fresh_validate()
    if not isinstance(meta_replicates, tuple) or not all(
        type(item) is PolicyFunctionalMetaReplicate for item in meta_replicates
    ):
        _fail("meta_replicates must be an immutable typed tuple")
    ordered = tuple(sorted(meta_replicates, key=lambda item: item.replicate_index))
    if tuple(item.replicate_index for item in ordered) != tuple(
        range(spec.meta_replicate_count)
    ):
        _fail("meta-replicates do not exactly cover the frozen B axis")
    for replicate in ordered:
        replicate._fresh_validate()
        _validate_completed_dimensions(spec, replicate)
    ordered_canaries = _canonical_canaries(spec, canaries)
    completed = sum(item.status is MetaReplicateStatus.COMPLETED for item in ordered)
    failed = sum(item.status is MetaReplicateStatus.FAILED for item in ordered)
    abstained = sum(item.status is MetaReplicateStatus.ABSTAINED for item in ordered)
    ranks = tuple(
        _rank_record(spec, item)
        for item in ordered
        if item.status is MetaReplicateStatus.COMPLETED
    )
    undetected = tuple(
        canary.kind.value
        for canary in ordered_canaries
        if not canary.detection.detected
    )
    reasons: list[str] = []
    if completed != spec.meta_replicate_count:
        reasons.append("reference_replicate_not_completed")
    if undetected:
        reasons.append("required_canary_not_detected:" + ",".join(undetected))
    if completed != spec.meta_replicate_count:
        # Individual completed ranks remain auditable, but no complete-case
        # histogram or discrepancy is emitted.
        return (
            ordered,
            ordered_canaries,
            ranks,
            (),
            None,
            (
                PolicyFunctionalSBCDecision.FAIL
                if undetected
                else PolicyFunctionalSBCDecision.ABSTAIN
            ),
            tuple(reasons),
            completed,
            failed,
            abstained,
        )
    histogram = tuple(
        sum(record.rank == rank for record in ranks)
        for rank in range(spec.posterior_draw_count + 1)
    )
    discrepancy = exact_rank_cdf_discrepancy(
        histogram,
        posterior_draw_count=spec.posterior_draw_count,
    )
    degenerate = _all_tie_degenerate(spec, ranks, completed=completed)
    if degenerate:
        reasons.append("degenerate_all_tie_finite_tape_functional_no_information")
    if undetected:
        decision = PolicyFunctionalSBCDecision.FAIL
    elif degenerate:
        decision = PolicyFunctionalSBCDecision.ABSTAIN
    elif _fraction(discrepancy) > _fraction(
        spec.threshold.maximum_rank_cdf_discrepancy
    ):
        reasons.append("rank_cdf_discrepancy_exceeds_caller_declared_threshold")
        decision = PolicyFunctionalSBCDecision.FAIL
    else:
        reasons.append("synthetic_rank_diagnostic_within_caller_declared_threshold")
        decision = PolicyFunctionalSBCDecision.SYNTHETIC_CORE_ARITHMETIC_PASS_UNVERIFIED
    return (
        ordered,
        ordered_canaries,
        ranks,
        histogram,
        discrepancy,
        decision,
        tuple(reasons),
        completed,
        failed,
        abstained,
    )


def _all_tie_degenerate(
    spec: PolicyFunctionalSBCSpec,
    ranks: tuple[PolicyFunctionalRankRecord, ...],
    *,
    completed: int,
) -> bool:
    return (
        completed == spec.meta_replicate_count
        and bool(ranks)
        and all(
            record.less_count == 0 and record.equal_count == spec.posterior_draw_count
            for record in ranks
        )
    )


def derive_policy_functional_sbc_result(
    *,
    spec: PolicyFunctionalSBCSpec,
    meta_replicates: tuple[PolicyFunctionalMetaReplicate, ...],
    canaries: tuple[PolicyFunctionalCanaryResult, ...],
) -> PolicyFunctionalSBCResult:
    """Derive the exact phase-one result without running or trusting an oracle."""
    if type(spec) is not PolicyFunctionalSBCSpec:
        raise TypeError("spec must be an exact PolicyFunctionalSBCSpec")
    components = _derive_components(spec, meta_replicates, canaries)
    return PolicyFunctionalSBCResult(
        spec=spec,
        meta_replicates=components[0],
        canaries=components[1],
        rank_records=components[2],
        rank_histogram=components[3],
        rank_cdf_discrepancy=components[4],
        degenerate_all_tie_functional=_all_tie_degenerate(
            spec,
            components[2],
            completed=components[7],
        ),
        decision=components[5],
        decision_reasons=components[6],
        completed_reference_replicates=components[7],
        failed_reference_replicates=components[8],
        abstained_reference_replicates=components[9],
        threshold_design_generation_event_keys=(spec.threshold.generation_event_keys),
        oracle_audit_generation_event_keys=(
            spec.complete_oracle_audit_generation_event_keys
        ),
        parameter_sbc_sha256=spec.parameter_sbc_sha256,
        parameter_sbc_used_for_policy_functional_decision=False,
        claims=PolicyFunctionalSBCClaimBoundary(),
    )


__all__ = [
    "POLICY_FUNCTIONAL_SBC_CLAIM_SCOPE",
    "POLICY_FUNCTIONAL_SBC_ESTIMAND_SCOPE",
    "POLICY_FUNCTIONAL_SBC_SCENARIO_FUNCTIONAL",
    "POLICY_FUNCTIONAL_SBC_SCHEMA_VERSION",
    "POLICY_FUNCTIONAL_SBC_TIE_RULE",
    "POLICY_FUNCTIONAL_SBC_TIE_DOMAIN",
    "CanaryDetection",
    "CanaryKind",
    "CallerDeclaredRankThreshold",
    "MetaReplicateFailureStage",
    "MetaReplicateStatus",
    "PolicyFunctionalCanaryResult",
    "PolicyFunctionalMetaReplicate",
    "PolicyFunctionalOracleRow",
    "PolicyFunctionalPosteriorRow",
    "PolicyFunctionalRankRecord",
    "PolicyFunctionalSBCClaimBoundary",
    "PolicyFunctionalSBCDecision",
    "PolicyFunctionalSBCError",
    "PolicyFunctionalSBCResult",
    "PolicyFunctionalSBCSpec",
    "derive_policy_functional_sbc_result",
    "exact_rank_cdf_discrepancy",
    "policy_functional_tie_event_key",
]
