"""Paired baseline and policy counterfactual execution primitives."""

from typing import TYPE_CHECKING

from policytwin.counterfactual.events import (
    HASH_BASED_RNG_IMPLEMENTATION_ID,
    MAX_SCENARIO_GENERATION_DRAWS,
    SCENARIO_GENERATION_DRAW_SCHEMA_VERSION,
    SCENARIO_GENERATION_RECEIPT_SCHEMA_VERSION,
    EventType,
    ExogenousShock,
    ExogenousShockTape,
    HashBasedRNG,
    ScenarioEventReceipt,
    ScenarioGenerationDraw,
    ScenarioGenerationReceipt,
    ScenarioGeneratorSpec,
    SemanticEventError,
    SemanticEventKey,
    ShockAccessCapability,
    ShockAccessError,
    ShockAccessViolation,
    ShockAccessViolationCode,
    ShockConsumption,
    ShockConsumptionLedger,
    ShockTransitionConfig,
    ShockTransitionReceipt,
    ShockTransitionResult,
    ShockTransitionState,
    hash_based_rng_live_dependency_sha256,
    hash_integer,
    hash_uniform,
    validate_exact_exogenous_shock_tape,
    validate_exact_scenario_generation_receipt,
)
from policytwin.counterfactual.paired import (
    GroupOutcome,
    GroupSpec,
    HorizonOutcome,
    HorizonSpec,
    MetricValue,
    PairedDifference,
    PairedOutcome,
    PairIntegrityAudit,
    PairIntegrityError,
    PolicyDefinition,
    PolicyWorldPair,
    PosteriorDraw,
    RandomnessDesign,
    StateArtifact,
    WorldOutcome,
    WorldRole,
    WorldRunRequest,
    WorldStatus,
    compare_randomness_designs,
    run_policy_world_pairs,
    validate_exact_policy_world_pair,
)
from policytwin.counterfactual.randomness import (
    EVENT_KEY_SCHEMA_VERSION,
    EventKey,
    deterministic_int,
)

if TYPE_CHECKING:
    from policytwin.counterfactual.internal_randomness_audit import (
        InternalRandomCouplingAudit,
        InternalRandomCouplingAuditError,
    )

_LAZY_INTERNAL_RANDOMNESS_EXPORTS = frozenset(
    {
        "INTERNAL_RANDOM_COUPLING_AUDIT_SCHEMA_VERSION",
        "INTERNAL_RANDOM_COUPLING_CLAIM_SCOPE",
        "InternalRandomCouplingAudit",
        "InternalRandomCouplingAuditError",
        "require_verifier_owned_internal_random_coupling_audit",
    }
)


def __getattr__(name: str) -> object:
    """Load the integrated-run audit lazily to avoid package import cycles."""
    if name in _LAZY_INTERNAL_RANDOMNESS_EXPORTS:
        from policytwin.counterfactual import internal_randomness_audit

        return getattr(internal_randomness_audit, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# The original EventKey remains stable for the existing CLOB and agent APIs.
# PairedEventKey names the richer key used for exogenous counterfactual events.
LegacyEventKey = EventKey
PairedEventKey = SemanticEventKey

__all__ = [
    "EVENT_KEY_SCHEMA_VERSION",
    "EventKey",
    "EventType",
    "ExogenousShock",
    "ExogenousShockTape",
    "GroupOutcome",
    "GroupSpec",
    "HASH_BASED_RNG_IMPLEMENTATION_ID",
    "HashBasedRNG",
    "HorizonOutcome",
    "HorizonSpec",
    "INTERNAL_RANDOM_COUPLING_AUDIT_SCHEMA_VERSION",
    "INTERNAL_RANDOM_COUPLING_CLAIM_SCOPE",
    "InternalRandomCouplingAudit",
    "InternalRandomCouplingAuditError",
    "LegacyEventKey",
    "MAX_SCENARIO_GENERATION_DRAWS",
    "MetricValue",
    "PairIntegrityAudit",
    "PairIntegrityError",
    "PairedDifference",
    "PairedEventKey",
    "PairedOutcome",
    "PolicyDefinition",
    "PolicyWorldPair",
    "PosteriorDraw",
    "RandomnessDesign",
    "SCENARIO_GENERATION_DRAW_SCHEMA_VERSION",
    "SCENARIO_GENERATION_RECEIPT_SCHEMA_VERSION",
    "ScenarioEventReceipt",
    "ScenarioGenerationDraw",
    "ScenarioGenerationReceipt",
    "ScenarioGeneratorSpec",
    "SemanticEventError",
    "SemanticEventKey",
    "ShockAccessCapability",
    "ShockAccessError",
    "ShockAccessViolation",
    "ShockAccessViolationCode",
    "ShockConsumption",
    "ShockConsumptionLedger",
    "ShockTransitionConfig",
    "ShockTransitionReceipt",
    "ShockTransitionResult",
    "ShockTransitionState",
    "StateArtifact",
    "WorldOutcome",
    "WorldRole",
    "WorldRunRequest",
    "WorldStatus",
    "compare_randomness_designs",
    "deterministic_int",
    "hash_integer",
    "hash_based_rng_live_dependency_sha256",
    "hash_uniform",
    "run_policy_world_pairs",
    "validate_exact_exogenous_shock_tape",
    "validate_exact_policy_world_pair",
    "validate_exact_scenario_generation_receipt",
    "require_verifier_owned_internal_random_coupling_audit",
]
