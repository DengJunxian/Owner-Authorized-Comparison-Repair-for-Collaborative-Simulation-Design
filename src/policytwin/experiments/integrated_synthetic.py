"""Frozen builders for the synthetic integrated PolicyTwin mechanism test."""

from __future__ import annotations

import inspect
import json
import math
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from threading import RLock
from typing import Literal, NoReturn, cast
from weakref import WeakKeyDictionary, WeakValueDictionary, finalize

import policytwin.counterfactual.events as counterfactual_events_module
from policytwin.agents import (
    AdapterArtifactKind,
    AdapterAuthorization,
    AgentArchetype,
    DeterministicMockLLMAdapter,
    DeterministicMockPlanningHook,
    LeakProbeDefinition,
    LeakProbeKind,
    LeakProbeManifest,
    LeakProbeReport,
    LeakProbeResult,
    ModelWeightProvenance,
    OrderPolicyConfig,
    RiskLimits,
    StructuredAgent,
    deterministic_mock_adapter_artifact_mapping,
    deterministic_mock_adapter_artifact_sha256,
)
from policytwin.agents.mock_planning import (
    issue_verifier_owned_integrated_mock_context,
)
from policytwin.behavior import BehaviorMechanisms, BehaviorParameters
from policytwin.cognition import PlanAction, RequestedUse
from policytwin.counterfactual.events import (
    HASH_BASED_RNG_IMPLEMENTATION_ID,
    MAX_SCENARIO_GENERATION_DRAWS,
    SCENARIO_GENERATION_DRAW_SCHEMA_VERSION,
    SCENARIO_GENERATION_RECEIPT_SCHEMA_VERSION,
    EventType,
    ExogenousShock,
    ExogenousShockTape,
    HashBasedRNG,
    ScenarioGenerationDraw,
    ScenarioGenerationReceipt,
    ScenarioGeneratorSpec,
    SemanticEventError,
    SemanticEventKey,
    ShockTransitionConfig,
    _issue_scenario_generation_receipt,
    hash_based_rng_live_dependency_sha256,
)
from policytwin.counterfactual.integrated_market import (
    INTEGRATED_MODEL_ID,
    IntegratedSyntheticMarketConfig,
    IntegratedSyntheticRunArtifact,
    IntegratedSyntheticWorldFactory,
    IntegratedSyntheticWorldRunner,
)
from policytwin.counterfactual.internal_randomness_audit import (
    InternalRandomCouplingAudit,
    InternalRandomCouplingAuditError,
    _issue_internal_random_audit_mint_ticket,
    _mint_internal_random_coupling_audit,
    require_verifier_owned_internal_random_coupling_audit,
)
from policytwin.counterfactual.paired import (
    GroupSpec,
    HorizonSpec,
    PairedDifference,
    PairedOutcome,
    PairValidationError,
    PolicyDefinition,
    PolicyWorldPair,
    PosteriorDraw,
    StateArtifact,
    WorldRole,
    validate_exact_policy_world_pair,
)
from policytwin.counterfactual.randomness import EventKey
from policytwin.implementation_fingerprint import (
    live_policytwin_implementation_sha256,
)
from policytwin.market import (
    AccountBalance,
    MarketConfig,
    OpeningOrder,
    OpeningState,
    Side,
)
from policytwin.protocol import (
    ClaimScope,
    CohortBasis,
    ProtocolExecutionAuthorization,
    ProtocolExecutionConsumer,
    ProtocolExecutionPurpose,
    ProtocolExecutionReceipt,
    StudyMode,
    StudyProtocol,
    consume_protocol_execution_authorization,
    validate_protocol_execution_receipt_binding,
)
from policytwin.provenance import (
    MetadataValidationError,
    canonical_json_bytes,
    read_json_bytes_object,
    sha256_bytes,
    sha256_file,
)
from policytwin.social import (
    DeidentificationPolicy,
    SyntheticSocialGraph,
    WeightedDirectedEdge,
)

_CUTOFF = datetime(2025, 1, 10, tzinfo=UTC)
_FIRST_EVENT = datetime(2025, 1, 2, tzinfo=UTC)
INTEGRATED_PROTOCOL_RUN_SCHEMA_VERSION = (
    "policytwin.protocol-bound-integrated-synthetic-run.v2"
)
INTEGRATED_PAIR_INPUT_SCHEMA_VERSION = "policytwin.integrated-pair-input.v2"
INTEGRATED_ESTIMAND_ID = "paired_distribution"
INTEGRATED_METRIC_IDS = (
    "cash",
    "filled_quantity",
    "inventory",
    "market_trade_volume",
    "mean_buy_probability",
    "mean_turnover_probability",
    "memory_entries",
    "net_worth",
    "quoted_spread",
    "social_belief",
    "submitted_orders",
)
_INTEGRATED_METRIC_SEMANTICS = {
    "cash": ("sum member cash", "cash_ticks", "cohort_sum_at_horizon"),
    "filled_quantity": (
        "sum trade quantity once per cohort-member side",
        "asset_units",
        "cumulative_cohort_sum_through_horizon",
    ),
    "inventory": ("sum member asset balance", "asset_units", "cohort_sum"),
    "market_trade_volume": (
        "sum all market trade quantity",
        "asset_units",
        "cumulative_market_sum_through_horizon",
    ),
    "mean_buy_probability": (
        "arithmetic mean of member decision buy probabilities",
        "probability",
        "cohort_decision_mean_through_horizon",
    ),
    "mean_turnover_probability": (
        "arithmetic mean of member buy plus sell probabilities",
        "probability",
        "cohort_decision_mean_through_horizon",
    ),
    "memory_entries": (
        "sum member episodic-memory entry counts",
        "entries",
        "cohort_sum_at_horizon",
    ),
    "net_worth": (
        "cohort cash plus cohort inventory times integer midquote",
        "cash_ticks",
        "cohort_sum_at_horizon",
    ),
    "quoted_spread": (
        "best ask minus best bid",
        "price_ticks",
        "market_snapshot_at_horizon",
    ),
    "social_belief": (
        "arithmetic mean member belief, exactly zero when none are present",
        "belief_units",
        "cohort_mean_at_horizon",
    ),
    "submitted_orders": (
        "count submitted agent commands belonging to cohort members",
        "orders",
        "cumulative_cohort_count_through_horizon",
    ),
}
_PROTOCOL_BOUND_SOURCE_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_PROTOCOL_BOUND_SOURCE_BYTES = 256 * 1024 * 1024


class IntegratedProtocolExecutionError(RuntimeError):
    """Raised when a study protocol does not describe the integrated run."""


class ProtocolBoundArtifactOrigin(StrEnum):
    """Same-process registry origin; neither value is external attestation."""

    EXECUTED_HERE = "executed_here"
    LOADED_AND_REPLAYED = "loaded_and_replayed"


class _ProtocolBoundArtifactMintTicket:
    """Opaque single-use authority for exactly one artifact origin/binding."""

    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("protocol-bound artifact mint tickets are verifier-issued")

    def __copy__(self) -> NoReturn:
        raise TypeError("protocol-bound artifact mint tickets cannot be copied")

    def __deepcopy__(self, memo: object) -> NoReturn:
        del memo
        raise TypeError("protocol-bound artifact mint tickets cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("protocol-bound artifact mint tickets cannot be serialized")


@dataclass(frozen=True, slots=True)
class _ProtocolBoundArtifactMintBinding:
    origin: ProtocolBoundArtifactOrigin
    protocol_sha256: str
    config_sha256: str
    input_sha256: str
    receipt_sha256: str
    expected_artifact_sha256: str | None


def integrated_metric_definition_sha256(metric_id: str) -> str:
    """Return the verifier-owned formula/unit/source commitment for a metric."""
    if metric_id not in _INTEGRATED_METRIC_SEMANTICS:
        raise IntegratedProtocolExecutionError(
            f"unknown integrated metric definition: {metric_id}"
        )
    formula, unit, aggregation = _INTEGRATED_METRIC_SEMANTICS[metric_id]
    implementation_source_sha256 = sha256_bytes(
        inspect.getsource(IntegratedSyntheticWorldRunner._group_outcome).encode("utf-8")
    )
    return sha256_bytes(
        canonical_json_bytes(
            {
                "aggregation": aggregation,
                "formula": formula,
                "implementation_id": "policytwin.integrated-group-outcome.v1",
                "implementation_source_sha256": implementation_source_sha256,
                "metric_id": metric_id,
                "registry_schema_version": 1,
                "unit": unit,
            }
        )
    )


def integrated_paired_estimand_definition_sha256() -> str:
    """Commit to the per-cell candidate-minus-baseline paired estimand."""
    implementation_source_sha256 = sha256_bytes(
        inspect.getsource(PairedDifference.__post_init__).encode("utf-8")
    )
    return sha256_bytes(
        canonical_json_bytes(
            {
                "aggregation": "per_draw_horizon_cohort_metric_distribution",
                "formula": "candidate_value - baseline_value",
                "implementation_id": "policytwin.paired-difference.v1",
                "implementation_source_sha256": implementation_source_sha256,
                "registry_schema_version": 1,
                "unit": "same_as_registered_metric",
            }
        )
    )


def _pair_input_mapping(pair: PolicyWorldPair) -> dict[str, object]:
    return {
        "baseline_policy": pair.baseline_policy.to_mapping(),
        "candidate_policy": pair.candidate_policy.to_mapping(),
        "groups": [item.to_mapping() for item in pair.groups],
        "horizons": [item.to_mapping() for item in pair.horizons],
        "initial_state": pair.initial_state.to_mapping(),
        "numeric_tolerance": pair.numeric_tolerance,
        "posterior_draw": pair.posterior_draw.to_mapping(),
        "scenario_generation_receipt": (
            None
            if pair.scenario_generation_receipt is None
            else pair.scenario_generation_receipt.to_mapping()
        ),
        "scenario_id": pair.scenario_id,
        "schema_version": INTEGRATED_PAIR_INPUT_SCHEMA_VERSION,
        "shock_tape": pair.shock_tape.to_mapping(),
        "shock_transition_config": pair.shock_transition_config.to_mapping(),
        "synthetic_only": pair.synthetic_only,
    }


def integrated_synthetic_pair_input_sha256(pair: PolicyWorldPair) -> str:
    """Commit to every paired-run input before authorization is issued."""
    if type(pair) is not PolicyWorldPair:
        raise TypeError("pair must be a PolicyWorldPair")
    try:
        validate_exact_policy_world_pair(pair)
    except PairValidationError as error:
        raise IntegratedProtocolExecutionError(
            "paired input contains non-canonical formal evidence"
        ) from error
    return sha256_bytes(canonical_json_bytes(_pair_input_mapping(pair)))


def _validate_formal_execution_types(
    *,
    pair: PolicyWorldPair,
    config: IntegratedSyntheticMarketConfig,
    protocol: StudyProtocol,
    authorization: ProtocolExecutionAuthorization,
) -> None:
    """Reject polymorphic evidence before consuming single-use authority."""
    if type(pair) is not PolicyWorldPair:
        raise TypeError("pair must use the exact PolicyWorldPair type")
    if type(config) is not IntegratedSyntheticMarketConfig:
        raise TypeError(
            "config must use the exact IntegratedSyntheticMarketConfig type"
        )
    if type(protocol) is not StudyProtocol:
        raise TypeError("protocol must use the exact StudyProtocol type")
    if type(authorization) is not ProtocolExecutionAuthorization:
        raise TypeError(
            "authorization must use the exact ProtocolExecutionAuthorization type"
        )
    if type(pair.shock_tape) is not ExogenousShockTape:
        raise IntegratedProtocolExecutionError(
            "formal execution requires the exact ExogenousShockTape type"
        )
    receipt = pair.scenario_generation_receipt
    if receipt is not None and type(receipt) is not ScenarioGenerationReceipt:
        raise IntegratedProtocolExecutionError(
            "formal execution requires the exact ScenarioGenerationReceipt type"
        )
    spec = config.scenario_generator_spec
    if spec is not None and type(spec) is not ScenarioGeneratorSpec:
        raise IntegratedProtocolExecutionError(
            "formal execution requires the exact ScenarioGeneratorSpec type"
        )
    try:
        validate_exact_policy_world_pair(pair)
    except PairValidationError as error:
        raise IntegratedProtocolExecutionError(
            "formal paired evidence failed exact canonical reconstruction"
        ) from error


def _validate_protocol_semantics(
    *,
    protocol: StudyProtocol,
    config: IntegratedSyntheticMarketConfig,
    pair: PolicyWorldPair,
) -> None:
    if type(protocol) is not StudyProtocol:
        raise IntegratedProtocolExecutionError(
            "formal execution requires the exact StudyProtocol type"
        )
    if type(config) is not IntegratedSyntheticMarketConfig:
        raise IntegratedProtocolExecutionError(
            "formal execution requires the exact integrated config type"
        )
    if type(pair) is not PolicyWorldPair:
        raise IntegratedProtocolExecutionError(
            "formal execution requires the exact PolicyWorldPair type"
        )
    if protocol.mode is not StudyMode.SYNTHETIC:
        raise IntegratedProtocolExecutionError(
            "integrated runner accepts only a synthetic study protocol"
        )
    if protocol.claim_scope is not ClaimScope.SYNTHETIC_ONLY:
        raise IntegratedProtocolExecutionError(
            "integrated runner cannot authorize a non-synthetic claim"
        )
    if protocol.knowledge_cutoff != config.cutoff:
        raise IntegratedProtocolExecutionError(
            "study protocol cutoff differs from the integrated configuration"
        )
    out_of_window = tuple(
        item.event_key.value
        for item in pair.shock_tape.entries
        if not protocol.policy_start <= item.event_key.timestamp <= protocol.policy_end
    )
    if out_of_window:
        raise IntegratedProtocolExecutionError(
            "every integrated policy-simulation shock must fall within the "
            "registered policy window"
        )
    if (
        protocol.estimand.definition_id != INTEGRATED_ESTIMAND_ID
        or protocol.estimand.definition_sha256
        != integrated_paired_estimand_definition_sha256()
    ):
        raise IntegratedProtocolExecutionError(
            "study protocol estimand differs from the integrated paired estimand"
        )
    metric_definitions = {item.definition_id: item for item in protocol.metrics}
    if set(metric_definitions) != set(INTEGRATED_METRIC_IDS):
        raise IntegratedProtocolExecutionError(
            "study protocol metric registry differs from the integrated runner"
        )
    for metric_id in INTEGRATED_METRIC_IDS:
        if metric_definitions[
            metric_id
        ].definition_sha256 != integrated_metric_definition_sha256(metric_id):
            raise IntegratedProtocolExecutionError(
                "study protocol metric definition differs from its verifier-owned "
                "formula, unit, aggregation, or implementation"
            )
    expected_model_draw = sha256_bytes(
        canonical_json_bytes(
            {
                "config_sha256": config.sha256,
                "model": INTEGRATED_MODEL_ID,
            }
        )
    )
    if pair.posterior_draw.model_draw_sha256 != expected_model_draw:
        raise IntegratedProtocolExecutionError(
            "paired model draw is not bound to the integrated configuration"
        )
    if (
        pair.shock_transition_config.state_field != config.transition_state_field
        or pair.shock_transition_config.shock_field != config.transition_shock_field
    ):
        raise IntegratedProtocolExecutionError(
            "paired shock transition differs from the integrated configuration"
        )
    if (
        protocol.baseline_policy.definition_id != pair.baseline_policy.policy_id
        or protocol.baseline_policy.definition_sha256 != pair.baseline_policy.sha256
        or protocol.candidate_policy.definition_id != pair.candidate_policy.policy_id
        or protocol.candidate_policy.definition_sha256 != pair.candidate_policy.sha256
    ):
        raise IntegratedProtocolExecutionError(
            "study protocol policy definitions differ from the paired input"
        )

    protocol_cohorts = {item.cohort_id: item for item in protocol.cohorts}
    pair_groups = {item.group_id: item for item in pair.groups}
    if set(protocol_cohorts) != set(pair_groups):
        raise IntegratedProtocolExecutionError(
            "study protocol cohorts differ from the paired groups"
        )
    for group_id, group in pair_groups.items():
        cohort = protocol_cohorts[group_id]
        if (
            cohort.definition_sha256 != group.cohort_sha256
            or cohort.basis is not CohortBasis.LATENT_STRATEGY
        ):
            raise IntegratedProtocolExecutionError(
                "synthetic strategy groups must remain explicitly latent cohorts"
            )

    protocol_horizons = {item.definition_id: item for item in protocol.horizons}
    pair_horizons = {item.horizon_id: item for item in pair.horizons}
    if set(protocol_horizons) != set(pair_horizons):
        raise IntegratedProtocolExecutionError(
            "study protocol horizons differ from the paired horizons"
        )
    for horizon_id, horizon in pair_horizons.items():
        expected = sha256_bytes(canonical_json_bytes(horizon.to_mapping()))
        if protocol_horizons[horizon_id].definition_sha256 != expected:
            raise IntegratedProtocolExecutionError(
                "study protocol horizon commitment differs from the paired input"
            )
    spec = config.scenario_generator_spec
    if spec is None:
        raise IntegratedProtocolExecutionError(
            "protocol-bound policy simulation requires a frozen forward generator"
        )
    if spec.frozen_at > protocol.model_frozen_at:
        raise IntegratedProtocolExecutionError(
            "scenario generator was not frozen by the protocol model-freeze time"
        )


def _validate_registered_outputs(
    protocol: StudyProtocol,
    outcome: PairedOutcome,
) -> None:
    observed = {
        (item.horizon_id, item.group_id, item.metric_name)
        for item in outcome.differences
    }
    registered = {
        (item.horizon_id, item.cohort_id, item.metric_id) for item in protocol.outputs
    }
    missing = sorted(registered - observed)
    unregistered = sorted(observed - registered)
    if missing or unregistered:
        raise IntegratedProtocolExecutionError(
            "integrated output cells must exactly match the study protocol; "
            f"missing={missing}, unregistered={unregistered}"
        )


@dataclass(frozen=True, slots=True, init=False, weakref_slot=True)
class ProtocolBoundIntegratedSyntheticRunArtifact:
    """Protocol receipt plus the complete paired and per-world run evidence."""

    protocol: StudyProtocol
    config: IntegratedSyntheticMarketConfig
    pair_input: PolicyWorldPair
    protocol_execution_receipt: ProtocolExecutionReceipt
    protocol_canonical_json: str
    config_sha256: str
    config_canonical_json: str
    input_sha256: str
    pair_input_canonical_json: str
    paired_outcome: PairedOutcome
    baseline_run: IntegratedSyntheticRunArtifact
    candidate_run: IntegratedSyntheticRunArtifact
    internal_random_coupling_audit: InternalRandomCouplingAudit
    cross_world_internal_random_coupling_verified: Literal[True] = True
    external_validity_evidence: Literal[False] = False
    synthetic_only: Literal[True] = True
    schema_version: str = INTEGRATED_PROTOCOL_RUN_SCHEMA_VERSION

    def __init__(self) -> None:
        raise TypeError(
            "ProtocolBoundIntegratedSyntheticRunArtifact is verifier-produced; "
            "use run_protocol_bound_integrated_synthetic_pair"
        )

    def _fresh_validate(self) -> None:
        """Revalidate every binding and deterministically replay both worlds."""
        if not isinstance(self.protocol, StudyProtocol):
            raise IntegratedProtocolExecutionError(
                "integrated artifact requires a typed study protocol"
            )
        if not isinstance(self.config, IntegratedSyntheticMarketConfig):
            raise IntegratedProtocolExecutionError(
                "integrated artifact requires a typed market configuration"
            )
        if not isinstance(self.pair_input, PolicyWorldPair):
            raise IntegratedProtocolExecutionError(
                "integrated artifact requires typed paired-run inputs"
            )
        receipt = self.protocol_execution_receipt
        if not isinstance(receipt, ProtocolExecutionReceipt):
            raise IntegratedProtocolExecutionError(
                "integrated artifact requires a typed protocol execution receipt"
            )
        if (
            receipt.consumer
            is not ProtocolExecutionConsumer.INTEGRATED_SYNTHETIC_SIMULATION
            or receipt.purpose
            is not ProtocolExecutionPurpose.PAIRED_SYNTHETIC_POLICY_SIMULATION
        ):
            raise IntegratedProtocolExecutionError(
                "protocol receipt has the wrong integrated consumer or purpose"
            )
        if ProtocolExecutionReceipt.from_mapping(receipt.to_mapping()) != receipt:
            raise IntegratedProtocolExecutionError(
                "protocol receipt does not survive strict reconstruction"
            )
        validate_protocol_execution_receipt_binding(self.protocol, receipt)
        try:
            protocol_value = json.loads(self.protocol_canonical_json)
        except (json.JSONDecodeError, TypeError) as error:
            raise IntegratedProtocolExecutionError(
                "integrated artifact protocol must be canonical JSON"
            ) from error
        if (
            canonical_json_bytes(protocol_value).decode("utf-8")
            != self.protocol_canonical_json
            or protocol_value != self.protocol.to_mapping()
            or sha256_bytes(self.protocol_canonical_json.encode("utf-8"))
            != receipt.protocol_sha256
            or self.protocol.sha256 != receipt.protocol_sha256
        ):
            raise IntegratedProtocolExecutionError(
                "integrated artifact protocol does not match its receipt"
            )
        if (
            receipt.config_sha256 != self.config_sha256
            or receipt.input_sha256 != self.input_sha256
        ):
            raise IntegratedProtocolExecutionError(
                "protocol receipt does not bind the artifact config and input"
            )
        try:
            config_value = json.loads(self.config_canonical_json)
            input_value = json.loads(self.pair_input_canonical_json)
        except (json.JSONDecodeError, TypeError) as error:
            raise IntegratedProtocolExecutionError(
                "integrated artifact config and input must be canonical JSON"
            ) from error
        if (
            canonical_json_bytes(config_value).decode("utf-8")
            != self.config_canonical_json
            or config_value != self.config.to_mapping()
            or sha256_bytes(self.config_canonical_json.encode("utf-8"))
            != self.config_sha256
            or self.config.sha256 != self.config_sha256
        ):
            raise IntegratedProtocolExecutionError(
                "integrated artifact configuration does not match its hash"
            )
        if (
            canonical_json_bytes(input_value).decode("utf-8")
            != self.pair_input_canonical_json
            or input_value != _pair_input_mapping(self.pair_input)
            or sha256_bytes(self.pair_input_canonical_json.encode("utf-8"))
            != self.input_sha256
            or integrated_synthetic_pair_input_sha256(self.pair_input)
            != self.input_sha256
        ):
            raise IntegratedProtocolExecutionError(
                "integrated artifact pair input does not match its hash"
            )
        if self.synthetic_only is not True:
            raise IntegratedProtocolExecutionError(
                "protocol-bound integrated artifacts must remain synthetic-only"
            )
        if self.external_validity_evidence is not False:
            raise IntegratedProtocolExecutionError(
                "protocol-bound internal random coupling is not external evidence"
            )
        if self.cross_world_internal_random_coupling_verified is not True:
            raise IntegratedProtocolExecutionError(
                "protocol-bound artifact lacks paired internal random coupling proof"
            )
        if self.schema_version != INTEGRATED_PROTOCOL_RUN_SCHEMA_VERSION:
            raise IntegratedProtocolExecutionError(
                "unsupported protocol-bound integrated artifact schema"
            )
        if (
            self.baseline_run.role is not WorldRole.BASELINE
            or self.candidate_run.role is not WorldRole.CANDIDATE
        ):
            raise IntegratedProtocolExecutionError(
                "integrated world artifacts have incorrect roles"
            )
        for run in (self.baseline_run, self.candidate_run):
            if (
                run.config_sha256 != self.config_sha256
                or run.parameter_sha256 != self.paired_outcome.parameter_sha256
                or run.shock_tape_sha256 != self.paired_outcome.shock_tape_sha256
                or run.initial_state_sha256 != self.paired_outcome.initial_state_sha256
            ):
                raise IntegratedProtocolExecutionError(
                    "integrated world artifact differs from the paired outcome"
                )
        if (
            self.baseline_run.policy_sha256
            != self.paired_outcome.baseline_policy_sha256
            or self.candidate_run.policy_sha256
            != self.paired_outcome.candidate_policy_sha256
        ):
            raise IntegratedProtocolExecutionError(
                "integrated world policy commitments differ from the paired outcome"
            )
        baseline_state = self.paired_outcome.baseline.final_state.to_mapping()
        candidate_state = self.paired_outcome.candidate.final_state.to_mapping()
        if (
            baseline_state.get("integrated_run_sha256") != self.baseline_run.sha256
            or candidate_state.get("integrated_run_sha256") != self.candidate_run.sha256
        ):
            raise IntegratedProtocolExecutionError(
                "paired final states do not bind the integrated world artifacts"
            )

        if type(self.internal_random_coupling_audit) is not InternalRandomCouplingAudit:
            raise IntegratedProtocolExecutionError(
                "integrated artifact requires an exact internal random coupling audit"
            )
        try:
            require_verifier_owned_internal_random_coupling_audit(
                self.internal_random_coupling_audit,
                baseline_run=self.baseline_run,
                candidate_run=self.candidate_run,
                protocol=self.protocol,
                config=self.config,
                pair_input=self.pair_input,
                protocol_execution_receipt=receipt,
            )
        except InternalRandomCouplingAuditError as error:
            raise IntegratedProtocolExecutionError(
                "paired internal random coupling verification failed"
            ) from error

        _validate_protocol_semantics(
            protocol=self.protocol,
            config=self.config,
            pair=self.pair_input,
        )
        _validate_registered_outputs(self.protocol, self.paired_outcome)
        if self.paired_outcome.pair_input != self.pair_input:
            raise IntegratedProtocolExecutionError(
                "paired outcome does not retain the authorized typed pair input"
            )

        replay_factory = IntegratedSyntheticWorldFactory(self.config)
        replayed_outcome = self.pair_input.run(replay_factory)
        replayed_baseline = replay_factory.runners[WorldRole.BASELINE].artifact
        replayed_candidate = replay_factory.runners[WorldRole.CANDIDATE].artifact
        if replayed_baseline is None or replayed_candidate is None:
            raise IntegratedProtocolExecutionError(
                "integrated artifact verification replay did not complete"
            )
        if (
            self.paired_outcome != replayed_outcome
            or self.baseline_run != replayed_baseline
            or self.candidate_run != replayed_candidate
        ):
            raise IntegratedProtocolExecutionError(
                "integrated artifact does not reproduce under deterministic replay"
            )

    def to_mapping(self) -> dict[str, object]:
        """Return complete canonical audit evidence, including the receipt."""
        return {
            "baseline_run": self.baseline_run.to_mapping(),
            "candidate_run": self.candidate_run.to_mapping(),
            "config": json.loads(self.config_canonical_json),
            "config_sha256": self.config_sha256,
            "cross_world_internal_random_coupling_verified": (
                self.cross_world_internal_random_coupling_verified
            ),
            "external_validity_evidence": self.external_validity_evidence,
            "input_sha256": self.input_sha256,
            "internal_random_coupling_audit": (
                self.internal_random_coupling_audit.to_mapping()
            ),
            "pair_input": json.loads(self.pair_input_canonical_json),
            "paired_outcome": self.paired_outcome.to_mapping(),
            "protocol": json.loads(self.protocol_canonical_json),
            "protocol_execution_receipt": (
                self.protocol_execution_receipt.to_mapping()
            ),
            "schema_version": self.schema_version,
            "synthetic_only": self.synthetic_only,
        }

    @property
    def canonical_bytes(self) -> bytes:
        """Return deterministic bytes for storage and later hash audit."""
        return canonical_json_bytes(self.to_mapping())

    @property
    def sha256(self) -> str:
        """Commit to protocol authority and all generated run evidence."""
        return sha256_bytes(self.canonical_bytes)


_VERIFIER_OWNED_INTEGRATED_ARTIFACTS: WeakValueDictionary[
    int, ProtocolBoundIntegratedSyntheticRunArtifact
] = WeakValueDictionary()
_VERIFIER_OWNED_INTEGRATED_ARTIFACT_SEALS: dict[int, str] = {}
_VERIFIER_OWNED_INTEGRATED_ARTIFACT_ORIGINS: dict[int, ProtocolBoundArtifactOrigin] = {}
_VERIFIER_OWNED_INTEGRATED_ARTIFACTS_LOCK = RLock()
_PROTOCOL_BOUND_ARTIFACT_MINT_TICKETS: WeakKeyDictionary[
    _ProtocolBoundArtifactMintTicket, _ProtocolBoundArtifactMintBinding
] = WeakKeyDictionary()


def _consume_integrated_execution_authorization_for_mint(
    *,
    authorization: ProtocolExecutionAuthorization,
    protocol: StudyProtocol,
    config_sha256: str,
    input_sha256: str,
    execution_started_at: datetime,
) -> tuple[ProtocolExecutionReceipt, _ProtocolBoundArtifactMintTicket]:
    """Consume execution authority and derive the only executed-origin ticket."""
    receipt = consume_protocol_execution_authorization(
        authorization,
        protocol=protocol,
        consumer=ProtocolExecutionConsumer.INTEGRATED_SYNTHETIC_SIMULATION,
        purpose=ProtocolExecutionPurpose.PAIRED_SYNTHETIC_POLICY_SIMULATION,
        config_sha256=config_sha256,
        input_sha256=input_sha256,
        execution_started_at=execution_started_at,
    )
    binding = _ProtocolBoundArtifactMintBinding(
        origin=ProtocolBoundArtifactOrigin.EXECUTED_HERE,
        protocol_sha256=protocol.sha256,
        config_sha256=config_sha256,
        input_sha256=input_sha256,
        receipt_sha256=receipt.sha256,
        expected_artifact_sha256=None,
    )
    ticket = object.__new__(_ProtocolBoundArtifactMintTicket)
    with _VERIFIER_OWNED_INTEGRATED_ARTIFACTS_LOCK:
        _PROTOCOL_BOUND_ARTIFACT_MINT_TICKETS[ticket] = binding
    return receipt, ticket


def _issue_loaded_replay_artifact_mint_ticket(
    *,
    protocol: StudyProtocol,
    config_sha256: str,
    input_sha256: str,
    receipt: ProtocolExecutionReceipt,
    expected_artifact_sha256: str,
) -> _ProtocolBoundArtifactMintTicket:
    """Issue a replay-only ticket that can never mint executed authority."""
    binding = _ProtocolBoundArtifactMintBinding(
        origin=ProtocolBoundArtifactOrigin.LOADED_AND_REPLAYED,
        protocol_sha256=protocol.sha256,
        config_sha256=config_sha256,
        input_sha256=input_sha256,
        receipt_sha256=receipt.sha256,
        expected_artifact_sha256=expected_artifact_sha256,
    )
    ticket = object.__new__(_ProtocolBoundArtifactMintTicket)
    with _VERIFIER_OWNED_INTEGRATED_ARTIFACTS_LOCK:
        _PROTOCOL_BOUND_ARTIFACT_MINT_TICKETS[ticket] = binding
    return ticket


def _forget_protocol_bound_integrated_artifact(identity: int) -> None:
    with _VERIFIER_OWNED_INTEGRATED_ARTIFACTS_LOCK:
        if _VERIFIER_OWNED_INTEGRATED_ARTIFACTS.get(identity) is None:
            _VERIFIER_OWNED_INTEGRATED_ARTIFACT_SEALS.pop(identity, None)
            _VERIFIER_OWNED_INTEGRATED_ARTIFACT_ORIGINS.pop(identity, None)


def _require_registered_protocol_bound_integrated_artifact(
    artifact: ProtocolBoundIntegratedSyntheticRunArtifact,
    *,
    allowed_origins: frozenset[ProtocolBoundArtifactOrigin],
) -> ProtocolBoundIntegratedSyntheticRunArtifact:
    """Require a sealed same-process identity from one explicit origin set."""
    if type(artifact) is not ProtocolBoundIntegratedSyntheticRunArtifact:
        raise IntegratedProtocolExecutionError(
            "integrated artifact has the wrong concrete type"
        )
    try:
        before_sha256 = artifact.sha256
    except Exception as error:
        raise IntegratedProtocolExecutionError(
            "integrated artifact cannot reproduce its mint-time seal"
        ) from error
    with _VERIFIER_OWNED_INTEGRATED_ARTIFACTS_LOCK:
        registered = _VERIFIER_OWNED_INTEGRATED_ARTIFACTS.get(id(artifact))
        mint_seal = _VERIFIER_OWNED_INTEGRATED_ARTIFACT_SEALS.get(id(artifact))
        origin = _VERIFIER_OWNED_INTEGRATED_ARTIFACT_ORIGINS.get(id(artifact))
    if registered is not artifact or mint_seal != before_sha256 or origin is None:
        raise IntegratedProtocolExecutionError(
            "integrated artifact was not produced by the verifier-owned runner "
            "or strict replay loader, or differs from its mint-time seal"
        )
    if origin not in allowed_origins:
        raise IntegratedProtocolExecutionError(
            "integrated artifact origin is not authorized for this operation; "
            f"origin={origin.value}"
        )
    artifact._fresh_validate()
    try:
        after_sha256 = artifact.sha256
    except Exception as error:
        raise IntegratedProtocolExecutionError(
            "integrated artifact changed during mint-seal validation"
        ) from error
    with _VERIFIER_OWNED_INTEGRATED_ARTIFACTS_LOCK:
        still_registered = _VERIFIER_OWNED_INTEGRATED_ARTIFACTS.get(id(artifact))
        current_seal = _VERIFIER_OWNED_INTEGRATED_ARTIFACT_SEALS.get(id(artifact))
        current_origin = _VERIFIER_OWNED_INTEGRATED_ARTIFACT_ORIGINS.get(id(artifact))
    if (
        still_registered is not artifact
        or before_sha256 != after_sha256
        or current_seal != after_sha256
        or current_origin is not origin
    ):
        raise IntegratedProtocolExecutionError(
            "integrated artifact changed after its verifier-owned mint"
        )
    return artifact


def require_verifier_owned_protocol_bound_integrated_artifact(
    artifact: ProtocolBoundIntegratedSyntheticRunArtifact,
) -> ProtocolBoundIntegratedSyntheticRunArtifact:
    """Require execution in this process, then replay all artifact evidence.

    Serialized receipts are audit evidence rather than minting authority.  This
    same-process identity check therefore rejects objects assembled with
    ``object.__new__`` or copied field-by-field, even when their public fields
    are internally consistent.  A source reconstructed by the strict historical
    loader is deliberately rejected because it was not executed here under an
    opaque authorization.
    """
    return _require_registered_protocol_bound_integrated_artifact(
        artifact,
        allowed_origins=frozenset({ProtocolBoundArtifactOrigin.EXECUTED_HERE}),
    )


def require_ledger_derivation_source_artifact(
    artifact: ProtocolBoundIntegratedSyntheticRunArtifact,
) -> ProtocolBoundIntegratedSyntheticRunArtifact:
    """Accept executed or strictly loaded sources only for pure ledger derivation.

    This function does not authorize simulation, prove historical execution, or
    upgrade any external, authorization, welfare, identity, or causal claim.
    """
    return _require_registered_protocol_bound_integrated_artifact(
        artifact,
        allowed_origins=frozenset(
            {
                ProtocolBoundArtifactOrigin.EXECUTED_HERE,
                ProtocolBoundArtifactOrigin.LOADED_AND_REPLAYED,
            }
        ),
    )


def _mint_protocol_bound_integrated_artifact(
    *,
    mint_ticket: _ProtocolBoundArtifactMintTicket,
    protocol: StudyProtocol,
    config: IntegratedSyntheticMarketConfig,
    pair_input: PolicyWorldPair,
    protocol_execution_receipt: ProtocolExecutionReceipt,
    protocol_canonical_json: str,
    config_sha256: str,
    config_canonical_json: str,
    input_sha256: str,
    pair_input_canonical_json: str,
    paired_outcome: PairedOutcome,
    baseline_run: IntegratedSyntheticRunArtifact,
    candidate_run: IntegratedSyntheticRunArtifact,
) -> ProtocolBoundIntegratedSyntheticRunArtifact:
    if type(mint_ticket) is not _ProtocolBoundArtifactMintTicket:
        raise IntegratedProtocolExecutionError(
            "artifact mint requires an exact verifier-issued ticket"
        )
    with _VERIFIER_OWNED_INTEGRATED_ARTIFACTS_LOCK:
        binding = _PROTOCOL_BOUND_ARTIFACT_MINT_TICKETS.pop(mint_ticket, None)
    if binding is None:
        raise IntegratedProtocolExecutionError(
            "artifact mint ticket was not verifier-issued or was already consumed"
        )
    if (
        binding.protocol_sha256 != protocol.sha256
        or binding.config_sha256 != config_sha256
        or binding.input_sha256 != input_sha256
        or binding.receipt_sha256 != protocol_execution_receipt.sha256
    ):
        raise IntegratedProtocolExecutionError(
            "artifact mint fields differ from the consumed ticket binding"
        )
    try:
        internal_random_audit_ticket = _issue_internal_random_audit_mint_ticket(
            baseline_run=baseline_run,
            candidate_run=candidate_run,
            protocol=protocol,
            config=config,
            pair_input=pair_input,
            protocol_execution_receipt=protocol_execution_receipt,
        )
        internal_random_coupling_audit = _mint_internal_random_coupling_audit(
            mint_ticket=internal_random_audit_ticket,
            baseline_run=baseline_run,
            candidate_run=candidate_run,
            protocol=protocol,
            config=config,
            pair_input=pair_input,
            protocol_execution_receipt=protocol_execution_receipt,
        )
    except InternalRandomCouplingAuditError as error:
        raise IntegratedProtocolExecutionError(
            "paired internal random coupling verification failed before artifact mint"
        ) from error
    artifact = object.__new__(ProtocolBoundIntegratedSyntheticRunArtifact)
    values: tuple[tuple[str, object], ...] = (
        ("protocol", protocol),
        ("config", config),
        ("pair_input", pair_input),
        ("protocol_execution_receipt", protocol_execution_receipt),
        ("protocol_canonical_json", protocol_canonical_json),
        ("config_sha256", config_sha256),
        ("config_canonical_json", config_canonical_json),
        ("input_sha256", input_sha256),
        ("pair_input_canonical_json", pair_input_canonical_json),
        ("paired_outcome", paired_outcome),
        ("baseline_run", baseline_run),
        ("candidate_run", candidate_run),
        ("internal_random_coupling_audit", internal_random_coupling_audit),
        ("cross_world_internal_random_coupling_verified", True),
        ("external_validity_evidence", False),
        ("synthetic_only", True),
        ("schema_version", INTEGRATED_PROTOCOL_RUN_SCHEMA_VERSION),
    )
    for field, value in values:
        object.__setattr__(artifact, field, value)
    artifact._fresh_validate()
    mint_seal = artifact.sha256
    if (
        binding.expected_artifact_sha256 is not None
        and binding.expected_artifact_sha256 != mint_seal
    ):
        raise IntegratedProtocolExecutionError(
            "replay mint differs from its externally supplied source commitment"
        )
    identity = id(artifact)
    with _VERIFIER_OWNED_INTEGRATED_ARTIFACTS_LOCK:
        _VERIFIER_OWNED_INTEGRATED_ARTIFACTS[identity] = artifact
        _VERIFIER_OWNED_INTEGRATED_ARTIFACT_SEALS[identity] = mint_seal
        _VERIFIER_OWNED_INTEGRATED_ARTIFACT_ORIGINS[identity] = binding.origin
    finalize(artifact, _forget_protocol_bound_integrated_artifact, identity)
    return artifact


def run_protocol_bound_integrated_synthetic_pair(
    *,
    pair: PolicyWorldPair,
    config: IntegratedSyntheticMarketConfig,
    protocol: StudyProtocol,
    authorization: ProtocolExecutionAuthorization,
    execution_started_at: datetime,
    parallel: bool = False,
    invocation_order: tuple[WorldRole, WorldRole] = (
        WorldRole.BASELINE,
        WorldRole.CANDIDATE,
    ),
) -> ProtocolBoundIntegratedSyntheticRunArtifact:
    """Consume exact study authority and run one verifier-owned synthetic pair."""
    _validate_formal_execution_types(
        pair=pair,
        config=config,
        protocol=protocol,
        authorization=authorization,
    )
    _validate_formal_generation_evidence(config=config, pair=pair)
    config_sha256 = config.sha256
    input_sha256 = integrated_synthetic_pair_input_sha256(pair)
    receipt, mint_ticket = _consume_integrated_execution_authorization_for_mint(
        authorization=authorization,
        protocol=protocol,
        config_sha256=config_sha256,
        input_sha256=input_sha256,
        execution_started_at=execution_started_at,
    )
    _validate_protocol_semantics(protocol=protocol, config=config, pair=pair)
    factory = IntegratedSyntheticWorldFactory(config)
    outcome = pair.run(
        factory,
        parallel=parallel,
        invocation_order=invocation_order,
    )
    _validate_registered_outputs(protocol, outcome)
    baseline = factory.runners[WorldRole.BASELINE].artifact
    candidate = factory.runners[WorldRole.CANDIDATE].artifact
    if baseline is None or candidate is None:
        raise IntegratedProtocolExecutionError(
            "integrated runners did not produce complete world artifacts"
        )
    return _mint_protocol_bound_integrated_artifact(
        mint_ticket=mint_ticket,
        protocol=protocol,
        config=config,
        pair_input=pair,
        protocol_execution_receipt=receipt,
        protocol_canonical_json=canonical_json_bytes(protocol.to_mapping()).decode(
            "utf-8"
        ),
        config_sha256=config_sha256,
        config_canonical_json=canonical_json_bytes(config.to_mapping()).decode("utf-8"),
        input_sha256=input_sha256,
        pair_input_canonical_json=canonical_json_bytes(
            _pair_input_mapping(pair)
        ).decode("utf-8"),
        paired_outcome=outcome,
        baseline_run=baseline,
        candidate_run=candidate,
    )


def _protocol_source_stat_fingerprint(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _open_protocol_source_parent(candidate: Path) -> tuple[int, str]:
    """Open every ancestor by descriptor without following symbolic links."""
    if not candidate.is_absolute() or candidate == Path(candidate.anchor):
        raise IntegratedProtocolExecutionError(
            "protocol-bound source path must identify a file"
        )
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor: int | None = None
    try:
        descriptor = os.open(candidate.anchor, directory_flags)
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise IntegratedProtocolExecutionError(
                "protocol-bound source anchor is not a directory"
            )
        for component in candidate.parts[1:-1]:
            child = os.open(component, directory_flags, dir_fd=descriptor)
            child_stat = os.fstat(child)
            if not stat.S_ISDIR(child_stat.st_mode):
                os.close(child)
                raise IntegratedProtocolExecutionError(
                    "protocol-bound source ancestor is not a directory"
                )
            os.close(descriptor)
            descriptor = child
        return descriptor, candidate.parts[-1]
    except OSError as error:
        if descriptor is not None:
            os.close(descriptor)
        raise IntegratedProtocolExecutionError(
            "cannot open protocol-bound source path without following links"
        ) from error
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        raise


def _restat_protocol_source_path(candidate: Path) -> os.stat_result:
    parent, leaf = _open_protocol_source_parent(candidate)
    try:
        linked = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
    except OSError as error:
        raise IntegratedProtocolExecutionError(
            "cannot recheck protocol-bound source path"
        ) from error
    finally:
        os.close(parent)
    if not stat.S_ISREG(linked.st_mode) or linked.st_nlink != 1:
        raise IntegratedProtocolExecutionError(
            "protocol-bound source path is not a singly linked regular file"
        )
    return linked


def _read_protocol_bound_source_bytes(
    path: Path,
) -> tuple[Path, bytes]:
    if not isinstance(path, Path):
        raise TypeError("protocol-bound source path must be a Path")
    candidate = Path(os.path.abspath(path.expanduser()))
    current = Path(candidate.anchor)
    for component in candidate.parts[1:]:
        current /= component
        if current.is_symlink():
            raise IntegratedProtocolExecutionError(
                "protocol-bound source path contains a symbolic link"
            )

    parent: int | None = None
    descriptor: int | None = None
    before: os.stat_result | None = None
    after: os.stat_result | None = None
    linked: os.stat_result | None = None
    content = b""
    try:
        parent, leaf = _open_protocol_source_parent(candidate)
        descriptor = os.open(
            leaf,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
            dir_fd=parent,
        )
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > _MAX_PROTOCOL_BOUND_SOURCE_BYTES
        ):
            raise IntegratedProtocolExecutionError(
                "protocol-bound source is not a bounded singly linked regular file"
            )
        chunks: list[bytes] = []
        total = 0
        while total <= _MAX_PROTOCOL_BOUND_SOURCE_BYTES:
            chunk = os.read(
                descriptor,
                min(1024 * 1024, _MAX_PROTOCOL_BOUND_SOURCE_BYTES + 1 - total),
            )
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        if total > _MAX_PROTOCOL_BOUND_SOURCE_BYTES:
            raise IntegratedProtocolExecutionError(
                "protocol-bound source exceeds its byte limit"
            )
        content = b"".join(chunks)
        after = os.fstat(descriptor)
        linked = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
    except OSError as error:
        raise IntegratedProtocolExecutionError(
            "cannot open or read protocol-bound source artifact"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if parent is not None:
            os.close(parent)
    if before is None or after is None or linked is None:
        raise IntegratedProtocolExecutionError(
            "protocol-bound source read did not establish file identity"
        )
    before_fingerprint = _protocol_source_stat_fingerprint(before)
    if (
        before_fingerprint != _protocol_source_stat_fingerprint(after)
        or before_fingerprint != _protocol_source_stat_fingerprint(linked)
        or len(content) != before.st_size
    ):
        raise IntegratedProtocolExecutionError(
            "protocol-bound source changed while it was read"
        )
    fresh = _restat_protocol_source_path(candidate)
    if before_fingerprint != _protocol_source_stat_fingerprint(fresh):
        raise IntegratedProtocolExecutionError(
            "protocol-bound source path identity changed while it was read"
        )
    return candidate, content


def _discard_protocol_bound_integrated_artifact(
    artifact: ProtocolBoundIntegratedSyntheticRunArtifact,
) -> None:
    identity = id(artifact)
    with _VERIFIER_OWNED_INTEGRATED_ARTIFACTS_LOCK:
        if _VERIFIER_OWNED_INTEGRATED_ARTIFACTS.get(identity) is artifact:
            _VERIFIER_OWNED_INTEGRATED_ARTIFACTS.pop(identity, None)
            _VERIFIER_OWNED_INTEGRATED_ARTIFACT_SEALS.pop(identity, None)
            _VERIFIER_OWNED_INTEGRATED_ARTIFACT_ORIGINS.pop(identity, None)


def verify_protocol_bound_integrated_synthetic_run_artifact_bytes(
    content: bytes,
    *,
    expected_sha256: str,
    expected_protocol: StudyProtocol,
    expected_config: IntegratedSyntheticMarketConfig,
    expected_pair: PolicyWorldPair,
) -> ProtocolBoundIntegratedSyntheticRunArtifact:
    """Verify committed historical bytes by a complete deterministic replay.

    The returned artifact is marked only in the process-local registry as
    ``loaded_and_replayed``.  The historical receipt is audit data, not consumed
    authority.  This pure-bytes boundary never consumes an execution
    authorization and cannot upgrade any external, historical-execution,
    authorization, welfare, identity, publication, or causal claim.
    """
    if type(content) is not bytes:
        raise TypeError("content must be exact bytes")
    if not content or len(content) > _MAX_PROTOCOL_BOUND_SOURCE_BYTES:
        raise IntegratedProtocolExecutionError(
            "protocol-bound source bytes must be non-empty and within the byte limit"
        )
    if (
        not isinstance(expected_sha256, str)
        or _PROTOCOL_BOUND_SOURCE_SHA256.fullmatch(expected_sha256) is None
    ):
        raise IntegratedProtocolExecutionError(
            "expected protocol-bound source SHA-256 is invalid"
        )
    if type(expected_protocol) is not StudyProtocol:
        raise TypeError("expected_protocol must be a StudyProtocol")
    if type(expected_config) is not IntegratedSyntheticMarketConfig:
        raise TypeError("expected_config must be an IntegratedSyntheticMarketConfig")
    if type(expected_pair) is not PolicyWorldPair:
        raise TypeError("expected_pair must be a PolicyWorldPair")
    if sha256_bytes(content) != expected_sha256:
        raise IntegratedProtocolExecutionError(
            "protocol-bound source differs from its external commitment"
        )
    try:
        supplied = read_json_bytes_object(
            content,
            source="protocol-bound integrated synthetic artifact bytes",
        )
    except MetadataValidationError as error:
        raise IntegratedProtocolExecutionError(str(error)) from error
    if canonical_json_bytes(supplied) != content:
        raise IntegratedProtocolExecutionError(
            "protocol-bound source must use canonical duplicate-free JSON"
        )
    receipt_value = supplied.get("protocol_execution_receipt")
    if not isinstance(receipt_value, Mapping):
        raise IntegratedProtocolExecutionError(
            "protocol-bound source lacks a structured execution receipt"
        )
    try:
        historical_receipt = ProtocolExecutionReceipt.from_mapping(
            cast(Mapping[str, object], receipt_value)
        )
    except Exception as error:
        raise IntegratedProtocolExecutionError(
            "protocol-bound source execution receipt is invalid"
        ) from error

    mint_ticket = _issue_loaded_replay_artifact_mint_ticket(
        protocol=expected_protocol,
        config_sha256=expected_config.sha256,
        input_sha256=integrated_synthetic_pair_input_sha256(expected_pair),
        receipt=historical_receipt,
        expected_artifact_sha256=expected_sha256,
    )

    _validate_protocol_semantics(
        protocol=expected_protocol,
        config=expected_config,
        pair=expected_pair,
    )
    replay_factory = IntegratedSyntheticWorldFactory(expected_config)
    replayed_outcome = expected_pair.run(replay_factory)
    _validate_registered_outputs(expected_protocol, replayed_outcome)
    replayed_baseline = replay_factory.runners[WorldRole.BASELINE].artifact
    replayed_candidate = replay_factory.runners[WorldRole.CANDIDATE].artifact
    if replayed_baseline is None or replayed_candidate is None:
        raise IntegratedProtocolExecutionError(
            "historical source verification replay did not complete"
        )
    artifact = _mint_protocol_bound_integrated_artifact(
        mint_ticket=mint_ticket,
        protocol=expected_protocol,
        config=expected_config,
        pair_input=expected_pair,
        protocol_execution_receipt=historical_receipt,
        protocol_canonical_json=canonical_json_bytes(
            expected_protocol.to_mapping()
        ).decode("utf-8"),
        config_sha256=expected_config.sha256,
        config_canonical_json=canonical_json_bytes(expected_config.to_mapping()).decode(
            "utf-8"
        ),
        input_sha256=integrated_synthetic_pair_input_sha256(expected_pair),
        pair_input_canonical_json=canonical_json_bytes(
            _pair_input_mapping(expected_pair)
        ).decode("utf-8"),
        paired_outcome=replayed_outcome,
        baseline_run=replayed_baseline,
        candidate_run=replayed_candidate,
    )
    if (
        artifact.paired_outcome.formal_execution_evidence is not False
        or artifact.synthetic_only is not True
        or any(
            run.integrated_mechanism_verified is not False
            or run.ledger_derived_group_outcomes is not False
            or run.formal_welfare_evidence is not False
            or run.synthetic_only is not True
            for run in (artifact.baseline_run, artifact.candidate_run)
        )
    ):
        _discard_protocol_bound_integrated_artifact(artifact)
        raise IntegratedProtocolExecutionError(
            "loaded source attempted to broaden its evidence claims"
        )
    if artifact.canonical_bytes != content or artifact.to_mapping() != supplied:
        _discard_protocol_bound_integrated_artifact(artifact)
        raise IntegratedProtocolExecutionError(
            "protocol-bound source differs from deterministic typed-input replay"
        )
    if artifact.sha256 != expected_sha256:
        _discard_protocol_bound_integrated_artifact(artifact)
        raise IntegratedProtocolExecutionError(
            "replayed protocol-bound source differs from its external commitment"
        )
    return artifact


def load_protocol_bound_integrated_synthetic_run_artifact(
    path: Path,
    *,
    expected_sha256: str,
    expected_protocol: StudyProtocol,
    expected_config: IntegratedSyntheticMarketConfig,
    expected_pair: PolicyWorldPair,
) -> ProtocolBoundIntegratedSyntheticRunArtifact:
    """Stably capture one path, then delegate all validation to the bytes API."""
    _, content = _read_protocol_bound_source_bytes(path)
    return verify_protocol_bound_integrated_synthetic_run_artifact_bytes(
        content,
        expected_sha256=expected_sha256,
        expected_protocol=expected_protocol,
        expected_config=expected_config,
        expected_pair=expected_pair,
    )


def _legacy(step: int, purpose: str) -> EventKey:
    return EventKey(
        experiment_id="integrated-synthetic-fixture",
        seed="integrated-opening-v1",
        replicate_id=0,
        simulation_step=step,
        purpose=purpose,
    )


def _parameters(scale: float) -> BehaviorParameters:
    return BehaviorParameters(
        risk_aversion=0.2 * scale,
        loss_aversion=0.3 * scale,
        disposition_strength=0.25 * scale,
        overconfidence=0.35 * scale,
        herding=0.4 * scale,
        attention_sensitivity=0.3 * scale,
        recency_weight=0.8,
        memory_decay=0.2 * scale,
        inventory_aversion=0.4 * scale,
    )


def integrated_synthetic_market_config() -> IntegratedSyntheticMarketConfig:
    """Build a small fully synthetic system with three latent strategies."""
    policy = OrderPolicyConfig(
        tick_size=10,
        base_order_quantity=1,
        temperature=0.35,
    )
    limits = RiskLimits(
        max_order_quantity=4,
        max_order_notional=2_000,
        max_inventory=100,
        max_open_orders=20,
        fee_bps=0,
        allow_market_orders=False,
    )
    mechanisms = BehaviorMechanisms()
    agents = (
        StructuredAgent(
            agent_id="agent-fundamental",
            archetype=AgentArchetype.FUNDAMENTAL,
            parameters=_parameters(0.8),
            mechanisms=mechanisms,
            trusted_cutoff=_CUTOFF,
            policy_config=policy,
            risk_limits=limits,
            cognitive_update_interval_steps=1,
        ),
        StructuredAgent(
            agent_id="agent-maker",
            archetype=AgentArchetype.MARKET_MAKER,
            parameters=_parameters(1.0),
            mechanisms=mechanisms,
            trusted_cutoff=_CUTOFF,
            policy_config=policy,
            risk_limits=limits,
            cognitive_update_interval_steps=1,
        ),
        StructuredAgent(
            agent_id="agent-trend",
            archetype=AgentArchetype.TREND,
            parameters=_parameters(1.2),
            mechanisms=mechanisms,
            trusted_cutoff=_CUTOFF,
            policy_config=policy,
            risk_limits=limits,
            cognitive_update_interval_steps=2,
        ),
    )
    graph = SyntheticSocialGraph(
        nodes=tuple(item.agent_id for item in agents),
        edges=(
            WeightedDirectedEdge("agent-fundamental", "agent-maker", 0.9),
            WeightedDirectedEdge("agent-maker", "agent-trend", 0.8),
            WeightedDirectedEdge("agent-trend", "agent-fundamental", 0.7),
        ),
    )
    opening = OpeningState(
        accounts=(
            AccountBalance("agent-fundamental", cash=100_000, asset=10),
            AccountBalance("agent-maker", cash=100_000, asset=20),
            AccountBalance("agent-trend", cash=100_000, asset=10),
            AccountBalance("ask-maker", cash=0, asset=100),
            AccountBalance("bid-maker", cash=100_000, asset=0),
            AccountBalance("exogenous-buyer", cash=100_000, asset=0),
            AccountBalance("exogenous-seller", cash=0, asset=100),
        ),
        orders=(
            OpeningOrder(
                event_key=_legacy(0, "opening-bid-top"),
                order_id="opening-bid-top",
                trader_id="bid-maker",
                side=Side.BUY,
                price=90,
                quantity=1,
                priority_time=0,
            ),
            OpeningOrder(
                event_key=_legacy(1, "opening-bid-backup"),
                order_id="opening-bid-backup",
                trader_id="bid-maker",
                side=Side.BUY,
                price=80,
                quantity=20,
                priority_time=0,
            ),
            OpeningOrder(
                event_key=_legacy(2, "opening-ask-top"),
                order_id="opening-ask-top",
                trader_id="ask-maker",
                side=Side.SELL,
                price=110,
                quantity=1,
                priority_time=0,
            ),
            OpeningOrder(
                event_key=_legacy(3, "opening-ask-backup"),
                order_id="opening-ask-backup",
                trader_id="ask-maker",
                side=Side.SELL,
                price=120,
                quantity=20,
                priority_time=0,
            ),
        ),
    )
    all_agents = tuple(item.agent_id for item in agents)
    return IntegratedSyntheticMarketConfig(
        cutoff=_CUTOFF,
        cutoff_policy_sha256=sha256_bytes(b"integrated-synthetic-cutoff-policy-v1"),
        source_lineage_sha256=sha256_bytes(b"integrated-synthetic-source-lineage-v1"),
        shared_seed="integrated-synthetic-shared-seed",
        market_config=MarketConfig(tick_size=10),
        opening_state=opening,
        agents=agents,
        social_graph=graph,
        group_members=(
            ("all-latent-strategies", all_agents),
            ("fundamental-latent", ("agent-fundamental",)),
            ("liquidity-latent", ("agent-maker",)),
        ),
        exogenous_order_quantity=2,
    )


def integrated_deterministic_mock_planning_hook(
    *, action: PlanAction = PlanAction.HOLD
) -> DeterministicMockPlanningHook:
    """Build actual offline synthetic-test authorization for the mock planner.

    The leak report is derived from executing the closed fixed-response adapter
    on every committed synthetic probe input.  It is structural mock evidence,
    not a claim that any external model passed a knowledge-leak evaluation.
    """
    if not isinstance(action, PlanAction):
        raise TypeError("action must be a PlanAction")
    adapter = DeterministicMockLLMAdapter(
        adapter_id="integrated-deterministic-mock-planner-v1",
        action=action.value,
        confidence=0.9,
        review_step=0,
        valid_until=_CUTOFF,
        rationale_code="DETERMINISTIC_SYNTHETIC_MOCK",
    )
    provenance = ModelWeightProvenance(
        adapter_id=adapter.adapter_id,
        artifact_kind=AdapterArtifactKind.DETERMINISTIC_MOCK,
        artifact_sha256=deterministic_mock_adapter_artifact_sha256(adapter),
        training_manifest_sha256=sha256_bytes(
            b"integrated-deterministic-mock-has-no-training-data"
        ),
        verification_manifest_sha256=sha256_bytes(
            canonical_json_bytes(
                {
                    "adapter_artifact": deterministic_mock_adapter_artifact_mapping(
                        adapter
                    ),
                    "verification": "execute_each_closed_synthetic_probe",
                }
            )
        ),
        provenance_manifest_sha256=sha256_bytes(
            canonical_json_bytes(
                {
                    "adapter_id": adapter.adapter_id,
                    "artifact_kind": "deterministic_mock",
                    "network": False,
                    "retrieval": False,
                    "synthetic_only": True,
                }
            )
        ),
        environment_lock_sha256=sha256_file(
            Path(__file__).resolve().parents[3] / "uv.lock"
        ),
        training_data_cutoff=_FIRST_EVENT - timedelta(days=1),
        weights_frozen_time=_FIRST_EVENT - timedelta(hours=12),
        weights_available_time=_FIRST_EVENT - timedelta(hours=6),
        network_access_permitted=False,
        retrieval_permitted=False,
        anonymous_structured_inputs_only=True,
        synthetic_only=True,
    )
    definitions = tuple(
        LeakProbeDefinition(
            probe_id=f"integrated-mock-{kind.value}",
            kind=kind,
            input_sha256=sha256_bytes(
                canonical_json_bytes(
                    {"kind": kind.value, "probe": "synthetic-closed-input-v1"}
                )
            ),
            grading_rule_sha256=sha256_bytes(
                canonical_json_bytes(
                    {
                        "allowed_response_fields": (
                            "action",
                            "confidence",
                            "input_sha256",
                            "rationale_code",
                            "review_step",
                            "schema_version",
                            "valid_until",
                        ),
                        "kind": kind.value,
                        "rule": "fixed_mock_response_contains_no_probe_payload",
                    }
                )
            ),
        )
        for kind in LeakProbeKind
    )
    manifest = LeakProbeManifest(
        provenance_sha256=provenance.sha256,
        trusted_cutoff=_CUTOFF,
        max_named_anonymous_sensitivity=0.0,
        definitions=definitions,
    )
    results = tuple(
        LeakProbeResult(
            probe_id=definition.probe_id,
            kind=definition.kind,
            input_sha256=definition.input_sha256,
            output_sha256=sha256_bytes(
                canonical_json_bytes(
                    adapter.complete({"input_sha256": definition.input_sha256})
                )
            ),
            forbidden_revelation_detected=False,
            sensitivity_score=(
                0.0
                if definition.kind is LeakProbeKind.NAMED_VS_ANONYMOUS_SENSITIVITY
                else None
            ),
        )
        for definition in definitions
    )
    report = LeakProbeReport(
        manifest_sha256=manifest.sha256,
        provenance_sha256=provenance.sha256,
        offline_enforced=True,
        retrieval_disabled=True,
        results=results,
    )
    deidentification_policy = DeidentificationPolicy(
        entity_names=("Synthetic Named Entity",),
        policy_names=("Synthetic Named Policy",),
        commitment_key=b"policytwin-integrated-mock-deidentification-key-v1",
    )
    (
        sanitized_context,
        sanitized_context_record,
        context_authority,
    ) = issue_verifier_owned_integrated_mock_context(deidentification_policy)
    authorization = AdapterAuthorization(
        provenance=provenance,
        requested_use=RequestedUse.SYNTHETIC_TEST,
        trusted_cutoff=_CUTOFF,
        cutoff_policy_sha256=sha256_bytes(b"integrated-synthetic-cutoff-policy-v1"),
        deidentification_policy_sha256=deidentification_policy.sha256,
        prompt_template_sha256=sha256_bytes(
            b"integrated-fixed-anonymous-public-features-template-v1"
        ),
        sanitized_context_record_sha256=sanitized_context_record.sha256,
        leak_probe_manifest=manifest,
        leak_probe_report=report,
    )
    return DeterministicMockPlanningHook(
        adapter=adapter,
        authorization=authorization,
        deidentification_policy=deidentification_policy,
        sanitized_context=sanitized_context,
        sanitized_context_record=sanitized_context_record,
        _context_authority=context_authority,
    )


def integrated_synthetic_mock_market_config(
    *, action: PlanAction = PlanAction.HOLD
) -> IntegratedSyntheticMarketConfig:
    """Enable the validated deterministic mock in the pre-cutoff fixture."""
    return replace(
        integrated_synthetic_market_config(),
        mock_planning_hook=integrated_deterministic_mock_planning_hook(action=action),
    )


def _generate_integrated_shock_tape(
    *,
    event_key: EventKey,
    config: IntegratedSyntheticMarketConfig,
    step_count: int,
) -> ExogenousShockTape:
    """Generate a tape; protocol builders also retain its verifier receipt."""
    tape, _ = _generate_integrated_shock_evidence(
        event_key=event_key,
        config=config,
        step_count=step_count,
    )
    return tape


def _generate_integrated_shock_evidence(
    *,
    event_key: EventKey,
    config: IntegratedSyntheticMarketConfig,
    step_count: int,
) -> tuple[ExogenousShockTape, ScenarioGenerationReceipt | None]:
    if (
        isinstance(step_count, bool)
        or not isinstance(step_count, int)
        or step_count <= 0
        or step_count > MAX_SCENARIO_GENERATION_DRAWS
    ):
        raise ValueError(
            "step_count must be within the frozen scenario-generation resource bound"
        )
    scenario_id = f"integrated-{event_key.value.removeprefix('ptk1:')[:20]}"
    rng = HashBasedRNG(event_key.value)
    shocks: list[ExogenousShock] = []
    draws: list[ScenarioGenerationDraw] = []
    first_event = (
        config.cutoff if config.scenario_generator_spec is not None else _FIRST_EVENT
    )
    for index in range(1, step_count + 1):
        semantic = SemanticEventKey(
            scenario_id=scenario_id,
            event_type=EventType.EXOGENOUS_SHOCK,
            entity_id="synthetic-fundamental",
            timestamp=first_event + timedelta(hours=index),
            agent_id=None,
        )
        unit = rng.uniform(
            semantic,
            domain="integrated-shock-magnitude",
        )
        bits = rng.uniform_bits(
            semantic,
            domain="integrated-shock-magnitude",
        )
        magnitude = 6.0 + 4.0 * unit
        sign = 1.0 if index % 2 else -1.0
        shocks.append(
            ExogenousShock(
                event_key=semantic,
                values=((config.transition_shock_field, sign * magnitude),),
            )
        )
        draws.append(
            ScenarioGenerationDraw(
                event_key=semantic,
                domain="integrated-shock-magnitude",
                draw_index=0,
                uniform_bits=bits,
                output_binary64_hex=unit.hex(),
            )
        )
    tape = ExogenousShockTape(tuple(shocks))
    if config.scenario_generator_spec is None:
        return tape, None
    receipt = _issue_scenario_generation_receipt(
        root_event_key=event_key,
        generator_spec=config.scenario_generator_spec,
        draws=tuple(draws),
        scenario_id=scenario_id,
        shock_tape_sha256=tape.sha256,
    )
    return tape, receipt


_SCENARIO_GENERATOR_EVENT_DEPENDENCY_ALLOWLIST = (
    "UTC",
    "EVENT_KEY_SCHEMA_VERSION",
    "EXOGENOUS_SHOCK_TAPE_SCHEMA_VERSION",
    "HASH_BASED_RNG_IMPLEMENTATION_ID",
    "MAX_SCENARIO_GENERATION_DRAWS",
    "SCENARIO_GENERATION_DRAW_SCHEMA_VERSION",
    "SCENARIO_GENERATION_RECEIPT_SCHEMA_VERSION",
    "SEMANTIC_EVENT_KEY_SCHEMA_VERSION",
    "EventKey",
    "EventType",
    "ExogenousShock",
    "ExogenousShockTape",
    "HashBasedRNG",
    "ScenarioGenerationDraw",
    "ScenarioGenerationReceipt",
    "ScenarioGeneratorSpec",
    "SemanticEventKey",
    "datetime",
    "_IDENTIFIER_PATTERN",
    "_LEGACY_KEY_PATTERN",
    "_MAX_DRAW_INDEX",
    "_SEMANTIC_KEY_PATTERN",
    "_UNIT_SCALE",
    "_aware_utc",
    "_bounded_draw_index",
    "_bounded_text",
    "_canonical_json_bytes",
    "_draw_digest",
    "_event_type",
    "_finite",
    "_identifier",
    "_issue_scenario_generation_receipt",
    "_legacy_event_key_from_mapping",
    "_legacy_event_key_mapping",
    "_non_negative_int",
    "_normalize_shock_values",
    "_optional_identifier",
    "_parse_datetime",
    "_require_mapping",
    "_seed",
    "_semantic_key_value",
    "_sha256_bytes",
    "_strict_fields",
    "_uniform_from_bits",
    "hash_based_rng_live_dependency_sha256",
    "validate_exact_exogenous_shock_tape",
    "validate_exact_scenario_generation_receipt",
)

_SCENARIO_GENERATOR_LOCAL_DEPENDENCY_ALLOWLIST = (
    "EventKey",
    "EventType",
    "ExogenousShock",
    "ExogenousShockTape",
    "HashBasedRNG",
    "IntegratedSyntheticMarketConfig",
    "MAX_SCENARIO_GENERATION_DRAWS",
    "ScenarioGenerationDraw",
    "ScenarioGenerationReceipt",
    "ScenarioGeneratorSpec",
    "SemanticEventKey",
    "_FIRST_EVENT",
    "_generate_integrated_shock_evidence",
    "_generate_integrated_shock_tape",
    "_issue_scenario_generation_receipt",
    "canonical_json_bytes",
    "sha256_bytes",
    "timedelta",
)


def _scenario_generator_event_dependency_sha256() -> str:
    """Fingerprint the explicit live event/shock dependency allowlist."""
    namespace = {
        name: getattr(counterfactual_events_module, name)
        for name in _SCENARIO_GENERATOR_EVENT_DEPENDENCY_ALLOWLIST
    }
    module_path = Path(counterfactual_events_module.__file__)
    return live_policytwin_implementation_sha256(
        module_path=module_path,
        namespace=namespace,
    )


def _scenario_generator_local_dependency_sha256() -> str:
    """Fingerprint live generator code and its run-specific config carrier."""
    return live_policytwin_implementation_sha256(
        module_path=Path(__file__),
        namespace={
            name: globals()[name]
            for name in _SCENARIO_GENERATOR_LOCAL_DEPENDENCY_ALLOWLIST
        },
    )


def integrated_synthetic_shock_generator_implementation_sha256() -> str:
    """Commit the forward generator to an explicit live dependency closure."""
    return sha256_bytes(
        canonical_json_bytes(
            {
                "event_dependency_allowlist": list(
                    _SCENARIO_GENERATOR_EVENT_DEPENDENCY_ALLOWLIST
                ),
                "event_dependency_sha256": (
                    _scenario_generator_event_dependency_sha256()
                ),
                "local_dependency_sha256": (
                    _scenario_generator_local_dependency_sha256()
                ),
                "local_dependency_allowlist": list(
                    _SCENARIO_GENERATOR_LOCAL_DEPENDENCY_ALLOWLIST
                ),
                "rng_dependency_sha256": (hash_based_rng_live_dependency_sha256()),
                "rng_implementation_id": HASH_BASED_RNG_IMPLEMENTATION_ID,
                "schema_versions": {
                    "draw": SCENARIO_GENERATION_DRAW_SCHEMA_VERSION,
                    "receipt": SCENARIO_GENERATION_RECEIPT_SCHEMA_VERSION,
                },
            }
        )
    )


def integrated_synthetic_shock_generator_config_sha256() -> str:
    """Commit the generator's fixed semantics; run-specific inputs bind separately."""
    return sha256_bytes(
        canonical_json_bytes(
            {
                "domain": "integrated-shock-magnitude",
                "draw_index": 0,
                "entity_id": "synthetic-fundamental",
                "event_spacing": "one_hour",
                "first_event": "trusted_cutoff",
                "magnitude": "6.0 + 4.0 * keyed_uniform",
                "rng_implementation_id": HASH_BASED_RNG_IMPLEMENTATION_ID,
                "scenario_id": "event_key_commitment_prefix_20",
                "sign": "positive_on_odd_sequence_negative_on_even_sequence",
                "step_count_source": "protocol_bound_pair_input",
                "transition_field_source": "integrated_market_config",
            }
        )
    )


def _validate_integrated_scenario_generation(
    *,
    config: IntegratedSyntheticMarketConfig,
    pair: PolicyWorldPair,
) -> None:
    """Regenerate a frozen forward tape from its persisted root receipt."""
    spec = config.scenario_generator_spec
    receipt = pair.scenario_generation_receipt
    if spec is None:
        raise IntegratedProtocolExecutionError(
            "scenario generation validation requires a frozen generator spec"
        )
    if receipt is None:
        raise IntegratedProtocolExecutionError(
            "frozen scenario-generator protocol execution requires a generation receipt"
        )
    if receipt.generator_spec != spec:
        raise IntegratedProtocolExecutionError(
            "scenario generation receipt is bound to another generator spec"
        )
    try:
        receipt.validate_live()
        regenerated_tape, regenerated_receipt = _generate_integrated_shock_evidence(
            event_key=receipt.root_event_key,
            config=config,
            step_count=len(pair.shock_tape),
        )
    except (SemanticEventError, TypeError, ValueError) as error:
        raise IntegratedProtocolExecutionError(
            "scenario generation receipt failed live deterministic replay"
        ) from error
    if regenerated_receipt is None:
        raise IntegratedProtocolExecutionError(
            "live frozen scenario regeneration omitted its receipt"
        )
    if regenerated_tape.canonical_bytes != pair.shock_tape.canonical_bytes:
        raise IntegratedProtocolExecutionError(
            "shock tape differs from live regeneration under its receipt root"
        )
    if regenerated_receipt.canonical_bytes != receipt.canonical_bytes:
        raise IntegratedProtocolExecutionError(
            "scenario generation receipt differs from live regeneration"
        )


def _validate_formal_generation_evidence(
    *,
    config: IntegratedSyntheticMarketConfig,
    pair: PolicyWorldPair,
) -> None:
    """Validate complete generator-specific evidence before authority is used."""
    spec = config.scenario_generator_spec
    if spec is None:
        raise IntegratedProtocolExecutionError(
            "protocol-bound policy simulation requires a frozen forward generator"
        )
    if (
        spec.generator_implementation_sha256
        != integrated_synthetic_shock_generator_implementation_sha256()
        or spec.generator_config_sha256
        != integrated_synthetic_shock_generator_config_sha256()
    ):
        raise IntegratedProtocolExecutionError(
            "scenario generator commitment differs from the live verifier-owned "
            "implementation or configuration"
        )
    _validate_integrated_scenario_generation(config=config, pair=pair)


def integrated_synthetic_forward_market_config() -> IntegratedSyntheticMarketConfig:
    """Build the same mechanism fixture with a frozen post-cutoff generator."""
    base = integrated_synthetic_market_config()
    return replace(
        base,
        scenario_generator_spec=ScenarioGeneratorSpec(
            generator_id="integrated-synthetic-forward-generator-v2",
            generator_config_sha256=(
                integrated_synthetic_shock_generator_config_sha256()
            ),
            generator_implementation_sha256=(
                integrated_synthetic_shock_generator_implementation_sha256()
            ),
            source_lineage_sha256=base.source_lineage_sha256,
            cutoff_policy_sha256=base.cutoff_policy_sha256,
            frozen_at=base.cutoff - timedelta(days=1),
            trusted_cutoff=base.cutoff,
        ),
    )


def integrated_synthetic_pair(
    *,
    event_key: EventKey,
    parameters: dict[str, float],
    config: IntegratedSyntheticMarketConfig | None = None,
    candidate_allow_buy: bool = True,
    candidate_allow_sell: bool = True,
    candidate_is_noop: bool = False,
    step_count: int = 4,
    include_all_groups: bool = False,
) -> PolicyWorldPair:
    """Build a keyed paired scenario over the integrated synthetic system."""
    if not isinstance(event_key, EventKey):
        raise TypeError("event_key must be an EventKey")
    if (
        isinstance(step_count, bool)
        or not isinstance(step_count, int)
        or step_count < 2
    ):
        raise ValueError("step_count must be an integer of at least two")
    if not all(
        isinstance(value, bool)
        for value in (
            candidate_allow_buy,
            candidate_allow_sell,
            candidate_is_noop,
            include_all_groups,
        )
    ):
        raise TypeError("candidate and group switches must be boolean")
    if set(parameters) != {"behavior_scale", "signal_scale"}:
        raise ValueError("integrated parameters have missing or unknown fields")
    for name, value in parameters.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
        ):
            raise ValueError(f"integrated parameter {name} must be finite")
    if config is None:
        config = integrated_synthetic_market_config()
    elif not isinstance(config, IntegratedSyntheticMarketConfig):
        raise TypeError("config must be an IntegratedSyntheticMarketConfig")
    tape, generation_receipt = _generate_integrated_shock_evidence(
        event_key=event_key,
        config=config,
        step_count=step_count,
    )
    scenario_id = tape.scenario_id
    groups = (
        tuple(
            GroupSpec.from_members(group_id, members)
            for group_id, members in config.group_members
        )
        if include_all_groups
        else (
            GroupSpec.from_members(
                "all-latent-strategies",
                config.members_for("all-latent-strategies"),
            ),
        )
    )
    baseline_config = (("allow_buy", True), ("allow_sell", True))
    candidate_config = (
        ("allow_buy", candidate_allow_buy),
        ("allow_sell", candidate_allow_sell),
    )
    if candidate_is_noop and candidate_config != baseline_config:
        raise ValueError("a no-op candidate must equal the baseline policy")
    parameter_items = tuple(
        sorted((name, float(value)) for name, value in parameters.items())
    )
    return PolicyWorldPair(
        scenario_id=scenario_id,
        posterior_draw=PosteriorDraw(
            draw_id=f"draw-{event_key.value[-16:]}",
            model_draw_sha256=sha256_bytes(
                canonical_json_bytes(
                    {
                        "config_sha256": config.sha256,
                        "model": INTEGRATED_MODEL_ID,
                    }
                )
            ),
            parameters=parameter_items,
        ),
        initial_state=StateArtifact.from_mapping(
            {config.transition_state_field: 100.0}
        ),
        shock_tape=tape,
        shock_transition_config=ShockTransitionConfig(
            state_field=config.transition_state_field,
            shock_field=config.transition_shock_field,
        ),
        baseline_policy=PolicyDefinition(
            policy_id="integrated-baseline",
            config=baseline_config,
        ),
        candidate_policy=PolicyDefinition(
            policy_id=(
                "integrated-noop" if candidate_is_noop else "integrated-buy-restriction"
            ),
            config=candidate_config,
            is_noop=candidate_is_noop,
        ),
        horizons=(
            HorizonSpec("early", 2),
            HorizonSpec("terminal", step_count),
        ),
        groups=groups,
        scenario_generation_receipt=generation_receipt,
    )


__all__ = [
    "INTEGRATED_ESTIMAND_ID",
    "INTEGRATED_METRIC_IDS",
    "INTEGRATED_PAIR_INPUT_SCHEMA_VERSION",
    "INTEGRATED_PROTOCOL_RUN_SCHEMA_VERSION",
    "IntegratedProtocolExecutionError",
    "ProtocolBoundArtifactOrigin",
    "ProtocolBoundIntegratedSyntheticRunArtifact",
    "integrated_metric_definition_sha256",
    "integrated_paired_estimand_definition_sha256",
    "integrated_deterministic_mock_planning_hook",
    "integrated_synthetic_forward_market_config",
    "integrated_synthetic_market_config",
    "integrated_synthetic_mock_market_config",
    "integrated_synthetic_pair",
    "integrated_synthetic_pair_input_sha256",
    "integrated_synthetic_shock_generator_config_sha256",
    "integrated_synthetic_shock_generator_implementation_sha256",
    "load_protocol_bound_integrated_synthetic_run_artifact",
    "require_ledger_derivation_source_artifact",
    "require_verifier_owned_protocol_bound_integrated_artifact",
    "run_protocol_bound_integrated_synthetic_pair",
    "verify_protocol_bound_integrated_synthetic_run_artifact_bytes",
]
