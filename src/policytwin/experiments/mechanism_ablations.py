"""Mechanism-executed synthetic ablations for the integrated market loop.

This module is intentionally narrower than the experiment matrix.  It executes
only mechanisms that the bundled synthetic integrated runner can actually turn
on or off.  Matrix labels for LLM use, posterior-vs-point calibration, or a
structural ensemble are not accepted here because those mechanisms are not
executed by this runner.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Literal, NoReturn

import policytwin.agents.core as agent_core_module
import policytwin.agents.policies as agent_policies_module
import policytwin.behavior.parameters as behavior_parameters_module
import policytwin.cognition.memory as cognition_memory_module
import policytwin.cognition.state as cognition_state_module
import policytwin.counterfactual.integrated_market as integrated_market_module
import policytwin.experiments.integrated_synthetic as integrated_synthetic_module
import policytwin.market.session as market_session_module
import policytwin.protocol as protocol_module
import policytwin.social.propagation as social_propagation_module
from policytwin.agents import AgentArchetype, StructuredAgent
from policytwin.behavior import BehaviorMechanisms, BehaviorParameters
from policytwin.counterfactual.integrated_market import (
    IntegratedSyntheticMarketConfig,
    IntegratedSyntheticRunArtifact,
    IntegratedSyntheticWorldFactory,
)
from policytwin.counterfactual.paired import PairedOutcome, PolicyWorldPair, WorldRole
from policytwin.counterfactual.randomness import EventKey
from policytwin.experiments.integrated_synthetic import (
    INTEGRATED_METRIC_IDS,
    integrated_metric_definition_sha256,
    integrated_synthetic_market_config,
    integrated_synthetic_pair,
)
from policytwin.protocol import (
    ClaimScope,
    CohortBasis,
    FailureThreshold,
    FrozenCohort,
    FrozenDefinition,
    OutputRole,
    ProtocolExecutionAuthorization,
    ProtocolExecutionConsumer,
    ProtocolExecutionPurpose,
    ProtocolExecutionReceipt,
    RandomnessProtocol,
    RegisteredOutput,
    StudyMode,
    StudyProtocol,
    SyntheticProtocolExecutionIssuer,
    consume_protocol_execution_authorization,
    validate_protocol_execution_receipt_binding,
)
from policytwin.provenance import (
    canonical_json_bytes,
    read_json_object,
    sha256_bytes,
    sha256_file,
    write_json_file,
    write_text_file,
)

MECHANISM_ABLATION_SPEC_SCHEMA_VERSION = "policytwin.synthetic-mechanism-ablation.v2"
MECHANISM_ABLATION_ARTIFACT_SCHEMA_VERSION = (
    "policytwin.synthetic-mechanism-ablation-artifact.v3"
)
# Backward-compatible request-schema name. Artifacts use their own version.
MECHANISM_ABLATION_SCHEMA_VERSION = MECHANISM_ABLATION_SPEC_SCHEMA_VERSION
MECHANISM_ABLATION_CLAIM_SCOPE = "synthetic_integrated_mechanism_ablation_only"
MECHANISM_ABLATION_EXECUTOR_ID = "policytwin.integrated_mechanism_ablations.v1"
MECHANISM_ABLATION_SPEC_DEFINITION_ID = "mechanism_ablation_spec"
MECHANISM_ABLATION_ESTIMAND_ID = "synthetic_mechanism_variant_minus_reference"
MECHANISM_ABLATION_OUTCOME_ID = "synthetic_mechanism_ablation_contrast"
MECHANISM_ABLATION_EXECUTION_INPUT_SCHEMA_VERSION = (
    "policytwin.synthetic-mechanism-ablation-execution-input.v1"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_NULL_HASH = "0" * 64


class MechanismAblationError(RuntimeError):
    """Base class for fail-closed mechanism-ablation failures."""


class MechanismAblationSchemaError(MechanismAblationError, ValueError):
    """Raised when immutable ablation inputs or artifacts are malformed."""


class MechanismAblationIntegrityError(MechanismAblationError):
    """Raised when a mechanism receipt or saved artifact does not replay."""


def _fail(message: str) -> NoReturn:
    raise MechanismAblationSchemaError(message)


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail(f"{field} must be a lowercase SHA-256 digest")
    return value


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        _fail(f"{field} must be a finite real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        _fail(f"{field} must be finite")
    return normalized


def _commit(value: object) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _event_key_mapping(event_key: EventKey) -> dict[str, object]:
    return {
        "experiment_id": event_key.experiment_id,
        "purpose": event_key.purpose,
        "replicate_id": event_key.replicate_id,
        "schema_version": event_key.schema_version,
        "seed": event_key.seed,
        "simulation_step": event_key.simulation_step,
        "value": event_key.value,
    }


def mechanism_ablation_estimand_definition_sha256() -> str:
    """Commit the synthetic variant-minus-reference estimand semantics."""
    return _commit(
        {
            "claim_scope": MECHANISM_ABLATION_CLAIM_SCOPE,
            "formula": "metric_cell(variant)-metric_cell(heterogeneous_reference)",
            "reference_variant": "heterogeneous_continuous_synthetic",
            "sampling_unit": "event_keyed_synthetic_replicate",
            "schema_version": 1,
        }
    )


class MechanismAblationVariant(StrEnum):
    """Closed set of dimensions with an executable integrated mechanism."""

    HETEROGENEOUS_CONTINUOUS = "heterogeneous_continuous_synthetic"
    PROMPTED_LABELS_MOCK = "prompted_behavior_labels_deterministic_mock"
    HOMOGENEOUS_AGENTS = "homogeneous_agents"
    RATIONAL_ONLY = "rational_only"
    NO_SOCIAL_GRAPH = "no_social_graph"
    NO_MEMORY = "no_memory"


_VARIANTS = tuple(MechanismAblationVariant)
_VARIANT_ORDER = {variant: index for index, variant in enumerate(_VARIANTS)}


class MechanismAblationStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class MechanismAblationSpec:
    """Frozen synthetic-only request for a small comparable ablation grid."""

    root_event_key: EventKey
    behavior_scale: float = 0.8
    signal_scale: float = 1.1
    replicate_count: int = 2
    step_count: int = 4
    variants: tuple[MechanismAblationVariant, ...] = _VARIANTS
    synthetic_only: Literal[True] = True
    schema_version: str = MECHANISM_ABLATION_SPEC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.root_event_key, EventKey):
            _fail("root_event_key must be an EventKey")
        if (
            not self.root_event_key.experiment_id
            or not self.root_event_key.seed
            or not self.root_event_key.purpose
            or isinstance(self.root_event_key.replicate_id, bool)
            or not isinstance(self.root_event_key.replicate_id, int)
            or self.root_event_key.replicate_id < 0
            or isinstance(self.root_event_key.simulation_step, bool)
            or not isinstance(self.root_event_key.simulation_step, int)
            or self.root_event_key.simulation_step < 0
            or not self.root_event_key.value.startswith("ptk1:")
        ):
            _fail("root_event_key has invalid or negative identity fields")
        behavior = _finite(self.behavior_scale, "behavior_scale")
        signal = _finite(self.signal_scale, "signal_scale")
        if not 0.0 <= behavior <= 1.0:
            _fail("behavior_scale must be in [0, 1]")
        if not 0.0 < signal <= 5.0:
            _fail("signal_scale must be in (0, 5]")
        object.__setattr__(self, "behavior_scale", behavior)
        object.__setattr__(self, "signal_scale", signal)
        if (
            isinstance(self.replicate_count, bool)
            or not isinstance(self.replicate_count, int)
            or self.replicate_count < 1
        ):
            _fail("replicate_count must be a positive integer")
        if (
            isinstance(self.step_count, bool)
            or not isinstance(self.step_count, int)
            or self.step_count < 2
        ):
            _fail("step_count must be an integer of at least two")
        if not isinstance(self.variants, tuple) or not self.variants:
            _fail("variants must be a non-empty immutable tuple")
        if not all(
            isinstance(item, MechanismAblationVariant) for item in self.variants
        ):
            _fail("variants contain an unsupported mechanism")
        ordered = tuple(sorted(self.variants, key=_VARIANT_ORDER.__getitem__))
        if len(set(ordered)) != len(ordered):
            _fail("variants must be unique")
        if MechanismAblationVariant.HETEROGENEOUS_CONTINUOUS not in ordered:
            _fail("the executable heterogeneous reference is required")
        object.__setattr__(self, "variants", ordered)
        if self.synthetic_only is not True:
            _fail("mechanism ablations are synthetic-only")
        if self.schema_version != MECHANISM_ABLATION_SPEC_SCHEMA_VERSION:
            _fail("unsupported mechanism-ablation schema version")

    def event_key_for(self, replicate_index: int) -> EventKey:
        """Return a variant-independent key for one comparable replicate."""
        if (
            isinstance(replicate_index, bool)
            or not isinstance(replicate_index, int)
            or not 0 <= replicate_index < self.replicate_count
        ):
            _fail("replicate_index is outside the frozen grid")
        root = self.root_event_key
        return EventKey(
            experiment_id=root.experiment_id,
            seed=root.seed,
            replicate_id=root.replicate_id + replicate_index,
            simulation_step=root.simulation_step,
            purpose=f"{root.purpose}:mechanism-ablation",
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "behavior_scale": self.behavior_scale,
            "replicate_count": self.replicate_count,
            "root_event_key": _event_key_mapping(self.root_event_key),
            "schema_version": self.schema_version,
            "signal_scale": self.signal_scale,
            "step_count": self.step_count,
            "synthetic_only": self.synthetic_only,
            "variants": [item.value for item in self.variants],
        }

    @property
    def sha256(self) -> str:
        return _commit(self.to_mapping())


@dataclass(frozen=True, slots=True)
class MechanismSourceComponent:
    """One implementation source file committed by the execution authority."""

    module: str
    source_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.module, str) or not self.module:
            _fail("source component module must be non-empty")
        _sha256(self.source_sha256, "source_sha256")

    def to_mapping(self) -> dict[str, object]:
        return {"module": self.module, "source_sha256": self.source_sha256}


@dataclass(frozen=True, slots=True)
class MechanismImplementationReceipt:
    """Hash binding to the exact integrated mechanism implementation."""

    executor_id: str
    components: tuple[MechanismSourceComponent, ...]
    executor_code_sha256: str
    synthetic_only: Literal[True] = True

    def __post_init__(self) -> None:
        if self.executor_id != MECHANISM_ABLATION_EXECUTOR_ID:
            _fail("unknown mechanism-ablation executor")
        if not isinstance(self.components, tuple) or not self.components:
            _fail("implementation receipt requires source components")
        if not all(
            isinstance(item, MechanismSourceComponent) for item in self.components
        ):
            _fail("implementation receipt has invalid source components")
        if self.components != tuple(
            sorted(self.components, key=lambda item: item.module)
        ):
            _fail("implementation source components must be canonical")
        if len({item.module for item in self.components}) != len(self.components):
            _fail("implementation source modules must be unique")
        _sha256(self.executor_code_sha256, "executor_code_sha256")
        expected = _commit([item.to_mapping() for item in self.components])
        if self.executor_code_sha256 != expected:
            _fail("executor_code_sha256 does not match source components")
        if self.synthetic_only is not True:
            _fail("mechanism implementation receipt must remain synthetic-only")

    def to_mapping(self) -> dict[str, object]:
        return {
            "components": [item.to_mapping() for item in self.components],
            "executor_code_sha256": self.executor_code_sha256,
            "executor_id": self.executor_id,
            "synthetic_only": self.synthetic_only,
        }

    @property
    def sha256(self) -> str:
        return _commit(self.to_mapping())


def _implementation_receipt() -> MechanismImplementationReceipt:
    modules = (
        ("policytwin.agents.core", agent_core_module.__file__),
        ("policytwin.agents.policies", agent_policies_module.__file__),
        ("policytwin.behavior.parameters", behavior_parameters_module.__file__),
        ("policytwin.cognition.memory", cognition_memory_module.__file__),
        ("policytwin.cognition.state", cognition_state_module.__file__),
        (
            "policytwin.counterfactual.integrated_market",
            integrated_market_module.__file__,
        ),
        (
            "policytwin.experiments.integrated_synthetic",
            integrated_synthetic_module.__file__,
        ),
        ("policytwin.experiments.mechanism_ablations", __file__),
        ("policytwin.market.session", market_session_module.__file__),
        ("policytwin.protocol", protocol_module.__file__),
        ("policytwin.social.propagation", social_propagation_module.__file__),
    )
    components: list[MechanismSourceComponent] = []
    for module_name, raw_path in modules:
        if raw_path is None:
            raise MechanismAblationIntegrityError(
                f"cannot locate mechanism source for {module_name}"
            )
        path = Path(raw_path)
        if not path.is_file():
            raise MechanismAblationIntegrityError(
                f"mechanism source is unavailable for {module_name}"
            )
        components.append(MechanismSourceComponent(module_name, sha256_file(path)))
    ordered = tuple(sorted(components, key=lambda item: item.module))
    return MechanismImplementationReceipt(
        executor_id=MECHANISM_ABLATION_EXECUTOR_ID,
        components=ordered,
        executor_code_sha256=_commit([item.to_mapping() for item in ordered]),
    )


def _execution_input_sha256(
    spec: MechanismAblationSpec,
    implementation: MechanismImplementationReceipt,
) -> str:
    return _commit(
        {
            "implementation_receipt": implementation.to_mapping(),
            "schema_version": MECHANISM_ABLATION_EXECUTION_INPUT_SCHEMA_VERSION,
            "spec": spec.to_mapping(),
        }
    )


def mechanism_ablation_execution_input_sha256(spec: MechanismAblationSpec) -> str:
    """Commit the complete grid request and live implementation before issuance."""
    if not isinstance(spec, MechanismAblationSpec):
        _fail("spec must be a MechanismAblationSpec")
    return _execution_input_sha256(spec, _implementation_receipt())


def _protocol_reference_pair(spec: MechanismAblationSpec) -> PolicyWorldPair:
    config = integrated_synthetic_market_config()
    return integrated_synthetic_pair(
        event_key=spec.event_key_for(0),
        parameters={
            "behavior_scale": spec.behavior_scale,
            "signal_scale": spec.signal_scale,
        },
        config=config,
        candidate_is_noop=True,
        step_count=spec.step_count,
        include_all_groups=True,
    )


def mechanism_ablation_registered_output_cells(
    spec: MechanismAblationSpec,
) -> tuple[tuple[str, str, str], ...]:
    """Return the exact horizon/cohort/metric registry for every variant."""
    return tuple(
        sorted(
            (item.horizon_id, item.cohort_id, item.metric_id)
            for item in mechanism_ablation_registered_outputs(spec)
        )
    )


def mechanism_ablation_registered_outputs(
    spec: MechanismAblationSpec,
) -> tuple[RegisteredOutput, ...]:
    """Return the verifier-owned output IDs, roles, and interval semantics."""
    if not isinstance(spec, MechanismAblationSpec):
        _fail("spec must be a MechanismAblationSpec")
    pair = _protocol_reference_pair(spec)
    outputs = (
        RegisteredOutput(
            output_id=f"mechanism_{horizon.horizon_id}_{group.group_id}_{metric_id}",
            metric_id=metric_id,
            cohort_id=group.group_id,
            horizon_id=horizon.horizon_id,
            role=(
                OutputRole.PRIMARY
                if metric_id == "quoted_spread"
                and horizon.horizon_id == "terminal"
                and group.group_id == "all-latent-strategies"
                else OutputRole.SECONDARY
            ),
            interval_method="none",
            interval_levels=(),
        )
        for horizon in pair.horizons
        for group in pair.groups
        for metric_id in INTEGRATED_METRIC_IDS
    )
    return tuple(sorted(outputs, key=lambda item: item.output_id))


def _protocol_definition(
    identifier: str,
    *,
    definition_sha256: str | None = None,
    description: str | None = None,
) -> FrozenDefinition:
    return FrozenDefinition(
        definition_id=identifier,
        description=(
            description
            if description is not None
            else f"Frozen synthetic definition for {identifier}."
        ),
        definition_sha256=(
            definition_sha256
            if definition_sha256 is not None
            else sha256_bytes(identifier.encode("utf-8"))
        ),
    )


def build_mechanism_ablation_protocol(spec: MechanismAblationSpec) -> StudyProtocol:
    """Build the complete synthetic study contract for one ablation grid.

    Keeping this factory in production code prevents research runs from relying
    on a test-only protocol fixture.  The resulting protocol remains locally
    issued, synthetic-only evidence; it cannot authorize a real policy claim.
    """

    if not isinstance(spec, MechanismAblationSpec):
        _fail("spec must be a MechanismAblationSpec")
    config = integrated_synthetic_market_config()
    pair = _protocol_reference_pair(spec)
    cutoff = config.cutoff.astimezone(UTC)
    protocol = StudyProtocol(
        schema_version=2,
        study_id=spec.root_event_key.experiment_id,
        mode=StudyMode.SYNTHETIC,
        claim_scope=ClaimScope.SYNTHETIC_ONLY,
        registered_at=cutoff,
        knowledge_cutoff=cutoff,
        model_frozen_at=cutoff,
        policy_start=cutoff + timedelta(days=1),
        policy_end=cutoff + timedelta(days=2),
        evaluation_reveal_at=cutoff + timedelta(days=3),
        estimand=_protocol_definition(
            MECHANISM_ABLATION_ESTIMAND_ID,
            definition_sha256=mechanism_ablation_estimand_definition_sha256(),
        ),
        baseline_policy=_protocol_definition(
            pair.baseline_policy.policy_id,
            definition_sha256=pair.baseline_policy.sha256,
        ),
        candidate_policy=_protocol_definition(
            pair.candidate_policy.policy_id,
            definition_sha256=pair.candidate_policy.sha256,
        ),
        cohorts=tuple(
            FrozenCohort(
                cohort_id=group.group_id,
                description="Frozen synthetic latent strategy cohort.",
                definition_sha256=group.cohort_sha256,
                basis=CohortBasis.LATENT_STRATEGY,
            )
            for group in pair.groups
        ),
        horizons=tuple(
            _protocol_definition(
                horizon.horizon_id,
                definition_sha256=_commit(horizon.to_mapping()),
            )
            for horizon in pair.horizons
        ),
        metrics=tuple(
            _protocol_definition(
                metric_id,
                definition_sha256=integrated_metric_definition_sha256(metric_id),
            )
            for metric_id in INTEGRATED_METRIC_IDS
        ),
        baselines=(
            _protocol_definition("synthetic_reference_model"),
            _protocol_definition(
                MECHANISM_ABLATION_SPEC_DEFINITION_ID,
                definition_sha256=spec.sha256,
                description="Frozen synthetic execution grid for this ablation.",
            ),
        ),
        outputs=mechanism_ablation_registered_outputs(spec),
        calibration_statistics=("synthetic_calibration_statistic",),
        heldout_validation_statistics=("synthetic_validation_statistic",),
        policy_evaluation_outcomes=(MECHANISM_ABLATION_OUTCOME_ID,),
        failure_thresholds=(
            FailureThreshold(
                check_id="temporal_leakage",
                comparator="higher_is_worse",
                yellow_value=0.0,
                red_value=0.0,
                abstain_at_red=True,
            ),
        ),
        abstention_rule=_protocol_definition("synthetic_abstention"),
        randomness_protocol=RandomnessProtocol.EVENT_KEYED_PAIRED,
    )
    _validate_study_protocol(protocol, spec)
    return protocol


def issue_mechanism_ablation_authorization(
    spec: MechanismAblationSpec,
    *,
    execution_id: str,
) -> tuple[StudyProtocol, ProtocolExecutionAuthorization, datetime]:
    """Issue one process-local, single-use synthetic ablation capability."""

    protocol = build_mechanism_ablation_protocol(spec)
    started_at = protocol.registered_at
    authorization = SyntheticProtocolExecutionIssuer(protocol).issue(
        execution_id=execution_id,
        consumer=ProtocolExecutionConsumer.MECHANISM_ABLATION_WORKFLOW,
        purpose=ProtocolExecutionPurpose.SYNTHETIC_MECHANISM_ABLATION,
        config_sha256=spec.sha256,
        input_sha256=mechanism_ablation_execution_input_sha256(spec),
        execution_started_at=started_at,
    )
    return protocol, authorization, started_at


def _registered_output_cells(
    protocol: StudyProtocol,
) -> tuple[tuple[str, str, str], ...]:
    cells = tuple(
        (item.horizon_id, item.cohort_id, item.metric_id) for item in protocol.outputs
    )
    if len(set(cells)) != len(cells):
        _fail("mechanism-ablation protocol output cells must be unique")
    return tuple(sorted(cells))


def _registered_output_mappings(
    protocol: StudyProtocol,
) -> tuple[dict[str, object], ...]:
    return tuple(
        item.to_mapping()
        for item in sorted(protocol.outputs, key=lambda item: item.output_id)
    )


def _validate_study_protocol(
    protocol: StudyProtocol,
    spec: MechanismAblationSpec,
) -> None:
    if not isinstance(protocol, StudyProtocol):
        _fail("mechanism ablations require a typed StudyProtocol")
    if protocol.mode is not StudyMode.SYNTHETIC:
        _fail("mechanism ablations accept only a synthetic study protocol")
    if protocol.claim_scope is not ClaimScope.SYNTHETIC_ONLY:
        _fail("mechanism ablations cannot authorize a non-synthetic claim")
    if protocol.study_id != spec.root_event_key.experiment_id:
        _fail("mechanism-ablation spec belongs to another study protocol")
    reference_pair = _protocol_reference_pair(spec)
    if (
        protocol.estimand.definition_id != MECHANISM_ABLATION_ESTIMAND_ID
        or protocol.estimand.definition_sha256
        != mechanism_ablation_estimand_definition_sha256()
    ):
        _fail("mechanism-ablation estimand differs from its frozen definition")
    if (
        protocol.baseline_policy.definition_id
        != reference_pair.baseline_policy.policy_id
        or protocol.baseline_policy.definition_sha256
        != reference_pair.baseline_policy.sha256
        or protocol.candidate_policy.definition_id
        != reference_pair.candidate_policy.policy_id
        or protocol.candidate_policy.definition_sha256
        != reference_pair.candidate_policy.sha256
    ):
        _fail("mechanism-ablation internal policy definitions changed")
    protocol_cohorts = {item.cohort_id: item for item in protocol.cohorts}
    pair_groups = {item.group_id: item for item in reference_pair.groups}
    if set(protocol_cohorts) != set(pair_groups):
        _fail("mechanism-ablation cohorts differ from the executable grid")
    for group_id, group in pair_groups.items():
        cohort = protocol_cohorts[group_id]
        if (
            cohort.definition_sha256 != group.cohort_sha256
            or cohort.basis is not CohortBasis.LATENT_STRATEGY
        ):
            _fail("mechanism-ablation cohort semantics changed")
    protocol_horizons = {item.definition_id: item for item in protocol.horizons}
    pair_horizons = {item.horizon_id: item for item in reference_pair.horizons}
    if set(protocol_horizons) != set(pair_horizons):
        _fail("mechanism-ablation horizons differ from the executable grid")
    for horizon_id, horizon in pair_horizons.items():
        if protocol_horizons[horizon_id].definition_sha256 != _commit(
            horizon.to_mapping()
        ):
            _fail("mechanism-ablation horizon semantics changed")
    protocol_metrics = {item.definition_id: item for item in protocol.metrics}
    if set(protocol_metrics) != set(INTEGRATED_METRIC_IDS):
        _fail("mechanism-ablation metrics differ from the executable grid")
    for metric_id in INTEGRATED_METRIC_IDS:
        if protocol_metrics[
            metric_id
        ].definition_sha256 != integrated_metric_definition_sha256(metric_id):
            _fail("mechanism-ablation metric semantics changed")
    expected_outputs = tuple(
        item.to_mapping() for item in mechanism_ablation_registered_outputs(spec)
    )
    if _registered_output_mappings(protocol) != expected_outputs:
        _fail("mechanism-ablation registered outputs differ from the frozen grid")
    if protocol.policy_evaluation_outcomes != (MECHANISM_ABLATION_OUTCOME_ID,):
        _fail("mechanism-ablation policy outcome registry changed")
    if protocol.randomness_protocol is not RandomnessProtocol.EVENT_KEYED_PAIRED:
        _fail("mechanism ablations require event-keyed paired randomness")
    registered_specs = tuple(
        item
        for item in protocol.baselines
        if item.definition_id == MECHANISM_ABLATION_SPEC_DEFINITION_ID
    )
    if (
        len(registered_specs) != 1
        or registered_specs[0].definition_sha256 != spec.sha256
    ):
        _fail("mechanism-ablation grid differs from its frozen protocol definition")
    if protocol.knowledge_cutoff != integrated_synthetic_market_config().cutoff:
        _fail("mechanism-ablation cutoff differs from the study protocol")


@dataclass(frozen=True, slots=True)
class AgentMechanismSetting:
    """Exact configured decision mechanism for one latent synthetic strategy."""

    agent_id: str
    archetype: AgentArchetype
    parameters: tuple[tuple[str, float], ...]
    mechanisms: tuple[tuple[str, bool], ...]
    cognitive_update_interval_steps: int
    policy_config_sha256: str
    risk_limits_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.agent_id, str) or not self.agent_id:
            _fail("agent setting requires an agent_id")
        if not isinstance(self.archetype, AgentArchetype):
            _fail("agent setting archetype has the wrong type")
        if (
            not isinstance(self.parameters, tuple)
            or self.parameters != tuple(sorted(self.parameters))
            or not self.parameters
        ):
            _fail("agent parameters must be a non-empty canonical tuple")
        if (
            not isinstance(self.mechanisms, tuple)
            or self.mechanisms != tuple(sorted(self.mechanisms))
            or not self.mechanisms
        ):
            _fail("agent mechanisms must be a non-empty canonical tuple")
        if any(
            not isinstance(name, str)
            or isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
            for name, value in self.parameters
        ):
            _fail("agent parameters contain invalid values")
        if any(
            not isinstance(name, str) or not isinstance(value, bool)
            for name, value in self.mechanisms
        ):
            _fail("agent mechanisms contain invalid switches")
        if (
            isinstance(self.cognitive_update_interval_steps, bool)
            or not isinstance(self.cognitive_update_interval_steps, int)
            or self.cognitive_update_interval_steps <= 0
        ):
            _fail("cognitive update interval must be positive")
        _sha256(self.policy_config_sha256, "policy_config_sha256")
        _sha256(self.risk_limits_sha256, "risk_limits_sha256")

    @classmethod
    def from_agent(cls, agent: StructuredAgent) -> AgentMechanismSetting:
        return cls(
            agent_id=agent.agent_id,
            archetype=agent.archetype,
            parameters=tuple(sorted(agent.parameters.to_mapping().items())),
            mechanisms=tuple(sorted(agent.mechanisms.to_mapping().items())),
            cognitive_update_interval_steps=agent.cognitive_update_interval_steps,
            policy_config_sha256=_commit(agent.policy_config.to_mapping()),
            risk_limits_sha256=_commit(agent.risk_limits.to_mapping()),
        )

    def structural_mapping(self) -> dict[str, object]:
        """Return mechanism settings excluding the identifier."""
        return {
            "archetype": self.archetype.value,
            "cognitive_update_interval_steps": self.cognitive_update_interval_steps,
            "mechanisms": dict(self.mechanisms),
            "parameters": dict(self.parameters),
            "policy_config_sha256": self.policy_config_sha256,
            "risk_limits_sha256": self.risk_limits_sha256,
        }

    def to_mapping(self) -> dict[str, object]:
        return {"agent_id": self.agent_id, **self.structural_mapping()}


class SyntheticBehaviorLabel(StrEnum):
    """Closed qualitative vocabulary emitted by the offline mock labeler."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


_LABEL_VALUE = {
    SyntheticBehaviorLabel.LOW: 0.15,
    SyntheticBehaviorLabel.MEDIUM: 0.45,
    SyntheticBehaviorLabel.HIGH: 0.8,
}


@dataclass(frozen=True, slots=True)
class PromptedBehaviorLabelAssignment:
    """One deterministic mock label and the continuous value actually executed."""

    agent_id: str
    parameter_name: str
    label: SyntheticBehaviorLabel
    executed_value: float
    labeler_id: str = "offline-deterministic-labeler.v1"

    def __post_init__(self) -> None:
        if not self.agent_id or not self.parameter_name:
            _fail("prompted-label assignment identity must be non-empty")
        if not isinstance(self.label, SyntheticBehaviorLabel):
            _fail("prompted-label assignment has an unsupported label")
        value = _finite(self.executed_value, "prompted-label executed_value")
        if value != _LABEL_VALUE[self.label]:
            _fail("prompted-label assignment disagrees with the frozen label map")
        object.__setattr__(self, "executed_value", value)
        if self.labeler_id != "offline-deterministic-labeler.v1":
            _fail("prompted-label assignment uses an unknown labeler")

    def to_mapping(self) -> dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "executed_value": self.executed_value,
            "label": self.label.value,
            "labeler_id": self.labeler_id,
            "parameter_name": self.parameter_name,
        }


@dataclass(frozen=True, slots=True)
class ExecutedMechanismSettings:
    """Executor-derived switches, never copied from an experiment label."""

    variant: MechanismAblationVariant
    population_definition: str
    behavior_definition: str
    social_enabled: bool
    memory_enabled: bool
    agents: tuple[AgentMechanismSetting, ...]
    prompted_label_assignments: tuple[PromptedBehaviorLabelAssignment, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.variant, MechanismAblationVariant):
            _fail("mechanism settings have an unsupported variant")
        if not self.population_definition or not self.behavior_definition:
            _fail("mechanism definitions must be explicit")
        if not isinstance(self.social_enabled, bool) or not isinstance(
            self.memory_enabled, bool
        ):
            _fail("social and memory settings must be boolean")
        if not self.agents or self.agents != tuple(
            sorted(self.agents, key=lambda item: item.agent_id)
        ):
            _fail("agent settings must use canonical order")
        structures = {_commit(item.structural_mapping()) for item in self.agents}
        all_switches = [value for item in self.agents for _, value in item.mechanisms]
        if self.variant is MechanismAblationVariant.HOMOGENEOUS_AGENTS:
            if (
                len(structures) != 1
                or self.population_definition != "homogeneous_policy"
            ):
                _fail("homogeneous variant did not execute a homogeneous policy")
        elif self.population_definition != "heterogeneous_policy":
            _fail("non-homogeneous variant changed the population definition")
        if self.variant is MechanismAblationVariant.RATIONAL_ONLY:
            if (
                any(all_switches)
                or self.behavior_definition != "behavior_switches_disabled"
            ):
                _fail("rational-only variant retained a behavioral switch")
        elif self.variant is MechanismAblationVariant.PROMPTED_LABELS_MOCK:
            if self.behavior_definition != "deterministic_mock_prompted_labels":
                _fail("prompted-label variant lacks its offline mock definition")
            expected = {
                (agent.agent_id, name, value)
                for agent in self.agents
                for name, value in agent.parameters
            }
            actual = {
                (item.agent_id, item.parameter_name, item.executed_value)
                for item in self.prompted_label_assignments
            }
            if actual != expected or len(actual) != len(
                self.prompted_label_assignments
            ):
                _fail("prompted-label receipts do not cover executed parameters")
        elif self.behavior_definition != "synthetic_continuous_uncalibrated":
            _fail("continuous synthetic behavior must not be called calibrated")
        if (
            self.variant is not MechanismAblationVariant.PROMPTED_LABELS_MOCK
            and self.prompted_label_assignments
        ):
            _fail("non-prompted variants cannot carry prompted-label assignments")
        if self.social_enabled != (
            self.variant is not MechanismAblationVariant.NO_SOCIAL_GRAPH
        ):
            _fail("social switch does not match the executed variant")
        if self.memory_enabled != (
            self.variant is not MechanismAblationVariant.NO_MEMORY
        ):
            _fail("memory switch does not match the executed variant")

    @classmethod
    def from_config(
        cls,
        variant: MechanismAblationVariant,
        config: IntegratedSyntheticMarketConfig,
    ) -> ExecutedMechanismSettings:
        assignments = (
            tuple(
                PromptedBehaviorLabelAssignment(
                    agent_id=agent.agent_id,
                    parameter_name=name,
                    label=_label_for_executed_value(value),
                    executed_value=value,
                )
                for agent in config.agents
                for name, value in sorted(agent.parameters.to_mapping().items())
            )
            if variant is MechanismAblationVariant.PROMPTED_LABELS_MOCK
            else ()
        )
        return cls(
            variant=variant,
            population_definition=(
                "homogeneous_policy"
                if variant is MechanismAblationVariant.HOMOGENEOUS_AGENTS
                else "heterogeneous_policy"
            ),
            behavior_definition=(
                "behavior_switches_disabled"
                if variant is MechanismAblationVariant.RATIONAL_ONLY
                else (
                    "deterministic_mock_prompted_labels"
                    if variant is MechanismAblationVariant.PROMPTED_LABELS_MOCK
                    else "synthetic_continuous_uncalibrated"
                )
            ),
            social_enabled=config.social_enabled,
            memory_enabled=config.memory_enabled,
            agents=tuple(
                AgentMechanismSetting.from_agent(item) for item in config.agents
            ),
            prompted_label_assignments=assignments,
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "agents": [item.to_mapping() for item in self.agents],
            "behavior_definition": self.behavior_definition,
            "memory_enabled": self.memory_enabled,
            "population_definition": self.population_definition,
            "prompted_label_assignments": [
                item.to_mapping() for item in self.prompted_label_assignments
            ],
            "social_enabled": self.social_enabled,
            "variant": self.variant.value,
        }

    @property
    def sha256(self) -> str:
        return _commit(self.to_mapping())


def _all_mechanisms_disabled() -> BehaviorMechanisms:
    return BehaviorMechanisms(
        risk_aversion=False,
        loss_aversion=False,
        disposition_strength=False,
        overconfidence=False,
        herding=False,
        attention_sensitivity=False,
        recency_weight=False,
        memory_decay=False,
        inventory_aversion=False,
    )


def _label_for_source_value(value: float) -> SyntheticBehaviorLabel:
    if value < 0.25:
        return SyntheticBehaviorLabel.LOW
    if value < 0.6:
        return SyntheticBehaviorLabel.MEDIUM
    return SyntheticBehaviorLabel.HIGH


def _label_for_executed_value(value: float) -> SyntheticBehaviorLabel:
    matches = tuple(label for label, anchor in _LABEL_VALUE.items() if value == anchor)
    if len(matches) != 1:
        _fail("executed prompted-label value is not a frozen label anchor")
    return matches[0]


def _prompted_mock_parameters(parameters: BehaviorParameters) -> BehaviorParameters:
    """Coarsen continuous fixture values through an explicit offline label map."""
    labeled = {
        name: _LABEL_VALUE[_label_for_source_value(value)]
        for name, value in parameters.to_mapping().items()
    }
    return BehaviorParameters(**labeled)


def _materialize_config(
    variant: MechanismAblationVariant,
) -> IntegratedSyntheticMarketConfig:
    base = integrated_synthetic_market_config()
    if variant is MechanismAblationVariant.HETEROGENEOUS_CONTINUOUS:
        return base
    if variant is MechanismAblationVariant.PROMPTED_LABELS_MOCK:
        return replace(
            base,
            agents=tuple(
                replace(agent, parameters=_prompted_mock_parameters(agent.parameters))
                for agent in base.agents
            ),
        )
    if variant is MechanismAblationVariant.HOMOGENEOUS_AGENTS:
        prototype = base.agents[0]
        agents = tuple(
            replace(
                agent,
                archetype=prototype.archetype,
                parameters=prototype.parameters,
                mechanisms=prototype.mechanisms,
                policy_config=prototype.policy_config,
                risk_limits=prototype.risk_limits,
                cognitive_update_interval_steps=(
                    prototype.cognitive_update_interval_steps
                ),
            )
            for agent in base.agents
        )
        return replace(base, agents=agents)
    if variant is MechanismAblationVariant.RATIONAL_ONLY:
        disabled = _all_mechanisms_disabled()
        return replace(
            base,
            agents=tuple(replace(agent, mechanisms=disabled) for agent in base.agents),
        )
    if variant is MechanismAblationVariant.NO_SOCIAL_GRAPH:
        return replace(base, social_enabled=False)
    if variant is MechanismAblationVariant.NO_MEMORY:
        return replace(base, memory_enabled=False)
    raise AssertionError(f"unhandled mechanism ablation: {variant}")


def _pair_input_mapping(pair: PolicyWorldPair) -> dict[str, object]:
    return {
        "baseline_policy": pair.baseline_policy.to_mapping(),
        "candidate_policy": pair.candidate_policy.to_mapping(),
        "groups": [item.to_mapping() for item in pair.groups],
        "horizons": [item.to_mapping() for item in pair.horizons],
        "initial_state": pair.initial_state.to_mapping(),
        "numeric_tolerance": pair.numeric_tolerance,
        "posterior_draw": pair.posterior_draw.to_mapping(),
        "scenario_id": pair.scenario_id,
        "shock_tape": pair.shock_tape.to_mapping(),
        "shock_transition_config": pair.shock_transition_config.to_mapping(),
        "synthetic_only": pair.synthetic_only,
    }


@dataclass(frozen=True, slots=True)
class AblationMetricCell:
    """One exposed synthetic outcome cell for cross-variant comparison."""

    horizon_id: str
    group_id: str
    metric_name: str
    value: float

    def __post_init__(self) -> None:
        for field in ("horizon_id", "group_id", "metric_name"):
            if not isinstance(getattr(self, field), str) or not getattr(self, field):
                _fail(f"{field} must be non-empty")
        object.__setattr__(self, "value", _finite(self.value, "metric value"))

    def to_mapping(self) -> dict[str, object]:
        return {
            "group_id": self.group_id,
            "horizon_id": self.horizon_id,
            "metric_name": self.metric_name,
            "value": self.value,
        }


def _metric_cells(outcome: PairedOutcome) -> tuple[AblationMetricCell, ...]:
    cells = tuple(
        AblationMetricCell(
            horizon_id=horizon.horizon_id,
            group_id=group.group_id,
            metric_name=metric.metric_name,
            value=metric.value,
        )
        for horizon in outcome.baseline.horizons
        for group in horizon.groups
        for metric in group.metrics
    )
    return tuple(
        sorted(
            cells, key=lambda item: (item.horizon_id, item.group_id, item.metric_name)
        )
    )


@dataclass(frozen=True, slots=True)
class MechanismExecutionReceipt:
    """Complete proof for one successfully replayed variant and replicate."""

    spec_sha256: str
    implementation_receipt_sha256: str
    variant_request_sha256: str
    event_key_value: str
    integrated_config_sha256: str
    mechanism_settings: ExecutedMechanismSettings
    baseline_policy_id: str
    baseline_policy_sha256: str
    candidate_policy_id: str
    candidate_policy_sha256: str
    pair_input_sha256: str
    shock_event_keys: tuple[str, ...]
    baseline_decision_event_keys: tuple[str, ...]
    candidate_decision_event_keys: tuple[str, ...]
    paired_outcome_sha256: str
    baseline_artifact_sha256: str
    candidate_artifact_sha256: str
    metric_cells: tuple[AblationMetricCell, ...]
    claim_scope: str = MECHANISM_ABLATION_CLAIM_SCOPE
    synthetic_only: Literal[True] = True

    def __post_init__(self) -> None:
        for field in (
            "spec_sha256",
            "implementation_receipt_sha256",
            "variant_request_sha256",
            "integrated_config_sha256",
            "baseline_policy_sha256",
            "candidate_policy_sha256",
            "pair_input_sha256",
            "paired_outcome_sha256",
            "baseline_artifact_sha256",
            "candidate_artifact_sha256",
        ):
            _sha256(getattr(self, field), field)
        for field in ("baseline_policy_id", "candidate_policy_id"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value:
                _fail(f"{field} must be non-empty")
        if not isinstance(
            self.event_key_value, str
        ) or not self.event_key_value.startswith("ptk1:"):
            _fail("receipt event key has an invalid format")
        if not isinstance(self.mechanism_settings, ExecutedMechanismSettings):
            _fail("receipt mechanism settings have the wrong type")
        if (
            not isinstance(self.shock_event_keys, tuple)
            or not self.shock_event_keys
            or not all(
                isinstance(item, str) and item.startswith("pse1:")
                for item in self.shock_event_keys
            )
            or len(set(self.shock_event_keys)) != len(self.shock_event_keys)
        ):
            _fail("shock EventKey manifest must be complete and unique")
        if (
            not isinstance(self.baseline_decision_event_keys, tuple)
            or not self.baseline_decision_event_keys
            or not all(
                isinstance(item, str) and item.startswith("ptk1:")
                for item in self.baseline_decision_event_keys
            )
            or len(set(self.baseline_decision_event_keys))
            != len(self.baseline_decision_event_keys)
        ):
            _fail("decision EventKey manifest must not be empty")
        if not isinstance(self.candidate_decision_event_keys, tuple):
            _fail("candidate decision EventKey manifest must be immutable")
        if self.baseline_decision_event_keys != self.candidate_decision_event_keys:
            _fail("no-op worlds must use identical decision EventKeys")
        if not self.metric_cells or self.metric_cells != tuple(
            sorted(
                self.metric_cells,
                key=lambda item: (item.horizon_id, item.group_id, item.metric_name),
            )
        ):
            _fail("metric cells must be non-empty and canonical")
        keys = [
            (item.horizon_id, item.group_id, item.metric_name)
            for item in self.metric_cells
        ]
        if len(set(keys)) != len(keys):
            _fail("metric cells must be unique")
        if self.claim_scope != MECHANISM_ABLATION_CLAIM_SCOPE:
            _fail("mechanism receipt claim scope changed")
        if self.synthetic_only is not True:
            _fail("mechanism receipt must remain synthetic-only")

    def to_mapping(self) -> dict[str, object]:
        return {
            "baseline_artifact_sha256": self.baseline_artifact_sha256,
            "baseline_decision_event_keys": list(self.baseline_decision_event_keys),
            "baseline_policy_id": self.baseline_policy_id,
            "baseline_policy_sha256": self.baseline_policy_sha256,
            "candidate_artifact_sha256": self.candidate_artifact_sha256,
            "candidate_decision_event_keys": list(self.candidate_decision_event_keys),
            "candidate_policy_id": self.candidate_policy_id,
            "candidate_policy_sha256": self.candidate_policy_sha256,
            "claim_scope": self.claim_scope,
            "event_key_value": self.event_key_value,
            "implementation_receipt_sha256": self.implementation_receipt_sha256,
            "integrated_config_sha256": self.integrated_config_sha256,
            "mechanism_settings": self.mechanism_settings.to_mapping(),
            "metric_cells": [item.to_mapping() for item in self.metric_cells],
            "pair_input_sha256": self.pair_input_sha256,
            "paired_outcome_sha256": self.paired_outcome_sha256,
            "shock_event_keys": list(self.shock_event_keys),
            "spec_sha256": self.spec_sha256,
            "synthetic_only": self.synthetic_only,
            "variant_request_sha256": self.variant_request_sha256,
        }

    @property
    def sha256(self) -> str:
        return _commit(self.to_mapping())


@dataclass(frozen=True, slots=True)
class MechanismExecutionFailure:
    """Machine-readable terminal state for one non-completed grid cell."""

    stage: str
    exception_type: str
    message: str

    def __post_init__(self) -> None:
        if not self.stage or not self.exception_type or not self.message:
            _fail("failure stage, exception type, and message must be non-empty")

    def to_mapping(self) -> dict[str, object]:
        return {
            "exception_type": self.exception_type,
            "message": self.message,
            "stage": self.stage,
        }


@dataclass(frozen=True, slots=True)
class MechanismAblationResult:
    """Exactly one expected grid cell, including terminal failure state."""

    variant: MechanismAblationVariant
    replicate_index: int
    event_key: EventKey
    requested_configuration_sha256: str
    implementation_receipt_sha256: str
    status: MechanismAblationStatus
    executed_configuration_sha256: str | None
    receipt: MechanismExecutionReceipt | None
    failure: MechanismExecutionFailure | None

    def __post_init__(self) -> None:
        if not isinstance(self.variant, MechanismAblationVariant):
            _fail("result variant has the wrong type")
        if (
            isinstance(self.replicate_index, bool)
            or not isinstance(self.replicate_index, int)
            or self.replicate_index < 0
        ):
            _fail("replicate_index must be non-negative")
        if not isinstance(self.event_key, EventKey):
            _fail("result event_key must be an EventKey")
        _sha256(self.requested_configuration_sha256, "requested_configuration_sha256")
        _sha256(self.implementation_receipt_sha256, "implementation_receipt_sha256")
        if not isinstance(self.status, MechanismAblationStatus):
            _fail("result status has the wrong type")
        if self.status is MechanismAblationStatus.COMPLETED:
            if self.executed_configuration_sha256 is None:
                _fail("completed result lacks an executed configuration")
            _sha256(self.executed_configuration_sha256, "executed_configuration_sha256")
            if not isinstance(self.receipt, MechanismExecutionReceipt):
                _fail("completed result lacks a mechanism receipt")
            if self.failure is not None:
                _fail("completed result cannot carry a failure")
            if (
                self.receipt.integrated_config_sha256
                != self.executed_configuration_sha256
                or self.receipt.event_key_value != self.event_key.value
                or self.receipt.implementation_receipt_sha256
                != self.implementation_receipt_sha256
                or self.receipt.variant_request_sha256
                != self.requested_configuration_sha256
                or self.receipt.mechanism_settings.variant is not self.variant
            ):
                _fail("completed result disagrees with its execution receipt")
        else:
            if self.receipt is not None or not isinstance(
                self.failure, MechanismExecutionFailure
            ):
                _fail("failed result must carry only a complete failure state")
            if self.executed_configuration_sha256 is not None:
                _sha256(
                    self.executed_configuration_sha256,
                    "executed_configuration_sha256",
                )

    def to_mapping(self) -> dict[str, object]:
        return {
            "event_key": _event_key_mapping(self.event_key),
            "executed_configuration_sha256": self.executed_configuration_sha256,
            "failure": None if self.failure is None else self.failure.to_mapping(),
            "implementation_receipt_sha256": self.implementation_receipt_sha256,
            "receipt": None if self.receipt is None else self.receipt.to_mapping(),
            "replicate_index": self.replicate_index,
            "requested_configuration_sha256": self.requested_configuration_sha256,
            "status": self.status.value,
            "variant": self.variant.value,
        }


def _requested_configuration_sha256(
    spec: MechanismAblationSpec,
    variant: MechanismAblationVariant,
) -> str:
    return _commit(
        {
            "base_config_sha256": integrated_synthetic_market_config().sha256,
            "claim_scope": MECHANISM_ABLATION_CLAIM_SCOPE,
            "spec_sha256": spec.sha256,
            "variant": variant.value,
        }
    )


def _artifacts(
    factory: IntegratedSyntheticWorldFactory,
) -> tuple[IntegratedSyntheticRunArtifact, IntegratedSyntheticRunArtifact]:
    baseline = factory.runners[WorldRole.BASELINE].artifact
    candidate = factory.runners[WorldRole.CANDIDATE].artifact
    if baseline is None or candidate is None:
        raise MechanismAblationIntegrityError(
            "integrated runner completed without world artifacts"
        )
    return baseline, candidate


def _execute_completed(
    *,
    spec: MechanismAblationSpec,
    variant: MechanismAblationVariant,
    event_key: EventKey,
    config: IntegratedSyntheticMarketConfig,
    implementation: MechanismImplementationReceipt,
    requested_sha256: str,
) -> MechanismExecutionReceipt:
    settings = ExecutedMechanismSettings.from_config(variant, config)
    pair = integrated_synthetic_pair(
        event_key=event_key,
        parameters={
            "behavior_scale": spec.behavior_scale,
            "signal_scale": spec.signal_scale,
        },
        config=config,
        candidate_is_noop=True,
        step_count=spec.step_count,
        include_all_groups=True,
    )
    factory = IntegratedSyntheticWorldFactory(config)
    outcome = pair.run(factory, parallel=True)
    if any(item.difference != 0.0 for item in outcome.differences):
        raise MechanismAblationIntegrityError(
            "no-op mechanism evidence produced a non-zero paired difference"
        )
    baseline, candidate = _artifacts(factory)
    baseline_keys = tuple(
        decision.event_key for step in baseline.steps for decision in step.decisions
    )
    candidate_keys = tuple(
        decision.event_key for step in candidate.steps for decision in step.decisions
    )
    shock_keys = tuple(item.shock_event_key for item in baseline.steps)
    if shock_keys != tuple(item.shock_event_key for item in candidate.steps):
        raise MechanismAblationIntegrityError(
            "paired ablation worlds consumed different shock EventKeys"
        )
    return MechanismExecutionReceipt(
        spec_sha256=spec.sha256,
        implementation_receipt_sha256=implementation.sha256,
        variant_request_sha256=requested_sha256,
        event_key_value=event_key.value,
        integrated_config_sha256=config.sha256,
        mechanism_settings=settings,
        baseline_policy_id=pair.baseline_policy.policy_id,
        baseline_policy_sha256=pair.baseline_policy.sha256,
        candidate_policy_id=pair.candidate_policy.policy_id,
        candidate_policy_sha256=pair.candidate_policy.sha256,
        pair_input_sha256=_commit(_pair_input_mapping(pair)),
        shock_event_keys=shock_keys,
        baseline_decision_event_keys=baseline_keys,
        candidate_decision_event_keys=candidate_keys,
        paired_outcome_sha256=outcome.sha256,
        baseline_artifact_sha256=baseline.sha256,
        candidate_artifact_sha256=candidate.sha256,
        metric_cells=_metric_cells(outcome),
    )


def _run_one(
    *,
    spec: MechanismAblationSpec,
    variant: MechanismAblationVariant,
    replicate_index: int,
    implementation: MechanismImplementationReceipt,
) -> MechanismAblationResult:
    event_key = spec.event_key_for(replicate_index)
    requested = _requested_configuration_sha256(spec, variant)
    config: IntegratedSyntheticMarketConfig | None = None
    stage = "materialize_configuration"
    try:
        config = _materialize_config(variant)
        stage = "execute_integrated_pair"
        receipt = _execute_completed(
            spec=spec,
            variant=variant,
            event_key=event_key,
            config=config,
            implementation=implementation,
            requested_sha256=requested,
        )
    except Exception as error:
        return MechanismAblationResult(
            variant=variant,
            replicate_index=replicate_index,
            event_key=event_key,
            requested_configuration_sha256=requested,
            implementation_receipt_sha256=implementation.sha256,
            status=MechanismAblationStatus.FAILED,
            executed_configuration_sha256=(None if config is None else config.sha256),
            receipt=None,
            failure=MechanismExecutionFailure(
                stage=stage,
                exception_type=type(error).__qualname__,
                message=str(error) or type(error).__qualname__,
            ),
        )
    return MechanismAblationResult(
        variant=variant,
        replicate_index=replicate_index,
        event_key=event_key,
        requested_configuration_sha256=requested,
        implementation_receipt_sha256=implementation.sha256,
        status=MechanismAblationStatus.COMPLETED,
        executed_configuration_sha256=config.sha256,
        receipt=receipt,
        failure=None,
    )


@dataclass(frozen=True, slots=True)
class MechanismAblationArtifact:
    """Complete grid, including failed rows, with replayable receipts."""

    protocol_execution_receipt: ProtocolExecutionReceipt
    protocol_canonical_json: str
    execution_input_sha256: str
    spec: MechanismAblationSpec
    implementation_receipt: MechanismImplementationReceipt
    results: tuple[MechanismAblationResult, ...]
    completed_count: int
    failed_count: int
    claim_scope: str = MECHANISM_ABLATION_CLAIM_SCOPE
    synthetic_only: Literal[True] = True
    schema_version: str = MECHANISM_ABLATION_ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        receipt = self.protocol_execution_receipt
        if not isinstance(receipt, ProtocolExecutionReceipt):
            _fail("artifact requires a typed protocol execution receipt")
        if (
            receipt.consumer
            is not ProtocolExecutionConsumer.MECHANISM_ABLATION_WORKFLOW
            or receipt.purpose
            is not ProtocolExecutionPurpose.SYNTHETIC_MECHANISM_ABLATION
        ):
            _fail("artifact protocol receipt has the wrong consumer or purpose")
        try:
            protocol_value = json.loads(self.protocol_canonical_json)
        except (json.JSONDecodeError, TypeError) as error:
            raise MechanismAblationSchemaError(
                "artifact study protocol must be canonical JSON"
            ) from error
        if not isinstance(protocol_value, dict) or not all(
            isinstance(key, str) for key in protocol_value
        ):
            _fail("artifact study protocol must be a JSON object")
        if (
            canonical_json_bytes(protocol_value).decode("utf-8")
            != self.protocol_canonical_json
            or sha256_bytes(self.protocol_canonical_json.encode("utf-8"))
            != receipt.protocol_sha256
        ):
            _fail("artifact study protocol does not match its execution receipt")
        if not isinstance(self.spec, MechanismAblationSpec):
            _fail("artifact spec has the wrong type")
        protocol = StudyProtocol.from_mapping(protocol_value)
        validate_protocol_execution_receipt_binding(protocol, receipt)
        _validate_study_protocol(protocol, self.spec)
        if not isinstance(self.implementation_receipt, MechanismImplementationReceipt):
            _fail("artifact implementation receipt has the wrong type")
        if not isinstance(self.results, tuple) or not all(
            isinstance(item, MechanismAblationResult) for item in self.results
        ):
            _fail("artifact results must be immutable typed rows")
        ordered = tuple(
            sorted(
                self.results,
                key=lambda item: (_VARIANT_ORDER[item.variant], item.replicate_index),
            )
        )
        if self.results != ordered:
            _fail("artifact results must use canonical grid order")
        expected = len(self.spec.variants) * self.spec.replicate_count
        identities = {(item.variant, item.replicate_index) for item in self.results}
        expected_identities = {
            (variant, replicate)
            for variant in self.spec.variants
            for replicate in range(self.spec.replicate_count)
        }
        if len(self.results) != expected or identities != expected_identities:
            _fail("artifact omitted, duplicated, or added an ablation grid cell")
        complete = sum(
            item.status is MechanismAblationStatus.COMPLETED for item in self.results
        )
        failed = len(self.results) - complete
        if self.completed_count != complete or self.failed_count != failed:
            _fail("artifact terminal-state counts do not match the full grid")
        if any(
            item.implementation_receipt_sha256 != self.implementation_receipt.sha256
            for item in self.results
        ):
            _fail("a result is bound to another implementation receipt")
        registered_cells = _registered_output_cells(protocol)
        for result in self.results:
            if result.receipt is None:
                continue
            expected_config = _materialize_config(result.variant)
            expected_settings = ExecutedMechanismSettings.from_config(
                result.variant, expected_config
            )
            expected_pair = integrated_synthetic_pair(
                event_key=result.event_key,
                parameters={
                    "behavior_scale": self.spec.behavior_scale,
                    "signal_scale": self.spec.signal_scale,
                },
                config=expected_config,
                candidate_is_noop=True,
                step_count=self.spec.step_count,
                include_all_groups=True,
            )
            expected_pair_input_sha256 = _commit(_pair_input_mapping(expected_pair))
            expected_policy_commitment = (
                protocol.baseline_policy.definition_id,
                protocol.baseline_policy.definition_sha256,
                protocol.candidate_policy.definition_id,
                protocol.candidate_policy.definition_sha256,
            )
            pair_policy_commitment = (
                expected_pair.baseline_policy.policy_id,
                expected_pair.baseline_policy.sha256,
                expected_pair.candidate_policy.policy_id,
                expected_pair.candidate_policy.sha256,
            )
            receipt_policy_commitment = (
                result.receipt.baseline_policy_id,
                result.receipt.baseline_policy_sha256,
                result.receipt.candidate_policy_id,
                result.receipt.candidate_policy_sha256,
            )
            if (
                pair_policy_commitment != expected_policy_commitment
                or receipt_policy_commitment != expected_policy_commitment
            ):
                _fail(
                    "completed mechanism receipt used policies outside the frozen "
                    "protocol"
                )
            if (
                result.receipt.spec_sha256 != self.spec.sha256
                or result.receipt.integrated_config_sha256 != expected_config.sha256
                or result.executed_configuration_sha256 != expected_config.sha256
                or result.receipt.mechanism_settings.sha256 != expected_settings.sha256
                or result.receipt.variant_request_sha256
                != _requested_configuration_sha256(self.spec, result.variant)
                or result.receipt.pair_input_sha256 != expected_pair_input_sha256
            ):
                _fail(
                    "completed mechanism receipt differs from its reconstructed "
                    "execution input"
                )
            observed_cells = tuple(
                (
                    item.horizon_id,
                    item.group_id,
                    item.metric_name,
                )
                for item in result.receipt.metric_cells
            )
            if observed_cells != registered_cells:
                _fail(
                    "completed mechanism output cells differ from the frozen protocol"
                )
        expected_input = _execution_input_sha256(self.spec, self.implementation_receipt)
        _sha256(self.execution_input_sha256, "execution_input_sha256")
        if (
            self.execution_input_sha256 != expected_input
            or receipt.config_sha256 != self.spec.sha256
            or receipt.input_sha256 != expected_input
        ):
            _fail("artifact protocol receipt does not bind the full grid input")
        for replicate in range(self.spec.replicate_count):
            keys = {
                item.event_key.value
                for item in self.results
                if item.replicate_index == replicate
            }
            if keys != {self.spec.event_key_for(replicate).value}:
                _fail("variants are not coupled to the same replicate EventKey")
            completed_receipts = tuple(
                item.receipt
                for item in self.results
                if item.replicate_index == replicate and item.receipt is not None
            )
            if (
                len({item.shock_event_keys for item in completed_receipts}) > 1
                or len(
                    {item.baseline_decision_event_keys for item in completed_receipts}
                )
                > 1
            ):
                _fail("completed variants used different comparable EventKey manifests")
        for variant in self.spec.variants:
            variant_rows = tuple(
                item for item in self.results if item.variant is variant
            )
            requested = {item.requested_configuration_sha256 for item in variant_rows}
            executed = {
                item.executed_configuration_sha256
                for item in variant_rows
                if item.executed_configuration_sha256 is not None
            }
            settings = {
                item.receipt.mechanism_settings.sha256
                for item in variant_rows
                if item.receipt is not None
            }
            if len(requested) != 1 or len(executed) > 1 or len(settings) > 1:
                _fail("a variant changed configuration across replicates")
        if self.claim_scope != MECHANISM_ABLATION_CLAIM_SCOPE:
            _fail("artifact claim scope changed")
        if self.synthetic_only is not True:
            _fail("artifact must remain synthetic-only")
        if self.schema_version != MECHANISM_ABLATION_ARTIFACT_SCHEMA_VERSION:
            _fail("artifact schema version changed")

    def result_for(
        self, variant: MechanismAblationVariant, replicate_index: int
    ) -> MechanismAblationResult:
        """Return one explicit terminal state from the frozen grid."""
        return _mechanism_result_for(self.results, variant, replicate_index)

    def metric_contrast(
        self,
        variant: MechanismAblationVariant,
        replicate_index: int,
    ) -> tuple[AblationMetricCell, ...]:
        """Return variant-minus-reference cells or fail on any failed run."""
        return _mechanism_metric_contrast(
            self.results,
            variant,
            replicate_index,
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "claim_scope": self.claim_scope,
            "completed_count": self.completed_count,
            "execution_input_sha256": self.execution_input_sha256,
            "failed_count": self.failed_count,
            "implementation_receipt": self.implementation_receipt.to_mapping(),
            "protocol": json.loads(self.protocol_canonical_json),
            "protocol_execution_receipt": (
                self.protocol_execution_receipt.to_mapping()
            ),
            "results": [item.to_mapping() for item in self.results],
            "schema_version": self.schema_version,
            "spec": self.spec.to_mapping(),
            "synthetic_only": self.synthetic_only,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_mapping())

    @property
    def sha256(self) -> str:
        return sha256_bytes(self.canonical_bytes)


def _mechanism_result_for(
    results: tuple[MechanismAblationResult, ...],
    variant: MechanismAblationVariant,
    replicate_index: int,
) -> MechanismAblationResult:
    for result in results:
        if result.variant is variant and result.replicate_index == replicate_index:
            return result
    raise KeyError((variant, replicate_index))


def _mechanism_metric_contrast(
    results: tuple[MechanismAblationResult, ...],
    variant: MechanismAblationVariant,
    replicate_index: int,
) -> tuple[AblationMetricCell, ...]:
    reference = _mechanism_result_for(
        results,
        MechanismAblationVariant.HETEROGENEOUS_CONTINUOUS,
        replicate_index,
    )
    variant_result = _mechanism_result_for(results, variant, replicate_index)
    if reference.receipt is None or variant_result.receipt is None:
        raise MechanismAblationIntegrityError(
            "contrast is unavailable because an expected run failed"
        )
    reference_values = {
        (item.horizon_id, item.group_id, item.metric_name): item.value
        for item in reference.receipt.metric_cells
    }
    variant_values = {
        (item.horizon_id, item.group_id, item.metric_name): item.value
        for item in variant_result.receipt.metric_cells
    }
    if reference_values.keys() != variant_values.keys():
        raise MechanismAblationIntegrityError(
            "variant and reference expose different metric cells"
        )
    return tuple(
        AblationMetricCell(horizon, group, metric, variant_values[key] - value)
        for key, value in sorted(reference_values.items())
        for horizon, group, metric in (key,)
    )


def _execute_mechanism_ablation_grid(
    spec: MechanismAblationSpec,
    implementation: MechanismImplementationReceipt | None = None,
) -> tuple[MechanismImplementationReceipt, tuple[MechanismAblationResult, ...]]:
    """Execute the deterministic grid without minting a research artifact."""
    if not isinstance(spec, MechanismAblationSpec):
        _fail("spec must be a MechanismAblationSpec")
    live_implementation = (
        _implementation_receipt() if implementation is None else implementation
    )
    results = tuple(
        _run_one(
            spec=spec,
            variant=variant,
            replicate_index=replicate,
            implementation=live_implementation,
        )
        for variant in spec.variants
        for replicate in range(spec.replicate_count)
    )
    return live_implementation, results


def _artifact_from_verified_receipt(
    *,
    spec: MechanismAblationSpec,
    implementation: MechanismImplementationReceipt,
    results: tuple[MechanismAblationResult, ...],
    protocol_execution_receipt: ProtocolExecutionReceipt,
    protocol_canonical_json: str,
) -> MechanismAblationArtifact:
    complete = sum(item.status is MechanismAblationStatus.COMPLETED for item in results)
    return MechanismAblationArtifact(
        protocol_execution_receipt=protocol_execution_receipt,
        protocol_canonical_json=protocol_canonical_json,
        execution_input_sha256=_execution_input_sha256(spec, implementation),
        spec=spec,
        implementation_receipt=implementation,
        results=results,
        completed_count=complete,
        failed_count=len(results) - complete,
    )


def run_mechanism_ablations(
    *,
    spec: MechanismAblationSpec,
    study_protocol: StudyProtocol,
    protocol_authorization: ProtocolExecutionAuthorization,
    execution_started_at: datetime,
) -> MechanismAblationArtifact:
    """Consume one exact authorization before executing the complete grid."""
    if not isinstance(spec, MechanismAblationSpec):
        _fail("spec must be a MechanismAblationSpec")
    implementation = _implementation_receipt()
    execution_input_sha256 = _execution_input_sha256(spec, implementation)
    receipt = consume_protocol_execution_authorization(
        protocol_authorization,
        protocol=study_protocol,
        consumer=ProtocolExecutionConsumer.MECHANISM_ABLATION_WORKFLOW,
        purpose=ProtocolExecutionPurpose.SYNTHETIC_MECHANISM_ABLATION,
        config_sha256=spec.sha256,
        input_sha256=execution_input_sha256,
        execution_started_at=execution_started_at,
    )
    _validate_study_protocol(study_protocol, spec)
    # Authorization is consumed before this helper reaches the first grid cell.
    _, results = _execute_mechanism_ablation_grid(spec, implementation)
    return _artifact_from_verified_receipt(
        spec=spec,
        implementation=implementation,
        results=results,
        protocol_execution_receipt=receipt,
        protocol_canonical_json=canonical_json_bytes(
            study_protocol.to_mapping()
        ).decode("utf-8"),
    )


@dataclass(frozen=True, slots=True)
class MechanismAblationAudit:
    """Positive deterministic replay evidence for a complete artifact."""

    artifact_sha256: str
    protocol_execution_receipt_sha256: str
    implementation_receipt_sha256: str
    replayed_result_count: int
    passed: Literal[True] = True

    def __post_init__(self) -> None:
        _sha256(self.artifact_sha256, "artifact_sha256")
        _sha256(
            self.protocol_execution_receipt_sha256,
            "protocol_execution_receipt_sha256",
        )
        _sha256(
            self.implementation_receipt_sha256,
            "implementation_receipt_sha256",
        )
        if self.replayed_result_count <= 0:
            _fail("audit must replay at least one result")
        if self.passed is not True:
            _fail("only a passing replay creates an audit receipt")

    def to_mapping(self) -> dict[str, object]:
        return {
            "artifact_sha256": self.artifact_sha256,
            "implementation_receipt_sha256": self.implementation_receipt_sha256,
            "passed": self.passed,
            "protocol_execution_receipt_sha256": (
                self.protocol_execution_receipt_sha256
            ),
            "replayed_result_count": self.replayed_result_count,
        }


def audit_mechanism_ablation_artifact(
    artifact: MechanismAblationArtifact,
) -> MechanismAblationAudit:
    """Replay all expected cells and reject any receipt or failure drift."""
    if not isinstance(artifact, MechanismAblationArtifact):
        _fail("artifact must be a MechanismAblationArtifact")
    implementation, results = _execute_mechanism_ablation_grid(artifact.spec)
    replay = _artifact_from_verified_receipt(
        spec=artifact.spec,
        implementation=implementation,
        results=results,
        protocol_execution_receipt=artifact.protocol_execution_receipt,
        protocol_canonical_json=artifact.protocol_canonical_json,
    )
    if replay.canonical_bytes != artifact.canonical_bytes:
        raise MechanismAblationIntegrityError(
            "mechanism-ablation artifact does not reproduce from trusted inputs"
        )
    return MechanismAblationAudit(
        artifact_sha256=artifact.sha256,
        protocol_execution_receipt_sha256=(artifact.protocol_execution_receipt.sha256),
        implementation_receipt_sha256=artifact.implementation_receipt.sha256,
        replayed_result_count=len(artifact.results),
    )


def write_mechanism_ablation_artifact(
    directory: Path,
    artifact: MechanismAblationArtifact,
) -> tuple[Path, Path]:
    """Write one immutable JSON artifact and a hash seal, refusing overwrite."""
    if not isinstance(directory, Path):
        _fail("directory must be a Path")
    if not isinstance(artifact, MechanismAblationArtifact):
        _fail("artifact has the wrong type")
    # Publication is an evidence boundary: never persist a merely well-shaped
    # object that has not reproduced from the verifier-owned implementation.
    audit_mechanism_ablation_artifact(artifact)
    directory.mkdir(parents=True, exist_ok=True)
    artifact_path = directory / "mechanism-ablation.json"
    seal_path = directory / "mechanism-ablation.sha256"
    if artifact_path.exists() or seal_path.exists():
        raise MechanismAblationIntegrityError(
            "mechanism-ablation artifact directory is already populated"
        )
    write_json_file(artifact_path, artifact.to_mapping())
    write_text_file(seal_path, f"{sha256_file(artifact_path)}\n")
    return artifact_path, seal_path


def audit_written_mechanism_ablation_artifact(
    directory: Path,
    *,
    expected_spec: MechanismAblationSpec,
    expected_protocol_receipt_sha256: str,
) -> MechanismAblationAudit:
    """Verify the file seal, retained authorization receipt, and live replay."""
    if not isinstance(expected_spec, MechanismAblationSpec):
        _fail("expected_spec must be a MechanismAblationSpec")
    expected_receipt_sha256 = _sha256(
        expected_protocol_receipt_sha256,
        "expected_protocol_receipt_sha256",
    )
    artifact_path = directory / "mechanism-ablation.json"
    seal_path = directory / "mechanism-ablation.sha256"
    try:
        seal = seal_path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError) as error:
        raise MechanismAblationIntegrityError(
            "cannot read mechanism-ablation hash seal"
        ) from error
    try:
        artifact_file_sha256 = sha256_file(artifact_path)
    except OSError as error:
        raise MechanismAblationIntegrityError(
            "cannot read stored mechanism-ablation artifact"
        ) from error
    if _SHA256.fullmatch(seal) is None or seal != artifact_file_sha256:
        raise MechanismAblationIntegrityError(
            "mechanism-ablation file hash does not match its seal"
        )
    try:
        stored = read_json_object(artifact_path)
    except Exception as error:
        raise MechanismAblationIntegrityError(
            "cannot parse stored mechanism-ablation artifact"
        ) from error
    raw_receipt = stored.get("protocol_execution_receipt")
    raw_protocol = stored.get("protocol")
    if not isinstance(raw_receipt, Mapping) or not all(
        isinstance(key, str) for key in raw_receipt
    ):
        raise MechanismAblationIntegrityError(
            "stored mechanism-ablation artifact lacks a protocol receipt"
        )
    if not isinstance(raw_protocol, Mapping) or not all(
        isinstance(key, str) for key in raw_protocol
    ):
        raise MechanismAblationIntegrityError(
            "stored mechanism-ablation artifact lacks a study protocol"
        )
    try:
        receipt = ProtocolExecutionReceipt.from_mapping(raw_receipt)
        protocol = StudyProtocol.from_mapping(raw_protocol)
        validate_protocol_execution_receipt_binding(protocol, receipt)
    except Exception as error:
        raise MechanismAblationIntegrityError(
            "stored mechanism-ablation protocol evidence is invalid"
        ) from error
    if receipt.sha256 != expected_receipt_sha256:
        raise MechanismAblationIntegrityError(
            "stored protocol receipt differs from its external commitment"
        )
    implementation, results = _execute_mechanism_ablation_grid(expected_spec)
    replay = _artifact_from_verified_receipt(
        spec=expected_spec,
        implementation=implementation,
        results=results,
        protocol_execution_receipt=receipt,
        protocol_canonical_json=canonical_json_bytes(protocol.to_mapping()).decode(
            "utf-8"
        ),
    )
    if stored != replay.to_mapping():
        raise MechanismAblationIntegrityError(
            "stored mechanism-ablation artifact differs from authorized evidence"
        )
    return MechanismAblationAudit(
        artifact_sha256=replay.sha256,
        protocol_execution_receipt_sha256=receipt.sha256,
        implementation_receipt_sha256=replay.implementation_receipt.sha256,
        replayed_result_count=len(replay.results),
    )


__all__ = [
    "MECHANISM_ABLATION_ARTIFACT_SCHEMA_VERSION",
    "MECHANISM_ABLATION_CLAIM_SCOPE",
    "MECHANISM_ABLATION_EXECUTION_INPUT_SCHEMA_VERSION",
    "MECHANISM_ABLATION_EXECUTOR_ID",
    "MECHANISM_ABLATION_ESTIMAND_ID",
    "MECHANISM_ABLATION_OUTCOME_ID",
    "MECHANISM_ABLATION_SPEC_DEFINITION_ID",
    "MECHANISM_ABLATION_SCHEMA_VERSION",
    "MECHANISM_ABLATION_SPEC_SCHEMA_VERSION",
    "AblationMetricCell",
    "AgentMechanismSetting",
    "ExecutedMechanismSettings",
    "MechanismAblationArtifact",
    "MechanismAblationAudit",
    "MechanismAblationError",
    "MechanismAblationIntegrityError",
    "MechanismAblationResult",
    "MechanismAblationSchemaError",
    "MechanismAblationSpec",
    "MechanismAblationStatus",
    "MechanismAblationVariant",
    "MechanismExecutionFailure",
    "MechanismExecutionReceipt",
    "MechanismImplementationReceipt",
    "MechanismSourceComponent",
    "PromptedBehaviorLabelAssignment",
    "SyntheticBehaviorLabel",
    "audit_mechanism_ablation_artifact",
    "audit_written_mechanism_ablation_artifact",
    "build_mechanism_ablation_protocol",
    "issue_mechanism_ablation_authorization",
    "mechanism_ablation_execution_input_sha256",
    "mechanism_ablation_estimand_definition_sha256",
    "mechanism_ablation_registered_output_cells",
    "mechanism_ablation_registered_outputs",
    "run_mechanism_ablations",
    "write_mechanism_ablation_artifact",
]
