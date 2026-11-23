"""Replay-audited synthetic evidence for cross-subsystem ablation dimensions.

The integrated mechanism executor owns agent/social/memory/behavior switches.
This companion executes the remaining dimensions through their actual typed
subsystems: ABC point versus posterior propagation, one structural member
versus a complete structural ensemble, frozen abstention versus full coverage,
a knowledge-firewall-authorized deterministic mock versus the structural
no-LLM planner, and a post-cutoff canary attack.  The mock is offline synthetic
interface evidence only; this module never calls or evaluates a real model.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from statistics import fmean
from tempfile import TemporaryDirectory
from typing import Literal, NoReturn

import policytwin.agents.llm_contracts as llm_contracts_module
import policytwin.agents.llm_firewall as llm_firewall_module
import policytwin.agents.llm_gateway as llm_gateway_module
import policytwin.agents.mock_planning as mock_planning_module
import policytwin.calibration.estimation as calibration_estimation_module
import policytwin.calibration.integrated as calibration_integrated_module
import policytwin.counterfactual.integrated_market as integrated_market_module
import policytwin.data.temporal_firewall as temporal_firewall_module
import policytwin.evaluation.baselines as baselines_module
import policytwin.evaluation.ensemble as ensemble_module
import policytwin.evaluation.selective as selective_module
import policytwin.evaluation.structural_generators as structural_generators_module
import policytwin.experiments.integrated_synthetic as integrated_synthetic_module
import policytwin.experiments.mechanism_ablations as mechanism_ablations_module
import policytwin.protocol as protocol_module
import policytwin.social.privacy as social_privacy_module
from policytwin.calibration import (
    ABCPosteriorEstimator,
    IntegratedSyntheticCalibrationAdapter,
    integrated_synthetic_calibration_config,
    sample_posterior,
    synthetic_calibration_input,
)
from policytwin.counterfactual import EventType, SemanticEventKey
from policytwin.counterfactual.integrated_market import (
    IntegratedSyntheticRunArtifact,
    IntegratedSyntheticWorldFactory,
)
from policytwin.counterfactual.paired import WorldRole
from policytwin.counterfactual.randomness import EventKey
from policytwin.data.temporal_firewall import (
    CutoffPolicy,
    PostCutoffDataError,
    RuleCode,
    TemporalAuditLog,
    TemporalFirewall,
    make_post_cutoff_canary,
)
from policytwin.evaluation import (
    AgentInteractionStructuralGenerator,
    BaselineInputBundle,
    EnsembleWeighting,
    FullPolicyTwinSyntheticBaseline,
    OrderFlowStructuralGenerator,
    OrderLevelGeneratorBaseline,
    SelectionRule,
    SelectivePredictionRecord,
    StructuralEnsembleSpec,
    StructuralOutcomeCell,
    SyntheticThresholdSourceReceipt,
    ThresholdSource,
    evaluate_generated_structural_ensemble,
    evaluate_selective_predictions,
)
from policytwin.experiments.integrated_synthetic import (
    integrated_synthetic_market_config,
    integrated_synthetic_mock_market_config,
    integrated_synthetic_pair,
    integrated_synthetic_pair_input_sha256,
)
from policytwin.experiments.mechanism_ablations import (
    MechanismAblationSpec,
    MechanismAblationStatus,
    MechanismAblationVariant,
    _execute_mechanism_ablation_grid,
    _mechanism_metric_contrast,
    _mechanism_result_for,
)
from policytwin.protocol import (
    ClaimScope,
    CohortBasis,
    OutputRole,
    ProtocolExecutionAuthorization,
    ProtocolExecutionConsumer,
    ProtocolExecutionPurpose,
    ProtocolExecutionReceipt,
    RandomnessProtocol,
    RegisteredOutput,
    StudyMode,
    StudyProtocol,
    consume_protocol_execution_authorization,
    validate_protocol_execution_receipt_binding,
)
from policytwin.provenance import canonical_json_bytes, sha256_bytes, sha256_file

EXTENDED_ABLATION_SPEC_SCHEMA_VERSION = "policytwin.synthetic-extended-ablation.v1"
EXTENDED_ABLATION_ARTIFACT_SCHEMA_VERSION = (
    "policytwin.synthetic-extended-ablation-artifact.v2"
)
# Backward-compatible request-schema name. Artifacts use their own version.
EXTENDED_ABLATION_SCHEMA_VERSION = EXTENDED_ABLATION_SPEC_SCHEMA_VERSION
EXTENDED_ABLATION_EXECUTOR_ID = "policytwin.synthetic_extended_ablations.v1"
EXTENDED_ABLATION_CLAIM_SCOPE = "synthetic_cross_subsystem_ablation_only"
EXTENDED_ABLATION_SPEC_DEFINITION_ID = "extended_ablation_spec"
EXTENDED_ABLATION_ESTIMAND_ID = "synthetic_cross_subsystem_contrast_registry"
EXTENDED_ABLATION_REFERENCE_POLICY_ID = "synthetic_cross_subsystem_reference"
EXTENDED_ABLATION_VARIANT_POLICY_ID = "synthetic_cross_subsystem_variants"
EXTENDED_ABLATION_COHORT_ID = "synthetic_all"
EXTENDED_ABLATION_HORIZON_ID = "synthetic_terminal"
EXTENDED_ABLATION_OUTCOME_ID = "synthetic_extended_ablation_contrasts"
EXTENDED_ABLATION_EXECUTION_INPUT_SCHEMA_VERSION = (
    "policytwin.synthetic-extended-ablation-execution-input.v1"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ExtendedAblationError(RuntimeError):
    """Base class for cross-subsystem ablation failures."""


class ExtendedAblationSchemaError(ExtendedAblationError, ValueError):
    """Raised when an ablation request or receipt is malformed."""


class ExtendedAblationIntegrityError(ExtendedAblationError):
    """Raised when saved evidence no longer replays exactly."""


def _fail(message: str) -> NoReturn:
    raise ExtendedAblationSchemaError(message)


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail(f"{field} must be a lowercase SHA-256 digest")
    return value


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


class ExtendedAblationDimension(StrEnum):
    """Closed dimensions handled by this cross-subsystem executor."""

    BEHAVIOR_REPRESENTATION = "behavior_representation"
    CALIBRATION_PROPAGATION = "calibration_propagation"
    MODEL_UNCERTAINTY = "model_uncertainty"
    ABSTENTION = "abstention"
    LLM = "llm"
    TEMPORAL_FIREWALL = "temporal_firewall_negative_control"


_DIMENSIONS = tuple(ExtendedAblationDimension)
_DIMENSION_ORDER = {item: index for index, item in enumerate(_DIMENSIONS)}
EXTENDED_ABLATION_METRIC_IDS = tuple(
    f"{dimension.value}_contrast" for dimension in _DIMENSIONS
)
_METRIC_BY_DIMENSION = dict(zip(_DIMENSIONS, EXTENDED_ABLATION_METRIC_IDS, strict=True))

_EXECUTED_MECHANISM_IDS: dict[ExtendedAblationDimension, tuple[str, str]] = {
    ExtendedAblationDimension.BEHAVIOR_REPRESENTATION: (
        "synthetic_continuous_uncalibrated",
        "deterministic_mock_prompted_labels",
    ),
    ExtendedAblationDimension.CALIBRATION_PROPAGATION: (
        "rejection_abc_posterior_propagation",
        "best_distance_abc_point_propagation",
    ),
    ExtendedAblationDimension.MODEL_UNCERTAINTY: (
        "complete_equal_weight_structural_ensemble",
        "single_agent_interaction_structure",
    ),
    ExtendedAblationDimension.ABSTENTION: (
        "frozen_threshold_selective_prediction",
        "retain_all_without_abstention",
    ),
    ExtendedAblationDimension.LLM: (
        "knowledge_firewall_authorized_deterministic_mock_planner",
        "structural_rule_no_llm_planner",
    ),
}


def extended_ablation_definition_sha256(definition_id: str) -> str:
    """Return a closed verifier-owned commitment for an extended-grid definition."""
    if definition_id == EXTENDED_ABLATION_ESTIMAND_ID:
        semantics: dict[str, object] = {
            "formula": "dimension_specific_variant_minus_reference_or_negative_control",
            "registered_dimensions": [item.value for item in _DIMENSIONS],
            "sampling_unit": "event_keyed_synthetic_execution",
        }
    elif definition_id == EXTENDED_ABLATION_REFERENCE_POLICY_ID:
        semantics = {"arm": "reference", "scope": "cross_subsystem_bundle"}
    elif definition_id == EXTENDED_ABLATION_VARIANT_POLICY_ID:
        semantics = {
            "arm": "variant",
            "dimensions": [item.value for item in _DIMENSIONS],
            "scope": "cross_subsystem_bundle",
        }
    elif definition_id == EXTENDED_ABLATION_COHORT_ID:
        semantics = {"cohort": "all_synthetic_execution_units"}
    elif definition_id == EXTENDED_ABLATION_HORIZON_ID:
        semantics = {"horizon": "dimension_terminal_artifact"}
    elif definition_id in EXTENDED_ABLATION_METRIC_IDS:
        dimension = _DIMENSIONS[EXTENDED_ABLATION_METRIC_IDS.index(definition_id)]
        semantics = {
            "dimension": dimension.value,
            "evidence": (
                "expected_rejection"
                if dimension is ExtendedAblationDimension.TEMPORAL_FIREWALL
                else "executed_contrast"
            ),
        }
        if dimension is ExtendedAblationDimension.TEMPORAL_FIREWALL:
            semantics["expected_rule_code"] = RuleCode.POST_CUTOFF_AVAILABLE_TIME.value
        else:
            reference_id, variant_id = _EXECUTED_MECHANISM_IDS[dimension]
            semantics["reference_mechanism_id"] = reference_id
            semantics["variant_mechanism_id"] = variant_id
    else:
        _fail("unknown extended-ablation semantic definition")
    return _commit(
        {
            "claim_scope": EXTENDED_ABLATION_CLAIM_SCOPE,
            "definition_id": definition_id,
            "schema_version": 1,
            "semantics": semantics,
        }
    )


def extended_ablation_registered_output_cells() -> tuple[tuple[str, str, str], ...]:
    """Return the exact row registry exposed as protocol-authorized evidence."""
    return tuple(
        (item.horizon_id, item.cohort_id, item.metric_id)
        for item in extended_ablation_registered_outputs()
    )


def extended_ablation_registered_outputs() -> tuple[RegisteredOutput, ...]:
    """Return the verifier-owned output IDs, roles, and interval semantics."""
    return tuple(
        RegisteredOutput(
            output_id=f"extended_{metric_id}",
            metric_id=metric_id,
            cohort_id=EXTENDED_ABLATION_COHORT_ID,
            horizon_id=EXTENDED_ABLATION_HORIZON_ID,
            role=(
                OutputRole.PRIMARY
                if metric_id == EXTENDED_ABLATION_METRIC_IDS[0]
                else OutputRole.SECONDARY
            ),
            interval_method="none",
            interval_levels=(),
        )
        for metric_id in EXTENDED_ABLATION_METRIC_IDS
    )


class ExtendedAblationStatus(StrEnum):
    COMPLETED = "completed"
    EXPECTED_REJECTION = "expected_rejection"
    BLOCKED = "blocked"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ExtendedAblationSpec:
    """Frozen synthetic-only request shared by every executed dimension."""

    root_event_key: EventKey
    calibration_simulations: int = 6
    posterior_draws: int = 4
    structural_draws: int = 4
    synthetic_only: Literal[True] = True
    schema_version: str = EXTENDED_ABLATION_SPEC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.root_event_key, EventKey):
            _fail("root_event_key must be an EventKey")
        if not self.root_event_key.value.startswith("ptk1:"):
            _fail("root_event_key has an invalid commitment")
        for field, minimum in (
            ("calibration_simulations", 4),
            ("posterior_draws", 2),
            ("structural_draws", 2),
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                _fail(f"{field} must be an integer >= {minimum}")
        if self.calibration_simulations < 3:
            _fail("calibration must retain at least three accepted candidates")
        if self.synthetic_only is not True:
            _fail("extended ablations are synthetic-only")
        if self.schema_version != EXTENDED_ABLATION_SPEC_SCHEMA_VERSION:
            _fail("unsupported extended-ablation schema version")

    def key(self, purpose: str, step: int = 0) -> EventKey:
        if (
            not purpose
            or isinstance(step, bool)
            or not isinstance(step, int)
            or step < 0
        ):
            _fail("derived EventKey purpose and step must be valid")
        root = self.root_event_key
        return EventKey(
            experiment_id=root.experiment_id,
            seed=root.seed,
            replicate_id=root.replicate_id,
            simulation_step=root.simulation_step + step,
            purpose=f"{root.purpose}:extended-ablation:{purpose}",
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "calibration_simulations": self.calibration_simulations,
            "posterior_draws": self.posterior_draws,
            "root_event_key": _event_key_mapping(self.root_event_key),
            "schema_version": self.schema_version,
            "structural_draws": self.structural_draws,
            "synthetic_only": self.synthetic_only,
        }

    @property
    def sha256(self) -> str:
        return _commit(self.to_mapping())


@dataclass(frozen=True, slots=True)
class ExtendedImplementationReceipt:
    """Commitment to every source module directly executed by this layer."""

    executor_id: str
    components: tuple[tuple[str, str], ...]
    source_manifest_sha256: str

    def __post_init__(self) -> None:
        if self.executor_id != EXTENDED_ABLATION_EXECUTOR_ID:
            _fail("unknown extended-ablation executor")
        if (
            not isinstance(self.components, tuple)
            or not self.components
            or self.components != tuple(sorted(self.components))
            or len({name for name, _ in self.components}) != len(self.components)
        ):
            _fail("implementation components must be non-empty and canonical")
        for name, digest in self.components:
            if not name:
                _fail("implementation component name must be non-empty")
            _sha256(digest, "implementation component digest")
        _sha256(self.source_manifest_sha256, "source_manifest_sha256")
        if self.source_manifest_sha256 != _commit(
            [{"module": name, "sha256": digest} for name, digest in self.components]
        ):
            _fail("source manifest digest does not match implementation components")

    @property
    def sha256(self) -> str:
        return _commit(self.to_mapping())

    def to_mapping(self) -> dict[str, object]:
        return {
            "components": [
                {"module": name, "sha256": digest} for name, digest in self.components
            ],
            "executor_id": self.executor_id,
            "source_manifest_sha256": self.source_manifest_sha256,
        }


def _implementation_receipt() -> ExtendedImplementationReceipt:
    modules = (
        ("policytwin.agents.llm_contracts", llm_contracts_module.__file__),
        ("policytwin.agents.llm_firewall", llm_firewall_module.__file__),
        ("policytwin.agents.llm_gateway", llm_gateway_module.__file__),
        ("policytwin.agents.mock_planning", mock_planning_module.__file__),
        ("policytwin.calibration.estimation", calibration_estimation_module.__file__),
        ("policytwin.calibration.integrated", calibration_integrated_module.__file__),
        (
            "policytwin.counterfactual.integrated_market",
            integrated_market_module.__file__,
        ),
        ("policytwin.data.temporal_firewall", temporal_firewall_module.__file__),
        ("policytwin.evaluation.baselines", baselines_module.__file__),
        ("policytwin.evaluation.ensemble", ensemble_module.__file__),
        ("policytwin.evaluation.selective", selective_module.__file__),
        (
            "policytwin.evaluation.structural_generators",
            structural_generators_module.__file__,
        ),
        (
            "policytwin.experiments.extended_ablations",
            __file__,
        ),
        (
            "policytwin.experiments.integrated_synthetic",
            integrated_synthetic_module.__file__,
        ),
        (
            "policytwin.experiments.mechanism_ablations",
            mechanism_ablations_module.__file__,
        ),
        ("policytwin.protocol", protocol_module.__file__),
        ("policytwin.social.privacy", social_privacy_module.__file__),
    )
    components: list[tuple[str, str]] = []
    for module_name, raw_path in modules:
        if raw_path is None or not Path(raw_path).is_file():
            raise ExtendedAblationIntegrityError(
                f"cannot locate implementation source for {module_name}"
            )
        components.append((module_name, sha256_file(Path(raw_path))))
    ordered = tuple(sorted(components))
    return ExtendedImplementationReceipt(
        executor_id=EXTENDED_ABLATION_EXECUTOR_ID,
        components=ordered,
        source_manifest_sha256=_commit(
            [{"module": name, "sha256": digest} for name, digest in ordered]
        ),
    )


def _execution_input_sha256(
    spec: ExtendedAblationSpec,
    implementation: ExtendedImplementationReceipt,
) -> str:
    return _commit(
        {
            "implementation_receipt": implementation.to_mapping(),
            "schema_version": EXTENDED_ABLATION_EXECUTION_INPUT_SCHEMA_VERSION,
            "spec": spec.to_mapping(),
        }
    )


def extended_ablation_execution_input_sha256(spec: ExtendedAblationSpec) -> str:
    """Commit the complete cross-subsystem grid and live implementation."""
    if not isinstance(spec, ExtendedAblationSpec):
        _fail("spec must be an ExtendedAblationSpec")
    return _execution_input_sha256(spec, _implementation_receipt())


def _validate_study_protocol(
    protocol: StudyProtocol,
    spec: ExtendedAblationSpec,
) -> None:
    if not isinstance(protocol, StudyProtocol):
        _fail("extended ablations require a typed StudyProtocol")
    if protocol.mode is not StudyMode.SYNTHETIC:
        _fail("extended ablations accept only a synthetic study protocol")
    if protocol.claim_scope is not ClaimScope.SYNTHETIC_ONLY:
        _fail("extended ablations cannot authorize a non-synthetic claim")
    if protocol.study_id != spec.root_event_key.experiment_id:
        _fail("extended-ablation spec belongs to another study protocol")
    if (
        protocol.estimand.definition_id != EXTENDED_ABLATION_ESTIMAND_ID
        or protocol.estimand.definition_sha256
        != extended_ablation_definition_sha256(EXTENDED_ABLATION_ESTIMAND_ID)
    ):
        _fail("extended-ablation estimand differs from its frozen definition")
    if (
        protocol.baseline_policy.definition_id != EXTENDED_ABLATION_REFERENCE_POLICY_ID
        or protocol.baseline_policy.definition_sha256
        != extended_ablation_definition_sha256(EXTENDED_ABLATION_REFERENCE_POLICY_ID)
        or protocol.candidate_policy.definition_id
        != EXTENDED_ABLATION_VARIANT_POLICY_ID
        or protocol.candidate_policy.definition_sha256
        != extended_ablation_definition_sha256(EXTENDED_ABLATION_VARIANT_POLICY_ID)
    ):
        _fail("extended-ablation comparison-arm definitions changed")
    protocol_cohorts = {item.cohort_id: item for item in protocol.cohorts}
    if set(protocol_cohorts) != {EXTENDED_ABLATION_COHORT_ID}:
        _fail("extended-ablation cohort registry changed")
    cohort = protocol_cohorts[EXTENDED_ABLATION_COHORT_ID]
    if (
        cohort.definition_sha256
        != extended_ablation_definition_sha256(EXTENDED_ABLATION_COHORT_ID)
        or cohort.basis is not CohortBasis.LATENT_STRATEGY
    ):
        _fail("extended-ablation cohort semantics changed")
    protocol_horizons = {item.definition_id: item for item in protocol.horizons}
    if set(protocol_horizons) != {EXTENDED_ABLATION_HORIZON_ID} or (
        protocol_horizons[EXTENDED_ABLATION_HORIZON_ID].definition_sha256
        != extended_ablation_definition_sha256(EXTENDED_ABLATION_HORIZON_ID)
    ):
        _fail("extended-ablation horizon registry changed")
    protocol_metrics = {item.definition_id: item for item in protocol.metrics}
    if set(protocol_metrics) != set(EXTENDED_ABLATION_METRIC_IDS):
        _fail("extended-ablation metric registry changed")
    for metric_id in EXTENDED_ABLATION_METRIC_IDS:
        if protocol_metrics[
            metric_id
        ].definition_sha256 != extended_ablation_definition_sha256(metric_id):
            _fail("extended-ablation metric semantics changed")
    observed_outputs = tuple(
        item.to_mapping()
        for item in sorted(protocol.outputs, key=lambda item: item.output_id)
    )
    expected_outputs = tuple(
        item.to_mapping()
        for item in sorted(
            extended_ablation_registered_outputs(), key=lambda item: item.output_id
        )
    )
    if observed_outputs != expected_outputs:
        _fail("extended-ablation registered outputs differ from the frozen grid")
    if protocol.policy_evaluation_outcomes != (EXTENDED_ABLATION_OUTCOME_ID,):
        _fail("extended-ablation policy outcome registry changed")
    if protocol.randomness_protocol is not RandomnessProtocol.EVENT_KEYED_PAIRED:
        _fail("extended ablations require event-keyed paired randomness")
    registered_specs = tuple(
        item
        for item in protocol.baselines
        if item.definition_id == EXTENDED_ABLATION_SPEC_DEFINITION_ID
    )
    if (
        len(registered_specs) != 1
        or registered_specs[0].definition_sha256 != spec.sha256
    ):
        _fail("extended-ablation grid differs from its frozen protocol definition")
    if protocol.knowledge_cutoff != integrated_synthetic_market_config().cutoff:
        _fail("extended-ablation cutoff differs from the study protocol")


@dataclass(frozen=True, slots=True)
class ExtendedExecutionReceipt:
    """Generic cross-binding for two actually executed mechanism paths."""

    dimension: ExtendedAblationDimension
    reference_mechanism_id: str
    variant_mechanism_id: str
    source_input_sha256: str
    implementation_receipt_sha256: str
    event_keys: tuple[str, ...]
    reference_settings_sha256: str
    variant_settings_sha256: str
    reference_output_sha256: str
    variant_output_sha256: str
    contrast_sha256: str
    evidence_kind: str = "executed_contrast"
    synthetic_only: Literal[True] = True

    def __post_init__(self) -> None:
        if not isinstance(self.dimension, ExtendedAblationDimension):
            _fail("execution receipt dimension has the wrong type")
        if not self.reference_mechanism_id or not self.variant_mechanism_id:
            _fail("execution receipt mechanism IDs must be non-empty")
        for field in (
            "source_input_sha256",
            "implementation_receipt_sha256",
            "reference_settings_sha256",
            "variant_settings_sha256",
            "reference_output_sha256",
            "variant_output_sha256",
            "contrast_sha256",
        ):
            _sha256(getattr(self, field), field)
        if (
            not isinstance(self.event_keys, tuple)
            or not self.event_keys
            or len(set(self.event_keys)) != len(self.event_keys)
            or not all(
                isinstance(item, str)
                and (item.startswith("ptk1:") or item.startswith("pse1:"))
                for item in self.event_keys
            )
        ):
            _fail("execution receipt requires unique explicit EventKeys")
        if self.evidence_kind != "executed_contrast" or self.synthetic_only is not True:
            _fail("execution receipt scope cannot be widened")

    @property
    def sha256(self) -> str:
        return _commit(self.to_mapping())

    def to_mapping(self) -> dict[str, object]:
        return {
            "contrast_sha256": self.contrast_sha256,
            "dimension": self.dimension.value,
            "event_keys": list(self.event_keys),
            "evidence_kind": self.evidence_kind,
            "implementation_receipt_sha256": self.implementation_receipt_sha256,
            "reference_mechanism_id": self.reference_mechanism_id,
            "reference_output_sha256": self.reference_output_sha256,
            "reference_settings_sha256": self.reference_settings_sha256,
            "source_input_sha256": self.source_input_sha256,
            "synthetic_only": self.synthetic_only,
            "variant_mechanism_id": self.variant_mechanism_id,
            "variant_output_sha256": self.variant_output_sha256,
            "variant_settings_sha256": self.variant_settings_sha256,
        }


@dataclass(frozen=True, slots=True)
class ExpectedRejectionReceipt:
    """Semantic receipt for a deliberate firewall attack rejected before use."""

    implementation_receipt_sha256: str
    policy_sha256: str
    attack_input_sha256: str
    event_key: str
    exception_type: str
    rule_code: str
    normalized_audit_sha256: str
    rejected_before_use: Literal[True] = True
    synthetic_only: Literal[True] = True

    def __post_init__(self) -> None:
        for field in (
            "implementation_receipt_sha256",
            "policy_sha256",
            "attack_input_sha256",
            "normalized_audit_sha256",
        ):
            _sha256(getattr(self, field), field)
        if not self.event_key.startswith("ptk1:"):
            _fail("negative-control receipt requires an explicit EventKey")
        if self.exception_type != "PostCutoffDataError":
            _fail("negative control did not reject with PostCutoffDataError")
        if self.rule_code != RuleCode.POST_CUTOFF_AVAILABLE_TIME.value:
            _fail("negative control did not record the post-cutoff rule")
        if self.rejected_before_use is not True or self.synthetic_only is not True:
            _fail("negative-control scope cannot be widened")

    @property
    def sha256(self) -> str:
        return _commit(self.to_mapping())

    def to_mapping(self) -> dict[str, object]:
        return {
            "attack_input_sha256": self.attack_input_sha256,
            "event_key": self.event_key,
            "exception_type": self.exception_type,
            "implementation_receipt_sha256": self.implementation_receipt_sha256,
            "normalized_audit_sha256": self.normalized_audit_sha256,
            "policy_sha256": self.policy_sha256,
            "rejected_before_use": self.rejected_before_use,
            "rule_code": self.rule_code,
            "synthetic_only": self.synthetic_only,
        }


@dataclass(frozen=True, slots=True)
class ExtendedAblationRow:
    """One terminal dimension row, including honest blockers."""

    dimension: ExtendedAblationDimension
    status: ExtendedAblationStatus
    execution: ExtendedExecutionReceipt | None = None
    rejection: ExpectedRejectionReceipt | None = None
    blocker_code: str | None = None
    blocker_message: str | None = None
    failure: ExtendedExecutionFailure | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.dimension, ExtendedAblationDimension) or not isinstance(
            self.status, ExtendedAblationStatus
        ):
            _fail("extended ablation row has an invalid identity or status")
        if self.status is ExtendedAblationStatus.COMPLETED:
            if (
                not isinstance(self.execution, ExtendedExecutionReceipt)
                or self.execution.dimension is not self.dimension
                or self.rejection is not None
                or self.blocker_code is not None
                or self.blocker_message is not None
                or self.failure is not None
            ):
                _fail("completed ablation row lacks a matching execution receipt")
        elif self.status is ExtendedAblationStatus.EXPECTED_REJECTION:
            if (
                not isinstance(self.rejection, ExpectedRejectionReceipt)
                or self.dimension is not ExtendedAblationDimension.TEMPORAL_FIREWALL
                or self.execution is not None
                or self.blocker_code is not None
                or self.blocker_message is not None
                or self.failure is not None
            ):
                _fail("expected-rejection row lacks its firewall receipt")
        elif self.status is ExtendedAblationStatus.BLOCKED:
            if (
                self.execution is not None
                or self.rejection is not None
                or not self.blocker_code
                or not self.blocker_message
                or self.failure is not None
            ):
                _fail("blocked ablation row must retain a complete blocker")
        elif (
            self.execution is not None
            or self.rejection is not None
            or self.blocker_code is not None
            or self.blocker_message is not None
            or not isinstance(self.failure, ExtendedExecutionFailure)
        ):
            _fail("failed ablation row must retain only its execution failure")

    def to_mapping(self) -> dict[str, object]:
        return {
            "blocker_code": self.blocker_code,
            "blocker_message": self.blocker_message,
            "dimension": self.dimension.value,
            "execution": None
            if self.execution is None
            else self.execution.to_mapping(),
            "failure": None if self.failure is None else self.failure.to_mapping(),
            "rejection": None
            if self.rejection is None
            else self.rejection.to_mapping(),
            "status": self.status.value,
        }


@dataclass(frozen=True, slots=True)
class ExtendedExecutionFailure:
    """Machine-readable failure retained instead of omitting a dimension."""

    stage: str
    exception_type: str
    message: str

    def __post_init__(self) -> None:
        if not self.stage or not self.exception_type or not self.message:
            _fail("extended execution failure fields must be non-empty")

    def to_mapping(self) -> dict[str, object]:
        return {
            "exception_type": self.exception_type,
            "message": self.message,
            "stage": self.stage,
        }


def _behavior_representation_row(
    spec: ExtendedAblationSpec,
    implementation: ExtendedImplementationReceipt,
) -> ExtendedAblationRow:
    mechanism_spec = MechanismAblationSpec(
        root_event_key=spec.key("behavior-representation"),
        replicate_count=1,
        step_count=3,
        variants=(
            MechanismAblationVariant.HETEROGENEOUS_CONTINUOUS,
            MechanismAblationVariant.PROMPTED_LABELS_MOCK,
        ),
    )
    _, results = _execute_mechanism_ablation_grid(mechanism_spec)
    reference = _mechanism_result_for(
        results, MechanismAblationVariant.HETEROGENEOUS_CONTINUOUS, 0
    )
    prompted = _mechanism_result_for(
        results,
        MechanismAblationVariant.PROMPTED_LABELS_MOCK,
        0,
    )
    if (
        reference.status is not MechanismAblationStatus.COMPLETED
        or prompted.status is not MechanismAblationStatus.COMPLETED
        or reference.receipt is None
        or prompted.receipt is None
    ):
        raise ExtendedAblationIntegrityError(
            "behavior-representation mechanism execution failed"
        )
    return ExtendedAblationRow(
        dimension=ExtendedAblationDimension.BEHAVIOR_REPRESENTATION,
        status=ExtendedAblationStatus.COMPLETED,
        execution=ExtendedExecutionReceipt(
            dimension=ExtendedAblationDimension.BEHAVIOR_REPRESENTATION,
            reference_mechanism_id="synthetic_continuous_uncalibrated",
            variant_mechanism_id="deterministic_mock_prompted_labels",
            source_input_sha256=mechanism_spec.sha256,
            implementation_receipt_sha256=implementation.sha256,
            event_keys=(reference.event_key.value,),
            reference_settings_sha256=(reference.receipt.mechanism_settings.sha256),
            variant_settings_sha256=prompted.receipt.mechanism_settings.sha256,
            reference_output_sha256=reference.receipt.paired_outcome_sha256,
            variant_output_sha256=prompted.receipt.paired_outcome_sha256,
            contrast_sha256=_commit(
                [
                    item.to_mapping()
                    for item in _mechanism_metric_contrast(
                        results, MechanismAblationVariant.PROMPTED_LABELS_MOCK, 0
                    )
                ]
            ),
        ),
    )


def _paired_effect_mapping(
    parameters: dict[str, float], event_key: EventKey
) -> dict[str, object]:
    pair = integrated_synthetic_pair(
        event_key=event_key,
        parameters=parameters,
        config=integrated_synthetic_market_config(),
        candidate_allow_buy=False,
        step_count=3,
        include_all_groups=True,
    )
    outcome = pair.run(
        IntegratedSyntheticWorldFactory(integrated_synthetic_market_config())
    )
    return {
        "event_key": event_key.value,
        "outcome_sha256": outcome.sha256,
        "parameters": dict(sorted(parameters.items())),
        "paired_differences": [item.to_mapping() for item in outcome.differences],
    }


def _calibration_row(
    spec: ExtendedAblationSpec,
    implementation: ExtendedImplementationReceipt,
) -> ExtendedAblationRow:
    adapter = IntegratedSyntheticCalibrationAdapter(step_count=2)
    config = integrated_synthetic_calibration_config(
        seed=f"{spec.root_event_key.seed}:extended-calibration",
        simulation_count=spec.calibration_simulations,
        accepted_particles=3,
        posterior_draws=spec.posterior_draws,
        ppc_draws=2,
        sbc_replicates=1,
        sbc_posterior_draws=2,
    )
    truth = {"behavior_scale": 0.7, "signal_scale": 1.25}
    observed_key = spec.key("calibration-observed")
    heldout_key = spec.key("calibration-heldout", 1)
    observed = synthetic_calibration_input(
        config,
        adapter.simulate(truth, event_key=observed_key),
        adapter.simulate(truth, event_key=heldout_key),
        calibration_generation_event_key=observed_key.value,
        held_out_generation_event_key=heldout_key.value,
        record_id="extended-ablation-synthetic-observation",
    )
    fit = ABCPosteriorEstimator().fit(
        config=config,
        observed=observed,
        simulator=adapter,
        namespace="extended-ablation",
    )
    posterior, posterior_keys = sample_posterior(
        config=config,
        fit=fit,
        namespace="extended-ablation",
    )
    point_parameters = dict(fit.particles[0].parameters)
    evaluation_keys = tuple(
        spec.key("calibration-policy-evaluation", index + 10)
        for index in range(spec.posterior_draws)
    )
    posterior_outputs = tuple(
        _paired_effect_mapping(dict(draw.parameters), key)
        for draw, key in zip(posterior.draws, evaluation_keys, strict=True)
    )
    point_outputs = tuple(
        _paired_effect_mapping(point_parameters, key) for key in evaluation_keys
    )
    source = {
        "config": config.to_mapping(),
        "fit": {
            "backend_id": fit.backend_id,
            "event_keys": list(fit.event_keys),
            "particles": [item.to_mapping() for item in fit.particles],
        },
        "input": observed.to_mapping(),
        "posterior": posterior.to_mapping(),
    }
    return ExtendedAblationRow(
        dimension=ExtendedAblationDimension.CALIBRATION_PROPAGATION,
        status=ExtendedAblationStatus.COMPLETED,
        execution=ExtendedExecutionReceipt(
            dimension=ExtendedAblationDimension.CALIBRATION_PROPAGATION,
            reference_mechanism_id="rejection_abc_posterior_propagation",
            variant_mechanism_id="best_distance_abc_point_propagation",
            source_input_sha256=_commit(source),
            implementation_receipt_sha256=implementation.sha256,
            event_keys=(
                observed_key.value,
                heldout_key.value,
                *fit.event_keys,
                *posterior_keys,
                *(item.value for item in evaluation_keys),
            ),
            reference_settings_sha256=posterior.sha256,
            variant_settings_sha256=_commit(
                {
                    "particle": fit.particles[0].to_mapping(),
                    "selection_rule": "minimum_abc_distance_then_canonical_tie_break",
                }
            ),
            reference_output_sha256=_commit(posterior_outputs),
            variant_output_sha256=_commit(point_outputs),
            contrast_sha256=_commit(
                {
                    "point_outputs": point_outputs,
                    "posterior_outputs": posterior_outputs,
                }
            ),
        ),
    )


def _structural_row(
    spec: ExtendedAblationSpec,
    implementation: ExtendedImplementationReceipt,
) -> ExtendedAblationRow:
    protocol_sha256 = _commit({"protocol": spec.sha256})
    cohort_sha256 = _commit({"cohort": "pre-policy-synthetic-all"})
    start = datetime(2025, 2, 1, tzinfo=UTC)
    input_bundle = BaselineInputBundle(
        scenario_id="extended-ablation-structural",
        study_protocol_sha256=protocol_sha256,
        data_manifest_sha256=_commit({"source": "synthetic-pre-policy"}),
        cohort_sha256=cohort_sha256,
        horizon_sha256=_commit({"horizon": "synthetic-terminal"}),
        outcome_definition_sha256=_commit({"outcome": "paired-effect"}),
        frozen_at=start - timedelta(days=1),
        target_outcomes_first_available_at=start,
        event_keys=tuple(
            SemanticEventKey(
                scenario_id="extended-ablation-structural",
                event_type=EventType.EXOGENOUS_SHOCK,
                entity_id=f"draw-{index}",
                timestamp=start + timedelta(minutes=index),
            )
            for index in range(spec.structural_draws)
        ),
        prepolicy_features=(("depth", 10.0), ("spread", 1.5)),
        prepolicy_residuals=(-0.5, 0.0, 0.5),
        synthetic_only=True,
    )
    cell = StructuralOutcomeCell(
        study_protocol_sha256=protocol_sha256,
        baseline_policy_sha256=_commit({"policy": "baseline"}),
        candidate_policy_sha256=_commit({"policy": "candidate"}),
        estimand_id="candidate_minus_baseline",
        outcome_id="synthetic_paired_effect",
        cohort_id="pre_policy_synthetic_all",
        cohort_sha256=cohort_sha256,
        horizon_id="synthetic_terminal",
    )
    full = FullPolicyTwinSyntheticBaseline(
        seed=f"{spec.root_event_key.seed}:agent-structure",
        posterior_artifact_sha256=_commit({"posterior": "synthetic-fixture"}),
    )
    orders = OrderLevelGeneratorBaseline(
        seed=f"{spec.root_event_key.seed}:order-structure",
        generator_checkpoint_sha256=_commit({"checkpoint": "synthetic-fixture"}),
    )
    agent_generator = AgentInteractionStructuralGenerator("agents", full)
    order_generator = OrderFlowStructuralGenerator("orders", orders)
    executions = (
        agent_generator.generate(input_bundle, cell),
        order_generator.generate(input_bundle, cell),
    )
    ensemble_spec = StructuralEnsembleSpec(
        study_protocol_sha256=protocol_sha256,
        members=(agent_generator.member, order_generator.member),
        weighting=EnsembleWeighting.EQUAL,
        weights_frozen_at=start - timedelta(days=1),
        target_outcomes_available_at=start,
        quantile_levels=(0.25, 0.5, 0.75),
        interval_level=0.5,
    )
    ensemble = evaluate_generated_structural_ensemble(ensemble_spec, cell, executions)
    single = executions[0]
    return ExtendedAblationRow(
        dimension=ExtendedAblationDimension.MODEL_UNCERTAINTY,
        status=ExtendedAblationStatus.COMPLETED,
        execution=ExtendedExecutionReceipt(
            dimension=ExtendedAblationDimension.MODEL_UNCERTAINTY,
            reference_mechanism_id="complete_equal_weight_structural_ensemble",
            variant_mechanism_id="single_agent_interaction_structure",
            source_input_sha256=input_bundle.sha256,
            implementation_receipt_sha256=implementation.sha256,
            event_keys=tuple(item.value for item in input_bundle.event_keys),
            reference_settings_sha256=ensemble_spec.sha256,
            variant_settings_sha256=agent_generator.member.sha256,
            reference_output_sha256=ensemble.sha256,
            variant_output_sha256=single.result.sha256,
            contrast_sha256=_commit(
                {
                    "ensemble": ensemble.to_mapping(),
                    "member_execution_receipts": [
                        item.receipt.to_mapping() for item in executions
                    ],
                    "single": single.result.to_mapping(),
                }
            ),
        ),
    )


def _abstention_row(
    spec: ExtendedAblationSpec,
    implementation: ExtendedImplementationReceipt,
) -> ExtendedAblationRow:
    target_available = datetime(2025, 2, 1, tzinfo=UTC)
    thresholds_frozen_at = target_available - timedelta(days=1)
    uncertainty_thresholds = (0.1, 0.2, 0.4)
    study_protocol_sha256 = _commit({"protocol": spec.sha256})
    threshold_source_artifact_sha256 = _commit(
        {"source": "frozen-synthetic-uncertainty-thresholds"}
    )
    rule = SelectionRule(
        schema_version=2,
        rule_id="extended_synthetic_abstention",
        study_protocol_sha256=study_protocol_sha256,
        threshold_source=ThresholdSource.SYNTHETIC_FIXTURE,
        threshold_source_artifact_sha256=threshold_source_artifact_sha256,
        thresholds_frozen_at=thresholds_frozen_at,
        target_outcomes_first_available_at=target_available,
        uncertainty_thresholds=uncertainty_thresholds,
        loss_metric_id="absolute_paired_effect_error",
        minimum_retained=2,
        synthetic_only=True,
        threshold_source_receipt=SyntheticThresholdSourceReceipt.issue(
            study_protocol_sha256=study_protocol_sha256,
            source_artifact_sha256=threshold_source_artifact_sha256,
            thresholds_frozen_at=thresholds_frozen_at,
            target_outcomes_first_available_at=target_available,
            uncertainty_thresholds=uncertainty_thresholds,
        ),
    )
    values = (
        ("prediction-1", 0.05, 0.5),
        ("prediction-2", 0.1, 1.0),
        ("prediction-3", 0.2, 2.0),
        ("prediction-4", 0.4, 4.0),
    )
    records = tuple(
        SelectivePredictionRecord(identifier, uncertainty, loss)
        for identifier, uncertainty, loss in values
    )
    event_keys = tuple(
        spec.key(f"abstention-{identifier}", index + 30)
        for index, (identifier, _, _) in enumerate(values)
    )
    record_source = tuple(
        {
            "event_key": event_key.value,
            "loss": record.loss,
            "prediction_id": record.prediction_id,
            "uncertainty_score": record.uncertainty_score,
        }
        for record, event_key in zip(records, event_keys, strict=True)
    )
    with_abstention = evaluate_selective_predictions(records, rule)
    without_abstention = {
        "coverage": 1.0,
        "mean_loss": fmean(item.loss for item in records),
        "mechanism": "retain_all_without_abstention",
        "record_manifest_sha256": _commit(record_source),
        "retained_count": len(records),
        "total_count": len(records),
    }
    return ExtendedAblationRow(
        dimension=ExtendedAblationDimension.ABSTENTION,
        status=ExtendedAblationStatus.COMPLETED,
        execution=ExtendedExecutionReceipt(
            dimension=ExtendedAblationDimension.ABSTENTION,
            reference_mechanism_id="frozen_threshold_selective_prediction",
            variant_mechanism_id="retain_all_without_abstention",
            source_input_sha256=_commit(record_source),
            implementation_receipt_sha256=implementation.sha256,
            event_keys=tuple(item.value for item in event_keys),
            reference_settings_sha256=rule.sha256,
            variant_settings_sha256=_commit(
                {"mechanism": "retain_all", "threshold": None}
            ),
            reference_output_sha256=with_abstention.sha256,
            variant_output_sha256=_commit(without_abstention),
            contrast_sha256=_commit(
                {
                    "with_abstention": with_abstention.to_mapping(),
                    "without_abstention": without_abstention,
                }
            ),
        ),
    )


def _llm_row(
    spec: ExtendedAblationSpec,
    implementation: ExtendedImplementationReceipt,
) -> ExtendedAblationRow:
    """Execute an offline mock/no-LLM contrast over one exact paired input."""
    event_key = spec.key("integrated-mock-llm", 35)
    no_llm_config = integrated_synthetic_market_config()
    mock_config = integrated_synthetic_mock_market_config()
    shared_parameters = {"behavior_scale": 0.8, "signal_scale": 1.1}
    no_llm_pair = integrated_synthetic_pair(
        event_key=event_key,
        parameters=shared_parameters,
        config=no_llm_config,
        candidate_is_noop=True,
        step_count=3,
        include_all_groups=True,
    )
    mock_pair = integrated_synthetic_pair(
        event_key=event_key,
        parameters=shared_parameters,
        config=mock_config,
        candidate_is_noop=True,
        step_count=3,
        include_all_groups=True,
    )
    no_llm_pair_input_sha256 = integrated_synthetic_pair_input_sha256(no_llm_pair)
    mock_pair_input_sha256 = integrated_synthetic_pair_input_sha256(mock_pair)
    no_llm_factory = IntegratedSyntheticWorldFactory(no_llm_config)
    mock_factory = IntegratedSyntheticWorldFactory(mock_config)
    no_llm_outcome = no_llm_pair.run(
        no_llm_factory,
        parallel=True,
    )
    mock_outcome = mock_pair.run(
        mock_factory,
        parallel=True,
        invocation_order=(
            no_llm_outcome.candidate.role,
            no_llm_outcome.baseline.role,
        ),
    )
    if (
        not no_llm_outcome.integrity_audit.passed
        or not mock_outcome.integrity_audit.passed
    ):
        raise ExtendedAblationIntegrityError(
            "integrated mock/no-LLM paired integrity audit failed"
        )

    def artifacts(
        factory: IntegratedSyntheticWorldFactory,
        baseline_role: WorldRole,
        candidate_role: WorldRole,
    ) -> tuple[IntegratedSyntheticRunArtifact, IntegratedSyntheticRunArtifact]:
        baseline = factory.runners[baseline_role].artifact
        candidate = factory.runners[candidate_role].artifact
        if baseline is None or candidate is None:
            raise ExtendedAblationIntegrityError(
                "integrated mock/no-LLM run omitted a world artifact"
            )
        return baseline, candidate

    no_llm_baseline, no_llm_candidate = artifacts(
        no_llm_factory,
        no_llm_outcome.baseline.role,
        no_llm_outcome.candidate.role,
    )
    mock_baseline, mock_candidate = artifacts(
        mock_factory,
        mock_outcome.baseline.role,
        mock_outcome.candidate.role,
    )
    no_llm_artifacts = (no_llm_baseline, no_llm_candidate)
    mock_artifacts = (mock_baseline, mock_candidate)

    def decision_keys(artifact: IntegratedSyntheticRunArtifact) -> tuple[str, ...]:
        return tuple(
            decision.event_key for step in artifact.steps for decision in step.decisions
        )

    no_llm_keys = decision_keys(no_llm_baseline)
    mock_keys = decision_keys(mock_baseline)
    if (
        no_llm_keys != decision_keys(no_llm_candidate)
        or no_llm_keys != mock_keys
        or mock_keys != decision_keys(mock_candidate)
        or len(set(mock_keys)) != len(mock_keys)
    ):
        raise ExtendedAblationIntegrityError(
            "mock/no-LLM variants did not retain identical unique decision EventKeys"
        )
    if any(
        decision.mock_adapter_id is not None
        for artifact in no_llm_artifacts
        for step in artifact.steps
        for decision in step.decisions
    ):
        raise ExtendedAblationIntegrityError(
            "no-LLM variant unexpectedly retained mock planning evidence"
        )
    hook = mock_config.mock_planning_hook
    if hook is None or not all(
        decision.mock_adapter_id == hook.adapter.adapter_id
        and decision.mock_authorization_sha256 == hook.authorization.sha256
        and decision.mock_request_sha256 is not None
        and decision.mock_rationale_code == hook.adapter.rationale_code
        for artifact in mock_artifacts
        for step in artifact.steps
        for decision in step.decisions
    ):
        raise ExtendedAblationIntegrityError(
            "mock variant omitted complete knowledge-firewall decision evidence"
        )
    if (
        tuple(item.to_mapping() for item in no_llm_outcome.baseline.exogenous_shocks)
        != tuple(item.to_mapping() for item in mock_outcome.baseline.exogenous_shocks)
        or no_llm_outcome.baseline.shock_tape_sha256
        != mock_outcome.baseline.shock_tape_sha256
    ):
        raise ExtendedAblationIntegrityError(
            "mock/no-LLM variants did not consume the same exogenous tape"
        )
    if (
        no_llm_pair.posterior_draw.parameters != mock_pair.posterior_draw.parameters
        or no_llm_pair.posterior_draw.parameter_sha256
        != mock_pair.posterior_draw.parameter_sha256
        or no_llm_pair.initial_state != mock_pair.initial_state
        or no_llm_pair.shock_transition_config != mock_pair.shock_transition_config
        or no_llm_pair.baseline_policy != mock_pair.baseline_policy
        or no_llm_pair.candidate_policy != mock_pair.candidate_policy
        or no_llm_pair.horizons != mock_pair.horizons
        or no_llm_pair.groups != mock_pair.groups
        or no_llm_pair.numeric_tolerance != mock_pair.numeric_tolerance
    ):
        raise ExtendedAblationIntegrityError(
            "mock/no-LLM variants changed a shared scientific input"
        )
    if (
        no_llm_pair.posterior_draw.model_draw_sha256
        == mock_pair.posterior_draw.model_draw_sha256
        or no_llm_baseline.config_sha256 != no_llm_config.sha256
        or no_llm_candidate.config_sha256 != no_llm_config.sha256
        or mock_baseline.config_sha256 != mock_config.sha256
        or mock_candidate.config_sha256 != mock_config.sha256
    ):
        raise ExtendedAblationIntegrityError(
            "mock/no-LLM model draw or run artifact is not config-bound"
        )

    shared_scientific_input = {
        "baseline_policy": no_llm_pair.baseline_policy.to_mapping(),
        "candidate_policy": no_llm_pair.candidate_policy.to_mapping(),
        "groups": [item.to_mapping() for item in no_llm_pair.groups],
        "horizons": [item.to_mapping() for item in no_llm_pair.horizons],
        "initial_state": no_llm_pair.initial_state.to_mapping(),
        "numeric_tolerance": no_llm_pair.numeric_tolerance,
        "parameters": list(no_llm_pair.posterior_draw.parameters),
        "shock_tape": no_llm_pair.shock_tape.to_mapping(),
        "shock_transition_config": (no_llm_pair.shock_transition_config.to_mapping()),
    }
    shared_scientific_input_sha256 = _commit(shared_scientific_input)

    no_llm_output = {
        "paired_outcome": no_llm_outcome.to_mapping(),
        "world_artifacts": [item.to_mapping() for item in no_llm_artifacts],
    }
    mock_output = {
        "paired_outcome": mock_outcome.to_mapping(),
        "world_artifacts": [item.to_mapping() for item in mock_artifacts],
    }
    if _commit(no_llm_output) == _commit(mock_output):
        raise ExtendedAblationIntegrityError(
            "integrated deterministic mock produced no executable contrast"
        )
    return ExtendedAblationRow(
        dimension=ExtendedAblationDimension.LLM,
        status=ExtendedAblationStatus.COMPLETED,
        execution=ExtendedExecutionReceipt(
            dimension=ExtendedAblationDimension.LLM,
            reference_mechanism_id=(
                "knowledge_firewall_authorized_deterministic_mock_planner"
            ),
            variant_mechanism_id="structural_rule_no_llm_planner",
            source_input_sha256=shared_scientific_input_sha256,
            implementation_receipt_sha256=implementation.sha256,
            event_keys=(event_key.value, *mock_keys),
            reference_settings_sha256=mock_config.sha256,
            variant_settings_sha256=no_llm_config.sha256,
            reference_output_sha256=_commit(mock_output),
            variant_output_sha256=_commit(no_llm_output),
            contrast_sha256=_commit(
                {
                    "claim_scope": "deterministic_mock_synthetic_only",
                    "mock": mock_output,
                    "no_llm": no_llm_output,
                    "mock_model_draw_sha256": (
                        mock_pair.posterior_draw.model_draw_sha256
                    ),
                    "mock_pair_input_sha256": mock_pair_input_sha256,
                    "no_llm_model_draw_sha256": (
                        no_llm_pair.posterior_draw.model_draw_sha256
                    ),
                    "no_llm_pair_input_sha256": no_llm_pair_input_sha256,
                    "shared_scientific_input": shared_scientific_input,
                    "shared_scientific_input_sha256": (shared_scientific_input_sha256),
                    "shared_decision_event_keys": list(mock_keys),
                }
            ),
        ),
    )


def _firewall_row(
    spec: ExtendedAblationSpec,
    implementation: ExtendedImplementationReceipt,
) -> ExtendedAblationRow:
    policy = CutoffPolicy(
        schema_version=1,
        cutoff=datetime(2025, 1, 31, tzinfo=UTC),
        training_end=datetime(2025, 1, 10, tzinfo=UTC),
        validation_end=datetime(2025, 1, 20, tzinfo=UTC),
        policy_start=datetime(2025, 1, 25, tzinfo=UTC),
        allowed_license_tags=("synthetic-test",),
    )
    canary = make_post_cutoff_canary(
        policy,
        b"extended-ablation-do-not-read-post-cutoff",
        license_tag="synthetic-test",
    )
    with TemporaryDirectory(prefix="policytwin-firewall-ablation-") as directory:
        firewall = TemporalFirewall(
            policy,
            TemporalAuditLog(Path(directory) / "temporal-audit.jsonl"),
        )
        try:
            firewall.validate_record(canary)
        except PostCutoffDataError as error:
            exception_type = type(error).__qualname__
        else:
            raise ExtendedAblationIntegrityError(
                "post-cutoff negative control was unexpectedly accepted"
            )
        events = firewall.audit_log.events()
    if len(events) != 1:
        raise ExtendedAblationIntegrityError(
            "post-cutoff negative control did not emit exactly one audit decision"
        )
    event = events[0]
    normalized = {
        key: value
        for key, value in event.items()
        if key not in {"occurred_at", "event_sha256", "previous_event_sha256"}
    }
    return ExtendedAblationRow(
        dimension=ExtendedAblationDimension.TEMPORAL_FIREWALL,
        status=ExtendedAblationStatus.EXPECTED_REJECTION,
        rejection=ExpectedRejectionReceipt(
            implementation_receipt_sha256=implementation.sha256,
            policy_sha256=policy.sha256,
            attack_input_sha256=_commit(
                {
                    "event_key": spec.key("firewall-attack", 40).value,
                    "record": canary.to_mapping(),
                }
            ),
            event_key=spec.key("firewall-attack", 40).value,
            exception_type=exception_type,
            rule_code=str(event["rule_code"]),
            normalized_audit_sha256=_commit(normalized),
        ),
    )


@dataclass(frozen=True, slots=True)
class ExtendedAblationArtifact:
    """Complete terminal registry for all cross-subsystem dimensions."""

    protocol_execution_receipt: ProtocolExecutionReceipt
    protocol_canonical_json: str
    execution_input_sha256: str
    spec: ExtendedAblationSpec
    implementation_receipt: ExtendedImplementationReceipt
    rows: tuple[ExtendedAblationRow, ...]
    completed_count: int
    expected_rejection_count: int
    blocked_count: int
    failure_count: int
    claim_scope: str = EXTENDED_ABLATION_CLAIM_SCOPE
    synthetic_only: Literal[True] = True
    schema_version: str = EXTENDED_ABLATION_ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        receipt = self.protocol_execution_receipt
        if not isinstance(receipt, ProtocolExecutionReceipt):
            _fail("extended artifact requires a protocol execution receipt")
        if (
            receipt.consumer is not ProtocolExecutionConsumer.EXTENDED_ABLATION_WORKFLOW
            or receipt.purpose
            is not ProtocolExecutionPurpose.SYNTHETIC_EXTENDED_ABLATION
        ):
            _fail(
                "extended artifact protocol receipt has the wrong consumer or purpose"
            )
        try:
            protocol_value = json.loads(self.protocol_canonical_json)
        except (json.JSONDecodeError, TypeError) as error:
            raise ExtendedAblationSchemaError(
                "extended artifact study protocol must be canonical JSON"
            ) from error
        if not isinstance(protocol_value, dict) or not all(
            isinstance(key, str) for key in protocol_value
        ):
            _fail("extended artifact study protocol must be a JSON object")
        if (
            canonical_json_bytes(protocol_value).decode("utf-8")
            != self.protocol_canonical_json
            or sha256_bytes(self.protocol_canonical_json.encode("utf-8"))
            != receipt.protocol_sha256
        ):
            _fail("extended artifact protocol does not match its execution receipt")
        if not isinstance(self.spec, ExtendedAblationSpec) or not isinstance(
            self.implementation_receipt, ExtendedImplementationReceipt
        ):
            _fail("extended artifact has invalid typed roots")
        protocol = StudyProtocol.from_mapping(protocol_value)
        validate_protocol_execution_receipt_binding(protocol, receipt)
        _validate_study_protocol(protocol, self.spec)
        expected_input = _execution_input_sha256(self.spec, self.implementation_receipt)
        _sha256(self.execution_input_sha256, "execution_input_sha256")
        if (
            self.execution_input_sha256 != expected_input
            or receipt.config_sha256 != self.spec.sha256
            or receipt.input_sha256 != expected_input
        ):
            _fail("extended artifact receipt does not bind the complete grid input")
        if (
            not isinstance(self.rows, tuple)
            or not all(isinstance(item, ExtendedAblationRow) for item in self.rows)
            or self.rows
            != tuple(
                sorted(self.rows, key=lambda item: _DIMENSION_ORDER[item.dimension])
            )
            or {item.dimension for item in self.rows} != set(_DIMENSIONS)
            or len(self.rows) != len(_DIMENSIONS)
        ):
            _fail("extended artifact must retain each dimension exactly once")
        observed_cells = tuple(
            (
                EXTENDED_ABLATION_HORIZON_ID,
                EXTENDED_ABLATION_COHORT_ID,
                _METRIC_BY_DIMENSION[row.dimension],
            )
            for row in self.rows
        )
        if observed_cells != extended_ablation_registered_output_cells():
            _fail("extended rows differ from the frozen protocol output registry")
        counts = {
            status: sum(item.status is status for item in self.rows)
            for status in ExtendedAblationStatus
        }
        if (
            self.completed_count != counts[ExtendedAblationStatus.COMPLETED]
            or self.expected_rejection_count
            != counts[ExtendedAblationStatus.EXPECTED_REJECTION]
            or self.blocked_count != counts[ExtendedAblationStatus.BLOCKED]
            or self.failure_count != counts[ExtendedAblationStatus.FAILED]
        ):
            _fail("extended artifact terminal counts do not match its rows")
        for row in self.rows:
            if row.execution is not None:
                expected_mechanism_ids = _EXECUTED_MECHANISM_IDS.get(row.dimension)
                if (
                    expected_mechanism_ids is None
                    or (
                        row.execution.reference_mechanism_id,
                        row.execution.variant_mechanism_id,
                    )
                    != expected_mechanism_ids
                ):
                    _fail(
                        "extended row mechanism IDs differ from the verifier-owned "
                        "dimension registry"
                    )
            receipt_hash = (
                row.execution.implementation_receipt_sha256
                if row.execution is not None
                else (
                    row.rejection.implementation_receipt_sha256
                    if row.rejection is not None
                    else None
                )
            )
            if (
                receipt_hash is not None
                and receipt_hash != self.implementation_receipt.sha256
            ):
                _fail("extended row is bound to another implementation receipt")
        live_implementation = _implementation_receipt()
        if self.implementation_receipt.sha256 != live_implementation.sha256:
            _fail("extended artifact implementation differs from the live verifier")
        replayed_rows = _execute_extended_ablation_rows(self.spec, live_implementation)
        if tuple(item.to_mapping() for item in self.rows) != tuple(
            item.to_mapping() for item in replayed_rows
        ):
            _fail("extended rows do not reproduce every registered receipt commitment")
        if self.claim_scope != EXTENDED_ABLATION_CLAIM_SCOPE:
            _fail("extended artifact claim scope changed")
        if (
            self.synthetic_only is not True
            or self.schema_version != EXTENDED_ABLATION_ARTIFACT_SCHEMA_VERSION
        ):
            _fail("extended artifact synthetic/schema contract changed")

    def row_for(self, dimension: ExtendedAblationDimension) -> ExtendedAblationRow:
        for row in self.rows:
            if row.dimension is dimension:
                return row
        raise KeyError(dimension)

    def to_mapping(self) -> dict[str, object]:
        return {
            "blocked_count": self.blocked_count,
            "claim_scope": self.claim_scope,
            "completed_count": self.completed_count,
            "execution_input_sha256": self.execution_input_sha256,
            "expected_rejection_count": self.expected_rejection_count,
            "failure_count": self.failure_count,
            "implementation_receipt": self.implementation_receipt.to_mapping(),
            "protocol": json.loads(self.protocol_canonical_json),
            "protocol_execution_receipt": (
                self.protocol_execution_receipt.to_mapping()
            ),
            "rows": [item.to_mapping() for item in self.rows],
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


def _capture_execution(
    dimension: ExtendedAblationDimension,
    runner: Callable[[], ExtendedAblationRow],
) -> ExtendedAblationRow:
    try:
        row = runner()
    except Exception as error:
        return ExtendedAblationRow(
            dimension=dimension,
            status=ExtendedAblationStatus.FAILED,
            failure=ExtendedExecutionFailure(
                stage=f"execute_{dimension.value}",
                exception_type=type(error).__qualname__,
                message=str(error) or type(error).__qualname__,
            ),
        )
    if row.dimension is not dimension:
        raise ExtendedAblationIntegrityError(
            "extended ablation runner returned another dimension"
        )
    return row


def _execute_extended_ablation_rows(
    spec: ExtendedAblationSpec,
    implementation: ExtendedImplementationReceipt,
) -> tuple[ExtendedAblationRow, ...]:
    """Execute rows without issuing or consuming study-protocol authority."""
    if not isinstance(spec, ExtendedAblationSpec):
        _fail("spec must be an ExtendedAblationSpec")
    return tuple(
        sorted(
            (
                _capture_execution(
                    ExtendedAblationDimension.BEHAVIOR_REPRESENTATION,
                    lambda: _behavior_representation_row(spec, implementation),
                ),
                _capture_execution(
                    ExtendedAblationDimension.CALIBRATION_PROPAGATION,
                    lambda: _calibration_row(spec, implementation),
                ),
                _capture_execution(
                    ExtendedAblationDimension.MODEL_UNCERTAINTY,
                    lambda: _structural_row(spec, implementation),
                ),
                _capture_execution(
                    ExtendedAblationDimension.ABSTENTION,
                    lambda: _abstention_row(spec, implementation),
                ),
                _capture_execution(
                    ExtendedAblationDimension.LLM,
                    lambda: _llm_row(spec, implementation),
                ),
                _capture_execution(
                    ExtendedAblationDimension.TEMPORAL_FIREWALL,
                    lambda: _firewall_row(spec, implementation),
                ),
            ),
            key=lambda item: _DIMENSION_ORDER[item.dimension],
        )
    )


def _artifact_from_verified_receipt(
    *,
    spec: ExtendedAblationSpec,
    implementation: ExtendedImplementationReceipt,
    rows: tuple[ExtendedAblationRow, ...],
    protocol_execution_receipt: ProtocolExecutionReceipt,
    protocol_canonical_json: str,
) -> ExtendedAblationArtifact:
    return ExtendedAblationArtifact(
        protocol_execution_receipt=protocol_execution_receipt,
        protocol_canonical_json=protocol_canonical_json,
        execution_input_sha256=_execution_input_sha256(spec, implementation),
        spec=spec,
        implementation_receipt=implementation,
        rows=rows,
        completed_count=sum(
            item.status is ExtendedAblationStatus.COMPLETED for item in rows
        ),
        expected_rejection_count=sum(
            item.status is ExtendedAblationStatus.EXPECTED_REJECTION for item in rows
        ),
        blocked_count=sum(
            item.status is ExtendedAblationStatus.BLOCKED for item in rows
        ),
        failure_count=sum(
            item.status is ExtendedAblationStatus.FAILED for item in rows
        ),
    )


def run_extended_ablations(
    *,
    spec: ExtendedAblationSpec,
    study_protocol: StudyProtocol,
    protocol_authorization: ProtocolExecutionAuthorization,
    execution_started_at: datetime,
) -> ExtendedAblationArtifact:
    """Consume one exact authorization before executing the complete grid."""
    if not isinstance(spec, ExtendedAblationSpec):
        _fail("spec must be an ExtendedAblationSpec")
    implementation = _implementation_receipt()
    execution_input_sha256 = _execution_input_sha256(spec, implementation)
    receipt = consume_protocol_execution_authorization(
        protocol_authorization,
        protocol=study_protocol,
        consumer=ProtocolExecutionConsumer.EXTENDED_ABLATION_WORKFLOW,
        purpose=ProtocolExecutionPurpose.SYNTHETIC_EXTENDED_ABLATION,
        config_sha256=spec.sha256,
        input_sha256=execution_input_sha256,
        execution_started_at=execution_started_at,
    )
    _validate_study_protocol(study_protocol, spec)
    # Authorization is consumed before any nested mechanism or subsystem row.
    rows = _execute_extended_ablation_rows(spec, implementation)
    return _artifact_from_verified_receipt(
        spec=spec,
        implementation=implementation,
        rows=rows,
        protocol_execution_receipt=receipt,
        protocol_canonical_json=canonical_json_bytes(
            study_protocol.to_mapping()
        ).decode("utf-8"),
    )


@dataclass(frozen=True, slots=True)
class ExtendedAblationAudit:
    artifact_sha256: str
    protocol_execution_receipt_sha256: str
    replayed_row_count: int
    passed: Literal[True] = True

    def __post_init__(self) -> None:
        _sha256(self.artifact_sha256, "artifact_sha256")
        _sha256(
            self.protocol_execution_receipt_sha256,
            "protocol_execution_receipt_sha256",
        )
        if self.replayed_row_count != len(_DIMENSIONS) or self.passed is not True:
            _fail("extended ablation audit must cover the complete registry")


def audit_extended_ablation_artifact(
    artifact: ExtendedAblationArtifact,
) -> ExtendedAblationAudit:
    """Replay every row and reject altered results, blockers, or source code."""
    if not isinstance(artifact, ExtendedAblationArtifact):
        _fail("artifact must be an ExtendedAblationArtifact")
    implementation = _implementation_receipt()
    rows = _execute_extended_ablation_rows(artifact.spec, implementation)
    replay = _artifact_from_verified_receipt(
        spec=artifact.spec,
        implementation=implementation,
        rows=rows,
        protocol_execution_receipt=artifact.protocol_execution_receipt,
        protocol_canonical_json=artifact.protocol_canonical_json,
    )
    if replay.canonical_bytes != artifact.canonical_bytes:
        raise ExtendedAblationIntegrityError(
            "extended-ablation artifact does not reproduce from trusted inputs"
        )
    return ExtendedAblationAudit(
        artifact_sha256=artifact.sha256,
        protocol_execution_receipt_sha256=(artifact.protocol_execution_receipt.sha256),
        replayed_row_count=len(artifact.rows),
    )


__all__ = [
    "EXTENDED_ABLATION_ARTIFACT_SCHEMA_VERSION",
    "EXTENDED_ABLATION_CLAIM_SCOPE",
    "EXTENDED_ABLATION_COHORT_ID",
    "EXTENDED_ABLATION_EXECUTION_INPUT_SCHEMA_VERSION",
    "EXTENDED_ABLATION_EXECUTOR_ID",
    "EXTENDED_ABLATION_ESTIMAND_ID",
    "EXTENDED_ABLATION_HORIZON_ID",
    "EXTENDED_ABLATION_METRIC_IDS",
    "EXTENDED_ABLATION_OUTCOME_ID",
    "EXTENDED_ABLATION_REFERENCE_POLICY_ID",
    "EXTENDED_ABLATION_SPEC_DEFINITION_ID",
    "EXTENDED_ABLATION_SCHEMA_VERSION",
    "EXTENDED_ABLATION_SPEC_SCHEMA_VERSION",
    "EXTENDED_ABLATION_VARIANT_POLICY_ID",
    "ExpectedRejectionReceipt",
    "ExtendedAblationArtifact",
    "ExtendedAblationAudit",
    "ExtendedAblationDimension",
    "ExtendedAblationError",
    "ExtendedAblationIntegrityError",
    "ExtendedAblationRow",
    "ExtendedAblationSchemaError",
    "ExtendedAblationSpec",
    "ExtendedAblationStatus",
    "ExtendedExecutionReceipt",
    "ExtendedExecutionFailure",
    "ExtendedImplementationReceipt",
    "audit_extended_ablation_artifact",
    "extended_ablation_definition_sha256",
    "extended_ablation_execution_input_sha256",
    "extended_ablation_registered_output_cells",
    "extended_ablation_registered_outputs",
    "run_extended_ablations",
]
