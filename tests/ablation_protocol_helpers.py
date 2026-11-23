"""Caller-owned study-protocol fixtures for authorized ablation tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from policytwin.experiments.extended_ablations import (
    EXTENDED_ABLATION_COHORT_ID,
    EXTENDED_ABLATION_ESTIMAND_ID,
    EXTENDED_ABLATION_HORIZON_ID,
    EXTENDED_ABLATION_METRIC_IDS,
    EXTENDED_ABLATION_OUTCOME_ID,
    EXTENDED_ABLATION_REFERENCE_POLICY_ID,
    EXTENDED_ABLATION_SPEC_DEFINITION_ID,
    EXTENDED_ABLATION_VARIANT_POLICY_ID,
    ExtendedAblationSpec,
    extended_ablation_definition_sha256,
    extended_ablation_execution_input_sha256,
    extended_ablation_registered_outputs,
)
from policytwin.experiments.integrated_synthetic import (
    INTEGRATED_METRIC_IDS,
    integrated_metric_definition_sha256,
    integrated_synthetic_market_config,
    integrated_synthetic_pair,
)
from policytwin.experiments.mechanism_ablations import (
    MECHANISM_ABLATION_ESTIMAND_ID,
    MECHANISM_ABLATION_OUTCOME_ID,
    MECHANISM_ABLATION_SPEC_DEFINITION_ID,
    MechanismAblationSpec,
    mechanism_ablation_estimand_definition_sha256,
    mechanism_ablation_execution_input_sha256,
    mechanism_ablation_registered_outputs,
)
from policytwin.protocol import (
    ClaimScope,
    CohortBasis,
    FailureThreshold,
    FrozenCohort,
    FrozenDefinition,
    ProtocolExecutionAuthorization,
    ProtocolExecutionConsumer,
    ProtocolExecutionPurpose,
    RandomnessProtocol,
    StudyMode,
    StudyProtocol,
    SyntheticProtocolExecutionIssuer,
)
from policytwin.provenance import canonical_json_bytes, sha256_bytes


def _definition(
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


def mechanism_protocol(spec: MechanismAblationSpec) -> StudyProtocol:
    config = integrated_synthetic_market_config()
    pair = integrated_synthetic_pair(
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
    cutoff = config.cutoff
    return StudyProtocol(
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
        estimand=_definition(
            MECHANISM_ABLATION_ESTIMAND_ID,
            definition_sha256=mechanism_ablation_estimand_definition_sha256(),
        ),
        baseline_policy=_definition(
            pair.baseline_policy.policy_id,
            definition_sha256=pair.baseline_policy.sha256,
        ),
        candidate_policy=_definition(
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
            _definition(
                horizon.horizon_id,
                definition_sha256=sha256_bytes(
                    canonical_json_bytes(horizon.to_mapping())
                ),
            )
            for horizon in pair.horizons
        ),
        metrics=tuple(
            _definition(
                metric_id,
                definition_sha256=integrated_metric_definition_sha256(metric_id),
            )
            for metric_id in INTEGRATED_METRIC_IDS
        ),
        baselines=(
            _definition("synthetic_reference_model"),
            _definition(
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
        abstention_rule=_definition("synthetic_abstention"),
        randomness_protocol=RandomnessProtocol.EVENT_KEYED_PAIRED,
    )


def extended_protocol(spec: ExtendedAblationSpec) -> StudyProtocol:
    cutoff = datetime(2025, 1, 10, tzinfo=UTC)
    return StudyProtocol(
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
        estimand=_definition(
            EXTENDED_ABLATION_ESTIMAND_ID,
            definition_sha256=extended_ablation_definition_sha256(
                EXTENDED_ABLATION_ESTIMAND_ID
            ),
        ),
        baseline_policy=_definition(
            EXTENDED_ABLATION_REFERENCE_POLICY_ID,
            definition_sha256=extended_ablation_definition_sha256(
                EXTENDED_ABLATION_REFERENCE_POLICY_ID
            ),
        ),
        candidate_policy=_definition(
            EXTENDED_ABLATION_VARIANT_POLICY_ID,
            definition_sha256=extended_ablation_definition_sha256(
                EXTENDED_ABLATION_VARIANT_POLICY_ID
            ),
        ),
        cohorts=(
            FrozenCohort(
                cohort_id=EXTENDED_ABLATION_COHORT_ID,
                description="All units in the synthetic cross-subsystem registry.",
                definition_sha256=extended_ablation_definition_sha256(
                    EXTENDED_ABLATION_COHORT_ID
                ),
                basis=CohortBasis.LATENT_STRATEGY,
            ),
        ),
        horizons=(
            _definition(
                EXTENDED_ABLATION_HORIZON_ID,
                definition_sha256=extended_ablation_definition_sha256(
                    EXTENDED_ABLATION_HORIZON_ID
                ),
            ),
        ),
        metrics=tuple(
            _definition(
                metric_id,
                definition_sha256=extended_ablation_definition_sha256(metric_id),
            )
            for metric_id in EXTENDED_ABLATION_METRIC_IDS
        ),
        baselines=(
            _definition("synthetic_reference_model"),
            _definition(
                EXTENDED_ABLATION_SPEC_DEFINITION_ID,
                definition_sha256=spec.sha256,
                description="Frozen synthetic execution grid for this ablation.",
            ),
        ),
        outputs=extended_ablation_registered_outputs(),
        calibration_statistics=("synthetic_calibration_statistic",),
        heldout_validation_statistics=("synthetic_validation_statistic",),
        policy_evaluation_outcomes=(EXTENDED_ABLATION_OUTCOME_ID,),
        failure_thresholds=(
            FailureThreshold(
                check_id="temporal_leakage",
                comparator="higher_is_worse",
                yellow_value=0.0,
                red_value=0.0,
                abstain_at_red=True,
            ),
        ),
        abstention_rule=_definition("synthetic_abstention"),
        randomness_protocol=RandomnessProtocol.EVENT_KEYED_PAIRED,
    )


def mechanism_authorization(
    spec: MechanismAblationSpec,
    *,
    execution_id: str,
) -> tuple[StudyProtocol, ProtocolExecutionAuthorization, datetime]:
    protocol = mechanism_protocol(spec)
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


def extended_authorization(
    spec: ExtendedAblationSpec,
    *,
    execution_id: str,
) -> tuple[StudyProtocol, ProtocolExecutionAuthorization, datetime]:
    protocol = extended_protocol(spec)
    started_at = protocol.registered_at
    authorization = SyntheticProtocolExecutionIssuer(protocol).issue(
        execution_id=execution_id,
        consumer=ProtocolExecutionConsumer.EXTENDED_ABLATION_WORKFLOW,
        purpose=ProtocolExecutionPurpose.SYNTHETIC_EXTENDED_ABLATION,
        config_sha256=spec.sha256,
        input_sha256=extended_ablation_execution_input_sha256(spec),
        execution_started_at=started_at,
    )
    return protocol, authorization, started_at
