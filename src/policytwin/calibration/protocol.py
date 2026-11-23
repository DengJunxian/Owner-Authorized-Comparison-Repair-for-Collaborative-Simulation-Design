"""Production study contracts for synthetic calibration workflows.

The contracts in this module replace test-only protocol fixtures.  They issue
only process-local synthetic capabilities and cannot authorize real-data or
real-policy claims.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from policytwin.calibration.schema import (
    CalibrationConfig,
    CalibrationError,
    CalibrationInput,
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
    RandomnessProtocol,
    RegisteredOutput,
    StudyMode,
    StudyProtocol,
    SyntheticProtocolExecutionIssuer,
)
from policytwin.provenance import sha256_bytes


def _definition(identifier: str, prefix: str = "") -> FrozenDefinition:
    return FrozenDefinition(
        definition_id=identifier,
        description=f"Frozen synthetic definition for {identifier}.",
        definition_sha256=sha256_bytes(f"{prefix}{identifier}".encode()),
    )


def build_synthetic_calibration_protocol(config: CalibrationConfig) -> StudyProtocol:
    """Build the complete synthetic-only protocol required by calibration."""

    if not isinstance(config, CalibrationConfig):
        raise CalibrationError("calibration protocol requires a CalibrationConfig")
    if config.synthetic_only is not True:
        raise CalibrationError("calibration protocol factory is synthetic-only")
    calibration_statistics = tuple(
        statistic.name for statistic in config.summaries.calibration
    )
    held_out_statistics = tuple(
        statistic.name for statistic in config.summaries.held_out
    )
    if set(calibration_statistics) & set(held_out_statistics):
        raise CalibrationError(
            "calibration and held-out protocol statistics must be disjoint"
        )
    return StudyProtocol(
        schema_version=2,
        study_id=config.study_id,
        mode=StudyMode.SYNTHETIC,
        claim_scope=ClaimScope.SYNTHETIC_ONLY,
        knowledge_cutoff=config.cutoff,
        model_frozen_at=config.cutoff,
        registered_at=config.cutoff + timedelta(days=1),
        policy_start=config.cutoff + timedelta(days=2),
        policy_end=config.cutoff + timedelta(days=3),
        evaluation_reveal_at=config.cutoff + timedelta(days=4),
        estimand=_definition("paired_distribution"),
        baseline_policy=_definition("control", "baseline:"),
        candidate_policy=_definition("candidate", "candidate:"),
        cohorts=(
            FrozenCohort(
                cohort_id="all_units",
                description="Frozen synthetic cohort.",
                definition_sha256=sha256_bytes(b"synthetic:all_units"),
                basis=CohortBasis.PRE_TREATMENT_OBSERVED,
            ),
        ),
        horizons=(_definition("short_horizon"),),
        metrics=(_definition("synthetic_metric"),),
        baselines=(_definition("zero_effect"),),
        outputs=(
            RegisteredOutput(
                output_id="synthetic_output",
                metric_id="synthetic_metric",
                cohort_id="all_units",
                horizon_id="short_horizon",
                role=OutputRole.PRIMARY,
                interval_method="posterior_credible_interval",
                interval_levels=(0.8, 0.95),
            ),
        ),
        calibration_statistics=calibration_statistics,
        heldout_validation_statistics=held_out_statistics,
        policy_evaluation_outcomes=("synthetic_policy_effect",),
        failure_thresholds=(
            FailureThreshold(
                check_id="temporal_leakage",
                comparator="higher_is_worse",
                yellow_value=0.0,
                red_value=0.0,
                abstain_at_red=True,
            ),
        ),
        abstention_rule=_definition("abstention"),
        randomness_protocol=RandomnessProtocol.EVENT_KEYED_PAIRED,
    )


def issue_synthetic_calibration_authorization(
    config: CalibrationConfig,
    observed: CalibrationInput,
    *,
    execution_id: str,
    execution_started_at: datetime | None = None,
) -> tuple[StudyProtocol, ProtocolExecutionAuthorization, datetime]:
    """Issue one single-use capability for the exact config and input."""

    if not isinstance(observed, CalibrationInput):
        raise CalibrationError("calibration authorization requires CalibrationInput")
    observed.validate_for(config)
    protocol = build_synthetic_calibration_protocol(config)
    started_at = config.cutoff if execution_started_at is None else execution_started_at
    authorization = SyntheticProtocolExecutionIssuer(protocol).issue(
        execution_id=execution_id,
        consumer=ProtocolExecutionConsumer.CALIBRATION_WORKFLOW,
        purpose=ProtocolExecutionPurpose.PRE_POLICY_CALIBRATION,
        config_sha256=config.sha256,
        input_sha256=observed.sha256,
        execution_started_at=started_at,
    )
    return protocol, authorization, started_at


__all__ = [
    "build_synthetic_calibration_protocol",
    "issue_synthetic_calibration_authorization",
]
