"""Trusted bridge from temporal snapshots to empirical evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from policytwin.data.temporal_firewall import (
    AccessPurpose,
    ArtifactKind,
    DataPartition,
    FrozenSnapshot,
    LineageReference,
    LineageRole,
    RuleCode,
    TemporalFirewall,
    TemporalRecord,
    TemporalSchemaError,
    TemporalTableRow,
    ValidatedArtifact,
    decode_temporal_table,
    lineage_references_sha256,
)
from policytwin.data.transformation_receipts import (
    TransformationReceiptError,
    transformation_implementation_sha256,
    verify_transformation_receipt,
)
from policytwin.empirical.artifacts import (
    capture_empirical_runtime_metadata,
    write_empirical_bundle,
)
from policytwin.empirical.authorization import (
    FrozenEvaluationAuthorization,
    _issue_frozen_evaluation_authorization,
)
from policytwin.empirical.evaluator import EmpiricalEvaluator
from policytwin.empirical.schema import (
    DerivationEvidenceClass,
    EmpiricalEvaluationArtifact,
    EmpiricalEvaluationConfig,
    EmpiricalEvaluationError,
    FreezeMode,
    FrozenEvaluationInputs,
    FrozenInputError,
    FrozenPanel,
)
from policytwin.protocol import (
    ProtocolExecutionAuthorization,
    ProtocolExecutionConsumer,
    ProtocolExecutionPurpose,
    ProtocolExecutionReceipt,
    ProtocolValidationError,
    StudyProtocol,
    consume_protocol_execution_authorization,
)
from policytwin.provenance import (
    MetadataValidationError,
    canonical_json_bytes,
    read_json_bytes_object,
)


@dataclass(frozen=True, slots=True)
class FrozenEmpiricalRun:
    """Completed production job and its written bundle manifest."""

    artifact: EmpiricalEvaluationArtifact
    protocol_execution_receipt: ProtocolExecutionReceipt
    manifest_path: Path

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, EmpiricalEvaluationArtifact):
            raise FrozenInputError("frozen run artifact has an invalid schema")
        if not isinstance(self.protocol_execution_receipt, ProtocolExecutionReceipt):
            raise FrozenInputError("frozen run protocol receipt has an invalid schema")
        if (
            self.protocol_execution_receipt.config_sha256 != self.artifact.config_sha256
            or self.protocol_execution_receipt.input_sha256
            != self.artifact.panel_sha256
        ):
            raise FrozenInputError("frozen run protocol receipt bindings changed")
        if not isinstance(self.manifest_path, Path) or not self.manifest_path.is_file():
            raise FrozenInputError("frozen run manifest was not written")


@dataclass(frozen=True, slots=True)
class _VerifiedFrozenEvaluation:
    """Strictly parsed inputs plus their single-use execution authority."""

    config: EmpiricalEvaluationConfig
    panel: FrozenPanel
    authorization: FrozenEvaluationAuthorization


def _artifact_by_path(
    artifacts: tuple[ValidatedArtifact, ...], relative_path: str
) -> ValidatedArtifact:
    matches = tuple(
        artifact
        for artifact in artifacts
        if artifact.artifact.relative_path == relative_path
    )
    if len(matches) != 1:
        raise FrozenInputError(
            f"frozen snapshot must contain exactly one artifact at {relative_path!r}"
        )
    return matches[0]


def _verify_panel_derivation_closure(
    *,
    firewall: TemporalFirewall,
    config: EmpiricalEvaluationConfig,
    panel: FrozenPanel,
    panel_artifact: ValidatedArtifact,
    artifacts: tuple[ValidatedArtifact, ...],
) -> str:
    """Replay every v2 panel cell from full TABLE payloads in this snapshot."""
    if not config.synthetic_test_only or not panel.synthetic_only:
        raise FrozenInputError(
            "real empirical evaluation remains blocked: source, checkpoint, "
            "prediction, and cohort derivation authority are unavailable"
        )
    if panel.schema_version != 2:
        raise FrozenInputError(
            "verified snapshot evaluation requires a receipt-backed panel "
            "with schema_version 2"
        )
    table_artifacts = tuple(
        artifact
        for artifact in artifacts
        if artifact.artifact.kind is ArtifactKind.TABLE
    )
    complete_tables: dict[str, tuple[TemporalTableRow, ...]] = {}
    try:
        for artifact in table_artifacts:
            path = artifact.artifact.relative_path
            complete_tables[path] = decode_temporal_table(artifact.payload)
    except (TemporalSchemaError, TransformationReceiptError) as error:
        raise FrozenInputError(f"cannot decode panel source table: {error}") from error
    if not complete_tables:
        raise FrozenInputError(
            "receipt-backed empirical panel requires at least one frozen TABLE"
        )

    policy_start = config.treatment.policy_start
    for unit in panel.units:
        for characteristic in unit.characteristics:
            if characteristic.record.event_time >= policy_start:
                raise FrozenInputError(
                    "pre-treatment characteristic derivation has a policy-post "
                    "output time"
                )
            if characteristic.record.available_time >= policy_start:
                raise FrozenInputError(
                    "pre-treatment characteristic derivation was not available "
                    "before policy start"
                )
            if characteristic.record.partition is DataPartition.POST_POLICY_EVALUATION:
                raise FrozenInputError(
                    "pre-treatment characteristic derivation uses the policy-post "
                    "partition"
                )
    for observation in panel.observations:
        if (
            observation.relative_period < 0
            and observation.record.event_time >= policy_start
        ):
            raise FrozenInputError(
                "negative relative-period derivation has a policy-post output time"
            )
        if (
            observation.relative_period >= 0
            and observation.record.event_time < policy_start
        ):
            raise FrozenInputError(
                "non-negative relative-period derivation has a pre-policy output time"
            )
        mapped_time = config.event_time_for_period(observation.relative_period)
        if mapped_time is not None and observation.record.event_time != mapped_time:
            raise FrozenInputError(
                "panel derivation output time differs from period_time_mapping"
            )

    input_records: list[TemporalRecord] = []
    for receipt in panel.derivation_receipts:
        if receipt.plan.purpose is not AccessPurpose.FROZEN_EVALUATION:
            raise FrozenInputError(
                "panel derivation receipt was not issued for frozen evaluation"
            )
        if receipt.plan.construction_contract_sha256 != config.sha256:
            raise FrozenInputError(
                "panel derivation plan differs from the frozen empirical config"
            )
        missing_paths = sorted(set(receipt.plan.source_paths) - set(complete_tables))
        if missing_paths:
            raise FrozenInputError(
                "panel derivation source TABLE is absent from the frozen snapshot; "
                f"missing={missing_paths[:3]!r}"
            )
        receipt_tables = {
            path: complete_tables[path] for path in receipt.plan.source_paths
        }
        try:
            verified = verify_transformation_receipt(
                receipt=receipt,
                complete_source_tables=receipt_tables,
                firewall=firewall,
            )
        except TransformationReceiptError as error:
            raise FrozenInputError(
                f"panel derivation receipt did not replay: {error}"
            ) from error
        input_records.extend(verified.input_records)

    records_by_id: dict[str, TemporalRecord] = {}
    for record in input_records:
        prior = records_by_id.get(record.record_id)
        if prior is not None and prior.to_mapping() != record.to_mapping():
            raise FrozenInputError(
                "panel receipts reuse one raw record_id with different metadata"
            )
        records_by_id[record.record_id] = record
    expected = tuple(
        LineageReference(record, LineageRole.RAW_INPUT)
        for _, record in sorted(records_by_id.items())
    )
    actual = panel_artifact.artifact.lineage
    expected_mappings = tuple(
        sorted(
            (reference.to_mapping() for reference in expected),
            key=canonical_json_bytes,
        )
    )
    actual_mappings = tuple(
        sorted(
            (reference.to_mapping() for reference in actual),
            key=canonical_json_bytes,
        )
    )
    if expected_mappings != actual_mappings:
        expected_ids = {reference.record_id for reference in expected}
        actual_ids = {reference.record_id for reference in actual}
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        mismatched = (
            sorted(expected_ids & actual_ids) if not missing and not extra else []
        )
        raise FrozenInputError(
            "panel replay input set does not exactly match committed artifact "
            "lineage; "
            f"missing={missing[:3]!r}, extra={extra[:3]!r}, "
            f"mismatched={mismatched[:3]!r}"
        )
    expected_sha256 = lineage_references_sha256(expected)
    if panel_artifact.artifact.lineage_manifest_sha256 != expected_sha256:
        raise FrozenInputError(
            "panel lineage commitment is not the exact replay input set"
        )
    return expected_sha256


def _verify_prediction_pre_reveal(
    *,
    config: EmpiricalEvaluationConfig,
    checkpoint_artifact: ValidatedArtifact,
    prediction_artifact: ValidatedArtifact,
    first_post_available_at: datetime,
) -> None:
    checkpoint_record = checkpoint_artifact.artifact.record
    prediction_record = prediction_artifact.artifact.record
    if checkpoint_record.ingestion_time >= first_post_available_at:
        raise FrozenInputError(
            "model checkpoint was frozen after policy-post data became available"
        )
    if prediction_record.ingestion_time >= first_post_available_at:
        raise FrozenInputError(
            "predictions were frozen after policy-post data became available"
        )
    if checkpoint_record.ingestion_time > prediction_record.ingestion_time:
        raise FrozenInputError("predictions were frozen before their model checkpoint")
    if checkpoint_record.ingestion_time > config.treatment.policy_start:
        raise FrozenInputError("model checkpoint was not frozen before policy start")
    if prediction_record.ingestion_time > config.treatment.policy_start:
        raise FrozenInputError("predictions were not frozen before policy start")
    checkpoint_mapping = checkpoint_record.to_mapping()
    if not any(
        reference.role is LineageRole.FEATURE_INPUT
        and reference.record.to_mapping() == checkpoint_mapping
        for reference in prediction_artifact.artifact.lineage
    ):
        raise FrozenInputError(
            "prediction artifact lineage does not include the frozen model checkpoint"
        )
    if any(
        reference.record.ingestion_time >= first_post_available_at
        for reference in prediction_artifact.artifact.lineage
    ):
        raise FrozenInputError(
            "prediction artifact lineage contains policy-post information"
        )


def _verify_frozen_snapshot_inputs_inner(
    *,
    firewall: TemporalFirewall,
    snapshot: FrozenSnapshot,
    expected_config: EmpiricalEvaluationConfig | None,
    expected_panel: FrozenPanel | None,
    config_relative_path: str,
    model_checkpoint_relative_path: str,
    prediction_relative_path: str,
    panel_relative_path: str,
    evaluation_started_at: datetime,
) -> _VerifiedFrozenEvaluation:
    """Verify and authorize inputs while the no-network context remains active.

    This low-level bridge must be called from ``firewall.frozen_evaluation()``.
    Most callers should use :func:`run_frozen_empirical_evaluation`, which keeps
    verification, parsing, estimation, and artifact writing inside one context.
    """
    if not firewall.frozen_evaluation_active:
        raise FrozenInputError(
            "snapshot authorization requires an active frozen_evaluation() context"
        )
    paths = (
        config_relative_path,
        model_checkpoint_relative_path,
        prediction_relative_path,
        panel_relative_path,
    )
    if len(set(paths)) != len(paths):
        raise FrozenInputError(
            "config, checkpoint, prediction, and panel paths must differ"
        )

    manifest = firewall.verify_snapshot(snapshot, AccessPurpose.FROZEN_EVALUATION)
    artifacts = firewall.load_snapshot(snapshot, AccessPurpose.FROZEN_EVALUATION)
    config_artifact = _artifact_by_path(artifacts, config_relative_path)
    checkpoint_artifact = _artifact_by_path(artifacts, model_checkpoint_relative_path)
    prediction_artifact = _artifact_by_path(artifacts, prediction_relative_path)
    panel_artifact = _artifact_by_path(artifacts, panel_relative_path)
    if config_artifact.artifact.kind is not ArtifactKind.TEXT:
        raise FrozenInputError("frozen empirical config must be a text JSON artifact")
    if checkpoint_artifact.artifact.kind is not ArtifactKind.MODEL_CHECKPOINT:
        raise FrozenInputError("model checkpoint path is not a checkpoint artifact")
    if prediction_artifact.artifact.kind is not ArtifactKind.FEATURE:
        raise FrozenInputError("prediction path is not a lineage-committed feature")
    if panel_artifact.artifact.kind is not ArtifactKind.FEATURE:
        raise FrozenInputError(
            "empirical panel path is not a lineage-committed feature artifact"
        )
    try:
        frozen_config_mapping = read_json_bytes_object(
            config_artifact.payload, source=config_relative_path
        )
        frozen_panel_mapping = read_json_bytes_object(
            panel_artifact.payload, source=panel_relative_path
        )
        config = EmpiricalEvaluationConfig.from_mapping(frozen_config_mapping)
        panel = FrozenPanel.from_mapping(frozen_panel_mapping)
    except MetadataValidationError as error:
        raise FrozenInputError(
            f"cannot parse frozen empirical JSON: {error}"
        ) from error
    except EmpiricalEvaluationError as error:
        raise FrozenInputError(
            f"frozen empirical JSON does not satisfy its strict schema: {error}"
        ) from error
    if config.synthetic_test_only != panel.synthetic_only:
        raise FrozenInputError("config and panel synthetic-only markers must agree")
    if (
        expected_config is not None
        and expected_config.to_mapping() != config.to_mapping()
    ):
        raise FrozenInputError(
            "in-memory empirical config does not match the frozen config artifact"
        )
    if expected_panel is not None and expected_panel.to_mapping() != panel.to_mapping():
        raise FrozenInputError(
            "in-memory panel does not match the frozen panel artifact"
        )

    post_records = tuple(
        observation.record
        for observation in panel.observations
        if observation.relative_period >= 0
    )
    if not post_records:
        raise FrozenInputError("empirical panel has no policy-post observations")
    first_post_available_at = min(record.available_time for record in post_records)
    panel_lineage_sha256 = _verify_panel_derivation_closure(
        firewall=firewall,
        config=config,
        panel=panel,
        panel_artifact=panel_artifact,
        artifacts=artifacts,
    )
    _verify_prediction_pre_reveal(
        config=config,
        checkpoint_artifact=checkpoint_artifact,
        prediction_artifact=prediction_artifact,
        first_post_available_at=first_post_available_at,
    )
    config_frozen_at = config_artifact.artifact.record.ingestion_time
    if config_frozen_at > prediction_artifact.artifact.record.ingestion_time:
        raise FrozenInputError("prediction artifact predates its frozen config")
    if config_frozen_at > config.treatment.policy_start:
        raise FrozenInputError("empirical protocol was not frozen before policy start")

    receipt = FrozenEvaluationInputs(
        schema_version=3,
        mode=FreezeMode.VERIFIED_TEMPORAL_SNAPSHOT,
        config_sha256=config.sha256,
        panel_sha256=panel.sha256,
        data_manifest_sha256=snapshot.manifest_sha256,
        model_checkpoint_sha256=checkpoint_artifact.artifact.record.content_hash,
        prediction_artifact_sha256=prediction_artifact.artifact.record.content_hash,
        panel_artifact_sha256=panel_artifact.artifact.record.content_hash,
        panel_lineage_sha256=panel_lineage_sha256,
        panel_derivation_registry_sha256=panel.derivation_registry_sha256,
        panel_derivation_receipt_count=len(panel.derivation_receipts),
        transformation_implementation_sha256=(transformation_implementation_sha256()),
        derivation_evidence_class=(DerivationEvidenceClass.SYNTHETIC_TABLE_REPLAY_ONLY),
        panel_derivation_replay_verified=True,
        source_authority_verified=False,
        checkpoint_derivation_replay_verified=False,
        prediction_derivation_replay_verified=False,
        cohort_derivation_replay_verified=False,
        snapshot_id=manifest.snapshot_id,
        model_frozen_at=checkpoint_artifact.artifact.record.ingestion_time,
        predictions_frozen_at=prediction_artifact.artifact.record.ingestion_time,
        post_policy_data_first_available_at=first_post_available_at,
        snapshot_frozen_at=manifest.created_at,
        evaluation_started_at=evaluation_started_at,
        policy_start=config.treatment.policy_start,
        knowledge_cutoff=firewall.policy.cutoff,
    )
    if receipt.data_manifest_sha256 != snapshot.manifest_sha256:
        raise FrozenInputError("snapshot commitment changed during verification")
    return _VerifiedFrozenEvaluation(
        config=config,
        panel=panel,
        authorization=_issue_frozen_evaluation_authorization(
            firewall=firewall,
            snapshot=snapshot,
            config=config,
            panel=panel,
            receipt=receipt,
        ),
    )


def verify_frozen_snapshot_inputs(
    *,
    firewall: TemporalFirewall,
    snapshot: FrozenSnapshot,
    config: EmpiricalEvaluationConfig,
    panel: FrozenPanel,
    config_relative_path: str,
    model_checkpoint_relative_path: str,
    prediction_relative_path: str,
    panel_relative_path: str,
    evaluation_started_at: datetime,
) -> FrozenEvaluationAuthorization:
    """Issue an audited capability after complete snapshot and lineage checks.

    ``config`` and ``panel`` are expected-value assertions for low-level
    integration.  The trusted objects bound into the capability are always
    reconstructed from snapshot bytes inside the active no-network context.
    Production callers should use :func:`run_frozen_empirical_evaluation`.
    """
    try:
        verified = _verify_frozen_snapshot_inputs_inner(
            firewall=firewall,
            snapshot=snapshot,
            expected_config=config,
            expected_panel=panel,
            config_relative_path=config_relative_path,
            model_checkpoint_relative_path=model_checkpoint_relative_path,
            prediction_relative_path=prediction_relative_path,
            panel_relative_path=panel_relative_path,
            evaluation_started_at=evaluation_started_at,
        )
    except EmpiricalEvaluationError as error:
        firewall.audit_log.record(
            action="authorize_empirical_evaluation",
            outcome="rejected",
            rule_code=RuleCode.MANIFEST_MISMATCH,
            policy_sha256=firewall.policy.sha256,
            cutoff=firewall.policy.cutoff.isoformat(),
            artifact_id=(
                snapshot.snapshot_id if isinstance(snapshot, FrozenSnapshot) else None
            ),
            record_id=None,
            partition=DataPartition.POST_POLICY_EVALUATION.value,
            details={"message": str(error)},
        )
        raise
    firewall.audit_log.record(
        action="authorize_empirical_evaluation",
        outcome="allowed",
        rule_code=RuleCode.OK,
        policy_sha256=firewall.policy.sha256,
        cutoff=firewall.policy.cutoff.isoformat(),
        artifact_id=snapshot.snapshot_id,
        record_id=None,
        partition=DataPartition.POST_POLICY_EVALUATION.value,
        details={
            "config_sha256": verified.config.sha256,
            "panel_sha256": verified.panel.sha256,
            "snapshot_manifest_sha256": snapshot.manifest_sha256,
        },
    )
    return verified.authorization


def _validate_empirical_protocol_contract(
    protocol: StudyProtocol,
    config: EmpiricalEvaluationConfig,
) -> None:
    if not config.synthetic_test_only:
        raise FrozenInputError(
            "real empirical protocol execution requires an external "
            "preregistration verifier, which is unavailable"
        )
    if protocol.study_id != config.study_id:
        raise FrozenInputError("empirical config belongs to a different study protocol")
    if (
        protocol.policy_start != config.treatment.policy_start
        or protocol.policy_end != config.treatment.policy_end
    ):
        raise FrozenInputError(
            "empirical policy window differs from the study protocol"
        )
    if protocol.candidate_policy.definition_id != config.treatment.policy_id:
        raise FrozenInputError(
            "empirical treatment differs from the registered candidate policy"
        )


def run_frozen_empirical_evaluation(
    *,
    firewall: TemporalFirewall,
    snapshot: FrozenSnapshot,
    config_relative_path: str,
    model_checkpoint_relative_path: str,
    prediction_relative_path: str,
    panel_relative_path: str,
    evaluation_started_at: datetime,
    study_protocol: StudyProtocol,
    protocol_authorization: ProtocolExecutionAuthorization,
    output_directory: Path,
    project_root: Path | None = None,
) -> FrozenEmpiricalRun:
    """Run load -> parse -> evaluate -> artifact write without network access."""
    root = (
        Path(__file__).resolve().parents[3]
        if project_root is None
        else project_root.resolve()
    )
    runtime_metadata = capture_empirical_runtime_metadata(root)
    with firewall.frozen_evaluation():
        try:
            verified = _verify_frozen_snapshot_inputs_inner(
                firewall=firewall,
                snapshot=snapshot,
                expected_config=None,
                expected_panel=None,
                config_relative_path=config_relative_path,
                model_checkpoint_relative_path=model_checkpoint_relative_path,
                prediction_relative_path=prediction_relative_path,
                panel_relative_path=panel_relative_path,
                evaluation_started_at=evaluation_started_at,
            )
        except EmpiricalEvaluationError as error:
            firewall.audit_log.record(
                action="authorize_empirical_evaluation",
                outcome="rejected",
                rule_code=RuleCode.MANIFEST_MISMATCH,
                policy_sha256=firewall.policy.sha256,
                cutoff=firewall.policy.cutoff.isoformat(),
                artifact_id=snapshot.snapshot_id,
                record_id=None,
                partition=DataPartition.POST_POLICY_EVALUATION.value,
                details={"message": str(error)},
            )
            raise
        firewall.audit_log.record(
            action="authorize_empirical_evaluation",
            outcome="allowed",
            rule_code=RuleCode.OK,
            policy_sha256=firewall.policy.sha256,
            cutoff=firewall.policy.cutoff.isoformat(),
            artifact_id=snapshot.snapshot_id,
            record_id=None,
            partition=DataPartition.POST_POLICY_EVALUATION.value,
            details={
                "config_sha256": verified.config.sha256,
                "panel_sha256": verified.panel.sha256,
                "snapshot_manifest_sha256": snapshot.manifest_sha256,
            },
        )
        try:
            protocol_receipt = consume_protocol_execution_authorization(
                protocol_authorization,
                protocol=study_protocol,
                consumer=ProtocolExecutionConsumer.EMPIRICAL_FROZEN_EVALUATION,
                purpose=(ProtocolExecutionPurpose.POLICY_POST_EMPIRICAL_EVALUATION),
                config_sha256=verified.config.sha256,
                input_sha256=verified.panel.sha256,
                execution_started_at=evaluation_started_at,
            )
        except ProtocolValidationError as error:
            firewall.audit_log.record(
                action="authorize_study_protocol_execution",
                outcome="rejected",
                rule_code=RuleCode.MANIFEST_MISMATCH,
                policy_sha256=firewall.policy.sha256,
                cutoff=firewall.policy.cutoff.isoformat(),
                artifact_id=snapshot.snapshot_id,
                record_id=None,
                partition=DataPartition.POST_POLICY_EVALUATION.value,
                details={"message": str(error)},
            )
            raise FrozenInputError(
                f"study protocol authorization rejected: {error}"
            ) from error
        try:
            _validate_empirical_protocol_contract(study_protocol, verified.config)
        except FrozenInputError as error:
            firewall.audit_log.record(
                action="authorize_study_protocol_execution",
                outcome="rejected",
                rule_code=RuleCode.MANIFEST_MISMATCH,
                policy_sha256=firewall.policy.sha256,
                cutoff=firewall.policy.cutoff.isoformat(),
                artifact_id=snapshot.snapshot_id,
                record_id=None,
                partition=DataPartition.POST_POLICY_EVALUATION.value,
                details={"message": str(error)},
            )
            raise
        firewall.audit_log.record(
            action="authorize_study_protocol_execution",
            outcome="allowed",
            rule_code=RuleCode.OK,
            policy_sha256=firewall.policy.sha256,
            cutoff=firewall.policy.cutoff.isoformat(),
            artifact_id=snapshot.snapshot_id,
            record_id=None,
            partition=DataPartition.POST_POLICY_EVALUATION.value,
            details={
                "protocol_execution_receipt_sha256": protocol_receipt.sha256,
                "study_protocol_sha256": study_protocol.sha256,
            },
        )
        artifact = EmpiricalEvaluator().evaluate(
            config=verified.config,
            panel=verified.panel,
            frozen_inputs=verified.authorization,
        )
        manifest_path = write_empirical_bundle(
            output_directory,
            config=verified.config,
            panel=verified.panel,
            artifact=artifact,
            protocol_execution_receipt=protocol_receipt,
            project_root=root,
            runtime_metadata=runtime_metadata,
        )
    return FrozenEmpiricalRun(
        artifact=artifact,
        protocol_execution_receipt=protocol_receipt,
        manifest_path=manifest_path,
    )


__all__ = [
    "FrozenEmpiricalRun",
    "run_frozen_empirical_evaluation",
    "verify_frozen_snapshot_inputs",
]
