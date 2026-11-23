"""Versioned empirical bundle writing and tamper audit."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn, cast

from policytwin.empirical.schema import (
    EmpiricalEvaluationArtifact,
    EmpiricalEvaluationConfig,
    EmpiricalEvaluationError,
    EmpiricalSchemaError,
    FrozenEvaluationInputs,
    FrozenPanel,
)
from policytwin.protocol import (
    ProtocolExecutionConsumer,
    ProtocolExecutionPurpose,
    ProtocolExecutionReceipt,
    ProtocolValidationError,
)
from policytwin.provenance import (
    EnvironmentMetadata,
    EventKeyMetadata,
    GitMetadata,
    JsonlEventWriter,
    MetadataValidationError,
    build_event_key_metadata,
    canonical_json_bytes,
    collect_environment_metadata,
    collect_git_metadata,
    read_json_bytes_object,
    read_jsonl_bytes_objects,
    sha256_bytes,
    sha256_file,
    write_json_file,
)

_BUNDLE_FILES = frozenset(
    {
        "audit.jsonl",
        "config.json",
        "frozen_panel.json",
        "frozen_inputs.json",
        "protocol_execution_receipt.json",
        "provenance.json",
        "results.json",
    }
)

_MAX_BUNDLE_FILE_BYTES = 256 * 1024 * 1024


class EmpiricalArtifactIntegrityError(RuntimeError):
    """Raised when a versioned empirical bundle is incomplete or modified."""


def _integrity_fail(message: str) -> NoReturn:
    raise EmpiricalArtifactIntegrityError(message)


@dataclass(frozen=True, slots=True)
class EmpiricalBundleAuditReport:
    """Positive internal cross-binding and estimator-reproducibility outcome.

    This report does not assert that a source TABLE was reopened or replayed and
    is not real-data execution authority.
    """

    valid: bool
    scientific_artifact_sha256: str
    scientific_artifact_schema_version: int
    config_sha256: str
    panel_sha256: str
    study_protocol_sha256: str
    protocol_execution_receipt_sha256: str
    specification_ids: tuple[str, ...]
    checked_files: tuple[str, ...]

    def to_mapping(self) -> dict[str, object]:
        return {
            "checked_files": list(self.checked_files),
            "config_sha256": self.config_sha256,
            "panel_sha256": self.panel_sha256,
            "protocol_execution_receipt_sha256": (
                self.protocol_execution_receipt_sha256
            ),
            "scientific_artifact_sha256": self.scientific_artifact_sha256,
            "scientific_artifact_schema_version": (
                self.scientific_artifact_schema_version
            ),
            "specification_ids": list(self.specification_ids),
            "study_protocol_sha256": self.study_protocol_sha256,
            "valid": self.valid,
        }


@dataclass(frozen=True, slots=True)
class EmpiricalRuntimeMetadata:
    """Environment evidence captured before post-policy payloads are opened."""

    git: GitMetadata
    environment: EnvironmentMetadata

    def __post_init__(self) -> None:
        if not isinstance(self.git, GitMetadata):
            _integrity_fail("runtime git metadata has an invalid schema")
        if not isinstance(self.environment, EnvironmentMetadata):
            _integrity_fail("runtime environment metadata has an invalid schema")


@dataclass(frozen=True, slots=True)
class _CapturedBundleFile:
    """One regular file read from a stable descriptor-relative identity."""

    payload: bytes
    fingerprint: tuple[int, int, int, int, int, int, int]


@dataclass(frozen=True, slots=True)
class _CapturedBundle:
    """Stable input bytes plus directory identity for one audit invocation."""

    directory_fingerprint: tuple[int, int, int, int, int, int, int]
    files: dict[str, _CapturedBundleFile]


def _fingerprint(value: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_exact_fd(fd: int, expected_size: int, name: str) -> bytes:
    if expected_size < 0 or expected_size > _MAX_BUNDLE_FILE_BYTES:
        _integrity_fail(f"empirical bundle file size is invalid: {name}")
    chunks: list[bytes] = []
    observed = 0
    while True:
        chunk = os.read(fd, min(1024 * 1024, expected_size + 1 - observed))
        if not chunk:
            break
        chunks.append(chunk)
        observed += len(chunk)
        if observed > expected_size:
            _integrity_fail(f"empirical bundle file changed while read: {name}")
    if observed != expected_size:
        _integrity_fail(f"empirical bundle file changed while read: {name}")
    return b"".join(chunks)


def _capture_bundle(directory: Path) -> _CapturedBundle:
    expected_files = _BUNDLE_FILES | {"manifest.json"}
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    directory_flags = flags | getattr(os, "O_DIRECTORY", 0)
    try:
        directory_fd = os.open(directory, directory_flags)
    except OSError as error:
        raise EmpiricalArtifactIntegrityError(
            f"cannot open empirical bundle directory: {error}"
        ) from error
    try:
        before_directory = os.fstat(directory_fd)
        if not stat.S_ISDIR(before_directory.st_mode):
            _integrity_fail("empirical bundle must be a regular directory")
        observed_files = frozenset(os.listdir(directory_fd))
        if observed_files != expected_files:
            _integrity_fail(
                "empirical bundle file set changed: "
                f"missing={sorted(expected_files - observed_files)!r}, "
                f"extra={sorted(observed_files - expected_files)!r}"
            )
        captures: dict[str, _CapturedBundleFile] = {}
        for name in sorted(expected_files):
            try:
                fd = os.open(name, flags, dir_fd=directory_fd)
            except OSError as error:
                raise EmpiricalArtifactIntegrityError(
                    f"cannot open empirical bundle file {name}: {error}"
                ) from error
            try:
                before = os.fstat(fd)
                if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                    _integrity_fail(
                        "empirical bundle file must be a singly linked regular "
                        f"file: {name}"
                    )
                payload = _read_exact_fd(fd, before.st_size, name)
                after = os.fstat(fd)
                if _fingerprint(before) != _fingerprint(after):
                    _integrity_fail(f"empirical bundle file changed while read: {name}")
                captures[name] = _CapturedBundleFile(
                    payload=payload,
                    fingerprint=_fingerprint(after),
                )
            finally:
                os.close(fd)
        after_directory = os.fstat(directory_fd)
        if (
            _fingerprint(before_directory) != _fingerprint(after_directory)
            or frozenset(os.listdir(directory_fd)) != expected_files
        ):
            _integrity_fail("empirical bundle directory changed while captured")
        return _CapturedBundle(
            directory_fingerprint=_fingerprint(after_directory),
            files=captures,
        )
    finally:
        os.close(directory_fd)


def _recheck_bundle(directory: Path, captured: _CapturedBundle) -> None:
    """Reject path replacement or byte changes before returning a positive audit."""
    current = _capture_bundle(directory)
    if current.directory_fingerprint != captured.directory_fingerprint:
        _integrity_fail("empirical bundle directory identity changed during audit")
    for name, original in captured.files.items():
        latest = current.files[name]
        if (
            latest.fingerprint != original.fingerprint
            or latest.payload != original.payload
        ):
            _integrity_fail(f"empirical bundle path changed during audit: {name}")


def capture_empirical_runtime_metadata(project_root: Path) -> EmpiricalRuntimeMetadata:
    """Collect local runtime identity before entering the no-network job."""
    root = project_root.resolve()
    return EmpiricalRuntimeMetadata(
        git=collect_git_metadata(root),
        environment=collect_environment_metadata(root),
    )


def _digest_mapping(path: Path, relative_path: str) -> dict[str, object]:
    file_path = path / relative_path
    return {
        "path": relative_path,
        "sha256": sha256_file(file_path),
        "size_bytes": file_path.stat().st_size,
    }


def _expected_audit_events(
    *,
    config: EmpiricalEvaluationConfig,
    panel: FrozenPanel,
    artifact: EmpiricalEvaluationArtifact,
    protocol_execution_receipt: ProtocolExecutionReceipt,
) -> tuple[dict[str, object], ...]:
    """Build the complete ordered event registry committed by a bundle."""
    events: list[dict[str, object]] = [
        {
            "action": "frozen_inputs_verified",
            "checkpoint_derivation_replay_verified": (
                artifact.frozen_inputs.checkpoint_derivation_replay_verified
            ),
            "cohort_derivation_replay_verified": (
                artifact.frozen_inputs.cohort_derivation_replay_verified
            ),
            "config_sha256": config.sha256,
            "data_manifest_sha256": artifact.frozen_inputs.data_manifest_sha256,
            "derivation_evidence_class": (
                artifact.frozen_inputs.derivation_evidence_class.value
            ),
            "model_checkpoint_sha256": (artifact.frozen_inputs.model_checkpoint_sha256),
            "outcome": "allowed",
            "panel_sha256": panel.sha256,
            "panel_artifact_sha256": artifact.frozen_inputs.panel_artifact_sha256,
            "panel_derivation_registry_sha256": (
                artifact.frozen_inputs.panel_derivation_registry_sha256
            ),
            "panel_derivation_receipt_count": (
                artifact.frozen_inputs.panel_derivation_receipt_count
            ),
            "panel_derivation_replay_verified": (
                artifact.frozen_inputs.panel_derivation_replay_verified
            ),
            "panel_lineage_sha256": artifact.frozen_inputs.panel_lineage_sha256,
            "prediction_artifact_sha256": (
                artifact.frozen_inputs.prediction_artifact_sha256
            ),
            "prediction_derivation_replay_verified": (
                artifact.frozen_inputs.prediction_derivation_replay_verified
            ),
            "protocol_execution_receipt_sha256": (protocol_execution_receipt.sha256),
            "schema_version": 3,
            "source_authority_verified": (
                artifact.frozen_inputs.source_authority_verified
            ),
            "study_protocol_sha256": protocol_execution_receipt.protocol_sha256,
            "synthetic_only": True,
            "transformation_implementation_sha256": (
                artifact.frozen_inputs.transformation_implementation_sha256
            ),
        }
    ]
    for result in artifact.specification_results:
        bootstrap = result.bootstrap_inference
        events.append(
            {
                "action": "specification_evaluated",
                "bootstrap_draw_count": len(result.bootstrap_draws),
                "bootstrap_invalid_draw_count": bootstrap.invalid_draws,
                "bootstrap_requested_draw_count": bootstrap.requested_draws,
                "bootstrap_support_gate_passed": bootstrap.support_gate_passed,
                "bootstrap_valid_draw_count": bootstrap.valid_draws,
                "bootstrap_valid_draw_gate_passed": (bootstrap.valid_draw_gate_passed),
                "causal_reporting_reasons": list(result.causal_reporting.reasons),
                "causal_reporting_status": result.causal_reporting.status.value,
                "estimate_count": len(result.estimates),
                "outcome": "retained",
                "pretrend_status": result.pretrend_diagnostic.status.value,
                "randomization_draw_count": len(result.randomization_inference.draws),
                "randomization_invalid_draw_count": (
                    result.randomization_inference.invalid_draws
                ),
                "randomization_requested_draw_count": (
                    result.randomization_inference.requested_draws
                ),
                "randomization_valid_draw_count": (
                    result.randomization_inference.valid_draws
                ),
                "schema_version": 3,
                "specification_id": result.specification_id,
            }
        )
    events.append(
        {
            "action": "full_specification_curve_sealed",
            "outcome": "allowed",
            "schema_version": 3,
            "scientific_artifact_sha256": artifact.sha256,
            "scientific_artifact_schema_version": artifact.schema_version,
            "selection_policy": artifact.selection_policy,
            "specification_ids": [
                result.specification_id for result in artifact.specification_results
            ],
            "synthetic_only": True,
        }
    )
    return tuple(events)


def write_empirical_bundle(
    directory: Path,
    *,
    config: EmpiricalEvaluationConfig,
    panel: FrozenPanel,
    artifact: EmpiricalEvaluationArtifact,
    protocol_execution_receipt: ProtocolExecutionReceipt,
    project_root: Path | None = None,
    runtime_metadata: EmpiricalRuntimeMetadata | None = None,
) -> Path:
    """Write a new immutable-by-convention empirical result bundle.

    Existing directories are never overwritten.  Synthetic bundles retain the
    complete frozen panel so every persisted scientific value can be reproduced.
    Real-data publication remains categorically blocked until an external
    preregistration and lawful-data verifier is available.
    """
    if artifact.config_sha256 != config.sha256:
        _integrity_fail("artifact/config commitment mismatch before bundle write")
    if artifact.panel_sha256 != panel.sha256:
        _integrity_fail("artifact/panel commitment mismatch before bundle write")
    if not config.synthetic_test_only or not panel.synthetic_only:
        _integrity_fail(
            "real empirical bundles require an external preregistration verifier"
        )
    try:
        reproduced_artifact = EmpiricalEvaluationArtifact.from_mapping(
            artifact.to_mapping(),
            config=config,
            panel=panel,
            frozen_inputs=artifact.frozen_inputs,
        )
    except EmpiricalEvaluationError as error:
        raise EmpiricalArtifactIntegrityError(
            f"empirical artifact cannot be reproduced before bundle write: {error}"
        ) from error
    if reproduced_artifact.sha256 != artifact.sha256:
        _integrity_fail("empirical artifact replay changed its scientific commitment")
    if not isinstance(protocol_execution_receipt, ProtocolExecutionReceipt):
        _integrity_fail("protocol execution receipt has an invalid schema")
    if (
        protocol_execution_receipt.consumer
        is not ProtocolExecutionConsumer.EMPIRICAL_FROZEN_EVALUATION
        or protocol_execution_receipt.purpose
        is not ProtocolExecutionPurpose.POLICY_POST_EMPIRICAL_EVALUATION
    ):
        _integrity_fail("protocol receipt cannot authorize empirical evaluation")
    if protocol_execution_receipt.config_sha256 != config.sha256:
        _integrity_fail("protocol receipt/config commitment mismatch")
    if protocol_execution_receipt.input_sha256 != panel.sha256:
        _integrity_fail("protocol receipt/panel commitment mismatch")
    if protocol_execution_receipt.study_id != config.study_id:
        _integrity_fail("protocol receipt belongs to a different empirical study")
    if (
        protocol_execution_receipt.policy_start != config.treatment.policy_start
        or protocol_execution_receipt.policy_end != config.treatment.policy_end
    ):
        _integrity_fail("protocol receipt policy window differs from empirical config")
    if (
        protocol_execution_receipt.execution_started_at
        != artifact.frozen_inputs.evaluation_started_at
    ):
        _integrity_fail("protocol receipt evaluation time differs from freeze evidence")
    if (
        artifact.causal_reporting_criteria_sha256
        != config.causal_reporting_criteria_sha256
    ):
        _integrity_fail("artifact causal-reporting criteria differ from the config")
    frozen_ids = tuple(item.specification_id for item in config.specifications)
    result_ids = tuple(item.specification_id for item in artifact.specification_results)
    if result_ids != frozen_ids:
        _integrity_fail("result curve is not identical to the frozen specification set")
    if any(
        result.randomization_inference.requested_draws
        != config.randomization_repetitions
        for result in artifact.specification_results
    ):
        _integrity_fail("randomization requested_draws differ from the frozen config")
    directory.mkdir(parents=True, exist_ok=False)
    write_json_file(directory / "config.json", config.to_mapping())
    write_json_file(directory / "frozen_panel.json", panel.to_mapping())
    write_json_file(
        directory / "frozen_inputs.json", artifact.frozen_inputs.to_mapping()
    )
    write_json_file(
        directory / "protocol_execution_receipt.json",
        protocol_execution_receipt.to_mapping(),
    )
    write_json_file(directory / "results.json", artifact.to_mapping())
    bootstrap_event_keys = tuple(
        draw.event_key
        for result in artifact.specification_results
        for draw in result.bootstrap_draws
    )
    randomization_event_keys = tuple(
        draw.event_key
        for result in artifact.specification_results
        for draw in result.randomization_inference.draws
    )
    event_keys = bootstrap_event_keys + randomization_event_keys
    root = (
        Path(__file__).resolve().parents[3]
        if project_root is None
        else project_root.resolve()
    )
    runtime = (
        capture_empirical_runtime_metadata(root)
        if runtime_metadata is None
        else runtime_metadata
    )
    if not isinstance(runtime, EmpiricalRuntimeMetadata):
        _integrity_fail("runtime_metadata has an invalid schema")
    provenance = {
        "checkpoint_derivation_replay_verified": (
            artifact.frozen_inputs.checkpoint_derivation_replay_verified
        ),
        "cohort_derivation_replay_verified": (
            artifact.frozen_inputs.cohort_derivation_replay_verified
        ),
        "config_sha256": config.sha256,
        "causal_reporting_criteria_sha256": (config.causal_reporting_criteria_sha256),
        "data_manifest_sha256": artifact.frozen_inputs.data_manifest_sha256,
        "derivation_evidence_class": (
            artifact.frozen_inputs.derivation_evidence_class.value
        ),
        "environment": runtime.environment.to_mapping(),
        "event_keys": build_event_key_metadata(
            schema_version="policytwin.event-key.v1",
            root_key=config.random_seed,
            keys=event_keys,
        ).to_mapping(),
        "git": runtime.git.to_mapping(),
        "model_checkpoint_sha256": (artifact.frozen_inputs.model_checkpoint_sha256),
        "panel_sha256": panel.sha256,
        "panel_artifact_sha256": (artifact.frozen_inputs.panel_artifact_sha256),
        "panel_derivation_registry_sha256": (
            artifact.frozen_inputs.panel_derivation_registry_sha256
        ),
        "panel_derivation_receipt_count": (
            artifact.frozen_inputs.panel_derivation_receipt_count
        ),
        "panel_derivation_replay_verified": (
            artifact.frozen_inputs.panel_derivation_replay_verified
        ),
        "panel_lineage_sha256": artifact.frozen_inputs.panel_lineage_sha256,
        "prediction_artifact_sha256": (
            artifact.frozen_inputs.prediction_artifact_sha256
        ),
        "prediction_derivation_replay_verified": (
            artifact.frozen_inputs.prediction_derivation_replay_verified
        ),
        "protocol_execution_receipt_sha256": protocol_execution_receipt.sha256,
        "schema_version": 7,
        "scientific_artifact_schema_version": artifact.schema_version,
        "source_authority_verified": artifact.frozen_inputs.source_authority_verified,
        "study_protocol_sha256": protocol_execution_receipt.protocol_sha256,
        "study_id": config.study_id,
        "synthetic_only": panel.synthetic_only,
        "transformation_implementation_sha256": (
            artifact.frozen_inputs.transformation_implementation_sha256
        ),
    }
    write_json_file(directory / "provenance.json", provenance)
    expected_audit_events = _expected_audit_events(
        config=config,
        panel=panel,
        artifact=artifact,
        protocol_execution_receipt=protocol_execution_receipt,
    )
    with JsonlEventWriter(directory / "audit.jsonl") as audit:
        for event in expected_audit_events:
            audit.write(event)
    manifest = {
        "artifacts": [
            _digest_mapping(directory, relative_path)
            for relative_path in sorted(_BUNDLE_FILES)
        ],
        "config_sha256": config.sha256,
        "panel_sha256": panel.sha256,
        "protocol_execution_receipt_sha256": protocol_execution_receipt.sha256,
        "schema_version": 7,
        "scientific_artifact_sha256": artifact.sha256,
        "scientific_artifact_schema_version": artifact.schema_version,
        "specification_ids": list(result_ids),
        "study_protocol_sha256": protocol_execution_receipt.protocol_sha256,
    }
    write_json_file(directory / "manifest.json", manifest)
    return directory / "manifest.json"


def _require_dict(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        _integrity_fail(f"{field} must be an object")
    return cast(dict[str, object], value)


def _require_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        _integrity_fail(f"{field} must be a non-empty string")
    return value


def _require_int(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _integrity_fail(f"{field} must be an integer >= {minimum}")
    return value


def _require_bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        _integrity_fail(f"{field} must be a boolean")
    return value


def audit_empirical_bundle(
    directory: Path,
    *,
    expected_scientific_artifact_sha256: str | None = None,
) -> EmpiricalBundleAuditReport:
    """Audit retained bytes, cross-bindings, and estimator reproducibility.

    A positive report does not reopen or replay the source TABLE snapshot and
    does not establish source authority or authorize a real-data evaluation.
    """
    expected_files = _BUNDLE_FILES | {"manifest.json"}
    captured = _capture_bundle(directory)
    try:
        manifest = read_json_bytes_object(
            captured.files["manifest.json"].payload, source="manifest.json"
        )
        config = read_json_bytes_object(
            captured.files["config.json"].payload, source="config.json"
        )
        panel_mapping = read_json_bytes_object(
            captured.files["frozen_panel.json"].payload,
            source="frozen_panel.json",
        )
        receipt = read_json_bytes_object(
            captured.files["frozen_inputs.json"].payload,
            source="frozen_inputs.json",
        )
        protocol_execution_receipt_mapping = read_json_bytes_object(
            captured.files["protocol_execution_receipt.json"].payload,
            source="protocol_execution_receipt.json",
        )
        provenance = read_json_bytes_object(
            captured.files["provenance.json"].payload, source="provenance.json"
        )
        results = read_json_bytes_object(
            captured.files["results.json"].payload, source="results.json"
        )
        audit_events = read_jsonl_bytes_objects(
            captured.files["audit.jsonl"].payload, source="audit.jsonl"
        )
    except MetadataValidationError as error:
        raise EmpiricalArtifactIntegrityError(str(error)) from error
    expected_manifest_fields = frozenset(
        {
            "artifacts",
            "config_sha256",
            "panel_sha256",
            "protocol_execution_receipt_sha256",
            "schema_version",
            "scientific_artifact_sha256",
            "scientific_artifact_schema_version",
            "specification_ids",
            "study_protocol_sha256",
        }
    )
    if frozenset(manifest) != expected_manifest_fields:
        _integrity_fail("empirical manifest schema is missing or has unknown fields")
    if _require_int(manifest["schema_version"], "manifest schema_version") != 7:
        _integrity_fail("empirical manifest schema_version must be 7")
    raw_digests = manifest["artifacts"]
    if not isinstance(raw_digests, list):
        _integrity_fail("empirical manifest artifacts must be an array")
    digest_paths: set[str] = set()
    for raw_digest in raw_digests:
        digest = _require_dict(raw_digest, "manifest artifact")
        if frozenset(digest) != {"path", "sha256", "size_bytes"}:
            _integrity_fail("manifest artifact fields are missing or unknown")
        relative_path = _require_string(digest["path"], "artifact path")
        if relative_path not in _BUNDLE_FILES or relative_path in digest_paths:
            _integrity_fail("manifest artifact path is unexpected or duplicated")
        digest_paths.add(relative_path)
        captured_file = captured.files[relative_path]
        if digest["sha256"] != sha256_bytes(captured_file.payload):
            _integrity_fail(f"artifact hash mismatch: {relative_path}")
        size_bytes = _require_int(digest["size_bytes"], "artifact size_bytes")
        if size_bytes != len(captured_file.payload):
            _integrity_fail(f"artifact size mismatch: {relative_path}")
    if digest_paths != set(_BUNDLE_FILES):
        _integrity_fail("manifest does not commit to every empirical bundle artifact")

    config_sha256 = sha256_bytes(canonical_json_bytes(config))
    panel_sha256 = _require_string(manifest["panel_sha256"], "panel_sha256")
    try:
        protocol_execution_receipt = ProtocolExecutionReceipt.from_mapping(
            protocol_execution_receipt_mapping
        )
    except ProtocolValidationError as error:
        raise EmpiricalArtifactIntegrityError(
            f"invalid protocol execution receipt: {error}"
        ) from error
    if (
        protocol_execution_receipt.consumer
        is not ProtocolExecutionConsumer.EMPIRICAL_FROZEN_EVALUATION
        or protocol_execution_receipt.purpose
        is not ProtocolExecutionPurpose.POLICY_POST_EMPIRICAL_EVALUATION
        or protocol_execution_receipt.config_sha256 != config_sha256
        or protocol_execution_receipt.input_sha256 != panel_sha256
    ):
        _integrity_fail("empirical protocol execution receipt bindings changed")
    if (
        manifest["study_protocol_sha256"] != protocol_execution_receipt.protocol_sha256
        or manifest["protocol_execution_receipt_sha256"]
        != protocol_execution_receipt.sha256
    ):
        _integrity_fail("empirical manifest protocol commitments mismatch")
    scientific_sha256 = sha256_bytes(canonical_json_bytes(results))
    expected_provenance_fields = frozenset(
        {
            "checkpoint_derivation_replay_verified",
            "cohort_derivation_replay_verified",
            "config_sha256",
            "causal_reporting_criteria_sha256",
            "data_manifest_sha256",
            "derivation_evidence_class",
            "environment",
            "event_keys",
            "git",
            "model_checkpoint_sha256",
            "panel_sha256",
            "panel_artifact_sha256",
            "panel_derivation_registry_sha256",
            "panel_derivation_receipt_count",
            "panel_derivation_replay_verified",
            "panel_lineage_sha256",
            "prediction_artifact_sha256",
            "prediction_derivation_replay_verified",
            "protocol_execution_receipt_sha256",
            "schema_version",
            "scientific_artifact_schema_version",
            "source_authority_verified",
            "study_protocol_sha256",
            "study_id",
            "synthetic_only",
            "transformation_implementation_sha256",
        }
    )
    if frozenset(provenance) != expected_provenance_fields:
        _integrity_fail("empirical provenance schema is missing or has unknown fields")
    if (
        _require_int(results.get("schema_version"), "result schema_version") != 7
        or _require_int(
            manifest.get("scientific_artifact_schema_version"),
            "manifest scientific_artifact_schema_version",
        )
        != 7
        or _require_int(provenance.get("schema_version"), "provenance schema_version")
        != 7
        or _require_int(
            provenance.get("scientific_artifact_schema_version"),
            "provenance scientific_artifact_schema_version",
        )
        != 7
    ):
        _integrity_fail(
            "empirical scientific artifact and bundle schema_version must be 7"
        )
    if config_sha256 != manifest["config_sha256"]:
        _integrity_fail("frozen config semantic hash mismatch")
    if results.get("config_sha256") != config_sha256:
        _integrity_fail("results do not reference the frozen config")
    try:
        parsed_config = EmpiricalEvaluationConfig.from_mapping(config)
        parsed_panel = FrozenPanel.from_mapping(panel_mapping)
        parsed_receipt = FrozenEvaluationInputs.from_mapping(receipt)
    except EmpiricalSchemaError as error:
        raise EmpiricalArtifactIntegrityError(
            "frozen empirical config, panel, or receipt does not satisfy its "
            "strict schema"
        ) from error
    if (
        not parsed_config.synthetic_test_only
        or not parsed_panel.synthetic_only
        or results.get("synthetic_only") is not True
        or _require_bool(provenance.get("synthetic_only"), "provenance synthetic_only")
        is not True
    ):
        _integrity_fail(
            "empirical bundle synthetic authority markers changed; real audit "
            "requires external temporal authorization"
        )
    if parsed_panel.sha256 != panel_sha256:
        _integrity_fail("retained frozen panel semantic hash mismatch")
    try:
        reproduced_artifact = EmpiricalEvaluationArtifact.from_mapping(
            results,
            config=parsed_config,
            panel=parsed_panel,
            frozen_inputs=parsed_receipt,
        )
    except EmpiricalEvaluationError as error:
        raise EmpiricalArtifactIntegrityError(
            "scientific artifact does not reproduce from the retained frozen "
            f"config and panel: {error}"
        ) from error
    if reproduced_artifact.sha256 != scientific_sha256:
        _integrity_fail("reproduced empirical artifact commitment changed")
    try:
        GitMetadata.from_mapping(_require_dict(provenance.get("git"), "provenance git"))
        EnvironmentMetadata.from_mapping(
            _require_dict(provenance.get("environment"), "provenance environment")
        )
        event_key_metadata = EventKeyMetadata.from_mapping(
            _require_dict(provenance.get("event_keys"), "provenance event_keys")
        )
    except MetadataValidationError as error:
        raise EmpiricalArtifactIntegrityError(
            f"empirical provenance does not satisfy its strict schema: {error}"
        ) from error
    expected_event_keys = tuple(
        draw.event_key
        for result in reproduced_artifact.specification_results
        for draw in result.bootstrap_draws
    ) + tuple(
        draw.event_key
        for result in reproduced_artifact.specification_results
        for draw in result.randomization_inference.draws
    )
    expected_event_key_metadata = build_event_key_metadata(
        schema_version="policytwin.event-key.v1",
        root_key=parsed_config.random_seed,
        keys=expected_event_keys,
    )
    if event_key_metadata != expected_event_key_metadata:
        _integrity_fail(
            "provenance event-key registry, order, count, or commitment changed"
        )
    if (
        protocol_execution_receipt.study_id != parsed_config.study_id
        or protocol_execution_receipt.policy_start
        != parsed_config.treatment.policy_start
        or protocol_execution_receipt.policy_end != parsed_config.treatment.policy_end
        or protocol_execution_receipt.execution_started_at.isoformat()
        != receipt.get("evaluation_started_at")
    ):
        _integrity_fail("empirical protocol receipt/config chronology mismatch")
    criteria_sha256 = parsed_config.causal_reporting_criteria_sha256
    if results.get("causal_reporting_criteria_sha256") != criteria_sha256:
        _integrity_fail("results causal-reporting criteria differ from the config")
    if results.get("panel_sha256") != panel_sha256:
        _integrity_fail("results do not reference the frozen panel")
    if results.get("frozen_inputs") != receipt:
        _integrity_fail("results and standalone freeze evidence differ")
    expected_receipt_fields = frozenset(
        {
            "checkpoint_derivation_replay_verified",
            "cohort_derivation_replay_verified",
            "config_sha256",
            "data_manifest_sha256",
            "derivation_evidence_class",
            "evaluation_started_at",
            "knowledge_cutoff",
            "mode",
            "model_checkpoint_sha256",
            "model_frozen_at",
            "panel_artifact_sha256",
            "panel_derivation_registry_sha256",
            "panel_derivation_receipt_count",
            "panel_derivation_replay_verified",
            "panel_lineage_sha256",
            "panel_sha256",
            "policy_start",
            "post_policy_data_first_available_at",
            "prediction_artifact_sha256",
            "prediction_derivation_replay_verified",
            "predictions_frozen_at",
            "schema_version",
            "snapshot_frozen_at",
            "snapshot_id",
            "source_authority_verified",
            "transformation_implementation_sha256",
        }
    )
    if (
        frozenset(receipt) != expected_receipt_fields
        or receipt.get("schema_version") != 3
    ):
        _integrity_fail("freeze evidence schema is missing or has unknown fields")
    if receipt.get("config_sha256") != config_sha256:
        _integrity_fail("freeze receipt does not reference the frozen config")
    if receipt.get("panel_sha256") != panel_sha256:
        _integrity_fail("freeze receipt does not reference the frozen panel")
    if receipt.get("panel_artifact_sha256") != panel_sha256:
        _integrity_fail("panel artifact bytes differ from the semantic panel")
    if (
        receipt.get("panel_derivation_registry_sha256")
        != parsed_panel.derivation_registry_sha256
    ):
        _integrity_fail(
            "freeze receipt panel-derivation registry differs from the retained panel"
        )
    if receipt.get("panel_derivation_receipt_count") != len(
        parsed_panel.derivation_receipts
    ):
        _integrity_fail(
            "freeze receipt panel-derivation count differs from the retained panel"
        )
    if parsed_panel.derivation_receipts:
        implementation_sha256s = {
            item.plan.implementation_source_sha256
            for item in parsed_panel.derivation_receipts
        }
        if implementation_sha256s != {
            receipt.get("transformation_implementation_sha256")
        }:
            _integrity_fail(
                "freeze receipt transformation implementation differs from the "
                "retained panel receipts"
            )
    if provenance.get("config_sha256") != config_sha256:
        _integrity_fail("provenance does not reference the frozen config")
    if provenance.get("study_id") != parsed_config.study_id:
        _integrity_fail("provenance study_id differs from the frozen config")
    if provenance.get("causal_reporting_criteria_sha256") != criteria_sha256:
        _integrity_fail("provenance causal-reporting criteria mismatch")
    if provenance.get("panel_sha256") != panel_sha256:
        _integrity_fail("provenance does not reference the frozen panel")
    if provenance.get("panel_artifact_sha256") != receipt.get("panel_artifact_sha256"):
        _integrity_fail("provenance panel-artifact commitment mismatch")
    if provenance.get("panel_lineage_sha256") != receipt.get("panel_lineage_sha256"):
        _integrity_fail("provenance panel-lineage commitment mismatch")
    if provenance.get("data_manifest_sha256") != receipt.get("data_manifest_sha256"):
        _integrity_fail("provenance data-manifest commitment mismatch")
    if provenance.get("model_checkpoint_sha256") != receipt.get(
        "model_checkpoint_sha256"
    ):
        _integrity_fail("provenance model-checkpoint commitment mismatch")
    if provenance.get("prediction_artifact_sha256") != receipt.get(
        "prediction_artifact_sha256"
    ):
        _integrity_fail("provenance prediction commitment mismatch")
    derivation_fields = (
        "panel_derivation_registry_sha256",
        "panel_derivation_receipt_count",
        "panel_derivation_replay_verified",
        "transformation_implementation_sha256",
        "derivation_evidence_class",
        "source_authority_verified",
        "checkpoint_derivation_replay_verified",
        "prediction_derivation_replay_verified",
        "cohort_derivation_replay_verified",
    )
    for field in (
        "panel_derivation_replay_verified",
        "source_authority_verified",
        "checkpoint_derivation_replay_verified",
        "prediction_derivation_replay_verified",
        "cohort_derivation_replay_verified",
    ):
        _require_bool(provenance.get(field), f"provenance {field}")
    _require_int(
        provenance.get("panel_derivation_receipt_count"),
        "provenance panel_derivation_receipt_count",
    )
    for field in (
        "panel_derivation_registry_sha256",
        "transformation_implementation_sha256",
        "derivation_evidence_class",
    ):
        _require_string(provenance.get(field), f"provenance {field}")
    for field in derivation_fields:
        if provenance.get(field) != receipt.get(field):
            _integrity_fail(f"provenance derivation boundary mismatch: {field}")
    if (
        provenance.get("study_protocol_sha256")
        != protocol_execution_receipt.protocol_sha256
        or provenance.get("protocol_execution_receipt_sha256")
        != protocol_execution_receipt.sha256
    ):
        _integrity_fail("provenance protocol execution commitment mismatch")
    if scientific_sha256 != manifest["scientific_artifact_sha256"]:
        _integrity_fail("scientific artifact semantic hash mismatch")
    if (
        expected_scientific_artifact_sha256 is not None
        and scientific_sha256 != expected_scientific_artifact_sha256
    ):
        _integrity_fail("scientific artifact does not match the external commitment")

    raw_spec_ids = manifest["specification_ids"]
    if not isinstance(raw_spec_ids, list) or not all(
        isinstance(item, str) and item for item in raw_spec_ids
    ):
        _integrity_fail("manifest specification_ids are invalid")
    specification_ids = tuple(cast(list[str], raw_spec_ids))
    config_specs = config.get("specifications")
    result_specs = results.get("specification_results")
    if not isinstance(config_specs, list) or not isinstance(result_specs, list):
        _integrity_fail("config/results specification curves are invalid")
    frozen_ids = tuple(
        _require_string(
            _require_dict(item, "config specification").get("specification_id"),
            "config specification_id",
        )
        for item in config_specs
    )
    result_ids = tuple(
        _require_string(
            _require_dict(item, "result specification").get("specification_id"),
            "result specification_id",
        )
        for item in result_specs
    )
    if specification_ids != frozen_ids or specification_ids != result_ids:
        _integrity_fail("full frozen specification curve was not retained")
    registered_ids = results.get("registered_specification_ids")
    if not isinstance(registered_ids, list) or tuple(registered_ids) != frozen_ids:
        _integrity_fail("results changed the registered specification commitment")
    for item in result_specs:
        result_mapping = _require_dict(item, "result specification")
        randomization = _require_dict(
            result_mapping.get("randomization_inference"),
            "randomization_inference",
        )
        expected_randomization_fields = {
            "draws",
            "invalid_draws",
            "minimum_valid_draws",
            "observed_statistic",
            "p_value",
            "requested_draws",
            "valid_draw_gate_passed",
            "valid_draws",
        }
        if frozenset(randomization) != expected_randomization_fields:
            _integrity_fail("randomization result fields changed")
        requested_draws = randomization.get("requested_draws")
        raw_draws = randomization.get("draws")
        if (
            isinstance(requested_draws, bool)
            or not isinstance(requested_draws, int)
            or requested_draws != parsed_config.randomization_repetitions
            or not isinstance(raw_draws, list)
            or len(raw_draws) != requested_draws
        ):
            _integrity_fail(
                "randomization requested_draws differ from the frozen config"
            )
        valid_draws = sum(
            _require_dict(draw, "randomization draw").get("statistic") is not None
            for draw in raw_draws
        )
        invalid_draws = requested_draws - valid_draws
        minimum_valid_draws = randomization.get("minimum_valid_draws")
        expected_gate = (
            invalid_draws == 0
            and minimum_valid_draws == parsed_config.minimum_valid_randomization_draws
            and valid_draws >= parsed_config.minimum_valid_randomization_draws
        )
        if (
            randomization.get("valid_draws") != valid_draws
            or randomization.get("invalid_draws") != invalid_draws
            or minimum_valid_draws != parsed_config.minimum_valid_randomization_draws
            or randomization.get("valid_draw_gate_passed") is not expected_gate
            or (randomization.get("p_value") is not None and not expected_gate)
        ):
            _integrity_fail("randomization draw accounting is inconsistent")
        decision = _require_dict(
            result_mapping.get("causal_reporting"), "causal_reporting"
        )
        if frozenset(decision) != {"criteria_sha256", "reasons", "status"}:
            _integrity_fail("causal-reporting decision fields changed")
        if decision.get("criteria_sha256") != criteria_sha256:
            _integrity_fail("specification causal-reporting criteria mismatch")
        status = decision.get("status")
        reasons = decision.get("reasons")
        if (
            status
            not in {
                "passes_internal_causal_diagnostics",
                "fails_internal_causal_diagnostics",
            }
            or not isinstance(reasons, list)
            or not all(isinstance(reason, str) and reason for reason in reasons)
        ):
            _integrity_fail("causal-reporting decision is invalid")
        if (status == "passes_internal_causal_diagnostics") != (not reasons):
            _integrity_fail("internal causal-diagnostic status and reasons disagree")
    if results.get("selection_policy") != "all_frozen_specifications_reported":
        _integrity_fail("results contain a forbidden specification-selection policy")

    expected_audit_events = _expected_audit_events(
        config=parsed_config,
        panel=parsed_panel,
        artifact=reproduced_artifact,
        protocol_execution_receipt=protocol_execution_receipt,
    )
    if canonical_json_bytes(audit_events) != canonical_json_bytes(
        expected_audit_events
    ):
        _integrity_fail(
            "audit event registry, strict schema, order, count, or values changed"
        )
    _recheck_bundle(directory, captured)
    return EmpiricalBundleAuditReport(
        valid=True,
        scientific_artifact_sha256=scientific_sha256,
        scientific_artifact_schema_version=7,
        config_sha256=config_sha256,
        panel_sha256=panel_sha256,
        study_protocol_sha256=protocol_execution_receipt.protocol_sha256,
        protocol_execution_receipt_sha256=protocol_execution_receipt.sha256,
        specification_ids=specification_ids,
        checked_files=tuple(sorted(expected_files)),
    )


__all__ = [
    "EmpiricalRuntimeMetadata",
    "EmpiricalArtifactIntegrityError",
    "EmpiricalBundleAuditReport",
    "audit_empirical_bundle",
    "capture_empirical_runtime_metadata",
    "write_empirical_bundle",
]
