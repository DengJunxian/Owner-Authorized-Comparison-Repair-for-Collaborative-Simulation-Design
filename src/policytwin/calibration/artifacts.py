"""Versioned calibration bundle writer and fail-closed tamper audit."""

from __future__ import annotations

import json
import os
import shutil
import stat
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from pathlib import Path
from typing import NoReturn, cast

from policytwin.calibration.diagnostics import (
    ParameterIdentifiability,
    PosteriorPredictiveChecks,
    PosteriorPredictiveStatistic,
    RecoveryMetric,
)
from policytwin.calibration.schema import (
    CalibrationConfig,
    CalibrationError,
    CalibrationInput,
    IdentifiabilityThresholds,
    ParameterTransform,
    Particle,
    PosteriorDraw,
    PosteriorSamples,
    PriorSpec,
    SummaryLevel,
    SummaryRole,
    SummarySourcePartition,
    SummaryStatisticsConfig,
    SummaryStatisticSpec,
    SummaryVector,
    TransformKind,
    TransformLineage,
    TransformLineageNode,
    parameter_mapping,
)
from policytwin.calibration.workflow import (
    CalibrationReport,
    CalibrationRun,
    SBCRankHistogram,
    SimulationBasedCalibration,
)
from policytwin.data.temporal_firewall import TemporalFirewallError, TemporalRecord
from policytwin.protocol import (
    ClaimScope,
    ProtocolExecutionConsumer,
    ProtocolExecutionPurpose,
    ProtocolExecutionReceipt,
    ProtocolValidationError,
    StudyMode,
)
from policytwin.provenance import (
    EnvironmentMetadata,
    EventKeyMetadata,
    GitMetadata,
    MetadataValidationError,
    build_event_key_metadata,
    canonical_json_bytes,
    collect_environment_metadata,
    collect_git_metadata,
    read_json_bytes_object,
    sha256_bytes,
)

_BUNDLE_FILES = frozenset(
    {
        "audit.jsonl",
        "calibration_report.json",
        "config.json",
        "event_keys.json",
        "input_receipt.json",
        "posterior_predictive_checks.json",
        "posterior_samples.json",
        "protocol_execution_receipt.json",
        "provenance.json",
        "rank_histogram.json",
        "simulator_config.json",
        "true_parameters.json",
    }
)
CALLER_SUPPLIED_GIT_IDENTITY_ATTESTATION = "caller_supplied_unattested"
LOCAL_GIT_IDENTITY_ATTESTATION = (
    "local_runtime_observation_not_independent_attestation"
)
_GIT_IDENTITY_ATTESTATIONS = frozenset(
    {
        CALLER_SUPPLIED_GIT_IDENTITY_ATTESTATION,
        LOCAL_GIT_IDENTITY_ATTESTATION,
    }
)


class CalibrationArtifactIntegrityError(RuntimeError):
    """Raised when a calibration artifact is incomplete or modified."""


@dataclass(frozen=True, slots=True)
class CalibrationArtifactFailureReceipt:
    """Terminal evidence for a writer failure; never a success commitment."""

    destination_name: str
    stage: str
    error_type: str
    error_message: str
    git_identity_attestation: str
    schema_version: int = 1
    status: str = "failed"
    synthetic_only: bool = True
    success_commitment: None = None

    def __post_init__(self) -> None:
        if (
            not self.destination_name
            or "/" in self.destination_name
            or "\\" in self.destination_name
        ):
            _fail("failure receipt destination_name is invalid")
        if not self.stage or not self.error_type or not self.error_message:
            _fail("failure receipt terminal fields must be non-empty")
        if self.git_identity_attestation not in _GIT_IDENTITY_ATTESTATIONS:
            _fail("failure receipt Git identity attestation is invalid")
        if (
            self.schema_version != 1
            or self.status != "failed"
            or self.synthetic_only is not True
            or self.success_commitment is not None
        ):
            _fail("failure receipt cannot claim successful completion")

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def to_mapping(self) -> dict[str, object]:
        return {
            "destination_name": self.destination_name,
            "error_message": self.error_message,
            "error_type": self.error_type,
            "git_identity_attestation": self.git_identity_attestation,
            "schema_version": self.schema_version,
            "stage": self.stage,
            "status": self.status,
            "success_commitment": self.success_commitment,
            "synthetic_only": self.synthetic_only,
        }


class CalibrationArtifactWriteError(CalibrationArtifactIntegrityError):
    """Raised with terminal, machine-readable evidence after a write failure."""

    def __init__(
        self,
        message: str,
        *,
        failure_receipt: CalibrationArtifactFailureReceipt,
        failure_receipt_path: Path | None,
    ) -> None:
        super().__init__(message)
        self.failure_receipt = failure_receipt
        self.failure_receipt_path = failure_receipt_path


def _fail(message: str) -> NoReturn:
    raise CalibrationArtifactIntegrityError(message)


def _payload_digest(payloads: dict[str, bytes], name: str) -> dict[str, object]:
    payload = payloads[name]
    return {
        "path": name,
        "sha256": sha256_bytes(payload),
        "size_bytes": len(payload),
    }


def _pretty_json_bytes(value: object) -> bytes:
    rendered = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    return f"{rendered}\n".encode()


def _git_identity_attestation(git_metadata: GitMetadata | None) -> str:
    return (
        LOCAL_GIT_IDENTITY_ATTESTATION
        if git_metadata is None
        else CALLER_SUPPLIED_GIT_IDENTITY_ATTESTATION
    )


def build_calibration_artifact_payloads(
    *,
    run: CalibrationRun,
    project_root: Path | None = None,
    git_metadata: GitMetadata | None = None,
) -> dict[str, bytes]:
    """Build exact calibration bundle bytes without touching an output path."""
    root = (
        Path(__file__).resolve().parents[3]
        if project_root is None
        else project_root.resolve()
    )
    git = collect_git_metadata(root) if git_metadata is None else git_metadata
    if not isinstance(git, GitMetadata):
        _fail("git_metadata must be GitMetadata when provided")
    event_metadata = build_event_key_metadata(
        schema_version="policytwin.event-key.v1",
        root_key=run.config.seed,
        keys=run.event_keys,
    )
    provenance = {
        "claim_scope": run.report.claim_scope,
        "config_sha256": run.config.sha256,
        "environment": collect_environment_metadata(root).to_mapping(),
        "event_keys": event_metadata.to_mapping(),
        "git": git.to_mapping(),
        "git_identity_attestation": _git_identity_attestation(git_metadata),
        "input_sha256": run.observed.sha256,
        "model_checkpoint_sha256": None,
        "protocol_execution_receipt_sha256": (run.protocol_execution_receipt.sha256),
        "schema_version": 3,
        "simulator_id": run.report.simulator_id,
        "simulator_config_sha256": run.report.simulator_config_sha256,
        "true_parameters_sha256": run.true_parameters_sha256,
        "synthetic_only": True,
        "study_protocol_sha256": run.protocol_execution_receipt.protocol_sha256,
    }
    payloads = {
        "audit.jsonl": _canonical_jsonl(_calibration_audit_events(run)),
        "calibration_report.json": _pretty_json_bytes(run.report.to_mapping()),
        "config.json": _pretty_json_bytes(run.config.to_mapping()),
        "event_keys.json": _pretty_json_bytes(
            {"event_keys": list(run.event_keys), "schema_version": 1}
        ),
        "input_receipt.json": _pretty_json_bytes(run.observed.to_mapping()),
        "posterior_predictive_checks.json": _pretty_json_bytes(run.ppc.to_mapping()),
        "posterior_samples.json": _pretty_json_bytes(run.posterior.to_mapping()),
        "protocol_execution_receipt.json": _pretty_json_bytes(
            run.protocol_execution_receipt.to_mapping()
        ),
        "provenance.json": _pretty_json_bytes(provenance),
        "rank_histogram.json": _pretty_json_bytes(run.sbc.to_mapping()),
        "simulator_config.json": _pretty_json_bytes(
            read_json_bytes_object(
                run.simulator_config_json,
                source="in-memory simulator configuration",
            )
        ),
        "true_parameters.json": _pretty_json_bytes(run.true_parameters_mapping),
    }
    manifest = {
        "artifacts": [
            _payload_digest(payloads, name) for name in sorted(_BUNDLE_FILES)
        ],
        "claim_scope": run.report.claim_scope,
        "config_sha256": run.config.sha256,
        "input_sha256": run.observed.sha256,
        "protocol_execution_receipt_sha256": (run.protocol_execution_receipt.sha256),
        "report_sha256": run.report.sha256,
        "simulator_config_sha256": run.report.simulator_config_sha256,
        "true_parameters_sha256": run.true_parameters_sha256,
        "schema_version": 4,
        "synthetic_only": True,
        "study_protocol_sha256": run.protocol_execution_receipt.protocol_sha256,
    }
    payloads["manifest.json"] = _pretty_json_bytes(manifest)
    return payloads


def _write_new_payload(path: Path, payload: bytes) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                _fail(f"cannot complete calibration artifact write: {path.name}")
            view = view[written:]
        os.fsync(descriptor)
    except OSError as error:
        raise CalibrationArtifactIntegrityError(
            f"cannot create calibration artifact {path.name}: {error}"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _remove_writer_tree(path: Path) -> None:
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return
    except OSError as error:
        raise CalibrationArtifactIntegrityError(
            "cannot inspect calibration writer staging for cleanup"
        ) from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        _fail("calibration writer cleanup target must be a real directory")
    try:
        shutil.rmtree(path)
    except OSError as error:
        raise CalibrationArtifactIntegrityError(
            "cannot remove calibration writer staging after failure"
        ) from error


def _publish_failure_receipt(
    directory: Path,
    receipt: CalibrationArtifactFailureReceipt,
) -> Path | None:
    """Best-effort atomic publication; the exception always retains the receipt."""
    parent = directory.parent
    failure_path = parent / f"{directory.name}.failure.json"
    if failure_path.exists() or failure_path.is_symlink():
        return None
    payload = canonical_json_bytes(receipt.to_mapping())
    descriptor: int | None = None
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{directory.name}.failure-staging-",
            dir=parent,
        )
        temporary_path = Path(temporary_name)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                return None
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        if failure_path.exists() or failure_path.is_symlink():
            return None
        os.replace(temporary_path, failure_path)
        temporary_path = None
        parent_descriptor = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
        return failure_path
    except OSError:
        return None
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_path is not None:
            with suppress(OSError):
                temporary_path.unlink(missing_ok=True)


def write_calibration_artifact(
    directory: Path,
    *,
    run: CalibrationRun,
    project_root: Path | None = None,
    git_metadata: GitMetadata | None = None,
) -> Path:
    """Audit a private staging tree, then atomically publish a complete artifact."""
    if directory.exists() or directory.is_symlink():
        raise CalibrationArtifactIntegrityError(
            "calibration artifact directories are immutable and cannot be overwritten"
        )
    try:
        directory.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise CalibrationArtifactIntegrityError(
            "cannot create calibration artifact parent directory"
        ) from error
    staging: Path | None = None
    published = False
    stage = "serialize_payloads"
    attestation = _git_identity_attestation(git_metadata)
    try:
        payloads = build_calibration_artifact_payloads(
            run=run,
            project_root=project_root,
            git_metadata=git_metadata,
        )
        stage = "create_staging"
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{directory.name}.staging-",
                dir=directory.parent,
            )
        )
        stage = "write_payloads"
        for name in sorted(frozenset(payloads) - {"manifest.json"}):
            _write_new_payload(staging / name, payloads[name])
        _write_new_payload(staging / "manifest.json", payloads["manifest.json"])
        stage = "audit_staging"
        manifest_digest = sha256_bytes(payloads["manifest.json"])
        staged = load_calibration_artifact(
            staging,
            expected_manifest_sha256=manifest_digest,
        )
        if (
            staged.run != run
            or staged.audit.git_identity_attestation != attestation
        ):
            _fail("immediate calibration artifact replay differs from input run")
        stage = "atomic_publish"
        if directory.exists() or directory.is_symlink():
            _fail(
                "calibration artifact directories are immutable and cannot be "
                "overwritten"
            )
        os.replace(staging, directory)
        staging = None
        published = True
        stage = "audit_published"
        final = load_calibration_artifact(
            directory,
            expected_manifest_sha256=manifest_digest,
        )
        if final.run != run or final.audit != staged.audit:
            _fail("published calibration artifact differs from audited staging")
        parent_descriptor = os.open(directory.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
        return directory / "manifest.json"
    except Exception as error:
        cleanup_error: Exception | None = None
        cleanup_target = directory if published else staging
        if cleanup_target is not None:
            try:
                _remove_writer_tree(cleanup_target)
            except Exception as caught:
                cleanup_error = caught
        receipt = CalibrationArtifactFailureReceipt(
            destination_name=directory.name,
            stage=stage,
            error_type=type(error).__name__,
            error_message=str(error)[:2048] or "calibration artifact write failed",
            git_identity_attestation=attestation,
        )
        receipt_path = _publish_failure_receipt(directory, receipt)
        raised = CalibrationArtifactWriteError(
            "calibration artifact write failed with terminal failure evidence",
            failure_receipt=receipt,
            failure_receipt_path=receipt_path,
        )
        if cleanup_error is not None:
            raise raised from cleanup_error
        raise raised from error


@dataclass(frozen=True, slots=True)
class CalibrationArtifactAuditReport:
    """Positive result from rehashing a complete calibration bundle."""

    valid: bool
    report_sha256: str
    config_sha256: str
    input_sha256: str
    study_protocol_sha256: str
    protocol_execution_receipt_sha256: str
    simulator_config_sha256: str
    true_parameters_sha256: str
    event_keys_sha256: str
    event_key_count: int
    git_identity_attestation: str
    checked_files: tuple[str, ...]

    def to_mapping(self) -> dict[str, object]:
        return {
            "checked_files": list(self.checked_files),
            "config_sha256": self.config_sha256,
            "event_key_count": self.event_key_count,
            "event_keys_sha256": self.event_keys_sha256,
            "git_identity_attestation": self.git_identity_attestation,
            "input_sha256": self.input_sha256,
            "protocol_execution_receipt_sha256": (
                self.protocol_execution_receipt_sha256
            ),
            "report_sha256": self.report_sha256,
            "simulator_config_sha256": self.simulator_config_sha256,
            "study_protocol_sha256": self.study_protocol_sha256,
            "true_parameters_sha256": self.true_parameters_sha256,
            "valid": self.valid,
        }


@dataclass(frozen=True, slots=True)
class CalibrationArtifactLoadResult:
    """Replay-verified load relative to a caller-supplied manifest digest.

    The raw manifest digest binds this result to bytes selected by the caller.  It
    does not establish who retained that digest, when it was retained, or any
    external custody, registration, or attestation claim.
    """

    run: CalibrationRun
    audit: CalibrationArtifactAuditReport
    manifest_raw_sha256: str


def _object(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        _fail(f"{field} must be an object")
    return cast(dict[str, object], value)


def _fields(value: dict[str, object], expected: frozenset[str], field: str) -> None:
    if frozenset(value) != expected:
        missing = sorted(expected - frozenset(value))
        unknown = sorted(frozenset(value) - expected)
        _fail(f"{field} fields changed: missing={missing!r}, unknown={unknown!r}")


def _array(value: object, field: str, *, nonempty: bool = True) -> list[object]:
    if not isinstance(value, list) or (nonempty and not value):
        _fail(f"{field} must be {'a non-empty' if nonempty else 'an'} array")
    return value


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{field} must be a non-empty string")
    return value


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{field} must be an integer")
    return value


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        _fail(f"{field} must be numeric")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise CalibrationArtifactIntegrityError(
            f"{field} must be a finite number"
        ) from error
    if not isfinite(result):
        _fail(f"{field} must be a finite number")
    return result


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        _fail(f"{field} must be boolean")
    return value


def _calibration_audit_events(run: CalibrationRun) -> tuple[dict[str, object], ...]:
    """Derive the complete ordered audit registry from replay-verified state."""
    events: list[dict[str, object]] = [
        {
            "action": "calibration_input_validated",
            "calibration_statistics": list(run.report.calibration_statistics),
            "held_out_statistics": list(run.report.held_out_statistics),
            "input_sha256": run.observed.sha256,
            "outcome": "allowed",
            "policy_post_records": 0,
            "protocol_execution_receipt_sha256": (
                run.protocol_execution_receipt.sha256
            ),
            "study_protocol_sha256": run.protocol_execution_receipt.protocol_sha256,
            "schema_version": 2,
        },
        {
            "accepted_particles": len(run.posterior.particles),
            "action": "posterior_distribution_estimated",
            "backend": run.posterior.backend_id,
            "outcome": "retained_full_distribution",
            "posterior_draws": len(run.posterior.draws),
            "schema_version": 1,
        },
    ]
    events.extend(
        {
            "action": "identifiability_diagnostic",
            "boundary_pileup": diagnostic.boundary_pileup,
            "estimable_for_reporting": diagnostic.estimable_for_reporting,
            "multimodal": diagnostic.multimodal,
            "outcome": "reported",
            "parameter": diagnostic.parameter,
            "schema_version": 1,
        }
        for diagnostic in run.identifiability
    )
    events.append(
        {
            "action": "synthetic_calibration_artifact_sealed",
            "claim_scope": run.report.claim_scope,
            "outcome": "synthetic_method_check_only",
            "real_data_status": run.report.real_data_status,
            "report_sha256": run.report.sha256,
            "true_parameters_sha256": run.true_parameters_sha256,
            "schema_version": 1,
        }
    )
    return tuple(events)


def _canonical_jsonl(events: tuple[dict[str, object], ...]) -> bytes:
    return b"".join(canonical_json_bytes(event) + b"\n" for event in events)


def _strings(value: object, field: str) -> tuple[str, ...]:
    return tuple(
        _string(item, f"{field} item") for item in _array(value, field, nonempty=False)
    )


def _parse_summary_vector(value: object, field: str) -> SummaryVector:
    mapping = _object(value, field)
    _fields(mapping, frozenset({"values"}), field)
    values = _object(mapping["values"], f"{field}.values")
    if not values:
        _fail(f"{field}.values must not be empty")
    return SummaryVector(
        tuple((name, _number(item, name)) for name, item in values.items())
    )


def _parse_config(mapping: dict[str, object]) -> CalibrationConfig:
    _fields(
        mapping,
        frozenset(
            {
                "accepted_particles",
                "cutoff",
                "identifiability",
                "posterior_draws",
                "ppc_draws",
                "priors",
                "sbc_posterior_draws",
                "sbc_replicates",
                "schema_version",
                "seed",
                "simulation_count",
                "study_id",
                "summaries",
                "synthetic_only",
            }
        ),
        "calibration config",
    )
    priors: list[PriorSpec] = []
    for index, raw in enumerate(_array(mapping["priors"], "config.priors")):
        item = _object(raw, f"config.priors[{index}]")
        _fields(
            item,
            frozenset({"distribution", "lower", "name", "transform", "upper"}),
            f"config.priors[{index}]",
        )
        if item["distribution"] != "uniform":
            _fail("calibration prior distribution changed")
        transform = _object(item["transform"], "prior.transform")
        _fields(transform, frozenset({"kind"}), "prior.transform")
        try:
            transform_kind = TransformKind(_string(transform["kind"], "transform.kind"))
        except ValueError as error:
            raise CalibrationArtifactIntegrityError(
                "unknown parameter transform"
            ) from error
        priors.append(
            PriorSpec(
                name=_string(item["name"], "prior.name"),
                lower=_number(item["lower"], "prior.lower"),
                upper=_number(item["upper"], "prior.upper"),
                transform=ParameterTransform(transform_kind),
            )
        )
    summaries_mapping = _object(mapping["summaries"], "config.summaries")
    _fields(summaries_mapping, frozenset({"statistics"}), "config.summaries")
    summaries: list[SummaryStatisticSpec] = []
    for index, raw in enumerate(
        _array(summaries_mapping["statistics"], "config.summaries.statistics")
    ):
        item = _object(raw, f"config.summaries.statistics[{index}]")
        _fields(
            item,
            frozenset(
                {
                    "definition_sha256",
                    "level",
                    "name",
                    "role",
                    "source_design_sha256",
                    "weight",
                }
            ),
            f"config.summaries.statistics[{index}]",
        )
        try:
            level = SummaryLevel(_string(item["level"], "summary.level"))
            role = SummaryRole(_string(item["role"], "summary.role"))
        except ValueError as error:
            raise CalibrationArtifactIntegrityError(
                "unknown summary enum value"
            ) from error
        summaries.append(
            SummaryStatisticSpec(
                name=_string(item["name"], "summary.name"),
                level=level,
                role=role,
                definition_sha256=_string(
                    item["definition_sha256"], "summary.definition_sha256"
                ),
                source_design_sha256=_string(
                    item["source_design_sha256"],
                    "summary.source_design_sha256",
                ),
                weight=_number(item["weight"], "summary.weight"),
            )
        )
    thresholds_mapping = _object(mapping["identifiability"], "identifiability")
    _fields(
        thresholds_mapping,
        frozenset(
            {
                "boundary_fraction",
                "boundary_mass_threshold",
                "correlation_threshold",
                "histogram_bins",
                "minimum_contraction",
            }
        ),
        "identifiability",
    )
    cutoff = _string(mapping["cutoff"], "config.cutoff")
    try:
        parsed_cutoff = datetime.fromisoformat(cutoff)
    except ValueError as error:
        raise CalibrationArtifactIntegrityError("config.cutoff is invalid") from error
    return CalibrationConfig(
        schema_version=_integer(mapping["schema_version"], "config.schema_version"),
        study_id=_string(mapping["study_id"], "config.study_id"),
        cutoff=parsed_cutoff,
        seed=_string(mapping["seed"], "config.seed"),
        priors=tuple(priors),
        summaries=SummaryStatisticsConfig(tuple(summaries)),
        simulation_count=_integer(mapping["simulation_count"], "simulation_count"),
        accepted_particles=_integer(
            mapping["accepted_particles"], "accepted_particles"
        ),
        posterior_draws=_integer(mapping["posterior_draws"], "posterior_draws"),
        ppc_draws=_integer(mapping["ppc_draws"], "ppc_draws"),
        sbc_replicates=_integer(mapping["sbc_replicates"], "sbc_replicates"),
        sbc_posterior_draws=_integer(
            mapping["sbc_posterior_draws"], "sbc_posterior_draws"
        ),
        identifiability=IdentifiabilityThresholds(
            minimum_contraction=_number(
                thresholds_mapping["minimum_contraction"], "minimum_contraction"
            ),
            boundary_fraction=_number(
                thresholds_mapping["boundary_fraction"], "boundary_fraction"
            ),
            boundary_mass_threshold=_number(
                thresholds_mapping["boundary_mass_threshold"],
                "boundary_mass_threshold",
            ),
            correlation_threshold=_number(
                thresholds_mapping["correlation_threshold"],
                "correlation_threshold",
            ),
            histogram_bins=_integer(
                thresholds_mapping["histogram_bins"], "histogram_bins"
            ),
        ),
        synthetic_only=_boolean(mapping["synthetic_only"], "synthetic_only"),
    )


def _parse_input(mapping: dict[str, object]) -> CalibrationInput:
    _fields(mapping, frozenset({"observed", "sources", "synthetic_only"}), "input")
    sources: list[SummarySourcePartition] = []
    for index, raw in enumerate(_array(mapping["sources"], "input.sources")):
        item = _object(raw, f"input.sources[{index}]")
        _fields(
            item,
            frozenset(
                {
                    "event_keys",
                    "observed",
                    "records",
                    "role",
                    "source_design_sha256",
                    "source_record_set_sha256",
                    "transform_lineage",
                }
            ),
            f"input.sources[{index}]",
        )
        try:
            role = SummaryRole(_string(item["role"], "source.role"))
        except ValueError as error:
            raise CalibrationArtifactIntegrityError(
                "unknown input source role"
            ) from error
        records = tuple(
            TemporalRecord.from_mapping(_object(record, "source record"))
            for record in _array(item["records"], "source.records")
        )
        lineage_mapping = _object(item["transform_lineage"], "source.transform_lineage")
        _fields(
            lineage_mapping,
            frozenset({"nodes", "root_node_id"}),
            "source.transform_lineage",
        )
        lineage_nodes: list[TransformLineageNode] = []
        for node_index, raw_node in enumerate(
            _array(lineage_mapping["nodes"], "source.transform_lineage.nodes")
        ):
            node = _object(
                raw_node,
                f"source.transform_lineage.nodes[{node_index}]",
            )
            _fields(
                node,
                frozenset(
                    {
                        "event_keys",
                        "implementation_sha256",
                        "node_id",
                        "operation",
                        "parent_node_ids",
                        "source_record_ids",
                    }
                ),
                f"source.transform_lineage.nodes[{node_index}]",
            )
            lineage_nodes.append(
                TransformLineageNode(
                    node_id=_string(node["node_id"], "lineage.node_id"),
                    operation=_string(node["operation"], "lineage.operation"),
                    implementation_sha256=_string(
                        node["implementation_sha256"],
                        "lineage.implementation_sha256",
                    ),
                    parent_node_ids=_strings(
                        node["parent_node_ids"], "lineage.parent_node_ids"
                    ),
                    source_record_ids=_strings(
                        node["source_record_ids"], "lineage.source_record_ids"
                    ),
                    event_keys=_strings(node["event_keys"], "lineage.event_keys"),
                )
            )
        transform_lineage = TransformLineage(
            root_node_id=_string(
                lineage_mapping["root_node_id"], "lineage.root_node_id"
            ),
            nodes=tuple(lineage_nodes),
        )
        sources.append(
            SummarySourcePartition(
                role=role,
                observed=_parse_summary_vector(item["observed"], "source.observed"),
                source_design_sha256=_string(
                    item["source_design_sha256"],
                    "source.source_design_sha256",
                ),
                source_record_set_sha256=_string(
                    item["source_record_set_sha256"],
                    "source.source_record_set_sha256",
                ),
                records=records,
                event_keys=_strings(item["event_keys"], "source.event_keys"),
                transform_lineage=transform_lineage,
            )
        )
    return CalibrationInput(
        observed=_parse_summary_vector(mapping["observed"], "input.observed"),
        sources=tuple(sources),
        synthetic_only=_boolean(mapping["synthetic_only"], "input.synthetic_only"),
    )


def _parse_posterior(mapping: dict[str, object]) -> PosteriorSamples:
    _fields(mapping, frozenset({"backend_id", "draws", "particles"}), "posterior")
    particles: list[Particle] = []
    for raw in _array(mapping["particles"], "posterior.particles"):
        item = _object(raw, "posterior particle")
        _fields(
            item,
            frozenset({"distance", "parameters", "simulation_event_key"}),
            "posterior particle",
        )
        particles.append(
            Particle(
                parameters=parameter_mapping(
                    {
                        name: _number(value, f"particle.parameters.{name}")
                        for name, value in _object(
                            item["parameters"], "particle.parameters"
                        ).items()
                    }
                ),
                distance=_number(item["distance"], "particle.distance"),
                simulation_event_key=_string(
                    item["simulation_event_key"], "particle.simulation_event_key"
                ),
            )
        )
    draws: list[PosteriorDraw] = []
    for raw in _array(mapping["draws"], "posterior.draws"):
        item = _object(raw, "posterior draw")
        _fields(
            item,
            frozenset({"event_key", "parameters", "source_particle"}),
            "posterior draw",
        )
        draws.append(
            PosteriorDraw(
                parameters=parameter_mapping(
                    {
                        name: _number(value, f"draw.parameters.{name}")
                        for name, value in _object(
                            item["parameters"], "draw.parameters"
                        ).items()
                    }
                ),
                source_particle=_integer(
                    item["source_particle"], "draw.source_particle"
                ),
                event_key=_string(item["event_key"], "draw.event_key"),
            )
        )
    return PosteriorSamples(
        backend_id=_string(mapping["backend_id"], "posterior.backend_id"),
        particles=tuple(particles),
        draws=tuple(draws),
    )


def _parse_ppc(mapping: dict[str, object]) -> PosteriorPredictiveChecks:
    _fields(mapping, frozenset({"event_keys", "statistics"}), "PPC")
    statistics: list[PosteriorPredictiveStatistic] = []
    for raw in _array(mapping["statistics"], "PPC.statistics"):
        item = _object(raw, "PPC statistic")
        _fields(
            item,
            frozenset(
                {
                    "covered",
                    "lower_90",
                    "median",
                    "name",
                    "observed",
                    "predictive_draws",
                    "role",
                    "upper_90",
                }
            ),
            "PPC statistic",
        )
        try:
            role = SummaryRole(_string(item["role"], "PPC role"))
        except ValueError as error:
            raise CalibrationArtifactIntegrityError("unknown PPC role") from error
        statistics.append(
            PosteriorPredictiveStatistic(
                name=_string(item["name"], "PPC name"),
                role=role,
                observed=_number(item["observed"], "PPC observed"),
                predictive_draws=tuple(
                    _number(value, "PPC predictive draw")
                    for value in _array(
                        item["predictive_draws"], "PPC predictive_draws"
                    )
                ),
                lower_90=_number(item["lower_90"], "PPC lower_90"),
                median=_number(item["median"], "PPC median"),
                upper_90=_number(item["upper_90"], "PPC upper_90"),
                covered=_boolean(item["covered"], "PPC covered"),
            )
        )
    return PosteriorPredictiveChecks(
        statistics=tuple(statistics),
        event_keys=_strings(mapping["event_keys"], "PPC.event_keys"),
    )


def _parse_recovery(
    value: object, field: str = "recovery"
) -> tuple[RecoveryMetric, ...]:
    results: list[RecoveryMetric] = []
    for raw in _array(value, field):
        item = _object(raw, "recovery metric")
        _fields(
            item,
            frozenset(
                {
                    "absolute_error",
                    "covered",
                    "lower_90",
                    "parameter",
                    "posterior_median",
                    "truth",
                    "upper_90",
                }
            ),
            "recovery metric",
        )
        results.append(
            RecoveryMetric(
                parameter=_string(item["parameter"], "recovery.parameter"),
                truth=_number(item["truth"], "recovery.truth"),
                posterior_median=_number(
                    item["posterior_median"], "recovery.posterior_median"
                ),
                absolute_error=_number(
                    item["absolute_error"], "recovery.absolute_error"
                ),
                lower_90=_number(item["lower_90"], "recovery.lower_90"),
                upper_90=_number(item["upper_90"], "recovery.upper_90"),
                covered=_boolean(item["covered"], "recovery.covered"),
            )
        )
    return tuple(results)


_IDENTIFIABILITY_FIELDS = frozenset(
    {
        "boundary_pileup",
        "confounded",
        "contraction",
        "estimable_for_reporting",
        "histogram_counts",
        "lower_boundary_mass",
        "maximum_absolute_correlation",
        "mode_count",
        "multimodal",
        "parameter",
        "posterior_standard_deviation",
        "prior_standard_deviation",
        "upper_boundary_mass",
    }
)


def _parse_identifiability(value: object) -> tuple[ParameterIdentifiability, ...]:
    results: list[ParameterIdentifiability] = []
    for raw in _array(value, "identifiability"):
        item = _object(raw, "identifiability diagnostic")
        _fields(item, _IDENTIFIABILITY_FIELDS, "identifiability diagnostic")
        results.append(
            ParameterIdentifiability(
                parameter=_string(item["parameter"], "identifiability.parameter"),
                posterior_standard_deviation=_number(
                    item["posterior_standard_deviation"],
                    "posterior_standard_deviation",
                ),
                prior_standard_deviation=_number(
                    item["prior_standard_deviation"], "prior_standard_deviation"
                ),
                contraction=_number(item["contraction"], "contraction"),
                lower_boundary_mass=_number(
                    item["lower_boundary_mass"], "lower_boundary_mass"
                ),
                upper_boundary_mass=_number(
                    item["upper_boundary_mass"], "upper_boundary_mass"
                ),
                boundary_pileup=_boolean(item["boundary_pileup"], "boundary_pileup"),
                histogram_counts=tuple(
                    _integer(count, "histogram count")
                    for count in _array(item["histogram_counts"], "histogram_counts")
                ),
                mode_count=_integer(item["mode_count"], "mode_count"),
                multimodal=_boolean(item["multimodal"], "multimodal"),
                maximum_absolute_correlation=_number(
                    item["maximum_absolute_correlation"],
                    "maximum_absolute_correlation",
                ),
                confounded=_boolean(item["confounded"], "confounded"),
                estimable_for_reporting=_boolean(
                    item["estimable_for_reporting"], "estimable_for_reporting"
                ),
            )
        )
    return tuple(results)


def _parse_sbc(mapping: dict[str, object]) -> SimulationBasedCalibration:
    _fields(
        mapping,
        frozenset({"event_keys", "parameters", "replicates", "synthetic_only"}),
        "SBC",
    )
    if _boolean(mapping["synthetic_only"], "SBC.synthetic_only") is not True:
        _fail("SBC must remain synthetic-only")
    parameters: list[SBCRankHistogram] = []
    for raw in _array(mapping["parameters"], "SBC.parameters"):
        item = _object(raw, "SBC parameter")
        _fields(
            item,
            frozenset(
                {
                    "histogram",
                    "interval_coverage_90",
                    "parameter",
                    "posterior_draws_per_replicate",
                    "ranks",
                }
            ),
            "SBC parameter",
        )
        parameters.append(
            SBCRankHistogram(
                parameter=_string(item["parameter"], "SBC.parameter"),
                posterior_draws_per_replicate=_integer(
                    item["posterior_draws_per_replicate"],
                    "SBC.posterior_draws_per_replicate",
                ),
                ranks=tuple(
                    _integer(rank, "SBC rank")
                    for rank in _array(item["ranks"], "SBC.ranks")
                ),
                histogram=tuple(
                    _integer(count, "SBC histogram count")
                    for count in _array(item["histogram"], "SBC.histogram")
                ),
                interval_coverage_90=_number(
                    item["interval_coverage_90"], "SBC.interval_coverage_90"
                ),
            )
        )
    return SimulationBasedCalibration(
        replicates=_integer(mapping["replicates"], "SBC.replicates"),
        parameters=tuple(parameters),
        event_keys=_strings(mapping["event_keys"], "SBC.event_keys"),
    )


_REPORT_FIELDS = frozenset(
    {
        "calibration_statistics",
        "claim_scope",
        "completion_status",
        "config_sha256",
        "estimator_backend",
        "event_key_count",
        "event_keys_sha256",
        "held_out_statistics",
        "identifiability",
        "input_sha256",
        "posterior_draw_count",
        "posterior_particle_count",
        "posterior_predictive_checks_sha256",
        "posterior_sha256",
        "ppc_held_out_coverage",
        "protocol_execution_receipt_sha256",
        "rank_histogram_sha256",
        "real_data_status",
        "recovery",
        "sbc_interval_coverage_90",
        "schema_version",
        "simulator_config_sha256",
        "simulator_id",
        "study_claim_scope",
        "study_protocol_sha256",
        "synthetic_only",
        "true_parameters_sha256",
        "warnings",
    }
)


def _parse_report(
    mapping: dict[str, object], config: CalibrationConfig
) -> CalibrationReport:
    _fields(mapping, _REPORT_FIELDS, "calibration report")
    if _boolean(mapping["synthetic_only"], "report.synthetic_only") is not True:
        _fail("calibration report must remain synthetic-only")
    sbc_coverage = _object(
        mapping["sbc_interval_coverage_90"], "report.sbc_interval_coverage_90"
    )
    expected_parameters = tuple(prior.name for prior in config.priors)
    if set(sbc_coverage) != set(expected_parameters):
        _fail("report SBC coverage parameters changed")
    return CalibrationReport(
        schema_version=_integer(mapping["schema_version"], "report.schema_version"),
        study_protocol_sha256=_string(
            mapping["study_protocol_sha256"], "report.study_protocol_sha256"
        ),
        study_claim_scope=_string(
            mapping["study_claim_scope"], "report.study_claim_scope"
        ),
        protocol_execution_receipt_sha256=_string(
            mapping["protocol_execution_receipt_sha256"],
            "report.protocol_execution_receipt_sha256",
        ),
        config_sha256=_string(mapping["config_sha256"], "report.config_sha256"),
        input_sha256=_string(mapping["input_sha256"], "report.input_sha256"),
        simulator_id=_string(mapping["simulator_id"], "report.simulator_id"),
        simulator_config_sha256=_string(
            mapping["simulator_config_sha256"], "report.simulator_config_sha256"
        ),
        true_parameters_sha256=_string(
            mapping["true_parameters_sha256"], "report.true_parameters_sha256"
        ),
        estimator_backend=_string(
            mapping["estimator_backend"], "report.estimator_backend"
        ),
        completion_status=_string(
            mapping["completion_status"], "report.completion_status"
        ),
        claim_scope=_string(mapping["claim_scope"], "report.claim_scope"),
        real_data_status=_string(
            mapping["real_data_status"], "report.real_data_status"
        ),
        calibration_statistics=_strings(
            mapping["calibration_statistics"], "report.calibration_statistics"
        ),
        held_out_statistics=_strings(
            mapping["held_out_statistics"], "report.held_out_statistics"
        ),
        posterior_particle_count=_integer(
            mapping["posterior_particle_count"], "report.posterior_particle_count"
        ),
        posterior_draw_count=_integer(
            mapping["posterior_draw_count"], "report.posterior_draw_count"
        ),
        posterior_sha256=_string(
            mapping["posterior_sha256"], "report.posterior_sha256"
        ),
        posterior_predictive_checks_sha256=_string(
            mapping["posterior_predictive_checks_sha256"],
            "report.posterior_predictive_checks_sha256",
        ),
        rank_histogram_sha256=_string(
            mapping["rank_histogram_sha256"], "report.rank_histogram_sha256"
        ),
        event_key_count=_integer(mapping["event_key_count"], "report.event_key_count"),
        event_keys_sha256=_string(
            mapping["event_keys_sha256"], "report.event_keys_sha256"
        ),
        recovery=_parse_recovery(mapping["recovery"]),
        ppc_held_out_coverage=_number(
            mapping["ppc_held_out_coverage"], "report.ppc_held_out_coverage"
        ),
        sbc_interval_coverage_90=tuple(
            (name, _number(sbc_coverage[name], f"SBC coverage {name}"))
            for name in expected_parameters
        ),
        identifiability=_parse_identifiability(mapping["identifiability"]),
        warnings=_strings(mapping["warnings"], "report.warnings"),
    )


_MAX_BUNDLE_FILE_BYTES = 128 * 1024 * 1024
_MAX_BUNDLE_TOTAL_BYTES = 512 * 1024 * 1024


def _fingerprint(info: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _read_exact_fd(fd: int, *, expected_size: int, name: str) -> bytes:
    if not 0 <= expected_size <= _MAX_BUNDLE_FILE_BYTES:
        _fail(f"calibration artifact size is invalid: {name}")
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        observed = 0
        while True:
            chunk = os.read(fd, min(1024 * 1024, expected_size + 1 - observed))
            if not chunk:
                break
            chunks.append(chunk)
            observed += len(chunk)
            if observed > expected_size:
                _fail(f"calibration artifact changed while read: {name}")
    except OSError as error:
        raise CalibrationArtifactIntegrityError(
            f"cannot read calibration artifact {name}: {error}"
        ) from error
    if observed != expected_size:
        _fail(f"calibration artifact changed while read: {name}")
    return b"".join(chunks)


@dataclass(slots=True)
class _CapturedFile:
    name: str
    fd: int
    payload: bytes
    fingerprint: tuple[int, int, int, int, int, int, int]


@dataclass(slots=True)
class _CapturedBundle:
    directory: Path
    directory_fd: int
    directory_fingerprint: tuple[int, int, int, int, int, int, int]
    expected_files: frozenset[str]
    files: dict[str, _CapturedFile]

    def close(self) -> None:
        for item in self.files.values():
            os.close(item.fd)
        os.close(self.directory_fd)


def _capture_bundle(directory: Path) -> _CapturedBundle:
    expected_files = _BUNDLE_FILES | {"manifest.json"}
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    directory_flags = flags | getattr(os, "O_DIRECTORY", 0)
    try:
        directory_fd = os.open(directory, directory_flags)
    except OSError as error:
        raise CalibrationArtifactIntegrityError(
            f"cannot open calibration artifact directory: {error}"
        ) from error
    files: dict[str, _CapturedFile] = {}
    captured_total_bytes = 0
    try:
        directory_info = os.fstat(directory_fd)
        if not stat.S_ISDIR(directory_info.st_mode):
            _fail("calibration artifact must be a regular directory")
        observed_files = frozenset(os.listdir(directory_fd))
        if observed_files != expected_files:
            _fail(
                "calibration artifact file set changed: "
                f"missing={sorted(expected_files - observed_files)!r}, "
                f"extra={sorted(observed_files - expected_files)!r}"
            )
        for name in sorted(expected_files):
            try:
                fd = os.open(name, flags, dir_fd=directory_fd)
            except OSError as error:
                raise CalibrationArtifactIntegrityError(
                    f"cannot open calibration artifact {name}: {error}"
                ) from error
            try:
                before = os.fstat(fd)
                if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                    _fail(
                        "calibration artifact must be a singly linked regular "
                        f"file: {name}"
                    )
                if (
                    before.st_size <= 0
                    or before.st_size > _MAX_BUNDLE_FILE_BYTES
                    or captured_total_bytes + before.st_size > _MAX_BUNDLE_TOTAL_BYTES
                ):
                    _fail(
                        "calibration artifact is empty or exceeds its file/total "
                        f"byte limit: {name}"
                    )
                fingerprint = _fingerprint(before)
                payload = _read_exact_fd(fd, expected_size=before.st_size, name=name)
                if _fingerprint(os.fstat(fd)) != fingerprint:
                    _fail(f"calibration artifact changed while captured: {name}")
                files[name] = _CapturedFile(name, fd, payload, fingerprint)
                captured_total_bytes += len(payload)
            except Exception:
                os.close(fd)
                raise
        if (
            _fingerprint(os.fstat(directory_fd)) != _fingerprint(directory_info)
            or frozenset(os.listdir(directory_fd)) != expected_files
        ):
            _fail("calibration artifact directory changed while captured")
        return _CapturedBundle(
            directory=directory,
            directory_fd=directory_fd,
            directory_fingerprint=_fingerprint(directory_info),
            expected_files=expected_files,
            files=files,
        )
    except Exception:
        for item in files.values():
            os.close(item.fd)
        os.close(directory_fd)
        raise


def _require_current_bundle(captured: _CapturedBundle) -> None:
    if frozenset(os.listdir(captured.directory_fd)) != captured.expected_files:
        _fail("calibration artifact file set changed during audit")
    for name, item in captured.files.items():
        try:
            path_info = os.stat(
                name, dir_fd=captured.directory_fd, follow_symlinks=False
            )
            fd_info = os.fstat(item.fd)
        except OSError as error:
            raise CalibrationArtifactIntegrityError(
                f"cannot recheck calibration artifact {name}: {error}"
            ) from error
        if (
            _fingerprint(path_info) != item.fingerprint
            or _fingerprint(fd_info) != item.fingerprint
            or _read_exact_fd(item.fd, expected_size=fd_info.st_size, name=name)
            != item.payload
        ):
            _fail(f"calibration artifact path changed during audit: {name}")
    try:
        path_directory = os.stat(captured.directory, follow_symlinks=False)
        fd_directory = os.fstat(captured.directory_fd)
    except OSError as error:
        raise CalibrationArtifactIntegrityError(
            f"cannot finalize calibration artifact directory: {error}"
        ) from error
    if (
        _fingerprint(path_directory) != captured.directory_fingerprint
        or _fingerprint(fd_directory) != captured.directory_fingerprint
        or frozenset(os.listdir(captured.directory_fd)) != captured.expected_files
    ):
        _fail("calibration artifact directory identity changed during audit")


def _payload_json(
    payloads: dict[str, bytes],
    name: str,
    *,
    source: str,
) -> dict[str, object]:
    return read_json_bytes_object(
        payloads[name],
        source=f"{source}/{name}",
    )


def _verify_calibration_artifact_payloads(
    payloads: dict[str, bytes],
    *,
    source: str,
    expected_report_sha256: str | None = None,
    expected_manifest_sha256: str | None = None,
) -> CalibrationArtifactLoadResult:
    """Replay one exact byte snapshot without reopening filesystem paths."""
    expected_files = _BUNDLE_FILES | {"manifest.json"}
    if type(payloads) is not dict or frozenset(payloads) != expected_files:
        _fail("calibration byte snapshot file set changed")
    if any(
        type(name) is not str
        or type(payload) is not bytes
        or len(payload) == 0
        or len(payload) > _MAX_BUNDLE_FILE_BYTES
        for name, payload in payloads.items()
    ):
        _fail("calibration byte snapshot contains an invalid payload")
    if sum(len(payload) for payload in payloads.values()) > _MAX_BUNDLE_TOTAL_BYTES:
        _fail("calibration byte snapshot exceeds its total byte limit")
    try:
        manifest_raw_sha256 = sha256_bytes(payloads["manifest.json"])
        if (
            expected_manifest_sha256 is not None
            and manifest_raw_sha256 != expected_manifest_sha256
        ):
            _fail(
                "calibration manifest bytes do not match the caller-supplied commitment"
            )
        try:
            manifest = _payload_json(payloads, "manifest.json", source=source)
            config_mapping = _payload_json(payloads, "config.json", source=source)
            input_mapping = _payload_json(payloads, "input_receipt.json", source=source)
            receipt_mapping = _payload_json(
                payloads, "protocol_execution_receipt.json", source=source
            )
            simulator_mapping = _payload_json(
                payloads, "simulator_config.json", source=source
            )
            truth_mapping = _payload_json(
                payloads, "true_parameters.json", source=source
            )
            posterior_mapping = _payload_json(
                payloads, "posterior_samples.json", source=source
            )
            ppc_mapping = _payload_json(
                payloads, "posterior_predictive_checks.json", source=source
            )
            sbc_mapping = _payload_json(payloads, "rank_histogram.json", source=source)
            report_mapping = _payload_json(
                payloads, "calibration_report.json", source=source
            )
            provenance = _payload_json(payloads, "provenance.json", source=source)
            event_keys_mapping = _payload_json(
                payloads, "event_keys.json", source=source
            )
        except MetadataValidationError as error:
            raise CalibrationArtifactIntegrityError(str(error)) from error

        _fields(
            manifest,
            frozenset(
                {
                    "artifacts",
                    "claim_scope",
                    "config_sha256",
                    "input_sha256",
                    "protocol_execution_receipt_sha256",
                    "report_sha256",
                    "schema_version",
                    "simulator_config_sha256",
                    "study_protocol_sha256",
                    "synthetic_only",
                    "true_parameters_sha256",
                }
            ),
            "calibration manifest",
        )
        if (
            _integer(manifest["schema_version"], "manifest.schema_version") != 4
            or _boolean(manifest["synthetic_only"], "manifest.synthetic_only")
            is not True
        ):
            _fail("calibration manifest schema or evidence scope is invalid")
        committed: set[str] = set()
        for raw in _array(manifest["artifacts"], "manifest.artifacts"):
            item = _object(raw, "manifest artifact")
            _fields(
                item,
                frozenset({"path", "sha256", "size_bytes"}),
                "manifest artifact",
            )
            name = _string(item["path"], "manifest artifact path")
            if name not in _BUNDLE_FILES or name in committed:
                _fail("manifest artifact path is invalid or duplicated")
            if _string(item["sha256"], "manifest artifact sha256") != sha256_bytes(
                payloads[name]
            ) or _integer(item["size_bytes"], "manifest artifact size_bytes") != len(
                payloads[name]
            ):
                _fail(f"calibration artifact hash or size mismatch: {name}")
            committed.add(name)
        if committed != set(_BUNDLE_FILES):
            _fail("manifest does not commit to every calibration file")

        try:
            config = _parse_config(config_mapping)
            observed = _parse_input(input_mapping)
            observed.validate_for(config)
            receipt = ProtocolExecutionReceipt.from_mapping(receipt_mapping)
            posterior = _parse_posterior(posterior_mapping)
            ppc = _parse_ppc(ppc_mapping)
            sbc = _parse_sbc(sbc_mapping)
            report = _parse_report(report_mapping, config)
        except (
            CalibrationError,
            ProtocolValidationError,
            TemporalFirewallError,
        ) as error:
            raise CalibrationArtifactIntegrityError(
                f"invalid typed calibration artifact: {error}"
            ) from error

        if (
            receipt.consumer is not ProtocolExecutionConsumer.CALIBRATION_WORKFLOW
            or receipt.purpose is not ProtocolExecutionPurpose.PRE_POLICY_CALIBRATION
            or receipt.study_mode is not StudyMode.SYNTHETIC
            or receipt.claim_scope is not ClaimScope.SYNTHETIC_ONLY
            or receipt.issuer_kind != "synthetic_fixture_issuer"
            or receipt.study_id != config.study_id
            or receipt.knowledge_cutoff != config.cutoff
            or receipt.config_sha256 != config.sha256
            or receipt.input_sha256 != observed.sha256
        ):
            _fail(
                "calibration receipt does not bind the exact synthetic workflow, "
                "study, cutoff, config, and input"
            )

        _fields(
            truth_mapping,
            frozenset({"parameters", "schema_version", "synthetic_only"}),
            "true parameters",
        )
        if (
            _integer(truth_mapping["schema_version"], "truth.schema_version") != 1
            or _boolean(truth_mapping["synthetic_only"], "truth.synthetic_only")
            is not True
        ):
            _fail("synthetic truth schema changed")
        raw_truth = _object(truth_mapping["parameters"], "true parameters")
        true_parameters = parameter_mapping(
            {
                name: _number(value, f"true parameter {name}")
                for name, value in raw_truth.items()
            }
        )
        _fields(
            event_keys_mapping,
            frozenset({"event_keys", "schema_version"}),
            "event-key manifest",
        )
        if (
            _integer(
                event_keys_mapping["schema_version"],
                "event-key manifest schema_version",
            )
            != 1
        ):
            _fail("event-key manifest schema changed")
        event_keys = _strings(event_keys_mapping["event_keys"], "event keys")

        simulator_config_json = canonical_json_bytes(simulator_mapping)
        try:
            typed_run = CalibrationRun(
                config=config,
                observed=observed,
                protocol_execution_receipt=receipt,
                simulator_config_json=simulator_config_json,
                true_parameters=true_parameters,
                posterior=posterior,
                ppc=ppc,
                recovery=report.recovery,
                sbc=sbc,
                identifiability=report.identifiability,
                report=report,
                event_keys=event_keys,
            )
        except CalibrationError as error:
            raise CalibrationArtifactIntegrityError(
                f"calibration semantic replay failed: {error}"
            ) from error

        report_sha256 = report.sha256
        config_sha256 = config.sha256
        input_sha256 = observed.sha256
        simulator_config_sha256 = sha256_bytes(simulator_config_json)
        protocol_receipt_sha256 = receipt.sha256
        if (
            _string(manifest["claim_scope"], "manifest.claim_scope")
            != report.claim_scope
            or manifest["config_sha256"] != config_sha256
            or manifest["input_sha256"] != input_sha256
            or manifest["study_protocol_sha256"] != receipt.protocol_sha256
            or manifest["protocol_execution_receipt_sha256"] != protocol_receipt_sha256
            or manifest["report_sha256"] != report_sha256
            or manifest["simulator_config_sha256"] != simulator_config_sha256
            or manifest["true_parameters_sha256"] != typed_run.true_parameters_sha256
        ):
            _fail("calibration manifest semantic commitments mismatch")
        if (
            expected_report_sha256 is not None
            and report_sha256 != expected_report_sha256
        ):
            _fail("calibration report does not match the externally expected digest")
        _fields(
            provenance,
            frozenset(
                {
                    "claim_scope",
                    "config_sha256",
                    "environment",
                    "event_keys",
                    "git",
                    "git_identity_attestation",
                    "input_sha256",
                    "model_checkpoint_sha256",
                    "protocol_execution_receipt_sha256",
                    "schema_version",
                    "simulator_id",
                    "simulator_config_sha256",
                    "study_protocol_sha256",
                    "synthetic_only",
                    "true_parameters_sha256",
                }
            ),
            "calibration provenance",
        )
        try:
            GitMetadata.from_mapping(_object(provenance["git"], "provenance.git"))
            EnvironmentMetadata.from_mapping(
                _object(provenance["environment"], "provenance.environment")
            )
        except MetadataValidationError as error:
            raise CalibrationArtifactIntegrityError(str(error)) from error
        if (
            _integer(provenance["schema_version"], "provenance.schema_version") != 3
            or _boolean(provenance["synthetic_only"], "provenance.synthetic_only")
            is not True
            or _string(
                provenance["git_identity_attestation"],
                "provenance.git_identity_attestation",
            )
            not in _GIT_IDENTITY_ATTESTATIONS
            or provenance["model_checkpoint_sha256"] is not None
            or _string(provenance["claim_scope"], "provenance.claim_scope")
            != report.claim_scope
            or _string(provenance["simulator_id"], "provenance.simulator_id")
            != report.simulator_id
            or _string(provenance["config_sha256"], "provenance.config_sha256")
            != config_sha256
            or _string(provenance["input_sha256"], "provenance.input_sha256")
            != input_sha256
            or _string(
                provenance["simulator_config_sha256"],
                "provenance.simulator_config_sha256",
            )
            != simulator_config_sha256
            or _string(
                provenance["true_parameters_sha256"],
                "provenance.true_parameters_sha256",
            )
            != typed_run.true_parameters_sha256
            or _string(
                provenance["study_protocol_sha256"],
                "provenance.study_protocol_sha256",
            )
            != receipt.protocol_sha256
            or _string(
                provenance["protocol_execution_receipt_sha256"],
                "provenance.protocol_execution_receipt_sha256",
            )
            != protocol_receipt_sha256
        ):
            _fail("calibration provenance commitments mismatch")
        event_raw = _object(provenance["event_keys"], "provenance.event_keys")
        try:
            event_metadata = EventKeyMetadata.from_mapping(event_raw)
        except MetadataValidationError as error:
            raise CalibrationArtifactIntegrityError(str(error)) from error
        expected_event_sha256 = sha256_bytes(canonical_json_bytes(list(event_keys)))
        if (
            event_metadata.schema_version != "policytwin.event-key.v1"
            or event_metadata.root_key != config.seed
            or event_metadata.keys != event_keys
            or event_metadata.count != len(event_keys)
            or event_metadata.sha256 != expected_event_sha256
            or report.event_key_count != len(event_keys)
            or report.event_keys_sha256 != expected_event_sha256
        ):
            _fail("calibration report/event-key commitment mismatch")
        expected_audit = _canonical_jsonl(_calibration_audit_events(typed_run))
        if payloads["audit.jsonl"] != expected_audit:
            _fail(
                "calibration audit log does not match the complete replay-derived "
                "event registry"
            )

        audit = CalibrationArtifactAuditReport(
            valid=True,
            report_sha256=report_sha256,
            config_sha256=config_sha256,
            input_sha256=input_sha256,
            study_protocol_sha256=receipt.protocol_sha256,
            protocol_execution_receipt_sha256=protocol_receipt_sha256,
            simulator_config_sha256=simulator_config_sha256,
            true_parameters_sha256=typed_run.true_parameters_sha256,
            event_keys_sha256=event_metadata.sha256,
            event_key_count=event_metadata.count,
            git_identity_attestation=cast(
                str, provenance["git_identity_attestation"]
            ),
            checked_files=tuple(sorted(expected_files)),
        )
        return CalibrationArtifactLoadResult(
            run=typed_run,
            audit=audit,
            manifest_raw_sha256=manifest_raw_sha256,
        )
    except CalibrationArtifactIntegrityError:
        raise
    except (
        CalibrationError,
        MetadataValidationError,
        ProtocolValidationError,
        TemporalFirewallError,
    ) as error:
        raise CalibrationArtifactIntegrityError(
            f"invalid calibration artifact: {error}"
        ) from error


def _load_and_audit_calibration_artifact(
    directory: Path,
    *,
    expected_report_sha256: str | None = None,
    expected_manifest_sha256: str | None = None,
) -> CalibrationArtifactLoadResult:
    """Capture one stable bundle and replay every built-in synthetic output."""
    captured = _capture_bundle(directory)
    try:
        result = _verify_calibration_artifact_payloads(
            {name: item.payload for name, item in captured.files.items()},
            source=str(directory),
            expected_report_sha256=expected_report_sha256,
            expected_manifest_sha256=expected_manifest_sha256,
        )
        _require_current_bundle(captured)
        return result
    finally:
        captured.close()


def audit_calibration_artifact(
    directory: Path,
    *,
    expected_report_sha256: str | None = None,
) -> CalibrationArtifactAuditReport:
    """Audit a stable byte snapshot and replay every built-in synthetic output."""
    return _load_and_audit_calibration_artifact(
        directory,
        expected_report_sha256=expected_report_sha256,
    ).audit


def load_calibration_artifact(
    directory: Path,
    *,
    expected_manifest_sha256: str,
) -> CalibrationArtifactLoadResult:
    """Load and replay a bundle relative to a caller's raw-manifest commitment.

    ``expected_manifest_sha256`` is the SHA-256 of the exact ``manifest.json``
    bytes, not the manifest's self-declared semantic hash.  Possession of that
    digest by this process is not evidence of independent or external custody.
    """
    if (
        type(expected_manifest_sha256) is not str
        or len(expected_manifest_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected_manifest_sha256
        )
    ):
        _fail("expected_manifest_sha256 must be a lowercase SHA-256 digest")
    return _load_and_audit_calibration_artifact(
        directory,
        expected_manifest_sha256=expected_manifest_sha256,
    )


def verify_calibration_artifact_snapshot(
    payloads: dict[str, bytes],
    *,
    expected_manifest_sha256: str,
) -> CalibrationArtifactLoadResult:
    """Replay an already captured complete calibration byte snapshot.

    Filesystem stability is the caller's responsibility at this boundary.  The
    function verifies exact membership, bounded byte payloads, the caller's raw
    manifest digest, every manifest leaf commitment, and the full deterministic
    synthetic calibration closure without reopening a path.  The supplied
    digest remains relative evidence and is not external custody or attestation.
    """
    if type(payloads) is not dict or not all(
        type(name) is str and type(payload) is bytes
        for name, payload in payloads.items()
    ):
        _fail("calibration snapshot must map file names to exact bytes")
    if (
        type(expected_manifest_sha256) is not str
        or len(expected_manifest_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected_manifest_sha256
        )
    ):
        _fail("expected_manifest_sha256 must be a lowercase SHA-256 digest")
    return _verify_calibration_artifact_payloads(
        payloads.copy(),
        source="captured calibration snapshot",
        expected_manifest_sha256=expected_manifest_sha256,
    )


__all__ = [
    "CALLER_SUPPLIED_GIT_IDENTITY_ATTESTATION",
    "LOCAL_GIT_IDENTITY_ATTESTATION",
    "CalibrationArtifactAuditReport",
    "CalibrationArtifactFailureReceipt",
    "CalibrationArtifactIntegrityError",
    "CalibrationArtifactLoadResult",
    "CalibrationArtifactWriteError",
    "audit_calibration_artifact",
    "build_calibration_artifact_payloads",
    "load_calibration_artifact",
    "verify_calibration_artifact_snapshot",
    "write_calibration_artifact",
]
