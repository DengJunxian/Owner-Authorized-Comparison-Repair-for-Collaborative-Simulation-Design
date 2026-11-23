"""Verifier-owned execution receipts for closed synthetic reporting inputs.

The receipt implemented here is deliberately narrow.  It can certify only the
bundled deterministic synthetic runner and it replays that runner during every
verification.  It is therefore evidence for the mechanics of a closed-input
execution path, not a signature, an empirical-data attestation, or authority
for a non-synthetic claim.
"""

from __future__ import annotations

import inspect
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import NoReturn, Self, cast

from policytwin.counterfactual.randomness import EventKey
from policytwin.experiments.matrix import (
    ExperimentMatrix,
    ExperimentResult,
    ExperimentRow,
    ReplicateRunner,
    matrix_replicate_event_key,
)
from policytwin.protocol import (
    ProtocolValidationError,
    StudyMode,
    StudyProtocol,
    load_study_protocol,
)
from policytwin.provenance import (
    MetadataValidationError,
    canonical_json_bytes,
    read_json_object,
    sha256_bytes,
    sha256_file,
    write_json_file,
    write_text_file,
)

EXECUTION_RECEIPT_SCHEMA_VERSION = "policytwin.synthetic-execution-receipt.v2"
EXECUTION_RECEIPT_AUTHORITY_ID = "policytwin.synthetic-replay-verifier.v1"
VERIFIER_OWNED_RUNNER_KIND = "typed_reporting_source_v1"
SHARED_DRAW_EVENT_KEY_MANIFEST_SCHEMA_VERSION = (
    "policytwin.shared-draw-event-key-manifest.v1"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


class ExecutionReceiptError(RuntimeError):
    """Base class for execution-receipt failures."""


class ExecutionReceiptSchemaError(ExecutionReceiptError, ValueError):
    """Raised when a receipt or verifier configuration is ambiguous."""


class ExecutionReceiptIntegrityError(ExecutionReceiptError):
    """Raised when live replay or an external byte commitment disagrees."""


def _schema_fail(message: str) -> NoReturn:
    raise ExecutionReceiptSchemaError(message)


def _integrity_fail(message: str) -> NoReturn:
    raise ExecutionReceiptIntegrityError(message)


def _strict_fields(
    value: Mapping[str, object], expected: frozenset[str], context: str
) -> None:
    actual = frozenset(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        _schema_fail(f"{context} is missing fields: {', '.join(missing)}")
    if unknown:
        _schema_fail(f"{context} has unknown fields: {', '.join(unknown)}")


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        _schema_fail(f"{field} must be an object with string keys")
    return cast(Mapping[str, object], value)


def _array(value: object, field: str) -> Sequence[object]:
    if not isinstance(value, list) or not value:
        _schema_fail(f"{field} must be a non-empty array")
    return cast(Sequence[object], value)


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        _schema_fail(f"{field} must be a stable identifier")
    return value


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _schema_fail(f"{field} must be a lowercase SHA-256 digest")
    return value


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        _schema_fail(f"{field} must be a positive integer")
    return value


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        _schema_fail(f"{field} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        _schema_fail(f"{field} must be finite")
    return parsed


class ExecutionInputRole(StrEnum):
    """The exact, closed source set accepted by the synthetic verifier."""

    RUN_CONFIG = "run_config"
    STUDY_PROTOCOL = "study_protocol"
    DATA_MANIFEST = "data_manifest"
    POLICY_AFTER_DATA = "policy_after_data"
    MODEL_CHECKPOINT = "model_checkpoint"
    PREDICTION_ARTIFACT = "prediction_artifact"
    POSTERIOR_ARTIFACT = "posterior_artifact"
    ENVIRONMENT_LOCK = "environment_lock"
    FREEZE_RECEIPT = "freeze_receipt"
    EVENT_KEY_MANIFEST = "event_key_manifest"


@dataclass(frozen=True, slots=True)
class SourceInputReceipt:
    """One regular file whose exact bytes were re-read by the verifier."""

    role: ExecutionInputRole
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.role, ExecutionInputRole):
            _schema_fail("source input role is invalid")
        _sha256(self.sha256, "source input sha256")
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes < 0
        ):
            _schema_fail("source input size_bytes must be a non-negative integer")

    def to_mapping(self) -> dict[str, object]:
        return {
            "role": self.role.value,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(
            value,
            frozenset({"role", "sha256", "size_bytes"}),
            "source input receipt",
        )
        raw_role = value["role"]
        if not isinstance(raw_role, str):
            _schema_fail("source input role must be a string")
        try:
            role = ExecutionInputRole(raw_role)
        except ValueError as error:
            raise ExecutionReceiptSchemaError(
                "source input role is unsupported"
            ) from error
        size = value["size_bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            _schema_fail("source input size_bytes must be a non-negative integer")
        return cls(
            role=role,
            sha256=_sha256(value["sha256"], "source input sha256"),
            size_bytes=size,
        )


@dataclass(frozen=True, slots=True)
class RunnerImplementationReceipt:
    """Live-code and immutable-configuration identity for the executed runner."""

    module: str
    qualname: str
    module_relative_path: str
    module_source_sha256: str
    class_source_sha256: str
    entrypoint_source_sha256: str
    runner_configuration_sha256: str
    runner_kind: str
    claim_scope: str
    mechanism_execution: bool

    def __post_init__(self) -> None:
        _identifier(self.module, "runner module")
        _identifier(self.qualname, "runner qualname")
        if (
            not isinstance(self.module_relative_path, str)
            or not self.module_relative_path
            or "\\" in self.module_relative_path
            or Path(self.module_relative_path).is_absolute()
            or any(
                part in {"", ".", ".."}
                for part in Path(self.module_relative_path).parts
            )
        ):
            _schema_fail("runner module_relative_path must be a safe relative path")
        for field in (
            "module_source_sha256",
            "class_source_sha256",
            "entrypoint_source_sha256",
            "runner_configuration_sha256",
        ):
            _sha256(getattr(self, field), field)
        _identifier(self.runner_kind, "runner_kind")
        _identifier(self.claim_scope, "claim_scope")
        if self.mechanism_execution is not True:
            _schema_fail("verifier-owned runner must execute its declared mechanism")

    def to_mapping(self) -> dict[str, object]:
        return {
            "claim_scope": self.claim_scope,
            "class_source_sha256": self.class_source_sha256,
            "entrypoint_source_sha256": self.entrypoint_source_sha256,
            "mechanism_execution": self.mechanism_execution,
            "module": self.module,
            "module_relative_path": self.module_relative_path,
            "module_source_sha256": self.module_source_sha256,
            "qualname": self.qualname,
            "runner_configuration_sha256": self.runner_configuration_sha256,
            "runner_kind": self.runner_kind,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(
            value,
            frozenset(
                {
                    "claim_scope",
                    "class_source_sha256",
                    "entrypoint_source_sha256",
                    "mechanism_execution",
                    "module",
                    "module_relative_path",
                    "module_source_sha256",
                    "qualname",
                    "runner_configuration_sha256",
                    "runner_kind",
                }
            ),
            "runner implementation receipt",
        )
        mechanism = value["mechanism_execution"]
        if not isinstance(mechanism, bool):
            _schema_fail("runner mechanism_execution must be boolean")
        relative = value["module_relative_path"]
        if not isinstance(relative, str):
            _schema_fail("runner module_relative_path must be a string")
        return cls(
            module=_identifier(value["module"], "runner module"),
            qualname=_identifier(value["qualname"], "runner qualname"),
            module_relative_path=relative,
            module_source_sha256=_sha256(
                value["module_source_sha256"], "runner module_source_sha256"
            ),
            class_source_sha256=_sha256(
                value["class_source_sha256"], "runner class_source_sha256"
            ),
            entrypoint_source_sha256=_sha256(
                value["entrypoint_source_sha256"],
                "runner entrypoint_source_sha256",
            ),
            runner_configuration_sha256=_sha256(
                value["runner_configuration_sha256"],
                "runner runner_configuration_sha256",
            ),
            runner_kind=_identifier(value["runner_kind"], "runner_kind"),
            claim_scope=_identifier(value["claim_scope"], "claim_scope"),
            mechanism_execution=mechanism,
        )


@dataclass(frozen=True, slots=True)
class VerifierOwnedSyntheticRunner:
    """Fixed offline runner whose outputs can be replayed by the verifier.

    Values are fixtures for testing reporting provenance.  They must never be
    represented as estimates from real market data or as a complex simulator.
    """

    values_by_row: tuple[tuple[str, tuple[float, ...]], ...]
    runner_kind: str = VERIFIER_OWNED_RUNNER_KIND
    claim_scope: str = "synthetic_mechanics_only"
    mechanism_execution: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.values_by_row, tuple) or not self.values_by_row:
            _schema_fail("values_by_row must be a non-empty immutable tuple")
        normalized: list[tuple[str, tuple[float, ...]]] = []
        for row_id, values in self.values_by_row:
            parsed_row = _identifier(row_id, "runner values row_id")
            if not isinstance(values, tuple) or len(values) < 2:
                _schema_fail(
                    "each verifier-owned value series needs at least two draws"
                )
            normalized.append(
                (
                    parsed_row,
                    tuple(_finite(item, "runner fixture value") for item in values),
                )
            )
        normalized.sort(key=lambda item: item[0])
        if len({item[0] for item in normalized}) != len(normalized):
            _schema_fail("verifier-owned runner row IDs must be unique")
        object.__setattr__(self, "values_by_row", tuple(normalized))
        if self.runner_kind != VERIFIER_OWNED_RUNNER_KIND:
            _schema_fail("verifier-owned runner_kind cannot be changed")
        if self.claim_scope != "synthetic_mechanics_only":
            _schema_fail("verifier-owned runner claim scope cannot be changed")
        if self.mechanism_execution is not True:
            _schema_fail("verifier-owned runner cannot disable its scalar mechanism")

    def execution_configuration_mapping(self) -> dict[str, object]:
        """Return the exact immutable fixture values used by replay."""

        return {
            "values_by_row": [
                {"row_id": row_id, "values": list(values)}
                for row_id, values in self.values_by_row
            ]
        }

    def run_replicate(
        self, row: ExperimentRow, replicate: int, event_key: EventKey
    ) -> float:
        """Return the predeclared synthetic value at an engine event address."""

        if not isinstance(event_key, EventKey):
            _integrity_fail("verifier-owned runner requires an engine EventKey")
        values = next(
            (series for row_id, series in self.values_by_row if row_id == row.row_id),
            None,
        )
        if values is None:
            _integrity_fail(f"no verifier-owned values for row {row.row_id!r}")
        if replicate < 0 or replicate >= len(values):
            _integrity_fail(
                f"verifier-owned values do not cover replicate {replicate} "
                f"for row {row.row_id!r}"
            )
        # Accessing the value forces construction and validation of the exact
        # engine address even though this particular fixture has no stochastic
        # component.
        if not event_key.value.startswith("ptk1:"):
            _integrity_fail("engine event key is invalid")
        return values[replicate]


@dataclass(frozen=True, slots=True)
class ExecutionSourceInputSet:
    """Paths plus external anchors for the exact ten-file source boundary."""

    run_config_path: Path
    expected_run_config_sha256: str
    study_protocol_path: Path
    expected_study_protocol_source_sha256: str
    data_manifest_path: Path
    policy_after_data_path: Path
    model_checkpoint_path: Path
    prediction_artifact_path: Path
    posterior_artifact_path: Path
    environment_lock_path: Path
    freeze_receipt_path: Path
    event_key_manifest_path: Path
    expected_event_key_manifest_sha256: str

    def __post_init__(self) -> None:
        for field in (
            "expected_run_config_sha256",
            "expected_study_protocol_source_sha256",
            "expected_event_key_manifest_sha256",
        ):
            _sha256(getattr(self, field), field)
        paths = tuple(path.expanduser().resolve() for path in self._paths())
        if len(set(paths)) != len(paths):
            _schema_fail("closed execution inputs must be ten distinct files")

    def _paths(self) -> tuple[Path, ...]:
        return (
            self.run_config_path,
            self.study_protocol_path,
            self.data_manifest_path,
            self.policy_after_data_path,
            self.model_checkpoint_path,
            self.prediction_artifact_path,
            self.posterior_artifact_path,
            self.environment_lock_path,
            self.freeze_receipt_path,
            self.event_key_manifest_path,
        )

    @property
    def paths(self) -> tuple[Path, ...]:
        """Return the complete path set in frozen role order."""

        return self._paths()

    @staticmethod
    def _commit_file(
        path: Path, *, role: ExecutionInputRole, expected_sha256: str
    ) -> SourceInputReceipt:
        expected = _sha256(expected_sha256, f"{role.value} expected sha256")
        if path.is_symlink() or not path.is_file():
            _integrity_fail(f"{role.value} must be a regular non-symlink file")
        try:
            observed = sha256_file(path)
            size = path.stat().st_size
        except OSError as error:
            raise ExecutionReceiptIntegrityError(
                f"cannot read closed input {role.value}: {error}"
            ) from error
        if observed != expected:
            _integrity_fail(f"closed input {role.value} hash mismatch")
        return SourceInputReceipt(role=role, sha256=observed, size_bytes=size)

    def commitments(
        self, row: ExperimentRow, protocol: StudyProtocol
    ) -> tuple[SourceInputReceipt, ...]:
        """Re-read every allowed input and reject any digest mismatch."""

        frozen = row.frozen_inputs
        role_path_digest = (
            (
                ExecutionInputRole.RUN_CONFIG,
                self.run_config_path,
                self.expected_run_config_sha256,
            ),
            (
                ExecutionInputRole.STUDY_PROTOCOL,
                self.study_protocol_path,
                self.expected_study_protocol_source_sha256,
            ),
            (
                ExecutionInputRole.DATA_MANIFEST,
                self.data_manifest_path,
                frozen.data_manifest_sha256,
            ),
            (
                ExecutionInputRole.POLICY_AFTER_DATA,
                self.policy_after_data_path,
                frozen.policy_after_data_sha256,
            ),
            (
                ExecutionInputRole.MODEL_CHECKPOINT,
                self.model_checkpoint_path,
                frozen.model_checkpoint_sha256,
            ),
            (
                ExecutionInputRole.PREDICTION_ARTIFACT,
                self.prediction_artifact_path,
                frozen.prediction_artifact_sha256,
            ),
            (
                ExecutionInputRole.POSTERIOR_ARTIFACT,
                self.posterior_artifact_path,
                frozen.posterior_artifact_sha256,
            ),
            (
                ExecutionInputRole.ENVIRONMENT_LOCK,
                self.environment_lock_path,
                frozen.environment_lock_sha256,
            ),
            (
                ExecutionInputRole.FREEZE_RECEIPT,
                self.freeze_receipt_path,
                frozen.freeze_receipt_sha256,
            ),
            (
                ExecutionInputRole.EVENT_KEY_MANIFEST,
                self.event_key_manifest_path,
                self.expected_event_key_manifest_sha256,
            ),
        )
        receipts = tuple(
            self._commit_file(path, role=role, expected_sha256=digest)
            for role, path, digest in role_path_digest
        )
        if tuple(item.role for item in receipts) != tuple(ExecutionInputRole):
            _integrity_fail("closed execution input role set changed")
        try:
            loaded = load_study_protocol(
                self.study_protocol_path,
                expected_source_sha256=self.expected_study_protocol_source_sha256,
            )
        except ProtocolValidationError as error:
            raise ExecutionReceiptIntegrityError(
                f"cannot verify study protocol input: {error}"
            ) from error
        if loaded.protocol.sha256 != protocol.sha256:
            _integrity_fail("study protocol input has another semantic commitment")
        return receipts


@dataclass(frozen=True, slots=True)
class SyntheticExecutionReceipt:
    """Complete post-execution receipt for one deterministic result row."""

    protocol_sha256: str
    protocol_source_sha256: str
    matrix_id: str
    matrix_sha256: str
    row_id: str
    config_sha256: str
    frozen_inputs_sha256: str
    draw_identity_sha256: str
    shared_draw_identity: bool
    source_inputs: tuple[SourceInputReceipt, ...]
    source_input_manifest_sha256: str
    runner: RunnerImplementationReceipt
    result_sha256: str
    result_file_sha256: str
    event_keys_sha256: str
    samples_sha256: str
    sample_count: int
    authority_source_sha256: str
    synthetic_only: bool = True
    authority_id: str = EXECUTION_RECEIPT_AUTHORITY_ID
    schema_version: str = EXECUTION_RECEIPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field in (
            "protocol_sha256",
            "protocol_source_sha256",
            "matrix_sha256",
            "config_sha256",
            "frozen_inputs_sha256",
            "draw_identity_sha256",
            "source_input_manifest_sha256",
            "result_sha256",
            "result_file_sha256",
            "event_keys_sha256",
            "samples_sha256",
            "authority_source_sha256",
        ):
            _sha256(getattr(self, field), field)
        _identifier(self.matrix_id, "receipt matrix_id")
        _identifier(self.row_id, "receipt row_id")
        if not isinstance(self.shared_draw_identity, bool):
            _schema_fail("receipt shared_draw_identity must be boolean")
        _positive_int(self.sample_count, "receipt sample_count")
        if not isinstance(self.source_inputs, tuple):
            _schema_fail("receipt source_inputs must be immutable")
        if tuple(item.role for item in self.source_inputs) != tuple(ExecutionInputRole):
            _schema_fail("receipt source_inputs must contain the exact closed role set")
        expected_manifest = sha256_bytes(
            canonical_json_bytes([item.to_mapping() for item in self.source_inputs])
        )
        if self.source_input_manifest_sha256 != expected_manifest:
            _integrity_fail("receipt source-input manifest hash mismatch")
        if not isinstance(self.runner, RunnerImplementationReceipt):
            _schema_fail("receipt runner implementation is invalid")
        if self.synthetic_only is not True:
            _schema_fail("this receipt schema is synthetic-only")
        if self.authority_id != EXECUTION_RECEIPT_AUTHORITY_ID:
            _schema_fail("execution receipt authority_id is unsupported")
        if self.schema_version != EXECUTION_RECEIPT_SCHEMA_VERSION:
            _schema_fail("execution receipt schema_version is unsupported")

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def to_mapping(self) -> dict[str, object]:
        return {
            "authority_id": self.authority_id,
            "authority_source_sha256": self.authority_source_sha256,
            "config_sha256": self.config_sha256,
            "draw_identity_sha256": self.draw_identity_sha256,
            "event_keys_sha256": self.event_keys_sha256,
            "frozen_inputs_sha256": self.frozen_inputs_sha256,
            "matrix_id": self.matrix_id,
            "matrix_sha256": self.matrix_sha256,
            "protocol_sha256": self.protocol_sha256,
            "protocol_source_sha256": self.protocol_source_sha256,
            "result_file_sha256": self.result_file_sha256,
            "result_sha256": self.result_sha256,
            "row_id": self.row_id,
            "runner": self.runner.to_mapping(),
            "sample_count": self.sample_count,
            "samples_sha256": self.samples_sha256,
            "schema_version": self.schema_version,
            "source_input_manifest_sha256": self.source_input_manifest_sha256,
            "source_inputs": [item.to_mapping() for item in self.source_inputs],
            "shared_draw_identity": self.shared_draw_identity,
            "synthetic_only": self.synthetic_only,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(
            value,
            frozenset(
                {
                    "authority_id",
                    "authority_source_sha256",
                    "config_sha256",
                    "draw_identity_sha256",
                    "event_keys_sha256",
                    "frozen_inputs_sha256",
                    "matrix_id",
                    "matrix_sha256",
                    "protocol_sha256",
                    "protocol_source_sha256",
                    "result_file_sha256",
                    "result_sha256",
                    "row_id",
                    "runner",
                    "sample_count",
                    "samples_sha256",
                    "schema_version",
                    "source_input_manifest_sha256",
                    "source_inputs",
                    "shared_draw_identity",
                    "synthetic_only",
                }
            ),
            "synthetic execution receipt",
        )
        synthetic = value["synthetic_only"]
        if not isinstance(synthetic, bool):
            _schema_fail("receipt synthetic_only must be boolean")
        shared_draw_identity = value["shared_draw_identity"]
        if not isinstance(shared_draw_identity, bool):
            _schema_fail("receipt shared_draw_identity must be boolean")
        authority_id = value["authority_id"]
        schema_version = value["schema_version"]
        if not isinstance(authority_id, str) or not isinstance(schema_version, str):
            _schema_fail("receipt authority and schema versions must be strings")
        return cls(
            protocol_sha256=_sha256(value["protocol_sha256"], "protocol_sha256"),
            protocol_source_sha256=_sha256(
                value["protocol_source_sha256"], "protocol_source_sha256"
            ),
            matrix_id=_identifier(value["matrix_id"], "matrix_id"),
            matrix_sha256=_sha256(value["matrix_sha256"], "matrix_sha256"),
            row_id=_identifier(value["row_id"], "row_id"),
            config_sha256=_sha256(value["config_sha256"], "config_sha256"),
            frozen_inputs_sha256=_sha256(
                value["frozen_inputs_sha256"], "frozen_inputs_sha256"
            ),
            draw_identity_sha256=_sha256(
                value["draw_identity_sha256"], "draw_identity_sha256"
            ),
            shared_draw_identity=shared_draw_identity,
            source_inputs=tuple(
                SourceInputReceipt.from_mapping(_mapping(item, "source input"))
                for item in _array(value["source_inputs"], "source_inputs")
            ),
            source_input_manifest_sha256=_sha256(
                value["source_input_manifest_sha256"],
                "source_input_manifest_sha256",
            ),
            runner=RunnerImplementationReceipt.from_mapping(
                _mapping(value["runner"], "runner")
            ),
            result_sha256=_sha256(value["result_sha256"], "result_sha256"),
            result_file_sha256=_sha256(
                value["result_file_sha256"], "result_file_sha256"
            ),
            event_keys_sha256=_sha256(value["event_keys_sha256"], "event_keys_sha256"),
            samples_sha256=_sha256(value["samples_sha256"], "samples_sha256"),
            sample_count=_positive_int(value["sample_count"], "sample_count"),
            authority_source_sha256=_sha256(
                value["authority_source_sha256"], "authority_source_sha256"
            ),
            synthetic_only=synthetic,
            authority_id=authority_id,
            schema_version=schema_version,
        )


@dataclass(frozen=True, slots=True)
class ExecutionReceiptAuditReport:
    """Positive proof returned only after closed-input replay succeeds."""

    valid: bool
    receipt_sha256: str
    row_id: str
    result_sha256: str
    replayed_sample_count: int

    def __post_init__(self) -> None:
        if self.valid is not True:
            _schema_fail("positive execution receipt audit must have valid=True")
        _sha256(self.receipt_sha256, "receipt audit sha256")
        _identifier(self.row_id, "receipt audit row_id")
        _sha256(self.result_sha256, "receipt audit result_sha256")
        _positive_int(self.replayed_sample_count, "replayed_sample_count")


@dataclass(frozen=True, slots=True)
class SyntheticExecutionAuthority:
    """Live verifier context; callers cannot replace its runner implementation."""

    protocol: StudyProtocol
    source_inputs: ExecutionSourceInputSet
    runner: VerifierOwnedSyntheticRunner
    repository_root: Path

    def __post_init__(self) -> None:
        if not isinstance(self.protocol, StudyProtocol):
            _schema_fail("execution authority requires a StudyProtocol")
        if self.protocol.mode is not StudyMode.SYNTHETIC:
            _schema_fail("execution receipts cannot authorize non-synthetic studies")
        if not isinstance(self.source_inputs, ExecutionSourceInputSet):
            _schema_fail("execution authority requires a closed source-input set")
        if type(self.runner) is not VerifierOwnedSyntheticRunner:
            _schema_fail("execution authority accepts only its bundled runner type")
        root = self.repository_root.expanduser().resolve()
        if not root.is_dir() or root.is_symlink():
            _schema_fail("execution authority repository_root must be a directory")
        object.__setattr__(self, "repository_root", root)
        self._implementation_receipt()

    @property
    def protocol_source_sha256(self) -> str:
        return self.source_inputs.expected_study_protocol_source_sha256

    def _implementation_receipt(self) -> RunnerImplementationReceipt:
        runner_type = type(self.runner)
        module_path_raw = inspect.getsourcefile(runner_type)
        if module_path_raw is None:
            _integrity_fail("cannot locate verifier-owned runner source")
        module_path = Path(module_path_raw).resolve()
        if module_path.is_symlink() or not module_path.is_file():
            _integrity_fail("verifier-owned runner source must be a regular file")
        try:
            relative = module_path.relative_to(self.repository_root).as_posix()
        except ValueError as error:
            raise ExecutionReceiptIntegrityError(
                "verifier-owned runner source is outside repository_root"
            ) from error
        try:
            class_source = inspect.getsource(runner_type).encode("utf-8")
            entrypoint_source = inspect.getsource(runner_type.run_replicate).encode(
                "utf-8"
            )
        except (OSError, TypeError) as error:
            raise ExecutionReceiptIntegrityError(
                f"cannot inspect verifier-owned runner source: {error}"
            ) from error
        configuration_sha256 = sha256_bytes(
            canonical_json_bytes(self.runner.execution_configuration_mapping())
        )
        return RunnerImplementationReceipt(
            module=runner_type.__module__,
            qualname=runner_type.__qualname__,
            module_relative_path=relative,
            module_source_sha256=sha256_file(module_path),
            class_source_sha256=sha256_bytes(class_source),
            entrypoint_source_sha256=sha256_bytes(entrypoint_source),
            runner_configuration_sha256=configuration_sha256,
            runner_kind=self.runner.runner_kind,
            claim_scope=self.runner.claim_scope,
            mechanism_execution=self.runner.mechanism_execution,
        )

    def _validate_matrix(self, matrix: ExperimentMatrix) -> None:
        if not isinstance(matrix, ExperimentMatrix):
            _schema_fail("execution authority requires an ExperimentMatrix")
        expected_rows = tuple(row.row_id for row in matrix.rows)
        fixture_rows = tuple(row_id for row_id, _ in self.runner.values_by_row)
        if fixture_rows != tuple(sorted(expected_rows)):
            _integrity_fail(
                "verifier-owned runner fixture row set must equal the matrix row set"
            )
        frozen_hashes = {row.frozen_inputs.sha256 for row in matrix.rows}
        if len(frozen_hashes) != 1:
            _integrity_fail("one receipt authority requires one frozen input reference")
        self._validate_shared_draw_event_manifest(matrix)
        for row in matrix.rows:
            if not row.frozen_inputs.synthetic_only:
                _integrity_fail("execution receipts cannot cover non-synthetic rows")
            contract = row.shared_draw_identity
            if (
                contract is not None
                and contract.sample_manifest_sha256
                != self.source_inputs.expected_event_key_manifest_sha256
            ):
                _integrity_fail(
                    "shared draw identity does not bind the closed event-key manifest"
                )
            values = dict(self.runner.values_by_row)[row.row_id]
            if len(values) < row.stop_rule.maximum_replicates:
                _integrity_fail(
                    f"verifier-owned values do not cover the frozen maximum for "
                    f"row {row.row_id!r}"
                )

    def _validate_shared_draw_event_manifest(self, matrix: ExperimentMatrix) -> None:
        """Cross-check contracted engine addresses against the closed manifest."""

        groups: dict[str, ExperimentRow] = {}
        for row in matrix.rows:
            contract = row.shared_draw_identity
            if contract is not None:
                groups.setdefault(contract.contract_id, row)
        if not groups:
            return
        try:
            manifest = read_json_object(self.source_inputs.event_key_manifest_path)
        except (MetadataValidationError, OSError, UnicodeError) as error:
            raise ExecutionReceiptIntegrityError(
                f"cannot read shared draw EventKey manifest: {error}"
            ) from error
        _strict_fields(
            manifest,
            frozenset({"contracts", "schema_version", "synthetic_only"}),
            "shared draw EventKey manifest",
        )
        if (
            manifest["schema_version"] != SHARED_DRAW_EVENT_KEY_MANIFEST_SCHEMA_VERSION
            or manifest["synthetic_only"] is not True
        ):
            _integrity_fail(
                "shared draw EventKey manifest must use the synthetic-only schema"
            )
        entries = _array(manifest["contracts"], "shared draw manifest contracts")
        parsed_ids: list[str] = []
        for index, raw_entry in enumerate(entries):
            entry = _mapping(raw_entry, f"shared draw manifest contracts[{index}]")
            _strict_fields(
                entry,
                frozenset(
                    {
                        "contract_id",
                        "event_keys",
                        "event_namespace",
                        "replicate_count",
                    }
                ),
                f"shared draw manifest contracts[{index}]",
            )
            contract_id = _identifier(entry["contract_id"], "manifest contract_id")
            manifest_row = groups.get(contract_id)
            if manifest_row is None:
                _integrity_fail(
                    "shared draw EventKey manifest contains an unregistered contract"
                )
            contract = manifest_row.shared_draw_identity
            if contract is None:  # pragma: no cover - guaranteed by groups
                raise AssertionError("shared draw group lost its contract")
            if (
                entry["event_namespace"] != contract.event_namespace
                or entry["replicate_count"] != contract.replicate_count
            ):
                _integrity_fail("shared draw EventKey manifest contract fields changed")
            raw_keys = _array(entry["event_keys"], "shared draw manifest event_keys")
            if not all(isinstance(item, str) for item in raw_keys):
                _integrity_fail("shared draw EventKey manifest keys must be strings")
            expected_keys = tuple(
                matrix_replicate_event_key(matrix, manifest_row, replicate).value
                for replicate in range(contract.replicate_count)
            )
            if tuple(raw_keys) != expected_keys:
                _integrity_fail(
                    "shared draw EventKey manifest does not match engine addresses"
                )
            parsed_ids.append(contract_id)
        if tuple(parsed_ids) != tuple(sorted(groups)):
            _integrity_fail(
                "shared draw EventKey manifest must exhaust contracts in sorted order"
            )

    def ensure_runner(self, matrix: ExperimentMatrix, runner: ReplicateRunner) -> None:
        """Reject a swapped or caller-defined runner before execution starts."""

        self._validate_matrix(matrix)
        if runner is not self.runner:
            _integrity_fail(
                "execution authority runner object differs from the scheduled runner"
            )
        self._implementation_receipt()
        self.source_inputs.commitments(matrix.rows[0], self.protocol)

    def _expected_receipt(
        self,
        matrix: ExperimentMatrix,
        row: ExperimentRow,
        result: ExperimentResult,
        result_path: Path,
    ) -> SyntheticExecutionReceipt:
        self._validate_matrix(matrix)
        if matrix.row(row.row_id) != row:
            _integrity_fail("receipt row is not the authoritative matrix row")
        if (
            result.row_id != row.row_id
            or result.config_sha256 != row.config_sha256
            or result.frozen_inputs_sha256 != row.frozen_inputs.sha256
            or result.draw_identity_sha256 != row.draw_identity_sha256
            or result.shared_draw_identity is not row.has_shared_draw_identity
        ):
            _integrity_fail("receipt result does not match its matrix row")
        implementation = self._implementation_receipt()
        if (
            result.runner_kind != implementation.runner_kind
            or result.claim_scope != implementation.claim_scope
            or result.mechanism_execution != implementation.mechanism_execution
        ):
            _integrity_fail(
                "result self-reported runner fields disagree with live implementation"
            )
        if not result.synthetic_only:
            _integrity_fail("synthetic execution receipt cannot bind a real result")
        if result_path.is_symlink() or not result_path.is_file():
            _integrity_fail("receipt result path must be a regular non-symlink file")
        source_inputs = self.source_inputs.commitments(row, self.protocol)
        source_manifest_sha256 = sha256_bytes(
            canonical_json_bytes([item.to_mapping() for item in source_inputs])
        )
        return SyntheticExecutionReceipt(
            protocol_sha256=self.protocol.sha256,
            protocol_source_sha256=self.protocol_source_sha256,
            matrix_id=matrix.matrix_id,
            matrix_sha256=matrix.sha256,
            row_id=row.row_id,
            config_sha256=row.config_sha256,
            frozen_inputs_sha256=row.frozen_inputs.sha256,
            draw_identity_sha256=row.draw_identity_sha256,
            shared_draw_identity=row.has_shared_draw_identity,
            source_inputs=source_inputs,
            source_input_manifest_sha256=source_manifest_sha256,
            runner=implementation,
            result_sha256=result.sha256,
            result_file_sha256=sha256_file(result_path),
            event_keys_sha256=sha256_bytes(
                canonical_json_bytes(list(result.event_keys))
            ),
            samples_sha256=sha256_bytes(canonical_json_bytes(list(result.samples))),
            sample_count=len(result.samples),
            authority_source_sha256=sha256_file(Path(__file__).resolve()),
        )

    def issue(
        self,
        matrix: ExperimentMatrix,
        row: ExperimentRow,
        result: ExperimentResult,
        result_path: Path,
    ) -> SyntheticExecutionReceipt:
        """Issue a receipt only after exact deterministic replay succeeds."""

        self._replay(matrix, row, result)
        return self._expected_receipt(matrix, row, result, result_path)

    def _replay(
        self, matrix: ExperimentMatrix, row: ExperimentRow, result: ExperimentResult
    ) -> None:
        replayed: list[float] = []
        replayed_keys: list[str] = []
        for replicate in range(len(result.samples)):
            event_key = matrix_replicate_event_key(matrix, row, replicate)
            replayed_keys.append(event_key.value)
            replayed.append(self.runner.run_replicate(row, replicate, event_key))
        if tuple(replayed_keys) != result.event_keys:
            _integrity_fail("receipt replay event-key sequence changed")
        if tuple(replayed) != result.samples:
            _integrity_fail("receipt replay output differs from stored samples")

    def verify(
        self,
        receipt: SyntheticExecutionReceipt,
        matrix: ExperimentMatrix,
        row: ExperimentRow,
        result: ExperimentResult,
        result_path: Path,
    ) -> ExecutionReceiptAuditReport:
        """Re-read inputs, re-hash live code, and replay every stored sample."""

        self._replay(matrix, row, result)
        expected = self._expected_receipt(matrix, row, result, result_path)
        if receipt != expected:
            _integrity_fail(
                "execution receipt disagrees with live inputs, code, or replay"
            )
        return ExecutionReceiptAuditReport(
            valid=True,
            receipt_sha256=receipt.sha256,
            row_id=row.row_id,
            result_sha256=result.sha256,
            replayed_sample_count=len(result.samples),
        )


def write_execution_receipt(directory: Path, receipt: SyntheticExecutionReceipt) -> str:
    """Add the two immutable receipt files to an existing result directory."""

    receipt_path = directory / "execution-receipt.json"
    seal_path = directory / "execution-receipt.sha256"
    if receipt_path.exists() or seal_path.exists():
        _integrity_fail("execution receipt files already exist")
    write_json_file(receipt_path, receipt.to_mapping())
    write_text_file(seal_path, f"{receipt.sha256}\n")
    return receipt.sha256


def read_execution_receipt(directory: Path) -> SyntheticExecutionReceipt:
    """Parse and self-seal-check one strict execution receipt."""

    receipt_path = directory / "execution-receipt.json"
    seal_path = directory / "execution-receipt.sha256"
    if (
        receipt_path.is_symlink()
        or seal_path.is_symlink()
        or not receipt_path.is_file()
        or not seal_path.is_file()
    ):
        _integrity_fail("execution receipt files are missing or invalid")
    try:
        receipt = SyntheticExecutionReceipt.from_mapping(read_json_object(receipt_path))
        seal = seal_path.read_text(encoding="ascii")
    except (
        ExecutionReceiptSchemaError,
        MetadataValidationError,
        OSError,
        UnicodeError,
    ) as error:
        raise ExecutionReceiptIntegrityError(str(error)) from error
    if seal != f"{receipt.sha256}\n":
        _integrity_fail("execution receipt semantic seal mismatch")
    return receipt


__all__ = [
    "EXECUTION_RECEIPT_AUTHORITY_ID",
    "EXECUTION_RECEIPT_SCHEMA_VERSION",
    "ExecutionInputRole",
    "ExecutionReceiptAuditReport",
    "ExecutionReceiptError",
    "ExecutionReceiptIntegrityError",
    "ExecutionReceiptSchemaError",
    "ExecutionSourceInputSet",
    "SHARED_DRAW_EVENT_KEY_MANIFEST_SCHEMA_VERSION",
    "RunnerImplementationReceipt",
    "SourceInputReceipt",
    "SyntheticExecutionAuthority",
    "SyntheticExecutionReceipt",
    "VERIFIER_OWNED_RUNNER_KIND",
    "VerifierOwnedSyntheticRunner",
    "read_execution_receipt",
    "write_execution_receipt",
]
