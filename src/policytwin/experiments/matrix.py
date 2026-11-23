"""Immutable preregistered experiment matrices and local scheduling.

The module intentionally has no Cartesian-product executor and no cluster
submission function.  A matrix consists only of explicit preregistered rows.
Local execution validates a completed result's strict schema and semantic hash;
the mere presence of a file never marks a row complete.
"""

from __future__ import annotations

import math
import os
import re
import stat
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn, Protocol, Self, cast

from policytwin.counterfactual import EventType, SemanticEventError, SemanticEventKey
from policytwin.counterfactual.randomness import EventKey, deterministic_int
from policytwin.experiments.distribution_mcse import (
    DistributionMCSEError,
    DistributionMCSEEstimate,
    DistributionMCSESpec,
    MCSEStopDecision,
    MonteCarloDraw,
    evaluate_distribution_mcse,
)
from policytwin.provenance import (
    JsonlEventWriter,
    MetadataValidationError,
    canonical_json_bytes,
    read_json_bytes_object,
    read_json_object,
    read_jsonl_bytes_objects,
    read_jsonl_objects,
    sha256_bytes,
    sha256_file,
    write_json_file,
    write_text_file,
)

if TYPE_CHECKING:
    from policytwin.experiments.execution_receipt import SyntheticExecutionAuthority

MATRIX_SCHEMA_VERSION = "policytwin.experiment-matrix.v2"
RESULT_SCHEMA_VERSION = "policytwin.experiment-matrix-result.v4"
DISTRIBUTION_RESULT_SCHEMA_VERSION = "policytwin.experiment-matrix-result.v5"
CLUSTER_PLAN_SCHEMA_VERSION = "policytwin.cluster-batch-plan.v1"
RUN_MANIFEST_SCHEMA_VERSION = "policytwin.experiment-matrix-run-manifest.v2"
DRAW_IDENTITY_SCHEMA_VERSION = "policytwin.shared-draw-identity.v1"
ROW_LOCAL_DRAW_IDENTITY_SCHEMA_VERSION = "policytwin.row-local-draw-identity.v1"

_RUN_LOG_PATHS = (
    "registry.jsonl",
    "dependencies.jsonl",
    "resources.jsonl",
    "timing.jsonl",
    "failures.jsonl",
)
_RUN_FIXED_ARTIFACT_PATHS = (
    "matrix.json",
    "matrix.sha256",
    *_RUN_LOG_PATHS,
    "cluster-plan.json",
)

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EVENT_KEY = re.compile(r"^ptk1:[0-9a-f]{64}$")


class ExperimentMatrixError(RuntimeError):
    """Base class for fail-closed matrix errors."""


class ExperimentMatrixSchemaError(ExperimentMatrixError, ValueError):
    """Raised when a matrix or row is ambiguous or internally inconsistent."""


class ExperimentResultIntegrityError(ExperimentMatrixError):
    """Raised when a claimed completed result cannot be verified."""


class SyntheticMatrixRunError(ExperimentMatrixError):
    """Deliberate synthetic runner failure used by scheduler tests."""


_MAX_AUDITED_ARTIFACT_BYTES = 128 * 1024 * 1024
_StatFingerprint = tuple[int, int, int, int, int, int, int]


def _stat_fingerprint(info: os.stat_result) -> _StatFingerprint:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _read_exact_descriptor(
    descriptor: int, *, expected_size: int, relative_path: str
) -> bytes:
    if not 0 <= expected_size <= _MAX_AUDITED_ARTIFACT_BYTES:
        raise ExperimentResultIntegrityError(
            f"audited artifact size is invalid: {relative_path}"
        )
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        observed_size = 0
        while observed_size < expected_size:
            chunk = os.read(
                descriptor,
                min(1024 * 1024, expected_size - observed_size),
            )
            if not chunk:
                raise ExperimentResultIntegrityError(
                    f"audited artifact changed while read: {relative_path}"
                )
            chunks.append(chunk)
            observed_size += len(chunk)
        if os.read(descriptor, 1):
            raise ExperimentResultIntegrityError(
                f"audited artifact grew while read: {relative_path}"
            )
        return b"".join(chunks)
    except OSError as error:
        raise ExperimentResultIntegrityError(
            f"cannot read audited artifact {relative_path}: {error}"
        ) from error


@dataclass(slots=True)
class _CapturedArtifactFile:
    relative_path: str
    descriptor: int
    payload: bytes
    fingerprint: _StatFingerprint


@dataclass(slots=True)
class _CapturedArtifactDirectory:
    path: Path
    name: str | None
    descriptor: int
    fingerprint: _StatFingerprint
    expected_entries: frozenset[str] = frozenset()
    files: dict[str, _CapturedArtifactFile] | None = None
    children: dict[str, _CapturedArtifactDirectory] | None = None
    parent: _CapturedArtifactDirectory | None = None

    def __post_init__(self) -> None:
        if self.files is None:
            self.files = {}
        if self.children is None:
            self.children = {}

    def close(self) -> None:
        assert self.children is not None
        assert self.files is not None
        for child in self.children.values():
            child.close()
        for item in self.files.values():
            os.close(item.descriptor)
        os.close(self.descriptor)


def _directory_open_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _file_open_flags() -> int:
    return os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)


def _open_captured_root(path: Path) -> _CapturedArtifactDirectory:
    try:
        path_info = os.stat(path, follow_symlinks=False)
        if not stat.S_ISDIR(path_info.st_mode):
            raise ExperimentResultIntegrityError(
                "audited artifact root must be a real directory"
            )
        descriptor = os.open(path, _directory_open_flags())
    except OSError as error:
        raise ExperimentResultIntegrityError(
            f"cannot safely open audited artifact root: {error}"
        ) from error
    try:
        opened_info = os.fstat(descriptor)
        if not stat.S_ISDIR(opened_info.st_mode) or _stat_fingerprint(
            opened_info
        ) != _stat_fingerprint(path_info):
            raise ExperimentResultIntegrityError(
                "audited artifact root changed while opened"
            )
        return _CapturedArtifactDirectory(
            path=path,
            name=None,
            descriptor=descriptor,
            fingerprint=_stat_fingerprint(opened_info),
        )
    except Exception:
        os.close(descriptor)
        raise


def _open_captured_child(
    parent: _CapturedArtifactDirectory, name: str
) -> _CapturedArtifactDirectory:
    assert parent.children is not None
    if name in parent.children:
        raise ExperimentResultIntegrityError(
            f"audited directory was captured twice: {name}"
        )
    try:
        path_info = os.stat(name, dir_fd=parent.descriptor, follow_symlinks=False)
        if not stat.S_ISDIR(path_info.st_mode):
            raise ExperimentResultIntegrityError(
                f"audited entry must be a real directory: {name}"
            )
        descriptor = os.open(name, _directory_open_flags(), dir_fd=parent.descriptor)
    except OSError as error:
        raise ExperimentResultIntegrityError(
            f"cannot safely open audited directory {name}: {error}"
        ) from error
    try:
        opened_info = os.fstat(descriptor)
        if not stat.S_ISDIR(opened_info.st_mode) or _stat_fingerprint(
            opened_info
        ) != _stat_fingerprint(path_info):
            raise ExperimentResultIntegrityError(
                f"audited directory changed while opened: {name}"
            )
        captured = _CapturedArtifactDirectory(
            path=parent.path / name,
            name=name,
            descriptor=descriptor,
            fingerprint=_stat_fingerprint(opened_info),
            parent=parent,
        )
        parent.children[name] = captured
        return captured
    except Exception:
        os.close(descriptor)
        raise


def _require_exact_directory_entries(
    captured: _CapturedArtifactDirectory, expected_entries: frozenset[str]
) -> None:
    try:
        observed = frozenset(os.listdir(captured.descriptor))
    except OSError as error:
        raise ExperimentResultIntegrityError(
            f"cannot list audited directory {captured.path}: {error}"
        ) from error
    if observed != expected_entries:
        raise ExperimentResultIntegrityError(
            "audited artifact file set changed: "
            f"missing={sorted(expected_entries - observed)!r}, "
            f"extra={sorted(observed - expected_entries)!r}"
        )
    captured.expected_entries = expected_entries


def _capture_artifact_file(
    directory: _CapturedArtifactDirectory, name: str, *, relative_path: str
) -> _CapturedArtifactFile:
    assert directory.files is not None
    if name in directory.files:
        raise ExperimentResultIntegrityError(
            f"audited file was captured twice: {relative_path}"
        )
    try:
        descriptor = os.open(name, _file_open_flags(), dir_fd=directory.descriptor)
    except OSError as error:
        raise ExperimentResultIntegrityError(
            f"cannot safely open audited artifact {relative_path}: {error}"
        ) from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ExperimentResultIntegrityError(
                "audited artifact must be a singly linked regular file: "
                f"{relative_path}"
            )
        fingerprint = _stat_fingerprint(before)
        payload = _read_exact_descriptor(
            descriptor,
            expected_size=before.st_size,
            relative_path=relative_path,
        )
        path_info = os.stat(
            name,
            dir_fd=directory.descriptor,
            follow_symlinks=False,
        )
        if (
            _stat_fingerprint(os.fstat(descriptor)) != fingerprint
            or _stat_fingerprint(path_info) != fingerprint
        ):
            raise ExperimentResultIntegrityError(
                f"audited artifact changed while captured: {relative_path}"
            )
        captured = _CapturedArtifactFile(
            relative_path=relative_path,
            descriptor=descriptor,
            payload=payload,
            fingerprint=fingerprint,
        )
        directory.files[name] = captured
        return captured
    except Exception:
        os.close(descriptor)
        raise


def _require_captured_tree_current(
    captured: _CapturedArtifactDirectory, *, external_root: bool = False
) -> None:
    assert captured.files is not None
    assert captured.children is not None
    try:
        descriptor_info = os.fstat(captured.descriptor)
        if _stat_fingerprint(descriptor_info) != captured.fingerprint:
            raise ExperimentResultIntegrityError(
                f"audited directory identity changed: {captured.path}"
            )
        if frozenset(os.listdir(captured.descriptor)) != captured.expected_entries:
            raise ExperimentResultIntegrityError(
                f"audited directory entries changed: {captured.path}"
            )
        if external_root:
            path_info = os.stat(captured.path, follow_symlinks=False)
        else:
            if captured.parent is None or captured.name is None:
                raise ExperimentResultIntegrityError(
                    "captured child directory lost its parent identity"
                )
            path_info = os.stat(
                captured.name,
                dir_fd=captured.parent.descriptor,
                follow_symlinks=False,
            )
        if _stat_fingerprint(path_info) != captured.fingerprint:
            raise ExperimentResultIntegrityError(
                f"audited directory path changed: {captured.path}"
            )
        for name, item in captured.files.items():
            path_file_info = os.stat(
                name,
                dir_fd=captured.descriptor,
                follow_symlinks=False,
            )
            descriptor_file_info = os.fstat(item.descriptor)
            if (
                _stat_fingerprint(path_file_info) != item.fingerprint
                or _stat_fingerprint(descriptor_file_info) != item.fingerprint
                or _read_exact_descriptor(
                    item.descriptor,
                    expected_size=descriptor_file_info.st_size,
                    relative_path=item.relative_path,
                )
                != item.payload
            ):
                raise ExperimentResultIntegrityError(
                    f"audited artifact path or content changed: {item.relative_path}"
                )
        for child in captured.children.values():
            _require_captured_tree_current(child)
    except OSError as error:
        raise ExperimentResultIntegrityError(
            f"cannot finalize audited artifact tree {captured.path}: {error}"
        ) from error


@dataclass(frozen=True, slots=True)
class _ResourceSnapshot:
    user_cpu_seconds: float | None
    system_cpu_seconds: float | None
    process_peak_rss_bytes: int | None
    unavailable_reason: str | None


@dataclass(frozen=True, slots=True)
class _OperationalStart:
    wall_counter: float
    process_cpu_counter: float
    resources: _ResourceSnapshot


def _fail(message: str) -> NoReturn:
    raise ExperimentMatrixSchemaError(message)


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        _fail(f"{field} must be a stable ASCII identifier")
    return value


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail(f"{field} must be a lowercase SHA-256 digest")
    return value


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        _fail(f"{field} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{field} must be finite")
    return result


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _fail(f"{field} must be a positive integer")
    return value


def _non_negative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(f"{field} must be a non-negative integer")
    return value


def _timestamp(value: object, field: str) -> datetime:
    if not isinstance(value, datetime):
        _fail(f"{field} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        _fail(f"{field} must include an explicit timezone")
    try:
        return value.astimezone(UTC)
    except (OverflowError, ValueError) as error:
        raise ExperimentMatrixSchemaError(
            f"{field} is outside the UTC range"
        ) from error


def _parse_timestamp(value: object, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        _fail(f"{field} must be a non-empty ISO 8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ExperimentMatrixSchemaError(f"{field} must be ISO 8601") from error
    return _timestamp(parsed, field)


def _strict_fields(
    value: Mapping[str, object], expected: frozenset[str], context: str
) -> None:
    if frozenset(value) != expected:
        missing = sorted(expected - frozenset(value))
        unknown = sorted(frozenset(value) - expected)
        _fail(f"{context} fields changed: missing={missing!r}, unknown={unknown!r}")


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        _fail(f"{field} must be an object with string keys")
    return cast(Mapping[str, object], value)


def _array(value: object, field: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        _fail(f"{field} must be an array")
    return cast(Sequence[object], value)


def _relative_artifact_path(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        _fail(f"{field} must be a non-empty POSIX relative path")
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        _fail(f"{field} must stay inside the matrix run directory")
    if path.as_posix() != value:
        _fail(f"{field} must use canonical POSIX spelling")
    return value


def _resource_snapshot() -> _ResourceSnapshot:
    """Read process usage where the platform's stdlib exposes it."""
    try:
        import resource
    except ImportError:
        return _ResourceSnapshot(None, None, None, "resource_module_unavailable")
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        raw_peak = int(usage.ru_maxrss)
    except (OSError, OverflowError, ValueError) as error:
        return _ResourceSnapshot(
            None,
            None,
            None,
            f"resource_query_failed:{type(error).__name__}",
        )
    # Darwin reports bytes; Linux and the other supported CI POSIX targets
    # report KiB.  This is a process-wide high-water mark, not isolated RSS.
    factor = 1 if sys.platform == "darwin" else 1024
    return _ResourceSnapshot(
        user_cpu_seconds=float(usage.ru_utime),
        system_cpu_seconds=float(usage.ru_stime),
        process_peak_rss_bytes=raw_peak * factor,
        unavailable_reason=None,
    )


def _start_operational_measurement() -> _OperationalStart:
    return _OperationalStart(
        wall_counter=time.perf_counter(),
        process_cpu_counter=time.process_time(),
        resources=_resource_snapshot(),
    )


def _finish_operational_measurement(
    started: _OperationalStart,
) -> dict[str, object]:
    finished_resources = _resource_snapshot()
    user_cpu: float | None = None
    system_cpu: float | None = None
    peak_delta: int | None = None
    reason = (
        finished_resources.unavailable_reason or started.resources.unavailable_reason
    )
    if (
        started.resources.user_cpu_seconds is not None
        and finished_resources.user_cpu_seconds is not None
    ):
        user_cpu = max(
            0.0,
            finished_resources.user_cpu_seconds - started.resources.user_cpu_seconds,
        )
    if (
        started.resources.system_cpu_seconds is not None
        and finished_resources.system_cpu_seconds is not None
    ):
        system_cpu = max(
            0.0,
            finished_resources.system_cpu_seconds
            - started.resources.system_cpu_seconds,
        )
    if (
        started.resources.process_peak_rss_bytes is not None
        and finished_resources.process_peak_rss_bytes is not None
    ):
        peak_delta = max(
            0,
            finished_resources.process_peak_rss_bytes
            - started.resources.process_peak_rss_bytes,
        )
    return {
        "measurement_scope": "operational_non_scientific_process",
        "process_cpu_seconds": max(
            0.0, time.process_time() - started.process_cpu_counter
        ),
        "process_peak_rss_bytes": finished_resources.process_peak_rss_bytes,
        "process_peak_rss_delta_bytes": peak_delta,
        "resource_measurement_reason": reason,
        "system_cpu_seconds": system_cpu,
        "user_cpu_seconds": user_cpu,
        "wall_elapsed_seconds": max(0.0, time.perf_counter() - started.wall_counter),
    }


def _not_executed_measurement(reason: str) -> dict[str, object]:
    _identifier(reason, "measurement reason")
    return {
        "measurement_scope": "operational_non_scientific_process",
        "process_cpu_seconds": None,
        "process_peak_rss_bytes": None,
        "process_peak_rss_delta_bytes": None,
        "resource_measurement_reason": reason,
        "system_cpu_seconds": None,
        "user_cpu_seconds": None,
        "wall_elapsed_seconds": None,
    }


class MatrixDimension(StrEnum):
    """The nine experiment axes stated in the frozen research request."""

    MODEL_VARIANT = "model_variant"
    BEHAVIOR_REPRESENTATION = "behavior_representation"
    SOCIAL = "social"
    MEMORY = "memory"
    LLM = "llm"
    CALIBRATION = "calibration"
    RNG = "rng"
    MODEL_UNCERTAINTY = "model_uncertainty"
    ABSTENTION = "abstention"


class ModelVariant(StrEnum):
    FULL_MODEL = "full_model"
    HOMOGENEOUS_AGENTS = "homogeneous_agents"
    RATIONAL_ONLY = "rational_only"


class BehaviorRepresentation(StrEnum):
    PROMPTED_BEHAVIOR_LABELS = "prompted_behavior_labels"
    CALIBRATED_CONTINUOUS_BEHAVIOR = "calibrated_continuous_behavior"


class SocialMode(StrEnum):
    ENABLED = "enabled"
    NO_SOCIAL_GRAPH = "no_social_graph"


class MemoryMode(StrEnum):
    ENABLED = "enabled"
    NO_MEMORY = "no_memory"


class LLMMode(StrEnum):
    ENABLED = "enabled"
    NO_LLM = "no_LLM"


class CalibrationMode(StrEnum):
    POINT_CALIBRATION = "point_calibration"
    POSTERIOR_CALIBRATION = "posterior_calibration"


class RNGMode(StrEnum):
    INDEPENDENT_RNG = "independent_RNG"
    SAME_SEQUENTIAL_SEED = "same_sequential_seed"
    EVENT_KEYED_RNG = "event_keyed_RNG"


class ModelUncertaintyMode(StrEnum):
    SINGLE_MODEL = "single_model"
    STRUCTURAL_ENSEMBLE = "structural_ensemble"


class AbstentionMode(StrEnum):
    WITH_ABSTENTION = "with_abstention"
    WITHOUT_ABSTENTION = "without_abstention"


DIMENSION_LEVELS: tuple[tuple[MatrixDimension, tuple[str, ...]], ...] = (
    (MatrixDimension.MODEL_VARIANT, tuple(item.value for item in ModelVariant)),
    (
        MatrixDimension.BEHAVIOR_REPRESENTATION,
        tuple(item.value for item in BehaviorRepresentation),
    ),
    (MatrixDimension.SOCIAL, tuple(item.value for item in SocialMode)),
    (MatrixDimension.MEMORY, tuple(item.value for item in MemoryMode)),
    (MatrixDimension.LLM, tuple(item.value for item in LLMMode)),
    (MatrixDimension.CALIBRATION, tuple(item.value for item in CalibrationMode)),
    (MatrixDimension.RNG, tuple(item.value for item in RNGMode)),
    (
        MatrixDimension.MODEL_UNCERTAINTY,
        tuple(item.value for item in ModelUncertaintyMode),
    ),
    (MatrixDimension.ABSTENTION, tuple(item.value for item in AbstentionMode)),
)


@dataclass(frozen=True, slots=True)
class ExperimentFactors:
    """One fully specified point in the nine-dimensional design space."""

    model_variant: ModelVariant
    behavior_representation: BehaviorRepresentation
    social: SocialMode
    memory: MemoryMode
    llm: LLMMode
    calibration: CalibrationMode
    rng: RNGMode
    model_uncertainty: ModelUncertaintyMode
    abstention: AbstentionMode

    def __post_init__(self) -> None:
        expected = (
            (self.model_variant, ModelVariant, "model_variant"),
            (
                self.behavior_representation,
                BehaviorRepresentation,
                "behavior_representation",
            ),
            (self.social, SocialMode, "social"),
            (self.memory, MemoryMode, "memory"),
            (self.llm, LLMMode, "llm"),
            (self.calibration, CalibrationMode, "calibration"),
            (self.rng, RNGMode, "rng"),
            (
                self.model_uncertainty,
                ModelUncertaintyMode,
                "model_uncertainty",
            ),
            (self.abstention, AbstentionMode, "abstention"),
        )
        for value, enum_type, field in expected:
            if not isinstance(value, enum_type):
                _fail(f"{field} has an invalid level")

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def to_mapping(self) -> dict[str, object]:
        return {
            "abstention": self.abstention.value,
            "behavior_representation": self.behavior_representation.value,
            "calibration": self.calibration.value,
            "llm": self.llm.value,
            "memory": self.memory.value,
            "model_uncertainty": self.model_uncertainty.value,
            "model_variant": self.model_variant.value,
            "rng": self.rng.value,
            "social": self.social.value,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        """Strictly reconstruct one frozen factor combination."""
        fields = frozenset(dimension.value for dimension in MatrixDimension)
        _strict_fields(value, fields, "experiment factors")
        enum_fields = (
            ("model_variant", ModelVariant),
            ("behavior_representation", BehaviorRepresentation),
            ("social", SocialMode),
            ("memory", MemoryMode),
            ("llm", LLMMode),
            ("calibration", CalibrationMode),
            ("rng", RNGMode),
            ("model_uncertainty", ModelUncertaintyMode),
            ("abstention", AbstentionMode),
        )
        parsed: dict[str, StrEnum] = {}
        for field, enum_type in enum_fields:
            raw = value[field]
            if not isinstance(raw, str):
                _fail(f"{field} must be a string")
            try:
                parsed[field] = enum_type(raw)
            except ValueError as error:
                raise ExperimentMatrixSchemaError(
                    f"{field} has an unsupported level"
                ) from error
        return cls(
            model_variant=cast(ModelVariant, parsed["model_variant"]),
            behavior_representation=cast(
                BehaviorRepresentation, parsed["behavior_representation"]
            ),
            social=cast(SocialMode, parsed["social"]),
            memory=cast(MemoryMode, parsed["memory"]),
            llm=cast(LLMMode, parsed["llm"]),
            calibration=cast(CalibrationMode, parsed["calibration"]),
            rng=cast(RNGMode, parsed["rng"]),
            model_uncertainty=cast(ModelUncertaintyMode, parsed["model_uncertainty"]),
            abstention=cast(AbstentionMode, parsed["abstention"]),
        )


@dataclass(frozen=True, slots=True)
class FrozenInputReference:
    """Commitments proving model, predictions, and policy-after data are frozen."""

    reference_id: str
    data_manifest_sha256: str
    policy_after_data_sha256: str
    model_checkpoint_sha256: str
    prediction_artifact_sha256: str
    posterior_artifact_sha256: str
    environment_lock_sha256: str
    freeze_receipt_sha256: str
    model_frozen_at: datetime
    predictions_frozen_at: datetime
    policy_after_data_frozen_at: datetime
    evaluation_cutoff: datetime
    evaluation_started_at: datetime
    synthetic_only: bool

    def __post_init__(self) -> None:
        _identifier(self.reference_id, "reference_id")
        for field in (
            "data_manifest_sha256",
            "policy_after_data_sha256",
            "model_checkpoint_sha256",
            "prediction_artifact_sha256",
            "posterior_artifact_sha256",
            "environment_lock_sha256",
            "freeze_receipt_sha256",
        ):
            _sha256(getattr(self, field), field)
        for field in (
            "model_frozen_at",
            "predictions_frozen_at",
            "policy_after_data_frozen_at",
            "evaluation_cutoff",
            "evaluation_started_at",
        ):
            object.__setattr__(self, field, _timestamp(getattr(self, field), field))
        if self.model_frozen_at > self.policy_after_data_frozen_at:
            _fail(
                "model must be frozen before policy-after data are frozen "
                "for evaluation"
            )
        if self.predictions_frozen_at > self.policy_after_data_frozen_at:
            _fail(
                "predictions must be frozen before policy-after data are frozen for "
                "evaluation"
            )
        if self.evaluation_cutoff > self.policy_after_data_frozen_at:
            _fail("evaluation cutoff cannot follow the policy-after data freeze")
        if self.policy_after_data_frozen_at > self.evaluation_started_at:
            _fail("policy-after data must be frozen before evaluation starts")
        if not isinstance(self.synthetic_only, bool):
            _fail("synthetic_only must be a boolean")

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def to_mapping(self) -> dict[str, object]:
        return {
            "data_manifest_sha256": self.data_manifest_sha256,
            "environment_lock_sha256": self.environment_lock_sha256,
            "evaluation_cutoff": self.evaluation_cutoff.isoformat(),
            "evaluation_started_at": self.evaluation_started_at.isoformat(),
            "freeze_receipt_sha256": self.freeze_receipt_sha256,
            "model_checkpoint_sha256": self.model_checkpoint_sha256,
            "model_frozen_at": self.model_frozen_at.isoformat(),
            "policy_after_data_frozen_at": self.policy_after_data_frozen_at.isoformat(),
            "policy_after_data_sha256": self.policy_after_data_sha256,
            "posterior_artifact_sha256": self.posterior_artifact_sha256,
            "prediction_artifact_sha256": self.prediction_artifact_sha256,
            "predictions_frozen_at": self.predictions_frozen_at.isoformat(),
            "reference_id": self.reference_id,
            "synthetic_only": self.synthetic_only,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        """Strictly load the complete pre-evaluation freeze commitment."""
        _strict_fields(
            value,
            frozenset(
                {
                    "data_manifest_sha256",
                    "environment_lock_sha256",
                    "evaluation_cutoff",
                    "evaluation_started_at",
                    "freeze_receipt_sha256",
                    "model_checkpoint_sha256",
                    "model_frozen_at",
                    "policy_after_data_frozen_at",
                    "policy_after_data_sha256",
                    "posterior_artifact_sha256",
                    "prediction_artifact_sha256",
                    "predictions_frozen_at",
                    "reference_id",
                    "synthetic_only",
                }
            ),
            "frozen input reference",
        )
        synthetic_only = value["synthetic_only"]
        if not isinstance(synthetic_only, bool):
            _fail("synthetic_only must be a boolean")
        return cls(
            reference_id=_identifier(value["reference_id"], "reference_id"),
            data_manifest_sha256=_sha256(
                value["data_manifest_sha256"], "data_manifest_sha256"
            ),
            policy_after_data_sha256=_sha256(
                value["policy_after_data_sha256"], "policy_after_data_sha256"
            ),
            model_checkpoint_sha256=_sha256(
                value["model_checkpoint_sha256"], "model_checkpoint_sha256"
            ),
            prediction_artifact_sha256=_sha256(
                value["prediction_artifact_sha256"], "prediction_artifact_sha256"
            ),
            posterior_artifact_sha256=_sha256(
                value["posterior_artifact_sha256"], "posterior_artifact_sha256"
            ),
            environment_lock_sha256=_sha256(
                value["environment_lock_sha256"], "environment_lock_sha256"
            ),
            freeze_receipt_sha256=_sha256(
                value["freeze_receipt_sha256"], "freeze_receipt_sha256"
            ),
            model_frozen_at=_parse_timestamp(
                value["model_frozen_at"], "model_frozen_at"
            ),
            predictions_frozen_at=_parse_timestamp(
                value["predictions_frozen_at"], "predictions_frozen_at"
            ),
            policy_after_data_frozen_at=_parse_timestamp(
                value["policy_after_data_frozen_at"],
                "policy_after_data_frozen_at",
            ),
            evaluation_cutoff=_parse_timestamp(
                value["evaluation_cutoff"], "evaluation_cutoff"
            ),
            evaluation_started_at=_parse_timestamp(
                value["evaluation_started_at"], "evaluation_started_at"
            ),
            synthetic_only=synthetic_only,
        )


@dataclass(frozen=True, slots=True)
class ResourceBudget:
    """Frozen per-row local/cluster resource request."""

    cpu_cores: int
    memory_mb: int
    wall_time_seconds: int

    def __post_init__(self) -> None:
        _positive_int(self.cpu_cores, "cpu_cores")
        _positive_int(self.memory_mb, "memory_mb")
        _positive_int(self.wall_time_seconds, "wall_time_seconds")

    def to_mapping(self) -> dict[str, object]:
        return {
            "cpu_cores": self.cpu_cores,
            "memory_mb": self.memory_mb,
            "wall_time_seconds": self.wall_time_seconds,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(
            value,
            frozenset({"cpu_cores", "memory_mb", "wall_time_seconds"}),
            "resource budget",
        )
        return cls(
            cpu_cores=_positive_int(value["cpu_cores"], "cpu_cores"),
            memory_mb=_positive_int(value["memory_mb"], "memory_mb"),
            wall_time_seconds=_positive_int(
                value["wall_time_seconds"], "wall_time_seconds"
            ),
        )


@dataclass(frozen=True, slots=True)
class MCSEStopRule:
    """Frozen Monte Carlo stopping rule evaluated with Welford moments."""

    metric_name: str
    minimum_replicates: int
    maximum_replicates: int
    check_every: int
    target_mcse: float

    def __post_init__(self) -> None:
        _identifier(self.metric_name, "metric_name")
        minimum = _positive_int(self.minimum_replicates, "minimum_replicates")
        maximum = _positive_int(self.maximum_replicates, "maximum_replicates")
        check = _positive_int(self.check_every, "check_every")
        if minimum < 2:
            _fail("minimum_replicates must be at least two for MCSE")
        if maximum < minimum:
            _fail("maximum_replicates must not be below minimum_replicates")
        if check > maximum:
            _fail("check_every must not exceed maximum_replicates")
        target = _finite(self.target_mcse, "target_mcse")
        if target < 0:
            _fail("target_mcse must not be negative")
        object.__setattr__(self, "target_mcse", target)

    def to_mapping(self) -> dict[str, object]:
        return {
            "check_every": self.check_every,
            "maximum_replicates": self.maximum_replicates,
            "metric_name": self.metric_name,
            "minimum_replicates": self.minimum_replicates,
            "target_mcse": self.target_mcse,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(
            value,
            frozenset(
                {
                    "check_every",
                    "maximum_replicates",
                    "metric_name",
                    "minimum_replicates",
                    "target_mcse",
                }
            ),
            "MCSE stop rule",
        )
        return cls(
            metric_name=_identifier(value["metric_name"], "metric_name"),
            minimum_replicates=_positive_int(
                value["minimum_replicates"], "minimum_replicates"
            ),
            maximum_replicates=_positive_int(
                value["maximum_replicates"], "maximum_replicates"
            ),
            check_every=_positive_int(value["check_every"], "check_every"),
            target_mcse=_finite(value["target_mcse"], "target_mcse"),
        )


@dataclass(frozen=True, slots=True)
class SharedDrawIdentity:
    """Frozen synthetic contract for draws intentionally shared across rows.

    The contract is deliberately synthetic-only.  Its sample-manifest digest is
    an external byte commitment (normally the execution authority's event-key
    manifest), while ``event_namespace`` and ``shared_seed`` determine the
    engine address.  Rows without this explicit contract receive a distinct
    row-local identity and can never pass a cross-row paired-draw audit merely
    because their caller happened to copy event-key strings.
    """

    contract_id: str
    sample_manifest_sha256: str
    event_namespace: str
    shared_seed: str
    replicate_count: int
    synthetic_only: bool = True
    schema_version: str = DRAW_IDENTITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _identifier(self.contract_id, "draw identity contract_id")
        _sha256(
            self.sample_manifest_sha256,
            "draw identity sample_manifest_sha256",
        )
        _identifier(self.event_namespace, "draw identity event_namespace")
        if (
            not isinstance(self.shared_seed, str)
            or not self.shared_seed
            or len(self.shared_seed) > 512
        ):
            _fail("draw identity shared_seed must contain between 1 and 512 chars")
        if _positive_int(self.replicate_count, "draw identity replicate_count") < 2:
            _fail("draw identity replicate_count must be at least two")
        if self.synthetic_only is not True:
            _fail(
                "shared draw identity currently has only a synthetic execution "
                "trust root"
            )
        if self.schema_version != DRAW_IDENTITY_SCHEMA_VERSION:
            _fail("unsupported shared draw identity schema version")

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def to_mapping(self) -> dict[str, object]:
        return {
            "contract_id": self.contract_id,
            "event_namespace": self.event_namespace,
            "replicate_count": self.replicate_count,
            "sample_manifest_sha256": self.sample_manifest_sha256,
            "schema_version": self.schema_version,
            "shared_seed": self.shared_seed,
            "synthetic_only": self.synthetic_only,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(
            value,
            frozenset(
                {
                    "contract_id",
                    "event_namespace",
                    "replicate_count",
                    "sample_manifest_sha256",
                    "schema_version",
                    "shared_seed",
                    "synthetic_only",
                }
            ),
            "shared draw identity",
        )
        shared_seed = value["shared_seed"]
        schema_version = value["schema_version"]
        synthetic_only = value["synthetic_only"]
        if not isinstance(shared_seed, str) or not isinstance(schema_version, str):
            _fail("shared draw seed and schema version must be strings")
        if not isinstance(synthetic_only, bool):
            _fail("shared draw synthetic_only must be a boolean")
        return cls(
            contract_id=_identifier(value["contract_id"], "contract_id"),
            sample_manifest_sha256=_sha256(
                value["sample_manifest_sha256"], "sample_manifest_sha256"
            ),
            event_namespace=_identifier(value["event_namespace"], "event_namespace"),
            shared_seed=shared_seed,
            replicate_count=_positive_int(value["replicate_count"], "replicate_count"),
            synthetic_only=synthetic_only,
            schema_version=schema_version,
        )


@dataclass(frozen=True, slots=True)
class WelfordState:
    """Immutable online moments used only in a fixed observation order."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.count, bool)
            or not isinstance(self.count, int)
            or self.count < 0
        ):
            _fail("Welford count must be a non-negative integer")
        object.__setattr__(self, "mean", _finite(self.mean, "Welford mean"))
        object.__setattr__(self, "m2", _finite(self.m2, "Welford m2"))
        if self.m2 < -1.0e-12:
            _fail("Welford m2 must not be negative")
        if self.count == 0 and (self.mean != 0.0 or self.m2 != 0.0):
            _fail("empty Welford state must have zero moments")

    def update(self, value: float) -> Self:
        normalized = _finite(value, "Welford value")
        count = self.count + 1
        delta = normalized - self.mean
        mean = self.mean + delta / count
        delta_after = normalized - mean
        m2 = self.m2 + delta * delta_after
        if -1.0e-12 < m2 < 0.0:
            m2 = 0.0
        return type(self)(count=count, mean=mean, m2=m2)

    @property
    def mcse(self) -> float | None:
        if self.count < 2:
            return None
        sample_variance = self.m2 / (self.count - 1)
        if sample_variance < 0.0:
            _fail("sample variance is negative")
        return math.sqrt(sample_variance / self.count)

    def to_mapping(self) -> dict[str, object]:
        return {
            "count": self.count,
            "m2": self.m2,
            "mcse": self.mcse,
            "mean": self.mean,
        }


@dataclass(frozen=True, slots=True)
class ExperimentRow:
    """One explicit immutable row; row IDs do not influence its config hash."""

    row_id: str
    factors: ExperimentFactors
    frozen_inputs: FrozenInputReference
    stop_rule: MCSEStopRule
    resources: ResourceBudget
    seed: str
    dependencies: tuple[str, ...] = ()
    distribution_mcse_specs: tuple[DistributionMCSESpec, ...] = ()
    shared_draw_identity: SharedDrawIdentity | None = None

    def __post_init__(self) -> None:
        _identifier(self.row_id, "row_id")
        if not isinstance(self.factors, ExperimentFactors):
            _fail("factors must be ExperimentFactors")
        if not isinstance(self.frozen_inputs, FrozenInputReference):
            _fail("frozen_inputs must be a FrozenInputReference")
        if not isinstance(self.stop_rule, MCSEStopRule):
            _fail("stop_rule must be an MCSEStopRule")
        if not isinstance(self.resources, ResourceBudget):
            _fail("resources must be a ResourceBudget")
        if not isinstance(self.seed, str) or not self.seed or len(self.seed) > 512:
            _fail("seed must contain between one and 512 characters")
        if not isinstance(self.dependencies, tuple):
            _fail("dependencies must be an immutable tuple")
        normalized = tuple(
            sorted(_identifier(item, "dependency") for item in self.dependencies)
        )
        if len(set(normalized)) != len(normalized):
            _fail("dependencies must be unique")
        if self.row_id in normalized:
            _fail("a row cannot depend on itself")
        object.__setattr__(self, "dependencies", normalized)
        if not isinstance(self.distribution_mcse_specs, tuple):
            _fail("distribution_mcse_specs must be an immutable tuple")
        if not all(
            isinstance(spec, DistributionMCSESpec)
            for spec in self.distribution_mcse_specs
        ):
            _fail("distribution_mcse_specs contains an invalid spec")
        ordered_specs = tuple(
            sorted(self.distribution_mcse_specs, key=lambda spec: spec.spec_id)
        )
        if len({spec.spec_id for spec in ordered_specs}) != len(ordered_specs):
            _fail("distribution MCSE spec IDs must be unique")
        if len({spec.sha256 for spec in ordered_specs}) != len(ordered_specs):
            _fail("distribution MCSE specs must be semantically unique")
        semantic_targets = tuple(
            (
                spec.statistic,
                spec.quantile_level,
                spec.tail_threshold,
                spec.tail_direction,
            )
            for spec in ordered_specs
        )
        if len(set(semantic_targets)) != len(semantic_targets):
            _fail("distribution MCSE statistics must not be duplicated")
        if ordered_specs:
            design = (
                self.stop_rule.minimum_replicates,
                self.stop_rule.maximum_replicates,
                self.stop_rule.check_every,
            )
            for spec in ordered_specs:
                if (
                    spec.minimum_replicates,
                    spec.maximum_replicates,
                    spec.check_every,
                ) != design:
                    _fail(
                        "distribution MCSE checkpoints must match the scalar "
                        "scheduler envelope"
                    )
                if spec.synthetic_only != self.frozen_inputs.synthetic_only:
                    _fail("distribution MCSE synthetic marker changed")
                if spec.frozen_at >= self.frozen_inputs.policy_after_data_frozen_at:
                    _fail(
                        "distribution MCSE rules must be frozen before policy-after "
                        "data are frozen"
                    )
                if (
                    spec.target_outcomes_available_at
                    < self.frozen_inputs.policy_after_data_frozen_at
                ):
                    _fail(
                        "distribution MCSE target availability precedes the frozen "
                        "policy-after dataset"
                    )
        object.__setattr__(self, "distribution_mcse_specs", ordered_specs)
        contract = self.shared_draw_identity
        if contract is not None:
            if not isinstance(contract, SharedDrawIdentity):
                _fail("shared_draw_identity must be a SharedDrawIdentity or None")
            if not self.frozen_inputs.synthetic_only:
                _fail("shared draw identity cannot authorize non-synthetic matrix rows")
            if self.seed != contract.shared_seed:
                _fail("row seed must equal its shared draw identity seed")
            if self.stop_rule.maximum_replicates != contract.replicate_count:
                _fail(
                    "shared draw identity replicate_count must equal the frozen "
                    "maximum_replicates"
                )

    @property
    def draw_identity_sha256(self) -> str:
        """Return the shared contract hash or a deliberately row-local hash."""

        if self.shared_draw_identity is not None:
            return self.shared_draw_identity.sha256
        return sha256_bytes(
            canonical_json_bytes(
                {
                    "factors_sha256": self.factors.sha256,
                    "frozen_inputs_sha256": self.frozen_inputs.sha256,
                    "maximum_replicates": self.stop_rule.maximum_replicates,
                    "row_id": self.row_id,
                    "schema_version": ROW_LOCAL_DRAW_IDENTITY_SCHEMA_VERSION,
                    "seed": self.seed,
                }
            )
        )

    @property
    def has_shared_draw_identity(self) -> bool:
        """Whether this row is covered by an explicit cross-row contract."""

        return self.shared_draw_identity is not None

    @property
    def config_sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.configuration_mapping()))

    def configuration_mapping(self) -> dict[str, object]:
        mapping: dict[str, object] = {
            "dependencies": list(self.dependencies),
            "shared_draw_identity": (
                None
                if self.shared_draw_identity is None
                else self.shared_draw_identity.to_mapping()
            ),
            "factors": self.factors.to_mapping(),
            "frozen_inputs": self.frozen_inputs.to_mapping(),
            "resources": self.resources.to_mapping(),
            "seed": self.seed,
            "stop_rule": self.stop_rule.to_mapping(),
        }
        if self.distribution_mcse_specs:
            mapping["distribution_mcse_specs"] = [
                spec.to_mapping() for spec in self.distribution_mcse_specs
            ]
        return mapping

    def to_mapping(self) -> dict[str, object]:
        return {
            **self.configuration_mapping(),
            "config_sha256": self.config_sha256,
            "row_id": self.row_id,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        """Strictly load one row and verify its stored config commitment."""
        base_fields = frozenset(
            {
                "config_sha256",
                "dependencies",
                "factors",
                "frozen_inputs",
                "resources",
                "row_id",
                "seed",
                "shared_draw_identity",
                "stop_rule",
            }
        )
        actual = frozenset(value)
        if actual not in {base_fields, base_fields | {"distribution_mcse_specs"}}:
            missing = sorted(base_fields - actual)
            unknown = sorted(actual - (base_fields | {"distribution_mcse_specs"}))
            _fail(
                "experiment row fields changed: "
                f"missing={missing!r}, unknown={unknown!r}"
            )
        dependencies = tuple(
            _identifier(item, "dependency")
            for item in _array(value["dependencies"], "dependencies")
        )
        raw_seed = value["seed"]
        if not isinstance(raw_seed, str):
            _fail("seed must be a string")
        raw_shared = value["shared_draw_identity"]
        shared = (
            None
            if raw_shared is None
            else SharedDrawIdentity.from_mapping(
                _mapping(raw_shared, "shared_draw_identity")
            )
        )
        specs = tuple(
            DistributionMCSESpec.from_mapping(
                _mapping(item, "distribution_mcse_specs item")
            )
            for item in _array(
                value.get("distribution_mcse_specs", []),
                "distribution_mcse_specs",
            )
        )
        row = cls(
            row_id=_identifier(value["row_id"], "row_id"),
            factors=ExperimentFactors.from_mapping(
                _mapping(value["factors"], "factors")
            ),
            frozen_inputs=FrozenInputReference.from_mapping(
                _mapping(value["frozen_inputs"], "frozen_inputs")
            ),
            stop_rule=MCSEStopRule.from_mapping(
                _mapping(value["stop_rule"], "stop_rule")
            ),
            resources=ResourceBudget.from_mapping(
                _mapping(value["resources"], "resources")
            ),
            seed=raw_seed,
            dependencies=dependencies,
            distribution_mcse_specs=specs,
            shared_draw_identity=shared,
        )
        if row.config_sha256 != _sha256(value["config_sha256"], "config_sha256"):
            _fail("row config_sha256 does not match its frozen configuration")
        if row.to_mapping() != dict(value):
            _fail("experiment row is not in canonical serialized form")
        return row


class MatrixRegistrationKind(StrEnum):
    PREREGISTERED = "preregistered_explicit_rows"
    SMOKE = "preregistered_smoke_rows"


@dataclass(frozen=True, slots=True)
class ExperimentMatrix:
    """A finite explicit row registry with a validated dependency DAG."""

    matrix_id: str
    rows: tuple[ExperimentRow, ...]
    registration_kind: MatrixRegistrationKind
    cartesian_execution_allowed: bool = False
    schema_version: str = MATRIX_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _identifier(self.matrix_id, "matrix_id")
        if not isinstance(self.rows, tuple) or not self.rows:
            _fail("rows must be a non-empty immutable tuple")
        if not all(isinstance(row, ExperimentRow) for row in self.rows):
            _fail("rows must contain only ExperimentRow values")
        if not isinstance(self.registration_kind, MatrixRegistrationKind):
            _fail("registration_kind is invalid")
        if self.cartesian_execution_allowed is not False:
            _fail("Cartesian execution is forbidden; preregister explicit rows")
        if self.schema_version != MATRIX_SCHEMA_VERSION:
            _fail("unsupported experiment matrix schema version")
        row_ids = tuple(row.row_id for row in self.rows)
        if len(set(row_ids)) != len(row_ids):
            _fail("row IDs must be unique")
        hashes = tuple(row.config_sha256 for row in self.rows)
        if len(set(hashes)) != len(hashes):
            _fail("every row must have a unique semantic config hash")
        factor_hashes = tuple(row.factors.sha256 for row in self.rows)
        if len(set(factor_hashes)) != len(factor_hashes):
            _fail("duplicate factor combinations must not be preregistered twice")
        contract_ids: dict[str, str] = {}
        contract_namespaces: dict[str, str] = {}
        contract_groups: dict[str, list[ExperimentRow]] = {}
        for row in self.rows:
            contract = row.shared_draw_identity
            if contract is None:
                continue
            prior = contract_ids.setdefault(contract.contract_id, contract.sha256)
            if prior != contract.sha256:
                _fail("one shared draw contract_id cannot name different commitments")
            prior_namespace = contract_namespaces.setdefault(
                contract.event_namespace, contract.sha256
            )
            if prior_namespace != contract.sha256:
                _fail(
                    "one shared draw event_namespace cannot address different contracts"
                )
            contract_groups.setdefault(contract.sha256, []).append(row)
        for group in contract_groups.values():
            if len({row.frozen_inputs.sha256 for row in group}) != 1:
                _fail("shared draw rows must use the same frozen input reference")
            if len({row.seed for row in group}) != 1:
                _fail("shared draw rows must use the same explicit seed")
            if len({row.stop_rule.maximum_replicates for row in group}) != 1:
                _fail("shared draw rows must use the same replicate envelope")
        known = set(row_ids)
        for row in self.rows:
            unknown = sorted(set(row.dependencies) - known)
            if unknown:
                _fail(f"row {row.row_id!r} has unknown dependencies: {unknown!r}")
        self._validate_acyclic()

    def _validate_acyclic(self) -> None:
        dependencies = {row.row_id: set(row.dependencies) for row in self.rows}
        completed: set[str] = set()
        remaining = set(dependencies)
        while remaining:
            ready = {
                row_id for row_id in remaining if dependencies[row_id] <= completed
            }
            if not ready:
                _fail("experiment dependency graph contains a cycle")
            completed.update(ready)
            remaining.difference_update(ready)

    @property
    def topological_rows(self) -> tuple[ExperimentRow, ...]:
        pending = list(self.rows)
        completed: set[str] = set()
        ordered: list[ExperimentRow] = []
        while pending:
            ready_index = next(
                (
                    index
                    for index, row in enumerate(pending)
                    if set(row.dependencies) <= completed
                ),
                None,
            )
            if ready_index is None:
                _fail("experiment dependency graph cannot be ordered")
            row = pending.pop(ready_index)
            ordered.append(row)
            completed.add(row.row_id)
        return tuple(ordered)

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def row(self, row_id: str) -> ExperimentRow:
        for row in self.rows:
            if row.row_id == row_id:
                return row
        raise ExperimentMatrixSchemaError(f"unknown row_id: {row_id!r}")

    def to_mapping(self) -> dict[str, object]:
        return {
            "cartesian_execution_allowed": self.cartesian_execution_allowed,
            "dimension_levels": [
                {"dimension": dimension.value, "levels": list(levels)}
                for dimension, levels in DIMENSION_LEVELS
            ],
            "matrix_id": self.matrix_id,
            "registration_kind": self.registration_kind.value,
            "rows": [row.to_mapping() for row in self.rows],
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        """Strictly reconstruct a matrix without trusting caller-built objects."""
        _strict_fields(
            value,
            frozenset(
                {
                    "cartesian_execution_allowed",
                    "dimension_levels",
                    "matrix_id",
                    "registration_kind",
                    "rows",
                    "schema_version",
                }
            ),
            "experiment matrix",
        )
        expected_levels = [
            {"dimension": dimension.value, "levels": list(levels)}
            for dimension, levels in DIMENSION_LEVELS
        ]
        if value["dimension_levels"] != expected_levels:
            _fail("matrix dimension_levels differ from the implemented registry")
        registration = value["registration_kind"]
        schema_version = value["schema_version"]
        if not isinstance(registration, str) or not isinstance(schema_version, str):
            _fail("matrix registration_kind and schema_version must be strings")
        try:
            registration_kind = MatrixRegistrationKind(registration)
        except ValueError as error:
            raise ExperimentMatrixSchemaError(
                "matrix registration_kind is unsupported"
            ) from error
        cartesian = value["cartesian_execution_allowed"]
        if not isinstance(cartesian, bool):
            _fail("cartesian_execution_allowed must be a boolean")
        matrix = cls(
            matrix_id=_identifier(value["matrix_id"], "matrix_id"),
            rows=tuple(
                ExperimentRow.from_mapping(_mapping(item, "matrix row"))
                for item in _array(value["rows"], "rows")
            ),
            registration_kind=registration_kind,
            cartesian_execution_allowed=cartesian,
            schema_version=schema_version,
        )
        if matrix.to_mapping() != dict(value):
            _fail("experiment matrix is not in canonical serialized form")
        return matrix


class ResultStopReason(StrEnum):
    MCSE_THRESHOLD = "mcse_threshold_reached"
    MAXIMUM_REPLICATES = "maximum_replicates_reached"
    SHARED_DRAW_CONTRACT = "shared_draw_contract_completed"


@dataclass(frozen=True, slots=True)
class DistributionMCSERegistryRecord:
    """Compact per-statistic precision record committed by the registry."""

    spec_id: str
    estimate: DistributionMCSEEstimate

    def __post_init__(self) -> None:
        _identifier(self.spec_id, "distribution MCSE registry spec_id")
        if not isinstance(self.estimate, DistributionMCSEEstimate):
            _fail("distribution MCSE registry estimate is invalid")

    def to_mapping(self) -> dict[str, object]:
        return {"spec_id": self.spec_id, **self.estimate.to_mapping()}

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        expected = frozenset(
            {
                "bootstrap_replicates",
                "decision",
                "draws_sha256",
                "estimate",
                "mcse",
                "sample_count",
                "schema_version",
                "spec_id",
                "spec_sha256",
                "statistic",
                "synthetic_only",
            }
        )
        _strict_fields(value, expected, "distribution MCSE registry record")
        return cls(
            spec_id=_identifier(value["spec_id"], "distribution MCSE spec_id"),
            estimate=DistributionMCSEEstimate.from_mapping(
                {key: item for key, item in value.items() if key != "spec_id"}
            ),
        )


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Strict scalar or multi-statistic result with fully recomputable draws."""

    row_id: str
    config_sha256: str
    frozen_inputs_sha256: str
    metric_name: str
    runner_kind: str
    claim_scope: str
    mechanism_execution: bool
    draw_identity_sha256: str
    shared_draw_identity: bool
    samples: tuple[float, ...]
    event_keys: tuple[str, ...]
    mean: float
    m2: float
    mcse: float
    stop_reason: ResultStopReason
    synthetic_only: bool
    distribution_mcse_specs: tuple[DistributionMCSESpec, ...] = ()
    distribution_draws: tuple[MonteCarloDraw, ...] = ()
    distribution_estimates: tuple[DistributionMCSEEstimate, ...] = ()
    schema_version: str = RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _identifier(self.row_id, "result row_id")
        _sha256(self.config_sha256, "result config_sha256")
        _sha256(self.frozen_inputs_sha256, "result frozen_inputs_sha256")
        _identifier(self.metric_name, "result metric_name")
        _identifier(self.runner_kind, "result runner_kind")
        _identifier(self.claim_scope, "result claim_scope")
        if not isinstance(self.mechanism_execution, bool):
            _fail("result mechanism_execution must be a boolean")
        _sha256(self.draw_identity_sha256, "result draw_identity_sha256")
        if not isinstance(self.shared_draw_identity, bool):
            _fail("result shared_draw_identity must be a boolean")
        if not isinstance(self.samples, tuple) or len(self.samples) < 2:
            _fail("result samples must be an immutable tuple with at least two draws")
        normalized_samples = tuple(
            _finite(value, "result sample") for value in self.samples
        )
        object.__setattr__(self, "samples", normalized_samples)
        if not isinstance(self.event_keys, tuple) or len(self.event_keys) != len(
            self.samples
        ):
            _fail("result event_keys must match the sample count")
        if any(
            not isinstance(value, str) or _EVENT_KEY.fullmatch(value) is None
            for value in self.event_keys
        ):
            _fail("result contains an invalid event key")
        if len(set(self.event_keys)) != len(self.event_keys):
            _fail("result event keys must be unique")
        for field in ("mean", "m2", "mcse"):
            object.__setattr__(self, field, _finite(getattr(self, field), field))
        if self.m2 < -1.0e-12 or self.mcse < 0.0:
            _fail("result m2 and mcse must not be negative")
        if not isinstance(self.stop_reason, ResultStopReason):
            _fail("result stop_reason is invalid")
        if not isinstance(self.synthetic_only, bool):
            _fail("result synthetic_only must be a boolean")
        state = WelfordState()
        for sample in self.samples:
            state = state.update(sample)
        expected_mcse = state.mcse
        if expected_mcse is None:
            _fail("result has too few samples for MCSE")
        if not math.isclose(self.mean, state.mean, rel_tol=1.0e-12, abs_tol=1.0e-12):
            _fail("result mean does not match its raw samples")
        if not math.isclose(self.m2, state.m2, rel_tol=1.0e-12, abs_tol=1.0e-12):
            _fail("result m2 does not match its raw samples")
        if not math.isclose(self.mcse, expected_mcse, rel_tol=1.0e-12, abs_tol=1.0e-12):
            _fail("result MCSE does not match its raw samples")
        if self.schema_version == RESULT_SCHEMA_VERSION:
            if any(
                (
                    self.distribution_mcse_specs,
                    self.distribution_draws,
                    self.distribution_estimates,
                )
            ):
                _fail("scalar result cannot contain distribution MCSE fields")
        elif self.schema_version == DISTRIBUTION_RESULT_SCHEMA_VERSION:
            self._validate_distribution_fields()
        else:
            _fail("unsupported experiment result schema version")

    def _validate_distribution_fields(self) -> None:
        if (
            not isinstance(self.distribution_mcse_specs, tuple)
            or not self.distribution_mcse_specs
            or not all(
                isinstance(item, DistributionMCSESpec)
                for item in self.distribution_mcse_specs
            )
        ):
            _fail("distribution result requires immutable MCSE specs")
        if tuple(spec.spec_id for spec in self.distribution_mcse_specs) != tuple(
            sorted(spec.spec_id for spec in self.distribution_mcse_specs)
        ):
            _fail("distribution MCSE specs must be ordered by spec_id")
        if len({spec.spec_id for spec in self.distribution_mcse_specs}) != len(
            self.distribution_mcse_specs
        ):
            _fail("distribution MCSE spec IDs must be unique")
        if (
            not isinstance(self.distribution_draws, tuple)
            or len(self.distribution_draws) != len(self.samples)
            or not all(
                isinstance(item, MonteCarloDraw) for item in self.distribution_draws
            )
        ):
            _fail("distribution result requires one immutable draw per sample")
        if tuple(draw.replicate_id for draw in self.distribution_draws) != tuple(
            range(len(self.distribution_draws))
        ):
            _fail("distribution draw IDs must be contiguous from zero")
        if tuple(draw.value for draw in self.distribution_draws) != self.samples:
            _fail("distribution draw values do not match result samples")
        if len({draw.event_key.value for draw in self.distribution_draws}) != len(
            self.distribution_draws
        ):
            _fail("distribution semantic event keys must be unique")
        if (
            not isinstance(self.distribution_estimates, tuple)
            or len(self.distribution_estimates) != len(self.distribution_mcse_specs)
            or not all(
                isinstance(item, DistributionMCSEEstimate)
                for item in self.distribution_estimates
            )
        ):
            _fail("distribution result requires one estimate per MCSE spec")
        designs = {
            (
                spec.minimum_replicates,
                spec.maximum_replicates,
                spec.check_every,
            )
            for spec in self.distribution_mcse_specs
        }
        if len(designs) != 1:
            _fail("distribution result MCSE specs must share checkpoints")
        for spec, observed in zip(
            self.distribution_mcse_specs,
            self.distribution_estimates,
            strict=True,
        ):
            if spec.synthetic_only != self.synthetic_only:
                _fail("distribution result synthetic marker changed")
            expected = evaluate_distribution_mcse(spec, self.distribution_draws)
            if observed != expected:
                _fail(f"distribution MCSE estimate changed for spec {spec.spec_id!r}")
        count = len(self.samples)
        design = next(iter(designs))
        minimum, maximum, cadence = design
        if self.stop_reason is ResultStopReason.MCSE_THRESHOLD:
            if count < minimum or count % cadence != 0:
                _fail("distribution MCSE stop occurred off its frozen checkpoint")
            if not all(
                item.decision is MCSEStopDecision.TARGET_REACHED
                for item in self.distribution_estimates
            ):
                _fail("distribution MCSE stopped before every target was reached")
        elif self.stop_reason is ResultStopReason.MAXIMUM_REPLICATES:
            if count != maximum or not any(
                item.decision is MCSEStopDecision.MAXIMUM_REACHED
                for item in self.distribution_estimates
            ):
                _fail("distribution maximum stop hid an unmet statistic")
        elif count != maximum:
            _fail("shared draw contract did not exhaust its frozen draw count")

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def to_mapping(self) -> dict[str, object]:
        mapping: dict[str, object] = {
            "config_sha256": self.config_sha256,
            "draw_identity_sha256": self.draw_identity_sha256,
            "event_keys": list(self.event_keys),
            "frozen_inputs_sha256": self.frozen_inputs_sha256,
            "m2": self.m2,
            "mcse": self.mcse,
            "mean": self.mean,
            "mechanism_execution": self.mechanism_execution,
            "metric_name": self.metric_name,
            "claim_scope": self.claim_scope,
            "row_id": self.row_id,
            "runner_kind": self.runner_kind,
            "samples": list(self.samples),
            "schema_version": self.schema_version,
            "shared_draw_identity": self.shared_draw_identity,
            "stop_reason": self.stop_reason.value,
            "synthetic_only": self.synthetic_only,
        }
        if self.schema_version == DISTRIBUTION_RESULT_SCHEMA_VERSION:
            mapping.update(
                {
                    "distribution_draws": [
                        draw.to_mapping() for draw in self.distribution_draws
                    ],
                    "distribution_estimates": [
                        estimate.to_mapping()
                        for estimate in self.distribution_estimates
                    ],
                    "distribution_mcse_specs": [
                        spec.to_mapping() for spec in self.distribution_mcse_specs
                    ],
                }
            )
        return mapping

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        schema_version = value.get("schema_version")
        if not isinstance(schema_version, str):
            _fail("schema_version must be a string")
        distribution_fields = frozenset(
            {
                "distribution_draws",
                "distribution_estimates",
                "distribution_mcse_specs",
            }
        )
        if schema_version == RESULT_SCHEMA_VERSION:
            distribution_fields = frozenset()
        elif schema_version != DISTRIBUTION_RESULT_SCHEMA_VERSION:
            _fail("unsupported experiment result schema version")
        _strict_fields(
            value,
            frozenset(
                {
                    "config_sha256",
                    "draw_identity_sha256",
                    "event_keys",
                    "frozen_inputs_sha256",
                    "m2",
                    "mcse",
                    "mean",
                    "mechanism_execution",
                    "metric_name",
                    "claim_scope",
                    "row_id",
                    "runner_kind",
                    "samples",
                    "schema_version",
                    "shared_draw_identity",
                    "stop_reason",
                    "synthetic_only",
                    *distribution_fields,
                }
            ),
            "experiment result",
        )
        raw_samples = _array(value["samples"], "samples")
        raw_keys = _array(value["event_keys"], "event_keys")
        stop_reason = value["stop_reason"]
        if not isinstance(stop_reason, str):
            _fail("stop_reason must be a string")
        try:
            parsed_reason = ResultStopReason(stop_reason)
        except ValueError as error:
            raise ExperimentMatrixSchemaError("unsupported stop_reason") from error
        synthetic_only = value["synthetic_only"]
        if not isinstance(synthetic_only, bool):
            _fail("synthetic_only must be a boolean")
        mechanism_execution = value["mechanism_execution"]
        if not isinstance(mechanism_execution, bool):
            _fail("mechanism_execution must be a boolean")
        shared_draw_identity = value["shared_draw_identity"]
        if not isinstance(shared_draw_identity, bool):
            _fail("shared_draw_identity must be a boolean")
        raw_specs = (
            _array(value["distribution_mcse_specs"], "distribution_mcse_specs")
            if distribution_fields
            else ()
        )
        raw_draws = (
            _array(value["distribution_draws"], "distribution_draws")
            if distribution_fields
            else ()
        )
        raw_estimates = (
            _array(value["distribution_estimates"], "distribution_estimates")
            if distribution_fields
            else ()
        )
        return cls(
            row_id=_identifier(value["row_id"], "row_id"),
            config_sha256=_sha256(value["config_sha256"], "config_sha256"),
            frozen_inputs_sha256=_sha256(
                value["frozen_inputs_sha256"], "frozen_inputs_sha256"
            ),
            metric_name=_identifier(value["metric_name"], "metric_name"),
            runner_kind=_identifier(value["runner_kind"], "runner_kind"),
            claim_scope=_identifier(value["claim_scope"], "claim_scope"),
            mechanism_execution=mechanism_execution,
            draw_identity_sha256=_sha256(
                value["draw_identity_sha256"], "draw_identity_sha256"
            ),
            shared_draw_identity=shared_draw_identity,
            samples=tuple(_finite(item, "sample") for item in raw_samples),
            event_keys=tuple(_event_key_text(item, "event key") for item in raw_keys),
            mean=_finite(value["mean"], "mean"),
            m2=_finite(value["m2"], "m2"),
            mcse=_finite(value["mcse"], "mcse"),
            stop_reason=parsed_reason,
            synthetic_only=synthetic_only,
            distribution_mcse_specs=tuple(
                DistributionMCSESpec.from_mapping(_mapping(item, "MCSE spec"))
                for item in raw_specs
            ),
            distribution_draws=tuple(
                MonteCarloDraw.from_mapping(_mapping(item, "Monte Carlo draw"))
                for item in raw_draws
            ),
            distribution_estimates=tuple(
                DistributionMCSEEstimate.from_mapping(_mapping(item, "MCSE estimate"))
                for item in raw_estimates
            ),
            schema_version=schema_version,
        )


def _event_key_text(value: object, field: str) -> str:
    if not isinstance(value, str) or _EVENT_KEY.fullmatch(value) is None:
        _fail(f"{field} is invalid")
    return value


class ReplicateRunner(Protocol):
    """One deterministic scalar replicate supplied to the local scheduler."""

    @property
    def runner_kind(self) -> str:
        """Return a stable implementation identifier."""
        ...

    @property
    def claim_scope(self) -> str:
        """Return the narrowest scientific claim the output can support."""
        ...

    @property
    def mechanism_execution(self) -> bool:
        """Report whether declared scientific mechanisms actually executed."""
        ...

    def run_replicate(
        self, row: ExperimentRow, replicate: int, event_key: EventKey
    ) -> float:
        """Return a finite value or raise an explicit failure."""
        ...


@dataclass(frozen=True, slots=True)
class SyntheticMatrixRunner:
    """Offline deterministic runner for integration tests, never empirical use."""

    runner_kind: str = "synthetic_matrix_runner"
    claim_scope: str = "scheduler_smoke_only"
    mechanism_execution: bool = False
    centers: tuple[tuple[str, float], ...] = ()
    noise_scale: float = 0.0
    fail_row_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.centers, tuple):
            _fail("centers must be an immutable tuple")
        if self.runner_kind != "synthetic_matrix_runner":
            _fail("SyntheticMatrixRunner runner_kind cannot be changed")
        if self.claim_scope != "scheduler_smoke_only":
            _fail("SyntheticMatrixRunner claim_scope cannot be changed")
        if self.mechanism_execution is not False:
            _fail("SyntheticMatrixRunner cannot claim mechanism execution")
        normalized_centers: list[tuple[str, float]] = []
        for item in self.centers:
            if not isinstance(item, tuple) or len(item) != 2:
                _fail("each center must be a row-value tuple")
            normalized_centers.append(
                (_identifier(item[0], "center row_id"), _finite(item[1], "center"))
            )
        normalized_centers.sort()
        if len({row_id for row_id, _ in normalized_centers}) != len(normalized_centers):
            _fail("center row IDs must be unique")
        object.__setattr__(self, "centers", tuple(normalized_centers))
        scale = _finite(self.noise_scale, "noise_scale")
        if scale < 0:
            _fail("noise_scale must not be negative")
        object.__setattr__(self, "noise_scale", scale)
        if not isinstance(self.fail_row_ids, tuple):
            _fail("fail_row_ids must be an immutable tuple")
        normalized_failures = tuple(
            sorted(_identifier(item, "fail row_id") for item in self.fail_row_ids)
        )
        if len(set(normalized_failures)) != len(normalized_failures):
            _fail("fail_row_ids must be unique")
        object.__setattr__(self, "fail_row_ids", normalized_failures)

    def run_replicate(
        self, row: ExperimentRow, replicate: int, event_key: EventKey
    ) -> float:
        if not row.frozen_inputs.synthetic_only:
            raise SyntheticMatrixRunError(
                "SyntheticMatrixRunner cannot execute a non-synthetic matrix row"
            )
        if row.row_id in self.fail_row_ids:
            raise SyntheticMatrixRunError(
                f"deliberate synthetic failure for row {row.row_id}"
            )
        center = next(
            (value for row_id, value in self.centers if row_id == row.row_id), 0.0
        )
        noise_integer = deterministic_int(event_key, -1_000_000, 1_000_000)
        return center + self.noise_scale * noise_integer / 1_000_000.0


def _replicate_key(
    matrix: ExperimentMatrix, row: ExperimentRow, replicate: int
) -> EventKey:
    return EventKey(
        experiment_id=f"{matrix.matrix_id}.{row.row_id}",
        seed=row.seed,
        replicate_id=replicate,
        simulation_step=0,
        purpose="experiment_matrix.replicate",
    )


def _semantic_replicate_key(
    row: ExperimentRow,
    replicate: int,
) -> SemanticEventKey:
    """Return the frozen semantic identity for a distributional replicate."""
    contract = row.shared_draw_identity
    scenario_id = (
        f"shared-{contract.event_namespace}"
        if contract is not None
        else f"matrix-row-{row.config_sha256[:32]}"
    )
    return SemanticEventKey(
        scenario_id=scenario_id,
        event_type=EventType.EXOGENOUS_SHOCK,
        entity_id=f"replicate-{replicate}",
        timestamp=row.frozen_inputs.evaluation_started_at,
    )


def matrix_replicate_event_key(
    matrix: ExperimentMatrix, row: ExperimentRow, replicate: int
) -> EventKey:
    """Return the exact engine address used for one matrix replicate."""

    if isinstance(replicate, bool) or not isinstance(replicate, int) or replicate < 0:
        _fail("replicate must be a non-negative integer")
    if matrix.row(row.row_id) != row:
        _fail("replicate row is not the authoritative matrix row")
    if row.distribution_mcse_specs or row.shared_draw_identity is not None:
        return _semantic_replicate_key(row, replicate).to_legacy_event_key(
            seed=row.seed,
            replicate_id=replicate,
            simulation_step=0,
        )
    return _replicate_key(matrix, row, replicate)


def _operation_key(
    matrix: ExperimentMatrix, row: ExperimentRow, sequence: int, purpose: str
) -> str:
    return EventKey(
        experiment_id=f"{matrix.matrix_id}.{row.row_id}",
        seed=row.seed,
        replicate_id=0,
        simulation_step=sequence,
        purpose=f"experiment_matrix.{purpose}",
    ).value


def _execute_row(
    matrix: ExperimentMatrix, row: ExperimentRow, runner: ReplicateRunner
) -> ExperimentResult:
    state = WelfordState()
    samples: list[float] = []
    keys: list[str] = []
    distribution_draws: list[MonteCarloDraw] = []
    distribution_estimates: tuple[DistributionMCSEEstimate, ...] = ()
    stop_reason = ResultStopReason.MAXIMUM_REPLICATES
    for replicate in range(row.stop_rule.maximum_replicates):
        semantic_key = (
            _semantic_replicate_key(row, replicate)
            if row.distribution_mcse_specs
            else None
        )
        key = matrix_replicate_event_key(matrix, row, replicate)
        value = _finite(
            runner.run_replicate(row, replicate, key), "replicate runner output"
        )
        samples.append(value)
        keys.append(key.value)
        state = state.update(value)
        if semantic_key is not None:
            distribution_draws.append(
                MonteCarloDraw(
                    replicate_id=replicate,
                    event_key=semantic_key,
                    value=value,
                )
            )
        should_check = (
            state.count >= row.stop_rule.minimum_replicates
            and state.count % row.stop_rule.check_every == 0
        )
        if not should_check:
            continue
        if row.distribution_mcse_specs:
            frozen_draws = tuple(distribution_draws)
            distribution_estimates = tuple(
                evaluate_distribution_mcse(spec, frozen_draws)
                for spec in row.distribution_mcse_specs
            )
            if row.shared_draw_identity is None and all(
                estimate.decision is MCSEStopDecision.TARGET_REACHED
                for estimate in distribution_estimates
            ):
                stop_reason = ResultStopReason.MCSE_THRESHOLD
                break
        else:
            mcse = state.mcse
            if (
                row.shared_draw_identity is None
                and mcse is not None
                and mcse <= row.stop_rule.target_mcse
            ):
                stop_reason = ResultStopReason.MCSE_THRESHOLD
                break
    mcse = state.mcse
    if mcse is None:
        _fail("row completed without enough replicates for MCSE")
    if row.distribution_mcse_specs:
        distribution_estimates = tuple(
            evaluate_distribution_mcse(spec, tuple(distribution_draws))
            for spec in row.distribution_mcse_specs
        )
    if row.shared_draw_identity is not None:
        stop_reason = ResultStopReason.SHARED_DRAW_CONTRACT
    return ExperimentResult(
        row_id=row.row_id,
        config_sha256=row.config_sha256,
        frozen_inputs_sha256=row.frozen_inputs.sha256,
        metric_name=row.stop_rule.metric_name,
        runner_kind=_identifier(runner.runner_kind, "runner_kind"),
        claim_scope=_identifier(runner.claim_scope, "claim_scope"),
        mechanism_execution=runner.mechanism_execution,
        draw_identity_sha256=row.draw_identity_sha256,
        shared_draw_identity=row.has_shared_draw_identity,
        samples=tuple(samples),
        event_keys=tuple(keys),
        mean=state.mean,
        m2=state.m2,
        mcse=mcse,
        stop_reason=stop_reason,
        synthetic_only=row.frozen_inputs.synthetic_only,
        distribution_mcse_specs=row.distribution_mcse_specs,
        distribution_draws=tuple(distribution_draws),
        distribution_estimates=distribution_estimates,
        schema_version=(
            DISTRIBUTION_RESULT_SCHEMA_VERSION
            if row.distribution_mcse_specs
            else RESULT_SCHEMA_VERSION
        ),
    )


def _validate_result_for_row(result: ExperimentResult, row: ExperimentRow) -> None:
    if result.row_id != row.row_id:
        raise ExperimentResultIntegrityError("result row_id does not match its row")
    if result.config_sha256 != row.config_sha256:
        raise ExperimentResultIntegrityError(
            "result config hash does not match its row"
        )
    if result.frozen_inputs_sha256 != row.frozen_inputs.sha256:
        raise ExperimentResultIntegrityError("result frozen-input hash changed")
    if result.draw_identity_sha256 != row.draw_identity_sha256:
        raise ExperimentResultIntegrityError("result draw identity commitment changed")
    if result.shared_draw_identity is not row.has_shared_draw_identity:
        raise ExperimentResultIntegrityError("result shared draw identity flag changed")
    if result.metric_name != row.stop_rule.metric_name:
        raise ExperimentResultIntegrityError("result metric changed")
    count = len(result.samples)
    if (
        not row.stop_rule.minimum_replicates
        <= count
        <= row.stop_rule.maximum_replicates
    ):
        raise ExperimentResultIntegrityError(
            "result replicate count violates stop rule"
        )
    if row.shared_draw_identity is not None:
        expected_shared_keys = tuple(
            _semantic_replicate_key(row, replicate)
            .to_legacy_event_key(
                seed=row.seed,
                replicate_id=replicate,
                simulation_step=0,
            )
            .value
            for replicate in range(count)
        )
        if result.event_keys != expected_shared_keys:
            raise ExperimentResultIntegrityError(
                "result event keys do not match the shared draw contract"
            )
    if row.distribution_mcse_specs:
        if result.schema_version != DISTRIBUTION_RESULT_SCHEMA_VERSION:
            raise ExperimentResultIntegrityError(
                "distribution row used the scalar result schema"
            )
        if result.distribution_mcse_specs != row.distribution_mcse_specs:
            raise ExperimentResultIntegrityError(
                "result distribution MCSE specs changed from the frozen row"
            )
        expected_semantic_keys = tuple(
            _semantic_replicate_key(row, replicate) for replicate in range(count)
        )
        if tuple(draw.event_key for draw in result.distribution_draws) != (
            expected_semantic_keys
        ):
            raise ExperimentResultIntegrityError(
                "result distribution semantic event keys changed"
            )
        expected_legacy_keys = tuple(
            key.to_legacy_event_key(
                seed=row.seed,
                replicate_id=replicate,
                simulation_step=0,
            ).value
            for replicate, key in enumerate(expected_semantic_keys)
        )
        if result.event_keys != expected_legacy_keys:
            raise ExperimentResultIntegrityError(
                "result legacy event keys do not commit to semantic replicate keys"
            )
        expected_estimates = tuple(
            evaluate_distribution_mcse(spec, result.distribution_draws)
            for spec in row.distribution_mcse_specs
        )
        if result.distribution_estimates != expected_estimates:
            raise ExperimentResultIntegrityError(
                "result distribution MCSE estimates are not reproducible"
            )
        if result.stop_reason is ResultStopReason.MCSE_THRESHOLD:
            if count % row.stop_rule.check_every != 0 or not all(
                estimate.decision is MCSEStopDecision.TARGET_REACHED
                for estimate in result.distribution_estimates
            ):
                raise ExperimentResultIntegrityError(
                    "distribution row stopped before every MCSE target was met"
                )
        elif result.stop_reason is ResultStopReason.MAXIMUM_REPLICATES:
            if count != row.stop_rule.maximum_replicates or not any(
                estimate.decision is MCSEStopDecision.MAXIMUM_REACHED
                for estimate in result.distribution_estimates
            ):
                raise ExperimentResultIntegrityError(
                    "distribution maximum stop hid an unmet statistic"
                )
        elif (
            row.shared_draw_identity is None
            or count != row.shared_draw_identity.replicate_count
        ):
            raise ExperimentResultIntegrityError(
                "distribution shared draw contract was not exhausted"
            )
    else:
        if result.schema_version != RESULT_SCHEMA_VERSION:
            raise ExperimentResultIntegrityError(
                "scalar row used the distribution result schema"
            )
        if result.stop_reason is ResultStopReason.MCSE_THRESHOLD:
            if result.mcse > row.stop_rule.target_mcse:
                raise ExperimentResultIntegrityError("result claims an unmet MCSE stop")
            if count % row.stop_rule.check_every != 0:
                raise ExperimentResultIntegrityError(
                    "MCSE stop occurred off its frozen cadence"
                )
        elif result.stop_reason is ResultStopReason.MAXIMUM_REPLICATES:
            if count != row.stop_rule.maximum_replicates:
                raise ExperimentResultIntegrityError(
                    "maximum-replicate stop ended too early"
                )
        elif (
            row.shared_draw_identity is None
            or count != row.shared_draw_identity.replicate_count
        ):
            raise ExperimentResultIntegrityError(
                "scalar shared draw contract was not exhausted"
            )
    if result.synthetic_only != row.frozen_inputs.synthetic_only:
        raise ExperimentResultIntegrityError("result synthetic marker changed")


def write_result_bundle(directory: Path, result: ExperimentResult) -> str:
    """Write a new two-file result bundle and return its semantic commitment."""
    if directory.exists():
        raise ExperimentResultIntegrityError("result directory already exists")
    directory.mkdir(parents=True, exist_ok=False)
    write_json_file(directory / "result.json", result.to_mapping())
    write_text_file(directory / "result.sha256", f"{result.sha256}\n")
    return result.sha256


@dataclass(frozen=True, slots=True)
class ResultAuditReport:
    valid: bool
    row_id: str
    result_sha256: str
    draw_identity_sha256: str
    shared_draw_identity: bool
    sample_count: int
    stop_reason: ResultStopReason
    distribution_mcse: tuple[DistributionMCSERegistryRecord, ...] = ()
    execution_receipt_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.valid is not True:
            _fail("a positive result audit report must have valid=True")
        _identifier(self.row_id, "audit row_id")
        _sha256(self.result_sha256, "audit result_sha256")
        _sha256(self.draw_identity_sha256, "audit draw_identity_sha256")
        if not isinstance(self.shared_draw_identity, bool):
            _fail("audit shared_draw_identity must be boolean")
        _positive_int(self.sample_count, "audit sample_count")
        if not isinstance(self.stop_reason, ResultStopReason):
            _fail("audit stop_reason is invalid")
        if not isinstance(self.distribution_mcse, tuple) or not all(
            isinstance(item, DistributionMCSERegistryRecord)
            for item in self.distribution_mcse
        ):
            _fail("audit distribution MCSE records are invalid")
        if self.execution_receipt_sha256 is not None:
            _sha256(
                self.execution_receipt_sha256,
                "audit execution_receipt_sha256",
            )


def _distribution_registry_records(
    result: ExperimentResult,
) -> tuple[DistributionMCSERegistryRecord, ...]:
    return tuple(
        DistributionMCSERegistryRecord(spec_id=spec.spec_id, estimate=estimate)
        for spec, estimate in zip(
            result.distribution_mcse_specs,
            result.distribution_estimates,
            strict=True,
        )
    )


def audit_result_bundle(
    directory: Path,
    row: ExperimentRow,
    *,
    expected_result_sha256: str | None = None,
    expected_execution_receipt_sha256: str | None = None,
    execution_authority: SyntheticExecutionAuthority | None = None,
    matrix: ExperimentMatrix | None = None,
) -> ResultAuditReport:
    """Verify one result from stable bytes and reject any in-audit replacement."""
    if not row.frozen_inputs.synthetic_only:
        raise ExperimentResultIntegrityError(
            "non-synthetic result audit is blocked until an external execution "
            "and temporal authority is implemented"
        )
    captured = _open_captured_root(directory)
    try:
        observed_files = frozenset(os.listdir(captured.descriptor))
        receipt_files = frozenset(
            {"execution-receipt.json", "execution-receipt.sha256"}
        )
        observed_receipt_files = observed_files & receipt_files
        if observed_receipt_files and observed_receipt_files != receipt_files:
            raise ExperimentResultIntegrityError(
                "execution receipt bundle is incomplete"
            )
        if expected_execution_receipt_sha256 is not None:
            _sha256(
                expected_execution_receipt_sha256,
                "expected_execution_receipt_sha256",
            )
            if observed_receipt_files != receipt_files:
                raise ExperimentResultIntegrityError(
                    "expected execution receipt is missing"
                )
        expected_files = frozenset({"result.json", "result.sha256"}) | (
            receipt_files if observed_receipt_files else frozenset()
        )
        _require_exact_directory_entries(captured, expected_files)
        for name in sorted(expected_files):
            _capture_artifact_file(captured, name, relative_path=name)
        _require_captured_tree_current(captured, external_root=True)
        assert captured.files is not None
        try:
            mapping = read_json_bytes_object(
                captured.files["result.json"].payload,
                source=str(directory / "result.json"),
            )
            result = ExperimentResult.from_mapping(mapping)
            seal = captured.files["result.sha256"].payload.decode("ascii")
        except (
            DistributionMCSEError,
            ExperimentMatrixSchemaError,
            MetadataValidationError,
            SemanticEventError,
            UnicodeError,
        ) as error:
            raise ExperimentResultIntegrityError(str(error)) from error
        if seal != f"{result.sha256}\n":
            raise ExperimentResultIntegrityError("result semantic hash seal mismatch")
        if expected_result_sha256 is not None:
            _sha256(expected_result_sha256, "expected_result_sha256")
            if result.sha256 != expected_result_sha256:
                raise ExperimentResultIntegrityError(
                    "result changed after registry completion"
                )
        _validate_result_for_row(result, row)
        execution_receipt_sha256: str | None = None
        if observed_receipt_files:
            if execution_authority is None:
                raise ExperimentResultIntegrityError(
                    "execution receipt requires its live verifier authority"
                )
            if matrix is None:
                raise ExperimentResultIntegrityError(
                    "execution receipt audit requires the authoritative matrix"
                )
            try:
                from policytwin.experiments.execution_receipt import (
                    ExecutionReceiptError,
                    SyntheticExecutionReceipt,
                )

                receipt = SyntheticExecutionReceipt.from_mapping(
                    read_json_bytes_object(
                        captured.files["execution-receipt.json"].payload,
                        source=str(directory / "execution-receipt.json"),
                    )
                )
                receipt_seal = captured.files[
                    "execution-receipt.sha256"
                ].payload.decode("ascii")
                if receipt_seal != f"{receipt.sha256}\n":
                    raise ExperimentResultIntegrityError(
                        "execution receipt semantic seal mismatch"
                    )
                if receipt.result_file_sha256 != sha256_bytes(
                    captured.files["result.json"].payload
                ):
                    raise ExperimentResultIntegrityError(
                        "execution receipt does not bind captured result bytes"
                    )
                _require_captured_tree_current(captured, external_root=True)
                receipt_audit = execution_authority.verify(
                    receipt,
                    matrix,
                    row,
                    result,
                    directory / "result.json",
                )
            except ExecutionReceiptError as error:
                raise ExperimentResultIntegrityError(str(error)) from error
            execution_receipt_sha256 = receipt_audit.receipt_sha256
            if (
                expected_execution_receipt_sha256 is not None
                and execution_receipt_sha256 != expected_execution_receipt_sha256
            ):
                raise ExperimentResultIntegrityError(
                    "execution receipt changed after registry completion"
                )
        report = ResultAuditReport(
            valid=True,
            row_id=result.row_id,
            result_sha256=result.sha256,
            draw_identity_sha256=result.draw_identity_sha256,
            shared_draw_identity=result.shared_draw_identity,
            sample_count=len(result.samples),
            stop_reason=result.stop_reason,
            distribution_mcse=_distribution_registry_records(result),
            execution_receipt_sha256=execution_receipt_sha256,
        )
        _require_captured_tree_current(captured, external_root=True)
        return report
    except OSError as error:
        raise ExperimentResultIntegrityError(
            f"cannot audit result bundle: {error}"
        ) from error
    finally:
        captured.close()


class RegistryStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED_DEPENDENCY = "blocked_dependency"


@dataclass(frozen=True, slots=True)
class RegistryEntry:
    row_id: str
    config_sha256: str
    draw_identity_sha256: str
    shared_draw_identity: bool
    status: RegistryStatus
    result_sha256: str | None
    failure_code: str | None
    distribution_mcse: tuple[DistributionMCSERegistryRecord, ...] = ()
    execution_receipt_sha256: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.row_id, "registry row_id")
        _sha256(self.config_sha256, "registry config_sha256")
        _sha256(self.draw_identity_sha256, "registry draw_identity_sha256")
        if not isinstance(self.shared_draw_identity, bool):
            _fail("registry shared_draw_identity must be boolean")
        if not isinstance(self.status, RegistryStatus):
            _fail("registry status is invalid")
        if self.result_sha256 is not None:
            _sha256(self.result_sha256, "registry result_sha256")
        if self.failure_code is not None:
            _identifier(self.failure_code, "failure_code")
        if self.execution_receipt_sha256 is not None:
            _sha256(
                self.execution_receipt_sha256,
                "registry execution_receipt_sha256",
            )
        if not isinstance(self.distribution_mcse, tuple) or not all(
            isinstance(item, DistributionMCSERegistryRecord)
            for item in self.distribution_mcse
        ):
            _fail("registry distribution MCSE records are invalid")
        spec_ids = tuple(item.spec_id for item in self.distribution_mcse)
        if spec_ids != tuple(sorted(spec_ids)) or len(set(spec_ids)) != len(spec_ids):
            _fail("registry distribution MCSE records must be unique and sorted")
        if self.status is RegistryStatus.COMPLETED:
            if self.result_sha256 is None or self.failure_code is not None:
                _fail("completed registry entry requires only a result hash")
        elif self.result_sha256 is not None:
            _fail("non-completed registry entry must not claim a result hash")
        if (
            self.status is not RegistryStatus.COMPLETED
            and self.execution_receipt_sha256 is not None
        ):
            _fail("non-completed registry entry cannot claim an execution receipt")
        if self.status is not RegistryStatus.COMPLETED and self.distribution_mcse:
            _fail("non-completed registry entry cannot claim MCSE decisions")
        if (
            self.status
            in {
                RegistryStatus.FAILED,
                RegistryStatus.BLOCKED_DEPENDENCY,
            }
            and self.failure_code is None
        ):
            _fail("failed or blocked registry entries require a failure_code")
        if (
            self.status
            in {
                RegistryStatus.PENDING,
                RegistryStatus.RUNNING,
            }
            and self.failure_code is not None
        ):
            _fail("pending or running registry entries cannot claim a failure")

    def to_mapping(self) -> dict[str, object]:
        mapping: dict[str, object] = {
            "config_sha256": self.config_sha256,
            "draw_identity_sha256": self.draw_identity_sha256,
            "failure_code": self.failure_code,
            "result_sha256": self.result_sha256,
            "row_id": self.row_id,
            "shared_draw_identity": self.shared_draw_identity,
            "status": self.status.value,
        }
        if self.distribution_mcse:
            mapping["distribution_mcse"] = [
                item.to_mapping() for item in self.distribution_mcse
            ]
        if self.execution_receipt_sha256 is not None:
            mapping["execution_receipt_sha256"] = self.execution_receipt_sha256
        return mapping

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        optional_fields = frozenset(
            field
            for field in ("distribution_mcse", "execution_receipt_sha256")
            if field in value
        )
        _strict_fields(
            value,
            frozenset(
                {
                    "config_sha256",
                    "draw_identity_sha256",
                    "failure_code",
                    "result_sha256",
                    "row_id",
                    "shared_draw_identity",
                    "status",
                    *optional_fields,
                }
            ),
            "registry entry",
        )
        raw_status = value["status"]
        if not isinstance(raw_status, str):
            _fail("registry status must be a string")
        try:
            status = RegistryStatus(raw_status)
        except ValueError as error:
            raise ExperimentMatrixSchemaError("registry status is invalid") from error
        raw_result = value["result_sha256"]
        raw_failure = value["failure_code"]
        raw_shared = value["shared_draw_identity"]
        if not isinstance(raw_shared, bool):
            _fail("registry shared_draw_identity must be boolean")
        raw_distribution = (
            _array(value["distribution_mcse"], "distribution_mcse")
            if "distribution_mcse" in optional_fields
            else ()
        )
        if "distribution_mcse" in optional_fields and not raw_distribution:
            _fail("distribution_mcse must not be empty when present")
        if raw_result is not None:
            raw_result = _sha256(raw_result, "registry result_sha256")
        if raw_failure is not None:
            raw_failure = _identifier(raw_failure, "registry failure_code")
        return cls(
            row_id=_identifier(value["row_id"], "registry row_id"),
            config_sha256=_sha256(value["config_sha256"], "registry config_sha256"),
            draw_identity_sha256=_sha256(
                value["draw_identity_sha256"], "registry draw_identity_sha256"
            ),
            shared_draw_identity=raw_shared,
            status=status,
            result_sha256=raw_result,
            failure_code=raw_failure,
            distribution_mcse=tuple(
                DistributionMCSERegistryRecord.from_mapping(
                    _mapping(item, "distribution MCSE registry record")
                )
                for item in raw_distribution
            ),
            execution_receipt_sha256=(
                _sha256(
                    value["execution_receipt_sha256"],
                    "registry execution_receipt_sha256",
                )
                if "execution_receipt_sha256" in optional_fields
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class RunArtifactCommitment:
    """Byte-level commitment to one immutable file in a matrix run."""

    path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "path", _relative_artifact_path(self.path, "run artifact path")
        )
        _sha256(self.sha256, "run artifact sha256")
        _non_negative_int(self.size_bytes, "run artifact size_bytes")

    def to_mapping(self) -> dict[str, object]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(
            value,
            frozenset({"path", "sha256", "size_bytes"}),
            "run artifact commitment",
        )
        return cls(
            path=_relative_artifact_path(value["path"], "run artifact path"),
            sha256=_sha256(value["sha256"], "run artifact sha256"),
            size_bytes=_non_negative_int(
                value["size_bytes"], "run artifact size_bytes"
            ),
        )


@dataclass(frozen=True, slots=True)
class MatrixRunManifest:
    """Root commitment to the complete declared matrix, logs, and outcomes."""

    matrix_id: str
    matrix_sha256: str
    row_count: int
    completed_count: int
    failed_count: int
    blocked_dependency_count: int
    failure_count: int
    final_entries: tuple[RegistryEntry, ...]
    artifacts: tuple[RunArtifactCommitment, ...]
    cluster_submission_state: str = "not_submitted"
    schema_version: str = RUN_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _identifier(self.matrix_id, "run manifest matrix_id")
        _sha256(self.matrix_sha256, "run manifest matrix_sha256")
        for field in (
            "row_count",
            "completed_count",
            "failed_count",
            "blocked_dependency_count",
            "failure_count",
        ):
            _non_negative_int(getattr(self, field), f"run manifest {field}")
        if not isinstance(self.final_entries, tuple) or not self.final_entries:
            _fail("run manifest final_entries must be a non-empty immutable tuple")
        if not all(isinstance(item, RegistryEntry) for item in self.final_entries):
            _fail("run manifest contains an invalid final entry")
        row_ids = tuple(entry.row_id for entry in self.final_entries)
        if len(set(row_ids)) != len(row_ids):
            _fail("run manifest final row IDs must be unique")
        if self.row_count != len(self.final_entries):
            _fail("run manifest row_count does not match final_entries")
        observed_completed = sum(
            entry.status is RegistryStatus.COMPLETED for entry in self.final_entries
        )
        observed_failed = sum(
            entry.status is RegistryStatus.FAILED for entry in self.final_entries
        )
        observed_blocked = sum(
            entry.status is RegistryStatus.BLOCKED_DEPENDENCY
            for entry in self.final_entries
        )
        if any(
            entry.status in {RegistryStatus.PENDING, RegistryStatus.RUNNING}
            for entry in self.final_entries
        ):
            _fail("run manifest may contain only terminal row states")
        if (
            self.completed_count != observed_completed
            or self.failed_count != observed_failed
            or self.blocked_dependency_count != observed_blocked
        ):
            _fail("run manifest terminal status counts are inconsistent")
        if self.failure_count != observed_failed + observed_blocked:
            _fail("run manifest failure_count must include failed and blocked rows")
        if not isinstance(self.artifacts, tuple) or not self.artifacts:
            _fail("run manifest artifacts must be a non-empty immutable tuple")
        if not all(isinstance(item, RunArtifactCommitment) for item in self.artifacts):
            _fail("run manifest contains an invalid artifact commitment")
        artifact_paths = tuple(item.path for item in self.artifacts)
        if artifact_paths != tuple(sorted(artifact_paths)):
            _fail("run manifest artifacts must be sorted by path")
        if len(set(artifact_paths)) != len(artifact_paths):
            _fail("run manifest artifact paths must be unique")
        if self.cluster_submission_state != "not_submitted":
            _fail("matrix run manifest cannot claim cluster submission")
        if self.schema_version != RUN_MANIFEST_SCHEMA_VERSION:
            _fail("unsupported matrix run manifest schema version")

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def to_mapping(self) -> dict[str, object]:
        return {
            "artifacts": [item.to_mapping() for item in self.artifacts],
            "blocked_dependency_count": self.blocked_dependency_count,
            "cluster_submission_state": self.cluster_submission_state,
            "completed_count": self.completed_count,
            "failed_count": self.failed_count,
            "failure_count": self.failure_count,
            "final_entries": [item.to_mapping() for item in self.final_entries],
            "matrix_id": self.matrix_id,
            "matrix_sha256": self.matrix_sha256,
            "row_count": self.row_count,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(
            value,
            frozenset(
                {
                    "artifacts",
                    "blocked_dependency_count",
                    "cluster_submission_state",
                    "completed_count",
                    "failed_count",
                    "failure_count",
                    "final_entries",
                    "matrix_id",
                    "matrix_sha256",
                    "row_count",
                    "schema_version",
                }
            ),
            "matrix run manifest",
        )
        raw_entries = _array(value["final_entries"], "final_entries")
        raw_artifacts = _array(value["artifacts"], "artifacts")
        cluster_state = value["cluster_submission_state"]
        schema_version = value["schema_version"]
        if not isinstance(cluster_state, str) or not isinstance(schema_version, str):
            _fail("run manifest string fields are invalid")
        return cls(
            matrix_id=_identifier(value["matrix_id"], "run manifest matrix_id"),
            matrix_sha256=_sha256(value["matrix_sha256"], "run manifest matrix_sha256"),
            row_count=_non_negative_int(value["row_count"], "row_count"),
            completed_count=_non_negative_int(
                value["completed_count"], "completed_count"
            ),
            failed_count=_non_negative_int(value["failed_count"], "failed_count"),
            blocked_dependency_count=_non_negative_int(
                value["blocked_dependency_count"], "blocked_dependency_count"
            ),
            failure_count=_non_negative_int(value["failure_count"], "failure_count"),
            final_entries=tuple(
                RegistryEntry.from_mapping(_mapping(item, "final entry"))
                for item in raw_entries
            ),
            artifacts=tuple(
                RunArtifactCommitment.from_mapping(
                    _mapping(item, "artifact commitment")
                )
                for item in raw_artifacts
            ),
            cluster_submission_state=cluster_state,
            schema_version=schema_version,
        )


@dataclass(frozen=True, slots=True)
class MatrixRunAuditReport:
    """Positive result returned only after a complete run audit succeeds."""

    valid: bool
    matrix_id: str
    matrix_sha256: str
    manifest_sha256: str
    row_count: int
    completed_count: int
    failure_count: int
    execution_receipts_verified: bool

    def __post_init__(self) -> None:
        if self.valid is not True:
            _fail("a positive matrix run audit report must have valid=True")
        _identifier(self.matrix_id, "matrix run audit matrix_id")
        _sha256(self.matrix_sha256, "matrix run audit matrix_sha256")
        _sha256(self.manifest_sha256, "matrix run audit manifest_sha256")
        _positive_int(self.row_count, "matrix run audit row_count")
        _non_negative_int(self.completed_count, "matrix run audit completed_count")
        _non_negative_int(self.failure_count, "matrix run audit failure_count")
        if not isinstance(self.execution_receipts_verified, bool):
            _fail("matrix run audit execution_receipts_verified must be boolean")


@dataclass(frozen=True, slots=True)
class SharedDrawIdentityAuditReport:
    """Verifier-backed proof that every row in one synthetic family is paired."""

    valid: bool
    contract_id: str
    draw_identity_sha256: str
    sample_manifest_sha256: str
    event_keys_sha256: str
    matrix_sha256: str
    run_manifest_sha256: str
    row_ids: tuple[str, ...]
    result_sha256s: tuple[str, ...]
    sample_count: int
    execution_receipts_verified: bool
    synthetic_only: bool = True
    claim_scope: str = "synthetic_pairing_mechanics_only"

    def __post_init__(self) -> None:
        if self.valid is not True:
            _fail("positive shared draw audit must have valid=True")
        _identifier(self.contract_id, "shared draw audit contract_id")
        for field in (
            "draw_identity_sha256",
            "sample_manifest_sha256",
            "event_keys_sha256",
            "matrix_sha256",
            "run_manifest_sha256",
        ):
            _sha256(getattr(self, field), f"shared draw audit {field}")
        if len(self.row_ids) < 2 or len(set(self.row_ids)) != len(self.row_ids):
            _fail("shared draw audit requires at least two unique rows")
        if len(self.result_sha256s) != len(self.row_ids):
            _fail("shared draw audit result commitments do not match its rows")
        for row_id in self.row_ids:
            _identifier(row_id, "shared draw audit row_id")
        for digest in self.result_sha256s:
            _sha256(digest, "shared draw audit result_sha256")
        _positive_int(self.sample_count, "shared draw audit sample_count")
        if self.execution_receipts_verified is not True:
            _fail("shared draw audit requires verifier-owned execution receipts")
        if self.synthetic_only is not True:
            _fail("shared draw audit is synthetic-only")
        if self.claim_scope != "synthetic_pairing_mechanics_only":
            _fail("shared draw audit claim scope cannot be widened")


def audit_shared_draw_identity(
    matrix: ExperimentMatrix,
    manifest: MatrixRunManifest,
    run_audit: MatrixRunAuditReport,
    contract_id: str,
    results: tuple[ExperimentResult, ...],
) -> SharedDrawIdentityAuditReport:
    """Fail closed unless a complete verifier-owned synthetic family is paired.

    Every matrix row carrying ``contract_id`` is mandatory.  A failed or
    dependency-blocked member therefore remains visible and makes comparison
    impossible instead of silently shrinking the sample or ablation set.
    """

    requested = _identifier(contract_id, "shared draw contract_id")
    if manifest.matrix_sha256 != matrix.sha256:
        raise ExperimentResultIntegrityError(
            "shared draw audit manifest does not match the matrix"
        )
    if (
        run_audit.matrix_sha256 != matrix.sha256
        or run_audit.manifest_sha256 != manifest.sha256
    ):
        raise ExperimentResultIntegrityError(
            "shared draw audit does not match the audited run manifest"
        )
    if not run_audit.execution_receipts_verified:
        raise ExperimentResultIntegrityError(
            "shared draw audit requires verifier-owned synthetic execution receipts"
        )
    rows = tuple(
        row
        for row in matrix.topological_rows
        if row.shared_draw_identity is not None
        and row.shared_draw_identity.contract_id == requested
    )
    if len(rows) < 2:
        raise ExperimentResultIntegrityError(
            "shared draw contract must cover at least two matrix rows"
        )
    contracts = tuple(
        cast(SharedDrawIdentity, row.shared_draw_identity) for row in rows
    )
    contract = contracts[0]
    if any(item != contract for item in contracts[1:]):
        raise ExperimentResultIntegrityError(
            "shared draw contract_id resolves to inconsistent commitments"
        )
    entries_by_row = {entry.row_id: entry for entry in manifest.final_entries}
    terminal = tuple(entries_by_row[row.row_id] for row in rows)
    incomplete = tuple(
        f"{entry.row_id}:{entry.status.value}"
        for entry in terminal
        if entry.status is not RegistryStatus.COMPLETED
    )
    if incomplete:
        raise ExperimentResultIntegrityError(
            "shared draw comparison cannot omit failed rows: " + ", ".join(incomplete)
        )
    if tuple(result.row_id for result in results) != tuple(row.row_id for row in rows):
        raise ExperimentResultIntegrityError(
            "shared draw results must include every contract row in matrix order"
        )
    for row, entry, result in zip(rows, terminal, results, strict=True):
        _validate_result_for_row(result, row)
        if (
            entry.result_sha256 != result.sha256
            or entry.draw_identity_sha256 != result.draw_identity_sha256
            or not entry.shared_draw_identity
        ):
            raise ExperimentResultIntegrityError(
                f"shared draw registry/result binding changed for {row.row_id!r}"
            )
        if entry.execution_receipt_sha256 is None:
            raise ExperimentResultIntegrityError(
                f"shared draw row {row.row_id!r} lacks an execution receipt"
            )
    reference_keys = results[0].event_keys
    if any(result.event_keys != reference_keys for result in results[1:]):
        raise ExperimentResultIntegrityError(
            "shared draw rows have mismatched EventKey sequences"
        )
    if len(reference_keys) != contract.replicate_count:
        raise ExperimentResultIntegrityError(
            "shared draw rows did not exhaust the frozen sample manifest"
        )
    return SharedDrawIdentityAuditReport(
        valid=True,
        contract_id=contract.contract_id,
        draw_identity_sha256=contract.sha256,
        sample_manifest_sha256=contract.sample_manifest_sha256,
        event_keys_sha256=sha256_bytes(canonical_json_bytes(list(reference_keys))),
        matrix_sha256=matrix.sha256,
        run_manifest_sha256=manifest.sha256,
        row_ids=tuple(row.row_id for row in rows),
        result_sha256s=tuple(result.sha256 for result in results),
        sample_count=len(reference_keys),
        execution_receipts_verified=True,
    )


@dataclass(frozen=True, slots=True)
class MatrixRunReport:
    matrix_id: str
    matrix_sha256: str
    manifest_sha256: str
    output_directory: str
    entries: tuple[RegistryEntry, ...]

    def __post_init__(self) -> None:
        _identifier(self.matrix_id, "run report matrix_id")
        _sha256(self.matrix_sha256, "run report matrix_sha256")
        _sha256(self.manifest_sha256, "run report manifest_sha256")
        if not isinstance(self.output_directory, str) or not self.output_directory:
            _fail("output_directory must be a non-empty string")
        if not isinstance(self.entries, tuple) or not self.entries:
            _fail("entries must be a non-empty immutable tuple")
        if not all(isinstance(entry, RegistryEntry) for entry in self.entries):
            _fail("entries must contain only RegistryEntry values")
        row_ids = tuple(entry.row_id for entry in self.entries)
        if len(set(row_ids)) != len(row_ids):
            _fail("run report row IDs must be unique")

    def entry(self, row_id: str) -> RegistryEntry:
        for entry in self.entries:
            if entry.row_id == row_id:
                return entry
        raise ExperimentMatrixSchemaError(f"unknown registry row: {row_id!r}")

    def to_mapping(self) -> dict[str, object]:
        return {
            "entries": [entry.to_mapping() for entry in self.entries],
            "matrix_id": self.matrix_id,
            "matrix_sha256": self.matrix_sha256,
            "manifest_sha256": self.manifest_sha256,
            "output_directory": self.output_directory,
        }


def run_local_matrix(
    matrix: ExperimentMatrix,
    output_directory: Path,
    runner: ReplicateRunner,
    *,
    execution_authority: SyntheticExecutionAuthority | None = None,
) -> MatrixRunReport:
    """Run dependency-ready rows locally and write five structured JSONL logs."""
    if not isinstance(matrix, ExperimentMatrix):
        _fail("matrix must be an ExperimentMatrix")
    if any(not row.frozen_inputs.synthetic_only for row in matrix.rows):
        raise ExperimentResultIntegrityError(
            "non-synthetic matrix execution is blocked until an external execution "
            "and temporal authority is implemented"
        )
    if execution_authority is not None:
        try:
            execution_authority.ensure_runner(matrix, runner)
        except Exception as error:
            raise ExperimentResultIntegrityError(
                f"execution receipt preflight failed: {error}"
            ) from error
    if output_directory.exists():
        _fail("matrix output directory already exists")
    output_directory.mkdir(parents=True, exist_ok=False)
    (output_directory / "runs").mkdir()
    write_json_file(output_directory / "matrix.json", matrix.to_mapping())
    write_text_file(output_directory / "matrix.sha256", f"{matrix.sha256}\n")
    final_entries: dict[str, RegistryEntry] = {}
    sequence = 0
    failure_count = 0
    with (
        JsonlEventWriter(output_directory / "registry.jsonl") as registry_log,
        JsonlEventWriter(output_directory / "dependencies.jsonl") as dependency_log,
        JsonlEventWriter(output_directory / "resources.jsonl") as resource_log,
        JsonlEventWriter(output_directory / "timing.jsonl") as timing_log,
        JsonlEventWriter(output_directory / "failures.jsonl") as failure_log,
    ):
        for row in matrix.topological_rows:
            pending = RegistryEntry(
                row_id=row.row_id,
                config_sha256=row.config_sha256,
                draw_identity_sha256=row.draw_identity_sha256,
                shared_draw_identity=row.has_shared_draw_identity,
                status=RegistryStatus.PENDING,
                result_sha256=None,
                failure_code=None,
            )
            registry_log.write(
                {
                    **pending.to_mapping(),
                    "event_key": _operation_key(
                        matrix, row, sequence, "registry.pending"
                    ),
                    "matrix_id": matrix.matrix_id,
                    "schema_version": 1,
                    "sequence": sequence,
                }
            )
            sequence += 1
            dependency_log.write(
                {
                    "dependencies": list(row.dependencies),
                    "event_key": _operation_key(
                        matrix, row, sequence, "dependencies.declared"
                    ),
                    "matrix_id": matrix.matrix_id,
                    "row_id": row.row_id,
                    "schema_version": 1,
                    "sequence": sequence,
                }
            )
            sequence += 1

        for row in matrix.topological_rows:
            blocked_by = tuple(
                dependency
                for dependency in row.dependencies
                if final_entries[dependency].status is not RegistryStatus.COMPLETED
            )
            if blocked_by:
                entry = RegistryEntry(
                    row_id=row.row_id,
                    config_sha256=row.config_sha256,
                    draw_identity_sha256=row.draw_identity_sha256,
                    shared_draw_identity=row.has_shared_draw_identity,
                    status=RegistryStatus.BLOCKED_DEPENDENCY,
                    result_sha256=None,
                    failure_code="dependency_not_completed",
                )
                final_entries[row.row_id] = entry
                registry_log.write(
                    {
                        **entry.to_mapping(),
                        "event_key": _operation_key(
                            matrix, row, sequence, "registry.blocked"
                        ),
                        "matrix_id": matrix.matrix_id,
                        "schema_version": 1,
                        "sequence": sequence,
                    }
                )
                sequence += 1
                failure_log.write(
                    {
                        "blocked_by": list(blocked_by),
                        "event_key": _operation_key(
                            matrix, row, sequence, "failure.dependency"
                        ),
                        "failure_code": "dependency_not_completed",
                        "failure_type": "blocked_dependency",
                        "matrix_id": matrix.matrix_id,
                        "row_id": row.row_id,
                        "schema_version": 1,
                        "sequence": sequence,
                    }
                )
                failure_count += 1
                sequence += 1
                blocked_measurement = _not_executed_measurement("blocked_dependency")
                resource_log.write(
                    {
                        **blocked_measurement,
                        "event": "not_executed",
                        "event_key": _operation_key(
                            matrix, row, sequence, "resource.blocked"
                        ),
                        "matrix_id": matrix.matrix_id,
                        "row_id": row.row_id,
                        "schema_version": 1,
                        "sequence": sequence,
                    }
                )
                sequence += 1
                timing_log.write(
                    {
                        **blocked_measurement,
                        "event": "blocked",
                        "event_key": _operation_key(
                            matrix, row, sequence, "timing.blocked"
                        ),
                        "matrix_id": matrix.matrix_id,
                        "row_id": row.row_id,
                        "schema_version": 1,
                        "sequence": sequence,
                        "time_basis": "operational_measurement_unavailable",
                    }
                )
                sequence += 1
                continue

            running = RegistryEntry(
                row_id=row.row_id,
                config_sha256=row.config_sha256,
                draw_identity_sha256=row.draw_identity_sha256,
                shared_draw_identity=row.has_shared_draw_identity,
                status=RegistryStatus.RUNNING,
                result_sha256=None,
                failure_code=None,
            )
            registry_log.write(
                {
                    **running.to_mapping(),
                    "event_key": _operation_key(
                        matrix, row, sequence, "registry.running"
                    ),
                    "matrix_id": matrix.matrix_id,
                    "schema_version": 1,
                    "sequence": sequence,
                }
            )
            sequence += 1
            resource_log.write(
                {
                    "config_sha256": row.config_sha256,
                    "event": "requested",
                    "event_key": _operation_key(
                        matrix, row, sequence, "resource.requested"
                    ),
                    "matrix_id": matrix.matrix_id,
                    "resources": row.resources.to_mapping(),
                    "row_id": row.row_id,
                    "schema_version": 1,
                    "sequence": sequence,
                }
            )
            sequence += 1
            timing_log.write(
                {
                    "event": "started",
                    "event_key": _operation_key(
                        matrix, row, sequence, "timing.started"
                    ),
                    "matrix_id": matrix.matrix_id,
                    "row_id": row.row_id,
                    "schema_version": 1,
                    "sequence": sequence,
                    "time_basis": "deterministic_logical_sequence",
                }
            )
            sequence += 1
            measurement_started = _start_operational_measurement()
            try:
                result = _execute_row(matrix, row, runner)
                result_directory = output_directory / "runs" / row.row_id
                result_sha256 = write_result_bundle(result_directory, result)
                execution_receipt_sha256: str | None = None
                if execution_authority is not None:
                    from policytwin.experiments.execution_receipt import (
                        write_execution_receipt,
                    )

                    receipt = execution_authority.issue(
                        matrix,
                        row,
                        result,
                        result_directory / "result.json",
                    )
                    execution_receipt_sha256 = write_execution_receipt(
                        result_directory, receipt
                    )
                audit_result_bundle(
                    result_directory,
                    row,
                    expected_result_sha256=result_sha256,
                    expected_execution_receipt_sha256=execution_receipt_sha256,
                    execution_authority=execution_authority,
                    matrix=matrix,
                )
            except Exception as error:
                measurement = _finish_operational_measurement(measurement_started)
                failure_code = "row_execution_failed"
                entry = RegistryEntry(
                    row_id=row.row_id,
                    config_sha256=row.config_sha256,
                    draw_identity_sha256=row.draw_identity_sha256,
                    shared_draw_identity=row.has_shared_draw_identity,
                    status=RegistryStatus.FAILED,
                    result_sha256=None,
                    failure_code=failure_code,
                )
                final_entries[row.row_id] = entry
                registry_log.write(
                    {
                        **entry.to_mapping(),
                        "event_key": _operation_key(
                            matrix, row, sequence, "registry.failed"
                        ),
                        "matrix_id": matrix.matrix_id,
                        "schema_version": 1,
                        "sequence": sequence,
                    }
                )
                sequence += 1
                failure_log.write(
                    {
                        "event_key": _operation_key(
                            matrix, row, sequence, "failure.execution"
                        ),
                        "failure_code": failure_code,
                        "failure_message": str(error)[:512],
                        "failure_type": type(error).__name__,
                        "matrix_id": matrix.matrix_id,
                        "row_id": row.row_id,
                        "schema_version": 1,
                        "sequence": sequence,
                    }
                )
                failure_count += 1
                sequence += 1
                resource_log.write(
                    {
                        **measurement,
                        "event": "observed_failed",
                        "event_key": _operation_key(
                            matrix, row, sequence, "resource.failed"
                        ),
                        "matrix_id": matrix.matrix_id,
                        "row_id": row.row_id,
                        "schema_version": 1,
                        "sequence": sequence,
                    }
                )
                sequence += 1
                timing_log.write(
                    {
                        **measurement,
                        "event": "failed",
                        "event_key": _operation_key(
                            matrix, row, sequence, "timing.failed"
                        ),
                        "matrix_id": matrix.matrix_id,
                        "row_id": row.row_id,
                        "schema_version": 1,
                        "sequence": sequence,
                        "time_basis": "measured_operational_elapsed",
                    }
                )
                sequence += 1
                continue

            measurement = _finish_operational_measurement(measurement_started)
            resource_log.write(
                {
                    **measurement,
                    "event": "observed_completed",
                    "event_key": _operation_key(
                        matrix, row, sequence, "resource.completed"
                    ),
                    "matrix_id": matrix.matrix_id,
                    "row_id": row.row_id,
                    "schema_version": 1,
                    "sequence": sequence,
                }
            )
            sequence += 1
            entry = RegistryEntry(
                row_id=row.row_id,
                config_sha256=row.config_sha256,
                draw_identity_sha256=row.draw_identity_sha256,
                shared_draw_identity=row.has_shared_draw_identity,
                status=RegistryStatus.COMPLETED,
                result_sha256=result_sha256,
                failure_code=None,
                distribution_mcse=_distribution_registry_records(result),
                execution_receipt_sha256=execution_receipt_sha256,
            )
            final_entries[row.row_id] = entry
            registry_log.write(
                {
                    **entry.to_mapping(),
                    "event_key": _operation_key(
                        matrix, row, sequence, "registry.completed"
                    ),
                    "matrix_id": matrix.matrix_id,
                    "schema_version": 1,
                    "sequence": sequence,
                }
            )
            sequence += 1
            timing_log.write(
                {
                    **measurement,
                    "event": "completed",
                    "event_key": _operation_key(
                        matrix, row, sequence, "timing.completed"
                    ),
                    "matrix_id": matrix.matrix_id,
                    "result_sha256": result_sha256,
                    "row_id": row.row_id,
                    "schema_version": 1,
                    "sequence": sequence,
                    "time_basis": "measured_operational_elapsed",
                }
            )
            sequence += 1

        seal_row = matrix.topological_rows[0]
        failure_log.write(
            {
                "event_key": _operation_key(
                    matrix, seal_row, sequence, "failure.registry_sealed"
                ),
                "event_type": "failure_registry_sealed",
                "failure_count": failure_count,
                "matrix_id": matrix.matrix_id,
                "schema_version": 1,
                "sequence": sequence,
            }
        )

    entries = tuple(final_entries[row.row_id] for row in matrix.topological_rows)
    generate_cluster_batch_plan(matrix, output_directory / "cluster-plan.json")
    manifest = write_matrix_run_manifest(output_directory, matrix, entries)
    audit_matrix_run(
        output_directory,
        matrix,
        expected_manifest_sha256=manifest.sha256,
        execution_authority=execution_authority,
    )
    return MatrixRunReport(
        matrix_id=matrix.matrix_id,
        matrix_sha256=matrix.sha256,
        manifest_sha256=manifest.sha256,
        output_directory=str(output_directory.resolve()),
        entries=entries,
    )


class ClusterSubmissionState(StrEnum):
    NOT_SUBMITTED = "not_submitted"


@dataclass(frozen=True, slots=True)
class ClusterBatchTask:
    task_index: int
    row_id: str
    config_sha256: str
    dependencies: tuple[str, ...]
    resources: ResourceBudget
    event_key: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.task_index, bool)
            or not isinstance(self.task_index, int)
            or self.task_index < 0
        ):
            _fail("task_index must be a non-negative integer")
        _identifier(self.row_id, "cluster task row_id")
        _sha256(self.config_sha256, "cluster task config_sha256")
        if not isinstance(self.dependencies, tuple):
            _fail("cluster task dependencies must be an immutable tuple")
        normalized = tuple(
            sorted(
                _identifier(item, "cluster task dependency")
                for item in self.dependencies
            )
        )
        if len(set(normalized)) != len(normalized):
            _fail("cluster task dependencies must be unique")
        object.__setattr__(self, "dependencies", normalized)
        if not isinstance(self.resources, ResourceBudget):
            _fail("cluster task resources are invalid")
        _event_key_text(self.event_key, "cluster task event_key")

    def to_mapping(self) -> dict[str, object]:
        return {
            "action": "execute_preregistered_row",
            "config_sha256": self.config_sha256,
            "dependencies": list(self.dependencies),
            "event_key": self.event_key,
            "resources": self.resources.to_mapping(),
            "row_id": self.row_id,
            "task_index": self.task_index,
        }


@dataclass(frozen=True, slots=True)
class ClusterBatchPlan:
    matrix_id: str
    matrix_sha256: str
    tasks: tuple[ClusterBatchTask, ...]
    submission_state: ClusterSubmissionState = ClusterSubmissionState.NOT_SUBMITTED
    schema_version: str = CLUSTER_PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _identifier(self.matrix_id, "cluster plan matrix_id")
        _sha256(self.matrix_sha256, "cluster plan matrix_sha256")
        if not isinstance(self.tasks, tuple) or not self.tasks:
            _fail("cluster plan tasks must be a non-empty immutable tuple")
        if not all(isinstance(task, ClusterBatchTask) for task in self.tasks):
            _fail("cluster plan contains an invalid task")
        if tuple(task.task_index for task in self.tasks) != tuple(
            range(len(self.tasks))
        ):
            _fail("cluster task indices must be contiguous and ordered")
        row_ids = tuple(task.row_id for task in self.tasks)
        if len(set(row_ids)) != len(row_ids):
            _fail("cluster task row IDs must be unique")
        config_hashes = tuple(task.config_sha256 for task in self.tasks)
        if len(set(config_hashes)) != len(config_hashes):
            _fail("cluster task config hashes must be unique")
        if self.submission_state is not ClusterSubmissionState.NOT_SUBMITTED:
            _fail("this module may generate but never submit cluster work")
        if self.schema_version != CLUSTER_PLAN_SCHEMA_VERSION:
            _fail("unsupported cluster plan schema version")

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def to_mapping(self) -> dict[str, object]:
        return {
            "matrix_id": self.matrix_id,
            "matrix_sha256": self.matrix_sha256,
            "schema_version": self.schema_version,
            "submission_state": self.submission_state.value,
            "tasks": [task.to_mapping() for task in self.tasks],
        }


def generate_cluster_batch_plan(
    matrix: ExperimentMatrix, destination: Path
) -> ClusterBatchPlan:
    """Write a scheduler-neutral plan; no process, network, or submit API is used."""
    if destination.exists():
        _fail("cluster batch plan destination already exists")
    tasks = tuple(
        ClusterBatchTask(
            task_index=index,
            row_id=row.row_id,
            config_sha256=row.config_sha256,
            dependencies=row.dependencies,
            resources=row.resources,
            event_key=EventKey(
                experiment_id=f"{matrix.matrix_id}.{row.row_id}",
                seed=row.seed,
                replicate_id=0,
                simulation_step=index,
                purpose="experiment_matrix.cluster_plan",
            ).value,
        )
        for index, row in enumerate(matrix.topological_rows)
    )
    plan = ClusterBatchPlan(
        matrix_id=matrix.matrix_id,
        matrix_sha256=matrix.sha256,
        tasks=tasks,
    )
    write_json_file(destination, plan.to_mapping())
    return plan


def _run_artifact_commitment(
    directory: Path, relative_path: str
) -> RunArtifactCommitment:
    normalized = _relative_artifact_path(relative_path, "run artifact path")
    path = directory / normalized
    if not path.is_file() or path.is_symlink():
        raise ExperimentResultIntegrityError(
            f"run artifact is missing or not a regular file: {normalized}"
        )
    return RunArtifactCommitment(
        path=normalized,
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
    )


def _expected_run_artifact_paths(
    entries: Sequence[RegistryEntry],
) -> tuple[str, ...]:
    paths = list(_RUN_FIXED_ARTIFACT_PATHS)
    for entry in entries:
        if entry.status is RegistryStatus.COMPLETED:
            paths.extend(
                (
                    f"runs/{entry.row_id}/result.json",
                    f"runs/{entry.row_id}/result.sha256",
                )
            )
            if entry.execution_receipt_sha256 is not None:
                paths.extend(
                    (
                        f"runs/{entry.row_id}/execution-receipt.json",
                        f"runs/{entry.row_id}/execution-receipt.sha256",
                    )
                )
    return tuple(sorted(paths))


def _captured_files_by_relative_path(
    root: _CapturedArtifactDirectory,
) -> dict[str, _CapturedArtifactFile]:
    assert root.files is not None
    assert root.children is not None
    result = {item.relative_path: item for item in root.files.values()}
    for child in root.children.values():
        for path, item in _captured_files_by_relative_path(child).items():
            if path in result:
                raise ExperimentResultIntegrityError(
                    f"captured matrix path is duplicated: {path}"
                )
            result[path] = item
    return result


def _captured_run_artifact_commitment(
    captured_files: Mapping[str, _CapturedArtifactFile], relative_path: str
) -> RunArtifactCommitment:
    normalized = _relative_artifact_path(relative_path, "run artifact path")
    try:
        item = captured_files[normalized]
    except KeyError as error:
        raise ExperimentResultIntegrityError(
            f"captured run artifact is missing: {normalized}"
        ) from error
    return RunArtifactCommitment(
        path=normalized,
        sha256=sha256_bytes(item.payload),
        size_bytes=len(item.payload),
    )


def _capture_matrix_result_directories(
    root: _CapturedArtifactDirectory,
    entries: Sequence[RegistryEntry],
) -> None:
    runs = _open_captured_child(root, "runs")
    completed = tuple(
        entry for entry in entries if entry.status is RegistryStatus.COMPLETED
    )
    expected_rows = frozenset(entry.row_id for entry in completed)
    _require_exact_directory_entries(runs, expected_rows)
    for entry in completed:
        row_directory = _open_captured_child(runs, entry.row_id)
        expected_files = {"result.json", "result.sha256"}
        if entry.execution_receipt_sha256 is not None:
            expected_files.update(
                {"execution-receipt.json", "execution-receipt.sha256"}
            )
        frozen_files = frozenset(expected_files)
        _require_exact_directory_entries(row_directory, frozen_files)
        for name in sorted(frozen_files):
            _capture_artifact_file(
                row_directory,
                name,
                relative_path=f"runs/{entry.row_id}/{name}",
            )


def write_matrix_run_manifest(
    directory: Path,
    matrix: ExperimentMatrix,
    entries: tuple[RegistryEntry, ...],
) -> MatrixRunManifest:
    """Seal a finished local matrix run; this never changes row outcomes."""
    if (directory / "run-manifest.json").exists() or (
        directory / "run-manifest.sha256"
    ).exists():
        raise ExperimentResultIntegrityError("matrix run manifest already exists")
    expected_rows = matrix.topological_rows
    if tuple(entry.row_id for entry in entries) != tuple(
        row.row_id for row in expected_rows
    ):
        raise ExperimentResultIntegrityError(
            "cannot seal a run whose final entries omit or reorder matrix rows"
        )
    for row, entry in zip(expected_rows, entries, strict=True):
        if entry.config_sha256 != row.config_sha256:
            raise ExperimentResultIntegrityError(
                f"cannot seal changed config hash for row {row.row_id!r}"
            )
        if (
            entry.draw_identity_sha256 != row.draw_identity_sha256
            or entry.shared_draw_identity is not row.has_shared_draw_identity
        ):
            raise ExperimentResultIntegrityError(
                f"cannot seal changed draw identity for row {row.row_id!r}"
            )
    artifacts = tuple(
        _run_artifact_commitment(directory, path)
        for path in _expected_run_artifact_paths(entries)
    )
    manifest = MatrixRunManifest(
        matrix_id=matrix.matrix_id,
        matrix_sha256=matrix.sha256,
        row_count=len(entries),
        completed_count=sum(
            entry.status is RegistryStatus.COMPLETED for entry in entries
        ),
        failed_count=sum(entry.status is RegistryStatus.FAILED for entry in entries),
        blocked_dependency_count=sum(
            entry.status is RegistryStatus.BLOCKED_DEPENDENCY for entry in entries
        ),
        failure_count=sum(
            entry.status in {RegistryStatus.FAILED, RegistryStatus.BLOCKED_DEPENDENCY}
            for entry in entries
        ),
        final_entries=entries,
        artifacts=artifacts,
    )
    write_json_file(directory / "run-manifest.json", manifest.to_mapping())
    write_text_file(directory / "run-manifest.sha256", f"{manifest.sha256}\n")
    return manifest


def _audit_event_identity(
    event: Mapping[str, object],
    matrix: ExperimentMatrix,
    row: ExperimentRow,
    purpose: str,
    sequences: list[int],
) -> None:
    if event.get("matrix_id") != matrix.matrix_id:
        raise ExperimentResultIntegrityError("matrix log event changed matrix_id")
    if event.get("row_id") != row.row_id:
        raise ExperimentResultIntegrityError("matrix log event changed row_id")
    schema_version = event.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != 1
    ):
        raise ExperimentResultIntegrityError("matrix log event schema changed")
    sequence = event.get("sequence")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        raise ExperimentResultIntegrityError("matrix log sequence is invalid")
    expected_key = _operation_key(matrix, row, sequence, purpose)
    if event.get("event_key") != expected_key:
        raise ExperimentResultIntegrityError(
            f"matrix log event key changed for {purpose}"
        )
    sequences.append(sequence)


def _audit_registry_log(
    events: Sequence[Mapping[str, object]],
    matrix: ExperimentMatrix,
    entries: tuple[RegistryEntry, ...],
    sequences: list[int],
) -> None:
    base_event_fields = frozenset(
        {
            "config_sha256",
            "draw_identity_sha256",
            "event_key",
            "failure_code",
            "matrix_id",
            "result_sha256",
            "row_id",
            "schema_version",
            "sequence",
            "shared_draw_identity",
            "status",
        }
    )
    states: dict[str, list[RegistryEntry]] = {
        row.row_id: [] for row in matrix.topological_rows
    }
    final_by_row = {entry.row_id: entry for entry in entries}
    purpose_by_status = {
        RegistryStatus.PENDING: "registry.pending",
        RegistryStatus.RUNNING: "registry.running",
        RegistryStatus.COMPLETED: "registry.completed",
        RegistryStatus.FAILED: "registry.failed",
        RegistryStatus.BLOCKED_DEPENDENCY: "registry.blocked",
    }
    for event in events:
        row_id = _identifier(event.get("row_id"), "registry event row_id")
        try:
            row = matrix.row(row_id)
        except ExperimentMatrixSchemaError as error:
            raise ExperimentResultIntegrityError(
                f"registry contains an unregistered row: {row_id!r}"
            ) from error
        raw_status = event.get("status")
        if not isinstance(raw_status, str):
            _fail("registry log status must be a string")
        try:
            status = RegistryStatus(raw_status)
        except ValueError as error:
            raise ExperimentMatrixSchemaError(
                "registry log status is invalid"
            ) from error
        terminal = final_by_row[row_id]
        expected_distribution_summary = bool(
            row.distribution_mcse_specs and status is RegistryStatus.COMPLETED
        )
        expected_execution_receipt = bool(
            status is RegistryStatus.COMPLETED
            and terminal.execution_receipt_sha256 is not None
        )
        optional_event_fields = {
            *({"distribution_mcse"} if expected_distribution_summary else set()),
            *({"execution_receipt_sha256"} if expected_execution_receipt else set()),
        }
        _strict_fields(
            event,
            frozenset(base_event_fields | optional_event_fields),
            f"registry {status.value} log event",
        )
        entry_fields = (
            "config_sha256",
            "draw_identity_sha256",
            "failure_code",
            "result_sha256",
            "row_id",
            "shared_draw_identity",
            "status",
        )
        entry_mapping = {key: event[key] for key in entry_fields}
        if expected_distribution_summary:
            entry_mapping["distribution_mcse"] = event["distribution_mcse"]
        if expected_execution_receipt:
            entry_mapping["execution_receipt_sha256"] = event[
                "execution_receipt_sha256"
            ]
        entry = RegistryEntry.from_mapping(entry_mapping)
        if entry.config_sha256 != row.config_sha256:
            raise ExperimentResultIntegrityError("registry config hash changed")
        if (
            entry.draw_identity_sha256 != row.draw_identity_sha256
            or entry.shared_draw_identity is not row.has_shared_draw_identity
        ):
            raise ExperimentResultIntegrityError("registry draw identity changed")
        _audit_event_identity(
            event, matrix, row, purpose_by_status[entry.status], sequences
        )
        states[row_id].append(entry)

    for row in matrix.topological_rows:
        final = final_by_row[row.row_id]
        expected_statuses = (
            (RegistryStatus.PENDING, RegistryStatus.BLOCKED_DEPENDENCY)
            if final.status is RegistryStatus.BLOCKED_DEPENDENCY
            else (RegistryStatus.PENDING, RegistryStatus.RUNNING, final.status)
        )
        observed = states[row.row_id]
        if tuple(item.status for item in observed) != expected_statuses:
            raise ExperimentResultIntegrityError(
                f"registry state history changed for row {row.row_id!r}"
            )
        if observed[-1] != final:
            raise ExperimentResultIntegrityError(
                f"registry terminal state disagrees with manifest for {row.row_id!r}"
            )


def _audit_dependency_log(
    events: Sequence[Mapping[str, object]],
    matrix: ExperimentMatrix,
    sequences: list[int],
) -> None:
    expected_fields = frozenset(
        {
            "dependencies",
            "event_key",
            "matrix_id",
            "row_id",
            "schema_version",
            "sequence",
        }
    )
    if len(events) != len(matrix.rows):
        raise ExperimentResultIntegrityError("dependency log omitted a matrix row")
    observed_rows: set[str] = set()
    for event in events:
        _strict_fields(event, expected_fields, "dependency log event")
        row_id = _identifier(event["row_id"], "dependency event row_id")
        if row_id in observed_rows:
            raise ExperimentResultIntegrityError("duplicate dependency declaration")
        observed_rows.add(row_id)
        try:
            row = matrix.row(row_id)
        except ExperimentMatrixSchemaError as error:
            raise ExperimentResultIntegrityError(
                "dependency log contains an unregistered row"
            ) from error
        dependencies = tuple(
            _identifier(item, "dependency log dependency")
            for item in _array(event["dependencies"], "dependencies")
        )
        if dependencies != row.dependencies:
            raise ExperimentResultIntegrityError(
                f"dependency declaration changed for row {row_id!r}"
            )
        _audit_event_identity(event, matrix, row, "dependencies.declared", sequences)
    if observed_rows != {row.row_id for row in matrix.rows}:
        raise ExperimentResultIntegrityError("dependency log row set changed")


def _audit_failure_log(
    events: Sequence[Mapping[str, object]],
    matrix: ExperimentMatrix,
    entries: tuple[RegistryEntry, ...],
    sequences: list[int],
) -> None:
    if not events:
        raise ExperimentResultIntegrityError("failure log is empty")
    final_by_row = {entry.row_id: entry for entry in entries}
    expected_failure_rows = {
        entry.row_id
        for entry in entries
        if entry.status in {RegistryStatus.FAILED, RegistryStatus.BLOCKED_DEPENDENCY}
    }
    observed_failure_rows: set[str] = set()
    seal_events = tuple(
        event
        for event in events
        if event.get("event_type") == "failure_registry_sealed"
    )
    if len(seal_events) != 1 or events[-1] is not seal_events[0]:
        raise ExperimentResultIntegrityError(
            "failure log must end in exactly one registry seal"
        )
    for event in events[:-1]:
        row_id = _identifier(event.get("row_id"), "failure event row_id")
        if row_id in observed_failure_rows:
            raise ExperimentResultIntegrityError("duplicate failure row record")
        observed_failure_rows.add(row_id)
        if row_id not in final_by_row:
            raise ExperimentResultIntegrityError(
                "failure log contains an unregistered row"
            )
        row = matrix.row(row_id)
        final = final_by_row[row_id]
        if final.status is RegistryStatus.FAILED:
            _strict_fields(
                event,
                frozenset(
                    {
                        "event_key",
                        "failure_code",
                        "failure_message",
                        "failure_type",
                        "matrix_id",
                        "row_id",
                        "schema_version",
                        "sequence",
                    }
                ),
                "execution failure event",
            )
            if event["failure_code"] != final.failure_code:
                raise ExperimentResultIntegrityError("failure code changed")
            if (
                not isinstance(event["failure_message"], str)
                or len(event["failure_message"]) > 512
                or not isinstance(event["failure_type"], str)
                or not event["failure_type"]
                or len(event["failure_type"]) > 128
            ):
                raise ExperimentResultIntegrityError(
                    "execution failure description is invalid"
                )
            purpose = "failure.execution"
        elif final.status is RegistryStatus.BLOCKED_DEPENDENCY:
            _strict_fields(
                event,
                frozenset(
                    {
                        "blocked_by",
                        "event_key",
                        "failure_code",
                        "failure_type",
                        "matrix_id",
                        "row_id",
                        "schema_version",
                        "sequence",
                    }
                ),
                "dependency failure event",
            )
            blocked_by = tuple(
                _identifier(item, "blocked dependency")
                for item in _array(event["blocked_by"], "blocked_by")
            )
            expected_blocked = tuple(
                dependency
                for dependency in row.dependencies
                if final_by_row[dependency].status is not RegistryStatus.COMPLETED
            )
            if blocked_by != expected_blocked or not blocked_by:
                raise ExperimentResultIntegrityError(
                    "blocked dependency evidence changed"
                )
            if (
                event["failure_code"] != final.failure_code
                or event["failure_type"] != "blocked_dependency"
            ):
                raise ExperimentResultIntegrityError(
                    "blocked dependency failure code changed"
                )
            purpose = "failure.dependency"
        else:
            raise ExperimentResultIntegrityError(
                "failure log records a row whose manifest status is successful"
            )
        _audit_event_identity(event, matrix, row, purpose, sequences)
    if observed_failure_rows != expected_failure_rows:
        raise ExperimentResultIntegrityError(
            "failure log omitted or added a terminal failure row"
        )
    seal = seal_events[0]
    _strict_fields(
        seal,
        frozenset(
            {
                "event_key",
                "event_type",
                "failure_count",
                "matrix_id",
                "schema_version",
                "sequence",
            }
        ),
        "failure registry seal",
    )
    expected_count = len(expected_failure_rows)
    observed_count = seal["failure_count"]
    if (
        isinstance(observed_count, bool)
        or not isinstance(observed_count, int)
        or observed_count != expected_count
    ):
        raise ExperimentResultIntegrityError("failure_count seal changed")
    seal_row = matrix.topological_rows[0]
    synthetic_seal_event = {**seal, "row_id": seal_row.row_id}
    _audit_event_identity(
        synthetic_seal_event,
        matrix,
        seal_row,
        "failure.registry_sealed",
        sequences,
    )


def _audit_resource_and_timing_logs(
    resource_events: Sequence[Mapping[str, object]],
    timing_events: Sequence[Mapping[str, object]],
    matrix: ExperimentMatrix,
    entries: tuple[RegistryEntry, ...],
    sequences: list[int],
) -> None:
    identity_fields = frozenset(
        {
            "event",
            "event_key",
            "matrix_id",
            "row_id",
            "schema_version",
            "sequence",
        }
    )
    measurement_fields = frozenset(
        {
            "measurement_scope",
            "process_cpu_seconds",
            "process_peak_rss_bytes",
            "process_peak_rss_delta_bytes",
            "resource_measurement_reason",
            "system_cpu_seconds",
            "user_cpu_seconds",
            "wall_elapsed_seconds",
        }
    )

    def validate_optional_real(value: object, field: str) -> float | None:
        if value is None:
            return None
        result = _finite(value, field)
        if result < 0.0:
            _fail(f"{field} must be non-negative")
        return result

    def validate_optional_integer(value: object, field: str) -> int | None:
        if value is None:
            return None
        return _non_negative_int(value, field)

    def validate_measurement(
        event: Mapping[str, object], *, executed: bool, context: str
    ) -> tuple[object, ...]:
        if event["measurement_scope"] != "operational_non_scientific_process":
            _fail(f"{context} measurement_scope changed")
        reason = event["resource_measurement_reason"]
        if reason is not None and (
            not isinstance(reason, str) or not reason or len(reason) > 256
        ):
            _fail(f"{context} resource_measurement_reason is invalid")
        process_cpu = validate_optional_real(
            event["process_cpu_seconds"], f"{context} process_cpu_seconds"
        )
        wall_elapsed = validate_optional_real(
            event["wall_elapsed_seconds"], f"{context} wall_elapsed_seconds"
        )
        user_cpu = validate_optional_real(
            event["user_cpu_seconds"], f"{context} user_cpu_seconds"
        )
        system_cpu = validate_optional_real(
            event["system_cpu_seconds"], f"{context} system_cpu_seconds"
        )
        peak_rss = validate_optional_integer(
            event["process_peak_rss_bytes"],
            f"{context} process_peak_rss_bytes",
        )
        peak_delta = validate_optional_integer(
            event["process_peak_rss_delta_bytes"],
            f"{context} process_peak_rss_delta_bytes",
        )
        if executed:
            if process_cpu is None or wall_elapsed is None:
                _fail(f"{context} executed measurement lacks wall or process CPU")
            platform_values = (peak_rss, user_cpu, system_cpu)
            if any(value is None for value in platform_values) and reason is None:
                _fail(f"{context} unavailable resource values require a reason")
            if (
                all(value is not None for value in platform_values)
                and reason is not None
            ):
                _fail(f"{context} available resource values must not claim failure")
        else:
            if (
                any(
                    value is not None
                    for value in (
                        process_cpu,
                        wall_elapsed,
                        user_cpu,
                        system_cpu,
                        peak_rss,
                        peak_delta,
                    )
                )
                or reason != "blocked_dependency"
            ):
                _fail(f"{context} blocked measurement must be explicitly unavailable")
        return tuple(event[field] for field in sorted(measurement_fields))

    final_by_row = {entry.row_id: entry for entry in entries}
    resource_labels: dict[str, list[str]] = {row.row_id: [] for row in matrix.rows}
    timing_labels: dict[str, list[str]] = {row.row_id: [] for row in matrix.rows}
    resource_measurements: dict[str, tuple[object, ...]] = {}
    timing_measurements: dict[str, tuple[object, ...]] = {}
    resource_purpose = {
        "requested": "resource.requested",
        "observed_completed": "resource.completed",
        "observed_failed": "resource.failed",
        "not_executed": "resource.blocked",
    }
    timing_purpose = {
        "started": "timing.started",
        "completed": "timing.completed",
        "failed": "timing.failed",
        "blocked": "timing.blocked",
    }
    for event in resource_events:
        row_id = _identifier(event.get("row_id"), "resource event row_id")
        try:
            row = matrix.row(row_id)
        except ExperimentMatrixSchemaError as error:
            raise ExperimentResultIntegrityError(
                "resource log contains an unregistered row"
            ) from error
        label = event.get("event")
        if not isinstance(label, str) or label not in resource_purpose:
            raise ExperimentResultIntegrityError("resource event type changed")
        if label == "requested":
            _strict_fields(
                event,
                identity_fields | {"config_sha256", "resources"},
                "resource requested event",
            )
            resources = _mapping(event["resources"], "resource requested resources")
            _strict_fields(
                resources,
                frozenset({"cpu_cores", "memory_mb", "wall_time_seconds"}),
                "resource requested resources",
            )
            observed_budget = ResourceBudget(
                cpu_cores=_positive_int(
                    resources["cpu_cores"], "resource requested cpu_cores"
                ),
                memory_mb=_positive_int(
                    resources["memory_mb"], "resource requested memory_mb"
                ),
                wall_time_seconds=_positive_int(
                    resources["wall_time_seconds"],
                    "resource requested wall_time_seconds",
                ),
            )
            if (
                _sha256(event["config_sha256"], "resource requested config_sha256")
                != row.config_sha256
                or observed_budget != row.resources
            ):
                raise ExperimentResultIntegrityError(
                    "resource request changed from the frozen row"
                )
        else:
            _strict_fields(
                event,
                identity_fields | measurement_fields,
                f"resource {label} event",
            )
            resource_measurements[row_id] = validate_measurement(
                event,
                executed=label != "not_executed",
                context=f"resource {label}",
            )
        resource_labels[row_id].append(label)
        _audit_event_identity(event, matrix, row, resource_purpose[label], sequences)
    for event in timing_events:
        row_id = _identifier(event.get("row_id"), "timing event row_id")
        try:
            row = matrix.row(row_id)
        except ExperimentMatrixSchemaError as error:
            raise ExperimentResultIntegrityError(
                "timing log contains an unregistered row"
            ) from error
        label = event.get("event")
        if not isinstance(label, str) or label not in timing_purpose:
            raise ExperimentResultIntegrityError("timing event type changed")
        if label == "started":
            _strict_fields(
                event,
                identity_fields | {"time_basis"},
                "timing started event",
            )
            if event["time_basis"] != "deterministic_logical_sequence":
                raise ExperimentResultIntegrityError("timing started basis changed")
        else:
            extra_fields = {"time_basis"}
            if label == "completed":
                extra_fields.add("result_sha256")
            _strict_fields(
                event,
                identity_fields | measurement_fields | extra_fields,
                f"timing {label} event",
            )
            expected_basis = (
                "operational_measurement_unavailable"
                if label == "blocked"
                else "measured_operational_elapsed"
            )
            if event["time_basis"] != expected_basis:
                raise ExperimentResultIntegrityError(f"timing {label} basis changed")
            timing_measurements[row_id] = validate_measurement(
                event,
                executed=label != "blocked",
                context=f"timing {label}",
            )
        timing_labels[row_id].append(label)
        final = final_by_row[row_id]
        if label == "completed" and (
            _sha256(event["result_sha256"], "timing result_sha256")
            != final.result_sha256
        ):
            raise ExperimentResultIntegrityError("timing log result commitment changed")
        _audit_event_identity(event, matrix, row, timing_purpose[label], sequences)
    for row in matrix.topological_rows:
        status = final_by_row[row.row_id].status
        expected_resources = (
            ["not_executed"]
            if status is RegistryStatus.BLOCKED_DEPENDENCY
            else [
                "requested",
                (
                    "observed_completed"
                    if status is RegistryStatus.COMPLETED
                    else "observed_failed"
                ),
            ]
        )
        expected_timings = (
            ["blocked"]
            if status is RegistryStatus.BLOCKED_DEPENDENCY
            else [
                "started",
                "completed" if status is RegistryStatus.COMPLETED else "failed",
            ]
        )
        if resource_labels[row.row_id] != expected_resources:
            raise ExperimentResultIntegrityError(
                f"resource log history changed for row {row.row_id!r}"
            )
        if timing_labels[row.row_id] != expected_timings:
            raise ExperimentResultIntegrityError(
                f"timing log history changed for row {row.row_id!r}"
            )
        if resource_measurements.get(row.row_id) != timing_measurements.get(row.row_id):
            raise ExperimentResultIntegrityError(
                f"resource and timing measurements disagree for row {row.row_id!r}"
            )


def _audit_matrix_run_path_view(
    directory: Path,
    matrix: ExperimentMatrix,
    *,
    expected_manifest_sha256: str | None = None,
    execution_authority: SyntheticExecutionAuthority | None = None,
    captured_log_payloads: Mapping[str, bytes] | None = None,
) -> MatrixRunAuditReport:
    """Audit one path view while the public wrapper pins every input inode."""
    if any(not row.frozen_inputs.synthetic_only for row in matrix.rows):
        raise ExperimentResultIntegrityError(
            "non-synthetic matrix audit is blocked until an external execution "
            "and temporal authority is implemented"
        )
    if not directory.is_dir() or directory.is_symlink():
        raise ExperimentResultIntegrityError("matrix run must be a regular directory")
    if expected_manifest_sha256 is not None:
        _sha256(expected_manifest_sha256, "expected_manifest_sha256")
    expected_top_level = frozenset(
        {
            *_RUN_FIXED_ARTIFACT_PATHS,
            "run-manifest.json",
            "run-manifest.sha256",
            "runs",
        }
    )
    observed_top_level = frozenset(child.name for child in directory.iterdir())
    if observed_top_level != expected_top_level:
        raise ExperimentResultIntegrityError(
            "matrix run top-level file set changed: "
            f"missing={sorted(expected_top_level - observed_top_level)!r}, "
            f"extra={sorted(observed_top_level - expected_top_level)!r}"
        )
    if any((directory / name).is_symlink() for name in expected_top_level):
        raise ExperimentResultIntegrityError("matrix run must not contain symlinks")
    try:
        manifest = MatrixRunManifest.from_mapping(
            read_json_object(directory / "run-manifest.json")
        )
        manifest_seal = (directory / "run-manifest.sha256").read_text(encoding="ascii")
    except (
        ExperimentMatrixSchemaError,
        MetadataValidationError,
        OSError,
        UnicodeError,
    ) as error:
        raise ExperimentResultIntegrityError(str(error)) from error
    if manifest_seal != f"{manifest.sha256}\n":
        raise ExperimentResultIntegrityError("matrix run manifest seal mismatch")
    if (
        expected_manifest_sha256 is not None
        and manifest.sha256 != expected_manifest_sha256
    ):
        raise ExperimentResultIntegrityError(
            "matrix run changed after its external manifest commitment"
        )
    if (
        manifest.matrix_id != matrix.matrix_id
        or manifest.matrix_sha256 != matrix.sha256
    ):
        raise ExperimentResultIntegrityError(
            "run manifest does not match the authoritative matrix"
        )
    expected_rows = matrix.topological_rows
    entries = manifest.final_entries
    if tuple(entry.row_id for entry in entries) != tuple(
        row.row_id for row in expected_rows
    ):
        raise ExperimentResultIntegrityError(
            "run manifest omitted, added, or reordered a matrix specification"
        )
    for row, entry in zip(expected_rows, entries, strict=True):
        if entry.config_sha256 != row.config_sha256:
            raise ExperimentResultIntegrityError(
                f"run manifest changed config hash for row {row.row_id!r}"
            )
        if (
            entry.draw_identity_sha256 != row.draw_identity_sha256
            or entry.shared_draw_identity is not row.has_shared_draw_identity
        ):
            raise ExperimentResultIntegrityError(
                f"run manifest changed draw identity for row {row.row_id!r}"
            )
    expected_artifact_paths = _expected_run_artifact_paths(entries)
    if tuple(item.path for item in manifest.artifacts) != expected_artifact_paths:
        raise ExperimentResultIntegrityError(
            "run manifest artifact set does not match all logs and final results"
        )
    for commitment in manifest.artifacts:
        observed = _run_artifact_commitment(directory, commitment.path)
        if observed != commitment:
            raise ExperimentResultIntegrityError(
                f"run artifact changed after sealing: {commitment.path}"
            )
    try:
        if read_json_object(directory / "matrix.json") != matrix.to_mapping():
            raise ExperimentResultIntegrityError(
                "matrix.json does not match the authoritative matrix"
            )
        matrix_seal = (directory / "matrix.sha256").read_text(encoding="ascii")
    except (MetadataValidationError, OSError, UnicodeError) as error:
        raise ExperimentResultIntegrityError(str(error)) from error
    if matrix_seal != f"{matrix.sha256}\n":
        raise ExperimentResultIntegrityError("matrix semantic hash seal mismatch")
    expected_plan = ClusterBatchPlan(
        matrix_id=matrix.matrix_id,
        matrix_sha256=matrix.sha256,
        tasks=tuple(
            ClusterBatchTask(
                task_index=index,
                row_id=row.row_id,
                config_sha256=row.config_sha256,
                dependencies=row.dependencies,
                resources=row.resources,
                event_key=EventKey(
                    experiment_id=f"{matrix.matrix_id}.{row.row_id}",
                    seed=row.seed,
                    replicate_id=0,
                    simulation_step=index,
                    purpose="experiment_matrix.cluster_plan",
                ).value,
            )
            for index, row in enumerate(expected_rows)
        ),
    )
    if read_json_object(directory / "cluster-plan.json") != expected_plan.to_mapping():
        raise ExperimentResultIntegrityError(
            "cluster plan changed or claims work was submitted"
        )
    runs_directory = directory / "runs"
    if not runs_directory.is_dir() or runs_directory.is_symlink():
        raise ExperimentResultIntegrityError("matrix runs path is invalid")
    expected_result_rows = {
        entry.row_id for entry in entries if entry.status is RegistryStatus.COMPLETED
    }
    observed_result_rows = {child.name for child in runs_directory.iterdir()}
    if observed_result_rows != expected_result_rows:
        raise ExperimentResultIntegrityError(
            "result directory set does not equal completed registry rows"
        )
    for entry in entries:
        if entry.status is RegistryStatus.COMPLETED:
            if entry.result_sha256 is None:
                raise ExperimentResultIntegrityError(
                    "completed run row has no result commitment"
                )
            result_audit = audit_result_bundle(
                runs_directory / entry.row_id,
                matrix.row(entry.row_id),
                expected_result_sha256=entry.result_sha256,
                expected_execution_receipt_sha256=(entry.execution_receipt_sha256),
                execution_authority=execution_authority,
                matrix=matrix,
            )
            if result_audit.distribution_mcse != entry.distribution_mcse:
                raise ExperimentResultIntegrityError(
                    "registry distribution MCSE decisions disagree with result"
                )
            if (
                result_audit.draw_identity_sha256 != entry.draw_identity_sha256
                or result_audit.shared_draw_identity is not entry.shared_draw_identity
            ):
                raise ExperimentResultIntegrityError(
                    "registry draw identity disagrees with result bundle"
                )
            if result_audit.execution_receipt_sha256 != entry.execution_receipt_sha256:
                raise ExperimentResultIntegrityError(
                    "registry execution receipt disagrees with result bundle"
                )
    try:
        if captured_log_payloads is None:
            log_events = {
                name: read_jsonl_objects(directory / name) for name in _RUN_LOG_PATHS
            }
        else:
            if frozenset(captured_log_payloads) != frozenset(_RUN_LOG_PATHS):
                raise ExperimentResultIntegrityError(
                    "captured matrix log byte set is incomplete or has unknown logs"
                )
            log_events = {
                name: read_jsonl_bytes_objects(
                    captured_log_payloads[name],
                    source=str(directory / name),
                )
                for name in _RUN_LOG_PATHS
            }
        registry_events = log_events["registry.jsonl"]
        dependency_events = log_events["dependencies.jsonl"]
        resource_events = log_events["resources.jsonl"]
        timing_events = log_events["timing.jsonl"]
        failure_events = log_events["failures.jsonl"]
    except MetadataValidationError as error:
        raise ExperimentResultIntegrityError(str(error)) from error
    sequences: list[int] = []
    try:
        _audit_registry_log(registry_events, matrix, entries, sequences)
        _audit_dependency_log(dependency_events, matrix, sequences)
        _audit_resource_and_timing_logs(
            resource_events, timing_events, matrix, entries, sequences
        )
        _audit_failure_log(failure_events, matrix, entries, sequences)
    except ExperimentMatrixSchemaError as error:
        raise ExperimentResultIntegrityError(str(error)) from error
    if sorted(sequences) != list(range(len(sequences))):
        raise ExperimentResultIntegrityError(
            "matrix log sequence is incomplete, duplicated, or reordered"
        )
    return MatrixRunAuditReport(
        valid=True,
        matrix_id=matrix.matrix_id,
        matrix_sha256=matrix.sha256,
        manifest_sha256=manifest.sha256,
        row_count=manifest.row_count,
        completed_count=manifest.completed_count,
        failure_count=manifest.failure_count,
        execution_receipts_verified=(
            manifest.completed_count > 0
            and all(
                entry.execution_receipt_sha256 is not None
                for entry in entries
                if entry.status is RegistryStatus.COMPLETED
            )
            and execution_authority is not None
        ),
    )


def audit_matrix_run(
    directory: Path,
    matrix: ExperimentMatrix,
    *,
    expected_manifest_sha256: str | None = None,
    execution_authority: SyntheticExecutionAuthority | None = None,
) -> MatrixRunAuditReport:
    """Audit a matrix run while pinning every committed path and exact byte set."""
    if any(not row.frozen_inputs.synthetic_only for row in matrix.rows):
        raise ExperimentResultIntegrityError(
            "non-synthetic matrix audit is blocked until an external execution "
            "and temporal authority is implemented"
        )
    expected_top_level = frozenset(
        {
            *_RUN_FIXED_ARTIFACT_PATHS,
            "run-manifest.json",
            "run-manifest.sha256",
            "runs",
        }
    )
    captured = _open_captured_root(directory)
    try:
        _require_exact_directory_entries(captured, expected_top_level)
        for name in sorted(expected_top_level - {"runs"}):
            _capture_artifact_file(captured, name, relative_path=name)
        assert captured.files is not None
        try:
            captured_manifest = MatrixRunManifest.from_mapping(
                read_json_bytes_object(
                    captured.files["run-manifest.json"].payload,
                    source=str(directory / "run-manifest.json"),
                )
            )
            captured_manifest_seal = captured.files[
                "run-manifest.sha256"
            ].payload.decode("ascii")
        except (
            ExperimentMatrixSchemaError,
            MetadataValidationError,
            UnicodeError,
        ) as error:
            raise ExperimentResultIntegrityError(str(error)) from error
        if captured_manifest_seal != f"{captured_manifest.sha256}\n":
            raise ExperimentResultIntegrityError("matrix run manifest seal mismatch")
        if (
            captured_manifest.matrix_id != matrix.matrix_id
            or captured_manifest.matrix_sha256 != matrix.sha256
        ):
            raise ExperimentResultIntegrityError(
                "run manifest does not match the authoritative matrix"
            )
        if expected_manifest_sha256 is not None:
            _sha256(expected_manifest_sha256, "expected_manifest_sha256")
            if captured_manifest.sha256 != expected_manifest_sha256:
                raise ExperimentResultIntegrityError(
                    "matrix run changed after its external manifest commitment"
                )
        _capture_matrix_result_directories(
            captured,
            captured_manifest.final_entries,
        )
        _require_captured_tree_current(captured, external_root=True)
        captured_files = _captured_files_by_relative_path(captured)
        expected_artifact_paths = _expected_run_artifact_paths(
            captured_manifest.final_entries
        )
        if tuple(item.path for item in captured_manifest.artifacts) != (
            expected_artifact_paths
        ):
            raise ExperimentResultIntegrityError(
                "run manifest artifact set does not match all logs and final results"
            )
        for commitment in captured_manifest.artifacts:
            if (
                _captured_run_artifact_commitment(
                    captured_files,
                    commitment.path,
                )
                != commitment
            ):
                raise ExperimentResultIntegrityError(
                    f"run artifact changed before stable capture: {commitment.path}"
                )
        report = _audit_matrix_run_path_view(
            directory,
            matrix,
            expected_manifest_sha256=expected_manifest_sha256,
            execution_authority=execution_authority,
            captured_log_payloads={
                name: captured.files[name].payload for name in _RUN_LOG_PATHS
            },
        )
        if (
            report.manifest_sha256 != captured_manifest.sha256
            or report.row_count != captured_manifest.row_count
            or report.completed_count != captured_manifest.completed_count
            or report.failure_count != captured_manifest.failure_count
        ):
            raise ExperimentResultIntegrityError(
                "path audit result disagrees with the captured run manifest"
            )
        _require_captured_tree_current(captured, external_root=True)
        final_files = _captured_files_by_relative_path(captured)
        if frozenset(final_files) != frozenset(captured_files):
            raise ExperimentResultIntegrityError(
                "captured matrix artifact set changed before audit completion"
            )
        for commitment in captured_manifest.artifacts:
            if (
                _captured_run_artifact_commitment(final_files, commitment.path)
                != commitment
            ):
                raise ExperimentResultIntegrityError(
                    f"run artifact changed before audit completion: {commitment.path}"
                )
        _require_captured_tree_current(captured, external_root=True)
        return report
    except OSError as error:
        raise ExperimentResultIntegrityError(
            f"cannot audit matrix run: {error}"
        ) from error
    finally:
        captured.close()


def load_experiment_matrix(
    matrix_path: Path, *, expected_matrix_sha256: str
) -> ExperimentMatrix:
    """Load a strict matrix using an external semantic-hash trust root."""
    expected = _sha256(expected_matrix_sha256, "expected_matrix_sha256")
    try:
        mapping = read_json_object(matrix_path)
        matrix = ExperimentMatrix.from_mapping(mapping)
        seal = matrix_path.with_name("matrix.sha256").read_text(encoding="ascii")
    except (MetadataValidationError, OSError, UnicodeError) as error:
        raise ExperimentResultIntegrityError(str(error)) from error
    if matrix.sha256 != expected:
        raise ExperimentResultIntegrityError(
            "matrix differs from its external semantic-hash commitment"
        )
    if seal != f"{matrix.sha256}\n":
        raise ExperimentResultIntegrityError("matrix semantic hash seal mismatch")
    return matrix


def audit_matrix_run_from_directory(
    directory: Path,
    *,
    expected_matrix_sha256: str,
    expected_manifest_sha256: str,
    execution_authority: SyntheticExecutionAuthority | None = None,
) -> MatrixRunAuditReport:
    """Audit a portable run without requiring an in-memory caller-built matrix."""
    matrix = load_experiment_matrix(
        directory / "matrix.json",
        expected_matrix_sha256=expected_matrix_sha256,
    )
    return audit_matrix_run(
        directory,
        matrix,
        expected_manifest_sha256=expected_manifest_sha256,
        execution_authority=execution_authority,
    )


def build_preregistered_smoke_matrix(
    frozen_inputs: FrozenInputReference,
    *,
    seed: str = "experiment-matrix-smoke-v1",
    stop_rule: MCSEStopRule | None = None,
    resources: ResourceBudget | None = None,
) -> ExperimentMatrix:
    """Return 12 explicit one-factor contrasts, never the 1,152-row product."""
    if not frozen_inputs.synthetic_only:
        _fail("the built-in smoke matrix requires explicitly synthetic frozen inputs")
    frozen_stop = stop_rule or MCSEStopRule(
        metric_name="scheduler_smoke_scalar",
        minimum_replicates=4,
        maximum_replicates=16,
        check_every=1,
        target_mcse=0.05,
    )
    frozen_resources = resources or ResourceBudget(
        cpu_cores=1,
        memory_mb=256,
        wall_time_seconds=60,
    )
    baseline = ExperimentFactors(
        model_variant=ModelVariant.FULL_MODEL,
        behavior_representation=BehaviorRepresentation.CALIBRATED_CONTINUOUS_BEHAVIOR,
        social=SocialMode.ENABLED,
        memory=MemoryMode.ENABLED,
        llm=LLMMode.ENABLED,
        calibration=CalibrationMode.POSTERIOR_CALIBRATION,
        rng=RNGMode.EVENT_KEYED_RNG,
        model_uncertainty=ModelUncertaintyMode.STRUCTURAL_ENSEMBLE,
        abstention=AbstentionMode.WITH_ABSTENTION,
    )
    contrasts: tuple[tuple[str, ExperimentFactors], ...] = (
        ("baseline_full", baseline),
        (
            "homogeneous_agents",
            replace(baseline, model_variant=ModelVariant.HOMOGENEOUS_AGENTS),
        ),
        ("rational_only", replace(baseline, model_variant=ModelVariant.RATIONAL_ONLY)),
        (
            "prompted_behavior",
            replace(
                baseline,
                behavior_representation=(
                    BehaviorRepresentation.PROMPTED_BEHAVIOR_LABELS
                ),
            ),
        ),
        ("no_social", replace(baseline, social=SocialMode.NO_SOCIAL_GRAPH)),
        ("no_memory", replace(baseline, memory=MemoryMode.NO_MEMORY)),
        ("no_llm", replace(baseline, llm=LLMMode.NO_LLM)),
        (
            "point_calibration",
            replace(baseline, calibration=CalibrationMode.POINT_CALIBRATION),
        ),
        ("independent_rng", replace(baseline, rng=RNGMode.INDEPENDENT_RNG)),
        (
            "same_sequential_seed",
            replace(baseline, rng=RNGMode.SAME_SEQUENTIAL_SEED),
        ),
        (
            "single_model",
            replace(
                baseline,
                model_uncertainty=ModelUncertaintyMode.SINGLE_MODEL,
            ),
        ),
        (
            "without_abstention",
            replace(baseline, abstention=AbstentionMode.WITHOUT_ABSTENTION),
        ),
    )
    rows = tuple(
        ExperimentRow(
            row_id=row_id,
            factors=factors,
            frozen_inputs=frozen_inputs,
            stop_rule=frozen_stop,
            resources=frozen_resources,
            seed=seed,
            dependencies=() if row_id == "baseline_full" else ("baseline_full",),
        )
        for row_id, factors in contrasts
    )
    return ExperimentMatrix(
        matrix_id="synthetic_preregistered_smoke",
        rows=rows,
        registration_kind=MatrixRegistrationKind.SMOKE,
    )


__all__ = [
    "AbstentionMode",
    "BehaviorRepresentation",
    "CalibrationMode",
    "ClusterBatchPlan",
    "ClusterBatchTask",
    "ClusterSubmissionState",
    "DIMENSION_LEVELS",
    "DISTRIBUTION_RESULT_SCHEMA_VERSION",
    "DistributionMCSERegistryRecord",
    "DRAW_IDENTITY_SCHEMA_VERSION",
    "ExperimentFactors",
    "ExperimentMatrix",
    "ExperimentMatrixError",
    "ExperimentMatrixSchemaError",
    "ExperimentResult",
    "ExperimentResultIntegrityError",
    "ExperimentRow",
    "FrozenInputReference",
    "LLMMode",
    "MCSEStopRule",
    "MatrixDimension",
    "MatrixRegistrationKind",
    "MatrixRunAuditReport",
    "MatrixRunManifest",
    "MatrixRunReport",
    "MemoryMode",
    "ModelUncertaintyMode",
    "ModelVariant",
    "RNGMode",
    "RegistryEntry",
    "RegistryStatus",
    "ReplicateRunner",
    "ResourceBudget",
    "RunArtifactCommitment",
    "SharedDrawIdentity",
    "SharedDrawIdentityAuditReport",
    "ResultAuditReport",
    "ResultStopReason",
    "SocialMode",
    "SyntheticMatrixRunError",
    "SyntheticMatrixRunner",
    "WelfordState",
    "audit_matrix_run",
    "audit_matrix_run_from_directory",
    "audit_result_bundle",
    "audit_shared_draw_identity",
    "build_preregistered_smoke_matrix",
    "generate_cluster_batch_plan",
    "matrix_replicate_event_key",
    "load_experiment_matrix",
    "run_local_matrix",
    "write_matrix_run_manifest",
    "write_result_bundle",
]
