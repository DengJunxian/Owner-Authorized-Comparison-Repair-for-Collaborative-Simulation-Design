"""Fail-closed temporal data firewall and immutable snapshot support.

This module deliberately validates metadata before reading payload bytes.  It
never repairs, truncates, or filters records that violate the configured
cutoff.  Every operational decision is written to a chained JSONL audit log.
"""

from __future__ import annotations

import fcntl
import gc
import json
import math
import os
import re
import secrets
import shutil
import socket
import stat
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import AbstractContextManager, ExitStack, suppress
from contextvars import ContextVar, Token
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path, PurePosixPath
from threading import RLock
from types import TracebackType
from typing import NoReturn, Self, TypeVar, cast
from unittest.mock import patch

from policytwin.provenance import (
    MetadataValidationError,
    canonical_json_bytes,
    read_jsonl_objects,
    sha256_bytes,
)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ZERO_SHA256 = "0" * 64
_MANIFEST_FILE = "manifest.json"
_MANIFEST_SEAL_FILE = "manifest.sha256"
_FROZEN_MARKER_FILE = ".policytwin-frozen"
_SNAPSHOT_PUBLICATION_SUFFIX = ".policytwin-published"
_RESERVED_SNAPSHOT_PATHS = frozenset(
    {_MANIFEST_FILE, _MANIFEST_SEAL_FILE, _FROZEN_MARKER_FILE}
)
POST_CUTOFF_CANARY_SOURCE = "policytwin.post-cutoff-canary"
_ORIGINAL_SOCKET_TYPE = socket.socket
_AUDIT_DECISION_LOCKS_GUARD = RLock()
_AUDIT_DECISION_LOCKS: dict[Path, RLock] = {}


def _audit_decision_lock(path: Path) -> RLock:
    with _AUDIT_DECISION_LOCKS_GUARD:
        return _AUDIT_DECISION_LOCKS.setdefault(path, RLock())


def _snapshot_publication_path(root: Path, token: str) -> Path:
    """Return the sibling path whose hard link authorizes snapshot loading."""
    return root.with_name(f".{root.name}{_SNAPSHOT_PUBLICATION_SUFFIX}-{token}")


class TemporalFirewallError(RuntimeError):
    """Base class for temporal-firewall failures."""


class TemporalSchemaError(TemporalFirewallError):
    """Raised when immutable temporal metadata are malformed or ambiguous."""


class TemporalLeakageError(TemporalFirewallError):
    """Raised when information would cross a declared temporal boundary."""


class PostCutoffDataError(TemporalLeakageError):
    """Raised when data available after the knowledge cutoff are encountered."""


class TransformationLeakageError(TemporalLeakageError):
    """Raised when a transformation uses future or forbidden inputs."""


class PartitionAccessError(TemporalLeakageError):
    """Raised when a job attempts to inspect a sealed partition."""


class ArtifactIntegrityError(TemporalFirewallError):
    """Raised when a payload or manifest fails integrity validation."""


class FrozenSnapshotError(TemporalFirewallError):
    """Raised when a frozen snapshot is mutable or modification is attempted."""


class NetworkAccessBlockedError(TemporalFirewallError):
    """Raised when network access is attempted during frozen evaluation."""


class AuditLogError(TemporalFirewallError):
    """Raised when the mandatory audit log cannot be trusted or written."""


class ArtifactKind(StrEnum):
    """Supported temporal artifact classes."""

    TABLE = "table"
    TEXT = "text"
    FEATURE = "feature"
    MODEL_CHECKPOINT = "model_checkpoint"


class DataPartition(StrEnum):
    """Mutually exclusive temporal research partitions."""

    TRAIN = "train"
    VALIDATION = "validation"
    POST_POLICY_EVALUATION = "post_policy_evaluation"


class AccessPurpose(StrEnum):
    """Declared reason for loading an artifact."""

    TRAINING = "training"
    VALIDATION = "validation"
    FROZEN_EVALUATION = "frozen_evaluation"


class LineageRole(StrEnum):
    """Role played by an upstream record in derived data."""

    RAW_INPUT = "raw_input"
    FEATURE_INPUT = "feature_input"
    SCALER_FIT = "scaler_fit"
    LABEL = "label"


class RuleCode(StrEnum):
    """Stable machine-readable temporal-firewall audit rules."""

    OK = "OK"
    INVALID_SCHEMA = "INVALID_SCHEMA"
    POST_CUTOFF_AVAILABLE_TIME = "POST_CUTOFF_AVAILABLE_TIME"
    EVENT_AFTER_CUTOFF = "EVENT_AFTER_CUTOFF"
    PARTITION_VIOLATION = "PARTITION_VIOLATION"
    PARTITION_ACCESS_DENIED = "PARTITION_ACCESS_DENIED"
    EVALUATION_CONTEXT_REQUIRED = "EVALUATION_CONTEXT_REQUIRED"
    SNAPSHOT_REQUIRED = "SNAPSHOT_REQUIRED"
    UNTRUSTED_LICENSE = "UNTRUSTED_LICENSE"
    PATH_VIOLATION = "PATH_VIOLATION"
    HASH_MISMATCH = "HASH_MISMATCH"
    FEATURE_LINEAGE_INVALID = "FEATURE_LINEAGE_INVALID"
    CHECKPOINT_LINEAGE_INVALID = "CHECKPOINT_LINEAGE_INVALID"
    FULL_SAMPLE_FIT = "FULL_SAMPLE_FIT"
    FORWARD_FILL_LEAKAGE = "FORWARD_FILL_LEAKAGE"
    CROSS_CUTOFF_WINDOW = "CROSS_CUTOFF_WINDOW"
    LABEL_LEAKAGE = "LABEL_LEAKAGE"
    MANIFEST_MISMATCH = "MANIFEST_MISMATCH"
    FROZEN_MUTATION = "FROZEN_MUTATION"
    NETWORK_DENIED = "NETWORK_DENIED"
    PRIOR_REJECTION = "PRIOR_REJECTION"


_LATCHING_RULES = frozenset(
    {
        RuleCode.INVALID_SCHEMA,
        RuleCode.POST_CUTOFF_AVAILABLE_TIME,
        RuleCode.EVENT_AFTER_CUTOFF,
        RuleCode.PARTITION_VIOLATION,
        RuleCode.UNTRUSTED_LICENSE,
        RuleCode.PATH_VIOLATION,
        RuleCode.HASH_MISMATCH,
        RuleCode.FEATURE_LINEAGE_INVALID,
        RuleCode.CHECKPOINT_LINEAGE_INVALID,
        RuleCode.FULL_SAMPLE_FIT,
        RuleCode.FORWARD_FILL_LEAKAGE,
        RuleCode.CROSS_CUTOFF_WINDOW,
        RuleCode.LABEL_LEAKAGE,
        RuleCode.MANIFEST_MISMATCH,
        RuleCode.FROZEN_MUTATION,
        RuleCode.NETWORK_DENIED,
        RuleCode.PRIOR_REJECTION,
    }
)


def _schema_fail(message: str) -> NoReturn:
    raise TemporalSchemaError(message)


def _normalize_datetime(value: object, field: str) -> datetime:
    if not isinstance(value, datetime):
        _schema_fail(f"{field} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        _schema_fail(f"{field} must include an explicit timezone")
    try:
        return value.astimezone(UTC)
    except (OverflowError, ValueError) as error:
        raise TemporalSchemaError(f"{field} is outside the UTC range") from error


def _parse_datetime(value: object, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        _schema_fail(f"{field} must be a non-empty ISO 8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise TemporalSchemaError(f"{field} must be ISO 8601") from error
    return _normalize_datetime(parsed, field)


def _validate_identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_PATTERN.fullmatch(value) is None:
        _schema_fail(f"{field} must be a stable ASCII identifier")
    return value


def _validate_text(value: object, field: str, *, maximum: int = 1024) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > maximum
        or "\x00" in value
    ):
        _schema_fail(f"{field} must be a non-empty string of at most {maximum} chars")
    return value


def _validate_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        _schema_fail(f"{field} must be a lowercase SHA-256 digest")
    return value


def _validate_fields(
    mapping: Mapping[str, object], expected: frozenset[str], context: str
) -> None:
    actual = frozenset(mapping)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        _schema_fail(f"{context} is missing fields: {', '.join(missing)}")
    if unknown:
        _schema_fail(f"{context} has unknown fields: {', '.join(unknown)}")


def _require_mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        _schema_fail(f"{field} must be an object with string keys")
    return cast(Mapping[str, object], value)


def _require_array(value: object, field: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        _schema_fail(f"{field} must be an array")
    return cast(Sequence[object], value)


def _require_int(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _schema_fail(f"{field} must be an integer greater than or equal to {minimum}")
    return value


def _require_bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        _schema_fail(f"{field} must be a boolean")
    return value


def _optional_sha256(value: object, field: str) -> str | None:
    if value is None:
        return None
    return _validate_sha256(value, field)


def _optional_datetime(value: object, field: str) -> datetime | None:
    if value is None:
        return None
    return _parse_datetime(value, field)


def _enum_value[T: StrEnum](value: object, enum_type: type[T], field: str) -> T:
    if not isinstance(value, str):
        _schema_fail(f"{field} must be a string")
    try:
        return enum_type(value)
    except ValueError as error:
        raise TemporalSchemaError(f"unsupported {field}: {value}") from error


def _validate_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "://" in value:
        _schema_fail("relative_path must be a non-empty portable relative path")
    candidate = PurePosixPath(value)
    if (
        candidate.is_absolute()
        or value != candidate.as_posix()
        or any(part in {"", ".", ".."} for part in candidate.parts)
        or value in _RESERVED_SNAPSHOT_PATHS
    ):
        _schema_fail("relative_path must not escape or collide with snapshot metadata")
    return value


@dataclass(frozen=True, slots=True)
class CutoffPolicy:
    """Immutable cutoff and partition policy.

    Event-time intervals are closed and disjoint, with an optional embargo
    between validation and policy activation:

    - train: ``event_time <= training_end``;
    - validation: ``training_end < event_time <= validation_end``;
    - post-policy evaluation: ``policy_start <= event_time <= cutoff``.
    """

    schema_version: int
    cutoff: datetime
    training_end: datetime
    validation_end: datetime
    policy_start: datetime
    allowed_license_tags: tuple[str, ...]

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            _schema_fail("CutoffPolicy schema_version must be 1")
        for field in ("cutoff", "training_end", "validation_end", "policy_start"):
            object.__setattr__(
                self, field, _normalize_datetime(getattr(self, field), field)
            )
        if (
            not self.training_end
            < self.validation_end
            < self.policy_start
            <= self.cutoff
        ):
            _schema_fail(
                "partition boundaries must satisfy "
                "training_end < validation_end < policy_start <= cutoff"
            )
        if (
            not isinstance(self.allowed_license_tags, tuple)
            or not self.allowed_license_tags
        ):
            _schema_fail("allowed_license_tags must be a non-empty immutable tuple")
        normalized = tuple(
            sorted(
                _validate_text(value, "allowed_license_tags item", maximum=128)
                for value in self.allowed_license_tags
            )
        )
        if len(set(normalized)) != len(normalized):
            _schema_fail("allowed_license_tags must be unique")
        object.__setattr__(self, "allowed_license_tags", normalized)

    def to_mapping(self) -> dict[str, object]:
        """Return the canonical machine-readable policy representation."""
        return {
            "allowed_license_tags": list(self.allowed_license_tags),
            "cutoff": self.cutoff.isoformat(),
            "policy_start": self.policy_start.isoformat(),
            "schema_version": self.schema_version,
            "training_end": self.training_end.isoformat(),
            "validation_end": self.validation_end.isoformat(),
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> Self:
        """Strictly construct a cutoff policy without defaults or unknown fields."""
        _validate_fields(
            mapping,
            frozenset(
                {
                    "schema_version",
                    "cutoff",
                    "training_end",
                    "validation_end",
                    "policy_start",
                    "allowed_license_tags",
                }
            ),
            "cutoff policy",
        )
        tags = tuple(
            _validate_text(item, "allowed_license_tags item", maximum=128)
            for item in _require_array(
                mapping["allowed_license_tags"], "allowed_license_tags"
            )
        )
        return cls(
            schema_version=_require_int(
                mapping["schema_version"], "schema_version", minimum=1
            ),
            cutoff=_parse_datetime(mapping["cutoff"], "cutoff"),
            training_end=_parse_datetime(mapping["training_end"], "training_end"),
            validation_end=_parse_datetime(mapping["validation_end"], "validation_end"),
            policy_start=_parse_datetime(mapping["policy_start"], "policy_start"),
            allowed_license_tags=tags,
        )

    @property
    def sha256(self) -> str:
        """Return the semantic SHA-256 commitment for this policy."""
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))


@dataclass(frozen=True, slots=True)
class TemporalRecord:
    """Required record-level temporal and provenance metadata."""

    record_id: str
    partition: DataPartition
    event_time: datetime
    available_time: datetime
    ingestion_time: datetime
    source: str
    version: str
    content_hash: str
    license_tag: str

    def __post_init__(self) -> None:
        _validate_identifier(self.record_id, "record_id")
        if not isinstance(self.partition, DataPartition):
            _schema_fail("partition must be a DataPartition")
        for field in ("event_time", "available_time", "ingestion_time"):
            object.__setattr__(
                self, field, _normalize_datetime(getattr(self, field), field)
            )
        if self.ingestion_time < self.available_time:
            _schema_fail("ingestion_time must not precede available_time")
        _validate_text(self.source, "source")
        _validate_text(self.version, "version", maximum=256)
        _validate_sha256(self.content_hash, "content_hash")
        _validate_text(self.license_tag, "license_tag", maximum=128)

    def to_mapping(self) -> dict[str, object]:
        """Return the canonical record representation."""
        return {
            "available_time": self.available_time.isoformat(),
            "content_hash": self.content_hash,
            "event_time": self.event_time.isoformat(),
            "ingestion_time": self.ingestion_time.isoformat(),
            "license_tag": self.license_tag,
            "partition": self.partition.value,
            "record_id": self.record_id,
            "source": self.source,
            "version": self.version,
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> Self:
        """Strictly reconstruct a temporal record from manifest JSON."""
        _validate_fields(
            mapping,
            frozenset(
                {
                    "record_id",
                    "partition",
                    "event_time",
                    "available_time",
                    "ingestion_time",
                    "source",
                    "version",
                    "content_hash",
                    "license_tag",
                }
            ),
            "temporal record",
        )
        return cls(
            record_id=_validate_identifier(mapping["record_id"], "record_id"),
            partition=_enum_value(mapping["partition"], DataPartition, "partition"),
            event_time=_parse_datetime(mapping["event_time"], "event_time"),
            available_time=_parse_datetime(mapping["available_time"], "available_time"),
            ingestion_time=_parse_datetime(mapping["ingestion_time"], "ingestion_time"),
            source=_validate_text(mapping["source"], "source"),
            version=_validate_text(mapping["version"], "version", maximum=256),
            content_hash=_validate_sha256(mapping["content_hash"], "content_hash"),
            license_tag=_validate_text(
                mapping["license_tag"], "license_tag", maximum=128
            ),
        )


type TableScalar = str | int | float | bool | None


def _normalize_table_values(
    values: Mapping[str, object],
) -> tuple[tuple[str, TableScalar], ...]:
    normalized: list[tuple[str, TableScalar]] = []
    for key, value in values.items():
        _validate_text(key, "table column name", maximum=256)
        if value is not None and not isinstance(value, str | int | float | bool):
            _schema_fail("table values must be JSON scalar values")
        if isinstance(value, float) and not math.isfinite(value):
            _schema_fail("table values must not contain non-finite floats")
        normalized.append((key, value))
    if not normalized:
        _schema_fail("table row values must not be empty")
    if len({key for key, _ in normalized}) != len(normalized):
        _schema_fail("table row column names must be unique")
    return tuple(sorted(normalized))


def table_values_sha256(values: Mapping[str, object]) -> str:
    """Hash one table row's scalar values independently of key order."""
    normalized = _normalize_table_values(values)
    return sha256_bytes(canonical_json_bytes(dict(normalized)))


@dataclass(frozen=True, slots=True)
class TemporalTableRow:
    """Strict JSONL table row with mandatory record-level temporal metadata."""

    record: TemporalRecord
    values: tuple[tuple[str, TableScalar], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.record, TemporalRecord):
            _schema_fail("table row record must be a TemporalRecord")
        if not isinstance(self.values, tuple):
            _schema_fail("table row values must be an immutable tuple")
        normalized = _normalize_table_values(dict(self.values))
        if len(normalized) != len(self.values):
            _schema_fail("table row values contain duplicate columns")
        object.__setattr__(self, "values", normalized)
        if self.record.content_hash != table_values_sha256(dict(normalized)):
            _schema_fail("table row content_hash does not match its scalar values")

    def to_mapping(self) -> dict[str, object]:
        """Return the canonical strict-table row representation."""
        return {"record": self.record.to_mapping(), "values": dict(self.values)}

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> Self:
        """Strictly reconstruct a row, rejecting absent temporal metadata."""
        _validate_fields(mapping, frozenset({"record", "values"}), "temporal table row")
        values = _normalize_table_values(
            _require_mapping(mapping["values"], "table row values")
        )
        return cls(
            record=TemporalRecord.from_mapping(
                _require_mapping(mapping["record"], "table row record")
            ),
            values=values,
        )


def encode_temporal_table(rows: Iterable[TemporalTableRow]) -> bytes:
    """Encode a non-empty strict temporal table as canonical JSONL."""
    row_tuple = tuple(rows)
    if not row_tuple or not all(isinstance(row, TemporalTableRow) for row in row_tuple):
        _schema_fail("temporal table requires at least one TemporalTableRow")
    record_ids = tuple(row.record.record_id for row in row_tuple)
    if len(set(record_ids)) != len(record_ids):
        _schema_fail("temporal table record IDs must be unique")
    return b"".join(canonical_json_bytes(row.to_mapping()) + b"\n" for row in row_tuple)


def decode_temporal_table(content: bytes) -> tuple[TemporalTableRow, ...]:
    """Strictly decode a complete temporal JSONL table without filtering rows.

    Decoding validates the row schema and row-level content commitments.  It
    does not authorize temporal use by itself; callers must still pass every
    decoded row through :class:`TemporalFirewall` for the intended purpose.
    """
    try:
        lines = content.decode("utf-8").splitlines()
    except UnicodeError as error:
        raise TemporalSchemaError(f"temporal table must be UTF-8: {error}") from error
    if not lines:
        _schema_fail("temporal table must not be empty")
    rows: list[TemporalTableRow] = []
    for line_number, line in enumerate(lines, start=1):
        if not line:
            _schema_fail(f"blank line in temporal table at row {line_number}")
        mapping = _parse_json_object_bytes(
            line.encode("utf-8"), f"temporal table row {line_number}"
        )
        rows.append(TemporalTableRow.from_mapping(mapping))
    record_ids = tuple(row.record.record_id for row in rows)
    if len(set(record_ids)) != len(record_ids):
        _schema_fail("temporal table contains duplicate record IDs")
    return tuple(rows)


# Kept private as an internal spelling for older call sites.  New derivation
# verifiers use the public, explicitly non-authorizing decoder above.
_parse_temporal_table = decode_temporal_table


@dataclass(frozen=True, slots=True)
class LineageReference:
    """Content-addressed reference to one actual upstream record."""

    record: TemporalRecord
    role: LineageRole

    def __post_init__(self) -> None:
        if not isinstance(self.record, TemporalRecord):
            _schema_fail("lineage record must be a complete TemporalRecord")
        if not isinstance(self.role, LineageRole):
            _schema_fail("lineage role must be a LineageRole")

    @property
    def record_id(self) -> str:
        """Return the stable upstream record ID."""
        return self.record.record_id

    @property
    def partition(self) -> DataPartition:
        """Return the upstream record partition."""
        return self.record.partition

    @property
    def available_time(self) -> datetime:
        """Return the upstream record availability time."""
        return self.record.available_time

    @property
    def content_hash(self) -> str:
        """Return the upstream payload commitment."""
        return self.record.content_hash

    def to_mapping(self) -> dict[str, object]:
        """Return the canonical lineage representation."""
        return {"record": self.record.to_mapping(), "role": self.role.value}

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> Self:
        """Strictly reconstruct a lineage reference."""
        _validate_fields(
            mapping,
            frozenset({"record", "role"}),
            "lineage reference",
        )
        return cls(
            record=TemporalRecord.from_mapping(
                _require_mapping(mapping["record"], "lineage record")
            ),
            role=_enum_value(mapping["role"], LineageRole, "lineage role"),
        )


def lineage_references_sha256(references: Iterable[LineageReference]) -> str:
    """Commit to complete upstream records and their declared lineage roles."""
    reference_tuple = tuple(references)
    if not reference_tuple or not all(
        isinstance(reference, LineageReference) for reference in reference_tuple
    ):
        _schema_fail("lineage commitment requires at least one reference")
    keys = tuple((reference.record_id, reference.role) for reference in reference_tuple)
    if len(set(keys)) != len(keys):
        _schema_fail("lineage commitment cannot contain duplicate references")
    mappings = [
        reference.to_mapping()
        for reference in sorted(
            reference_tuple, key=lambda item: (item.record_id, item.role.value)
        )
    ]
    return sha256_bytes(canonical_json_bytes(mappings))


@dataclass(frozen=True, slots=True)
class TemporalArtifact:
    """One file plus temporal metadata and content-addressed lineage."""

    relative_path: str
    kind: ArtifactKind
    record: TemporalRecord
    lineage: tuple[LineageReference, ...] = ()
    lineage_manifest_sha256: str | None = None
    config_sha256: str | None = None
    trained_through: datetime | None = None

    def __post_init__(self) -> None:
        _validate_relative_path(self.relative_path)
        if not isinstance(self.kind, ArtifactKind):
            _schema_fail("kind must be an ArtifactKind")
        if not isinstance(self.record, TemporalRecord):
            _schema_fail("record must be a TemporalRecord")
        if not isinstance(self.lineage, tuple) or not all(
            isinstance(reference, LineageReference) for reference in self.lineage
        ):
            _schema_fail("lineage must be an immutable tuple of LineageReference")
        lineage_keys = tuple((item.record_id, item.role) for item in self.lineage)
        if len(set(lineage_keys)) != len(lineage_keys):
            _schema_fail("lineage references must be unique")
        if self.lineage_manifest_sha256 is not None:
            _validate_sha256(self.lineage_manifest_sha256, "lineage_manifest_sha256")
        if self.config_sha256 is not None:
            _validate_sha256(self.config_sha256, "config_sha256")
        if self.trained_through is not None:
            object.__setattr__(
                self,
                "trained_through",
                _normalize_datetime(self.trained_through, "trained_through"),
            )
        if self.kind in {ArtifactKind.TABLE, ArtifactKind.TEXT}:
            if (
                self.lineage
                or self.lineage_manifest_sha256 is not None
                or self.config_sha256 is not None
                or self.trained_through is not None
            ):
                _schema_fail("raw table/text artifacts must not claim derived lineage")
            if self.kind is ArtifactKind.TABLE and not self.relative_path.endswith(
                ".jsonl"
            ):
                _schema_fail("table artifacts must use strict temporal JSONL files")
        elif self.kind is ArtifactKind.FEATURE:
            if not self.lineage or self.lineage_manifest_sha256 is None:
                _schema_fail("feature artifacts require committed input lineage")
            if self.config_sha256 is not None or self.trained_through is not None:
                _schema_fail("feature artifacts cannot contain checkpoint metadata")
        elif self.kind is ArtifactKind.MODEL_CHECKPOINT and (
            not self.lineage
            or self.lineage_manifest_sha256 is None
            or self.config_sha256 is None
            or self.trained_through is None
        ):
            _schema_fail("model checkpoints require complete training lineage")
        if self.kind in {ArtifactKind.FEATURE, ArtifactKind.MODEL_CHECKPOINT}:
            expected_lineage_sha256 = lineage_references_sha256(self.lineage)
            if self.lineage_manifest_sha256 != expected_lineage_sha256:
                _schema_fail(
                    "lineage_manifest_sha256 does not match the complete "
                    "upstream record lineage"
                )

    def to_mapping(self) -> dict[str, object]:
        """Return the canonical artifact representation."""
        return {
            "config_sha256": self.config_sha256,
            "kind": self.kind.value,
            "lineage": [item.to_mapping() for item in self.lineage],
            "lineage_manifest_sha256": self.lineage_manifest_sha256,
            "record": self.record.to_mapping(),
            "relative_path": self.relative_path,
            "trained_through": (
                self.trained_through.isoformat()
                if self.trained_through is not None
                else None
            ),
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> Self:
        """Strictly reconstruct an artifact from a snapshot manifest."""
        _validate_fields(
            mapping,
            frozenset(
                {
                    "relative_path",
                    "kind",
                    "record",
                    "lineage",
                    "lineage_manifest_sha256",
                    "config_sha256",
                    "trained_through",
                }
            ),
            "temporal artifact",
        )
        lineage = tuple(
            LineageReference.from_mapping(_require_mapping(item, "lineage reference"))
            for item in _require_array(mapping["lineage"], "lineage")
        )
        return cls(
            relative_path=_validate_relative_path(mapping["relative_path"]),
            kind=_enum_value(mapping["kind"], ArtifactKind, "artifact kind"),
            record=TemporalRecord.from_mapping(
                _require_mapping(mapping["record"], "record")
            ),
            lineage=lineage,
            lineage_manifest_sha256=_optional_sha256(
                mapping["lineage_manifest_sha256"], "lineage_manifest_sha256"
            ),
            config_sha256=_optional_sha256(mapping["config_sha256"], "config_sha256"),
            trained_through=_optional_datetime(
                mapping["trained_through"], "trained_through"
            ),
        )


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    """One immutable artifact entry in a temporal snapshot manifest."""

    artifact: TemporalArtifact
    size_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, TemporalArtifact):
            _schema_fail("manifest artifact must be a TemporalArtifact")
        _require_int(self.size_bytes, "size_bytes")

    def to_mapping(self) -> dict[str, object]:
        """Return the canonical manifest-entry representation."""
        return {"artifact": self.artifact.to_mapping(), "size_bytes": self.size_bytes}

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> Self:
        """Strictly reconstruct a manifest entry."""
        _validate_fields(
            mapping, frozenset({"artifact", "size_bytes"}), "manifest entry"
        )
        return cls(
            artifact=TemporalArtifact.from_mapping(
                _require_mapping(mapping["artifact"], "artifact")
            ),
            size_bytes=_require_int(mapping["size_bytes"], "size_bytes"),
        )


@dataclass(frozen=True, slots=True)
class TemporalDataManifest:
    """Canonical manifest for a content-addressed frozen snapshot."""

    schema_version: int
    snapshot_id: str
    policy_sha256: str
    created_at: datetime
    frozen: bool
    entries: tuple[ManifestEntry, ...]

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            _schema_fail("TemporalDataManifest schema_version must be 1")
        _validate_identifier(self.snapshot_id, "snapshot_id")
        _validate_sha256(self.policy_sha256, "policy_sha256")
        object.__setattr__(
            self, "created_at", _normalize_datetime(self.created_at, "created_at")
        )
        if self.frozen is not True:
            _schema_fail("temporal snapshot manifest must be frozen")
        if not isinstance(self.entries, tuple) or not self.entries:
            _schema_fail("manifest entries must be a non-empty immutable tuple")
        if not all(isinstance(entry, ManifestEntry) for entry in self.entries):
            _schema_fail("manifest entries contain an invalid value")
        ordered = tuple(
            sorted(self.entries, key=lambda item: item.artifact.relative_path)
        )
        paths = tuple(entry.artifact.relative_path for entry in ordered)
        record_ids = tuple(entry.artifact.record.record_id for entry in ordered)
        if len(set(paths)) != len(paths):
            _schema_fail("manifest artifact paths must be unique")
        if len(set(record_ids)) != len(record_ids):
            _schema_fail("manifest record IDs must be unique")
        object.__setattr__(self, "entries", ordered)

    def to_mapping(self) -> dict[str, object]:
        """Return the canonical manifest body used for hashing."""
        return {
            "created_at": self.created_at.isoformat(),
            "entries": [entry.to_mapping() for entry in self.entries],
            "frozen": self.frozen,
            "policy_sha256": self.policy_sha256,
            "schema_version": self.schema_version,
            "snapshot_id": self.snapshot_id,
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> Self:
        """Strictly reconstruct a snapshot manifest body."""
        _validate_fields(
            mapping,
            frozenset(
                {
                    "schema_version",
                    "snapshot_id",
                    "policy_sha256",
                    "created_at",
                    "frozen",
                    "entries",
                }
            ),
            "temporal data manifest",
        )
        entries = tuple(
            ManifestEntry.from_mapping(_require_mapping(item, "manifest entry"))
            for item in _require_array(mapping["entries"], "entries")
        )
        return cls(
            schema_version=_require_int(
                mapping["schema_version"], "schema_version", minimum=1
            ),
            snapshot_id=_validate_identifier(mapping["snapshot_id"], "snapshot_id"),
            policy_sha256=_validate_sha256(mapping["policy_sha256"], "policy_sha256"),
            created_at=_parse_datetime(mapping["created_at"], "created_at"),
            frozen=_require_bool(mapping["frozen"], "frozen"),
            entries=entries,
        )


@dataclass(frozen=True, slots=True)
class FrozenSnapshot:
    """Trusted handle to an immutable snapshot commitment."""

    root: Path
    snapshot_id: str
    manifest_sha256: str
    policy_sha256: str
    publication_token: str

    def __post_init__(self) -> None:
        if not isinstance(self.root, Path) or not self.root.is_absolute():
            _schema_fail("frozen snapshot root must be an absolute Path")
        _validate_identifier(self.snapshot_id, "snapshot_id")
        _validate_sha256(self.manifest_sha256, "manifest_sha256")
        _validate_sha256(self.policy_sha256, "policy_sha256")
        _validate_sha256(self.publication_token, "publication_token")


@dataclass(frozen=True, slots=True)
class ValidatedArtifact:
    """Immutable bytes returned only after temporal and integrity checks."""

    artifact: TemporalArtifact
    payload: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, TemporalArtifact):
            _schema_fail("validated artifact metadata are invalid")
        if not isinstance(self.payload, bytes):
            _schema_fail("validated artifact payload must be immutable bytes")


def temporal_records_sha256(records: Iterable[TemporalRecord]) -> str:
    """Commit to the complete, sorted metadata of actual input records."""
    record_tuple = tuple(records)
    if not record_tuple or not all(
        isinstance(record, TemporalRecord) for record in record_tuple
    ):
        _schema_fail("record commitment requires at least one TemporalRecord")
    record_ids = tuple(record.record_id for record in record_tuple)
    if len(set(record_ids)) != len(record_ids):
        _schema_fail("record commitment cannot contain duplicate record IDs")
    mappings = [
        record.to_mapping()
        for record in sorted(record_tuple, key=lambda item: item.record_id)
    ]
    return sha256_bytes(canonical_json_bytes(mappings))


@dataclass(frozen=True, slots=True)
class StandardizationSpec:
    """Actual records and declared endpoint used to fit a scaler."""

    fit_records: tuple[TemporalRecord, ...]
    fit_end: datetime
    fit_on_full_sample: bool = False

    def __post_init__(self) -> None:
        temporal_records_sha256(self.fit_records)
        object.__setattr__(
            self, "fit_end", _normalize_datetime(self.fit_end, "fit_end")
        )
        if not isinstance(self.fit_on_full_sample, bool):
            _schema_fail("fit_on_full_sample must be a boolean")

    @property
    def fit_records_sha256(self) -> str:
        """Return the commitment to actual scaler-fit inputs."""
        return temporal_records_sha256(self.fit_records)


@dataclass(frozen=True, slots=True)
class ForwardFillSpec:
    """A claimed forward fill from one actual record to another."""

    source_record: TemporalRecord
    target_record: TemporalRecord
    as_of_time: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.source_record, TemporalRecord) or not isinstance(
            self.target_record, TemporalRecord
        ):
            _schema_fail("forward-fill records must be TemporalRecord values")
        object.__setattr__(
            self, "as_of_time", _normalize_datetime(self.as_of_time, "as_of_time")
        )


@dataclass(frozen=True, slots=True)
class WindowAggregationSpec:
    """Declared aggregation window bound to its actual input records."""

    records: tuple[TemporalRecord, ...]
    window_start: datetime
    window_end: datetime
    output_time: datetime
    output_partition: DataPartition

    def __post_init__(self) -> None:
        temporal_records_sha256(self.records)
        for field in ("window_start", "window_end", "output_time"):
            object.__setattr__(
                self, field, _normalize_datetime(getattr(self, field), field)
            )
        if self.window_end < self.window_start:
            _schema_fail("window_end must not precede window_start")
        if not isinstance(self.output_partition, DataPartition):
            _schema_fail("output_partition must be a DataPartition")

    @property
    def input_records_sha256(self) -> str:
        """Return the commitment to actual aggregation inputs."""
        return temporal_records_sha256(self.records)


@dataclass(frozen=True, slots=True)
class LabelConstructionSpec:
    """A label horizon bound to actual label-source records."""

    records: tuple[TemporalRecord, ...]
    prediction_time: datetime
    label_start: datetime
    label_end: datetime
    output_partition: DataPartition

    def __post_init__(self) -> None:
        temporal_records_sha256(self.records)
        for field in ("prediction_time", "label_start", "label_end"):
            object.__setattr__(
                self, field, _normalize_datetime(getattr(self, field), field)
            )
        if self.label_start < self.prediction_time:
            _schema_fail("label_start must not precede prediction_time")
        if self.label_end < self.label_start:
            _schema_fail("label_end must not precede label_start")
        if not isinstance(self.output_partition, DataPartition):
            _schema_fail("output_partition must be a DataPartition")

    @property
    def input_records_sha256(self) -> str:
        """Return the commitment to actual label inputs."""
        return temporal_records_sha256(self.records)


class TemporalAuditLog:
    """Mandatory append-only, hash-chained JSONL audit sink."""

    _FIELDS = frozenset(
        {
            "schema_version",
            "sequence",
            "event_type",
            "occurred_at",
            "action",
            "outcome",
            "rule_code",
            "policy_sha256",
            "cutoff",
            "artifact_id",
            "record_id",
            "partition",
            "details",
            "previous_event_sha256",
            "event_sha256",
        }
    )

    def __init__(self, path: Path) -> None:
        expanded = path.expanduser()
        self._path = Path(os.path.abspath(os.fspath(expanded)))
        self._lock = RLock()
        self._decision_lock = _audit_decision_lock(self._path)
        events = self._read_and_validate()
        self._sequence = len(events)
        self._previous_hash = (
            cast(str, events[-1]["event_sha256"]) if events else _ZERO_SHA256
        )

    @property
    def path(self) -> Path:
        """Return the absolute audit-log path."""
        return self._path

    def _read_and_validate(
        self, *, allow_locked_empty: bool = False
    ) -> tuple[dict[str, object], ...]:
        if not self._path.exists():
            return ()
        if self._path.is_symlink() or not self._path.is_file():
            raise AuditLogError("audit log must be a regular non-symlink file")
        if self._path.stat().st_size == 0:
            if allow_locked_empty:
                return ()
            raise AuditLogError("temporal audit log is empty")
        try:
            events = read_jsonl_objects(self._path)
        except MetadataValidationError as error:
            raise AuditLogError(f"invalid temporal audit log: {error}") from error
        previous_hash = _ZERO_SHA256
        for expected_sequence, event in enumerate(events):
            if frozenset(event) != self._FIELDS:
                raise AuditLogError("audit event fields are missing or unknown")
            try:
                if event["schema_version"] != 1:
                    _schema_fail("audit schema_version must be 1")
                if event["event_type"] != "temporal_firewall":
                    _schema_fail("audit event_type is invalid")
                _parse_datetime(event["occurred_at"], "audit occurred_at")
                if event["outcome"] not in {"allowed", "rejected"}:
                    _schema_fail("audit outcome is invalid")
                _enum_value(event["rule_code"], RuleCode, "audit rule_code")
                _validate_sha256(event["policy_sha256"], "audit policy_sha256")
                _parse_datetime(event["cutoff"], "audit cutoff")
                _require_mapping(event["details"], "audit details")
                for field in ("artifact_id", "record_id"):
                    value = event[field]
                    if value is not None:
                        _validate_text(value, f"audit {field}")
                partition = event["partition"]
                if partition is not None:
                    _enum_value(partition, DataPartition, "audit partition")
            except TemporalSchemaError as error:
                raise AuditLogError(
                    f"audit event schema is invalid: {error}"
                ) from error
            sequence = event["sequence"]
            if (
                isinstance(sequence, bool)
                or not isinstance(sequence, int)
                or sequence != expected_sequence
            ):
                raise AuditLogError("audit event sequence is not contiguous")
            if event["previous_event_sha256"] != previous_hash:
                raise AuditLogError("audit event hash chain is broken")
            event_sha256 = event["event_sha256"]
            if (
                not isinstance(event_sha256, str)
                or _SHA256_PATTERN.fullmatch(event_sha256) is None
            ):
                raise AuditLogError("audit event digest is invalid")
            core = dict(event)
            del core["event_sha256"]
            if sha256_bytes(canonical_json_bytes(core)) != event_sha256:
                raise AuditLogError("audit event digest does not match its content")
            previous_hash = event_sha256
        return events

    def record(
        self,
        *,
        action: str,
        outcome: str,
        rule_code: RuleCode,
        policy_sha256: str,
        cutoff: str,
        artifact_id: str | None,
        record_id: str | None,
        partition: str | None,
        details: Mapping[str, object],
    ) -> None:
        """Synchronously append and fsync one chained audit event."""
        _validate_text(action, "audit action", maximum=128)
        if outcome not in {"allowed", "rejected"}:
            _schema_fail("audit outcome must be allowed or rejected")
        if not isinstance(rule_code, RuleCode):
            _schema_fail("audit rule_code must be a RuleCode")
        _validate_sha256(policy_sha256, "audit policy_sha256")
        _parse_datetime(cutoff, "audit cutoff")
        with self._lock:
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                flags = os.O_APPEND | os.O_RDWR
                if hasattr(os, "O_NOFOLLOW"):
                    flags |= os.O_NOFOLLOW
                try:
                    descriptor = os.open(
                        self._path, flags | os.O_CREAT | os.O_EXCL, 0o600
                    )
                    created_exclusively = True
                except FileExistsError:
                    descriptor = os.open(self._path, flags, 0o600)
                    created_exclusively = False
                with os.fdopen(descriptor, "a+b") as stream:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                    opened = os.fstat(stream.fileno())
                    linked = os.lstat(self._path)
                    if (opened.st_dev, opened.st_ino) != (
                        linked.st_dev,
                        linked.st_ino,
                    ):
                        raise AuditLogError("audit log path changed during open")
                    current = self._read_and_validate(
                        allow_locked_empty=created_exclusively
                    )
                    self._sequence = len(current)
                    self._previous_hash = (
                        cast(str, current[-1]["event_sha256"])
                        if current
                        else _ZERO_SHA256
                    )
                    core: dict[str, object] = {
                        "action": action,
                        "artifact_id": artifact_id,
                        "cutoff": cutoff,
                        "details": dict(details),
                        "event_type": "temporal_firewall",
                        "occurred_at": datetime.now(UTC).isoformat(),
                        "outcome": outcome,
                        "partition": partition,
                        "policy_sha256": policy_sha256,
                        "previous_event_sha256": self._previous_hash,
                        "record_id": record_id,
                        "rule_code": rule_code.value,
                        "schema_version": 1,
                        "sequence": self._sequence,
                    }
                    event_sha256 = sha256_bytes(canonical_json_bytes(core))
                    event = {**core, "event_sha256": event_sha256}
                    rendered = canonical_json_bytes(event) + b"\n"
                    stream.seek(0, os.SEEK_END)
                    stream.write(rendered)
                    stream.flush()
                    os.fsync(stream.fileno())
                    linked_after = os.lstat(self._path)
                    if (opened.st_dev, opened.st_ino) != (
                        linked_after.st_dev,
                        linked_after.st_ino,
                    ):
                        raise AuditLogError("audit log path changed during append")
            except OSError as error:
                raise AuditLogError(
                    f"cannot write mandatory audit log: {error}"
                ) from error
            self._sequence += 1
            self._previous_hash = event_sha256

    def events(self) -> tuple[dict[str, object], ...]:
        """Return all validated audit events."""
        with self._lock:
            return self._read_and_validate()


class _CheckFailure(Exception):
    def __init__(
        self,
        error_type: type[TemporalFirewallError],
        rule_code: RuleCode,
        message: str,
        details: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.rule_code = rule_code
        self.message = message
        self.details = dict(details or {})


def _check_fail(
    error_type: type[TemporalFirewallError],
    rule_code: RuleCode,
    message: str,
    details: Mapping[str, object] | None = None,
) -> NoReturn:
    raise _CheckFailure(error_type, rule_code, message, details)


_T = TypeVar("_T")
_EVALUATION_FIREWALL: ContextVar[object | None] = ContextVar(
    "policytwin_frozen_evaluation_firewall", default=None
)
_NETWORK_CONTEXT_LOCK = RLock()
_ACTIVE_NETWORK_FIREWALLS: list[object] = []
_NETWORK_AUDIT_EVENTS = frozenset(
    {
        "os.exec",
        "os.fork",
        "os.forkpty",
        "os.posix_spawn",
        "os.spawn",
        "os.system",
        "socket.__new__",
        "socket.bind",
        "socket.connect",
        "socket.getaddrinfo",
        "socket.gethostbyaddr",
        "socket.gethostbyname",
        "socket.sendmsg",
        "socket.sendto",
        "subprocess.Popen",
    }
)


def _strict_object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _schema_fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> NoReturn:
    _schema_fail(f"non-finite JSON number is forbidden: {value}")


def _parse_json_object_bytes(content: bytes, context: str) -> dict[str, object]:
    try:
        value = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_strict_object_pairs,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise TemporalSchemaError(f"invalid JSON in {context}: {error}") from error
    if not isinstance(value, dict):
        _schema_fail(f"top-level JSON in {context} must be an object")
    return cast(dict[str, object], value)


def _partition_rank(partition: DataPartition) -> int:
    return {
        DataPartition.TRAIN: 0,
        DataPartition.VALIDATION: 1,
        DataPartition.POST_POLICY_EVALUATION: 2,
    }[partition]


def _open_socket_file_descriptors() -> tuple[int, ...]:
    descriptors: list[int] = []
    for candidate in gc.get_objects():
        if isinstance(candidate, _ORIGINAL_SOCKET_TYPE):
            try:
                descriptor = candidate.fileno()
            except OSError:
                continue
            if descriptor >= 0:
                descriptors.append(descriptor)
    return tuple(sorted(descriptors))


class TemporalFirewall:
    """Validate all temporal data access through a mandatory audit boundary."""

    def __init__(self, policy: CutoffPolicy, audit_log: TemporalAuditLog) -> None:
        if not isinstance(policy, CutoffPolicy):
            _schema_fail("policy must be a CutoffPolicy")
        if not isinstance(audit_log, TemporalAuditLog):
            _schema_fail(
                "audit_log must be a TemporalAuditLog; no-op logs are forbidden"
            )
        self._policy = policy
        self._audit = audit_log
        self._latched_failure: tuple[RuleCode, str] | None = None
        for event in self._audit.events():
            rule_code = RuleCode(cast(str, event["rule_code"]))
            if event["outcome"] == "rejected" and rule_code in _LATCHING_RULES:
                self._latched_failure = (rule_code, cast(str, event["action"]))

    @property
    def policy(self) -> CutoffPolicy:
        """Return the immutable cutoff policy."""
        return self._policy

    @property
    def audit_log(self) -> TemporalAuditLog:
        """Return the mandatory audit sink."""
        return self._audit

    @property
    def frozen_evaluation_active(self) -> bool:
        """Return whether this firewall owns the active no-network context.

        This is intentionally an identity check rather than a caller-supplied
        flag.  Downstream bridges use it to ensure a verified capability is
        consumed before the frozen evaluation boundary is exited.
        """
        return _EVALUATION_FIREWALL.get() is self

    def _refresh_latched_failure(self) -> None:
        for event in self._audit.events():
            rule_code = RuleCode(cast(str, event["rule_code"]))
            if event["outcome"] == "rejected" and rule_code in _LATCHING_RULES:
                self._latched_failure = (rule_code, cast(str, event["action"]))

    def _run_check(
        self,
        *,
        action: str,
        check: Callable[[], _T],
        artifact_id: str | None = None,
        record: TemporalRecord | None = None,
        success_details: Mapping[str, object] | None = None,
    ) -> _T:
        with self._audit._decision_lock:
            self._refresh_latched_failure()
            return self._run_check_locked(
                action=action,
                check=check,
                artifact_id=artifact_id,
                record=record,
                success_details=success_details,
            )

    def _run_check_locked(
        self,
        *,
        action: str,
        check: Callable[[], _T],
        artifact_id: str | None = None,
        record: TemporalRecord | None = None,
        success_details: Mapping[str, object] | None = None,
    ) -> _T:
        if self._latched_failure is not None:
            previous_rule, previous_action = self._latched_failure
            message = (
                "temporal firewall is latched after a prior rejected validity "
                "check; start a new audited job only after resolving the failure"
            )
            self._audit.record(
                action=action,
                outcome="rejected",
                rule_code=RuleCode.PRIOR_REJECTION,
                policy_sha256=self._policy.sha256,
                cutoff=self._policy.cutoff.isoformat(),
                artifact_id=artifact_id,
                record_id=record.record_id if record is not None else None,
                partition=record.partition.value if record is not None else None,
                details={
                    "message": message,
                    "previous_action": previous_action,
                    "previous_rule_code": previous_rule.value,
                },
            )
            raise TemporalLeakageError(message)
        try:
            result = check()
        except _CheckFailure as failure:
            details = {"message": failure.message, **failure.details}
            if failure.rule_code in _LATCHING_RULES:
                self._latched_failure = (failure.rule_code, action)
            self._audit.record(
                action=action,
                outcome="rejected",
                rule_code=failure.rule_code,
                policy_sha256=self._policy.sha256,
                cutoff=self._policy.cutoff.isoformat(),
                artifact_id=artifact_id,
                record_id=record.record_id if record is not None else None,
                partition=record.partition.value if record is not None else None,
                details=details,
            )
            raise failure.error_type(failure.message) from failure
        self._audit.record(
            action=action,
            outcome="allowed",
            rule_code=RuleCode.OK,
            policy_sha256=self._policy.sha256,
            cutoff=self._policy.cutoff.isoformat(),
            artifact_id=artifact_id,
            record_id=record.record_id if record is not None else None,
            partition=record.partition.value if record is not None else None,
            details=dict(success_details or {}),
        )
        return result

    def _partition_end(self, partition: DataPartition) -> datetime:
        if partition is DataPartition.TRAIN:
            return self._policy.training_end
        if partition is DataPartition.VALIDATION:
            return self._policy.validation_end
        return self._policy.cutoff

    def _expected_partition(self, event_time: datetime) -> DataPartition:
        if event_time > self._policy.cutoff:
            _check_fail(
                PostCutoffDataError,
                RuleCode.EVENT_AFTER_CUTOFF,
                "event_time is after the configured cutoff",
                {
                    "event_time": event_time.isoformat(),
                    "cutoff": self._policy.cutoff.isoformat(),
                },
            )
        if event_time <= self._policy.training_end:
            return DataPartition.TRAIN
        if event_time <= self._policy.validation_end:
            return DataPartition.VALIDATION
        if event_time < self._policy.policy_start:
            _check_fail(
                PartitionAccessError,
                RuleCode.PARTITION_VIOLATION,
                "event_time falls in the embargo between validation and policy start",
                {
                    "event_time": event_time.isoformat(),
                    "validation_end": self._policy.validation_end.isoformat(),
                    "policy_start": self._policy.policy_start.isoformat(),
                },
            )
        return DataPartition.POST_POLICY_EVALUATION

    def _check_record(self, record: TemporalRecord) -> None:
        if not isinstance(record, TemporalRecord):
            _check_fail(
                TemporalSchemaError,
                RuleCode.INVALID_SCHEMA,
                "value is not a TemporalRecord",
            )
        if record.available_time > self._policy.cutoff:
            _check_fail(
                PostCutoffDataError,
                RuleCode.POST_CUTOFF_AVAILABLE_TIME,
                "available_time is after the configured cutoff",
                {
                    "available_time": record.available_time.isoformat(),
                    "cutoff": self._policy.cutoff.isoformat(),
                    "source": record.source,
                    "version": record.version,
                },
            )
        if record.license_tag not in self._policy.allowed_license_tags:
            _check_fail(
                TemporalLeakageError,
                RuleCode.UNTRUSTED_LICENSE,
                "license_tag is not permitted by the cutoff policy",
                {"license_tag": record.license_tag},
            )
        expected = self._expected_partition(record.event_time)
        if record.partition is not expected:
            _check_fail(
                PartitionAccessError,
                RuleCode.PARTITION_VIOLATION,
                "record partition does not match its event_time",
                {
                    "declared_partition": record.partition.value,
                    "expected_partition": expected.value,
                    "event_time": record.event_time.isoformat(),
                },
            )
        partition_end = self._partition_end(record.partition)
        if record.available_time > partition_end:
            _check_fail(
                TemporalLeakageError,
                RuleCode.PARTITION_VIOLATION,
                "record was not available by the end of its declared partition",
                {
                    "available_time": record.available_time.isoformat(),
                    "partition_end": partition_end.isoformat(),
                },
            )

    def _check_purpose_context(self, purpose: AccessPurpose) -> None:
        if not isinstance(purpose, AccessPurpose):
            _check_fail(
                TemporalSchemaError,
                RuleCode.INVALID_SCHEMA,
                "purpose must be an AccessPurpose",
            )
        if (
            purpose is AccessPurpose.FROZEN_EVALUATION
            and _EVALUATION_FIREWALL.get() is not self
        ):
            _check_fail(
                PartitionAccessError,
                RuleCode.EVALUATION_CONTEXT_REQUIRED,
                "frozen evaluation data require an active no-network context",
            )

    def _check_access(self, partition: DataPartition, purpose: AccessPurpose) -> None:
        self._check_purpose_context(purpose)
        if purpose is AccessPurpose.FROZEN_EVALUATION:
            return
        if purpose is AccessPurpose.TRAINING and partition is not DataPartition.TRAIN:
            _check_fail(
                PartitionAccessError,
                RuleCode.PARTITION_ACCESS_DENIED,
                "training jobs may load only the training partition",
                {"partition": partition.value, "purpose": purpose.value},
            )
        if purpose is AccessPurpose.VALIDATION and partition not in {
            DataPartition.TRAIN,
            DataPartition.VALIDATION,
        }:
            _check_fail(
                PartitionAccessError,
                RuleCode.PARTITION_ACCESS_DENIED,
                "validation jobs cannot inspect post-policy evaluation data",
                {"partition": partition.value, "purpose": purpose.value},
            )

    def _check_lineage_reference(
        self, reference: LineageReference, artifact: TemporalArtifact
    ) -> None:
        self._check_record(reference.record)
        if _partition_rank(reference.partition) > _partition_rank(
            artifact.record.partition
        ):
            rule_code = (
                RuleCode.CHECKPOINT_LINEAGE_INVALID
                if artifact.kind is ArtifactKind.MODEL_CHECKPOINT
                else RuleCode.FEATURE_LINEAGE_INVALID
            )
            _check_fail(
                TransformationLeakageError,
                rule_code,
                "artifact lineage uses a later partition",
                {
                    "lineage_record_id": reference.record_id,
                    "lineage_partition": reference.partition.value,
                    "artifact_partition": artifact.record.partition.value,
                },
            )

    def _check_artifact(self, artifact: TemporalArtifact) -> None:
        if not isinstance(artifact, TemporalArtifact):
            _check_fail(
                TemporalSchemaError,
                RuleCode.INVALID_SCHEMA,
                "value is not a TemporalArtifact",
            )
        self._check_record(artifact.record)
        for reference in artifact.lineage:
            self._check_lineage_reference(reference, artifact)
        if artifact.kind is ArtifactKind.FEATURE:
            for reference in artifact.lineage:
                if reference.role is LineageRole.LABEL:
                    _check_fail(
                        TransformationLeakageError,
                        RuleCode.FEATURE_LINEAGE_INVALID,
                        "label records cannot enter feature lineage",
                        {"lineage_record_id": reference.record_id},
                    )
                if (
                    reference.available_time > artifact.record.event_time
                    or artifact.record.available_time < reference.available_time
                    or artifact.record.available_time < artifact.record.event_time
                ):
                    _check_fail(
                        TransformationLeakageError,
                        RuleCode.FEATURE_LINEAGE_INVALID,
                        "feature availability precedes its as-of time or input lineage",
                        {
                            "feature_available_time": (
                                artifact.record.available_time.isoformat()
                            ),
                            "lineage_available_time": (
                                reference.available_time.isoformat()
                            ),
                            "lineage_record_id": reference.record_id,
                        },
                    )
        if artifact.kind is ArtifactKind.MODEL_CHECKPOINT:
            assert artifact.trained_through is not None
            if artifact.config_sha256 != self._policy.sha256:
                _check_fail(
                    TransformationLeakageError,
                    RuleCode.CHECKPOINT_LINEAGE_INVALID,
                    "checkpoint config commitment does not match the cutoff policy",
                )
            if (
                artifact.trained_through > self._policy.training_end
                or artifact.record.available_time < artifact.trained_through
            ):
                _check_fail(
                    TransformationLeakageError,
                    RuleCode.CHECKPOINT_LINEAGE_INVALID,
                    "checkpoint training or availability time is inconsistent",
                    {
                        "checkpoint_available_time": (
                            artifact.record.available_time.isoformat()
                        ),
                        "trained_through": artifact.trained_through.isoformat(),
                        "training_end": self._policy.training_end.isoformat(),
                    },
                )
            for reference in artifact.lineage:
                if (
                    reference.partition is not DataPartition.TRAIN
                    or reference.available_time > self._policy.training_end
                    or reference.available_time > artifact.trained_through
                    or reference.available_time > artifact.record.available_time
                ):
                    _check_fail(
                        TransformationLeakageError,
                        RuleCode.CHECKPOINT_LINEAGE_INVALID,
                        "checkpoint lineage is not restricted to training inputs",
                        {
                            "lineage_record_id": reference.record_id,
                            "lineage_partition": reference.partition.value,
                            "lineage_role": reference.role.value,
                        },
                    )

    def validate_record(self, record: TemporalRecord) -> None:
        """Validate one record without reading payload data."""
        self._run_check(
            action="validate_record",
            check=lambda: self._check_record(record),
            record=record if isinstance(record, TemporalRecord) else None,
        )

    def validate_records(
        self, records: Iterable[TemporalRecord]
    ) -> tuple[TemporalRecord, ...]:
        """Atomically validate a batch; never return a filtered subset."""
        record_tuple = tuple(records)

        def check() -> tuple[TemporalRecord, ...]:
            if not record_tuple:
                _check_fail(
                    TemporalSchemaError,
                    RuleCode.INVALID_SCHEMA,
                    "record batch must not be empty",
                )
            if not all(isinstance(record, TemporalRecord) for record in record_tuple):
                _check_fail(
                    TemporalSchemaError,
                    RuleCode.INVALID_SCHEMA,
                    "record batch contains a non-TemporalRecord value",
                )
            if len({record.record_id for record in record_tuple}) != len(record_tuple):
                _check_fail(
                    TemporalSchemaError,
                    RuleCode.INVALID_SCHEMA,
                    "record batch contains duplicate record IDs",
                )
            for record in record_tuple:
                self._check_record(record)
            return record_tuple

        return self._run_check(
            action="validate_record_batch",
            check=check,
            success_details={"record_count": len(record_tuple)},
        )

    def validate_artifact(
        self, artifact: TemporalArtifact, purpose: AccessPurpose
    ) -> None:
        """Validate artifact metadata, lineage, and partition access."""

        action_kind = (
            artifact.kind.value
            if isinstance(artifact, TemporalArtifact)
            else "artifact"
        )
        purpose_label = (
            purpose.value if isinstance(purpose, AccessPurpose) else "invalid"
        )

        def check() -> None:
            self._check_artifact(artifact)
            self._check_access(artifact.record.partition, purpose)

        self._run_check(
            action=f"validate_{action_kind}",
            check=check,
            artifact_id=(
                artifact.relative_path
                if isinstance(artifact, TemporalArtifact)
                else None
            ),
            record=artifact.record if isinstance(artifact, TemporalArtifact) else None,
            success_details={"purpose": purpose_label},
        )

    def _read_regular_file(
        self,
        root: Path,
        relative_path: str,
        *,
        require_read_only: bool,
    ) -> bytes:
        raw_root = root.expanduser()
        try:
            if raw_root.is_symlink():
                _check_fail(
                    ArtifactIntegrityError,
                    RuleCode.PATH_VIOLATION,
                    "artifact root must not be a symlink",
                )
            resolved_root = raw_root.resolve(strict=True)
        except OSError as error:
            _check_fail(
                ArtifactIntegrityError,
                RuleCode.PATH_VIOLATION,
                f"cannot resolve artifact root: {error}",
            )
        if not resolved_root.is_dir():
            _check_fail(
                ArtifactIntegrityError,
                RuleCode.PATH_VIOLATION,
                "artifact root must be a directory",
            )
        current = resolved_root
        try:
            for part in PurePosixPath(relative_path).parts:
                current /= part
                item_stat = os.lstat(current)
                if stat.S_ISLNK(item_stat.st_mode):
                    _check_fail(
                        ArtifactIntegrityError,
                        RuleCode.PATH_VIOLATION,
                        "artifact path must not contain symlinks",
                        {"relative_path": relative_path},
                    )
            before = os.lstat(current)
            if not stat.S_ISREG(before.st_mode):
                _check_fail(
                    ArtifactIntegrityError,
                    RuleCode.PATH_VIOLATION,
                    "artifact path must refer to a regular file",
                )
            if require_read_only and before.st_mode & 0o222:
                _check_fail(
                    FrozenSnapshotError,
                    RuleCode.FROZEN_MUTATION,
                    "frozen snapshot artifact has writable permission bits",
                    {"relative_path": relative_path},
                )
            with current.open("rb") as stream:
                opened_before = os.fstat(stream.fileno())
                content = stream.read()
                opened_after = os.fstat(stream.fileno())
            after = os.lstat(current)
        except _CheckFailure:
            raise
        except OSError as error:
            _check_fail(
                ArtifactIntegrityError,
                RuleCode.PATH_VIOLATION,
                f"cannot read artifact: {error}",
                {"relative_path": relative_path},
            )
        signatures = {
            (
                value.st_dev,
                value.st_ino,
                value.st_size,
                value.st_mtime_ns,
            )
            for value in (before, opened_before, opened_after, after)
        }
        if len(signatures) != 1:
            _check_fail(
                ArtifactIntegrityError,
                RuleCode.HASH_MISMATCH,
                "artifact changed while it was being read",
                {"relative_path": relative_path},
            )
        return content

    def _load_artifact_inner(
        self,
        root: Path,
        artifact: TemporalArtifact,
        purpose: AccessPurpose,
        expected_kind: ArtifactKind | None,
    ) -> ValidatedArtifact:
        self._check_artifact(artifact)
        if expected_kind is not None and artifact.kind is not expected_kind:
            _check_fail(
                TemporalSchemaError,
                RuleCode.INVALID_SCHEMA,
                f"expected {expected_kind.value}, got {artifact.kind.value}",
            )
        self._check_access(artifact.record.partition, purpose)
        if purpose is AccessPurpose.FROZEN_EVALUATION:
            _check_fail(
                PartitionAccessError,
                RuleCode.SNAPSHOT_REQUIRED,
                "frozen evaluation payloads must be loaded from a verified snapshot",
            )
        payload = self._read_regular_file(
            root, artifact.relative_path, require_read_only=False
        )
        observed_hash = sha256_bytes(payload)
        if observed_hash != artifact.record.content_hash:
            _check_fail(
                ArtifactIntegrityError,
                RuleCode.HASH_MISMATCH,
                "artifact content hash does not match its temporal record",
                {
                    "expected_sha256": artifact.record.content_hash,
                    "observed_sha256": observed_hash,
                },
            )
        if artifact.kind is ArtifactKind.TABLE:
            self._check_table_payload(payload, purpose, artifact.record)
        if artifact.kind is ArtifactKind.TEXT:
            try:
                payload.decode("utf-8")
            except UnicodeError as error:
                _check_fail(
                    ArtifactIntegrityError,
                    RuleCode.INVALID_SCHEMA,
                    f"text artifact is not valid UTF-8: {error}",
                )
        return ValidatedArtifact(artifact=artifact, payload=payload)

    def _check_table_payload(
        self,
        payload: bytes,
        purpose: AccessPurpose,
        container_record: TemporalRecord,
    ) -> tuple[TemporalTableRow, ...]:
        try:
            rows = _parse_temporal_table(payload)
        except TemporalSchemaError as error:
            _check_fail(
                TemporalSchemaError,
                RuleCode.INVALID_SCHEMA,
                f"table does not use the strict temporal JSONL schema: {error}",
            )
        for row in rows:
            try:
                self._check_record(row.record)
                self._check_access(row.record.partition, purpose)
            except _CheckFailure as failure:
                failure.details.setdefault("table_record_id", row.record.record_id)
                raise
            if (
                container_record.available_time < row.record.available_time
                or container_record.ingestion_time < row.record.ingestion_time
            ):
                _check_fail(
                    TemporalLeakageError,
                    RuleCode.PARTITION_VIOLATION,
                    "table container predates one of its records",
                    {"table_record_id": row.record.record_id},
                )
        return rows

    def _load_expected_kind(
        self,
        root: Path,
        artifact: TemporalArtifact,
        purpose: AccessPurpose,
        kind: ArtifactKind,
    ) -> ValidatedArtifact:
        artifact_id = (
            artifact.relative_path if isinstance(artifact, TemporalArtifact) else None
        )
        record = artifact.record if isinstance(artifact, TemporalArtifact) else None
        purpose_label = (
            purpose.value if isinstance(purpose, AccessPurpose) else "invalid"
        )
        return self._run_check(
            action=f"load_{kind.value}",
            check=lambda: self._load_artifact_inner(root, artifact, purpose, kind),
            artifact_id=artifact_id,
            record=record,
            success_details={"purpose": purpose_label},
        )

    def load_table(
        self, root: Path, artifact: TemporalArtifact, purpose: AccessPurpose
    ) -> ValidatedArtifact:
        """Load an integrity-checked table without silently dropping rows."""
        return self._load_expected_kind(root, artifact, purpose, ArtifactKind.TABLE)

    def load_text(
        self, root: Path, artifact: TemporalArtifact, purpose: AccessPurpose
    ) -> ValidatedArtifact:
        """Load an integrity-checked UTF-8 text artifact."""
        return self._load_expected_kind(root, artifact, purpose, ArtifactKind.TEXT)

    def load_feature(
        self, root: Path, artifact: TemporalArtifact, purpose: AccessPurpose
    ) -> ValidatedArtifact:
        """Load a feature only after validating its complete input lineage."""
        return self._load_expected_kind(root, artifact, purpose, ArtifactKind.FEATURE)

    def load_checkpoint(
        self, root: Path, artifact: TemporalArtifact, purpose: AccessPurpose
    ) -> ValidatedArtifact:
        """Load checkpoint bytes without unsafe deserialization."""
        return self._load_expected_kind(
            root, artifact, purpose, ArtifactKind.MODEL_CHECKPOINT
        )

    def check_standardization(self, spec: StandardizationSpec) -> None:
        """Reject scalers fit outside the committed training record set."""

        def check() -> None:
            if spec.fit_on_full_sample or any(
                record.partition is not DataPartition.TRAIN
                for record in spec.fit_records
            ):
                _check_fail(
                    TransformationLeakageError,
                    RuleCode.FULL_SAMPLE_FIT,
                    "scaler fitting is restricted to the training partition",
                    {"fit_records_sha256": spec.fit_records_sha256},
                )
            if spec.fit_end > self._policy.training_end:
                _check_fail(
                    TransformationLeakageError,
                    RuleCode.FULL_SAMPLE_FIT,
                    "scaler fit_end exceeds the training boundary",
                    {"fit_end": spec.fit_end.isoformat()},
                )
            latest_event = max(record.event_time for record in spec.fit_records)
            if spec.fit_end != latest_event:
                _check_fail(
                    TransformationLeakageError,
                    RuleCode.FULL_SAMPLE_FIT,
                    "scaler fit_end does not match its committed fit records",
                    {
                        "fit_end": spec.fit_end.isoformat(),
                        "latest_fit_event": latest_event.isoformat(),
                    },
                )
            for record in spec.fit_records:
                self._check_record(record)
                if (
                    record.available_time > self._policy.training_end
                    or record.available_time > spec.fit_end
                ):
                    _check_fail(
                        TransformationLeakageError,
                        RuleCode.FULL_SAMPLE_FIT,
                        "scaler input was unavailable by its declared fit_end",
                        {
                            "record_id": record.record_id,
                            "available_time": record.available_time.isoformat(),
                            "fit_end": spec.fit_end.isoformat(),
                        },
                    )

        self._run_check(
            action="check_standardization",
            check=check,
            success_details={"fit_records_sha256": spec.fit_records_sha256},
        )

    def check_forward_fill(self, spec: ForwardFillSpec, purpose: AccessPurpose) -> None:
        """Reject backfill, future-source fill, and cross-partition fill."""

        def check() -> None:
            self._check_record(spec.source_record)
            self._check_record(spec.target_record)
            self._check_access(spec.target_record.partition, purpose)
            if (
                spec.source_record.partition is not spec.target_record.partition
                or spec.source_record.source != spec.target_record.source
                or spec.source_record.version != spec.target_record.version
                or spec.source_record.event_time > spec.target_record.event_time
                or spec.source_record.event_time > spec.as_of_time
                or spec.source_record.available_time > spec.as_of_time
                or spec.as_of_time > spec.target_record.event_time
                or spec.as_of_time > self._policy.cutoff
            ):
                _check_fail(
                    TransformationLeakageError,
                    RuleCode.FORWARD_FILL_LEAKAGE,
                    "forward fill would use a future, cross-source, or "
                    "cross-partition value",
                    {
                        "source_record_id": spec.source_record.record_id,
                        "target_record_id": spec.target_record.record_id,
                        "as_of_time": spec.as_of_time.isoformat(),
                    },
                )

        self._run_check(action="check_forward_fill", check=check)

    def check_window_aggregation(
        self, spec: WindowAggregationSpec, purpose: AccessPurpose
    ) -> None:
        """Reject windows that include unavailable or post-cutoff information."""

        def check() -> None:
            if spec.window_end > self._policy.cutoff:
                _check_fail(
                    TransformationLeakageError,
                    RuleCode.CROSS_CUTOFF_WINDOW,
                    "aggregation window crosses the configured cutoff",
                    {
                        "window_start": spec.window_start.isoformat(),
                        "window_end": spec.window_end.isoformat(),
                        "cutoff": self._policy.cutoff.isoformat(),
                    },
                )
            if (
                spec.window_end > spec.output_time
                or spec.output_time > self._policy.cutoff
            ):
                _check_fail(
                    TransformationLeakageError,
                    RuleCode.CROSS_CUTOFF_WINDOW,
                    "aggregation output_time is earlier than its window "
                    "or after cutoff",
                )
            expected = self._expected_partition(spec.output_time)
            if expected is not spec.output_partition:
                _check_fail(
                    PartitionAccessError,
                    RuleCode.PARTITION_VIOLATION,
                    "aggregation output partition does not match output_time",
                )
            self._check_access(spec.output_partition, purpose)
            for record in spec.records:
                self._check_record(record)
                if not spec.window_start <= record.event_time <= spec.window_end:
                    _check_fail(
                        TransformationLeakageError,
                        RuleCode.CROSS_CUTOFF_WINDOW,
                        "aggregation input lies outside the declared window",
                        {"record_id": record.record_id},
                    )
                if record.available_time > spec.output_time:
                    _check_fail(
                        TransformationLeakageError,
                        RuleCode.CROSS_CUTOFF_WINDOW,
                        "aggregation input was unavailable at output_time",
                        {"record_id": record.record_id},
                    )
                if _partition_rank(record.partition) > _partition_rank(
                    spec.output_partition
                ):
                    _check_fail(
                        TransformationLeakageError,
                        RuleCode.CROSS_CUTOFF_WINDOW,
                        "aggregation uses a later partition",
                        {"record_id": record.record_id},
                    )

        self._run_check(
            action="check_window_aggregation",
            check=check,
            success_details={"input_records_sha256": spec.input_records_sha256},
        )

    def check_label_construction(
        self, spec: LabelConstructionSpec, purpose: AccessPurpose
    ) -> None:
        """Keep label horizons inside the declared partition and cutoff."""

        def check() -> None:
            expected = self._expected_partition(spec.prediction_time)
            if expected is not spec.output_partition:
                _check_fail(
                    TransformationLeakageError,
                    RuleCode.LABEL_LEAKAGE,
                    "label output partition does not match prediction_time",
                )
            self._check_access(spec.output_partition, purpose)
            boundary = self._partition_end(spec.output_partition)
            if spec.label_end > boundary or spec.label_end > self._policy.cutoff:
                _check_fail(
                    TransformationLeakageError,
                    RuleCode.LABEL_LEAKAGE,
                    "label horizon crosses its partition or the configured cutoff",
                    {
                        "label_start": spec.label_start.isoformat(),
                        "label_end": spec.label_end.isoformat(),
                        "partition_end": boundary.isoformat(),
                    },
                )
            for record in spec.records:
                self._check_record(record)
                if (
                    record.partition is not spec.output_partition
                    or not spec.label_start <= record.event_time <= spec.label_end
                    or record.available_time > boundary
                ):
                    _check_fail(
                        TransformationLeakageError,
                        RuleCode.LABEL_LEAKAGE,
                        "label input is outside the committed label horizon "
                        "or partition",
                        {"record_id": record.record_id},
                    )

        self._run_check(
            action="check_label_construction",
            check=check,
            success_details={"input_records_sha256": spec.input_records_sha256},
        )

    @staticmethod
    def _remove_tree(path: Path) -> None:
        if not os.path.lexists(path):
            return
        root_stat = os.lstat(path)
        if stat.S_ISLNK(root_stat.st_mode):
            path.unlink()
            return
        if not stat.S_ISDIR(root_stat.st_mode):
            path.unlink()
            return

        def raise_walk_error(error: OSError) -> NoReturn:
            raise error

        for directory, child_directories, files in os.walk(
            path, topdown=False, followlinks=False, onerror=raise_walk_error
        ):
            directory_path = Path(directory)
            for name in files:
                child = directory_path / name
                child_stat = os.lstat(child)
                if stat.S_ISREG(child_stat.st_mode):
                    child.chmod(0o600)
            for name in child_directories:
                child = directory_path / name
                child_stat = os.lstat(child)
                if stat.S_ISDIR(child_stat.st_mode):
                    child.chmod(0o700)
            directory_path.chmod(0o700)
        shutil.rmtree(path)
        if os.path.lexists(path):
            raise OSError(f"snapshot cleanup did not remove {path}")

    @staticmethod
    def _remove_tree_if_identity(path: Path, identity: tuple[int, int]) -> None:
        """Remove only the directory inode exclusively claimed by this producer."""
        try:
            info = os.lstat(path)
        except FileNotFoundError:
            return
        if (
            stat.S_ISLNK(info.st_mode)
            or not stat.S_ISDIR(info.st_mode)
            or (info.st_dev, info.st_ino) != identity
        ):
            raise OSError("snapshot path changed identity before cleanup")
        TemporalFirewall._remove_tree(path)
        if os.path.lexists(path):
            raise OSError("snapshot cleanup left the claimed inode path in place")

    @staticmethod
    def _open_snapshot_directory_at(root_descriptor: int, parts: Sequence[str]) -> int:
        descriptor = os.dup(root_descriptor)
        try:
            for part in parts:
                child = os.open(
                    part,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=descriptor,
                )
                info = os.fstat(child)
                if not stat.S_ISDIR(info.st_mode):
                    os.close(child)
                    raise OSError("snapshot path component is not a directory")
                os.close(descriptor)
                descriptor = child
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    @staticmethod
    def _write_snapshot_file_at(
        root_descriptor: int, relative_path: str, content: bytes
    ) -> None:
        parts = PurePosixPath(relative_path).parts
        parent_descriptor = os.dup(root_descriptor)
        try:
            for part in parts[:-1]:
                with suppress(FileExistsError):
                    os.mkdir(part, mode=0o700, dir_fd=parent_descriptor)
                child_descriptor = TemporalFirewall._open_snapshot_directory_at(
                    parent_descriptor, (part,)
                )
                os.close(parent_descriptor)
                parent_descriptor = child_descriptor
            descriptor = os.open(
                parts[-1],
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=parent_descriptor,
            )
            try:
                before = os.fstat(descriptor)
                if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                    raise OSError("snapshot output is not a private regular file")
                written = 0
                while written < len(content):
                    count = os.write(descriptor, content[written:])
                    if count <= 0:
                        raise OSError("short write while freezing snapshot")
                    written += count
                os.fsync(descriptor)
                os.fchmod(descriptor, 0o444)
                after = os.fstat(descriptor)
                if (
                    (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
                    or after.st_size != len(content)
                    or after.st_nlink != 1
                ):
                    raise OSError("snapshot output changed while it was written")
            finally:
                os.close(descriptor)
        finally:
            os.close(parent_descriptor)

    @staticmethod
    def _seal_snapshot_directories(root_descriptor: int, directories: set[str]) -> None:
        for relative in sorted(
            directories,
            key=lambda value: len(PurePosixPath(value).parts),
            reverse=True,
        ):
            descriptor = TemporalFirewall._open_snapshot_directory_at(
                root_descriptor, PurePosixPath(relative).parts
            )
            try:
                os.fchmod(descriptor, 0o555)
            finally:
                os.close(descriptor)
        os.fchmod(root_descriptor, 0o555)

    def freeze_snapshot(
        self,
        source_root: Path,
        destination_root: Path,
        artifacts: Iterable[TemporalArtifact],
        *,
        snapshot_id: str,
        purpose: AccessPurpose,
        created_at: datetime | None = None,
    ) -> FrozenSnapshot:
        """Copy validated inputs to a new read-only, content-addressed snapshot."""
        artifact_tuple = tuple(artifacts)
        publication_state = [False]
        destination_state: list[Path] = []
        destination_identity_state: list[tuple[int, int]] = []

        def remove_claimed_snapshot(path: Path) -> None:
            if not destination_identity_state:
                raise OSError("claimed snapshot identity is unavailable for cleanup")
            self._remove_tree_if_identity(path, destination_identity_state[0])

        def claimed_snapshot_is_current(path: Path) -> bool:
            if not destination_identity_state:
                return False
            try:
                info = os.lstat(path)
            except OSError:
                return False
            return (
                stat.S_ISDIR(info.st_mode)
                and not stat.S_ISLNK(info.st_mode)
                and (info.st_dev, info.st_ino) == destination_identity_state[0]
            )

        publication_token = secrets.token_hex(32)
        success_details: dict[str, object] = {
            "artifact_count": len(artifact_tuple),
            "publication_token_sha256": sha256_bytes(publication_token.encode("ascii")),
            "purpose": (
                purpose.value if isinstance(purpose, AccessPurpose) else "invalid"
            ),
        }
        audit_snapshot_id = (
            snapshot_id
            if isinstance(snapshot_id, str)
            and _IDENTIFIER_PATTERN.fullmatch(snapshot_id) is not None
            else None
        )

        def check() -> FrozenSnapshot:
            try:
                stable_snapshot_id = _validate_identifier(snapshot_id, "snapshot_id")
                if not isinstance(source_root, Path) or not isinstance(
                    destination_root, Path
                ):
                    _schema_fail("snapshot roots must be Path values")
                destination = destination_root.expanduser().resolve()
                created = (
                    datetime.now(UTC)
                    if created_at is None
                    else _normalize_datetime(created_at, "created_at")
                )
            except (OSError, TemporalSchemaError) as error:
                _check_fail(
                    TemporalSchemaError,
                    RuleCode.INVALID_SCHEMA,
                    f"invalid snapshot request: {error}",
                )
            destination_state.append(destination)
            publication_path = _snapshot_publication_path(
                destination, publication_token
            )
            if not artifact_tuple:
                _check_fail(
                    TemporalSchemaError,
                    RuleCode.INVALID_SCHEMA,
                    "a frozen snapshot must contain at least one artifact",
                )
            for artifact in artifact_tuple:
                self._check_artifact(artifact)
                self._check_access(artifact.record.partition, purpose)
            paths = tuple(artifact.relative_path for artifact in artifact_tuple)
            if len(set(paths)) != len(paths):
                _check_fail(
                    TemporalSchemaError,
                    RuleCode.INVALID_SCHEMA,
                    "snapshot artifacts contain duplicate relative paths",
                )
            latest_ingestion = max(
                timestamp
                for artifact in artifact_tuple
                for timestamp in (
                    artifact.record.ingestion_time,
                    *(
                        reference.record.ingestion_time
                        for reference in artifact.lineage
                    ),
                )
            )
            if created < latest_ingestion:
                _check_fail(
                    TemporalSchemaError,
                    RuleCode.INVALID_SCHEMA,
                    "snapshot created_at precedes an artifact ingestion_time",
                    {
                        "created_at": created.isoformat(),
                        "latest_ingestion_time": latest_ingestion.isoformat(),
                    },
                )
            if destination.exists() or os.path.lexists(publication_path):
                _check_fail(
                    FrozenSnapshotError,
                    RuleCode.FROZEN_MUTATION,
                    "snapshot destination or publication receipt already exists "
                    "and will not be overwritten",
                    {
                        "destination": str(destination),
                        "publication_receipt": str(publication_path),
                    },
                )
            try:
                self._audit.path.relative_to(destination)
            except ValueError:
                pass
            else:
                _check_fail(
                    FrozenSnapshotError,
                    RuleCode.PATH_VIOLATION,
                    "audit log cannot be stored inside the frozen snapshot",
                )

            payloads: list[tuple[TemporalArtifact, bytes]] = []
            for artifact in artifact_tuple:
                payload = self._read_regular_file(
                    source_root,
                    artifact.relative_path,
                    require_read_only=False,
                )
                observed_hash = sha256_bytes(payload)
                if observed_hash != artifact.record.content_hash:
                    _check_fail(
                        ArtifactIntegrityError,
                        RuleCode.HASH_MISMATCH,
                        "source artifact hash does not match its temporal record",
                        {
                            "relative_path": artifact.relative_path,
                            "expected_sha256": artifact.record.content_hash,
                            "observed_sha256": observed_hash,
                        },
                    )
                if artifact.kind is ArtifactKind.TABLE:
                    self._check_table_payload(payload, purpose, artifact.record)
                if artifact.kind is ArtifactKind.TEXT:
                    try:
                        payload.decode("utf-8")
                    except UnicodeError as error:
                        _check_fail(
                            ArtifactIntegrityError,
                            RuleCode.INVALID_SCHEMA,
                            f"text artifact is not valid UTF-8: {error}",
                        )
                payloads.append((artifact, payload))

            published = False
            try:
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists() or os.path.lexists(publication_path):
                    _check_fail(
                        FrozenSnapshotError,
                        RuleCode.FROZEN_MUTATION,
                        "snapshot destination or publication receipt appeared "
                        "during publication",
                    )
                destination.mkdir(mode=0o700)
                published = True
                publication_state[0] = True
                with ExitStack() as descriptors:
                    root_descriptor = os.open(
                        destination,
                        os.O_RDONLY
                        | getattr(os, "O_DIRECTORY", 0)
                        | getattr(os, "O_NOFOLLOW", 0),
                    )
                    descriptors.callback(os.close, root_descriptor)
                    opened_root = os.fstat(root_descriptor)
                    linked_root = os.lstat(destination)
                    root_identity = (opened_root.st_dev, opened_root.st_ino)
                    if (
                        not stat.S_ISDIR(opened_root.st_mode)
                        or root_identity != (linked_root.st_dev, linked_root.st_ino)
                        or os.listdir(root_descriptor)
                    ):
                        raise OSError(
                            "claimed snapshot directory changed or was not empty"
                        )
                    destination_identity_state.append(root_identity)
                    entries: list[ManifestEntry] = []
                    snapshot_directories: set[str] = set()
                    for artifact, payload in payloads:
                        self._write_snapshot_file_at(
                            root_descriptor, artifact.relative_path, payload
                        )
                        parent = PurePosixPath(artifact.relative_path).parent
                        while parent != PurePosixPath("."):
                            snapshot_directories.add(parent.as_posix())
                            parent = parent.parent
                        entries.append(
                            ManifestEntry(artifact=artifact, size_bytes=len(payload))
                        )
                    manifest = TemporalDataManifest(
                        schema_version=1,
                        snapshot_id=stable_snapshot_id,
                        policy_sha256=self._policy.sha256,
                        created_at=created,
                        frozen=True,
                        entries=tuple(entries),
                    )
                    manifest_bytes = canonical_json_bytes(manifest.to_mapping()) + b"\n"
                    manifest_sha256 = sha256_bytes(manifest_bytes)
                    seal = f"{manifest_sha256}  {_MANIFEST_FILE}\n".encode("ascii")
                    self._write_snapshot_file_at(
                        root_descriptor, _MANIFEST_FILE, manifest_bytes
                    )
                    self._write_snapshot_file_at(
                        root_descriptor, _MANIFEST_SEAL_FILE, seal
                    )
                    self._write_snapshot_file_at(
                        root_descriptor, _FROZEN_MARKER_FILE, seal
                    )
                    self._seal_snapshot_directories(
                        root_descriptor, snapshot_directories
                    )
                    linked_after = os.lstat(destination)
                    opened_after = os.fstat(root_descriptor)
                    if (
                        (linked_after.st_dev, linked_after.st_ino) != root_identity
                        or (opened_after.st_dev, opened_after.st_ino) != root_identity
                        or opened_after.st_mode & 0o222
                    ):
                        raise OSError(
                            "snapshot destination changed during "
                            "descriptor-anchored build"
                        )
                directory_fd = os.open(destination.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
                success_details["manifest_sha256"] = manifest_sha256
                return FrozenSnapshot(
                    root=destination,
                    snapshot_id=stable_snapshot_id,
                    manifest_sha256=manifest_sha256,
                    policy_sha256=self._policy.sha256,
                    publication_token=publication_token,
                )
            except _CheckFailure as failure:
                try:
                    if published:
                        remove_claimed_snapshot(destination)
                except OSError as failed_check_cleanup_error:
                    _check_fail(
                        FrozenSnapshotError,
                        RuleCode.FROZEN_MUTATION,
                        f"{failure.message}; snapshot cleanup failed: "
                        f"{failed_check_cleanup_error}",
                    )
                raise
            except (OSError, TemporalSchemaError, MetadataValidationError) as error:
                publication_cleanup_error: OSError | None = None
                try:
                    if published:
                        remove_claimed_snapshot(destination)
                except OSError as caught_publication_cleanup_error:
                    publication_cleanup_error = caught_publication_cleanup_error
                cleanup_suffix = (
                    ""
                    if publication_cleanup_error is None
                    else f"; snapshot cleanup failed: {publication_cleanup_error}"
                )
                _check_fail(
                    FrozenSnapshotError,
                    RuleCode.FROZEN_MUTATION,
                    f"cannot freeze snapshot: {error}{cleanup_suffix}",
                )

        with self._audit._decision_lock:
            self._refresh_latched_failure()
            try:
                snapshot = self._run_check_locked(
                    action="freeze_snapshot",
                    check=check,
                    artifact_id=audit_snapshot_id,
                    success_details=success_details,
                )
            except AuditLogError as error:
                if publication_state[0] and destination_state:
                    try:
                        remove_claimed_snapshot(destination_state[0])
                    except OSError as audit_cleanup_error:
                        raise AuditLogError(
                            f"{error}; unpublished snapshot cleanup failed: "
                            f"{audit_cleanup_error}"
                        ) from error
                    try:
                        os.lstat(destination_state[0])
                    except FileNotFoundError:
                        pass
                    except OSError as audit_cleanup_error:
                        raise AuditLogError(
                            f"{error}; cannot prove unpublished snapshot cleanup: "
                            f"{audit_cleanup_error}"
                        ) from error
                    else:
                        raise AuditLogError(
                            f"{error}; unpublished snapshot cleanup left the "
                            "snapshot in place"
                        ) from error
                raise

            marker_path = snapshot.root / _FROZEN_MARKER_FILE
            publication_path = _snapshot_publication_path(
                snapshot.root, snapshot.publication_token
            )
            receipt_linked = False
            expected_receipt_identity: tuple[int, int] | None = None

            def remove_linked_receipt() -> None:
                if not receipt_linked or expected_receipt_identity is None:
                    return
                try:
                    info = os.lstat(publication_path)
                except FileNotFoundError:
                    return
                if (
                    stat.S_ISLNK(info.st_mode)
                    or not stat.S_ISREG(info.st_mode)
                    or (info.st_dev, info.st_ino) != expected_receipt_identity
                ):
                    raise OSError(
                        "snapshot publication receipt changed identity before cleanup"
                    )
                publication_path.unlink()
                try:
                    os.lstat(publication_path)
                except FileNotFoundError:
                    return
                raise OSError("snapshot publication receipt remains after cleanup")

            try:
                # This exclusive hard-link creation is intentionally the final
                # filesystem publication operation. The loader also requires
                # the succeeding final audit authorization.
                if not claimed_snapshot_is_current(snapshot.root):
                    raise OSError(
                        "claimed snapshot path changed before final authorization"
                    )
                marker_info = os.lstat(marker_path)
                expected_receipt_identity = (marker_info.st_dev, marker_info.st_ino)
                os.link(marker_path, publication_path, follow_symlinks=False)
                receipt_linked = True
                publication_signature = self._snapshot_publication_signature(
                    snapshot.root, snapshot.publication_token
                )
            except (OSError, _CheckFailure) as error:
                authorization_cleanup_errors: list[str] = []
                try:
                    remove_linked_receipt()
                except OSError as cleanup_error:
                    authorization_cleanup_errors.append(
                        f"publication receipt cleanup failed: {cleanup_error}"
                    )
                try:
                    remove_claimed_snapshot(snapshot.root)
                except OSError as cleanup_error:
                    authorization_cleanup_errors.append(
                        f"unpublished snapshot cleanup failed: {cleanup_error}"
                    )
                else:
                    try:
                        os.lstat(snapshot.root)
                    except FileNotFoundError:
                        pass
                    except OSError as cleanup_error:
                        authorization_cleanup_errors.append(
                            "cannot prove unpublished snapshot cleanup: "
                            f"{cleanup_error}"
                        )
                    else:
                        authorization_cleanup_errors.append(
                            "unpublished snapshot remains after cleanup"
                        )
                reason = (
                    error.message if isinstance(error, _CheckFailure) else str(error)
                )
                message = f"cannot authorize frozen snapshot publication: {reason}"
                if authorization_cleanup_errors:
                    message = f"{message}; " + "; ".join(authorization_cleanup_errors)
                self._audit.record(
                    action="finalize_snapshot_publication",
                    outcome="rejected",
                    rule_code=RuleCode.FROZEN_MUTATION,
                    policy_sha256=self._policy.sha256,
                    cutoff=self._policy.cutoff.isoformat(),
                    artifact_id=audit_snapshot_id,
                    record_id=None,
                    partition=None,
                    details={"message": message},
                )
                raise FrozenSnapshotError(message) from error
            try:
                self._audit.record(
                    action="finalize_snapshot_publication",
                    outcome="allowed",
                    rule_code=RuleCode.OK,
                    policy_sha256=self._policy.sha256,
                    cutoff=self._policy.cutoff.isoformat(),
                    artifact_id=audit_snapshot_id,
                    record_id=None,
                    partition=None,
                    details={
                        "manifest_sha256": snapshot.manifest_sha256,
                        "publication_token_sha256": sha256_bytes(
                            snapshot.publication_token.encode("ascii")
                        ),
                    },
                )
            except AuditLogError as error:
                cleanup_errors: list[str] = []
                try:
                    remove_linked_receipt()
                except OSError as cleanup_error:
                    cleanup_errors.append(f"receipt cleanup failed: {cleanup_error}")
                try:
                    remove_claimed_snapshot(snapshot.root)
                except OSError as cleanup_error:
                    cleanup_errors.append(f"snapshot cleanup failed: {cleanup_error}")
                else:
                    try:
                        os.lstat(snapshot.root)
                    except FileNotFoundError:
                        pass
                    except OSError as cleanup_error:
                        cleanup_errors.append(
                            f"cannot prove snapshot cleanup: {cleanup_error}"
                        )
                    else:
                        cleanup_errors.append(
                            "snapshot cleanup left the snapshot in place"
                        )
                if cleanup_errors:
                    raise AuditLogError(
                        f"{error}; " + "; ".join(cleanup_errors)
                    ) from error
                raise
            recheck_failure: _CheckFailure | None = None
            try:
                publication_changed = (
                    not claimed_snapshot_is_current(snapshot.root)
                    or self._snapshot_publication_signature(
                        snapshot.root, snapshot.publication_token
                    )
                    != publication_signature
                )
            except _CheckFailure as failure:
                recheck_failure = failure
                publication_changed = True
            if publication_changed:
                message = (
                    "snapshot publication receipt changed during final audit "
                    "authorization"
                )
                revocation_error: AuditLogError | None = None
                try:
                    self._audit.record(
                        action="revoke_snapshot_publication",
                        outcome="rejected",
                        rule_code=RuleCode.FROZEN_MUTATION,
                        policy_sha256=self._policy.sha256,
                        cutoff=self._policy.cutoff.isoformat(),
                        artifact_id=audit_snapshot_id,
                        record_id=None,
                        partition=None,
                        details={
                            "manifest_sha256": snapshot.manifest_sha256,
                            "message": message,
                            "publication_token_sha256": sha256_bytes(
                                snapshot.publication_token.encode("ascii")
                            ),
                        },
                    )
                except AuditLogError as error:
                    revocation_error = error
                cleanup_errors = []
                try:
                    remove_linked_receipt()
                except OSError as error:
                    cleanup_errors.append(f"receipt cleanup failed: {error}")
                try:
                    remove_claimed_snapshot(snapshot.root)
                except OSError as error:
                    cleanup_errors.append(f"snapshot cleanup failed: {error}")
                else:
                    try:
                        os.lstat(snapshot.root)
                    except FileNotFoundError:
                        pass
                    except OSError as error:
                        cleanup_errors.append(f"cannot prove snapshot cleanup: {error}")
                    else:
                        cleanup_errors.append(
                            "snapshot cleanup left the snapshot in place"
                        )
                if revocation_error is not None and cleanup_errors:
                    raise AuditLogError(
                        f"{revocation_error}; " + "; ".join(cleanup_errors)
                    ) from revocation_error
                if cleanup_errors:
                    raise FrozenSnapshotError(
                        f"{message}; " + "; ".join(cleanup_errors)
                    ) from recheck_failure
                raise FrozenSnapshotError(message) from recheck_failure
            return snapshot

    def _snapshot_tree_signature(
        self,
        root: Path,
        expected_files: set[str],
        expected_directories: set[str],
    ) -> tuple[tuple[str, int, int, int, int, int, int], ...]:
        actual_files: set[str] = set()
        actual_directories: set[str] = set()
        signatures: list[tuple[str, int, int, int, int, int, int]] = []
        try:
            root_stat = os.lstat(root)
            if (
                stat.S_ISLNK(root_stat.st_mode)
                or not stat.S_ISDIR(root_stat.st_mode)
                or root_stat.st_mode & 0o222
            ):
                _check_fail(
                    FrozenSnapshotError,
                    RuleCode.FROZEN_MUTATION,
                    "snapshot root is not an immutable regular directory",
                )
            signatures.append(
                (
                    ".",
                    root_stat.st_dev,
                    root_stat.st_ino,
                    root_stat.st_size,
                    root_stat.st_mtime_ns,
                    root_stat.st_mode,
                    root_stat.st_nlink,
                )
            )
            for item in root.rglob("*"):
                relative = item.relative_to(root).as_posix()
                item_stat = os.lstat(item)
                if stat.S_ISLNK(item_stat.st_mode):
                    _check_fail(
                        ArtifactIntegrityError,
                        RuleCode.PATH_VIOLATION,
                        "frozen snapshot contains a symlink",
                        {"relative_path": relative},
                    )
                if stat.S_ISDIR(item_stat.st_mode):
                    if item_stat.st_mode & 0o222:
                        _check_fail(
                            FrozenSnapshotError,
                            RuleCode.FROZEN_MUTATION,
                            "frozen snapshot contains a writable directory",
                            {"relative_path": relative},
                        )
                    actual_directories.add(relative)
                elif stat.S_ISREG(item_stat.st_mode):
                    expected_links = 2 if relative == _FROZEN_MARKER_FILE else 1
                    if (
                        item_stat.st_mode & 0o222
                        or item_stat.st_nlink != expected_links
                    ):
                        _check_fail(
                            FrozenSnapshotError,
                            RuleCode.FROZEN_MUTATION,
                            "frozen snapshot file has writable permissions or an "
                            "invalid hard-link count",
                            {
                                "expected_links": expected_links,
                                "observed_links": item_stat.st_nlink,
                                "relative_path": relative,
                            },
                        )
                    actual_files.add(relative)
                else:
                    _check_fail(
                        ArtifactIntegrityError,
                        RuleCode.PATH_VIOLATION,
                        "frozen snapshot contains a non-regular filesystem entry",
                        {"relative_path": relative},
                    )
                signatures.append(
                    (
                        relative,
                        item_stat.st_dev,
                        item_stat.st_ino,
                        item_stat.st_size,
                        item_stat.st_mtime_ns,
                        item_stat.st_mode,
                        item_stat.st_nlink,
                    )
                )
        except _CheckFailure:
            raise
        except OSError as error:
            _check_fail(
                ArtifactIntegrityError,
                RuleCode.PATH_VIOLATION,
                f"cannot enumerate frozen snapshot: {error}",
            )
        if actual_files != expected_files or actual_directories != expected_directories:
            _check_fail(
                ArtifactIntegrityError,
                RuleCode.MANIFEST_MISMATCH,
                "frozen snapshot file set does not exactly match its manifest",
                {
                    "missing_files": sorted(expected_files - actual_files),
                    "unexpected_files": sorted(actual_files - expected_files),
                    "unexpected_directories": sorted(
                        actual_directories - expected_directories
                    ),
                },
            )
        return tuple(sorted(signatures))

    def _snapshot_publication_signature(
        self, root: Path, publication_token: str
    ) -> tuple[tuple[int, int, int, int, int], tuple[int, int, int, int, int]]:
        publication_path = _snapshot_publication_path(root, publication_token)
        marker_path = root / _FROZEN_MARKER_FILE
        try:
            publication_stat = os.lstat(publication_path)
            marker_stat = os.lstat(marker_path)
        except OSError as error:
            _check_fail(
                ArtifactIntegrityError,
                RuleCode.MANIFEST_MISMATCH,
                f"snapshot has no valid publication receipt: {error}",
            )
        if (
            stat.S_ISLNK(publication_stat.st_mode)
            or not stat.S_ISREG(publication_stat.st_mode)
            or publication_stat.st_mode & 0o222
            or publication_stat.st_nlink != 2
            or (marker_stat.st_dev, marker_stat.st_ino)
            != (publication_stat.st_dev, publication_stat.st_ino)
        ):
            _check_fail(
                ArtifactIntegrityError,
                RuleCode.MANIFEST_MISMATCH,
                "snapshot publication receipt is not an immutable two-link file "
                "linked to its frozen marker",
            )

        def signature(info: os.stat_result) -> tuple[int, int, int, int, int]:
            return (
                info.st_dev,
                info.st_ino,
                info.st_mode,
                info.st_nlink,
                info.st_mtime_ns,
            )

        return signature(publication_stat), signature(marker_stat)

    def _load_snapshot_inner(
        self, snapshot: FrozenSnapshot, purpose: AccessPurpose
    ) -> tuple[TemporalDataManifest, tuple[ValidatedArtifact, ...]]:
        if not isinstance(snapshot, FrozenSnapshot):
            _check_fail(
                TemporalSchemaError,
                RuleCode.INVALID_SCHEMA,
                "snapshot must be a FrozenSnapshot",
            )
        self._check_purpose_context(purpose)
        if snapshot.policy_sha256 != self._policy.sha256:
            _check_fail(
                ArtifactIntegrityError,
                RuleCode.MANIFEST_MISMATCH,
                "snapshot handle belongs to a different cutoff policy",
            )
        root = snapshot.root
        try:
            root_stat = os.lstat(root)
        except OSError as error:
            _check_fail(
                ArtifactIntegrityError,
                RuleCode.PATH_VIOLATION,
                f"cannot inspect snapshot root: {error}",
            )
        if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
            _check_fail(
                ArtifactIntegrityError,
                RuleCode.PATH_VIOLATION,
                "snapshot root must be a regular directory, not a symlink",
            )
        if root_stat.st_mode & 0o222:
            _check_fail(
                FrozenSnapshotError,
                RuleCode.FROZEN_MUTATION,
                "snapshot root has writable permission bits",
            )
        publication_signature = self._snapshot_publication_signature(
            root, snapshot.publication_token
        )
        manifest_bytes = self._read_regular_file(
            root, _MANIFEST_FILE, require_read_only=True
        )
        seal_bytes = self._read_regular_file(
            root, _MANIFEST_SEAL_FILE, require_read_only=True
        )
        marker_bytes = self._read_regular_file(
            root, _FROZEN_MARKER_FILE, require_read_only=True
        )
        try:
            manifest_body = _parse_json_object_bytes(manifest_bytes, _MANIFEST_FILE)
            manifest = TemporalDataManifest.from_mapping(manifest_body)
        except TemporalSchemaError as error:
            _check_fail(
                ArtifactIntegrityError,
                RuleCode.MANIFEST_MISMATCH,
                f"snapshot manifest schema is invalid: {error}",
            )
        observed_sha256 = sha256_bytes(manifest_bytes)
        expected_seal = f"{observed_sha256}  {_MANIFEST_FILE}\n".encode()
        if seal_bytes != expected_seal or marker_bytes != expected_seal:
            _check_fail(
                ArtifactIntegrityError,
                RuleCode.MANIFEST_MISMATCH,
                "snapshot manifest seal or frozen marker is invalid",
            )
        if observed_sha256 != snapshot.manifest_sha256:
            _check_fail(
                ArtifactIntegrityError,
                RuleCode.MANIFEST_MISMATCH,
                "snapshot manifest does not match the trusted commitment",
            )
        if (
            manifest.snapshot_id != snapshot.snapshot_id
            or manifest.policy_sha256 != self._policy.sha256
        ):
            _check_fail(
                ArtifactIntegrityError,
                RuleCode.MANIFEST_MISMATCH,
                "snapshot identity or cutoff policy commitment changed",
            )
        audit_events = self._audit.events()
        token_sha256 = sha256_bytes(snapshot.publication_token.encode("ascii"))
        matching_publication_events = tuple(
            event
            for event in audit_events
            if event["action"] == "finalize_snapshot_publication"
            and event["artifact_id"] == snapshot.snapshot_id
            and event["outcome"] == "allowed"
            and event["rule_code"] == RuleCode.OK.value
            and event["policy_sha256"] == self._policy.sha256
            and cast(Mapping[str, object], event["details"]).get("manifest_sha256")
            == observed_sha256
            and cast(Mapping[str, object], event["details"]).get(
                "publication_token_sha256"
            )
            == token_sha256
        )
        if not matching_publication_events:
            _check_fail(
                ArtifactIntegrityError,
                RuleCode.MANIFEST_MISMATCH,
                "snapshot publication receipt has no matching successful audit event",
            )
        latest_authorization = max(
            cast(int, event["sequence"]) for event in matching_publication_events
        )
        if any(
            event["action"] == "revoke_snapshot_publication"
            and event["artifact_id"] == snapshot.snapshot_id
            and event["outcome"] == "rejected"
            and event["policy_sha256"] == self._policy.sha256
            and cast(int, event["sequence"]) > latest_authorization
            and cast(Mapping[str, object], event["details"]).get("manifest_sha256")
            == observed_sha256
            and cast(Mapping[str, object], event["details"]).get(
                "publication_token_sha256"
            )
            == token_sha256
            for event in audit_events
        ):
            _check_fail(
                ArtifactIntegrityError,
                RuleCode.MANIFEST_MISMATCH,
                "snapshot publication authorization was revoked after mutation",
            )

        expected_files = set(_RESERVED_SNAPSHOT_PATHS)
        expected_directories: set[str] = set()
        for entry in manifest.entries:
            expected_files.add(entry.artifact.relative_path)
            parent = PurePosixPath(entry.artifact.relative_path).parent
            while parent != PurePosixPath("."):
                expected_directories.add(parent.as_posix())
                parent = parent.parent
        tree_signature = self._snapshot_tree_signature(
            root, expected_files, expected_directories
        )

        for entry in manifest.entries:
            self._check_artifact(entry.artifact)
            self._check_access(entry.artifact.record.partition, purpose)
        loaded: list[ValidatedArtifact] = []
        for entry in manifest.entries:
            payload = self._read_regular_file(
                root, entry.artifact.relative_path, require_read_only=True
            )
            if len(payload) != entry.size_bytes:
                _check_fail(
                    ArtifactIntegrityError,
                    RuleCode.HASH_MISMATCH,
                    "frozen artifact size changed",
                    {"relative_path": entry.artifact.relative_path},
                )
            observed = sha256_bytes(payload)
            if observed != entry.artifact.record.content_hash:
                _check_fail(
                    ArtifactIntegrityError,
                    RuleCode.HASH_MISMATCH,
                    "frozen artifact content hash changed",
                    {
                        "relative_path": entry.artifact.relative_path,
                        "expected_sha256": entry.artifact.record.content_hash,
                        "observed_sha256": observed,
                    },
                )
            if entry.artifact.kind is ArtifactKind.TABLE:
                self._check_table_payload(payload, purpose, entry.artifact.record)
            if entry.artifact.kind is ArtifactKind.TEXT:
                try:
                    payload.decode("utf-8")
                except UnicodeError as error:
                    _check_fail(
                        ArtifactIntegrityError,
                        RuleCode.INVALID_SCHEMA,
                        f"frozen text artifact is not UTF-8: {error}",
                    )
            loaded.append(ValidatedArtifact(artifact=entry.artifact, payload=payload))
        if tree_signature != self._snapshot_tree_signature(
            root, expected_files, expected_directories
        ) or publication_signature != self._snapshot_publication_signature(
            root, snapshot.publication_token
        ):
            _check_fail(
                ArtifactIntegrityError,
                RuleCode.FROZEN_MUTATION,
                "frozen snapshot changed during verification",
            )
        return manifest, tuple(loaded)

    def verify_snapshot(
        self, snapshot: FrozenSnapshot, purpose: AccessPurpose
    ) -> TemporalDataManifest:
        """Verify a trusted snapshot commitment without returning payload bytes."""
        artifact_id = (
            snapshot.snapshot_id if isinstance(snapshot, FrozenSnapshot) else None
        )
        purpose_label = (
            purpose.value if isinstance(purpose, AccessPurpose) else "invalid"
        )
        manifest, _ = self._run_check(
            action="verify_snapshot",
            check=lambda: self._load_snapshot_inner(snapshot, purpose),
            artifact_id=artifact_id,
            success_details={"purpose": purpose_label},
        )
        return manifest

    def load_snapshot(
        self, snapshot: FrozenSnapshot, purpose: AccessPurpose
    ) -> tuple[ValidatedArtifact, ...]:
        """Atomically validate and load every artifact; never filter a snapshot."""
        artifact_id = (
            snapshot.snapshot_id if isinstance(snapshot, FrozenSnapshot) else None
        )
        purpose_label = (
            purpose.value if isinstance(purpose, AccessPurpose) else "invalid"
        )
        _, artifacts = self._run_check(
            action="load_snapshot",
            check=lambda: self._load_snapshot_inner(snapshot, purpose),
            artifact_id=artifact_id,
            success_details={"purpose": purpose_label},
        )
        return artifacts

    def refuse_in_place_mutation(
        self,
        snapshot: FrozenSnapshot,
        relative_path: str,
        *,
        operation: str,
    ) -> NoReturn:
        """Explicitly reject write/delete APIs against a frozen snapshot."""
        artifact_id = (
            snapshot.snapshot_id if isinstance(snapshot, FrozenSnapshot) else None
        )

        def check() -> NoReturn:
            if not isinstance(snapshot, FrozenSnapshot):
                _check_fail(
                    TemporalSchemaError,
                    RuleCode.INVALID_SCHEMA,
                    "snapshot must be a FrozenSnapshot",
                )
            try:
                safe_path = _validate_relative_path(relative_path)
                stable_operation = _validate_text(operation, "operation", maximum=64)
            except TemporalSchemaError as error:
                _check_fail(
                    TemporalSchemaError,
                    RuleCode.INVALID_SCHEMA,
                    f"invalid frozen mutation request: {error}",
                )
            _check_fail(
                FrozenSnapshotError,
                RuleCode.FROZEN_MUTATION,
                "frozen snapshots cannot be modified in place",
                {"relative_path": safe_path, "operation": stable_operation},
            )

        self._run_check(
            action="refuse_snapshot_mutation",
            check=check,
            artifact_id=artifact_id,
        )
        raise AssertionError("unreachable")

    def replace_snapshot_artifact(
        self, snapshot: FrozenSnapshot, relative_path: str, payload: bytes
    ) -> NoReturn:
        """Reject replacement of an artifact in a frozen snapshot."""
        del payload
        self.refuse_in_place_mutation(snapshot, relative_path, operation="replace")

    def delete_snapshot_artifact(
        self, snapshot: FrozenSnapshot, relative_path: str
    ) -> NoReturn:
        """Reject deletion of an artifact in a frozen snapshot."""
        self.refuse_in_place_mutation(snapshot, relative_path, operation="delete")

    def _deny_network(self, operation: str) -> NoReturn:
        message = f"network-capable operation blocked: {operation}"
        with self._audit._decision_lock:
            self._refresh_latched_failure()
            self._latched_failure = (RuleCode.NETWORK_DENIED, "network_access")
            self._audit.record(
                action="network_access",
                outcome="rejected",
                rule_code=RuleCode.NETWORK_DENIED,
                policy_sha256=self._policy.sha256,
                cutoff=self._policy.cutoff.isoformat(),
                artifact_id=None,
                record_id=None,
                partition=DataPartition.POST_POLICY_EVALUATION.value,
                details={"message": message, "operation": operation},
            )
        raise NetworkAccessBlockedError(message)

    def _network_blocker(self, operation: str) -> Callable[..., NoReturn]:
        def blocked(*args: object, **kwargs: object) -> NoReturn:
            del args, kwargs
            self._deny_network(operation)

        return blocked

    def frozen_evaluation(self) -> AbstractContextManager[TemporalFirewall]:
        """Enter a Python-level no-network context for sealed evaluation."""
        return _FrozenEvaluationContext(self)


class _FrozenEvaluationContext(AbstractContextManager[TemporalFirewall]):
    def __init__(self, firewall: TemporalFirewall) -> None:
        self._firewall = firewall
        self._stack: ExitStack | None = None
        self._token: Token[object | None] | None = None
        self._lock_acquired = False

    def __enter__(self) -> TemporalFirewall:
        _NETWORK_CONTEXT_LOCK.acquire()
        self._lock_acquired = True
        stack = ExitStack()
        token = _EVALUATION_FIREWALL.set(self._firewall)
        _ACTIVE_NETWORK_FIREWALLS.append(self._firewall)
        targets: tuple[tuple[object, str], ...] = (
            (socket, "socket"),
            (socket, "socketpair"),
            (socket, "create_connection"),
            (socket, "create_server"),
            (socket, "getaddrinfo"),
            (socket, "gethostbyname"),
            (socket, "gethostbyname_ex"),
            (socket, "gethostbyaddr"),
            (subprocess, "Popen"),
            (os, "system"),
        )
        try:
            open_socket_descriptors = _open_socket_file_descriptors()
            if open_socket_descriptors:
                self._firewall._deny_network("preexisting_open_socket")
            for target, attribute in targets:
                if hasattr(target, attribute):
                    stack.enter_context(
                        patch.object(
                            target,
                            attribute,
                            self._firewall._network_blocker(
                                f"{getattr(target, '__name__', type(target).__name__)}."
                                f"{attribute}"
                            ),
                        )
                    )
            self._firewall.audit_log.record(
                action="enter_frozen_evaluation",
                outcome="allowed",
                rule_code=RuleCode.OK,
                policy_sha256=self._firewall.policy.sha256,
                cutoff=self._firewall.policy.cutoff.isoformat(),
                artifact_id=None,
                record_id=None,
                partition=DataPartition.POST_POLICY_EVALUATION.value,
                details={"network_mode": "python_process_blocked"},
            )
        except BaseException as error:
            stack.close()
            _EVALUATION_FIREWALL.reset(token)
            active = _ACTIVE_NETWORK_FIREWALLS.pop()
            _NETWORK_CONTEXT_LOCK.release()
            self._lock_acquired = False
            if active is not self._firewall:
                raise TemporalFirewallError(
                    "frozen evaluation network guard stack is corrupted"
                ) from error
            raise
        self._stack = stack
        self._token = token
        return self._firewall

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        if self._stack is None or self._token is None:
            raise TemporalFirewallError("frozen evaluation context was not entered")
        self._stack.close()
        _EVALUATION_FIREWALL.reset(self._token)
        self._stack = None
        self._token = None
        try:
            self._firewall.audit_log.record(
                action="exit_frozen_evaluation",
                outcome="allowed",
                rule_code=RuleCode.OK,
                policy_sha256=self._firewall.policy.sha256,
                cutoff=self._firewall.policy.cutoff.isoformat(),
                artifact_id=None,
                record_id=None,
                partition=DataPartition.POST_POLICY_EVALUATION.value,
                details={"network_mode": "restored"},
            )
        finally:
            active = _ACTIVE_NETWORK_FIREWALLS.pop()
            if self._lock_acquired:
                _NETWORK_CONTEXT_LOCK.release()
                self._lock_acquired = False
            if active is not self._firewall:
                raise TemporalFirewallError(
                    "frozen evaluation network guard stack is corrupted"
                )


def make_post_cutoff_canary(
    policy: CutoffPolicy,
    payload: bytes,
    *,
    license_tag: str | None = None,
) -> TemporalRecord:
    """Build a sentinel that every compliant firewall must reject before load."""
    if not isinstance(policy, CutoffPolicy):
        _schema_fail("policy must be a CutoffPolicy")
    if not isinstance(payload, bytes):
        _schema_fail("canary payload must be immutable bytes")
    selected_license = (
        policy.allowed_license_tags[0] if license_tag is None else license_tag
    )
    try:
        post_cutoff = policy.cutoff + timedelta(microseconds=1)
    except OverflowError as error:
        raise TemporalSchemaError("cutoff cannot represent a canary instant") from error
    return TemporalRecord(
        record_id="post-cutoff-canary",
        partition=DataPartition.TRAIN,
        event_time=policy.training_end,
        available_time=post_cutoff,
        ingestion_time=post_cutoff,
        source=POST_CUTOFF_CANARY_SOURCE,
        version="canary-v1",
        content_hash=sha256_bytes(payload),
        license_tag=selected_license,
    )


def _temporal_python_audit_hook(event: str, args: tuple[object, ...]) -> None:
    del args
    active = _ACTIVE_NETWORK_FIREWALLS[-1] if _ACTIVE_NETWORK_FIREWALLS else None
    if event in _NETWORK_AUDIT_EVENTS and isinstance(active, TemporalFirewall):
        active._deny_network(f"python_audit:{event}")


sys.addaudithook(_temporal_python_audit_hook)


__all__ = [
    "AccessPurpose",
    "ArtifactIntegrityError",
    "ArtifactKind",
    "AuditLogError",
    "CutoffPolicy",
    "DataPartition",
    "ForwardFillSpec",
    "FrozenSnapshot",
    "FrozenSnapshotError",
    "LabelConstructionSpec",
    "LineageReference",
    "LineageRole",
    "ManifestEntry",
    "NetworkAccessBlockedError",
    "POST_CUTOFF_CANARY_SOURCE",
    "PartitionAccessError",
    "PostCutoffDataError",
    "RuleCode",
    "StandardizationSpec",
    "TableScalar",
    "TemporalArtifact",
    "TemporalAuditLog",
    "TemporalDataManifest",
    "TemporalFirewall",
    "TemporalFirewallError",
    "TemporalLeakageError",
    "TemporalRecord",
    "TemporalSchemaError",
    "TemporalTableRow",
    "TransformationLeakageError",
    "ValidatedArtifact",
    "WindowAggregationSpec",
    "decode_temporal_table",
    "encode_temporal_table",
    "lineage_references_sha256",
    "make_post_cutoff_canary",
    "table_values_sha256",
    "temporal_records_sha256",
]
