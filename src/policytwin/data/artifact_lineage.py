"""Content-addressed artifact lineage and policy-post taint capabilities.

This module closes an API-level laundering path that record-local temporal
checks cannot close by themselves.  Every non-source artifact names complete
parents by both ID and content hash.  Taint propagates through parent edges and
across byte-identical aliases, so copying or renaming a policy-post payload does
not make it admissible for calibration or model selection.

The implementation is deliberately synthetic-only.  It proves the trust-root
mechanics without claiming that an untrusted producer disclosed hidden parents.
"""

from __future__ import annotations

import os
import re
import stat
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import NoReturn, Self, cast

from policytwin.data.temporal_firewall import (
    DataPartition,
    RuleCode,
    TemporalAuditLog,
    TemporalFirewallError,
    TemporalRecord,
)
from policytwin.provenance import (
    MetadataValidationError,
    canonical_json_bytes,
    read_json_bytes_object,
    sha256_bytes,
    sha256_file,
    write_json_file,
)

ARTIFACT_LINEAGE_SCHEMA_VERSION = "policytwin.artifact-lineage.v1"

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class LineageRuleCode(StrEnum):
    """Stable artifact-lineage validation and authorization outcomes."""

    OK = "OK"
    INVALID_SCHEMA = "INVALID_SCHEMA"
    REGISTRY_HASH_MISMATCH = "REGISTRY_HASH_MISMATCH"
    REGISTRY_COMMITMENT_MISMATCH = "REGISTRY_COMMITMENT_MISMATCH"
    PATH_VIOLATION = "PATH_VIOLATION"
    ARTIFACT_HASH_MISMATCH = "ARTIFACT_HASH_MISMATCH"
    MISSING_PARENT = "MISSING_PARENT"
    PARENT_HASH_MISMATCH = "PARENT_HASH_MISMATCH"
    LINEAGE_CYCLE = "LINEAGE_CYCLE"
    PARTITION_DOWNGRADE = "PARTITION_DOWNGRADE"
    LICENSE_DENIED = "LICENSE_DENIED"
    TAINTED_LINEAGE = "TAINTED_LINEAGE"
    PURPOSE_DENIED = "PURPOSE_DENIED"
    CAPABILITY_INVALID = "CAPABILITY_INVALID"


class ArtifactLineageError(RuntimeError):
    """Base class carrying a stable machine-readable lineage rule."""

    def __init__(self, message: str, rule_code: LineageRuleCode) -> None:
        super().__init__(message)
        self.rule_code = rule_code


class ArtifactLineageSchemaError(ArtifactLineageError, ValueError):
    """Raised when a registry or node has an invalid immutable schema."""


class ArtifactRegistryIntegrityError(ArtifactLineageError):
    """Raised when registry or payload bytes differ from commitments."""


class ArtifactUseDeniedError(ArtifactLineageError):
    """Raised when an artifact is inadmissible for the requested purpose."""


class ArtifactCapabilityError(ArtifactLineageError):
    """Raised for forged, reused, serialized, or purpose-confused authority."""


class LineageArtifactKind(StrEnum):
    """Closed artifact roles used to derive temporal taint."""

    RAW_SOURCE = "raw_source"
    DERIVED_DATA = "derived_data"
    MODEL_CHECKPOINT = "model_checkpoint"
    PREDICTION = "prediction"
    EVALUATION_INPUT = "evaluation_input"
    EVALUATION_RESULT = "evaluation_result"
    REPORT = "report"


class ArtifactUsePurpose(StrEnum):
    """Closed downstream purposes for opaque use authorization."""

    CALIBRATION = "calibration"
    MODEL_SELECTION = "model_selection"
    HELD_OUT_VALIDATION = "held_out_validation"
    FROZEN_EVALUATION = "frozen_evaluation"
    REPORTING = "reporting"
    REPRODUCIBILITY_AUDIT = "reproducibility_audit"


class LineageTaint(StrEnum):
    """Facts that prevent pre-policy model development use."""

    POLICY_POST_PARTITION = "policy_post_partition"
    AVAILABLE_AFTER_KNOWLEDGE_CUTOFF = "available_after_knowledge_cutoff"
    EVALUATION_DERIVED = "evaluation_derived"


_PRE_POLICY_PURPOSES = frozenset(
    {
        ArtifactUsePurpose.CALIBRATION,
        ArtifactUsePurpose.MODEL_SELECTION,
        ArtifactUsePurpose.HELD_OUT_VALIDATION,
    }
)


def _raise[T: ArtifactLineageError](
    error_type: type[T], rule_code: LineageRuleCode, message: str
) -> NoReturn:
    raise error_type(message, rule_code)


def _schema(message: str) -> NoReturn:
    _raise(ArtifactLineageSchemaError, LineageRuleCode.INVALID_SCHEMA, message)


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        _schema(f"{field} must be a stable ASCII identifier")
    return value


def _digest(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _schema(f"{field} must be a lowercase SHA-256 digest")
    return value


def _aware(value: object, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        _schema(f"{field} must be a timezone-aware datetime")
    try:
        offset = value.utcoffset()
        converted = value.astimezone(UTC)
    except (OverflowError, ValueError) as error:
        raise ArtifactLineageSchemaError(
            f"{field} is outside the UTC range", LineageRuleCode.INVALID_SCHEMA
        ) from error
    if offset is None:
        _schema(f"{field} must be a timezone-aware datetime")
    return converted


def _parse_time(value: object, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        _schema(f"{field} must be an ISO 8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ArtifactLineageSchemaError(
            f"{field} must be ISO 8601", LineageRuleCode.INVALID_SCHEMA
        ) from error
    return _aware(parsed, field)


def _fields(
    mapping: Mapping[str, object], expected: frozenset[str], context: str
) -> None:
    actual = frozenset(mapping)
    if actual != expected:
        _schema(
            f"{context} fields differ; missing={sorted(expected - actual)!r}, "
            f"unknown={sorted(actual - expected)!r}"
        )


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        _schema(f"{field} must be an object with string keys")
    return cast(Mapping[str, object], value)


def _array(value: object, field: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        _schema(f"{field} must be an array")
    return cast(Sequence[object], value)


def _enum[T: StrEnum](value: object, enum_type: type[T], field: str) -> T:
    if not isinstance(value, str):
        _schema(f"{field} must be a string")
    try:
        return enum_type(value)
    except ValueError as error:
        raise ArtifactLineageSchemaError(
            f"unsupported {field}: {value}", LineageRuleCode.INVALID_SCHEMA
        ) from error


def _relative_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "://" in value:
        _schema("relative_path must be a portable non-empty relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        _schema("relative_path must not escape the artifact root")
    return value


def _nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _schema(f"{field} must be a non-negative integer")
    return value


def _partition_rank(partition: DataPartition) -> int:
    return {
        DataPartition.TRAIN: 0,
        DataPartition.VALIDATION: 1,
        DataPartition.POST_POLICY_EVALUATION: 2,
    }[partition]


@dataclass(frozen=True, slots=True)
class ArtifactParentReference:
    """Parent identity committed by stable ID and exact payload hash."""

    artifact_id: str
    content_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.artifact_id, "parent artifact_id")
        _digest(self.content_sha256, "parent content_sha256")

    def to_mapping(self) -> dict[str, object]:
        return {
            "artifact_id": self.artifact_id,
            "content_sha256": self.content_sha256,
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> Self:
        _fields(
            mapping,
            frozenset({"artifact_id", "content_sha256"}),
            "artifact parent reference",
        )
        return cls(
            artifact_id=_identifier(mapping["artifact_id"], "parent artifact_id"),
            content_sha256=_digest(mapping["content_sha256"], "parent content_sha256"),
        )


@dataclass(frozen=True, slots=True)
class ArtifactLineageNode:
    """One content-addressed artifact and its complete immediate parents."""

    relative_path: str
    kind: LineageArtifactKind
    record: TemporalRecord
    size_bytes: int
    parents: tuple[ArtifactParentReference, ...] = ()

    def __post_init__(self) -> None:
        _relative_path(self.relative_path)
        if not isinstance(self.kind, LineageArtifactKind):
            _schema("artifact kind must be a LineageArtifactKind")
        if not isinstance(self.record, TemporalRecord):
            _schema("artifact record must be a TemporalRecord")
        _nonnegative_int(self.size_bytes, "size_bytes")
        if not isinstance(self.parents, tuple) or not all(
            type(parent) is ArtifactParentReference for parent in self.parents
        ):
            _schema("parents must be an immutable ArtifactParentReference tuple")
        ordered = tuple(sorted(self.parents, key=lambda parent: parent.artifact_id))
        if len({parent.artifact_id for parent in ordered}) != len(ordered):
            _schema("artifact parent IDs must be unique")
        if any(parent.artifact_id == self.artifact_id for parent in ordered):
            _schema("artifact cannot directly reference itself")
        object.__setattr__(self, "parents", ordered)
        if self.kind is LineageArtifactKind.RAW_SOURCE:
            if ordered:
                _schema("raw source artifacts must not claim derived parents")
        elif not ordered:
            _schema("every non-source artifact must declare at least one parent")
        if (
            self.kind is LineageArtifactKind.EVALUATION_RESULT
            and self.record.partition is not DataPartition.POST_POLICY_EVALUATION
        ):
            _schema("evaluation results must remain in the policy-post partition")

    @property
    def artifact_id(self) -> str:
        return self.record.record_id

    @property
    def content_sha256(self) -> str:
        return self.record.content_hash

    def to_mapping(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "parents": [parent.to_mapping() for parent in self.parents],
            "record": self.record.to_mapping(),
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> Self:
        _fields(
            mapping,
            frozenset({"relative_path", "kind", "record", "size_bytes", "parents"}),
            "artifact lineage node",
        )
        try:
            record = TemporalRecord.from_mapping(_mapping(mapping["record"], "record"))
        except TemporalFirewallError as error:
            raise ArtifactLineageSchemaError(
                f"invalid temporal record: {error}", LineageRuleCode.INVALID_SCHEMA
            ) from error
        return cls(
            relative_path=_relative_path(mapping["relative_path"]),
            kind=_enum(mapping["kind"], LineageArtifactKind, "artifact kind"),
            record=record,
            size_bytes=_nonnegative_int(mapping["size_bytes"], "size_bytes"),
            parents=tuple(
                ArtifactParentReference.from_mapping(_mapping(item, "parent"))
                for item in _array(mapping["parents"], "parents")
            ),
        )


@dataclass(frozen=True, slots=True)
class ArtifactLineageRegistry:
    """Immutable DAG plus content-alias taint propagation contract."""

    schema_version: str
    registry_id: str
    protocol_sha256: str
    cutoff_policy_sha256: str
    knowledge_cutoff: datetime
    created_at: datetime
    allowed_license_tags: tuple[str, ...]
    artifacts: tuple[ArtifactLineageNode, ...]
    synthetic_only: bool = True

    def __post_init__(self) -> None:
        if self.schema_version != ARTIFACT_LINEAGE_SCHEMA_VERSION:
            _schema("unsupported artifact lineage schema version")
        _identifier(self.registry_id, "registry_id")
        _digest(self.protocol_sha256, "protocol_sha256")
        _digest(self.cutoff_policy_sha256, "cutoff_policy_sha256")
        object.__setattr__(
            self,
            "knowledge_cutoff",
            _aware(self.knowledge_cutoff, "knowledge_cutoff"),
        )
        object.__setattr__(self, "created_at", _aware(self.created_at, "created_at"))
        if self.synthetic_only is not True:
            _schema("the implemented lineage trust root is synthetic-only")
        if (
            not isinstance(self.allowed_license_tags, tuple)
            or not self.allowed_license_tags
        ):
            _schema("allowed_license_tags must be a non-empty immutable tuple")
        tags = tuple(
            sorted(_identifier(tag, "license tag") for tag in self.allowed_license_tags)
        )
        if len(set(tags)) != len(tags):
            _schema("allowed_license_tags must be unique")
        object.__setattr__(self, "allowed_license_tags", tags)
        if not isinstance(self.artifacts, tuple) or not self.artifacts:
            _schema("artifacts must be a non-empty immutable tuple")
        if not all(type(node) is ArtifactLineageNode for node in self.artifacts):
            _schema("artifacts contain an invalid lineage node")
        ordered = tuple(sorted(self.artifacts, key=lambda node: node.artifact_id))
        ids = tuple(node.artifact_id for node in ordered)
        paths = tuple(node.relative_path for node in ordered)
        if len(set(ids)) != len(ids):
            _schema("artifact IDs must be unique")
        if len(set(paths)) != len(paths):
            _schema("artifact paths must be unique")
        if any(node.record.license_tag not in tags for node in ordered):
            _raise(
                ArtifactLineageSchemaError,
                LineageRuleCode.LICENSE_DENIED,
                "artifact license is absent from the frozen allowed-license set",
            )
        if any(node.record.ingestion_time > self.created_at for node in ordered):
            _schema("registry cannot predate an artifact ingestion time")
        object.__setattr__(self, "artifacts", ordered)
        self._validate_graph()

    def _validate_graph(self) -> None:
        by_id = {node.artifact_id: node for node in self.artifacts}
        hash_sizes: dict[str, int] = {}
        for node in self.artifacts:
            known_size = hash_sizes.setdefault(node.content_sha256, node.size_bytes)
            if known_size != node.size_bytes:
                _schema("byte-identical artifact aliases disagree on size")
            for reference in node.parents:
                parent = by_id.get(reference.artifact_id)
                if parent is None:
                    _raise(
                        ArtifactLineageSchemaError,
                        LineageRuleCode.MISSING_PARENT,
                        f"artifact {node.artifact_id} references a missing parent: "
                        f"{reference.artifact_id}",
                    )
                if parent.content_sha256 != reference.content_sha256:
                    _raise(
                        ArtifactLineageSchemaError,
                        LineageRuleCode.PARENT_HASH_MISMATCH,
                        f"parent hash commitment changed for {reference.artifact_id}",
                    )
                if _partition_rank(parent.record.partition) > _partition_rank(
                    node.record.partition
                ):
                    _raise(
                        ArtifactLineageSchemaError,
                        LineageRuleCode.PARTITION_DOWNGRADE,
                        f"artifact {node.artifact_id} declares an earlier partition "
                        "than its parent",
                    )
                if (
                    parent.record.event_time > node.record.event_time
                    or parent.record.available_time > node.record.available_time
                    or parent.record.ingestion_time > node.record.ingestion_time
                ):
                    _schema(f"artifact {node.artifact_id} predates one of its parents")

        state: dict[str, int] = {}

        def visit(artifact_id: str) -> None:
            mark = state.get(artifact_id, 0)
            if mark == 1:
                _raise(
                    ArtifactLineageSchemaError,
                    LineageRuleCode.LINEAGE_CYCLE,
                    f"lineage cycle detected at artifact {artifact_id}",
                )
            if mark == 2:
                return
            state[artifact_id] = 1
            for parent in by_id[artifact_id].parents:
                visit(parent.artifact_id)
            state[artifact_id] = 2

        for artifact_id in by_id:
            visit(artifact_id)

    def artifact(self, artifact_id: str) -> ArtifactLineageNode:
        stable = _identifier(artifact_id, "artifact_id")
        for node in self.artifacts:
            if node.artifact_id == stable:
                return node
        _raise(
            ArtifactUseDeniedError,
            LineageRuleCode.PURPOSE_DENIED,
            f"artifact is not present in the frozen registry: {stable}",
        )

    def _direct_taints(self, node: ArtifactLineageNode) -> set[LineageTaint]:
        taints: set[LineageTaint] = set()
        if node.record.partition is DataPartition.POST_POLICY_EVALUATION:
            taints.add(LineageTaint.POLICY_POST_PARTITION)
        if node.record.available_time > self.knowledge_cutoff:
            taints.add(LineageTaint.AVAILABLE_AFTER_KNOWLEDGE_CUTOFF)
        if node.kind in {
            LineageArtifactKind.EVALUATION_INPUT,
            LineageArtifactKind.EVALUATION_RESULT,
            LineageArtifactKind.REPORT,
        }:
            taints.add(LineageTaint.EVALUATION_DERIVED)
        return taints

    def _taint_map(self) -> dict[str, set[LineageTaint]]:
        by_id = {node.artifact_id: node for node in self.artifacts}
        taints = {
            node.artifact_id: self._direct_taints(node) for node in self.artifacts
        }
        aliases: dict[str, tuple[str, ...]] = {
            digest: tuple(node.artifact_id for node in nodes)
            for digest, nodes in self._hash_groups().items()
        }
        changed = True
        while changed:
            changed = False
            for node in self.artifacts:
                combined = set(taints[node.artifact_id])
                for parent in node.parents:
                    combined.update(taints[parent.artifact_id])
                for alias_id in aliases[node.content_sha256]:
                    combined.update(taints[alias_id])
                if combined != taints[node.artifact_id]:
                    taints[node.artifact_id] = combined
                    changed = True
            # A taint learned by a later alias in the loop must flow back to all
            # aliases before the fixed point is accepted.
            for alias_ids in aliases.values():
                union = set().union(*(taints[item] for item in alias_ids))
                for artifact_id in alias_ids:
                    if union != taints[artifact_id]:
                        taints[artifact_id] = set(union)
                        changed = True
        if set(taints) != set(by_id):
            _schema("internal taint map does not cover the registry")
        return taints

    def _hash_groups(self) -> dict[str, tuple[ArtifactLineageNode, ...]]:
        grouped: dict[str, list[ArtifactLineageNode]] = defaultdict(list)
        for node in self.artifacts:
            grouped[node.content_sha256].append(node)
        return {
            digest: tuple(sorted(nodes, key=lambda node: node.artifact_id))
            for digest, nodes in grouped.items()
        }

    def effective_taints(self, artifact_id: str) -> tuple[LineageTaint, ...]:
        """Return deterministic transitive and content-alias taints."""
        node = self.artifact(artifact_id)
        return tuple(sorted(self._taint_map()[node.artifact_id], key=str))

    def lineage_artifact_ids(self, artifact_id: str) -> tuple[str, ...]:
        """Return parents and byte aliases that can transmit information."""
        start = self.artifact(artifact_id)
        by_id = {node.artifact_id: node for node in self.artifacts}
        hash_groups = self._hash_groups()
        pending = [start.artifact_id]
        found: set[str] = set()
        while pending:
            current_id = pending.pop()
            if current_id in found:
                continue
            found.add(current_id)
            current = by_id[current_id]
            pending.extend(parent.artifact_id for parent in current.parents)
            pending.extend(
                alias.artifact_id for alias in hash_groups[current.content_sha256]
            )
        return tuple(sorted(found))

    def lineage_sha256(self, artifact_id: str) -> str:
        ids = set(self.lineage_artifact_ids(artifact_id))
        mappings = [
            node.to_mapping() for node in self.artifacts if node.artifact_id in ids
        ]
        return sha256_bytes(canonical_json_bytes(mappings))

    def to_mapping(self) -> dict[str, object]:
        return {
            "allowed_license_tags": list(self.allowed_license_tags),
            "artifacts": [node.to_mapping() for node in self.artifacts],
            "created_at": self.created_at.isoformat(),
            "cutoff_policy_sha256": self.cutoff_policy_sha256,
            "knowledge_cutoff": self.knowledge_cutoff.isoformat(),
            "protocol_sha256": self.protocol_sha256,
            "registry_id": self.registry_id,
            "schema_version": self.schema_version,
            "synthetic_only": self.synthetic_only,
        }

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> Self:
        _fields(
            mapping,
            frozenset(
                {
                    "allowed_license_tags",
                    "artifacts",
                    "created_at",
                    "cutoff_policy_sha256",
                    "knowledge_cutoff",
                    "protocol_sha256",
                    "registry_id",
                    "schema_version",
                    "synthetic_only",
                }
            ),
            "artifact lineage registry",
        )
        synthetic = mapping["synthetic_only"]
        if not isinstance(synthetic, bool):
            _schema("synthetic_only must be boolean")
        return cls(
            schema_version=cast(str, mapping["schema_version"]),
            registry_id=_identifier(mapping["registry_id"], "registry_id"),
            protocol_sha256=_digest(mapping["protocol_sha256"], "protocol_sha256"),
            cutoff_policy_sha256=_digest(
                mapping["cutoff_policy_sha256"], "cutoff_policy_sha256"
            ),
            knowledge_cutoff=_parse_time(
                mapping["knowledge_cutoff"], "knowledge_cutoff"
            ),
            created_at=_parse_time(mapping["created_at"], "created_at"),
            allowed_license_tags=tuple(
                _identifier(item, "license tag")
                for item in _array(
                    mapping["allowed_license_tags"], "allowed_license_tags"
                )
            ),
            artifacts=tuple(
                ArtifactLineageNode.from_mapping(_mapping(item, "artifact node"))
                for item in _array(mapping["artifacts"], "artifacts")
            ),
            synthetic_only=synthetic,
        )


@dataclass(frozen=True, slots=True)
class ArtifactUseReceipt:
    """Serializable evidence of an admissibility decision, not authority."""

    registry_sha256: str
    registry_source_sha256: str
    protocol_sha256: str
    artifact_id: str
    artifact_sha256: str
    lineage_sha256: str
    purpose: ArtifactUsePurpose
    effective_taints: tuple[LineageTaint, ...]

    def __post_init__(self) -> None:
        for field in (
            "registry_sha256",
            "registry_source_sha256",
            "protocol_sha256",
            "artifact_sha256",
            "lineage_sha256",
        ):
            _digest(getattr(self, field), field)
        _identifier(self.artifact_id, "artifact_id")
        if not isinstance(self.purpose, ArtifactUsePurpose):
            _schema("receipt purpose must be ArtifactUsePurpose")
        if not isinstance(self.effective_taints, tuple) or not all(
            isinstance(taint, LineageTaint) for taint in self.effective_taints
        ):
            _schema("receipt taints must be an immutable LineageTaint tuple")
        if tuple(sorted(set(self.effective_taints), key=str)) != self.effective_taints:
            _schema("receipt taints must be unique and sorted")

    def to_mapping(self) -> dict[str, object]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_sha256": self.artifact_sha256,
            "effective_taints": [taint.value for taint in self.effective_taints],
            "lineage_sha256": self.lineage_sha256,
            "protocol_sha256": self.protocol_sha256,
            "purpose": self.purpose.value,
            "registry_sha256": self.registry_sha256,
            "registry_source_sha256": self.registry_source_sha256,
        }

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))


@dataclass(frozen=True, slots=True)
class AuthorizedArtifact:
    """Immutable payload returned only by a consumed opaque capability."""

    node: ArtifactLineageNode
    payload: bytes
    receipt: ArtifactUseReceipt

    def __post_init__(self) -> None:
        if type(self.node) is not ArtifactLineageNode:
            _schema("authorized artifact node is invalid")
        if not isinstance(self.payload, bytes):
            _schema("authorized artifact payload must be immutable bytes")
        if type(self.receipt) is not ArtifactUseReceipt:
            _schema("authorized artifact receipt is invalid")
        if (
            self.node.artifact_id != self.receipt.artifact_id
            or self.node.content_sha256 != self.receipt.artifact_sha256
            or sha256_bytes(self.payload) != self.node.content_sha256
            or len(self.payload) != self.node.size_bytes
        ):
            _schema("authorized payload differs from its receipt or node")


def _audit_rule(rule_code: LineageRuleCode) -> RuleCode:
    if rule_code is LineageRuleCode.OK:
        return RuleCode.OK
    if rule_code in {
        LineageRuleCode.REGISTRY_HASH_MISMATCH,
        LineageRuleCode.ARTIFACT_HASH_MISMATCH,
    }:
        return RuleCode.HASH_MISMATCH
    if rule_code is LineageRuleCode.PATH_VIOLATION:
        return RuleCode.PATH_VIOLATION
    if rule_code is LineageRuleCode.LICENSE_DENIED:
        return RuleCode.UNTRUSTED_LICENSE
    if rule_code in {
        LineageRuleCode.TAINTED_LINEAGE,
        LineageRuleCode.PARTITION_DOWNGRADE,
    }:
        return RuleCode.CHECKPOINT_LINEAGE_INVALID
    if rule_code in {
        LineageRuleCode.PURPOSE_DENIED,
        LineageRuleCode.CAPABILITY_INVALID,
    }:
        return RuleCode.PARTITION_ACCESS_DENIED
    return RuleCode.MANIFEST_MISMATCH


def _record_audit(
    audit_log: TemporalAuditLog,
    *,
    action: str,
    outcome: str,
    lineage_rule: LineageRuleCode,
    cutoff_policy_sha256: str,
    knowledge_cutoff: datetime,
    registry_source_sha256: str,
    artifact: ArtifactLineageNode | None,
    purpose: ArtifactUsePurpose | None,
    message: str,
    details: Mapping[str, object] | None = None,
) -> None:
    audit_log.record(
        action=action,
        outcome=outcome,
        rule_code=_audit_rule(lineage_rule),
        policy_sha256=cutoff_policy_sha256,
        cutoff=knowledge_cutoff.isoformat(),
        artifact_id=None if artifact is None else artifact.artifact_id,
        record_id=None if artifact is None else artifact.record.record_id,
        partition=None if artifact is None else artifact.record.partition.value,
        details={
            "lineage_rule_code": lineage_rule.value,
            "message": message,
            "purpose": None if purpose is None else purpose.value,
            "registry_source_sha256": registry_source_sha256,
            **dict(details or {}),
        },
    )


def _read_regular_file(path: Path) -> bytes:
    candidate = Path(os.path.abspath(path.expanduser()))
    current = Path(candidate.anchor)
    try:
        for part in candidate.parts[1:]:
            current /= part
            if current.is_symlink():
                _raise(
                    ArtifactRegistryIntegrityError,
                    LineageRuleCode.PATH_VIOLATION,
                    f"lineage path contains a symbolic link: {current}",
                )
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(candidate, flags)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                _raise(
                    ArtifactRegistryIntegrityError,
                    LineageRuleCode.PATH_VIOLATION,
                    "lineage path is not a regular file",
                )
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except ArtifactLineageError:
        raise
    except OSError as error:
        raise ArtifactRegistryIntegrityError(
            f"cannot read local lineage file: {candidate}",
            LineageRuleCode.PATH_VIOLATION,
        ) from error
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        _raise(
            ArtifactRegistryIntegrityError,
            LineageRuleCode.ARTIFACT_HASH_MISMATCH,
            "lineage file changed while it was read",
        )
    return b"".join(chunks)


def _artifact_path(root: Path, relative_path: str) -> Path:
    raw_root = Path(os.path.abspath(root.expanduser()))
    if raw_root.is_symlink() or not raw_root.is_dir():
        _raise(
            ArtifactRegistryIntegrityError,
            LineageRuleCode.PATH_VIOLATION,
            "artifact root must be a regular local directory",
        )
    candidate = Path(os.path.abspath(raw_root / relative_path))
    if not candidate.is_relative_to(raw_root):
        _raise(
            ArtifactRegistryIntegrityError,
            LineageRuleCode.PATH_VIOLATION,
            "artifact path escapes its root",
        )
    return candidate


class LoadedArtifactLineageRegistry:
    """Opaque local registry handle that rechecks every byte before use."""

    __slots__ = (
        "_artifact_root",
        "_audit_log",
        "_registry",
        "_registry_path",
        "_source_sha256",
    )

    _artifact_root: Path
    _audit_log: TemporalAuditLog
    _registry: ArtifactLineageRegistry
    _registry_path: Path
    _source_sha256: str

    def __init__(self) -> None:
        raise ArtifactCapabilityError(
            "loaded lineage registries may only be issued by the strict local loader",
            LineageRuleCode.CAPABILITY_INVALID,
        )

    def __copy__(self) -> NoReturn:
        raise ArtifactCapabilityError(
            "loaded lineage registry handles cannot be copied",
            LineageRuleCode.CAPABILITY_INVALID,
        )

    def __deepcopy__(self, memo: object) -> NoReturn:
        del memo
        self.__copy__()

    def __reduce__(self) -> NoReturn:
        raise TypeError("LoadedArtifactLineageRegistry cannot be serialized")

    @property
    def registry(self) -> ArtifactLineageRegistry:
        return self._registry

    @property
    def source_sha256(self) -> str:
        return self._source_sha256

    def _verify_bytes(self) -> None:
        if self not in _VERIFIED_REGISTRIES:
            raise ArtifactCapabilityError(
                "lineage registry handle was not issued by the strict loader",
                LineageRuleCode.CAPABILITY_INVALID,
            )
        registry_bytes = _read_regular_file(self._registry_path)
        if sha256_bytes(registry_bytes) != self._source_sha256:
            _raise(
                ArtifactRegistryIntegrityError,
                LineageRuleCode.REGISTRY_HASH_MISMATCH,
                "artifact lineage registry bytes changed",
            )
        try:
            reparsed = ArtifactLineageRegistry.from_mapping(
                read_json_bytes_object(registry_bytes, source=str(self._registry_path))
            )
        except (MetadataValidationError, ArtifactLineageError) as error:
            raise ArtifactRegistryIntegrityError(
                "artifact lineage registry no longer has valid semantics",
                LineageRuleCode.REGISTRY_HASH_MISMATCH,
            ) from error
        if reparsed.to_mapping() != self._registry.to_mapping():
            _raise(
                ArtifactRegistryIntegrityError,
                LineageRuleCode.REGISTRY_HASH_MISMATCH,
                "artifact lineage registry semantics changed",
            )
        for node in self._registry.artifacts:
            payload = _read_regular_file(
                _artifact_path(self._artifact_root, node.relative_path)
            )
            if (
                len(payload) != node.size_bytes
                or sha256_bytes(payload) != node.content_sha256
            ):
                _raise(
                    ArtifactRegistryIntegrityError,
                    LineageRuleCode.ARTIFACT_HASH_MISMATCH,
                    f"artifact bytes changed: {node.artifact_id}",
                )

    def _check_purpose(
        self, node: ArtifactLineageNode, purpose: ArtifactUsePurpose
    ) -> tuple[LineageTaint, ...]:
        if not isinstance(purpose, ArtifactUsePurpose):
            _raise(
                ArtifactUseDeniedError,
                LineageRuleCode.PURPOSE_DENIED,
                "purpose must be an ArtifactUsePurpose",
            )
        taints = self._registry.effective_taints(node.artifact_id)
        if purpose in _PRE_POLICY_PURPOSES and taints:
            _raise(
                ArtifactUseDeniedError,
                LineageRuleCode.TAINTED_LINEAGE,
                "policy-post, post-cutoff, or evaluation-derived lineage cannot "
                f"enter {purpose.value}: {[item.value for item in taints]!r}",
            )
        if (
            purpose is ArtifactUsePurpose.CALIBRATION
            and node.record.partition is not DataPartition.TRAIN
        ):
            _raise(
                ArtifactUseDeniedError,
                LineageRuleCode.PURPOSE_DENIED,
                "calibration may consume only the training partition",
            )
        if (
            purpose
            in {
                ArtifactUsePurpose.MODEL_SELECTION,
                ArtifactUsePurpose.HELD_OUT_VALIDATION,
            }
            and node.record.partition is DataPartition.POST_POLICY_EVALUATION
        ):
            _raise(
                ArtifactUseDeniedError,
                LineageRuleCode.PURPOSE_DENIED,
                "pre-policy validation cannot consume policy-post artifacts",
            )
        return taints

    def authorize(
        self, artifact_id: str, purpose: ArtifactUsePurpose
    ) -> ArtifactUseAuthorization:
        """Issue a single-use purpose-bound capability or audit a rejection."""
        node: ArtifactLineageNode | None = None
        try:
            self._verify_bytes()
            node = self._registry.artifact(artifact_id)
            taints = self._check_purpose(node, purpose)
        except ArtifactLineageError as error:
            _record_audit(
                self._audit_log,
                action="authorize_artifact_use",
                outcome="rejected",
                lineage_rule=error.rule_code,
                cutoff_policy_sha256=self._registry.cutoff_policy_sha256,
                knowledge_cutoff=self._registry.knowledge_cutoff,
                registry_source_sha256=self._source_sha256,
                artifact=node,
                purpose=purpose if isinstance(purpose, ArtifactUsePurpose) else None,
                message=str(error),
            )
            raise
        receipt = ArtifactUseReceipt(
            registry_sha256=self._registry.sha256,
            registry_source_sha256=self._source_sha256,
            protocol_sha256=self._registry.protocol_sha256,
            artifact_id=node.artifact_id,
            artifact_sha256=node.content_sha256,
            lineage_sha256=self._registry.lineage_sha256(node.artifact_id),
            purpose=purpose,
            effective_taints=taints,
        )
        authorization = object.__new__(ArtifactUseAuthorization)
        authorization._loaded = self
        authorization._receipt = receipt
        _ISSUED_AUTHORIZATIONS.add(authorization)
        _record_audit(
            self._audit_log,
            action="authorize_artifact_use",
            outcome="allowed",
            lineage_rule=LineageRuleCode.OK,
            cutoff_policy_sha256=self._registry.cutoff_policy_sha256,
            knowledge_cutoff=self._registry.knowledge_cutoff,
            registry_source_sha256=self._source_sha256,
            artifact=node,
            purpose=purpose,
            message="purpose-bound artifact capability issued",
            details={"receipt_sha256": receipt.sha256},
        )
        return authorization

    def _consume(
        self,
        authorization: ArtifactUseAuthorization,
        expected_purpose: ArtifactUsePurpose,
    ) -> AuthorizedArtifact:
        receipt = authorization._receipt
        node = self._registry.artifact(receipt.artifact_id)
        try:
            if expected_purpose is not receipt.purpose:
                _raise(
                    ArtifactCapabilityError,
                    LineageRuleCode.CAPABILITY_INVALID,
                    "artifact capability purpose does not match its consumer",
                )
            self._verify_bytes()
            current_taints = self._check_purpose(node, receipt.purpose)
            if (
                receipt.registry_sha256 != self._registry.sha256
                or receipt.registry_source_sha256 != self._source_sha256
                or receipt.protocol_sha256 != self._registry.protocol_sha256
                or receipt.artifact_sha256 != node.content_sha256
                or receipt.lineage_sha256
                != self._registry.lineage_sha256(node.artifact_id)
                or receipt.effective_taints != current_taints
            ):
                _raise(
                    ArtifactCapabilityError,
                    LineageRuleCode.CAPABILITY_INVALID,
                    "artifact capability commitments changed before consumption",
                )
            payload = _read_regular_file(
                _artifact_path(self._artifact_root, node.relative_path)
            )
            authorized = AuthorizedArtifact(node=node, payload=payload, receipt=receipt)
        except ArtifactLineageError as error:
            _record_audit(
                self._audit_log,
                action="consume_artifact_use",
                outcome="rejected",
                lineage_rule=error.rule_code,
                cutoff_policy_sha256=self._registry.cutoff_policy_sha256,
                knowledge_cutoff=self._registry.knowledge_cutoff,
                registry_source_sha256=self._source_sha256,
                artifact=node,
                purpose=receipt.purpose,
                message=str(error),
            )
            raise
        _record_audit(
            self._audit_log,
            action="consume_artifact_use",
            outcome="allowed",
            lineage_rule=LineageRuleCode.OK,
            cutoff_policy_sha256=self._registry.cutoff_policy_sha256,
            knowledge_cutoff=self._registry.knowledge_cutoff,
            registry_source_sha256=self._source_sha256,
            artifact=node,
            purpose=receipt.purpose,
            message="artifact payload reverified and capability consumed",
            details={"receipt_sha256": receipt.sha256},
        )
        return authorized


class ArtifactUseAuthorization:
    """Opaque, non-serializable, single-use authority for one purpose."""

    __slots__ = ("_loaded", "_receipt")

    _loaded: LoadedArtifactLineageRegistry
    _receipt: ArtifactUseReceipt

    def __init__(self) -> None:
        raise ArtifactCapabilityError(
            "artifact authorization may only be issued by a verified registry",
            LineageRuleCode.CAPABILITY_INVALID,
        )

    def __copy__(self) -> NoReturn:
        raise ArtifactCapabilityError(
            "artifact authorization cannot be copied",
            LineageRuleCode.CAPABILITY_INVALID,
        )

    def __deepcopy__(self, memo: object) -> NoReturn:
        del memo
        self.__copy__()

    def __reduce__(self) -> NoReturn:
        raise TypeError("ArtifactUseAuthorization cannot be serialized")

    @property
    def receipt(self) -> ArtifactUseReceipt:
        """Return serializable evidence; this is not execution authority."""
        return self._receipt

    def consume(self, expected_purpose: ArtifactUsePurpose) -> AuthorizedArtifact:
        """Consume once after binding the downstream consumer's purpose."""
        if self not in _ISSUED_AUTHORIZATIONS:
            raise ArtifactCapabilityError(
                "artifact authorization is forged or already consumed",
                LineageRuleCode.CAPABILITY_INVALID,
            )
        _ISSUED_AUTHORIZATIONS.remove(self)
        return self._loaded._consume(self, expected_purpose)


_VERIFIED_REGISTRIES: set[LoadedArtifactLineageRegistry] = set()
_ISSUED_AUTHORIZATIONS: set[ArtifactUseAuthorization] = set()


def write_artifact_lineage_registry(
    path: Path, registry: ArtifactLineageRegistry
) -> str:
    """Write one new synthetic registry and return its exact source hash."""
    if type(registry) is not ArtifactLineageRegistry:
        _schema("registry must be an ArtifactLineageRegistry")
    if path.exists() or path.is_symlink():
        _raise(
            ArtifactRegistryIntegrityError,
            LineageRuleCode.PATH_VIOLATION,
            "artifact lineage registry path must be new",
        )
    write_json_file(path, registry.to_mapping())
    return sha256_file(path)


def load_artifact_lineage_registry(
    path: Path,
    *,
    expected_source_sha256: str,
    artifact_root: Path,
    expected_protocol_sha256: str,
    expected_cutoff_policy_sha256: str,
    expected_knowledge_cutoff: datetime,
    audit_log: TemporalAuditLog,
) -> LoadedArtifactLineageRegistry:
    """Strictly load, bind, audit, and verify an external local registry."""
    expected_source = _digest(expected_source_sha256, "expected_source_sha256")
    protocol_sha256 = _digest(expected_protocol_sha256, "expected_protocol_sha256")
    cutoff_policy_sha256 = _digest(
        expected_cutoff_policy_sha256, "expected_cutoff_policy_sha256"
    )
    cutoff = _aware(expected_knowledge_cutoff, "expected_knowledge_cutoff")
    if type(audit_log) is not TemporalAuditLog:
        _schema("audit_log must be an exact TemporalAuditLog")
    registry: ArtifactLineageRegistry | None = None
    try:
        content = _read_regular_file(path)
        if sha256_bytes(content) != expected_source:
            _raise(
                ArtifactRegistryIntegrityError,
                LineageRuleCode.REGISTRY_HASH_MISMATCH,
                "external artifact registry source hash does not match",
            )
        try:
            mapping = read_json_bytes_object(content, source=str(path))
        except MetadataValidationError as error:
            raise ArtifactLineageSchemaError(
                f"external artifact registry JSON is invalid: {error}",
                LineageRuleCode.INVALID_SCHEMA,
            ) from error
        registry = ArtifactLineageRegistry.from_mapping(mapping)
        if (
            registry.protocol_sha256 != protocol_sha256
            or registry.cutoff_policy_sha256 != cutoff_policy_sha256
            or registry.knowledge_cutoff != cutoff
        ):
            _raise(
                ArtifactRegistryIntegrityError,
                LineageRuleCode.REGISTRY_COMMITMENT_MISMATCH,
                "external registry belongs to a different protocol or cutoff",
            )
        loaded = object.__new__(LoadedArtifactLineageRegistry)
        loaded._artifact_root = Path(os.path.abspath(artifact_root.expanduser()))
        loaded._audit_log = audit_log
        loaded._registry = registry
        loaded._registry_path = Path(os.path.abspath(path.expanduser()))
        loaded._source_sha256 = expected_source
        _VERIFIED_REGISTRIES.add(loaded)
        try:
            loaded._verify_bytes()
        except ArtifactLineageError:
            _VERIFIED_REGISTRIES.discard(loaded)
            raise
    except ArtifactLineageError as error:
        _record_audit(
            audit_log,
            action="load_artifact_lineage_registry",
            outcome="rejected",
            lineage_rule=error.rule_code,
            cutoff_policy_sha256=cutoff_policy_sha256,
            knowledge_cutoff=cutoff,
            registry_source_sha256=expected_source,
            artifact=None,
            purpose=None,
            message=str(error),
        )
        raise
    _record_audit(
        audit_log,
        action="load_artifact_lineage_registry",
        outcome="allowed",
        lineage_rule=LineageRuleCode.OK,
        cutoff_policy_sha256=cutoff_policy_sha256,
        knowledge_cutoff=cutoff,
        registry_source_sha256=expected_source,
        artifact=None,
        purpose=None,
        message="external artifact registry and every payload were verified",
        details={
            "artifact_count": len(registry.artifacts),
            "registry_sha256": registry.sha256,
        },
    )
    return loaded


__all__ = [
    "ARTIFACT_LINEAGE_SCHEMA_VERSION",
    "ArtifactCapabilityError",
    "ArtifactLineageError",
    "ArtifactLineageNode",
    "ArtifactLineageRegistry",
    "ArtifactLineageSchemaError",
    "ArtifactParentReference",
    "ArtifactRegistryIntegrityError",
    "ArtifactUseAuthorization",
    "ArtifactUseDeniedError",
    "ArtifactUsePurpose",
    "ArtifactUseReceipt",
    "AuthorizedArtifact",
    "LineageArtifactKind",
    "LineageRuleCode",
    "LineageTaint",
    "LoadedArtifactLineageRegistry",
    "load_artifact_lineage_registry",
    "write_artifact_lineage_registry",
]
