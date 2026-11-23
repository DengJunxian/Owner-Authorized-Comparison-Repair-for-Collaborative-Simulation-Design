"""Deidentified, wall-clock-free audit records for the social layer.

The ledger deliberately accepts commitments and counters only.  Raw subject
identifiers, free-form messages, exception text, prompts, and arbitrary detail
mappings have no representation in this schema.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import stat
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from threading import RLock
from typing import NoReturn, Self, cast

SOCIAL_AUDIT_SCHEMA_VERSION = 1
NULL_COMMITMENT = "0" * 64

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_EVENT_KEY_PATTERN = re.compile(r"^ptk1:[0-9a-f]{64}$")
_EVENT_FIELDS = frozenset(
    {
        "schema_version",
        "sequence",
        "stage",
        "outcome",
        "rule_code",
        "event_key",
        "policy_sha256",
        "input_commitment",
        "output_commitment",
        "counts",
        "previous_event_sha256",
        "event_sha256",
    }
)
_COUNT_FIELDS = frozenset({"accepted_count", "rejected_count"})


class SocialAuditError(RuntimeError):
    """Base class for social-audit failures."""


class SocialAuditSchemaError(SocialAuditError, ValueError):
    """Raised when an audit value does not satisfy the fixed schema."""


class SocialAuditIntegrityError(SocialAuditError):
    """Raised when an in-memory or persisted audit chain cannot be trusted."""


class SocialAuditPersistenceError(SocialAuditError):
    """Raised when synchronous persistence does not complete."""


class SocialAuditStage(StrEnum):
    """Closed set of social-information processing boundaries."""

    INGESTION = "ingestion"
    ANONYMIZATION = "anonymization"
    DEIDENTIFICATION = "deidentification"
    PROPAGATION_INPUT = "propagation_input"
    EXPOSURE = "exposure"
    ATTENTION = "attention"
    BELIEF_UPDATE = "belief_update"
    REPOST = "repost"
    PROPAGATION_OUTPUT = "propagation_output"
    PROPAGATION = "propagation"
    MEMORY = "memory"
    PLANNING = "planning"
    LLM_INPUT = "llm_input"
    LLM_OUTPUT = "llm_output"
    LLM_GATEWAY = "llm_gateway"
    DISTILLED_INPUT = "distilled_input"
    DISTILLED_OUTPUT = "distilled_output"
    DISTILLATION = "distillation"
    PUBLICATION = "publication"


class SocialAuditOutcome(StrEnum):
    """Closed set of audit decisions."""

    ALLOWED = "allowed"
    REJECTED = "rejected"


class SocialAuditRule(StrEnum):
    """Stable rule codes; none carries caller-controlled text."""

    OK = "OK"
    TEMPORAL_BOUNDARY = "TEMPORAL_BOUNDARY"
    SCHEMA_VIOLATION = "SCHEMA_VIOLATION"
    ATTENTION_CAPACITY = "ATTENTION_CAPACITY"
    BELOW_THRESHOLD = "BELOW_THRESHOLD"
    MISSING_CORRECTION_TARGET = "MISSING_CORRECTION_TARGET"
    PRIVACY_VIOLATION = "PRIVACY_VIOLATION"
    ILLEGAL_ADAPTER_OUTPUT = "ILLEGAL_ADAPTER_OUTPUT"
    INVALID_SCHEMA = "INVALID_SCHEMA"
    POST_CUTOFF_REJECTED = "POST_CUTOFF_REJECTED"
    TEMPORAL_LINEAGE_REJECTED = "TEMPORAL_LINEAGE_REJECTED"
    PRIVACY_REJECTED = "PRIVACY_REJECTED"
    PRIVATE_STATE_REJECTED = "PRIVATE_STATE_REJECTED"
    INTEGRITY_FAILURE = "INTEGRITY_FAILURE"
    AUDIT_WRITE_FAILED = "AUDIT_WRITE_FAILED"


class SubjectKind(StrEnum):
    """Domain separators for keyed, non-reversible subject commitments."""

    AGENT = "agent"
    GRAPH_NODE = "graph_node"
    INFORMATION_EVENT = "information_event"
    DELIVERY = "delivery"
    MEMORY = "memory"


def _fail(message: str) -> NoReturn:
    raise SocialAuditSchemaError(message)


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SocialAuditSchemaError("audit value is not canonical JSON") from error


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        _fail(f"{field} must be a lowercase SHA-256 commitment")
    return value


def _event_key(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or _EVENT_KEY_PATTERN.fullmatch(value) is None:
        _fail("event_key must be null or a committed PolicyTwin event key")
    return value


def _count(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(f"{field} must be a non-negative integer")
    return value


def _strict_fields(
    value: dict[str, object], expected: frozenset[str], context: str
) -> None:
    actual = frozenset(value)
    if actual != expected:
        _fail(f"{context} has missing or unknown fields")


def _strict_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail("audit JSON contains a duplicate field")
        result[key] = value
    return result


def _reject_constant(value: str) -> NoReturn:
    del value
    _fail("audit JSON contains a non-finite number")


def _parse_json_object(line: bytes) -> dict[str, object]:
    try:
        value = json.loads(
            line.decode("utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise SocialAuditSchemaError("audit JSONL contains invalid JSON") from error
    if not isinstance(value, dict):
        _fail("each audit JSONL line must contain one object")
    return cast(dict[str, object], value)


def subject_commitment(subject_id: str, *, secret: bytes, kind: SubjectKind) -> str:
    """Return a domain-separated HMAC without exposing a low-entropy raw ID.

    The secret is intentionally required: a plain hash of a short account or
    agent identifier is vulnerable to dictionary reversal.  Neither the secret
    nor the raw identifier is retained by the audit ledger.
    """

    if (
        not isinstance(subject_id, str)
        or not subject_id
        or len(subject_id) > 1024
        or "\x00" in subject_id
    ):
        _fail("subject_id must be a bounded non-empty string")
    if not isinstance(secret, bytes) or len(secret) < 32:
        _fail("subject commitment secret must contain at least 32 bytes")
    if not isinstance(kind, SubjectKind):
        _fail("kind must be a SubjectKind")
    material = (
        b"policytwin.social-subject.v1\x00"
        + kind.value.encode("ascii")
        + b"\x00"
        + subject_id.encode("utf-8")
    )
    return hmac.new(secret, material, hashlib.sha256).hexdigest()


@dataclass(frozen=True, slots=True)
class AuditCounts:
    """The only counters admitted to a social audit event."""

    accepted_count: int
    rejected_count: int

    def __post_init__(self) -> None:
        _count(self.accepted_count, "accepted_count")
        _count(self.rejected_count, "rejected_count")

    def to_mapping(self) -> dict[str, int]:
        """Return the fixed counter representation."""
        return {
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
        }

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        """Strictly reconstruct counters, rejecting arbitrary detail fields."""
        if not isinstance(value, dict) or not all(
            isinstance(key, str) for key in value
        ):
            _fail("counts must be an object")
        mapping = cast(dict[str, object], value)
        _strict_fields(mapping, _COUNT_FIELDS, "counts")
        return cls(
            accepted_count=_count(mapping["accepted_count"], "accepted_count"),
            rejected_count=_count(mapping["rejected_count"], "rejected_count"),
        )


@dataclass(frozen=True, slots=True)
class SocialAuditEvent:
    """One fixed-field, hash-chained, wall-clock-free social audit event."""

    schema_version: int
    sequence: int
    stage: SocialAuditStage
    outcome: SocialAuditOutcome
    rule_code: SocialAuditRule
    event_key: str | None
    policy_sha256: str
    input_commitment: str
    output_commitment: str
    counts: AuditCounts
    previous_event_sha256: str
    event_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != SOCIAL_AUDIT_SCHEMA_VERSION:
            _fail("unsupported social audit schema version")
        _count(self.sequence, "sequence")
        if not isinstance(self.stage, SocialAuditStage):
            _fail("stage must be a SocialAuditStage")
        if not isinstance(self.outcome, SocialAuditOutcome):
            _fail("outcome must be a SocialAuditOutcome")
        if not isinstance(self.rule_code, SocialAuditRule):
            _fail("rule_code must be a SocialAuditRule")
        _event_key(self.event_key)
        _sha256(self.policy_sha256, "policy_sha256")
        _sha256(self.input_commitment, "input_commitment")
        _sha256(self.output_commitment, "output_commitment")
        if not isinstance(self.counts, AuditCounts):
            _fail("counts must be AuditCounts")
        _sha256(self.previous_event_sha256, "previous_event_sha256")
        _sha256(self.event_sha256, "event_sha256")
        expected = _sha256_bytes(_canonical_json_bytes(self.core_mapping()))
        if self.event_sha256 != expected:
            _fail("event_sha256 does not match the fixed event fields")

    def core_mapping(self) -> dict[str, object]:
        """Return the exact fields covered by ``event_sha256``."""
        return {
            "counts": self.counts.to_mapping(),
            "event_key": self.event_key,
            "input_commitment": self.input_commitment,
            "outcome": self.outcome.value,
            "output_commitment": self.output_commitment,
            "policy_sha256": self.policy_sha256,
            "previous_event_sha256": self.previous_event_sha256,
            "rule_code": self.rule_code.value,
            "schema_version": self.schema_version,
            "sequence": self.sequence,
            "stage": self.stage.value,
        }

    def to_mapping(self) -> dict[str, object]:
        """Return the complete fixed-field JSON representation."""
        return {**self.core_mapping(), "event_sha256": self.event_sha256}

    @classmethod
    def create(
        cls,
        *,
        sequence: int,
        stage: SocialAuditStage,
        outcome: SocialAuditOutcome,
        rule_code: SocialAuditRule,
        event_key: str | None,
        policy_sha256: str,
        input_commitment: str,
        output_commitment: str,
        counts: AuditCounts,
        previous_event_sha256: str,
    ) -> Self:
        """Construct one event and derive its content hash."""
        core = {
            "counts": counts.to_mapping(),
            "event_key": _event_key(event_key),
            "input_commitment": _sha256(input_commitment, "input_commitment"),
            "outcome": outcome.value,
            "output_commitment": _sha256(output_commitment, "output_commitment"),
            "policy_sha256": _sha256(policy_sha256, "policy_sha256"),
            "previous_event_sha256": _sha256(
                previous_event_sha256, "previous_event_sha256"
            ),
            "rule_code": rule_code.value,
            "schema_version": SOCIAL_AUDIT_SCHEMA_VERSION,
            "sequence": _count(sequence, "sequence"),
            "stage": stage.value,
        }
        return cls(
            schema_version=SOCIAL_AUDIT_SCHEMA_VERSION,
            sequence=sequence,
            stage=stage,
            outcome=outcome,
            rule_code=rule_code,
            event_key=event_key,
            policy_sha256=policy_sha256,
            input_commitment=input_commitment,
            output_commitment=output_commitment,
            counts=counts,
            previous_event_sha256=previous_event_sha256,
            event_sha256=_sha256_bytes(_canonical_json_bytes(core)),
        )

    @classmethod
    def from_mapping(cls, value: dict[str, object]) -> Self:
        """Strictly reconstruct an event and verify its content hash."""
        _strict_fields(value, _EVENT_FIELDS, "social audit event")
        stage_value = value["stage"]
        outcome_value = value["outcome"]
        rule_value = value["rule_code"]
        if not isinstance(stage_value, str):
            _fail("audit enum values must be strings")
        if not isinstance(outcome_value, str):
            _fail("audit enum values must be strings")
        if not isinstance(rule_value, str):
            _fail("audit enum values must be strings")
        try:
            stage = SocialAuditStage(stage_value)
            outcome = SocialAuditOutcome(outcome_value)
            rule_code = SocialAuditRule(rule_value)
        except (TypeError, ValueError) as error:
            raise SocialAuditSchemaError(
                "audit event contains an unknown enum"
            ) from error
        return cls(
            schema_version=_count(value["schema_version"], "schema_version"),
            sequence=_count(value["sequence"], "sequence"),
            stage=stage,
            outcome=outcome,
            rule_code=rule_code,
            event_key=_event_key(value["event_key"]),
            policy_sha256=_sha256(value["policy_sha256"], "policy_sha256"),
            input_commitment=_sha256(value["input_commitment"], "input_commitment"),
            output_commitment=_sha256(value["output_commitment"], "output_commitment"),
            counts=AuditCounts.from_mapping(value["counts"]),
            previous_event_sha256=_sha256(
                value["previous_event_sha256"], "previous_event_sha256"
            ),
            event_sha256=_sha256(value["event_sha256"], "event_sha256"),
        )


def verify_social_audit_chain(events: tuple[SocialAuditEvent, ...]) -> None:
    """Verify sequence continuity and every link in a social audit chain."""
    if not isinstance(events, tuple) or not all(
        isinstance(event, SocialAuditEvent) for event in events
    ):
        _fail("events must be an immutable tuple of SocialAuditEvent values")
    previous = NULL_COMMITMENT
    for sequence, event in enumerate(events):
        expected = _sha256_bytes(_canonical_json_bytes(event.core_mapping()))
        if (
            event.sequence != sequence
            or event.previous_event_sha256 != previous
            or event.event_sha256 != expected
        ):
            raise SocialAuditIntegrityError("social audit hash chain is invalid")
        previous = event.event_sha256


def _decode_jsonl(content: bytes) -> tuple[SocialAuditEvent, ...]:
    if not content:
        return ()
    lines = content.splitlines()
    if not lines or any(not line for line in lines):
        raise SocialAuditSchemaError("social audit JSONL contains a blank line")
    events = tuple(
        SocialAuditEvent.from_mapping(_parse_json_object(line)) for line in lines
    )
    verify_social_audit_chain(events)
    return events


class SocialAuditLog:
    """Thread-safe audit ledger with optional fsync-on-record JSONL output."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = (
            None
            if path is None
            else Path(os.path.abspath(os.fspath(path.expanduser())))
        )
        self._events: list[SocialAuditEvent] = []
        self._lock = RLock()
        self._file_identity: tuple[int, int] | None = None
        self._persisted_size = 0
        self._persisted_sha256 = _sha256_bytes(b"")
        self._compromised = False
        if self._path is not None:
            self._open_persisted_log()

    @property
    def path(self) -> Path | None:
        """Return the optional absolute persistence path."""
        return self._path

    @property
    def events(self) -> tuple[SocialAuditEvent, ...]:
        """Return an immutable view of all accepted audit events."""
        with self._lock:
            return tuple(self._events)

    @property
    def compromised(self) -> bool:
        """Return whether persistence or integrity verification has failed."""
        with self._lock:
            return self._compromised

    def _open_persisted_log(self) -> None:
        assert self._path is not None
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            flags = os.O_RDWR | os.O_CREAT
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(self._path, flags, 0o600)
            with os.fdopen(descriptor, "r+b") as stream:
                opened = os.fstat(stream.fileno())
                linked = os.lstat(self._path)
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or (opened.st_dev, opened.st_ino) != (linked.st_dev, linked.st_ino)
                    or opened.st_nlink != 1
                ):
                    raise OSError("social audit path is not a unique regular file")
                content = stream.read()
        except (OSError, SocialAuditError) as error:
            self._compromised = True
            raise SocialAuditIntegrityError(
                "cannot initialize the social audit log"
            ) from error
        self._events = list(_decode_jsonl(content))
        self._file_identity = (opened.st_dev, opened.st_ino)
        self._persisted_size = len(content)
        self._persisted_sha256 = _sha256_bytes(content)

    def _assert_integrity_locked(self) -> None:
        if self._compromised:
            raise SocialAuditIntegrityError("social audit log is compromised")
        try:
            verify_social_audit_chain(tuple(self._events))
            if self._path is None:
                return
            linked = os.lstat(self._path)
            if (
                not stat.S_ISREG(linked.st_mode)
                or linked.st_nlink != 1
                or self._file_identity != (linked.st_dev, linked.st_ino)
                or linked.st_size != self._persisted_size
                or _sha256_file(self._path) != self._persisted_sha256
            ):
                raise OSError("persisted social audit prefix changed")
        except (OSError, SocialAuditError) as error:
            self._compromised = True
            raise SocialAuditIntegrityError(
                "social audit integrity verification failed"
            ) from error

    def assert_integrity(self) -> None:
        """Fail closed if the memory chain or persisted JSONL changed."""
        with self._lock:
            self._assert_integrity_locked()

    def record(
        self,
        *,
        stage: SocialAuditStage,
        outcome: SocialAuditOutcome,
        rule_code: SocialAuditRule,
        event_key: str | None,
        policy_sha256: str,
        input_commitment: str,
        output_commitment: str,
        accepted_count: int,
        rejected_count: int,
    ) -> SocialAuditEvent:
        """Synchronously record one typed decision containing no free text."""
        if not isinstance(stage, SocialAuditStage):
            _fail("stage must be a SocialAuditStage")
        if not isinstance(outcome, SocialAuditOutcome):
            _fail("outcome must be a SocialAuditOutcome")
        if not isinstance(rule_code, SocialAuditRule):
            _fail("rule_code must be a SocialAuditRule")
        counts = AuditCounts(
            accepted_count=accepted_count,
            rejected_count=rejected_count,
        )
        with self._lock:
            self._assert_integrity_locked()
            previous = (
                self._events[-1].event_sha256 if self._events else NULL_COMMITMENT
            )
            event = SocialAuditEvent.create(
                sequence=len(self._events),
                stage=stage,
                outcome=outcome,
                rule_code=rule_code,
                event_key=event_key,
                policy_sha256=policy_sha256,
                input_commitment=input_commitment,
                output_commitment=output_commitment,
                counts=counts,
                previous_event_sha256=previous,
            )
            if self._path is not None:
                self._persist(event)
            self._events.append(event)
            return event

    def _persist(self, event: SocialAuditEvent) -> None:
        assert self._path is not None
        rendered = _canonical_json_bytes(event.to_mapping()) + b"\n"
        write_started = False
        try:
            flags = os.O_RDWR
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(self._path, flags)
            with os.fdopen(descriptor, "r+b") as stream:
                opened = os.fstat(stream.fileno())
                if self._file_identity != (opened.st_dev, opened.st_ino):
                    raise OSError("social audit file identity changed")
                stream.seek(0, os.SEEK_END)
                if stream.tell() != self._persisted_size:
                    raise OSError("social audit file size changed")
                write_started = True
                stream.write(rendered)
                stream.flush()
                os.fsync(stream.fileno())
            persisted_sha256 = _sha256_file(self._path)
        except OSError as error:
            rolled_back = self._truncate_failed_append() if write_started else True
            self._compromised = True
            suffix = "rolled back" if rolled_back else "rollback failed"
            raise SocialAuditPersistenceError(
                f"social audit append failed and was {suffix}"
            ) from error
        self._persisted_size += len(rendered)
        self._persisted_sha256 = persisted_sha256

    def _truncate_failed_append(self) -> bool:
        if self._path is None or self._file_identity is None:
            return False
        try:
            descriptor = os.open(self._path, os.O_RDWR)
            with os.fdopen(descriptor, "r+b") as stream:
                opened = os.fstat(stream.fileno())
                if self._file_identity != (opened.st_dev, opened.st_ino):
                    return False
                stream.truncate(self._persisted_size)
                stream.flush()
                os.fsync(stream.fileno())
            return (
                os.lstat(self._path).st_size == self._persisted_size
                and _sha256_file(self._path) == self._persisted_sha256
            )
        except OSError:
            return False


__all__ = [
    "AuditCounts",
    "NULL_COMMITMENT",
    "SOCIAL_AUDIT_SCHEMA_VERSION",
    "SocialAuditError",
    "SocialAuditEvent",
    "SocialAuditIntegrityError",
    "SocialAuditLog",
    "SocialAuditOutcome",
    "SocialAuditPersistenceError",
    "SocialAuditRule",
    "SocialAuditSchemaError",
    "SocialAuditStage",
    "SubjectKind",
    "subject_commitment",
    "verify_social_audit_chain",
]
