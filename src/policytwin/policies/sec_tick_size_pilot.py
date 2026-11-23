"""Evidence-bound, configurable SEC Tick Size Pilot policy interfaces.

No SEC rule values or treatment assignments are embedded here.  The repository
currently contains no primary-source specification or official assignment
artifact, so executable examples must be explicitly marked synthetic.  A real
enabled configuration fails closed unless every policy rule is bound to a
locally verified, pre-cutoff primary source.
"""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Literal, NoReturn, Protocol, Self, cast

from policytwin.provenance import (
    JsonlEventWriter,
    canonical_json_bytes,
    read_json_bytes_object,
    sha256_bytes,
)

POLICY_SCHEMA_VERSION = "policytwin.sec-tick-size-pilot.v1"
SOURCE_REGISTRY_SCHEMA_VERSION = "policytwin.policy-source-registry.v1"
LEGACY_ASSIGNMENT_SCHEMA_VERSION = "policytwin.security-assignment.v1"
ASSIGNMENT_SCHEMA_VERSION = "policytwin.security-assignment.v2"
COMPLIANCE_AUDIT_SCHEMA_VERSION = "policytwin.policy-compliance-audit.v3"
SYNTHETIC_EVALUATOR_SCHEMA_VERSION = "policytwin.synthetic-evaluator.v1"
POLICY_MECHANICS_CLAIM_SCOPE: Literal[
    "synthetic_policy_rule_mechanics_non_authorizing"
] = "synthetic_policy_rule_mechanics_non_authorizing"

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ZERO_SHA256 = "0" * 64


class PolicySchemaError(ValueError):
    """Raised when policy configuration or evidence is malformed."""


class PolicyEvidenceError(PolicySchemaError):
    """Raised when a rule lacks verified, pre-cutoff evidence."""


class AssignmentIntegrityError(PolicySchemaError):
    """Raised when a security assignment artifact cannot be trusted."""


class PolicyGroup(StrEnum):
    """Closed SEC Tick Size Pilot assignment labels."""

    CONTROL = "control"
    TEST_GROUP_1 = "test_group_1"
    TEST_GROUP_2 = "test_group_2"
    TEST_GROUP_3 = "test_group_3"


class AssignmentResolutionStatus(StrEnum):
    """Auditable outcome of a timestamped assignment lookup."""

    ASSIGNED = "assigned"
    LEGACY_SYNTHETIC = "legacy_synthetic"
    NOT_YET_EFFECTIVE = "not_yet_effective"
    DELETED = "deleted"
    UNKNOWN_SECURITY = "unknown_security"
    NOT_EVALUATED = "not_evaluated"


class RuleEvidence(StrEnum):
    """Scientific status of one configured rule value."""

    VERIFIED_PRIMARY = "verified_primary"
    PENDING_VERIFICATION = "pending_verification"
    SYNTHETIC_FIXTURE = "synthetic_fixture"


class ComplianceAction(StrEnum):
    """Policy mechanism being evaluated before market mutation."""

    QUOTE = "quote"
    TRADE = "trade"


class ExceptionScope(StrEnum):
    """A documented exception cannot escape its declared mechanism."""

    QUOTE_TICK = "quote_tick"
    TRADE_TICK = "trade_tick"
    TRADE_AT = "trade_at"


class ComplianceOutcome(StrEnum):
    """Machine-readable policy outcome."""

    ALLOWED = "allowed"
    REJECTED = "rejected"
    NO_OP = "no_op"


class ComplianceRuleCode(StrEnum):
    """Stable policy compliance decisions."""

    OK = "OK"
    NO_OP_DISABLED = "NO_OP_DISABLED"
    OUTSIDE_POLICY_WINDOW = "OUTSIDE_POLICY_WINDOW"
    UNASSIGNED_SECURITY = "UNASSIGNED_SECURITY"
    CONTROL_BASELINE = "CONTROL_BASELINE"
    OFF_QUOTE_GRID = "OFF_QUOTE_GRID"
    OFF_TRADE_GRID = "OFF_TRADE_GRID"
    EXCEPTION_NOT_CONFIGURED = "EXCEPTION_NOT_CONFIGURED"
    EXCEPTION_NOT_APPLICABLE = "EXCEPTION_NOT_APPLICABLE"
    EXCEPTION_EVALUATOR_REQUIRED = "EXCEPTION_EVALUATOR_REQUIRED"
    EXCEPTION_DENIED = "EXCEPTION_DENIED"
    EXCEPTION_ALLOWED = "EXCEPTION_ALLOWED"
    EVALUATOR_FAILED = "EVALUATOR_FAILED"
    EVALUATOR_RECEIPT_INVALID = "EVALUATOR_RECEIPT_INVALID"
    TRADE_AT_CONTEXT_REQUIRED = "TRADE_AT_CONTEXT_REQUIRED"
    TRADE_AT_EVALUATOR_REQUIRED = "TRADE_AT_EVALUATOR_REQUIRED"
    TRADE_AT_REJECTED = "TRADE_AT_REJECTED"
    TRADE_AT_ALLOWED = "TRADE_AT_ALLOWED"


def _fail(message: str) -> NoReturn:
    raise PolicySchemaError(message)


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        _fail(f"{field} must be a stable ASCII identifier")
    return value


def _text(value: object, field: str, *, maximum: int = 1024) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > maximum
        or "\x00" in value
    ):
        _fail(f"{field} must be a bounded non-empty string")
    return value


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail(f"{field} must be a lowercase SHA-256 digest")
    return value


def _aware_utc(value: object, field: str) -> datetime:
    if not isinstance(value, datetime):
        _fail(f"{field} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        _fail(f"{field} must include an explicit timezone")
    try:
        return value.astimezone(UTC)
    except (OverflowError, ValueError) as error:
        raise PolicySchemaError(f"{field} is outside the UTC range") from error


def _parse_datetime(value: object, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        _fail(f"{field} must be an ISO 8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise PolicySchemaError(f"{field} must be ISO 8601") from error
    return _aware_utc(parsed, field)


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _fail(f"{field} must be a positive integer")
    return value


def _strict_fields(
    mapping: Mapping[str, object], expected: frozenset[str], context: str
) -> None:
    actual = frozenset(mapping)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        _fail(f"{context} fields differ; missing={missing}, unknown={unknown}")


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        _fail(f"{field} must be an object with string keys")
    return cast(Mapping[str, object], value)


def _sequence(value: object, field: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        _fail(f"{field} must be an array")
    return cast(Sequence[object], value)


def _local_absolute(path: Path) -> Path:
    return Path(os.path.abspath(path.expanduser()))


def _reject_symlink_components(path: Path, error_type: type[PolicySchemaError]) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            raise error_type(f"local policy path contains a symbolic link: {current}")


def _read_committed_bytes(
    path: Path,
    expected_sha256: str,
    error_type: type[PolicySchemaError],
) -> tuple[Path, bytes]:
    """Read, hash, and identify one regular local file through one descriptor."""
    expected = _sha256(expected_sha256, "expected_sha256")
    candidate = _local_absolute(path)
    _reject_symlink_components(candidate, error_type)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(candidate, flags)
    except OSError as error:
        raise error_type(
            f"cannot open committed local policy file: {candidate}"
        ) from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise error_type("committed local policy path is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    except OSError as error:
        raise error_type(
            f"cannot read committed local policy file: {candidate}"
        ) from error
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after:
        raise error_type("committed local policy file changed while it was read")
    content = b"".join(chunks)
    if sha256_bytes(content) != expected:
        raise error_type("committed local policy file hash does not match")
    return candidate, content


def _read_committed_json(
    path: Path,
    expected_sha256: str,
    error_type: type[PolicySchemaError],
) -> tuple[Path, Mapping[str, object]]:
    candidate, content = _read_committed_bytes(path, expected_sha256, error_type)
    try:
        value = read_json_bytes_object(content, source=str(candidate))
    except ValueError as error:
        raise error_type(f"committed policy JSON is invalid: {candidate}") from error
    return candidate, value


def _enum_value[T: StrEnum](value: object, enum_type: type[T], field: str) -> T:
    if not isinstance(value, str):
        _fail(f"{field} must be a string")
    try:
        return enum_type(value)
    except ValueError as error:
        raise PolicySchemaError(f"unsupported {field}: {value}") from error


def _source_ids(values: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        _fail("source_ids must be an immutable tuple")
    normalized = tuple(sorted(_identifier(item, "source_id") for item in values))
    if len(set(normalized)) != len(normalized):
        _fail("source_ids must be unique")
    return normalized


@dataclass(frozen=True, slots=True)
class EvidenceBinding:
    """Status and local source identities supporting one rule."""

    rule_id: str
    status: RuleEvidence
    source_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.rule_id, "rule_id")
        if not isinstance(self.status, RuleEvidence):
            _fail("evidence status must be a RuleEvidence")
        object.__setattr__(self, "source_ids", _source_ids(self.source_ids))
        if self.status is RuleEvidence.VERIFIED_PRIMARY and not self.source_ids:
            _fail("verified primary evidence requires at least one source_id")
        if self.status is RuleEvidence.PENDING_VERIFICATION and self.source_ids:
            _fail("pending evidence must not claim verified source bindings")

    def to_mapping(self) -> dict[str, object]:
        return {
            "rule_id": self.rule_id,
            "source_ids": list(self.source_ids),
            "status": self.status.value,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(
            value, frozenset({"rule_id", "status", "source_ids"}), "evidence binding"
        )
        return cls(
            rule_id=_identifier(value["rule_id"], "rule_id"),
            status=_enum_value(value["status"], RuleEvidence, "evidence status"),
            source_ids=tuple(
                _identifier(item, "source_id")
                for item in _sequence(value["source_ids"], "source_ids")
            ),
        )


@dataclass(frozen=True, slots=True)
class PolicyWindow:
    """Explicit policy interval; no boundary convention is assumed."""

    starts_at: datetime
    ends_at: datetime
    start_inclusive: bool
    end_inclusive: bool
    evidence: EvidenceBinding

    def __post_init__(self) -> None:
        object.__setattr__(self, "starts_at", _aware_utc(self.starts_at, "starts_at"))
        object.__setattr__(self, "ends_at", _aware_utc(self.ends_at, "ends_at"))
        if self.ends_at <= self.starts_at:
            _fail("policy window ends_at must be after starts_at")
        if not isinstance(self.start_inclusive, bool) or not isinstance(
            self.end_inclusive, bool
        ):
            _fail("policy window boundary flags must be booleans")
        if not isinstance(self.evidence, EvidenceBinding):
            _fail("policy window requires EvidenceBinding")

    def contains(self, timestamp: datetime) -> bool:
        value = _aware_utc(timestamp, "timestamp")
        after_start = (
            value >= self.starts_at if self.start_inclusive else value > self.starts_at
        )
        before_end = (
            value <= self.ends_at if self.end_inclusive else value < self.ends_at
        )
        return after_start and before_end

    def to_mapping(self) -> dict[str, object]:
        return {
            "end_inclusive": self.end_inclusive,
            "ends_at": self.ends_at.isoformat(),
            "evidence": self.evidence.to_mapping(),
            "start_inclusive": self.start_inclusive,
            "starts_at": self.starts_at.isoformat(),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        """Parse the exact persisted policy-window schema."""
        _strict_fields(
            value,
            frozenset(
                {
                    "starts_at",
                    "ends_at",
                    "start_inclusive",
                    "end_inclusive",
                    "evidence",
                }
            ),
            "policy window",
        )
        start_inclusive = value["start_inclusive"]
        end_inclusive = value["end_inclusive"]
        if not isinstance(start_inclusive, bool) or not isinstance(end_inclusive, bool):
            _fail("policy window boundary flags must be booleans")
        return cls(
            starts_at=_parse_datetime(value["starts_at"], "starts_at"),
            ends_at=_parse_datetime(value["ends_at"], "ends_at"),
            start_inclusive=start_inclusive,
            end_inclusive=end_inclusive,
            evidence=EvidenceBinding.from_mapping(
                _mapping(value["evidence"], "evidence")
            ),
        )


@dataclass(frozen=True, slots=True)
class TickConstraint:
    """Integer price grid configured from evidence, never silently rounded."""

    increment_units: int
    origin_units: int
    evidence: EvidenceBinding

    def __post_init__(self) -> None:
        _positive_int(self.increment_units, "increment_units")
        if isinstance(self.origin_units, bool) or not isinstance(
            self.origin_units, int
        ):
            _fail("origin_units must be an integer")
        if not isinstance(self.evidence, EvidenceBinding):
            _fail("tick constraint requires EvidenceBinding")

    def allows(self, price_units: int) -> bool:
        if isinstance(price_units, bool) or not isinstance(price_units, int):
            _fail("price_units must be an integer")
        if price_units <= 0:
            _fail("price_units must be positive")
        return (price_units - self.origin_units) % self.increment_units == 0

    def to_mapping(self) -> dict[str, object]:
        return {
            "evidence": self.evidence.to_mapping(),
            "increment_units": self.increment_units,
            "origin_units": self.origin_units,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        """Parse the exact persisted integer-tick schema."""
        _strict_fields(
            value,
            frozenset({"increment_units", "origin_units", "evidence"}),
            "tick constraint",
        )
        origin = value["origin_units"]
        if isinstance(origin, bool) or not isinstance(origin, int):
            _fail("origin_units must be an integer")
        return cls(
            increment_units=_positive_int(value["increment_units"], "increment_units"),
            origin_units=origin,
            evidence=EvidenceBinding.from_mapping(
                _mapping(value["evidence"], "evidence")
            ),
        )


@dataclass(frozen=True, slots=True)
class TradeAtRestriction:
    """Interface marker; detailed trade-at semantics live in an evaluator."""

    restriction_id: str
    evaluator_id: str
    context_required: bool
    evidence: EvidenceBinding

    def __post_init__(self) -> None:
        _identifier(self.restriction_id, "restriction_id")
        _identifier(self.evaluator_id, "evaluator_id")
        if not isinstance(self.context_required, bool):
            _fail("context_required must be a boolean")
        if not isinstance(self.evidence, EvidenceBinding):
            _fail("trade-at restriction requires EvidenceBinding")

    def to_mapping(self) -> dict[str, object]:
        return {
            "context_required": self.context_required,
            "evaluator_id": self.evaluator_id,
            "evidence": self.evidence.to_mapping(),
            "restriction_id": self.restriction_id,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        """Parse one evaluator-bound trade-at interface marker."""
        _strict_fields(
            value,
            frozenset(
                {"restriction_id", "evaluator_id", "context_required", "evidence"}
            ),
            "trade-at restriction",
        )
        context_required = value["context_required"]
        if not isinstance(context_required, bool):
            _fail("context_required must be a boolean")
        return cls(
            restriction_id=_identifier(value["restriction_id"], "restriction_id"),
            evaluator_id=_identifier(value["evaluator_id"], "evaluator_id"),
            context_required=context_required,
            evidence=EvidenceBinding.from_mapping(
                _mapping(value["evidence"], "evidence")
            ),
        )


@dataclass(frozen=True, slots=True)
class DocumentedException:
    """Evidence-bound exception with closed group and mechanism scopes."""

    exception_id: str
    evaluator_id: str
    groups: tuple[PolicyGroup, ...]
    scopes: tuple[ExceptionScope, ...]
    evidence: EvidenceBinding

    def __post_init__(self) -> None:
        _identifier(self.exception_id, "exception_id")
        _identifier(self.evaluator_id, "evaluator_id")
        if not isinstance(self.groups, tuple) or not all(
            isinstance(item, PolicyGroup) for item in self.groups
        ):
            _fail("documented exception groups must be an immutable PolicyGroup tuple")
        if not isinstance(self.scopes, tuple) or not all(
            isinstance(item, ExceptionScope) for item in self.scopes
        ):
            _fail("documented exception scopes must be an immutable scope tuple")
        ordered_groups = tuple(sorted(self.groups, key=lambda item: item.value))
        ordered_scopes = tuple(sorted(self.scopes, key=lambda item: item.value))
        if not ordered_groups:
            _fail("documented exception requires PolicyGroup scopes")
        if not ordered_scopes:
            _fail("documented exception requires mechanism scopes")
        if len(set(ordered_groups)) != len(ordered_groups) or len(
            set(ordered_scopes)
        ) != len(ordered_scopes):
            _fail("documented exception scopes must be unique")
        object.__setattr__(self, "groups", ordered_groups)
        object.__setattr__(self, "scopes", ordered_scopes)
        if not isinstance(self.evidence, EvidenceBinding):
            _fail("documented exception requires EvidenceBinding")

    def to_mapping(self) -> dict[str, object]:
        return {
            "evaluator_id": self.evaluator_id,
            "evidence": self.evidence.to_mapping(),
            "exception_id": self.exception_id,
            "groups": [item.value for item in self.groups],
            "scopes": [item.value for item in self.scopes],
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        """Parse one closed-scope documented exception."""
        _strict_fields(
            value,
            frozenset({"exception_id", "evaluator_id", "groups", "scopes", "evidence"}),
            "documented exception",
        )
        return cls(
            exception_id=_identifier(value["exception_id"], "exception_id"),
            evaluator_id=_identifier(value["evaluator_id"], "evaluator_id"),
            groups=tuple(
                _enum_value(item, PolicyGroup, "exception group")
                for item in _sequence(value["groups"], "groups")
            ),
            scopes=tuple(
                _enum_value(item, ExceptionScope, "exception scope")
                for item in _sequence(value["scopes"], "scopes")
            ),
            evidence=EvidenceBinding.from_mapping(
                _mapping(value["evidence"], "evidence")
            ),
        )


@dataclass(frozen=True, slots=True)
class BaselineMarketRules:
    """Venue baseline rules used by control and every no-op resolution."""

    quote_tick_constraint: TickConstraint
    trade_tick_constraint: TickConstraint

    def __post_init__(self) -> None:
        if not isinstance(self.quote_tick_constraint, TickConstraint) or not isinstance(
            self.trade_tick_constraint, TickConstraint
        ):
            _fail("baseline quote and trade constraints must be TickConstraint values")

    def to_mapping(self) -> dict[str, object]:
        return {
            "quote_tick_constraint": self.quote_tick_constraint.to_mapping(),
            "trade_tick_constraint": self.trade_tick_constraint.to_mapping(),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        """Parse the complete baseline quote/trade grids."""
        _strict_fields(
            value,
            frozenset({"quote_tick_constraint", "trade_tick_constraint"}),
            "baseline market rules",
        )
        return cls(
            quote_tick_constraint=TickConstraint.from_mapping(
                _mapping(value["quote_tick_constraint"], "quote_tick_constraint")
            ),
            trade_tick_constraint=TickConstraint.from_mapping(
                _mapping(value["trade_tick_constraint"], "trade_tick_constraint")
            ),
        )


@dataclass(frozen=True, slots=True)
class GroupRule:
    """Explicit mechanisms for one assignment group."""

    group: PolicyGroup
    quote_tick_constraint: TickConstraint | None
    trade_tick_constraint: TickConstraint | None
    trade_at_restriction: TradeAtRestriction | None = None
    exception_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.group, PolicyGroup):
            _fail("group rule requires PolicyGroup")
        if self.quote_tick_constraint is not None and not isinstance(
            self.quote_tick_constraint, TickConstraint
        ):
            _fail("group quote constraint must be TickConstraint or None")
        if self.trade_tick_constraint is not None and not isinstance(
            self.trade_tick_constraint, TickConstraint
        ):
            _fail("group trade constraint must be TickConstraint or None")
        if self.trade_at_restriction is not None and not isinstance(
            self.trade_at_restriction, TradeAtRestriction
        ):
            _fail("group trade-at restriction must be TradeAtRestriction or None")
        if self.group is PolicyGroup.CONTROL:
            if any(
                item is not None
                for item in (
                    self.quote_tick_constraint,
                    self.trade_tick_constraint,
                    self.trade_at_restriction,
                )
            ):
                _fail("control group must inherit baseline rules without overrides")
            if self.exception_ids:
                _fail("control group must not configure policy exceptions")
        elif self.quote_tick_constraint is None or self.trade_tick_constraint is None:
            _fail("each test group must explicitly configure quote and trade ticks")
        if not isinstance(self.exception_ids, tuple):
            _fail("group exception_ids must be an immutable tuple")
        normalized = tuple(
            sorted(_identifier(item, "exception_id") for item in self.exception_ids)
        )
        if len(set(normalized)) != len(normalized):
            _fail("group exception_ids must be unique")
        object.__setattr__(self, "exception_ids", normalized)

    def to_mapping(self) -> dict[str, object]:
        return {
            "exception_ids": list(self.exception_ids),
            "group": self.group.value,
            "quote_tick_constraint": (
                None
                if self.quote_tick_constraint is None
                else self.quote_tick_constraint.to_mapping()
            ),
            "trade_at_restriction": (
                None
                if self.trade_at_restriction is None
                else self.trade_at_restriction.to_mapping()
            ),
            "trade_tick_constraint": (
                None
                if self.trade_tick_constraint is None
                else self.trade_tick_constraint.to_mapping()
            ),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        """Parse one exact group mechanism declaration."""
        _strict_fields(
            value,
            frozenset(
                {
                    "group",
                    "quote_tick_constraint",
                    "trade_tick_constraint",
                    "trade_at_restriction",
                    "exception_ids",
                }
            ),
            "group rule",
        )
        quote = value["quote_tick_constraint"]
        trade = value["trade_tick_constraint"]
        trade_at = value["trade_at_restriction"]
        return cls(
            group=_enum_value(value["group"], PolicyGroup, "group"),
            quote_tick_constraint=(
                None
                if quote is None
                else TickConstraint.from_mapping(_mapping(quote, "quote constraint"))
            ),
            trade_tick_constraint=(
                None
                if trade is None
                else TickConstraint.from_mapping(_mapping(trade, "trade constraint"))
            ),
            trade_at_restriction=(
                None
                if trade_at is None
                else TradeAtRestriction.from_mapping(
                    _mapping(trade_at, "trade-at restriction")
                )
            ),
            exception_ids=tuple(
                _identifier(item, "exception_id")
                for item in _sequence(value["exception_ids"], "exception_ids")
            ),
        )


@dataclass(frozen=True, slots=True)
class PolicySourceRecord:
    """Local primary-source metadata; raw text has no field in this schema."""

    source_id: str
    relative_path: str
    event_time: datetime
    available_time: datetime
    ingestion_time: datetime
    source: str
    version: str
    document_id: str
    content_hash: str
    license_tag: str
    evidence: RuleEvidence
    synthetic_only: bool

    def __post_init__(self) -> None:
        _identifier(self.source_id, "source_id")
        object.__setattr__(
            self, "event_time", _aware_utc(self.event_time, "event_time")
        )
        object.__setattr__(
            self, "available_time", _aware_utc(self.available_time, "available_time")
        )
        object.__setattr__(
            self, "ingestion_time", _aware_utc(self.ingestion_time, "ingestion_time")
        )
        if self.ingestion_time < self.available_time:
            _fail("source ingestion_time must not precede available_time")
        _text(self.source, "source", maximum=256)
        _text(self.version, "version", maximum=256)
        _text(self.document_id, "document_id", maximum=256)
        _sha256(self.content_hash, "content_hash")
        _text(self.license_tag, "license_tag", maximum=128)
        if not isinstance(self.evidence, RuleEvidence):
            _fail("source evidence must be RuleEvidence")
        if not isinstance(self.synthetic_only, bool):
            _fail("source synthetic_only must be a boolean")
        if self.synthetic_only != (self.evidence is RuleEvidence.SYNTHETIC_FIXTURE):
            _fail("source synthetic marker and evidence status disagree")
        path = PurePosixPath(self.relative_path)
        if (
            not self.relative_path
            or path.is_absolute()
            or path.as_posix() != self.relative_path
            or any(part in {"", ".", ".."} for part in path.parts)
            or "://" in self.relative_path
            or "\\" in self.relative_path
        ):
            _fail("relative_path must be a safe portable local path")

    def to_mapping(self) -> dict[str, object]:
        return {
            "available_time": self.available_time.isoformat(),
            "content_hash": self.content_hash,
            "document_id": self.document_id,
            "event_time": self.event_time.isoformat(),
            "evidence": self.evidence.value,
            "ingestion_time": self.ingestion_time.isoformat(),
            "license_tag": self.license_tag,
            "relative_path": self.relative_path,
            "source": self.source,
            "source_id": self.source_id,
            "synthetic_only": self.synthetic_only,
            "version": self.version,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(
            value,
            frozenset(
                {
                    "source_id",
                    "relative_path",
                    "event_time",
                    "available_time",
                    "ingestion_time",
                    "source",
                    "version",
                    "document_id",
                    "content_hash",
                    "license_tag",
                    "evidence",
                    "synthetic_only",
                }
            ),
            "policy source record",
        )
        synthetic_only = value["synthetic_only"]
        if not isinstance(synthetic_only, bool):
            _fail("synthetic_only must be a boolean")
        return cls(
            source_id=_identifier(value["source_id"], "source_id"),
            relative_path=_text(value["relative_path"], "relative_path"),
            event_time=_parse_datetime(value["event_time"], "event_time"),
            available_time=_parse_datetime(value["available_time"], "available_time"),
            ingestion_time=_parse_datetime(value["ingestion_time"], "ingestion_time"),
            source=_text(value["source"], "source"),
            version=_text(value["version"], "version"),
            document_id=_text(value["document_id"], "document_id"),
            content_hash=_sha256(value["content_hash"], "content_hash"),
            license_tag=_text(value["license_tag"], "license_tag"),
            evidence=_enum_value(value["evidence"], RuleEvidence, "evidence"),
            synthetic_only=synthetic_only,
        )


@dataclass(frozen=True, slots=True)
class PolicySourceRegistry:
    """Validated local source files under one trusted cutoff."""

    registry_id: str
    records: tuple[PolicySourceRecord, ...]
    manifest_source_sha256: str
    trusted_cutoff: datetime
    synthetic_only: bool
    _verified_local_files: bool = field(default=False, init=False, repr=False)
    _manifest_source_path: Path | None = field(default=None, init=False, repr=False)
    _source_root: Path | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        _identifier(self.registry_id, "registry_id")
        _sha256(self.manifest_source_sha256, "manifest_source_sha256")
        if not isinstance(self.synthetic_only, bool):
            _fail("source registry synthetic_only must be a boolean")
        if not isinstance(self.records, tuple) or not all(
            isinstance(item, PolicySourceRecord) for item in self.records
        ):
            _fail("source registry records must be an immutable record tuple")
        object.__setattr__(
            self, "trusted_cutoff", _aware_utc(self.trusted_cutoff, "trusted_cutoff")
        )
        ordered = tuple(sorted(self.records, key=lambda item: item.source_id))
        if len({item.source_id for item in ordered}) != len(ordered):
            _fail("source registry ids must be unique")
        if any(item.available_time > self.trusted_cutoff for item in ordered):
            raise PolicyEvidenceError("source became available after trusted cutoff")
        if any(item.synthetic_only != self.synthetic_only for item in ordered):
            _fail("source registry cannot mix synthetic and real records")
        object.__setattr__(self, "records", ordered)

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    @property
    def verified_local_files(self) -> bool:
        """Whether the strict local loader verified all committed bytes."""
        return self._verified_local_files

    def to_mapping(self) -> dict[str, object]:
        return {
            "manifest_source_sha256": self.manifest_source_sha256,
            "records": [record.to_mapping() for record in self.records],
            "registry_id": self.registry_id,
            "schema_version": SOURCE_REGISTRY_SCHEMA_VERSION,
            "synthetic_only": self.synthetic_only,
            "trusted_cutoff": self.trusted_cutoff.isoformat(),
        }

    @classmethod
    def from_synthetic_replay_mapping(cls, value: Mapping[str, object]) -> Self:
        """Rebuild only the explicit empty synthetic replay fixture registry.

        Real or source-bearing registries must continue through
        :func:`load_policy_source_registry`, which verifies local primary-source
        bytes.  The replay loader deliberately has no equivalent shortcut.
        """
        _strict_fields(
            value,
            frozenset(
                {
                    "schema_version",
                    "registry_id",
                    "records",
                    "manifest_source_sha256",
                    "trusted_cutoff",
                    "synthetic_only",
                }
            ),
            "synthetic replay source registry",
        )
        if value["schema_version"] != SOURCE_REGISTRY_SCHEMA_VERSION:
            _fail("unsupported policy source registry schema")
        if value["synthetic_only"] is not True:
            raise PolicyEvidenceError(
                "policy replay accepts only explicit synthetic source fixtures"
            )
        records = _sequence(value["records"], "records")
        if records:
            raise PolicyEvidenceError(
                "source-bearing registries require verified local source files"
            )
        return cls(
            registry_id=_identifier(value["registry_id"], "registry_id"),
            records=(),
            manifest_source_sha256=_sha256(
                value["manifest_source_sha256"], "manifest_source_sha256"
            ),
            trusted_cutoff=_parse_datetime(value["trusted_cutoff"], "trusted_cutoff"),
            synthetic_only=True,
        )

    def record(self, source_id: str) -> PolicySourceRecord:
        stable = _identifier(source_id, "source_id")
        for record in self.records:
            if record.source_id == stable:
                return record
        raise PolicyEvidenceError(
            f"policy source is not in the local registry: {stable}"
        )

    def validate_binding(
        self, binding: EvidenceBinding, *, synthetic_only: bool
    ) -> None:
        if not isinstance(binding, EvidenceBinding):
            _fail("source validation requires an EvidenceBinding")
        if synthetic_only != self.synthetic_only:
            raise PolicyEvidenceError(
                "binding mode differs from the source registry synthetic status"
            )
        if binding.source_ids and not self.verified_local_files:
            raise PolicyEvidenceError(
                "source-bound rule requires a registry loaded from verified local files"
            )
        if synthetic_only:
            if binding.status is not RuleEvidence.SYNTHETIC_FIXTURE:
                raise PolicyEvidenceError(
                    "synthetic policy rule lacks synthetic evidence status"
                )
            for source_id in binding.source_ids:
                record = self.record(source_id)
                if (
                    record.evidence is not RuleEvidence.SYNTHETIC_FIXTURE
                    or not record.synthetic_only
                ):
                    raise PolicyEvidenceError(
                        "synthetic policy rule references a non-synthetic source"
                    )
            return
        if binding.status is not RuleEvidence.VERIFIED_PRIMARY:
            raise PolicyEvidenceError(
                "real policy rule is pending primary-source verification"
            )
        for source_id in binding.source_ids:
            record = self.record(source_id)
            if (
                record.evidence is not RuleEvidence.VERIFIED_PRIMARY
                or record.synthetic_only
            ):
                raise PolicyEvidenceError(
                    "real policy rule references an unverified source"
                )

    def assert_source_integrity(self) -> None:
        """Recheck exact registry and source bytes when local sources are bound."""
        if not self.verified_local_files:
            if self.synthetic_only and not self.records:
                return
            raise PolicyEvidenceError(
                "policy source registry was not loaded from verified local files"
            )
        if self._manifest_source_path is None or self._source_root is None:
            raise PolicyEvidenceError("verified source registry lost its local paths")
        _, value = _read_committed_json(
            self._manifest_source_path,
            self.manifest_source_sha256,
            PolicyEvidenceError,
        )
        _strict_fields(
            value,
            frozenset({"schema_version", "registry_id", "synthetic_only", "records"}),
            "policy source registry",
        )
        if (
            value["schema_version"] != SOURCE_REGISTRY_SCHEMA_VERSION
            or value["registry_id"] != self.registry_id
            or value["synthetic_only"] is not self.synthetic_only
        ):
            raise PolicyEvidenceError("policy source registry semantics changed")
        parsed_records = tuple(
            sorted(
                (
                    PolicySourceRecord.from_mapping(_mapping(item, "source record"))
                    for item in _sequence(value["records"], "records")
                ),
                key=lambda item: item.source_id,
            )
        )
        if tuple(item.to_mapping() for item in parsed_records) != tuple(
            item.to_mapping() for item in self.records
        ):
            raise PolicyEvidenceError("policy source registry records changed")
        for record in self.records:
            candidate = _local_absolute(self._source_root / record.relative_path)
            if not candidate.is_relative_to(self._source_root):
                raise PolicyEvidenceError("policy source path escapes its local root")
            _read_committed_bytes(candidate, record.content_hash, PolicyEvidenceError)


def load_policy_source_registry(
    manifest_path: Path,
    *,
    expected_sha256: str,
    trusted_cutoff: datetime,
    source_root: Path | None = None,
) -> PolicySourceRegistry:
    """Load only local hash-matched source files; no network fallback exists."""
    expected = _sha256(expected_sha256, "expected_sha256")
    path, value = _read_committed_json(manifest_path, expected, PolicyEvidenceError)
    _strict_fields(
        value,
        frozenset({"schema_version", "registry_id", "synthetic_only", "records"}),
        "policy source registry",
    )
    if value["schema_version"] != SOURCE_REGISTRY_SCHEMA_VERSION:
        _fail("unsupported policy source registry schema")
    synthetic_only = value["synthetic_only"]
    if not isinstance(synthetic_only, bool):
        _fail("registry synthetic_only must be a boolean")
    records = tuple(
        PolicySourceRecord.from_mapping(_mapping(item, "source record"))
        for item in _sequence(value["records"], "records")
    )
    root = _local_absolute(path.parent if source_root is None else source_root)
    _reject_symlink_components(root, PolicyEvidenceError)
    if not root.is_dir():
        raise PolicyEvidenceError("policy source root is not a local directory")
    for record in records:
        candidate = _local_absolute(root / record.relative_path)
        if not candidate.is_relative_to(root):
            raise PolicyEvidenceError(
                "policy source path escapes or is not a regular file"
            )
        _read_committed_bytes(candidate, record.content_hash, PolicyEvidenceError)
    registry = PolicySourceRegistry(
        registry_id=_identifier(value["registry_id"], "registry_id"),
        records=records,
        manifest_source_sha256=expected,
        trusted_cutoff=_aware_utc(trusted_cutoff, "trusted_cutoff"),
        synthetic_only=synthetic_only,
    )
    object.__setattr__(registry, "_verified_local_files", True)
    object.__setattr__(registry, "_manifest_source_path", path)
    object.__setattr__(registry, "_source_root", root)
    registry.assert_source_integrity()
    return registry


@dataclass(frozen=True, slots=True)
class SecurityAssignment:
    """One immutable assignment interval or a legacy synthetic static row.

    Temporal intervals are left-closed and right-open.  A missing ``deleted_at``
    means the final interval remains open.  Static rows are accepted only by the
    legacy synthetic manifest schema and acquire meaning from a containing
    synthetic policy window at lookup time.
    """

    security_id: str
    group: PolicyGroup
    effective_at: datetime | None = None
    deleted_at: datetime | None = None
    change_reason: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.security_id, "security_id")
        if not isinstance(self.group, PolicyGroup):
            _fail("assignment group must be PolicyGroup")
        if self.effective_at is None:
            if self.deleted_at is not None or self.change_reason is not None:
                raise AssignmentIntegrityError(
                    "legacy static assignment cannot contain temporal fields"
                )
            return
        object.__setattr__(
            self,
            "effective_at",
            _aware_utc(self.effective_at, "assignment effective_at"),
        )
        if self.deleted_at is not None:
            object.__setattr__(
                self,
                "deleted_at",
                _aware_utc(self.deleted_at, "assignment deleted_at"),
            )
            if self.deleted_at <= self.effective_at:
                raise AssignmentIntegrityError(
                    "assignment deleted_at must be after effective_at"
                )
        if self.change_reason is None:
            raise AssignmentIntegrityError(
                "temporal assignment requires a stable change_reason"
            )
        _identifier(self.change_reason, "assignment change_reason")

    @property
    def is_temporal(self) -> bool:
        return self.effective_at is not None

    def contains(self, timestamp: datetime) -> bool:
        """Return whether a timestamp is inside this temporal interval."""
        if self.effective_at is None:
            raise AssignmentIntegrityError(
                "legacy static assignment has no intrinsic time interval"
            )
        value = _aware_utc(timestamp, "assignment lookup timestamp")
        return value >= self.effective_at and (
            self.deleted_at is None or value < self.deleted_at
        )

    def to_mapping(self) -> dict[str, object]:
        if not self.is_temporal:
            return {"group": self.group.value, "security_id": self.security_id}
        if self.effective_at is None or self.change_reason is None:
            raise AssignmentIntegrityError("temporal assignment lost required fields")
        return {
            "change_reason": self.change_reason,
            "deleted_at": (
                None if self.deleted_at is None else self.deleted_at.isoformat()
            ),
            "effective_at": self.effective_at.isoformat(),
            "group": self.group.value,
            "security_id": self.security_id,
        }

    @classmethod
    def from_legacy_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(value, frozenset({"security_id", "group"}), "assignment")
        return cls(
            security_id=_identifier(value["security_id"], "security_id"),
            group=_enum_value(value["group"], PolicyGroup, "group"),
        )

    @classmethod
    def from_temporal_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(
            value,
            frozenset(
                {
                    "security_id",
                    "group",
                    "effective_at",
                    "deleted_at",
                    "change_reason",
                }
            ),
            "assignment history row",
        )
        deleted_at = value["deleted_at"]
        if deleted_at is not None and not isinstance(deleted_at, str):
            _fail("assignment deleted_at must be an ISO 8601 timestamp or null")
        return cls(
            security_id=_identifier(value["security_id"], "security_id"),
            group=_enum_value(value["group"], PolicyGroup, "group"),
            effective_at=_parse_datetime(
                value["effective_at"], "assignment effective_at"
            ),
            deleted_at=(
                None
                if deleted_at is None
                else _parse_datetime(deleted_at, "assignment deleted_at")
            ),
            change_reason=_identifier(
                value["change_reason"], "assignment change_reason"
            ),
        )


@dataclass(frozen=True, slots=True)
class AssignmentResolution:
    """Timestamp-bound result retained in every compliance decision."""

    security_id: str
    timestamp: datetime
    status: AssignmentResolutionStatus
    group: PolicyGroup | None = None
    effective_at: datetime | None = None
    deleted_at: datetime | None = None

    def __post_init__(self) -> None:
        _identifier(self.security_id, "assignment resolution security_id")
        object.__setattr__(
            self,
            "timestamp",
            _aware_utc(self.timestamp, "assignment resolution timestamp"),
        )
        if not isinstance(self.status, AssignmentResolutionStatus):
            _fail("assignment resolution status is invalid")
        if self.group is not None and not isinstance(self.group, PolicyGroup):
            _fail("assignment resolution group must be PolicyGroup or None")
        if self.effective_at is not None:
            object.__setattr__(
                self,
                "effective_at",
                _aware_utc(self.effective_at, "resolution effective_at"),
            )
        if self.deleted_at is not None:
            object.__setattr__(
                self,
                "deleted_at",
                _aware_utc(self.deleted_at, "resolution deleted_at"),
            )
        if self.status is AssignmentResolutionStatus.ASSIGNED:
            if self.group is None or self.effective_at is None:
                _fail("temporal assigned resolution requires group and effective_at")
            if self.timestamp < self.effective_at:
                _fail("assigned resolution cannot precede effective_at")
            if self.deleted_at is not None and self.timestamp >= self.deleted_at:
                _fail("assigned resolution must precede deleted_at")
        elif self.status is AssignmentResolutionStatus.LEGACY_SYNTHETIC:
            if (
                self.group is None
                or self.effective_at is not None
                or self.deleted_at is not None
            ):
                _fail(
                    "legacy synthetic resolution requires group without an "
                    "intrinsic interval"
                )
        elif self.status is AssignmentResolutionStatus.NOT_YET_EFFECTIVE:
            if (
                self.group is not None
                or self.effective_at is None
                or self.deleted_at is not None
            ):
                _fail(
                    "not-yet-effective resolution requires only the first effective_at"
                )
            if self.timestamp >= self.effective_at:
                _fail("not-yet-effective resolution must precede effective_at")
        elif self.status is AssignmentResolutionStatus.DELETED:
            if (
                self.group is not None
                or self.effective_at is not None
                or self.deleted_at is None
            ):
                _fail("deleted resolution requires only the terminal deleted_at")
            if self.timestamp < self.deleted_at:
                _fail("deleted resolution cannot precede deleted_at")
        elif self.status in {
            AssignmentResolutionStatus.UNKNOWN_SECURITY,
            AssignmentResolutionStatus.NOT_EVALUATED,
        }:
            if any(
                value is not None
                for value in (self.group, self.effective_at, self.deleted_at)
            ):
                _fail(
                    "unknown or unevaluated resolution cannot carry assignment fields"
                )
        else:
            _fail("assignment resolution status has no validated state contract")

    @classmethod
    def not_evaluated(cls, security_id: str, timestamp: datetime) -> Self:
        return cls(
            security_id=security_id,
            timestamp=timestamp,
            status=AssignmentResolutionStatus.NOT_EVALUATED,
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "deleted_at": (
                None if self.deleted_at is None else self.deleted_at.isoformat()
            ),
            "effective_at": (
                None if self.effective_at is None else self.effective_at.isoformat()
            ),
            "group": None if self.group is None else self.group.value,
            "security_id": self.security_id,
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class SecurityAssignmentManifest:
    """Canonical static-synthetic or timestamped assignment payload."""

    manifest_id: str
    policy_id: str
    available_time: datetime
    assignments: tuple[SecurityAssignment, ...]
    evidence: EvidenceBinding
    synthetic_only: bool

    def __post_init__(self) -> None:
        _identifier(self.manifest_id, "manifest_id")
        _identifier(self.policy_id, "policy_id")
        object.__setattr__(
            self, "available_time", _aware_utc(self.available_time, "available_time")
        )
        if not isinstance(self.assignments, tuple):
            _fail("assignments must be an immutable tuple")
        if not all(isinstance(item, SecurityAssignment) for item in self.assignments):
            _fail("assignments must contain only SecurityAssignment values")
        if not self.assignments:
            raise AssignmentIntegrityError("assignment manifest must not be empty")
        if not isinstance(self.evidence, EvidenceBinding):
            _fail("assignment manifest requires EvidenceBinding")
        if not isinstance(self.synthetic_only, bool):
            _fail("assignment synthetic_only must be a boolean")
        temporal_modes = {item.is_temporal for item in self.assignments}
        if len(temporal_modes) != 1:
            raise AssignmentIntegrityError(
                "assignment manifest cannot mix static and temporal rows"
            )
        if self.is_legacy_static:
            if not self.synthetic_only:
                raise PolicyEvidenceError(
                    "legacy static assignment schema is synthetic-only"
                )
            ordered = tuple(sorted(self.assignments, key=lambda item: item.security_id))
            if self.assignments != ordered:
                raise AssignmentIntegrityError(
                    "legacy assignment rows must use canonical security order"
                )
            if len({item.security_id for item in ordered}) != len(ordered):
                raise AssignmentIntegrityError(
                    "legacy security assignments contain duplicates"
                )
            return

        ordered_temporal = tuple(
            sorted(
                self.assignments,
                key=lambda item: (
                    item.security_id,
                    cast(datetime, item.effective_at),
                ),
            )
        )
        if self.assignments != ordered_temporal:
            raise AssignmentIntegrityError(
                "assignment history rows must be in canonical chronological order"
            )
        by_security: dict[str, list[SecurityAssignment]] = {}
        for assignment in self.assignments:
            by_security.setdefault(assignment.security_id, []).append(assignment)
        for security_id, history in by_security.items():
            for previous, current in zip(history[:-1], history[1:], strict=True):
                if previous.deleted_at is None:
                    raise AssignmentIntegrityError(
                        f"open assignment interval precedes another row: {security_id}"
                    )
                if current.effective_at is None:
                    raise AssignmentIntegrityError("temporal history lost effective_at")
                if current.effective_at < previous.deleted_at:
                    raise AssignmentIntegrityError(
                        f"assignment history intervals overlap: {security_id}"
                    )
                if current.effective_at > previous.deleted_at:
                    raise AssignmentIntegrityError(
                        f"assignment history contains an internal gap: {security_id}"
                    )

    @property
    def is_legacy_static(self) -> bool:
        return bool(self.assignments) and not self.assignments[0].is_temporal

    @property
    def schema_version(self) -> str:
        return (
            LEGACY_ASSIGNMENT_SCHEMA_VERSION
            if self.is_legacy_static
            else ASSIGNMENT_SCHEMA_VERSION
        )

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def to_mapping(self) -> dict[str, object]:
        mapping: dict[str, object] = {
            "available_time": self.available_time.isoformat(),
            "evidence": self.evidence.to_mapping(),
            "manifest_id": self.manifest_id,
            "policy_id": self.policy_id,
            "schema_version": self.schema_version,
            "synthetic_only": self.synthetic_only,
        }
        mapping["assignments" if self.is_legacy_static else "assignment_history"] = [
            item.to_mapping() for item in self.assignments
        ]
        return mapping

    def resolve(
        self,
        security_id: str,
        timestamp: datetime,
        *,
        legacy_window: PolicyWindow | None = None,
    ) -> AssignmentResolution:
        """Resolve exactly one left-closed/right-open assignment interval."""
        stable = _identifier(security_id, "security_id")
        value = _aware_utc(timestamp, "assignment lookup timestamp")
        history = tuple(
            assignment
            for assignment in self.assignments
            if assignment.security_id == stable
        )
        if not history:
            return AssignmentResolution(
                security_id=stable,
                timestamp=value,
                status=AssignmentResolutionStatus.UNKNOWN_SECURITY,
            )
        if self.is_legacy_static:
            if (
                not self.synthetic_only
                or legacy_window is None
                or not legacy_window.contains(value)
            ):
                raise AssignmentIntegrityError(
                    "legacy static assignment requires a containing synthetic "
                    "policy window"
                )
            return AssignmentResolution(
                security_id=stable,
                timestamp=value,
                status=AssignmentResolutionStatus.LEGACY_SYNTHETIC,
                group=history[0].group,
            )
        first = history[0]
        if first.effective_at is None:
            raise AssignmentIntegrityError("temporal history lost first effective_at")
        if value < first.effective_at:
            return AssignmentResolution(
                security_id=stable,
                timestamp=value,
                status=AssignmentResolutionStatus.NOT_YET_EFFECTIVE,
                effective_at=first.effective_at,
            )
        for assignment in history:
            if assignment.contains(value):
                return AssignmentResolution(
                    security_id=stable,
                    timestamp=value,
                    status=AssignmentResolutionStatus.ASSIGNED,
                    group=assignment.group,
                    effective_at=assignment.effective_at,
                    deleted_at=assignment.deleted_at,
                )
        terminal = history[-1]
        if terminal.deleted_at is not None and value >= terminal.deleted_at:
            return AssignmentResolution(
                security_id=stable,
                timestamp=value,
                status=AssignmentResolutionStatus.DELETED,
                deleted_at=terminal.deleted_at,
            )
        raise AssignmentIntegrityError(
            "validated assignment history did not resolve the requested timestamp"
        )

    def group_for(
        self,
        security_id: str,
        timestamp: datetime,
        *,
        legacy_window: PolicyWindow | None = None,
    ) -> PolicyGroup | None:
        """Return a timestamped group, never an atemporal real assignment."""
        return self.resolve(security_id, timestamp, legacy_window=legacy_window).group

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        schema_version = value.get("schema_version")
        if schema_version == LEGACY_ASSIGNMENT_SCHEMA_VERSION:
            _strict_fields(
                value,
                frozenset(
                    {
                        "schema_version",
                        "manifest_id",
                        "policy_id",
                        "available_time",
                        "evidence",
                        "synthetic_only",
                        "assignments",
                    }
                ),
                "legacy security assignment manifest",
            )
        elif schema_version == ASSIGNMENT_SCHEMA_VERSION:
            _strict_fields(
                value,
                frozenset(
                    {
                        "schema_version",
                        "manifest_id",
                        "policy_id",
                        "available_time",
                        "evidence",
                        "synthetic_only",
                        "assignment_history",
                    }
                ),
                "temporal security assignment manifest",
            )
        else:
            _fail("unsupported security assignment schema")
        synthetic_only = value["synthetic_only"]
        if not isinstance(synthetic_only, bool):
            _fail("assignment synthetic_only must be a boolean")
        if schema_version == LEGACY_ASSIGNMENT_SCHEMA_VERSION and not synthetic_only:
            raise PolicyEvidenceError(
                "legacy static assignment schema is synthetic-only"
            )
        rows = _sequence(
            value[
                "assignments"
                if schema_version == LEGACY_ASSIGNMENT_SCHEMA_VERSION
                else "assignment_history"
            ],
            "assignment rows",
        )
        assignments = tuple(
            (
                SecurityAssignment.from_legacy_mapping(_mapping(raw, "assignment"))
                if schema_version == LEGACY_ASSIGNMENT_SCHEMA_VERSION
                else SecurityAssignment.from_temporal_mapping(
                    _mapping(raw, "assignment history row")
                )
            )
            for raw in rows
        )
        return cls(
            manifest_id=_identifier(value["manifest_id"], "manifest_id"),
            policy_id=_identifier(value["policy_id"], "policy_id"),
            available_time=_parse_datetime(value["available_time"], "available_time"),
            assignments=assignments,
            evidence=EvidenceBinding.from_mapping(
                _mapping(value["evidence"], "evidence")
            ),
            synthetic_only=synthetic_only,
        )


@dataclass(frozen=True, slots=True)
class LoadedSecurityAssignments:
    """Assignment payload plus the exact local file commitment."""

    manifest: SecurityAssignmentManifest
    source_sha256: str
    source_path: Path

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, SecurityAssignmentManifest):
            _fail("loaded assignments require a SecurityAssignmentManifest")
        _sha256(self.source_sha256, "assignment source_sha256")
        if not isinstance(self.source_path, Path):
            _fail("assignment source_path must be a Path")
        resolved, mapping = _read_committed_json(
            self.source_path, self.source_sha256, AssignmentIntegrityError
        )
        parsed = SecurityAssignmentManifest.from_mapping(mapping)
        if parsed.sha256 != self.manifest.sha256:
            raise AssignmentIntegrityError(
                "loaded assignment semantic content differs from its source file"
            )
        object.__setattr__(self, "source_path", resolved)

    def assert_source_integrity(self) -> None:
        """Recheck the assignment bytes before a policy engine accepts them."""
        _, mapping = _read_committed_json(
            self.source_path, self.source_sha256, AssignmentIntegrityError
        )
        parsed = SecurityAssignmentManifest.from_mapping(mapping)
        if parsed.sha256 != self.manifest.sha256:
            raise AssignmentIntegrityError(
                "loaded assignment semantic content differs from its source file"
            )

    def read_source_bytes(self) -> bytes:
        """Return the exact assignment bytes after a fresh descriptor check."""
        _, content = _read_committed_bytes(
            self.source_path, self.source_sha256, AssignmentIntegrityError
        )
        return content


def load_security_assignment_manifest(
    path: Path,
    *,
    expected_sha256: str,
    registry: PolicySourceRegistry,
    trusted_cutoff: datetime,
) -> LoadedSecurityAssignments:
    """Strictly load a local assignment list and verify temporal evidence."""
    expected = _sha256(expected_sha256, "expected_sha256")
    resolved, mapping = _read_committed_json(path, expected, AssignmentIntegrityError)
    manifest = SecurityAssignmentManifest.from_mapping(mapping)
    cutoff = _aware_utc(trusted_cutoff, "trusted_cutoff")
    if cutoff != registry.trusted_cutoff:
        raise PolicyEvidenceError(
            "assignment cutoff differs from the verified source registry cutoff"
        )
    if manifest.available_time > cutoff:
        raise PolicyEvidenceError("assignment became available after trusted cutoff")
    if manifest.synthetic_only != registry.synthetic_only:
        raise PolicyEvidenceError(
            "assignment and source registry synthetic status differ"
        )
    registry.validate_binding(manifest.evidence, synthetic_only=manifest.synthetic_only)
    return LoadedSecurityAssignments(manifest, expected, resolved)


@dataclass(frozen=True, slots=True)
class SecTickSizePilotConfig:
    """Complete immutable policy definition with no real-rule defaults."""

    policy_id: str
    enabled: bool
    trusted_cutoff: datetime
    window: PolicyWindow
    baseline_rules: BaselineMarketRules
    group_rules: tuple[GroupRule, ...]
    exceptions: tuple[DocumentedException, ...]
    source_registry: PolicySourceRegistry
    synthetic_only: bool

    def __post_init__(self) -> None:
        _identifier(self.policy_id, "policy_id")
        if not isinstance(self.enabled, bool) or not isinstance(
            self.synthetic_only, bool
        ):
            _fail("enabled and synthetic_only must be booleans")
        if not isinstance(self.window, PolicyWindow):
            _fail("policy config requires an explicit PolicyWindow")
        if not isinstance(self.baseline_rules, BaselineMarketRules):
            _fail("policy config requires BaselineMarketRules")
        if not isinstance(self.source_registry, PolicySourceRegistry):
            _fail("policy config requires a PolicySourceRegistry")
        if not isinstance(self.group_rules, tuple) or not all(
            isinstance(item, GroupRule) for item in self.group_rules
        ):
            _fail("group_rules must be an immutable GroupRule tuple")
        if not isinstance(self.exceptions, tuple) or not all(
            isinstance(item, DocumentedException) for item in self.exceptions
        ):
            _fail("exceptions must be an immutable DocumentedException tuple")
        object.__setattr__(
            self, "trusted_cutoff", _aware_utc(self.trusted_cutoff, "trusted_cutoff")
        )
        if self.source_registry.trusted_cutoff != self.trusted_cutoff:
            _fail("source registry and policy trusted_cutoff differ")
        if self.source_registry.synthetic_only != self.synthetic_only:
            _fail("source registry and policy synthetic status differ")
        if not self.synthetic_only and not self.source_registry.verified_local_files:
            raise PolicyEvidenceError(
                "real policy config requires a verified local source registry"
            )
        self.source_registry.assert_source_integrity()
        ordered_rules = tuple(
            sorted(self.group_rules, key=lambda item: item.group.value)
        )
        if len(ordered_rules) != len(PolicyGroup) or {
            item.group for item in ordered_rules
        } != set(PolicyGroup):
            _fail(
                "group_rules must define control and all three test groups exactly once"
            )
        object.__setattr__(self, "group_rules", ordered_rules)
        ordered_exceptions = tuple(
            sorted(self.exceptions, key=lambda item: item.exception_id)
        )
        if len({item.exception_id for item in ordered_exceptions}) != len(
            ordered_exceptions
        ):
            _fail("documented exception ids must be unique")
        known_exceptions = {item.exception_id for item in ordered_exceptions}
        if any(
            exception_id not in known_exceptions
            for rule in ordered_rules
            for exception_id in rule.exception_ids
        ):
            _fail("group rule references an undefined documented exception")
        object.__setattr__(self, "exceptions", ordered_exceptions)
        for binding in self._evidence_bindings():
            self.source_registry.validate_binding(
                binding, synthetic_only=self.synthetic_only
            )

    def _evidence_bindings(self) -> tuple[EvidenceBinding, ...]:
        bindings = [
            self.window.evidence,
            self.baseline_rules.quote_tick_constraint.evidence,
            self.baseline_rules.trade_tick_constraint.evidence,
        ]
        for rule in self.group_rules:
            if rule.quote_tick_constraint is not None:
                bindings.append(rule.quote_tick_constraint.evidence)
            if rule.trade_tick_constraint is not None:
                bindings.append(rule.trade_tick_constraint.evidence)
            if rule.trade_at_restriction is not None:
                bindings.append(rule.trade_at_restriction.evidence)
        bindings.extend(item.evidence for item in self.exceptions)
        return tuple(bindings)

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def rule_for(self, group: PolicyGroup) -> GroupRule:
        for rule in self.group_rules:
            if rule.group is group:
                return rule
        raise PolicySchemaError("configured group rule is missing")

    def exception(self, exception_id: str) -> DocumentedException | None:
        for item in self.exceptions:
            if item.exception_id == exception_id:
                return item
        return None

    def to_mapping(self) -> dict[str, object]:
        return {
            "baseline_rules": self.baseline_rules.to_mapping(),
            "enabled": self.enabled,
            "exceptions": [item.to_mapping() for item in self.exceptions],
            "group_rules": [item.to_mapping() for item in self.group_rules],
            "policy_id": self.policy_id,
            "schema_version": POLICY_SCHEMA_VERSION,
            "source_registry_sha256": self.source_registry.sha256,
            "synthetic_only": self.synthetic_only,
            "trusted_cutoff": self.trusted_cutoff.isoformat(),
            "window": self.window.to_mapping(),
        }

    @classmethod
    def from_replay_mapping(
        cls,
        value: Mapping[str, object],
        *,
        source_registry: PolicySourceRegistry,
    ) -> Self:
        """Parse a committed policy definition against one rebuilt registry."""
        _strict_fields(
            value,
            frozenset(
                {
                    "schema_version",
                    "policy_id",
                    "enabled",
                    "trusted_cutoff",
                    "window",
                    "baseline_rules",
                    "group_rules",
                    "exceptions",
                    "source_registry_sha256",
                    "synthetic_only",
                }
            ),
            "SEC Tick Size Pilot replay policy",
        )
        if value["schema_version"] != POLICY_SCHEMA_VERSION:
            _fail("unsupported SEC Tick Size Pilot policy schema")
        if value["synthetic_only"] is not True or not source_registry.synthetic_only:
            raise PolicyEvidenceError(
                "policy replay accepts only explicit synthetic policy fixtures"
            )
        if value["source_registry_sha256"] != source_registry.sha256:
            raise PolicyEvidenceError(
                "policy replay source registry differs from its commitment"
            )
        enabled = value["enabled"]
        if not isinstance(enabled, bool):
            _fail("enabled must be a boolean")
        return cls(
            policy_id=_identifier(value["policy_id"], "policy_id"),
            enabled=enabled,
            trusted_cutoff=_parse_datetime(value["trusted_cutoff"], "trusted_cutoff"),
            window=PolicyWindow.from_mapping(_mapping(value["window"], "window")),
            baseline_rules=BaselineMarketRules.from_mapping(
                _mapping(value["baseline_rules"], "baseline_rules")
            ),
            group_rules=tuple(
                GroupRule.from_mapping(_mapping(item, "group rule"))
                for item in _sequence(value["group_rules"], "group_rules")
            ),
            exceptions=tuple(
                DocumentedException.from_mapping(_mapping(item, "exception"))
                for item in _sequence(value["exceptions"], "exceptions")
            ),
            source_registry=source_registry,
            synthetic_only=True,
        )


@dataclass(frozen=True, slots=True)
class ComplianceRequest:
    """One pre-mutation quote or trade compliance query."""

    event_key: str
    security_id: str
    timestamp: datetime
    action: ComplianceAction
    price_units: int
    exception_id: str | None = None
    trade_at_context_sha256: str | None = None

    def __post_init__(self) -> None:
        _text(self.event_key, "event_key", maximum=256)
        _identifier(self.security_id, "security_id")
        object.__setattr__(self, "timestamp", _aware_utc(self.timestamp, "timestamp"))
        if not isinstance(self.action, ComplianceAction):
            _fail("action must be ComplianceAction")
        _positive_int(self.price_units, "price_units")
        if self.exception_id is not None:
            _identifier(self.exception_id, "exception_id")
        if self.trade_at_context_sha256 is not None:
            _sha256(self.trade_at_context_sha256, "trade_at_context_sha256")

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def to_mapping(self) -> dict[str, object]:
        return {
            "action": self.action.value,
            "event_key": self.event_key,
            "exception_id": self.exception_id,
            "price_units": self.price_units,
            "security_id": self.security_id,
            "timestamp": self.timestamp.isoformat(),
            "trade_at_context_sha256": self.trade_at_context_sha256,
        }


@dataclass(frozen=True, slots=True)
class EvaluationReceipt:
    """Hash-bound result returned by an exception or trade-at evaluator."""

    evaluator_id: str
    subject_id: str
    request_sha256: str
    context_sha256: str | None
    allowed: bool
    proof_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.evaluator_id, "receipt evaluator_id")
        _identifier(self.subject_id, "receipt subject_id")
        _sha256(self.request_sha256, "receipt request_sha256")
        if self.context_sha256 is not None:
            _sha256(self.context_sha256, "receipt context_sha256")
        if not isinstance(self.allowed, bool):
            _fail("receipt allowed must be a boolean")
        _sha256(self.proof_sha256, "receipt proof_sha256")

    def to_mapping(self) -> dict[str, object]:
        return {
            "allowed": self.allowed,
            "context_sha256": self.context_sha256,
            "evaluator_id": self.evaluator_id,
            "proof_sha256": self.proof_sha256,
            "request_sha256": self.request_sha256,
            "subject_id": self.subject_id,
        }

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))


@dataclass(frozen=True, slots=True)
class EvaluationAudit:
    """Non-sensitive audit summary for one evaluator invocation."""

    expected_evaluator_id: str
    expected_subject_id: str
    receipt: EvaluationReceipt | None
    receipt_valid: bool
    allowed: bool | None

    def __post_init__(self) -> None:
        _identifier(self.expected_evaluator_id, "expected_evaluator_id")
        _identifier(self.expected_subject_id, "expected_subject_id")
        if self.receipt is not None and type(self.receipt) is not EvaluationReceipt:
            _fail("evaluation receipt must be EvaluationReceipt or None")
        if not isinstance(self.receipt_valid, bool):
            _fail("receipt_valid must be a boolean")
        if self.allowed is not None and not isinstance(self.allowed, bool):
            _fail("evaluation allowed must be bool or None")
        if self.receipt_valid and (self.receipt is None or self.allowed is None):
            _fail("a valid evaluator receipt requires its hash and decision")
        if (
            self.receipt_valid
            and self.receipt is not None
            and (
                self.receipt.evaluator_id != self.expected_evaluator_id
                or self.receipt.subject_id != self.expected_subject_id
                or self.receipt.allowed is not self.allowed
            )
        ):
            _fail("valid evaluator audit fields disagree with its receipt")
        if not self.receipt_valid and self.allowed is not None:
            _fail("an invalid evaluator receipt cannot carry a decision")

    @property
    def receipt_sha256(self) -> str | None:
        return None if self.receipt is None else self.receipt.sha256

    def to_mapping(self) -> dict[str, object]:
        return {
            "allowed": self.allowed,
            "expected_evaluator_id": self.expected_evaluator_id,
            "expected_subject_id": self.expected_subject_id,
            "receipt": None if self.receipt is None else self.receipt.to_mapping(),
            "receipt_sha256": self.receipt_sha256,
            "receipt_valid": self.receipt_valid,
        }


class DocumentedExceptionEvaluator(Protocol):
    """Project-specific proof that one documented exception applies."""

    def evaluate(
        self, exception: DocumentedException, request: ComplianceRequest
    ) -> EvaluationReceipt: ...


class TradeAtRestrictionEvaluator(Protocol):
    """Project-specific implementation of source-verified trade-at semantics."""

    def evaluate(
        self, restriction: TradeAtRestriction, request: ComplianceRequest
    ) -> EvaluationReceipt: ...


@dataclass(frozen=True, slots=True)
class SyntheticFixedEvaluator:
    """Closed deterministic evaluator for explicitly synthetic policy fixtures.

    This is deliberately the only evaluator implementation accepted by the
    current policy engine.  It prevents a duck-typed callback from blessing a
    restricted action with a caller-authored ``proof_sha256``.  Real policy
    execution remains blocked until a separately verified, primary-source-
    bound evaluator authority exists.
    """

    role: str
    decisions: tuple[tuple[str, bool], ...]
    proof_domain: str
    schema_version: str = SYNTHETIC_EVALUATOR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.role not in {"documented_exception", "trade_at"}:
            _fail("synthetic evaluator role is outside the closed allowlist")
        if not isinstance(self.decisions, tuple) or not self.decisions:
            _fail("synthetic evaluator requires an immutable decision tuple")
        normalized: list[tuple[str, bool]] = []
        for item in self.decisions:
            if not isinstance(item, tuple) or len(item) != 2:
                _fail("synthetic evaluator decisions must be (evaluator_id, allowed)")
            evaluator_id, allowed = item
            normalized.append(
                (
                    _identifier(evaluator_id, "synthetic evaluator_id"),
                    allowed
                    if isinstance(allowed, bool)
                    else _fail("synthetic evaluator decision must be boolean"),
                )
            )
        ordered = tuple(sorted(normalized, key=lambda item: item[0]))
        if len({item[0] for item in ordered}) != len(ordered):
            _fail("synthetic evaluator ids must be unique")
        object.__setattr__(self, "decisions", ordered)
        _identifier(self.proof_domain, "synthetic evaluator proof_domain")
        if self.schema_version != SYNTHETIC_EVALUATOR_SCHEMA_VERSION:
            _fail("unsupported synthetic evaluator schema version")

    def to_mapping(self) -> dict[str, object]:
        """Return the same frozen descriptor committed by market replay."""
        return {
            "decisions": [
                {"allowed": allowed, "evaluator_id": evaluator_id}
                for evaluator_id, allowed in self.decisions
            ],
            "mode": "fixed_receipt",
            "proof_domain": self.proof_domain,
            "role": self.role,
            "schema_version": self.schema_version,
        }

    @property
    def descriptor_sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def _subject_identity(
        self, subject: DocumentedException | TradeAtRestriction
    ) -> tuple[str, str]:
        if self.role == "documented_exception" and type(subject) is DocumentedException:
            return subject.evaluator_id, subject.exception_id
        if self.role == "trade_at" and type(subject) is TradeAtRestriction:
            return subject.evaluator_id, subject.restriction_id
        _fail("synthetic evaluator received a subject outside its frozen role")

    def proof_sha256(
        self,
        subject: DocumentedException | TradeAtRestriction,
        request: ComplianceRequest,
    ) -> str:
        evaluator_id, _ = self._subject_identity(subject)
        decisions = dict(self.decisions)
        if evaluator_id not in decisions:
            _fail("synthetic evaluator descriptor does not cover evaluator_id")
        return sha256_bytes(
            canonical_json_bytes(
                {
                    "allowed": decisions[evaluator_id],
                    "descriptor_sha256": self.descriptor_sha256,
                    "proof_domain": self.proof_domain,
                    "request": request.to_mapping(),
                    "subject": subject.to_mapping(),
                }
            )
        )

    def evaluate(
        self,
        subject: DocumentedException | TradeAtRestriction,
        request: ComplianceRequest,
    ) -> EvaluationReceipt:
        evaluator_id, subject_id = self._subject_identity(subject)
        decisions = dict(self.decisions)
        if evaluator_id not in decisions:
            _fail("synthetic evaluator descriptor does not cover evaluator_id")
        return EvaluationReceipt(
            evaluator_id=evaluator_id,
            subject_id=subject_id,
            request_sha256=request.sha256,
            context_sha256=request.trade_at_context_sha256,
            allowed=decisions[evaluator_id],
            proof_sha256=self.proof_sha256(subject, request),
        )


def _evaluation_audit(
    receipt: object,
    *,
    evaluator: object,
    subject: DocumentedException | TradeAtRestriction,
    evaluator_id: str,
    subject_id: str,
    request: ComplianceRequest,
    synthetic_only: bool,
) -> EvaluationAudit:
    typed_receipt = receipt if type(receipt) is EvaluationReceipt else None
    trusted_evaluator = (
        evaluator
        if type(evaluator) is SyntheticFixedEvaluator
        and isinstance(evaluator, SyntheticFixedEvaluator)
        else None
    )
    valid = (
        synthetic_only
        and trusted_evaluator is not None
        and typed_receipt is not None
        and typed_receipt.evaluator_id == evaluator_id
        and typed_receipt.subject_id == subject_id
        and typed_receipt.request_sha256 == request.sha256
        and typed_receipt.context_sha256 == request.trade_at_context_sha256
        and typed_receipt.allowed == dict(trusted_evaluator.decisions).get(evaluator_id)
        and typed_receipt.proof_sha256
        == trusted_evaluator.proof_sha256(subject, request)
    )
    return EvaluationAudit(
        expected_evaluator_id=evaluator_id,
        expected_subject_id=subject_id,
        receipt=typed_receipt,
        receipt_valid=valid,
        allowed=typed_receipt.allowed if valid and typed_receipt is not None else None,
    )


@dataclass(frozen=True, slots=True)
class ComplianceDecision:
    """Policy-mechanics decision that cannot authorize formal SEC execution."""

    outcome: ComplianceOutcome
    rule_code: ComplianceRuleCode
    request: ComplianceRequest
    action: ComplianceAction
    security_id: str
    group: PolicyGroup | None
    assignment_resolution: AssignmentResolution
    policy_applied: bool
    baseline_used: bool
    required_increment_units: int
    exception_id: str | None
    trade_at_evaluated: bool
    evaluation_audits: tuple[EvaluationAudit, ...]
    policy_sha256: str
    assignment_sha256: str
    assignment_source_sha256: str
    source_registry_sha256: str
    input_sha256: str
    claim_scope: Literal["synthetic_policy_rule_mechanics_non_authorizing"] = (
        POLICY_MECHANICS_CLAIM_SCOPE
    )
    formal_execution_evidence: Literal[False] = False

    def __post_init__(self) -> None:
        if self.claim_scope != POLICY_MECHANICS_CLAIM_SCOPE:
            _fail("compliance decision cannot expand its non-authorizing claim scope")
        if self.formal_execution_evidence is not False:
            _fail("low-level policy decision cannot be formal execution evidence")
        if not isinstance(self.outcome, ComplianceOutcome):
            _fail("decision outcome must be ComplianceOutcome")
        if not isinstance(self.rule_code, ComplianceRuleCode):
            _fail("decision rule_code must be ComplianceRuleCode")
        if not isinstance(self.request, ComplianceRequest):
            _fail("decision request must be ComplianceRequest")
        if not isinstance(self.action, ComplianceAction):
            _fail("decision action must be ComplianceAction")
        _identifier(self.security_id, "decision security_id")
        if self.group is not None and not isinstance(self.group, PolicyGroup):
            _fail("decision group must be PolicyGroup or None")
        if not isinstance(self.assignment_resolution, AssignmentResolution):
            _fail("decision requires an immutable assignment resolution")
        if (
            self.assignment_resolution.security_id != self.security_id
            or self.assignment_resolution.timestamp != self.request.timestamp
            or self.assignment_resolution.group is not self.group
        ):
            _fail("decision assignment resolution disagrees with its request or group")
        if not isinstance(self.policy_applied, bool) or not isinstance(
            self.baseline_used, bool
        ):
            _fail("decision policy flags must be booleans")
        if self.policy_applied == self.baseline_used:
            _fail("exactly one of policy_applied and baseline_used must be true")
        if self.policy_applied and self.group in {None, PolicyGroup.CONTROL}:
            _fail("policy cannot be applied to control or an unassigned security")
        _positive_int(self.required_increment_units, "required_increment_units")
        if self.exception_id is not None:
            _identifier(self.exception_id, "decision exception_id")
        if not isinstance(self.trade_at_evaluated, bool):
            _fail("trade_at_evaluated must be a boolean")
        if not isinstance(self.evaluation_audits, tuple) or not all(
            isinstance(item, EvaluationAudit) for item in self.evaluation_audits
        ):
            _fail("evaluation_audits must be an immutable EvaluationAudit tuple")
        audit_keys = tuple(
            (item.expected_evaluator_id, item.expected_subject_id)
            for item in self.evaluation_audits
        )
        if len(set(audit_keys)) != len(audit_keys):
            _fail("evaluation_audits must not contain duplicate evaluator subjects")
        for evaluation in self.evaluation_audits:
            if evaluation.receipt_valid:
                receipt = evaluation.receipt
                if receipt is None or (
                    receipt.request_sha256 != self.request.sha256
                    or receipt.context_sha256 != self.request.trade_at_context_sha256
                ):
                    _fail("valid evaluator receipt is not bound to this request")
        _sha256(self.policy_sha256, "policy_sha256")
        _sha256(self.assignment_sha256, "assignment_sha256")
        _sha256(self.assignment_source_sha256, "assignment_source_sha256")
        _sha256(self.source_registry_sha256, "source_registry_sha256")
        _sha256(self.input_sha256, "input_sha256")
        if (
            self.request.action is not self.action
            or self.request.security_id != self.security_id
            or self.request.sha256 != self.input_sha256
        ):
            _fail("decision fields do not match the committed compliance request")
        allowed_codes = {
            ComplianceRuleCode.OK,
            ComplianceRuleCode.EXCEPTION_ALLOWED,
            ComplianceRuleCode.TRADE_AT_ALLOWED,
        }
        no_op_codes = {
            ComplianceRuleCode.NO_OP_DISABLED,
            ComplianceRuleCode.OUTSIDE_POLICY_WINDOW,
            ComplianceRuleCode.UNASSIGNED_SECURITY,
            ComplianceRuleCode.CONTROL_BASELINE,
        }
        expected_outcome = (
            ComplianceOutcome.ALLOWED
            if self.rule_code in allowed_codes
            else (
                ComplianceOutcome.NO_OP
                if self.rule_code in no_op_codes
                else ComplianceOutcome.REJECTED
            )
        )
        if self.outcome is not expected_outcome:
            _fail("decision outcome disagrees with its compliance rule code")
        if (self.rule_code is ComplianceRuleCode.EXCEPTION_ALLOWED) != (
            self.exception_id is not None
        ):
            _fail("only an allowed documented exception may set exception_id")

    def to_mapping(self) -> dict[str, object]:
        return {
            "action": self.action.value,
            "assignment_resolution": self.assignment_resolution.to_mapping(),
            "assignment_source_sha256": self.assignment_source_sha256,
            "assignment_sha256": self.assignment_sha256,
            "baseline_used": self.baseline_used,
            "claim_scope": self.claim_scope,
            "exception_id": self.exception_id,
            "evaluation_audits": [item.to_mapping() for item in self.evaluation_audits],
            "formal_execution_evidence": self.formal_execution_evidence,
            "group": None if self.group is None else self.group.value,
            "input_sha256": self.input_sha256,
            "outcome": self.outcome.value,
            "policy_applied": self.policy_applied,
            "policy_sha256": self.policy_sha256,
            "request": self.request.to_mapping(),
            "required_increment_units": self.required_increment_units,
            "rule_code": self.rule_code.value,
            "security_id": self.security_id,
            "source_registry_sha256": self.source_registry_sha256,
            "trade_at_evaluated": self.trade_at_evaluated,
        }

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))


@dataclass(frozen=True, slots=True)
class PolicyComplianceAuditEvent:
    """One deterministic hash-chained compliance decision."""

    sequence: int
    decision: ComplianceDecision
    previous_event_sha256: str
    event_sha256: str

    def __post_init__(self) -> None:
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int):
            _fail("audit sequence must be an integer")
        if self.sequence < 0:
            _fail("audit sequence must be non-negative")
        if type(self.decision) is not ComplianceDecision:
            _fail("audit event requires an immutable ComplianceDecision")
        _sha256(self.previous_event_sha256, "previous_event_sha256")
        _sha256(self.event_sha256, "event_sha256")
        if self.event_sha256 != sha256_bytes(canonical_json_bytes(self.core_mapping())):
            _fail("audit event hash does not match its content")

    def core_mapping(self) -> dict[str, object]:
        return {
            "decision": self.decision.to_mapping(),
            "event_type": "policy_compliance",
            "previous_event_sha256": self.previous_event_sha256,
            "schema_version": COMPLIANCE_AUDIT_SCHEMA_VERSION,
            "sequence": self.sequence,
        }

    def to_mapping(self) -> dict[str, object]:
        return {**self.core_mapping(), "event_sha256": self.event_sha256}


class PolicyComplianceAuditLog:
    """In-memory scientific ledger with deterministic JSONL export."""

    def __init__(self) -> None:
        self._events: list[PolicyComplianceAuditEvent] = []

    @property
    def events(self) -> tuple[PolicyComplianceAuditEvent, ...]:
        return tuple(self._events)

    def record(self, decision: ComplianceDecision) -> PolicyComplianceAuditEvent:
        if type(decision) is not ComplianceDecision:
            _fail("audit log accepts only an immutable ComplianceDecision")
        previous = self._events[-1].event_sha256 if self._events else _ZERO_SHA256
        sequence = len(self._events)
        core = {
            "decision": decision.to_mapping(),
            "event_type": "policy_compliance",
            "previous_event_sha256": previous,
            "schema_version": COMPLIANCE_AUDIT_SCHEMA_VERSION,
            "sequence": sequence,
        }
        event = PolicyComplianceAuditEvent(
            sequence=sequence,
            decision=decision,
            previous_event_sha256=previous,
            event_sha256=sha256_bytes(canonical_json_bytes(core)),
        )
        self._events.append(event)
        return event

    def assert_integrity(self) -> None:
        previous = _ZERO_SHA256
        for sequence, event in enumerate(self._events):
            if (
                event.sequence != sequence
                or event.previous_event_sha256 != previous
                or event.event_sha256
                != sha256_bytes(canonical_json_bytes(event.core_mapping()))
            ):
                raise PolicySchemaError("policy compliance audit chain is invalid")
            previous = event.event_sha256

    def write_jsonl(self, path: Path) -> None:
        self.assert_integrity()
        with JsonlEventWriter(path) as writer:
            for event in self._events:
                writer.write(event.to_mapping())


@dataclass(frozen=True, slots=True, init=False)
class SecTickSizePilotPolicy:
    """Pre-mutation compliance engine for one evidence-bound configuration."""

    config: SecTickSizePilotConfig
    assignments: LoadedSecurityAssignments
    audit_log: PolicyComplianceAuditLog = field(repr=False)

    def __init__(
        self,
        config: SecTickSizePilotConfig,
        assignments: LoadedSecurityAssignments,
        *,
        audit_log: PolicyComplianceAuditLog | None = None,
    ) -> None:
        if not isinstance(config, SecTickSizePilotConfig):
            _fail("policy engine requires SecTickSizePilotConfig")
        if not isinstance(assignments, LoadedSecurityAssignments):
            _fail("policy engine requires strictly loaded security assignments")
        config.source_registry.assert_source_integrity()
        assignments.assert_source_integrity()
        if assignments.manifest.policy_id != config.policy_id:
            _fail("assignment manifest policy_id differs from policy config")
        if assignments.manifest.available_time > config.trusted_cutoff:
            raise PolicyEvidenceError(
                "assignment became available after trusted cutoff"
            )
        if assignments.manifest.synthetic_only != config.synthetic_only:
            _fail("assignment and policy synthetic status differ")
        config.source_registry.validate_binding(
            assignments.manifest.evidence, synthetic_only=config.synthetic_only
        )
        if audit_log is not None and type(audit_log) is not PolicyComplianceAuditLog:
            _fail("audit_log must be an unmodified PolicyComplianceAuditLog")
        if audit_log is not None and audit_log.events:
            _fail("policy engine requires an empty, exclusive audit log")
        object.__setattr__(self, "config", config)
        object.__setattr__(self, "assignments", assignments)
        object.__setattr__(
            self,
            "audit_log",
            PolicyComplianceAuditLog() if audit_log is None else audit_log,
        )

    def evaluate(
        self,
        request: ComplianceRequest,
        *,
        exception_evaluator: DocumentedExceptionEvaluator | None = None,
        trade_at_evaluator: TradeAtRestrictionEvaluator | None = None,
    ) -> ComplianceDecision:
        """Evaluate and audit one quote or trade without mutating a market."""
        if type(request) is not ComplianceRequest:
            _fail("policy evaluation requires an immutable ComplianceRequest")
        self.config.source_registry.assert_source_integrity()
        self.assignments.assert_source_integrity()
        assignment_resolution = self._resolve_assignment(request)
        group = assignment_resolution.group
        baseline_used = False
        policy_applied = False
        fallback_rule: ComplianceRuleCode | None = None
        if not self.config.enabled:
            baseline_used = True
            fallback_rule = ComplianceRuleCode.NO_OP_DISABLED
        elif not self.config.window.contains(request.timestamp):
            baseline_used = True
            fallback_rule = ComplianceRuleCode.OUTSIDE_POLICY_WINDOW
        elif group is None:
            baseline_used = True
            fallback_rule = ComplianceRuleCode.UNASSIGNED_SECURITY
        elif group is PolicyGroup.CONTROL:
            baseline_used = True
            fallback_rule = ComplianceRuleCode.CONTROL_BASELINE
        else:
            policy_applied = True

        if baseline_used:
            constraint = (
                self.config.baseline_rules.quote_tick_constraint
                if request.action is ComplianceAction.QUOTE
                else self.config.baseline_rules.trade_tick_constraint
            )
            group_rule = None
        else:
            if group is None:
                raise PolicySchemaError("active policy resolution lost its group")
            group_rule = self.config.rule_for(group)
            configured = (
                group_rule.quote_tick_constraint
                if request.action is ComplianceAction.QUOTE
                else group_rule.trade_tick_constraint
            )
            if configured is None:
                raise PolicySchemaError("active test group lacks a tick constraint")
            constraint = configured

        if request.exception_id is not None:
            exception = self.config.exception(request.exception_id)
            exception_is_configured = (
                exception is not None
                and group is not None
                and group_rule is not None
                and request.exception_id in group_rule.exception_ids
                and group in exception.groups
            )
            if not exception_is_configured:
                return self._finish(
                    request,
                    group=group,
                    policy_applied=policy_applied,
                    baseline_used=baseline_used,
                    required_increment=constraint.increment_units,
                    outcome=ComplianceOutcome.REJECTED,
                    rule_code=ComplianceRuleCode.EXCEPTION_NOT_CONFIGURED,
                )

        if not constraint.allows(request.price_units):
            scope = (
                ExceptionScope.QUOTE_TICK
                if request.action is ComplianceAction.QUOTE
                else ExceptionScope.TRADE_TICK
            )
            exception_result = self._evaluate_exception(
                request,
                group=group,
                scope=scope,
                group_rule=group_rule,
                evaluator=exception_evaluator,
            )
            if exception_result is not None:
                return exception_result
            return self._finish(
                request,
                group=group,
                policy_applied=policy_applied,
                baseline_used=baseline_used,
                required_increment=constraint.increment_units,
                outcome=ComplianceOutcome.REJECTED,
                rule_code=(
                    ComplianceRuleCode.OFF_QUOTE_GRID
                    if request.action is ComplianceAction.QUOTE
                    else ComplianceRuleCode.OFF_TRADE_GRID
                ),
            )

        if (
            policy_applied
            and request.action is ComplianceAction.TRADE
            and group_rule is not None
            and group_rule.trade_at_restriction is not None
        ):
            restriction = group_rule.trade_at_restriction
            if restriction.context_required and request.trade_at_context_sha256 is None:
                return self._finish(
                    request,
                    group=group,
                    policy_applied=True,
                    baseline_used=False,
                    required_increment=constraint.increment_units,
                    outcome=ComplianceOutcome.REJECTED,
                    rule_code=ComplianceRuleCode.TRADE_AT_CONTEXT_REQUIRED,
                )
            if trade_at_evaluator is None:
                return self._finish(
                    request,
                    group=group,
                    policy_applied=True,
                    baseline_used=False,
                    required_increment=constraint.increment_units,
                    outcome=ComplianceOutcome.REJECTED,
                    rule_code=ComplianceRuleCode.TRADE_AT_EVALUATOR_REQUIRED,
                )
            try:
                receipt: object = trade_at_evaluator.evaluate(restriction, request)
            except Exception:
                failed_audit = EvaluationAudit(
                    expected_evaluator_id=restriction.evaluator_id,
                    expected_subject_id=restriction.restriction_id,
                    receipt=None,
                    receipt_valid=False,
                    allowed=None,
                )
                return self._finish(
                    request,
                    group=group,
                    policy_applied=True,
                    baseline_used=False,
                    required_increment=constraint.increment_units,
                    outcome=ComplianceOutcome.REJECTED,
                    rule_code=ComplianceRuleCode.EVALUATOR_FAILED,
                    trade_at_evaluated=True,
                    evaluation_audits=(failed_audit,),
                )
            trade_at_audit = _evaluation_audit(
                receipt,
                evaluator=trade_at_evaluator,
                subject=restriction,
                evaluator_id=restriction.evaluator_id,
                subject_id=restriction.restriction_id,
                request=request,
                synthetic_only=self.config.synthetic_only,
            )
            if not trade_at_audit.receipt_valid:
                return self._finish(
                    request,
                    group=group,
                    policy_applied=True,
                    baseline_used=False,
                    required_increment=constraint.increment_units,
                    outcome=ComplianceOutcome.REJECTED,
                    rule_code=ComplianceRuleCode.EVALUATOR_RECEIPT_INVALID,
                    trade_at_evaluated=True,
                    evaluation_audits=(trade_at_audit,),
                )
            if not trade_at_audit.allowed:
                exception_result = self._evaluate_exception(
                    request,
                    group=group,
                    scope=ExceptionScope.TRADE_AT,
                    group_rule=group_rule,
                    evaluator=exception_evaluator,
                    prior_audits=(trade_at_audit,),
                )
                if exception_result is not None:
                    return exception_result
                return self._finish(
                    request,
                    group=group,
                    policy_applied=True,
                    baseline_used=False,
                    required_increment=constraint.increment_units,
                    outcome=ComplianceOutcome.REJECTED,
                    rule_code=ComplianceRuleCode.TRADE_AT_REJECTED,
                    trade_at_evaluated=True,
                    evaluation_audits=(trade_at_audit,),
                )
            if request.exception_id is not None:
                return self._finish(
                    request,
                    group=group,
                    policy_applied=True,
                    baseline_used=False,
                    required_increment=constraint.increment_units,
                    outcome=ComplianceOutcome.REJECTED,
                    rule_code=ComplianceRuleCode.EXCEPTION_NOT_APPLICABLE,
                    trade_at_evaluated=True,
                    evaluation_audits=(trade_at_audit,),
                )
            return self._finish(
                request,
                group=group,
                policy_applied=True,
                baseline_used=False,
                required_increment=constraint.increment_units,
                outcome=ComplianceOutcome.ALLOWED,
                rule_code=ComplianceRuleCode.TRADE_AT_ALLOWED,
                trade_at_evaluated=True,
                evaluation_audits=(trade_at_audit,),
            )

        if request.exception_id is not None:
            return self._finish(
                request,
                group=group,
                policy_applied=policy_applied,
                baseline_used=baseline_used,
                required_increment=constraint.increment_units,
                outcome=ComplianceOutcome.REJECTED,
                rule_code=ComplianceRuleCode.EXCEPTION_NOT_APPLICABLE,
            )

        return self._finish(
            request,
            group=group,
            policy_applied=policy_applied,
            baseline_used=baseline_used,
            required_increment=constraint.increment_units,
            outcome=(
                ComplianceOutcome.NO_OP if baseline_used else ComplianceOutcome.ALLOWED
            ),
            rule_code=fallback_rule or ComplianceRuleCode.OK,
        )

    def _evaluate_exception(
        self,
        request: ComplianceRequest,
        *,
        group: PolicyGroup | None,
        scope: ExceptionScope,
        group_rule: GroupRule | None,
        evaluator: DocumentedExceptionEvaluator | None,
        prior_audits: tuple[EvaluationAudit, ...] = (),
    ) -> ComplianceDecision | None:
        if request.exception_id is None:
            return None
        exception = self.config.exception(request.exception_id)
        configured = (
            exception is not None
            and group is not None
            and group_rule is not None
            and request.exception_id in group_rule.exception_ids
            and group in exception.groups
            and scope in exception.scopes
        )
        if not configured or exception is None:
            return self._finish(
                request,
                group=group,
                policy_applied=group_rule is not None,
                baseline_used=group_rule is None,
                required_increment=self._required_increment(request, group, group_rule),
                outcome=ComplianceOutcome.REJECTED,
                rule_code=ComplianceRuleCode.EXCEPTION_NOT_CONFIGURED,
                evaluation_audits=prior_audits,
            )
        if evaluator is None:
            return self._finish(
                request,
                group=group,
                policy_applied=True,
                baseline_used=False,
                required_increment=self._required_increment(request, group, group_rule),
                outcome=ComplianceOutcome.REJECTED,
                rule_code=ComplianceRuleCode.EXCEPTION_EVALUATOR_REQUIRED,
                evaluation_audits=prior_audits,
            )
        try:
            receipt: object = evaluator.evaluate(exception, request)
        except Exception:
            failed_audit = EvaluationAudit(
                expected_evaluator_id=exception.evaluator_id,
                expected_subject_id=exception.exception_id,
                receipt=None,
                receipt_valid=False,
                allowed=None,
            )
            return self._finish(
                request,
                group=group,
                policy_applied=True,
                baseline_used=False,
                required_increment=self._required_increment(request, group, group_rule),
                outcome=ComplianceOutcome.REJECTED,
                rule_code=ComplianceRuleCode.EVALUATOR_FAILED,
                trade_at_evaluated=scope is ExceptionScope.TRADE_AT,
                evaluation_audits=(*prior_audits, failed_audit),
            )
        exception_audit = _evaluation_audit(
            receipt,
            evaluator=evaluator,
            subject=exception,
            evaluator_id=exception.evaluator_id,
            subject_id=exception.exception_id,
            request=request,
            synthetic_only=self.config.synthetic_only,
        )
        if not exception_audit.receipt_valid:
            return self._finish(
                request,
                group=group,
                policy_applied=True,
                baseline_used=False,
                required_increment=self._required_increment(request, group, group_rule),
                outcome=ComplianceOutcome.REJECTED,
                rule_code=ComplianceRuleCode.EVALUATOR_RECEIPT_INVALID,
                trade_at_evaluated=scope is ExceptionScope.TRADE_AT,
                evaluation_audits=(*prior_audits, exception_audit),
            )
        if not exception_audit.allowed:
            return self._finish(
                request,
                group=group,
                policy_applied=True,
                baseline_used=False,
                required_increment=self._required_increment(request, group, group_rule),
                outcome=ComplianceOutcome.REJECTED,
                rule_code=ComplianceRuleCode.EXCEPTION_DENIED,
                trade_at_evaluated=scope is ExceptionScope.TRADE_AT,
                evaluation_audits=(*prior_audits, exception_audit),
            )
        return self._finish(
            request,
            group=group,
            policy_applied=True,
            baseline_used=False,
            required_increment=self._required_increment(request, group, group_rule),
            outcome=ComplianceOutcome.ALLOWED,
            rule_code=ComplianceRuleCode.EXCEPTION_ALLOWED,
            exception_id=exception.exception_id,
            trade_at_evaluated=scope is ExceptionScope.TRADE_AT,
            evaluation_audits=(*prior_audits, exception_audit),
        )

    def _resolve_assignment(self, request: ComplianceRequest) -> AssignmentResolution:
        if not self.config.window.contains(request.timestamp):
            return AssignmentResolution.not_evaluated(
                request.security_id, request.timestamp
            )
        return self.assignments.manifest.resolve(
            request.security_id,
            request.timestamp,
            legacy_window=(
                self.config.window
                if self.assignments.manifest.is_legacy_static
                else None
            ),
        )

    def _required_increment(
        self,
        request: ComplianceRequest,
        group: PolicyGroup | None,
        group_rule: GroupRule | None,
    ) -> int:
        del group
        if group_rule is None:
            constraint = (
                self.config.baseline_rules.quote_tick_constraint
                if request.action is ComplianceAction.QUOTE
                else self.config.baseline_rules.trade_tick_constraint
            )
        else:
            configured = (
                group_rule.quote_tick_constraint
                if request.action is ComplianceAction.QUOTE
                else group_rule.trade_tick_constraint
            )
            if configured is None:
                raise PolicySchemaError("group constraint is missing")
            constraint = configured
        return constraint.increment_units

    def _finish(
        self,
        request: ComplianceRequest,
        *,
        group: PolicyGroup | None,
        policy_applied: bool,
        baseline_used: bool,
        required_increment: int,
        outcome: ComplianceOutcome,
        rule_code: ComplianceRuleCode,
        exception_id: str | None = None,
        trade_at_evaluated: bool = False,
        evaluation_audits: tuple[EvaluationAudit, ...] = (),
    ) -> ComplianceDecision:
        decision = ComplianceDecision(
            outcome=outcome,
            rule_code=rule_code,
            request=request,
            action=request.action,
            security_id=request.security_id,
            group=group,
            assignment_resolution=self._resolve_assignment(request),
            policy_applied=policy_applied,
            baseline_used=baseline_used,
            required_increment_units=required_increment,
            exception_id=exception_id,
            trade_at_evaluated=trade_at_evaluated,
            evaluation_audits=evaluation_audits,
            policy_sha256=self.config.sha256,
            assignment_sha256=self.assignments.manifest.sha256,
            assignment_source_sha256=self.assignments.source_sha256,
            source_registry_sha256=self.config.source_registry.sha256,
            input_sha256=request.sha256,
        )
        self.audit_log.record(decision)
        return decision


__all__ = [
    "ASSIGNMENT_SCHEMA_VERSION",
    "COMPLIANCE_AUDIT_SCHEMA_VERSION",
    "LEGACY_ASSIGNMENT_SCHEMA_VERSION",
    "POLICY_SCHEMA_VERSION",
    "POLICY_MECHANICS_CLAIM_SCOPE",
    "SOURCE_REGISTRY_SCHEMA_VERSION",
    "SYNTHETIC_EVALUATOR_SCHEMA_VERSION",
    "AssignmentIntegrityError",
    "AssignmentResolution",
    "AssignmentResolutionStatus",
    "BaselineMarketRules",
    "ComplianceAction",
    "ComplianceDecision",
    "ComplianceOutcome",
    "ComplianceRequest",
    "ComplianceRuleCode",
    "DocumentedException",
    "DocumentedExceptionEvaluator",
    "EvidenceBinding",
    "EvaluationAudit",
    "EvaluationReceipt",
    "ExceptionScope",
    "GroupRule",
    "LoadedSecurityAssignments",
    "PolicyComplianceAuditEvent",
    "PolicyComplianceAuditLog",
    "PolicyEvidenceError",
    "PolicyGroup",
    "PolicySchemaError",
    "PolicySourceRecord",
    "PolicySourceRegistry",
    "PolicyWindow",
    "RuleEvidence",
    "SecTickSizePilotConfig",
    "SecTickSizePilotPolicy",
    "SecurityAssignment",
    "SecurityAssignmentManifest",
    "SyntheticFixedEvaluator",
    "TickConstraint",
    "TradeAtRestriction",
    "TradeAtRestrictionEvaluator",
    "load_policy_source_registry",
    "load_security_assignment_manifest",
]
