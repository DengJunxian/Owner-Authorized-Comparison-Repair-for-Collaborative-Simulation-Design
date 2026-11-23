"""Fail-closed SEC Tick Size Pilot study-design contracts.

This module separates research-design bookkeeping from the executable policy
mechanics in :mod:`policytwin.policies.sec_tick_size_pilot`.  It contains no
tick values, exception semantics, treatment list, or permission to execute a
real SEC policy.  Dates supplied by callers are local-market dates with
explicit session boundaries; the module never invents a UTC conversion.
"""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import NoReturn, Self, cast

from policytwin.policies.sec_tick_size_pilot import PolicyGroup
from policytwin.provenance import canonical_json_bytes, sha256_bytes

STUDY_CONTRACT_SCHEMA_VERSION = "policytwin.sec-pilot-study-contract.v1"
ASSIGNMENT_STUDY_SCHEMA_VERSION = "policytwin.sec-pilot-study-assignment.v1"
ITT_DESIGN_SCHEMA_VERSION = "policytwin.sec-pilot-itt-design.v1"
READINESS_AUDIT_SCHEMA_VERSION = "policytwin.sec-pilot-readiness-audit.v1"

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ZERO_SHA256 = "0" * 64


class SecPilotStudyContractError(ValueError):
    """Raised when a study contract is malformed or scientifically unsafe."""


class SecPilotAssignmentChainError(SecPilotStudyContractError):
    """Raised when an assignment-state hash chain is incomplete or altered."""


class SecPilotEvidenceIntegrityError(SecPilotStudyContractError):
    """Raised when declared local evidence cannot pass byte-level validation."""


def _fail(message: str) -> NoReturn:
    raise SecPilotStudyContractError(message)


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        _fail(f"{field} must be a stable ASCII identifier")
    return value


def _sha256(value: object, field: str, *, allow_zero: bool = False) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail(f"{field} must be a lowercase SHA-256 digest")
    if not allow_zero and value == _ZERO_SHA256:
        _fail(f"{field} must not use the all-zero placeholder digest")
    return value


def _local_date(value: object, field: str) -> date:
    if type(value) is not date:
        _fail(f"{field} must be a typed local date, not a datetime")
    return value


def _strict_fields(
    mapping: Mapping[str, object], expected: frozenset[str], context: str
) -> None:
    actual = frozenset(mapping)
    if actual != expected:
        _fail(
            f"{context} fields differ; missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        _fail(f"{field} must be an object with string keys")
    return cast(Mapping[str, object], value)


def _sequence(value: object, field: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        _fail(f"{field} must be an array")
    return cast(Sequence[object], value)


def _parse_local_date(value: object, field: str) -> date:
    if not isinstance(value, str):
        _fail(f"{field} must be an ISO local-date string")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise SecPilotStudyContractError(
            f"{field} must be an ISO local-date string"
        ) from error
    return parsed


def _enum_value[T: StrEnum](value: object, enum_type: type[T], field: str) -> T:
    if not isinstance(value, str):
        _fail(f"{field} must be a string")
    try:
        return enum_type(value)
    except ValueError as error:
        raise SecPilotStudyContractError(f"unsupported {field}: {value}") from error


def _hash_mapping(mapping: Mapping[str, object]) -> str:
    return sha256_bytes(canonical_json_bytes(mapping))


class LocalSessionBoundary(StrEnum):
    """Ordered boundary within one named local-market date."""

    START_OF_LOCAL_DATE = "start_of_local_date"
    SESSION_OPEN = "session_open"
    SESSION_CLOSE = "session_close"
    AFTER_SESSION_CLOSE = "after_session_close"
    END_OF_LOCAL_DATE = "end_of_local_date"


_BOUNDARY_ORDER = {
    LocalSessionBoundary.START_OF_LOCAL_DATE: 0,
    LocalSessionBoundary.SESSION_OPEN: 1,
    LocalSessionBoundary.SESSION_CLOSE: 2,
    LocalSessionBoundary.AFTER_SESSION_CLOSE: 3,
    LocalSessionBoundary.END_OF_LOCAL_DATE: 4,
}


@dataclass(frozen=True, slots=True)
class LocalSessionPoint:
    """A date-plus-boundary on a named calendar, deliberately without UTC."""

    calendar_id: str
    local_date: date
    boundary: LocalSessionBoundary

    def __post_init__(self) -> None:
        _identifier(self.calendar_id, "calendar_id")
        _local_date(self.local_date, "local_date")
        if not isinstance(self.boundary, LocalSessionBoundary):
            _fail("boundary must be a LocalSessionBoundary")

    @property
    def ordering_key(self) -> tuple[date, int]:
        """Return an ordering key meaningful only within ``calendar_id``."""
        return (self.local_date, _BOUNDARY_ORDER[self.boundary])

    def assert_same_calendar(self, other: LocalSessionPoint) -> None:
        """Reject cross-calendar comparisons instead of guessing a timezone."""
        if not isinstance(other, LocalSessionPoint):
            _fail("session comparison requires LocalSessionPoint")
        if self.calendar_id != other.calendar_id:
            _fail("local session points use different calendars")

    def occurs_before(self, other: LocalSessionPoint) -> bool:
        """Compare two points only after their calendar identity is checked."""
        self.assert_same_calendar(other)
        return self.ordering_key < other.ordering_key

    def occurs_at_or_before(self, other: LocalSessionPoint) -> bool:
        """Compare two points only after their calendar identity is checked."""
        self.assert_same_calendar(other)
        return self.ordering_key <= other.ordering_key

    def to_mapping(self) -> dict[str, object]:
        return {
            "boundary": self.boundary.value,
            "calendar_id": self.calendar_id,
            "local_date": self.local_date.isoformat(),
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> Self:
        """Strictly reconstruct a local session point."""
        _strict_fields(
            mapping,
            frozenset({"calendar_id", "local_date", "boundary"}),
            "local session point",
        )
        return cls(
            calendar_id=_identifier(mapping["calendar_id"], "calendar_id"),
            local_date=_parse_local_date(mapping["local_date"], "local_date"),
            boundary=_enum_value(
                mapping["boundary"], LocalSessionBoundary, "session boundary"
            ),
        )


class RolloutStage(StrEnum):
    """Closed administrative rollout milestones; not per-security eligibility."""

    TEST_GROUPS_1_2_INITIAL = "test_groups_1_2_initial"
    TEST_GROUPS_1_2_EXPANDED = "test_groups_1_2_expanded"
    TEST_GROUPS_1_2_FULL = "test_groups_1_2_full"
    TEST_GROUP_3_INITIAL = "test_group_3_initial"
    TEST_GROUP_3_EXPANDED = "test_group_3_expanded"
    TEST_GROUP_3_FULL = "test_group_3_full"


_ROLLOUT_ORDER = tuple(RolloutStage)


@dataclass(frozen=True, slots=True)
class RolloutMilestone:
    """One source-supplied rollout boundary without embedded rule values."""

    stage: RolloutStage
    effective_at: LocalSessionPoint

    def __post_init__(self) -> None:
        if not isinstance(self.stage, RolloutStage):
            _fail("rollout stage must be a RolloutStage")
        if not isinstance(self.effective_at, LocalSessionPoint):
            _fail("rollout effective_at must be a LocalSessionPoint")
        if self.effective_at.boundary is not LocalSessionBoundary.SESSION_OPEN:
            _fail("rollout milestones must declare a local session-open boundary")

    def to_mapping(self) -> dict[str, object]:
        return {
            "effective_at": self.effective_at.to_mapping(),
            "stage": self.stage.value,
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> Self:
        """Strictly reconstruct a rollout milestone."""
        _strict_fields(mapping, frozenset({"stage", "effective_at"}), "rollout")
        return cls(
            stage=_enum_value(mapping["stage"], RolloutStage, "rollout stage"),
            effective_at=LocalSessionPoint.from_mapping(
                _mapping(mapping["effective_at"], "rollout effective_at")
            ),
        )


class TimelineEvidenceClass(StrEnum):
    """Authority attached to schedule dates, never inferred from a URL."""

    LOCATOR_ONLY_UNVERIFIED = "locator_only_unverified"
    LOCAL_PRIMARY_CORPUS_BOUND = "local_primary_corpus_bound"


class EnforcementStatus(StrEnum):
    """Schedule-level enforcement state, not a security-level determination."""

    NOT_STARTED = "not_started"
    PHASED_ROLLOUT = "phased_rollout"
    FULL_ROLLOUT = "full_rollout"
    ENDED = "ended"


class EffectiveAssignmentMode(StrEnum):
    """Whether the schedule has reached the effective-control transition."""

    ORIGINAL_ASSIGNMENT_STATE = "original_assignment_state"
    EFFECTIVE_CONTROL_STATE = "effective_control_state"


class AdministrativePlanStatus(StrEnum):
    """Position relative to the configured administrative expiration boundary."""

    BEFORE_EXPIRATION_BOUNDARY = "before_expiration_boundary"
    EXPIRATION_BOUNDARY_REACHED = "expiration_boundary_reached"


class DataCollectionStatus(StrEnum):
    """Position relative to the configured data-collection end boundary."""

    BEFORE_END_BOUNDARY = "before_end_boundary"
    END_BOUNDARY_REACHED = "end_boundary_reached"


@dataclass(frozen=True, slots=True)
class PolicyRegimeResolution:
    """Multi-axis timeline resolution that avoids collapsing distinct dates."""

    query: LocalSessionPoint
    latest_rollout_stage: RolloutStage | None
    enforcement_status: EnforcementStatus
    effective_assignment_mode: EffectiveAssignmentMode
    plan_status: AdministrativePlanStatus
    data_collection_status: DataCollectionStatus


@dataclass(frozen=True, slots=True)
class PolicyRegimeTimeline:
    """Immutable SEC study timeline with explicit, non-UTC local boundaries.

    This schedule is only a research-design contract.  Even a locally bound
    instance cannot authorize quote, trade, exception, or trade-at mechanics.
    """

    timeline_id: str
    calendar_id: str
    rollout_milestones: tuple[RolloutMilestone, ...]
    enforcement_ends_at: LocalSessionPoint
    effective_control_starts_at: LocalSessionPoint
    formal_plan_expires_at: LocalSessionPoint
    data_collection_ends_at: LocalSessionPoint
    evidence_class: TimelineEvidenceClass
    source_manifest_sha256: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.timeline_id, "timeline_id")
        _identifier(self.calendar_id, "calendar_id")
        if not isinstance(self.rollout_milestones, tuple):
            _fail("rollout_milestones must be an immutable tuple")
        if not all(
            isinstance(item, RolloutMilestone) for item in self.rollout_milestones
        ):
            _fail("rollout_milestones contain an invalid value")
        if tuple(item.stage for item in self.rollout_milestones) != _ROLLOUT_ORDER:
            _fail("rollout_milestones must contain the closed six-stage sequence")
        points = tuple(item.effective_at for item in self.rollout_milestones)
        boundaries = (
            *points,
            self.enforcement_ends_at,
            self.effective_control_starts_at,
            self.formal_plan_expires_at,
            self.data_collection_ends_at,
        )
        if not all(isinstance(item, LocalSessionPoint) for item in boundaries):
            _fail("timeline boundaries must be LocalSessionPoint values")
        if any(item.calendar_id != self.calendar_id for item in boundaries):
            _fail("timeline boundaries must all use timeline calendar_id")
        if any(
            current.ordering_key < previous.ordering_key
            for previous, current in zip(points, points[1:], strict=False)
        ):
            _fail("rollout milestones must be chronological")
        if not points[-1].occurs_before(self.enforcement_ends_at):
            _fail("full rollout must precede the enforcement-end boundary")
        if not self.enforcement_ends_at.occurs_before(self.effective_control_starts_at):
            _fail("effective-control transition must follow enforcement end")
        if self.formal_plan_expires_at.occurs_before(self.effective_control_starts_at):
            _fail("formal Plan expiration cannot precede effective control")
        if self.data_collection_ends_at.occurs_before(self.formal_plan_expires_at):
            _fail("data-collection end cannot precede formal Plan expiration")
        if not isinstance(self.evidence_class, TimelineEvidenceClass):
            _fail("timeline evidence_class is invalid")
        if self.evidence_class is TimelineEvidenceClass.LOCAL_PRIMARY_CORPUS_BOUND:
            if self.source_manifest_sha256 is None:
                _fail("locally bound timeline requires source_manifest_sha256")
            _sha256(self.source_manifest_sha256, "source_manifest_sha256")
        elif self.source_manifest_sha256 is not None:
            _fail("locator-only timeline cannot claim a local source-manifest hash")

    @property
    def sha256(self) -> str:
        """Return a deterministic commitment to all timeline boundaries."""
        return _hash_mapping(self.to_mapping())

    def resolve(self, query: LocalSessionPoint) -> PolicyRegimeResolution:
        """Resolve all schedule axes without inferring a security assignment."""
        if not isinstance(query, LocalSessionPoint):
            _fail("timeline query must be a LocalSessionPoint")
        if query.calendar_id != self.calendar_id:
            _fail("timeline query uses a different calendar")
        latest: RolloutStage | None = None
        latest_index = -1
        for index, milestone in enumerate(self.rollout_milestones):
            if milestone.effective_at.ordering_key <= query.ordering_key:
                latest = milestone.stage
                latest_index = index
        if query.ordering_key >= self.enforcement_ends_at.ordering_key:
            enforcement = EnforcementStatus.ENDED
        elif latest is None:
            enforcement = EnforcementStatus.NOT_STARTED
        elif latest_index < len(self.rollout_milestones) - 1:
            enforcement = EnforcementStatus.PHASED_ROLLOUT
        else:
            enforcement = EnforcementStatus.FULL_ROLLOUT
        return PolicyRegimeResolution(
            query=query,
            latest_rollout_stage=latest,
            enforcement_status=enforcement,
            effective_assignment_mode=(
                EffectiveAssignmentMode.EFFECTIVE_CONTROL_STATE
                if query.ordering_key >= self.effective_control_starts_at.ordering_key
                else EffectiveAssignmentMode.ORIGINAL_ASSIGNMENT_STATE
            ),
            plan_status=(
                AdministrativePlanStatus.EXPIRATION_BOUNDARY_REACHED
                if query.ordering_key >= self.formal_plan_expires_at.ordering_key
                else AdministrativePlanStatus.BEFORE_EXPIRATION_BOUNDARY
            ),
            data_collection_status=(
                DataCollectionStatus.END_BOUNDARY_REACHED
                if query.ordering_key >= self.data_collection_ends_at.ordering_key
                else DataCollectionStatus.BEFORE_END_BOUNDARY
            ),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "calendar_id": self.calendar_id,
            "data_collection_ends_at": self.data_collection_ends_at.to_mapping(),
            "effective_control_starts_at": (
                self.effective_control_starts_at.to_mapping()
            ),
            "enforcement_ends_at": self.enforcement_ends_at.to_mapping(),
            "evidence_class": self.evidence_class.value,
            "formal_plan_expires_at": self.formal_plan_expires_at.to_mapping(),
            "rollout_milestones": [
                item.to_mapping() for item in self.rollout_milestones
            ],
            "schema_version": STUDY_CONTRACT_SCHEMA_VERSION,
            "source_manifest_sha256": self.source_manifest_sha256,
            "timeline_id": self.timeline_id,
        }


class EffectiveGroupChangeReason(StrEnum):
    """Closed reasons for source-bound effective assignment state changes."""

    ORIGINAL_ASSIGNMENT_EFFECTIVE = "original_assignment_effective"
    SOURCE_BOUND_BELOW_PRICE_THRESHOLD = "source_bound_below_price_threshold"
    SOURCE_BOUND_ELIGIBILITY_CHANGE = "source_bound_eligibility_change"
    POST_ENFORCEMENT_EFFECTIVE_CONTROL = "post_enforcement_effective_control"
    SOURCE_BOUND_REMOVAL = "source_bound_removal"


@dataclass(frozen=True, slots=True)
class EffectiveGroupChange:
    """One hash-chained effective-state change; it never edits the ITT arm."""

    security_id: str
    sequence_number: int
    effective_at: LocalSessionPoint
    effective_group: PolicyGroup | None
    reason: EffectiveGroupChangeReason
    source_record_id: str
    source_content_sha256: str
    previous_record_sha256: str
    record_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.security_id, "security_id")
        if (
            isinstance(self.sequence_number, bool)
            or not isinstance(self.sequence_number, int)
            or self.sequence_number < 0
        ):
            _fail("sequence_number must be a non-negative integer")
        if not isinstance(self.effective_at, LocalSessionPoint):
            _fail("effective_at must be a LocalSessionPoint")
        if self.effective_group is not None and not isinstance(
            self.effective_group, PolicyGroup
        ):
            _fail("effective_group must be PolicyGroup or None")
        if not isinstance(self.reason, EffectiveGroupChangeReason):
            _fail("effective group change reason is invalid")
        _identifier(self.source_record_id, "source_record_id")
        _sha256(self.source_content_sha256, "source_content_sha256")
        _sha256(
            self.previous_record_sha256,
            "previous_record_sha256",
            allow_zero=self.sequence_number == 0,
        )
        _sha256(self.record_sha256, "record_sha256")
        if self.reason is EffectiveGroupChangeReason.SOURCE_BOUND_REMOVAL:
            if self.effective_group is not None:
                _fail("source-bound removal must set effective_group to None")
        elif self.effective_group is None:
            _fail("only a source-bound removal may set effective_group to None")
        if self.record_sha256 != _hash_mapping(self.payload_mapping()):
            raise SecPilotAssignmentChainError(
                "effective-group change record hash does not match its payload"
            )

    @classmethod
    def create(
        cls,
        *,
        security_id: str,
        sequence_number: int,
        effective_at: LocalSessionPoint,
        effective_group: PolicyGroup | None,
        reason: EffectiveGroupChangeReason,
        source_record_id: str,
        source_content_sha256: str,
        previous_record_sha256: str,
    ) -> Self:
        """Create and seal one effective-state record."""
        payload = cls._payload_from_values(
            security_id=security_id,
            sequence_number=sequence_number,
            effective_at=effective_at,
            effective_group=effective_group,
            reason=reason,
            source_record_id=source_record_id,
            source_content_sha256=source_content_sha256,
            previous_record_sha256=previous_record_sha256,
        )
        return cls(
            security_id=security_id,
            sequence_number=sequence_number,
            effective_at=effective_at,
            effective_group=effective_group,
            reason=reason,
            source_record_id=source_record_id,
            source_content_sha256=source_content_sha256,
            previous_record_sha256=previous_record_sha256,
            record_sha256=_hash_mapping(payload),
        )

    @staticmethod
    def _payload_from_values(
        *,
        security_id: str,
        sequence_number: int,
        effective_at: LocalSessionPoint,
        effective_group: PolicyGroup | None,
        reason: EffectiveGroupChangeReason,
        source_record_id: str,
        source_content_sha256: str,
        previous_record_sha256: str,
    ) -> dict[str, object]:
        return {
            "effective_at": effective_at.to_mapping(),
            "effective_group": (
                None if effective_group is None else effective_group.value
            ),
            "previous_record_sha256": previous_record_sha256,
            "reason": reason.value,
            "security_id": security_id,
            "sequence_number": sequence_number,
            "source_content_sha256": source_content_sha256,
            "source_record_id": source_record_id,
        }

    def payload_mapping(self) -> dict[str, object]:
        """Return the exact hash-domain payload."""
        return self._payload_from_values(
            security_id=self.security_id,
            sequence_number=self.sequence_number,
            effective_at=self.effective_at,
            effective_group=self.effective_group,
            reason=self.reason,
            source_record_id=self.source_record_id,
            source_content_sha256=self.source_content_sha256,
            previous_record_sha256=self.previous_record_sha256,
        )

    def to_mapping(self) -> dict[str, object]:
        return {**self.payload_mapping(), "record_sha256": self.record_sha256}

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> Self:
        """Strictly reconstruct and re-hash one effective-state record."""
        _strict_fields(
            mapping,
            frozenset(
                {
                    "security_id",
                    "sequence_number",
                    "effective_at",
                    "effective_group",
                    "reason",
                    "source_record_id",
                    "source_content_sha256",
                    "previous_record_sha256",
                    "record_sha256",
                }
            ),
            "effective-group change",
        )
        sequence = mapping["sequence_number"]
        if isinstance(sequence, bool) or not isinstance(sequence, int):
            _fail("sequence_number must be an integer")
        raw_group = mapping["effective_group"]
        group = (
            None
            if raw_group is None
            else _enum_value(raw_group, PolicyGroup, "effective_group")
        )
        return cls(
            security_id=_identifier(mapping["security_id"], "security_id"),
            sequence_number=sequence,
            effective_at=LocalSessionPoint.from_mapping(
                _mapping(mapping["effective_at"], "effective_at")
            ),
            effective_group=group,
            reason=_enum_value(
                mapping["reason"], EffectiveGroupChangeReason, "change reason"
            ),
            source_record_id=_identifier(
                mapping["source_record_id"], "source_record_id"
            ),
            source_content_sha256=_sha256(
                mapping["source_content_sha256"], "source_content_sha256"
            ),
            previous_record_sha256=_sha256(
                mapping["previous_record_sha256"],
                "previous_record_sha256",
                allow_zero=sequence == 0,
            ),
            record_sha256=_sha256(mapping["record_sha256"], "record_sha256"),
        )


@dataclass(frozen=True, slots=True)
class EffectiveAssignmentResolution:
    """Original ITT arm plus the date-specific effective state."""

    security_id: str
    query: LocalSessionPoint
    original_assigned_group: PolicyGroup
    effective_group: PolicyGroup | None
    effective_change_reason: EffectiveGroupChangeReason | None
    effective_record_sha256: str | None


@dataclass(frozen=True, slots=True)
class SecurityStudyAssignment:
    """Immutable original arm and complete, hash-chained effective history."""

    security_id: str
    original_assigned_group: PolicyGroup
    effective_history: tuple[EffectiveGroupChange, ...]

    def __post_init__(self) -> None:
        _identifier(self.security_id, "security_id")
        if not isinstance(self.original_assigned_group, PolicyGroup):
            _fail("original_assigned_group must be PolicyGroup")
        if not isinstance(self.effective_history, tuple) or not self.effective_history:
            raise SecPilotAssignmentChainError(
                "effective history must be a non-empty immutable tuple"
            )
        if not all(
            isinstance(item, EffectiveGroupChange) for item in self.effective_history
        ):
            _fail("effective_history contains an invalid value")
        first = self.effective_history[0]
        if first.sequence_number != 0 or first.previous_record_sha256 != _ZERO_SHA256:
            raise SecPilotAssignmentChainError(
                "effective history must start at sequence zero and the zero-chain root"
            )
        if (
            first.reason is not EffectiveGroupChangeReason.ORIGINAL_ASSIGNMENT_EFFECTIVE
            or first.effective_group is not self.original_assigned_group
        ):
            raise SecPilotAssignmentChainError(
                "first effective state must preserve the original assigned group"
            )
        previous: EffectiveGroupChange | None = None
        for index, current in enumerate(self.effective_history):
            if current.security_id != self.security_id:
                raise SecPilotAssignmentChainError(
                    "effective history contains another security_id"
                )
            if current.sequence_number != index:
                raise SecPilotAssignmentChainError(
                    "effective history sequence is incomplete"
                )
            if previous is not None:
                if current.previous_record_sha256 != previous.record_sha256:
                    raise SecPilotAssignmentChainError(
                        "effective history previous-record hash does not match"
                    )
                if (
                    current.effective_at.ordering_key
                    <= previous.effective_at.ordering_key
                ):
                    raise SecPilotAssignmentChainError(
                        "effective history dates must be strictly increasing"
                    )
                if (
                    current.effective_at.calendar_id
                    != previous.effective_at.calendar_id
                ):
                    raise SecPilotAssignmentChainError(
                        "effective history cannot mix local calendars"
                    )
                if (
                    current.reason
                    is EffectiveGroupChangeReason.ORIGINAL_ASSIGNMENT_EFFECTIVE
                ):
                    raise SecPilotAssignmentChainError(
                        "original-assignment reason is allowed only at chain root"
                    )
                if current.effective_group is previous.effective_group:
                    raise SecPilotAssignmentChainError(
                        "effective-state record must change the effective group"
                    )
            previous = current

    @property
    def original_group(self) -> PolicyGroup:
        """Return the frozen ITT arm; effective changes cannot modify it."""
        return self.original_assigned_group

    def resolve(self, query: LocalSessionPoint) -> EffectiveAssignmentResolution:
        """Resolve effective state while retaining the immutable original arm."""
        if not isinstance(query, LocalSessionPoint):
            _fail("assignment query must be a LocalSessionPoint")
        calendar_id = self.effective_history[0].effective_at.calendar_id
        if query.calendar_id != calendar_id:
            _fail("assignment query uses a different calendar")
        selected: EffectiveGroupChange | None = None
        for change in self.effective_history:
            if change.effective_at.ordering_key <= query.ordering_key:
                selected = change
        return EffectiveAssignmentResolution(
            security_id=self.security_id,
            query=query,
            original_assigned_group=self.original_assigned_group,
            effective_group=None if selected is None else selected.effective_group,
            effective_change_reason=None if selected is None else selected.reason,
            effective_record_sha256=None
            if selected is None
            else selected.record_sha256,
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "effective_history": [item.to_mapping() for item in self.effective_history],
            "original_assigned_group": self.original_assigned_group.value,
            "security_id": self.security_id,
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> Self:
        """Strictly reconstruct one original/effective assignment contract."""
        _strict_fields(
            mapping,
            frozenset({"security_id", "original_assigned_group", "effective_history"}),
            "security study assignment",
        )
        return cls(
            security_id=_identifier(mapping["security_id"], "security_id"),
            original_assigned_group=_enum_value(
                mapping["original_assigned_group"],
                PolicyGroup,
                "original_assigned_group",
            ),
            effective_history=tuple(
                EffectiveGroupChange.from_mapping(
                    _mapping(item, "effective_history item")
                )
                for item in _sequence(mapping["effective_history"], "effective_history")
            ),
        )


@dataclass(frozen=True, slots=True)
class FrozenStudyAssignmentManifest:
    """Frozen four-arm original assignment plus effective-state histories."""

    manifest_id: str
    calendar_id: str
    source_artifact_sha256: str
    records: tuple[SecurityStudyAssignment, ...]
    frozen: bool
    manifest_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.manifest_id, "manifest_id")
        _identifier(self.calendar_id, "calendar_id")
        _sha256(self.source_artifact_sha256, "source_artifact_sha256")
        if not isinstance(self.frozen, bool) or not self.frozen:
            _fail("study assignment manifest must be frozen")
        if not isinstance(self.records, tuple) or not self.records:
            _fail("study assignment records must be a non-empty immutable tuple")
        if not all(isinstance(item, SecurityStudyAssignment) for item in self.records):
            _fail("study assignment manifest contains an invalid record")
        ordered = tuple(sorted(self.records, key=lambda item: item.security_id))
        if ordered != self.records:
            _fail("study assignment records must use canonical security order")
        if len({item.security_id for item in self.records}) != len(self.records):
            _fail("study assignment manifest contains duplicate securities")
        if {item.original_assigned_group for item in self.records} != set(PolicyGroup):
            _fail("study assignment manifest must preserve all four original arms")
        for record in self.records:
            if any(
                change.effective_at.calendar_id != self.calendar_id
                for change in record.effective_history
            ):
                _fail("assignment histories must use manifest calendar_id")
        _sha256(self.manifest_sha256, "manifest_sha256")
        if self.manifest_sha256 != _hash_mapping(self.payload_mapping()):
            raise SecPilotAssignmentChainError(
                "study assignment manifest hash does not match its payload"
            )

    @classmethod
    def create(
        cls,
        *,
        manifest_id: str,
        calendar_id: str,
        source_artifact_sha256: str,
        records: tuple[SecurityStudyAssignment, ...],
    ) -> Self:
        """Create and seal a frozen four-arm assignment manifest."""
        payload = cls._payload_from_values(
            manifest_id=manifest_id,
            calendar_id=calendar_id,
            source_artifact_sha256=source_artifact_sha256,
            records=records,
            frozen=True,
        )
        return cls(
            manifest_id=manifest_id,
            calendar_id=calendar_id,
            source_artifact_sha256=source_artifact_sha256,
            records=records,
            frozen=True,
            manifest_sha256=_hash_mapping(payload),
        )

    @staticmethod
    def _payload_from_values(
        *,
        manifest_id: str,
        calendar_id: str,
        source_artifact_sha256: str,
        records: tuple[SecurityStudyAssignment, ...],
        frozen: bool,
    ) -> dict[str, object]:
        return {
            "calendar_id": calendar_id,
            "frozen": frozen,
            "manifest_id": manifest_id,
            "records": [item.to_mapping() for item in records],
            "schema_version": ASSIGNMENT_STUDY_SCHEMA_VERSION,
            "source_artifact_sha256": source_artifact_sha256,
        }

    def payload_mapping(self) -> dict[str, object]:
        """Return the exact assignment-manifest hash domain."""
        return self._payload_from_values(
            manifest_id=self.manifest_id,
            calendar_id=self.calendar_id,
            source_artifact_sha256=self.source_artifact_sha256,
            records=self.records,
            frozen=self.frozen,
        )

    def to_mapping(self) -> dict[str, object]:
        return {**self.payload_mapping(), "manifest_sha256": self.manifest_sha256}

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> Self:
        """Strictly reconstruct and re-hash a frozen assignment manifest."""
        _strict_fields(
            mapping,
            frozenset(
                {
                    "schema_version",
                    "manifest_id",
                    "calendar_id",
                    "source_artifact_sha256",
                    "records",
                    "frozen",
                    "manifest_sha256",
                }
            ),
            "study assignment manifest",
        )
        if mapping["schema_version"] != ASSIGNMENT_STUDY_SCHEMA_VERSION:
            _fail("unsupported study assignment schema_version")
        frozen = mapping["frozen"]
        if not isinstance(frozen, bool):
            _fail("study assignment frozen must be a boolean")
        return cls(
            manifest_id=_identifier(mapping["manifest_id"], "manifest_id"),
            calendar_id=_identifier(mapping["calendar_id"], "calendar_id"),
            source_artifact_sha256=_sha256(
                mapping["source_artifact_sha256"], "source_artifact_sha256"
            ),
            records=tuple(
                SecurityStudyAssignment.from_mapping(_mapping(item, "records item"))
                for item in _sequence(mapping["records"], "records")
            ),
            frozen=frozen,
            manifest_sha256=_sha256(mapping["manifest_sha256"], "manifest_sha256"),
        )

    def record_for(self, security_id: str) -> SecurityStudyAssignment:
        """Return one frozen assignment record or fail closed."""
        stable = _identifier(security_id, "security_id")
        for record in self.records:
            if record.security_id == stable:
                return record
        raise SecPilotStudyContractError(
            f"security {stable!r} is absent from the frozen assignment manifest"
        )


class PrimaryCohortBasis(StrEnum):
    """Allowed schema values; only original assignment is valid for primary ITT."""

    ORIGINAL_ASSIGNED_GROUP = "original_assigned_group"
    EFFECTIVE_GROUP = "effective_group"


@dataclass(frozen=True, slots=True)
class FrozenITTContrast:
    """One immutable contrast over original assignment arms."""

    contrast_id: str
    treated_original_arms: tuple[PolicyGroup, ...]
    control_original_arms: tuple[PolicyGroup, ...]

    def __post_init__(self) -> None:
        _identifier(self.contrast_id, "contrast_id")
        for field in ("treated_original_arms", "control_original_arms"):
            value = getattr(self, field)
            if not isinstance(value, tuple) or not value:
                _fail(f"{field} must be a non-empty immutable tuple")
            if not all(isinstance(item, PolicyGroup) for item in value):
                _fail(f"{field} must contain only PolicyGroup values")
            if len(set(value)) != len(value):
                _fail(f"{field} must not contain duplicate arms")
        if set(self.treated_original_arms) & set(self.control_original_arms):
            _fail("ITT contrast treatment and control arms must be disjoint")

    def to_mapping(self) -> dict[str, object]:
        return {
            "contrast_id": self.contrast_id,
            "control_original_arms": [
                item.value for item in self.control_original_arms
            ],
            "treated_original_arms": [
                item.value for item in self.treated_original_arms
            ],
        }


@dataclass(frozen=True, slots=True)
class SecPilotITTDesign:
    """Frozen four-arm ITT design that rejects effective-state primary cohorts."""

    design_id: str
    assignment_manifest_sha256: str
    primary_cohort_basis: PrimaryCohortBasis
    contrasts: tuple[FrozenITTContrast, ...]
    frozen: bool
    design_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.design_id, "design_id")
        _sha256(self.assignment_manifest_sha256, "assignment_manifest_sha256")
        if not isinstance(self.primary_cohort_basis, PrimaryCohortBasis):
            _fail("primary_cohort_basis is invalid")
        if self.primary_cohort_basis is not PrimaryCohortBasis.ORIGINAL_ASSIGNED_GROUP:
            _fail("primary SEC ITT cohort must use original_assigned_group")
        if not isinstance(self.contrasts, tuple) or not self.contrasts:
            _fail("ITT contrasts must be a non-empty immutable tuple")
        if not all(isinstance(item, FrozenITTContrast) for item in self.contrasts):
            _fail("ITT contrasts contain an invalid value")
        if len({item.contrast_id for item in self.contrasts}) != len(self.contrasts):
            _fail("ITT contrast IDs must be unique")
        if not isinstance(self.frozen, bool) or not self.frozen:
            _fail("SEC ITT design must be frozen")
        _sha256(self.design_sha256, "design_sha256")
        if self.design_sha256 != _hash_mapping(self.payload_mapping()):
            _fail("SEC ITT design hash does not match its payload")

    @classmethod
    def create(
        cls,
        *,
        design_id: str,
        assignment_manifest: FrozenStudyAssignmentManifest,
        contrasts: tuple[FrozenITTContrast, ...],
        primary_cohort_basis: PrimaryCohortBasis = (
            PrimaryCohortBasis.ORIGINAL_ASSIGNED_GROUP
        ),
    ) -> Self:
        """Create, validate, and seal an original-assignment ITT design."""
        if not isinstance(assignment_manifest, FrozenStudyAssignmentManifest):
            _fail("assignment_manifest must be FrozenStudyAssignmentManifest")
        payload = cls._payload_from_values(
            design_id=design_id,
            assignment_manifest_sha256=assignment_manifest.manifest_sha256,
            primary_cohort_basis=primary_cohort_basis,
            contrasts=contrasts,
            frozen=True,
        )
        result = cls(
            design_id=design_id,
            assignment_manifest_sha256=assignment_manifest.manifest_sha256,
            primary_cohort_basis=primary_cohort_basis,
            contrasts=contrasts,
            frozen=True,
            design_sha256=_hash_mapping(payload),
        )
        result.assert_assignment_manifest(assignment_manifest)
        return result

    @staticmethod
    def _payload_from_values(
        *,
        design_id: str,
        assignment_manifest_sha256: str,
        primary_cohort_basis: PrimaryCohortBasis,
        contrasts: tuple[FrozenITTContrast, ...],
        frozen: bool,
    ) -> dict[str, object]:
        return {
            "assignment_manifest_sha256": assignment_manifest_sha256,
            "contrasts": [item.to_mapping() for item in contrasts],
            "design_id": design_id,
            "frozen": frozen,
            "primary_cohort_basis": primary_cohort_basis.value,
            "schema_version": ITT_DESIGN_SCHEMA_VERSION,
        }

    def payload_mapping(self) -> dict[str, object]:
        """Return the exact frozen-design hash domain."""
        return self._payload_from_values(
            design_id=self.design_id,
            assignment_manifest_sha256=self.assignment_manifest_sha256,
            primary_cohort_basis=self.primary_cohort_basis,
            contrasts=self.contrasts,
            frozen=self.frozen,
        )

    def to_mapping(self) -> dict[str, object]:
        return {**self.payload_mapping(), "design_sha256": self.design_sha256}

    def assert_assignment_manifest(
        self, assignment_manifest: FrozenStudyAssignmentManifest
    ) -> None:
        """Fail if the design is paired with another assignment epoch."""
        if not isinstance(assignment_manifest, FrozenStudyAssignmentManifest):
            _fail("assignment_manifest must be FrozenStudyAssignmentManifest")
        if assignment_manifest.manifest_sha256 != self.assignment_manifest_sha256:
            _fail("SEC ITT design assignment manifest hash differs")
        available_arms = {
            item.original_assigned_group for item in assignment_manifest.records
        }
        for contrast in self.contrasts:
            requested = set(contrast.treated_original_arms) | set(
                contrast.control_original_arms
            )
            if not requested <= available_arms:
                _fail("ITT contrast references an absent original assignment arm")


class RequiredEvidenceComponent(StrEnum):
    """Local evidence classes required before even source-level review."""

    PRIMARY_SOURCE_CORPUS = "primary_source_corpus"
    ASSIGNMENT_CHANGE_HISTORY = "assignment_change_history"
    VENUE_BASELINE_RULES = "venue_baseline_rules"
    DOCUMENTED_EXCEPTIONS = "documented_exceptions"


@dataclass(frozen=True, slots=True)
class SourceLocator:
    """An online discovery pointer that explicitly does not count as evidence."""

    component: RequiredEvidenceComponent
    url: str

    def __post_init__(self) -> None:
        if not isinstance(self.component, RequiredEvidenceComponent):
            _fail("source locator component is invalid")
        if not isinstance(self.url, str) or not self.url.startswith("https://"):
            _fail("source locator must be an HTTPS URL")


@dataclass(frozen=True, slots=True)
class LocalEvidenceArtifact:
    """One frozen local file commitment used by the readiness audit."""

    component: RequiredEvidenceComponent
    local_path: str
    expected_sha256: str
    manifest_id: str
    frozen: bool

    def __post_init__(self) -> None:
        if not isinstance(self.component, RequiredEvidenceComponent):
            _fail("local evidence component is invalid")
        if (
            not isinstance(self.local_path, str)
            or not self.local_path
            or "://" in self.local_path
            or "\x00" in self.local_path
        ):
            _fail("local evidence path must name a local filesystem path")
        _sha256(self.expected_sha256, "expected_sha256")
        _identifier(self.manifest_id, "manifest_id")
        if not isinstance(self.frozen, bool) or not self.frozen:
            _fail("local evidence artifact must be frozen")

    def verify(self) -> str:
        """Read one regular local file once and validate its byte commitment."""
        candidate = Path(os.path.abspath(Path(self.local_path).expanduser()))
        current = Path(candidate.anchor)
        for part in candidate.parts[1:]:
            current /= part
            if current.is_symlink():
                raise SecPilotEvidenceIntegrityError(
                    f"local evidence path contains a symbolic link: {current}"
                )
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(candidate, flags)
        except OSError as error:
            raise SecPilotEvidenceIntegrityError(
                f"cannot open declared local evidence artifact: {candidate}"
            ) from error
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise SecPilotEvidenceIntegrityError(
                    "declared local evidence artifact is not a regular file"
                )
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(descriptor)
        except OSError as error:
            raise SecPilotEvidenceIntegrityError(
                f"cannot read declared local evidence artifact: {candidate}"
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
            raise SecPilotEvidenceIntegrityError(
                "declared local evidence artifact changed while it was read"
            )
        observed = sha256_bytes(b"".join(chunks))
        if observed != self.expected_sha256:
            raise SecPilotEvidenceIntegrityError(
                "declared local evidence artifact hash does not match"
            )
        return observed


class SecPilotReadinessStatus(StrEnum):
    """Study-contract readiness; neither value authorizes real execution."""

    ABSTAIN = "abstain"
    READY_FOR_MANUAL_SOURCE_VALIDATION = "ready_for_manual_source_validation"


@dataclass(frozen=True, slots=True)
class SecPilotReadinessReport:
    """Machine-readable readiness outcome with a permanent execution denial."""

    status: SecPilotReadinessStatus
    missing_components: tuple[RequiredEvidenceComponent, ...]
    verified_components: tuple[RequiredEvidenceComponent, ...]
    reasons: tuple[str, ...]
    timeline_sha256: str
    assignment_manifest_sha256: str
    itt_design_sha256: str
    authorizes_real_policy_execution: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.status, SecPilotReadinessStatus):
            _fail("readiness status is invalid")
        if self.authorizes_real_policy_execution:
            _fail("study readiness report can never authorize real policy execution")
        for field in (
            "timeline_sha256",
            "assignment_manifest_sha256",
            "itt_design_sha256",
        ):
            _sha256(getattr(self, field), field)

    @property
    def sha256(self) -> str:
        """Return a deterministic commitment to the readiness outcome."""
        return _hash_mapping(self.to_mapping())

    def to_mapping(self) -> dict[str, object]:
        return {
            "assignment_manifest_sha256": self.assignment_manifest_sha256,
            "authorizes_real_policy_execution": (self.authorizes_real_policy_execution),
            "itt_design_sha256": self.itt_design_sha256,
            "missing_components": [item.value for item in self.missing_components],
            "reasons": list(self.reasons),
            "schema_version": READINESS_AUDIT_SCHEMA_VERSION,
            "status": self.status.value,
            "timeline_sha256": self.timeline_sha256,
            "verified_components": [item.value for item in self.verified_components],
        }


@dataclass(frozen=True, slots=True)
class SecPilotReadinessAudit:
    """Offline, fail-closed audit of source-level study prerequisites."""

    timeline: PolicyRegimeTimeline
    assignment_manifest: FrozenStudyAssignmentManifest
    itt_design: SecPilotITTDesign
    local_evidence: tuple[LocalEvidenceArtifact, ...] = ()
    source_locators: tuple[SourceLocator, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.timeline, PolicyRegimeTimeline):
            _fail("readiness audit requires PolicyRegimeTimeline")
        if not isinstance(self.assignment_manifest, FrozenStudyAssignmentManifest):
            _fail("readiness audit requires FrozenStudyAssignmentManifest")
        if not isinstance(self.itt_design, SecPilotITTDesign):
            _fail("readiness audit requires SecPilotITTDesign")
        if not isinstance(self.local_evidence, tuple) or not all(
            isinstance(item, LocalEvidenceArtifact) for item in self.local_evidence
        ):
            _fail("local_evidence must be an immutable LocalEvidenceArtifact tuple")
        if not isinstance(self.source_locators, tuple) or not all(
            isinstance(item, SourceLocator) for item in self.source_locators
        ):
            _fail("source_locators must be an immutable SourceLocator tuple")
        components = tuple(item.component for item in self.local_evidence)
        if len(set(components)) != len(components):
            _fail("readiness audit contains duplicate local evidence components")
        if self.timeline.calendar_id != self.assignment_manifest.calendar_id:
            _fail("timeline and assignment manifest calendars differ")
        self.itt_design.assert_assignment_manifest(self.assignment_manifest)

    def evaluate(self) -> SecPilotReadinessReport:
        """Verify local bytes; absent prerequisites produce explicit abstention."""
        by_component = {item.component: item for item in self.local_evidence}
        verified = tuple(
            component
            for component in RequiredEvidenceComponent
            if component in by_component
            and by_component[component].verify()
            == by_component[component].expected_sha256
        )
        missing = tuple(
            component
            for component in RequiredEvidenceComponent
            if component not in by_component
        )
        reasons = [f"missing_local_evidence:{item.value}" for item in missing]
        if (
            self.timeline.evidence_class
            is TimelineEvidenceClass.LOCATOR_ONLY_UNVERIFIED
        ):
            reasons.append("timeline_is_locator_only_unverified")
        if not missing and self.timeline.source_manifest_sha256 is not None:
            source_artifact = by_component[
                RequiredEvidenceComponent.PRIMARY_SOURCE_CORPUS
            ]
            if source_artifact.expected_sha256 != self.timeline.source_manifest_sha256:
                raise SecPilotEvidenceIntegrityError(
                    "timeline source-manifest hash differs from local source corpus"
                )
            assignment_artifact = by_component[
                RequiredEvidenceComponent.ASSIGNMENT_CHANGE_HISTORY
            ]
            if (
                assignment_artifact.expected_sha256
                != self.assignment_manifest.source_artifact_sha256
            ):
                raise SecPilotEvidenceIntegrityError(
                    "assignment source-artifact hash differs from local change history"
                )
        status = (
            SecPilotReadinessStatus.ABSTAIN
            if reasons
            else SecPilotReadinessStatus.READY_FOR_MANUAL_SOURCE_VALIDATION
        )
        return SecPilotReadinessReport(
            status=status,
            missing_components=missing,
            verified_components=verified,
            reasons=tuple(reasons),
            timeline_sha256=self.timeline.sha256,
            assignment_manifest_sha256=self.assignment_manifest.manifest_sha256,
            itt_design_sha256=self.itt_design.design_sha256,
        )


__all__ = [
    "ASSIGNMENT_STUDY_SCHEMA_VERSION",
    "ITT_DESIGN_SCHEMA_VERSION",
    "READINESS_AUDIT_SCHEMA_VERSION",
    "STUDY_CONTRACT_SCHEMA_VERSION",
    "AdministrativePlanStatus",
    "DataCollectionStatus",
    "EffectiveAssignmentMode",
    "EffectiveAssignmentResolution",
    "EffectiveGroupChange",
    "EffectiveGroupChangeReason",
    "EnforcementStatus",
    "FrozenITTContrast",
    "FrozenStudyAssignmentManifest",
    "LocalEvidenceArtifact",
    "LocalSessionBoundary",
    "LocalSessionPoint",
    "PolicyRegimeResolution",
    "PolicyRegimeTimeline",
    "PrimaryCohortBasis",
    "RequiredEvidenceComponent",
    "RolloutMilestone",
    "RolloutStage",
    "SecPilotAssignmentChainError",
    "SecPilotEvidenceIntegrityError",
    "SecPilotITTDesign",
    "SecPilotReadinessAudit",
    "SecPilotReadinessReport",
    "SecPilotReadinessStatus",
    "SecPilotStudyContractError",
    "SecurityStudyAssignment",
    "SourceLocator",
    "TimelineEvidenceClass",
]
