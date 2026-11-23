"""Anonymous episodic memory and pure semantic-belief transitions."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from policytwin.social.events import (
    InformationEvent,
    InformationEventKind,
)

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_EVENT_TYPE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class CognitiveMemoryError(ValueError):
    """Raised when a memory object or transition is malformed."""


class CognitiveMemoryTemporalError(CognitiveMemoryError):
    """Raised when memory would incorporate unavailable information."""


def _identifier(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise CognitiveMemoryError(
            f"{field_name} must be a stable anonymous identifier"
        )
    return value


def _sha256(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise CognitiveMemoryError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _aware_utc(value: object, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise CognitiveMemoryError(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise CognitiveMemoryError(f"{field_name} must include an explicit timezone")
    try:
        return value.astimezone(UTC)
    except (OverflowError, ValueError) as error:
        raise CognitiveMemoryError(f"{field_name} is outside the UTC range") from error


def _unit_interval(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CognitiveMemoryError(f"{field_name} must be a finite real number")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise CognitiveMemoryError(f"{field_name} must be finite and in [0, 1]")
    return normalized


def _non_negative(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CognitiveMemoryError(f"{field_name} must be a finite real number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise CognitiveMemoryError(f"{field_name} must be finite and non-negative")
    return normalized


def _belief_value(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CognitiveMemoryError(f"{field_name} must be a finite real number")
    normalized = float(value)
    if not math.isfinite(normalized) or not -1.0 <= normalized <= 1.0:
        raise CognitiveMemoryError(f"{field_name} must be finite and in [-1, 1]")
    return normalized


def _event_type(value: object) -> str:
    if isinstance(value, StrEnum):
        normalized = value.value
    elif isinstance(value, str):
        normalized = value
    else:
        raise CognitiveMemoryError("event_type must be a string or StrEnum")
    if _EVENT_TYPE_PATTERN.fullmatch(normalized) is None:
        raise CognitiveMemoryError("event_type must be a stable code without free text")
    return normalized


def _entry_order(entry: EpisodicMemoryEntry) -> tuple[object, ...]:
    return (
        entry.observed_time,
        entry.available_time,
        entry.event_time,
        entry.content_sha256,
        entry.anonymous_topic_id,
        entry.anonymous_source_id,
        entry.event_type,
        entry.attention,
        entry.strength,
    )


def _validate_information_event(
    event: InformationEvent,
    *,
    trusted_cutoff: datetime,
    cutoff_policy_sha256: str,
    lineage_sha256: str,
) -> None:
    """Repeat the event firewall at the mutable-system integration boundary."""
    if not isinstance(event, InformationEvent):
        raise CognitiveMemoryError("event must be an InformationEvent")
    if event.trusted_cutoff != trusted_cutoff:
        raise CognitiveMemoryTemporalError(
            "event trusted_cutoff does not match the cognitive state"
        )
    if event.cutoff_policy_sha256 != cutoff_policy_sha256:
        raise CognitiveMemoryTemporalError(
            "event cutoff policy commitment does not match the cognitive state"
        )
    if event.lineage_sha256 != lineage_sha256:
        raise CognitiveMemoryTemporalError(
            "event lineage commitment does not match the cognitive state"
        )
    if event.as_of_time > trusted_cutoff:
        raise CognitiveMemoryTemporalError("event as_of_time is after trusted_cutoff")
    if event.event_time > event.as_of_time:
        raise CognitiveMemoryTemporalError("event_time is after event as_of_time")
    if event.available_time > event.as_of_time:
        raise CognitiveMemoryTemporalError(
            "event was unavailable at its declared as_of_time"
        )
    if event.available_time > trusted_cutoff:
        raise CognitiveMemoryTemporalError(
            "event available_time is after trusted_cutoff"
        )


@dataclass(frozen=True, slots=True)
class EpisodicMemoryEntry:
    """One text-free, anonymous record of an attended information event."""

    anonymous_topic_id: str
    anonymous_source_id: str
    content_sha256: str
    event_type: str | StrEnum
    event_time: datetime
    available_time: datetime
    observed_time: datetime
    attention: float
    strength: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "anonymous_topic_id",
            _identifier(self.anonymous_topic_id, "anonymous_topic_id"),
        )
        object.__setattr__(
            self,
            "anonymous_source_id",
            _identifier(self.anonymous_source_id, "anonymous_source_id"),
        )
        object.__setattr__(
            self,
            "content_sha256",
            _sha256(self.content_sha256, "content_sha256"),
        )
        object.__setattr__(self, "event_type", _event_type(self.event_type))
        event_time = _aware_utc(self.event_time, "event_time")
        available_time = _aware_utc(self.available_time, "available_time")
        observed_time = _aware_utc(self.observed_time, "observed_time")
        if event_time > observed_time:
            raise CognitiveMemoryTemporalError("event_time is after observed_time")
        if available_time > observed_time:
            raise CognitiveMemoryTemporalError(
                "information was unavailable at observed_time"
            )
        object.__setattr__(self, "event_time", event_time)
        object.__setattr__(self, "available_time", available_time)
        object.__setattr__(self, "observed_time", observed_time)
        object.__setattr__(
            self, "attention", _unit_interval(self.attention, "attention")
        )
        object.__setattr__(self, "strength", _unit_interval(self.strength, "strength"))

    @classmethod
    def from_information_event(
        cls,
        event: InformationEvent,
        *,
        attention: float,
        strength: float,
        observed_time: datetime | None = None,
    ) -> EpisodicMemoryEntry:
        """Project an event into memory without retaining its raw payload."""
        if not isinstance(event, InformationEvent):
            raise CognitiveMemoryError("event must be an InformationEvent")
        return cls(
            anonymous_topic_id=event.event_id,
            anonymous_source_id=event.anonymous_source_id,
            content_sha256=event.content_sha256,
            event_type=event.event_kind,
            event_time=event.event_time,
            available_time=event.available_time,
            observed_time=event.as_of_time if observed_time is None else observed_time,
            attention=attention,
            strength=strength,
        )

    def validate_cutoff(self, trusted_cutoff: datetime) -> None:
        """Fail closed unless all knowledge times are no later than cutoff."""
        cutoff = _aware_utc(trusted_cutoff, "trusted_cutoff")
        if (
            self.event_time > cutoff
            or self.available_time > cutoff
            or self.observed_time > cutoff
        ):
            raise CognitiveMemoryTemporalError(
                "episodic entry crosses the trusted cutoff"
            )

    def to_mapping(self) -> dict[str, object]:
        """Return the complete text-free episodic record."""
        return {
            "anonymous_source_id": self.anonymous_source_id,
            "anonymous_topic_id": self.anonymous_topic_id,
            "attention": self.attention,
            "available_time": self.available_time.isoformat(),
            "content_sha256": self.content_sha256,
            "event_time": self.event_time.isoformat(),
            "event_type": str(self.event_type),
            "observed_time": self.observed_time.isoformat(),
            "strength": self.strength,
        }


@dataclass(frozen=True, slots=True)
class EpisodicMemory:
    """A fixed-capacity memory retaining the most recently observed entries."""

    capacity: int
    trusted_cutoff: datetime
    cutoff_policy_sha256: str
    lineage_sha256: str
    entries: tuple[EpisodicMemoryEntry, ...] = ()

    def __post_init__(self) -> None:
        if (
            isinstance(self.capacity, bool)
            or not isinstance(self.capacity, int)
            or self.capacity < 0
        ):
            raise CognitiveMemoryError("capacity must be a non-negative integer")
        cutoff = _aware_utc(self.trusted_cutoff, "trusted_cutoff")
        object.__setattr__(self, "trusted_cutoff", cutoff)
        object.__setattr__(
            self,
            "cutoff_policy_sha256",
            _sha256(self.cutoff_policy_sha256, "cutoff_policy_sha256"),
        )
        object.__setattr__(
            self,
            "lineage_sha256",
            _sha256(self.lineage_sha256, "lineage_sha256"),
        )
        if not isinstance(self.entries, tuple):
            raise CognitiveMemoryError("entries must be an immutable tuple")
        if len(self.entries) > self.capacity:
            raise CognitiveMemoryError("entries exceed the fixed memory capacity")
        for entry in self.entries:
            if not isinstance(entry, EpisodicMemoryEntry):
                raise CognitiveMemoryError(
                    "entries must contain only EpisodicMemoryEntry values"
                )
            entry.validate_cutoff(cutoff)
        object.__setattr__(
            self, "entries", tuple(sorted(self.entries, key=_entry_order))
        )

    def append_with_eviction(
        self, entry: EpisodicMemoryEntry
    ) -> tuple[EpisodicMemory, EpisodicMemoryEntry | None]:
        """Append purely and evict the oldest entry by a stable semantic order."""
        if not isinstance(entry, EpisodicMemoryEntry):
            raise CognitiveMemoryError("entry must be an EpisodicMemoryEntry")
        entry.validate_cutoff(self.trusted_cutoff)
        candidates = tuple(sorted((*self.entries, entry), key=_entry_order))
        evicted = candidates[0] if len(candidates) > self.capacity else None
        kept = candidates[-self.capacity :] if self.capacity else ()
        return (
            EpisodicMemory(
                capacity=self.capacity,
                trusted_cutoff=self.trusted_cutoff,
                cutoff_policy_sha256=self.cutoff_policy_sha256,
                lineage_sha256=self.lineage_sha256,
                entries=kept,
            ),
            evicted,
        )

    def append(self, entry: EpisodicMemoryEntry) -> EpisodicMemory:
        """Return memory after deterministic append and possible eviction."""
        memory, _ = self.append_with_eviction(entry)
        return memory

    def append_information_event(
        self,
        event: InformationEvent,
        *,
        attention: float,
        strength: float,
        observed_time: datetime | None = None,
    ) -> EpisodicMemory:
        """Validate and append one temporally firewalled information event."""
        _validate_information_event(
            event,
            trusted_cutoff=self.trusted_cutoff,
            cutoff_policy_sha256=self.cutoff_policy_sha256,
            lineage_sha256=self.lineage_sha256,
        )
        entry = EpisodicMemoryEntry.from_information_event(
            event,
            attention=attention,
            strength=strength,
            observed_time=observed_time,
        )
        return self.append(entry)

    def to_mapping(self) -> dict[str, object]:
        """Return capacity, firewall commitments, and canonical entries."""
        return {
            "capacity": self.capacity,
            "cutoff_policy_sha256": self.cutoff_policy_sha256,
            "entries": [entry.to_mapping() for entry in self.entries],
            "lineage_sha256": self.lineage_sha256,
            "trusted_cutoff": self.trusted_cutoff.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class SemanticBelief:
    """One bounded belief with content and correction lineage but no text."""

    topic_id: str
    value: float = 0.0
    last_update_time: datetime | None = None
    evidence_event_ids: tuple[str, ...] = ()
    evidence_content_sha256: tuple[str, ...] = ()
    correction_lineage: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "topic_id", _identifier(self.topic_id, "topic_id"))
        object.__setattr__(self, "value", _belief_value(self.value, "value"))
        if self.last_update_time is not None:
            object.__setattr__(
                self,
                "last_update_time",
                _aware_utc(self.last_update_time, "last_update_time"),
            )
        if not isinstance(self.evidence_event_ids, tuple) or not isinstance(
            self.evidence_content_sha256, tuple
        ):
            raise CognitiveMemoryError("evidence lineage must use immutable tuples")
        event_ids = tuple(
            _identifier(value, "evidence_event_id") for value in self.evidence_event_ids
        )
        content_hashes = tuple(
            _sha256(value, "evidence_content_sha256")
            for value in self.evidence_content_sha256
        )
        if len(event_ids) != len(content_hashes):
            raise CognitiveMemoryError(
                "event and content evidence lineage lengths must match"
            )
        if len(set(event_ids)) != len(event_ids):
            raise CognitiveMemoryError("evidence event identifiers must be unique")
        if event_ids and self.last_update_time is None:
            raise CognitiveMemoryError("evidence lineage requires last_update_time")
        if not event_ids and self.value != 0.0:
            raise CognitiveMemoryError("non-zero belief requires evidence lineage")
        if not isinstance(self.correction_lineage, tuple):
            raise CognitiveMemoryError("correction_lineage must be an immutable tuple")
        normalized_corrections: list[tuple[str, str]] = []
        correction_ids: set[str] = set()
        for pair in self.correction_lineage:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise CognitiveMemoryError(
                    "correction lineage entries must be (correction, target) tuples"
                )
            correction_id = _identifier(pair[0], "correction_event_id")
            target_id = _identifier(pair[1], "correction_target_event_id")
            if correction_id == target_id:
                raise CognitiveMemoryError("a correction cannot target itself")
            if correction_id in correction_ids:
                raise CognitiveMemoryError(
                    "correction event identifiers must be unique"
                )
            if correction_id not in event_ids or target_id not in event_ids:
                raise CognitiveMemoryError(
                    "correction and target must both appear in evidence lineage"
                )
            correction_ids.add(correction_id)
            normalized_corrections.append((correction_id, target_id))
        object.__setattr__(self, "evidence_event_ids", event_ids)
        object.__setattr__(self, "evidence_content_sha256", content_hashes)
        object.__setattr__(self, "correction_lineage", tuple(normalized_corrections))

    def to_mapping(self) -> dict[str, object]:
        """Return the belief and its complete anonymous evidence lineage."""
        return {
            "correction_lineage": [
                {"correction_event_id": correction, "target_event_id": target}
                for correction, target in self.correction_lineage
            ],
            "evidence_content_sha256": list(self.evidence_content_sha256),
            "evidence_event_ids": list(self.evidence_event_ids),
            "last_update_time": (
                None
                if self.last_update_time is None
                else self.last_update_time.isoformat()
            ),
            "topic_id": self.topic_id,
            "value": self.value,
        }


@dataclass(frozen=True, slots=True)
class SemanticState:
    """Canonical semantic beliefs bound to one temporal and data lineage."""

    trusted_cutoff: datetime
    cutoff_policy_sha256: str
    lineage_sha256: str
    beliefs: tuple[SemanticBelief, ...] = ()

    def __post_init__(self) -> None:
        cutoff = _aware_utc(self.trusted_cutoff, "trusted_cutoff")
        object.__setattr__(self, "trusted_cutoff", cutoff)
        object.__setattr__(
            self,
            "cutoff_policy_sha256",
            _sha256(self.cutoff_policy_sha256, "cutoff_policy_sha256"),
        )
        object.__setattr__(
            self,
            "lineage_sha256",
            _sha256(self.lineage_sha256, "lineage_sha256"),
        )
        if not isinstance(self.beliefs, tuple):
            raise CognitiveMemoryError("beliefs must be an immutable tuple")
        topics: set[str] = set()
        for belief in self.beliefs:
            if not isinstance(belief, SemanticBelief):
                raise CognitiveMemoryError(
                    "beliefs must contain only SemanticBelief values"
                )
            if belief.topic_id in topics:
                raise CognitiveMemoryError("semantic topic identifiers must be unique")
            if belief.last_update_time is not None and belief.last_update_time > cutoff:
                raise CognitiveMemoryTemporalError(
                    "semantic belief was updated after trusted_cutoff"
                )
            topics.add(belief.topic_id)
        object.__setattr__(
            self,
            "beliefs",
            tuple(sorted(self.beliefs, key=lambda belief: belief.topic_id)),
        )

    def belief_for(self, topic_id: str) -> SemanticBelief | None:
        """Return one belief by anonymous topic identifier."""
        normalized_topic = _identifier(topic_id, "topic_id")
        return next(
            (belief for belief in self.beliefs if belief.topic_id == normalized_topic),
            None,
        )

    def value_for(self, topic_id: str) -> float:
        """Return a topic value, using neutral zero for an unseen event."""
        belief = self.belief_for(topic_id)
        return 0.0 if belief is None else belief.value

    def _with_belief(self, updated: SemanticBelief) -> SemanticState:
        retained = tuple(
            belief for belief in self.beliefs if belief.topic_id != updated.topic_id
        )
        return SemanticState(
            trusted_cutoff=self.trusted_cutoff,
            cutoff_policy_sha256=self.cutoff_policy_sha256,
            lineage_sha256=self.lineage_sha256,
            beliefs=(*retained, updated),
        )

    def _validate_event(self, event: InformationEvent) -> None:
        _validate_information_event(
            event,
            trusted_cutoff=self.trusted_cutoff,
            cutoff_policy_sha256=self.cutoff_policy_sha256,
            lineage_sha256=self.lineage_sha256,
        )

    def update_from_event(
        self,
        event: InformationEvent,
        *,
        target_belief: float,
        learning_rate: float = 1.0,
        attention: float = 1.0,
    ) -> SemanticState:
        """Move an event-keyed belief toward a bounded target without mutation."""
        self._validate_event(event)
        if event.event_kind is InformationEventKind.CORRECTION:
            raise CognitiveMemoryError(
                "correction events must use apply_correction or update dispatch"
            )
        target = _belief_value(target_belief, "target_belief")
        rate = _unit_interval(learning_rate, "learning_rate")
        attended = _unit_interval(attention, "attention")
        existing = self.belief_for(event.event_id)
        if existing is not None and event.event_id in existing.evidence_event_ids:
            evidence_index = existing.evidence_event_ids.index(event.event_id)
            if existing.evidence_content_sha256[evidence_index] != event.content_sha256:
                raise CognitiveMemoryError(
                    "event_id was reused with a conflicting content hash"
                )
            return self
        if existing is None:
            existing = SemanticBelief(topic_id=event.event_id)
        if (
            existing.last_update_time is not None
            and event.as_of_time < existing.last_update_time
        ):
            raise CognitiveMemoryTemporalError(
                "event update would move semantic state backward in time"
            )
        event_weight = min(1.0, event.magnitude) * event.credibility
        effective_rate = rate * attended * event_weight
        value = existing.value + effective_rate * (target - existing.value)
        updated = SemanticBelief(
            topic_id=existing.topic_id,
            value=value,
            last_update_time=event.as_of_time,
            evidence_event_ids=(*existing.evidence_event_ids, event.event_id),
            evidence_content_sha256=(
                *existing.evidence_content_sha256,
                event.content_sha256,
            ),
            correction_lineage=existing.correction_lineage,
        )
        return self._with_belief(updated)

    def apply_correction(
        self,
        event: InformationEvent,
        *,
        correction_fraction: float,
        attention: float = 1.0,
    ) -> SemanticState:
        """Partially reduce a targeted belief while preserving both lineages."""
        self._validate_event(event)
        if event.event_kind is not InformationEventKind.CORRECTION:
            raise CognitiveMemoryError("event must be a correction")
        if event.corrects_event_id is None:
            raise CognitiveMemoryError("correction is missing corrects_event_id")
        fraction = _unit_interval(correction_fraction, "correction_fraction")
        attended = _unit_interval(attention, "attention")
        existing = self.belief_for(event.corrects_event_id)
        if (
            existing is None
            or event.corrects_event_id not in existing.evidence_event_ids
        ):
            raise CognitiveMemoryError(
                "correction target is absent from semantic state"
            )
        if event.event_id in existing.evidence_event_ids:
            evidence_index = existing.evidence_event_ids.index(event.event_id)
            if existing.evidence_content_sha256[evidence_index] != event.content_sha256:
                raise CognitiveMemoryError(
                    "correction event_id has a conflicting content hash"
                )
            if (event.event_id, event.corrects_event_id) in existing.correction_lineage:
                return self
            raise CognitiveMemoryError(
                "correction event_id already exists with incompatible lineage"
            )
        if (
            existing.last_update_time is not None
            and event.as_of_time < existing.last_update_time
        ):
            raise CognitiveMemoryTemporalError(
                "correction would move semantic state backward in time"
            )
        event_weight = min(1.0, event.magnitude) * event.credibility
        effective_fraction = fraction * attended * event_weight
        updated = SemanticBelief(
            topic_id=existing.topic_id,
            value=existing.value * (1.0 - effective_fraction),
            last_update_time=event.as_of_time,
            evidence_event_ids=(*existing.evidence_event_ids, event.event_id),
            evidence_content_sha256=(
                *existing.evidence_content_sha256,
                event.content_sha256,
            ),
            correction_lineage=(
                *existing.correction_lineage,
                (event.event_id, event.corrects_event_id),
            ),
        )
        return self._with_belief(updated)

    def update(
        self,
        event: InformationEvent,
        *,
        target_belief: float | None = None,
        learning_rate: float = 1.0,
        attention: float = 1.0,
        correction_fraction: float | None = None,
    ) -> SemanticState:
        """Dispatch ordinary and correction events under a strict argument schema."""
        if not isinstance(event, InformationEvent):
            raise CognitiveMemoryError("event must be an InformationEvent")
        if event.event_kind is InformationEventKind.CORRECTION:
            if target_belief is not None:
                raise CognitiveMemoryError(
                    "correction update must not supply target_belief"
                )
            if correction_fraction is None:
                raise CognitiveMemoryError(
                    "correction update requires correction_fraction"
                )
            if learning_rate != 1.0:
                raise CognitiveMemoryError(
                    "correction update must not supply learning_rate"
                )
            return self.apply_correction(
                event,
                correction_fraction=correction_fraction,
                attention=attention,
            )
        if target_belief is None:
            raise CognitiveMemoryError("ordinary update requires target_belief")
        if correction_fraction is not None:
            raise CognitiveMemoryError(
                "ordinary update must not supply correction_fraction"
            )
        return self.update_from_event(
            event,
            target_belief=target_belief,
            learning_rate=learning_rate,
            attention=attention,
        )

    def decay(
        self,
        *,
        as_of_time: datetime,
        decay_rate: float,
        elapsed_steps: int = 1,
    ) -> SemanticState:
        """Exponentially decay belief magnitudes without changing evidence lineage."""
        normalized_as_of = _aware_utc(as_of_time, "as_of_time")
        if normalized_as_of > self.trusted_cutoff:
            raise CognitiveMemoryTemporalError("decay as_of_time is after cutoff")
        rate = _non_negative(decay_rate, "decay_rate")
        if (
            isinstance(elapsed_steps, bool)
            or not isinstance(elapsed_steps, int)
            or elapsed_steps < 0
        ):
            raise CognitiveMemoryError("elapsed_steps must be a non-negative integer")
        if elapsed_steps == 0 or not self.beliefs:
            return self
        factor = math.exp(-rate * elapsed_steps)
        decayed: list[SemanticBelief] = []
        for belief in self.beliefs:
            if belief.last_update_time is None:
                decayed.append(belief)
                continue
            if normalized_as_of < belief.last_update_time:
                raise CognitiveMemoryTemporalError(
                    "decay would move semantic state backward in time"
                )
            decayed.append(
                SemanticBelief(
                    topic_id=belief.topic_id,
                    value=belief.value * factor,
                    last_update_time=normalized_as_of,
                    evidence_event_ids=belief.evidence_event_ids,
                    evidence_content_sha256=belief.evidence_content_sha256,
                    correction_lineage=belief.correction_lineage,
                )
            )
        return SemanticState(
            trusted_cutoff=self.trusted_cutoff,
            cutoff_policy_sha256=self.cutoff_policy_sha256,
            lineage_sha256=self.lineage_sha256,
            beliefs=tuple(decayed),
        )

    def to_mapping(self) -> dict[str, object]:
        """Return the canonical semantic state and firewall commitments."""
        return {
            "beliefs": [belief.to_mapping() for belief in self.beliefs],
            "cutoff_policy_sha256": self.cutoff_policy_sha256,
            "lineage_sha256": self.lineage_sha256,
            "trusted_cutoff": self.trusted_cutoff.isoformat(),
        }


__all__ = [
    "CognitiveMemoryError",
    "CognitiveMemoryTemporalError",
    "EpisodicMemory",
    "EpisodicMemoryEntry",
    "SemanticBelief",
    "SemanticState",
]
