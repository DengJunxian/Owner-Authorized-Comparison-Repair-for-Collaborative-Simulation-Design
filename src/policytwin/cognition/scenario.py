"""Receipt-bound cognition for model-generated future scenario events.

Observed memory remains governed by :mod:`policytwin.cognition.memory` and can
never cross its trusted cutoff.  The types here are deliberately separate:
they retain only anonymous, text-free events carrying engine-issued scenario
receipts, so a simulated future cannot be confused with an observed record.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import NoReturn

from policytwin.cognition.memory import SemanticBelief
from policytwin.counterfactual import EventKey
from policytwin.provenance import canonical_json_bytes, sha256_bytes
from policytwin.social.audit import (
    NULL_COMMITMENT,
    SocialAuditLog,
    SocialAuditOutcome,
    SocialAuditRule,
    SocialAuditStage,
)
from policytwin.social.events import (
    InformationEventKind,
    ScenarioInformationEvent,
)

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ScenarioCognitionError(ValueError):
    """Raised when generated-scenario cognitive state is invalid."""


class ScenarioCognitionTemporalError(ScenarioCognitionError):
    """Raised when a generated event moves its scenario state backward."""


def _fail(message: str) -> NoReturn:
    raise ScenarioCognitionError(message)


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        _fail(f"{field} must be a stable anonymous identifier")
    return value


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail(f"{field} must be a lowercase SHA-256 digest")
    return value


def _aware(value: object, field: str) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        _fail(f"{field} must be a timezone-aware datetime")
    try:
        return value.astimezone(UTC)
    except (OverflowError, ValueError) as error:
        raise ScenarioCognitionError(f"{field} is outside the UTC range") from error


def _unit(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        _fail(f"{field} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        _fail(f"{field} must be finite and in [0, 1]")
    return normalized


def _belief(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        _fail(f"{field} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or not -1.0 <= normalized <= 1.0:
        _fail(f"{field} must be finite and in [-1, 1]")
    return normalized


def _commit(value: object) -> str:
    return sha256_bytes(canonical_json_bytes(value))


@dataclass(frozen=True, slots=True)
class ScenarioMemoryEntry:
    """One attended, text-free event in simulated scenario time."""

    event_id: str
    anonymous_source_id: str
    event_kind: InformationEventKind
    content_sha256: str
    scenario_time: datetime
    scenario_receipt_sha256: str
    attention: float
    strength: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", _identifier(self.event_id, "event_id"))
        object.__setattr__(
            self,
            "anonymous_source_id",
            _identifier(self.anonymous_source_id, "anonymous_source_id"),
        )
        if not isinstance(self.event_kind, InformationEventKind):
            _fail("event_kind must be an InformationEventKind")
        object.__setattr__(
            self,
            "content_sha256",
            _sha256(self.content_sha256, "content_sha256"),
        )
        object.__setattr__(
            self, "scenario_time", _aware(self.scenario_time, "scenario_time")
        )
        object.__setattr__(
            self,
            "scenario_receipt_sha256",
            _sha256(self.scenario_receipt_sha256, "scenario_receipt_sha256"),
        )
        object.__setattr__(self, "attention", _unit(self.attention, "attention"))
        object.__setattr__(self, "strength", _unit(self.strength, "strength"))

    @classmethod
    def from_event(
        cls,
        event: ScenarioInformationEvent,
        *,
        attention: float,
        strength: float,
    ) -> ScenarioMemoryEntry:
        """Create an entry only from a validated generated event."""
        if not isinstance(event, ScenarioInformationEvent):
            _fail("scenario memory accepts only ScenarioInformationEvent")
        return cls(
            event_id=event.event_id,
            anonymous_source_id=event.anonymous_source_id,
            event_kind=event.event_kind,
            content_sha256=event.content_sha256,
            scenario_time=event.as_of_time,
            scenario_receipt_sha256=event.receipt.engine_signature_sha256,
            attention=attention,
            strength=strength,
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "anonymous_source_id": self.anonymous_source_id,
            "attention": self.attention,
            "content_sha256": self.content_sha256,
            "event_id": self.event_id,
            "event_kind": self.event_kind.value,
            "scenario_receipt_sha256": self.scenario_receipt_sha256,
            "scenario_time": self.scenario_time.isoformat(),
            "strength": self.strength,
        }


def _entry_order(entry: ScenarioMemoryEntry) -> tuple[object, ...]:
    return (
        entry.scenario_time,
        entry.event_id,
        entry.scenario_receipt_sha256,
    )


@dataclass(frozen=True, slots=True)
class ScenarioEpisodicMemory:
    """Fixed-capacity memory tied to one paired scenario and generator."""

    scenario_id: str
    capacity: int
    trusted_cutoff: datetime
    cutoff_policy_sha256: str
    source_lineage_sha256: str
    generator_spec_sha256: str
    entries: tuple[ScenarioMemoryEntry, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "scenario_id", _identifier(self.scenario_id, "scenario_id")
        )
        if (
            isinstance(self.capacity, bool)
            or not isinstance(self.capacity, int)
            or self.capacity < 0
        ):
            _fail("capacity must be a non-negative integer")
        object.__setattr__(
            self, "trusted_cutoff", _aware(self.trusted_cutoff, "trusted_cutoff")
        )
        for field in (
            "cutoff_policy_sha256",
            "source_lineage_sha256",
            "generator_spec_sha256",
        ):
            object.__setattr__(self, field, _sha256(getattr(self, field), field))
        if not isinstance(self.entries, tuple) or not all(
            isinstance(item, ScenarioMemoryEntry) for item in self.entries
        ):
            _fail("entries must be an immutable tuple of ScenarioMemoryEntry")
        if len(self.entries) > self.capacity:
            _fail("scenario memory entries exceed capacity")
        event_ids = tuple(item.event_id for item in self.entries)
        if len(set(event_ids)) != len(event_ids):
            _fail("scenario memory event IDs must be unique")
        if any(item.scenario_time <= self.trusted_cutoff for item in self.entries):
            _fail("scenario memory entries must be strictly post-cutoff")
        object.__setattr__(
            self,
            "entries",
            tuple(sorted(self.entries, key=_entry_order)),
        )

    def _validate_event(self, event: ScenarioInformationEvent) -> None:
        if not isinstance(event, ScenarioInformationEvent):
            _fail("scenario memory accepts only generated scenario information")
        spec = event.receipt.generator_spec
        if (
            event.receipt.scenario_id != self.scenario_id
            or event.trusted_cutoff != self.trusted_cutoff
            or event.cutoff_policy_sha256 != self.cutoff_policy_sha256
            or event.lineage_sha256 != self.source_lineage_sha256
            or spec.sha256 != self.generator_spec_sha256
        ):
            _fail("generated event differs from scenario-memory commitments")

    def append(
        self,
        event: ScenarioInformationEvent,
        *,
        attention: float,
        strength: float,
    ) -> ScenarioEpisodicMemory:
        """Append atomically and evict by a deterministic semantic order."""
        self._validate_event(event)
        entry = ScenarioMemoryEntry.from_event(
            event,
            attention=attention,
            strength=strength,
        )
        existing = next(
            (item for item in self.entries if item.event_id == entry.event_id), None
        )
        if existing is not None:
            if existing != entry:
                _fail("scenario event_id was reused with conflicting content")
            return self
        candidates = tuple(sorted((*self.entries, entry), key=_entry_order))
        kept = candidates[-self.capacity :] if self.capacity else ()
        return ScenarioEpisodicMemory(
            scenario_id=self.scenario_id,
            capacity=self.capacity,
            trusted_cutoff=self.trusted_cutoff,
            cutoff_policy_sha256=self.cutoff_policy_sha256,
            source_lineage_sha256=self.source_lineage_sha256,
            generator_spec_sha256=self.generator_spec_sha256,
            entries=kept,
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "capacity": self.capacity,
            "cutoff_policy_sha256": self.cutoff_policy_sha256,
            "entries": [item.to_mapping() for item in self.entries],
            "generator_spec_sha256": self.generator_spec_sha256,
            "scenario_id": self.scenario_id,
            "source_lineage_sha256": self.source_lineage_sha256,
            "trusted_cutoff": self.trusted_cutoff.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class ScenarioSemanticState:
    """Generated-event beliefs that cannot accept observed information."""

    scenario_id: str
    trusted_cutoff: datetime
    cutoff_policy_sha256: str
    source_lineage_sha256: str
    generator_spec_sha256: str
    beliefs: tuple[SemanticBelief, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "scenario_id", _identifier(self.scenario_id, "scenario_id")
        )
        object.__setattr__(
            self, "trusted_cutoff", _aware(self.trusted_cutoff, "trusted_cutoff")
        )
        for field in (
            "cutoff_policy_sha256",
            "source_lineage_sha256",
            "generator_spec_sha256",
        ):
            object.__setattr__(self, field, _sha256(getattr(self, field), field))
        if not isinstance(self.beliefs, tuple) or not all(
            isinstance(item, SemanticBelief) for item in self.beliefs
        ):
            _fail("beliefs must be an immutable tuple of SemanticBelief")
        topics = tuple(item.topic_id for item in self.beliefs)
        if len(set(topics)) != len(topics):
            _fail("scenario semantic topics must be unique")
        if any(
            item.last_update_time is not None
            and item.last_update_time <= self.trusted_cutoff
            for item in self.beliefs
        ):
            _fail("scenario beliefs must be updated strictly after cutoff")
        object.__setattr__(
            self, "beliefs", tuple(sorted(self.beliefs, key=lambda item: item.topic_id))
        )

    def _validate_event(self, event: ScenarioInformationEvent) -> None:
        if not isinstance(event, ScenarioInformationEvent):
            _fail("scenario semantic state accepts only generated information")
        spec = event.receipt.generator_spec
        if (
            event.receipt.scenario_id != self.scenario_id
            or event.trusted_cutoff != self.trusted_cutoff
            or event.cutoff_policy_sha256 != self.cutoff_policy_sha256
            or event.lineage_sha256 != self.source_lineage_sha256
            or spec.sha256 != self.generator_spec_sha256
        ):
            _fail("generated event differs from semantic-state commitments")

    def belief_for(self, topic_id: str) -> SemanticBelief | None:
        normalized = _identifier(topic_id, "topic_id")
        return next(
            (item for item in self.beliefs if item.topic_id == normalized),
            None,
        )

    def _with_belief(self, updated: SemanticBelief) -> ScenarioSemanticState:
        retained = tuple(
            item for item in self.beliefs if item.topic_id != updated.topic_id
        )
        return ScenarioSemanticState(
            scenario_id=self.scenario_id,
            trusted_cutoff=self.trusted_cutoff,
            cutoff_policy_sha256=self.cutoff_policy_sha256,
            source_lineage_sha256=self.source_lineage_sha256,
            generator_spec_sha256=self.generator_spec_sha256,
            beliefs=(*retained, updated),
        )

    def update(
        self,
        event: ScenarioInformationEvent,
        *,
        target_belief: float | None = None,
        attention: float = 1.0,
        correction_fraction: float | None = None,
    ) -> ScenarioSemanticState:
        """Apply an ordinary signal or a partial correction in scenario time."""
        self._validate_event(event)
        attended = _unit(attention, "attention")
        event_weight = min(1.0, event.magnitude) * event.credibility * attended
        if event.event_kind is InformationEventKind.CORRECTION:
            if target_belief is not None or correction_fraction is None:
                _fail("scenario correction requires correction_fraction only")
            if event.corrects_event_id is None:
                _fail("scenario correction lacks its target")
            existing = self.belief_for(event.corrects_event_id)
            if existing is None:
                _fail("scenario correction target is absent")
            fraction = _unit(correction_fraction, "correction_fraction")
            if (
                existing.last_update_time is not None
                and event.as_of_time < existing.last_update_time
            ):
                raise ScenarioCognitionTemporalError(
                    "scenario correction moves belief backward"
                )
            if event.event_id in existing.evidence_event_ids:
                return self
            updated = SemanticBelief(
                topic_id=existing.topic_id,
                value=existing.value * (1.0 - fraction * event_weight),
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
        if target_belief is None or correction_fraction is not None:
            _fail("ordinary scenario update requires target_belief only")
        target = _belief(target_belief, "target_belief")
        existing = self.belief_for(event.event_id)
        if existing is not None:
            if event.event_id in existing.evidence_event_ids:
                index = existing.evidence_event_ids.index(event.event_id)
                if existing.evidence_content_sha256[index] != event.content_sha256:
                    _fail("scenario event_id has conflicting content")
                return self
        else:
            existing = SemanticBelief(topic_id=event.event_id)
        if (
            existing.last_update_time is not None
            and event.as_of_time < existing.last_update_time
        ):
            raise ScenarioCognitionTemporalError("scenario event moves belief backward")
        updated = SemanticBelief(
            topic_id=event.event_id,
            value=existing.value + event_weight * (target - existing.value),
            last_update_time=event.as_of_time,
            evidence_event_ids=(*existing.evidence_event_ids, event.event_id),
            evidence_content_sha256=(
                *existing.evidence_content_sha256,
                event.content_sha256,
            ),
            correction_lineage=existing.correction_lineage,
        )
        return self._with_belief(updated)

    def to_mapping(self) -> dict[str, object]:
        return {
            "beliefs": [item.to_mapping() for item in self.beliefs],
            "cutoff_policy_sha256": self.cutoff_policy_sha256,
            "generator_spec_sha256": self.generator_spec_sha256,
            "scenario_id": self.scenario_id,
            "source_lineage_sha256": self.source_lineage_sha256,
            "trusted_cutoff": self.trusted_cutoff.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class ScenarioCognitiveSnapshot:
    """Bound generated memory and semantic state for one scenario."""

    episodic_memory: ScenarioEpisodicMemory
    semantic_state: ScenarioSemanticState

    def __post_init__(self) -> None:
        valid_memory = isinstance(self.episodic_memory, ScenarioEpisodicMemory)
        valid_semantic = isinstance(self.semantic_state, ScenarioSemanticState)
        if not valid_memory or not valid_semantic:
            _fail("scenario snapshot requires generated memory and semantic state")
        memory_binding = (
            self.episodic_memory.scenario_id,
            self.episodic_memory.trusted_cutoff,
            self.episodic_memory.cutoff_policy_sha256,
            self.episodic_memory.source_lineage_sha256,
            self.episodic_memory.generator_spec_sha256,
        )
        semantic_binding = (
            self.semantic_state.scenario_id,
            self.semantic_state.trusted_cutoff,
            self.semantic_state.cutoff_policy_sha256,
            self.semantic_state.source_lineage_sha256,
            self.semantic_state.generator_spec_sha256,
        )
        if memory_binding != semantic_binding:
            _fail("scenario memory and semantic commitments differ")

    def to_mapping(self) -> dict[str, object]:
        return {
            "episodic_memory": self.episodic_memory.to_mapping(),
            "semantic_state": self.semantic_state.to_mapping(),
        }


@dataclass(frozen=True, slots=True)
class AuditedScenarioCognitiveLayer:
    """Atomic, audited transition gateway for generated scenario cognition."""

    audit_log: SocialAuditLog
    cutoff_policy_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.audit_log, SocialAuditLog):
            _fail("audit_log must be a SocialAuditLog")
        object.__setattr__(
            self,
            "cutoff_policy_sha256",
            _sha256(self.cutoff_policy_sha256, "cutoff_policy_sha256"),
        )

    def attend(
        self,
        snapshot: ScenarioCognitiveSnapshot,
        event: ScenarioInformationEvent,
        *,
        event_key: EventKey,
        attention: float,
        strength: float,
        target_belief: float | None = None,
        correction_fraction: float | None = None,
    ) -> ScenarioCognitiveSnapshot:
        """Atomically update scenario memory and belief with an audit record."""
        if not isinstance(snapshot, ScenarioCognitiveSnapshot):
            _fail("snapshot must be a ScenarioCognitiveSnapshot")
        if not isinstance(event, ScenarioInformationEvent):
            _fail("event must be a ScenarioInformationEvent")
        if not isinstance(event_key, EventKey):
            _fail("event_key must be an EventKey")
        input_value = {
            "attention": attention,
            "correction_fraction": correction_fraction,
            "event": event.to_mapping(),
            "snapshot": snapshot.to_mapping(),
            "strength": strength,
            "target_belief": target_belief,
        }
        input_commitment = _commit(input_value)
        try:
            if event.cutoff_policy_sha256 != self.cutoff_policy_sha256:
                _fail("scenario event cutoff policy differs from audited layer")
            memory = snapshot.episodic_memory.append(
                event,
                attention=attention,
                strength=strength,
            )
            semantic = snapshot.semantic_state.update(
                event,
                target_belief=target_belief,
                attention=attention,
                correction_fraction=correction_fraction,
            )
            result = ScenarioCognitiveSnapshot(memory, semantic)
        except ScenarioCognitionError as error:
            self.audit_log.record(
                stage=SocialAuditStage.MEMORY,
                outcome=SocialAuditOutcome.REJECTED,
                rule_code=(
                    SocialAuditRule.TEMPORAL_BOUNDARY
                    if isinstance(error, ScenarioCognitionTemporalError)
                    else SocialAuditRule.SCHEMA_VIOLATION
                ),
                event_key=event_key.value,
                policy_sha256=self.cutoff_policy_sha256,
                input_commitment=input_commitment,
                output_commitment=NULL_COMMITMENT,
                accepted_count=0,
                rejected_count=1,
            )
            raise
        self.audit_log.record(
            stage=SocialAuditStage.MEMORY,
            outcome=SocialAuditOutcome.ALLOWED,
            rule_code=SocialAuditRule.OK,
            event_key=event_key.value,
            policy_sha256=self.cutoff_policy_sha256,
            input_commitment=input_commitment,
            output_commitment=_commit(result.to_mapping()),
            accepted_count=1,
            rejected_count=0,
        )
        return result


__all__ = [
    "AuditedScenarioCognitiveLayer",
    "ScenarioCognitionError",
    "ScenarioCognitionTemporalError",
    "ScenarioCognitiveSnapshot",
    "ScenarioEpisodicMemory",
    "ScenarioMemoryEntry",
    "ScenarioSemanticState",
]
