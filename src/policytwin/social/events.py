"""Privacy-preserving, temporally guarded information events."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from policytwin.counterfactual.events import ScenarioEventReceipt
from policytwin.provenance import canonical_json_bytes, sha256_bytes

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class InformationEventError(ValueError):
    """Raised when an information event is malformed or semantically unsafe."""


class InformationEventTemporalError(InformationEventError):
    """Raised when an event crosses its declared knowledge-time boundary."""


class InformationEventKind(StrEnum):
    """Permitted information channels without retaining source text."""

    NEWS = "news"
    ANNOUNCEMENT = "announcement"
    SOCIAL_POST = "social_post"
    RUMOR = "rumor"
    CORRECTION = "correction"


def _identifier(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise InformationEventError(
            f"{field_name} must be a stable anonymous identifier"
        )
    return value


def _aware_utc(value: object, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise InformationEventError(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise InformationEventError(f"{field_name} must include an explicit timezone")
    try:
        return value.astimezone(UTC)
    except (OverflowError, ValueError) as error:
        raise InformationEventError(f"{field_name} is outside the UTC range") from error


def _finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise InformationEventError(f"{field_name} must be a finite real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise InformationEventError(f"{field_name} must be finite")
    return normalized


def _sha256(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise InformationEventError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _scenario_payload_mapping(
    *,
    event_id: str,
    anonymous_source_id: str,
    event_kind: InformationEventKind,
    sentiment: float,
    magnitude: float,
    credibility: float,
    content_sha256: str,
    corrects_event_id: str | None,
) -> dict[str, object]:
    return {
        "anonymous_source_id": anonymous_source_id,
        "content_sha256": content_sha256,
        "corrects_event_id": corrects_event_id,
        "credibility": credibility,
        "event_id": event_id,
        "event_kind": event_kind.value,
        "magnitude": magnitude,
        "sentiment": sentiment,
    }


def scenario_information_payload_sha256(
    *,
    event_id: str,
    anonymous_source_id: str,
    event_kind: InformationEventKind,
    sentiment: float,
    magnitude: float,
    credibility: float,
    content_sha256: str,
    corrects_event_id: str | None = None,
) -> str:
    """Commit a text-free generated payload before engine receipt issuance."""
    normalized_event_id = _identifier(event_id, "event_id")
    normalized_source = _identifier(anonymous_source_id, "anonymous_source_id")
    if not isinstance(event_kind, InformationEventKind):
        raise InformationEventError("event_kind must be an InformationEventKind")
    normalized_sentiment = _finite_number(sentiment, "sentiment")
    if not -1.0 <= normalized_sentiment <= 1.0:
        raise InformationEventError("sentiment must be in [-1, 1]")
    normalized_magnitude = _finite_number(magnitude, "magnitude")
    if normalized_magnitude < 0.0:
        raise InformationEventError("magnitude must be non-negative")
    normalized_credibility = _finite_number(credibility, "credibility")
    if not 0.0 <= normalized_credibility <= 1.0:
        raise InformationEventError("credibility must be in [0, 1]")
    normalized_content = _sha256(content_sha256, "content_sha256")
    normalized_correction = corrects_event_id
    if event_kind is InformationEventKind.CORRECTION:
        normalized_correction = _identifier(normalized_correction, "corrects_event_id")
        if normalized_correction == normalized_event_id:
            raise InformationEventError("a correction cannot correct itself")
    elif normalized_correction is not None:
        raise InformationEventError(
            "corrects_event_id is only valid for correction events"
        )
    return sha256_bytes(
        canonical_json_bytes(
            _scenario_payload_mapping(
                event_id=normalized_event_id,
                anonymous_source_id=normalized_source,
                event_kind=event_kind,
                sentiment=normalized_sentiment,
                magnitude=normalized_magnitude,
                credibility=normalized_credibility,
                content_sha256=normalized_content,
                corrects_event_id=normalized_correction,
            )
        )
    )


@dataclass(frozen=True, slots=True)
class InformationEvent:
    """An anonymous information event represented only by numeric semantics.

    ``sentiment`` is a signed score in ``[-1, 1]``, ``magnitude`` is a
    non-negative intensity, and ``credibility`` is in ``[0, 1]``. Raw content,
    author names, handles, and participant identities are deliberately absent.
    """

    event_id: str
    anonymous_source_id: str
    event_kind: InformationEventKind
    sentiment: float
    magnitude: float
    credibility: float
    content_sha256: str
    lineage_sha256: str
    cutoff_policy_sha256: str
    event_time: datetime
    available_time: datetime
    as_of_time: datetime
    trusted_cutoff: datetime
    corrects_event_id: str | None = None

    def __post_init__(self) -> None:
        event_id = _identifier(self.event_id, "event_id")
        anonymous_source_id = _identifier(
            self.anonymous_source_id, "anonymous_source_id"
        )
        if not isinstance(self.event_kind, InformationEventKind):
            raise InformationEventError("event_kind must be an InformationEventKind")

        sentiment = _finite_number(self.sentiment, "sentiment")
        if not -1.0 <= sentiment <= 1.0:
            raise InformationEventError("sentiment must be in [-1, 1]")
        magnitude = _finite_number(self.magnitude, "magnitude")
        if magnitude < 0.0:
            raise InformationEventError("magnitude must be non-negative")
        credibility = _finite_number(self.credibility, "credibility")
        if not 0.0 <= credibility <= 1.0:
            raise InformationEventError("credibility must be in [0, 1]")

        event_time = _aware_utc(self.event_time, "event_time")
        available_time = _aware_utc(self.available_time, "available_time")
        as_of_time = _aware_utc(self.as_of_time, "as_of_time")
        trusted_cutoff = _aware_utc(self.trusted_cutoff, "trusted_cutoff")
        if as_of_time > trusted_cutoff:
            raise InformationEventTemporalError("as_of_time is after trusted_cutoff")
        if event_time > as_of_time:
            raise InformationEventTemporalError("event_time is after as_of_time")
        if available_time > as_of_time:
            raise InformationEventTemporalError("available_time is after as_of_time")
        if available_time > trusted_cutoff:
            raise InformationEventTemporalError(
                "available_time is after trusted_cutoff"
            )

        corrects_event_id = self.corrects_event_id
        if self.event_kind is InformationEventKind.CORRECTION:
            corrects_event_id = _identifier(corrects_event_id, "corrects_event_id")
            if corrects_event_id == event_id:
                raise InformationEventError("a correction cannot correct itself")
        elif corrects_event_id is not None:
            raise InformationEventError(
                "corrects_event_id is only valid for correction events"
            )

        object.__setattr__(self, "event_id", event_id)
        object.__setattr__(self, "anonymous_source_id", anonymous_source_id)
        object.__setattr__(self, "sentiment", sentiment)
        object.__setattr__(self, "magnitude", magnitude)
        object.__setattr__(self, "credibility", credibility)
        object.__setattr__(
            self,
            "content_sha256",
            _sha256(self.content_sha256, "content_sha256"),
        )
        object.__setattr__(
            self,
            "lineage_sha256",
            _sha256(self.lineage_sha256, "lineage_sha256"),
        )
        object.__setattr__(
            self,
            "cutoff_policy_sha256",
            _sha256(self.cutoff_policy_sha256, "cutoff_policy_sha256"),
        )
        object.__setattr__(self, "event_time", event_time)
        object.__setattr__(self, "available_time", available_time)
        object.__setattr__(self, "as_of_time", as_of_time)
        object.__setattr__(self, "trusted_cutoff", trusted_cutoff)
        object.__setattr__(self, "corrects_event_id", corrects_event_id)

    def to_mapping(self) -> dict[str, object]:
        """Return numeric semantics and hashes without retaining raw text."""
        return {
            "anonymous_source_id": self.anonymous_source_id,
            "as_of_time": self.as_of_time.isoformat(),
            "available_time": self.available_time.isoformat(),
            "content_sha256": self.content_sha256,
            "corrects_event_id": self.corrects_event_id,
            "credibility": self.credibility,
            "cutoff_policy_sha256": self.cutoff_policy_sha256,
            "event_id": self.event_id,
            "event_kind": self.event_kind.value,
            "event_time": self.event_time.isoformat(),
            "lineage_sha256": self.lineage_sha256,
            "magnitude": self.magnitude,
            "sentiment": self.sentiment,
            "trusted_cutoff": self.trusted_cutoff.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class ScenarioInformationEvent:
    """Anonymous information generated inside a post-cutoff scenario.

    Unlike :class:`InformationEvent`, this type has no observed
    ``available_time``.  Its scenario time and pre-cutoff generator lineage are
    authorized by an engine-issued :class:`ScenarioEventReceipt`.  Keeping the
    classes separate prevents a simulated future event from weakening the
    admissibility rules for observed records.
    """

    event_id: str
    anonymous_source_id: str
    event_kind: InformationEventKind
    sentiment: float
    magnitude: float
    credibility: float
    content_sha256: str
    receipt: ScenarioEventReceipt
    corrects_event_id: str | None = None

    def __post_init__(self) -> None:
        event_id = _identifier(self.event_id, "event_id")
        anonymous_source_id = _identifier(
            self.anonymous_source_id, "anonymous_source_id"
        )
        if not isinstance(self.event_kind, InformationEventKind):
            raise InformationEventError("event_kind must be an InformationEventKind")
        sentiment = _finite_number(self.sentiment, "sentiment")
        if not -1.0 <= sentiment <= 1.0:
            raise InformationEventError("sentiment must be in [-1, 1]")
        magnitude = _finite_number(self.magnitude, "magnitude")
        if magnitude < 0.0:
            raise InformationEventError("magnitude must be non-negative")
        credibility = _finite_number(self.credibility, "credibility")
        if not 0.0 <= credibility <= 1.0:
            raise InformationEventError("credibility must be in [0, 1]")
        content_sha256 = _sha256(self.content_sha256, "content_sha256")
        if not isinstance(self.receipt, ScenarioEventReceipt):
            raise InformationEventError(
                "scenario information requires an engine-issued receipt"
            )
        corrects_event_id = self.corrects_event_id
        if self.event_kind is InformationEventKind.CORRECTION:
            corrects_event_id = _identifier(corrects_event_id, "corrects_event_id")
            if corrects_event_id == event_id:
                raise InformationEventError("a correction cannot correct itself")
        elif corrects_event_id is not None:
            raise InformationEventError(
                "corrects_event_id is only valid for correction events"
            )
        payload_sha256 = scenario_information_payload_sha256(
            event_id=event_id,
            anonymous_source_id=anonymous_source_id,
            event_kind=self.event_kind,
            sentiment=sentiment,
            magnitude=magnitude,
            credibility=credibility,
            content_sha256=content_sha256,
            corrects_event_id=corrects_event_id,
        )
        if payload_sha256 != self.receipt.payload_sha256:
            raise InformationEventError(
                "scenario information payload differs from its engine receipt"
            )
        object.__setattr__(self, "event_id", event_id)
        object.__setattr__(self, "anonymous_source_id", anonymous_source_id)
        object.__setattr__(self, "sentiment", sentiment)
        object.__setattr__(self, "magnitude", magnitude)
        object.__setattr__(self, "credibility", credibility)
        object.__setattr__(self, "content_sha256", content_sha256)
        object.__setattr__(self, "corrects_event_id", corrects_event_id)

    @property
    def event_time(self) -> datetime:
        """Return the simulated event time, not an observed-record time."""
        return self.receipt.scenario_time

    @property
    def as_of_time(self) -> datetime:
        """Return the current within-scenario time."""
        return self.receipt.scenario_time

    @property
    def trusted_cutoff(self) -> datetime:
        """Return the cutoff of the generator's lawful information set."""
        return self.receipt.trusted_cutoff

    @property
    def lineage_sha256(self) -> str:
        """Return the frozen generator source-lineage commitment."""
        return self.receipt.generator_spec.source_lineage_sha256

    @property
    def cutoff_policy_sha256(self) -> str:
        """Return the cutoff-policy commitment bound by the generator."""
        return self.receipt.generator_spec.cutoff_policy_sha256

    def to_mapping(self) -> dict[str, object]:
        """Return generated semantics without inventing ``available_time``."""
        return {
            **_scenario_payload_mapping(
                event_id=self.event_id,
                anonymous_source_id=self.anonymous_source_id,
                event_kind=self.event_kind,
                sentiment=self.sentiment,
                magnitude=self.magnitude,
                credibility=self.credibility,
                content_sha256=self.content_sha256,
                corrects_event_id=self.corrects_event_id,
            ),
            "as_of_time": self.as_of_time.isoformat(),
            "event_time": self.event_time.isoformat(),
            "lineage_sha256": self.lineage_sha256,
            "provenance": "model_generated_scenario",
            "scenario_event_receipt": self.receipt.to_mapping(),
            "trusted_cutoff": self.trusted_cutoff.isoformat(),
        }


__all__ = [
    "InformationEvent",
    "InformationEventError",
    "InformationEventKind",
    "InformationEventTemporalError",
    "ScenarioInformationEvent",
    "scenario_information_payload_sha256",
]
