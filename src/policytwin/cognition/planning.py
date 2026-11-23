"""Strict, text-free slow cognitive plans for constrained fast policies."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Self

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_RATIONALE_CODE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class CognitivePlanError(ValueError):
    """Raised when a slow cognitive plan violates its strict schema."""


class CognitivePlanTemporalError(CognitivePlanError):
    """Raised when a plan crosses its trusted knowledge-time boundary."""


class PlanAction(StrEnum):
    """Only directional actions understood by the current fast policy."""

    HOLD = "hold"
    BUY = "buy"
    SELL = "sell"


class RequestedUse(StrEnum):
    """Closed scientific uses for a slow-planning adapter."""

    SYNTHETIC_TEST = "synthetic_test"
    MAIN_POLICY = "main_policy"
    ABLATION_ONLY = "ablation_only"


_ACTION_ORDER = (PlanAction.HOLD, PlanAction.BUY, PlanAction.SELL)
_PLAN_FIELDS = frozenset(
    {
        "adapter_authorization_sha256",
        "adapter_id",
        "agent_id",
        "allowed_actions",
        "as_of_time",
        "confidence",
        "input_sha256",
        "rationale_code",
        "requested_use",
        "review_step",
        "topic_id",
        "trusted_cutoff",
        "valid_until",
    }
)


def _identifier(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise CognitivePlanError(f"{field_name} must be a stable anonymous identifier")
    return value


def _rationale_code(value: object) -> str:
    if not isinstance(value, str) or _RATIONALE_CODE_PATTERN.fullmatch(value) is None:
        raise CognitivePlanError(
            "rationale_code must be a stable code without free text"
        )
    return value


def _sha256(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise CognitivePlanError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _aware_utc(value: object, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise CognitivePlanError(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise CognitivePlanError(f"{field_name} must include an explicit timezone")
    try:
        return value.astimezone(UTC)
    except (OverflowError, ValueError) as error:
        raise CognitivePlanError(f"{field_name} is outside the UTC range") from error


def _parse_datetime(value: object, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise CognitivePlanError(f"{field_name} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise CognitivePlanError(f"{field_name} is not valid ISO-8601") from error
    return _aware_utc(parsed, field_name)


def _confidence(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CognitivePlanError("confidence must be a finite real number")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise CognitivePlanError("confidence must be finite and in [0, 1]")
    return normalized


def _review_step(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CognitivePlanError("review_step must be a non-negative integer")
    return value


def _allowed_actions(value: object, *, serialized: bool) -> tuple[PlanAction, ...]:
    if serialized:
        if not isinstance(value, list):
            raise CognitivePlanError("allowed_actions must be a JSON list")
        raw_actions = tuple(value)
    else:
        if not isinstance(value, tuple):
            raise CognitivePlanError("allowed_actions must be an immutable tuple")
        raw_actions = value
    if not raw_actions:
        raise CognitivePlanError("allowed_actions must not be empty")
    actions: list[PlanAction] = []
    for raw_action in raw_actions:
        if serialized:
            if not isinstance(raw_action, str):
                raise CognitivePlanError(
                    "serialized allowed_actions must contain only strings"
                )
            try:
                action = PlanAction(raw_action)
            except ValueError as error:
                raise CognitivePlanError(
                    f"unsupported plan action: {raw_action}"
                ) from error
        else:
            if not isinstance(raw_action, PlanAction):
                raise CognitivePlanError(
                    "allowed_actions must contain only PlanAction values"
                )
            action = raw_action
        actions.append(action)
    if len(set(actions)) != len(actions):
        raise CognitivePlanError("allowed_actions must not contain duplicates")
    action_set = frozenset(actions)
    return tuple(action for action in _ACTION_ORDER if action in action_set)


@dataclass(frozen=True, slots=True)
class CognitivePlan:
    """An immutable, bounded plan containing codes and hashes but no prose."""

    agent_id: str
    topic_id: str
    as_of_time: datetime
    valid_until: datetime
    trusted_cutoff: datetime
    allowed_actions: tuple[PlanAction, ...]
    confidence: float
    review_step: int
    input_sha256: str
    adapter_id: str
    adapter_authorization_sha256: str
    requested_use: RequestedUse
    rationale_code: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "agent_id", _identifier(self.agent_id, "agent_id"))
        object.__setattr__(self, "topic_id", _identifier(self.topic_id, "topic_id"))
        as_of_time = _aware_utc(self.as_of_time, "as_of_time")
        valid_until = _aware_utc(self.valid_until, "valid_until")
        trusted_cutoff = _aware_utc(self.trusted_cutoff, "trusted_cutoff")
        if as_of_time > trusted_cutoff:
            raise CognitivePlanTemporalError("as_of_time is after trusted_cutoff")
        if valid_until < as_of_time:
            raise CognitivePlanTemporalError("valid_until is before as_of_time")
        if valid_until > trusted_cutoff:
            raise CognitivePlanTemporalError("valid_until is after trusted_cutoff")
        object.__setattr__(self, "as_of_time", as_of_time)
        object.__setattr__(self, "valid_until", valid_until)
        object.__setattr__(self, "trusted_cutoff", trusted_cutoff)
        object.__setattr__(
            self,
            "allowed_actions",
            _allowed_actions(self.allowed_actions, serialized=False),
        )
        object.__setattr__(self, "confidence", _confidence(self.confidence))
        object.__setattr__(self, "review_step", _review_step(self.review_step))
        object.__setattr__(
            self,
            "input_sha256",
            _sha256(self.input_sha256, "input_sha256"),
        )
        object.__setattr__(
            self,
            "adapter_id",
            _identifier(self.adapter_id, "adapter_id"),
        )
        object.__setattr__(
            self,
            "adapter_authorization_sha256",
            _sha256(
                self.adapter_authorization_sha256,
                "adapter_authorization_sha256",
            ),
        )
        if not isinstance(self.requested_use, RequestedUse):
            raise CognitivePlanError("requested_use must be a RequestedUse")
        object.__setattr__(self, "rationale_code", _rationale_code(self.rationale_code))

    def permits(self, action: PlanAction) -> bool:
        """Return whether the strict slow plan permits a fast action."""
        if not isinstance(action, PlanAction):
            raise CognitivePlanError("action must be a PlanAction")
        return action in self.allowed_actions

    def is_valid(self, at_time: datetime) -> bool:
        """Return validity at a timezone-aware time without relaxing cutoff."""
        normalized_time = _aware_utc(at_time, "at_time")
        return self.as_of_time <= normalized_time <= self.valid_until

    def to_mapping(self) -> dict[str, object]:
        """Return the exact machine-readable planning schema."""
        return {
            "adapter_authorization_sha256": self.adapter_authorization_sha256,
            "adapter_id": self.adapter_id,
            "agent_id": self.agent_id,
            "allowed_actions": [action.value for action in self.allowed_actions],
            "as_of_time": self.as_of_time.isoformat(),
            "confidence": self.confidence,
            "input_sha256": self.input_sha256,
            "rationale_code": self.rationale_code,
            "requested_use": self.requested_use.value,
            "review_step": self.review_step,
            "topic_id": self.topic_id,
            "trusted_cutoff": self.trusted_cutoff.isoformat(),
            "valid_until": self.valid_until.isoformat(),
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> Self:
        """Parse an exact allowlist, rejecting missing and unknown fields."""
        if not isinstance(mapping, Mapping):
            raise CognitivePlanError("plan must be a mapping")
        if any(not isinstance(key, str) for key in mapping):
            raise CognitivePlanError("plan field names must be strings")
        actual = frozenset(mapping)
        missing = sorted(_PLAN_FIELDS - actual)
        unknown = sorted(actual - _PLAN_FIELDS)
        if missing:
            raise CognitivePlanError(f"plan is missing fields: {', '.join(missing)}")
        if unknown:
            raise CognitivePlanError(f"plan has unknown fields: {', '.join(unknown)}")
        return cls(
            agent_id=_identifier(mapping["agent_id"], "agent_id"),
            topic_id=_identifier(mapping["topic_id"], "topic_id"),
            as_of_time=_parse_datetime(mapping["as_of_time"], "as_of_time"),
            valid_until=_parse_datetime(mapping["valid_until"], "valid_until"),
            trusted_cutoff=_parse_datetime(mapping["trusted_cutoff"], "trusted_cutoff"),
            allowed_actions=_allowed_actions(
                mapping["allowed_actions"], serialized=True
            ),
            confidence=_confidence(mapping["confidence"]),
            review_step=_review_step(mapping["review_step"]),
            input_sha256=_sha256(mapping["input_sha256"], "input_sha256"),
            adapter_id=_identifier(mapping["adapter_id"], "adapter_id"),
            adapter_authorization_sha256=_sha256(
                mapping["adapter_authorization_sha256"],
                "adapter_authorization_sha256",
            ),
            requested_use=_requested_use(mapping["requested_use"]),
            rationale_code=_rationale_code(mapping["rationale_code"]),
        )


def _requested_use(value: object) -> RequestedUse:
    if not isinstance(value, str):
        raise CognitivePlanError("requested_use must be a string")
    try:
        return RequestedUse(value)
    except ValueError as error:
        raise CognitivePlanError("requested_use is unsupported") from error


__all__ = [
    "CognitivePlan",
    "CognitivePlanError",
    "CognitivePlanTemporalError",
    "PlanAction",
    "RequestedUse",
]
