"""Privacy-preserving public social observations for simulated agents."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime


class SocialObservationError(ValueError):
    """Raised when an aggregate social observation is not safe to consume."""


def _aware_utc(value: datetime, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise SocialObservationError(f"{field} must be timezone-aware")
    try:
        offset = value.utcoffset()
    except (OverflowError, ValueError) as error:
        raise SocialObservationError(f"{field} has an invalid timezone") from error
    if offset is None:
        raise SocialObservationError(f"{field} must be timezone-aware")
    return value.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class VisibleOrderFlow:
    """An aggregate order-flow signal with no participant-level state.

    ``net_flow`` is normalized to ``[-1, 1]``. Positive values indicate more
    visible buy flow and negative values indicate more visible sell flow. The
    schema deliberately contains no agent identifiers, accounts, inventories,
    cognitive states, or private labels.
    """

    net_flow: float
    sample_size: int
    event_time: datetime
    available_time: datetime
    source: str
    aggregation_id: str

    def __post_init__(self) -> None:
        if isinstance(self.net_flow, bool) or not isinstance(
            self.net_flow, int | float
        ):
            raise SocialObservationError("net_flow must be a finite number")
        normalized_flow = float(self.net_flow)
        if not math.isfinite(normalized_flow) or not -1.0 <= normalized_flow <= 1.0:
            raise SocialObservationError("net_flow must be finite and in [-1, 1]")
        object.__setattr__(self, "net_flow", normalized_flow)

        if (
            isinstance(self.sample_size, bool)
            or not isinstance(self.sample_size, int)
            or self.sample_size < 0
        ):
            raise SocialObservationError("sample_size must be a non-negative integer")
        object.__setattr__(
            self, "event_time", _aware_utc(self.event_time, "event_time")
        )
        object.__setattr__(
            self,
            "available_time",
            _aware_utc(self.available_time, "available_time"),
        )
        if not isinstance(self.source, str) or not self.source.strip():
            raise SocialObservationError("source must be a non-empty string")
        if not isinstance(self.aggregation_id, str) or not self.aggregation_id.strip():
            raise SocialObservationError("aggregation_id must be a non-empty string")

    def validate_available(self, *, as_of_time: datetime, cutoff: datetime) -> None:
        """Fail closed if the aggregate was not available for this decision."""
        normalized_as_of = _aware_utc(as_of_time, "as_of_time")
        normalized_cutoff = _aware_utc(cutoff, "cutoff")
        if normalized_as_of > normalized_cutoff:
            raise SocialObservationError("as_of_time must not be after cutoff")
        if self.event_time > normalized_as_of:
            raise SocialObservationError("order-flow event_time is after as_of_time")
        if self.available_time > normalized_as_of:
            raise SocialObservationError("order flow was unavailable at decision time")
        if self.available_time > normalized_cutoff:
            raise SocialObservationError("order flow was unavailable before cutoff")

    def to_mapping(self) -> dict[str, object]:
        """Return a machine-readable, privacy-preserving representation."""
        return {
            "aggregation_id": self.aggregation_id,
            "available_time": self.available_time.isoformat(),
            "event_time": self.event_time.isoformat(),
            "net_flow": self.net_flow,
            "sample_size": self.sample_size,
            "source": self.source,
        }


__all__ = ["SocialObservationError", "VisibleOrderFlow"]
