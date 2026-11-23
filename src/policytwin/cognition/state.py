"""Immutable cognitive state and temporally guarded pure updates."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime

from policytwin.behavior import BehaviorMechanisms, BehaviorParameters


class CognitiveStateError(ValueError):
    """Raised when a cognitive state or update input is invalid."""


class CognitiveTemporalLeakageError(CognitiveStateError):
    """Raised when a signal was not legally available at the knowledge boundary."""


def _finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CognitiveStateError(f"{field_name} must be a finite real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise CognitiveStateError(f"{field_name} must be finite")
    return normalized


def _aware_utc(value: object, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise CognitiveStateError(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise CognitiveStateError(f"{field_name} must include an explicit timezone")
    try:
        return value.astimezone(UTC)
    except (OverflowError, ValueError) as error:
        raise CognitiveStateError(f"{field_name} is outside the UTC range") from error


def _optional_aware_utc(value: object, field_name: str) -> datetime | None:
    if value is None:
        return None
    return _aware_utc(value, field_name)


@dataclass(frozen=True, slots=True)
class CognitiveState:
    """A strategy-readable belief state with explicit knowledge-time lineage.

    ``signal_memory`` is the signal-derived component of ``belief``. Keeping it
    separate lets the updater decay signal memory without overwriting priors or
    other belief components supplied by a strategy.
    """

    belief: float = 0.0
    signal_memory: float = 0.0
    last_update_time: datetime | None = None
    observation_count: int = 0
    last_event_time: datetime | None = None
    last_available_time: datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "belief", _finite_number(self.belief, "belief"))
        object.__setattr__(
            self,
            "signal_memory",
            _finite_number(self.signal_memory, "signal_memory"),
        )
        if (
            isinstance(self.observation_count, bool)
            or not isinstance(self.observation_count, int)
            or self.observation_count < 0
        ):
            raise CognitiveStateError(
                "observation_count must be a non-negative integer"
            )

        last_update = _optional_aware_utc(self.last_update_time, "last_update_time")
        last_event = _optional_aware_utc(self.last_event_time, "last_event_time")
        last_available = _optional_aware_utc(
            self.last_available_time, "last_available_time"
        )
        object.__setattr__(self, "last_update_time", last_update)
        object.__setattr__(self, "last_event_time", last_event)
        object.__setattr__(self, "last_available_time", last_available)

        if (
            last_available is not None
            and last_update is not None
            and last_available > last_update
        ):
            raise CognitiveStateError(
                "last_available_time cannot be after last_update_time"
            )

    def to_mapping(self) -> dict[str, object]:
        """Return a JSON-compatible, timezone-explicit representation."""
        return {
            "belief": self.belief,
            "last_available_time": (
                None
                if self.last_available_time is None
                else self.last_available_time.isoformat()
            ),
            "last_event_time": (
                None
                if self.last_event_time is None
                else self.last_event_time.isoformat()
            ),
            "last_update_time": (
                None
                if self.last_update_time is None
                else self.last_update_time.isoformat()
            ),
            "observation_count": self.observation_count,
            "signal_memory": self.signal_memory,
        }

    def validate_available(
        self, *, as_of_time: datetime, trusted_cutoff: datetime
    ) -> None:
        """Reject non-zero or updated state without pre-cutoff lineage."""
        normalized_as_of = _aware_utc(as_of_time, "as_of_time")
        normalized_cutoff = _aware_utc(trusted_cutoff, "trusted_cutoff")
        if normalized_as_of > normalized_cutoff:
            raise CognitiveTemporalLeakageError("as_of_time is after trusted_cutoff")

        has_materialized_content = (
            self.belief != 0.0
            or self.signal_memory != 0.0
            or self.observation_count != 0
        )
        lineage = (
            self.last_update_time,
            self.last_event_time,
            self.last_available_time,
        )
        if has_materialized_content and any(value is None for value in lineage):
            raise CognitiveTemporalLeakageError(
                "non-zero cognitive state requires complete temporal lineage"
            )
        if (
            self.last_update_time is not None
            and self.last_update_time > normalized_as_of
        ):
            raise CognitiveTemporalLeakageError(
                "cognitive state's last_update_time is after as_of_time"
            )
        if self.last_event_time is not None and self.last_event_time > normalized_as_of:
            raise CognitiveTemporalLeakageError(
                "cognitive state contains an event after as_of_time"
            )
        if self.last_available_time is not None:
            if self.last_available_time > normalized_as_of:
                raise CognitiveTemporalLeakageError(
                    "cognitive state was unavailable at as_of_time"
                )
            if self.last_available_time > normalized_cutoff:
                raise CognitiveTemporalLeakageError(
                    "cognitive state was unavailable before trusted_cutoff"
                )


@dataclass(frozen=True, slots=True)
class CognitiveStateUpdater:
    """Pure update rule parameterized by immutable behavior configuration."""

    parameters: BehaviorParameters
    mechanisms: BehaviorMechanisms = field(default_factory=BehaviorMechanisms)

    def __post_init__(self) -> None:
        if not isinstance(self.parameters, BehaviorParameters):
            raise CognitiveStateError(
                "parameters must be an immutable BehaviorParameters instance"
            )
        if not isinstance(self.mechanisms, BehaviorMechanisms):
            raise CognitiveStateError(
                "mechanisms must be an immutable BehaviorMechanisms instance"
            )

    def update(
        self,
        state: CognitiveState,
        *,
        observed_signal: float,
        event_time: datetime,
        available_time: datetime,
        as_of_time: datetime,
        trusted_cutoff: datetime,
        elapsed_steps: int,
    ) -> CognitiveState:
        """Return a new state after a fail-closed temporal availability check."""
        if not isinstance(state, CognitiveState):
            raise CognitiveStateError("state must be a CognitiveState instance")
        signal = _finite_number(observed_signal, "observed_signal")
        if (
            isinstance(elapsed_steps, bool)
            or not isinstance(elapsed_steps, int)
            or elapsed_steps < 0
        ):
            raise CognitiveStateError("elapsed_steps must be a non-negative integer")

        normalized_event_time = _aware_utc(event_time, "event_time")
        normalized_available_time = _aware_utc(available_time, "available_time")
        normalized_as_of_time = _aware_utc(as_of_time, "as_of_time")
        normalized_cutoff = _aware_utc(trusted_cutoff, "trusted_cutoff")

        state.validate_available(
            as_of_time=normalized_as_of_time,
            trusted_cutoff=normalized_cutoff,
        )

        if normalized_as_of_time > normalized_cutoff:
            raise CognitiveTemporalLeakageError("as_of_time is after trusted_cutoff")
        if normalized_event_time > normalized_as_of_time:
            raise CognitiveTemporalLeakageError("event_time is after as_of_time")
        if normalized_available_time > normalized_cutoff:
            raise CognitiveTemporalLeakageError(
                "available_time is after trusted_cutoff"
            )
        if normalized_available_time > normalized_as_of_time:
            raise CognitiveTemporalLeakageError("available_time is after as_of_time")
        if (
            state.last_update_time is not None
            and normalized_as_of_time < state.last_update_time
        ):
            raise CognitiveTemporalLeakageError(
                "as_of_time is before the state's last_update_time"
            )

        retention = 1.0
        if self.mechanisms.memory_decay:
            retention = math.exp(-self.parameters.memory_decay * elapsed_steps)
        retained_memory = state.signal_memory * retention

        signal_weight = 1.0
        if self.mechanisms.recency_weight:
            signal_weight = self.parameters.recency_weight
        updated_memory = retained_memory + signal_weight * signal
        updated_belief = state.belief - state.signal_memory + updated_memory

        return CognitiveState(
            belief=updated_belief,
            signal_memory=updated_memory,
            last_update_time=normalized_as_of_time,
            observation_count=state.observation_count + 1,
            last_event_time=normalized_event_time,
            last_available_time=normalized_available_time,
        )

    def to_mapping(self) -> dict[str, object]:
        """Return the complete immutable update configuration."""
        return {
            "mechanisms": self.mechanisms.to_mapping(),
            "parameters": self.parameters.to_mapping(),
        }


__all__ = [
    "CognitiveState",
    "CognitiveStateError",
    "CognitiveStateUpdater",
    "CognitiveTemporalLeakageError",
]
