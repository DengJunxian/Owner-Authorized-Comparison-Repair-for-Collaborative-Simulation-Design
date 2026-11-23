"""Legacy mock advice values plus the strict LLM adapter protocol.

``LLMAdviceRequest`` and ``MockLLMAdapter`` remain compatibility fixtures for
fast-agent tests. They lack cutoff and lineage commitments and therefore must
not enter a production planner. New code uses ``LLMAdapter`` only through the
validated gateway.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from policytwin.agents.llm_gateway import LLMAdapter
from policytwin.counterfactual import EventKey


@dataclass(frozen=True, slots=True)
class LLMAdviceRequest:
    """Sanitized, public-only request accepted by an LLM adapter."""

    event_key: EventKey
    archetype: str
    public_signals: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.event_key, EventKey):
            raise ValueError("event_key must be an EventKey")
        if not isinstance(self.archetype, str) or not self.archetype:
            raise ValueError("archetype must be a non-empty string")
        if not isinstance(self.public_signals, tuple):
            raise ValueError("public_signals must be an immutable tuple")
        for name, value in self.public_signals:
            if not name or not math.isfinite(value):
                raise ValueError("public_signals must contain finite named values")


@dataclass(frozen=True, slots=True)
class LLMAdvice:
    """Bounded directional score returned by an adapter."""

    directional_score: float
    rationale_code: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.directional_score, bool)
            or not isinstance(self.directional_score, int | float)
            or not math.isfinite(self.directional_score)
            or not -1.0 <= self.directional_score <= 1.0
        ):
            raise ValueError("directional_score must be finite and in [-1, 1]")
        object.__setattr__(self, "directional_score", float(self.directional_score))
        if not isinstance(self.rationale_code, str) or not self.rationale_code:
            raise ValueError("rationale_code must be a non-empty string")


class LegacyLLMAdviceAdapter(Protocol):
    """Compatibility protocol that is ineligible for production planning."""

    def advise(self, request: LLMAdviceRequest) -> LLMAdvice:
        """Return advice without receiving private portfolio or cognitive state."""
        ...


@dataclass(frozen=True, slots=True)
class MockLLMAdapter:
    """Deterministic offline mock for interface and ablation tests."""

    response: LLMAdvice

    def advise(self, request: LLMAdviceRequest) -> LLMAdvice:
        """Return the configured response without I/O or mutable state."""
        if not isinstance(request, LLMAdviceRequest):
            raise TypeError("request must be an LLMAdviceRequest")
        return self.response


__all__ = [
    "LLMAdapter",
    "LLMAdvice",
    "LLMAdviceRequest",
    "LegacyLLMAdviceAdapter",
    "MockLLMAdapter",
]
