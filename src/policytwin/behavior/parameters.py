"""Immutable behavioral parameters and mechanism switches."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields


def _validate_parameter(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a finite real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    if normalized < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return normalized


@dataclass(frozen=True, slots=True)
class BehaviorParameters:
    """Continuous non-negative parameters for optional behavior mechanisms."""

    risk_aversion: float
    loss_aversion: float
    disposition_strength: float
    overconfidence: float
    herding: float
    attention_sensitivity: float
    recency_weight: float
    memory_decay: float
    inventory_aversion: float

    def __post_init__(self) -> None:
        for field in fields(self):
            value = _validate_parameter(field.name, getattr(self, field.name))
            object.__setattr__(self, field.name, value)
        if self.recency_weight > 1.0:
            raise ValueError("recency_weight must be between 0 and 1")

    def to_mapping(self) -> dict[str, float]:
        """Return a stable representation suitable for configuration artifacts."""
        return {
            "attention_sensitivity": self.attention_sensitivity,
            "disposition_strength": self.disposition_strength,
            "herding": self.herding,
            "inventory_aversion": self.inventory_aversion,
            "loss_aversion": self.loss_aversion,
            "memory_decay": self.memory_decay,
            "overconfidence": self.overconfidence,
            "recency_weight": self.recency_weight,
            "risk_aversion": self.risk_aversion,
        }


@dataclass(frozen=True, slots=True)
class BehaviorMechanisms:
    """Independent switches for every modeled behavioral mechanism.

    A disabled switch makes its corresponding parameter inert; parameter values
    remain in the immutable configuration so an ablation is fully traceable.
    """

    risk_aversion: bool = True
    loss_aversion: bool = True
    disposition_strength: bool = True
    overconfidence: bool = True
    herding: bool = True
    attention_sensitivity: bool = True
    recency_weight: bool = True
    memory_decay: bool = True
    inventory_aversion: bool = True

    def __post_init__(self) -> None:
        for field in fields(self):
            if not isinstance(getattr(self, field.name), bool):
                raise ValueError(f"{field.name} mechanism switch must be boolean")

    def to_mapping(self) -> dict[str, bool]:
        """Return all switches without combining or inferring mechanisms."""
        return {
            "attention_sensitivity": self.attention_sensitivity,
            "disposition_strength": self.disposition_strength,
            "herding": self.herding,
            "inventory_aversion": self.inventory_aversion,
            "loss_aversion": self.loss_aversion,
            "memory_decay": self.memory_decay,
            "overconfidence": self.overconfidence,
            "recency_weight": self.recency_weight,
            "risk_aversion": self.risk_aversion,
        }


__all__ = ["BehaviorMechanisms", "BehaviorParameters"]
