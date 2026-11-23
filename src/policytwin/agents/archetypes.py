"""Uncalibrated continuous profiles for synthetic strategy archetypes."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field

from policytwin.agents.models import AgentArchetype
from policytwin.behavior import BehaviorMechanisms, BehaviorParameters


@dataclass(frozen=True, slots=True)
class SyntheticArchetypeProfile:
    """Numerical test profile; it is not an observed investor identity."""

    archetype: AgentArchetype
    parameters: BehaviorParameters
    mechanisms: BehaviorMechanisms
    calibration_status: str = dataclass_field(
        default="synthetic_uncalibrated",
        init=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.archetype, AgentArchetype):
            raise ValueError("archetype must be an AgentArchetype")
        if not isinstance(self.parameters, BehaviorParameters):
            raise ValueError("parameters must be BehaviorParameters")
        if not isinstance(self.mechanisms, BehaviorMechanisms):
            raise ValueError("mechanisms must be BehaviorMechanisms")


def synthetic_archetype_profile(
    archetype: AgentArchetype,
) -> SyntheticArchetypeProfile:
    """Return explicit, continuous defaults for one uncalibrated archetype."""
    if archetype is AgentArchetype.RETAIL_NOISE:
        values = BehaviorParameters(0.6, 1.8, 1.2, 1.0, 0.8, 0.9, 0.8, 0.3, 0.4)
    elif archetype is AgentArchetype.FUNDAMENTAL:
        values = BehaviorParameters(1.0, 1.5, 0.4, 0.3, 0.1, 0.2, 0.5, 0.1, 0.7)
    elif archetype is AgentArchetype.TREND:
        values = BehaviorParameters(0.8, 1.4, 0.5, 0.7, 0.4, 0.5, 0.9, 0.2, 0.6)
    elif archetype is AgentArchetype.INSTITUTIONAL_EXECUTION:
        values = BehaviorParameters(1.3, 1.2, 0.2, 0.2, 0.1, 0.1, 0.4, 0.1, 1.5)
    elif archetype is AgentArchetype.MARKET_MAKER:
        values = BehaviorParameters(1.5, 1.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.1, 2.0)
    elif archetype is AgentArchetype.ARBITRAGE:
        values = BehaviorParameters(0.7, 1.0, 0.1, 0.4, 0.0, 0.2, 0.7, 0.1, 1.0)
    else:
        raise ValueError(f"unsupported archetype: {archetype}")
    return SyntheticArchetypeProfile(
        archetype=archetype,
        parameters=values,
        mechanisms=BehaviorMechanisms(),
    )


__all__ = ["SyntheticArchetypeProfile", "synthetic_archetype_profile"]
