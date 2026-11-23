"""Composition root for slow cognition and fast constrained order decisions."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime

from policytwin.agents.models import (
    AgentArchetype,
    AgentDecisionError,
    AgentSchemaError,
    AgentTemporalLeakageError,
    DecisionTrace,
    OrderPolicyConfig,
    OwnPortfolioState,
    PublicMarketObservation,
    RiskLimits,
    SignalKind,
)
from policytwin.agents.policies import FastOrderPolicy
from policytwin.behavior import BehaviorMechanisms, BehaviorParameters
from policytwin.cognition import CognitiveState, CognitiveStateUpdater
from policytwin.counterfactual import EventKey

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _trusted_utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise AgentSchemaError("trusted_cutoff must be timezone-aware")
    try:
        offset = value.utcoffset()
    except (OverflowError, ValueError) as error:
        raise AgentSchemaError("trusted_cutoff has an invalid timezone") from error
    if offset is None:
        raise AgentSchemaError("trusted_cutoff must be timezone-aware")
    return value.astimezone(UTC)


def _primary_signal_kind(archetype: AgentArchetype) -> SignalKind:
    if archetype is AgentArchetype.RETAIL_NOISE:
        return SignalKind.NOISE
    if archetype is AgentArchetype.FUNDAMENTAL:
        return SignalKind.FUNDAMENTAL
    if archetype is AgentArchetype.TREND:
        return SignalKind.TREND
    if archetype is AgentArchetype.INSTITUTIONAL_EXECUTION:
        return SignalKind.EXECUTION_URGENCY
    if archetype is AgentArchetype.MARKET_MAKER:
        return SignalKind.FUNDAMENTAL
    if archetype is AgentArchetype.ARBITRAGE:
        return SignalKind.ARBITRAGE
    raise AssertionError(f"unhandled archetype: {archetype}")


@dataclass(frozen=True, slots=True)
class StructuredAgent:
    """One synthetic agent with explicit cognition and policy boundaries.

    ``update_cognitive_state`` is called only on the configured slow schedule.
    ``decide`` can be called at every fast event and cannot mutate cognition.
    """

    agent_id: str
    archetype: AgentArchetype
    parameters: BehaviorParameters
    mechanisms: BehaviorMechanisms
    trusted_cutoff: datetime
    policy_config: OrderPolicyConfig
    risk_limits: RiskLimits
    cognitive_update_interval_steps: int = 5
    _fast_policy: FastOrderPolicy = field(
        default_factory=FastOrderPolicy,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if (
            not isinstance(self.agent_id, str)
            or _IDENTIFIER_PATTERN.fullmatch(self.agent_id) is None
        ):
            raise AgentSchemaError("agent_id must be a stable ASCII identifier")
        if not isinstance(self.archetype, AgentArchetype):
            raise AgentSchemaError("archetype must be an AgentArchetype")
        if not isinstance(self.parameters, BehaviorParameters):
            raise AgentSchemaError("parameters must be BehaviorParameters")
        if not isinstance(self.mechanisms, BehaviorMechanisms):
            raise AgentSchemaError("mechanisms must be BehaviorMechanisms")
        if not isinstance(self.policy_config, OrderPolicyConfig):
            raise AgentSchemaError("policy_config must be an OrderPolicyConfig")
        if not isinstance(self.risk_limits, RiskLimits):
            raise AgentSchemaError("risk_limits must be RiskLimits")
        object.__setattr__(self, "trusted_cutoff", _trusted_utc(self.trusted_cutoff))
        if (
            isinstance(self.cognitive_update_interval_steps, bool)
            or not isinstance(self.cognitive_update_interval_steps, int)
            or self.cognitive_update_interval_steps <= 0
        ):
            raise AgentSchemaError(
                "cognitive_update_interval_steps must be a positive integer"
            )
        if not isinstance(self._fast_policy, FastOrderPolicy):
            raise AgentSchemaError("_fast_policy must be a FastOrderPolicy")

    def _validate_context(
        self,
        *,
        portfolio: OwnPortfolioState,
        observation: PublicMarketObservation,
    ) -> None:
        if not isinstance(portfolio, OwnPortfolioState):
            raise AgentSchemaError("portfolio must be OwnPortfolioState")
        if not isinstance(observation, PublicMarketObservation):
            raise AgentSchemaError("observation must be PublicMarketObservation")
        if portfolio.agent_id != self.agent_id:
            raise AgentDecisionError("agent may access only its own portfolio state")
        if observation.cutoff != self.trusted_cutoff:
            raise AgentTemporalLeakageError(
                "observation cutoff does not match the agent's trusted cutoff"
            )
        if observation.as_of_time > self.trusted_cutoff:
            raise AgentTemporalLeakageError("decision time is after trusted cutoff")

    def cognition_due(self, *, elapsed_steps: int) -> bool:
        """Return whether the explicit slow update schedule is due."""
        if (
            isinstance(elapsed_steps, bool)
            or not isinstance(elapsed_steps, int)
            or elapsed_steps < 0
        ):
            raise AgentSchemaError("elapsed_steps must be a non-negative integer")
        return elapsed_steps >= self.cognitive_update_interval_steps

    def update_cognitive_state(
        self,
        state: CognitiveState,
        *,
        portfolio: OwnPortfolioState,
        observation: PublicMarketObservation,
        elapsed_steps: int,
    ) -> CognitiveState:
        """Return a new slow state, rejecting off-schedule implicit updates."""
        self._validate_context(portfolio=portfolio, observation=observation)
        if not isinstance(state, CognitiveState):
            raise AgentSchemaError("state must be CognitiveState")
        if not self.cognition_due(elapsed_steps=elapsed_steps):
            raise AgentDecisionError("cognitive update requested before slow interval")
        signal = observation.signal(_primary_signal_kind(self.archetype))
        updater = CognitiveStateUpdater(
            parameters=self.parameters,
            mechanisms=self.mechanisms,
        )
        return updater.update(
            state,
            observed_signal=signal.value,
            event_time=signal.event_time,
            available_time=signal.available_time,
            as_of_time=observation.as_of_time,
            trusted_cutoff=self.trusted_cutoff,
            elapsed_steps=elapsed_steps,
        )

    def decide(
        self,
        *,
        state: CognitiveState,
        portfolio: OwnPortfolioState,
        observation: PublicMarketObservation,
        event_key: EventKey,
    ) -> DecisionTrace:
        """Run only the fast policy against an already materialized slow state."""
        self._validate_context(portfolio=portfolio, observation=observation)
        if not isinstance(state, CognitiveState):
            raise AgentSchemaError("state must be CognitiveState")
        if not isinstance(event_key, EventKey):
            raise AgentSchemaError("event_key must be EventKey")
        if (
            state.last_update_time is not None
            and state.last_update_time > observation.as_of_time
        ):
            raise AgentTemporalLeakageError(
                "cognitive state was updated after the decision time"
            )
        return self._fast_policy.decide(
            agent_id=self.agent_id,
            archetype=self.archetype,
            parameters=self.parameters,
            mechanisms=self.mechanisms,
            cognitive_state=state,
            portfolio=portfolio,
            observation=observation,
            trusted_cutoff=self.trusted_cutoff,
            event_key=event_key,
            config=self.policy_config,
            risk_limits=self.risk_limits,
        )


__all__ = ["StructuredAgent"]
