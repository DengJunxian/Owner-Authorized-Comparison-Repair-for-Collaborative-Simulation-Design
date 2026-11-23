"""Fast, stateless order policy for the six synthetic archetypes."""

from __future__ import annotations

import math
from datetime import UTC, datetime

from policytwin.agents.models import (
    ActionIntent,
    ActionProbabilities,
    AgentArchetype,
    AgentDecisionError,
    AgentSchemaError,
    AgentTemporalLeakageError,
    DecisionAction,
    DecisionTrace,
    OrderKind,
    OrderPolicyConfig,
    OwnPortfolioState,
    PublicMarketObservation,
    RiskLimits,
    SignalContribution,
    SignalKind,
)
from policytwin.agents.risk import RiskAuthorization, RiskGateway
from policytwin.behavior import BehaviorMechanisms, BehaviorParameters
from policytwin.cognition import CognitiveState
from policytwin.counterfactual import EventKey
from policytwin.counterfactual.randomness import (
    AGENT_ACTION_CHOICE_DOMAIN,
    RandomDrawReceipt,
    deterministic_int_with_receipt,
)

_ACTION_DRAW_SCALE = 10**12


def _primary_signal(archetype: AgentArchetype) -> tuple[SignalKind, float]:
    if archetype is AgentArchetype.RETAIL_NOISE:
        return SignalKind.NOISE, 0.8
    if archetype is AgentArchetype.FUNDAMENTAL:
        return SignalKind.FUNDAMENTAL, 1.0
    if archetype is AgentArchetype.TREND:
        return SignalKind.TREND, 1.0
    if archetype is AgentArchetype.INSTITUTIONAL_EXECUTION:
        return SignalKind.EXECUTION_URGENCY, 1.2
    if archetype is AgentArchetype.MARKET_MAKER:
        return SignalKind.FUNDAMENTAL, 0.2
    if archetype is AgentArchetype.ARBITRAGE:
        return SignalKind.ARBITRAGE, 1.5
    raise AssertionError(f"unhandled archetype: {archetype}")


def _order_kind(archetype: AgentArchetype) -> OrderKind:
    # The current public observation contains only top-of-book prices, not a
    # committed depth curve. A limit price is therefore required to make the
    # cash/notional authorization a hard bound for every built-in archetype.
    # ``archetype`` remains explicit so a future versioned policy can choose a
    # different safe order contract without silently changing this estimand.
    _ = archetype
    return OrderKind.LIMIT


def _softmax(
    *,
    buy_logit: float,
    hold_logit: float,
    sell_logit: float,
    temperature: float,
    authorization: RiskAuthorization,
) -> ActionProbabilities:
    logits = [
        buy_logit / temperature if authorization.buy_quantity > 0 else -math.inf,
        hold_logit / temperature,
        sell_logit / temperature if authorization.sell_quantity > 0 else -math.inf,
    ]
    maximum = max(logits)
    weights = [
        0.0 if value == -math.inf else math.exp(value - maximum) for value in logits
    ]
    total = sum(weights)
    return ActionProbabilities(
        buy=weights[0] / total,
        hold=weights[1] / total,
        sell=weights[2] / total,
    )


def _sample_action(
    *, event_key: EventKey, probabilities: ActionProbabilities
) -> tuple[DecisionAction, RandomDrawReceipt]:
    integer_draw, receipt = deterministic_int_with_receipt(
        event_key,
        domain=AGENT_ACTION_CHOICE_DOMAIN,
        draw_index=0,
        lower=0,
        upper=_ACTION_DRAW_SCALE - 1,
    )
    draw = (integer_draw + 0.5) / _ACTION_DRAW_SCALE
    if draw < probabilities.buy:
        return DecisionAction.BUY, receipt
    if draw < probabilities.buy + probabilities.hold:
        return DecisionAction.HOLD, receipt
    return DecisionAction.SELL, receipt


def _trace_signal(
    *, name: str, raw: float, weight: float, enabled: bool
) -> SignalContribution:
    effective_weight = weight if enabled else 0.0
    return SignalContribution(
        name=name,
        raw_value=raw,
        weight=effective_weight,
        contribution=raw * effective_weight,
        enabled=enabled,
    )


class FastOrderPolicy:
    """Pure score, risk-gate, and action-sampling policy.

    The policy consumes a previously updated slow ``CognitiveState`` but never
    mutates it. All stochastic choice is addressed by a semantic ``EventKey``.
    """

    def __init__(self, risk_gateway: RiskGateway | None = None) -> None:
        self._risk_gateway = risk_gateway or RiskGateway()

    def decide(
        self,
        *,
        agent_id: str,
        archetype: AgentArchetype,
        parameters: BehaviorParameters,
        mechanisms: BehaviorMechanisms,
        cognitive_state: CognitiveState,
        portfolio: OwnPortfolioState,
        observation: PublicMarketObservation,
        trusted_cutoff: datetime,
        event_key: EventKey,
        config: OrderPolicyConfig,
        risk_limits: RiskLimits,
    ) -> DecisionTrace:
        """Produce one complete, constrained, reproducible decision trace."""
        if not isinstance(portfolio, OwnPortfolioState):
            raise AgentSchemaError("portfolio must be OwnPortfolioState")
        if not isinstance(agent_id, str) or portfolio.agent_id != agent_id:
            raise AgentDecisionError("policy may access only its own portfolio state")
        if not isinstance(observation, PublicMarketObservation):
            raise AgentSchemaError("observation must be PublicMarketObservation")
        if not isinstance(cognitive_state, CognitiveState):
            raise AgentSchemaError("cognitive_state must be CognitiveState")
        if not isinstance(archetype, AgentArchetype):
            raise AgentSchemaError("archetype must be AgentArchetype")
        if not isinstance(parameters, BehaviorParameters):
            raise AgentSchemaError("parameters must be BehaviorParameters")
        if not isinstance(mechanisms, BehaviorMechanisms):
            raise AgentSchemaError("mechanisms must be BehaviorMechanisms")
        if not isinstance(event_key, EventKey):
            raise AgentSchemaError("event_key must be EventKey")
        if not isinstance(config, OrderPolicyConfig):
            raise AgentSchemaError("config must be OrderPolicyConfig")
        if not isinstance(risk_limits, RiskLimits):
            raise AgentSchemaError("risk_limits must be RiskLimits")
        if not isinstance(trusted_cutoff, datetime) or trusted_cutoff.tzinfo is None:
            raise AgentTemporalLeakageError("trusted_cutoff must be timezone-aware")
        try:
            offset = trusted_cutoff.utcoffset()
        except (OverflowError, ValueError) as error:
            raise AgentTemporalLeakageError(
                "trusted_cutoff has an invalid timezone"
            ) from error
        if offset is None:
            raise AgentTemporalLeakageError("trusted_cutoff must be timezone-aware")
        normalized_cutoff = trusted_cutoff.astimezone(UTC)
        if observation.cutoff != normalized_cutoff:
            raise AgentTemporalLeakageError(
                "observation cutoff does not match the trusted cutoff"
            )
        try:
            cognitive_state.validate_available(
                as_of_time=observation.as_of_time,
                trusted_cutoff=normalized_cutoff,
            )
        except ValueError as error:
            raise AgentTemporalLeakageError(str(error)) from error
        primary_kind, base_weight = _primary_signal(archetype)
        primary = observation.signal(primary_kind)
        attention = observation.optional_signal_value(SignalKind.ATTENTION)
        volatility = abs(observation.optional_signal_value(SignalKind.VOLATILITY))
        order_flow = observation.visible_order_flow.net_flow

        subjective = _trace_signal(
            name="subjective_signal",
            raw=primary.value,
            weight=base_weight,
            enabled=True,
        )
        overconfidence = _trace_signal(
            name="overconfidence",
            raw=primary.value,
            weight=base_weight * parameters.overconfidence,
            enabled=mechanisms.overconfidence,
        )
        attention_effect = _trace_signal(
            name="attention_sensitivity",
            raw=primary.value * abs(attention),
            weight=parameters.attention_sensitivity,
            enabled=mechanisms.attention_sensitivity,
        )
        herding = _trace_signal(
            name="herding",
            raw=order_flow,
            weight=parameters.herding,
            enabled=mechanisms.herding,
        )
        cognitive_memory = _trace_signal(
            name="cognitive_memory",
            raw=cognitive_state.signal_memory,
            weight=0.35,
            enabled=True,
        )

        inventory_ratio = portfolio.inventory / risk_limits.max_inventory
        inventory_multiplier = 2.0 if archetype is AgentArchetype.MARKET_MAKER else 0.5
        inventory = _trace_signal(
            name="inventory_aversion",
            raw=-inventory_ratio,
            weight=parameters.inventory_aversion * inventory_multiplier,
            enabled=mechanisms.inventory_aversion,
        )

        unrealized_return = 0.0
        if portfolio.inventory > 0 and portfolio.average_cost is not None:
            unrealized_return = (
                observation.mid_price - portfolio.average_cost
            ) / portfolio.average_cost
        disposition = _trace_signal(
            name="disposition_strength",
            raw=max(0.0, unrealized_return),
            weight=parameters.disposition_strength * 4.0,
            enabled=mechanisms.disposition_strength,
        )
        loss_aversion = _trace_signal(
            name="loss_aversion",
            raw=max(0.0, -unrealized_return),
            weight=parameters.loss_aversion,
            enabled=mechanisms.loss_aversion,
        )
        risk_aversion = _trace_signal(
            name="risk_aversion",
            raw=volatility,
            weight=parameters.risk_aversion,
            enabled=mechanisms.risk_aversion,
        )

        contributions = (
            subjective,
            cognitive_memory,
            overconfidence,
            herding,
            attention_effect,
            disposition,
            loss_aversion,
            risk_aversion,
            inventory,
        )
        direction = (
            subjective.contribution
            + cognitive_memory.contribution
            + overconfidence.contribution
            + herding.contribution
            + attention_effect.contribution
            + inventory.contribution
        )
        turnover_bonus = 0.5 * abs(overconfidence.contribution) + (
            0.1 * parameters.overconfidence if mechanisms.overconfidence else 0.0
        )
        buy_logit = direction + turnover_bonus
        sell_logit = -direction + turnover_bonus + disposition.contribution
        hold_logit = 0.5 + loss_aversion.contribution + risk_aversion.contribution

        kind = _order_kind(archetype)
        authorization = self._risk_gateway.authorize(
            portfolio=portfolio,
            observation=observation,
            policy=config,
            limits=risk_limits,
            order_kind=kind,
        )
        probabilities = _softmax(
            buy_logit=buy_logit,
            hold_logit=hold_logit,
            sell_logit=sell_logit,
            temperature=config.temperature,
            authorization=authorization,
        )
        action, action_choice_draw = _sample_action(
            event_key=event_key,
            probabilities=probabilities,
        )
        if action is DecisionAction.BUY:
            final = ActionIntent(
                action=action,
                order_kind=kind,
                quantity=authorization.buy_quantity,
                limit_price=(
                    authorization.buy_limit_price if kind is OrderKind.LIMIT else None
                ),
            )
        elif action is DecisionAction.SELL:
            final = ActionIntent(
                action=action,
                order_kind=kind,
                quantity=authorization.sell_quantity,
                limit_price=(
                    authorization.sell_limit_price if kind is OrderKind.LIMIT else None
                ),
            )
        else:
            final = ActionIntent(
                action=DecisionAction.HOLD,
                order_kind=None,
                quantity=0,
                limit_price=None,
            )

        return DecisionTrace(
            event_key=event_key,
            agent_id=agent_id,
            archetype=archetype,
            available_market_state=observation,
            available_own_state=portfolio,
            cognitive_state=cognitive_state,
            parameters=parameters,
            mechanisms=mechanisms,
            policy_config=config,
            risk_limits=risk_limits,
            trusted_cutoff=normalized_cutoff,
            signals=contributions,
            action_probabilities=probabilities,
            action_choice_draw=action_choice_draw,
            final_action=final,
            constraints=authorization.checks,
        )


__all__ = ["FastOrderPolicy"]
