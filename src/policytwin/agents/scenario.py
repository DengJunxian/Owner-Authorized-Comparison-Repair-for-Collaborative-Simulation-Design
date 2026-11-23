"""Fast agents for receipt-bound model-generated future scenarios.

Observed ``PublicSignal`` and ``PublicMarketObservation`` deliberately remain
cutoff-bound.  The types in this module contain no ``available_time`` and can
only operate under a frozen :class:`ScenarioAgentAuthority` that exactly
matches an engine-issued post-cutoff receipt.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime

from policytwin.agents.models import (
    ActionIntent,
    ActionProbabilities,
    AgentArchetype,
    AgentDecisionError,
    AgentSchemaError,
    AgentTemporalLeakageError,
    ConstraintCheck,
    ConstraintCode,
    DecisionAction,
    OrderKind,
    OrderPolicyConfig,
    OwnPortfolioState,
    RiskLimits,
    SignalContribution,
    SignalKind,
)
from policytwin.behavior import BehaviorMechanisms, BehaviorParameters
from policytwin.cognition import ScenarioCognitiveSnapshot
from policytwin.counterfactual import EventKey, ScenarioEventReceipt
from policytwin.counterfactual.randomness import (
    AGENT_ACTION_CHOICE_DOMAIN,
    RNG_IMPLEMENTATION_COMMITMENT,
    RNG_IMPLEMENTATION_ID,
    RandomDrawReceipt,
    deterministic_int_with_receipt,
)

SCENARIO_AGENT_DECISION_SCHEMA_VERSION = "policytwin.scenario-agent-decision.v2"
_ACTION_DRAW_SCALE = 10**12
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ScenarioAgentError(ValueError):
    """Raised when generated-scenario agent evidence is invalid."""


def _identifier(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ScenarioAgentError(f"{field_name} must be a stable ASCII identifier")
    return value


def _sha256(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ScenarioAgentError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _aware(value: object, field_name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ScenarioAgentError(f"{field_name} must be timezone-aware")
    try:
        offset = value.utcoffset()
        normalized = value.astimezone(UTC)
    except (OverflowError, ValueError) as error:
        raise ScenarioAgentError(f"{field_name} has an invalid timezone") from error
    if offset is None:
        raise ScenarioAgentError(f"{field_name} must be timezone-aware")
    return normalized


def _finite(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ScenarioAgentError(f"{field_name} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ScenarioAgentError(f"{field_name} must be a finite number")
    return normalized


@dataclass(frozen=True, slots=True)
class ScenarioAgentAuthority:
    """Frozen scenario/generator commitments accepted by one fast-agent run."""

    scenario_id: str
    trusted_cutoff: datetime
    cutoff_policy_sha256: str
    source_lineage_sha256: str
    generator_spec_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "scenario_id", _identifier(self.scenario_id, "scenario_id")
        )
        object.__setattr__(
            self, "trusted_cutoff", _aware(self.trusted_cutoff, "trusted_cutoff")
        )
        for field_name in (
            "cutoff_policy_sha256",
            "source_lineage_sha256",
            "generator_spec_sha256",
        ):
            object.__setattr__(
                self, field_name, _sha256(getattr(self, field_name), field_name)
            )

    @classmethod
    def from_receipt(cls, receipt: ScenarioEventReceipt) -> ScenarioAgentAuthority:
        if not isinstance(receipt, ScenarioEventReceipt):
            raise ScenarioAgentError("authority requires an engine-issued receipt")
        return cls(
            scenario_id=receipt.scenario_id,
            trusted_cutoff=receipt.trusted_cutoff,
            cutoff_policy_sha256=receipt.generator_spec.cutoff_policy_sha256,
            source_lineage_sha256=receipt.generator_spec.source_lineage_sha256,
            generator_spec_sha256=receipt.generator_spec.sha256,
        )

    def validate_receipt(self, receipt: ScenarioEventReceipt) -> None:
        if not isinstance(receipt, ScenarioEventReceipt):
            raise ScenarioAgentError("scenario input lacks an engine-issued receipt")
        if receipt.scenario_time <= self.trusted_cutoff:
            raise ScenarioAgentError("scenario receipt is not strictly post-cutoff")
        if (
            receipt.scenario_id != self.scenario_id
            or receipt.trusted_cutoff != self.trusted_cutoff
            or receipt.generator_spec.cutoff_policy_sha256 != self.cutoff_policy_sha256
            or receipt.generator_spec.source_lineage_sha256
            != self.source_lineage_sha256
            or receipt.generator_spec.sha256 != self.generator_spec_sha256
        ):
            raise ScenarioAgentError(
                "scenario receipt differs from the frozen fast-agent authority"
            )

    def to_mapping(self) -> dict[str, object]:
        return {
            "cutoff_policy_sha256": self.cutoff_policy_sha256,
            "generator_spec_sha256": self.generator_spec_sha256,
            "scenario_id": self.scenario_id,
            "source_lineage_sha256": self.source_lineage_sha256,
            "trusted_cutoff": self.trusted_cutoff.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class ScenarioPublicSignal:
    """One generated signal; its time and provenance come from its container."""

    kind: SignalKind
    value: float
    signal_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, SignalKind):
            raise ScenarioAgentError("kind must be a SignalKind")
        object.__setattr__(self, "value", _finite(self.value, "value"))
        object.__setattr__(self, "signal_id", _identifier(self.signal_id, "signal_id"))

    def to_mapping(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "signal_id": self.signal_id,
            "value": self.value,
        }


@dataclass(frozen=True, slots=True)
class ScenarioVisibleOrderFlow:
    """Anonymous aggregate generated within one receipt-bound scenario step."""

    net_flow: float
    sample_size: int
    aggregation_id: str

    def __post_init__(self) -> None:
        flow = _finite(self.net_flow, "net_flow")
        if not -1.0 <= flow <= 1.0:
            raise ScenarioAgentError("net_flow must be in [-1, 1]")
        object.__setattr__(self, "net_flow", flow)
        if (
            isinstance(self.sample_size, bool)
            or not isinstance(self.sample_size, int)
            or self.sample_size < 0
        ):
            raise ScenarioAgentError("sample_size must be a non-negative integer")
        object.__setattr__(
            self,
            "aggregation_id",
            _identifier(self.aggregation_id, "aggregation_id"),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "aggregation_id": self.aggregation_id,
            "net_flow": self.net_flow,
            "sample_size": self.sample_size,
        }


@dataclass(frozen=True, slots=True)
class ScenarioPublicMarketObservation:
    """Public market projection for an exact generated future event receipt."""

    receipt: ScenarioEventReceipt
    mid_price: int
    best_bid: int | None
    best_ask: int | None
    signals: tuple[ScenarioPublicSignal, ...]
    visible_order_flow: ScenarioVisibleOrderFlow
    market_state_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.receipt, ScenarioEventReceipt):
            raise ScenarioAgentError(
                "scenario observation requires an engine-issued receipt"
            )
        if self.receipt.scenario_time <= self.receipt.trusted_cutoff:
            raise ScenarioAgentError("scenario observation is not post-cutoff")
        if (
            isinstance(self.mid_price, bool)
            or not isinstance(self.mid_price, int)
            or self.mid_price <= 0
        ):
            raise ScenarioAgentError("mid_price must be a positive integer")
        for field_name in ("best_bid", "best_ask"):
            value = getattr(self, field_name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise ScenarioAgentError(
                    f"{field_name} must be null or a positive integer"
                )
        if (
            self.best_bid is not None
            and self.best_ask is not None
            and self.best_bid >= self.best_ask
        ):
            raise ScenarioAgentError("best_bid must be below best_ask")
        if not isinstance(self.signals, tuple) or not all(
            isinstance(item, ScenarioPublicSignal) for item in self.signals
        ):
            raise ScenarioAgentError(
                "signals must be an immutable ScenarioPublicSignal tuple"
            )
        kinds = tuple(item.kind for item in self.signals)
        if len(set(kinds)) != len(kinds):
            raise ScenarioAgentError("scenario signal kinds must be unique")
        if not isinstance(self.visible_order_flow, ScenarioVisibleOrderFlow):
            raise ScenarioAgentError(
                "visible_order_flow must be ScenarioVisibleOrderFlow"
            )
        object.__setattr__(
            self,
            "market_state_sha256",
            _sha256(self.market_state_sha256, "market_state_sha256"),
        )

    @property
    def scenario_id(self) -> str:
        return self.receipt.scenario_id

    @property
    def as_of_time(self) -> datetime:
        return self.receipt.scenario_time

    @property
    def trusted_cutoff(self) -> datetime:
        return self.receipt.trusted_cutoff

    def validate_authority(self, authority: ScenarioAgentAuthority) -> None:
        if not isinstance(authority, ScenarioAgentAuthority):
            raise ScenarioAgentError("scenario authority has the wrong type")
        authority.validate_receipt(self.receipt)

    def signal(self, kind: SignalKind) -> ScenarioPublicSignal:
        for signal in self.signals:
            if signal.kind is kind:
                return signal
        raise AgentDecisionError(f"required scenario signal is missing: {kind.value}")

    def optional_signal_value(self, kind: SignalKind) -> float:
        for signal in self.signals:
            if signal.kind is kind:
                return signal.value
        return 0.0

    def to_mapping(self) -> dict[str, object]:
        return {
            "best_ask": self.best_ask,
            "best_bid": self.best_bid,
            "market_state_sha256": self.market_state_sha256,
            "mid_price": self.mid_price,
            "provenance": "model_generated_scenario",
            "scenario_event_receipt": self.receipt.to_mapping(),
            "signals": [
                item.to_mapping()
                for item in sorted(self.signals, key=lambda value: value.kind.value)
            ],
            "visible_order_flow": self.visible_order_flow.to_mapping(),
        }


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


def _max_affordable_quantity(*, cash: int, price: int, fee_bps: int, upper: int) -> int:
    lower = 0
    high = upper
    while lower < high:
        candidate = (lower + high + 1) // 2
        notional = price * candidate
        total = notional + notional * fee_bps // 10_000
        if total <= cash:
            lower = candidate
        else:
            high = candidate - 1
    return lower


@dataclass(frozen=True, slots=True)
class _ScenarioRiskAuthorization:
    buy_quantity: int
    sell_quantity: int
    buy_limit_price: int
    sell_limit_price: int
    checks: tuple[ConstraintCheck, ...]


def _floor_tick(price: int, tick_size: int) -> int:
    return max(tick_size, price - price % tick_size)


def _ceil_tick(price: int, tick_size: int) -> int:
    remainder = price % tick_size
    return price if remainder == 0 else price + tick_size - remainder


def _authorize(
    *,
    portfolio: OwnPortfolioState,
    observation: ScenarioPublicMarketObservation,
    policy: OrderPolicyConfig,
    limits: RiskLimits,
) -> _ScenarioRiskAuthorization:
    buy_base = observation.best_bid or observation.mid_price
    sell_base = observation.best_ask or observation.mid_price
    buy_limit = _floor_tick(
        max(policy.tick_size, buy_base - policy.limit_offset_ticks * policy.tick_size),
        policy.tick_size,
    )
    sell_limit = _ceil_tick(
        sell_base + policy.limit_offset_ticks * policy.tick_size,
        policy.tick_size,
    )
    requested = policy.base_order_quantity
    buy_quantity = requested
    sell_quantity = requested
    checks: list[ConstraintCheck] = []

    def cap(
        *,
        action: DecisionAction,
        current: int,
        maximum: int,
        code: ConstraintCode,
        detail: str,
        force_record: bool = False,
    ) -> int:
        permitted = min(current, max(0, maximum))
        if permitted < current or force_record:
            checks.append(
                ConstraintCheck(
                    code=code,
                    action=action,
                    requested_quantity=requested,
                    permitted_quantity=permitted,
                    detail=detail,
                )
            )
        return permitted

    for action in (DecisionAction.BUY, DecisionAction.SELL):
        current = buy_quantity if action is DecisionAction.BUY else sell_quantity
        current = cap(
            action=action,
            current=current,
            maximum=limits.max_order_quantity,
            code=ConstraintCode.ORDER_QUANTITY_LIMIT,
            detail=f"{action.value} quantity capped by max_order_quantity",
        )
        reference = buy_limit if action is DecisionAction.BUY else sell_limit
        current = cap(
            action=action,
            current=current,
            maximum=limits.max_order_notional // reference,
            code=ConstraintCode.ORDER_NOTIONAL_LIMIT,
            detail=f"{action.value} quantity capped by max_order_notional",
        )
        if action is DecisionAction.BUY:
            buy_quantity = current
        else:
            sell_quantity = current
    buy_quantity = cap(
        action=DecisionAction.BUY,
        current=buy_quantity,
        maximum=(
            limits.max_inventory - portfolio.inventory - portfolio.pending_buy_quantity
        ),
        code=ConstraintCode.POSITION_LIMIT,
        detail="buy quantity capped by inventory plus pending buys",
    )
    sell_quantity = cap(
        action=DecisionAction.SELL,
        current=sell_quantity,
        maximum=portfolio.available_inventory,
        code=ConstraintCode.INVENTORY_AVAILABLE,
        detail="sell quantity capped by unreserved inventory",
    )
    buy_quantity = cap(
        action=DecisionAction.BUY,
        current=buy_quantity,
        maximum=_max_affordable_quantity(
            cash=max(0, portfolio.available_cash - limits.cash_buffer),
            price=buy_limit,
            fee_bps=limits.fee_bps,
            upper=buy_quantity,
        ),
        code=ConstraintCode.CASH_BUDGET,
        detail="buy quantity capped by unreserved cash, buffer, and fees",
    )
    if portfolio.open_order_count >= limits.max_open_orders:
        buy_quantity = cap(
            action=DecisionAction.BUY,
            current=buy_quantity,
            maximum=0,
            code=ConstraintCode.OPEN_ORDER_LIMIT,
            detail="buy blocked by max_open_orders",
            force_record=True,
        )
        sell_quantity = cap(
            action=DecisionAction.SELL,
            current=sell_quantity,
            maximum=0,
            code=ConstraintCode.OPEN_ORDER_LIMIT,
            detail="sell blocked by max_open_orders",
            force_record=True,
        )
    return _ScenarioRiskAuthorization(
        buy_quantity=buy_quantity,
        sell_quantity=sell_quantity,
        buy_limit_price=buy_limit,
        sell_limit_price=sell_limit,
        checks=tuple(checks),
    )


def _softmax(
    *,
    buy_logit: float,
    hold_logit: float,
    sell_logit: float,
    temperature: float,
    authorization: _ScenarioRiskAuthorization,
) -> ActionProbabilities:
    logits = (
        buy_logit / temperature if authorization.buy_quantity else -math.inf,
        hold_logit / temperature,
        sell_logit / temperature if authorization.sell_quantity else -math.inf,
    )
    maximum = max(logits)
    weights = tuple(
        0.0 if item == -math.inf else math.exp(item - maximum) for item in logits
    )
    total = math.fsum(weights)
    return ActionProbabilities(
        buy=weights[0] / total,
        hold=weights[1] / total,
        sell=weights[2] / total,
    )


def _sample_action(
    event_key: EventKey, probabilities: ActionProbabilities
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


@dataclass(frozen=True, slots=True)
class ScenarioDecisionTrace:
    """Complete replayable fast decision for one generated future event."""

    event_key: EventKey
    agent_id: str
    archetype: AgentArchetype
    authority: ScenarioAgentAuthority
    available_market_state: ScenarioPublicMarketObservation
    available_own_state: OwnPortfolioState
    cognitive_snapshot: ScenarioCognitiveSnapshot
    parameters: BehaviorParameters
    mechanisms: BehaviorMechanisms
    policy_config: OrderPolicyConfig
    risk_limits: RiskLimits
    signals: tuple[SignalContribution, ...]
    action_probabilities: ActionProbabilities
    action_choice_draw: RandomDrawReceipt
    final_action: ActionIntent
    constraints: tuple[ConstraintCheck, ...]
    calibration_status: str = field(default="synthetic_uncalibrated", init=False)
    schema_version: str = SCENARIO_AGENT_DECISION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.event_key, EventKey):
            raise ScenarioAgentError("event_key must be an EventKey")
        object.__setattr__(self, "agent_id", _identifier(self.agent_id, "agent_id"))
        if not isinstance(self.authority, ScenarioAgentAuthority):
            raise ScenarioAgentError("authority must be ScenarioAgentAuthority")
        if not isinstance(self.available_market_state, ScenarioPublicMarketObservation):
            raise ScenarioAgentError(
                "available_market_state must be ScenarioPublicMarketObservation"
            )
        self.available_market_state.validate_authority(self.authority)
        if self.event_key.experiment_id != self.authority.scenario_id:
            raise ScenarioAgentError("event_key belongs to another scenario")
        if not isinstance(self.available_own_state, OwnPortfolioState):
            raise ScenarioAgentError("available_own_state must be OwnPortfolioState")
        if self.available_own_state.agent_id != self.agent_id:
            raise ScenarioAgentError("trace may contain only the agent's own state")
        if not isinstance(self.cognitive_snapshot, ScenarioCognitiveSnapshot):
            raise ScenarioAgentError(
                "cognitive_snapshot must be ScenarioCognitiveSnapshot"
            )
        _validate_snapshot(
            self.cognitive_snapshot,
            authority=self.authority,
            as_of_time=self.available_market_state.as_of_time,
        )
        if not isinstance(self.archetype, AgentArchetype):
            raise ScenarioAgentError("archetype must be an AgentArchetype")
        if not isinstance(self.parameters, BehaviorParameters):
            raise ScenarioAgentError("parameters must be BehaviorParameters")
        if not isinstance(self.mechanisms, BehaviorMechanisms):
            raise ScenarioAgentError("mechanisms must be BehaviorMechanisms")
        if not isinstance(self.policy_config, OrderPolicyConfig):
            raise ScenarioAgentError("policy_config must be OrderPolicyConfig")
        if not isinstance(self.risk_limits, RiskLimits):
            raise ScenarioAgentError("risk_limits must be RiskLimits")
        if not isinstance(self.signals, tuple) or not all(
            isinstance(item, SignalContribution) for item in self.signals
        ):
            raise ScenarioAgentError("signals must contain SignalContribution values")
        if not isinstance(self.action_probabilities, ActionProbabilities):
            raise ScenarioAgentError("action_probabilities has the wrong type")
        if type(self.action_choice_draw) is not RandomDrawReceipt:
            raise ScenarioAgentError(
                "action_choice_draw must be an exact RandomDrawReceipt"
            )
        draw = self.action_choice_draw
        try:
            recomputed_draw = RandomDrawReceipt.from_mapping(draw.to_mapping())
        except (RuntimeError, TypeError, ValueError) as error:
            raise ScenarioAgentError(
                "action_choice_draw failed independent deterministic replay"
            ) from error
        if recomputed_draw != draw:
            raise ScenarioAgentError(
                "action_choice_draw changed after independent deterministic replay"
            )
        if (
            draw.event_key != self.event_key
            or draw.domain != AGENT_ACTION_CHOICE_DOMAIN
            or draw.draw_index != 0
            or draw.lower != 0
            or draw.upper != _ACTION_DRAW_SCALE - 1
            or draw.rng_implementation_id != RNG_IMPLEMENTATION_ID
            or draw.rng_implementation_commitment != RNG_IMPLEMENTATION_COMMITMENT
        ):
            raise ScenarioAgentError(
                "action choice receipt differs from the fixed decision draw contract"
            )
        if not isinstance(self.final_action, ActionIntent):
            raise ScenarioAgentError("final_action has the wrong type")
        probability_draw = (draw.integer_value + 0.5) / _ACTION_DRAW_SCALE
        expected_action = (
            DecisionAction.BUY
            if probability_draw < self.action_probabilities.buy
            else (
                DecisionAction.HOLD
                if probability_draw
                < self.action_probabilities.buy + self.action_probabilities.hold
                else DecisionAction.SELL
            )
        )
        if self.final_action.action is not expected_action:
            raise ScenarioAgentError(
                "final action differs from the receipted probability draw"
            )
        if not isinstance(self.constraints, tuple) or not all(
            isinstance(item, ConstraintCheck) for item in self.constraints
        ):
            raise ScenarioAgentError("constraints has the wrong type")
        if self.schema_version != SCENARIO_AGENT_DECISION_SCHEMA_VERSION:
            raise ScenarioAgentError("unsupported scenario decision schema")

    @property
    def receipt(self) -> ScenarioEventReceipt:
        return self.available_market_state.receipt

    def to_mapping(self) -> dict[str, object]:
        return {
            "action_probabilities": self.action_probabilities.to_mapping(),
            "action_choice_draw": self.action_choice_draw.to_mapping(),
            "agent_id": self.agent_id,
            "archetype": self.archetype.value,
            "authority": self.authority.to_mapping(),
            "available_state": {
                "market": self.available_market_state.to_mapping(),
                "own": self.available_own_state.to_mapping(),
            },
            "calibration_status": self.calibration_status,
            "cognitive_snapshot": self.cognitive_snapshot.to_mapping(),
            "constraints": [item.to_mapping() for item in self.constraints],
            "event_key": {
                "experiment_id": self.event_key.experiment_id,
                "purpose": self.event_key.purpose,
                "replicate_id": self.event_key.replicate_id,
                "schema_version": self.event_key.schema_version,
                "seed": self.event_key.seed,
                "simulation_step": self.event_key.simulation_step,
                "value": self.event_key.value,
            },
            "final_action": self.final_action.to_mapping(),
            "mechanisms": self.mechanisms.to_mapping(),
            "parameters": self.parameters.to_mapping(),
            "policy_config": self.policy_config.to_mapping(),
            "provenance": "model_generated_scenario",
            "risk_limits": self.risk_limits.to_mapping(),
            "schema_version": self.schema_version,
            "signals": [item.to_mapping() for item in self.signals],
        }


def _validate_snapshot(
    snapshot: ScenarioCognitiveSnapshot,
    *,
    authority: ScenarioAgentAuthority,
    as_of_time: datetime,
) -> None:
    memory = snapshot.episodic_memory
    semantic = snapshot.semantic_state
    binding = (
        authority.scenario_id,
        authority.trusted_cutoff,
        authority.cutoff_policy_sha256,
        authority.source_lineage_sha256,
        authority.generator_spec_sha256,
    )
    if (
        memory.scenario_id,
        memory.trusted_cutoff,
        memory.cutoff_policy_sha256,
        memory.source_lineage_sha256,
        memory.generator_spec_sha256,
    ) != binding or (
        semantic.scenario_id,
        semantic.trusted_cutoff,
        semantic.cutoff_policy_sha256,
        semantic.source_lineage_sha256,
        semantic.generator_spec_sha256,
    ) != binding:
        raise ScenarioAgentError(
            "scenario cognition differs from the frozen fast-agent authority"
        )
    normalized_time = _aware(as_of_time, "as_of_time")
    if any(item.scenario_time > normalized_time for item in memory.entries) or any(
        item.last_update_time is not None and item.last_update_time > normalized_time
        for item in semantic.beliefs
    ):
        raise ScenarioAgentError("scenario cognition contains future generated state")


class ScenarioFastOrderPolicy:
    """Pure fast policy over generated state with hard budget/risk constraints."""

    def decide(
        self,
        *,
        agent_id: str,
        archetype: AgentArchetype,
        parameters: BehaviorParameters,
        mechanisms: BehaviorMechanisms,
        cognitive_snapshot: ScenarioCognitiveSnapshot,
        portfolio: OwnPortfolioState,
        observation: ScenarioPublicMarketObservation,
        authority: ScenarioAgentAuthority,
        event_key: EventKey,
        config: OrderPolicyConfig,
        risk_limits: RiskLimits,
    ) -> ScenarioDecisionTrace:
        if not isinstance(portfolio, OwnPortfolioState):
            raise AgentSchemaError("portfolio must be OwnPortfolioState")
        if portfolio.agent_id != agent_id:
            raise AgentDecisionError("policy may access only its own portfolio state")
        if not isinstance(observation, ScenarioPublicMarketObservation):
            raise AgentSchemaError(
                "observation must be ScenarioPublicMarketObservation"
            )
        if not isinstance(authority, ScenarioAgentAuthority):
            raise AgentSchemaError("authority must be ScenarioAgentAuthority")
        observation.validate_authority(authority)
        if not isinstance(cognitive_snapshot, ScenarioCognitiveSnapshot):
            raise AgentSchemaError(
                "cognitive_snapshot must be ScenarioCognitiveSnapshot"
            )
        _validate_snapshot(
            cognitive_snapshot,
            authority=authority,
            as_of_time=observation.as_of_time,
        )
        if event_key.experiment_id != authority.scenario_id:
            raise AgentTemporalLeakageError("event_key belongs to another scenario")
        if not isinstance(archetype, AgentArchetype):
            raise AgentSchemaError("archetype must be AgentArchetype")
        if not isinstance(parameters, BehaviorParameters):
            raise AgentSchemaError("parameters must be BehaviorParameters")
        if not isinstance(mechanisms, BehaviorMechanisms):
            raise AgentSchemaError("mechanisms must be BehaviorMechanisms")
        if not isinstance(config, OrderPolicyConfig):
            raise AgentSchemaError("config must be OrderPolicyConfig")
        if not isinstance(risk_limits, RiskLimits):
            raise AgentSchemaError("risk_limits must be RiskLimits")

        primary_kind, base_weight = _primary_signal(archetype)
        primary = observation.signal(primary_kind)
        attention = observation.optional_signal_value(SignalKind.ATTENTION)
        volatility = abs(observation.optional_signal_value(SignalKind.VOLATILITY))
        order_flow = observation.visible_order_flow.net_flow
        beliefs = cognitive_snapshot.semantic_state.beliefs
        memory_signal = (
            0.0
            if not beliefs
            else math.fsum(item.value for item in beliefs) / len(beliefs)
        )
        subjective = _trace_signal(
            name="subjective_signal",
            raw=primary.value,
            weight=base_weight,
            enabled=True,
        )
        cognitive_memory = _trace_signal(
            name="scenario_cognitive_memory",
            raw=memory_signal,
            weight=0.35,
            enabled=True,
        )
        overconfidence = _trace_signal(
            name="overconfidence",
            raw=primary.value,
            weight=base_weight * parameters.overconfidence,
            enabled=mechanisms.overconfidence,
        )
        herding = _trace_signal(
            name="herding",
            raw=order_flow,
            weight=parameters.herding,
            enabled=mechanisms.herding,
        )
        attention_effect = _trace_signal(
            name="attention_sensitivity",
            raw=primary.value * abs(attention),
            weight=parameters.attention_sensitivity,
            enabled=mechanisms.attention_sensitivity,
        )
        inventory_ratio = portfolio.inventory / risk_limits.max_inventory
        inventory = _trace_signal(
            name="inventory_aversion",
            raw=-inventory_ratio,
            weight=parameters.inventory_aversion
            * (2.0 if archetype is AgentArchetype.MARKET_MAKER else 0.5),
            enabled=mechanisms.inventory_aversion,
        )
        unrealized_return = 0.0
        if portfolio.inventory and portfolio.average_cost is not None:
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
        direction = math.fsum(
            (
                subjective.contribution,
                cognitive_memory.contribution,
                overconfidence.contribution,
                herding.contribution,
                attention_effect.contribution,
                inventory.contribution,
            )
        )
        turnover_bonus = 0.5 * abs(overconfidence.contribution) + (
            0.1 * parameters.overconfidence if mechanisms.overconfidence else 0.0
        )
        authorization = _authorize(
            portfolio=portfolio,
            observation=observation,
            policy=config,
            limits=risk_limits,
        )
        probabilities = _softmax(
            buy_logit=direction + turnover_bonus,
            hold_logit=0.5 + loss_aversion.contribution + risk_aversion.contribution,
            sell_logit=-direction + turnover_bonus + disposition.contribution,
            temperature=config.temperature,
            authorization=authorization,
        )
        action, action_choice_draw = _sample_action(event_key, probabilities)
        if action is DecisionAction.BUY:
            intent = ActionIntent(
                action=action,
                order_kind=OrderKind.LIMIT,
                quantity=authorization.buy_quantity,
                limit_price=authorization.buy_limit_price,
            )
        elif action is DecisionAction.SELL:
            intent = ActionIntent(
                action=action,
                order_kind=OrderKind.LIMIT,
                quantity=authorization.sell_quantity,
                limit_price=authorization.sell_limit_price,
            )
        else:
            intent = ActionIntent(
                action=DecisionAction.HOLD,
                order_kind=None,
                quantity=0,
                limit_price=None,
            )
        return ScenarioDecisionTrace(
            event_key=event_key,
            agent_id=agent_id,
            archetype=archetype,
            authority=authority,
            available_market_state=observation,
            available_own_state=portfolio,
            cognitive_snapshot=cognitive_snapshot,
            parameters=parameters,
            mechanisms=mechanisms,
            policy_config=config,
            risk_limits=risk_limits,
            signals=contributions,
            action_probabilities=probabilities,
            action_choice_draw=action_choice_draw,
            final_action=intent,
            constraints=authorization.checks,
        )


__all__ = [
    "SCENARIO_AGENT_DECISION_SCHEMA_VERSION",
    "ScenarioAgentAuthority",
    "ScenarioAgentError",
    "ScenarioDecisionTrace",
    "ScenarioFastOrderPolicy",
    "ScenarioPublicMarketObservation",
    "ScenarioPublicSignal",
    "ScenarioVisibleOrderFlow",
]
