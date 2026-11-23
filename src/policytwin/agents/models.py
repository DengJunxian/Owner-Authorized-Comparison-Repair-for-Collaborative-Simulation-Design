"""Immutable schemas for auditable heterogeneous-agent decisions."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import UTC, datetime
from enum import StrEnum

from policytwin.behavior import BehaviorMechanisms, BehaviorParameters
from policytwin.cognition import CognitiveState
from policytwin.counterfactual import EventKey
from policytwin.counterfactual.randomness import (
    AGENT_ACTION_CHOICE_DOMAIN,
    RNG_IMPLEMENTATION_COMMITMENT,
    RNG_IMPLEMENTATION_ID,
    RandomDrawReceipt,
)
from policytwin.social import SocialObservationError, VisibleOrderFlow

AGENT_DECISION_SCHEMA_VERSION = "policytwin.agent-decision.v2"
_ACTION_DRAW_SCALE = 10**12
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class AgentError(RuntimeError):
    """Base class for structured-agent failures."""


class AgentSchemaError(ValueError):
    """Raised when an agent input or output schema is invalid."""


class AgentTemporalLeakageError(AgentError):
    """Raised when an observation was unavailable at the trusted cutoff."""


class AgentDecisionError(AgentError):
    """Raised when a decision cannot be produced without violating a contract."""


class AgentArchetype(StrEnum):
    """Synthetic strategy archetypes, not observed investor identities."""

    RETAIL_NOISE = "retail_noise"
    FUNDAMENTAL = "fundamental"
    TREND = "trend"
    INSTITUTIONAL_EXECUTION = "institutional_execution"
    MARKET_MAKER = "market_maker"
    ARBITRAGE = "arbitrage"


class SignalKind(StrEnum):
    """Explicit public signal channels understood by the synthetic policies."""

    NOISE = "noise"
    FUNDAMENTAL = "fundamental"
    TREND = "trend"
    EXECUTION_URGENCY = "execution_urgency"
    ATTENTION = "attention"
    VOLATILITY = "volatility"
    ARBITRAGE = "arbitrage"


class DecisionAction(StrEnum):
    """Directional decision before conversion to a market command."""

    HOLD = "hold"
    BUY = "buy"
    SELL = "sell"


class OrderKind(StrEnum):
    """Order instruction used when the directional action is not hold."""

    LIMIT = "limit"
    MARKET = "market"


class ConstraintCode(StrEnum):
    """Stable codes for hard risk-gateway interventions."""

    CASH_BUDGET = "cash_budget"
    INVENTORY_AVAILABLE = "inventory_available"
    POSITION_LIMIT = "position_limit"
    ORDER_QUANTITY_LIMIT = "order_quantity_limit"
    ORDER_NOTIONAL_LIMIT = "order_notional_limit"
    OPEN_ORDER_LIMIT = "open_order_limit"
    MARKET_ORDER_DISABLED = "market_order_disabled"


def _aware_utc(value: datetime, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise AgentSchemaError(f"{field} must be timezone-aware")
    try:
        offset = value.utcoffset()
    except (OverflowError, ValueError) as error:
        raise AgentSchemaError(f"{field} has an invalid timezone") from error
    if offset is None:
        raise AgentSchemaError(f"{field} must be timezone-aware")
    return value.astimezone(UTC)


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise AgentSchemaError(f"{field} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise AgentSchemaError(f"{field} must be a finite number")
    return normalized


def _non_negative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AgentSchemaError(f"{field} must be a non-negative integer")
    return value


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise AgentSchemaError(f"{field} must be a stable ASCII identifier")
    return value


@dataclass(frozen=True, slots=True)
class PublicSignal:
    """One public signal with explicit temporal availability and provenance."""

    kind: SignalKind
    value: float
    event_time: datetime
    available_time: datetime
    source: str
    signal_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, SignalKind):
            raise AgentSchemaError("kind must be a SignalKind")
        object.__setattr__(self, "value", _finite(self.value, "value"))
        object.__setattr__(
            self, "event_time", _aware_utc(self.event_time, "event_time")
        )
        object.__setattr__(
            self,
            "available_time",
            _aware_utc(self.available_time, "available_time"),
        )
        if not isinstance(self.source, str) or not self.source.strip():
            raise AgentSchemaError("source must be a non-empty string")
        _identifier(self.signal_id, "signal_id")

    def validate_available(self, *, as_of_time: datetime, cutoff: datetime) -> None:
        """Fail closed unless this signal was observable at decision time."""
        normalized_as_of = _aware_utc(as_of_time, "as_of_time")
        normalized_cutoff = _aware_utc(cutoff, "cutoff")
        if self.event_time > normalized_as_of:
            raise AgentTemporalLeakageError(
                f"signal {self.signal_id} event_time is after decision time"
            )
        if self.available_time > normalized_as_of:
            raise AgentTemporalLeakageError(
                f"signal {self.signal_id} was unavailable at decision time"
            )
        if self.available_time > normalized_cutoff:
            raise AgentTemporalLeakageError(
                f"signal {self.signal_id} was unavailable before cutoff"
            )

    def to_mapping(self) -> dict[str, object]:
        """Return a machine-readable representation."""
        return {
            "available_time": self.available_time.isoformat(),
            "event_time": self.event_time.isoformat(),
            "kind": self.kind.value,
            "signal_id": self.signal_id,
            "source": self.source,
            "value": self.value,
        }


@dataclass(frozen=True, slots=True)
class PublicMarketObservation:
    """Sanitized public market state available to one agent.

    This is intentionally not constructed from or aliased to ``MarketSnapshot``:
    the latter contains every participant's account and order ownership.
    """

    as_of_time: datetime
    cutoff: datetime
    mid_price: int
    best_bid: int | None
    best_ask: int | None
    signals: tuple[PublicSignal, ...]
    visible_order_flow: VisibleOrderFlow

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "as_of_time", _aware_utc(self.as_of_time, "as_of_time")
        )
        object.__setattr__(self, "cutoff", _aware_utc(self.cutoff, "cutoff"))
        if self.as_of_time > self.cutoff:
            raise AgentTemporalLeakageError("decision time must not be after cutoff")
        if (
            isinstance(self.mid_price, bool)
            or not isinstance(self.mid_price, int)
            or self.mid_price <= 0
        ):
            raise AgentSchemaError("mid_price must be a positive integer")
        for field in ("best_bid", "best_ask"):
            value = getattr(self, field)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise AgentSchemaError(f"{field} must be null or a positive integer")
        if (
            self.best_bid is not None
            and self.best_ask is not None
            and self.best_bid >= self.best_ask
        ):
            raise AgentSchemaError("best_bid must be below best_ask")
        if not isinstance(self.signals, tuple):
            raise AgentSchemaError("signals must be an immutable tuple")
        kinds: set[SignalKind] = set()
        for signal in self.signals:
            if not isinstance(signal, PublicSignal):
                raise AgentSchemaError("signals must contain only PublicSignal values")
            if signal.kind in kinds:
                raise AgentSchemaError(
                    f"duplicate public signal kind: {signal.kind.value}"
                )
            kinds.add(signal.kind)
            signal.validate_available(as_of_time=self.as_of_time, cutoff=self.cutoff)
        if not isinstance(self.visible_order_flow, VisibleOrderFlow):
            raise AgentSchemaError("visible_order_flow must be a VisibleOrderFlow")
        try:
            self.visible_order_flow.validate_available(
                as_of_time=self.as_of_time,
                cutoff=self.cutoff,
            )
        except SocialObservationError as error:
            raise AgentTemporalLeakageError(str(error)) from error

    def signal(self, kind: SignalKind) -> PublicSignal:
        """Return a required public signal or fail rather than impute silently."""
        for signal in self.signals:
            if signal.kind is kind:
                return signal
        raise AgentDecisionError(f"required public signal is missing: {kind.value}")

    def optional_signal_value(self, kind: SignalKind) -> float:
        """Return an explicitly optional channel, with zero as its neutral value."""
        for signal in self.signals:
            if signal.kind is kind:
                return signal.value
        return 0.0

    def to_mapping(self) -> dict[str, object]:
        """Return the exact public state included in a decision trace."""
        return {
            "as_of_time": self.as_of_time.isoformat(),
            "best_ask": self.best_ask,
            "best_bid": self.best_bid,
            "cutoff": self.cutoff.isoformat(),
            "mid_price": self.mid_price,
            "signals": [
                signal.to_mapping()
                for signal in sorted(self.signals, key=lambda item: item.kind.value)
            ],
            "visible_order_flow": self.visible_order_flow.to_mapping(),
        }


@dataclass(frozen=True, slots=True)
class OwnPortfolioState:
    """Only the deciding agent's balance, reservations, and cost basis."""

    agent_id: str
    cash: int
    inventory: int
    reserved_cash: int
    reserved_inventory: int
    average_cost: int | None
    open_order_count: int = 0
    pending_buy_quantity: int = 0

    def __post_init__(self) -> None:
        _identifier(self.agent_id, "agent_id")
        for field in (
            "cash",
            "inventory",
            "reserved_cash",
            "reserved_inventory",
            "open_order_count",
            "pending_buy_quantity",
        ):
            _non_negative_int(getattr(self, field), field)
        if self.reserved_cash > self.cash:
            raise AgentSchemaError("reserved_cash must not exceed cash")
        if self.reserved_inventory > self.inventory:
            raise AgentSchemaError("reserved_inventory must not exceed inventory")
        if self.average_cost is not None and (
            isinstance(self.average_cost, bool)
            or not isinstance(self.average_cost, int)
            or self.average_cost <= 0
        ):
            raise AgentSchemaError("average_cost must be null or a positive integer")
        if self.inventory > 0 and self.average_cost is None:
            raise AgentSchemaError(
                "average_cost is required when inventory is positive"
            )

    @property
    def available_cash(self) -> int:
        """Cash not already reserved by active or pending orders."""
        return self.cash - self.reserved_cash

    @property
    def available_inventory(self) -> int:
        """Inventory not already reserved by active or pending orders."""
        return self.inventory - self.reserved_inventory

    def to_mapping(self) -> dict[str, object]:
        """Return the private state captured only in this agent's trace."""
        return {
            "agent_id": self.agent_id,
            "available_cash": self.available_cash,
            "available_inventory": self.available_inventory,
            "average_cost": self.average_cost,
            "cash": self.cash,
            "inventory": self.inventory,
            "open_order_count": self.open_order_count,
            "pending_buy_quantity": self.pending_buy_quantity,
            "reserved_cash": self.reserved_cash,
            "reserved_inventory": self.reserved_inventory,
        }


@dataclass(frozen=True, slots=True)
class RiskLimits:
    """Hard pre-trade limits; market permission needs a future depth contract."""

    max_order_quantity: int
    max_order_notional: int
    max_inventory: int
    max_open_orders: int
    cash_buffer: int = 0
    fee_bps: int = 0
    allow_market_orders: bool = True

    def __post_init__(self) -> None:
        for field in (
            "max_order_quantity",
            "max_order_notional",
            "max_inventory",
            "max_open_orders",
        ):
            value = _non_negative_int(getattr(self, field), field)
            if value == 0:
                raise AgentSchemaError(f"{field} must be positive")
        _non_negative_int(self.cash_buffer, "cash_buffer")
        _non_negative_int(self.fee_bps, "fee_bps")
        if self.fee_bps > 10_000:
            raise AgentSchemaError("fee_bps must not exceed 10000")
        if not isinstance(self.allow_market_orders, bool):
            raise AgentSchemaError("allow_market_orders must be a boolean")

    def to_mapping(self) -> dict[str, object]:
        """Return a machine-readable risk contract."""
        return {
            "allow_market_orders": self.allow_market_orders,
            "cash_buffer": self.cash_buffer,
            "fee_bps": self.fee_bps,
            "max_inventory": self.max_inventory,
            "max_open_orders": self.max_open_orders,
            "max_order_notional": self.max_order_notional,
            "max_order_quantity": self.max_order_quantity,
        }


@dataclass(frozen=True, slots=True)
class OrderPolicyConfig:
    """Immutable numerical configuration for the fast order policy."""

    tick_size: int
    base_order_quantity: int
    temperature: float = 1.0
    limit_offset_ticks: int = 0

    def __post_init__(self) -> None:
        for field in ("tick_size", "base_order_quantity"):
            value = _non_negative_int(getattr(self, field), field)
            if value == 0:
                raise AgentSchemaError(f"{field} must be positive")
        temperature = _finite(self.temperature, "temperature")
        if temperature <= 0.0:
            raise AgentSchemaError("temperature must be positive")
        object.__setattr__(self, "temperature", temperature)
        _non_negative_int(self.limit_offset_ticks, "limit_offset_ticks")

    def to_mapping(self) -> dict[str, object]:
        """Return a machine-readable policy configuration."""
        return {
            "base_order_quantity": self.base_order_quantity,
            "limit_offset_ticks": self.limit_offset_ticks,
            "temperature": self.temperature,
            "tick_size": self.tick_size,
        }


@dataclass(frozen=True, slots=True)
class SignalContribution:
    """One named contribution to the fast policy's action scores."""

    name: str
    raw_value: float
    weight: float
    contribution: float
    enabled: bool

    def __post_init__(self) -> None:
        _identifier(self.name, "name")
        for field in ("raw_value", "weight", "contribution"):
            object.__setattr__(self, field, _finite(getattr(self, field), field))
        if not isinstance(self.enabled, bool):
            raise AgentSchemaError("enabled must be a boolean")

    def to_mapping(self) -> dict[str, object]:
        """Return the auditable signal decomposition."""
        return {
            "contribution": self.contribution,
            "enabled": self.enabled,
            "name": self.name,
            "raw_value": self.raw_value,
            "weight": self.weight,
        }


@dataclass(frozen=True, slots=True)
class ActionProbabilities:
    """Normalized probabilities in a fixed buy/hold/sell action space."""

    buy: float
    hold: float
    sell: float

    def __post_init__(self) -> None:
        for field in ("buy", "hold", "sell"):
            value = _finite(getattr(self, field), field)
            if not 0.0 <= value <= 1.0:
                raise AgentSchemaError(f"{field} probability must be in [0, 1]")
            object.__setattr__(self, field, value)
        if not math.isclose(self.buy + self.hold + self.sell, 1.0, abs_tol=1e-12):
            raise AgentSchemaError("action probabilities must sum to one")

    def to_mapping(self) -> dict[str, object]:
        """Return probabilities in stable action order."""
        return {"buy": self.buy, "hold": self.hold, "sell": self.sell}


@dataclass(frozen=True, slots=True)
class ActionIntent:
    """A final, risk-constrained instruction produced by the policy."""

    action: DecisionAction
    order_kind: OrderKind | None
    quantity: int
    limit_price: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.action, DecisionAction):
            raise AgentSchemaError("action must be a DecisionAction")
        _non_negative_int(self.quantity, "quantity")
        if self.action is DecisionAction.HOLD:
            if (
                self.order_kind is not None
                or self.quantity != 0
                or self.limit_price is not None
            ):
                raise AgentSchemaError("hold action must not contain an order")
            return
        if not isinstance(self.order_kind, OrderKind):
            raise AgentSchemaError("trading action must have an OrderKind")
        if self.quantity <= 0:
            raise AgentSchemaError("trading action quantity must be positive")
        if self.order_kind is OrderKind.LIMIT:
            if (
                isinstance(self.limit_price, bool)
                or not isinstance(self.limit_price, int)
                or self.limit_price <= 0
            ):
                raise AgentSchemaError("limit order must have a positive limit_price")
        elif self.limit_price is not None:
            raise AgentSchemaError("market order must not have a limit_price")

    def to_mapping(self) -> dict[str, object]:
        """Return the final action representation."""
        return {
            "action": self.action.value,
            "limit_price": self.limit_price,
            "order_kind": self.order_kind.value
            if self.order_kind is not None
            else None,
            "quantity": self.quantity,
        }


@dataclass(frozen=True, slots=True)
class ConstraintCheck:
    """One hard-limit adjustment recorded in the decision trace."""

    code: ConstraintCode
    action: DecisionAction
    requested_quantity: int
    permitted_quantity: int
    detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.code, ConstraintCode):
            raise AgentSchemaError("code must be a ConstraintCode")
        if self.action is DecisionAction.HOLD or not isinstance(
            self.action, DecisionAction
        ):
            raise AgentSchemaError("constraint action must be buy or sell")
        _non_negative_int(self.requested_quantity, "requested_quantity")
        _non_negative_int(self.permitted_quantity, "permitted_quantity")
        if self.permitted_quantity > self.requested_quantity:
            raise AgentSchemaError(
                "permitted quantity must not exceed requested quantity"
            )
        if not isinstance(self.detail, str) or not self.detail:
            raise AgentSchemaError("constraint detail must be non-empty")

    def to_mapping(self) -> dict[str, object]:
        """Return the hard-limit intervention."""
        return {
            "action": self.action.value,
            "code": self.code.value,
            "detail": self.detail,
            "permitted_quantity": self.permitted_quantity,
            "requested_quantity": self.requested_quantity,
        }


@dataclass(frozen=True, slots=True)
class DecisionTrace:
    """Complete structured trace for one deterministic agent decision."""

    event_key: EventKey
    agent_id: str
    archetype: AgentArchetype
    available_market_state: PublicMarketObservation
    available_own_state: OwnPortfolioState
    cognitive_state: CognitiveState
    parameters: BehaviorParameters
    mechanisms: BehaviorMechanisms
    policy_config: OrderPolicyConfig
    risk_limits: RiskLimits
    trusted_cutoff: datetime
    signals: tuple[SignalContribution, ...]
    action_probabilities: ActionProbabilities
    action_choice_draw: RandomDrawReceipt
    final_action: ActionIntent
    constraints: tuple[ConstraintCheck, ...]
    calibration_status: str = dataclass_field(
        default="synthetic_uncalibrated",
        init=False,
    )
    schema_version: str = AGENT_DECISION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.event_key, EventKey):
            raise AgentSchemaError("event_key must be an EventKey")
        _identifier(self.agent_id, "agent_id")
        if not isinstance(self.available_market_state, PublicMarketObservation):
            raise AgentSchemaError(
                "available_market_state must be a PublicMarketObservation"
            )
        if not isinstance(self.available_own_state, OwnPortfolioState):
            raise AgentSchemaError("available_own_state must be OwnPortfolioState")
        if self.agent_id != self.available_own_state.agent_id:
            raise AgentSchemaError("decision agent_id must match own portfolio state")
        if not isinstance(self.archetype, AgentArchetype):
            raise AgentSchemaError("archetype must be an AgentArchetype")
        if not isinstance(self.cognitive_state, CognitiveState):
            raise AgentSchemaError("cognitive_state must be a CognitiveState")
        if not isinstance(self.parameters, BehaviorParameters):
            raise AgentSchemaError("parameters must be BehaviorParameters")
        if not isinstance(self.mechanisms, BehaviorMechanisms):
            raise AgentSchemaError("mechanisms must be BehaviorMechanisms")
        if not isinstance(self.policy_config, OrderPolicyConfig):
            raise AgentSchemaError("policy_config must be an OrderPolicyConfig")
        if not isinstance(self.risk_limits, RiskLimits):
            raise AgentSchemaError("risk_limits must be RiskLimits")
        object.__setattr__(
            self,
            "trusted_cutoff",
            _aware_utc(self.trusted_cutoff, "trusted_cutoff"),
        )
        if self.trusted_cutoff != self.available_market_state.cutoff:
            raise AgentTemporalLeakageError(
                "trace trusted_cutoff must match the available market state"
            )
        if not isinstance(self.action_probabilities, ActionProbabilities):
            raise AgentSchemaError("action_probabilities must be ActionProbabilities")
        if type(self.action_choice_draw) is not RandomDrawReceipt:
            raise AgentSchemaError(
                "action_choice_draw must be an exact RandomDrawReceipt"
            )
        draw = self.action_choice_draw
        try:
            recomputed_draw = RandomDrawReceipt.from_mapping(draw.to_mapping())
        except (RuntimeError, TypeError, ValueError) as error:
            raise AgentSchemaError(
                "action_choice_draw failed independent deterministic replay"
            ) from error
        if recomputed_draw != draw:
            raise AgentSchemaError(
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
            raise AgentSchemaError(
                "action choice receipt differs from the fixed decision draw contract"
            )
        if not isinstance(self.final_action, ActionIntent):
            raise AgentSchemaError("final_action must be an ActionIntent")
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
            raise AgentSchemaError(
                "final action differs from the receipted probability draw"
            )
        if not isinstance(self.signals, tuple) or not all(
            isinstance(item, SignalContribution) for item in self.signals
        ):
            raise AgentSchemaError("signals must be an immutable contribution tuple")
        if not isinstance(self.constraints, tuple) or not all(
            isinstance(item, ConstraintCheck) for item in self.constraints
        ):
            raise AgentSchemaError("constraints must be an immutable tuple")
        if self.schema_version != AGENT_DECISION_SCHEMA_VERSION:
            raise AgentSchemaError("unsupported decision trace schema version")

    def to_mapping(self) -> dict[str, object]:
        """Return a canonical JSON-compatible trace payload."""
        return {
            "action_probabilities": self.action_probabilities.to_mapping(),
            "action_choice_draw": self.action_choice_draw.to_mapping(),
            "agent_id": self.agent_id,
            "archetype": self.archetype.value,
            "available_state": {
                "market": self.available_market_state.to_mapping(),
                "own": self.available_own_state.to_mapping(),
            },
            "cognitive_state": self.cognitive_state.to_mapping(),
            "calibration_status": self.calibration_status,
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
            "risk_limits": self.risk_limits.to_mapping(),
            "schema_version": self.schema_version,
            "signals": [item.to_mapping() for item in self.signals],
            "trusted_cutoff": self.trusted_cutoff.isoformat(),
        }


__all__ = [
    "AGENT_DECISION_SCHEMA_VERSION",
    "ActionIntent",
    "ActionProbabilities",
    "AgentArchetype",
    "AgentDecisionError",
    "AgentError",
    "AgentSchemaError",
    "AgentTemporalLeakageError",
    "ConstraintCheck",
    "ConstraintCode",
    "DecisionAction",
    "DecisionTrace",
    "OrderKind",
    "OrderPolicyConfig",
    "OwnPortfolioState",
    "PublicMarketObservation",
    "PublicSignal",
    "RiskLimits",
    "SignalContribution",
    "SignalKind",
]
