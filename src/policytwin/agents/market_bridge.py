"""Fail-closed conversion from an agent decision trace to a CLOB command.

The bridge does not run a strategy or infer missing state.  It replays the
decision, verifies that its private projection matches the committed market
snapshot, and only then emits a deterministic command and audit receipt.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from threading import RLock
from typing import NoReturn

from policytwin.agents.models import (
    DecisionAction,
    DecisionTrace,
    OrderKind,
)
from policytwin.agents.policies import FastOrderPolicy
from policytwin.agents.scenario import (
    ScenarioDecisionTrace,
    ScenarioFastOrderPolicy,
)
from policytwin.market import (
    LimitOrder,
    MarketCommand,
    MarketConfig,
    MarketSnapshot,
    Side,
)
from policytwin.provenance import (
    canonical_json_bytes,
    read_jsonl_objects,
    sha256_bytes,
)

AGENT_MARKET_BRIDGE_SCHEMA_VERSION = "policytwin.agent-market-bridge.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_NULL_HASH = "0" * 64


class AgentMarketBridgeRule(StrEnum):
    """Stable conversion and audit outcomes."""

    OK = "OK"
    INVALID_SCHEMA = "INVALID_SCHEMA"
    TRACE_REPLAY_MISMATCH = "TRACE_REPLAY_MISMATCH"
    MARKET_CONFIG_MISMATCH = "MARKET_CONFIG_MISMATCH"
    STALE_MARKET_STATE = "STALE_MARKET_STATE"
    FEE_BOUND_MISMATCH = "FEE_BOUND_MISMATCH"
    UNSUPPORTED_ORDER_KIND = "UNSUPPORTED_ORDER_KIND"
    INVALID_PRICE_GRID = "INVALID_PRICE_GRID"
    AUDIT_PERSISTENCE_FAILURE = "AUDIT_PERSISTENCE_FAILURE"


class AgentMarketBridgeError(RuntimeError):
    """Raised after a rejected conversion has been recorded."""

    def __init__(self, message: str, *, rule_code: AgentMarketBridgeRule) -> None:
        super().__init__(message)
        self.rule_code = rule_code
        self.audit_event: AgentMarketBridgeAuditEvent | None = None


def _fail(message: str, rule: AgentMarketBridgeRule) -> NoReturn:
    raise AgentMarketBridgeError(message, rule_code=rule)


def _sha(value: str, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail(f"{field} must be a SHA-256 digest", AgentMarketBridgeRule.INVALID_SCHEMA)
    return value


def _trace_sha256(trace: DecisionTrace) -> str:
    return sha256_bytes(canonical_json_bytes(trace.to_mapping()))


def _scenario_trace_sha256(trace: ScenarioDecisionTrace) -> str:
    return sha256_bytes(canonical_json_bytes(trace.to_mapping()))


def _command_sha256(command: MarketCommand | None) -> str:
    if command is None:
        return _NULL_HASH
    return sha256_bytes(canonical_json_bytes(command.to_mapping()))


@dataclass(frozen=True, slots=True)
class DecisionCommandBinding:
    """Tamper-sensitive binding between one trace, state, config, and command."""

    trace_sha256: str
    market_snapshot_sha256: str
    market_config_sha256: str
    command_sha256: str
    event_key: str
    trader_id: str
    order_id: str | None
    hold: bool
    schema_version: str = AGENT_MARKET_BRIDGE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field in (
            "trace_sha256",
            "market_snapshot_sha256",
            "market_config_sha256",
            "command_sha256",
        ):
            _sha(getattr(self, field), field)
        if not isinstance(self.event_key, str) or not self.event_key:
            _fail("event_key must be non-empty", AgentMarketBridgeRule.INVALID_SCHEMA)
        if not isinstance(self.trader_id, str) or not self.trader_id:
            _fail("trader_id must be non-empty", AgentMarketBridgeRule.INVALID_SCHEMA)
        if not isinstance(self.hold, bool):
            _fail("hold must be boolean", AgentMarketBridgeRule.INVALID_SCHEMA)
        if self.hold != (self.order_id is None):
            _fail(
                "hold and order_id must agree",
                AgentMarketBridgeRule.INVALID_SCHEMA,
            )
        if self.hold != (self.command_sha256 == _NULL_HASH):
            _fail(
                "hold and command commitment must agree",
                AgentMarketBridgeRule.INVALID_SCHEMA,
            )
        if self.schema_version != AGENT_MARKET_BRIDGE_SCHEMA_VERSION:
            _fail(
                "unsupported bridge schema version",
                AgentMarketBridgeRule.INVALID_SCHEMA,
            )

    def to_mapping(self) -> dict[str, object]:
        """Return the complete binding payload."""
        return {
            "command_sha256": self.command_sha256,
            "event_key": self.event_key,
            "hold": self.hold,
            "market_config_sha256": self.market_config_sha256,
            "market_snapshot_sha256": self.market_snapshot_sha256,
            "order_id": self.order_id,
            "schema_version": self.schema_version,
            "trace_sha256": self.trace_sha256,
            "trader_id": self.trader_id,
        }

    @property
    def sha256(self) -> str:
        """Return the binding commitment."""
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))


@dataclass(frozen=True, slots=True)
class AgentMarketBridgeAuditEvent:
    """One hash-chained accepted or rejected conversion."""

    sequence: int
    outcome: str
    rule_code: AgentMarketBridgeRule
    trace_sha256: str
    market_snapshot_sha256: str
    market_config_sha256: str
    binding_sha256: str
    event_key: str | None
    previous_event_sha256: str
    event_sha256: str
    schema_version: str = AGENT_MARKET_BRIDGE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence < 0
        ):
            _fail(
                "audit sequence must be non-negative",
                AgentMarketBridgeRule.INVALID_SCHEMA,
            )
        if self.outcome not in {"allowed", "rejected"}:
            _fail(
                "audit outcome is invalid",
                AgentMarketBridgeRule.INVALID_SCHEMA,
            )
        if not isinstance(self.rule_code, AgentMarketBridgeRule):
            _fail(
                "audit rule code is invalid",
                AgentMarketBridgeRule.INVALID_SCHEMA,
            )
        for field in (
            "trace_sha256",
            "market_snapshot_sha256",
            "market_config_sha256",
            "binding_sha256",
            "previous_event_sha256",
            "event_sha256",
        ):
            _sha(getattr(self, field), field)
        if self.event_key is not None and (
            not isinstance(self.event_key, str) or not self.event_key
        ):
            _fail(
                "audit event_key must be null or non-empty",
                AgentMarketBridgeRule.INVALID_SCHEMA,
            )
        if self.outcome == "allowed" and (
            self.rule_code is not AgentMarketBridgeRule.OK
            or self.binding_sha256 == _NULL_HASH
            or self.event_key is None
        ):
            _fail(
                "allowed audit row lacks a successful binding",
                AgentMarketBridgeRule.INVALID_SCHEMA,
            )
        if self.outcome == "rejected" and (
            self.rule_code is AgentMarketBridgeRule.OK
            or self.binding_sha256 != _NULL_HASH
        ):
            _fail(
                "rejected audit row has success fields",
                AgentMarketBridgeRule.INVALID_SCHEMA,
            )
        if self.schema_version != AGENT_MARKET_BRIDGE_SCHEMA_VERSION:
            _fail(
                "unsupported audit schema version",
                AgentMarketBridgeRule.INVALID_SCHEMA,
            )
        if sha256_bytes(canonical_json_bytes(self.core_mapping())) != self.event_sha256:
            _fail(
                "audit event hash does not match its fields",
                AgentMarketBridgeRule.INVALID_SCHEMA,
            )

    def core_mapping(self) -> dict[str, object]:
        """Return fields covered by the event hash."""
        return {
            "binding_sha256": self.binding_sha256,
            "event_key": self.event_key,
            "event_type": "agent_market_bridge",
            "market_config_sha256": self.market_config_sha256,
            "market_snapshot_sha256": self.market_snapshot_sha256,
            "outcome": self.outcome,
            "previous_event_sha256": self.previous_event_sha256,
            "rule_code": self.rule_code.value,
            "schema_version": self.schema_version,
            "sequence": self.sequence,
            "trace_sha256": self.trace_sha256,
        }

    def to_mapping(self) -> dict[str, object]:
        """Return the persisted audit representation."""
        return {**self.core_mapping(), "event_sha256": self.event_sha256}


@dataclass(frozen=True, slots=True)
class DecisionCommandConversion:
    """Successful conversion; a hold deliberately has no command."""

    command: MarketCommand | None
    binding: DecisionCommandBinding
    audit_event: AgentMarketBridgeAuditEvent

    def __post_init__(self) -> None:
        if self.binding.command_sha256 != _command_sha256(self.command):
            _fail(
                "conversion command disagrees with binding",
                AgentMarketBridgeRule.INVALID_SCHEMA,
            )
        if self.audit_event.binding_sha256 != self.binding.sha256:
            _fail(
                "conversion audit disagrees with binding",
                AgentMarketBridgeRule.INVALID_SCHEMA,
            )


class AgentMarketBridge:
    """Replay and verify decisions before they enter the market scheduler."""

    def __init__(self, audit_path: Path | None = None) -> None:
        self._events: list[AgentMarketBridgeAuditEvent] = []
        self._lock = RLock()
        self._path = audit_path.expanduser().resolve() if audit_path else None
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            try:
                self._path.open("xb").close()
            except OSError as error:
                raise AgentMarketBridgeError(
                    "cannot initialize bridge audit log",
                    rule_code=AgentMarketBridgeRule.AUDIT_PERSISTENCE_FAILURE,
                ) from error

    @property
    def events(self) -> tuple[AgentMarketBridgeAuditEvent, ...]:
        """Return immutable audit events."""
        with self._lock:
            return tuple(self._events)

    @property
    def sha256(self) -> str:
        """Return the complete audit-ledger commitment."""
        return sha256_bytes(
            canonical_json_bytes([event.to_mapping() for event in self.events])
        )

    def convert(
        self,
        *,
        trace: DecisionTrace,
        market_snapshot: MarketSnapshot,
        market_config: MarketConfig,
    ) -> DecisionCommandConversion:
        """Convert one current, reproducible decision or fail with an audit row."""
        trace_hash = _NULL_HASH
        snapshot_hash = _NULL_HASH
        config_hash = _NULL_HASH
        event_key: str | None = None
        try:
            if not isinstance(trace, DecisionTrace):
                _fail(
                    "trace must be a DecisionTrace",
                    AgentMarketBridgeRule.INVALID_SCHEMA,
                )
            trace_hash = _trace_sha256(trace)
            event_key = trace.event_key.value
            if not isinstance(market_snapshot, MarketSnapshot):
                _fail(
                    "market_snapshot must be a MarketSnapshot",
                    AgentMarketBridgeRule.INVALID_SCHEMA,
                )
            snapshot_hash = market_snapshot.sha256
            if not isinstance(market_config, MarketConfig):
                _fail(
                    "market_config must be a MarketConfig",
                    AgentMarketBridgeRule.INVALID_SCHEMA,
                )
            config_hash = market_config.sha256
            command = self._validated_command(trace, market_snapshot, market_config)
            binding = DecisionCommandBinding(
                trace_sha256=trace_hash,
                market_snapshot_sha256=snapshot_hash,
                market_config_sha256=config_hash,
                command_sha256=_command_sha256(command),
                event_key=event_key,
                trader_id=trace.agent_id,
                order_id=command.order_id if isinstance(command, LimitOrder) else None,
                hold=command is None,
            )
        except AgentMarketBridgeError as error:
            audit = self._append(
                outcome="rejected",
                rule_code=error.rule_code,
                trace_sha256=trace_hash,
                snapshot_sha256=snapshot_hash,
                config_sha256=config_hash,
                binding_sha256=_NULL_HASH,
                event_key=event_key,
            )
            error.audit_event = audit
            raise
        audit = self._append(
            outcome="allowed",
            rule_code=AgentMarketBridgeRule.OK,
            trace_sha256=trace_hash,
            snapshot_sha256=snapshot_hash,
            config_sha256=config_hash,
            binding_sha256=binding.sha256,
            event_key=event_key,
        )
        return DecisionCommandConversion(command, binding, audit)

    def convert_scenario(
        self,
        *,
        trace: ScenarioDecisionTrace,
        market_snapshot: MarketSnapshot,
        market_config: MarketConfig,
    ) -> DecisionCommandConversion:
        """Convert an explicitly receipt-bound future-scenario decision.

        This is intentionally a distinct entry point.  ``convert`` still
        accepts only the observed/pre-cutoff ``DecisionTrace`` schema.
        """
        trace_hash = _NULL_HASH
        snapshot_hash = _NULL_HASH
        config_hash = _NULL_HASH
        event_key: str | None = None
        try:
            if not isinstance(trace, ScenarioDecisionTrace):
                _fail(
                    "trace must be a ScenarioDecisionTrace",
                    AgentMarketBridgeRule.INVALID_SCHEMA,
                )
            trace_hash = _scenario_trace_sha256(trace)
            event_key = trace.event_key.value
            if not isinstance(market_snapshot, MarketSnapshot):
                _fail(
                    "market_snapshot must be a MarketSnapshot",
                    AgentMarketBridgeRule.INVALID_SCHEMA,
                )
            snapshot_hash = market_snapshot.sha256
            if not isinstance(market_config, MarketConfig):
                _fail(
                    "market_config must be a MarketConfig",
                    AgentMarketBridgeRule.INVALID_SCHEMA,
                )
            config_hash = market_config.sha256
            command = self._validated_scenario_command(
                trace, market_snapshot, market_config
            )
            binding = DecisionCommandBinding(
                trace_sha256=trace_hash,
                market_snapshot_sha256=snapshot_hash,
                market_config_sha256=config_hash,
                command_sha256=_command_sha256(command),
                event_key=event_key,
                trader_id=trace.agent_id,
                order_id=command.order_id if isinstance(command, LimitOrder) else None,
                hold=command is None,
            )
        except AgentMarketBridgeError as error:
            audit = self._append(
                outcome="rejected",
                rule_code=error.rule_code,
                trace_sha256=trace_hash,
                snapshot_sha256=snapshot_hash,
                config_sha256=config_hash,
                binding_sha256=_NULL_HASH,
                event_key=event_key,
            )
            error.audit_event = audit
            raise
        audit = self._append(
            outcome="allowed",
            rule_code=AgentMarketBridgeRule.OK,
            trace_sha256=trace_hash,
            snapshot_sha256=snapshot_hash,
            config_sha256=config_hash,
            binding_sha256=binding.sha256,
            event_key=event_key,
        )
        return DecisionCommandConversion(command, binding, audit)

    def _validated_command(
        self,
        trace: DecisionTrace,
        snapshot: MarketSnapshot,
        config: MarketConfig,
    ) -> MarketCommand | None:
        replayed = FastOrderPolicy().decide(
            agent_id=trace.agent_id,
            archetype=trace.archetype,
            parameters=trace.parameters,
            mechanisms=trace.mechanisms,
            cognitive_state=trace.cognitive_state,
            portfolio=trace.available_own_state,
            observation=trace.available_market_state,
            trusted_cutoff=trace.trusted_cutoff,
            event_key=trace.event_key,
            config=trace.policy_config,
            risk_limits=trace.risk_limits,
        )
        if replayed != trace:
            _fail(
                "decision trace does not exactly replay",
                AgentMarketBridgeRule.TRACE_REPLAY_MISMATCH,
            )
        if trace.policy_config.tick_size != config.tick_size:
            _fail(
                "agent and market tick sizes differ",
                AgentMarketBridgeRule.MARKET_CONFIG_MISMATCH,
            )
        if trace.risk_limits.fee_bps < max(config.maker_fee_bps, config.taker_fee_bps):
            _fail(
                "agent risk fee bound is below the market fee",
                AgentMarketBridgeRule.FEE_BOUND_MISMATCH,
            )
        self._verify_snapshot_projection(trace, snapshot)
        intent = trace.final_action
        if intent.action is DecisionAction.HOLD:
            return None
        if intent.order_kind is not OrderKind.LIMIT or intent.limit_price is None:
            _fail(
                "only price-bounded limit intents can enter the CLOB",
                AgentMarketBridgeRule.UNSUPPORTED_ORDER_KIND,
            )
        if intent.limit_price % config.tick_size != 0:
            _fail(
                "intent price is outside the market grid",
                AgentMarketBridgeRule.INVALID_PRICE_GRID,
            )
        digest = sha256_bytes(
            canonical_json_bytes(
                {
                    "config_sha256": config.sha256,
                    "snapshot_sha256": snapshot.sha256,
                    "trace_sha256": _trace_sha256(trace),
                }
            )
        )
        return LimitOrder(
            event_key=trace.event_key,
            submitted_at=trace.event_key.simulation_step,
            order_id=f"agent-order-{digest[:24]}",
            trader_id=trace.agent_id,
            side=(Side.BUY if intent.action is DecisionAction.BUY else Side.SELL),
            quantity=intent.quantity,
            price=intent.limit_price,
        )

    def _validated_scenario_command(
        self,
        trace: ScenarioDecisionTrace,
        snapshot: MarketSnapshot,
        config: MarketConfig,
    ) -> MarketCommand | None:
        replayed = ScenarioFastOrderPolicy().decide(
            agent_id=trace.agent_id,
            archetype=trace.archetype,
            parameters=trace.parameters,
            mechanisms=trace.mechanisms,
            cognitive_snapshot=trace.cognitive_snapshot,
            portfolio=trace.available_own_state,
            observation=trace.available_market_state,
            authority=trace.authority,
            event_key=trace.event_key,
            config=trace.policy_config,
            risk_limits=trace.risk_limits,
        )
        if replayed != trace:
            _fail(
                "scenario decision trace does not exactly replay",
                AgentMarketBridgeRule.TRACE_REPLAY_MISMATCH,
            )
        if trace.policy_config.tick_size != config.tick_size:
            _fail(
                "agent and market tick sizes differ",
                AgentMarketBridgeRule.MARKET_CONFIG_MISMATCH,
            )
        if trace.risk_limits.fee_bps < max(config.maker_fee_bps, config.taker_fee_bps):
            _fail(
                "agent risk fee bound is below the market fee",
                AgentMarketBridgeRule.FEE_BOUND_MISMATCH,
            )
        self._verify_scenario_snapshot_projection(trace, snapshot)
        intent = trace.final_action
        if intent.action is DecisionAction.HOLD:
            return None
        if intent.order_kind is not OrderKind.LIMIT or intent.limit_price is None:
            _fail(
                "only price-bounded limit intents can enter the CLOB",
                AgentMarketBridgeRule.UNSUPPORTED_ORDER_KIND,
            )
        if intent.limit_price % config.tick_size != 0:
            _fail(
                "intent price is outside the market grid",
                AgentMarketBridgeRule.INVALID_PRICE_GRID,
            )
        digest = sha256_bytes(
            canonical_json_bytes(
                {
                    "config_sha256": config.sha256,
                    "snapshot_sha256": snapshot.sha256,
                    "trace_sha256": _scenario_trace_sha256(trace),
                }
            )
        )
        return LimitOrder(
            event_key=trace.event_key,
            submitted_at=trace.event_key.simulation_step,
            order_id=f"scenario-agent-order-{digest[:24]}",
            trader_id=trace.agent_id,
            side=(Side.BUY if intent.action is DecisionAction.BUY else Side.SELL),
            quantity=intent.quantity,
            price=intent.limit_price,
        )

    def _verify_snapshot_projection(
        self, trace: DecisionTrace, snapshot: MarketSnapshot
    ) -> None:
        own = trace.available_own_state
        accounts = tuple(
            account
            for account in snapshot.accounts
            if account.trader_id == own.agent_id
        )
        if len(accounts) != 1:
            _fail(
                "agent account is absent or duplicated in market state",
                AgentMarketBridgeRule.STALE_MARKET_STATE,
            )
        account = accounts[0]
        open_orders = tuple(
            order
            for order in snapshot.resting_orders
            if order.trader_id == own.agent_id
        )
        if (
            account.cash != own.cash
            or account.asset != own.inventory
            or account.reserved_cash != own.reserved_cash
            or account.reserved_asset != own.reserved_inventory
            or len(open_orders) != own.open_order_count
            or own.pending_buy_quantity != 0
        ):
            _fail(
                "agent private projection is stale or has unverifiable pending orders",
                AgentMarketBridgeRule.STALE_MARKET_STATE,
            )
        bids = tuple(
            order.price for order in snapshot.resting_orders if order.side is Side.BUY
        )
        asks = tuple(
            order.price for order in snapshot.resting_orders if order.side is Side.SELL
        )
        if not bids or not asks:
            _fail(
                "two-sided top of book is required for a verified decision",
                AgentMarketBridgeRule.STALE_MARKET_STATE,
            )
        best_bid = max(bids)
        best_ask = min(asks)
        observation = trace.available_market_state
        if (
            observation.best_bid != best_bid
            or observation.best_ask != best_ask
            or observation.mid_price != (best_bid + best_ask) // 2
        ):
            _fail(
                "public top-of-book projection is stale",
                AgentMarketBridgeRule.STALE_MARKET_STATE,
            )

    def _verify_scenario_snapshot_projection(
        self, trace: ScenarioDecisionTrace, snapshot: MarketSnapshot
    ) -> None:
        own = trace.available_own_state
        accounts = tuple(
            account
            for account in snapshot.accounts
            if account.trader_id == own.agent_id
        )
        if len(accounts) != 1:
            _fail(
                "agent account is absent or duplicated in market state",
                AgentMarketBridgeRule.STALE_MARKET_STATE,
            )
        account = accounts[0]
        open_orders = tuple(
            order
            for order in snapshot.resting_orders
            if order.trader_id == own.agent_id
        )
        if (
            account.cash != own.cash
            or account.asset != own.inventory
            or account.reserved_cash != own.reserved_cash
            or account.reserved_asset != own.reserved_inventory
            or len(open_orders) != own.open_order_count
            or own.pending_buy_quantity != 0
        ):
            _fail(
                "agent private projection is stale or has unverifiable pending orders",
                AgentMarketBridgeRule.STALE_MARKET_STATE,
            )
        bids = tuple(
            order.price for order in snapshot.resting_orders if order.side is Side.BUY
        )
        asks = tuple(
            order.price for order in snapshot.resting_orders if order.side is Side.SELL
        )
        if not bids or not asks:
            _fail(
                "two-sided top of book is required for a verified decision",
                AgentMarketBridgeRule.STALE_MARKET_STATE,
            )
        observation = trace.available_market_state
        best_bid = max(bids)
        best_ask = min(asks)
        if (
            observation.market_state_sha256 != snapshot.sha256
            or observation.best_bid != best_bid
            or observation.best_ask != best_ask
            or observation.mid_price != (best_bid + best_ask) // 2
        ):
            _fail(
                "generated public market projection is stale",
                AgentMarketBridgeRule.STALE_MARKET_STATE,
            )

    def _append(
        self,
        *,
        outcome: str,
        rule_code: AgentMarketBridgeRule,
        trace_sha256: str,
        snapshot_sha256: str,
        config_sha256: str,
        binding_sha256: str,
        event_key: str | None,
    ) -> AgentMarketBridgeAuditEvent:
        with self._lock:
            previous = self._events[-1].event_sha256 if self._events else _NULL_HASH
            core: dict[str, object] = {
                "binding_sha256": binding_sha256,
                "event_key": event_key,
                "event_type": "agent_market_bridge",
                "market_config_sha256": config_sha256,
                "market_snapshot_sha256": snapshot_sha256,
                "outcome": outcome,
                "previous_event_sha256": previous,
                "rule_code": rule_code.value,
                "schema_version": AGENT_MARKET_BRIDGE_SCHEMA_VERSION,
                "sequence": len(self._events),
                "trace_sha256": trace_sha256,
            }
            event = AgentMarketBridgeAuditEvent(
                sequence=len(self._events),
                outcome=outcome,
                rule_code=rule_code,
                trace_sha256=trace_sha256,
                market_snapshot_sha256=snapshot_sha256,
                market_config_sha256=config_sha256,
                binding_sha256=binding_sha256,
                event_key=event_key,
                previous_event_sha256=previous,
                event_sha256=sha256_bytes(canonical_json_bytes(core)),
            )
            self._persist(event)
            self._events.append(event)
            return event

    def _persist(self, event: AgentMarketBridgeAuditEvent) -> None:
        if self._path is None:
            return
        try:
            with self._path.open("ab") as stream:
                stream.write(canonical_json_bytes(event.to_mapping()) + b"\n")
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as error:
            raise AgentMarketBridgeError(
                "cannot persist bridge audit event",
                rule_code=AgentMarketBridgeRule.AUDIT_PERSISTENCE_FAILURE,
            ) from error

    def assert_integrity(self) -> None:
        """Verify the in-memory chain and, when configured, exact JSONL bytes."""
        previous = _NULL_HASH
        for sequence, event in enumerate(self.events):
            if event.sequence != sequence or event.previous_event_sha256 != previous:
                _fail(
                    "bridge audit sequence or chain changed",
                    AgentMarketBridgeRule.INVALID_SCHEMA,
                )
            if sha256_bytes(canonical_json_bytes(event.core_mapping())) != (
                event.event_sha256
            ):
                _fail(
                    "bridge audit event hash changed",
                    AgentMarketBridgeRule.INVALID_SCHEMA,
                )
            previous = event.event_sha256
        if self._path is not None:
            observed = read_jsonl_objects(self._path)
            expected = tuple(event.to_mapping() for event in self.events)
            if observed != expected:
                _fail(
                    "bridge audit JSONL differs from memory",
                    AgentMarketBridgeRule.INVALID_SCHEMA,
                )


__all__ = [
    "AGENT_MARKET_BRIDGE_SCHEMA_VERSION",
    "AgentMarketBridge",
    "AgentMarketBridgeAuditEvent",
    "AgentMarketBridgeError",
    "AgentMarketBridgeRule",
    "DecisionCommandBinding",
    "DecisionCommandConversion",
]
