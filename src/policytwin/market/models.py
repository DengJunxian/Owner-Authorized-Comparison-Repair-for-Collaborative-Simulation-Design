"""Immutable schemas and deterministic logs for the discrete-event market."""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, NoReturn

from policytwin.counterfactual import EventKey
from policytwin.provenance import (
    JsonlEventWriter,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)

if TYPE_CHECKING:
    from policytwin.market.policy_gateway import PolicyGatewayConfig
    from policytwin.market.policy_replay import SyntheticPolicyReplayDescriptor
    from policytwin.policies.sec_tick_size_pilot import PolicyComplianceAuditEvent

MARKET_SCHEMA_VERSION = "policytwin.clob.v2"
SCHEDULER_SCHEMA_VERSION = "policytwin.clob.scheduler.v1"
FEE_DENOMINATOR_BPS = 10_000
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

type AuditScalar = str | int | bool | None


class Side(StrEnum):
    """Order side."""

    BUY = "buy"
    SELL = "sell"


class OrderStatus(StrEnum):
    """Terminal or active order lifecycle state."""

    RESTING = "resting"
    FILLED = "filled"
    CANCELED = "canceled"
    EXPIRED = "expired"


class MarketPolicyMode(StrEnum):
    """Whether the market is an explicitly synthetic no-policy baseline."""

    NO_POLICY_SYNTHETIC = "no_policy_synthetic"
    POLICY_REQUIRED = "policy_required"


class MarketRuleCode(StrEnum):
    """Stable machine-readable market validation and invariant rules."""

    OK = "OK"
    INVALID_CONFIG = "INVALID_CONFIG"
    INVALID_OPENING_STATE = "INVALID_OPENING_STATE"
    INVALID_EVENT = "INVALID_EVENT"
    INVALID_QUANTITY = "INVALID_QUANTITY"
    INVALID_PRICE_GRID = "INVALID_PRICE_GRID"
    DUPLICATE_EVENT_KEY = "DUPLICATE_EVENT_KEY"
    DUPLICATE_ORDER_ID = "DUPLICATE_ORDER_ID"
    UNKNOWN_TRADER = "UNKNOWN_TRADER"
    UNKNOWN_ORDER = "UNKNOWN_ORDER"
    ORDER_NOT_ACTIVE = "ORDER_NOT_ACTIVE"
    UNAUTHORIZED_CANCEL = "UNAUTHORIZED_CANCEL"
    INSUFFICIENT_CASH = "INSUFFICIENT_CASH"
    INSUFFICIENT_ASSET = "INSUFFICIENT_ASSET"
    SELF_TRADE = "SELF_TRADE"
    CASH_CONSERVATION = "CASH_CONSERVATION"
    ASSET_CONSERVATION = "ASSET_CONSERVATION"
    NEGATIVE_ORDER_QUANTITY = "NEGATIVE_ORDER_QUANTITY"
    RESERVATION_MISMATCH = "RESERVATION_MISMATCH"
    CROSSED_BOOK = "CROSSED_BOOK"
    SNAPSHOT_STRUCTURE = "SNAPSHOT_STRUCTURE"
    ENGINE_LATCHED = "ENGINE_LATCHED"
    ENGINE_SEALED = "ENGINE_SEALED"
    AUDIT_WRITE_FAILED = "AUDIT_WRITE_FAILED"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    STATE_COMMITMENT_MISMATCH = "STATE_COMMITMENT_MISMATCH"
    REPLAY_MISMATCH = "REPLAY_MISMATCH"
    POLICY_GATEWAY_REQUIRED = "POLICY_GATEWAY_REQUIRED"
    POLICY_GATEWAY_MISMATCH = "POLICY_GATEWAY_MISMATCH"
    POLICY_GATEWAY_ERROR = "POLICY_GATEWAY_ERROR"
    POLICY_QUOTE_REJECTED = "POLICY_QUOTE_REJECTED"
    POLICY_TRADE_REJECTED = "POLICY_TRADE_REJECTED"
    REPLAY_POLICY_CONTEXT_REQUIRED = "REPLAY_POLICY_CONTEXT_REQUIRED"
    HALT_TRIGGERED = "HALT_TRIGGERED"
    HALT_COMMAND_REJECTED = "HALT_COMMAND_REJECTED"
    HALT_COMMAND_ALLOWED = "HALT_COMMAND_ALLOWED"
    HALT_RECOVERED = "HALT_RECOVERED"
    HALT_GATEWAY_CORE_REJECTED = "HALT_GATEWAY_CORE_REJECTED"


class MarketError(RuntimeError):
    """Base class for deterministic market failures."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.audit_events: tuple[MarketAuditEvent, ...] = ()
        self.policy_audit_events: tuple[PolicyComplianceAuditEvent, ...] = ()


class MarketValidationError(MarketError):
    """Raised when a command or configuration is invalid."""


class MarketStateError(MarketError):
    """Raised when a requested state transition is not permitted."""


class ConservationViolation(MarketStateError):
    """Raised when cash or asset conservation is violated."""


class MarketInvariantError(MarketStateError):
    """Raised when another order-book invariant fails."""


class MarketAuditError(MarketError):
    """Raised when a mandatory market audit event cannot be persisted."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.audit_tail_rolled_back: bool | None = None


class EngineLatchedError(MarketStateError):
    """Raised after a previous failure has made the engine unusable."""


@dataclass(frozen=True, slots=True)
class MarketConfig:
    """Immutable integer-valued market mechanism configuration."""

    tick_size: int
    latency_steps: int = 0
    maker_fee_bps: int = 0
    taker_fee_bps: int = 0
    policy_mode: MarketPolicyMode = MarketPolicyMode.NO_POLICY_SYNTHETIC
    schema_version: str = MARKET_SCHEMA_VERSION
    scheduler_schema_version: str = SCHEDULER_SCHEMA_VERSION

    def to_mapping(self) -> dict[str, object]:
        """Return the canonical configuration representation."""
        return {
            "latency_steps": self.latency_steps,
            "maker_fee_bps": self.maker_fee_bps,
            "policy_mode": (
                self.policy_mode.value
                if isinstance(self.policy_mode, MarketPolicyMode)
                else self.policy_mode
            ),
            "scheduler_schema_version": self.scheduler_schema_version,
            "schema_version": self.schema_version,
            "taker_fee_bps": self.taker_fee_bps,
            "tick_size": self.tick_size,
        }

    @property
    def sha256(self) -> str:
        """Return the semantic configuration digest."""
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))


@dataclass(frozen=True, slots=True)
class AccountBalance:
    """Gross opening cash and asset holdings for one test participant."""

    trader_id: str
    cash: int
    asset: int

    def to_mapping(self) -> dict[str, object]:
        return {"asset": self.asset, "cash": self.cash, "trader_id": self.trader_id}


def event_key_to_mapping(event_key: EventKey) -> dict[str, object]:
    """Commit to both the semantic EventKey fields and its derived value."""
    return {
        "experiment_id": event_key.experiment_id,
        "purpose": event_key.purpose,
        "replicate_id": event_key.replicate_id,
        "schema_version": event_key.schema_version,
        "seed": event_key.seed,
        "simulation_step": event_key.simulation_step,
        "value": event_key.value,
    }


@dataclass(frozen=True, slots=True)
class OpeningOrder:
    """A pre-existing, non-crossing resting limit order."""

    event_key: EventKey
    order_id: str
    trader_id: str
    side: Side
    price: int
    quantity: int
    priority_time: int = 0

    def to_mapping(self) -> dict[str, object]:
        return {
            "event_key": event_key_to_mapping(self.event_key),
            "order_id": self.order_id,
            "price": self.price,
            "priority_time": self.priority_time,
            "quantity": self.quantity,
            "side": self.side.value if isinstance(self.side, Side) else self.side,
            "trader_id": self.trader_id,
        }


@dataclass(frozen=True, slots=True)
class OpeningState:
    """Gross accounts, fee pool, and optional pre-existing book."""

    accounts: tuple[AccountBalance, ...]
    venue_cash: int = 0
    orders: tuple[OpeningOrder, ...] = ()

    def to_mapping(self) -> dict[str, object]:
        return {
            "accounts": [
                account.to_mapping()
                for account in sorted(self.accounts, key=lambda item: item.trader_id)
            ],
            "orders": [
                order.to_mapping()
                for order in sorted(
                    self.orders,
                    key=lambda item: (
                        item.priority_time,
                        item.event_key.value,
                        item.order_id,
                    ),
                )
            ],
            "venue_cash": self.venue_cash,
        }


@dataclass(frozen=True, slots=True)
class LimitOrder:
    """One limit-order command submitted at an integer logical time."""

    event_key: EventKey
    submitted_at: int
    order_id: str
    trader_id: str
    side: Side
    quantity: int
    price: int

    def to_mapping(self) -> dict[str, object]:
        return {
            "command_type": "limit_order",
            "event_key": event_key_to_mapping(self.event_key),
            "order_id": self.order_id,
            "price": self.price,
            "quantity": self.quantity,
            "side": self.side.value if isinstance(self.side, Side) else self.side,
            "submitted_at": self.submitted_at,
            "trader_id": self.trader_id,
        }


@dataclass(frozen=True, slots=True)
class MarketOrder:
    """One immediate-or-cancel market-order command."""

    event_key: EventKey
    submitted_at: int
    order_id: str
    trader_id: str
    side: Side
    quantity: int

    def to_mapping(self) -> dict[str, object]:
        return {
            "command_type": "market_order",
            "event_key": event_key_to_mapping(self.event_key),
            "order_id": self.order_id,
            "quantity": self.quantity,
            "side": self.side.value if isinstance(self.side, Side) else self.side,
            "submitted_at": self.submitted_at,
            "trader_id": self.trader_id,
        }


@dataclass(frozen=True, slots=True)
class CancelOrder:
    """Cancellation of one active order by its owner."""

    event_key: EventKey
    submitted_at: int
    trader_id: str
    target_order_id: str

    def to_mapping(self) -> dict[str, object]:
        return {
            "command_type": "cancel_order",
            "event_key": event_key_to_mapping(self.event_key),
            "submitted_at": self.submitted_at,
            "target_order_id": self.target_order_id,
            "trader_id": self.trader_id,
        }


type MarketCommand = LimitOrder | MarketOrder | CancelOrder


def command_to_mapping(command: MarketCommand) -> dict[str, object]:
    """Return the canonical representation of a supported command."""
    return command.to_mapping()


@dataclass(frozen=True, slots=True)
class ScheduledCommand:
    """One validated command in its canonical deterministic execution order."""

    arrival_time: int
    event_key: str
    command: MarketCommand

    def to_mapping(self) -> dict[str, object]:
        return {
            "arrival_time": self.arrival_time,
            "command": command_to_mapping(self.command),
            "event_key": self.event_key,
        }


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    """Gross and reserved holdings after a deterministic transition."""

    trader_id: str
    cash: int
    asset: int
    reserved_cash: int
    reserved_asset: int

    @property
    def available_cash(self) -> int:
        return self.cash - self.reserved_cash

    @property
    def available_asset(self) -> int:
        return self.asset - self.reserved_asset

    def to_mapping(self) -> dict[str, object]:
        return {
            "asset": self.asset,
            "available_asset": self.available_asset,
            "available_cash": self.available_cash,
            "cash": self.cash,
            "reserved_asset": self.reserved_asset,
            "reserved_cash": self.reserved_cash,
            "trader_id": self.trader_id,
        }


@dataclass(frozen=True, slots=True)
class RestingOrderSnapshot:
    """One active price-time-prioritized limit order."""

    order_id: str
    trader_id: str
    side: Side
    price: int
    remaining_quantity: int
    arrival_time: int
    priority_source: int
    event_key: str
    reserved_cash: int
    reserved_asset: int

    def priority_key(self) -> tuple[int, int, str, str]:
        return (
            self.arrival_time,
            self.priority_source,
            self.event_key,
            self.order_id,
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "arrival_time": self.arrival_time,
            "event_key": self.event_key,
            "order_id": self.order_id,
            "price": self.price,
            "priority_source": self.priority_source,
            "remaining_quantity": self.remaining_quantity,
            "reserved_asset": self.reserved_asset,
            "reserved_cash": self.reserved_cash,
            "side": self.side.value,
            "trader_id": self.trader_id,
        }


@dataclass(frozen=True, slots=True)
class OrderRecordSnapshot:
    """Lifecycle record retained after fill, cancel, or expiry."""

    order_id: str
    trader_id: str
    side: Side
    order_type: str
    original_quantity: int
    filled_quantity: int
    remaining_quantity: int
    status: OrderStatus

    def to_mapping(self) -> dict[str, object]:
        return {
            "filled_quantity": self.filled_quantity,
            "order_id": self.order_id,
            "order_type": self.order_type,
            "original_quantity": self.original_quantity,
            "remaining_quantity": self.remaining_quantity,
            "side": self.side.value,
            "status": self.status.value,
            "trader_id": self.trader_id,
        }


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    """Immutable, canonical scientific state of the market."""

    venue_cash: int
    accounts: tuple[AccountSnapshot, ...]
    resting_orders: tuple[RestingOrderSnapshot, ...]
    order_records: tuple[OrderRecordSnapshot, ...]

    def to_mapping(self) -> dict[str, object]:
        return {
            "accounts": [account.to_mapping() for account in self.accounts],
            "order_records": [record.to_mapping() for record in self.order_records],
            "resting_orders": [order.to_mapping() for order in self.resting_orders],
            "schema_version": MARKET_SCHEMA_VERSION,
            "venue_cash": self.venue_cash,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_mapping())

    @property
    def sha256(self) -> str:
        return sha256_bytes(self.canonical_bytes)


@dataclass(frozen=True, slots=True)
class TradeLogEntry:
    """One deterministic fill at the resting order's price."""

    sequence: int
    trade_id: str
    simulation_time: int
    cause_event_key: str
    fill_index: int
    maker_order_id: str
    taker_order_id: str
    maker_trader_id: str
    taker_trader_id: str
    buyer_id: str
    seller_id: str
    aggressor_side: Side
    price: int
    quantity: int
    maker_fee: int
    taker_fee: int

    def to_mapping(self) -> dict[str, object]:
        return {
            "aggressor_side": self.aggressor_side.value,
            "buyer_id": self.buyer_id,
            "cause_event_key": self.cause_event_key,
            "fill_index": self.fill_index,
            "maker_fee": self.maker_fee,
            "maker_order_id": self.maker_order_id,
            "maker_trader_id": self.maker_trader_id,
            "price": self.price,
            "quantity": self.quantity,
            "seller_id": self.seller_id,
            "sequence": self.sequence,
            "simulation_time": self.simulation_time,
            "taker_fee": self.taker_fee,
            "taker_order_id": self.taker_order_id,
            "taker_trader_id": self.taker_trader_id,
            "trade_id": self.trade_id,
        }


@dataclass(frozen=True, slots=True)
class QuoteLogEntry:
    """Best-bid/offer snapshot after opening or an accepted command."""

    sequence: int
    simulation_time: int
    cause_event_key: str | None
    best_bid_price: int | None
    best_bid_quantity: int
    best_ask_price: int | None
    best_ask_quantity: int

    def to_mapping(self) -> dict[str, object]:
        return {
            "best_ask_price": self.best_ask_price,
            "best_ask_quantity": self.best_ask_quantity,
            "best_bid_price": self.best_bid_price,
            "best_bid_quantity": self.best_bid_quantity,
            "cause_event_key": self.cause_event_key,
            "sequence": self.sequence,
            "simulation_time": self.simulation_time,
        }


@dataclass(frozen=True, slots=True)
class MarketAuditEvent:
    """One hash-chained, wall-clock-free scientific audit decision."""

    sequence: int
    action: str
    outcome: str
    rule_code: MarketRuleCode
    simulation_time: int | None
    event_key: str | None
    details: tuple[tuple[str, AuditScalar], ...]
    previous_event_sha256: str
    event_sha256: str

    def core_mapping(self) -> dict[str, object]:
        return {
            "action": self.action,
            "details": dict(self.details),
            "event_key": self.event_key,
            "event_type": "market_audit",
            "outcome": self.outcome,
            "previous_event_sha256": self.previous_event_sha256,
            "rule_code": self.rule_code.value,
            "schema_version": MARKET_SCHEMA_VERSION,
            "sequence": self.sequence,
            "simulation_time": self.simulation_time,
        }

    def to_mapping(self) -> dict[str, object]:
        return {**self.core_mapping(), "event_sha256": self.event_sha256}


class MarketAuditLog:
    """Thread-safe deterministic audit ledger with optional synchronous JSONL."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path.expanduser().resolve() if path is not None else None
        self._events: list[MarketAuditEvent] = []
        self._lock = RLock()
        self._file_identity: tuple[int, int] | None = None
        self._persisted_size = 0
        self._persisted_sha256 = sha256_bytes(b"")
        self._compromised = False
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            try:
                self._path.open("xb").close()
                file_stat = os.lstat(self._path)
            except OSError as error:
                self._compromised = True
                self._append_persistence_failure(
                    attempted_action="initialize_audit_log",
                    attempted_outcome="allowed",
                    simulation_time=None,
                    event_key=None,
                    message="market audit log creation failed",
                    tail_rollback_succeeded=True,
                )
                failure = MarketAuditError(f"cannot create market audit log: {error}")
                failure.audit_events = tuple(self._events)
                failure.audit_tail_rolled_back = True
                raise failure from error
            self._file_identity = (file_stat.st_dev, file_stat.st_ino)

    @property
    def path(self) -> Path | None:
        return self._path

    @property
    def events(self) -> tuple[MarketAuditEvent, ...]:
        with self._lock:
            return tuple(self._events)

    @property
    def compromised(self) -> bool:
        """Return whether required persistence failed or the file was altered."""
        with self._lock:
            return self._compromised

    def assert_integrity(self) -> None:
        """Fail closed if the in-memory chain or persisted prefix changed."""
        with self._lock:
            if self._compromised:
                failure = MarketAuditError("market audit log is compromised")
                failure.audit_events = tuple(self._events)
                failure.audit_tail_rolled_back = False
                raise failure
            try:
                previous = "0" * 64
                for sequence, event in enumerate(self._events):
                    expected = sha256_bytes(canonical_json_bytes(event.core_mapping()))
                    if (
                        event.sequence != sequence
                        or event.previous_event_sha256 != previous
                        or event.event_sha256 != expected
                    ):
                        raise OSError("in-memory audit hash chain is invalid")
                    previous = event.event_sha256
                if self._path is not None:
                    path_stat = os.lstat(self._path)
                    if (
                        not stat.S_ISREG(path_stat.st_mode)
                        or self._file_identity != (path_stat.st_dev, path_stat.st_ino)
                        or path_stat.st_size != self._persisted_size
                        or sha256_file(self._path) != self._persisted_sha256
                    ):
                        raise OSError("persisted audit prefix is invalid")
            except OSError as error:
                self._compromised = True
                self._append_persistence_failure(
                    attempted_action="verify_audit_log",
                    attempted_outcome="allowed",
                    simulation_time=None,
                    event_key=None,
                    message="market audit integrity verification failed",
                    tail_rollback_succeeded=True,
                )
                failure = MarketAuditError(str(error))
                failure.audit_events = tuple(self._events)
                failure.audit_tail_rolled_back = True
                raise failure from error

    def append(
        self,
        *,
        action: str,
        outcome: str,
        rule_code: MarketRuleCode,
        simulation_time: int | None,
        event_key: str | None,
        details: Mapping[str, AuditScalar],
    ) -> MarketAuditEvent:
        """Append and optionally fsync one canonical audit event."""
        with self._lock:
            if self._compromised:
                self._raise_compromised_append(
                    attempted_action=action,
                    attempted_outcome=outcome,
                    simulation_time=simulation_time,
                    event_key=event_key,
                )
            event = self._create_event(
                action=action,
                outcome=outcome,
                rule_code=rule_code,
                simulation_time=simulation_time,
                event_key=event_key,
                details=details,
            )
            if self._path is not None:
                rendered = canonical_json_bytes(event.to_mapping()) + b"\n"
                write_started = False
                try:
                    path_stat = os.lstat(self._path)
                    if (
                        not stat.S_ISREG(path_stat.st_mode)
                        or self._file_identity != (path_stat.st_dev, path_stat.st_ino)
                        or path_stat.st_size != self._persisted_size
                        or sha256_file(self._path) != self._persisted_sha256
                    ):
                        raise OSError(
                            "audit log identity, size, or content changed externally"
                        )
                    with self._path.open("r+b") as stream:
                        stream_stat = os.fstat(stream.fileno())
                        if self._file_identity != (
                            stream_stat.st_dev,
                            stream_stat.st_ino,
                        ):
                            raise OSError("audit log was replaced while opening")
                        stream.seek(0, os.SEEK_END)
                        write_started = True
                        stream.write(rendered)
                        stream.flush()
                        os.fsync(stream.fileno())
                    persisted_sha256 = sha256_file(self._path)
                except OSError as error:
                    rolled_back = (
                        self._truncate_failed_append() if write_started else True
                    )
                    self._compromised = True
                    self._append_persistence_failure(
                        attempted_action=action,
                        attempted_outcome=outcome,
                        simulation_time=simulation_time,
                        event_key=event_key,
                        message="mandatory market audit persistence failed",
                        tail_rollback_succeeded=rolled_back,
                    )
                    failure = MarketAuditError(
                        f"cannot persist mandatory market audit event: {error}"
                    )
                    failure.audit_events = tuple(self._events)
                    failure.audit_tail_rolled_back = rolled_back
                    raise failure from error
                self._persisted_size += len(rendered)
                self._persisted_sha256 = persisted_sha256
            self._events.append(event)
            return event

    def _create_event(
        self,
        *,
        action: str,
        outcome: str,
        rule_code: MarketRuleCode,
        simulation_time: int | None,
        event_key: str | None,
        details: Mapping[str, AuditScalar],
    ) -> MarketAuditEvent:
        normalized_details = tuple(sorted(details.items()))
        previous = self._events[-1].event_sha256 if self._events else "0" * 64
        provisional = MarketAuditEvent(
            sequence=len(self._events),
            action=action,
            outcome=outcome,
            rule_code=rule_code,
            simulation_time=simulation_time,
            event_key=event_key,
            details=normalized_details,
            previous_event_sha256=previous,
            event_sha256="0" * 64,
        )
        return MarketAuditEvent(
            sequence=provisional.sequence,
            action=provisional.action,
            outcome=provisional.outcome,
            rule_code=provisional.rule_code,
            simulation_time=provisional.simulation_time,
            event_key=provisional.event_key,
            details=provisional.details,
            previous_event_sha256=provisional.previous_event_sha256,
            event_sha256=sha256_bytes(canonical_json_bytes(provisional.core_mapping())),
        )

    def _append_persistence_failure(
        self,
        *,
        attempted_action: str,
        attempted_outcome: str,
        simulation_time: int | None,
        event_key: str | None,
        message: str,
        tail_rollback_succeeded: bool,
    ) -> None:
        self._events.append(
            self._create_event(
                action="audit_append",
                outcome="rejected",
                rule_code=MarketRuleCode.AUDIT_WRITE_FAILED,
                simulation_time=simulation_time,
                event_key=event_key,
                details={
                    "attempted_action": attempted_action,
                    "attempted_outcome": attempted_outcome,
                    "message": message,
                    "persisted": False,
                    "tail_rollback_succeeded": tail_rollback_succeeded,
                },
            )
        )

    def _raise_compromised_append(
        self,
        *,
        attempted_action: str,
        attempted_outcome: str,
        simulation_time: int | None,
        event_key: str | None,
    ) -> NoReturn:
        self._append_persistence_failure(
            attempted_action=attempted_action,
            attempted_outcome=attempted_outcome,
            simulation_time=simulation_time,
            event_key=event_key,
            message="market audit log is already compromised",
            tail_rollback_succeeded=True,
        )
        failure = MarketAuditError("market audit log is compromised")
        failure.audit_events = tuple(self._events)
        failure.audit_tail_rolled_back = True
        raise failure

    def _truncate_failed_append(self) -> bool:
        if self._path is None or self._file_identity is None:
            return False
        try:
            path_stat = os.lstat(self._path)
            if (
                self._file_identity != (path_stat.st_dev, path_stat.st_ino)
                or path_stat.st_size < self._persisted_size
            ):
                return False
            with self._path.open("r+b") as stream:
                stream.truncate(self._persisted_size)
                stream.flush()
                with suppress(OSError):
                    os.fsync(stream.fileno())
            return (
                os.lstat(self._path).st_size == self._persisted_size
                and sha256_file(self._path) == self._persisted_sha256
            )
        except OSError:
            return False


@dataclass(frozen=True, slots=True)
class ReplayResult:
    """Complete deterministic replay artifact for one sealed command batch."""

    config: MarketConfig
    opening_state: OpeningState
    scheduled_commands: tuple[ScheduledCommand, ...]
    config_sha256: str
    opening_state_sha256: str
    command_stream_sha256: str
    event_keys: tuple[str, ...]
    state: MarketSnapshot
    trades: tuple[TradeLogEntry, ...]
    quotes: tuple[QuoteLogEntry, ...]
    audit_events: tuple[MarketAuditEvent, ...]
    policy_gateway_config: PolicyGatewayConfig | None
    policy_gateway_sha256: str | None
    policy_replay_descriptor: SyntheticPolicyReplayDescriptor | None
    policy_audit_events: tuple[PolicyComplianceAuditEvent, ...]

    def to_mapping(self) -> dict[str, object]:
        return {
            "audit_events": [event.to_mapping() for event in self.audit_events],
            "command_stream_sha256": self.command_stream_sha256,
            "config": self.config.to_mapping(),
            "config_sha256": self.config_sha256,
            "event_keys": list(self.event_keys),
            "opening_state": self.opening_state.to_mapping(),
            "opening_state_sha256": self.opening_state_sha256,
            "policy_audit_events": [
                event.to_mapping() for event in self.policy_audit_events
            ],
            "policy_gateway_config": (
                None
                if self.policy_gateway_config is None
                else self.policy_gateway_config.to_mapping()
            ),
            "policy_gateway_sha256": self.policy_gateway_sha256,
            "policy_replay_descriptor": (
                None
                if self.policy_replay_descriptor is None
                else self.policy_replay_descriptor.to_mapping()
            ),
            "policy_replay_descriptor_sha256": (
                None
                if self.policy_replay_descriptor is None
                else self.policy_replay_descriptor.sha256
            ),
            "quotes": [quote.to_mapping() for quote in self.quotes],
            "schema_version": MARKET_SCHEMA_VERSION,
            "scheduled_commands": [
                scheduled.to_mapping() for scheduled in self.scheduled_commands
            ],
            "state": self.state.to_mapping(),
            "state_sha256": self.state.sha256,
            "trades": [trade.to_mapping() for trade in self.trades],
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_mapping())

    @property
    def sha256(self) -> str:
        return sha256_bytes(self.canonical_bytes)

    def write_logs(self, destination: Path) -> None:
        """Write deterministic trade, quote, and audit JSONL artifacts."""
        destination.mkdir(parents=True, exist_ok=False)
        logs: tuple[tuple[str, str, tuple[Mapping[str, object], ...]], ...] = (
            (
                "commands.jsonl",
                "scheduled_commands",
                tuple(command.to_mapping() for command in self.scheduled_commands),
            ),
            (
                "trades.jsonl",
                "trades",
                tuple(trade.to_mapping() for trade in self.trades),
            ),
            (
                "quotes.jsonl",
                "quotes",
                tuple(quote.to_mapping() for quote in self.quotes),
            ),
            (
                "market_audit.jsonl",
                "market_audit",
                tuple(event.to_mapping() for event in self.audit_events),
            ),
            (
                "policy_audit.jsonl",
                "policy_audit",
                tuple(event.to_mapping() for event in self.policy_audit_events),
            ),
        )
        for filename, log_kind, records in logs:
            with JsonlEventWriter(destination / filename) as writer:
                writer.write(
                    {
                        "log_kind": log_kind,
                        "record_type": "header",
                        "schema_version": MARKET_SCHEMA_VERSION,
                    }
                )
                for record in records:
                    writer.write(record)

    def write_replay_bundle(self, destination: Path) -> Path:
        """Persist a strict synthetic policy reconstruction bundle."""
        from policytwin.market.policy_replay import (
            write_synthetic_policy_replay_bundle,
        )

        return write_synthetic_policy_replay_bundle(self, destination)


def is_stable_identifier(value: object) -> bool:
    """Return whether a value is a portable, deterministic identifier."""
    return isinstance(value, str) and _IDENTIFIER_PATTERN.fullmatch(value) is not None


__all__ = [
    "AccountBalance",
    "AccountSnapshot",
    "CancelOrder",
    "ConservationViolation",
    "EngineLatchedError",
    "FEE_DENOMINATOR_BPS",
    "LimitOrder",
    "MARKET_SCHEMA_VERSION",
    "MarketAuditError",
    "MarketAuditEvent",
    "MarketAuditLog",
    "MarketCommand",
    "MarketConfig",
    "MarketError",
    "MarketInvariantError",
    "MarketOrder",
    "MarketPolicyMode",
    "MarketRuleCode",
    "MarketSnapshot",
    "MarketStateError",
    "MarketValidationError",
    "OpeningOrder",
    "OpeningState",
    "OrderRecordSnapshot",
    "OrderStatus",
    "QuoteLogEntry",
    "ReplayResult",
    "RestingOrderSnapshot",
    "SCHEDULER_SCHEMA_VERSION",
    "ScheduledCommand",
    "Side",
    "TradeLogEntry",
    "command_to_mapping",
    "event_key_to_mapping",
    "is_stable_identifier",
]
