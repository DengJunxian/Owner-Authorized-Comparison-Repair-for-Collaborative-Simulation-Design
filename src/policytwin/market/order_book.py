"""Correctness-first deterministic discrete-event central limit order book."""

from __future__ import annotations

import copy
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import NoReturn, cast

from policytwin.counterfactual import EVENT_KEY_SCHEMA_VERSION, EventKey
from policytwin.market.models import (
    FEE_DENOMINATOR_BPS,
    MARKET_SCHEMA_VERSION,
    SCHEDULER_SCHEMA_VERSION,
    AccountBalance,
    AccountSnapshot,
    AuditScalar,
    CancelOrder,
    ConservationViolation,
    EngineLatchedError,
    LimitOrder,
    MarketAuditError,
    MarketAuditEvent,
    MarketAuditLog,
    MarketCommand,
    MarketConfig,
    MarketError,
    MarketInvariantError,
    MarketOrder,
    MarketPolicyMode,
    MarketRuleCode,
    MarketSnapshot,
    MarketStateError,
    MarketValidationError,
    OpeningOrder,
    OpeningState,
    OrderRecordSnapshot,
    OrderStatus,
    QuoteLogEntry,
    ReplayResult,
    RestingOrderSnapshot,
    ScheduledCommand,
    Side,
    TradeLogEntry,
    command_to_mapping,
    is_stable_identifier,
)
from policytwin.market.policy_gateway import (
    GatewayComplianceRecord,
    PolicyGatewayConfig,
    PolicyGatewayError,
    SecTickSizePolicyGateway,
)
from policytwin.market.policy_replay import (
    SyntheticPolicyReplayDescriptor,
    load_synthetic_policy_replay_bundle,
    verify_loaded_replay_result,
)
from policytwin.policies.sec_tick_size_pilot import (
    ComplianceOutcome,
    PolicyComplianceAuditEvent,
)
from policytwin.provenance import canonical_json_bytes, sha256_bytes


@dataclass(slots=True)
class _AccountLedger:
    trader_id: str
    cash: int
    asset: int
    reserved_cash: int = 0
    reserved_asset: int = 0

    @property
    def available_cash(self) -> int:
        return self.cash - self.reserved_cash

    @property
    def available_asset(self) -> int:
        return self.asset - self.reserved_asset


@dataclass(slots=True)
class _WorkingOrder:
    order_id: str
    trader_id: str
    side: Side
    original_quantity: int
    remaining_quantity: int
    price: int | None
    arrival_time: int
    priority_source: int
    event_key: str
    order_type: str
    reserved_cash: int = 0
    reserved_asset: int = 0

    def priority_key(self) -> tuple[int, int, str, str]:
        return (
            self.arrival_time,
            self.priority_source,
            self.event_key,
            self.order_id,
        )


@dataclass(slots=True)
class _OrderHistory:
    order_id: str
    trader_id: str
    side: Side
    order_type: str
    original_quantity: int
    filled_quantity: int
    remaining_quantity: int
    status: OrderStatus


@dataclass(frozen=True, slots=True)
class _PlannedFill:
    maker_order_id: str
    quantity: int


class _RuleFailure(Exception):
    def __init__(
        self,
        exception_type: type[MarketError],
        rule_code: MarketRuleCode,
        message: str,
        details: Mapping[str, AuditScalar] | None = None,
    ) -> None:
        super().__init__(message)
        self.exception_type = exception_type
        self.rule_code = rule_code
        self.message = message
        self.details = dict(details or {})


def _rule_fail(
    exception_type: type[MarketError],
    rule_code: MarketRuleCode,
    message: str,
    details: Mapping[str, AuditScalar] | None = None,
) -> NoReturn:
    raise _RuleFailure(exception_type, rule_code, message, details)


type _MutableBackup = tuple[
    dict[str, _AccountLedger],
    dict[str, _WorkingOrder],
    dict[str, _OrderHistory],
    int,
    list[TradeLogEntry],
    list[QuoteLogEntry],
    set[str],
]


class CentralLimitOrderBook:
    """Single-market deterministic batch scheduler and matching engine.

    ``schedule`` is safe for concurrent producers but only buffers commands.
    ``run`` seals the batch, sorts it by logical arrival time and semantic event
    key, and executes it on one thread. No thread completion order enters the
    scientific state.
    """

    def __init__(
        self,
        config: MarketConfig,
        opening_state: OpeningState,
        *,
        audit_log: MarketAuditLog | None = None,
        policy_gateway: SecTickSizePolicyGateway | None = None,
    ) -> None:
        self._config = config
        self._opening_state = opening_state
        self._policy_gateway = policy_gateway
        if audit_log is not None and type(audit_log) is not MarketAuditLog:
            raise MarketAuditError("audit_log must be an unmodified MarketAuditLog")
        self._audit = MarketAuditLog() if audit_log is None else audit_log
        if self._audit.events:
            raise MarketAuditError(
                "market engine requires an empty, exclusive audit log"
            )
        self._accounts: dict[str, _AccountLedger] = {}
        self._orders: dict[str, _WorkingOrder] = {}
        self._history: dict[str, _OrderHistory] = {}
        self._venue_cash = 0
        self._baseline_cash = 0
        self._baseline_asset = 0
        self._opening_state_sha256 = ""
        self._seen_event_keys: set[str] = set()
        self._scheduled: list[MarketCommand] = []
        self._schedule_lock = RLock()
        self._sealed = False
        self._failed = False
        self._failure_message: str | None = None
        self._trades: list[TradeLogEntry] = []
        self._quotes: list[QuoteLogEntry] = []
        self._result: ReplayResult | None = None
        self._opening_policy_records: list[GatewayComplianceRecord] = []
        try:
            self._validate_config()
            self._initialize_opening_state()
            self._assert_internal_invariants()
            self._append_quote(simulation_time=0, cause_event_key=None)
        except _RuleFailure as failure:
            self._failed = True
            self._failure_message = failure.message
            state_hash = self._safe_state_hash()
            self._audit.append(
                action="initialize_market",
                outcome="rejected",
                rule_code=failure.rule_code,
                simulation_time=None,
                event_key=cast(str | None, failure.details.get("event_key")),
                details={
                    "after_state_sha256": state_hash,
                    "before_state_sha256": state_hash,
                    "message": failure.message,
                    **failure.details,
                },
            )
            error = failure.exception_type(failure.message)
            error.audit_events = self._audit.events
            error.policy_audit_events = self._policy_events()
            raise error from failure
        except MarketAuditError:
            raise
        except Exception as unexpected:
            self._failed = True
            self._failure_message = "unexpected market initialization failure"
            state_hash = self._safe_state_hash()
            self._audit.append(
                action="initialize_market",
                outcome="rejected",
                rule_code=MarketRuleCode.INTERNAL_ERROR,
                simulation_time=None,
                event_key=None,
                details={
                    "after_state_sha256": state_hash,
                    "before_state_sha256": state_hash,
                    "exception_type": type(unexpected).__qualname__,
                    "message": self._failure_message,
                },
            )
            error = MarketInvariantError(self._failure_message)
            error.audit_events = self._audit.events
            error.policy_audit_events = self._policy_events()
            raise error from unexpected
        state_hash = self._state_hash()
        self._audit.append(
            action="initialize_market",
            outcome="allowed",
            rule_code=MarketRuleCode.OK,
            simulation_time=0,
            event_key=None,
            details={
                "after_state_sha256": state_hash,
                "before_state_sha256": state_hash,
                "config_sha256": self._config.sha256,
                "opening_order_count": len(self._orders),
                **self._policy_audit_details(self._opening_policy_records),
            },
        )

    @property
    def config(self) -> MarketConfig:
        return self._config

    @property
    def audit_log(self) -> MarketAuditLog:
        return self._audit

    @property
    def audit_events(self) -> tuple[MarketAuditEvent, ...]:
        return self._audit.events

    @property
    def trades(self) -> tuple[TradeLogEntry, ...]:
        return tuple(self._trades)

    @property
    def quotes(self) -> tuple[QuoteLogEntry, ...]:
        return tuple(self._quotes)

    @property
    def failed(self) -> bool:
        return self._failed

    @property
    def snapshot(self) -> MarketSnapshot:
        return self._build_snapshot()

    def schedule(self, command: MarketCommand) -> None:
        """Buffer a command without assigning priority from call order."""
        with self._schedule_lock:
            self._ensure_audit_healthy()
            if self._failed:
                self._record_latched_attempt("schedule", command)
            if self._sealed:
                self._reject_sealed_schedule("schedule", command)
            self._scheduled.append(command)

    def schedule_all(self, commands: Iterable[MarketCommand]) -> None:
        """Atomically buffer a finite command collection."""
        with self._schedule_lock:
            self._ensure_audit_healthy()
        try:
            materialized = tuple(commands)
        except Exception as error:
            with self._schedule_lock:
                if self._failed:
                    self._record_latched_attempt("schedule_all", None)
                self._failed = True
                self._failure_message = "command iterable failed while materializing"
                self._audit.append(
                    action="schedule_all",
                    outcome="rejected",
                    rule_code=MarketRuleCode.INTERNAL_ERROR,
                    simulation_time=None,
                    event_key=None,
                    details={
                        "exception_type": type(error).__qualname__,
                        "message": self._failure_message,
                    },
                )
                failure = MarketValidationError(self._failure_message)
                failure.audit_events = self._audit.events
                raise failure from error
        with self._schedule_lock:
            self._ensure_audit_healthy()
            if self._failed:
                self._record_latched_attempt("schedule_all", None)
            if self._sealed:
                self._reject_sealed_schedule("schedule_all", None)
            self._scheduled.extend(materialized)

    def run(self) -> ReplayResult:
        """Seal, deterministically order, execute, and return one replay result."""
        with self._schedule_lock:
            self._ensure_audit_healthy()
            try:
                self._assert_policy_gateway_runtime()
            except _RuleFailure as failure:
                self._reject_preflight(failure)
            if self._result is not None and not self._failed:
                return self._result
            if self._failed:
                self._record_latched_attempt("run", None)
            if self._sealed:
                self._record_latched_attempt("run", None)
            self._sealed = True
            commands = tuple(self._scheduled)
            scheduled = self._prepare_schedule(commands)
            scheduled_commands = tuple(
                ScheduledCommand(
                    arrival_time=arrival_time,
                    event_key=event_key,
                    command=command,
                )
                for arrival_time, event_key, command in scheduled
            )
            command_stream_sha256 = sha256_bytes(
                canonical_json_bytes(
                    [scheduled.to_mapping() for scheduled in scheduled_commands]
                )
            )
            for arrival_time, _, command in scheduled:
                self._process_command(command, arrival_time)
            policy_replay_descriptor = (
                None
                if self._policy_gateway is None
                else SyntheticPolicyReplayDescriptor.capture(self._policy_gateway)
            )
            result = ReplayResult(
                config=self._config,
                opening_state=self._opening_state,
                scheduled_commands=scheduled_commands,
                config_sha256=self._config.sha256,
                opening_state_sha256=self._opening_state_sha256,
                command_stream_sha256=command_stream_sha256,
                event_keys=tuple(event_key for _, event_key, _ in scheduled),
                state=self._build_snapshot(),
                trades=tuple(self._trades),
                quotes=tuple(self._quotes),
                audit_events=self._audit.events,
                policy_gateway_config=(
                    None
                    if self._policy_gateway is None
                    else self._policy_gateway.config
                ),
                policy_gateway_sha256=(
                    None
                    if self._policy_gateway is None
                    else self._policy_gateway.sha256
                ),
                policy_replay_descriptor=policy_replay_descriptor,
                policy_audit_events=(
                    ()
                    if self._policy_gateway is None
                    else self._policy_gateway.policy_audit_events
                ),
            )
            self._result = result
            return result

    @classmethod
    def replay(cls, artifact: ReplayResult) -> ReplayResult:
        """Re-execute a complete in-memory replay artifact and verify exact bytes."""
        engine: CentralLimitOrderBook | None = None
        replayed: ReplayResult | None = None
        try:
            if not isinstance(artifact, ReplayResult):
                raise TypeError("artifact has the wrong runtime type")
            if (artifact.policy_gateway_config is None) != (
                artifact.policy_gateway_sha256 is None
            ):
                raise ValueError("policy gateway config and hash presence differ")
            if (artifact.policy_gateway_sha256 is None) != (
                artifact.policy_replay_descriptor is None
            ):
                raise ValueError(
                    "policy gateway and reconstruction descriptor presence differ"
                )
            if artifact.policy_gateway_config is not None and (
                type(artifact.policy_gateway_config) is not PolicyGatewayConfig
                or artifact.policy_gateway_config.sha256
                != artifact.policy_gateway_sha256
            ):
                raise ValueError("policy gateway config differs from its commitment")
            descriptor = artifact.policy_replay_descriptor
            if descriptor is None:
                engine = cls(artifact.config, artifact.opening_state)
                engine.schedule_all(
                    scheduled.command for scheduled in artifact.scheduled_commands
                )
                replayed = engine.run()
            else:
                if (
                    type(descriptor) is not SyntheticPolicyReplayDescriptor
                    or descriptor.gateway_config != artifact.policy_gateway_config
                    or descriptor.gateway_config.sha256
                    != artifact.policy_gateway_sha256
                ):
                    raise ValueError(
                        "policy reconstruction descriptor differs from the gateway"
                    )
                with tempfile.TemporaryDirectory(
                    prefix="policytwin-policy-replay-",
                    dir=Path(tempfile.gettempdir()).resolve(),
                ) as temporary:
                    assignment_path = Path(temporary) / "assignments.json"
                    assignment_path.write_bytes(descriptor.assignment_source_bytes)
                    gateway = descriptor.reconstruct(assignment_path)
                    engine = cls(
                        artifact.config,
                        artifact.opening_state,
                        policy_gateway=gateway,
                    )
                    engine.schedule_all(
                        scheduled.command for scheduled in artifact.scheduled_commands
                    )
                    replayed = engine.run()
            if replayed.canonical_bytes == artifact.canonical_bytes:
                return replayed
            raise ValueError("canonical replay bytes differ")
        except Exception as unexpected:
            audit = MarketAuditLog() if engine is None else engine._audit
            message = "replayed artifact differs from its commitment"
            expected_sha256 = "unavailable"
            if isinstance(artifact, ReplayResult):
                with suppress(Exception):
                    expected_sha256 = artifact.sha256
            actual_sha256 = replayed.sha256 if replayed is not None else "unavailable"
            if engine is not None:
                engine._failed = True
                engine._failure_message = message
            audit.append(
                action="replay",
                outcome="rejected",
                rule_code=(MarketRuleCode.REPLAY_MISMATCH),
                simulation_time=None,
                event_key=None,
                details={
                    "actual_sha256": actual_sha256,
                    "exception_type": type(unexpected).__qualname__,
                    "expected_sha256": expected_sha256,
                    "message": message,
                },
            )
            failure = MarketInvariantError(message)
            failure.audit_events = audit.events
            failure.policy_audit_events = (
                artifact.policy_audit_events
                if isinstance(artifact, ReplayResult)
                else ()
            )
            raise failure from unexpected

    @classmethod
    def replay_bundle(cls, directory: Path) -> ReplayResult:
        """Load, reconstruct, execute, and verify one synthetic policy bundle."""
        engine: CentralLimitOrderBook | None = None
        try:
            loaded = load_synthetic_policy_replay_bundle(directory)
            gateway = loaded.reconstruct_gateway()
            engine = cls(
                loaded.config,
                loaded.opening_state,
                policy_gateway=gateway,
            )
            engine.schedule_all(
                scheduled.command for scheduled in loaded.scheduled_commands
            )
            result = engine.run()
            verify_loaded_replay_result(loaded, result)
            return result
        except Exception as unexpected:
            audit = MarketAuditLog() if engine is None else engine._audit
            audit.append(
                action="replay_bundle",
                outcome="rejected",
                rule_code=MarketRuleCode.REPLAY_MISMATCH,
                simulation_time=None,
                event_key=None,
                details={
                    "exception_type": type(unexpected).__qualname__,
                    "message": "synthetic policy replay bundle was rejected",
                },
            )
            failure = MarketInvariantError(
                "synthetic policy replay bundle was rejected"
            )
            failure.audit_events = audit.events
            failure.policy_audit_events = (
                ()
                if engine is None or engine._policy_gateway is None
                else engine._policy_gateway.policy_audit_events
            )
            raise failure from unexpected

    def assert_invariants(self) -> None:
        """Audit and latch any invariant failure in the current live state."""
        self._ensure_audit_healthy()
        if self._failed:
            self._record_latched_attempt("assert_invariants", None)
        try:
            self._assert_internal_invariants()
        except _RuleFailure as failure:
            self._latch_external_invariant_failure(
                action="assert_invariants",
                snapshot=self._build_snapshot(),
                failure=failure,
            )
        except Exception as unexpected:
            self._latch_external_invariant_failure(
                action="assert_invariants",
                snapshot=self._build_snapshot(),
                failure=_RuleFailure(
                    MarketInvariantError,
                    MarketRuleCode.INTERNAL_ERROR,
                    "unexpected invariant validation failure",
                    {"exception_type": type(unexpected).__qualname__},
                ),
            )

    def verify_snapshot(self, snapshot: MarketSnapshot) -> None:
        """Validate a candidate and require its exact current-state commitment."""
        self._ensure_audit_healthy()
        if self._failed:
            self._record_latched_attempt("verify_snapshot", None)
        try:
            self._validate_snapshot(snapshot)
            if snapshot.sha256 != self._state_hash():
                _rule_fail(
                    MarketInvariantError,
                    MarketRuleCode.STATE_COMMITMENT_MISMATCH,
                    "candidate snapshot differs from the committed market state",
                )
        except _RuleFailure as failure:
            self._latch_external_invariant_failure(
                action="verify_snapshot", snapshot=snapshot, failure=failure
            )
        except Exception as unexpected:
            self._latch_external_invariant_failure(
                action="verify_snapshot",
                snapshot=snapshot,
                failure=_RuleFailure(
                    MarketInvariantError,
                    MarketRuleCode.INTERNAL_ERROR,
                    "unexpected candidate snapshot validation failure",
                    {"exception_type": type(unexpected).__qualname__},
                ),
            )

    def order_record(self, order_id: str) -> OrderRecordSnapshot | None:
        """Return one immutable lifecycle record, if the ID has been accepted."""
        history = self._history.get(order_id)
        return self._history_snapshot(history) if history is not None else None

    def _validate_config(self) -> None:
        config = self._config
        if not isinstance(config, MarketConfig):
            _rule_fail(
                MarketValidationError,
                MarketRuleCode.INVALID_CONFIG,
                "config must be a MarketConfig",
            )
        if (
            config.schema_version != MARKET_SCHEMA_VERSION
            or config.scheduler_schema_version != SCHEDULER_SCHEMA_VERSION
        ):
            _rule_fail(
                MarketValidationError,
                MarketRuleCode.INVALID_CONFIG,
                "market or scheduler schema version is unsupported",
            )
        for field, value, minimum, maximum in (
            ("tick_size", config.tick_size, 1, 10**12),
            ("latency_steps", config.latency_steps, 0, 10**12),
            ("maker_fee_bps", config.maker_fee_bps, 0, FEE_DENOMINATOR_BPS),
            ("taker_fee_bps", config.taker_fee_bps, 0, FEE_DENOMINATOR_BPS),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not minimum <= value <= maximum
            ):
                _rule_fail(
                    MarketValidationError,
                    MarketRuleCode.INVALID_CONFIG,
                    f"{field} must be an integer between {minimum} and {maximum}",
                )
        if not isinstance(config.policy_mode, MarketPolicyMode):
            _rule_fail(
                MarketValidationError,
                MarketRuleCode.INVALID_CONFIG,
                "policy_mode must be a MarketPolicyMode",
            )
        if config.policy_mode is MarketPolicyMode.POLICY_REQUIRED:
            if type(self._policy_gateway) is not SecTickSizePolicyGateway:
                _rule_fail(
                    MarketValidationError,
                    MarketRuleCode.POLICY_GATEWAY_REQUIRED,
                    "policy-required market needs an exclusive compliance gateway",
                )
            try:
                self._policy_gateway.assert_integrity()
                self._policy_gateway.claim_for_market()
            except PolicyGatewayError as error:
                _rule_fail(
                    MarketValidationError,
                    MarketRuleCode.POLICY_GATEWAY_MISMATCH,
                    "policy gateway commitment or evidence is invalid",
                    {"exception_type": type(error).__qualname__},
                )
        elif self._policy_gateway is not None:
            _rule_fail(
                MarketValidationError,
                MarketRuleCode.POLICY_GATEWAY_MISMATCH,
                "no-policy synthetic market cannot accept a policy gateway",
            )

    def _initialize_opening_state(self) -> None:
        opening = self._opening_state
        if not isinstance(opening, OpeningState):
            _rule_fail(
                MarketValidationError,
                MarketRuleCode.INVALID_OPENING_STATE,
                "opening_state must be an OpeningState",
            )
        if not isinstance(opening.accounts, tuple) or not opening.accounts:
            _rule_fail(
                MarketValidationError,
                MarketRuleCode.INVALID_OPENING_STATE,
                "opening accounts must be a non-empty immutable tuple",
            )
        accounts: dict[str, _AccountLedger] = {}
        for account in opening.accounts:
            self._validate_opening_account(account)
            if account.trader_id in accounts:
                _rule_fail(
                    MarketValidationError,
                    MarketRuleCode.INVALID_OPENING_STATE,
                    "opening account IDs must be unique",
                    {"trader_id": account.trader_id},
                )
            accounts[account.trader_id] = _AccountLedger(
                trader_id=account.trader_id,
                cash=account.cash,
                asset=account.asset,
            )
        if (
            isinstance(opening.venue_cash, bool)
            or not isinstance(opening.venue_cash, int)
            or opening.venue_cash < 0
        ):
            _rule_fail(
                MarketValidationError,
                MarketRuleCode.INVALID_OPENING_STATE,
                "opening venue_cash must be a non-negative integer",
            )
        if not isinstance(opening.orders, tuple):
            _rule_fail(
                MarketValidationError,
                MarketRuleCode.INVALID_OPENING_STATE,
                "opening orders must be an immutable tuple",
            )
        self._accounts = accounts
        opening_order_ids: set[str] = set()
        opening_event_keys: set[str] = set()
        required_cash: Counter[str] = Counter()
        required_asset: Counter[str] = Counter()
        bids: list[int] = []
        asks: list[int] = []
        for opening_order in opening.orders:
            self._validate_opening_order_schema(opening_order)
            event_key = opening_order.event_key.value
            if opening_order.order_id in opening_order_ids:
                _rule_fail(
                    MarketValidationError,
                    MarketRuleCode.DUPLICATE_ORDER_ID,
                    "opening order IDs must be unique",
                    {"event_key": event_key, "order_id": opening_order.order_id},
                )
            if event_key in opening_event_keys:
                _rule_fail(
                    MarketValidationError,
                    MarketRuleCode.DUPLICATE_EVENT_KEY,
                    "opening event keys must be unique",
                    {"event_key": event_key},
                )
            opening_order_ids.add(opening_order.order_id)
            opening_event_keys.add(event_key)
            self._evaluate_policy_quote(
                cause_event_key=event_key,
                logical_time=opening_order.priority_time,
                price_units=opening_order.price,
                records=self._opening_policy_records,
            )
            if opening_order.side is Side.BUY:
                required_cash[opening_order.trader_id] += self._buy_reserve(
                    opening_order.price,
                    opening_order.quantity,
                    self._config.maker_fee_bps,
                )
                bids.append(opening_order.price)
            else:
                required_asset[opening_order.trader_id] += opening_order.quantity
                asks.append(opening_order.price)
        for trader_id, ledger in accounts.items():
            if required_cash[trader_id] > ledger.cash:
                _rule_fail(
                    MarketStateError,
                    MarketRuleCode.INSUFFICIENT_CASH,
                    "opening buy orders exceed account cash",
                    {
                        "required_cash": required_cash[trader_id],
                        "trader_id": trader_id,
                    },
                )
            if required_asset[trader_id] > ledger.asset:
                _rule_fail(
                    MarketStateError,
                    MarketRuleCode.INSUFFICIENT_ASSET,
                    "opening sell orders exceed account asset",
                    {
                        "required_asset": required_asset[trader_id],
                        "trader_id": trader_id,
                    },
                )
        if bids and asks and max(bids) >= min(asks):
            _rule_fail(
                MarketInvariantError,
                MarketRuleCode.CROSSED_BOOK,
                "resting order book must not remain crossed",
                {"best_ask": min(asks), "best_bid": max(bids)},
            )
        sorted_orders = sorted(
            opening.orders,
            key=lambda order: (
                order.priority_time,
                order.event_key.value,
                order.order_id,
            ),
        )
        self._venue_cash = opening.venue_cash
        self._baseline_cash = (
            sum(account.cash for account in self._accounts.values()) + self._venue_cash
        )
        self._baseline_asset = sum(account.asset for account in self._accounts.values())
        for opening_order in sorted_orders:
            self._add_opening_order(opening_order)
        self._assert_internal_invariants()
        opening_mapping = {
            "accounts": [
                account.to_mapping()
                for account in sorted(opening.accounts, key=lambda item: item.trader_id)
            ],
            "orders": [order.to_mapping() for order in sorted_orders],
            "venue_cash": opening.venue_cash,
        }
        self._opening_state_sha256 = sha256_bytes(canonical_json_bytes(opening_mapping))

    def _validate_opening_order_schema(self, opening: OpeningOrder) -> None:
        if not isinstance(opening, OpeningOrder):
            _rule_fail(
                MarketValidationError,
                MarketRuleCode.INVALID_OPENING_STATE,
                "opening order has an invalid schema",
            )
        self._validate_event_key(opening.event_key)
        self._validate_order_fields(
            order_id=opening.order_id,
            trader_id=opening.trader_id,
            side=opening.side,
            quantity=opening.quantity,
            price=opening.price,
        )
        if (
            isinstance(opening.priority_time, bool)
            or not isinstance(opening.priority_time, int)
            or opening.priority_time > 0
        ):
            _rule_fail(
                MarketValidationError,
                MarketRuleCode.INVALID_OPENING_STATE,
                "opening priority_time must be an integer at or before zero",
                {
                    "event_key": opening.event_key.value,
                    "order_id": opening.order_id,
                },
            )

    def _validate_opening_account(self, account: AccountBalance) -> None:
        if not isinstance(account, AccountBalance):
            _rule_fail(
                MarketValidationError,
                MarketRuleCode.INVALID_OPENING_STATE,
                "opening account has an invalid schema",
            )
        if not is_stable_identifier(account.trader_id):
            _rule_fail(
                MarketValidationError,
                MarketRuleCode.INVALID_OPENING_STATE,
                "opening trader_id must be a stable identifier",
            )
        for field, value in (("cash", account.cash), ("asset", account.asset)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                _rule_fail(
                    MarketValidationError,
                    MarketRuleCode.INVALID_OPENING_STATE,
                    f"opening {field} must be a non-negative integer",
                    {"trader_id": account.trader_id},
                )

    def _add_opening_order(self, opening: OpeningOrder) -> None:
        if not isinstance(opening, OpeningOrder):
            _rule_fail(
                MarketValidationError,
                MarketRuleCode.INVALID_OPENING_STATE,
                "opening order has an invalid schema",
            )
        self._validate_event_key(opening.event_key)
        event_key = opening.event_key.value
        self._validate_order_fields(
            order_id=opening.order_id,
            trader_id=opening.trader_id,
            side=opening.side,
            quantity=opening.quantity,
            price=opening.price,
        )
        if (
            isinstance(opening.priority_time, bool)
            or not isinstance(opening.priority_time, int)
            or opening.priority_time > 0
        ):
            _rule_fail(
                MarketValidationError,
                MarketRuleCode.INVALID_OPENING_STATE,
                "opening priority_time must be an integer at or before zero",
                {"event_key": event_key, "order_id": opening.order_id},
            )
        if opening.order_id in self._history:
            _rule_fail(
                MarketValidationError,
                MarketRuleCode.DUPLICATE_ORDER_ID,
                "opening order IDs must be unique",
                {"event_key": event_key, "order_id": opening.order_id},
            )
        if event_key in self._seen_event_keys:
            _rule_fail(
                MarketValidationError,
                MarketRuleCode.DUPLICATE_EVENT_KEY,
                "opening event keys must be unique",
                {"event_key": event_key},
            )
        order = _WorkingOrder(
            order_id=opening.order_id,
            trader_id=opening.trader_id,
            side=opening.side,
            original_quantity=opening.quantity,
            remaining_quantity=opening.quantity,
            price=opening.price,
            arrival_time=opening.priority_time,
            priority_source=0,
            event_key=event_key,
            order_type="opening_limit",
        )
        self._reserve_limit_order(order, maker_only=True)
        self._orders[order.order_id] = order
        self._history[order.order_id] = _OrderHistory(
            order_id=order.order_id,
            trader_id=order.trader_id,
            side=order.side,
            order_type=order.order_type,
            original_quantity=order.original_quantity,
            filled_quantity=0,
            remaining_quantity=order.remaining_quantity,
            status=OrderStatus.RESTING,
        )
        self._seen_event_keys.add(event_key)

    def _prepare_schedule(
        self, commands: tuple[MarketCommand, ...]
    ) -> tuple[tuple[int, str, MarketCommand], ...]:
        prepared: list[tuple[int, str, MarketCommand]] = []
        ordered_commands = sorted(commands, key=self._preflight_sort_key)
        valid_event_keys: list[str] = []
        for command in ordered_commands:
            if not isinstance(command, LimitOrder | MarketOrder | CancelOrder):
                continue
            try:
                self._validate_event_key(command.event_key)
            except _RuleFailure:
                continue
            valid_event_keys.append(command.event_key.value)
        preflight_counts = Counter(valid_event_keys)
        preflight_duplicates = sorted(
            event_key
            for event_key, count in preflight_counts.items()
            if count > 1 or event_key in self._seen_event_keys
        )
        if preflight_duplicates:
            self._reject_preflight(
                _RuleFailure(
                    MarketValidationError,
                    MarketRuleCode.DUPLICATE_EVENT_KEY,
                    "scheduled event keys must be globally unique",
                    {
                        "duplicate_count": len(preflight_duplicates),
                        "event_key": preflight_duplicates[0],
                    },
                )
            )
        for command in ordered_commands:
            try:
                self._validate_scheduling_fields(command)
                self._validate_command(command)
                event_key = command.event_key.value
                prepared.append(
                    (
                        command.submitted_at + self._config.latency_steps,
                        event_key,
                        command,
                    )
                )
            except _RuleFailure as failure:
                details = dict(failure.details)
                safe_event_key = self._safe_command_event_key(command)
                if safe_event_key is not None:
                    details.setdefault("event_key", safe_event_key)
                self._reject_preflight(
                    _RuleFailure(
                        failure.exception_type,
                        failure.rule_code,
                        failure.message,
                        details,
                    )
                )
        prepared.sort(key=lambda item: (item[0], item[1]))
        counts = Counter(event_key for _, event_key, _ in prepared)
        duplicates = sorted(
            event_key
            for event_key, count in counts.items()
            if count > 1 or event_key in self._seen_event_keys
        )
        if duplicates:
            self._reject_preflight(
                _RuleFailure(
                    MarketValidationError,
                    MarketRuleCode.DUPLICATE_EVENT_KEY,
                    "scheduled event keys must be globally unique",
                    {"event_key": duplicates[0], "duplicate_count": len(duplicates)},
                )
            )
        return tuple(prepared)

    def _preflight_sort_key(
        self, command: MarketCommand
    ) -> tuple[int, str, str, str, str]:
        if not isinstance(command, LimitOrder | MarketOrder | CancelOrder):
            command_type = self._stable_type_name(command)
            return (-2, "invalid-event-key", command_type, "", "")
        submitted_at = command.submitted_at
        logical_time = (
            submitted_at
            if isinstance(submitted_at, int) and not isinstance(submitted_at, bool)
            else -1
        )
        event_identity = self._safe_event_identity(command.event_key)
        order_id = (
            command.target_order_id
            if isinstance(command, CancelOrder)
            else command.order_id
        )
        return (
            logical_time,
            event_identity,
            type(command).__qualname__,
            order_id if isinstance(order_id, str) else self._stable_type_name(order_id),
            command.trader_id
            if isinstance(command.trader_id, str)
            else self._stable_type_name(command.trader_id),
        )

    def _safe_event_identity(self, event_key: object) -> str:
        if not isinstance(event_key, EventKey):
            return f"invalid:{self._stable_type_name(event_key)}"
        try:
            self._validate_event_key(event_key)
        except _RuleFailure:
            material = {
                "experiment_id": self._safe_identity_value(event_key.experiment_id),
                "purpose": self._safe_identity_value(event_key.purpose),
                "replicate_id": self._safe_identity_value(event_key.replicate_id),
                "schema_version": self._safe_identity_value(event_key.schema_version),
                "seed": self._safe_identity_value(event_key.seed),
                "simulation_step": self._safe_identity_value(event_key.simulation_step),
            }
            return f"invalid:{sha256_bytes(canonical_json_bytes(material))}"
        return event_key.value

    def _safe_identity_value(self, value: object) -> str | int | bool | None:
        if value is None or isinstance(value, str | int | bool):
            return value
        return f"type:{self._stable_type_name(value)}"

    def _stable_type_name(self, value: object) -> str:
        value_type = type(value)
        return f"{value_type.__module__}.{value_type.__qualname__}"

    def _validate_scheduling_fields(self, command: MarketCommand) -> None:
        if not isinstance(command, LimitOrder | MarketOrder | CancelOrder):
            _rule_fail(
                MarketValidationError,
                MarketRuleCode.INVALID_EVENT,
                "scheduled value is not a supported market command",
            )
        self._validate_event_key(command.event_key)
        if (
            isinstance(command.submitted_at, bool)
            or not isinstance(command.submitted_at, int)
            or command.submitted_at < 0
        ):
            _rule_fail(
                MarketValidationError,
                MarketRuleCode.INVALID_EVENT,
                "submitted_at must be a non-negative integer logical time",
                {"event_key": command.event_key.value},
            )

    def _validate_event_key(self, event_key: EventKey) -> None:
        if not isinstance(event_key, EventKey):
            _rule_fail(
                MarketValidationError,
                MarketRuleCode.INVALID_EVENT,
                "event_key must be an EventKey",
            )
        if (
            event_key.schema_version != EVENT_KEY_SCHEMA_VERSION
            or not is_stable_identifier(event_key.experiment_id)
            or not isinstance(event_key.seed, str)
            or not event_key.seed
            or len(event_key.seed) > 512
            or isinstance(event_key.replicate_id, bool)
            or not isinstance(event_key.replicate_id, int)
            or event_key.replicate_id < 0
            or isinstance(event_key.simulation_step, bool)
            or not isinstance(event_key.simulation_step, int)
            or event_key.simulation_step < 0
            or not isinstance(event_key.purpose, str)
            or not event_key.purpose
            or len(event_key.purpose) > 512
        ):
            _rule_fail(
                MarketValidationError,
                MarketRuleCode.INVALID_EVENT,
                "event_key fields are invalid or use an unsupported schema",
            )

    def _process_command(self, command: MarketCommand, arrival_time: int) -> None:
        backup = self._capture_backup()
        before_hash = self._state_hash()
        trades_before = len(self._trades)
        policy_records: list[GatewayComplianceRecord] = []
        action = self._command_action(command)
        event_key = command.event_key.value
        try:
            self._validate_command(command)
            self._assert_policy_gateway_runtime()
            if isinstance(command, LimitOrder):
                self._evaluate_policy_quote(
                    cause_event_key=event_key,
                    logical_time=arrival_time,
                    price_units=command.price,
                    records=policy_records,
                )
            self._seen_event_keys.add(event_key)
            if isinstance(command, CancelOrder):
                self._process_cancel(command)
            else:
                self._process_new_order(command, arrival_time, policy_records)
            self._assert_internal_invariants()
            self._append_quote(arrival_time, event_key)
            after_hash = self._state_hash()
            self._audit.append(
                action=action,
                outcome="allowed",
                rule_code=MarketRuleCode.OK,
                simulation_time=arrival_time,
                event_key=event_key,
                details={
                    "after_state_sha256": after_hash,
                    "before_state_sha256": before_hash,
                    "command_sha256": sha256_bytes(
                        canonical_json_bytes(command_to_mapping(command))
                    ),
                    "order_id": self._command_order_id(command),
                    "trade_count": len(self._trades) - trades_before,
                    "policy_cancel_audited": isinstance(command, CancelOrder),
                    **self._policy_audit_details(policy_records),
                },
            )
        except _RuleFailure as failure:
            self._restore_backup(backup)
            after_hash = self._state_hash()
            self._failed = True
            self._failure_message = failure.message
            self._audit.append(
                action=action,
                outcome="rejected",
                rule_code=failure.rule_code,
                simulation_time=arrival_time,
                event_key=event_key,
                details={
                    "after_state_sha256": after_hash,
                    "before_state_sha256": before_hash,
                    "command_sha256": sha256_bytes(
                        canonical_json_bytes(command_to_mapping(command))
                    ),
                    "message": failure.message,
                    "order_id": self._command_order_id(command),
                    "policy_cancel_audited": isinstance(command, CancelOrder),
                    **self._policy_audit_details(policy_records),
                    **failure.details,
                },
            )
            error = failure.exception_type(failure.message)
            error.audit_events = self._audit.events
            error.policy_audit_events = self._policy_events()
            raise error from failure
        except MarketAuditError as audit_error:
            if audit_error.audit_tail_rolled_back is not False:
                self._restore_backup(backup)
            self._failed = True
            self._failure_message = "mandatory market audit append failed"
            audit_error.policy_audit_events = self._policy_events()
            raise
        except Exception as unexpected:
            self._restore_backup(backup)
            after_hash = self._state_hash()
            self._failed = True
            self._failure_message = "unexpected command processing failure"
            self._audit.append(
                action=action,
                outcome="rejected",
                rule_code=MarketRuleCode.INTERNAL_ERROR,
                simulation_time=arrival_time,
                event_key=event_key,
                details={
                    "after_state_sha256": after_hash,
                    "before_state_sha256": before_hash,
                    "exception_type": type(unexpected).__qualname__,
                    "message": self._failure_message,
                    "order_id": self._command_order_id(command),
                    "policy_cancel_audited": isinstance(command, CancelOrder),
                    **self._policy_audit_details(policy_records),
                },
            )
            error = MarketInvariantError(self._failure_message)
            error.audit_events = self._audit.events
            error.policy_audit_events = self._policy_events()
            raise error from unexpected

    def _validate_command(self, command: MarketCommand) -> None:
        self._validate_scheduling_fields(command)
        if isinstance(command, LimitOrder):
            self._validate_order_fields(
                order_id=command.order_id,
                trader_id=command.trader_id,
                side=command.side,
                quantity=command.quantity,
                price=command.price,
            )
        elif isinstance(command, MarketOrder):
            self._validate_order_fields(
                order_id=command.order_id,
                trader_id=command.trader_id,
                side=command.side,
                quantity=command.quantity,
                price=None,
            )
        else:
            if not is_stable_identifier(command.trader_id):
                _rule_fail(
                    MarketValidationError,
                    MarketRuleCode.INVALID_EVENT,
                    "cancel trader_id must be a stable identifier",
                )
            if not is_stable_identifier(command.target_order_id):
                _rule_fail(
                    MarketValidationError,
                    MarketRuleCode.INVALID_EVENT,
                    "cancel target_order_id must be a stable identifier",
                )
            if command.trader_id not in self._accounts:
                _rule_fail(
                    MarketValidationError,
                    MarketRuleCode.UNKNOWN_TRADER,
                    "cancel trader is not present in opening accounts",
                    {"trader_id": command.trader_id},
                )

    def _validate_order_fields(
        self,
        *,
        order_id: object,
        trader_id: object,
        side: object,
        quantity: object,
        price: object | None,
    ) -> None:
        if not is_stable_identifier(order_id):
            _rule_fail(
                MarketValidationError,
                MarketRuleCode.INVALID_EVENT,
                "order_id must be a stable identifier",
            )
        if not is_stable_identifier(trader_id):
            _rule_fail(
                MarketValidationError,
                MarketRuleCode.INVALID_EVENT,
                "trader_id must be a stable identifier",
                {"order_id": cast(str, order_id)},
            )
        if trader_id not in self._accounts:
            _rule_fail(
                MarketValidationError,
                MarketRuleCode.UNKNOWN_TRADER,
                "order trader is not present in opening accounts",
                {"order_id": cast(str, order_id), "trader_id": cast(str, trader_id)},
            )
        if not isinstance(side, Side):
            _rule_fail(
                MarketValidationError,
                MarketRuleCode.INVALID_EVENT,
                "side must be Side.BUY or Side.SELL",
                {"order_id": cast(str, order_id)},
            )
        if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity <= 0:
            _rule_fail(
                MarketValidationError,
                MarketRuleCode.INVALID_QUANTITY,
                "order quantity must be a positive integer",
                {"order_id": cast(str, order_id)},
            )
        if price is not None and (
            isinstance(price, bool)
            or not isinstance(price, int)
            or price <= 0
            or price % self._config.tick_size != 0
        ):
            _rule_fail(
                MarketValidationError,
                MarketRuleCode.INVALID_PRICE_GRID,
                "limit price must be positive and aligned to tick_size",
                {
                    "order_id": cast(str, order_id),
                    "price": price if isinstance(price, int) else None,
                    "tick_size": self._config.tick_size,
                },
            )

    def _process_new_order(
        self,
        command: LimitOrder | MarketOrder,
        arrival_time: int,
        policy_records: list[GatewayComplianceRecord],
    ) -> None:
        if command.order_id in self._history:
            _rule_fail(
                MarketValidationError,
                MarketRuleCode.DUPLICATE_ORDER_ID,
                "order_id cannot be reused",
                {"order_id": command.order_id},
            )
        price = command.price if isinstance(command, LimitOrder) else None
        order_type = "limit" if isinstance(command, LimitOrder) else "market"
        incoming = _WorkingOrder(
            order_id=command.order_id,
            trader_id=command.trader_id,
            side=command.side,
            original_quantity=command.quantity,
            remaining_quantity=command.quantity,
            price=price,
            arrival_time=arrival_time,
            priority_source=1,
            event_key=command.event_key.value,
            order_type=order_type,
        )
        plan = self._plan_fills(incoming)
        for fill_index, planned in enumerate(plan):
            maker = self._orders.get(planned.maker_order_id)
            if maker is None or maker.price is None:
                _rule_fail(
                    MarketInvariantError,
                    MarketRuleCode.SNAPSHOT_STRUCTURE,
                    "planned maker disappeared before policy validation",
                    {"order_id": planned.maker_order_id},
                )
            self._evaluate_policy_trade(
                cause_event_key=incoming.event_key,
                logical_time=arrival_time,
                price_units=maker.price,
                fill_index=fill_index,
                records=policy_records,
            )
        self._history[incoming.order_id] = _OrderHistory(
            order_id=incoming.order_id,
            trader_id=incoming.trader_id,
            side=incoming.side,
            order_type=incoming.order_type,
            original_quantity=incoming.original_quantity,
            filled_quantity=0,
            remaining_quantity=incoming.remaining_quantity,
            status=OrderStatus.EXPIRED,
        )
        if isinstance(command, LimitOrder):
            self._reserve_limit_order(incoming, maker_only=False)
        else:
            self._reserve_market_order(incoming, plan)
        for fill_index, planned in enumerate(plan):
            maker = self._orders.get(planned.maker_order_id)
            if maker is None:
                _rule_fail(
                    MarketInvariantError,
                    MarketRuleCode.SNAPSHOT_STRUCTURE,
                    "planned maker disappeared before atomic commit",
                    {"order_id": planned.maker_order_id},
                )
            self._apply_fill(
                incoming,
                maker,
                planned.quantity,
                fill_index=fill_index,
                simulation_time=arrival_time,
            )
        self._finalize_incoming(incoming)

    def _process_cancel(self, command: CancelOrder) -> None:
        order = self._orders.get(command.target_order_id)
        if order is None:
            if command.target_order_id in self._history:
                _rule_fail(
                    MarketStateError,
                    MarketRuleCode.ORDER_NOT_ACTIVE,
                    "target order is no longer active",
                    {"order_id": command.target_order_id},
                )
            _rule_fail(
                MarketStateError,
                MarketRuleCode.UNKNOWN_ORDER,
                "target order does not exist",
                {"order_id": command.target_order_id},
            )
        if order.trader_id != command.trader_id:
            _rule_fail(
                MarketStateError,
                MarketRuleCode.UNAUTHORIZED_CANCEL,
                "only the order owner may cancel an active order",
                {"order_id": command.target_order_id},
            )
        self._release_reservations(order)
        del self._orders[order.order_id]
        history = self._history[order.order_id]
        history.status = OrderStatus.CANCELED

    def _plan_fills(self, incoming: _WorkingOrder) -> tuple[_PlannedFill, ...]:
        candidates = self._eligible_opposite_orders(incoming)
        remaining = incoming.remaining_quantity
        fills: list[_PlannedFill] = []
        for maker in candidates:
            if remaining == 0:
                break
            if maker.trader_id == incoming.trader_id:
                _rule_fail(
                    MarketStateError,
                    MarketRuleCode.SELF_TRADE,
                    "self-trading is explicitly disabled in the MVP",
                    {"order_id": incoming.order_id, "maker_order_id": maker.order_id},
                )
            quantity = min(remaining, maker.remaining_quantity)
            fills.append(_PlannedFill(maker.order_id, quantity))
            remaining -= quantity
        return tuple(fills)

    def _eligible_opposite_orders(self, incoming: _WorkingOrder) -> list[_WorkingOrder]:
        opposite = Side.SELL if incoming.side is Side.BUY else Side.BUY
        candidates = [
            order for order in self._orders.values() if order.side is opposite
        ]
        if incoming.price is not None:
            if incoming.side is Side.BUY:
                candidates = [
                    order
                    for order in candidates
                    if cast(int, order.price) <= incoming.price
                ]
            else:
                candidates = [
                    order
                    for order in candidates
                    if cast(int, order.price) >= incoming.price
                ]
        if incoming.side is Side.BUY:
            candidates.sort(
                key=lambda order: (cast(int, order.price), order.priority_key())
            )
        else:
            candidates.sort(
                key=lambda order: (-cast(int, order.price), order.priority_key())
            )
        return candidates

    def _reserve_limit_order(self, order: _WorkingOrder, *, maker_only: bool) -> None:
        if order.price is None:
            _rule_fail(
                MarketInvariantError,
                MarketRuleCode.SNAPSHOT_STRUCTURE,
                "limit order is missing its price",
            )
        account = self._accounts[order.trader_id]
        if order.side is Side.BUY:
            fee_rate = (
                self._config.maker_fee_bps
                if maker_only
                else max(self._config.maker_fee_bps, self._config.taker_fee_bps)
            )
            reserve = self._buy_reserve(order.price, order.remaining_quantity, fee_rate)
            if account.available_cash < reserve:
                _rule_fail(
                    MarketStateError,
                    MarketRuleCode.INSUFFICIENT_CASH,
                    "buyer has insufficient available cash for the limit order",
                    {"order_id": order.order_id, "required_cash": reserve},
                )
            account.reserved_cash += reserve
            order.reserved_cash = reserve
        else:
            if account.available_asset < order.remaining_quantity:
                _rule_fail(
                    MarketStateError,
                    MarketRuleCode.INSUFFICIENT_ASSET,
                    "seller has insufficient available asset for the limit order",
                    {
                        "order_id": order.order_id,
                        "required_asset": order.remaining_quantity,
                    },
                )
            account.reserved_asset += order.remaining_quantity
            order.reserved_asset = order.remaining_quantity

    def _reserve_market_order(
        self, order: _WorkingOrder, plan: tuple[_PlannedFill, ...]
    ) -> None:
        account = self._accounts[order.trader_id]
        if order.side is Side.BUY:
            reserve = 0
            for planned in plan:
                maker = self._orders[planned.maker_order_id]
                notional = cast(int, maker.price) * planned.quantity
                reserve += notional + self._fee(notional, self._config.taker_fee_bps)
            if account.available_cash < reserve:
                _rule_fail(
                    MarketStateError,
                    MarketRuleCode.INSUFFICIENT_CASH,
                    "buyer has insufficient available cash for marketable depth",
                    {"order_id": order.order_id, "required_cash": reserve},
                )
            account.reserved_cash += reserve
            order.reserved_cash = reserve
        else:
            if account.available_asset < order.original_quantity:
                _rule_fail(
                    MarketStateError,
                    MarketRuleCode.INSUFFICIENT_ASSET,
                    "seller has insufficient available asset for the market order",
                    {
                        "order_id": order.order_id,
                        "required_asset": order.original_quantity,
                    },
                )
            account.reserved_asset += order.original_quantity
            order.reserved_asset = order.original_quantity

    def _apply_fill(
        self,
        incoming: _WorkingOrder,
        maker: _WorkingOrder,
        quantity: int,
        *,
        fill_index: int,
        simulation_time: int,
    ) -> None:
        if (
            quantity <= 0
            or quantity > incoming.remaining_quantity
            or quantity > maker.remaining_quantity
        ):
            _rule_fail(
                MarketInvariantError,
                MarketRuleCode.NEGATIVE_ORDER_QUANTITY,
                "planned fill quantity is inconsistent with remaining orders",
            )
        price = cast(int, maker.price)
        notional = price * quantity
        maker_fee = self._fee(notional, self._config.maker_fee_bps)
        taker_fee = self._fee(notional, self._config.taker_fee_bps)
        buyer = incoming if incoming.side is Side.BUY else maker
        seller = maker if incoming.side is Side.BUY else incoming
        buyer_fee = taker_fee if buyer is incoming else maker_fee
        seller_fee = maker_fee if seller is maker else taker_fee
        buyer_account = self._accounts[buyer.trader_id]
        seller_account = self._accounts[seller.trader_id]
        buyer_payment = notional + buyer_fee
        if buyer.reserved_cash < buyer_payment:
            _rule_fail(
                MarketInvariantError,
                MarketRuleCode.RESERVATION_MISMATCH,
                "buyer reservation is insufficient for an atomic fill",
                {"order_id": buyer.order_id},
            )
        if seller.reserved_asset < quantity:
            _rule_fail(
                MarketInvariantError,
                MarketRuleCode.RESERVATION_MISMATCH,
                "seller reservation is insufficient for an atomic fill",
                {"order_id": seller.order_id},
            )
        buyer_account.cash -= buyer_payment
        buyer_account.reserved_cash -= buyer_payment
        buyer.reserved_cash -= buyer_payment
        seller_account.cash += notional - seller_fee
        self._venue_cash += buyer_fee + seller_fee
        seller_account.asset -= quantity
        seller_account.reserved_asset -= quantity
        seller.reserved_asset -= quantity
        buyer_account.asset += quantity
        incoming.remaining_quantity -= quantity
        maker.remaining_quantity -= quantity
        incoming_history = self._history[incoming.order_id]
        maker_history = self._history[maker.order_id]
        incoming_history.filled_quantity += quantity
        incoming_history.remaining_quantity -= quantity
        maker_history.filled_quantity += quantity
        maker_history.remaining_quantity -= quantity
        if maker.side is Side.BUY:
            desired = self._buy_reserve(
                price, maker.remaining_quantity, self._config.maker_fee_bps
            )
            self._set_cash_reserve(maker, desired)
        if maker.remaining_quantity == 0:
            self._release_reservations(maker)
            del self._orders[maker.order_id]
            maker_history.status = OrderStatus.FILLED
        trade_material = {
            "cause_event_key": incoming.event_key,
            "fill_index": fill_index,
            "maker_order_id": maker.order_id,
            "quantity": quantity,
            "taker_order_id": incoming.order_id,
        }
        trade_id = f"ptt1:{sha256_bytes(canonical_json_bytes(trade_material))}"
        self._trades.append(
            TradeLogEntry(
                sequence=len(self._trades),
                trade_id=trade_id,
                simulation_time=simulation_time,
                cause_event_key=incoming.event_key,
                fill_index=fill_index,
                maker_order_id=maker.order_id,
                taker_order_id=incoming.order_id,
                maker_trader_id=maker.trader_id,
                taker_trader_id=incoming.trader_id,
                buyer_id=buyer.trader_id,
                seller_id=seller.trader_id,
                aggressor_side=incoming.side,
                price=price,
                quantity=quantity,
                maker_fee=maker_fee,
                taker_fee=taker_fee,
            )
        )

    def _finalize_incoming(self, incoming: _WorkingOrder) -> None:
        history = self._history[incoming.order_id]
        if incoming.order_type == "limit" and incoming.remaining_quantity > 0:
            if incoming.side is Side.BUY:
                desired = self._buy_reserve(
                    cast(int, incoming.price),
                    incoming.remaining_quantity,
                    self._config.maker_fee_bps,
                )
                self._set_cash_reserve(incoming, desired)
            self._orders[incoming.order_id] = incoming
            history.status = OrderStatus.RESTING
            return
        self._release_reservations(incoming)
        history.status = (
            OrderStatus.FILLED
            if incoming.remaining_quantity == 0
            else OrderStatus.EXPIRED
        )

    def _set_cash_reserve(self, order: _WorkingOrder, desired: int) -> None:
        account = self._accounts[order.trader_id]
        delta = desired - order.reserved_cash
        if delta > 0 and account.available_cash < delta:
            _rule_fail(
                MarketInvariantError,
                MarketRuleCode.RESERVATION_MISMATCH,
                "price-improvement normalization requires unavailable cash",
                {"order_id": order.order_id},
            )
        account.reserved_cash += delta
        order.reserved_cash = desired

    def _release_reservations(self, order: _WorkingOrder) -> None:
        account = self._accounts[order.trader_id]
        account.reserved_cash -= order.reserved_cash
        account.reserved_asset -= order.reserved_asset
        order.reserved_cash = 0
        order.reserved_asset = 0

    def _fee(self, notional: int, rate_bps: int) -> int:
        return notional * rate_bps // FEE_DENOMINATOR_BPS

    def _buy_reserve(self, price: int, quantity: int, rate_bps: int) -> int:
        notional = price * quantity
        return notional + self._fee(notional, rate_bps)

    def _append_quote(self, simulation_time: int, cause_event_key: str | None) -> None:
        bids = [order for order in self._orders.values() if order.side is Side.BUY]
        asks = [order for order in self._orders.values() if order.side is Side.SELL]
        best_bid = max((cast(int, order.price) for order in bids), default=None)
        best_ask = min((cast(int, order.price) for order in asks), default=None)
        bid_quantity = (
            sum(order.remaining_quantity for order in bids if order.price == best_bid)
            if best_bid is not None
            else 0
        )
        ask_quantity = (
            sum(order.remaining_quantity for order in asks if order.price == best_ask)
            if best_ask is not None
            else 0
        )
        self._quotes.append(
            QuoteLogEntry(
                sequence=len(self._quotes),
                simulation_time=simulation_time,
                cause_event_key=cause_event_key,
                best_bid_price=best_bid,
                best_bid_quantity=bid_quantity,
                best_ask_price=best_ask,
                best_ask_quantity=ask_quantity,
            )
        )

    def _build_snapshot(self) -> MarketSnapshot:
        accounts = tuple(
            AccountSnapshot(
                trader_id=account.trader_id,
                cash=account.cash,
                asset=account.asset,
                reserved_cash=account.reserved_cash,
                reserved_asset=account.reserved_asset,
            )
            for account in sorted(
                self._accounts.values(), key=lambda item: item.trader_id
            )
        )
        orders = tuple(
            RestingOrderSnapshot(
                order_id=order.order_id,
                trader_id=order.trader_id,
                side=order.side,
                price=cast(int, order.price),
                remaining_quantity=order.remaining_quantity,
                arrival_time=order.arrival_time,
                priority_source=order.priority_source,
                event_key=order.event_key,
                reserved_cash=order.reserved_cash,
                reserved_asset=order.reserved_asset,
            )
            for order in sorted(
                self._orders.values(),
                key=lambda item: (
                    item.side.value,
                    cast(int, item.price),
                    item.priority_key(),
                ),
            )
        )
        histories = tuple(
            self._history_snapshot(history)
            for history in sorted(
                self._history.values(), key=lambda item: item.order_id
            )
        )
        return MarketSnapshot(
            venue_cash=self._venue_cash,
            accounts=accounts,
            resting_orders=orders,
            order_records=histories,
        )

    def _history_snapshot(self, history: _OrderHistory) -> OrderRecordSnapshot:
        return OrderRecordSnapshot(
            order_id=history.order_id,
            trader_id=history.trader_id,
            side=history.side,
            order_type=history.order_type,
            original_quantity=history.original_quantity,
            filled_quantity=history.filled_quantity,
            remaining_quantity=history.remaining_quantity,
            status=history.status,
        )

    def _assert_internal_invariants(self) -> None:
        self._validate_snapshot(self._build_snapshot())

    def _validate_snapshot(self, snapshot: MarketSnapshot) -> None:
        if not isinstance(snapshot, MarketSnapshot):
            _rule_fail(
                MarketInvariantError,
                MarketRuleCode.SNAPSHOT_STRUCTURE,
                "snapshot must be a MarketSnapshot",
            )
        if (
            not isinstance(snapshot.accounts, tuple)
            or not isinstance(snapshot.resting_orders, tuple)
            or not isinstance(snapshot.order_records, tuple)
            or isinstance(snapshot.venue_cash, bool)
            or not isinstance(snapshot.venue_cash, int)
        ):
            _rule_fail(
                MarketInvariantError,
                MarketRuleCode.SNAPSHOT_STRUCTURE,
                "snapshot collections and venue_cash have invalid runtime types",
            )
        expected_accounts = set(self._accounts)
        observed_accounts: list[str] = []
        for account in snapshot.accounts:
            if not isinstance(account, AccountSnapshot) or not is_stable_identifier(
                account.trader_id
            ):
                _rule_fail(
                    MarketInvariantError,
                    MarketRuleCode.SNAPSHOT_STRUCTURE,
                    "snapshot account has an invalid schema or trader_id",
                )
            observed_accounts.append(account.trader_id)
        if (
            len(set(observed_accounts)) != len(observed_accounts)
            or set(observed_accounts) != expected_accounts
        ):
            _rule_fail(
                MarketInvariantError,
                MarketRuleCode.SNAPSHOT_STRUCTURE,
                "snapshot account identities differ from opening state",
            )
        cash_total = snapshot.venue_cash
        asset_total = 0
        if snapshot.venue_cash < 0:
            _rule_fail(
                MarketInvariantError,
                MarketRuleCode.CASH_CONSERVATION,
                "venue cash must not be negative",
            )
        account_map: dict[str, AccountSnapshot] = {}
        for account in snapshot.accounts:
            account_map[account.trader_id] = account
            values = (
                account.cash,
                account.asset,
                account.reserved_cash,
                account.reserved_asset,
            )
            if (
                any(
                    isinstance(value, bool) or not isinstance(value, int) or value < 0
                    for value in values
                )
                or account.reserved_cash > account.cash
                or account.reserved_asset > account.asset
            ):
                _rule_fail(
                    MarketInvariantError,
                    MarketRuleCode.RESERVATION_MISMATCH,
                    "account totals and reservations must be non-negative and covered",
                    {"trader_id": account.trader_id},
                )
            cash_total += account.cash
            asset_total += account.asset
        if cash_total != self._baseline_cash:
            _rule_fail(
                ConservationViolation,
                MarketRuleCode.CASH_CONSERVATION,
                "cash is not conserved across accounts and the venue fee pool",
                {"expected_cash": self._baseline_cash, "observed_cash": cash_total},
            )
        if asset_total != self._baseline_asset:
            _rule_fail(
                ConservationViolation,
                MarketRuleCode.ASSET_CONSERVATION,
                "asset quantity is not conserved across accounts",
                {"expected_asset": self._baseline_asset, "observed_asset": asset_total},
            )
        active_ids: set[str] = set()
        expected_reserved_cash: Counter[str] = Counter()
        expected_reserved_asset: Counter[str] = Counter()
        bids: list[int] = []
        asks: list[int] = []
        for order in snapshot.resting_orders:
            if not isinstance(order, RestingOrderSnapshot):
                _rule_fail(
                    MarketInvariantError,
                    MarketRuleCode.SNAPSHOT_STRUCTURE,
                    "resting order has an invalid schema",
                )
            integer_fields = (
                order.price,
                order.remaining_quantity,
                order.arrival_time,
                order.priority_source,
                order.reserved_cash,
                order.reserved_asset,
            )
            if (
                not is_stable_identifier(order.order_id)
                or not is_stable_identifier(order.trader_id)
                or not isinstance(order.side, Side)
                or not isinstance(order.event_key, str)
                or not order.event_key.startswith("ptk1:")
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in integer_fields
                )
                or order.reserved_cash < 0
                or order.reserved_asset < 0
            ):
                _rule_fail(
                    MarketInvariantError,
                    MarketRuleCode.SNAPSHOT_STRUCTURE,
                    "resting order fields have invalid runtime types",
                )
            if order.order_id in active_ids or order.trader_id not in account_map:
                _rule_fail(
                    MarketInvariantError,
                    MarketRuleCode.SNAPSHOT_STRUCTURE,
                    "active order IDs must be unique and owned by an account",
                )
            active_ids.add(order.order_id)
            if order.remaining_quantity <= 0:
                _rule_fail(
                    MarketInvariantError,
                    MarketRuleCode.NEGATIVE_ORDER_QUANTITY,
                    "active orders must have strictly positive remaining quantity",
                    {"order_id": order.order_id},
                )
            if order.price <= 0 or order.price % self._config.tick_size != 0:
                _rule_fail(
                    MarketInvariantError,
                    MarketRuleCode.INVALID_PRICE_GRID,
                    "active order price is off the configured grid",
                    {"order_id": order.order_id, "price": order.price},
                )
            if order.side is Side.BUY:
                desired = self._buy_reserve(
                    order.price,
                    order.remaining_quantity,
                    self._config.maker_fee_bps,
                )
                if order.reserved_cash != desired or order.reserved_asset != 0:
                    _rule_fail(
                        MarketInvariantError,
                        MarketRuleCode.RESERVATION_MISMATCH,
                        "resting buy reservation does not match remaining quantity",
                        {"order_id": order.order_id},
                    )
                expected_reserved_cash[order.trader_id] += order.reserved_cash
                bids.append(order.price)
            elif order.side is Side.SELL:
                if (
                    order.reserved_asset != order.remaining_quantity
                    or order.reserved_cash != 0
                ):
                    _rule_fail(
                        MarketInvariantError,
                        MarketRuleCode.RESERVATION_MISMATCH,
                        "resting sell reservation does not match remaining quantity",
                        {"order_id": order.order_id},
                    )
                expected_reserved_asset[order.trader_id] += order.reserved_asset
                asks.append(order.price)
            else:
                _rule_fail(
                    MarketInvariantError,
                    MarketRuleCode.SNAPSHOT_STRUCTURE,
                    "active order side is invalid",
                )
        if bids and asks and max(bids) >= min(asks):
            _rule_fail(
                MarketInvariantError,
                MarketRuleCode.CROSSED_BOOK,
                "resting order book must not remain crossed",
                {"best_ask": min(asks), "best_bid": max(bids)},
            )
        for trader_id, account in account_map.items():
            if (
                account.reserved_cash != expected_reserved_cash[trader_id]
                or account.reserved_asset != expected_reserved_asset[trader_id]
            ):
                _rule_fail(
                    MarketInvariantError,
                    MarketRuleCode.RESERVATION_MISMATCH,
                    "account reservation does not equal active order reservations",
                    {"trader_id": trader_id},
                )
        record_ids: set[str] = set()
        record_map: dict[str, OrderRecordSnapshot] = {}
        for record in snapshot.order_records:
            if not isinstance(record, OrderRecordSnapshot):
                _rule_fail(
                    MarketInvariantError,
                    MarketRuleCode.SNAPSHOT_STRUCTURE,
                    "order lifecycle record has an invalid schema",
                )
            if (
                not is_stable_identifier(record.order_id)
                or not is_stable_identifier(record.trader_id)
                or not isinstance(record.side, Side)
                or not isinstance(record.order_type, str)
                or record.order_type not in {"limit", "market", "opening_limit"}
                or not isinstance(record.status, OrderStatus)
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in (
                        record.original_quantity,
                        record.filled_quantity,
                        record.remaining_quantity,
                    )
                )
            ):
                _rule_fail(
                    MarketInvariantError,
                    MarketRuleCode.SNAPSHOT_STRUCTURE,
                    "order lifecycle fields have invalid runtime types",
                )
            if record.order_id in record_ids:
                _rule_fail(
                    MarketInvariantError,
                    MarketRuleCode.SNAPSHOT_STRUCTURE,
                    "order lifecycle record IDs must be unique",
                )
            record_ids.add(record.order_id)
            record_map[record.order_id] = record
            if (
                record.original_quantity <= 0
                or record.filled_quantity < 0
                or record.remaining_quantity < 0
                or record.filled_quantity + record.remaining_quantity
                != record.original_quantity
            ):
                _rule_fail(
                    MarketInvariantError,
                    MarketRuleCode.NEGATIVE_ORDER_QUANTITY,
                    "order lifecycle quantities are inconsistent",
                    {"order_id": record.order_id},
                )
            if (record.status is OrderStatus.RESTING) != (
                record.order_id in active_ids
            ):
                _rule_fail(
                    MarketInvariantError,
                    MarketRuleCode.SNAPSHOT_STRUCTURE,
                    "active orders and lifecycle statuses disagree",
                    {"order_id": record.order_id},
                )
            if record.status is OrderStatus.FILLED and record.remaining_quantity != 0:
                _rule_fail(
                    MarketInvariantError,
                    MarketRuleCode.NEGATIVE_ORDER_QUANTITY,
                    "filled order has a non-zero remaining quantity",
                    {"order_id": record.order_id},
                )
        if record_ids != set(self._history):
            _rule_fail(
                MarketInvariantError,
                MarketRuleCode.SNAPSHOT_STRUCTURE,
                "snapshot lifecycle records differ from engine history",
            )
        for order in snapshot.resting_orders:
            linked_record = record_map.get(order.order_id)
            if linked_record is None or (
                linked_record.trader_id != order.trader_id
                or linked_record.side is not order.side
                or linked_record.remaining_quantity != order.remaining_quantity
                or linked_record.status is not OrderStatus.RESTING
            ):
                _rule_fail(
                    MarketInvariantError,
                    MarketRuleCode.SNAPSHOT_STRUCTURE,
                    "resting order and lifecycle record fields disagree",
                    {"order_id": order.order_id},
                )
        for record in snapshot.order_records:
            expected = self._history.get(record.order_id)
            if expected is None or record != self._history_snapshot(expected):
                _rule_fail(
                    MarketInvariantError,
                    MarketRuleCode.SNAPSHOT_STRUCTURE,
                    "snapshot lifecycle record differs from engine history",
                    {"order_id": record.order_id},
                )

    def _capture_backup(self) -> _MutableBackup:
        return (
            copy.deepcopy(self._accounts),
            copy.deepcopy(self._orders),
            copy.deepcopy(self._history),
            self._venue_cash,
            list(self._trades),
            list(self._quotes),
            set(self._seen_event_keys),
        )

    def _restore_backup(self, backup: _MutableBackup) -> None:
        (
            self._accounts,
            self._orders,
            self._history,
            self._venue_cash,
            self._trades,
            self._quotes,
            self._seen_event_keys,
        ) = backup

    def _state_hash(self) -> str:
        return self._build_snapshot().sha256

    def _safe_state_hash(self) -> str:
        try:
            return self._state_hash()
        except Exception:
            return "unavailable"

    def _safe_snapshot_hash(self, snapshot: object) -> str:
        if not isinstance(snapshot, MarketSnapshot):
            return "unavailable"
        try:
            return snapshot.sha256
        except Exception:
            return "unavailable"

    def _reject_preflight(self, failure: _RuleFailure) -> NoReturn:
        self._failed = True
        self._failure_message = failure.message
        state_hash = self._state_hash()
        event_key = cast(str | None, failure.details.get("event_key"))
        self._audit.append(
            action="schedule_batch",
            outcome="rejected",
            rule_code=failure.rule_code,
            simulation_time=None,
            event_key=event_key,
            details={
                "after_state_sha256": state_hash,
                "before_state_sha256": state_hash,
                "message": failure.message,
                **failure.details,
            },
        )
        error = failure.exception_type(failure.message)
        error.audit_events = self._audit.events
        error.policy_audit_events = self._policy_events()
        raise error from failure

    def _latch_external_invariant_failure(
        self,
        *,
        action: str,
        snapshot: MarketSnapshot,
        failure: _RuleFailure,
    ) -> NoReturn:
        self._failed = True
        self._failure_message = failure.message
        current_hash = self._state_hash()
        candidate_hash = self._safe_snapshot_hash(snapshot)
        self._audit.append(
            action=action,
            outcome="rejected",
            rule_code=failure.rule_code,
            simulation_time=None,
            event_key=None,
            details={
                "after_state_sha256": current_hash,
                "before_state_sha256": current_hash,
                "candidate_state_sha256": candidate_hash,
                "message": failure.message,
                **failure.details,
            },
        )
        error = failure.exception_type(failure.message)
        error.audit_events = self._audit.events
        error.policy_audit_events = self._policy_events()
        raise error from failure

    def _record_latched_attempt(
        self, action: str, command: MarketCommand | None
    ) -> NoReturn:
        message = self._failure_message or "market engine is latched after a failure"
        self._audit.append(
            action=action,
            outcome="rejected",
            rule_code=MarketRuleCode.ENGINE_LATCHED,
            simulation_time=None,
            event_key=self._safe_command_event_key(command),
            details={"message": message},
        )
        failure = EngineLatchedError(message)
        failure.audit_events = self._audit.events
        failure.policy_audit_events = self._policy_events()
        raise failure

    def _ensure_audit_healthy(self) -> None:
        try:
            self._audit.assert_integrity()
        except MarketAuditError:
            self._failed = True
            self._failure_message = "mandatory market audit log is compromised"
            raise

    def _assert_policy_gateway_runtime(self) -> None:
        if self._policy_gateway is None:
            return
        try:
            self._policy_gateway.assert_integrity()
        except PolicyGatewayError as error:
            _rule_fail(
                MarketValidationError,
                MarketRuleCode.POLICY_GATEWAY_ERROR,
                "policy gateway failed its runtime integrity check",
                {
                    "exception_type": type(error).__qualname__,
                    "policy_gateway_sha256": self._policy_gateway.sha256,
                },
            )

    def _evaluate_policy_quote(
        self,
        *,
        cause_event_key: str,
        logical_time: int,
        price_units: int,
        records: list[GatewayComplianceRecord],
    ) -> None:
        gateway = self._policy_gateway
        if gateway is None:
            return
        try:
            record = gateway.evaluate_quote(
                cause_event_key=cause_event_key,
                logical_time=logical_time,
                price_units=price_units,
            )
        except PolicyGatewayError as error:
            _rule_fail(
                MarketValidationError,
                MarketRuleCode.POLICY_GATEWAY_ERROR,
                "quote policy evaluation could not produce an audited decision",
                {
                    "exception_type": type(error).__qualname__,
                    "policy_gateway_sha256": gateway.sha256,
                },
            )
        records.append(record)
        if record.decision.outcome is ComplianceOutcome.REJECTED:
            _rule_fail(
                MarketValidationError,
                MarketRuleCode.POLICY_QUOTE_REJECTED,
                "limit quote was rejected by the mandatory policy gateway",
                {
                    **self._policy_audit_details(records),
                    "policy_audit_event_sha256": (
                        record.policy_audit_event.event_sha256
                    ),
                    "policy_decision_sha256": record.decision.sha256,
                    "policy_request_sha256": record.decision.request.sha256,
                    "policy_rule_code": record.decision.rule_code.value,
                },
            )

    def _evaluate_policy_trade(
        self,
        *,
        cause_event_key: str,
        logical_time: int,
        price_units: int,
        fill_index: int,
        records: list[GatewayComplianceRecord],
    ) -> None:
        gateway = self._policy_gateway
        if gateway is None:
            return
        try:
            record = gateway.evaluate_trade(
                cause_event_key=cause_event_key,
                logical_time=logical_time,
                price_units=price_units,
                fill_index=fill_index,
            )
        except PolicyGatewayError as error:
            _rule_fail(
                MarketValidationError,
                MarketRuleCode.POLICY_GATEWAY_ERROR,
                "trade policy evaluation could not produce an audited decision",
                {
                    "exception_type": type(error).__qualname__,
                    "fill_index": fill_index,
                    "policy_gateway_sha256": gateway.sha256,
                },
            )
        records.append(record)
        if record.decision.outcome is ComplianceOutcome.REJECTED:
            _rule_fail(
                MarketValidationError,
                MarketRuleCode.POLICY_TRADE_REJECTED,
                "planned fill was rejected by the mandatory policy gateway",
                {
                    **self._policy_audit_details(records),
                    "fill_index": fill_index,
                    "policy_audit_event_sha256": (
                        record.policy_audit_event.event_sha256
                    ),
                    "policy_decision_sha256": record.decision.sha256,
                    "policy_request_sha256": record.decision.request.sha256,
                    "policy_rule_code": record.decision.rule_code.value,
                },
            )

    def _policy_audit_details(
        self, records: list[GatewayComplianceRecord]
    ) -> dict[str, AuditScalar]:
        gateway_sha256 = (
            None if self._policy_gateway is None else self._policy_gateway.sha256
        )
        record_sha256 = sha256_bytes(
            canonical_json_bytes([record.to_mapping() for record in records])
        )
        last = records[-1] if records else None
        return {
            "policy_decision_count": len(records),
            "policy_decision_sequence_sha256": record_sha256,
            "policy_gateway_sha256": gateway_sha256,
            "policy_last_audit_event_sha256": (
                None if last is None else last.policy_audit_event.event_sha256
            ),
            "policy_last_decision_sha256": (
                None if last is None else last.decision.sha256
            ),
            "policy_last_request_sha256": (
                None if last is None else last.decision.request.sha256
            ),
            "policy_mode": self._config.policy_mode.value,
            "policy_security_id": (
                None
                if self._policy_gateway is None
                else self._policy_gateway.config.security_id
            ),
        }

    def _policy_events(self) -> tuple[PolicyComplianceAuditEvent, ...]:
        return (
            ()
            if self._policy_gateway is None
            else self._policy_gateway.policy_audit_events
        )

    def _reject_sealed_schedule(
        self, action: str, command: MarketCommand | None
    ) -> NoReturn:
        message = "cannot schedule after the batch is sealed"
        try:
            self._audit.append(
                action=action,
                outcome="rejected",
                rule_code=MarketRuleCode.ENGINE_SEALED,
                simulation_time=None,
                event_key=self._safe_command_event_key(command),
                details={"message": message},
            )
        except MarketAuditError:
            self._failed = True
            self._failure_message = "mandatory market audit append failed"
            raise
        failure = MarketStateError(message)
        failure.audit_events = self._audit.events
        raise failure

    def _command_action(self, command: MarketCommand) -> str:
        if isinstance(command, LimitOrder):
            return "limit_order"
        if isinstance(command, MarketOrder):
            return "market_order"
        return "cancel_order"

    def _command_order_id(self, command: MarketCommand) -> str:
        return (
            command.target_order_id
            if isinstance(command, CancelOrder)
            else command.order_id
        )

    def _safe_command_event_key(self, command: MarketCommand | None) -> str | None:
        if command is None or not hasattr(command, "event_key"):
            return None
        event_key = command.event_key
        return self._safe_event_identity(event_key)


__all__ = ["CentralLimitOrderBook"]
