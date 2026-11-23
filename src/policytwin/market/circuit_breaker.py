"""Synthetic single-market circuit breaker with deterministic recovery.

The core CLOB remains the sole matching and accounting implementation.  This
module is an audited command gateway around :class:`ReplayMarketSession`: it
may reject commands while the synthetic venue is halted, but every command it
accepts is executed by an exact prefix replay of the core book.

Nothing in this module encodes or claims the rules of a real trading venue.
"""

from __future__ import annotations

import heapq
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from threading import RLock
from typing import cast

from policytwin.counterfactual import EVENT_KEY_SCHEMA_VERSION, EventKey
from policytwin.market.models import (
    CancelOrder,
    LimitOrder,
    MarketAuditEvent,
    MarketAuditLog,
    MarketCommand,
    MarketConfig,
    MarketOrder,
    MarketPolicyMode,
    MarketRuleCode,
    OpeningState,
    ReplayResult,
    ScheduledCommand,
    TradeLogEntry,
    command_to_mapping,
    event_key_to_mapping,
    is_stable_identifier,
)
from policytwin.market.order_book import CentralLimitOrderBook
from policytwin.market.session import ReplayMarketSession
from policytwin.provenance import JsonlEventWriter, canonical_json_bytes, sha256_bytes

CIRCUIT_BREAKER_SCHEMA_VERSION = "policytwin.synthetic-circuit-breaker.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EVENT_KEY_PREFIX = "ptk1:"


class CircuitBreakerError(RuntimeError):
    """Base class for fail-closed circuit-breaker failures."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.audit_events: tuple[MarketAuditEvent, ...] = ()


class HaltPolicyError(CircuitBreakerError):
    """Raised when a synthetic halt policy is invalid or unsupported."""


class HaltReplayError(CircuitBreakerError):
    """Raised when a halt replay artifact does not reproduce exactly."""


class HaltCommandRule(StrEnum):
    """Explicit command disposition while the venue is halted."""

    REJECT = "reject"
    ALLOW = "allow"


class HaltReferenceMode(StrEnum):
    """Reference-price choice for the synthetic price-move trigger."""

    FIXED = "fixed"
    ROLLING_FIRST_TRADE = "rolling_first_trade"


class ThresholdBoundary(StrEnum):
    """Supported trigger-threshold boundary."""

    INCLUSIVE = "greater_than_or_equal"


class RollingWindowBoundary(StrEnum):
    """Supported rolling-window boundary."""

    LEFT_INCLUSIVE_CURRENT_EXCLUDED = "left_inclusive_current_excluded"


class MonitoringBoundary(StrEnum):
    """Supported monitoring-interval boundary."""

    INCLUSIVE = "start_and_end_inclusive"


class RecoveryOrdering(StrEnum):
    """Total order between commands and internal recovery events."""

    LOGICAL_TIME_THEN_EVENT_KEY = "logical_time_then_event_key"


class HaltStatus(StrEnum):
    """Synthetic venue halt state."""

    ACTIVE = "active"
    HALTED = "halted"


class HaltTransitionKind(StrEnum):
    """State transition represented in the halt ledger."""

    TRIGGER = "trigger"
    RECOVERY = "recovery"


@dataclass(frozen=True, slots=True)
class HaltPolicy:
    """Frozen synthetic circuit-breaker policy.

    A trade triggers a halt when its absolute move from the selected reference
    is at least ``trigger_threshold_bps``.  The equality boundary is therefore
    included.  Rolling references use the earliest prior trade in
    ``[trade_time - rolling_window_steps, trade_time)`` and otherwise fall back
    to ``reference_price``.  Monitoring start and end are both inclusive.

    A recovery is an internal event at ``trigger_time + halt_duration_steps``.
    Commands and recovery events at the same logical time are ordered solely by
    their complete ``EventKey.value``.  Consequently, a same-time command whose
    key sorts before the recovery key still observes the halted state.
    """

    policy_id: str
    reference_mode: HaltReferenceMode
    reference_price: int
    rolling_window_steps: int
    trigger_threshold_bps: int
    halt_duration_steps: int
    monitoring_start: int
    monitoring_end: int
    limit_order_during_halt: HaltCommandRule
    market_order_during_halt: HaltCommandRule
    cancel_during_halt: HaltCommandRule
    threshold_boundary: ThresholdBoundary = ThresholdBoundary.INCLUSIVE
    rolling_window_boundary: RollingWindowBoundary = (
        RollingWindowBoundary.LEFT_INCLUSIVE_CURRENT_EXCLUDED
    )
    monitoring_boundary: MonitoringBoundary = MonitoringBoundary.INCLUSIVE
    recovery_ordering: RecoveryOrdering = RecoveryOrdering.LOGICAL_TIME_THEN_EVENT_KEY
    synthetic_only: bool = True
    schema_version: str = CIRCUIT_BREAKER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not is_stable_identifier(self.policy_id):
            raise HaltPolicyError("policy_id must be a stable identifier")
        enum_fields: tuple[tuple[str, object, type[StrEnum]], ...] = (
            ("reference_mode", self.reference_mode, HaltReferenceMode),
            (
                "limit_order_during_halt",
                self.limit_order_during_halt,
                HaltCommandRule,
            ),
            (
                "market_order_during_halt",
                self.market_order_during_halt,
                HaltCommandRule,
            ),
            ("cancel_during_halt", self.cancel_during_halt, HaltCommandRule),
            ("threshold_boundary", self.threshold_boundary, ThresholdBoundary),
            (
                "rolling_window_boundary",
                self.rolling_window_boundary,
                RollingWindowBoundary,
            ),
            ("monitoring_boundary", self.monitoring_boundary, MonitoringBoundary),
            ("recovery_ordering", self.recovery_ordering, RecoveryOrdering),
        )
        for field, value, expected in enum_fields:
            if type(value) is not expected:
                raise HaltPolicyError(f"{field} must use the supported enum schema")
        positive_fields = (
            ("reference_price", self.reference_price),
            ("rolling_window_steps", self.rolling_window_steps),
            ("trigger_threshold_bps", self.trigger_threshold_bps),
            ("halt_duration_steps", self.halt_duration_steps),
        )
        for field, value in positive_fields:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise HaltPolicyError(f"{field} must be a positive integer")
        if self.trigger_threshold_bps > 1_000_000:
            raise HaltPolicyError("trigger_threshold_bps exceeds the supported bound")
        for field, value in (
            ("monitoring_start", self.monitoring_start),
            ("monitoring_end", self.monitoring_end),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise HaltPolicyError(f"{field} must be a non-negative integer")
        if self.monitoring_end < self.monitoring_start:
            raise HaltPolicyError("monitoring_end must not precede monitoring_start")
        if self.synthetic_only is not True:
            raise HaltPolicyError(
                "only explicitly synthetic halt policies are supported"
            )
        if self.schema_version != CIRCUIT_BREAKER_SCHEMA_VERSION:
            raise HaltPolicyError("unsupported circuit-breaker schema version")

    def to_mapping(self) -> dict[str, object]:
        """Return the canonical policy representation including boundaries."""
        return {
            "cancel_during_halt": self.cancel_during_halt.value,
            "halt_activation": "after_trigger_command_atomic_completion",
            "halt_duration_steps": self.halt_duration_steps,
            "limit_order_during_halt": self.limit_order_during_halt.value,
            "market_order_during_halt": self.market_order_during_halt.value,
            "monitoring_boundary": self.monitoring_boundary.value,
            "monitoring_end": self.monitoring_end,
            "monitoring_start": self.monitoring_start,
            "policy_id": self.policy_id,
            "recovery_ordering": self.recovery_ordering.value,
            "reference_mode": self.reference_mode.value,
            "reference_price": self.reference_price,
            "rolling_window_boundary": self.rolling_window_boundary.value,
            "rolling_window_steps": self.rolling_window_steps,
            "schema_version": self.schema_version,
            "synthetic_only": self.synthetic_only,
            "threshold_boundary": self.threshold_boundary.value,
            "trigger_threshold_bps": self.trigger_threshold_bps,
        }

    @property
    def sha256(self) -> str:
        """Return the semantic policy commitment."""
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def command_rule(self, command: MarketCommand) -> HaltCommandRule:
        """Return the explicit halted-state rule for one supported command."""
        if isinstance(command, LimitOrder):
            return self.limit_order_during_halt
        if isinstance(command, MarketOrder):
            return self.market_order_during_halt
        if isinstance(command, CancelOrder):
            return self.cancel_during_halt
        raise HaltPolicyError("unsupported command type cannot pass the halt gateway")


@dataclass(frozen=True, slots=True)
class HaltTransition:
    """One hash-committed trigger or deterministic recovery transition."""

    sequence: int
    kind: HaltTransitionKind
    status_after: HaltStatus
    logical_time: int
    event_key: EventKey
    trigger_event_key: str
    recovery_event_key: str
    halted_until: int
    observed_price: int | None
    reference_price: int | None
    cause_trade_id: str | None
    policy_sha256: str
    transition_sha256: str
    schema_version: str = CIRCUIT_BREAKER_SCHEMA_VERSION

    def core_mapping(self) -> dict[str, object]:
        """Return the transition fields covered by ``transition_sha256``."""
        return {
            "cause_trade_id": self.cause_trade_id,
            "event_key": event_key_to_mapping(self.event_key),
            "halted_until": self.halted_until,
            "kind": self.kind.value,
            "logical_time": self.logical_time,
            "observed_price": self.observed_price,
            "policy_sha256": self.policy_sha256,
            "recovery_event_key": self.recovery_event_key,
            "reference_price": self.reference_price,
            "schema_version": self.schema_version,
            "sequence": self.sequence,
            "status_after": self.status_after.value,
            "trigger_event_key": self.trigger_event_key,
        }

    def to_mapping(self) -> dict[str, object]:
        """Return the complete machine-readable transition."""
        return {**self.core_mapping(), "transition_sha256": self.transition_sha256}

    def assert_integrity(self) -> None:
        """Verify schema, state semantics, and transition hash."""
        if self.schema_version != CIRCUIT_BREAKER_SCHEMA_VERSION:
            raise HaltReplayError("unsupported halt-transition schema")
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence < 0
        ):
            raise HaltReplayError("halt-transition sequence is invalid")
        if (
            type(self.kind) is not HaltTransitionKind
            or type(self.status_after) is not HaltStatus
        ):
            raise HaltReplayError("halt-transition enum fields are invalid")
        expected_status = (
            HaltStatus.HALTED
            if self.kind is HaltTransitionKind.TRIGGER
            else HaltStatus.ACTIVE
        )
        if self.status_after is not expected_status:
            raise HaltReplayError("halt-transition status does not match its kind")
        if not _valid_event_key(self.event_key):
            raise HaltReplayError("halt-transition event key is invalid")
        if self.event_key.value != (
            self.trigger_event_key
            if self.kind is HaltTransitionKind.TRIGGER
            else self.recovery_event_key
        ):
            raise HaltReplayError("halt-transition event identity is inconsistent")
        for field in ("trigger_event_key", "recovery_event_key"):
            if not _valid_event_key_value(getattr(self, field)):
                raise HaltReplayError(f"{field} is invalid")
        for field in ("policy_sha256", "transition_sha256"):
            if not _valid_sha256(getattr(self, field)):
                raise HaltReplayError(f"{field} is invalid")
        if (
            sha256_bytes(canonical_json_bytes(self.core_mapping()))
            != self.transition_sha256
        ):
            raise HaltReplayError("halt-transition hash mismatch")


@dataclass(frozen=True, slots=True)
class HaltReplayArtifact:
    """Complete replay artifact binding policy, gateway ledger, and core CLOB."""

    policy: HaltPolicy
    market_config: MarketConfig
    opening_state: OpeningState
    attempted_commands: tuple[ScheduledCommand, ...]
    accepted_commands: tuple[ScheduledCommand, ...]
    transitions: tuple[HaltTransition, ...]
    halt_audit_events: tuple[MarketAuditEvent, ...]
    final_market_replay: ReplayResult
    policy_sha256: str
    attempted_commands_sha256: str
    accepted_commands_sha256: str
    transitions_sha256: str
    halt_audit_sha256: str
    final_market_replay_sha256: str
    artifact_sha256: str
    schema_version: str = CIRCUIT_BREAKER_SCHEMA_VERSION

    def core_mapping(self) -> dict[str, object]:
        """Return all artifact fields covered by ``artifact_sha256``."""
        return {
            "accepted_commands": [item.to_mapping() for item in self.accepted_commands],
            "accepted_commands_sha256": self.accepted_commands_sha256,
            "attempted_commands": [
                item.to_mapping() for item in self.attempted_commands
            ],
            "attempted_commands_sha256": self.attempted_commands_sha256,
            "final_market_replay": self.final_market_replay.to_mapping(),
            "final_market_replay_sha256": self.final_market_replay_sha256,
            "halt_audit_events": [item.to_mapping() for item in self.halt_audit_events],
            "halt_audit_sha256": self.halt_audit_sha256,
            "market_config": self.market_config.to_mapping(),
            "opening_state": self.opening_state.to_mapping(),
            "policy": self.policy.to_mapping(),
            "policy_sha256": self.policy_sha256,
            "schema_version": self.schema_version,
            "transitions": [item.to_mapping() for item in self.transitions],
            "transitions_sha256": self.transitions_sha256,
        }

    def to_mapping(self) -> dict[str, object]:
        """Return the complete machine-readable replay artifact."""
        return {**self.core_mapping(), "artifact_sha256": self.artifact_sha256}

    @property
    def canonical_bytes(self) -> bytes:
        """Return canonical bytes for exact replay comparison."""
        return canonical_json_bytes(self.to_mapping())

    def assert_integrity(self) -> None:
        """Verify every component commitment and cross-artifact binding."""
        if self.schema_version != CIRCUIT_BREAKER_SCHEMA_VERSION:
            raise HaltReplayError("unsupported halt replay schema")
        if type(self.policy) is not HaltPolicy:
            raise HaltReplayError("halt replay policy has the wrong type")
        if self.policy.sha256 != self.policy_sha256:
            raise HaltReplayError("halt policy differs from its commitment")
        if self.market_config.policy_mode is not MarketPolicyMode.NO_POLICY_SYNTHETIC:
            raise HaltReplayError(
                "halt replay is restricted to the synthetic no-policy CLOB"
            )
        if self.final_market_replay.config != self.market_config:
            raise HaltReplayError("final market replay uses a different configuration")
        if self.final_market_replay.opening_state != self.opening_state:
            raise HaltReplayError("final market replay uses a different opening state")
        components: tuple[tuple[str, object, str], ...] = (
            (
                "attempted commands",
                [item.to_mapping() for item in self.attempted_commands],
                self.attempted_commands_sha256,
            ),
            (
                "accepted commands",
                [item.to_mapping() for item in self.accepted_commands],
                self.accepted_commands_sha256,
            ),
            (
                "halt transitions",
                [item.to_mapping() for item in self.transitions],
                self.transitions_sha256,
            ),
            (
                "halt audit",
                [item.to_mapping() for item in self.halt_audit_events],
                self.halt_audit_sha256,
            ),
        )
        for label, value, expected in components:
            if (
                not _valid_sha256(expected)
                or sha256_bytes(canonical_json_bytes(value)) != expected
            ):
                raise HaltReplayError(f"{label} commitment mismatch")
        if (
            not _valid_sha256(self.final_market_replay_sha256)
            or self.final_market_replay.sha256 != self.final_market_replay_sha256
        ):
            raise HaltReplayError("final market replay commitment mismatch")
        if self.accepted_commands != self.final_market_replay.scheduled_commands:
            raise HaltReplayError("accepted commands differ from the core replay")
        _assert_scheduled_order(
            self.attempted_commands, self.market_config.latency_steps
        )
        _assert_scheduled_order(
            self.accepted_commands, self.market_config.latency_steps
        )
        attempted_by_key = {item.event_key: item for item in self.attempted_commands}
        if len(attempted_by_key) != len(self.attempted_commands):
            raise HaltReplayError("attempted command event keys are not unique")
        for accepted in self.accepted_commands:
            if attempted_by_key.get(accepted.event_key) != accepted:
                raise HaltReplayError(
                    "accepted command is absent from attempted commands"
                )
        previous = "0" * 64
        for sequence, event in enumerate(self.halt_audit_events):
            if (
                event.sequence != sequence
                or event.previous_event_sha256 != previous
                or sha256_bytes(canonical_json_bytes(event.core_mapping()))
                != event.event_sha256
            ):
                raise HaltReplayError("halt audit hash chain mismatch")
            previous = event.event_sha256
        for sequence, transition in enumerate(self.transitions):
            transition.assert_integrity()
            if (
                transition.sequence != sequence
                or transition.policy_sha256 != self.policy_sha256
            ):
                raise HaltReplayError("halt transition ledger order or policy changed")
        if not _valid_sha256(self.artifact_sha256):
            raise HaltReplayError("halt artifact hash is invalid")
        if (
            sha256_bytes(canonical_json_bytes(self.core_mapping()))
            != self.artifact_sha256
        ):
            raise HaltReplayError("halt artifact hash mismatch")

    def write_logs(self, destination: Path) -> None:
        """Write deterministic JSONL halt and transition logs plus core logs."""
        destination.mkdir(parents=True, exist_ok=False)
        logs: tuple[tuple[str, str, tuple[Mapping[str, object], ...]], ...] = (
            (
                "halt_audit.jsonl",
                "halt_audit",
                tuple(item.to_mapping() for item in self.halt_audit_events),
            ),
            (
                "halt_transitions.jsonl",
                "halt_transitions",
                tuple(item.to_mapping() for item in self.transitions),
            ),
        )
        for filename, log_kind, records in logs:
            with JsonlEventWriter(destination / filename) as writer:
                writer.write(
                    {
                        "artifact_sha256": self.artifact_sha256,
                        "log_kind": log_kind,
                        "policy_sha256": self.policy_sha256,
                        "record_type": "header",
                        "schema_version": CIRCUIT_BREAKER_SCHEMA_VERSION,
                    }
                )
                for record in records:
                    writer.write(record)
        self.final_market_replay.write_logs(destination / "core_market_logs")


@dataclass(frozen=True, slots=True)
class _RecoveryEvent:
    logical_time: int
    event_key: EventKey
    trigger_event_key: str


@dataclass(frozen=True, slots=True)
class _TradeObservation:
    simulation_time: int
    sequence: int
    price: int


type _QueuedPayload = MarketCommand | _RecoveryEvent
type _QueueEntry = tuple[int, str, _QueuedPayload]


class SyntheticCircuitBreakerMarket:
    """Thread-safe scheduler for one synthetic halted-market replay."""

    def __init__(
        self,
        market_config: MarketConfig,
        opening_state: OpeningState,
        halt_policy: HaltPolicy,
    ) -> None:
        if type(market_config) is not MarketConfig:
            raise HaltPolicyError("market_config must be an immutable MarketConfig")
        if market_config.policy_mode is not MarketPolicyMode.NO_POLICY_SYNTHETIC:
            raise HaltPolicyError(
                "synthetic circuit breaker cannot bypass or compose with a "
                "policy gateway"
            )
        if type(opening_state) is not OpeningState:
            raise HaltPolicyError("opening_state must be an immutable OpeningState")
        if type(halt_policy) is not HaltPolicy:
            raise HaltPolicyError("halt_policy must be an immutable HaltPolicy")
        # A fresh core construction provides authoritative config/opening checks.
        CentralLimitOrderBook(market_config, opening_state)
        self._market_config = market_config
        self._opening_state = opening_state
        self._policy = halt_policy
        self._scheduled: list[MarketCommand] = []
        self._lock = RLock()
        self._sealed = False
        self._result: HaltReplayArtifact | None = None

    @property
    def halt_policy(self) -> HaltPolicy:
        """Return the frozen synthetic halt policy."""
        return self._policy

    def schedule(self, command: MarketCommand) -> None:
        """Buffer one command without using producer completion order."""
        with self._lock:
            if self._sealed:
                raise CircuitBreakerError("circuit-breaker market is sealed")
            _validate_command_identity(command)
            self._scheduled.append(command)

    def schedule_all(self, commands: Iterable[MarketCommand]) -> None:
        """Atomically buffer a finite command collection."""
        try:
            materialized = tuple(commands)
        except Exception as error:
            raise CircuitBreakerError("commands could not be materialized") from error
        for command in materialized:
            _validate_command_identity(command)
        with self._lock:
            if self._sealed:
                raise CircuitBreakerError("circuit-breaker market is sealed")
            self._scheduled.extend(materialized)

    def run(self) -> HaltReplayArtifact:
        """Execute the deterministic command/recovery event queue and seal it."""
        with self._lock:
            if self._result is not None:
                return self._result
            if self._sealed:
                raise CircuitBreakerError("circuit-breaker market is sealed")
            self._sealed = True
            self._result = self._run_commands(tuple(self._scheduled))
            return self._result

    @classmethod
    def replay(cls, artifact: HaltReplayArtifact) -> HaltReplayArtifact:
        """Reconstruct and exactly compare a complete halt replay artifact."""
        try:
            if type(artifact) is not HaltReplayArtifact:
                raise HaltReplayError("artifact has the wrong runtime type")
            artifact.assert_integrity()
            market = cls(
                artifact.market_config,
                artifact.opening_state,
                artifact.policy,
            )
            market.schedule_all(item.command for item in artifact.attempted_commands)
            replayed = market.run()
            replayed.assert_integrity()
            if replayed.canonical_bytes != artifact.canonical_bytes:
                raise HaltReplayError("halt replay canonical bytes differ")
            return replayed
        except HaltReplayError:
            raise
        except Exception as error:
            failure = HaltReplayError("halt replay could not be verified")
            if isinstance(artifact, HaltReplayArtifact):
                failure.audit_events = artifact.halt_audit_events
            raise failure from error

    def _run_commands(self, commands: tuple[MarketCommand, ...]) -> HaltReplayArtifact:
        attempted = _scheduled_commands(commands, self._market_config.latency_steps)
        audit = MarketAuditLog()
        audit.append(
            action="initialize_circuit_breaker",
            outcome="allowed",
            rule_code=MarketRuleCode.OK,
            simulation_time=0,
            event_key=None,
            details={
                "market_config_sha256": self._market_config.sha256,
                "message": "explicitly synthetic single-market circuit breaker",
                "policy_sha256": self._policy.sha256,
            },
        )
        queue: list[_QueueEntry] = [
            (item.arrival_time, item.event_key, item.command) for item in attempted
        ]
        heapq.heapify(queue)
        queued_keys = {item.event_key for item in attempted}
        session = ReplayMarketSession(self._market_config, self._opening_state)
        accepted: list[ScheduledCommand] = []
        transitions: list[HaltTransition] = []
        observations: list[_TradeObservation] = []
        status = HaltStatus.ACTIVE
        active_trigger_key: str | None = None
        active_recovery_key: str | None = None
        halted_until: int | None = None

        while queue:
            logical_time, queued_key, payload = heapq.heappop(queue)
            if isinstance(payload, _RecoveryEvent):
                if (
                    status is not HaltStatus.HALTED
                    or active_trigger_key != payload.trigger_event_key
                    or active_recovery_key != payload.event_key.value
                    or halted_until != payload.logical_time
                ):
                    failure = HaltReplayError(
                        "recovery event does not match active halt"
                    )
                    failure.audit_events = audit.events
                    raise failure
                transition = _transition(
                    sequence=len(transitions),
                    kind=HaltTransitionKind.RECOVERY,
                    logical_time=logical_time,
                    event_key=payload.event_key,
                    trigger_event_key=payload.trigger_event_key,
                    recovery_event_key=payload.event_key.value,
                    halted_until=logical_time,
                    observed_price=None,
                    reference_price=None,
                    cause_trade_id=None,
                    policy_sha256=self._policy.sha256,
                )
                transitions.append(transition)
                status = HaltStatus.ACTIVE
                audit.append(
                    action="circuit_breaker_recovery",
                    outcome="allowed",
                    rule_code=MarketRuleCode.HALT_RECOVERED,
                    simulation_time=logical_time,
                    event_key=payload.event_key.value,
                    details={
                        "policy_sha256": self._policy.sha256,
                        "status_after": status.value,
                        "transition_sha256": transition.transition_sha256,
                        "trigger_event_key": payload.trigger_event_key,
                    },
                )
                active_trigger_key = None
                active_recovery_key = None
                halted_until = None
                continue

            command = payload
            if command.event_key.value != queued_key:
                failure = HaltReplayError("queued command event key changed")
                failure.audit_events = audit.events
                raise failure
            scheduled = ScheduledCommand(
                arrival_time=logical_time,
                event_key=queued_key,
                command=command,
            )
            if status is HaltStatus.HALTED:
                rule = self._policy.command_rule(command)
                if rule is HaltCommandRule.REJECT:
                    audit.append(
                        action=_command_action(command),
                        outcome="rejected",
                        rule_code=MarketRuleCode.HALT_COMMAND_REJECTED,
                        simulation_time=logical_time,
                        event_key=queued_key,
                        details={
                            "command_sha256": sha256_bytes(
                                canonical_json_bytes(command_to_mapping(command))
                            ),
                            "halted_until": cast(int, halted_until),
                            "policy_rule": rule.value,
                            "policy_sha256": self._policy.sha256,
                            "recovery_event_key": cast(str, active_recovery_key),
                            "status_before": status.value,
                        },
                    )
                    continue
                audit_rule = MarketRuleCode.HALT_COMMAND_ALLOWED
            else:
                audit_rule = MarketRuleCode.OK

            before_trade_count = len(session.replay.trades)
            try:
                session.submit(command)
            except Exception as error:
                audit.append(
                    action=_command_action(command),
                    outcome="rejected",
                    rule_code=MarketRuleCode.HALT_GATEWAY_CORE_REJECTED,
                    simulation_time=logical_time,
                    event_key=queued_key,
                    details={
                        "command_sha256": sha256_bytes(
                            canonical_json_bytes(command_to_mapping(command))
                        ),
                        "exception_type": type(error).__qualname__,
                        "message": "core CLOB rejected a gateway-allowed command",
                        "policy_sha256": self._policy.sha256,
                    },
                )
                failure = HaltReplayError(
                    "core CLOB rejected a gateway-allowed command; run failed closed"
                )
                failure.audit_events = audit.events
                raise failure from error
            accepted.append(scheduled)
            audit.append(
                action=_command_action(command),
                outcome="allowed",
                rule_code=audit_rule,
                simulation_time=logical_time,
                event_key=queued_key,
                details={
                    "command_sha256": sha256_bytes(
                        canonical_json_bytes(command_to_mapping(command))
                    ),
                    "core_replay_sha256": session.replay.sha256,
                    "policy_sha256": self._policy.sha256,
                    "status_before": status.value,
                },
            )

            new_trades = session.replay.trades[before_trade_count:]
            for trade in new_trades:
                if status is HaltStatus.ACTIVE:
                    reference = _reference_price(self._policy, observations, trade)
                    if _trade_triggers(self._policy, trade, reference):
                        recovery_key = derive_recovery_event_key(
                            command.event_key,
                            trigger_time=trade.simulation_time,
                            halt_duration_steps=self._policy.halt_duration_steps,
                        )
                        if recovery_key.value in queued_keys:
                            failure = HaltReplayError(
                                "derived recovery EventKey collides with another event"
                            )
                            failure.audit_events = audit.events
                            raise failure
                        recovery_time = (
                            trade.simulation_time + self._policy.halt_duration_steps
                        )
                        recovery = _RecoveryEvent(
                            logical_time=recovery_time,
                            event_key=recovery_key,
                            trigger_event_key=command.event_key.value,
                        )
                        queued_keys.add(recovery_key.value)
                        heapq.heappush(
                            queue,
                            (recovery_time, recovery_key.value, recovery),
                        )
                        transition = _transition(
                            sequence=len(transitions),
                            kind=HaltTransitionKind.TRIGGER,
                            logical_time=trade.simulation_time,
                            event_key=command.event_key,
                            trigger_event_key=command.event_key.value,
                            recovery_event_key=recovery_key.value,
                            halted_until=recovery_time,
                            observed_price=trade.price,
                            reference_price=reference,
                            cause_trade_id=trade.trade_id,
                            policy_sha256=self._policy.sha256,
                        )
                        transitions.append(transition)
                        status = HaltStatus.HALTED
                        active_trigger_key = command.event_key.value
                        active_recovery_key = recovery_key.value
                        halted_until = recovery_time
                        audit.append(
                            action="circuit_breaker_trigger",
                            outcome="allowed",
                            rule_code=MarketRuleCode.HALT_TRIGGERED,
                            simulation_time=trade.simulation_time,
                            event_key=command.event_key.value,
                            details={
                                "cause_trade_id": trade.trade_id,
                                "halted_until": recovery_time,
                                "observed_price": trade.price,
                                "policy_sha256": self._policy.sha256,
                                "recovery_event_key": recovery_key.value,
                                "reference_price": reference,
                                "status_after": status.value,
                                "threshold_bps": self._policy.trigger_threshold_bps,
                                "transition_sha256": transition.transition_sha256,
                            },
                        )
                observations.append(
                    _TradeObservation(
                        simulation_time=trade.simulation_time,
                        sequence=trade.sequence,
                        price=trade.price,
                    )
                )

        session.assert_replay()
        if status is not HaltStatus.ACTIVE:
            failure = HaltReplayError("event queue ended before deterministic recovery")
            failure.audit_events = audit.events
            raise failure
        return _artifact(
            policy=self._policy,
            market_config=self._market_config,
            opening_state=self._opening_state,
            attempted=attempted,
            accepted=tuple(accepted),
            transitions=tuple(transitions),
            audit_events=audit.events,
            final_market_replay=session.replay,
        )


def derive_recovery_event_key(
    trigger_event_key: EventKey,
    *,
    trigger_time: int,
    halt_duration_steps: int,
) -> EventKey:
    """Derive the role-neutral deterministic key for one recovery event."""
    if not _valid_event_key(trigger_event_key):
        raise HaltPolicyError("trigger_event_key is invalid")
    for field, value in (
        ("trigger_time", trigger_time),
        ("halt_duration_steps", halt_duration_steps),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise HaltPolicyError(f"{field} must be a non-negative integer")
    if halt_duration_steps == 0:
        raise HaltPolicyError("halt_duration_steps must be positive")
    return EventKey(
        experiment_id=trigger_event_key.experiment_id,
        seed=trigger_event_key.seed,
        replicate_id=trigger_event_key.replicate_id,
        simulation_step=trigger_time + halt_duration_steps,
        purpose=f"circuit-breaker-recovery:{trigger_event_key.value}",
    )


def _artifact(
    *,
    policy: HaltPolicy,
    market_config: MarketConfig,
    opening_state: OpeningState,
    attempted: tuple[ScheduledCommand, ...],
    accepted: tuple[ScheduledCommand, ...],
    transitions: tuple[HaltTransition, ...],
    audit_events: tuple[MarketAuditEvent, ...],
    final_market_replay: ReplayResult,
) -> HaltReplayArtifact:
    attempted_sha = _mapping_sequence_sha(item.to_mapping() for item in attempted)
    accepted_sha = _mapping_sequence_sha(item.to_mapping() for item in accepted)
    transitions_sha = _mapping_sequence_sha(item.to_mapping() for item in transitions)
    audit_sha = _mapping_sequence_sha(item.to_mapping() for item in audit_events)
    provisional = HaltReplayArtifact(
        policy=policy,
        market_config=market_config,
        opening_state=opening_state,
        attempted_commands=attempted,
        accepted_commands=accepted,
        transitions=transitions,
        halt_audit_events=audit_events,
        final_market_replay=final_market_replay,
        policy_sha256=policy.sha256,
        attempted_commands_sha256=attempted_sha,
        accepted_commands_sha256=accepted_sha,
        transitions_sha256=transitions_sha,
        halt_audit_sha256=audit_sha,
        final_market_replay_sha256=final_market_replay.sha256,
        artifact_sha256="0" * 64,
    )
    result = replace(
        provisional,
        artifact_sha256=sha256_bytes(canonical_json_bytes(provisional.core_mapping())),
    )
    result.assert_integrity()
    return result


def _transition(
    *,
    sequence: int,
    kind: HaltTransitionKind,
    logical_time: int,
    event_key: EventKey,
    trigger_event_key: str,
    recovery_event_key: str,
    halted_until: int,
    observed_price: int | None,
    reference_price: int | None,
    cause_trade_id: str | None,
    policy_sha256: str,
) -> HaltTransition:
    status = (
        HaltStatus.HALTED if kind is HaltTransitionKind.TRIGGER else HaltStatus.ACTIVE
    )
    provisional = HaltTransition(
        sequence=sequence,
        kind=kind,
        status_after=status,
        logical_time=logical_time,
        event_key=event_key,
        trigger_event_key=trigger_event_key,
        recovery_event_key=recovery_event_key,
        halted_until=halted_until,
        observed_price=observed_price,
        reference_price=reference_price,
        cause_trade_id=cause_trade_id,
        policy_sha256=policy_sha256,
        transition_sha256="0" * 64,
    )
    result = replace(
        provisional,
        transition_sha256=sha256_bytes(
            canonical_json_bytes(provisional.core_mapping())
        ),
    )
    result.assert_integrity()
    return result


def _scheduled_commands(
    commands: tuple[MarketCommand, ...], latency_steps: int
) -> tuple[ScheduledCommand, ...]:
    scheduled: list[ScheduledCommand] = []
    event_keys: set[str] = set()
    for command in commands:
        _validate_command_identity(command)
        event_key = command.event_key.value
        if event_key in event_keys:
            raise CircuitBreakerError("scheduled event keys must be globally unique")
        event_keys.add(event_key)
        scheduled.append(
            ScheduledCommand(
                arrival_time=command.submitted_at + latency_steps,
                event_key=event_key,
                command=command,
            )
        )
    scheduled.sort(key=lambda item: (item.arrival_time, item.event_key))
    return tuple(scheduled)


def _assert_scheduled_order(
    commands: tuple[ScheduledCommand, ...], latency_steps: int
) -> None:
    keys = tuple((item.arrival_time, item.event_key) for item in commands)
    if keys != tuple(sorted(keys)):
        raise HaltReplayError("scheduled commands are not deterministically ordered")
    for item in commands:
        if (
            item.event_key != item.command.event_key.value
            or item.arrival_time != item.command.submitted_at + latency_steps
        ):
            raise HaltReplayError("scheduled command identity or arrival is invalid")


def _reference_price(
    policy: HaltPolicy,
    observations: list[_TradeObservation],
    trade: TradeLogEntry,
) -> int:
    if policy.reference_mode is HaltReferenceMode.FIXED:
        return policy.reference_price
    left = trade.simulation_time - policy.rolling_window_steps
    eligible = (
        item
        for item in observations
        if left <= item.simulation_time < trade.simulation_time
    )
    first = min(
        eligible, key=lambda item: (item.simulation_time, item.sequence), default=None
    )
    return policy.reference_price if first is None else first.price


def _trade_triggers(
    policy: HaltPolicy, trade: TradeLogEntry, reference_price: int
) -> bool:
    if not (policy.monitoring_start <= trade.simulation_time <= policy.monitoring_end):
        return False
    absolute_move_scaled = abs(trade.price - reference_price) * 10_000
    threshold_scaled = reference_price * policy.trigger_threshold_bps
    return absolute_move_scaled >= threshold_scaled


def _validate_command_identity(command: MarketCommand) -> None:
    if not isinstance(command, LimitOrder | MarketOrder | CancelOrder):
        raise CircuitBreakerError("unsupported market command type")
    if not _valid_event_key(command.event_key):
        raise CircuitBreakerError("command EventKey is invalid")
    if (
        isinstance(command.submitted_at, bool)
        or not isinstance(command.submitted_at, int)
        or command.submitted_at < 0
    ):
        raise CircuitBreakerError("submitted_at must be a non-negative integer")


def _valid_event_key(event_key: object) -> bool:
    return (
        isinstance(event_key, EventKey)
        and event_key.schema_version == EVENT_KEY_SCHEMA_VERSION
        and is_stable_identifier(event_key.experiment_id)
        and isinstance(event_key.seed, str)
        and bool(event_key.seed)
        and len(event_key.seed) <= 512
        and isinstance(event_key.replicate_id, int)
        and not isinstance(event_key.replicate_id, bool)
        and event_key.replicate_id >= 0
        and isinstance(event_key.simulation_step, int)
        and not isinstance(event_key.simulation_step, bool)
        and event_key.simulation_step >= 0
        and isinstance(event_key.purpose, str)
        and bool(event_key.purpose)
        and len(event_key.purpose) <= 512
    )


def _valid_event_key_value(value: object) -> bool:
    return (
        isinstance(value, str)
        and value.startswith(_EVENT_KEY_PREFIX)
        and _SHA256.fullmatch(value.removeprefix(_EVENT_KEY_PREFIX)) is not None
    )


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _mapping_sequence_sha(values: Iterable[Mapping[str, object]]) -> str:
    return sha256_bytes(canonical_json_bytes(list(values)))


def _command_action(command: MarketCommand) -> str:
    if isinstance(command, LimitOrder):
        return "circuit_breaker_limit_order"
    if isinstance(command, MarketOrder):
        return "circuit_breaker_market_order"
    return "circuit_breaker_cancel_order"


__all__ = [
    "CIRCUIT_BREAKER_SCHEMA_VERSION",
    "CircuitBreakerError",
    "HaltCommandRule",
    "HaltPolicy",
    "HaltPolicyError",
    "HaltReferenceMode",
    "HaltReplayArtifact",
    "HaltReplayError",
    "HaltStatus",
    "HaltTransition",
    "HaltTransitionKind",
    "MonitoringBoundary",
    "RecoveryOrdering",
    "RollingWindowBoundary",
    "SyntheticCircuitBreakerMarket",
    "ThresholdBoundary",
    "derive_recovery_event_key",
]
