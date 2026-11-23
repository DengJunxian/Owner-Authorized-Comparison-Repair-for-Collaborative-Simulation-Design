"""Frozen synthetic benchmark for paired execution in branching CLOB markets.

The benchmark compares four random-number couplings inside the same scripted
multi-agent limit-order market.  Six policy branches exercise order omission,
partial fill, cancellation, circuit breaking, activity exit, and latency
reordering.  A separate fault-injection suite measures the incremental
detection supplied by receipts, replay, and a fail-closed evidence gate.

All outputs are synthetic.  Event-keyed execution defines the semantic
coupling reference; it does not identify a real-world joint counterfactual.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from statistics import fmean, median
from typing import NoReturn, Self, cast

from policytwin.counterfactual import EventKey
from policytwin.market import (
    AccountBalance,
    CancelOrder,
    CentralLimitOrderBook,
    HaltCommandRule,
    HaltPolicy,
    HaltReferenceMode,
    MarketCommand,
    MarketConfig,
    MarketOrder,
    OpeningOrder,
    OpeningState,
    OrderStatus,
    QuoteLogEntry,
    Side,
    SyntheticCircuitBreakerMarket,
    TradeLogEntry,
)
from policytwin.provenance import canonical_json_bytes, sha256_bytes

PAIRED_EXECUTION_BENCHMARK_SCHEMA_VERSION = "policytwin.paired-execution-benchmark.v1"
PAIRED_EXECUTION_BENCHMARK_CLAIM_SCOPE = (
    "synthetic_clob_coupling_and_audit_layer_evaluation_only"
)
_AGENTS = (
    "fundamental",
    "trend",
    "maker-a",
    "maker-b",
    "liquidity",
    "entrant",
)
_AFFECTED_AGENT = {
    "order_prohibition": "trend",
    "partial_fill": "liquidity",
    "cancellation": "maker-a",
    "circuit_breaker": "fundamental",
    "agent_activity_exit": "entrant",
    "latency_reorder": "maker-b",
}
_DRAW_FIELDS = ("side", "quantity")
_T_CRITICAL_95_DF31 = 2.039513446


class PairedExecutionBenchmarkError(RuntimeError):
    """Raised when the frozen benchmark contract is violated."""


class CouplingMode(StrEnum):
    """Evaluated cross-world random-number coupling."""

    EVENT_KEYED = "event_keyed"
    SHARED_SEQUENTIAL = "shared_sequential"
    PER_AGENT_SUBSTREAM = "per_agent_substream"
    INDEPENDENT = "independent"


class BranchScenario(StrEnum):
    """Supported policy-sensitive market branch."""

    ORDER_PROHIBITION = "order_prohibition"
    PARTIAL_FILL = "partial_fill"
    CANCELLATION = "cancellation"
    CIRCUIT_BREAKER = "circuit_breaker"
    AGENT_ACTIVITY_EXIT = "agent_activity_exit"
    LATENCY_REORDER = "latency_reorder"


class AuditLayer(StrEnum):
    """Cumulative evidence layers in the component study."""

    EVENT_KEY_ONLY = "event_key_only"
    CONSUMPTION_RECEIPTS = "event_key_plus_receipts"
    DETERMINISTIC_REPLAY = "event_key_receipts_replay"
    FULL_EVIDENCE_GATE = "full_evidence_gate"


class InjectedFault(StrEnum):
    """Registered faults used to test incremental detection."""

    DRAW_SHIFT = "draw_shift"
    DUPLICATE_KEY = "duplicate_key"
    MISSING_COMMON_EVENT = "missing_common_event"
    OUTPUT_MUTATION = "output_mutation"
    COMMAND_REORDER = "command_reorder"
    POST_CUTOFF_RECORD = "post_cutoff_record"
    MISSING_REQUIRED_METRIC = "missing_required_metric"


def _fail(message: str) -> NoReturn:
    raise PairedExecutionBenchmarkError(message)


def _strict_fields(
    value: Mapping[str, object], expected: frozenset[str], field: str
) -> None:
    if frozenset(value) != expected:
        _fail(f"{field} has missing or unknown fields")


def _integer(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"{field} must be an integer >= {minimum}")
    return value


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{field} must be a non-empty string")
    return value


def _string_sequence(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        _fail(f"{field} must be an array")
    result = tuple(_string(item, field) for item in value)
    if len(result) != len(set(result)):
        _fail(f"{field} must not contain duplicates")
    return result


@dataclass(frozen=True, slots=True)
class PairedExecutionBenchmarkPlan:
    """Immutable workload and analysis registration."""

    experiment_id: str
    seed: str
    replicate_count: int
    step_count: int
    branch_step: int
    batch_count: int
    batch_size: int
    fault_repetitions: int
    branch_scenarios: tuple[str, ...]
    coupling_modes: tuple[str, ...]
    audit_layers: tuple[str, ...]
    injected_faults: tuple[str, ...]
    source_snapshot_manifest_sha256: str
    schema_version: str = PAIRED_EXECUTION_BENCHMARK_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PAIRED_EXECUTION_BENCHMARK_SCHEMA_VERSION:
            _fail("unsupported benchmark schema version")
        _string(self.experiment_id, "experiment_id")
        _string(self.seed, "seed")
        _integer(self.replicate_count, "replicate_count", minimum=32)
        _integer(self.step_count, "step_count", minimum=8)
        _integer(self.branch_step, "branch_step", minimum=2)
        if self.branch_step >= self.step_count:
            _fail("branch_step must precede the final market step")
        _integer(self.batch_count, "batch_count", minimum=32)
        _integer(self.batch_size, "batch_size", minimum=8)
        if self.batch_size != 32:
            _fail("batch_size must equal 32 for the frozen t interval")
        _integer(self.fault_repetitions, "fault_repetitions", minimum=8)
        if self.branch_scenarios != tuple(item.value for item in BranchScenario):
            _fail("branch_scenarios must equal the complete frozen enum order")
        if self.coupling_modes != tuple(item.value for item in CouplingMode):
            _fail("coupling_modes must equal the complete frozen enum order")
        if self.audit_layers != tuple(item.value for item in AuditLayer):
            _fail("audit_layers must equal the complete frozen enum order")
        if self.injected_faults != tuple(item.value for item in InjectedFault):
            _fail("injected_faults must equal the complete frozen enum order")
        if len(self.source_snapshot_manifest_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.source_snapshot_manifest_sha256
        ):
            _fail("source_snapshot_manifest_sha256 must be a lowercase digest")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(value, frozenset(cls.__dataclass_fields__), "benchmark plan")
        return cls(
            experiment_id=_string(value["experiment_id"], "experiment_id"),
            seed=_string(value["seed"], "seed"),
            replicate_count=_integer(
                value["replicate_count"], "replicate_count", minimum=32
            ),
            step_count=_integer(value["step_count"], "step_count", minimum=8),
            branch_step=_integer(value["branch_step"], "branch_step", minimum=2),
            batch_count=_integer(value["batch_count"], "batch_count", minimum=32),
            batch_size=_integer(value["batch_size"], "batch_size", minimum=8),
            fault_repetitions=_integer(
                value["fault_repetitions"], "fault_repetitions", minimum=8
            ),
            branch_scenarios=_string_sequence(
                value["branch_scenarios"], "branch_scenarios"
            ),
            coupling_modes=_string_sequence(value["coupling_modes"], "coupling_modes"),
            audit_layers=_string_sequence(value["audit_layers"], "audit_layers"),
            injected_faults=_string_sequence(
                value["injected_faults"], "injected_faults"
            ),
            source_snapshot_manifest_sha256=_string(
                value["source_snapshot_manifest_sha256"],
                "source_snapshot_manifest_sha256",
            ),
            schema_version=_string(value["schema_version"], "schema_version"),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "audit_layers": list(self.audit_layers),
            "batch_count": self.batch_count,
            "batch_size": self.batch_size,
            "branch_scenarios": list(self.branch_scenarios),
            "branch_step": self.branch_step,
            "coupling_modes": list(self.coupling_modes),
            "experiment_id": self.experiment_id,
            "fault_repetitions": self.fault_repetitions,
            "injected_faults": list(self.injected_faults),
            "replicate_count": self.replicate_count,
            "schema_version": self.schema_version,
            "seed": self.seed,
            "source_snapshot_manifest_sha256": (self.source_snapshot_manifest_sha256),
            "step_count": self.step_count,
        }

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))


@dataclass(frozen=True, slots=True)
class _DrawEvent:
    step: int
    agent_id: str
    field: str

    @property
    def identity(self) -> str:
        return f"step={self.step}|agent={self.agent_id}|field={self.field}"


@dataclass(frozen=True, slots=True)
class _WorldResult:
    traded_notional: float
    trade_volume: float
    terminal_spread: float | None
    terminal_mid: float | None
    replay_sha256: str
    branch_activated: bool
    command_count: int


def _digest_bits(*parts: object) -> int:
    payload = canonical_json_bytes({"parts": [str(item) for item in parts]})
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _events(plan: PairedExecutionBenchmarkPlan) -> tuple[_DrawEvent, ...]:
    return tuple(
        _DrawEvent(step, agent_id, field)
        for step in range(plan.step_count)
        for agent_id in _AGENTS
        for field in _DRAW_FIELDS
    )


def _base_bits(
    plan: PairedExecutionBenchmarkPlan,
    replicate_id: int,
    scenario: BranchScenario,
    event: _DrawEvent,
    *,
    namespace: str = "shared",
) -> int:
    return _digest_bits(
        plan.seed,
        namespace,
        replicate_id,
        scenario.value,
        event.identity,
    )


def _world_draws(
    plan: PairedExecutionBenchmarkPlan,
    replicate_id: int,
    scenario: BranchScenario,
    mode: CouplingMode,
    *,
    candidate: bool,
) -> dict[str, int]:
    events = _events(plan)
    baseline = {
        event.identity: _base_bits(plan, replicate_id, scenario, event)
        for event in events
    }
    if not candidate or mode is CouplingMode.EVENT_KEYED:
        return baseline
    if mode is CouplingMode.INDEPENDENT:
        return {
            event.identity: _base_bits(
                plan, replicate_id, scenario, event, namespace="candidate-independent"
            )
            for event in events
        }
    branch_index = next(
        index for index, event in enumerate(events) if event.step >= plan.branch_step
    )
    if mode is CouplingMode.SHARED_SEQUENTIAL:
        shifted: dict[str, int] = {}
        shift = 5
        for index, event in enumerate(events):
            if index < branch_index:
                shifted[event.identity] = baseline[event.identity]
                continue
            source_index = index + shift
            if source_index < len(events):
                shifted[event.identity] = baseline[events[source_index].identity]
            else:
                shifted[event.identity] = _digest_bits(
                    plan.seed,
                    "shared-sequential-tail",
                    replicate_id,
                    scenario.value,
                    source_index,
                )
        return shifted
    if mode is CouplingMode.PER_AGENT_SUBSTREAM:
        affected = _AFFECTED_AGENT[scenario.value]
        shifted = dict(baseline)
        agent_events = [event for event in events if event.agent_id == affected]
        shift = 2
        for index, event in enumerate(agent_events):
            if event.step < plan.branch_step:
                continue
            source_index = index + shift
            if source_index < len(agent_events):
                shifted[event.identity] = baseline[agent_events[source_index].identity]
            else:
                shifted[event.identity] = _digest_bits(
                    plan.seed,
                    "per-agent-tail",
                    replicate_id,
                    scenario.value,
                    affected,
                    source_index,
                )
        return shifted
    raise AssertionError("unhandled coupling mode")


def _event_key(
    plan: PairedExecutionBenchmarkPlan,
    replicate_id: int,
    scenario: BranchScenario,
    simulation_step: int,
    purpose: str,
) -> EventKey:
    return EventKey(
        experiment_id=plan.experiment_id,
        seed=f"{plan.seed}-{scenario.value}",
        replicate_id=replicate_id,
        simulation_step=simulation_step,
        purpose=purpose,
    )


def _opening(
    plan: PairedExecutionBenchmarkPlan,
    replicate_id: int,
    scenario: BranchScenario,
) -> OpeningState:
    trader_ids = (*_AGENTS, "book-bid", "book-ask")
    accounts = tuple(
        AccountBalance(trader_id, cash=10_000_000, asset=100_000)
        for trader_id in trader_ids
    )
    orders: list[OpeningOrder] = []
    for index, (side, price, quantity, owner) in enumerate(
        (
            (Side.BUY, 90, 500, "book-bid"),
            (Side.BUY, 80, 500, "book-bid"),
            (Side.BUY, 70, 500, "book-bid"),
            (Side.SELL, 110, 500, "book-ask"),
            (Side.SELL, 120, 500, "book-ask"),
            (Side.SELL, 130, 500, "book-ask"),
            (Side.BUY, 85, 50, "maker-a"),
        )
    ):
        orders.append(
            OpeningOrder(
                event_key=_event_key(
                    plan,
                    replicate_id,
                    scenario,
                    index,
                    f"opening-{owner}-{index}",
                ),
                order_id=f"opening-{owner}-{index}",
                trader_id=owner,
                side=side,
                price=price,
                quantity=quantity,
                priority_time=-10 + index,
            )
        )
    return OpeningState(accounts=accounts, orders=tuple(orders))


def _command_key(
    plan: PairedExecutionBenchmarkPlan,
    replicate_id: int,
    scenario: BranchScenario,
    step: int,
    agent_index: int,
    purpose: str,
) -> EventKey:
    return _event_key(
        plan,
        replicate_id,
        scenario,
        1_000 + step * 100 + agent_index,
        purpose,
    )


def _base_commands(
    plan: PairedExecutionBenchmarkPlan,
    replicate_id: int,
    scenario: BranchScenario,
    draws: Mapping[str, int],
    *,
    candidate: bool,
) -> tuple[MarketCommand, ...]:
    commands: list[MarketCommand] = []
    for step in range(plan.step_count):
        for agent_index, agent_id in enumerate(_AGENTS):
            if (
                candidate
                and scenario is BranchScenario.ORDER_PROHIBITION
                and agent_id == "trend"
                and step >= plan.branch_step
            ):
                side_event = _DrawEvent(step, agent_id, "side")
                if draws[side_event.identity] % 2 == 0:
                    continue
            if (
                candidate
                and scenario is BranchScenario.AGENT_ACTIVITY_EXIT
                and agent_id == "entrant"
                and step >= plan.branch_step
            ):
                continue
            side_bits = draws[_DrawEvent(step, agent_id, "side").identity]
            quantity_bits = draws[_DrawEvent(step, agent_id, "quantity").identity]
            side = Side.BUY if side_bits % 2 == 0 else Side.SELL
            quantity = 1 + quantity_bits % 5
            submitted_at = step * 100 + agent_index
            if (
                candidate
                and scenario is BranchScenario.LATENCY_REORDER
                and agent_id == "maker-b"
                and step >= plan.branch_step
            ):
                submitted_at += 20
            commands.append(
                MarketOrder(
                    event_key=_command_key(
                        plan,
                        replicate_id,
                        scenario,
                        step,
                        agent_index,
                        f"agent-order-{step}-{agent_id}",
                    ),
                    submitted_at=submitted_at,
                    order_id=f"agent-order-{step}-{agent_id}",
                    trader_id=agent_id,
                    side=side,
                    quantity=quantity,
                )
            )
    branch_time = plan.branch_step * 100 - 2
    if scenario is BranchScenario.PARTIAL_FILL:
        commands.append(
            MarketOrder(
                event_key=_command_key(
                    plan,
                    replicate_id,
                    scenario,
                    plan.branch_step,
                    90,
                    "candidate-partial-sweep" if candidate else "baseline-small-sweep",
                ),
                submitted_at=branch_time,
                order_id=(
                    "candidate-partial-sweep" if candidate else "baseline-small-sweep"
                ),
                trader_id="liquidity",
                side=Side.BUY,
                quantity=5_000 if candidate else 5,
            )
        )
    if scenario is BranchScenario.CANCELLATION and candidate:
        commands.append(
            CancelOrder(
                event_key=_command_key(
                    plan,
                    replicate_id,
                    scenario,
                    plan.branch_step,
                    91,
                    "candidate-cancel-resting-bid",
                ),
                submitted_at=branch_time,
                trader_id="maker-a",
                target_order_id="opening-maker-a-6",
            )
        )
    return tuple(commands)


def _standard_world(
    plan: PairedExecutionBenchmarkPlan,
    replicate_id: int,
    scenario: BranchScenario,
    draws: Mapping[str, int],
    *,
    candidate: bool,
) -> _WorldResult:
    opening = _opening(plan, replicate_id, scenario)
    commands = _base_commands(plan, replicate_id, scenario, draws, candidate=candidate)
    book = CentralLimitOrderBook(MarketConfig(tick_size=1), opening)
    book.schedule_all(commands)
    replay = book.run()
    book.assert_invariants()
    replayed = CentralLimitOrderBook.replay(replay)
    if replayed.canonical_bytes != replay.canonical_bytes:
        _fail("standard CLOB replay differs")
    branch_activated = False
    if candidate and scenario is BranchScenario.ORDER_PROHIBITION:
        baseline_count = plan.step_count * len(_AGENTS)
        branch_activated = len(commands) < baseline_count
    elif candidate and scenario is BranchScenario.PARTIAL_FILL:
        record = next(
            item
            for item in replay.state.order_records
            if item.order_id == "candidate-partial-sweep"
        )
        branch_activated = (
            record.status is OrderStatus.EXPIRED
            and record.filled_quantity > 0
            and record.remaining_quantity > 0
        )
    elif candidate and scenario is BranchScenario.CANCELLATION:
        record = next(
            item
            for item in replay.state.order_records
            if item.order_id == "opening-maker-a-6"
        )
        branch_activated = record.status is OrderStatus.CANCELED
    elif candidate and scenario is BranchScenario.AGENT_ACTIVITY_EXIT:
        branch_activated = not any(
            getattr(item, "trader_id", None) == "entrant"
            and item.submitted_at >= plan.branch_step * 100
            for item in commands
        )
    elif candidate and scenario is BranchScenario.LATENCY_REORDER:
        branch_activated = any(
            item.command.trader_id == "maker-b"
            and item.arrival_time != item.command.event_key.simulation_step - 1_000
            for item in replay.scheduled_commands
        )
    return _market_result(
        replay.trades,
        replay.quotes,
        replay.sha256,
        branch_activated=branch_activated,
        command_count=len(commands),
    )


def _circuit_world(
    plan: PairedExecutionBenchmarkPlan,
    replicate_id: int,
    scenario: BranchScenario,
    draws: Mapping[str, int],
    *,
    candidate: bool,
) -> _WorldResult:
    opening = _opening(plan, replicate_id, scenario)
    commands = _base_commands(plan, replicate_id, scenario, draws, candidate=candidate)
    if candidate:
        policy = HaltPolicy(
            policy_id="synthetic-branch-benchmark-halt",
            reference_mode=HaltReferenceMode.FIXED,
            reference_price=100,
            rolling_window_steps=5,
            trigger_threshold_bps=500,
            halt_duration_steps=50,
            monitoring_start=0,
            monitoring_end=100_000,
            limit_order_during_halt=HaltCommandRule.REJECT,
            market_order_during_halt=HaltCommandRule.REJECT,
            cancel_during_halt=HaltCommandRule.ALLOW,
        )
        market = SyntheticCircuitBreakerMarket(
            MarketConfig(tick_size=1), opening, policy
        )
        market.schedule_all(commands)
        artifact = market.run()
        artifact.assert_integrity()
        replayed = SyntheticCircuitBreakerMarket.replay(artifact)
        if replayed.canonical_bytes != artifact.canonical_bytes:
            _fail("circuit-breaker replay differs")
        replay = artifact.final_market_replay
        return _market_result(
            replay.trades,
            replay.quotes,
            artifact.artifact_sha256,
            branch_activated=any(
                item.kind.value == "trigger" for item in artifact.transitions
            ),
            command_count=len(artifact.accepted_commands),
        )
    return _standard_world(plan, replicate_id, scenario, draws, candidate=False)


def _market_result(
    trades: Sequence[TradeLogEntry],
    quotes: Sequence[QuoteLogEntry],
    replay_sha256: str,
    *,
    branch_activated: bool,
    command_count: int,
) -> _WorldResult:
    notional = math.fsum(float(item.price) * float(item.quantity) for item in trades)
    volume = math.fsum(float(item.quantity) for item in trades)
    final_quote = quotes[-1]
    bid = final_quote.best_bid_price
    ask = final_quote.best_ask_price
    spread = float(ask - bid) if bid is not None and ask is not None else None
    mid = float(ask + bid) / 2.0 if bid is not None and ask is not None else None
    return _WorldResult(
        traded_notional=notional,
        trade_volume=volume,
        terminal_spread=spread,
        terminal_mid=mid,
        replay_sha256=replay_sha256,
        branch_activated=branch_activated,
        command_count=command_count,
    )


def _sample_variance(values: Sequence[float]) -> float:
    if len(values) < 2:
        return math.nan
    mean = fmean(values)
    return math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1)


def _optional_difference(
    candidate: float | None, baseline: float | None
) -> float | None:
    if candidate is None or baseline is None:
        return None
    return candidate - baseline


def _batch_indices(
    plan: PairedExecutionBenchmarkPlan, batch_id: int
) -> tuple[int, ...]:
    return tuple(
        _digest_bits(plan.seed, "batch", batch_id, index) % plan.replicate_count
        for index in range(plan.batch_size)
    )


def _summary(
    plan: PairedExecutionBenchmarkPlan,
    rows: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    oracle_means: dict[str, float] = {}
    effects: dict[tuple[str, str], list[float]] = {}
    oracle_by_replicate: dict[tuple[str, int], float] = {}
    for scenario in BranchScenario:
        for mode in CouplingMode:
            selected = [
                row
                for row in rows
                if row["scenario"] == scenario.value
                and row["coupling_mode"] == mode.value
            ]
            values = [cast(float, row["effect_traded_notional"]) for row in selected]
            effects[(scenario.value, mode.value)] = values
            if mode is CouplingMode.EVENT_KEYED:
                oracle_means[scenario.value] = fmean(values)
                for row in selected:
                    oracle_by_replicate[
                        (scenario.value, cast(int, row["replicate_id"]))
                    ] = cast(float, row["effect_traded_notional"])

    summaries: list[dict[str, object]] = []
    coverage: list[dict[str, object]] = []
    for scenario in BranchScenario:
        target = oracle_means[scenario.value]
        for mode in CouplingMode:
            values = effects[(scenario.value, mode.value)]
            errors = [
                value - oracle_by_replicate[(scenario.value, replicate_id)]
                for replicate_id, value in enumerate(values)
            ]
            mismatches = [
                cast(int, row["shock_mismatch_count"])
                for row in rows
                if row["scenario"] == scenario.value
                and row["coupling_mode"] == mode.value
            ]
            runtimes = [
                cast(int, row["runtime_ns"])
                for row in rows
                if row["scenario"] == scenario.value
                and row["coupling_mode"] == mode.value
            ]
            summaries.append(
                {
                    "bias_vs_event_keyed_mean": fmean(values) - target,
                    "coupling_mode": mode.value,
                    "effect_mean": fmean(values),
                    "effect_sample_variance": _sample_variance(values),
                    "median_runtime_ns": int(median(runtimes)),
                    "replicate_count": len(values),
                    "rmse_vs_event_keyed_replicate": math.sqrt(
                        fmean([error * error for error in errors])
                    ),
                    "scenario": scenario.value,
                    "shock_mismatch_rate": math.fsum(mismatches)
                    / (
                        len(mismatches)
                        * plan.step_count
                        * len(_AGENTS)
                        * len(_DRAW_FIELDS)
                    ),
                }
            )
            covered = 0
            direction_correct = 0
            intervals: list[list[float]] = []
            for batch_id in range(plan.batch_count):
                sample = [values[index] for index in _batch_indices(plan, batch_id)]
                sample_mean = fmean(sample)
                se = math.sqrt(_sample_variance(sample) / len(sample))
                interval = [
                    sample_mean - _T_CRITICAL_95_DF31 * se,
                    sample_mean + _T_CRITICAL_95_DF31 * se,
                ]
                intervals.append(interval)
                covered += interval[0] <= target <= interval[1]
                direction_correct += (
                    (sample_mean > 0 and target > 0)
                    or (sample_mean < 0 and target < 0)
                    or (sample_mean == 0 and target == 0)
                )
            coverage.append(
                {
                    "batch_count": plan.batch_count,
                    "batch_size": plan.batch_size,
                    "ci_coverage": covered / plan.batch_count,
                    "ci_definition": "two_sided_95pct_student_t_df31",
                    "coupling_mode": mode.value,
                    "effect_direction_accuracy": direction_correct / plan.batch_count,
                    "oracle_mean": target,
                    "scenario": scenario.value,
                }
            )

    oracle_order = tuple(
        sorted(
            (scenario.value for scenario in BranchScenario),
            key=lambda item: (oracle_means[item], item),
        )
    )
    ranking: list[dict[str, object]] = []
    for mode in CouplingMode:
        exact = 0
        concordant = 0
        total_pairs = 0
        for batch_id in range(plan.batch_count):
            indices = _batch_indices(plan, batch_id)
            batch_means = {
                scenario.value: fmean(
                    [effects[(scenario.value, mode.value)][index] for index in indices]
                )
                for scenario in BranchScenario
            }
            order = tuple(
                sorted(batch_means, key=lambda item: (batch_means[item], item))
            )
            exact += order == oracle_order
            for left_index, left in enumerate(oracle_order):
                for right in oracle_order[left_index + 1 :]:
                    concordant += order.index(left) < order.index(right)
                    total_pairs += 1
        ranking.append(
            {
                "batch_count": plan.batch_count,
                "batch_size": plan.batch_size,
                "coupling_mode": mode.value,
                "exact_policy_ranking_rate": exact / plan.batch_count,
                "pairwise_ranking_concordance": concordant / total_pairs,
                "reference_order_low_to_high_effect": list(oracle_order),
            }
        )
    return summaries, coverage, ranking


@dataclass(frozen=True, slots=True)
class _AuditEvidence:
    expected_events: tuple[str, ...]
    observed_events: tuple[tuple[str, int], ...]
    output_sha256: str
    replay_sha256: str
    command_order_sha256: str
    replay_command_order_sha256: str
    cutoff: datetime
    available_time: datetime
    required_metric_present: bool


def _clean_evidence(
    plan: PairedExecutionBenchmarkPlan, repetition: int
) -> _AuditEvidence:
    expected = tuple(f"event-{repetition}-{index}" for index in range(12))
    observed = tuple(
        (event, _digest_bits(plan.seed, "fault-clean", event)) for event in expected
    )
    output = sha256_bytes(canonical_json_bytes({"value": repetition}))
    order = sha256_bytes(canonical_json_bytes(list(expected)))
    cutoff = datetime(2026, 1, 1, tzinfo=UTC)
    return _AuditEvidence(
        expected_events=expected,
        observed_events=observed,
        output_sha256=output,
        replay_sha256=output,
        command_order_sha256=order,
        replay_command_order_sha256=order,
        cutoff=cutoff,
        available_time=cutoff - timedelta(days=1),
        required_metric_present=True,
    )


def _inject(evidence: _AuditEvidence, fault: InjectedFault) -> _AuditEvidence:
    values = {
        field: getattr(evidence, field) for field in evidence.__dataclass_fields__
    }
    observed = list(evidence.observed_events)
    if fault is InjectedFault.DRAW_SHIFT:
        event, bits = observed[-1]
        observed[-1] = (event, bits ^ 1)
        values["observed_events"] = tuple(observed)
    elif fault is InjectedFault.DUPLICATE_KEY:
        observed[-1] = observed[-2]
        values["observed_events"] = tuple(observed)
    elif fault is InjectedFault.MISSING_COMMON_EVENT:
        values["observed_events"] = tuple(observed[:-1])
    elif fault is InjectedFault.OUTPUT_MUTATION:
        values["output_sha256"] = "1" * 64
    elif fault is InjectedFault.COMMAND_REORDER:
        values["command_order_sha256"] = "2" * 64
    elif fault is InjectedFault.POST_CUTOFF_RECORD:
        values["available_time"] = evidence.cutoff + timedelta(seconds=1)
    elif fault is InjectedFault.MISSING_REQUIRED_METRIC:
        values["required_metric_present"] = False
    return _AuditEvidence(**values)


def _audit_detects(
    plan: PairedExecutionBenchmarkPlan,
    layer: AuditLayer,
    evidence: _AuditEvidence,
) -> bool:
    if layer is AuditLayer.EVENT_KEY_ONLY:
        return False
    observed_keys = tuple(key for key, _ in evidence.observed_events)
    receipt_failure = (
        observed_keys != evidence.expected_events
        or len(observed_keys) != len(set(observed_keys))
        or any(
            bits != _digest_bits(plan.seed, "fault-clean", key)
            for key, bits in evidence.observed_events
        )
    )
    if receipt_failure:
        return True
    if layer is AuditLayer.CONSUMPTION_RECEIPTS:
        return False
    replay_failure = (
        evidence.output_sha256 != evidence.replay_sha256
        or evidence.command_order_sha256 != evidence.replay_command_order_sha256
    )
    if replay_failure:
        return True
    if layer is AuditLayer.DETERMINISTIC_REPLAY:
        return False
    return (
        evidence.available_time > evidence.cutoff
        or not evidence.required_metric_present
    )


def _fault_study(plan: PairedExecutionBenchmarkPlan) -> dict[str, object]:
    results: list[dict[str, object]] = []
    clean_false_positives: list[dict[str, object]] = []
    for layer in AuditLayer:
        false_positives = 0
        for repetition in range(plan.fault_repetitions):
            false_positives += _audit_detects(
                plan, layer, _clean_evidence(plan, repetition)
            )
        clean_false_positives.append(
            {
                "audit_layer": layer.value,
                "clean_false_positive_count": false_positives,
                "clean_repetition_count": plan.fault_repetitions,
            }
        )
        for fault in InjectedFault:
            detected = 0
            for repetition in range(plan.fault_repetitions):
                evidence = _inject(_clean_evidence(plan, repetition), fault)
                detected += _audit_detects(plan, layer, evidence)
            results.append(
                {
                    "audit_layer": layer.value,
                    "detected_count": detected,
                    "fault": fault.value,
                    "repetition_count": plan.fault_repetitions,
                }
            )

    representative = _clean_evidence(plan, 0)
    base = canonical_json_bytes({"event_keys": list(representative.expected_events)})
    receipts = canonical_json_bytes(
        {
            "event_keys": list(representative.expected_events),
            "receipts": list(representative.observed_events),
        }
    )
    replay = canonical_json_bytes(
        {
            "event_keys": list(representative.expected_events),
            "receipts": list(representative.observed_events),
            "replay": {
                "command_order_sha256": representative.command_order_sha256,
                "output_sha256": representative.output_sha256,
            },
        }
    )
    gate = canonical_json_bytes(
        {
            "available_time": representative.available_time.isoformat(),
            "cutoff": representative.cutoff.isoformat(),
            "event_keys": list(representative.expected_events),
            "receipts": list(representative.observed_events),
            "replay": {
                "command_order_sha256": representative.command_order_sha256,
                "output_sha256": representative.output_sha256,
            },
            "required_metric_present": representative.required_metric_present,
        }
    )
    return {
        "clean_false_positives": clean_false_positives,
        "fault_detection": results,
        "storage_bytes_per_representative_run": [
            {"audit_layer": AuditLayer.EVENT_KEY_ONLY.value, "bytes": len(base)},
            {
                "audit_layer": AuditLayer.CONSUMPTION_RECEIPTS.value,
                "bytes": len(receipts),
            },
            {
                "audit_layer": AuditLayer.DETERMINISTIC_REPLAY.value,
                "bytes": len(replay),
            },
            {
                "audit_layer": AuditLayer.FULL_EVIDENCE_GATE.value,
                "bytes": len(gate),
            },
        ],
    }


def run_paired_execution_benchmark(
    plan: PairedExecutionBenchmarkPlan,
) -> dict[str, object]:
    """Run the complete registered synthetic benchmark."""
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for scenario in BranchScenario:
        for replicate_id in range(plan.replicate_count):
            baseline_draws = _world_draws(
                plan,
                replicate_id,
                scenario,
                CouplingMode.EVENT_KEYED,
                candidate=False,
            )
            for mode in CouplingMode:
                started = time.perf_counter_ns()
                try:
                    candidate_draws = _world_draws(
                        plan,
                        replicate_id,
                        scenario,
                        mode,
                        candidate=True,
                    )
                    if scenario is BranchScenario.CIRCUIT_BREAKER:
                        baseline = _circuit_world(
                            plan,
                            replicate_id,
                            scenario,
                            baseline_draws,
                            candidate=False,
                        )
                        candidate = _circuit_world(
                            plan,
                            replicate_id,
                            scenario,
                            candidate_draws,
                            candidate=True,
                        )
                    else:
                        baseline = _standard_world(
                            plan,
                            replicate_id,
                            scenario,
                            baseline_draws,
                            candidate=False,
                        )
                        candidate = _standard_world(
                            plan,
                            replicate_id,
                            scenario,
                            candidate_draws,
                            candidate=True,
                        )
                    elapsed = time.perf_counter_ns() - started
                    mismatches = sum(
                        baseline_draws[key] != candidate_draws[key]
                        for key in baseline_draws
                    )
                    row: dict[str, object] = {
                        "baseline_command_count": baseline.command_count,
                        "baseline_replay_sha256": baseline.replay_sha256,
                        "branch_activated": candidate.branch_activated,
                        "candidate_command_count": candidate.command_count,
                        "candidate_replay_sha256": candidate.replay_sha256,
                        "coupling_mode": mode.value,
                        "effect_terminal_mid": _optional_difference(
                            candidate.terminal_mid, baseline.terminal_mid
                        ),
                        "effect_terminal_spread": _optional_difference(
                            candidate.terminal_spread, baseline.terminal_spread
                        ),
                        "effect_trade_volume": (
                            candidate.trade_volume - baseline.trade_volume
                        ),
                        "effect_traded_notional": (
                            candidate.traded_notional - baseline.traded_notional
                        ),
                        "replicate_id": replicate_id,
                        "runtime_ns": elapsed,
                        "scenario": scenario.value,
                        "shock_mismatch_count": mismatches,
                        "shock_registered_count": len(baseline_draws),
                    }
                    row["record_sha256"] = sha256_bytes(canonical_json_bytes(row))
                    rows.append(row)
                except Exception as error:
                    failures.append(
                        {
                            "coupling_mode": mode.value,
                            "error_type": type(error).__name__,
                            "message_sha256": sha256_bytes(str(error).encode("utf-8")),
                            "replicate_id": replicate_id,
                            "scenario": scenario.value,
                        }
                    )
    expected_rows = plan.replicate_count * len(BranchScenario) * len(CouplingMode)
    if failures or len(rows) != expected_rows:
        _fail("registered market benchmark has failed or missing rows")
    activation = {
        scenario.value: sum(
            cast(bool, row["branch_activated"])
            for row in rows
            if row["scenario"] == scenario.value
            and row["coupling_mode"] == CouplingMode.EVENT_KEYED.value
        )
        for scenario in BranchScenario
    }
    if any(count == 0 for count in activation.values()):
        _fail("at least one registered branch never activated")
    summaries, coverage, ranking = _summary(plan, rows)
    fault_injection = _fault_study(plan)
    scientific_core = {
        "branch_activation_count_event_keyed": activation,
        "claim_scope": PAIRED_EXECUTION_BENCHMARK_CLAIM_SCOPE,
        "config_sha256": plan.sha256,
        "coupling_summaries": [
            {key: value for key, value in summary.items() if key != "median_runtime_ns"}
            for summary in summaries
        ],
        "failures": failures,
        "fault_injection": fault_injection,
        "market_rows": [
            {key: value for key, value in row.items() if key != "runtime_ns"}
            for row in rows
        ],
        "policy_ranking": ranking,
        "registered_market_row_count": expected_rows,
        "sampled_ci_coverage": coverage,
        "schema_version": PAIRED_EXECUTION_BENCHMARK_SCHEMA_VERSION,
        "source_snapshot_manifest_sha256": (plan.source_snapshot_manifest_sha256),
        "synthetic_only": True,
    }
    artifact: dict[str, object] = {
        "branch_activation_count_event_keyed": activation,
        "claim_scope": PAIRED_EXECUTION_BENCHMARK_CLAIM_SCOPE,
        "config_sha256": plan.sha256,
        "coupling_summaries": summaries,
        "failures": failures,
        "fault_injection": fault_injection,
        "market_rows": rows,
        "policy_ranking": ranking,
        "registered_market_row_count": expected_rows,
        "sampled_ci_coverage": coverage,
        "schema_version": PAIRED_EXECUTION_BENCHMARK_SCHEMA_VERSION,
        "scientific_core_sha256": sha256_bytes(canonical_json_bytes(scientific_core)),
        "source_snapshot_manifest_sha256": (plan.source_snapshot_manifest_sha256),
        "synthetic_only": True,
        "timing_interpretation": (
            "machine_specific_wall_clock_diagnostic_excluded_from_"
            "scientific_core_sha256"
        ),
    }
    artifact["artifact_sha256"] = sha256_bytes(canonical_json_bytes(artifact))
    return artifact


__all__ = [
    "AuditLayer",
    "BranchScenario",
    "CouplingMode",
    "InjectedFault",
    "PAIRED_EXECUTION_BENCHMARK_CLAIM_SCOPE",
    "PAIRED_EXECUTION_BENCHMARK_SCHEMA_VERSION",
    "PairedExecutionBenchmarkError",
    "PairedExecutionBenchmarkPlan",
    "run_paired_execution_benchmark",
]
