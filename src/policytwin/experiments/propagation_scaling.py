"""Population-scale robustness for target-local RNG drift in the same CLOB."""

from __future__ import annotations

import hashlib
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, NoReturn, Self, cast

from policytwin.counterfactual import EventKey
from policytwin.market.models import (
    AccountBalance,
    CancelOrder,
    MarketCommand,
    MarketConfig,
    MarketOrder,
    MarketSnapshot,
    OpeningOrder,
    OpeningState,
    Side,
    command_to_mapping,
)
from policytwin.market.session import ReplayMarketSession
from policytwin.provenance import canonical_json_bytes, sha256_bytes

PROPAGATION_SCALING_SCHEMA_VERSION = "policytwin.propagation-scaling.v1"
SCENARIOS = (
    "rejected_order_retry",
    "partial_fill_remainder",
    "cancellation_replacement",
)
TARGET_POSITIONS = ("first", "middle", "last")
DRAW_ROLES = ("side", "quantity")


class ScalingCoupling(StrEnum):
    EVENT_KEYED = "event_keyed"
    PER_AGENT_SUBSTREAM = "per_agent_substream"
    SHARED_SEQUENTIAL = "shared_sequential"


class PropagationScalingError(RuntimeError):
    """Raised when the frozen scaling study cannot execute exactly."""


def _fail(message: str) -> NoReturn:
    raise PropagationScalingError(message)


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{field} must be a non-empty string")
    return value


def _integer(value: object, field: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"{field} must be an integer >= {minimum}")
    return value


def _as_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail("artifact value must be an integer")
    return value


def _as_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        _fail("artifact value must be numeric")
    return float(value)


@dataclass(frozen=True, slots=True)
class PropagationScalingPlan:
    experiment_id: str
    seed: str
    agent_counts: tuple[int, ...]
    target_positions: tuple[str, ...]
    scenarios: tuple[str, ...]
    coupling_modes: tuple[str, ...]
    replicate_count: int
    step_count: int
    branch_epoch: int
    propagation_rate_threshold: float
    metrics: tuple[str, ...]
    expected_claims: tuple[str, ...]
    schema_version: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        if frozenset(value) != frozenset(cls.__dataclass_fields__):
            _fail("scaling config has missing or unknown fields")
        sequence_fields = (
            "agent_counts",
            "target_positions",
            "scenarios",
            "coupling_modes",
            "metrics",
            "expected_claims",
        )
        sequences: dict[str, Sequence[object]] = {}
        for field in sequence_fields:
            raw = value[field]
            if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
                _fail(f"{field} must be an array")
            sequences[field] = cast(Sequence[object], raw)
        threshold = value["propagation_rate_threshold"]
        if isinstance(threshold, bool) or not isinstance(threshold, int | float):
            _fail("propagation_rate_threshold must be numeric")
        plan = cls(
            experiment_id=_string(value["experiment_id"], "experiment_id"),
            seed=_string(value["seed"], "seed"),
            agent_counts=tuple(
                _integer(item, "agent_count", 4) for item in sequences["agent_counts"]
            ),
            target_positions=tuple(
                _string(item, "target_position")
                for item in sequences["target_positions"]
            ),
            scenarios=tuple(
                _string(item, "scenario") for item in sequences["scenarios"]
            ),
            coupling_modes=tuple(
                _string(item, "coupling_mode") for item in sequences["coupling_modes"]
            ),
            replicate_count=_integer(value["replicate_count"], "replicate_count", 1),
            step_count=_integer(value["step_count"], "step_count", 6),
            branch_epoch=_integer(value["branch_epoch"], "branch_epoch", 1),
            propagation_rate_threshold=float(threshold),
            metrics=tuple(_string(item, "metric") for item in sequences["metrics"]),
            expected_claims=tuple(
                _string(item, "expected_claim") for item in sequences["expected_claims"]
            ),
            schema_version=_string(value["schema_version"], "schema_version"),
        )
        if plan.schema_version != PROPAGATION_SCALING_SCHEMA_VERSION:
            _fail("unsupported propagation scaling schema")
        if plan.agent_counts != tuple(sorted(set(plan.agent_counts))):
            _fail("agent_counts must be unique and increasing")
        if plan.target_positions != TARGET_POSITIONS:
            _fail("target_positions must equal the frozen enumeration")
        if plan.scenarios != SCENARIOS:
            _fail("scenarios must equal the frozen enumeration")
        if plan.coupling_modes != tuple(item.value for item in ScalingCoupling):
            _fail("coupling_modes must equal the frozen enumeration")
        if plan.branch_epoch >= plan.step_count - 1:
            _fail("branch_epoch must leave downstream epochs")
        if not 0.0 <= plan.propagation_rate_threshold < 1.0:
            _fail("propagation threshold must be in [0, 1)")
        return plan

    def to_mapping(self) -> dict[str, object]:
        return {
            "agent_counts": list(self.agent_counts),
            "branch_epoch": self.branch_epoch,
            "coupling_modes": list(self.coupling_modes),
            "expected_claims": list(self.expected_claims),
            "experiment_id": self.experiment_id,
            "metrics": list(self.metrics),
            "propagation_rate_threshold": self.propagation_rate_threshold,
            "replicate_count": self.replicate_count,
            "scenarios": list(self.scenarios),
            "schema_version": self.schema_version,
            "seed": self.seed,
            "step_count": self.step_count,
            "target_positions": list(self.target_positions),
        }

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))


def _agents(count: int) -> tuple[str, ...]:
    return tuple(f"agent-{index:03d}" for index in range(count))


def _target_index(count: int, position: str) -> int:
    return {"first": 0, "middle": count // 2, "last": count - 1}[position]


def _event_identity(epoch: int, agent: str, role: str) -> str:
    return f"scheduled|epoch={epoch}|agent={agent}|role={role}"


def _branch_identity(scenario: str, epoch: int, agent: str, role: str) -> str:
    return f"branch|scenario={scenario}|epoch={epoch}|agent={agent}|role={role}"


def _draw_bits(
    plan: PropagationScalingPlan,
    count: int,
    position: str,
    scenario: str,
    replicate: int,
    key: str,
) -> int:
    return int.from_bytes(
        hashlib.sha256(
            canonical_json_bytes(
                {
                    "agent_count": count,
                    "key": key,
                    "replicate": replicate,
                    "scenario": scenario,
                    "seed": plan.seed,
                    "target_position": position,
                }
            )
        ).digest()[:8],
        "big",
    )


class _Drawer:
    def __init__(
        self,
        plan: PropagationScalingPlan,
        agents: tuple[str, ...],
        target_position: str,
        scenario: str,
        replicate: int,
        mode: ScalingCoupling,
    ) -> None:
        self._plan = plan
        self._agents = agents
        self._target_position = target_position
        self._scenario = scenario
        self._replicate = replicate
        self._mode = mode
        scheduled = tuple(
            _event_identity(epoch, agent, role)
            for epoch in range(plan.step_count)
            for agent in agents
            for role in DRAW_ROLES
        )
        self._shared = scheduled
        self._by_agent = {
            agent: tuple(key for key in scheduled if f"agent={agent}|" in key)
            for agent in agents
        }
        self._shared_cursor = 0
        self._agent_cursors = {agent: 0 for agent in agents}

    def draw(self, requested: str, agent: str) -> tuple[int, str]:
        if self._mode is ScalingCoupling.EVENT_KEYED:
            source = requested
        elif self._mode is ScalingCoupling.SHARED_SEQUENTIAL:
            cursor = self._shared_cursor
            self._shared_cursor += 1
            source = (
                self._shared[cursor]
                if cursor < len(self._shared)
                else f"tail|shared|position={cursor}"
            )
        else:
            cursor = self._agent_cursors[agent]
            self._agent_cursors[agent] += 1
            stream = self._by_agent[agent]
            source = (
                stream[cursor]
                if cursor < len(stream)
                else f"tail|agent={agent}|position={cursor}"
            )
        return (
            _draw_bits(
                self._plan,
                len(self._agents),
                self._target_position,
                self._scenario,
                self._replicate,
                source,
            ),
            source,
        )


def _market_event_key(
    plan: PropagationScalingPlan,
    count: int,
    position: str,
    scenario: str,
    replicate: int,
    step: int,
    purpose: str,
) -> EventKey:
    return EventKey(
        experiment_id=plan.experiment_id,
        seed=f"{plan.seed}:{count}:{position}:{scenario}",
        replicate_id=replicate,
        simulation_step=step,
        purpose=purpose,
    )


def _opening(
    plan: PropagationScalingPlan,
    agents: tuple[str, ...],
    position: str,
    scenario: str,
    replicate: int,
    target: str,
) -> OpeningState:
    participant_ids = (*agents, "book-bid", "book-ask")
    accounts = tuple(
        AccountBalance(trader_id=item, cash=100_000_000, asset=1_000_000)
        for item in participant_ids
    )
    raw_orders: tuple[tuple[Side, int, int, str, str], ...] = (
        (Side.BUY, 90, 5_000, "book-bid", "book-bid-90"),
        (Side.BUY, 80, 5_000, "book-bid", "book-bid-80"),
        (Side.BUY, 70, 5_000, "book-bid", "book-bid-70"),
        (Side.SELL, 110, 5_000, "book-ask", "book-ask-110"),
        (Side.SELL, 120, 5_000, "book-ask", "book-ask-120"),
        (Side.SELL, 130, 5_000, "book-ask", "book-ask-130"),
    )
    if scenario == "cancellation_replacement":
        raw_orders = (*raw_orders, (Side.BUY, 60, 20, target, "target-resting"))
    orders = tuple(
        OpeningOrder(
            event_key=_market_event_key(
                plan,
                len(agents),
                position,
                scenario,
                replicate,
                index,
                f"opening-{order_id}",
            ),
            order_id=order_id,
            trader_id=owner,
            side=side,
            price=price,
            quantity=quantity,
        )
        for index, (side, price, quantity, owner, order_id) in enumerate(raw_orders)
    )
    return OpeningState(accounts=accounts, orders=orders)


def _action(
    snapshot: MarketSnapshot, side_bits: int, quantity_bits: int, agent: str
) -> tuple[Side, int]:
    bids = [order for order in snapshot.resting_orders if order.side is Side.BUY]
    asks = [order for order in snapshot.resting_orders if order.side is Side.SELL]
    best_bid = max((item.price for item in bids), default=0)
    best_ask = min((item.price for item in asks), default=0)
    bid_quantity = sum(
        item.remaining_quantity for item in bids if item.price == best_bid
    )
    ask_quantity = sum(
        item.remaining_quantity for item in asks if item.price == best_ask
    )
    account = next(item for item in snapshot.accounts if item.trader_id == agent)
    signal = (
        3 * bid_quantity
        + 5 * ask_quantity
        + best_bid
        + best_ask
        + 7 * (account.asset + account.cash // 100)
    )
    side = Side.BUY if (side_bits + signal) % 2 == 0 else Side.SELL
    quantity = 1 + (quantity_bits + signal) % 9
    return side, quantity


def _market_order(
    plan: PropagationScalingPlan,
    count: int,
    position: str,
    scenario: str,
    replicate: int,
    epoch: int,
    agent_index: int,
    slot: int,
    agent: str,
    side: Side,
    quantity: int,
    label: str,
) -> MarketOrder:
    step = 10_000 + epoch * 1_000 + agent_index * 10 + slot
    order_id = f"{label}-e{epoch}-{agent}"
    return MarketOrder(
        event_key=_market_event_key(
            plan, count, position, scenario, replicate, step, order_id
        ),
        submitted_at=step,
        order_id=order_id,
        trader_id=agent,
        side=side,
        quantity=quantity,
    )


def _run_world(
    plan: PropagationScalingPlan,
    count: int,
    target_position: str,
    scenario: str,
    replicate: int,
    mode: ScalingCoupling,
    *,
    branch_active: bool,
) -> dict[str, object]:
    agents = _agents(count)
    target = agents[_target_index(count, target_position)]
    drawer = _Drawer(plan, agents, target_position, scenario, replicate, mode)
    session = ReplayMarketSession(
        MarketConfig(tick_size=1),
        _opening(plan, agents, target_position, scenario, replicate, target),
    )
    draw_trace: list[dict[str, object]] = []
    command_trace: list[dict[str, object]] = []
    state_trace: list[str] = []
    for epoch in range(plan.step_count):
        snapshot = session.snapshot
        commands: list[MarketCommand] = []
        for agent_index, agent in enumerate(agents):
            values: dict[str, int] = {}
            for role in DRAW_ROLES:
                requested = _event_identity(epoch, agent, role)
                value, source = drawer.draw(requested, agent)
                values[role] = value
                draw_trace.append(
                    {
                        "agent": agent,
                        "epoch": epoch,
                        "requested_event": requested,
                        "source_event": source,
                        "value_u64": value,
                    }
                )
            side, quantity = _action(
                snapshot, values["side"], values["quantity"], agent
            )
            active = branch_active and epoch == plan.branch_epoch and agent == target
            if not active:
                commands.append(
                    _market_order(
                        plan,
                        count,
                        target_position,
                        scenario,
                        replicate,
                        epoch,
                        agent_index,
                        0,
                        agent,
                        side,
                        quantity,
                        "scheduled",
                    )
                )
                continue
            branch_values: dict[str, int] = {}
            for role in ("branch_decision", "branch_quantity"):
                requested = _branch_identity(scenario, epoch, target, role)
                value, source = drawer.draw(requested, target)
                branch_values[role] = value
                draw_trace.append(
                    {
                        "agent": target,
                        "epoch": epoch,
                        "requested_event": requested,
                        "source_event": source,
                        "value_u64": value,
                    }
                )
            replacement_side, replacement_quantity = _action(
                snapshot,
                branch_values["branch_decision"],
                branch_values["branch_quantity"],
                target,
            )
            if scenario == "partial_fill_remainder":
                commands.append(
                    _market_order(
                        plan,
                        count,
                        target_position,
                        scenario,
                        replicate,
                        epoch,
                        agent_index,
                        0,
                        target,
                        side,
                        16_000,
                        "partial-trigger",
                    )
                )
            elif scenario == "cancellation_replacement":
                step = 10_000 + epoch * 1_000 + agent_index * 10
                commands.append(
                    CancelOrder(
                        event_key=_market_event_key(
                            plan,
                            count,
                            target_position,
                            scenario,
                            replicate,
                            step,
                            f"cancel-e{epoch}-{target}",
                        ),
                        submitted_at=step,
                        trader_id=target,
                        target_order_id="target-resting",
                    )
                )
            commands.append(
                _market_order(
                    plan,
                    count,
                    target_position,
                    scenario,
                    replicate,
                    epoch,
                    agent_index,
                    1,
                    target,
                    replacement_side,
                    1 + replacement_quantity % 15,
                    f"branch-{scenario}",
                )
            )
        session.submit_all(tuple(commands))
        for command in commands:
            mapping = command_to_mapping(command)
            command_trace.append(
                {
                    "agent": str(mapping["trader_id"]),
                    "command": mapping,
                    "epoch": epoch,
                }
            )
        state_trace.append(session.snapshot.sha256)
    session.assert_replay()
    replay = session.replay
    final_snapshot = replay.state.to_mapping()
    return {
        "command_trace": command_trace,
        "draw_trace": draw_trace,
        "final_state_sha256": replay.state.sha256,
        "final_snapshot": final_snapshot,
        "state_trace": state_trace,
        "traded_notional": sum(item.price * item.quantity for item in replay.trades),
        "trades": [item.to_mapping() for item in replay.trades],
    }


def _draws(world: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    return {
        str(item["requested_event"]): item
        for item in cast(list[Mapping[str, object]], world["draw_trace"])
    }


def _commands(
    world: Mapping[str, object],
) -> dict[tuple[str, int], tuple[bytes, ...]]:
    grouped: dict[tuple[str, int], list[bytes]] = defaultdict(list)
    for row in cast(list[Mapping[str, object]], world["command_trace"]):
        grouped[(str(row["agent"]), _as_int(row["epoch"]))].append(
            canonical_json_bytes(row["command"])
        )
    return {key: tuple(value) for key, value in grouped.items()}


def _agent_projection(
    world: Mapping[str, object], agent: str
) -> tuple[tuple[bytes, ...], tuple[bytes, ...], tuple[int, int]]:
    snapshot = cast(dict[str, Any], world["final_snapshot"])
    orders = tuple(
        canonical_json_bytes(item)
        for item in cast(list[dict[str, object]], snapshot["order_records"])
        if item["trader_id"] == agent
    )
    fills = tuple(
        canonical_json_bytes(item)
        for item in cast(list[dict[str, object]], world["trades"])
        if agent in {item["buyer_id"], item["seller_id"]}
    )
    account = next(
        item
        for item in cast(list[dict[str, object]], snapshot["accounts"])
        if item["trader_id"] == agent
    )
    return orders, fills, (_as_int(account["cash"]), _as_int(account["asset"]))


def _compare(
    plan: PropagationScalingPlan,
    count: int,
    target_position: str,
    scenario: str,
    replicate: int,
    mode: ScalingCoupling,
    baseline: Mapping[str, object],
    reference: Mapping[str, object],
    candidate: Mapping[str, object],
) -> dict[str, object]:
    agents = _agents(count)
    target = agents[_target_index(count, target_position)]
    reference_draws = _draws(reference)
    candidate_draws = _draws(candidate)
    if set(reference_draws) != set(candidate_draws):
        _fail("candidate draw request set differs from event-keyed reference")
    mismatches = {
        key
        for key in reference_draws
        if reference_draws[key]["value_u64"] != candidate_draws[key]["value_u64"]
    }
    target_keys = {
        key
        for key, row in reference_draws.items()
        if row["agent"] == target and _as_int(row["epoch"]) >= plan.branch_epoch
    }
    unaffected_keys = {
        key
        for key, row in reference_draws.items()
        if row["agent"] != target and _as_int(row["epoch"]) >= plan.branch_epoch
    }
    reference_commands = _commands(reference)
    candidate_commands = _commands(candidate)
    command_divergence: dict[str, list[int]] = defaultdict(list)
    downstream: set[str] = set()
    for agent in agents:
        if agent == target:
            continue
        for epoch in range(plan.branch_epoch, plan.step_count):
            key = (agent, epoch)
            if reference_commands.get(key, ()) != candidate_commands.get(key, ()):
                command_divergence[agent].append(epoch)
        if command_divergence[agent] or _agent_projection(
            reference, agent
        ) != _agent_projection(candidate, agent):
            downstream.add(agent)
    reference_states = cast(list[str], reference["state_trace"])
    candidate_states = cast(list[str], candidate["state_trace"])
    divergent_epochs = [
        epoch
        for epoch, (left, right) in enumerate(
            zip(reference_states, candidate_states, strict=True)
        )
        if left != right
    ]
    first_unaffected = min(
        (epoch for epochs in command_divergence.values() for epoch in epochs),
        default=None,
    )
    reference_contrast = _as_float(reference["traded_notional"]) - _as_float(
        baseline["traded_notional"]
    )
    candidate_contrast = _as_float(candidate["traded_notional"]) - _as_float(
        baseline["traded_notional"]
    )
    row: dict[str, object] = {
        "agent_count": count,
        "candidate_paired_contrast": candidate_contrast,
        "coupling_mode": mode.value,
        "cross_agent_propagation_rate": len(downstream) / (count - 1),
        "downstream_divergent_agent_count": len(downstream),
        "final_output_diverged": candidate["final_state_sha256"]
        != reference["final_state_sha256"],
        "first_divergence_epoch": min(divergent_epochs) if divergent_epochs else None,
        "local_rng_containment": not bool(mismatches & unaffected_keys),
        "paired_contrast_error": candidate_contrast - reference_contrast,
        "propagation_depth_epochs": 0
        if first_unaffected is None
        else plan.step_count - first_unaffected,
        "reference_paired_contrast": reference_contrast,
        "replicate_id": replicate,
        "scenario": scenario,
        "system_level_propagation": len(downstream) / (count - 1)
        > plan.propagation_rate_threshold,
        "target_agent": target,
        "target_agent_draw_mismatch_rate": len(mismatches & target_keys)
        / len(target_keys),
        "target_position": target_position,
        "unaffected_agent_command_divergent_pair_count": sum(
            len(epochs) for epochs in command_divergence.values()
        ),
        "unaffected_agent_draw_mismatch_rate": len(mismatches & unaffected_keys)
        / len(unaffected_keys),
    }
    if mode is ScalingCoupling.EVENT_KEYED and any(
        (
            row["target_agent_draw_mismatch_rate"],
            row["unaffected_agent_draw_mismatch_rate"],
            row["cross_agent_propagation_rate"],
            row["paired_contrast_error"],
        )
    ):
        _fail("event-keyed scaling reference diverged from itself")
    if (
        mode is ScalingCoupling.PER_AGENT_SUBSTREAM
        and row["unaffected_agent_draw_mismatch_rate"] != 0.0
    ):
        _fail("per-agent scaling study leaked target drift into other streams")
    return row


def run_propagation_scaling_study(
    plan: PropagationScalingPlan,
) -> dict[str, object]:
    """Run every frozen population, target, branch, coupling, and replicate cell."""
    rows: list[dict[str, object]] = []
    for count in plan.agent_counts:
        for target_position in plan.target_positions:
            for scenario in plan.scenarios:
                for replicate in range(plan.replicate_count):
                    baseline = _run_world(
                        plan,
                        count,
                        target_position,
                        scenario,
                        replicate,
                        ScalingCoupling.EVENT_KEYED,
                        branch_active=False,
                    )
                    reference = _run_world(
                        plan,
                        count,
                        target_position,
                        scenario,
                        replicate,
                        ScalingCoupling.EVENT_KEYED,
                        branch_active=True,
                    )
                    for mode in ScalingCoupling:
                        candidate = (
                            reference
                            if mode is ScalingCoupling.EVENT_KEYED
                            else _run_world(
                                plan,
                                count,
                                target_position,
                                scenario,
                                replicate,
                                mode,
                                branch_active=True,
                            )
                        )
                        rows.append(
                            _compare(
                                plan,
                                count,
                                target_position,
                                scenario,
                                replicate,
                                mode,
                                baseline,
                                reference,
                                candidate,
                            )
                        )
    aggregates: list[dict[str, object]] = []
    for count in plan.agent_counts:
        for mode in ScalingCoupling:
            selected = [
                row
                for row in rows
                if row["agent_count"] == count and row["coupling_mode"] == mode.value
            ]
            propagation = [
                _as_float(row["cross_agent_propagation_rate"]) for row in selected
            ]
            aggregates.append(
                {
                    "agent_count": count,
                    "absolute_paired_contrast_error_mean": statistics.fmean(
                        abs(_as_float(row["paired_contrast_error"])) for row in selected
                    ),
                    "coupling_mode": mode.value,
                    "propagation_any_rate": statistics.fmean(
                        float(bool(row["system_level_propagation"])) for row in selected
                    ),
                    "propagation_rate_max": max(propagation),
                    "propagation_rate_mean": statistics.fmean(propagation),
                    "propagation_rate_median": statistics.median(propagation),
                    "propagation_rate_min": min(propagation),
                    "row_count": len(selected),
                    "target_draw_mismatch_rate_mean": statistics.fmean(
                        _as_float(row["target_agent_draw_mismatch_rate"])
                        for row in selected
                    ),
                    "unaffected_draw_mismatch_rate_mean": statistics.fmean(
                        _as_float(row["unaffected_agent_draw_mismatch_rate"])
                        for row in selected
                    ),
                }
            )
    return {
        "aggregates": aggregates,
        "config": plan.to_mapping(),
        "config_sha256": plan.sha256,
        "explicit_advance_skip_or_cursor_mutation_used": False,
        "row_count": len(rows),
        "rows": rows,
        "schema_version": PROPAGATION_SCALING_SCHEMA_VERSION,
        "synthetic_only": True,
    }


__all__ = [
    "PROPAGATION_SCALING_SCHEMA_VERSION",
    "PropagationScalingError",
    "PropagationScalingPlan",
    "ScalingCoupling",
    "run_propagation_scaling_study",
]
