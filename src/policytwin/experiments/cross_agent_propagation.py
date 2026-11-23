"""Post-registration study of RNG drift propagation through a shared CLOB."""

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

CROSS_AGENT_PROPAGATION_SCHEMA_VERSION = "policytwin.cross-agent-propagation.v1"
AGENTS = (
    "fundamental",
    "trend",
    "maker-a",
    "maker-b",
    "liquidity",
    "entrant",
)
DRAW_ROLES = ("side", "quantity")


class PropagationStudyError(RuntimeError):
    """Raised when the frozen propagation study cannot be executed exactly."""


class PropagationCoupling(StrEnum):
    EVENT_KEYED = "event_keyed"
    PER_AGENT_SUBSTREAM = "per_agent_substream"
    SHARED_SEQUENTIAL = "shared_sequential"


def _fail(message: str) -> NoReturn:
    raise PropagationStudyError(message)


def _strict_fields(value: Mapping[str, object], expected: frozenset[str]) -> None:
    if frozenset(value) != expected:
        _fail("propagation configuration has missing or unknown fields")


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
        _fail("artifact field must be an integer")
    return value


def _as_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        _fail("artifact field must be numeric")
    return float(value)


@dataclass(frozen=True, slots=True)
class PropagationScenario:
    name: str
    target_agent: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        if frozenset(value) != {"name", "target_agent"}:
            _fail("propagation scenario fields differ from the frozen schema")
        name = _string(value["name"], "scenario.name")
        if name not in {
            "rejected_order_retry",
            "partial_fill_remainder",
            "cancellation_replacement",
        }:
            _fail("unsupported propagation scenario")
        target = _string(value["target_agent"], "scenario.target_agent")
        if target not in AGENTS:
            _fail("target agent is absent from the frozen population")
        return cls(name=name, target_agent=target)


@dataclass(frozen=True, slots=True)
class PropagationPlan:
    experiment_id: str
    seed: str
    replicate_count: int
    step_count: int
    branch_epoch: int
    scenarios: tuple[PropagationScenario, ...]
    coupling_modes: tuple[str, ...]
    metrics: tuple[str, ...]
    propagation_rate_threshold: float
    expected_claims: tuple[str, ...]
    schema_version: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(value, frozenset(cls.__dataclass_fields__))
        raw_scenarios = value["scenarios"]
        if not isinstance(raw_scenarios, Sequence) or isinstance(raw_scenarios, str):
            _fail("scenarios must be an array")
        scenarios = tuple(
            PropagationScenario.from_mapping(cast(Mapping[str, object], item))
            for item in raw_scenarios
        )
        raw_modes = value["coupling_modes"]
        raw_metrics = value["metrics"]
        raw_claims = value["expected_claims"]
        if any(
            not isinstance(item, Sequence) or isinstance(item, str)
            for item in (raw_modes, raw_metrics, raw_claims)
        ):
            _fail("modes, metrics, and claims must be arrays")
        modes = tuple(
            _string(item, "coupling_mode") for item in cast(Sequence[object], raw_modes)
        )
        expected_modes = tuple(item.value for item in PropagationCoupling)
        if modes != expected_modes:
            _fail("coupling_modes must equal the frozen enum order")
        metrics = tuple(
            _string(item, "metric") for item in cast(Sequence[object], raw_metrics)
        )
        claims = tuple(
            _string(item, "expected_claim")
            for item in cast(Sequence[object], raw_claims)
        )
        threshold = value["propagation_rate_threshold"]
        if isinstance(threshold, bool) or not isinstance(threshold, int | float):
            _fail("propagation threshold must be numeric")
        plan = cls(
            experiment_id=_string(value["experiment_id"], "experiment_id"),
            seed=_string(value["seed"], "seed"),
            replicate_count=_integer(value["replicate_count"], "replicate_count", 1),
            step_count=_integer(value["step_count"], "step_count", 6),
            branch_epoch=_integer(value["branch_epoch"], "branch_epoch", 1),
            scenarios=scenarios,
            coupling_modes=modes,
            metrics=metrics,
            propagation_rate_threshold=float(threshold),
            expected_claims=claims,
            schema_version=_string(value["schema_version"], "schema_version"),
        )
        if plan.schema_version != CROSS_AGENT_PROPAGATION_SCHEMA_VERSION:
            _fail("unsupported propagation-study schema")
        if plan.branch_epoch >= plan.step_count - 1:
            _fail("branch_epoch must leave downstream epochs")
        if not 0.0 <= plan.propagation_rate_threshold < 1.0:
            _fail("propagation threshold must be in [0, 1)")
        if len({item.name for item in scenarios}) != len(scenarios):
            _fail("propagation scenarios must be unique")
        return plan

    def to_mapping(self) -> dict[str, object]:
        return {
            "branch_epoch": self.branch_epoch,
            "coupling_modes": list(self.coupling_modes),
            "expected_claims": list(self.expected_claims),
            "experiment_id": self.experiment_id,
            "metrics": list(self.metrics),
            "propagation_rate_threshold": self.propagation_rate_threshold,
            "replicate_count": self.replicate_count,
            "scenarios": [
                {"name": item.name, "target_agent": item.target_agent}
                for item in self.scenarios
            ],
            "schema_version": self.schema_version,
            "seed": self.seed,
            "step_count": self.step_count,
        }

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))


def _event_identity(epoch: int, agent: str, role: str) -> str:
    return f"scheduled|epoch={epoch}|agent={agent}|role={role}"


def _branch_identity(scenario: str, epoch: int, agent: str, role: str) -> str:
    return f"branch|scenario={scenario}|epoch={epoch}|agent={agent}|role={role}"


def _draw_bits(plan: PropagationPlan, scenario: str, replicate: int, key: str) -> int:
    return int.from_bytes(
        hashlib.sha256(
            canonical_json_bytes(
                {
                    "key": key,
                    "replicate": replicate,
                    "scenario": scenario,
                    "seed": plan.seed,
                }
            )
        ).digest()[:8],
        "big",
    )


class _Drawer:
    def __init__(
        self,
        plan: PropagationPlan,
        scenario: PropagationScenario,
        replicate: int,
        mode: PropagationCoupling,
    ) -> None:
        self._plan = plan
        self._scenario = scenario
        self._replicate = replicate
        self._mode = mode
        scheduled = [
            _event_identity(epoch, agent, role)
            for epoch in range(plan.step_count)
            for agent in AGENTS
            for role in DRAW_ROLES
        ]
        self._shared = tuple(scheduled)
        self._by_agent = {
            agent: tuple(key for key in scheduled if f"agent={agent}|" in key)
            for agent in AGENTS
        }
        self._shared_cursor = 0
        self._agent_cursors = {agent: 0 for agent in AGENTS}

    def draw(self, requested: str, agent: str) -> tuple[int, str]:
        if self._mode is PropagationCoupling.EVENT_KEYED:
            return (
                _draw_bits(self._plan, self._scenario.name, self._replicate, requested),
                requested,
            )
        if self._mode is PropagationCoupling.SHARED_SEQUENTIAL:
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
            _draw_bits(self._plan, self._scenario.name, self._replicate, source),
            source,
        )


def _market_event_key(
    plan: PropagationPlan,
    scenario: PropagationScenario,
    replicate: int,
    step: int,
    purpose: str,
) -> EventKey:
    return EventKey(
        experiment_id=plan.experiment_id,
        seed=f"{plan.seed}:{scenario.name}",
        replicate_id=replicate,
        simulation_step=step,
        purpose=purpose,
    )


def _opening(
    plan: PropagationPlan, scenario: PropagationScenario, replicate: int
) -> OpeningState:
    participant_ids = (*AGENTS, "book-bid", "book-ask")
    accounts = tuple(
        AccountBalance(trader_id=item, cash=10_000_000, asset=100_000)
        for item in participant_ids
    )
    raw_orders = (
        (Side.BUY, 90, 500, "book-bid", "book-bid-90"),
        (Side.BUY, 80, 500, "book-bid", "book-bid-80"),
        (Side.BUY, 70, 500, "book-bid", "book-bid-70"),
        (Side.SELL, 110, 500, "book-ask", "book-ask-110"),
        (Side.SELL, 120, 500, "book-ask", "book-ask-120"),
        (Side.SELL, 130, 500, "book-ask", "book-ask-130"),
        # Kept below the public bid stack so maker-a's ordinary small sell
        # orders cannot self-trade before the cancellation branch.
        (Side.BUY, 60, 20, "maker-a", "maker-a-resting"),
    )
    orders = tuple(
        OpeningOrder(
            event_key=_market_event_key(
                plan, scenario, replicate, index, f"opening-{order_id}"
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


def _account_signal(snapshot: MarketSnapshot, agent: str) -> int:
    account = next(item for item in snapshot.accounts if item.trader_id == agent)
    return account.asset + account.cash // 100


def _action(
    snapshot: MarketSnapshot, side_bits: int, quantity_bits: int, agent: str
) -> tuple[Side, int]:
    quote_signal = 0
    if snapshot.resting_orders:
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
        quote_signal = 3 * bid_quantity + 5 * ask_quantity + best_bid + best_ask
    signal = quote_signal + 7 * _account_signal(snapshot, agent)
    side = Side.BUY if (side_bits + signal) % 2 == 0 else Side.SELL
    quantity = 1 + (quantity_bits + signal) % 9
    return side, quantity


def _market_order(
    plan: PropagationPlan,
    scenario: PropagationScenario,
    replicate: int,
    epoch: int,
    agent_index: int,
    slot: int,
    agent: str,
    side: Side,
    quantity: int,
    label: str,
) -> MarketOrder:
    step = 10_000 + epoch * 100 + agent_index * 10 + slot
    order_id = f"{label}-e{epoch}-{agent}"
    return MarketOrder(
        event_key=_market_event_key(plan, scenario, replicate, step, order_id),
        submitted_at=step,
        order_id=order_id,
        trader_id=agent,
        side=side,
        quantity=quantity,
    )


def _run_world(
    plan: PropagationPlan,
    scenario: PropagationScenario,
    replicate: int,
    mode: PropagationCoupling,
    *,
    branch_active: bool,
) -> dict[str, object]:
    drawer = _Drawer(plan, scenario, replicate, mode)
    session = ReplayMarketSession(
        MarketConfig(tick_size=1), _opening(plan, scenario, replicate)
    )
    draw_trace: list[dict[str, object]] = []
    command_trace: list[dict[str, object]] = []
    state_trace: list[dict[str, object]] = []
    for epoch in range(plan.step_count):
        snapshot = session.snapshot
        commands: list[MarketCommand] = []
        for agent_index, agent in enumerate(AGENTS):
            values: dict[str, int] = {}
            for role in DRAW_ROLES:
                requested = _event_identity(epoch, agent, role)
                value, source = drawer.draw(requested, agent)
                values[role] = value
                draw_trace.append(
                    {
                        "agent": agent,
                        "epoch": epoch,
                        "is_branch_draw": False,
                        "requested_event": requested,
                        "role": role,
                        "source_event": source,
                        "value_u64": value,
                    }
                )
            side, quantity = _action(
                snapshot, values["side"], values["quantity"], agent
            )
            is_target_branch = (
                branch_active
                and epoch == plan.branch_epoch
                and agent == scenario.target_agent
            )
            if not is_target_branch:
                commands.append(
                    _market_order(
                        plan,
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
                requested = _branch_identity(
                    scenario.name, epoch, scenario.target_agent, role
                )
                value, source = drawer.draw(requested, agent)
                branch_values[role] = value
                draw_trace.append(
                    {
                        "agent": agent,
                        "epoch": epoch,
                        "is_branch_draw": True,
                        "requested_event": requested,
                        "role": role,
                        "source_event": source,
                        "value_u64": value,
                    }
                )
            replacement_side, replacement_quantity = _action(
                snapshot,
                branch_values["branch_decision"],
                branch_values["branch_quantity"],
                agent,
            )
            if scenario.name == "partial_fill_remainder":
                commands.append(
                    _market_order(
                        plan,
                        scenario,
                        replicate,
                        epoch,
                        agent_index,
                        0,
                        agent,
                        side,
                        1_600,
                        "partial-trigger",
                    )
                )
            elif scenario.name == "cancellation_replacement":
                cancel_step = 10_000 + epoch * 100 + agent_index * 10
                commands.append(
                    CancelOrder(
                        event_key=_market_event_key(
                            plan,
                            scenario,
                            replicate,
                            cancel_step,
                            f"cancel-e{epoch}-{agent}",
                        ),
                        submitted_at=cancel_step,
                        trader_id=agent,
                        target_order_id="maker-a-resting",
                    )
                )
            # The retry/remainder/replacement is an ordinary business-path draw,
            # not an explicit cursor mutation.
            commands.append(
                _market_order(
                    plan,
                    scenario,
                    replicate,
                    epoch,
                    agent_index,
                    1,
                    agent,
                    replacement_side,
                    1 + replacement_quantity % 15,
                    f"branch-{scenario.name}",
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
        quote = session.replay.quotes[-1]
        state_trace.append(
            {
                "best_ask_price": quote.best_ask_price,
                "best_ask_quantity": quote.best_ask_quantity,
                "best_bid_price": quote.best_bid_price,
                "best_bid_quantity": quote.best_bid_quantity,
                "epoch": epoch,
                "state_sha256": session.snapshot.sha256,
            }
        )
    session.assert_replay()
    replay = session.replay
    traded_notional = sum(item.price * item.quantity for item in replay.trades)
    return {
        "command_trace": command_trace,
        "draw_trace": draw_trace,
        "final_replay_sha256": replay.sha256,
        "final_snapshot": replay.state.to_mapping(),
        "final_state_sha256": replay.state.sha256,
        "replay_verified": True,
        "state_trace": state_trace,
        "traded_notional": traded_notional,
        "trades": [item.to_mapping() for item in replay.trades],
    }


def _by_requested(world: Mapping[str, object]) -> dict[str, dict[str, object]]:
    trace = cast(list[dict[str, object]], world["draw_trace"])
    return {str(item["requested_event"]): item for item in trace}


def _command_by_agent_epoch(
    world: Mapping[str, object],
) -> dict[tuple[str, int], tuple[bytes, ...]]:
    result: dict[tuple[str, int], list[bytes]] = defaultdict(list)
    for row in cast(list[dict[str, object]], world["command_trace"]):
        result[(str(row["agent"]), _as_int(row["epoch"]))].append(
            canonical_json_bytes(row["command"])
        )
    return {key: tuple(value) for key, value in result.items()}


def _agent_orders(world: Mapping[str, object], agent: str) -> tuple[bytes, ...]:
    snapshot = cast(dict[str, Any], world["final_snapshot"])
    return tuple(
        canonical_json_bytes(item)
        for item in cast(list[dict[str, object]], snapshot["order_records"])
        if item["trader_id"] == agent
    )


def _agent_fills(world: Mapping[str, object], agent: str) -> tuple[bytes, ...]:
    return tuple(
        canonical_json_bytes(item)
        for item in cast(list[dict[str, object]], world["trades"])
        if agent in {item["buyer_id"], item["seller_id"]}
    )


def _agent_inventory(world: Mapping[str, object], agent: str) -> tuple[int, int]:
    snapshot = cast(dict[str, Any], world["final_snapshot"])
    account = next(
        item
        for item in cast(list[dict[str, object]], snapshot["accounts"])
        if item["trader_id"] == agent
    )
    return _as_int(account["cash"]), _as_int(account["asset"])


def _compare_worlds(
    plan: PropagationPlan,
    scenario: PropagationScenario,
    replicate: int,
    mode: PropagationCoupling,
    baseline: Mapping[str, object],
    reference: Mapping[str, object],
    candidate: Mapping[str, object],
) -> dict[str, object]:
    reference_draws = _by_requested(reference)
    candidate_draws = _by_requested(candidate)
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
        if row["agent"] == scenario.target_agent
        and _as_int(row["epoch"]) >= plan.branch_epoch
    }
    unaffected_keys = {
        key
        for key, row in reference_draws.items()
        if row["agent"] != scenario.target_agent
        and _as_int(row["epoch"]) >= plan.branch_epoch
    }
    reference_commands = _command_by_agent_epoch(reference)
    candidate_commands = _command_by_agent_epoch(candidate)
    divergent_commands: dict[str, list[int]] = defaultdict(list)
    for agent in AGENTS:
        if agent == scenario.target_agent:
            continue
        for epoch in range(plan.branch_epoch, plan.step_count):
            key = (agent, epoch)
            if reference_commands.get(key, ()) != candidate_commands.get(key, ()):
                divergent_commands[agent].append(epoch)
    order_divergent = {
        agent
        for agent in AGENTS
        if agent != scenario.target_agent
        and _agent_orders(reference, agent) != _agent_orders(candidate, agent)
    }
    fill_divergent = {
        agent
        for agent in AGENTS
        if agent != scenario.target_agent
        and _agent_fills(reference, agent) != _agent_fills(candidate, agent)
    }
    inventory_divergent = {
        agent
        for agent in AGENTS
        if agent != scenario.target_agent
        and _agent_inventory(reference, agent) != _agent_inventory(candidate, agent)
    }
    downstream = (
        set(divergent_commands)
        | order_divergent
        | fill_divergent
        | (inventory_divergent)
    )
    reference_states = cast(list[dict[str, object]], reference["state_trace"])
    candidate_states = cast(list[dict[str, object]], candidate["state_trace"])
    divergent_state_epochs = [
        _as_int(left["epoch"])
        for left, right in zip(reference_states, candidate_states, strict=True)
        if left["state_sha256"] != right["state_sha256"]
    ]
    trace_order = [
        str(item["requested_event"])
        for item in cast(list[dict[str, object]], reference["draw_trace"])
    ]
    first_semantic = next((key for key in trace_order if key in mismatches), None)
    first_unaffected = min(
        (epoch for epochs in divergent_commands.values() for epoch in epochs),
        default=None,
    )
    unaffected_count = len(AGENTS) - 1
    reference_contrast = _as_float(reference["traded_notional"]) - _as_float(
        baseline["traded_notional"]
    )
    candidate_contrast = _as_float(candidate["traded_notional"]) - _as_float(
        baseline["traded_notional"]
    )
    return {
        "candidate_paired_contrast": candidate_contrast,
        "coupling_mode": mode.value,
        "cross_agent_propagation_rate": len(downstream) / unaffected_count,
        "downstream_divergent_agent_count": len(downstream),
        "fill_divergent_unaffected_agent_count": len(fill_divergent),
        "final_output_diverged": candidate["final_state_sha256"]
        != reference["final_state_sha256"],
        "first_divergence_epoch": (
            min(divergent_state_epochs) if divergent_state_epochs else None
        ),
        "first_semantic_divergence": first_semantic,
        "first_unaffected_command_divergence_epoch": first_unaffected,
        "inventory_divergent_unaffected_agent_count": len(inventory_divergent),
        "local_rng_containment": not bool(mismatches & unaffected_keys),
        "market_state_divergent_epoch_count": len(divergent_state_epochs),
        "order_divergent_unaffected_agent_count": len(order_divergent),
        "paired_contrast_error": candidate_contrast - reference_contrast,
        "propagation_depth_epochs": (
            0 if first_unaffected is None else plan.step_count - first_unaffected
        ),
        "reference_paired_contrast": reference_contrast,
        "replicate_id": replicate,
        "scenario": scenario.name,
        "system_level_propagation": (
            len(downstream) / unaffected_count > plan.propagation_rate_threshold
        ),
        "target_agent": scenario.target_agent,
        "target_agent_draw_mismatch_count": len(mismatches & target_keys),
        "target_agent_draw_mismatch_rate": len(mismatches & target_keys)
        / len(target_keys),
        "unaffected_agent_command_divergent_pair_count": sum(
            len(epochs) for epochs in divergent_commands.values()
        ),
        "unaffected_agent_draw_mismatch_count": len(mismatches & unaffected_keys),
        "unaffected_agent_draw_mismatch_rate": len(mismatches & unaffected_keys)
        / len(unaffected_keys),
    }


def run_propagation_study(
    plan: PropagationPlan,
) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    """Run the frozen branch-by-coupling grid and retain every trace."""
    rows: list[dict[str, object]] = []
    traces: list[dict[str, object]] = []
    for scenario in plan.scenarios:
        for replicate in range(plan.replicate_count):
            baseline = _run_world(
                plan,
                scenario,
                replicate,
                PropagationCoupling.EVENT_KEYED,
                branch_active=False,
            )
            reference = _run_world(
                plan,
                scenario,
                replicate,
                PropagationCoupling.EVENT_KEYED,
                branch_active=True,
            )
            for mode in PropagationCoupling:
                candidate = (
                    reference
                    if mode is PropagationCoupling.EVENT_KEYED
                    else _run_world(
                        plan,
                        scenario,
                        replicate,
                        mode,
                        branch_active=True,
                    )
                )
                rows.append(
                    _compare_worlds(
                        plan,
                        scenario,
                        replicate,
                        mode,
                        baseline,
                        reference,
                        candidate,
                    )
                )
                traces.append(
                    {
                        "candidate": candidate,
                        "coupling_mode": mode.value,
                        "reference": reference,
                        "replicate_id": replicate,
                        "scenario": scenario.name,
                    }
                )
    aggregates: list[dict[str, object]] = []
    for scenario in plan.scenarios:
        for mode in PropagationCoupling:
            selected = [
                row
                for row in rows
                if row["scenario"] == scenario.name
                and row["coupling_mode"] == mode.value
            ]
            aggregates.append(
                {
                    "coupling_mode": mode.value,
                    "local_containment_rate": statistics.fmean(
                        bool(row["local_rng_containment"]) for row in selected
                    ),
                    "mean_cross_agent_propagation_rate": statistics.fmean(
                        _as_float(row["cross_agent_propagation_rate"])
                        for row in selected
                    ),
                    "mean_paired_contrast_absolute_error": statistics.fmean(
                        abs(_as_float(row["paired_contrast_error"])) for row in selected
                    ),
                    "mean_target_agent_draw_mismatch_rate": statistics.fmean(
                        _as_float(row["target_agent_draw_mismatch_rate"])
                        for row in selected
                    ),
                    "mean_unaffected_agent_draw_mismatch_rate": statistics.fmean(
                        _as_float(row["unaffected_agent_draw_mismatch_rate"])
                        for row in selected
                    ),
                    "replicate_count": len(selected),
                    "scenario": scenario.name,
                    "system_propagation_rate": statistics.fmean(
                        bool(row["system_level_propagation"]) for row in selected
                    ),
                }
            )
    artifact: dict[str, object] = {
        "aggregates": aggregates,
        "explicit_advance_skip_or_cursor_mutation_used": False,
        "plan": plan.to_mapping(),
        "plan_sha256": plan.sha256,
        "rows": rows,
        "schema_version": CROSS_AGENT_PROPAGATION_SCHEMA_VERSION,
        "synthetic_only": True,
        "trace_record_count": len(traces),
    }
    artifact["scientific_core_sha256"] = sha256_bytes(canonical_json_bytes(artifact))
    return artifact, tuple(traces)


__all__ = [
    "AGENTS",
    "CROSS_AGENT_PROPAGATION_SCHEMA_VERSION",
    "PropagationCoupling",
    "PropagationPlan",
    "PropagationScenario",
    "PropagationStudyError",
    "run_propagation_study",
]
