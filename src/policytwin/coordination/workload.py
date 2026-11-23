"""Epoch-view workload adapter using the repository's existing integer CLOB."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from policytwin.coordination.plan import RunSpec, event_lookup
from policytwin.coordination.schema import (
    AgentView,
    CoordinationError,
    TaskRequest,
    TaskResult,
    commitment,
    integer,
    object_fields,
)
from policytwin.counterfactual import EventKey
from policytwin.counterfactual.coupling_contract import (
    CouplingContract,
    EventKeyedBackend,
    World,
)
from policytwin.market.models import (
    AccountBalance,
    CancelOrder,
    MarketCommand,
    MarketOrder,
    MarketSnapshot,
    OpeningOrder,
    OpeningState,
    Side,
)


def command_key(spec: RunSpec, step: int, purpose: str) -> EventKey:
    """Keep scientific command identities free of worker and plan metadata."""
    return EventKey(
        "cscwd2027-coordination", spec.run_id, spec.replicate, step, purpose
    )


def opening_state(spec: RunSpec) -> OpeningState:
    """Use the same deep integer book and account scale as propagation scaling."""
    accounts = tuple(
        AccountBalance(agent, 100_000_000, 1_000_000)
        for agent in (*spec.agents, "book-bid", "book-ask")
    )
    levels = [(Side.BUY, p, "book-bid") for p in (90, 80, 70)]
    levels += [(Side.SELL, p, "book-ask") for p in (110, 120, 130)]
    orders = [
        OpeningOrder(
            command_key(spec, i, f"opening-{owner}-{p}"),
            f"opening-{owner}-{p}",
            owner,
            side,
            p,
            5000,
        )
        for i, (side, p, owner) in enumerate(levels)
    ]
    if spec.branch == "cancellation":
        orders.append(
            OpeningOrder(
                command_key(spec, 6, "opening-target"),
                "target-resting",
                spec.target,
                Side.BUY,
                60,
                20,
            )
        )
    return OpeningState(accounts=accounts, orders=tuple(orders))


def agent_view(snapshot: MarketSnapshot, agent: str) -> AgentView:
    """Project exactly the quantities used by the inherited feedback policy."""
    bids = [o for o in snapshot.resting_orders if o.side is Side.BUY]
    asks = [o for o in snapshot.resting_orders if o.side is Side.SELL]
    best_bid = max((o.price for o in bids), default=0)
    best_ask = min((o.price for o in asks), default=0)
    account = next(a for a in snapshot.accounts if a.trader_id == agent)
    return AgentView(
        snapshot.sha256,
        best_bid,
        best_ask,
        sum(o.remaining_quantity for o in bids if o.price == best_bid),
        sum(o.remaining_quantity for o in asks if o.price == best_ask),
        account.cash,
        account.asset,
    )


def action(view: AgentView, side_bits: int, quantity_bits: int) -> tuple[Side, int]:
    """Match propagation_scaling._action on the same view and input values."""
    signal = (
        3 * view.bid_quantity
        + 5 * view.ask_quantity
        + view.best_bid
        + view.best_ask
        + 7 * (view.asset + view.cash // 100)
    )
    return (
        Side.BUY if (side_bits + signal) % 2 == 0 else Side.SELL,
        1 + (quantity_bits + signal) % 9,
    )


def compute_task(
    spec: RunSpec,
    contract: CouplingContract,
    request: TaskRequest,
    worker_id: int,
    worker_pid: int,
    *,
    collect_draws: bool = True,
) -> TaskResult:
    """Compute an agent decision in the receiving process, including branch draws."""
    if request.run_id != spec.run_id or request.task_id != spec.task_id(
        request.world, request.epoch, request.agent
    ):
        raise CoordinationError("request scientific identity mismatch")
    events = event_lookup(contract)
    backend = EventKeyedBackend(spec.seed)
    values: dict[str, int] = {}
    draws: list[dict[str, Any]] = []
    for role in (
        "side",
        "quantity",
        "replacement_side",
        "replacement_quantity",
        "jitter",
    ):
        event = events[request.epoch, request.agent, role]
        if not event.world_scope.applies_to(World(request.world)):
            continue
        consumed = spec.consumes(request.world, request.epoch, request.agent, role)
        value = integer(backend.draw(event), "sample") if consumed else None
        if value is not None:
            values[role] = value
        if collect_draws:
            draws.append(
                {
                    "event_identity": event.identity,
                    "address": event.to_mapping(),
                    "world": request.world,
                    "state": "consumed" if consumed else "masked",
                    "value": value,
                }
            )
    commands: list[dict[str, Any]] = []
    active = spec.active(request.world, request.epoch, request.agent)
    if active:
        side, quantity = action(request.view, values["side"], values["quantity"])
        index = spec.agents.index(request.agent)
        step = 10_000 + request.epoch * 1000 + index * 10
        if "replacement_side" in values:
            commands.append(
                CancelOrder(
                    command_key(spec, step, f"cancel-{request.epoch}"),
                    step,
                    request.agent,
                    "target-resting",
                ).to_mapping()
            )
            side, quantity = action(
                request.view, values["replacement_side"], values["replacement_quantity"]
            )
            step += 1
        if "jitter" in values:
            step += 20 + values["jitter"] % 10
        name = f"order-e{request.epoch}-{request.agent}"
        commands.append(
            MarketOrder(
                command_key(spec, step, name), step, name, request.agent, side, quantity
            ).to_mapping()
        )
    result = TaskResult(
        run_id=spec.run_id,
        task_id=request.task_id,
        message_id="msg-" + request.task_id.removeprefix("task-"),
        world=request.world,
        epoch=request.epoch,
        agent=request.agent,
        worker_id=worker_id,
        worker_pid=worker_pid,
        plan_sha256=request.plan_sha256,
        view_sha256=commitment(request.view.to_mapping()),
        status="active" if active else "inactive",
        draws=tuple(draws),
        commands=tuple(commands),
        payload_sha256="0" * 64,
    )
    return replace(result, payload_sha256=commitment(result.payload()))


def parse_command(value: object) -> MarketCommand:
    """Convert a closed worker wire command to the existing market API."""
    if not isinstance(value, dict):
        raise CoordinationError("command must be an object")
    kind = value.get("command_type")
    common = {"command_type", "event_key", "submitted_at", "trader_id"}
    fields = common | (
        {"target_order_id"}
        if kind == "cancel_order"
        else {"order_id", "side", "quantity"}
    )
    obj = object_fields(value, fields)
    key = dict(
        object_fields(
            obj["event_key"],
            {
                "experiment_id",
                "seed",
                "replicate_id",
                "simulation_step",
                "purpose",
                "schema_version",
                "value",
            },
        )
    )
    digest = key.pop("value")
    event = EventKey(**key)
    if event.value != digest:
        raise CoordinationError("command event commitment mismatch")
    submitted = integer(obj["submitted_at"], "submitted_at")
    if kind == "cancel_order":
        return CancelOrder(event, submitted, obj["trader_id"], obj["target_order_id"])
    if kind != "market_order":
        raise CoordinationError("unsupported command kind")
    return MarketOrder(
        event,
        submitted,
        obj["order_id"],
        obj["trader_id"],
        Side(obj["side"]),
        integer(obj["quantity"], "quantity", 1),
    )
