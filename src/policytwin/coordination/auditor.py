"""Executable conformance predicates; no access to injection labels or hooks."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from policytwin.coordination.plan import (
    CoordinationExecutionPlan,
    RunSpec,
    compile_contract,
)
from policytwin.coordination.schema import TaskRequest, TaskResult, commitment
from policytwin.coordination.workload import opening_state
from policytwin.counterfactual.coupling_contract import (
    ConsumptionObservation,
    World,
    audit_pair,
)
from policytwin.market.models import MarketConfig
from policytwin.market.session import ReplayMarketSession


@dataclass(frozen=True, slots=True)
class Finding:
    """One auditable violation witness, not a ground-truth fault label."""

    component: str
    code: str
    world: str
    epoch: int
    task_id: str


def event_conformance(spec: RunSpec, artifact: dict[str, Any]) -> tuple[Finding, ...]:
    """Evaluate the existing C/O0/O1 relation on runtime draw records."""
    contract = compile_contract(spec)
    by_id = contract.by_identity
    observations: list[ConsumptionObservation] = []
    findings: list[Finding] = []
    if artifact.get("contract_sha256") != contract.sha256:
        findings.append(Finding("events", "contract_binding", "pair", -1, ""))
    for world in ("baseline", "policy"):
        for row in artifact["worlds"][world]["epochs"]:
            for raw in row["results"]:
                try:
                    result = TaskResult.from_mapping(raw)
                    for draw in result.draws:
                        event = by_id.get(draw["event_identity"])
                        if (
                            event is None
                            or draw["address"] != event.to_mapping()
                            or draw["world"] != world
                        ):
                            findings.append(
                                Finding(
                                    "events",
                                    "event_metadata",
                                    world,
                                    row["epoch"],
                                    result.task_id,
                                )
                            )
                            continue
                        observation = (
                            ConsumptionObservation.consumed(
                                World(world), event, draw["value"]
                            )
                            if draw["state"] == "consumed"
                            else ConsumptionObservation.masked(World(world), event)
                        )
                        observations.append(observation)
                except (TypeError, ValueError, KeyError) as error:
                    findings.append(
                        Finding(
                            "events",
                            f"schema:{type(error).__name__}",
                            world,
                            row["epoch"],
                            "",
                        )
                    )
    report = audit_pair(contract, observations)
    for issue in report.violations:
        findings.append(
            Finding(
                "events",
                issue.code.value,
                "pair" if issue.world is None else issue.world.value,
                -1,
                issue.event_identity,
            )
        )
    return tuple(findings)


def _view(state: dict[str, Any], agent: str) -> dict[str, Any]:
    orders = state["resting_orders"]
    bids = [o for o in orders if o["side"] == "buy"]
    asks = [o for o in orders if o["side"] == "sell"]
    bid = max((o["price"] for o in bids), default=0)
    ask = min((o["price"] for o in asks), default=0)
    account = next(a for a in state["accounts"] if a["trader_id"] == agent)
    return {
        "state_sha256": commitment(state),
        "best_bid": bid,
        "best_ask": ask,
        "bid_quantity": sum(o["remaining_quantity"] for o in bids if o["price"] == bid),
        "ask_quantity": sum(o["remaining_quantity"] for o in asks if o["price"] == ask),
        "cash": account["cash"],
        "asset": account["asset"],
    }


def coordination_conformance(
    spec: RunSpec,
    plan: CoordinationExecutionPlan,
    artifact: dict[str, Any],
) -> tuple[Finding, ...]:
    """Check independently declared tasks, ownership, messages, order and versions."""
    findings: list[Finding] = []

    def flag(component: str, code: str, world: str, epoch: int, task: str = "") -> None:
        findings.append(Finding(component, code, world, epoch, task))

    for world in ("baseline", "policy"):
        rows = artifact["worlds"][world]["epochs"]
        if [r["epoch"] for r in rows] != list(range(spec.epoch_count)):
            flag("expected", "epoch_coverage", world, -1)
        state = ReplayMarketSession(
            MarketConfig(tick_size=1), opening_state(spec)
        ).snapshot.to_mapping()
        for row in rows:
            epoch = row["epoch"]
            if type(epoch) is not int or not 0 <= epoch < spec.epoch_count:
                flag("expected", "unexpected_epoch", world, -1)
                continue
            expected = {spec.task_id(world, epoch, a): a for a in spec.agents}
            requests: dict[str, TaskRequest] = {}
            results: list[TaskResult] = []
            deliveries: list[TaskResult] = []
            try:
                request_rows = [TaskRequest.from_mapping(r) for r in row["requests"]]
                requests = {r.task_id: r for r in request_rows}
                if Counter(r.task_id for r in request_rows) != Counter(expected.keys()):
                    flag("expected", "dispatch_coverage", world, epoch)
                for dispatched in request_rows:
                    if (
                        dispatched.run_id,
                        dispatched.world,
                        dispatched.epoch,
                        dispatched.agent,
                    ) != (spec.run_id, world, epoch, expected.get(dispatched.task_id)):
                        flag(
                            "message",
                            "dispatch_identity",
                            world,
                            epoch,
                            dispatched.task_id,
                        )
                results = [TaskResult.from_mapping(r) for r in row["results"]]
                deliveries = [TaskResult.from_mapping(r) for r in row["deliveries"]]
            except (KeyError, TypeError, ValueError):
                flag("schema", "malformed_record", world, epoch)
            if Counter(r.task_id for r in results) != Counter(expected.keys()):
                flag("expected", "task_effect_coverage", world, epoch)
            if Counter(r.task_id for r in deliveries) != Counter(expected.keys()):
                flag("expected", "message_effect_coverage", world, epoch)
            if row["state_before_sha256"] != commitment(state):
                flag("view", "state_chain", world, epoch)
            result_ids = {r.message_id: r for r in results}
            for result in (*results, *deliveries):
                agent = expected.get(result.task_id)
                if agent is None:
                    flag("expected", "unexpected_task", world, epoch, result.task_id)
                    continue
                if (result.run_id, result.world, result.epoch, result.agent) != (
                    spec.run_id,
                    world,
                    epoch,
                    agent,
                ):
                    flag("message", "semantic_identity", world, epoch, result.task_id)
                if result.worker_id != plan.owner(spec, agent):
                    flag("ownership", "wrong_owner", world, epoch, result.task_id)
                if result.plan_sha256 != plan.sha256:
                    flag("version", "stale_plan", world, epoch, result.task_id)
                if result.payload_sha256 != commitment(result.payload()):
                    flag("message", "payload_commitment", world, epoch, result.task_id)
                if result.message_id != "msg-" + result.task_id.removeprefix("task-"):
                    flag("message", "message_identity", world, epoch, result.task_id)
                expected_view = _view(state, agent)
                request = requests.get(result.task_id)
                if (
                    result.view_sha256 != commitment(expected_view)
                    or request is None
                    or request.view.to_mapping() != expected_view
                ):
                    flag("view", "stale_or_wrong_view", world, epoch, result.task_id)
                if request is not None and request.plan_sha256 != plan.sha256:
                    flag("version", "dispatch_plan", world, epoch, result.task_id)
                status = "active" if spec.active(world, epoch, agent) else "inactive"
                if result.status != status or (
                    status == "inactive" and result.commands
                ):
                    flag("message", "activation_status", world, epoch, result.task_id)
                for draw in result.draws:
                    role = draw["address"].get("draw_role")
                    consumed = spec.consumes(world, epoch, agent, role)
                    if (draw["state"] == "consumed") != consumed:
                        flag("events", "activation_state", world, epoch, result.task_id)
            for delivered in deliveries:
                original = result_ids.get(delivered.message_id)
                if original is None or original.to_mapping() != delivered.to_mapping():
                    flag("message", "delivery_binding", world, epoch, delivered.task_id)
            expected_messages = [
                "msg-" + task.removeprefix("task-") for task in expected
            ]
            if row["committed_message_ids"] != expected_messages:
                flag("logical_order", "commit_order_or_multiplicity", world, epoch)
            message_by_id = {r.message_id: r for r in deliveries}
            recorded_commands = [
                command
                for mid in row["committed_message_ids"]
                if mid in message_by_id
                for command in message_by_id[mid].commands
            ]
            if row["applied_commands"] != recorded_commands:
                flag("message", "command_commit_binding", world, epoch)
            state = row["state_after"]
    return tuple(findings)
