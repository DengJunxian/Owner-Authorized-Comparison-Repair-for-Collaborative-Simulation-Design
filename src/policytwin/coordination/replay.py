"""Actual task and CLOB re-execution, separate from receipt-set conformance."""

from __future__ import annotations

from typing import Any

from policytwin.coordination.auditor import Finding
from policytwin.coordination.plan import (
    CoordinationExecutionPlan,
    RunSpec,
    compile_contract,
)
from policytwin.coordination.schema import TaskRequest
from policytwin.coordination.workload import (
    agent_view,
    compute_task,
    opening_state,
    parse_command,
)
from policytwin.market.models import MarketConfig
from policytwin.market.session import ReplayMarketSession


def replay_valid(
    spec: RunSpec,
    plan: CoordinationExecutionPlan,
    artifact: dict[str, Any],
) -> tuple[Finding, ...]:
    """Recompute each agent action from the independently reconstructed state."""
    contract = compile_contract(spec)
    findings: list[Finding] = []
    values: dict[str, int] = {}
    for world in ("baseline", "policy"):
        session = ReplayMarketSession(MarketConfig(tick_size=1), opening_state(spec))
        observed = artifact["worlds"][world]
        by_epoch = {row["epoch"]: row for row in observed["epochs"]}
        for epoch in range(spec.epoch_count):
            commands: list[dict[str, Any]] = []
            for agent in spec.agents:
                request = TaskRequest(
                    spec.run_id,
                    spec.task_id(world, epoch, agent),
                    world,
                    epoch,
                    agent,
                    plan.sha256,
                    agent_view(session.snapshot, agent),
                )
                result = compute_task(
                    spec,
                    contract,
                    request,
                    plan.owner(spec, agent),
                    0,
                    collect_draws=False,
                )
                commands.extend(result.commands)
            if commands:
                session.submit_all(
                    tuple(parse_command(command) for command in commands)
                )
            row = by_epoch.get(epoch)
            if row is None or row["applied_commands"] != commands:
                findings.append(
                    Finding("replay", "command_reexecution", world, epoch, "")
                )
            if row is None or row["state_after"] != session.snapshot.to_mapping():
                findings.append(
                    Finding("replay", "state_reexecution", world, epoch, "")
                )
        replay = session.replay
        values[world] = sum(t.price * t.quantity for t in replay.trades)
        if (
            observed["final_state"] != replay.state.to_mapping()
            or observed["trades"] != [t.to_mapping() for t in replay.trades]
            or observed["traded_notional"] != values[world]
        ):
            findings.append(Finding("replay", "output_reexecution", world, -1, ""))
    if (
        artifact.get("metrics", {}).get("paired_traded_notional")
        != values["policy"] - values["baseline"]
    ):
        findings.append(Finding("replay", "metric_reexecution", "pair", -1, ""))
    return tuple(findings)
