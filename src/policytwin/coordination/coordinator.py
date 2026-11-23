"""Epoch-barrier coordinator over the existing deterministic CLOB transition."""

from __future__ import annotations

import os
from dataclasses import asdict
from time import perf_counter, sleep
from typing import Any

from policytwin.coordination.faults import (
    Fault,
    alter_commit_order,
    alter_deliveries,
    alter_jobs,
    mutate_quantity,
)
from policytwin.coordination.plan import (
    CoordinationExecutionPlan,
    RunSpec,
    compile_contract,
)
from policytwin.coordination.schema import TaskRequest, commitment
from policytwin.coordination.worker_runtime import WorkerPool
from policytwin.coordination.workload import (
    agent_view,
    compute_task,
    opening_state,
    parse_command,
)
from policytwin.market.models import MarketConfig
from policytwin.market.session import ReplayMarketSession


def scientific_projection(world: dict[str, Any]) -> dict[str, Any]:
    """Predeclared scientific trace: excludes transport, assignment and process IDs."""
    return {
        "epochs": [
            {
                "epoch": row["epoch"],
                "commands": row["applied_commands"],
                "state": row["state_after"],
            }
            for row in world["epochs"]
        ],
        "final_state": world["final_state"],
        "trades": world["trades"],
        "traded_notional": world["traded_notional"],
        "completed": world["completed"],
    }


def run_world(
    spec: RunSpec,
    plan: CoordinationExecutionPlan,
    world: str,
    pool: WorkerPool | None,
    *,
    fault: Fault | None = None,
    transport: str = "normal",
    collect_draws: bool = True,
    retain_trace: bool = True,
) -> dict[str, Any]:
    """Execute one world, retaining incomplete/native-rejected traces on failure."""
    if transport not in ("normal", "delay", "reverse"):
        raise ValueError("unknown transport mode")
    started = perf_counter()
    contract = compile_contract(spec)
    session = ReplayMarketSession(MarketConfig(tick_size=1), opening_state(spec))
    epochs: list[dict[str, Any]] = []
    native_error: str | None = None
    execution_seconds = 0.0
    market_seconds = 0.0
    for epoch in range(spec.epoch_count):
        snapshot = session.snapshot
        requests = [
            TaskRequest(
                spec.run_id,
                spec.task_id(world, epoch, agent),
                world,
                epoch,
                agent,
                plan.sha256,
                agent_view(snapshot, agent),
            )
            for agent in spec.agents
        ]
        jobs = alter_jobs(
            [(plan.owner(spec, r.agent), r) for r in requests], fault, plan
        )
        mutation_tasks = frozenset(
            r.task_id
            for _, r in jobs
            if fault is not None
            and fault.matches(r)
            and fault.kind == "payload_mutation"
        )
        row: dict[str, Any] = {
            "epoch": epoch,
            "state_before_sha256": snapshot.sha256,
            "requests": [request.to_mapping() for _, request in jobs],
            "results": [],
            "deliveries": [],
            "committed_message_ids": [],
            "applied_commands": [],
            "state_after": snapshot.to_mapping(),
        }
        if retain_trace:
            epochs.append(row)
        try:
            begin = perf_counter()
            if pool is None:
                arrivals = []
                for owner, request in jobs:
                    result = compute_task(
                        spec,
                        contract,
                        request,
                        owner,
                        os.getpid(),
                        collect_draws=collect_draws,
                    )
                    if request.task_id in mutation_tasks:
                        result = mutate_quantity(result)
                    arrivals.append(result)
            else:
                arrivals = pool.execute(
                    jobs,
                    collect_draws=collect_draws,
                    quantity_mutation_tasks=mutation_tasks,
                )
            execution_seconds += perf_counter() - begin
            if retain_trace:
                row["results"] = [item.to_mapping() for item in arrivals]
                row["physical_arrival_ids"] = [item.message_id for item in arrivals]
            released = sorted(arrivals, key=lambda item: spec.agents.index(item.agent))
            if transport == "delay":
                # Transport time is not simulated market time or a random draw.
                sleep(0.001)
            if transport == "reverse":
                released.reverse()
            deliveries = alter_deliveries(released, fault)
            if retain_trace:
                row["deliveries"] = [item.to_mapping() for item in deliveries]
            ordered = alter_commit_order(deliveries, fault, spec)
            row["committed_message_ids"] = [item.message_id for item in ordered]
            commands = [command for item in ordered for command in item.commands]
            row["applied_commands"] = commands
            begin = perf_counter()
            if commands:
                session.submit_all(
                    tuple(parse_command(command) for command in commands)
                )
            market_seconds += perf_counter() - begin
            if retain_trace:
                row["state_after"] = session.snapshot.to_mapping()
        except Exception as error:
            native_error = f"{type(error).__name__}: {error}"
            row["native_error"] = native_error
            break
    replay = session.replay
    world_result: dict[str, Any] = {
        "world": world,
        "epochs": epochs,
        "native_error": native_error,
        "completed": native_error is None,
        "trace_retained": retain_trace,
        "final_state": replay.state.to_mapping(),
        "trades": [trade.to_mapping() for trade in replay.trades],
        "traded_notional": sum(trade.price * trade.quantity for trade in replay.trades),
        "runtime_seconds": perf_counter() - started,
        "task_dispatch_compute_seconds": execution_seconds,
        "market_transition_seconds": market_seconds,
    }
    world_result["scientific_sha256"] = commitment(scientific_projection(world_result))
    return world_result


def run_pair(
    spec: RunSpec,
    plan: CoordinationExecutionPlan,
    *,
    processes: bool = True,
    fault: Fault | None = None,
    transport: str = "normal",
    collect_draws: bool = True,
    retain_trace: bool = True,
) -> dict[str, Any]:
    """Run both worlds in one fresh worker pool, without filtering failures."""
    started = perf_counter()
    pool = WorkerPool(spec, plan) if processes else None
    try:
        worlds = {
            world: run_world(
                spec,
                plan,
                world,
                pool,
                fault=fault,
                transport=transport,
                collect_draws=collect_draws,
                retain_trace=retain_trace,
            )
            for world in ("baseline", "policy")
        }
        artifact: dict[str, Any] = {
            "schema_version": "policytwin.coordination-pair.v1",
            "spec": asdict(spec),
            "plan": asdict(plan),
            "contract_sha256": compile_contract(spec).sha256,
            "process_mode": "spawn" if processes else "in_process_reference",
            "worker_pids": {}
            if pool is None
            else {str(k): v for k, v in pool.pids.items()},
            "worlds": worlds,
            "metrics": {
                "paired_traded_notional": worlds["policy"]["traded_notional"]
                - worlds["baseline"]["traded_notional"]
            },
            "startup_seconds": 0.0 if pool is None else pool.startup_seconds,
            "worker_compute_elapsed_sum_seconds": 0.0
            if pool is None
            else pool.compute_seconds,
        }
    finally:
        if pool is not None:
            pool.close()
    artifact["execution_total_seconds"] = perf_counter() - started
    return artifact


def draw_projection(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    """Stable draw records for paired implementation comparisons."""
    draws = [
        draw
        for world in artifact["worlds"].values()
        for epoch in world["epochs"]
        for result in epoch["results"]
        for draw in result["draws"]
    ]
    return sorted(draws, key=lambda draw: (draw["world"], draw["event_identity"]))
