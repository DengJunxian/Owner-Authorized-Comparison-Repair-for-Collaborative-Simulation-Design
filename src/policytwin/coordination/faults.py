"""Runtime-only injection hooks. Verifiers never import this module."""

from __future__ import annotations

from dataclasses import dataclass, replace

from policytwin.coordination.plan import CoordinationExecutionPlan, RunSpec
from policytwin.coordination.schema import (
    CoordinationError,
    TaskRequest,
    TaskResult,
    commitment,
)

FAULTS = (
    "task_duplication",
    "task_omission",
    "wrong_owner",
    "duplicate_effect",
    "missing_message",
    "logical_order",
    "payload_mutation",
    "stale_plan",
)


@dataclass(frozen=True, slots=True)
class Fault:
    """A preselected runtime intervention, kept outside auditor input."""

    kind: str
    world: str
    epoch: int
    agent: str

    def __post_init__(self) -> None:
        if self.kind not in FAULTS or self.world not in ("baseline", "policy"):
            raise CoordinationError("invalid fault")

    def matches(self, request: TaskRequest) -> bool:
        """Select a task using only prefixed coordinates."""
        return (request.world, request.epoch, request.agent) == (
            self.world,
            self.epoch,
            self.agent,
        )


def alter_jobs(
    jobs: list[tuple[int, TaskRequest]],
    fault: Fault | None,
    plan: CoordinationExecutionPlan,
) -> list[tuple[int, TaskRequest]]:
    """Apply dispatch/assignment/configuration faults before worker computation."""
    result: list[tuple[int, TaskRequest]] = []
    for owner, request in jobs:
        if fault is not None and fault.matches(request):
            if fault.kind == "task_omission":
                continue
            if fault.kind == "wrong_owner":
                owner = (owner + 1) % plan.worker_count
            if fault.kind == "stale_plan":
                request = replace(request, plan_sha256="0" * 64)
            if fault.kind == "task_duplication":
                result.append((owner, request))
        result.append((owner, request))
    return result


def mutate_quantity(result: TaskResult) -> TaskResult:
    """A worker-side computation fault with a self-consistent payload hash."""
    commands = [dict(command) for command in result.commands]
    for command in commands:
        if command["command_type"] == "market_order":
            command["quantity"] += 1
            break
    mutated = replace(result, commands=tuple(commands))
    return replace(mutated, payload_sha256=commitment(mutated.payload()))


def alter_deliveries(
    arrivals: list[TaskResult],
    fault: Fault | None,
) -> list[TaskResult]:
    """Drop or duplicate messages on the actual coordinator delivery path."""
    result: list[TaskResult] = []
    for message in arrivals:
        match = fault is not None and (message.world, message.epoch, message.agent) == (
            fault.world,
            fault.epoch,
            fault.agent,
        )
        if match and fault is not None and fault.kind == "missing_message":
            continue
        result.append(message)
        if match and fault is not None and fault.kind == "duplicate_effect":
            result.append(message)
    return result


def alter_commit_order(
    messages: list[TaskResult],
    fault: Fault | None,
    spec: RunSpec,
) -> list[TaskResult]:
    """Change coordinator commit order after canonical sorting."""
    ordered = sorted(messages, key=lambda item: spec.agents.index(item.agent))
    if fault is not None and fault.kind == "logical_order":
        for index, message in enumerate(ordered[:-1]):
            if (message.world, message.epoch, message.agent) == (
                fault.world,
                fault.epoch,
                fault.agent,
            ):
                ordered[index], ordered[index + 1] = ordered[index + 1], ordered[index]
                break
    return ordered
