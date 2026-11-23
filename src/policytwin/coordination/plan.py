"""Separate scientific inputs from worker assignment and message scheduling."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Self

from policytwin.coordination.schema import (
    CoordinationError,
    commitment,
    identifier,
    integer,
    object_fields,
)
from policytwin.counterfactual.coupling_contract import (
    CouplingContract,
    EventDeclaration,
    Requirement,
    ScenarioSpecification,
    SemanticEventAddress,
    WorldScope,
    generate_contract,
)

BRANCHES = ("cancellation", "agent_exit", "latency_reorder")
ROLES = ("side", "quantity", "replacement_side", "replacement_quantity", "jitter")


@dataclass(frozen=True, slots=True)
class RunSpec:
    """Scientific workload; invariant to process identity or assignment."""

    study: str
    seed: str
    replicate: int
    agent_count: int
    branch: str
    epoch_count: int = 8
    branch_epoch: int = 3
    schema_version: str = "policytwin.coordination-workload.v1"

    def __post_init__(self) -> None:
        identifier(self.study, "study")
        identifier(self.seed, "seed")
        integer(self.replicate, "replicate")
        integer(self.agent_count, "agent_count", 4)
        integer(self.epoch_count, "epoch_count", 2)
        integer(self.branch_epoch, "branch_epoch", 1)
        if self.agent_count > 96 or self.epoch_count > 32:
            raise CoordinationError("workload exceeds the declared finite bounds")
        if self.branch_epoch >= self.epoch_count or self.branch not in BRANCHES:
            raise CoordinationError("invalid branch or epoch")
        if self.schema_version != "policytwin.coordination-workload.v1":
            raise CoordinationError("invalid workload schema")

    @property
    def agents(self) -> tuple[str, ...]:
        """Stable finite population independent of worker partition."""
        return tuple(f"agent-{i:03d}" for i in range(self.agent_count))

    @property
    def target(self) -> str:
        """The middle population index is fixed before all runs."""
        return self.agents[self.agent_count // 2]

    @property
    def run_id(self) -> str:
        """Bind the complete scientific configuration."""
        return "run-" + commitment(self.to_mapping())

    def to_mapping(self) -> dict[str, Any]:
        """Serialize the immutable scientific inputs."""
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        """Parse a closed workload configuration."""
        return cls(**object_fields(value, set(cls.__dataclass_fields__)))

    def task_id(self, world: str, epoch: int, agent: str) -> str:
        """Name a semantic task without worker metadata."""
        if world not in ("baseline", "policy") or agent not in self.agents:
            raise CoordinationError("invalid task coordinate")
        if not 0 <= epoch < self.epoch_count:
            raise CoordinationError("task epoch out of range")
        return "task-" + commitment([self.run_id, world, epoch, agent])

    def active(self, world: str, epoch: int, agent: str) -> bool:
        """Agent exit is a normal inactive task state, not missing evidence."""
        return not (
            world == "policy"
            and self.branch == "agent_exit"
            and epoch >= self.branch_epoch
            and agent == self.target
        )

    def consumes(self, world: str, epoch: int, agent: str, role: str) -> bool:
        """Declare bounded role activation independently of observed draws."""
        if role in ("side", "quantity"):
            return self.active(world, epoch, agent)
        if world != "policy" or agent != self.target:
            return False
        if role in ("replacement_side", "replacement_quantity"):
            return self.branch == "cancellation" and epoch == self.branch_epoch
        return self.branch == "latency_reorder" and epoch >= self.branch_epoch


@dataclass(frozen=True, slots=True)
class CoordinationExecutionPlan:
    """Static ownership and logical scheduling rules; no RNG namespace fields."""

    worker_count: int
    assignment: str = "round_robin"
    version: str = "cscwd-coordination-v1"
    schedule: str = "epoch-view-agent-index-v1"
    schema_version: str = "policytwin.coordination-plan.v1"

    def __post_init__(self) -> None:
        if type(self.worker_count) is not int or self.worker_count not in (1, 2, 4):
            raise CoordinationError("worker_count must be 1, 2 or 4")
        if self.assignment not in ("round_robin", "contiguous"):
            raise CoordinationError("invalid assignment")
        identifier(self.version, "version")
        if self.schedule != "epoch-view-agent-index-v1":
            raise CoordinationError("unsupported logical schedule")
        if self.schema_version != "policytwin.coordination-plan.v1":
            raise CoordinationError("invalid plan schema")

    def owner(self, spec: RunSpec, agent: str) -> int:
        """Compute ownership from the expected population, not a runtime trace."""
        index = spec.agents.index(agent)
        if self.assignment == "round_robin":
            return index % self.worker_count
        return min(index * self.worker_count // spec.agent_count, self.worker_count - 1)

    def to_mapping(self) -> dict[str, Any]:
        """Serialize scheduling metadata separately from scientific inputs."""
        return asdict(self)

    @property
    def sha256(self) -> str:
        """Content-address this particular partition plan."""
        return commitment(self.to_mapping())

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        """Parse an independently stored execution plan."""
        return cls(**object_fields(value, set(cls.__dataclass_fields__)))


def compile_contract(spec: RunSpec) -> CouplingContract:
    """Build a finite potential-event set before either world runs."""
    declarations = tuple(
        EventDeclaration(
            event_type="agent_decision",
            draw_role=role,
            distribution_family="discrete_uniform_inclusive",
            distribution_parameters=(("lower", 0), ("upper", (1 << 32) - 1)),
            coupling_group="paired-inputs",
            world_scope=WorldScope.SHARED
            if role in ("side", "quantity")
            else WorldScope.POLICY_ONLY,
            requirement=Requirement.OPTIONAL,
        )
        for role in ROLES
    )
    return generate_contract(
        ScenarioSpecification(
            base_experiment_id="cscwd2027-coordination",
            scenario_namespace=spec.run_id,
            logical_epochs=tuple(range(spec.epoch_count)),
            entity_ids=spec.agents,
            declarations=declarations,
            configuration_version="coordination-workload-v1",
        ),
        spec.replicate,
    )


def event_lookup(
    contract: CouplingContract,
) -> dict[tuple[int, str, str], SemanticEventAddress]:
    """Index complete event declarations by their semantic workload coordinate."""
    return {(e.logical_epoch, e.entity_id, e.draw_role): e for e in contract.events}
