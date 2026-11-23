"""Coupling-only intervention over the existing CLOB and unchanged local gate.

Worker entry points explicitly enter the same declared coupling context as
the coordinator and replay. No old source, recipe or result is modified.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Any, Self
from unittest.mock import patch

from policytwin.coordination import (
    auditor,
    coordinator,
    plan,
    release_gate,
    replay,
    worker_runtime,
    workload,
)
from policytwin.coordination.schema import commitment
from policytwin.counterfactual.coupling_contract import (
    CouplingContract,
    EventDeclaration,
    Requirement,
    ScenarioSpecification,
    World,
    WorldScope,
    generate_contract,
)
from policytwin.design_contract.contract import Evidence
from policytwin.design_contract.runtime import SimulationRecipe, configured
from policytwin.market.models import MarketConfig
from policytwin.provenance import canonical_json_bytes
from scripts.cscwd2027_design_contract.store import save

COUPLINGS = ("shared", "world_separated")
ORIGINAL_COMPUTE = workload.compute_task
ORIGINAL_WORKER = worker_runtime._worker


def digest(path: Path) -> str:
    """Hash an existing file without rewriting it."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compile_coupling(spec: plan.RunSpec, *, coupling: str) -> CouplingContract:
    """Keep per-world draw marginals fixed and vary only cross-world scope."""
    if coupling not in COUPLINGS:
        raise ValueError("unknown coupling")
    declarations = []
    for role in plan.ROLES:
        scopes = (
            (
                (WorldScope.SHARED,)
                if coupling == "shared"
                else (WorldScope.BASELINE_ONLY, WorldScope.POLICY_ONLY)
            )
            if role in ("side", "quantity")
            else (WorldScope.POLICY_ONLY,)
        )
        for scope in scopes:
            declarations.append(
                EventDeclaration(
                    "agent_decision",
                    role,
                    "discrete_uniform_inclusive",
                    (("lower", 0), ("upper", (1 << 32) - 1)),
                    "paired-inputs",
                    scope,
                    Requirement.OPTIONAL,
                )
            )
    return generate_contract(
        ScenarioSpecification(
            "cscwd2027-coordination",
            spec.run_id,
            tuple(range(spec.epoch_count)),
            spec.agents,
            tuple(declarations),
            "coordination-workload-v1",
        ),
        spec.replicate,
    )


def marginal_schema(contract: CouplingContract, world: str) -> list[tuple[Any, ...]]:
    """Project the declared one-world draw law, not observed outcome moments."""
    return sorted(
        (
            e.logical_epoch,
            e.entity_id,
            e.draw_role,
            e.distribution_family,
            e.distribution_parameters,
            e.requirement.value,
        )
        for e in contract.events
        if e.world_scope.applies_to(World(world))
    )


def compute_task(spec, contract, request, worker_id, worker_pid, **kwargs):
    """Supply the unchanged decision function its world's unique role map."""
    selected = replace(
        contract,
        events=tuple(
            e
            for e in contract.events
            if e.draw_role not in ("side", "quantity")
            or e.world_scope.applies_to(World(request.world))
        ),
    )
    return ORIGINAL_COMPUTE(spec, selected, request, worker_id, worker_pid, **kwargs)


def coupled_worker(spec_data, worker_id, incoming, outgoing, *, coupling):
    """Picklable spawn target: explicitly configure the child, not just parent."""
    with coupling_context(coupling):
        ORIGINAL_WORKER(spec_data, worker_id, incoming, outgoing)


@contextmanager
def coupling_context(coupling: str) -> Iterator[None]:
    """Scope every producer/auditor/replay alias and restore it on all exits."""
    if coupling not in COUPLINGS:
        raise ValueError("unknown coupling")
    factory = partial(compile_coupling, coupling=coupling)
    with ExitStack() as stack:
        for module in (
            plan,
            coordinator,
            worker_runtime,
            auditor,
            replay,
            release_gate,
        ):
            stack.enter_context(patch.object(module, "compile_contract", factory))
        for module in (workload, coordinator, worker_runtime, replay):
            stack.enter_context(patch.object(module, "compute_task", compute_task))
        stack.enter_context(
            patch.object(
                worker_runtime, "_worker", partial(coupled_worker, coupling=coupling)
            )
        )
        yield


@dataclass(frozen=True)
class CoupledRecipe:
    """A fixed scientific recipe with an independently declared joint draw law."""

    recipe: SimulationRecipe
    coupling: str

    def __post_init__(self):
        if self.coupling not in COUPLINGS:
            raise ValueError("unknown coupling")

    def to_mapping(self) -> dict[str, Any]:
        return {"recipe": self.recipe.to_mapping(), "coupling": self.coupling}

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> Self:
        if set(value) != {"recipe", "coupling"}:
            raise ValueError("unknown coupled-recipe field")
        return cls(SimulationRecipe.from_mapping(value["recipe"]), value["coupling"])

    def key_payload(self) -> dict[str, Any]:
        authority = self.recipe.authority()
        return {
            **authority,
            "source_sha256": commitment(
                {
                    "inherited": authority["source_sha256"],
                    "coupling_adapter": digest(Path(__file__)),
                }
            ),
            "market_config": MarketConfig(tick_size=1).to_mapping(),
            "coupling_contract_sha256": compile_coupling(
                self.recipe.spec, coupling=self.coupling
            ).sha256,
        }

    @property
    def scientific_key(self) -> str:
        return commitment(self.key_payload())

    @property
    def key_without_coupling(self) -> str:
        return commitment(
            {
                k: v
                for k, v in self.key_payload().items()
                if k != "coupling_contract_sha256"
            }
        )


def execute(row: dict[str, Any], root: Path, freeze_sha256: str) -> dict[str, Any]:
    """Run both real spawn worlds and the unchanged complete evidence gate."""
    folder = root / "objects" / row["id"]
    folder.mkdir(parents=True, exist_ok=False)
    recipe = CoupledRecipe.from_mapping(row["coupled_recipe"])
    allocation = plan.CoordinationExecutionPlan.from_mapping(row["plan"])
    spec = recipe.recipe.spec
    result: dict[str, Any] = {"status": "failed", "design": row, "id": row["id"]}
    begin = perf_counter()
    try:
        with configured(recipe.recipe), coupling_context(recipe.coupling):
            registration = release_gate.register(spec, allocation)
            save(
                folder / "attempt.json",
                {
                    "design": row,
                    "freeze_sha256": freeze_sha256,
                    "key_payload": recipe.key_payload(),
                    "registration": asdict(registration),
                    "coupling_contract": compile_coupling(
                        spec, coupling=recipe.coupling
                    ).to_mapping(),
                },
            )
            start = perf_counter()
            capsule = release_gate.seal(
                coordinator.run_pair(spec, allocation, processes=True), registration
            )
            execute_seconds = perf_counter() - start
            start = perf_counter()
            with (folder / "capsule.json.gz").open("xb") as stream:
                stream.write(gzip.compress(canonical_json_bytes(capsule), mtime=0))
            store_seconds = perf_counter() - start
            start = perf_counter()
            decision = release_gate.report(
                spec, allocation, capsule, registration, folder / "local_report", "full"
            )
            gate_seconds = perf_counter() - start
        if len(set(capsule["worker_pids"].values())) != allocation.worker_count:
            raise RuntimeError("spawn worker count differs from submitted plan")
        result.update(
            {
                "status": "completed",
                "local_pass": decision["release_authorized"],
                "scientific_key": recipe.scientific_key,
                "key_without_coupling": recipe.key_without_coupling,
                "key_payload": recipe.key_payload(),
                "capsule_sha256": capsule["capsule_sha256"],
                "metric": capsule["metrics"]["paired_traded_notional"],
                "world_notionals": {
                    w: v["traded_notional"] for w, v in capsule["worlds"].items()
                },
                "local_decision": decision,
                "timing_seconds": {
                    "spawn_execution": execute_seconds,
                    "capsule_storage": store_seconds,
                    "full_gate_and_report": gate_seconds,
                },
                "files": {
                    str(p.relative_to(folder)): digest(p)
                    for p in sorted(folder.rglob("*"))
                    if p.is_file()
                },
            }
        )
    except Exception as error:
        result.update(
            {
                "local_pass": False,
                "metric": None,
                "error": f"{type(error).__name__}: {error}",
            }
        )
    result["elapsed_seconds"] = perf_counter() - begin
    save(folder / "result.json", result)
    return result


def load(identifier: str, root: Path) -> tuple[Evidence, dict[str, Any]]:
    """Validate actual local files and the current coupled-recipe authority."""
    if re.fullmatch(r"[A-Za-z0-9_-]+", identifier) is None:
        raise ValueError("invalid object ID")
    folder = root / "objects" / identifier
    value = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    recipe = CoupledRecipe.from_mapping(value["design"]["coupled_recipe"])
    allocation = plan.CoordinationExecutionPlan.from_mapping(value["design"]["plan"])
    capsule = {}
    passed = value["status"] == "completed" and value["local_pass"]
    if value["status"] == "completed":
        for name, expected in value["files"].items():
            if digest(folder / name) != expected:
                raise RuntimeError("stored evidence commitment differs")
        if value["key_payload"] != recipe.key_payload():
            raise RuntimeError("coupling verification context changed")
        if value["scientific_key"] != recipe.scientific_key:
            raise RuntimeError("scientific key differs")
        if value["key_without_coupling"] != recipe.key_without_coupling:
            raise RuntimeError("key projection differs")
        capsule = json.loads(gzip.decompress((folder / "capsule.json.gz").read_bytes()))
        if capsule["capsule_sha256"] != commitment(
            {k: v for k, v in capsule.items() if k != "capsule_sha256"}
        ):
            raise RuntimeError("capsule content differs")
        if passed:
            report = json.loads(
                (folder / "local_report/numerical_report.json").read_text()
            )
            if report["capsule_sha256"] != capsule["capsule_sha256"]:
                raise RuntimeError("numerical file binding differs")
            if report["metrics"] != capsule["metrics"]:
                raise RuntimeError("numerical metric binding differs")
    return Evidence(
        identifier,
        value.get("scientific_key", ""),
        allocation.sha256,
        value.get("capsule_sha256", ""),
        bool(passed),
        value.get("metric"),
        recipe.recipe.spec.branch,
        recipe.recipe.spec.replicate,
    ), capsule
