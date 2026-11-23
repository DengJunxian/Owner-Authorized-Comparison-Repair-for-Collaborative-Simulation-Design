"""Executable fixtures for the new protocol; these are not formal-study seeds."""

from __future__ import annotations

import ast
import copy
import inspect
import os
from dataclasses import asdict, replace
from fractions import Fraction
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from policytwin.coordination.auditor import coordination_conformance, event_conformance
from policytwin.coordination.coordinator import (
    draw_projection,
    run_pair,
    run_world,
    scientific_projection,
)
from policytwin.coordination.faults import FAULTS, Fault
from policytwin.coordination.plan import (
    CoordinationExecutionPlan,
    RunSpec,
    compile_contract,
)
from policytwin.coordination.release_gate import evaluate, register, report, seal
from policytwin.coordination.replay import replay_valid
from policytwin.coordination.schema import (
    AgentView,
    CoordinationError,
    TaskRequest,
    TaskResult,
    commitment,
)
from policytwin.coordination.worker_runtime import WorkerPool
from policytwin.coordination.workload import (
    action,
    agent_view,
    compute_task,
    opening_state,
)
from policytwin.experiments.propagation_scaling import _action
from policytwin.market.models import MarketConfig
from policytwin.market.session import ReplayMarketSession


def spec(branch: str = "cancellation") -> RunSpec:
    return RunSpec("unit-only", "unit-only-seed", 0, 6, branch, 4, 1)


@pytest.mark.parametrize("branch", ("cancellation", "agent_exit", "latency_reorder"))
def test_reference_conforms_and_reexecutes(branch: str) -> None:
    workload = spec(branch)
    plan = CoordinationExecutionPlan(2)
    registration = register(workload, plan)
    artifact = run_pair(workload, plan, processes=False)
    assert all(w["completed"] for w in artifact["worlds"].values())
    assert not event_conformance(workload, artifact)
    assert not coordination_conformance(workload, plan, artifact)
    assert not replay_valid(workload, plan, artifact)
    assert evaluate(workload, plan, seal(artifact, registration), registration)[
        "release_authorized"
    ]


@pytest.mark.parametrize("workers", (1, 2, 4))
@pytest.mark.parametrize("branch", ("cancellation", "agent_exit", "latency_reorder"))
def test_real_process_partition_exactly_matches_reference(
    workers: int, branch: str
) -> None:
    workload = spec(branch)
    plan = CoordinationExecutionPlan(workers)
    reference = run_pair(workload, plan, processes=False)
    observed = run_pair(workload, plan)
    assert len(set(observed["worker_pids"].values())) == workers
    assert os.getpid() not in observed["worker_pids"].values()
    used = {
        r["worker_pid"]
        for w in observed["worlds"].values()
        for e in w["epochs"]
        for r in e["results"]
    }
    assert used == set(observed["worker_pids"].values())
    assert draw_projection(observed) == draw_projection(reference)
    for world in ("baseline", "policy"):
        assert scientific_projection(
            observed["worlds"][world]
        ) == scientific_projection(reference["worlds"][world])
    assert observed["metrics"] == reference["metrics"]
    assert not coordination_conformance(workload, plan, observed)


@pytest.mark.parametrize("transport", ("normal", "reverse", "delay"))
def test_benign_transport_changes_preserve_scientific_trace(transport: str) -> None:
    workload = spec()
    plan = CoordinationExecutionPlan(4)
    expected = run_pair(workload, plan, processes=False)
    artifact = run_pair(workload, plan, transport=transport)
    assert not coordination_conformance(workload, plan, artifact)
    assert artifact["metrics"] == expected["metrics"]
    assert (
        artifact["worlds"]["policy"]["scientific_sha256"]
        == expected["worlds"]["policy"]["scientific_sha256"]
    )
    if transport == "reverse":
        row = artifact["worlds"]["policy"]["epochs"][0]
        assert [r["message_id"] for r in row["deliveries"]] == list(
            reversed(row["committed_message_ids"])
        )


@pytest.mark.parametrize("kind", FAULTS)
def test_faults_are_retained_and_full_report_is_blocked(
    kind: str, tmp_path: Path
) -> None:
    workload = spec()
    plan = CoordinationExecutionPlan(4)
    registration = register(workload, plan)
    artifact = run_pair(
        workload,
        plan,
        processes=False,
        fault=Fault(kind, "policy", 1, workload.agents[1]),
    )
    capsule = seal(artifact, registration)
    decision = report(workload, plan, capsule, registration, tmp_path / "full")
    assert decision["findings"]
    assert not decision["numerical_file_exists"]
    assert (tmp_path / "full/decision.json").is_file()
    assert artifact["worlds"]["policy"]["epochs"]


@pytest.mark.parametrize("kind", ("wrong_owner", "payload_mutation", "stale_plan"))
def test_fault_is_on_real_worker_path(kind: str) -> None:
    workload = spec()
    plan = CoordinationExecutionPlan(2)
    registration = register(workload, plan)
    artifact = run_pair(
        workload, plan, fault=Fault(kind, "policy", 1, workload.agents[1])
    )
    assert not evaluate(workload, plan, seal(artifact, registration), registration)[
        "release_authorized"
    ]
    row = artifact["worlds"]["policy"]["epochs"][1]
    result = next(r for r in row["results"] if r["agent"] == workload.agents[1])
    assert result["worker_pid"] in artifact["worker_pids"].values()


def test_wrong_owner_can_be_numerically_quiet_but_violates_plan(tmp_path: Path) -> None:
    workload = spec()
    plan = CoordinationExecutionPlan(4)
    registration = register(workload, plan)
    reference = run_pair(workload, plan, processes=False)
    artifact = run_pair(
        workload,
        plan,
        processes=False,
        fault=Fault("wrong_owner", "policy", 1, workload.agents[1]),
    )
    assert artifact["metrics"] == reference["metrics"]
    assert not replay_valid(workload, plan, artifact)
    capsule = seal(artifact, registration)
    assert report(
        workload, plan, capsule, registration, tmp_path / "keys", "keys_only"
    )["numerical_file_exists"]
    assert not report(workload, plan, capsule, registration, tmp_path / "full")[
        "numerical_file_exists"
    ]
    minus = report(
        workload, plan, capsule, registration, tmp_path / "minus", "minus_gate"
    )
    assert minus["findings"] and minus["numerical_file_exists"]


@pytest.mark.parametrize(
    "field", ("spec", "metrics", "registration", "finished_at_utc")
)
def test_capsule_mutation_fails_closed(field: str, tmp_path: Path) -> None:
    workload = spec()
    plan = CoordinationExecutionPlan(1)
    registration = register(workload, plan)
    capsule = seal(run_pair(workload, plan, processes=False), registration)
    changed = copy.deepcopy(capsule)
    if field == "metrics":
        changed[field].clear()
    elif field == "spec":
        changed[field]["seed"] = "stale-seed"
    elif field == "registration":
        changed[field]["plan_sha256"] = "0" * 64
    else:
        changed[field] = "2099-01-01T00:00:00+00:00"
    assert not report(workload, plan, changed, registration, tmp_path / "report")[
        "numerical_file_exists"
    ]


def test_task_schema_and_record_commitment() -> None:
    workload = spec()
    plan = CoordinationExecutionPlan(1)
    snapshot = ReplayMarketSession(
        MarketConfig(tick_size=1), opening_state(workload)
    ).snapshot
    task = TaskRequest(
        workload.run_id,
        workload.task_id("policy", 1, workload.target),
        "policy",
        1,
        workload.target,
        plan.sha256,
        agent_view(snapshot, workload.target),
    )
    assert TaskRequest.from_mapping(task.to_mapping()) == task
    bad = task.to_mapping() | {"unexpected": True}
    with pytest.raises(CoordinationError):
        TaskRequest.from_mapping(bad)
    with pytest.raises(CoordinationError):
        replace(task, epoch=True)
    result = compute_task(workload, compile_contract(workload), task, 0, 0)
    assert TaskResult.from_mapping(result.to_mapping()) == result
    assert result.payload_sha256 == commitment(result.payload())


@given(
    side=st.integers(min_value=0, max_value=2**32 - 1),
    quantity=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_policy_projection_matches_existing_feedback_rule(
    side: int, quantity: int
) -> None:
    workload = spec()
    snapshot = ReplayMarketSession(
        MarketConfig(tick_size=1), opening_state(workload)
    ).snapshot
    for agent in workload.agents:
        assert action(agent_view(snapshot, agent), side, quantity) == _action(
            snapshot, side, quantity, agent
        )


def test_scientific_inputs_exclude_partition_metadata() -> None:
    workload = spec()
    addresses = [e.to_mapping() for e in compile_contract(workload).events]
    assert all(
        "worker" not in str(address) and "plan_sha256" not in address
        for address in addresses
    )
    plans = [
        CoordinationExecutionPlan(n, assignment)
        for n in (1, 2, 4)
        for assignment in ("round_robin", "contiguous")
    ]
    assert len({p.sha256 for p in plans}) == 6
    assert "worker_count" not in asdict(workload)


def test_gate_selection_counterexample_is_exact() -> None:
    # Z ~ Bernoulli(1/2); G = [Z == 1]. The accepted mean is 1, not 1/2.
    mass = {0: Fraction(1, 2), 1: Fraction(1, 2)}
    accepted = {z: p for z, p in mass.items() if z == 1}
    normalization = sum(accepted.values())
    assert sum(z * p / normalization for z, p in accepted.items()) == 1
    assert sum(z * p for z, p in mass.items()) == Fraction(1, 2)
    # Retaining only all-pass two-replicate batches retains only (1,1).
    batch = {(a, b): pa * pb for a, pa in mass.items() for b, pb in mass.items()}
    passed = {pair: p for pair, p in batch.items() if all(pair)}
    assert sum(passed.values()) == Fraction(1, 4)
    assert (
        sum(Fraction(sum(pair), 2) * p for pair, p in passed.items())
        / sum(passed.values())
        == 1
    )


def test_equal_obligation_conventional_adapter_matches_full() -> None:
    workload = spec()
    plan = CoordinationExecutionPlan(2)
    registration = register(workload, plan)
    capsule = seal(run_pair(workload, plan, processes=False), registration)
    full = evaluate(workload, plan, capsule, registration, "full")
    conventional = evaluate(workload, plan, capsule, registration, "conventional")
    assert full["findings"] == conventional["findings"]
    assert full["release_authorized"] == conventional["release_authorized"]


@pytest.mark.parametrize("value", (float("nan"), float("inf"), True, -1))
def test_view_rejects_nonfinite_noninteger_or_negative_values(value: object) -> None:
    with pytest.raises(CoordinationError):
        AgentView.from_mapping(
            {
                "state_sha256": "0" * 64,
                "best_bid": value,
                "best_ask": 0,
                "bid_quantity": 0,
                "ask_quantity": 0,
                "cash": 0,
                "asset": 0,
            }
        )


def test_worker_crash_is_retained_and_children_are_reaped() -> None:
    workload = spec()
    pool = WorkerPool(workload, CoordinationExecutionPlan(2))
    try:
        assert pool.execute([]) == []
        pool.processes[0].terminate()
        pool.processes[0].join(timeout=2)
        world = run_world(workload, pool.plan, "policy", pool)
        assert not world["completed"]
        assert world["native_error"]
        assert len(world["epochs"]) == 1
    finally:
        pool.close()
    assert not any(process.is_alive() for process in pool.processes)


def test_no_evidence_retains_same_numerical_workload() -> None:
    workload = spec()
    plan = CoordinationExecutionPlan(1)
    full = run_pair(workload, plan, processes=False)
    minimal = run_pair(
        workload, plan, processes=False, collect_draws=False, retain_trace=False
    )
    assert minimal["metrics"] == full["metrics"]
    for world in ("baseline", "policy"):
        assert minimal["worlds"][world]["epochs"] == []
        assert (
            minimal["worlds"][world]["final_state"]
            == full["worlds"][world]["final_state"]
        )
        assert minimal["worlds"][world]["trades"] == full["worlds"][world]["trades"]


@pytest.mark.parametrize("field", ("world", "agent", "epoch", "run_id"))
def test_dispatch_semantic_identity_is_checked(field: str) -> None:
    workload = spec()
    plan = CoordinationExecutionPlan(1)
    artifact = run_pair(workload, plan, processes=False)
    request = artifact["worlds"]["policy"]["epochs"][1]["requests"][0]
    request[field] = {
        "world": "baseline",
        "agent": workload.agents[1],
        "epoch": 0,
        "run_id": "wrong-run",
    }[field]
    assert any(
        f.code == "dispatch_identity"
        for f in coordination_conformance(workload, plan, artifact)
    )


def test_verifiers_have_no_fault_hook_import() -> None:
    from policytwin.coordination import auditor, release_gate, replay

    for module in (auditor, release_gate, replay):
        tree = ast.parse(inspect.getsource(module))
        imports = [
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        ]
        assert not any(name.endswith(("faults", "coordinator")) for name in imports)
