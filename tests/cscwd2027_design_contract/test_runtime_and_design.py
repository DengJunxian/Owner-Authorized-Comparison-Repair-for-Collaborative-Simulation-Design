"""Pre-freeze engineering checks, using seeds outside the formal design grid."""

from __future__ import annotations

from dataclasses import replace

import pytest

from policytwin.coordination import auditor, coordinator, replay, workload
from policytwin.coordination.plan import CoordinationExecutionPlan, RunSpec
from policytwin.coordination.release_gate import evaluate, register, seal
from policytwin.design_contract.runtime import SimulationRecipe, configured
from scripts.cscwd2027_design_contract.design import attempts, cases, recipes, sessions
from scripts.cscwd2027_design_contract.store import compare


def recipe(depth: int) -> SimulationRecipe:
    return SimulationRecipe(
        RunSpec("design-unit", "design-unit-not-formal", 0, 4, "cancellation", 2, 1),
        depth,
    )


def test_opening_depth_and_context_authority() -> None:
    deep, shallow = recipe(5000), recipe(24)
    assert deep.opening() == workload.opening_state(deep.spec)
    assert shallow.authority()["opening_sha256"] != deep.authority()["opening_sha256"]
    assert shallow.scientific_key != deep.scientific_key
    assert {
        o.quantity for o in shallow.opening().orders if o.trader_id.startswith("book-")
    } == {24}
    assert (
        next(
            o.quantity
            for o in shallow.opening().orders
            if o.order_id == "target-resting"
        )
        == 20
    )


def test_all_three_aliases_restore_after_exception() -> None:
    modules = (coordinator, auditor, replay)
    old = tuple(vars(module)["opening_state"] for module in modules)
    with pytest.raises(RuntimeError), configured(recipe(24)):
        values = [vars(module)["opening_state"] for module in modules]
        assert values[0] is values[1] is values[2]
        raise RuntimeError("fixture failure")
    assert old == tuple(vars(module)["opening_state"] for module in modules)


def test_shallow_real_process_run_has_correct_local_verification() -> None:
    value = recipe(24)
    plan = CoordinationExecutionPlan(2)
    registration = register(value.spec, plan)
    with configured(value):
        artifact = seal(
            coordinator.run_pair(value.spec, plan, processes=True), registration
        )
        assert evaluate(value.spec, plan, artifact, registration)["release_authorized"]
    assert len(set(artifact["worker_pids"].values())) == 2


def test_deep_adapter_preserves_inherited_scientific_trace() -> None:
    value, plan = recipe(5000), CoordinationExecutionPlan(1)
    inherited = coordinator.run_pair(value.spec, plan, processes=False)
    with configured(value):
        adapted = coordinator.run_pair(value.spec, plan, processes=False)
    result = compare(inherited, adapted)
    assert (
        result["science_equal"]
        and result["draw_equal"]
        and result["contrast_error"] == 0
    )


def test_frozen_grid_recipe_identity_and_counts() -> None:
    blocks = sessions()
    assert len(blocks) == 32
    jobs = [r for s in blocks for r in attempts(s)]
    assert len(jobs) == len({r["id"] for r in jobs}) == 992
    assert sum(len(cases(s)) for s in blocks) == 224
    for block in blocks:
        old, new = recipes(block, 1), recipes(block, 2)
        assert sum(old[k] == new[k] for k in old) == 8
        assert all(
            replace(old[k].spec, branch_epoch=2) == new[k].spec
            for k in old
            if k[0] == block["target"]
        )
        selected = {
            r["candidate"]: r["recipe"]
            for r in attempts(block)
            if r["phase"] == "selective"
        }
        assert set(selected) == {block["target"]}
