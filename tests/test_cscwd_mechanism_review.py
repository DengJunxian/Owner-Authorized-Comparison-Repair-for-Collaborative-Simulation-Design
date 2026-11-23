"""Hand-checkable fixtures for the post-review coupling and repair studies."""

from dataclasses import replace

import pytest

from policytwin.coordination import plan, workload
from policytwin.coordination.plan import CoordinationExecutionPlan, RunSpec
from policytwin.cscwd_mechanism.coupling import (
    CoupledRecipe,
    compile_coupling,
    coupling_context,
    execute,
    load,
    marginal_schema,
)
from policytwin.cscwd_mechanism.repair import (
    localization_score,
    repair_panel,
    semantic_truth,
)
from policytwin.design_contract.contract import (
    ComparisonContract,
    Evidence,
    ExpectedSlot,
    Member,
    revision_plan,
)
from policytwin.design_contract.runtime import SimulationRecipe


@pytest.fixture
def recipe():
    return SimulationRecipe(
        RunSpec("engineering-fixture", "fixed", 0, 6, "cancellation", 4, 1), 36
    )


def test_one_world_draw_laws_are_unchanged(recipe):
    shared = compile_coupling(recipe.spec, coupling="shared")
    separate = compile_coupling(recipe.spec, coupling="world_separated")
    assert shared == plan.compile_contract(recipe.spec)
    assert shared.sha256 != separate.sha256
    for world in ("baseline", "policy"):
        assert marginal_schema(shared, world) == marginal_schema(separate, world)


def test_coupling_projection_drops_exactly_one_binding(recipe):
    a, b = (CoupledRecipe(recipe, c) for c in ("shared", "world_separated"))
    assert a.scientific_key != b.scientific_key
    assert a.key_without_coupling == b.key_without_coupling
    assert {k for k in a.key_payload() if a.key_payload()[k] != b.key_payload()[k]} == {
        "coupling_contract_sha256"
    }


@pytest.mark.parametrize("field,value", [("seed", "other"), ("replicate", 1)])
def test_random_identity_is_bound_even_without_coupling(recipe, field, value):
    changed = replace(recipe, spec=replace(recipe.spec, **{field: value}))
    assert recipe.spec.run_id != changed.spec.run_id
    assert (
        CoupledRecipe(recipe, "shared").key_without_coupling
        != CoupledRecipe(changed, "shared").key_without_coupling
    )


def test_alias_restoration_on_exception():
    original, compute = plan.compile_contract, workload.compute_task
    with pytest.raises(RuntimeError), coupling_context("world_separated"):
        assert plan.compile_contract is not original
        raise RuntimeError("fixture")
    assert plan.compile_contract is original
    assert workload.compute_task is compute


@pytest.mark.parametrize("coupling", ["shared", "world_separated"])
def test_actual_spawn_and_unchanged_full_gate(recipe, coupling, tmp_path):
    row = {
        "id": "engineering-" + coupling,
        "coupled_recipe": CoupledRecipe(recipe, coupling).to_mapping(),
        "plan": CoordinationExecutionPlan(2).to_mapping(),
    }
    result = execute(row, tmp_path, "engineering-not-study")
    assert result["status"] == "completed", result
    assert result["local_pass"], result
    token, capsule = load(row["id"], tmp_path)
    assert token.local_pass
    assert token.value == capsule["metrics"]["paired_traded_notional"]
    assert len(set(capsule["worker_pids"].values())) == 2


def fixtures():
    tokens = {
        c: Evidence(c, "key-" + c, "plan", "capsule", True, i, c, 0)
        for i, c in enumerate(("a", "b"))
    }
    contract = ComparisonContract(
        "fixture", 1, tuple(ExpectedSlot(c, 0, "key-" + c) for c in tokens), ("plan",)
    )
    return contract, tokens


@pytest.mark.parametrize(
    "method", ["slot_directed_batch", "slot_directed_serial", "whole_panel_refresh"]
)
def test_actual_repair_missing_and_duplicate(method, tmp_path):
    contract, tokens = fixtures()
    members = [Member("a", 0, tokens["a"]), Member("a", 0, tokens["a"])]
    result, repaired = repair_panel(
        contract, members, method, lambda c, r: tokens[c], tmp_path
    )
    assert not result["initial_release"] and result["final_release"]
    assert result["object_transfers"] == (2 if method == "whole_panel_refresh" else 1)
    assert result["round_trips"] == 1
    assert result["new_pairs"] == 0
    assert len(repaired) == 2


def test_duplicate_only_repairs_without_transfer(tmp_path):
    contract, tokens = fixtures()
    members = [Member(c, 0, t) for c, t in tokens.items()]
    result, _ = repair_panel(
        contract,
        members + [members[0]],
        "slot_directed_batch",
        lambda c, r: tokens[c],
        tmp_path,
    )
    assert (
        result["final_release"]
        and result["object_transfers"] == result["round_trips"] == 0
    )


def test_repair_plan_is_invariant_to_outcomes():
    contract, tokens = fixtures()
    members = [Member("a", 0, tokens["a"])]
    changed = [replace(m, evidence=replace(m.evidence, value=-99999)) for m in members]
    assert revision_plan(contract, members) == revision_plan(contract, changed)


def test_semantic_oracle_uses_final_residual_not_operation_count():
    contract, tokens = fixtures()
    recipes = {
        c: {"spec": {"branch": c, "replicate": 0}, "liquidity": 6} for c in tokens
    }
    catalog = {
        c: {"id": c, "recipe": recipes[c], "plan": {"workers": 2}} for c in tokens
    }
    expected = {(c, 0): r for c, r in recipes.items()}
    members = [{"object_id": c} for c in tokens]
    assert semantic_truth(expected, [{"workers": 2}], members, catalog, tokens) == []
    issues = semantic_truth(
        expected, [{"workers": 2}], [members[0], members[0]], catalog, tokens
    )
    assert issues == [["a", 0, "duplicate_slot"], ["b", 0, "missing_slot"]]
    score = localization_score(issues, issues)
    assert score["precision"] == [2, 2] and score["recall"] == [2, 2]
    assert localization_score([], [])["precision"] is None


def test_exact_allocations_without_observing_outcomes():
    from policytwin.cscwd_mechanism.design import design
    from scripts.cscwd2027_design_contract.design import sessions

    contexts, attempts = design(sessions())
    assert len(contexts) == 52 and len(attempts) == 1632
    assert sum("primary" in c["families"] for c in contexts) == 32
    assert {c["replicates"] for c in contexts} == {4, 8, 16, 32}
    assert len({r["id"] for r in attempts}) == 1632


def test_no_coupling_projection_keeps_clean_but_releases_wrong_c():
    from policytwin.cscwd_mechanism.analysis import variant_decide
    from scripts.cscwd2027_design_contract.evaluate import panel_valid

    contract, tokens = fixtures()
    panel = [Member(c, 0, t) for c, t in tokens.items()]
    projections = {"key-a": "noC-a", "key-b": "noC-b", "other-C-a": "noC-a"}
    clean = variant_decide(contract, panel, "without_coupling", projections)
    assert clean["authorized"] and panel_valid(contract, clean["numerical"])
    panel[0] = replace(
        panel[0], evidence=replace(tokens["a"], scientific_key="other-C-a")
    )
    wrong = variant_decide(contract, panel, "without_coupling", projections)
    assert wrong["authorized"] and not panel_valid(contract, wrong["numerical"])
