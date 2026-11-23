"""Minimal verifiable metric, dependency and full file-path checks before sealing."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from policytwin.coordination.schema import commitment
from policytwin.cscwd_mechanism.coupling import CoupledRecipe
from policytwin.typed_revision.contract import (
    EvaluationToken,
    TraceToken,
    decide,
    derive,
    relational_work_order,
    trajectory_key,
    verify_derivation,
    work_order,
)
from policytwin.typed_revision.design import (
    STUDY,
    available_slots,
    recipes,
    request_contract,
    schedule,
    sessions,
)
from policytwin.typed_revision.metrics import MetricSpec, evaluate, evaluation_key
from policytwin.typed_revision.runtime import (
    acquired_trace,
    checked_evaluation,
    run_new_pair,
)


def fixture_capsule():
    worlds = {}
    for world, amounts in (
        ("baseline", [(10, 2), (20, 3)]),
        ("policy", [(10, 4), (20, 2)]),
    ):
        worlds[world] = {
            "completed": True,
            "epochs": [
                {
                    "epoch": epoch,
                    "applied_commands": [{"event_key": {"value": f"{world}-{epoch}"}}],
                }
                for epoch in range(4)
            ],
            "trades": [
                {
                    "cause_event_key": f"{world}-{epoch}",
                    "price": price,
                    "quantity": quantity,
                }
                for epoch, (price, quantity) in zip((0, 2), amounts, strict=True)
            ],
            "traded_notional": sum(p * q for p, q in amounts),
        }
    value = {
        "spec": {"epoch_count": 4, "branch": "cancellation", "replicate": 0},
        "plan": {"x": 1},
        "worlds": worlds,
    }
    value["capsule_sha256"] = commitment(value)
    return value


def token(value=7, metric="old"):
    return EvaluationToken(
        "e", "cancellation", 0, "t", metric, "c", "p", "r", True, value
    )


def trace():
    return TraceToken("t0", "cancellation", 0, "t", "c", "p", True)


def test_manually_checkable_metric_fixture():
    capsule = fixture_capsule()
    assert evaluate(capsule, MetricSpec("notional")) == {
        "baseline": 80,
        "policy": 80,
        "contrast": 0,
    }
    assert evaluate(capsule, MetricSpec("volume")) == {
        "baseline": 5,
        "policy": 6,
        "contrast": 1,
    }
    assert evaluate(capsule, MetricSpec("volume", 2)) == {
        "baseline": 3,
        "policy": 2,
        "contrast": -1,
    }


@pytest.mark.parametrize(
    "mutation", ["incomplete", "unknown_cause", "noninteger", "outside"]
)
def test_metric_rejects_inapplicable_or_broken_projection(mutation):
    capsule = fixture_capsule()
    metric = MetricSpec("volume")
    if mutation == "incomplete":
        capsule["worlds"]["baseline"]["completed"] = False
    elif mutation == "unknown_cause":
        capsule["worlds"]["policy"]["trades"][0]["cause_event_key"] = "unknown"
    elif mutation == "noninteger":
        capsule["worlds"]["policy"]["trades"][0]["quantity"] = 1.5
    else:
        metric = MetricSpec("volume", 4)
    with pytest.raises(ValueError):
        evaluate(capsule, metric)


def test_derivation_binds_metric_source_parent_and_values():
    capsule = fixture_capsule()
    t = replace(
        trace(),
        capsule_sha256=capsule["capsule_sha256"],
        plan_sha256=commitment(capsule["plan"]),
    )
    e, receipt = derive(t, capsule, MetricSpec("volume"))
    assert e.value == 1
    assert verify_derivation(t, capsule, receipt) == e
    for field, new in (
        ("metric_source_sha256", "other"),
        ("capsule_sha256", "other"),
        ("evaluation_key", "other"),
    ):
        bad = {**receipt, field: new}
        with pytest.raises(ValueError):
            verify_derivation(t, capsule, bad)
    bad = deepcopy(receipt)
    bad["values"]["contrast"] = 12
    with pytest.raises(ValueError):
        verify_derivation(t, capsule, bad)


def test_mutated_parent_is_not_reevaluated_as_verified():
    capsule = fixture_capsule()
    t = replace(
        trace(),
        capsule_sha256=capsule["capsule_sha256"],
        plan_sha256=commitment(capsule["plan"]),
    )
    capsule["worlds"]["baseline"]["trades"][0]["quantity"] += 1
    with pytest.raises(ValueError, match="content"):
        derive(t, capsule, MetricSpec("volume"))


def test_metric_and_trajectory_dependencies_are_separate():
    recipe = recipes(sessions()[0], "initial")["cancellation", 0]
    key = trajectory_key(recipe)
    assert evaluation_key(key, MetricSpec("notional")) != evaluation_key(
        key, MetricSpec("volume")
    )
    assert evaluation_key(key, MetricSpec("volume")) != evaluation_key(
        key, MetricSpec("volume", 2)
    )
    assert trajectory_key(CoupledRecipe(recipe.recipe, "world_separated")) != key
    changed = replace(
        recipe.recipe, spec=replace(recipe.recipe.spec, seed="changed-seed")
    )
    assert trajectory_key(CoupledRecipe(changed, "shared")) != key
    opening = replace(
        recipe.recipe, liquidity_per_level=recipe.recipe.liquidity_per_level + 1
    )
    assert trajectory_key(CoupledRecipe(opening, "shared")) != key


def test_three_way_partition_and_opaque_control():
    expected = {("cancellation", 0): ("t", "new")}
    assert work_order(expected, ("p",), [trace()], [token()])[0]["action"] == "evaluate"
    assert (
        work_order(expected, ("p",), [trace()], [token(metric="new")])[0]["action"]
        == "reuse"
    )
    assert work_order(expected, ("p",), [], [])[0]["action"] == "execute"
    assert (
        work_order(expected, ("p",), [trace()], [token()], mode="opaque")[0]["action"]
        == "execute"
    )
    assert (
        work_order(expected, ("forbidden",), [trace()], [token(metric="new")])[0][
            "action"
        ]
        == "execute"
    )


def test_planner_is_outcome_independent_and_relationally_equal():
    expected = {("cancellation", 0): ("t", "new")}
    wanted = work_order(expected, ("p",), [trace()], [token(1, "new")])
    assert wanted == work_order(expected, ("p",), [trace()], [token(-999999, "new")])
    assert wanted == relational_work_order(
        expected, ("p",), [trace()], [token(12, "new")]
    )


def test_exact_coverage_metric_binding_and_lexicographic_tie():
    expected = {("cancellation", 0): ("t", "new")}
    assert not decide("d", expected, ("p",), [token()])["authorized"]
    assert decide("d", expected, ("p",), [token()], omit_evaluation_binding=True)[
        "authorized"
    ]
    assert not decide("d", expected, ("p",), [token(metric="new")] * 2)["authorized"]
    assert not decide(
        "d", expected, ("p",), [replace(token(metric="new"), local_pass=False)]
    )["authorized"]
    two = {**expected, ("agent_exit", 0): ("u", "g")}
    es = [
        token(0, "new"),
        replace(token(0, "g"), candidate="agent_exit", trajectory_key="u"),
    ]
    assert decide("d", two, ("p",), es)["report"]["top1"] == "agent_exit"


def test_schedule_binding_partial_views_and_mixed_invalidation():
    blocks = schedule()
    assert len(blocks) == 8
    assert sum(len(b["requests"]) * len(b["availability"]) for b in blocks) == 72
    for session in sessions():
        before = request_contract(session, "metric_quantity")
        after = request_contract(session, "combined_revision")
        changed = [
            a["candidate"]
            for a, b in zip(before["slots"], after["slots"], strict=True)
            if a["trajectory_key"] != b["trajectory_key"]
        ]
        assert changed == [session["target"]] * STUDY["replicates"]
        assert set(map(tuple, available_slots(session, 6))) <= set(
            map(tuple, available_slots(session, 12))
        )


def test_actual_spawn_acquisition_replay_and_metric_derivation(tmp_path):
    """Unique changed file/derivation path; this engineering fixture is not science."""
    recipe = recipes(sessions()[0], "initial")["agent_exit", 0]
    t, capsule, result = run_new_pair(
        "engineering-pair", recipe, tmp_path, "engineering-only"
    )
    assert result["local_pass"]
    assert t.local_pass
    first = checked_evaluation(
        t, capsule, MetricSpec("volume"), tmp_path / "evaluation.json"
    )
    t2, capsule2 = acquired_trace(
        "engineering-pair", tmp_path, tmp_path / "fresh-replay"
    )
    assert t == t2
    assert derive(t2, capsule2, MetricSpec("volume"))[0] == first
