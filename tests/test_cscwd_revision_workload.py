"""Engineering fixtures only; these synthetic tokens are not study evidence."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from policytwin.coordination.schema import commitment
from policytwin.provenance import canonical_json_bytes
from policytwin.revision_workload.design import (
    DOMAINS,
    INITIAL,
    METHODS,
    STUDY,
    Target,
    policies,
    request,
    schedule,
    slots,
    targets,
)
from policytwin.revision_workload.planner import (
    DependencyCache,
    SemanticMemo,
    build_work,
    catalog_identity,
    plan,
    ranked,
    validate_request,
)
from policytwin.revision_workload.policy import (
    authorized,
    compile_policies,
    compile_policy,
    expression,
)
from policytwin.typed_revision.contract import (
    EvaluationToken,
    TraceToken,
    trajectory_key,
)
from policytwin.typed_revision.metrics import MetricSpec, evaluation_key


@pytest.fixture(scope="module")
def fixture() -> dict[str, Any]:
    session = {**schedule()[0]["session"], "id": "engineering-only", "agents": 4}
    cache = DependencyCache()
    layouts = cache.build(session, 32)
    initial = next(o for o in layouts if o["target"] == INITIAL.to_mapping())
    changed = next(
        o
        for o in layouts
        if o["target"] == Target(2, "shared", "volume", 4, 2, 12).to_mapping()
    )
    traces, evaluations = {}, {}
    for label, layout in (("initial", initial), ("changed", changed)):
        for slot, (trace_key, _) in layout["expected"].items():
            identifier = f"fixture-{label}-{slot[0]}-{slot[1]}"
            trace = TraceToken(
                identifier,
                *slot,
                trace_key,
                commitment(identifier),
                layout["plan"],
                True,
            )
            traces[label, slot] = trace
            for quantity, first in (("notional", 0), ("volume", 0), ("volume", 4)):
                key = evaluation_key(trace_key, MetricSpec(quantity, first))
                evaluations[label, slot, quantity, first] = EvaluationToken(
                    f"eval-{identifier}-{quantity}-{first}",
                    *slot,
                    trace_key,
                    key,
                    trace.capsule_sha256,
                    trace.plan_sha256,
                    commitment([identifier, key]),
                    True,
                    0,
                )
    return {
        "session": session,
        "cache": cache,
        "layouts": layouts,
        "traces": traces,
        "evaluations": evaluations,
    }


def view(
    fixture: dict[str, Any], episode: str, stage: int
) -> tuple[list[TraceToken], list[EvaluationToken]]:
    session = fixture["session"]
    count = 3 if episode == "C" and stage >= 2 else 2
    traces, evaluations = [], []
    for slot in slots(session, count):
        traces.append(fixture["traces"]["initial", slot])
        evaluations.append(fixture["evaluations"]["initial", slot, "notional", 0])
        if episode == "A" and stage >= 1:
            evaluations.append(fixture["evaluations"]["initial", slot, "volume", 0])
        if episode == "B" and stage >= 2:
            evaluations.append(fixture["evaluations"]["initial", slot, "volume", 4])
    if episode == "A" and stage >= 2:
        for slot in slots(session, 2):
            if slot[0] == session["target"]:
                traces.append(fixture["traces"]["changed", slot])
                if stage >= 3:
                    evaluations.append(
                        fixture["evaluations"]["changed", slot, "volume", 4]
                    )
    return traces, evaluations


def invoke(
    fixture: dict[str, Any],
    episode: str,
    stage: int,
    method: str,
    memo: SemanticMemo | None = None,
) -> dict[str, Any]:
    traces, evaluations = view(fixture, episode, stage)
    session = fixture["session"]
    return plan(
        session,
        32,
        request(session, episode, stage),
        policies(episode, stage),
        traces,
        evaluations,
        method,
        fixture["cache"],
        memo or SemanticMemo(),
    )


def test_real_nested_domains() -> None:
    domains = [{t.identity for t in targets(size)} for size in (32, 256, 1024)]
    assert [len(d) for d in domains] == [32, 256, 1024]
    assert domains[0] < domains[1] < domains[2]
    assert all(0 <= t.first_epoch < 8 and 1 <= t.epoch < 8 for t in targets(1024))
    assert 0 in DOMAINS[1024]["allowance"]


def test_complete_frozen_metadata_uses_canonical_serialization() -> None:
    import json

    assert json.loads(canonical_json_bytes(STUDY))["domains"] == {
        str(size): domain for size, domain in DOMAINS.items()
    }
    assert json.loads(canonical_json_bytes(schedule())) == schedule()


def test_schedule_requests_and_controls_are_predeclared() -> None:
    blocks = schedule()
    assert len(blocks) == 4
    assert sum(len(e["stages"]) for b in blocks for e in b["episodes"]) * 3 == 156
    assert (
        sum(s["physical"] for b in blocks for e in b["episodes"] for s in e["stages"])
        == 16
    )
    for block in blocks:
        for episode in block["episodes"]:
            requests = [s["request"]["request_sha256"] for s in episode["stages"]]
            if episode["episode"] == "A":
                assert len(set(requests)) == 1
            elif episode["episode"] in {"B", "C"}:
                assert len(set(requests[:3])) == 1 and requests[3] != requests[2]


def test_linear_cross_role_complete_dependencies() -> None:
    compiled = compile_policies(policies("A", 0))
    assert compiled[1].reads == ("coupling", "epoch", "first_epoch", "quantity")
    assert compiled[2].reads == ("new_pairs", "reevaluations", "workers")
    reads, function = expression(
        {
            "op": "le",
            "left": {"add": [{"field": "new_pairs"}, {"field": "reevaluations"}]},
            "right": {"mul": [6, {"field": "workers"}]},
        }
    )
    assert reads == {"new_pairs", "reevaluations", "workers"}
    assert function({"new_pairs": 6, "reevaluations": 6, "workers": 2})
    assert not function({"new_pairs": 6, "reevaluations": 6, "workers": 1})


def test_expression_identity_changes_even_with_same_version() -> None:
    original = policies("C", 0)[2]
    changed = deepcopy(original)
    changed["expression"]["args"][0]["right"] = 3
    changed["policy_sha256"] = commitment(
        {k: v for k, v in changed.items() if k != "policy_sha256"}
    )
    assert original["version"] == changed["version"]
    assert compile_policy(original).identity != compile_policy(changed).identity
    assert (
        compile_policy(original, "a" * 64).identity
        != compile_policy(original, "b" * 64).identity
    )


def test_five_routes_full_work_and_selection_parity(fixture: dict[str, Any]) -> None:
    for episode in ("A", "B", "C", "control"):
        memos = {m: SemanticMemo() for m in METHODS}
        for stage in range(1 if episode == "control" else 4):
            found = [
                invoke(fixture, episode, stage, method, memos[method])
                for method in METHODS
            ]
            assert all(r["selected"] == found[0]["selected"] for r in found)
            assert all(
                r["counts"]["selection_predicate_calls"] == (6 if r["selected"] else 0)
                for r in found
            )
    traces, evaluations = view(fixture, "A", 3)
    assert build_work(fixture["layouts"], traces, evaluations, False) == build_work(
        fixture["layouts"], traces, evaluations, True
    )


def test_irrelevant_arrival_reuses_negatives_and_work_change_reconsiders(
    fixture: dict[str, Any],
) -> None:
    memo = SemanticMemo()
    first = invoke(fixture, "A", 0, "persistent", memo)
    irrelevant = invoke(fixture, "A", 1, "persistent", memo)
    relevant = invoke(fixture, "A", 2, "persistent", memo)
    assert first["selected"] is None and irrelevant["selected"] is None
    assert first["catalog_sha256"] != irrelevant["catalog_sha256"]
    assert irrelevant["counts"]["negative_hits"] > 0
    assert (
        irrelevant["counts"]["search_predicate_calls"]
        < first["counts"]["search_predicate_calls"]
    )
    before_traces, before_evals = view(fixture, "A", 0)
    after_traces, after_evals = view(fixture, "A", 1)
    wanted = [
        o
        for o in fixture["layouts"]
        if o["target"] == Target(2, "shared", "volume", 4, 2, 12).to_mapping()
    ]
    assert build_work(wanted, before_traces, before_evals, False) == build_work(
        wanted, after_traces, after_evals, False
    )
    assert relevant["selected"]["new_pairs"] == 6
    assert relevant["selected"]["reevaluations"] == 6


def test_policy_change_reconsiders_and_store_arrival_restores_agreement(
    fixture: dict[str, Any],
) -> None:
    memo = SemanticMemo()
    outcomes = [invoke(fixture, "C", stage, "persistent", memo) for stage in range(4)]
    assert [r["selected"] is not None for r in outcomes] == [True, False, True, True]
    assert outcomes[2]["selected"]["new_pairs"] == 3
    assert (
        outcomes[2]["selected"] != outcomes[3]["selected"]
    )  # Same target, revised waived worker preference.


def test_metric_revision_preserves_trajectory_and_changes_work(
    fixture: dict[str, Any],
) -> None:
    memo = SemanticMemo()
    outcomes = [invoke(fixture, "B", stage, "persistent", memo) for stage in range(4)]
    assert [r["selected"] is not None for r in outcomes] == [True, False, True, True]
    old, new = outcomes[0]["selected"], outcomes[2]["selected"]
    assert [s["trajectory_key"] for s in old["contract"]["slots"]] == [
        s["trajectory_key"] for s in new["contract"]["slots"]
    ]
    assert [s["evaluation_key"] for s in old["contract"]["slots"]] != [
        s["evaluation_key"] for s in new["contract"]["slots"]
    ]


def test_current_full_offer_binding_and_stale_approval(fixture: dict[str, Any]) -> None:
    result = invoke(fixture, "A", 2, "persistent")
    new = invoke(fixture, "A", 3, "persistent")
    session = fixture["session"]
    compiled = compile_policies(policies("A", 2))
    req = request(session, "A", 2)
    assert authorized(
        req, result["catalog_sha256"], result["selected"], compiled, result["approvals"]
    )
    assert not authorized(
        req, new["catalog_sha256"], new["selected"], compiled, result["approvals"]
    )
    for mutation in ("waived", "target", "work", "count"):
        selected = deepcopy(result["selected"])
        if mutation == "waived":
            selected["waived"].append("invented")
        elif mutation == "target":
            selected["target"]["first_epoch"] = 0
        elif mutation == "work":
            selected["work_order"][0]["object_id"] = "other-object"
        else:
            selected["new_pairs"] += 1
        assert not authorized(
            req, result["catalog_sha256"], selected, compiled, result["approvals"]
        )


@pytest.mark.parametrize("value", [True, 1.0, "1", -1, 8])
def test_hard_requirement_strict_types_and_domain(
    fixture: dict[str, Any], value: Any
) -> None:
    req = request(fixture["session"], "control", 0)
    req["hard"] = [{"field": "epoch", "value": value}]
    req["request_sha256"] = commitment(
        {k: v for k, v in req.items() if k != "request_sha256"}
    )
    with pytest.raises(ValueError):
        validate_request(req, fixture["session"])


def test_duplicate_hard_requirements_rejected(fixture: dict[str, Any]) -> None:
    req = request(fixture["session"], "control", 0)
    req["hard"] = [{"field": "epoch", "value": 1}, {"field": "epoch", "value": 2}]
    req["request_sha256"] = commitment(
        {k: v for k, v in req.items() if k != "request_sha256"}
    )
    with pytest.raises(ValueError):
        validate_request(req, fixture["session"])


def test_request_revision_recomputes_rank_and_hard_feasibility(
    fixture: dict[str, Any],
) -> None:
    traces, evaluations = view(fixture, "control", 0)
    options = build_work(fixture["layouts"], traces, evaluations, False)
    req = request(fixture["session"], "control", 0)
    original = ranked(options, req)[0]
    req["proposed"]["epoch"] = 2
    req["hard"] = [{"field": "workers", "value": 1}]
    req["request_sha256"] = commitment(
        {k: v for k, v in req.items() if k != "request_sha256"}
    )
    revised = ranked(options, req)[0]
    assert revised["target"]["workers"] == 1
    assert revised["target_id"] != original["target_id"]
    found = [
        plan(
            fixture["session"],
            32,
            req,
            policies("control", 0),
            traces,
            evaluations,
            m,
            fixture["cache"],
            SemanticMemo(),
        )["selected"]
        for m in METHODS
    ]
    assert all(r == found[0] for r in found)


def test_dependency_cache_cross_session_rejected(fixture: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        fixture["cache"].build({**fixture["session"], "id": "other-session"}, 32)


def test_real_dependency_compiler_matches_inherited_keys(
    fixture: dict[str, Any],
) -> None:
    from policytwin.revision_workload.design import recipes

    target = INITIAL
    for recipe in recipes(fixture["session"], target).values():
        assert fixture["cache"].trajectories[recipe] == trajectory_key(recipe)


def test_lean_baseline_has_no_projection_and_generic_positive_is_not_authority(
    fixture: dict[str, Any],
) -> None:
    baseline = invoke(fixture, "control", 0, "exhaustive")
    assert baseline["counts"]["projection_calls"] == 0
    assert baseline["counts"]["new_witnesses"] == []
    memo = SemanticMemo()
    first = invoke(fixture, "control", 0, "generic_memo", memo)
    second = invoke(fixture, "control", 0, "generic_memo", memo)
    assert first["selected"] == second["selected"]
    assert second["counts"]["positive_hits"] > 0
    assert second["counts"]["selection_predicate_calls"] == 6


def test_catalog_irrelevant_addition_invalidates_approval(
    fixture: dict[str, Any],
) -> None:
    result = invoke(fixture, "control", 0, "persistent")
    traces, evaluations = view(fixture, "control", 0)
    slot = traces[0].slot
    evaluations.append(fixture["evaluations"]["initial", slot, "volume", 0])
    changed_catalog = catalog_identity(traces, evaluations)
    assert not authorized(
        request(fixture["session"], "control", 0),
        changed_catalog,
        result["selected"],
        compile_policies(policies("control", 0)),
        result["approvals"],
    )
