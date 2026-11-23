"""Hand-checkable owner/work relations and the one new physical path."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace

import pytest

from policytwin.coordination.schema import commitment
from policytwin.negotiated_repair.consent import (
    approvals,
    authorized,
    proposal,
    validate_request,
)
from policytwin.negotiated_repair.design import (
    Target,
    available,
    contract,
    recipes,
    schedule,
    sessions,
    targets,
    ticket,
)
from policytwin.negotiated_repair.runtime import new_pair, repair
from policytwin.negotiated_repair.solver import (
    alternatives,
    conflict_cores,
    feasible,
    frontier,
    minimal_options,
    selected,
    sql_order,
)
from policytwin.typed_revision.contract import EvaluationToken, TraceToken, derive
from policytwin.typed_revision.metrics import MetricSpec
from policytwin.typed_revision.runtime import expectations


def catalog(session, count=6):
    declared = contract(session, Target(1, "shared", "notional", 2, 12))
    expected = expectations(declared)
    traces, evaluations = [], []
    for branch, rep in available(session, count):
        t, e = expected[branch, rep]
        traces.append(
            TraceToken(
                f"t-{branch}-{rep}",
                branch,
                rep,
                t,
                "c",
                declared["allowed_plans"][0],
                True,
            )
        )
        evaluations.append(
            EvaluationToken(
                f"e-{branch}-{rep}",
                branch,
                rep,
                t,
                e,
                "c",
                declared["allowed_plans"][0],
                "r",
                True,
                rep,
            )
        )
    return traces, evaluations


def test_declared_domain_and_balanced_catalog():
    assert len(targets()) == 32
    assert len(schedule()) == 4
    assert sum(len(b["tickets"]) for b in schedule()) == 24
    for session in sessions():
        partial = available(session, 6)
        assert len(partial) == len(set(partial)) == 6
        assert sorted(
            sum(b == branch for b, _ in partial) for branch in {b for b, _ in partial}
        ) == [2, 2, 2]
        assert set(partial) <= set(available(session, 12))
    with pytest.raises(ValueError):
        Target(3, "shared", "notional", 2, 6)


@pytest.mark.parametrize(
    "kind,serial_pairs",
    [("candidate_conflict", 8), ("coupling_conflict", 12), ("plan_conflict", 12)],
)
def test_minimum_concession_frontier_and_full_csp(kind, serial_pairs):
    session = sessions()[0]
    traces, evals = catalog(session)
    request = ticket(session, kind)
    options = alternatives(request, traces, evals)
    chosen = selected(options, request, "frontier")
    serial = selected(options, request, "serial")
    assert chosen == selected(options, request, "csp_dag")
    assert selected(options, request, "fixed_dag") is None
    assert chosen is not None and serial is not None
    assert chosen["new_pairs"] == 6
    assert serial["new_pairs"] == serial_pairs
    assert len(chosen["waived"]) == len(serial["waived"]) == 1
    assert len(frontier(minimal_options(options, request["clauses"]))) == 2
    cores = conflict_cores(options, request["clauses"])
    assert len(cores) == 1 and len(cores[0]) == 2
    core_clauses = [c for c in request["clauses"] if c["id"] in cores[0]]
    assert not any(feasible(o, core_clauses) for o in options)
    for removed in core_clauses:
        assert any(
            feasible(o, [c for c in core_clauses if c != removed]) for o in options
        )


def test_hard_conflict_abstains_and_controls_match():
    session = sessions()[0]
    for kind, count, pairs in (
        ("nonnegotiable", 6, None),
        ("partial_control", 6, 6),
        ("complete_control", 12, 0),
    ):
        traces, evals = catalog(session, count)
        request = ticket(session, kind)
        options = alternatives(request, traces, evals)
        for method in ("frontier", "csp_dag", "serial", "fixed_dag"):
            chosen = selected(options, request, method)
            if pairs is None:
                assert chosen is None
            else:
                assert (
                    chosen is not None
                    and chosen["new_pairs"] == pairs
                    and not chosen["waived"]
                )


def test_owner_approval_binds_request_target_waiver_and_work():
    session = sessions()[0]
    traces, evals = catalog(session)
    request = ticket(session, "coupling_conflict")
    chosen = selected(alternatives(request, traces, evals), request, "frontier")
    assert chosen is not None
    records = approvals(request, chosen)
    assert authorized(request, chosen, records)
    assert not authorized(request, chosen, records[:-1])
    for field, value in (
        ("contract_sha256", "0" * 64),
        ("owned_waivers", []),
        ("request_sha256", "old"),
    ):
        bad = deepcopy(records)
        owner = next(i for i, r in enumerate(bad) if r["owned_waivers"])
        bad[owner][field] = value
        bad[owner]["approval_sha256"] = commitment(
            {k: v for k, v in bad[owner].items() if k != "approval_sha256"}
        )
        assert not authorized(request, chosen, bad)
    changed = deepcopy(chosen)
    changed["work_order"][0]["action"] = "execute"
    assert proposal(request, changed) != proposal(request, chosen)
    assert not authorized(request, changed, records)


def test_hard_clause_and_request_owner_cannot_be_silently_waived():
    request = ticket(sessions()[0], "partial_control")
    bad = deepcopy(request)
    bad["clauses"][0]["owner"] = "producer"
    bad["request_sha256"] = commitment(
        {k: v for k, v in bad.items() if k != "request_sha256"}
    )
    with pytest.raises(ValueError, match="owner"):
        validate_request(bad)


def test_planning_does_not_read_numerical_values():
    session = sessions()[0]
    traces, evals = catalog(session)
    request = ticket(session, "candidate_conflict")
    before = alternatives(request, traces, evals)
    after = alternatives(request, traces, [replace(e, value=-1000000) for e in evals])
    assert before == after


def test_sql_dag_retains_full_permissions_and_dependencies():
    session = sessions()[0]
    traces, evals = catalog(session, 12)
    wanted = contract(session, Target(1, "shared", "notional", 2, 12))
    order = sql_order(expectations(wanted), wanted["allowed_plans"][0], traces, evals)
    assert all(o["action"] == "reuse" for o in order)
    assert all(
        o["action"] == "execute"
        for o in sql_order(expectations(wanted), "forbidden", traces, evals)
    )
    late = contract(session, Target(1, "shared", "late_volume", 2, 12))
    assert all(
        o["action"] == "evaluate"
        for o in sql_order(
            expectations(late), wanted["allowed_plans"][0], traces, evals
        )
    )


def test_new_physical_plan_and_pure_derivation(tmp_path):
    """One new adapter path; this engineering-only seed is outside the schedule."""
    session = {**sessions()[0], "id": "engineering-negotiation-only"}
    target = Target(2, "world_separated", "late_volume", 1, 12)
    recipe = recipes(session, target)["agent_exit", 0]
    trace, capsule = new_pair(
        "engineering-negotiated-pair", recipe, 1, tmp_path, "engineering-only"
    )
    assert trace.local_pass
    token, receipt = derive(trace, capsule, MetricSpec("volume", 2))
    assert token.value == receipt["values"]["contrast"]
    request = ticket(session, "nonnegotiable")
    request["proposed"].update(target.to_mapping())
    for clause in request["clauses"]:
        clause["value"] = request["proposed"][clause["field"]]
    request["request_sha256"] = commitment(
        {k: v for k, v in request.items() if k != "request_sha256"}
    )
    declared = contract(session, target)
    declared["slots"] = [
        s for s in declared["slots"] if (s["candidate"], s["replicate"]) == trace.slot
    ]
    declared["contract_sha256"] = commitment(
        {k: v for k, v in declared.items() if k != "contract_sha256"}
    )
    one_slot_order = [
        {
            "candidate": trace.candidate,
            "replicate": trace.replicate,
            "trajectory_key": trace.trajectory_key,
            "evaluation_key": token.evaluation_key,
            "action": "evaluate",
            "object_id": trace.object_id,
        }
    ]
    option = {
        "target": target.to_mapping(),
        "target_id": target.identity,
        "contract": declared,
        "work_order": one_slot_order,
        "new_pairs": 0,
        "reevaluations": 1,
        "reused_evaluations": 0,
        "waived": [],
    }
    reference = {
        "traces": [asdict(trace)],
        "evaluations": {"late_volume": [asdict(token)]},
    }
    result = repair(
        request,
        "frontier",
        option,
        {trace.slot: (trace, token)},
        tmp_path,
        "engineering-only",
        reference,
    )
    assert result["authorized"] and result["exact"]
