"""Manually checkable engineering fixtures; excluded from scientific results."""

from __future__ import annotations

from copy import deepcopy
from itertools import product

import pytest

from policytwin.consent_projection.policy import (
    authorized,
    evaluate,
    signature,
    validate_ast,
    validate_policies,
    votes,
)
from policytwin.consent_projection.search import ALL_METHODS, search
from policytwin.consent_projection.study import schedule, selected_slots, sessions
from policytwin.coordination.schema import commitment
from policytwin.negotiated_repair.design import OWNERS, ROLES, Target, contract


def policy(owner, expression):
    body = {
        "schema": "closed-owner-policy.v1",
        "owner": owner,
        "expression": expression,
    }
    return {**body, "policy_sha256": commitment(body)}


def fixture(work_field="new_pairs"):
    session = sessions()[0]
    proposed = Target(2, "shared", "late_volume", 2, 12).to_mapping()
    body = {
        "schema": "role-owned-revision-request.v1",
        "session": session,
        "kind": "fixture",
        "proposed": proposed,
        "availability": 0,
        "clauses": [
            {
                "id": OWNERS[k] + "." + k,
                "owner": OWNERS[k],
                "field": k,
                "value": v,
                "hard": False,
            }
            for k, v in proposed.items()
        ],
    }
    request = {**body, "request_sha256": commitment(body)}
    changed = {
        **proposed,
        ("epoch" if work_field == "new_pairs" else "metric"): (
            1 if work_field == "new_pairs" else "notional"
        ),
    }
    options = []
    for index, target in enumerate((proposed, changed)):
        options.append(
            {
                "target": target,
                "target_id": commitment(target),
                "new_pairs": 0,
                "reevaluations": 0,
                "work_order": [],
                "contract": contract(session, Target.from_mapping(target)),
            }
        )
        options[-1][work_field] = 12 if index == 0 else 0
    policies = [
        policy(
            r,
            {"op": "le", "field": work_field, "value": 6}
            if r == "producer"
            else {"op": "const", "value": True},
        )
        for r in ROLES
    ]
    return request, options, policies


@pytest.mark.parametrize("field", ["new_pairs", "reevaluations"])
def test_complete_work_projection_preserves_acceptable_alternative(field):
    request, options, policies = fixture(field)
    outcomes = {
        m: search(options, request, policies, "catalog", m) for m in ALL_METHODS
    }
    expected = outcomes["exhaustive"]["selected"]
    assert expected is not None and expected[field] == 0
    assert all(
        outcomes[m]["selected"] == expected
        for m in ("certificates", "policy_sql", "full_key_memo")
    )
    assert outcomes["omit_work"]["selected"] is None
    assert outcomes["work_pruned"]["selected"] is None
    assert outcomes["full_key_memo"]["skipped_candidates"] == 0


def test_ast_dependencies_cover_all_branches_not_only_short_circuit():
    node = {
        "op": "or",
        "args": [
            {"op": "eq", "field": "workers", "value": 1},
            {"op": "not", "arg": {"op": "le", "field": "new_pairs", "value": 6}},
        ],
    }
    assert validate_ast(node, "producer") == {"workers", "new_pairs"}
    assert evaluate(node, {"target": {"workers": 1}, "new_pairs": 12})


@pytest.mark.parametrize(
    "node",
    [
        {"op": "eq", "field": "epoch", "value": 1},
        {"op": "le", "field": "workers", "value": True},
        {"op": "eq", "field": "workers", "value": 1.0},
        {"op": "call", "function": "outcome"},
        {"op": "const", "value": True, "undeclared": 4},
    ],
)
def test_policy_language_rejects_foreign_or_open_fields(node):
    with pytest.raises(ValueError):
        validate_ast(node, "producer")


def test_signature_binds_context_and_only_declared_dependencies():
    request, options, policies = fixture()
    p = next(p for p in policies if p["owner"] == "producer")
    first = signature(p, options[0], request["request_sha256"], "catalog")
    assert first != signature(p, options[1], request["request_sha256"], "catalog")
    assert first != signature(p, options[0], request["request_sha256"], "changed")
    other = deepcopy(options[0])
    other["target"]["metric"] = "notional"
    assert first == signature(p, other, request["request_sha256"], "catalog")


def test_exact_votes_not_reused_between_targets_or_contexts():
    request, options, policies = fixture()
    chosen = search(options, request, policies, "catalog", "certificates")["selected"]
    records = votes(request, chosen, policies, "catalog")
    assert authorized(request, chosen, policies, "catalog", records)
    assert not authorized(request, chosen, policies, "other", records)
    assert not authorized(request, chosen, policies, "catalog", records[:-1])
    assert not authorized(request, chosen, policies, "catalog", records + records[:1])
    changed = deepcopy(policies)
    changed[0]["expression"] = {"op": "const", "value": False}
    changed[0]["policy_sha256"] = commitment(
        {k: v for k, v in changed[0].items() if k != "policy_sha256"}
    )
    assert not authorized(request, chosen, changed, "catalog", records)
    with pytest.raises(ValueError):
        validate_policies(policies[:-1])


def test_independent_sql_truth_tables_and_rank_match():
    request, options, _ = fixture()
    atoms = [{"op": "const", "value": value} for value in (False, True)]
    atoms += [{"op": "eq", "field": "epoch", "value": v} for v in (1, 2)]
    for left, right in product(atoms, repeat=2):
        for op in ("and", "or"):
            policies = [
                policy("designer", {"op": op, "args": [left, right]}),
                policy("evaluator", {"op": "const", "value": True}),
                policy("producer", {"op": "const", "value": True}),
            ]
            base = search(options, request, policies, "catalog", "exhaustive")
            for method in ("certificates", "policy_sql", "full_key_memo"):
                assert (
                    search(
                        list(reversed(options)),
                        request,
                        list(reversed(policies)),
                        "catalog",
                        method,
                    )["selected"]
                    == base["selected"]
                )


def test_schedule_is_preoutcome_and_physical_indices_not_feasibility_selected():
    first = schedule()
    assert first == schedule()
    assert len(first) == 4
    assert {b["session"]["agents"] for b in first} == {12, 48}
    tickets = [t for b in first for c in b["contexts"] for t in c["tickets"]]
    assert len(tickets) == 384
    assert sum(t["physical"] for t in tickets) == 12
    assert all(t["physical"] == (t["index"] == 0) for t in tickets)
    for ticket in tickets:
        validate_policies(ticket["policies"])
    for session in sessions():
        for count in (3, 7, 11):
            assert len(set(selected_slots(session, count))) == count
