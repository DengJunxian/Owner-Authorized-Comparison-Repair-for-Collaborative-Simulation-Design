"""Engineering fixtures for refusal-capable owners; not experimental evidence."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from policytwin.coordination.schema import commitment
from policytwin.negotiated_repair.design import Target, available, contract
from policytwin.negotiated_repair.solver import alternatives
from policytwin.owner_policy.protocol import (
    authorized,
    negotiate,
    validate_policies,
    votes,
)
from policytwin.owner_policy.study import (
    METHODS,
    TICKETS,
    make_ticket,
    schedule,
    sessions,
)
from policytwin.typed_revision.contract import EvaluationToken, TraceToken
from policytwin.typed_revision.runtime import expectations


def fixture(kind):
    session = sessions()[0]
    declared = contract(session, Target(1, "shared", "notional", 2, 12))
    traces, evaluations = [], []
    for branch, rep in available(session, 6):
        t, e = expectations(declared)[branch, rep]
        traces.append(
            TraceToken(
                f"t-{branch}-{rep}",
                branch,
                rep,
                t,
                "capsule",
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
                "capsule",
                declared["allowed_plans"][0],
                "receipt",
                True,
                rep,
            )
        )
    ticket = make_ticket(session, kind)
    return (
        ticket,
        alternatives(ticket["request"], traces, evaluations),
        traces,
        evaluations,
    )


def test_schedule_has_new_ids_and_distinct_six_cases():
    blocks = schedule()
    assert len(blocks) == 4
    assert len({b["session"]["id"] for b in blocks}) == 4
    for block in blocks:
        assert block["session"]["id"].startswith("owner-policy-20261030-v1")
        assert {t["request"]["kind"] for t in block["tickets"]} == set(TICKETS)
        for ticket in block["tickets"]:
            validate_policies(ticket["policies"])


@pytest.mark.parametrize("kind", TICKETS)
def test_exhaustive_and_sql_target_parity_and_safe_veto(kind):
    ticket, options, _, _ = fixture(kind)
    decisions = {
        m: negotiate(options, ticket["request"], ticket["policies"], "catalog", m)
        for m in METHODS
    }
    assert decisions["owner_search"]["selected"] == decisions["policy_csp"]["selected"]
    if kind == "empty_agreement":
        assert all(d["selected"] is None for d in decisions.values())
    elif kind == "permissive_control":
        assert all(
            d["selected"] == decisions["owner_search"]["selected"]
            for d in decisions.values()
        )
    elif kind == "multi_concession":
        assert decisions["one_shot"]["selected"] is None
        assert decisions["work_pruned"]["selected"] is None
        assert len(decisions["owner_search"]["selected"]["waived"]) == 3
        assert decisions["owner_search"]["selected"]["new_pairs"] == 6
    else:
        assert decisions["one_shot"]["selected"] is None
        assert decisions["work_pruned"]["selected"] is not None
        assert decisions["owner_search"]["selected"] is not None


def test_votes_bind_policy_request_work_and_catalog_and_reject_duplicates():
    ticket, options, _, _ = fixture("candidate_veto")
    request, policies = ticket["request"], ticket["policies"]
    chosen = negotiate(options, request, policies, "catalog", "owner_search")[
        "selected"
    ]
    records = votes(request, chosen, policies, "catalog")
    assert authorized(request, chosen, policies, "catalog", records)
    assert not authorized(request, chosen, policies, "changed", records)
    assert not authorized(request, chosen, policies, "catalog", records + records[:1])
    assert not authorized(request, chosen, policies, "catalog", records[:-1])
    for field, value in (
        ("accepted", False),
        ("policy_sha256", "other"),
        ("request_sha256", "other"),
        ("proposal_sha256", "other"),
    ):
        changed = deepcopy(records)
        changed[0][field] = value
        changed[0]["vote_sha256"] = commitment(
            {k: v for k, v in changed[0].items() if k != "vote_sha256"}
        )
        assert not authorized(request, chosen, policies, "catalog", changed)
    changed_policy = deepcopy(policies)
    producer = next(p for p in changed_policy if p["owner"] == "producer")
    producer["max_new_pairs"] = 0
    producer["policy_sha256"] = commitment(
        {k: v for k, v in producer.items() if k != "policy_sha256"}
    )
    assert not authorized(request, chosen, changed_policy, "catalog", records)
    forged = votes(request, chosen, changed_policy, "catalog")
    for vote in forged:
        vote["accepted"] = True
        vote["vote_sha256"] = commitment(
            {k: v for k, v in vote.items() if k != "vote_sha256"}
        )
    assert not authorized(request, chosen, changed_policy, "catalog", forged)


def test_policies_fail_closed_on_missing_owner_and_foreign_fields():
    policies = fixture("permissive_control")[0]["policies"]
    with pytest.raises(ValueError, match="owner coverage"):
        validate_policies(policies[:-1])
    changed = deepcopy(policies)
    changed[0]["allowed"]["allowance"] = [12]
    changed[0]["policy_sha256"] = commitment(
        {k: v for k, v in changed[0].items() if k != "policy_sha256"}
    )
    with pytest.raises(ValueError, match="ownership"):
        validate_policies(changed)


def test_selection_is_outcome_independent_and_input_order_invariant():
    ticket, options, traces, evaluations = fixture("multi_concession")
    altered = alternatives(
        ticket["request"], traces, [replace(e, value=-1000000000) for e in evaluations]
    )
    assert options == altered
    for method in METHODS:
        before = negotiate(
            options, ticket["request"], ticket["policies"], "catalog", method
        )
        after = negotiate(
            list(reversed(altered)),
            ticket["request"],
            list(reversed(ticket["policies"])),
            "catalog",
            method,
        )
        assert before == after
