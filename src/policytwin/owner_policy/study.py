"""Pre-outcome design: refusal-capable owners and complete offer search."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from policytwin.coordination.schema import commitment
from policytwin.negotiated_repair.design import OWNERS, ROLES
from policytwin.negotiated_repair.design import STUDY as PRIOR_STUDY
from policytwin.negotiated_repair.design import sessions as prior_sessions
from policytwin.negotiated_repair.design import ticket as prior_ticket

METHODS = ("one_shot", "work_pruned", "owner_search", "policy_csp")
TICKETS = (
    "candidate_veto",
    "coupling_veto",
    "plan_veto",
    "multi_concession",
    "empty_agreement",
    "permissive_control",
)
STUDY: dict[str, Any] = {
    "label": "post-registration refusal-capable owner policy study v1",
    "session_namespace": "owner-policy-20261030-v1",
    "inherited_seed_prefix": PRIOR_STUDY["seed_root"],
    "seed_derivation": "inherited prefix plus new unique session id and replicate",
    "agents": [6, 24],
    "regimes": ["deep", "responsive"],
    "sessions_per_cell": 1,
    "replicates": 4,
    "epochs": 4,
    "targets": deepcopy(PRIOR_STUDY["targets"]),
    "partial_catalog_size": 6,
    "methods": list(METHODS),
    "tickets": list(TICKETS),
    "owner_policies": "declared allowed own-field values and producer new-pair cap",
    "selection": (
        "all hard-feasible targets, owner acceptance, then waivers/new/reeval/id"
    ),
    "metrics": [
        "released",
        "unresolved",
        "false_abstention_given_feasible_agreement",
        "selected_target_sql_equality",
        "new_pairs",
        "trace_reevaluations",
        "reused_evaluations",
        "concessions",
        "offers_evaluated",
        "vetoed_offers",
        "fresh_scientific_draw_metric_and_choice_equality",
        "invalid_vote_release",
    ],
    "thresholds": "exact integer/rational/scientific/draw equality; caps 6/12 pairs",
    "analysis": (
        "descriptive blocked counts; tickets/methods/votes are repeated conditions"
    ),
    "failure_policy": (
        "retain all failed attempts; no retries, replacement seeds or targets"
    ),
    "claims": [
        "acceptance must precede concession/work pruning if owners can refuse",
        "one-shot and work-pruned search may abstain despite an acceptable target",
        "complete SQL with identical owner predicates should match full search",
        "all released panels must match fresh accepted-target references",
        "no human negotiation, security, natural frequency or wall-clock speedup claim",
    ],
}


def sessions() -> list[dict[str, Any]]:
    return [
        {**s, "id": f"owner-policy-20261030-v1-a{s['agents']}-{s['regime']}"}
        for s in prior_sessions()
    ]


def make_ticket(session: dict[str, Any], kind: str) -> dict[str, Any]:
    if kind not in TICKETS:
        raise ValueError("unknown owner-policy ticket")
    base_kind = {
        "candidate_veto": "candidate_conflict",
        "coupling_veto": "coupling_conflict",
        "plan_veto": "plan_conflict",
        "multi_concession": "candidate_conflict",
        "empty_agreement": "coupling_conflict",
        "permissive_control": "candidate_conflict",
    }[kind]
    request = prior_ticket(session, base_kind)
    request["kind"] = kind
    if kind == "multi_concession":
        request["proposed"].update(coupling="world_separated", workers=1)
        for clause in request["clauses"]:
            clause["value"] = request["proposed"][clause["field"]]
            clause["hard"] = clause["field"] == "metric"
    request["request_sha256"] = commitment(
        {k: v for k, v in request.items() if k != "request_sha256"}
    )
    policies = []
    for owner in ROLES:
        allowed = {
            field: list(STUDY["targets"][field])
            for field, role in OWNERS.items()
            if role == owner
        }
        if owner == "designer" and kind == "candidate_veto":
            allowed["epoch"] = [2]
        if owner == "evaluator" and kind in {"coupling_veto", "empty_agreement"}:
            allowed["coupling"] = ["world_separated"]
        if owner == "producer" and kind == "plan_veto":
            allowed["workers"] = [1]
        body = {
            "schema": "declared-owner-policy.v1",
            "owner": owner,
            "allowed": allowed,
            "max_new_pairs": 6
            if kind in {"multi_concession", "empty_agreement"} and owner == "producer"
            else 12,
        }
        policies.append({**body, "policy_sha256": commitment(body)})
    return {"request": request, "policies": policies}


def schedule() -> list[dict[str, Any]]:
    return [
        {"session": s, "tickets": [make_ticket(s, k) for k in TICKETS]}
        for s in sessions()
    ]
