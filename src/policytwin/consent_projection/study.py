"""Outcome-independent generator for previously unused role/store combinations."""

from __future__ import annotations

import random
from typing import Any

from policytwin.coordination.plan import BRANCHES
from policytwin.coordination.schema import commitment
from policytwin.negotiated_repair.design import OWNERS, ROLES
from policytwin.negotiated_repair.design import STUDY as OLD_STUDY

SEED = 202610300731
STUDY: dict[str, Any] = {
    "label": "post-registration consent dependency projection v1",
    "namespace": "consent-projection-20261030-v1",
    "generator_seed": SEED,
    "agents": [12, 48],
    "regimes": ["deep", "responsive"],
    "catalog_sizes": [3, 7, 11],
    "requests_per_context": 32,
    "replicates": 4,
    "epochs": 4,
    "targets": OLD_STUDY["targets"],
    "hard_clause_probability": 0.2,
    "policy_shape_probabilities": {
        "const_true": 0.2,
        "atom": 0.4,
        "and": 0.2,
        "or": 0.1,
        "not": 0.1,
    },
    "atom_selection": (
        "uniform among owned target fields; producer additionally "
        "new_pairs and reevaluations"
    ),
    "integer_atom_operator": "eq for target fields; le for work counts",
    "work_thresholds": [0, 6, 12],
    "catalog_selection": (
        "lowest global slot hash; unequal candidate coverage permitted"
    ),
    "physical_requests": (
        "index 0 in each context; no feasibility or outcome replacement"
    ),
    "methods": ["exhaustive", "certificates", "work_pruned", "policy_sql"],
    "planning_controls": [
        "omit_work removes both new_pairs and reevaluations from rejection signatures",
        "full_key_memo caches only exact complete proposal inputs",
    ],
    "rank": "waiver count, new pairs, reevaluations, target id",
    "metrics": [
        "selected_target_and_work_equality",
        "false_abstention",
        "selection_disagreement",
        "search_predicate_calls",
        "selection_check_calls",
        "authorization_predicate_calls",
        "skipped_candidates",
        "exact_fresh_report",
        "actual_pairs",
        "local_failures",
    ],
    "thresholds": "exact scientific/draw/metric/integer-rational decision equality",
    "analysis": (
        "descriptive counts by scientific block and catalog; "
        "generated policies are repeated designed conditions"
    ),
    "claims": [
        "complete policy dependencies preserve the least acceptable target",
        "work-only pruning can discard agreement",
        "omitted work dependencies can make rejection reuse incomplete",
        "full SQL constraints can realize the same selection",
        "predicate-call counts are not latency or human effort",
    ],
    "failure_policy": (
        "retain failed attempts and no-agreement cases; no retries or replacement seeds"
    ),
}


def sessions() -> list[dict[str, Any]]:
    return [
        {
            "id": f"{STUDY['namespace']}-a{agents}-{regime}",
            "index": 2 * ai + ri,
            "agents": agents,
            "regime": regime,
            "liquidity": 5000 if regime == "deep" else 6 * agents,
            "target": BRANCHES[(2 * ai + ri) % len(BRANCHES)],
        }
        for ai, agents in enumerate(STUDY["agents"])
        for ri, regime in enumerate(STUDY["regimes"])
    ]


def atom(rng: random.Random, owner: str) -> dict[str, Any]:
    fields = [k for k, role in OWNERS.items() if role == owner]
    if owner == "producer":
        fields += ["new_pairs", "reevaluations"]
    field = rng.choice(fields)
    if field in {"new_pairs", "reevaluations"}:
        return {
            "op": "le",
            "field": field,
            "value": rng.choice(STUDY["work_thresholds"]),
        }
    return {"op": "eq", "field": field, "value": rng.choice(STUDY["targets"][field])}


def make_policy(rng: random.Random, owner: str) -> dict[str, Any]:
    shape = rng.choices(
        list(STUDY["policy_shape_probabilities"]),
        weights=list(STUDY["policy_shape_probabilities"].values()),
    )[0]
    if shape == "const_true":
        expression = {"op": "const", "value": True}
    elif shape == "atom":
        expression = atom(rng, owner)
    elif shape == "not":
        expression = {"op": "not", "arg": atom(rng, owner)}
    else:
        expression = {"op": shape, "args": [atom(rng, owner), atom(rng, owner)]}
    body = {
        "schema": "closed-owner-policy.v1",
        "owner": owner,
        "expression": expression,
    }
    return {**body, "policy_sha256": commitment(body)}


def schedule() -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    blocks = []
    for session in sessions():
        contexts = []
        for count in STUDY["catalog_sizes"]:
            tickets = []
            for index in range(STUDY["requests_per_context"]):
                proposed = {k: rng.choice(v) for k, v in STUDY["targets"].items()}
                clauses = [
                    {
                        "id": OWNERS[k] + "." + k,
                        "owner": OWNERS[k],
                        "field": k,
                        "value": value,
                        "hard": rng.random() < STUDY["hard_clause_probability"],
                    }
                    for k, value in proposed.items()
                ]
                body = {
                    "schema": "role-owned-revision-request.v1",
                    "session": session,
                    "kind": f"store{count}-request{index:02d}",
                    "proposed": proposed,
                    "clauses": clauses,
                    "availability": count,
                }
                request = {**body, "request_sha256": commitment(body)}
                tickets.append(
                    {
                        "index": index,
                        "request": request,
                        "policies": [make_policy(rng, owner) for owner in ROLES],
                        "physical": index == 0,
                    }
                )
            contexts.append({"catalog_size": count, "tickets": tickets})
        blocks.append({"session": session, "contexts": contexts})
    return blocks


def selected_slots(session: dict[str, Any], count: int) -> list[tuple[str, int]]:
    slots = [(b, r) for b in BRANCHES for r in range(4)]
    return sorted(
        slots,
        key=lambda s: commitment([STUDY["namespace"], session["id"], *s, "catalog"]),
    )[:count]
