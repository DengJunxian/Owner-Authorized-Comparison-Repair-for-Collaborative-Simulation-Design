"""Refusal-capable software votes and acceptance-before-pruning search."""

from __future__ import annotations

import sqlite3
from typing import Any

from policytwin.coordination.schema import commitment
from policytwin.negotiated_repair.consent import proposal, validate_request
from policytwin.negotiated_repair.design import OWNERS, ROLES, STUDY
from policytwin.negotiated_repair.solver import (
    failed_clauses,
    feasible,
    frontier,
    minimal_options,
)


def validate_policies(policies: list[dict[str, Any]]) -> None:
    if len(policies) != len(ROLES) or {p.get("owner") for p in policies} != set(ROLES):
        raise ValueError("policy owner coverage differs")
    for policy in policies:
        body = {k: v for k, v in policy.items() if k != "policy_sha256"}
        if commitment(body) != policy["policy_sha256"]:
            raise ValueError("policy commitment differs")
        allowed = policy["allowed"]
        if set(allowed) != {k for k, role in OWNERS.items() if role == policy["owner"]}:
            raise ValueError("policy field ownership differs")
        if any(
            not values
            or len(values) != len(set(values))
            or any(v not in STUDY["targets"][k] for v in values)
            for k, values in allowed.items()
        ):
            raise ValueError("invalid owner field values")
        if type(policy["max_new_pairs"]) is not int or policy["max_new_pairs"] < 0:
            raise ValueError("invalid new-pair cap")


def accepts(policy: dict[str, Any], option: dict[str, Any]) -> bool:
    return all(
        option["target"][k] in values for k, values in policy["allowed"].items()
    ) and (
        policy["owner"] != "producer" or option["new_pairs"] <= policy["max_new_pairs"]
    )


def votes(
    request: dict[str, Any],
    option: dict[str, Any],
    policies: list[dict[str, Any]],
    catalog_sha256: str,
) -> list[dict[str, Any]]:
    validate_policies(policies)
    offer = proposal(request, option)
    result = []
    for policy in sorted(policies, key=lambda p: p["owner"]):
        body = {
            "schema": "owner-policy-vote.v1",
            "owner": policy["owner"],
            "policy_sha256": policy["policy_sha256"],
            "proposal_sha256": offer["proposal_sha256"],
            "request_sha256": request["request_sha256"],
            "catalog_sha256": catalog_sha256,
            "accepted": accepts(policy, option),
        }
        result.append({**body, "vote_sha256": commitment(body)})
    return result


def authorized(
    request: dict[str, Any],
    option: dict[str, Any],
    policies: list[dict[str, Any]],
    catalog_sha256: str,
    records: list[dict[str, Any]],
) -> bool:
    try:
        expected = votes(request, option, policies, catalog_sha256)
        return (
            all(v["accepted"] for v in expected)
            and sorted(records, key=lambda r: r["owner"]) == expected
        )
    except (KeyError, TypeError, ValueError):
        return False


def hard_options(
    options: list[dict[str, Any]], request: dict[str, Any]
) -> list[dict[str, Any]]:
    validate_request(request)
    hard = [c for c in request["clauses"] if c["hard"]]
    return [
        {**o, "waived": failed_clauses(o, request["clauses"])}
        for o in options
        if feasible(o, hard)
    ]


def rank(option: dict[str, Any]) -> tuple[Any, ...]:
    return (
        len(option["waived"]),
        option["new_pairs"],
        option["reevaluations"],
        option["target_id"],
    )


def sql_candidates(
    options: list[dict[str, Any]],
    request: dict[str, Any],
    policies: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Independent SQL expression of hard clauses, owner acceptance and ranking."""
    validate_request(request)
    validate_policies(policies)
    with sqlite3.connect(":memory:") as db:
        db.execute("CREATE TABLE targets(id TEXT,n INT,l INT,cap INT)")
        db.execute("CREATE TABLE vals(id TEXT,field TEXT,value TEXT)")
        db.execute("CREATE TABLE clauses(id TEXT,field TEXT,value TEXT,hard INT)")
        db.execute("CREATE TABLE allowed(field TEXT,value TEXT)")
        db.executemany(
            "INSERT INTO targets VALUES(?,?,?,?)",
            [
                (
                    o["target_id"],
                    o["new_pairs"],
                    o["reevaluations"],
                    o["target"]["allowance"],
                )
                for o in options
            ],
        )
        db.executemany(
            "INSERT INTO vals VALUES(?,?,?)",
            [
                (o["target_id"], k, str(v))
                for o in options
                for k, v in o["target"].items()
            ],
        )
        db.executemany(
            "INSERT INTO clauses VALUES(?,?,?,?)",
            [
                (c["id"], c["field"], str(c["value"]), c["hard"])
                for c in request["clauses"]
            ],
        )
        db.executemany(
            "INSERT INTO allowed VALUES(?,?)",
            [
                (k, str(v))
                for p in policies
                for k, values in p["allowed"].items()
                for v in values
            ],
        )
        cap = next(p["max_new_pairs"] for p in policies if p["owner"] == "producer")
        ids = [
            r[0]
            for r in db.execute(
                """
            SELECT t.id FROM targets t WHERE t.n<=t.cap AND t.n<=?
            AND NOT EXISTS(SELECT 1 FROM vals v JOIN clauses c ON v.field=c.field
              WHERE v.id=t.id AND c.hard=1 AND v.value<>c.value)
            AND NOT EXISTS(SELECT 1 FROM vals v WHERE v.id=t.id AND NOT EXISTS
              (SELECT 1 FROM allowed a WHERE a.field=v.field AND a.value=v.value))
            ORDER BY (SELECT COUNT(*) FROM vals v JOIN clauses c ON v.field=c.field
              WHERE v.id=t.id AND v.value<>c.value),t.n,t.l,t.id
            """,
                (cap,),
            )
        ]
        result = []
        for identity in ids:
            option = next(o for o in options if o["target_id"] == identity)
            waived = [
                r[0]
                for r in db.execute(
                    """
                SELECT c.id FROM vals v JOIN clauses c ON v.field=c.field
                WHERE v.id=? AND v.value<>c.value ORDER BY c.id
                """,
                    (identity,),
                )
            ]
            result.append({**option, "waived": waived})
        return result


def negotiate(
    options: list[dict[str, Any]],
    request: dict[str, Any],
    policies: list[dict[str, Any]],
    catalog_sha256: str,
    method: str,
) -> dict[str, Any]:
    validate_policies(policies)
    if method == "policy_csp":
        candidates = sql_candidates(options, request, policies)
    elif method == "owner_search":
        candidates = sorted(hard_options(options, request), key=rank)
    elif method in {"one_shot", "work_pruned"}:
        candidates = sorted(
            frontier(minimal_options(options, request["clauses"])), key=rank
        )
        if method == "one_shot":
            candidates = candidates[:1]
    else:
        raise ValueError("unknown owner-policy method")
    offered = []
    chosen = None
    for option in candidates:
        records = votes(request, option, policies, catalog_sha256)
        accepted = authorized(request, option, policies, catalog_sha256, records)
        offered.append(
            {
                "target_id": option["target_id"],
                "waived": option["waived"],
                "votes": records,
                "accepted": accepted,
            }
        )
        if accepted:
            chosen = option
            break
    return {
        "method": method,
        "offered": offered,
        "selected": chosen,
        "status": "agreed" if chosen is not None else "no_agreement",
    }
