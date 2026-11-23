"""Finite ordered search, rejection certificates and a full SQL comparator."""

from __future__ import annotations

import sqlite3
from typing import Any

from policytwin.consent_projection.policy import evaluate, signature, validate_policies
from policytwin.coordination.schema import commitment
from policytwin.negotiated_repair.consent import validate_request
from policytwin.negotiated_repair.solver import frontier, minimal_options
from policytwin.owner_policy.protocol import hard_options, rank

METHODS = ("exhaustive", "certificates", "work_pruned", "policy_sql")
ALL_METHODS = (*METHODS, "omit_work", "full_key_memo")
FIELDS = (
    "epoch",
    "coupling",
    "metric",
    "workers",
    "allowance",
    "new_pairs",
    "reevaluations",
)


def memo_key(
    policy: dict[str, Any],
    option: dict[str, Any],
    request: dict[str, Any],
    catalog_hash: str,
    method: str,
) -> str:
    if method == "full_key_memo":
        return commitment(
            [policy["policy_sha256"], request["request_sha256"], catalog_hash, option]
        )
    return signature(
        policy,
        option,
        request["request_sha256"],
        catalog_hash,
        omit_work=method == "omit_work",
    )


def sql_expression(node: dict[str, Any], parameters: list[Any]) -> str:
    op = node["op"]
    if op == "const":
        parameters.append(int(node["value"]))
        return "?"
    if op == "not":
        return "(NOT " + sql_expression(node["arg"], parameters) + ")"
    if op in {"and", "or"}:
        return (
            "("
            + (" AND " if op == "and" else " OR ").join(
                sql_expression(n, parameters) for n in node["args"]
            )
            + ")"
        )
    if node["field"] not in FIELDS or op not in {"eq", "le"}:
        raise ValueError("invalid SQL policy")
    parameters.append(node["value"])
    return str(node["field"]) + (" = ?" if op == "eq" else " <= ?")


def sql_selected(
    options: list[dict[str, Any]],
    request: dict[str, Any],
    policies: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """SQL compiles all policy expressions and hard clauses, not Python outcomes."""
    validate_policies(policies)
    validate_request(request)
    with sqlite3.connect(":memory:") as db:
        db.execute(
            "CREATE TABLE q(id TEXT, epoch INT,coupling TEXT,metric TEXT,"
            "workers INT,allowance INT,new_pairs INT,reevaluations INT)"
        )
        db.executemany(
            "INSERT INTO q VALUES(?,?,?,?,?,?,?,?)",
            [
                (
                    o["target_id"],
                    *(o["target"][k] if k in o["target"] else o[k] for k in FIELDS),
                )
                for o in options
            ],
        )
        conditions = ["new_pairs <= allowance"]
        parameters: list[Any] = []
        for clause in request["clauses"]:
            if clause["hard"]:
                conditions.append(clause["field"] + " = ?")
                parameters.append(clause["value"])
        for policy in sorted(policies, key=lambda p: p["owner"]):
            conditions.append(sql_expression(policy["expression"], parameters))
        costs = []
        for clause in request["clauses"]:
            costs.append("(" + clause["field"] + " <> ?)")
            parameters.append(clause["value"])
        row = db.execute(
            "SELECT id FROM q WHERE "
            + " AND ".join(conditions)
            + " ORDER BY "
            + "+".join(costs)
            + ",new_pairs,reevaluations,id LIMIT 1",
            parameters,
        ).fetchone()
    if row is None:
        return None
    return next(o for o in hard_options(options, request) if o["target_id"] == row[0])


def search(
    options: list[dict[str, Any]],
    request: dict[str, Any],
    policies: list[dict[str, Any]],
    catalog_hash: str,
    method: str,
) -> dict[str, Any]:
    validate_policies(policies)
    if method not in ALL_METHODS:
        raise ValueError("unknown search method")
    ordered = sorted(hard_options(options, request), key=rank)
    if method == "work_pruned":
        ordered = sorted(
            frontier(minimal_options(options, request["clauses"])), key=rank
        )
    if method == "policy_sql":
        selected = sql_selected(options, request, policies)
        if selected is not None and not all(
            evaluate(p["expression"], selected) for p in policies
        ):
            raise RuntimeError("SQL selected an unacceptable target")
        return {
            "method": method,
            "selected": selected,
            "search_predicate_calls": None,
            "visited_candidates": None,
            "skipped_candidates": None,
            "rejections": [],
            "events": [],
            "selection_check_calls": 0 if selected is None else 3,
        }
    rejected: dict[str, dict[str, Any]] = {}
    calls, skipped, visited = 0, 0, 0
    events = []
    selected = None
    for option in ordered:
        visited += 1
        if method in {"certificates", "omit_work", "full_key_memo"}:
            matched = next(
                (
                    memo_key(p, option, request, catalog_hash, method)
                    for p in policies
                    if memo_key(p, option, request, catalog_hash, method) in rejected
                ),
                None,
            )
            if matched is not None:
                skipped += 1
                events.append(
                    {"target_id": option["target_id"], "skip_certificate": matched}
                )
                continue
        accepted = True
        for policy in sorted(policies, key=lambda p: p["owner"]):
            calls += 1
            answer = evaluate(policy["expression"], option)
            events.append(
                {
                    "target_id": option["target_id"],
                    "owner": policy["owner"],
                    "accepted": answer,
                }
            )
            if not answer:
                accepted = False
                sig = memo_key(policy, option, request, catalog_hash, method)
                rejected[sig] = {
                    "signature": sig,
                    "owner": policy["owner"],
                    "witness_target": option["target_id"],
                    "policy_sha256": policy["policy_sha256"],
                }
                break
        if accepted:
            selected = option
            break
    if selected is not None and not all(
        evaluate(p["expression"], selected) for p in policies
    ):
        raise RuntimeError("search selected an unacceptable target")
    return {
        "method": method,
        "selected": selected,
        "search_predicate_calls": calls,
        "visited_candidates": visited,
        "skipped_candidates": skipped,
        "rejections": list(rejected.values()),
        "events": events,
        "selection_check_calls": 0 if selected is None else 3,
    }
