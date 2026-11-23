"""Closed owner-policy language and exact proposal-specific approvals."""

from __future__ import annotations

from typing import Any

from policytwin.coordination.schema import commitment
from policytwin.negotiated_repair.consent import proposal
from policytwin.negotiated_repair.design import OWNERS, ROLES, STUDY

WORK_FIELDS = {"new_pairs", "reevaluations"}


def validate_ast(node: dict[str, Any], owner: str) -> set[str]:
    op = node.get("op")
    if op == "const":
        if set(node) != {"op", "value"} or type(node["value"]) is not bool:
            raise ValueError("invalid constant")
        return set()
    if op in {"and", "or"}:
        if set(node) != {"op", "args"} or not 1 <= len(node["args"]) <= 2:
            raise ValueError("invalid Boolean arguments")
        return set().union(*(validate_ast(n, owner) for n in node["args"]))
    if op == "not":
        if set(node) != {"op", "arg"}:
            raise ValueError("invalid negation")
        return validate_ast(node["arg"], owner)
    if op not in {"eq", "le"} or set(node) != {"op", "field", "value"}:
        raise ValueError("unknown policy expression")
    field, value = node["field"], node["value"]
    if field in WORK_FIELDS:
        if owner != "producer" or type(value) is not int or not 0 <= value <= 12:
            raise ValueError("invalid work dependency")
    elif field not in OWNERS or OWNERS[field] != owner:
        raise ValueError("foreign owner field")
    elif not any(
        type(value) is type(v) and value == v for v in STUDY["targets"][field]
    ):
        raise ValueError("invalid target value")
    if op == "le" and type(value) is not int:
        raise ValueError("ordered comparison needs integer")
    return {field}


def validate_policies(policies: list[dict[str, Any]]) -> None:
    if len(policies) != 3 or {p.get("owner") for p in policies} != set(ROLES):
        raise ValueError("owner coverage differs")
    for policy in policies:
        if set(policy) != {"schema", "owner", "expression", "policy_sha256"}:
            raise ValueError("policy fields differ")
        if policy["schema"] != "closed-owner-policy.v1":
            raise ValueError("unknown policy schema")
        if (
            commitment({k: v for k, v in policy.items() if k != "policy_sha256"})
            != policy["policy_sha256"]
        ):
            raise ValueError("policy commitment differs")
        validate_ast(policy["expression"], policy["owner"])


def field_value(option: dict[str, Any], field: str) -> Any:
    return option[field] if field in WORK_FIELDS else option["target"][field]


def evaluate(node: dict[str, Any], option: dict[str, Any]) -> bool:
    op = node["op"]
    if op == "const":
        return bool(node["value"])
    if op == "not":
        return not evaluate(node["arg"], option)
    if op == "and":
        return all(evaluate(n, option) for n in node["args"])
    if op == "or":
        return any(evaluate(n, option) for n in node["args"])
    value = field_value(option, node["field"])
    if op == "eq":
        return bool(value == node["value"])
    if op == "le":
        return bool(value <= node["value"])
    raise ValueError("unknown expression")


def signature(
    policy: dict[str, Any],
    option: dict[str, Any],
    request_hash: str,
    catalog_hash: str,
    *,
    omit_work: bool = False,
) -> str:
    fields = validate_ast(policy["expression"], policy["owner"])
    if omit_work:
        fields -= WORK_FIELDS
    return commitment(
        {
            "policy": policy["policy_sha256"],
            "request": request_hash,
            "catalog": catalog_hash,
            "projection": {k: field_value(option, k) for k in sorted(fields)},
        }
    )


def votes(
    request: dict[str, Any],
    option: dict[str, Any],
    policies: list[dict[str, Any]],
    catalog_hash: str,
) -> list[dict[str, Any]]:
    validate_policies(policies)
    offer = proposal(request, option)
    records = []
    for policy in sorted(policies, key=lambda p: p["owner"]):
        body = {
            "schema": "projection-owner-vote.v1",
            "owner": policy["owner"],
            "policy_sha256": policy["policy_sha256"],
            "request_sha256": request["request_sha256"],
            "catalog_sha256": catalog_hash,
            "proposal_sha256": offer["proposal_sha256"],
            "accepted": evaluate(policy["expression"], option),
        }
        records.append({**body, "vote_sha256": commitment(body)})
    return records


def authorized(
    request: dict[str, Any],
    option: dict[str, Any],
    policies: list[dict[str, Any]],
    catalog_hash: str,
    records: list[dict[str, Any]],
) -> bool:
    try:
        expected = votes(request, option, policies, catalog_hash)
        return (
            all(v["accepted"] for v in expected)
            and sorted(records, key=lambda r: r["owner"]) == expected
        )
    except (KeyError, TypeError, ValueError):
        return False
