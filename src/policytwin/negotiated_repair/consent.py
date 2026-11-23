"""Executable software-owner approval; no human or remote attestation claim."""

from __future__ import annotations

from typing import Any

from policytwin.coordination.schema import commitment
from policytwin.negotiated_repair.design import OWNERS, ROLES
from policytwin.negotiated_repair.solver import failed_clauses, feasible


def validate_request(request: dict[str, Any]) -> None:
    body = {k: v for k, v in request.items() if k != "request_sha256"}
    if commitment(body) != request["request_sha256"]:
        raise ValueError("request commitment differs")
    clauses = request["clauses"]
    if len(clauses) != len(OWNERS) or {c["field"] for c in clauses} != set(OWNERS):
        raise ValueError("role clause coverage differs")
    for clause in clauses:
        field = clause["field"]
        if (
            clause["owner"] != OWNERS[field]
            or clause["id"] != OWNERS[field] + "." + field
        ):
            raise ValueError("clause owner differs")
        if (
            type(clause["hard"]) is not bool
            or clause["value"] != request["proposed"][field]
        ):
            raise ValueError("invalid clause or proposal")


def proposal(request: dict[str, Any], option: dict[str, Any]) -> dict[str, Any]:
    validate_request(request)
    hard = [c for c in request["clauses"] if c["hard"]]
    if not feasible(option, hard) or option["waived"] != failed_clauses(
        option, request["clauses"]
    ):
        raise ValueError("counteroffer lacks a valid concession set")
    body = {
        "schema": "comparison-counteroffer.v1",
        "request_sha256": request["request_sha256"],
        "target_id": option["target_id"],
        "contract_sha256": option["contract"]["contract_sha256"],
        "waived": option["waived"],
        "work_order_sha256": commitment(option["work_order"]),
    }
    return {**body, "proposal_sha256": commitment(body)}


def approvals(request: dict[str, Any], option: dict[str, Any]) -> list[dict[str, Any]]:
    offer = proposal(request, option)
    result = []
    for owner in ROLES:
        body = {
            "schema": "software-owner-approval.v1",
            "owner": owner,
            "proposal_sha256": offer["proposal_sha256"],
            "request_sha256": request["request_sha256"],
            "contract_sha256": option["contract"]["contract_sha256"],
            "owned_waivers": [
                c["id"]
                for c in request["clauses"]
                if c["owner"] == owner and c["id"] in option["waived"]
            ],
        }
        result.append({**body, "approval_sha256": commitment(body)})
    return result


def authorized(
    request: dict[str, Any], option: dict[str, Any], records: list[dict[str, Any]]
) -> bool:
    """Compare exact owner approvals with the requested counteroffer relation."""
    try:
        expected = approvals(request, option)
    except (KeyError, ValueError, TypeError):
        return False
    if len(records) != len(ROLES) or {r.get("owner") for r in records} != set(ROLES):
        return False
    return sorted(records, key=lambda r: r["owner"]) == sorted(
        expected, key=lambda r: r["owner"]
    )
