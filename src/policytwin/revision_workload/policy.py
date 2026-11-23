"""Closed typed role policies, dependency scopes, and current exact approval."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from policytwin.coordination.plan import CoordinationExecutionPlan
from policytwin.coordination.schema import commitment
from policytwin.revision_workload.design import ROLES, Target

FIELD_TYPES = {
    "epoch": int,
    "coupling": str,
    "quantity": str,
    "first_epoch": int,
    "workers": int,
    "allowance": int,
    "new_pairs": int,
    "reevaluations": int,
}
OWNED_READS = {
    "designer": {"epoch"},
    "evaluator": {"epoch", "coupling", "quantity", "first_epoch"},
    "producer": {"workers", "allowance", "new_pairs", "reevaluations"},
}
SEMANTICS = {
    "schema": "typed-role-linear-ast.v1",
    "fields": {name: typ.__name__ for name, typ in FIELD_TYPES.items()},
    "epoch": "intervention epoch in the designated branch; others retain epoch1",
    "first_epoch": "inclusive recorded command epoch of the paired trade metric",
    "allowance": "maximum number of new paired executions",
    "work": "current complete typed DAG partitions all twelve requested slots",
    "arithmetic": "unbounded integer add and nonnegative constant multiplication",
}


@dataclass(frozen=True)
class CompiledPolicy:
    owner: str
    identity: str
    reads: tuple[str, ...]
    expression: dict[str, Any]
    function: Callable[[dict[str, Any]], bool]

    def projection(self, values: dict[str, Any]) -> tuple[Any, ...]:
        return (
            self.identity,
            *((name, type(values[name]).__name__, values[name]) for name in self.reads),
        )


def operand(value: Any) -> tuple[type, set[str], Callable[[dict[str, Any]], Any]]:
    if type(value) in (int, str):
        return type(value), set(), lambda row: value
    if not isinstance(value, dict):
        raise ValueError("invalid typed operand")
    if set(value) == {"field"}:
        name = value["field"]
        if name not in FIELD_TYPES:
            raise ValueError("unknown policy field")
        return FIELD_TYPES[name], {name}, lambda row: row[name]
    if (
        set(value) == {"add"}
        and isinstance(value["add"], list)
        and len(value["add"]) >= 2
    ):
        parts = [operand(v) for v in value["add"]]
        if any(typ is not int for typ, _, _ in parts):
            raise ValueError("noninteger sum")
        return (
            int,
            set().union(*(dep for _, dep, _ in parts)),
            lambda row: sum(fn(row) for _, _, fn in parts),
        )
    if (
        set(value) == {"mul"}
        and isinstance(value["mul"], list)
        and len(value["mul"]) == 2
    ):
        coefficient, arg = value["mul"]
        if type(coefficient) is not int or coefficient < 0:
            raise ValueError("invalid integer multiplier")
        typ, deps, fn = operand(arg)
        if typ is not int:
            raise ValueError("noninteger multiplication")
        return int, deps, lambda row: coefficient * fn(row)
    raise ValueError("unsupported typed operand")


def expression(
    value: dict[str, Any],
) -> tuple[set[str], Callable[[dict[str, Any]], bool]]:
    if set(value) == {"op", "args"} and value["op"] == "and":
        if not isinstance(value["args"], list) or len(value["args"]) < 2:
            raise ValueError("and needs at least two arguments")
        children = [expression(v) for v in value["args"]]
        return set().union(*(d for d, _ in children)), lambda row: all(
            f(row) for _, f in children
        )
    if set(value) != {"op", "left", "right"} or value["op"] not in {"eq", "le"}:
        raise ValueError("unsupported closed policy expression")
    lt, ld, lf = operand(value["left"])
    rt, rd, rf = operand(value["right"])
    if lt is not rt or (value["op"] == "le" and lt is not int):
        raise ValueError("policy comparison type mismatch")
    fn = (
        (lambda row: lf(row) == rf(row))
        if value["op"] == "eq"
        else (lambda row: lf(row) <= rf(row))
    )
    return ld | rd, fn


def compile_policy(
    raw: dict[str, Any], interpreter_sha256: str | None = None
) -> CompiledPolicy:
    if set(raw) != {"schema", "owner", "version", "expression", "policy_sha256"}:
        raise ValueError("policy fields differ")
    if (
        raw["schema"] != "revision-role-policy.v1"
        or raw["owner"] not in ROLES
        or type(raw["version"]) is not str
    ):
        raise ValueError("policy schema, owner or version differs")
    body = {k: v for k, v in raw.items() if k != "policy_sha256"}
    if commitment(body) != raw["policy_sha256"]:
        raise ValueError("policy commitment differs")
    dependencies, fn = expression(raw["expression"])
    if not dependencies <= OWNED_READS[raw["owner"]]:
        raise ValueError("undeclared cross-role dependency")
    interpreter = (
        interpreter_sha256 or hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    )
    identity = commitment(
        {
            "policy": raw["policy_sha256"],
            "interpreter": interpreter,
            "typed_semantics": SEMANTICS,
        }
    )
    return CompiledPolicy(
        raw["owner"], identity, tuple(sorted(dependencies)), raw["expression"], fn
    )


def compile_policies(raw: list[dict[str, Any]]) -> list[CompiledPolicy]:
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    compiled = [compile_policy(p, digest) for p in raw]
    if tuple(p.owner for p in compiled) != ROLES:
        raise ValueError("all current role policies in the declared order are required")
    return compiled


def values(option: dict[str, Any]) -> dict[str, Any]:
    return {
        **option["target"],
        "new_pairs": option["new_pairs"],
        "reevaluations": option["reevaluations"],
    }


def sql_operand(value: Any, parameters: list[Any]) -> str:
    """Independent SQL encoding; validation is performed by the closed compiler."""
    if type(value) in (int, str):
        parameters.append(value)
        return "?"
    if "field" in value:
        return '"' + str(value["field"]) + '"'
    if "add" in value:
        return "(" + "+".join(sql_operand(v, parameters) for v in value["add"]) + ")"
    if "mul" in value:
        return "(" + "*".join(sql_operand(v, parameters) for v in value["mul"]) + ")"
    raise ValueError("unknown SQL operand")


def sql_expression(value: dict[str, Any], parameters: list[Any]) -> str:
    if value["op"] == "and":
        return (
            "("
            + " AND ".join(sql_expression(v, parameters) for v in value["args"])
            + ")"
        )
    op = {"eq": "=", "le": "<="}[value["op"]]
    return (
        "("
        + sql_operand(value["left"], parameters)
        + op
        + sql_operand(value["right"], parameters)
        + ")"
    )


def approvals(
    request: dict[str, Any],
    catalog_hash: str,
    option: dict[str, Any],
    policies: list[CompiledPolicy],
) -> list[dict[str, Any]]:
    target = Target.from_mapping(option["target"])
    contract = option["contract"]
    if (
        target.identity != option["target_id"]
        or contract["target"] != option["target"]
        or contract["metric"] != target.metric.to_mapping()
        or contract["session"] != request["session"]
        or contract["allowed_plans"]
        != [CoordinationExecutionPlan(target.workers).sha256]
        or commitment({k: v for k, v in contract.items() if k != "contract_sha256"})
        != contract["contract_sha256"]
    ):
        raise ValueError("offer target, metric, plan or contract differs")
    expected = {
        (s["candidate"], s["replicate"]): (s["trajectory_key"], s["evaluation_key"])
        for s in contract["slots"]
    }
    if len(expected) != 12 or len(option["work_order"]) != 12:
        raise ValueError("offer slot coverage differs")
    coordinates = [(a["candidate"], a["replicate"]) for a in option["work_order"]]
    if len(set(coordinates)) != 12 or set(coordinates) != set(expected):
        raise ValueError("offer work coverage differs")
    for action in option["work_order"]:
        if expected[action["candidate"], action["replicate"]] != (
            action["trajectory_key"],
            action["evaluation_key"],
        ) or action["action"] not in {"execute", "evaluate", "reuse"}:
            raise ValueError("offer work dependency differs")
    for count, kind in (
        ("new_pairs", "execute"),
        ("reevaluations", "evaluate"),
        ("reused_evaluations", "reuse"),
    ):
        if type(option[count]) is not int or option[count] != sum(
            a["action"] == kind for a in option["work_order"]
        ):
            raise ValueError("offer work count differs")
    waived = sorted(
        f for f, v in request["proposed"].items() if target.to_mapping()[f] != v
    )
    if (
        option["waived"] != waived
        or option["new_pairs"] > target.allowance
        or any(target.to_mapping()[c["field"]] != c["value"] for c in request["hard"])
    ):
        raise ValueError("offer concessions or hard requirements differ")
    offer = {
        k: option[k]
        for k in (
            "target",
            "target_id",
            "waived",
            "contract",
            "work_order",
            "new_pairs",
            "reevaluations",
            "reused_evaluations",
        )
    }
    scope = {
        "schema": "current-role-approval.v1",
        "request_sha256": request["request_sha256"],
        "catalog_sha256": catalog_hash,
        "contract_sha256": option["contract"]["contract_sha256"],
        "work_order_sha256": commitment(option["work_order"]),
        "current_policies": [p.identity for p in policies],
        "offer_sha256": commitment(offer),
    }
    result = []
    current = values(option)
    for policy in policies:
        body = {
            **scope,
            "owner": policy.owner,
            "policy_identity": policy.identity,
            "accepted": policy.function(current),
        }
        result.append({**body, "approval_sha256": commitment(body)})
    return result


def authorized(
    request: dict[str, Any],
    catalog_hash: str,
    option: dict[str, Any],
    policies: list[CompiledPolicy],
    records: list[dict[str, Any]],
) -> bool:
    try:
        expected = approvals(request, catalog_hash, option, policies)
    except (KeyError, TypeError, ValueError):
        return False
    return records == expected and all(r["accepted"] for r in expected)
