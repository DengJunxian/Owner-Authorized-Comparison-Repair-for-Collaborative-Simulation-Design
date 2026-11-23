"""Finite owner constraints and their store-relative evidence-work frontier."""

from __future__ import annotations

import sqlite3
from itertools import combinations
from typing import Any

from policytwin.negotiated_repair.design import STUDY, contract, targets
from policytwin.typed_revision.contract import EvaluationToken, TraceToken, work_order
from policytwin.typed_revision.runtime import expectations


def sql_order(
    expected: dict[tuple[str, int], tuple[str, str]],
    plan: str,
    traces: list[TraceToken],
    evaluations: list[EvaluationToken],
) -> list[dict[str, Any]]:
    """A complete native two-level dependency builder, expressed in SQL."""
    with sqlite3.connect(":memory:") as db:
        db.execute("CREATE TABLE wanted (b TEXT, r INTEGER, t TEXT, e TEXT)")
        db.execute(
            "CREATE TABLE traces "
            "(b TEXT, r INTEGER, t TEXT, p TEXT, ok INTEGER, id TEXT)"
        )
        db.execute(
            "CREATE TABLE evals "
            "(b TEXT, r INTEGER, t TEXT, e TEXT, p TEXT, ok INTEGER, id TEXT)"
        )
        db.executemany(
            "INSERT INTO wanted VALUES (?,?,?,?)",
            [(*s, *v) for s, v in expected.items()],
        )
        db.executemany(
            "INSERT INTO traces VALUES (?,?,?,?,?,?)",
            [
                (*t.slot, t.trajectory_key, t.plan_sha256, t.local_pass, t.object_id)
                for t in traces
            ],
        )
        db.executemany(
            "INSERT INTO evals VALUES (?,?,?,?,?,?,?)",
            [
                (
                    *e.slot,
                    e.trajectory_key,
                    e.evaluation_key,
                    e.plan_sha256,
                    e.local_pass,
                    e.object_id,
                )
                for e in evaluations
            ],
        )
        query = """SELECT w.b,w.r,w.t,w.e,
          (SELECT MIN(id) FROM evals e
            WHERE e.b=w.b AND e.r=w.r AND e.t=w.t AND e.e=w.e
              AND e.p=? AND e.ok=1),
          (SELECT MIN(id) FROM traces t
            WHERE t.b=w.b AND t.r=w.r AND t.t=w.t AND t.p=? AND t.ok=1)
          FROM wanted w ORDER BY w.b,w.r"""
        return [
            {
                "candidate": b,
                "replicate": r,
                "trajectory_key": t,
                "evaluation_key": e,
                "action": "reuse" if eid else "evaluate" if tid else "execute",
                "object_id": eid or tid,
            }
            for b, r, t, e, eid, tid in db.execute(query, (plan, plan))
        ]


def alternatives(
    request: dict[str, Any],
    traces: list[TraceToken],
    evaluations: list[EvaluationToken],
) -> list[dict[str, Any]]:
    rows = []
    for target in targets():
        declared = contract(request["session"], target)
        expected = expectations(declared)
        order = work_order(
            expected, tuple(declared["allowed_plans"]), traces, evaluations
        )
        independent = sql_order(
            expected, declared["allowed_plans"][0], traces, evaluations
        )
        if order != independent:
            raise RuntimeError("complete SQL dependency builder disagrees")
        rows.append(
            {
                "target": target.to_mapping(),
                "target_id": target.identity,
                "contract": declared,
                "work_order": order,
                "new_pairs": sum(w["action"] == "execute" for w in order),
                "reevaluations": sum(w["action"] == "evaluate" for w in order),
                "reused_evaluations": sum(w["action"] == "reuse" for w in order),
                "sql_work_order_equal": True,
            }
        )
    return rows


def failed_clauses(option: dict[str, Any], clauses: list[dict[str, Any]]) -> list[str]:
    return sorted(
        c["id"] for c in clauses if option["target"][c["field"]] != c["value"]
    )


def feasible(option: dict[str, Any], clauses: list[dict[str, Any]]) -> bool:
    return option["new_pairs"] <= option["target"]["allowance"] and not failed_clauses(
        option, clauses
    )


def conflict_cores(
    options: list[dict[str, Any]], clauses: list[dict[str, Any]]
) -> list[list[str]]:
    """All inclusion-minimal inconsistent subsets over the declared finite domain."""
    cores: list[list[str]] = []
    for size in range(1, len(clauses) + 1):
        for subset in combinations(clauses, size):
            identifiers = sorted(c["id"] for c in subset)
            if any(set(core) <= set(identifiers) for core in cores):
                continue
            if not any(feasible(o, list(subset)) for o in options):
                cores.append(identifiers)
    return sorted(cores)


def minimal_options(
    options: list[dict[str, Any]], clauses: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    hard = [c for c in clauses if c["hard"]]
    possible = []
    for option in options:
        if feasible(option, hard):
            possible.append({**option, "waived": failed_clauses(option, clauses)})
    if not possible:
        return []
    minimum = min(len(o["waived"]) for o in possible)
    return [o for o in possible if len(o["waived"]) == minimum]


def frontier(options: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
        return (
            a["new_pairs"] <= b["new_pairs"]
            and a["reevaluations"] <= b["reevaluations"]
            and (a["new_pairs"], a["reevaluations"])
            != (b["new_pairs"], b["reevaluations"])
        )

    return [b for b in options if not any(dominates(a, b) for a in options)]


def selected(
    options: list[dict[str, Any]], request: dict[str, Any], method: str
) -> dict[str, Any] | None:
    if method == "csp_dag":
        return sql_selected(options, request)
    possible = minimal_options(options, request["clauses"])
    if method == "fixed_dag":
        possible = [o for o in possible if not o["waived"]]
    elif method == "frontier":
        possible = frontier(possible)
    elif method != "serial":
        raise ValueError("unknown acquisition policy")
    if not possible:
        return None
    if method == "serial":
        rank = {field: i for i, field in enumerate(STUDY["serial_relaxation_order"])}
        return min(
            possible,
            key=lambda o: (
                tuple(sorted(rank[c.split(".")[1]] for c in o["waived"])),
                o["target_id"],
            ),
        )
    return min(
        possible, key=lambda o: (o["new_pairs"], o["reevaluations"], o["target_id"])
    )


def sql_selected(
    options: list[dict[str, Any]], request: dict[str, Any]
) -> dict[str, Any] | None:
    """Generic exhaustive finite CSP + the same full two-level DAG obligations."""
    with sqlite3.connect(":memory:") as db:
        db.execute(
            "CREATE TABLE opts (id TEXT, epoch INTEGER, coupling TEXT, "
            "metric TEXT, workers INTEGER, allowance INTEGER, n INTEGER, g INTEGER)"
        )
        db.execute(
            "CREATE TABLE clauses "
            "(id TEXT, field TEXT, iv INTEGER, sv TEXT, hard INTEGER)"
        )
        db.executemany(
            "INSERT INTO opts VALUES (?,?,?,?,?,?,?,?)",
            [
                (
                    o["target_id"],
                    *[
                        o["target"][k]
                        for k in ("epoch", "coupling", "metric", "workers", "allowance")
                    ],
                    o["new_pairs"],
                    o["reevaluations"],
                )
                for o in options
            ],
        )
        db.executemany(
            "INSERT INTO clauses VALUES (?,?,?,?,?)",
            [
                (
                    c["id"],
                    c["field"],
                    c["value"] if type(c["value"]) is int else None,
                    c["value"] if type(c["value"]) is str else None,
                    c["hard"],
                )
                for c in request["clauses"]
            ],
        )
        db.execute("""CREATE VIEW violations AS
          SELECT o.id AS target_id,c.id AS clause_id,c.hard
          FROM opts o CROSS JOIN clauses c WHERE
          CASE c.field WHEN 'epoch' THEN o.epoch<>c.iv
          WHEN 'coupling' THEN o.coupling<>c.sv
          WHEN 'metric' THEN o.metric<>c.sv WHEN 'workers' THEN o.workers<>c.iv
          WHEN 'allowance' THEN o.allowance<>c.iv END""")
        row = db.execute("""SELECT o.id FROM opts o
          WHERE o.n<=o.allowance AND NOT EXISTS
          (SELECT 1 FROM violations v WHERE v.target_id=o.id AND v.hard=1)
          ORDER BY (SELECT COUNT(*) FROM violations v WHERE v.target_id=o.id),
          o.n,o.g,o.id LIMIT 1""").fetchone()
        if row is None:
            return None
        found = next(o for o in options if o["target_id"] == row[0])
        waived = [
            r[0]
            for r in db.execute(
                "SELECT clause_id FROM violations WHERE target_id=? ORDER BY clause_id",
                (row[0],),
            )
        ]
        return {**found, "waived": waived}
