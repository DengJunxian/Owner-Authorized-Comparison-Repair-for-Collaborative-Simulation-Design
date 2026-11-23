"""Measured complete planners with distinct semantic-result cache lifetimes."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from time import perf_counter_ns, process_time_ns
from typing import Any

from policytwin.coordination.plan import CoordinationExecutionPlan
from policytwin.coordination.schema import commitment
from policytwin.revision_workload.design import (
    FIELDS,
    METHODS,
    Target,
    recipes,
    targets,
)
from policytwin.revision_workload.policy import (
    CompiledPolicy,
    approvals,
    authorized,
    compile_policies,
    sql_expression,
    values,
)
from policytwin.typed_revision.contract import EvaluationToken, TraceToken, work_order
from policytwin.typed_revision.metrics import implementation_hash


@dataclass
class DependencyCache:
    """Equal private immutable-dependency caches; never cache current work/rank."""

    trajectories: dict[Any, str] = field(default_factory=dict)
    evaluations: dict[tuple[str, str, int], str] = field(default_factory=dict)
    layouts: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    metric_source: str | None = None
    compiled_trajectories: int = 0
    compiled_evaluations: int = 0
    session_sha256: str | None = None

    def build(self, session: dict[str, Any], size: int) -> list[dict[str, Any]]:
        identity = commitment(session)
        if self.session_sha256 is not None and self.session_sha256 != identity:
            raise ValueError("dependency cache belongs to another scientific session")
        self.session_sha256 = identity
        if size in self.layouts:
            return self.layouts[size]
        if self.metric_source is None:
            self.metric_source = implementation_hash()
        result = []
        for target in targets(size):
            expected = {}
            for slot, recipe in recipes(session, target).items():
                if recipe not in self.trajectories:
                    self.trajectories[recipe] = commitment(
                        {k: v for k, v in recipe.key_payload().items() if k != "metric"}
                    )
                    self.compiled_trajectories += 1
                trajectory = self.trajectories[recipe]
                key = trajectory, target.quantity, target.first_epoch
                if key not in self.evaluations:
                    self.evaluations[key] = commitment(
                        {
                            "trajectory_key": trajectory,
                            "metric": target.metric.to_mapping(),
                            "metric_source_sha256": self.metric_source,
                        }
                    )
                    self.compiled_evaluations += 1
                expected[slot] = trajectory, self.evaluations[key]
            result.append(
                {
                    "target": target.to_mapping(),
                    "target_id": target.identity,
                    "plan": CoordinationExecutionPlan(target.workers).sha256,
                    "expected": expected,
                }
            )
        self.layouts[size] = result
        return result


@dataclass
class SemanticMemo:
    entries: dict[tuple[Any, ...], dict[str, Any]] = field(default_factory=dict)


def catalog_identity(
    traces: list[TraceToken], evaluations: list[EvaluationToken]
) -> str:
    return commitment(
        {
            "traces": [asdict(t) for t in sorted(traces, key=lambda x: x.object_id)],
            "evaluations": [
                asdict(e) for e in sorted(evaluations, key=lambda x: x.object_id)
            ],
        }
    )


def declared_contract(
    session: dict[str, Any], option: dict[str, Any]
) -> dict[str, Any]:
    target = Target.from_mapping(option["target"])
    body = {
        "schema": "staged-comparison-contract.v1",
        "session": session,
        "target": target.to_mapping(),
        "metric": target.metric.to_mapping(),
        "allowed_plans": [option["plan"]],
        "slots": [
            {"candidate": b, "replicate": r, "trajectory_key": t, "evaluation_key": e}
            for (b, r), (t, e) in sorted(option["expected"].items())
        ],
        "report_rule": "maximize exact paired metric mean; lexicographic ties",
    }
    return {**body, "contract_sha256": commitment(body)}


def sql_work(
    layouts: list[dict[str, Any]],
    traces: list[TraceToken],
    evaluations: list[EvaluationToken],
) -> dict[str, list[dict[str, Any]]]:
    """Batch native SQL dependency DAG; no Python work-order calls."""
    result: dict[str, list[dict[str, Any]]] = {o["target_id"]: [] for o in layouts}
    with sqlite3.connect(":memory:") as db:
        db.executescript("""
        CREATE TABLE wanted (id TEXT,b TEXT,r INTEGER,t TEXT,e TEXT,p TEXT);
        CREATE TABLE traces (b TEXT,r INTEGER,t TEXT,p TEXT,ok INTEGER,id TEXT);
        CREATE TABLE evals (b TEXT,r INTEGER,t TEXT,e TEXT,p TEXT,ok INTEGER,id TEXT);
        CREATE INDEX ti ON traces(b,r,t,p,ok);
        CREATE INDEX ei ON evals(b,r,t,e,p,ok);
        """)
        db.executemany(
            "INSERT INTO wanted VALUES (?,?,?,?,?,?)",
            [
                (o["target_id"], *slot, *keys, o["plan"])
                for o in layouts
                for slot, keys in o["expected"].items()
            ],
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
        query = """SELECT w.id,w.b,w.r,w.t,w.e,
        (SELECT MIN(id) FROM evals e WHERE e.b=w.b AND e.r=w.r AND e.t=w.t
          AND e.e=w.e AND e.p=w.p AND e.ok=1),
        (SELECT MIN(id) FROM traces t WHERE t.b=w.b AND t.r=w.r AND t.t=w.t
          AND t.p=w.p AND t.ok=1)
        FROM wanted w ORDER BY w.id,w.b,w.r"""
        for identifier, b, r, t, e, eid, tid in db.execute(query):
            result[identifier].append(
                {
                    "candidate": b,
                    "replicate": r,
                    "trajectory_key": t,
                    "evaluation_key": e,
                    "action": "reuse" if eid else "evaluate" if tid else "execute",
                    "object_id": eid or tid,
                }
            )
    return result


def build_work(
    layouts: list[dict[str, Any]],
    traces: list[TraceToken],
    evaluations: list[EvaluationToken],
    use_sql: bool,
) -> list[dict[str, Any]]:
    independent = sql_work(layouts, traces, evaluations) if use_sql else {}
    result = []
    for layout in layouts:
        order = (
            independent[layout["target_id"]]
            if use_sql
            else work_order(layout["expected"], (layout["plan"],), traces, evaluations)
        )
        result.append(
            {
                **layout,
                "work_order": order,
                "new_pairs": sum(a["action"] == "execute" for a in order),
                "reevaluations": sum(a["action"] == "evaluate" for a in order),
                "reused_evaluations": sum(a["action"] == "reuse" for a in order),
            }
        )
    return result


def ranked(
    options: list[dict[str, Any]], request: dict[str, Any]
) -> list[dict[str, Any]]:
    result = []
    for option in options:
        target = option["target"]
        if option["new_pairs"] > target["allowance"] or any(
            target[c["field"]] != c["value"] for c in request["hard"]
        ):
            continue
        waived = sorted(f for f in FIELDS if target[f] != request["proposed"][f])
        result.append({**option, "waived": waived})
    return sorted(
        result,
        key=lambda o: (
            len(o["waived"]),
            o["new_pairs"],
            o["reevaluations"],
            o["target_id"],
        ),
    )


def validate_request(request: dict[str, Any], session: dict[str, Any]) -> None:
    if (
        set(request)
        != {"schema", "session", "episode", "proposed", "hard", "request_sha256"}
        or request["schema"] != "revision-request.v1"
        or request["session"] != session
    ):
        raise ValueError("request schema or scientific session differs")
    if (
        commitment({k: v for k, v in request.items() if k != "request_sha256"})
        != request["request_sha256"]
    ):
        raise ValueError("request commitment differs")
    Target.from_mapping(request["proposed"])
    seen = set()
    for clause in request["hard"]:
        if (
            set(clause) != {"field", "value"}
            or clause["field"] not in FIELDS
            or clause["field"] in seen
        ):
            raise ValueError("unknown or duplicate hard requirement")
        seen.add(clause["field"])
        Target.from_mapping({**request["proposed"], clause["field"]: clause["value"]})


def sql_select(
    options: list[dict[str, Any]],
    request: dict[str, Any],
    policies: list[CompiledPolicy],
) -> dict[str, Any] | None:
    with sqlite3.connect(":memory:") as db:
        db.execute(
            "CREATE TABLE options (id TEXT,epoch INTEGER,coupling TEXT,"
            "quantity TEXT,first_epoch INTEGER,workers INTEGER,allowance INTEGER,"
            "new_pairs INTEGER,reevaluations INTEGER)"
        )
        db.executemany(
            "INSERT INTO options VALUES (?,?,?,?,?,?,?,?,?)",
            [
                (
                    o["target_id"],
                    *(o["target"][f] for f in FIELDS),
                    o["new_pairs"],
                    o["reevaluations"],
                )
                for o in options
            ],
        )
        params: list[Any] = []
        conditions = ["new_pairs<=allowance"]
        for clause in request["hard"]:
            conditions.append('"' + clause["field"] + '"=?')
            params.append(clause["value"])
        for policy in policies:
            conditions.append(sql_expression(policy.expression, params))
        discrepancy = "+".join('("' + f + '"<>?)' for f in FIELDS)
        params.extend(request["proposed"][f] for f in FIELDS)
        row = db.execute(
            "SELECT id FROM options WHERE "
            + " AND ".join(conditions)
            + " ORDER BY "
            + discrepancy
            + ",new_pairs,reevaluations,id LIMIT 1",
            params,
        ).fetchone()
    if row is None:
        return None
    chosen = next(o for o in options if o["target_id"] == row[0])
    return {
        **chosen,
        "waived": sorted(
            f for f in FIELDS if chosen["target"][f] != request["proposed"][f]
        ),
    }


def owner_search(
    options: list[dict[str, Any]],
    request_hash: str,
    catalog_hash: str,
    policies: list[CompiledPolicy],
    method: str,
    memo: SemanticMemo,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    counts: dict[str, Any] = {
        "search_predicate_calls": 0,
        "projection_calls": 0,
        "visited_candidates": 0,
        "skipped_candidates": 0,
        "negative_hits": 0,
        "positive_hits": 0,
        "new_witnesses": [],
        "calls_by_owner": {p.owner: 0 for p in policies},
    }
    for option in options:
        counts["visited_candidates"] += 1
        row = values(option)
        if method == "exhaustive":
            passed = True
            for policy in policies:
                counts["search_predicate_calls"] += 1
                counts["calls_by_owner"][policy.owner] += 1
                if not policy.function(row):
                    passed = False
                    break
            if passed:
                return option, counts
            continue
        keys = []
        for policy in policies:
            key = policy.projection(row)
            counts["projection_calls"] += 1
            if method == "context_local":
                key = (request_hash, catalog_hash, *key)
            keys.append(key)
        rejected = next(
            (
                memo.entries[k]
                for k in keys
                if k in memo.entries and not memo.entries[k]["accepted"]
            ),
            None,
        )
        if rejected is not None:
            counts["negative_hits"] += 1
            counts["skipped_candidates"] += 1
            continue
        passed = True
        for policy, key in zip(policies, keys, strict=True):
            if method == "generic_memo" and key in memo.entries:
                counts["positive_hits"] += 1
                accepted = bool(memo.entries[key]["accepted"])
            else:
                counts["search_predicate_calls"] += 1
                counts["calls_by_owner"][policy.owner] += 1
                accepted = policy.function(row)
                if not accepted or method == "generic_memo":
                    witness = {
                        "owner": policy.owner,
                        "policy_identity": policy.identity,
                        "projection": list(key),
                        "accepted": accepted,
                        "first_target": option["target"],
                        "first_work": {
                            "new_pairs": option["new_pairs"],
                            "reevaluations": option["reevaluations"],
                        },
                        "first_request": request_hash,
                        "first_catalog": catalog_hash,
                    }
                    memo.entries[key] = witness
                    counts["new_witnesses"].append(witness)
            if not accepted:
                passed = False
                break
        if passed:
            return option, counts
    return None, counts


def plan(
    session: dict[str, Any],
    size: int,
    request: dict[str, Any],
    raw_policies: list[dict[str, Any]],
    traces: list[TraceToken],
    evaluations: list[EvaluationToken],
    method: str,
    dependencies: DependencyCache,
    memo: SemanticMemo,
) -> dict[str, Any]:
    """Time the real route; independent parity audits are outside all routes."""
    begin_wall, begin_cpu = perf_counter_ns(), process_time_ns()
    phases: dict[str, dict[str, int]] = {}
    last_wall, last_cpu = begin_wall, begin_cpu

    def end_phase(name: str) -> None:
        nonlocal last_wall, last_cpu
        wall, cpu = perf_counter_ns(), process_time_ns()
        phases[name] = {"wall_ns": wall - last_wall, "cpu_ns": cpu - last_cpu}
        last_wall, last_cpu = wall, cpu

    if method not in METHODS:
        raise ValueError("unknown planner")
    validate_request(request, session)
    compiled = compile_policies(raw_policies)
    catalog_hash = catalog_identity(traces, evaluations)
    end_phase("validation_and_policy_compile")
    before_t, before_e = (
        dependencies.compiled_trajectories,
        dependencies.compiled_evaluations,
    )
    layouts = dependencies.build(session, size)
    cold_t, cold_e = (
        dependencies.compiled_trajectories - before_t,
        dependencies.compiled_evaluations - before_e,
    )
    end_phase("immutable_dependencies")
    options = build_work(layouts, traces, evaluations, method == "complete_sql")
    end_phase("current_typed_work")
    ordered = ranked(options, request) if method != "complete_sql" else []
    end_phase("current_hard_filter_and_rank")
    if method == "complete_sql":
        chosen = sql_select(options, request, compiled)
        counts: dict[str, Any] = {
            "search_predicate_calls": None,
            "projection_calls": 0,
            "visited_candidates": None,
            "skipped_candidates": None,
            "negative_hits": 0,
            "positive_hits": 0,
            "new_witnesses": [],
            "calls_by_owner": None,
        }
    else:
        chosen, counts = owner_search(
            ordered, request["request_sha256"], catalog_hash, compiled, method, memo
        )
    end_phase("owner_search")
    if chosen is None:
        selected, consent = None, []
    else:
        selected = {k: v for k, v in chosen.items() if k not in {"expected", "plan"}}
        selected["contract"] = declared_contract(session, chosen)
        consent = approvals(request, catalog_hash, selected, compiled)
        if not authorized(request, catalog_hash, selected, compiled, consent):
            raise RuntimeError("selected target is not currently approved")
    end_phase("contract_and_exact_approval")
    result = {
        "method": method,
        "selected": selected,
        "approvals": consent,
        "catalog_sha256": catalog_hash,
        "policies": raw_policies,
        "counts": {
            **counts,
            "selection_predicate_calls": 6 if selected else 0,
            "cold_trajectory_compilations": cold_t,
            "cold_evaluation_compilations": cold_e,
            "memo_entries": len(memo.entries),
        },
    }
    serialized = json.dumps(
        result, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    end_phase("result_serialization")
    result["timing"] = {
        "phases": phases,
        "wall_ns": perf_counter_ns() - begin_wall,
        "cpu_ns": process_time_ns() - begin_cpu,
        "serialized_bytes": len(serialized),
    }
    return result
