"""Actual file-backed acquisition and checked derivation through software roles."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from policytwin.coordination import release_gate
from policytwin.coordination.coordinator import draw_projection, scientific_projection
from policytwin.coordination.plan import CoordinationExecutionPlan
from policytwin.coordination.schema import commitment
from policytwin.cscwd_mechanism.coupling import (
    CoupledRecipe,
    coupling_context,
    execute,
    load,
)
from policytwin.design_contract.runtime import configured
from policytwin.provenance import canonical_json_bytes
from policytwin.typed_revision.contract import (
    EvaluationToken,
    TraceToken,
    decide,
    derive,
    relational_work_order,
    trajectory_key,
    verify_derivation,
    work_order,
)
from policytwin.typed_revision.design import STUDY, recipes, requested_metric
from policytwin.typed_revision.metrics import MetricSpec
from scripts.cscwd2027_design_contract.store import save


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def acquired_trace(
    identifier: str, root: Path, context: Path | None = None
) -> tuple[TraceToken, dict[str, Any]]:
    """Check physical commitments, and freshly replay any reused capsule."""
    evidence, capsule = load(identifier, root)
    result = read(root / "objects" / identifier / "result.json")
    recipe = CoupledRecipe.from_mapping(result["design"]["coupled_recipe"])
    if context is not None and evidence.local_pass:
        attempt = read(root / "objects" / identifier / "attempt.json")
        allocation = CoordinationExecutionPlan.from_mapping(result["design"]["plan"])
        registration = release_gate.Registration(**attempt["registration"])
        with configured(recipe.recipe), coupling_context(recipe.coupling):
            checked = release_gate.report(
                recipe.recipe.spec,
                allocation,
                capsule,
                registration,
                context,
                "full",
            )
        passed = checked["release_authorized"]
    else:
        passed = evidence.local_pass
    return TraceToken(
        identifier,
        evidence.actual_candidate,
        evidence.actual_replicate,
        trajectory_key(recipe),
        evidence.capsule_sha256,
        evidence.plan_sha256,
        bool(passed),
    ), capsule


def run_new_pair(
    identifier: str,
    recipe: CoupledRecipe,
    root: Path,
    seal: str,
) -> tuple[TraceToken, dict[str, Any], dict[str, Any]]:
    row = {
        "id": identifier,
        "session": recipe.recipe.spec.study,
        "candidate": recipe.recipe.spec.branch,
        "replicate": recipe.recipe.spec.replicate,
        "coupled_recipe": recipe.to_mapping(),
        "plan": STUDY["plan"],
    }
    result = execute(row, root, seal)
    trace, capsule = acquired_trace(identifier, root)
    return trace, capsule, result


def checked_evaluation(
    trace: TraceToken,
    capsule: dict[str, Any],
    metric: MetricSpec,
    path: Path,
) -> EvaluationToken:
    token, receipt = derive(trace, capsule, metric)
    save(path, receipt)
    verified = verify_derivation(trace, capsule, read(path))
    if verified != token:
        raise RuntimeError("persisted derivation differs")
    return verified


def expectations(contract: dict[str, Any]) -> dict[tuple[str, int], tuple[str, str]]:
    return {
        (s["candidate"], s["replicate"]): (s["trajectory_key"], s["evaluation_key"])
        for s in contract["slots"]
    }


def trace_matches(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return all(
        scientific_projection(a["worlds"][w]) == scientific_projection(b["worlds"][w])
        for w in ("baseline", "policy")
    ) and draw_projection(a) == draw_projection(b)


def receipt_bytes(path: Path) -> int:
    return path.stat().st_size


def execute_case(
    contract: dict[str, Any],
    view: dict[str, Any],
    method: str,
    initial: dict[tuple[str, int], tuple[TraceToken, EvaluationToken]],
    root: Path,
    seal: str,
) -> dict[str, Any]:
    """Persist role messages before execution and perform every assigned action."""
    identifier = (
        f"{contract['session']['id']}-{contract['request']}-a{view['count']}-{method}"
    )
    folder = root / "handoffs" / identifier
    folder.mkdir(parents=True, exist_ok=False)
    selected = [initial[tuple(slot)] for slot in view["slots"]]
    traces = [pair[0] for pair in selected]
    evaluations = [pair[1] for pair in selected]
    expected = expectations(contract)
    allowed = tuple(contract["allowed_plans"])
    order = work_order(expected, allowed, traces, evaluations, mode=method)
    relational = relational_work_order(expected, allowed, traces, evaluations)
    save(folder / "designer_request.json", contract)
    save(
        folder / "producer_work_order.json",
        {
            "contract_sha256": contract["contract_sha256"],
            "method": method,
            "available_slots": view["slots"],
            "actions": order,
            "timestamp_before_acquisition": datetime.now(UTC).isoformat(),
        },
    )
    metric = MetricSpec.from_mapping(contract["evaluator_change"])
    desired = recipes(contract["session"], contract["request"])
    incoming = decide(contract["contract_sha256"], expected, allowed, evaluations)
    responses, members, final_traces = [], [], []
    begin = perf_counter()
    for action in order:
        slot = action["candidate"], action["replicate"]
        if action["action"] == "reuse":
            token = next(e for e in evaluations if e.object_id == action["object_id"])
            trace = next(t for t in traces if t.capsule_sha256 == token.capsule_sha256)
            responses.append({"action": "reuse", "token": asdict(token)})
        else:
            if action["action"] == "execute":
                trace, capsule, execution = run_new_pair(
                    f"{identifier}-{slot[0]}-r{slot[1]}", desired[slot], root, seal
                )
            else:
                trace, capsule = acquired_trace(
                    action["object_id"],
                    root,
                    folder / "replay" / f"{slot[0]}-r{slot[1]}",
                )
                execution = None
            if trace.local_pass:
                evaluation_path = folder / "evaluations" / f"{slot[0]}-r{slot[1]}.json"
                token = checked_evaluation(trace, capsule, metric, evaluation_path)
            else:
                token = EvaluationToken(
                    "rejected-" + trace.object_id,
                    *slot,
                    trace.trajectory_key,
                    action["evaluation_key"],
                    trace.capsule_sha256,
                    trace.plan_sha256,
                    "",
                    False,
                    None,
                )
            responses.append(
                {
                    "action": action["action"],
                    "trace": asdict(trace),
                    "token": asdict(token),
                    "new_pair_status": None
                    if execution is None
                    else execution["status"],
                }
            )
        members.append(token)
        final_traces.append(trace)
    final = decide(contract["contract_sha256"], expected, allowed, members)
    final_order = work_order(expected, allowed, final_traces, members)
    unsafe_order = work_order(
        expected, allowed, traces, evaluations, mode="minus_evaluation_binding"
    )
    unsafe_members = []
    by_slot = {m.slot: m for m in members}
    for item in unsafe_order:
        if item["action"] == "reuse":
            unsafe_members.append(
                next(e for e in evaluations if e.object_id == item["object_id"])
            )
        else:
            unsafe_members.append(by_slot[item["candidate"], item["replicate"]])
    unsafe = decide(
        contract["contract_sha256"],
        expected,
        allowed,
        unsafe_members,
        omit_evaluation_binding=True,
    )
    safe_check_of_unsafe = decide(
        contract["contract_sha256"], expected, allowed, unsafe_members
    )
    response = {
        "contract_sha256": contract["contract_sha256"],
        "actions": responses,
    }
    save(folder / "producer_response.json", response)
    save(folder / "evaluator_decision.json", final)
    save(folder / "metric_binding_ablation.json", unsafe)
    actions = Counter(a["action"] for a in order)
    bytes_transferred = sum(
        (root / "objects" / t.object_id / "capsule.json.gz").stat().st_size
        for t in final_traces
        if t.local_pass
    )
    result = {
        "id": identifier,
        "session": contract["session"]["id"],
        "agents": contract["session"]["agents"],
        "regime": contract["session"]["regime"],
        "request": contract["request"],
        "availability": view["count"],
        "method": method,
        "work_order": order,
        "relational_agreement": relational == order if method == "typed" else None,
        "initial_release": incoming["authorized"],
        "final_decision": final,
        "trace_tokens": [asdict(t) for t in final_traces],
        "new_pairs": actions["execute"],
        "reevaluations": actions["evaluate"],
        "reused_evaluations": actions["reuse"],
        "budget_feasible": actions["execute"] <= STUDY["execution_budget_profile"],
        "repeat_request_all_reuse": all(a["action"] == "reuse" for a in final_order),
        "request_rounds": int(any(a["action"] != "reuse" for a in order)),
        "request_response_bytes": len(canonical_json_bytes(contract))
        + len(canonical_json_bytes(response))
        + bytes_transferred,
        "elapsed_seconds": perf_counter() - begin,
        "unsafe_decision": unsafe,
        "unsafe_invalid_release": unsafe["authorized"]
        and not safe_check_of_unsafe["authorized"],
        "safe_findings_on_ablation": safe_check_of_unsafe["findings"],
    }
    save(folder / "result.json", result)
    print(
        "TYPED_HANDOFF", identifier, actions["execute"], final["authorized"], flush=True
    )
    return result


def execute_session(block: dict[str, Any], root: Path, seal: str) -> dict[str, Any]:
    """Run initial evidence, fresh references and every partial-view handoff."""
    session = block["session"]
    initial = {}
    statuses = []
    for slot, recipe in recipes(session, "initial").items():
        trace, capsule, result = run_new_pair(
            f"{session['id']}-initial-{slot[0]}-r{slot[1]}", recipe, root, seal
        )
        statuses.append({"id": trace.object_id, "local_pass": trace.local_pass})
        if trace.local_pass:
            evaluation = checked_evaluation(
                trace,
                capsule,
                requested_metric("initial"),
                root / "initial_evaluations" / f"{trace.object_id}.json",
            )
        else:
            evaluation = EvaluationToken(
                "rejected-" + trace.object_id,
                *slot,
                trace.trajectory_key,
                "",
                trace.capsule_sha256,
                trace.plan_sha256,
                "",
                False,
                None,
            )
        initial[slot] = trace, evaluation
    fresh = []
    for contract in block["requests"]:
        expected = expectations(contract)
        metric = MetricSpec.from_mapping(contract["evaluator_change"])
        members, traces = [], []
        for slot, recipe in recipes(session, contract["request"]).items():
            trace, capsule, result = run_new_pair(
                f"{session['id']}-{contract['request']}-fresh-{slot[0]}-r{slot[1]}",
                recipe,
                root,
                seal,
            )
            statuses.append({"id": trace.object_id, "local_pass": trace.local_pass})
            traces.append(asdict(trace))
            if trace.local_pass:
                members.append(
                    checked_evaluation(
                        trace,
                        capsule,
                        metric,
                        root / "fresh_evaluations" / f"{trace.object_id}.json",
                    )
                )
        decision = decide(
            contract["contract_sha256"],
            expected,
            tuple(contract["allowed_plans"]),
            members,
        )
        fresh.append(
            {
                "session": session["id"],
                "request": contract["request"],
                "decision": decision,
                "traces": traces,
                "actual_pairs": len(traces),
            }
        )
    save(root / "fresh_panels" / f"{session['id']}.json", fresh)
    completed = []
    for contract in block["requests"]:
        for view in block["availability"]:
            # Assignment is counterbalanced by identity, not outcome or timing.
            methods = sorted(
                STUDY["methods"],
                key=lambda m: commitment([session["id"], contract["request"], view, m]),
            )
            for method in methods:
                result = execute_case(contract, view, method, initial, root, seal)
                completed.append(result["id"])
    return {
        "session": session["id"],
        "initial_and_fresh": statuses,
        "handoffs": completed,
    }
