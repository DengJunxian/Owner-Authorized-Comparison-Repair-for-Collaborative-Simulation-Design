"""The outcome-independent scientific blocks and cooperative change requests."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from policytwin.coordination.plan import BRANCHES, CoordinationExecutionPlan, RunSpec
from policytwin.coordination.schema import commitment
from policytwin.cscwd_mechanism.coupling import CoupledRecipe
from policytwin.design_contract.runtime import SimulationRecipe
from policytwin.typed_revision.contract import trajectory_key
from policytwin.typed_revision.metrics import MetricSpec, evaluation_key

STUDY: dict[str, Any] = {
    "label": "post-registration typed revision and incomplete-store study v1",
    "seed_root": "cscwd-typed-revision-20261027-v1",
    "agents": [6, 24],
    "regimes": ["deep", "responsive"],
    "sessions_per_cell": 2,
    "replicates": 4,
    "epochs": 4,
    "initial_branch_epoch": 1,
    "revised_branch_epoch": 2,
    "coupling": "shared",
    "candidates": list(BRANCHES),
    "plan": CoordinationExecutionPlan(2).to_mapping(),
    "initial_metric": MetricSpec("notional").to_mapping(),
    "requests": ["metric_quantity", "metric_window", "combined_revision"],
    "available_original_pairs": [0, 6, 12],
    "availability_rule": (
        "lowest sha256(session, candidate, replicate, availability-v1)"
    ),
    "target_rule": "session index modulo candidate count",
    "execution_budget_profile": 6,
    "budget_interpretation": "work-order feasibility, not stopped or timed service",
    "methods": ["opaque", "typed"],
    "positive_control": "equal-obligation relational typed planner",
    "ablation": "omit evaluation binding; keep trajectory, coverage and permission",
    "decision": "maximize exact mean paired metric; lexicographic ties",
    "metrics": [
        "actual_new_pairs",
        "reevaluations",
        "reused_evaluations",
        "exact_fresh_metric_agreement",
        "exact_fresh_trace_and_draw_agreement",
        "invalid_release",
        "top1_disagreement",
        "top2_disagreement",
        "finite_panel_regret",
        "request_response_bytes",
        "request_rounds",
        "budget_feasibility",
        "idempotent_repeat_request",
    ],
    "threshold": "exact integers, rational means and complete trace equality",
    "analysis": "finite descriptive counts; sessions are blocks, not view repetitions",
    "failure_policy": "retain every failed/incomplete attempt; no retry or new seed",
    "expected_claims": [
        "typed dependency separation preserves specified fresh reports",
        "pure metric changes reuse complete valid traces without simulation",
        "mixed revisions invalidate their trajectory dependencies",
        "partial stores require real new execution for unsupported trajectories",
        "stale derived evaluations can change revised finite-panel decisions",
        "equally specified relational implementation can implement the same rule",
    ],
}


def sessions() -> list[dict[str, Any]]:
    result = []
    for agents in STUDY["agents"]:
        for regime in STUDY["regimes"]:
            for batch in range(STUDY["sessions_per_cell"]):
                index = len(result)
                result.append(
                    {
                        "id": f"typed-a{agents}-{regime}-s{batch:02d}",
                        "index": index,
                        "agents": agents,
                        "regime": regime,
                        "batch": batch,
                        "liquidity": 5000 if regime == "deep" else 6 * agents,
                        "target": BRANCHES[index % len(BRANCHES)],
                    }
                )
    return result


def recipes(
    session: dict[str, Any], request: str
) -> dict[tuple[str, int], CoupledRecipe]:
    if request not in {"initial", *STUDY["requests"]}:
        raise ValueError("unknown revision request")
    result = {}
    for candidate in BRANCHES:
        for replicate in range(STUDY["replicates"]):
            spec = RunSpec(
                "cscwd-typed-" + session["regime"],
                commitment([STUDY["seed_root"], session["id"], replicate]),
                replicate,
                session["agents"],
                candidate,
                STUDY["epochs"],
                STUDY["initial_branch_epoch"],
            )
            if request == "combined_revision" and candidate == session["target"]:
                spec = replace(spec, branch_epoch=STUDY["revised_branch_epoch"])
            result[candidate, replicate] = CoupledRecipe(
                SimulationRecipe(spec, session["liquidity"]), STUDY["coupling"]
            )
    return result


def requested_metric(request: str) -> MetricSpec:
    if request == "initial":
        return MetricSpec("notional")
    if request == "metric_quantity":
        return MetricSpec("volume")
    if request in {"metric_window", "combined_revision"}:
        return MetricSpec("volume", 2)
    raise ValueError("unknown request")


def request_contract(session: dict[str, Any], request: str) -> dict[str, Any]:
    metric = requested_metric(request)
    slots = []
    for (candidate, replicate), recipe in recipes(session, request).items():
        science = trajectory_key(recipe)
        slots.append(
            {
                "candidate": candidate,
                "replicate": replicate,
                "trajectory_key": science,
                "evaluation_key": evaluation_key(science, metric),
                "coupled_recipe": recipe.to_mapping(),
            }
        )
    body = {
        "schema": "typed-comparison-contract.v1",
        "session": session,
        "request": request,
        "designer_change": "branch epoch 1 to 2 for " + session["target"]
        if request == "combined_revision"
        else "retain candidate recipes",
        "evaluator_change": metric.to_mapping(),
        "allowed_plans": [CoordinationExecutionPlan.from_mapping(STUDY["plan"]).sha256],
        "report_rule": STUDY["decision"],
        "slots": slots,
    }
    return {**body, "contract_sha256": commitment(body)}


def available_slots(session: dict[str, Any], count: int) -> list[list[Any]]:
    if count not in STUDY["available_original_pairs"]:
        raise ValueError("unsupported availability view")
    ordered = sorted(
        recipes(session, "initial"),
        key=lambda s: commitment([session["id"], *s, "availability-v1"]),
    )
    return [list(s) for s in ordered[:count]]


def schedule() -> list[dict[str, Any]]:
    return [
        {
            "session": session,
            "requests": [request_contract(session, r) for r in STUDY["requests"]],
            "availability": [
                {"count": count, "slots": available_slots(session, count)}
                for count in STUDY["available_original_pairs"]
            ],
        }
        for session in sessions()
    ]
