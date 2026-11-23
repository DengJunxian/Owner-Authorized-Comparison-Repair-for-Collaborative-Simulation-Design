"""Real file-backed counteroffers, approvals, acquisition and fresh comparison."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from policytwin.coordination.plan import CoordinationExecutionPlan
from policytwin.coordination.schema import commitment
from policytwin.cscwd_mechanism.coupling import CoupledRecipe, execute
from policytwin.negotiated_repair.consent import approvals, authorized, proposal
from policytwin.negotiated_repair.design import (
    METHODS,
    STUDY,
    Target,
    available,
    contract,
    metric,
    recipes,
)
from policytwin.negotiated_repair.solver import (
    alternatives,
    conflict_cores,
    frontier,
    minimal_options,
    selected,
)
from policytwin.typed_revision.contract import EvaluationToken, TraceToken, decide
from policytwin.typed_revision.runtime import (
    acquired_trace,
    checked_evaluation,
    expectations,
    read,
    trace_matches,
)
from scripts.cscwd2027_design_contract.store import save


def new_pair(
    identifier: str, recipe: CoupledRecipe, workers: int, root: Path, seal: str
) -> tuple[TraceToken, dict[str, Any]]:
    row = {
        "id": identifier,
        "session": recipe.recipe.spec.study,
        "candidate": recipe.recipe.spec.branch,
        "replicate": recipe.recipe.spec.replicate,
        "coupled_recipe": recipe.to_mapping(),
        "plan": CoordinationExecutionPlan(workers).to_mapping(),
    }
    execute(row, root, seal)
    return acquired_trace(identifier, root)


def checked_or_rejected(
    trace: TraceToken,
    capsule: dict[str, Any],
    target: Target,
    path: Path,
    expected_eval: str,
) -> EvaluationToken:
    if trace.local_pass:
        return checked_evaluation(trace, capsule, metric(target), path)
    return EvaluationToken(
        "rejected-" + trace.object_id,
        *trace.slot,
        trace.trajectory_key,
        expected_eval,
        trace.capsule_sha256,
        trace.plan_sha256,
        "",
        False,
        None,
    )


def fresh_target(
    session: dict[str, Any], target: Target, root: Path, seal: str, number: int
) -> dict[str, Any]:
    """Reference trajectories are shared across pure metrics, never acquisitions."""
    # Allowance and metric do not alter trajectories. The plan does alter authority.
    base = Target(target.epoch, target.coupling, "notional", target.workers, 12)
    trace_tokens = []
    values: dict[str, list[dict[str, Any]]] = {"notional": [], "late_volume": []}
    for slot, recipe in recipes(session, base).items():
        identifier = f"{session['id']}-reference-{number}-{slot[0]}-r{slot[1]}"
        trace, capsule = new_pair(identifier, recipe, target.workers, root, seal)
        trace_tokens.append(asdict(trace))
        for name in values:
            measured = Target(target.epoch, target.coupling, name, target.workers, 12)
            declared = contract(session, measured)
            evaluation = checked_or_rejected(
                trace,
                capsule,
                measured,
                root / "reference_evaluations" / f"{identifier}-{name}.json",
                expectations(declared)[slot][1],
            )
            values[name].append(asdict(evaluation))
    result = {
        "trajectory_target": base.to_mapping(),
        "traces": trace_tokens,
        "evaluations": values,
    }
    save(root / "references" / f"{session['id']}-{number}.json", result)
    return result


def repair(
    request: dict[str, Any],
    method: str,
    option: dict[str, Any] | None,
    catalog: dict[tuple[str, int], tuple[TraceToken, EvaluationToken]],
    root: Path,
    seal: str,
    reference: dict[str, Any] | None,
) -> dict[str, Any]:
    identifier = request["session"]["id"] + "-" + request["kind"] + "-" + method
    folder = root / "handoffs" / identifier
    folder.mkdir(parents=True, exist_ok=False)
    save(folder / "request.json", request)
    if option is None:
        result = {
            "id": identifier,
            "session": request["session"]["id"],
            "kind": request["kind"],
            "method": method,
            "selected": None,
            "status": "abstain_unresolved",
            "new_pairs": 0,
            "reevaluations": 0,
            "reused_evaluations": 0,
            "authorized": False,
            "exact": None,
            "compared_slots": 0,
            "matching_slots": 0,
            "authority_mutations": [],
        }
        save(folder / "result.json", result)
        return result
    target = Target.from_mapping(option["target"])
    declared = option["contract"]
    expected = expectations(declared)
    offer = proposal(request, option)
    consent = approvals(request, option)
    save(folder / "counteroffer.json", offer)
    save(folder / "owner_approvals.json", consent)
    save(folder / "accepted_contract.json", declared)
    save(
        folder / "work_order.json",
        {
            "proposal_sha256": offer["proposal_sha256"],
            "recorded_before_acquisition_utc": datetime.now(UTC).isoformat(),
            "actions": option["work_order"],
        },
    )
    if not authorized(request, option, read(folder / "owner_approvals.json")):
        raise RuntimeError("counteroffer is not approved before acquisition")
    members = []
    traces = []
    desired = recipes(request["session"], target)
    for action in option["work_order"]:
        slot = action["candidate"], action["replicate"]
        if action["action"] == "execute":
            trace, capsule = new_pair(
                f"{identifier}-{slot[0]}-r{slot[1]}",
                desired[slot],
                target.workers,
                root,
                seal,
            )
        else:
            old_trace, old_eval = catalog[slot]
            if action["object_id"] not in (old_trace.object_id, old_eval.object_id):
                raise RuntimeError("selector attempted to read a different catalog")
            trace, capsule = acquired_trace(
                old_trace.object_id, root, folder / "replay" / f"{slot[0]}-r{slot[1]}"
            )
        # Reevaluating the metric is part of validation even when the stored
        # evaluation is reused. Work-type counts do not count function calls.
        token = checked_or_rejected(
            trace,
            capsule,
            target,
            folder / "evaluations" / f"{slot[0]}-r{slot[1]}.json",
            expected[slot][1],
        )
        if action["action"] == "reuse" and token != catalog[slot][1]:
            raise RuntimeError("reused evaluation differs from its checked parent")
        traces.append(trace)
        members.append(token)
    numerical = decide(
        declared["contract_sha256"], expected, tuple(declared["allowed_plans"]), members
    )
    released = numerical["authorized"] and authorized(request, option, consent)
    save(
        folder / "producer_response.json",
        {
            "traces": [asdict(t) for t in traces],
            "evaluations": [asdict(e) for e in members],
        },
    )
    save(
        folder / "report_decision.json",
        {
            "release_authorized": released,
            "proposal_sha256": offer["proposal_sha256"],
            "decision": numerical,
        },
    )
    exact = None
    matching = 0
    compared = 0
    if reference is not None:
        ref_traces = {
            tuple((t["candidate"], t["replicate"])): TraceToken(**t)
            for t in reference["traces"]
        }
        ref_evaluations = [
            EvaluationToken(**v) for v in reference["evaluations"][target.metric]
        ]
        ref_decision = decide(
            declared["contract_sha256"],
            expected,
            tuple(declared["allowed_plans"]),
            ref_evaluations,
        )
        if released and ref_decision["authorized"]:
            reference_values = {e.slot: e.value for e in ref_evaluations}
            for trace, evaluation in zip(traces, members, strict=True):
                ref = ref_traces[trace.slot]
                _, capsule = acquired_trace(trace.object_id, root)
                _, ref_capsule = acquired_trace(ref.object_id, root)
                compared += 1
                matching += int(
                    trace_matches(capsule, ref_capsule)
                    and evaluation.value == reference_values[trace.slot]
                )
            observed = numerical["report"]
            intended = ref_decision["report"]
            exact = matching == compared and all(
                observed[k] == intended[k] for k in ("means", "top1", "top2")
            )
    mutations = []
    if option["waived"]:
        for kind in STUDY["authority_mutations"]:
            changed = deepcopy(consent)
            if kind == "missing_owner":
                owner = option["waived"][0].split(".")[0]
                changed = [r for r in changed if r["owner"] != owner]
            elif kind == "stale_target":
                changed[0]["contract_sha256"] = "0" * 64
                changed[0]["approval_sha256"] = commitment(
                    {k: v for k, v in changed[0].items() if k != "approval_sha256"}
                )
            else:
                changed[0]["owned_waivers"] = ["unauthorized.waiver"]
                changed[0]["approval_sha256"] = commitment(
                    {k: v for k, v in changed[0].items() if k != "approval_sha256"}
                )
            safe_release = bool(
                numerical["authorized"] and authorized(request, option, changed)
            )
            mutations.append(
                {
                    "kind": kind,
                    "approvals": changed,
                    "full_release": safe_release,
                    "omit_consent_release": numerical["authorized"],
                    "wrong_authority_if_omitted": bool(
                        numerical["authorized"]
                        and not authorized(request, option, changed)
                    ),
                }
            )
    save(folder / "authority_ablation.json", mutations)
    actions = Counter(w["action"] for w in option["work_order"])
    result = {
        "id": identifier,
        "session": request["session"]["id"],
        "kind": request["kind"],
        "method": method,
        "selected": option,
        "status": "released" if released else "abstain_execution",
        "new_pairs": actions["execute"],
        "reevaluations": actions["evaluate"],
        "reused_evaluations": actions["reuse"],
        "authorized": bool(released),
        "exact": exact,
        "compared_slots": compared,
        "matching_slots": matching,
        "authority_mutations": mutations,
    }
    save(folder / "result.json", result)
    print(
        "NEGOTIATED_HANDOFF",
        identifier,
        actions["execute"],
        released,
        exact,
        flush=True,
    )
    return result


def execute_session(block: dict[str, Any], root: Path, seal: str) -> dict[str, Any]:
    session = block["session"]
    initial_target = Target(1, "shared", "notional", 2, 12)
    initial: dict[tuple[str, int], tuple[TraceToken, EvaluationToken]] = {}
    initial_contract = contract(session, initial_target)
    for slot, recipe in recipes(session, initial_target).items():
        identifier = f"{session['id']}-initial-{slot[0]}-r{slot[1]}"
        trace, capsule = new_pair(identifier, recipe, 2, root, seal)
        token = checked_or_rejected(
            trace,
            capsule,
            initial_target,
            root / "initial_evaluations" / f"{identifier}.json",
            expectations(initial_contract)[slot][1],
        )
        initial[slot] = trace, token
    plans = []
    for request in block["tickets"]:
        catalog = {s: initial[s] for s in available(session, request["availability"])}
        options = alternatives(
            request, [v[0] for v in catalog.values()], [v[1] for v in catalog.values()]
        )
        choices = {m: selected(options, request, m) for m in METHODS}
        sql_equal = choices["frontier"] == choices["csp_dag"]
        if not sql_equal:
            raise RuntimeError("generic complete CSP disagrees with frontier policy")
        evidence = {
            "request": request,
            "available_slots": [list(s) for s in catalog],
            "alternatives": options,
            "conflict_cores": conflict_cores(options, request["clauses"]),
            "minimum_concession_frontier": frontier(
                minimal_options(options, request["clauses"])
            ),
            "selected": choices,
            "sql_planner_equal": sql_equal,
            "counteroffers": {
                m: None if o is None else proposal(request, o)
                for m, o in choices.items()
            },
            "owner_approvals": {
                m: None if o is None else approvals(request, o)
                for m, o in choices.items()
            },
            "created_before_any_case_acquisition_utc": datetime.now(UTC).isoformat(),
        }
        save(root / "planning" / f"{session['id']}-{request['kind']}.json", evidence)
        plans.append((request, catalog, choices))
    # Every role's counteroffer is fixed before independent reference outcomes.
    reference_targets: dict[tuple[Any, ...], Target] = {}
    for _, _, choices in plans:
        for option in choices.values():
            if option is not None:
                target = Target.from_mapping(option["target"])
                reference_targets[target.epoch, target.coupling, target.workers] = (
                    target
                )
    references = {
        key: fresh_target(session, t, root, seal, i)
        for i, (key, t) in enumerate(sorted(reference_targets.items()))
    }
    outcomes = []
    for request, catalog, choices in plans:
        for method in sorted(
            METHODS, key=lambda m: commitment([session["id"], request["kind"], m])
        ):
            option = choices[method]
            key = (
                None
                if option is None
                else (
                    option["target"]["epoch"],
                    option["target"]["coupling"],
                    option["target"]["workers"],
                )
            )
            result = repair(
                request,
                method,
                option,
                catalog,
                root,
                seal,
                None if key is None else references[key],
            )
            outcomes.append(result["id"])
    return {
        "session": session["id"],
        "handoffs": outcomes,
        "reference_trajectory_panels": len(references),
    }
