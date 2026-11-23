"""Recompute every cooperative work/authority endpoint from retained records."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from policytwin.negotiated_repair.consent import authorized
from policytwin.negotiated_repair.design import METHODS, Target, available
from policytwin.negotiated_repair.solver import alternatives, conflict_cores, selected
from policytwin.typed_revision.contract import (
    EvaluationToken,
    TraceToken,
    decide,
    verify_derivation,
)
from policytwin.typed_revision.runtime import (
    acquired_trace,
    expectations,
    read,
    trace_matches,
)
from scripts.cscwd2027_design_contract.store import save


def summarize(root: Path, destination: Path | None = None) -> dict[str, Any]:
    out = root if destination is None else destination
    rows = []
    planning = []
    chronology_violations = []
    references = [read(p) for p in sorted((root / "references").glob("*.json"))]
    for path in sorted((root / "planning").glob("*.json")):
        entry = read(path)
        request = entry["request"]
        catalog_traces = []
        catalog_evaluations = []
        for branch, rep in available(request["session"], request["availability"]):
            identifier = f"{request['session']['id']}-initial-{branch}-r{rep}"
            trace, capsule = acquired_trace(identifier, root)
            receipt = root / "initial_evaluations" / f"{identifier}.json"
            if trace.local_pass:
                evaluation = verify_derivation(trace, capsule, read(receipt))
            else:
                # Preserve a failed initial trace, as the actual run did.
                evaluation = EvaluationToken(
                    "rejected-" + trace.object_id,
                    *trace.slot,
                    trace.trajectory_key,
                    "",
                    trace.capsule_sha256,
                    trace.plan_sha256,
                    "",
                    False,
                    None,
                )
            catalog_traces.append(trace)
            catalog_evaluations.append(evaluation)
        rebuilt_options = alternatives(request, catalog_traces, catalog_evaluations)
        # A rejected initial evaluation's unused key is not an eligibility input.
        if rebuilt_options != entry["alternatives"]:
            raise RuntimeError("catalog-derived target/work alternatives differ")
        cores = conflict_cores(entry["alternatives"], request["clauses"])
        if cores != entry["conflict_cores"]:
            raise RuntimeError("conflict certificates differ on recomputation")
        for method in METHODS:
            option = selected(entry["alternatives"], request, method)
            if option != entry["selected"][method]:
                raise RuntimeError("pre-outcome counteroffer differs")
            identifier = request["session"]["id"] + "-" + request["kind"] + "-" + method
            folder = root / "handoffs" / identifier
            result = read(folder / "result.json")
            if result["selected"] != option:
                raise RuntimeError("acquired target differs from agreed planner target")
            if option is not None:
                records = read(folder / "owner_approvals.json")
                if records != entry["owner_approvals"][method]:
                    raise RuntimeError(
                        "owner decision changed after reference outcomes"
                    )
                if not authorized(request, option, records):
                    raise RuntimeError("accepted counteroffer lacks owner approvals")
                actions = read(folder / "work_order.json")["actions"]
                if actions != option["work_order"]:
                    raise RuntimeError("actual work differs from approved work")
                response = read(folder / "producer_response.json")
                members = []
                observed_capsules = {}
                for raw_trace, raw_evaluation in zip(
                    response["traces"], response["evaluations"], strict=True
                ):
                    trace, capsule = acquired_trace(raw_trace["object_id"], root)
                    if trace != TraceToken(**raw_trace):
                        raise RuntimeError("response trace binding differs")
                    receipt = (
                        folder
                        / "evaluations"
                        / f"{trace.candidate}-r{trace.replicate}.json"
                    )
                    if trace.local_pass:
                        evaluated = verify_derivation(trace, capsule, read(receipt))
                        if evaluated != EvaluationToken(**raw_evaluation):
                            raise RuntimeError("response evaluation differs")
                    else:
                        evaluated = EvaluationToken(**raw_evaluation)
                    members.append(evaluated)
                    observed_capsules[trace.slot] = capsule
                declared = option["contract"]
                numerical = decide(
                    declared["contract_sha256"],
                    expectations(declared),
                    tuple(declared["allowed_plans"]),
                    members,
                )
                if result["authorized"] != numerical["authorized"]:
                    raise RuntimeError("recomputed numerical authorization differs")
                if numerical["authorized"]:
                    target = option["target"]
                    reference = next(
                        r
                        for r in references
                        if r["traces"][0]["object_id"].startswith(
                            request["session"]["id"] + "-reference-"
                        )
                        and all(
                            r["trajectory_target"][k] == target[k]
                            for k in ("epoch", "coupling", "workers")
                        )
                    )
                    ref_tokens = [
                        EvaluationToken(**v)
                        for v in reference["evaluations"][target["metric"]]
                    ]
                    reference_decision = decide(
                        declared["contract_sha256"],
                        expectations(declared),
                        tuple(declared["allowed_plans"]),
                        ref_tokens,
                    )
                    if reference_decision["authorized"]:
                        matching = 0
                        by_slot = {e.slot: e for e in members}
                        for raw in reference["traces"]:
                            ref_trace, ref_capsule = acquired_trace(
                                raw["object_id"], root
                            )
                            saved = (
                                root
                                / "reference_evaluations"
                                / f"{ref_trace.object_id}-{target['metric']}.json"
                            )
                            fresh_value = verify_derivation(
                                ref_trace, ref_capsule, read(saved)
                            )
                            matching += int(
                                trace_matches(
                                    observed_capsules[ref_trace.slot], ref_capsule
                                )
                                and by_slot[ref_trace.slot].value == fresh_value.value
                            )
                        exact = matching == len(members) and all(
                            numerical["report"][k] == reference_decision["report"][k]
                            for k in ("means", "top1", "top2")
                        )
                        if (
                            result["compared_slots"],
                            result["matching_slots"],
                            result["exact"],
                        ) != (len(members), matching, exact):
                            raise RuntimeError("complete fresh comparison differs")
                for mutation in result["authority_mutations"]:
                    check = authorized(request, option, mutation["approvals"])
                    if (
                        mutation["full_release"] != (numerical["authorized"] and check)
                        or mutation["omit_consent_release"] != numerical["authorized"]
                    ):
                        raise RuntimeError("authority ablation differs")
                for action in actions:
                    if action["action"] == "execute":
                        obj = (
                            f"{identifier}-{action['candidate']}-r{action['replicate']}"
                        )
                        attempt = root / "objects" / obj / "attempt.json"
                        order_time = datetime.fromisoformat(
                            read(folder / "work_order.json")[
                                "recorded_before_acquisition_utc"
                            ]
                        )
                        if (
                            not attempt.exists()
                            or datetime.fromisoformat(
                                read(attempt)["registration"]["registered_at_utc"]
                            )
                            < order_time
                        ):
                            chronology_violations.append(obj)
                actual = sum(
                    1 for p in (root / "objects").glob(identifier + "-*/result.json")
                )
                if actual != result["new_pairs"]:
                    raise RuntimeError("new-pair count differs from actual executions")
            rows.append(result)
        planning.append(
            {
                "session": request["session"]["id"],
                "kind": request["kind"],
                "cores": cores,
                "sql_planner_equal": entry["sql_planner_equal"],
                "alternatives": len(entry["alternatives"]),
                "frontier_size": len(entry["minimum_concession_frontier"]),
                "feasible_with_waivers": entry["selected"]["frontier"] is not None,
            }
        )
    attempts = [read(p) for p in sorted((root / "objects").glob("*/result.json"))]
    attempt_rows = [
        {
            "id": a["id"],
            "status": a["status"],
            "local_pass": a["local_pass"],
            "coupling": a["design"]["coupled_recipe"]["coupling"],
            "workers": a["design"]["plan"]["worker_count"],
            "metric": a.get("metric"),
            "error": a.get("error"),
        }
        for a in attempts
    ]
    methods = {}
    for method in METHODS:
        group = [r for r in rows if r["method"] == method]
        mutations = [m for r in group for m in r["authority_mutations"]]
        methods[method] = {
            "panels": len(group),
            "released": sum(r["authorized"] for r in group),
            "unresolved_abstentions": sum(
                r["status"] == "abstain_unresolved" for r in group
            ),
            "execution_abstentions": sum(
                r["status"] == "abstain_execution" for r in group
            ),
            "new_pairs": sum(r["new_pairs"] for r in group),
            "reevaluations": sum(r["reevaluations"] for r in group),
            "reused_evaluations": sum(r["reused_evaluations"] for r in group),
            "exact_reports": sum(r["exact"] is True for r in group),
            "assessable_reports": sum(r["exact"] is not None for r in group),
            "compared_slots": sum(r["compared_slots"] for r in group),
            "matching_slots": sum(r["matching_slots"] for r in group),
            "concessions": sum(
                len(r["selected"]["waived"]) for r in group if r["selected"]
            ),
            "concession_owners": dict(
                Counter(
                    c.split(".")[0]
                    for r in group
                    if r["selected"]
                    for c in r["selected"]["waived"]
                )
            ),
            "authority_mutations": len(mutations),
            "authority_mutations_rejected": sum(
                not m["full_release"] for m in mutations
            ),
            "omit_consent_invalid_release": sum(
                m["wrong_authority_if_omitted"] for m in mutations
            ),
        }
    paired = []
    for plan in planning:
        found = {
            r["method"]: r
            for r in rows
            if r["session"] == plan["session"] and r["kind"] == plan["kind"]
        }
        a, b = found["frontier"], found["serial"]
        paired.append(
            {
                "session": plan["session"],
                "kind": plan["kind"],
                "frontier_new_pairs": a["new_pairs"],
                "serial_new_pairs": b["new_pairs"],
                "new_pair_difference": a["new_pairs"] - b["new_pairs"],
                "same_target": (a["selected"] is None and b["selected"] is None)
                or (
                    a["selected"] is not None
                    and b["selected"] is not None
                    and Target.from_mapping(a["selected"]["target"])
                    == Target.from_mapping(b["selected"]["target"])
                ),
                "interpretation": "different approved targets are not numerical errors",
            }
        )
    summary = {
        "schema": "negotiated-repair-summary.v1",
        "scientific_sessions": 4,
        "tickets": len(planning),
        "method_panels": len(rows),
        "actual_paired_attempts": len(attempts),
        "local_passes": sum(a["local_pass"] for a in attempts),
        "failed_or_rejected_attempts": [
            a["id"] for a in attempts if not a["local_pass"]
        ],
        "chronology_violations": chronology_violations,
        "initial_pairs": sum("-initial-" in a["id"] for a in attempts),
        "reference_pairs": sum("-reference-" in a["id"] for a in attempts),
        "methods": methods,
        "sql_planner_matches": sum(p["sql_planner_equal"] for p in planning),
        "conflicting_tickets": sum(bool(p["cores"]) for p in planning),
        "minimal_conflict_cores": sum(len(p["cores"]) for p in planning),
        "hard_infeasible_tickets": sum(
            not p["feasible_with_waivers"] for p in planning
        ),
        "source_of_gain": (
            "minimum-concession choice between owner-approved alternatives, "
            "not faster execution of one unchanged target"
        ),
        "cross_platform_or_external_implementation": False,
        "human_negotiation_or_network_measurement": False,
    }
    if summary["initial_pairs"] + summary["reference_pairs"] + sum(
        m["new_pairs"] for m in methods.values()
    ) != len(attempts):
        raise RuntimeError("actual attempt partition does not close")
    save(out / "analysis/rows.json", rows)
    save(out / "analysis/planning.json", planning)
    save(out / "analysis/attempts.json", attempt_rows)
    save(out / "analysis/paired_work.json", paired)
    save(out / "summary.json", summary)
    return summary
