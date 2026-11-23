"""Complete retained-record recomputation of the refusal-capable owner study."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from policytwin.coordination.schema import commitment
from policytwin.negotiated_repair.solver import alternatives
from policytwin.owner_policy.protocol import authorized, negotiate
from policytwin.owner_policy.runtime import vote_mutations
from policytwin.owner_policy.study import METHODS
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
    rows, planning, violations = [], [], []
    references = [read(p) for p in sorted((root / "references").glob("*.json"))]
    for path in sorted((root / "planning").glob("*.json")):
        entry = read(path)
        request, policies = entry["request"], entry["policies"]
        catalog = []
        for item in entry["catalog"]:
            trace, capsule = acquired_trace(item["trace"]["object_id"], root)
            if trace != TraceToken(**item["trace"]):
                raise RuntimeError("initial catalog trace differs")
            if trace.local_pass:
                evaluation = verify_derivation(
                    trace,
                    capsule,
                    read(root / "initial_evaluations" / f"{trace.object_id}.json"),
                )
                if evaluation != EvaluationToken(**item["evaluation"]):
                    raise RuntimeError("initial evaluation differs")
            else:
                evaluation = EvaluationToken(**item["evaluation"])
            catalog.append({"trace": asdict(trace), "evaluation": asdict(evaluation)})
        if commitment(catalog) != entry["catalog_sha256"]:
            raise RuntimeError("catalog commitment differs")
        options = alternatives(
            request,
            [TraceToken(**r["trace"]) for r in catalog],
            [EvaluationToken(**r["evaluation"]) for r in catalog],
        )
        if options != entry["alternatives"]:
            raise RuntimeError("available target/work alternatives differ")
        decisions = {
            m: negotiate(options, request, policies, entry["catalog_sha256"], m)
            for m in METHODS
        }
        if decisions != entry["decisions"]:
            raise RuntimeError("owner search differs on recomputation")
        sql_equal = (
            decisions["owner_search"]["selected"] == decisions["policy_csp"]["selected"]
        )
        for method in METHODS:
            decision = decisions[method]
            option = decision["selected"]
            identifier = f"{request['session']['id']}-{request['kind']}-{method}"
            owner_folder = root / "owner_handoffs" / identifier
            row = read(owner_folder / "result.json")
            recomputed_counts = {
                "proposal_count": len(decision["offered"]),
                "rejected_proposals": sum(
                    not offer["accepted"] for offer in decision["offered"]
                ),
                "owner_calls": sum(
                    len(offer["votes"]) for offer in decision["offered"]
                ),
                "waivers_count": len(option["waived"]) if option else 0,
                "new_pairs": sum(w["action"] == "execute" for w in option["work_order"])
                if option
                else 0,
                "reevaluations": sum(
                    w["action"] == "evaluate" for w in option["work_order"]
                )
                if option
                else 0,
                "reused_evaluations": sum(
                    w["action"] == "reuse" for w in option["work_order"]
                )
                if option
                else 0,
            }
            if any(row[k] != value for k, value in recomputed_counts.items()):
                raise RuntimeError("recorded proposal/work/waiver counts differ")
            if (row["id"], row["session"], row["case"], row["method"]) != (
                identifier,
                request["session"]["id"],
                request["kind"],
                method,
            ):
                raise RuntimeError("result coordinate binding differs")
            if (
                row["selected"] != option
                or read(owner_folder / "negotiation.json") != decision
            ):
                raise RuntimeError("executed selection differs")
            actual = len(list((root / "objects").glob(identifier + "-*/result.json")))
            if actual != row["new_pairs"]:
                raise RuntimeError("actual new-pair count differs")
            if option is None:
                if row["authorized"] or actual or row["exact"] is not None:
                    raise RuntimeError("unagreed request acquired or released")
                if row["status"] != "no_agreement":
                    raise RuntimeError("unagreed request status differs")
            else:
                records = read(owner_folder / "owner_votes.json")
                if records != decision["offered"][-1]["votes"] or not authorized(
                    request, option, policies, entry["catalog_sha256"], records
                ):
                    raise RuntimeError("policy agreement differs")
                folder = root / "handoffs" / identifier
                response = read(folder / "producer_response.json")
                if read(folder / "work_order.json")["actions"] != option["work_order"]:
                    raise RuntimeError("acquired work order differs")
                members, capsules = [], {}
                for raw, ev in zip(
                    response["traces"], response["evaluations"], strict=True
                ):
                    trace, capsule = acquired_trace(raw["object_id"], root)
                    if trace != TraceToken(**raw):
                        raise RuntimeError("response trace differs")
                    evaluation = EvaluationToken(**ev)
                    if trace.local_pass:
                        rebuilt = verify_derivation(
                            trace,
                            capsule,
                            read(
                                folder
                                / "evaluations"
                                / f"{trace.candidate}-r{trace.replicate}.json"
                            ),
                        )
                        if rebuilt != evaluation:
                            raise RuntimeError("response derivation differs")
                    members.append(evaluation)
                    capsules[trace.slot] = capsule
                declared = option["contract"]
                numerical = decide(
                    declared["contract_sha256"],
                    expectations(declared),
                    tuple(declared["allowed_plans"]),
                    members,
                )
                if row["authorized"] != numerical["authorized"]:
                    raise RuntimeError("numerical release differs")
                expected_status = (
                    "released" if numerical["authorized"] else "execution_abstention"
                )
                if row["status"] != expected_status:
                    raise RuntimeError("execution status differs")
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
                ref_tokens = []
                matching = 0
                by_slot = {e.slot: e for e in members}
                for raw in reference["traces"]:
                    trace, capsule = acquired_trace(raw["object_id"], root)
                    if trace.local_pass:
                        token = verify_derivation(
                            trace,
                            capsule,
                            read(
                                root
                                / "reference_evaluations"
                                / f"{trace.object_id}-{target['metric']}.json"
                            ),
                        )
                    else:
                        token = next(
                            EvaluationToken(**e)
                            for e in reference["evaluations"][target["metric"]]
                            if (e["candidate"], e["replicate"]) == trace.slot
                        )
                    ref_tokens.append(token)
                    if trace.local_pass and by_slot[trace.slot].local_pass:
                        matching += int(
                            trace_matches(capsules[trace.slot], capsule)
                            and by_slot[trace.slot].value == token.value
                        )
                ref_decision = decide(
                    declared["contract_sha256"],
                    expectations(declared),
                    tuple(declared["allowed_plans"]),
                    ref_tokens,
                )
                if numerical["authorized"] and ref_decision["authorized"]:
                    exact = matching == len(members) and all(
                        numerical["report"][k] == ref_decision["report"][k]
                        for k in ("means", "top1", "top2")
                    )
                    if (row["exact"], row["compared_slots"], row["matching_slots"]) != (
                        exact,
                        len(members),
                        matching,
                    ):
                        raise RuntimeError(
                            "fresh scientific/draw/metric comparison differs"
                        )
                expected_mutations = vote_mutations(
                    request,
                    option,
                    policies,
                    entry["catalog_sha256"],
                    records,
                    bool(numerical["authorized"]),
                )
                if row["policy_invalid_release_mutations"] != expected_mutations:
                    raise RuntimeError("policy mutation outcome differs")
                if (
                    read(owner_folder / "policy_vote_mutations.json")
                    != expected_mutations
                ):
                    raise RuntimeError("retained policy mutation records differ")
                for action in option["work_order"]:
                    if action["action"] == "execute":
                        obj = (
                            f"{identifier}-{action['candidate']}-r{action['replicate']}"
                        )
                        stamp = read(root / "objects" / obj / "attempt.json")[
                            "registration"
                        ]["registered_at_utc"]
                        if datetime.fromisoformat(stamp) < datetime.fromisoformat(
                            entry["recorded_before_reference_utc"]
                        ):
                            violations.append(obj)
            row["false_abstention"] = (
                option is None and decisions["owner_search"]["selected"] is not None
            )
            rows.append(row)
        plan_time = datetime.fromisoformat(entry["recorded_before_reference_utc"])
        for p in (root / "objects").glob(
            request["session"]["id"] + "-reference-*/attempt.json"
        ):
            if (
                datetime.fromisoformat(read(p)["registration"]["registered_at_utc"])
                < plan_time
            ):
                violations.append(str(p.relative_to(root)))
        planning.append(
            {
                "session": request["session"]["id"],
                "case": request["kind"],
                "sql_equal": sql_equal,
                "agreement_exists": decisions["owner_search"]["selected"] is not None,
            }
        )
    attempts = [read(p) for p in sorted((root / "objects").glob("*/result.json"))]
    methods = {}
    for method in METHODS:
        group = [r for r in rows if r["method"] == method]
        mutations = [m for r in group for m in r["policy_invalid_release_mutations"]]
        methods[method] = {
            "requests": len(group),
            "released": sum(r["authorized"] for r in group),
            "no_agreement": sum(r["status"] == "no_agreement" for r in group),
            "execution_abstentions": sum(
                r["status"] == "execution_abstention" for r in group
            ),
            "false_abstentions": sum(r["false_abstention"] for r in group),
            **{
                k: sum(r[k] for r in group)
                for k in (
                    "new_pairs",
                    "reevaluations",
                    "reused_evaluations",
                    "compared_slots",
                    "matching_slots",
                    "waivers_count",
                    "proposal_count",
                    "rejected_proposals",
                    "owner_calls",
                )
            },
            "exact_reports": sum(r["exact"] is True for r in group),
            "assessable_reports": sum(r["exact"] is not None for r in group),
            "policy_mutations": len(mutations),
            "policy_mutations_rejected": sum(not m["full_release"] for m in mutations),
            "omit_policy_invalid_release": sum(
                m["omit_policy_release"] for m in mutations
            ),
        }
    summary = {
        "schema": "owner-policy-summary.v1",
        "scientific_sessions": 4,
        "tickets": len(planning),
        "method_panels": len(rows),
        "actual_paired_attempts": len(attempts),
        "local_passes": sum(a["local_pass"] for a in attempts),
        "failed_or_rejected_attempts": [
            a["id"] for a in attempts if not a["local_pass"]
        ],
        "chronology_violations": violations,
        "initial_pairs": sum("-initial-" in a["id"] for a in attempts),
        "reference_pairs": sum("-reference-" in a["id"] for a in attempts),
        "sql_planner_matches": sum(p["sql_equal"] for p in planning),
        "agreement_exists": sum(p["agreement_exists"] for p in planning),
        "methods": methods,
        "human_or_network_measurement": False,
        "interpretation": (
            "finite owner-policy search completeness, not a faster generic CSP"
        ),
    }
    if summary["initial_pairs"] + summary["reference_pairs"] + sum(
        m["new_pairs"] for m in methods.values()
    ) != len(attempts):
        raise RuntimeError("paired execution partition does not close")
    save(out / "analysis/rows.json", rows)
    save(out / "analysis/planning.json", planning)
    save(out / "analysis/attempts.json", attempts)
    save(out / "summary.json", summary)
    return summary
