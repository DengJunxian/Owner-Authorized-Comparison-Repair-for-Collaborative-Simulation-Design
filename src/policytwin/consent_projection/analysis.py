"""Recompute planning, local checks, accepted-target equality and work counts."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from policytwin.consent_projection.policy import authorized
from policytwin.consent_projection.search import ALL_METHODS, METHODS, search
from policytwin.consent_projection.study import schedule
from policytwin.coordination.schema import commitment
from policytwin.negotiated_repair.solver import alternatives
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


def checked_catalog(root: Path, record: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for item in record["catalog"]:
        trace, capsule = acquired_trace(item["trace"]["object_id"], root)
        if trace != TraceToken(**item["trace"]):
            raise RuntimeError("catalog trace differs")
        token = EvaluationToken(**item["evaluation"])
        if trace.local_pass and token != verify_derivation(
            trace,
            capsule,
            read(root / "initial_evaluations" / f"{trace.object_id}.json"),
        ):
            raise RuntimeError("catalog evaluation differs")
        result.append({"trace": asdict(trace), "evaluation": asdict(token)})
    if commitment(result) != record["catalog_sha256"]:
        raise RuntimeError("catalog commitment differs")
    return result


def verify_physical(
    root: Path, entry: dict[str, Any], method: str, references: list[dict[str, Any]]
) -> dict[str, Any]:
    request, policies = entry["request"], entry["policies"]
    choice = entry["decisions"][method]
    option = choice["selected"]
    identifier = f"{request['session']['id']}-{request['kind']}-{method}"
    outer = root / "projection_handoffs" / identifier
    row: dict[str, Any] = read(outer / "result.json")
    if read(outer / "selection.json") != choice:
        raise RuntimeError("physical choice differs")
    if (row["id"], row["method"], row["session"], row["catalog_size"]) != (
        identifier,
        method,
        request["session"]["id"],
        request["availability"],
    ):
        raise RuntimeError("physical coordinate differs")
    actual = len(list((root / "objects").glob(identifier + "-*/result.json")))
    if actual != row["new_pairs"]:
        raise RuntimeError("actual acquisitions differ")
    if option is None:
        if (
            actual
            or row["authorized"]
            or row["exact"] is not None
            or row["status"] != "no_agreement"
        ):
            raise RuntimeError("unagreed request acquired/released")
        return row
    for key, action in (
        ("new_pairs", "execute"),
        ("reevaluations", "evaluate"),
        ("reused_evaluations", "reuse"),
    ):
        if row[key] != sum(w["action"] == action for w in option["work_order"]):
            raise RuntimeError("work count differs")
    if not authorized(
        request, option, policies, entry["catalog_sha256"], read(outer / "votes.json")
    ):
        raise RuntimeError("owner agreement differs")
    folder = root / "handoffs" / identifier
    if read(folder / "work_order.json")["actions"] != option["work_order"]:
        raise RuntimeError("physical work order differs")
    response = read(folder / "producer_response.json")
    members, capsules = [], {}
    for raw, ev in zip(response["traces"], response["evaluations"], strict=True):
        trace, capsule = acquired_trace(raw["object_id"], root)
        if trace != TraceToken(**raw):
            raise RuntimeError("response trace differs")
        token = EvaluationToken(**ev)
        if trace.local_pass and token != verify_derivation(
            trace,
            capsule,
            read(folder / "evaluations" / f"{trace.candidate}-r{trace.replicate}.json"),
        ):
            raise RuntimeError("response metric differs")
        members.append(token)
        capsules[trace.slot] = capsule
    declared = option["contract"]
    numerical = decide(
        declared["contract_sha256"],
        expectations(declared),
        tuple(declared["allowed_plans"]),
        members,
    )
    if row["authorized"] != numerical["authorized"] or row["status"] != (
        "released" if numerical["authorized"] else "execution_abstention"
    ):
        raise RuntimeError("numerical release differs")
    reference = next(
        r
        for r in references
        if r["traces"][0]["object_id"].startswith(
            request["session"]["id"] + "-reference-"
        )
        and all(
            r["trajectory_target"][k] == option["target"][k]
            for k in ("epoch", "coupling", "workers")
        )
    )
    by_slot = {e.slot: e for e in members}
    ref_tokens = []
    matching = 0
    for raw in reference["traces"]:
        trace, capsule = acquired_trace(raw["object_id"], root)
        if trace != TraceToken(**raw):
            raise RuntimeError("reference trace differs")
        token = next(
            EvaluationToken(**e)
            for e in reference["evaluations"][option["target"]["metric"]]
            if (e["candidate"], e["replicate"]) == trace.slot
        )
        if trace.local_pass and token != verify_derivation(
            trace,
            capsule,
            read(
                root
                / "reference_evaluations"
                / f"{trace.object_id}-{option['target']['metric']}.json"
            ),
        ):
            raise RuntimeError("reference derivation differs")
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
        if (row["exact"], row["matching_slots"], row["compared_slots"]) != (
            exact,
            matching,
            len(members),
        ):
            raise RuntimeError("accepted-target equality differs")
    elif row["exact"] is not None:
        raise RuntimeError("invalid equality claim")
    if row["authorization_predicate_calls"] != 9:
        raise RuntimeError("authorization operation count differs")
    return row


def aggregate(
    rows: list[dict[str, Any]], physical: list[dict[str, Any]]
) -> dict[str, Any]:
    output = {}
    for method in ALL_METHODS:
        group = [r for r in rows if r["method"] == method]
        actual = [r for r in physical if r["method"] == method]
        output[method] = {
            "requests": len(group),
            "agreement": sum(r["selected"] for r in group),
            "no_agreement": sum(not r["selected"] for r in group),
            "false_abstentions": sum(r["false_abstention"] for r in group),
            "selection_disagreements": sum(not r["selection_equal"] for r in group),
            "search_predicate_calls": None
            if method == "policy_sql"
            else sum(r["search_predicate_calls"] for r in group),
            "selection_check_calls": sum(r["selection_check_calls"] for r in group),
            "skipped_candidates": None
            if method == "policy_sql"
            else sum(r["skipped_candidates"] for r in group),
            "visited_candidates": None
            if method == "policy_sql"
            else sum(r["visited_candidates"] for r in group),
            "physical_requests": len(actual),
            "released": sum(r["authorized"] for r in actual),
            "execution_abstentions": sum(
                r["status"] == "execution_abstention" for r in actual
            ),
            "exact_reports": sum(r["exact"] is True for r in actual),
            **{
                k: sum(r[k] for r in actual)
                for k in (
                    "new_pairs",
                    "reevaluations",
                    "reused_evaluations",
                    "matching_slots",
                    "compared_slots",
                    "authorization_predicate_calls",
                )
            },
        }
    return output


def summarize(root: Path, destination: Path | None = None) -> dict[str, Any]:
    out = root if destination is None else destination
    references = [read(p) for p in sorted((root / "references").glob("*.json"))]
    catalogs = {p.stem: read(p) for p in sorted((root / "catalogs").glob("*.json"))}
    recomputed = {}
    rows, physical, violations = [], [], []
    expected = {
        f"{b['session']['id']}-{t['request']['kind']}": t
        for b in schedule()
        for c in b["contexts"]
        for t in c["tickets"]
    }
    paths = sorted((root / "planning").glob("*.json"))
    if {p.stem for p in paths} != set(expected):
        raise RuntimeError("planning case coverage differs from pre-outcome schedule")
    for path in paths:
        entry = read(path)
        if {k: entry[k] for k in expected[path.stem]} != expected[path.stem]:
            raise RuntimeError("retained request/policy differs from schedule")
        request, policies = entry["request"], entry["policies"]
        cid = entry["catalog_id"]
        if cid not in recomputed:
            record = catalogs[cid]
            checked = checked_catalog(root, record)
            options = alternatives(
                request,
                [TraceToken(**x["trace"]) for x in checked],
                [EvaluationToken(**x["evaluation"]) for x in checked],
            )
            if options != record["alternatives"]:
                raise RuntimeError("typed work alternatives differ")
            recomputed[cid] = options
        if entry["catalog_sha256"] != catalogs[cid]["catalog_sha256"]:
            raise RuntimeError("planning catalog differs")
        decisions = {
            m: search(recomputed[cid], request, policies, entry["catalog_sha256"], m)
            for m in ALL_METHODS
        }
        if decisions != entry["decisions"]:
            raise RuntimeError("planning decisions differ")
        baseline = decisions["exhaustive"]["selected"]
        for method, decision in decisions.items():
            selected = decision["selected"]
            rows.append(
                {
                    "session": request["session"]["id"],
                    "agents": request["session"]["agents"],
                    "regime": request["session"]["regime"],
                    "catalog_size": request["availability"],
                    "request": request["kind"],
                    "method": method,
                    "selected": selected is not None,
                    "target_id": None if selected is None else selected["target_id"],
                    "selection_equal": selected == baseline,
                    "false_abstention": selected is None and baseline is not None,
                    **{
                        k: decision[k]
                        for k in (
                            "search_predicate_calls",
                            "selection_check_calls",
                            "skipped_candidates",
                            "visited_candidates",
                        )
                    },
                }
            )
            if entry["physical"] and method in METHODS:
                physical.append(verify_physical(root, entry, method, references))
        if entry["physical"]:
            stamp = datetime.fromisoformat(entry["recorded_before_acquisition_utc"])
            for path2 in (root / "objects").glob(
                request["session"]["id"] + "-*/attempt.json"
            ):
                identifier = path2.parent.name
                if (
                    "-reference-" in identifier or f"-{request['kind']}-" in identifier
                ) and datetime.fromisoformat(
                    read(path2)["registration"]["registered_at_utc"]
                ) < stamp:
                    violations.append(identifier)
    attempts = [read(p) for p in sorted((root / "objects").glob("*/result.json"))]
    methods = aggregate(rows, physical)
    summary = {
        "schema": "consent-projection-summary.v1",
        "scientific_blocks": len({r["session"] for r in rows}),
        "catalog_contexts": len(catalogs),
        "planning_requests": len(rows) // len(ALL_METHODS),
        "physical_requests": len(physical) // len(METHODS),
        "method_panels": len(physical),
        "actual_paired_attempts": len(attempts),
        "local_passes": sum(r["local_pass"] for r in attempts),
        "failed_or_rejected_attempts": [
            r["id"] for r in attempts if not r["local_pass"]
        ],
        "chronology_violations": sorted(set(violations)),
        "initial_pairs": sum("-initial-" in r["id"] for r in attempts),
        "reference_pairs": sum("-reference-" in r["id"] for r in attempts),
        "methods": methods,
        "by_catalog": {
            str(size): aggregate(
                [r for r in rows if r["catalog_size"] == size],
                [r for r in physical if r["catalog_size"] == size],
            )
            for size in (3, 7, 11)
        },
        "by_block": {
            s: aggregate(
                [r for r in rows if r["session"] == s],
                [r for r in physical if r["session"] == s],
            )
            for s in sorted({r["session"] for r in rows})
        },
        "interpretation": (
            "generated finite-policy distribution, not natural frequency; "
            "predicate calls not runtime; SQL has full obligation parity"
        ),
    }
    if summary["initial_pairs"] + summary["reference_pairs"] + sum(
        methods[m]["new_pairs"] for m in METHODS
    ) != len(attempts):
        raise RuntimeError("physical attempt partition does not close")
    save(out / "analysis/planning_rows.json", rows)
    save(out / "analysis/physical_rows.json", physical)
    save(out / "analysis/attempts.json", attempts)
    save(out / "summary.json", summary)
    return summary
