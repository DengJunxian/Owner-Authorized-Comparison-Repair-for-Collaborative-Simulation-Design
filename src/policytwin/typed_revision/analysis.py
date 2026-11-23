"""Exact finite-panel analysis; failed references are never zero-valued outcomes."""

from __future__ import annotations

from collections import Counter, defaultdict
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any

from policytwin.cscwd_mechanism.coupling import load
from policytwin.typed_revision.runtime import read, trace_matches
from scripts.cscwd2027_design_contract.store import save


def decision_comparison(
    decision: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    if not decision["authorized"] or not reference["authorized"]:
        return {"assessable": False, "exact": None, "top1_flip": None, "regret": None}
    proposed, wanted = decision["report"], reference["report"]
    means = {c: Fraction(*v) for c, v in wanted["means"].items()}
    regret = max(means.values()) - means[proposed["top1"]]
    return {
        "assessable": True,
        "exact": all(proposed[k] == wanted[k] for k in ("means", "top1", "top2")),
        "top1_flip": proposed["top1"] != wanted["top1"],
        "top2_flip": proposed["top2"] != wanted["top2"],
        "regret": [regret.numerator, regret.denominator],
    }


def summarize(root: Path, *, output: Path | None = None) -> dict[str, Any]:
    target = output or root
    fresh = {
        (row["session"], row["request"]): row
        for path in sorted((root / "fresh_panels").glob("*.json"))
        for row in read(path)
    }

    @lru_cache(maxsize=32)
    def capsule(identifier: str) -> dict[str, Any]:
        _, value = load(identifier, root)
        return value

    rows = []
    for path in sorted((root / "handoffs").glob("*/result.json")):
        value = read(path)
        reference = fresh[value["session"], value["request"]]
        comparison = decision_comparison(value["final_decision"], reference["decision"])
        ablation = decision_comparison(value["unsafe_decision"], reference["decision"])
        ref_traces = {(t["candidate"], t["replicate"]): t for t in reference["traces"]}
        comparisons = []
        for trace in value["trace_tokens"]:
            wanted = ref_traces[trace["candidate"], trace["replicate"]]
            assessable = trace["local_pass"] and wanted["local_pass"]
            comparisons.append(
                {
                    "candidate": trace["candidate"],
                    "replicate": trace["replicate"],
                    "assessable": assessable,
                    "exact": trace_matches(
                        capsule(trace["object_id"]), capsule(wanted["object_id"])
                    )
                    if assessable
                    else None,
                }
            )
        rows.append(
            {
                k: value[k]
                for k in (
                    "id",
                    "session",
                    "agents",
                    "regime",
                    "request",
                    "availability",
                    "method",
                    "new_pairs",
                    "reevaluations",
                    "reused_evaluations",
                    "relational_agreement",
                    "budget_feasible",
                    "repeat_request_all_reuse",
                    "request_rounds",
                    "request_response_bytes",
                    "elapsed_seconds",
                    "initial_release",
                    "unsafe_invalid_release",
                    "safe_findings_on_ablation",
                )
            }
            | {
                "final_release": value["final_decision"]["authorized"],
                "fresh_release": reference["decision"]["authorized"],
                "comparison": comparison,
                "ablation_comparison": ablation,
                "slot_comparisons": comparisons,
            }
        )
    cells = defaultdict(list)
    for row in rows:
        cells[row["request"], row["availability"], row["method"]].append(row)
    aggregates = []
    for (request, availability, method), group in sorted(cells.items()):
        aggregates.append(
            {
                "request": request,
                "availability": availability,
                "method": method,
                "panels": len(group),
                "released": sum(r["final_release"] for r in group),
                "assessable": sum(r["comparison"]["assessable"] for r in group),
                "exact_reports": sum(r["comparison"]["exact"] is True for r in group),
                "new_pairs": sum(r["new_pairs"] for r in group),
                "reevaluations": sum(r["reevaluations"] for r in group),
                "budget_feasible": sum(r["budget_feasible"] for r in group),
                "unsafe_invalid_release": sum(
                    r["unsafe_invalid_release"] for r in group
                ),
                "unsafe_assessable": sum(
                    r["ablation_comparison"]["assessable"] for r in group
                ),
                "unsafe_top1_flips": sum(
                    r["ablation_comparison"]["top1_flip"] is True for r in group
                ),
            }
        )
    attempts = []
    for path in sorted((root / "objects").glob("*/result.json")):
        value = read(path)
        attempts.append(
            {
                "id": value["id"],
                "status": value["status"],
                "local_pass": value["local_pass"],
                "elapsed_seconds": value["elapsed_seconds"],
                "error": value.get("error"),
            }
        )
    typed = [r for r in rows if r["method"] == "typed"]
    opaque = [r for r in rows if r["method"] == "opaque"]
    summary = {
        "sessions": len({r["session"] for r in rows}),
        "requested_panels_per_method": len(typed),
        "fresh_reference_panels": len(fresh),
        "actual_paired_attempts": len(attempts),
        "local_pass": sum(a["local_pass"] for a in attempts),
        "local_rejections": [a for a in attempts if not a["local_pass"]],
        "typed": {
            "new_pairs": sum(r["new_pairs"] for r in typed),
            "reevaluations": sum(r["reevaluations"] for r in typed),
            "released": sum(r["final_release"] for r in typed),
            "exact_reports": sum(r["comparison"]["exact"] is True for r in typed),
            "assessable_reports": sum(r["comparison"]["assessable"] for r in typed),
            "exact_slots": sum(
                s["exact"] is True for r in typed for s in r["slot_comparisons"]
            ),
            "assessable_slots": sum(
                s["assessable"] for r in typed for s in r["slot_comparisons"]
            ),
            "relational_agreement": sum(r["relational_agreement"] for r in typed),
            "budget_feasible": sum(r["budget_feasible"] for r in typed),
            "idempotent_repeats": sum(r["repeat_request_all_reuse"] for r in typed),
            "unsafe_invalid_release": sum(r["unsafe_invalid_release"] for r in typed),
            "unsafe_top1_flips": sum(
                r["ablation_comparison"]["top1_flip"] is True for r in typed
            ),
            "unsafe_assessable": sum(
                r["ablation_comparison"]["assessable"] for r in typed
            ),
        },
        "opaque": {
            "new_pairs": sum(r["new_pairs"] for r in opaque),
            "released": sum(r["final_release"] for r in opaque),
            "exact_reports": sum(r["comparison"]["exact"] is True for r in opaque),
            "budget_feasible": sum(r["budget_feasible"] for r in opaque),
        },
        "strata": aggregates,
        "denominators": {
            "scientific_blocks": (
                "eight new-seed sessions; views and methods repeat within blocks"
            ),
            "references": (
                "one independently executed fresh panel per session/request, "
                "shared across three availability views"
            ),
            "budget": ("six-pair feasibility from recorded pre-outcome work orders"),
            "unsafe": (
                "metric-binding ablation; genuine old derivations, "
                "current trajectory checks"
            ),
        },
        "attempt_statuses": dict(Counter(a["status"] for a in attempts)),
    }
    save(target / "analysis" / "rows.json", rows)
    save(target / "analysis" / "attempts.json", attempts)
    save(target / "summary.json", summary)
    return summary
