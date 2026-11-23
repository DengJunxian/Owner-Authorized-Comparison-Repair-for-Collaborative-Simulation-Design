"""Frozen finite-panel analysis; all cases, abstentions, failures and ties retained."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

from scripts.cscwd2027_design_contract.freeze import DEST, check
from scripts.cscwd2027_design_contract.protection import digest
from scripts.cscwd2027_design_contract.store import RESULTS


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Separate panel validity, deterministic label errors and positive regret."""
    counts: Counter[str] = Counter()
    ids: dict[str, list[str]] = {}
    regret: list[Fraction] = []
    for row in rows:
        counts["sets"] += 1
        written = row["decision"]["numerical_file_exists"]
        control = row["nominal_valid_or_recoverable"]
        evaluable = row["all_observed_local_valid"] and row["reference_available"]
        marks = {
            "all_observed_local_valid": row["all_observed_local_valid"],
            "written": written,
            "unauthorized_release": written and not row["output_panel_valid"],
            "valid_or_recoverable_controls": control,
            "evaluable_controls": control and evaluable,
            "unavailable_controls": control and not evaluable,
            "valid_control_release": control
            and evaluable
            and written
            and row["output_panel_valid"],
            "unnecessary_abstention": control and evaluable and not written,
            "top1_disagreement": row["consequences"]["top1_disagreement"] is True,
            "top2_disagreement": row["consequences"]["top2_disagreement"] is True,
            "optimal_set_error": row["consequences"]["optimal_set_error"] is True,
            "zero_regret_disagreement": row["consequences"]["zero_regret_disagreement"]
            is True,
            "assessable_decisions": row["consequences"]["assessable"],
        }
        for key, present in marks.items():
            counts[key] += present
            if present:
                ids.setdefault(key, []).append(row["case_id"])
        if row["consequences"]["regret"] is not None:
            regret.append(Fraction(*row["consequences"]["regret"]))
    mean = sum(regret, Fraction()) / len(regret) if regret else None
    maximum = max(regret) if regret else None
    return {
        **dict(counts),
        "ids": ids,
        "regret_mean": None if mean is None else [mean.numerator, mean.denominator],
        "regret_max": None
        if maximum is None
        else [maximum.numerator, maximum.denominator],
        "regret_denominator": len(regret),
    }


def analyze() -> dict[str, Any]:
    """Bind raw files and actual numerical outputs before deriving any paper count."""
    seal = check()
    planned = json.loads((DEST / "attempts.json").read_text())
    results = []
    missing = []
    findings = []
    for row in planned:
        folder = RESULTS / "objects" / row["id"]
        path = folder / "result.json"
        if not path.is_file():
            missing.append(row["id"])
            continue
        result = json.loads(path.read_text())
        results.append(result)
        attempt = json.loads((folder / "attempt.json").read_text())
        if result["design"] != row or attempt["design"] != row:
            findings.append(row["id"] + ":design")
        if attempt["freeze_sha256"] != digest(DEST / "freeze.json"):
            findings.append(row["id"] + ":freeze")
        if datetime.fromisoformat(attempt["started_at_utc"]) < datetime.fromisoformat(
            seal["created_at_utc"]
        ):
            findings.append(row["id"] + ":predates_freeze")
        if result["status"] == "completed":
            if len(set(result["worker_pids"].values())) != row["plan"]["worker_count"]:
                findings.append(row["id"] + ":worker_count")
            for name, sha in result["files"].items():
                if digest(folder / name) != sha:
                    findings.append(row["id"] + ":file_binding")
    evaluation = json.loads((RESULTS / "evaluation/results.json").read_text())
    for row in evaluation["handoffs"]:
        folder = RESULTS / "evaluation/handoffs" / row["case_id"] / row["mode"]
        numeric = folder / "numerical_comparison.json"
        if numeric.is_file() != row["decision"]["numerical_file_exists"]:
            findings.append(row["case_id"] + ":report_presence")
        if (
            numeric.is_file()
            and json.loads(numeric.read_text()) != row["decision"]["numerical"]
        ):
            findings.append(row["case_id"] + ":report_content")
    modes = sorted({r["mode"] for r in evaluation["handoffs"]})
    kinds = sorted({r["case_kind"] for r in evaluation["handoffs"]})
    mode_stats = {
        mode: aggregate([r for r in evaluation["handoffs"] if r["mode"] == mode])
        for mode in modes
    }
    case_stats = {
        kind: {
            mode: aggregate(
                [
                    r
                    for r in evaluation["handoffs"]
                    if r["mode"] == mode and r["case_kind"] == kind
                ]
            )
            for mode in modes
        }
        for kind in kinds
    }
    revision: Counter[str] = Counter()
    revision_ids: dict[str, list[str]] = {}
    for row in evaluation["revisions"]:
        revision["sessions"] += 1
        flags = {
            "reference_available": row["reference_available"],
            "selective_release": row["selective"]["numerical_file_exists"],
            "budget_infeasible": row["budget_infeasible"],
            "selective_correct_top1": row["selective_consequences"]["assessable"]
            and not row["selective_consequences"]["top1_disagreement"],
            "stale_top1_disagreement": row["stale_consequences"]["top1_disagreement"]
            is True,
            "stale_optimal_set_error": row["stale_consequences"]["optimal_set_error"]
            is True,
            "stale_zero_regret_disagreement": row["stale_consequences"][
                "zero_regret_disagreement"
            ]
            is True,
        }
        for key, flag in flags.items():
            revision[key] += flag
            if flag:
                revision_ids.setdefault(key, []).append(row["session"])
        for key in (
            "selective_new_pairs",
            "fresh_new_pairs",
            "reused_pairs",
            "initial_pairs",
        ):
            revision[key] += row[key]
        for comparison in row["per_slot_comparisons"]:
            revision["slot_comparisons"] += 1
            revision["slot_science_equal"] += comparison["science_equal"]
            revision["slot_draw_equal"] += comparison["draw_equal"]
            revision["slot_zero_error"] += comparison["contrast_error"] == 0
    remap = {
        "pairs": len(evaluation["remap_comparisons"]),
        **{
            key: sum(r[key] for r in evaluation["remap_comparisons"])
            for key in ("complete", "science_equal", "draw_equal")
        },
    }
    branch_tables: dict[str, Counter[str]] = {}
    for row in results:
        if row["design"]["phase"] != "initial":
            continue
        spec = row["design"]["recipe"]["spec"]
        key = f"{spec['agent_count']}:{spec['study']}:{row['design']['candidate']}"
        counter = branch_tables.setdefault(key, Counter())
        counter["pairs"] += 1
        counter["local_valid"] += row["local_pass"]
        if row["metric"] is not None:
            counter["nonzero_contrast"] += row["metric"] != 0
    manifest = {
        str(p.relative_to(RESULTS)): digest(p)
        for p in sorted(RESULTS.rglob("*"))
        if p.is_file()
        and "derived" not in p.relative_to(RESULTS).parts
        and "protection" not in p.relative_to(RESULTS).parts
    }
    return {
        "label": "post-registration frozen design-contract study",
        "planned_attempts": len(planned),
        "actual_attempts": len(results),
        "failed_attempts": [r["id"] for r in results if r["status"] != "completed"],
        "locally_rejected_attempts": [r["id"] for r in results if not r["local_pass"]],
        "missing_attempts": missing,
        "integrity_findings": findings,
        "handoff_count": len(evaluation["handoffs"]) // len(modes),
        "mode_statistics": mode_stats,
        "case_statistics": case_stats,
        "revision": {**dict(revision), "ids": revision_ids},
        "remap": remap,
        "initial_branch_sensitivity": {k: dict(v) for k, v in branch_tables.items()},
        "all_revision_rows": evaluation["revisions"],
        "target_session_counts": dict(
            Counter(r["target"] for r in evaluation["revisions"])
        ),
        "raw_file_manifest": manifest,
        "freeze_sha256": digest(DEST / "freeze.json"),
        "analysis_source_sha256": digest(Path(__file__)),
        "interpretation": (
            "32 fixed design blocks, designed handoffs and same-code finite-panel "
            "references; not human efficacy, market validity, independent "
            "reproduction or a new caching algorithm"
        ),
    }


def main() -> None:
    """Write statistics in the separate new result namespace."""
    value = analyze()
    target = RESULTS / "derived"
    target.mkdir(parents=True, exist_ok=True)
    (target / "summary.json").write_text(json.dumps(value, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: value[k]
                for k in (
                    "actual_attempts",
                    "planned_attempts",
                    "missing_attempts",
                    "failed_attempts",
                    "locally_rejected_attempts",
                    "integrity_findings",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
