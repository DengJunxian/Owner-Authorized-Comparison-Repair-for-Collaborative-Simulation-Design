"""Post-seal presentation of already declared endpoints, never a study runner.

The pre-execution analysis/present/plot scripts remain byte-identical. This
helper expands their structured output into audit tables and prose receipts.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from fractions import Fraction
from typing import Any

from scripts.cscwd2027_design_contract.freeze import DEST, check
from scripts.cscwd2027_design_contract.protection import ROOT, digest, verify
from scripts.cscwd2027_design_contract.store import RESULTS

PAPER = ROOT / "paper/cscwd2027_design_contract"
REPORT = ROOT / "reports/cscwd2027/design_contract_v1"


def fraction(value: list[int] | None) -> str:
    """Print exact finite-panel regret, leaving undefined values explicit."""
    return "undefined" if value is None else str(Fraction(*value))


def main(*, scoped_extraction: bool = False) -> None:
    """Generate complete supplementary tables and the implementation receipt."""
    freeze = check()
    protected = (
        {"scope": "scoped extraction; local preservation manifests not replicated"}
        if scoped_extraction
        else verify()
    )
    source = RESULTS / "derived/summary.json"
    data: dict[str, Any] = json.loads(source.read_text())
    evaluation = json.loads((RESULTS / "evaluation/results.json").read_text())
    config = json.loads((DEST / "study.json").read_text())
    sessions = json.loads((DEST / "sessions.json").read_text())
    jobs = json.loads((DEST / "attempts.json").read_text())
    if data["missing_attempts"] or data["integrity_findings"]:
        raise RuntimeError("cannot deliver incomplete or unbound data")
    full = data["mode_statistics"]["comparison_contract"]
    local = data["mode_statistics"]["local_full_only"]
    rev = data["revision"]
    decisions = evaluation["handoffs"]
    full_by_id = {
        r["case_id"]: r for r in decisions if r["mode"] == "comparison_contract"
    }
    relational = [r for r in decisions if r["mode"] == "conventional_relational"]
    agreement = sum(
        r["decision"]["authorized"]
        == full_by_id[r["case_id"]]["decision"]["authorized"]
        and r["consequences"] == full_by_id[r["case_id"]]["consequences"]
        for r in relational
    )
    plan_audit = []
    for row in data["all_revision_rows"]:
        selected = [
            j
            for j in jobs
            if j["session"] == row["session"] and j["phase"] == "selective"
        ]
        selected_ids = {j["id"] for j in selected}
        rerun_matches = {(j["candidate"], j["replicate"]) for j in selected} == {
            (j["candidate"], j["replicate"]) for j in row["plan"]["rerun"]
        }
        numeric = row["selective"]["numerical"]
        bindings = [] if numeric is None else numeric["effective_bindings"]
        reuse_matches = {
            (m["candidate"], m["replicate"], m["object_id"])
            for m in bindings
            if m["object_id"] not in selected_ids
        } == {
            (m["candidate"], m["replicate"], m["object_id"])
            for m in row["plan"]["reuse"]
        }
        plan_audit.append(
            {
                "session": row["session"],
                "rerun_matches": rerun_matches,
                "reuse_matches": reuse_matches,
                "complete_report": numeric is not None,
                "slot_comparisons_complete": all(
                    c["complete"] for c in row["per_slot_comparisons"]
                ),
            }
        )
    extras = {
        "DesignSmallAgents": min(config["agents"]),
        "DesignLargeAgents": max(config["agents"]),
        "DesignDeepDepth": int(config["depth"]["deep"]),
        "DesignDepthFactor": int(config["depth"]["responsive"].split("*")[0]),
        "DesignSessionsPerCell": config["sessions_per_cell"],
        "DesignReplicates": config["replicates"],
        "DesignCandidateCount": len(config["candidates"]),
        "DesignDefaultWorkers": config["plans"][0]["worker_count"],
        "DesignAlternateWorkers": config["plans"][1]["worker_count"],
        "DesignInitialEpoch": config["branch_epoch"],
        "DesignRevisedEpoch": config["revised_branch_epoch"],
        "DesignTopSetSize": 2,  # Frozen evaluator's top2 endpoint, not a new choice.
        "DesignNewPairBudget": config["new_pair_budget"],
        "DesignRelationalAgreement": agreement,
        "DesignCollectorDecisions": len(relational),
        "DesignLocalTopOneWrong": local["top1_disagreement"],
        "DesignLocalTopTwoWrong": local["top2_disagreement"],
        "DesignLocalPositiveRegret": local["optimal_set_error"],
        "DesignLocalZeroRegret": local["zero_regret_disagreement"],
        "DesignLocalWritten": local["written"],
        "DesignSlotDrawMatches": rev["slot_draw_equal"],
        "DesignSelectiveTotal": rev["initial_pairs"] + rev["selective_new_pairs"],
        "DesignFreshTotal": rev["initial_pairs"] + rev["fresh_new_pairs"],
        "DesignFailedAttempts": len(data["failed_attempts"]),
        "DesignLocalRejected": len(data["locally_rejected_attempts"]),
        "DesignUnsafeSets": sum(
            r["case_kind"]
            in {"foreign_session", "missing_slot", "duplicate_slot", "stale_revision"}
            for r in decisions
            if r["mode"] == "comparison_contract"
        ),
        "DesignGroupedPositiveRegret": data["mode_statistics"]["spec_grouped"][
            "optimal_set_error"
        ],
        "DesignGroupedWritten": data["mode_statistics"]["spec_grouped"]["written"],
        "DesignStaleZeroRegret": rev["stale_zero_regret_disagreement"],
        "DesignSelectiveReleased": rev["selective_release"],
        "DesignPlanMatches": sum(
            r["rerun_matches"] and r["reuse_matches"] and r["complete_report"]
            for r in plan_audit
        ),
    }
    generated = PAPER / "generated"
    outputs = []

    def tex(name: str, value: str) -> None:
        path = generated / name
        path.write_text(
            "% Artifact-derived post-seal presentation only.\n" + value + "\n"
        )
        outputs.append(path)

    tex(
        "design_detail_numbers.tex",
        "\n".join(f"\\newcommand{{\\{k}}}{{{v:,}}}" for k, v in extras.items()),
    )
    rows = [
        "\\begin{tabular}{lrrrrrr}\\toprule",
        "Collector & Reports & Unauth. & Top-1 diff. & Top-2 diff. & Worse & Tied diff."
        "\\\\\\midrule",
    ]
    labels = {
        "local_full_only": "Local full",
        "spec_grouped": "Spec-grouped",
        "strict_plan_lock": "Fixed-plan lock",
        "comparison_contract": "Comparison contract",
        "conventional_relational": "Relational control",
        "minus_scientific_binding": "Minus scientific binding",
        "minus_slot_coverage": "Minus slot coverage",
    }
    for mode, label in labels.items():
        r = data["mode_statistics"][mode]
        fields = (
            "written",
            "unauthorized_release",
            "top1_disagreement",
            "top2_disagreement",
            "optimal_set_error",
            "zero_regret_disagreement",
        )
        rows.append(label + " & " + " & ".join(str(r[k]) for k in fields) + "\\\\")
    tex(
        "design_consequences_table.tex",
        "\n".join(rows + ["\\bottomrule\\end{tabular}"]),
    )
    rows = [
        "\\begin{tabular}{lrrr}\\toprule",
        "Agents/regime/candidate & Pairs & Valid & Nonzero $\\Delta$\\\\\\midrule",
    ]
    for key, r in data["initial_branch_sensitivity"].items():
        parts = key.split(":")
        regime = parts[1].removeprefix("cscwd-design-panel-")
        branch = {"agent_exit": "exit", "latency_reorder": "latency"}.get(
            parts[2], parts[2]
        )
        label = f"{parts[0]}/{regime}/{branch}"
        rows.append(
            f"{label} & {r['pairs']} & {r['local_valid']} & "
            f"{r.get('nonzero_contrast', 0)}\\\\"
        )
    tex("design_branch_table.tex", "\n".join(rows + ["\\bottomrule\\end{tabular}"]))
    cells: dict[str, Counter[str]] = {}
    for r in data["all_revision_rows"]:
        c = cells.setdefault(f"{r['agents']}/{r['regime']}", Counter())
        c["sessions"] += 1
        c["selective_release"] += r["selective"]["numerical_file_exists"]
        for key in (
            "top1_disagreement",
            "top2_disagreement",
            "optimal_set_error",
            "zero_regret_disagreement",
        ):
            c[key] += r["stale_consequences"][key] is True
    rows = [
        "\\begin{tabular}{lrrrr}\\toprule",
        "Cell & Sessions & Reuse reports & Stale diff. & Stale worse\\\\\\midrule",
    ]
    for key, r in cells.items():
        rows.append(
            f"{key} & {r['sessions']} & {r['selective_release']} & "
            f"{r['top1_disagreement']} & {r['optimal_set_error']}\\\\"
        )
    tex("design_revision_cells.tex", "\n".join(rows + ["\\bottomrule\\end{tabular}"]))
    receipt = {
        "freeze_sha256": digest(DEST / "freeze.json"),
        "freeze_created_at_utc": freeze["created_at_utc"],
        "old_evidence_protection": protected,
        "predeclared_attempts": data["planned_attempts"],
        "actual_attempts": data["actual_attempts"],
        "failed_attempt_ids": data["failed_attempts"],
        "locally_rejected_ids": data["locally_rejected_attempts"],
        "integrity_findings": data["integrity_findings"],
        "postseal_helper": "presentation only; no new endpoint or simulation",
        "numeric_values": extras,
        "revision_by_cell": {k: dict(v) for k, v in cells.items()},
        "posthoc_schedule_implementation_audit": plan_audit,
        "all_slot_comparisons_complete": all(
            r["slot_comparisons_complete"] for r in plan_audit
        ),
        "all_remap_comparisons_complete": all(
            r["complete"] for r in evaluation["remap_comparisons"]
        ),
        "regret": {
            mode: {k: fraction(r[k]) for k in ("regret_mean", "regret_max")}
            for mode, r in data["mode_statistics"].items()
        },
        "source": {str(source.relative_to(ROOT)): digest(source)},
        "outputs": {str(p.relative_to(ROOT)): digest(p) for p in outputs},
    }
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "presentation_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    lines = [
        "# Execution-contract and collaborative-design experiment",
        "",
        "Status: completed local post-registration study; not a submission receipt.",
        "",
        f"Pre-execution freeze: `{receipt['freeze_sha256']}` "
        f"at `{freeze['created_at_utc']}`.",
        f"Actual/planned paired attempts: {data['actual_attempts']}/"
        f"{data['planned_attempts']}; "
        f"sessions: {len(sessions)}; handoffs: {data['handoff_count']}.",
        f"Failed: {len(data['failed_attempts'])}; "
        f"locally rejected: {len(data['locally_rejected_attempts'])}; "
        f"integrity findings: {len(data['integrity_findings'])}.",
        "",
        "## Scientific result",
        "",
        f"Local verification writes {local['unauthorized_release']} "
        "unauthorized panels; "
        "spec-grouping writes "
        f"{data['mode_statistics']['spec_grouped']['unauthorized_release']}; "
        f"the comparison contract writes {full['unauthorized_release']} and accepts "
        f"{full['valid_control_release']}/{full['evaluable_controls']} "
        "evaluable controls.",
        f"Equal-obligation relational agreement: {agreement}/{len(relational)}.",
        f"Selective new/reused pairs: {rev['selective_new_pairs']}/"
        f"{rev['reused_pairs']}; "
        f"fresh new pairs: {rev['fresh_new_pairs']}; exact scientific slots: "
        f"{rev['slot_science_equal']}/{rev['slot_comparisons']}.",
        f"Stale top-1 disagreements: {rev['stale_top1_disagreement']}; "
        f"positive-regret choices: {rev['stale_optimal_set_error']}; "
        f"zero-regret disagreements: {rev['stale_zero_regret_disagreement']}.",
        "",
        "## Boundaries and negative results",
        "",
        "The declared one-candidate revision fixes the theoretical work saving; "
        "the experiment tests correctness of actual reused evidence. "
        "No latency speedup is claimed.",
        "The relational tie is retained. All nulls, top-2 disagreements, ties, "
        "false negatives, abstentions and native failures are in the summary "
        "and actual files.",
        "Branch epoch changes the event namespace, including baseline draws; "
        "this is a versioned-design study, not a controlled causal timing estimate.",
        "One CLOB, one host, same-code finite-panel reference, trusted local store; "
        "no human-team, independent reproduction or market-validity claim.",
        "",
        "See handoff_results.md, branch_and_revision_results.md, "
        "the structured summary, "
        "and evaluation/results.json for every case and emitted file.",
    ]
    (REPORT / "experiment_results.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"generated_files": len(outputs), "numeric_values": extras}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scoped-extraction", action="store_true")
    main(scoped_extraction=parser.parse_args().scoped_extraction)
