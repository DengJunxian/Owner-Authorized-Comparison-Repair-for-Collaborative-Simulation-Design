"""Predeclared artifact-derived counts and tables; no outcome-based selection."""

from __future__ import annotations

import json

from scripts.cscwd2027_design_contract.freeze import check
from scripts.cscwd2027_design_contract.protection import ROOT, digest
from scripts.cscwd2027_design_contract.store import RESULTS

LABELS = {
    "local_full_only": "Local full",
    "spec_grouped": "Spec-grouped",
    "strict_plan_lock": "Fixed-plan lock",
    "comparison_contract": "Comparison contract",
    "conventional_relational": "Relational control",
    "minus_scientific_binding": "Minus scientific binding",
    "minus_slot_coverage": "Minus slot coverage",
}


def main() -> None:
    """Generate all main counts and complete audit tables from the new summary."""
    check()
    source = RESULTS / "derived/summary.json"
    data = json.loads(source.read_text())
    if data["missing_attempts"] or data["integrity_findings"]:
        raise RuntimeError(
            "missing or unbound evidence cannot be presented as complete"
        )
    paper = ROOT / "paper/cscwd2027_design_contract/generated"
    report = ROOT / "reports/cscwd2027/design_contract_v1"
    paper.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    full = data["mode_statistics"]["comparison_contract"]
    local = data["mode_statistics"]["local_full_only"]
    grouped = data["mode_statistics"]["spec_grouped"]
    revision = data["revision"]
    values = {
        "DesignSessions": revision["sessions"],
        "DesignAttempts": data["actual_attempts"],
        "DesignHandoffs": data["handoff_count"],
        "DesignLocalUnauthorized": local["unauthorized_release"],
        "DesignGroupedUnauthorized": grouped["unauthorized_release"],
        "DesignFullUnauthorized": full["unauthorized_release"],
        "DesignValidControls": full["evaluable_controls"],
        "DesignValidAccepted": full["valid_control_release"],
        "DesignSelectiveCorrect": revision["selective_correct_top1"],
        "DesignStaleWrong": revision["stale_top1_disagreement"],
        "DesignStaleWorse": revision["stale_optimal_set_error"],
        "DesignSelectedNew": revision["selective_new_pairs"],
        "DesignFreshNew": revision["fresh_new_pairs"],
        "DesignReused": revision["reused_pairs"],
        "DesignSlotMatches": revision["slot_science_equal"],
        "DesignSlotPairs": revision["slot_comparisons"],
        "DesignRemapMatches": data["remap"]["science_equal"],
        "DesignRemapPairs": data["remap"]["pairs"],
    }
    outputs = []

    def write(name: str, text: str) -> None:
        path = paper / name
        path.write_text(
            "% Generated from the separate design-contract artifacts.\n" + text + "\n"
        )
        outputs.append(path)

    write(
        "design_numbers.tex",
        "\n".join(
            f"\\newcommand{{\\{key}}}{{{value:,}}}" for key, value in values.items()
        ),
    )
    rows = [
        "\\begin{tabular}{lrrr}\\toprule",
        "Collector & Unauth. & Valid & Top-1 diff.\\\\\\midrule",
    ]
    md = [
        "# Comparison-contract handoff results",
        "",
        "Finite, designed incoming panels; actual numerical report files are counted.",
        "Label recovery is assessed by emitted bindings, not fault labels.",
        "Unavailable/native-failed controls are separate, not false positives.",
        "",
        "| Collector | Sets | Written | Unauthorized release "
        "| Valid controls accepted/evaluable | Unnecessary abstention "
        "| Top-1 diff. | Worse finite-panel choice | Zero-regret diff. |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, label in LABELS.items():
        r = data["mode_statistics"][mode]
        rows.append(
            f"{label} & {r['unauthorized_release']} & "
            f"{r['valid_control_release']}/{r['evaluable_controls']} & "
            f"{r['top1_disagreement']}\\\\"
        )
        md.append(
            f"| {label} | {r['sets']} | {r['written']} | "
            f"{r['unauthorized_release']} | {r['valid_control_release']}/"
            f"{r['evaluable_controls']} | {r['unnecessary_abstention']} | "
            f"{r['top1_disagreement']} | {r['optimal_set_error']} | "
            f"{r['zero_regret_disagreement']} |"
        )
    write("design_handoff_table.tex", "\n".join(rows + ["\\bottomrule\\end{tabular}"]))
    rows = [
        "\\begin{tabular}{lrrr}\\toprule",
        "Revision path & New pairs & Reused & Complete\\\\\\midrule",
        f"Full fresh panel & {revision['fresh_new_pairs']} & 0 & "
        f"{revision['reference_available']}/{revision['sessions']}\\\\",
        f"Contract reuse & {revision['selective_new_pairs']} & "
        f"{revision['reused_pairs']} & {revision['selective_release']}/"
        f"{revision['sessions']}\\\\",
        "\\bottomrule\\end{tabular}",
    ]
    write("design_revision_table.tex", "\n".join(rows))
    (report / "handoff_results.md").write_text("\n".join(md) + "\n")
    md = [
        "# Complete branch/regime sensitivity and revision audit",
        "",
        "Initial pairs are counted once, excluding remap/fresh/selective copies.",
        "No numerical null is removed.",
        "",
        "| Regime:candidate | Pairs | Locally valid | Nonzero contrast |",
        "|---|---:|---:|---:|",
    ]
    for key, value in data["initial_branch_sensitivity"].items():
        md.append(
            f"| {key} | {value['pairs']} | {value['local_valid']} | "
            f"{value.get('nonzero_contrast', 0)} |"
        )
    md += [
        "",
        "Target session counts: "
        + json.dumps(data["target_session_counts"], sort_keys=True),
        "",
        "| Session | Target | Rerun slots | Budget infeasible "
        "| Selective correct | Stale top-1 diff. |",
        "|---|---|---:|---|---|---|",
    ]
    for row in data["all_revision_rows"]:
        md.append(
            f"| {row['session']} | {row['target']} | "
            f"{len(row['plan']['rerun'])} | {row['budget_infeasible']} | "
            f"{row['selective_consequences']['top1_disagreement'] is False} | "
            f"{row['stale_consequences']['top1_disagreement']} |"
        )
    (report / "branch_and_revision_results.md").write_text("\n".join(md) + "\n")
    (paper / "design_manifest.json").write_text(
        json.dumps(
            {
                "input": str(source.relative_to(ROOT)),
                "input_sha256": digest(source),
                "outputs": {str(p.relative_to(ROOT)): digest(p) for p in outputs},
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(values))


if __name__ == "__main__":
    main()
