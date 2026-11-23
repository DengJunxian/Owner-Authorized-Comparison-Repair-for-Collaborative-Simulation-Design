"""Predefined presentation of revision2 counts; no outcome-based selection."""

from __future__ import annotations

import json
from collections import Counter

from scripts.cscwd2027_revision2.design import CONFIG
from scripts.cscwd2027_revision2.freeze import ROOT, check, digest


def main() -> None:
    """Generate manuscript values and complete branch/operator review tables."""
    check()
    source = ROOT / "results/cscwd2027/revision2/derived/summary.json"
    summary = json.loads(source.read_text())
    if (
        not summary["data_complete"]
        or summary["failed"]
        or summary["integrity_findings"]
    ):
        raise RuntimeError("complete verified data required")
    paper = ROOT / "paper/cscwd2027_revision2/generated"
    reports = ROOT / "reports/cscwd2027/revision2"
    paper.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    extension = summary["extension"]
    mutants = {k: v for k, v in summary["common_mode"].items() if k != "clean"}
    clean = summary["common_mode"]["clean"]
    values = {
        "RevisionAttempts": summary["attempts"],
        "ExtendedProcessPairs": sum(r["attempts"] for r in extension),
        "ExtendedTraceMatches": sum(r["trace_equal"] for r in extension),
        "ExtendedReplicates": CONFIG["replicates"],
        "ExtendedMaxAgents": max(r["agents"] for r in extension),
        "ExtendedMaxEpochs": max(r["epochs"] for r in extension),
        "CommonDefectAttempts": sum(r["attempts"] for r in mutants.values()),
        "CommonFullReleases": sum(r["full_release"] for r in mutants.values()),
        "WitnessInvalidReleases": sum(r["witness_release"] for r in mutants.values()),
        "CommonCleanAttempts": clean["attempts"],
        "CommonCleanAccepted": clean["witness_release"],
        "CommonContrastChanged": sum(r["contrast_changed"] for r in mutants.values()),
        "CommonContrastUnchanged": sum(
            r["contrast_unchanged"] for r in mutants.values()
        ),
    }
    outputs = []

    def write(name: str, lines: list[str]) -> None:
        path = paper / name
        path.write_text(
            "% Generated from the separate revision2 artifacts.\n"
            + "\n".join(lines)
            + "\n"
        )
        outputs.append(path)

    write(
        "revision_numbers.tex",
        [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in values.items()],
    )
    cells: dict[tuple[int, int, str], Counter[str]] = {}
    for row in extension:
        counter = cells.setdefault(
            (row["agents"], row["epochs"], row["assignment"]), Counter()
        )
        for key in ("attempts", "trace_equal", "zero_contrast_error", "witness_accept"):
            counter[key] += row[key]
    lines = [
        "\\begin{tabular}{rrlrr}\\toprule",
        "Agents & Epochs & Map & Exact trace & Accepted\\\\\\midrule",
    ]
    for (agents, epochs, mapping), value in sorted(cells.items()):
        label = "Round-robin" if mapping == "round_robin" else "Contiguous"
        lines.append(
            f"{agents} & {epochs} & {label} & "
            f"{value['trace_equal']}/{value['attempts']} & "
            f"{value['witness_accept']}/{value['attempts']}\\\\"
        )
    write("extension_table.tex", lines + ["\\bottomrule\\end{tabular}"])
    labels = {
        "quantity_rotation": "Quantity rotation",
        "side_flip": "Side flip",
        "omit_balance_feedback": "Balance omitted",
        "trade_price_shift": "Trade-price shift",
        "clean": "Clean controls",
    }
    lines = [
        "\\begin{tabular}{lrrr}\\toprule",
        "Shared defect & Pairs & Full & +Witness\\\\\\midrule",
    ]
    for kind in (*CONFIG["common_mode_mutants"], "clean"):
        value = summary["common_mode"][kind]
        lines.append(
            f"{labels[kind]} & {value['attempts']} & "
            f"{value['full_release']} & {value['witness_release']}\\\\"
        )
    write("common_mode_table.tex", lines + ["\\bottomrule\\end{tabular}"])
    lines = [
        "# Post-registration extension results",
        "",
        "| Agents | Epochs | Branch | Map | Exact trace | "
        "Zero contrast error | Witnessed acceptance |",
        "|---:|---:|---|---|---:|---:|---:|",
    ]
    for r in extension:
        n = r["attempts"]
        lines.append(
            f"| {r['agents']} | {r['epochs']} | {r['branch']} | "
            f"{r['assignment']} | {r['trace_equal']}/{n} | "
            f"{r['zero_contrast_error']}/{n} | {r['witness_accept']}/{n} |"
        )
    (reports / "extension_results.md").write_text("\n".join(lines) + "\n")
    lines = [
        "# Shared-implementation diagnostic",
        "",
        "The same defect is active in execution and replay. These are TCB stress",
        "cases, outside the original accidental runtime-fault universe. All actual",
        "report files, unchanged contrasts and findings remain in the raw records.",
        "",
        "| Kind | Attempts | Triggered | Full reports | Witness reports | "
        "Changed contrast | Unchanged contrast |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, r in summary["common_mode"].items():
        lines.append(
            f"| {key} | {r['attempts']} | {r['triggered']} | {r['full_release']} | "
            f"{r['witness_release']} | {r.get('contrast_changed', 0)} | "
            f"{r.get('contrast_unchanged', 0)} |"
        )
    lines += [
        "",
        "Witnesses read serialized views, draws, commands, trades and accounts.",
        "They import no simulator, policy, RNG, replay or contract compiler.",
        "Their normative formulas and the serializer remain trusted. They do not",
        "validate complete matching behavior, the ontology, or every shared defect.",
    ]
    (reports / "common_mode_results.md").write_text("\n".join(lines) + "\n")
    lines = [
        "# Obligation-separation audit of unchanged original v1 records",
        "",
        "This is a secondary analysis of recorded findings, not a new registered",
        "fault campaign. Incomplete or non-evaluated replay cases are kept separate.",
        "",
        "| Fault | Attempts | Replay evaluated | Coord pass / replay fail | "
        "Replay pass / coord fail | Draw equal / contrast changed |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in summary["original_obligation_separation"]:
        lines.append(
            f"| {r['kind']} | {r['attempts']} | {r.get('replay_evaluated', 0)} | "
            f"{r.get('coord_pass_replay_fail', 0)} | "
            f"{r.get('replay_pass_coord_fail', 0)} | "
            f"{r.get('draw_equal_contrast_changed', 0)} |"
        )
    (reports / "obligation_separation.md").write_text("\n".join(lines) + "\n")
    (paper / "revision_manifest.json").write_text(
        json.dumps(
            {
                "input": str(source.relative_to(ROOT)),
                "input_sha256": digest(source),
                "presentation_source_sha256": digest(
                    ROOT / "scripts/cscwd2027_revision2/present.py"
                ),
                "outputs": {str(p.relative_to(ROOT)): digest(p) for p in outputs},
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"outputs": len(outputs), "values": values}))


if __name__ == "__main__":
    main()
