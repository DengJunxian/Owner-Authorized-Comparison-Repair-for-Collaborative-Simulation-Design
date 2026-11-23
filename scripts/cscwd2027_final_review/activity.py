"""Post-hoc workload-activity audit of the unchanged revision2 references.

These descriptive checks were requested during final review, after the original
outcomes existed. They are not additional preregistered endpoints or new runs.
"""

from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.cscwd2027_final_review.protection import ROOT, digest


def world_activity(world: dict[str, Any]) -> dict[str, int]:
    """Count actual active tasks and fill-bearing epochs from event identities."""
    epochs = world["epochs"]
    causes = {t["cause_event_key"] for t in world["trades"]}
    return {
        "world_epochs": len(epochs),
        "fill_bearing_epochs": sum(
            any(c["event_key"]["value"] in causes for c in row["applied_commands"])
            for row in epochs
        ),
        "active_tasks": sum(
            result["status"] == "active" for row in epochs for result in row["results"]
        ),
        "inactive_tasks": sum(
            result["status"] == "inactive"
            for row in epochs
            for result in row["results"]
        ),
        "fills": len(world["trades"]),
    }


def pair_activity(capsule: dict[str, Any]) -> dict[str, int]:
    """Measure activity and intervention sensitivity without an external oracle."""
    worlds = capsule["worlds"]
    baseline, policy = worlds["baseline"], worlds["policy"]
    totals: Counter[str] = Counter()
    for world in (baseline, policy):
        totals.update(world_activity(world))

    def projection(world: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {"commands": row["applied_commands"], "state": row["state_after"]}
            for row in world["epochs"]
        ]

    def balances(world: dict[str, Any]) -> list[tuple[str, int, int]]:
        return sorted(
            (a["trader_id"], a["asset"], a["cash"])
            for a in world["final_state"]["accounts"]
        )

    totals.update(
        {
            "pairs": 1,
            "nonzero_paired_contrasts": capsule["metrics"]["paired_traded_notional"]
            != 0,
            "different_world_traces": projection(baseline) != projection(policy),
            "different_terminal_balances": balances(baseline) != balances(policy),
        }
    )
    return dict(totals)


def analyze() -> dict[str, Any]:
    """Read all 72 reference pairs once; do not count worker maps as new samples."""
    source = ROOT / "results/cscwd2027/revision2/derived/summary.json"
    summary = json.loads(source.read_text())
    if (
        not summary["data_complete"]
        or summary["failed"]
        or summary["integrity_findings"]
    ):
        raise RuntimeError("complete original grid required")
    schedule_path = ROOT / "configs/cscwd2027/revision2/schedule.json"
    schedule = json.loads(schedule_path.read_text())
    groups: dict[tuple[int, int, str], Counter[str]] = {}
    raw = {}
    for design in schedule:
        if design["study"] != "extension" or design["processes"]:
            continue
        name = f"results/cscwd2027/revision2/formal/{design['id']}/capsule.json.gz"
        path = ROOT / name
        sha = digest(path)
        if summary["raw_file_manifest"].get(name) != sha:
            raise RuntimeError(f"unbound input capsule: {name}")
        raw[name] = sha
        capsule = json.loads(gzip.decompress(path.read_bytes()))
        if capsule["spec"] != design["spec"]:
            raise RuntimeError("reference specification mismatch")
        key = (
            capsule["spec"]["agent_count"],
            capsule["spec"]["epoch_count"],
            capsule["spec"]["branch"],
        )
        groups.setdefault(key, Counter()).update(pair_activity(capsule))
    return {
        "label": "post-hoc descriptive final-review audit; no new simulation runs",
        "selection": "all extension reference pairs; worker variants excluded",
        "reference_pairs": len(raw),
        "cells": [
            {"agents": a, "epochs": h, "branch": b, **dict(c)}
            for (a, h, b), c in sorted(groups.items())
        ],
        "source_summary_sha256": digest(source),
        "source_schedule_sha256": digest(schedule_path),
        "analysis_source_sha256": digest(Path(__file__)),
        "raw_inputs": raw,
        "definitions": {
            "fill_bearing_epoch": (
                "at least one final fill caused by a command committed "
                "in that world epoch"
            ),
            "world_trace_difference": (
                "applied commands or epoch states differ between baseline and policy"
            ),
            "terminal_balance_difference": (
                "at least one terminal cash or asset balance differs"
            ),
            "nonzero_contrast": (
                "policy minus baseline total traded notional is not zero"
            ),
        },
    }


def main() -> None:
    """Generate descriptive artifacts and manuscript macros in new locations."""
    data = analyze()
    results = ROOT / "results/cscwd2027/final_review/derived"
    reports = ROOT / "reports/cscwd2027/final_review"
    generated = ROOT / "paper/cscwd2027_final_review/generated"
    for folder in (results, reports, generated):
        folder.mkdir(parents=True, exist_ok=True)
    result_path = results / "activity_summary.json"
    result_path.write_text(json.dumps(data, indent=2) + "\n")
    cells = data["cells"]
    latency = [c for c in cells if c["branch"] == "latency_reorder"]
    values = {
        "ActivityReferencePairs": data["reference_pairs"],
        "ActivityWorldEpochs": sum(c["world_epochs"] for c in cells),
        "ActivityFilledEpochs": sum(c["fill_bearing_epochs"] for c in cells),
        "ActivityLatencyPairs": sum(c["pairs"] for c in latency),
        "ActivityLatencyNonzero": sum(c["nonzero_paired_contrasts"] for c in latency),
    }
    numbers_path = generated / "final_review_numbers.tex"
    numbers_path.write_text(
        "% Generated post-hoc descriptive audit, not new registered endpoints.\n"
        + "".join(f"\\newcommand{{\\{k}}}{{{v:,}}}\n" for k, v in values.items())
    )
    lines = [
        "\\begin{tabular}{llrr}\\toprule",
        "Cell & Branch & Filled epochs & Nonzero $\\Delta$\\\\\\midrule",
    ]
    md = [
        "# Descriptive activity audit of existing references",
        "",
        "Post-hoc final-review analysis; no new runs or registered endpoints.",
        "All references are included once; worker variants are not new samples.",
        "Fill-bearing epochs have a trade caused by an applied command.",
        "Trace difference compares applied commands and epoch states across worlds.",
        "",
        "| Agents | Epochs | Branch | Reference pairs | Fill-bearing world epochs "
        "| World trace differs | Nonzero paired contrast "
        "| Different terminal balances |",
        "|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "agent_exit": "Exit",
        "cancellation": "Cancel/replace",
        "latency_reorder": "Latency",
    }
    for c in cells:
        n = c["pairs"]
        e = c["world_epochs"]
        lines.append(
            f"{c['agents']}/{c['epochs']} & {labels[c['branch']]} & "
            f"{c['fill_bearing_epochs']}/{e} & "
            f"{c['nonzero_paired_contrasts']}/{n}\\\\"
        )
        md.append(
            f"| {c['agents']} | {c['epochs']} | {c['branch']} | {n} | "
            f"{c['fill_bearing_epochs']}/{e} | {c['different_world_traces']}/{n} | "
            f"{c['nonzero_paired_contrasts']}/{n} | "
            f"{c['different_terminal_balances']}/{n} |"
        )
    table_path = generated / "activity_table.tex"
    table_path.write_text("\n".join(lines + ["\\bottomrule\\end{tabular}"]) + "\n")
    (reports / "workload_activity_audit.md").write_text("\n".join(md) + "\n")
    (generated / "final_review_manifest.json").write_text(
        json.dumps(
            {
                "input": str(result_path.relative_to(ROOT)),
                "input_sha256": digest(result_path),
                "outputs": {
                    str(p.relative_to(ROOT)): digest(p)
                    for p in (numbers_path, table_path)
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(values))


if __name__ == "__main__":
    main()
