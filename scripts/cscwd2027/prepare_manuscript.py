"""Presentation-only extraction after the frozen scientific derivation.

This script adds no estimator, threshold, sample selection or fault classification.
Every displayed value comes directly from the frozen config/derived summary or
the separate engineering audit. Its own source hash is recorded with the outputs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    sources = [
        root / "configs/cscwd2027/study_v1.yaml",
        root / "configs/cscwd2027/freeze_manifest_v1.json",
        root / "results/cscwd2027/derived_v1/summary.json",
        root / "reports/cscwd2027/tcb_audit.json",
        root / "reports/cscwd2027/hardware_addendum.json",
    ]
    config, freeze, summary, tcb, hardware = [
        json.loads(p.read_text()) for p in sources
    ]
    if not summary["data_complete"]:
        raise RuntimeError("manuscript values require a complete derived grid")
    destination = root / "paper/cscwd2027/generated"
    destination.mkdir(parents=True, exist_ok=True)
    values = {
        "SmallAgents": config["agents"][0],
        "LargeAgents": config["agents"][1],
        "EpochCount": config["epochs"],
        "BranchEpoch": config["branch_epoch"],
        "ReplicateCount": config["replicates"],
        "ReferencePairs": freeze["counts"]["e1_reference"],
        "WarmupBlocks": config["cost_warmup_blocks"],
        "MeasuredBlocks": config["cost_measured_blocks"],
        "BootstrapResamples": config["bootstrap_resamples"],
        "EngineeringTests": tcb["junit"]["tests"],
        "HostLogicalCpus": hardware["logical_cpus"],
        "HostMemoryGiB": hardware["memory_gib"],
        "FaultClassCount": len(summary["faults"]),
    }
    for mode, macro in (
        ("keys_only", "KeysInvalidReleases"),
        ("coordination", "CoordInvalidReleases"),
        ("minus_replay", "NoReplayInvalidReleases"),
        ("minus_gate", "NoGateInvalidReleases"),
    ):
        values[macro] = sum(
            c["modes"][mode]["invalid_release"]["count"]
            for c in summary["faults"].values()
        )
    values["MissingMessageDrawAgreement"] = summary["faults"]["missing_message"][
        "draw_equal"
    ]
    full_cost = [c for c in summary["cost"] if c["mode"] == "full"]
    ratios = [c["paired_ratio"]["mean"] for c in full_cost]
    values["FullOverheadMin"] = f"{min(ratios):.2f}"
    values["FullOverheadMax"] = f"{max(ratios):.2f}"
    comparisons = [c for c in summary["cost"] if c["mode"] != "no_evidence"]
    values["CostPairedComparisons"] = sum(c["paired_ratio"]["n"] for c in comparisons)
    values["CostOutcomeMismatches"] = sum(c["outcome_mismatches"] for c in comparisons)
    outputs = []

    def write(name: str, lines: list[str]) -> None:
        path = destination / name
        path.write_text(
            "% Generated presentation; no new scientific analysis.\n"
            + "\n".join(lines)
            + "\n"
        )
        outputs.append(path)

    write(
        "presentation_numbers.tex",
        [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in values.items()],
    )
    labels = {
        "draw_equal": "Draw records",
        "trace_equal": "Command/state trace",
        "account_order_fill_equal": "Accounts, orders and fills",
        "zero_contrast_error": "Zero contrast discrepancy",
        "accepted": "Full acceptance",
    }
    write(
        "partition_table.tex",
        [
            "\\begin{tabular}{lr}\\toprule",
            "Exact comparison & Matched pairs\\\\\\midrule",
        ]
        + [
            f"{label} & {summary['e1'][key]}/{summary['e1']['attempts']}\\\\"
            for key, label in labels.items()
        ]
        + ["\\bottomrule\\end{tabular}"],
    )
    modes = {
        "keys_only": "Keys only",
        "receipts": "Event receipts",
        "coordination": "Coordination",
        "full": "Full",
        "conventional": "Conventional full",
        "minus_ownership_version": "Minus ownership/version",
        "minus_expected": "Minus coverage checks",
        "minus_replay": "Minus replay",
        "minus_gate": "Minus release gate",
    }
    lines = [
        "\\begin{tabular}{lrrr}\\toprule",
        "Path & Detected & Invalid release & Valid acceptance\\\\\\midrule",
    ]
    for mode, label in modes.items():
        counts = [cell["modes"][mode] for cell in summary["faults"].values()]
        detected = sum(c["detected"] for c in counts)
        released = sum(c["invalid_release"]["count"] for c in counts)
        total = sum(c["invalid_release"]["denominator"] for c in counts)
        valid = sum(c["modes"][mode]["count"] for c in summary["controls"].values())
        nvalid = sum(c["attempts"] for c in summary["controls"].values())
        lines.append(
            f"{label} & {detected}/{total} & {released}/{total} & {valid}/{nvalid}\\\\"
        )
    write("ablation_table.tex", lines + ["\\bottomrule\\end{tabular}"])
    fault_labels = {
        "task_duplication": "Task duplicate",
        "task_omission": "Task missing",
        "wrong_owner": "Wrong owner",
        "duplicate_effect": "Effect duplicate",
        "missing_message": "Message missing",
        "logical_order": "Commit order",
        "payload_mutation": "Payload",
        "stale_plan": "Stale plan",
    }
    lines = [
        "\\begin{tabular}{lrr}\\toprule",
        "Fault & Changed/same/incomplete & Full release\\\\\\midrule",
    ]
    for kind, label in fault_labels.items():
        cell = summary["faults"][kind]
        release = cell["modes"]["full"]["invalid_release"]
        lines.append(
            f"{label} & {cell['changed']}/{cell['unchanged']}/{cell['incomplete']} & "
            f"{release['count']}/{release['denominator']}\\\\"
        )
    write("fault_outcomes_compact.tex", lines + ["\\bottomrule\\end{tabular}"])
    for agents in config["agents"]:
        lines = [
            "\\begin{tabular}{rlrrrr}\\toprule",
            "Workers & Mode & Median (s) & Ratio & 95\\% interval & KiB\\\\\\midrule",
        ]
        for cell in summary["cost"]:
            if cell["agents"] != agents:
                continue
            ratio = cell["paired_ratio"]
            low, high = ratio["ci95"]
            label = {
                "no_evidence": "No evidence",
                "keys_only": "Keys",
                "receipts": "Receipts",
                "full": "Full",
                "conventional": "Conventional",
            }[cell["mode"]]
            seconds = cell["end_to_end_seconds"]["median"]
            kib = cell["gzip_capsule_bytes"]["mean"] / 1024
            lines.append(
                f"{cell['workers']} & {label} & {seconds:.2f} & {ratio['mean']:.2f} & "
                f"[{low:.2f}, {high:.2f}] & {kib:.1f}\\\\"
            )
        write(f"cost_a{agents}.tex", lines + ["\\bottomrule\\end{tabular}"])
    write(
        "tcb_table.tex",
        [
            "\\begin{tabular}{lrr}\\toprule",
            "Component & LOC & AST statements\\\\\\midrule",
        ]
        + [
            f"{r['component'].replace('&', 'and')} & "
            f"{r['nonblank_noncomment_lines']} & {r['ast_statements']}\\\\"
            for r in tcb["components"]
        ]
        + ["\\bottomrule\\end{tabular}"],
    )
    manifest = {
        "purpose": "presentation extraction; no new estimators",
        "inputs": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "outputs": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in outputs
        },
    }
    (destination / "presentation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
