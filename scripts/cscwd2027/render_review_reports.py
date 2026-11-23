"""Present the complete frozen summaries and retained negative cases for review."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    """Produce reviewer-readable reports without selecting new scientific samples."""
    summary = json.loads(
        (ROOT / "results/cscwd2027/derived_v1/summary.json").read_text()
    )
    if not summary["data_complete"]:
        raise RuntimeError("review reports require the complete frozen analysis")
    records = [
        json.loads(p.read_text())
        for p in sorted((ROOT / "results/cscwd2027/study_v1").glob("*/result.json"))
    ]
    destination = ROOT / "reports/cscwd2027"

    def write(name: str, lines: list[str]) -> None:
        (destination / name).write_text("\n".join(lines) + "\n")

    common = [
        "",
        "Generated from `results/cscwd2027/derived_v1/summary.json` and retained",
        "attempt receipts. This is presentation of the frozen analysis, not a new",
        "experiment, threshold or choice of samples.",
        "",
    ]
    e1 = summary["e1"]
    write(
        "partition_results.md",
        ["# Partition study"]
        + common
        + [
            "| Comparison | Exact agreements |",
            "|---|---:|",
        ]
        + [f"| {k} | {v}/{e1['attempts']} |" for k, v in e1.items() if k != "attempts"]
        + [
            "",
            "Worker variants share matched scenarios. These are finite exact checks",
            "under a fixed logical schedule, not independent replications or proof of",
            "arbitrary scheduler/platform equivalence. The reference shares",
            "the policy and CLOB implementation.",
        ],
    )
    fault_lines = (
        ["# Runtime faults and valid-transport controls"]
        + common
        + [
            "| Operation | Triggered/attempted | Changed | Same | "
            "Incomplete | Full release |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for kind, cell in summary["faults"].items():
        release = cell["modes"]["full"]["invalid_release"]
        fault_lines.append(
            f"| {kind} | {cell['triggered']}/{cell['attempts']} | {cell['changed']} | "
            f"{cell['unchanged']} | {cell['incomplete']} | "
            f"{release['count']}/{release['denominator']} |"
        )
    fault_lines += [
        "",
        "A numerical null means a completed contrast equals its clean reference.",
        "Native incomplete runs are not assigned zero error. Duplicate tasks/effects",
        "can be stopped by the existing CLOB before the added evidence checks.",
        "",
        "| Valid transport | Exact trace | Full accepted |",
        "|---|---:|---:|",
    ]
    for kind, cell in summary["controls"].items():
        fault_lines.append(
            f"| {kind} | {cell['exact_trace']}/{cell['attempts']} | "
            f"{cell['modes']['full']['count']}/{cell['attempts']} |"
        )
    write("fault_control_results.md", fault_lines)
    totals: dict[str, Any] = {}
    for mode in next(iter(summary["faults"].values()))["modes"]:
        cells = [c["modes"][mode] for c in summary["faults"].values()]
        totals[mode] = {
            "detected": sum(c["detected"] for c in cells),
            "invalid_release": sum(c["invalid_release"]["count"] for c in cells),
            "triggered": sum(c["invalid_release"]["denominator"] for c in cells),
            "valid_accepted": sum(
                c["modes"][mode]["count"] for c in summary["controls"].values()
            ),
            "false_negative_ids": sorted(
                identifier for c in cells for identifier in c["false_negative_ids"]
            ),
        }
    same: list[str] = []
    different: list[str] = []
    for record in records:
        decisions = record.get("decisions", {})
        if not {"full", "conventional"} <= decisions.keys():
            continue
        fields = (
            "findings",
            "diagnostic_pass",
            "release_authorized",
            "numerical_file_exists",
        )
        target = (
            same
            if all(decisions["full"][f] == decisions["conventional"][f] for f in fields)
            else different
        )
        target.append(record["id"])
    derived = {
        "modes": totals,
        "conventional_equal_ids": same,
        "conventional_different_ids": different,
    }
    (destination / "ablation_review_data.json").write_text(
        json.dumps(derived, indent=2) + "\n"
    )
    lines = (
        ["# Equal-obligation control and leave-one-check-out results"]
        + common
        + [
            "| Report path | Detected incl. native | "
            "Invalid released | Valid accepted |",
            "|---|---:|---:|---:|",
        ]
    )
    control_total = sum(c["attempts"] for c in summary["controls"].values())
    for mode, cell in totals.items():
        lines.append(
            f"| {mode} | {cell['detected']}/{cell['triggered']} | "
            f"{cell['invalid_release']}/{cell['triggered']} | "
            f"{cell['valid_accepted']}/{control_total} |"
        )
    lines += [
        "",
        "Full/conventional decision agreement: "
        f"{len(same)}/{len(same) + len(different)}.",
        "The conventional adapter calls the same primitives. Its tie is an equivalence",
        "control, not evidence of superiority or independent implementation.",
        "The coverage deletion removes coverage findings only; expected identities and",
        "logical commitments remain available to other predicates. It does not remove",
        "the entire compiler. False-negative IDs for every mode remain in",
        "`ablation_review_data.json` and the frozen summary.",
    ]
    write("strong_baseline_and_ablation.md", lines)
    costs = (
        ["# Cold end-to-end evidence cost"]
        + common
        + [
            "| Agents | Workers | Mode | Median seconds | IQR seconds | "
            "Mean paired ratio [95% interval] | Mean gzip KiB |",
            "|---:|---:|---|---:|---:|---:|---:|",
        ]
    )
    for c in summary["cost"]:
        t = c["end_to_end_seconds"]
        ratio = c["paired_ratio"]
        costs.append(
            f"| {c['agents']} | {c['workers']} | {c['mode']} | {t['median']:.4f} | "
            f"[{t['q25']:.4f}, {t['q75']:.4f}] | {ratio['mean']:.3f} "
            f"[{ratio['ci95'][0]:.3f}, {ratio['ci95'][1]:.3f}] | "
            f"{c['gzip_capsule_bytes']['mean'] / 1024:.2f} |"
        )
    costs += [
        "",
        "## Descriptive stage audit",
        "",
        "The following medians summarize already-recorded stage clocks using the same",
        "measured blocks. They are post-run descriptive presentation, not new primary",
        "estimators or a claim that overlapping timers add to the critical path.",
        "",
        "| Agents | Workers | Execute | Serialize/store | Events | "
        "Coordination | Replay | Lock/metric/cutoff |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    stages = []
    for agents in (6, 24):
        for workers in (1, 2, 4):
            selected = [
                r
                for r in records
                if r["design"]["group"] == "e3"
                and not r["design"]["warmup"]
                and r["design"]["cost_mode"] == "full"
                and r["design"]["spec"]["agent_count"] == agents
                and r["design"]["plan"]["worker_count"] == workers
            ]
            row = {"agents": agents, "workers": workers, "n": len(selected)}
            for field in ("execution_seconds", "serialization_storage_seconds"):
                row[field] = median(r[field] for r in selected)
            for field in (
                "event_conformance",
                "coordination_conformance",
                "reexecution",
                "lock_metric_cutoff",
            ):
                row[field] = median(
                    r["decisions"]["full"]["component_seconds"][field] for r in selected
                )
            stages.append(row)
            values = [
                row[k]
                for k in (
                    "execution_seconds",
                    "serialization_storage_seconds",
                    "event_conformance",
                    "coordination_conformance",
                    "reexecution",
                    "lock_metric_cutoff",
                )
            ]
            costs.append(
                f"| {agents} | {workers} | "
                + " | ".join(f"{v:.4f}" for v in values)
                + " |"
            )
    (destination / "cost_stage_audit.json").write_text(
        json.dumps(stages, indent=2) + "\n"
    )
    costs += [
        "",
        "Fresh spawn, IPC, native CLOB checks, compression/storage and actual report",
        "writing are included in cold latency. Registration and reference comparison",
        "are outside that boundary. Worker elapsed sums overlap with coordinator time.",
        "The baseline uses the same RNG and native CLOB invariants; this is not an RNG",
        "algorithm benchmark. Intervals describe paired block variation on one host.",
    ]
    write("cost_results.md", costs)
    write(
        "revised_claim_evidence_matrix.md",
        ["# Claim–evidence map"]
        + common
        + [
            "| Claim | Evidence | Boundary |",
            "|---|---|---|",
            "| Static worker partitions preserve tested scientific traces | "
            f"E1 {e1['trace_equal']}/{e1['attempts']}; `partition_results.md`; "
            "Proposition 1 | Fixed schedule and same-repository reference |",
            "| Full report path blocks the declared runtime violations | "
            f"{summary['full_invalid_releases']['count']}/"
            f"{summary['full_invalid_releases']['denominator']} invalid files; "
            "`fault_control_results.md` | Native containment is included and "
            "separated; finite fault universe |",
            "| Draw agreement alone does not establish correct message effects | "
            "Missing-message rows, exact draw comparison, changed contrasts | "
            "This concrete workload/fault design |",
            "| Replay and release authorization address distinct obligations | "
            "`strong_baseline_and_ablation.md`; actual report files | "
            "Other checks can overlap; no universal minimality result |",
            "| Equal obligations give equal observed decisions | "
            f"{len(same)}/{len(same) + len(different)} full/conventional agreement | "
            "Shared primitive composition, not an independent implementation |",
            "| Evidence has an end-to-end measured cost | `cost_results.md`; "
            "E3 measured-block observations | Cold latency, one host, "
            "background load uncontrolled |",
            "| Acceptance yields a conditional input law | Proposition 2; "
            "exact rational fixtures | No unconditional unbiasedness claim "
            "from an all-pass batch |",
            "| Historical evidence is preserved | Protected baseline plus "
            "original metadata/results checks | Older v2 source-byte gap "
            "is retained, not retroactively repaired |",
            "",
            "Not claimed: SOTA, malicious-fault security, real market validity,",
            "cross-platform equivalence, external reproduction, universal speedup, or",
            "exclusive capability over a conventional equal-obligation implementation.",
        ],
    )
    print(
        json.dumps(
            {"reports_generated": 5, "full_conventional_disagreements": len(different)}
        )
    )


if __name__ == "__main__":
    main()
