"""Prefrozen statistics: exact equality/counts and paired block cost intervals."""

from __future__ import annotations

import argparse
import json
import random
from statistics import mean, median
from typing import Any

from policytwin.coordination.faults import FAULTS
from policytwin.coordination.release_gate import MODES
from policytwin.provenance import canonical_json_bytes
from scripts.cscwd2027.freeze_study import DEST, ROOT, check
from scripts.cscwd2027.study_design import CONFIG, COST_MODES


def quantile(values: list[float], probability: float) -> float:
    """Linear interpolation of ordered values, including singleton fixtures."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def bootstrap_mean(values: list[float], seed: int = 20271025) -> dict[str, Any]:
    """Percentile 95% interval from resampling paired per-block ratios."""
    if not values:
        return {"n": 0, "mean": None, "ci95": None}
    rng = random.Random(seed)
    simulations = [mean(rng.choices(values, k=len(values))) for _ in range(2000)]
    return {
        "n": len(values),
        "mean": mean(values),
        "ci95": [quantile(simulations, 0.025), quantile(simulations, 0.975)],
    }


def rate(rows: list[dict[str, Any]], mode: str, field: str) -> dict[str, int]:
    """Retain the fixed attempted denominator; absent decisions never count positive."""
    return {
        "count": sum(
            bool(r.get("decisions", {}).get(mode, {}).get(field)) for r in rows
        ),
        "denominator": len(rows),
    }


def summarize(
    rows: list[dict[str, Any]], expected: list[dict[str, Any]]
) -> dict[str, Any]:
    """Generate all scientific totals from per-attempt files, including failures."""
    expected_ids = {r["id"] for r in expected}
    present_ids = {r["id"] for r in rows}
    e1 = [r for r in rows if r["design"]["group"] == "e1"]
    faults = [r for r in rows if r["design"]["group"] == "e2"]
    controls = [r for r in rows if r["design"]["group"] == "e2b"]
    cost = [
        r for r in rows if r["design"]["group"] == "e3" and not r["design"]["warmup"]
    ]
    failures = [r["id"] for r in rows if r["status"] != "completed"]
    summary: dict[str, Any] = {
        "expected_attempts": len(expected),
        "recorded_attempts": len(rows),
        "missing_ids": sorted(expected_ids - present_ids),
        "failed_ids": failures,
        "unexpected_ids": sorted(present_ids - expected_ids),
        "analysis": {
            "cost_interval": "95% percentile bootstrap of paired block ratios",
            "bootstrap_seed": CONFIG["bootstrap_seed"],
            "resamples": CONFIG["bootstrap_resamples"],
            "fault_interval": (
                "none: designed finite fault universe, not random classes"
            ),
        },
        "e1": {
            "attempts": len(e1),
            **{
                name: sum(bool(r.get("comparison", {}).get(name)) for r in e1)
                for name in ("draw_equal", "trace_equal", "account_order_fill_equal")
            },
            "zero_contrast_error": sum(
                r.get("comparison", {}).get("contrast_error") == 0 for r in e1
            ),
            "accepted": rate(e1, "full", "numerical_file_exists")["count"],
        },
        "e2": {
            "attempts": len(faults),
            "triggered": sum(r.get("injection_triggered", False) for r in faults),
            "incomplete": sum(
                not r.get("comparison", {}).get("complete", False) for r in faults
            ),
            "numerical_changed": sum(
                r.get("comparison", {}).get("numerical_changed") is True for r in faults
            ),
            "numerically_unchanged": sum(
                r.get("comparison", {}).get("numerical_changed") is False
                for r in faults
            ),
        },
        "faults": {},
        "controls": {},
        "cost": [],
    }
    for kind in FAULTS:
        subset = [r for r in faults if r["design"]["fault"]["kind"] == kind]
        triggered = [r for r in subset if r.get("injection_triggered")]
        summary["faults"][kind] = {
            "attempts": len(subset),
            "triggered": len(triggered),
            "changed": sum(
                r.get("comparison", {}).get("numerical_changed") is True for r in subset
            ),
            "unchanged": sum(
                r.get("comparison", {}).get("numerical_changed") is False
                for r in subset
            ),
            "incomplete": sum(
                not r.get("comparison", {}).get("complete", False) for r in subset
            ),
            "draw_equal": sum(
                r.get("comparison", {}).get("draw_equal", False) for r in subset
            ),
            "modes": {
                mode: {
                    "detected": sum(
                        bool(r.get("decisions", {}).get(mode, {}).get("findings"))
                        for r in triggered
                    ),
                    "invalid_release": rate(triggered, mode, "numerical_file_exists"),
                    "contained": sum(
                        not r.get("decisions", {})
                        .get(mode, {})
                        .get("numerical_file_exists", False)
                        for r in triggered
                    ),
                    "false_negative_ids": [
                        r["id"]
                        for r in triggered
                        if not r.get("decisions", {}).get(mode, {}).get("findings")
                    ],
                }
                for mode in MODES
            },
        }
    for transport in ("normal", "delay", "reverse"):
        subset = [r for r in controls if r["design"]["transport"] == transport]
        summary["controls"][transport] = {
            "attempts": len(subset),
            "exact_trace": sum(
                r.get("comparison", {}).get("trace_equal", False) for r in subset
            ),
            "modes": {
                mode: rate(subset, mode, "numerical_file_exists") for mode in MODES
            },
        }
    for agents in (6, 24):
        for workers in (1, 2, 4):
            cell = [
                r
                for r in cost
                if r["design"]["spec"]["agent_count"] == agents
                and r["design"]["plan"]["worker_count"] == workers
                and r["status"] == "completed"
            ]
            base = {
                r["design"]["block"]: r
                for r in cell
                if r["design"]["cost_mode"] == "no_evidence"
            }
            for mode in COST_MODES:
                selected = [r for r in cell if r["design"]["cost_mode"] == mode]
                paired = [r for r in selected if r["design"]["block"] in base]
                ratios = [
                    r["end_to_end_seconds"]
                    / base[r["design"]["block"]]["end_to_end_seconds"]
                    for r in paired
                ]
                row = {
                    "agents": agents,
                    "workers": workers,
                    "mode": mode,
                    "n": len(selected),
                    "paired_ratio": bootstrap_mean(ratios),
                    "outcome_mismatches": sum(
                        r["outcome_sha256"]
                        != base[r["design"]["block"]]["outcome_sha256"]
                        for r in paired
                    ),
                }
                for field in (
                    "end_to_end_seconds",
                    "execution_seconds",
                    "startup_seconds",
                    "market_transition_seconds",
                    "serialization_storage_seconds",
                    "raw_capsule_bytes",
                    "gzip_capsule_bytes",
                    "reported_bytes",
                ):
                    values = [r[field] for r in selected]
                    row[field] = (
                        {
                            "mean": mean(values),
                            "median": median(values),
                            "q25": quantile(values, 0.25),
                            "q75": quantile(values, 0.75),
                        }
                        if values
                        else None
                    )
                summary["cost"].append(row)
    summary["full_invalid_releases"] = rate(
        [r for r in faults if r.get("injection_triggered")],
        "full",
        "numerical_file_exists",
    )
    summary["data_complete"] = (
        not summary["missing_ids"] and not failures and not summary["unexpected_ids"]
    )
    return summary


def write_outputs(summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """Persist derived data; only complete grids can feed manuscript numbers."""
    destination = ROOT / "results/cscwd2027/derived_v1"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "summary.json").write_bytes(canonical_json_bytes(summary))
    (destination / "cost_observations.json").write_bytes(
        canonical_json_bytes(
            [
                {
                    k: r[k]
                    for k in (
                        "id",
                        "design",
                        "status",
                        "end_to_end_seconds",
                        "gzip_capsule_bytes",
                    )
                }
                for r in rows
                if r["design"]["group"] == "e3" and r["status"] == "completed"
            ]
        )
    )
    report = ROOT / "reports/cscwd2027/formal_study_results.md"
    lines = [
        "# CSCWD formal-study results",
        "",
        f"Data complete: {summary['data_complete']}.",
        f"Recorded/expected attempts: {summary['recorded_attempts']}/"
        f"{summary['expected_attempts']}.",
        "",
        "All failures, non-triggered injections and numerical nulls are retained.",
        "",
        "## Partition equality",
        "",
        json.dumps(summary["e1"], indent=2),
        "",
        "## Fault attempts",
        "",
        json.dumps(summary["e2"], indent=2),
        "",
        "| Fault | Attempts | Changed / same / incomplete | Full invalid releases |",
        "|---|---:|---:|---:|",
    ]
    for name, cell in summary["faults"].items():
        released = cell["modes"]["full"]["invalid_release"]
        lines.append(
            f"| {name} | {cell['attempts']} | {cell['changed']} / "
            f"{cell['unchanged']} / {cell['incomplete']} | "
            f"{released['count']}/{released['denominator']} |"
        )
    report.write_text("\n".join(lines) + "\n")
    if summary["data_complete"]:
        paper = ROOT / "paper/cscwd2027/generated"
        paper.mkdir(parents=True, exist_ok=True)
        values = {
            "StudyAttempts": summary["recorded_attempts"],
            "PartitionPairs": summary["e1"]["attempts"],
            "PartitionEqual": summary["e1"]["trace_equal"],
            "FaultAttempts": summary["e2"]["attempts"],
            "FaultTriggered": summary["e2"]["triggered"],
            "FullInvalidReleases": summary["full_invalid_releases"]["count"],
            "NumericalChanged": summary["e2"]["numerical_changed"],
            "NumericalUnchanged": summary["e2"]["numerically_unchanged"],
            "IncompleteFaults": summary["e2"]["incomplete"],
            "ControlPairs": sum(c["attempts"] for c in summary["controls"].values()),
            "ControlAccepted": sum(
                c["modes"]["full"]["count"] for c in summary["controls"].values()
            ),
        }
        (paper / "numbers.tex").write_text(
            "% Generated only from complete structured artifacts.\n"
            + "\n".join(
                f"\\newcommand{{\\{key}}}{{{value}}}" for key, value in values.items()
            )
            + "\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    check()
    expected = json.loads((DEST / "schedule_v1.json").read_text())
    rows = [
        json.loads(p.read_text())
        for p in sorted((ROOT / "results/cscwd2027/study_v1").glob("*/result.json"))
    ]
    summary = summarize(rows, expected)
    if not summary["data_complete"] and not args.allow_partial:
        raise RuntimeError(
            "formal grid incomplete; use --allow-partial only for status"
        )
    write_outputs(summary, rows)
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "data_complete",
                    "recorded_attempts",
                    "expected_attempts",
                    "failed_ids",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
