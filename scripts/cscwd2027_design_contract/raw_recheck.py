"""Post-hoc read-only arithmetic audit from trades and actual comparison files.

No simulation, decision threshold, sample selection or frozen endpoint changes.
"""

from __future__ import annotations

import gzip
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any

from scripts.cscwd2027_design_contract.deliver import REPORT
from scripts.cscwd2027_design_contract.freeze import DEST, check
from scripts.cscwd2027_design_contract.protection import digest
from scripts.cscwd2027_design_contract.store import RESULTS


def main() -> None:
    """Audit all paired metrics, means, membership and decision consequences."""
    check()
    jobs = json.loads((DEST / "attempts.json").read_text())
    contracts = json.loads((DEST / "contracts.json").read_text())
    handoffs = {r["id"]: r for r in json.loads((DEST / "handoffs.json").read_text())}
    evaluation = json.loads((RESULTS / "evaluation/results.json").read_text())
    findings: list[str] = []
    values = {}
    for job in jobs:
        folder = RESULTS / "objects" / job["id"]
        cap = json.loads(gzip.decompress((folder / "capsule.json.gz").read_bytes()))
        totals = {}
        for w in ("baseline", "policy"):
            totals[w] = sum(
                t["price"] * t["quantity"] for t in cap["worlds"][w]["trades"]
            )
            if totals[w] != cap["worlds"][w]["traded_notional"]:
                findings.append(job["id"] + ":world_notional")
        metric = totals["policy"] - totals["baseline"]
        result = json.loads((folder / "result.json").read_text())
        if (
            metric != result["metric"]
            or metric != cap["metrics"]["paired_traded_notional"]
        ):
            findings.append(job["id"] + ":paired_metric")
        values[job["id"]] = metric
    checked_reports = 0

    def means(numeric: dict[str, Any], label: str) -> dict[str, Fraction]:
        nonlocal checked_reports
        checked_reports += 1
        groups: dict[str, list[int]] = {}
        for m in numeric["effective_bindings"]:
            groups.setdefault(m["candidate"], []).append(values[m["object_id"]])
        result = {k: Fraction(sum(v), len(v)) for k, v in groups.items()}
        order = sorted(result, key=lambda c: (-result[c], c))
        if result != {k: Fraction(*v) for k, v in numeric["means"].items()}:
            findings.append(label + ":means")
        if numeric["top1"] != order[0] or numeric["top2"] != sorted(order[:2]):
            findings.append(label + ":selection")
        if numeric["group_sizes"] != {k: len(v) for k, v in groups.items()}:
            findings.append(label + ":group_sizes")
        return result

    reference_means = {}
    for sid in contracts:
        for version in (1, 2):
            path = (
                RESULTS
                / f"evaluation/references/{sid}/v{version}/numerical_comparison.json"
            )
            reference_means[sid, version] = means(
                json.loads(path.read_text()), f"{sid}:v{version}"
            )
    counts: dict[str, Counter[str]] = {}
    for row in evaluation["handoffs"]:
        path = (
            RESULTS
            / f"evaluation/handoffs/{row['case_id']}/{row['mode']}"
            / "numerical_comparison.json"
        )
        if path.is_file() != row["decision"]["numerical_file_exists"]:
            findings.append(row["case_id"] + ":file_presence")
        counter = counts.setdefault(row["mode"], Counter())
        counter["sets"] += 1
        if not path.is_file():
            if row["consequences"]["regret"] is not None:
                findings.append(row["case_id"] + ":abstention_regret")
            continue
        counter["reports"] += 1
        numeric = json.loads(path.read_text())
        means(numeric, row["case_id"] + ":" + row["mode"])
        version = handoffs[row["case_id"]]["version"]
        contract = contracts[row["session"]][str(version)]
        wanted = Counter(
            (s["candidate"], s["replicate"], s["scientific_key"])
            for s in contract["slots"]
        )
        observed = Counter(
            (m["candidate"], m["replicate"], m["scientific_key"])
            for m in numeric["effective_bindings"]
        )
        valid = wanted == observed and all(
            m["local_pass"] and m["plan_sha256"] in contract["allowed_plans"]
            for m in numeric["effective_bindings"]
        )
        if valid != row["output_panel_valid"]:
            findings.append(row["case_id"] + ":membership")
        counter["unauthorized"] += not valid
        ref = reference_means[row["session"], version]
        ref_order = sorted(ref, key=lambda c: (-ref[c], c))
        regret = max(ref.values()) - ref[numeric["top1"]]
        actual = {
            "top1_disagreement": numeric["top1"] != ref_order[0],
            "top2_disagreement": numeric["top2"] != sorted(ref_order[:2]),
            "optimal_set_error": regret > 0,
            "regret": [regret.numerator, regret.denominator],
        }
        if any(row["consequences"][k] != v for k, v in actual.items()):
            findings.append(row["case_id"] + ":consequences")
        for k in ("top1_disagreement", "top2_disagreement", "optimal_set_error"):
            counter[k] += actual[k]
    value = {
        "label": "post-hoc arithmetic and file-binding audit, not new endpoints",
        "pass": not findings,
        "findings": findings,
        "raw_capsules_recomputed": len(values),
        "actual_numerical_files_checked": checked_reports,
        "mode_counts": {k: dict(v) for k, v in counts.items()},
        "source_sha256": digest(Path(__file__)),
        "freeze_sha256": digest(DEST / "freeze.json"),
        "evaluation_sha256": digest(RESULTS / "evaluation/results.json"),
        "scope": "same-repository arithmetic audit, not independent simulation",
    }
    (REPORT / "raw_arithmetic_recheck.json").write_text(
        json.dumps(value, indent=2) + "\n"
    )
    print(
        json.dumps(
            {k: value[k] for k in ("pass", "raw_capsules_recomputed", "findings")}
        )
    )


if __name__ == "__main__":
    main()
