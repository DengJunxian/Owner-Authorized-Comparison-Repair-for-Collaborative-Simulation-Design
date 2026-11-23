"""Frozen finite-count analysis and artifact-level verification for revision two."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from policytwin.coordination.schema import commitment
from scripts.cscwd2027.run_studies import compare
from scripts.cscwd2027_revision2.freeze import ROOT, check, digest


def original_obligation_audit() -> list[dict[str, Any]]:
    """Reinterpret recorded findings only, without rewriting original evidence."""
    records: dict[str, Counter[str]] = {}
    for path in sorted((ROOT / "results/cscwd2027/study_v1").glob("e2-*/result.json")):
        result = json.loads(path.read_text())
        if result["design"]["group"] != "e2":
            continue
        kind = result["design"]["fault"]["kind"]
        counts = records.setdefault(kind, Counter())
        counts["attempts"] += 1
        decision = result["decisions"]["full"]
        components = {f["component"] for f in decision["findings"]}
        evaluated = "reexecution" in decision["component_seconds"]
        complete = result["comparison"]["complete"]
        coord = "coordination_conformance" in decision[
            "component_seconds"
        ] and not components.intersection(
            {
                "expected",
                "message",
                "view",
                "ownership",
                "version",
                "logical_order",
                "schema",
            }
        )
        replay = evaluated and "replay" not in components
        counts["complete"] += complete
        counts["replay_evaluated"] += evaluated
        counts["coord_pass_replay_fail"] += coord and evaluated and not replay
        counts["replay_pass_coord_fail"] += replay and not coord
        counts["draw_equal_contrast_changed"] += (
            result["comparison"]["draw_equal"]
            and result["comparison"]["numerical_changed"] is True
        )
    return [{"kind": key, **dict(value)} for key, value in sorted(records.items())]


def verify_reports(
    folder: Path, capsule: dict[str, Any], result: dict[str, Any]
) -> None:
    """Check saved flags against the two actual numerical-report files."""
    for mode, key in (("full", "full"), ("full_plus_witnesses", "extended")):
        directory = folder / "reports" / mode
        if json.loads((directory / "decision.json").read_text()) != result[key]:
            raise ValueError("decision file differs from result receipt")
        numeric = directory / "numerical_report.json"
        if (
            numeric.is_file() != result[key]["numerical_file_exists"]
            or numeric.is_file() != result[key]["release_authorized"]
        ):
            raise ValueError("actual report presence mismatch")
        if numeric.is_file():
            data = json.loads(numeric.read_text())
            if (
                data["capsule_sha256"] != capsule["capsule_sha256"]
                or data["metrics"] != capsule["metrics"]
            ):
                raise ValueError("actual report value binding mismatch")


def analyze() -> dict[str, Any]:
    """Require every scheduled attempt and recompute all exact comparisons."""
    freeze = check()
    rows = json.loads((ROOT / "configs/cscwd2027/revision2/schedule.json").read_text())
    location = ROOT / "results/cscwd2027/revision2/formal"
    results = []
    failed = []
    missing = []
    findings = []
    file_manifest = {}
    for row in rows:
        folder = location / row["id"]
        if not (folder / "result.json").is_file():
            missing.append(row["id"])
            continue
        result = json.loads((folder / "result.json").read_text())
        results.append(result)
        if result["status"] != "completed":
            failed.append(row["id"])
            continue
        try:
            attempt = json.loads((folder / "attempt.json").read_text())
            if attempt["design"] != row or result["design"] != row:
                raise ValueError("design changed")
            if attempt["freeze_sha256"] != commitment(freeze):
                raise ValueError("pre-execution freeze binding mismatch")
            if datetime.fromisoformat(
                attempt["started_at_utc"]
            ) < datetime.fromisoformat(freeze["created_at_utc"]):
                raise ValueError("attempt predates freeze")
            capsule = json.loads(
                gzip.decompress((folder / "capsule.json.gz").read_bytes())
            )
            if capsule["capsule_sha256"] != commitment(
                {k: v for k, v in capsule.items() if k != "capsule_sha256"}
            ):
                raise ValueError("capsule content commitment mismatch")
            if capsule["spec"] != row["spec"] or capsule["plan"] != row["plan"]:
                raise ValueError("capsule scientific/execution specification mismatch")
            if capsule["registration"] != attempt["registration"]:
                raise ValueError("registration mismatch")
            if result["worker_pids"] != capsule["worker_pids"]:
                raise ValueError("worker identity receipt mismatch")
            if row["processes"] and (
                capsule["process_mode"] != "spawn"
                or len(set(capsule["worker_pids"].values())) != 4
            ):
                raise ValueError("actual four-worker execution missing")
            verify_reports(folder, capsule, result)
            if row["reference_id"]:
                reference = json.loads(
                    gzip.decompress(
                        (
                            location / row["reference_id"] / "capsule.json.gz"
                        ).read_bytes()
                    )
                )
                if compare(reference, capsule) != result["comparison"]:
                    raise ValueError("exact matched comparison drift")
            for path in folder.rglob("*"):
                if path.is_file():
                    file_manifest[str(path.relative_to(ROOT))] = digest(path)
        except (ValueError, KeyError, OSError, TypeError) as error:
            findings.append({"id": row["id"], "error": str(error)})
    extension: dict[tuple[int, int, str, str], Counter[str]] = {}
    common: dict[str, Counter[str]] = {}
    for result in results:
        if result["status"] != "completed":
            continue
        row = result["design"]
        if row["study"] == "extension" and row["processes"]:
            spec, plan = row["spec"], row["plan"]
            key = (
                spec["agent_count"],
                spec["epoch_count"],
                spec["branch"],
                plan["assignment"],
            )
            counter = extension.setdefault(key, Counter())
            counter["attempts"] += 1
            for field in (
                "draw_equal",
                "trace_equal",
                "account_order_fill_equal",
                "complete",
            ):
                counter[field] += result["comparison"][field]
            counter["zero_contrast_error"] += (
                result["comparison"]["contrast_error"] == 0
            )
            counter["full_accept"] += result["full"]["numerical_file_exists"]
            counter["witness_accept"] += result["extended"]["numerical_file_exists"]
        elif row["study"] == "common_mode":
            kind = row["mutant"] or "clean"
            counter = common.setdefault(kind, Counter())
            counter["attempts"] += 1
            counter["completed"] += result["completed"]
            counter["triggered"] += result["mutator_calls"] > 0
            counter["full_release"] += result["full"]["numerical_file_exists"]
            counter["witness_release"] += result["extended"]["numerical_file_exists"]
            counter["witness_pass"] += result["extended"]["witnesses"]["pass"]
            if result["comparison"]:
                counter["trace_changed"] += not result["comparison"]["trace_equal"]
                counter["contrast_changed"] += (
                    result["comparison"]["numerical_changed"] is True
                )
                counter["contrast_unchanged"] += (
                    result["comparison"]["numerical_changed"] is False
                )
    return {
        "label": "post-registration revision2",
        "attempts": len(results),
        "expected_attempts": len(rows),
        "missing": missing,
        "failed": failed,
        "integrity_findings": findings,
        "data_complete": not missing and len(results) == len(rows),
        "extension": [
            {
                "agents": k[0],
                "epochs": k[1],
                "branch": k[2],
                "assignment": k[3],
                **dict(v),
            }
            for k, v in sorted(extension.items())
        ],
        "common_mode": {key: dict(value) for key, value in sorted(common.items())},
        "original_obligation_separation": original_obligation_audit(),
        "raw_file_manifest": file_manifest,
        "analysis_source_sha256": digest(Path(__file__)),
        "freeze_sha256": commitment(freeze),
        "interpretation": (
            "designed finite operation counts; no independent simulation "
            "or population-wide detection claim"
        ),
    }


def main() -> None:
    """Generate complete-data-only structured statistics and paper macros."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    value = analyze()
    destination = ROOT / "results/cscwd2027/revision2/derived"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "summary.json").write_text(json.dumps(value, indent=2) + "\n")
    if not args.partial and (
        not value["data_complete"] or value["failed"] or value["integrity_findings"]
    ):
        raise RuntimeError(
            "complete, integrity-checked study required for paper values"
        )
    print(
        json.dumps(
            {
                k: value[k]
                for k in (
                    "attempts",
                    "expected_attempts",
                    "missing",
                    "failed",
                    "integrity_findings",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
