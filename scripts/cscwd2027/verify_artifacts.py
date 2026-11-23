"""Post-run file binding audit, without changing frozen estimators or evidence."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from policytwin.coordination.faults import Fault
from policytwin.coordination.plan import CoordinationExecutionPlan, RunSpec
from policytwin.coordination.release_gate import implementation_sha256
from policytwin.coordination.schema import commitment
from scripts.cscwd2027.freeze_study import DEST, ROOT, check
from scripts.cscwd2027.run_studies import compare, injection_status


def require(condition: bool, message: str) -> None:
    """Raise a bounded audit finding instead of silently repairing evidence."""
    if not condition:
        raise ValueError(message)


def read_capsule(folder: Path) -> dict[str, Any]:
    """Read an existing compressed capsule; never extract arbitrary paths."""
    value: dict[str, Any] = json.loads(
        gzip.decompress((folder / "capsule.json.gz").read_bytes())
    )
    return value


def verify_attempt(
    folder: Path,
    design: dict[str, Any],
    reference: dict[str, Any] | None,
    frozen_at: str,
    source_hash: str,
) -> dict[str, Any]:
    """Recheck recorded outcomes against the files and prescribed projections."""
    result = json.loads((folder / "result.json").read_text())
    attempt = json.loads((folder / "attempt.json").read_text())
    require(result["design"] == attempt["design"] == design, "saved design mismatch")
    require(result["id"] == design["id"], "attempt identity mismatch")
    require(result["status"] == "completed", "runner failure retained")
    packed = (folder / "capsule.json.gz").read_bytes()
    raw = gzip.decompress(packed)
    capsule = json.loads(raw)
    spec = RunSpec.from_mapping(design["spec"])
    plan = CoordinationExecutionPlan.from_mapping(design["plan"])
    require(capsule["spec"] == asdict(spec), "capsule specification mismatch")
    require(capsule["plan"] == asdict(plan), "capsule plan mismatch")
    require(
        commitment({k: v for k, v in capsule.items() if k != "capsule_sha256"})
        == capsule["capsule_sha256"],
        "capsule commitment mismatch",
    )
    registration = attempt["registration"]
    require(capsule["registration"] == registration, "registration substitution")
    require(
        registration["implementation_sha256"] == source_hash,
        "runtime source binding mismatch",
    )
    times = [
        datetime.fromisoformat(v)
        for v in (
            frozen_at,
            registration["registered_at_utc"],
            attempt["started_at_utc"],
            capsule["finished_at_utc"],
            registration["cutoff_utc"],
        )
    ]
    require(all(t.tzinfo is not None for t in times), "missing time-zone metadata")
    require(times == sorted(times), "freeze/registration/execution ordering mismatch")
    pids = capsule["worker_pids"]
    if design["processes"]:
        require(capsule["process_mode"] == "spawn", "missing actual-process marker")
        require(
            len(pids) == len(set(pids.values())) == plan.worker_count
            and all(type(v) is int and v > 0 for v in pids.values()),
            "invalid worker process identities",
        )
    else:
        require(not pids, "reference contains worker process identities")
    require(result["worker_pids"] == pids, "result process identities mismatch")
    for world in capsule["worlds"].values():
        notional = sum(t["price"] * t["quantity"] for t in world["trades"])
        require(notional == world["traded_notional"], "trade/notional mismatch")
    contrast = (
        capsule["worlds"]["policy"]["traded_notional"]
        - capsule["worlds"]["baseline"]["traded_notional"]
    )
    require(
        capsule["metrics"] == result["metrics"] == {"paired_traded_notional": contrast},
        "reported contrast mismatch",
    )
    require(result["raw_capsule_bytes"] == len(raw), "raw size mismatch")
    require(result["gzip_capsule_bytes"] == len(packed), "compressed size mismatch")
    for mode, decision in result["decisions"].items():
        directory = folder / "reports" / mode
        stored = json.loads((directory / "decision.json").read_text())
        require(stored == decision, "saved report decision mismatch")
        numeric = directory / "numerical_report.json"
        require(
            numeric.is_file()
            == decision["numerical_file_exists"]
            == decision["release_authorized"],
            "actual report presence mismatch",
        )
        if numeric.is_file():
            report = json.loads(numeric.read_text())
            require(
                report
                == {
                    "run_id": spec.run_id,
                    "mode": mode,
                    "capsule_sha256": capsule["capsule_sha256"],
                    "metrics": capsule["metrics"],
                },
                "actual numerical report binding mismatch",
            )
    require(
        result["reported_bytes"]
        == sum(p.stat().st_size for p in (folder / "reports").rglob("*.json")),
        "report-file size mismatch",
    )
    if reference is not None:
        require(result["comparison"] == compare(reference, capsule), "comparison drift")
        if design["fault"] is not None:
            witness = injection_status(
                Fault(**design["fault"]), reference, capsule, plan, spec
            )
            stored_witness = json.loads((folder / "injection_witness.json").read_text())
            require(stored_witness == witness, "injection witness mismatch")
            require(
                result["injection_triggered"] == witness["triggered"],
                "injection label mismatch",
            )
    return {
        "id": design["id"],
        "capsule_sha256": capsule["capsule_sha256"],
        "actual_processes": len(pids),
        "reports_checked": len(result["decisions"]),
    }


def main() -> None:
    """Audit the complete grid and retain a file-level integrity manifest."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    freeze = check()
    designs = json.loads((DEST / "schedule_v1.json").read_text())
    result_root = ROOT / "results/cscwd2027/study_v1"
    findings: list[dict[str, str]] = []
    checked = []
    missing = []
    source_hash = implementation_sha256()

    @lru_cache(maxsize=4)
    def reference(identifier: str) -> dict[str, Any]:
        return read_capsule(result_root / identifier)

    for design in designs:
        folder = result_root / design["id"]
        if not (folder / "result.json").is_file():
            missing.append(design["id"])
            continue
        try:
            checked.append(
                verify_attempt(
                    folder,
                    design,
                    reference(design["reference_id"])
                    if design["reference_id"]
                    else None,
                    freeze["created_at_utc"],
                    source_hash,
                )
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            findings.append({"id": design["id"], "finding": str(error)})
    expected_ids = {d["id"] for d in designs}
    unexpected = sorted(
        p.name
        for p in result_root.iterdir()
        if p.is_dir() and p.name not in expected_ids
    )
    report = {
        "purpose": "post-run integrity audit; no new estimator or sample selection",
        "expected_attempts": len(designs),
        "verified_attempts": len(checked),
        "reports_checked": sum(r["reports_checked"] for r in checked),
        "missing_ids": missing,
        "unexpected_ids": unexpected,
        "findings": findings,
        "all_files_bound": not (missing or findings or unexpected),
        "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    destination = ROOT / "reports/cscwd2027"
    (destination / "artifact_integrity_audit.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    if report["all_files_bound"]:
        members = {}
        for path in sorted(result_root.rglob("*")):
            if path.is_file():
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                members[str(path.relative_to(ROOT))] = {
                    "sha256": digest,
                    "bytes": path.stat().st_size,
                }
        (ROOT / "results/cscwd2027/derived_v1/raw_file_manifest.json").write_text(
            json.dumps({"members": members, "audit": report}, indent=2) + "\n"
        )
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "verified_attempts",
                    "expected_attempts",
                    "reports_checked",
                    "all_files_bound",
                )
            }
        )
    )
    if findings or unexpected or (missing and not args.allow_partial):
        raise RuntimeError(
            "artifact audit incomplete or failed; inspect retained report"
        )


if __name__ == "__main__":
    main()
