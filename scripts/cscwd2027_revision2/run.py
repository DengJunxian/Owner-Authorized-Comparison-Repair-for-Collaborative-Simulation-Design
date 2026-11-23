"""Resume-safe extension and shared-defect diagnostic execution."""

from __future__ import annotations

import argparse
import gzip
import json
import traceback
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from policytwin.coordination.coordinator import run_pair
from policytwin.coordination.plan import CoordinationExecutionPlan, RunSpec
from policytwin.coordination.release_gate import register, report, seal
from policytwin.coordination.schema import commitment
from policytwin.coordination_validation.gate import witnessed_report
from policytwin.provenance import canonical_json_bytes
from scripts.cscwd2027.run_studies import compare
from scripts.cscwd2027_revision2.freeze import ROOT, check
from scripts.cscwd2027_revision2.mutators import common_mode

RESULTS = ROOT / "results/cscwd2027/revision2/formal"


def save(path: Path, data: object) -> None:
    """Create a record once; never overwrite a completed or failed attempt."""
    with path.open("xb") as stream:
        stream.write(canonical_json_bytes(data))


def load(identifier: str) -> dict[str, Any]:
    """Read a separately executed, matched clean reference."""
    value: dict[str, Any] = json.loads(
        gzip.decompress((RESULTS / identifier / "capsule.json.gz").read_bytes())
    )
    return value


def execute(row: dict[str, Any], freeze_sha256: str) -> dict[str, Any]:
    """Register before dispatch; keep hooks active during execution and both reports."""
    destination = RESULTS / row["id"]
    destination.mkdir(parents=True, exist_ok=False)
    spec, plan = (
        RunSpec.from_mapping(row["spec"]),
        CoordinationExecutionPlan.from_mapping(row["plan"]),
    )
    registration = register(spec, plan)
    save(
        destination / "attempt.json",
        {
            "design": row,
            "registration": asdict(registration),
            "freeze_sha256": freeze_sha256,
            "started_at_utc": datetime.now(UTC).isoformat(),
        },
    )
    start = perf_counter()
    result: dict[str, Any] = {"id": row["id"], "design": row, "status": "failed"}
    try:
        with common_mode(row["mutant"]) as counter:
            capsule = seal(
                run_pair(spec, plan, processes=row["processes"]), registration
            )
            with (destination / "capsule.json.gz").open("xb") as stream:
                stream.write(gzip.compress(canonical_json_bytes(capsule), mtime=0))
            full = report(
                spec, plan, capsule, registration, destination / "reports/full"
            )
            extended = witnessed_report(
                spec,
                plan,
                capsule,
                registration,
                destination / "reports/full_plus_witnesses",
            )
        comparison = (
            None
            if row["reference_id"] is None
            else compare(load(row["reference_id"]), capsule)
        )
        result.update(
            {
                "status": "completed",
                "full": full,
                "extended": extended,
                "comparison": comparison,
                "mutator_calls": counter["calls"],
                "metrics": capsule["metrics"],
                "worker_pids": capsule["worker_pids"],
                "completed": all(w["completed"] for w in capsule["worlds"].values()),
                "capsule_sha256": capsule["capsule_sha256"],
                "runtime_seconds": perf_counter() - start,
            }
        )
        if row["mutant"]:
            save(
                destination / "injection_witness.json",
                {
                    "kind": row["mutant"],
                    "hook_calls": counter["calls"],
                    "trace_changed_from_clean": comparison is not None
                    and not comparison["trace_equal"],
                    "scope": (
                        "same defect applied in-process to execution and replay; "
                        "source files unchanged"
                    ),
                },
            )
    except Exception as error:
        result.update(
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
            runtime_seconds=perf_counter() - start,
        )
    save(destination / "result.json", result)
    return result


def main() -> None:
    """Use the full frozen list; preserve every failure and interrupted attempt."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", choices=("extension", "common_mode"))
    args = parser.parse_args()
    manifest = check()
    rows = json.loads((ROOT / "configs/cscwd2027/revision2/schedule.json").read_text())
    RESULTS.mkdir(parents=True, exist_ok=True)
    for row in rows:
        if args.study and row["study"] != args.study:
            continue
        folder = RESULTS / row["id"]
        if (folder / "result.json").is_file():
            continue
        if folder.exists():
            raise RuntimeError(f"partial attempt retained: {row['id']}")
        check()
        result = execute(row, commitment(manifest))
        print(
            json.dumps(
                {
                    "id": row["id"],
                    "status": result["status"],
                    "seconds": result["runtime_seconds"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
