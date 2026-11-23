"""Resume-safe execution of the prefrozen grid; no network or result selection."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import platform
import sys
import traceback
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from policytwin.coordination.coordinator import (
    draw_projection,
    run_pair,
    scientific_projection,
)
from policytwin.coordination.faults import Fault
from policytwin.coordination.plan import CoordinationExecutionPlan, RunSpec
from policytwin.coordination.release_gate import MODES, register, report, seal
from policytwin.coordination.schema import commitment
from policytwin.provenance import canonical_json_bytes
from scripts.cscwd2027.freeze_study import DEST, ROOT, check

RESULTS = ROOT / "results/cscwd2027/study_v1"


def save(path: Path, value: object) -> None:
    """Never overwrite an attempt or receipt."""
    with path.open("xb") as stream:
        stream.write(canonical_json_bytes(value))


def load_capsule(identifier: str) -> dict[str, Any]:
    """Read the independently executed matched reference by its frozen ID."""
    capsule: dict[str, Any] = json.loads(
        gzip.decompress((RESULTS / identifier / "capsule.json.gz").read_bytes())
    )
    if not isinstance(capsule, dict):
        raise ValueError("reference capsule must be a JSON object")
    return capsule


def compare(reference: dict[str, Any], observed: dict[str, Any]) -> dict[str, Any]:
    """Exact, predeclared projection comparison; no outcome-dependent tolerance."""
    complete = all(w["completed"] for w in observed["worlds"].values())
    agreement = {}
    first = []
    for world in ("baseline", "policy"):
        left, right = reference["worlds"][world], observed["worlds"][world]
        agreement[world] = scientific_projection(left) == scientific_projection(right)
        by_epoch = {e["epoch"]: e for e in right["epochs"]}
        for row in left["epochs"]:
            other = by_epoch.get(row["epoch"])
            if other is None or any(
                row[k] != other[k] for k in ("applied_commands", "state_after")
            ):
                first.append({"world": world, "epoch": row["epoch"]})
                break
    difference = (
        observed["metrics"]["paired_traded_notional"]
        - reference["metrics"]["paired_traded_notional"]
    )
    return {
        "draw_equal": draw_projection(reference) == draw_projection(observed),
        "trace_equal": all(agreement.values()),
        "world_trace_equal": agreement,
        "contrast_error": difference if complete else None,
        "numerical_changed": difference != 0 if complete else None,
        "first_divergences": first,
        "complete": complete,
        "account_order_fill_equal": all(
            reference["worlds"][w]["final_state"]
            == observed["worlds"][w]["final_state"]
            and reference["worlds"][w]["trades"] == observed["worlds"][w]["trades"]
            for w in ("baseline", "policy")
        ),
    }


def injection_status(
    fault: Fault,
    reference: dict[str, Any],
    observed: dict[str, Any],
    plan: CoordinationExecutionPlan,
    spec: RunSpec,
) -> dict[str, Any]:
    """Check that the frozen operation actually occurred, without any auditor result."""
    epoch = next(
        (
            e
            for e in observed["worlds"][fault.world]["epochs"]
            if e["epoch"] == fault.epoch
        ),
        None,
    )
    clean = reference["worlds"][fault.world]["epochs"][fault.epoch]
    if epoch is None:
        return {"triggered": False, "reason": "execution_did_not_reach_target"}
    requests = [r for r in epoch["requests"] if r["agent"] == fault.agent]
    results = [r for r in epoch["results"] if r["agent"] == fault.agent]
    deliveries = [r for r in epoch["deliveries"] if r["agent"] == fault.agent]
    clean_result = next(r for r in clean["results"] if r["agent"] == fault.agent)
    tests = {
        "task_duplication": len(requests) == 2,
        "task_omission": len(requests) == 0,
        "wrong_owner": any(
            r["worker_id"] != plan.owner(spec, fault.agent) for r in results
        ),
        "duplicate_effect": len(deliveries) == 2,
        "missing_message": len(results) == 1 and len(deliveries) == 0,
        "logical_order": epoch["committed_message_ids"]
        != clean["committed_message_ids"],
        "payload_mutation": any(
            r["commands"] != clean_result["commands"] for r in results
        ),
        "stale_plan": any(r["plan_sha256"] != plan.sha256 for r in results),
    }
    return {
        "triggered": tests[fault.kind],
        "reason": "runtime_operation_witness",
        "original_target_result": clean_result,
        "observed_target_results": results,
    }


def execute(row: dict[str, Any]) -> dict[str, Any]:
    """Register before dispatch; retain every outcome and actual report file."""
    folder = RESULTS / row["id"]
    folder.mkdir(parents=True, exist_ok=False)
    spec = RunSpec.from_mapping(row["spec"])
    plan = CoordinationExecutionPlan.from_mapping(row["plan"])
    fault = None if row["fault"] is None else Fault(**row["fault"])
    registration = register(spec, plan)
    save(
        folder / "attempt.json",
        {
            "design": row,
            "registration": asdict(registration),
            "started_at_utc": datetime.now(UTC).isoformat(),
            "load_average": os.getloadavg(),
        },
    )
    started = perf_counter()
    result: dict[str, Any] = {"id": row["id"], "design": row, "status": "failed"}
    try:
        mode = row.get("cost_mode", "full")
        artifact = run_pair(
            spec,
            plan,
            processes=row["processes"],
            fault=fault,
            transport=row["transport"],
            collect_draws=mode not in ("no_evidence", "keys_only"),
            retain_trace=mode != "no_evidence",
        )
        before = perf_counter()
        capsule = seal(artifact, registration)
        raw = canonical_json_bytes(capsule)
        packed = gzip.compress(raw, mtime=0)
        with (folder / "capsule.json.gz").open("xb") as stream:
            stream.write(packed)
        serialization_seconds = perf_counter() - before
        modes = (
            MODES
            if row["group"] in ("e2", "e2b")
            else ("keys_only" if mode == "no_evidence" else mode,)
        )
        decisions = {}
        for selected in modes:
            destination = folder / "reports" / selected
            decision = report(spec, plan, capsule, registration, destination, selected)
            numeric = destination / "numerical_report.json"
            if numeric.is_file():
                content = json.loads(numeric.read_text())
                if (
                    content["capsule_sha256"] != capsule["capsule_sha256"]
                    or content["metrics"] != artifact["metrics"]
                ):
                    raise RuntimeError(
                        "numerical artifact is not bound to the evaluated capsule"
                    )
            decisions[selected] = decision
        elapsed = perf_counter() - started
        result.update(
            {
                "status": "completed",
                "end_to_end_seconds": elapsed,
                "execution_seconds": artifact["execution_total_seconds"],
                "startup_seconds": artifact["startup_seconds"],
                "worker_compute_elapsed_sum_seconds": artifact[
                    "worker_compute_elapsed_sum_seconds"
                ],
                "market_transition_seconds": sum(
                    w["market_transition_seconds"] for w in artifact["worlds"].values()
                ),
                "serialization_storage_seconds": serialization_seconds,
                "raw_capsule_bytes": len(raw),
                "gzip_capsule_bytes": len(packed),
                "reported_bytes": sum(
                    p.stat().st_size for p in (folder / "reports").rglob("*.json")
                ),
                "metrics": artifact["metrics"],
                "decisions": decisions,
                "worker_pids": artifact["worker_pids"],
                "native_errors": {
                    w: a["native_error"] for w, a in artifact["worlds"].items()
                },
                "tasks": 2 * spec.epoch_count * spec.agent_count,
                "recorded_draws": len(draw_projection(artifact)),
                "outcome_sha256": commitment(
                    {
                        w: {"state": a["final_state"], "trades": a["trades"]}
                        for w, a in artifact["worlds"].items()
                    }
                ),
            }
        )
        if row["reference_id"]:
            reference = load_capsule(row["reference_id"])
            result["comparison"] = compare(reference, artifact)
            if fault is not None:
                witness = injection_status(fault, reference, artifact, plan, spec)
                save(folder / "injection_witness.json", witness)
                result["injection_triggered"] = witness["triggered"]
        result["attempt_total_seconds"] = perf_counter() - started
    except Exception as error:
        result.update(
            {
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "attempt_total_seconds": perf_counter() - started,
            }
        )
    save(folder / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", nargs="*", default=[])
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    freeze = check()
    RESULTS.mkdir(parents=True, exist_ok=True)
    host = RESULTS / "host.json"
    if not host.exists():
        save(
            host,
            {
                "python": sys.version,
                "platform": platform.platform(),
                "machine": platform.machine(),
                "logical_cpus": os.cpu_count(),
                "process_start": "spawn",
                "timer": "time.perf_counter",
                "memory_bytes": "unavailable",
                "concurrent_external_load": "not controlled",
                "freeze_created_at_utc": freeze["created_at_utc"],
            },
        )
    grid = json.loads((DEST / "schedule_v1.json").read_text())
    selected = [r for r in grid if not args.groups or r["group"] in args.groups]
    completed = 0
    for row in selected:
        folder = RESULTS / row["id"]
        if (folder / "result.json").is_file():
            continue  # A failed result is also retained, never silently retried.
        if folder.exists():
            raise RuntimeError(
                "incomplete prior attempt retained; manual recovery required: "
                f"{row['id']}"
            )
        result = execute(row)
        completed += 1
        print(
            json.dumps(
                {
                    "id": row["id"],
                    "status": result["status"],
                    "seconds": result["attempt_total_seconds"],
                    "new_attempts_this_invocation": completed,
                }
            ),
            flush=True,
        )
        if args.limit and completed >= args.limit:
            break
    check()
    print(
        json.dumps({"status": "INVOCATION_COMPLETE", "new_attempts": completed}),
        flush=True,
    )


if __name__ == "__main__":
    main()
