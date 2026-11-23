"""Isolated engineering checks before the new study is frozen."""

from __future__ import annotations

import argparse
import gzip
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from policytwin.coordination.coordinator import run_pair
from policytwin.coordination.plan import CoordinationExecutionPlan, RunSpec
from policytwin.coordination.release_gate import register, report, seal
from policytwin.coordination_validation.gate import witnessed_report
from scripts.cscwd2027_revision2.mutators import common_mode

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    """Retain engineering inputs/outputs without using formal seed namespaces."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--mutant")
    parser.add_argument("--agents", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--processes", action="store_true")
    args = parser.parse_args()
    folder = ROOT / "results/cscwd2027/revision2/pilot" / args.name
    folder.mkdir(parents=True, exist_ok=False)
    spec = RunSpec(
        "r2-engineering-pilot",
        "r2-pilot-" + args.name,
        0,
        args.agents,
        "cancellation",
        args.epochs,
        1,
    )
    plan = CoordinationExecutionPlan(4 if args.processes else 1, "contiguous")
    registration = register(spec, plan)
    (folder / "inputs.json").write_text(
        json.dumps(
            {"spec": asdict(spec), "plan": asdict(plan), "mutant": args.mutant},
            indent=2,
        )
    )
    begin = perf_counter()
    with common_mode(args.mutant) as counter:
        capsule = seal(run_pair(spec, plan, processes=args.processes), registration)
        full = report(spec, plan, capsule, registration, folder / "full")
        witnessed = witnessed_report(
            spec, plan, capsule, registration, folder / "witnessed"
        )
    (folder / "capsule.json.gz").write_bytes(
        gzip.compress(json.dumps(capsule, sort_keys=True).encode(), mtime=0)
    )
    value = {
        "elapsed": perf_counter() - begin,
        "full_release": full["numerical_file_exists"],
        "witnessed_release": witnessed["numerical_file_exists"],
        "mutator_calls": counter["calls"],
        "findings": witnessed["witnesses"]["findings"],
    }
    (folder / "result.json").write_text(json.dumps(value, indent=2))
    print(json.dumps({k: v for k, v in value.items() if k != "findings"}))


if __name__ == "__main__":
    main()
