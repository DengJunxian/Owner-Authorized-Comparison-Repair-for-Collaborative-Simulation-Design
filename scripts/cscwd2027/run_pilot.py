"""Isolated engineering pilot; none of its seeds enter the formal studies."""

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
from policytwin.provenance import canonical_json_bytes

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    """Measure a small fixed pilot and preserve all attempts and file sizes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v1")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--branch-epoch", type=int, default=3)
    args = parser.parse_args()
    if args.version not in ("v1", "v2"):
        parser.error("unsupported pilot version")
    root = ROOT / "results/cscwd2027/pilot" / args.version
    root.mkdir(parents=True, exist_ok=False)
    (root / "configuration.json").write_bytes(canonical_json_bytes(vars(args)))
    rows = []
    for agents, workers in ((6, 1), (24, 1), (24, 4)):
        spec = RunSpec(
            f"pilot-{args.version}",
            f"pilot-{args.version}-budget-only",
            0,
            agents,
            "cancellation",
            args.epochs,
            args.branch_epoch,
        )
        plan = CoordinationExecutionPlan(workers)
        folder = root / f"a{agents}-w{workers}"
        folder.mkdir()
        registration = register(spec, plan)
        (folder / "registration.json").write_bytes(
            canonical_json_bytes(asdict(registration))
        )
        before = perf_counter()
        artifact = run_pair(spec, plan)
        capsule = seal(artifact, registration)
        raw = canonical_json_bytes(capsule)
        packed = gzip.compress(raw, mtime=0)
        (folder / "capsule.json.gz").write_bytes(packed)
        decision = report(spec, plan, capsule, registration, folder / "report")
        row = {
            "agents": agents,
            "workers": workers,
            "execution_seconds": artifact["execution_total_seconds"],
            "pipeline_seconds": perf_counter() - before,
            "raw_bytes": len(raw),
            "gzip_bytes": len(packed),
            "accepted": decision["numerical_file_exists"],
            "component_seconds": decision["component_seconds"],
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    (root / "summary.json").write_bytes(canonical_json_bytes(rows))


if __name__ == "__main__":
    main()
