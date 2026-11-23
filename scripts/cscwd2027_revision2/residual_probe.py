"""Separately sealed exploratory counterexample probe for the trusted RNG boundary."""

from __future__ import annotations

import argparse
import gzip
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

from policytwin.coordination.coordinator import run_pair
from policytwin.coordination.plan import CoordinationExecutionPlan, RunSpec
from policytwin.coordination.release_gate import Registration, register, report, seal
from policytwin.coordination_validation.gate import witnessed_report
from policytwin.counterfactual.coupling_contract import EventKeyedBackend
from scripts.cscwd2027.run_studies import compare
from scripts.cscwd2027_revision2.freeze import ROOT, check, digest

CONFIG = ROOT / "configs/cscwd2027/revision2/residual_probe_config.json"
FREEZE = ROOT / "configs/cscwd2027/revision2/residual_probe_freeze.json"


def freeze_probe() -> None:
    """Seal one explicit negative-boundary question before its outcomes exist."""
    check()
    config = {
        "label": "separate exploratory probe, not the 336-attempt grid",
        "question": "Do replay and witnesses detect a shared all-zero sampler?",
        "spec": asdict(
            RunSpec(
                "r2-residual-rng", "r2-residual-fixed-seed-v1", 0, 6, "agent_exit", 4, 1
            )
        ),
        "plan": asdict(CoordinationExecutionPlan(1)),
        "operator": "return zero for every active draw in execution and replay",
        "attempts": ["clean", "shared_zero_sampler"],
        "metrics": [
            "actual_full_report",
            "actual_witnessed_report",
            "exact_trace_change",
            "draw_change",
            "contrast_error",
        ],
        "selection": (
            "one counterexample search; no rate estimate or post-hoc replacement"
        ),
    }
    with CONFIG.open("x") as stream:
        json.dump(config, stream, indent=2)
    paths = [
        Path(__file__),
        CONFIG,
        ROOT / "src/policytwin/coordination_validation/witnesses.py",
        ROOT / "src/policytwin/coordination_validation/gate.py",
    ]
    value = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "authority": "local pre-probe seal; exploratory and not blind",
        "members": {str(p.relative_to(ROOT)): digest(p) for p in paths},
        "parent_freeze_sha256": digest(
            ROOT / "configs/cscwd2027/revision2/freeze.json"
        ),
    }
    with FREEZE.open("x") as stream:
        json.dump(value, stream, indent=2)
    with FREEZE.with_suffix(".sha256").open("x") as stream:
        stream.write(digest(FREEZE) + "\n")
    print(
        json.dumps(
            {
                "status": "RESIDUAL_PROBE_FROZEN",
                "sha256": digest(FREEZE),
                "created_at_utc": value["created_at_utc"],
            }
        )
    )


def run_probe() -> None:
    """Retain both clean and modified capsules, including any undetected artifact."""
    check()
    value = json.loads(FREEZE.read_text())
    if digest(FREEZE) != FREEZE.with_suffix(".sha256").read_text().strip():
        raise RuntimeError("residual probe seal changed")
    if any(digest(ROOT / p) != sha for p, sha in value["members"].items()):
        raise RuntimeError("residual probe source/config changed")
    config = json.loads(CONFIG.read_text())
    spec, plan = (
        RunSpec.from_mapping(config["spec"]),
        CoordinationExecutionPlan.from_mapping(config["plan"]),
    )
    location = ROOT / "results/cscwd2027/revision2/residual_probe"
    location.mkdir(parents=True, exist_ok=False)
    capsules = {}
    results: dict[str, dict[str, Any]] = {}
    for mode in config["attempts"]:
        folder = location / mode
        folder.mkdir()
        registration = register(spec, plan)
        (folder / "attempt.json").write_text(
            json.dumps(
                {
                    "registration": asdict(registration),
                    "probe_freeze_sha256": digest(FREEZE),
                    "mode": mode,
                    "started_at_utc": datetime.now(UTC).isoformat(),
                },
                indent=2,
            )
        )

        def execute(
            registration: Registration,
            folder: Path,
        ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
            capsule = seal(run_pair(spec, plan, processes=False), registration)
            full = report(spec, plan, capsule, registration, folder / "full")
            witnessed = witnessed_report(
                spec, plan, capsule, registration, folder / "witnessed"
            )
            return capsule, full, witnessed

        if mode == "shared_zero_sampler":
            with patch.object(EventKeyedBackend, "draw", return_value=0) as hook:
                capsule, full, witnessed = execute(registration, folder)
                calls = hook.call_count
        else:
            capsule, full, witnessed = execute(registration, folder)
            calls = 0
        capsules[mode] = capsule
        (folder / "capsule.json.gz").write_bytes(
            gzip.compress(json.dumps(capsule, sort_keys=True).encode(), mtime=0)
        )
        results[mode] = {"full": full, "witnessed": witnessed, "calls": calls}
        (folder / "result.json").write_text(json.dumps(results[mode], indent=2))
    result = {
        "scope": config["label"],
        "probe_freeze_sha256": digest(FREEZE),
        "clean_full_release": results["clean"]["full"]["numerical_file_exists"],
        "clean_witnessed_release": results["clean"]["witnessed"][
            "numerical_file_exists"
        ],
        "mutated_full_release": results["shared_zero_sampler"]["full"][
            "numerical_file_exists"
        ],
        "mutated_witnessed_release": results["shared_zero_sampler"]["witnessed"][
            "numerical_file_exists"
        ],
        "hook_calls": results["shared_zero_sampler"]["calls"],
        "comparison": compare(capsules["clean"], capsules["shared_zero_sampler"]),
        "interpretation": (
            "An altered sampler is outside the partial action/ledger specification. "
            "One counterexample is not a population false-negative rate."
        ),
    }
    (location / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


def main() -> None:
    """Freeze and run are separate commands; never reselect a scenario."""
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("freeze", "run"))
    args = parser.parse_args()
    if args.operation == "freeze":
        freeze_probe()
    else:
        run_probe()


if __name__ == "__main__":
    main()
