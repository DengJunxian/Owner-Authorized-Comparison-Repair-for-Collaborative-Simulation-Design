"""Additional report route composing v1 conformance with specification witnesses."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from policytwin.coordination.plan import CoordinationExecutionPlan, RunSpec
from policytwin.coordination.release_gate import Registration, evaluate
from policytwin.coordination_validation.witnesses import specification_witnesses


def witnessed_report(
    spec: RunSpec,
    plan: CoordinationExecutionPlan,
    capsule: dict[str, Any],
    registration: Registration,
    destination: Path,
) -> dict[str, Any]:
    """Write a number only after both unchanged v1 and orthogonal checks pass."""
    destination.mkdir(parents=True, exist_ok=False)
    full = evaluate(spec, plan, capsule, registration, "full")
    try:
        witnesses = specification_witnesses(capsule)
    except (KeyError, ValueError, TypeError) as error:
        witnesses = {"pass": False, "findings": [{"code": type(error).__name__}]}
    authorized = full["release_authorized"] and witnesses["pass"]
    if authorized:
        with (destination / "numerical_report.json").open("x") as stream:
            json.dump(
                {
                    "run_id": spec.run_id,
                    "mode": "full_plus_witnesses",
                    "capsule_sha256": capsule["capsule_sha256"],
                    "metrics": capsule["metrics"],
                },
                stream,
                sort_keys=True,
            )
    decision = {
        "full": full,
        "witnesses": witnesses,
        "release_authorized": authorized,
        "numerical_file_exists": (destination / "numerical_report.json").is_file(),
    }
    with (destination / "decision.json").open("x") as stream:
        json.dump(decision, stream, sort_keys=True)
    return decision
