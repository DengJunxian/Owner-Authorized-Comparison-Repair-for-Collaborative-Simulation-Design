"""Artifact-bound checks and actual file-level numerical-report authorization."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import perf_counter
from typing import Any

from policytwin.coordination.auditor import (
    Finding,
    coordination_conformance,
    event_conformance,
)
from policytwin.coordination.plan import (
    CoordinationExecutionPlan,
    RunSpec,
    compile_contract,
)
from policytwin.coordination.replay import replay_valid
from policytwin.coordination.schema import commitment
from policytwin.provenance import canonical_json_bytes

MODES = (
    "keys_only",
    "receipts",
    "coordination",
    "full",
    "conventional",
    "minus_ownership_version",
    "minus_expected",
    "minus_replay",
    "minus_gate",
)


def implementation_sha256() -> str:
    """Bind live source bytes including reused scientific dependencies."""
    package = Path(__file__).resolve().parents[1]
    paths = list((package / "coordination").glob("*.py"))
    paths += list((package / "market").glob("*.py"))
    paths += list((package / "counterfactual").glob("*.py"))
    paths.append(package / "provenance.py")
    members = [
        (str(path.relative_to(package)), hashlib.sha256(path.read_bytes()).hexdigest())
        for path in sorted(paths)
    ]
    return commitment(members)


@dataclass(frozen=True, slots=True)
class Registration:
    """External-to-capsule authority fixed before a paired attempt begins."""

    spec_sha256: str
    plan_sha256: str
    contract_sha256: str
    implementation_sha256: str
    registered_at_utc: str
    cutoff_utc: str
    final_epoch: int
    required_metrics: tuple[str, ...] = ("paired_traded_notional",)


def register(spec: RunSpec, plan: CoordinationExecutionPlan) -> Registration:
    """Create a local pre-execution binding, not an external timestamp claim."""
    now = datetime.now(UTC)
    return Registration(
        commitment(spec.to_mapping()),
        plan.sha256,
        compile_contract(spec).sha256,
        implementation_sha256(),
        now.isoformat(),
        (now + timedelta(days=1)).isoformat(),
        spec.epoch_count - 1,
    )


def seal(artifact: dict[str, Any], registration: Registration) -> dict[str, Any]:
    """Commit to all recorded execution and provenance fields after execution."""
    capsule = {
        **artifact,
        "registration": asdict(registration),
        "finished_at_utc": datetime.now(UTC).isoformat(),
    }
    capsule["capsule_sha256"] = commitment(capsule)
    return capsule


def lock_findings(
    spec: RunSpec,
    plan: CoordinationExecutionPlan,
    capsule: dict[str, Any],
    registration: Registration,
) -> tuple[Finding, ...]:
    """Bind external registration, live source, capsule, metric and cutoff."""
    checks = {
        "spec_binding": commitment(capsule["spec"])
        == registration.spec_sha256
        == commitment(spec.to_mapping()),
        "plan_binding": commitment(capsule["plan"])
        == registration.plan_sha256
        == plan.sha256,
        "contract_binding": capsule["contract_sha256"]
        == registration.contract_sha256
        == compile_contract(spec).sha256,
        "source_binding": implementation_sha256() == registration.implementation_sha256,
        "registration_binding": commitment(capsule["registration"])
        == commitment(asdict(registration)),
        "capsule_binding": capsule.get("capsule_sha256")
        == commitment({k: v for k, v in capsule.items() if k != "capsule_sha256"}),
        "required_metric": all(
            type(capsule.get("metrics", {}).get(name)) is int
            for name in registration.required_metrics
        ),
        "epoch_cutoff": registration.final_epoch == spec.epoch_count - 1
        and all(
            row["epoch"] <= registration.final_epoch
            for world in capsule["worlds"].values()
            for row in world["epochs"]
        ),
    }
    try:
        begin = datetime.fromisoformat(registration.registered_at_utc)
        end = datetime.fromisoformat(capsule["finished_at_utc"])
        cutoff = datetime.fromisoformat(registration.cutoff_utc)
        checks["time_order"] = (
            begin.tzinfo is not None
            and end.tzinfo is not None
            and cutoff.tzinfo is not None
            and begin <= end <= cutoff
        )
    except (TypeError, ValueError):
        checks["time_order"] = False
    return tuple(
        Finding("lock", name, "pair", -1, "")
        for name, passed in checks.items()
        if not passed
    )


def evaluate(
    spec: RunSpec,
    plan: CoordinationExecutionPlan,
    capsule: dict[str, Any],
    registration: Registration,
    mode: str = "full",
) -> dict[str, Any]:
    """Evaluate the selected obligation set without consulting fault ground truth."""
    if mode not in MODES:
        raise ValueError("unknown reporting mode")
    started = perf_counter()
    findings: list[Finding] = []
    timings: dict[str, float] = {}
    try:
        complete = set(capsule["worlds"]) == {"baseline", "policy"} and all(
            capsule["worlds"][w]["completed"] is True for w in ("baseline", "policy")
        )
    except (KeyError, TypeError):
        complete = False
    if not complete:
        findings.append(Finding("native", "incomplete_execution", "pair", -1, ""))
    try:
        if mode != "keys_only":
            before = perf_counter()
            findings.extend(event_conformance(spec, capsule))
            timings["event_conformance"] = perf_counter() - before
        if mode not in ("keys_only", "receipts"):
            before = perf_counter()
            coordination = coordination_conformance(spec, plan, capsule)
            if mode == "minus_ownership_version":
                coordination = tuple(
                    f
                    for f in coordination
                    if f.component not in ("ownership", "version")
                )
            if mode == "minus_expected":
                coordination = tuple(
                    f for f in coordination if f.component != "expected"
                )
            findings.extend(coordination)
            timings["coordination_conformance"] = perf_counter() - before
        if mode not in ("keys_only", "receipts", "coordination", "minus_replay"):
            before = perf_counter()
            findings.extend(replay_valid(spec, plan, capsule))
            timings["reexecution"] = perf_counter() - before
        if mode not in ("keys_only", "receipts", "coordination"):
            before = perf_counter()
            findings.extend(lock_findings(spec, plan, capsule, registration))
            timings["lock_metric_cutoff"] = perf_counter() - before
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        findings.append(
            Finding(
                "schema", f"invalid_evidence:{type(error).__name__}", "pair", -1, ""
            )
        )
    # The conventional baseline explicitly composes the same verified primitives.
    # It is an equal-obligation adapter, not an independent verifier implementation.
    diagnostic_pass = not findings
    reportable_metric = (
        type(capsule.get("metrics", {}).get("paired_traded_notional")) is int
    )
    released = (
        complete and reportable_metric and (diagnostic_pass or mode == "minus_gate")
    )
    return {
        "mode": mode,
        "diagnostic_pass": diagnostic_pass,
        "release_authorized": released,
        "findings": [asdict(f) for f in findings],
        "component_seconds": timings,
        "verification_seconds": perf_counter() - started,
    }


def report(
    spec: RunSpec,
    plan: CoordinationExecutionPlan,
    capsule: dict[str, Any],
    registration: Registration,
    destination: Path,
    mode: str = "full",
) -> dict[str, Any]:
    """Re-evaluate the capsule and write a numerical artifact only when authorized."""
    destination.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    decision = evaluate(spec, plan, capsule, registration, mode)
    if decision["release_authorized"]:
        payload = {
            "run_id": spec.run_id,
            "mode": mode,
            "capsule_sha256": capsule.get("capsule_sha256"),
            "metrics": capsule["metrics"],
        }
        with (destination / "numerical_report.json").open("xb") as stream:
            stream.write(canonical_json_bytes(payload))
    decision["numerical_file_exists"] = (
        destination / "numerical_report.json"
    ).is_file()
    decision["report_pipeline_seconds"] = perf_counter() - started
    with (destination / "decision.json").open("xb") as stream:
        stream.write(canonical_json_bytes(decision))
    return decision
