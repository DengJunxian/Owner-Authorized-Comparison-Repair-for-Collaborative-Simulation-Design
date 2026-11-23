"""Trusted local evidence objects, sealed recipes and actual per-run reports."""

from __future__ import annotations

import gzip
import json
import re
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
from policytwin.coordination.plan import CoordinationExecutionPlan
from policytwin.coordination.release_gate import register, report, seal
from policytwin.coordination.schema import commitment
from policytwin.design_contract.contract import Evidence, Member
from policytwin.design_contract.runtime import SimulationRecipe, configured
from policytwin.provenance import canonical_json_bytes
from scripts.cscwd2027_design_contract.protection import ROOT, digest

RESULTS = ROOT / "results/cscwd2027/design_contract_v1"


def save(path: Path, value: object) -> None:
    """Create each raw record once, including failures."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical_json_bytes(value))


def execute(row: dict[str, Any], freeze_sha256: str) -> dict[str, Any]:
    """Run the existing real-process simulator under the declared recipe."""
    folder = RESULTS / "objects" / row["id"]
    folder.mkdir(parents=True, exist_ok=False)
    recipe = SimulationRecipe.from_mapping(row["recipe"])
    plan = CoordinationExecutionPlan.from_mapping(row["plan"])
    authority = recipe.authority()
    registration = register(recipe.spec, plan)
    save(
        folder / "attempt.json",
        {
            "design": row,
            "freeze_sha256": freeze_sha256,
            "started_at_utc": datetime.now(UTC).isoformat(),
            "recipe_authority": authority,
            "registration": asdict(registration),
        },
    )
    begin = perf_counter()
    result: dict[str, Any] = {"id": row["id"], "status": "failed", "design": row}
    try:
        with configured(recipe):
            capsule = seal(run_pair(recipe.spec, plan, processes=True), registration)
            with (folder / "capsule.json.gz").open("xb") as stream:
                stream.write(gzip.compress(canonical_json_bytes(capsule), mtime=0))
            decision = report(
                recipe.spec,
                plan,
                capsule,
                registration,
                folder / "local_report",
                "full",
            )
        local_pass = decision["release_authorized"] and authority == recipe.authority()
        if len(set(capsule["worker_pids"].values())) != plan.worker_count:
            raise RuntimeError("actual worker count differs from the plan")
        result.update(
            {
                "status": "completed",
                "local_pass": bool(local_pass),
                "scientific_key": recipe.scientific_key,
                "authority": authority,
                "capsule_sha256": capsule["capsule_sha256"],
                "metric": capsule["metrics"]["paired_traded_notional"]
                if local_pass
                else None,
                "worker_pids": capsule["worker_pids"],
                "local_decision": decision,
                "files": {
                    str(p.relative_to(folder)): digest(p)
                    for p in sorted(folder.rglob("*"))
                    if p.is_file()
                },
            }
        )
    except Exception as error:
        result.update(
            {
                "error": f"{type(error).__name__}: {error}",
                "local_pass": False,
                "metric": None,
            }
        )
    result["elapsed_seconds"] = perf_counter() - begin
    result["stored_bytes"] = sum(
        p.stat().st_size for p in folder.rglob("*") if p.is_file()
    )
    save(folder / "result.json", result)
    return result


def load(object_id: str) -> tuple[Evidence, dict[str, Any]]:
    """Validate committed local report/cache context before exposing a token.

    This is a trusted local store; an adversarial replacement of it is not
    claimed detectable. Each selected recipe/source context is checked anew.
    """
    if re.fullmatch(r"[A-Za-z0-9_-]+", object_id) is None:
        raise ValueError("invalid evidence object id")
    folder = RESULTS / "objects" / object_id
    value = json.loads((folder / "result.json").read_text())
    recipe = SimulationRecipe.from_mapping(value["design"]["recipe"])
    plan = CoordinationExecutionPlan.from_mapping(value["design"]["plan"])
    passed = value["status"] == "completed" and value["local_pass"]
    capsule: dict[str, Any] = {}
    if value["status"] == "completed":
        if any(digest(folder / name) != sha for name, sha in value["files"].items()):
            raise RuntimeError("stored object file commitment changed")
        if (
            value["authority"] != recipe.authority()
            or value["scientific_key"] != recipe.scientific_key
        ):
            raise RuntimeError("verification context changed")
        capsule = json.loads(gzip.decompress((folder / "capsule.json.gz").read_bytes()))
        if capsule["capsule_sha256"] != commitment(
            {k: v for k, v in capsule.items() if k != "capsule_sha256"}
        ):
            raise RuntimeError("stored capsule content changed")
        if passed:
            numeric = json.loads(
                (folder / "local_report/numerical_report.json").read_text()
            )
            if (
                numeric["capsule_sha256"] != capsule["capsule_sha256"]
                or numeric["metrics"] != capsule["metrics"]
            ):
                raise RuntimeError("local numerical file binding differs")
    token = Evidence(
        object_id,
        value.get("scientific_key", ""),
        plan.sha256,
        value.get("capsule_sha256", ""),
        bool(passed),
        value.get("metric"),
        recipe.spec.branch,
        recipe.spec.replicate,
    )
    return token, capsule


def members(
    descriptors: list[dict[str, Any]], tokens: dict[str, Evidence]
) -> list[Member]:
    """Resolve handoff descriptors without reading their operation labels."""
    return [
        Member(m["candidate"], m["replicate"], tokens[m["object_id"]])
        for m in descriptors
    ]


def compare(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Exact per-slot scientific/draw comparison, not just a final-choice check."""
    complete = bool(left and right) and all(
        w["completed"] for p in (left, right) for w in p["worlds"].values()
    )
    return {
        "complete": complete,
        "draw_equal": bool(left and right)
        and draw_projection(left) == draw_projection(right),
        "science_equal": bool(left and right)
        and all(
            scientific_projection(left["worlds"][w])
            == scientific_projection(right["worlds"][w])
            for w in ("baseline", "policy")
        ),
        "contrast_error": right["metrics"]["paired_traded_notional"]
        - left["metrics"]["paired_traded_notional"]
        if complete
        else None,
    }
