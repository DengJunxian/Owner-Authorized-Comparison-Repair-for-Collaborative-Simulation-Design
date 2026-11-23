"""Outcome-independent design sessions, recipes, handoffs and revisions."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, replace
from typing import Any

from policytwin.coordination.plan import BRANCHES, CoordinationExecutionPlan, RunSpec
from policytwin.coordination.schema import commitment
from policytwin.design_contract.contract import ComparisonContract, ExpectedSlot
from policytwin.design_contract.runtime import SimulationRecipe

CONFIG: dict[str, Any] = {
    "label": "post-registration comparison-contract study v1",
    "seed_root": "cscwd-design-contract-20261025-v1",
    "agents": [6, 24],
    "regimes": ["deep", "responsive"],
    "depth": {"deep": "5000", "responsive": "6*agents"},
    "epochs": 4,
    "branch_epoch": 1,
    "revised_branch_epoch": 2,
    "sessions_per_cell": 8,
    "replicates": 4,
    "candidates": list(BRANCHES),
    "target_rule": "global session index modulo three, not outcome dependent",
    "plans": [
        asdict(CoordinationExecutionPlan(2)),
        asdict(CoordinationExecutionPlan(4, "contiguous")),
    ],
    "parent_shards": 2,
    "revision_order": "sha256(session_id + revision-order) parity",
    "collector_modes": [
        "local_full_only",
        "spec_grouped",
        "strict_plan_lock",
        "comparison_contract",
        "conventional_relational",
        "minus_scientific_binding",
        "minus_slot_coverage",
    ],
    "decision": "maximize exact mean paired traded notional; lexicographic ties",
    "new_pair_budget": 4,
    "threshold": "exact equality, no fitted numerical tolerance",
    "primary_metrics": [
        "unauthorized_panel_release",
        "valid_panel_acceptance",
        "top1_disagreement",
        "top2_disagreement",
        "optimal_set_error",
        "zero_regret_disagreement",
        "decision_regret",
        "actual_rerun_count",
        "actual_reuse_count",
        "per_slot_scientific_agreement",
        "remap_scientific_agreement",
    ],
    "failure_policy": (
        "retain failures, incompletions and nulls; no automatic retries or new seeds"
    ),
    "analysis": (
        "finite counts by session/cell/case; no population fault interval or p-value"
    ),
}


def number(text: str) -> int:
    """Map a predeclared identity to a deterministic integer, not an outcome."""
    return int(hashlib.sha256(text.encode()).hexdigest(), 16)


def sessions() -> list[dict[str, Any]]:
    """Create all scientific blocks before running any of them."""
    result: list[dict[str, Any]] = []
    for agents in CONFIG["agents"]:
        for regime in CONFIG["regimes"]:
            for batch in range(CONFIG["sessions_per_cell"]):
                index = len(result)
                identifier = f"a{agents}-{regime}-s{batch:02d}"
                result.append(
                    {
                        "id": identifier,
                        "index": index,
                        "agents": agents,
                        "regime": regime,
                        "batch": batch,
                        "liquidity": 5000 if regime == "deep" else 6 * agents,
                        "target": BRANCHES[index % len(BRANCHES)],
                        "revision_order": ["fresh", "selective"]
                        if number(identifier + "revision-order") % 2 == 0
                        else ["selective", "fresh"],
                    }
                )
    return result


def recipes(
    session: dict[str, Any], version: int
) -> dict[tuple[str, int], SimulationRecipe]:
    """Retain every unchanged recipe literally; phase/version is not a seed input."""
    result = {}
    for candidate in BRANCHES:
        for replicate in range(CONFIG["replicates"]):
            seed = commitment([CONFIG["seed_root"], session["id"], replicate])
            spec = RunSpec(
                "cscwd-design-panel-" + session["regime"],
                seed,
                replicate,
                session["agents"],
                candidate,
                CONFIG["epochs"],
                CONFIG["branch_epoch"],
            )
            if version == 2 and candidate == session["target"]:
                spec = replace(spec, branch_epoch=CONFIG["revised_branch_epoch"])
            result[candidate, replicate] = SimulationRecipe(spec, session["liquidity"])
    return result


def contract(session: dict[str, Any], version: int) -> ComparisonContract:
    """Compile required slots from the frozen design, never from received artifacts."""
    values = recipes(session, version)
    contexts = {
        (r.spec.agent_count, r.spec.epoch_count, r.liquidity_per_level)
        for r in values.values()
    }
    if len(contexts) != 1:
        raise ValueError("incomparable population/horizon/liquidity contexts")
    return ComparisonContract(
        session["id"],
        version,
        tuple(
            ExpectedSlot(c, r, recipe.scientific_key)
            for (c, r), recipe in values.items()
        ),
        tuple(
            CoordinationExecutionPlan.from_mapping(p).sha256 for p in CONFIG["plans"]
        ),
    )


def object_id(
    session: dict[str, Any], phase: str, candidate: str, replicate: int
) -> str:
    """Storage phase is operational metadata only."""
    return f"{session['id']}-{phase}-{candidate}-r{replicate}"


def attempts(session: dict[str, Any]) -> list[dict[str, Any]]:
    """Enumerate 31 fixed actual paired attempts in one block."""
    rows = []
    phases = ["initial", "remap", *session["revision_order"]]
    for phase in phases:
        version = 1 if phase in {"initial", "remap"} else 2
        for (candidate, replicate), recipe in recipes(session, version).items():
            if phase == "remap" and replicate != 0:
                continue
            if phase == "selective" and candidate != session["target"]:
                continue
            rows.append(
                {
                    "id": object_id(session, phase, candidate, replicate),
                    "session": session["id"],
                    "phase": phase,
                    "candidate": candidate,
                    "replicate": replicate,
                    "recipe": recipe.to_mapping(),
                    "plan": CONFIG["plans"][1 if phase == "remap" else 0],
                }
            )
    return rows


def member(
    session: dict[str, Any], phase: str, candidate: str, replicate: int
) -> dict[str, Any]:
    """Create a handoff descriptor without error labels."""
    return {
        "candidate": candidate,
        "replicate": replicate,
        "object_id": object_id(session, phase, candidate, replicate),
    }


def cases(session: dict[str, Any]) -> list[dict[str, Any]]:
    """Freeze every handoff, donor and target independently of numerical outcomes."""
    base = [
        member(session, "initial", c, r)
        for c in BRANCHES
        for r in range(CONFIG["replicates"])
    ]
    remap = [
        member(session, "remap" if r == 0 else "initial", c, r)
        for c in BRANCHES
        for r in range(CONFIG["replicates"])
    ]
    labels = {c: BRANCHES[(i + 1) % len(BRANCHES)] for i, c in enumerate(BRANCHES)}
    label_cycle = [{**m, "candidate": labels[m["candidate"]]} for m in base]
    donor = {**session, "batch": (session["batch"] + 1) % CONFIG["sessions_per_cell"]}
    donor["id"] = f"a{session['agents']}-{session['regime']}-s{donor['batch']:02d}"
    target = number(session["id"] + "handoff-target") % len(base)
    old = base[target]
    foreign = [dict(m) for m in base]
    foreign[target]["object_id"] = object_id(
        donor, "initial", old["candidate"], old["replicate"]
    )
    values = [
        ("clean", 1, base),
        ("authorized_remap", 1, remap),
        ("label_cycle", 1, label_cycle),
        ("foreign_session", 1, foreign),
        ("missing_slot", 1, base[:target] + base[target + 1 :]),
        ("duplicate_slot", 1, [*base, dict(old)]),
        ("stale_revision", 2, base),
    ]
    return [
        {
            "id": session["id"] + "-" + name,
            "kind": name,
            "session": session["id"],
            "version": version,
            "members": sorted(
                rows, key=lambda m: number(session["id"] + name + m["object_id"])
            ),
        }
        for name, version, rows in values
    ]
