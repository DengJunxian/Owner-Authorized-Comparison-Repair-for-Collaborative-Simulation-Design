"""Outcome-independent contexts and attempts for coupling-only sensitivity."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from policytwin.coordination.plan import BRANCHES, CoordinationExecutionPlan
from policytwin.coordination.schema import commitment
from policytwin.cscwd_mechanism.coupling import COUPLINGS, CoupledRecipe
from scripts.cscwd2027_design_contract.design import CONFIG, number, recipes


def identity(
    session: str, horizon: int, candidate: str, replicate: int, coupling: str
) -> str:
    return f"{session}-h{horizon}-{candidate}-r{replicate}-{coupling}"


def design(
    sessions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    contexts = {}
    for session in sessions:
        settings = [(4, 4, "primary")]
        if session["batch"] == 0:
            settings += [(4, r, "replicate_sweep") for r in (4, 8, 16, 32)]
        if session["agents"] == 6 and session["batch"] in (0, 1):
            settings += [(h, 4, "horizon_sweep") for h in (4, 16, 32)]
        for horizon, count, family in settings:
            identifier = f"{session['id']}-h{horizon}-n{count}"
            if identifier not in contexts:
                contexts[identifier] = {
                    "id": identifier,
                    "session": session,
                    "horizon": horizon,
                    "replicates": count,
                    "families": [],
                    "target_replicate": number(session["id"] + "mechanism-slot")
                    % count,
                }
            contexts[identifier]["families"].append(family)
    attempts = {}
    for context in contexts.values():
        session = context["session"]
        base = recipes(session, 2)
        for candidate in BRANCHES:
            prototype = base[candidate, 0]
            for replicate in range(context["replicates"]):
                recipe = replace(
                    prototype,
                    spec=replace(
                        prototype.spec,
                        epoch_count=context["horizon"],
                        replicate=replicate,
                        seed=commitment(
                            [CONFIG["seed_root"], session["id"], replicate]
                        ),
                    ),
                )
                for coupling in COUPLINGS:
                    identifier = identity(
                        session["id"],
                        context["horizon"],
                        candidate,
                        replicate,
                        coupling,
                    )
                    attempts[identifier] = {
                        "id": identifier,
                        "session": session["id"],
                        "horizon": context["horizon"],
                        "candidate": candidate,
                        "replicate": replicate,
                        "coupling": coupling,
                        "coupled_recipe": CoupledRecipe(recipe, coupling).to_mapping(),
                        "plan": CoordinationExecutionPlan(2).to_mapping(),
                    }
    # Counterbalance coupling execution order by an identity hash, not values.
    return list(contexts.values()), sorted(
        attempts.values(), key=lambda r: number(r["id"] + "execution-order")
    )
