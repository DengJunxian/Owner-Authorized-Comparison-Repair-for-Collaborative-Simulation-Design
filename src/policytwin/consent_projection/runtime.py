"""Outcome-blind selection followed by actual paired acquisition and replay."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from policytwin.consent_projection.policy import authorized, votes
from policytwin.consent_projection.search import ALL_METHODS, METHODS, search
from policytwin.consent_projection.study import selected_slots
from policytwin.coordination.schema import commitment
from policytwin.negotiated_repair.design import Target, contract, recipes
from policytwin.negotiated_repair.runtime import (
    checked_or_rejected,
    fresh_target,
    new_pair,
    repair,
)
from policytwin.negotiated_repair.solver import alternatives
from policytwin.typed_revision.runtime import expectations, read
from scripts.cscwd2027_design_contract.store import save


def execute_session(block: dict[str, Any], root: Path, seal: str) -> dict[str, Any]:
    session = block["session"]
    initial_target = Target(1, "shared", "notional", 2, 12)
    declared = contract(session, initial_target)
    initial = {}
    for slot, recipe in recipes(session, initial_target).items():
        identifier = f"{session['id']}-initial-{slot[0]}-r{slot[1]}"
        trace, capsule = new_pair(identifier, recipe, 2, root, seal)
        token = checked_or_rejected(
            trace,
            capsule,
            initial_target,
            root / "initial_evaluations" / f"{identifier}.json",
            expectations(declared)[slot][1],
        )
        initial[slot] = trace, token
    physical = []
    for context in block["contexts"]:
        count = context["catalog_size"]
        catalog = {s: initial[s] for s in selected_slots(session, count)}
        catalog_records = [
            {"trace": asdict(t), "evaluation": asdict(e)} for t, e in catalog.values()
        ]
        catalog_hash = commitment(catalog_records)
        options = alternatives(
            context["tickets"][0]["request"],
            [t for t, _ in catalog.values()],
            [e for _, e in catalog.values()],
        )
        context_id = f"{session['id']}-store{count}"
        save(
            root / "catalogs" / f"{context_id}.json",
            {
                "session": session,
                "catalog_size": count,
                "catalog": catalog_records,
                "catalog_sha256": catalog_hash,
                "alternatives": options,
            },
        )
        for ticket in context["tickets"]:
            request, policies = ticket["request"], ticket["policies"]
            decisions = {
                m: search(options, request, policies, catalog_hash, m)
                for m in ALL_METHODS
            }
            if (
                not decisions["exhaustive"]["selected"]
                == decisions["certificates"]["selected"]
                == decisions["policy_sql"]["selected"]
            ):
                raise RuntimeError("complete selection methods disagree")
            entry = {
                **ticket,
                "catalog_id": context_id,
                "catalog_sha256": catalog_hash,
                "decisions": decisions,
                "recorded_before_acquisition_utc": datetime.now(UTC).isoformat(),
            }
            save(root / "planning" / f"{session['id']}-{request['kind']}.json", entry)
            if ticket["physical"]:
                physical.append((entry, catalog))
    reference_targets = {}
    for entry, _ in physical:
        for method in METHODS:
            option = entry["decisions"][method]["selected"]
            if option is not None:
                target = Target.from_mapping(option["target"])
                reference_targets[target.epoch, target.coupling, target.workers] = (
                    target
                )
    references = {
        key: fresh_target(session, target, root, seal, i)
        for i, (key, target) in enumerate(sorted(reference_targets.items()))
    }
    rows = []
    for entry, catalog in physical:
        request, policies = entry["request"], entry["policies"]
        for method in sorted(
            METHODS, key=lambda m: commitment([session["id"], request["kind"], m])
        ):
            option = entry["decisions"][method]["selected"]
            identifier = f"{session['id']}-{request['kind']}-{method}"
            folder = root / "projection_handoffs" / identifier
            save(folder / "selection.json", entry["decisions"][method])
            if option is None:
                row = {
                    "id": identifier,
                    "session": session["id"],
                    "catalog_size": request["availability"],
                    "method": method,
                    "status": "no_agreement",
                    "authorized": False,
                    "exact": None,
                    "new_pairs": 0,
                    "reevaluations": 0,
                    "reused_evaluations": 0,
                    "compared_slots": 0,
                    "matching_slots": 0,
                    "authorization_predicate_calls": 0,
                }
            else:
                records = votes(request, option, policies, entry["catalog_sha256"])
                save(folder / "votes.json", records)
                if not authorized(
                    request, option, policies, entry["catalog_sha256"], records
                ):
                    raise RuntimeError("physical acquisition lacks owner agreement")
                target = Target.from_mapping(option["target"])
                inherited = repair(
                    request,
                    method,
                    option,
                    catalog,
                    root,
                    seal,
                    references[target.epoch, target.coupling, target.workers],
                )
                owner_ok = authorized(
                    request,
                    option,
                    policies,
                    entry["catalog_sha256"],
                    read(folder / "votes.json"),
                )
                released = bool(inherited["authorized"] and owner_ok)
                row = {
                    "id": identifier,
                    "session": session["id"],
                    "catalog_size": request["availability"],
                    "method": method,
                    "status": "released" if released else "execution_abstention",
                    "authorized": released,
                    "exact": inherited["exact"] if released else None,
                    **{
                        k: inherited[k]
                        for k in (
                            "new_pairs",
                            "reevaluations",
                            "reused_evaluations",
                            "compared_slots",
                            "matching_slots",
                        )
                    },
                    "authorization_predicate_calls": 9,
                }
            save(folder / "result.json", row)
            rows.append(row["id"])
            print(
                "PROJECTION_HANDOFF",
                identifier,
                row["status"],
                row["new_pairs"],
                flush=True,
            )
    return {
        "session": session["id"],
        "planning_cases": 96,
        "physical_handoffs": rows,
        "reference_trajectory_panels": len(references),
    }
