"""Physical paired acquisitions after refusal-capable owner agreement."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from policytwin.coordination.schema import commitment
from policytwin.negotiated_repair.design import Target, available, contract, recipes
from policytwin.negotiated_repair.runtime import (
    checked_or_rejected,
    fresh_target,
    new_pair,
    repair,
)
from policytwin.negotiated_repair.solver import alternatives
from policytwin.owner_policy.protocol import authorized, negotiate
from policytwin.owner_policy.study import METHODS
from policytwin.typed_revision.runtime import expectations, read
from scripts.cscwd2027_design_contract.store import save


def vote_mutations(
    request: dict[str, Any],
    option: dict[str, Any],
    policies: list[dict[str, Any]],
    catalog_hash: str,
    records: list[dict[str, Any]],
    numerical_pass: bool,
) -> list[dict[str, Any]]:
    result = []
    for kind in (
        "missing_owner",
        "stale_policy",
        "changed_acceptance",
        "stale_catalog",
    ):
        changed = deepcopy(records)
        if kind == "missing_owner":
            changed = changed[:-1]
        else:
            field = {
                "stale_policy": "policy_sha256",
                "changed_acceptance": "accepted",
                "stale_catalog": "catalog_sha256",
            }[kind]
            changed[0][field] = False if field == "accepted" else "0" * 64
            changed[0]["vote_sha256"] = commitment(
                {k: v for k, v in changed[0].items() if k != "vote_sha256"}
            )
        permitted = authorized(request, option, policies, catalog_hash, changed)
        result.append(
            {
                "kind": kind,
                "votes": changed,
                "full_release": bool(numerical_pass and permitted),
                "omit_policy_release": bool(numerical_pass),
            }
        )
    return result


def execute_session(block: dict[str, Any], root: Path, seal: str) -> dict[str, Any]:
    session = block["session"]
    initial_target = Target(1, "shared", "notional", 2, 12)
    initial = {}
    initial_contract = contract(session, initial_target)
    for slot, recipe in recipes(session, initial_target).items():
        identifier = f"{session['id']}-initial-{slot[0]}-r{slot[1]}"
        trace, capsule = new_pair(identifier, recipe, 2, root, seal)
        token = checked_or_rejected(
            trace,
            capsule,
            initial_target,
            root / "initial_evaluations" / f"{identifier}.json",
            expectations(initial_contract)[slot][1],
        )
        initial[slot] = trace, token
    plans = []
    for ticket in block["tickets"]:
        request, policies = ticket["request"], ticket["policies"]
        catalog = {s: initial[s] for s in available(session, 6)}
        catalog_records = [
            {"trace": asdict(t), "evaluation": asdict(e)} for t, e in catalog.values()
        ]
        catalog_hash = commitment(catalog_records)
        options = alternatives(
            request, [t for t, _ in catalog.values()], [e for _, e in catalog.values()]
        )
        decisions = {
            m: negotiate(options, request, policies, catalog_hash, m) for m in METHODS
        }
        if decisions["owner_search"]["selected"] != decisions["policy_csp"]["selected"]:
            raise RuntimeError(
                "complete policy CSP disagrees with exhaustive owner search"
            )
        entry = {
            "request": request,
            "policies": policies,
            "catalog": catalog_records,
            "catalog_sha256": catalog_hash,
            "alternatives": options,
            "decisions": decisions,
            "recorded_before_reference_utc": datetime.now(UTC).isoformat(),
        }
        save(root / "planning" / f"{session['id']}-{request['kind']}.json", entry)
        plans.append((entry, catalog))
    reference_targets = {}
    for entry, _ in plans:
        for decision in entry["decisions"].values():
            option = decision["selected"]
            if option is not None:
                target = Target.from_mapping(option["target"])
                reference_targets[target.epoch, target.coupling, target.workers] = (
                    target
                )
    references = {
        key: fresh_target(session, target, root, seal, index)
        for index, (key, target) in enumerate(sorted(reference_targets.items()))
    }
    rows = []
    for entry, catalog in plans:
        request, policies = entry["request"], entry["policies"]
        for method in sorted(
            METHODS, key=lambda m: commitment([session["id"], request["kind"], m])
        ):
            decision = entry["decisions"][method]
            option = decision["selected"]
            identifier = f"{session['id']}-{request['kind']}-{method}"
            folder = root / "owner_handoffs" / identifier
            save(folder / "negotiation.json", decision)
            if option is None:
                inherited = {
                    "authorized": False,
                    "exact": None,
                    "new_pairs": 0,
                    "reevaluations": 0,
                    "reused_evaluations": 0,
                    "compared_slots": 0,
                    "matching_slots": 0,
                }
                outer_authorized = False
                mutations = []
            else:
                records = decision["offered"][-1]["votes"]
                save(folder / "owner_votes.json", records)
                if not authorized(
                    request, option, policies, entry["catalog_sha256"], records
                ):
                    raise RuntimeError("acquisition attempted without policy agreement")
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
                outer_authorized = bool(
                    inherited["authorized"]
                    and authorized(
                        request,
                        option,
                        policies,
                        entry["catalog_sha256"],
                        read(folder / "owner_votes.json"),
                    )
                )
                mutations = vote_mutations(
                    request,
                    option,
                    policies,
                    entry["catalog_sha256"],
                    records,
                    bool(inherited["authorized"]),
                )
                save(folder / "policy_vote_mutations.json", mutations)
            row = {
                "id": identifier,
                "session": session["id"],
                "case": request["kind"],
                "method": method,
                "selected": option,
                "proposal_count": len(decision["offered"]),
                "rejected_proposals": sum(
                    not o["accepted"] for o in decision["offered"]
                ),
                "owner_calls": 3 * len(decision["offered"]),
                "authorized": outer_authorized,
                "exact": inherited["exact"] if outer_authorized else None,
                "new_pairs": inherited["new_pairs"],
                "reevaluations": inherited["reevaluations"],
                "reused_evaluations": inherited["reused_evaluations"],
                "matching_slots": inherited["matching_slots"],
                "compared_slots": inherited["compared_slots"],
                "waivers_count": len(option["waived"]) if option is not None else 0,
                "policy_invalid_release_mutations": mutations,
                "status": "no_agreement"
                if option is None
                else "released"
                if outer_authorized
                else "execution_abstention",
            }
            save(folder / "result.json", row)
            rows.append(identifier)
            print(
                "OWNER_POLICY_HANDOFF",
                identifier,
                row["status"],
                row["new_pairs"],
                flush=True,
            )
    return {
        "session": session["id"],
        "handoffs": rows,
        "reference_trajectory_panels": len(references),
    }
