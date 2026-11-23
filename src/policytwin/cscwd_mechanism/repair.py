"""Observable-residual truth and actual offline evidence-store repair.

The truth projection compares complete semantic recipes/plans from a sealed
schedule. It never calls the hash-key diagnostic or uses historical fault
labels as the final-state answer.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from policytwin.coordination.schema import commitment
from policytwin.design_contract.contract import (
    ComparisonContract,
    Evidence,
    Member,
    decide,
    revision_plan,
)
from policytwin.provenance import canonical_json_bytes


def semantic_truth(
    expected: dict[tuple[str, int], dict[str, Any]],
    allowed_plans: list[dict[str, Any]],
    members: list[dict[str, Any]],
    catalog: dict[str, dict[str, Any]],
    verified: dict[str, Evidence],
) -> list[list[Any]]:
    """Enumerate final observable obligations independently of hash-key checks."""
    seen = defaultdict(list)
    for member in members:
        row = catalog[member["object_id"]]
        spec = row["recipe"]["spec"]
        coordinate = spec["branch"], spec["replicate"]
        token = verified[member["object_id"]]
        if coordinate != (token.actual_candidate, token.actual_replicate):
            raise RuntimeError("trusted token and sealed semantic catalog differ")
        seen[coordinate].append(row)
    issues = set()
    for coordinate, recipe in expected.items():
        values = seen.get(coordinate, [])
        if not values:
            issues.add((*coordinate, "missing_slot"))
        if len(values) > 1:
            issues.add((*coordinate, "duplicate_slot"))
        for row in values:
            if canonical_json_bytes(row["recipe"]) != canonical_json_bytes(recipe):
                issues.add((*coordinate, "key_mismatch"))
    for coordinate, values in seen.items():
        if coordinate not in expected:
            issues.add((*coordinate, "unexpected_slot"))
        for row in values:
            token = verified[row["id"]]
            if not token.local_pass or type(token.value) is not int:
                issues.add((*coordinate, "local_invalid"))
            if row["plan"] not in allowed_plans:
                issues.add((*coordinate, "unpermitted_plan"))
    return [list(i) for i in sorted(issues)]


def localization_score(
    predicted: list[list[Any]], truth: list[list[Any]]
) -> dict[str, Any]:
    """Score exact reason/slot sets and slot-level counts; empty rates are n/a."""
    predicted_reasons = {tuple(v) for v in predicted}
    true_reasons = {tuple(v) for v in truth}
    predicted_slots = {v[:2] for v in predicted_reasons}
    true_slots = {v[:2] for v in true_reasons}
    intersection = len(predicted_slots & true_slots)
    return {
        "exact_reason_set": predicted_reasons == true_reasons,
        "exact_slot_set": predicted_slots == true_slots,
        "true_slots": len(true_slots),
        "predicted_slots": len(predicted_slots),
        "correct_slots": intersection,
        "precision": [intersection, len(predicted_slots)] if predicted_slots else None,
        "recall": [intersection, len(true_slots)] if true_slots else None,
    }


def selected_members(
    contract: ComparisonContract, members: list[Member]
) -> list[Member]:
    """Retain one eligible object per slot, without consulting numerical values."""
    lookup = {m.evidence.object_id: m.evidence for m in members}
    retained = revision_plan(contract, members)["reuse"]
    return [
        Member(r["candidate"], r["replicate"], lookup[r["object_id"]]) for r in retained
    ]


def transfer(
    contract: ComparisonContract,
    slots: list[dict[str, Any]],
    fetch: Callable[[str, int], Evidence],
    object_root: Path,
) -> tuple[list[Evidence], dict[str, Any]]:
    """Serialize the request/envelope and count actual existing object-file bytes."""
    request = {"contract_sha256": contract.sha256, "slots": slots}
    objects, records = [], []
    for slot in slots:
        token = fetch(slot["candidate"], slot["replicate"])
        objects.append(token)
        files = []
        for path in sorted((object_root / token.object_id).rglob("*")):
            if path.is_file():
                raw = path.read_bytes()
                files.append(
                    {
                        "name": str(path.relative_to(object_root / token.object_id)),
                        "bytes": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                    }
                )
        records.append(
            {"object_id": token.object_id, "token": asdict(token), "files": files}
        )
    response = {"contract_sha256": contract.sha256, "objects": records}
    request_bytes = len(canonical_json_bytes(request))
    response_bytes = len(canonical_json_bytes(response)) + sum(
        f["bytes"] for o in records for f in o["files"]
    )
    return objects, {
        "request": request,
        "response_manifest": response,
        "request_bytes": request_bytes,
        "response_bytes": response_bytes,
        "request_sha256": commitment(request),
        "envelope_sha256": commitment(response),
    }


def repair_panel(
    contract: ComparisonContract,
    members: list[Member],
    method: str,
    fetch: Callable[[str, int], Evidence],
    object_root: Path,
) -> tuple[dict[str, Any], list[Member]]:
    """Execute three deterministic repair protocols with identical authorization."""
    if method not in {
        "slot_directed_batch",
        "slot_directed_serial",
        "whole_panel_refresh",
    }:
        raise ValueError("unknown repair method")
    before = decide(contract, members, "comparison_contract")
    packets = []
    if before["authorized"]:
        after_members = selected_members(contract, members)
    else:
        if method == "whole_panel_refresh":
            after_members = []
            requested = [asdict(slot) for slot in contract.slots]
        else:
            after_members = selected_members(contract, members)
            requested = revision_plan(contract, members)["rerun"]
            keys = {s.coordinate: s.scientific_key for s in contract.slots}
            requested = [
                {**s, "scientific_key": keys[s["candidate"], s["replicate"]]}
                for s in requested
            ]
        batches = (
            [[s] for s in requested]
            if method == "slot_directed_serial"
            else [requested]
        )
        for batch in batches:
            if not batch:
                continue
            objects, packet = transfer(contract, batch, fetch, object_root)
            packets.append(packet)
            after_members.extend(
                Member(e.actual_candidate, e.actual_replicate, e) for e in objects
            )
    after = decide(contract, after_members, "comparison_contract")
    return {
        "method": method,
        "initial_release": before["authorized"],
        "final_release": after["authorized"],
        "final_decision": after,
        "retained_objects": len(after_members)
        - sum(len(p["response_manifest"]["objects"]) for p in packets),
        "object_transfers": sum(
            len(p["response_manifest"]["objects"]) for p in packets
        ),
        "round_trips": len(packets),
        "request_response_bytes": sum(
            p["request_bytes"] + p["response_bytes"] for p in packets
        ),
        "new_pairs": 0,
        "packets": packets,
    }, after_members


def operation_history_check(case: dict[str, Any]) -> dict[str, Any]:
    """Retain operation histories without mistaking them for final residual truth."""
    operations = case["operations"]
    final = [m["object_id"] for m in case["members"]]
    if operations and operations[-1]["after_objects"] != final:
        raise RuntimeError("retained operation history and final panel differ")
    return {
        "injected_operations": len(operations),
        "operation_families": dict(Counter(o["operation"] for o in operations)),
        "intermediate_object_sequences": [o["after_objects"] for o in operations],
        "final_object_sequence": final,
        "interpretation": (
            "diagnosis target is the final observable residual, "
            "not every historical operation"
        ),
    }
