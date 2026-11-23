"""Actual comparison-report files and frozen handoff/revision evaluation."""

from __future__ import annotations

import json
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any

from policytwin.design_contract.contract import (
    ComparisonContract,
    Evidence,
    ExpectedSlot,
    Member,
    decide,
    revision_plan,
)
from scripts.cscwd2027_design_contract.design import CONFIG, member, object_id
from scripts.cscwd2027_design_contract.freeze import DEST, check
from scripts.cscwd2027_design_contract.store import (
    RESULTS,
    compare,
    load,
    members,
    save,
)


def get_contract(
    session: str, version: int, contracts: dict[str, Any]
) -> ComparisonContract:
    """Use the independent pre-run contract, not a received artifact-derived set."""
    raw = contracts[session][str(version)]
    return ComparisonContract(
        raw["session_id"],
        raw["version"],
        tuple(ExpectedSlot(**s) for s in raw["slots"]),
        tuple(raw["allowed_plans"]),
        raw["rule"],
    )


def emit(
    path: Path, contract: ComparisonContract, values: list[Member], mode: str
) -> dict[str, Any]:
    """Reevaluate before creating the actual numerical comparison artifact."""
    path.mkdir(parents=True, exist_ok=False)
    result = decide(contract, values, mode)
    if result["authorized"]:
        save(path / "numerical_comparison.json", result["numerical"])
    save(path / "decision.json", result)
    return {
        **result,
        "numerical_file_exists": (path / "numerical_comparison.json").is_file(),
    }


def panel_valid(contract: ComparisonContract, numerical: dict[str, Any] | None) -> bool:
    """Audit emitted effective bindings; do not use incoming operation labels."""
    if numerical is None:
        return False
    expected = Counter(
        (s.candidate, s.replicate, s.scientific_key) for s in contract.slots
    )
    actual = Counter(
        (m["candidate"], m["replicate"], m["scientific_key"])
        for m in numerical["effective_bindings"]
    )
    return expected == actual and all(
        m["local_pass"] and m["plan_sha256"] in contract.allowed_plans
        for m in numerical["effective_bindings"]
    )


def consequences(
    reference: dict[str, Any] | None, numerical: dict[str, Any] | None
) -> dict[str, Any]:
    """Keep abstention, tied choices and worse finite-panel choices distinct."""
    if reference is None or numerical is None:
        return {
            "assessable": False,
            "top1_disagreement": None,
            "top2_disagreement": None,
            "optimal_set_error": None,
            "regret": None,
            "zero_regret_disagreement": None,
        }
    means = {key: Fraction(*value) for key, value in reference["means"].items()}
    selected = numerical["top1"]
    regret = max(means.values()) - means[selected]
    mismatch = selected != reference["top1"]
    return {
        "assessable": True,
        "top1_disagreement": mismatch,
        "top2_disagreement": numerical["top2"] != reference["top2"],
        "optimal_set_error": regret > 0,
        "regret": [regret.numerator, regret.denominator],
        "zero_regret_disagreement": mismatch and regret == 0,
    }


def main() -> None:
    """Evaluate every declared panel, source-valid control and design revision."""
    check()
    blocks = json.loads((DEST / "sessions.json").read_text())
    jobs = json.loads((DEST / "attempts.json").read_text())
    handoffs = json.loads((DEST / "handoffs.json").read_text())
    contracts = json.loads((DEST / "contracts.json").read_text())
    tokens: dict[str, Evidence] = {}
    for job in jobs:
        tokens[job["id"]] = load(job["id"])[0]
    destination = RESULTS / "evaluation"
    destination.mkdir(parents=True, exist_ok=False)
    references = {}
    revisions = []
    remaps = []
    for session in blocks:
        sid = session["id"]
        c0, c1 = (get_contract(sid, v, contracts) for v in (1, 2))
        initial = [
            member(session, "initial", s.candidate, s.replicate) for s in c0.slots
        ]
        fresh = [member(session, "fresh", s.candidate, s.replicate) for s in c1.slots]
        old = members(initial, tokens)
        for version, contract, data in ((1, c0, old), (2, c1, members(fresh, tokens))):
            result = emit(
                destination / "references" / sid / f"v{version}",
                contract,
                data,
                "comparison_contract",
            )
            references[sid, version] = result["numerical"]
        planned = revision_plan(c1, old)
        updated = [
            member(
                session,
                "selective" if s.candidate == session["target"] else "initial",
                s.candidate,
                s.replicate,
            )
            for s in c1.slots
        ]
        revised = emit(
            destination / "revisions" / sid,
            c1,
            members(updated, tokens),
            "comparison_contract",
        )
        stale = emit(
            destination / "stale-local-cache" / sid, c1, old, "local_full_only"
        )
        comparisons = []
        for spec in c1.slots:
            left = load(object_id(session, "fresh", spec.candidate, spec.replicate))[1]
            phase = "selective" if spec.candidate == session["target"] else "initial"
            right = load(object_id(session, phase, spec.candidate, spec.replicate))[1]
            comparisons.append(
                {
                    "candidate": spec.candidate,
                    "replicate": spec.replicate,
                    **compare(left, right),
                }
            )
        revisions.append(
            {
                "session": sid,
                "agents": session["agents"],
                "regime": session["regime"],
                "target": session["target"],
                "plan": planned,
                "budget_infeasible": len(planned["rerun"]) > CONFIG["new_pair_budget"],
                "selective": revised,
                "stale_local": stale,
                "reference_available": references[sid, 2] is not None,
                "selective_consequences": consequences(
                    references[sid, 2], revised["numerical"]
                ),
                "stale_consequences": consequences(
                    references[sid, 2], stale["numerical"]
                ),
                "per_slot_comparisons": comparisons,
                "selective_new_pairs": sum(
                    j["phase"] == "selective" and j["session"] == sid for j in jobs
                ),
                "fresh_new_pairs": sum(
                    j["phase"] == "fresh" and j["session"] == sid for j in jobs
                ),
                "reused_pairs": sum(s.candidate != session["target"] for s in c1.slots),
                "initial_pairs": len(c0.slots),
            }
        )
        for candidate in c0.candidates:
            initial_cap = load(object_id(session, "initial", candidate, 0))[1]
            remap_cap = load(object_id(session, "remap", candidate, 0))[1]
            remaps.append(
                {
                    "session": sid,
                    "candidate": candidate,
                    **compare(initial_cap, remap_cap),
                }
            )
    decisions = []
    for case in handoffs:
        contract = get_contract(case["session"], case["version"], contracts)
        data = members(case["members"], tokens)
        for mode in CONFIG["collector_modes"]:
            result = emit(
                destination / "handoffs" / case["id"] / mode, contract, data, mode
            )
            decisions.append(
                {
                    "case_id": case["id"],
                    "case_kind": case["kind"],
                    "session": case["session"],
                    "mode": mode,
                    "all_observed_local_valid": all(
                        m.evidence.local_pass for m in data
                    ),
                    "nominal_valid_or_recoverable": case["kind"]
                    in {"clean", "authorized_remap", "label_cycle"},
                    "reference_available": references[case["session"], case["version"]]
                    is not None,
                    "output_panel_valid": panel_valid(contract, result["numerical"]),
                    "decision": result,
                    "consequences": consequences(
                        references[case["session"], case["version"]],
                        result["numerical"],
                    ),
                }
            )
    save(
        destination / "results.json",
        {"handoffs": decisions, "revisions": revisions, "remap_comparisons": remaps},
    )
    check()
    print(
        json.dumps(
            {
                "handoff_decisions": len(decisions),
                "revisions": len(revisions),
                "remap_comparisons": len(remaps),
            }
        )
    )


if __name__ == "__main__":
    main()
