"""Second-review analyses, mixed handoffs and full-gate workflow measurements.

All new paths are outside frozen source/result namespaces. The original store
is read-only; its output root is scoped only when executing new timing objects.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import statistics
import subprocess
from collections import Counter
from dataclasses import asdict, replace
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from time import perf_counter
from typing import Any
from unittest.mock import patch

from policytwin.coordination.plan import BRANCHES, CoordinationExecutionPlan
from policytwin.coordination.release_gate import Registration, report
from policytwin.coordination.schema import commitment
from policytwin.design_contract.contract import (
    MODES,
    ComparisonContract,
    Evidence,
    ExpectedSlot,
    Member,
    decide,
    revision_plan,
)
from policytwin.design_contract.runtime import SimulationRecipe, configured
from policytwin.provenance import canonical_json_bytes
from scripts.cscwd2027_design_contract import store
from scripts.cscwd2027_design_contract.design import CONFIG as ORIGINAL_CONFIG
from scripts.cscwd2027_design_contract.design import member, number, object_id, recipes
from scripts.cscwd2027_design_contract.evaluate import (
    consequences,
    emit,
    get_contract,
    panel_valid,
)
from scripts.cscwd2027_design_contract.freeze import DEST
from scripts.cscwd2027_design_contract.freeze import check as original_check
from scripts.cscwd2027_design_contract.store import compare, save
from scripts.cscwd2027_review_response import without_permission

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/cscwd2027/review_consequences_v2.json"
SEAL = ROOT / "configs/cscwd2027/review_consequences_v2_freeze.json"
OUT = ROOT / "results/cscwd2027/review_consequences_v2"
OLD = ROOT / "results/cscwd2027/design_contract_v1"
ALL_MODES = (*MODES, "without_permission")


def read(path: Path) -> Any:
    """Read a structured local artifact."""
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    """Bind a complete local file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rational(value: Fraction | int) -> list[int]:
    """Serialize exact arithmetic without a floating-point metric rewrite."""
    value = Fraction(value)
    return [value.numerator, value.denominator]


def gap(reference: dict[str, Any]) -> Fraction:
    """Gap between the top two exact finite-panel means, retaining ties."""
    means = sorted((Fraction(*v) for v in reference["means"].values()), reverse=True)
    return means[0] - means[1]


def cuts(values: list[Fraction]) -> list[Fraction]:
    """Deterministic descriptive quartiles, independent of incoming reports."""
    return list(statistics.quantiles(values, n=4, method="inclusive"))


def explain(contract: ComparisonContract, panel: list[Member]) -> dict[str, Any]:
    """Return slot/reason diagnostics without operation labels or numeric values."""
    expected = {s.coordinate: s.scientific_key for s in contract.slots}
    observed: dict[tuple[str, int], list[Evidence]] = {}
    for m in panel:
        coord = m.evidence.actual_candidate, m.evidence.actual_replicate
        observed.setdefault(coord, []).append(m.evidence)
    issues = set()
    for coord, key in expected.items():
        values = observed.get(coord, [])
        if not values:
            issues.add((*coord, "missing_slot"))
        if len(values) > 1:
            issues.add((*coord, "duplicate_slot"))
        for evidence in values:
            if evidence.scientific_key != key:
                issues.add((*coord, "key_mismatch"))
    for coord, values in observed.items():
        if coord not in expected:
            issues.add((*coord, "unexpected_slot"))
        for evidence in values:
            if not evidence.local_pass or type(evidence.value) is not int:
                issues.add((*coord, "local_invalid"))
            if evidence.plan_sha256 not in contract.allowed_plans:
                issues.add((*coord, "unpermitted_plan"))
    return {
        "issues": [list(i) for i in sorted(issues)],
        "repair_plan": revision_plan(contract, panel),
        "selection": "lexicographically smallest eligible object_id; values unused",
    }


def conditional(
    rows: list[dict[str, Any]], thresholds: list[Fraction]
) -> dict[str, Any]:
    """Add descriptive consequences without replacing original denominators."""
    output = {}
    for mode in sorted({row["mode"] for row in rows}):
        selected = [r for r in rows if r["mode"] == mode]
        invalid = [
            r
            for r in selected
            if r["released"] and not r["valid"] and r["effects"]["assessable"]
        ]
        positive = [
            Fraction(*r["effects"]["regret"])
            for r in invalid
            if r["effects"]["optimal_set_error"]
        ]
        strata = []
        for index in range(4):
            group = [
                r
                for r in invalid
                if sum(Fraction(*r["reference_gap"]) > q for q in thresholds) == index
            ]
            positive_group = [
                Fraction(*r["effects"]["regret"])
                for r in group
                if r["effects"]["optimal_set_error"]
            ]
            strata.append(
                {
                    "quartile": index + 1,
                    "unauthorized_reports": len(group),
                    "flips": sum(r["effects"]["top1_disagreement"] for r in group),
                    "nonoptimal": len(positive_group),
                    "positive_regret_median": rational(
                        statistics.median(positive_group)
                    )
                    if positive_group
                    else None,
                    "positive_regret_max": rational(max(positive_group))
                    if positive_group
                    else None,
                }
            )
        output[mode] = {
            "panels": len(selected),
            "reports": sum(r["released"] for r in selected),
            "unauthorized_reports": len(invalid),
            "all_assessable_reports": sum(
                r["released"] and r["effects"]["assessable"] for r in selected
            ),
            "all_report_flips": sum(
                bool(r["effects"]["top1_disagreement"]) for r in selected
            ),
            "conditional_flips": sum(
                r["effects"]["top1_disagreement"] for r in invalid
            ),
            "conditional_nonoptimal": len(positive),
            "positive_regret_median": rational(statistics.median(positive))
            if positive
            else None,
            "positive_regret_max": rational(max(positive)) if positive else None,
            "gap_strata": strata,
        }
    return output


def verify() -> dict[str, Any]:
    """Keep both scientific substrate and new protocol fixed."""
    original_check()
    seal = read(SEAL)
    for path, expected in seal["members"].items():
        if sha(ROOT / path) != expected:
            raise RuntimeError(f"sealed input changed: {path}")
    return seal


def freeze() -> None:
    """Seal the analysis, random generator and timing boundaries before running."""
    original_check()
    inputs = [
        CONFIG,
        Path(__file__).resolve(),
        ROOT / "tests/test_cscwd_review2.py",
        ROOT / "uv.lock",
        ROOT / "pyproject.toml",
        ROOT / "AGENTS.md",
        DEST / "freeze.json",
        DEST / "contracts.json",
        DEST / "sessions.json",
        DEST / "handoffs.json",
        OLD / "evaluation/results.json",
    ]
    save(
        SEAL,
        {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "clock": "local ordering; not external attestation",
            "config": read(CONFIG),
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "python": platform.python_version(),
            "machine": platform.machine(),
            "members": {str(p.relative_to(ROOT)): sha(p) for p in inputs},
        },
    )
    print("REVIEW2_FROZEN", sha(SEAL), flush=True)


def original_reference(sid: str, version: int) -> dict[str, Any]:
    """Read the retained complete intended panel."""
    return read(
        OLD / "evaluation/references" / sid / f"v{version}/numerical_comparison.json"
    )


def original_panel(session: dict[str, Any], phase: str) -> list[Member]:
    """Read all verified source objects of a complete original panel."""
    return [
        Member(c, r, store.load(object_id(session, phase, c, r))[0])
        for c in BRANCHES
        for r in range(ORIGINAL_CONFIG["replicates"])
    ]


def artifact_analysis() -> None:
    """Run post-hoc consequence analysis and the sealed mixed-panel generator."""
    config = verify()["config"]
    dest = OUT / "artifacts"
    dest.mkdir(parents=True, exist_ok=False)
    sessions = read(DEST / "sessions.json")
    contracts = read(DEST / "contracts.json")
    old_rows = read(OLD / "evaluation/results.json")["handoffs"]
    references = {
        (s["id"], v): original_reference(s["id"], v) for s in sessions for v in (1, 2)
    }
    lookup = {case["id"]: case for case in read(DEST / "handoffs.json")}
    rows = []
    for r in old_rows:
        reference = references[r["session"], lookup[r["case_id"]]["version"]]
        rows.append(
            {
                "mode": r["mode"],
                "case_id": r["case_id"],
                "released": r["decision"]["numerical_file_exists"],
                "valid": r["output_panel_valid"],
                "effects": r["consequences"],
                "reference_gap": rational(gap(reference)),
            }
        )
    quartiles = cuts([gap(value) for value in references.values()])
    original = conditional(rows, quartiles)
    cache: dict[str, Evidence] = {}

    def token(identifier: str) -> Evidence:
        if identifier not in cache:
            cache[identifier] = store.load(identifier)[0]
        return cache[identifier]

    localization = []
    by_id = {s["id"]: s for s in sessions}
    for case in lookup.values():
        session = by_id[case["session"]]
        authority = get_contract(session["id"], case["version"], contracts)
        panel = [
            Member(m["candidate"], m["replicate"], token(m["object_id"]))
            for m in case["members"]
        ]
        diagnostic = explain(authority, panel)
        if case["kind"] not in {
            "foreign_session",
            "missing_slot",
            "duplicate_slot",
            "stale_revision",
        }:
            continue
        if case["kind"] == "stale_revision":
            expected = [
                [session["target"], r, "key_mismatch"]
                for r in range(ORIGINAL_CONFIG["replicates"])
            ]
        else:
            index = number(session["id"] + "handoff-target") % (
                len(BRANCHES) * ORIGINAL_CONFIG["replicates"]
            )
            c, r = [
                (c, r) for c in BRANCHES for r in range(ORIGINAL_CONFIG["replicates"])
            ][index]
            reason = {
                "foreign_session": "key_mismatch",
                "missing_slot": "missing_slot",
                "duplicate_slot": "duplicate_slot",
            }[case["kind"]]
            expected = [[c, r, reason]]
        localization.append(
            {
                "case_id": case["id"],
                "expected": sorted(expected),
                "diagnostic": diagnostic,
                "exact": diagnostic["issues"] == sorted(expected),
            }
        )
    rng = random.Random(config["mixed_handoffs"]["seed"])
    generated = []
    for session in sessions:
        contract = get_contract(session["id"], 2, contracts)
        authority = replace(contract, allowed_plans=contract.allowed_plans[:1])
        for iteration in range(config["mixed_handoffs"]["panels_per_session"]):
            panel = [
                member(session, "fresh", c, r)
                for c in BRANCHES
                for r in range(ORIGINAL_CONFIG["replicates"])
            ]
            operations = []
            for _ in range(rng.randrange(4)):
                operation = rng.choice(config["mixed_handoffs"]["operations"])
                position = rng.randrange(len(panel))
                chosen = panel[position]
                note = {
                    "operation": operation,
                    "position": position,
                    "before": dict(chosen),
                }
                if operation == "missing_slot":
                    panel.pop(position)
                elif operation == "duplicate_slot":
                    panel.insert(rng.randrange(len(panel) + 1), dict(chosen))
                elif operation == "label_substitution":
                    panel[position] = {
                        **chosen,
                        "candidate": rng.choice(
                            [c for c in BRANCHES if c != chosen["candidate"]]
                        ),
                    }
                elif operation == "foreign_session":
                    donor = sessions[
                        (session["index"] // 8) * 8 + (session["batch"] + 1) % 8
                    ]
                    e = token(chosen["object_id"])
                    panel[position] = {
                        **chosen,
                        "object_id": object_id(
                            donor, "fresh", e.actual_candidate, e.actual_replicate
                        ),
                    }
                elif operation == "stale_revision":
                    r = rng.randrange(ORIGINAL_CONFIG["replicates"])
                    replacement = member(session, "initial", session["target"], r)
                    found = next(
                        (
                            i
                            for i, m in enumerate(panel)
                            if (
                                token(m["object_id"]).actual_candidate,
                                token(m["object_id"]).actual_replicate,
                            )
                            == (session["target"], r)
                        ),
                        position,
                    )
                    panel[found] = replacement
                    note["replacement_position"] = found
                else:
                    c = rng.choice([c for c in BRANCHES if c != session["target"]])
                    replacement = member(session, "remap", c, 0)
                    found = next(
                        (
                            i
                            for i, m in enumerate(panel)
                            if (
                                token(m["object_id"]).actual_candidate,
                                token(m["object_id"]).actual_replicate,
                            )
                            == (c, 0)
                        ),
                        position,
                    )
                    panel[found] = replacement
                    note["replacement_position"] = found
                note["after_objects"] = [m["object_id"] for m in panel]
                operations.append(note)
            generated.append(
                {
                    "id": f"{session['id']}-mixed-{iteration:02d}",
                    "session": session["id"],
                    "contract": asdict(authority),
                    "members": panel,
                    "operations": operations,
                }
            )
    save(dest / "generated_panels.json", generated)
    mixed = []
    for case in generated:
        authority = get_contract(case["session"], 2, contracts)
        authority = replace(authority, allowed_plans=authority.allowed_plans[:1])
        panel = [
            Member(m["candidate"], m["replicate"], token(m["object_id"]))
            for m in case["members"]
        ]
        reference = references[case["session"], 2]
        diagnostic = explain(authority, panel)
        for mode in ALL_MODES:
            result = (
                without_permission(authority, panel)
                if mode == "without_permission"
                else decide(authority, panel, mode)
            )
            target = dest / "reports" / case["id"] / mode
            if result["authorized"]:
                save(target / "numerical_comparison.json", result["numerical"])
            save(target / "decision.json", result)
            mixed.append(
                {
                    "case_id": case["id"],
                    "mode": mode,
                    "session": case["session"],
                    "operation_count": len(case["operations"]),
                    "released": result["authorized"],
                    "valid": panel_valid(authority, result["numerical"]),
                    "effects": consequences(reference, result["numerical"]),
                    "reference_gap": rational(gap(reference)),
                    "diagnostic": diagnostic,
                }
            )
    mixed_cuts = cuts([gap(references[s["id"], 2]) for s in sessions])
    save(
        dest / "summary.json",
        {
            "label": config["label"],
            "original_posthoc": original,
            "original_gap_cuts": [rational(q) for q in quartiles],
            "original_rows": rows,
            "localization": localization,
            "localization_exact": sum(r["exact"] for r in localization),
            "mixed_rows": mixed,
            "mixed": conditional(mixed, mixed_cuts),
            "mixed_gap_cuts": [rational(q) for q in mixed_cuts],
            "operation_count_distribution": dict(
                Counter(len(c["operations"]) for c in generated)
            ),
            "mixed_panels": len(generated),
            "added_simulation_pairs": 0,
        },
    )
    verify()
    print("REVIEW2_ARTIFACTS_COMPLETE", len(generated), flush=True)


def load_from(folder: Path, identifier: str) -> tuple[Evidence, dict[str, Any]]:
    """Scope new-output reading without changing a frozen implementation file."""
    with patch.object(store, "RESULTS", folder):
        return store.load(identifier)


def execute_new(
    folder: Path, session: dict[str, Any], recipe: SimulationRecipe, phase: str
) -> tuple[Member, dict[str, Any]]:
    """Run the unchanged complete simulator/evidence path at a new output root."""
    row = {
        "id": object_id(session, phase, recipe.spec.branch, recipe.spec.replicate),
        "session": session["id"],
        "phase": phase,
        "candidate": recipe.spec.branch,
        "replicate": recipe.spec.replicate,
        "recipe": recipe.to_mapping(),
        "plan": ORIGINAL_CONFIG["plans"][0],
    }
    with patch.object(store, "RESULTS", folder):
        store.execute(row, sha(SEAL))
        token, capsule = store.load(row["id"])
    return Member(recipe.spec.branch, recipe.spec.replicate, token), capsule


def revalidate_old(
    session: dict[str, Any], c: str, r: int, dest: Path
) -> tuple[Member, dict[str, Any], dict[str, Any]]:
    """Freshly replay an old capsule into new report files; never edit its store."""
    identifier = object_id(session, "initial", c, r)
    token, capsule = store.load(identifier)
    result = read(OLD / "objects" / identifier / "result.json")
    recipe = SimulationRecipe.from_mapping(result["design"]["recipe"])
    plan = CoordinationExecutionPlan.from_mapping(result["design"]["plan"])
    fields = dict(capsule["registration"])
    fields["required_metrics"] = tuple(fields["required_metrics"])
    with configured(recipe):
        decision = report(
            recipe.spec, plan, capsule, Registration(**fields), dest, "full"
        )
    if not decision["release_authorized"]:
        token = replace(token, local_pass=False, value=None)
    return Member(c, r, token), capsule, decision


def workflow() -> None:
    """Measure complete revision paths, including replay of every reused object."""
    verify()
    dest = OUT / "workflow"
    dest.mkdir(parents=True, exist_ok=False)
    sessions = read(DEST / "sessions.json")
    contracts = read(DEST / "contracts.json")
    for session in sessions:
        authority = get_contract(session["id"], 2, contracts)
        session_out = dest / session["id"]
        available = original_panel(session, "initial")
        request = revision_plan(authority, available)
        needs = {(v["candidate"], v["replicate"]) for v in request["rerun"]}
        order = (
            ("fresh", "reuse")
            if number(session["id"] + "review-e2e-order") % 2 == 0
            else ("reuse", "fresh")
        )
        records: dict[str, Any] = {}
        tokens: dict[str, list[Member]] = {}
        capsules: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
        for phase in order:
            root = session_out / phase
            started = perf_counter()
            if phase == "reuse":
                timed_available = original_panel(session, "initial")
                timed_request = revision_plan(authority, timed_available)
                needs = {
                    (v["candidate"], v["replicate"]) for v in timed_request["rerun"]
                }
            members, caps, replay_decisions = [], {}, []
            new_count = 0
            for coord, recipe in recipes(session, 2).items():
                if phase == "fresh" or coord in needs:
                    value, cap = execute_new(root, session, recipe, phase)
                    new_count += 1
                else:
                    value, cap, decision = revalidate_old(
                        session,
                        *coord,
                        root / "reused_replay" / f"{coord[0]}-{coord[1]}",
                    )
                    replay_decisions.append(decision)
                members.append(value)
                caps[coord] = cap
            decision = emit(root / "panel", authority, members, "comparison_contract")
            elapsed = perf_counter() - started
            records[phase] = {
                "elapsed_seconds": elapsed,
                "new_pairs": new_count,
                "reused_pairs": len(members) - new_count,
                "fresh_replay_of_reused": replay_decisions,
                "decision": decision,
            }
            tokens[phase], capsules[phase] = members, caps
        slot_checks = [
            {
                "candidate": c,
                "replicate": r,
                **compare(capsules["fresh"][c, r], capsules["reuse"][c, r]),
            }
            for c, r in recipes(session, 2)
        ]
        start_subset = replace(
            authority,
            slots=tuple(s for s in authority.slots if s.candidate != session["target"]),
        )
        old_subset = [
            m for m in available if m.evidence.actual_candidate != session["target"]
        ]
        subset_decision = emit(
            session_out / "two_candidate_start",
            start_subset,
            old_subset,
            "comparison_contract",
        )
        addition_plan = revision_plan(authority, old_subset)
        addition = emit(
            session_out / "add_third_candidate",
            authority,
            tokens["reuse"],
            "comparison_contract",
        )
        add_record = {
            "initial_slots": len(start_subset.slots),
            "initial_decision": subset_decision,
            "plan": addition_plan,
            "new_execution_shared_with_timing": True,
            "decision": addition,
        }
        # New replicates extend the same seed recipe, without altering old specs.
        extensions, extension_caps = [], []
        extra_slots = []
        for c in BRANCHES:
            template = recipes(session, 2)[c, 0]
            for r in range(
                ORIGINAL_CONFIG["replicates"], 2 * ORIGINAL_CONFIG["replicates"]
            ):
                seed = commitment([ORIGINAL_CONFIG["seed_root"], session["id"], r])
                recipe = replace(
                    template, spec=replace(template.spec, seed=seed, replicate=r)
                )
                extra_slots.append(ExpectedSlot(c, r, recipe.scientific_key))
                value, cap = execute_new(
                    session_out / "replicate_extension", session, recipe, "extend"
                )
                extensions.append(value)
                extension_caps.append(
                    {
                        "candidate": c,
                        "replicate": r,
                        "completed": bool(cap) and value.evidence.local_pass,
                    }
                )
        extended = replace(authority, version=3, slots=(*authority.slots, *extra_slots))
        old_revised = original_panel(session, "fresh")
        extension_plan = revision_plan(extended, old_revised)
        reused_panel = emit(
            session_out / "replicate_extension/panel",
            extended,
            [*old_revised, *extensions],
            "comparison_contract",
        )
        shared_reference = emit(
            session_out / "replicate_extension/shared_new_reference",
            extended,
            [*tokens["fresh"], *extensions],
            "comparison_contract",
        )
        equality = (
            reused_panel["numerical"] is not None
            and shared_reference["numerical"] is not None
            and reused_panel["numerical"]["means"]
            == shared_reference["numerical"]["means"]
            and reused_panel["numerical"]["top1"]
            == shared_reference["numerical"]["top1"]
        )
        result = {
            "session": session["id"],
            "agents": session["agents"],
            "regime": session["regime"],
            "target": session["target"],
            "order": order,
            "revision_plan": request,
            "paths": records,
            "slot_checks": slot_checks,
            "add_candidate": add_record,
            "replicate_extension": {
                "plan": extension_plan,
                "new_pairs": len(extensions),
                "reused_pairs": len(old_revised),
                "same_means_and_choice": equality,
                "shared_new_slot_reference": True,
                "new_slot_status": extension_caps,
            },
        }
        save(session_out / "result.json", result)
        with (dest / "progress.jsonl").open("ab") as log:
            log.write(
                canonical_json_bytes({"session": session["id"], "completed": True})
                + b"\n"
            )
        print(
            "WORKFLOW_SESSION",
            session["id"],
            f"fresh={records['fresh']['elapsed_seconds']:.3f}s",
            f"reuse={records['reuse']['elapsed_seconds']:.3f}s",
            flush=True,
        )
    verify()


def summarize() -> None:
    """Retain all timed blocks, failed objects and zero/nonzero contrasts."""
    verify()
    sessions = read(DEST / "sessions.json")
    rows = [read(OUT / "workflow" / s["id"] / "result.json") for s in sessions]
    objects = sorted((OUT / "workflow").glob("*/**/objects/*/result.json"))
    statuses = Counter(read(p)["status"] for p in objects)
    rejected = sum(not read(p).get("local_pass", False) for p in objects)
    ratios = [
        r["paths"]["reuse"]["elapsed_seconds"] / r["paths"]["fresh"]["elapsed_seconds"]
        for r in rows
    ]
    timing = {}
    for a, regime in sorted({(r["agents"], r["regime"]) for r in rows}):
        group = [r for r in rows if (r["agents"], r["regime"]) == (a, regime)]
        timing[f"{a}/{regime}"] = {
            "sessions": len(group),
            "fresh_median_s": statistics.median(
                r["paths"]["fresh"]["elapsed_seconds"] for r in group
            ),
            "reuse_median_s": statistics.median(
                r["paths"]["reuse"]["elapsed_seconds"] for r in group
            ),
            "reuse_fresh_ratio_median": statistics.median(
                r["paths"]["reuse"]["elapsed_seconds"]
                / r["paths"]["fresh"]["elapsed_seconds"]
                for r in group
            ),
        }
    checks = [c for row in rows for c in row["slot_checks"]]
    summary = {
        "label": read(CONFIG)["label"],
        "sessions": len(rows),
        "new_pairs": len(objects),
        "statuses": dict(statuses),
        "locally_rejected": rejected,
        "timing_cells": timing,
        "reuse_fresh_ratio_median": statistics.median(ratios),
        "reuse_fresh_ratio_range": [min(ratios), max(ratios)],
        "scientific_trace_matches": sum(c["science_equal"] for c in checks),
        "slot_comparisons": len(checks),
        "draw_matches": sum(c["draw_equal"] for c in checks),
        "add_candidate_released": sum(
            r["add_candidate"]["decision"]["authorized"] for r in rows
        ),
        "extension_same_means_and_choice": sum(
            r["replicate_extension"]["same_means_and_choice"] for r in rows
        ),
        "extension_reference": "independently fresh old slots plus shared newly "
        "executed slots; not an independent full eight-replicate rerun",
        "rows": rows,
        "artifact_summary": str((OUT / "artifacts/summary.json").relative_to(ROOT)),
    }
    save(OUT / "summary.json", summary)
    save(
        OUT / "manifest.json",
        {
            "freeze_sha256": sha(SEAL),
            "summary_sha256": sha(OUT / "summary.json"),
            "artifact_summary_sha256": sha(OUT / "artifacts/summary.json"),
            "object_results": {str(p.relative_to(ROOT)): sha(p) for p in objects},
        },
    )
    print(
        json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=("freeze", "artifacts", "workflow", "summarize", "check")
    )
    action = parser.parse_args().action
    if action == "freeze":
        freeze()
    elif action == "artifacts":
        artifact_analysis()
    elif action == "workflow":
        workflow()
    elif action == "summarize":
        summarize()
    else:
        verify()
        print("REVIEW2_FREEZE_OK")
