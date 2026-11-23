"""Pre-run specified mechanism analyses; all publication counts derive here."""

from __future__ import annotations

import json
import statistics
from dataclasses import replace
from fractions import Fraction
from functools import cache
from pathlib import Path
from typing import Any

from policytwin.coordination.plan import BRANCHES, CoordinationExecutionPlan
from policytwin.cscwd_mechanism.coupling import (
    COUPLINGS,
    CoupledRecipe,
    compile_coupling,
    load,
    marginal_schema,
)
from policytwin.cscwd_mechanism.design import identity
from policytwin.cscwd_mechanism.repair import (
    localization_score,
    operation_history_check,
    repair_panel,
    semantic_truth,
)
from policytwin.design_contract.contract import (
    ComparisonContract,
    ExpectedSlot,
    Member,
    decide,
)
from scripts.cscwd2027_design_contract import store
from scripts.cscwd2027_design_contract.evaluate import (
    consequences,
    get_contract,
    panel_valid,
)
from scripts.cscwd2027_review2 import explain


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rational(value: Fraction | int | float) -> list[int]:
    value = Fraction(value)
    return [value.numerator, value.denominator]


def distribution(values: list[int | float | Fraction]) -> dict[str, Any]:
    if not values:
        return {
            "n": 0,
            "median": None,
            "q1": None,
            "q3": None,
            "min": None,
            "max": None,
        }
    quartiles = (
        statistics.quantiles(values, n=4, method="inclusive")
        if len(values) > 1
        else [values[0]] * 3
    )
    return {
        "n": len(values),
        "median": float(statistics.median(values)),
        "q1": float(quartiles[0]),
        "q3": float(quartiles[2]),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def variant_decide(contract, members, mode, projections):
    if mode != "without_coupling":
        return decide(contract, members, mode)
    projected = replace(
        contract,
        slots=tuple(
            replace(s, scientific_key=projections[s.scientific_key])
            for s in contract.slots
        ),
    )
    panel = [
        replace(
            m,
            evidence=replace(
                m.evidence, scientific_key=projections[m.evidence.scientific_key]
            ),
        )
        for m in members
    ]
    result = decide(projected, panel, "comparison_contract")
    if result["authorized"]:
        actual = {m.evidence.object_id: m.evidence for m in members}
        # Record the actual full object identities, not the ablated key projection,
        # so an independent full-contract audit can distinguish clean from wrong C.
        for binding in result["numerical"]["effective_bindings"]:
            binding["scientific_key"] = actual[binding["object_id"]].scientific_key
        result["numerical"]["mode"] = mode
        result["numerical"]["contract_sha256"] = contract.sha256
    result["acceptance_projection"] = "all key fields except coupling_contract_sha256"
    return result


def coupling_analysis(
    root: Path,
    contexts: list[dict[str, Any]],
    attempts: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    @cache
    def token(identifier):
        return load(identifier, root)[0]

    rows_by_id = {r["id"]: r for r in attempts}
    projections = {}
    for row in attempts:
        recipe = CoupledRecipe.from_mapping(row["coupled_recipe"])
        projections[recipe.scientific_key] = recipe.key_without_coupling
    local = [read(root / "objects" / r["id"] / "result.json") for r in attempts]
    schemas = []
    for row in attempts:
        if row["coupling"] != "shared":
            continue
        recipe = CoupledRecipe.from_mapping(row["coupled_recipe"]).recipe
        a, b = (compile_coupling(recipe.spec, coupling=c) for c in COUPLINGS)
        schemas.append(
            {
                "id": row["id"],
                "match": all(
                    marginal_schema(a, w) == marginal_schema(b, w)
                    for w in ("baseline", "policy")
                ),
            }
        )
    output, variance = [], []
    for context in contexts:
        session = context["session"]
        sid, h, count = session["id"], context["horizon"], context["replicates"]
        for candidate in BRANCHES:
            values = {}
            for coupling in COUPLINGS:
                group = [
                    token(identity(sid, h, candidate, r, coupling))
                    for r in range(count)
                ]
                values[coupling] = (
                    [Fraction(e.value) for e in group]
                    if all(e.local_pass for e in group)
                    else None
                )
            samples = {
                c: rational(statistics.variance(v)) if v is not None else None
                for c, v in values.items()
            }
            ratio = (
                Fraction(*samples["world_separated"]) / Fraction(*samples["shared"])
                if all(samples.values()) and Fraction(*samples["shared"]) != 0
                else None
            )
            variance.append(
                {
                    "context": context["id"],
                    "session": sid,
                    "candidate": candidate,
                    "agents": session["agents"],
                    "regime": session["regime"],
                    "horizon": h,
                    "replicates": count,
                    "families": context["families"],
                    "sample_variance": samples,
                    "separate_shared_ratio": rational(ratio)
                    if ratio is not None
                    else None,
                }
            )
        for requested in COUPLINGS:
            opposite = COUPLINGS[1 - COUPLINGS.index(requested)]
            expected = []
            clean = []
            for candidate in BRANCHES:
                for replicate in range(count):
                    identifier = identity(sid, h, candidate, replicate, requested)
                    recipe = CoupledRecipe.from_mapping(
                        rows_by_id[identifier]["coupled_recipe"]
                    )
                    expected.append(
                        ExpectedSlot(candidate, replicate, recipe.scientific_key)
                    )
                    clean.append(Member(candidate, replicate, token(identifier)))
            contract = ComparisonContract(
                context["id"] + "-" + requested,
                1,
                tuple(expected),
                (CoordinationExecutionPlan(2).sha256,),
            )
            reference = decide(contract, clean, "comparison_contract")["numerical"]
            target = session["target"], context["target_replicate"]
            for case in config["panel_cases"]:
                members = list(clean)
                position = next(
                    i for i, m in enumerate(members) if m.coordinate == target
                )
                if case == "coupling_switch_target":
                    members = [
                        Member(
                            m.candidate,
                            m.replicate,
                            token(identity(sid, h, m.candidate, m.replicate, opposite)),
                        )
                        if m.candidate == session["target"]
                        else m
                        for m in members
                    ]
                elif case == "missing_slot":
                    members.pop(position)
                elif case == "duplicate_slot":
                    members.append(members[position])
                panel_id = context["id"] + "-" + requested + "-" + case
                for mode in config["collectors"]:
                    result = variant_decide(contract, members, mode, projections)
                    dest = root / "handoffs" / panel_id / mode
                    if result["authorized"]:
                        store.save(
                            dest / "numerical_comparison.json", result["numerical"]
                        )
                    store.save(dest / "decision.json", result)
                    effects = consequences(reference, result["numerical"])
                    output.append(
                        {
                            "panel": panel_id,
                            "context": context["id"],
                            "session": sid,
                            "agents": session["agents"],
                            "regime": session["regime"],
                            "families": context["families"],
                            "horizon": h,
                            "replicates": count,
                            "requested_coupling": requested,
                            "case": case,
                            "mode": mode,
                            "released": result["authorized"],
                            "valid": panel_valid(contract, result["numerical"]),
                            "effects": effects,
                        }
                    )
    store.save(root / "analysis/coupling_rows.json", output)
    store.save(root / "analysis/variance_rows.json", variance)
    store.save(root / "analysis/marginal_schema_checks.json", schemas)
    primary = [
        r
        for r in output
        if "primary" in r["families"] and r["case"] == "coupling_switch_target"
    ]
    summary = {}
    for mode in config["collectors"]:
        group = [r for r in primary if r["mode"] == mode]
        summary[mode] = {
            "panels": len(group),
            "released": sum(r["released"] for r in group),
            "unauthorized": sum(r["released"] and not r["valid"] for r in group),
            "assessable_reports": sum(r["effects"]["assessable"] for r in group),
            "top1_flips": sum(bool(r["effects"]["top1_disagreement"]) for r in group),
            "positive_regret": sum(
                bool(r["effects"]["optimal_set_error"]) for r in group
            ),
        }
    groups = {}
    for agents in (6, 24):
        for regime in ("deep", "responsive"):
            for candidate in BRANCHES:
                group = [
                    r
                    for r in variance
                    if "primary" in r["families"]
                    and r["agents"] == agents
                    and r["regime"] == regime
                    and r["candidate"] == candidate
                ]
                groups[f"a{agents}-{regime}-{candidate}"] = {
                    c: distribution(
                        [
                            Fraction(*r["sample_variance"][c])
                            for r in group
                            if r["sample_variance"][c] is not None
                        ]
                    )
                    for c in COUPLINGS
                }
    return {
        "pairs": len(attempts),
        "completed": sum(r["status"] == "completed" for r in local),
        "local_pass": sum(r["local_pass"] for r in local),
        "failed_ids": [r["id"] for r in local if r["status"] != "completed"],
        "local_reject_ids": [r["id"] for r in local if not r["local_pass"]],
        "marginal_schema_pairs": len(schemas),
        "marginal_schema_matches": sum(r["match"] for r in schemas),
        "contexts": len(contexts),
        "panels": len(output) // len(config["collectors"]),
        "primary_switch": summary,
        "primary_variance_by_cell_branch": groups,
    }


def repair_analysis(project: Path, root: Path) -> dict[str, Any]:
    old = project / "results/cscwd2027/design_contract_v1"
    config = project / "configs/cscwd2027/design_contract_v1"
    cases = read(
        project
        / "results/cscwd2027/review_consequences_v2/artifacts/generated_panels.json"
    )
    catalog = {r["id"]: r for r in read(config / "attempts.json")}
    contracts = read(config / "contracts.json")

    @cache
    def token(identifier):
        return store.load(identifier)[0]

    rows = []
    for case in cases:
        sid = case["session"]
        authority = get_contract(sid, 2, contracts)
        authority = replace(authority, allowed_plans=authority.allowed_plans[:1])
        fresh = {
            (r["candidate"], r["replicate"]): r
            for r in catalog.values()
            if r["session"] == sid and r["phase"] == "fresh"
        }
        expected = {coordinate: r["recipe"] for coordinate, r in fresh.items()}
        raw = case["members"]
        tokens = {m["object_id"]: token(m["object_id"]) for m in raw}
        members = [
            Member(m["candidate"], m["replicate"], tokens[m["object_id"]]) for m in raw
        ]
        truth = semantic_truth(
            expected, [next(iter(fresh.values()))["plan"]], raw, catalog, tokens
        )
        diagnostic = explain(authority, members)
        history = operation_history_check(case)
        reference = decide(
            authority,
            [Member(c, r, token(row["id"])) for (c, r), row in fresh.items()],
            "comparison_contract",
        )["numerical"]
        repairs = []
        for method in (
            "slot_directed_batch",
            "slot_directed_serial",
            "whole_panel_refresh",
        ):
            result, _ = repair_panel(
                authority,
                members,
                method,
                lambda c, r, fresh=fresh: token(fresh[c, r]["id"]),
                old / "objects",
            )
            result["same_reference_means_and_choice"] = result["final_release"] and all(
                result["final_decision"]["numerical"][k] == reference[k]
                for k in ("means", "top1", "top2")
            )
            repairs.append(result)
        rows.append(
            {
                "id": case["id"],
                "session": sid,
                "truth": truth,
                "predicted": diagnostic["issues"],
                "score": localization_score(diagnostic["issues"], truth),
                "history": history,
                "repairs": repairs,
            }
        )
    store.save(root / "analysis/repair_rows.json", rows)
    summary = {}
    for method in (
        "slot_directed_batch",
        "slot_directed_serial",
        "whole_panel_refresh",
    ):
        results = [
            next(r for r in row["repairs"] if r["method"] == method) for row in rows
        ]
        invalid = [r for r in results if not r["initial_release"]]
        summary[method] = {
            "panels": len(results),
            "initially_invalid": len(invalid),
            "final_release": sum(r["final_release"] for r in results),
            "same_reference_means_and_choice": sum(
                r["same_reference_means_and_choice"] for r in results
            ),
            "object_transfers": sum(r["object_transfers"] for r in results),
            "round_trips": sum(r["round_trips"] for r in results),
            "request_response_bytes": sum(r["request_response_bytes"] for r in results),
            "new_pairs": sum(r["new_pairs"] for r in results),
            "invalid_panel_transfer_distribution": distribution(
                [r["object_transfers"] for r in invalid]
            ),
            "invalid_panel_round_trip_distribution": distribution(
                [r["round_trips"] for r in invalid]
            ),
        }
    true_slots = sum(r["score"]["true_slots"] for r in rows)
    predicted_slots = sum(r["score"]["predicted_slots"] for r in rows)
    correct_slots = sum(r["score"]["correct_slots"] for r in rows)
    return {
        "panels": len(rows),
        "exact_reason_sets": sum(r["score"]["exact_reason_set"] for r in rows),
        "exact_slot_sets": sum(r["score"]["exact_slot_set"] for r in rows),
        "micro_slot_precision": [correct_slots, predicted_slots],
        "micro_slot_recall": [correct_slots, true_slots],
        "nonzero_operation_zero_residual": [
            r["id"]
            for r in rows
            if r["history"]["injected_operations"] and not r["truth"]
        ],
        "mismatch_ids": [r["id"] for r in rows if not r["score"]["exact_reason_set"]],
        "methods": summary,
    }


def auxiliary_analysis(project: Path, root: Path) -> dict[str, Any]:
    publication = read(
        project / "results/cscwd2027/review_consequences_v2/publication_summary.json"
    )
    workflow = publication["workflow"]["rows"]
    timing = []
    for row in workflow:
        fresh, reuse = (row["paths"][p]["elapsed_seconds"] for p in ("fresh", "reuse"))
        timing.append(
            {
                "session": row["session"],
                "agents": row["agents"],
                "regime": row["regime"],
                "fresh_seconds": fresh,
                "reuse_seconds": reuse,
                "paired_change_seconds": reuse - fresh,
                "reuse_fresh_ratio": reuse / fresh,
                "original_order": row["order"],
            }
        )
    old = publication["artifact_analysis"]
    handoffs = {
        r["id"]: r
        for r in read(project / "configs/cscwd2027/design_contract_v1/handoffs.json")
    }
    consequences_rows = []
    for family in ("original_rows", "mixed_rows"):
        for row in old[family]:
            if not (
                row["released"] and not row["valid"] and row["effects"]["assessable"]
            ):
                continue
            sid = row["session"]
            reference_version = (
                2 if family == "mixed_rows" else handoffs[row["case_id"]]["version"]
            )
            reference = read(
                project
                / "results/cscwd2027/design_contract_v1/evaluation/references"
                / sid
                / f"v{reference_version}/numerical_comparison.json"
            )
            scale = max(
                Fraction(1), max(abs(Fraction(*m)) for m in reference["means"].values())
            )
            consequences_rows.append(
                {
                    "family": family,
                    "case_id": row["case_id"],
                    "mode": row["mode"],
                    "regret": row["effects"]["regret"],
                    "top1_flip": row["effects"]["top1_disagreement"],
                    "reference_version": reference_version,
                    "reference_scale": rational(scale),
                    "normalized_regret": rational(
                        Fraction(*row["effects"]["regret"]) / scale
                    ),
                }
            )
    store.save(root / "analysis/old_timing_rows.json", timing)
    store.save(root / "analysis/old_consequence_rows.json", consequences_rows)
    timing_cells = {}
    for agents, regime in ((a, r) for a in (6, 24) for r in ("deep", "responsive")):
        group = [r for r in timing if r["agents"] == agents and r["regime"] == regime]
        timing_cells[f"a{agents}-{regime}"] = {
            k: distribution([r[k] for r in group])
            for k in (
                "fresh_seconds",
                "reuse_seconds",
                "paired_change_seconds",
                "reuse_fresh_ratio",
            )
        }
    regrets = {}
    for family, case, mode in sorted(
        {
            (
                r["family"],
                r["case_id"].split("-s", 1)[1].split("-", 1)[1]
                if r["family"] == "original_rows"
                else "mixed",
                r["mode"],
            )
            for r in consequences_rows
        }
    ):
        group = [
            r
            for r in consequences_rows
            if r["family"] == family
            and r["mode"] == mode
            and (family == "mixed_rows" or r["case_id"].endswith("-" + case))
        ]
        regrets[family + "/" + case + "/" + mode] = {
            "unauthorized_reports": len(group),
            "top1_flips": sum(r["top1_flip"] for r in group),
            "absolute_regret": distribution([Fraction(*r["regret"]) for r in group]),
            "normalized_regret": distribution(
                [Fraction(*r["normalized_regret"]) for r in group]
            ),
        }
    return {
        "timing_sessions": len(timing),
        "timing_boundary": (
            "original complete fresh/selective workflows, descriptive on one host"
        ),
        "phase_decomposition": "unavailable for the old complete workflow boundary",
        "timing_cells": timing_cells,
        "slower_reuse_sessions": [r for r in timing if r["reuse_fresh_ratio"] > 1],
        "old_unauthorized_assessable_rows": len(consequences_rows),
        "regret_by_fault_and_method": regrets,
    }
