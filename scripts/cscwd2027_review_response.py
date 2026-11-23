"""Separate, pre-sealed analysis of reviewer-requested comparison obligations."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from dataclasses import replace
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from time import perf_counter_ns
from typing import Any

from policytwin.design_contract.contract import ComparisonContract, Member, decide
from scripts.cscwd2027_design_contract.design import object_id
from scripts.cscwd2027_design_contract.evaluate import get_contract, panel_valid
from scripts.cscwd2027_design_contract.freeze import DEST, check
from scripts.cscwd2027_design_contract.store import load, save

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/cscwd2027/review_response_v1.json"
OUT = ROOT / "results/cscwd2027/review_response_v1"
FREEZE = ROOT / "configs/cscwd2027/review_response_v1_freeze.json"


def sha(path: Path) -> str:
    """Hash an existing input without modifying it."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def without_permission(
    contract: ComparisonContract, panel: list[Member]
) -> dict[str, Any]:
    """Ablate only the plan allowlist; keep every other full-stack obligation."""
    relaxed = replace(
        contract,
        allowed_plans=tuple(
            sorted(
                set(contract.allowed_plans) | {m.evidence.plan_sha256 for m in panel}
            )
        ),
    )
    result = decide(relaxed, panel, "comparison_contract")
    result["ablation"] = "comparison-level plan permission removed"
    if result["numerical"] is not None:
        result["numerical"]["contract_sha256"] = contract.sha256
        result["numerical"]["mode"] = "without_permission"
    return result


def mean_gap(numerical: dict[str, Any]) -> Fraction:
    """Finite-panel top-two gap; no inferential or welfare interpretation."""
    values = sorted((Fraction(*v) for v in numerical["means"].values()), reverse=True)
    return values[0] - values[1]


def freeze() -> None:
    """Seal the complete diagnostic before executing it."""
    check()
    paths = [
        CONFIG,
        Path(__file__),
        ROOT / "tests/test_cscwd_review_response.py",
        DEST / "freeze.json",
        DEST / "contracts.json",
        DEST / "sessions.json",
    ]
    save(
        FREEZE,
        {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "clock": "local, not external timestamp attestation",
            "members": {str(p.relative_to(ROOT)): sha(p) for p in paths},
            "config": json.loads(CONFIG.read_text(encoding="utf-8")),
        },
    )
    print("REVIEW_DIAGNOSTIC_FROZEN", sha(FREEZE))


def verify() -> dict[str, Any]:
    """Verify this diagnostic and the original scientific source seal."""
    check()
    value = json.loads(FREEZE.read_text(encoding="utf-8"))
    for path, expected in value["members"].items():
        if sha(ROOT / path) != expected:
            raise RuntimeError(f"diagnostic input changed: {path}")
    return value


def run() -> None:
    """Evaluate every session without editing any original result object."""
    seal = verify()
    config = seal["config"]
    OUT.mkdir(parents=True, exist_ok=False)
    save(OUT / "started.json", {"freeze_sha256": sha(FREEZE), "config": config})
    try:
        analyze(config)
    except Exception as error:
        save(OUT / "failure.json", {"error": f"{type(error).__name__}: {error}"})
        raise
    verify()
    print("REVIEW_DIAGNOSTIC_COMPLETE", str(OUT.relative_to(ROOT)))


def analyze(config: dict[str, Any]) -> None:
    """Preserve full permission decisions, exact means and batch timer samples."""
    sessions = json.loads((DEST / "sessions.json").read_text(encoding="utf-8"))
    contracts = json.loads((DEST / "contracts.json").read_text(encoding="utf-8"))
    decisions, gaps, initial_panels = [], [], []
    for session in sessions:
        contract = get_contract(session["id"], 1, contracts)
        initial, remapped = [], []
        for slot in contract.slots:
            evidence = load(
                object_id(session, "initial", slot.candidate, slot.replicate)
            )[0]
            initial.append(Member(slot.candidate, slot.replicate, evidence))
            if slot.replicate == 0:
                evidence = load(object_id(session, "remap", slot.candidate, 0))[0]
            remapped.append(Member(slot.candidate, slot.replicate, evidence))
        initial_panels.append((contract, initial))
        restricted = replace(contract, allowed_plans=contract.allowed_plans[:1])
        for case, authority in (
            ("unpermitted_plan", restricted),
            ("permitted_control", contract),
        ):
            for mode in (
                "comparison_contract",
                "conventional_relational",
                "without_permission",
            ):
                result = (
                    without_permission(authority, remapped)
                    if mode == "without_permission"
                    else decide(authority, remapped, mode)
                )
                decisions.append(
                    {
                        "session": session["id"],
                        "case": case,
                        "mode": mode,
                        "all_local_valid": all(m.evidence.local_pass for m in remapped),
                        "authorized": result["authorized"],
                        "output_panel_valid": panel_valid(
                            authority, result["numerical"]
                        ),
                        "decision": result,
                    }
                )
        for version in (1, 2):
            path = (
                ROOT
                / "results/cscwd2027/design_contract_v1/evaluation/references"
                / session["id"]
                / f"v{version}/numerical_comparison.json"
            )
            numerical = json.loads(path.read_text(encoding="utf-8"))
            gap = mean_gap(numerical)
            gaps.append(
                {
                    "session": session["id"],
                    "agents": session["agents"],
                    "regime": session["regime"],
                    "version": version,
                    "means": numerical["means"],
                    "top_two_gap": [gap.numerator, gap.denominator],
                }
            )
    timings: dict[str, Any] = {}
    for mode in config["timing"]["modes"]:
        elapsed = []
        for batch in range(
            config["timing"]["warmup_batches"] + config["timing"]["batches"]
        ):
            begin = perf_counter_ns()
            for contract, panel in initial_panels:
                decide(contract, panel, mode)
            duration = perf_counter_ns() - begin
            if batch >= config["timing"]["warmup_batches"]:
                elapsed.append(duration)
        samples = [v / len(initial_panels) / 1_000_000 for v in elapsed]
        timings[mode] = {
            "batch_ns": elapsed,
            "panels_per_batch": len(initial_panels),
            "median_ms_per_panel": statistics.median(samples),
            "q1_ms": statistics.quantiles(samples, n=4)[0],
            "q3_ms": statistics.quantiles(samples, n=4)[2],
        }
    total = len(initial_panels[0][0].slots)
    per_candidate = total // len(initial_panels[0][0].candidates)
    summary = {
        "label": config["label"],
        "new_simulation_pairs": 0,
        "sessions": len(sessions),
        "decisions": decisions,
        "gaps": gaps,
        "permission_counts": {
            case: {
                mode: {
                    "panels": len(
                        rows := [
                            r
                            for r in decisions
                            if r["case"] == case and r["mode"] == mode
                        ]
                    ),
                    "released": sum(r["authorized"] for r in rows),
                    "invalid_released": sum(
                        r["authorized"] and not r["output_panel_valid"] for r in rows
                    ),
                }
                for mode in config["timing"]["modes"] + ["without_permission"]
            }
            for case in ("unpermitted_plan", "permitted_control")
        },
        "comparison_timing": timings,
        "revision_footprints": [
            {
                "revision": "one candidate epoch",
                "reuse": total - per_candidate,
                "new": per_candidate,
                "evidence": "measured in the original study",
            },
            {
                "revision": "four to eight replicates",
                "reuse": total,
                "new": total,
                "evidence": "analytical",
            },
            {
                "revision": "add one candidate",
                "reuse": total,
                "new": per_candidate,
                "evidence": "analytical",
            },
            {
                "revision": "metric definition",
                "reuse": 0,
                "new": total,
                "evidence": "analytical",
            },
            {
                "revision": "shared opening state",
                "reuse": 0,
                "new": total,
                "evidence": "analytical",
            },
            {
                "revision": "source-only refactor",
                "reuse": 0,
                "new": total,
                "evidence": "analytical",
            },
        ],
    }
    save(OUT / "summary.json", summary)
    save(
        OUT / "manifest.json",
        {"freeze_sha256": sha(FREEZE), "summary_sha256": sha(OUT / "summary.json")},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "run", "check"))
    action = parser.parse_args().action
    if action == "freeze":
        freeze()
    elif action == "run":
        run()
    else:
        verify()
        print("REVIEW_DIAGNOSTIC_FREEZE_OK")
