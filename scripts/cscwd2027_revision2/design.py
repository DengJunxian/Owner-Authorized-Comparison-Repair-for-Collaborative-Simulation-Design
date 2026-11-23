"""Fixed post-registration study grid; distinct from original v1 records."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any

from policytwin.coordination.plan import CoordinationExecutionPlan, RunSpec
from scripts.cscwd2027_revision2.mutators import MUTANTS

CONFIG: dict[str, Any] = {
    "label": "post-registration robustness and common-mode diagnostic",
    "seed_root": "cscwd2027-revision2-20261025-v1",
    "replicates": 8,
    "extension_cells": [[6, 16], [24, 8], [48, 4]],
    "branches": ["cancellation", "agent_exit", "latency_reorder"],
    "branch_epoch": "floor(horizon/3)",
    "extension_plans": [
        [1, "round_robin", False],
        [4, "round_robin", True],
        [4, "contiguous", True],
    ],
    "common_mode_agents": 6,
    "common_mode_epochs": 4,
    "common_mode_mutants": list(MUTANTS),
    "common_mode_execution": (
        "in-process, same defect active during execution and replay"
    ),
    "metrics": [
        "exact_draw_agreement",
        "exact_scientific_trace_agreement",
        "paired_contrast_error",
        "first_divergence_epoch",
        "actual_full_release",
        "actual_witnessed_release",
        "witness_finding_codes",
        "common_mode_trigger_calls",
        "clean_false_positive",
    ],
    "threshold": "exact equality, no fitted tolerance",
    "failure_policy": (
        "retain failed/incomplete/nontriggered attempts; no replacement seeds"
    ),
    "analysis": (
        "finite counts by cell/branch/operator; no population fault CI or selection"
    ),
    "expected_claims": [
        "tested worker-partition invariance extends across both static mappings",
        "coordination conformance and replay address non-equivalent obligations",
        "same-implementation replay may accept a shared defect",
        "specification witnesses can expose declared shared-defect instances",
        "partial witnesses do not establish independent simulation or TCB validity",
    ],
}


def seed(study: str, agents: int, epochs: int, branch: str, replicate: int) -> str:
    """Keep compared plans and mutants on the same scientific seed."""
    value = [CONFIG["seed_root"], study, agents, epochs, branch, replicate]
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def schedule() -> list[dict[str, Any]]:
    """List every attempted pair, with references before their comparisons."""
    rows = []
    for agents, epochs in CONFIG["extension_cells"]:
        for branch in CONFIG["branches"]:
            for replicate in range(CONFIG["replicates"]):
                stem = f"extension-a{agents}-h{epochs}-{branch}-r{replicate:02d}"
                spec = RunSpec(
                    "cscwd-r2-extension",
                    seed("extension", agents, epochs, branch, replicate),
                    replicate,
                    agents,
                    branch,
                    epochs,
                    epochs // 3,
                )
                for workers, assignment, processes in CONFIG["extension_plans"]:
                    plan = CoordinationExecutionPlan(workers, assignment)
                    identifier = stem + (
                        "-reference" if not processes else f"-w{workers}-{assignment}"
                    )
                    rows.append(
                        {
                            "id": identifier,
                            "study": "extension",
                            "spec": asdict(spec),
                            "plan": asdict(plan),
                            "processes": processes,
                            "mutant": None,
                            "reference_id": None
                            if not processes
                            else stem + "-reference",
                        }
                    )
    for branch in CONFIG["branches"]:
        for replicate in range(CONFIG["replicates"]):
            agents, epochs = CONFIG["common_mode_agents"], CONFIG["common_mode_epochs"]
            spec = RunSpec(
                "cscwd-r2-common-mode",
                seed("common-mode", agents, epochs, branch, replicate),
                replicate,
                agents,
                branch,
                epochs,
                1,
            )
            stem = f"common-{branch}-r{replicate:02d}"
            for mutant in (None, *MUTANTS):
                rows.append(
                    {
                        "id": stem + ("-clean" if mutant is None else "-" + mutant),
                        "study": "common_mode",
                        "spec": asdict(spec),
                        "plan": asdict(CoordinationExecutionPlan(1)),
                        "processes": False,
                        "mutant": mutant,
                        "reference_id": None if mutant is None else stem + "-clean",
                    }
                )
    return rows
