"""Pre-result deterministic grids and seed derivation for CSCWD studies."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from policytwin.coordination.faults import FAULTS, Fault
from policytwin.coordination.plan import BRANCHES, CoordinationExecutionPlan, RunSpec
from policytwin.coordination.schema import commitment

SEED_ROOT = "policytwin-cscwd2027-coordination-v1"
COST_MODES = ("no_evidence", "keys_only", "receipts", "full", "conventional")
CONFIG = {
    "schema": "cscwd2027-study-v1",
    "seed_root": SEED_ROOT,
    "agents": [6, 24],
    "workers": [1, 2, 4],
    "branches": list(BRANCHES),
    "replicates": 32,
    "epochs": 4,
    "branch_epoch": 1,
    "partition": "round_robin",
    "optional_contiguous_partition": "not_run_budget",
    "cost_measured_blocks": 20,
    "cost_warmup_blocks": 3,
    "cost_modes": list(COST_MODES),
    "bootstrap_resamples": 2000,
    "bootstrap_seed": 20271025,
    "comparison": "exact canonical equality; zero paired contrast discrepancy",
    "failure_policy": (
        "retain all attempts; no replacement seeds; resume completed IDs only"
    ),
    "expected_claims": [
        "partition invariance only for the fixed logical schedule",
        "event alignment does not establish coordination conformance",
        "full conventional composition may equal the proposed full obligations",
        "no claim of speedup or unbiased accepted-sample law",
    ],
    "budget_amendment": (
        "64 extra clean in-process pairs for E2/E2B numerical ground truth"
    ),
}


def seed(study: str, agents: int, branch: str, replicate: int) -> str:
    """Derive seeds from canonical tuples; never use Python's randomized hash."""
    return commitment([SEED_ROOT, study, agents, branch, replicate])


def workload(study: str, agents: int, branch: str, replicate: int) -> RunSpec:
    """Construct the four-epoch workload fixed after the separate budget pilot."""
    return RunSpec(
        study, seed(study, agents, branch, replicate), replicate, agents, branch, 4, 1
    )


def entry(
    identifier: str, group: str, spec: RunSpec, workers: int, **extra: Any
) -> dict[str, Any]:
    """One independently addressable attempt; source of all runner parameters."""
    return {
        "id": identifier,
        "group": group,
        "spec": asdict(spec),
        "plan": asdict(CoordinationExecutionPlan(workers)),
        "processes": True,
        "transport": "normal",
        "fault": None,
        "reference_id": None,
        "warmup": False,
        **extra,
    }


def schedule() -> list[dict[str, Any]]:
    """Return the complete execution order before any formal result is observed."""
    rows = []
    for agents in (6, 24):
        for branch in BRANCHES:
            for rep in range(32):
                spec = workload("e1-v1", agents, branch, rep)
                ref = f"e1-a{agents}-{branch}-r{rep:02d}-reference"
                rows.append(entry(ref, "e1_reference", spec, 1, processes=False))
                for workers in (1, 2, 4):
                    rows.append(
                        entry(
                            f"e1-a{agents}-{branch}-r{rep:02d}-w{workers}",
                            "e1",
                            spec,
                            workers,
                            reference_id=ref,
                        )
                    )
    for rep in range(32):
        branch = BRANCHES[rep % 3]
        spec = workload("e2-v1", 24, branch, rep)
        ref = f"e2-r{rep:02d}-reference"
        rows.append(entry(ref, "e2_reference", spec, 4, processes=False))
        index = (7 * rep + 1) % 23  # Never the last agent: F6 has a successor.
        if index == 12:  # F7 mutates an active non-exited task by design.
            index = 13
        for kind in FAULTS:
            fault = Fault(
                kind, ("baseline", "policy")[rep % 2], 1 + rep % 3, spec.agents[index]
            )
            rows.append(
                entry(
                    f"e2-{kind}-r{rep:02d}",
                    "e2",
                    spec,
                    4,
                    reference_id=ref,
                    fault=asdict(fault),
                )
            )
    for rep in range(32):
        spec = workload("e2b-v1", 24, BRANCHES[rep % 3], rep)
        ref = f"e2b-r{rep:02d}-reference"
        rows.append(entry(ref, "e2b_reference", spec, 4, processes=False))
        for transport in ("normal", "delay", "reverse"):
            rows.append(
                entry(
                    f"e2b-{transport}-r{rep:02d}",
                    "e2b",
                    spec,
                    4,
                    reference_id=ref,
                    transport=transport,
                )
            )
    for agents in (6, 24):
        for workers in (1, 2, 4):
            for warmup, count in ((True, 3), (False, 20)):
                phase = "warmup" if warmup else "measured"
                for block in range(count):
                    spec = workload(f"e3-{phase}-v1", agents, "cancellation", block)
                    # Same scenario within a block; deterministic random mode order.
                    ordered = sorted(
                        COST_MODES,
                        key=lambda mode: commitment(
                            [
                                SEED_ROOT,
                                "cost-order",
                                agents,
                                workers,
                                phase,
                                block,
                                mode,
                            ]
                        ),
                    )
                    for position, mode in enumerate(ordered):
                        rows.append(
                            entry(
                                f"e3-a{agents}-w{workers}-{phase}-{block:02d}-{mode}",
                                "e3",
                                spec,
                                workers,
                                warmup=warmup,
                                cost_mode=mode,
                                block=block,
                                order_position=position,
                            )
                        )
    return rows
