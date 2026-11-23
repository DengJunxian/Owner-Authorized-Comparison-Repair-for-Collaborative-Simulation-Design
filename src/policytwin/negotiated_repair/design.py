"""Pre-outcome design for owner-authorized comparison repair."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Any

from policytwin.coordination.plan import BRANCHES, CoordinationExecutionPlan, RunSpec
from policytwin.coordination.schema import commitment
from policytwin.cscwd_mechanism.coupling import CoupledRecipe
from policytwin.design_contract.runtime import SimulationRecipe
from policytwin.typed_revision.contract import trajectory_key
from policytwin.typed_revision.metrics import MetricSpec, evaluation_key

ROLES = ("designer", "evaluator", "producer")
OWNERS = {
    "epoch": "designer",
    "coupling": "evaluator",
    "metric": "evaluator",
    "workers": "producer",
    "allowance": "producer",
}
TICKETS = (
    "candidate_conflict",
    "coupling_conflict",
    "plan_conflict",
    "nonnegotiable",
    "partial_control",
    "complete_control",
)
METHODS = ("fixed_dag", "serial", "frontier", "csp_dag")
STUDY: dict[str, Any] = {
    "label": "post-registration owner-authorized comparison repair v1",
    "seed_root": "cscwd-negotiated-repair-20261027-v1",
    "agents": [6, 24],
    "regimes": ["deep", "responsive"],
    "sessions_per_cell": 1,
    "replicates": 4,
    "epochs": 4,
    "targets": {
        "epoch": [1, 2],
        "coupling": ["shared", "world_separated"],
        "metric": ["notional", "late_volume"],
        "workers": [1, 2],
        "allowance": [6, 12],
    },
    "tickets": list(TICKETS),
    "methods": list(METHODS),
    "partial_store": 6,
    "complete_store": 12,
    "availability": (
        "two lowest hash(session, candidate, replicate, catalog-v1) per candidate; "
        "complete control has all four"
    ),
    "frontier_policy": (
        "minimum soft-clause concessions; Pareto new pairs and trace-only "
        "reevaluations; lexicographic work and target identity"
    ),
    "serial_relaxation_order": ["allowance", "workers", "coupling", "epoch", "metric"],
    "consent": (
        "three cooperating software roles approve exact request, target, "
        "waived clauses and work order before acquisition"
    ),
    "authority_mutations": ["missing_owner", "stale_target", "altered_waiver"],
    "metrics": [
        "actual_pairs",
        "trace_reevaluations",
        "evaluation_reuse",
        "concessions",
        "concession_owners",
        "feasible_panels",
        "abstention",
        "fresh_metric_trace_draw_decision_equality",
        "core_minimality",
        "sql_planner_agreement",
        "approval_rejection",
        "omit_consent_invalid_release",
    ],
    "decision": "maximize exact mean paired metric; lexicographic ties",
    "threshold": (
        "exact integer/rational/trace equality; allowances 6/12 from panel size"
    ),
    "analysis": "descriptive paired work counts; four scientific session blocks",
    "failure_policy": (
        "retain every failure/incompletion; no retries or replacement seeds"
    ),
    "expected_claims": [
        "conflicts precede acquisition and identify required owner concessions",
        "agreed counteroffers bind exact new scientific identity before execution",
        "evidence-work selection differs from fixed-target caching/rebuilding",
        "fully annotated generic CSP and DAG can implement the same policy",
        "finite frontier minimizes work among minimum-concession alternatives",
        "same numerical evidence does not authorize a missing or stale approval",
    ],
}


@dataclass(frozen=True, slots=True)
class Target:
    epoch: int
    coupling: str
    metric: str
    workers: int
    allowance: int

    def __post_init__(self) -> None:
        if any(
            getattr(self, k) not in values for k, values in STUDY["targets"].items()
        ):
            raise ValueError("target is outside the declared domain")

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> Target:
        if set(value) != set(cls.__dataclass_fields__):
            raise ValueError("target fields differ")
        return cls(**value)

    @property
    def identity(self) -> str:
        return commitment(self.to_mapping())


def targets() -> list[Target]:
    keys = list(STUDY["targets"])
    return sorted(
        (
            Target(**dict(zip(keys, values, strict=True)))
            for values in product(*(STUDY["targets"][k] for k in keys))
        ),
        key=lambda t: t.identity,
    )


def sessions() -> list[dict[str, Any]]:
    return [
        {
            "id": f"negotiated-a{agents}-{regime}",
            "index": 2 * ai + ri,
            "agents": agents,
            "regime": regime,
            "liquidity": 5000 if regime == "deep" else 6 * agents,
            "target": BRANCHES[(2 * ai + ri) % len(BRANCHES)],
        }
        for ai, agents in enumerate(STUDY["agents"])
        for ri, regime in enumerate(STUDY["regimes"])
    ]


def metric(target: Target) -> MetricSpec:
    return (
        MetricSpec("notional")
        if target.metric == "notional"
        else MetricSpec("volume", 2)
    )


def recipes(
    session: dict[str, Any], target: Target
) -> dict[tuple[str, int], CoupledRecipe]:
    result: dict[tuple[str, int], CoupledRecipe] = {}
    for candidate in BRANCHES:
        for replicate in range(STUDY["replicates"]):
            spec = RunSpec(
                "cscwd-negotiated-" + session["regime"],
                commitment([STUDY["seed_root"], session["id"], replicate]),
                replicate,
                session["agents"],
                candidate,
                STUDY["epochs"],
                target.epoch if candidate == session["target"] else 1,
            )
            result[candidate, replicate] = CoupledRecipe(
                SimulationRecipe(spec, session["liquidity"]), target.coupling
            )
    return result


def contract(session: dict[str, Any], target: Target) -> dict[str, Any]:
    slots = []
    for (candidate, replicate), recipe in recipes(session, target).items():
        key = trajectory_key(recipe)
        slots.append(
            {
                "candidate": candidate,
                "replicate": replicate,
                "trajectory_key": key,
                "evaluation_key": evaluation_key(key, metric(target)),
                "coupled_recipe": recipe.to_mapping(),
            }
        )
    body = {
        "schema": "negotiated-comparison-contract.v1",
        "session": session,
        "target": target.to_mapping(),
        "metric": metric(target).to_mapping(),
        "allowed_plans": [CoordinationExecutionPlan(target.workers).sha256],
        "slots": slots,
        "report_rule": STUDY["decision"],
    }
    return {**body, "contract_sha256": commitment(body)}


def ticket(session: dict[str, Any], kind: str) -> dict[str, Any]:
    if kind not in TICKETS:
        raise ValueError("unknown ticket")
    proposed = Target(1, "shared", "late_volume", 2, 6).to_mapping()
    soft: set[str] = set()
    if kind == "candidate_conflict":
        proposed["epoch"] = 2
        soft = {"epoch", "allowance"}
    elif kind == "coupling_conflict":
        proposed["coupling"] = "world_separated"
        soft = {"coupling", "allowance"}
    elif kind == "plan_conflict":
        proposed["workers"] = 1
        soft = {"workers", "allowance"}
    elif kind == "nonnegotiable":
        proposed["coupling"] = "world_separated"
    elif kind == "complete_control":
        proposed["metric"] = "notional"
    clauses = [
        {
            "id": OWNERS[k] + "." + k,
            "owner": OWNERS[k],
            "field": k,
            "value": v,
            "hard": k not in soft,
        }
        for k, v in proposed.items()
    ]
    body = {
        "schema": "role-owned-revision-request.v1",
        "session": session,
        "kind": kind,
        "proposed": proposed,
        "clauses": clauses,
        "availability": 12 if kind == "complete_control" else 6,
    }
    return {**body, "request_sha256": commitment(body)}


def available(session: dict[str, Any], count: int) -> list[tuple[str, int]]:
    if count not in (6, 12):
        raise ValueError("unsupported catalog size")
    return [
        slot
        for candidate in BRANCHES
        for slot in sorted(
            ((candidate, r) for r in range(STUDY["replicates"])),
            key=lambda s: commitment([session["id"], *s, "catalog-v1"]),
        )[: count // len(BRANCHES)]
    ]


def schedule() -> list[dict[str, Any]]:
    return [
        {"session": s, "tickets": [ticket(s, k) for k in TICKETS]} for s in sessions()
    ]
