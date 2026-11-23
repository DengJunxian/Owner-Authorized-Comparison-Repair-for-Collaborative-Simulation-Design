"""Real nested target domains and predeclared requirement-driven revisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Self

from policytwin.coordination.plan import BRANCHES, RunSpec
from policytwin.coordination.schema import commitment
from policytwin.cscwd_mechanism.coupling import CoupledRecipe
from policytwin.design_contract.runtime import SimulationRecipe
from policytwin.typed_revision.metrics import MetricSpec

NAMESPACE = "revision-workload-20261031-v1"
FIELDS = ("epoch", "coupling", "quantity", "first_epoch", "workers", "allowance")
ROLES = ("designer", "evaluator", "producer")
METHODS = ("exhaustive", "context_local", "persistent", "generic_memo", "complete_sql")
PHYSICAL_METHODS = ("exhaustive", "persistent", "complete_sql")
DOMAINS: dict[int, dict[str, list[Any]]] = {
    32: dict(
        zip(
            FIELDS,
            (
                [1, 2],
                ["shared", "world_separated"],
                ["notional", "volume"],
                [0, 4],
                [1, 2],
                [12],
            ),
            strict=True,
        )
    ),
    256: dict(
        zip(
            FIELDS,
            (
                [1, 2, 3, 4],
                ["shared", "world_separated"],
                ["notional", "volume"],
                [0, 2, 4, 6],
                [1, 2],
                [6, 12],
            ),
            strict=True,
        )
    ),
    1024: dict(
        zip(
            FIELDS,
            (
                [1, 2, 3, 4],
                ["shared", "world_separated"],
                ["notional", "volume"],
                list(range(8)),
                [1, 2],
                [0, 3, 6, 12],
            ),
            strict=True,
        )
    ),
}
STUDY: dict[str, Any] = {
    "namespace": NAMESPACE,
    "label": "post-registration staged revision workload",
    "agents": [12, 48],
    "regimes": ["deep", "responsive"],
    "epochs": 8,
    "replicates": 4,
    "domains": {str(size): domain for size, domain in DOMAINS.items()},
    "timing_repeats": 5,
    "methods": METHODS,
    "physical_methods": PHYSICAL_METHODS,
    "catalog_selection": (
        "two or three lowest identity hashes per branch, never outcomes"
    ),
    "physical_cases": (
        "domain256: A stage3, B stage2, C stage2, control stage0 in each block"
    ),
    "rank": (
        "number of changed requested fields, new pairs, reevaluations, target SHA256"
    ),
    "allowance_semantics": (
        "target allowance caps new paired executions, unchanged from earlier study"
    ),
    "operation_cap": (
        "producer separately requires new pairs + reevaluations <= 6 * workers"
    ),
    "warmup": "none; record cold dependency compilation and all warm routes",
    "dependency_cache": (
        "private per method/repeat/block, retained across nested domains "
        "in order32,256,1024"
    ),
    "semantic_memo": (
        "empty per method/repeat/episode/domain, retained through its declared stages"
    ),
    "method_order": "five cyclic shifts cover every position once for each context",
    "timing_scope": (
        "validation, dependencies, typed work, ranking, search, "
        "fresh exact votes, serialization"
    ),
    "thresholds": "exact selection/work/scientific/draw/metric/mean/top1/top2 equality",
    "analysis": (
        "all contexts, stratified by domain/block/episode/stage; "
        "medians and ranges/IQR, no timing inference"
    ),
    "claims": [
        "complete dependency projections permit negative reuse "
        "across irrelevant context changes",
        "changed work or policy dependencies trigger reconsideration",
        "exact current approval remains distinct from reusable semantic results",
        "generic annotated memoization and complete SQL "
        "can realize the same obligations",
        "real total planner cost may exceed lean exhaustive "
        "despite fewer predicate calls",
    ],
    "failure_policy": (
        "retain every failure, no agreement and measurement; no retries or replacement"
    ),
}


@dataclass(frozen=True, slots=True)
class Target:
    epoch: int
    coupling: str
    quantity: str
    first_epoch: int
    workers: int
    allowance: int

    def __post_init__(self) -> None:
        for field, value in asdict(self).items():
            if not any(
                type(value) is type(v) and value == v for v in DOMAINS[1024][field]
            ):
                raise ValueError("invalid target " + field)

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> Self:
        if set(value) != set(FIELDS):
            raise ValueError("target fields differ")
        return cls(**value)

    @property
    def identity(self) -> str:
        return commitment(self.to_mapping())

    @property
    def metric(self) -> MetricSpec:
        return MetricSpec(self.quantity, self.first_epoch)


INITIAL = Target(1, "shared", "notional", 0, 2, 12)


def targets(size: int) -> list[Target]:
    return [Target(*values) for values in product(*(DOMAINS[size][f] for f in FIELDS))]


def recipes(
    session: dict[str, Any], target: Target
) -> dict[tuple[str, int], CoupledRecipe]:
    return {
        (branch, replicate): CoupledRecipe(
            SimulationRecipe(
                RunSpec(
                    "cscwd-revision-" + session["regime"],
                    commitment([NAMESPACE, session["id"], replicate]),
                    replicate,
                    session["agents"],
                    branch,
                    8,
                    target.epoch if branch == session["target"] else 1,
                ),
                session["liquidity"],
            ),
            target.coupling,
        )
        for branch in BRANCHES
        for replicate in range(4)
    }


def slots(session: dict[str, Any], count_per_branch: int) -> list[tuple[str, int]]:
    return [
        (branch, rep)
        for branch in BRANCHES
        for rep in sorted(
            range(4), key=lambda r: commitment([NAMESPACE, session["id"], branch, r])
        )[:count_per_branch]
    ]


def field(name: str) -> dict[str, Any]:
    return {"field": name}


def compare(op: str, left: Any, right: Any) -> dict[str, Any]:
    return {"op": op, "left": left, "right": right}


def all_of(*expressions: dict[str, Any]) -> dict[str, Any]:
    return {"op": "and", "args": list(expressions)}


def policies(episode: str, stage: int) -> list[dict[str, Any]]:
    late = episode == "A" or (episode == "B" and stage >= 1)
    max_new = 3 if episode == "C" and stage >= 1 else 12 if episode == "control" else 6
    max_reevaluation = 0 if episode == "B" else 12
    expressions = {
        "designer": all_of(
            compare("le", 2 if episode == "A" else 1, field("epoch")),
            compare("le", field("epoch"), 4 if episode in {"A", "control"} else 2),
        ),
        "evaluator": all_of(
            compare("eq", field("coupling"), "shared"),
            compare("eq", field("quantity"), "volume" if late else "notional"),
            compare("le", field("epoch"), field("first_epoch"))
            if late
            else compare("eq", field("first_epoch"), 0),
        ),
        "producer": all_of(
            compare("le", field("new_pairs"), max_new),
            compare("le", field("reevaluations"), max_reevaluation),
            compare(
                "le",
                {"add": [field("new_pairs"), field("reevaluations")]},
                {"mul": [6, field("workers")]},
            ),
        ),
    }
    result = []
    for owner in ROLES:
        version = (
            2
            if (owner == "evaluator" and episode == "B" and stage >= 1)
            or (owner == "producer" and episode == "C" and stage >= 1)
            else 1
        )
        body = {
            "schema": "revision-role-policy.v1",
            "owner": owner,
            "version": f"{episode}.{owner}.{version}",
            "expression": expressions[owner],
        }
        result.append({**body, "policy_sha256": commitment(body)})
    return result


def request(session: dict[str, Any], episode: str, stage: int) -> dict[str, Any]:
    proposed = INITIAL.to_mapping()
    if episode == "A":
        proposed.update(epoch=2, quantity="volume", first_epoch=4)
    if episode == "B" and stage == 3:
        proposed.update(quantity="volume", first_epoch=4)
    if episode == "C" and stage == 3:
        proposed.update(workers=1)
    body = {
        "schema": "revision-request.v1",
        "session": session,
        "episode": episode,
        "proposed": proposed,
        "hard": [],
    }
    return {**body, "request_sha256": commitment(body)}


def schedule() -> list[dict[str, Any]]:
    blocks = []
    for ai, agents in enumerate(STUDY["agents"]):
        for ri, regime in enumerate(STUDY["regimes"]):
            session = {
                "id": f"{NAMESPACE}-a{agents}-{regime}",
                "agents": agents,
                "regime": regime,
                "liquidity": 5000 if regime == "deep" else 6 * agents,
                "target": BRANCHES[(2 * ai + ri) % 3],
            }
            episodes = []
            for episode in ("A", "B", "C", "control"):
                stages = []
                for stage in range(1 if episode == "control" else 4):
                    stages.append(
                        {
                            "stage": stage,
                            "request": request(session, episode, stage),
                            "policies": policies(episode, stage),
                            "physical": stage
                            == {"A": 3, "B": 2, "C": 2, "control": 0}[episode],
                            "catalog": {
                                "base_per_branch": 3
                                if episode == "C" and stage >= 2
                                else 2,
                                "unrelated_volume0": episode == "A" and stage >= 1,
                                "revised_epoch2": episode == "A" and stage >= 2,
                                "revised_volume4": episode == "A" and stage >= 3,
                                "base_volume4": episode == "B" and stage >= 2,
                            },
                        }
                    )
                episodes.append({"episode": episode, "stages": stages})
            blocks.append({"session": session, "episodes": episodes})
    return blocks
