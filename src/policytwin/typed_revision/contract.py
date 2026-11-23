"""Outcome-independent three-way acquisition over typed evidence dependencies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Any

from policytwin.coordination.schema import commitment
from policytwin.cscwd_mechanism.coupling import CoupledRecipe
from policytwin.typed_revision.metrics import (
    MetricSpec,
    evaluate,
    evaluation_key,
    implementation_hash,
)


@lru_cache(maxsize=4096)
def trajectory_key(recipe: CoupledRecipe) -> str:
    """Keep dynamics dependencies, excluding only the inherited metric label.

    Recipes are immutable. The executable seal checks all source dependencies
    before and after a job; caching avoids recompiling the same event set for
    repeated availability views, without weakening those file checks.
    """
    return commitment({k: v for k, v in recipe.key_payload().items() if k != "metric"})


@dataclass(frozen=True, slots=True)
class TraceToken:
    object_id: str
    candidate: str
    replicate: int
    trajectory_key: str
    capsule_sha256: str
    plan_sha256: str
    local_pass: bool

    @property
    def slot(self) -> tuple[str, int]:
        return self.candidate, self.replicate


@dataclass(frozen=True, slots=True)
class EvaluationToken:
    object_id: str
    candidate: str
    replicate: int
    trajectory_key: str
    evaluation_key: str
    capsule_sha256: str
    plan_sha256: str
    receipt_sha256: str
    local_pass: bool
    value: int | None

    @property
    def slot(self) -> tuple[str, int]:
        return self.candidate, self.replicate


def derive(
    trace: TraceToken, capsule: dict[str, Any], metric: MetricSpec
) -> tuple[EvaluationToken, dict[str, Any]]:
    """Issue a derivation receipt only for the exact complete verified capsule."""
    if not trace.local_pass:
        raise ValueError("cannot derive from rejected execution")
    raw = {k: v for k, v in capsule.items() if k != "capsule_sha256"}
    if commitment(raw) != trace.capsule_sha256:
        raise ValueError("parent capsule content differs")
    if capsule["capsule_sha256"] != trace.capsule_sha256:
        raise ValueError("parent capsule binding differs")
    if (capsule["spec"]["branch"], capsule["spec"]["replicate"]) != trace.slot:
        raise ValueError("parent coordinates differ")
    if commitment(capsule["plan"]) != trace.plan_sha256:
        raise ValueError("parent plan differs")
    values = evaluate(capsule, metric)
    key = evaluation_key(trace.trajectory_key, metric)
    body = {
        "schema": "checked-metric-derivation.v1",
        "parent_object_id": trace.object_id,
        "candidate": trace.candidate,
        "replicate": trace.replicate,
        "trajectory_key": trace.trajectory_key,
        "evaluation_key": key,
        "capsule_sha256": trace.capsule_sha256,
        "plan_sha256": trace.plan_sha256,
        "metric": metric.to_mapping(),
        "metric_source_sha256": implementation_hash(),
        "values": values,
    }
    receipt = {**body, "receipt_sha256": commitment(body)}
    return EvaluationToken(
        "eval-" + receipt["receipt_sha256"],
        trace.candidate,
        trace.replicate,
        trace.trajectory_key,
        key,
        trace.capsule_sha256,
        trace.plan_sha256,
        receipt["receipt_sha256"],
        True,
        values["contrast"],
    ), receipt


def verify_derivation(
    trace: TraceToken, capsule: dict[str, Any], receipt: dict[str, Any]
) -> EvaluationToken:
    """Recompute the pure rule; never trust a stored numerical assertion alone."""
    metric = MetricSpec.from_mapping(receipt["metric"])
    token, expected = derive(trace, capsule, metric)
    if expected != receipt:
        raise ValueError("derivation binding or numerical value differs")
    return token


def work_order(
    expected: dict[tuple[str, int], tuple[str, str]],
    allowed_plans: tuple[str, ...],
    traces: list[TraceToken],
    evaluations: list[EvaluationToken],
    *,
    mode: str = "typed",
) -> list[dict[str, Any]]:
    """Partition all requested slots without consulting numerical outcomes."""
    if mode not in {"typed", "opaque", "fresh", "minus_evaluation_binding"}:
        raise ValueError("unsupported acquisition policy")
    rows = []
    for slot, (wanted_trace, wanted_eval) in sorted(expected.items()):
        valid_evals = sorted(
            (
                e
                for e in evaluations
                if e.slot == slot
                and e.local_pass
                and e.plan_sha256 in allowed_plans
                and e.trajectory_key == wanted_trace
                and (
                    e.evaluation_key == wanted_eval
                    or mode == "minus_evaluation_binding"
                )
            ),
            key=lambda e: e.object_id,
        )
        valid_traces = sorted(
            (
                t
                for t in traces
                if t.slot == slot
                and t.local_pass
                and t.plan_sha256 in allowed_plans
                and t.trajectory_key == wanted_trace
            ),
            key=lambda t: t.object_id,
        )
        if valid_evals and mode != "fresh":
            action, selected = "reuse", valid_evals[0].object_id
        elif valid_traces and mode in {"typed", "minus_evaluation_binding"}:
            action, selected = "evaluate", valid_traces[0].object_id
        else:
            action, selected = "execute", None
        rows.append(
            {
                "candidate": slot[0],
                "replicate": slot[1],
                "trajectory_key": wanted_trace,
                "evaluation_key": wanted_eval,
                "action": action,
                "object_id": selected,
            }
        )
    return rows


def relational_work_order(
    expected: dict[tuple[str, int], tuple[str, str]],
    allowed_plans: tuple[str, ...],
    traces: list[TraceToken],
    evaluations: list[EvaluationToken],
) -> list[dict[str, Any]]:
    """Equal-obligation relational projection, independently expressed in code."""
    trace_relation: dict[tuple[str, int, str], list[str]] = {}
    eval_relation: dict[tuple[str, int, str, str], list[str]] = {}
    for trace in traces:
        if trace.local_pass and trace.plan_sha256 in allowed_plans:
            trace_relation.setdefault((*trace.slot, trace.trajectory_key), []).append(
                trace.object_id
            )
    for evaluation in evaluations:
        if evaluation.local_pass and evaluation.plan_sha256 in allowed_plans:
            eval_relation.setdefault(
                (
                    *evaluation.slot,
                    evaluation.trajectory_key,
                    evaluation.evaluation_key,
                ),
                [],
            ).append(evaluation.object_id)
    result = []
    for (candidate, replicate), (science, metric) in sorted(expected.items()):
        found_eval = eval_relation.get((candidate, replicate, science, metric), [])
        found_trace = trace_relation.get((candidate, replicate, science), [])
        result.append(
            {
                "candidate": candidate,
                "replicate": replicate,
                "trajectory_key": science,
                "evaluation_key": metric,
                "action": "reuse"
                if found_eval
                else "evaluate"
                if found_trace
                else "execute",
                "object_id": min(found_eval)
                if found_eval
                else min(found_trace)
                if found_trace
                else None,
            }
        )
    return result


def decide(
    contract_sha256: str,
    expected: dict[tuple[str, int], tuple[str, str]],
    allowed_plans: tuple[str, ...],
    members: list[EvaluationToken],
    *,
    omit_evaluation_binding: bool = False,
) -> dict[str, Any]:
    """Authorize exact typed coverage before computing the requested comparison."""
    coordinates = [m.slot for m in members]
    findings = []
    if len(coordinates) != len(expected) or set(coordinates) != set(expected):
        findings.append("exact_slot_coverage")
    for member in members:
        if not member.local_pass or type(member.value) is not int:
            findings.append("local_derivation")
        if member.plan_sha256 not in allowed_plans:
            findings.append("plan_permission")
        wanted = expected.get(member.slot)
        if wanted is None or wanted[0] != member.trajectory_key:
            findings.append("trajectory_binding")
        if not omit_evaluation_binding and (
            wanted is None or wanted[1] != member.evaluation_key
        ):
            findings.append("evaluation_binding")
    if findings:
        return {"authorized": False, "findings": sorted(set(findings)), "report": None}
    candidates = sorted({s[0] for s in expected})
    means = {}
    for candidate in candidates:
        values = [m.value for m in members if m.candidate == candidate]
        means[candidate] = Fraction(
            sum(v for v in values if v is not None), len(values)
        )
    order = sorted(candidates, key=lambda c: (-means[c], c))
    return {
        "authorized": True,
        "findings": [],
        "report": {
            "contract_sha256": contract_sha256,
            "means": {c: [v.numerator, v.denominator] for c, v in means.items()},
            "top1": order[0],
            "top2": sorted(order[:2]),
            "members": [asdict(m) for m in members],
        },
    }
