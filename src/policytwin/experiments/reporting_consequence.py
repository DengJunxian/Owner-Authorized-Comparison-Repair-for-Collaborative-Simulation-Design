"""Frozen decision/reporting rules applied to registered synthetic evidence."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, NoReturn, Self, cast

from policytwin.experiments.paired_execution_benchmark import (
    AuditLayer,
    BranchScenario,
    CouplingMode,
    PairedExecutionBenchmarkPlan,
    _batch_indices,
)
from policytwin.provenance import canonical_json_bytes, sha256_bytes
from policytwin.ranking_metrics import selection_regret, top_set

REPORTING_CONSEQUENCE_SCHEMA_VERSION = "policytwin.reporting-consequence.v1"
_T_CRITICAL_95_DF31 = 2.039513446


class ReportingConsequenceError(RuntimeError):
    """Raised when reporting rules or registered inputs are incomplete."""


class ReportingDecision(StrEnum):
    RECOMMEND = "recommend_intervention"
    DO_NOT_RECOMMEND = "do_not_recommend"
    INCONCLUSIVE = "inconclusive"
    ABSTAIN = "abstain_high_variance"
    PROHIBITED = "prohibited_incomplete_evidence"


def _fail(message: str) -> NoReturn:
    raise ReportingConsequenceError(message)


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{field} must be a non-empty string")
    return value


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        _fail(f"{field} must be numeric")
    return float(value)


def _integer(value: object, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"{field} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class ReportingConsequencePlan:
    experiment_id: str
    source_experiment_id: str
    effect_recommendation_threshold: float
    effect_variance_abstention_threshold: float
    interval_level: float
    interval_crossing_zero_is_inconclusive: bool
    top_k: int
    top_k_boundary_ties_included: bool
    incomplete_evidence_prohibits_numeric_report: bool
    metrics: tuple[str, ...]
    expected_claims: tuple[str, ...]
    schema_version: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        if frozenset(value) != frozenset(cls.__dataclass_fields__):
            _fail("reporting-consequence config has missing or unknown fields")
        raw_metrics = value["metrics"]
        raw_claims = value["expected_claims"]
        if (
            not isinstance(raw_metrics, Sequence)
            or isinstance(raw_metrics, str)
            or not isinstance(raw_claims, Sequence)
            or isinstance(raw_claims, str)
        ):
            _fail("metrics and expected_claims must be arrays")
        for field in (
            "interval_crossing_zero_is_inconclusive",
            "top_k_boundary_ties_included",
            "incomplete_evidence_prohibits_numeric_report",
        ):
            if not isinstance(value[field], bool):
                _fail(f"{field} must be Boolean")
        plan = cls(
            experiment_id=_string(value["experiment_id"], "experiment_id"),
            source_experiment_id=_string(
                value["source_experiment_id"], "source_experiment_id"
            ),
            effect_recommendation_threshold=_number(
                value["effect_recommendation_threshold"],
                "effect_recommendation_threshold",
            ),
            effect_variance_abstention_threshold=_number(
                value["effect_variance_abstention_threshold"],
                "effect_variance_abstention_threshold",
            ),
            interval_level=_number(value["interval_level"], "interval_level"),
            interval_crossing_zero_is_inconclusive=cast(
                bool, value["interval_crossing_zero_is_inconclusive"]
            ),
            top_k=_integer(value["top_k"], "top_k", 1),
            top_k_boundary_ties_included=cast(
                bool, value["top_k_boundary_ties_included"]
            ),
            incomplete_evidence_prohibits_numeric_report=cast(
                bool, value["incomplete_evidence_prohibits_numeric_report"]
            ),
            metrics=tuple(
                _string(item, "metric") for item in cast(Sequence[object], raw_metrics)
            ),
            expected_claims=tuple(
                _string(item, "expected_claim")
                for item in cast(Sequence[object], raw_claims)
            ),
            schema_version=_string(value["schema_version"], "schema_version"),
        )
        if plan.schema_version != REPORTING_CONSEQUENCE_SCHEMA_VERSION:
            _fail("unsupported reporting-consequence schema")
        if plan.source_experiment_id != "aaai27-paired-execution-v2":
            _fail("reporting study must use the immutable registered experiment")
        if plan.interval_level != 0.95:
            _fail("only the frozen 95% interval rule is implemented")
        if plan.top_k != 2 or not plan.top_k_boundary_ties_included:
            _fail("the frozen top-2 rule must retain boundary ties")
        if plan.effect_variance_abstention_threshold <= 0:
            _fail("variance abstention threshold must be positive")
        return plan

    def to_mapping(self) -> dict[str, object]:
        return {
            "effect_recommendation_threshold": self.effect_recommendation_threshold,
            "effect_variance_abstention_threshold": (
                self.effect_variance_abstention_threshold
            ),
            "expected_claims": list(self.expected_claims),
            "experiment_id": self.experiment_id,
            "incomplete_evidence_prohibits_numeric_report": (
                self.incomplete_evidence_prohibits_numeric_report
            ),
            "interval_crossing_zero_is_inconclusive": (
                self.interval_crossing_zero_is_inconclusive
            ),
            "interval_level": self.interval_level,
            "metrics": list(self.metrics),
            "schema_version": self.schema_version,
            "source_experiment_id": self.source_experiment_id,
            "top_k": self.top_k,
            "top_k_boundary_ties_included": self.top_k_boundary_ties_included,
        }

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))


def reporting_decision(
    plan: ReportingConsequencePlan,
    values: Sequence[float],
    *,
    evidence_complete: bool,
) -> tuple[ReportingDecision, dict[str, float | bool]]:
    """Apply the pre-frozen decision rule with explicit precedence."""
    if not values:
        _fail("reporting decision requires at least one contrast")
    if not evidence_complete and plan.incomplete_evidence_prohibits_numeric_report:
        return ReportingDecision.PROHIBITED, {
            "evidence_complete": False,
            "interval_crosses_zero": False,
            "mean": math.nan,
            "sample_variance": math.nan,
        }
    mean = statistics.fmean(values)
    variance = statistics.variance(values) if len(values) > 1 else 0.0
    standard_error = math.sqrt(variance / len(values))
    interval = (
        mean - _T_CRITICAL_95_DF31 * standard_error,
        mean + _T_CRITICAL_95_DF31 * standard_error,
    )
    crosses_zero = interval[0] <= 0.0 <= interval[1]
    diagnostics: dict[str, float | bool] = {
        "evidence_complete": True,
        "interval_crosses_zero": crosses_zero,
        "interval_high": interval[1],
        "interval_low": interval[0],
        "mean": mean,
        "sample_variance": variance,
    }
    if variance > plan.effect_variance_abstention_threshold:
        return ReportingDecision.ABSTAIN, diagnostics
    if plan.interval_crossing_zero_is_inconclusive and crosses_zero:
        return ReportingDecision.INCONCLUSIVE, diagnostics
    if mean < plan.effect_recommendation_threshold:
        return ReportingDecision.RECOMMEND, diagnostics
    return ReportingDecision.DO_NOT_RECOMMEND, diagnostics


def top_k_set(values: Mapping[str, float], k: int) -> frozenset[str]:
    """Return the top-k set with every boundary tie retained."""
    if k < 1 or len(values) < k:
        _fail("top-k input is smaller than k")
    levels = sorted({float(value) for value in values.values()}, reverse=True)
    boundary = levels[min(k - 1, len(levels) - 1)]
    return frozenset(key for key, value in values.items() if value >= boundary)


def _registered_effects(
    artifact: Mapping[str, Any], plan: PairedExecutionBenchmarkPlan
) -> dict[tuple[str, str], list[float]]:
    rows = cast(list[dict[str, object]], artifact["market_rows"])
    result: dict[tuple[str, str], list[float]] = {}
    for scenario in BranchScenario:
        for mode in CouplingMode:
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["scenario"] == scenario.value
                    and row["coupling_mode"] == mode.value
                ),
                key=lambda row: _integer(row["replicate_id"], "replicate_id"),
            )
            values = [
                _number(row["effect_traded_notional"], "effect_traded_notional")
                for row in selected
            ]
            if len(values) != plan.replicate_count:
                _fail("registered reporting cell is incomplete")
            result[(scenario.value, mode.value)] = values
    return result


def _evidence_consequences(artifact: Mapping[str, Any]) -> dict[str, object]:
    fault = cast(dict[str, Any], artifact["fault_injection"])
    detection_rows = cast(list[dict[str, object]], fault["fault_detection"])
    clean_rows = cast(list[dict[str, object]], fault["clean_false_positives"])
    rows: list[dict[str, object]] = []
    aggregates: list[dict[str, object]] = []
    for layer in AuditLayer:
        selected = [row for row in detection_rows if row["audit_layer"] == layer.value]
        fault_instances = sum(
            _integer(row["repetition_count"], "repetition_count") for row in selected
        )
        detected = sum(
            _integer(row["detected_count"], "detected_count") for row in selected
        )
        clean = next(row for row in clean_rows if row["audit_layer"] == layer.value)
        clean_count = _integer(clean["clean_repetition_count"], "clean_count")
        clean_alarms = _integer(
            clean["clean_false_positive_count"], "clean_false_positive_count"
        )
        for row in selected:
            repetitions = _integer(row["repetition_count"], "repetition_count")
            detections = _integer(row["detected_count"], "detected_count")
            rows.append(
                {
                    "audit_layer": layer.value,
                    "detected_fault_instances": detections,
                    "false_numerical_report_instances": repetitions - detections,
                    "fault": row["fault"],
                    "fault_instances": repetitions,
                    "invalid_artifact_release_instances": repetitions - detections,
                    "missed_abstention_instances": repetitions - detections,
                }
            )
        aggregates.append(
            {
                "audit_layer": layer.value,
                "clean_control_count": clean_count,
                "false_numerical_report_rate": (
                    (fault_instances - detected) / fault_instances
                ),
                "full_gate_prevented_report_count": (
                    detected if layer is AuditLayer.FULL_EVIDENCE_GATE else 0
                ),
                "invalid_artifact_release_rate": (
                    (fault_instances - detected) / fault_instances
                ),
                "missed_abstention_rate": (
                    (fault_instances - detected) / fault_instances
                ),
                "unnecessary_abstention_rate": clean_alarms / clean_count,
            }
        )
    return {"aggregates": aggregates, "rows": rows}


def run_reporting_consequence_study(
    study_plan: ReportingConsequencePlan,
    artifact: Mapping[str, Any],
    registered_plan: PairedExecutionBenchmarkPlan,
) -> dict[str, object]:
    """Separate conformance, numeric, and final reporting consequences."""
    if registered_plan.experiment_id != study_plan.source_experiment_id:
        _fail("registered plan does not match the frozen reporting source")
    effects = _registered_effects(artifact, registered_plan)
    reference_means = {
        scenario.value: statistics.fmean(
            effects[(scenario.value, CouplingMode.EVENT_KEYED.value)]
        )
        for scenario in BranchScenario
    }
    reference_decisions = {
        scenario.value: reporting_decision(
            study_plan,
            effects[(scenario.value, CouplingMode.EVENT_KEYED.value)],
            evidence_complete=True,
        )[0]
        for scenario in BranchScenario
    }
    reference_top1 = top_set(reference_means)
    reference_top2 = top_k_set(reference_means, study_plan.top_k)
    batch_rows: list[dict[str, object]] = []
    for batch_id in range(registered_plan.batch_count):
        indices = _batch_indices(registered_plan, batch_id)
        keyed_samples = {
            scenario.value: [
                effects[(scenario.value, CouplingMode.EVENT_KEYED.value)][index]
                for index in indices
            ]
            for scenario in BranchScenario
        }
        keyed_decisions = {
            scenario: reporting_decision(study_plan, values, evidence_complete=True)[0]
            for scenario, values in keyed_samples.items()
        }
        keyed_means = {
            scenario: statistics.fmean(values)
            for scenario, values in keyed_samples.items()
        }
        for mode in CouplingMode:
            samples = {
                scenario.value: [
                    effects[(scenario.value, mode.value)][index] for index in indices
                ]
                for scenario in BranchScenario
            }
            decisions = {
                scenario: reporting_decision(
                    study_plan, values, evidence_complete=True
                )[0]
                for scenario, values in samples.items()
            }
            means = {
                scenario: statistics.fmean(values)
                for scenario, values in samples.items()
            }
            numerical_errors = {
                scenario: means[scenario] - keyed_means[scenario] for scenario in means
            }
            batch_rows.append(
                {
                    "batch_id": batch_id,
                    "coupling_mode": mode.value,
                    "decision_disagreement_rate_vs_paired_event_keyed_batch": (
                        sum(
                            decisions[scenario] != keyed_decisions[scenario]
                            for scenario in decisions
                        )
                        / len(decisions)
                    ),
                    "decision_level_regret": selection_regret(reference_means, means),
                    "execution_conformance_error": (
                        mode is not CouplingMode.EVENT_KEYED
                        and any(error != 0.0 for error in numerical_errors.values())
                    ),
                    "final_reporting_decision_error_rate": (
                        sum(
                            decisions[scenario] != reference_decisions[scenario]
                            for scenario in decisions
                        )
                        / len(decisions)
                    ),
                    "mean_absolute_numerical_contrast_error": statistics.fmean(
                        abs(error) for error in numerical_errors.values()
                    ),
                    "reporting_decisions": {
                        scenario: decision.value
                        for scenario, decision in decisions.items()
                    },
                    "top1_set": sorted(top_set(means)),
                    "top2_set": sorted(top_k_set(means, study_plan.top_k)),
                    "top2_set_disagreement": (
                        top_k_set(means, study_plan.top_k) != reference_top2
                    ),
                    "wrong_top1_selection": top_set(means) != reference_top1,
                }
            )
    aggregates: list[dict[str, object]] = []
    for mode in CouplingMode:
        selected = [row for row in batch_rows if row["coupling_mode"] == mode.value]
        aggregates.append(
            {
                "coupling_mode": mode.value,
                "decision_disagreement_rate": statistics.fmean(
                    _number(
                        row["decision_disagreement_rate_vs_paired_event_keyed_batch"],
                        "decision_disagreement_rate",
                    )
                    for row in selected
                ),
                "decision_level_regret_mean": statistics.fmean(
                    _number(row["decision_level_regret"], "decision_level_regret")
                    for row in selected
                ),
                "execution_conformance_error_rate": statistics.fmean(
                    cast(bool, row["execution_conformance_error"]) for row in selected
                ),
                "final_reporting_decision_error_rate": statistics.fmean(
                    _number(
                        row["final_reporting_decision_error_rate"],
                        "final_reporting_decision_error_rate",
                    )
                    for row in selected
                ),
                "mean_absolute_numerical_contrast_error": statistics.fmean(
                    _number(
                        row["mean_absolute_numerical_contrast_error"],
                        "mean_absolute_numerical_contrast_error",
                    )
                    for row in selected
                ),
                "top2_set_disagreement_rate": statistics.fmean(
                    cast(bool, row["top2_set_disagreement"]) for row in selected
                ),
                "wrong_top1_selection_rate": statistics.fmean(
                    cast(bool, row["wrong_top1_selection"]) for row in selected
                ),
            }
        )
    result: dict[str, object] = {
        "coupling_aggregates": aggregates,
        "coupling_batch_rows": batch_rows,
        "evidence_stack": _evidence_consequences(artifact),
        "plan": study_plan.to_mapping(),
        "plan_sha256": study_plan.sha256,
        "reference_branch_means": reference_means,
        "reference_decisions": {
            scenario: decision.value
            for scenario, decision in reference_decisions.items()
        },
        "reference_top1_set": sorted(reference_top1),
        "reference_top2_set_with_boundary_ties": sorted(reference_top2),
        "schema_version": REPORTING_CONSEQUENCE_SCHEMA_VERSION,
        "synthetic_only": True,
    }
    result["scientific_core_sha256"] = sha256_bytes(canonical_json_bytes(result))
    return result


__all__ = [
    "REPORTING_CONSEQUENCE_SCHEMA_VERSION",
    "ReportingConsequenceError",
    "ReportingConsequencePlan",
    "ReportingDecision",
    "reporting_decision",
    "run_reporting_consequence_study",
    "top_k_set",
]
