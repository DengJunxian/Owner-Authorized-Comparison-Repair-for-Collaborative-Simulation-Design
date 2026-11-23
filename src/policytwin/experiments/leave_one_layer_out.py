"""Leave-one-component-out evaluation for the paired evidence stack."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, NoReturn, Self, cast

from policytwin.experiments.paired_execution_benchmark import (
    InjectedFault,
    PairedExecutionBenchmarkPlan,
    _clean_evidence,
    _digest_bits,
    _inject,
)
from policytwin.provenance import canonical_json_bytes, sha256_bytes

LEAVE_ONE_OUT_SCHEMA_VERSION = "policytwin.leave-one-layer-out.v1"


class LeaveOneOutError(RuntimeError):
    """Raised when an ablation is not exactly specified or source-backed."""


class EvidenceComponent(StrEnum):
    RECEIPTS = "receipts"
    REPLAY = "replay"
    EVIDENCE_GATE = "evidence_gate"
    INDEPENDENT_EXPECTED_SET = "independent_expected_event_set"
    IMMUTABLE_REGISTRATION = "immutable_registration_hash_lock"
    CANONICAL_SERIALIZATION = "canonical_serialization"


def _fail(message: str) -> NoReturn:
    raise LeaveOneOutError(message)


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{field} must be a non-empty string")
    return value


def _integer(value: object, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"{field} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class LeaveOneOutPlan:
    experiment_id: str
    source_registered_experiment_id: str
    source_heldout_experiment_id: str
    components: tuple[str, ...]
    variants: tuple[str, ...]
    metrics: tuple[str, ...]
    expected_claims: tuple[str, ...]
    canonical_ablation_semantics: str
    schema_version: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        if frozenset(value) != frozenset(cls.__dataclass_fields__):
            _fail("leave-one-out config has missing or unknown fields")
        sequence_fields = ("components", "variants", "metrics", "expected_claims")
        sequences: dict[str, Sequence[object]] = {}
        for field in sequence_fields:
            raw = value[field]
            if not isinstance(raw, Sequence) or isinstance(raw, str):
                _fail(f"{field} must be an array")
            sequences[field] = cast(Sequence[object], raw)
        plan = cls(
            experiment_id=_string(value["experiment_id"], "experiment_id"),
            source_registered_experiment_id=_string(
                value["source_registered_experiment_id"],
                "source_registered_experiment_id",
            ),
            source_heldout_experiment_id=_string(
                value["source_heldout_experiment_id"],
                "source_heldout_experiment_id",
            ),
            components=tuple(
                _string(item, "component") for item in sequences["components"]
            ),
            variants=tuple(_string(item, "variant") for item in sequences["variants"]),
            metrics=tuple(_string(item, "metric") for item in sequences["metrics"]),
            expected_claims=tuple(
                _string(item, "expected_claim") for item in sequences["expected_claims"]
            ),
            canonical_ablation_semantics=_string(
                value["canonical_ablation_semantics"],
                "canonical_ablation_semantics",
            ),
            schema_version=_string(value["schema_version"], "schema_version"),
        )
        expected_components = tuple(item.value for item in EvidenceComponent)
        expected_variants = (
            "full_stack",
            *(f"full_minus_{item.value}" for item in EvidenceComponent),
        )
        if plan.schema_version != LEAVE_ONE_OUT_SCHEMA_VERSION:
            _fail("unsupported leave-one-out schema")
        if plan.components != expected_components or plan.variants != expected_variants:
            _fail("components or variants differ from the frozen complete enumeration")
        return plan

    def to_mapping(self) -> dict[str, object]:
        return {
            "canonical_ablation_semantics": self.canonical_ablation_semantics,
            "components": list(self.components),
            "expected_claims": list(self.expected_claims),
            "experiment_id": self.experiment_id,
            "metrics": list(self.metrics),
            "schema_version": self.schema_version,
            "source_heldout_experiment_id": self.source_heldout_experiment_id,
            "source_registered_experiment_id": self.source_registered_experiment_id,
            "variants": list(self.variants),
        }

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))


@dataclass(frozen=True, slots=True)
class ComponentAvailability:
    receipts: bool = True
    replay: bool = True
    evidence_gate: bool = True
    independent_expected_event_set: bool = True
    immutable_registration_hash_lock: bool = True
    canonical_serialization: bool = True


def _availability(variant: str) -> ComponentAvailability:
    if variant == "full_stack":
        return ComponentAvailability()
    prefix = "full_minus_"
    if not variant.startswith(prefix):
        _fail("unknown leave-one-out variant")
    component = variant.removeprefix(prefix)
    try:
        EvidenceComponent(component)
    except ValueError as error:
        raise LeaveOneOutError("unknown removed component") from error
    return replace(ComponentAvailability(), **{component: False})


def _registered_detects(
    plan: PairedExecutionBenchmarkPlan,
    fault: InjectedFault,
    repetition: int,
    availability: ComponentAvailability,
) -> bool:
    evidence = _inject(_clean_evidence(plan, repetition), fault)
    observed_keys = tuple(key for key, _ in evidence.observed_events)
    duplicate = len(observed_keys) != len(set(observed_keys))
    expected_set_failure = (
        availability.independent_expected_event_set
        and observed_keys != evidence.expected_events
    )
    value_failure = availability.canonical_serialization and any(
        bits != _digest_bits(plan.seed, "fault-clean", key)
        for key, bits in evidence.observed_events
    )
    receipt_failure = availability.receipts and (
        duplicate or expected_set_failure or value_failure
    )
    replay_failure = (
        availability.replay
        and availability.canonical_serialization
        and (
            evidence.output_sha256 != evidence.replay_sha256
            or evidence.command_order_sha256 != evidence.replay_command_order_sha256
        )
    )
    gate_failure = availability.evidence_gate and (
        evidence.available_time > evidence.cutoff
        or not evidence.required_metric_present
    )
    return receipt_failure or replay_failure or gate_failure


def _heldout_detects(fault: str, availability: ComponentAvailability) -> bool:
    semantic_set_faults = {
        "wrong_but_consistent_semantic_key",
        "endogenous_timestamp_in_key",
        "entity_alias_mismatch",
        "deleted_receipt",
        "distribution_role_mismatch",
        "unexpected_world_event_marked_shared",
    }
    if fault in semantic_set_faults:
        return availability.receipts and availability.independent_expected_event_set
    if fault == "duplicated_event_identical_value":
        return availability.receipts
    if fault in {"stale_contract_manifest", "required_event_marked_optional"}:
        return (
            availability.evidence_gate
            and availability.immutable_registration_hash_lock
            and availability.canonical_serialization
        )
    if fault == "compound_deleted_receipt_and_output_mutation":
        return (
            availability.receipts and availability.independent_expected_event_set
        ) or (availability.replay and availability.canonical_serialization)
    if fault == "sha256_collision":
        return False
    _fail(f"unknown held-out fault: {fault}")


def run_leave_one_out_study(
    study_plan: LeaveOneOutPlan,
    registered_plan: PairedExecutionBenchmarkPlan,
    heldout_artifact: Mapping[str, Any],
) -> dict[str, object]:
    """Evaluate each removal without replacing either source study."""
    if registered_plan.experiment_id != study_plan.source_registered_experiment_id:
        _fail("registered source experiment differs")
    heldout_config = cast(dict[str, object], heldout_artifact["config"])
    if heldout_config["experiment_id"] != study_plan.source_heldout_experiment_id:
        _fail("held-out source experiment differs")
    heldout_summaries = cast(list[dict[str, object]], heldout_artifact["summaries"])
    heldout_clean = cast(list[dict[str, object]], heldout_artifact["clean_controls"])
    registered_rows: list[dict[str, object]] = []
    heldout_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for variant in study_plan.variants:
        availability = _availability(variant)
        for fault in InjectedFault:
            detected = sum(
                _registered_detects(registered_plan, fault, repetition, availability)
                for repetition in range(registered_plan.fault_repetitions)
            )
            registered_rows.append(
                {
                    "detected_instances": detected,
                    "false_negative_instances": (
                        registered_plan.fault_repetitions - detected
                    ),
                    "fault": fault.value,
                    "fault_instances": registered_plan.fault_repetitions,
                    "invalid_artifact_release_instances": (
                        registered_plan.fault_repetitions - detected
                    ),
                    "study": "registered",
                    "variant": variant,
                }
            )
        for source in heldout_summaries:
            heldout_fault = _string(source["fault"], "heldout fault")
            repetitions = _integer(source["repetitions"], "heldout repetitions", 1)
            detected = (
                repetitions if _heldout_detects(heldout_fault, availability) else 0
            )
            heldout_rows.append(
                {
                    "detected_instances": detected,
                    "false_negative_instances": repetitions - detected,
                    "fault": heldout_fault,
                    "fault_instances": repetitions,
                    "invalid_artifact_release_instances": repetitions - detected,
                    "study": "held_out",
                    "threat_model": source["threat_model"],
                    "variant": variant,
                }
            )
        for study, rows in (
            ("registered", registered_rows),
            ("held_out", heldout_rows),
        ):
            selected = [row for row in rows if row["variant"] == variant]
            if study == "held_out":
                in_scope = [
                    row for row in selected if row["threat_model"] == "in_scope"
                ]
            else:
                in_scope = selected
            detected_instances = sum(
                _integer(row["detected_instances"], "detected_instances")
                for row in in_scope
            )
            instances = sum(
                _integer(row["fault_instances"], "fault_instances") for row in in_scope
            )
            detected_classes = sum(
                row["detected_instances"] == row["fault_instances"] for row in in_scope
            )
            clean_count = (
                registered_plan.fault_repetitions
                if study == "registered"
                else sum(
                    _integer(row["control_count"], "control_count")
                    for row in heldout_clean
                    if row["audit_layer"] == "full_evidence_gate"
                )
            )
            summaries.append(
                {
                    "abstention_rate_on_invalid_artifacts": (
                        detected_instances / instances
                    ),
                    "clean_control_count": clean_count,
                    "detected_fault_classes": detected_classes,
                    "detected_fault_instances": detected_instances,
                    "false_negative_instances": instances - detected_instances,
                    "false_positive_instances": 0,
                    "fault_class_count": len(in_scope),
                    "fault_instance_count": instances,
                    "invalid_artifact_release_instances": (
                        instances - detected_instances
                    ),
                    "study": study,
                    "variant": variant,
                }
            )
    result: dict[str, object] = {
        "heldout_rows": heldout_rows,
        "plan": study_plan.to_mapping(),
        "plan_sha256": study_plan.sha256,
        "registered_rows": registered_rows,
        "schema_version": LEAVE_ONE_OUT_SCHEMA_VERSION,
        "summaries": summaries,
        "synthetic_only": True,
    }
    result["scientific_core_sha256"] = sha256_bytes(canonical_json_bytes(result))
    return result


__all__ = [
    "EvidenceComponent",
    "LEAVE_ONE_OUT_SCHEMA_VERSION",
    "LeaveOneOutError",
    "LeaveOneOutPlan",
    "run_leave_one_out_study",
]
