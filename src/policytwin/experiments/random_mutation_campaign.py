"""Pre-frozen randomized evidence mutation campaign for revision v4."""

from __future__ import annotations

import hashlib
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import NoReturn, Self, cast

from policytwin.counterfactual.coupling_contract import (
    ConsumptionObservation,
    CouplingContract,
    EventDeclaration,
    EventKeyedBackend,
    Requirement,
    ScenarioSpecification,
    SemanticEventAddress,
    World,
    WorldScope,
    audit_pair,
    generate_contract,
)
from policytwin.provenance import canonical_json_bytes, sha256_bytes
from policytwin.verification.minimal_evidence_verifier import (
    MINIMAL_EVIDENCE_CAPSULE_SCHEMA_VERSION,
    capsule_sha256,
    verify_capsule,
)

RANDOM_MUTATION_CAMPAIGN_SCHEMA_VERSION = "policytwin.random-mutation-campaign.v1"
MUTATION_OPERATORS = (
    "delete_receipt",
    "insert_unexpected_event",
    "duplicate_event",
    "rename_semantic_key",
    "flip_world_scope",
    "mutate_draw_role",
    "mutate_distribution_parameters",
    "reorder_commands",
    "mutate_output",
    "stale_manifest",
    "remove_required_metric",
    "change_cutoff_metadata",
)


class RandomMutationCampaignError(RuntimeError):
    """Raised when a campaign plan or generated mutant is invalid."""


def _fail(message: str) -> NoReturn:
    raise RandomMutationCampaignError(message)


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{field} must be a non-empty string")
    return value


def _integer(value: object, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"{field} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class MutationOperator:
    name: str
    probability_weight: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        if frozenset(value) != {"name", "probability_weight"}:
            _fail("mutation operator has missing or unknown fields")
        return cls(
            name=_string(value["name"], "operator.name"),
            probability_weight=_integer(
                value["probability_weight"], "operator.probability_weight", 1
            ),
        )

    def to_mapping(self) -> dict[str, object]:
        return {"name": self.name, "probability_weight": self.probability_weight}


@dataclass(frozen=True, slots=True)
class RandomMutationPlan:
    experiment_id: str
    seed: str
    trial_count: int
    clean_control_count: int
    fault_count_min: int
    fault_count_max: int
    operators: tuple[MutationOperator, ...]
    agent_ids: tuple[str, ...]
    logical_epochs: tuple[int, ...]
    branches: tuple[str, ...]
    worlds: tuple[str, ...]
    required_metrics: tuple[str, ...]
    expected_claims: tuple[str, ...]
    schema_version: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        if frozenset(value) != frozenset(cls.__dataclass_fields__):
            _fail("mutation campaign config has missing or unknown fields")
        sequences: dict[str, Sequence[object]] = {}
        for field in (
            "operators",
            "agent_ids",
            "logical_epochs",
            "branches",
            "worlds",
            "required_metrics",
            "expected_claims",
        ):
            raw = value[field]
            if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
                _fail(f"{field} must be an array")
            sequences[field] = cast(Sequence[object], raw)
        operators = tuple(
            MutationOperator.from_mapping(cast(Mapping[str, object], item))
            for item in sequences["operators"]
        )
        plan = cls(
            experiment_id=_string(value["experiment_id"], "experiment_id"),
            seed=_string(value["seed"], "seed"),
            trial_count=_integer(value["trial_count"], "trial_count", 1),
            clean_control_count=_integer(
                value["clean_control_count"], "clean_control_count", 1
            ),
            fault_count_min=_integer(value["fault_count_min"], "fault_count_min", 1),
            fault_count_max=_integer(value["fault_count_max"], "fault_count_max", 1),
            operators=operators,
            agent_ids=tuple(
                _string(item, "agent_id") for item in sequences["agent_ids"]
            ),
            logical_epochs=tuple(
                _integer(item, "logical_epoch") for item in sequences["logical_epochs"]
            ),
            branches=tuple(_string(item, "branch") for item in sequences["branches"]),
            worlds=tuple(_string(item, "world") for item in sequences["worlds"]),
            required_metrics=tuple(
                _string(item, "required_metric")
                for item in sequences["required_metrics"]
            ),
            expected_claims=tuple(
                _string(item, "expected_claim") for item in sequences["expected_claims"]
            ),
            schema_version=_string(value["schema_version"], "schema_version"),
        )
        if plan.schema_version != RANDOM_MUTATION_CAMPAIGN_SCHEMA_VERSION:
            _fail("unsupported mutation campaign schema")
        if tuple(item.name for item in plan.operators) != MUTATION_OPERATORS:
            _fail("mutation operators must equal the frozen complete enumeration")
        if plan.fault_count_max > len(plan.operators):
            _fail("fault_count_max exceeds the unique operator count")
        if plan.fault_count_min > plan.fault_count_max:
            _fail("fault count bounds are reversed")
        if plan.worlds != tuple(item.value for item in World):
            _fail("worlds must be baseline then policy")
        if len(set(plan.agent_ids)) != len(plan.agent_ids) or not plan.agent_ids:
            _fail("agent_ids must be non-empty and unique")
        if tuple(sorted(set(plan.logical_epochs))) != plan.logical_epochs:
            _fail("logical_epochs must be strictly increasing")
        if len(set(plan.branches)) != len(plan.branches) or not plan.branches:
            _fail("branches must be non-empty and unique")
        if not plan.required_metrics:
            _fail("required_metrics must not be empty")
        return plan

    def to_mapping(self) -> dict[str, object]:
        return {
            "agent_ids": list(self.agent_ids),
            "branches": list(self.branches),
            "clean_control_count": self.clean_control_count,
            "expected_claims": list(self.expected_claims),
            "experiment_id": self.experiment_id,
            "fault_count_max": self.fault_count_max,
            "fault_count_min": self.fault_count_min,
            "logical_epochs": list(self.logical_epochs),
            "operators": [item.to_mapping() for item in self.operators],
            "required_metrics": list(self.required_metrics),
            "schema_version": self.schema_version,
            "seed": self.seed,
            "trial_count": self.trial_count,
            "worlds": list(self.worlds),
        }

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))


@dataclass(slots=True)
class _EvidenceState:
    observations: list[ConsumptionObservation]
    expected_manifest_sha256: str
    observed_manifest_sha256: str
    expected_command_sha256: str
    observed_command_sha256: str
    expected_output_sha256: str
    observed_output_sha256: str
    required_metrics: tuple[str, ...]
    present_metrics: list[str]
    cutoff_epoch: int
    max_available_epoch: int


def _digest_int(plan: RandomMutationPlan, trial: int, label: str) -> int:
    return int.from_bytes(
        hashlib.sha256(
            canonical_json_bytes({"label": label, "seed": plan.seed, "trial": trial})
        ).digest()[:8],
        "big",
    )


def _contract(plan: RandomMutationPlan, replicate: int) -> CouplingContract:
    declarations = tuple(
        EventDeclaration(
            event_type=branch,
            draw_role="quantity",
            distribution_family="discrete_uniform_inclusive",
            distribution_parameters=(("lower", 1), ("upper", 17)),
            coupling_group="shared-action",
            world_scope=WorldScope.SHARED,
            requirement=Requirement.REQUIRED,
        )
        for branch in plan.branches
    )
    return generate_contract(
        ScenarioSpecification(
            base_experiment_id=plan.experiment_id,
            scenario_namespace="random-mutation-campaign",
            logical_epochs=plan.logical_epochs,
            entity_ids=plan.agent_ids,
            declarations=declarations,
            configuration_version=plan.sha256[:16],
        ),
        replicate,
    )


def _clean_state(
    plan: RandomMutationPlan, replicate: int
) -> tuple[CouplingContract, _EvidenceState]:
    contract = _contract(plan, replicate)
    backend = EventKeyedBackend(f"{plan.seed}:clean:{replicate}")
    observations = [
        ConsumptionObservation.consumed(world, event, backend.draw(event))
        for world in World
        for event in contract.events
    ]
    manifest = hashlib.sha256(f"manifest:{replicate}".encode()).hexdigest()
    command = hashlib.sha256(f"commands:{replicate}".encode()).hexdigest()
    output = hashlib.sha256(f"output:{replicate}".encode()).hexdigest()
    return contract, _EvidenceState(
        observations=observations,
        expected_manifest_sha256=manifest,
        observed_manifest_sha256=manifest,
        expected_command_sha256=command,
        observed_command_sha256=command,
        expected_output_sha256=output,
        observed_output_sha256=output,
        required_metrics=plan.required_metrics,
        present_metrics=list(plan.required_metrics),
        cutoff_epoch=max(plan.logical_epochs) + 10,
        max_available_epoch=max(plan.logical_epochs),
    )


def _weighted_operator_sample(
    plan: RandomMutationPlan, trial: int, count: int
) -> tuple[str, ...]:
    remaining = list(plan.operators)
    selected: list[str] = []
    for position in range(count):
        total = sum(item.probability_weight for item in remaining)
        draw = _digest_int(plan, trial, f"operator:{position}") % total
        cumulative = 0
        for index, item in enumerate(remaining):
            cumulative += item.probability_weight
            if draw < cumulative:
                selected.append(item.name)
                remaining.pop(index)
                break
    return tuple(selected)


def _coordinate(
    plan: RandomMutationPlan, trial: int, operation_index: int
) -> dict[str, object]:
    def choose(values: Sequence[object], label: str) -> object:
        return values[
            _digest_int(plan, trial, f"coordinate:{operation_index}:{label}")
            % len(values)
        ]

    return {
        "agent": choose(plan.agent_ids, "agent"),
        "branch": choose(plan.branches, "branch"),
        "epoch": choose(plan.logical_epochs, "epoch"),
        "magnitude": 1
        + _digest_int(plan, trial, f"coordinate:{operation_index}:magnitude") % 17,
        "trace_position": _digest_int(
            plan, trial, f"coordinate:{operation_index}:trace_position"
        ),
        "world": choose(plan.worlds, "world"),
    }


def _matches_coordinate(
    observation: ConsumptionObservation, coordinate: Mapping[str, object]
) -> bool:
    event = observation.observed_event
    return (
        event is not None
        and event.entity_id == coordinate["agent"]
        and event.logical_epoch == coordinate["epoch"]
        and event.event_type == coordinate["branch"]
        and observation.world.value == coordinate["world"]
    )


def _selected_index(state: _EvidenceState, coordinate: Mapping[str, object]) -> int:
    candidates = [
        index
        for index, observation in enumerate(state.observations)
        if _matches_coordinate(observation, coordinate)
    ]
    if not candidates:
        candidates = list(range(len(state.observations)))
    if not candidates:
        _fail("mutation removed every receipt before a later receipt mutation")
    return candidates[
        _integer(coordinate["trace_position"], "trace_position") % len(candidates)
    ]


def _changed_digest(
    value: str, operation: str, coordinate: Mapping[str, object]
) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {"coordinate": dict(coordinate), "operation": operation, "value": value}
        )
    ).hexdigest()


def _apply_mutation(
    state: _EvidenceState,
    operation: str,
    coordinate: Mapping[str, object],
) -> tuple[str, str] | None:
    receipt_operations = set(MUTATION_OPERATORS[:7])
    selected_key: tuple[str, str] | None = None
    if operation in receipt_operations:
        index = _selected_index(state, coordinate)
        selected = state.observations[index]
        selected_key = (selected.world.value, selected.event_identity)
        if operation == "delete_receipt":
            state.observations.pop(index)
        elif operation == "insert_unexpected_event":
            state.observations.append(
                replace(
                    selected,
                    event_identity="pcc1:"
                    + _changed_digest(selected.event_identity, operation, coordinate),
                )
            )
        elif operation == "duplicate_event":
            state.observations.insert(index + 1, selected)
        elif operation == "rename_semantic_key":
            state.observations[index] = replace(
                selected,
                event_identity="pcc1:"
                + _changed_digest(selected.event_identity, operation, coordinate),
            )
        else:
            if selected.observed_event is None:
                _fail("metadata mutation selected an observation without metadata")
            event = selected.observed_event
            if operation == "flip_world_scope":
                changed = replace(
                    event,
                    world_scope=(
                        WorldScope.POLICY_ONLY
                        if selected.world is World.BASELINE
                        else WorldScope.BASELINE_ONLY
                    ),
                )
            elif operation == "mutate_draw_role":
                changed = replace(event, draw_role="mutated-quantity-role")
            else:
                parameters = dict(event.distribution_parameters)
                parameters["upper"] = _integer(
                    parameters["upper"], "distribution upper"
                ) + _integer(coordinate["magnitude"], "mutation magnitude")
                changed = replace(
                    event, distribution_parameters=tuple(sorted(parameters.items()))
                )
            state.observations[index] = replace(selected, observed_event=changed)
    elif operation == "reorder_commands":
        state.observed_command_sha256 = _changed_digest(
            state.observed_command_sha256, operation, coordinate
        )
    elif operation == "mutate_output":
        state.observed_output_sha256 = _changed_digest(
            state.observed_output_sha256, operation, coordinate
        )
    elif operation == "stale_manifest":
        state.observed_manifest_sha256 = _changed_digest(
            state.observed_manifest_sha256, operation, coordinate
        )
    elif operation == "remove_required_metric":
        if state.present_metrics:
            index = _integer(coordinate["magnitude"], "mutation magnitude") % len(
                state.present_metrics
            )
            state.present_metrics.pop(index)
    elif operation == "change_cutoff_metadata":
        state.max_available_epoch = state.cutoff_epoch + _integer(
            coordinate["magnitude"], "mutation magnitude"
        )
    else:
        _fail(f"unknown mutation operator: {operation}")
    return selected_key


def _metadata(event: SemanticEventAddress) -> dict[str, object]:
    return event.to_mapping()


def _capsule(contract: CouplingContract, state: _EvidenceState) -> dict[str, object]:
    event_rows = [
        {
            "event_identity": event.identity,
            "metadata": _metadata(event),
            "requirement": event.requirement.value,
            "world_scope": event.world_scope.value,
        }
        for event in contract.events
    ]
    observations = {
        world.value: [
            {
                "event_identity": item.event_identity,
                "metadata": (
                    {}
                    if item.observed_event is None
                    else _metadata(item.observed_event)
                ),
                "state": item.state.value,
                "value_commitment": item.value_commitment,
            }
            for item in state.observations
            if item.world is world
        ]
        for world in World
    }
    return {
        "contract": {
            "events": event_rows,
            "sha256": capsule_sha256(event_rows),
        },
        "execution": {
            "cutoff_epoch": state.cutoff_epoch,
            "expected_command_sha256": state.expected_command_sha256,
            "expected_manifest_sha256": state.expected_manifest_sha256,
            "expected_output_sha256": state.expected_output_sha256,
            "max_available_epoch": state.max_available_epoch,
            "observed_command_sha256": state.observed_command_sha256,
            "observed_manifest_sha256": state.observed_manifest_sha256,
            "observed_output_sha256": state.observed_output_sha256,
            "present_metrics": list(state.present_metrics),
            "required_metrics": list(state.required_metrics),
        },
        "observations": observations,
        "schema_version": MINIMAL_EVIDENCE_CAPSULE_SCHEMA_VERSION,
    }


def _full_stack_detects(contract: CouplingContract, state: _EvidenceState) -> bool:
    return (
        not audit_pair(contract, tuple(state.observations)).passed
        or state.expected_manifest_sha256 != state.observed_manifest_sha256
        or state.expected_command_sha256 != state.observed_command_sha256
        or state.expected_output_sha256 != state.observed_output_sha256
        or not set(state.required_metrics).issubset(state.present_metrics)
        or state.max_available_epoch > state.cutoff_epoch
    )


def _observation_signature(observation: ConsumptionObservation) -> bytes:
    return canonical_json_bytes(
        {
            "event_identity": observation.event_identity,
            "metadata": (
                None
                if observation.observed_event is None
                else observation.observed_event.to_mapping()
            ),
            "state": observation.state.value,
            "value_commitment": observation.value_commitment,
            "world": observation.world.value,
        }
    )


def _containment_rate(
    clean: Sequence[ConsumptionObservation],
    mutated: Sequence[ConsumptionObservation],
    targeted: set[tuple[str, str]],
) -> float:
    clean_map = {
        (item.world.value, item.event_identity): _observation_signature(item)
        for item in clean
        if (item.world.value, item.event_identity) not in targeted
    }
    mutated_map = {
        (item.world.value, item.event_identity): _observation_signature(item)
        for item in mutated
        if (item.world.value, item.event_identity) not in targeted
        and (item.world.value, item.event_identity) in clean_map
    }
    if not clean_map:
        return 1.0
    unchanged = sum(mutated_map.get(key) == value for key, value in clean_map.items())
    return unchanged / len(clean_map)


def run_random_mutation_campaign(
    plan: RandomMutationPlan,
) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    """Run randomized, non-overwriting compound mutations and clean controls."""
    rows: list[dict[str, object]] = []
    capsules: list[dict[str, object]] = []
    operator_counts: Counter[str] = Counter()
    for trial in range(plan.trial_count):
        contract, state = _clean_state(plan, trial)
        clean_observations = tuple(state.observations)
        clean_capsule = _capsule(contract, state)
        count = plan.fault_count_min + _digest_int(plan, trial, "fault_count") % (
            plan.fault_count_max - plan.fault_count_min + 1
        )
        operations = _weighted_operator_sample(plan, trial, count)
        coordinates: list[dict[str, object]] = []
        targeted: set[tuple[str, str]] = set()
        for index, operation in enumerate(operations):
            coordinate = _coordinate(plan, trial, index)
            coordinates.append({"operation": operation, **coordinate})
            selected = _apply_mutation(state, operation, coordinate)
            if selected is not None:
                targeted.add(selected)
            operator_counts[operation] += 1
        mutated_capsule = _capsule(contract, state)
        if canonical_json_bytes(clean_capsule) == canonical_json_bytes(mutated_capsule):
            _fail("generated mutation is equivalent to its clean control")
        full_detected = _full_stack_detects(contract, state)
        minimal_result = verify_capsule(mutated_capsule)
        row = {
            "containment_rate": _containment_rate(
                clean_observations, state.observations, targeted
            ),
            "coordinates": coordinates,
            "fault_count": count,
            "full_stack_detected": full_detected,
            "minimal_verifier_detected": not minimal_result.accepted,
            "operator_order": list(operations),
            "replicate_id": trial,
            "verifier_agreement": full_detected == (not minimal_result.accepted),
            "violation_codes": list(minimal_result.violations),
        }
        rows.append(row)
        capsules.append(
            {
                "capsule": mutated_capsule,
                "capsule_sha256": minimal_result.capsule_sha256,
                "replicate_id": trial,
            }
        )

    clean_rows: list[dict[str, object]] = []
    clean_capsule_example: dict[str, object] | None = None
    for control in range(plan.clean_control_count):
        contract, state = _clean_state(plan, 1_000_000 + control)
        capsule = _capsule(contract, state)
        if clean_capsule_example is None:
            clean_capsule_example = capsule
        full_alarm = _full_stack_detects(contract, state)
        minimal_alarm = not verify_capsule(capsule).accepted
        clean_rows.append(
            {
                "control_id": control,
                "full_stack_alarm": full_alarm,
                "minimal_verifier_alarm": minimal_alarm,
            }
        )
    undetected = [row for row in rows if not row["full_stack_detected"]]
    undetected.sort(
        key=lambda row: (
            _integer(row["fault_count"], "fault_count"),
            _integer(row["replicate_id"], "replicate_id"),
        )
    )
    false_negative_patterns = Counter(
        "+".join(cast(list[str], row["operator_order"])) for row in undetected
    )
    artifact: dict[str, object] = {
        "clean_control_false_positives": {
            "full_stack": sum(bool(row["full_stack_alarm"]) for row in clean_rows),
            "minimal_verifier": sum(
                bool(row["minimal_verifier_alarm"]) for row in clean_rows
            ),
            "total": len(clean_rows),
        },
        "clean_rows": clean_rows,
        "config": plan.to_mapping(),
        "config_sha256": plan.sha256,
        "containment_rate_mean": statistics.fmean(
            cast(float, row["containment_rate"]) for row in rows
        ),
        "detection_rate": sum(bool(row["full_stack_detected"]) for row in rows)
        / len(rows),
        "false_negative_patterns": dict(sorted(false_negative_patterns.items())),
        "minimal_undetected_counterexample": undetected[0] if undetected else None,
        "minimal_verifier_agreement_rate": sum(
            bool(row["verifier_agreement"]) for row in rows
        )
        / len(rows),
        "mutation_score": sum(bool(row["full_stack_detected"]) for row in rows)
        / len(rows),
        "operator_instance_counts": dict(sorted(operator_counts.items())),
        "operator_universe_size": len(plan.operators),
        "rows": rows,
        "schema_version": RANDOM_MUTATION_CAMPAIGN_SCHEMA_VERSION,
        "synthetic_only": True,
        "trial_count": len(rows),
        "unique_operator_set_count": len(
            {tuple(cast(list[str], row["operator_order"])) for row in rows}
        ),
    }
    if clean_capsule_example is None:
        raise AssertionError("clean controls unexpectedly empty")
    artifact["clean_capsule_example"] = clean_capsule_example
    return artifact, tuple(capsules)


__all__ = [
    "MUTATION_OPERATORS",
    "RANDOM_MUTATION_CAMPAIGN_SCHEMA_VERSION",
    "MutationOperator",
    "RandomMutationCampaignError",
    "RandomMutationPlan",
    "run_random_mutation_campaign",
]
