from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import pytest

from policytwin.counterfactual.coupling_contract import (
    ConformanceViolationCode,
    ConsumptionObservation,
    CouplingContract,
    CouplingContractError,
    EventDeclaration,
    Requirement,
    ScenarioSpecification,
    SemanticEventAddress,
    World,
    WorldScope,
    audit_pair,
    conform,
    generate_contract,
)


def _contract() -> CouplingContract:
    return generate_contract(
        ScenarioSpecification(
            base_experiment_id="conform-fixture",
            scenario_namespace="hand-checkable",
            logical_epochs=(0,),
            entity_ids=("agent-a",),
            declarations=(
                EventDeclaration(
                    event_type="decision",
                    draw_role="quantity",
                    distribution_family="discrete_uniform_inclusive",
                    distribution_parameters=(("lower", 1), ("upper", 5)),
                    coupling_group="shared-action",
                    world_scope=WorldScope.SHARED,
                    requirement=Requirement.REQUIRED,
                ),
                EventDeclaration(
                    event_type="retry",
                    draw_role="retry_quantity",
                    distribution_family="discrete_uniform_inclusive",
                    distribution_parameters=(("lower", 1), ("upper", 3)),
                    coupling_group="policy-retry",
                    world_scope=WorldScope.POLICY_ONLY,
                    requirement=Requirement.OPTIONAL,
                ),
            ),
            configuration_version="v1",
        ),
        0,
    )


def _events() -> tuple[CouplingContract, SemanticEventAddress, SemanticEventAddress]:
    contract = _contract()
    shared = next(
        event for event in contract.events if event.world_scope is WorldScope.SHARED
    )
    policy_only = next(
        event
        for event in contract.events
        if event.world_scope is WorldScope.POLICY_ONLY
    )
    return contract, shared, policy_only


def _clean() -> tuple[
    CouplingContract,
    tuple[ConsumptionObservation, ...],
    tuple[ConsumptionObservation, ...],
]:
    contract, shared, policy_only = _events()
    baseline = (ConsumptionObservation.consumed(World.BASELINE, shared, 3),)
    policy = (
        ConsumptionObservation.consumed(World.POLICY, shared, 3),
        ConsumptionObservation.masked(World.POLICY, policy_only),
    )
    return contract, baseline, policy


def _mutate_role(observation: ConsumptionObservation) -> ConsumptionObservation:
    assert observation.observed_event is not None
    return replace(
        observation,
        observed_event=replace(
            observation.observed_event, draw_role="replacement_quantity"
        ),
    )


def _mutate_family(observation: ConsumptionObservation) -> ConsumptionObservation:
    assert observation.observed_event is not None
    return replace(
        observation,
        observed_event=replace(
            observation.observed_event, distribution_family="uniform_0_1"
        ),
    )


def _mutate_scope(observation: ConsumptionObservation) -> ConsumptionObservation:
    assert observation.observed_event is not None
    return replace(
        observation,
        observed_event=replace(
            observation.observed_event, world_scope=WorldScope.POLICY_ONLY
        ),
    )


def test_conform_accepts_required_shared_optional_masked_and_not_applicable() -> None:
    contract, baseline, policy = _clean()
    assert conform(contract, baseline, policy)
    report = audit_pair(contract, (*baseline, *policy))
    assert report.passed
    assert report.violations == ()


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    (
        (
            _mutate_role,
            ConformanceViolationCode.DRAW_ROLE_MISMATCH,
        ),
        (
            _mutate_family,
            ConformanceViolationCode.DISTRIBUTION_FAMILY_MISMATCH,
        ),
        (
            _mutate_scope,
            ConformanceViolationCode.WORLD_SCOPE_MISMATCH,
        ),
    ),
)
def test_conform_detects_receipt_metadata_substitution(
    mutation: Callable[[ConsumptionObservation], ConsumptionObservation],
    expected_code: ConformanceViolationCode,
) -> None:
    contract, baseline, policy = _clean()
    changed = (mutation(policy[0]), policy[1])
    report = audit_pair(contract, (*baseline, *changed))
    assert not report.passed
    assert expected_code in {violation.code for violation in report.violations}
    assert ConformanceViolationCode.SEMANTIC_KEY_SUBSTITUTION in {
        violation.code for violation in report.violations
    }


def test_conform_detects_missing_duplicate_unexpected_and_key_substitution() -> None:
    contract, baseline, policy = _clean()
    substituted = replace(policy[0], event_identity="pcc1:" + "f" * 64)
    report = audit_pair(contract, (*baseline, substituted, substituted, policy[1]))
    codes = {violation.code for violation in report.violations}
    assert not report.passed
    assert ConformanceViolationCode.MISSING in codes
    assert ConformanceViolationCode.UNEXPECTED in codes


def test_required_mask_and_shared_commitment_mismatch_fail() -> None:
    contract, baseline, policy = _clean()
    assert policy[0].observed_event is not None
    required_masked = ConsumptionObservation.masked(
        World.POLICY,
        policy[0].observed_event,
    )
    report = audit_pair(contract, (*baseline, required_masked, policy[1]))
    assert ConformanceViolationCode.INVALID_STATUS in {
        violation.code for violation in report.violations
    }

    mismatch = replace(
        policy[0],
        value_commitment="0" * 64,
    )
    report = audit_pair(contract, (*baseline, mismatch, policy[1]))
    assert ConformanceViolationCode.SHARED_VALUE_MISMATCH in {
        violation.code for violation in report.violations
    }


def test_conform_requires_world_separated_inputs() -> None:
    contract, baseline, policy = _clean()
    with pytest.raises(CouplingContractError, match="O0"):
        conform(contract, (*baseline, policy[0]), policy)
