"""Small artificial fixtures for set-level conformance, ties and reuse."""

from __future__ import annotations

from dataclasses import replace

import pytest

from policytwin.design_contract.contract import (
    ComparisonContract,
    Evidence,
    ExpectedSlot,
    Member,
    decide,
    revision_plan,
)
from scripts.cscwd2027_design_contract.evaluate import consequences, panel_valid


def fixture() -> tuple[ComparisonContract, list[Member]]:
    """Two alternatives x two replicates; every local artifact is valid."""
    slots = tuple(ExpectedSlot(c, r, f"{c}-{r}") for c in ("a", "b") for r in range(2))
    contract = ComparisonContract("manual", 1, slots, ("p0", "p1"))
    members = [
        Member(
            s.candidate,
            s.replicate,
            Evidence(
                f"obj-{s.candidate}-{s.replicate}",
                s.scientific_key,
                "p0",
                f"cap-{s.scientific_key}",
                True,
                2 if s.candidate == "a" else 1,
                s.candidate,
                s.replicate,
            ),
        )
        for s in slots
    ]
    return contract, members


def test_exact_panel_and_authorized_remap() -> None:
    contract, members = fixture()
    changed = [
        replace(m, evidence=replace(m.evidence, plan_sha256="p1")) for m in members
    ]
    assert decide(contract, changed, "comparison_contract")["authorized"]
    assert not decide(contract, changed, "strict_plan_lock")["authorized"]


def test_local_validity_does_not_bind_intended_science() -> None:
    contract, members = fixture()
    foreign = replace(
        members[0], evidence=replace(members[0].evidence, scientific_key="foreign")
    )
    values = [foreign, *members[1:]]
    assert decide(contract, values, "local_full_only")["authorized"]
    assert decide(contract, values, "spec_grouped")["authorized"]
    assert not decide(contract, values, "comparison_contract")["authorized"]


@pytest.mark.parametrize("operation", ["missing", "duplicate"])
def test_manual_expected_set_is_not_received_set(operation: str) -> None:
    contract, members = fixture()
    changed = members[1:] if operation == "missing" else [*members, members[0]]
    assert decide(contract, changed, "local_full_only")["authorized"]
    assert not decide(contract, changed, "comparison_contract")["authorized"]
    assert decide(contract, changed, "minus_slot_coverage")["authorized"]


def test_self_describing_labels_recover_without_contract_advantage() -> None:
    contract, members = fixture()
    swapped = [
        replace(m, candidate="b" if m.candidate == "a" else "a") for m in members
    ]
    local = decide(contract, swapped, "local_full_only")
    grouped = decide(contract, swapped, "spec_grouped")
    full = decide(contract, swapped, "comparison_contract")
    assert local["numerical"]["top1"] == "b"
    assert grouped["numerical"]["top1"] == full["numerical"]["top1"] == "a"
    assert panel_valid(contract, grouped["numerical"])
    assert not panel_valid(contract, local["numerical"])
    assert len(full["normalizations"]) == 4


def test_relational_equal_obligation_control() -> None:
    contract, members = fixture()
    for values in (members, members[1:], [*members, members[0]]):
        assert (
            decide(contract, values, "comparison_contract")["authorized"]
            == decide(contract, values, "conventional_relational")["authorized"]
        )


def test_revision_is_key_based_not_value_based() -> None:
    contract, members = fixture()
    changed = replace(
        contract,
        version=2,
        slots=(replace(contract.slots[0], scientific_key="new"), *contract.slots[1:]),
    )
    plan = revision_plan(changed, members)
    assert plan["rerun"] == [{"candidate": "a", "replicate": 0}]
    assert len(plan["reuse"]) == 3
    inflated = [replace(m, evidence=replace(m.evidence, value=-99999)) for m in members]
    assert revision_plan(changed, inflated) == plan


def test_tie_aware_error_and_undefined_abstention() -> None:
    contract, members = fixture()
    same = [replace(m, evidence=replace(m.evidence, value=1)) for m in members]
    reference = decide(contract, same, "comparison_contract")["numerical"]
    alternate = {**reference, "top1": "b"}
    outcome = consequences(reference, alternate)
    assert outcome["top1_disagreement"]
    assert not outcome["optimal_set_error"]
    assert outcome["zero_regret_disagreement"]
    assert outcome["regret"] == [0, 1]
    assert consequences(reference, None)["regret"] is None


def test_duplicate_expected_slots_rejected() -> None:
    contract, _ = fixture()
    with pytest.raises(ValueError, match="duplicate"):
        replace(contract, slots=(*contract.slots, contract.slots[0]))
