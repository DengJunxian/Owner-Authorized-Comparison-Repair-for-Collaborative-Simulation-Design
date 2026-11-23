"""Hand-checkable fixtures for the separate reviewer diagnostic."""

from dataclasses import replace
from fractions import Fraction

from policytwin.design_contract.contract import (
    ComparisonContract,
    Evidence,
    ExpectedSlot,
    Member,
    decide,
)
from scripts.cscwd2027_review_response import mean_gap, without_permission


def fixture():
    contract = ComparisonContract(
        "fixture",
        1,
        (ExpectedSlot("a", 0, "ka"), ExpectedSlot("b", 0, "kb")),
        ("allowed",),
    )
    panel = [
        Member(c, 0, Evidence(c, "k" + c, "other", c, True, i, c, 0))
        for i, c in enumerate(("a", "b"))
    ]
    return contract, panel


def test_plan_permission_has_an_independent_negative_fixture():
    contract, panel = fixture()
    assert decide(contract, panel, "comparison_contract")["findings"] == [
        "permitted_plan"
    ]
    assert without_permission(contract, panel)["authorized"]
    assert decide(
        replace(contract, allowed_plans=("other",)), panel, "comparison_contract"
    )["authorized"]


def test_permission_ablation_preserves_other_obligations():
    contract, panel = fixture()
    assert not without_permission(contract, panel[:1])["authorized"]
    bad = replace(panel[0].evidence, scientific_key="foreign")
    assert not without_permission(
        contract, [replace(panel[0], evidence=bad), panel[1]]
    )["authorized"]
    bad = replace(panel[0].evidence, local_pass=False)
    assert not without_permission(
        contract, [replace(panel[0], evidence=bad), panel[1]]
    )["authorized"]


def test_exact_top_two_gaps_include_ties():
    assert mean_gap({"means": {"a": [9, 4], "b": [2, 1], "c": [-3, 1]}}) == Fraction(
        1, 4
    )
    assert mean_gap({"means": {"a": [2, 1], "b": [2, 1], "c": [-3, 1]}}) == 0
