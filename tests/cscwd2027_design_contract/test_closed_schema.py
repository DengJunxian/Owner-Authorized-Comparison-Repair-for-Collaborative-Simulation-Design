"""No silent rule or integer/boolean substitutions in comparison authority."""

from __future__ import annotations

from dataclasses import replace

import pytest

from policytwin.design_contract.contract import ComparisonContract, ExpectedSlot


def test_unsupported_decision_rule_is_rejected() -> None:
    contract = ComparisonContract("fixture", 1, (ExpectedSlot("a", 0, "key"),), ("p",))
    with pytest.raises(ValueError, match="unsupported"):
        replace(contract, rule="minimize something else")


def test_boolean_replicate_is_not_integer_zero_or_one() -> None:
    with pytest.raises(ValueError, match="coordinate"):
        ExpectedSlot("a", True, "key")
