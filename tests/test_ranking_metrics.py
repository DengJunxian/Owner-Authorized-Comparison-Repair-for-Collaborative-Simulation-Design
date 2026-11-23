from __future__ import annotations

import math

import pytest

from policytwin.ranking_metrics import (
    RankingMetricError,
    exact_weak_order_match,
    kendall_tau_b,
    pairwise_concordance,
    selection_regret,
    top1_set_accuracy,
)


def test_hand_checkable_tie_fixture() -> None:
    # Pair order is (a,b), (a,c), (b,c).  The reference signs are (-1,-1,0).
    reference = {"a": 1.0, "b": 2.0, "c": 2.0}
    same = {"a": 10.0, "b": 20.0, "c": 20.0}
    broken_tie = {"a": 10.0, "b": 20.0, "c": 21.0}

    assert exact_weak_order_match(reference, same)
    assert kendall_tau_b(reference, same) == pytest.approx(1.0)
    assert pairwise_concordance(reference, same) == pytest.approx(1.0)

    assert not exact_weak_order_match(reference, broken_tie)
    assert kendall_tau_b(reference, broken_tie) == pytest.approx(2 / math.sqrt(6))
    assert pairwise_concordance(reference, broken_tie) == pytest.approx(2 / 3)


def test_reversal_and_new_tie_are_not_concordant() -> None:
    reference = {"a": 1.0, "b": 2.0, "c": 3.0}
    observed = {"a": 3.0, "b": 2.0, "c": 2.0}
    assert kendall_tau_b(reference, observed) == pytest.approx(-2 / math.sqrt(6))
    assert pairwise_concordance(reference, observed) == 0.0
    assert not exact_weak_order_match(reference, observed)


def test_top_set_and_regret_tie_rules() -> None:
    reference = {"a": 1.0, "b": 4.0, "c": 3.0}
    observed = {"a": 0.0, "b": 2.0, "c": 2.0}
    assert not top1_set_accuracy(reference, observed)
    assert selection_regret(reference, observed) == 0.0

    wrong = {"a": 2.0, "b": 0.0, "c": 3.0}
    assert selection_regret(reference, wrong) == 1.0


def test_input_validation_and_constant_tau() -> None:
    with pytest.raises(RankingMetricError, match="same"):
        pairwise_concordance({"a": 1.0, "b": 2.0}, {"a": 1.0, "c": 2.0})
    with pytest.raises(RankingMetricError, match="finite"):
        kendall_tau_b({"a": 1.0, "b": 2.0}, {"a": 1.0, "b": math.nan})
    with pytest.raises(RankingMetricError, match="undefined"):
        kendall_tau_b({"a": 0.0, "b": 0.0}, {"a": 1.0, "b": 1.0})
