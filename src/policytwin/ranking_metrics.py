"""Tie-aware ranking metrics for the paired-execution benchmark.

The registered scientific reference is the vector of full-run event-keyed
branch means.  Candidate rankings are computed within each frozen replicate
batch.  A ranking is a weak order: ties are part of the result and are never
silently broken by branch identifiers.
"""

from __future__ import annotations

import math
from collections.abc import Mapping


class RankingMetricError(ValueError):
    """Raised when ranking inputs cannot define the requested metric."""


def _validated_pair(
    reference: Mapping[str, float], observed: Mapping[str, float]
) -> tuple[str, ...]:
    keys = tuple(sorted(reference))
    if len(keys) < 2 or set(keys) != set(observed):
        raise RankingMetricError(
            "reference and observed rankings need the same two or more branches"
        )
    if any(
        isinstance(value, bool) or not isinstance(value, int | float)
        for value in (*reference.values(), *observed.values())
    ):
        raise RankingMetricError("ranking values must be numeric")
    if any(
        not math.isfinite(float(value))
        for value in (*reference.values(), *observed.values())
    ):
        raise RankingMetricError("ranking values must be finite")
    return keys


def _sign(value: float) -> int:
    return (value > 0.0) - (value < 0.0)


def pairwise_signature(values: Mapping[str, float]) -> tuple[int, ...]:
    """Return all unordered branch comparisons in sorted branch-label order."""
    keys = tuple(sorted(values))
    if len(keys) < 2:
        raise RankingMetricError("a ranking needs at least two branches")
    return tuple(
        _sign(float(values[left]) - float(values[right]))
        for index, left in enumerate(keys)
        for right in keys[index + 1 :]
    )


def exact_weak_order_match(
    reference: Mapping[str, float], observed: Mapping[str, float]
) -> bool:
    """Return whether order and every tie agree exactly."""
    _validated_pair(reference, observed)
    return pairwise_signature(reference) == pairwise_signature(observed)


def kendall_tau_b(
    reference: Mapping[str, float], observed: Mapping[str, float]
) -> float:
    """Compute Kendall's tau-b with pairs tied in both rankings excluded.

    ``ties_reference_only`` and ``ties_observed_only`` follow the standard
    tau-b denominator.  Counting a pair tied in both rankings in both terms
    would incorrectly make a ranking disagree with itself.
    """
    keys = _validated_pair(reference, observed)
    concordant = 0
    discordant = 0
    ties_reference_only = 0
    ties_observed_only = 0
    for index, left in enumerate(keys):
        for right in keys[index + 1 :]:
            reference_sign = _sign(
                float(reference[left]) - float(reference[right])
            )
            observed_sign = _sign(float(observed[left]) - float(observed[right]))
            if reference_sign and observed_sign:
                if reference_sign == observed_sign:
                    concordant += 1
                else:
                    discordant += 1
            elif reference_sign == 0 and observed_sign != 0:
                ties_reference_only += 1
            elif reference_sign != 0 and observed_sign == 0:
                ties_observed_only += 1
            # A pair tied in both rankings contributes to neither correction.
    denominator = math.sqrt(
        (concordant + discordant + ties_reference_only)
        * (concordant + discordant + ties_observed_only)
    )
    if denominator == 0.0:
        raise RankingMetricError("Kendall tau-b is undefined for constant rankings")
    return (concordant - discordant) / denominator


def pairwise_concordance(
    reference: Mapping[str, float], observed: Mapping[str, float]
) -> float:
    """Return the fraction of all branch pairs with the same comparison sign.

    The denominator is ``n * (n - 1) / 2``.  A tie agrees only with a tie;
    breaking a reference tie or introducing a new tie counts as disagreement.
    """
    _validated_pair(reference, observed)
    left = pairwise_signature(reference)
    right = pairwise_signature(observed)
    return sum(a == b for a, b in zip(left, right, strict=True)) / len(left)


def top_set(values: Mapping[str, float]) -> frozenset[str]:
    """Return every branch tied at the maximum value."""
    if not values:
        raise RankingMetricError("top-set input must not be empty")
    maximum = max(float(value) for value in values.values())
    return frozenset(key for key, value in values.items() if float(value) == maximum)


def top1_set_accuracy(
    reference: Mapping[str, float], observed: Mapping[str, float]
) -> bool:
    """Return whether the complete observed and reference top sets match."""
    _validated_pair(reference, observed)
    return top_set(reference) == top_set(observed)


def selection_regret(
    reference: Mapping[str, float], observed: Mapping[str, float]
) -> float:
    """Return reference regret using the best member of the observed top set.

    Tied observed recommendations are treated as a set.  This avoids injecting
    an unregistered label-order choice.  Top-set disagreement remains visible
    through :func:`top1_set_accuracy`.
    """
    _validated_pair(reference, observed)
    reference_best = max(float(value) for value in reference.values())
    selected_reference_value = max(
        float(reference[key]) for key in top_set(observed)
    )
    return reference_best - selected_reference_value


__all__ = [
    "RankingMetricError",
    "exact_weak_order_match",
    "kendall_tau_b",
    "pairwise_concordance",
    "pairwise_signature",
    "selection_regret",
    "top1_set_accuracy",
    "top_set",
]
