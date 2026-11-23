"""Exact finite-lattice order-witness reference for rank-test mechanics.

The reference in this module is intentionally narrower than a continuous
normal--normal simulator.  For each meta-replicate it produces one oracle and
``M`` posterior order witnesses from independent semantic :class:`EventKey`
identities.  A witness is the exact midpoint

``(2 * k + 1) / 2**257``, where ``k`` is a hash-derived 256-bit integer.

Under the explicitly declared iid finite-lattice law, the witnesses are
exchangeable.  A fixed SHA-256 grid is only a deterministic conformance fixture
and is not itself statistical evidence of independence.  The witnesses are
order-isomorphic to a strictly increasing toy posterior functional, but they
are not policy-effect magnitudes or exact continuous-normal draws.  Exact
integer dynamic programming derives the reachable critical rule, its actual
finite-sample size, and power against a closed set of in-code-frozen rank-PMF
distortions.  Nothing here validates the integrated ABC posterior, market
mechanisms, real coverage, causal effects, external registration, or release
eligibility.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from functools import lru_cache
from math import comb, gcd, lcm
from typing import Literal, NoReturn

from policytwin.counterfactual.randomness import EventKey, deterministic_int
from policytwin.evaluation.ledger_derived import (
    ExactRational,
    LedgerDerivedEvidenceError,
)
from policytwin.evaluation.ledger_policy_functional_sbc import (
    exact_rank_cdf_discrepancy,
)
from policytwin.provenance import canonical_json_bytes, sha256_bytes

POLICY_FUNCTIONAL_ORDER_WITNESS_SCHEMA_VERSION = (
    "policytwin.policy-functional-order-witness.v2"
)
# Result-schema revisions that only rename or clarify diagnostics must not redraw
# the frozen witness grid.  This value deliberately preserves the v1 generation
# mapping bytes; changing it is a scientific RNG-design migration, not a schema
# migration.
POLICY_FUNCTIONAL_ORDER_WITNESS_GENERATION_SCHEMA_VERSION: Literal[
    "policytwin.policy-functional-order-witness.v1"
] = "policytwin.policy-functional-order-witness.v1"
POLICY_FUNCTIONAL_ORDER_WITNESS_CLAIM_SCOPE: Literal[
    "exact_rank_test_arithmetic_under_declared_iid_finite_lattice_law_only"
] = "exact_rank_test_arithmetic_under_declared_iid_finite_lattice_law_only"
POLICY_FUNCTIONAL_ORDER_WITNESS_FUNCTIONAL: Literal[
    "strictly_increasing_toy_paired_effect_order_only_not_effect_magnitude"
] = "strictly_increasing_toy_paired_effect_order_only_not_effect_magnitude"
POLICY_FUNCTIONAL_ORDER_WITNESS_REJECTION_RULE: Literal[
    "reject_when_exact_rank_cdf_discrepancy_strictly_exceeds_reachable_critical"
] = "reject_when_exact_rank_cdf_discrepancy_strictly_exceeds_reachable_critical"
POLICY_FUNCTIONAL_ORDER_WITNESS_RNG_IMPLEMENTATION: Literal[
    "legacy_event_key_sha256_rejection_midpoint_v1"
] = "legacy_event_key_sha256_rejection_midpoint_v1"

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_EVENT_KEY = re.compile(r"^ptk1:[0-9a-f]{64}$")
_WITNESS_BITS = 256
_WITNESS_DENOMINATOR = 1 << (_WITNESS_BITS + 1)
_MAX_META_REPLICATES = 256
_MAX_RANK_CATEGORIES = 16
_MAX_WITNESS_CELLS = 4_096


class PolicyFunctionalOrderWitnessError(ValueError):
    """Raised when the exact order-witness contract is violated."""


def _fail(message: str) -> NoReturn:
    raise PolicyFunctionalOrderWitnessError(message)


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        _fail(f"{field} must be a stable ASCII identifier")
    return value


def _seed(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 512:
        _fail(f"{field} must contain between 1 and 512 characters")
    return value


def _positive_int(value: object, field: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"{field} must be an integer of at least {minimum}")
    return value


def _exact(value: object, field: str) -> ExactRational:
    if type(value) is not ExactRational:
        raise TypeError(f"{field} must be an exact ExactRational")
    try:
        fresh = ExactRational(value.numerator, value.denominator)
    except (LedgerDerivedEvidenceError, TypeError, ValueError) as error:
        raise PolicyFunctionalOrderWitnessError(
            f"{field} is not a canonical exact rational"
        ) from error
    if fresh != value:
        _fail(f"{field} is not canonical")
    return value


def _fraction(value: ExactRational, field: str = "exact value") -> Fraction:
    exact = _exact(value, field)
    return Fraction(exact.numerator, exact.denominator)


def _from_fraction(value: Fraction) -> ExactRational:
    return ExactRational.from_ratio(value.numerator, value.denominator)


def _probability(value: object, field: str) -> ExactRational:
    exact = _exact(value, field)
    if not Fraction() < _fraction(exact, field) < Fraction(1):
        _fail(f"{field} must lie strictly between zero and one")
    return exact


def _event_key(value: object, field: str) -> str:
    if not isinstance(value, str) or _EVENT_KEY.fullmatch(value) is None:
        _fail(f"{field} must be a canonical EventKey commitment")
    return value


def _validate_dimensions(
    sample_size: object, category_count: object
) -> tuple[int, int]:
    sample = _positive_int(sample_size, "sample_size")
    categories = _positive_int(category_count, "category_count", minimum=3)
    if sample > _MAX_META_REPLICATES:
        _fail("sample_size exceeds the exact-DP resource bound")
    if categories > _MAX_RANK_CATEGORIES:
        _fail("category_count exceeds the exact-DP resource bound")
    if sample * categories > _MAX_WITNESS_CELLS:
        _fail("rank design exceeds the exact-DP cell bound")
    return sample, categories


def _validate_histogram(histogram: object) -> tuple[int, ...]:
    if (
        not isinstance(histogram, tuple)
        or len(histogram) < 3
        or len(histogram) > _MAX_RANK_CATEGORIES
        or any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0
            for count in histogram
        )
    ):
        _fail("rank histogram is invalid")
    _validate_dimensions(sum(histogram), len(histogram))
    return histogram


def _normalized_weights(weights: object, category_count: int) -> tuple[int, ...]:
    if (
        not isinstance(weights, tuple)
        or len(weights) != category_count
        or any(
            isinstance(weight, bool) or not isinstance(weight, int) or weight < 0
            for weight in weights
        )
        or sum(weights) <= 0
    ):
        _fail("rank-PMF integer weights are invalid")
    divisor = 0
    for weight in weights:
        divisor = gcd(divisor, weight)
    if divisor != 1:
        _fail("rank-PMF integer weights must be reduced by their common divisor")
    return weights


@lru_cache(maxsize=16_384)
def _bounded_sequence_weight(
    sample_size: int,
    category_count: int,
    bound_numerator: int,
    bound_denominator: int,
    weights: tuple[int, ...],
    *,
    inclusive: bool,
) -> int:
    """Return exact labelled-sequence weight inside one CDF boundary."""
    bound = Fraction(bound_numerator, bound_denominator)
    if bound < 0:
        return 0
    total_weight = sum(weights)
    if total_weight <= 0:
        _fail("rank-PMF total weight must be positive")
    paths: dict[int, int] = {0: 1}
    weight_powers = tuple(
        tuple(weight**count for count in range(sample_size + 1)) for weight in weights
    )
    for category, _weight in enumerate(weights, start=1):
        next_paths: dict[int, int] = {}
        for assigned, path_weight in paths.items():
            remaining = sample_size - assigned
            counts: tuple[int, ...] | range
            if category == category_count:
                counts = (remaining,)
            else:
                counts = range(remaining + 1)
            for count in counts:
                cumulative = assigned + count
                discrepancy = abs(
                    Fraction(cumulative, sample_size)
                    - Fraction(category, category_count)
                )
                inside = discrepancy <= bound if inclusive else discrepancy < bound
                if not inside:
                    continue
                contribution = (
                    path_weight
                    * comb(remaining, count)
                    * weight_powers[category - 1][count]
                )
                next_paths[cumulative] = next_paths.get(cumulative, 0) + contribution
        paths = next_paths
    return paths.get(sample_size, 0)


def _tail_probability(
    *,
    sample_size: int,
    category_count: int,
    bound: Fraction,
    weights: tuple[int, ...],
    inclusive_acceptance: bool,
) -> Fraction:
    normalized = _normalized_weights(weights, category_count)
    accepted = _bounded_sequence_weight(
        sample_size,
        category_count,
        bound.numerator,
        bound.denominator,
        normalized,
        inclusive=inclusive_acceptance,
    )
    total = sum(normalized) ** sample_size
    return Fraction(total - accepted, total)


def exact_discrete_rank_tail_probability(
    histogram: tuple[int, ...],
) -> ExactRational:
    """Return exact ``P_null(D >= D_observed)`` for uniform discrete ranks."""
    normalized = _validate_histogram(histogram)
    discrepancy = exact_rank_cdf_discrepancy(
        normalized,
        posterior_draw_count=len(normalized) - 1,
    )
    tail = _tail_probability(
        sample_size=sum(normalized),
        category_count=len(normalized),
        bound=_fraction(discrepancy, "observed discrepancy"),
        weights=tuple(1 for _ in normalized),
        inclusive_acceptance=False,
    )
    return _from_fraction(tail)


@lru_cache(maxsize=256)
def _critical_components(
    sample_size: int,
    category_count: int,
    alpha_numerator: int,
    alpha_denominator: int,
) -> tuple[Fraction, Fraction, Fraction | None, Fraction | None]:
    alpha = Fraction(alpha_numerator, alpha_denominator)
    candidates = sorted(
        {
            abs(Fraction(cumulative, sample_size) - Fraction(category, category_count))
            for cumulative in range(sample_size + 1)
            for category in range(1, category_count + 1)
        }
    )
    uniform = tuple(1 for _ in range(category_count))

    def rejection_size(candidate: Fraction) -> Fraction:
        return _tail_probability(
            sample_size=sample_size,
            category_count=category_count,
            bound=candidate,
            weights=uniform,
            inclusive_acceptance=True,
        )

    lower = 0
    upper = len(candidates) - 1
    while lower < upper:
        midpoint = (lower + upper) // 2
        if rejection_size(candidates[midpoint]) <= alpha:
            upper = midpoint
        else:
            lower = midpoint + 1
    critical = candidates[lower]
    achieved = rejection_size(critical)
    previous = None
    previous_size = None
    for index in range(lower - 1, -1, -1):
        candidate = candidates[index]
        candidate_size = rejection_size(candidate)
        size_before_candidate = (
            Fraction(1) if index == 0 else rejection_size(candidates[index - 1])
        )
        if size_before_candidate > candidate_size:
            previous = candidate
            previous_size = candidate_size
            break
    return critical, achieved, previous, previous_size


@dataclass(frozen=True, slots=True)
class ExactReachableCriticalRule:
    """Minimal reachable strict critical rule with exact null size."""

    sample_size: int
    category_count: int
    alpha: ExactRational
    critical_discrepancy: ExactRational
    achieved_size: ExactRational
    previous_reachable_discrepancy: ExactRational | None
    previous_reachable_size: ExactRational | None
    rejection_rule: str = POLICY_FUNCTIONAL_ORDER_WITNESS_REJECTION_RULE

    def __post_init__(self) -> None:
        self._fresh_validate()

    def _fresh_validate(self) -> None:
        sample_size, category_count = _validate_dimensions(
            self.sample_size,
            self.category_count,
        )
        alpha = _probability(self.alpha, "alpha")
        critical, achieved, previous, previous_size = _critical_components(
            sample_size,
            category_count,
            alpha.numerator,
            alpha.denominator,
        )
        actual = (
            _fraction(self.critical_discrepancy, "critical_discrepancy"),
            _fraction(self.achieved_size, "achieved_size"),
            (
                None
                if self.previous_reachable_discrepancy is None
                else _fraction(
                    self.previous_reachable_discrepancy,
                    "previous_reachable_discrepancy",
                )
            ),
            (
                None
                if self.previous_reachable_size is None
                else _fraction(
                    self.previous_reachable_size,
                    "previous_reachable_size",
                )
            ),
        )
        if actual != (critical, achieved, previous, previous_size):
            _fail("reachable critical-rule fields do not exactly rederive")
        if achieved > _fraction(alpha, "alpha"):
            _fail("reachable critical rule exceeds its target size")
        if previous_size is not None and previous_size <= _fraction(alpha, "alpha"):
            _fail("reachable critical rule is not the minimal admissible rule")
        if self.rejection_rule != POLICY_FUNCTIONAL_ORDER_WITNESS_REJECTION_RULE:
            _fail("rank rejection rule changed")

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def to_mapping(self) -> dict[str, object]:
        self._fresh_validate()
        return {
            "achieved_size": self.achieved_size.to_mapping(),
            "alpha": self.alpha.to_mapping(),
            "category_count": self.category_count,
            "critical_discrepancy": self.critical_discrepancy.to_mapping(),
            "previous_reachable_discrepancy": (
                None
                if self.previous_reachable_discrepancy is None
                else self.previous_reachable_discrepancy.to_mapping()
            ),
            "previous_reachable_size": (
                None
                if self.previous_reachable_size is None
                else self.previous_reachable_size.to_mapping()
            ),
            "rejection_rule": self.rejection_rule,
            "sample_size": self.sample_size,
        }


def derive_exact_reachable_critical_rule(
    *,
    sample_size: int,
    category_count: int,
    alpha: ExactRational,
) -> ExactReachableCriticalRule:
    """Derive the least conservative reachable rule satisfying ``size <= alpha``."""
    sample, categories = _validate_dimensions(sample_size, category_count)
    target = _probability(alpha, "alpha")
    critical, achieved, previous, previous_size = _critical_components(
        sample,
        categories,
        target.numerator,
        target.denominator,
    )
    return ExactReachableCriticalRule(
        sample_size=sample,
        category_count=categories,
        alpha=target,
        critical_discrepancy=_from_fraction(critical),
        achieved_size=_from_fraction(achieved),
        previous_reachable_discrepancy=(
            None if previous is None else _from_fraction(previous)
        ),
        previous_reachable_size=(
            None if previous_size is None else _from_fraction(previous_size)
        ),
    )


class RankDistortionKind(StrEnum):
    """Closed in-code-frozen posterior-quantile rank distortions."""

    BETA_HALF_ONE_DIRECTIONAL = "beta_half_one_directional"
    BETA_ONE_HALF_DIRECTIONAL = "beta_one_half_directional"
    EDGE_RANK_CONCENTRATION = "edge_rank_concentration"
    CENTER_RANK_CONCENTRATION = "center_rank_concentration"


def _fractions_to_weights(probabilities: tuple[Fraction, ...]) -> tuple[int, ...]:
    denominator = 1
    for probability in probabilities:
        denominator = lcm(denominator, probability.denominator)
    weights = tuple(
        probability.numerator * (denominator // probability.denominator)
        for probability in probabilities
    )
    divisor = 0
    for weight in weights:
        divisor = gcd(divisor, weight)
    return tuple(weight // divisor for weight in weights)


def _distortion_weights(
    kind: RankDistortionKind,
    category_count: int,
) -> tuple[int, ...]:
    posterior_draw_count = category_count - 1
    values: tuple[Fraction, ...]
    if kind in {
        RankDistortionKind.BETA_HALF_ONE_DIRECTIONAL,
        RankDistortionKind.BETA_ONE_HALF_DIRECTIONAL,
    }:
        probabilities: list[Fraction] = []
        for rank in range(category_count):
            integral = sum(
                (
                    Fraction(
                        ((-1) ** term) * comb(posterior_draw_count - rank, term),
                        2 * (rank + term) + 1,
                    )
                    for term in range(posterior_draw_count - rank + 1)
                ),
                Fraction(),
            )
            probabilities.append(comb(posterior_draw_count, rank) * integral)
        values = tuple(probabilities)
        if kind is RankDistortionKind.BETA_ONE_HALF_DIRECTIONAL:
            values = tuple(reversed(values))
    elif kind is RankDistortionKind.EDGE_RANK_CONCENTRATION:
        interior = Fraction(1, 2 * category_count)
        values = tuple(
            interior
            + (Fraction(1, 4) if rank in {0, category_count - 1} else Fraction())
            for rank in range(category_count)
        )
    elif kind is RankDistortionKind.CENTER_RANK_CONCENTRATION:
        values = tuple(
            Fraction(1, 2 * category_count)
            + Fraction(
                comb(posterior_draw_count, rank),
                1 << (posterior_draw_count + 1),
            )
            for rank in range(category_count)
        )
    else:  # pragma: no cover - the closed enum makes this unreachable
        _fail("unsupported rank-distortion kind")
    if any(value <= 0 for value in values) or sum(values, Fraction()) != Fraction(1):
        _fail("frozen rank distortion does not define a probability mass")
    return _fractions_to_weights(values)


@dataclass(frozen=True, slots=True)
class ExactRankDistortionPower:
    """Exact power for one closed rank-PMF distortion and critical rule."""

    kind: RankDistortionKind
    weights: tuple[int, ...]
    exact_power: ExactRational
    minimum_power: ExactRational
    target_met: bool
    critical_rule: ExactReachableCriticalRule

    def __post_init__(self) -> None:
        self._fresh_validate()

    def _fresh_validate(self) -> None:
        if not isinstance(self.kind, RankDistortionKind):
            _fail("rank-distortion kind has the wrong type")
        if type(self.critical_rule) is not ExactReachableCriticalRule:
            raise TypeError("critical_rule must be an ExactReachableCriticalRule")
        self.critical_rule._fresh_validate()
        expected_weights = _distortion_weights(
            self.kind,
            self.critical_rule.category_count,
        )
        weights = _normalized_weights(self.weights, len(expected_weights))
        if weights != expected_weights:
            _fail("rank-distortion weights do not match the frozen kind")
        target = _probability(self.minimum_power, "minimum_power")
        expected_power = _tail_probability(
            sample_size=self.critical_rule.sample_size,
            category_count=self.critical_rule.category_count,
            bound=_fraction(self.critical_rule.critical_discrepancy),
            weights=weights,
            inclusive_acceptance=True,
        )
        if _fraction(
            self.exact_power, "exact_power"
        ) != expected_power or self.target_met is not (
            expected_power >= _fraction(target)
        ):
            _fail("rank-distortion power fields do not exactly rederive")

    def to_mapping(self) -> dict[str, object]:
        self._fresh_validate()
        return {
            "critical_rule_sha256": self.critical_rule.sha256,
            "exact_power": self.exact_power.to_mapping(),
            "kind": self.kind.value,
            "minimum_power": self.minimum_power.to_mapping(),
            "target_met": self.target_met,
            "weights": list(self.weights),
            "weights_total": sum(self.weights),
        }


def derive_frozen_rank_distortion_powers(
    *,
    critical_rule: ExactReachableCriticalRule,
    minimum_power: ExactRational,
) -> tuple[ExactRankDistortionPower, ...]:
    """Derive exact power for every closed in-code-frozen rank distortion."""
    if type(critical_rule) is not ExactReachableCriticalRule:
        raise TypeError("critical_rule must be an ExactReachableCriticalRule")
    critical_rule._fresh_validate()
    target = _probability(minimum_power, "minimum_power")
    critical = _fraction(critical_rule.critical_discrepancy)
    powers = []
    for kind in RankDistortionKind:
        weights = _distortion_weights(kind, critical_rule.category_count)
        power = _tail_probability(
            sample_size=critical_rule.sample_size,
            category_count=critical_rule.category_count,
            bound=critical,
            weights=weights,
            inclusive_acceptance=True,
        )
        powers.append(
            ExactRankDistortionPower(
                kind=kind,
                weights=weights,
                exact_power=_from_fraction(power),
                minimum_power=target,
                target_met=power >= _fraction(target),
                critical_rule=critical_rule,
            )
        )
    return tuple(powers)


@dataclass(frozen=True, slots=True, init=False)
class PolicyFunctionalOrderWitnessConfig:
    """Factory-only immutable finite-lattice reference design."""

    reference_id: str
    seed: str
    tie_seed: str
    meta_replicate_count: int
    posterior_draw_count: int
    alpha: ExactRational
    minimum_frozen_rank_law_power: ExactRational
    witness_bits: Literal[256]
    functional: str
    rng_implementation: str
    schema_version: str

    def __init__(self) -> None:
        raise TypeError(
            "PolicyFunctionalOrderWitnessConfig is factory-produced; use "
            "build_policy_functional_order_witness_config"
        )

    def __copy__(self) -> NoReturn:
        raise TypeError("order-witness configs cannot be copied")

    def __deepcopy__(self, memo: object) -> NoReturn:
        del memo
        raise TypeError("order-witness configs cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("order-witness configs cannot be serialized")

    def _fresh_validate(self) -> None:
        _identifier(self.reference_id, "reference_id")
        seed = _seed(self.seed, "seed")
        tie_seed = _seed(self.tie_seed, "tie_seed")
        if seed == tie_seed:
            _fail("witness and tie seeds must be distinct")
        sample, categories = _validate_dimensions(
            self.meta_replicate_count,
            self.posterior_draw_count + 1,
        )
        if sample < 8:
            _fail("meta_replicate_count must be at least 8")
        if sample != self.meta_replicate_count or categories != (
            self.posterior_draw_count + 1
        ):
            _fail("order-witness dimensions changed")
        _probability(self.alpha, "alpha")
        _probability(
            self.minimum_frozen_rank_law_power,
            "minimum_frozen_rank_law_power",
        )
        if (
            self.witness_bits != _WITNESS_BITS
            or self.functional != POLICY_FUNCTIONAL_ORDER_WITNESS_FUNCTIONAL
            or self.rng_implementation
            != POLICY_FUNCTIONAL_ORDER_WITNESS_RNG_IMPLEMENTATION
            or self.schema_version != POLICY_FUNCTIONAL_ORDER_WITNESS_SCHEMA_VERSION
        ):
            _fail("order-witness schema or scientific semantics changed")

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    @property
    def generation_design_sha256(self) -> str:
        """Bind witness generation without thresholds, power, or tie settings."""
        self._fresh_validate()
        return sha256_bytes(
            canonical_json_bytes(
                {
                    "functional": self.functional,
                    "reference_id": self.reference_id,
                    "rng_implementation": self.rng_implementation,
                    "schema_version": (
                        POLICY_FUNCTIONAL_ORDER_WITNESS_GENERATION_SCHEMA_VERSION
                    ),
                    "witness_bits": self.witness_bits,
                }
            )
        )

    @property
    def tie_design_sha256(self) -> str:
        """Bind only the rank and tie semantics, never the test threshold."""
        self._fresh_validate()
        return sha256_bytes(
            canonical_json_bytes(
                {
                    "generation_design_sha256": self.generation_design_sha256,
                    "posterior_draw_count": self.posterior_draw_count,
                    "rng_implementation": self.rng_implementation,
                    "tie_rule": "uniform_integer_over_exact_tie_positions",
                }
            )
        )

    def to_mapping(self) -> dict[str, object]:
        self._fresh_validate()
        return {
            "alpha": self.alpha.to_mapping(),
            "functional": self.functional,
            "generation_schema_version": (
                POLICY_FUNCTIONAL_ORDER_WITNESS_GENERATION_SCHEMA_VERSION
            ),
            "generation_design_sha256": self.generation_design_sha256,
            "meta_replicate_count": self.meta_replicate_count,
            "minimum_frozen_rank_law_power": (
                self.minimum_frozen_rank_law_power.to_mapping()
            ),
            "posterior_draw_count": self.posterior_draw_count,
            "reference_id": self.reference_id,
            "rng_implementation": self.rng_implementation,
            "schema_version": self.schema_version,
            "seed": self.seed,
            "seed_sha256": sha256_bytes(self.seed.encode("utf-8")),
            "tie_seed": self.tie_seed,
            "tie_seed_sha256": sha256_bytes(self.tie_seed.encode("utf-8")),
            "tie_design_sha256": self.tie_design_sha256,
            "witness_bits": self.witness_bits,
        }


def _mint_config(
    *,
    reference_id: str,
    seed: str,
    tie_seed: str,
    meta_replicate_count: int,
    posterior_draw_count: int,
    alpha: ExactRational,
    minimum_frozen_rank_law_power: ExactRational,
) -> PolicyFunctionalOrderWitnessConfig:
    config = object.__new__(PolicyFunctionalOrderWitnessConfig)
    values: tuple[tuple[str, object], ...] = (
        ("reference_id", reference_id),
        ("seed", seed),
        ("tie_seed", tie_seed),
        ("meta_replicate_count", meta_replicate_count),
        ("posterior_draw_count", posterior_draw_count),
        ("alpha", alpha),
        ("minimum_frozen_rank_law_power", minimum_frozen_rank_law_power),
        ("witness_bits", _WITNESS_BITS),
        ("functional", POLICY_FUNCTIONAL_ORDER_WITNESS_FUNCTIONAL),
        ("rng_implementation", POLICY_FUNCTIONAL_ORDER_WITNESS_RNG_IMPLEMENTATION),
        ("schema_version", POLICY_FUNCTIONAL_ORDER_WITNESS_SCHEMA_VERSION),
    )
    for field, value in values:
        object.__setattr__(config, field, value)
    config._fresh_validate()
    return config


def build_policy_functional_order_witness_config(
    *,
    reference_id: str = "policy-functional-order-witness-v1",
    seed: str = "policy-functional-order-witness-grid-v1",
    tie_seed: str = "policy-functional-order-witness-ties-v1",
    meta_replicate_count: int = 192,
    posterior_draw_count: int = 7,
    alpha: ExactRational | None = None,
    minimum_frozen_rank_law_power: ExactRational | None = None,
) -> PolicyFunctionalOrderWitnessConfig:
    """Build the bounded reference design before deriving any witness outcome."""
    return _mint_config(
        reference_id=reference_id,
        seed=seed,
        tie_seed=tie_seed,
        meta_replicate_count=meta_replicate_count,
        posterior_draw_count=posterior_draw_count,
        alpha=ExactRational.from_ratio(1, 20) if alpha is None else alpha,
        minimum_frozen_rank_law_power=(
            ExactRational.from_ratio(9, 10)
            if minimum_frozen_rank_law_power is None
            else minimum_frozen_rank_law_power
        ),
    )


@dataclass(frozen=True, slots=True)
class OrderWitnessRankRecord:
    """One exact finite-lattice oracle rank and all replay ingredients."""

    replicate_index: int
    oracle_witness: ExactRational
    posterior_witnesses: tuple[ExactRational, ...]
    oracle_event_key: str
    posterior_event_keys: tuple[str, ...]
    tie_event_key: str
    less_count: int
    equal_count: int
    tie_draw: int
    rank: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.replicate_index, bool)
            or not isinstance(self.replicate_index, int)
            or self.replicate_index < 0
        ):
            _fail("replicate_index must be a non-negative integer")
        oracle = _exact(self.oracle_witness, "oracle_witness")
        if oracle.denominator != _WITNESS_DENOMINATOR or oracle.numerator % 2 != 1:
            _fail("oracle witness is not a canonical 256-bit midpoint")
        if (
            not isinstance(self.posterior_witnesses, tuple)
            or not self.posterior_witnesses
        ):
            _fail("posterior_witnesses must be a non-empty immutable tuple")
        for index, witness in enumerate(self.posterior_witnesses):
            exact = _exact(witness, f"posterior_witnesses[{index}]")
            if exact.denominator != _WITNESS_DENOMINATOR or exact.numerator % 2 != 1:
                _fail("posterior witness is not a canonical 256-bit midpoint")
        _event_key(self.oracle_event_key, "oracle_event_key")
        if not isinstance(self.posterior_event_keys, tuple) or len(
            self.posterior_event_keys
        ) != len(self.posterior_witnesses):
            _fail("posterior EventKeys do not align with posterior witnesses")
        for index, value in enumerate(self.posterior_event_keys):
            _event_key(value, f"posterior_event_keys[{index}]")
        _event_key(self.tie_event_key, "tie_event_key")
        keys = (self.oracle_event_key, *self.posterior_event_keys, self.tie_event_key)
        if len(set(keys)) != len(keys):
            _fail("one rank record reused an EventKey identity")
        oracle_fraction = _fraction(oracle)
        posterior = tuple(_fraction(value) for value in self.posterior_witnesses)
        less = sum(value < oracle_fraction for value in posterior)
        equal = sum(value == oracle_fraction for value in posterior)
        if self.less_count != less or self.equal_count != equal:
            _fail("order-witness comparison counts do not rederive")
        if (
            isinstance(self.tie_draw, bool)
            or not isinstance(self.tie_draw, int)
            or not 0 <= self.tie_draw <= equal
            or self.rank != less + self.tie_draw
        ):
            _fail("order-witness randomized rank is invalid")

    def to_mapping(self) -> dict[str, object]:
        return {
            "equal_count": self.equal_count,
            "less_count": self.less_count,
            "oracle_event_key": self.oracle_event_key,
            "oracle_witness": self.oracle_witness.to_mapping(),
            "posterior_event_keys": list(self.posterior_event_keys),
            "posterior_witnesses": [
                value.to_mapping() for value in self.posterior_witnesses
            ],
            "rank": self.rank,
            "replicate_index": self.replicate_index,
            "tie_draw": self.tie_draw,
            "tie_event_key": self.tie_event_key,
        }


def _witness_key(
    config: PolicyFunctionalOrderWitnessConfig,
    *,
    replicate_index: int,
    role: str,
    draw_index: int,
) -> EventKey:
    return EventKey(
        experiment_id=config.reference_id,
        seed=config.seed,
        replicate_id=replicate_index,
        simulation_step=draw_index,
        purpose=f"order-witness.{role}.{config.generation_design_sha256}",
    )


def _tie_key(
    config: PolicyFunctionalOrderWitnessConfig,
    *,
    replicate_index: int,
) -> EventKey:
    return EventKey(
        experiment_id=config.reference_id,
        seed=config.tie_seed,
        replicate_id=replicate_index,
        simulation_step=0,
        purpose=f"order-witness.randomized-tie.{config.tie_design_sha256}",
    )


def _midpoint(key: EventKey) -> ExactRational:
    integer = deterministic_int(key, 0, (1 << _WITNESS_BITS) - 1)
    return ExactRational.from_ratio(2 * integer + 1, _WITNESS_DENOMINATOR)


def _derive_records(
    config: PolicyFunctionalOrderWitnessConfig,
) -> tuple[tuple[OrderWitnessRankRecord, ...], tuple[str, ...]]:
    config._fresh_validate()
    records = []
    event_keys: list[str] = []
    for replicate in range(config.meta_replicate_count):
        oracle_key = _witness_key(
            config,
            replicate_index=replicate,
            role="oracle",
            draw_index=0,
        )
        posterior_keys = tuple(
            _witness_key(
                config,
                replicate_index=replicate,
                role="posterior",
                draw_index=index,
            )
            for index in range(config.posterior_draw_count)
        )
        tie_key = _tie_key(config, replicate_index=replicate)
        oracle = _midpoint(oracle_key)
        posterior = tuple(_midpoint(key) for key in posterior_keys)
        oracle_fraction = _fraction(oracle)
        posterior_fractions = tuple(_fraction(value) for value in posterior)
        less = sum(value < oracle_fraction for value in posterior_fractions)
        equal = sum(value == oracle_fraction for value in posterior_fractions)
        tie_draw = deterministic_int(tie_key, 0, equal)
        records.append(
            OrderWitnessRankRecord(
                replicate_index=replicate,
                oracle_witness=oracle,
                posterior_witnesses=posterior,
                oracle_event_key=oracle_key.value,
                posterior_event_keys=tuple(key.value for key in posterior_keys),
                tie_event_key=tie_key.value,
                less_count=less,
                equal_count=equal,
                tie_draw=tie_draw,
                rank=less + tie_draw,
            )
        )
        event_keys.extend(
            (oracle_key.value, *(key.value for key in posterior_keys), tie_key.value)
        )
    if len(set(event_keys)) != len(event_keys):
        _fail("order-witness reference reused an EventKey identity")
    return tuple(records), tuple(event_keys)


@dataclass(frozen=True, slots=True, init=False)
class PolicyFunctionalOrderWitnessResult:
    """Factory-only fully rederived finite-lattice rank reference."""

    config: PolicyFunctionalOrderWitnessConfig
    records: tuple[OrderWitnessRankRecord, ...]
    rank_histogram: tuple[int, ...]
    rank_cdf_discrepancy: ExactRational
    exact_tail_p_value: ExactRational
    observed_rejected: bool
    critical_rule: ExactReachableCriticalRule
    distortion_powers: tuple[ExactRankDistortionPower, ...]
    event_keys_sha256: str
    event_key_count: int
    exact_tie_count: int
    frozen_rank_distortion_power_target_met: bool
    finite_lattice_exchangeability_derived_under_declared_iid_law: Literal[True]
    fixed_hash_grid_statistical_independence_verified: Literal[False]
    theoretical_null_arithmetic: Literal[True]
    exact_reachable_type_i_size_rederived: Literal[True]
    exact_frozen_rank_law_power_rederived: Literal[True]
    exact_rank_pmf_power_only: Literal[True]
    estimator_canary_power_verified: Literal[False]
    order_witness_not_effect_magnitude: Literal[True]
    continuous_normal_exact: Literal[False]
    integrated_abc_posterior_exchangeability_verified: Literal[False]
    integrated_market_policy_validity_verified: Literal[False]
    external_preregistration_verified: Literal[False]
    independent_reproduction: Literal[False]
    real_world_coverage_claim: Literal[False]
    real_world_causal_claim: Literal[False]
    publication_eligible: Literal[False]
    scientific_abstention_required: Literal[True]
    claim_scope: str
    schema_version: str

    def __init__(self) -> None:
        raise TypeError(
            "PolicyFunctionalOrderWitnessResult is factory-produced; use "
            "run_policy_functional_order_witness"
        )

    def __copy__(self) -> NoReturn:
        raise TypeError("order-witness results cannot be copied")

    def __deepcopy__(self, memo: object) -> NoReturn:
        del memo
        raise TypeError("order-witness results cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("order-witness results cannot be serialized")

    def _fresh_validate(self) -> None:
        if type(self.config) is not PolicyFunctionalOrderWitnessConfig:
            raise TypeError("result config lost its exact factory-produced type")
        components = _derive_components(self.config)
        actual = (
            self.records,
            self.rank_histogram,
            self.rank_cdf_discrepancy,
            self.exact_tail_p_value,
            self.observed_rejected,
            self.critical_rule,
            self.distortion_powers,
            self.event_keys_sha256,
            self.event_key_count,
            self.exact_tie_count,
            self.frozen_rank_distortion_power_target_met,
        )
        if actual != components:
            _fail("order-witness result fields do not freshly rederive")
        if (
            self.finite_lattice_exchangeability_derived_under_declared_iid_law
            is not True
            or self.fixed_hash_grid_statistical_independence_verified is not False
            or self.theoretical_null_arithmetic is not True
            or self.exact_reachable_type_i_size_rederived is not True
            or self.exact_frozen_rank_law_power_rederived is not True
            or self.exact_rank_pmf_power_only is not True
            or self.estimator_canary_power_verified is not False
            or self.order_witness_not_effect_magnitude is not True
            or self.continuous_normal_exact is not False
            or self.integrated_abc_posterior_exchangeability_verified is not False
            or self.integrated_market_policy_validity_verified is not False
            or self.external_preregistration_verified is not False
            or self.independent_reproduction is not False
            or self.real_world_coverage_claim is not False
            or self.real_world_causal_claim is not False
            or self.publication_eligible is not False
            or self.scientific_abstention_required is not True
            or self.claim_scope != POLICY_FUNCTIONAL_ORDER_WITNESS_CLAIM_SCOPE
            or self.schema_version != POLICY_FUNCTIONAL_ORDER_WITNESS_SCHEMA_VERSION
        ):
            _fail("order-witness scientific claim boundary changed")

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_mapping())

    @property
    def sha256(self) -> str:
        return sha256_bytes(self.canonical_bytes)

    def to_mapping(self) -> dict[str, object]:
        self._fresh_validate()
        return {
            "claim_scope": self.claim_scope,
            "config": self.config.to_mapping(),
            "config_sha256": self.config.sha256,
            "continuous_normal_exact": self.continuous_normal_exact,
            "estimator_canary_power_verified": self.estimator_canary_power_verified,
            "fixed_hash_grid_statistical_independence_verified": (
                self.fixed_hash_grid_statistical_independence_verified
            ),
            "frozen_rank_distortion_power_target_met": (
                self.frozen_rank_distortion_power_target_met
            ),
            "distortion_powers": [
                value.to_mapping() for value in self.distortion_powers
            ],
            "event_key_count": self.event_key_count,
            "event_keys_sha256": self.event_keys_sha256,
            "exact_frozen_rank_law_power_rederived": (
                self.exact_frozen_rank_law_power_rederived
            ),
            "exact_rank_pmf_power_only": self.exact_rank_pmf_power_only,
            "exact_reachable_type_i_size_rederived": (
                self.exact_reachable_type_i_size_rederived
            ),
            "exact_tail_p_value": self.exact_tail_p_value.to_mapping(),
            "exact_tie_count": self.exact_tie_count,
            "external_preregistration_verified": (
                self.external_preregistration_verified
            ),
            "finite_lattice_exchangeability_derived_under_declared_iid_law": (
                self.finite_lattice_exchangeability_derived_under_declared_iid_law
            ),
            "independent_reproduction": self.independent_reproduction,
            "integrated_abc_posterior_exchangeability_verified": (
                self.integrated_abc_posterior_exchangeability_verified
            ),
            "integrated_market_policy_validity_verified": (
                self.integrated_market_policy_validity_verified
            ),
            "observed_rejected": self.observed_rejected,
            "order_witness_not_effect_magnitude": (
                self.order_witness_not_effect_magnitude
            ),
            "publication_eligible": self.publication_eligible,
            "rank_cdf_discrepancy": self.rank_cdf_discrepancy.to_mapping(),
            "rank_histogram": list(self.rank_histogram),
            "real_world_causal_claim": self.real_world_causal_claim,
            "real_world_coverage_claim": self.real_world_coverage_claim,
            "records": [value.to_mapping() for value in self.records],
            "schema_version": self.schema_version,
            "scientific_abstention_required": self.scientific_abstention_required,
            "theoretical_null_arithmetic": self.theoretical_null_arithmetic,
            "critical_rule": self.critical_rule.to_mapping(),
        }


def _derive_components(
    config: PolicyFunctionalOrderWitnessConfig,
) -> tuple[
    tuple[OrderWitnessRankRecord, ...],
    tuple[int, ...],
    ExactRational,
    ExactRational,
    bool,
    ExactReachableCriticalRule,
    tuple[ExactRankDistortionPower, ...],
    str,
    int,
    int,
    bool,
]:
    records, event_keys = _derive_records(config)
    histogram = tuple(
        sum(record.rank == rank for record in records)
        for rank in range(config.posterior_draw_count + 1)
    )
    discrepancy = exact_rank_cdf_discrepancy(
        histogram,
        posterior_draw_count=config.posterior_draw_count,
    )
    tail = exact_discrete_rank_tail_probability(histogram)
    critical = derive_exact_reachable_critical_rule(
        sample_size=config.meta_replicate_count,
        category_count=config.posterior_draw_count + 1,
        alpha=config.alpha,
    )
    powers = derive_frozen_rank_distortion_powers(
        critical_rule=critical,
        minimum_power=config.minimum_frozen_rank_law_power,
    )
    rejected = _fraction(discrepancy) > _fraction(
        critical.critical_discrepancy,
        "critical_discrepancy",
    )
    return (
        records,
        histogram,
        discrepancy,
        tail,
        rejected,
        critical,
        powers,
        sha256_bytes(canonical_json_bytes(list(event_keys))),
        len(event_keys),
        sum(record.equal_count for record in records),
        all(power.target_met for power in powers),
    )


def run_policy_functional_order_witness(
    config: PolicyFunctionalOrderWitnessConfig,
) -> PolicyFunctionalOrderWitnessResult:
    """Run and fully rederive the exact finite-lattice order-witness reference."""
    if type(config) is not PolicyFunctionalOrderWitnessConfig:
        raise TypeError("config must be a PolicyFunctionalOrderWitnessConfig")
    config._fresh_validate()
    components = _derive_components(config)
    result = object.__new__(PolicyFunctionalOrderWitnessResult)
    values: tuple[tuple[str, object], ...] = (
        ("config", config),
        ("records", components[0]),
        ("rank_histogram", components[1]),
        ("rank_cdf_discrepancy", components[2]),
        ("exact_tail_p_value", components[3]),
        ("observed_rejected", components[4]),
        ("critical_rule", components[5]),
        ("distortion_powers", components[6]),
        ("event_keys_sha256", components[7]),
        ("event_key_count", components[8]),
        ("exact_tie_count", components[9]),
        ("frozen_rank_distortion_power_target_met", components[10]),
        ("finite_lattice_exchangeability_derived_under_declared_iid_law", True),
        ("fixed_hash_grid_statistical_independence_verified", False),
        ("theoretical_null_arithmetic", True),
        ("exact_reachable_type_i_size_rederived", True),
        ("exact_frozen_rank_law_power_rederived", True),
        ("exact_rank_pmf_power_only", True),
        ("estimator_canary_power_verified", False),
        ("order_witness_not_effect_magnitude", True),
        ("continuous_normal_exact", False),
        ("integrated_abc_posterior_exchangeability_verified", False),
        ("integrated_market_policy_validity_verified", False),
        ("external_preregistration_verified", False),
        ("independent_reproduction", False),
        ("real_world_coverage_claim", False),
        ("real_world_causal_claim", False),
        ("publication_eligible", False),
        ("scientific_abstention_required", True),
        ("claim_scope", POLICY_FUNCTIONAL_ORDER_WITNESS_CLAIM_SCOPE),
        ("schema_version", POLICY_FUNCTIONAL_ORDER_WITNESS_SCHEMA_VERSION),
    )
    for field, value in values:
        object.__setattr__(result, field, value)
    result._fresh_validate()
    return result


__all__ = [
    "POLICY_FUNCTIONAL_ORDER_WITNESS_CLAIM_SCOPE",
    "POLICY_FUNCTIONAL_ORDER_WITNESS_FUNCTIONAL",
    "POLICY_FUNCTIONAL_ORDER_WITNESS_GENERATION_SCHEMA_VERSION",
    "POLICY_FUNCTIONAL_ORDER_WITNESS_REJECTION_RULE",
    "POLICY_FUNCTIONAL_ORDER_WITNESS_RNG_IMPLEMENTATION",
    "POLICY_FUNCTIONAL_ORDER_WITNESS_SCHEMA_VERSION",
    "ExactRankDistortionPower",
    "ExactReachableCriticalRule",
    "OrderWitnessRankRecord",
    "PolicyFunctionalOrderWitnessConfig",
    "PolicyFunctionalOrderWitnessError",
    "PolicyFunctionalOrderWitnessResult",
    "RankDistortionKind",
    "build_policy_functional_order_witness_config",
    "derive_exact_reachable_critical_rule",
    "derive_frozen_rank_distortion_powers",
    "exact_discrete_rank_tail_probability",
    "run_policy_functional_order_witness",
]
