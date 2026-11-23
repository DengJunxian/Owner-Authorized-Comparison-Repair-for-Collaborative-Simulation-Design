"""Binary64 analytic-conjugate regression for policy-functional ranks.

This module deliberately does not validate the integrated ABC posterior or a
real market.  It uses a normal--normal model with a closed-form posterior to
exercise the rank-null implementation, an exact finite-sample discrete-uniform
tail calculation, and frozen fixed-key synthetic alternatives.  A data-conditioned
functional exposes an ignored-data failure that a parameter-only rank can miss.

All operating characteristics concern one fixed, deterministic synthetic
EventKey grid generated with binary64 transforms.  The analytic posterior
formula is closed form, but the floating-point sampler is not an exact
continuous-normal construction.  These results are not external preregistration,
type-I-error or power theorems, real coverage, policy validity, or publication
evidence.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from fractions import Fraction
from functools import lru_cache
from math import comb
from typing import Literal, NoReturn, Self

from policytwin.counterfactual.events import (
    HASH_BASED_RNG_IMPLEMENTATION_ID,
    EventType,
    HashBasedRNG,
    SemanticEventKey,
)
from policytwin.evaluation.ledger_derived import ExactRational
from policytwin.evaluation.ledger_policy_functional_sbc import (
    exact_rank_cdf_discrepancy,
)
from policytwin.provenance import canonical_json_bytes, sha256_bytes

POLICY_FUNCTIONAL_REFERENCE_SCHEMA_VERSION = (
    "policytwin.policy-functional-conjugate-reference.v3"
)
POLICY_FUNCTIONAL_REFERENCE_CLAIM_SCOPE: Literal[
    "fixed_key_binary64_conjugate_synthetic_rank_regression_only"
] = "fixed_key_binary64_conjugate_synthetic_rank_regression_only"

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_MAX_META_REPLICATES = 512
_MAX_RANK_CATEGORIES = 32
_MAX_OPERATING_TRIALS = 512
_MAX_OBSERVATION_COUNT = 256
_MAX_SCENARIO_REPLICATE_COUNT = 256
_MAX_LOGICAL_NORMAL_DRAWS = 1_000_000
_REFERENCE_EVENT_EPOCH = datetime(2000, 1, 1, tzinfo=UTC)


class PolicyFunctionalReferenceError(ValueError):
    """Raised when the analytic reference contract is invalid."""


def _fail(message: str) -> NoReturn:
    raise PolicyFunctionalReferenceError(message)


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        _fail(f"{field} must be a stable ASCII identifier")
    return value


def _positive_int(value: object, field: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"{field} must be an integer at least {minimum}")
    return value


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        _fail(f"{field} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{field} must be finite")
    return result


def _aware(value: object, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        _fail(f"{field} must be timezone-aware")
    return value.astimezone(UTC)


def _fraction(value: ExactRational) -> Fraction:
    if type(value) is not ExactRational:
        raise TypeError("exact probability must be an ExactRational")
    return Fraction(value.numerator, value.denominator)


def _probability(value: ExactRational, field: str) -> ExactRational:
    fraction = _fraction(value)
    if not Fraction() < fraction < Fraction(1):
        _fail(f"{field} must lie strictly between zero and one")
    return value


def _digest(value: object) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        _fail(f"{field} must be an object with string keys")
    return value


def _exact_from_mapping(value: object, field: str) -> ExactRational:
    mapping = _mapping(value, field)
    if frozenset(mapping) != frozenset({"numerator", "denominator"}):
        _fail(f"{field} has missing or unknown fields")
    numerator = mapping["numerator"]
    denominator = mapping["denominator"]
    if (
        isinstance(numerator, bool)
        or not isinstance(numerator, int)
        or isinstance(denominator, bool)
        or not isinstance(denominator, int)
    ):
        _fail(f"{field} numerator and denominator must be integers")
    return ExactRational(numerator=numerator, denominator=denominator)


def _datetime_from_mapping(value: object, field: str) -> datetime:
    if not isinstance(value, str):
        _fail(f"{field} must be an ISO 8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise PolicyFunctionalReferenceError(
            f"{field} must be an ISO 8601 timestamp"
        ) from error
    return _aware(parsed, field)


class ReferenceFunctional(StrEnum):
    """Closed generalized test quantities used by the reference."""

    PARAMETER_ONLY = "parameter_only"
    PRE_POLICY_DATA_AND_SCENARIO_CONDITIONED = (
        "pre_policy_data_and_scenario_conditioned"
    )


class ReferencePosteriorMode(StrEnum):
    """Closed binary64 analytic and controlled synthetic implementations."""

    ANALYTIC_CONJUGATE_BINARY64 = "analytic_conjugate_binary64"
    LOCATION_SHIFT = "location_shift"
    UNDERDISPERSED = "underdispersed"
    IGNORED_DATA_PRIOR = "ignored_data_prior"


class ReferenceOperatingExpectation(StrEnum):
    """Frozen CI-regression direction for one fixed-key cell."""

    MAXIMUM_REJECTION_RATE = "maximum_rejection_rate"
    MINIMUM_REJECTION_RATE = "minimum_rejection_rate"
    PARAMETER_RANK_BLIND_SPOT_MAXIMUM = "parameter_rank_blind_spot_maximum"


@dataclass(frozen=True, slots=True)
class ConjugatePolicyFunctionalReferenceConfig:
    """Immutable normal--normal reference and operating-characteristic design."""

    reference_id: str
    seed: str
    frozen_at: datetime
    meta_replicate_count: int
    posterior_draw_count: int
    scenario_replicate_count: int
    observation_count: int
    operating_trial_count: int
    prior_mean: float
    prior_sd: float
    likelihood_sd: float
    posterior_shift_sd: float
    underdispersion_scale: float
    alpha: ExactRational
    maximum_fixed_key_null_rejection_rate: ExactRational
    minimum_fixed_key_detection_rate: ExactRational
    maximum_fixed_key_parameter_blind_spot_rejection_rate: ExactRational
    schema_version: str = POLICY_FUNCTIONAL_REFERENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _identifier(self.reference_id, "reference_id")
        _identifier(self.seed, "seed")
        object.__setattr__(self, "frozen_at", _aware(self.frozen_at, "frozen_at"))
        _positive_int(self.meta_replicate_count, "meta_replicate_count", minimum=8)
        _positive_int(self.posterior_draw_count, "posterior_draw_count", minimum=2)
        _positive_int(
            self.scenario_replicate_count,
            "scenario_replicate_count",
        )
        _positive_int(self.observation_count, "observation_count")
        _positive_int(self.operating_trial_count, "operating_trial_count", minimum=8)
        if self.meta_replicate_count > _MAX_META_REPLICATES:
            _fail("meta_replicate_count exceeds the exact-test resource bound")
        if self.posterior_draw_count + 1 > _MAX_RANK_CATEGORIES:
            _fail("posterior_draw_count exceeds the exact-test category bound")
        if self.operating_trial_count > _MAX_OPERATING_TRIALS:
            _fail("operating_trial_count exceeds the resource bound")
        if self.observation_count > _MAX_OBSERVATION_COUNT:
            _fail("observation_count exceeds the resource bound")
        if self.scenario_replicate_count > _MAX_SCENARIO_REPLICATE_COUNT:
            _fail("scenario_replicate_count exceeds the resource bound")
        logical_normal_draws = (
            self.operating_trial_count
            * self.meta_replicate_count
            * (
                1
                + self.observation_count
                + self.posterior_draw_count
                + self.scenario_replicate_count
            )
        )
        if logical_normal_draws > _MAX_LOGICAL_NORMAL_DRAWS:
            _fail("conjugate reference exceeds the total logical-draw budget")
        object.__setattr__(self, "prior_mean", _finite(self.prior_mean, "prior_mean"))
        prior_sd = _finite(self.prior_sd, "prior_sd")
        likelihood_sd = _finite(self.likelihood_sd, "likelihood_sd")
        shift = _finite(self.posterior_shift_sd, "posterior_shift_sd")
        dispersion = _finite(
            self.underdispersion_scale,
            "underdispersion_scale",
        )
        if prior_sd <= 0.0 or likelihood_sd <= 0.0:
            _fail("reference standard deviations must be positive")
        if shift <= 0.0:
            _fail("posterior_shift_sd must be positive")
        if not 0.0 < dispersion < 1.0:
            _fail("underdispersion_scale must lie strictly between zero and one")
        object.__setattr__(self, "prior_sd", prior_sd)
        object.__setattr__(self, "likelihood_sd", likelihood_sd)
        object.__setattr__(self, "posterior_shift_sd", shift)
        object.__setattr__(self, "underdispersion_scale", dispersion)
        _probability(self.alpha, "alpha")
        _probability(
            self.maximum_fixed_key_null_rejection_rate,
            "maximum_fixed_key_null_rejection_rate",
        )
        _probability(
            self.minimum_fixed_key_detection_rate,
            "minimum_fixed_key_detection_rate",
        )
        _probability(
            self.maximum_fixed_key_parameter_blind_spot_rejection_rate,
            "maximum_fixed_key_parameter_blind_spot_rejection_rate",
        )
        if _fraction(self.minimum_fixed_key_detection_rate) <= _fraction(
            self.maximum_fixed_key_parameter_blind_spot_rejection_rate
        ):
            _fail(
                "minimum fixed-key detection rate must exceed the maximum "
                "parameter blind-spot rejection rate"
            )
        if self.schema_version != POLICY_FUNCTIONAL_REFERENCE_SCHEMA_VERSION:
            _fail("unsupported conjugate reference schema version")

    def to_mapping(self) -> dict[str, object]:
        return {
            "alpha": self.alpha.to_mapping(),
            "frozen_at": self.frozen_at.isoformat(),
            "likelihood_sd": self.likelihood_sd,
            "maximum_fixed_key_null_rejection_rate": (
                self.maximum_fixed_key_null_rejection_rate.to_mapping()
            ),
            "maximum_fixed_key_parameter_blind_spot_rejection_rate": (
                self.maximum_fixed_key_parameter_blind_spot_rejection_rate.to_mapping()
            ),
            "meta_replicate_count": self.meta_replicate_count,
            "minimum_fixed_key_detection_rate": (
                self.minimum_fixed_key_detection_rate.to_mapping()
            ),
            "observation_count": self.observation_count,
            "operating_trial_count": self.operating_trial_count,
            "posterior_draw_count": self.posterior_draw_count,
            "posterior_shift_sd": self.posterior_shift_sd,
            "prior_mean": self.prior_mean,
            "prior_sd": self.prior_sd,
            "reference_id": self.reference_id,
            "rng_implementation_id": HASH_BASED_RNG_IMPLEMENTATION_ID,
            "scenario_replicate_count": self.scenario_replicate_count,
            "schema_version": self.schema_version,
            "seed": self.seed,
            "seed_sha256": sha256_bytes(self.seed.encode("utf-8")),
            "underdispersion_scale": self.underdispersion_scale,
        }

    @property
    def sha256(self) -> str:
        return _digest(self.to_mapping())

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        """Strictly reconstruct the complete synthetic RNG configuration."""
        expected = frozenset(
            {
                "alpha",
                "frozen_at",
                "likelihood_sd",
                "maximum_fixed_key_null_rejection_rate",
                "maximum_fixed_key_parameter_blind_spot_rejection_rate",
                "meta_replicate_count",
                "minimum_fixed_key_detection_rate",
                "observation_count",
                "operating_trial_count",
                "posterior_draw_count",
                "posterior_shift_sd",
                "prior_mean",
                "prior_sd",
                "reference_id",
                "rng_implementation_id",
                "scenario_replicate_count",
                "schema_version",
                "seed",
                "seed_sha256",
                "underdispersion_scale",
            }
        )
        if frozenset(value) != expected:
            _fail("conjugate reference config has missing or unknown fields")
        if value["rng_implementation_id"] != HASH_BASED_RNG_IMPLEMENTATION_ID:
            _fail("conjugate reference RNG implementation changed")
        seed = _identifier(value["seed"], "seed")
        if value["seed_sha256"] != sha256_bytes(seed.encode("utf-8")):
            _fail("conjugate reference seed commitment does not match")
        schema_version = value["schema_version"]
        if not isinstance(schema_version, str):
            _fail("schema_version must be a string")
        return cls(
            reference_id=_identifier(value["reference_id"], "reference_id"),
            seed=seed,
            frozen_at=_datetime_from_mapping(value["frozen_at"], "frozen_at"),
            meta_replicate_count=_positive_int(
                value["meta_replicate_count"], "meta_replicate_count"
            ),
            posterior_draw_count=_positive_int(
                value["posterior_draw_count"], "posterior_draw_count"
            ),
            scenario_replicate_count=_positive_int(
                value["scenario_replicate_count"], "scenario_replicate_count"
            ),
            observation_count=_positive_int(
                value["observation_count"], "observation_count"
            ),
            operating_trial_count=_positive_int(
                value["operating_trial_count"], "operating_trial_count"
            ),
            prior_mean=_finite(value["prior_mean"], "prior_mean"),
            prior_sd=_finite(value["prior_sd"], "prior_sd"),
            likelihood_sd=_finite(value["likelihood_sd"], "likelihood_sd"),
            posterior_shift_sd=_finite(
                value["posterior_shift_sd"], "posterior_shift_sd"
            ),
            underdispersion_scale=_finite(
                value["underdispersion_scale"], "underdispersion_scale"
            ),
            alpha=_exact_from_mapping(value["alpha"], "alpha"),
            maximum_fixed_key_null_rejection_rate=_exact_from_mapping(
                value["maximum_fixed_key_null_rejection_rate"],
                "maximum_fixed_key_null_rejection_rate",
            ),
            minimum_fixed_key_detection_rate=_exact_from_mapping(
                value["minimum_fixed_key_detection_rate"],
                "minimum_fixed_key_detection_rate",
            ),
            maximum_fixed_key_parameter_blind_spot_rejection_rate=(
                _exact_from_mapping(
                    value["maximum_fixed_key_parameter_blind_spot_rejection_rate"],
                    "maximum_fixed_key_parameter_blind_spot_rejection_rate",
                )
            ),
            schema_version=schema_version,
        )


def conjugate_policy_functional_reference_config(
    *,
    seed: str = "policytwin-conjugate-reference-v1",
) -> ConjugatePolicyFunctionalReferenceConfig:
    """Return the bounded fixed-key CI regression without a chronology claim."""
    return ConjugatePolicyFunctionalReferenceConfig(
        reference_id="policy-functional-conjugate-reference-v1",
        seed=seed,
        frozen_at=datetime(2000, 1, 1, tzinfo=UTC),
        meta_replicate_count=48,
        posterior_draw_count=7,
        scenario_replicate_count=3,
        observation_count=4,
        operating_trial_count=80,
        prior_mean=0.0,
        prior_sd=1.0,
        likelihood_sd=1.0,
        posterior_shift_sd=2.0,
        underdispersion_scale=0.35,
        alpha=ExactRational.from_ratio(1, 20),
        maximum_fixed_key_null_rejection_rate=ExactRational.from_ratio(1, 10),
        minimum_fixed_key_detection_rate=ExactRational.from_ratio(4, 5),
        maximum_fixed_key_parameter_blind_spot_rejection_rate=(
            ExactRational.from_ratio(3, 20)
        ),
    )


@lru_cache(maxsize=4096)
def _strict_cdf_sequence_count(
    sample_size: int,
    category_count: int,
    bound_numerator: int,
    bound_denominator: int,
) -> int:
    """Count labelled rank sequences whose max CDF distance is below a bound."""
    bound = Fraction(bound_numerator, bound_denominator)
    if bound <= 0:
        return 0
    paths: dict[int, int] = {0: 1}
    for category in range(1, category_count + 1):
        next_paths: dict[int, int] = {}
        for assigned, path_count in paths.items():
            for count in range(sample_size - assigned + 1):
                cumulative = assigned + count
                discrepancy = abs(
                    Fraction(cumulative, sample_size)
                    - Fraction(category, category_count)
                )
                if discrepancy < bound:
                    next_paths[cumulative] = next_paths.get(cumulative, 0) + (
                        path_count * comb(sample_size - assigned, count)
                    )
        paths = next_paths
    return paths.get(sample_size, 0)


@dataclass(frozen=True, slots=True)
class ExactDiscreteRankUniformityTest:
    """Exact conservative tail test for iid discrete-uniform rank sequences."""

    histogram: tuple[int, ...]
    alpha: ExactRational

    def __post_init__(self) -> None:
        if (
            not isinstance(self.histogram, tuple)
            or len(self.histogram) < 3
            or len(self.histogram) > _MAX_RANK_CATEGORIES
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in self.histogram
            )
        ):
            _fail("rank histogram is invalid")
        if not 1 <= sum(self.histogram) <= _MAX_META_REPLICATES:
            _fail("rank histogram sample size is outside the exact-test bound")
        _probability(self.alpha, "alpha")

    @property
    def sample_size(self) -> int:
        return sum(self.histogram)

    @property
    def category_count(self) -> int:
        return len(self.histogram)

    @property
    def discrepancy(self) -> ExactRational:
        return exact_rank_cdf_discrepancy(
            self.histogram,
            posterior_draw_count=self.category_count - 1,
        )

    @property
    def exact_tail_p_value(self) -> ExactRational:
        discrepancy = _fraction(self.discrepancy)
        sequence_space_size = self.category_count**self.sample_size
        strictly_below = _strict_cdf_sequence_count(
            self.sample_size,
            self.category_count,
            discrepancy.numerator,
            discrepancy.denominator,
        )
        return ExactRational.from_ratio(
            sequence_space_size - strictly_below,
            sequence_space_size,
        )

    @property
    def rejected(self) -> bool:
        return _fraction(self.exact_tail_p_value) <= _fraction(self.alpha)

    def to_mapping(self) -> dict[str, object]:
        return {
            "alpha": self.alpha.to_mapping(),
            "category_count": self.category_count,
            "discrepancy": self.discrepancy.to_mapping(),
            "exact_tail_p_value": self.exact_tail_p_value.to_mapping(),
            "histogram": list(self.histogram),
            "rejected": self.rejected,
            "sample_size": self.sample_size,
            "tail_rule": "reject_when_exact_P_null_D_ge_observed_le_alpha",
        }


@dataclass(frozen=True, slots=True)
class ReferenceTrialResult:
    """One complete rank-uniformity trial and its EventKey commitment."""

    trial_index: int
    test: ExactDiscreteRankUniformityTest
    event_keys_sha256: str
    event_key_count: int
    tie_count: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.trial_index, bool)
            or not isinstance(self.trial_index, int)
            or self.trial_index < 0
        ):
            _fail("trial_index must be a non-negative integer")
        if type(self.test) is not ExactDiscreteRankUniformityTest:
            raise TypeError("test must be an ExactDiscreteRankUniformityTest")
        if not isinstance(self.event_keys_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", self.event_keys_sha256
        ):
            _fail("event_keys_sha256 must be a lowercase SHA-256 digest")
        _positive_int(self.event_key_count, "event_key_count")
        if isinstance(self.tie_count, bool) or not isinstance(self.tie_count, int):
            _fail("tie_count must be a non-negative integer")
        if self.tie_count < 0:
            _fail("tie_count must be a non-negative integer")

    def to_mapping(self) -> dict[str, object]:
        return {
            "event_key_count": self.event_key_count,
            "event_keys_sha256": self.event_keys_sha256,
            "test": self.test.to_mapping(),
            "tie_count": self.tie_count,
            "trial_index": self.trial_index,
        }


@dataclass(frozen=True, slots=True)
class ReferenceOperatingCell:
    """Frozen regression cell over disjoint deterministic key trials."""

    functional: ReferenceFunctional
    posterior_mode: ReferencePosteriorMode
    trials: tuple[ReferenceTrialResult, ...]
    expectation: ReferenceOperatingExpectation
    threshold: ExactRational

    def __post_init__(self) -> None:
        if not isinstance(self.functional, ReferenceFunctional) or not isinstance(
            self.posterior_mode, ReferencePosteriorMode
        ):
            _fail("operating cell uses an invalid functional or posterior mode")
        if (
            not isinstance(self.trials, tuple)
            or not self.trials
            or not all(type(item) is ReferenceTrialResult for item in self.trials)
            or tuple(item.trial_index for item in self.trials)
            != tuple(range(len(self.trials)))
        ):
            _fail("operating trials must exactly cover the ordered trial axis")
        if not isinstance(self.expectation, ReferenceOperatingExpectation):
            _fail("operating expectation has the wrong type")
        _probability(self.threshold, "operating threshold")
        alphas = {item.test.alpha for item in self.trials}
        dimensions = {
            (item.test.sample_size, item.test.category_count) for item in self.trials
        }
        if len(alphas) != 1 or len(dimensions) != 1:
            _fail("operating trials changed alpha or rank dimensions")

    @property
    def rejection_count(self) -> int:
        return sum(item.test.rejected for item in self.trials)

    @property
    def rejection_rate(self) -> ExactRational:
        return ExactRational.from_ratio(self.rejection_count, len(self.trials))

    @property
    def passed(self) -> bool:
        rate = _fraction(self.rejection_rate)
        threshold = _fraction(self.threshold)
        if self.expectation is ReferenceOperatingExpectation.MINIMUM_REJECTION_RATE:
            return rate >= threshold
        return rate <= threshold

    def to_mapping(self) -> dict[str, object]:
        return {
            "expectation": self.expectation.value,
            "functional": self.functional.value,
            "passed": self.passed,
            "posterior_mode": self.posterior_mode.value,
            "rejection_count": self.rejection_count,
            "rejection_rate": self.rejection_rate.to_mapping(),
            "threshold": self.threshold.to_mapping(),
            "trials": [item.to_mapping() for item in self.trials],
        }


def _event_key(
    config: ConjugatePolicyFunctionalReferenceConfig,
    *,
    trial_index: int,
    meta_replicate_index: int,
) -> SemanticEventKey:
    return SemanticEventKey(
        scenario_id=(
            f"{config.reference_id}-t{trial_index:04d}-b{meta_replicate_index:04d}"
        ),
        event_type=EventType.EXOGENOUS_SHOCK,
        entity_id="conjugate-reference",
        timestamp=_REFERENCE_EVENT_EPOCH
        + timedelta(
            seconds=(trial_index * config.meta_replicate_count) + meta_replicate_index
        ),
    )


def _draw_receipt(key: SemanticEventKey, domain: str, draw_index: int) -> str:
    return f"{key.value}|{domain}|{draw_index}"


def _consume_normal(
    rng: HashBasedRNG,
    key: SemanticEventKey,
    *,
    domain: str,
    draw_index: int,
    receipts: list[str],
    consumed: set[str],
) -> float:
    """Consume one logical normal identity and record it before sampling."""
    receipt = _draw_receipt(key, domain, draw_index)
    if receipt in consumed:
        _fail("reference trial reused a normal EventKey/domain/index identity")
    consumed.add(receipt)
    receipts.append(receipt)
    return rng.normal(key, domain=domain, draw_index=draw_index)


def _consume_integer(
    rng: HashBasedRNG,
    key: SemanticEventKey,
    lower: int,
    upper: int,
    *,
    domain: str,
    draw_index: int,
    receipts: list[str],
    consumed: set[str],
) -> int:
    """Consume one logical integer identity and record it before sampling."""
    receipt = _draw_receipt(key, domain, draw_index)
    if receipt in consumed:
        _fail("reference trial reused an integer EventKey/domain/index identity")
    consumed.add(receipt)
    receipts.append(receipt)
    return rng.integer(
        key,
        lower,
        upper,
        domain=domain,
        draw_index=draw_index,
    )


def _posterior_parameters(
    config: ConjugatePolicyFunctionalReferenceConfig,
    observations: tuple[float, ...],
) -> tuple[float, float]:
    prior_precision = 1.0 / (config.prior_sd * config.prior_sd)
    likelihood_precision = 1.0 / (config.likelihood_sd * config.likelihood_sd)
    posterior_precision = prior_precision + (
        config.observation_count * likelihood_precision
    )
    posterior_mean = (
        config.prior_mean * prior_precision + sum(observations) * likelihood_precision
    ) / posterior_precision
    return posterior_mean, math.sqrt(1.0 / posterior_precision)


def _posterior_draws(
    config: ConjugatePolicyFunctionalReferenceConfig,
    *,
    mode: ReferencePosteriorMode,
    posterior_mean: float,
    posterior_sd: float,
    standard_normals: tuple[float, ...],
) -> tuple[float, ...]:
    if mode is ReferencePosteriorMode.ANALYTIC_CONJUGATE_BINARY64:
        return tuple(
            posterior_mean + posterior_sd * value for value in standard_normals
        )
    if mode is ReferencePosteriorMode.LOCATION_SHIFT:
        return tuple(
            posterior_mean
            + config.posterior_shift_sd * posterior_sd
            + posterior_sd * value
            for value in standard_normals
        )
    if mode is ReferencePosteriorMode.UNDERDISPERSED:
        return tuple(
            posterior_mean + config.underdispersion_scale * posterior_sd * value
            for value in standard_normals
        )
    if mode is ReferencePosteriorMode.IGNORED_DATA_PRIOR:
        return tuple(
            config.prior_mean + config.prior_sd * value for value in standard_normals
        )
    _fail("unsupported posterior mode")


def _functional_value(
    *,
    functional: ReferenceFunctional,
    parameter: float,
    pre_policy_mean: float,
    scenario_shocks: tuple[float, ...],
) -> Fraction:
    if functional is ReferenceFunctional.PARAMETER_ONLY:
        return Fraction.from_float(parameter)
    scenario_mean = sum(scenario_shocks) / len(scenario_shocks)
    value = parameter * pre_policy_mean + (parameter * parameter * scenario_mean / 10.0)
    return Fraction.from_float(value)


def _expectation(
    config: ConjugatePolicyFunctionalReferenceConfig,
    *,
    functional: ReferenceFunctional,
    mode: ReferencePosteriorMode,
) -> tuple[ReferenceOperatingExpectation, ExactRational]:
    if mode is ReferencePosteriorMode.ANALYTIC_CONJUGATE_BINARY64:
        return (
            ReferenceOperatingExpectation.MAXIMUM_REJECTION_RATE,
            config.maximum_fixed_key_null_rejection_rate,
        )
    if (
        mode is ReferencePosteriorMode.IGNORED_DATA_PRIOR
        and functional is ReferenceFunctional.PARAMETER_ONLY
    ):
        return (
            ReferenceOperatingExpectation.PARAMETER_RANK_BLIND_SPOT_MAXIMUM,
            config.maximum_fixed_key_parameter_blind_spot_rejection_rate,
        )
    return (
        ReferenceOperatingExpectation.MINIMUM_REJECTION_RATE,
        config.minimum_fixed_key_detection_rate,
    )


def _derive_reference(
    config: ConjugatePolicyFunctionalReferenceConfig,
) -> tuple[tuple[ReferenceOperatingCell, ...], tuple[str, ...]]:
    rng = HashBasedRNG(config.seed)
    trial_results: dict[
        tuple[ReferenceFunctional, ReferencePosteriorMode],
        list[ReferenceTrialResult],
    ] = {
        (functional, mode): []
        for functional in ReferenceFunctional
        for mode in ReferencePosteriorMode
    }
    complete_event_keys: list[str] = []
    for trial_index in range(config.operating_trial_count):
        histograms = {
            coordinate: [0] * (config.posterior_draw_count + 1)
            for coordinate in trial_results
        }
        tie_counts = {coordinate: 0 for coordinate in trial_results}
        trial_event_keys: list[str] = []
        trial_consumed: set[str] = set()
        for meta_index in range(config.meta_replicate_count):
            key = _event_key(
                config,
                trial_index=trial_index,
                meta_replicate_index=meta_index,
            )
            truth_z = _consume_normal(
                rng,
                key,
                domain="reference-truth",
                draw_index=0,
                receipts=trial_event_keys,
                consumed=trial_consumed,
            )
            truth = config.prior_mean + config.prior_sd * truth_z
            observations = tuple(
                truth
                + config.likelihood_sd
                * _consume_normal(
                    rng,
                    key,
                    domain="reference-data",
                    draw_index=index,
                    receipts=trial_event_keys,
                    consumed=trial_consumed,
                )
                for index in range(config.observation_count)
            )
            posterior_mean, posterior_sd = _posterior_parameters(
                config,
                observations,
            )
            posterior_normals = tuple(
                _consume_normal(
                    rng,
                    key,
                    domain="reference-posterior",
                    draw_index=index,
                    receipts=trial_event_keys,
                    consumed=trial_consumed,
                )
                for index in range(config.posterior_draw_count)
            )
            scenario_shocks = tuple(
                _consume_normal(
                    rng,
                    key,
                    domain="reference-scenario",
                    draw_index=index,
                    receipts=trial_event_keys,
                    consumed=trial_consumed,
                )
                for index in range(config.scenario_replicate_count)
            )
            pre_policy_mean = sum(observations) / len(observations)
            for mode in ReferencePosteriorMode:
                draws = _posterior_draws(
                    config,
                    mode=mode,
                    posterior_mean=posterior_mean,
                    posterior_sd=posterior_sd,
                    standard_normals=posterior_normals,
                )
                for functional in ReferenceFunctional:
                    coordinate = (functional, mode)
                    truth_value = _functional_value(
                        functional=functional,
                        parameter=truth,
                        pre_policy_mean=pre_policy_mean,
                        scenario_shocks=scenario_shocks,
                    )
                    draw_values = tuple(
                        _functional_value(
                            functional=functional,
                            parameter=value,
                            pre_policy_mean=pre_policy_mean,
                            scenario_shocks=scenario_shocks,
                        )
                        for value in draws
                    )
                    less = sum(value < truth_value for value in draw_values)
                    equal = sum(value == truth_value for value in draw_values)
                    if equal:
                        tie_counts[coordinate] += equal
                        tie_draw = _consume_integer(
                            rng,
                            key,
                            0,
                            equal,
                            domain=f"reference-tie-{functional.value}-{mode.value}",
                            draw_index=0,
                            receipts=trial_event_keys,
                            consumed=trial_consumed,
                        )
                    else:
                        tie_draw = 0
                    histograms[coordinate][less + tie_draw] += 1
        if len(trial_consumed) != len(trial_event_keys):
            _fail("reference trial reused a random EventKey/domain/index identity")
        complete_event_keys.extend(trial_event_keys)
        trial_key_sha256 = _digest(trial_event_keys)
        for coordinate, histogram in histograms.items():
            trial_results[coordinate].append(
                ReferenceTrialResult(
                    trial_index=trial_index,
                    test=ExactDiscreteRankUniformityTest(
                        histogram=tuple(histogram),
                        alpha=config.alpha,
                    ),
                    event_keys_sha256=trial_key_sha256,
                    event_key_count=len(trial_event_keys),
                    tie_count=tie_counts[coordinate],
                )
            )
    if len(set(complete_event_keys)) != len(complete_event_keys):
        _fail("reference operating grid reused a random identity")
    cells = []
    for functional, mode in sorted(
        trial_results,
        key=lambda item: (item[0].value, item[1].value),
    ):
        expectation, threshold = _expectation(
            config,
            functional=functional,
            mode=mode,
        )
        cells.append(
            ReferenceOperatingCell(
                functional=functional,
                posterior_mode=mode,
                trials=tuple(trial_results[(functional, mode)]),
                expectation=expectation,
                threshold=threshold,
            )
        )
    return tuple(cells), tuple(complete_event_keys)


def _fixed_key_target_groups(
    cells: tuple[ReferenceOperatingCell, ...],
) -> tuple[bool, bool, bool, bool]:
    """Separate fixed-grid null, alternative, and blind-spot regressions."""
    null_cells = tuple(
        cell
        for cell in cells
        if cell.posterior_mode is ReferencePosteriorMode.ANALYTIC_CONJUGATE_BINARY64
    )
    blind_spot_cells = tuple(
        cell
        for cell in cells
        if cell.functional is ReferenceFunctional.PARAMETER_ONLY
        and cell.posterior_mode is ReferencePosteriorMode.IGNORED_DATA_PRIOR
    )
    alternative_cells = tuple(
        cell
        for cell in cells
        if cell.posterior_mode is not ReferencePosteriorMode.ANALYTIC_CONJUGATE_BINARY64
        and not (
            cell.functional is ReferenceFunctional.PARAMETER_ONLY
            and cell.posterior_mode is ReferencePosteriorMode.IGNORED_DATA_PRIOR
        )
    )
    if not null_cells or not alternative_cells or len(blind_spot_cells) != 1:
        _fail("fixed-key target groups are incomplete")
    null_pass = all(cell.passed for cell in null_cells)
    alternative_pass = all(cell.passed for cell in alternative_cells)
    blind_spot_pass = blind_spot_cells[0].passed
    return (
        null_pass and alternative_pass and blind_spot_pass,
        null_pass,
        alternative_pass,
        blind_spot_pass,
    )


def _ignored_data_blind_spot_exposed(
    cells: tuple[ReferenceOperatingCell, ...],
) -> bool:
    """Require both gates and a strictly stronger conditioned rejection rate."""
    parameter = next(
        cell
        for cell in cells
        if cell.functional is ReferenceFunctional.PARAMETER_ONLY
        and cell.posterior_mode is ReferencePosteriorMode.IGNORED_DATA_PRIOR
    )
    conditioned = next(
        cell
        for cell in cells
        if cell.functional
        is ReferenceFunctional.PRE_POLICY_DATA_AND_SCENARIO_CONDITIONED
        and cell.posterior_mode is ReferencePosteriorMode.IGNORED_DATA_PRIOR
    )
    return (
        parameter.passed
        and conditioned.passed
        and _fraction(conditioned.rejection_rate) > _fraction(parameter.rejection_rate)
    )


@dataclass(frozen=True, slots=True, init=False)
class ConjugatePolicyFunctionalReferenceResult:
    """Fully rederived fixed-key binary64 conjugate regression benchmark."""

    config: ConjugatePolicyFunctionalReferenceConfig
    cells: tuple[ReferenceOperatingCell, ...]
    event_keys_sha256: str
    event_key_count: int
    fixed_key_operating_targets_met: bool
    fixed_key_null_targets_met: bool
    fixed_key_alternative_detection_targets_met: bool
    fixed_key_parameter_blind_spot_target_met: bool
    analytic_normal_normal_posterior_formula_used: Literal[True]
    exact_discrete_rank_tail_arithmetic_rederived: Literal[True]
    binary64_forward_sampling_used: Literal[True]
    continuous_normal_sampler_exact: Literal[False]
    cross_platform_binary64_replay_verified: Literal[False]
    iid_discrete_uniform_rank_null_verified_by_sampler: Literal[False]
    logical_random_identity_uniqueness_checked: Literal[True]
    administrative_freeze_time_excluded_from_rng_identity: Literal[True]
    ignored_data_parameter_rank_blind_spot_exposed: bool
    integrated_abc_posterior_exchangeability_verified: Literal[False]
    integrated_market_policy_validity_verified: Literal[False]
    external_preregistration_verified: Literal[False]
    independent_reproduction: Literal[False]
    standalone_environment_provenance_complete: Literal[False]
    real_world_coverage_claim: Literal[False]
    real_world_causal_claim: Literal[False]
    publication_eligible: Literal[False]
    scientific_abstention_required: Literal[True]
    claim_scope: str
    schema_version: str

    def __init__(self) -> None:
        raise TypeError(
            "ConjugatePolicyFunctionalReferenceResult is factory-produced; use "
            "run_conjugate_policy_functional_reference"
        )

    def __copy__(self) -> NoReturn:
        raise TypeError("conjugate reference results cannot be copied")

    def __deepcopy__(self, memo: object) -> NoReturn:
        del memo
        raise TypeError("conjugate reference results cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("conjugate reference results cannot be serialized")

    def _fresh_validate(self) -> None:
        if type(self.config) is not ConjugatePolicyFunctionalReferenceConfig:
            raise TypeError("reference result lost its exact config")
        expected_cells, event_keys = _derive_reference(self.config)
        if self.cells != expected_cells:
            _fail("reference operating cells do not freshly rederive")
        if self.event_keys_sha256 != _digest(event_keys) or self.event_key_count != len(
            event_keys
        ):
            _fail("reference EventKey manifest does not freshly rederive")
        (
            expected_pass,
            expected_null_pass,
            expected_alternative_pass,
            expected_blind_spot_pass,
        ) = _fixed_key_target_groups(self.cells)
        expected_blind_spot = _ignored_data_blind_spot_exposed(self.cells)
        if (
            self.administrative_freeze_time_excluded_from_rng_identity is not True
            or self.fixed_key_operating_targets_met is not expected_pass
            or self.fixed_key_null_targets_met is not expected_null_pass
            or self.fixed_key_alternative_detection_targets_met
            is not expected_alternative_pass
            or self.fixed_key_parameter_blind_spot_target_met
            is not expected_blind_spot_pass
            or self.ignored_data_parameter_rank_blind_spot_exposed
            is not expected_blind_spot
            or self.analytic_normal_normal_posterior_formula_used is not True
            or self.exact_discrete_rank_tail_arithmetic_rederived is not True
            or self.binary64_forward_sampling_used is not True
            or self.continuous_normal_sampler_exact is not False
            or self.cross_platform_binary64_replay_verified is not False
            or self.iid_discrete_uniform_rank_null_verified_by_sampler is not False
            or self.logical_random_identity_uniqueness_checked is not True
            or self.integrated_abc_posterior_exchangeability_verified is not False
            or self.integrated_market_policy_validity_verified is not False
            or self.external_preregistration_verified is not False
            or self.independent_reproduction is not False
            or self.standalone_environment_provenance_complete is not False
            or self.real_world_coverage_claim is not False
            or self.real_world_causal_claim is not False
            or self.publication_eligible is not False
            or self.scientific_abstention_required is not True
            or self.claim_scope != POLICY_FUNCTIONAL_REFERENCE_CLAIM_SCOPE
            or self.schema_version != POLICY_FUNCTIONAL_REFERENCE_SCHEMA_VERSION
        ):
            _fail("conjugate reference scientific claim boundary changed")

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_mapping())

    @property
    def sha256(self) -> str:
        return sha256_bytes(self.canonical_bytes)

    def to_mapping(self) -> dict[str, object]:
        self._fresh_validate()
        return {
            "administrative_freeze_time_excluded_from_rng_identity": (
                self.administrative_freeze_time_excluded_from_rng_identity
            ),
            "analytic_normal_normal_posterior_formula_used": (
                self.analytic_normal_normal_posterior_formula_used
            ),
            "binary64_forward_sampling_used": self.binary64_forward_sampling_used,
            "cells": [cell.to_mapping() for cell in self.cells],
            "claim_scope": self.claim_scope,
            "config": self.config.to_mapping(),
            "config_sha256": self.config.sha256,
            "event_key_count": self.event_key_count,
            "event_keys_sha256": self.event_keys_sha256,
            "exact_discrete_rank_tail_arithmetic_rederived": (
                self.exact_discrete_rank_tail_arithmetic_rederived
            ),
            "continuous_normal_sampler_exact": self.continuous_normal_sampler_exact,
            "cross_platform_binary64_replay_verified": (
                self.cross_platform_binary64_replay_verified
            ),
            "external_preregistration_verified": (
                self.external_preregistration_verified
            ),
            "ignored_data_parameter_rank_blind_spot_exposed": (
                self.ignored_data_parameter_rank_blind_spot_exposed
            ),
            "independent_reproduction": self.independent_reproduction,
            "integrated_abc_posterior_exchangeability_verified": (
                self.integrated_abc_posterior_exchangeability_verified
            ),
            "integrated_market_policy_validity_verified": (
                self.integrated_market_policy_validity_verified
            ),
            "iid_discrete_uniform_rank_null_verified_by_sampler": (
                self.iid_discrete_uniform_rank_null_verified_by_sampler
            ),
            "logical_random_identity_uniqueness_checked": (
                self.logical_random_identity_uniqueness_checked
            ),
            "publication_eligible": self.publication_eligible,
            "real_world_causal_claim": self.real_world_causal_claim,
            "real_world_coverage_claim": self.real_world_coverage_claim,
            "fixed_key_operating_targets_met": self.fixed_key_operating_targets_met,
            "fixed_key_null_targets_met": self.fixed_key_null_targets_met,
            "fixed_key_alternative_detection_targets_met": (
                self.fixed_key_alternative_detection_targets_met
            ),
            "fixed_key_parameter_blind_spot_target_met": (
                self.fixed_key_parameter_blind_spot_target_met
            ),
            "schema_version": self.schema_version,
            "scientific_abstention_required": self.scientific_abstention_required,
            "standalone_environment_provenance_complete": (
                self.standalone_environment_provenance_complete
            ),
        }


def run_conjugate_policy_functional_reference(
    config: ConjugatePolicyFunctionalReferenceConfig,
) -> ConjugatePolicyFunctionalReferenceResult:
    """Run and rederive the fixed-key binary64 conjugate regression."""
    if type(config) is not ConjugatePolicyFunctionalReferenceConfig:
        raise TypeError("config must be a ConjugatePolicyFunctionalReferenceConfig")
    cells, event_keys = _derive_reference(config)
    passed, null_pass, alternative_pass, blind_spot_pass = _fixed_key_target_groups(
        cells
    )
    result = object.__new__(ConjugatePolicyFunctionalReferenceResult)
    values: tuple[tuple[str, object], ...] = (
        ("config", config),
        ("cells", cells),
        ("event_keys_sha256", _digest(event_keys)),
        ("event_key_count", len(event_keys)),
        ("fixed_key_operating_targets_met", passed),
        ("fixed_key_null_targets_met", null_pass),
        ("fixed_key_alternative_detection_targets_met", alternative_pass),
        ("fixed_key_parameter_blind_spot_target_met", blind_spot_pass),
        ("administrative_freeze_time_excluded_from_rng_identity", True),
        ("analytic_normal_normal_posterior_formula_used", True),
        ("exact_discrete_rank_tail_arithmetic_rederived", True),
        ("binary64_forward_sampling_used", True),
        ("continuous_normal_sampler_exact", False),
        ("cross_platform_binary64_replay_verified", False),
        ("iid_discrete_uniform_rank_null_verified_by_sampler", False),
        ("logical_random_identity_uniqueness_checked", True),
        (
            "ignored_data_parameter_rank_blind_spot_exposed",
            _ignored_data_blind_spot_exposed(cells),
        ),
        ("integrated_abc_posterior_exchangeability_verified", False),
        ("integrated_market_policy_validity_verified", False),
        ("external_preregistration_verified", False),
        ("independent_reproduction", False),
        ("standalone_environment_provenance_complete", False),
        ("real_world_coverage_claim", False),
        ("real_world_causal_claim", False),
        ("publication_eligible", False),
        ("scientific_abstention_required", True),
        ("claim_scope", POLICY_FUNCTIONAL_REFERENCE_CLAIM_SCOPE),
        ("schema_version", POLICY_FUNCTIONAL_REFERENCE_SCHEMA_VERSION),
    )
    for field, value in values:
        object.__setattr__(result, field, value)
    result._fresh_validate()
    return result


def replay_conjugate_policy_functional_reference(
    artifact: Mapping[str, object],
) -> ConjugatePolicyFunctionalReferenceResult:
    """Rebuild a persisted mapping and reject any semantic or byte drift."""
    config_mapping = _mapping(artifact.get("config"), "artifact.config")
    config = ConjugatePolicyFunctionalReferenceConfig.from_mapping(config_mapping)
    replay = run_conjugate_policy_functional_reference(config)
    if canonical_json_bytes(dict(artifact)) != replay.canonical_bytes:
        _fail("persisted conjugate reference artifact does not freshly rederive")
    return replay


__all__ = [
    "POLICY_FUNCTIONAL_REFERENCE_CLAIM_SCOPE",
    "POLICY_FUNCTIONAL_REFERENCE_SCHEMA_VERSION",
    "ConjugatePolicyFunctionalReferenceConfig",
    "ConjugatePolicyFunctionalReferenceResult",
    "ExactDiscreteRankUniformityTest",
    "PolicyFunctionalReferenceError",
    "ReferenceFunctional",
    "ReferenceOperatingCell",
    "ReferenceOperatingExpectation",
    "ReferencePosteriorMode",
    "ReferenceTrialResult",
    "conjugate_policy_functional_reference_config",
    "replay_conjugate_policy_functional_reference",
    "run_conjugate_policy_functional_reference",
]
