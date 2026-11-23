"""Deterministic MCSE diagnostics for distributional policy summaries.

The matrix scheduler's scalar mean MCSE is not a valid substitute for the
uncertainty of a quantile or tail probability. This module evaluates those
predeclared statistics with an event-keyed nonparametric bootstrap and returns
an explicit checkpoint decision. It never chooses a statistic or threshold
from the observed target outcome.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from statistics import fmean
from typing import NoReturn, Self, cast

from policytwin.counterfactual import HashBasedRNG, SemanticEventKey
from policytwin.provenance import canonical_json_bytes, sha256_bytes

DISTRIBUTION_MCSE_SCHEMA_VERSION = "policytwin.distribution-mcse.v1"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class DistributionMCSEError(ValueError):
    """Raised when an MCSE contract or draw sequence is invalid."""


def _fail(message: str) -> NoReturn:
    raise DistributionMCSEError(message)


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        _fail(f"{field} must be a stable identifier")
    return value


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        _fail(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{field} must be finite")
    return result


def _positive_int(value: object, field: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"{field} must be an integer of at least {minimum}")
    return value


def _aware(value: object, field: str) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        _fail(f"{field} must be a timezone-aware datetime")
    return value.astimezone(UTC)


def _parse_datetime(value: object, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        _fail(f"{field} must be a non-empty ISO 8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise DistributionMCSEError(f"{field} must be ISO 8601") from error
    return _aware(parsed, field)


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        _fail(f"{field} must be an object with string keys")
    return cast(Mapping[str, object], value)


def _strict_fields(
    value: Mapping[str, object], expected: frozenset[str], context: str
) -> None:
    if frozenset(value) != expected:
        _fail(f"{context} has missing or unknown fields")


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        _fail(f"{field} must be boolean")
    return value


def _optional_float(value: object, field: str) -> float | None:
    if value is None:
        return None
    return _finite(value, field)


def _quantile(values: tuple[float, ...], level: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * level
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + weight * (ordered[upper] - ordered[lower])


class MCSEStatistic(StrEnum):
    """Predeclared distributional summaries supported by the stop rule."""

    MEAN = "mean"
    QUANTILE = "quantile"
    SIGN_PROBABILITY = "sign_probability"
    TAIL_PROBABILITY = "tail_probability"


class TailDirection(StrEnum):
    """Which frozen tail is counted by a tail-probability statistic."""

    LOWER = "lower"
    UPPER = "upper"


class MCSEStopDecision(StrEnum):
    """Checkpoint outcome without hiding an unmet precision target."""

    CONTINUE = "continue"
    TARGET_REACHED = "target_reached"
    MAXIMUM_REACHED = "maximum_reached"


@dataclass(frozen=True, slots=True)
class DistributionMCSESpec:
    """Immutable statistic, precision target, and bootstrap design."""

    spec_id: str
    statistic: MCSEStatistic
    target_mcse: float
    minimum_replicates: int
    maximum_replicates: int
    check_every: int
    bootstrap_replicates: int
    bootstrap_seed: str
    frozen_at: datetime
    target_outcomes_available_at: datetime
    quantile_level: float | None = None
    tail_threshold: float | None = None
    tail_direction: TailDirection | None = None
    policy_post_data_used: bool = False
    synthetic_only: bool = True
    schema_version: str = DISTRIBUTION_MCSE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _identifier(self.spec_id, "spec_id")
        if not isinstance(self.statistic, MCSEStatistic):
            _fail("statistic must be an MCSEStatistic")
        target = _finite(self.target_mcse, "target_mcse")
        if target < 0.0:
            _fail("target_mcse must be non-negative")
        object.__setattr__(self, "target_mcse", target)
        minimum = _positive_int(
            self.minimum_replicates, "minimum_replicates", minimum=2
        )
        maximum = _positive_int(
            self.maximum_replicates, "maximum_replicates", minimum=minimum
        )
        check_every = _positive_int(self.check_every, "check_every")
        if minimum % check_every != 0 or maximum % check_every != 0:
            _fail("minimum and maximum replicates must be MCSE checkpoints")
        _positive_int(self.bootstrap_replicates, "bootstrap_replicates", minimum=2)
        if not isinstance(self.bootstrap_seed, str) or not self.bootstrap_seed:
            _fail("bootstrap_seed must be non-empty")
        frozen = _aware(self.frozen_at, "frozen_at")
        available = _aware(
            self.target_outcomes_available_at,
            "target_outcomes_available_at",
        )
        if frozen >= available:
            _fail("MCSE rule must be frozen before target outcomes are available")
        object.__setattr__(self, "frozen_at", frozen)
        object.__setattr__(self, "target_outcomes_available_at", available)
        if self.policy_post_data_used is not False:
            _fail("policy-post data cannot define an MCSE rule")
        if not isinstance(self.synthetic_only, bool):
            _fail("synthetic_only must be boolean")
        if self.statistic is MCSEStatistic.QUANTILE:
            level = _finite(self.quantile_level, "quantile_level")
            if not 0.0 < level < 1.0:
                _fail("quantile_level must lie strictly inside (0, 1)")
            object.__setattr__(self, "quantile_level", level)
            if self.tail_threshold is not None or self.tail_direction is not None:
                _fail("quantile rule cannot include tail fields")
        elif self.statistic is MCSEStatistic.TAIL_PROBABILITY:
            object.__setattr__(
                self,
                "tail_threshold",
                _finite(self.tail_threshold, "tail_threshold"),
            )
            if not isinstance(self.tail_direction, TailDirection):
                _fail("tail probability requires a TailDirection")
            if self.quantile_level is not None:
                _fail("tail rule cannot include a quantile level")
        elif any(
            value is not None
            for value in (
                self.quantile_level,
                self.tail_threshold,
                self.tail_direction,
            )
        ):
            _fail("mean and sign rules cannot include quantile or tail fields")
        if self.schema_version != DISTRIBUTION_MCSE_SCHEMA_VERSION:
            _fail("unsupported distribution MCSE schema version")

    @property
    def sha256(self) -> str:
        """Return the complete stop-rule commitment."""
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def to_mapping(self) -> dict[str, object]:
        """Return the frozen rule, including the explicit reproducibility key."""
        return {
            "bootstrap_replicates": self.bootstrap_replicates,
            "bootstrap_seed": self.bootstrap_seed,
            "check_every": self.check_every,
            "frozen_at": self.frozen_at.isoformat(),
            "maximum_replicates": self.maximum_replicates,
            "minimum_replicates": self.minimum_replicates,
            "policy_post_data_used": self.policy_post_data_used,
            "quantile_level": self.quantile_level,
            "schema_version": self.schema_version,
            "spec_id": self.spec_id,
            "statistic": self.statistic.value,
            "synthetic_only": self.synthetic_only,
            "tail_direction": (
                None if self.tail_direction is None else self.tail_direction.value
            ),
            "tail_threshold": self.tail_threshold,
            "target_mcse": self.target_mcse,
            "target_outcomes_available_at": (
                self.target_outcomes_available_at.isoformat()
            ),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        """Strictly reconstruct a frozen distributional stop rule."""
        _strict_fields(
            value,
            frozenset(
                {
                    "bootstrap_replicates",
                    "bootstrap_seed",
                    "check_every",
                    "frozen_at",
                    "maximum_replicates",
                    "minimum_replicates",
                    "policy_post_data_used",
                    "quantile_level",
                    "schema_version",
                    "spec_id",
                    "statistic",
                    "synthetic_only",
                    "tail_direction",
                    "tail_threshold",
                    "target_mcse",
                    "target_outcomes_available_at",
                }
            ),
            "distribution MCSE spec",
        )
        raw_statistic = value["statistic"]
        if not isinstance(raw_statistic, str):
            _fail("statistic must be a string")
        try:
            statistic = MCSEStatistic(raw_statistic)
        except ValueError as error:
            raise DistributionMCSEError("unsupported MCSE statistic") from error
        raw_direction = value["tail_direction"]
        if raw_direction is None:
            direction = None
        elif isinstance(raw_direction, str):
            try:
                direction = TailDirection(raw_direction)
            except ValueError as error:
                raise DistributionMCSEError("unsupported tail direction") from error
        else:
            _fail("tail_direction must be a string or null")
        bootstrap_seed = value["bootstrap_seed"]
        schema_version = value["schema_version"]
        if not isinstance(bootstrap_seed, str) or not isinstance(schema_version, str):
            _fail("bootstrap_seed and schema_version must be strings")
        return cls(
            spec_id=_identifier(value["spec_id"], "spec_id"),
            statistic=statistic,
            target_mcse=_finite(value["target_mcse"], "target_mcse"),
            minimum_replicates=_positive_int(
                value["minimum_replicates"], "minimum_replicates", minimum=2
            ),
            maximum_replicates=_positive_int(
                value["maximum_replicates"], "maximum_replicates", minimum=2
            ),
            check_every=_positive_int(value["check_every"], "check_every"),
            bootstrap_replicates=_positive_int(
                value["bootstrap_replicates"],
                "bootstrap_replicates",
                minimum=2,
            ),
            bootstrap_seed=bootstrap_seed,
            frozen_at=_parse_datetime(value["frozen_at"], "frozen_at"),
            target_outcomes_available_at=_parse_datetime(
                value["target_outcomes_available_at"],
                "target_outcomes_available_at",
            ),
            quantile_level=_optional_float(value["quantile_level"], "quantile_level"),
            tail_threshold=_optional_float(value["tail_threshold"], "tail_threshold"),
            tail_direction=direction,
            policy_post_data_used=_boolean(
                value["policy_post_data_used"], "policy_post_data_used"
            ),
            synthetic_only=_boolean(value["synthetic_only"], "synthetic_only"),
            schema_version=schema_version,
        )


@dataclass(frozen=True, slots=True)
class MonteCarloDraw:
    """One explicit replicate and semantic event identity."""

    replicate_id: int
    event_key: SemanticEventKey
    value: float

    def __post_init__(self) -> None:
        _positive_int(self.replicate_id, "replicate_id", minimum=0)
        if not isinstance(self.event_key, SemanticEventKey):
            _fail("event_key must be a SemanticEventKey")
        object.__setattr__(self, "value", _finite(self.value, "draw value"))

    def to_mapping(self) -> dict[str, object]:
        """Return the draw identity and value."""
        return {
            "event_key": self.event_key.to_mapping(),
            "replicate_id": self.replicate_id,
            "value": self.value,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        """Strictly reconstruct a draw and verify its semantic event key."""
        _strict_fields(
            value,
            frozenset({"event_key", "replicate_id", "value"}),
            "Monte Carlo draw",
        )
        return cls(
            replicate_id=_positive_int(
                value["replicate_id"], "replicate_id", minimum=0
            ),
            event_key=SemanticEventKey.from_mapping(
                _mapping(value["event_key"], "event_key")
            ),
            value=_finite(value["value"], "draw value"),
        )


@dataclass(frozen=True, slots=True)
class DistributionMCSEEstimate:
    """One recomputable checkpoint estimate and precision decision."""

    spec_sha256: str
    draws_sha256: str
    statistic: MCSEStatistic
    estimate: float
    mcse: float
    sample_count: int
    bootstrap_replicates: int
    decision: MCSEStopDecision
    synthetic_only: bool
    schema_version: str = DISTRIBUTION_MCSE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field in ("spec_sha256", "draws_sha256"):
            value = getattr(self, field)
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                _fail(f"{field} must be a SHA-256 digest")
        if not isinstance(self.statistic, MCSEStatistic):
            _fail("estimate statistic is invalid")
        object.__setattr__(self, "estimate", _finite(self.estimate, "estimate"))
        mcse = _finite(self.mcse, "mcse")
        if mcse < 0.0:
            _fail("mcse must be non-negative")
        object.__setattr__(self, "mcse", mcse)
        _positive_int(self.sample_count, "sample_count", minimum=2)
        _positive_int(self.bootstrap_replicates, "bootstrap_replicates", minimum=2)
        if not isinstance(self.decision, MCSEStopDecision):
            _fail("decision must be an MCSEStopDecision")
        if not isinstance(self.synthetic_only, bool):
            _fail("synthetic_only must be boolean")
        if self.schema_version != DISTRIBUTION_MCSE_SCHEMA_VERSION:
            _fail("unsupported distribution MCSE estimate schema")

    def to_mapping(self) -> dict[str, object]:
        """Return the machine-readable checkpoint result."""
        return {
            "bootstrap_replicates": self.bootstrap_replicates,
            "decision": self.decision.value,
            "draws_sha256": self.draws_sha256,
            "estimate": self.estimate,
            "mcse": self.mcse,
            "sample_count": self.sample_count,
            "schema_version": self.schema_version,
            "spec_sha256": self.spec_sha256,
            "statistic": self.statistic.value,
            "synthetic_only": self.synthetic_only,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        """Strictly reconstruct one recorded checkpoint estimate."""
        _strict_fields(
            value,
            frozenset(
                {
                    "bootstrap_replicates",
                    "decision",
                    "draws_sha256",
                    "estimate",
                    "mcse",
                    "sample_count",
                    "schema_version",
                    "spec_sha256",
                    "statistic",
                    "synthetic_only",
                }
            ),
            "distribution MCSE estimate",
        )
        raw_statistic = value["statistic"]
        raw_decision = value["decision"]
        schema_version = value["schema_version"]
        if not all(
            isinstance(item, str)
            for item in (raw_statistic, raw_decision, schema_version)
        ):
            _fail("estimate enum and schema fields must be strings")
        try:
            statistic = MCSEStatistic(cast(str, raw_statistic))
            decision = MCSEStopDecision(cast(str, raw_decision))
        except ValueError as error:
            raise DistributionMCSEError(
                "unsupported MCSE estimate enum value"
            ) from error
        return cls(
            spec_sha256=cast(str, value["spec_sha256"]),
            draws_sha256=cast(str, value["draws_sha256"]),
            statistic=statistic,
            estimate=_finite(value["estimate"], "estimate"),
            mcse=_finite(value["mcse"], "mcse"),
            sample_count=_positive_int(
                value["sample_count"], "sample_count", minimum=2
            ),
            bootstrap_replicates=_positive_int(
                value["bootstrap_replicates"],
                "bootstrap_replicates",
                minimum=2,
            ),
            decision=decision,
            synthetic_only=_boolean(value["synthetic_only"], "synthetic_only"),
            schema_version=cast(str, schema_version),
        )

    @property
    def sha256(self) -> str:
        """Return the checkpoint artifact commitment."""
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))


def _statistic(values: tuple[float, ...], spec: DistributionMCSESpec) -> float:
    if spec.statistic is MCSEStatistic.MEAN:
        return fmean(values)
    if spec.statistic is MCSEStatistic.SIGN_PROBABILITY:
        return sum(value > 0.0 for value in values) / len(values)
    if spec.statistic is MCSEStatistic.QUANTILE:
        assert spec.quantile_level is not None
        return _quantile(values, spec.quantile_level)
    assert spec.tail_threshold is not None
    assert spec.tail_direction is not None
    if spec.tail_direction is TailDirection.LOWER:
        return sum(value <= spec.tail_threshold for value in values) / len(values)
    return sum(value >= spec.tail_threshold for value in values) / len(values)


def evaluate_distribution_mcse(
    spec: DistributionMCSESpec,
    draws: tuple[MonteCarloDraw, ...],
) -> DistributionMCSEEstimate:
    """Evaluate a predeclared statistic with deterministic keyed bootstrap MCSE."""
    if not isinstance(spec, DistributionMCSESpec):
        _fail("spec must be a DistributionMCSESpec")
    if (
        not isinstance(draws, tuple)
        or len(draws) < 2
        or not all(isinstance(draw, MonteCarloDraw) for draw in draws)
    ):
        _fail("draws must be an immutable tuple with at least two draws")
    ordered = tuple(sorted(draws, key=lambda draw: draw.replicate_id))
    if tuple(draw.replicate_id for draw in ordered) != tuple(range(len(ordered))):
        _fail("replicate IDs must be unique and contiguous from zero")
    if len({draw.event_key.value for draw in ordered}) != len(ordered):
        _fail("Monte Carlo event keys must be unique")
    if len(ordered) > spec.maximum_replicates:
        _fail("draw count exceeds the frozen maximum")
    values = tuple(draw.value for draw in ordered)
    rng = HashBasedRNG(spec.bootstrap_seed)
    bootstrapped: list[float] = []
    for bootstrap in range(spec.bootstrap_replicates):
        sample = tuple(
            values[
                rng.integer(
                    ordered[position].event_key,
                    0,
                    len(values) - 1,
                    domain=f"distribution_mcse_{spec.statistic.value}",
                    draw_index=bootstrap * spec.maximum_replicates + position,
                )
            ]
            for position in range(len(values))
        )
        bootstrapped.append(_statistic(sample, spec))
    bootstrap_mean = fmean(bootstrapped)
    mcse = math.sqrt(
        math.fsum((value - bootstrap_mean) ** 2 for value in bootstrapped)
        / (len(bootstrapped) - 1)
    )
    count = len(ordered)
    at_checkpoint = count >= spec.minimum_replicates and count % spec.check_every == 0
    if at_checkpoint and mcse <= spec.target_mcse:
        decision = MCSEStopDecision.TARGET_REACHED
    elif count == spec.maximum_replicates:
        decision = MCSEStopDecision.MAXIMUM_REACHED
    else:
        decision = MCSEStopDecision.CONTINUE
    return DistributionMCSEEstimate(
        spec_sha256=spec.sha256,
        draws_sha256=sha256_bytes(
            canonical_json_bytes([draw.to_mapping() for draw in ordered])
        ),
        statistic=spec.statistic,
        estimate=_statistic(values, spec),
        mcse=mcse,
        sample_count=count,
        bootstrap_replicates=spec.bootstrap_replicates,
        decision=decision,
        synthetic_only=spec.synthetic_only,
    )


__all__ = [
    "DISTRIBUTION_MCSE_SCHEMA_VERSION",
    "DistributionMCSEError",
    "DistributionMCSEEstimate",
    "DistributionMCSESpec",
    "MCSEStatistic",
    "MCSEStopDecision",
    "MonteCarloDraw",
    "TailDirection",
    "evaluate_distribution_mcse",
]
