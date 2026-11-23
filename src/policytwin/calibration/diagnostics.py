"""Posterior predictive, recovery, and identifiability diagnostics."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean

from policytwin.calibration.estimation import SimulatorAdapter, calibration_event_key
from policytwin.calibration.schema import (
    CalibrationConfig,
    CalibrationError,
    CalibrationInput,
    PosteriorSamples,
    PriorSpec,
    SummaryRole,
)
from policytwin.provenance import canonical_json_bytes, sha256_bytes


def quantile(values: Sequence[float], probability: float) -> float:
    """Return a deterministic linearly interpolated empirical quantile."""
    if not values:
        raise CalibrationError("a quantile requires at least one value")
    if not 0.0 <= probability <= 1.0:
        raise CalibrationError("quantile probability must lie in [0, 1]")
    ordered = sorted(float(item) for item in values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    lower_value = ordered[lower]
    upper_value = ordered[upper]
    # The weighted-sum form can move an interval endpoint by one ULP even
    # when both adjacent observations are bit-identical.  Interpolating the
    # difference preserves that exact fixed point and clipping keeps the
    # binary64 result inside its ordered bracket.
    interpolated = lower_value + (upper_value - lower_value) * fraction
    return min(upper_value, max(lower_value, interpolated))


def _sample_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = fmean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _correlation(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_mean = fmean(left)
    right_mean = fmean(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True)
    )
    left_scale = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_scale = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    if left_scale == 0.0 or right_scale == 0.0:
        return 0.0
    return numerator / (left_scale * right_scale)


def _histogram(values: Sequence[float], prior: PriorSpec, bins: int) -> tuple[int, ...]:
    counts = [0] * bins
    width = prior.upper - prior.lower
    for value in values:
        relative = (value - prior.lower) / width
        index = min(bins - 1, max(0, int(relative * bins)))
        counts[index] += 1
    return tuple(counts)


def _mode_count(counts: Sequence[int]) -> int:
    """Count separated occupied local maxima in a fixed frozen histogram."""
    if not counts or max(counts) == 0:
        return 0
    minimum_peak = max(2, math.ceil(sum(counts) * 0.08))
    peaks = 0
    for index, count in enumerate(counts):
        left = counts[index - 1] if index > 0 else -1
        right = counts[index + 1] if index + 1 < len(counts) else -1
        if count >= minimum_peak and count > left and count >= right:
            peaks += 1
    return peaks


@dataclass(frozen=True, slots=True)
class PosteriorPredictiveStatistic:
    """Full predictive draws and frozen 90% interval for one statistic."""

    name: str
    role: SummaryRole
    observed: float
    predictive_draws: tuple[float, ...]
    lower_90: float
    median: float
    upper_90: float
    covered: bool

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise CalibrationError("PPC statistic name must be non-empty")
        if not isinstance(self.role, SummaryRole):
            raise CalibrationError("PPC statistic role is invalid")
        if not self.predictive_draws or not all(
            math.isfinite(value) for value in self.predictive_draws
        ):
            raise CalibrationError("PPC predictive draws must be finite and non-empty")
        if not all(
            math.isfinite(value)
            for value in (self.observed, self.lower_90, self.median, self.upper_90)
        ):
            raise CalibrationError("PPC statistic values must be finite")
        if not self.lower_90 <= self.median <= self.upper_90:
            raise CalibrationError("PPC interval ordering is invalid")
        if self.covered is not (self.lower_90 <= self.observed <= self.upper_90):
            raise CalibrationError("PPC coverage flag does not match its interval")

    def to_mapping(self) -> dict[str, object]:
        return {
            "covered": self.covered,
            "lower_90": self.lower_90,
            "median": self.median,
            "name": self.name,
            "observed": self.observed,
            "predictive_draws": list(self.predictive_draws),
            "role": self.role.value,
            "upper_90": self.upper_90,
        }


@dataclass(frozen=True, slots=True)
class PosteriorPredictiveChecks:
    """PPC results retaining calibration and held-out roles separately."""

    statistics: tuple[PosteriorPredictiveStatistic, ...]
    event_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.statistics, tuple)
            or not self.statistics
            or not all(
                isinstance(item, PosteriorPredictiveStatistic)
                for item in self.statistics
            )
        ):
            raise CalibrationError("PPC must retain typed statistic outputs")
        names = tuple(item.name for item in self.statistics)
        if len(set(names)) != len(names):
            raise CalibrationError("PPC statistic names must be unique")
        if (
            not isinstance(self.event_keys, tuple)
            or not self.event_keys
            or not all(isinstance(item, str) and item for item in self.event_keys)
            or len(set(self.event_keys)) != len(self.event_keys)
        ):
            raise CalibrationError("PPC event keys must be non-empty and unique")
        if any(
            len(item.predictive_draws) != len(self.event_keys)
            for item in self.statistics
        ):
            raise CalibrationError("PPC draw counts do not match simulator event keys")

    def to_mapping(self) -> dict[str, object]:
        return {
            "event_keys": list(self.event_keys),
            "statistics": [item.to_mapping() for item in self.statistics],
        }

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))


def posterior_predictive_checks(
    *,
    config: CalibrationConfig,
    observed: CalibrationInput,
    posterior: PosteriorSamples,
    simulator: SimulatorAdapter,
    namespace: str = "main",
) -> PosteriorPredictiveChecks:
    """Simulate full posterior summaries without feeding held-out values into fit."""
    observed.validate_for(config)
    predictions: dict[str, list[float]] = {
        item.name: [] for item in config.summaries.statistics
    }
    event_keys: list[str] = []
    for index in range(config.ppc_draws):
        posterior_draw = posterior.draws[index % len(posterior.draws)]
        key = calibration_event_key(
            config,
            replicate=index,
            step=0,
            purpose=f"{namespace}:ppc_simulator",
        )
        result = simulator.simulate(dict(posterior_draw.parameters), event_key=key)
        result_values = result.as_dict()
        if set(result_values) != set(predictions):
            raise CalibrationError("PPC simulator output changed the frozen summaries")
        for name, value in result_values.items():
            predictions[name].append(value)
        event_keys.append(key.value)
    observed_values = observed.observed.as_dict()
    specs = {item.name: item for item in config.summaries.statistics}
    statistics: list[PosteriorPredictiveStatistic] = []
    for name in sorted(predictions):
        draws = tuple(predictions[name])
        lower = quantile(draws, 0.05)
        median = quantile(draws, 0.5)
        upper = quantile(draws, 0.95)
        value = observed_values[name]
        statistics.append(
            PosteriorPredictiveStatistic(
                name=name,
                role=specs[name].role,
                observed=value,
                predictive_draws=draws,
                lower_90=lower,
                median=median,
                upper_90=upper,
                covered=lower <= value <= upper,
            )
        )
    return PosteriorPredictiveChecks(tuple(statistics), tuple(event_keys))


@dataclass(frozen=True, slots=True)
class RecoveryMetric:
    """Synthetic-only posterior recovery for one known true parameter."""

    parameter: str
    truth: float
    posterior_median: float
    absolute_error: float
    lower_90: float
    upper_90: float
    covered: bool

    def __post_init__(self) -> None:
        if not isinstance(self.parameter, str) or not self.parameter:
            raise CalibrationError("recovery parameter name must be non-empty")
        if not all(
            math.isfinite(value)
            for value in (
                self.truth,
                self.posterior_median,
                self.absolute_error,
                self.lower_90,
                self.upper_90,
            )
        ):
            raise CalibrationError("recovery outputs must be finite")
        if self.absolute_error < 0.0 or self.lower_90 > self.upper_90:
            raise CalibrationError("recovery error or interval is invalid")
        if not math.isclose(
            self.absolute_error,
            abs(self.posterior_median - self.truth),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise CalibrationError("recovery absolute error is inconsistent")
        if self.covered is not (self.lower_90 <= self.truth <= self.upper_90):
            raise CalibrationError("recovery coverage flag is inconsistent")

    def to_mapping(self) -> dict[str, object]:
        return {
            "absolute_error": self.absolute_error,
            "covered": self.covered,
            "lower_90": self.lower_90,
            "parameter": self.parameter,
            "posterior_median": self.posterior_median,
            "truth": self.truth,
            "upper_90": self.upper_90,
        }


def synthetic_recovery(
    config: CalibrationConfig,
    posterior: PosteriorSamples,
    true_parameters: Mapping[str, float],
) -> tuple[RecoveryMetric, ...]:
    """Compare the posterior distribution with synthetic generating values."""
    expected = {item.name for item in config.priors}
    if set(true_parameters) != expected:
        raise CalibrationError("true parameters do not match the frozen priors")
    metrics: list[RecoveryMetric] = []
    for prior in config.priors:
        truth = float(true_parameters[prior.name])
        if not prior.lower <= truth <= prior.upper:
            raise CalibrationError(f"true parameter lies outside prior: {prior.name}")
        draws = [dict(item.parameters)[prior.name] for item in posterior.draws]
        median = quantile(draws, 0.5)
        lower = quantile(draws, 0.05)
        upper = quantile(draws, 0.95)
        metrics.append(
            RecoveryMetric(
                parameter=prior.name,
                truth=truth,
                posterior_median=median,
                absolute_error=abs(median - truth),
                lower_90=lower,
                upper_90=upper,
                covered=lower <= truth <= upper,
            )
        )
    return tuple(metrics)


@dataclass(frozen=True, slots=True)
class ParameterIdentifiability:
    """Posterior contraction, boundary mass, modality, and confounding flags."""

    parameter: str
    posterior_standard_deviation: float
    prior_standard_deviation: float
    contraction: float
    lower_boundary_mass: float
    upper_boundary_mass: float
    boundary_pileup: bool
    histogram_counts: tuple[int, ...]
    mode_count: int
    multimodal: bool
    maximum_absolute_correlation: float
    confounded: bool
    estimable_for_reporting: bool

    def to_mapping(self) -> dict[str, object]:
        return {
            "boundary_pileup": self.boundary_pileup,
            "confounded": self.confounded,
            "contraction": self.contraction,
            "estimable_for_reporting": self.estimable_for_reporting,
            "histogram_counts": list(self.histogram_counts),
            "lower_boundary_mass": self.lower_boundary_mass,
            "maximum_absolute_correlation": self.maximum_absolute_correlation,
            "mode_count": self.mode_count,
            "multimodal": self.multimodal,
            "parameter": self.parameter,
            "posterior_standard_deviation": self.posterior_standard_deviation,
            "prior_standard_deviation": self.prior_standard_deviation,
            "upper_boundary_mass": self.upper_boundary_mass,
        }


def identifiability_diagnostics(
    config: CalibrationConfig,
    posterior: PosteriorSamples,
) -> tuple[ParameterIdentifiability, ...]:
    """Diagnose weak identification instead of reporting false precision."""
    by_name = {
        prior.name: [dict(draw.parameters)[prior.name] for draw in posterior.draws]
        for prior in config.priors
    }
    results: list[ParameterIdentifiability] = []
    thresholds = config.identifiability
    for prior in config.priors:
        values = by_name[prior.name]
        posterior_std = _sample_std(values)
        contraction = 1.0 - posterior_std / prior.standard_deviation
        width = prior.upper - prior.lower
        boundary_width = width * thresholds.boundary_fraction
        lower_mass = fmean(value <= prior.lower + boundary_width for value in values)
        upper_mass = fmean(value >= prior.upper - boundary_width for value in values)
        boundary = max(lower_mass, upper_mass) >= thresholds.boundary_mass_threshold
        counts = _histogram(values, prior, thresholds.histogram_bins)
        modes = _mode_count(counts)
        multimodal = modes > 1
        correlations = [
            abs(_correlation(values, other_values))
            for name, other_values in by_name.items()
            if name != prior.name
        ]
        maximum_correlation = max(correlations, default=0.0)
        confounded = maximum_correlation >= thresholds.correlation_threshold
        estimable = (
            contraction >= thresholds.minimum_contraction
            and not boundary
            and not multimodal
            and not confounded
        )
        results.append(
            ParameterIdentifiability(
                parameter=prior.name,
                posterior_standard_deviation=posterior_std,
                prior_standard_deviation=prior.standard_deviation,
                contraction=contraction,
                lower_boundary_mass=lower_mass,
                upper_boundary_mass=upper_mass,
                boundary_pileup=boundary,
                histogram_counts=counts,
                mode_count=modes,
                multimodal=multimodal,
                maximum_absolute_correlation=maximum_correlation,
                confounded=confounded,
                estimable_for_reporting=estimable,
            )
        )
    return tuple(results)


__all__ = [
    "ParameterIdentifiability",
    "PosteriorPredictiveChecks",
    "PosteriorPredictiveStatistic",
    "RecoveryMetric",
    "identifiability_diagnostics",
    "posterior_predictive_checks",
    "quantile",
    "synthetic_recovery",
]
