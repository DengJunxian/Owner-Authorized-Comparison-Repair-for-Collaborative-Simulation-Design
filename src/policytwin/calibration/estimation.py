"""Pluggable posterior-estimator contracts and a deterministic ABC baseline."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from policytwin.calibration.schema import (
    CalibrationConfig,
    CalibrationError,
    CalibrationInput,
    Particle,
    PosteriorDraw,
    PosteriorSamples,
    SummaryVector,
    parameter_mapping,
)
from policytwin.counterfactual.randomness import EventKey, deterministic_int

_UNIT_MAX = (1 << 53) - 1


@runtime_checkable
class SimulatorAdapter(Protocol):
    """Typed boundary around any simulator that emits frozen summaries."""

    @property
    def simulator_id(self) -> str:
        """Return a stable implementation/configuration identifier."""
        ...

    def to_mapping(self) -> dict[str, object]:
        """Return every simulator setting in a canonical JSON-compatible form."""
        ...

    def simulate(
        self,
        parameters: Mapping[str, float],
        *,
        event_key: EventKey,
    ) -> SummaryVector:
        """Generate one full summary vector using only the explicit key."""
        ...


@dataclass(frozen=True, slots=True)
class EstimatorFit:
    """Accepted posterior support and every random event commitment."""

    backend_id: str
    particles: tuple[Particle, ...]
    event_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.backend_id, str) or not self.backend_id:
            raise CalibrationError("backend_id must be non-empty")
        if (
            not isinstance(self.particles, tuple)
            or not self.particles
            or not all(isinstance(item, Particle) for item in self.particles)
        ):
            raise CalibrationError("an estimator fit must retain posterior particles")
        if (
            not isinstance(self.event_keys, tuple)
            or not self.event_keys
            or not all(isinstance(item, str) and item for item in self.event_keys)
            or len(set(self.event_keys)) != len(self.event_keys)
        ):
            raise CalibrationError("estimator event keys must be non-empty and unique")
        simulation_keys = tuple(item.simulation_event_key for item in self.particles)
        if len(set(simulation_keys)) != len(simulation_keys):
            raise CalibrationError("estimator particles reused a simulation event key")
        if not set(simulation_keys).issubset(self.event_keys):
            raise CalibrationError(
                "estimator event manifest omits a particle simulation key"
            )


@runtime_checkable
class PosteriorEstimator(Protocol):
    """Backend contract reserved for ABC, SNPE, NRE, and future estimators."""

    @property
    def backend_id(self) -> str:
        """Return the stable backend identity."""
        ...

    def fit(
        self,
        *,
        config: CalibrationConfig,
        observed: CalibrationInput,
        simulator: SimulatorAdapter,
        namespace: str = "main",
    ) -> EstimatorFit:
        """Infer a posterior support without inspecting held-out outcomes."""
        ...


def calibration_event_key(
    config: CalibrationConfig,
    *,
    replicate: int,
    step: int,
    purpose: str,
    seed: str | None = None,
) -> EventKey:
    """Create the complete explicit key for one calibration random event."""
    if replicate < 0 or step < 0:
        raise CalibrationError("event-key indices must not be negative")
    return EventKey(
        experiment_id=config.study_id,
        seed=config.seed if seed is None else seed,
        replicate_id=replicate,
        simulation_step=step,
        purpose=f"calibration:{purpose}",
    )


def _unit_draw(key: EventKey) -> float:
    integer = deterministic_int(key, 0, _UNIT_MAX)
    return (integer + 0.5) / (_UNIT_MAX + 1.0)


def sample_prior(
    config: CalibrationConfig,
    *,
    replicate: int,
    namespace: str,
    seed: str | None = None,
) -> tuple[dict[str, float], tuple[str, ...]]:
    """Draw all parameters independently from their frozen bounded priors."""
    parameters: dict[str, float] = {}
    keys: list[str] = []
    for index, prior in enumerate(config.priors):
        key = calibration_event_key(
            config,
            replicate=replicate,
            step=index,
            purpose=f"{namespace}:prior:{prior.name}",
            seed=seed,
        )
        unit = _unit_draw(key)
        parameters[prior.name] = prior.lower + (prior.upper - prior.lower) * unit
        keys.append(key.value)
    return parameters, tuple(keys)


def summary_distance(
    config: CalibrationConfig,
    observed: SummaryVector,
    simulated: SummaryVector,
) -> float:
    """Weighted normalized distance using calibration statistics only."""
    observed_values = observed.as_dict()
    simulated_values = simulated.as_dict()
    expected = {item.name for item in config.summaries.statistics}
    if set(observed_values) != expected or set(simulated_values) != expected:
        raise CalibrationError("summary vector does not match the frozen statistic set")
    total = 0.0
    total_weight = 0.0
    for statistic in config.summaries.calibration:
        reference = observed_values[statistic.name]
        scale = max(abs(reference), 1.0)
        residual = (simulated_values[statistic.name] - reference) / scale
        total += statistic.weight * residual * residual
        total_weight += statistic.weight
    return math.sqrt(total / total_weight)


@dataclass(frozen=True, slots=True)
class ABCPosteriorEstimator:
    """Small deterministic rejection-ABC baseline for CI and recovery tests."""

    backend_id: str = "rejection_abc_v1"

    def fit(
        self,
        *,
        config: CalibrationConfig,
        observed: CalibrationInput,
        simulator: SimulatorAdapter,
        namespace: str = "main",
    ) -> EstimatorFit:
        """Keep the frozen number of lowest-distance prior particles."""
        observed.validate_for(config)
        if not isinstance(namespace, str) or not namespace:
            raise CalibrationError("namespace must be non-empty")
        expected = {item.name for item in config.summaries.statistics}
        candidates: list[Particle] = []
        event_keys: list[str] = []
        for simulation_index in range(config.simulation_count):
            parameters, prior_keys = sample_prior(
                config,
                replicate=simulation_index,
                namespace=namespace,
            )
            simulation_key = calibration_event_key(
                config,
                replicate=simulation_index,
                step=len(config.priors),
                purpose=f"{namespace}:simulator",
            )
            simulated = simulator.simulate(parameters, event_key=simulation_key)
            if set(simulated.as_dict()) != expected:
                raise CalibrationError(
                    "simulator omitted or added a frozen summary statistic"
                )
            candidates.append(
                Particle(
                    parameters=parameter_mapping(parameters),
                    distance=summary_distance(config, observed.observed, simulated),
                    simulation_event_key=simulation_key.value,
                )
            )
            event_keys.extend(prior_keys)
            event_keys.append(simulation_key.value)
        candidates.sort(
            key=lambda item: (
                item.distance,
                item.parameters,
                item.simulation_event_key,
            )
        )
        return EstimatorFit(
            backend_id=self.backend_id,
            particles=tuple(candidates[: config.accepted_particles]),
            event_keys=tuple(event_keys),
        )


def sample_posterior(
    *,
    config: CalibrationConfig,
    fit: EstimatorFit,
    draw_count: int | None = None,
    namespace: str = "main",
) -> tuple[PosteriorSamples, tuple[str, ...]]:
    """Keyed empirical resampling from all accepted ABC particles."""
    expected_names = {prior.name for prior in config.priors}
    priors = {prior.name: prior for prior in config.priors}
    for particle in fit.particles:
        parameters = dict(particle.parameters)
        if set(parameters) != expected_names:
            raise CalibrationError(
                "posterior particle parameters do not match the frozen priors"
            )
        for name, value in parameters.items():
            prior = priors[name]
            if not prior.lower <= value <= prior.upper:
                raise CalibrationError(
                    f"posterior particle lies outside frozen prior support: {name}"
                )
    count = config.posterior_draws if draw_count is None else draw_count
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise CalibrationError("posterior draw count must be positive")
    draws: list[PosteriorDraw] = []
    keys: list[str] = []
    for index in range(count):
        key = calibration_event_key(
            config,
            replicate=index,
            step=0,
            purpose=f"{namespace}:posterior_resample",
        )
        selected = deterministic_int(key, 0, len(fit.particles) - 1)
        draws.append(
            PosteriorDraw(
                parameters=fit.particles[selected].parameters,
                source_particle=selected,
                event_key=key.value,
            )
        )
        keys.append(key.value)
    return (
        PosteriorSamples(
            backend_id=fit.backend_id,
            particles=fit.particles,
            draws=tuple(draws),
        ),
        tuple(keys),
    )


__all__ = [
    "ABCPosteriorEstimator",
    "EstimatorFit",
    "PosteriorEstimator",
    "SimulatorAdapter",
    "calibration_event_key",
    "sample_posterior",
    "sample_prior",
    "summary_distance",
]
