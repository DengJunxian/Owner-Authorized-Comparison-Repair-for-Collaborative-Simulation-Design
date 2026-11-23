"""Synthetic-only joint recovery diagnostics for ``BehaviorParameters``.

The benchmark exercises the live fast-order and cognitive-update mechanisms.
It is deliberately a small CI diagnostic, not evidence that behavioral
parameters are identified in market data.  In particular, the calibration
moments contain no lagged-memory probe, so ``memory_decay`` should remain weakly
identified; independent lagged probes are reserved for held-out validation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from policytwin.agents import (
    AgentArchetype,
    FastOrderPolicy,
    OrderPolicyConfig,
    OwnPortfolioState,
    PublicMarketObservation,
    PublicSignal,
    RiskLimits,
    SignalKind,
)
from policytwin.behavior import BehaviorMechanisms, BehaviorParameters
from policytwin.calibration.diagnostics import (
    ParameterIdentifiability,
    PosteriorPredictiveChecks,
    RecoveryMetric,
    identifiability_diagnostics,
    posterior_predictive_checks,
    synthetic_recovery,
)
from policytwin.calibration.estimation import (
    ABCPosteriorEstimator,
    PosteriorEstimator,
    calibration_event_key,
    sample_posterior,
)
from policytwin.calibration.schema import (
    CalibrationConfig,
    CalibrationError,
    CalibrationInput,
    IdentifiabilityThresholds,
    ParameterTransform,
    PosteriorSamples,
    PriorSpec,
    SummaryLevel,
    SummaryRole,
    SummaryStatisticsConfig,
    SummaryStatisticSpec,
    SummaryVector,
    TransformKind,
)
from policytwin.calibration.synthetic import (
    synthetic_calibration_input,
    synthetic_source_design_sha256,
)
from policytwin.calibration.workflow import (
    SimulationBasedCalibration,
    run_simulation_based_calibration,
)
from policytwin.cognition import CognitiveState, CognitiveStateUpdater
from policytwin.counterfactual import EventKey, deterministic_int
from policytwin.provenance import canonical_json_bytes, sha256_bytes
from policytwin.social import VisibleOrderFlow

BEHAVIOR_RECOVERY_SIMULATOR_ID = "synthetic_behavior_probe_v1"
BEHAVIOR_PARAMETER_NAMES = (
    "risk_aversion",
    "loss_aversion",
    "disposition_strength",
    "overconfidence",
    "herding",
    "attention_sensitivity",
    "recency_weight",
    "memory_decay",
    "inventory_aversion",
)

_SUPPORTS: dict[str, tuple[float, float]] = {
    "risk_aversion": (0.0, 2.0),
    "loss_aversion": (0.0, 2.5),
    "disposition_strength": (0.0, 2.0),
    "overconfidence": (0.0, 1.5),
    "herding": (0.0, 1.5),
    "attention_sensitivity": (0.0, 1.5),
    "recency_weight": (0.05, 0.95),
    "memory_decay": (0.01, 0.60),
    "inventory_aversion": (0.0, 2.5),
}
_DECISION_TIME = datetime(2024, 12, 20, tzinfo=UTC)
_CUTOFF = datetime(2025, 1, 1, tzinfo=UTC)
_AGENT_ID = "synthetic-behavior-probe"
_UNIT_MAX = (1 << 53) - 1


def _derived_key(parent: EventKey, purpose: str) -> EventKey:
    return EventKey(
        experiment_id=parent.experiment_id,
        seed=parent.seed,
        replicate_id=parent.replicate_id,
        simulation_step=parent.simulation_step,
        purpose=f"{parent.purpose}:behavior_probe:{purpose}",
    )


def _signal(kind: SignalKind, value: float, suffix: str) -> PublicSignal:
    return PublicSignal(
        kind=kind,
        value=value,
        event_time=_DECISION_TIME - timedelta(minutes=2),
        available_time=_DECISION_TIME - timedelta(minutes=1),
        source="policytwin.calibration.synthetic_behavior_probe",
        signal_id=f"{kind.value}-{suffix}",
    )


def _observation(
    suffix: str,
    *,
    fundamental: float = 0.0,
    flow: float = 0.0,
    attention: float = 0.0,
    volatility: float = 0.0,
) -> PublicMarketObservation:
    return PublicMarketObservation(
        as_of_time=_DECISION_TIME,
        cutoff=_CUTOFF,
        mid_price=10_000,
        best_bid=9_990,
        best_ask=10_010,
        signals=tuple(
            _signal(kind, value, suffix)
            for kind, value in (
                (SignalKind.NOISE, 0.0),
                (SignalKind.FUNDAMENTAL, fundamental),
                (SignalKind.TREND, fundamental),
                (SignalKind.EXECUTION_URGENCY, 0.0),
                (SignalKind.ATTENTION, attention),
                (SignalKind.VOLATILITY, volatility),
                (SignalKind.ARBITRAGE, 0.0),
            )
        ),
        visible_order_flow=VisibleOrderFlow(
            net_flow=flow,
            sample_size=100,
            event_time=_DECISION_TIME - timedelta(minutes=2),
            available_time=_DECISION_TIME - timedelta(minutes=1),
            source="policytwin.calibration.synthetic_behavior_probe",
            aggregation_id=f"flow-{suffix}",
        ),
    )


def _portfolio(
    *, inventory: int = 1_000, average_cost: int = 10_000
) -> OwnPortfolioState:
    return OwnPortfolioState(
        agent_id=_AGENT_ID,
        cash=1_000_000_000_000,
        inventory=inventory,
        reserved_cash=0,
        reserved_inventory=0,
        average_cost=average_cost,
    )


def _unit_noise(event_key: EventKey, name: str) -> float:
    key = _derived_key(event_key, f"summary_noise:{name}")
    draw = (deterministic_int(key, 0, _UNIT_MAX) + 0.5) / (_UNIT_MAX + 1.0)
    return 2.0 * draw - 1.0


@dataclass(frozen=True, slots=True)
class SyntheticBehaviorProbeSimulator:
    """Live-mechanism probes for all nine continuous behavior parameters."""

    noise_scale: float = 0.001

    def __post_init__(self) -> None:
        if (
            isinstance(self.noise_scale, bool)
            or not isinstance(self.noise_scale, int | float)
            or not math.isfinite(self.noise_scale)
            or self.noise_scale < 0.0
        ):
            raise CalibrationError("behavior probe noise_scale must be non-negative")
        object.__setattr__(self, "noise_scale", float(self.noise_scale))

    @property
    def simulator_id(self) -> str:
        return BEHAVIOR_RECOVERY_SIMULATOR_ID

    def to_mapping(self) -> dict[str, object]:
        return {
            "calibration_memory_design": "current_signal_only",
            "held_out_memory_design": "independent_short_and_long_lags",
            "noise_scale": self.noise_scale,
            "probe_implementation": "FastOrderPolicy+CognitiveStateUpdater",
            "schema_version": 1,
            "simulator_id": self.simulator_id,
            "synthetic_only": True,
        }

    def _parameters(self, values: Mapping[str, float]) -> BehaviorParameters:
        if set(values) != set(BEHAVIOR_PARAMETER_NAMES):
            raise CalibrationError(
                "behavior probe requires exactly the nine BehaviorParameters fields"
            )
        normalized: dict[str, float] = {}
        for name in BEHAVIOR_PARAMETER_NAMES:
            value = values[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
            ):
                raise CalibrationError(f"behavior parameter {name} must be finite")
            number = float(value)
            lower, upper = _SUPPORTS[name]
            if not lower <= number <= upper:
                raise CalibrationError(
                    f"behavior parameter {name} lies outside probe support"
                )
            normalized[name] = number
        return BehaviorParameters(**normalized)

    @staticmethod
    def _trace(
        parameters: BehaviorParameters,
        *,
        event_key: EventKey,
        suffix: str,
        archetype: AgentArchetype = AgentArchetype.FUNDAMENTAL,
        observation: PublicMarketObservation | None = None,
        portfolio: OwnPortfolioState | None = None,
        state: CognitiveState | None = None,
    ) -> tuple[float, float, float]:
        trace = FastOrderPolicy().decide(
            agent_id=_AGENT_ID,
            archetype=archetype,
            parameters=parameters,
            mechanisms=BehaviorMechanisms(),
            cognitive_state=CognitiveState() if state is None else state,
            portfolio=_portfolio() if portfolio is None else portfolio,
            observation=(_observation(suffix) if observation is None else observation),
            trusted_cutoff=_CUTOFF,
            event_key=_derived_key(event_key, f"decision:{suffix}"),
            config=OrderPolicyConfig(
                tick_size=10,
                base_order_quantity=1,
                temperature=1.0,
            ),
            risk_limits=RiskLimits(
                max_order_quantity=10,
                max_order_notional=10_000_000,
                max_inventory=1_000_000,
                max_open_orders=10,
            ),
        )
        probabilities = trace.action_probabilities
        return probabilities.buy, probabilities.hold, probabilities.sell

    @staticmethod
    def _memory(
        parameters: BehaviorParameters,
        *,
        initial_memory: float,
        observed_signal: float,
        elapsed_steps: int,
    ) -> float:
        old_time = _DECISION_TIME - timedelta(minutes=10)
        state = CognitiveState(
            belief=initial_memory,
            signal_memory=initial_memory,
            last_update_time=old_time,
            observation_count=1,
            last_event_time=old_time,
            last_available_time=old_time,
        )
        updated = CognitiveStateUpdater(
            parameters=parameters,
            mechanisms=BehaviorMechanisms(),
        ).update(
            state,
            observed_signal=observed_signal,
            event_time=_DECISION_TIME - timedelta(minutes=2),
            available_time=_DECISION_TIME - timedelta(minutes=1),
            as_of_time=_DECISION_TIME,
            trusted_cutoff=_CUTOFF,
            elapsed_steps=elapsed_steps,
        )
        return updated.signal_memory

    def simulate(
        self,
        parameters: Mapping[str, float],
        *,
        event_key: EventKey,
    ) -> SummaryVector:
        """Generate registered synthetic probes using only the explicit key."""
        if not isinstance(event_key, EventKey):
            raise CalibrationError("behavior probe event_key must be an EventKey")
        values = self._parameters(parameters)

        def trace(
            suffix: str,
            *,
            archetype: AgentArchetype = AgentArchetype.FUNDAMENTAL,
            observation: PublicMarketObservation | None = None,
            portfolio: OwnPortfolioState | None = None,
        ) -> tuple[float, float, float]:
            return self._trace(
                values,
                event_key=event_key,
                suffix=suffix,
                archetype=archetype,
                observation=observation,
                portfolio=portfolio,
            )

        flat = trace("cal-flat")
        risk = trace(
            "cal-risk",
            observation=_observation("cal-risk", volatility=0.8),
        )
        loss = trace(
            "cal-loss",
            portfolio=_portfolio(average_cost=11_112),
        )
        gain = trace(
            "cal-gain",
            portfolio=_portfolio(average_cost=9_090),
        )
        signal = trace(
            "cal-signal",
            observation=_observation("cal-signal", fundamental=0.6),
        )
        herd = trace(
            "cal-herd",
            observation=_observation("cal-herd", flow=0.8),
        )
        attention_zero = trace(
            "cal-attention-zero",
            observation=_observation(
                "cal-attention-zero", fundamental=0.6, attention=0.0
            ),
        )
        attention_high = trace(
            "cal-attention-high",
            observation=_observation(
                "cal-attention-high", fundamental=0.6, attention=1.0
            ),
        )
        inventory_low = trace(
            "cal-inventory-low",
            archetype=AgentArchetype.MARKET_MAKER,
            portfolio=_portfolio(inventory=1_000),
        )
        inventory_high = trace(
            "cal-inventory-high",
            archetype=AgentArchetype.MARKET_MAKER,
            portfolio=_portfolio(inventory=700_000),
        )

        held_flat = trace("held-flat")
        held_risk = trace(
            "held-risk",
            observation=_observation("held-risk", volatility=0.45),
        )
        held_loss = trace(
            "held-loss",
            portfolio=_portfolio(average_cost=10_527),
        )
        held_gain = trace(
            "held-gain",
            portfolio=_portfolio(average_cost=9_523),
        )
        held_signal = trace(
            "held-signal",
            observation=_observation("held-signal", fundamental=-0.4),
        )
        held_herd = trace(
            "held-herd",
            observation=_observation("held-herd", flow=-0.55),
        )
        held_attention_zero = trace(
            "held-attention-zero",
            observation=_observation(
                "held-attention-zero", fundamental=-0.4, attention=0.0
            ),
        )
        held_attention_high = trace(
            "held-attention-high",
            observation=_observation(
                "held-attention-high", fundamental=-0.4, attention=0.7
            ),
        )
        held_inventory_low = trace(
            "held-inventory-low",
            archetype=AgentArchetype.MARKET_MAKER,
            portfolio=_portfolio(inventory=100_000),
        )
        held_inventory_high = trace(
            "held-inventory-high",
            archetype=AgentArchetype.MARKET_MAKER,
            portfolio=_portfolio(inventory=500_000),
        )

        summaries = {
            "macro_current_memory": self._memory(
                values,
                initial_memory=0.0,
                observed_signal=0.8,
                elapsed_steps=4,
            ),
            "macro_risk_hold_delta": risk[1] - flat[1],
            "meso_attention_direction_delta": (attention_high[0] - attention_high[2])
            - (attention_zero[0] - attention_zero[2]),
            "meso_herding_direction_delta": (herd[0] - herd[2]) - (flat[0] - flat[2]),
            "meso_inventory_reduction_delta": (inventory_high[2] - inventory_high[0])
            - (inventory_low[2] - inventory_low[0]),
            "micro_disposition_sell_delta": gain[2] - flat[2],
            "micro_loss_hold_delta": loss[1] - flat[1],
            "micro_overconfidence_turnover_delta": (1.0 - signal[1]) - (1.0 - flat[1]),
            "held_macro_memory_long_lag": self._memory(
                values,
                initial_memory=0.8,
                observed_signal=0.0,
                elapsed_steps=8,
            ),
            "held_macro_memory_short_lag": self._memory(
                values,
                initial_memory=0.8,
                observed_signal=0.0,
                elapsed_steps=2,
            ),
            "held_macro_risk_hold_delta": held_risk[1] - held_flat[1],
            "held_meso_attention_direction_delta": (
                held_attention_high[0] - held_attention_high[2]
            )
            - (held_attention_zero[0] - held_attention_zero[2]),
            "held_meso_herding_direction_delta": (held_herd[0] - held_herd[2])
            - (held_flat[0] - held_flat[2]),
            "held_meso_inventory_reduction_delta": (
                held_inventory_high[2] - held_inventory_high[0]
            )
            - (held_inventory_low[2] - held_inventory_low[0]),
            "held_micro_disposition_sell_delta": held_gain[2] - held_flat[2],
            "held_micro_loss_hold_delta": held_loss[1] - held_flat[1],
            "held_micro_overconfidence_direction_delta": (
                held_signal[0] - held_signal[2]
            )
            - (held_flat[0] - held_flat[2]),
        }
        return SummaryVector(
            tuple(
                (
                    name,
                    value + self.noise_scale * _unit_noise(event_key, name),
                )
                for name, value in summaries.items()
            )
        )


def behavior_recovery_config(
    *,
    seed: str = "synthetic-behavior-recovery-v1",
    simulation_count: int = 1_200,
    accepted_particles: int = 60,
    posterior_draws: int = 240,
    ppc_draws: int = 60,
    sbc_replicates: int = 2,
    sbc_posterior_draws: int = 60,
) -> CalibrationConfig:
    """Build the frozen nine-parameter synthetic CI configuration."""

    calibration_levels = {
        "macro_current_memory": SummaryLevel.MACRO,
        "macro_risk_hold_delta": SummaryLevel.MACRO,
        "meso_attention_direction_delta": SummaryLevel.MESO,
        "meso_herding_direction_delta": SummaryLevel.MESO,
        "meso_inventory_reduction_delta": SummaryLevel.MESO,
        "micro_disposition_sell_delta": SummaryLevel.MICRO,
        "micro_loss_hold_delta": SummaryLevel.MICRO,
        "micro_overconfidence_turnover_delta": SummaryLevel.MICRO,
    }
    held_out_levels = {
        "held_macro_memory_long_lag": SummaryLevel.MACRO,
        "held_macro_memory_short_lag": SummaryLevel.MACRO,
        "held_macro_risk_hold_delta": SummaryLevel.MACRO,
        "held_meso_attention_direction_delta": SummaryLevel.MESO,
        "held_meso_herding_direction_delta": SummaryLevel.MESO,
        "held_meso_inventory_reduction_delta": SummaryLevel.MESO,
        "held_micro_disposition_sell_delta": SummaryLevel.MICRO,
        "held_micro_loss_hold_delta": SummaryLevel.MICRO,
        "held_micro_overconfidence_direction_delta": SummaryLevel.MICRO,
    }

    def statistic(
        name: str,
        level: SummaryLevel,
        role: SummaryRole,
    ) -> SummaryStatisticSpec:
        return SummaryStatisticSpec(
            name=name,
            level=level,
            role=role,
            definition_sha256=sha256_bytes(
                canonical_json_bytes(
                    {
                        "name": name,
                        "probe_implementation": (
                            "FastOrderPolicy+CognitiveStateUpdater"
                        ),
                        "simulator_id": BEHAVIOR_RECOVERY_SIMULATOR_ID,
                        "version": 1,
                    }
                )
            ),
            source_design_sha256=synthetic_source_design_sha256(role),
        )

    priors = tuple(
        PriorSpec(
            name=name,
            lower=_SUPPORTS[name][0],
            upper=_SUPPORTS[name][1],
            transform=(
                ParameterTransform(TransformKind.LOGIT)
                if name in {"recency_weight", "memory_decay"}
                else ParameterTransform()
            ),
        )
        for name in BEHAVIOR_PARAMETER_NAMES
    )
    statistics = tuple(
        statistic(name, level, SummaryRole.CALIBRATION)
        for name, level in calibration_levels.items()
    ) + tuple(
        statistic(name, level, SummaryRole.HELD_OUT_VALIDATION)
        for name, level in held_out_levels.items()
    )
    return CalibrationConfig(
        schema_version=2,
        study_id="synthetic-behavior-parameter-recovery",
        cutoff=_CUTOFF,
        seed=seed,
        priors=priors,
        summaries=SummaryStatisticsConfig(statistics),
        simulation_count=simulation_count,
        accepted_particles=accepted_particles,
        posterior_draws=posterior_draws,
        ppc_draws=ppc_draws,
        sbc_replicates=sbc_replicates,
        sbc_posterior_draws=sbc_posterior_draws,
        identifiability=IdentifiabilityThresholds(
            minimum_contraction=0.25,
            boundary_fraction=0.15,
            boundary_mass_threshold=0.35,
            correlation_threshold=0.85,
            histogram_bins=8,
        ),
        synthetic_only=True,
    )


def default_behavior_recovery_truth() -> BehaviorParameters:
    """Return the preregistered interior generating point for the CI fixture."""
    return BehaviorParameters(
        risk_aversion=0.8,
        loss_aversion=1.4,
        disposition_strength=0.7,
        overconfidence=0.6,
        herding=0.55,
        attention_sensitivity=0.65,
        recency_weight=0.58,
        memory_decay=0.22,
        inventory_aversion=1.1,
    )


@dataclass(frozen=True, slots=True)
class BehaviorRecoveryBenchmarkRun:
    """Machine-readable result of the synthetic nine-parameter CI diagnostic."""

    config: CalibrationConfig
    observed: CalibrationInput
    simulator_config_sha256: str
    truth: BehaviorParameters
    posterior: PosteriorSamples
    ppc: PosteriorPredictiveChecks
    recovery: tuple[RecoveryMetric, ...]
    sbc: SimulationBasedCalibration
    identifiability: tuple[ParameterIdentifiability, ...]
    event_keys: tuple[str, ...]
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.config, CalibrationConfig):
            raise CalibrationError("behavior benchmark config is invalid")
        if not isinstance(self.observed, CalibrationInput):
            raise CalibrationError("behavior benchmark input is invalid")
        if not isinstance(self.truth, BehaviorParameters):
            raise CalibrationError("behavior benchmark truth is invalid")
        self.observed.validate_for(self.config)
        expected = set(BEHAVIOR_PARAMETER_NAMES)
        if {prior.name for prior in self.config.priors} != expected:
            raise CalibrationError("behavior benchmark config changed parameter set")
        if {item.parameter for item in self.recovery} != expected:
            raise CalibrationError("behavior recovery omitted a parameter")
        if {item.parameter for item in self.identifiability} != expected:
            raise CalibrationError("behavior identifiability omitted a parameter")
        if {item.parameter for item in self.sbc.parameters} != expected:
            raise CalibrationError("behavior SBC omitted a parameter")
        if len(set(self.event_keys)) != len(self.event_keys):
            raise CalibrationError("behavior benchmark reused an EventKey")
        if len(self.simulator_config_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.simulator_config_sha256
        ):
            raise CalibrationError("simulator_config_sha256 must be a SHA-256")
        if not any("SYNTHETIC ONLY" in warning for warning in self.warnings):
            raise CalibrationError("behavior benchmark must retain its claim warning")

    def to_mapping(self) -> dict[str, object]:
        return {
            "authorization_status": "ci_diagnostic_not_research_authorized",
            "calibration_statistics": [
                item.name for item in self.config.summaries.calibration
            ],
            "claim_scope": "synthetic_behavior_recovery_mechanics_only",
            "completion_status": "synthetic_ci_diagnostic_completed",
            "config_sha256": self.config.sha256,
            "event_key_count": len(self.event_keys),
            "event_keys_sha256": sha256_bytes(
                canonical_json_bytes(list(self.event_keys))
            ),
            "held_out_statistics": [
                item.name for item in self.config.summaries.held_out
            ],
            "identifiability": [item.to_mapping() for item in self.identifiability],
            "input_sha256": self.observed.sha256,
            "parameter_names": list(BEHAVIOR_PARAMETER_NAMES),
            "posterior_predictive_checks_sha256": self.ppc.sha256,
            "posterior_sha256": self.posterior.sha256,
            "rank_histogram_sha256": self.sbc.sha256,
            "real_data_status": "blocked_not_run",
            "recovery": [item.to_mapping() for item in self.recovery],
            "sbc_interval_coverage_90": {
                item.parameter: item.interval_coverage_90
                for item in self.sbc.parameters
            },
            "schema_version": 1,
            "simulator_config_sha256": self.simulator_config_sha256,
            "simulator_id": BEHAVIOR_RECOVERY_SIMULATOR_ID,
            "synthetic_only": True,
            "truth": self.truth.to_mapping(),
            "warnings": list(self.warnings),
        }

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))


def run_behavior_recovery_benchmark(
    *,
    config: CalibrationConfig | None = None,
    simulator: SyntheticBehaviorProbeSimulator | None = None,
    truth: BehaviorParameters | None = None,
    estimator: PosteriorEstimator | None = None,
) -> BehaviorRecoveryBenchmarkRun:
    """Run joint synthetic recovery, PPC, SBC, and identifiability diagnostics."""
    frozen_config = behavior_recovery_config() if config is None else config
    probe = SyntheticBehaviorProbeSimulator() if simulator is None else simulator
    generating_truth = default_behavior_recovery_truth() if truth is None else truth
    if not isinstance(frozen_config, CalibrationConfig):
        raise CalibrationError("behavior recovery config must be a CalibrationConfig")
    if not isinstance(probe, SyntheticBehaviorProbeSimulator):
        raise CalibrationError("behavior recovery simulator is invalid")
    if not isinstance(generating_truth, BehaviorParameters):
        raise CalibrationError("behavior recovery truth must be BehaviorParameters")
    if frozen_config.synthetic_only is not True:
        raise CalibrationError("behavior recovery benchmark is synthetic-only")
    if {prior.name for prior in frozen_config.priors} != set(BEHAVIOR_PARAMETER_NAMES):
        raise CalibrationError("behavior recovery requires all nine parameters")
    true_parameters = generating_truth.to_mapping()
    calibration_key = calibration_event_key(
        frozen_config,
        replicate=0,
        step=0,
        purpose="behavior_benchmark:observed:calibration",
    )
    held_out_key = calibration_event_key(
        frozen_config,
        replicate=0,
        step=1,
        purpose="behavior_benchmark:observed:held_out",
    )
    observed = synthetic_calibration_input(
        frozen_config,
        probe.simulate(true_parameters, event_key=calibration_key),
        probe.simulate(true_parameters, event_key=held_out_key),
        record_id="synthetic-behavior-recovery-input",
        calibration_generation_event_key=calibration_key.value,
        held_out_generation_event_key=held_out_key.value,
    )
    backend = ABCPosteriorEstimator() if estimator is None else estimator
    fit = backend.fit(
        config=frozen_config,
        observed=observed,
        simulator=probe,
        namespace="behavior_benchmark:fit",
    )
    posterior, posterior_keys = sample_posterior(
        config=frozen_config,
        fit=fit,
        namespace="behavior_benchmark:posterior",
    )
    ppc = posterior_predictive_checks(
        config=frozen_config,
        observed=observed,
        posterior=posterior,
        simulator=probe,
        namespace="behavior_benchmark",
    )
    recovery = synthetic_recovery(frozen_config, posterior, true_parameters)
    identifiability = identifiability_diagnostics(frozen_config, posterior)
    sbc = run_simulation_based_calibration(
        config=frozen_config,
        simulator=probe,
        estimator=backend,
    )
    warnings = [
        "SYNTHETIC ONLY: this CI diagnostic is not evidence about real investor "
        "parameters or policy effects.",
        "Real-data behavior recovery remains blocked pending lawful pre-policy "
        "lineage and an externally frozen study authorization.",
        "The calibration moments intentionally omit lagged-memory variation; "
        "memory_decay must not be reported as a precise estimate when diagnostics "
        "mark it unidentifiable.",
    ]
    for diagnostic in identifiability:
        if not diagnostic.estimable_for_reporting:
            warnings.append(
                f"Parameter {diagnostic.parameter} is not identifiable enough for "
                "precise reporting."
            )
        if diagnostic.boundary_pileup:
            warnings.append(
                f"Parameter {diagnostic.parameter} has posterior boundary pile-up."
            )
        if diagnostic.multimodal:
            warnings.append(
                f"Parameter {diagnostic.parameter} has a multimodal posterior."
            )
    for statistic in ppc.statistics:
        if statistic.role is SummaryRole.HELD_OUT_VALIDATION and not statistic.covered:
            warnings.append(f"Held-out PPC failed for statistic {statistic.name}.")
    event_keys = (
        observed.event_keys
        + fit.event_keys
        + posterior_keys
        + ppc.event_keys
        + sbc.event_keys
    )
    return BehaviorRecoveryBenchmarkRun(
        config=frozen_config,
        observed=observed,
        simulator_config_sha256=sha256_bytes(canonical_json_bytes(probe.to_mapping())),
        truth=generating_truth,
        posterior=posterior,
        ppc=ppc,
        recovery=recovery,
        sbc=sbc,
        identifiability=identifiability,
        event_keys=event_keys,
        warnings=tuple(warnings),
    )


__all__ = [
    "BEHAVIOR_PARAMETER_NAMES",
    "BEHAVIOR_RECOVERY_SIMULATOR_ID",
    "BehaviorRecoveryBenchmarkRun",
    "SyntheticBehaviorProbeSimulator",
    "behavior_recovery_config",
    "default_behavior_recovery_truth",
    "run_behavior_recovery_benchmark",
]
