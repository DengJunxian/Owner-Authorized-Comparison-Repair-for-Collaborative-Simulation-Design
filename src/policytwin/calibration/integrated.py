"""Calibration adapter backed by the complete synthetic PolicyTwin loop."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime

from policytwin.calibration.schema import (
    CalibrationConfig,
    CalibrationError,
    IdentifiabilityThresholds,
    ParameterTransform,
    PriorSpec,
    SummaryLevel,
    SummaryRole,
    SummaryStatisticsConfig,
    SummaryStatisticSpec,
    SummaryVector,
    TransformKind,
)
from policytwin.calibration.synthetic import synthetic_source_design_sha256
from policytwin.counterfactual.integrated_market import (
    IntegratedSyntheticWorldFactory,
)
from policytwin.counterfactual.randomness import EventKey
from policytwin.provenance import canonical_json_bytes, sha256_bytes

INTEGRATED_CALIBRATION_SIMULATOR_ID = "integrated_synthetic_policytwin_v1"


@dataclass(frozen=True, slots=True)
class IntegratedSyntheticCalibrationAdapter:
    """Run the paired engine and expose frozen micro/meso/macro summaries."""

    step_count: int = 4

    def __post_init__(self) -> None:
        if (
            isinstance(self.step_count, bool)
            or not isinstance(self.step_count, int)
            or self.step_count < 2
        ):
            raise ValueError("step_count must be an integer of at least two")

    @property
    def simulator_id(self) -> str:
        return INTEGRATED_CALIBRATION_SIMULATOR_ID

    def to_mapping(self) -> dict[str, object]:
        # Importing the experiment package while ``policytwin.calibration`` is
        # still being initialized creates a calibration -> experiments ->
        # evaluation -> calibration cycle.  The integrated runner is needed
        # only when this adapter is exercised, so keep the public calibration
        # package importable without changing simulation semantics.
        from policytwin.experiments.integrated_synthetic import (
            integrated_synthetic_market_config,
        )

        config = integrated_synthetic_market_config()
        return {
            "integrated_config_sha256": config.sha256,
            "schema_version": 1,
            "simulator_id": self.simulator_id,
            "step_count": self.step_count,
            "synthetic_only": True,
        }

    def simulate(
        self,
        parameters: Mapping[str, float],
        *,
        event_key: EventKey,
    ) -> SummaryVector:
        from policytwin.experiments.integrated_synthetic import (
            integrated_synthetic_market_config,
            integrated_synthetic_pair,
        )

        if set(parameters) != {"behavior_scale", "signal_scale"}:
            raise CalibrationError(
                "integrated parameters have missing or unknown fields"
            )
        values: dict[str, float] = {}
        for name, value in parameters.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
            ):
                raise CalibrationError(f"integrated parameter {name} must be finite")
            values[name] = float(value)
        pair = integrated_synthetic_pair(
            event_key=event_key,
            parameters=values,
            candidate_is_noop=True,
            step_count=self.step_count,
        )
        factory = IntegratedSyntheticWorldFactory(integrated_synthetic_market_config())
        outcome = pair.run(factory)
        terminal = next(
            item for item in outcome.baseline.horizons if item.horizon_id == "terminal"
        )
        group = terminal.groups[0]
        metrics = {item.metric_name: item.value for item in group.metrics}
        agents = len(factory.config.agents)
        submitted = metrics["submitted_orders"]
        return SummaryVector(
            (
                ("macro_quoted_spread", metrics["quoted_spread"]),
                ("macro_net_worth", metrics["net_worth"]),
                ("meso_memory_load", metrics["memory_entries"] / agents),
                ("meso_mean_buy_probability", metrics["mean_buy_probability"]),
                (
                    "micro_turnover_probability",
                    metrics["mean_turnover_probability"],
                ),
                (
                    "micro_submission_rate",
                    submitted / (agents * self.step_count),
                ),
            )
        )


def integrated_synthetic_calibration_config(
    *,
    seed: str = "integrated-synthetic-calibration-v1",
    simulation_count: int = 24,
    accepted_particles: int = 6,
    posterior_draws: int = 12,
    ppc_draws: int = 8,
    sbc_replicates: int = 2,
    sbc_posterior_draws: int = 8,
) -> CalibrationConfig:
    """Return a small CI configuration with disjoint statistic roles."""

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
                        "simulator_id": INTEGRATED_CALIBRATION_SIMULATOR_ID,
                        "version": 1,
                    }
                )
            ),
            source_design_sha256=synthetic_source_design_sha256(role),
        )

    return CalibrationConfig(
        schema_version=2,
        study_id="integrated-synthetic-calibration",
        cutoff=datetime(2025, 1, 10, tzinfo=UTC),
        seed=seed,
        priors=(
            PriorSpec(
                "behavior_scale",
                0.2,
                1.0,
                ParameterTransform(TransformKind.LOGIT),
            ),
            PriorSpec(
                "signal_scale",
                0.5,
                2.0,
                ParameterTransform(TransformKind.LOGIT),
            ),
        ),
        summaries=SummaryStatisticsConfig(
            (
                statistic(
                    "micro_submission_rate",
                    SummaryLevel.MICRO,
                    SummaryRole.CALIBRATION,
                ),
                statistic(
                    "micro_turnover_probability",
                    SummaryLevel.MICRO,
                    SummaryRole.HELD_OUT_VALIDATION,
                ),
                statistic(
                    "meso_mean_buy_probability",
                    SummaryLevel.MESO,
                    SummaryRole.CALIBRATION,
                ),
                statistic(
                    "meso_memory_load",
                    SummaryLevel.MESO,
                    SummaryRole.HELD_OUT_VALIDATION,
                ),
                statistic(
                    "macro_net_worth",
                    SummaryLevel.MACRO,
                    SummaryRole.CALIBRATION,
                ),
                statistic(
                    "macro_quoted_spread",
                    SummaryLevel.MACRO,
                    SummaryRole.HELD_OUT_VALIDATION,
                ),
            )
        ),
        simulation_count=simulation_count,
        accepted_particles=accepted_particles,
        posterior_draws=posterior_draws,
        ppc_draws=ppc_draws,
        sbc_replicates=sbc_replicates,
        sbc_posterior_draws=sbc_posterior_draws,
        identifiability=IdentifiabilityThresholds(),
        synthetic_only=True,
    )


__all__ = [
    "INTEGRATED_CALIBRATION_SIMULATOR_ID",
    "IntegratedSyntheticCalibrationAdapter",
    "integrated_synthetic_calibration_config",
]
