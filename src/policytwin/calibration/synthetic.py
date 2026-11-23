"""Deterministic synthetic calibration design used only for tests and CI."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from policytwin.calibration.schema import (
    CalibrationConfig,
    CalibrationError,
    CalibrationInput,
    IdentifiabilityThresholds,
    ParameterTransform,
    PriorSpec,
    SummaryLevel,
    SummaryRole,
    SummarySourcePartition,
    SummaryStatisticsConfig,
    SummaryStatisticSpec,
    SummaryVector,
    TransformKind,
    TransformLineage,
    TransformLineageNode,
    derive_source_design_sha256,
    derive_source_set_sha256,
)
from policytwin.counterfactual.randomness import EventKey, deterministic_int
from policytwin.data.temporal_firewall import DataPartition, TemporalRecord
from policytwin.provenance import canonical_json_bytes, sha256_bytes

_UNIT_MAX = (1 << 53) - 1
_SYNTHETIC_SOURCE_TRANSFORM_SHA256 = sha256_bytes(
    canonical_json_bytes(
        {
            "implementation": (
                "policytwin.calibration.synthetic.synthetic_calibration_input"
            ),
            "schema_version": 1,
        }
    )
)


def _synthetic_source_lineage(
    *, record_id: str, event_key: str, suffix: str
) -> TransformLineage:
    return TransformLineage(
        root_node_id=f"{suffix}-summary-transform",
        nodes=(
            TransformLineageNode(
                node_id=f"{suffix}-summary-transform",
                operation="synthetic_summary_extraction",
                implementation_sha256=_SYNTHETIC_SOURCE_TRANSFORM_SHA256,
                source_record_ids=(record_id,),
                event_keys=(event_key,),
            ),
        ),
    )


def synthetic_source_design_sha256(role: SummaryRole) -> str:
    """Return the verifier-derived one-record synthetic source design."""
    if not isinstance(role, SummaryRole):
        raise CalibrationError("synthetic source design role must be a SummaryRole")
    partition = (
        DataPartition.TRAIN
        if role is SummaryRole.CALIBRATION
        else DataPartition.VALIDATION
    )
    record_id = "source-design-record"
    event_key = f"ptk1:{sha256_bytes(b'source-design-event-key')}"
    lineage = _synthetic_source_lineage(
        record_id=record_id,
        event_key=event_key,
        suffix="source-design",
    )
    return derive_source_design_sha256(
        record_partitions=((record_id, partition),),
        event_keys=(event_key,),
        transform_lineage=lineage,
    )


def _derived_key(parent: EventKey, name: str) -> EventKey:
    return EventKey(
        experiment_id=parent.experiment_id,
        seed=parent.seed,
        replicate_id=parent.replicate_id,
        simulation_step=parent.simulation_step,
        purpose=f"{parent.purpose}:synthetic_noise:{name}",
    )


def _normal(parent: EventKey, name: str) -> float:
    first_key = _derived_key(parent, f"{name}:box_muller_0")
    second_key = _derived_key(parent, f"{name}:box_muller_1")
    first = (deterministic_int(first_key, 0, _UNIT_MAX) + 0.5) / (_UNIT_MAX + 1.0)
    second = (deterministic_int(second_key, 0, _UNIT_MAX) + 0.5) / (_UNIT_MAX + 1.0)
    return math.sqrt(-2.0 * math.log(first)) * math.cos(2.0 * math.pi * second)


@dataclass(frozen=True, slots=True)
class SyntheticMarketSimulator:
    """Small identifiable simulator; it is not a real market model."""

    noise_scale: float = 0.005

    def __post_init__(self) -> None:
        if not math.isfinite(self.noise_scale) or self.noise_scale < 0.0:
            raise CalibrationError("noise_scale must be finite and non-negative")
        object.__setattr__(self, "noise_scale", float(self.noise_scale))

    @property
    def simulator_id(self) -> str:
        return "synthetic_summary_simulator_v1"

    @property
    def simulator_config_sha256(self) -> str:
        """Commit to noise settings that materially change generated summaries."""
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def to_mapping(self) -> dict[str, object]:
        """Return the complete synthetic simulator configuration."""
        return {
            "noise_scale": self.noise_scale,
            "schema_version": 1,
            "simulator_id": self.simulator_id,
            "synthetic_only": True,
        }

    def simulate(
        self,
        parameters: Mapping[str, float],
        *,
        event_key: EventKey,
    ) -> SummaryVector:
        """Return micro/meso/macro summaries from two known parameters."""
        if set(parameters) != {"liquidity", "signal_strength"}:
            raise CalibrationError("synthetic simulator parameter set changed")
        signal = float(parameters["signal_strength"])
        liquidity = float(parameters["liquidity"])
        if not -1.0 <= signal <= 1.0 or not 0.5 <= liquidity <= 2.0:
            raise CalibrationError("synthetic parameters lie outside their support")

        def noise(name: str) -> float:
            return self.noise_scale * _normal(event_key, name)

        return SummaryVector(
            (
                (
                    "macro_volatility",
                    0.5
                    + 0.4 * abs(signal)
                    + 0.2 / liquidity
                    + noise("macro_volatility"),
                ),
                ("macro_volume", 5.0 + signal + liquidity + noise("macro_volume")),
                ("meso_depth", 2.0 * liquidity - 0.1 * signal + noise("meso_depth")),
                (
                    "meso_spread",
                    1.2 - 0.35 * liquidity + 0.2 * signal + noise("meso_spread"),
                ),
                ("micro_order_arrival", signal + noise("micro_order_arrival")),
                (
                    "micro_order_size",
                    0.5 * signal + 0.75 * liquidity + noise("micro_order_size"),
                ),
            )
        )


def synthetic_calibration_config(
    *,
    seed: str = "synthetic-calibration-v1",
    simulation_count: int = 500,
    accepted_particles: int = 50,
    posterior_draws: int = 200,
    ppc_draws: int = 100,
    sbc_replicates: int = 6,
    sbc_posterior_draws: int = 100,
) -> CalibrationConfig:
    """Build the frozen CI configuration with disjoint statistic roles."""

    def statistic(
        name: str, level: SummaryLevel, role: SummaryRole
    ) -> SummaryStatisticSpec:
        return SummaryStatisticSpec(
            name,
            level,
            role,
            definition_sha256=sha256_bytes(
                canonical_json_bytes(
                    {
                        "name": name,
                        "simulator_id": "synthetic_summary_simulator_v1",
                        "synthetic_formula_version": 1,
                    }
                )
            ),
            source_design_sha256=synthetic_source_design_sha256(role),
        )

    return CalibrationConfig(
        schema_version=2,
        study_id="synthetic-calibration-closure",
        cutoff=datetime(2025, 1, 1, tzinfo=UTC),
        seed=seed,
        priors=(
            PriorSpec(
                "liquidity",
                0.5,
                2.0,
                ParameterTransform(TransformKind.LOGIT),
            ),
            PriorSpec("signal_strength", -1.0, 1.0),
        ),
        summaries=SummaryStatisticsConfig(
            (
                statistic(
                    "micro_order_arrival",
                    SummaryLevel.MICRO,
                    SummaryRole.CALIBRATION,
                ),
                statistic(
                    "micro_order_size",
                    SummaryLevel.MICRO,
                    SummaryRole.HELD_OUT_VALIDATION,
                ),
                statistic("meso_depth", SummaryLevel.MESO, SummaryRole.CALIBRATION),
                statistic(
                    "meso_spread",
                    SummaryLevel.MESO,
                    SummaryRole.HELD_OUT_VALIDATION,
                ),
                statistic("macro_volume", SummaryLevel.MACRO, SummaryRole.CALIBRATION),
                statistic(
                    "macro_volatility",
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


def synthetic_calibration_input(
    config: CalibrationConfig,
    calibration_summary: SummaryVector,
    held_out_summary: SummaryVector,
    *,
    record_id: str = "synthetic-calibration-input",
    calibration_generation_event_key: str,
    held_out_generation_event_key: str,
) -> CalibrationInput:
    """Bind two independently generated trajectories to typed source roles."""
    calibration_names = {item.name for item in config.summaries.calibration}
    held_out_names = {item.name for item in config.summaries.held_out}
    calibration_values = calibration_summary.as_dict()
    held_out_values = held_out_summary.as_dict()
    expected = calibration_names | held_out_names
    if set(calibration_values) != expected or set(held_out_values) != expected:
        raise CalibrationError(
            "role trajectory summaries must each expose the complete frozen schema"
        )
    calibration_vector = SummaryVector(
        tuple((name, calibration_values[name]) for name in sorted(calibration_names))
    )
    held_out_vector = SummaryVector(
        tuple((name, held_out_values[name]) for name in sorted(held_out_names))
    )
    observed = SummaryVector(calibration_vector.values + held_out_vector.values)
    role_inputs = (
        (
            SummaryRole.CALIBRATION,
            DataPartition.TRAIN,
            calibration_vector,
            calibration_generation_event_key,
            "calibration",
            config.summaries.calibration[0].source_design_sha256,
        ),
        (
            SummaryRole.HELD_OUT_VALIDATION,
            DataPartition.VALIDATION,
            held_out_vector,
            held_out_generation_event_key,
            "held-out",
            config.summaries.held_out[0].source_design_sha256,
        ),
    )
    sources: list[SummarySourcePartition] = []
    for role, partition, vector, event_key, suffix, source_design_hash in role_inputs:
        record = TemporalRecord(
            record_id=f"{record_id}-{suffix}",
            partition=partition,
            event_time=config.cutoff - timedelta(days=3),
            available_time=config.cutoff - timedelta(days=2),
            ingestion_time=config.cutoff - timedelta(days=1),
            source="policytwin.calibration.synthetic",
            version="v2",
            content_hash=vector.sha256,
            license_tag="synthetic-test-only",
        )
        transform_lineage = _synthetic_source_lineage(
            record_id=record.record_id,
            event_key=event_key,
            suffix=suffix,
        )
        source_record_set_sha256 = derive_source_set_sha256(
            records=(record,),
            event_keys=(event_key,),
            transform_lineage=transform_lineage,
        )
        sources.append(
            SummarySourcePartition(
                role=role,
                observed=vector,
                source_design_sha256=source_design_hash,
                source_record_set_sha256=source_record_set_sha256,
                records=(record,),
                event_keys=(event_key,),
                transform_lineage=transform_lineage,
            )
        )
    result = CalibrationInput(
        observed,
        tuple(sources),
        synthetic_only=True,
    )
    result.validate_for(config)
    return result


def generate_synthetic_observation(
    config: CalibrationConfig,
    simulator: SyntheticMarketSimulator,
    true_parameters: Mapping[str, float],
) -> CalibrationInput:
    """Generate the one known-truth observation for recovery tests."""
    calibration_key = EventKey(
        experiment_id=config.study_id,
        seed=config.seed,
        replicate_id=0,
        simulation_step=0,
        purpose="calibration:synthetic_observed:calibration",
    )
    held_out_key = EventKey(
        experiment_id=config.study_id,
        seed=config.seed,
        replicate_id=0,
        simulation_step=1,
        purpose="calibration:synthetic_observed:held_out",
    )
    calibration_summary = simulator.simulate(true_parameters, event_key=calibration_key)
    held_out_summary = simulator.simulate(true_parameters, event_key=held_out_key)
    return synthetic_calibration_input(
        config,
        calibration_summary,
        held_out_summary,
        calibration_generation_event_key=calibration_key.value,
        held_out_generation_event_key=held_out_key.value,
    )


__all__ = [
    "SyntheticMarketSimulator",
    "generate_synthetic_observation",
    "synthetic_calibration_config",
    "synthetic_calibration_input",
]
