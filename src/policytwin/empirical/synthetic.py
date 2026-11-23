"""Deterministic known-effect fixtures for tests and CI only."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from policytwin.counterfactual.randomness import EventKey, deterministic_int
from policytwin.data.temporal_firewall import DataPartition, TemporalRecord
from policytwin.empirical.schema import (
    EmpiricalEvaluationConfig,
    EmpiricalSpecification,
    FrozenEvaluationInputs,
    FrozenPanel,
    MissingDataPolicy,
    ObservationStatus,
    PanelObservation,
    PanelUnit,
    PostTreatmentGroup,
    PreTreatmentCharacteristic,
    RandomizationDesign,
    SubgroupDefinition,
    TreatmentDefinition,
)
from policytwin.provenance import canonical_json_bytes, sha256_bytes


@dataclass(frozen=True, slots=True)
class KnownEffectDesign:
    """Small deterministic DGP used only to test estimator behavior."""

    seed: str = "known-effect-v1"
    clusters_per_arm: int = 8
    units_per_cluster: int = 4
    treatment_effect: float = 2.0
    subgroup_spread: float = 0.0
    noise_scale: float = 0.5
    treated_pretrend: float = 0.0
    bootstrap_repetitions: int = 99
    randomization_repetitions: int = 99

    def __post_init__(self) -> None:
        if self.clusters_per_arm < 2 or self.units_per_cluster < 2:
            raise ValueError(
                "known-effect fixture needs at least two clusters and units"
            )
        if self.noise_scale < 0:
            raise ValueError("noise_scale must not be negative")


@dataclass(frozen=True, slots=True)
class KnownEffectFixture:
    """Config, frozen panel, receipt, and true synthetic estimand."""

    config: EmpiricalEvaluationConfig
    panel: FrozenPanel
    frozen_inputs: FrozenEvaluationInputs
    true_average_effect: float


def _draw(seed: str, unit_id: str, period: int, purpose: str) -> float:
    key = EventKey(
        experiment_id="empirical-known-effect",
        seed=seed,
        replicate_id=0,
        simulation_step=period,
        purpose=f"{purpose}:{unit_id}",
    )
    integer = deterministic_int(key, 0, 2_000_000)
    return integer / 1_000_000.0 - 1.0


def _record(
    *,
    record_id: str,
    event_time: datetime,
    content_hash: str,
    pre_treatment: bool,
) -> TemporalRecord:
    return TemporalRecord(
        record_id=record_id,
        partition=(
            DataPartition.VALIDATION
            if pre_treatment
            else DataPartition.POST_POLICY_EVALUATION
        ),
        event_time=event_time,
        available_time=event_time,
        ingestion_time=event_time + timedelta(hours=1),
        source="policytwin.synthetic.known_effect",
        version="v1",
        content_hash=content_hash,
        license_tag="synthetic-test-only",
    )


def _characteristic(
    unit_id: str, name: str, value: float, policy_start: datetime
) -> PreTreatmentCharacteristic:
    payload = {
        "measured_relative_period": -4,
        "name": name,
        "unit_id": unit_id,
        "value": value,
    }
    return PreTreatmentCharacteristic(
        unit_id=unit_id,
        name=name,
        value=value,
        measured_relative_period=-4,
        record=_record(
            record_id=f"characteristic-{name}-{unit_id}",
            event_time=policy_start - timedelta(days=120),
            content_hash=sha256_bytes(canonical_json_bytes(payload)),
            pre_treatment=True,
        ),
    )


def _observation(
    *,
    unit_id: str,
    period: int,
    outcome: float,
    policy_start: datetime,
) -> PanelObservation:
    post_group = (
        PostTreatmentGroup(name="realized_response", value="observed_post")
        if period >= 0
        else None
    )
    payload = {
        "outcome": outcome,
        "post_treatment_group": (
            post_group.to_mapping() if post_group is not None else None
        ),
        "relative_period": period,
        "status": ObservationStatus.OBSERVED.value,
        "unit_id": unit_id,
    }
    event_time = policy_start + timedelta(days=30 * period)
    return PanelObservation(
        unit_id=unit_id,
        relative_period=period,
        status=ObservationStatus.OBSERVED,
        outcome=outcome,
        record=_record(
            record_id=f"outcome-{unit_id}-{period}",
            event_time=event_time,
            content_hash=sha256_bytes(canonical_json_bytes(payload)),
            pre_treatment=period < 0,
        ),
        post_treatment_group=post_group,
    )


def build_known_effect_fixture(design: KnownEffectDesign) -> KnownEffectFixture:
    """Build a sealed synthetic panel with a known common-timing effect."""
    policy_start = datetime(2020, 7, 1, tzinfo=UTC)
    periods = (-3, -2, -1, 0, 1, 2)
    treated_ids: list[str] = []
    control_ids: list[str] = []
    units: list[PanelUnit] = []
    unit_traits: dict[str, float] = {}
    unit_covariates: dict[str, float] = {}
    for arm_prefix, target in (("treated", treated_ids), ("control", control_ids)):
        for cluster_index in range(design.clusters_per_arm):
            cluster_id = f"{arm_prefix}-cluster-{cluster_index}"
            for unit_index in range(design.units_per_cluster):
                unit_id = f"{arm_prefix}-{cluster_index}-{unit_index}"
                trait = float(unit_index % 2)
                target.append(unit_id)
                unit_traits[unit_id] = trait
                covariate = cluster_index / design.clusters_per_arm + unit_index / (
                    10.0 * design.units_per_cluster
                )
                unit_covariates[unit_id] = covariate
                units.append(
                    PanelUnit(
                        unit_id=unit_id,
                        cluster_id=cluster_id,
                        characteristics=(
                            _characteristic(
                                unit_id, "baseline_trait", trait, policy_start
                            ),
                            _characteristic(
                                unit_id,
                                "baseline_covariate",
                                covariate,
                                policy_start,
                            ),
                        ),
                    )
                )
    treatment = TreatmentDefinition(
        policy_id="synthetic-known-effect-policy",
        policy_start=policy_start,
        policy_end=policy_start + timedelta(days=90),
        randomization_design=RandomizationDesign.UNIT_COMPLETE,
        treated_unit_ids=tuple(treated_ids),
        control_unit_ids=tuple(control_ids),
    )
    specification = EmpiricalSpecification(
        specification_id="pre_registered_main",
        pre_periods=(-3, -2, -1),
        post_periods=(0, 1, 2),
        event_periods=periods,
        reference_period=-1,
        max_absolute_lead_tolerance=0.35,
        max_slope_gap_per_day=0.35,
        covariates=("baseline_covariate",),
        quantiles=(0.25, 0.5, 0.75),
        subgroup_ids=("trait_low", "trait_high"),
    )
    config = EmpiricalEvaluationConfig(
        schema_version=4,
        study_id=f"known-effect-{design.seed}",
        treatment=treatment,
        specifications=(specification,),
        subgroups=(
            SubgroupDefinition(
                subgroup_id="trait_low",
                characteristic_name="baseline_trait",
                upper_exclusive=0.5,
            ),
            SubgroupDefinition(
                subgroup_id="trait_high",
                characteristic_name="baseline_trait",
                lower_inclusive=0.5,
            ),
        ),
        bootstrap_repetitions=design.bootstrap_repetitions,
        randomization_repetitions=design.randomization_repetitions,
        confidence_level=0.95,
        random_seed=design.seed,
        missing_data_policy=MissingDataPolicy.COMPLETE_CASE,
        max_missing_fraction=0.0,
        max_attrition_fraction=0.0,
        max_composition_shift=0.0,
        maximum_invalid_bootstrap_fraction=1.0,
        synthetic_test_only=True,
    )
    observations: list[PanelObservation] = []
    for unit_id in treatment.unit_ids:
        treated = unit_id in treatment.treated_unit_ids
        trait = unit_traits[unit_id]
        covariate = unit_covariates[unit_id]
        unit_intercept = 10.0 + _draw(design.seed, unit_id, 0, "unit")
        heterogeneous_effect = design.treatment_effect + design.subgroup_spread * (
            1.0 if trait >= 0.5 else -1.0
        )
        for period in periods:
            noise = design.noise_scale * _draw(
                design.seed, unit_id, period, "unit-period"
            )
            untreated = unit_intercept + 0.4 * period + 0.2 * covariate + noise
            pretrend = (
                design.treated_pretrend * period if treated and period < 0 else 0.0
            )
            effect = heterogeneous_effect if treated and period >= 0 else 0.0
            observations.append(
                _observation(
                    unit_id=unit_id,
                    period=period,
                    outcome=untreated + pretrend + effect,
                    policy_start=policy_start,
                )
            )
    panel = FrozenPanel(
        schema_version=1,
        panel_id=f"known-effect-panel-{design.seed}",
        units=tuple(units),
        expected_periods=periods,
        observations=tuple(observations),
        synthetic_only=True,
    )
    cutoff = policy_start + timedelta(days=75)
    frozen_at = cutoff + timedelta(days=1)
    receipt = FrozenEvaluationInputs.for_synthetic_fixture(
        config,
        panel,
        frozen_at=frozen_at,
        evaluation_started_at=max(
            frozen_at + timedelta(days=1),
            treatment.policy_end + timedelta(days=2),
        ),
        knowledge_cutoff=cutoff,
    )
    return KnownEffectFixture(
        config=config,
        panel=panel,
        frozen_inputs=receipt,
        true_average_effect=design.treatment_effect,
    )


__all__ = ["KnownEffectDesign", "KnownEffectFixture", "build_known_effect_fixture"]
