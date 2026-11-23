"""A deterministic, fully synthetic paired smoke experiment.

This module exercises configuration, event-keyed randomness, paired outcomes,
artifact production, and audit plumbing. It is not a market model and cannot
support an empirical or real-world causal claim.
"""

from __future__ import annotations

from dataclasses import dataclass

from policytwin.config import SmokeExperimentConfig
from policytwin.counterfactual.randomness import EventKey, deterministic_int

CLAIM_SCOPE = "synthetic_engineering_smoke"
EMPIRICAL_STATUS = "blocked_no_real_market_data"


@dataclass(frozen=True, slots=True)
class SmokeEvent:
    """One deterministic synthetic event written to the JSONL audit log."""

    sequence: int
    replicate_id: int
    simulation_step: int
    event_key: str
    shock_ticks: int
    baseline_value_ticks: int
    policy_value_ticks: int

    def to_mapping(self) -> dict[str, object]:
        """Return the machine-readable event payload."""
        return {
            "baseline_value_ticks": self.baseline_value_ticks,
            "event_key": self.event_key,
            "event_type": "synthetic_paired_step",
            "policy_value_ticks": self.policy_value_ticks,
            "replicate_id": self.replicate_id,
            "schema_version": 1,
            "sequence": self.sequence,
            "shock_ticks": self.shock_ticks,
            "simulation_step": self.simulation_step,
            "world_scope": "shared_exogenous",
        }


@dataclass(frozen=True, slots=True)
class SyntheticPairOutcome:
    """Final paired outcome for one replicate."""

    replicate_id: int
    baseline_outcome_ticks: int
    policy_outcome_ticks: int
    difference_ticks: int

    def to_mapping(self) -> dict[str, int]:
        """Return the deterministic result representation."""
        return {
            "baseline_outcome_ticks": self.baseline_outcome_ticks,
            "difference_ticks": self.difference_ticks,
            "policy_outcome_ticks": self.policy_outcome_ticks,
            "replicate_id": self.replicate_id,
        }


@dataclass(frozen=True, slots=True)
class SmokeResult:
    """Complete result of the synthetic engineering smoke experiment."""

    experiment_id: str
    outcomes: tuple[SyntheticPairOutcome, ...]

    def to_mapping(self, run_id: str) -> dict[str, object]:
        """Return a stable result artifact with explicit scientific limits."""
        differences = tuple(outcome.difference_ticks for outcome in self.outcomes)
        return {
            "claim_scope": CLAIM_SCOPE,
            "empirical_status": EMPIRICAL_STATUS,
            "experiment_id": self.experiment_id,
            "outcomes": [outcome.to_mapping() for outcome in self.outcomes],
            "run_id": run_id,
            "run_status": "completed",
            "schema_version": 1,
            "summary": {
                "difference_count": len(differences),
                "difference_max_ticks": max(differences),
                "difference_min_ticks": min(differences),
                "difference_sum_ticks": sum(differences),
                "mean_denominator": len(differences),
                "mean_numerator_ticks": sum(differences),
            },
        }


def run_smoke_experiment(
    config: SmokeExperimentConfig,
) -> tuple[SmokeResult, tuple[SmokeEvent, ...], tuple[str, ...]]:
    """Run a deterministic paired arithmetic experiment from semantic keys."""
    events: list[SmokeEvent] = []
    outcomes: list[SyntheticPairOutcome] = []
    event_keys: list[str] = []
    sequence = 0

    for replicate_id in range(config.replicates):
        baseline_value = config.initial_value_ticks
        policy_value = config.initial_value_ticks

        for simulation_step in range(config.horizon_steps):
            key = EventKey(
                experiment_id=config.experiment_id,
                seed=config.seed,
                replicate_id=replicate_id,
                simulation_step=simulation_step,
                purpose="shared_exogenous_smoke_shock",
            )
            shock = deterministic_int(
                key,
                -config.shock_radius_ticks,
                config.shock_radius_ticks,
            )
            baseline_value += shock
            policy_value += shock + config.policy_shift_ticks
            event_keys.append(key.value)
            events.append(
                SmokeEvent(
                    sequence=sequence,
                    replicate_id=replicate_id,
                    simulation_step=simulation_step,
                    event_key=key.value,
                    shock_ticks=shock,
                    baseline_value_ticks=baseline_value,
                    policy_value_ticks=policy_value,
                )
            )
            sequence += 1

        outcomes.append(
            SyntheticPairOutcome(
                replicate_id=replicate_id,
                baseline_outcome_ticks=baseline_value,
                policy_outcome_ticks=policy_value,
                difference_ticks=policy_value - baseline_value,
            )
        )

    return (
        SmokeResult(experiment_id=config.experiment_id, outcomes=tuple(outcomes)),
        tuple(events),
        tuple(event_keys),
    )
