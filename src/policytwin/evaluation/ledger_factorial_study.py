"""Protocol-bound synthetic source grid for factorial ledger evaluation.

This module freezes and executes a balanced posterior-draw by scenario grid.
It deliberately stops at exact, source-bound ledger artifacts; statistical
summaries are derived by :mod:`policytwin.evaluation.ledger_factorial`.
Nothing here establishes external preregistration, real-data provenance, or a
real-world causal effect.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from threading import RLock
from typing import Literal, NoReturn
from weakref import WeakKeyDictionary, WeakValueDictionary, finalize

import policytwin.evaluation.ledger_factorial as ledger_factorial_module
from policytwin.calibration import CalibrationRun
from policytwin.calibration.integrated import INTEGRATED_CALIBRATION_SIMULATOR_ID
from policytwin.counterfactual.integrated_market import (
    IntegratedSyntheticMarketConfig,
)
from policytwin.counterfactual.paired import PolicyWorldPair, PosteriorDraw, WorldRole
from policytwin.counterfactual.randomness import EventKey
from policytwin.evaluation.ledger_derived import (
    DifferenceScope,
    ExactRational,
    LedgerDerivedEvaluation,
    derive_protocol_bound_ledger_evidence,
)
from policytwin.evaluation.ledger_factorial import (
    FactorialBenefitDirection,
    FactorialGridCell,
    FactorialOutcome,
    FactorialOutputRole,
    FactorialOutputSpec,
    FactorialPrecisionTarget,
    FactorialUndefinedPolicy,
    LedgerFactorialEvaluation,
    LedgerFactorialSpec,
    derive_ledger_factorial_evaluation,
)
from policytwin.evaluation.ledger_posterior import (
    CalibrationPosteriorCommitment,
    LedgerExecutionEnvironment,
    LedgerPosteriorOutputSpec,
    ledger_output_spec,
)
from policytwin.experiments.integrated_synthetic import (
    ProtocolBoundIntegratedSyntheticRunArtifact,
    integrated_synthetic_pair,
    integrated_synthetic_pair_input_sha256,
    require_verifier_owned_protocol_bound_integrated_artifact,
    run_protocol_bound_integrated_synthetic_pair,
)
from policytwin.implementation_fingerprint import (
    ImplementationFingerprintError,
    live_policytwin_dependency_closure_sha256,
    live_policytwin_implementation_sha256,
)
from policytwin.protocol import (
    CohortBasis,
    FrozenDefinition,
    OutputRole,
    ProtocolExecutionAuthorization,
    ProtocolExecutionConsumer,
    ProtocolExecutionPurpose,
    ProtocolExecutionReceipt,
    StudyMode,
    StudyProtocol,
    SyntheticProtocolExecutionIssuer,
    consume_protocol_execution_authorization,
    validate_protocol_execution_receipt_binding,
)
from policytwin.provenance import (
    MetadataValidationError,
    canonical_json_bytes,
    read_json_bytes_object,
    sha256_bytes,
)

LEDGER_FACTORIAL_STUDY_PLAN_SCHEMA_VERSION = "policytwin.ledger-factorial-study-plan.v1"
LEDGER_FACTORIAL_SOURCE_BATCH_SCHEMA_VERSION = (
    "policytwin.ledger-factorial-source-batch.v1"
)
LEDGER_FACTORIAL_PLAN_DEFINITION_ID = "ledger_factorial_analysis_plan"
LEDGER_FACTORIAL_SOURCE_CLAIM_SCOPE = (
    "synthetic_protocol_bound_factorial_source_grid_mechanics_only"
)
LEDGER_FACTORIAL_SOURCE_BOUND_RESULT_SCHEMA_VERSION = (
    "policytwin.ledger-factorial-source-bound-result.v1"
)
LEDGER_FACTORIAL_SOURCE_BOUND_CLAIM_SCOPE = (
    "synthetic_source_bound_finite_grid_factorial_arithmetic_only"
)
LEDGER_FACTORIAL_MAX_SOURCE_COUNT = 128

_MAX_FACTORIAL_SOURCE_BATCH_BYTES = 256 * 1024 * 1024

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EVENT_KEY = re.compile(r"^ptk1:[0-9a-f]{64}$")


class LedgerFactorialStudyError(RuntimeError):
    """Raised when the frozen factorial source grid cannot be proved."""


def _fail(message: str) -> NoReturn:
    raise LedgerFactorialStudyError(message)


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        _fail(f"{field} must be a stable ASCII identifier")
    return value


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail(f"{field} must be a lowercase SHA-256 digest")
    return value


def _aware(value: object, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        _fail(f"{field} must be timezone-aware")
    return value.astimezone(UTC)


def _at_least_two(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 2:
        _fail(f"{field} must be an integer of at least two")
    return value


class FactorialScenarioDesign(StrEnum):
    """The only supported v1 design is a complete balanced cross."""

    BALANCED_POSTERIOR_BY_SCENARIO = "balanced_posterior_by_scenario"


class LedgerFactorialSourceBatchOrigin(StrEnum):
    """Process-local origin; neither value is external execution attestation."""

    EXECUTED_HERE = "executed_here"
    LOADED_AND_REPLAYED = "loaded_and_replayed"


def ledger_factorial_study_implementation_sha256() -> str:
    """Commit the study layer and the live factorial-analysis closure."""
    try:
        return sha256_bytes(
            canonical_json_bytes(
                {
                    "analysis_dependency_closure_sha256": (
                        live_policytwin_dependency_closure_sha256(
                            root_modules=(ledger_factorial_module,)
                        )
                    ),
                    "schema_version": 2,
                    "study_module_sha256": live_policytwin_implementation_sha256(
                        module_path=Path(__file__), namespace=globals()
                    ),
                }
            )
        )
    except ImplementationFingerprintError as error:
        raise LedgerFactorialStudyError(
            "cannot capture the live factorial study implementation"
        ) from error


def _calibrated_step_count(
    *,
    run: CalibrationRun,
    config: IntegratedSyntheticMarketConfig,
) -> int:
    try:
        mapping = read_json_bytes_object(
            run.simulator_config_json,
            source="factorial calibration simulator configuration",
        )
    except MetadataValidationError as error:
        raise LedgerFactorialStudyError(str(error)) from error
    if set(mapping) != {
        "integrated_config_sha256",
        "schema_version",
        "simulator_id",
        "step_count",
        "synthetic_only",
    }:
        _fail("factorial calibration simulator configuration has drifted")
    step_count = mapping["step_count"]
    if (
        mapping["schema_version"] != 1
        or mapping["simulator_id"] != INTEGRATED_CALIBRATION_SIMULATOR_ID
        or mapping["synthetic_only"] is not True
        or isinstance(step_count, bool)
        or not isinstance(step_count, int)
        or step_count < 2
    ):
        _fail("factorial calibration simulator semantics have drifted")
    if config.scenario_generator_spec is None:
        _fail("factorial evaluation requires a frozen forward scenario generator")
    if (
        replace(config, scenario_generator_spec=None).sha256
        != mapping["integrated_config_sha256"]
    ):
        _fail("factorial evaluation config is not the calibrated integrated model")
    return step_count


def _posterior_draw_id(run: CalibrationRun, posterior_ordinal: int) -> str:
    draw = run.posterior.draws[posterior_ordinal]
    return (
        "factorial-posterior-"
        + sha256_bytes(
            canonical_json_bytes(
                {
                    "ordinal": posterior_ordinal,
                    "posterior_sha256": run.posterior.sha256,
                    "resample_event_key": draw.event_key,
                }
            )
        )[:24]
    )


def _scenario_event_key(
    *,
    plan_id: str,
    event_key_seed: str,
    scenario_replicate: int,
) -> EventKey:
    return EventKey(
        experiment_id=plan_id,
        seed=event_key_seed,
        replicate_id=scenario_replicate,
        simulation_step=0,
        purpose="ledger-factorial-scenario",
    )


def _stable_pair(
    *,
    run: CalibrationRun,
    posterior_ordinal: int,
    event_key: EventKey,
    config: IntegratedSyntheticMarketConfig,
    candidate_allow_buy: bool,
    candidate_allow_sell: bool,
    candidate_is_noop: bool,
    step_count: int,
    include_all_groups: bool,
) -> PolicyWorldPair:
    draw = run.posterior.draws[posterior_ordinal]
    generated = integrated_synthetic_pair(
        event_key=event_key,
        parameters=dict(draw.parameters),
        config=config,
        candidate_allow_buy=(True if candidate_is_noop else candidate_allow_buy),
        candidate_allow_sell=(True if candidate_is_noop else candidate_allow_sell),
        candidate_is_noop=candidate_is_noop,
        step_count=step_count,
        include_all_groups=include_all_groups,
    )
    stable_draw = PosteriorDraw(
        draw_id=_posterior_draw_id(run, posterior_ordinal),
        model_draw_sha256=generated.posterior_draw.model_draw_sha256,
        parameters=tuple(draw.parameters),
    )
    return replace(generated, posterior_draw=stable_draw)


@dataclass(frozen=True, slots=True)
class LedgerFactorialSlotPlan:
    """One immutable row-major posterior/scenario source slot."""

    ordinal: int
    posterior_ordinal: int
    scenario_replicate: int
    posterior_draw_id: str
    calibration_draw_sha256: str
    posterior_resample_event_key: str
    parameter_sha256: str
    scenario_event_key_canonical_text: str
    scenario_event_key: str
    scenario_id: str
    shock_tape_sha256: str
    pair_input_sha256: str
    initial_state_sha256: str

    def __post_init__(self) -> None:
        for field in ("ordinal", "posterior_ordinal", "scenario_replicate"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                _fail(f"{field} must be a non-negative integer")
        _identifier(self.posterior_draw_id, "posterior_draw_id")
        _identifier(self.scenario_id, "scenario_id")
        if not isinstance(self.posterior_resample_event_key, str) or not (
            self.posterior_resample_event_key
        ):
            _fail("posterior_resample_event_key must be non-empty")
        if not isinstance(self.scenario_event_key_canonical_text, str) or not (
            self.scenario_event_key_canonical_text
        ):
            _fail("scenario event key canonical text must be non-empty")
        if (
            not isinstance(self.scenario_event_key, str)
            or _EVENT_KEY.fullmatch(self.scenario_event_key) is None
        ):
            _fail("scenario_event_key must be a canonical PolicyTwin event key")
        for field in (
            "calibration_draw_sha256",
            "parameter_sha256",
            "shock_tape_sha256",
            "pair_input_sha256",
            "initial_state_sha256",
        ):
            _sha256(getattr(self, field), field)

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def to_mapping(self) -> dict[str, object]:
        return {
            "calibration_draw_sha256": self.calibration_draw_sha256,
            "initial_state_sha256": self.initial_state_sha256,
            "ordinal": self.ordinal,
            "pair_input_sha256": self.pair_input_sha256,
            "parameter_sha256": self.parameter_sha256,
            "posterior_draw_id": self.posterior_draw_id,
            "posterior_ordinal": self.posterior_ordinal,
            "posterior_resample_event_key": self.posterior_resample_event_key,
            "scenario_event_key": self.scenario_event_key,
            "scenario_event_key_canonical_text": (
                self.scenario_event_key_canonical_text
            ),
            "scenario_id": self.scenario_id,
            "scenario_replicate": self.scenario_replicate,
            "shock_tape_sha256": self.shock_tape_sha256,
        }


def _slot_plan(
    *,
    ordinal: int,
    posterior_ordinal: int,
    scenario_replicate: int,
    run: CalibrationRun,
    event_key: EventKey,
    pair: PolicyWorldPair,
) -> LedgerFactorialSlotPlan:
    draw = run.posterior.draws[posterior_ordinal]
    return LedgerFactorialSlotPlan(
        ordinal=ordinal,
        posterior_ordinal=posterior_ordinal,
        scenario_replicate=scenario_replicate,
        posterior_draw_id=pair.posterior_draw.draw_id,
        calibration_draw_sha256=sha256_bytes(canonical_json_bytes(draw.to_mapping())),
        posterior_resample_event_key=draw.event_key,
        parameter_sha256=pair.posterior_draw.parameter_sha256,
        scenario_event_key_canonical_text=event_key.canonical_text(),
        scenario_event_key=event_key.value,
        scenario_id=pair.scenario_id,
        shock_tape_sha256=pair.shock_tape.sha256,
        pair_input_sha256=integrated_synthetic_pair_input_sha256(pair),
        initial_state_sha256=pair.initial_state.sha256,
    )


@dataclass(frozen=True, slots=True, init=False)
class LedgerFactorialStudyPlan:
    """Factory-produced complete balanced source grid."""

    plan_id: str
    frozen_at: datetime
    event_key_seed: str
    posterior_source: CalibrationPosteriorCommitment
    calibration_run: CalibrationRun
    config: IntegratedSyntheticMarketConfig
    pairs: tuple[PolicyWorldPair, ...]
    slots: tuple[LedgerFactorialSlotPlan, ...]
    outputs: tuple[LedgerPosteriorOutputSpec, ...]
    analysis_spec: LedgerFactorialSpec
    environment: LedgerExecutionEnvironment
    factorial_implementation_sha256: str
    posterior_draw_count: int
    scenario_replicate_count: int
    step_count: int
    candidate_allow_buy: bool
    candidate_allow_sell: bool
    candidate_is_noop: bool
    include_all_groups: bool
    scenario_design: FactorialScenarioDesign
    failure_policy: str
    synthetic_only: Literal[True]
    externally_registered: Literal[False]
    schema_version: str

    def __init__(self) -> None:
        raise TypeError(
            "LedgerFactorialStudyPlan is factory-produced; use "
            "build_ledger_factorial_study_plan"
        )

    @property
    def slot_manifest_sha256(self) -> str:
        return sha256_bytes(
            canonical_json_bytes([item.to_mapping() for item in self.slots])
        )

    @property
    def output_manifest_sha256(self) -> str:
        return sha256_bytes(
            canonical_json_bytes([item.to_mapping() for item in self.outputs])
        )

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_mapping())

    @property
    def sha256(self) -> str:
        return sha256_bytes(self.canonical_bytes)

    @property
    def frozen_definition(self) -> FrozenDefinition:
        return FrozenDefinition(
            definition_id=LEDGER_FACTORIAL_PLAN_DEFINITION_ID,
            description=(
                "Locally frozen balanced synthetic posterior-by-scenario source "
                "grid; not externally preregistered."
            ),
            definition_sha256=self.sha256,
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "candidate_allow_buy": self.candidate_allow_buy,
            "candidate_allow_sell": self.candidate_allow_sell,
            "candidate_is_noop": self.candidate_is_noop,
            "analysis_spec": self.analysis_spec.to_mapping(),
            "analysis_spec_sha256": self.analysis_spec.sha256,
            "environment": self.environment.to_mapping(),
            "event_key_seed": self.event_key_seed,
            "externally_registered": self.externally_registered,
            "factorial_implementation_sha256": (self.factorial_implementation_sha256),
            "failure_policy": self.failure_policy,
            "frozen_at": self.frozen_at.isoformat(),
            "include_all_groups": self.include_all_groups,
            "output_manifest_sha256": self.output_manifest_sha256,
            "outputs": [item.to_mapping() for item in self.outputs],
            "plan_id": self.plan_id,
            "posterior_draw_count": self.posterior_draw_count,
            "posterior_source": self.posterior_source.to_mapping(),
            "scenario_design": self.scenario_design.value,
            "scenario_replicate_count": self.scenario_replicate_count,
            "schema_version": self.schema_version,
            "slot_manifest_sha256": self.slot_manifest_sha256,
            "slots": [item.to_mapping() for item in self.slots],
            "step_count": self.step_count,
            "synthetic_only": self.synthetic_only,
        }

    def _fresh_validate(self) -> None:
        _identifier(self.plan_id, "plan_id")
        _aware(self.frozen_at, "frozen_at")
        if not isinstance(self.event_key_seed, str) or not self.event_key_seed:
            _fail("event_key_seed must be non-empty")
        if (
            self.schema_version != LEDGER_FACTORIAL_STUDY_PLAN_SCHEMA_VERSION
            or self.synthetic_only is not True
            or self.externally_registered is not False
            or self.scenario_design
            is not FactorialScenarioDesign.BALANCED_POSTERIOR_BY_SCENARIO
            or self.failure_policy
            != "fixed_MxR_all_slots_required_any_failure_aborts_no_replacement"
        ):
            _fail("factorial plan scope, design, or failure policy changed")
        _at_least_two(self.posterior_draw_count, "posterior_draw_count")
        _at_least_two(self.scenario_replicate_count, "scenario_replicate_count")
        if not all(
            isinstance(value, bool)
            for value in (
                self.candidate_allow_buy,
                self.candidate_allow_sell,
                self.candidate_is_noop,
                self.include_all_groups,
            )
        ):
            _fail("factorial policy and group switches must be boolean")
        if self.candidate_is_noop and not (
            self.candidate_allow_buy and self.candidate_allow_sell
        ):
            _fail("a no-op factorial policy must equal the baseline switches")
        if type(self.environment) is not LedgerExecutionEnvironment:
            _fail("factorial environment has the wrong exact type")
        self.environment.require_current()
        if (
            self.factorial_implementation_sha256
            != ledger_factorial_study_implementation_sha256()
        ):
            _fail("factorial study implementation changed after plan freeze")
        if type(self.calibration_run) is not CalibrationRun:
            _fail("factorial plan lost its typed calibration source")
        if self.posterior_source != CalibrationPosteriorCommitment.from_run(
            self.calibration_run
        ):
            _fail("factorial calibration posterior changed after plan freeze")
        if self.posterior_draw_count != self.posterior_source.posterior_draw_count:
            _fail("factorial plan must retain the complete calibration posterior")
        if not isinstance(self.config, IntegratedSyntheticMarketConfig):
            _fail("factorial plan lost its typed integrated config")
        calibrated_steps = _calibrated_step_count(
            run=self.calibration_run, config=self.config
        )
        if self.step_count != calibrated_steps:
            _fail("factorial step count differs from calibrated simulator semantics")
        expected_count = self.posterior_draw_count * self.scenario_replicate_count
        if expected_count > LEDGER_FACTORIAL_MAX_SOURCE_COUNT:
            _fail(
                "factorial source grid exceeds its fixed source-count limit of "
                f"{LEDGER_FACTORIAL_MAX_SOURCE_COUNT}"
            )
        if (
            not isinstance(self.pairs, tuple)
            or not isinstance(self.slots, tuple)
            or len(self.pairs) != expected_count
            or len(self.slots) != expected_count
        ):
            _fail("factorial plan is not a complete M by R grid")
        expected_pairs: list[PolicyWorldPair] = []
        expected_slots: list[LedgerFactorialSlotPlan] = []
        for posterior_ordinal in range(self.posterior_draw_count):
            for scenario_replicate in range(self.scenario_replicate_count):
                ordinal = (
                    posterior_ordinal * self.scenario_replicate_count
                    + scenario_replicate
                )
                key = _scenario_event_key(
                    plan_id=self.plan_id,
                    event_key_seed=self.event_key_seed,
                    scenario_replicate=scenario_replicate,
                )
                pair = _stable_pair(
                    run=self.calibration_run,
                    posterior_ordinal=posterior_ordinal,
                    event_key=key,
                    config=self.config,
                    candidate_allow_buy=self.candidate_allow_buy,
                    candidate_allow_sell=self.candidate_allow_sell,
                    candidate_is_noop=self.candidate_is_noop,
                    step_count=self.step_count,
                    include_all_groups=self.include_all_groups,
                )
                expected_pairs.append(pair)
                expected_slots.append(
                    _slot_plan(
                        ordinal=ordinal,
                        posterior_ordinal=posterior_ordinal,
                        scenario_replicate=scenario_replicate,
                        run=self.calibration_run,
                        event_key=key,
                        pair=pair,
                    )
                )
        if self.pairs != tuple(expected_pairs) or self.slots != tuple(expected_slots):
            _fail("factorial source grid does not rederive from its frozen axes")
        if tuple(item.ordinal for item in self.slots) != tuple(range(expected_count)):
            _fail("factorial slots are not canonical row-major ordinals")
        for scenario_replicate in range(self.scenario_replicate_count):
            column = tuple(
                self.slots[
                    posterior_ordinal * self.scenario_replicate_count
                    + scenario_replicate
                ]
                for posterior_ordinal in range(self.posterior_draw_count)
            )
            if (
                len(
                    {
                        (
                            item.scenario_event_key,
                            item.scenario_id,
                            item.shock_tape_sha256,
                            item.initial_state_sha256,
                        )
                        for item in column
                    }
                )
                != 1
            ):
                _fail("one scenario replicate differs across posterior rows")
        for posterior_ordinal in range(self.posterior_draw_count):
            row = self.slots[
                posterior_ordinal * self.scenario_replicate_count : (
                    posterior_ordinal + 1
                )
                * self.scenario_replicate_count
            ]
            if (
                len(
                    {
                        (
                            item.posterior_draw_id,
                            item.calibration_draw_sha256,
                            item.posterior_resample_event_key,
                            item.parameter_sha256,
                        )
                        for item in row
                    }
                )
                != 1
            ):
                _fail("one posterior draw differs across scenario columns")
        if len({item.pair_input_sha256 for item in self.slots}) != expected_count:
            _fail("factorial pair inputs must be unique for every source slot")
        if len({item.scenario_event_key for item in self.slots}) != (
            self.scenario_replicate_count
        ):
            _fail("scenario keys must depend on the scenario axis only")
        if len({item.posterior_draw_id for item in self.slots}) != (
            self.posterior_draw_count
        ):
            _fail("posterior draw identities must depend on the posterior axis only")
        if len({item.initial_state_sha256 for item in self.slots}) != 1:
            _fail("factorial v1 requires one globally fixed opening state")
        _validate_outputs(outputs=self.outputs, config=self.config, pairs=self.pairs)
        expected_analysis_outputs = tuple(
            sorted(
                (
                    FactorialOutputSpec(
                        output_id=item.output_id,
                        role=FactorialOutputRole(item.role.value),
                        undefined_policy=FactorialUndefinedPolicy(
                            item.undefined_policy.value
                        ),
                        benefit_direction=FactorialBenefitDirection(
                            item.benefit_direction.value
                        ),
                        source_definition_sha256=sha256_bytes(
                            canonical_json_bytes(item.to_mapping())
                        ),
                    )
                    for item in self.outputs
                ),
                key=lambda item: item.output_id,
            )
        )
        if (
            type(self.analysis_spec) is not LedgerFactorialSpec
            or self.analysis_spec.posterior_draw_count != self.posterior_draw_count
            or self.analysis_spec.scenario_replicate_count
            != self.scenario_replicate_count
            or self.analysis_spec.outputs != expected_analysis_outputs
            or self.analysis_spec.frozen_at != self.frozen_at
        ):
            _fail("factorial analysis spec differs from the protocol-bound grid plan")


def _validate_outputs(
    *,
    outputs: tuple[LedgerPosteriorOutputSpec, ...],
    config: IntegratedSyntheticMarketConfig,
    pairs: tuple[PolicyWorldPair, ...],
) -> None:
    if not outputs or not all(
        isinstance(item, LedgerPosteriorOutputSpec) for item in outputs
    ):
        _fail("factorial study requires typed frozen ledger outputs")
    if len({item.output_id for item in outputs}) != len(outputs):
        _fail("factorial output IDs must be unique")
    if sum(item.role is OutputRole.PRIMARY for item in outputs) != 1:
        _fail("factorial study requires exactly one primary ledger output")
    first = pairs[0]
    horizons = {item.horizon_id for item in first.horizons}
    groups = {item.group_id for item in first.groups}
    for pair in pairs[1:]:
        if pair.horizons != first.horizons or pair.groups != first.groups:
            _fail("factorial source cells disagree on horizons or groups")
    for output in outputs:
        if output.interval_levels:
            _fail(
                "factorial v1 does not implement registered interval levels; "
                "the frozen interval manifest must be empty"
            )
        if output.horizon_id not in horizons:
            _fail("factorial output references an unknown horizon")
        if output.scope is DifferenceScope.GROUP and output.group_id not in groups:
            _fail("factorial output references an unknown frozen group")
        expected = ledger_output_spec(
            output_id=output.output_id,
            horizon_id=output.horizon_id,
            scope=output.scope,
            metric_id=output.metric_id,
            config=config,
            role=output.role,
            benefit_direction=output.benefit_direction,
            undefined_policy=output.undefined_policy,
            group_id=output.group_id,
            interval_levels=output.interval_levels,
        )
        if output != expected:
            _fail("factorial output differs from the live ledger metric registry")


def build_ledger_factorial_study_plan(
    *,
    plan_id: str,
    frozen_at: datetime,
    calibration_run: CalibrationRun,
    config: IntegratedSyntheticMarketConfig,
    event_key_seed: str,
    scenario_replicate_count: int,
    outputs: tuple[LedgerPosteriorOutputSpec, ...],
    bootstrap_replicates: int,
    bootstrap_seed: str,
    maximum_bootstrap_mean_variance: ExactRational,
    outcomes_available_at: datetime,
    candidate_allow_buy: bool = False,
    candidate_allow_sell: bool = True,
    candidate_is_noop: bool = False,
    step_count: int = 4,
    include_all_groups: bool = False,
) -> LedgerFactorialStudyPlan:
    """Freeze every M by R source input before any source outcome is run."""
    stable_plan_id = _identifier(plan_id, "plan_id")
    stable_time = _aware(frozen_at, "frozen_at")
    replicate_count = _at_least_two(
        scenario_replicate_count, "scenario_replicate_count"
    )
    if not isinstance(event_key_seed, str) or not event_key_seed:
        _fail("event_key_seed must be non-empty")
    if not isinstance(config, IntegratedSyntheticMarketConfig):
        raise TypeError("config must be an IntegratedSyntheticMarketConfig")
    posterior_source = CalibrationPosteriorCommitment.from_run(calibration_run)
    source_count = posterior_source.posterior_draw_count * replicate_count
    if source_count > LEDGER_FACTORIAL_MAX_SOURCE_COUNT:
        _fail(
            "factorial source grid exceeds its fixed source-count limit of "
            f"{LEDGER_FACTORIAL_MAX_SOURCE_COUNT}"
        )
    calibrated_steps = _calibrated_step_count(run=calibration_run, config=config)
    if step_count != calibrated_steps:
        _fail("factorial step_count differs from the calibrated simulator")
    if candidate_is_noop:
        candidate_allow_buy = True
        candidate_allow_sell = True
    pairs: list[PolicyWorldPair] = []
    slots: list[LedgerFactorialSlotPlan] = []
    for posterior_ordinal in range(posterior_source.posterior_draw_count):
        for scenario_replicate in range(replicate_count):
            ordinal = posterior_ordinal * replicate_count + scenario_replicate
            key = _scenario_event_key(
                plan_id=stable_plan_id,
                event_key_seed=event_key_seed,
                scenario_replicate=scenario_replicate,
            )
            pair = _stable_pair(
                run=calibration_run,
                posterior_ordinal=posterior_ordinal,
                event_key=key,
                config=config,
                candidate_allow_buy=candidate_allow_buy,
                candidate_allow_sell=candidate_allow_sell,
                candidate_is_noop=candidate_is_noop,
                step_count=step_count,
                include_all_groups=include_all_groups,
            )
            pairs.append(pair)
            slots.append(
                _slot_plan(
                    ordinal=ordinal,
                    posterior_ordinal=posterior_ordinal,
                    scenario_replicate=scenario_replicate,
                    run=calibration_run,
                    event_key=key,
                    pair=pair,
                )
            )
    pair_tuple = tuple(pairs)
    _validate_outputs(outputs=outputs, config=config, pairs=pair_tuple)
    analysis_outputs = tuple(
        sorted(
            (
                FactorialOutputSpec(
                    output_id=item.output_id,
                    role=FactorialOutputRole(item.role.value),
                    undefined_policy=FactorialUndefinedPolicy(
                        item.undefined_policy.value
                    ),
                    benefit_direction=FactorialBenefitDirection(
                        item.benefit_direction.value
                    ),
                    source_definition_sha256=sha256_bytes(
                        canonical_json_bytes(item.to_mapping())
                    ),
                )
                for item in outputs
            ),
            key=lambda item: item.output_id,
        )
    )
    primary_output_id = next(
        item.output_id for item in analysis_outputs if item.primary
    )
    stable_outcomes_available_at = _aware(
        outcomes_available_at, "outcomes_available_at"
    )
    analysis_spec = LedgerFactorialSpec(
        spec_id="factorial-analysis-"
        + sha256_bytes(
            canonical_json_bytes(
                {
                    "frozen_at": stable_time.isoformat(),
                    "plan_id": stable_plan_id,
                }
            )
        )[:24],
        posterior_draw_count=posterior_source.posterior_draw_count,
        scenario_replicate_count=replicate_count,
        outputs=analysis_outputs,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
        precision_target=FactorialPrecisionTarget(
            output_id=primary_output_id,
            maximum_bootstrap_mean_variance=maximum_bootstrap_mean_variance,
            frozen_at=stable_time,
            outcomes_available_at=stable_outcomes_available_at,
        ),
        frozen_at=stable_time,
        outcomes_available_at=stable_outcomes_available_at,
    )
    result = object.__new__(LedgerFactorialStudyPlan)
    fields: tuple[tuple[str, object], ...] = (
        ("plan_id", stable_plan_id),
        ("frozen_at", stable_time),
        ("event_key_seed", event_key_seed),
        ("posterior_source", posterior_source),
        ("calibration_run", calibration_run),
        ("config", config),
        ("pairs", pair_tuple),
        ("slots", tuple(slots)),
        ("outputs", outputs),
        ("analysis_spec", analysis_spec),
        ("environment", LedgerExecutionEnvironment.capture()),
        (
            "factorial_implementation_sha256",
            ledger_factorial_study_implementation_sha256(),
        ),
        ("posterior_draw_count", posterior_source.posterior_draw_count),
        ("scenario_replicate_count", replicate_count),
        ("step_count", step_count),
        ("candidate_allow_buy", candidate_allow_buy),
        ("candidate_allow_sell", candidate_allow_sell),
        ("candidate_is_noop", candidate_is_noop),
        ("include_all_groups", include_all_groups),
        (
            "scenario_design",
            FactorialScenarioDesign.BALANCED_POSTERIOR_BY_SCENARIO,
        ),
        (
            "failure_policy",
            "fixed_MxR_all_slots_required_any_failure_aborts_no_replacement",
        ),
        ("synthetic_only", True),
        ("externally_registered", False),
        ("schema_version", LEDGER_FACTORIAL_STUDY_PLAN_SCHEMA_VERSION),
    )
    for field, value in fields:
        object.__setattr__(result, field, value)
    result._fresh_validate()
    return result


def _validate_plan_protocol(
    *, plan: LedgerFactorialStudyPlan, protocol: StudyProtocol
) -> None:
    if type(plan) is not LedgerFactorialStudyPlan:
        raise TypeError("plan must be an exact LedgerFactorialStudyPlan")
    if not isinstance(protocol, StudyProtocol):
        raise TypeError("protocol must be a StudyProtocol")
    plan._fresh_validate()
    if protocol.mode is not StudyMode.SYNTHETIC:
        _fail("factorial source execution accepts only a synthetic protocol")
    if plan.frozen_at > protocol.registered_at:
        _fail("factorial source plan was not frozen by protocol registration")
    if tuple(
        item
        for item in protocol.evidence_plans
        if item.definition_id == LEDGER_FACTORIAL_PLAN_DEFINITION_ID
    ) != (plan.frozen_definition,):
        _fail("study protocol does not bind exactly this factorial source plan")
    if plan.config.cutoff != protocol.knowledge_cutoff:
        _fail("factorial config cutoff differs from its protocol")
    pair = plan.pairs[0]
    if (
        protocol.baseline_policy.definition_id != pair.baseline_policy.policy_id
        or protocol.baseline_policy.definition_sha256 != pair.baseline_policy.sha256
        or protocol.candidate_policy.definition_id != pair.candidate_policy.policy_id
        or protocol.candidate_policy.definition_sha256 != pair.candidate_policy.sha256
    ):
        _fail("factorial policies differ from the source protocol")
    protocol_horizons = {
        item.definition_id: item.definition_sha256 for item in protocol.horizons
    }
    expected_horizons = {
        item.horizon_id: sha256_bytes(canonical_json_bytes(item.to_mapping()))
        for item in pair.horizons
    }
    if protocol_horizons != expected_horizons:
        _fail("factorial horizons differ from the source protocol")
    protocol_cohorts = {item.cohort_id: item for item in protocol.cohorts}
    expected_cohorts = {item.group_id: item.cohort_sha256 for item in pair.groups}
    if set(protocol_cohorts) != set(expected_cohorts) or any(
        protocol_cohorts[group_id].definition_sha256 != digest
        or protocol_cohorts[group_id].basis is not CohortBasis.LATENT_STRATEGY
        for group_id, digest in expected_cohorts.items()
    ):
        _fail("factorial cohorts differ from the source protocol")


@dataclass(frozen=True, slots=True, init=False)
class PreparedLedgerFactorialStudy:
    """Opaque fixed-grid authorities that have not yet been consumed."""

    plan: LedgerFactorialStudyPlan
    protocol: StudyProtocol
    batch_authorization: ProtocolExecutionAuthorization
    source_authorizations: tuple[ProtocolExecutionAuthorization, ...]
    execution_started_at: datetime

    def __init__(self) -> None:
        raise TypeError(
            "PreparedLedgerFactorialStudy is issuer-produced; use "
            "authorize_ledger_factorial_study"
        )


def authorize_ledger_factorial_study(
    *,
    plan: LedgerFactorialStudyPlan,
    protocol: StudyProtocol,
    execution_started_at: datetime,
) -> PreparedLedgerFactorialStudy:
    """Issue the batch and every M by R source authority before execution."""
    _validate_plan_protocol(plan=plan, protocol=protocol)
    started = _aware(execution_started_at, "execution_started_at")
    issuer = SyntheticProtocolExecutionIssuer(protocol)
    batch_authorization = issuer.issue(
        execution_id=f"{plan.plan_id}-factorial-batch",
        consumer=ProtocolExecutionConsumer.LEDGER_FACTORIAL_EVALUATION,
        purpose=ProtocolExecutionPurpose.SYNTHETIC_LEDGER_FACTORIAL_EVALUATION,
        config_sha256=plan.sha256,
        input_sha256=plan.slot_manifest_sha256,
        execution_started_at=started,
    )
    source_authorizations = tuple(
        issuer.issue(
            execution_id=(
                f"{plan.plan_id}-source-p{slot.posterior_ordinal:04d}"
                f"-s{slot.scenario_replicate:04d}"
            ),
            consumer=ProtocolExecutionConsumer.INTEGRATED_SYNTHETIC_SIMULATION,
            purpose=ProtocolExecutionPurpose.PAIRED_SYNTHETIC_POLICY_SIMULATION,
            config_sha256=plan.config.sha256,
            input_sha256=slot.pair_input_sha256,
            execution_started_at=started,
        )
        for slot in plan.slots
    )
    prepared = object.__new__(PreparedLedgerFactorialStudy)
    object.__setattr__(prepared, "plan", plan)
    object.__setattr__(prepared, "protocol", protocol)
    object.__setattr__(prepared, "batch_authorization", batch_authorization)
    object.__setattr__(prepared, "source_authorizations", source_authorizations)
    object.__setattr__(prepared, "execution_started_at", started)
    return prepared


@dataclass(frozen=True, slots=True)
class LedgerFactorialSlotExecution:
    """One completed source slot and its exact ledger lineage."""

    ordinal: int
    posterior_ordinal: int
    scenario_replicate: int
    slot_plan_sha256: str
    pair_input_sha256: str
    source_execution_id: str
    source_receipt_sha256: str
    source_artifact_sha256: str
    ledger_evaluation_sha256: str
    exogenous_event_manifest_sha256: str
    status: Literal["completed"] = "completed"

    def __post_init__(self) -> None:
        for field in ("ordinal", "posterior_ordinal", "scenario_replicate"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                _fail(f"{field} must be a non-negative integer")
        _identifier(self.source_execution_id, "source_execution_id")
        for field in (
            "slot_plan_sha256",
            "pair_input_sha256",
            "source_receipt_sha256",
            "source_artifact_sha256",
            "ledger_evaluation_sha256",
            "exogenous_event_manifest_sha256",
        ):
            _sha256(getattr(self, field), field)
        if self.status != "completed":
            _fail("factorial source batch cannot retain a failed slot as complete")

    def to_mapping(self) -> dict[str, object]:
        return {
            "exogenous_event_manifest_sha256": (self.exogenous_event_manifest_sha256),
            "ledger_evaluation_sha256": self.ledger_evaluation_sha256,
            "ordinal": self.ordinal,
            "pair_input_sha256": self.pair_input_sha256,
            "posterior_ordinal": self.posterior_ordinal,
            "scenario_replicate": self.scenario_replicate,
            "slot_plan_sha256": self.slot_plan_sha256,
            "source_artifact_sha256": self.source_artifact_sha256,
            "source_execution_id": self.source_execution_id,
            "source_receipt_sha256": self.source_receipt_sha256,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True, init=False, weakref_slot=True)
class LedgerFactorialSourceBatch:
    """Complete grid value with one explicit process-local verifier origin.

    Scientific consumers must call
    :func:`require_verified_ledger_factorial_source_batch` when current-process
    execution authority is required, or
    :func:`require_ledger_factorial_derivation_source_batch` for pure derivation
    from either executed or strictly loaded-and-replayed sources. The concrete
    type and its booleans alone are not origin authority. No uncertainty claim
    is made at this layer.
    """

    plan_sha256: str
    protocol_sha256: str
    batch_receipt: ProtocolExecutionReceipt
    batch_receipt_sha256: str
    executions: tuple[LedgerFactorialSlotExecution, ...]
    ledger_evaluations: tuple[LedgerDerivedEvaluation, ...]
    source_grid_complete: Literal[True]
    posterior_axis_distinct_from_scenario_axis: Literal[True]
    exogenous_scenarios_shared_across_posterior_rows: Literal[True]
    failed_slots_silently_dropped: Literal[False]
    synthetic_only: Literal[True]
    externally_preregistered: Literal[False]
    independent_reproduction: Literal[False]
    publication_eligible: Literal[False]
    claim_scope: str
    schema_version: str

    def __init__(self) -> None:
        raise TypeError(
            "LedgerFactorialSourceBatch is verifier-produced; use "
            "run_prepared_ledger_factorial_study or the strict bytes verifier"
        )

    def __copy__(self) -> NoReturn:
        raise TypeError("factorial source batches cannot be copied")

    def __deepcopy__(self, memo: object) -> NoReturn:
        del memo
        raise TypeError("factorial source batches cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("factorial source batches cannot be serialized")

    def _fresh_validate(self) -> None:
        _sha256(self.plan_sha256, "plan_sha256")
        _sha256(self.protocol_sha256, "protocol_sha256")
        _sha256(self.batch_receipt_sha256, "batch_receipt_sha256")
        if self.batch_receipt.sha256 != self.batch_receipt_sha256:
            _fail("factorial batch receipt hash mismatch")
        if not self.executions or len(self.executions) != len(self.ledger_evaluations):
            _fail("factorial source batch lost a source or ledger evaluation")
        if any(
            execution.ledger_evaluation_sha256 != ledger.sha256
            for execution, ledger in zip(
                self.executions, self.ledger_evaluations, strict=True
            )
        ):
            _fail("factorial source batch ledger hashes do not match typed ledgers")
        if tuple(item.ordinal for item in self.executions) != tuple(
            range(len(self.executions))
        ):
            _fail("factorial source executions are not complete row-major slots")
        if (
            self.source_grid_complete is not True
            or self.posterior_axis_distinct_from_scenario_axis is not True
            or self.exogenous_scenarios_shared_across_posterior_rows is not True
            or self.failed_slots_silently_dropped is not False
            or self.synthetic_only is not True
            or self.externally_preregistered is not False
            or self.independent_reproduction is not False
            or self.publication_eligible is not False
            or self.claim_scope != LEDGER_FACTORIAL_SOURCE_CLAIM_SCOPE
            or self.schema_version != LEDGER_FACTORIAL_SOURCE_BATCH_SCHEMA_VERSION
        ):
            _fail("factorial source batch claim boundary changed")

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_mapping())

    @property
    def sha256(self) -> str:
        return sha256_bytes(self.canonical_bytes)

    def to_mapping(self) -> dict[str, object]:
        return {
            "batch_receipt": self.batch_receipt.to_mapping(),
            "batch_receipt_sha256": self.batch_receipt_sha256,
            "claim_scope": self.claim_scope,
            "executions": [item.to_mapping() for item in self.executions],
            "exogenous_scenarios_shared_across_posterior_rows": (
                self.exogenous_scenarios_shared_across_posterior_rows
            ),
            "externally_preregistered": self.externally_preregistered,
            "failed_slots_silently_dropped": self.failed_slots_silently_dropped,
            "independent_reproduction": self.independent_reproduction,
            "ledger_evaluation_sha256s": [
                item.sha256 for item in self.ledger_evaluations
            ],
            "plan_sha256": self.plan_sha256,
            "posterior_axis_distinct_from_scenario_axis": (
                self.posterior_axis_distinct_from_scenario_axis
            ),
            "protocol_sha256": self.protocol_sha256,
            "publication_eligible": self.publication_eligible,
            "schema_version": self.schema_version,
            "source_grid_complete": self.source_grid_complete,
            "synthetic_only": self.synthetic_only,
        }


class _LedgerFactorialSourceBatchMintTicket:
    """Opaque one-use proof of one exact verifier-owned origin and binding."""

    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("factorial source-batch mint tickets are verifier-issued")

    def __copy__(self) -> NoReturn:
        raise TypeError("factorial source-batch mint tickets cannot be copied")

    def __deepcopy__(self, memo: object) -> NoReturn:
        del memo
        raise TypeError("factorial source-batch mint tickets cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("factorial source-batch mint tickets cannot be serialized")


@dataclass(frozen=True, slots=True)
class _LedgerFactorialSourceBatchMintBinding:
    origin: LedgerFactorialSourceBatchOrigin
    plan_sha256: str
    protocol_sha256: str
    batch_receipt_sha256: str
    expected_batch_sha256: str | None
    expected_source_artifact_manifest_sha256: str | None


_FACTORIAL_SOURCE_BATCH_MINT_TICKETS: WeakKeyDictionary[
    _LedgerFactorialSourceBatchMintTicket,
    _LedgerFactorialSourceBatchMintBinding,
] = WeakKeyDictionary()
_VERIFIED_FACTORIAL_SOURCE_BATCHES: WeakValueDictionary[
    int, LedgerFactorialSourceBatch
] = WeakValueDictionary()
_VERIFIED_FACTORIAL_SOURCE_BATCH_SEALS: dict[int, str] = {}
_VERIFIED_FACTORIAL_SOURCE_BATCH_ORIGINS: dict[
    int, LedgerFactorialSourceBatchOrigin
] = {}
_VERIFIED_FACTORIAL_SOURCE_BATCHES_LOCK = RLock()


def _factorial_source_batch_seal(batch: LedgerFactorialSourceBatch) -> str:
    return sha256_bytes(batch.canonical_bytes)


def _factorial_source_artifact_manifest_sha256(
    batch: LedgerFactorialSourceBatch,
) -> str:
    return sha256_bytes(
        canonical_json_bytes([item.source_artifact_sha256 for item in batch.executions])
    )


def _forget_factorial_source_batch(identity: int) -> None:
    with _VERIFIED_FACTORIAL_SOURCE_BATCHES_LOCK:
        if _VERIFIED_FACTORIAL_SOURCE_BATCHES.get(identity) is None:
            _VERIFIED_FACTORIAL_SOURCE_BATCH_SEALS.pop(identity, None)
            _VERIFIED_FACTORIAL_SOURCE_BATCH_ORIGINS.pop(identity, None)


def _mint_factorial_source_batch(
    *,
    ticket: _LedgerFactorialSourceBatchMintTicket,
    batch: LedgerFactorialSourceBatch,
) -> LedgerFactorialSourceBatch:
    if type(ticket) is not _LedgerFactorialSourceBatchMintTicket:
        _fail("factorial source-batch mint requires an exact verifier ticket")
    if type(batch) is not LedgerFactorialSourceBatch:
        _fail("factorial source-batch mint requires an exact batch value")
    with _VERIFIED_FACTORIAL_SOURCE_BATCHES_LOCK:
        binding = _FACTORIAL_SOURCE_BATCH_MINT_TICKETS.pop(ticket, None)
    if binding is None:
        _fail("factorial source-batch mint ticket is invalid or already consumed")
    if type(binding.origin) is not LedgerFactorialSourceBatchOrigin or (
        binding.plan_sha256 != batch.plan_sha256
        or binding.protocol_sha256 != batch.protocol_sha256
        or binding.batch_receipt_sha256 != batch.batch_receipt_sha256
        or (
            binding.expected_batch_sha256 is not None
            and binding.expected_batch_sha256 != batch.sha256
        )
        or (
            binding.expected_source_artifact_manifest_sha256 is not None
            and binding.expected_source_artifact_manifest_sha256
            != _factorial_source_artifact_manifest_sha256(batch)
        )
    ):
        _fail("factorial source batch differs from its verifier mint-ticket binding")
    batch._fresh_validate()
    identity = id(batch)
    seal = _factorial_source_batch_seal(batch)
    with _VERIFIED_FACTORIAL_SOURCE_BATCHES_LOCK:
        _VERIFIED_FACTORIAL_SOURCE_BATCHES[identity] = batch
        _VERIFIED_FACTORIAL_SOURCE_BATCH_SEALS[identity] = seal
        _VERIFIED_FACTORIAL_SOURCE_BATCH_ORIGINS[identity] = binding.origin
    finalize(batch, _forget_factorial_source_batch, identity)
    return batch


def _require_registered_ledger_factorial_source_batch(
    batch: LedgerFactorialSourceBatch,
    *,
    allowed_origins: frozenset[LedgerFactorialSourceBatchOrigin],
) -> LedgerFactorialSourceBatch:
    """Require one stable same-process batch from an explicit origin set."""
    if type(batch) is not LedgerFactorialSourceBatch:
        _fail("factorial source batch has the wrong concrete type")
    try:
        before_seal = _factorial_source_batch_seal(batch)
    except Exception as error:
        raise LedgerFactorialStudyError(
            "factorial source batch cannot reproduce its mint-time seal"
        ) from error
    with _VERIFIED_FACTORIAL_SOURCE_BATCHES_LOCK:
        registered = _VERIFIED_FACTORIAL_SOURCE_BATCHES.get(id(batch))
        mint_seal = _VERIFIED_FACTORIAL_SOURCE_BATCH_SEALS.get(id(batch))
        origin = _VERIFIED_FACTORIAL_SOURCE_BATCH_ORIGINS.get(id(batch))
    if registered is not batch or mint_seal != before_seal or origin is None:
        _fail(
            "factorial source batch was not minted by the verifier-owned runner "
            "or strict replay loader, or differs from its mint-time seal"
        )
    if origin not in allowed_origins:
        _fail(
            "factorial source batch origin is not authorized for this operation; "
            f"origin={origin.value}"
        )
    batch._fresh_validate()
    after_seal = _factorial_source_batch_seal(batch)
    with _VERIFIED_FACTORIAL_SOURCE_BATCHES_LOCK:
        still_registered = _VERIFIED_FACTORIAL_SOURCE_BATCHES.get(id(batch))
        current_seal = _VERIFIED_FACTORIAL_SOURCE_BATCH_SEALS.get(id(batch))
        current_origin = _VERIFIED_FACTORIAL_SOURCE_BATCH_ORIGINS.get(id(batch))
    if (
        still_registered is not batch
        or before_seal != after_seal
        or current_seal != after_seal
        or current_origin is not origin
    ):
        _fail("factorial source batch changed after its verifier-owned mint")
    return batch


def require_executed_ledger_factorial_source_batch(
    batch: LedgerFactorialSourceBatch,
) -> LedgerFactorialSourceBatch:
    """Require a batch minted only after current-process authority consumption."""
    return _require_registered_ledger_factorial_source_batch(
        batch,
        allowed_origins=frozenset({LedgerFactorialSourceBatchOrigin.EXECUTED_HERE}),
    )


def require_verified_ledger_factorial_source_batch(
    batch: LedgerFactorialSourceBatch,
) -> LedgerFactorialSourceBatch:
    """Backward-compatible executed-only source-batch origin guard."""
    return require_executed_ledger_factorial_source_batch(batch)


def require_ledger_factorial_derivation_source_batch(
    batch: LedgerFactorialSourceBatch,
) -> LedgerFactorialSourceBatch:
    """Accept executed or strictly replayed batches for pure derivation only.

    This guard does not authorize simulation, prove historical authority
    consumption, or upgrade any external, execution, publication, or causal
    claim.
    """
    return _require_registered_ledger_factorial_source_batch(
        batch,
        allowed_origins=frozenset(
            {
                LedgerFactorialSourceBatchOrigin.EXECUTED_HERE,
                LedgerFactorialSourceBatchOrigin.LOADED_AND_REPLAYED,
            }
        ),
    )


def _source_execution(
    *,
    slot: LedgerFactorialSlotPlan,
    artifact: ProtocolBoundIntegratedSyntheticRunArtifact,
    ledger: LedgerDerivedEvaluation,
) -> LedgerFactorialSlotExecution:
    pair = artifact.pair_input
    if (
        artifact.input_sha256 != slot.pair_input_sha256
        or integrated_synthetic_pair_input_sha256(pair) != slot.pair_input_sha256
        or pair.posterior_draw.draw_id != slot.posterior_draw_id
        or pair.posterior_draw.parameter_sha256 != slot.parameter_sha256
        or pair.scenario_id != slot.scenario_id
        or pair.shock_tape.sha256 != slot.shock_tape_sha256
        or pair.initial_state.sha256 != slot.initial_state_sha256
    ):
        _fail("factorial source artifact differs from its frozen grid slot")
    event_manifest = tuple(
        item.event_key.to_mapping() for item in pair.shock_tape.entries
    )
    return LedgerFactorialSlotExecution(
        ordinal=slot.ordinal,
        posterior_ordinal=slot.posterior_ordinal,
        scenario_replicate=slot.scenario_replicate,
        slot_plan_sha256=slot.sha256,
        pair_input_sha256=slot.pair_input_sha256,
        source_execution_id=artifact.protocol_execution_receipt.execution_id,
        source_receipt_sha256=artifact.protocol_execution_receipt.sha256,
        source_artifact_sha256=artifact.sha256,
        ledger_evaluation_sha256=ledger.sha256,
        exogenous_event_manifest_sha256=sha256_bytes(
            canonical_json_bytes(event_manifest)
        ),
    )


def _derive_source_batch(
    *,
    plan: LedgerFactorialStudyPlan,
    protocol: StudyProtocol,
    batch_receipt: ProtocolExecutionReceipt,
    source_artifacts: Sequence[ProtocolBoundIntegratedSyntheticRunArtifact],
) -> LedgerFactorialSourceBatch:
    _validate_plan_protocol(plan=plan, protocol=protocol)
    validate_protocol_execution_receipt_binding(protocol, batch_receipt)
    if (
        batch_receipt.consumer
        is not ProtocolExecutionConsumer.LEDGER_FACTORIAL_EVALUATION
        or batch_receipt.purpose
        is not ProtocolExecutionPurpose.SYNTHETIC_LEDGER_FACTORIAL_EVALUATION
        or batch_receipt.config_sha256 != plan.sha256
        or batch_receipt.input_sha256 != plan.slot_manifest_sha256
    ):
        _fail("factorial batch receipt does not bind the frozen source grid")
    artifacts = tuple(source_artifacts)
    if len(artifacts) != len(plan.slots):
        _fail("factorial source artifact count does not cover the complete grid")
    by_input: dict[str, ProtocolBoundIntegratedSyntheticRunArtifact] = {}
    for artifact in artifacts:
        if type(artifact) is not ProtocolBoundIntegratedSyntheticRunArtifact:
            raise TypeError("factorial source artifact has the wrong exact type")
        receipt = artifact.protocol_execution_receipt
        if (
            artifact.protocol != protocol
            or artifact.protocol.sha256 != protocol.sha256
            or artifact.config != plan.config
            or artifact.config_sha256 != plan.config.sha256
            or receipt.consumer
            is not ProtocolExecutionConsumer.INTEGRATED_SYNTHETIC_SIMULATION
            or receipt.purpose
            is not ProtocolExecutionPurpose.PAIRED_SYNTHETIC_POLICY_SIMULATION
            or receipt.protocol_sha256 != protocol.sha256
            or receipt.config_sha256 != plan.config.sha256
            or receipt.input_sha256 != artifact.input_sha256
        ):
            _fail("factorial source artifact protocol or receipt binding differs")
        if artifact.input_sha256 in by_input:
            _fail("factorial source artifact was reused across grid slots")
        by_input[artifact.input_sha256] = artifact
    if set(by_input) != {item.pair_input_sha256 for item in plan.slots}:
        _fail("factorial source artifacts contain missing or replacement slots")
    ordered = tuple(by_input[item.pair_input_sha256] for item in plan.slots)
    ledgers = tuple(derive_protocol_bound_ledger_evidence(item) for item in ordered)
    executions = tuple(
        _source_execution(slot=slot, artifact=artifact, ledger=ledger)
        for slot, artifact, ledger in zip(plan.slots, ordered, ledgers, strict=True)
    )
    if len({item.source_execution_id for item in executions}) != len(executions):
        _fail("factorial source execution IDs must be unique")
    for scenario_replicate in range(plan.scenario_replicate_count):
        column = tuple(
            executions[
                posterior_ordinal * plan.scenario_replicate_count + scenario_replicate
            ]
            for posterior_ordinal in range(plan.posterior_draw_count)
        )
        if len({item.exogenous_event_manifest_sha256 for item in column}) != 1:
            _fail("factorial exogenous events differ across posterior rows")
    result = object.__new__(LedgerFactorialSourceBatch)
    fields: tuple[tuple[str, object], ...] = (
        ("plan_sha256", plan.sha256),
        ("protocol_sha256", protocol.sha256),
        ("batch_receipt", batch_receipt),
        ("batch_receipt_sha256", batch_receipt.sha256),
        ("executions", executions),
        ("ledger_evaluations", ledgers),
        ("source_grid_complete", True),
        ("posterior_axis_distinct_from_scenario_axis", True),
        ("exogenous_scenarios_shared_across_posterior_rows", True),
        ("failed_slots_silently_dropped", False),
        ("synthetic_only", True),
        ("externally_preregistered", False),
        ("independent_reproduction", False),
        ("publication_eligible", False),
        ("claim_scope", LEDGER_FACTORIAL_SOURCE_CLAIM_SCOPE),
        ("schema_version", LEDGER_FACTORIAL_SOURCE_BATCH_SCHEMA_VERSION),
    )
    for field, value in fields:
        object.__setattr__(result, field, value)
    result._fresh_validate()
    return result


def _derive_and_mint_executed_factorial_source_batch(
    *,
    plan: LedgerFactorialStudyPlan,
    protocol: StudyProtocol,
    batch_receipt: ProtocolExecutionReceipt,
    source_artifacts: Sequence[ProtocolBoundIntegratedSyntheticRunArtifact],
) -> LedgerFactorialSourceBatch:
    """Guard executed source identities, derive the exact batch, and mint it.

    Deliberately no caller-supplied batch value is accepted.  Loaded replay
    artifacts fail before derivation and before a ticket exists.  The ticket is
    created only after complete source replay, binds the exact batch and ordered
    source manifest, and is consumed by the immediately following call.
    """
    artifacts = tuple(source_artifacts)
    for artifact in artifacts:
        require_verifier_owned_protocol_bound_integrated_artifact(artifact)
    batch = _derive_source_batch(
        plan=plan,
        protocol=protocol,
        batch_receipt=batch_receipt,
        source_artifacts=artifacts,
    )
    mint_ticket = object.__new__(_LedgerFactorialSourceBatchMintTicket)
    with _VERIFIED_FACTORIAL_SOURCE_BATCHES_LOCK:
        _FACTORIAL_SOURCE_BATCH_MINT_TICKETS[mint_ticket] = (
            _LedgerFactorialSourceBatchMintBinding(
                origin=LedgerFactorialSourceBatchOrigin.EXECUTED_HERE,
                plan_sha256=batch.plan_sha256,
                protocol_sha256=batch.protocol_sha256,
                batch_receipt_sha256=batch.batch_receipt_sha256,
                expected_batch_sha256=batch.sha256,
                expected_source_artifact_manifest_sha256=(
                    _factorial_source_artifact_manifest_sha256(batch)
                ),
            )
        )
    return _mint_factorial_source_batch(ticket=mint_ticket, batch=batch)


def verify_ledger_factorial_source_batch_bytes(
    content: bytes,
    *,
    expected_sha256: str,
    plan: LedgerFactorialStudyPlan,
    protocol: StudyProtocol,
    batch_receipt: ProtocolExecutionReceipt,
    source_artifacts: Sequence[ProtocolBoundIntegratedSyntheticRunArtifact],
) -> LedgerFactorialSourceBatch:
    """Strictly replay canonical batch bytes without minting execution authority.

    Every supplied source is independently accepted only through the pure ledger
    derivation boundary inside :func:`_derive_source_batch`. The historical batch
    receipt remains audit data: this verifier never consumes an authorization and
    the returned batch receives only ``loaded_and_replayed`` origin.
    """
    if type(content) is not bytes:
        raise TypeError("factorial source-batch content must be exact bytes")
    expected = _sha256(expected_sha256, "expected_sha256")
    if not content or len(content) > _MAX_FACTORIAL_SOURCE_BATCH_BYTES:
        _fail("factorial source-batch bytes are empty or exceed their byte limit")
    if sha256_bytes(content) != expected:
        _fail("factorial source-batch bytes differ from their external commitment")
    try:
        mapping = read_json_bytes_object(
            content,
            source="ledger factorial source-batch bytes",
        )
    except MetadataValidationError as error:
        raise LedgerFactorialStudyError(str(error)) from error
    if canonical_json_bytes(mapping) != content:
        _fail("factorial source-batch bytes must use canonical duplicate-free JSON")
    if type(batch_receipt) is not ProtocolExecutionReceipt:
        raise TypeError("batch_receipt must be an exact ProtocolExecutionReceipt")
    receipt_mapping = mapping.get("batch_receipt")
    if not isinstance(receipt_mapping, Mapping):
        _fail("factorial source-batch bytes lack a structured batch receipt")
    try:
        embedded_receipt = ProtocolExecutionReceipt.from_mapping(receipt_mapping)
    except Exception as error:
        raise LedgerFactorialStudyError(
            "factorial source-batch receipt cannot be strictly reconstructed"
        ) from error
    if embedded_receipt != batch_receipt:
        _fail("factorial source-batch bytes differ from the supplied batch receipt")
    try:
        rederived = _derive_source_batch(
            plan=plan,
            protocol=protocol,
            batch_receipt=batch_receipt,
            source_artifacts=source_artifacts,
        )
    except Exception as error:
        raise LedgerFactorialStudyError(
            f"factorial source-batch source replay failed: {error}"
        ) from error
    if rederived.canonical_bytes != content:
        _fail("factorial source-batch bytes differ from complete source rederivation")
    mint_ticket = object.__new__(_LedgerFactorialSourceBatchMintTicket)
    with _VERIFIED_FACTORIAL_SOURCE_BATCHES_LOCK:
        _FACTORIAL_SOURCE_BATCH_MINT_TICKETS[mint_ticket] = (
            _LedgerFactorialSourceBatchMintBinding(
                origin=LedgerFactorialSourceBatchOrigin.LOADED_AND_REPLAYED,
                plan_sha256=rederived.plan_sha256,
                protocol_sha256=rederived.protocol_sha256,
                batch_receipt_sha256=rederived.batch_receipt_sha256,
                expected_batch_sha256=expected,
                expected_source_artifact_manifest_sha256=(
                    _factorial_source_artifact_manifest_sha256(rederived)
                ),
            )
        )
    return _mint_factorial_source_batch(ticket=mint_ticket, batch=rederived)


def run_prepared_ledger_factorial_study(
    prepared: PreparedLedgerFactorialStudy,
    *,
    parallel_slots: bool = False,
    parallel_worlds: bool = True,
) -> tuple[
    LedgerFactorialSourceBatch,
    tuple[ProtocolBoundIntegratedSyntheticRunArtifact, ...],
]:
    """Consume authority, execute every fixed slot, then mint one exact batch."""
    if type(prepared) is not PreparedLedgerFactorialStudy:
        raise TypeError("prepared must be an exact PreparedLedgerFactorialStudy")
    plan = prepared.plan
    batch_receipt = consume_protocol_execution_authorization(
        prepared.batch_authorization,
        protocol=prepared.protocol,
        consumer=ProtocolExecutionConsumer.LEDGER_FACTORIAL_EVALUATION,
        purpose=ProtocolExecutionPurpose.SYNTHETIC_LEDGER_FACTORIAL_EVALUATION,
        config_sha256=plan.sha256,
        input_sha256=plan.slot_manifest_sha256,
        execution_started_at=prepared.execution_started_at,
    )
    plan._fresh_validate()

    def run_one(
        material: tuple[PolicyWorldPair, ProtocolExecutionAuthorization],
    ) -> ProtocolBoundIntegratedSyntheticRunArtifact:
        pair, authorization = material
        return run_protocol_bound_integrated_synthetic_pair(
            pair=pair,
            config=plan.config,
            protocol=prepared.protocol,
            authorization=authorization,
            execution_started_at=prepared.execution_started_at,
            parallel=parallel_worlds,
            invocation_order=(WorldRole.BASELINE, WorldRole.CANDIDATE),
        )

    material = tuple(zip(plan.pairs, prepared.source_authorizations, strict=True))
    if parallel_slots:
        with ThreadPoolExecutor(max_workers=min(4, len(material))) as executor:
            artifacts = tuple(executor.map(run_one, material))
    else:
        artifacts = tuple(run_one(item) for item in material)
    batch = _derive_and_mint_executed_factorial_source_batch(
        plan=plan,
        protocol=prepared.protocol,
        batch_receipt=batch_receipt,
        source_artifacts=artifacts,
    )
    return batch, artifacts


def _factorial_grid_from_source_batch(
    *,
    plan: LedgerFactorialStudyPlan,
    batch: LedgerFactorialSourceBatch,
) -> tuple[FactorialGridCell, ...]:
    """Rebuild every exact analysis cell from retained typed ledgers."""
    plan._fresh_validate()
    if type(batch) is not LedgerFactorialSourceBatch:
        raise TypeError("batch must be an exact LedgerFactorialSourceBatch")
    batch._fresh_validate()
    if batch.plan_sha256 != plan.sha256 or len(batch.executions) != len(plan.slots):
        _fail("factorial source batch differs from the frozen study plan")
    ordered_outputs = tuple(sorted(plan.outputs, key=lambda item: item.output_id))
    cells: list[FactorialGridCell] = []
    for slot, execution, ledger in zip(
        plan.slots,
        batch.executions,
        batch.ledger_evaluations,
        strict=True,
    ):
        if (
            execution.ordinal != slot.ordinal
            or execution.posterior_ordinal != slot.posterior_ordinal
            or execution.scenario_replicate != slot.scenario_replicate
            or execution.slot_plan_sha256 != slot.sha256
            or execution.pair_input_sha256 != slot.pair_input_sha256
            or execution.ledger_evaluation_sha256 != ledger.sha256
        ):
            _fail("factorial source execution differs from its grid slot")
        outcomes: list[FactorialOutcome] = []
        for output in ordered_outputs:
            matches = tuple(
                item
                for item in ledger.differences
                if (
                    item.horizon_id,
                    item.scope,
                    item.group_id,
                    item.metric_id,
                )
                == (
                    output.horizon_id,
                    output.scope,
                    output.group_id,
                    output.metric_id,
                )
            )
            if len(matches) != 1:
                _fail(
                    "registered factorial output is missing or duplicated in a "
                    "source ledger"
                )
            outcomes.append(
                FactorialOutcome.from_transition(
                    output_id=output.output_id,
                    baseline=matches[0].baseline,
                    candidate=matches[0].candidate,
                )
            )
        posterior_commitment = sha256_bytes(
            canonical_json_bytes(
                {
                    "calibration_draw_sha256": slot.calibration_draw_sha256,
                    "parameter_sha256": slot.parameter_sha256,
                    "posterior_draw_id": slot.posterior_draw_id,
                    "posterior_ordinal": slot.posterior_ordinal,
                    "posterior_resample_event_key": (slot.posterior_resample_event_key),
                }
            )
        )
        scenario_commitment = sha256_bytes(
            canonical_json_bytes(
                {
                    "exogenous_event_manifest_sha256": (
                        execution.exogenous_event_manifest_sha256
                    ),
                    "scenario_event_key": slot.scenario_event_key,
                    "scenario_event_key_canonical_text": (
                        slot.scenario_event_key_canonical_text
                    ),
                    "scenario_id": slot.scenario_id,
                    "scenario_replicate": slot.scenario_replicate,
                    "shock_tape_sha256": slot.shock_tape_sha256,
                    "initial_state_sha256": slot.initial_state_sha256,
                }
            )
        )
        cells.append(
            FactorialGridCell(
                posterior_index=slot.posterior_ordinal,
                scenario_index=slot.scenario_replicate,
                ordinal=slot.ordinal,
                posterior_commitment_sha256=posterior_commitment,
                scenario_commitment_sha256=scenario_commitment,
                outcomes=tuple(outcomes),
            )
        )
    return tuple(cells)


@dataclass(frozen=True, slots=True, init=False, weakref_slot=True)
class SourceBoundLedgerFactorialEvaluation:
    """Fresh source-derived factorial result with deliberately narrow claims.

    Consumers of process-local source origin must call
    :func:`require_source_bound_ledger_factorial_evaluation`. Canonical bytes
    remain value evidence and require full typed-source replay after loading.
    """

    plan: LedgerFactorialStudyPlan
    protocol: StudyProtocol
    source_batch: LedgerFactorialSourceBatch
    grid: tuple[FactorialGridCell, ...]
    analysis: LedgerFactorialEvaluation
    plan_sha256: str
    protocol_sha256: str
    source_batch_sha256: str
    analysis_sha256: str
    source_artifact_sha256s: tuple[str, ...]
    source_grid_rederived_from_verified_artifacts: Literal[True]
    fixed_grid_factorial_arithmetic_verified: Literal[True]
    registered_axes_separated: Literal[True]
    real_world_uncertainty_components_identified: Literal[False]
    source_artifacts_independently_verified: Literal[False]
    source_execution_identity_externally_attested: Literal[False]
    rooted_bundle_custody_verified: Literal[False]
    analysis_chronology_externally_verified: Literal[False]
    synthetic_only: Literal[True]
    real_world_causal_claim: Literal[False]
    credible_interval_claim: Literal[False]
    confidence_interval_claim: Literal[False]
    iid_mcse_reported: Literal[False]
    independent_reproduction: Literal[False]
    publication_eligible: Literal[False]
    abstention_required: bool
    claim_scope: str
    schema_version: str

    def __init__(self) -> None:
        raise TypeError(
            "SourceBoundLedgerFactorialEvaluation is verifier-produced; use "
            "derive_source_bound_ledger_factorial_evaluation"
        )

    def __copy__(self) -> NoReturn:
        raise TypeError("source-bound factorial evaluations cannot be copied")

    def __deepcopy__(self, memo: object) -> NoReturn:
        del memo
        raise TypeError("source-bound factorial evaluations cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("source-bound factorial evaluations cannot be serialized")

    def _fresh_validate(self) -> None:
        if type(self.plan) is not LedgerFactorialStudyPlan:
            _fail("source-bound factorial result lost its typed plan")
        if not isinstance(self.protocol, StudyProtocol):
            _fail("source-bound factorial result lost its typed protocol")
        if type(self.source_batch) is not LedgerFactorialSourceBatch:
            _fail("source-bound factorial result lost its typed source batch")
        if not isinstance(self.grid, tuple) or not all(
            type(item) is FactorialGridCell for item in self.grid
        ):
            _fail("source-bound factorial result lost its typed complete grid")
        if type(self.analysis) is not LedgerFactorialEvaluation:
            _fail("source-bound factorial result lost its typed analysis")
        self.plan._fresh_validate()
        _validate_plan_protocol(plan=self.plan, protocol=self.protocol)
        require_ledger_factorial_derivation_source_batch(self.source_batch)
        for field in (
            "plan_sha256",
            "protocol_sha256",
            "source_batch_sha256",
            "analysis_sha256",
        ):
            _sha256(getattr(self, field), field)
        if (
            self.plan_sha256 != self.plan.sha256
            or self.protocol_sha256 != self.protocol.sha256
            or self.source_batch_sha256 != self.source_batch.sha256
            or self.analysis_sha256 != self.analysis.sha256
            or self.source_batch.plan_sha256 != self.plan.sha256
            or self.source_batch.protocol_sha256 != self.protocol.sha256
            or self.analysis.spec != self.plan.analysis_spec
        ):
            _fail("source-bound factorial result commitment closure changed")
        expected_source_hashes = tuple(
            item.source_artifact_sha256 for item in self.source_batch.executions
        )
        if self.source_artifact_sha256s != expected_source_hashes or len(
            set(self.source_artifact_sha256s)
        ) != len(self.source_artifact_sha256s):
            _fail("source-bound factorial source-artifact manifest changed")
        expected_grid = _factorial_grid_from_source_batch(
            plan=self.plan,
            batch=self.source_batch,
        )
        if self.grid != expected_grid:
            _fail("source-bound factorial grid differs from typed source ledgers")
        expected_analysis = derive_ledger_factorial_evaluation(
            spec=self.plan.analysis_spec,
            grid=expected_grid,
        )
        if self.analysis.canonical_bytes != expected_analysis.canonical_bytes:
            _fail("source-bound factorial analysis differs from fresh grid derivation")
        if self.abstention_required is not (not self.analysis.precision_audit.passed):
            _fail("source-bound factorial abstention differs from precision audit")
        if (
            self.source_grid_rederived_from_verified_artifacts is not True
            or self.fixed_grid_factorial_arithmetic_verified is not True
            or self.registered_axes_separated is not True
            or self.real_world_uncertainty_components_identified is not False
            or self.source_artifacts_independently_verified is not False
            or self.source_execution_identity_externally_attested is not False
            or self.rooted_bundle_custody_verified is not False
            or self.analysis_chronology_externally_verified is not False
            or self.synthetic_only is not True
            or self.real_world_causal_claim is not False
            or self.credible_interval_claim is not False
            or self.confidence_interval_claim is not False
            or self.iid_mcse_reported is not False
            or self.independent_reproduction is not False
            or self.publication_eligible is not False
            or self.claim_scope != LEDGER_FACTORIAL_SOURCE_BOUND_CLAIM_SCOPE
            or self.schema_version
            != LEDGER_FACTORIAL_SOURCE_BOUND_RESULT_SCHEMA_VERSION
        ):
            _fail("source-bound factorial scientific claim boundary changed")

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_mapping())

    @property
    def sha256(self) -> str:
        return sha256_bytes(self.canonical_bytes)

    def to_mapping(self) -> dict[str, object]:
        return {
            "abstention_required": self.abstention_required,
            "analysis": self.analysis.to_mapping(),
            "analysis_chronology_externally_verified": (
                self.analysis_chronology_externally_verified
            ),
            "analysis_sha256": self.analysis_sha256,
            "claim_scope": self.claim_scope,
            "confidence_interval_claim": self.confidence_interval_claim,
            "credible_interval_claim": self.credible_interval_claim,
            "fixed_grid_factorial_arithmetic_verified": (
                self.fixed_grid_factorial_arithmetic_verified
            ),
            "grid": [item.to_mapping() for item in self.grid],
            "iid_mcse_reported": self.iid_mcse_reported,
            "independent_reproduction": self.independent_reproduction,
            "plan": self.plan.to_mapping(),
            "plan_sha256": self.plan_sha256,
            "protocol": self.protocol.to_mapping(),
            "protocol_sha256": self.protocol_sha256,
            "publication_eligible": self.publication_eligible,
            "real_world_causal_claim": self.real_world_causal_claim,
            "real_world_uncertainty_components_identified": (
                self.real_world_uncertainty_components_identified
            ),
            "registered_axes_separated": self.registered_axes_separated,
            "rooted_bundle_custody_verified": self.rooted_bundle_custody_verified,
            "schema_version": self.schema_version,
            "source_artifact_sha256s": list(self.source_artifact_sha256s),
            "source_artifacts_independently_verified": (
                self.source_artifacts_independently_verified
            ),
            "source_batch": self.source_batch.to_mapping(),
            "source_batch_sha256": self.source_batch_sha256,
            "source_execution_identity_externally_attested": (
                self.source_execution_identity_externally_attested
            ),
            "source_grid_rederived_from_verified_artifacts": (
                self.source_grid_rederived_from_verified_artifacts
            ),
            "synthetic_only": self.synthetic_only,
        }


class _SourceBoundFactorialResultMintTicket:
    """Opaque one-use authority issued only after full typed-source replay."""

    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("source-bound factorial mint tickets are verifier-issued")

    def __copy__(self) -> NoReturn:
        raise TypeError("source-bound factorial mint tickets cannot be copied")

    def __deepcopy__(self, memo: object) -> NoReturn:
        del memo
        raise TypeError("source-bound factorial mint tickets cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("source-bound factorial mint tickets cannot be serialized")


@dataclass(frozen=True, slots=True)
class _SourceBoundFactorialResultMintBinding:
    plan_sha256: str
    protocol_sha256: str
    source_batch_sha256: str
    analysis_sha256: str
    source_artifact_manifest_sha256: str


_SOURCE_BOUND_FACTORIAL_RESULT_MINT_TICKETS: WeakKeyDictionary[
    _SourceBoundFactorialResultMintTicket,
    _SourceBoundFactorialResultMintBinding,
] = WeakKeyDictionary()
_VERIFIED_SOURCE_BOUND_FACTORIAL_RESULTS: WeakValueDictionary[
    int, SourceBoundLedgerFactorialEvaluation
] = WeakValueDictionary()
_VERIFIED_SOURCE_BOUND_FACTORIAL_RESULT_SEALS: dict[int, str] = {}
_VERIFIED_SOURCE_BOUND_FACTORIAL_RESULTS_LOCK = RLock()


def _source_bound_factorial_result_seal(
    result: SourceBoundLedgerFactorialEvaluation,
) -> str:
    return sha256_bytes(result.canonical_bytes)


def _forget_source_bound_factorial_result(identity: int) -> None:
    with _VERIFIED_SOURCE_BOUND_FACTORIAL_RESULTS_LOCK:
        if _VERIFIED_SOURCE_BOUND_FACTORIAL_RESULTS.get(identity) is None:
            _VERIFIED_SOURCE_BOUND_FACTORIAL_RESULT_SEALS.pop(identity, None)


def _mint_source_bound_factorial_result(
    *,
    ticket: _SourceBoundFactorialResultMintTicket,
    result: SourceBoundLedgerFactorialEvaluation,
) -> SourceBoundLedgerFactorialEvaluation:
    if type(ticket) is not _SourceBoundFactorialResultMintTicket:
        _fail("source-bound factorial mint requires an exact verifier ticket")
    with _VERIFIED_SOURCE_BOUND_FACTORIAL_RESULTS_LOCK:
        binding = _SOURCE_BOUND_FACTORIAL_RESULT_MINT_TICKETS.pop(ticket, None)
    if binding is None:
        _fail("source-bound factorial mint ticket is invalid or already consumed")
    source_manifest_sha256 = sha256_bytes(
        canonical_json_bytes(list(result.source_artifact_sha256s))
    )
    if (
        binding.plan_sha256 != result.plan_sha256
        or binding.protocol_sha256 != result.protocol_sha256
        or binding.source_batch_sha256 != result.source_batch_sha256
        or binding.analysis_sha256 != result.analysis_sha256
        or binding.source_artifact_manifest_sha256 != source_manifest_sha256
    ):
        _fail("source-bound factorial result differs from its verifier mint ticket")
    result._fresh_validate()
    identity = id(result)
    seal = _source_bound_factorial_result_seal(result)
    with _VERIFIED_SOURCE_BOUND_FACTORIAL_RESULTS_LOCK:
        _VERIFIED_SOURCE_BOUND_FACTORIAL_RESULTS[identity] = result
        _VERIFIED_SOURCE_BOUND_FACTORIAL_RESULT_SEALS[identity] = seal
    finalize(result, _forget_source_bound_factorial_result, identity)
    return result


def require_source_bound_ledger_factorial_evaluation(
    result: SourceBoundLedgerFactorialEvaluation,
) -> SourceBoundLedgerFactorialEvaluation:
    """Require a same-process result minted by full typed-source rederivation.

    The guard proves only verifier-owned process-local origin and value
    stability.  It does not establish external custody, historical execution,
    independent reproduction, or any wider scientific claim.
    """
    if type(result) is not SourceBoundLedgerFactorialEvaluation:
        _fail("source-bound factorial result has the wrong concrete type")
    try:
        before_seal = _source_bound_factorial_result_seal(result)
    except Exception as error:
        raise LedgerFactorialStudyError(
            "source-bound factorial result cannot reproduce its mint-time seal"
        ) from error
    with _VERIFIED_SOURCE_BOUND_FACTORIAL_RESULTS_LOCK:
        registered = _VERIFIED_SOURCE_BOUND_FACTORIAL_RESULTS.get(id(result))
        mint_seal = _VERIFIED_SOURCE_BOUND_FACTORIAL_RESULT_SEALS.get(id(result))
    if registered is not result or mint_seal != before_seal:
        _fail(
            "source-bound factorial result was not minted by full source "
            "rederivation or differs from its mint-time seal"
        )
    result._fresh_validate()
    try:
        after_seal = _source_bound_factorial_result_seal(result)
    except Exception as error:
        raise LedgerFactorialStudyError(
            "source-bound factorial result changed during seal validation"
        ) from error
    with _VERIFIED_SOURCE_BOUND_FACTORIAL_RESULTS_LOCK:
        still_registered = _VERIFIED_SOURCE_BOUND_FACTORIAL_RESULTS.get(id(result))
        current_seal = _VERIFIED_SOURCE_BOUND_FACTORIAL_RESULT_SEALS.get(id(result))
    if (
        still_registered is not result
        or before_seal != after_seal
        or current_seal != after_seal
    ):
        _fail("source-bound factorial result changed after its verifier-owned mint")
    return result


def require_executed_source_bound_ledger_factorial_evaluation(
    result: SourceBoundLedgerFactorialEvaluation,
) -> SourceBoundLedgerFactorialEvaluation:
    """Require a stable result whose source batch was executed in this process.

    The ordinary result guard deliberately accepts strict historical replay for
    pure derivation.  This stronger boundary composes that value/provenance
    check with the executed-only source-batch guard, preventing a loaded replay
    result from being relabelled as current-process execution.
    """
    verified = require_source_bound_ledger_factorial_evaluation(result)
    require_executed_ledger_factorial_source_batch(verified.source_batch)
    return verified


def derive_source_bound_ledger_factorial_evaluation(
    *,
    plan: LedgerFactorialStudyPlan,
    protocol: StudyProtocol,
    source_batch: LedgerFactorialSourceBatch,
    source_artifacts: Sequence[ProtocolBoundIntegratedSyntheticRunArtifact],
) -> SourceBoundLedgerFactorialEvaluation:
    """Freshly replay sources, rebuild the grid, then derive exact analysis."""
    if type(source_batch) is not LedgerFactorialSourceBatch:
        raise TypeError("source_batch must be an exact LedgerFactorialSourceBatch")
    require_ledger_factorial_derivation_source_batch(source_batch)
    rederived_batch = _derive_source_batch(
        plan=plan,
        protocol=protocol,
        batch_receipt=source_batch.batch_receipt,
        source_artifacts=source_artifacts,
    )
    if source_batch.canonical_bytes != rederived_batch.canonical_bytes:
        _fail("supplied factorial source batch differs from fresh source replay")
    grid = _factorial_grid_from_source_batch(plan=plan, batch=rederived_batch)
    analysis = derive_ledger_factorial_evaluation(
        spec=plan.analysis_spec,
        grid=grid,
    )
    result = object.__new__(SourceBoundLedgerFactorialEvaluation)
    fields: tuple[tuple[str, object], ...] = (
        ("plan", plan),
        ("protocol", protocol),
        ("source_batch", source_batch),
        ("grid", grid),
        ("analysis", analysis),
        ("plan_sha256", plan.sha256),
        ("protocol_sha256", protocol.sha256),
        ("source_batch_sha256", source_batch.sha256),
        ("analysis_sha256", analysis.sha256),
        (
            "source_artifact_sha256s",
            tuple(item.source_artifact_sha256 for item in rederived_batch.executions),
        ),
        ("source_grid_rederived_from_verified_artifacts", True),
        ("fixed_grid_factorial_arithmetic_verified", True),
        ("registered_axes_separated", True),
        ("real_world_uncertainty_components_identified", False),
        ("source_artifacts_independently_verified", False),
        ("source_execution_identity_externally_attested", False),
        ("rooted_bundle_custody_verified", False),
        ("analysis_chronology_externally_verified", False),
        ("synthetic_only", True),
        ("real_world_causal_claim", False),
        ("credible_interval_claim", False),
        ("confidence_interval_claim", False),
        ("iid_mcse_reported", False),
        ("independent_reproduction", False),
        ("publication_eligible", False),
        ("abstention_required", not analysis.precision_audit.passed),
        ("claim_scope", LEDGER_FACTORIAL_SOURCE_BOUND_CLAIM_SCOPE),
        ("schema_version", LEDGER_FACTORIAL_SOURCE_BOUND_RESULT_SCHEMA_VERSION),
    )
    for field, value in fields:
        object.__setattr__(result, field, value)
    result._fresh_validate()
    source_manifest_sha256 = sha256_bytes(
        canonical_json_bytes(list(result.source_artifact_sha256s))
    )
    mint_ticket = object.__new__(_SourceBoundFactorialResultMintTicket)
    with _VERIFIED_SOURCE_BOUND_FACTORIAL_RESULTS_LOCK:
        _SOURCE_BOUND_FACTORIAL_RESULT_MINT_TICKETS[mint_ticket] = (
            _SourceBoundFactorialResultMintBinding(
                plan_sha256=result.plan_sha256,
                protocol_sha256=result.protocol_sha256,
                source_batch_sha256=result.source_batch_sha256,
                analysis_sha256=result.analysis_sha256,
                source_artifact_manifest_sha256=source_manifest_sha256,
            )
        )
    return _mint_source_bound_factorial_result(ticket=mint_ticket, result=result)


def verify_source_bound_ledger_factorial_evaluation_bytes(
    content: bytes,
    *,
    expected_sha256: str,
    plan: LedgerFactorialStudyPlan,
    protocol: StudyProtocol,
    source_batch: LedgerFactorialSourceBatch,
    source_artifacts: Sequence[ProtocolBoundIntegratedSyntheticRunArtifact],
) -> SourceBoundLedgerFactorialEvaluation:
    """Verify canonical bytes by replaying the entire typed source closure."""
    if type(content) is not bytes:
        raise TypeError("content must be exact bytes")
    expected = _sha256(expected_sha256, "expected_sha256")
    if not content or len(content) > 256 * 1024 * 1024:
        _fail("source-bound factorial result bytes are empty or exceed their limit")
    if sha256_bytes(content) != expected:
        _fail("source-bound factorial result differs from its external commitment")
    try:
        mapping = read_json_bytes_object(
            content,
            source="source-bound ledger factorial evaluation bytes",
        )
    except MetadataValidationError as error:
        raise LedgerFactorialStudyError(str(error)) from error
    if canonical_json_bytes(mapping) != content:
        _fail("source-bound factorial result must be canonical duplicate-free JSON")
    rederived = derive_source_bound_ledger_factorial_evaluation(
        plan=plan,
        protocol=protocol,
        source_batch=source_batch,
        source_artifacts=source_artifacts,
    )
    if content != rederived.canonical_bytes:
        _fail("source-bound factorial result differs from fresh source derivation")
    return rederived


__all__ = [
    "LEDGER_FACTORIAL_PLAN_DEFINITION_ID",
    "LEDGER_FACTORIAL_MAX_SOURCE_COUNT",
    "LEDGER_FACTORIAL_SOURCE_BATCH_SCHEMA_VERSION",
    "LEDGER_FACTORIAL_SOURCE_BOUND_CLAIM_SCOPE",
    "LEDGER_FACTORIAL_SOURCE_BOUND_RESULT_SCHEMA_VERSION",
    "LEDGER_FACTORIAL_SOURCE_CLAIM_SCOPE",
    "LEDGER_FACTORIAL_STUDY_PLAN_SCHEMA_VERSION",
    "FactorialScenarioDesign",
    "LedgerFactorialSlotExecution",
    "LedgerFactorialSlotPlan",
    "LedgerFactorialSourceBatchOrigin",
    "LedgerFactorialSourceBatch",
    "LedgerFactorialStudyError",
    "LedgerFactorialStudyPlan",
    "PreparedLedgerFactorialStudy",
    "SourceBoundLedgerFactorialEvaluation",
    "authorize_ledger_factorial_study",
    "build_ledger_factorial_study_plan",
    "derive_source_bound_ledger_factorial_evaluation",
    "ledger_factorial_study_implementation_sha256",
    "require_executed_ledger_factorial_source_batch",
    "require_executed_source_bound_ledger_factorial_evaluation",
    "require_ledger_factorial_derivation_source_batch",
    "require_source_bound_ledger_factorial_evaluation",
    "require_verified_ledger_factorial_source_batch",
    "run_prepared_ledger_factorial_study",
    "verify_ledger_factorial_source_batch_bytes",
    "verify_source_bound_ledger_factorial_evaluation_bytes",
]
