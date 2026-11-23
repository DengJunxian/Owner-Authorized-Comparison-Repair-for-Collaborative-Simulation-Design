"""Same-environment construction contract for factorial ledger study plans.

The contract records every public argument accepted by
``build_ledger_factorial_study_plan`` and the typed-source commitments needed
to reconstruct exactly one frozen synthetic plan.  It does not execute the
grid, provide external custody, establish preregistration, or support any
real-world scientific claim.

Reconstruction deliberately imposes a much smaller source-count bound than
the pure arithmetic layer.  The bound is checked before the pair factory is
called so untrusted construction bytes cannot trigger an unexpectedly large
``M x R`` source-grid allocation.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal, NoReturn, Self, cast

from policytwin.calibration import CalibrationRun
from policytwin.counterfactual.integrated_market import (
    IntegratedSyntheticMarketConfig,
)
from policytwin.evaluation.ledger_bundle import LedgerOutputConstructionSpec
from policytwin.evaluation.ledger_derived import ExactRational
from policytwin.evaluation.ledger_factorial_study import (
    LEDGER_FACTORIAL_MAX_SOURCE_COUNT,
    LedgerFactorialStudyError,
    LedgerFactorialStudyPlan,
    build_ledger_factorial_study_plan,
)
from policytwin.experiments.integrated_synthetic import (
    integrated_synthetic_forward_market_config,
)
from policytwin.provenance import (
    MetadataValidationError,
    canonical_json_bytes,
    read_json_bytes_object,
    sha256_bytes,
)

LEDGER_FACTORIAL_PLAN_CONSTRUCTION_SCHEMA_VERSION = (
    "policytwin.ledger-factorial-plan-construction.v1"
)
LEDGER_FACTORIAL_PLAN_CONFIG_FACTORY_ID: Literal[
    "integrated_synthetic_forward_market_config.v1"
] = "integrated_synthetic_forward_market_config.v1"
LEDGER_FACTORIAL_PLAN_CONSTRUCTION_CLAIM_SCOPE: Literal[
    "synthetic_factorial_plan_reconstruction_mechanics_only"
] = "synthetic_factorial_plan_reconstruction_mechanics_only"
LEDGER_FACTORIAL_PLAN_CONSTRUCTION_MAX_SOURCE_COUNT = LEDGER_FACTORIAL_MAX_SOURCE_COUNT

_MAX_CONSTRUCTION_SPEC_BYTES = 1024 * 1024
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _fail(message: str) -> NoReturn:
    raise LedgerFactorialStudyError(message)


def _strict_fields(
    value: Mapping[str, object],
    expected: frozenset[str],
    label: str,
) -> None:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        _fail(f"{label} must be an object with string keys")
    actual = frozenset(value)
    if actual != expected:
        _fail(
            f"{label} fields differ from the closed schema; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{label} must be non-empty text")
    return value


def _identifier(value: object, label: str) -> str:
    result = _text(value, label)
    if _IDENTIFIER.fullmatch(result) is None:
        _fail(f"{label} must be a stable identifier")
    return result


def _sha256(value: object, label: str) -> str:
    result = _text(value, label)
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        _fail(f"{label} must be a lowercase SHA-256")
    return result


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        _fail(f"{label} must be boolean")
    return value


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _fail(f"{label} must be a positive integer")
    return value


def _at_least_two(value: object, label: str) -> int:
    result = _positive_int(value, label)
    if result < 2:
        _fail(f"{label} must be at least two")
    return result


def _canonical_utc(value: object, label: str) -> datetime:
    if (
        type(value) is not datetime
        or value.tzinfo is None
        or value.utcoffset() != timedelta(0)
    ):
        _fail(f"{label} must use canonical UTC")
    return value


def _datetime_from_text(value: object, label: str) -> datetime:
    text = _text(value, label)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as error:
        raise LedgerFactorialStudyError(f"{label} is invalid ISO-8601") from error
    return _canonical_utc(parsed, label)


def _calibration_event_keys_sha256(run: CalibrationRun) -> str:
    return sha256_bytes(canonical_json_bytes(list(run.event_keys)))


def _source_count(
    *,
    posterior_draw_count: object,
    scenario_replicate_count: object,
    expected_source_count: object,
) -> int:
    posterior_count = _at_least_two(posterior_draw_count, "posterior_draw_count")
    scenario_count = _at_least_two(scenario_replicate_count, "scenario_replicate_count")
    expected = _positive_int(expected_source_count, "expected_source_count")
    product = posterior_count * scenario_count
    if expected != product:
        _fail(
            "expected_source_count must equal posterior_draw_count times "
            "scenario_replicate_count"
        )
    if product > LEDGER_FACTORIAL_PLAN_CONSTRUCTION_MAX_SOURCE_COUNT:
        _fail("factorial plan construction source count exceeds its fixed bound")
    return product


def _exact_rational_from_mapping(value: object) -> ExactRational:
    if not isinstance(value, Mapping):
        _fail("maximum_bootstrap_mean_variance must be an exact-rational object")
    _strict_fields(
        value,
        frozenset({"denominator", "numerator"}),
        "maximum_bootstrap_mean_variance",
    )
    numerator = value["numerator"]
    denominator = value["denominator"]
    if isinstance(numerator, bool) or not isinstance(numerator, int):
        _fail("maximum_bootstrap_mean_variance numerator must be an integer")
    if isinstance(denominator, bool) or not isinstance(denominator, int):
        _fail("maximum_bootstrap_mean_variance denominator must be an integer")
    try:
        return ExactRational(numerator=numerator, denominator=denominator)
    except (TypeError, ValueError, RuntimeError) as error:
        raise LedgerFactorialStudyError(
            "maximum_bootstrap_mean_variance is not a canonical exact rational"
        ) from error


@dataclass(frozen=True, slots=True)
class LedgerFactorialPlanConstructionSpec:
    """Closed recipe for one exact synthetic factorial study plan."""

    plan_id: str
    frozen_at: datetime
    event_key_seed: str
    posterior_draw_count: int
    scenario_replicate_count: int
    expected_source_count: int
    outputs: tuple[LedgerOutputConstructionSpec, ...]
    bootstrap_replicates: int
    bootstrap_seed: str
    maximum_bootstrap_mean_variance: ExactRational
    outcomes_available_at: datetime
    candidate_allow_buy: bool
    candidate_allow_sell: bool
    candidate_is_noop: bool
    step_count: int
    include_all_groups: bool
    config_sha256: str
    calibration_config_sha256: str
    calibration_report_sha256: str
    calibration_posterior_sha256: str
    calibration_event_keys_sha256: str
    expected_analysis_spec_sha256: str
    expected_slot_manifest_sha256: str
    expected_output_manifest_sha256: str
    expected_plan_sha256: str
    config_factory_id: Literal["integrated_synthetic_forward_market_config.v1"] = (
        LEDGER_FACTORIAL_PLAN_CONFIG_FACTORY_ID
    )
    claim_scope: Literal["synthetic_factorial_plan_reconstruction_mechanics_only"] = (
        LEDGER_FACTORIAL_PLAN_CONSTRUCTION_CLAIM_SCOPE
    )
    synthetic_only: Literal[True] = True
    policy_post_data_used: Literal[False] = False
    source_grid_executed: Literal[False] = False
    external_custody_verified: Literal[False] = False
    externally_registered: Literal[False] = False
    analysis_chronology_externally_verified: Literal[False] = False
    real_world_causal_claim: Literal[False] = False
    credible_interval_claim: Literal[False] = False
    confidence_interval_claim: Literal[False] = False
    iid_mcse_reported: Literal[False] = False
    independent_reproduction: Literal[False] = False
    publication_eligible: Literal[False] = False
    same_environment_reconstruction_only: Literal[True] = True
    schema_version: str = LEDGER_FACTORIAL_PLAN_CONSTRUCTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _identifier(self.plan_id, "plan_id")
        _canonical_utc(self.frozen_at, "frozen_at")
        available = _canonical_utc(self.outcomes_available_at, "outcomes_available_at")
        if self.frozen_at >= available:
            _fail("factorial construction must be frozen before target outcomes")
        _text(self.event_key_seed, "event_key_seed")
        _source_count(
            posterior_draw_count=self.posterior_draw_count,
            scenario_replicate_count=self.scenario_replicate_count,
            expected_source_count=self.expected_source_count,
        )
        if (
            not isinstance(self.outputs, tuple)
            or not self.outputs
            or not all(
                type(item) is LedgerOutputConstructionSpec for item in self.outputs
            )
        ):
            _fail("factorial construction outputs must be a non-empty typed tuple")
        _at_least_two(self.bootstrap_replicates, "bootstrap_replicates")
        _identifier(self.bootstrap_seed, "bootstrap_seed")
        if type(self.maximum_bootstrap_mean_variance) is not ExactRational:
            _fail("maximum_bootstrap_mean_variance must use ExactRational")
        if self.maximum_bootstrap_mean_variance.numerator < 0:
            _fail("maximum_bootstrap_mean_variance cannot be negative")
        for field in (
            "candidate_allow_buy",
            "candidate_allow_sell",
            "candidate_is_noop",
            "include_all_groups",
        ):
            _boolean(getattr(self, field), field)
        if self.candidate_is_noop and not (
            self.candidate_allow_buy and self.candidate_allow_sell
        ):
            _fail("no-op factorial construction requires both allow switches true")
        _positive_int(self.step_count, "step_count")
        for field in (
            "config_sha256",
            "calibration_config_sha256",
            "calibration_report_sha256",
            "calibration_posterior_sha256",
            "calibration_event_keys_sha256",
            "expected_analysis_spec_sha256",
            "expected_slot_manifest_sha256",
            "expected_output_manifest_sha256",
            "expected_plan_sha256",
        ):
            _sha256(getattr(self, field), field)
        if (
            self.config_factory_id != LEDGER_FACTORIAL_PLAN_CONFIG_FACTORY_ID
            or self.claim_scope != LEDGER_FACTORIAL_PLAN_CONSTRUCTION_CLAIM_SCOPE
            or self.synthetic_only is not True
            or self.same_environment_reconstruction_only is not True
            or any(
                value is not False
                for value in (
                    self.policy_post_data_used,
                    self.source_grid_executed,
                    self.external_custody_verified,
                    self.externally_registered,
                    self.analysis_chronology_externally_verified,
                    self.real_world_causal_claim,
                    self.credible_interval_claim,
                    self.confidence_interval_claim,
                    self.iid_mcse_reported,
                    self.independent_reproduction,
                    self.publication_eligible,
                )
            )
            or self.schema_version != LEDGER_FACTORIAL_PLAN_CONSTRUCTION_SCHEMA_VERSION
        ):
            _fail("factorial construction schema or scientific claim boundary changed")

    @classmethod
    def from_plan(
        cls,
        plan: LedgerFactorialStudyPlan,
    ) -> Self:
        """Extract every builder input and prove exact plan reconstruction."""
        if type(plan) is not LedgerFactorialStudyPlan:
            raise TypeError("plan must be an exact LedgerFactorialStudyPlan")
        # Bound untrusted typed values before the plan's fresh validation
        # rederives its complete M x R pair grid.
        _source_count(
            posterior_draw_count=plan.posterior_draw_count,
            scenario_replicate_count=plan.scenario_replicate_count,
            expected_source_count=len(plan.slots),
        )
        plan._fresh_validate()
        result = cls(
            plan_id=plan.plan_id,
            frozen_at=plan.frozen_at,
            event_key_seed=plan.event_key_seed,
            posterior_draw_count=plan.posterior_draw_count,
            scenario_replicate_count=plan.scenario_replicate_count,
            expected_source_count=len(plan.slots),
            outputs=tuple(
                LedgerOutputConstructionSpec.from_output(item) for item in plan.outputs
            ),
            bootstrap_replicates=plan.analysis_spec.bootstrap_replicates,
            bootstrap_seed=plan.analysis_spec.bootstrap_seed,
            maximum_bootstrap_mean_variance=(
                plan.analysis_spec.precision_target.maximum_bootstrap_mean_variance
            ),
            outcomes_available_at=plan.analysis_spec.outcomes_available_at,
            candidate_allow_buy=plan.candidate_allow_buy,
            candidate_allow_sell=plan.candidate_allow_sell,
            candidate_is_noop=plan.candidate_is_noop,
            step_count=plan.step_count,
            include_all_groups=plan.include_all_groups,
            config_sha256=plan.config.sha256,
            calibration_config_sha256=plan.calibration_run.config.sha256,
            calibration_report_sha256=plan.calibration_run.report.sha256,
            calibration_posterior_sha256=plan.calibration_run.posterior.sha256,
            calibration_event_keys_sha256=_calibration_event_keys_sha256(
                plan.calibration_run
            ),
            expected_analysis_spec_sha256=plan.analysis_spec.sha256,
            expected_slot_manifest_sha256=plan.slot_manifest_sha256,
            expected_output_manifest_sha256=plan.output_manifest_sha256,
            expected_plan_sha256=plan.sha256,
        )
        rebuilt = result.rebuild(
            calibration_run=plan.calibration_run,
            config=plan.config,
        )
        if rebuilt.canonical_bytes != plan.canonical_bytes:
            _fail("factorial construction recipe does not reproduce the supplied plan")
        return result

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
        *,
        calibration_run: CalibrationRun,
        config: IntegratedSyntheticMarketConfig,
    ) -> Self:
        expected = frozenset(
            {
                "analysis_chronology_externally_verified",
                "bootstrap_replicates",
                "bootstrap_seed",
                "calibration_config_sha256",
                "calibration_event_keys_sha256",
                "calibration_posterior_sha256",
                "calibration_report_sha256",
                "candidate_allow_buy",
                "candidate_allow_sell",
                "candidate_is_noop",
                "claim_scope",
                "confidence_interval_claim",
                "config_factory_id",
                "config_sha256",
                "credible_interval_claim",
                "event_key_seed",
                "expected_analysis_spec_sha256",
                "expected_output_manifest_sha256",
                "expected_plan_sha256",
                "expected_slot_manifest_sha256",
                "expected_source_count",
                "external_custody_verified",
                "externally_registered",
                "frozen_at",
                "iid_mcse_reported",
                "include_all_groups",
                "independent_reproduction",
                "maximum_bootstrap_mean_variance",
                "outcomes_available_at",
                "outputs",
                "plan_id",
                "policy_post_data_used",
                "posterior_draw_count",
                "publication_eligible",
                "real_world_causal_claim",
                "same_environment_reconstruction_only",
                "scenario_replicate_count",
                "schema_version",
                "source_grid_executed",
                "step_count",
                "synthetic_only",
            }
        )
        _strict_fields(value, expected, "factorial plan construction")
        raw_outputs = value["outputs"]
        if not isinstance(raw_outputs, list):
            _fail("factorial construction outputs must be a list")
        outputs: list[LedgerOutputConstructionSpec] = []
        for raw_output in raw_outputs:
            if not isinstance(raw_output, Mapping):
                _fail("factorial construction output must be an object")
            outputs.append(LedgerOutputConstructionSpec.from_mapping(raw_output))
        result = cls(
            plan_id=_identifier(value["plan_id"], "plan_id"),
            frozen_at=_datetime_from_text(value["frozen_at"], "frozen_at"),
            event_key_seed=_text(value["event_key_seed"], "event_key_seed"),
            posterior_draw_count=_at_least_two(
                value["posterior_draw_count"], "posterior_draw_count"
            ),
            scenario_replicate_count=_at_least_two(
                value["scenario_replicate_count"], "scenario_replicate_count"
            ),
            expected_source_count=_positive_int(
                value["expected_source_count"], "expected_source_count"
            ),
            outputs=tuple(outputs),
            bootstrap_replicates=_at_least_two(
                value["bootstrap_replicates"], "bootstrap_replicates"
            ),
            bootstrap_seed=_identifier(value["bootstrap_seed"], "bootstrap_seed"),
            maximum_bootstrap_mean_variance=_exact_rational_from_mapping(
                value["maximum_bootstrap_mean_variance"]
            ),
            outcomes_available_at=_datetime_from_text(
                value["outcomes_available_at"], "outcomes_available_at"
            ),
            candidate_allow_buy=_boolean(
                value["candidate_allow_buy"], "candidate_allow_buy"
            ),
            candidate_allow_sell=_boolean(
                value["candidate_allow_sell"], "candidate_allow_sell"
            ),
            candidate_is_noop=_boolean(value["candidate_is_noop"], "candidate_is_noop"),
            step_count=_positive_int(value["step_count"], "step_count"),
            include_all_groups=_boolean(
                value["include_all_groups"], "include_all_groups"
            ),
            config_sha256=_sha256(value["config_sha256"], "config_sha256"),
            calibration_config_sha256=_sha256(
                value["calibration_config_sha256"], "calibration_config_sha256"
            ),
            calibration_report_sha256=_sha256(
                value["calibration_report_sha256"], "calibration_report_sha256"
            ),
            calibration_posterior_sha256=_sha256(
                value["calibration_posterior_sha256"],
                "calibration_posterior_sha256",
            ),
            calibration_event_keys_sha256=_sha256(
                value["calibration_event_keys_sha256"],
                "calibration_event_keys_sha256",
            ),
            expected_analysis_spec_sha256=_sha256(
                value["expected_analysis_spec_sha256"],
                "expected_analysis_spec_sha256",
            ),
            expected_slot_manifest_sha256=_sha256(
                value["expected_slot_manifest_sha256"],
                "expected_slot_manifest_sha256",
            ),
            expected_output_manifest_sha256=_sha256(
                value["expected_output_manifest_sha256"],
                "expected_output_manifest_sha256",
            ),
            expected_plan_sha256=_sha256(
                value["expected_plan_sha256"], "expected_plan_sha256"
            ),
            config_factory_id=cast(
                Literal["integrated_synthetic_forward_market_config.v1"],
                _text(value["config_factory_id"], "config_factory_id"),
            ),
            claim_scope=cast(
                Literal["synthetic_factorial_plan_reconstruction_mechanics_only"],
                _text(value["claim_scope"], "claim_scope"),
            ),
            synthetic_only=cast(
                Literal[True], _boolean(value["synthetic_only"], "synthetic_only")
            ),
            policy_post_data_used=cast(
                Literal[False],
                _boolean(value["policy_post_data_used"], "policy_post_data_used"),
            ),
            source_grid_executed=cast(
                Literal[False],
                _boolean(value["source_grid_executed"], "source_grid_executed"),
            ),
            external_custody_verified=cast(
                Literal[False],
                _boolean(
                    value["external_custody_verified"], "external_custody_verified"
                ),
            ),
            externally_registered=cast(
                Literal[False],
                _boolean(value["externally_registered"], "externally_registered"),
            ),
            analysis_chronology_externally_verified=cast(
                Literal[False],
                _boolean(
                    value["analysis_chronology_externally_verified"],
                    "analysis_chronology_externally_verified",
                ),
            ),
            real_world_causal_claim=cast(
                Literal[False],
                _boolean(value["real_world_causal_claim"], "real_world_causal_claim"),
            ),
            credible_interval_claim=cast(
                Literal[False],
                _boolean(value["credible_interval_claim"], "credible_interval_claim"),
            ),
            confidence_interval_claim=cast(
                Literal[False],
                _boolean(
                    value["confidence_interval_claim"], "confidence_interval_claim"
                ),
            ),
            iid_mcse_reported=cast(
                Literal[False],
                _boolean(value["iid_mcse_reported"], "iid_mcse_reported"),
            ),
            independent_reproduction=cast(
                Literal[False],
                _boolean(value["independent_reproduction"], "independent_reproduction"),
            ),
            publication_eligible=cast(
                Literal[False],
                _boolean(value["publication_eligible"], "publication_eligible"),
            ),
            same_environment_reconstruction_only=cast(
                Literal[True],
                _boolean(
                    value["same_environment_reconstruction_only"],
                    "same_environment_reconstruction_only",
                ),
            ),
            schema_version=_text(value["schema_version"], "schema_version"),
        )
        if canonical_json_bytes(result.to_mapping()) != canonical_json_bytes(
            dict(value)
        ):
            _fail("factorial construction mapping is not in canonical semantic form")
        result.rebuild(calibration_run=calibration_run, config=config)
        return result

    def rebuild(
        self,
        *,
        calibration_run: CalibrationRun,
        config: IntegratedSyntheticMarketConfig,
    ) -> LedgerFactorialStudyPlan:
        """Reconstruct the plan after bounded typed-source verification."""
        if type(calibration_run) is not CalibrationRun:
            raise TypeError("calibration_run must be an exact CalibrationRun")
        if type(config) is not IntegratedSyntheticMarketConfig:
            raise TypeError("config must be an IntegratedSyntheticMarketConfig")

        # This check must precede output building and the M x R pair factory.
        _source_count(
            posterior_draw_count=self.posterior_draw_count,
            scenario_replicate_count=self.scenario_replicate_count,
            expected_source_count=self.expected_source_count,
        )
        if len(calibration_run.posterior.draws) != self.posterior_draw_count:
            _fail("factorial construction posterior draw count changed")
        if integrated_synthetic_forward_market_config() != config:
            _fail("factorial construction config is not the allowlisted frozen factory")
        if config.sha256 != self.config_sha256:
            _fail("factorial construction config hash does not match the typed config")
        if (
            calibration_run.config.sha256 != self.calibration_config_sha256
            or calibration_run.report.sha256 != self.calibration_report_sha256
            or calibration_run.posterior.sha256 != self.calibration_posterior_sha256
            or _calibration_event_keys_sha256(calibration_run)
            != self.calibration_event_keys_sha256
        ):
            _fail("factorial construction calibration source commitment changed")

        outputs = tuple(item.build(config) for item in self.outputs)
        rebuilt = build_ledger_factorial_study_plan(
            plan_id=self.plan_id,
            frozen_at=self.frozen_at,
            calibration_run=calibration_run,
            config=config,
            event_key_seed=self.event_key_seed,
            scenario_replicate_count=self.scenario_replicate_count,
            outputs=outputs,
            bootstrap_replicates=self.bootstrap_replicates,
            bootstrap_seed=self.bootstrap_seed,
            maximum_bootstrap_mean_variance=self.maximum_bootstrap_mean_variance,
            outcomes_available_at=self.outcomes_available_at,
            candidate_allow_buy=self.candidate_allow_buy,
            candidate_allow_sell=self.candidate_allow_sell,
            candidate_is_noop=self.candidate_is_noop,
            step_count=self.step_count,
            include_all_groups=self.include_all_groups,
        )
        if (
            rebuilt.analysis_spec.sha256 != self.expected_analysis_spec_sha256
            or rebuilt.slot_manifest_sha256 != self.expected_slot_manifest_sha256
            or rebuilt.output_manifest_sha256 != self.expected_output_manifest_sha256
            or rebuilt.sha256 != self.expected_plan_sha256
        ):
            _fail("reconstructed factorial plan differs from its commitments")
        return rebuilt

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_mapping())

    @property
    def sha256(self) -> str:
        return sha256_bytes(self.canonical_bytes)

    def to_mapping(self) -> dict[str, object]:
        return {
            "analysis_chronology_externally_verified": (
                self.analysis_chronology_externally_verified
            ),
            "bootstrap_replicates": self.bootstrap_replicates,
            "bootstrap_seed": self.bootstrap_seed,
            "calibration_config_sha256": self.calibration_config_sha256,
            "calibration_event_keys_sha256": self.calibration_event_keys_sha256,
            "calibration_posterior_sha256": self.calibration_posterior_sha256,
            "calibration_report_sha256": self.calibration_report_sha256,
            "candidate_allow_buy": self.candidate_allow_buy,
            "candidate_allow_sell": self.candidate_allow_sell,
            "candidate_is_noop": self.candidate_is_noop,
            "claim_scope": self.claim_scope,
            "confidence_interval_claim": self.confidence_interval_claim,
            "config_factory_id": self.config_factory_id,
            "config_sha256": self.config_sha256,
            "credible_interval_claim": self.credible_interval_claim,
            "event_key_seed": self.event_key_seed,
            "expected_analysis_spec_sha256": self.expected_analysis_spec_sha256,
            "expected_output_manifest_sha256": self.expected_output_manifest_sha256,
            "expected_plan_sha256": self.expected_plan_sha256,
            "expected_slot_manifest_sha256": self.expected_slot_manifest_sha256,
            "expected_source_count": self.expected_source_count,
            "external_custody_verified": self.external_custody_verified,
            "externally_registered": self.externally_registered,
            "frozen_at": self.frozen_at.isoformat(),
            "iid_mcse_reported": self.iid_mcse_reported,
            "include_all_groups": self.include_all_groups,
            "independent_reproduction": self.independent_reproduction,
            "maximum_bootstrap_mean_variance": (
                self.maximum_bootstrap_mean_variance.to_mapping()
            ),
            "outcomes_available_at": self.outcomes_available_at.isoformat(),
            "outputs": [item.to_mapping() for item in self.outputs],
            "plan_id": self.plan_id,
            "policy_post_data_used": self.policy_post_data_used,
            "posterior_draw_count": self.posterior_draw_count,
            "publication_eligible": self.publication_eligible,
            "real_world_causal_claim": self.real_world_causal_claim,
            "same_environment_reconstruction_only": (
                self.same_environment_reconstruction_only
            ),
            "scenario_replicate_count": self.scenario_replicate_count,
            "schema_version": self.schema_version,
            "source_grid_executed": self.source_grid_executed,
            "step_count": self.step_count,
            "synthetic_only": self.synthetic_only,
        }


def verify_ledger_factorial_plan_construction_bytes(
    content: bytes,
    *,
    expected_sha256: str,
    calibration_run: CalibrationRun,
    config: IntegratedSyntheticMarketConfig,
) -> LedgerFactorialPlanConstructionSpec:
    """Verify canonical bytes and reconstruct their exact typed study plan.

    ``expected_sha256`` is only a caller-supplied value commitment.  It does
    not prove external publication, custody, chronology, or preregistration.
    """
    if not isinstance(content, bytes):
        raise TypeError("factorial construction content must be bytes")
    expected = _sha256(expected_sha256, "expected_sha256")
    if not content or len(content) > _MAX_CONSTRUCTION_SPEC_BYTES:
        _fail("factorial construction content is empty or exceeds its byte limit")
    if sha256_bytes(content) != expected:
        _fail("factorial construction bytes differ from the supplied root commitment")
    try:
        mapping = read_json_bytes_object(
            content,
            source="factorial plan construction bytes",
        )
    except MetadataValidationError as error:
        raise LedgerFactorialStudyError(str(error)) from error
    if canonical_json_bytes(mapping) != content:
        _fail("factorial construction bytes must use canonical duplicate-free JSON")
    return LedgerFactorialPlanConstructionSpec.from_mapping(
        mapping,
        calibration_run=calibration_run,
        config=config,
    )


__all__ = [
    "LEDGER_FACTORIAL_PLAN_CONFIG_FACTORY_ID",
    "LEDGER_FACTORIAL_PLAN_CONSTRUCTION_CLAIM_SCOPE",
    "LEDGER_FACTORIAL_PLAN_CONSTRUCTION_MAX_SOURCE_COUNT",
    "LEDGER_FACTORIAL_PLAN_CONSTRUCTION_SCHEMA_VERSION",
    "LedgerFactorialPlanConstructionSpec",
    "verify_ledger_factorial_plan_construction_bytes",
]
