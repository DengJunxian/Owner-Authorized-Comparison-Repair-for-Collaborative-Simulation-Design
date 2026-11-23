"""Same-environment construction contract for synthetic ledger study plans.

The serialized study plan intentionally omits large typed runtime inputs.  This
module records the small, closed set of builder arguments needed to reconstruct
that plan from a separately verified calibration run and the one allowlisted
synthetic market configuration.  The plan hash intentionally binds machine,
Git state, lock, and live implementation material, so this contract is not
portable evidence, external registration, or execution attestation.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Literal, NoReturn, Self, cast

from policytwin.calibration import CalibrationRun
from policytwin.counterfactual.integrated_market import (
    IntegratedSyntheticMarketConfig,
)
from policytwin.evaluation.ledger_derived import DifferenceScope
from policytwin.evaluation.ledger_posterior import (
    LedgerBenefitDirection,
    LedgerPosteriorError,
    LedgerPosteriorOutputSpec,
    LedgerPosteriorStudyPlan,
    UndefinedMetricPolicy,
    build_ledger_posterior_study_plan,
    ledger_output_spec,
)
from policytwin.experiments.integrated_synthetic import (
    integrated_synthetic_forward_market_config,
)
from policytwin.protocol import OutputRole
from policytwin.provenance import (
    MetadataValidationError,
    canonical_json_bytes,
    read_json_bytes_object,
    sha256_bytes,
)

LEDGER_PLAN_CONSTRUCTION_SCHEMA_VERSION = "policytwin.ledger-plan-construction.v1"
LEDGER_PLAN_CONFIG_FACTORY_ID: Literal[
    "integrated_synthetic_forward_market_config.v1"
] = "integrated_synthetic_forward_market_config.v1"
LEDGER_PLAN_CONSTRUCTION_CLAIM_SCOPE: Literal[
    "synthetic_plan_reconstruction_mechanics_only"
] = "synthetic_plan_reconstruction_mechanics_only"

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_CONSTRUCTION_SPEC_BYTES = 1024 * 1024


def _fail(message: str) -> NoReturn:
    raise LedgerPosteriorError(message)


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
    if _SHA256.fullmatch(result) is None:
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


def _enum_value[EnumT: Enum](
    enum_type: type[EnumT], value: object, label: str
) -> EnumT:
    text = _text(value, label)
    try:
        return enum_type(text)
    except (TypeError, ValueError) as error:
        raise LedgerPosteriorError(f"{label} is unsupported") from error


def _calibration_event_keys_sha256(run: CalibrationRun) -> str:
    return sha256_bytes(canonical_json_bytes(list(run.event_keys)))


@dataclass(frozen=True, slots=True)
class LedgerOutputConstructionSpec:
    """Caller-independent builder arguments for one frozen output cell."""

    output_id: str
    horizon_id: str
    scope: DifferenceScope
    metric_id: str
    role: OutputRole
    benefit_direction: LedgerBenefitDirection
    undefined_policy: UndefinedMetricPolicy
    group_id: str | None = None
    interval_levels: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.output_id, "output_id")
        _identifier(self.horizon_id, "horizon_id")
        _identifier(self.metric_id, "metric_id")
        if not isinstance(self.scope, DifferenceScope):
            _fail("output construction scope has the wrong type")
        if self.scope is DifferenceScope.GROUP:
            _identifier(self.group_id, "group_id")
        elif self.group_id is not None:
            _fail("market output construction cannot carry a group ID")
        if not isinstance(self.role, OutputRole):
            _fail("output construction role has the wrong type")
        if not isinstance(self.benefit_direction, LedgerBenefitDirection):
            _fail("output construction benefit direction has the wrong type")
        if not isinstance(self.undefined_policy, UndefinedMetricPolicy):
            _fail("output construction undefined policy has the wrong type")
        if not isinstance(self.interval_levels, tuple):
            _fail("output construction interval levels must be immutable")
        # The authoritative output constructor performs the full level checks.
        for level in self.interval_levels:
            if isinstance(level, bool) or not isinstance(level, int | float):
                _fail("output construction interval levels must be numeric")

    @classmethod
    def from_output(cls, output: LedgerPosteriorOutputSpec) -> Self:
        if type(output) is not LedgerPosteriorOutputSpec:
            raise TypeError("output must be a LedgerPosteriorOutputSpec")
        return cls(
            output_id=output.output_id,
            horizon_id=output.horizon_id,
            scope=output.scope,
            metric_id=output.metric_id,
            role=output.role,
            benefit_direction=output.benefit_direction,
            undefined_policy=output.undefined_policy,
            group_id=output.group_id,
            interval_levels=output.interval_levels,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        expected = frozenset(
            {
                "benefit_direction",
                "group_id",
                "horizon_id",
                "interval_levels",
                "metric_id",
                "output_id",
                "role",
                "scope",
                "undefined_policy",
            }
        )
        _strict_fields(value, expected, "ledger output construction")
        raw_levels = value["interval_levels"]
        if not isinstance(raw_levels, list):
            _fail("output construction interval_levels must be a list")
        levels: list[float] = []
        for raw_level in raw_levels:
            if isinstance(raw_level, bool) or not isinstance(raw_level, int | float):
                _fail("output construction interval level must be numeric")
            levels.append(float(raw_level))
        group_id = value["group_id"]
        if group_id is not None and not isinstance(group_id, str):
            _fail("output construction group_id must be text or null")
        return cls(
            output_id=_identifier(value["output_id"], "output_id"),
            horizon_id=_identifier(value["horizon_id"], "horizon_id"),
            scope=_enum_value(DifferenceScope, value["scope"], "scope"),
            metric_id=_identifier(value["metric_id"], "metric_id"),
            role=_enum_value(OutputRole, value["role"], "role"),
            benefit_direction=_enum_value(
                LedgerBenefitDirection,
                value["benefit_direction"],
                "benefit_direction",
            ),
            undefined_policy=_enum_value(
                UndefinedMetricPolicy,
                value["undefined_policy"],
                "undefined_policy",
            ),
            group_id=group_id,
            interval_levels=tuple(levels),
        )

    def build(
        self,
        config: IntegratedSyntheticMarketConfig,
    ) -> LedgerPosteriorOutputSpec:
        return ledger_output_spec(
            output_id=self.output_id,
            horizon_id=self.horizon_id,
            scope=self.scope,
            metric_id=self.metric_id,
            config=config,
            role=self.role,
            benefit_direction=self.benefit_direction,
            undefined_policy=self.undefined_policy,
            group_id=self.group_id,
            interval_levels=self.interval_levels,
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "benefit_direction": self.benefit_direction.value,
            "group_id": self.group_id,
            "horizon_id": self.horizon_id,
            "interval_levels": list(self.interval_levels),
            "metric_id": self.metric_id,
            "output_id": self.output_id,
            "role": self.role.value,
            "scope": self.scope.value,
            "undefined_policy": self.undefined_policy.value,
        }


@dataclass(frozen=True, slots=True)
class LedgerPosteriorPlanConstructionSpec:
    """Closed recipe that reconstructs one exact typed synthetic study plan."""

    plan_id: str
    frozen_at: datetime
    event_key_seed: str
    outputs: tuple[LedgerOutputConstructionSpec, ...]
    candidate_allow_buy: bool
    candidate_allow_sell: bool
    candidate_is_noop: bool
    step_count: int
    include_all_groups: bool
    config_sha256: str
    calibration_report_sha256: str
    calibration_posterior_sha256: str
    calibration_event_keys_sha256: str
    expected_plan_sha256: str
    config_factory_id: Literal["integrated_synthetic_forward_market_config.v1"] = (
        LEDGER_PLAN_CONFIG_FACTORY_ID
    )
    claim_scope: Literal["synthetic_plan_reconstruction_mechanics_only"] = (
        LEDGER_PLAN_CONSTRUCTION_CLAIM_SCOPE
    )
    synthetic_only: Literal[True] = True
    externally_registered: Literal[False] = False
    same_environment_reconstruction_only: Literal[True] = True
    portable_reconstruction_proven: Literal[False] = False
    independent_reproduction: Literal[False] = False
    schema_version: str = LEDGER_PLAN_CONSTRUCTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _identifier(self.plan_id, "plan_id")
        if (
            type(self.frozen_at) is not datetime
            or self.frozen_at.tzinfo is None
            or self.frozen_at.utcoffset() != timedelta(0)
        ):
            _fail("plan construction frozen_at must use canonical UTC")
        if not isinstance(self.event_key_seed, str) or not self.event_key_seed:
            _fail("plan construction event_key_seed must be non-empty")
        if (
            not isinstance(self.outputs, tuple)
            or not self.outputs
            or not all(
                type(item) is LedgerOutputConstructionSpec for item in self.outputs
            )
        ):
            _fail("plan construction outputs must be a non-empty typed tuple")
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
            _fail("no-op plan construction requires both allow switches to be true")
        _positive_int(self.step_count, "step_count")
        for field in (
            "config_sha256",
            "calibration_report_sha256",
            "calibration_posterior_sha256",
            "calibration_event_keys_sha256",
            "expected_plan_sha256",
        ):
            _sha256(getattr(self, field), field)
        if (
            self.config_factory_id != LEDGER_PLAN_CONFIG_FACTORY_ID
            or self.claim_scope != LEDGER_PLAN_CONSTRUCTION_CLAIM_SCOPE
            or self.synthetic_only is not True
            or self.externally_registered is not False
            or self.same_environment_reconstruction_only is not True
            or self.portable_reconstruction_proven is not False
            or self.independent_reproduction is not False
            or self.schema_version != LEDGER_PLAN_CONSTRUCTION_SCHEMA_VERSION
        ):
            _fail("plan construction scope, factory, or schema changed")

    @classmethod
    def from_plan(
        cls,
        plan: LedgerPosteriorStudyPlan,
        *,
        include_all_groups: bool,
    ) -> Self:
        """Extract builder inputs and prove they reproduce the supplied plan."""
        if type(plan) is not LedgerPosteriorStudyPlan:
            raise TypeError("plan must be an exact LedgerPosteriorStudyPlan")
        plan._fresh_validate()
        first_entry = plan.draw_entries[0]
        key_mapping = json.loads(first_entry.evaluation_event_key_canonical_text)
        if not isinstance(key_mapping, dict):
            _fail("plan event key must decode to an object")
        event_key_seed = _text(key_mapping.get("seed"), "event_key_seed")
        first_pair = plan.pairs[0]
        candidate_config = dict(first_pair.candidate_policy.config)
        if set(candidate_config) != {"allow_buy", "allow_sell"}:
            _fail("plan candidate switches cannot be reconstructed")
        allow_buy = _boolean(candidate_config["allow_buy"], "candidate_allow_buy")
        allow_sell = _boolean(candidate_config["allow_sell"], "candidate_allow_sell")
        terminal_steps = {
            item.end_step
            for item in first_pair.horizons
            if item.horizon_id == "terminal"
        }
        if len(terminal_steps) != 1:
            _fail("plan terminal step cannot be reconstructed")
        result = cls(
            plan_id=plan.plan_id,
            frozen_at=plan.frozen_at,
            event_key_seed=event_key_seed,
            outputs=tuple(
                LedgerOutputConstructionSpec.from_output(item) for item in plan.outputs
            ),
            candidate_allow_buy=allow_buy,
            candidate_allow_sell=allow_sell,
            candidate_is_noop=first_pair.candidate_policy.is_noop,
            step_count=terminal_steps.pop(),
            include_all_groups=_boolean(include_all_groups, "include_all_groups"),
            config_sha256=plan.config.sha256,
            calibration_report_sha256=plan.calibration_run.report.sha256,
            calibration_posterior_sha256=plan.calibration_run.posterior.sha256,
            calibration_event_keys_sha256=_calibration_event_keys_sha256(
                plan.calibration_run
            ),
            expected_plan_sha256=plan.sha256,
        )
        rebuilt = result.rebuild(
            calibration_run=plan.calibration_run,
            config=plan.config,
        )
        if rebuilt.canonical_bytes != plan.canonical_bytes:
            _fail("plan construction recipe does not reproduce the supplied plan")
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
                "calibration_event_keys_sha256",
                "calibration_posterior_sha256",
                "calibration_report_sha256",
                "candidate_allow_buy",
                "candidate_allow_sell",
                "candidate_is_noop",
                "claim_scope",
                "config_factory_id",
                "config_sha256",
                "event_key_seed",
                "expected_plan_sha256",
                "externally_registered",
                "frozen_at",
                "include_all_groups",
                "independent_reproduction",
                "outputs",
                "plan_id",
                "portable_reconstruction_proven",
                "same_environment_reconstruction_only",
                "schema_version",
                "step_count",
                "synthetic_only",
            }
        )
        _strict_fields(value, expected, "ledger plan construction")
        raw_outputs = value["outputs"]
        if not isinstance(raw_outputs, list):
            _fail("plan construction outputs must be a list")
        outputs: list[LedgerOutputConstructionSpec] = []
        for raw_output in raw_outputs:
            if not isinstance(raw_output, Mapping):
                _fail("plan construction output must be an object")
            outputs.append(LedgerOutputConstructionSpec.from_mapping(raw_output))
        frozen_text = _text(value["frozen_at"], "frozen_at")
        try:
            frozen_at = datetime.fromisoformat(frozen_text)
        except ValueError as error:
            raise LedgerPosteriorError(
                "plan construction frozen_at is invalid ISO-8601"
            ) from error
        result = cls(
            plan_id=_identifier(value["plan_id"], "plan_id"),
            frozen_at=frozen_at,
            event_key_seed=_text(value["event_key_seed"], "event_key_seed"),
            outputs=tuple(outputs),
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
            calibration_report_sha256=_sha256(
                value["calibration_report_sha256"],
                "calibration_report_sha256",
            ),
            calibration_posterior_sha256=_sha256(
                value["calibration_posterior_sha256"],
                "calibration_posterior_sha256",
            ),
            calibration_event_keys_sha256=_sha256(
                value["calibration_event_keys_sha256"],
                "calibration_event_keys_sha256",
            ),
            expected_plan_sha256=_sha256(
                value["expected_plan_sha256"], "expected_plan_sha256"
            ),
            config_factory_id=cast(
                Literal["integrated_synthetic_forward_market_config.v1"],
                _text(value["config_factory_id"], "config_factory_id"),
            ),
            claim_scope=cast(
                Literal["synthetic_plan_reconstruction_mechanics_only"],
                _text(value["claim_scope"], "claim_scope"),
            ),
            synthetic_only=cast(
                Literal[True], _boolean(value["synthetic_only"], "synthetic_only")
            ),
            externally_registered=cast(
                Literal[False],
                _boolean(value["externally_registered"], "externally_registered"),
            ),
            same_environment_reconstruction_only=cast(
                Literal[True],
                _boolean(
                    value["same_environment_reconstruction_only"],
                    "same_environment_reconstruction_only",
                ),
            ),
            portable_reconstruction_proven=cast(
                Literal[False],
                _boolean(
                    value["portable_reconstruction_proven"],
                    "portable_reconstruction_proven",
                ),
            ),
            independent_reproduction=cast(
                Literal[False],
                _boolean(
                    value["independent_reproduction"],
                    "independent_reproduction",
                ),
            ),
            schema_version=_text(value["schema_version"], "schema_version"),
        )
        if canonical_json_bytes(result.to_mapping()) != canonical_json_bytes(
            dict(value)
        ):
            _fail("plan construction mapping is not in canonical semantic form")
        result.rebuild(calibration_run=calibration_run, config=config)
        return result

    def rebuild(
        self,
        *,
        calibration_run: CalibrationRun,
        config: IntegratedSyntheticMarketConfig,
    ) -> LedgerPosteriorStudyPlan:
        """Reconstruct the plan from typed sources and reject every mismatch."""
        if type(calibration_run) is not CalibrationRun:
            raise TypeError("calibration_run must be an exact CalibrationRun")
        if type(config) is not IntegratedSyntheticMarketConfig:
            raise TypeError("config must be an IntegratedSyntheticMarketConfig")
        if integrated_synthetic_forward_market_config() != config:
            _fail("plan construction config is not the allowlisted frozen factory")
        if config.sha256 != self.config_sha256:
            _fail("plan construction config hash does not match the typed config")
        if (
            calibration_run.report.sha256 != self.calibration_report_sha256
            or calibration_run.posterior.sha256 != self.calibration_posterior_sha256
            or _calibration_event_keys_sha256(calibration_run)
            != self.calibration_event_keys_sha256
        ):
            _fail("plan construction calibration source commitment changed")
        outputs = tuple(item.build(config) for item in self.outputs)
        rebuilt = build_ledger_posterior_study_plan(
            plan_id=self.plan_id,
            frozen_at=self.frozen_at,
            calibration_run=calibration_run,
            config=config,
            event_key_seed=self.event_key_seed,
            outputs=outputs,
            candidate_allow_buy=self.candidate_allow_buy,
            candidate_allow_sell=self.candidate_allow_sell,
            candidate_is_noop=self.candidate_is_noop,
            step_count=self.step_count,
            include_all_groups=self.include_all_groups,
        )
        if rebuilt.sha256 != self.expected_plan_sha256:
            _fail("reconstructed plan differs from its construction plan commitment")
        return rebuilt

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_mapping())

    @property
    def sha256(self) -> str:
        return sha256_bytes(self.canonical_bytes)

    def to_mapping(self) -> dict[str, object]:
        return {
            "calibration_event_keys_sha256": self.calibration_event_keys_sha256,
            "calibration_posterior_sha256": self.calibration_posterior_sha256,
            "calibration_report_sha256": self.calibration_report_sha256,
            "candidate_allow_buy": self.candidate_allow_buy,
            "candidate_allow_sell": self.candidate_allow_sell,
            "candidate_is_noop": self.candidate_is_noop,
            "claim_scope": self.claim_scope,
            "config_factory_id": self.config_factory_id,
            "config_sha256": self.config_sha256,
            "event_key_seed": self.event_key_seed,
            "expected_plan_sha256": self.expected_plan_sha256,
            "externally_registered": self.externally_registered,
            "frozen_at": self.frozen_at.isoformat(),
            "include_all_groups": self.include_all_groups,
            "independent_reproduction": self.independent_reproduction,
            "outputs": [item.to_mapping() for item in self.outputs],
            "plan_id": self.plan_id,
            "portable_reconstruction_proven": self.portable_reconstruction_proven,
            "same_environment_reconstruction_only": (
                self.same_environment_reconstruction_only
            ),
            "schema_version": self.schema_version,
            "step_count": self.step_count,
            "synthetic_only": self.synthetic_only,
        }


def verify_ledger_posterior_plan_construction_bytes(
    content: bytes,
    *,
    expected_sha256: str,
    calibration_run: CalibrationRun,
    config: IntegratedSyntheticMarketConfig,
) -> LedgerPosteriorPlanConstructionSpec:
    """Verify externally supplied bytes, then reconstruct their exact typed plan.

    The expected digest is authority relative to ``content`` only.  This
    function does not prove who held, published, or timestamped that digest and
    therefore cannot make ``externally_registered`` true.
    """
    if not isinstance(content, bytes):
        raise TypeError("plan construction content must be bytes")
    expected = _sha256(expected_sha256, "expected_sha256")
    if not content or len(content) > _MAX_CONSTRUCTION_SPEC_BYTES:
        _fail("plan construction content is empty or exceeds its byte limit")
    if sha256_bytes(content) != expected:
        _fail("plan construction bytes differ from the supplied root commitment")
    try:
        mapping = read_json_bytes_object(
            content,
            source="ledger plan construction bytes",
        )
    except MetadataValidationError as error:
        raise LedgerPosteriorError(str(error)) from error
    if canonical_json_bytes(mapping) != content:
        _fail("plan construction bytes must use canonical duplicate-free JSON")
    return LedgerPosteriorPlanConstructionSpec.from_mapping(
        mapping,
        calibration_run=calibration_run,
        config=config,
    )


__all__ = [
    "LEDGER_PLAN_CONFIG_FACTORY_ID",
    "LEDGER_PLAN_CONSTRUCTION_CLAIM_SCOPE",
    "LEDGER_PLAN_CONSTRUCTION_SCHEMA_VERSION",
    "LedgerOutputConstructionSpec",
    "LedgerPosteriorPlanConstructionSpec",
    "verify_ledger_posterior_plan_construction_bytes",
]
