"""Protocol-first aggregation of freshly generated synthetic ledger draws.

The boundary in this module is intentionally narrow.  It consumes every draw
from one locally replayed synthetic calibration posterior, freezes the complete
scenario and output manifest before any source run, and then derives exact
candidate-minus-baseline ledger distributions.  It does not establish external
preregistration, a real-data posterior, welfare, or a real-world causal effect.
"""

from __future__ import annotations

import json
import os
import re
import stat
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from fractions import Fraction
from math import isfinite
from pathlib import Path
from typing import Literal, NoReturn, Self, cast

import policytwin.calibration.integrated as calibration_integrated_module
import policytwin.calibration.workflow as calibration_workflow_module
import policytwin.evaluation.ledger_derived as ledger_derived_module
import policytwin.experiments.integrated_synthetic as integrated_synthetic_module
from policytwin.calibration import CalibrationRun
from policytwin.calibration.integrated import INTEGRATED_CALIBRATION_SIMULATOR_ID
from policytwin.counterfactual.integrated_market import (
    IntegratedSyntheticMarketConfig,
)
from policytwin.counterfactual.paired import PolicyWorldPair, WorldRole
from policytwin.counterfactual.randomness import EVENT_KEY_SCHEMA_VERSION, EventKey
from policytwin.evaluation.ledger_derived import (
    DifferenceScope,
    ExactMetric,
    ExactMetricState,
    ExactRational,
    LedgerDerivedEvaluation,
    LedgerMetricDefinition,
    LedgerMetricScope,
    PairedLedgerDifference,
    derive_protocol_bound_ledger_evidence,
    ledger_evidence_verifier_implementation_sha256,
    ledger_metric_registry,
    ledger_metric_registry_sha256,
)
from policytwin.experiments.integrated_synthetic import (
    ProtocolBoundIntegratedSyntheticRunArtifact,
    integrated_synthetic_pair,
    integrated_synthetic_pair_input_sha256,
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
    collect_environment_metadata,
    collect_git_metadata,
    read_json_bytes_object,
    sha256_bytes,
)

LEDGER_POSTERIOR_PLAN_SCHEMA_VERSION = "policytwin.ledger-posterior-plan.v2"
LEDGER_POSTERIOR_RESULT_SCHEMA_VERSION = "policytwin.ledger-posterior-result.v2"
LEDGER_POSTERIOR_PLAN_DEFINITION_ID = "ledger_posterior_analysis_plan"
LEDGER_POSTERIOR_CLAIM_SCOPE = (
    "synthetic_protocol_bound_ledger_aggregation_mechanics_only"
)
LEDGER_POSTERIOR_OUTPUT_ROLE_SCOPE: Literal[
    "ledger_plan_local_synthetic_diagnostic_not_protocol_output_role"
] = "ledger_plan_local_synthetic_diagnostic_not_protocol_output_role"
LEDGER_POSTERIOR_ESTIMAND_ID = "exact_candidate_minus_baseline_per_frozen_draw"
LEDGER_POSTERIOR_ESTIMAND_DEFINITION_SHA256 = sha256_bytes(
    canonical_json_bytes(
        {
            "aggregation": "retain_every_frozen_draw_without_imputation",
            "formula": "candidate exact metric minus baseline exact metric",
            "sampling_unit": "one_keyed_scenario_per_calibration_posterior_draw",
            "schema_version": 1,
            "undefined_semantics": "retain_baseline_by_candidate_state_transition",
        }
    )
)

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_RESULT_BYTES = 256 * 1024 * 1024


class LedgerPosteriorError(RuntimeError):
    """Raised when a frozen multi-draw ledger contract cannot be proved."""


def _fail(message: str) -> NoReturn:
    raise LedgerPosteriorError(message)


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
        _fail(f"{field} must be a timezone-aware datetime")
    return value.astimezone(UTC)


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _fail(f"{field} must be a positive integer")
    return value


def _project_root() -> Path:
    root = Path(__file__).resolve().parents[3]
    if not (root / "pyproject.toml").is_file() or not (root / "uv.lock").is_file():
        _fail("cannot locate the frozen PolicyTwin project root")
    return root


def policytwin_source_tree_sha256() -> str:
    """Commit to every Python source byte used by the local package.

    The whole package is intentionally included instead of a hand-maintained
    function allowlist.  This is a local source snapshot, not an external build
    attestation.
    """
    root = _project_root() / "src" / "policytwin"
    paths = tuple(sorted(root.rglob("*.py")))
    if not paths:
        _fail("PolicyTwin source tree is empty")
    material: list[dict[str, object]] = []
    for path in paths:
        if path.is_symlink():
            _fail("PolicyTwin source snapshot contains a symbolic link")
        try:
            before = path.stat()
            content = path.read_bytes()
            after = path.stat()
        except OSError as error:
            raise LedgerPosteriorError(
                "cannot capture PolicyTwin source tree"
            ) from error
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size != len(content)
            or (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
        ):
            _fail("PolicyTwin source changed while its snapshot was captured")
        material.append(
            {
                "path": path.relative_to(root.parent.parent).as_posix(),
                "sha256": sha256_bytes(content),
                "size_bytes": len(content),
            }
        )
    return sha256_bytes(canonical_json_bytes(material))


def ledger_posterior_implementation_sha256() -> str:
    """Hash this module plus all live functions and local class methods.

    The disk source snapshot alone cannot detect an in-process monkeypatch.
    Capturing the currently bound callables closes that gap for the aggregation
    layer while retaining an honest same-process, non-attested claim boundary.
    """
    try:
        return live_policytwin_implementation_sha256(
            module_path=Path(__file__), namespace=globals()
        )
    except ImplementationFingerprintError as error:
        raise LedgerPosteriorError(
            "cannot capture the complete live ledger posterior implementation"
        ) from error


class LedgerBenefitDirection(StrEnum):
    """Pre-outcome interpretation of one output; descriptive is not ordered."""

    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"
    DESCRIPTIVE = "descriptive_not_ordered"


class UndefinedMetricPolicy(StrEnum):
    """Frozen handling of scientifically undefined world metrics."""

    FAIL_CLOSED = "fail_closed"
    RETAIN_STATE_TRANSITIONS = "retain_state_transitions_without_numeric_summary"


class PosteriorScenarioDesign(StrEnum):
    """Closed cross-draw scenario design; no iid MCSE is claimed here."""

    ONE_KEYED_SCENARIO_PER_DRAW = "one_keyed_scenario_per_posterior_draw"


@dataclass(frozen=True, slots=True)
class LedgerExecutionEnvironment:
    """Local code/environment commitment captured before source execution."""

    source_tree_sha256: str
    environment_lock_sha256: str
    python_version: str
    python_implementation: str
    platform: str
    machine: str
    git_commit: str | None
    git_dirty: bool
    ledger_verifier_implementation_sha256: str
    ledger_posterior_implementation_sha256: str
    source_execution_dependency_closure_sha256: str

    def __post_init__(self) -> None:
        for field in (
            "source_tree_sha256",
            "environment_lock_sha256",
            "ledger_verifier_implementation_sha256",
            "ledger_posterior_implementation_sha256",
            "source_execution_dependency_closure_sha256",
        ):
            _sha256(getattr(self, field), field)
        for field in ("python_version", "python_implementation", "platform", "machine"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value:
                _fail(f"{field} must be non-empty text")
        if (
            self.git_commit is not None
            and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", self.git_commit) is None
        ):
            _fail("git_commit must be null or a full object ID")
        if not isinstance(self.git_dirty, bool):
            _fail("git_dirty must be boolean")

    @classmethod
    def capture(cls) -> Self:
        root = _project_root()
        git = collect_git_metadata(root)
        environment = collect_environment_metadata(root)
        if environment.lock_sha256 is None:
            _fail("the evaluation environment must have a frozen uv.lock")
        return cls(
            source_tree_sha256=policytwin_source_tree_sha256(),
            environment_lock_sha256=environment.lock_sha256,
            python_version=environment.python_version,
            python_implementation=environment.python_implementation,
            platform=environment.platform,
            machine=environment.machine,
            git_commit=git.commit,
            git_dirty=git.dirty,
            ledger_verifier_implementation_sha256=(
                ledger_evidence_verifier_implementation_sha256()
            ),
            ledger_posterior_implementation_sha256=(
                ledger_posterior_implementation_sha256()
            ),
            source_execution_dependency_closure_sha256=(
                live_policytwin_dependency_closure_sha256(
                    root_modules=(
                        calibration_integrated_module,
                        calibration_workflow_module,
                        integrated_synthetic_module,
                        ledger_derived_module,
                    )
                )
            ),
        )

    def require_current(self) -> None:
        current = type(self).capture()
        if current != self:
            _fail(
                "source tree, environment, Git state, ledger verifier, or "
                "aggregation/integrated execution implementation changed after freeze"
            )

    def to_mapping(self) -> dict[str, object]:
        return {
            "environment_lock_sha256": self.environment_lock_sha256,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "ledger_verifier_implementation_sha256": (
                self.ledger_verifier_implementation_sha256
            ),
            "ledger_posterior_implementation_sha256": (
                self.ledger_posterior_implementation_sha256
            ),
            "source_execution_dependency_closure_sha256": (
                self.source_execution_dependency_closure_sha256
            ),
            "machine": self.machine,
            "platform": self.platform,
            "python_implementation": self.python_implementation,
            "python_version": self.python_version,
            "source_tree_sha256": self.source_tree_sha256,
        }


@dataclass(frozen=True, slots=True)
class CalibrationPosteriorCommitment:
    """Exact lineage of the locally replayed synthetic calibration posterior."""

    calibration_config_sha256: str
    calibration_input_sha256: str
    calibration_protocol_receipt_sha256: str
    calibration_report_sha256: str
    posterior_sha256: str
    posterior_draw_count: int
    simulator_config_sha256: str
    simulator_id: str
    estimator_backend: str
    synthetic_only: Literal[True] = True

    def __post_init__(self) -> None:
        for field in (
            "calibration_config_sha256",
            "calibration_input_sha256",
            "calibration_protocol_receipt_sha256",
            "calibration_report_sha256",
            "posterior_sha256",
            "simulator_config_sha256",
        ):
            _sha256(getattr(self, field), field)
        if self.posterior_draw_count < 2:
            _fail("posterior source must contain at least two retained draws")
        _identifier(self.simulator_id, "simulator_id")
        _identifier(self.estimator_backend, "estimator_backend")
        if self.synthetic_only is not True:
            _fail("current posterior source must remain synthetic-only")

    @classmethod
    def from_run(cls, run: CalibrationRun) -> Self:
        if type(run) is not CalibrationRun:
            raise TypeError("run must be an exact CalibrationRun")
        try:
            verified = replace(run)
        except Exception as error:
            raise LedgerPosteriorError(
                "calibration source failed complete local replay"
            ) from error
        if verified.report.simulator_id != INTEGRATED_CALIBRATION_SIMULATOR_ID:
            _fail("ledger posterior source must use the integrated calibration model")
        return cls(
            calibration_config_sha256=verified.config.sha256,
            calibration_input_sha256=verified.observed.sha256,
            calibration_protocol_receipt_sha256=(
                verified.protocol_execution_receipt.sha256
            ),
            calibration_report_sha256=verified.report.sha256,
            posterior_sha256=verified.posterior.sha256,
            posterior_draw_count=len(verified.posterior.draws),
            simulator_config_sha256=verified.report.simulator_config_sha256,
            simulator_id=verified.report.simulator_id,
            estimator_backend=verified.posterior.backend_id,
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "calibration_config_sha256": self.calibration_config_sha256,
            "calibration_input_sha256": self.calibration_input_sha256,
            "calibration_protocol_receipt_sha256": (
                self.calibration_protocol_receipt_sha256
            ),
            "calibration_report_sha256": self.calibration_report_sha256,
            "estimator_backend": self.estimator_backend,
            "posterior_draw_count": self.posterior_draw_count,
            "posterior_sha256": self.posterior_sha256,
            "simulator_config_sha256": self.simulator_config_sha256,
            "simulator_id": self.simulator_id,
            "synthetic_only": self.synthetic_only,
        }


def _validated_integrated_calibration_simulator_mapping(
    *,
    run: CalibrationRun,
    evaluation_config: IntegratedSyntheticMarketConfig,
) -> dict[str, object]:
    """Recheck the exact calibrated-model dependency at every trust boundary."""
    try:
        mapping = read_json_bytes_object(
            run.simulator_config_json,
            source="calibration simulator configuration",
        )
    except MetadataValidationError as error:
        raise LedgerPosteriorError(str(error)) from error
    if set(mapping) != {
        "integrated_config_sha256",
        "schema_version",
        "simulator_id",
        "step_count",
        "synthetic_only",
    }:
        _fail("integrated calibration simulator configuration has drifted")
    step_count = mapping["step_count"]
    if (
        mapping["schema_version"] != 1
        or mapping["simulator_id"] != INTEGRATED_CALIBRATION_SIMULATOR_ID
        or mapping["synthetic_only"] is not True
        or isinstance(step_count, bool)
        or not isinstance(step_count, int)
        or step_count < 2
    ):
        _fail("integrated calibration simulator semantics have drifted")
    if evaluation_config.scenario_generator_spec is None:
        _fail("posterior policy evaluation requires a frozen forward generator")
    base_config_sha256 = replace(evaluation_config, scenario_generator_spec=None).sha256
    if mapping["integrated_config_sha256"] != base_config_sha256:
        _fail("policy evaluation config is not the calibrated integrated model")
    return mapping


def _require_factory_rederived_pair(
    *,
    run: CalibrationRun,
    ordinal: int,
    pair: PolicyWorldPair,
    event_key: EventKey,
    config: IntegratedSyntheticMarketConfig,
    calibrated_step_count: int,
) -> None:
    """Prove that the pair, including its shock tape, came from its manifest key."""
    candidate_config = dict(pair.candidate_policy.config)
    if set(candidate_config) != {"allow_buy", "allow_sell"} or not all(
        isinstance(candidate_config[name], bool) for name in ("allow_buy", "allow_sell")
    ):
        _fail("candidate policy switches differ from the integrated pair factory")
    horizons = {item.horizon_id: item.end_step for item in pair.horizons}
    if (
        set(horizons) != {"early", "terminal"}
        or horizons["early"] != 2
        or horizons["terminal"] != calibrated_step_count
    ):
        _fail("evaluation horizons differ from calibrated simulator step semantics")
    draw = run.posterior.draws[ordinal]
    try:
        candidates = tuple(
            integrated_synthetic_pair(
                event_key=event_key,
                parameters=dict(draw.parameters),
                config=config,
                candidate_allow_buy=cast(bool, candidate_config["allow_buy"]),
                candidate_allow_sell=cast(bool, candidate_config["allow_sell"]),
                candidate_is_noop=pair.candidate_policy.is_noop,
                step_count=calibrated_step_count,
                include_all_groups=include_all_groups,
            )
            for include_all_groups in (False, True)
        )
    except (TypeError, ValueError) as error:
        raise LedgerPosteriorError(
            "integrated pair cannot be replayed from its frozen factory inputs"
        ) from error
    if pair not in candidates:
        _fail(
            "paired scenario or shock tape does not rederive from its frozen event key"
        )


@dataclass(frozen=True, slots=True)
class LedgerPosteriorOutputSpec:
    """One output cell frozen before any source outcome is visible."""

    output_id: str
    horizon_id: str
    scope: DifferenceScope
    group_id: str | None
    metric_id: str
    metric_definition_sha256: str
    unit_context_sha256: str
    role: OutputRole
    benefit_direction: LedgerBenefitDirection
    undefined_policy: UndefinedMetricPolicy
    role_scope: Literal[
        "ledger_plan_local_synthetic_diagnostic_not_protocol_output_role"
    ] = LEDGER_POSTERIOR_OUTPUT_ROLE_SCOPE
    interval_levels: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.output_id, "output_id")
        _identifier(self.horizon_id, "horizon_id")
        _identifier(self.metric_id, "metric_id")
        _sha256(self.metric_definition_sha256, "metric_definition_sha256")
        _sha256(self.unit_context_sha256, "unit_context_sha256")
        if not isinstance(self.scope, DifferenceScope):
            _fail("output scope has the wrong type")
        if self.scope is DifferenceScope.GROUP:
            _identifier(self.group_id, "group_id")
        elif self.group_id is not None:
            _fail("market output cannot carry a group ID")
        if not isinstance(self.role, OutputRole):
            _fail("output role has the wrong type")
        if self.role_scope != LEDGER_POSTERIOR_OUTPUT_ROLE_SCOPE:
            _fail("ledger output role must remain plan-local synthetic metadata")
        if not isinstance(self.benefit_direction, LedgerBenefitDirection):
            _fail("benefit direction has the wrong type")
        if not isinstance(self.undefined_policy, UndefinedMetricPolicy):
            _fail("undefined policy has the wrong type")
        if self.role is OutputRole.PRIMARY and (
            self.undefined_policy is not UndefinedMetricPolicy.FAIL_CLOSED
        ):
            _fail("primary outputs must fail closed on every undefined draw")
        if not isinstance(self.interval_levels, tuple):
            _fail("interval levels must be immutable")
        levels: list[float] = []
        for value in self.interval_levels:
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not isfinite(value)
                or not 0.0 < float(value) < 1.0
            ):
                _fail("interval levels must be finite values strictly inside (0, 1)")
            levels.append(float(value))
        if tuple(sorted(set(levels))) != tuple(levels):
            _fail("interval levels must be unique and strictly sorted")

    def to_mapping(self) -> dict[str, object]:
        return {
            "benefit_direction": self.benefit_direction.value,
            "group_id": self.group_id,
            "horizon_id": self.horizon_id,
            "interval_levels": list(self.interval_levels),
            "interval_level_ratios": [
                {
                    "denominator": Fraction(str(level)).denominator,
                    "numerator": Fraction(str(level)).numerator,
                }
                for level in self.interval_levels
            ],
            "metric_definition_sha256": self.metric_definition_sha256,
            "metric_id": self.metric_id,
            "output_id": self.output_id,
            "role": self.role.value,
            "role_scope": self.role_scope,
            "scope": self.scope.value,
            "undefined_policy": self.undefined_policy.value,
            "unit_context_sha256": self.unit_context_sha256,
        }


def ledger_output_spec(
    *,
    output_id: str,
    horizon_id: str,
    scope: DifferenceScope,
    metric_id: str,
    config: IntegratedSyntheticMarketConfig,
    role: OutputRole,
    benefit_direction: LedgerBenefitDirection,
    undefined_policy: UndefinedMetricPolicy,
    group_id: str | None = None,
    interval_levels: tuple[float, ...] = (),
) -> LedgerPosteriorOutputSpec:
    """Build one output from the live closed registry and exact unit context."""
    if not isinstance(config, IntegratedSyntheticMarketConfig):
        raise TypeError("config must be an IntegratedSyntheticMarketConfig")
    definitions = {item.metric_id: item for item in ledger_metric_registry()}
    definition = definitions.get(metric_id)
    if definition is None:
        _fail("output references an unknown ledger metric")
    expected_scope = (
        LedgerMetricScope.GROUP
        if scope is DifferenceScope.GROUP
        else LedgerMetricScope.MARKET
    )
    if definition.scope is not expected_scope:
        _fail("output scope differs from the registered ledger metric scope")
    definition_sha256 = sha256_bytes(canonical_json_bytes(definition.to_mapping()))
    unit_context_sha256 = sha256_bytes(
        canonical_json_bytes(
            {
                "config_sha256": config.sha256,
                "market_config": config.market_config.to_mapping(),
                "metric_definition_sha256": definition_sha256,
                "price_representation": "integer_configured_cash_units",
                "synthetic_asset_id": "integrated_synthetic_single_asset",
            }
        )
    )
    return LedgerPosteriorOutputSpec(
        output_id=output_id,
        horizon_id=horizon_id,
        scope=scope,
        group_id=group_id,
        metric_id=metric_id,
        metric_definition_sha256=definition_sha256,
        unit_context_sha256=unit_context_sha256,
        role=role,
        benefit_direction=benefit_direction,
        undefined_policy=undefined_policy,
        interval_levels=interval_levels,
    )


@dataclass(frozen=True, slots=True)
class PosteriorDrawPlanEntry:
    """One immutable posterior/scenario slot registered before execution."""

    ordinal: int
    posterior_draw_id: str
    calibration_draw_sha256: str
    posterior_resample_event_key: str
    source_particle_index: int
    source_particle_sha256: str
    parameters: tuple[tuple[str, float], ...]
    parameter_sha256: str
    evaluation_event_key_canonical_text: str
    evaluation_event_key: str
    pair_draw_id: str
    pair_input_sha256: str
    model_draw_sha256: str
    scenario_id: str
    initial_state_sha256: str
    shock_tape_sha256: str
    weight_numerator: int
    weight_denominator: int

    def __post_init__(self) -> None:
        if isinstance(self.ordinal, bool) or not isinstance(self.ordinal, int):
            _fail("draw ordinal must be an integer")
        if self.ordinal < 0:
            _fail("draw ordinal must be non-negative")
        for field in ("posterior_draw_id", "pair_draw_id", "scenario_id"):
            _identifier(getattr(self, field), field)
        for field in (
            "calibration_draw_sha256",
            "source_particle_sha256",
            "parameter_sha256",
            "pair_input_sha256",
            "model_draw_sha256",
            "initial_state_sha256",
            "shock_tape_sha256",
        ):
            _sha256(getattr(self, field), field)
        if (
            isinstance(self.source_particle_index, bool)
            or not isinstance(self.source_particle_index, int)
            or self.source_particle_index < 0
        ):
            _fail("source particle index must be non-negative")
        if not isinstance(self.posterior_resample_event_key, str) or not (
            self.posterior_resample_event_key
        ):
            _fail("posterior resample event key must be non-empty")
        if not isinstance(self.evaluation_event_key, str) or not (
            self.evaluation_event_key.startswith("ptk1:")
            and len(self.evaluation_event_key) == 69
        ):
            _fail("evaluation event key must be a complete PolicyTwin key")
        if not isinstance(self.evaluation_event_key_canonical_text, str):
            _fail("evaluation event key canonical text must be retained")
        try:
            canonical_key = json.loads(self.evaluation_event_key_canonical_text)
        except json.JSONDecodeError as error:
            raise LedgerPosteriorError(
                "evaluation event key canonical text is invalid JSON"
            ) from error
        if (
            canonical_json_bytes(canonical_key).decode("utf-8")
            != self.evaluation_event_key_canonical_text
        ):
            _fail("evaluation event key text is not canonical JSON")
        if not isinstance(self.parameters, tuple) or not self.parameters:
            _fail("draw parameters must be a non-empty immutable tuple")
        names: list[str] = []
        normalized: list[tuple[str, float]] = []
        for name, value in self.parameters:
            names.append(_identifier(name, "parameter name"))
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not isfinite(value)
            ):
                _fail("draw parameters must be finite")
            normalized.append((name, float(value)))
        if tuple(sorted(normalized)) != tuple(normalized) or len(set(names)) != len(
            names
        ):
            _fail("draw parameters must be unique and sorted")
        expected_parameter_sha256 = sha256_bytes(
            canonical_json_bytes(
                [{"name": name, "value": value} for name, value in normalized]
            )
        )
        if self.parameter_sha256 != expected_parameter_sha256:
            _fail("draw parameter commitment does not reproduce")
        if self.weight_numerator != 1 or self.weight_denominator <= 1:
            _fail("the fixed first version requires equal positive draw weights")

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def to_mapping(self) -> dict[str, object]:
        return {
            "calibration_draw_sha256": self.calibration_draw_sha256,
            "evaluation_event_key": self.evaluation_event_key,
            "evaluation_event_key_canonical_text": (
                self.evaluation_event_key_canonical_text
            ),
            "initial_state_sha256": self.initial_state_sha256,
            "model_draw_sha256": self.model_draw_sha256,
            "ordinal": self.ordinal,
            "pair_draw_id": self.pair_draw_id,
            "pair_input_sha256": self.pair_input_sha256,
            "parameter_sha256": self.parameter_sha256,
            "parameters": [
                {"name": name, "value": value} for name, value in self.parameters
            ],
            "posterior_draw_id": self.posterior_draw_id,
            "posterior_resample_event_key": self.posterior_resample_event_key,
            "scenario_id": self.scenario_id,
            "shock_tape_sha256": self.shock_tape_sha256,
            "source_particle_index": self.source_particle_index,
            "source_particle_sha256": self.source_particle_sha256,
            "weight_denominator": self.weight_denominator,
            "weight_numerator": self.weight_numerator,
        }


@dataclass(frozen=True, slots=True, init=False)
class LedgerPosteriorStudyPlan:
    """Factory-produced plan whose hash is embedded in the source protocol."""

    plan_id: str
    frozen_at: datetime
    posterior_source: CalibrationPosteriorCommitment
    calibration_run: CalibrationRun
    config: IntegratedSyntheticMarketConfig
    pairs: tuple[PolicyWorldPair, ...]
    draw_entries: tuple[PosteriorDrawPlanEntry, ...]
    outputs: tuple[LedgerPosteriorOutputSpec, ...]
    environment: LedgerExecutionEnvironment
    scenario_design: PosteriorScenarioDesign
    metric_registry: tuple[LedgerMetricDefinition, ...]
    metric_registry_sha256: str
    estimand_id: str
    estimand_definition_sha256: str
    failure_policy: str
    fixed_draw_count: int
    synthetic_only: Literal[True]
    externally_registered: Literal[False]
    analysis_chronology_externally_verified: Literal[False]
    schema_version: str

    def __init__(self) -> None:
        raise TypeError(
            "LedgerPosteriorStudyPlan is factory-produced; use "
            "build_ledger_posterior_study_plan"
        )

    def _fresh_validate(self) -> None:
        """Rebuild every caller-reachable binding from typed frozen sources."""
        _identifier(self.plan_id, "plan_id")
        _aware(self.frozen_at, "frozen_at")
        if (
            self.schema_version != LEDGER_POSTERIOR_PLAN_SCHEMA_VERSION
            or self.synthetic_only is not True
            or self.externally_registered is not False
            or self.analysis_chronology_externally_verified is not False
            or self.scenario_design
            is not PosteriorScenarioDesign.ONE_KEYED_SCENARIO_PER_DRAW
            or self.estimand_id != LEDGER_POSTERIOR_ESTIMAND_ID
            or self.estimand_definition_sha256
            != LEDGER_POSTERIOR_ESTIMAND_DEFINITION_SHA256
            or self.failure_policy
            != "fixed_N_all_slots_required_any_source_failure_aborts_no_replacement"
        ):
            _fail("ledger posterior plan scope, estimand, or failure rule changed")
        if type(self.environment) is not LedgerExecutionEnvironment:
            _fail("ledger posterior environment has the wrong exact type")
        self.environment.require_current()
        if type(self.calibration_run) is not CalibrationRun:
            _fail("ledger posterior plan lost its typed calibration source")
        expected_source = CalibrationPosteriorCommitment.from_run(self.calibration_run)
        if self.posterior_source != expected_source:
            _fail("ledger posterior plan changed its calibration source")
        if not isinstance(self.config, IntegratedSyntheticMarketConfig):
            _fail("ledger posterior plan lost its typed integrated config")
        simulator_mapping = _validated_integrated_calibration_simulator_mapping(
            run=self.calibration_run,
            evaluation_config=self.config,
        )
        calibrated_step_count = cast(int, simulator_mapping["step_count"])
        if not isinstance(self.pairs, tuple) or not all(
            isinstance(item, PolicyWorldPair) for item in self.pairs
        ):
            _fail("ledger posterior pairs must be an immutable typed tuple")
        if not isinstance(self.draw_entries, tuple) or not all(
            isinstance(item, PosteriorDrawPlanEntry) for item in self.draw_entries
        ):
            _fail("ledger posterior entries must be an immutable typed tuple")
        if (
            self.fixed_draw_count != expected_source.posterior_draw_count
            or len(self.pairs) != self.fixed_draw_count
            or len(self.draw_entries) != self.fixed_draw_count
        ):
            _fail("ledger posterior plan does not cover the full posterior")
        for ordinal, (pair, entry) in enumerate(
            zip(self.pairs, self.draw_entries, strict=True)
        ):
            try:
                key_mapping = read_json_bytes_object(
                    entry.evaluation_event_key_canonical_text.encode("utf-8"),
                    source="frozen evaluation event key",
                )
            except MetadataValidationError as error:
                raise LedgerPosteriorError(str(error)) from error
            if set(key_mapping) != {
                "experiment_id",
                "purpose",
                "replicate_id",
                "schema_version",
                "seed",
                "simulation_step",
            }:
                _fail("frozen evaluation event key schema changed")
            experiment_id = key_mapping["experiment_id"]
            purpose = key_mapping["purpose"]
            seed = key_mapping["seed"]
            schema_version = key_mapping["schema_version"]
            replicate_id = key_mapping["replicate_id"]
            simulation_step = key_mapping["simulation_step"]
            if (
                not isinstance(experiment_id, str)
                or not isinstance(purpose, str)
                or not isinstance(seed, str)
                or not isinstance(schema_version, str)
                or isinstance(replicate_id, bool)
                or not isinstance(replicate_id, int)
                or isinstance(simulation_step, bool)
                or not isinstance(simulation_step, int)
            ):
                _fail("frozen evaluation event key fields have invalid types")
            event_key = EventKey(
                experiment_id=experiment_id,
                seed=seed,
                replicate_id=replicate_id,
                simulation_step=simulation_step,
                purpose=purpose,
                schema_version=schema_version,
            )
            if (
                experiment_id != self.plan_id
                or purpose != "ledger-posterior-policy-evaluation"
                or replicate_id != ordinal
                or simulation_step != 0
                or schema_version != EVENT_KEY_SCHEMA_VERSION
                or event_key.value != entry.evaluation_event_key
            ):
                _fail("evaluation event key differs from its frozen plan slot")
            _require_factory_rederived_pair(
                run=self.calibration_run,
                ordinal=ordinal,
                pair=pair,
                event_key=event_key,
                config=self.config,
                calibrated_step_count=calibrated_step_count,
            )
            expected_entry = _posterior_entry(
                ordinal=ordinal,
                run=self.calibration_run,
                pair=pair,
                event_key=event_key,
            )
            if entry != expected_entry:
                _fail("draw entry does not rederive from posterior, scenario, and pair")
        if tuple(item.ordinal for item in self.draw_entries) != tuple(
            range(self.fixed_draw_count)
        ):
            _fail("ledger posterior draw ordinals are incomplete")
        current_registry = ledger_metric_registry()
        if (
            self.metric_registry != current_registry
            or self.metric_registry_sha256 != ledger_metric_registry_sha256()
        ):
            _fail("ledger posterior metric registry changed after freeze")
        _validate_output_contract(
            outputs=self.outputs,
            config=self.config,
            pairs=self.pairs,
        )

    @property
    def draw_manifest_sha256(self) -> str:
        return sha256_bytes(
            canonical_json_bytes([item.to_mapping() for item in self.draw_entries])
        )

    @property
    def output_manifest_sha256(self) -> str:
        return sha256_bytes(
            canonical_json_bytes([item.to_mapping() for item in self.outputs])
        )

    @property
    def sha256(self) -> str:
        return sha256_bytes(self.canonical_bytes)

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_mapping())

    @property
    def frozen_definition(self) -> FrozenDefinition:
        return FrozenDefinition(
            definition_id=LEDGER_POSTERIOR_PLAN_DEFINITION_ID,
            description=(
                "Local synthetic ledger draw/output plan frozen before this "
                "authorized invocation; not externally preregistered."
            ),
            definition_sha256=self.sha256,
        )

    def to_mapping(self) -> dict[str, object]:
        """Return the acyclic payload; protocol/receipt/result hashes are absent."""
        return {
            "analysis_chronology_externally_verified": (
                self.analysis_chronology_externally_verified
            ),
            "config_sha256": self.config.sha256,
            "draw_entries": [item.to_mapping() for item in self.draw_entries],
            "draw_manifest_sha256": self.draw_manifest_sha256,
            "environment": self.environment.to_mapping(),
            "estimand_definition_sha256": self.estimand_definition_sha256,
            "estimand_id": self.estimand_id,
            "externally_registered": self.externally_registered,
            "failure_policy": self.failure_policy,
            "fixed_draw_count": self.fixed_draw_count,
            "frozen_at": self.frozen_at.isoformat(),
            "metric_registry": [item.to_mapping() for item in self.metric_registry],
            "metric_registry_sha256": self.metric_registry_sha256,
            "output_manifest_sha256": self.output_manifest_sha256,
            "outputs": [item.to_mapping() for item in self.outputs],
            "plan_id": self.plan_id,
            "posterior_source": self.posterior_source.to_mapping(),
            "scenario_design": self.scenario_design.value,
            "schema_version": self.schema_version,
            "synthetic_only": self.synthetic_only,
        }


def _validate_output_contract(
    *,
    outputs: tuple[LedgerPosteriorOutputSpec, ...],
    config: IntegratedSyntheticMarketConfig,
    pairs: tuple[PolicyWorldPair, ...],
) -> None:
    if not outputs:
        _fail("ledger posterior plan requires at least one output")
    if not all(isinstance(item, LedgerPosteriorOutputSpec) for item in outputs):
        _fail("ledger posterior outputs have the wrong type")
    output_ids = tuple(item.output_id for item in outputs)
    cell_ids = tuple(
        (item.horizon_id, item.scope, item.group_id, item.metric_id) for item in outputs
    )
    if len(set(output_ids)) != len(output_ids) or len(set(cell_ids)) != len(cell_ids):
        _fail("ledger posterior output IDs and scientific cells must be unique")
    if not any(item.role is OutputRole.PRIMARY for item in outputs):
        _fail("ledger posterior plan requires at least one primary output")
    horizons = {item.horizon_id for item in pairs[0].horizons}
    groups = {item.group_id for item in pairs[0].groups}
    for output in outputs:
        if output.horizon_id not in horizons:
            _fail("output references an unregistered horizon")
        if output.scope is DifferenceScope.GROUP and output.group_id not in groups:
            _fail("output references an unregistered frozen group")
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
            _fail("output definition or unit context differs from the live registry")


def _posterior_entry(
    *,
    ordinal: int,
    run: CalibrationRun,
    pair: PolicyWorldPair,
    event_key: EventKey,
) -> PosteriorDrawPlanEntry:
    draw = run.posterior.draws[ordinal]
    particle = run.posterior.particles[draw.source_particle]
    calibration_draw_sha256 = sha256_bytes(canonical_json_bytes(draw.to_mapping()))
    posterior_draw_id = (
        "posterior-"
        + sha256_bytes(
            canonical_json_bytes(
                {
                    "ordinal": ordinal,
                    "posterior_sha256": run.posterior.sha256,
                    "resample_event_key": draw.event_key,
                }
            )
        )[:24]
    )
    return PosteriorDrawPlanEntry(
        ordinal=ordinal,
        posterior_draw_id=posterior_draw_id,
        calibration_draw_sha256=calibration_draw_sha256,
        posterior_resample_event_key=draw.event_key,
        source_particle_index=draw.source_particle,
        source_particle_sha256=sha256_bytes(
            canonical_json_bytes(particle.to_mapping())
        ),
        parameters=tuple(draw.parameters),
        parameter_sha256=pair.posterior_draw.parameter_sha256,
        evaluation_event_key_canonical_text=event_key.canonical_text(),
        evaluation_event_key=event_key.value,
        pair_draw_id=pair.posterior_draw.draw_id,
        pair_input_sha256=integrated_synthetic_pair_input_sha256(pair),
        model_draw_sha256=pair.posterior_draw.model_draw_sha256,
        scenario_id=pair.scenario_id,
        initial_state_sha256=pair.initial_state.sha256,
        shock_tape_sha256=pair.shock_tape.sha256,
        weight_numerator=1,
        weight_denominator=len(run.posterior.draws),
    )


def build_ledger_posterior_study_plan(
    *,
    plan_id: str,
    frozen_at: datetime,
    calibration_run: CalibrationRun,
    config: IntegratedSyntheticMarketConfig,
    event_key_seed: str,
    outputs: tuple[LedgerPosteriorOutputSpec, ...],
    candidate_allow_buy: bool = False,
    candidate_allow_sell: bool = True,
    candidate_is_noop: bool = False,
    step_count: int = 4,
    include_all_groups: bool = False,
) -> LedgerPosteriorStudyPlan:
    """Freeze every posterior draw, scenario key, and output before execution."""
    stable_plan_id = _identifier(plan_id, "plan_id")
    stable_time = _aware(frozen_at, "frozen_at")
    if not isinstance(config, IntegratedSyntheticMarketConfig):
        raise TypeError("config must be an IntegratedSyntheticMarketConfig")
    if not isinstance(event_key_seed, str) or not event_key_seed:
        _fail("event_key_seed must be non-empty")
    posterior_source = CalibrationPosteriorCommitment.from_run(calibration_run)
    simulator_mapping = _validated_integrated_calibration_simulator_mapping(
        run=calibration_run,
        evaluation_config=config,
    )
    if step_count != simulator_mapping["step_count"]:
        _fail("evaluation step_count differs from the calibrated simulator")
    pairs: list[PolicyWorldPair] = []
    entries: list[PosteriorDrawPlanEntry] = []
    for ordinal, draw in enumerate(calibration_run.posterior.draws):
        event_key = EventKey(
            experiment_id=stable_plan_id,
            seed=event_key_seed,
            replicate_id=ordinal,
            simulation_step=0,
            purpose="ledger-posterior-policy-evaluation",
        )
        pair = integrated_synthetic_pair(
            event_key=event_key,
            parameters=dict(draw.parameters),
            config=config,
            candidate_allow_buy=(True if candidate_is_noop else candidate_allow_buy),
            candidate_allow_sell=(True if candidate_is_noop else candidate_allow_sell),
            candidate_is_noop=candidate_is_noop,
            step_count=step_count,
            include_all_groups=include_all_groups,
        )
        pairs.append(pair)
        entries.append(
            _posterior_entry(
                ordinal=ordinal,
                run=calibration_run,
                pair=pair,
                event_key=event_key,
            )
        )
    pair_tuple = tuple(pairs)
    entry_tuple = tuple(entries)
    if tuple(item.ordinal for item in entry_tuple) != tuple(range(len(entry_tuple))):
        _fail("posterior draw manifest ordinals are incomplete")
    for values, field in (
        ((item.posterior_draw_id for item in entry_tuple), "posterior draw IDs"),
        ((item.posterior_resample_event_key for item in entry_tuple), "resample keys"),
        ((item.evaluation_event_key for item in entry_tuple), "evaluation keys"),
        ((item.pair_draw_id for item in entry_tuple), "paired draw IDs"),
        ((item.pair_input_sha256 for item in entry_tuple), "pair inputs"),
        ((item.scenario_id for item in entry_tuple), "scenario IDs"),
    ):
        material = tuple(values)
        if len(set(material)) != len(material):
            _fail(f"{field} must be unique across every frozen slot")
    if len(pair_tuple) != posterior_source.posterior_draw_count:
        _fail("draw manifest does not cover the full calibration posterior")
    _validate_output_contract(outputs=outputs, config=config, pairs=pair_tuple)
    environment = LedgerExecutionEnvironment.capture()
    result = object.__new__(LedgerPosteriorStudyPlan)
    fields: tuple[tuple[str, object], ...] = (
        ("plan_id", stable_plan_id),
        ("frozen_at", stable_time),
        ("posterior_source", posterior_source),
        ("calibration_run", calibration_run),
        ("config", config),
        ("pairs", pair_tuple),
        ("draw_entries", entry_tuple),
        ("outputs", outputs),
        ("environment", environment),
        ("scenario_design", PosteriorScenarioDesign.ONE_KEYED_SCENARIO_PER_DRAW),
        ("metric_registry", ledger_metric_registry()),
        ("metric_registry_sha256", ledger_metric_registry_sha256()),
        ("estimand_id", LEDGER_POSTERIOR_ESTIMAND_ID),
        (
            "estimand_definition_sha256",
            LEDGER_POSTERIOR_ESTIMAND_DEFINITION_SHA256,
        ),
        (
            "failure_policy",
            "fixed_N_all_slots_required_any_source_failure_aborts_no_replacement",
        ),
        ("fixed_draw_count", len(entry_tuple)),
        ("synthetic_only", True),
        ("externally_registered", False),
        ("analysis_chronology_externally_verified", False),
        ("schema_version", LEDGER_POSTERIOR_PLAN_SCHEMA_VERSION),
    )
    for field, value in fields:
        object.__setattr__(result, field, value)
    result._fresh_validate()
    return result


@dataclass(frozen=True, slots=True, init=False)
class PreparedLedgerPosteriorStudy:
    """Opaque in-memory batch whose authorities have not yet been consumed."""

    plan: LedgerPosteriorStudyPlan
    protocol: StudyProtocol
    batch_authorization: ProtocolExecutionAuthorization
    source_authorizations: tuple[ProtocolExecutionAuthorization, ...]
    execution_started_at: datetime

    def __init__(self) -> None:
        raise TypeError(
            "PreparedLedgerPosteriorStudy is issuer-produced; use "
            "authorize_ledger_posterior_study"
        )


def _validate_plan_protocol(
    plan: LedgerPosteriorStudyPlan,
    protocol: StudyProtocol,
) -> None:
    if type(plan) is not LedgerPosteriorStudyPlan:
        raise TypeError("plan must be an exact LedgerPosteriorStudyPlan")
    if not isinstance(protocol, StudyProtocol):
        raise TypeError("protocol must be a StudyProtocol")
    plan._fresh_validate()
    if protocol.mode is not StudyMode.SYNTHETIC:
        _fail("ledger posterior execution accepts only a synthetic protocol")
    if plan.frozen_at > protocol.registered_at:
        _fail("ledger posterior plan was not frozen by protocol registration")
    evidence_plans = getattr(protocol, "evidence_plans", ())
    matches = tuple(
        item
        for item in evidence_plans
        if item.definition_id == LEDGER_POSTERIOR_PLAN_DEFINITION_ID
    )
    if matches != (plan.frozen_definition,):
        _fail("study protocol does not bind exactly this ledger posterior plan")
    if plan.config.cutoff != protocol.knowledge_cutoff:
        _fail("ledger posterior config cutoff differs from its protocol")
    for pair in plan.pairs:
        if (
            protocol.baseline_policy.definition_id != pair.baseline_policy.policy_id
            or protocol.baseline_policy.definition_sha256 != pair.baseline_policy.sha256
            or protocol.candidate_policy.definition_id
            != pair.candidate_policy.policy_id
            or protocol.candidate_policy.definition_sha256
            != pair.candidate_policy.sha256
        ):
            _fail("ledger posterior policies differ from the source protocol")
        protocol_horizons = {
            item.definition_id: item.definition_sha256 for item in protocol.horizons
        }
        expected_horizons = {
            item.horizon_id: sha256_bytes(canonical_json_bytes(item.to_mapping()))
            for item in pair.horizons
        }
        if protocol_horizons != expected_horizons:
            _fail("ledger posterior horizons differ from the source protocol")
        protocol_cohorts = {item.cohort_id: item for item in protocol.cohorts}
        expected_cohorts = {item.group_id: item.cohort_sha256 for item in pair.groups}
        if set(protocol_cohorts) != set(expected_cohorts) or any(
            protocol_cohorts[group_id].definition_sha256 != cohort_sha256
            or protocol_cohorts[group_id].basis is not CohortBasis.LATENT_STRATEGY
            for group_id, cohort_sha256 in expected_cohorts.items()
        ):
            _fail("ledger posterior cohorts differ from the source protocol")


def authorize_ledger_posterior_study(
    *,
    plan: LedgerPosteriorStudyPlan,
    protocol: StudyProtocol,
    execution_started_at: datetime,
) -> PreparedLedgerPosteriorStudy:
    """Issue all fixed-N authorities after the plan enters the protocol hash."""
    _validate_plan_protocol(plan, protocol)
    started_at = _aware(execution_started_at, "execution_started_at")
    plan.environment.require_current()
    issuer = SyntheticProtocolExecutionIssuer(protocol)
    batch_authorization = issuer.issue(
        execution_id=f"{plan.plan_id}-ledger-batch",
        consumer=ProtocolExecutionConsumer.LEDGER_POSTERIOR_EVALUATION,
        purpose=ProtocolExecutionPurpose.SYNTHETIC_LEDGER_POSTERIOR_EVALUATION,
        config_sha256=plan.sha256,
        input_sha256=plan.draw_manifest_sha256,
        execution_started_at=started_at,
    )
    source_authorizations = tuple(
        issuer.issue(
            execution_id=f"{plan.plan_id}-source-{entry.ordinal:06d}",
            consumer=ProtocolExecutionConsumer.INTEGRATED_SYNTHETIC_SIMULATION,
            purpose=ProtocolExecutionPurpose.PAIRED_SYNTHETIC_POLICY_SIMULATION,
            config_sha256=plan.config.sha256,
            input_sha256=entry.pair_input_sha256,
            execution_started_at=started_at,
        )
        for entry in plan.draw_entries
    )
    prepared = object.__new__(PreparedLedgerPosteriorStudy)
    object.__setattr__(prepared, "plan", plan)
    object.__setattr__(prepared, "protocol", protocol)
    object.__setattr__(prepared, "batch_authorization", batch_authorization)
    object.__setattr__(prepared, "source_authorizations", source_authorizations)
    object.__setattr__(prepared, "execution_started_at", started_at)
    return prepared


@dataclass(frozen=True, slots=True)
class LedgerPosteriorDrawExecution:
    """Post-run receipt for one and only one frozen plan slot."""

    ordinal: int
    posterior_draw_id: str
    plan_entry_sha256: str
    pair_input_sha256: str
    parameter_sha256: str
    model_draw_sha256: str
    scenario_id: str
    shock_tape_sha256: str
    source_execution_id: str
    source_protocol_execution_receipt_sha256: str
    source_artifact_sha256: str
    ledger_evaluation_sha256: str
    baseline_integrated_run_sha256: str
    candidate_integrated_run_sha256: str
    exogenous_event_manifest_sha256: str
    status: Literal["completed"] = "completed"

    def __post_init__(self) -> None:
        if isinstance(self.ordinal, bool) or not isinstance(self.ordinal, int):
            _fail("draw execution ordinal must be an integer")
        _identifier(self.posterior_draw_id, "posterior_draw_id")
        _identifier(self.scenario_id, "scenario_id")
        _identifier(self.source_execution_id, "source_execution_id")
        for field in (
            "plan_entry_sha256",
            "pair_input_sha256",
            "parameter_sha256",
            "model_draw_sha256",
            "shock_tape_sha256",
            "source_protocol_execution_receipt_sha256",
            "source_artifact_sha256",
            "ledger_evaluation_sha256",
            "baseline_integrated_run_sha256",
            "candidate_integrated_run_sha256",
            "exogenous_event_manifest_sha256",
        ):
            _sha256(getattr(self, field), field)
        if self.status != "completed":
            _fail("fixed-N result cannot silently retain a failed draw")

    def to_mapping(self) -> dict[str, object]:
        return {
            "baseline_integrated_run_sha256": self.baseline_integrated_run_sha256,
            "candidate_integrated_run_sha256": self.candidate_integrated_run_sha256,
            "exogenous_event_manifest_sha256": (self.exogenous_event_manifest_sha256),
            "ledger_evaluation_sha256": self.ledger_evaluation_sha256,
            "model_draw_sha256": self.model_draw_sha256,
            "ordinal": self.ordinal,
            "pair_input_sha256": self.pair_input_sha256,
            "parameter_sha256": self.parameter_sha256,
            "plan_entry_sha256": self.plan_entry_sha256,
            "posterior_draw_id": self.posterior_draw_id,
            "scenario_id": self.scenario_id,
            "shock_tape_sha256": self.shock_tape_sha256,
            "source_artifact_sha256": self.source_artifact_sha256,
            "source_execution_id": self.source_execution_id,
            "source_protocol_execution_receipt_sha256": (
                self.source_protocol_execution_receipt_sha256
            ),
            "status": self.status,
        }

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))


@dataclass(frozen=True, slots=True)
class LedgerPosteriorObservation:
    """Exact world values for one frozen draw; no float conversion is allowed."""

    ordinal: int
    posterior_draw_id: str
    draw_execution_sha256: str
    baseline: ExactMetric
    candidate: ExactMetric
    difference: ExactMetric

    def __post_init__(self) -> None:
        if isinstance(self.ordinal, bool) or not isinstance(self.ordinal, int):
            _fail("observation ordinal must be an integer")
        _identifier(self.posterior_draw_id, "posterior_draw_id")
        _sha256(self.draw_execution_sha256, "draw_execution_sha256")
        if not all(
            isinstance(item, ExactMetric)
            for item in (self.baseline, self.candidate, self.difference)
        ):
            _fail("ledger posterior observations require exact metrics")
        if (
            self.baseline.state is ExactMetricState.DEFINED
            and self.candidate.state is ExactMetricState.DEFINED
        ):
            assert self.baseline.value is not None and self.candidate.value is not None
            expected = ExactMetric(
                ExactMetricState.DEFINED,
                self.candidate.value.minus(self.baseline.value),
            )
        else:
            expected = ExactMetric.undefined(ExactMetricState.UNDEFINED_WORLD_METRIC)
        if self.difference != expected:
            _fail("posterior observation is not exact candidate minus baseline")

    def to_mapping(self) -> dict[str, object]:
        return {
            "baseline": self.baseline.to_mapping(),
            "candidate": self.candidate.to_mapping(),
            "difference": self.difference.to_mapping(),
            "draw_execution_sha256": self.draw_execution_sha256,
            "ordinal": self.ordinal,
            "posterior_draw_id": self.posterior_draw_id,
        }


@dataclass(frozen=True, slots=True)
class MetricStateTransitionCount:
    """Count of one explicit baseline-to-candidate definedness transition."""

    baseline_state: ExactMetricState
    candidate_state: ExactMetricState
    count: int

    def __post_init__(self) -> None:
        if not isinstance(self.baseline_state, ExactMetricState) or not isinstance(
            self.candidate_state, ExactMetricState
        ):
            _fail("state transition uses an unknown metric state")
        _positive_int(self.count, "state transition count")

    def to_mapping(self) -> dict[str, object]:
        return {
            "baseline_state": self.baseline_state.value,
            "candidate_state": self.candidate_state.value,
            "count": self.count,
        }


@dataclass(frozen=True, slots=True)
class ExactEmpiricalInterval:
    """Equal-tailed empirical order-statistic interval without interpolation."""

    level: float
    lower_order_index: int
    upper_order_index: int
    lower: ExactRational
    upper: ExactRational

    def __post_init__(self) -> None:
        if not 0.0 < self.level < 1.0:
            _fail("empirical interval level must lie inside (0, 1)")
        if (
            self.lower_order_index < 0
            or self.upper_order_index < self.lower_order_index
        ):
            _fail("empirical interval indices are invalid")
        if Fraction(self.lower.numerator, self.lower.denominator) > Fraction(
            self.upper.numerator, self.upper.denominator
        ):
            _fail("empirical interval endpoints are reversed")

    def to_mapping(self) -> dict[str, object]:
        level_ratio = Fraction(str(self.level))
        return {
            "level": self.level,
            "level_ratio": {
                "denominator": level_ratio.denominator,
                "numerator": level_ratio.numerator,
            },
            "lower": self.lower.to_mapping(),
            "lower_order_index": self.lower_order_index,
            "method": "equal_tailed_empirical_inverse_cdf_no_interpolation",
            "upper": self.upper.to_mapping(),
            "upper_order_index": self.upper_order_index,
        }


@dataclass(frozen=True, slots=True)
class LedgerPosteriorNumericSummary:
    """Exact all-draw summary, emitted only when every world value is defined."""

    draw_count: int
    mean_difference: ExactRational
    minimum_difference: ExactRational
    maximum_difference: ExactRational
    negative_count: int
    zero_count: int
    positive_count: int
    strict_benefit_count: int | None
    strict_harm_count: int | None
    intervals: tuple[ExactEmpiricalInterval, ...]

    def __post_init__(self) -> None:
        _positive_int(self.draw_count, "summary draw_count")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (self.negative_count, self.zero_count, self.positive_count)
        ):
            _fail("summary sign counts must be non-negative integers")
        if (
            self.negative_count + self.zero_count + self.positive_count
            != self.draw_count
        ):
            _fail("summary sign counts do not cover every draw")
        if (self.strict_benefit_count is None) != (self.strict_harm_count is None):
            _fail("benefit and harm counts must both be present or both absent")
        if self.strict_benefit_count is not None and (
            self.strict_benefit_count < 0
            or self.strict_harm_count is None
            or self.strict_harm_count < 0
            or self.strict_benefit_count + self.strict_harm_count > self.draw_count
        ):
            _fail("benefit/harm counts are invalid")

    def to_mapping(self) -> dict[str, object]:
        return {
            "draw_count": self.draw_count,
            "intervals": [item.to_mapping() for item in self.intervals],
            "maximum_difference": self.maximum_difference.to_mapping(),
            "mean_difference": self.mean_difference.to_mapping(),
            "minimum_difference": self.minimum_difference.to_mapping(),
            "negative_count": self.negative_count,
            "positive_count": self.positive_count,
            "strict_benefit_count": self.strict_benefit_count,
            "strict_harm_count": self.strict_harm_count,
            "zero_count": self.zero_count,
        }


@dataclass(frozen=True, slots=True)
class LedgerPosteriorDistributionCell:
    """Complete fixed-N cell including every undefined state transition."""

    output: LedgerPosteriorOutputSpec
    observations: tuple[LedgerPosteriorObservation, ...]
    state_transitions: tuple[MetricStateTransitionCount, ...]
    both_defined_count: int
    numeric_summary: LedgerPosteriorNumericSummary | None

    def __post_init__(self) -> None:
        if not isinstance(self.output, LedgerPosteriorOutputSpec):
            _fail("distribution cell requires a frozen output")
        if not isinstance(self.observations, tuple) or not self.observations:
            _fail("distribution cell must retain every frozen observation")
        ordinals = tuple(item.ordinal for item in self.observations)
        if ordinals != tuple(range(len(ordinals))):
            _fail("distribution observations do not cover canonical draw ordinals")
        expected = Counter(
            (item.baseline.state, item.candidate.state) for item in self.observations
        )
        observed = {
            (item.baseline_state, item.candidate_state): item.count
            for item in self.state_transitions
        }
        if observed != expected:
            _fail("state-transition counts do not reproduce from retained draws")
        defined = expected[(ExactMetricState.DEFINED, ExactMetricState.DEFINED)]
        if self.both_defined_count != defined:
            _fail("both-defined count does not reproduce")
        if (
            defined != len(self.observations)
            and self.output.undefined_policy is UndefinedMetricPolicy.FAIL_CLOSED
        ):
            _fail("fail-closed output has an undefined retained observation")
        if self.numeric_summary is None:
            if defined == len(self.observations):
                _fail("all-defined output cannot omit its exact numeric summary")
        elif defined != len(self.observations):
            _fail("partial definedness cannot be silently summarized complete-case")
        elif self.numeric_summary != _numeric_summary(self.output, self.observations):
            _fail("numeric summary does not rederive from every retained observation")

    def to_mapping(self) -> dict[str, object]:
        return {
            "both_defined_count": self.both_defined_count,
            "numeric_summary": (
                None
                if self.numeric_summary is None
                else self.numeric_summary.to_mapping()
            ),
            "observations": [item.to_mapping() for item in self.observations],
            "output": self.output.to_mapping(),
            "state_transitions": [item.to_mapping() for item in self.state_transitions],
        }


def _rational_key(value: ExactRational) -> Fraction:
    return Fraction(value.numerator, value.denominator)


def _numeric_summary(
    output: LedgerPosteriorOutputSpec,
    observations: tuple[LedgerPosteriorObservation, ...],
) -> LedgerPosteriorNumericSummary:
    values = tuple(cast(ExactRational, item.difference.value) for item in observations)
    ordered = tuple(sorted(values, key=_rational_key))
    total = ExactRational.from_ratio(0)
    for value in values:
        total = total.plus(value)
    signs = tuple((value.numerator > 0) - (value.numerator < 0) for value in values)
    if output.benefit_direction is LedgerBenefitDirection.DESCRIPTIVE:
        benefit: int | None = None
        harm: int | None = None
    elif output.benefit_direction is LedgerBenefitDirection.HIGHER_IS_BETTER:
        benefit = sum(item > 0 for item in signs)
        harm = sum(item < 0 for item in signs)
    else:
        benefit = sum(item < 0 for item in signs)
        harm = sum(item > 0 for item in signs)
    intervals: list[ExactEmpiricalInterval] = []
    count = len(ordered)
    for level in output.interval_levels:
        level_ratio = Fraction(str(level))
        tail = (Fraction(1, 1) - level_ratio) / 2

        def inverse_cdf_index(probability: Fraction) -> int:
            scaled = count * probability
            ceiling = (scaled.numerator + scaled.denominator - 1) // (
                scaled.denominator
            )
            return min(count - 1, max(0, ceiling - 1))

        lower_index = inverse_cdf_index(tail)
        upper_index = inverse_cdf_index(Fraction(1, 1) - tail)
        intervals.append(
            ExactEmpiricalInterval(
                level=level,
                lower_order_index=lower_index,
                upper_order_index=upper_index,
                lower=ordered[lower_index],
                upper=ordered[upper_index],
            )
        )
    return LedgerPosteriorNumericSummary(
        draw_count=count,
        mean_difference=ExactRational.from_ratio(
            total.numerator, total.denominator * count
        ),
        minimum_difference=ordered[0],
        maximum_difference=ordered[-1],
        negative_count=sum(item < 0 for item in signs),
        zero_count=sum(item == 0 for item in signs),
        positive_count=sum(item > 0 for item in signs),
        strict_benefit_count=benefit,
        strict_harm_count=harm,
        intervals=tuple(intervals),
    )


@dataclass(frozen=True, slots=True, init=False)
class LedgerPosteriorEvaluation:
    """Verifier-produced fixed-N aggregation; public construction is closed."""

    plan: LedgerPosteriorStudyPlan
    plan_sha256: str
    protocol_sha256: str
    batch_protocol_execution_receipt: ProtocolExecutionReceipt
    batch_protocol_execution_receipt_sha256: str
    posterior_source: CalibrationPosteriorCommitment
    draw_executions: tuple[LedgerPosteriorDrawExecution, ...]
    cells: tuple[LedgerPosteriorDistributionCell, ...]
    metric_registry_sha256: str
    verifier_implementation_source_sha256: str
    aggregation_implementation_source_sha256: str
    source_execution_dependency_closure_sha256: str
    source_tree_sha256: str
    environment_lock_sha256: str
    claim_scope: str
    synthetic_only: Literal[True]
    locally_bound_before_this_authorized_invocation: Literal[False]
    batch_authorization_consumption_persistently_verified: Literal[False]
    all_plan_slots_accounted: Literal[True]
    exact_ledger_arithmetic_verified: Literal[True]
    aggregate_rederived_from_sources: Literal[True]
    undefined_values_imputed: Literal[False]
    failed_draws_silently_dropped: Literal[False]
    externally_preregistered: Literal[False]
    analysis_chronology_externally_verified: Literal[False]
    posterior_provenance_independently_verified: Literal[False]
    source_execution_identity_externally_attested: Literal[False]
    source_artifacts_independently_verified: Literal[False]
    posterior_distribution_claim: Literal[False]
    credible_interval_claim: Literal[False]
    confidence_interval_claim: Literal[False]
    formal_welfare_evidence: Literal[False]
    integrated_mechanism_verified: Literal[False]
    real_world_causal_claim: Literal[False]
    observed_investor_identity: Literal[False]
    independent_reproduction: Literal[False]
    publication_eligible: Literal[False]
    separates_parameter_and_scenario_uncertainty: Literal[False]
    iid_mcse_reported: Literal[False]
    schema_version: str

    def __init__(self) -> None:
        raise TypeError(
            "LedgerPosteriorEvaluation is verifier-produced; use "
            "run_prepared_ledger_posterior_study"
        )

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_mapping())

    @property
    def sha256(self) -> str:
        return sha256_bytes(self.canonical_bytes)

    def to_mapping(self) -> dict[str, object]:
        return {
            "aggregate_rederived_from_sources": self.aggregate_rederived_from_sources,
            "aggregation_implementation_source_sha256": (
                self.aggregation_implementation_source_sha256
            ),
            "all_plan_slots_accounted": self.all_plan_slots_accounted,
            "analysis_chronology_externally_verified": (
                self.analysis_chronology_externally_verified
            ),
            "batch_protocol_execution_receipt": (
                self.batch_protocol_execution_receipt.to_mapping()
            ),
            "batch_protocol_execution_receipt_sha256": (
                self.batch_protocol_execution_receipt_sha256
            ),
            "batch_authorization_consumption_persistently_verified": (
                self.batch_authorization_consumption_persistently_verified
            ),
            "cells": [item.to_mapping() for item in self.cells],
            "claim_scope": self.claim_scope,
            "confidence_interval_claim": self.confidence_interval_claim,
            "credible_interval_claim": self.credible_interval_claim,
            "draw_executions": [item.to_mapping() for item in self.draw_executions],
            "environment_lock_sha256": self.environment_lock_sha256,
            "exact_ledger_arithmetic_verified": (self.exact_ledger_arithmetic_verified),
            "externally_preregistered": self.externally_preregistered,
            "failed_draws_silently_dropped": self.failed_draws_silently_dropped,
            "formal_welfare_evidence": self.formal_welfare_evidence,
            "iid_mcse_reported": self.iid_mcse_reported,
            "independent_reproduction": self.independent_reproduction,
            "source_execution_dependency_closure_sha256": (
                self.source_execution_dependency_closure_sha256
            ),
            "integrated_mechanism_verified": self.integrated_mechanism_verified,
            "locally_bound_before_this_authorized_invocation": (
                self.locally_bound_before_this_authorized_invocation
            ),
            "metric_registry_sha256": self.metric_registry_sha256,
            "observed_investor_identity": self.observed_investor_identity,
            "plan": self.plan.to_mapping(),
            "plan_sha256": self.plan_sha256,
            "posterior_distribution_claim": self.posterior_distribution_claim,
            "posterior_provenance_independently_verified": (
                self.posterior_provenance_independently_verified
            ),
            "posterior_source": self.posterior_source.to_mapping(),
            "protocol_sha256": self.protocol_sha256,
            "publication_eligible": self.publication_eligible,
            "real_world_causal_claim": self.real_world_causal_claim,
            "schema_version": self.schema_version,
            "separates_parameter_and_scenario_uncertainty": (
                self.separates_parameter_and_scenario_uncertainty
            ),
            "source_artifacts_independently_verified": (
                self.source_artifacts_independently_verified
            ),
            "source_execution_identity_externally_attested": (
                self.source_execution_identity_externally_attested
            ),
            "source_tree_sha256": self.source_tree_sha256,
            "synthetic_only": self.synthetic_only,
            "undefined_values_imputed": self.undefined_values_imputed,
            "verifier_implementation_source_sha256": (
                self.verifier_implementation_source_sha256
            ),
        }

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
        *,
        plan: LedgerPosteriorStudyPlan,
        protocol: StudyProtocol,
        batch_receipt: ProtocolExecutionReceipt,
        source_artifacts: Sequence[ProtocolBoundIntegratedSyntheticRunArtifact],
    ) -> Self:
        """Ignore nested authority and compare with a complete fresh derivation."""
        if not isinstance(value, Mapping) or not all(
            isinstance(key, str) for key in value
        ):
            _fail("ledger posterior result must be an object with string keys")
        expected = _derive_ledger_posterior_evaluation(
            plan=plan,
            protocol=protocol,
            batch_receipt=batch_receipt,
            source_artifacts=source_artifacts,
        )
        try:
            supplied = canonical_json_bytes(dict(value))
        except Exception as error:
            raise LedgerPosteriorError(
                "ledger posterior result is not canonical JSON material"
            ) from error
        if supplied != expected.canonical_bytes:
            _fail("ledger posterior result differs from fresh source-bound derivation")
        return cast(Self, expected)


def _difference_key(
    difference: PairedLedgerDifference,
) -> tuple[str, DifferenceScope, str | None, str]:
    return (
        difference.horizon_id,
        difference.scope,
        difference.group_id,
        difference.metric_id,
    )


def _draw_execution(
    *,
    entry: PosteriorDrawPlanEntry,
    artifact: ProtocolBoundIntegratedSyntheticRunArtifact,
    ledger: LedgerDerivedEvaluation,
) -> LedgerPosteriorDrawExecution:
    shock_events = tuple(
        item.event_key.to_mapping() for item in artifact.pair_input.shock_tape.entries
    )
    return LedgerPosteriorDrawExecution(
        ordinal=entry.ordinal,
        posterior_draw_id=entry.posterior_draw_id,
        plan_entry_sha256=entry.sha256,
        pair_input_sha256=artifact.input_sha256,
        parameter_sha256=artifact.pair_input.posterior_draw.parameter_sha256,
        model_draw_sha256=artifact.pair_input.posterior_draw.model_draw_sha256,
        scenario_id=artifact.pair_input.scenario_id,
        shock_tape_sha256=artifact.pair_input.shock_tape.sha256,
        source_execution_id=artifact.protocol_execution_receipt.execution_id,
        source_protocol_execution_receipt_sha256=(
            artifact.protocol_execution_receipt.sha256
        ),
        source_artifact_sha256=artifact.sha256,
        ledger_evaluation_sha256=ledger.sha256,
        baseline_integrated_run_sha256=artifact.baseline_run.sha256,
        candidate_integrated_run_sha256=artifact.candidate_run.sha256,
        exogenous_event_manifest_sha256=sha256_bytes(
            canonical_json_bytes(shock_events)
        ),
    )


def _validate_source_binding(
    *,
    plan: LedgerPosteriorStudyPlan,
    protocol: StudyProtocol,
    entry: PosteriorDrawPlanEntry,
    artifact: ProtocolBoundIntegratedSyntheticRunArtifact,
) -> None:
    if type(artifact) is not ProtocolBoundIntegratedSyntheticRunArtifact:
        raise TypeError("source artifact has the wrong exact type")
    pair = artifact.pair_input
    if (
        artifact.protocol != protocol
        or artifact.protocol.sha256 != protocol.sha256
        or artifact.config != plan.config
        or artifact.config_sha256 != plan.config.sha256
        or artifact.input_sha256 != entry.pair_input_sha256
        or integrated_synthetic_pair_input_sha256(pair) != entry.pair_input_sha256
        or pair.posterior_draw.draw_id != entry.pair_draw_id
        or pair.posterior_draw.parameters != entry.parameters
        or pair.posterior_draw.parameter_sha256 != entry.parameter_sha256
        or pair.posterior_draw.model_draw_sha256 != entry.model_draw_sha256
        or pair.scenario_id != entry.scenario_id
        or pair.initial_state.sha256 != entry.initial_state_sha256
        or pair.shock_tape.sha256 != entry.shock_tape_sha256
    ):
        _fail("source artifact differs from its frozen posterior/scenario slot")
    receipt = artifact.protocol_execution_receipt
    if (
        receipt.consumer
        is not ProtocolExecutionConsumer.INTEGRATED_SYNTHETIC_SIMULATION
        or receipt.purpose
        is not ProtocolExecutionPurpose.PAIRED_SYNTHETIC_POLICY_SIMULATION
        or receipt.protocol_sha256 != protocol.sha256
        or receipt.config_sha256 != plan.config.sha256
        or receipt.input_sha256 != entry.pair_input_sha256
    ):
        _fail("source receipt differs from its frozen slot")


def _build_distribution_cell(
    *,
    output: LedgerPosteriorOutputSpec,
    entries: tuple[PosteriorDrawPlanEntry, ...],
    executions: tuple[LedgerPosteriorDrawExecution, ...],
    ledgers: tuple[LedgerDerivedEvaluation, ...],
) -> LedgerPosteriorDistributionCell:
    target = (output.horizon_id, output.scope, output.group_id, output.metric_id)
    observations: list[LedgerPosteriorObservation] = []
    for entry, execution, ledger in zip(entries, executions, ledgers, strict=True):
        matches = tuple(
            item for item in ledger.differences if _difference_key(item) == target
        )
        if len(matches) != 1:
            _fail("registered ledger output is missing or duplicated in a source draw")
        difference = matches[0]
        observations.append(
            LedgerPosteriorObservation(
                ordinal=entry.ordinal,
                posterior_draw_id=entry.posterior_draw_id,
                draw_execution_sha256=execution.sha256,
                baseline=difference.baseline,
                candidate=difference.candidate,
                difference=difference.difference,
            )
        )
    observation_tuple = tuple(observations)
    counts = Counter(
        (item.baseline.state, item.candidate.state) for item in observation_tuple
    )
    transitions = tuple(
        MetricStateTransitionCount(
            baseline_state=baseline_state,
            candidate_state=candidate_state,
            count=count,
        )
        for (baseline_state, candidate_state), count in sorted(
            counts.items(), key=lambda item: (item[0][0].value, item[0][1].value)
        )
    )
    both_defined = counts[(ExactMetricState.DEFINED, ExactMetricState.DEFINED)]
    if both_defined != len(observation_tuple):
        if output.undefined_policy is UndefinedMetricPolicy.FAIL_CLOSED:
            _fail(f"registered output {output.output_id} has an undefined source draw")
        summary = None
    else:
        summary = _numeric_summary(output, observation_tuple)
    return LedgerPosteriorDistributionCell(
        output=output,
        observations=observation_tuple,
        state_transitions=transitions,
        both_defined_count=both_defined,
        numeric_summary=summary,
    )


def _derive_ledger_posterior_evaluation(
    *,
    plan: LedgerPosteriorStudyPlan,
    protocol: StudyProtocol,
    batch_receipt: ProtocolExecutionReceipt,
    source_artifacts: Sequence[ProtocolBoundIntegratedSyntheticRunArtifact],
) -> LedgerPosteriorEvaluation:
    _validate_plan_protocol(plan, protocol)
    plan.environment.require_current()
    if CalibrationPosteriorCommitment.from_run(plan.calibration_run) != (
        plan.posterior_source
    ):
        _fail("calibration posterior source changed after the plan was frozen")
    if not isinstance(batch_receipt, ProtocolExecutionReceipt):
        _fail("batch result requires a typed protocol execution receipt")
    validate_protocol_execution_receipt_binding(protocol, batch_receipt)
    if (
        batch_receipt.consumer
        is not ProtocolExecutionConsumer.LEDGER_POSTERIOR_EVALUATION
        or batch_receipt.purpose
        is not ProtocolExecutionPurpose.SYNTHETIC_LEDGER_POSTERIOR_EVALUATION
        or batch_receipt.config_sha256 != plan.sha256
        or batch_receipt.input_sha256 != plan.draw_manifest_sha256
    ):
        _fail("batch receipt does not bind the frozen ledger posterior plan")
    artifacts = tuple(source_artifacts)
    if len(artifacts) != plan.fixed_draw_count:
        _fail("source artifact count does not cover the fixed posterior draw plan")
    artifacts_by_input: dict[str, ProtocolBoundIntegratedSyntheticRunArtifact] = {}
    for artifact in artifacts:
        if type(artifact) is not ProtocolBoundIntegratedSyntheticRunArtifact:
            raise TypeError("source artifact has the wrong exact type")
        if artifact.input_sha256 in artifacts_by_input:
            _fail("one source artifact or pair input was reused across plan slots")
        artifacts_by_input[artifact.input_sha256] = artifact
    expected_inputs = {item.pair_input_sha256 for item in plan.draw_entries}
    if set(artifacts_by_input) != expected_inputs:
        _fail("source artifacts contain a missing, extra, or replacement draw")
    ordered_artifacts = tuple(
        artifacts_by_input[item.pair_input_sha256] for item in plan.draw_entries
    )
    ledgers: list[LedgerDerivedEvaluation] = []
    executions: list[LedgerPosteriorDrawExecution] = []
    for entry, artifact in zip(plan.draw_entries, ordered_artifacts, strict=True):
        _validate_source_binding(
            plan=plan,
            protocol=protocol,
            entry=entry,
            artifact=artifact,
        )
        ledger = derive_protocol_bound_ledger_evidence(artifact)
        ledgers.append(ledger)
        executions.append(
            _draw_execution(entry=entry, artifact=artifact, ledger=ledger)
        )
    execution_tuple = tuple(executions)
    for values, field in (
        (
            (item.source_execution_id for item in execution_tuple),
            "source execution IDs",
        ),
        (
            (item.source_protocol_execution_receipt_sha256 for item in execution_tuple),
            "source receipts",
        ),
        (
            (item.source_artifact_sha256 for item in execution_tuple),
            "source artifacts",
        ),
        (
            (item.ledger_evaluation_sha256 for item in execution_tuple),
            "ledger evaluations",
        ),
    ):
        material = tuple(values)
        if len(set(material)) != len(material):
            _fail(f"{field} must be unique across the complete draw plan")
    ledger_tuple = tuple(ledgers)
    cells = tuple(
        _build_distribution_cell(
            output=output,
            entries=plan.draw_entries,
            executions=execution_tuple,
            ledgers=ledger_tuple,
        )
        for output in plan.outputs
    )
    result = object.__new__(LedgerPosteriorEvaluation)
    fields: tuple[tuple[str, object], ...] = (
        ("plan", plan),
        ("plan_sha256", plan.sha256),
        ("protocol_sha256", protocol.sha256),
        ("batch_protocol_execution_receipt", batch_receipt),
        ("batch_protocol_execution_receipt_sha256", batch_receipt.sha256),
        ("posterior_source", plan.posterior_source),
        ("draw_executions", execution_tuple),
        ("cells", cells),
        ("metric_registry_sha256", plan.metric_registry_sha256),
        (
            "verifier_implementation_source_sha256",
            plan.environment.ledger_verifier_implementation_sha256,
        ),
        (
            "aggregation_implementation_source_sha256",
            plan.environment.ledger_posterior_implementation_sha256,
        ),
        (
            "source_execution_dependency_closure_sha256",
            plan.environment.source_execution_dependency_closure_sha256,
        ),
        ("source_tree_sha256", plan.environment.source_tree_sha256),
        ("environment_lock_sha256", plan.environment.environment_lock_sha256),
        ("claim_scope", LEDGER_POSTERIOR_CLAIM_SCOPE),
        ("synthetic_only", True),
        ("locally_bound_before_this_authorized_invocation", False),
        ("batch_authorization_consumption_persistently_verified", False),
        ("all_plan_slots_accounted", True),
        ("exact_ledger_arithmetic_verified", True),
        ("aggregate_rederived_from_sources", True),
        ("undefined_values_imputed", False),
        ("failed_draws_silently_dropped", False),
        ("externally_preregistered", False),
        ("analysis_chronology_externally_verified", False),
        ("posterior_provenance_independently_verified", False),
        ("source_execution_identity_externally_attested", False),
        ("source_artifacts_independently_verified", False),
        ("posterior_distribution_claim", False),
        ("credible_interval_claim", False),
        ("confidence_interval_claim", False),
        ("formal_welfare_evidence", False),
        ("integrated_mechanism_verified", False),
        ("real_world_causal_claim", False),
        ("observed_investor_identity", False),
        ("independent_reproduction", False),
        ("publication_eligible", False),
        ("separates_parameter_and_scenario_uncertainty", False),
        ("iid_mcse_reported", False),
        ("schema_version", LEDGER_POSTERIOR_RESULT_SCHEMA_VERSION),
    )
    for field, value in fields:
        object.__setattr__(result, field, value)
    plan.environment.require_current()
    return result


def run_prepared_ledger_posterior_study(
    prepared: PreparedLedgerPosteriorStudy,
    *,
    parallel_draws: bool = False,
    parallel_worlds: bool = True,
) -> tuple[
    LedgerPosteriorEvaluation,
    tuple[ProtocolBoundIntegratedSyntheticRunArtifact, ...],
]:
    """Consume batch authority first, then execute every precommitted source slot."""
    if type(prepared) is not PreparedLedgerPosteriorStudy:
        raise TypeError("prepared must be an exact PreparedLedgerPosteriorStudy")
    plan = prepared.plan
    protocol = prepared.protocol
    batch_receipt = consume_protocol_execution_authorization(
        prepared.batch_authorization,
        protocol=protocol,
        consumer=ProtocolExecutionConsumer.LEDGER_POSTERIOR_EVALUATION,
        purpose=ProtocolExecutionPurpose.SYNTHETIC_LEDGER_POSTERIOR_EVALUATION,
        config_sha256=plan.sha256,
        input_sha256=plan.draw_manifest_sha256,
        execution_started_at=prepared.execution_started_at,
    )
    plan.environment.require_current()

    def run_one(
        material: tuple[
            PolicyWorldPair,
            ProtocolExecutionAuthorization,
        ],
    ) -> ProtocolBoundIntegratedSyntheticRunArtifact:
        pair, authorization = material
        return run_protocol_bound_integrated_synthetic_pair(
            pair=pair,
            config=plan.config,
            protocol=protocol,
            authorization=authorization,
            execution_started_at=prepared.execution_started_at,
            parallel=parallel_worlds,
            invocation_order=(WorldRole.BASELINE, WorldRole.CANDIDATE),
        )

    source_material = tuple(
        zip(plan.pairs, prepared.source_authorizations, strict=True)
    )
    if parallel_draws:
        with ThreadPoolExecutor(max_workers=min(4, len(source_material))) as executor:
            artifacts = tuple(executor.map(run_one, source_material))
    else:
        artifacts = tuple(run_one(item) for item in source_material)
    result = _derive_ledger_posterior_evaluation(
        plan=plan,
        protocol=protocol,
        batch_receipt=batch_receipt,
        source_artifacts=artifacts,
    )
    return result, artifacts


def _fingerprint(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def verify_ledger_posterior_evaluation_bytes(
    content: bytes,
    *,
    expected_sha256: str,
    plan: LedgerPosteriorStudyPlan,
    protocol: StudyProtocol,
    batch_receipt: ProtocolExecutionReceipt,
    source_artifacts: Sequence[ProtocolBoundIntegratedSyntheticRunArtifact],
) -> LedgerPosteriorEvaluation:
    """Verify committed canonical bytes by a complete source-bound derivation.

    The caller-supplied digest commits only to ``content``.  It does not prove
    external custody, registration, historical execution, or independent
    reproduction, and this verifier does not broaden any result claim.
    """
    if type(content) is not bytes:
        raise TypeError("content must be exact bytes")
    expected = _sha256(expected_sha256, "expected_sha256")
    if not content or len(content) > _MAX_RESULT_BYTES:
        _fail("ledger posterior result bytes are empty or exceed their byte limit")
    if sha256_bytes(content) != expected:
        _fail("ledger posterior result hash differs from its external commitment")
    try:
        mapping = read_json_bytes_object(
            content,
            source="ledger posterior evaluation bytes",
        )
    except MetadataValidationError as error:
        raise LedgerPosteriorError(str(error)) from error
    if canonical_json_bytes(mapping) != content:
        _fail("ledger posterior result must use canonical duplicate-free JSON bytes")
    return LedgerPosteriorEvaluation.from_mapping(
        mapping,
        plan=plan,
        protocol=protocol,
        batch_receipt=batch_receipt,
        source_artifacts=source_artifacts,
    )


def load_ledger_posterior_evaluation(
    path: Path,
    *,
    expected_sha256: str,
    plan: LedgerPosteriorStudyPlan,
    protocol: StudyProtocol,
    batch_receipt: ProtocolExecutionReceipt,
    source_artifacts: Sequence[ProtocolBoundIntegratedSyntheticRunArtifact],
) -> LedgerPosteriorEvaluation:
    """Load a bounded canonical file and rederive it from all typed sources.

    The typed source artifacts remain same-process authority in this version;
    consequently the returned object keeps independent-reproduction and
    external-attestation flags false.
    """
    if not isinstance(path, Path):
        _fail("ledger posterior result path must be a Path")
    candidate = Path(os.path.abspath(path.expanduser()))
    current = Path(candidate.anchor)
    descriptor: int | None = None
    try:
        for part in candidate.parts[1:]:
            current /= part
            if current.is_symlink():
                _fail("ledger posterior result path contains a symbolic link")
        descriptor = os.open(
            candidate,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
        )
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > _MAX_RESULT_BYTES
        ):
            _fail("ledger posterior result is not a bounded singly linked file")
        chunks: list[bytes] = []
        total = 0
        while total <= _MAX_RESULT_BYTES:
            chunk = os.read(
                descriptor,
                min(1024 * 1024, _MAX_RESULT_BYTES + 1 - total),
            )
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        if total > _MAX_RESULT_BYTES:
            _fail("ledger posterior result exceeds its byte limit")
        after = os.fstat(descriptor)
        content = b"".join(chunks)
    except OSError as error:
        raise LedgerPosteriorError("cannot open ledger posterior result") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    try:
        path_after = os.stat(candidate, follow_symlinks=False)
    except OSError as error:
        raise LedgerPosteriorError("cannot recheck ledger posterior result") from error
    if (
        _fingerprint(before) != _fingerprint(after)
        or _fingerprint(after) != _fingerprint(path_after)
        or len(content) != before.st_size
    ):
        _fail("ledger posterior result changed while it was read")
    return verify_ledger_posterior_evaluation_bytes(
        content,
        expected_sha256=expected_sha256,
        plan=plan,
        protocol=protocol,
        batch_receipt=batch_receipt,
        source_artifacts=source_artifacts,
    )


__all__ = [
    "LEDGER_POSTERIOR_CLAIM_SCOPE",
    "LEDGER_POSTERIOR_ESTIMAND_DEFINITION_SHA256",
    "LEDGER_POSTERIOR_ESTIMAND_ID",
    "LEDGER_POSTERIOR_PLAN_DEFINITION_ID",
    "LEDGER_POSTERIOR_PLAN_SCHEMA_VERSION",
    "LEDGER_POSTERIOR_RESULT_SCHEMA_VERSION",
    "LEDGER_POSTERIOR_OUTPUT_ROLE_SCOPE",
    "CalibrationPosteriorCommitment",
    "ExactEmpiricalInterval",
    "LedgerBenefitDirection",
    "LedgerExecutionEnvironment",
    "LedgerPosteriorDistributionCell",
    "LedgerPosteriorDrawExecution",
    "LedgerPosteriorError",
    "LedgerPosteriorEvaluation",
    "LedgerPosteriorNumericSummary",
    "LedgerPosteriorObservation",
    "LedgerPosteriorOutputSpec",
    "LedgerPosteriorStudyPlan",
    "MetricStateTransitionCount",
    "PosteriorDrawPlanEntry",
    "PosteriorScenarioDesign",
    "PreparedLedgerPosteriorStudy",
    "UndefinedMetricPolicy",
    "authorize_ledger_posterior_study",
    "build_ledger_posterior_study_plan",
    "ledger_posterior_implementation_sha256",
    "ledger_output_spec",
    "load_ledger_posterior_evaluation",
    "policytwin_source_tree_sha256",
    "run_prepared_ledger_posterior_study",
    "verify_ledger_posterior_evaluation_bytes",
]
