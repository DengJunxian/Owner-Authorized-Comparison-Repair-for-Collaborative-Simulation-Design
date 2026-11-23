"""Fair, explicit baseline adapters for paired distributional evaluation.

The structural adapters in this module are deliberately small synthetic
mechanisms.  They are executable comparators for CI and architecture tests,
not evidence that any model is calibrated to a real market.  Every completed
row carries an engine-created execution receipt binding the frozen common
input, semantic event keys, implementation source, per-event mechanism trace,
and output draws.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from statistics import fmean
from typing import Literal, NoReturn, Protocol, runtime_checkable

from policytwin.counterfactual import HashBasedRNG, SemanticEventKey
from policytwin.provenance import canonical_json_bytes, sha256_bytes, sha256_file

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SEMANTIC_EVENT_KEY = re.compile(r"^pse1:[0-9a-f]{64}$")
BASELINE_MECHANISM_RECEIPT_SCHEMA_VERSION = (
    "policytwin.synthetic-baseline-mechanism-receipt.v1"
)
BASELINE_MECHANISM_CLAIM_SCOPE = "synthetic_comparator_mechanism_only"
BASELINE_SUITE_SCHEMA_VERSION = "policytwin.baseline-suite.v2"


class BaselineEvaluationError(ValueError):
    """Raised when a baseline suite violates its frozen comparison contract."""


def _fail(message: str) -> NoReturn:
    raise BaselineEvaluationError(message)


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        _fail(f"{field} must be a stable identifier")
    return value


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail(f"{field} must be a lowercase SHA-256 digest")
    return value


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        _fail(f"{field} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        _fail(f"{field} must be finite")
    return normalized


def _aware(value: object, field: str) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        _fail(f"{field} must be a timezone-aware datetime")
    return value.astimezone(UTC)


def _named_values(
    values: tuple[tuple[str, float], ...], field: str
) -> tuple[tuple[str, float], ...]:
    if not isinstance(values, tuple):
        _fail(f"{field} must be an immutable tuple")
    normalized = tuple(
        sorted(
            (_identifier(name, f"{field} name"), _finite(value, f"{field} value"))
            for name, value in values
        )
    )
    if len({name for name, _ in normalized}) != len(normalized):
        _fail(f"{field} names must be unique")
    return normalized


def _source_components() -> tuple[tuple[str, str], ...]:
    """Hash the actual executor source rather than trusting a caller digest."""
    path = Path(__file__).resolve()
    if not path.is_file():
        _fail("baseline executor source is unavailable")
    return (("policytwin.evaluation.baselines", sha256_file(path)),)


def _implementation_sha256(
    implementation_id: str,
    description: str,
    source_components: tuple[tuple[str, str], ...],
) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "description": description,
                "implementation_id": implementation_id,
                "source_components": [
                    {"module": module, "source_sha256": digest}
                    for module, digest in source_components
                ],
            }
        )
    )


@dataclass(frozen=True, slots=True)
class BaselineMechanismDescriptor:
    """Closed identity of code that actually computes one comparator."""

    implementation_id: str
    description: str
    source_components: tuple[tuple[str, str], ...]
    implementation_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.implementation_id, "implementation_id")
        if (
            not isinstance(self.description, str)
            or not self.description.strip()
            or len(self.description) > 512
        ):
            _fail("mechanism description must be a non-empty bounded string")
        if not isinstance(self.source_components, tuple) or not self.source_components:
            _fail("mechanism source_components must be a non-empty tuple")
        normalized = tuple(sorted(self.source_components))
        if len({module for module, _ in normalized}) != len(normalized):
            _fail("mechanism source component names must be unique")
        for module, digest in normalized:
            _identifier(module, "source component module")
            _sha256(digest, "source component sha256")
        object.__setattr__(self, "source_components", normalized)
        _sha256(self.implementation_sha256, "implementation_sha256")
        if self.implementation_sha256 != _implementation_sha256(
            self.implementation_id,
            self.description,
            normalized,
        ):
            _fail("implementation_sha256 does not match mechanism source")

    @classmethod
    def current(
        cls, implementation_id: str, description: str
    ) -> BaselineMechanismDescriptor:
        """Build a descriptor from executor bytes visible in this process."""
        components = _source_components()
        return cls(
            implementation_id=implementation_id,
            description=description,
            source_components=components,
            implementation_sha256=_implementation_sha256(
                implementation_id,
                description,
                components,
            ),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "description": self.description,
            "implementation_id": self.implementation_id,
            "implementation_sha256": self.implementation_sha256,
            "source_components": [
                {"module": module, "source_sha256": digest}
                for module, digest in self.source_components
            ],
        }


@dataclass(frozen=True, slots=True)
class BaselineMechanismStep:
    """Auditable, non-sensitive state used for one event-keyed draw."""

    event_key: str
    state: tuple[tuple[str, float], ...]
    output: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.event_key, str)
            or _SEMANTIC_EVENT_KEY.fullmatch(self.event_key) is None
        ):
            _fail("mechanism step event_key must be semantic")
        object.__setattr__(self, "state", _named_values(self.state, "mechanism state"))
        object.__setattr__(self, "output", _finite(self.output, "mechanism output"))

    def to_mapping(self) -> dict[str, object]:
        return {
            "event_key": self.event_key,
            "output": self.output,
            "state": [{"name": name, "value": value} for name, value in self.state],
        }


@dataclass(frozen=True, slots=True)
class BaselineExecution:
    """Raw execution returned to the suite before it issues a receipt."""

    draws: tuple[float, ...]
    steps: tuple[BaselineMechanismStep, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.draws, tuple) or not self.draws:
            _fail("baseline execution draws must be a non-empty tuple")
        draws = tuple(_finite(value, "baseline execution draw") for value in self.draws)
        object.__setattr__(self, "draws", draws)
        if (
            not isinstance(self.steps, tuple)
            or len(self.steps) != len(draws)
            or not all(isinstance(step, BaselineMechanismStep) for step in self.steps)
        ):
            _fail("baseline mechanism steps must align with draws")
        if tuple(step.output for step in self.steps) != draws:
            _fail("mechanism step outputs do not match baseline draws")


@dataclass(frozen=True, slots=True)
class BaselineMechanismReceipt:
    """Content-addressed evidence that a baseline mechanism actually ran."""

    descriptor: BaselineMechanismDescriptor
    spec_sha256: str
    input_bundle_sha256: str
    event_keys: tuple[str, ...]
    event_keys_sha256: str
    draws_sha256: str
    trace_sha256: str
    steps: tuple[BaselineMechanismStep, ...]
    mechanism_execution: Literal[True] = True
    synthetic_only: Literal[True] = True
    claim_scope: str = BASELINE_MECHANISM_CLAIM_SCOPE
    schema_version: str = BASELINE_MECHANISM_RECEIPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.descriptor, BaselineMechanismDescriptor):
            _fail("receipt descriptor is invalid")
        _sha256(self.spec_sha256, "receipt spec_sha256")
        _sha256(self.input_bundle_sha256, "receipt input_bundle_sha256")
        if (
            not isinstance(self.event_keys, tuple)
            or not self.event_keys
            or len(set(self.event_keys)) != len(self.event_keys)
            or not all(
                isinstance(key, str) and _SEMANTIC_EVENT_KEY.fullmatch(key) is not None
                for key in self.event_keys
            )
        ):
            _fail("receipt event keys must be unique semantic commitments")
        for field in ("event_keys_sha256", "draws_sha256", "trace_sha256"):
            _sha256(getattr(self, field), field)
        if (
            not isinstance(self.steps, tuple)
            or len(self.steps) != len(self.event_keys)
            or not all(isinstance(step, BaselineMechanismStep) for step in self.steps)
        ):
            _fail("receipt steps must align with event keys")
        if tuple(step.event_key for step in self.steps) != self.event_keys:
            _fail("receipt step keys do not match event_keys")
        if self.event_keys_sha256 != sha256_bytes(
            canonical_json_bytes(list(self.event_keys))
        ):
            _fail("receipt event_keys_sha256 does not match event keys")
        draws = tuple(step.output for step in self.steps)
        if self.draws_sha256 != sha256_bytes(canonical_json_bytes(list(draws))):
            _fail("receipt draws_sha256 does not match mechanism outputs")
        if self.trace_sha256 != sha256_bytes(
            canonical_json_bytes([step.to_mapping() for step in self.steps])
        ):
            _fail("receipt trace_sha256 does not match mechanism trace")
        if self.mechanism_execution is not True or self.synthetic_only is not True:
            _fail("baseline mechanism receipt must be executed and synthetic-only")
        if self.claim_scope != BASELINE_MECHANISM_CLAIM_SCOPE:
            _fail("baseline mechanism receipt claim scope cannot be widened")
        if self.schema_version != BASELINE_MECHANISM_RECEIPT_SCHEMA_VERSION:
            _fail("unsupported baseline mechanism receipt schema")

    @property
    def draws(self) -> tuple[float, ...]:
        return tuple(step.output for step in self.steps)

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def to_mapping(self) -> dict[str, object]:
        return {
            "claim_scope": self.claim_scope,
            "descriptor": self.descriptor.to_mapping(),
            "draws_sha256": self.draws_sha256,
            "event_keys": list(self.event_keys),
            "event_keys_sha256": self.event_keys_sha256,
            "input_bundle_sha256": self.input_bundle_sha256,
            "mechanism_execution": self.mechanism_execution,
            "schema_version": self.schema_version,
            "spec_sha256": self.spec_sha256,
            "steps": [step.to_mapping() for step in self.steps],
            "synthetic_only": self.synthetic_only,
            "trace_sha256": self.trace_sha256,
        }


class BaselineKind(StrEnum):
    """Preregistered comparator families from the research protocol."""

    ZERO_EFFECT = "zero_effect"
    HISTORICAL_BOOTSTRAP = "historical_bootstrap"
    REDUCED_FORM_TRANSFER = "reduced_form_transfer"
    CLASSIC_CALIBRATED_ABM = "classic_calibrated_abm"
    POINT_CALIBRATED_ABM = "point_calibrated_abm"
    LLM_ABM = "llm_abm"
    ORDER_GENERATOR = "order_generator"
    FULL_POLICYTWIN = "full_policytwin"


class BaselineStatus(StrEnum):
    """Terminal status retained for every registered baseline."""

    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"


RESEARCH_BASELINE_KINDS: frozenset[BaselineKind] = frozenset(BaselineKind)
"""Comparator families required by the complete PolicyTwin protocol."""


@dataclass(frozen=True, slots=True)
class BaselineInputBundle:
    """Identical frozen input capability presented to every baseline adapter."""

    scenario_id: str
    study_protocol_sha256: str
    data_manifest_sha256: str
    cohort_sha256: str
    horizon_sha256: str
    outcome_definition_sha256: str
    frozen_at: datetime
    target_outcomes_first_available_at: datetime
    event_keys: tuple[SemanticEventKey, ...]
    prepolicy_features: tuple[tuple[str, float], ...]
    prepolicy_residuals: tuple[float, ...]
    synthetic_only: bool
    policy_post_data_used: bool = False

    def __post_init__(self) -> None:
        _identifier(self.scenario_id, "scenario_id")
        for field in (
            "study_protocol_sha256",
            "data_manifest_sha256",
            "cohort_sha256",
            "horizon_sha256",
            "outcome_definition_sha256",
        ):
            _sha256(getattr(self, field), field)
        object.__setattr__(self, "frozen_at", _aware(self.frozen_at, "frozen_at"))
        object.__setattr__(
            self,
            "target_outcomes_first_available_at",
            _aware(
                self.target_outcomes_first_available_at,
                "target_outcomes_first_available_at",
            ),
        )
        if self.frozen_at >= self.target_outcomes_first_available_at:
            _fail("baseline inputs must be frozen before target outcomes are available")
        if (
            not isinstance(self.event_keys, tuple)
            or not self.event_keys
            or not all(isinstance(key, SemanticEventKey) for key in self.event_keys)
        ):
            _fail("event_keys must be a non-empty immutable semantic-key tuple")
        ordered = tuple(sorted(self.event_keys, key=lambda key: key.value))
        if len({key.value for key in ordered}) != len(ordered):
            _fail("baseline event keys must be unique")
        if any(key.scenario_id != self.scenario_id for key in ordered):
            _fail("baseline event keys must share the input scenario")
        object.__setattr__(self, "event_keys", ordered)
        object.__setattr__(
            self,
            "prepolicy_features",
            _named_values(self.prepolicy_features, "prepolicy_features"),
        )
        if not isinstance(self.prepolicy_residuals, tuple):
            _fail("prepolicy_residuals must be an immutable tuple")
        object.__setattr__(
            self,
            "prepolicy_residuals",
            tuple(
                _finite(value, "prepolicy residual")
                for value in self.prepolicy_residuals
            ),
        )
        if not isinstance(self.synthetic_only, bool):
            _fail("synthetic_only must be boolean")
        if self.policy_post_data_used is not False:
            _fail("policy-post data are forbidden in baseline construction")

    @property
    def sha256(self) -> str:
        """Return the common-input commitment used by the fairness audit."""
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def to_mapping(self) -> dict[str, object]:
        """Return the canonical input representation."""
        return {
            "cohort_sha256": self.cohort_sha256,
            "data_manifest_sha256": self.data_manifest_sha256,
            "event_keys": [key.to_mapping() for key in self.event_keys],
            "frozen_at": self.frozen_at.isoformat(),
            "horizon_sha256": self.horizon_sha256,
            "outcome_definition_sha256": self.outcome_definition_sha256,
            "policy_post_data_used": self.policy_post_data_used,
            "prepolicy_features": [
                {"name": name, "value": value}
                for name, value in self.prepolicy_features
            ],
            "prepolicy_residuals": list(self.prepolicy_residuals),
            "scenario_id": self.scenario_id,
            "study_protocol_sha256": self.study_protocol_sha256,
            "synthetic_only": self.synthetic_only,
            "target_outcomes_first_available_at": (
                self.target_outcomes_first_available_at.isoformat()
            ),
        }


@dataclass(frozen=True, slots=True)
class BaselineSpec:
    """One immutable baseline identity and complete adapter configuration."""

    baseline_id: str
    kind: BaselineKind
    config: tuple[tuple[str, str | int | float | bool], ...]
    required: bool = True

    def __post_init__(self) -> None:
        _identifier(self.baseline_id, "baseline_id")
        if not isinstance(self.kind, BaselineKind):
            _fail("kind must be a BaselineKind")
        if not isinstance(self.config, tuple):
            _fail("baseline config must be an immutable tuple")
        normalized: list[tuple[str, str | int | float | bool]] = []
        for name, value in self.config:
            stable_name = _identifier(name, "baseline config name")
            if isinstance(value, float) and not math.isfinite(value):
                _fail("baseline config floats must be finite")
            if not isinstance(value, str | int | float | bool):
                _fail("baseline config values must be JSON scalars")
            normalized.append((stable_name, value))
        normalized.sort()
        if len({name for name, _ in normalized}) != len(normalized):
            _fail("baseline config names must be unique")
        object.__setattr__(self, "config", tuple(normalized))
        if not isinstance(self.required, bool):
            _fail("required must be boolean")

    @property
    def sha256(self) -> str:
        """Return the complete baseline configuration commitment."""
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def to_mapping(self) -> dict[str, object]:
        """Return the canonical specification."""
        return {
            "baseline_id": self.baseline_id,
            "config": {name: value for name, value in self.config},
            "kind": self.kind.value,
            "required": self.required,
        }


@dataclass(frozen=True, slots=True)
class BaselineResult:
    """One terminal baseline row, including blocked and failed rows."""

    baseline_id: str
    kind: BaselineKind
    required: bool
    status: BaselineStatus
    spec_sha256: str
    input_bundle_sha256: str
    event_keys: tuple[str, ...]
    draws: tuple[float, ...]
    failure_code: str | None
    synthetic_only: bool
    mechanism_receipt: BaselineMechanismReceipt | None = None

    def __post_init__(self) -> None:
        _identifier(self.baseline_id, "baseline_id")
        if not isinstance(self.kind, BaselineKind) or not isinstance(
            self.status, BaselineStatus
        ):
            _fail("baseline result enum value is invalid")
        if not isinstance(self.required, bool):
            _fail("baseline result required flag must be boolean")
        _sha256(self.spec_sha256, "spec_sha256")
        _sha256(self.input_bundle_sha256, "input_bundle_sha256")
        if not isinstance(self.event_keys, tuple) or not all(
            isinstance(value, str) and value for value in self.event_keys
        ):
            _fail("event_keys must be an immutable string tuple")
        if not isinstance(self.draws, tuple):
            _fail("draws must be an immutable tuple")
        object.__setattr__(
            self,
            "draws",
            tuple(_finite(value, "baseline draw") for value in self.draws),
        )
        if self.status is BaselineStatus.COMPLETED:
            if not self.draws or len(self.draws) != len(self.event_keys):
                _fail("completed baseline must provide one draw per event key")
            if self.failure_code is not None:
                _fail("completed baseline cannot carry a failure code")
            if not isinstance(self.mechanism_receipt, BaselineMechanismReceipt):
                _fail("completed baseline must carry a mechanism execution receipt")
            receipt = self.mechanism_receipt
            if (
                receipt.spec_sha256 != self.spec_sha256
                or receipt.input_bundle_sha256 != self.input_bundle_sha256
                or receipt.event_keys != self.event_keys
                or receipt.draws != self.draws
            ):
                _fail("baseline result disagrees with its mechanism receipt")
        else:
            if self.draws or self.event_keys:
                _fail("blocked or failed baselines cannot carry partial draws")
            _identifier(self.failure_code, "failure_code")
            if self.mechanism_receipt is not None:
                _fail("blocked or failed baselines cannot carry execution receipts")
        if not isinstance(self.synthetic_only, bool):
            _fail("synthetic_only must be boolean")

    def to_mapping(self) -> dict[str, object]:
        """Return the complete terminal row."""
        return {
            "baseline_id": self.baseline_id,
            "draws": list(self.draws),
            "event_keys": list(self.event_keys),
            "failure_code": self.failure_code,
            "input_bundle_sha256": self.input_bundle_sha256,
            "kind": self.kind.value,
            "mechanism_receipt": (
                None
                if self.mechanism_receipt is None
                else self.mechanism_receipt.to_mapping()
            ),
            "required": self.required,
            "spec_sha256": self.spec_sha256,
            "status": self.status.value,
            "synthetic_only": self.synthetic_only,
        }


@runtime_checkable
class BaselineAdapter(Protocol):
    """Adapter contract for one preregistered, identically scoped baseline."""

    @property
    def spec(self) -> BaselineSpec:
        """Return the complete immutable baseline specification."""
        ...

    def run(self, inputs: BaselineInputBundle) -> tuple[float, ...]:
        """Return one outcome draw per common semantic event key."""
        ...


@runtime_checkable
class AuditedBaselineAdapter(Protocol):
    """Executable adapter that exposes code identity and per-key trace."""

    @property
    def spec(self) -> BaselineSpec: ...

    def describe_mechanism(self) -> BaselineMechanismDescriptor:
        """Return a descriptor derived from current executor bytes."""
        ...

    def execute(self, inputs: BaselineInputBundle) -> BaselineExecution:
        """Execute once and return aligned mechanism state and outputs."""
        ...


def _issue_mechanism_receipt(
    *,
    adapter: AuditedBaselineAdapter,
    inputs: BaselineInputBundle,
    execution: BaselineExecution,
) -> BaselineMechanismReceipt:
    event_keys = tuple(key.value for key in inputs.event_keys)
    if tuple(step.event_key for step in execution.steps) != event_keys:
        _fail("adapter mechanism trace does not use the common event keys")
    descriptor = adapter.describe_mechanism()
    return BaselineMechanismReceipt(
        descriptor=descriptor,
        spec_sha256=adapter.spec.sha256,
        input_bundle_sha256=inputs.sha256,
        event_keys=event_keys,
        event_keys_sha256=sha256_bytes(canonical_json_bytes(list(event_keys))),
        draws_sha256=sha256_bytes(canonical_json_bytes(list(execution.draws))),
        trace_sha256=sha256_bytes(
            canonical_json_bytes([step.to_mapping() for step in execution.steps])
        ),
        steps=execution.steps,
    )


def execute_baseline_adapter(
    adapter: AuditedBaselineAdapter,
    inputs: BaselineInputBundle,
) -> tuple[BaselineExecution, BaselineMechanismReceipt]:
    """Execute one audited adapter and issue an engine-owned receipt."""
    if not isinstance(adapter, AuditedBaselineAdapter):
        _fail("adapter does not implement the audited execution interface")
    if not isinstance(inputs, BaselineInputBundle):
        _fail("baseline execution requires a BaselineInputBundle")
    execution = adapter.execute(inputs)
    if not isinstance(execution, BaselineExecution):
        _fail("adapter returned an invalid BaselineExecution")
    return execution, _issue_mechanism_receipt(
        adapter=adapter,
        inputs=inputs,
        execution=execution,
    )


def audit_baseline_execution(
    adapter: AuditedBaselineAdapter,
    inputs: BaselineInputBundle,
    receipt: BaselineMechanismReceipt,
) -> None:
    """Re-execute a synthetic adapter and require byte-equivalent evidence."""
    if not isinstance(adapter, AuditedBaselineAdapter):
        _fail("adapter does not implement the audited execution interface")
    if not isinstance(inputs, BaselineInputBundle) or not isinstance(
        receipt, BaselineMechanismReceipt
    ):
        _fail("baseline audit requires typed inputs and receipt")
    _, current = execute_baseline_adapter(adapter, inputs)
    if current != receipt:
        _fail("baseline mechanism receipt does not reproduce")


@dataclass(frozen=True, slots=True)
class ZeroEffectBaseline:
    """Executable zero-effect benchmark."""

    baseline_id: str = "zero_effect"

    @property
    def spec(self) -> BaselineSpec:
        return BaselineSpec(
            baseline_id=self.baseline_id,
            kind=BaselineKind.ZERO_EFFECT,
            config=(("effect", 0.0),),
        )

    def describe_mechanism(self) -> BaselineMechanismDescriptor:
        return BaselineMechanismDescriptor.current(
            "zero-effect.v1",
            "Return an exact zero effect for every frozen semantic event key.",
        )

    def execute(self, inputs: BaselineInputBundle) -> BaselineExecution:
        if not isinstance(inputs, BaselineInputBundle):
            _fail("zero baseline requires BaselineInputBundle")
        steps = tuple(
            BaselineMechanismStep(
                event_key=key.value,
                state=(("constant_effect", 0.0),),
                output=0.0,
            )
            for key in inputs.event_keys
        )
        return BaselineExecution(tuple(step.output for step in steps), steps)

    def run(self, inputs: BaselineInputBundle) -> tuple[float, ...]:
        return self.execute(inputs).draws


@dataclass(frozen=True, slots=True)
class HistoricalBootstrapBaseline:
    """Keyed bootstrap from a frozen pre-policy residual distribution."""

    seed: str
    baseline_id: str = "historical_bootstrap"

    def __post_init__(self) -> None:
        if not isinstance(self.seed, str) or not self.seed:
            _fail("historical bootstrap seed must be non-empty")

    @property
    def spec(self) -> BaselineSpec:
        return BaselineSpec(
            baseline_id=self.baseline_id,
            kind=BaselineKind.HISTORICAL_BOOTSTRAP,
            config=(
                ("sampling", "event_keyed_with_replacement"),
                ("seed_sha256", sha256_bytes(self.seed.encode())),
            ),
        )

    def describe_mechanism(self) -> BaselineMechanismDescriptor:
        return BaselineMechanismDescriptor.current(
            "historical-bootstrap.v1",
            "Sample a frozen pre-policy residual by stateless semantic event key.",
        )

    def execute(self, inputs: BaselineInputBundle) -> BaselineExecution:
        if not inputs.prepolicy_residuals:
            _fail("historical bootstrap requires frozen pre-policy residuals")
        rng = HashBasedRNG(self.seed)
        steps: list[BaselineMechanismStep] = []
        for key in inputs.event_keys:
            residual_index = rng.integer(
                key,
                0,
                len(inputs.prepolicy_residuals) - 1,
                domain="historical_baseline_residual",
            )
            output = inputs.prepolicy_residuals[residual_index]
            steps.append(
                BaselineMechanismStep(
                    event_key=key.value,
                    state=(("residual_index", float(residual_index)),),
                    output=output,
                )
            )
        frozen_steps = tuple(steps)
        return BaselineExecution(
            tuple(step.output for step in frozen_steps), frozen_steps
        )

    def run(self, inputs: BaselineInputBundle) -> tuple[float, ...]:
        return self.execute(inputs).draws


@dataclass(frozen=True, slots=True)
class FrozenLinearTransferBaseline:
    """Reduced-form transfer benchmark with coefficients frozen pre-outcome."""

    intercept: float
    coefficients: tuple[tuple[str, float], ...]
    baseline_id: str = "reduced_form_transfer"

    def __post_init__(self) -> None:
        object.__setattr__(self, "intercept", _finite(self.intercept, "intercept"))
        object.__setattr__(
            self,
            "coefficients",
            _named_values(self.coefficients, "coefficients"),
        )
        if not self.coefficients:
            _fail("linear transfer baseline requires at least one coefficient")

    @property
    def spec(self) -> BaselineSpec:
        config: list[tuple[str, str | int | float | bool]] = [
            ("intercept", self.intercept),
            ("method", "frozen_linear_transfer"),
        ]
        config.extend(
            (f"coefficient.{name}", value) for name, value in self.coefficients
        )
        return BaselineSpec(
            baseline_id=self.baseline_id,
            kind=BaselineKind.REDUCED_FORM_TRANSFER,
            config=tuple(config),
        )

    def describe_mechanism(self) -> BaselineMechanismDescriptor:
        return BaselineMechanismDescriptor.current(
            "frozen-linear-transfer.v1",
            "Apply preregistered linear coefficients to frozen pre-policy features.",
        )

    def execute(self, inputs: BaselineInputBundle) -> BaselineExecution:
        features = dict(inputs.prepolicy_features)
        missing = tuple(name for name, _ in self.coefficients if name not in features)
        if missing:
            _fail(f"reduced-form transfer is missing frozen features: {missing!r}")
        prediction = self.intercept + math.fsum(
            coefficient * features[name] for name, coefficient in self.coefficients
        )
        state = tuple(
            (f"term_{index}", coefficient * features[name])
            for index, (name, coefficient) in enumerate(self.coefficients)
        ) + (("intercept", self.intercept),)
        steps = tuple(
            BaselineMechanismStep(key.value, state, prediction)
            for key in inputs.event_keys
        )
        return BaselineExecution(tuple(step.output for step in steps), steps)

    def run(self, inputs: BaselineInputBundle) -> tuple[float, ...]:
        return self.execute(inputs).draws


def _synthetic_features(inputs: BaselineInputBundle) -> tuple[float, float]:
    if not isinstance(inputs, BaselineInputBundle):
        _fail("structural comparator requires BaselineInputBundle")
    if inputs.synthetic_only is not True:
        _fail("synthetic structural adapters cannot consume real-data bundles")
    features = dict(inputs.prepolicy_features)
    missing = tuple(name for name in ("depth", "spread") if name not in features)
    if missing:
        _fail(f"structural comparator is missing frozen features: {missing!r}")
    spread = _finite(features["spread"], "spread")
    depth = _finite(features["depth"], "depth")
    if spread < 0.0 or depth <= 0.0:
        _fail("structural comparator requires spread >= 0 and depth > 0")
    return spread, depth


def _seed(value: object, field: str = "seed") -> str:
    if not isinstance(value, str) or not value or len(value) > 512:
        _fail(f"{field} must be a non-empty bounded string")
    return value


def _unit_interval(value: object, field: str) -> float:
    normalized = _finite(value, field)
    if not 0.0 <= normalized <= 1.0:
        _fail(f"{field} must lie in [0, 1]")
    return normalized


@dataclass(frozen=True, slots=True)
class ClassicCalibratedABMBaseline:
    """Synthetic chartist/fundamentalist ABM with frozen calibration identity."""

    seed: str
    calibration_artifact_sha256: str
    fundamentalist_share: float = 0.65
    adjustment_speed: float = 0.8
    policy_tick_change: float = 1.0
    baseline_id: str = "classic_calibrated_abm"

    def __post_init__(self) -> None:
        _seed(self.seed)
        _sha256(self.calibration_artifact_sha256, "calibration_artifact_sha256")
        object.__setattr__(
            self,
            "fundamentalist_share",
            _unit_interval(self.fundamentalist_share, "fundamentalist_share"),
        )
        for field in ("adjustment_speed", "policy_tick_change"):
            value = _finite(getattr(self, field), field)
            if value <= 0.0:
                _fail(f"{field} must be positive")
            object.__setattr__(self, field, value)
        _identifier(self.baseline_id, "baseline_id")

    @property
    def spec(self) -> BaselineSpec:
        return BaselineSpec(
            baseline_id=self.baseline_id,
            kind=BaselineKind.CLASSIC_CALIBRATED_ABM,
            config=(
                ("adjustment_speed", self.adjustment_speed),
                ("calibration_artifact_sha256", self.calibration_artifact_sha256),
                ("fundamentalist_share", self.fundamentalist_share),
                ("mechanism", "chartist_fundamentalist_interaction"),
                ("policy_tick_change", self.policy_tick_change),
                ("seed_sha256", sha256_bytes(self.seed.encode())),
                ("synthetic_only", True),
            ),
        )

    def describe_mechanism(self) -> BaselineMechanismDescriptor:
        return BaselineMechanismDescriptor.current(
            "classic-chartist-fundamentalist-abm.v1",
            "Combine keyed fundamental-value and chartist demand, then clear "
            "through a nonlinear price adjustment rule.",
        )

    def execute(self, inputs: BaselineInputBundle) -> BaselineExecution:
        spread, depth = _synthetic_features(inputs)
        rng = HashBasedRNG(self.seed)
        steps: list[BaselineMechanismStep] = []
        for key in inputs.event_keys:
            fundamental_signal = -spread / depth + 0.15 * rng.normal(
                key, domain="classic_abm_fundamental_signal"
            )
            chartist_signal = rng.normal(key, domain="classic_abm_chartist_signal")
            aggregate_demand = (
                self.fundamentalist_share * fundamental_signal
                + (1.0 - self.fundamentalist_share) * chartist_signal
            )
            output = (
                self.adjustment_speed
                * self.policy_tick_change
                * math.tanh(aggregate_demand)
            )
            steps.append(
                BaselineMechanismStep(
                    key.value,
                    (
                        ("aggregate_demand", aggregate_demand),
                        ("chartist_signal", chartist_signal),
                        ("fundamental_signal", fundamental_signal),
                        ("fundamentalist_share", self.fundamentalist_share),
                    ),
                    output,
                )
            )
        frozen = tuple(steps)
        return BaselineExecution(tuple(step.output for step in frozen), frozen)

    def run(self, inputs: BaselineInputBundle) -> tuple[float, ...]:
        return self.execute(inputs).draws


@dataclass(frozen=True, slots=True)
class PointCalibratedABMBaseline:
    """Synthetic representative-agent ABM using one frozen parameter vector."""

    seed: str
    point_estimate_sha256: str
    spread_sensitivity: float = -0.12
    inverse_depth_sensitivity: float = 0.6
    noise_scale: float = 0.05
    baseline_id: str = "point_calibrated_abm"

    def __post_init__(self) -> None:
        _seed(self.seed)
        _sha256(self.point_estimate_sha256, "point_estimate_sha256")
        for field in (
            "spread_sensitivity",
            "inverse_depth_sensitivity",
            "noise_scale",
        ):
            value = _finite(getattr(self, field), field)
            if field == "noise_scale" and value < 0.0:
                _fail("noise_scale must be non-negative")
            object.__setattr__(self, field, value)
        _identifier(self.baseline_id, "baseline_id")

    @property
    def spec(self) -> BaselineSpec:
        return BaselineSpec(
            baseline_id=self.baseline_id,
            kind=BaselineKind.POINT_CALIBRATED_ABM,
            config=(
                ("inverse_depth_sensitivity", self.inverse_depth_sensitivity),
                ("mechanism", "representative_agent_point_parameters"),
                ("noise_scale", self.noise_scale),
                ("point_estimate_sha256", self.point_estimate_sha256),
                ("seed_sha256", sha256_bytes(self.seed.encode())),
                ("spread_sensitivity", self.spread_sensitivity),
                ("synthetic_only", True),
            ),
        )

    def describe_mechanism(self) -> BaselineMechanismDescriptor:
        return BaselineMechanismDescriptor.current(
            "point-representative-agent-abm.v1",
            "Apply one frozen representative-agent parameter vector to spread, "
            "inverse depth, and keyed innovations.",
        )

    def execute(self, inputs: BaselineInputBundle) -> BaselineExecution:
        spread, depth = _synthetic_features(inputs)
        rng = HashBasedRNG(self.seed)
        steps: list[BaselineMechanismStep] = []
        for key in inputs.event_keys:
            innovation = rng.normal(key, domain="point_abm_innovation")
            deterministic_component = (
                self.spread_sensitivity * spread
                + self.inverse_depth_sensitivity / depth
            )
            output = deterministic_component + self.noise_scale * innovation
            steps.append(
                BaselineMechanismStep(
                    key.value,
                    (
                        ("deterministic_component", deterministic_component),
                        ("innovation", innovation),
                        ("inverse_depth", 1.0 / depth),
                    ),
                    output,
                )
            )
        frozen = tuple(steps)
        return BaselineExecution(tuple(step.output for step in frozen), frozen)

    def run(self, inputs: BaselineInputBundle) -> tuple[float, ...]:
        return self.execute(inputs).draws


@dataclass(frozen=True, slots=True)
class DeterministicMockLLMABMBaseline:
    """Offline rule-based mock of an LLM-ABM; no model or network is invoked."""

    seed: str
    prompt_schema_sha256: str
    buy_threshold: float = 0.25
    sell_threshold: float = -0.25
    action_scale: float = 0.4
    baseline_id: str = "llm_abm"

    def __post_init__(self) -> None:
        _seed(self.seed)
        _sha256(self.prompt_schema_sha256, "prompt_schema_sha256")
        for field in ("buy_threshold", "sell_threshold", "action_scale"):
            object.__setattr__(self, field, _finite(getattr(self, field), field))
        if self.sell_threshold >= self.buy_threshold:
            _fail("mock LLM sell_threshold must be below buy_threshold")
        if self.action_scale <= 0.0:
            _fail("mock LLM action_scale must be positive")
        _identifier(self.baseline_id, "baseline_id")

    @property
    def spec(self) -> BaselineSpec:
        return BaselineSpec(
            baseline_id=self.baseline_id,
            kind=BaselineKind.LLM_ABM,
            config=(
                ("action_scale", self.action_scale),
                ("adapter", "deterministic_mock_only"),
                ("buy_threshold", self.buy_threshold),
                ("network_access", False),
                ("prompt_schema_sha256", self.prompt_schema_sha256),
                ("seed_sha256", sha256_bytes(self.seed.encode())),
                ("sell_threshold", self.sell_threshold),
                ("synthetic_only", True),
            ),
        )

    def describe_mechanism(self) -> BaselineMechanismDescriptor:
        return BaselineMechanismDescriptor.current(
            "deterministic-mock-llm-abm.v1",
            "Map anonymous frozen numeric features to a closed buy-hold-sell rule; "
            "make no external model or network call.",
        )

    def execute(self, inputs: BaselineInputBundle) -> BaselineExecution:
        spread, depth = _synthetic_features(inputs)
        rng = HashBasedRNG(self.seed)
        steps: list[BaselineMechanismStep] = []
        for key in inputs.event_keys:
            anonymous_salience = (
                2.0 * rng.uniform(key, domain="mock_llm_anonymous_salience") - 1.0
            )
            score = anonymous_salience + spread / (1.0 + depth)
            action = 1.0 if score >= self.buy_threshold else 0.0
            if score <= self.sell_threshold:
                action = -1.0
            confidence = min(1.0, abs(score))
            output = action * confidence * self.action_scale
            steps.append(
                BaselineMechanismStep(
                    key.value,
                    (
                        ("action_code", action),
                        ("anonymous_salience", anonymous_salience),
                        ("confidence", confidence),
                        ("score", score),
                    ),
                    output,
                )
            )
        frozen = tuple(steps)
        return BaselineExecution(tuple(step.output for step in frozen), frozen)

    def run(self, inputs: BaselineInputBundle) -> tuple[float, ...]:
        return self.execute(inputs).draws


@dataclass(frozen=True, slots=True)
class OrderLevelGeneratorBaseline:
    """Synthetic order-arrival generator with keyed buy/sell count imbalance."""

    seed: str
    generator_checkpoint_sha256: str
    arrival_probability: float = 0.45
    opportunities: int = 8
    impact_scale: float = 1.5
    baseline_id: str = "order_generator"

    def __post_init__(self) -> None:
        _seed(self.seed)
        _sha256(self.generator_checkpoint_sha256, "generator_checkpoint_sha256")
        object.__setattr__(
            self,
            "arrival_probability",
            _unit_interval(self.arrival_probability, "arrival_probability"),
        )
        if (
            isinstance(self.opportunities, bool)
            or not isinstance(self.opportunities, int)
            or self.opportunities < 1
            or self.opportunities > 128
        ):
            _fail("opportunities must be an integer in [1, 128]")
        object.__setattr__(
            self, "impact_scale", _finite(self.impact_scale, "impact_scale")
        )
        if self.impact_scale <= 0.0:
            _fail("impact_scale must be positive")
        _identifier(self.baseline_id, "baseline_id")

    @property
    def spec(self) -> BaselineSpec:
        return BaselineSpec(
            baseline_id=self.baseline_id,
            kind=BaselineKind.ORDER_GENERATOR,
            config=(
                ("arrival_probability", self.arrival_probability),
                ("generator_checkpoint_sha256", self.generator_checkpoint_sha256),
                ("impact_scale", self.impact_scale),
                ("mechanism", "keyed_order_arrival_imbalance"),
                ("opportunities", self.opportunities),
                ("seed_sha256", sha256_bytes(self.seed.encode())),
                ("synthetic_only", True),
            ),
        )

    def describe_mechanism(self) -> BaselineMechanismDescriptor:
        return BaselineMechanismDescriptor.current(
            "order-level-arrival-generator.v1",
            "Generate keyed buy and sell arrivals, aggregate order imbalance, "
            "and map it through a depth-sensitive impact rule.",
        )

    def execute(self, inputs: BaselineInputBundle) -> BaselineExecution:
        spread, depth = _synthetic_features(inputs)
        rng = HashBasedRNG(self.seed)
        steps: list[BaselineMechanismStep] = []
        for key in inputs.event_keys:
            buys = sum(
                rng.bernoulli(
                    key,
                    self.arrival_probability,
                    domain="order_generator_buy_arrival",
                    draw_index=index,
                )
                for index in range(self.opportunities)
            )
            sells = sum(
                rng.bernoulli(
                    key,
                    self.arrival_probability,
                    domain="order_generator_sell_arrival",
                    draw_index=index,
                )
                for index in range(self.opportunities)
            )
            imbalance = float(buys - sells)
            output = self.impact_scale * imbalance * (1.0 + spread) / (1.0 + depth)
            steps.append(
                BaselineMechanismStep(
                    key.value,
                    (
                        ("buy_arrivals", float(buys)),
                        ("order_imbalance", imbalance),
                        ("sell_arrivals", float(sells)),
                    ),
                    output,
                )
            )
        frozen = tuple(steps)
        return BaselineExecution(tuple(step.output for step in frozen), frozen)

    def run(self, inputs: BaselineInputBundle) -> tuple[float, ...]:
        return self.execute(inputs).draws


@dataclass(frozen=True, slots=True)
class FullPolicyTwinSyntheticBaseline:
    """Small heterogeneous/social/memory mechanism for synthetic integration."""

    seed: str
    posterior_artifact_sha256: str
    behavioral_share: float = 0.55
    social_strength: float = 0.35
    memory_weight: float = 0.4
    inventory_aversion: float = 0.3
    policy_intensity: float = 1.0
    baseline_id: str = "full_policytwin"

    def __post_init__(self) -> None:
        _seed(self.seed)
        _sha256(self.posterior_artifact_sha256, "posterior_artifact_sha256")
        for field in ("behavioral_share", "social_strength", "memory_weight"):
            object.__setattr__(self, field, _unit_interval(getattr(self, field), field))
        for field in ("inventory_aversion", "policy_intensity"):
            value = _finite(getattr(self, field), field)
            if value < 0.0:
                _fail(f"{field} must be non-negative")
            object.__setattr__(self, field, value)
        _identifier(self.baseline_id, "baseline_id")

    @property
    def spec(self) -> BaselineSpec:
        return BaselineSpec(
            baseline_id=self.baseline_id,
            kind=BaselineKind.FULL_POLICYTWIN,
            config=(
                ("behavioral_share", self.behavioral_share),
                ("inventory_aversion", self.inventory_aversion),
                ("mechanism", "heterogeneous_social_memory_interaction"),
                ("memory_weight", self.memory_weight),
                ("policy_intensity", self.policy_intensity),
                ("posterior_artifact_sha256", self.posterior_artifact_sha256),
                ("seed_sha256", sha256_bytes(self.seed.encode())),
                ("social_strength", self.social_strength),
                ("synthetic_only", True),
            ),
        )

    def describe_mechanism(self) -> BaselineMechanismDescriptor:
        return BaselineMechanismDescriptor.current(
            "full-policytwin-synthetic-interaction.v1",
            "Combine keyed heterogeneous private signals, social exposure, decayed "
            "memory, and inventory restraint in a nonlinear policy response.",
        )

    def execute(self, inputs: BaselineInputBundle) -> BaselineExecution:
        spread, depth = _synthetic_features(inputs)
        rng = HashBasedRNG(self.seed)
        steps: list[BaselineMechanismStep] = []
        for key in inputs.event_keys:
            rational_signal = -spread / (1.0 + depth)
            behavioral_signal = rng.normal(
                key, domain="full_policytwin_behavioral_signal"
            )
            social_signal = (
                2.0 * rng.uniform(key, domain="full_policytwin_social_signal") - 1.0
            )
            memory_signal = rng.normal(key, domain="full_policytwin_memory_signal")
            inventory = rng.normal(key, domain="full_policytwin_inventory")
            combined = (
                (1.0 - self.behavioral_share) * rational_signal
                + self.behavioral_share * behavioral_signal
                + self.social_strength * social_signal
                + self.memory_weight * memory_signal
                - self.inventory_aversion * inventory
            )
            output = self.policy_intensity * math.tanh(combined)
            steps.append(
                BaselineMechanismStep(
                    key.value,
                    (
                        ("behavioral_signal", behavioral_signal),
                        ("combined_signal", combined),
                        ("inventory", inventory),
                        ("memory_signal", memory_signal),
                        ("rational_signal", rational_signal),
                        ("social_signal", social_signal),
                    ),
                    output,
                )
            )
        frozen = tuple(steps)
        return BaselineExecution(tuple(step.output for step in frozen), frozen)

    def run(self, inputs: BaselineInputBundle) -> tuple[float, ...]:
        return self.execute(inputs).draws


@dataclass(frozen=True, slots=True)
class BlockedBaselineAdapter:
    """Explicit placeholder for a registered comparator with no implementation."""

    spec: BaselineSpec
    failure_code: str

    def __post_init__(self) -> None:
        if not isinstance(self.spec, BaselineSpec):
            _fail("blocked baseline requires a BaselineSpec")
        _identifier(self.failure_code, "failure_code")

    def run(self, inputs: BaselineInputBundle) -> tuple[float, ...]:
        del inputs
        raise BaselineEvaluationError(self.failure_code)


@dataclass(frozen=True, slots=True)
class BaselineSuiteResult:
    """All registered terminal rows plus a non-renormalizing fairness decision."""

    input_bundle_sha256: str
    results: tuple[BaselineResult, ...]
    comparable: bool
    missing_or_failed_required: tuple[str, ...]

    def __post_init__(self) -> None:
        _sha256(self.input_bundle_sha256, "input_bundle_sha256")
        if not isinstance(self.results, tuple) or not self.results:
            _fail("baseline suite must retain every terminal result")
        if not all(isinstance(result, BaselineResult) for result in self.results):
            _fail("baseline suite contains an invalid result")
        identifiers = tuple(result.baseline_id for result in self.results)
        if tuple(sorted(identifiers)) != identifiers or len(set(identifiers)) != len(
            identifiers
        ):
            _fail("baseline suite results must be unique and canonically ordered")
        if not isinstance(self.comparable, bool):
            _fail("comparable must be boolean")
        if not isinstance(self.missing_or_failed_required, tuple) or not all(
            isinstance(value, str) and _IDENTIFIER.fullmatch(value) is not None
            for value in self.missing_or_failed_required
        ):
            _fail("missing_or_failed_required must be an identifier tuple")
        if tuple(sorted(self.missing_or_failed_required)) != (
            self.missing_or_failed_required
        ) or len(set(self.missing_or_failed_required)) != len(
            self.missing_or_failed_required
        ):
            _fail("missing_or_failed_required must be unique and ordered")
        if self.comparable != (not self.missing_or_failed_required):
            _fail("comparable flag disagrees with required terminal rows")
        if any(
            result.input_bundle_sha256 != self.input_bundle_sha256
            for result in self.results
        ):
            _fail("every baseline result must bind the common input bundle")
        terminal_failures = tuple(
            result.baseline_id
            for result in self.results
            if result.required and result.status is not BaselineStatus.COMPLETED
        )
        if self.missing_or_failed_required != terminal_failures:
            _fail("required failure list disagrees with terminal rows")

    @property
    def sha256(self) -> str:
        """Return the complete suite commitment, including failures."""
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def require_comparable(self) -> None:
        """Fail before a comparison table if a required row did not complete."""
        if not self.comparable:
            _fail(
                "required baselines are blocked or failed: "
                f"{self.missing_or_failed_required!r}"
            )

    def to_mapping(self) -> dict[str, object]:
        """Return all rows without converting failures to missing values."""
        return {
            "comparable": self.comparable,
            "input_bundle_sha256": self.input_bundle_sha256,
            "missing_or_failed_required": list(self.missing_or_failed_required),
            "results": [result.to_mapping() for result in self.results],
            "schema_version": BASELINE_SUITE_SCHEMA_VERSION,
        }


def run_baseline_suite(
    *,
    inputs: BaselineInputBundle,
    registered_specs: tuple[BaselineSpec, ...],
    adapters: Mapping[str, BaselineAdapter],
) -> BaselineSuiteResult:
    """Run every registered row on the same bundle and preserve all failures."""
    if not isinstance(inputs, BaselineInputBundle):
        _fail("inputs must be a BaselineInputBundle")
    if not isinstance(registered_specs, tuple) or not registered_specs:
        _fail("registered_specs must be a non-empty immutable tuple")
    if not all(isinstance(spec, BaselineSpec) for spec in registered_specs):
        _fail("registered_specs contains an invalid specification")
    if not isinstance(adapters, Mapping) or not all(
        isinstance(name, str) for name in adapters
    ):
        _fail("adapters must be a string-keyed mapping")
    ordered_specs = tuple(sorted(registered_specs, key=lambda item: item.baseline_id))
    if len({spec.baseline_id for spec in ordered_specs}) != len(ordered_specs):
        _fail("registered baseline IDs must be unique")
    unknown_adapters = sorted(
        set(adapters) - {spec.baseline_id for spec in ordered_specs}
    )
    if unknown_adapters:
        _fail(f"unregistered baseline adapters were supplied: {unknown_adapters!r}")
    results: list[BaselineResult] = []
    missing_required: list[str] = []
    event_values = tuple(key.value for key in inputs.event_keys)
    for spec in ordered_specs:
        adapter = adapters.get(spec.baseline_id)
        receipt: BaselineMechanismReceipt | None = None
        if adapter is None:
            status = BaselineStatus.BLOCKED
            failure_code = "ADAPTER_NOT_IMPLEMENTED"
            draws: tuple[float, ...] = ()
        else:
            try:
                adapter_spec = adapter.spec
                if not isinstance(adapter_spec, BaselineSpec):
                    raise BaselineEvaluationError("ADAPTER_SPEC_INVALID")
                if adapter_spec.sha256 != spec.sha256:
                    raise BaselineEvaluationError("ADAPTER_SPEC_MISMATCH")
                if isinstance(adapter, BlockedBaselineAdapter):
                    status = BaselineStatus.BLOCKED
                    failure_code = adapter.failure_code
                    draws = ()
                else:
                    if not isinstance(adapter, AuditedBaselineAdapter):
                        raise TypeError("adapter lacks audited execution interface")
                    execution, receipt = execute_baseline_adapter(adapter, inputs)
                    draws = execution.draws
                    if len(draws) != len(inputs.event_keys):
                        raise BaselineEvaluationError("DRAW_COUNT_MISMATCH")
                    draws = tuple(_finite(value, "baseline draw") for value in draws)
            except Exception as error:
                status = BaselineStatus.FAILED
                failure_code = (
                    str(error)
                    if isinstance(error, BaselineEvaluationError)
                    and _IDENTIFIER.fullmatch(str(error)) is not None
                    else "ADAPTER_EXECUTION_FAILED"
                )
                draws = ()
                receipt = None
            else:
                if not isinstance(adapter, BlockedBaselineAdapter):
                    status = BaselineStatus.COMPLETED
                    failure_code = None
        if spec.required and status is not BaselineStatus.COMPLETED:
            missing_required.append(spec.baseline_id)
        results.append(
            BaselineResult(
                baseline_id=spec.baseline_id,
                kind=spec.kind,
                required=spec.required,
                status=status,
                spec_sha256=spec.sha256,
                input_bundle_sha256=inputs.sha256,
                event_keys=(event_values if status is BaselineStatus.COMPLETED else ()),
                draws=draws,
                failure_code=failure_code,
                synthetic_only=inputs.synthetic_only,
                mechanism_receipt=receipt,
            )
        )
    return BaselineSuiteResult(
        input_bundle_sha256=inputs.sha256,
        results=tuple(results),
        comparable=not missing_required,
        missing_or_failed_required=tuple(missing_required),
    )


def completed_baseline_means(
    result: BaselineSuiteResult,
) -> tuple[tuple[str, float], ...]:
    """Return means only after every required baseline completed."""
    result.require_comparable()
    return tuple(
        (row.baseline_id, fmean(row.draws))
        for row in result.results
        if row.status is BaselineStatus.COMPLETED
    )


def validate_research_baseline_registry(
    registered_specs: tuple[BaselineSpec, ...],
) -> None:
    """Require one mandatory row for every preregistered comparator family."""
    if not isinstance(registered_specs, tuple) or not all(
        isinstance(spec, BaselineSpec) for spec in registered_specs
    ):
        _fail("research baseline registry must be an immutable spec tuple")
    counts = {
        kind: sum(spec.kind is kind for spec in registered_specs)
        for kind in RESEARCH_BASELINE_KINDS
    }
    missing = sorted(kind.value for kind, count in counts.items() if count == 0)
    duplicates = sorted(kind.value for kind, count in counts.items() if count > 1)
    optional = sorted(
        spec.baseline_id for spec in registered_specs if not spec.required
    )
    if missing or duplicates or optional:
        _fail(
            "research baseline registry is incomplete: "
            f"missing={missing!r}, duplicates={duplicates!r}, optional={optional!r}"
        )


def run_research_baseline_suite(
    *,
    inputs: BaselineInputBundle,
    registered_specs: tuple[BaselineSpec, ...],
    adapters: Mapping[str, BaselineAdapter],
) -> BaselineSuiteResult:
    """Run the complete preregistered comparator set without silent omission."""
    validate_research_baseline_registry(registered_specs)
    return run_baseline_suite(
        inputs=inputs,
        registered_specs=registered_specs,
        adapters=adapters,
    )


__all__ = [
    "BASELINE_MECHANISM_CLAIM_SCOPE",
    "BASELINE_MECHANISM_RECEIPT_SCHEMA_VERSION",
    "BASELINE_SUITE_SCHEMA_VERSION",
    "AuditedBaselineAdapter",
    "BaselineAdapter",
    "BaselineExecution",
    "BaselineEvaluationError",
    "BaselineInputBundle",
    "BaselineKind",
    "BaselineMechanismDescriptor",
    "BaselineMechanismReceipt",
    "BaselineMechanismStep",
    "BaselineResult",
    "BaselineSpec",
    "BaselineStatus",
    "BaselineSuiteResult",
    "BlockedBaselineAdapter",
    "ClassicCalibratedABMBaseline",
    "DeterministicMockLLMABMBaseline",
    "FrozenLinearTransferBaseline",
    "FullPolicyTwinSyntheticBaseline",
    "HistoricalBootstrapBaseline",
    "OrderLevelGeneratorBaseline",
    "PointCalibratedABMBaseline",
    "RESEARCH_BASELINE_KINDS",
    "ZeroEffectBaseline",
    "audit_baseline_execution",
    "completed_baseline_means",
    "execute_baseline_adapter",
    "run_baseline_suite",
    "run_research_baseline_suite",
    "validate_research_baseline_registry",
]
