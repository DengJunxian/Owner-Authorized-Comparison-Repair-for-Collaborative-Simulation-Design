"""Fail-closed paired baseline/candidate counterfactual orchestration."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from statistics import fmean, variance
from typing import Literal, NoReturn, Protocol, Self, cast

from policytwin.counterfactual.events import (
    ExogenousShock,
    ExogenousShockTape,
    ScenarioGenerationReceipt,
    SemanticEventError,
    ShockAccessCapability,
    ShockConsumption,
    ShockConsumptionLedger,
    ShockTransitionConfig,
    ShockTransitionState,
    _finalize_shock_access,
    _issue_shock_access,
    validate_exact_exogenous_shock_tape,
    validate_exact_scenario_generation_receipt,
)
from policytwin.provenance import JsonlEventWriter, canonical_json_bytes, sha256_bytes

PAIR_SCHEMA_VERSION = "policytwin.counterfactual-pair.v5"
PAIR_CLAIM_SCOPE: Literal["synthetic_pair_summary_arithmetic_only"] = (
    "synthetic_pair_summary_arithmetic_only"
)
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

type PolicyScalar = str | int | float | bool | None


class PairValidationError(ValueError):
    """Raised when paired-run scientific inputs are malformed."""


class PairIntegrityError(RuntimeError):
    """Raised when two world results do not satisfy the pairing contract."""

    def __init__(self, message: str, audit: PairIntegrityAudit) -> None:
        super().__init__(message)
        self.audit = audit


class PairExecutionError(PairIntegrityError):
    """Raised when a world runner fails before producing an auditable result."""


def _fail(message: str) -> NoReturn:
    raise PairValidationError(message)


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        _fail(f"{field} must be a stable ASCII identifier")
    return value


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail(f"{field} must be a lowercase SHA-256 digest")
    return value


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _fail(f"{field} must be a positive integer")
    return value


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        _fail(f"{field} must be a finite real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        _fail(f"{field} must be finite")
    return normalized


def _policy_scalar(value: object, field: str) -> PolicyScalar:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    _fail(f"{field} must be a finite JSON scalar")


class WorldRole(StrEnum):
    """The two members of an atomic policy comparison."""

    BASELINE = "baseline"
    CANDIDATE = "candidate"


class WorldStatus(StrEnum):
    """Explicit world completion status."""

    COMPLETED = "completed"
    FAILED = "failed"


class PairAuditRule(StrEnum):
    """Stable integrity rules emitted for every pair."""

    WORLD_ROLES = "WORLD_ROLES"
    WORLD_COMPLETION = "WORLD_COMPLETION"
    SCENARIO_COMMITMENT = "SCENARIO_COMMITMENT"
    INITIAL_STATE_COMMITMENT = "INITIAL_STATE_COMMITMENT"
    POSTERIOR_DRAW_COMMITMENT = "POSTERIOR_DRAW_COMMITMENT"
    PARAMETER_COMMITMENT = "PARAMETER_COMMITMENT"
    POLICY_COMMITMENT = "POLICY_COMMITMENT"
    SHOCK_TAPE_COMMITMENT = "SHOCK_TAPE_COMMITMENT"
    EXOGENOUS_REALIZATIONS = "EXOGENOUS_REALIZATIONS"
    SHOCK_ACCESS_VIOLATIONS = "SHOCK_ACCESS_VIOLATIONS"
    SHOCK_TRANSITION_COMPLETENESS = "SHOCK_TRANSITION_COMPLETENESS"
    SHOCK_TRANSITION_CONFIG = "SHOCK_TRANSITION_CONFIG"
    SHOCK_TRANSITION_REALIZATIONS = "SHOCK_TRANSITION_REALIZATIONS"
    SHOCK_TRANSITION_REPLAY = "SHOCK_TRANSITION_REPLAY"
    SCENARIO_EVENT_RECEIPTS = "SCENARIO_EVENT_RECEIPTS"
    SHOCK_STATE_BINDING = "SHOCK_STATE_BINDING"
    RUNNER_SHOCK_DECLARATION = "RUNNER_SHOCK_DECLARATION"
    HORIZON_CELLS = "HORIZON_CELLS"
    GROUP_CELLS = "GROUP_CELLS"
    GROUP_MEMBERSHIP_COMMITMENT = "GROUP_MEMBERSHIP_COMMITMENT"
    METRIC_CELLS = "METRIC_CELLS"
    NOOP_DIFFERENCE = "NOOP_DIFFERENCE"
    BASELINE_INVARIANCE = "BASELINE_INVARIANCE"
    RUNNER_COMPLETED = "RUNNER_COMPLETED"


class RandomnessDesign(StrEnum):
    """Randomness couplings whose difference variance may be compared."""

    INDEPENDENT_SEEDS = "independent_seeds"
    SAME_ORDER_SEED = "same_order_seed"
    EVENT_KEYED = "event_keyed"


@dataclass(frozen=True, slots=True)
class HorizonSpec:
    """One preregistered outcome horizon on a shared trajectory."""

    horizon_id: str
    end_step: int

    def __post_init__(self) -> None:
        _identifier(self.horizon_id, "horizon_id")
        _positive_int(self.end_step, "end_step")

    def to_mapping(self) -> dict[str, object]:
        return {"end_step": self.end_step, "horizon_id": self.horizon_id}


@dataclass(frozen=True, slots=True)
class GroupSpec:
    """One pre-policy fixed cohort and its membership commitment."""

    group_id: str
    cohort_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.group_id, "group_id")
        _sha256(self.cohort_sha256, "cohort_sha256")

    @classmethod
    def from_members(cls, group_id: str, members: tuple[str, ...]) -> Self:
        """Commit to a non-empty immutable pre-policy membership list."""
        if not isinstance(members, tuple) or not members:
            _fail("group members must be a non-empty immutable tuple")
        normalized = tuple(
            sorted(_identifier(item, "group member") for item in members)
        )
        if len(set(normalized)) != len(normalized):
            _fail("group members must be unique")
        return cls(
            group_id=group_id,
            cohort_sha256=sha256_bytes(canonical_json_bytes(list(normalized))),
        )

    def to_mapping(self) -> dict[str, object]:
        return {"cohort_sha256": self.cohort_sha256, "group_id": self.group_id}


def _group_commitment_sha256(groups: tuple[GroupSpec, ...]) -> str:
    """Commit to group identifiers and fixed pre-policy membership hashes."""
    return sha256_bytes(
        canonical_json_bytes(
            [
                item.to_mapping()
                for item in sorted(groups, key=lambda item: item.group_id)
            ]
        )
    )


@dataclass(frozen=True, slots=True)
class PosteriorDraw:
    """One posterior sample and the parameter vector used in both worlds."""

    draw_id: str
    model_draw_sha256: str
    parameters: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        _identifier(self.draw_id, "draw_id")
        _sha256(self.model_draw_sha256, "model_draw_sha256")
        if not isinstance(self.parameters, tuple) or not self.parameters:
            _fail("parameters must be a non-empty immutable tuple")
        normalized: list[tuple[str, float]] = []
        for item in self.parameters:
            if not isinstance(item, tuple) or len(item) != 2:
                _fail("each parameter must be a name-value tuple")
            name, value = item
            normalized.append(
                (_identifier(name, "parameter name"), _finite(value, name))
            )
        normalized.sort()
        if len({name for name, _ in normalized}) != len(normalized):
            _fail("parameter names must be unique")
        object.__setattr__(self, "parameters", tuple(normalized))

    @property
    def parameter_sha256(self) -> str:
        """Return the exact parameter-vector commitment."""
        return sha256_bytes(
            canonical_json_bytes(
                [{"name": name, "value": value} for name, value in self.parameters]
            )
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "draw_id": self.draw_id,
            "model_draw_sha256": self.model_draw_sha256,
            "parameter_sha256": self.parameter_sha256,
            "parameters": [
                {"name": name, "value": value} for name, value in self.parameters
            ],
        }


@dataclass(frozen=True, slots=True)
class StateArtifact:
    """Immutable canonical JSON opening state shared by both worlds."""

    canonical_json: str

    def __post_init__(self) -> None:
        if not isinstance(self.canonical_json, str) or not self.canonical_json:
            _fail("canonical_json must be a non-empty string")
        try:
            parsed = json.loads(self.canonical_json)
        except json.JSONDecodeError as error:
            raise PairValidationError("canonical_json is invalid JSON") from error
        canonical = canonical_json_bytes(parsed).decode("utf-8")
        if canonical != self.canonical_json:
            _fail("opening state JSON must use canonical serialization")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        """Freeze a JSON-compatible state mapping into canonical bytes."""
        return cls(canonical_json=canonical_json_bytes(dict(value)).decode("utf-8"))

    @property
    def sha256(self) -> str:
        return sha256_bytes(self.canonical_json.encode("utf-8"))

    def to_mapping(self) -> dict[str, object]:
        value = json.loads(self.canonical_json)
        if not isinstance(value, dict):
            _fail("opening state must decode to an object")
        return cast(dict[str, object], value)


@dataclass(frozen=True, slots=True)
class PolicyDefinition:
    """Immutable policy configuration; the only allowed treatment difference."""

    policy_id: str
    config: tuple[tuple[str, PolicyScalar], ...]
    is_noop: bool = False

    def __post_init__(self) -> None:
        _identifier(self.policy_id, "policy_id")
        if not isinstance(self.config, tuple):
            _fail("policy config must be an immutable tuple")
        normalized: list[tuple[str, PolicyScalar]] = []
        for item in self.config:
            if not isinstance(item, tuple) or len(item) != 2:
                _fail("each policy config item must be a name-value tuple")
            name, value = item
            normalized.append(
                (_identifier(name, "policy config name"), _policy_scalar(value, name))
            )
        normalized.sort()
        if len({name for name, _ in normalized}) != len(normalized):
            _fail("policy config names must be unique")
        object.__setattr__(self, "config", tuple(normalized))
        if not isinstance(self.is_noop, bool):
            _fail("is_noop must be a boolean")

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def to_mapping(self) -> dict[str, object]:
        return {
            "config": {name: value for name, value in self.config},
            "is_noop": self.is_noop,
            "policy_id": self.policy_id,
        }


@dataclass(frozen=True, slots=True)
class WorldDefinition:
    """One role and its immutable policy."""

    role: WorldRole
    policy: PolicyDefinition

    def __post_init__(self) -> None:
        if not isinstance(self.role, WorldRole):
            _fail("role must be a WorldRole")
        if not isinstance(self.policy, PolicyDefinition):
            _fail("policy must be a PolicyDefinition")


@dataclass(frozen=True, slots=True)
class MetricValue:
    """One finite scalar outcome metric."""

    metric_name: str
    value: float

    def __post_init__(self) -> None:
        _identifier(self.metric_name, "metric_name")
        object.__setattr__(self, "value", _finite(self.value, "metric value"))

    def to_mapping(self) -> dict[str, object]:
        return {"metric_name": self.metric_name, "value": self.value}


@dataclass(frozen=True, slots=True)
class GroupOutcome:
    """All metrics for one fixed cohort at one horizon."""

    group_id: str
    metrics: tuple[MetricValue, ...]

    def __post_init__(self) -> None:
        _identifier(self.group_id, "group_id")
        if not isinstance(self.metrics, tuple) or not self.metrics:
            _fail("group metrics must be a non-empty immutable tuple")
        ordered = tuple(sorted(self.metrics, key=lambda item: item.metric_name))
        if not all(isinstance(item, MetricValue) for item in ordered):
            _fail("group metrics must contain MetricValue items")
        if len({item.metric_name for item in ordered}) != len(ordered):
            _fail("metric names must be unique within a group cell")
        object.__setattr__(self, "metrics", ordered)

    def to_mapping(self) -> dict[str, object]:
        return {
            "group_id": self.group_id,
            "metrics": [metric.to_mapping() for metric in self.metrics],
        }


@dataclass(frozen=True, slots=True)
class HorizonOutcome:
    """All fixed-group outcomes computed from one trajectory prefix."""

    horizon_id: str
    end_step: int
    groups: tuple[GroupOutcome, ...]

    def __post_init__(self) -> None:
        _identifier(self.horizon_id, "horizon_id")
        _positive_int(self.end_step, "end_step")
        if not isinstance(self.groups, tuple) or not self.groups:
            _fail("horizon groups must be a non-empty immutable tuple")
        ordered = tuple(sorted(self.groups, key=lambda item: item.group_id))
        if not all(isinstance(item, GroupOutcome) for item in ordered):
            _fail("horizon groups must contain GroupOutcome items")
        if len({item.group_id for item in ordered}) != len(ordered):
            _fail("group ids must be unique within a horizon")
        object.__setattr__(self, "groups", ordered)

    def to_mapping(self) -> dict[str, object]:
        return {
            "end_step": self.end_step,
            "groups": [group.to_mapping() for group in self.groups],
            "horizon_id": self.horizon_id,
        }


@dataclass(frozen=True, slots=True)
class WorldFailure:
    """Machine-readable world failure retained without result selection."""

    rule_code: str
    exception_type: str

    def __post_init__(self) -> None:
        _identifier(self.rule_code, "failure rule_code")
        _identifier(self.exception_type, "failure exception_type")

    def to_mapping(self) -> dict[str, object]:
        return {
            "exception_type": self.exception_type,
            "rule_code": self.rule_code,
        }


@dataclass(frozen=True, slots=True)
class WorldRunRequest:
    """Complete immutable scientific inputs supplied to one isolated world."""

    scenario_id: str
    world: WorldDefinition
    initial_state: StateArtifact
    posterior_draw: PosteriorDraw
    shock_access: ShockAccessCapability
    horizons: tuple[HorizonSpec, ...]
    groups: tuple[GroupSpec, ...]

    def __post_init__(self) -> None:
        _identifier(self.scenario_id, "scenario_id")
        if not isinstance(self.world, WorldDefinition):
            _fail("world request requires a WorldDefinition")
        if not isinstance(self.initial_state, StateArtifact):
            _fail("world request requires a StateArtifact")
        if not isinstance(self.posterior_draw, PosteriorDraw):
            _fail("world request requires a PosteriorDraw")
        if not isinstance(self.shock_access, ShockAccessCapability):
            _fail("world request requires an engine-issued shock capability")
        if self.shock_access.scenario_id != self.scenario_id:
            _fail("shock capability scenario does not match the world request")
        if (
            not isinstance(self.horizons, tuple)
            or not self.horizons
            or not all(isinstance(item, HorizonSpec) for item in self.horizons)
        ):
            _fail("world request requires immutable horizon specifications")
        if (
            not isinstance(self.groups, tuple)
            or not self.groups
            or not all(isinstance(item, GroupSpec) for item in self.groups)
        ):
            _fail("world request requires immutable group specifications")

    @property
    def shock_tape(self) -> ShockAccessCapability:
        """Compatibility view that exposes keys and consuming lookup, not entries."""
        return self.shock_access

    @property
    def group_commitment_sha256(self) -> str:
        """Return the fixed pre-policy group membership commitment."""
        return _group_commitment_sha256(self.groups)


@dataclass(frozen=True, slots=True)
class WorldOutcome:
    """One world's result plus every commitment needed for pair auditing."""

    scenario_id: str
    role: WorldRole
    status: WorldStatus
    horizons: tuple[HorizonOutcome, ...]
    exogenous_shocks: tuple[ExogenousShock, ...]
    shock_tape_sha256: str
    initial_state_sha256: str
    model_draw_sha256: str
    parameter_sha256: str
    group_commitment_sha256: str
    policy_sha256: str
    final_state: StateArtifact
    final_state_sha256: str
    internal_event_count: int
    failure: WorldFailure | None = None
    shock_consumption_ledger: ShockConsumptionLedger | None = None

    def __post_init__(self) -> None:
        _identifier(self.scenario_id, "scenario_id")
        if not isinstance(self.role, WorldRole):
            _fail("world outcome role must be a WorldRole")
        if not isinstance(self.status, WorldStatus):
            _fail("world outcome status must be a WorldStatus")
        for field in (
            "shock_tape_sha256",
            "initial_state_sha256",
            "model_draw_sha256",
            "parameter_sha256",
            "group_commitment_sha256",
            "policy_sha256",
            "final_state_sha256",
        ):
            _sha256(getattr(self, field), field)
        if not isinstance(self.final_state, StateArtifact):
            _fail("world outcome final_state must be a StateArtifact")
        if self.final_state.sha256 != self.final_state_sha256:
            _fail("world outcome final-state hash does not match its artifact")
        if (
            isinstance(self.internal_event_count, bool)
            or not isinstance(self.internal_event_count, int)
            or self.internal_event_count < 0
        ):
            _fail("internal_event_count must be a non-negative integer")
        ordered_horizons = tuple(
            sorted(self.horizons, key=lambda item: (item.end_step, item.horizon_id))
        )
        if not all(isinstance(item, HorizonOutcome) for item in ordered_horizons):
            _fail("horizons must contain HorizonOutcome items")
        if len({item.horizon_id for item in ordered_horizons}) != len(ordered_horizons):
            _fail("world outcome horizon ids must be unique")
        object.__setattr__(self, "horizons", ordered_horizons)
        ordered_shocks = tuple(
            sorted(self.exogenous_shocks, key=lambda item: item.event_key.value)
        )
        if not all(isinstance(item, ExogenousShock) for item in ordered_shocks):
            _fail("exogenous_shocks must contain ExogenousShock items")
        if len({item.event_key.value for item in ordered_shocks}) != len(
            ordered_shocks
        ):
            _fail("world outcome exogenous event keys must be unique")
        object.__setattr__(self, "exogenous_shocks", ordered_shocks)
        if self.status is WorldStatus.COMPLETED and self.failure is not None:
            _fail("completed world must not carry a failure")
        if self.status is WorldStatus.FAILED and not isinstance(
            self.failure, WorldFailure
        ):
            _fail("failed world must carry a WorldFailure")
        if self.shock_consumption_ledger is not None:
            if not isinstance(self.shock_consumption_ledger, ShockConsumptionLedger):
                _fail("shock_consumption_ledger has the wrong type")
            if self.shock_consumption_ledger.world_id != self.role.value:
                _fail("shock consumption ledger is bound to the wrong world role")
            if self.shock_consumption_ledger.role != self.role.value:
                _fail("shock transition receipts are bound to the wrong world role")
            if self.shock_consumption_ledger.scenario_id != self.scenario_id:
                _fail("shock consumption ledger is bound to the wrong scenario")
            if (
                self.shock_consumption_ledger.shock_tape_sha256
                != self.shock_tape_sha256
            ):
                _fail("shock consumption ledger is bound to the wrong tape")

    def to_mapping(self) -> dict[str, object]:
        return {
            "exogenous_shocks": [item.to_mapping() for item in self.exogenous_shocks],
            "failure": None if self.failure is None else self.failure.to_mapping(),
            "final_state": self.final_state.to_mapping(),
            "final_state_sha256": self.final_state_sha256,
            "horizons": [item.to_mapping() for item in self.horizons],
            "group_commitment_sha256": self.group_commitment_sha256,
            "initial_state_sha256": self.initial_state_sha256,
            "internal_event_count": self.internal_event_count,
            "model_draw_sha256": self.model_draw_sha256,
            "parameter_sha256": self.parameter_sha256,
            "policy_sha256": self.policy_sha256,
            "role": self.role.value,
            "scenario_id": self.scenario_id,
            "shock_tape_sha256": self.shock_tape_sha256,
            "shock_consumption_ledger": (
                None
                if self.shock_consumption_ledger is None
                else self.shock_consumption_ledger.to_mapping()
            ),
            "shock_consumption_ledger_sha256": (
                None
                if self.shock_consumption_ledger is None
                else self.shock_consumption_ledger.sha256
            ),
            "status": self.status.value,
        }

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))


class WorldRunner(Protocol):
    """Isolated deterministic simulator for one world."""

    def run(self, request: WorldRunRequest) -> WorldOutcome:
        """Execute one world from the supplied immutable inputs."""
        ...


class WorldRunnerFactory(Protocol):
    """Create isolated runners so worlds never share mutable state."""

    def create(self, world: WorldDefinition) -> WorldRunner:
        """Return a fresh runner for exactly one world definition."""
        ...


class _WorldExecutionFailure(Exception):
    def __init__(
        self,
        role: WorldRole,
        original: Exception,
        ledger: ShockConsumptionLedger | None,
    ) -> None:
        super().__init__(type(original).__qualname__)
        self.role = role
        self.original = original
        self.ledger = ledger


@dataclass(frozen=True, slots=True)
class PairAuditRecord:
    """One fixed-field machine-readable pair-integrity decision."""

    rule: PairAuditRule
    passed: bool
    expected: str | None
    baseline: str | None
    candidate: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.rule, PairAuditRule):
            _fail("pair audit rule must be a PairAuditRule")
        if not isinstance(self.passed, bool):
            _fail("pair audit passed must be a boolean")
        for field in ("expected", "baseline", "candidate"):
            value = getattr(self, field)
            if value is not None and not isinstance(value, str):
                _fail(f"pair audit {field} must be a string or null")

    def to_mapping(self) -> dict[str, object]:
        return {
            "baseline": self.baseline,
            "candidate": self.candidate,
            "expected": self.expected,
            "passed": self.passed,
            "rule": self.rule.value,
        }


@dataclass(frozen=True, slots=True)
class PairIntegrityAudit:
    """Complete, non-bypassable pairing audit."""

    records: tuple[PairAuditRecord, ...]
    pair_input: PolicyWorldPair | None = None
    baseline_world: WorldOutcome | None = None
    candidate_world: WorldOutcome | None = None
    baseline_invariance_inputs: tuple[PairedOutcome, PairedOutcome] | None = None
    schema_version: str = PAIR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PAIR_SCHEMA_VERSION:
            _fail("unsupported pair integrity audit schema")
        if not isinstance(self.records, tuple) or not self.records:
            _fail("pair integrity audit records must be a non-empty tuple")
        if not all(isinstance(item, PairAuditRecord) for item in self.records):
            _fail("pair integrity audit records have the wrong type")
        if len({item.rule for item in self.records}) != len(self.records):
            _fail("pair integrity audit rules must be unique")

        complete_evidence = (
            self.pair_input,
            self.baseline_world,
            self.candidate_world,
        )
        if all(item is not None for item in complete_evidence):
            if self.baseline_invariance_inputs is not None:
                _fail("pair audit cannot mix pair and invariance evidence")
            pair = cast(PolicyWorldPair, self.pair_input)
            baseline = cast(WorldOutcome, self.baseline_world)
            candidate = cast(WorldOutcome, self.candidate_world)
            expected = _derive_pair_records(pair, baseline, candidate)
            if self.records != expected:
                _fail("pair integrity records do not reproduce from typed evidence")
            return
        if any(item is not None for item in complete_evidence):
            _fail("pair integrity audit requires complete typed pair/world evidence")

        if self.baseline_invariance_inputs is not None:
            inputs = self.baseline_invariance_inputs
            if (
                not isinstance(inputs, tuple)
                or len(inputs) != 2
                or not all(isinstance(item, PairedOutcome) for item in inputs)
            ):
                _fail("baseline invariance audit requires two typed paired outcomes")
            expected = (_baseline_invariance_record(inputs[0], inputs[1]),)
            if self.records != expected:
                _fail("baseline invariance record does not reproduce from evidence")
            return

        allowed_failure_rules = {
            PairAuditRule.RUNNER_COMPLETED,
            PairAuditRule.SHOCK_ACCESS_VIOLATIONS,
        }
        if (
            PairAuditRule.RUNNER_COMPLETED not in {item.rule for item in self.records}
            or any(item.rule not in allowed_failure_rules for item in self.records)
            or any(item.passed for item in self.records)
        ):
            _fail("audit without typed evidence may describe only runner failure")

    @property
    def passed(self) -> bool:
        return bool(self.records) and all(record.passed for record in self.records)

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def to_mapping(self) -> dict[str, object]:
        if self.pair_input is not None:
            evidence: dict[str, object] = {
                "audit_kind": "completed_pair",
                "baseline_world_sha256": cast(WorldOutcome, self.baseline_world).sha256,
                "candidate_world_sha256": cast(
                    WorldOutcome, self.candidate_world
                ).sha256,
                "pair_input_sha256": self.pair_input.sha256,
            }
        elif self.baseline_invariance_inputs is not None:
            evidence = {
                "audit_kind": "baseline_invariance",
                "first_outcome_sha256": self.baseline_invariance_inputs[0].sha256,
                "second_outcome_sha256": self.baseline_invariance_inputs[1].sha256,
            }
        else:
            evidence = {"audit_kind": "runner_failure"}
        return {
            "evidence": evidence,
            "passed": self.passed,
            "records": [record.to_mapping() for record in self.records],
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
        *,
        pair_input: PolicyWorldPair,
        baseline_world: WorldOutcome,
        candidate_world: WorldOutcome,
    ) -> Self:
        """Rebuild an audit from typed evidence and reject altered serialized data."""
        expected = cls(
            records=_derive_pair_records(pair_input, baseline_world, candidate_world),
            pair_input=pair_input,
            baseline_world=baseline_world,
            candidate_world=candidate_world,
        )
        if canonical_json_bytes(dict(value)) != canonical_json_bytes(
            expected.to_mapping()
        ):
            _fail("serialized pair integrity audit does not reproduce from evidence")
        return expected

    def event_mappings(self) -> tuple[dict[str, object], ...]:
        """Return deterministic structured events suitable for JSONL artifacts."""
        return tuple(
            {
                **record.to_mapping(),
                "audit_sha256": self.sha256,
                "event_type": "pair_integrity_audit",
                "schema_version": self.schema_version,
                "sequence": sequence,
            }
            for sequence, record in enumerate(self.records)
        )

    def write_jsonl(self, path: Path) -> None:
        """Exclusively persist every integrity decision as deterministic JSONL."""
        with JsonlEventWriter(path) as writer:
            for event in self.event_mappings():
                writer.write(event)

    def require_passed(self) -> None:
        """Raise with this audit if any integrity rule failed."""
        if not self.passed:
            raise PairIntegrityError("paired-run integrity audit failed", self)

    @classmethod
    def baseline_invariance(cls, first: PairedOutcome, second: PairedOutcome) -> Self:
        """Audit that changing only the candidate left baseline bytes unchanged."""
        return cls(
            records=(_baseline_invariance_record(first, second),),
            baseline_invariance_inputs=(first, second),
        )


def _baseline_invariance_record(
    first: PairedOutcome, second: PairedOutcome
) -> PairAuditRecord:
    same_inputs = (
        first.scenario_id == second.scenario_id
        and first.draw_id == second.draw_id
        and first.initial_state_sha256 == second.initial_state_sha256
        and first.model_draw_sha256 == second.model_draw_sha256
        and first.parameter_sha256 == second.parameter_sha256
        and first.shock_tape_sha256 == second.shock_tape_sha256
        and first.group_commitment_sha256 == second.group_commitment_sha256
        and first.baseline_policy_sha256 == second.baseline_policy_sha256
    )
    return PairAuditRecord(
        rule=PairAuditRule.BASELINE_INVARIANCE,
        passed=same_inputs and first.baseline.sha256 == second.baseline.sha256,
        expected=first.baseline.sha256,
        baseline=second.baseline.sha256,
        candidate=None,
    )


def _record(
    rule: PairAuditRule,
    expected: object,
    baseline: object,
    candidate: object,
) -> PairAuditRecord:
    rendered_expected = str(expected)
    rendered_baseline = str(baseline)
    rendered_candidate = str(candidate)
    return PairAuditRecord(
        rule=rule,
        passed=baseline == expected and candidate == expected,
        expected=rendered_expected,
        baseline=rendered_baseline,
        candidate=rendered_candidate,
    )


def _outcome_cells(outcome: WorldOutcome) -> dict[tuple[str, str, str], float]:
    cells: dict[tuple[str, str, str], float] = {}
    for horizon in outcome.horizons:
        for group in horizon.groups:
            for metric in group.metrics:
                cells[(horizon.horizon_id, group.group_id, metric.metric_name)] = (
                    metric.value
                )
    return cells


def _ledger_realizations(outcome: WorldOutcome) -> tuple[dict[str, object], ...]:
    ledger = outcome.shock_consumption_ledger
    if ledger is None:
        return ()
    return tuple(item.to_mapping() for item in ledger.realizations)


def _ledger_violations(outcome: WorldOutcome) -> tuple[dict[str, object], ...] | str:
    ledger = outcome.shock_consumption_ledger
    if ledger is None:
        return "missing_engine_ledger"
    return tuple(item.to_mapping() for item in ledger.violations)


def _ledger_transition_complete(outcome: WorldOutcome) -> bool | str:
    ledger = outcome.shock_consumption_ledger
    if ledger is None:
        return "missing_engine_ledger"
    return ledger.complete


def _ledger_transition_config(outcome: WorldOutcome) -> str:
    ledger = outcome.shock_consumption_ledger
    if ledger is None:
        return "missing_engine_ledger"
    return ledger.transition_config.sha256


def _ledger_transition_realizations(
    outcome: WorldOutcome,
) -> tuple[tuple[str, str], ...] | str:
    ledger = outcome.shock_consumption_ledger
    if ledger is None:
        return "missing_engine_ledger"
    return tuple(
        (item.event_key.value, item.realization_sha256)
        for item in ledger.transition_receipts
    )


def _ledger_transition_replay(outcome: WorldOutcome) -> bool | str:
    ledger = outcome.shock_consumption_ledger
    if ledger is None:
        return "missing_engine_ledger"
    return ledger.verify_transitions()


def _ledger_scenario_event_receipts(
    outcome: WorldOutcome,
) -> tuple[dict[str, object], ...] | str:
    """Return finalized future-event commitments, never runner-local state."""
    ledger = outcome.shock_consumption_ledger
    if ledger is None:
        return "missing_engine_ledger"
    return tuple(item.to_mapping() for item in ledger.scenario_event_receipts)


def _scenario_event_receipts_match(
    baseline: WorldOutcome,
    candidate: WorldOutcome,
) -> bool:
    """Require valid, exactly equal external-information payload commitments."""
    baseline_ledger = baseline.shock_consumption_ledger
    candidate_ledger = candidate.shock_consumption_ledger
    if baseline_ledger is None or candidate_ledger is None:
        return False
    return (
        baseline_ledger.verify_scenario_events()
        and candidate_ledger.verify_scenario_events()
        and _ledger_scenario_event_receipts(baseline)
        == _ledger_scenario_event_receipts(candidate)
    )


def _world_state_matches_transition(
    outcome: WorldOutcome,
    config: ShockTransitionConfig,
) -> bool:
    """Verify the typed final world artifact installed the engine's post-state."""
    ledger = outcome.shock_consumption_ledger
    if ledger is None or not ledger.verify_transitions():
        return False
    state_mapping = outcome.final_state.to_mapping()
    raw_value = state_mapping.get(config.state_field)
    try:
        value = _finite(raw_value, config.state_field)
    except PairValidationError:
        return False
    installed = ShockTransitionState.from_mapping({config.state_field: value})
    return installed == ledger.final_transition_state


def _runner_declaration_matches_ledger(outcome: WorldOutcome) -> bool:
    """Accept no legacy declaration, or require it to match engine evidence."""
    if not outcome.exogenous_shocks:
        return True
    return tuple(item.to_mapping() for item in outcome.exogenous_shocks) == (
        _ledger_realizations(outcome)
    )


def _derive_pair_records(
    pair: PolicyWorldPair,
    baseline: WorldOutcome,
    candidate: WorldOutcome,
) -> tuple[PairAuditRecord, ...]:
    expected_horizons = tuple(
        (item.horizon_id, item.end_step) for item in pair.horizons
    )
    baseline_horizons = tuple(
        (item.horizon_id, item.end_step) for item in baseline.horizons
    )
    candidate_horizons = tuple(
        (item.horizon_id, item.end_step) for item in candidate.horizons
    )
    group_ids = tuple(item.group_id for item in pair.groups)
    expected_groups = tuple(
        (horizon.horizon_id, group_ids) for horizon in pair.horizons
    )
    baseline_groups = tuple(
        (horizon.horizon_id, tuple(group.group_id for group in horizon.groups))
        for horizon in baseline.horizons
    )
    candidate_groups = tuple(
        (horizon.horizon_id, tuple(group.group_id for group in horizon.groups))
        for horizon in candidate.horizons
    )
    expected_shocks = tuple(item.to_mapping() for item in pair.shock_tape.entries)
    expected_transition_realizations = tuple(
        (item.event_key.value, ShockConsumption(item).realization_sha256)
        for item in sorted(
            pair.shock_tape.entries,
            key=lambda item: (item.event_key.timestamp, item.event_key.value),
        )
    )
    group_commitment = pair.group_commitment_sha256
    baseline_cells = _outcome_cells(baseline)
    candidate_cells = _outcome_cells(candidate)
    records = [
        _record(
            PairAuditRule.WORLD_ROLES,
            (WorldRole.BASELINE, WorldRole.CANDIDATE),
            (baseline.role, candidate.role),
            (baseline.role, candidate.role),
        ),
        _record(
            PairAuditRule.WORLD_COMPLETION,
            WorldStatus.COMPLETED,
            baseline.status,
            candidate.status,
        ),
        _record(
            PairAuditRule.SCENARIO_COMMITMENT,
            pair.scenario_id,
            baseline.scenario_id,
            candidate.scenario_id,
        ),
        _record(
            PairAuditRule.INITIAL_STATE_COMMITMENT,
            pair.initial_state.sha256,
            baseline.initial_state_sha256,
            candidate.initial_state_sha256,
        ),
        _record(
            PairAuditRule.POSTERIOR_DRAW_COMMITMENT,
            pair.posterior_draw.model_draw_sha256,
            baseline.model_draw_sha256,
            candidate.model_draw_sha256,
        ),
        _record(
            PairAuditRule.PARAMETER_COMMITMENT,
            pair.posterior_draw.parameter_sha256,
            baseline.parameter_sha256,
            candidate.parameter_sha256,
        ),
        PairAuditRecord(
            rule=PairAuditRule.POLICY_COMMITMENT,
            passed=(
                baseline.policy_sha256 == pair.baseline_policy.sha256
                and candidate.policy_sha256 == pair.candidate_policy.sha256
            ),
            expected=f"{pair.baseline_policy.sha256}|{pair.candidate_policy.sha256}",
            baseline=baseline.policy_sha256,
            candidate=candidate.policy_sha256,
        ),
        _record(
            PairAuditRule.SHOCK_TAPE_COMMITMENT,
            pair.shock_tape.sha256,
            baseline.shock_tape_sha256,
            candidate.shock_tape_sha256,
        ),
        _record(
            PairAuditRule.EXOGENOUS_REALIZATIONS,
            expected_shocks,
            _ledger_realizations(baseline),
            _ledger_realizations(candidate),
        ),
        _record(
            PairAuditRule.SHOCK_ACCESS_VIOLATIONS,
            (),
            _ledger_violations(baseline),
            _ledger_violations(candidate),
        ),
        _record(
            PairAuditRule.SHOCK_TRANSITION_COMPLETENESS,
            True,
            _ledger_transition_complete(baseline),
            _ledger_transition_complete(candidate),
        ),
        _record(
            PairAuditRule.SHOCK_TRANSITION_CONFIG,
            pair.shock_transition_config.sha256,
            _ledger_transition_config(baseline),
            _ledger_transition_config(candidate),
        ),
        _record(
            PairAuditRule.SHOCK_TRANSITION_REALIZATIONS,
            expected_transition_realizations,
            _ledger_transition_realizations(baseline),
            _ledger_transition_realizations(candidate),
        ),
        _record(
            PairAuditRule.SHOCK_TRANSITION_REPLAY,
            True,
            _ledger_transition_replay(baseline),
            _ledger_transition_replay(candidate),
        ),
        PairAuditRecord(
            rule=PairAuditRule.SCENARIO_EVENT_RECEIPTS,
            passed=_scenario_event_receipts_match(baseline, candidate),
            expected="valid_exactly_equal_finalized_receipts",
            baseline=str(_ledger_scenario_event_receipts(baseline)),
            candidate=str(_ledger_scenario_event_receipts(candidate)),
        ),
        _record(
            PairAuditRule.SHOCK_STATE_BINDING,
            True,
            _world_state_matches_transition(baseline, pair.shock_transition_config),
            _world_state_matches_transition(candidate, pair.shock_transition_config),
        ),
        PairAuditRecord(
            rule=PairAuditRule.RUNNER_SHOCK_DECLARATION,
            passed=(
                _runner_declaration_matches_ledger(baseline)
                and _runner_declaration_matches_ledger(candidate)
            ),
            expected="empty_or_equal_to_engine_ledger",
            baseline=str(
                tuple(item.to_mapping() for item in baseline.exogenous_shocks)
            ),
            candidate=str(
                tuple(item.to_mapping() for item in candidate.exogenous_shocks)
            ),
        ),
        _record(
            PairAuditRule.HORIZON_CELLS,
            expected_horizons,
            baseline_horizons,
            candidate_horizons,
        ),
        _record(
            PairAuditRule.GROUP_CELLS,
            expected_groups,
            baseline_groups,
            candidate_groups,
        ),
        _record(
            PairAuditRule.GROUP_MEMBERSHIP_COMMITMENT,
            group_commitment,
            baseline.group_commitment_sha256,
            candidate.group_commitment_sha256,
        ),
        PairAuditRecord(
            rule=PairAuditRule.METRIC_CELLS,
            passed=baseline_cells.keys() == candidate_cells.keys(),
            expected=str(sorted(baseline_cells)),
            baseline=str(sorted(baseline_cells)),
            candidate=str(sorted(candidate_cells)),
        ),
    ]
    noop_passed = True
    if (
        pair.candidate_policy.is_noop
        and baseline_cells.keys() == candidate_cells.keys()
    ):
        noop_passed = all(
            math.isclose(
                candidate_cells[cell],
                baseline_cells[cell],
                rel_tol=0.0,
                abs_tol=pair.numeric_tolerance,
            )
            for cell in baseline_cells
        )
    records.append(
        PairAuditRecord(
            rule=PairAuditRule.NOOP_DIFFERENCE,
            passed=noop_passed,
            expected=(
                f"abs_difference<={pair.numeric_tolerance}"
                if pair.candidate_policy.is_noop
                else "not_applicable"
            ),
            baseline=None,
            candidate=None,
        )
    )
    return tuple(records)


def _evaluate_pair(
    pair: PolicyWorldPair,
    baseline: WorldOutcome,
    candidate: WorldOutcome,
) -> PairIntegrityAudit:
    return PairIntegrityAudit(
        records=_derive_pair_records(pair, baseline, candidate),
        pair_input=pair,
        baseline_world=baseline,
        candidate_world=candidate,
    )


@dataclass(frozen=True, slots=True)
class PairedDifference:
    """Candidate-minus-baseline value for one posterior/horizon/group cell."""

    draw_id: str
    horizon_id: str
    group_id: str
    metric_name: str
    baseline_value: float
    candidate_value: float
    difference: float

    def __post_init__(self) -> None:
        for field in ("draw_id", "horizon_id", "group_id", "metric_name"):
            _identifier(getattr(self, field), field)
        baseline = _finite(self.baseline_value, "baseline_value")
        candidate = _finite(self.candidate_value, "candidate_value")
        difference = _finite(self.difference, "difference")
        if difference != candidate - baseline:
            _fail("paired difference must equal candidate minus baseline")

    def to_mapping(self) -> dict[str, object]:
        return {
            "baseline_value": self.baseline_value,
            "candidate_value": self.candidate_value,
            "difference": self.difference,
            "draw_id": self.draw_id,
            "group_id": self.group_id,
            "horizon_id": self.horizon_id,
            "metric_name": self.metric_name,
        }


def _derive_differences(
    pair: PolicyWorldPair,
    baseline: WorldOutcome,
    candidate: WorldOutcome,
) -> tuple[PairedDifference, ...]:
    baseline_cells = _outcome_cells(baseline)
    candidate_cells = _outcome_cells(candidate)
    if baseline_cells.keys() != candidate_cells.keys():
        _fail("cannot derive paired differences from mismatched metric cells")
    return tuple(
        PairedDifference(
            draw_id=pair.posterior_draw.draw_id,
            horizon_id=horizon_id,
            group_id=group_id,
            metric_name=metric_name,
            baseline_value=baseline_cells[cell],
            candidate_value=candidate_cells[cell],
            difference=candidate_cells[cell] - baseline_cells[cell],
        )
        for cell in sorted(baseline_cells)
        for horizon_id, group_id, metric_name in (cell,)
    )


@dataclass(frozen=True, slots=True)
class PairedOutcome:
    """Synthetic paired arithmetic, not standalone formal execution evidence."""

    scenario_id: str
    draw_id: str
    initial_state_sha256: str
    model_draw_sha256: str
    parameter_sha256: str
    shock_tape_sha256: str
    groups: tuple[GroupSpec, ...]
    group_commitment_sha256: str
    baseline_policy_sha256: str
    candidate_policy_sha256: str
    pair_input: PolicyWorldPair
    baseline: WorldOutcome
    candidate: WorldOutcome
    differences: tuple[PairedDifference, ...]
    integrity_audit: PairIntegrityAudit
    claim_scope: Literal["synthetic_pair_summary_arithmetic_only"] = PAIR_CLAIM_SCOPE
    formal_execution_evidence: Literal[False] = False
    synthetic_only: Literal[True] = True
    schema_version: str = PAIR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PAIR_SCHEMA_VERSION:
            _fail("unsupported paired outcome schema")
        if self.claim_scope != PAIR_CLAIM_SCOPE:
            _fail("paired outcome claim scope cannot authorize execution evidence")
        if self.formal_execution_evidence is not False:
            _fail("standalone paired outcome cannot be formal execution evidence")
        if self.synthetic_only is not True:
            _fail("paired outcome must remain synthetic-only")
        if not isinstance(self.pair_input, PolicyWorldPair):
            _fail("paired outcome requires typed pair input evidence")
        if not isinstance(self.baseline, WorldOutcome) or not isinstance(
            self.candidate, WorldOutcome
        ):
            _fail("paired outcome requires typed world evidence")

        pair = self.pair_input
        expected_scalars: tuple[tuple[str, object, object], ...] = (
            ("scenario_id", self.scenario_id, pair.scenario_id),
            ("draw_id", self.draw_id, pair.posterior_draw.draw_id),
            (
                "initial_state_sha256",
                self.initial_state_sha256,
                pair.initial_state.sha256,
            ),
            (
                "model_draw_sha256",
                self.model_draw_sha256,
                pair.posterior_draw.model_draw_sha256,
            ),
            (
                "parameter_sha256",
                self.parameter_sha256,
                pair.posterior_draw.parameter_sha256,
            ),
            ("shock_tape_sha256", self.shock_tape_sha256, pair.shock_tape.sha256),
            (
                "group_commitment_sha256",
                self.group_commitment_sha256,
                pair.group_commitment_sha256,
            ),
            (
                "baseline_policy_sha256",
                self.baseline_policy_sha256,
                pair.baseline_policy.sha256,
            ),
            (
                "candidate_policy_sha256",
                self.candidate_policy_sha256,
                pair.candidate_policy.sha256,
            ),
        )
        for field, observed, expected in expected_scalars:
            if observed != expected:
                _fail(f"paired outcome {field} does not reproduce from pair input")
        if self.groups != pair.groups:
            _fail(
                "paired outcome group membership commitment does not reproduce "
                "from pair input"
            )

        expected_audit = _evaluate_pair(pair, self.baseline, self.candidate)
        if self.integrity_audit != expected_audit:
            _fail("paired outcome audit does not reproduce from typed evidence")
        expected_audit.require_passed()

        expected_differences = _derive_differences(pair, self.baseline, self.candidate)
        if self.differences != expected_differences:
            _fail("paired differences do not reproduce from typed world evidence")

    @classmethod
    def from_worlds(
        cls,
        pair_input: PolicyWorldPair,
        baseline: WorldOutcome,
        candidate: WorldOutcome,
    ) -> Self:
        """Derive every outcome field from typed inputs and world evidence."""
        audit = _evaluate_pair(pair_input, baseline, candidate)
        audit.require_passed()
        return cls(
            scenario_id=pair_input.scenario_id,
            draw_id=pair_input.posterior_draw.draw_id,
            initial_state_sha256=pair_input.initial_state.sha256,
            model_draw_sha256=pair_input.posterior_draw.model_draw_sha256,
            parameter_sha256=pair_input.posterior_draw.parameter_sha256,
            shock_tape_sha256=pair_input.shock_tape.sha256,
            groups=pair_input.groups,
            group_commitment_sha256=pair_input.group_commitment_sha256,
            baseline_policy_sha256=pair_input.baseline_policy.sha256,
            candidate_policy_sha256=pair_input.candidate_policy.sha256,
            pair_input=pair_input,
            baseline=baseline,
            candidate=candidate,
            differences=_derive_differences(pair_input, baseline, candidate),
            integrity_audit=audit,
        )

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
        *,
        pair_input: PolicyWorldPair,
        baseline: WorldOutcome,
        candidate: WorldOutcome,
    ) -> Self:
        """Rebuild from typed evidence; serialized derived fields have no authority."""
        expected = cls.from_worlds(pair_input, baseline, candidate)
        if canonical_json_bytes(dict(value)) != expected.canonical_bytes:
            _fail("serialized paired outcome does not reproduce from typed evidence")
        return expected

    def to_mapping(self) -> dict[str, object]:
        return {
            "baseline": self.baseline.to_mapping(),
            "baseline_policy_sha256": self.baseline_policy_sha256,
            "candidate": self.candidate.to_mapping(),
            "candidate_policy_sha256": self.candidate_policy_sha256,
            "claim_scope": self.claim_scope,
            "differences": [item.to_mapping() for item in self.differences],
            "draw_id": self.draw_id,
            "formal_execution_evidence": self.formal_execution_evidence,
            "group_commitment_sha256": self.group_commitment_sha256,
            "groups": [item.to_mapping() for item in self.groups],
            "initial_state_sha256": self.initial_state_sha256,
            "integrity_audit": self.integrity_audit.to_mapping(),
            "model_draw_sha256": self.model_draw_sha256,
            "pair_input": self.pair_input.to_mapping(),
            "parameter_sha256": self.parameter_sha256,
            "scenario_id": self.scenario_id,
            "schema_version": self.schema_version,
            "shock_tape_sha256": self.shock_tape_sha256,
            "synthetic_only": self.synthetic_only,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_mapping())

    @property
    def sha256(self) -> str:
        return sha256_bytes(self.canonical_bytes)


@dataclass(frozen=True, slots=True)
class PolicyWorldPair:
    """Atomic comparison sharing every input except the policy definition."""

    scenario_id: str
    posterior_draw: PosteriorDraw
    initial_state: StateArtifact
    shock_tape: ExogenousShockTape
    shock_transition_config: ShockTransitionConfig
    baseline_policy: PolicyDefinition
    candidate_policy: PolicyDefinition
    horizons: tuple[HorizonSpec, ...]
    groups: tuple[GroupSpec, ...]
    scenario_generation_receipt: ScenarioGenerationReceipt | None = None
    numeric_tolerance: float = 0.0
    synthetic_only: Literal[True] = True

    def __post_init__(self) -> None:
        if type(self) is not PolicyWorldPair:
            _fail("formal paired inputs must use the exact PolicyWorldPair type")
        _identifier(self.scenario_id, "scenario_id")
        try:
            validate_exact_exogenous_shock_tape(self.shock_tape)
        except SemanticEventError as error:
            raise PairValidationError(
                "pair shock tape failed exact canonical validation"
            ) from error
        if self.shock_tape.scenario_id != self.scenario_id:
            _fail("shock tape scenario_id does not match the pair")
        receipt = self.scenario_generation_receipt
        if receipt is not None:
            if type(receipt) is not ScenarioGenerationReceipt:
                _fail("pair scenario generation receipt has the wrong type")
            try:
                validate_exact_scenario_generation_receipt(receipt)
            except SemanticEventError as error:
                raise PairValidationError(
                    "pair scenario generation receipt failed exact canonical validation"
                ) from error
            if receipt.scenario_id != self.scenario_id:
                _fail("scenario generation receipt is bound to another scenario")
            if receipt.shock_tape_sha256 != self.shock_tape.sha256:
                _fail("scenario generation receipt does not bind the pair shock tape")
        if type(self.shock_transition_config) is not ShockTransitionConfig:
            _fail("pair requires a frozen ShockTransitionConfig")
        initial_mapping = self.initial_state.to_mapping()
        if self.shock_transition_config.state_field not in initial_mapping:
            _fail("opening state lacks the configured shock-transition field")
        _finite(
            initial_mapping[self.shock_transition_config.state_field],
            self.shock_transition_config.state_field,
        )
        for shock in self.shock_tape.entries:
            try:
                shock.value(self.shock_transition_config.shock_field)
            except KeyError:
                _fail("shock tape lacks the configured transition field")
        ordered_horizons = tuple(
            sorted(self.horizons, key=lambda item: (item.end_step, item.horizon_id))
        )
        if not ordered_horizons or not all(
            isinstance(item, HorizonSpec) for item in ordered_horizons
        ):
            _fail("horizons must be a non-empty tuple of HorizonSpec")
        if len({item.horizon_id for item in ordered_horizons}) != len(ordered_horizons):
            _fail("horizon ids must be unique")
        object.__setattr__(self, "horizons", ordered_horizons)
        ordered_groups = tuple(sorted(self.groups, key=lambda item: item.group_id))
        if not ordered_groups or not all(
            isinstance(item, GroupSpec) for item in ordered_groups
        ):
            _fail("groups must be a non-empty tuple of GroupSpec")
        if len({item.group_id for item in ordered_groups}) != len(ordered_groups):
            _fail("group ids must be unique")
        object.__setattr__(self, "groups", ordered_groups)
        tolerance = _finite(self.numeric_tolerance, "numeric_tolerance")
        if tolerance < 0.0:
            _fail("numeric_tolerance must be non-negative")
        object.__setattr__(self, "numeric_tolerance", tolerance)
        if self.synthetic_only is not True:
            _fail("current paired engine accepts synthetic inputs only")

    def _definition(self, role: WorldRole) -> WorldDefinition:
        policy = (
            self.baseline_policy
            if role is WorldRole.BASELINE
            else self.candidate_policy
        )
        return WorldDefinition(role=role, policy=policy)

    @property
    def group_commitment_sha256(self) -> str:
        """Commit to all preregistered groups and their fixed members."""
        return _group_commitment_sha256(self.groups)

    def to_mapping(self) -> dict[str, object]:
        """Return every immutable scientific input used to derive a pair."""
        return {
            "baseline_policy": self.baseline_policy.to_mapping(),
            "candidate_policy": self.candidate_policy.to_mapping(),
            "groups": [item.to_mapping() for item in self.groups],
            "horizons": [item.to_mapping() for item in self.horizons],
            "initial_state": self.initial_state.to_mapping(),
            "numeric_tolerance": self.numeric_tolerance,
            "posterior_draw": self.posterior_draw.to_mapping(),
            "scenario_generation_receipt": (
                None
                if self.scenario_generation_receipt is None
                else self.scenario_generation_receipt.to_mapping()
            ),
            "scenario_id": self.scenario_id,
            "shock_tape": self.shock_tape.to_mapping(),
            "shock_transition_config": self.shock_transition_config.to_mapping(),
            "synthetic_only": self.synthetic_only,
        }

    @property
    def sha256(self) -> str:
        """Commit to all typed pair inputs."""
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    @property
    def initial_transition_state(self) -> ShockTransitionState:
        """Extract the frozen reducer state from the committed opening state."""
        initial_mapping = self.initial_state.to_mapping()
        value = _finite(
            initial_mapping[self.shock_transition_config.state_field],
            self.shock_transition_config.state_field,
        )
        return ShockTransitionState.from_mapping(
            {self.shock_transition_config.state_field: value}
        )

    def _request(
        self,
        world: WorldDefinition,
        shock_access: ShockAccessCapability,
    ) -> WorldRunRequest:
        return WorldRunRequest(
            scenario_id=self.scenario_id,
            world=world,
            initial_state=self.initial_state,
            posterior_draw=self.posterior_draw,
            shock_access=shock_access,
            horizons=self.horizons,
            groups=self.groups,
        )

    def run(
        self,
        factory: WorldRunnerFactory,
        *,
        parallel: bool = False,
        invocation_order: tuple[WorldRole, WorldRole] = (
            WorldRole.BASELINE,
            WorldRole.CANDIDATE,
        ),
    ) -> PairedOutcome:
        """Run isolated worlds and fail closed unless every pairing rule passes."""
        validate_exact_policy_world_pair(self)
        if set(invocation_order) != {WorldRole.BASELINE, WorldRole.CANDIDATE}:
            _fail("invocation_order must contain baseline and candidate exactly once")
        definitions = {role: self._definition(role) for role in invocation_order}
        shock_access = {
            role: _issue_shock_access(
                self.shock_tape,
                world_id=role.value,
                role=("baseline" if role is WorldRole.BASELINE else "candidate"),
                transition_config=self.shock_transition_config,
                initial_transition_state=self.initial_transition_state,
            )
            for role in invocation_order
        }

        def execution_failure(
            role: WorldRole,
            error: Exception,
            ledger: ShockConsumptionLedger | None = None,
        ) -> NoReturn:
            exception_type = type(error).__qualname__
            records = [
                PairAuditRecord(
                    rule=PairAuditRule.RUNNER_COMPLETED,
                    passed=False,
                    expected="completed",
                    baseline=(exception_type if role is WorldRole.BASELINE else None),
                    candidate=(exception_type if role is WorldRole.CANDIDATE else None),
                )
            ]
            if ledger is not None:
                records.append(
                    PairAuditRecord(
                        rule=PairAuditRule.SHOCK_ACCESS_VIOLATIONS,
                        passed=False,
                        expected="runner_completed_before_ledger_audit",
                        baseline=(
                            str(ledger.to_mapping())
                            if role is WorldRole.BASELINE
                            else None
                        ),
                        candidate=(
                            str(ledger.to_mapping())
                            if role is WorldRole.CANDIDATE
                            else None
                        ),
                    )
                )
            audit = PairIntegrityAudit(records=tuple(records))
            raise PairExecutionError("paired world runner failed", audit) from error

        runners: dict[WorldRole, WorldRunner] = {}
        for role in invocation_order:
            try:
                runners[role] = factory.create(definitions[role])
            except Exception as error:
                execution_failure(role, error)

        def execute(role: WorldRole) -> WorldOutcome:
            capability = shock_access[role]
            try:
                outcome = runners[role].run(
                    self._request(definitions[role], capability)
                )
            except Exception as error:
                ledger = _finalize_shock_access(capability)
                raise _WorldExecutionFailure(role, error, ledger) from error
            ledger = _finalize_shock_access(capability)
            if outcome.shock_consumption_ledger is not None:
                authored_ledger_error = PairValidationError(
                    "world runner must not author a shock consumption ledger"
                )
                raise _WorldExecutionFailure(
                    role, authored_ledger_error, ledger
                ) from authored_ledger_error
            return replace(outcome, shock_consumption_ledger=ledger)

        try:
            if parallel:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = {
                        role: executor.submit(execute, role)
                        for role in invocation_order
                    }
                    outcomes = {
                        role: futures[role].result() for role in invocation_order
                    }
            else:
                outcomes = {role: execute(role) for role in invocation_order}
        except _WorldExecutionFailure as error:
            execution_failure(error.role, error.original, error.ledger)

        baseline = outcomes[WorldRole.BASELINE]
        candidate = outcomes[WorldRole.CANDIDATE]
        return PairedOutcome.from_worlds(self, baseline, candidate)


def validate_exact_policy_world_pair(pair: PolicyWorldPair) -> None:
    """Strictly reconstruct every formal pair input without polymorphic hooks."""
    if type(pair) is not PolicyWorldPair:
        _fail("paired execution requires the exact PolicyWorldPair type")
    exact_fields: tuple[tuple[object, type[object], str], ...] = (
        (pair.posterior_draw, PosteriorDraw, "PosteriorDraw"),
        (pair.initial_state, StateArtifact, "StateArtifact"),
        (pair.shock_tape, ExogenousShockTape, "ExogenousShockTape"),
        (pair.shock_transition_config, ShockTransitionConfig, "ShockTransitionConfig"),
        (pair.baseline_policy, PolicyDefinition, "baseline PolicyDefinition"),
        (pair.candidate_policy, PolicyDefinition, "candidate PolicyDefinition"),
    )
    for value, expected, label in exact_fields:
        if type(value) is not expected:
            _fail(f"paired execution requires the exact {label} type")
    if (
        type(pair.scenario_id) is not str
        or type(pair.numeric_tolerance) is not float
        or pair.synthetic_only is not True
        or type(pair.posterior_draw.draw_id) is not str
        or type(pair.posterior_draw.model_draw_sha256) is not str
        or type(pair.posterior_draw.parameters) is not tuple
        or type(pair.initial_state.canonical_json) is not str
        or type(pair.shock_transition_config.state_field) is not str
        or type(pair.shock_transition_config.shock_field) is not str
        or type(pair.shock_transition_config.scale) is not float
        or type(pair.shock_transition_config.implementation_id) is not str
    ):
        _fail("paired execution requires exact built-in scalar input values")
    if not all(
        type(item) is tuple
        and len(item) == 2
        and type(item[0]) is str
        and type(item[1]) is float
        for item in pair.posterior_draw.parameters
    ):
        _fail("paired execution requires exact posterior parameter values")
    for policy in (pair.baseline_policy, pair.candidate_policy):
        if (
            type(policy.policy_id) is not str
            or type(policy.config) is not tuple
            or type(policy.is_noop) is not bool
            or not all(
                type(item) is tuple
                and len(item) == 2
                and type(item[0]) is str
                and (item[1] is None or type(item[1]) in {str, int, float, bool})
                for item in policy.config
            )
        ):
            _fail("paired execution requires exact policy values")
    if type(pair.horizons) is not tuple or not all(
        type(item) is HorizonSpec
        and type(item.horizon_id) is str
        and type(item.end_step) is int
        for item in pair.horizons
    ):
        _fail("paired execution requires exact HorizonSpec tuple values")
    if type(pair.groups) is not tuple or not all(
        type(item) is GroupSpec
        and type(item.group_id) is str
        and type(item.cohort_sha256) is str
        for item in pair.groups
    ):
        _fail("paired execution requires exact GroupSpec tuple values")
    receipt = pair.scenario_generation_receipt
    if receipt is not None and type(receipt) is not ScenarioGenerationReceipt:
        _fail("paired execution requires the exact ScenarioGenerationReceipt type")
    try:
        validate_exact_exogenous_shock_tape(pair.shock_tape)
        if receipt is not None:
            validate_exact_scenario_generation_receipt(receipt)
        reconstructed = PolicyWorldPair(
            scenario_id=pair.scenario_id,
            posterior_draw=PosteriorDraw(
                draw_id=pair.posterior_draw.draw_id,
                model_draw_sha256=pair.posterior_draw.model_draw_sha256,
                parameters=pair.posterior_draw.parameters,
            ),
            initial_state=StateArtifact(
                canonical_json=pair.initial_state.canonical_json
            ),
            shock_tape=pair.shock_tape,
            shock_transition_config=ShockTransitionConfig(
                state_field=pair.shock_transition_config.state_field,
                shock_field=pair.shock_transition_config.shock_field,
                scale=pair.shock_transition_config.scale,
                implementation_id=pair.shock_transition_config.implementation_id,
            ),
            baseline_policy=PolicyDefinition(
                policy_id=pair.baseline_policy.policy_id,
                config=pair.baseline_policy.config,
                is_noop=pair.baseline_policy.is_noop,
            ),
            candidate_policy=PolicyDefinition(
                policy_id=pair.candidate_policy.policy_id,
                config=pair.candidate_policy.config,
                is_noop=pair.candidate_policy.is_noop,
            ),
            horizons=tuple(
                HorizonSpec(item.horizon_id, item.end_step) for item in pair.horizons
            ),
            groups=tuple(
                GroupSpec(item.group_id, item.cohort_sha256) for item in pair.groups
            ),
            scenario_generation_receipt=receipt,
            numeric_tolerance=pair.numeric_tolerance,
            synthetic_only=True,
        )
    except (PairValidationError, SemanticEventError) as error:
        raise PairValidationError(
            "paired execution input failed strict canonical reconstruction"
        ) from error
    if reconstructed != pair:
        _fail("paired execution input failed strict canonical reconstruction")


class PairFactoryProvider(Protocol):
    """Supply a scenario-isolated world factory for a pair."""

    def __call__(self, pair: PolicyWorldPair) -> WorldRunnerFactory: ...


def run_policy_world_pairs(
    pairs: Sequence[PolicyWorldPair],
    factory_provider: PairFactoryProvider,
    *,
    max_workers: int = 1,
    parallel_worlds: bool = False,
) -> tuple[PairedOutcome, ...]:
    """Run multiple posterior draws with canonical completion-order independence."""
    materialized = tuple(pairs)
    if not materialized:
        _fail("pairs must not be empty")
    _positive_int(max_workers, "max_workers")
    identities = tuple(
        (pair.scenario_id, pair.posterior_draw.draw_id) for pair in materialized
    )
    if len(set(identities)) != len(identities):
        _fail("pair scenario/draw identities must be unique")

    def execute(pair: PolicyWorldPair) -> PairedOutcome:
        return pair.run(factory_provider(pair), parallel=parallel_worlds)

    if max_workers == 1:
        results = tuple(execute(pair) for pair in materialized)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = tuple(executor.map(execute, materialized))
    return tuple(sorted(results, key=lambda item: (item.scenario_id, item.draw_id)))


@dataclass(frozen=True, slots=True)
class RandomnessVariance:
    """Sample variance of paired differences under one coupling design."""

    design: RandomnessDesign
    sample_size: int
    mean_difference: float
    variance: float

    def to_mapping(self) -> dict[str, object]:
        return {
            "design": self.design.value,
            "mean_difference": self.mean_difference,
            "sample_size": self.sample_size,
            "variance": self.variance,
        }


@dataclass(frozen=True, slots=True)
class RandomnessVarianceComparison:
    """Side-by-side diagnostic; only event-keyed runs satisfy pair integrity."""

    estimates: tuple[RandomnessVariance, ...]

    def for_design(self, design: RandomnessDesign) -> RandomnessVariance:
        for estimate in self.estimates:
            if estimate.design is design:
                return estimate
        raise KeyError(design)

    def to_mapping(self) -> dict[str, object]:
        return {
            "estimates": [item.to_mapping() for item in self.estimates],
            "schema_version": PAIR_SCHEMA_VERSION,
        }


def compare_randomness_designs(
    samples: Mapping[RandomnessDesign, Sequence[int | float]]
    | Mapping[str, Sequence[int | float]],
) -> RandomnessVarianceComparison:
    """Compare unbiased sample variance for three explicitly labeled designs."""
    normalized: dict[RandomnessDesign, tuple[float, ...]] = {}
    for raw_design, raw_values in samples.items():
        try:
            design = (
                raw_design
                if isinstance(raw_design, RandomnessDesign)
                else RandomnessDesign(raw_design)
            )
        except ValueError as error:
            raise PairValidationError("unknown randomness design") from error
        if design in normalized:
            _fail("randomness design is duplicated")
        values = tuple(_finite(value, "difference sample") for value in raw_values)
        if len(values) < 2:
            _fail("each randomness design requires at least two samples")
        normalized[design] = values
    if set(normalized) != set(RandomnessDesign):
        _fail("all three randomness designs are required")
    estimates = tuple(
        RandomnessVariance(
            design=design,
            sample_size=len(normalized[design]),
            mean_difference=fmean(normalized[design]),
            variance=variance(normalized[design]),
        )
        for design in RandomnessDesign
    )
    return RandomnessVarianceComparison(estimates=estimates)


__all__ = [
    "PAIR_CLAIM_SCOPE",
    "PAIR_SCHEMA_VERSION",
    "GroupOutcome",
    "GroupSpec",
    "HorizonOutcome",
    "HorizonSpec",
    "MetricValue",
    "PairAuditRecord",
    "PairAuditRule",
    "PairExecutionError",
    "PairFactoryProvider",
    "PairIntegrityAudit",
    "PairIntegrityError",
    "PairValidationError",
    "PairedDifference",
    "PairedOutcome",
    "PolicyDefinition",
    "PolicyWorldPair",
    "PosteriorDraw",
    "RandomnessDesign",
    "RandomnessVariance",
    "RandomnessVarianceComparison",
    "StateArtifact",
    "WorldDefinition",
    "WorldFailure",
    "WorldOutcome",
    "WorldRole",
    "WorldRunRequest",
    "WorldRunner",
    "WorldRunnerFactory",
    "WorldStatus",
    "compare_randomness_designs",
    "run_policy_world_pairs",
    "validate_exact_policy_world_pair",
]
