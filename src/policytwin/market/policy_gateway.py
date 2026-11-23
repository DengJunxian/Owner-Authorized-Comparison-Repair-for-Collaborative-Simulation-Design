"""Mandatory, hash-bound bridge between policy decisions and market mutation.

The gateway intentionally contains no SEC rule values.  It binds an already
validated policy object, one security, an exact logical-time clock, and any
event-specific evaluator context into a deterministic commitment.  The order
book is the only caller in production: a policy-required market cannot be
constructed without this bridge.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from threading import RLock
from typing import NoReturn, Self, cast

from policytwin.policies.sec_tick_size_pilot import (
    ComplianceAction,
    ComplianceDecision,
    ComplianceRequest,
    DocumentedExceptionEvaluator,
    PolicyComplianceAuditEvent,
    PolicySourceRegistry,
    SecTickSizePilotConfig,
    SecTickSizePilotPolicy,
    SecurityAssignmentManifest,
    SyntheticFixedEvaluator,
    TradeAtRestrictionEvaluator,
)
from policytwin.provenance import canonical_json_bytes, sha256_bytes

POLICY_GATEWAY_SCHEMA_VERSION = "policytwin.market-policy-gateway.v2"
SYNTHETIC_EVALUATOR_SCHEMA_VERSION = "policytwin.synthetic-evaluator.v1"
SYNTHETIC_EVALUATOR_REGISTRY_SCHEMA_VERSION = (
    "policytwin.synthetic-evaluator-registry.v1"
)

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class PolicyGatewayError(RuntimeError):
    """Raised when the frozen bridge cannot issue a trustworthy decision."""


def _fail(message: str) -> NoReturn:
    raise PolicyGatewayError(message)


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        _fail(f"{field} must be a stable ASCII identifier")
    return value


def _event_key(value: object) -> str:
    if not isinstance(value, str) or not value or len(value) > 220 or "\x00" in value:
        _fail("event_key must be bounded non-empty text")
    return value


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail(f"{field} must be a lowercase SHA-256 digest")
    return value


def _aware_utc(value: object, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        _fail(f"{field} must be a timezone-aware datetime")
    try:
        offset = value.utcoffset()
        converted = value.astimezone(UTC)
    except (OverflowError, ValueError) as error:
        raise PolicyGatewayError(f"{field} is outside the UTC range") from error
    if offset is None:
        _fail(f"{field} must be a timezone-aware datetime")
    return converted


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        _fail(f"{field} must be an object with string keys")
    return cast(Mapping[str, object], value)


def _sequence(value: object, field: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        _fail(f"{field} must be an array")
    return cast(Sequence[object], value)


def _strict_fields(
    value: Mapping[str, object], expected: frozenset[str], field: str
) -> None:
    actual = frozenset(value)
    if actual != expected:
        _fail(
            f"{field} fields differ; missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _parse_datetime(value: object, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        _fail(f"{field} must be an ISO 8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise PolicyGatewayError(f"{field} must be ISO 8601") from error
    return _aware_utc(parsed, field)


@dataclass(frozen=True, slots=True)
class PolicyClockPoint:
    """One exact logical-time to civil-time mapping."""

    logical_time: int
    timestamp: datetime

    def __post_init__(self) -> None:
        if isinstance(self.logical_time, bool) or not isinstance(
            self.logical_time, int
        ):
            _fail("logical_time must be an integer")
        object.__setattr__(
            self, "timestamp", _aware_utc(self.timestamp, "clock timestamp")
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "logical_time": self.logical_time,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(
            value, frozenset({"logical_time", "timestamp"}), "policy clock point"
        )
        logical_time = value["logical_time"]
        if isinstance(logical_time, bool) or not isinstance(logical_time, int):
            _fail("logical_time must be an integer")
        return cls(
            logical_time=logical_time,
            timestamp=_parse_datetime(value["timestamp"], "clock timestamp"),
        )


@dataclass(frozen=True, slots=True)
class FrozenPolicyClock:
    """Finite exact clock; interpolation and implicit wall time are forbidden."""

    points: tuple[PolicyClockPoint, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.points, tuple) or not all(
            type(point) is PolicyClockPoint for point in self.points
        ):
            _fail("clock points must be an immutable PolicyClockPoint tuple")
        ordered = tuple(sorted(self.points, key=lambda point: point.logical_time))
        if not ordered:
            _fail("frozen policy clock must contain at least one point")
        if len({point.logical_time for point in ordered}) != len(ordered):
            _fail("frozen policy clock logical times must be unique")
        object.__setattr__(self, "points", ordered)

    def resolve(self, logical_time: int) -> datetime:
        """Resolve exactly one configured logical time or fail closed."""
        if isinstance(logical_time, bool) or not isinstance(logical_time, int):
            _fail("policy logical_time must be an integer")
        for point in self.points:
            if point.logical_time == logical_time:
                return point.timestamp
        raise PolicyGatewayError(
            f"logical time {logical_time} is absent from the frozen policy clock"
        )

    def to_mapping(self) -> dict[str, object]:
        return {"points": [point.to_mapping() for point in self.points]}

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(value, frozenset({"points"}), "frozen policy clock")
        return cls(
            tuple(
                PolicyClockPoint.from_mapping(_mapping(item, "policy clock point"))
                for item in _sequence(value["points"], "clock points")
            )
        )

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))


@dataclass(frozen=True, slots=True)
class PolicyRequestBinding:
    """Frozen optional exception or trade-at context for one market event."""

    cause_event_key: str
    action: ComplianceAction
    fill_index: int | None = None
    exception_id: str | None = None
    trade_at_context_sha256: str | None = None

    def __post_init__(self) -> None:
        _event_key(self.cause_event_key)
        if not isinstance(self.action, ComplianceAction):
            _fail("request binding action must be ComplianceAction")
        if self.action is ComplianceAction.QUOTE:
            if self.fill_index is not None or self.trade_at_context_sha256 is not None:
                _fail("quote binding cannot contain fill or trade-at context")
        elif (
            isinstance(self.fill_index, bool)
            or not isinstance(self.fill_index, int)
            or self.fill_index < 0
        ):
            _fail("trade binding requires a non-negative fill_index")
        if self.exception_id is not None:
            _identifier(self.exception_id, "exception_id")
        if self.trade_at_context_sha256 is not None:
            _sha256(self.trade_at_context_sha256, "trade_at_context_sha256")

    @property
    def lookup_key(self) -> tuple[str, str, int | None]:
        return (self.cause_event_key, self.action.value, self.fill_index)

    def to_mapping(self) -> dict[str, object]:
        return {
            "action": self.action.value,
            "cause_event_key": self.cause_event_key,
            "exception_id": self.exception_id,
            "fill_index": self.fill_index,
            "trade_at_context_sha256": self.trade_at_context_sha256,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(
            value,
            frozenset(
                {
                    "action",
                    "cause_event_key",
                    "exception_id",
                    "fill_index",
                    "trade_at_context_sha256",
                }
            ),
            "policy request binding",
        )
        raw_action = value["action"]
        if not isinstance(raw_action, str):
            _fail("request binding action must be text")
        try:
            action = ComplianceAction(raw_action)
        except ValueError as error:
            raise PolicyGatewayError("unsupported request binding action") from error
        fill_index = value["fill_index"]
        if fill_index is not None and (
            isinstance(fill_index, bool) or not isinstance(fill_index, int)
        ):
            _fail("fill_index must be an integer or null")
        exception_id = value["exception_id"]
        if exception_id is not None and not isinstance(exception_id, str):
            _fail("exception_id must be text or null")
        trade_context = value["trade_at_context_sha256"]
        if trade_context is not None and not isinstance(trade_context, str):
            _fail("trade_at_context_sha256 must be text or null")
        return cls(
            cause_event_key=_event_key(value["cause_event_key"]),
            action=action,
            fill_index=fill_index,
            exception_id=exception_id,
            trade_at_context_sha256=trade_context,
        )


@dataclass(frozen=True, slots=True)
class PolicyGatewayConfig:
    """Immutable commitment binding the market to one policy and assignment."""

    gateway_id: str
    security_id: str
    clock: FrozenPolicyClock
    policy_sha256: str
    assignment_sha256: str
    assignment_source_sha256: str
    source_registry_sha256: str
    evaluator_registry_sha256: str
    synthetic_only: bool
    request_bindings: tuple[PolicyRequestBinding, ...] = ()
    schema_version: str = POLICY_GATEWAY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _identifier(self.gateway_id, "gateway_id")
        _identifier(self.security_id, "security_id")
        if type(self.clock) is not FrozenPolicyClock:
            _fail("gateway clock must be a FrozenPolicyClock")
        for field, value in (
            ("policy_sha256", self.policy_sha256),
            ("assignment_sha256", self.assignment_sha256),
            ("assignment_source_sha256", self.assignment_source_sha256),
            ("source_registry_sha256", self.source_registry_sha256),
            ("evaluator_registry_sha256", self.evaluator_registry_sha256),
        ):
            _sha256(value, field)
        if not isinstance(self.synthetic_only, bool):
            _fail("synthetic_only must be a boolean")
        if not isinstance(self.request_bindings, tuple) or not all(
            type(binding) is PolicyRequestBinding for binding in self.request_bindings
        ):
            _fail("request_bindings must be an immutable PolicyRequestBinding tuple")
        ordered = tuple(
            sorted(self.request_bindings, key=lambda binding: binding.lookup_key)
        )
        if len({binding.lookup_key for binding in ordered}) != len(ordered):
            _fail("policy request bindings must have unique lookup keys")
        object.__setattr__(self, "request_bindings", ordered)
        if self.schema_version != POLICY_GATEWAY_SCHEMA_VERSION:
            _fail("unsupported policy gateway schema version")

    @classmethod
    def bind(
        cls,
        *,
        gateway_id: str,
        security_id: str,
        clock: FrozenPolicyClock,
        policy: SecTickSizePilotPolicy,
        request_bindings: tuple[PolicyRequestBinding, ...] = (),
        exception_evaluator_descriptor: SyntheticEvaluatorDescriptor | None = None,
        trade_at_evaluator_descriptor: SyntheticEvaluatorDescriptor | None = None,
    ) -> PolicyGatewayConfig:
        """Build a commitment from one already validated policy object."""
        if type(policy) is not SecTickSizePilotPolicy:
            _fail("gateway binding requires an exact SecTickSizePilotPolicy")
        exception_descriptor, trade_at_descriptor = _normalize_evaluator_descriptors(
            exception_evaluator_descriptor, trade_at_evaluator_descriptor
        )
        return cls(
            gateway_id=gateway_id,
            security_id=security_id,
            clock=clock,
            policy_sha256=policy.config.sha256,
            assignment_sha256=policy.assignments.manifest.sha256,
            assignment_source_sha256=policy.assignments.source_sha256,
            source_registry_sha256=policy.config.source_registry.sha256,
            evaluator_registry_sha256=_evaluator_registry_sha256(
                exception_descriptor, trade_at_descriptor
            ),
            synthetic_only=policy.config.synthetic_only,
            request_bindings=request_bindings,
        )

    def binding_for(
        self,
        *,
        cause_event_key: str,
        action: ComplianceAction,
        fill_index: int | None,
    ) -> PolicyRequestBinding | None:
        lookup = (_event_key(cause_event_key), action.value, fill_index)
        for binding in self.request_bindings:
            if binding.lookup_key == lookup:
                return binding
        return None

    def to_mapping(self) -> dict[str, object]:
        return {
            "assignment_sha256": self.assignment_sha256,
            "assignment_source_sha256": self.assignment_source_sha256,
            "clock": self.clock.to_mapping(),
            "evaluator_registry_sha256": self.evaluator_registry_sha256,
            "gateway_id": self.gateway_id,
            "policy_sha256": self.policy_sha256,
            "request_bindings": [
                binding.to_mapping() for binding in self.request_bindings
            ],
            "schema_version": self.schema_version,
            "security_id": self.security_id,
            "source_registry_sha256": self.source_registry_sha256,
            "synthetic_only": self.synthetic_only,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        """Parse the exact hash-bound gateway reconstruction schema."""
        _strict_fields(
            value,
            frozenset(
                {
                    "assignment_sha256",
                    "assignment_source_sha256",
                    "clock",
                    "evaluator_registry_sha256",
                    "gateway_id",
                    "policy_sha256",
                    "request_bindings",
                    "schema_version",
                    "security_id",
                    "source_registry_sha256",
                    "synthetic_only",
                }
            ),
            "policy gateway config",
        )
        synthetic_only = value["synthetic_only"]
        if not isinstance(synthetic_only, bool):
            _fail("synthetic_only must be a boolean")
        return cls(
            gateway_id=_identifier(value["gateway_id"], "gateway_id"),
            security_id=_identifier(value["security_id"], "security_id"),
            clock=FrozenPolicyClock.from_mapping(_mapping(value["clock"], "clock")),
            policy_sha256=_sha256(value["policy_sha256"], "policy_sha256"),
            assignment_sha256=_sha256(value["assignment_sha256"], "assignment_sha256"),
            assignment_source_sha256=_sha256(
                value["assignment_source_sha256"], "assignment_source_sha256"
            ),
            source_registry_sha256=_sha256(
                value["source_registry_sha256"], "source_registry_sha256"
            ),
            evaluator_registry_sha256=_sha256(
                value["evaluator_registry_sha256"], "evaluator_registry_sha256"
            ),
            synthetic_only=synthetic_only,
            request_bindings=tuple(
                PolicyRequestBinding.from_mapping(
                    _mapping(item, "policy request binding")
                )
                for item in _sequence(value["request_bindings"], "request_bindings")
            ),
            schema_version=(
                value["schema_version"]
                if isinstance(value["schema_version"], str)
                else "invalid"
            ),
        )

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))


@dataclass(frozen=True, slots=True)
class GatewayComplianceRecord:
    """One policy decision cross-bound to its policy audit event and market cause."""

    cause_event_key: str
    fill_index: int | None
    gateway_sha256: str
    decision: ComplianceDecision
    policy_audit_event: PolicyComplianceAuditEvent

    def __post_init__(self) -> None:
        _event_key(self.cause_event_key)
        _sha256(self.gateway_sha256, "gateway_sha256")
        if type(self.decision) is not ComplianceDecision:
            _fail("gateway record requires an exact ComplianceDecision")
        if type(self.policy_audit_event) is not PolicyComplianceAuditEvent:
            _fail("gateway record requires an exact policy audit event")
        if self.policy_audit_event.decision != self.decision:
            _fail("gateway decision differs from its policy audit event")
        if self.decision.action is ComplianceAction.QUOTE:
            if self.fill_index is not None:
                _fail("quote gateway record cannot have a fill index")
            expected_request_key = self.cause_event_key
        else:
            if (
                isinstance(self.fill_index, bool)
                or not isinstance(self.fill_index, int)
                or self.fill_index < 0
            ):
                _fail("trade gateway record requires a fill index")
            expected_request_key = f"{self.cause_event_key}:trade:{self.fill_index}"
        if self.decision.request.event_key != expected_request_key:
            _fail("policy request key is not derived from the market event key")

    def to_mapping(self) -> dict[str, object]:
        return {
            "cause_event_key": self.cause_event_key,
            "decision_sha256": self.decision.sha256,
            "fill_index": self.fill_index,
            "gateway_sha256": self.gateway_sha256,
            "policy_audit_event_sha256": self.policy_audit_event.event_sha256,
            "request_sha256": self.decision.request.sha256,
        }

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))


class SyntheticEvaluatorRole(StrEnum):
    """Closed evaluator roles supported by persisted synthetic replay."""

    DOCUMENTED_EXCEPTION = "documented_exception"
    TRADE_AT = "trade_at"


class SyntheticEvaluatorMode(StrEnum):
    """Closed deterministic evaluator implementations; no imports are accepted."""

    NOT_CONFIGURED = "not_configured"
    FIXED_RECEIPT = "fixed_receipt"


@dataclass(frozen=True, slots=True)
class SyntheticEvaluatorDecision:
    """One allowlisted synthetic evaluator identity and fixed decision."""

    evaluator_id: str
    allowed: bool

    def __post_init__(self) -> None:
        _identifier(self.evaluator_id, "synthetic evaluator_id")
        if not isinstance(self.allowed, bool):
            _fail("synthetic evaluator decision must be a boolean")

    def to_mapping(self) -> dict[str, object]:
        return {"allowed": self.allowed, "evaluator_id": self.evaluator_id}

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(
            value,
            frozenset({"allowed", "evaluator_id"}),
            "synthetic evaluator decision",
        )
        allowed = value["allowed"]
        if not isinstance(allowed, bool):
            _fail("synthetic evaluator decision must be a boolean")
        return cls(
            evaluator_id=_identifier(value["evaluator_id"], "evaluator_id"),
            allowed=allowed,
        )


@dataclass(frozen=True, slots=True)
class SyntheticEvaluatorDescriptor:
    """Portable allowlist descriptor for deterministic synthetic receipts."""

    role: SyntheticEvaluatorRole
    mode: SyntheticEvaluatorMode
    decisions: tuple[SyntheticEvaluatorDecision, ...] = ()
    proof_domain: str | None = None
    schema_version: str = SYNTHETIC_EVALUATOR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.role, SyntheticEvaluatorRole):
            _fail("synthetic evaluator role is invalid")
        if not isinstance(self.mode, SyntheticEvaluatorMode):
            _fail("synthetic evaluator mode is invalid")
        if not isinstance(self.decisions, tuple) or not all(
            type(item) is SyntheticEvaluatorDecision for item in self.decisions
        ):
            _fail("synthetic evaluator decisions must be an immutable typed tuple")
        ordered = tuple(sorted(self.decisions, key=lambda item: item.evaluator_id))
        if len({item.evaluator_id for item in ordered}) != len(ordered):
            _fail("synthetic evaluator ids must be unique")
        object.__setattr__(self, "decisions", ordered)
        if self.schema_version != SYNTHETIC_EVALUATOR_SCHEMA_VERSION:
            _fail("unsupported synthetic evaluator schema version")
        if self.mode is SyntheticEvaluatorMode.NOT_CONFIGURED:
            if ordered or self.proof_domain is not None:
                _fail("not-configured evaluator cannot carry decisions or proof domain")
        else:
            if not ordered:
                _fail("fixed synthetic evaluator requires at least one decision")
            _identifier(self.proof_domain, "proof_domain")

    @classmethod
    def not_configured(
        cls, role: SyntheticEvaluatorRole
    ) -> SyntheticEvaluatorDescriptor:
        return cls(role=role, mode=SyntheticEvaluatorMode.NOT_CONFIGURED)

    @classmethod
    def fixed(
        cls,
        role: SyntheticEvaluatorRole,
        *,
        decisions: tuple[SyntheticEvaluatorDecision, ...],
        proof_domain: str,
    ) -> SyntheticEvaluatorDescriptor:
        return cls(
            role=role,
            mode=SyntheticEvaluatorMode.FIXED_RECEIPT,
            decisions=decisions,
            proof_domain=proof_domain,
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "decisions": [item.to_mapping() for item in self.decisions],
            "mode": self.mode.value,
            "proof_domain": self.proof_domain,
            "role": self.role.value,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(
            value,
            frozenset({"decisions", "mode", "proof_domain", "role", "schema_version"}),
            "synthetic evaluator descriptor",
        )
        raw_role = value["role"]
        raw_mode = value["mode"]
        if not isinstance(raw_role, str) or not isinstance(raw_mode, str):
            _fail("synthetic evaluator role and mode must be text")
        try:
            role = SyntheticEvaluatorRole(raw_role)
            mode = SyntheticEvaluatorMode(raw_mode)
        except (TypeError, ValueError) as error:
            raise PolicyGatewayError(
                "synthetic evaluator descriptor is outside the allowlist"
            ) from error
        proof_domain = value["proof_domain"]
        if proof_domain is not None and not isinstance(proof_domain, str):
            _fail("proof_domain must be text or null")
        return cls(
            role=role,
            mode=mode,
            decisions=tuple(
                SyntheticEvaluatorDecision.from_mapping(
                    _mapping(item, "synthetic evaluator decision")
                )
                for item in _sequence(value["decisions"], "decisions")
            ),
            proof_domain=proof_domain,
            schema_version=(
                value["schema_version"]
                if isinstance(value["schema_version"], str)
                else "invalid"
            ),
        )

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))


def _normalize_evaluator_descriptors(
    exception_descriptor: SyntheticEvaluatorDescriptor | None,
    trade_at_descriptor: SyntheticEvaluatorDescriptor | None,
) -> tuple[SyntheticEvaluatorDescriptor, SyntheticEvaluatorDescriptor]:
    normalized_exception = (
        SyntheticEvaluatorDescriptor.not_configured(
            SyntheticEvaluatorRole.DOCUMENTED_EXCEPTION
        )
        if exception_descriptor is None
        else exception_descriptor
    )
    normalized_trade_at = (
        SyntheticEvaluatorDescriptor.not_configured(SyntheticEvaluatorRole.TRADE_AT)
        if trade_at_descriptor is None
        else trade_at_descriptor
    )
    if (
        type(normalized_exception) is not SyntheticEvaluatorDescriptor
        or normalized_exception.role is not SyntheticEvaluatorRole.DOCUMENTED_EXCEPTION
        or type(normalized_trade_at) is not SyntheticEvaluatorDescriptor
        or normalized_trade_at.role is not SyntheticEvaluatorRole.TRADE_AT
    ):
        _fail("synthetic evaluator descriptors have the wrong role")
    return normalized_exception, normalized_trade_at


def _evaluator_registry_sha256(
    exception_descriptor: SyntheticEvaluatorDescriptor,
    trade_at_descriptor: SyntheticEvaluatorDescriptor,
) -> str:
    """Commit both exact synthetic evaluator semantics as one execution input."""
    exception, trade_at = _normalize_evaluator_descriptors(
        exception_descriptor, trade_at_descriptor
    )
    return sha256_bytes(
        canonical_json_bytes(
            {
                "claim_scope": "synthetic_evaluator_execution_commitment_only",
                "exception_evaluator": exception.to_mapping(),
                "schema_version": SYNTHETIC_EVALUATOR_REGISTRY_SCHEMA_VERSION,
                "trade_at_evaluator": trade_at.to_mapping(),
            }
        )
    )


@dataclass(frozen=True, slots=True)
class SyntheticGatewayReplayMaterial:
    """Immutable gateway material captured before a replay artifact is sealed."""

    gateway_config: PolicyGatewayConfig
    policy_config: SecTickSizePilotConfig
    source_registry: PolicySourceRegistry
    assignment_manifest: SecurityAssignmentManifest
    assignment_source_bytes: bytes
    exception_evaluator: SyntheticEvaluatorDescriptor
    trade_at_evaluator: SyntheticEvaluatorDescriptor


class SecTickSizePolicyGateway:
    """Exclusive runtime gateway used by a policy-required order book."""

    def __init__(
        self,
        config: PolicyGatewayConfig,
        policy: SecTickSizePilotPolicy,
        *,
        exception_evaluator: DocumentedExceptionEvaluator | None = None,
        trade_at_evaluator: TradeAtRestrictionEvaluator | None = None,
        exception_evaluator_descriptor: SyntheticEvaluatorDescriptor | None = None,
        trade_at_evaluator_descriptor: SyntheticEvaluatorDescriptor | None = None,
    ) -> None:
        if type(config) is not PolicyGatewayConfig:
            _fail("gateway requires an exact PolicyGatewayConfig")
        if type(policy) is not SecTickSizePilotPolicy:
            _fail("gateway requires an exact SecTickSizePilotPolicy")
        if not config.synthetic_only or not policy.config.synthetic_only:
            _fail(
                "real SEC policy execution is blocked until primary-source-bound "
                "exception and trade-at evaluators are implemented"
            )
        if policy.audit_log.events:
            _fail("gateway requires an unused, exclusive policy audit log")
        if exception_evaluator is not None or trade_at_evaluator is not None:
            _fail(
                "caller-provided evaluators are not replayable; use the closed "
                "synthetic evaluator descriptor allowlist"
            )
        exception_descriptor, trade_at_descriptor = _normalize_evaluator_descriptors(
            exception_evaluator_descriptor, trade_at_evaluator_descriptor
        )
        self._validate_evaluator_descriptor_coverage(
            policy.config, exception_descriptor, trade_at_descriptor
        )
        self._config = config
        self._policy = policy
        self._exception_evaluator_descriptor = exception_descriptor
        self._trade_at_evaluator_descriptor = trade_at_descriptor
        self._exception_evaluator = self._build_evaluator(exception_descriptor)
        self._trade_at_evaluator = self._build_evaluator(trade_at_descriptor)
        self._claim_lock = RLock()
        self._claimed = False
        self.assert_integrity()

    @property
    def config(self) -> PolicyGatewayConfig:
        return self._config

    @property
    def sha256(self) -> str:
        return self._config.sha256

    @property
    def policy_audit_events(self) -> tuple[PolicyComplianceAuditEvent, ...]:
        return self._policy.audit_log.events

    @property
    def replay_material(self) -> SyntheticGatewayReplayMaterial:
        """Capture complete source-bound material for synthetic reconstruction."""
        self.assert_integrity()
        if not self._config.synthetic_only or not self._policy.config.synthetic_only:
            raise PolicyGatewayError(
                "real SEC policy replay requires locally verified primary sources"
            )
        source_bytes = self._policy.assignments.read_source_bytes()
        return SyntheticGatewayReplayMaterial(
            gateway_config=self._config,
            policy_config=self._policy.config,
            source_registry=self._policy.config.source_registry,
            assignment_manifest=self._policy.assignments.manifest,
            assignment_source_bytes=source_bytes,
            exception_evaluator=self._exception_evaluator_descriptor,
            trade_at_evaluator=self._trade_at_evaluator_descriptor,
        )

    @staticmethod
    def _build_evaluator(
        descriptor: SyntheticEvaluatorDescriptor,
    ) -> SyntheticFixedEvaluator | None:
        return (
            None
            if descriptor.mode is SyntheticEvaluatorMode.NOT_CONFIGURED
            else SyntheticFixedEvaluator(
                role=descriptor.role.value,
                decisions=tuple(
                    (item.evaluator_id, item.allowed) for item in descriptor.decisions
                ),
                proof_domain=cast(str, descriptor.proof_domain),
                schema_version=descriptor.schema_version,
            )
        )

    @staticmethod
    def _validate_evaluator_descriptor_coverage(
        policy_config: SecTickSizePilotConfig,
        exception_descriptor: SyntheticEvaluatorDescriptor,
        trade_at_descriptor: SyntheticEvaluatorDescriptor,
    ) -> None:
        expected_exception_ids = {
            item.evaluator_id for item in policy_config.exceptions
        }
        expected_trade_at_ids = {
            rule.trade_at_restriction.evaluator_id
            for rule in policy_config.group_rules
            if rule.trade_at_restriction is not None
        }
        for descriptor, expected in (
            (exception_descriptor, expected_exception_ids),
            (trade_at_descriptor, expected_trade_at_ids),
        ):
            configured = {item.evaluator_id for item in descriptor.decisions}
            if descriptor.mode is SyntheticEvaluatorMode.FIXED_RECEIPT:
                if configured != expected:
                    _fail("fixed synthetic evaluator ids differ from policy mechanisms")
            elif configured:
                _fail("not-configured evaluator must not declare evaluator ids")

    def assert_integrity(self) -> None:
        """Recheck all source and semantic commitments without mutating audit."""
        try:
            self._policy.config.source_registry.assert_source_integrity()
            self._policy.assignments.assert_source_integrity()
            self._policy.audit_log.assert_integrity()
        except Exception as error:
            raise PolicyGatewayError(
                "bound policy evidence or audit is invalid"
            ) from error
        expected = (
            self._policy.config.sha256,
            self._policy.assignments.manifest.sha256,
            self._policy.assignments.source_sha256,
            self._policy.config.source_registry.sha256,
            _evaluator_registry_sha256(
                self._exception_evaluator_descriptor,
                self._trade_at_evaluator_descriptor,
            ),
            self._policy.config.synthetic_only,
        )
        actual = (
            self._config.policy_sha256,
            self._config.assignment_sha256,
            self._config.assignment_source_sha256,
            self._config.source_registry_sha256,
            self._config.evaluator_registry_sha256,
            self._config.synthetic_only,
        )
        if actual != expected:
            _fail(
                "gateway commitment differs from the bound policy artifacts or "
                "evaluator registry commitment"
            )

    def claim_for_market(self) -> None:
        """Atomically grant this stateful audit stream to exactly one market."""
        with self._claim_lock:
            if self._claimed:
                _fail("policy gateway is already claimed by another market")
            if self._policy.audit_log.events:
                _fail("policy gateway audit is not empty at market claim time")
            self._claimed = True

    def evaluate_quote(
        self, *, cause_event_key: str, logical_time: int, price_units: int
    ) -> GatewayComplianceRecord:
        """Evaluate one limit price before it can enter or cross the book."""
        return self._evaluate(
            cause_event_key=cause_event_key,
            logical_time=logical_time,
            action=ComplianceAction.QUOTE,
            price_units=price_units,
            fill_index=None,
        )

    def evaluate_trade(
        self,
        *,
        cause_event_key: str,
        logical_time: int,
        price_units: int,
        fill_index: int,
    ) -> GatewayComplianceRecord:
        """Evaluate one planned fill before any economic state is mutated."""
        return self._evaluate(
            cause_event_key=cause_event_key,
            logical_time=logical_time,
            action=ComplianceAction.TRADE,
            price_units=price_units,
            fill_index=fill_index,
        )

    def _evaluate(
        self,
        *,
        cause_event_key: str,
        logical_time: int,
        action: ComplianceAction,
        price_units: int,
        fill_index: int | None,
    ) -> GatewayComplianceRecord:
        with self._claim_lock:
            if not self._claimed:
                _fail("policy gateway must be claimed by a market before evaluation")
        self.assert_integrity()
        timestamp = self._config.clock.resolve(logical_time)
        binding = self._config.binding_for(
            cause_event_key=cause_event_key,
            action=action,
            fill_index=fill_index,
        )
        request_key = (
            cause_event_key
            if action is ComplianceAction.QUOTE
            else f"{cause_event_key}:trade:{fill_index}"
        )
        request = ComplianceRequest(
            event_key=request_key,
            security_id=self._config.security_id,
            timestamp=timestamp,
            action=action,
            price_units=price_units,
            exception_id=None if binding is None else binding.exception_id,
            trade_at_context_sha256=(
                None if binding is None else binding.trade_at_context_sha256
            ),
        )
        before = len(self._policy.audit_log.events)
        try:
            decision = self._policy.evaluate(
                request,
                exception_evaluator=self._exception_evaluator,
                trade_at_evaluator=self._trade_at_evaluator,
            )
        except Exception as error:
            raise PolicyGatewayError(
                "policy evaluation failed before a decision"
            ) from error
        events = self._policy.audit_log.events
        if len(events) != before + 1 or events[-1].decision != decision:
            _fail("policy evaluation did not append exactly one matching audit event")
        self._policy.audit_log.assert_integrity()
        return GatewayComplianceRecord(
            cause_event_key=cause_event_key,
            fill_index=fill_index,
            gateway_sha256=self.sha256,
            decision=decision,
            policy_audit_event=events[-1],
        )


__all__ = [
    "FrozenPolicyClock",
    "GatewayComplianceRecord",
    "POLICY_GATEWAY_SCHEMA_VERSION",
    "PolicyClockPoint",
    "PolicyGatewayConfig",
    "PolicyGatewayError",
    "PolicyRequestBinding",
    "SecTickSizePolicyGateway",
    "SYNTHETIC_EVALUATOR_SCHEMA_VERSION",
    "SYNTHETIC_EVALUATOR_REGISTRY_SCHEMA_VERSION",
    "SyntheticEvaluatorDecision",
    "SyntheticEvaluatorDescriptor",
    "SyntheticEvaluatorMode",
    "SyntheticEvaluatorRole",
    "SyntheticGatewayReplayMaterial",
]
