"""Strict semantic event identities and stateless exogenous shock tapes.

``SemanticEventKey`` is deliberately distinct from the original smoke-run
``EventKey``.  An explicit conversion method is provided for market commands;
the conversion commits to the complete semantic identity rather than trying to
infer semantic fields from the legacy key.

Every random draw is a pure hash function of an explicit seed, semantic key,
domain, and draw index.  There is no cursor, generator state, or call-order
dependence.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import InitVar, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from threading import Lock
from typing import Literal, NoReturn, Self, cast

from policytwin.counterfactual.randomness import EVENT_KEY_SCHEMA_VERSION, EventKey

SEMANTIC_EVENT_KEY_SCHEMA_VERSION = "policytwin.semantic-event-key.v1"
EXOGENOUS_SHOCK_TAPE_SCHEMA_VERSION = "policytwin.exogenous-shock-tape.v1"
HASH_BASED_RNG_IMPLEMENTATION_ID = "policytwin.sha256-binary64-box-muller.v1"
SCENARIO_GENERATION_DRAW_SCHEMA_VERSION = "policytwin.scenario-generation-draw.v1"
SCENARIO_GENERATION_RECEIPT_SCHEMA_VERSION = "policytwin.scenario-generation-receipt.v1"
SHOCK_CONSUMPTION_LEDGER_SCHEMA_VERSION = "policytwin.shock-consumption-ledger.v3"
SHOCK_TRANSITION_RECEIPT_SCHEMA_VERSION = "policytwin.shock-transition-receipt.v1"
SHOCK_TRANSITION_IMPLEMENTATION_ID = "policytwin.additive-numeric-shock.v1"
SCENARIO_EVENT_RECEIPT_SCHEMA_VERSION = "policytwin.scenario-event-receipt.v1"

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SEMANTIC_KEY_PATTERN = re.compile(r"^pse1:[0-9a-f]{64}$")
_LEGACY_KEY_PATTERN = re.compile(r"^ptk1:[0-9a-f]{64}$")
_TWO_PI = 2.0 * math.pi
_UNIT_SCALE = 1 << 53
MAX_SCENARIO_GENERATION_DRAWS = 4_096
_MAX_DRAW_INDEX = (1 << 31) - 1


class SemanticEventError(ValueError):
    """Raised when a semantic event identity or shock tape is invalid."""


class ShockAccessError(SemanticEventError):
    """Raised when a world violates the engine-issued shock-access contract."""

    def __init__(
        self,
        message: str,
        *,
        rule_code: ShockAccessViolationCode,
        event_key_value: str | None,
    ) -> None:
        super().__init__(message)
        self.rule_code = rule_code
        self.event_key_value = event_key_value


class EventType(StrEnum):
    """Closed semantic event categories shared across paired worlds."""

    EXOGENOUS_SHOCK = "exogenous_shock"
    INFORMATION = "information"
    AGENT_DECISION = "agent_decision"
    ORDER_SUBMISSION = "order_submission"
    ORDER_CANCEL = "order_cancel"
    EVALUATION_QUOTE = "evaluation_quote"
    EVALUATION_TRADE = "evaluation_trade"
    EVALUATION_ORDER_OUTCOME = "evaluation_order_outcome"
    EVALUATION_INFORMATION_RESPONSE = "evaluation_information_response"


class ShockAccessViolationCode(StrEnum):
    """Stable fail-closed violations recorded by a shock capability."""

    INVALID_KEY = "INVALID_KEY"
    UNDECLARED_EVENT = "UNDECLARED_EVENT"
    DUPLICATE_CONSUMPTION = "DUPLICATE_CONSUMPTION"
    CAPABILITY_CLOSED = "CAPABILITY_CLOSED"
    WRONG_PRE_STATE = "WRONG_PRE_STATE"
    OUT_OF_ORDER_TRANSITION = "OUT_OF_ORDER_TRANSITION"
    UNAPPLIED_REALIZATION = "UNAPPLIED_REALIZATION"
    OMITTED_TRANSITION = "OMITTED_TRANSITION"
    SCENARIO_EVENT_BEFORE_TRANSITION = "SCENARIO_EVENT_BEFORE_TRANSITION"
    DUPLICATE_SCENARIO_EVENT = "DUPLICATE_SCENARIO_EVENT"
    INVALID_SCENARIO_EVENT = "INVALID_SCENARIO_EVENT"


def _fail(message: str) -> NoReturn:
    raise SemanticEventError(message)


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_PATTERN.fullmatch(value) is None:
        _fail(f"{field} must be a stable ASCII identifier")
    return value


def _optional_identifier(value: object, field: str) -> str | None:
    if value is None:
        return None
    return _identifier(value, field)


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        _fail(f"{field} must be a finite real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        _fail(f"{field} must be finite")
    return normalized


def _non_negative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(f"{field} must be a non-negative integer")
    return value


def _bounded_draw_index(value: object, field: str) -> int:
    result = _non_negative_int(value, field)
    if result > _MAX_DRAW_INDEX:
        _fail(f"{field} exceeds the frozen resource bound")
    return result


def _bounded_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 512:
        _fail(f"{field} must contain between 1 and 512 characters")
    return value


def _aware_utc(value: object, field: str) -> datetime:
    if not isinstance(value, datetime):
        _fail(f"{field} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        _fail(f"{field} must include an explicit timezone")
    try:
        return value.astimezone(UTC)
    except (OverflowError, ValueError) as error:
        raise SemanticEventError(f"{field} is outside the UTC range") from error


def _parse_datetime(value: object, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        _fail(f"{field} must be a non-empty ISO 8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise SemanticEventError(f"{field} must be ISO 8601") from error
    return _aware_utc(parsed, field)


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SemanticEventError("event value is not canonical JSON") from error


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _strict_fields(
    value: Mapping[str, object], expected: frozenset[str], context: str
) -> None:
    if frozenset(value) != expected:
        _fail(f"{context} has missing or unknown fields")


def _require_mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        _fail(f"{field} must be an object with string keys")
    return cast(Mapping[str, object], value)


def _event_type(value: object) -> EventType:
    if not isinstance(value, str):
        _fail("event_type must be a string")
    try:
        return EventType(value)
    except ValueError as error:
        raise SemanticEventError("unsupported event_type") from error


def _semantic_key_value(value: object) -> str:
    if not isinstance(value, str) or _SEMANTIC_KEY_PATTERN.fullmatch(value) is None:
        _fail("semantic event key commitment is invalid")
    return value


@dataclass(frozen=True, slots=True)
class SemanticEventKey:
    """Complete semantic identity for one potential event.

    ``scenario_id`` identifies the paired scenario and must be identical in the
    policy and baseline worlds.  It is not a world identifier.  ``agent_id`` is
    optional only for system-wide information and exogenous shocks.
    """

    scenario_id: str
    event_type: EventType
    entity_id: str
    timestamp: datetime
    agent_id: str | None = None
    schema_version: str = SEMANTIC_EVENT_KEY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self) is not SemanticEventKey:
            _fail("semantic event keys must use the exact verifier type")
        if type(self.scenario_id) is not str or type(self.entity_id) is not str:
            _fail("semantic event key identifiers must use exact strings")
        if self.agent_id is not None and type(self.agent_id) is not str:
            _fail("semantic event key agent_id must use an exact string")
        if type(self.event_type) is not EventType:
            _fail("semantic event key event_type must use the exact enum type")
        if type(self.timestamp) is not datetime:
            _fail("semantic event key timestamp must use the exact datetime type")
        if type(self.schema_version) is not str:
            _fail("semantic event key schema_version must use an exact string")
        _identifier(self.scenario_id, "scenario_id")
        if not isinstance(self.event_type, EventType):
            _fail("event_type must be an EventType")
        _identifier(self.entity_id, "entity_id")
        object.__setattr__(self, "timestamp", _aware_utc(self.timestamp, "timestamp"))
        _optional_identifier(self.agent_id, "agent_id")
        if self.schema_version != SEMANTIC_EVENT_KEY_SCHEMA_VERSION:
            _fail("unsupported semantic event key schema version")
        if (
            self.event_type
            in {
                EventType.AGENT_DECISION,
                EventType.ORDER_SUBMISSION,
                EventType.ORDER_CANCEL,
            }
            and self.agent_id is None
        ):
            _fail("agent-specific event types require agent_id")

    def identity_mapping(self) -> dict[str, object]:
        """Return the exact semantic fields covered by the key commitment."""
        return {
            "agent_id": self.agent_id,
            "entity_id": self.entity_id,
            "event_type": self.event_type.value,
            "scenario_id": self.scenario_id,
            "schema_version": self.schema_version,
            "timestamp": self.timestamp.isoformat(),
        }

    @property
    def canonical_bytes(self) -> bytes:
        """Return the canonical semantic identity bytes."""
        return _canonical_json_bytes(self.identity_mapping())

    @property
    def value(self) -> str:
        """Return the stable semantic event commitment."""
        return f"pse1:{_sha256_bytes(self.canonical_bytes)}"

    def to_mapping(self) -> dict[str, object]:
        """Return semantic fields and their derived commitment."""
        return {**self.identity_mapping(), "value": self.value}

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        """Strictly reconstruct and verify a semantic event key."""
        _strict_fields(
            value,
            frozenset(
                {
                    "scenario_id",
                    "event_type",
                    "entity_id",
                    "timestamp",
                    "agent_id",
                    "schema_version",
                    "value",
                }
            ),
            "semantic event key",
        )
        schema_version = value["schema_version"]
        if not isinstance(schema_version, str):
            _fail("schema_version must be a string")
        result = cls(
            scenario_id=_identifier(value["scenario_id"], "scenario_id"),
            event_type=_event_type(value["event_type"]),
            entity_id=_identifier(value["entity_id"], "entity_id"),
            timestamp=_parse_datetime(value["timestamp"], "timestamp"),
            agent_id=_optional_identifier(value["agent_id"], "agent_id"),
            schema_version=schema_version,
        )
        if _semantic_key_value(value["value"]) != result.value:
            _fail("semantic event key value does not match its fields")
        return result

    def to_legacy_event_key(
        self,
        *,
        seed: str,
        replicate_id: int,
        simulation_step: int,
    ) -> EventKey:
        """Create an explicit legacy key suitable for current market commands."""
        if not isinstance(seed, str) or not seed or len(seed) > 512:
            _fail("seed must contain between 1 and 512 characters")
        _non_negative_int(replicate_id, "replicate_id")
        _non_negative_int(simulation_step, "simulation_step")
        return EventKey(
            experiment_id=self.scenario_id,
            seed=seed,
            replicate_id=replicate_id,
            simulation_step=simulation_step,
            purpose=f"semantic.{self.value}",
        )


def _seed(value: object) -> str:
    if not isinstance(value, str) or not value or len(value) > 512:
        _fail("seed must contain between 1 and 512 characters")
    return value


def _draw_digest(
    *,
    seed: str,
    event_key: SemanticEventKey,
    domain: str,
    draw_index: int,
    attempt: int = 0,
) -> bytes:
    _seed(seed)
    if not isinstance(event_key, SemanticEventKey):
        _fail("random draws require a SemanticEventKey")
    _identifier(domain, "domain")
    _non_negative_int(draw_index, "draw_index")
    _non_negative_int(attempt, "attempt")
    material = _canonical_json_bytes(
        {
            "attempt": attempt,
            "domain": domain,
            "draw_index": draw_index,
            "event_key": event_key.value,
            "schema_version": "policytwin.hash-rng.v1",
            "seed": seed,
        }
    )
    return hashlib.sha256(material).digest()


def _uniform_from_bits(bits: int) -> float:
    if (
        isinstance(bits, bool)
        or not isinstance(bits, int)
        or not 0 <= bits < _UNIT_SCALE
    ):
        _fail("uniform bits must be a 53-bit non-negative integer")
    return (bits + 0.5) / _UNIT_SCALE


@dataclass(frozen=True, slots=True)
class HashBasedRNG:
    """Visible immutable seed plus stateless keyed random functions."""

    seed: str

    def __post_init__(self) -> None:
        _seed(self.seed)

    @property
    def seed_sha256(self) -> str:
        """Return a provenance-safe commitment to the explicit seed."""
        return _sha256_bytes(self.seed.encode("utf-8"))

    def uniform(
        self,
        event_key: SemanticEventKey,
        *,
        domain: str,
        draw_index: int = 0,
    ) -> float:
        """Return a deterministic value strictly inside ``(0, 1)``."""
        return _uniform_from_bits(
            self.uniform_bits(
                event_key,
                domain=domain,
                draw_index=draw_index,
            )
        )

    def uniform_bits(
        self,
        event_key: SemanticEventKey,
        *,
        domain: str,
        draw_index: int = 0,
    ) -> int:
        """Return the exact 53 source bits used by :meth:`uniform`."""
        digest = _draw_digest(
            seed=self.seed,
            event_key=event_key,
            domain=domain,
            draw_index=draw_index,
        )
        return int.from_bytes(digest[:8], "big") >> 11

    def integer(
        self,
        event_key: SemanticEventKey,
        lower: int,
        upper: int,
        *,
        domain: str,
        draw_index: int = 0,
    ) -> int:
        """Return an unbiased inclusive integer independent of call order."""
        if (
            isinstance(lower, bool)
            or not isinstance(lower, int)
            or isinstance(upper, bool)
            or not isinstance(upper, int)
            or lower > upper
        ):
            _fail("integer bounds must be integers satisfying lower <= upper")
        _identifier(domain, "domain")
        _non_negative_int(draw_index, "draw_index")
        span = upper - lower + 1
        full_domain = 1 << 256
        acceptance_limit = full_domain - (full_domain % span)
        attempt = 0
        while True:
            candidate = int.from_bytes(
                _draw_digest(
                    seed=self.seed,
                    event_key=event_key,
                    domain=domain,
                    draw_index=draw_index,
                    attempt=attempt,
                ),
                "big",
            )
            if candidate < acceptance_limit:
                return lower + candidate % span
            attempt += 1

    def normal(
        self,
        event_key: SemanticEventKey,
        *,
        domain: str,
        draw_index: int = 0,
    ) -> float:
        """Return one deterministic standard-normal Box-Muller draw."""
        _non_negative_int(draw_index, "draw_index")
        first = self.uniform(
            event_key,
            domain=domain,
            draw_index=2 * draw_index,
        )
        second = self.uniform(
            event_key,
            domain=domain,
            draw_index=2 * draw_index + 1,
        )
        return math.sqrt(-2.0 * math.log(first)) * math.cos(_TWO_PI * second)

    def bernoulli(
        self,
        event_key: SemanticEventKey,
        probability: float,
        *,
        domain: str,
        draw_index: int = 0,
    ) -> bool:
        """Return a deterministic Bernoulli draw with explicit probability."""
        normalized = _finite(probability, "probability")
        if not 0.0 <= normalized <= 1.0:
            _fail("probability must be in [0, 1]")
        if normalized == 0.0:
            return False
        if normalized == 1.0:
            return True
        return (
            self.uniform(
                event_key,
                domain=domain,
                draw_index=draw_index,
            )
            < normalized
        )


def hash_uniform(
    event_key: SemanticEventKey,
    *,
    seed: str,
    domain: str,
    draw_index: int = 0,
) -> float:
    """Functional wrapper for a stateless open-unit draw."""
    return HashBasedRNG(seed).uniform(
        event_key,
        domain=domain,
        draw_index=draw_index,
    )


def hash_integer(
    event_key: SemanticEventKey,
    lower: int,
    upper: int,
    *,
    seed: str,
    domain: str,
    draw_index: int = 0,
) -> int:
    """Functional wrapper for a stateless inclusive integer draw."""
    return HashBasedRNG(seed).integer(
        event_key,
        lower,
        upper,
        domain=domain,
        draw_index=draw_index,
    )


def _normalize_shock_values(
    values: tuple[tuple[str, float], ...],
) -> tuple[tuple[str, float], ...]:
    if not isinstance(values, tuple) or not values:
        _fail("shock values must be a non-empty immutable tuple")
    normalized: list[tuple[str, float]] = []
    for item in values:
        if not isinstance(item, tuple) or len(item) != 2:
            _fail("each shock value must be a name-value tuple")
        name, value = item
        normalized.append((_identifier(name, "shock field"), _finite(value, name)))
    if len({name for name, _ in normalized}) != len(normalized):
        _fail("shock field names must be unique")
    return tuple(sorted(normalized))


@dataclass(frozen=True, slots=True)
class ExogenousShock:
    """One immutable synthetic vector indexed by an exogenous semantic key."""

    event_key: SemanticEventKey
    values: tuple[tuple[str, float], ...]
    synthetic_only: Literal[True] = True

    def __post_init__(self) -> None:
        if type(self) is not ExogenousShock:
            _fail("exogenous shocks must use the exact verifier type")
        if type(self.event_key) is not SemanticEventKey:
            _fail("event_key must be a SemanticEventKey")
        if type(self.values) is not tuple or not all(
            type(item) is tuple
            and len(item) == 2
            and type(item[0]) is str
            and type(item[1]) in {int, float}
            for item in self.values
        ):
            _fail("shock values must use exact immutable built-in values")
        if self.event_key.event_type is not EventType.EXOGENOUS_SHOCK:
            _fail("shock event_key must use EXOGENOUS_SHOCK event_type")
        object.__setattr__(self, "values", _normalize_shock_values(self.values))
        if self.synthetic_only is not True:
            _fail("exogenous shocks in this module must be synthetic_only")

    def value(self, field: str) -> float:
        """Return one named shock component or fail rather than impute."""
        _identifier(field, "field")
        for name, value in self.values:
            if name == field:
                return value
        raise KeyError(field)

    def to_mapping(self) -> dict[str, object]:
        """Return the canonical shock representation."""
        return {
            "event_key": self.event_key.to_mapping(),
            "synthetic_only": self.synthetic_only,
            "values": [{"name": name, "value": value} for name, value in self.values],
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        """Strictly reconstruct a synthetic shock."""
        _strict_fields(
            value,
            frozenset({"event_key", "values", "synthetic_only"}),
            "exogenous shock",
        )
        if value["synthetic_only"] is not True:
            _fail("exogenous shock must be explicitly synthetic_only")
        raw_values = value["values"]
        if not isinstance(raw_values, Sequence) or isinstance(
            raw_values, str | bytes | bytearray
        ):
            _fail("shock values must be an array")
        parsed: list[tuple[str, float]] = []
        for raw_item in raw_values:
            item = _require_mapping(raw_item, "shock value")
            _strict_fields(item, frozenset({"name", "value"}), "shock value")
            name = _identifier(item["name"], "shock field")
            parsed.append((name, _finite(item["value"], name)))
        return cls(
            event_key=SemanticEventKey.from_mapping(
                _require_mapping(value["event_key"], "event_key")
            ),
            values=tuple(parsed),
            synthetic_only=True,
        )


@dataclass(frozen=True, slots=True)
class ShockTransitionState:
    """Immutable numeric state owned by the keyed-shock transition engine."""

    values: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _normalize_shock_values(self.values))

    @classmethod
    def from_mapping(cls, value: Mapping[str, int | float]) -> Self:
        """Create a canonical transition state without accepting booleans/NaNs."""
        return cls(
            values=tuple(
                (_identifier(name, "transition state field"), _finite(item, name))
                for name, item in value.items()
            )
        )

    def value(self, field: str) -> float:
        """Return one state component or fail rather than invent a default."""
        _identifier(field, "transition state field")
        for name, value in self.values:
            if name == field:
                return value
        raise KeyError(field)

    def with_value(self, field: str, value: int | float) -> Self:
        """Return a new state with exactly one existing numeric field replaced."""
        stable_field = _identifier(field, "transition state field")
        normalized = _finite(value, stable_field)
        if all(name != stable_field for name, _ in self.values):
            raise KeyError(stable_field)
        return type(self)(
            values=tuple(
                (name, normalized if name == stable_field else current)
                for name, current in self.values
            )
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "values": [{"name": name, "value": value} for name, value in self.values]
        }

    @property
    def sha256(self) -> str:
        """Commit to the complete numeric state."""
        return _sha256_bytes(_canonical_json_bytes(self.to_mapping()))


@dataclass(frozen=True, slots=True)
class ShockTransitionConfig:
    """Frozen configuration for the initial additive synthetic transition.

    The implementation is deliberately closed rather than accepting a runner
    callback: ``post[state_field] = pre[state_field] + scale * shock[shock_field]``.
    A new transition algorithm requires a new implementation identifier and
    implementation commitment.
    """

    state_field: str
    shock_field: str
    scale: float = 1.0
    implementation_id: str = SHOCK_TRANSITION_IMPLEMENTATION_ID

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "state_field", _identifier(self.state_field, "state_field")
        )
        object.__setattr__(
            self, "shock_field", _identifier(self.shock_field, "shock_field")
        )
        object.__setattr__(self, "scale", _finite(self.scale, "scale"))
        if self.implementation_id != SHOCK_TRANSITION_IMPLEMENTATION_ID:
            _fail("unsupported shock transition implementation")

    @property
    def implementation_sha256(self) -> str:
        """Commit to the exact closed transition semantics, not caller code."""
        return _sha256_bytes(
            _canonical_json_bytes(
                {
                    "arithmetic": "binary64",
                    "implementation_id": self.implementation_id,
                    "semantics": "post[state]=pre[state]+scale*shock[field]",
                    "unknown_field_behavior": "fail_closed",
                }
            )
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "implementation_id": self.implementation_id,
            "implementation_sha256": self.implementation_sha256,
            "scale": self.scale,
            "shock_field": self.shock_field,
            "state_field": self.state_field,
        }

    @property
    def sha256(self) -> str:
        """Commit to both implementation identity and configured fields."""
        return _sha256_bytes(_canonical_json_bytes(self.to_mapping()))

    def apply(
        self, state: ShockTransitionState, shock: ExogenousShock
    ) -> ShockTransitionState:
        """Deterministically replay the closed transition implementation."""
        if not isinstance(state, ShockTransitionState):
            _fail("shock transition requires a ShockTransitionState")
        if not isinstance(shock, ExogenousShock):
            _fail("shock transition requires an ExogenousShock")
        current = state.value(self.state_field)
        realization = shock.value(self.shock_field)
        return state.with_value(
            self.state_field,
            current + self.scale * realization,
        )


_SHOCK_TRANSITION_RECEIPT_ISSUER = object()


@dataclass(frozen=True, slots=True)
class ShockTransitionReceipt:
    """Engine-issued proof of one keyed realization's state transition."""

    sequence: int
    world_id: str
    role: Literal["baseline", "candidate"]
    scenario_id: str
    event_key: SemanticEventKey
    realization_sha256: str
    pre_state: ShockTransitionState
    pre_state_sha256: str
    transition_implementation_sha256: str
    transition_config_sha256: str
    post_state: ShockTransitionState
    post_state_sha256: str
    _issuer: InitVar[object]
    schema_version: str = SHOCK_TRANSITION_RECEIPT_SCHEMA_VERSION

    def __post_init__(self, _issuer: object) -> None:
        if _issuer is not _SHOCK_TRANSITION_RECEIPT_ISSUER:
            raise ShockAccessError(
                "shock transition receipts can only be issued by the engine",
                rule_code=ShockAccessViolationCode.INVALID_KEY,
                event_key_value=None,
            )
        _non_negative_int(self.sequence, "sequence")
        _identifier(self.world_id, "world_id")
        if self.role not in ("baseline", "candidate"):
            _fail("role must be baseline or candidate")
        _identifier(self.scenario_id, "scenario_id")
        if not isinstance(self.event_key, SemanticEventKey):
            _fail("receipt event_key must be a SemanticEventKey")
        if self.event_key.scenario_id != self.scenario_id:
            _fail("receipt event_key is bound to the wrong scenario")
        for field in (
            "realization_sha256",
            "pre_state_sha256",
            "transition_implementation_sha256",
            "transition_config_sha256",
            "post_state_sha256",
        ):
            value = getattr(self, field)
            if (
                not isinstance(value, str)
                or re.fullmatch(r"[0-9a-f]{64}", value) is None
            ):
                _fail(f"{field} must be a lowercase SHA-256 digest")
        if not isinstance(self.pre_state, ShockTransitionState):
            _fail("receipt pre_state must be a ShockTransitionState")
        if not isinstance(self.post_state, ShockTransitionState):
            _fail("receipt post_state must be a ShockTransitionState")
        if self.pre_state.sha256 != self.pre_state_sha256:
            _fail("receipt pre-state hash does not match its state")
        if self.post_state.sha256 != self.post_state_sha256:
            _fail("receipt post-state hash does not match its state")
        if self.schema_version != SHOCK_TRANSITION_RECEIPT_SCHEMA_VERSION:
            _fail("unsupported shock transition receipt schema version")

    def _signed_mapping(self) -> dict[str, object]:
        return {
            "event_key": self.event_key.to_mapping(),
            "post_state": self.post_state.to_mapping(),
            "post_state_sha256": self.post_state_sha256,
            "pre_state": self.pre_state.to_mapping(),
            "pre_state_sha256": self.pre_state_sha256,
            "realization_sha256": self.realization_sha256,
            "role": self.role,
            "scenario_id": self.scenario_id,
            "schema_version": self.schema_version,
            "sequence": self.sequence,
            "transition_config_sha256": self.transition_config_sha256,
            "transition_implementation_sha256": (self.transition_implementation_sha256),
            "world_id": self.world_id,
        }

    @property
    def engine_signature_sha256(self) -> str:
        """Content signature over all scientific receipt fields."""
        return _sha256_bytes(
            b"policytwin-engine-shock-transition-receipt-v1\0"
            + _canonical_json_bytes(self._signed_mapping())
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            **self._signed_mapping(),
            "engine_signature_sha256": self.engine_signature_sha256,
        }


@dataclass(frozen=True, slots=True)
class ShockTransitionResult:
    """The only result returned by the engine-owned transition operation."""

    shock: ExogenousShock
    post_state: ShockTransitionState
    receipt: ShockTransitionReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.shock, ExogenousShock):
            _fail("transition result shock has the wrong type")
        if not isinstance(self.post_state, ShockTransitionState):
            _fail("transition result post_state has the wrong type")
        if not isinstance(self.receipt, ShockTransitionReceipt):
            _fail("transition result receipt has the wrong type")
        if self.receipt.post_state != self.post_state:
            _fail("transition result state does not match its engine receipt")


@dataclass(frozen=True, slots=True)
class ScenarioGeneratorSpec:
    """Frozen pre-cutoff identity of a model-generated scenario mechanism.

    This object is a commitment, not evidence that arbitrary input data were
    lawful.  A real study must obtain ``source_lineage_sha256`` from the data
    firewall.  Its purpose is to make generated future scenario events
    categorically distinct from observed records with post-cutoff availability.
    """

    generator_id: str
    generator_config_sha256: str
    generator_implementation_sha256: str
    source_lineage_sha256: str
    cutoff_policy_sha256: str
    frozen_at: datetime
    trusted_cutoff: datetime
    synthetic_only: Literal[True] = True

    def __post_init__(self) -> None:
        if type(self) is not ScenarioGeneratorSpec:
            _fail("scenario generator specs must use the exact verifier type")
        if any(
            type(getattr(self, field)) is not str
            for field in (
                "generator_id",
                "generator_config_sha256",
                "generator_implementation_sha256",
                "source_lineage_sha256",
                "cutoff_policy_sha256",
            )
        ):
            _fail("scenario generator spec text fields must use exact strings")
        if (
            type(self.frozen_at) is not datetime
            or type(self.trusted_cutoff) is not datetime
        ):
            _fail("scenario generator spec times must use exact datetimes")
        _identifier(self.generator_id, "generator_id")
        for field in (
            "generator_config_sha256",
            "generator_implementation_sha256",
            "source_lineage_sha256",
            "cutoff_policy_sha256",
        ):
            value = getattr(self, field)
            if (
                not isinstance(value, str)
                or re.fullmatch(r"[0-9a-f]{64}", value) is None
            ):
                _fail(f"{field} must be a lowercase SHA-256 digest")
        frozen_at = _aware_utc(self.frozen_at, "frozen_at")
        trusted_cutoff = _aware_utc(self.trusted_cutoff, "trusted_cutoff")
        if frozen_at > trusted_cutoff:
            _fail("scenario generator must be frozen no later than trusted_cutoff")
        object.__setattr__(self, "frozen_at", frozen_at)
        object.__setattr__(self, "trusted_cutoff", trusted_cutoff)
        if self.synthetic_only is not True:
            _fail("scenario generator specs in this module are synthetic-only")

    def to_mapping(self) -> dict[str, object]:
        """Return the complete frozen generator commitment."""
        return {
            "cutoff_policy_sha256": self.cutoff_policy_sha256,
            "frozen_at": self.frozen_at.isoformat(),
            "generator_config_sha256": self.generator_config_sha256,
            "generator_id": self.generator_id,
            "generator_implementation_sha256": (self.generator_implementation_sha256),
            "source_lineage_sha256": self.source_lineage_sha256,
            "synthetic_only": self.synthetic_only,
            "trusted_cutoff": self.trusted_cutoff.isoformat(),
        }

    @property
    def sha256(self) -> str:
        """Return the content commitment for the generator specification."""
        return _sha256_bytes(_canonical_json_bytes(self.to_mapping()))

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        """Strictly reconstruct a frozen scenario-generator specification."""
        _strict_fields(
            value,
            frozenset(
                {
                    "cutoff_policy_sha256",
                    "frozen_at",
                    "generator_config_sha256",
                    "generator_id",
                    "generator_implementation_sha256",
                    "source_lineage_sha256",
                    "synthetic_only",
                    "trusted_cutoff",
                }
            ),
            "scenario generator spec",
        )
        if value["synthetic_only"] is not True:
            _fail("scenario generator spec must be explicitly synthetic_only")
        return cls(
            generator_id=_identifier(value["generator_id"], "generator_id"),
            generator_config_sha256=cast(str, value["generator_config_sha256"]),
            generator_implementation_sha256=cast(
                str, value["generator_implementation_sha256"]
            ),
            source_lineage_sha256=cast(str, value["source_lineage_sha256"]),
            cutoff_policy_sha256=cast(str, value["cutoff_policy_sha256"]),
            frozen_at=_parse_datetime(value["frozen_at"], "frozen_at"),
            trusted_cutoff=_parse_datetime(value["trusted_cutoff"], "trusted_cutoff"),
            synthetic_only=True,
        )


def _legacy_event_key_mapping(event_key: EventKey) -> dict[str, object]:
    if not isinstance(event_key, EventKey):
        _fail("root_event_key must be an EventKey")
    mapping = {
        "experiment_id": event_key.experiment_id,
        "purpose": event_key.purpose,
        "replicate_id": event_key.replicate_id,
        "schema_version": event_key.schema_version,
        "seed": event_key.seed,
        "simulation_step": event_key.simulation_step,
        "value": event_key.value,
    }
    _legacy_event_key_from_mapping(mapping)
    return mapping


def _legacy_event_key_from_mapping(value: Mapping[str, object]) -> EventKey:
    _strict_fields(
        value,
        frozenset(
            {
                "experiment_id",
                "purpose",
                "replicate_id",
                "schema_version",
                "seed",
                "simulation_step",
                "value",
            }
        ),
        "root legacy event key",
    )
    if value["schema_version"] != EVENT_KEY_SCHEMA_VERSION:
        _fail("unsupported root legacy event key schema version")
    result = EventKey(
        experiment_id=_bounded_text(value["experiment_id"], "experiment_id"),
        seed=_bounded_text(value["seed"], "seed"),
        replicate_id=_bounded_draw_index(value["replicate_id"], "replicate_id"),
        simulation_step=_bounded_draw_index(
            value["simulation_step"], "simulation_step"
        ),
        purpose=_bounded_text(value["purpose"], "purpose"),
        schema_version=EVENT_KEY_SCHEMA_VERSION,
    )
    commitment = value["value"]
    if (
        not isinstance(commitment, str)
        or _LEGACY_KEY_PATTERN.fullmatch(commitment) is None
        or commitment != result.value
    ):
        _fail("root legacy event key value does not match its complete mapping")
    return result


def hash_based_rng_live_dependency_sha256() -> str:
    """Commit to the live transitive code used for recorded uniform draws."""
    return _sha256_bytes(
        _canonical_json_bytes(
            {
                "constants": {
                    "implementation_id": HASH_BASED_RNG_IMPLEMENTATION_ID,
                    "unit_scale": _UNIT_SCALE,
                },
                "sources": {
                    "canonical_json": inspect.getsource(_canonical_json_bytes),
                    "draw_digest": inspect.getsource(_draw_digest),
                    "hash_based_rng": inspect.getsource(HashBasedRNG),
                    "legacy_event_key": inspect.getsource(EventKey),
                    "semantic_event_key": inspect.getsource(SemanticEventKey),
                    "uniform_from_bits": inspect.getsource(_uniform_from_bits),
                },
            }
        )
    )


@dataclass(frozen=True, slots=True)
class ScenarioGenerationDraw:
    """Exact semantic identity and binary64 output of one generator draw."""

    event_key: SemanticEventKey
    domain: str
    draw_index: int
    uniform_bits: int
    output_binary64_hex: str
    schema_version: str = SCENARIO_GENERATION_DRAW_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self) is not ScenarioGenerationDraw:
            _fail("scenario generation draws must use the exact verifier type")
        if type(self.event_key) is not SemanticEventKey:
            _fail("scenario generation draw requires a SemanticEventKey")
        if type(self.domain) is not str or type(self.output_binary64_hex) is not str:
            _fail("scenario generation draw text fields must use exact strings")
        if type(self.draw_index) is not int or type(self.uniform_bits) is not int:
            _fail("scenario generation draw integer fields must use exact integers")
        _identifier(self.domain, "domain")
        _bounded_draw_index(self.draw_index, "draw_index")
        if (
            isinstance(self.uniform_bits, bool)
            or not isinstance(self.uniform_bits, int)
            or not 0 <= self.uniform_bits < _UNIT_SCALE
        ):
            _fail("uniform_bits must be a 53-bit non-negative integer")
        expected_hex = _uniform_from_bits(self.uniform_bits).hex()
        if self.output_binary64_hex != expected_hex:
            _fail("draw output bits do not match uniform_bits")
        if self.schema_version != SCENARIO_GENERATION_DRAW_SCHEMA_VERSION:
            _fail("unsupported scenario generation draw schema version")

    def to_mapping(self) -> dict[str, object]:
        """Return the complete portable draw manifest row."""
        return {
            "domain": self.domain,
            "draw_index": self.draw_index,
            "event_key": self.event_key.to_mapping(),
            "output_binary64_hex": self.output_binary64_hex,
            "schema_version": self.schema_version,
            "uniform_bits": self.uniform_bits,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        """Strictly reconstruct one draw-manifest row."""
        _strict_fields(
            value,
            frozenset(
                {
                    "domain",
                    "draw_index",
                    "event_key",
                    "output_binary64_hex",
                    "schema_version",
                    "uniform_bits",
                }
            ),
            "scenario generation draw",
        )
        schema_version = value["schema_version"]
        output_hex = value["output_binary64_hex"]
        if not isinstance(schema_version, str) or not isinstance(output_hex, str):
            _fail("scenario generation draw string fields have the wrong type")
        return cls(
            event_key=SemanticEventKey.from_mapping(
                _require_mapping(value["event_key"], "draw event_key")
            ),
            domain=_identifier(value["domain"], "domain"),
            draw_index=_bounded_draw_index(value["draw_index"], "draw_index"),
            uniform_bits=_non_negative_int(value["uniform_bits"], "uniform_bits"),
            output_binary64_hex=output_hex,
            schema_version=schema_version,
        )


_SCENARIO_GENERATION_RECEIPT_ISSUER = object()


@dataclass(frozen=True, slots=True)
class ScenarioGenerationReceipt:
    """Persistable proof of a frozen synthetic shock-generator execution."""

    root_event_key: EventKey
    generator_spec: ScenarioGeneratorSpec
    rng_implementation_id: str
    rng_dependency_sha256: str
    rng_seed_sha256: str
    draws: tuple[ScenarioGenerationDraw, ...]
    scenario_id: str
    shock_tape_sha256: str
    _issuer: InitVar[object]
    synthetic_only: Literal[True] = True
    schema_version: str = SCENARIO_GENERATION_RECEIPT_SCHEMA_VERSION

    def __post_init__(self, _issuer: object) -> None:
        if type(self) is not ScenarioGenerationReceipt:
            _fail("scenario generation receipts must use the exact verifier type")
        if _issuer is not _SCENARIO_GENERATION_RECEIPT_ISSUER:
            _fail("scenario generation receipts are verifier-issued")
        _legacy_event_key_mapping(self.root_event_key)
        if type(self.root_event_key) is not EventKey:
            _fail("scenario generation receipt root must use the exact EventKey type")
        if type(self.generator_spec) is not ScenarioGeneratorSpec:
            _fail("scenario generation receipt requires a generator spec")
        if type(self.draws) is not tuple:
            _fail("scenario generation receipt draws must use an exact tuple")
        if any(
            type(getattr(self, field)) is not str
            for field in (
                "rng_implementation_id",
                "rng_dependency_sha256",
                "rng_seed_sha256",
                "scenario_id",
                "shock_tape_sha256",
                "schema_version",
            )
        ):
            _fail("scenario generation receipt text fields must use exact strings")
        if self.rng_implementation_id != HASH_BASED_RNG_IMPLEMENTATION_ID:
            _fail("scenario generation receipt RNG implementation is not live")
        for field in (
            "rng_dependency_sha256",
            "rng_seed_sha256",
            "shock_tape_sha256",
        ):
            digest = getattr(self, field)
            if (
                not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            ):
                _fail(f"{field} must be a lowercase SHA-256 digest")
        _identifier(self.scenario_id, "scenario_id")
        if self.synthetic_only is not True:
            _fail("scenario generation receipts must be synthetic_only")
        if self.schema_version != SCENARIO_GENERATION_RECEIPT_SCHEMA_VERSION:
            _fail("unsupported scenario generation receipt schema version")
        self.validate_live()

    def validate_live(self) -> None:
        """Fail closed if the receipt no longer reproduces under live RNG code."""
        if self.rng_dependency_sha256 != hash_based_rng_live_dependency_sha256():
            _fail("scenario generation receipt RNG dependency commitment is stale")
        rng = HashBasedRNG(self.root_event_key.value)
        if self.rng_seed_sha256 != rng.seed_sha256:
            _fail("scenario generation receipt seed commitment does not match root")
        if (
            not isinstance(self.draws, tuple)
            or not self.draws
            or len(self.draws) > MAX_SCENARIO_GENERATION_DRAWS
        ):
            _fail("scenario generation draws violate the frozen resource bound")
        if not all(type(item) is ScenarioGenerationDraw for item in self.draws):
            _fail("scenario generation receipt draws have the wrong type")
        identities: set[tuple[str, str, int]] = set()
        for draw in self.draws:
            if draw.event_key.scenario_id != self.scenario_id:
                _fail("scenario generation draw belongs to another scenario")
            identity = (draw.event_key.value, draw.domain, draw.draw_index)
            if identity in identities:
                _fail("scenario generation receipt contains a duplicate draw")
            identities.add(identity)
            expected_bits = rng.uniform_bits(
                draw.event_key,
                domain=draw.domain,
                draw_index=draw.draw_index,
            )
            if draw.uniform_bits != expected_bits:
                _fail("scenario generation draw does not reproduce from root key")

    def _signed_mapping(self) -> dict[str, object]:
        return {
            "draws": [item.to_mapping() for item in self.draws],
            "generator_spec": self.generator_spec.to_mapping(),
            "generator_spec_sha256": self.generator_spec.sha256,
            "rng_dependency_sha256": self.rng_dependency_sha256,
            "rng_implementation_id": self.rng_implementation_id,
            "rng_seed_sha256": self.rng_seed_sha256,
            "root_event_key": _legacy_event_key_mapping(self.root_event_key),
            "scenario_id": self.scenario_id,
            "schema_version": self.schema_version,
            "shock_tape_sha256": self.shock_tape_sha256,
            "synthetic_only": self.synthetic_only,
        }

    @property
    def verifier_seal_sha256(self) -> str:
        """Return an unkeyed deterministic content seal, not an attestation."""
        return _sha256_bytes(
            b"policytwin-scenario-generation-receipt-v1\0"
            + _canonical_json_bytes(self._signed_mapping())
        )

    def to_mapping(self) -> dict[str, object]:
        """Return the strict persistent receipt representation."""
        return {
            **self._signed_mapping(),
            "verifier_seal_sha256": self.verifier_seal_sha256,
        }

    @property
    def canonical_bytes(self) -> bytes:
        """Return canonical bytes for protocol input commitments."""
        return _canonical_json_bytes(self.to_mapping())

    @property
    def sha256(self) -> str:
        """Return the complete receipt commitment."""
        return _sha256_bytes(self.canonical_bytes)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        """Strictly reconstruct, reseal-check, and live-replay a receipt."""
        _strict_fields(
            value,
            frozenset(
                {
                    "draws",
                    "generator_spec",
                    "generator_spec_sha256",
                    "rng_dependency_sha256",
                    "rng_implementation_id",
                    "rng_seed_sha256",
                    "root_event_key",
                    "scenario_id",
                    "schema_version",
                    "shock_tape_sha256",
                    "synthetic_only",
                    "verifier_seal_sha256",
                }
            ),
            "scenario generation receipt",
        )
        raw_draws = value["draws"]
        if (
            not isinstance(raw_draws, Sequence)
            or isinstance(raw_draws, str | bytes | bytearray)
            or not raw_draws
            or len(raw_draws) > MAX_SCENARIO_GENERATION_DRAWS
        ):
            _fail("scenario generation receipt draws violate the resource bound")
        if value["synthetic_only"] is not True:
            _fail("scenario generation receipt must be explicitly synthetic_only")
        string_fields = (
            "generator_spec_sha256",
            "rng_dependency_sha256",
            "rng_implementation_id",
            "rng_seed_sha256",
            "scenario_id",
            "schema_version",
            "shock_tape_sha256",
            "verifier_seal_sha256",
        )
        if any(not isinstance(value[field], str) for field in string_fields):
            _fail("scenario generation receipt string fields have the wrong type")
        spec = ScenarioGeneratorSpec.from_mapping(
            _require_mapping(value["generator_spec"], "generator_spec")
        )
        if value["generator_spec_sha256"] != spec.sha256:
            _fail("scenario generation receipt generator spec hash is invalid")
        result = cls(
            root_event_key=_legacy_event_key_from_mapping(
                _require_mapping(value["root_event_key"], "root_event_key")
            ),
            generator_spec=spec,
            rng_implementation_id=cast(str, value["rng_implementation_id"]),
            rng_dependency_sha256=cast(str, value["rng_dependency_sha256"]),
            rng_seed_sha256=cast(str, value["rng_seed_sha256"]),
            draws=tuple(
                ScenarioGenerationDraw.from_mapping(
                    _require_mapping(item, "scenario generation draw")
                )
                for item in raw_draws
            ),
            scenario_id=cast(str, value["scenario_id"]),
            shock_tape_sha256=cast(str, value["shock_tape_sha256"]),
            _issuer=_SCENARIO_GENERATION_RECEIPT_ISSUER,
            synthetic_only=True,
            schema_version=cast(str, value["schema_version"]),
        )
        if value["verifier_seal_sha256"] != result.verifier_seal_sha256:
            _fail("scenario generation receipt deterministic seal is invalid")
        return result


def _issue_scenario_generation_receipt(
    *,
    root_event_key: EventKey,
    generator_spec: ScenarioGeneratorSpec,
    draws: tuple[ScenarioGenerationDraw, ...],
    scenario_id: str,
    shock_tape_sha256: str,
) -> ScenarioGenerationReceipt:
    """Issue a receipt after exact keyed draws and the tape are materialized."""
    rng = HashBasedRNG(root_event_key.value)
    return ScenarioGenerationReceipt(
        root_event_key=root_event_key,
        generator_spec=generator_spec,
        rng_implementation_id=HASH_BASED_RNG_IMPLEMENTATION_ID,
        rng_dependency_sha256=hash_based_rng_live_dependency_sha256(),
        rng_seed_sha256=rng.seed_sha256,
        draws=draws,
        scenario_id=scenario_id,
        shock_tape_sha256=shock_tape_sha256,
        _issuer=_SCENARIO_GENERATION_RECEIPT_ISSUER,
    )


_SCENARIO_EVENT_RECEIPT_ISSUER = object()


@dataclass(frozen=True, slots=True)
class ScenarioEventReceipt:
    """Engine-issued proof that future information is simulated, not observed.

    The receipt is intentionally role-neutral.  For the same semantic shock,
    payload, and frozen generator, baseline and candidate worlds receive equal
    receipts even when their endogenous event counts differ.
    """

    scenario_id: str
    event_key: SemanticEventKey
    scenario_time: datetime
    trusted_cutoff: datetime
    generator_spec: ScenarioGeneratorSpec
    payload_sha256: str
    realization_sha256: str
    shock_tape_sha256: str
    transition_implementation_sha256: str
    transition_config_sha256: str
    _issuer: InitVar[object]
    schema_version: str = SCENARIO_EVENT_RECEIPT_SCHEMA_VERSION

    def __post_init__(self, _issuer: object) -> None:
        if _issuer is not _SCENARIO_EVENT_RECEIPT_ISSUER:
            raise ShockAccessError(
                "scenario event receipts can only be issued after an engine transition",
                rule_code=ShockAccessViolationCode.INVALID_SCENARIO_EVENT,
                event_key_value=None,
            )
        _identifier(self.scenario_id, "scenario_id")
        if not isinstance(self.event_key, SemanticEventKey):
            _fail("scenario receipt event_key must be a SemanticEventKey")
        if self.event_key.event_type is not EventType.EXOGENOUS_SHOCK:
            _fail("scenario receipts must originate from an exogenous shock key")
        if self.event_key.scenario_id != self.scenario_id:
            _fail("scenario receipt key is bound to another scenario")
        scenario_time = _aware_utc(self.scenario_time, "scenario_time")
        trusted_cutoff = _aware_utc(self.trusted_cutoff, "trusted_cutoff")
        if scenario_time != self.event_key.timestamp:
            _fail("scenario_time must equal the semantic shock timestamp")
        if scenario_time <= trusted_cutoff:
            _fail("scenario event receipts are reserved for post-cutoff simulation")
        if not isinstance(self.generator_spec, ScenarioGeneratorSpec):
            _fail("scenario receipt requires a ScenarioGeneratorSpec")
        if self.generator_spec.trusted_cutoff != trusted_cutoff:
            _fail("scenario generator and receipt cutoffs differ")
        for field in (
            "payload_sha256",
            "realization_sha256",
            "shock_tape_sha256",
            "transition_implementation_sha256",
            "transition_config_sha256",
        ):
            value = getattr(self, field)
            if (
                not isinstance(value, str)
                or re.fullmatch(r"[0-9a-f]{64}", value) is None
            ):
                _fail(f"{field} must be a lowercase SHA-256 digest")
        object.__setattr__(self, "scenario_time", scenario_time)
        object.__setattr__(self, "trusted_cutoff", trusted_cutoff)
        if self.schema_version != SCENARIO_EVENT_RECEIPT_SCHEMA_VERSION:
            _fail("unsupported scenario-event receipt schema version")

    def _signed_mapping(self) -> dict[str, object]:
        return {
            "event_key": self.event_key.to_mapping(),
            "generator_spec": self.generator_spec.to_mapping(),
            "generator_spec_sha256": self.generator_spec.sha256,
            "payload_sha256": self.payload_sha256,
            "realization_sha256": self.realization_sha256,
            "scenario_id": self.scenario_id,
            "scenario_time": self.scenario_time.isoformat(),
            "schema_version": self.schema_version,
            "shock_tape_sha256": self.shock_tape_sha256,
            "transition_config_sha256": self.transition_config_sha256,
            "transition_implementation_sha256": (self.transition_implementation_sha256),
            "trusted_cutoff": self.trusted_cutoff.isoformat(),
        }

    @property
    def engine_signature_sha256(self) -> str:
        """Return a role-neutral engine commitment to the generated event."""
        return _sha256_bytes(
            b"policytwin-engine-scenario-event-receipt-v1\0"
            + _canonical_json_bytes(self._signed_mapping())
        )

    def to_mapping(self) -> dict[str, object]:
        """Return serializable evidence; it is not an execution capability."""
        return {
            **self._signed_mapping(),
            "engine_signature_sha256": self.engine_signature_sha256,
        }


@dataclass(frozen=True, slots=True)
class ExogenousShockTape:
    """Immutable content-addressed shock tape with key-based lookup.

    Entries are canonicalized by semantic-key commitment, so construction,
    serialization, hashing, and lookup do not depend on caller ordering.
    A tape belongs to one paired ``scenario_id``.
    """

    entries: tuple[ExogenousShock, ...]
    synthetic_only: Literal[True] = True
    schema_version: str = EXOGENOUS_SHOCK_TAPE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self) is not ExogenousShockTape:
            _fail("shock tapes must use the exact verifier type")
        if type(self.entries) is not tuple or not self.entries:
            _fail("shock tape entries must be a non-empty immutable tuple")
        if not all(type(entry) is ExogenousShock for entry in self.entries):
            _fail("shock tape entries must contain only ExogenousShock values")
        if type(self.schema_version) is not str:
            _fail("shock tape schema_version must use an exact string")
        ordered = tuple(sorted(self.entries, key=lambda item: item.event_key.value))
        keys = tuple(entry.event_key.value for entry in ordered)
        if len(set(keys)) != len(keys):
            _fail("shock tape semantic event keys must be unique")
        scenario_ids = {entry.event_key.scenario_id for entry in ordered}
        if len(scenario_ids) != 1:
            _fail("one shock tape cannot mix paired scenario identifiers")
        object.__setattr__(self, "entries", ordered)
        if self.synthetic_only is not True:
            _fail("shock tapes in this module must be synthetic_only")
        if self.schema_version != EXOGENOUS_SHOCK_TAPE_SCHEMA_VERSION:
            _fail("unsupported exogenous shock tape schema version")

    @property
    def scenario_id(self) -> str:
        """Return the single paired scenario identifier on this tape."""
        return self.entries[0].event_key.scenario_id

    @property
    def keys(self) -> tuple[SemanticEventKey, ...]:
        """Return keys in canonical commitment order."""
        return tuple(entry.event_key for entry in self.entries)

    def shock_for(self, event_key: SemanticEventKey) -> ExogenousShock:
        """Look up a shock by full semantic key without advancing state."""
        if not isinstance(event_key, SemanticEventKey):
            _fail("shock lookup requires a SemanticEventKey")
        return self._shock_for_commitment(event_key.value, expected=event_key)

    def _shock_for_commitment(
        self,
        commitment: str,
        *,
        expected: SemanticEventKey | None = None,
    ) -> ExogenousShock:
        _semantic_key_value(commitment)
        for entry in self.entries:
            if entry.event_key.value == commitment:
                if expected is not None and entry.event_key != expected:
                    _fail("semantic key commitment collision detected")
                return entry
        raise KeyError(commitment)

    def __getitem__(self, key: SemanticEventKey | str) -> ExogenousShock:
        if isinstance(key, SemanticEventKey):
            return self.shock_for(key)
        return self._shock_for_commitment(_semantic_key_value(key))

    def __len__(self) -> int:
        return len(self.entries)

    def to_mapping(self) -> dict[str, object]:
        """Return the canonical order-independent tape representation."""
        return {
            "entries": [entry.to_mapping() for entry in self.entries],
            "schema_version": self.schema_version,
            "synthetic_only": self.synthetic_only,
        }

    @property
    def canonical_bytes(self) -> bytes:
        """Return canonical tape bytes for artifacts and replay."""
        return _canonical_json_bytes(self.to_mapping())

    @property
    def sha256(self) -> str:
        """Return the semantic tape commitment."""
        return _sha256_bytes(self.canonical_bytes)

    @classmethod
    def generate(
        cls,
        keys: tuple[SemanticEventKey, ...],
        *,
        rng: HashBasedRNG,
        field: str = "magnitude",
        lower: float = -1.0,
        upper: float = 1.0,
        domain: str = "exogenous_shock",
    ) -> Self:
        """Generate one scalar shock per key without input-order dependence."""
        if not isinstance(keys, tuple) or not keys:
            _fail("keys must be a non-empty immutable tuple")
        if not all(isinstance(key, SemanticEventKey) for key in keys):
            _fail("keys must contain only SemanticEventKey values")
        if not isinstance(rng, HashBasedRNG):
            _fail("rng must be an explicit HashBasedRNG")
        stable_field = _identifier(field, "field")
        stable_domain = _identifier(domain, "domain")
        normalized_lower = _finite(lower, "lower")
        normalized_upper = _finite(upper, "upper")
        if normalized_lower >= normalized_upper:
            _fail("lower must be strictly below upper")
        shocks: list[ExogenousShock] = []
        for key in keys:
            if key.event_type is not EventType.EXOGENOUS_SHOCK:
                _fail("all generated shock keys must use EXOGENOUS_SHOCK")
            unit = rng.uniform(key, domain=stable_domain)
            magnitude = normalized_lower + (normalized_upper - normalized_lower) * unit
            shocks.append(
                ExogenousShock(
                    event_key=key,
                    values=((stable_field, magnitude),),
                    synthetic_only=True,
                )
            )
        return cls(entries=tuple(shocks), synthetic_only=True)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        """Strictly reconstruct and canonicalize a shock tape."""
        _strict_fields(
            value,
            frozenset({"entries", "schema_version", "synthetic_only"}),
            "exogenous shock tape",
        )
        if value["synthetic_only"] is not True:
            _fail("shock tape must be explicitly synthetic_only")
        schema_version = value["schema_version"]
        if not isinstance(schema_version, str):
            _fail("schema_version must be a string")
        raw_entries = value["entries"]
        if not isinstance(raw_entries, Sequence) or isinstance(
            raw_entries, str | bytes | bytearray
        ):
            _fail("shock tape entries must be an array")
        return cls(
            entries=tuple(
                ExogenousShock.from_mapping(
                    _require_mapping(raw_entry, "shock tape entry")
                )
                for raw_entry in raw_entries
            ),
            synthetic_only=True,
            schema_version=schema_version,
        )


def validate_exact_scenario_generation_receipt(
    receipt: ScenarioGenerationReceipt,
) -> None:
    """Validate an exact built-in receipt by strict reconstruction and replay."""
    if type(receipt) is not ScenarioGenerationReceipt:
        _fail("formal scenario generation evidence requires the exact receipt type")
    if type(receipt.root_event_key) is not EventKey:
        _fail("formal scenario generation evidence requires the exact root key type")
    if type(receipt.generator_spec) is not ScenarioGeneratorSpec:
        _fail("formal scenario generation evidence requires the exact spec type")
    spec = receipt.generator_spec
    if any(
        type(getattr(spec, field)) is not str
        for field in (
            "generator_id",
            "generator_config_sha256",
            "generator_implementation_sha256",
            "source_lineage_sha256",
            "cutoff_policy_sha256",
        )
    ) or any(
        type(getattr(spec, field)) is not datetime
        for field in ("frozen_at", "trusted_cutoff")
    ):
        _fail("formal scenario generation evidence has non-exact spec values")
    if type(receipt.draws) is not tuple or not all(
        type(draw) is ScenarioGenerationDraw
        and type(draw.event_key) is SemanticEventKey
        for draw in receipt.draws
    ):
        _fail("formal scenario generation evidence has derived nested draw types")
    if any(
        type(getattr(receipt, field)) is not str
        for field in (
            "rng_implementation_id",
            "rng_dependency_sha256",
            "rng_seed_sha256",
            "scenario_id",
            "shock_tape_sha256",
            "schema_version",
        )
    ):
        _fail("formal scenario generation evidence has non-exact receipt values")
    for draw in receipt.draws:
        key = draw.event_key
        if (
            type(draw.domain) is not str
            or type(draw.draw_index) is not int
            or type(draw.uniform_bits) is not int
            or type(draw.output_binary64_hex) is not str
            or type(draw.schema_version) is not str
            or type(key.scenario_id) is not str
            or type(key.event_type) is not EventType
            or type(key.entity_id) is not str
            or type(key.timestamp) is not datetime
            or (key.agent_id is not None and type(key.agent_id) is not str)
            or type(key.schema_version) is not str
        ):
            _fail("formal scenario generation evidence has non-exact draw values")
    receipt.validate_live()
    mapping = receipt.to_mapping()
    canonical = _canonical_json_bytes(mapping)
    reconstructed = ScenarioGenerationReceipt.from_mapping(mapping)
    if (
        type(reconstructed) is not ScenarioGenerationReceipt
        or reconstructed != receipt
        or reconstructed.to_mapping() != mapping
        or reconstructed.canonical_bytes != canonical
        or receipt.canonical_bytes != canonical
        or receipt.sha256 != _sha256_bytes(canonical)
    ):
        _fail("scenario generation receipt failed strict canonical reconstruction")


def validate_exact_exogenous_shock_tape(tape: ExogenousShockTape) -> None:
    """Validate exact built-in tape and nested values without polymorphic hooks."""
    if type(tape) is not ExogenousShockTape:
        _fail("formal shock evidence requires the exact ExogenousShockTape type")
    if (
        type(tape.entries) is not tuple
        or type(tape.schema_version) is not str
        or tape.synthetic_only is not True
    ):
        _fail("formal shock evidence contains non-exact tape values")
    for shock in tape.entries:
        if type(shock) is not ExogenousShock:
            _fail("formal shock evidence contains a derived shock type")
        if type(shock.event_key) is not SemanticEventKey:
            _fail("formal shock evidence contains a derived semantic key type")
        key = shock.event_key
        if (
            type(key.scenario_id) is not str
            or type(key.event_type) is not EventType
            or type(key.entity_id) is not str
            or type(key.timestamp) is not datetime
            or (key.agent_id is not None and type(key.agent_id) is not str)
            or type(key.schema_version) is not str
            or shock.synthetic_only is not True
        ):
            _fail("formal shock evidence contains non-exact shock values")
        if type(shock.values) is not tuple or not all(
            type(item) is tuple
            and len(item) == 2
            and type(item[0]) is str
            and type(item[1]) is float
            for item in shock.values
        ):
            _fail("formal shock evidence contains non-canonical nested values")
    mapping = tape.to_mapping()
    canonical = _canonical_json_bytes(mapping)
    reconstructed = ExogenousShockTape.from_mapping(mapping)
    if (
        type(reconstructed) is not ExogenousShockTape
        or reconstructed != tape
        or reconstructed.to_mapping() != mapping
        or reconstructed.canonical_bytes != canonical
        or tape.canonical_bytes != canonical
        or tape.sha256 != _sha256_bytes(canonical)
    ):
        _fail("shock tape failed strict canonical reconstruction")


@dataclass(frozen=True, slots=True)
class ShockAccessViolation:
    """One immutable access violation without caller-supplied free text."""

    rule_code: ShockAccessViolationCode
    event_key_value: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.rule_code, ShockAccessViolationCode):
            _fail("shock access rule_code must be a ShockAccessViolationCode")
        if self.event_key_value is not None:
            _semantic_key_value(self.event_key_value)

    def to_mapping(self) -> dict[str, object]:
        """Return the stable machine-readable violation fields."""
        return {
            "event_key_value": self.event_key_value,
            "rule_code": self.rule_code.value,
        }


@dataclass(frozen=True, slots=True)
class ShockConsumption:
    """One realization actually obtained through an engine-issued capability."""

    shock: ExogenousShock

    def __post_init__(self) -> None:
        if not isinstance(self.shock, ExogenousShock):
            _fail("shock consumption must contain an ExogenousShock")

    @property
    def event_key(self) -> SemanticEventKey:
        """Return the consumed semantic event identity."""
        return self.shock.event_key

    @property
    def realization_sha256(self) -> str:
        """Commit to the complete key and realized vector."""
        return _sha256_bytes(_canonical_json_bytes(self.shock.to_mapping()))

    def to_mapping(self) -> dict[str, object]:
        """Return the realization plus its independently checkable commitment."""
        return {
            "realization_sha256": self.realization_sha256,
            "shock": self.shock.to_mapping(),
        }


@dataclass(frozen=True, slots=True)
class ShockConsumptionLedger:
    """Engine-owned retrieval and replayable transition evidence for one world."""

    world_id: str
    role: Literal["baseline", "candidate"]
    scenario_id: str
    shock_tape_sha256: str
    declared_event_keys: tuple[str, ...]
    transition_config: ShockTransitionConfig
    initial_transition_state: ShockTransitionState
    consumptions: tuple[ShockConsumption, ...]
    transition_receipts: tuple[ShockTransitionReceipt, ...]
    violations: tuple[ShockAccessViolation, ...]
    scenario_event_receipts: tuple[ScenarioEventReceipt, ...] = ()
    schema_version: str = SHOCK_CONSUMPTION_LEDGER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _identifier(self.world_id, "world_id")
        if self.role not in ("baseline", "candidate"):
            _fail("ledger role must be baseline or candidate")
        _identifier(self.scenario_id, "scenario_id")
        if re.fullmatch(r"[0-9a-f]{64}", self.shock_tape_sha256) is None:
            _fail("shock_tape_sha256 must be a lowercase SHA-256 digest")
        if self.schema_version != SHOCK_CONSUMPTION_LEDGER_SCHEMA_VERSION:
            _fail("unsupported shock consumption ledger schema version")
        declared = tuple(sorted(self.declared_event_keys))
        if not declared:
            _fail("declared_event_keys must not be empty")
        if any(_SEMANTIC_KEY_PATTERN.fullmatch(item) is None for item in declared):
            _fail("declared_event_keys contains an invalid commitment")
        if len(set(declared)) != len(declared):
            _fail("declared_event_keys must be unique")
        object.__setattr__(self, "declared_event_keys", declared)
        if not isinstance(self.transition_config, ShockTransitionConfig):
            _fail("transition_config has the wrong type")
        if not isinstance(self.initial_transition_state, ShockTransitionState):
            _fail("initial_transition_state has the wrong type")
        if not all(isinstance(item, ShockConsumption) for item in self.consumptions):
            _fail("consumptions must contain only ShockConsumption values")
        consumptions = tuple(
            sorted(self.consumptions, key=lambda item: item.event_key.value)
        )
        consumed_keys = tuple(item.event_key.value for item in consumptions)
        if len(set(consumed_keys)) != len(consumed_keys):
            _fail("consumption ledger event keys must be unique")
        if any(item.event_key.scenario_id != self.scenario_id for item in consumptions):
            _fail("consumption ledger cannot mix scenario identifiers")
        object.__setattr__(self, "consumptions", consumptions)
        if not all(
            isinstance(item, ShockTransitionReceipt)
            for item in self.transition_receipts
        ):
            _fail("transition_receipts has the wrong item type")
        receipt_keys = tuple(item.event_key.value for item in self.transition_receipts)
        if len(set(receipt_keys)) != len(receipt_keys):
            _fail("transition receipt event keys must be unique")
        if any(
            item.world_id != self.world_id
            or item.role != self.role
            or item.scenario_id != self.scenario_id
            for item in self.transition_receipts
        ):
            _fail("transition receipt is bound to the wrong world or scenario")
        if not isinstance(self.scenario_event_receipts, tuple) or not all(
            isinstance(item, ScenarioEventReceipt)
            for item in self.scenario_event_receipts
        ):
            _fail("scenario_event_receipts has the wrong item type")
        scenario_receipts = tuple(
            sorted(
                self.scenario_event_receipts,
                key=lambda item: item.event_key.value,
            )
        )
        scenario_keys = tuple(item.event_key.value for item in scenario_receipts)
        if len(set(scenario_keys)) != len(scenario_keys):
            _fail("scenario event receipt keys must be unique")
        if any(item.scenario_id != self.scenario_id for item in scenario_receipts):
            _fail("scenario event receipts cannot mix scenario identifiers")
        if (
            scenario_receipts
            and len({item.generator_spec.sha256 for item in scenario_receipts}) != 1
        ):
            _fail("one ledger cannot mix scenario generator specifications")
        object.__setattr__(self, "scenario_event_receipts", scenario_receipts)
        if not all(isinstance(item, ShockAccessViolation) for item in self.violations):
            _fail("violations must contain only ShockAccessViolation values")
        violations = tuple(
            sorted(
                self.violations,
                key=lambda item: (item.rule_code.value, item.event_key_value or ""),
            )
        )
        object.__setattr__(self, "violations", violations)

    @property
    def complete(self) -> bool:
        """Whether every event has one valid, deterministic state transition."""
        consumed = tuple(item.event_key.value for item in self.consumptions)
        return (
            consumed == self.declared_event_keys
            and not self.violations
            and self.verify_transitions()
            and self.verify_scenario_events()
        )

    @property
    def transition_schedule(self) -> tuple[str, ...]:
        """Return the declared semantic keys in deterministic event-time order."""
        keys = {item.event_key.value: item.event_key for item in self.consumptions}
        if set(keys) != set(self.declared_event_keys):
            return ()
        return tuple(
            item.value
            for item in sorted(
                keys.values(),
                key=lambda item: (item.timestamp, item.value),
            )
        )

    def verify_transitions(self) -> bool:
        """Replay every receipt from its typed pre-state and realization."""
        schedule = self.transition_schedule
        if not schedule or len(self.transition_receipts) != len(schedule):
            return False
        by_key = {item.event_key.value: item for item in self.consumptions}
        state = self.initial_transition_state
        for sequence, (expected_key, receipt) in enumerate(
            zip(schedule, self.transition_receipts, strict=True)
        ):
            consumption = by_key[expected_key]
            if (
                receipt.sequence != sequence
                or receipt.event_key.value != expected_key
                or receipt.realization_sha256 != consumption.realization_sha256
                or receipt.pre_state != state
                or receipt.pre_state_sha256 != state.sha256
                or receipt.transition_implementation_sha256
                != self.transition_config.implementation_sha256
                or receipt.transition_config_sha256 != self.transition_config.sha256
            ):
                return False
            try:
                replayed = self.transition_config.apply(state, consumption.shock)
            except (KeyError, SemanticEventError):
                return False
            if (
                replayed != receipt.post_state
                or replayed.sha256 != receipt.post_state_sha256
            ):
                return False
            state = replayed
        return True

    def verify_scenario_events(self) -> bool:
        """Verify an all-or-none future payload receipt set against transitions.

        A capability that does not generate future information has no scenario
        receipts and remains valid.  Once it enters scenario-event mode, every
        declared shock must have exactly one receipt.  This prevents both
        worlds from silently omitting an inconvenient future event while still
        allowing ordinary pre-cutoff paired runners to use the same ledger.
        """
        if not self.scenario_event_receipts:
            return True
        receipt_keys = tuple(
            item.event_key.value for item in self.scenario_event_receipts
        )
        if receipt_keys != self.declared_event_keys:
            return False
        transitions = {item.event_key.value: item for item in self.transition_receipts}
        for receipt in self.scenario_event_receipts:
            transition = transitions.get(receipt.event_key.value)
            if transition is None:
                return False
            if (
                receipt.scenario_id != self.scenario_id
                or receipt.shock_tape_sha256 != self.shock_tape_sha256
                or receipt.realization_sha256 != transition.realization_sha256
                or receipt.transition_implementation_sha256
                != transition.transition_implementation_sha256
                or receipt.transition_config_sha256
                != transition.transition_config_sha256
                or receipt.generator_spec.trusted_cutoff != receipt.trusted_cutoff
                or receipt.scenario_time != receipt.event_key.timestamp
                or receipt.scenario_time <= receipt.trusted_cutoff
            ):
                return False
        return True

    @property
    def final_transition_state(self) -> ShockTransitionState:
        """Return the last engine-issued post-state, or the frozen opening state."""
        if not self.transition_receipts:
            return self.initial_transition_state
        return self.transition_receipts[-1].post_state

    @property
    def realizations(self) -> tuple[ExogenousShock, ...]:
        """Return consumed realizations in canonical semantic-key order."""
        return tuple(item.shock for item in self.consumptions)

    def to_mapping(self) -> dict[str, object]:
        """Return canonical, scheduling-independent engine evidence."""
        return {
            "complete": self.complete,
            "consumptions": [item.to_mapping() for item in self.consumptions],
            "declared_event_keys": list(self.declared_event_keys),
            "initial_transition_state": self.initial_transition_state.to_mapping(),
            "initial_transition_state_sha256": self.initial_transition_state.sha256,
            "final_transition_state_sha256": self.final_transition_state.sha256,
            "role": self.role,
            "scenario_id": self.scenario_id,
            "scenario_event_receipts": [
                item.to_mapping() for item in self.scenario_event_receipts
            ],
            "scenario_events_verified": self.verify_scenario_events(),
            "schema_version": self.schema_version,
            "shock_tape_sha256": self.shock_tape_sha256,
            "transition_config": self.transition_config.to_mapping(),
            "transition_config_sha256": self.transition_config.sha256,
            "transition_receipts": [
                item.to_mapping() for item in self.transition_receipts
            ],
            "transitions_verified": self.verify_transitions(),
            "violations": [item.to_mapping() for item in self.violations],
            "world_id": self.world_id,
        }

    @property
    def canonical_bytes(self) -> bytes:
        """Return deterministic bytes for replay comparison."""
        return _canonical_json_bytes(self.to_mapping())

    @property
    def sha256(self) -> str:
        """Return the ledger content commitment."""
        return _sha256_bytes(self.canonical_bytes)


_SHOCK_ACCESS_ISSUER = object()


class ShockAccessCapability:
    """Single-world, single-use access to a frozen exogenous tape.

    Construction and finalization are module-controlled.  Python does not
    provide a formal security boundary against code with arbitrary process
    introspection; the capability instead removes runner-authored consumption
    declarations from the integrity decision and fails closed on ordinary API
    misuse.
    """

    __slots__ = (
        "_closed",
        "_consumed",
        "_current_transition_state",
        "_initial_transition_state",
        "_lock",
        "_role",
        "_scenario_event_receipts",
        "_tape",
        "_transition_config",
        "_transition_receipts",
        "_transition_schedule",
        "_violations",
        "_world_id",
    )

    def __init__(
        self,
        tape: ExogenousShockTape,
        world_id: str,
        *,
        role: Literal["baseline", "candidate"] | None = None,
        transition_config: ShockTransitionConfig | None = None,
        initial_transition_state: ShockTransitionState | None = None,
        _issuer: object,
    ) -> None:
        if _issuer is not _SHOCK_ACCESS_ISSUER:
            raise ShockAccessError(
                "shock capabilities can only be issued by the pair engine",
                rule_code=ShockAccessViolationCode.INVALID_KEY,
                event_key_value=None,
            )
        if not isinstance(tape, ExogenousShockTape):
            _fail("shock capability requires an ExogenousShockTape")
        if role not in ("baseline", "candidate"):
            _fail("shock capability role must be baseline or candidate")
        if not isinstance(transition_config, ShockTransitionConfig):
            _fail("shock capability requires a ShockTransitionConfig")
        if not isinstance(initial_transition_state, ShockTransitionState):
            _fail("shock capability requires an initial ShockTransitionState")
        self._tape = tape
        self._world_id = _identifier(world_id, "world_id")
        self._role: Literal["baseline", "candidate"] = role
        self._transition_config = transition_config
        self._initial_transition_state = initial_transition_state
        self._current_transition_state = initial_transition_state
        self._transition_schedule = tuple(
            sorted(tape.keys, key=lambda item: (item.timestamp, item.value))
        )
        self._consumed: dict[str, ExogenousShock] = {}
        self._transition_receipts: list[ShockTransitionReceipt] = []
        self._scenario_event_receipts: dict[str, ScenarioEventReceipt] = {}
        self._violations: list[ShockAccessViolation] = []
        self._closed = False
        self._lock = Lock()

    @property
    def scenario_id(self) -> str:
        """Return the scenario to which this capability is bound."""
        return self._tape.scenario_id

    @property
    def sha256(self) -> str:
        """Return the underlying tape commitment without exposing its values."""
        return self._tape.sha256

    @property
    def keys(self) -> tuple[SemanticEventKey, ...]:
        """Return the declared semantic keys without their realized values."""
        return self._tape.keys

    @property
    def transition_schedule(self) -> tuple[SemanticEventKey, ...]:
        """Return the required engine transition order by event time and key."""
        return self._transition_schedule

    @property
    def initial_transition_state(self) -> ShockTransitionState:
        """Return the immutable opening state for the engine-owned reducer."""
        return self._initial_transition_state

    @property
    def current_transition_state(self) -> ShockTransitionState:
        """Return the latest engine-owned state snapshot."""
        with self._lock:
            return self._current_transition_state

    def _reject(
        self,
        rule_code: ShockAccessViolationCode,
        event_key_value: str | None,
        message: str,
    ) -> NoReturn:
        self._violations.append(
            ShockAccessViolation(
                rule_code=rule_code,
                event_key_value=event_key_value,
            )
        )
        raise ShockAccessError(
            message,
            rule_code=rule_code,
            event_key_value=event_key_value,
        )

    def consume(self, event_key: SemanticEventKey) -> ExogenousShock:
        """Legacy retrieval only; a pair cannot pass without applying it."""
        with self._lock:
            if not isinstance(event_key, SemanticEventKey):
                self._reject(
                    ShockAccessViolationCode.INVALID_KEY,
                    None,
                    "shock access requires a SemanticEventKey",
                )
            commitment = event_key.value
            if self._closed:
                self._reject(
                    ShockAccessViolationCode.CAPABILITY_CLOSED,
                    commitment,
                    "shock capability is closed",
                )
            if commitment in self._consumed:
                self._reject(
                    ShockAccessViolationCode.DUPLICATE_CONSUMPTION,
                    commitment,
                    "semantic shock was consumed more than once",
                )
            try:
                shock = self._tape.shock_for(event_key)
            except KeyError:
                self._reject(
                    ShockAccessViolationCode.UNDECLARED_EVENT,
                    commitment,
                    "semantic shock is not declared on this tape",
                )
            self._consumed[commitment] = shock
            return shock

    def apply_transition(
        self,
        event_key: SemanticEventKey,
        pre_state: ShockTransitionState,
    ) -> ShockTransitionResult:
        """Apply and sign one frozen transition inside the engine capability.

        The caller supplies the state it believes is current.  The capability
        compares it with its private state, enforces event-time order, obtains
        the realization, runs the closed reducer, and issues the receipt in one
        locked operation.  A runner cannot inject a receipt into this ledger.
        """
        with self._lock:
            if not isinstance(event_key, SemanticEventKey):
                self._reject(
                    ShockAccessViolationCode.INVALID_KEY,
                    None,
                    "shock transition requires a SemanticEventKey",
                )
            commitment = event_key.value
            if self._closed:
                self._reject(
                    ShockAccessViolationCode.CAPABILITY_CLOSED,
                    commitment,
                    "shock capability is closed",
                )
            if not isinstance(pre_state, ShockTransitionState):
                self._reject(
                    ShockAccessViolationCode.WRONG_PRE_STATE,
                    commitment,
                    "shock transition pre-state has the wrong type",
                )
            if pre_state != self._current_transition_state:
                self._reject(
                    ShockAccessViolationCode.WRONG_PRE_STATE,
                    commitment,
                    "shock transition pre-state does not match engine state",
                )
            sequence = len(self._transition_receipts)
            if sequence >= len(self._transition_schedule):
                self._reject(
                    ShockAccessViolationCode.DUPLICATE_CONSUMPTION,
                    commitment,
                    "all declared shock transitions were already applied",
                )
            expected = self._transition_schedule[sequence]
            if event_key != expected:
                self._reject(
                    ShockAccessViolationCode.OUT_OF_ORDER_TRANSITION,
                    commitment,
                    "shock transition does not match the frozen event schedule",
                )
            if commitment in self._consumed:
                self._reject(
                    ShockAccessViolationCode.DUPLICATE_CONSUMPTION,
                    commitment,
                    "semantic shock was consumed before its transition",
                )
            try:
                shock = self._tape.shock_for(event_key)
                post_state = self._transition_config.apply(pre_state, shock)
            except KeyError:
                self._reject(
                    ShockAccessViolationCode.INVALID_KEY,
                    commitment,
                    "transition state or realization lacks a configured field",
                )
            consumption = ShockConsumption(shock=shock)
            receipt = ShockTransitionReceipt(
                sequence=sequence,
                world_id=self._world_id,
                role=self._role,
                scenario_id=self._tape.scenario_id,
                event_key=event_key,
                realization_sha256=consumption.realization_sha256,
                pre_state=pre_state,
                pre_state_sha256=pre_state.sha256,
                transition_implementation_sha256=(
                    self._transition_config.implementation_sha256
                ),
                transition_config_sha256=self._transition_config.sha256,
                post_state=post_state,
                post_state_sha256=post_state.sha256,
                _issuer=_SHOCK_TRANSITION_RECEIPT_ISSUER,
            )
            self._consumed[commitment] = shock
            self._transition_receipts.append(receipt)
            self._current_transition_state = post_state
            return ShockTransitionResult(
                shock=shock,
                post_state=post_state,
                receipt=receipt,
            )

    def issue_scenario_event_receipt(
        self,
        event_key: SemanticEventKey,
        *,
        generator_spec: ScenarioGeneratorSpec,
        payload_sha256: str,
    ) -> ScenarioEventReceipt:
        """Bind one generated future payload to an applied exogenous shock.

        The operation cannot authorize an observed post-cutoff record.  It only
        attests that a text-free scenario payload was produced by a generator
        frozen no later than the trusted cutoff and from a shock transition
        already executed by this engine capability.
        """
        with self._lock:
            if not isinstance(event_key, SemanticEventKey):
                self._reject(
                    ShockAccessViolationCode.INVALID_SCENARIO_EVENT,
                    None,
                    "scenario event issuance requires a SemanticEventKey",
                )
            commitment = event_key.value
            if self._closed:
                self._reject(
                    ShockAccessViolationCode.CAPABILITY_CLOSED,
                    commitment,
                    "shock capability is closed",
                )
            if not isinstance(generator_spec, ScenarioGeneratorSpec):
                self._reject(
                    ShockAccessViolationCode.INVALID_SCENARIO_EVENT,
                    commitment,
                    "scenario event issuance requires a frozen generator spec",
                )
            if (
                not isinstance(payload_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", payload_sha256) is None
            ):
                self._reject(
                    ShockAccessViolationCode.INVALID_SCENARIO_EVENT,
                    commitment,
                    "scenario payload must be a lowercase SHA-256 digest",
                )
            if event_key.scenario_id != self._tape.scenario_id:
                self._reject(
                    ShockAccessViolationCode.INVALID_SCENARIO_EVENT,
                    commitment,
                    "scenario event key belongs to another paired scenario",
                )
            if event_key.timestamp <= generator_spec.trusted_cutoff:
                self._reject(
                    ShockAccessViolationCode.INVALID_SCENARIO_EVENT,
                    commitment,
                    "scenario event time must be strictly after the trusted cutoff",
                )
            transition_receipt = next(
                (
                    receipt
                    for receipt in self._transition_receipts
                    if receipt.event_key == event_key
                ),
                None,
            )
            if transition_receipt is None:
                self._reject(
                    ShockAccessViolationCode.SCENARIO_EVENT_BEFORE_TRANSITION,
                    commitment,
                    "scenario event cannot be issued before its shock transition",
                )
            if commitment in self._scenario_event_receipts:
                self._reject(
                    ShockAccessViolationCode.DUPLICATE_SCENARIO_EVENT,
                    commitment,
                    "a scenario event was already issued for this shock key",
                )
            receipt = ScenarioEventReceipt(
                scenario_id=self._tape.scenario_id,
                event_key=event_key,
                scenario_time=event_key.timestamp,
                trusted_cutoff=generator_spec.trusted_cutoff,
                generator_spec=generator_spec,
                payload_sha256=payload_sha256,
                realization_sha256=transition_receipt.realization_sha256,
                shock_tape_sha256=self._tape.sha256,
                transition_implementation_sha256=(
                    transition_receipt.transition_implementation_sha256
                ),
                transition_config_sha256=(transition_receipt.transition_config_sha256),
                _issuer=_SCENARIO_EVENT_RECEIPT_ISSUER,
            )
            self._scenario_event_receipts[commitment] = receipt
            return receipt

    def shock_for(self, event_key: SemanticEventKey) -> ExogenousShock:
        """Compatibility spelling for the consuming lookup operation."""
        return self.consume(event_key)

    def _finalize(self, *, _issuer: object) -> ShockConsumptionLedger:
        if _issuer is not _SHOCK_ACCESS_ISSUER:
            raise ShockAccessError(
                "shock capability can only be finalized by the pair engine",
                rule_code=ShockAccessViolationCode.INVALID_KEY,
                event_key_value=None,
            )
        with self._lock:
            if self._closed:
                raise ShockAccessError(
                    "shock capability was already finalized",
                    rule_code=ShockAccessViolationCode.CAPABILITY_CLOSED,
                    event_key_value=None,
                )
            self._closed = True
            transitioned = {item.event_key.value for item in self._transition_receipts}
            violations = list(self._violations)
            for commitment in sorted(set(self._consumed) - transitioned):
                violations.append(
                    ShockAccessViolation(
                        rule_code=ShockAccessViolationCode.UNAPPLIED_REALIZATION,
                        event_key_value=commitment,
                    )
                )
            for key in self._transition_schedule:
                if key.value not in transitioned:
                    violations.append(
                        ShockAccessViolation(
                            rule_code=ShockAccessViolationCode.OMITTED_TRANSITION,
                            event_key_value=key.value,
                        )
                    )
            return ShockConsumptionLedger(
                world_id=self._world_id,
                role=self._role,
                scenario_id=self._tape.scenario_id,
                shock_tape_sha256=self._tape.sha256,
                declared_event_keys=tuple(key.value for key in self._tape.keys),
                transition_config=self._transition_config,
                initial_transition_state=self._initial_transition_state,
                consumptions=tuple(
                    ShockConsumption(shock=shock) for shock in self._consumed.values()
                ),
                transition_receipts=tuple(self._transition_receipts),
                violations=tuple(violations),
                scenario_event_receipts=tuple(self._scenario_event_receipts.values()),
            )


def _issue_shock_access(
    tape: ExogenousShockTape,
    *,
    world_id: str,
    role: Literal["baseline", "candidate"],
    transition_config: ShockTransitionConfig,
    initial_transition_state: ShockTransitionState,
) -> ShockAccessCapability:
    """Issue a capability for one isolated world (pair-engine use only)."""
    return ShockAccessCapability(
        tape,
        world_id,
        role=role,
        transition_config=transition_config,
        initial_transition_state=initial_transition_state,
        _issuer=_SHOCK_ACCESS_ISSUER,
    )


def _finalize_shock_access(
    capability: ShockAccessCapability,
) -> ShockConsumptionLedger:
    """Close one capability and return its immutable engine ledger."""
    if not isinstance(capability, ShockAccessCapability):
        _fail("only a ShockAccessCapability can be finalized")
    return capability._finalize(_issuer=_SHOCK_ACCESS_ISSUER)


__all__ = [
    "EXOGENOUS_SHOCK_TAPE_SCHEMA_VERSION",
    "HASH_BASED_RNG_IMPLEMENTATION_ID",
    "MAX_SCENARIO_GENERATION_DRAWS",
    "SCENARIO_GENERATION_DRAW_SCHEMA_VERSION",
    "SCENARIO_GENERATION_RECEIPT_SCHEMA_VERSION",
    "SEMANTIC_EVENT_KEY_SCHEMA_VERSION",
    "EventType",
    "ExogenousShock",
    "ExogenousShockTape",
    "HashBasedRNG",
    "SCENARIO_EVENT_RECEIPT_SCHEMA_VERSION",
    "SHOCK_CONSUMPTION_LEDGER_SCHEMA_VERSION",
    "SHOCK_TRANSITION_IMPLEMENTATION_ID",
    "SHOCK_TRANSITION_RECEIPT_SCHEMA_VERSION",
    "SemanticEventError",
    "SemanticEventKey",
    "ScenarioEventReceipt",
    "ScenarioGenerationDraw",
    "ScenarioGenerationReceipt",
    "ScenarioGeneratorSpec",
    "ShockAccessCapability",
    "ShockAccessError",
    "ShockAccessViolation",
    "ShockAccessViolationCode",
    "ShockConsumption",
    "ShockConsumptionLedger",
    "ShockTransitionConfig",
    "ShockTransitionReceipt",
    "ShockTransitionResult",
    "ShockTransitionState",
    "hash_integer",
    "hash_based_rng_live_dependency_sha256",
    "hash_uniform",
    "validate_exact_exogenous_shock_tape",
    "validate_exact_scenario_generation_receipt",
]
