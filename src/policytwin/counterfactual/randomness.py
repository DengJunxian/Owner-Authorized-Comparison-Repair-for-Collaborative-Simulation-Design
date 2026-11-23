"""Semantic event keys and stateless deterministic synthetic draws.

The legacy :func:`deterministic_int` function remains byte-for-byte stable for
existing artifacts.  New scientific code must use
:func:`deterministic_int_with_receipt`: its explicit domain and draw index
prevent accidental stream aliasing, while the returned receipt can be
independently recomputed rather than trusted as a self-reported value.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Self, cast

EVENT_KEY_SCHEMA_VERSION = "policytwin.event-key.v1"
RANDOM_DRAW_RECEIPT_SCHEMA_VERSION = "policytwin.random-draw-receipt.v1"
RNG_IMPLEMENTATION_ID = "policytwin.sha256-domain-separated-rejection.v1"
DISCRETE_UNIFORM_DISTRIBUTION = "discrete_uniform_inclusive"
AGENT_ACTION_CHOICE_DOMAIN = "policytwin.agent.order_action_choice.v1"
INTEGRATED_OBSERVATION_NOISE_DOMAIN = "policytwin.integrated.observation_noise.v1"

_DOMAIN_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_MIN_SIGNED_64 = -(1 << 63)
_MAX_SIGNED_64 = (1 << 63) - 1
_MAX_DRAW_INDEX = (1 << 31) - 1
_MAX_REJECTION_ATTEMPTS = 1024
_UINT_64_MASK = (1 << 64) - 1


def _canonical_bytes(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


_RNG_IMPLEMENTATION_SPEC: Mapping[str, object] = {
    "acceptance_rule": "candidate < 2^256 - (2^256 mod span)",
    "attempt_encoding": "unsigned-big-endian-8-bytes",
    "candidate_hash": "sha256",
    "distribution": DISCRETE_UNIFORM_DISTRIBUTION,
    "input_encoding": "canonical-json-utf8",
    "input_prefix_hex": b"policytwin-rng-v1\x00".hex(),
    "output_rule": "lower + candidate mod span",
    "rng_implementation_id": RNG_IMPLEMENTATION_ID,
}
RNG_ALGORITHM_SPEC_COMMITMENT = hashlib.sha256(
    _canonical_bytes(_RNG_IMPLEMENTATION_SPEC)
).hexdigest()
# Assigned after every allowlisted implementation dependency is defined.  The
# declaration allows receipt methods to reference the final module value.
RNG_IMPLEMENTATION_COMMITMENT: str


@dataclass(frozen=True, slots=True)
class EventKey:
    """Stable identity for one potential random event.

    The key contains no world identifier because it names a shock shared by the
    policy and baseline worlds in the smoke experiment.
    """

    experiment_id: str
    seed: str
    replicate_id: int
    simulation_step: int
    purpose: str
    schema_version: str = EVENT_KEY_SCHEMA_VERSION

    def canonical_text(self) -> str:
        """Return the unambiguous representation used to derive the key."""
        return json.dumps(
            {
                "experiment_id": self.experiment_id,
                "purpose": self.purpose,
                "replicate_id": self.replicate_id,
                "schema_version": self.schema_version,
                "seed": self.seed,
                "simulation_step": self.simulation_step,
            },
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def value(self) -> str:
        """Return the content-derived event key stored in artifacts and logs."""
        digest = hashlib.sha256(self.canonical_text().encode("utf-8")).hexdigest()
        return f"ptk1:{digest}"


def _validate_receipted_event_key(event_key: EventKey) -> None:
    if not isinstance(event_key, EventKey):
        raise TypeError("event_key must be an EventKey")
    for name, maximum in (("experiment_id", 256), ("seed", 512), ("purpose", 512)):
        value = getattr(event_key, name)
        if not isinstance(value, str) or not value or len(value) > maximum:
            raise ValueError(f"event_key {name} must be a bounded non-empty string")
    for name in ("replicate_id", "simulation_step"):
        value = getattr(event_key, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or value > _MAX_SIGNED_64
        ):
            raise ValueError(f"event_key {name} must be a non-negative 64-bit integer")
    if event_key.schema_version != EVENT_KEY_SCHEMA_VERSION:
        raise ValueError("event_key schema version is unsupported")


def _event_key_mapping(event_key: EventKey) -> dict[str, object]:
    _validate_receipted_event_key(event_key)
    return {
        "experiment_id": event_key.experiment_id,
        "purpose": event_key.purpose,
        "replicate_id": event_key.replicate_id,
        "schema_version": event_key.schema_version,
        "seed": event_key.seed,
        "simulation_step": event_key.simulation_step,
        "value": event_key.value,
    }


def _parse_event_key(value: object) -> EventKey:
    if not isinstance(value, Mapping):
        raise ValueError("random draw event_key must be a mapping")
    expected = {
        "experiment_id",
        "purpose",
        "replicate_id",
        "schema_version",
        "seed",
        "simulation_step",
        "value",
    }
    if set(value) != expected:
        raise ValueError("random draw event_key fields are incomplete or unexpected")
    text_fields: dict[str, str] = {}
    for name in ("experiment_id", "purpose", "schema_version", "seed", "value"):
        item = value[name]
        if type(item) is not str:
            raise ValueError(f"random draw event_key {name} must be exact text")
        text_fields[name] = item
    integer_fields: dict[str, int] = {}
    for name in ("replicate_id", "simulation_step"):
        item = value[name]
        if type(item) is not int:
            raise ValueError(f"random draw event_key {name} must be an exact integer")
        integer_fields[name] = item
    event_key = EventKey(
        experiment_id=text_fields["experiment_id"],
        seed=text_fields["seed"],
        replicate_id=integer_fields["replicate_id"],
        simulation_step=integer_fields["simulation_step"],
        purpose=text_fields["purpose"],
        schema_version=text_fields["schema_version"],
    )
    _validate_receipted_event_key(event_key)
    if text_fields["value"] != event_key.value:
        raise ValueError("random draw event_key value does not match its fields")
    return event_key


def _validate_draw_request(
    *, event_key: EventKey, domain: str, draw_index: int, lower: int, upper: int
) -> None:
    _validate_receipted_event_key(event_key)
    if not isinstance(domain, str) or _DOMAIN_PATTERN.fullmatch(domain) is None:
        raise ValueError("domain must be an explicit bounded ASCII identifier")
    if (
        isinstance(draw_index, bool)
        or not isinstance(draw_index, int)
        or draw_index < 0
        or draw_index > _MAX_DRAW_INDEX
    ):
        raise ValueError("draw_index must be a bounded non-negative integer")
    for name, value in (("lower", lower), ("upper", upper)):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < _MIN_SIGNED_64
            or value > _MAX_SIGNED_64
        ):
            raise ValueError(f"{name} must be a signed 64-bit integer")
    if lower > upper:
        raise ValueError("lower bound must not exceed upper bound")


def _draw_input_mapping(
    *, event_key: EventKey, domain: str, draw_index: int, lower: int, upper: int
) -> dict[str, object]:
    return {
        "distribution": DISCRETE_UNIFORM_DISTRIBUTION,
        "domain": domain,
        "draw_index": draw_index,
        "event_key": _event_key_mapping(event_key),
        "lower": lower,
        "rng_implementation_id": RNG_IMPLEMENTATION_ID,
        "upper": upper,
    }


@dataclass(frozen=True, slots=True, init=False)
class RandomDrawReceipt:
    """Factory-produced evidence for one exact deterministic integer draw.

    Direct construction is intentionally unavailable.  ``from_mapping`` does
    not trust serialized result fields: it invokes the same public draw
    factory and requires exact canonical equality.
    """

    event_key: EventKey
    domain: str
    draw_index: int
    distribution: str
    lower: int
    upper: int
    rng_implementation_id: str
    rng_algorithm_spec_commitment: str
    rng_implementation_commitment: str
    input_commitment: str
    accepted_attempt: int
    accepted_candidate_bits_hex: str
    integer_value: int
    output_bits_hex: str
    schema_version: str

    def __new__(cls) -> Self:
        raise TypeError(
            "RandomDrawReceipt is factory-only; use deterministic_int_with_receipt"
        )

    @classmethod
    def _create(
        cls,
        *,
        event_key: EventKey,
        domain: str,
        draw_index: int,
        lower: int,
        upper: int,
        input_commitment: str,
        accepted_attempt: int,
        accepted_candidate_bits_hex: str,
        integer_value: int,
    ) -> Self:
        assert_live_rng_implementation()
        instance = object.__new__(cls)
        values: Mapping[str, object] = {
            "event_key": event_key,
            "domain": domain,
            "draw_index": draw_index,
            "distribution": DISCRETE_UNIFORM_DISTRIBUTION,
            "lower": lower,
            "upper": upper,
            "rng_implementation_id": RNG_IMPLEMENTATION_ID,
            "rng_algorithm_spec_commitment": RNG_ALGORITHM_SPEC_COMMITMENT,
            "rng_implementation_commitment": RNG_IMPLEMENTATION_COMMITMENT,
            "input_commitment": input_commitment,
            "accepted_attempt": accepted_attempt,
            "accepted_candidate_bits_hex": accepted_candidate_bits_hex,
            "integer_value": integer_value,
            "output_bits_hex": f"{integer_value & _UINT_64_MASK:016x}",
            "schema_version": RANDOM_DRAW_RECEIPT_SCHEMA_VERSION,
        }
        for name, value in values.items():
            object.__setattr__(instance, name, value)
        instance._validate()
        return instance

    def _validate(self) -> None:
        assert_live_rng_implementation()
        _validate_draw_request(
            event_key=self.event_key,
            domain=self.domain,
            draw_index=self.draw_index,
            lower=self.lower,
            upper=self.upper,
        )
        if self.distribution != DISCRETE_UNIFORM_DISTRIBUTION:
            raise ValueError("random draw distribution is unsupported")
        if self.rng_implementation_id != RNG_IMPLEMENTATION_ID:
            raise ValueError("random draw RNG implementation ID changed")
        if self.rng_algorithm_spec_commitment != RNG_ALGORITHM_SPEC_COMMITMENT:
            raise ValueError("random draw RNG algorithm specification changed")
        if self.rng_implementation_commitment != RNG_IMPLEMENTATION_COMMITMENT:
            raise ValueError("random draw RNG implementation commitment changed")
        for name in (
            "rng_algorithm_spec_commitment",
            "rng_implementation_commitment",
            "input_commitment",
        ):
            if _SHA256_PATTERN.fullmatch(getattr(self, name)) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if (
            isinstance(self.accepted_attempt, bool)
            or not isinstance(self.accepted_attempt, int)
            or self.accepted_attempt < 0
            or self.accepted_attempt >= _MAX_REJECTION_ATTEMPTS
        ):
            raise ValueError("accepted_attempt exceeds the bounded rejection budget")
        if (
            not isinstance(self.accepted_candidate_bits_hex, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.accepted_candidate_bits_hex) is None
        ):
            raise ValueError("accepted candidate must contain exactly 256 bits")
        if (
            isinstance(self.integer_value, bool)
            or not isinstance(self.integer_value, int)
            or not self.lower <= self.integer_value <= self.upper
        ):
            raise ValueError("integer_value lies outside its committed bounds")
        if self.output_bits_hex != f"{self.integer_value & _UINT_64_MASK:016x}":
            raise ValueError(
                "output_bits_hex is not the exact signed-64 representation"
            )
        if self.schema_version != RANDOM_DRAW_RECEIPT_SCHEMA_VERSION:
            raise ValueError("random draw receipt schema version changed")

    def to_mapping(self) -> dict[str, object]:
        """Return the closed canonical receipt mapping."""
        return {
            "accepted_attempt": self.accepted_attempt,
            "accepted_candidate_bits_hex": self.accepted_candidate_bits_hex,
            "distribution": self.distribution,
            "domain": self.domain,
            "draw_index": self.draw_index,
            "event_key": _event_key_mapping(self.event_key),
            "input_commitment": self.input_commitment,
            "integer_value": self.integer_value,
            "lower": self.lower,
            "output_bits_hex": self.output_bits_hex,
            "rng_algorithm_spec_commitment": self.rng_algorithm_spec_commitment,
            "rng_implementation_commitment": self.rng_implementation_commitment,
            "rng_implementation_id": self.rng_implementation_id,
            "schema_version": self.schema_version,
            "upper": self.upper,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        """Recompute and strictly compare a serialized draw receipt."""
        assert_live_rng_implementation()
        expected_fields = {
            "accepted_attempt",
            "accepted_candidate_bits_hex",
            "distribution",
            "domain",
            "draw_index",
            "event_key",
            "input_commitment",
            "integer_value",
            "lower",
            "output_bits_hex",
            "rng_algorithm_spec_commitment",
            "rng_implementation_commitment",
            "rng_implementation_id",
            "schema_version",
            "upper",
        }
        if set(value) != expected_fields:
            raise ValueError("random draw receipt fields are incomplete or unexpected")
        text_fields = (
            "accepted_candidate_bits_hex",
            "distribution",
            "domain",
            "input_commitment",
            "output_bits_hex",
            "rng_algorithm_spec_commitment",
            "rng_implementation_commitment",
            "rng_implementation_id",
            "schema_version",
        )
        for name in text_fields:
            if type(value[name]) is not str:
                raise ValueError(f"random draw {name} must be exact text")
        integer_fields = (
            "accepted_attempt",
            "draw_index",
            "integer_value",
            "lower",
            "upper",
        )
        for name in integer_fields:
            if type(value[name]) is not int:
                raise ValueError(f"random draw {name} must be an exact integer")
        event_key = _parse_event_key(value["event_key"])
        domain = value["domain"]
        draw_index = value["draw_index"]
        lower = value["lower"]
        upper = value["upper"]
        assert isinstance(domain, str)
        assert isinstance(draw_index, int)
        assert isinstance(lower, int)
        assert isinstance(upper, int)
        _, recomputed = deterministic_int_with_receipt(
            event_key,
            domain=domain,
            draw_index=draw_index,
            lower=lower,
            upper=upper,
        )
        if _canonical_bytes(value) != _canonical_bytes(recomputed.to_mapping()):
            raise ValueError(
                "random draw receipt differs from independent recomputation"
            )
        return cast(Self, recomputed)

    @property
    def sha256(self) -> str:
        """Return the canonical receipt commitment."""
        return hashlib.sha256(_canonical_bytes(self.to_mapping())).hexdigest()


def deterministic_int_with_receipt(
    event_key: EventKey,
    *,
    domain: str,
    draw_index: int,
    lower: int,
    upper: int,
) -> tuple[int, RandomDrawReceipt]:
    """Return an exact domain-separated integer draw and its receipt.

    ``domain`` and ``draw_index`` are keyword-only and have no defaults.  The
    bounded request schema prevents unbounded serialized integers and the loop
    has a fixed fail-closed rejection budget.
    """
    assert_live_rng_implementation()
    _validate_draw_request(
        event_key=event_key,
        domain=domain,
        draw_index=draw_index,
        lower=lower,
        upper=upper,
    )
    input_mapping = _draw_input_mapping(
        event_key=event_key,
        domain=domain,
        draw_index=draw_index,
        lower=lower,
        upper=upper,
    )
    input_bytes = _canonical_bytes(input_mapping)
    input_commitment = hashlib.sha256(input_bytes).hexdigest()
    span = upper - lower + 1
    hash_domain = 1 << 256
    acceptance_limit = hash_domain - (hash_domain % span)
    for attempt in range(_MAX_REJECTION_ATTEMPTS):
        material = (
            b"policytwin-rng-v1\x00"
            + input_bytes
            + b"\x00"
            + attempt.to_bytes(8, "big")
        )
        digest = hashlib.sha256(material).digest()
        candidate = int.from_bytes(digest, "big")
        if candidate < acceptance_limit:
            integer_value = lower + candidate % span
            receipt = RandomDrawReceipt._create(
                event_key=event_key,
                domain=domain,
                draw_index=draw_index,
                lower=lower,
                upper=upper,
                input_commitment=input_commitment,
                accepted_attempt=attempt,
                accepted_candidate_bits_hex=digest.hex(),
                integer_value=integer_value,
            )
            return integer_value, receipt
    raise RuntimeError("bounded deterministic rejection sampler exhausted")


def deterministic_int(event_key: EventKey, lower: int, upper: int) -> int:
    """Draw an integer from a semantic key without mutable RNG state.

    Rejection sampling avoids modulo bias. The result depends only on the event
    key and bounds, never on call order, process identity, or a world label.
    """
    if lower > upper:
        msg = "lower bound must not exceed upper bound"
        raise ValueError(msg)

    span = upper - lower + 1
    domain = 1 << 256
    acceptance_limit = domain - (domain % span)
    attempt = 0
    while True:
        material = f"{event_key.value}|{lower}|{upper}|{attempt}".encode()
        candidate = int.from_bytes(hashlib.sha256(material).digest(), "big")
        if candidate < acceptance_limit:
            return lower + candidate % span
        attempt += 1


def _live_source(value: Any, name: str) -> str:
    """Return bounded source for one allowlisted live RNG dependency."""
    try:
        source = inspect.getsource(value)
    except (OSError, TypeError) as error:
        raise RuntimeError(f"cannot inspect live RNG dependency: {name}") from error
    if not source or len(source.encode("utf-8")) > 256 * 1024:
        raise RuntimeError(f"live RNG dependency source is invalid: {name}")
    return source


def _dependency_cache_identity(name: str, value: object) -> tuple[str, int, int]:
    """Return a process-local cache identity; it is never artifact evidence."""
    target = getattr(value, "__func__", value)
    return name, id(target), id(getattr(target, "__code__", None))


def _rng_runtime_constants() -> tuple[tuple[str, object], ...]:
    return (
        ("action_choice_domain", AGENT_ACTION_CHOICE_DOMAIN),
        ("algorithm_spec_commitment", RNG_ALGORITHM_SPEC_COMMITMENT),
        ("distribution", DISCRETE_UNIFORM_DISTRIBUTION),
        ("domain_pattern", _DOMAIN_PATTERN.pattern),
        ("implementation_id", RNG_IMPLEMENTATION_ID),
        ("input_prefix_hex", b"policytwin-rng-v1\x00".hex()),
        ("max_draw_index", _MAX_DRAW_INDEX),
        ("max_rejection_attempts", _MAX_REJECTION_ATTEMPTS),
        ("max_signed_64", _MAX_SIGNED_64),
        ("min_signed_64", _MIN_SIGNED_64),
        ("observation_noise_domain", INTEGRATED_OBSERVATION_NOISE_DOMAIN),
        ("receipt_schema_version", RANDOM_DRAW_RECEIPT_SCHEMA_VERSION),
    )


def _rng_live_dependencies() -> tuple[tuple[str, object], ...]:
    return (
        ("EventKey.canonical_text", EventKey.canonical_text),
        ("EventKey.value", cast(property, EventKey.value).fget),
        ("RandomDrawReceipt.__new__", RandomDrawReceipt.__new__),
        ("RandomDrawReceipt._create", RandomDrawReceipt._create),
        ("RandomDrawReceipt._validate", RandomDrawReceipt._validate),
        ("RandomDrawReceipt.from_mapping", RandomDrawReceipt.from_mapping),
        ("RandomDrawReceipt.to_mapping", RandomDrawReceipt.to_mapping),
        ("_canonical_bytes", _canonical_bytes),
        ("_dependency_cache_identity", _dependency_cache_identity),
        ("_draw_input_mapping", _draw_input_mapping),
        ("_event_key_mapping", _event_key_mapping),
        ("_live_source", _live_source),
        ("_parse_event_key", _parse_event_key),
        ("_rng_live_dependencies", _rng_live_dependencies),
        ("_rng_runtime_constants", _rng_runtime_constants),
        ("_validate_draw_request", _validate_draw_request),
        ("_validate_receipted_event_key", _validate_receipted_event_key),
        ("assert_live_rng_implementation", assert_live_rng_implementation),
        ("deterministic_int_with_receipt", deterministic_int_with_receipt),
        ("rng_implementation_commitment", rng_implementation_commitment),
    )


# These are mutable, process-local caches rather than implementation constants.
# Keep their names lowercase so the generic implementation fingerprinter does
# not mistake address-bearing cache identities for portable static constants.
_rng_commitment_cache_key: tuple[object, ...] | None = None
_rng_commitment_cache_value: str | None = None


def rng_implementation_commitment() -> str:
    """Hash the allowlisted live code that defines receipted integer draws.

    This is an in-process implementation commitment, not an external provenance
    anchor.  Run metadata and the Git/environment commitments remain necessary
    to bind the wider source tree and execution environment.
    """
    global _rng_commitment_cache_key, _rng_commitment_cache_value
    dependencies = _rng_live_dependencies()
    runtime_constants = _rng_runtime_constants()
    cache_key: tuple[object, ...] = (
        tuple(_dependency_cache_identity(name, value) for name, value in dependencies),
        runtime_constants,
    )
    if (
        cache_key == _rng_commitment_cache_key
        and _rng_commitment_cache_value is not None
    ):
        return _rng_commitment_cache_value
    mapping: dict[str, object] = {
        "algorithm_spec_commitment": RNG_ALGORITHM_SPEC_COMMITMENT,
        "dependencies": {
            name: _live_source(value, name) for name, value in dependencies
        },
        "rng_implementation_id": RNG_IMPLEMENTATION_ID,
        "runtime_constants": dict(runtime_constants),
    }
    current_dependencies = _rng_live_dependencies()
    current_constants = _rng_runtime_constants()
    current_key: tuple[object, ...] = (
        tuple(
            _dependency_cache_identity(name, value)
            for name, value in current_dependencies
        ),
        current_constants,
    )
    if current_key != cache_key:
        raise RuntimeError("receipted RNG implementation changed during inspection")
    commitment = hashlib.sha256(_canonical_bytes(mapping)).hexdigest()
    _rng_commitment_cache_key = cache_key
    _rng_commitment_cache_value = commitment
    return commitment


def assert_live_rng_implementation() -> str:
    """Fail closed if allowlisted live RNG code differs from module startup."""
    current = rng_implementation_commitment()
    try:
        expected = RNG_IMPLEMENTATION_COMMITMENT
    except NameError as error:
        raise RuntimeError("receipted RNG startup commitment is unavailable") from error
    if current != expected:
        raise RuntimeError("receipted RNG implementation drift detected")
    return current


RNG_IMPLEMENTATION_COMMITMENT = rng_implementation_commitment()
