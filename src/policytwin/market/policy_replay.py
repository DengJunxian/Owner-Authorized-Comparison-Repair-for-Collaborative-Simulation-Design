"""Strict, allowlisted persistence for synthetic policy-required CLOB replay.

The loader in this module is intentionally narrower than a general serializer.
It accepts one closed synthetic schema, reconstructs only project-owned SEC
fixture objects, and never imports a class named by artifact data.  Real SEC
policy evidence must continue through the primary-source loaders.
"""

from __future__ import annotations

import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn, Self, cast

from policytwin.counterfactual import EVENT_KEY_SCHEMA_VERSION, EventKey
from policytwin.market.models import (
    MARKET_SCHEMA_VERSION,
    SCHEDULER_SCHEMA_VERSION,
    AccountBalance,
    CancelOrder,
    LimitOrder,
    MarketCommand,
    MarketConfig,
    MarketOrder,
    MarketPolicyMode,
    MarketValidationError,
    OpeningOrder,
    OpeningState,
    ScheduledCommand,
    Side,
)
from policytwin.market.policy_gateway import (
    PolicyGatewayConfig,
    SecTickSizePolicyGateway,
    SyntheticEvaluatorDescriptor,
    SyntheticEvaluatorRole,
)
from policytwin.policies.sec_tick_size_pilot import (
    AssignmentIntegrityError,
    PolicyEvidenceError,
    PolicySourceRegistry,
    SecTickSizePilotConfig,
    SecTickSizePilotPolicy,
    SecurityAssignmentManifest,
    load_security_assignment_manifest,
)
from policytwin.provenance import (
    canonical_json_bytes,
    read_json_bytes_object,
    sha256_bytes,
    sha256_file,
)

if TYPE_CHECKING:
    from policytwin.market.models import ReplayResult

POLICY_REPLAY_DESCRIPTOR_SCHEMA_VERSION = "policytwin.policy-replay-descriptor.v1"
POLICY_REPLAY_BUNDLE_SCHEMA_VERSION = "policytwin.policy-replay-bundle.v1"
POLICY_REPLAY_MANIFEST_SCHEMA_VERSION = "policytwin.policy-replay-manifest.v1"
LEDGER_COMMITMENT_SCHEMA_VERSION = "policytwin.replay-ledger-commitment.v1"

_ARTIFACT_FILENAME = "artifact.json"
_ASSIGNMENT_FILENAME = "assignments.json"
_MANIFEST_FILENAME = "manifest.json"
_EXPECTED_FILES = frozenset(
    {_ARTIFACT_FILENAME, _ASSIGNMENT_FILENAME, _MANIFEST_FILENAME}
)
_SHA256_ZERO = "0" * 64
_MAX_ARTIFACT_BYTES = 16 * 1024 * 1024
_MAX_ASSIGNMENT_BYTES = 8 * 1024 * 1024


class PolicyReplayError(MarketValidationError):
    """Raised when persisted replay evidence cannot be trusted."""


def _fail(message: str) -> NoReturn:
    raise PolicyReplayError(message)


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


def _text(value: object, field: str, *, maximum: int = 512) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > maximum
        or "\x00" in value
    ):
        _fail(f"{field} must be bounded non-empty text")
    return value


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{field} must be an integer")
    return value


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        _fail(f"{field} must be a boolean")
    return value


def _sha256(value: object, field: str) -> str:
    text = _text(value, field, maximum=64)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        _fail(f"{field} must be a lowercase SHA-256 digest")
    return text


def _canonical_mapping_text(value: Mapping[str, object]) -> str:
    return canonical_json_bytes(value).decode("utf-8")


def _mapping_from_canonical_text(value: str, field: str) -> Mapping[str, object]:
    content = value.encode("utf-8")
    parsed = read_json_bytes_object(content, source=field)
    if canonical_json_bytes(parsed) != content:
        _fail(f"{field} must use canonical JSON encoding")
    return parsed


@dataclass(frozen=True, slots=True)
class ImplementationCommitment:
    """Hash of one fixed project implementation component."""

    component: str
    sha256: str

    def __post_init__(self) -> None:
        _text(self.component, "implementation component", maximum=128)
        _sha256(self.sha256, "implementation sha256")

    def to_mapping(self) -> dict[str, str]:
        return {"component": self.component, "sha256": self.sha256}

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(
            value,
            frozenset({"component", "sha256"}),
            "implementation commitment",
        )
        return cls(
            component=_text(value["component"], "component", maximum=128),
            sha256=_sha256(value["sha256"], "sha256"),
        )


def _implementation_paths() -> dict[str, Path]:
    package_root = Path(__file__).parents[1]
    return {
        "counterfactual.randomness": package_root / "counterfactual" / "randomness.py",
        "market.models": package_root / "market" / "models.py",
        "market.order_book": package_root / "market" / "order_book.py",
        "market.policy_gateway": package_root / "market" / "policy_gateway.py",
        "market.policy_replay": package_root / "market" / "policy_replay.py",
        "policies.sec_tick_size_pilot": (
            package_root / "policies" / "sec_tick_size_pilot.py"
        ),
        "provenance": package_root / "provenance.py",
    }


def current_implementation_commitments() -> tuple[ImplementationCommitment, ...]:
    """Commit the closed implementation allowlist used during reconstruction."""
    return tuple(
        ImplementationCommitment(component, sha256_file(path))
        for component, path in sorted(_implementation_paths().items())
    )


@dataclass(frozen=True, slots=True)
class SyntheticPolicyReplayDescriptor:
    """Complete immutable reconstruction contract for one synthetic gateway."""

    gateway_config: PolicyGatewayConfig
    policy_config_json: str
    source_registry_json: str
    assignment_manifest_json: str
    assignment_source_sha256: str
    assignment_source_bytes: bytes
    exception_evaluator: SyntheticEvaluatorDescriptor
    trade_at_evaluator: SyntheticEvaluatorDescriptor
    implementation_commitments: tuple[ImplementationCommitment, ...]
    synthetic_only: bool = True
    schema_version: str = POLICY_REPLAY_DESCRIPTOR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.gateway_config) is not PolicyGatewayConfig:
            _fail("replay descriptor requires an exact PolicyGatewayConfig")
        if self.schema_version != POLICY_REPLAY_DESCRIPTOR_SCHEMA_VERSION:
            _fail("unsupported policy replay descriptor schema")
        if self.synthetic_only is not True or not self.gateway_config.synthetic_only:
            raise PolicyEvidenceError(
                "real SEC policy replay requires verified primary-source artifacts"
            )
        if not isinstance(self.assignment_source_bytes, bytes):
            _fail("assignment source must be immutable bytes")
        if len(self.assignment_source_bytes) > _MAX_ASSIGNMENT_BYTES:
            _fail("assignment source exceeds the replay size limit")
        source_sha256 = sha256_bytes(self.assignment_source_bytes)
        if (
            _sha256(self.assignment_source_sha256, "assignment_source_sha256")
            != source_sha256
            or source_sha256 != self.gateway_config.assignment_source_sha256
        ):
            raise AssignmentIntegrityError(
                "assignment bytes differ from the gateway commitment"
            )
        if (
            type(self.exception_evaluator) is not SyntheticEvaluatorDescriptor
            or self.exception_evaluator.role
            is not SyntheticEvaluatorRole.DOCUMENTED_EXCEPTION
            or type(self.trade_at_evaluator) is not SyntheticEvaluatorDescriptor
            or self.trade_at_evaluator.role is not SyntheticEvaluatorRole.TRADE_AT
        ):
            _fail("replay evaluator descriptors have invalid roles")
        policy_mapping = _mapping_from_canonical_text(
            self.policy_config_json, "policy_config_json"
        )
        source_mapping = _mapping_from_canonical_text(
            self.source_registry_json, "source_registry_json"
        )
        assignment_mapping = _mapping_from_canonical_text(
            self.assignment_manifest_json, "assignment_manifest_json"
        )
        registry = PolicySourceRegistry.from_synthetic_replay_mapping(source_mapping)
        policy = SecTickSizePilotConfig.from_replay_mapping(
            policy_mapping, source_registry=registry
        )
        assignment = SecurityAssignmentManifest.from_mapping(assignment_mapping)
        if (
            policy.sha256 != self.gateway_config.policy_sha256
            or registry.sha256 != self.gateway_config.source_registry_sha256
            or assignment.sha256 != self.gateway_config.assignment_sha256
            or assignment.policy_id != policy.policy_id
            or assignment.synthetic_only is not True
        ):
            _fail("policy, source, assignment, and gateway commitments disagree")
        if not isinstance(self.implementation_commitments, tuple) or not all(
            type(item) is ImplementationCommitment
            for item in self.implementation_commitments
        ):
            _fail("implementation commitments must be an immutable typed tuple")
        expected = current_implementation_commitments()
        if self.implementation_commitments != expected:
            _fail("runtime implementation differs from the replay commitment")

    @classmethod
    def capture(cls, gateway: SecTickSizePolicyGateway) -> Self:
        """Capture only project-owned, explicitly synthetic gateway material."""
        if type(gateway) is not SecTickSizePolicyGateway:
            _fail("replay capture requires an exact SecTickSizePolicyGateway")
        material = gateway.replay_material
        return cls(
            gateway_config=material.gateway_config,
            policy_config_json=_canonical_mapping_text(
                material.policy_config.to_mapping()
            ),
            source_registry_json=_canonical_mapping_text(
                material.source_registry.to_mapping()
            ),
            assignment_manifest_json=_canonical_mapping_text(
                material.assignment_manifest.to_mapping()
            ),
            assignment_source_sha256=sha256_bytes(material.assignment_source_bytes),
            assignment_source_bytes=material.assignment_source_bytes,
            exception_evaluator=material.exception_evaluator,
            trade_at_evaluator=material.trade_at_evaluator,
            implementation_commitments=current_implementation_commitments(),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "assignment": {
                "filename": _ASSIGNMENT_FILENAME,
                "manifest": dict(
                    _mapping_from_canonical_text(
                        self.assignment_manifest_json, "assignment_manifest_json"
                    )
                ),
                "manifest_sha256": self.gateway_config.assignment_sha256,
                "source_sha256": self.assignment_source_sha256,
            },
            "exception_evaluator": self.exception_evaluator.to_mapping(),
            "gateway_config": self.gateway_config.to_mapping(),
            "gateway_config_sha256": self.gateway_config.sha256,
            "implementation_commitments": [
                item.to_mapping() for item in self.implementation_commitments
            ],
            "policy_config": dict(
                _mapping_from_canonical_text(
                    self.policy_config_json, "policy_config_json"
                )
            ),
            "policy_config_sha256": self.gateway_config.policy_sha256,
            "schema_version": self.schema_version,
            "source_registry": dict(
                _mapping_from_canonical_text(
                    self.source_registry_json, "source_registry_json"
                )
            ),
            "source_registry_sha256": self.gateway_config.source_registry_sha256,
            "synthetic_only": self.synthetic_only,
            "trade_at_evaluator": self.trade_at_evaluator.to_mapping(),
        }

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object], *, assignment_source_bytes: bytes
    ) -> Self:
        _strict_fields(
            value,
            frozenset(
                {
                    "assignment",
                    "exception_evaluator",
                    "gateway_config",
                    "gateway_config_sha256",
                    "implementation_commitments",
                    "policy_config",
                    "policy_config_sha256",
                    "schema_version",
                    "source_registry",
                    "source_registry_sha256",
                    "synthetic_only",
                    "trade_at_evaluator",
                }
            ),
            "policy replay descriptor",
        )
        if value["synthetic_only"] is not True:
            raise PolicyEvidenceError(
                "replay loader accepts only explicit synthetic fixtures"
            )
        assignment = _mapping(value["assignment"], "assignment")
        _strict_fields(
            assignment,
            frozenset({"filename", "manifest", "manifest_sha256", "source_sha256"}),
            "assignment descriptor",
        )
        if assignment["filename"] != _ASSIGNMENT_FILENAME:
            _fail("assignment descriptor filename is not allowlisted")
        gateway = PolicyGatewayConfig.from_mapping(
            _mapping(value["gateway_config"], "gateway_config")
        )
        if gateway.sha256 != _sha256(
            value["gateway_config_sha256"], "gateway_config_sha256"
        ):
            _fail("gateway config hash differs from its descriptor")
        if gateway.policy_sha256 != _sha256(
            value["policy_config_sha256"], "policy_config_sha256"
        ):
            _fail("policy config hash differs from the gateway")
        if gateway.source_registry_sha256 != _sha256(
            value["source_registry_sha256"], "source_registry_sha256"
        ):
            _fail("source registry hash differs from the gateway")
        if gateway.assignment_sha256 != _sha256(
            assignment["manifest_sha256"], "assignment manifest_sha256"
        ):
            _fail("assignment manifest hash differs from the gateway")
        return cls(
            gateway_config=gateway,
            policy_config_json=_canonical_mapping_text(
                _mapping(value["policy_config"], "policy_config")
            ),
            source_registry_json=_canonical_mapping_text(
                _mapping(value["source_registry"], "source_registry")
            ),
            assignment_manifest_json=_canonical_mapping_text(
                _mapping(assignment["manifest"], "assignment manifest")
            ),
            assignment_source_sha256=_sha256(
                assignment["source_sha256"], "assignment source_sha256"
            ),
            assignment_source_bytes=assignment_source_bytes,
            exception_evaluator=SyntheticEvaluatorDescriptor.from_mapping(
                _mapping(value["exception_evaluator"], "exception_evaluator")
            ),
            trade_at_evaluator=SyntheticEvaluatorDescriptor.from_mapping(
                _mapping(value["trade_at_evaluator"], "trade_at_evaluator")
            ),
            implementation_commitments=tuple(
                ImplementationCommitment.from_mapping(
                    _mapping(item, "implementation commitment")
                )
                for item in _sequence(
                    value["implementation_commitments"],
                    "implementation_commitments",
                )
            ),
            synthetic_only=True,
            schema_version=_text(value["schema_version"], "schema_version"),
        )

    def reconstruct(self, assignment_path: Path) -> SecTickSizePolicyGateway:
        """Rebuild one fresh gateway without accepting caller-provided objects."""
        if sha256_file(assignment_path) != self.assignment_source_sha256:
            raise AssignmentIntegrityError(
                "persisted assignment bytes differ from the replay descriptor"
            )
        registry = PolicySourceRegistry.from_synthetic_replay_mapping(
            _mapping_from_canonical_text(
                self.source_registry_json, "source_registry_json"
            )
        )
        policy_config = SecTickSizePilotConfig.from_replay_mapping(
            _mapping_from_canonical_text(self.policy_config_json, "policy_config_json"),
            source_registry=registry,
        )
        assignments = load_security_assignment_manifest(
            assignment_path,
            expected_sha256=self.assignment_source_sha256,
            registry=registry,
            trusted_cutoff=policy_config.trusted_cutoff,
        )
        if assignments.manifest.to_mapping() != dict(
            _mapping_from_canonical_text(
                self.assignment_manifest_json, "assignment_manifest_json"
            )
        ):
            raise AssignmentIntegrityError(
                "persisted assignment semantics differ from the descriptor"
            )
        policy = SecTickSizePilotPolicy(policy_config, assignments)
        return SecTickSizePolicyGateway(
            self.gateway_config,
            policy,
            exception_evaluator_descriptor=self.exception_evaluator,
            trade_at_evaluator_descriptor=self.trade_at_evaluator,
        )


@dataclass(frozen=True, slots=True)
class LedgerCommitment:
    """Independent commitment to one deterministic hash-chained ledger."""

    ledger: str
    record_count: int
    records_sha256: str
    chain_tail_sha256: str
    schema_version: str = LEDGER_COMMITMENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.ledger not in {"market", "policy"}:
            _fail("ledger commitment kind is invalid")
        if self.record_count < 0:
            _fail("ledger record_count must be non-negative")
        _sha256(self.records_sha256, "ledger records_sha256")
        _sha256(self.chain_tail_sha256, "ledger chain_tail_sha256")
        if self.schema_version != LEDGER_COMMITMENT_SCHEMA_VERSION:
            _fail("unsupported ledger commitment schema")

    def to_mapping(self) -> dict[str, object]:
        return {
            "chain_tail_sha256": self.chain_tail_sha256,
            "ledger": self.ledger,
            "record_count": self.record_count,
            "records_sha256": self.records_sha256,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(
            value,
            frozenset(
                {
                    "chain_tail_sha256",
                    "ledger",
                    "record_count",
                    "records_sha256",
                    "schema_version",
                }
            ),
            "ledger commitment",
        )
        return cls(
            ledger=_text(value["ledger"], "ledger", maximum=16),
            record_count=_integer(value["record_count"], "record_count"),
            records_sha256=_sha256(value["records_sha256"], "records_sha256"),
            chain_tail_sha256=_sha256(value["chain_tail_sha256"], "chain_tail_sha256"),
            schema_version=_text(value["schema_version"], "schema_version"),
        )


def ledger_commitment(
    ledger: str, records: Sequence[Mapping[str, object]]
) -> LedgerCommitment:
    """Build a count, canonical hash, and hash-chain tail commitment."""
    mappings = tuple(dict(record) for record in records)
    tail = _SHA256_ZERO
    if mappings:
        raw_tail = mappings[-1].get("event_sha256")
        tail = _sha256(raw_tail, "ledger event_sha256")
    return LedgerCommitment(
        ledger=ledger,
        record_count=len(mappings),
        records_sha256=sha256_bytes(canonical_json_bytes(mappings)),
        chain_tail_sha256=tail,
    )


@dataclass(frozen=True, slots=True)
class LoadedSyntheticPolicyReplay:
    """Strictly parsed replay inputs and independent output commitments."""

    config: MarketConfig
    opening_state: OpeningState
    scheduled_commands: tuple[ScheduledCommand, ...]
    descriptor: SyntheticPolicyReplayDescriptor
    assignment_path: Path
    expected_result_sha256: str
    expected_state_sha256: str
    expected_trades_sha256: str
    expected_quotes_sha256: str
    market_ledger: LedgerCommitment
    policy_ledger: LedgerCommitment

    def reconstruct_gateway(self) -> SecTickSizePolicyGateway:
        return self.descriptor.reconstruct(self.assignment_path)


def _parse_event_key(value: Mapping[str, object]) -> EventKey:
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
        "event key",
    )
    if value["schema_version"] != EVENT_KEY_SCHEMA_VERSION:
        _fail("unsupported event key schema")
    key = EventKey(
        experiment_id=_text(value["experiment_id"], "experiment_id", maximum=128),
        seed=_text(value["seed"], "seed"),
        replicate_id=_integer(value["replicate_id"], "replicate_id"),
        simulation_step=_integer(value["simulation_step"], "simulation_step"),
        purpose=_text(value["purpose"], "purpose"),
        schema_version=EVENT_KEY_SCHEMA_VERSION,
    )
    if key.replicate_id < 0 or key.simulation_step < 0 or value["value"] != key.value:
        _fail("event key fields differ from the derived commitment")
    return key


def _parse_side(value: object) -> Side:
    if not isinstance(value, str):
        _fail("market side must be text")
    try:
        return Side(value)
    except (TypeError, ValueError) as error:
        raise PolicyReplayError("unsupported market side") from error


def _parse_command(value: Mapping[str, object]) -> MarketCommand:
    command_type = value.get("command_type")
    common = {
        "command_type",
        "event_key",
        "submitted_at",
        "trader_id",
    }
    if command_type == "limit_order":
        _strict_fields(
            value,
            frozenset(common | {"order_id", "side", "quantity", "price"}),
            "limit order",
        )
        return LimitOrder(
            event_key=_parse_event_key(_mapping(value["event_key"], "event_key")),
            submitted_at=_integer(value["submitted_at"], "submitted_at"),
            order_id=_text(value["order_id"], "order_id", maximum=128),
            trader_id=_text(value["trader_id"], "trader_id", maximum=128),
            side=_parse_side(value["side"]),
            quantity=_integer(value["quantity"], "quantity"),
            price=_integer(value["price"], "price"),
        )
    if command_type == "market_order":
        _strict_fields(
            value,
            frozenset(common | {"order_id", "side", "quantity"}),
            "market order",
        )
        return MarketOrder(
            event_key=_parse_event_key(_mapping(value["event_key"], "event_key")),
            submitted_at=_integer(value["submitted_at"], "submitted_at"),
            order_id=_text(value["order_id"], "order_id", maximum=128),
            trader_id=_text(value["trader_id"], "trader_id", maximum=128),
            side=_parse_side(value["side"]),
            quantity=_integer(value["quantity"], "quantity"),
        )
    if command_type == "cancel_order":
        _strict_fields(
            value,
            frozenset(common | {"target_order_id"}),
            "cancel order",
        )
        return CancelOrder(
            event_key=_parse_event_key(_mapping(value["event_key"], "event_key")),
            submitted_at=_integer(value["submitted_at"], "submitted_at"),
            trader_id=_text(value["trader_id"], "trader_id", maximum=128),
            target_order_id=_text(
                value["target_order_id"], "target_order_id", maximum=128
            ),
        )
    _fail("artifact command_type is outside the market allowlist")


def _parse_market_config(value: Mapping[str, object]) -> MarketConfig:
    _strict_fields(
        value,
        frozenset(
            {
                "tick_size",
                "latency_steps",
                "maker_fee_bps",
                "taker_fee_bps",
                "policy_mode",
                "schema_version",
                "scheduler_schema_version",
            }
        ),
        "market config",
    )
    if (
        value["schema_version"] != MARKET_SCHEMA_VERSION
        or value["scheduler_schema_version"] != SCHEDULER_SCHEMA_VERSION
        or value["policy_mode"] != MarketPolicyMode.POLICY_REQUIRED.value
    ):
        _fail("replay bundle requires the supported policy-required market schema")
    return MarketConfig(
        tick_size=_integer(value["tick_size"], "tick_size"),
        latency_steps=_integer(value["latency_steps"], "latency_steps"),
        maker_fee_bps=_integer(value["maker_fee_bps"], "maker_fee_bps"),
        taker_fee_bps=_integer(value["taker_fee_bps"], "taker_fee_bps"),
        policy_mode=MarketPolicyMode.POLICY_REQUIRED,
    )


def _parse_opening_state(value: Mapping[str, object]) -> OpeningState:
    _strict_fields(
        value, frozenset({"accounts", "orders", "venue_cash"}), "opening state"
    )
    accounts = tuple(
        _parse_account(_mapping(item, "opening account"))
        for item in _sequence(value["accounts"], "accounts")
    )
    orders = tuple(
        _parse_opening_order(_mapping(item, "opening order"))
        for item in _sequence(value["orders"], "orders")
    )
    return OpeningState(
        accounts=accounts,
        venue_cash=_integer(value["venue_cash"], "venue_cash"),
        orders=orders,
    )


def _parse_account(value: Mapping[str, object]) -> AccountBalance:
    _strict_fields(value, frozenset({"asset", "cash", "trader_id"}), "opening account")
    return AccountBalance(
        trader_id=_text(value["trader_id"], "trader_id", maximum=128),
        cash=_integer(value["cash"], "cash"),
        asset=_integer(value["asset"], "asset"),
    )


def _parse_opening_order(value: Mapping[str, object]) -> OpeningOrder:
    _strict_fields(
        value,
        frozenset(
            {
                "event_key",
                "order_id",
                "trader_id",
                "side",
                "price",
                "quantity",
                "priority_time",
            }
        ),
        "opening order",
    )
    return OpeningOrder(
        event_key=_parse_event_key(_mapping(value["event_key"], "event_key")),
        order_id=_text(value["order_id"], "order_id", maximum=128),
        trader_id=_text(value["trader_id"], "trader_id", maximum=128),
        side=_parse_side(value["side"]),
        price=_integer(value["price"], "price"),
        quantity=_integer(value["quantity"], "quantity"),
        priority_time=_integer(value["priority_time"], "priority_time"),
    )


def _parse_scheduled_commands(
    value: object, *, config: MarketConfig
) -> tuple[ScheduledCommand, ...]:
    parsed: list[ScheduledCommand] = []
    for item in _sequence(value, "scheduled_commands"):
        mapping = _mapping(item, "scheduled command")
        _strict_fields(
            mapping,
            frozenset({"arrival_time", "command", "event_key"}),
            "scheduled command",
        )
        command = _parse_command(_mapping(mapping["command"], "command"))
        arrival_time = _integer(mapping["arrival_time"], "arrival_time")
        if (
            arrival_time != command.submitted_at + config.latency_steps
            or mapping["event_key"] != command.event_key.value
        ):
            _fail("scheduled command differs from its derived ordering fields")
        parsed.append(
            ScheduledCommand(
                arrival_time=arrival_time,
                event_key=command.event_key.value,
                command=command,
            )
        )
    ordered = tuple(
        sorted(parsed, key=lambda item: (item.arrival_time, item.event_key))
    )
    if tuple(parsed) != ordered:
        _fail("scheduled commands are not in deterministic execution order")
    return ordered


def _write_new(path: Path, content: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        written = 0
        while written < len(content):
            written += os.write(descriptor, content[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_synthetic_policy_replay_bundle(
    result: ReplayResult, destination: Path
) -> Path:
    """Persist one complete policy replay bundle with exact file commitments."""
    descriptor = result.policy_replay_descriptor
    if (
        result.config.policy_mode is not MarketPolicyMode.POLICY_REQUIRED
        or descriptor is None
        or type(descriptor) is not SyntheticPolicyReplayDescriptor
    ):
        _fail("only a synthetic policy-required result can write this bundle")
    if result.policy_gateway_sha256 != descriptor.gateway_config.sha256:
        _fail("result gateway commitment differs from its replay descriptor")
    market_records = tuple(event.to_mapping() for event in result.audit_events)
    policy_records = tuple(event.to_mapping() for event in result.policy_audit_events)
    market_ledger = ledger_commitment("market", market_records)
    policy_ledger = ledger_commitment("policy", policy_records)
    artifact = {
        "descriptor": descriptor.to_mapping(),
        "descriptor_sha256": descriptor.sha256,
        "expected": {
            "market_ledger": market_ledger.to_mapping(),
            "policy_ledger": policy_ledger.to_mapping(),
            "quotes_sha256": sha256_bytes(
                canonical_json_bytes(tuple(item.to_mapping() for item in result.quotes))
            ),
            "result_sha256": result.sha256,
            "state_sha256": result.state.sha256,
            "trades_sha256": sha256_bytes(
                canonical_json_bytes(tuple(item.to_mapping() for item in result.trades))
            ),
        },
        "inputs": {
            "command_stream_sha256": result.command_stream_sha256,
            "config": result.config.to_mapping(),
            "config_sha256": result.config_sha256,
            "opening_state": result.opening_state.to_mapping(),
            "opening_state_sha256": result.opening_state_sha256,
            "scheduled_commands": [
                item.to_mapping() for item in result.scheduled_commands
            ],
        },
        "schema_version": POLICY_REPLAY_BUNDLE_SCHEMA_VERSION,
        "synthetic_only": True,
    }
    artifact_bytes = canonical_json_bytes(artifact)
    assignment_bytes = descriptor.assignment_source_bytes
    directory = Path(os.path.abspath(destination.expanduser()))
    directory.mkdir(parents=True, exist_ok=False)
    _write_new(directory / _ARTIFACT_FILENAME, artifact_bytes)
    _write_new(directory / _ASSIGNMENT_FILENAME, assignment_bytes)
    manifest = {
        "files": [
            {
                "name": _ARTIFACT_FILENAME,
                "sha256": sha256_bytes(artifact_bytes),
                "size_bytes": len(artifact_bytes),
            },
            {
                "name": _ASSIGNMENT_FILENAME,
                "sha256": sha256_bytes(assignment_bytes),
                "size_bytes": len(assignment_bytes),
            },
        ],
        "schema_version": POLICY_REPLAY_MANIFEST_SCHEMA_VERSION,
    }
    _write_new(directory / _MANIFEST_FILENAME, canonical_json_bytes(manifest))
    return directory


def _absolute_local(path: Path) -> Path:
    return Path(os.path.abspath(path.expanduser()))


def _reject_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            _fail(f"replay path contains a symbolic link: {current}")


def _read_regular(path: Path, *, maximum: int) -> bytes:
    _reject_symlink_components(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise PolicyReplayError(f"cannot open replay file: {path}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
            _fail("replay path is not a bounded regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum + 1))
            if not chunk:
                break
            chunks.append(chunk)
            if sum(len(item) for item in chunks) > maximum:
                _fail("replay file exceeds its size limit")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        _fail("replay file changed while it was read")
    return b"".join(chunks)


def _verify_bundle_files(directory: Path) -> tuple[bytes, bytes]:
    root = _absolute_local(directory)
    _reject_symlink_components(root)
    try:
        root_stat = os.lstat(root)
    except OSError as error:
        raise PolicyReplayError("replay bundle directory does not exist") from error
    if not stat.S_ISDIR(root_stat.st_mode):
        _fail("replay bundle path must be a real directory")
    names = frozenset(item.name for item in root.iterdir())
    if names != _EXPECTED_FILES:
        _fail("replay bundle file set differs from the closed schema")
    manifest_bytes = _read_regular(root / _MANIFEST_FILENAME, maximum=64 * 1024)
    manifest = read_json_bytes_object(manifest_bytes, source=_MANIFEST_FILENAME)
    if canonical_json_bytes(manifest) != manifest_bytes:
        _fail("replay manifest must use canonical JSON encoding")
    _strict_fields(
        manifest,
        frozenset({"files", "schema_version"}),
        "replay manifest",
    )
    if manifest["schema_version"] != POLICY_REPLAY_MANIFEST_SCHEMA_VERSION:
        _fail("unsupported replay manifest schema")
    expected_entries: dict[str, tuple[str, int]] = {}
    for raw in _sequence(manifest["files"], "manifest files"):
        entry = _mapping(raw, "manifest file")
        _strict_fields(
            entry,
            frozenset({"name", "sha256", "size_bytes"}),
            "manifest file",
        )
        name = _text(entry["name"], "manifest filename", maximum=64)
        if (
            name not in {_ARTIFACT_FILENAME, _ASSIGNMENT_FILENAME}
            or name in expected_entries
        ):
            _fail("manifest contains an unallowlisted or duplicate file")
        size = _integer(entry["size_bytes"], "size_bytes")
        if size < 0:
            _fail("manifest size_bytes must be non-negative")
        expected_entries[name] = (_sha256(entry["sha256"], "file sha256"), size)
    if frozenset(expected_entries) != {_ARTIFACT_FILENAME, _ASSIGNMENT_FILENAME}:
        _fail("manifest does not commit the complete replay file set")
    artifact_bytes = _read_regular(
        root / _ARTIFACT_FILENAME, maximum=_MAX_ARTIFACT_BYTES
    )
    assignment_bytes = _read_regular(
        root / _ASSIGNMENT_FILENAME, maximum=_MAX_ASSIGNMENT_BYTES
    )
    for name, content in (
        (_ARTIFACT_FILENAME, artifact_bytes),
        (_ASSIGNMENT_FILENAME, assignment_bytes),
    ):
        expected_hash, expected_size = expected_entries[name]
        if len(content) != expected_size or sha256_bytes(content) != expected_hash:
            _fail(f"{name} differs from the replay manifest commitment")
    return artifact_bytes, assignment_bytes


def load_synthetic_policy_replay_bundle(
    directory: Path,
) -> LoadedSyntheticPolicyReplay:
    """Strictly load committed synthetic inputs; never deserialize arbitrary code."""
    root = _absolute_local(directory)
    artifact_bytes, assignment_bytes = _verify_bundle_files(root)
    artifact = read_json_bytes_object(artifact_bytes, source=_ARTIFACT_FILENAME)
    if canonical_json_bytes(artifact) != artifact_bytes:
        _fail("replay artifact must use canonical JSON encoding")
    _strict_fields(
        artifact,
        frozenset(
            {
                "descriptor",
                "descriptor_sha256",
                "expected",
                "inputs",
                "schema_version",
                "synthetic_only",
            }
        ),
        "replay artifact",
    )
    if (
        artifact["schema_version"] != POLICY_REPLAY_BUNDLE_SCHEMA_VERSION
        or artifact["synthetic_only"] is not True
    ):
        raise PolicyEvidenceError(
            "loader accepts only the explicit synthetic replay bundle schema"
        )
    descriptor = SyntheticPolicyReplayDescriptor.from_mapping(
        _mapping(artifact["descriptor"], "descriptor"),
        assignment_source_bytes=assignment_bytes,
    )
    if descriptor.sha256 != _sha256(artifact["descriptor_sha256"], "descriptor_sha256"):
        _fail("policy replay descriptor differs from its commitment")
    inputs = _mapping(artifact["inputs"], "inputs")
    _strict_fields(
        inputs,
        frozenset(
            {
                "command_stream_sha256",
                "config",
                "config_sha256",
                "opening_state",
                "opening_state_sha256",
                "scheduled_commands",
            }
        ),
        "replay inputs",
    )
    config = _parse_market_config(_mapping(inputs["config"], "config"))
    if config.sha256 != _sha256(inputs["config_sha256"], "config_sha256"):
        _fail("market config differs from its commitment")
    opening_state = _parse_opening_state(
        _mapping(inputs["opening_state"], "opening_state")
    )
    opening_hash = sha256_bytes(canonical_json_bytes(opening_state.to_mapping()))
    if opening_hash != _sha256(inputs["opening_state_sha256"], "opening_state_sha256"):
        _fail("opening state differs from its commitment")
    scheduled = _parse_scheduled_commands(inputs["scheduled_commands"], config=config)
    command_hash = sha256_bytes(
        canonical_json_bytes(tuple(item.to_mapping() for item in scheduled))
    )
    if command_hash != _sha256(
        inputs["command_stream_sha256"], "command_stream_sha256"
    ):
        _fail("scheduled command stream differs from its commitment")
    expected = _mapping(artifact["expected"], "expected")
    _strict_fields(
        expected,
        frozenset(
            {
                "market_ledger",
                "policy_ledger",
                "quotes_sha256",
                "result_sha256",
                "state_sha256",
                "trades_sha256",
            }
        ),
        "replay expected outputs",
    )
    market_ledger = LedgerCommitment.from_mapping(
        _mapping(expected["market_ledger"], "market_ledger")
    )
    policy_ledger = LedgerCommitment.from_mapping(
        _mapping(expected["policy_ledger"], "policy_ledger")
    )
    if market_ledger.ledger != "market" or policy_ledger.ledger != "policy":
        _fail("market and policy ledger commitments are swapped")
    return LoadedSyntheticPolicyReplay(
        config=config,
        opening_state=opening_state,
        scheduled_commands=scheduled,
        descriptor=descriptor,
        assignment_path=root / _ASSIGNMENT_FILENAME,
        expected_result_sha256=_sha256(expected["result_sha256"], "result_sha256"),
        expected_state_sha256=_sha256(expected["state_sha256"], "state_sha256"),
        expected_trades_sha256=_sha256(expected["trades_sha256"], "trades_sha256"),
        expected_quotes_sha256=_sha256(expected["quotes_sha256"], "quotes_sha256"),
        market_ledger=market_ledger,
        policy_ledger=policy_ledger,
    )


def verify_loaded_replay_result(
    loaded: LoadedSyntheticPolicyReplay, result: ReplayResult
) -> None:
    """Compare exact outputs plus independent market and policy audit commitments."""
    market_ledger = ledger_commitment(
        "market", tuple(item.to_mapping() for item in result.audit_events)
    )
    policy_ledger = ledger_commitment(
        "policy", tuple(item.to_mapping() for item in result.policy_audit_events)
    )
    trades_sha256 = sha256_bytes(
        canonical_json_bytes(tuple(item.to_mapping() for item in result.trades))
    )
    quotes_sha256 = sha256_bytes(
        canonical_json_bytes(tuple(item.to_mapping() for item in result.quotes))
    )
    if (
        result.sha256 != loaded.expected_result_sha256
        or result.state.sha256 != loaded.expected_state_sha256
        or trades_sha256 != loaded.expected_trades_sha256
        or quotes_sha256 != loaded.expected_quotes_sha256
        or market_ledger != loaded.market_ledger
        or policy_ledger != loaded.policy_ledger
    ):
        _fail("replayed market, policy, or audit ledger differs from its commitment")


def replay_error_with_audit(error: Exception) -> PolicyReplayError:
    """Attach one stable rejection event to a persistence-layer failure."""
    failure = (
        error
        if isinstance(error, PolicyReplayError)
        else PolicyReplayError("synthetic policy replay bundle was rejected")
    )
    # The persistence loader has no live engine ledger yet.  A compact immutable
    # emergency record preserves fail-closed observability for callers.
    failure.audit_events = ()
    failure.policy_audit_events = ()
    return failure


__all__ = [
    "ImplementationCommitment",
    "LEDGER_COMMITMENT_SCHEMA_VERSION",
    "LedgerCommitment",
    "LoadedSyntheticPolicyReplay",
    "POLICY_REPLAY_BUNDLE_SCHEMA_VERSION",
    "POLICY_REPLAY_DESCRIPTOR_SCHEMA_VERSION",
    "POLICY_REPLAY_MANIFEST_SCHEMA_VERSION",
    "PolicyReplayError",
    "SyntheticPolicyReplayDescriptor",
    "current_implementation_commitments",
    "ledger_commitment",
    "load_synthetic_policy_replay_bundle",
    "verify_loaded_replay_result",
    "write_synthetic_policy_replay_bundle",
]
