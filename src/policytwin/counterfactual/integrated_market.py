"""Synthetic end-to-end agent, cognition, social, CLOB, and paired-world runner.

This module closes the research-engineering loop with deliberately synthetic
inputs.  It is not an SEC implementation and it is not empirical evidence.
Every exogenous realization is consumed through the pair engine, every agent
decision is made from a public projection plus its own account, and every
accepted command advances the market through a complete deterministic replay.
"""

from __future__ import annotations

import math
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal, NoReturn, Self, cast

from policytwin.agents import (
    AgentMarketBridge,
    DecisionAction,
    DeterministicMockPlanningHook,
    OwnPortfolioState,
    PublicCognitiveFeature,
    PublicFeatureCode,
    PublicMarketObservation,
    PublicSignal,
    ScenarioAgentAuthority,
    ScenarioFastOrderPolicy,
    ScenarioPublicMarketObservation,
    ScenarioPublicSignal,
    ScenarioVisibleOrderFlow,
    SignalKind,
    StrictLLMRequest,
    StructuredAgent,
)
from policytwin.behavior import BehaviorParameters
from policytwin.cognition import (
    AuditedCognitiveLayer,
    AuditedScenarioCognitiveLayer,
    CognitivePlan,
    CognitiveSnapshot,
    CognitiveState,
    EpisodicMemory,
    PlanAction,
    RequestedUse,
    ScenarioCognitivePlan,
    ScenarioCognitiveSnapshot,
    ScenarioEpisodicMemory,
    ScenarioSemanticState,
    SemanticState,
)
from policytwin.counterfactual.events import (
    ExogenousShock,
    ScenarioEventReceipt,
    ScenarioGeneratorSpec,
    ShockTransitionState,
)
from policytwin.counterfactual.paired import (
    GroupOutcome,
    GroupSpec,
    HorizonOutcome,
    MetricValue,
    StateArtifact,
    WorldDefinition,
    WorldOutcome,
    WorldRole,
    WorldRunner,
    WorldRunRequest,
    WorldStatus,
)
from policytwin.counterfactual.randomness import (
    AGENT_ACTION_CHOICE_DOMAIN,
    EVENT_KEY_SCHEMA_VERSION,
    INTEGRATED_OBSERVATION_NOISE_DOMAIN,
    RNG_ALGORITHM_SPEC_COMMITMENT,
    RNG_IMPLEMENTATION_COMMITMENT,
    RNG_IMPLEMENTATION_ID,
    EventKey,
    RandomDrawReceipt,
    assert_live_rng_implementation,
    deterministic_int_with_receipt,
)
from policytwin.market import (
    AccountBalance,
    AccountSnapshot,
    CancelOrder,
    CentralLimitOrderBook,
    MarketConfig,
    MarketOrder,
    MarketPolicyMode,
    MarketSnapshot,
    OpeningOrder,
    OpeningState,
    ReplayMarketSession,
    ReplayResult,
    ScheduledCommand,
    Side,
    TradeLogEntry,
)
from policytwin.market.models import (
    MARKET_SCHEMA_VERSION,
    SCHEDULER_SCHEMA_VERSION,
    LimitOrder,
    MarketCommand,
    command_to_mapping,
)
from policytwin.provenance import (
    canonical_json_bytes,
    read_json_bytes_object,
    sha256_bytes,
)
from policytwin.social import (
    InformationEvent,
    InformationEventKind,
    NodeBelief,
    PropagationConfig,
    ScenarioInformationEvent,
    SeededInformationEvent,
    SocialAuditLog,
    SocialDiffusionEngine,
    SyntheticSocialGraph,
    VisibleOrderFlow,
    scenario_information_payload_sha256,
)

INTEGRATED_CONFIG_SCHEMA_VERSION = "policytwin.integrated-synthetic-market.v2"
INTEGRATED_MARKET_SCHEMA_VERSION = INTEGRATED_CONFIG_SCHEMA_VERSION
INTEGRATED_RUN_ARTIFACT_SCHEMA_VERSION = (
    "policytwin.integrated-synthetic-run-artifact.v4"
)
RANDOM_CONSUMPTION_ENTRY_SCHEMA_VERSION = (
    "policytwin.integrated-random-consumption-entry.v1"
)
RANDOM_CONSUMPTION_LEDGER_SCHEMA_VERSION = (
    "policytwin.integrated-random-consumption-ledger.v1"
)
INTEGRATED_MARKET_CLAIM_SCOPE = "synthetic_mechanism_closure_only"
INTEGRATED_RUN_ARTIFACT_CLAIM_SCOPE = "synthetic_clob_replay_only"
INTEGRATED_MODEL_ID = "integrated-synthetic-market-v1"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_NULL_HASH = "0" * 64
_POSTERIOR_FIELDS = frozenset({"behavior_scale", "signal_scale"})
_POLICY_FIELDS = frozenset({"allow_buy", "allow_sell"})
_MAX_INTEGRATED_ARTIFACT_BYTES = 64 * 1024 * 1024
_MAX_RANDOM_LEDGER_STEPS = 100_000
_MAX_RANDOM_LEDGER_AGENTS = 10_000
_MAX_RANDOM_LEDGER_ENTRIES = 1_000_000
_OBSERVATION_DRAW_LOWER = -1000
_OBSERVATION_DRAW_UPPER = 1000
_ACTION_DRAW_LOWER = 0
_ACTION_DRAW_UPPER = 10**12 - 1
_MARKET_STEP_STRIDE = 1000
_EXOGENOUS_COMMAND_OFFSET = _MARKET_STEP_STRIDE - 1


class IntegratedMarketError(RuntimeError):
    """Raised when the synthetic closed-loop contract cannot be proven."""


class IntegratedDecisionOutcome(StrEnum):
    """Closed result of applying one slow plan to one fast decision."""

    APPLIED = "applied"
    HOLD = "hold"
    PLAN_VETO = "plan_veto"


class RandomConsumptionKind(StrEnum):
    """Closed internal draw vocabulary for one integrated agent decision."""

    OBSERVATION_NOISE = "observation_noise"
    ACTION_CHOICE = "action_choice"


def _fail(message: str) -> NoReturn:
    raise IntegratedMarketError(message)


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        _fail(f"{field} must be a stable ASCII identifier")
    return value


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail(f"{field} must be a lowercase SHA-256 digest")
    return value


def _aware_utc(value: object, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        _fail(f"{field} must be timezone-aware")
    try:
        normalized = value.astimezone(UTC)
    except (OverflowError, ValueError) as error:
        raise IntegratedMarketError(f"{field} is outside UTC range") from error
    return normalized


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        _fail(f"{field} must be a finite real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        _fail(f"{field} must be finite")
    return normalized


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


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _fail(f"{field} must be a positive integer")
    return value


def _commit(value: object) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _stat_fingerprint(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _agent_command_time(sequence: int, agent_index: int) -> int:
    """Place every agent command after the prior shock and before this shock."""
    if sequence <= 0 or agent_index < 0 or agent_index >= _EXOGENOUS_COMMAND_OFFSET:
        _fail("integrated agent command lies outside the per-step time budget")
    return sequence * _MARKET_STEP_STRIDE + agent_index


def _exogenous_command_time(sequence: int) -> int:
    if sequence <= 0:
        _fail("integrated exogenous command sequence must be positive")
    return sequence * _MARKET_STEP_STRIDE + _EXOGENOUS_COMMAND_OFFSET


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
        "integrated replay event key",
    )
    if value["schema_version"] != EVENT_KEY_SCHEMA_VERSION:
        _fail("integrated replay event key uses an unsupported schema")
    key = EventKey(
        experiment_id=_text(value["experiment_id"], "experiment_id", maximum=128),
        seed=_text(value["seed"], "seed"),
        replicate_id=_integer(value["replicate_id"], "replicate_id"),
        simulation_step=_integer(value["simulation_step"], "simulation_step"),
        purpose=_text(value["purpose"], "purpose"),
    )
    if key.replicate_id < 0 or key.simulation_step < 0 or value["value"] != key.value:
        _fail("integrated replay event key differs from its derived value")
    return key


def _parse_random_draw_receipt(value: object, field: str) -> RandomDrawReceipt:
    """Recompute an untrusted random draw mapping under the frozen RNG code."""
    try:
        return RandomDrawReceipt.from_mapping(_mapping(value, field))
    except (TypeError, ValueError) as error:
        raise IntegratedMarketError(
            f"{field} failed independent deterministic replay"
        ) from error


def _parse_side(value: object) -> Side:
    if not isinstance(value, str):
        _fail("integrated replay side must be text")
    try:
        return Side(value)
    except ValueError as error:
        raise IntegratedMarketError("integrated replay side is unsupported") from error


def _parse_command(value: Mapping[str, object]) -> MarketCommand:
    command_type = value.get("command_type")
    common = {"command_type", "event_key", "submitted_at", "trader_id"}
    if command_type == "limit_order":
        _strict_fields(
            value,
            frozenset(common | {"order_id", "side", "quantity", "price"}),
            "integrated replay limit order",
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
            "integrated replay market order",
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
            "integrated replay cancel order",
        )
        return CancelOrder(
            event_key=_parse_event_key(_mapping(value["event_key"], "event_key")),
            submitted_at=_integer(value["submitted_at"], "submitted_at"),
            trader_id=_text(value["trader_id"], "trader_id", maximum=128),
            target_order_id=_text(
                value["target_order_id"], "target_order_id", maximum=128
            ),
        )
    _fail("integrated replay command type is outside the closed allowlist")


def _parse_market_config(value: Mapping[str, object]) -> MarketConfig:
    _strict_fields(
        value,
        frozenset(
            {
                "latency_steps",
                "maker_fee_bps",
                "policy_mode",
                "scheduler_schema_version",
                "schema_version",
                "taker_fee_bps",
                "tick_size",
            }
        ),
        "integrated replay market config",
    )
    if (
        value["schema_version"] != MARKET_SCHEMA_VERSION
        or value["scheduler_schema_version"] != SCHEDULER_SCHEMA_VERSION
        or value["policy_mode"] != MarketPolicyMode.NO_POLICY_SYNTHETIC.value
    ):
        _fail("integrated replay requires the current synthetic no-policy schema")
    return MarketConfig(
        tick_size=_integer(value["tick_size"], "tick_size"),
        latency_steps=_integer(value["latency_steps"], "latency_steps"),
        maker_fee_bps=_integer(value["maker_fee_bps"], "maker_fee_bps"),
        taker_fee_bps=_integer(value["taker_fee_bps"], "taker_fee_bps"),
        policy_mode=MarketPolicyMode.NO_POLICY_SYNTHETIC,
    )


def _parse_account(value: Mapping[str, object]) -> AccountBalance:
    _strict_fields(
        value,
        frozenset({"asset", "cash", "trader_id"}),
        "integrated replay opening account",
    )
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
                "price",
                "priority_time",
                "quantity",
                "side",
                "trader_id",
            }
        ),
        "integrated replay opening order",
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


def _parse_opening_state(value: Mapping[str, object]) -> OpeningState:
    _strict_fields(
        value,
        frozenset({"accounts", "orders", "venue_cash"}),
        "integrated replay opening state",
    )
    return OpeningState(
        accounts=tuple(
            _parse_account(_mapping(item, "opening account"))
            for item in _sequence(value["accounts"], "accounts")
        ),
        venue_cash=_integer(value["venue_cash"], "venue_cash"),
        orders=tuple(
            _parse_opening_order(_mapping(item, "opening order"))
            for item in _sequence(value["orders"], "orders")
        ),
    )


def _parse_scheduled_commands(
    value: object, *, config: MarketConfig
) -> tuple[ScheduledCommand, ...]:
    scheduled: list[ScheduledCommand] = []
    for item in _sequence(value, "scheduled_commands"):
        mapping = _mapping(item, "scheduled command")
        _strict_fields(
            mapping,
            frozenset({"arrival_time", "command", "event_key"}),
            "integrated replay scheduled command",
        )
        command = _parse_command(_mapping(mapping["command"], "command"))
        arrival_time = _integer(mapping["arrival_time"], "arrival_time")
        if (
            arrival_time != command.submitted_at + config.latency_steps
            or mapping["event_key"] != command.event_key.value
        ):
            _fail("integrated scheduled command differs from its derived fields")
        scheduled.append(
            ScheduledCommand(
                arrival_time=arrival_time,
                event_key=command.event_key.value,
                command=command,
            )
        )
    ordered = tuple(
        sorted(scheduled, key=lambda item: (item.arrival_time, item.event_key))
    )
    if tuple(scheduled) != ordered:
        _fail("integrated scheduled commands are not in deterministic order")
    return ordered


def _replay_from_serialized_mapping(value: Mapping[str, object]) -> ReplayResult:
    """Rebuild a no-policy synthetic replay; serialized outputs have no authority."""
    _strict_fields(
        value,
        frozenset(
            {
                "audit_events",
                "command_stream_sha256",
                "config",
                "config_sha256",
                "event_keys",
                "opening_state",
                "opening_state_sha256",
                "policy_audit_events",
                "policy_gateway_config",
                "policy_gateway_sha256",
                "policy_replay_descriptor",
                "policy_replay_descriptor_sha256",
                "quotes",
                "schema_version",
                "scheduled_commands",
                "state",
                "state_sha256",
                "trades",
            }
        ),
        "integrated market replay",
    )
    if value["schema_version"] != MARKET_SCHEMA_VERSION:
        _fail("integrated market replay uses an unsupported schema")
    if any(
        value[field] is not None
        for field in (
            "policy_gateway_config",
            "policy_gateway_sha256",
            "policy_replay_descriptor",
            "policy_replay_descriptor_sha256",
        )
    ) or _sequence(value["policy_audit_events"], "policy_audit_events"):
        _fail("integrated market replay cannot contain policy-gateway evidence")
    config = _parse_market_config(_mapping(value["config"], "config"))
    opening_state = _parse_opening_state(
        _mapping(value["opening_state"], "opening_state")
    )
    scheduled = _parse_scheduled_commands(value["scheduled_commands"], config=config)
    try:
        engine = CentralLimitOrderBook(config, opening_state)
        if scheduled:
            engine.schedule_all(item.command for item in scheduled)
        replayed = engine.run()
        engine.assert_invariants()
    except Exception as error:
        raise IntegratedMarketError(
            "integrated serialized market replay failed live reconstruction"
        ) from error
    if replayed.canonical_bytes != canonical_json_bytes(dict(value)):
        _fail("integrated serialized market replay differs from live replay")
    return replayed


def _live_replay(artifact: ReplayResult) -> ReplayResult:
    try:
        return CentralLimitOrderBook.replay(artifact)
    except Exception as error:
        raise IntegratedMarketError(
            "integrated market replay failed independent live replay"
        ) from error


def _replay_prefix(
    artifact: ReplayResult, command_event_keys: tuple[str, ...]
) -> ReplayResult:
    if len(command_event_keys) > len(artifact.scheduled_commands):
        _fail("horizon command_count is outside the terminal replay")
    commands_by_key = {
        item.event_key: item.command for item in artifact.scheduled_commands
    }
    if len(commands_by_key) != len(artifact.scheduled_commands):
        _fail("terminal replay contains duplicate command event keys")
    if any(key not in commands_by_key for key in command_event_keys):
        _fail("horizon command event key is absent from the terminal replay")
    try:
        engine = CentralLimitOrderBook(artifact.config, artifact.opening_state)
        commands = tuple(commands_by_key[key] for key in command_event_keys)
        if commands:
            engine.schedule_all(commands)
        replayed = engine.run()
        engine.assert_invariants()
        return replayed
    except Exception as error:
        raise IntegratedMarketError(
            "integrated horizon boundary failed independent live replay"
        ) from error


def _agent_mapping(agent: StructuredAgent) -> dict[str, object]:
    return {
        "agent_id": agent.agent_id,
        "archetype": agent.archetype.value,
        "cognitive_update_interval_steps": agent.cognitive_update_interval_steps,
        "mechanisms": agent.mechanisms.to_mapping(),
        "parameters": agent.parameters.to_mapping(),
        "policy_config": agent.policy_config.to_mapping(),
        "risk_limits": agent.risk_limits.to_mapping(),
        "trusted_cutoff": agent.trusted_cutoff.isoformat(),
    }


@dataclass(frozen=True, slots=True)
class IntegratedSyntheticMarketConfig:
    """Frozen synthetic system inputs shared by both policy worlds."""

    cutoff: datetime
    cutoff_policy_sha256: str
    source_lineage_sha256: str
    shared_seed: str
    market_config: MarketConfig
    opening_state: OpeningState
    agents: tuple[StructuredAgent, ...]
    social_graph: SyntheticSocialGraph
    group_members: tuple[tuple[str, tuple[str, ...]], ...]
    transition_state_field: str = "latent_fundamental_ticks"
    transition_shock_field: str = "fundamental_ticks"
    exogenous_buyer_id: str = "exogenous-buyer"
    exogenous_seller_id: str = "exogenous-seller"
    exogenous_order_quantity: int = 2
    social_enabled: bool = True
    social_max_hops: int = 3
    social_attention_capacity: int = 8
    social_repost_threshold: float = 0.1
    memory_enabled: bool = True
    memory_capacity: int = 8
    mock_planning_hook: DeterministicMockPlanningHook | None = None
    scenario_generator_spec: ScenarioGeneratorSpec | None = None
    synthetic_only: Literal[True] = True
    schema_version: str = INTEGRATED_MARKET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        cutoff = _aware_utc(self.cutoff, "cutoff")
        object.__setattr__(self, "cutoff", cutoff)
        for field in ("cutoff_policy_sha256", "source_lineage_sha256"):
            _sha256(getattr(self, field), field)
        _identifier(self.shared_seed, "shared_seed")
        if not isinstance(self.market_config, MarketConfig):
            _fail("market_config must be a MarketConfig")
        if not isinstance(self.opening_state, OpeningState):
            _fail("opening_state must be an OpeningState")
        if not isinstance(self.agents, tuple) or not self.agents:
            _fail("agents must be a non-empty immutable tuple")
        if not all(isinstance(item, StructuredAgent) for item in self.agents):
            _fail("agents must contain StructuredAgent values")
        if len(self.agents) > _EXOGENOUS_COMMAND_OFFSET:
            _fail(
                "agents exceed the 999-command capacity of one frozen logical "
                "market step"
            )
        ordered_agents = tuple(sorted(self.agents, key=lambda item: item.agent_id))
        agent_ids = tuple(item.agent_id for item in ordered_agents)
        if len(set(agent_ids)) != len(agent_ids):
            _fail("agent IDs must be unique")
        for agent in ordered_agents:
            if agent.trusted_cutoff != cutoff:
                _fail("every agent must share the configured trusted cutoff")
            if agent.policy_config.tick_size != self.market_config.tick_size:
                _fail("agent and market tick sizes must match")
            if agent.risk_limits.fee_bps < max(
                self.market_config.maker_fee_bps,
                self.market_config.taker_fee_bps,
            ):
                _fail("agent fee bound is below the market fee")
        object.__setattr__(self, "agents", ordered_agents)
        if not isinstance(self.social_graph, SyntheticSocialGraph):
            _fail("integrated synthetic runner requires a SyntheticSocialGraph")
        if self.social_graph.nodes != tuple(sorted(agent_ids)):
            _fail("social graph nodes must exactly equal the agent IDs")
        account_ids = {item.trader_id for item in self.opening_state.accounts}
        required_accounts = {
            *agent_ids,
            self.exogenous_buyer_id,
            self.exogenous_seller_id,
        }
        if not required_accounts.issubset(account_ids):
            _fail("opening state omits an agent or exogenous liquidity account")
        _identifier(self.exogenous_buyer_id, "exogenous_buyer_id")
        _identifier(self.exogenous_seller_id, "exogenous_seller_id")
        if self.exogenous_buyer_id == self.exogenous_seller_id:
            _fail("exogenous buyer and seller must differ")
        _identifier(self.transition_state_field, "transition_state_field")
        _identifier(self.transition_shock_field, "transition_shock_field")
        _positive_int(self.exogenous_order_quantity, "exogenous_order_quantity")
        if not isinstance(self.social_enabled, bool):
            _fail("social_enabled must be boolean")
        if not isinstance(self.memory_enabled, bool):
            _fail("memory_enabled must be boolean")
        if (
            isinstance(self.social_max_hops, bool)
            or not isinstance(self.social_max_hops, int)
            or self.social_max_hops < 0
        ):
            _fail("social_max_hops must be non-negative")
        _positive_int(self.social_attention_capacity, "social_attention_capacity")
        threshold = _finite(self.social_repost_threshold, "social_repost_threshold")
        if threshold < 0.0:
            _fail("social_repost_threshold must be non-negative")
        object.__setattr__(self, "social_repost_threshold", threshold)
        _positive_int(self.memory_capacity, "memory_capacity")
        if self.mock_planning_hook is not None:
            if not isinstance(self.mock_planning_hook, DeterministicMockPlanningHook):
                _fail("mock_planning_hook has the wrong type")
            authorization = self.mock_planning_hook.authorization
            if (
                authorization.trusted_cutoff != cutoff
                or authorization.cutoff_policy_sha256 != self.cutoff_policy_sha256
            ):
                _fail("mock planning authorization differs from the cutoff config")
            if (
                self.mock_planning_hook.sanitized_context_record.lineage_sha256
                != self.source_lineage_sha256
            ):
                _fail("mock planning context lineage differs from the market config")
            if self.scenario_generator_spec is not None:
                _fail(
                    "deterministic mock planning is pre-cutoff synthetic evidence "
                    "only and cannot enter the post-cutoff scenario path"
                )
        if self.scenario_generator_spec is not None:
            if not isinstance(self.scenario_generator_spec, ScenarioGeneratorSpec):
                _fail("scenario_generator_spec has the wrong type")
            spec = self.scenario_generator_spec
            if (
                spec.trusted_cutoff != cutoff
                or spec.cutoff_policy_sha256 != self.cutoff_policy_sha256
                or spec.source_lineage_sha256 != self.source_lineage_sha256
            ):
                _fail("scenario generator differs from cutoff or source-lineage config")
        if not isinstance(self.group_members, tuple) or not self.group_members:
            _fail("group_members must be a non-empty immutable tuple")
        normalized_groups: list[tuple[str, tuple[str, ...]]] = []
        seen_groups: set[str] = set()
        for group_id, members in self.group_members:
            stable_group = _identifier(group_id, "group_id")
            if stable_group in seen_groups:
                _fail("group IDs must be unique")
            if not isinstance(members, tuple) or not members:
                _fail("every group must have immutable members")
            stable_members = tuple(
                sorted(_identifier(item, "member") for item in members)
            )
            if len(set(stable_members)) != len(stable_members):
                _fail("group members must be unique")
            if not set(stable_members).issubset(agent_ids):
                _fail("groups may contain only configured synthetic agents")
            seen_groups.add(stable_group)
            normalized_groups.append((stable_group, stable_members))
        object.__setattr__(
            self,
            "group_members",
            tuple(sorted(normalized_groups, key=lambda item: item[0])),
        )
        if self.synthetic_only is not True:
            _fail("integrated runner is synthetic-only")
        if self.schema_version != INTEGRATED_MARKET_SCHEMA_VERSION:
            _fail("unsupported integrated-market schema version")

    def members_for(self, group_id: str) -> tuple[str, ...]:
        """Return frozen members for a registered latent strategy group."""
        for identifier, members in self.group_members:
            if identifier == group_id:
                return members
        raise IntegratedMarketError(f"unregistered group: {group_id}")

    def to_mapping(self) -> dict[str, object]:
        """Return every setting that can change the integrated trajectory."""
        mapping: dict[str, object] = {
            "agents": [_agent_mapping(item) for item in self.agents],
            "claim_scope": INTEGRATED_MARKET_CLAIM_SCOPE,
            "cutoff": self.cutoff.isoformat(),
            "cutoff_policy_sha256": self.cutoff_policy_sha256,
            "exogenous_buyer_id": self.exogenous_buyer_id,
            "exogenous_order_quantity": self.exogenous_order_quantity,
            "exogenous_seller_id": self.exogenous_seller_id,
            "group_members": [
                {"group_id": group_id, "members": list(members)}
                for group_id, members in self.group_members
            ],
            "market_config": self.market_config.to_mapping(),
            "memory_enabled": self.memory_enabled,
            "memory_capacity": self.memory_capacity,
            "opening_state": self.opening_state.to_mapping(),
            "schema_version": self.schema_version,
            "scenario_generator_spec": (
                None
                if self.scenario_generator_spec is None
                else self.scenario_generator_spec.to_mapping()
            ),
            "shared_seed_sha256": sha256_bytes(self.shared_seed.encode("utf-8")),
            "social_attention_capacity": self.social_attention_capacity,
            "social_enabled": self.social_enabled,
            "social_graph": self.social_graph.to_mapping(),
            "social_max_hops": self.social_max_hops,
            "social_repost_threshold": self.social_repost_threshold,
            "source_lineage_sha256": self.source_lineage_sha256,
            "synthetic_only": self.synthetic_only,
            "transition_shock_field": self.transition_shock_field,
            "transition_state_field": self.transition_state_field,
        }
        # Keep the historical no-hook config commitment stable while binding
        # every effective mock setting whenever the optional mechanism is live.
        if self.mock_planning_hook is not None:
            mapping["mock_planning_hook"] = self.mock_planning_hook.to_mapping()
        return mapping

    @property
    def sha256(self) -> str:
        """Return the integrated-system commitment."""
        return _commit(self.to_mapping())


@dataclass(frozen=True, slots=True)
class IntegratedDecisionReceipt:
    """Hash-only proof that a slow plan gated one fast decision."""

    agent_id: str
    event_key: str
    action: str
    outcome: IntegratedDecisionOutcome
    buy_probability: float
    hold_probability: float
    sell_probability: float
    plan_sha256: str
    trace_sha256: str
    observation_noise_draw: RandomDrawReceipt
    action_choice_draw: RandomDrawReceipt
    bridge_binding_sha256: str
    market_state_before_sha256: str
    market_state_after_sha256: str
    mock_adapter_id: str | None = None
    mock_authorization_sha256: str | None = None
    mock_request_sha256: str | None = None
    mock_rationale_code: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.agent_id, "agent_id")
        if not isinstance(self.event_key, str) or not self.event_key:
            _fail("decision event_key must be non-empty")
        _identifier(self.action, "action")
        if not isinstance(self.outcome, IntegratedDecisionOutcome):
            _fail("decision outcome has the wrong type")
        if self.action not in {item.value for item in DecisionAction}:
            _fail("decision action is unsupported")
        probabilities = tuple(
            _finite(getattr(self, field), field)
            for field in (
                "buy_probability",
                "hold_probability",
                "sell_probability",
            )
        )
        if any(value < 0.0 or value > 1.0 for value in probabilities):
            _fail("decision probabilities must be in [0, 1]")
        if not math.isclose(math.fsum(probabilities), 1.0, abs_tol=1e-12):
            _fail("decision probabilities must sum to one")
        for field in ("observation_noise_draw", "action_choice_draw"):
            receipt = getattr(self, field)
            if not isinstance(receipt, RandomDrawReceipt):
                _fail(f"{field} must be a factory-produced RandomDrawReceipt")
            try:
                recomputed = RandomDrawReceipt.from_mapping(receipt.to_mapping())
            except (TypeError, ValueError) as error:
                raise IntegratedMarketError(
                    f"{field} failed independent deterministic replay"
                ) from error
            if recomputed != receipt or receipt.event_key.value != self.event_key:
                _fail(f"{field} differs from the decision event key")
        if (
            self.observation_noise_draw.domain != INTEGRATED_OBSERVATION_NOISE_DOMAIN
            or self.observation_noise_draw.draw_index != 0
            or self.observation_noise_draw.lower != _OBSERVATION_DRAW_LOWER
            or self.observation_noise_draw.upper != _OBSERVATION_DRAW_UPPER
        ):
            _fail("observation noise draw differs from the frozen draw contract")
        if (
            self.action_choice_draw.domain != AGENT_ACTION_CHOICE_DOMAIN
            or self.action_choice_draw.draw_index != 0
            or self.action_choice_draw.lower != _ACTION_DRAW_LOWER
            or self.action_choice_draw.upper != _ACTION_DRAW_UPPER
        ):
            _fail("action choice draw differs from the frozen draw contract")
        probability_draw = (self.action_choice_draw.integer_value + 0.5) / (
            _ACTION_DRAW_UPPER + 1
        )
        expected_action = (
            DecisionAction.BUY
            if probability_draw < self.buy_probability
            else (
                DecisionAction.HOLD
                if probability_draw < self.buy_probability + self.hold_probability
                else DecisionAction.SELL
            )
        )
        if self.action != expected_action.value:
            _fail("decision action differs from its receipted probability draw")
        for field in (
            "plan_sha256",
            "trace_sha256",
            "bridge_binding_sha256",
            "market_state_before_sha256",
            "market_state_after_sha256",
        ):
            _sha256(getattr(self, field), field)
        if self.outcome is IntegratedDecisionOutcome.PLAN_VETO:
            if self.bridge_binding_sha256 != _NULL_HASH:
                _fail("plan-veto receipt cannot contain a bridge binding")
            if self.market_state_before_sha256 != self.market_state_after_sha256:
                _fail("plan veto cannot mutate market state")
        mock_fields = (
            self.mock_adapter_id,
            self.mock_authorization_sha256,
            self.mock_request_sha256,
            self.mock_rationale_code,
        )
        if any(item is not None for item in mock_fields):
            if any(item is None for item in mock_fields):
                _fail("mock planning evidence must be complete or absent")
            _identifier(self.mock_adapter_id, "mock_adapter_id")
            _sha256(
                self.mock_authorization_sha256,
                "mock_authorization_sha256",
            )
            _sha256(self.mock_request_sha256, "mock_request_sha256")
            _identifier(self.mock_rationale_code, "mock_rationale_code")

    def to_mapping(self) -> dict[str, object]:
        mapping: dict[str, object] = {
            "action": self.action,
            "agent_id": self.agent_id,
            "action_choice_draw": self.action_choice_draw.to_mapping(),
            "bridge_binding_sha256": self.bridge_binding_sha256,
            "buy_probability": self.buy_probability,
            "event_key": self.event_key,
            "market_state_after_sha256": self.market_state_after_sha256,
            "market_state_before_sha256": self.market_state_before_sha256,
            "observation_noise_draw": self.observation_noise_draw.to_mapping(),
            "outcome": self.outcome.value,
            "hold_probability": self.hold_probability,
            "plan_sha256": self.plan_sha256,
            "trace_sha256": self.trace_sha256,
            "sell_probability": self.sell_probability,
        }
        if self.mock_adapter_id is not None:
            mapping["mock_planning_evidence"] = {
                "adapter_id": self.mock_adapter_id,
                "authorization_sha256": self.mock_authorization_sha256,
                "rationale_code": self.mock_rationale_code,
                "request_sha256": self.mock_request_sha256,
            }
        return mapping

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        """Strictly rebuild one decision receipt from its closed schema."""
        base_fields = frozenset(
            {
                "action",
                "agent_id",
                "action_choice_draw",
                "bridge_binding_sha256",
                "buy_probability",
                "event_key",
                "hold_probability",
                "market_state_after_sha256",
                "market_state_before_sha256",
                "observation_noise_draw",
                "outcome",
                "plan_sha256",
                "sell_probability",
                "trace_sha256",
            }
        )
        actual = frozenset(value)
        if actual not in (base_fields, base_fields | {"mock_planning_evidence"}):
            _strict_fields(value, base_fields, "integrated decision receipt")
        try:
            outcome = IntegratedDecisionOutcome(
                _text(value["outcome"], "outcome", maximum=32)
            )
        except ValueError as error:
            raise IntegratedMarketError(
                "integrated decision outcome is unsupported"
            ) from error
        mock_adapter_id: str | None = None
        mock_authorization_sha256: str | None = None
        mock_request_sha256: str | None = None
        mock_rationale_code: str | None = None
        if "mock_planning_evidence" in value:
            evidence = _mapping(
                value["mock_planning_evidence"], "mock_planning_evidence"
            )
            _strict_fields(
                evidence,
                frozenset(
                    {
                        "adapter_id",
                        "authorization_sha256",
                        "rationale_code",
                        "request_sha256",
                    }
                ),
                "mock planning evidence",
            )
            mock_adapter_id = _identifier(evidence["adapter_id"], "adapter_id")
            mock_authorization_sha256 = _sha256(
                evidence["authorization_sha256"], "authorization_sha256"
            )
            mock_request_sha256 = _sha256(evidence["request_sha256"], "request_sha256")
            mock_rationale_code = _identifier(
                evidence["rationale_code"], "rationale_code"
            )
        return cls(
            agent_id=_identifier(value["agent_id"], "agent_id"),
            event_key=_text(value["event_key"], "event_key"),
            action=_identifier(value["action"], "action"),
            outcome=outcome,
            buy_probability=_finite(value["buy_probability"], "buy_probability"),
            hold_probability=_finite(value["hold_probability"], "hold_probability"),
            sell_probability=_finite(value["sell_probability"], "sell_probability"),
            plan_sha256=_sha256(value["plan_sha256"], "plan_sha256"),
            trace_sha256=_sha256(value["trace_sha256"], "trace_sha256"),
            observation_noise_draw=_parse_random_draw_receipt(
                value["observation_noise_draw"], "observation_noise_draw"
            ),
            action_choice_draw=_parse_random_draw_receipt(
                value["action_choice_draw"], "action_choice_draw"
            ),
            bridge_binding_sha256=_sha256(
                value["bridge_binding_sha256"], "bridge_binding_sha256"
            ),
            market_state_before_sha256=_sha256(
                value["market_state_before_sha256"],
                "market_state_before_sha256",
            ),
            market_state_after_sha256=_sha256(
                value["market_state_after_sha256"],
                "market_state_after_sha256",
            ),
            mock_adapter_id=mock_adapter_id,
            mock_authorization_sha256=mock_authorization_sha256,
            mock_request_sha256=mock_request_sha256,
            mock_rationale_code=mock_rationale_code,
        )

    @property
    def sha256(self) -> str:
        return _commit(self.to_mapping())


@dataclass(frozen=True, slots=True)
class IntegratedStepReceipt:
    """One shock-to-social-to-cognition-to-orders-to-market transition."""

    sequence: int
    previous_step_sha256: str
    shock_event_key: str
    shock_transition_signature_sha256: str
    information_event_sha256: str
    diffusion_sha256: str
    cognitive_state_sha256: str
    decisions: tuple[IntegratedDecisionReceipt, ...]
    exogenous_command_sha256: str
    market_replay_sha256: str
    market_state_sha256: str
    social_audit_tail_sha256: str
    bridge_audit_sha256: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence <= 0
        ):
            _fail("integrated step sequence must be positive")
        if not isinstance(self.shock_event_key, str) or not self.shock_event_key:
            _fail("shock_event_key must be non-empty")
        for field in (
            "previous_step_sha256",
            "shock_transition_signature_sha256",
            "information_event_sha256",
            "diffusion_sha256",
            "cognitive_state_sha256",
            "exogenous_command_sha256",
            "market_replay_sha256",
            "market_state_sha256",
            "social_audit_tail_sha256",
            "bridge_audit_sha256",
        ):
            _sha256(getattr(self, field), field)
        if not isinstance(self.decisions, tuple) or not all(
            isinstance(item, IntegratedDecisionReceipt) for item in self.decisions
        ):
            _fail("decisions must contain immutable decision receipts")
        if (
            tuple(sorted(self.decisions, key=lambda item: item.agent_id))
            != self.decisions
        ):
            _fail("decision receipts must use canonical agent order")

    def to_mapping(self) -> dict[str, object]:
        return {
            "bridge_audit_sha256": self.bridge_audit_sha256,
            "cognitive_state_sha256": self.cognitive_state_sha256,
            "decisions": [item.to_mapping() for item in self.decisions],
            "diffusion_sha256": self.diffusion_sha256,
            "exogenous_command_sha256": self.exogenous_command_sha256,
            "information_event_sha256": self.information_event_sha256,
            "market_replay_sha256": self.market_replay_sha256,
            "market_state_sha256": self.market_state_sha256,
            "previous_step_sha256": self.previous_step_sha256,
            "sequence": self.sequence,
            "shock_event_key": self.shock_event_key,
            "shock_transition_signature_sha256": (
                self.shock_transition_signature_sha256
            ),
            "social_audit_tail_sha256": self.social_audit_tail_sha256,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        """Strictly rebuild one hash-linked mechanism step."""
        _strict_fields(
            value,
            frozenset(
                {
                    "bridge_audit_sha256",
                    "cognitive_state_sha256",
                    "decisions",
                    "diffusion_sha256",
                    "exogenous_command_sha256",
                    "information_event_sha256",
                    "market_replay_sha256",
                    "market_state_sha256",
                    "previous_step_sha256",
                    "sequence",
                    "shock_event_key",
                    "shock_transition_signature_sha256",
                    "social_audit_tail_sha256",
                }
            ),
            "integrated step receipt",
        )
        return cls(
            sequence=_integer(value["sequence"], "sequence"),
            previous_step_sha256=_sha256(
                value["previous_step_sha256"], "previous_step_sha256"
            ),
            shock_event_key=_text(value["shock_event_key"], "shock_event_key"),
            shock_transition_signature_sha256=_sha256(
                value["shock_transition_signature_sha256"],
                "shock_transition_signature_sha256",
            ),
            information_event_sha256=_sha256(
                value["information_event_sha256"], "information_event_sha256"
            ),
            diffusion_sha256=_sha256(value["diffusion_sha256"], "diffusion_sha256"),
            cognitive_state_sha256=_sha256(
                value["cognitive_state_sha256"], "cognitive_state_sha256"
            ),
            decisions=tuple(
                IntegratedDecisionReceipt.from_mapping(
                    _mapping(item, "integrated decision receipt")
                )
                for item in _sequence(value["decisions"], "decisions")
            ),
            exogenous_command_sha256=_sha256(
                value["exogenous_command_sha256"], "exogenous_command_sha256"
            ),
            market_replay_sha256=_sha256(
                value["market_replay_sha256"], "market_replay_sha256"
            ),
            market_state_sha256=_sha256(
                value["market_state_sha256"], "market_state_sha256"
            ),
            social_audit_tail_sha256=_sha256(
                value["social_audit_tail_sha256"], "social_audit_tail_sha256"
            ),
            bridge_audit_sha256=_sha256(
                value["bridge_audit_sha256"], "bridge_audit_sha256"
            ),
        )

    @property
    def sha256(self) -> str:
        return _commit(self.to_mapping())


@dataclass(frozen=True, slots=True)
class RandomConsumptionEntry:
    """One position-bound internal random draw in a single-world run."""

    step_sequence: int
    agent_id: str
    kind: RandomConsumptionKind
    receipt: RandomDrawReceipt
    schema_version: str = RANDOM_CONSUMPTION_ENTRY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            isinstance(self.step_sequence, bool)
            or not isinstance(self.step_sequence, int)
            or self.step_sequence <= 0
            or self.step_sequence > _MAX_RANDOM_LEDGER_STEPS
        ):
            _fail("random consumption step exceeds the bounded ledger range")
        _identifier(self.agent_id, "random consumption agent_id")
        if not isinstance(self.kind, RandomConsumptionKind):
            _fail("random consumption kind is unsupported")
        if not isinstance(self.receipt, RandomDrawReceipt):
            _fail("random consumption receipt was not factory-produced")
        try:
            recomputed = RandomDrawReceipt.from_mapping(self.receipt.to_mapping())
        except (TypeError, ValueError) as error:
            raise IntegratedMarketError(
                "random consumption receipt failed deterministic replay"
            ) from error
        if recomputed != self.receipt:
            _fail("random consumption receipt changed after deterministic replay")
        expected = (
            (
                INTEGRATED_OBSERVATION_NOISE_DOMAIN,
                _OBSERVATION_DRAW_LOWER,
                _OBSERVATION_DRAW_UPPER,
            )
            if self.kind is RandomConsumptionKind.OBSERVATION_NOISE
            else (
                AGENT_ACTION_CHOICE_DOMAIN,
                _ACTION_DRAW_LOWER,
                _ACTION_DRAW_UPPER,
            )
        )
        if (
            self.receipt.domain,
            self.receipt.lower,
            self.receipt.upper,
        ) != expected or self.receipt.draw_index != 0:
            _fail("random consumption entry differs from its fixed domain contract")
        if self.schema_version != RANDOM_CONSUMPTION_ENTRY_SCHEMA_VERSION:
            _fail("random consumption entry schema version changed")

    @property
    def draw_identity(self) -> tuple[str, str, int]:
        """Return the collision-detection identity for this consumed draw."""
        return (
            self.receipt.event_key.value,
            self.receipt.domain,
            self.receipt.draw_index,
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "kind": self.kind.value,
            "receipt": self.receipt.to_mapping(),
            "schema_version": self.schema_version,
            "step_sequence": self.step_sequence,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(
            value,
            frozenset(
                {"agent_id", "kind", "receipt", "schema_version", "step_sequence"}
            ),
            "random consumption entry",
        )
        try:
            kind = RandomConsumptionKind(_text(value["kind"], "kind", maximum=32))
        except ValueError as error:
            raise IntegratedMarketError(
                "random consumption entry kind is unsupported"
            ) from error
        return cls(
            step_sequence=_integer(value["step_sequence"], "step_sequence"),
            agent_id=_identifier(value["agent_id"], "agent_id"),
            kind=kind,
            receipt=_parse_random_draw_receipt(value["receipt"], "receipt"),
            schema_version=_text(value["schema_version"], "schema_version"),
        )


def _random_consumption_source_steps_sha256(
    *,
    expected_step_count: int,
    expected_agent_ids: tuple[str, ...],
    entries: tuple[RandomConsumptionEntry, ...],
) -> str:
    """Re-derive the source-step projection from canonical ledger positions."""
    position = 0
    steps: list[dict[str, object]] = []
    for step_sequence in range(1, expected_step_count + 1):
        decisions: list[dict[str, object]] = []
        for agent_id in expected_agent_ids:
            observation = entries[position]
            action = entries[position + 1]
            position += 2
            decisions.append(
                {
                    "action_choice_draw_sha256": action.receipt.sha256,
                    "agent_id": agent_id,
                    "observation_noise_draw_sha256": observation.receipt.sha256,
                }
            )
        steps.append({"decisions": decisions, "sequence": step_sequence})
    return _commit(steps)


@dataclass(frozen=True, slots=True)
class RandomConsumptionLedger:
    """Canonical and complete single-world internal draw ledger.

    Completeness is relative to the retained integrated step ledger only.  The
    explicit false flag prevents this artifact from being interpreted as proof
    of cross-world coupling, which requires a separate paired-world audit.
    """

    expected_step_count: int
    expected_agent_ids: tuple[str, ...]
    entries: tuple[RandomConsumptionEntry, ...]
    source_steps_sha256: str
    rng_implementation_id: str = RNG_IMPLEMENTATION_ID
    rng_algorithm_spec_commitment: str = RNG_ALGORITHM_SPEC_COMMITMENT
    rng_implementation_commitment: str = RNG_IMPLEMENTATION_COMMITMENT
    single_world_random_consumption_complete: Literal[True] = True
    cross_world_internal_random_coupling_verified: Literal[False] = False
    schema_version: str = RANDOM_CONSUMPTION_LEDGER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        try:
            live_rng_commitment = assert_live_rng_implementation()
        except RuntimeError as error:
            raise IntegratedMarketError(
                "random ledger RNG implementation drift detected"
            ) from error
        if (
            isinstance(self.expected_step_count, bool)
            or not isinstance(self.expected_step_count, int)
            or self.expected_step_count <= 0
            or self.expected_step_count > _MAX_RANDOM_LEDGER_STEPS
        ):
            _fail("random ledger expected_step_count is outside resource limits")
        if (
            not isinstance(self.expected_agent_ids, tuple)
            or not self.expected_agent_ids
            or len(self.expected_agent_ids) > _MAX_RANDOM_LEDGER_AGENTS
            or tuple(sorted(self.expected_agent_ids)) != self.expected_agent_ids
            or len(set(self.expected_agent_ids)) != len(self.expected_agent_ids)
        ):
            _fail("random ledger expected agents must be unique canonical IDs")
        for agent_id in self.expected_agent_ids:
            _identifier(agent_id, "expected_agent_id")
        expected_count = self.expected_step_count * len(self.expected_agent_ids) * 2
        if expected_count > _MAX_RANDOM_LEDGER_ENTRIES:
            _fail("random ledger exceeds the maximum receipt count")
        if (
            not isinstance(self.entries, tuple)
            or len(self.entries) != expected_count
            or not all(
                isinstance(item, RandomConsumptionEntry) for item in self.entries
            )
        ):
            _fail("random ledger has missing or extra consumption entries")
        expected_positions = tuple(
            (step, agent_id, kind)
            for step in range(1, self.expected_step_count + 1)
            for agent_id in self.expected_agent_ids
            for kind in (
                RandomConsumptionKind.OBSERVATION_NOISE,
                RandomConsumptionKind.ACTION_CHOICE,
            )
        )
        actual_positions = tuple(
            (item.step_sequence, item.agent_id, item.kind) for item in self.entries
        )
        if actual_positions != expected_positions:
            _fail("random ledger entries are missing, extra, or non-canonical")
        identities = tuple(item.draw_identity for item in self.entries)
        if len(set(identities)) != len(identities):
            _fail("random ledger contains a duplicate random draw consumption")
        _sha256(self.source_steps_sha256, "source_steps_sha256")
        expected_source_steps_sha256 = _random_consumption_source_steps_sha256(
            expected_step_count=self.expected_step_count,
            expected_agent_ids=self.expected_agent_ids,
            entries=self.entries,
        )
        if self.source_steps_sha256 != expected_source_steps_sha256:
            _fail(
                "random ledger source steps differ from canonical consumption entries"
            )
        if self.rng_implementation_id != RNG_IMPLEMENTATION_ID:
            _fail("random ledger RNG implementation ID changed")
        if self.rng_algorithm_spec_commitment != RNG_ALGORITHM_SPEC_COMMITMENT:
            _fail("random ledger RNG algorithm specification changed")
        if (
            self.rng_implementation_commitment != RNG_IMPLEMENTATION_COMMITMENT
            or self.rng_implementation_commitment != live_rng_commitment
        ):
            _fail("random ledger RNG implementation commitment changed")
        if self.single_world_random_consumption_complete is not True:
            _fail("random ledger must prove complete single-world consumption")
        if self.cross_world_internal_random_coupling_verified is not False:
            _fail("single-world ledger cannot verify cross-world random coupling")
        if self.schema_version != RANDOM_CONSUMPTION_LEDGER_SCHEMA_VERSION:
            _fail("random consumption ledger schema version changed")

    @classmethod
    def from_steps(cls, steps: tuple[IntegratedStepReceipt, ...]) -> Self:
        """Derive the only accepted ledger from retained decision receipts."""
        if (
            not isinstance(steps, tuple)
            or not steps
            or len(steps) > _MAX_RANDOM_LEDGER_STEPS
        ):
            _fail("random ledger requires bounded integrated steps")
        expected_agent_ids = tuple(item.agent_id for item in steps[0].decisions)
        if not expected_agent_ids:
            _fail("random ledger requires at least one decision per step")
        entries: list[RandomConsumptionEntry] = []
        for expected_sequence, step in enumerate(steps, start=1):
            if step.sequence != expected_sequence:
                _fail("random ledger source step sequence is incomplete")
            if tuple(item.agent_id for item in step.decisions) != expected_agent_ids:
                _fail("random ledger source agent set changes across steps")
            for decision in step.decisions:
                entries.extend(
                    (
                        RandomConsumptionEntry(
                            step_sequence=step.sequence,
                            agent_id=decision.agent_id,
                            kind=RandomConsumptionKind.OBSERVATION_NOISE,
                            receipt=decision.observation_noise_draw,
                        ),
                        RandomConsumptionEntry(
                            step_sequence=step.sequence,
                            agent_id=decision.agent_id,
                            kind=RandomConsumptionKind.ACTION_CHOICE,
                            receipt=decision.action_choice_draw,
                        ),
                    )
                )
        canonical_entries = tuple(entries)
        return cls(
            expected_step_count=len(steps),
            expected_agent_ids=expected_agent_ids,
            entries=canonical_entries,
            source_steps_sha256=_random_consumption_source_steps_sha256(
                expected_step_count=len(steps),
                expected_agent_ids=expected_agent_ids,
                entries=canonical_entries,
            ),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "cross_world_internal_random_coupling_verified": (
                self.cross_world_internal_random_coupling_verified
            ),
            "entries": [item.to_mapping() for item in self.entries],
            "expected_agent_ids": list(self.expected_agent_ids),
            "expected_step_count": self.expected_step_count,
            "rng_algorithm_spec_commitment": self.rng_algorithm_spec_commitment,
            "rng_implementation_commitment": self.rng_implementation_commitment,
            "rng_implementation_id": self.rng_implementation_id,
            "schema_version": self.schema_version,
            "single_world_random_consumption_complete": (
                self.single_world_random_consumption_complete
            ),
            "source_steps_sha256": self.source_steps_sha256,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        """Strictly rebuild a bounded single-world random consumption ledger."""
        _strict_fields(
            value,
            frozenset(
                {
                    "cross_world_internal_random_coupling_verified",
                    "entries",
                    "expected_agent_ids",
                    "expected_step_count",
                    "rng_algorithm_spec_commitment",
                    "rng_implementation_commitment",
                    "rng_implementation_id",
                    "schema_version",
                    "single_world_random_consumption_complete",
                    "source_steps_sha256",
                }
            ),
            "random consumption ledger",
        )
        return cls(
            expected_step_count=_integer(
                value["expected_step_count"], "expected_step_count"
            ),
            expected_agent_ids=tuple(
                _identifier(item, "expected_agent_id")
                for item in _sequence(value["expected_agent_ids"], "expected_agent_ids")
            ),
            entries=tuple(
                RandomConsumptionEntry.from_mapping(
                    _mapping(item, "random consumption entry")
                )
                for item in _sequence(value["entries"], "entries")
            ),
            source_steps_sha256=_sha256(
                value["source_steps_sha256"], "source_steps_sha256"
            ),
            rng_implementation_id=_text(
                value["rng_implementation_id"], "rng_implementation_id"
            ),
            rng_algorithm_spec_commitment=_sha256(
                value["rng_algorithm_spec_commitment"],
                "rng_algorithm_spec_commitment",
            ),
            rng_implementation_commitment=_sha256(
                value["rng_implementation_commitment"],
                "rng_implementation_commitment",
            ),
            single_world_random_consumption_complete=cast(
                Literal[True],
                _boolean(
                    value["single_world_random_consumption_complete"],
                    "single_world_random_consumption_complete",
                ),
            ),
            cross_world_internal_random_coupling_verified=cast(
                Literal[False],
                _boolean(
                    value["cross_world_internal_random_coupling_verified"],
                    "cross_world_internal_random_coupling_verified",
                ),
            ),
            schema_version=_text(value["schema_version"], "schema_version"),
        )

    @property
    def sha256(self) -> str:
        return _commit(self.to_mapping())


@dataclass(frozen=True, slots=True)
class IntegratedHorizonReplayBoundary:
    """Frozen command-prefix boundary for one registered synthetic horizon."""

    horizon_id: str
    end_step: int
    command_count: int
    command_event_keys: tuple[str, ...]
    replay_sha256: str
    state_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.horizon_id, "horizon_id")
        _positive_int(self.end_step, "end_step")
        if (
            isinstance(self.command_count, bool)
            or not isinstance(self.command_count, int)
            or self.command_count < 0
        ):
            _fail("horizon command_count must be a non-negative integer")
        if (
            not isinstance(self.command_event_keys, tuple)
            or len(self.command_event_keys) != self.command_count
            or not all(
                isinstance(item, str) and item.startswith("ptk1:")
                for item in self.command_event_keys
            )
            or len(set(self.command_event_keys)) != len(self.command_event_keys)
        ):
            _fail("horizon command event keys must exactly identify the prefix")
        _sha256(self.replay_sha256, "replay_sha256")
        _sha256(self.state_sha256, "state_sha256")

    def to_mapping(self) -> dict[str, object]:
        return {
            "command_count": self.command_count,
            "command_event_keys": list(self.command_event_keys),
            "end_step": self.end_step,
            "horizon_id": self.horizon_id,
            "replay_sha256": self.replay_sha256,
            "state_sha256": self.state_sha256,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(
            value,
            frozenset(
                {
                    "command_count",
                    "command_event_keys",
                    "end_step",
                    "horizon_id",
                    "replay_sha256",
                    "state_sha256",
                }
            ),
            "integrated horizon replay boundary",
        )
        return cls(
            horizon_id=_identifier(value["horizon_id"], "horizon_id"),
            end_step=_integer(value["end_step"], "end_step"),
            command_count=_integer(value["command_count"], "command_count"),
            command_event_keys=tuple(
                _text(item, "command_event_key")
                for item in _sequence(value["command_event_keys"], "command_event_keys")
            ),
            replay_sha256=_sha256(value["replay_sha256"], "replay_sha256"),
            state_sha256=_sha256(value["state_sha256"], "state_sha256"),
        )

    @property
    def sha256(self) -> str:
        return _commit(self.to_mapping())


@dataclass(frozen=True, slots=True)
class IntegratedSyntheticRunArtifact:
    """Replayable synthetic CLOB evidence, not full mechanism or welfare proof."""

    role: WorldRole
    config_sha256: str
    policy_sha256: str
    parameter_sha256: str
    shock_tape_sha256: str
    initial_state_sha256: str
    steps: tuple[IntegratedStepReceipt, ...]
    random_consumption_ledger: RandomConsumptionLedger
    horizon_boundaries: tuple[IntegratedHorizonReplayBoundary, ...]
    market_replay: ReplayResult
    market_session_sha256: str
    market_replay_sha256: str
    market_state_sha256: str
    social_audit_sha256: str
    bridge_audit_sha256: str
    claim_scope: str = INTEGRATED_RUN_ARTIFACT_CLAIM_SCOPE
    integrated_mechanism_verified: Literal[False] = False
    cross_world_internal_random_coupling_verified: Literal[False] = False
    ledger_derived_group_outcomes: Literal[False] = False
    formal_welfare_evidence: Literal[False] = False
    synthetic_only: Literal[True] = True
    schema_version: str = INTEGRATED_RUN_ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.role, WorldRole):
            _fail("run artifact role has the wrong type")
        for field in (
            "config_sha256",
            "policy_sha256",
            "parameter_sha256",
            "shock_tape_sha256",
            "initial_state_sha256",
            "market_session_sha256",
            "market_replay_sha256",
            "market_state_sha256",
            "social_audit_sha256",
            "bridge_audit_sha256",
        ):
            _sha256(getattr(self, field), field)
        if not isinstance(self.steps, tuple) or not self.steps:
            _fail("run artifact requires integrated step receipts")
        previous = _NULL_HASH
        for sequence, step in enumerate(self.steps, start=1):
            if step.sequence != sequence or step.previous_step_sha256 != previous:
                _fail("integrated step chain is incomplete or reordered")
            previous = step.sha256
        if not isinstance(self.random_consumption_ledger, RandomConsumptionLedger):
            _fail("run artifact requires a RandomConsumptionLedger")
        expected_random_ledger = RandomConsumptionLedger.from_steps(self.steps)
        if self.random_consumption_ledger != expected_random_ledger:
            _fail("random consumption ledger differs from the complete step decisions")
        if self.steps[-1].market_replay_sha256 != self.market_replay_sha256:
            _fail("terminal step and run market replay differ")
        if self.steps[-1].market_state_sha256 != self.market_state_sha256:
            _fail("terminal step and run market state differ")
        if type(self.market_replay) is not ReplayResult:
            _fail("run artifact requires an exact ReplayResult")
        replayed = _live_replay(self.market_replay)
        if replayed.canonical_bytes != self.market_replay.canonical_bytes:
            _fail("run artifact market replay differs from live replay")
        if self.market_replay.sha256 != self.market_replay_sha256:
            _fail("embedded market replay differs from its run commitment")
        if self.market_replay.state.sha256 != self.market_state_sha256:
            _fail("embedded market replay terminal state differs from the run")

        scheduled_by_key = {
            item.event_key: item.command
            for item in self.market_replay.scheduled_commands
        }
        if len(scheduled_by_key) != len(self.market_replay.scheduled_commands):
            _fail("integrated market replay contains duplicate command event keys")
        scheduled_by_sha256: dict[str, list[tuple[str, MarketCommand]]] = {}
        for event_key, command in scheduled_by_key.items():
            scheduled_by_sha256.setdefault(
                _commit(command_to_mapping(command)), []
            ).append((event_key, command))

        causal_command_keys: list[str] = []
        step_command_prefixes: dict[int, tuple[str, ...]] = {}
        current_replay = _replay_prefix(self.market_replay, ())
        for step in self.steps:
            agent_time_start = _agent_command_time(step.sequence, 0)
            exogenous_time = _exogenous_command_time(step.sequence)
            for decision in step.decisions:
                if decision.market_state_before_sha256 != current_replay.state.sha256:
                    _fail(
                        "decision information state differs from causal market history"
                    )
                decision_command = scheduled_by_key.get(decision.event_key)
                if decision.outcome is IntegratedDecisionOutcome.APPLIED:
                    if decision_command is None:
                        _fail("applied decision has no retained market command")
                    if (
                        decision_command.trader_id != decision.agent_id
                        or decision_command.submitted_at < agent_time_start
                        or decision_command.submitted_at >= exogenous_time
                        or decision_command.event_key.simulation_step
                        != decision_command.submitted_at
                    ):
                        _fail("applied decision command lies outside its causal step")
                    causal_command_keys.append(decision.event_key)
                    current_replay = _replay_prefix(
                        self.market_replay, tuple(causal_command_keys)
                    )
                elif decision_command is not None:
                    _fail("non-applied decision unexpectedly has a market command")
                if decision.market_state_after_sha256 != current_replay.state.sha256:
                    _fail("decision output state differs from causal market replay")

            exogenous_matches = scheduled_by_sha256.get(
                step.exogenous_command_sha256, []
            )
            if len(exogenous_matches) != 1:
                _fail(
                    "step exogenous command does not uniquely match the market ledger"
                )
            exogenous_key, exogenous_command = exogenous_matches[0]
            if (
                not isinstance(exogenous_command, MarketOrder)
                or exogenous_command.submitted_at != exogenous_time
                or exogenous_command.event_key.simulation_step != exogenous_time
                or exogenous_key in causal_command_keys
            ):
                _fail("step exogenous command lies outside its causal boundary")
            causal_command_keys.append(exogenous_key)
            current_replay = _replay_prefix(
                self.market_replay, tuple(causal_command_keys)
            )
            if (
                current_replay.sha256 != step.market_replay_sha256
                or current_replay.state.sha256 != step.market_state_sha256
            ):
                _fail("integrated step differs from live causal-prefix replay")
            step_command_prefixes[step.sequence] = tuple(causal_command_keys)

        terminal_schedule = tuple(
            item.event_key for item in self.market_replay.scheduled_commands
        )
        if tuple(causal_command_keys) != terminal_schedule:
            _fail("integrated causal command ledger differs from terminal schedule")
        if not isinstance(self.horizon_boundaries, tuple) or not (
            self.horizon_boundaries
        ):
            _fail("run artifact requires registered horizon replay boundaries")
        if not all(
            isinstance(item, IntegratedHorizonReplayBoundary)
            for item in self.horizon_boundaries
        ):
            _fail("horizon boundaries have the wrong type")
        ordered_boundaries = tuple(
            sorted(
                self.horizon_boundaries,
                key=lambda item: (item.end_step, item.horizon_id),
            )
        )
        if ordered_boundaries != self.horizon_boundaries:
            _fail("horizon boundaries must use canonical horizon order")
        if len({item.horizon_id for item in self.horizon_boundaries}) != len(
            self.horizon_boundaries
        ):
            _fail("horizon boundary IDs must be unique")
        previous_command_count = -1
        previous_command_event_keys: tuple[str, ...] = ()
        prefix_cache: dict[tuple[str, ...], ReplayResult] = {}
        for boundary in self.horizon_boundaries:
            if boundary.end_step > len(self.steps):
                _fail("horizon boundary exceeds the integrated step ledger")
            step = self.steps[boundary.end_step - 1]
            expected_prefix = step_command_prefixes[boundary.end_step]
            if (
                boundary.replay_sha256 != step.market_replay_sha256
                or boundary.state_sha256 != step.market_state_sha256
                or boundary.command_count != len(expected_prefix)
                or boundary.command_event_keys != expected_prefix
            ):
                _fail("horizon boundary differs from its exact causal step prefix")
            if boundary.command_count < previous_command_count:
                _fail("horizon command counts must be monotone")
            if boundary.command_event_keys[: len(previous_command_event_keys)] != (
                previous_command_event_keys
            ):
                _fail("later horizon commands must extend the earlier prefix")
            prefix = prefix_cache.get(boundary.command_event_keys)
            if prefix is None:
                prefix = _replay_prefix(self.market_replay, boundary.command_event_keys)
                prefix_cache[boundary.command_event_keys] = prefix
            if (
                prefix.sha256 != boundary.replay_sha256
                or prefix.state.sha256 != boundary.state_sha256
            ):
                _fail("horizon boundary differs from live command-prefix replay")
            previous_command_count = boundary.command_count
            previous_command_event_keys = boundary.command_event_keys
        if self.claim_scope != INTEGRATED_RUN_ARTIFACT_CLAIM_SCOPE:
            _fail("integrated run artifact claim scope changed")
        if self.integrated_mechanism_verified is not False:
            _fail("standalone CLOB replay cannot verify the integrated mechanism")
        if self.cross_world_internal_random_coupling_verified is not False:
            _fail("single-world artifact cannot verify cross-world random coupling")
        if self.ledger_derived_group_outcomes is not False:
            _fail("group outcomes are not yet derived from the retained ledger")
        if self.formal_welfare_evidence is not False:
            _fail("standalone CLOB replay cannot authorize welfare evidence")
        if self.synthetic_only is not True:
            _fail("integrated artifact must remain synthetic-only")
        if self.schema_version != INTEGRATED_RUN_ARTIFACT_SCHEMA_VERSION:
            _fail("integrated artifact schema version changed")

    def to_mapping(self) -> dict[str, object]:
        return {
            "bridge_audit_sha256": self.bridge_audit_sha256,
            "claim_scope": self.claim_scope,
            "config_sha256": self.config_sha256,
            "cross_world_internal_random_coupling_verified": (
                self.cross_world_internal_random_coupling_verified
            ),
            "formal_welfare_evidence": self.formal_welfare_evidence,
            "horizon_boundaries": [
                item.to_mapping() for item in self.horizon_boundaries
            ],
            "initial_state_sha256": self.initial_state_sha256,
            "integrated_mechanism_verified": self.integrated_mechanism_verified,
            "ledger_derived_group_outcomes": self.ledger_derived_group_outcomes,
            "market_replay": self.market_replay.to_mapping(),
            "market_replay_sha256": self.market_replay_sha256,
            "market_session_sha256": self.market_session_sha256,
            "market_state_sha256": self.market_state_sha256,
            "parameter_sha256": self.parameter_sha256,
            "policy_sha256": self.policy_sha256,
            "random_consumption_ledger": self.random_consumption_ledger.to_mapping(),
            "role": self.role.value,
            "schema_version": self.schema_version,
            "shock_tape_sha256": self.shock_tape_sha256,
            "social_audit_sha256": self.social_audit_sha256,
            "steps": [item.to_mapping() for item in self.steps],
            "synthetic_only": self.synthetic_only,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        """Strictly load and live-replay a synthetic integrated run artifact."""
        _strict_fields(
            value,
            frozenset(
                {
                    "bridge_audit_sha256",
                    "claim_scope",
                    "config_sha256",
                    "cross_world_internal_random_coupling_verified",
                    "formal_welfare_evidence",
                    "horizon_boundaries",
                    "initial_state_sha256",
                    "integrated_mechanism_verified",
                    "ledger_derived_group_outcomes",
                    "market_replay",
                    "market_replay_sha256",
                    "market_session_sha256",
                    "market_state_sha256",
                    "parameter_sha256",
                    "policy_sha256",
                    "random_consumption_ledger",
                    "role",
                    "schema_version",
                    "shock_tape_sha256",
                    "social_audit_sha256",
                    "steps",
                    "synthetic_only",
                }
            ),
            "integrated synthetic run artifact",
        )
        try:
            role = WorldRole(_text(value["role"], "role", maximum=32))
        except ValueError as error:
            raise IntegratedMarketError(
                "integrated run artifact role is unsupported"
            ) from error
        return cls(
            role=role,
            config_sha256=_sha256(value["config_sha256"], "config_sha256"),
            policy_sha256=_sha256(value["policy_sha256"], "policy_sha256"),
            parameter_sha256=_sha256(value["parameter_sha256"], "parameter_sha256"),
            shock_tape_sha256=_sha256(value["shock_tape_sha256"], "shock_tape_sha256"),
            initial_state_sha256=_sha256(
                value["initial_state_sha256"], "initial_state_sha256"
            ),
            steps=tuple(
                IntegratedStepReceipt.from_mapping(
                    _mapping(item, "integrated step receipt")
                )
                for item in _sequence(value["steps"], "steps")
            ),
            random_consumption_ledger=RandomConsumptionLedger.from_mapping(
                _mapping(
                    value["random_consumption_ledger"],
                    "random_consumption_ledger",
                )
            ),
            horizon_boundaries=tuple(
                IntegratedHorizonReplayBoundary.from_mapping(
                    _mapping(item, "integrated horizon replay boundary")
                )
                for item in _sequence(value["horizon_boundaries"], "horizon_boundaries")
            ),
            market_replay=_replay_from_serialized_mapping(
                _mapping(value["market_replay"], "market_replay")
            ),
            market_session_sha256=_sha256(
                value["market_session_sha256"], "market_session_sha256"
            ),
            market_replay_sha256=_sha256(
                value["market_replay_sha256"], "market_replay_sha256"
            ),
            market_state_sha256=_sha256(
                value["market_state_sha256"], "market_state_sha256"
            ),
            social_audit_sha256=_sha256(
                value["social_audit_sha256"], "social_audit_sha256"
            ),
            bridge_audit_sha256=_sha256(
                value["bridge_audit_sha256"], "bridge_audit_sha256"
            ),
            claim_scope=_text(value["claim_scope"], "claim_scope"),
            integrated_mechanism_verified=cast(
                Literal[False],
                _boolean(
                    value["integrated_mechanism_verified"],
                    "integrated_mechanism_verified",
                ),
            ),
            cross_world_internal_random_coupling_verified=cast(
                Literal[False],
                _boolean(
                    value["cross_world_internal_random_coupling_verified"],
                    "cross_world_internal_random_coupling_verified",
                ),
            ),
            ledger_derived_group_outcomes=cast(
                Literal[False],
                _boolean(
                    value["ledger_derived_group_outcomes"],
                    "ledger_derived_group_outcomes",
                ),
            ),
            formal_welfare_evidence=cast(
                Literal[False],
                _boolean(value["formal_welfare_evidence"], "formal_welfare_evidence"),
            ),
            synthetic_only=cast(
                Literal[True], _boolean(value["synthetic_only"], "synthetic_only")
            ),
            schema_version=_text(value["schema_version"], "schema_version"),
        )

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_mapping())

    @property
    def sha256(self) -> str:
        return sha256_bytes(self.canonical_bytes)


def load_integrated_synthetic_run_artifact(
    path: Path, *, expected_sha256: str
) -> IntegratedSyntheticRunArtifact:
    """Load canonical JSON under an external digest, then independently replay it."""
    if not isinstance(path, Path):
        _fail("integrated artifact path must be a Path")
    expected = _sha256(expected_sha256, "expected_sha256")
    candidate = Path(os.path.abspath(path.expanduser()))
    current = Path(candidate.anchor)
    try:
        for part in candidate.parts[1:]:
            current /= part
            if current.is_symlink():
                _fail("integrated artifact path contains a symbolic link")
        descriptor = os.open(
            candidate,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
        )
    except OSError as error:
        raise IntegratedMarketError("cannot open integrated run artifact") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > _MAX_INTEGRATED_ARTIFACT_BYTES
        ):
            _fail("integrated run artifact is not a bounded regular file")
        chunks: list[bytes] = []
        total = 0
        while total <= _MAX_INTEGRATED_ARTIFACT_BYTES:
            chunk = os.read(
                descriptor,
                min(1024 * 1024, _MAX_INTEGRATED_ARTIFACT_BYTES + 1 - total),
            )
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        if total > _MAX_INTEGRATED_ARTIFACT_BYTES:
            _fail("integrated run artifact exceeds its byte limit")
        after = os.fstat(descriptor)
        content = b"".join(chunks)
    finally:
        os.close(descriptor)
    if _stat_fingerprint(before) != _stat_fingerprint(after) or (
        len(content) != before.st_size
    ):
        _fail("integrated run artifact changed while it was read")
    try:
        linked = candidate.stat(follow_symlinks=False)
    except OSError as error:
        raise IntegratedMarketError("cannot restat integrated run artifact") from error
    if not stat.S_ISREG(linked.st_mode) or _stat_fingerprint(
        linked
    ) != _stat_fingerprint(after):
        _fail("integrated run artifact path identity changed while it was read")
    if sha256_bytes(content) != expected:
        _fail("integrated run artifact differs from its external commitment")
    try:
        parsed = read_json_bytes_object(content, source=str(candidate))
    except Exception as error:
        raise IntegratedMarketError("cannot parse integrated run artifact") from error
    if canonical_json_bytes(parsed) != content:
        _fail("integrated run artifact must use canonical JSON encoding")
    artifact = IntegratedSyntheticRunArtifact.from_mapping(parsed)
    if artifact.sha256 != expected:
        _fail("integrated run artifact semantic replay differs from its commitment")
    return artifact


@dataclass(frozen=True, slots=True)
class _HorizonState:
    step: int
    command_count: int
    command_event_keys: tuple[str, ...]
    market: MarketSnapshot
    replay_sha256: str
    trades: tuple[TradeLogEntry, ...]
    submitted_agent_commands: tuple[LimitOrder, ...]
    decisions: tuple[IntegratedDecisionReceipt, ...]
    cognitive_snapshots: tuple[
        tuple[str, CognitiveSnapshot | ScenarioCognitiveSnapshot], ...
    ]
    node_beliefs: tuple[NodeBelief, ...]


class IntegratedSyntheticWorldRunner(WorldRunner):
    """One isolated world whose components interact through typed boundaries."""

    def __init__(
        self,
        world: WorldDefinition,
        config: IntegratedSyntheticMarketConfig,
    ) -> None:
        if not isinstance(world, WorldDefinition):
            _fail("world must be a WorldDefinition")
        if not isinstance(config, IntegratedSyntheticMarketConfig):
            _fail("config must be an IntegratedSyntheticMarketConfig")
        self._world = world
        self._config = config
        self.artifact: IntegratedSyntheticRunArtifact | None = None

    def run(self, request: WorldRunRequest) -> WorldOutcome:
        """Execute the closed synthetic trajectory and return paired cells."""
        expected_model_draw_sha256 = _commit(
            {
                "config_sha256": self._config.sha256,
                "model": INTEGRATED_MODEL_ID,
            }
        )
        if request.posterior_draw.model_draw_sha256 != expected_model_draw_sha256:
            _fail("posterior model draw is not bound to this runner configuration")
        if self._config.scenario_generator_spec is not None:
            return self._run_scenario(request)
        if request.world != self._world:
            _fail("runner received a different world definition")
        if request.scenario_id != request.shock_access.scenario_id:
            _fail("world request and shock capability scenarios differ")
        policy = self._policy_flags(request)
        agents, signal_scale = self._posterior_agents(request)
        self._validate_groups(request.groups)
        schedule = request.shock_access.transition_schedule
        if max(item.end_step for item in request.horizons) > len(schedule):
            _fail("a requested horizon exceeds the shock schedule")
        if request.shock_access.initial_transition_state.values != (
            (
                self._config.transition_state_field,
                request.shock_access.initial_transition_state.value(
                    self._config.transition_state_field
                ),
            ),
        ):
            _fail("integrated runner requires exactly one configured transition state")

        social_audit = SocialAuditLog()
        bridge = AgentMarketBridge()
        session = ReplayMarketSession(
            self._config.market_config,
            self._config.opening_state,
        )
        cognitive_layer = AuditedCognitiveLayer(
            social_audit,
            self._config.cutoff_policy_sha256,
        )
        cognitive_states = {item.agent_id: CognitiveState() for item in agents}
        cognitive_snapshots = {
            item.agent_id: CognitiveSnapshot(
                EpisodicMemory(
                    capacity=self._config.memory_capacity,
                    trusted_cutoff=self._config.cutoff,
                    cutoff_policy_sha256=self._config.cutoff_policy_sha256,
                    lineage_sha256=self._config.source_lineage_sha256,
                ),
                SemanticState(
                    trusted_cutoff=self._config.cutoff,
                    cutoff_policy_sha256=self._config.cutoff_policy_sha256,
                    lineage_sha256=self._config.source_lineage_sha256,
                ),
            )
            for item in agents
        }
        plans: dict[str, CognitivePlan] = {}
        elapsed = {
            item.agent_id: item.cognitive_update_interval_steps for item in agents
        }
        node_beliefs: tuple[NodeBelief, ...] = ()
        step_receipts: list[IntegratedStepReceipt] = []
        horizon_states: dict[int, _HorizonState] = {}
        submitted_agent_commands: list[LimitOrder] = []
        all_decisions: list[IntegratedDecisionReceipt] = []
        consumed_shocks: list[ExogenousShock] = []
        transition_state = request.shock_access.initial_transition_state
        previous_mid = self._mid(session.snapshot)
        visible_flow = 0.0
        previous_step_hash = _NULL_HASH

        for sequence, shock_key in enumerate(schedule, start=1):
            transition = request.shock_access.apply_transition(
                shock_key, transition_state
            )
            transition_state = transition.post_state
            shock = transition.shock
            consumed_shocks.append(shock)
            event = self._information_event(shock, sequence)
            diffusion = SocialDiffusionEngine(
                graph=self._config.social_graph,
                config=PropagationConfig(
                    experiment_id=request.scenario_id,
                    shared_seed=self._config.shared_seed,
                    replicate_id=0,
                    policy_sha256=self._config.cutoff_policy_sha256,
                    as_of_time=event.as_of_time,
                    trusted_cutoff=self._config.cutoff,
                    social_enabled=self._config.social_enabled,
                    max_hops=self._config.social_max_hops,
                    attention_capacity=self._config.social_attention_capacity,
                    repost_threshold=self._config.social_repost_threshold,
                ),
                audit_log=social_audit,
            ).run(
                (SeededInformationEvent(event, agents[0].agent_id),),
                initial_beliefs=node_beliefs,
            )
            node_beliefs = diffusion.node_beliefs
            accepted_nodes = {
                item.node_id for item in diffusion.attention if item.accepted
            }
            for agent in agents:
                if self._config.memory_enabled and agent.agent_id in accepted_nodes:
                    cognition_key = self._event_key(
                        request.scenario_id,
                        sequence,
                        f"cognition:{shock_key.value}:{agent.agent_id}",
                    )
                    cognitive_snapshots[agent.agent_id] = cognitive_layer.attend(
                        cognitive_snapshots[agent.agent_id],
                        event,
                        event_key=cognition_key,
                        attention=1.0,
                        strength=event.credibility,
                        target_belief=event.sentiment,
                    )

            decisions: list[IntegratedDecisionReceipt] = []
            step_flow = 0
            for agent_index, agent in enumerate(agents):
                before = session.snapshot
                command_time = _agent_command_time(sequence, agent_index)
                decision_key = self._event_key(
                    request.scenario_id,
                    command_time,
                    f"agent-decision:{shock_key.value}:{agent.agent_id}",
                )
                observation, observation_noise_draw = self._observation(
                    agent=agent,
                    snapshot=before,
                    shock=shock,
                    transition_state=transition_state,
                    diffusion=diffusion.node_beliefs,
                    previous_mid=previous_mid,
                    visible_flow=visible_flow,
                    event_key=decision_key,
                    signal_scale=signal_scale,
                )
                portfolio = self._portfolio(agent.agent_id, before)
                if agent.cognition_due(elapsed_steps=elapsed[agent.agent_id]):
                    if self._config.memory_enabled:
                        cognitive_states[agent.agent_id] = agent.update_cognitive_state(
                            cognitive_states[agent.agent_id],
                            portfolio=portfolio,
                            observation=observation,
                            elapsed_steps=elapsed[agent.agent_id],
                        )
                    else:
                        # The no-memory ablation retains the current public signal
                        # path in the fast policy but carries no signal-derived slow
                        # state, episodic entries, or semantic belief across events.
                        cognitive_states[agent.agent_id] = CognitiveState()
                    elapsed[agent.agent_id] = 0
                    plans[agent.agent_id] = self._plan(
                        agent=agent,
                        state=cognitive_states[agent.agent_id],
                        snapshot=cognitive_snapshots[agent.agent_id],
                        policy=policy,
                        event=event,
                        event_key=decision_key,
                        audit_log=social_audit,
                        sequence=sequence,
                    )
                elif agent.agent_id not in plans:
                    _fail("agent lacks an initial slow cognitive plan")
                elapsed[agent.agent_id] += 1
                trace = agent.decide(
                    state=cognitive_states[agent.agent_id],
                    portfolio=portfolio,
                    observation=observation,
                    event_key=decision_key,
                )
                plan = plans[agent.agent_id]
                if (
                    plan.agent_id != self._expected_plan_agent_id(agent.agent_id)
                    or plan.requested_use is not RequestedUse.SYNTHETIC_TEST
                    or not plan.is_valid(observation.as_of_time)
                ):
                    _fail("slow cognitive plan is stale or bound to another agent")
                plan_action = self._plan_action(trace.final_action.action)
                permitted = plan.permits(plan_action)
                trace_hash = _commit(trace.to_mapping())
                plan_hash = _commit(plan.to_mapping())
                before_hash = before.sha256
                binding_hash = _NULL_HASH
                if not permitted:
                    outcome = IntegratedDecisionOutcome.PLAN_VETO
                else:
                    conversion = bridge.convert(
                        trace=trace,
                        market_snapshot=before,
                        market_config=self._config.market_config,
                    )
                    binding_hash = conversion.binding.sha256
                    if conversion.command is None:
                        outcome = IntegratedDecisionOutcome.HOLD
                    else:
                        session.submit(conversion.command)
                        if not isinstance(conversion.command, LimitOrder):
                            _fail("agent bridge emitted a non-limit command")
                        submitted_agent_commands.append(conversion.command)
                        step_flow += 1 if conversion.command.side is Side.BUY else -1
                        outcome = IntegratedDecisionOutcome.APPLIED
                decision_receipt = IntegratedDecisionReceipt(
                    agent_id=agent.agent_id,
                    event_key=decision_key.value,
                    action=trace.final_action.action.value,
                    outcome=outcome,
                    buy_probability=trace.action_probabilities.buy,
                    hold_probability=trace.action_probabilities.hold,
                    sell_probability=trace.action_probabilities.sell,
                    plan_sha256=plan_hash,
                    trace_sha256=trace_hash,
                    observation_noise_draw=observation_noise_draw,
                    action_choice_draw=trace.action_choice_draw,
                    bridge_binding_sha256=binding_hash,
                    market_state_before_sha256=before_hash,
                    market_state_after_sha256=session.snapshot.sha256,
                    mock_adapter_id=(
                        None
                        if self._config.mock_planning_hook is None
                        else plan.adapter_id
                    ),
                    mock_authorization_sha256=(
                        None
                        if self._config.mock_planning_hook is None
                        else plan.adapter_authorization_sha256
                    ),
                    mock_request_sha256=(
                        None
                        if self._config.mock_planning_hook is None
                        else plan.input_sha256
                    ),
                    mock_rationale_code=(
                        None
                        if self._config.mock_planning_hook is None
                        else plan.rationale_code
                    ),
                )
                decisions.append(decision_receipt)
                all_decisions.append(decision_receipt)

            exogenous = self._exogenous_command(
                request.scenario_id,
                sequence,
                shock,
            )
            session.submit(exogenous)
            previous_mid = self._mid(session.snapshot)
            visible_flow = step_flow / max(1, len(agents))
            social_events = social_audit.events
            social_tail = (
                social_events[-1].event_sha256 if social_events else _NULL_HASH
            )
            cognitive_hash = _commit(
                {
                    agent_id: {
                        "fast_state": cognitive_states[agent_id].to_mapping(),
                        "memory_state": snapshot.to_mapping(),
                        "plan": plans[agent_id].to_mapping(),
                    }
                    for agent_id, snapshot in sorted(cognitive_snapshots.items())
                }
            )
            step = IntegratedStepReceipt(
                sequence=sequence,
                previous_step_sha256=previous_step_hash,
                shock_event_key=shock_key.value,
                shock_transition_signature_sha256=(
                    transition.receipt.engine_signature_sha256
                ),
                information_event_sha256=_commit(event.to_mapping()),
                diffusion_sha256=_commit(diffusion.to_mapping()),
                cognitive_state_sha256=cognitive_hash,
                decisions=tuple(decisions),
                exogenous_command_sha256=_commit(exogenous.to_mapping()),
                market_replay_sha256=session.replay.sha256,
                market_state_sha256=session.snapshot.sha256,
                social_audit_tail_sha256=social_tail,
                bridge_audit_sha256=bridge.sha256,
            )
            step_receipts.append(step)
            previous_step_hash = step.sha256
            horizon_states[sequence] = _HorizonState(
                step=sequence,
                command_count=len(session.commands),
                command_event_keys=tuple(
                    item.event_key.value for item in session.commands
                ),
                market=session.snapshot,
                replay_sha256=session.replay.sha256,
                trades=session.replay.trades,
                submitted_agent_commands=tuple(submitted_agent_commands),
                decisions=tuple(all_decisions),
                cognitive_snapshots=tuple(sorted(cognitive_snapshots.items())),
                node_beliefs=node_beliefs,
            )

        session.assert_replay()
        bridge.assert_integrity()
        social_audit.assert_integrity()
        social_audit_sha = _commit([item.to_mapping() for item in social_audit.events])
        artifact = IntegratedSyntheticRunArtifact(
            role=request.world.role,
            config_sha256=self._config.sha256,
            policy_sha256=request.world.policy.sha256,
            parameter_sha256=request.posterior_draw.parameter_sha256,
            shock_tape_sha256=request.shock_access.sha256,
            initial_state_sha256=request.initial_state.sha256,
            steps=tuple(step_receipts),
            random_consumption_ledger=RandomConsumptionLedger.from_steps(
                tuple(step_receipts)
            ),
            horizon_boundaries=tuple(
                IntegratedHorizonReplayBoundary(
                    horizon_id=horizon.horizon_id,
                    end_step=horizon.end_step,
                    command_count=horizon_states[horizon.end_step].command_count,
                    command_event_keys=(
                        horizon_states[horizon.end_step].command_event_keys
                    ),
                    replay_sha256=horizon_states[horizon.end_step].replay_sha256,
                    state_sha256=horizon_states[horizon.end_step].market.sha256,
                )
                for horizon in request.horizons
            ),
            market_replay=session.replay,
            market_session_sha256=session.sha256,
            market_replay_sha256=session.replay.sha256,
            market_state_sha256=session.snapshot.sha256,
            social_audit_sha256=social_audit_sha,
            bridge_audit_sha256=bridge.sha256,
        )
        self.artifact = artifact
        horizons = tuple(
            HorizonOutcome(
                horizon_id=horizon.horizon_id,
                end_step=horizon.end_step,
                groups=tuple(
                    self._group_outcome(group, horizon_states[horizon.end_step])
                    for group in request.groups
                ),
            )
            for horizon in request.horizons
        )
        final_state = StateArtifact.from_mapping(
            {
                self._config.transition_state_field: transition_state.value(
                    self._config.transition_state_field
                ),
                "bridge_audit_sha256": bridge.sha256,
                "integrated_run_sha256": artifact.sha256,
                "market_replay_sha256": session.replay.sha256,
                "market_session_sha256": session.sha256,
                "market_state_sha256": session.snapshot.sha256,
                "social_audit_sha256": social_audit_sha,
                "step_ledger_tail_sha256": step_receipts[-1].sha256,
            }
        )
        return WorldOutcome(
            scenario_id=request.scenario_id,
            role=request.world.role,
            status=WorldStatus.COMPLETED,
            horizons=horizons,
            exogenous_shocks=tuple(consumed_shocks),
            shock_tape_sha256=request.shock_access.sha256,
            initial_state_sha256=request.initial_state.sha256,
            model_draw_sha256=request.posterior_draw.model_draw_sha256,
            parameter_sha256=request.posterior_draw.parameter_sha256,
            group_commitment_sha256=request.group_commitment_sha256,
            policy_sha256=request.world.policy.sha256,
            final_state=final_state,
            final_state_sha256=final_state.sha256,
            internal_event_count=(
                len(session.replay.audit_events)
                + len(social_audit.events)
                + len(bridge.events)
                + len(step_receipts)
            ),
        )

    def _run_scenario(self, request: WorldRunRequest) -> WorldOutcome:
        """Execute a strictly receipt-bound post-cutoff synthetic trajectory."""
        if request.world != self._world:
            _fail("runner received a different world definition")
        if request.scenario_id != request.shock_access.scenario_id:
            _fail("world request and shock capability scenarios differ")
        generator_spec = self._config.scenario_generator_spec
        if generator_spec is None:
            _fail("post-cutoff runner requires a frozen scenario generator")
        schedule = request.shock_access.transition_schedule
        if not schedule or any(
            item.timestamp <= self._config.cutoff for item in schedule
        ):
            _fail("scenario runner requires an entirely post-cutoff shock schedule")
        if max(item.end_step for item in request.horizons) > len(schedule):
            _fail("a requested horizon exceeds the shock schedule")
        if request.shock_access.initial_transition_state.values != (
            (
                self._config.transition_state_field,
                request.shock_access.initial_transition_state.value(
                    self._config.transition_state_field
                ),
            ),
        ):
            _fail("integrated runner requires exactly one configured transition state")
        policy = self._policy_flags(request)
        agents, signal_scale = self._posterior_agents(request)
        self._validate_groups(request.groups)
        authority = ScenarioAgentAuthority(
            scenario_id=request.scenario_id,
            trusted_cutoff=self._config.cutoff,
            cutoff_policy_sha256=generator_spec.cutoff_policy_sha256,
            source_lineage_sha256=generator_spec.source_lineage_sha256,
            generator_spec_sha256=generator_spec.sha256,
        )

        social_audit = SocialAuditLog()
        bridge = AgentMarketBridge()
        session = ReplayMarketSession(
            self._config.market_config,
            self._config.opening_state,
        )
        cognitive_layer = AuditedScenarioCognitiveLayer(
            social_audit,
            self._config.cutoff_policy_sha256,
        )

        def empty_snapshot() -> ScenarioCognitiveSnapshot:
            return ScenarioCognitiveSnapshot(
                episodic_memory=ScenarioEpisodicMemory(
                    scenario_id=request.scenario_id,
                    capacity=self._config.memory_capacity,
                    trusted_cutoff=self._config.cutoff,
                    cutoff_policy_sha256=generator_spec.cutoff_policy_sha256,
                    source_lineage_sha256=generator_spec.source_lineage_sha256,
                    generator_spec_sha256=generator_spec.sha256,
                ),
                semantic_state=ScenarioSemanticState(
                    scenario_id=request.scenario_id,
                    trusted_cutoff=self._config.cutoff,
                    cutoff_policy_sha256=generator_spec.cutoff_policy_sha256,
                    source_lineage_sha256=generator_spec.source_lineage_sha256,
                    generator_spec_sha256=generator_spec.sha256,
                ),
            )

        cognitive_snapshots = {item.agent_id: empty_snapshot() for item in agents}
        slow_snapshots = {item.agent_id: empty_snapshot() for item in agents}
        elapsed = {
            item.agent_id: item.cognitive_update_interval_steps for item in agents
        }
        node_beliefs: tuple[NodeBelief, ...] = ()
        step_receipts: list[IntegratedStepReceipt] = []
        horizon_states: dict[int, _HorizonState] = {}
        submitted_agent_commands: list[LimitOrder] = []
        all_decisions: list[IntegratedDecisionReceipt] = []
        consumed_shocks: list[ExogenousShock] = []
        transition_state = request.shock_access.initial_transition_state
        previous_mid = self._mid(session.snapshot)
        visible_flow = 0.0
        previous_step_hash = _NULL_HASH

        for sequence, shock_key in enumerate(schedule, start=1):
            transition = request.shock_access.apply_transition(
                shock_key, transition_state
            )
            transition_state = transition.post_state
            shock = transition.shock
            consumed_shocks.append(shock)
            event = self._scenario_information_event(
                request=request,
                shock=shock,
                sequence=sequence,
                generator_spec=generator_spec,
            )
            authority.validate_receipt(event.receipt)
            diffusion = SocialDiffusionEngine(
                graph=self._config.social_graph,
                config=PropagationConfig(
                    experiment_id=request.scenario_id,
                    shared_seed=self._config.shared_seed,
                    replicate_id=0,
                    policy_sha256=self._config.cutoff_policy_sha256,
                    as_of_time=event.as_of_time,
                    trusted_cutoff=self._config.cutoff,
                    scenario_event_receipt=event.receipt,
                    social_enabled=self._config.social_enabled,
                    max_hops=self._config.social_max_hops,
                    attention_capacity=self._config.social_attention_capacity,
                    repost_threshold=self._config.social_repost_threshold,
                ),
                audit_log=social_audit,
            ).run(
                (SeededInformationEvent(event, agents[0].agent_id),),
                initial_beliefs=node_beliefs,
            )
            node_beliefs = diffusion.node_beliefs
            accepted_nodes = {
                item.node_id for item in diffusion.attention if item.accepted
            }
            for agent in agents:
                if self._config.memory_enabled and agent.agent_id in accepted_nodes:
                    cognitive_snapshots[agent.agent_id] = cognitive_layer.attend(
                        cognitive_snapshots[agent.agent_id],
                        event,
                        event_key=self._event_key(
                            request.scenario_id,
                            sequence,
                            f"scenario-cognition:{shock_key.value}:{agent.agent_id}",
                        ),
                        attention=1.0,
                        strength=event.credibility,
                        target_belief=event.sentiment,
                    )

            decisions: list[IntegratedDecisionReceipt] = []
            plans: dict[str, ScenarioCognitivePlan] = {}
            step_flow = 0
            for agent_index, agent in enumerate(agents):
                before = session.snapshot
                command_time = _agent_command_time(sequence, agent_index)
                decision_key = self._event_key(
                    request.scenario_id,
                    command_time,
                    f"scenario-agent-decision:{shock_key.value}:{agent.agent_id}",
                )
                observation, observation_noise_draw = self._scenario_observation(
                    agent=agent,
                    snapshot=before,
                    shock=shock,
                    transition_state=transition_state,
                    diffusion=diffusion.node_beliefs,
                    previous_mid=previous_mid,
                    visible_flow=visible_flow,
                    event_key=decision_key,
                    signal_scale=signal_scale,
                    receipt=event.receipt,
                )
                portfolio = self._portfolio(agent.agent_id, before)
                if agent.cognition_due(elapsed_steps=elapsed[agent.agent_id]):
                    slow_snapshots[agent.agent_id] = (
                        cognitive_snapshots[agent.agent_id]
                        if self._config.memory_enabled
                        else empty_snapshot()
                    )
                    elapsed[agent.agent_id] = 0
                elapsed[agent.agent_id] += 1
                plan = self._scenario_plan(
                    agent=agent,
                    snapshot=slow_snapshots[agent.agent_id],
                    policy=policy,
                    event=event,
                    sequence=sequence,
                )
                plans[agent.agent_id] = plan
                trace = ScenarioFastOrderPolicy().decide(
                    agent_id=agent.agent_id,
                    archetype=agent.archetype,
                    parameters=agent.parameters,
                    mechanisms=agent.mechanisms,
                    cognitive_snapshot=slow_snapshots[agent.agent_id],
                    portfolio=portfolio,
                    observation=observation,
                    authority=authority,
                    event_key=decision_key,
                    config=agent.policy_config,
                    risk_limits=agent.risk_limits,
                )
                if agent.agent_id != plan.agent_id or not plan.is_valid_for(
                    event.receipt
                ):
                    _fail("scenario plan is stale or bound to another agent/event")
                permitted = plan.permits(self._plan_action(trace.final_action.action))
                trace_hash = _commit(trace.to_mapping())
                plan_hash = _commit(plan.to_mapping())
                before_hash = before.sha256
                binding_hash = _NULL_HASH
                if not permitted:
                    outcome = IntegratedDecisionOutcome.PLAN_VETO
                else:
                    conversion = bridge.convert_scenario(
                        trace=trace,
                        market_snapshot=before,
                        market_config=self._config.market_config,
                    )
                    binding_hash = conversion.binding.sha256
                    if conversion.command is None:
                        outcome = IntegratedDecisionOutcome.HOLD
                    else:
                        session.submit(conversion.command)
                        if not isinstance(conversion.command, LimitOrder):
                            _fail("scenario agent bridge emitted a non-limit command")
                        submitted_agent_commands.append(conversion.command)
                        step_flow += 1 if conversion.command.side is Side.BUY else -1
                        outcome = IntegratedDecisionOutcome.APPLIED
                decision_receipt = IntegratedDecisionReceipt(
                    agent_id=agent.agent_id,
                    event_key=decision_key.value,
                    action=trace.final_action.action.value,
                    outcome=outcome,
                    buy_probability=trace.action_probabilities.buy,
                    hold_probability=trace.action_probabilities.hold,
                    sell_probability=trace.action_probabilities.sell,
                    plan_sha256=plan_hash,
                    trace_sha256=trace_hash,
                    observation_noise_draw=observation_noise_draw,
                    action_choice_draw=trace.action_choice_draw,
                    bridge_binding_sha256=binding_hash,
                    market_state_before_sha256=before_hash,
                    market_state_after_sha256=session.snapshot.sha256,
                )
                decisions.append(decision_receipt)
                all_decisions.append(decision_receipt)

            exogenous = self._exogenous_command(
                request.scenario_id,
                sequence,
                shock,
            )
            session.submit(exogenous)
            previous_mid = self._mid(session.snapshot)
            visible_flow = step_flow / max(1, len(agents))
            social_events = social_audit.events
            social_tail = (
                social_events[-1].event_sha256 if social_events else _NULL_HASH
            )
            cognitive_hash = _commit(
                {
                    agent_id: {
                        "current_generated_state": snapshot.to_mapping(),
                        "plan": plans[agent_id].to_mapping(),
                        "slow_generated_state": slow_snapshots[agent_id].to_mapping(),
                    }
                    for agent_id, snapshot in sorted(cognitive_snapshots.items())
                }
            )
            step = IntegratedStepReceipt(
                sequence=sequence,
                previous_step_sha256=previous_step_hash,
                shock_event_key=shock_key.value,
                shock_transition_signature_sha256=(
                    transition.receipt.engine_signature_sha256
                ),
                information_event_sha256=_commit(event.to_mapping()),
                diffusion_sha256=_commit(diffusion.to_mapping()),
                cognitive_state_sha256=cognitive_hash,
                decisions=tuple(decisions),
                exogenous_command_sha256=_commit(exogenous.to_mapping()),
                market_replay_sha256=session.replay.sha256,
                market_state_sha256=session.snapshot.sha256,
                social_audit_tail_sha256=social_tail,
                bridge_audit_sha256=bridge.sha256,
            )
            step_receipts.append(step)
            previous_step_hash = step.sha256
            horizon_states[sequence] = _HorizonState(
                step=sequence,
                command_count=len(session.commands),
                command_event_keys=tuple(
                    item.event_key.value for item in session.commands
                ),
                market=session.snapshot,
                replay_sha256=session.replay.sha256,
                trades=session.replay.trades,
                submitted_agent_commands=tuple(submitted_agent_commands),
                decisions=tuple(all_decisions),
                cognitive_snapshots=tuple(sorted(cognitive_snapshots.items())),
                node_beliefs=node_beliefs,
            )

        session.assert_replay()
        bridge.assert_integrity()
        social_audit.assert_integrity()
        social_audit_sha = _commit([item.to_mapping() for item in social_audit.events])
        artifact = IntegratedSyntheticRunArtifact(
            role=request.world.role,
            config_sha256=self._config.sha256,
            policy_sha256=request.world.policy.sha256,
            parameter_sha256=request.posterior_draw.parameter_sha256,
            shock_tape_sha256=request.shock_access.sha256,
            initial_state_sha256=request.initial_state.sha256,
            steps=tuple(step_receipts),
            random_consumption_ledger=RandomConsumptionLedger.from_steps(
                tuple(step_receipts)
            ),
            horizon_boundaries=tuple(
                IntegratedHorizonReplayBoundary(
                    horizon_id=horizon.horizon_id,
                    end_step=horizon.end_step,
                    command_count=horizon_states[horizon.end_step].command_count,
                    command_event_keys=(
                        horizon_states[horizon.end_step].command_event_keys
                    ),
                    replay_sha256=horizon_states[horizon.end_step].replay_sha256,
                    state_sha256=horizon_states[horizon.end_step].market.sha256,
                )
                for horizon in request.horizons
            ),
            market_replay=session.replay,
            market_session_sha256=session.sha256,
            market_replay_sha256=session.replay.sha256,
            market_state_sha256=session.snapshot.sha256,
            social_audit_sha256=social_audit_sha,
            bridge_audit_sha256=bridge.sha256,
        )
        self.artifact = artifact
        horizons = tuple(
            HorizonOutcome(
                horizon_id=horizon.horizon_id,
                end_step=horizon.end_step,
                groups=tuple(
                    self._group_outcome(group, horizon_states[horizon.end_step])
                    for group in request.groups
                ),
            )
            for horizon in request.horizons
        )
        final_state = StateArtifact.from_mapping(
            {
                self._config.transition_state_field: transition_state.value(
                    self._config.transition_state_field
                ),
                "bridge_audit_sha256": bridge.sha256,
                "integrated_run_sha256": artifact.sha256,
                "market_replay_sha256": session.replay.sha256,
                "market_session_sha256": session.sha256,
                "market_state_sha256": session.snapshot.sha256,
                "scenario_generator_spec_sha256": generator_spec.sha256,
                "social_audit_sha256": social_audit_sha,
                "step_ledger_tail_sha256": step_receipts[-1].sha256,
            }
        )
        return WorldOutcome(
            scenario_id=request.scenario_id,
            role=request.world.role,
            status=WorldStatus.COMPLETED,
            horizons=horizons,
            exogenous_shocks=tuple(consumed_shocks),
            shock_tape_sha256=request.shock_access.sha256,
            initial_state_sha256=request.initial_state.sha256,
            model_draw_sha256=request.posterior_draw.model_draw_sha256,
            parameter_sha256=request.posterior_draw.parameter_sha256,
            group_commitment_sha256=request.group_commitment_sha256,
            policy_sha256=request.world.policy.sha256,
            final_state=final_state,
            final_state_sha256=final_state.sha256,
            internal_event_count=(
                len(session.replay.audit_events)
                + len(social_audit.events)
                + len(bridge.events)
                + len(step_receipts)
            ),
        )

    def _policy_flags(self, request: WorldRunRequest) -> dict[str, bool]:
        values = dict(request.world.policy.config)
        if set(values) != _POLICY_FIELDS:
            _fail("integrated policy config has missing or unknown fields")
        if not all(isinstance(values[field], bool) for field in _POLICY_FIELDS):
            _fail("integrated policy flags must be boolean")
        return {field: bool(values[field]) for field in sorted(_POLICY_FIELDS)}

    def _posterior_agents(
        self, request: WorldRunRequest
    ) -> tuple[tuple[StructuredAgent, ...], float]:
        values = dict(request.posterior_draw.parameters)
        if set(values) != _POSTERIOR_FIELDS:
            _fail("integrated posterior parameters have missing or unknown fields")
        behavior_scale = _finite(values["behavior_scale"], "behavior_scale")
        signal_scale = _finite(values["signal_scale"], "signal_scale")
        if not 0.0 <= behavior_scale <= 1.0:
            _fail("behavior_scale must be in [0, 1]")
        if not 0.0 < signal_scale <= 5.0:
            _fail("signal_scale must be in (0, 5]")
        materialized: list[StructuredAgent] = []
        for agent in self._config.agents:
            raw = agent.parameters.to_mapping()
            scaled = {name: value * behavior_scale for name, value in raw.items()}
            materialized.append(replace(agent, parameters=BehaviorParameters(**scaled)))
        return tuple(materialized), signal_scale

    def _validate_groups(self, groups: tuple[GroupSpec, ...]) -> None:
        for group in groups:
            members = self._config.members_for(group.group_id)
            if GroupSpec.from_members(group.group_id, members) != group:
                _fail("requested group commitment differs from frozen members")

    def _information_event(
        self, shock: ExogenousShock, sequence: int
    ) -> InformationEvent:
        timestamp = shock.event_key.timestamp
        if timestamp > self._config.cutoff:
            _fail("current integrated fixture only accepts pre-cutoff synthetic events")
        magnitude = abs(shock.value(self._config.transition_shock_field))
        sentiment = (
            0.0
            if magnitude == 0.0
            else math.copysign(1.0, shock.value(self._config.transition_shock_field))
        )
        payload = shock.to_mapping()
        return InformationEvent(
            event_id=f"shock-info-{sequence}",
            anonymous_source_id="synthetic-exogenous-tape",
            event_kind=InformationEventKind.NEWS,
            sentiment=sentiment,
            magnitude=min(1.0, magnitude),
            credibility=1.0,
            content_sha256=_commit(payload),
            lineage_sha256=self._config.source_lineage_sha256,
            cutoff_policy_sha256=self._config.cutoff_policy_sha256,
            event_time=timestamp,
            available_time=timestamp,
            as_of_time=timestamp,
            trusted_cutoff=self._config.cutoff,
        )

    def _scenario_information_event(
        self,
        *,
        request: WorldRunRequest,
        shock: ExogenousShock,
        sequence: int,
        generator_spec: ScenarioGeneratorSpec,
    ) -> ScenarioInformationEvent:
        timestamp = shock.event_key.timestamp
        if timestamp <= self._config.cutoff:
            _fail("scenario information must be strictly post-cutoff")
        magnitude = abs(shock.value(self._config.transition_shock_field))
        sentiment = (
            0.0
            if magnitude == 0.0
            else math.copysign(1.0, shock.value(self._config.transition_shock_field))
        )
        event_id = f"scenario-shock-info-{sequence}"
        source_id = "synthetic-scenario-generator"
        content_sha256 = _commit(shock.to_mapping())
        bounded_magnitude = min(1.0, magnitude)
        payload_sha256 = scenario_information_payload_sha256(
            event_id=event_id,
            anonymous_source_id=source_id,
            event_kind=InformationEventKind.NEWS,
            sentiment=sentiment,
            magnitude=bounded_magnitude,
            credibility=1.0,
            content_sha256=content_sha256,
        )
        receipt = request.shock_access.issue_scenario_event_receipt(
            shock.event_key,
            generator_spec=generator_spec,
            payload_sha256=payload_sha256,
        )
        return ScenarioInformationEvent(
            event_id=event_id,
            anonymous_source_id=source_id,
            event_kind=InformationEventKind.NEWS,
            sentiment=sentiment,
            magnitude=bounded_magnitude,
            credibility=1.0,
            content_sha256=content_sha256,
            receipt=receipt,
        )

    def _observation(
        self,
        *,
        agent: StructuredAgent,
        snapshot: MarketSnapshot,
        shock: ExogenousShock,
        transition_state: ShockTransitionState,
        diffusion: tuple[NodeBelief, ...],
        previous_mid: int,
        visible_flow: float,
        event_key: EventKey,
        signal_scale: float,
    ) -> tuple[PublicMarketObservation, RandomDrawReceipt]:
        bid, ask = self._top(snapshot)
        mid = (bid + ask) // 2
        belief = next(
            (item.belief for item in diffusion if item.node_id == agent.agent_id),
            0.0,
        )
        realization = shock.value(self._config.transition_shock_field)
        fundamental = transition_state.value(self._config.transition_state_field)
        noise_integer, noise_receipt = deterministic_int_with_receipt(
            event_key,
            domain=INTEGRATED_OBSERVATION_NOISE_DOMAIN,
            draw_index=0,
            lower=_OBSERVATION_DRAW_LOWER,
            upper=_OBSERVATION_DRAW_UPPER,
        )
        noise = noise_integer / 1000.0
        values = {
            SignalKind.NOISE: noise,
            SignalKind.FUNDAMENTAL: signal_scale
            * (fundamental - mid)
            / max(1, self._config.market_config.tick_size),
            SignalKind.TREND: signal_scale
            * (mid - previous_mid)
            / max(1, self._config.market_config.tick_size),
            SignalKind.EXECUTION_URGENCY: 0.25 * signal_scale,
            SignalKind.ATTENTION: belief,
            SignalKind.VOLATILITY: abs(realization),
            SignalKind.ARBITRAGE: signal_scale
            * (fundamental - mid)
            / max(1, self._config.market_config.tick_size),
        }
        timestamp = shock.event_key.timestamp
        observation = PublicMarketObservation(
            as_of_time=timestamp,
            cutoff=self._config.cutoff,
            mid_price=mid,
            best_bid=bid,
            best_ask=ask,
            signals=tuple(
                PublicSignal(
                    kind=kind,
                    value=values[kind],
                    event_time=timestamp,
                    available_time=timestamp,
                    source="synthetic-integrated-event-tape",
                    signal_id=f"signal-{event_key.simulation_step}-{kind.value}",
                )
                for kind in SignalKind
            ),
            visible_order_flow=VisibleOrderFlow(
                net_flow=visible_flow,
                sample_size=len(self._config.agents),
                event_time=timestamp,
                available_time=timestamp,
                source="synthetic-integrated-public-order-flow",
                aggregation_id=f"flow-{event_key.simulation_step}",
            ),
        )
        return observation, noise_receipt

    def _scenario_observation(
        self,
        *,
        agent: StructuredAgent,
        snapshot: MarketSnapshot,
        shock: ExogenousShock,
        transition_state: ShockTransitionState,
        diffusion: tuple[NodeBelief, ...],
        previous_mid: int,
        visible_flow: float,
        event_key: EventKey,
        signal_scale: float,
        receipt: ScenarioEventReceipt,
    ) -> tuple[ScenarioPublicMarketObservation, RandomDrawReceipt]:
        if not isinstance(receipt, ScenarioEventReceipt):
            _fail("scenario observation requires an engine-issued receipt")
        if receipt.event_key != shock.event_key:
            _fail("scenario observation receipt and shock differ")
        bid, ask = self._top(snapshot)
        mid = (bid + ask) // 2
        belief = next(
            (item.belief for item in diffusion if item.node_id == agent.agent_id),
            0.0,
        )
        realization = shock.value(self._config.transition_shock_field)
        fundamental = transition_state.value(self._config.transition_state_field)
        noise_integer, noise_receipt = deterministic_int_with_receipt(
            event_key,
            domain=INTEGRATED_OBSERVATION_NOISE_DOMAIN,
            draw_index=0,
            lower=_OBSERVATION_DRAW_LOWER,
            upper=_OBSERVATION_DRAW_UPPER,
        )
        noise = noise_integer / 1000.0
        values = {
            SignalKind.NOISE: noise,
            SignalKind.FUNDAMENTAL: signal_scale
            * (fundamental - mid)
            / max(1, self._config.market_config.tick_size),
            SignalKind.TREND: signal_scale
            * (mid - previous_mid)
            / max(1, self._config.market_config.tick_size),
            SignalKind.EXECUTION_URGENCY: 0.25 * signal_scale,
            SignalKind.ATTENTION: belief,
            SignalKind.VOLATILITY: abs(realization),
            SignalKind.ARBITRAGE: signal_scale
            * (fundamental - mid)
            / max(1, self._config.market_config.tick_size),
        }
        observation = ScenarioPublicMarketObservation(
            receipt=receipt,
            mid_price=mid,
            best_bid=bid,
            best_ask=ask,
            signals=tuple(
                ScenarioPublicSignal(
                    kind=kind,
                    value=values[kind],
                    signal_id=(
                        f"scenario-signal-{event_key.simulation_step}-{kind.value}"
                    ),
                )
                for kind in SignalKind
            ),
            visible_order_flow=ScenarioVisibleOrderFlow(
                net_flow=visible_flow,
                sample_size=len(self._config.agents),
                aggregation_id=f"scenario-flow-{event_key.simulation_step}",
            ),
            market_state_sha256=snapshot.sha256,
        )
        return observation, noise_receipt

    def _portfolio(self, agent_id: str, snapshot: MarketSnapshot) -> OwnPortfolioState:
        account = self._account(agent_id, snapshot)
        open_orders = tuple(
            item for item in snapshot.resting_orders if item.trader_id == agent_id
        )
        return OwnPortfolioState(
            agent_id=agent_id,
            cash=account.cash,
            inventory=account.asset,
            reserved_cash=account.reserved_cash,
            reserved_inventory=account.reserved_asset,
            average_cost=self._mid(snapshot) if account.asset else None,
            open_order_count=len(open_orders),
            pending_buy_quantity=0,
        )

    def _plan(
        self,
        *,
        agent: StructuredAgent,
        state: CognitiveState,
        snapshot: CognitiveSnapshot,
        policy: dict[str, bool],
        event: InformationEvent,
        event_key: EventKey,
        audit_log: SocialAuditLog,
        sequence: int,
    ) -> CognitivePlan:
        allowed = [PlanAction.HOLD]
        if policy["allow_buy"]:
            allowed.append(PlanAction.BUY)
        if policy["allow_sell"]:
            allowed.append(PlanAction.SELL)
        hook = self._config.mock_planning_hook
        if hook is not None:
            request = StrictLLMRequest(
                event_key=event_key,
                adapter_authorization_sha256=hook.authorization.sha256,
                requested_use=RequestedUse.SYNTHETIC_TEST,
                agent_commitment=self._expected_plan_agent_id(agent.agent_id),
                topic_id=event.event_id,
                as_of_time=event.as_of_time,
                trusted_cutoff=self._config.cutoff,
                cutoff_policy_sha256=self._config.cutoff_policy_sha256,
                input_lineage_sha256=self._config.source_lineage_sha256,
                prompt_template_sha256=hook.authorization.prompt_template_sha256,
                deidentification_policy_sha256=(hook.deidentification_policy.sha256),
                sanitized_context=hook.sanitized_context,
                sanitized_context_record=hook.sanitized_context_record,
                public_features=(
                    PublicCognitiveFeature(
                        code=PublicFeatureCode.BELIEF,
                        value=event.sentiment,
                        event_time=event.event_time,
                        available_time=event.available_time,
                        lineage_sha256=self._config.source_lineage_sha256,
                    ),
                    PublicCognitiveFeature(
                        code=PublicFeatureCode.CONFIDENCE,
                        value=event.credibility,
                        event_time=event.event_time,
                        available_time=event.available_time,
                        lineage_sha256=self._config.source_lineage_sha256,
                    ),
                ),
                allowed_actions=tuple(allowed),
            )
            return hook.gateway(audit_log).plan(request)
        input_sha = _commit(
            {
                "agent_id": agent.agent_id,
                "cognitive_state": state.to_mapping(),
                "memory": snapshot.to_mapping(),
                "policy": policy,
                "sequence": sequence,
            }
        )
        authorization_sha = _commit(
            {
                "adapter_id": "synthetic-structural-planner-v1",
                "claim_scope": INTEGRATED_MARKET_CLAIM_SCOPE,
                "config_sha256": self._config.sha256,
            }
        )
        return CognitivePlan(
            agent_id=agent.agent_id,
            topic_id=event.event_id,
            as_of_time=event.as_of_time,
            valid_until=self._config.cutoff,
            trusted_cutoff=self._config.cutoff,
            allowed_actions=tuple(allowed),
            confidence=min(1.0, abs(state.belief)),
            review_step=sequence,
            input_sha256=input_sha,
            adapter_id="synthetic-structural-planner-v1",
            adapter_authorization_sha256=authorization_sha,
            requested_use=RequestedUse.SYNTHETIC_TEST,
            rationale_code="STRUCTURAL_SYNTHETIC_RULE",
        )

    def _expected_plan_agent_id(self, agent_id: str) -> str:
        hook = self._config.mock_planning_hook
        if hook is None:
            return agent_id
        return _commit(
            {
                "anonymous_agent_id": agent_id,
                "mock_planning_hook_sha256": hook.sha256,
            }
        )

    def _scenario_plan(
        self,
        *,
        agent: StructuredAgent,
        snapshot: ScenarioCognitiveSnapshot,
        policy: dict[str, bool],
        event: ScenarioInformationEvent,
        sequence: int,
    ) -> ScenarioCognitivePlan:
        allowed = [PlanAction.HOLD]
        if policy["allow_buy"]:
            allowed.append(PlanAction.BUY)
        if policy["allow_sell"]:
            allowed.append(PlanAction.SELL)
        beliefs = snapshot.semantic_state.beliefs
        confidence = (
            0.0
            if not beliefs
            else min(
                1.0,
                abs(math.fsum(item.value for item in beliefs) / len(beliefs)),
            )
        )
        return ScenarioCognitivePlan(
            agent_id=agent.agent_id,
            topic_id=event.event_id,
            receipt=event.receipt,
            allowed_actions=tuple(allowed),
            confidence=confidence,
            review_step=sequence,
            input_sha256=_commit(
                {
                    "agent_id": agent.agent_id,
                    "generated_cognitive_snapshot": snapshot.to_mapping(),
                    "policy": policy,
                    "receipt_sha256": event.receipt.engine_signature_sha256,
                    "sequence": sequence,
                }
            ),
            adapter_id="synthetic-scenario-planner-v1",
            adapter_authorization_sha256=_commit(
                {
                    "adapter_id": "synthetic-scenario-planner-v1",
                    "claim_scope": INTEGRATED_MARKET_CLAIM_SCOPE,
                    "config_sha256": self._config.sha256,
                    "generator_spec_sha256": event.receipt.generator_spec.sha256,
                }
            ),
            requested_use=RequestedUse.SYNTHETIC_TEST,
            rationale_code="STRUCTURAL_SYNTHETIC_RULE",
        )

    def _exogenous_command(
        self,
        scenario_id: str,
        sequence: int,
        shock: ExogenousShock,
    ) -> MarketOrder:
        realization = shock.value(self._config.transition_shock_field)
        side = Side.BUY if realization >= 0.0 else Side.SELL
        command_time = _exogenous_command_time(sequence)
        return MarketOrder(
            event_key=shock.event_key.to_legacy_event_key(
                seed=self._config.shared_seed,
                replicate_id=0,
                simulation_step=command_time,
            ),
            submitted_at=command_time,
            order_id=f"exogenous-{scenario_id}-{sequence}",
            trader_id=(
                self._config.exogenous_buyer_id
                if side is Side.BUY
                else self._config.exogenous_seller_id
            ),
            side=side,
            quantity=self._config.exogenous_order_quantity,
        )

    def _group_outcome(self, group: GroupSpec, state: _HorizonState) -> GroupOutcome:
        members = frozenset(self._config.members_for(group.group_id))
        mid = self._mid(state.market)
        accounts = tuple(
            account for account in state.market.accounts if account.trader_id in members
        )
        cash = sum(item.cash for item in accounts)
        inventory = sum(item.asset for item in accounts)
        net_worth = cash + inventory * mid
        submitted = sum(
            item.trader_id in members for item in state.submitted_agent_commands
        )
        group_decisions = tuple(
            item for item in state.decisions if item.agent_id in members
        )
        mean_buy_probability = math.fsum(
            item.buy_probability for item in group_decisions
        ) / len(group_decisions)
        mean_turnover_probability = math.fsum(
            item.buy_probability + item.sell_probability for item in group_decisions
        ) / len(group_decisions)
        filled = sum(
            item.quantity
            * (int(item.buyer_id in members) + int(item.seller_id in members))
            for item in state.trades
        )
        memory_entries = sum(
            len(snapshot.episodic_memory.entries)
            for agent_id, snapshot in state.cognitive_snapshots
            if agent_id in members
        )
        beliefs = [
            item.belief for item in state.node_beliefs if item.node_id in members
        ]
        bid, ask = self._top(state.market)
        market_volume = sum(item.quantity for item in state.trades)
        metrics = (
            MetricValue("cash", float(cash)),
            MetricValue("filled_quantity", float(filled)),
            MetricValue("inventory", float(inventory)),
            MetricValue("market_trade_volume", float(market_volume)),
            MetricValue("mean_buy_probability", mean_buy_probability),
            MetricValue("mean_turnover_probability", mean_turnover_probability),
            MetricValue("memory_entries", float(memory_entries)),
            MetricValue("net_worth", float(net_worth)),
            MetricValue("quoted_spread", float(ask - bid)),
            MetricValue(
                "social_belief",
                0.0 if not beliefs else math.fsum(beliefs) / len(beliefs),
            ),
            MetricValue("submitted_orders", float(submitted)),
        )
        return GroupOutcome(group_id=group.group_id, metrics=metrics)

    def _event_key(
        self, scenario_id: str, simulation_step: int, purpose: str
    ) -> EventKey:
        return EventKey(
            experiment_id=scenario_id,
            seed=self._config.shared_seed,
            replicate_id=0,
            simulation_step=simulation_step,
            purpose=purpose,
        )

    @staticmethod
    def _plan_action(action: DecisionAction) -> PlanAction:
        if action is DecisionAction.BUY:
            return PlanAction.BUY
        if action is DecisionAction.SELL:
            return PlanAction.SELL
        return PlanAction.HOLD

    @staticmethod
    def _account(agent_id: str, snapshot: MarketSnapshot) -> AccountSnapshot:
        matches = tuple(
            item for item in snapshot.accounts if item.trader_id == agent_id
        )
        if len(matches) != 1:
            _fail("agent account is absent or duplicated")
        return matches[0]

    @staticmethod
    def _top(snapshot: MarketSnapshot) -> tuple[int, int]:
        bids = [item.price for item in snapshot.resting_orders if item.side is Side.BUY]
        asks = [
            item.price for item in snapshot.resting_orders if item.side is Side.SELL
        ]
        if not bids or not asks:
            _fail("integrated runner requires a two-sided book at every decision")
        return max(bids), min(asks)

    @classmethod
    def _mid(cls, snapshot: MarketSnapshot) -> int:
        bid, ask = cls._top(snapshot)
        return (bid + ask) // 2


class IntegratedSyntheticWorldFactory:
    """Create isolated runners and retain their synthetic audit artifacts."""

    def __init__(self, config: IntegratedSyntheticMarketConfig) -> None:
        if not isinstance(config, IntegratedSyntheticMarketConfig):
            _fail("factory config has the wrong type")
        self._config = config
        self.runners: dict[WorldRole, IntegratedSyntheticWorldRunner] = {}

    @property
    def config(self) -> IntegratedSyntheticMarketConfig:
        return self._config

    def create(self, world: WorldDefinition) -> IntegratedSyntheticWorldRunner:
        if world.role in self.runners:
            _fail("factory cannot create a second runner for one world role")
        runner = IntegratedSyntheticWorldRunner(world, self._config)
        self.runners[world.role] = runner
        return runner


__all__ = [
    "INTEGRATED_MARKET_CLAIM_SCOPE",
    "INTEGRATED_CONFIG_SCHEMA_VERSION",
    "INTEGRATED_MARKET_SCHEMA_VERSION",
    "INTEGRATED_RUN_ARTIFACT_CLAIM_SCOPE",
    "INTEGRATED_RUN_ARTIFACT_SCHEMA_VERSION",
    "INTEGRATED_MODEL_ID",
    "RANDOM_CONSUMPTION_ENTRY_SCHEMA_VERSION",
    "RANDOM_CONSUMPTION_LEDGER_SCHEMA_VERSION",
    "IntegratedDecisionOutcome",
    "IntegratedDecisionReceipt",
    "IntegratedHorizonReplayBoundary",
    "IntegratedMarketError",
    "IntegratedStepReceipt",
    "IntegratedSyntheticMarketConfig",
    "IntegratedSyntheticRunArtifact",
    "IntegratedSyntheticWorldFactory",
    "IntegratedSyntheticWorldRunner",
    "RandomConsumptionEntry",
    "RandomConsumptionKind",
    "RandomConsumptionLedger",
    "load_integrated_synthetic_run_artifact",
]
