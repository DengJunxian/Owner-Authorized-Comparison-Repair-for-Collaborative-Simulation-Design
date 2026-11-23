"""Deterministic policy metrics and frozen-threshold validity certificates.

This module evaluates already produced synthetic or frozen result records.  It
does not run a market model, choose thresholds, or tune a policy specification.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from statistics import fmean
from typing import NoReturn

from policytwin.counterfactual.events import EventType, SemanticEventKey
from policytwin.protocol import CohortBasis, FrozenCohort
from policytwin.provenance import canonical_json_bytes, sha256_bytes

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EVENT_KEY = re.compile(r"^(?:pse1|ptk1):[0-9a-f]{64}$")


class PolicyEvaluationError(RuntimeError):
    """Base class for fail-closed policy-evaluation errors."""


class PolicyEvaluationSchemaError(PolicyEvaluationError, ValueError):
    """Raised for incomplete or inconsistent evaluation inputs."""


def _fail(message: str) -> NoReturn:
    raise PolicyEvaluationSchemaError(message)


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        _fail(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{field} must be finite")
    return result


def _positive(value: object, field: str, *, allow_zero: bool = False) -> float:
    result = _finite(value, field)
    if result < 0.0 or (not allow_zero and result == 0.0):
        qualifier = "non-negative" if allow_zero else "positive"
        _fail(f"{field} must be {qualifier}")
    return result


def _probability(value: object, field: str) -> float:
    result = _finite(value, field)
    if not 0.0 <= result <= 1.0:
        _fail(f"{field} must lie in [0, 1]")
    return result


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        _fail(f"{field} must be a stable identifier")
    return value


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail(f"{field} must be a lowercase SHA-256 digest")
    return value


def _mapping_fields(
    mapping: Mapping[str, object], expected: frozenset[str], field: str
) -> None:
    if not all(isinstance(key, str) for key in mapping):
        _fail(f"{field} keys must be strings")
    observed = frozenset(mapping)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        _fail(f"{field} fields differ: missing={missing}, extra={extra}")


def _values(values: object, field: str, *, nonempty: bool = True) -> tuple[float, ...]:
    if not isinstance(values, tuple) or (nonempty and not values):
        _fail(f"{field} must be an immutable{' non-empty' if nonempty else ''} tuple")
    return tuple(_finite(value, f"{field} item") for value in values)


def _aware(value: object, field: str) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        _fail(f"{field} must be a timezone-aware datetime")
    return value.astimezone(UTC)


def _quantile(values: tuple[float, ...], probability: float) -> float:
    if not values:
        _fail("quantile input must not be empty")
    probability = _probability(probability, "quantile probability")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + weight * (ordered[upper] - ordered[lower])


def _maximum_drawdown(values: tuple[float, ...]) -> float:
    if not values or any(value <= 0.0 for value in values):
        _fail("drawdown paths must contain positive values")
    peak = values[0]
    maximum = 0.0
    for value in values:
        peak = max(peak, value)
        maximum = max(maximum, (peak - value) / peak)
    return maximum


class TradeSide(StrEnum):
    """Aggressor side used for signed spread decompositions."""

    BUY = "buy"
    SELL = "sell"

    @property
    def sign(self) -> float:
        """Return +1 for buyer-initiated and -1 for seller-initiated trades."""
        return 1.0 if self is TradeSide.BUY else -1.0


class ScenarioAggregation(StrEnum):
    """Frozen cross-scenario reduction for path-valued market diagnostics."""

    EQUAL_WEIGHTED_PATH_MEAN = "equal_weighted_path_mean"


class ScenarioUniverseRole(StrEnum):
    """Semantic role of one independently sampled market scenario universe."""

    PRICE_RETURN_PATH = "price_return_path"
    RISK_RETURN = "risk_return"
    PORTFOLIO_VALUE_PATH = "portfolio_value_path"


class ObservationUniverseRole(StrEnum):
    """Closed roles for event-addressed market observation collections."""

    QUOTE = "quote_observation"
    TRADE = "trade_observation"
    ORDER_OUTCOME = "order_outcome"
    INFORMATION_RESPONSE = "information_response"


@dataclass(frozen=True, slots=True)
class ScenarioMemberCommitment:
    """Content commitment for one canonically identified scenario member."""

    scenario_id: str
    source_event_key: SemanticEventKey
    payload_sha256: str
    trajectory_sha256: str | None

    def __post_init__(self) -> None:
        _identifier(self.scenario_id, "scenario member scenario_id")
        if not isinstance(self.source_event_key, SemanticEventKey):
            _fail("scenario member source_event_key must be a SemanticEventKey")
        if self.source_event_key.scenario_id != self.scenario_id:
            _fail("scenario member source event key is bound to the wrong scenario")
        _sha256(self.payload_sha256, "scenario member payload_sha256")
        if self.trajectory_sha256 is not None:
            _sha256(
                self.trajectory_sha256,
                "scenario member trajectory_sha256",
            )

    def to_mapping(self) -> dict[str, object]:
        """Return the member commitment in canonical machine-readable form."""
        return {
            "payload_sha256": self.payload_sha256,
            "scenario_id": self.scenario_id,
            "source_event_key": self.source_event_key.to_mapping(),
            "trajectory_sha256": self.trajectory_sha256,
        }


@dataclass(frozen=True, slots=True)
class ScenarioUniverseManifest:
    """Self-verifying commitment to every member of one scenario universe."""

    schema_version: int
    role: ScenarioUniverseRole
    members: tuple[ScenarioMemberCommitment, ...]
    scenario_count: int
    manifest_sha256: str

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 2:
            _fail("scenario universe schema_version must equal 2")
        if not isinstance(self.role, ScenarioUniverseRole):
            _fail("scenario universe role must be a ScenarioUniverseRole")
        if (
            not isinstance(self.members, tuple)
            or not self.members
            or not all(
                isinstance(member, ScenarioMemberCommitment) for member in self.members
            )
        ):
            _fail(
                "scenario universe members must be a non-empty immutable tuple "
                "of ScenarioMemberCommitment values"
            )
        scenario_ids = tuple(member.scenario_id for member in self.members)
        if len(set(scenario_ids)) != len(scenario_ids):
            _fail("scenario universe IDs must be unique")
        source_keys = tuple(member.source_event_key.value for member in self.members)
        if len(set(source_keys)) != len(source_keys):
            _fail("scenario universe source event keys must be unique")
        if scenario_ids != tuple(sorted(scenario_ids)):
            _fail("scenario universe members must use canonical scenario-ID order")
        path_role = self.role in {
            ScenarioUniverseRole.PRICE_RETURN_PATH,
            ScenarioUniverseRole.PORTFOLIO_VALUE_PATH,
        }
        if path_role and any(
            member.trajectory_sha256 is None for member in self.members
        ):
            _fail("path scenario members require trajectory commitments")
        if self.role is ScenarioUniverseRole.RISK_RETURN and any(
            member.trajectory_sha256 is not None for member in self.members
        ):
            _fail("scalar risk-return members must not claim trajectories")
        if (
            isinstance(self.scenario_count, bool)
            or not isinstance(self.scenario_count, int)
            or self.scenario_count != len(self.members)
        ):
            _fail("scenario universe count must equal the number of members")
        digest = _sha256(self.manifest_sha256, "scenario universe manifest_sha256")
        expected = sha256_bytes(canonical_json_bytes(self._committed_mapping()))
        if digest != expected:
            _fail("scenario universe manifest hash mismatch")

    @property
    def scenario_ids(self) -> tuple[str, ...]:
        """Return the canonical stable IDs committed by this universe."""
        return tuple(member.scenario_id for member in self.members)

    @classmethod
    def from_price_paths(
        cls,
        price_paths: tuple[PricePath, ...],
    ) -> ScenarioUniverseManifest:
        """Build a manifest by hashing complete price/time trajectories."""
        if (
            not isinstance(price_paths, tuple)
            or not price_paths
            or not all(isinstance(path, PricePath) for path in price_paths)
        ):
            _fail("price paths must be a non-empty immutable tuple")
        members = tuple(
            ScenarioMemberCommitment(
                scenario_id=path.scenario_id,
                source_event_key=path.source_event_key,
                payload_sha256=sha256_bytes(
                    canonical_json_bytes(
                        {
                            "prices": list(path.prices),
                            "role": ScenarioUniverseRole.PRICE_RETURN_PATH.value,
                            "schema_version": 1,
                            "times": list(path.times),
                        }
                    )
                ),
                trajectory_sha256=_trajectory_sha256(path.times, path.prices),
            )
            for path in price_paths
        )
        return cls._from_members(ScenarioUniverseRole.PRICE_RETURN_PATH, members)

    @classmethod
    def from_risk_returns(
        cls,
        scenario_ids: tuple[str, ...],
        returns: tuple[float, ...],
        source_event_keys: tuple[SemanticEventKey, ...],
        *,
        horizon: float,
        sampling_unit: str,
    ) -> ScenarioUniverseManifest:
        """Build a manifest by hashing every ID-aligned fixed-horizon return."""
        if not isinstance(scenario_ids, tuple) or not scenario_ids:
            _fail("risk-return scenario IDs must be a non-empty immutable tuple")
        values = _values(returns, "risk-return manifest values")
        if (
            not isinstance(source_event_keys, tuple)
            or len(scenario_ids) != len(values)
            or len(scenario_ids) != len(source_event_keys)
        ):
            _fail("risk-return manifest IDs, values, and event keys must align")
        normalized_horizon = _positive(horizon, "risk-return manifest horizon")
        normalized_unit = _identifier(
            sampling_unit,
            "risk-return manifest sampling_unit",
        )
        members = tuple(
            ScenarioMemberCommitment(
                scenario_id=_identifier(scenario_id, "risk-return scenario_id"),
                source_event_key=source_event_key,
                payload_sha256=sha256_bytes(
                    canonical_json_bytes(
                        {
                            "return": value,
                            "risk_return_horizon": normalized_horizon,
                            "risk_return_sampling_unit": normalized_unit,
                            "role": ScenarioUniverseRole.RISK_RETURN.value,
                            "schema_version": 1,
                        }
                    )
                ),
                trajectory_sha256=None,
            )
            for scenario_id, value, source_event_key in zip(
                scenario_ids,
                values,
                source_event_keys,
                strict=True,
            )
        )
        return cls._from_members(ScenarioUniverseRole.RISK_RETURN, members)

    @classmethod
    def from_portfolio_value_paths(
        cls,
        scenario_ids: tuple[str, ...],
        times: tuple[tuple[float, ...], ...],
        values: tuple[tuple[float, ...], ...],
        source_event_keys: tuple[SemanticEventKey, ...],
    ) -> ScenarioUniverseManifest:
        """Build a manifest by hashing complete portfolio-value trajectories."""
        if (
            not isinstance(scenario_ids, tuple)
            or not scenario_ids
            or not isinstance(times, tuple)
            or not isinstance(values, tuple)
            or not isinstance(source_event_keys, tuple)
            or len(scenario_ids) != len(times)
            or len(scenario_ids) != len(values)
            or len(scenario_ids) != len(source_event_keys)
        ):
            _fail(
                "portfolio-value manifest IDs, times, values, and event keys must align"
            )
        normalized_times = tuple(
            _values(path, "portfolio-value manifest times") for path in times
        )
        normalized_values = tuple(
            _values(path, "portfolio-value manifest values") for path in values
        )
        if any(
            len(path_times) != len(path_values)
            for path_times, path_values in zip(
                normalized_times,
                normalized_values,
                strict=True,
            )
        ):
            _fail("portfolio-value manifest trajectories must align")
        members = tuple(
            ScenarioMemberCommitment(
                scenario_id=_identifier(
                    scenario_id,
                    "portfolio-value scenario_id",
                ),
                source_event_key=source_event_key,
                payload_sha256=sha256_bytes(
                    canonical_json_bytes(
                        {
                            "role": ScenarioUniverseRole.PORTFOLIO_VALUE_PATH.value,
                            "schema_version": 1,
                            "times": list(path_times),
                            "values": list(path_values),
                        }
                    )
                ),
                trajectory_sha256=_trajectory_sha256(path_times, path_values),
            )
            for scenario_id, path_times, path_values, source_event_key in zip(
                scenario_ids,
                normalized_times,
                normalized_values,
                source_event_keys,
                strict=True,
            )
        )
        return cls._from_members(ScenarioUniverseRole.PORTFOLIO_VALUE_PATH, members)

    @classmethod
    def _from_members(
        cls,
        role: ScenarioUniverseRole,
        members: tuple[ScenarioMemberCommitment, ...],
    ) -> ScenarioUniverseManifest:
        normalized = tuple(sorted(members, key=lambda member: member.scenario_id))
        if len({member.scenario_id for member in normalized}) != len(normalized):
            _fail("scenario universe IDs must be unique")
        payload = {
            "members": [member.to_mapping() for member in normalized],
            "role": role.value,
            "scenario_count": len(normalized),
            "schema_version": 2,
        }
        return cls(
            schema_version=2,
            role=role,
            members=normalized,
            scenario_count=len(normalized),
            manifest_sha256=sha256_bytes(canonical_json_bytes(payload)),
        )

    def _committed_mapping(self) -> dict[str, object]:
        return {
            "members": [member.to_mapping() for member in self.members],
            "role": self.role.value,
            "scenario_count": self.scenario_count,
            "schema_version": self.schema_version,
        }

    def to_mapping(self) -> dict[str, object]:
        """Return a machine-readable, independently verifiable manifest."""
        return self._committed_mapping() | {
            "manifest_sha256": self.manifest_sha256,
            "scenario_ids": list(self.scenario_ids),
        }


_OBSERVATION_EVENT_TYPES: dict[ObservationUniverseRole, EventType] = {
    ObservationUniverseRole.QUOTE: EventType.EVALUATION_QUOTE,
    ObservationUniverseRole.TRADE: EventType.EVALUATION_TRADE,
    ObservationUniverseRole.ORDER_OUTCOME: EventType.EVALUATION_ORDER_OUTCOME,
    ObservationUniverseRole.INFORMATION_RESPONSE: (
        EventType.EVALUATION_INFORMATION_RESPONSE
    ),
}


@dataclass(frozen=True, slots=True)
class ObservationMemberCommitment:
    """Complete content commitment for one event-addressed observation."""

    scenario_id: str
    record_id: str
    source_event_key: SemanticEventKey
    payload_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.scenario_id, "observation member scenario_id")
        _identifier(self.record_id, "observation member record_id")
        if not isinstance(self.source_event_key, SemanticEventKey):
            _fail("observation member source_event_key must be a SemanticEventKey")
        if self.source_event_key.scenario_id != self.scenario_id:
            _fail("observation member source event key is bound to the wrong scenario")
        if self.source_event_key.entity_id != self.record_id:
            _fail("observation member source event key is bound to the wrong record")
        _sha256(self.payload_sha256, "observation member payload_sha256")

    def to_mapping(self) -> dict[str, object]:
        """Return the exact member commitment."""
        return {
            "payload_sha256": self.payload_sha256,
            "record_id": self.record_id,
            "scenario_id": self.scenario_id,
            "source_event_key": self.source_event_key.to_mapping(),
        }


@dataclass(frozen=True, slots=True)
class ObservationUniverseManifest:
    """Self-verifying registry for every quote, trade, order, or information event."""

    schema_version: int
    role: ObservationUniverseRole
    members: tuple[ObservationMemberCommitment, ...]
    record_count: int
    scenario_count: int
    manifest_sha256: str

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            _fail("observation universe schema_version must equal 1")
        if not isinstance(self.role, ObservationUniverseRole):
            _fail("observation universe role must be an ObservationUniverseRole")
        if (
            not isinstance(self.members, tuple)
            or not self.members
            or not all(
                isinstance(member, ObservationMemberCommitment)
                for member in self.members
            )
        ):
            _fail("observation universe members must be a non-empty immutable tuple")
        expected_event_type = _OBSERVATION_EVENT_TYPES[self.role]
        if any(
            member.source_event_key.event_type is not expected_event_type
            for member in self.members
        ):
            _fail("observation universe member has the wrong semantic event type")
        identities = tuple(
            (member.scenario_id, member.record_id) for member in self.members
        )
        if len(set(identities)) != len(identities):
            _fail("observation universe record identities must be unique")
        source_keys = tuple(member.source_event_key.value for member in self.members)
        if len(set(source_keys)) != len(source_keys):
            _fail("observation universe source event keys must be unique")
        canonical = tuple(sorted(self.members, key=_observation_member_sort_key))
        if self.members != canonical:
            _fail("observation universe members must use canonical event order")
        if (
            isinstance(self.record_count, bool)
            or not isinstance(self.record_count, int)
            or self.record_count != len(self.members)
        ):
            _fail("observation universe record_count must equal the member count")
        scenario_count = len({member.scenario_id for member in self.members})
        if (
            isinstance(self.scenario_count, bool)
            or not isinstance(self.scenario_count, int)
            or self.scenario_count != scenario_count
        ):
            _fail("observation universe scenario_count is inconsistent")
        digest = _sha256(self.manifest_sha256, "observation universe manifest_sha256")
        expected = sha256_bytes(canonical_json_bytes(self._committed_mapping()))
        if digest != expected:
            _fail("observation universe manifest hash mismatch")

    @property
    def scenario_ids(self) -> tuple[str, ...]:
        """Return every represented scenario in canonical order."""
        return tuple(sorted({member.scenario_id for member in self.members}))

    @classmethod
    def from_quotes(
        cls, observations: tuple[QuoteObservation, ...]
    ) -> ObservationUniverseManifest:
        """Commit to all quote fields and semantic identities."""
        records = tuple(
            (
                item.scenario_id,
                item.source_event_key.entity_id,
                item.source_event_key,
                {
                    "ask": item.ask,
                    "ask_depth": item.ask_depth,
                    "bid": item.bid,
                    "bid_depth": item.bid_depth,
                    "latent_fundamental": item.latent_fundamental,
                    "time": item.time,
                },
            )
            for item in observations
        )
        return cls._from_records(ObservationUniverseRole.QUOTE, records)

    @classmethod
    def from_trades(
        cls, observations: tuple[TradeObservation, ...]
    ) -> ObservationUniverseManifest:
        """Commit to all trade fields and semantic identities."""
        records = tuple(
            (
                item.scenario_id,
                item.source_event_key.entity_id,
                item.source_event_key,
                {
                    "midquote_after_horizon": item.midquote_after_horizon,
                    "midquote_at_execution": item.midquote_at_execution,
                    "price": item.price,
                    "quantity": item.quantity,
                    "realized_spread_horizon": item.realized_spread_horizon,
                    "side": item.side.value,
                },
            )
            for item in observations
        )
        return cls._from_records(ObservationUniverseRole.TRADE, records)

    @classmethod
    def from_orders(
        cls, observations: tuple[OrderOutcome, ...]
    ) -> ObservationUniverseManifest:
        """Commit to all submitted-order outcome fields and identities."""
        records = tuple(
            (
                item.scenario_id,
                item.order_id,
                item.source_event_key,
                {
                    "filled_quantity": item.filled_quantity,
                    "filled_time": item.filled_time,
                    "submitted_quantity": item.submitted_quantity,
                    "submitted_time": item.submitted_time,
                },
            )
            for item in observations
        )
        return cls._from_records(ObservationUniverseRole.ORDER_OUTCOME, records)

    @classmethod
    def from_information_responses(
        cls, observations: tuple[InformationResponse, ...]
    ) -> ObservationUniverseManifest:
        """Commit to every retained information-response trajectory."""
        records = tuple(
            (
                item.scenario_id,
                item.event_id,
                item.source_event_key,
                {
                    "absolute_deviations": list(item.absolute_deviations),
                    "event_time": item.event_time,
                    "times": list(item.times),
                },
            )
            for item in observations
        )
        return cls._from_records(ObservationUniverseRole.INFORMATION_RESPONSE, records)

    @classmethod
    def _from_records(
        cls,
        role: ObservationUniverseRole,
        records: tuple[tuple[str, str, SemanticEventKey, Mapping[str, object]], ...],
    ) -> ObservationUniverseManifest:
        if not isinstance(records, tuple) or not records:
            _fail("observation records must be a non-empty immutable tuple")
        members = tuple(
            ObservationMemberCommitment(
                scenario_id=scenario_id,
                record_id=record_id,
                source_event_key=source_event_key,
                payload_sha256=sha256_bytes(
                    canonical_json_bytes(
                        {
                            "payload": dict(payload),
                            "role": role.value,
                            "schema_version": 1,
                        }
                    )
                ),
            )
            for scenario_id, record_id, source_event_key, payload in records
        )
        normalized = tuple(sorted(members, key=_observation_member_sort_key))
        payload = {
            "members": [member.to_mapping() for member in normalized],
            "record_count": len(normalized),
            "role": role.value,
            "scenario_count": len({member.scenario_id for member in normalized}),
            "schema_version": 1,
        }
        return cls(
            schema_version=1,
            role=role,
            members=normalized,
            record_count=len(normalized),
            scenario_count=len({member.scenario_id for member in normalized}),
            manifest_sha256=sha256_bytes(canonical_json_bytes(payload)),
        )

    def _committed_mapping(self) -> dict[str, object]:
        return {
            "members": [member.to_mapping() for member in self.members],
            "record_count": self.record_count,
            "role": self.role.value,
            "scenario_count": self.scenario_count,
            "schema_version": self.schema_version,
        }

    def to_mapping(self) -> dict[str, object]:
        """Return a complete, independently hash-verifiable registry."""
        return self._committed_mapping() | {
            "manifest_sha256": self.manifest_sha256,
            "scenario_ids": list(self.scenario_ids),
        }


def _observation_member_sort_key(
    member: ObservationMemberCommitment,
) -> tuple[str, str, str, str]:
    return (
        member.scenario_id,
        member.source_event_key.timestamp.isoformat(),
        member.record_id,
        member.source_event_key.value,
    )


def _validate_observation_event_key(
    value: object,
    scenario_id: str,
    event_type: EventType,
    field: str,
    *,
    expected_entity_id: str | None = None,
) -> SemanticEventKey:
    if not isinstance(value, SemanticEventKey):
        _fail(f"{field} source_event_key must be a SemanticEventKey")
    if value.scenario_id != scenario_id:
        _fail(f"{field} source event key is bound to the wrong scenario")
    if value.event_type is not event_type:
        _fail(f"{field} source event key has the wrong semantic event type")
    if expected_entity_id is not None and value.entity_id != expected_entity_id:
        _fail(f"{field} source event key is bound to the wrong record")
    return value


def _trajectory_sha256(
    times: tuple[float, ...],
    values: tuple[float, ...],
) -> str:
    """Commit to one complete aligned time/value trajectory."""
    return sha256_bytes(
        canonical_json_bytes(
            {
                "schema_version": 1,
                "times": list(times),
                "values": list(values),
            }
        )
    )


@dataclass(frozen=True, slots=True)
class QuoteObservation:
    """Best quote, displayed depth, and optional latent fundamental value."""

    scenario_id: str
    time: float
    bid: float
    ask: float
    bid_depth: float
    ask_depth: float
    latent_fundamental: float | None
    source_event_key: SemanticEventKey

    def __post_init__(self) -> None:
        _identifier(self.scenario_id, "quote scenario_id")
        _validate_observation_event_key(
            self.source_event_key,
            self.scenario_id,
            EventType.EVALUATION_QUOTE,
            "quote",
        )
        object.__setattr__(self, "time", _finite(self.time, "quote time"))
        for field in ("bid", "ask"):
            object.__setattr__(self, field, _positive(getattr(self, field), field))
        if self.ask < self.bid:
            _fail("quote ask must not be below bid")
        for field in ("bid_depth", "ask_depth"):
            object.__setattr__(
                self, field, _positive(getattr(self, field), field, allow_zero=True)
            )
        if self.latent_fundamental is not None:
            object.__setattr__(
                self,
                "latent_fundamental",
                _positive(self.latent_fundamental, "latent_fundamental"),
            )

    @property
    def midpoint(self) -> float:
        """Return the best-quote midpoint."""
        return (self.bid + self.ask) / 2.0


@dataclass(frozen=True, slots=True)
class TradeObservation:
    """Execution record with contemporaneous and fixed-horizon midquotes."""

    scenario_id: str
    side: TradeSide
    price: float
    quantity: float
    midquote_at_execution: float
    midquote_after_horizon: float
    realized_spread_horizon: float
    source_event_key: SemanticEventKey

    def __post_init__(self) -> None:
        _identifier(self.scenario_id, "trade scenario_id")
        _validate_observation_event_key(
            self.source_event_key,
            self.scenario_id,
            EventType.EVALUATION_TRADE,
            "trade",
        )
        if not isinstance(self.side, TradeSide):
            _fail("trade side must be a TradeSide")
        for field in (
            "price",
            "quantity",
            "midquote_at_execution",
            "midquote_after_horizon",
        ):
            object.__setattr__(
                self, field, _positive(getattr(self, field), f"trade {field}")
            )
        object.__setattr__(
            self,
            "realized_spread_horizon",
            _positive(self.realized_spread_horizon, "realized_spread_horizon"),
        )


@dataclass(frozen=True, slots=True)
class OrderOutcome:
    """Submitted quantity and any eventual fill used for execution metrics."""

    order_id: str
    submitted_time: float
    submitted_quantity: float
    filled_quantity: float
    filled_time: float | None
    scenario_id: str
    source_event_key: SemanticEventKey

    def __post_init__(self) -> None:
        _identifier(self.order_id, "order_id")
        _identifier(self.scenario_id, "order scenario_id")
        _validate_observation_event_key(
            self.source_event_key,
            self.scenario_id,
            EventType.EVALUATION_ORDER_OUTCOME,
            "order",
            expected_entity_id=self.order_id,
        )
        object.__setattr__(
            self, "submitted_time", _finite(self.submitted_time, "submitted_time")
        )
        object.__setattr__(
            self,
            "submitted_quantity",
            _positive(self.submitted_quantity, "submitted_quantity"),
        )
        object.__setattr__(
            self,
            "filled_quantity",
            _positive(self.filled_quantity, "filled_quantity", allow_zero=True),
        )
        if self.filled_quantity > self.submitted_quantity:
            _fail("filled quantity must not exceed submitted quantity")
        if (self.filled_quantity > 0.0) != (self.filled_time is not None):
            _fail("positive fills and filled_time must be present together")
        if self.filled_time is not None:
            object.__setattr__(
                self, "filled_time", _finite(self.filled_time, "filled_time")
            )
            if self.filled_time < self.submitted_time:
                _fail("filled_time must not precede submission")


@dataclass(frozen=True, slots=True)
class PricePath:
    """One scenario path for crash, volatility, and drawdown calculations."""

    scenario_id: str
    times: tuple[float, ...]
    prices: tuple[float, ...]
    source_event_key: SemanticEventKey

    def __post_init__(self) -> None:
        _identifier(self.scenario_id, "price path scenario_id")
        if not isinstance(self.source_event_key, SemanticEventKey):
            _fail("price path source_event_key must be a SemanticEventKey")
        if self.source_event_key.scenario_id != self.scenario_id:
            _fail("price path source event key is bound to the wrong scenario")
        times = _values(self.times, "price path times")
        prices = _values(self.prices, "price path prices")
        if len(times) != len(prices) or len(times) < 2:
            _fail("price path requires aligned times and at least two prices")
        if any(
            second <= first for first, second in zip(times, times[1:], strict=False)
        ):
            _fail("price path times must be strictly increasing")
        if any(price <= 0.0 for price in prices):
            _fail("price path prices must be positive")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "prices", prices)


@dataclass(frozen=True, slots=True)
class InformationResponse:
    """Absolute post-information pricing errors at fixed observation times."""

    event_id: str
    event_time: float
    times: tuple[float, ...]
    absolute_deviations: tuple[float, ...]
    scenario_id: str
    source_event_key: SemanticEventKey

    def __post_init__(self) -> None:
        _identifier(self.event_id, "information event_id")
        _identifier(self.scenario_id, "information scenario_id")
        _validate_observation_event_key(
            self.source_event_key,
            self.scenario_id,
            EventType.EVALUATION_INFORMATION_RESPONSE,
            "information response",
            expected_entity_id=self.event_id,
        )
        object.__setattr__(self, "event_time", _finite(self.event_time, "event_time"))
        times = _values(self.times, "information times")
        deviations = _values(self.absolute_deviations, "information deviations")
        if len(times) != len(deviations):
            _fail("information response times and deviations must align")
        if times[0] != self.event_time or any(
            second <= first for first, second in zip(times, times[1:], strict=False)
        ):
            _fail(
                "information response times must start exactly at event_time and "
                "then strictly increase"
            )
        if deviations[0] <= 0.0 or any(value < 0.0 for value in deviations):
            _fail("information deviations must be non-negative with positive start")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "absolute_deviations", deviations)


@dataclass(frozen=True, slots=True)
class MarketMetricConfig:
    """Frozen definitions and thresholds for market metrics."""

    risk_alpha: float
    jump_log_return_threshold: float
    flash_crash_drawdown_threshold: float
    flash_crash_window: float
    realized_spread_horizon: float
    time_unit: str
    price_unit: str
    return_sampling_interval: float
    volatility_annualization_factor: float
    risk_return_horizon: float
    risk_return_sampling_unit: str
    portfolio_value_sampling_interval: float
    maximum_drawdown_horizon: float
    scenario_aggregation: ScenarioAggregation

    def __post_init__(self) -> None:
        alpha = _probability(self.risk_alpha, "risk_alpha")
        if alpha <= 0.0 or alpha >= 1.0:
            _fail("risk_alpha must lie strictly between zero and one")
        object.__setattr__(self, "risk_alpha", alpha)
        object.__setattr__(
            self,
            "jump_log_return_threshold",
            _positive(self.jump_log_return_threshold, "jump_log_return_threshold"),
        )
        crash = _probability(
            self.flash_crash_drawdown_threshold,
            "flash_crash_drawdown_threshold",
        )
        if crash <= 0.0:
            _fail("flash_crash_drawdown_threshold must be positive")
        object.__setattr__(self, "flash_crash_drawdown_threshold", crash)
        object.__setattr__(
            self,
            "flash_crash_window",
            _positive(self.flash_crash_window, "flash_crash_window"),
        )
        object.__setattr__(
            self,
            "realized_spread_horizon",
            _positive(self.realized_spread_horizon, "realized_spread_horizon"),
        )
        _identifier(self.time_unit, "time_unit")
        _identifier(self.price_unit, "price_unit")
        object.__setattr__(
            self,
            "return_sampling_interval",
            _positive(self.return_sampling_interval, "return_sampling_interval"),
        )
        object.__setattr__(
            self,
            "volatility_annualization_factor",
            _positive(
                self.volatility_annualization_factor,
                "volatility_annualization_factor",
            ),
        )
        object.__setattr__(
            self,
            "risk_return_horizon",
            _positive(self.risk_return_horizon, "risk_return_horizon"),
        )
        _identifier(self.risk_return_sampling_unit, "risk_return_sampling_unit")
        object.__setattr__(
            self,
            "portfolio_value_sampling_interval",
            _positive(
                self.portfolio_value_sampling_interval,
                "portfolio_value_sampling_interval",
            ),
        )
        object.__setattr__(
            self,
            "maximum_drawdown_horizon",
            _positive(self.maximum_drawdown_horizon, "maximum_drawdown_horizon"),
        )
        if (
            self.scenario_aggregation
            is not ScenarioAggregation.EQUAL_WEIGHTED_PATH_MEAN
        ):
            _fail("unsupported scenario_aggregation")

    def to_mapping(self) -> dict[str, object]:
        return {
            "flash_crash_drawdown_threshold": self.flash_crash_drawdown_threshold,
            "flash_crash_window": self.flash_crash_window,
            "jump_log_return_threshold": self.jump_log_return_threshold,
            "price_unit": self.price_unit,
            "portfolio_value_sampling_interval": (
                self.portfolio_value_sampling_interval
            ),
            "realized_spread_horizon": self.realized_spread_horizon,
            "return_sampling_interval": self.return_sampling_interval,
            "risk_return_horizon": self.risk_return_horizon,
            "risk_return_sampling_unit": self.risk_return_sampling_unit,
            "risk_alpha": self.risk_alpha,
            "scenario_aggregation": self.scenario_aggregation.value,
            "time_unit": self.time_unit,
            "maximum_drawdown_horizon": self.maximum_drawdown_horizon,
            "volatility_annualization_factor": (self.volatility_annualization_factor),
        }

    @property
    def sha256(self) -> str:
        """Commit to every definition, unit, horizon, and aggregation choice."""
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))


@dataclass(frozen=True, slots=True)
class MarketEvaluationInput:
    """Frozen market observations used only by metric calculations."""

    quotes: tuple[QuoteObservation, ...]
    quote_observation_manifest: ObservationUniverseManifest
    trades: tuple[TradeObservation, ...]
    trade_observation_manifest: ObservationUniverseManifest
    orders: tuple[OrderOutcome, ...]
    order_outcome_manifest: ObservationUniverseManifest
    price_paths: tuple[PricePath, ...]
    price_path_scenario_manifest: ScenarioUniverseManifest
    portfolio_returns: tuple[float, ...]
    portfolio_return_scenario_ids: tuple[str, ...]
    portfolio_return_source_event_keys: tuple[SemanticEventKey, ...]
    risk_return_scenario_manifest: ScenarioUniverseManifest
    risk_return_horizon: float
    risk_return_sampling_unit: str
    portfolio_value_paths: tuple[tuple[float, ...], ...]
    portfolio_value_times: tuple[tuple[float, ...], ...]
    portfolio_value_scenario_ids: tuple[str, ...]
    portfolio_value_source_event_keys: tuple[SemanticEventKey, ...]
    portfolio_value_scenario_manifest: ScenarioUniverseManifest
    information_responses: tuple[InformationResponse, ...]
    information_response_manifest: ObservationUniverseManifest

    def __post_init__(self) -> None:
        typed_fields: tuple[tuple[str, type[object]], ...] = (
            ("quotes", QuoteObservation),
            ("trades", TradeObservation),
            ("orders", OrderOutcome),
            ("price_paths", PricePath),
            ("information_responses", InformationResponse),
        )
        for field, expected_type in typed_fields:
            value = getattr(self, field)
            if (
                not isinstance(value, tuple)
                or not value
                or not all(isinstance(item, expected_type) for item in value)
            ):
                _fail(f"{field} must be a non-empty immutable tuple")
        observation_inputs = (
            (
                self.quotes,
                self.quote_observation_manifest,
                ObservationUniverseRole.QUOTE,
                ObservationUniverseManifest.from_quotes(self.quotes),
                "quote",
            ),
            (
                self.trades,
                self.trade_observation_manifest,
                ObservationUniverseRole.TRADE,
                ObservationUniverseManifest.from_trades(self.trades),
                "trade",
            ),
            (
                self.orders,
                self.order_outcome_manifest,
                ObservationUniverseRole.ORDER_OUTCOME,
                ObservationUniverseManifest.from_orders(self.orders),
                "order outcome",
            ),
            (
                self.information_responses,
                self.information_response_manifest,
                ObservationUniverseRole.INFORMATION_RESPONSE,
                ObservationUniverseManifest.from_information_responses(
                    self.information_responses
                ),
                "information response",
            ),
        )
        for records, manifest, role, expected_manifest, field in observation_inputs:
            self._require_canonical_observation_order(records, field)
            self._validate_observation_manifest(
                manifest,
                role,
                expected_manifest,
                field,
            )
        observation_scenarios = {
            manifest.scenario_ids
            for _, manifest, _, _, _ in observation_inputs
            if isinstance(manifest, ObservationUniverseManifest)
        }
        if len(observation_scenarios) != 1:
            _fail(
                "quote, trade, order, and information observation universes "
                "must contain the same scenario IDs"
            )
        price_path_ids = tuple(path.scenario_id for path in self.price_paths)
        self._require_canonical_input_order(price_path_ids, "price path")
        self._validate_scenario_manifest(
            self.price_path_scenario_manifest,
            ScenarioUniverseRole.PRICE_RETURN_PATH,
            ScenarioUniverseManifest.from_price_paths(self.price_paths),
            "price path",
        )
        object.__setattr__(
            self,
            "portfolio_returns",
            _values(self.portfolio_returns, "portfolio_returns"),
        )
        if not isinstance(self.portfolio_return_scenario_ids, tuple) or len(
            self.portfolio_return_scenario_ids
        ) != len(self.portfolio_returns):
            _fail("portfolio return scenario IDs must align with returns")
        return_ids = tuple(
            _identifier(item, "portfolio return scenario_id")
            for item in self.portfolio_return_scenario_ids
        )
        if len(set(return_ids)) != len(return_ids):
            _fail("portfolio return scenario IDs must be unique")
        object.__setattr__(self, "portfolio_return_scenario_ids", return_ids)
        return_event_keys = self._validate_source_event_keys(
            self.portfolio_return_source_event_keys,
            return_ids,
            "risk return",
        )
        object.__setattr__(
            self,
            "portfolio_return_source_event_keys",
            return_event_keys,
        )
        self._require_canonical_input_order(return_ids, "risk return")
        object.__setattr__(
            self,
            "risk_return_horizon",
            _positive(self.risk_return_horizon, "risk_return_horizon"),
        )
        _identifier(self.risk_return_sampling_unit, "risk_return_sampling_unit")
        self._validate_scenario_manifest(
            self.risk_return_scenario_manifest,
            ScenarioUniverseRole.RISK_RETURN,
            ScenarioUniverseManifest.from_risk_returns(
                return_ids,
                self.portfolio_returns,
                return_event_keys,
                horizon=self.risk_return_horizon,
                sampling_unit=self.risk_return_sampling_unit,
            ),
            "risk return",
        )
        if (
            not isinstance(self.portfolio_value_paths, tuple)
            or not self.portfolio_value_paths
        ):
            _fail("portfolio_value_paths must be a non-empty immutable tuple")
        normalized_paths = tuple(
            _values(path, "portfolio value path") for path in self.portfolio_value_paths
        )
        if any(value <= 0.0 for path in normalized_paths for value in path):
            _fail("portfolio value paths must be positive")
        object.__setattr__(self, "portfolio_value_paths", normalized_paths)
        if not isinstance(self.portfolio_value_times, tuple) or len(
            self.portfolio_value_times
        ) != len(normalized_paths):
            _fail("portfolio value times must align with value paths")
        normalized_times = tuple(
            _values(path, "portfolio value times")
            for path in self.portfolio_value_times
        )
        if any(
            len(times) != len(values)
            or any(
                current <= previous
                for previous, current in zip(times, times[1:], strict=False)
            )
            for times, values in zip(normalized_times, normalized_paths, strict=True)
        ):
            _fail("portfolio value times must align and be strictly increasing")
        object.__setattr__(self, "portfolio_value_times", normalized_times)
        if not isinstance(self.portfolio_value_scenario_ids, tuple) or len(
            self.portfolio_value_scenario_ids
        ) != len(normalized_paths):
            _fail("portfolio value scenario IDs must align with paths")
        value_ids = tuple(
            _identifier(item, "portfolio value scenario_id")
            for item in self.portfolio_value_scenario_ids
        )
        if len(set(value_ids)) != len(value_ids):
            _fail("portfolio value scenario IDs must be unique")
        object.__setattr__(self, "portfolio_value_scenario_ids", value_ids)
        value_event_keys = self._validate_source_event_keys(
            self.portfolio_value_source_event_keys,
            value_ids,
            "portfolio value",
        )
        object.__setattr__(
            self,
            "portfolio_value_source_event_keys",
            value_event_keys,
        )
        self._require_canonical_input_order(value_ids, "portfolio value")
        self._validate_scenario_manifest(
            self.portfolio_value_scenario_manifest,
            ScenarioUniverseRole.PORTFOLIO_VALUE_PATH,
            ScenarioUniverseManifest.from_portfolio_value_paths(
                value_ids,
                normalized_times,
                normalized_paths,
                value_event_keys,
            ),
            "portfolio value",
        )
        quote_scenarios = self.quote_observation_manifest.scenario_ids
        if any(
            not any(
                quote.scenario_id == scenario_id
                and quote.latent_fundamental is not None
                for quote in self.quotes
            )
            for scenario_id in quote_scenarios
        ):
            _fail("every quote scenario requires a latent fundamental value")

    @staticmethod
    def _validate_observation_manifest(
        manifest: object,
        expected_role: ObservationUniverseRole,
        expected_manifest: ObservationUniverseManifest,
        field: str,
    ) -> None:
        if not isinstance(manifest, ObservationUniverseManifest):
            _fail(f"{field} manifest must be an ObservationUniverseManifest")
        if manifest.role is not expected_role:
            _fail(f"{field} observation manifest has the wrong role")
        if manifest != expected_manifest:
            _fail(
                f"{field} observation manifest does not match retained event "
                "identities and payloads"
            )

    @staticmethod
    def _require_canonical_observation_order(
        records: tuple[object, ...], field: str
    ) -> None:
        ordered = tuple(
            sorted(
                records,
                key=lambda record: (
                    record.scenario_id,  # type: ignore[attr-defined]
                    record.source_event_key.timestamp.isoformat(),  # type: ignore[attr-defined]
                    record.source_event_key.entity_id,  # type: ignore[attr-defined]
                    record.source_event_key.value,  # type: ignore[attr-defined]
                ),
            )
        )
        if records != ordered:
            _fail(f"{field} observations must use canonical semantic-event order")

    @staticmethod
    def _validate_scenario_manifest(
        manifest: object,
        expected_role: ScenarioUniverseRole,
        expected_manifest: ScenarioUniverseManifest,
        field: str,
    ) -> None:
        if not isinstance(manifest, ScenarioUniverseManifest):
            _fail(f"{field} scenario manifest must be a ScenarioUniverseManifest")
        if manifest.role is not expected_role:
            _fail(f"{field} scenario manifest has the wrong role")
        if manifest != expected_manifest:
            _fail(
                f"{field} scenario manifest does not match observed scenario "
                "IDs, source event keys, and payloads"
            )

    @staticmethod
    def _require_canonical_input_order(
        scenario_ids: tuple[str, ...],
        field: str,
    ) -> None:
        if scenario_ids != tuple(sorted(scenario_ids)):
            _fail(f"{field} scenarios must use canonical scenario-ID order")

    @staticmethod
    def _validate_source_event_keys(
        event_keys: object,
        scenario_ids: tuple[str, ...],
        field: str,
    ) -> tuple[SemanticEventKey, ...]:
        if (
            not isinstance(event_keys, tuple)
            or len(event_keys) != len(scenario_ids)
            or not all(isinstance(key, SemanticEventKey) for key in event_keys)
        ):
            _fail(f"{field} source event keys must align with scenario IDs")
        normalized = tuple(
            key for key in event_keys if isinstance(key, SemanticEventKey)
        )
        if any(
            key.scenario_id != scenario_id
            for scenario_id, key in zip(scenario_ids, normalized, strict=True)
        ):
            _fail(f"{field} source event key is bound to the wrong scenario")
        if len({key.value for key in normalized}) != len(normalized):
            _fail(f"{field} source event keys must be unique")
        return normalized


@dataclass(frozen=True, slots=True)
class MarketMetrics:
    """Market-level policy evaluation metrics with explicit units in names."""

    quoted_spread: float
    effective_spread: float
    realized_spread: float
    displayed_depth: float
    fill_probability: float
    mean_time_to_fill_among_filled: float
    price_impact: float
    realized_volatility: float
    jump_frequency: float
    value_at_risk: float
    expected_shortfall: float
    maximum_drawdown: float
    flash_crash_probability: float
    mean_relative_fundamental_deviation: float
    mean_information_response_half_life_among_recovered: float | None
    information_half_life_observed_fraction: float
    metric_config_sha256: str
    scenario_aggregation: ScenarioAggregation
    price_path_scenario_universe: ScenarioUniverseManifest
    risk_return_scenario_universe: ScenarioUniverseManifest
    portfolio_value_scenario_universe: ScenarioUniverseManifest
    quote_observation_universe: ObservationUniverseManifest
    trade_observation_universe: ObservationUniverseManifest
    order_outcome_universe: ObservationUniverseManifest
    information_response_universe: ObservationUniverseManifest

    def __post_init__(self) -> None:
        for field in (
            "quoted_spread",
            "displayed_depth",
            "mean_time_to_fill_among_filled",
            "realized_volatility",
            "mean_relative_fundamental_deviation",
        ):
            object.__setattr__(
                self,
                field,
                _positive(getattr(self, field), field, allow_zero=True),
            )
        for field in (
            "effective_spread",
            "realized_spread",
            "price_impact",
            "value_at_risk",
            "expected_shortfall",
        ):
            object.__setattr__(self, field, _finite(getattr(self, field), field))
        for field in (
            "fill_probability",
            "jump_frequency",
            "maximum_drawdown",
            "flash_crash_probability",
            "information_half_life_observed_fraction",
        ):
            object.__setattr__(
                self,
                field,
                _probability(getattr(self, field), field),
            )
        if self.expected_shortfall < self.value_at_risk:
            _fail("expected_shortfall must not be below value_at_risk")
        if self.mean_information_response_half_life_among_recovered is not None:
            object.__setattr__(
                self,
                "mean_information_response_half_life_among_recovered",
                _positive(
                    self.mean_information_response_half_life_among_recovered,
                    "mean_information_response_half_life_among_recovered",
                    allow_zero=True,
                ),
            )
        _sha256(self.metric_config_sha256, "metric_config_sha256")
        if not isinstance(self.scenario_aggregation, ScenarioAggregation):
            _fail("scenario_aggregation must be a ScenarioAggregation")
        expected_roles = (
            (
                self.price_path_scenario_universe,
                ScenarioUniverseRole.PRICE_RETURN_PATH,
                "price path",
            ),
            (
                self.risk_return_scenario_universe,
                ScenarioUniverseRole.RISK_RETURN,
                "risk return",
            ),
            (
                self.portfolio_value_scenario_universe,
                ScenarioUniverseRole.PORTFOLIO_VALUE_PATH,
                "portfolio value",
            ),
        )
        for manifest, role, field in expected_roles:
            if not isinstance(manifest, ScenarioUniverseManifest):
                _fail(f"{field} scenario universe must be a manifest")
            if manifest.role is not role:
                _fail(f"{field} scenario universe has the wrong role")
        expected_observation_roles = (
            (
                self.quote_observation_universe,
                ObservationUniverseRole.QUOTE,
                "quote",
            ),
            (
                self.trade_observation_universe,
                ObservationUniverseRole.TRADE,
                "trade",
            ),
            (
                self.order_outcome_universe,
                ObservationUniverseRole.ORDER_OUTCOME,
                "order outcome",
            ),
            (
                self.information_response_universe,
                ObservationUniverseRole.INFORMATION_RESPONSE,
                "information response",
            ),
        )
        for (
            observation_manifest,
            observation_role,
            observation_field,
        ) in expected_observation_roles:
            if not isinstance(observation_manifest, ObservationUniverseManifest):
                _fail(f"{observation_field} observation universe must be a manifest")
            if observation_manifest.role is not observation_role:
                _fail(f"{observation_field} observation universe has the wrong role")

    def to_mapping(self) -> dict[str, object]:
        return {
            "displayed_depth": self.displayed_depth,
            "effective_spread": self.effective_spread,
            "expected_shortfall": self.expected_shortfall,
            "fill_probability": self.fill_probability,
            "flash_crash_probability": self.flash_crash_probability,
            "information_half_life_observed_fraction": (
                self.information_half_life_observed_fraction
            ),
            "mean_information_response_half_life_among_recovered": (
                self.mean_information_response_half_life_among_recovered
            ),
            "jump_frequency": self.jump_frequency,
            "maximum_drawdown": self.maximum_drawdown,
            "mean_relative_fundamental_deviation": (
                self.mean_relative_fundamental_deviation
            ),
            "mean_time_to_fill_among_filled": (self.mean_time_to_fill_among_filled),
            "metric_config_sha256": self.metric_config_sha256,
            "price_impact": self.price_impact,
            "quoted_spread": self.quoted_spread,
            "realized_spread": self.realized_spread,
            "realized_volatility": self.realized_volatility,
            "scenario_aggregation": self.scenario_aggregation.value,
            "price_path_scenario_universe": (
                self.price_path_scenario_universe.to_mapping()
            ),
            "risk_return_scenario_universe": (
                self.risk_return_scenario_universe.to_mapping()
            ),
            "portfolio_value_scenario_universe": (
                self.portfolio_value_scenario_universe.to_mapping()
            ),
            "quote_observation_universe": (
                self.quote_observation_universe.to_mapping()
            ),
            "trade_observation_universe": (
                self.trade_observation_universe.to_mapping()
            ),
            "order_outcome_universe": self.order_outcome_universe.to_mapping(),
            "information_response_universe": (
                self.information_response_universe.to_mapping()
            ),
            "value_at_risk": self.value_at_risk,
        }


def _log_returns(path: PricePath) -> tuple[float, ...]:
    return tuple(
        math.log(current / previous)
        for previous, current in zip(path.prices, path.prices[1:], strict=False)
    )


def _is_flash_crash(path: PricePath, config: MarketMetricConfig) -> bool:
    for start_index, start_time in enumerate(path.times[:-1]):
        start_price = path.prices[start_index]
        for end_index in range(start_index + 1, len(path.times)):
            if path.times[end_index] - start_time > config.flash_crash_window:
                break
            drawdown = (start_price - path.prices[end_index]) / start_price
            if drawdown >= config.flash_crash_drawdown_threshold:
                return True
    return False


def evaluate_market_metrics(
    data: MarketEvaluationInput, config: MarketMetricConfig
) -> MarketMetrics:
    """Compute frozen market definitions without selecting a favorable horizon."""
    if not isinstance(data, MarketEvaluationInput) or not isinstance(
        config, MarketMetricConfig
    ):
        _fail("market metrics require validated input and config schemas")
    if any(
        not math.isclose(
            trade.realized_spread_horizon,
            config.realized_spread_horizon,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        for trade in data.trades
    ):
        _fail("trade realized-spread horizon differs from the frozen metric config")
    for path in data.price_paths:
        if any(
            not math.isclose(
                current - previous,
                config.return_sampling_interval,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            for previous, current in zip(path.times, path.times[1:], strict=False)
        ):
            _fail("price-path sampling interval differs from the frozen metric config")
    if (
        not math.isclose(
            data.risk_return_horizon,
            config.risk_return_horizon,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        or data.risk_return_sampling_unit != config.risk_return_sampling_unit
    ):
        _fail("portfolio return horizon or sampling unit differs from metric config")
    for times in data.portfolio_value_times:
        if any(
            not math.isclose(
                current - previous,
                config.portfolio_value_sampling_interval,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            for previous, current in zip(times, times[1:], strict=False)
        ) or not math.isclose(
            times[-1] - times[0],
            config.maximum_drawdown_horizon,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            _fail("portfolio value path cadence or horizon differs from metric config")
    observation_scenario_ids = data.quote_observation_manifest.scenario_ids
    quoted = fmean(
        fmean(
            quote.ask - quote.bid
            for quote in data.quotes
            if quote.scenario_id == scenario_id
        )
        for scenario_id in observation_scenario_ids
    )
    depth = fmean(
        fmean(
            quote.bid_depth + quote.ask_depth
            for quote in data.quotes
            if quote.scenario_id == scenario_id
        )
        for scenario_id in observation_scenario_ids
    )
    effective = fmean(
        fmean(
            2.0 * trade.side.sign * (trade.price - trade.midquote_at_execution)
            for trade in data.trades
            if trade.scenario_id == scenario_id
        )
        for scenario_id in observation_scenario_ids
    )
    realized = fmean(
        fmean(
            2.0 * trade.side.sign * (trade.price - trade.midquote_after_horizon)
            for trade in data.trades
            if trade.scenario_id == scenario_id
        )
        for scenario_id in observation_scenario_ids
    )
    impact = fmean(
        fmean(
            2.0
            * trade.side.sign
            * (trade.midquote_after_horizon - trade.midquote_at_execution)
            for trade in data.trades
            if trade.scenario_id == scenario_id
        )
        for scenario_id in observation_scenario_ids
    )
    filled_by_scenario = tuple(
        tuple(
            order
            for order in data.orders
            if order.scenario_id == scenario_id and order.filled_time is not None
        )
        for scenario_id in observation_scenario_ids
    )
    if any(not filled for filled in filled_by_scenario):
        _fail(
            "mean time to fill among filled orders is undefined when a scenario "
            "has no filled orders"
        )
    fill_probability = fmean(
        len(filled) / sum(order.scenario_id == scenario_id for order in data.orders)
        for scenario_id, filled in zip(
            observation_scenario_ids, filled_by_scenario, strict=True
        )
    )
    mean_time_to_fill_among_filled = fmean(
        fmean(
            order.filled_time - order.submitted_time
            for order in filled
            if order.filled_time is not None
        )
        for filled in filled_by_scenario
    )
    path_returns = tuple(_log_returns(path) for path in data.price_paths)
    path_volatilities = tuple(
        math.sqrt(math.fsum(value * value for value in returns))
        * config.volatility_annualization_factor
        for returns in path_returns
    )
    realized_volatility = fmean(path_volatilities)
    jump_frequency = fmean(
        sum(abs(value) >= config.jump_log_return_threshold for value in returns)
        / len(returns)
        for returns in path_returns
    )
    losses = tuple(-value for value in data.portfolio_returns)
    value_at_risk = _quantile(losses, config.risk_alpha)
    tail_losses = tuple(value for value in losses if value >= value_at_risk)
    expected_shortfall = fmean(tail_losses)
    maximum_drawdown = fmean(
        _maximum_drawdown(path) for path in data.portfolio_value_paths
    )
    flash_probability = fmean(
        float(_is_flash_crash(path, config)) for path in data.price_paths
    )
    mean_fundamental_deviation = fmean(
        fmean(
            abs(quote.midpoint - quote.latent_fundamental) / quote.latent_fundamental
            for quote in data.quotes
            if quote.scenario_id == scenario_id and quote.latent_fundamental is not None
        )
        for scenario_id in observation_scenario_ids
    )
    half_lives_by_scenario: list[tuple[float, ...]] = []
    observed_fractions: list[float] = []
    for scenario_id in observation_scenario_ids:
        responses = tuple(
            response
            for response in data.information_responses
            if response.scenario_id == scenario_id
        )
        scenario_half_lives: list[float] = []
        for response in responses:
            threshold = response.absolute_deviations[0] / 2.0
            for time, deviation in zip(
                response.times, response.absolute_deviations, strict=True
            ):
                if deviation <= threshold:
                    scenario_half_lives.append(time - response.event_time)
                    break
        half_lives_by_scenario.append(tuple(scenario_half_lives))
        observed_fractions.append(len(scenario_half_lives) / len(responses))
    recovered_scenario_means = tuple(
        fmean(values) for values in half_lives_by_scenario if values
    )
    observed_fraction = fmean(observed_fractions)
    return MarketMetrics(
        quoted_spread=quoted,
        effective_spread=effective,
        realized_spread=realized,
        displayed_depth=depth,
        fill_probability=fill_probability,
        mean_time_to_fill_among_filled=mean_time_to_fill_among_filled,
        price_impact=impact,
        realized_volatility=realized_volatility,
        jump_frequency=jump_frequency,
        value_at_risk=value_at_risk,
        expected_shortfall=expected_shortfall,
        maximum_drawdown=maximum_drawdown,
        flash_crash_probability=flash_probability,
        mean_relative_fundamental_deviation=mean_fundamental_deviation,
        mean_information_response_half_life_among_recovered=(
            fmean(recovered_scenario_means) if recovered_scenario_means else None
        ),
        information_half_life_observed_fraction=observed_fraction,
        metric_config_sha256=config.sha256,
        scenario_aggregation=config.scenario_aggregation,
        price_path_scenario_universe=data.price_path_scenario_manifest,
        risk_return_scenario_universe=data.risk_return_scenario_manifest,
        portfolio_value_scenario_universe=data.portfolio_value_scenario_manifest,
        quote_observation_universe=data.quote_observation_manifest,
        trade_observation_universe=data.trade_observation_manifest,
        order_outcome_universe=data.order_outcome_manifest,
        information_response_universe=data.information_response_manifest,
    )


@dataclass(frozen=True, slots=True)
class AgentOutcome:
    """One agent outcome with a pre-treatment group label."""

    agent_id: str
    cohort: FrozenCohort
    pnl: float
    execution_cost: float
    submitted_quantity: float
    filled_quantity: float
    inventory_path: tuple[float, ...]
    wealth_path: tuple[float, ...]
    survived: bool
    utility_equivalent_change: float
    fees_paid: float = 0.0
    taxes_paid: float = 0.0
    transfer_received: float = 0.0
    crowded_out: bool = False

    def __post_init__(self) -> None:
        _identifier(self.agent_id, "agent_id")
        if not isinstance(self.cohort, FrozenCohort):
            _fail("agent cohort must be a FrozenCohort")
        if self.cohort.basis is CohortBasis.POST_TREATMENT_DESCRIPTIVE:
            _fail("group labels must be defined from pre-treatment information")
        for field in ("pnl", "execution_cost", "utility_equivalent_change"):
            object.__setattr__(self, field, _finite(getattr(self, field), field))
        for field in ("submitted_quantity", "filled_quantity"):
            object.__setattr__(
                self, field, _positive(getattr(self, field), field, allow_zero=True)
            )
        if self.filled_quantity > self.submitted_quantity:
            _fail("agent filled quantity exceeds submitted quantity")
        object.__setattr__(
            self, "inventory_path", _values(self.inventory_path, "inventory_path")
        )
        wealth = _values(self.wealth_path, "wealth_path")
        if any(value <= 0.0 for value in wealth):
            _fail("wealth_path must contain positive values")
        object.__setattr__(self, "wealth_path", wealth)
        if not isinstance(self.survived, bool):
            _fail("survived must be a boolean")
        for field in ("fees_paid", "taxes_paid"):
            object.__setattr__(
                self,
                field,
                _positive(getattr(self, field), field, allow_zero=True),
            )
        object.__setattr__(
            self,
            "transfer_received",
            _finite(self.transfer_received, "transfer_received"),
        )
        if not isinstance(self.crowded_out, bool):
            _fail("crowded_out must be a boolean")

    @property
    def group_id(self) -> str:
        """Return the frozen cohort identifier used for aggregation."""
        return self.cohort.cohort_id


@dataclass(frozen=True, slots=True)
class GroupMetrics:
    """Distribution-aware aggregate for one frozen pre-treatment group."""

    group_id: str
    cohort_basis: CohortBasis
    cohort_definition_sha256: str
    agent_count: int
    mean_pnl: float
    mean_execution_cost: float
    fill_rate: float
    mean_inventory_risk: float
    mean_drawdown: float
    survival_probability: float
    mean_utility_equivalent_change: float
    mean_wealth_change: float
    mean_return_to_rms_risk: float
    mean_fee_burden: float
    mean_tax_burden: float
    crowd_out_probability: float
    mean_transfer_received: float
    total_fees_paid: float
    total_taxes_paid: float

    def __post_init__(self) -> None:
        _identifier(self.group_id, "group metric group_id")
        if not isinstance(self.cohort_basis, CohortBasis):
            _fail("group metric cohort_basis must be a CohortBasis")
        if _SHA256.fullmatch(self.cohort_definition_sha256) is None:
            _fail("group metric cohort_definition_sha256 must be a SHA-256")
        if (
            isinstance(self.agent_count, bool)
            or not isinstance(self.agent_count, int)
            or self.agent_count < 1
        ):
            _fail("group metric agent_count must be a positive integer")
        probability_fields = (
            "fill_rate",
            "mean_drawdown",
            "survival_probability",
            "crowd_out_probability",
        )
        for field in probability_fields:
            object.__setattr__(self, field, _probability(getattr(self, field), field))
        non_negative_fields = (
            "mean_inventory_risk",
            "mean_fee_burden",
            "mean_tax_burden",
            "total_fees_paid",
            "total_taxes_paid",
        )
        for field in non_negative_fields:
            object.__setattr__(
                self, field, _positive(getattr(self, field), field, allow_zero=True)
            )
        unrestricted_fields = (
            "mean_pnl",
            "mean_execution_cost",
            "mean_utility_equivalent_change",
            "mean_wealth_change",
            "mean_return_to_rms_risk",
            "mean_transfer_received",
        )
        for field in unrestricted_fields:
            object.__setattr__(self, field, _finite(getattr(self, field), field))

    def to_mapping(self) -> dict[str, object]:
        return {
            "agent_count": self.agent_count,
            "fill_rate": self.fill_rate,
            "group_id": self.group_id,
            "cohort_basis": self.cohort_basis.value,
            "cohort_definition_sha256": self.cohort_definition_sha256,
            "mean_drawdown": self.mean_drawdown,
            "mean_execution_cost": self.mean_execution_cost,
            "mean_inventory_risk": self.mean_inventory_risk,
            "mean_pnl": self.mean_pnl,
            "mean_utility_equivalent_change": self.mean_utility_equivalent_change,
            "mean_wealth_change": self.mean_wealth_change,
            "mean_return_to_rms_risk": self.mean_return_to_rms_risk,
            "mean_fee_burden": self.mean_fee_burden,
            "mean_tax_burden": self.mean_tax_burden,
            "crowd_out_probability": self.crowd_out_probability,
            "mean_transfer_received": self.mean_transfer_received,
            "survival_probability": self.survival_probability,
            "total_fees_paid": self.total_fees_paid,
            "total_taxes_paid": self.total_taxes_paid,
        }


def evaluate_group_metrics(
    outcomes: tuple[AgentOutcome, ...],
) -> tuple[GroupMetrics, ...]:
    """Compute every observed pre-treatment group; never auto-select a subgroup."""
    if (
        not isinstance(outcomes, tuple)
        or not outcomes
        or not all(isinstance(item, AgentOutcome) for item in outcomes)
    ):
        _fail("group evaluation requires a non-empty immutable AgentOutcome tuple")
    agent_ids = tuple(item.agent_id for item in outcomes)
    if len(set(agent_ids)) != len(agent_ids):
        _fail("agent outcome IDs must be unique")
    results: list[GroupMetrics] = []
    for group_id in sorted({item.group_id for item in outcomes}):
        members = tuple(item for item in outcomes if item.group_id == group_id)
        cohort_commitments = {
            (item.cohort.basis, item.cohort.definition_sha256) for item in members
        }
        if len(cohort_commitments) != 1:
            _fail(f"group {group_id!r} has conflicting cohort commitments")
        cohort_basis, cohort_definition_sha256 = next(iter(cohort_commitments))
        submitted = math.fsum(item.submitted_quantity for item in members)
        filled = math.fsum(item.filled_quantity for item in members)
        if submitted <= 0.0:
            _fail(f"group {group_id!r} has no submitted quantity")
        inventory_risks = tuple(
            math.sqrt(fmean(value * value for value in item.inventory_path))
            for item in members
        )
        return_to_risk: list[float] = []
        for item in members:
            returns = tuple(
                (current - previous) / previous
                for previous, current in zip(
                    item.wealth_path,
                    item.wealth_path[1:],
                    strict=False,
                )
            )
            if not returns:
                return_to_risk.append(0.0)
                continue
            rms = math.sqrt(fmean(value * value for value in returns))
            return_to_risk.append(0.0 if rms == 0.0 else fmean(returns) / rms)
        results.append(
            GroupMetrics(
                group_id=group_id,
                cohort_basis=cohort_basis,
                cohort_definition_sha256=cohort_definition_sha256,
                agent_count=len(members),
                mean_pnl=fmean(item.pnl for item in members),
                mean_execution_cost=fmean(item.execution_cost for item in members),
                fill_rate=filled / submitted,
                mean_inventory_risk=fmean(inventory_risks),
                mean_drawdown=fmean(
                    _maximum_drawdown(item.wealth_path) for item in members
                ),
                survival_probability=fmean(float(item.survived) for item in members),
                mean_utility_equivalent_change=fmean(
                    item.utility_equivalent_change for item in members
                ),
                mean_wealth_change=fmean(
                    item.wealth_path[-1] - item.wealth_path[0] for item in members
                ),
                mean_return_to_rms_risk=fmean(return_to_risk),
                mean_fee_burden=fmean(
                    item.fees_paid / item.wealth_path[0] for item in members
                ),
                mean_tax_burden=fmean(
                    item.taxes_paid / item.wealth_path[0] for item in members
                ),
                crowd_out_probability=fmean(
                    float(item.crowded_out) for item in members
                ),
                mean_transfer_received=fmean(
                    item.transfer_received for item in members
                ),
                total_fees_paid=math.fsum(item.fees_paid for item in members),
                total_taxes_paid=math.fsum(item.taxes_paid for item in members),
            )
        )
    return tuple(results)


class IntervalKind(StrEnum):
    """Statistical meaning of a reported interval; names are not interchangeable."""

    POSTERIOR_CREDIBLE = "posterior_credible_interval"
    BOOTSTRAP_CONFIDENCE = "bootstrap_confidence_interval"
    PREDICTIVE = "predictive_interval"
    RANDOMIZATION = "randomization_interval"


class DistributionSampleRole(StrEnum):
    """Closed provenance roles for distribution-scoring inputs."""

    PREDICTIVE = "predictive"
    HELD_OUT_PREDECLARED = "held_out_predeclared"


@dataclass(frozen=True, slots=True)
class DistributionSampleArtifact:
    """Immutable synthetic sample plus provenance and temporal commitments.

    Non-synthetic construction remains disabled until a typed temporal-firewall
    verifier receipt can be validated here. Bare hashes, timestamps, and a
    caller-provided boolean are not authority for real-study scoring.
    """

    schema_version: int
    artifact_id: str
    role: DistributionSampleRole
    values: tuple[float, ...]
    values_sha256: str
    data_manifest_sha256: str
    study_protocol_sha256: str
    available_at: datetime
    frozen_at: datetime
    estimand_id: str
    sampling_unit: str
    synthetic_only: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            _fail("distribution sample schema_version must equal 1")
        _identifier(self.artifact_id, "distribution sample artifact_id")
        if not isinstance(self.role, DistributionSampleRole):
            _fail("distribution sample role must be a DistributionSampleRole")
        normalized = _values(self.values, "distribution sample values")
        object.__setattr__(self, "values", normalized)
        _sha256(self.values_sha256, "distribution sample values_sha256")
        expected_values_sha256 = sha256_bytes(canonical_json_bytes(list(normalized)))
        if self.values_sha256 != expected_values_sha256:
            _fail("distribution sample values hash does not match values")
        _sha256(
            self.data_manifest_sha256,
            "distribution sample data_manifest_sha256",
        )
        _sha256(
            self.study_protocol_sha256,
            "distribution sample study_protocol_sha256",
        )
        available = _aware(self.available_at, "distribution sample available_at")
        frozen = _aware(self.frozen_at, "distribution sample frozen_at")
        if available > frozen:
            _fail("distribution sample cannot be frozen before it is available")
        object.__setattr__(self, "available_at", available)
        object.__setattr__(self, "frozen_at", frozen)
        _identifier(self.estimand_id, "distribution sample estimand_id")
        _identifier(self.sampling_unit, "distribution sample sampling_unit")
        if self.synthetic_only is not True:
            _fail(
                "non-synthetic distribution samples require a future typed "
                "temporal-firewall verifier receipt"
            )

    @property
    def sha256(self) -> str:
        """Return the full sample and provenance commitment."""
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def to_mapping(self) -> dict[str, object]:
        """Return the canonical typed sample artifact."""
        return {
            "artifact_id": self.artifact_id,
            "available_at": self.available_at.isoformat(),
            "data_manifest_sha256": self.data_manifest_sha256,
            "estimand_id": self.estimand_id,
            "frozen_at": self.frozen_at.isoformat(),
            "role": self.role.value,
            "sampling_unit": self.sampling_unit,
            "schema_version": self.schema_version,
            "study_protocol_sha256": self.study_protocol_sha256,
            "synthetic_only": self.synthetic_only,
            "values": list(self.values),
            "values_sha256": self.values_sha256,
        }


@dataclass(frozen=True, slots=True)
class PairedRankingObservation:
    """One value whose cross-artifact identity is frozen before evaluation.

    Ranking is scientifically meaningful only when the same unit can be
    matched between the predictive and held-out artifacts.  ``event_key`` is
    therefore part of the identity rather than an incidental random seed.
    """

    item_id: str
    event_key: str
    value: float

    def __post_init__(self) -> None:
        _identifier(self.item_id, "ranking item_id")
        if (
            not isinstance(self.event_key, str)
            or _EVENT_KEY.fullmatch(self.event_key) is None
        ):
            _fail("ranking event_key must be a semantic PolicyTwin event key")
        object.__setattr__(self, "value", _finite(self.value, "ranking value"))

    def identity_mapping(self) -> dict[str, str]:
        """Return the value-free identity committed before outcomes exist."""
        return {"event_key": self.event_key, "item_id": self.item_id}

    def to_mapping(self) -> dict[str, object]:
        return {**self.identity_mapping(), "value": self.value}


def ranking_identity_manifest_sha256(
    observations: tuple[PairedRankingObservation, ...],
) -> str:
    """Hash a canonical, value-free cross-world ranking identity manifest."""
    if not isinstance(observations, tuple) or len(observations) < 2:
        _fail("ranking identity manifest requires at least two observations")
    if not all(isinstance(item, PairedRankingObservation) for item in observations):
        _fail("ranking identity manifest contains an invalid observation")
    ordered = tuple(
        sorted(observations, key=lambda item: (item.item_id, item.event_key))
    )
    identities = tuple((item.item_id, item.event_key) for item in ordered)
    if len({item_id for item_id, _ in identities}) != len(identities):
        _fail("ranking item identities must be unique")
    if len({event_key for _, event_key in identities}) != len(identities):
        _fail("ranking event keys must be unique")
    return sha256_bytes(
        canonical_json_bytes([item.identity_mapping() for item in ordered])
    )


@dataclass(frozen=True, slots=True)
class PairedRankingSampleArtifact:
    """Frozen values plus an exact value-free identity commitment.

    Bare arrays are deliberately unsupported: a correlation between arrays
    with coincidentally equal lengths is not evidence that their units match.
    Real-study construction remains closed until a temporal/lineage verifier
    can authorize both the identity manifest and the values.
    """

    schema_version: int
    artifact_id: str
    role: DistributionSampleRole
    observations: tuple[PairedRankingObservation, ...]
    identity_manifest_sha256: str
    data_manifest_sha256: str
    study_protocol_sha256: str
    available_at: datetime
    frozen_at: datetime
    estimand_id: str
    sampling_unit: str
    synthetic_only: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            _fail("ranking sample schema_version must equal 1")
        _identifier(self.artifact_id, "ranking sample artifact_id")
        if not isinstance(self.role, DistributionSampleRole):
            _fail("ranking sample role must be a DistributionSampleRole")
        if not isinstance(self.observations, tuple) or len(self.observations) < 2:
            _fail("ranking sample requires at least two observations")
        if not all(
            isinstance(item, PairedRankingObservation) for item in self.observations
        ):
            _fail("ranking sample contains an invalid observation")
        ordered = tuple(
            sorted(self.observations, key=lambda item: (item.item_id, item.event_key))
        )
        object.__setattr__(self, "observations", ordered)
        _sha256(self.identity_manifest_sha256, "ranking identity_manifest_sha256")
        if self.identity_manifest_sha256 != ranking_identity_manifest_sha256(ordered):
            _fail("ranking identity manifest hash does not match observations")
        _sha256(self.data_manifest_sha256, "ranking data_manifest_sha256")
        _sha256(self.study_protocol_sha256, "ranking study_protocol_sha256")
        available = _aware(self.available_at, "ranking sample available_at")
        frozen = _aware(self.frozen_at, "ranking sample frozen_at")
        if available > frozen:
            _fail("ranking sample cannot be frozen before it is available")
        object.__setattr__(self, "available_at", available)
        object.__setattr__(self, "frozen_at", frozen)
        _identifier(self.estimand_id, "ranking sample estimand_id")
        _identifier(self.sampling_unit, "ranking sample sampling_unit")
        if self.synthetic_only is not True:
            _fail(
                "non-synthetic ranking samples require a typed temporal and "
                "identity-lineage verifier receipt"
            )

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def to_mapping(self) -> dict[str, object]:
        return {
            "artifact_id": self.artifact_id,
            "available_at": self.available_at.isoformat(),
            "data_manifest_sha256": self.data_manifest_sha256,
            "estimand_id": self.estimand_id,
            "frozen_at": self.frozen_at.isoformat(),
            "identity_manifest_sha256": self.identity_manifest_sha256,
            "observations": [item.to_mapping() for item in self.observations],
            "role": self.role.value,
            "sampling_unit": self.sampling_unit,
            "schema_version": self.schema_version,
            "study_protocol_sha256": self.study_protocol_sha256,
            "synthetic_only": self.synthetic_only,
        }


@dataclass(frozen=True, slots=True)
class PairedRankingMetrics:
    """Rank agreement for exactly matched predictive and held-out units."""

    spearman_rho: float
    kendall_tau_b: float
    matched_count: int
    identity_manifest_sha256: str
    predictive_artifact_sha256: str
    held_out_artifact_sha256: str
    estimand_id: str
    sampling_unit: str

    def __post_init__(self) -> None:
        for field in ("spearman_rho", "kendall_tau_b"):
            value = _finite(getattr(self, field), field)
            if not -1.0 <= value <= 1.0:
                _fail(f"{field} must lie in [-1, 1]")
            object.__setattr__(self, field, value)
        if (
            isinstance(self.matched_count, bool)
            or not isinstance(self.matched_count, int)
            or self.matched_count < 2
        ):
            _fail("matched_count must be an integer of at least two")
        for field in (
            "identity_manifest_sha256",
            "predictive_artifact_sha256",
            "held_out_artifact_sha256",
        ):
            _sha256(getattr(self, field), field)
        _identifier(self.estimand_id, "ranking metrics estimand_id")
        _identifier(self.sampling_unit, "ranking metrics sampling_unit")

    def to_mapping(self) -> dict[str, object]:
        return {
            "estimand_id": self.estimand_id,
            "held_out_artifact_sha256": self.held_out_artifact_sha256,
            "identity_manifest_sha256": self.identity_manifest_sha256,
            "kendall_tau_b": self.kendall_tau_b,
            "matched_count": self.matched_count,
            "predictive_artifact_sha256": self.predictive_artifact_sha256,
            "sampling_unit": self.sampling_unit,
            "spearman_rho": self.spearman_rho,
        }


def _average_ranks(values: tuple[float, ...]) -> tuple[float, ...]:
    indexed = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        for index, _ in indexed[start:end]:
            ranks[index] = average_rank
        start = end
    return tuple(ranks)


def _pearson(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    left_mean = fmean(left)
    right_mean = fmean(right)
    numerator = math.fsum(
        (first - left_mean) * (second - right_mean)
        for first, second in zip(left, right, strict=True)
    )
    left_ss = math.fsum((value - left_mean) ** 2 for value in left)
    right_ss = math.fsum((value - right_mean) ** 2 for value in right)
    denominator = math.sqrt(left_ss * right_ss)
    if denominator == 0.0:
        _fail("rank correlation is undefined for a constant ranking")
    return max(-1.0, min(1.0, numerator / denominator))


def _kendall_tau_b(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    concordant = 0
    discordant = 0
    tied_left = 0
    tied_right = 0
    for first in range(len(left) - 1):
        for second in range(first + 1, len(left)):
            left_delta = left[first] - left[second]
            right_delta = right[first] - right[second]
            if left_delta == 0.0 and right_delta == 0.0:
                continue
            if left_delta == 0.0:
                tied_left += 1
            elif right_delta == 0.0:
                tied_right += 1
            elif left_delta * right_delta > 0.0:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + tied_left) * (concordant + discordant + tied_right)
    )
    if denominator == 0.0:
        _fail("Kendall tau-b is undefined for the supplied rankings")
    return (concordant - discordant) / denominator


def evaluate_paired_ranking_metrics(
    predictive_sample: PairedRankingSampleArtifact,
    held_out_sample: PairedRankingSampleArtifact,
) -> PairedRankingMetrics:
    """Evaluate rankings only after exact unit/EventKey identity matching."""
    if not isinstance(predictive_sample, PairedRankingSampleArtifact) or not isinstance(
        held_out_sample, PairedRankingSampleArtifact
    ):
        _fail("ranking evaluation requires typed ranking sample artifacts")
    if (
        predictive_sample.role is not DistributionSampleRole.PREDICTIVE
        or held_out_sample.role is not DistributionSampleRole.HELD_OUT_PREDECLARED
    ):
        _fail("ranking sample roles must be predictive and held_out_predeclared")
    if predictive_sample.study_protocol_sha256 != held_out_sample.study_protocol_sha256:
        _fail("ranking samples must share one frozen study protocol")
    if predictive_sample.frozen_at >= held_out_sample.available_at:
        _fail(
            "predictive ranking must be frozen before held-out outcomes are available"
        )
    if predictive_sample.estimand_id != held_out_sample.estimand_id:
        _fail("ranking samples must share one estimand")
    if predictive_sample.sampling_unit != held_out_sample.sampling_unit:
        _fail("ranking samples must share one sampling unit")
    if (
        predictive_sample.identity_manifest_sha256
        != held_out_sample.identity_manifest_sha256
    ):
        _fail("ranking samples do not share the frozen identity manifest")
    predictive_identities = tuple(
        item.identity_mapping() for item in predictive_sample.observations
    )
    held_out_identities = tuple(
        item.identity_mapping() for item in held_out_sample.observations
    )
    if predictive_identities != held_out_identities:
        _fail("ranking identity rows differ despite their manifest commitment")
    predictive_values = tuple(item.value for item in predictive_sample.observations)
    held_out_values = tuple(item.value for item in held_out_sample.observations)
    return PairedRankingMetrics(
        spearman_rho=_pearson(
            _average_ranks(predictive_values), _average_ranks(held_out_values)
        ),
        kendall_tau_b=_kendall_tau_b(predictive_values, held_out_values),
        matched_count=len(predictive_values),
        identity_manifest_sha256=predictive_sample.identity_manifest_sha256,
        predictive_artifact_sha256=predictive_sample.sha256,
        held_out_artifact_sha256=held_out_sample.sha256,
        estimand_id=predictive_sample.estimand_id,
        sampling_unit=predictive_sample.sampling_unit,
    )


@dataclass(frozen=True, slots=True)
class DistributionMetricConfig:
    """Frozen scoring, tail, and uncertainty-semantics definitions."""

    quantile_levels: tuple[float, ...]
    interval_level: float
    tail_event_threshold: float
    reported_interval_levels: tuple[float, ...] = ()
    interval_kind: IntervalKind = IntervalKind.PREDICTIVE
    interval_method_id: str = "central_empirical_quantile"
    estimand_id: str = "policy_minus_baseline_difference"
    sampling_unit: str = "paired_scenario"
    minimum_valid_draws: int = 2
    mmd_bandwidth: float = 1.0

    def __post_init__(self) -> None:
        levels = _values(self.quantile_levels, "quantile_levels")
        if len(set(levels)) != len(levels) or any(
            level <= 0.0 or level >= 1.0 for level in levels
        ):
            _fail("quantile_levels must be unique and strictly inside (0, 1)")
        object.__setattr__(self, "quantile_levels", levels)
        interval = _probability(self.interval_level, "interval_level")
        if interval <= 0.0 or interval >= 1.0:
            _fail("interval_level must lie strictly inside (0, 1)")
        object.__setattr__(self, "interval_level", interval)
        if not isinstance(self.reported_interval_levels, tuple):
            _fail("reported_interval_levels must be an immutable tuple")
        reported = tuple(
            _probability(value, "reported interval level")
            for value in self.reported_interval_levels
        )
        if any(value <= 0.0 or value >= 1.0 for value in reported):
            _fail("reported interval levels must lie strictly inside (0, 1)")
        all_intervals = tuple(sorted(set((interval, *reported))))
        object.__setattr__(self, "reported_interval_levels", all_intervals)
        if not isinstance(self.interval_kind, IntervalKind):
            _fail("interval_kind must be an IntervalKind")
        _identifier(self.interval_method_id, "interval_method_id")
        if (
            self.interval_kind is not IntervalKind.PREDICTIVE
            or self.interval_method_id != "central_empirical_quantile"
        ):
            _fail(
                "tuple-based distribution scoring supports only "
                "predictive_interval with central_empirical_quantile; "
                "credible, confidence, and randomization intervals require "
                "provenance-typed estimator artifacts"
            )
        _identifier(self.estimand_id, "estimand_id")
        _identifier(self.sampling_unit, "sampling_unit")
        if (
            isinstance(self.minimum_valid_draws, bool)
            or not isinstance(self.minimum_valid_draws, int)
            or self.minimum_valid_draws < 2
        ):
            _fail("minimum_valid_draws must be an integer of at least two")
        object.__setattr__(
            self,
            "mmd_bandwidth",
            _positive(self.mmd_bandwidth, "mmd_bandwidth"),
        )
        object.__setattr__(
            self,
            "tail_event_threshold",
            _finite(self.tail_event_threshold, "tail_event_threshold"),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "estimand_id": self.estimand_id,
            "interval_kind": self.interval_kind.value,
            "interval_level": self.interval_level,
            "interval_method_id": self.interval_method_id,
            "minimum_valid_draws": self.minimum_valid_draws,
            "mmd_bandwidth": self.mmd_bandwidth,
            "quantile_levels": list(self.quantile_levels),
            "reported_interval_levels": list(self.reported_interval_levels),
            "sampling_unit": self.sampling_unit,
            "tail_event_threshold": self.tail_event_threshold,
        }


@dataclass(frozen=True, slots=True)
class IntervalEstimate:
    """One central interval with explicit level, score, and held-out coverage."""

    level: float
    lower: float
    upper: float
    interval_score: float
    empirical_coverage: float

    def __post_init__(self) -> None:
        level = _probability(self.level, "interval level")
        if level <= 0.0 or level >= 1.0:
            _fail("interval level must lie strictly inside (0, 1)")
        lower = _finite(self.lower, "interval lower")
        upper = _finite(self.upper, "interval upper")
        if lower > upper:
            _fail("interval lower must not exceed upper")
        score = _positive(self.interval_score, "interval score", allow_zero=True)
        coverage = _probability(self.empirical_coverage, "empirical coverage")
        object.__setattr__(self, "level", level)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "interval_score", score)
        object.__setattr__(self, "empirical_coverage", coverage)

    def to_mapping(self) -> dict[str, object]:
        return {
            "empirical_coverage": self.empirical_coverage,
            "interval_score": self.interval_score,
            "level": self.level,
            "lower": self.lower,
            "upper": self.upper,
        }


@dataclass(frozen=True, slots=True)
class DistributionMetrics:
    """Proper scores and distribution comparisons for frozen samples."""

    quantiles: tuple[tuple[float, float], ...]
    sign_probability: float
    crps: float
    energy_score: float
    interval_lower: float
    interval_upper: float
    interval_score: float
    empirical_coverage: float
    wasserstein_distance: float
    tail_event_probability: float
    brier_score: float
    intervals: tuple[IntervalEstimate, ...]
    interval_kind: IntervalKind
    interval_method_id: str
    estimand_id: str
    sampling_unit: str
    predictive_draw_count: int
    empirical_draw_count: int
    mean_absolute_error: float
    root_mean_squared_error: float
    direction_accuracy: float
    worst_five_percent_mean: float
    maximum_mean_discrepancy_squared: float
    predictive_sample_artifact_sha256: str
    held_out_sample_artifact_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.quantiles, tuple) or not self.quantiles:
            _fail("quantiles must be a non-empty immutable tuple")
        normalized_quantiles: list[tuple[float, float]] = []
        for item in self.quantiles:
            if not isinstance(item, tuple) or len(item) != 2:
                _fail("each quantile must be a (level, value) tuple")
            level = _probability(item[0], "quantile level")
            if level <= 0.0 or level >= 1.0:
                _fail("quantile levels must lie strictly inside (0, 1)")
            normalized_quantiles.append((level, _finite(item[1], "quantile value")))
        levels = tuple(level for level, _ in normalized_quantiles)
        if len(set(levels)) != len(levels):
            _fail("quantile levels must be unique")
        ordered_quantiles = sorted(normalized_quantiles)
        if any(
            current_value < previous_value
            for (_, previous_value), (_, current_value) in zip(
                ordered_quantiles,
                ordered_quantiles[1:],
                strict=False,
            )
        ):
            _fail("quantile values must be non-decreasing by level")
        object.__setattr__(self, "quantiles", tuple(normalized_quantiles))
        for field in (
            "sign_probability",
            "empirical_coverage",
            "tail_event_probability",
            "brier_score",
            "direction_accuracy",
        ):
            object.__setattr__(
                self,
                field,
                _probability(getattr(self, field), field),
            )
        for field in (
            "crps",
            "energy_score",
            "interval_score",
            "wasserstein_distance",
            "mean_absolute_error",
            "root_mean_squared_error",
            "maximum_mean_discrepancy_squared",
        ):
            object.__setattr__(
                self,
                field,
                _positive(getattr(self, field), field, allow_zero=True),
            )
        lower = _finite(self.interval_lower, "interval_lower")
        upper = _finite(self.interval_upper, "interval_upper")
        if lower > upper:
            _fail("interval_lower must not exceed interval_upper")
        object.__setattr__(self, "interval_lower", lower)
        object.__setattr__(self, "interval_upper", upper)
        object.__setattr__(
            self,
            "worst_five_percent_mean",
            _finite(self.worst_five_percent_mean, "worst_five_percent_mean"),
        )
        if (
            not isinstance(self.intervals, tuple)
            or not self.intervals
            or not all(isinstance(item, IntervalEstimate) for item in self.intervals)
        ):
            _fail("intervals must be a non-empty immutable tuple")
        interval_levels = tuple(item.level for item in self.intervals)
        if len(set(interval_levels)) != len(interval_levels):
            _fail("interval levels must be unique")
        if interval_levels != tuple(sorted(interval_levels)):
            _fail("intervals must use canonical increasing level order")
        if not any(
            item.lower == lower
            and item.upper == upper
            and item.interval_score == self.interval_score
            and item.empirical_coverage == self.empirical_coverage
            for item in self.intervals
        ):
            _fail("primary interval fields must match one retained interval")
        if not isinstance(self.interval_kind, IntervalKind):
            _fail("interval_kind must be an IntervalKind")
        _identifier(self.interval_method_id, "interval_method_id")
        _identifier(self.estimand_id, "estimand_id")
        _identifier(self.sampling_unit, "sampling_unit")
        for field in ("predictive_draw_count", "empirical_draw_count"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                _fail(f"{field} must be an integer of at least two")
        _sha256(
            self.predictive_sample_artifact_sha256,
            "predictive_sample_artifact_sha256",
        )
        _sha256(
            self.held_out_sample_artifact_sha256,
            "held_out_sample_artifact_sha256",
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "brier_score": self.brier_score,
            "crps": self.crps,
            "empirical_coverage": self.empirical_coverage,
            "energy_score": self.energy_score,
            "empirical_draw_count": self.empirical_draw_count,
            "estimand_id": self.estimand_id,
            "interval_kind": self.interval_kind.value,
            "interval_lower": self.interval_lower,
            "interval_method_id": self.interval_method_id,
            "interval_score": self.interval_score,
            "interval_upper": self.interval_upper,
            "intervals": [interval.to_mapping() for interval in self.intervals],
            "direction_accuracy": self.direction_accuracy,
            "maximum_mean_discrepancy_squared": (self.maximum_mean_discrepancy_squared),
            "mean_absolute_error": self.mean_absolute_error,
            "predictive_draw_count": self.predictive_draw_count,
            "predictive_sample_artifact_sha256": (
                self.predictive_sample_artifact_sha256
            ),
            "quantiles": [
                {"level": level, "value": value} for level, value in self.quantiles
            ],
            "root_mean_squared_error": self.root_mean_squared_error,
            "sampling_unit": self.sampling_unit,
            "sign_probability": self.sign_probability,
            "tail_event_probability": self.tail_event_probability,
            "wasserstein_distance": self.wasserstein_distance,
            "worst_five_percent_mean": self.worst_five_percent_mean,
            "held_out_sample_artifact_sha256": (self.held_out_sample_artifact_sha256),
        }


def _crps(forecast: tuple[float, ...], observation: float) -> float:
    first = fmean(abs(value - observation) for value in forecast)
    pairwise = fmean(abs(left - right) for left in forecast for right in forecast)
    return max(0.0, first - 0.5 * pairwise)


def _wasserstein(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    """Return exact empirical one-dimensional W1 for arbitrary sample sizes."""
    ordered_left = tuple(sorted(left))
    ordered_right = tuple(sorted(right))
    left_index = 0
    right_index = 0
    previous = min(ordered_left[0], ordered_right[0])
    distance = 0.0
    while left_index < len(ordered_left) or right_index < len(ordered_right):
        left_value = (
            ordered_left[left_index] if left_index < len(ordered_left) else math.inf
        )
        right_value = (
            ordered_right[right_index] if right_index < len(ordered_right) else math.inf
        )
        current = min(left_value, right_value)
        distance += (current - previous) * abs(
            left_index / len(ordered_left) - right_index / len(ordered_right)
        )
        while left_index < len(ordered_left) and ordered_left[left_index] == current:
            left_index += 1
        while (
            right_index < len(ordered_right) and ordered_right[right_index] == current
        ):
            right_index += 1
        previous = current
    return distance


def _interval_estimate(
    predictive: tuple[float, ...],
    empirical: tuple[float, ...],
    level: float,
) -> IntervalEstimate:
    alpha = 1.0 - level
    lower = _quantile(predictive, alpha / 2.0)
    upper = _quantile(predictive, 1.0 - alpha / 2.0)
    scores = tuple(
        (upper - lower)
        + (2.0 / alpha) * (lower - value) * float(value < lower)
        + (2.0 / alpha) * (value - upper) * float(value > upper)
        for value in empirical
    )
    return IntervalEstimate(
        level=level,
        lower=lower,
        upper=upper,
        interval_score=fmean(scores),
        empirical_coverage=fmean(float(lower <= value <= upper) for value in empirical),
    )


def _mmd_squared(
    left: tuple[float, ...], right: tuple[float, ...], bandwidth: float
) -> float:
    scale = 2.0 * bandwidth * bandwidth

    def kernel(first: float, second: float) -> float:
        return math.exp(-((first - second) ** 2) / scale)

    left_term = fmean(kernel(first, second) for first in left for second in left)
    right_term = fmean(kernel(first, second) for first in right for second in right)
    cross_term = fmean(kernel(first, second) for first in left for second in right)
    return max(0.0, left_term + right_term - 2.0 * cross_term)


def evaluate_distribution_metrics(
    predictive_sample: DistributionSampleArtifact,
    held_out_sample: DistributionSampleArtifact,
    config: DistributionMetricConfig,
) -> DistributionMetrics:
    """Score a frozen predictive distribution against held-out empirical values."""
    if not isinstance(config, DistributionMetricConfig):
        _fail("distribution config has an invalid schema")
    if not isinstance(predictive_sample, DistributionSampleArtifact) or not isinstance(
        held_out_sample, DistributionSampleArtifact
    ):
        _fail("distribution scoring requires typed sample artifacts")
    if (
        predictive_sample.role is not DistributionSampleRole.PREDICTIVE
        or held_out_sample.role is not DistributionSampleRole.HELD_OUT_PREDECLARED
    ):
        _fail("distribution sample roles must be predictive and held_out_predeclared")
    if predictive_sample.study_protocol_sha256 != held_out_sample.study_protocol_sha256:
        _fail("distribution samples must share one frozen study protocol")
    if predictive_sample.frozen_at >= held_out_sample.available_at:
        _fail("predictive sample must be frozen before held-out outcomes are available")
    for sample in (predictive_sample, held_out_sample):
        if sample.estimand_id != config.estimand_id:
            _fail("distribution sample estimand does not match scoring config")
        if sample.sampling_unit != config.sampling_unit:
            _fail("distribution sample sampling unit does not match scoring config")
    predictive = predictive_sample.values
    empirical = held_out_sample.values
    if (
        len(predictive) < config.minimum_valid_draws
        or len(empirical) < config.minimum_valid_draws
    ):
        _fail("predictive and empirical samples do not meet minimum_valid_draws")
    intervals = tuple(
        _interval_estimate(predictive, empirical, level)
        for level in config.reported_interval_levels
    )
    primary = next(
        interval for interval in intervals if interval.level == config.interval_level
    )
    tail_probability = fmean(
        float(value <= config.tail_event_threshold) for value in predictive
    )
    brier = fmean(
        (tail_probability - float(value <= config.tail_event_threshold)) ** 2
        for value in empirical
    )
    crps = fmean(_crps(predictive, value) for value in empirical)
    predictive_mean = fmean(predictive)
    errors = tuple(predictive_mean - value for value in empirical)
    predicted_positive = _quantile(predictive, 0.5) > 0.0
    tail_count = max(1, math.ceil(len(predictive) * 0.05))
    return DistributionMetrics(
        quantiles=tuple(
            (level, _quantile(predictive, level)) for level in config.quantile_levels
        ),
        sign_probability=fmean(float(value > 0.0) for value in predictive),
        crps=crps,
        energy_score=crps,
        interval_lower=primary.lower,
        interval_upper=primary.upper,
        interval_score=primary.interval_score,
        empirical_coverage=primary.empirical_coverage,
        wasserstein_distance=_wasserstein(predictive, empirical),
        tail_event_probability=tail_probability,
        brier_score=brier,
        intervals=intervals,
        interval_kind=config.interval_kind,
        interval_method_id=config.interval_method_id,
        estimand_id=config.estimand_id,
        sampling_unit=config.sampling_unit,
        predictive_draw_count=len(predictive),
        empirical_draw_count=len(empirical),
        mean_absolute_error=fmean(abs(value) for value in errors),
        root_mean_squared_error=math.sqrt(fmean(value * value for value in errors)),
        direction_accuracy=fmean(
            float(predicted_positive == (value > 0.0)) for value in empirical
        ),
        worst_five_percent_mean=fmean(sorted(predictive)[:tail_count]),
        maximum_mean_discrepancy_squared=_mmd_squared(
            predictive, empirical, config.mmd_bandwidth
        ),
        predictive_sample_artifact_sha256=predictive_sample.sha256,
        held_out_sample_artifact_sha256=held_out_sample.sha256,
    )


class ValidityLevel(StrEnum):
    """Publication decision emitted by the validity certificate."""

    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"
    ABSTAIN = "abstain"


class ValidityCheckStatus(StrEnum):
    """Status of one pre-registered validity check."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    UNKNOWN = "unknown"


class ValidityEvidenceAuthority(StrEnum):
    """Authority boundary for the values retained by a certificate.

    The current public builder accepts caller-authored synthetic diagnostics
    only.  Hashing those values makes later alteration detectable, but does
    not prove that the underlying leakage, PPC, ensemble, seed, coverage,
    group, or mechanism audits ran.  Consequently this is deliberately the
    only constructible authority state until typed upstream receipts exist.
    """

    SELF_REPORTED_SYNTHETIC_FIXTURE = "self_reported_synthetic_fixture"


_VALIDITY_CHECK_IDS = (
    "temporal_leakage",
    "pre_policy_posterior_predictive",
    "ood_policy_distance",
    "model_disagreement",
    "seed_instability",
    "prompt_instability",
    "parameter_support_failure",
    "empirical_coverage_history",
    "unsupported_group_labels",
    "missing_required_mechanisms",
)


@dataclass(frozen=True, slots=True)
class ValidityThresholds:
    """Frozen thresholds; no target-event result can modify this object."""

    max_ppc_failure_green: float
    max_ppc_failure_yellow: float
    max_ood_distance_green: float
    max_ood_distance_yellow: float
    max_model_disagreement_green: float
    max_model_disagreement_yellow: float
    max_seed_instability_green: float
    max_seed_instability_yellow: float
    min_coverage_green: float
    min_coverage_yellow: float
    minimum_coverage_history: int
    supported_group_labels: tuple[str, ...]
    required_mechanisms: tuple[str, ...]
    max_prompt_instability_green: float = 0.10
    max_prompt_instability_yellow: float = 0.20
    max_parameter_support_failure_green: float = 0.05
    max_parameter_support_failure_yellow: float = 0.10

    def __post_init__(self) -> None:
        upper_pairs = (
            ("ppc_failure", self.max_ppc_failure_green, self.max_ppc_failure_yellow),
            ("ood_distance", self.max_ood_distance_green, self.max_ood_distance_yellow),
            (
                "model_disagreement",
                self.max_model_disagreement_green,
                self.max_model_disagreement_yellow,
            ),
            (
                "seed_instability",
                self.max_seed_instability_green,
                self.max_seed_instability_yellow,
            ),
            (
                "prompt_instability",
                self.max_prompt_instability_green,
                self.max_prompt_instability_yellow,
            ),
            (
                "parameter_support_failure",
                self.max_parameter_support_failure_green,
                self.max_parameter_support_failure_yellow,
            ),
        )
        for name, green, yellow in upper_pairs:
            if name in {"ppc_failure", "parameter_support_failure"}:
                green_value = _probability(green, f"max_{name}_green")
                yellow_value = _probability(yellow, f"max_{name}_yellow")
            else:
                green_value = _positive(green, f"max_{name}_green", allow_zero=True)
                yellow_value = _positive(yellow, f"max_{name}_yellow", allow_zero=True)
            if green_value > yellow_value:
                _fail(f"{name} green threshold must not exceed yellow threshold")
            object.__setattr__(self, f"max_{name}_green", green_value)
            object.__setattr__(self, f"max_{name}_yellow", yellow_value)
        coverage_green = _probability(self.min_coverage_green, "min_coverage_green")
        coverage_yellow = _probability(self.min_coverage_yellow, "min_coverage_yellow")
        if coverage_green < coverage_yellow:
            _fail("coverage green threshold must not be below yellow threshold")
        object.__setattr__(self, "min_coverage_green", coverage_green)
        object.__setattr__(self, "min_coverage_yellow", coverage_yellow)
        if (
            isinstance(self.minimum_coverage_history, bool)
            or not isinstance(self.minimum_coverage_history, int)
            or self.minimum_coverage_history < 1
        ):
            _fail("minimum_coverage_history must be a positive integer")
        for field in ("supported_group_labels", "required_mechanisms"):
            values = getattr(self, field)
            if not isinstance(values, tuple) or not values:
                _fail(f"{field} must be a non-empty immutable tuple")
            normalized_identifiers = tuple(
                _identifier(value, f"{field} item") for value in values
            )
            if len(set(normalized_identifiers)) != len(normalized_identifiers):
                _fail(f"{field} must not contain duplicates")
            object.__setattr__(self, field, normalized_identifiers)

    @property
    def sha256(self) -> str:
        """Return the immutable threshold commitment."""
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def to_mapping(self) -> dict[str, object]:
        return {
            "max_model_disagreement_green": self.max_model_disagreement_green,
            "max_model_disagreement_yellow": self.max_model_disagreement_yellow,
            "max_ood_distance_green": self.max_ood_distance_green,
            "max_ood_distance_yellow": self.max_ood_distance_yellow,
            "max_ppc_failure_green": self.max_ppc_failure_green,
            "max_ppc_failure_yellow": self.max_ppc_failure_yellow,
            "max_prompt_instability_green": self.max_prompt_instability_green,
            "max_prompt_instability_yellow": self.max_prompt_instability_yellow,
            "max_parameter_support_failure_green": (
                self.max_parameter_support_failure_green
            ),
            "max_parameter_support_failure_yellow": (
                self.max_parameter_support_failure_yellow
            ),
            "max_seed_instability_green": self.max_seed_instability_green,
            "max_seed_instability_yellow": self.max_seed_instability_yellow,
            "min_coverage_green": self.min_coverage_green,
            "min_coverage_yellow": self.min_coverage_yellow,
            "minimum_coverage_history": self.minimum_coverage_history,
            "required_mechanisms": list(self.required_mechanisms),
            "supported_group_labels": list(self.supported_group_labels),
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> ValidityThresholds:
        """Parse the exact frozen threshold schema without defaults."""
        expected = frozenset(
            {
                "max_model_disagreement_green",
                "max_model_disagreement_yellow",
                "max_ood_distance_green",
                "max_ood_distance_yellow",
                "max_parameter_support_failure_green",
                "max_parameter_support_failure_yellow",
                "max_ppc_failure_green",
                "max_ppc_failure_yellow",
                "max_prompt_instability_green",
                "max_prompt_instability_yellow",
                "max_seed_instability_green",
                "max_seed_instability_yellow",
                "min_coverage_green",
                "min_coverage_yellow",
                "minimum_coverage_history",
                "required_mechanisms",
                "supported_group_labels",
            }
        )
        _mapping_fields(mapping, expected, "validity thresholds")

        def number(field: str) -> float:
            return _finite(mapping[field], f"validity thresholds {field}")

        def identifiers(field: str) -> tuple[str, ...]:
            raw = mapping[field]
            if not isinstance(raw, list):
                _fail(f"validity thresholds {field} must be an array")
            return tuple(
                _identifier(item, f"validity thresholds {field} item") for item in raw
            )

        history = mapping["minimum_coverage_history"]
        if isinstance(history, bool) or not isinstance(history, int):
            _fail("validity thresholds minimum_coverage_history must be an integer")
        return cls(
            max_ppc_failure_green=number("max_ppc_failure_green"),
            max_ppc_failure_yellow=number("max_ppc_failure_yellow"),
            max_ood_distance_green=number("max_ood_distance_green"),
            max_ood_distance_yellow=number("max_ood_distance_yellow"),
            max_model_disagreement_green=number("max_model_disagreement_green"),
            max_model_disagreement_yellow=number("max_model_disagreement_yellow"),
            max_seed_instability_green=number("max_seed_instability_green"),
            max_seed_instability_yellow=number("max_seed_instability_yellow"),
            min_coverage_green=number("min_coverage_green"),
            min_coverage_yellow=number("min_coverage_yellow"),
            minimum_coverage_history=history,
            supported_group_labels=identifiers("supported_group_labels"),
            required_mechanisms=identifiers("required_mechanisms"),
            max_prompt_instability_green=number("max_prompt_instability_green"),
            max_prompt_instability_yellow=number("max_prompt_instability_yellow"),
            max_parameter_support_failure_green=number(
                "max_parameter_support_failure_green"
            ),
            max_parameter_support_failure_yellow=number(
                "max_parameter_support_failure_yellow"
            ),
        )


@dataclass(frozen=True, slots=True)
class ValidityEvidence:
    """Diagnostic values supplied independently of certificate thresholds.

    This value object carries no execution authority.  The current builder
    therefore labels it as a self-reported synthetic fixture regardless of how
    favorable the values are.
    """

    temporal_leakage_detected: bool | None
    pre_policy_ppc_failure_rate: float | None
    ood_policy_distance: float | None
    model_disagreement: float | None
    seed_instability: float | None
    empirical_coverage_history: tuple[float, ...] | None
    observed_group_labels: tuple[str, ...] | None
    active_mechanisms: tuple[str, ...] | None
    prompt_instability: float | None = None
    parameter_support_failure_rate: float | None = None

    def __post_init__(self) -> None:
        if self.temporal_leakage_detected is not None and not isinstance(
            self.temporal_leakage_detected, bool
        ):
            _fail("temporal_leakage_detected must be boolean or null")
        numeric_fields = (
            "pre_policy_ppc_failure_rate",
            "ood_policy_distance",
            "model_disagreement",
            "seed_instability",
            "prompt_instability",
            "parameter_support_failure_rate",
        )
        for field in numeric_fields:
            value = getattr(self, field)
            if value is not None:
                normalized = _positive(value, field, allow_zero=True)
                if field in {
                    "pre_policy_ppc_failure_rate",
                    "parameter_support_failure_rate",
                }:
                    normalized = _probability(normalized, field)
                object.__setattr__(self, field, normalized)
        if self.empirical_coverage_history is not None:
            if not isinstance(self.empirical_coverage_history, tuple):
                _fail("empirical_coverage_history must be an immutable tuple or null")
            object.__setattr__(
                self,
                "empirical_coverage_history",
                tuple(
                    _probability(value, "coverage history item")
                    for value in self.empirical_coverage_history
                ),
            )
        for field in ("observed_group_labels", "active_mechanisms"):
            value = getattr(self, field)
            if value is not None:
                if not isinstance(value, tuple):
                    _fail(f"{field} must be an immutable tuple or null")
                normalized_identifiers = tuple(
                    _identifier(item, f"{field} item") for item in value
                )
                if len(set(normalized_identifiers)) != len(normalized_identifiers):
                    _fail(f"{field} must not contain duplicates")
                object.__setattr__(self, field, normalized_identifiers)

    @property
    def sha256(self) -> str:
        """Return the evidence commitment independent of thresholds."""
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def to_mapping(self) -> dict[str, object]:
        return {
            "active_mechanisms": (
                list(self.active_mechanisms)
                if self.active_mechanisms is not None
                else None
            ),
            "empirical_coverage_history": (
                list(self.empirical_coverage_history)
                if self.empirical_coverage_history is not None
                else None
            ),
            "model_disagreement": self.model_disagreement,
            "observed_group_labels": (
                list(self.observed_group_labels)
                if self.observed_group_labels is not None
                else None
            ),
            "ood_policy_distance": self.ood_policy_distance,
            "parameter_support_failure_rate": self.parameter_support_failure_rate,
            "pre_policy_ppc_failure_rate": self.pre_policy_ppc_failure_rate,
            "prompt_instability": self.prompt_instability,
            "seed_instability": self.seed_instability,
            "temporal_leakage_detected": self.temporal_leakage_detected,
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> ValidityEvidence:
        """Parse complete synthetic evidence retained by a certificate."""
        expected = frozenset(
            {
                "active_mechanisms",
                "empirical_coverage_history",
                "model_disagreement",
                "observed_group_labels",
                "ood_policy_distance",
                "parameter_support_failure_rate",
                "pre_policy_ppc_failure_rate",
                "prompt_instability",
                "seed_instability",
                "temporal_leakage_detected",
            }
        )
        _mapping_fields(mapping, expected, "validity evidence")

        temporal = mapping["temporal_leakage_detected"]
        if temporal is not None and not isinstance(temporal, bool):
            _fail("validity evidence temporal_leakage_detected must be boolean or null")

        def optional_number(field: str) -> float | None:
            value = mapping[field]
            return (
                None if value is None else _finite(value, f"validity evidence {field}")
            )

        def optional_numbers(field: str) -> tuple[float, ...] | None:
            value = mapping[field]
            if value is None:
                return None
            if not isinstance(value, list):
                _fail(f"validity evidence {field} must be an array or null")
            return tuple(
                _finite(item, f"validity evidence {field} item") for item in value
            )

        def optional_identifiers(field: str) -> tuple[str, ...] | None:
            value = mapping[field]
            if value is None:
                return None
            if not isinstance(value, list):
                _fail(f"validity evidence {field} must be an array or null")
            return tuple(
                _identifier(item, f"validity evidence {field} item") for item in value
            )

        return cls(
            temporal_leakage_detected=temporal,
            pre_policy_ppc_failure_rate=optional_number("pre_policy_ppc_failure_rate"),
            ood_policy_distance=optional_number("ood_policy_distance"),
            model_disagreement=optional_number("model_disagreement"),
            seed_instability=optional_number("seed_instability"),
            empirical_coverage_history=optional_numbers("empirical_coverage_history"),
            observed_group_labels=optional_identifiers("observed_group_labels"),
            active_mechanisms=optional_identifiers("active_mechanisms"),
            prompt_instability=optional_number("prompt_instability"),
            parameter_support_failure_rate=optional_number(
                "parameter_support_failure_rate"
            ),
        )


@dataclass(frozen=True, slots=True)
class ValidityCheck:
    """One named check, observed value, and frozen-threshold outcome."""

    check_id: str
    status: ValidityCheckStatus
    observed: object
    reason: str

    def __post_init__(self) -> None:
        _identifier(self.check_id, "validity check_id")
        if not isinstance(self.status, ValidityCheckStatus):
            _fail("validity check status is invalid")
        if not isinstance(self.reason, str) or not self.reason.strip():
            _fail("validity check reason must be non-empty")

    def to_mapping(self) -> dict[str, object]:
        return {
            "check_id": self.check_id,
            "observed": self.observed,
            "reason": self.reason,
            "status": self.status.value,
        }


def _derive_validity_level(checks: tuple[ValidityCheck, ...]) -> ValidityLevel:
    temporal = next(
        (check for check in checks if check.check_id == "temporal_leakage"), None
    )
    if temporal is not None and temporal.status in {
        ValidityCheckStatus.FAIL,
        ValidityCheckStatus.UNKNOWN,
    }:
        return ValidityLevel.ABSTAIN
    statuses = {check.status for check in checks}
    if ValidityCheckStatus.FAIL in statuses:
        return ValidityLevel.RED
    if ValidityCheckStatus.UNKNOWN in statuses:
        return ValidityLevel.ABSTAIN
    if ValidityCheckStatus.WARN in statuses:
        return ValidityLevel.YELLOW
    return ValidityLevel.GREEN


@dataclass(frozen=True, slots=True)
class ValidityCertificate:
    """Complete certificate with all required checks and immutable commitments."""

    schema_version: int
    level: ValidityLevel
    diagnostic_level: ValidityLevel
    evidence_authority: ValidityEvidenceAuthority
    thresholds: ValidityThresholds
    evidence: ValidityEvidence
    thresholds_sha256: str
    evidence_sha256: str
    checks: tuple[ValidityCheck, ...]
    synthetic_only: bool

    def __post_init__(self) -> None:
        if self.schema_version != 3:
            _fail("validity certificate schema_version must be 3")
        if not isinstance(self.level, ValidityLevel) or not isinstance(
            self.diagnostic_level, ValidityLevel
        ):
            _fail("validity certificate level is invalid")
        if (
            self.evidence_authority
            is not ValidityEvidenceAuthority.SELF_REPORTED_SYNTHETIC_FIXTURE
        ):
            _fail(
                "validity evidence requires typed upstream verifier receipts; "
                "caller-authored evidence has synthetic-fixture authority only"
            )
        if not isinstance(self.thresholds, ValidityThresholds) or not isinstance(
            self.evidence, ValidityEvidence
        ):
            _fail("validity certificate must retain typed thresholds and evidence")
        if not isinstance(self.checks, tuple):
            _fail("validity certificate checks must be an immutable tuple")
        if not all(isinstance(check, ValidityCheck) for check in self.checks):
            _fail("validity certificate checks have an invalid schema")
        check_ids = tuple(check.check_id for check in self.checks)
        if check_ids != _VALIDITY_CHECK_IDS:
            _fail(
                "validity certificate must contain the exact registered check IDs "
                "in canonical order"
            )
        expected_checks = _derive_validity_checks(self.evidence, self.thresholds)
        if self.checks != expected_checks:
            _fail("validity certificate checks do not derive from retained evidence")
        expected_diagnostic_level = _derive_validity_level(expected_checks)
        if self.diagnostic_level is not expected_diagnostic_level:
            _fail("validity diagnostic level disagrees with its check statuses")
        if self.level is not ValidityLevel.ABSTAIN:
            _fail(
                "self-reported synthetic validity evidence must abstain; "
                "favorable caller scalars cannot authorize GREEN"
            )
        if self.thresholds_sha256 != self.thresholds.sha256:
            _fail("validity certificate threshold hash does not match retained values")
        if self.evidence_sha256 != self.evidence.sha256:
            _fail("validity certificate evidence hash does not match retained values")
        if self.synthetic_only is not True:
            _fail(
                "non-synthetic validity certification requires typed verifier receipts"
            )

    @property
    def sha256(self) -> str:
        """Return the certificate commitment."""
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def to_mapping(self) -> dict[str, object]:
        return {
            "checks": [check.to_mapping() for check in self.checks],
            "diagnostic_level": self.diagnostic_level.value,
            "evidence": self.evidence.to_mapping(),
            "evidence_authority": self.evidence_authority.value,
            "evidence_sha256": self.evidence_sha256,
            "level": self.level.value,
            "schema_version": self.schema_version,
            "synthetic_only": self.synthetic_only,
            "thresholds": self.thresholds.to_mapping(),
            "thresholds_sha256": self.thresholds_sha256,
        }


def _upper_check(
    check_id: str, value: float | None, green: float, yellow: float
) -> ValidityCheck:
    if value is None:
        return ValidityCheck(
            check_id, ValidityCheckStatus.UNKNOWN, None, "required evidence is missing"
        )
    if value <= green:
        return ValidityCheck(
            check_id, ValidityCheckStatus.PASS, value, "within green threshold"
        )
    if value <= yellow:
        return ValidityCheck(
            check_id, ValidityCheckStatus.WARN, value, "within yellow threshold"
        )
    return ValidityCheck(
        check_id, ValidityCheckStatus.FAIL, value, "exceeds yellow threshold"
    )


def _derive_validity_checks(
    evidence: ValidityEvidence,
    thresholds: ValidityThresholds,
) -> tuple[ValidityCheck, ...]:
    """Derive the canonical ten checks from retained evidence and thresholds."""
    if evidence.temporal_leakage_detected is None:
        temporal = ValidityCheck(
            "temporal_leakage",
            ValidityCheckStatus.UNKNOWN,
            None,
            "temporal leakage audit is missing",
        )
    elif evidence.temporal_leakage_detected:
        temporal = ValidityCheck(
            "temporal_leakage",
            ValidityCheckStatus.FAIL,
            True,
            "temporal leakage was detected",
        )
    else:
        temporal = ValidityCheck(
            "temporal_leakage", ValidityCheckStatus.PASS, False, "no leakage detected"
        )
    ppc = _upper_check(
        "pre_policy_posterior_predictive",
        evidence.pre_policy_ppc_failure_rate,
        thresholds.max_ppc_failure_green,
        thresholds.max_ppc_failure_yellow,
    )
    ood = _upper_check(
        "ood_policy_distance",
        evidence.ood_policy_distance,
        thresholds.max_ood_distance_green,
        thresholds.max_ood_distance_yellow,
    )
    disagreement = _upper_check(
        "model_disagreement",
        evidence.model_disagreement,
        thresholds.max_model_disagreement_green,
        thresholds.max_model_disagreement_yellow,
    )
    seeds = _upper_check(
        "seed_instability",
        evidence.seed_instability,
        thresholds.max_seed_instability_green,
        thresholds.max_seed_instability_yellow,
    )
    prompts = _upper_check(
        "prompt_instability",
        evidence.prompt_instability,
        thresholds.max_prompt_instability_green,
        thresholds.max_prompt_instability_yellow,
    )
    parameter_support = _upper_check(
        "parameter_support_failure",
        evidence.parameter_support_failure_rate,
        thresholds.max_parameter_support_failure_green,
        thresholds.max_parameter_support_failure_yellow,
    )
    history = evidence.empirical_coverage_history
    if history is None or len(history) < thresholds.minimum_coverage_history:
        coverage = ValidityCheck(
            "empirical_coverage_history",
            ValidityCheckStatus.UNKNOWN,
            list(history) if history is not None else None,
            "coverage history is missing or too short",
        )
    else:
        observed_coverage = fmean(history)
        if observed_coverage >= thresholds.min_coverage_green:
            coverage_status = ValidityCheckStatus.PASS
            coverage_reason = "coverage meets green threshold"
        elif observed_coverage >= thresholds.min_coverage_yellow:
            coverage_status = ValidityCheckStatus.WARN
            coverage_reason = "coverage meets only yellow threshold"
        else:
            coverage_status = ValidityCheckStatus.FAIL
            coverage_reason = "coverage is below yellow threshold"
        coverage = ValidityCheck(
            "empirical_coverage_history",
            coverage_status,
            observed_coverage,
            coverage_reason,
        )
    if evidence.observed_group_labels is None:
        groups = ValidityCheck(
            "unsupported_group_labels",
            ValidityCheckStatus.UNKNOWN,
            None,
            "group-label audit is missing",
        )
    else:
        unsupported = tuple(
            sorted(
                set(evidence.observed_group_labels)
                - set(thresholds.supported_group_labels)
            )
        )
        groups = ValidityCheck(
            "unsupported_group_labels",
            ValidityCheckStatus.FAIL if unsupported else ValidityCheckStatus.PASS,
            list(unsupported),
            "unsupported group labels detected"
            if unsupported
            else "all group labels supported",
        )
    if evidence.active_mechanisms is None:
        mechanisms = ValidityCheck(
            "missing_required_mechanisms",
            ValidityCheckStatus.UNKNOWN,
            None,
            "mechanism audit is missing",
        )
    else:
        missing = tuple(
            sorted(
                set(thresholds.required_mechanisms) - set(evidence.active_mechanisms)
            )
        )
        mechanisms = ValidityCheck(
            "missing_required_mechanisms",
            ValidityCheckStatus.FAIL if missing else ValidityCheckStatus.PASS,
            list(missing),
            "required mechanisms are missing"
            if missing
            else "all required mechanisms active",
        )
    checks = (
        temporal,
        ppc,
        ood,
        disagreement,
        seeds,
        prompts,
        parameter_support,
        coverage,
        groups,
        mechanisms,
    )
    return checks


def build_validity_certificate(
    evidence: ValidityEvidence,
    thresholds: ValidityThresholds,
    *,
    synthetic_only: bool,
) -> ValidityCertificate:
    """Apply only pre-frozen thresholds and retain all validity inputs."""
    if not isinstance(evidence, ValidityEvidence) or not isinstance(
        thresholds, ValidityThresholds
    ):
        _fail("validity certificate requires validated evidence and thresholds")
    if synthetic_only is not True:
        _fail(
            "non-synthetic validity certification requires typed verifier "
            "receipts; caller-supplied ValidityEvidence cannot authorize it"
        )
    checks = _derive_validity_checks(evidence, thresholds)
    return ValidityCertificate(
        schema_version=3,
        level=ValidityLevel.ABSTAIN,
        diagnostic_level=_derive_validity_level(checks),
        evidence_authority=(ValidityEvidenceAuthority.SELF_REPORTED_SYNTHETIC_FIXTURE),
        thresholds=thresholds,
        evidence=evidence,
        thresholds_sha256=thresholds.sha256,
        evidence_sha256=evidence.sha256,
        checks=checks,
        synthetic_only=synthetic_only,
    )


__all__ = [
    "AgentOutcome",
    "DistributionMetricConfig",
    "DistributionMetrics",
    "DistributionSampleArtifact",
    "DistributionSampleRole",
    "GroupMetrics",
    "InformationResponse",
    "IntervalEstimate",
    "IntervalKind",
    "MarketEvaluationInput",
    "MarketMetricConfig",
    "MarketMetrics",
    "OrderOutcome",
    "PairedRankingMetrics",
    "PairedRankingObservation",
    "PairedRankingSampleArtifact",
    "PolicyEvaluationError",
    "PolicyEvaluationSchemaError",
    "PricePath",
    "QuoteObservation",
    "ScenarioAggregation",
    "ScenarioMemberCommitment",
    "ScenarioUniverseManifest",
    "ScenarioUniverseRole",
    "TradeObservation",
    "TradeSide",
    "ValidityCertificate",
    "ValidityCheck",
    "ValidityCheckStatus",
    "ValidityEvidenceAuthority",
    "ValidityEvidence",
    "ValidityLevel",
    "ValidityThresholds",
    "build_validity_certificate",
    "evaluate_distribution_metrics",
    "evaluate_group_metrics",
    "evaluate_market_metrics",
    "evaluate_paired_ranking_metrics",
    "ranking_identity_manifest_sha256",
]
