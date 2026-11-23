"""Verifier-owned exact outcomes derived from retained integrated CLOB ledgers.

This module deliberately accepts only the protocol-bound synthetic run artifact.
It independently replays every registered command prefix and reconstructs cash,
asset, and venue-fee settlement from the retained trade ledger.  The resulting
metrics are synthetic mechanism evidence, not welfare or real-world causal
evidence.
"""

from __future__ import annotations

import os
import re
import stat
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from math import gcd
from pathlib import Path
from typing import Literal, NoReturn, Self, cast

from policytwin.counterfactual.integrated_market import (
    IntegratedHorizonReplayBoundary,
    IntegratedSyntheticRunArtifact,
)
from policytwin.counterfactual.paired import GroupSpec, WorldRole
from policytwin.experiments.integrated_synthetic import (
    ProtocolBoundIntegratedSyntheticRunArtifact,
    require_ledger_derivation_source_artifact,
)
from policytwin.implementation_fingerprint import (
    ImplementationFingerprintError,
    live_policytwin_implementation_sha256,
)
from policytwin.market import (
    CancelOrder,
    CentralLimitOrderBook,
    LimitOrder,
    MarketOrder,
    MarketSnapshot,
    OpeningState,
    OrderStatus,
    ReplayResult,
    Side,
    TradeLogEntry,
)
from policytwin.market.models import FEE_DENOMINATOR_BPS
from policytwin.protocol import CohortBasis
from policytwin.provenance import (
    canonical_json_bytes,
    read_json_bytes_object,
    sha256_bytes,
)

LEDGER_DERIVED_SCHEMA_VERSION = "policytwin.ledger-derived-evaluation.v2"
LEDGER_DERIVED_CLAIM_SCOPE = "synthetic_protocol_bound_ledger_evidence_only"
LEDGER_METRIC_REGISTRY_VERSION = "policytwin.ledger-metric-registry.v2"
LEDGER_EVIDENCE_VERIFIER_IMPLEMENTATION_ID = "policytwin.ledger-evidence-verifier.v2"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_ARTIFACT_BYTES = 256 * 1024 * 1024


class LedgerDerivedEvidenceError(RuntimeError):
    """Raised when exact ledger evidence cannot be independently established."""


def _fail(message: str) -> NoReturn:
    raise LedgerDerivedEvidenceError(message)


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{field} must be an integer")
    return value


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 256:
        _fail(f"{field} must be bounded non-empty text")
    return value


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail(f"{field} must be a lowercase SHA-256 digest")
    return value


def _bounded_text(value: object, field: str, *, limit: int = 2048) -> str:
    if not isinstance(value, str) or not value or len(value) > limit:
        _fail(f"{field} must be bounded non-empty text")
    return value


@dataclass(frozen=True, slots=True)
class ExactRational:
    """Canonical exact rational used instead of floating-point outcome values."""

    numerator: int
    denominator: int = 1

    def __post_init__(self) -> None:
        numerator = _integer(self.numerator, "numerator")
        denominator = _integer(self.denominator, "denominator")
        if denominator <= 0:
            _fail("exact-rational denominator must be positive")
        divisor = gcd(numerator, denominator)
        if divisor != 1:
            _fail("exact rational must be in canonical reduced form")

    @classmethod
    def from_ratio(cls, numerator: int, denominator: int = 1) -> ExactRational:
        """Reduce one integer ratio to its unique canonical representation."""
        numerator = _integer(numerator, "numerator")
        denominator = _integer(denominator, "denominator")
        if denominator == 0:
            _fail("cannot construct an exact rational with denominator zero")
        if denominator < 0:
            numerator = -numerator
            denominator = -denominator
        divisor = gcd(numerator, denominator)
        return cls(numerator // divisor, denominator // divisor)

    def plus(self, other: ExactRational) -> ExactRational:
        """Return the exact sum."""
        if not isinstance(other, ExactRational):
            raise TypeError("other must be an ExactRational")
        return ExactRational.from_ratio(
            self.numerator * other.denominator + other.numerator * self.denominator,
            self.denominator * other.denominator,
        )

    def minus(self, other: ExactRational) -> ExactRational:
        """Return the exact difference."""
        if not isinstance(other, ExactRational):
            raise TypeError("other must be an ExactRational")
        return ExactRational.from_ratio(
            self.numerator * other.denominator - other.numerator * self.denominator,
            self.denominator * other.denominator,
        )

    def times_integer(self, multiplier: int) -> ExactRational:
        """Return the exact product with an integer."""
        return ExactRational.from_ratio(
            self.numerator * _integer(multiplier, "multiplier"), self.denominator
        )

    def to_mapping(self) -> dict[str, int]:
        return {"denominator": self.denominator, "numerator": self.numerator}


class ExactMetricState(StrEnum):
    """Closed vocabulary for defined and scientifically undefined values."""

    DEFINED = "defined"
    UNDEFINED_ZERO_DENOMINATOR = "undefined_zero_denominator"
    UNDEFINED_ONE_SIDED_BBO = "undefined_one_sided_bbo"
    UNDEFINED_EMPTY_BBO = "undefined_empty_bbo"
    UNDEFINED_OPENING_BBO = "undefined_opening_bbo"
    UNDEFINED_WORLD_METRIC = "undefined_world_metric"


@dataclass(frozen=True, slots=True)
class ExactMetric:
    """One exact value or an explicit non-numeric state."""

    state: ExactMetricState
    value: ExactRational | None

    def __post_init__(self) -> None:
        if not isinstance(self.state, ExactMetricState):
            _fail("metric state has the wrong type")
        if self.state is ExactMetricState.DEFINED:
            if not isinstance(self.value, ExactRational):
                _fail("defined exact metric requires an ExactRational")
        elif self.value is not None:
            _fail("undefined exact metric cannot carry a numeric value")

    @classmethod
    def integer(cls, value: int) -> ExactMetric:
        """Build a defined integer metric."""
        return cls(ExactMetricState.DEFINED, ExactRational.from_ratio(value))

    @classmethod
    def rational(cls, numerator: int, denominator: int) -> ExactMetric:
        """Build a defined rational metric."""
        return cls(
            ExactMetricState.DEFINED,
            ExactRational.from_ratio(numerator, denominator),
        )

    @classmethod
    def undefined(cls, state: ExactMetricState) -> ExactMetric:
        """Build an explicit undefined metric."""
        if state is ExactMetricState.DEFINED:
            _fail("undefined metric cannot use the defined state")
        return cls(state, None)

    def to_mapping(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "value": None if self.value is None else self.value.to_mapping(),
        }


class LedgerMetricScope(StrEnum):
    """Closed scope vocabulary for verifier-derived ledger diagnostics."""

    GROUP = "group"
    MARKET = "market"


@dataclass(frozen=True, slots=True)
class LedgerMetricDefinition:
    """Immutable code-owned definition for one exact ledger diagnostic.

    These definitions are deliberately distinct from the study protocol's
    registered metric definitions.  They are verifier-derived synthetic
    diagnostics and are not retroactively described as preregistered outputs.
    """

    metric_id: str
    scope: LedgerMetricScope
    formula: str
    unit: str
    observation_clock: str
    aggregation: str
    implementation_id: str
    implementation_source_sha256: str
    allowed_undefined_states: tuple[ExactMetricState, ...] = ()
    externally_registered: Literal[False] = False
    registry_version: str = LEDGER_METRIC_REGISTRY_VERSION

    def __post_init__(self) -> None:
        _identifier(self.metric_id, "metric_id")
        if not isinstance(self.scope, LedgerMetricScope):
            _fail("ledger metric definition scope has the wrong type")
        for field in ("formula", "unit", "observation_clock", "aggregation"):
            _bounded_text(getattr(self, field), field)
        _identifier(self.implementation_id, "implementation_id")
        _sha256(
            self.implementation_source_sha256,
            "implementation_source_sha256",
        )
        if (
            not isinstance(self.allowed_undefined_states, tuple)
            or not all(
                isinstance(item, ExactMetricState)
                and item is not ExactMetricState.DEFINED
                for item in self.allowed_undefined_states
            )
            or len(set(self.allowed_undefined_states))
            != len(self.allowed_undefined_states)
        ):
            _fail("allowed undefined metric states must be a unique closed tuple")
        if self.externally_registered is not False:
            _fail("ledger-derived metric definitions are not externally registered")
        if self.registry_version != LEDGER_METRIC_REGISTRY_VERSION:
            _fail("ledger metric registry version mismatch")

    def to_mapping(self) -> dict[str, object]:
        return {
            "aggregation": self.aggregation,
            "allowed_undefined_states": [
                item.value for item in self.allowed_undefined_states
            ],
            "externally_registered": self.externally_registered,
            "formula": self.formula,
            "implementation_id": self.implementation_id,
            "implementation_source_sha256": self.implementation_source_sha256,
            "metric_id": self.metric_id,
            "observation_clock": self.observation_clock,
            "registry_version": self.registry_version,
            "scope": self.scope.value,
            "unit": self.unit,
        }

    @property
    def definition_sha256(self) -> str:
        """Commit to the complete code-owned metric definition."""
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))


def _definition(
    metric_id: str,
    scope: LedgerMetricScope,
    formula: str,
    unit: str,
    aggregation: str,
    *allowed_undefined_states: ExactMetricState,
) -> LedgerMetricDefinition:
    implementation_id, implementation_source_sha256 = _metric_implementation_commitment(
        scope
    )
    return LedgerMetricDefinition(
        metric_id=metric_id,
        scope=scope,
        formula=formula,
        unit=unit,
        observation_clock="exact registered horizon command-prefix close",
        aggregation=aggregation,
        implementation_id=implementation_id,
        implementation_source_sha256=implementation_source_sha256,
        allowed_undefined_states=allowed_undefined_states,
    )


def _ledger_live_callable_bundle_sha256() -> str:
    """Hash the module bytes and every bound PolicyTwin function and method.

    A whole-module live bundle avoids the transitive-helper omissions possible
    with a hand-maintained allowlist.  In particular, runtime drift in helpers
    such as fee settlement or exact subtraction changes this commitment before
    a protocol-first run may begin.
    """
    try:
        return live_policytwin_implementation_sha256(
            module_path=Path(__file__), namespace=globals()
        )
    except ImplementationFingerprintError as error:
        raise LedgerDerivedEvidenceError(
            "cannot capture the complete live ledger verifier bundle"
        ) from error


def _metric_implementation_commitment(
    scope: LedgerMetricScope,
) -> tuple[str, str]:
    if scope is LedgerMetricScope.GROUP:
        implementation_id = "policytwin.ledger-group-outcome.v1"
    elif scope is LedgerMetricScope.MARKET:
        implementation_id = "policytwin.ledger-market-outcome.v1"
    else:
        _fail("unsupported ledger metric implementation scope")
    return implementation_id, sha256_bytes(
        canonical_json_bytes(
            {
                "live_verifier_bundle_sha256": (_ledger_live_callable_bundle_sha256()),
                "scope": scope.value,
            }
        )
    )


def _build_ledger_metric_registry() -> tuple[LedgerMetricDefinition, ...]:
    return tuple(
        sorted(
            (
                _definition(
                    "ledger.group.cash_level",
                    LedgerMetricScope.GROUP,
                    "sum of closing gross cash over frozen cohort members",
                    "integer cash units",
                    "sum over frozen cohort members",
                ),
                _definition(
                    "ledger.group.conditional_fill_rate",
                    LedgerMetricScope.GROUP,
                    "filled quantity attributed to cohort post-opening submitted "
                    "order IDs divided by cohort post-opening submitted quantity",
                    "exact ratio",
                    "cohort order-ID attribution; a within-cohort trade contributes "
                    "once per cohort-owned side",
                    ExactMetricState.UNDEFINED_ZERO_DENOMINATOR,
                ),
                _definition(
                    "ledger.group.gross_filled_quantity",
                    LedgerMetricScope.GROUP,
                    "sum trade quantity once for each buyer or seller belonging to "
                    "the frozen cohort",
                    "integer asset units",
                    "sum over retained trade sides",
                ),
                _definition(
                    "ledger.group.inventory_level",
                    LedgerMetricScope.GROUP,
                    "sum of closing gross asset balances over frozen cohort members",
                    "integer asset units",
                    "sum over frozen cohort members",
                ),
                _definition(
                    "ledger.group.midpoint_pnl",
                    LedgerMetricScope.GROUP,
                    "closing cohort midpoint wealth minus opening cohort midpoint "
                    "wealth",
                    "exact cash-equivalent units",
                    "cohort cash plus inventory times contemporaneous two-sided "
                    "BBO midpoint",
                    ExactMetricState.UNDEFINED_OPENING_BBO,
                    ExactMetricState.UNDEFINED_ONE_SIDED_BBO,
                    ExactMetricState.UNDEFINED_EMPTY_BBO,
                ),
                _definition(
                    "ledger.group.midpoint_wealth",
                    LedgerMetricScope.GROUP,
                    "closing cohort cash plus closing inventory times closing "
                    "two-sided BBO midpoint",
                    "exact cash-equivalent units",
                    "sum balances then mark inventory at exact BBO midpoint",
                    ExactMetricState.UNDEFINED_ONE_SIDED_BBO,
                    ExactMetricState.UNDEFINED_EMPTY_BBO,
                ),
                _definition(
                    "ledger.group.submitted_filled_quantity",
                    LedgerMetricScope.GROUP,
                    "sum trade quantity once for each maker or taker order ID "
                    "submitted post-opening by the frozen cohort",
                    "integer asset units",
                    "cohort post-opening order-ID attribution",
                ),
                _definition(
                    "ledger.group.submitted_order_count",
                    LedgerMetricScope.GROUP,
                    "count of post-opening limit and market commands submitted by "
                    "frozen cohort members",
                    "integer orders",
                    "count over registered command prefix",
                ),
                _definition(
                    "ledger.group.submitted_order_quantity",
                    LedgerMetricScope.GROUP,
                    "sum quantity of post-opening limit and market commands submitted "
                    "by frozen cohort members",
                    "integer asset units",
                    "sum over registered command prefix",
                ),
                _definition(
                    "ledger.group.trading_fees_paid",
                    LedgerMetricScope.GROUP,
                    "sum exact buyer and seller fees paid by frozen cohort members",
                    "integer cash units",
                    "sum independently recomputed settlement fee legs",
                ),
                _definition(
                    "ledger.market.best_ask_depth",
                    LedgerMetricScope.MARKET,
                    "sum remaining quantity at the lowest closing sell price",
                    "integer asset units",
                    "closing active order book",
                    ExactMetricState.UNDEFINED_ONE_SIDED_BBO,
                    ExactMetricState.UNDEFINED_EMPTY_BBO,
                ),
                _definition(
                    "ledger.market.best_ask_price",
                    LedgerMetricScope.MARKET,
                    "lowest closing active sell limit price",
                    "integer price ticks in configured cash units",
                    "closing active order book",
                    ExactMetricState.UNDEFINED_ONE_SIDED_BBO,
                    ExactMetricState.UNDEFINED_EMPTY_BBO,
                ),
                _definition(
                    "ledger.market.best_bid_depth",
                    LedgerMetricScope.MARKET,
                    "sum remaining quantity at the highest closing buy price",
                    "integer asset units",
                    "closing active order book",
                    ExactMetricState.UNDEFINED_ONE_SIDED_BBO,
                    ExactMetricState.UNDEFINED_EMPTY_BBO,
                ),
                _definition(
                    "ledger.market.best_bid_price",
                    LedgerMetricScope.MARKET,
                    "highest closing active buy limit price",
                    "integer price ticks in configured cash units",
                    "closing active order book",
                    ExactMetricState.UNDEFINED_ONE_SIDED_BBO,
                    ExactMetricState.UNDEFINED_EMPTY_BBO,
                ),
                _definition(
                    "ledger.market.bbo_depth",
                    LedgerMetricScope.MARKET,
                    "closing best-bid depth plus closing best-ask depth",
                    "integer asset units",
                    "closing two-sided active order book",
                    ExactMetricState.UNDEFINED_ONE_SIDED_BBO,
                    ExactMetricState.UNDEFINED_EMPTY_BBO,
                ),
                _definition(
                    "ledger.market.quoted_spread",
                    LedgerMetricScope.MARKET,
                    "closing best ask price minus closing best bid price",
                    "integer price units",
                    "closing two-sided active order book",
                    ExactMetricState.UNDEFINED_ONE_SIDED_BBO,
                    ExactMetricState.UNDEFINED_EMPTY_BBO,
                ),
                _definition(
                    "ledger.market.trade_count",
                    LedgerMetricScope.MARKET,
                    "number of retained trade ledger entries",
                    "integer trades",
                    "count over registered command prefix",
                ),
                _definition(
                    "ledger.market.volume",
                    LedgerMetricScope.MARKET,
                    "sum quantity over retained trade ledger entries",
                    "integer asset units",
                    "sum over registered command prefix",
                ),
                _definition(
                    "ledger.market.venue_cash_level",
                    LedgerMetricScope.MARKET,
                    "closing venue cash balance",
                    "integer cash units",
                    "independently posted settlement close",
                ),
                _definition(
                    "ledger.market.venue_fee_revenue",
                    LedgerMetricScope.MARKET,
                    "closing venue cash minus opening venue cash",
                    "integer cash units",
                    "independently posted settlement close",
                ),
            ),
            key=lambda item: item.metric_id,
        )
    )


def ledger_metric_registry() -> tuple[LedgerMetricDefinition, ...]:
    """Return the closed immutable registry for ledger-derived diagnostics."""
    return _build_ledger_metric_registry()


def ledger_metric_registry_sha256() -> str:
    """Return the registry commitment embedded in every result."""
    registry = ledger_metric_registry()
    return sha256_bytes(canonical_json_bytes([item.to_mapping() for item in registry]))


@dataclass(frozen=True, slots=True)
class LedgerMetric:
    """Named verifier-owned metric cell."""

    metric_id: str
    value: ExactMetric

    def __post_init__(self) -> None:
        _identifier(self.metric_id, "metric_id")
        if not isinstance(self.value, ExactMetric):
            _fail("ledger metric requires an ExactMetric")

    def to_mapping(self) -> dict[str, object]:
        return {"metric_id": self.metric_id, "value": self.value.to_mapping()}


class BookState(StrEnum):
    """Whether a horizon has both, one, or neither side of the BBO."""

    TWO_SIDED = "two_sided"
    BID_ONLY = "bid_only"
    ASK_ONLY = "ask_only"
    EMPTY = "empty"


class SettlementCommodity(StrEnum):
    CASH = "cash"
    ASSET = "asset"


class SettlementPartyKind(StrEnum):
    ACCOUNT = "account"
    VENUE = "venue"


@dataclass(frozen=True, slots=True)
class SettlementPosting:
    """One signed leg of a two-commodity trade settlement."""

    trade_id: str
    posting_index: int
    party_kind: SettlementPartyKind
    account_id: str | None
    commodity: SettlementCommodity
    component: str
    delta: int

    def __post_init__(self) -> None:
        _identifier(self.trade_id, "trade_id")
        if _integer(self.posting_index, "posting_index") < 0:
            _fail("posting_index must be non-negative")
        if not isinstance(self.party_kind, SettlementPartyKind) or not isinstance(
            self.commodity, SettlementCommodity
        ):
            _fail("settlement posting enum has the wrong type")
        _identifier(self.component, "component")
        _integer(self.delta, "delta")
        if self.party_kind is SettlementPartyKind.ACCOUNT:
            _identifier(self.account_id, "account_id")
        elif self.account_id is not None:
            _fail("venue postings cannot impersonate an account")

    def to_mapping(self) -> dict[str, object]:
        return {
            "account_id": self.account_id,
            "commodity": self.commodity.value,
            "component": self.component,
            "delta": self.delta,
            "party_kind": self.party_kind.value,
            "posting_index": self.posting_index,
            "trade_id": self.trade_id,
        }


@dataclass(frozen=True, slots=True)
class TradeSettlement:
    """Independently recomputed settlement evidence for one retained fill."""

    sequence: int
    trade_id: str
    notional: int
    buyer_fee: int
    seller_fee: int
    postings: tuple[SettlementPosting, ...]

    def __post_init__(self) -> None:
        if _integer(self.sequence, "trade sequence") < 0:
            _fail("trade sequence must be non-negative")
        _identifier(self.trade_id, "trade_id")
        for field in ("notional", "buyer_fee", "seller_fee"):
            if _integer(getattr(self, field), field) < 0:
                _fail(f"{field} must be non-negative")
        if (
            not isinstance(self.postings, tuple)
            or len(self.postings) != 8
            or not all(isinstance(item, SettlementPosting) for item in self.postings)
            or tuple(item.posting_index for item in self.postings) != tuple(range(8))
            or any(item.trade_id != self.trade_id for item in self.postings)
        ):
            _fail("trade settlement postings are incomplete or reordered")
        for commodity in SettlementCommodity:
            if (
                sum(item.delta for item in self.postings if item.commodity is commodity)
                != 0
            ):
                _fail("trade settlement postings do not balance by commodity")

    def to_mapping(self) -> dict[str, object]:
        return {
            "buyer_fee": self.buyer_fee,
            "notional": self.notional,
            "postings": [item.to_mapping() for item in self.postings],
            "seller_fee": self.seller_fee,
            "sequence": self.sequence,
            "trade_id": self.trade_id,
        }


@dataclass(frozen=True, slots=True)
class AccountClosure:
    """Opening plus independently posted balances matched to replayed balances."""

    account_id: str
    opening_cash: int
    opening_asset: int
    posted_cash: int
    posted_asset: int
    replay_cash: int
    replay_asset: int

    def __post_init__(self) -> None:
        _identifier(self.account_id, "account_id")
        for field in (
            "opening_cash",
            "opening_asset",
            "posted_cash",
            "posted_asset",
            "replay_cash",
            "replay_asset",
        ):
            _integer(getattr(self, field), field)
        if (
            self.posted_cash != self.replay_cash
            or self.posted_asset != self.replay_asset
        ):
            _fail("independent account postings do not close to replayed balances")

    def to_mapping(self) -> dict[str, object]:
        return {
            "account_id": self.account_id,
            "opening_asset": self.opening_asset,
            "opening_cash": self.opening_cash,
            "posted_asset": self.posted_asset,
            "posted_cash": self.posted_cash,
            "replay_asset": self.replay_asset,
            "replay_cash": self.replay_cash,
        }


@dataclass(frozen=True, slots=True)
class VenueClosure:
    """Independent fee postings matched to the replayed venue cash balance."""

    opening_cash: int
    posted_cash: int
    replay_cash: int

    def __post_init__(self) -> None:
        for field in ("opening_cash", "posted_cash", "replay_cash"):
            _integer(getattr(self, field), field)
        if self.posted_cash != self.replay_cash:
            _fail("independent venue postings do not close to replayed venue cash")

    def to_mapping(self) -> dict[str, int]:
        return {
            "opening_cash": self.opening_cash,
            "posted_cash": self.posted_cash,
            "replay_cash": self.replay_cash,
        }


@dataclass(frozen=True, slots=True)
class OrderLifecycleClosure:
    """Trade-ledger quantities and terminal lifecycle independently reconciled."""

    order_id: str
    trader_id: str
    source_event_key: str
    side: Side
    order_type: str
    status: OrderStatus
    original_quantity: int
    ledger_filled_quantity: int
    record_filled_quantity: int
    record_remaining_quantity: int
    is_resting: bool
    cancel_event_key: str | None

    def __post_init__(self) -> None:
        _identifier(self.order_id, "order_id")
        _identifier(self.trader_id, "trader_id")
        if not isinstance(
            self.source_event_key, str
        ) or not self.source_event_key.startswith("ptk1:"):
            _fail("order lifecycle source event key is invalid")
        if not isinstance(self.side, Side) or not isinstance(self.status, OrderStatus):
            _fail("order lifecycle enum has the wrong type")
        if self.order_type not in {"limit", "market", "opening_limit"}:
            _fail("order lifecycle type is unsupported")
        for field in (
            "original_quantity",
            "ledger_filled_quantity",
            "record_filled_quantity",
            "record_remaining_quantity",
        ):
            if _integer(getattr(self, field), field) < 0:
                _fail(f"{field} must be non-negative")
        if self.original_quantity <= 0:
            _fail("order original quantity must be positive")
        if (
            self.ledger_filled_quantity != self.record_filled_quantity
            or self.record_filled_quantity + self.record_remaining_quantity
            != self.original_quantity
        ):
            _fail("order lifecycle does not close to retained trade quantities")
        if not isinstance(self.is_resting, bool):
            _fail("order lifecycle resting flag must be boolean")
        if self.status is OrderStatus.RESTING:
            if not self.is_resting or self.record_remaining_quantity <= 0:
                _fail("resting lifecycle status does not identify an active order")
        elif self.is_resting:
            _fail("terminal lifecycle status cannot identify an active order")
        if self.status is OrderStatus.FILLED and self.record_remaining_quantity != 0:
            _fail("filled lifecycle status has non-zero remaining quantity")
        if self.status is OrderStatus.CANCELED:
            if (
                self.record_remaining_quantity <= 0
                or not isinstance(self.cancel_event_key, str)
                or not self.cancel_event_key.startswith("ptk1:")
            ):
                _fail("canceled lifecycle lacks a valid cancel event")
        elif self.cancel_event_key is not None:
            _fail("non-canceled lifecycle cannot carry a cancel event")
        if self.status is OrderStatus.EXPIRED and (
            self.order_type != "market" or self.record_remaining_quantity <= 0
        ):
            _fail("only a residual market order can expire")
        if self.order_type == "market" and self.status is OrderStatus.RESTING:
            _fail("market orders cannot remain resting")

    def to_mapping(self) -> dict[str, object]:
        return {
            "cancel_event_key": self.cancel_event_key,
            "is_resting": self.is_resting,
            "ledger_filled_quantity": self.ledger_filled_quantity,
            "order_id": self.order_id,
            "order_type": self.order_type,
            "original_quantity": self.original_quantity,
            "record_filled_quantity": self.record_filled_quantity,
            "record_remaining_quantity": self.record_remaining_quantity,
            "side": self.side.value,
            "source_event_key": self.source_event_key,
            "status": self.status.value,
            "trader_id": self.trader_id,
        }


@dataclass(frozen=True, slots=True)
class OrderReservationClosure:
    """Expected maker-side reserve matched to one active resting order."""

    order_id: str
    trader_id: str
    side: Side
    remaining_quantity: int
    expected_reserved_cash: int
    expected_reserved_asset: int
    replay_reserved_cash: int
    replay_reserved_asset: int

    def __post_init__(self) -> None:
        _identifier(self.order_id, "order_id")
        _identifier(self.trader_id, "trader_id")
        if not isinstance(self.side, Side):
            _fail("order reservation side has the wrong type")
        for field in (
            "remaining_quantity",
            "expected_reserved_cash",
            "expected_reserved_asset",
            "replay_reserved_cash",
            "replay_reserved_asset",
        ):
            if _integer(getattr(self, field), field) < 0:
                _fail(f"{field} must be non-negative")
        if self.remaining_quantity <= 0:
            _fail("active reservation requires positive remaining quantity")
        if (
            self.expected_reserved_cash != self.replay_reserved_cash
            or self.expected_reserved_asset != self.replay_reserved_asset
        ):
            _fail("active order reserve differs from independent expectation")

    def to_mapping(self) -> dict[str, object]:
        return {
            "expected_reserved_asset": self.expected_reserved_asset,
            "expected_reserved_cash": self.expected_reserved_cash,
            "order_id": self.order_id,
            "remaining_quantity": self.remaining_quantity,
            "replay_reserved_asset": self.replay_reserved_asset,
            "replay_reserved_cash": self.replay_reserved_cash,
            "side": self.side.value,
            "trader_id": self.trader_id,
        }


@dataclass(frozen=True, slots=True)
class AccountReservationClosure:
    """Sum of active-order reserves matched to one account snapshot."""

    account_id: str
    gross_cash: int
    gross_asset: int
    expected_reserved_cash: int
    expected_reserved_asset: int
    replay_reserved_cash: int
    replay_reserved_asset: int

    def __post_init__(self) -> None:
        _identifier(self.account_id, "account_id")
        for field in (
            "gross_cash",
            "gross_asset",
            "expected_reserved_cash",
            "expected_reserved_asset",
            "replay_reserved_cash",
            "replay_reserved_asset",
        ):
            if _integer(getattr(self, field), field) < 0:
                _fail(f"{field} must be non-negative")
        if (
            self.expected_reserved_cash != self.replay_reserved_cash
            or self.expected_reserved_asset != self.replay_reserved_asset
        ):
            _fail("account reserve differs from the active-order reserve sum")
        if (
            self.replay_reserved_cash > self.gross_cash
            or self.replay_reserved_asset > self.gross_asset
        ):
            _fail("account reserve exceeds its gross cash or asset balance")

    def to_mapping(self) -> dict[str, object]:
        return {
            "account_id": self.account_id,
            "expected_reserved_asset": self.expected_reserved_asset,
            "expected_reserved_cash": self.expected_reserved_cash,
            "gross_asset": self.gross_asset,
            "gross_cash": self.gross_cash,
            "replay_reserved_asset": self.replay_reserved_asset,
            "replay_reserved_cash": self.replay_reserved_cash,
        }


@dataclass(frozen=True, slots=True)
class LedgerGroupOutcome:
    """Exact horizon metrics for one frozen protocol cohort."""

    group_id: str
    cohort_basis: CohortBasis
    cohort_sha256: str
    members: tuple[str, ...]
    metrics: tuple[LedgerMetric, ...]

    def __post_init__(self) -> None:
        _identifier(self.group_id, "group_id")
        if not isinstance(self.cohort_basis, CohortBasis):
            _fail("cohort basis has the wrong type")
        if not isinstance(self.cohort_sha256, str) or len(self.cohort_sha256) != 64:
            _fail("cohort commitment must be a SHA-256 digest")
        if (
            not isinstance(self.members, tuple)
            or not self.members
            or self.members != tuple(sorted(self.members))
            or len(set(self.members)) != len(self.members)
        ):
            _fail("cohort members must be a canonical non-empty tuple")
        if GroupSpec.from_members(self.group_id, self.members).cohort_sha256 != (
            self.cohort_sha256
        ):
            _fail("cohort members differ from their frozen commitment")
        _validate_metrics(self.metrics, "group")

    def to_mapping(self) -> dict[str, object]:
        return {
            "cohort_basis": self.cohort_basis.value,
            "cohort_sha256": self.cohort_sha256,
            "group_id": self.group_id,
            "members": list(self.members),
            "metrics": [item.to_mapping() for item in self.metrics],
        }


@dataclass(frozen=True, slots=True)
class LedgerMarketOutcome:
    """Exact whole-market metrics and explicit BBO availability state."""

    book_state: BookState
    metrics: tuple[LedgerMetric, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.book_state, BookState):
            _fail("book state has the wrong type")
        _validate_metrics(self.metrics, "market")

    def to_mapping(self) -> dict[str, object]:
        return {
            "book_state": self.book_state.value,
            "metrics": [item.to_mapping() for item in self.metrics],
        }


@dataclass(frozen=True, slots=True)
class LedgerHorizonEvidence:
    """One independently replayed and settled registered horizon."""

    horizon_id: str
    end_step: int
    command_event_keys: tuple[str, ...]
    replay_sha256: str
    state_sha256: str
    settlements: tuple[TradeSettlement, ...]
    account_closures: tuple[AccountClosure, ...]
    venue_closure: VenueClosure
    order_lifecycle_closures: tuple[OrderLifecycleClosure, ...]
    order_reservation_closures: tuple[OrderReservationClosure, ...]
    account_reservation_closures: tuple[AccountReservationClosure, ...]
    market: LedgerMarketOutcome
    groups: tuple[LedgerGroupOutcome, ...]

    def __post_init__(self) -> None:
        _identifier(self.horizon_id, "horizon_id")
        if _integer(self.end_step, "end_step") <= 0:
            _fail("horizon end_step must be positive")
        if not isinstance(self.command_event_keys, tuple) or len(
            set(self.command_event_keys)
        ) != len(self.command_event_keys):
            _fail("horizon command keys must be an immutable unique prefix")
        for field in ("replay_sha256", "state_sha256"):
            value = getattr(self, field)
            if not isinstance(value, str) or len(value) != 64:
                _fail(f"{field} must be a SHA-256 digest")
        if tuple(item.sequence for item in self.settlements) != tuple(
            range(len(self.settlements))
        ):
            _fail("horizon trade settlements are incomplete or reordered")
        if not isinstance(self.account_closures, tuple) or tuple(
            item.account_id for item in self.account_closures
        ) != tuple(sorted(item.account_id for item in self.account_closures)):
            _fail("account closures are not in canonical account order")
        if not isinstance(self.venue_closure, VenueClosure):
            _fail("horizon requires a venue closure")
        if not isinstance(self.order_lifecycle_closures, tuple) or tuple(
            item.order_id for item in self.order_lifecycle_closures
        ) != tuple(sorted(item.order_id for item in self.order_lifecycle_closures)):
            _fail("order lifecycle closures are not in canonical order")
        if not isinstance(self.order_reservation_closures, tuple) or tuple(
            item.order_id for item in self.order_reservation_closures
        ) != tuple(sorted(item.order_id for item in self.order_reservation_closures)):
            _fail("order reservation closures are not in canonical order")
        if not isinstance(self.account_reservation_closures, tuple) or tuple(
            item.account_id for item in self.account_reservation_closures
        ) != tuple(
            sorted(item.account_id for item in self.account_reservation_closures)
        ):
            _fail("account reservation closures are not in canonical order")
        if not isinstance(self.market, LedgerMarketOutcome):
            _fail("horizon requires a market outcome")
        if not isinstance(self.groups, tuple) or tuple(
            item.group_id for item in self.groups
        ) != tuple(sorted(item.group_id for item in self.groups)):
            _fail("group outcomes are not in canonical group order")

    def to_mapping(self) -> dict[str, object]:
        return {
            "account_closures": [item.to_mapping() for item in self.account_closures],
            "account_reservation_closures": [
                item.to_mapping() for item in self.account_reservation_closures
            ],
            "command_event_keys": list(self.command_event_keys),
            "end_step": self.end_step,
            "groups": [item.to_mapping() for item in self.groups],
            "horizon_id": self.horizon_id,
            "market": self.market.to_mapping(),
            "order_lifecycle_closures": [
                item.to_mapping() for item in self.order_lifecycle_closures
            ],
            "order_reservation_closures": [
                item.to_mapping() for item in self.order_reservation_closures
            ],
            "replay_sha256": self.replay_sha256,
            "settlements": [item.to_mapping() for item in self.settlements],
            "state_sha256": self.state_sha256,
            "venue_closure": self.venue_closure.to_mapping(),
        }


@dataclass(frozen=True, slots=True)
class LedgerWorldEvidence:
    """All registered ledger-derived horizons for one policy world."""

    role: WorldRole
    policy_sha256: str
    integrated_run_sha256: str
    horizons: tuple[LedgerHorizonEvidence, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.role, WorldRole):
            _fail("ledger world role has the wrong type")
        for field in ("policy_sha256", "integrated_run_sha256"):
            value = getattr(self, field)
            if not isinstance(value, str) or len(value) != 64:
                _fail(f"{field} must be a SHA-256 digest")
        if not isinstance(self.horizons, tuple) or not self.horizons:
            _fail("ledger world requires registered horizons")

    def to_mapping(self) -> dict[str, object]:
        return {
            "horizons": [item.to_mapping() for item in self.horizons],
            "integrated_run_sha256": self.integrated_run_sha256,
            "policy_sha256": self.policy_sha256,
            "role": self.role.value,
        }


class DifferenceScope(StrEnum):
    MARKET = "market"
    GROUP = "group"


@dataclass(frozen=True, slots=True)
class PairedLedgerDifference:
    """Exact candidate-minus-baseline difference for one registered cell."""

    horizon_id: str
    scope: DifferenceScope
    group_id: str | None
    metric_id: str
    baseline: ExactMetric
    candidate: ExactMetric
    difference: ExactMetric

    def __post_init__(self) -> None:
        _identifier(self.horizon_id, "horizon_id")
        _identifier(self.metric_id, "metric_id")
        if not isinstance(self.scope, DifferenceScope):
            _fail("paired difference scope has the wrong type")
        if self.scope is DifferenceScope.GROUP:
            _identifier(self.group_id, "group_id")
        elif self.group_id is not None:
            _fail("market differences cannot carry a group ID")
        expected = _subtract_metrics(self.candidate, self.baseline)
        if self.difference != expected:
            _fail("paired difference is not exact candidate minus baseline")

    def to_mapping(self) -> dict[str, object]:
        return {
            "baseline": self.baseline.to_mapping(),
            "candidate": self.candidate.to_mapping(),
            "difference": self.difference.to_mapping(),
            "group_id": self.group_id,
            "horizon_id": self.horizon_id,
            "metric_id": self.metric_id,
            "scope": self.scope.value,
        }


@dataclass(frozen=True, slots=True, init=False)
class LedgerDerivedEvaluation:
    """Immutable verifier-produced ledger evidence; direct construction is closed."""

    source_artifact_sha256: str
    protocol_sha256: str
    config_sha256: str
    input_sha256: str
    baseline: LedgerWorldEvidence
    candidate: LedgerWorldEvidence
    differences: tuple[PairedLedgerDifference, ...]
    metric_registry: tuple[LedgerMetricDefinition, ...]
    metric_registry_sha256: str
    verifier_implementation_id: str
    verifier_implementation_source_sha256: str
    claim_scope: str
    metric_registry_externally_registered: Literal[False]
    source_protocol_metric_registry_reused: Literal[False]
    synthetic_only: Literal[True]
    ledger_derived_group_outcomes: Literal[True]
    ledger_derived_market_outcomes: Literal[True]
    formal_welfare_evidence: Literal[False]
    integrated_mechanism_verified: Literal[False]
    schema_version: str

    def __init__(self) -> None:
        raise TypeError(
            "LedgerDerivedEvaluation is verifier-produced; use "
            "derive_protocol_bound_ledger_evidence"
        )

    def to_mapping(self) -> dict[str, object]:
        """Return canonical machine-readable evidence."""
        return {
            "baseline": self.baseline.to_mapping(),
            "candidate": self.candidate.to_mapping(),
            "claim_scope": self.claim_scope,
            "config_sha256": self.config_sha256,
            "differences": [item.to_mapping() for item in self.differences],
            "formal_welfare_evidence": self.formal_welfare_evidence,
            "input_sha256": self.input_sha256,
            "integrated_mechanism_verified": self.integrated_mechanism_verified,
            "ledger_derived_group_outcomes": self.ledger_derived_group_outcomes,
            "ledger_derived_market_outcomes": self.ledger_derived_market_outcomes,
            "metric_registry": [item.to_mapping() for item in self.metric_registry],
            "metric_registry_externally_registered": (
                self.metric_registry_externally_registered
            ),
            "metric_registry_sha256": self.metric_registry_sha256,
            "protocol_sha256": self.protocol_sha256,
            "schema_version": self.schema_version,
            "source_protocol_metric_registry_reused": (
                self.source_protocol_metric_registry_reused
            ),
            "source_artifact_sha256": self.source_artifact_sha256,
            "synthetic_only": self.synthetic_only,
            "verifier_implementation_id": self.verifier_implementation_id,
            "verifier_implementation_source_sha256": (
                self.verifier_implementation_source_sha256
            ),
        }

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
        *,
        source_artifact: ProtocolBoundIntegratedSyntheticRunArtifact,
    ) -> Self:
        """Re-derive a persisted result from its only authoritative source.

        Nested caller-provided fields are never trusted or individually
        reconstructed.  A byte-for-byte canonical semantic comparison against
        a fresh verifier derivation rejects mutation, omission, insertion, and
        internally consistent resealing.
        """
        if not isinstance(value, Mapping) or not all(
            isinstance(key, str) for key in value
        ):
            _fail("ledger-derived result must be an object with string keys")
        expected = derive_protocol_bound_ledger_evidence(source_artifact)
        try:
            supplied = canonical_json_bytes(dict(value))
        except Exception as error:
            raise LedgerDerivedEvidenceError(
                "ledger-derived result is not canonical JSON material"
            ) from error
        if supplied != expected.canonical_bytes:
            _fail("ledger-derived result differs from fresh source-bound derivation")
        return cast(Self, expected)

    @property
    def canonical_bytes(self) -> bytes:
        """Return deterministic canonical JSON bytes."""
        return canonical_json_bytes(self.to_mapping())

    @property
    def sha256(self) -> str:
        """Commit to every source binding, posting, metric, and difference."""
        return sha256_bytes(self.canonical_bytes)


def _validate_metrics(metrics: tuple[LedgerMetric, ...], scope: str) -> None:
    if not isinstance(metrics, tuple) or not metrics:
        _fail(f"{scope} metrics must be a non-empty immutable tuple")
    if not all(isinstance(item, LedgerMetric) for item in metrics):
        _fail(f"{scope} metrics have the wrong type")
    metric_ids = tuple(item.metric_id for item in metrics)
    if metric_ids != tuple(sorted(metric_ids)) or len(set(metric_ids)) != len(
        metric_ids
    ):
        _fail(f"{scope} metrics must use unique canonical metric order")
    try:
        metric_scope = LedgerMetricScope(scope)
    except ValueError as error:
        raise LedgerDerivedEvidenceError("unsupported ledger metric scope") from error
    definitions = tuple(
        item for item in ledger_metric_registry() if item.scope is metric_scope
    )
    if metric_ids != tuple(item.metric_id for item in definitions):
        _fail(f"{scope} metrics differ from the closed ledger metric registry")
    for metric, definition in zip(metrics, definitions, strict=True):
        if (
            metric.value.state is not ExactMetricState.DEFINED
            and metric.value.state not in definition.allowed_undefined_states
        ):
            _fail(
                f"{metric.metric_id} uses an undefined state forbidden by its "
                "definition"
            )


def _subtract_metrics(candidate: ExactMetric, baseline: ExactMetric) -> ExactMetric:
    if not isinstance(candidate, ExactMetric) or not isinstance(baseline, ExactMetric):
        _fail("paired subtraction requires exact metrics")
    if (
        candidate.state is not ExactMetricState.DEFINED
        or baseline.state is not ExactMetricState.DEFINED
    ):
        return ExactMetric.undefined(ExactMetricState.UNDEFINED_WORLD_METRIC)
    assert candidate.value is not None
    assert baseline.value is not None
    return ExactMetric(ExactMetricState.DEFINED, candidate.value.minus(baseline.value))


def _metric_map(metrics: tuple[LedgerMetric, ...]) -> dict[str, ExactMetric]:
    return {item.metric_id: item.value for item in metrics}


def _make_metrics(values: dict[str, ExactMetric]) -> tuple[LedgerMetric, ...]:
    return tuple(
        LedgerMetric(metric_id, value) for metric_id, value in sorted(values.items())
    )


def _book_metrics(snapshot: MarketSnapshot) -> tuple[BookState, dict[str, ExactMetric]]:
    bids = tuple(item for item in snapshot.resting_orders if item.side is Side.BUY)
    asks = tuple(item for item in snapshot.resting_orders if item.side is Side.SELL)
    best_bid = max((item.price for item in bids), default=None)
    best_ask = min((item.price for item in asks), default=None)
    if best_bid is not None and best_ask is not None:
        state = BookState.TWO_SIDED
        bid_depth = sum(
            item.remaining_quantity for item in bids if item.price == best_bid
        )
        ask_depth = sum(
            item.remaining_quantity for item in asks if item.price == best_ask
        )
        return state, {
            "ledger.market.best_ask_depth": ExactMetric.integer(ask_depth),
            "ledger.market.best_ask_price": ExactMetric.integer(best_ask),
            "ledger.market.best_bid_depth": ExactMetric.integer(bid_depth),
            "ledger.market.best_bid_price": ExactMetric.integer(best_bid),
            "ledger.market.bbo_depth": ExactMetric.integer(bid_depth + ask_depth),
            "ledger.market.quoted_spread": ExactMetric.integer(best_ask - best_bid),
        }
    undefined = (
        ExactMetricState.UNDEFINED_ONE_SIDED_BBO
        if best_bid is not None or best_ask is not None
        else ExactMetricState.UNDEFINED_EMPTY_BBO
    )
    state = (
        BookState.BID_ONLY
        if best_bid is not None
        else BookState.ASK_ONLY
        if best_ask is not None
        else BookState.EMPTY
    )
    return state, {
        "ledger.market.best_ask_depth": (
            ExactMetric.integer(
                sum(item.remaining_quantity for item in asks if item.price == best_ask)
            )
            if best_ask is not None
            else ExactMetric.undefined(undefined)
        ),
        "ledger.market.best_ask_price": (
            ExactMetric.integer(best_ask)
            if best_ask is not None
            else ExactMetric.undefined(undefined)
        ),
        "ledger.market.best_bid_depth": (
            ExactMetric.integer(
                sum(item.remaining_quantity for item in bids if item.price == best_bid)
            )
            if best_bid is not None
            else ExactMetric.undefined(undefined)
        ),
        "ledger.market.best_bid_price": (
            ExactMetric.integer(best_bid)
            if best_bid is not None
            else ExactMetric.undefined(undefined)
        ),
        "ledger.market.bbo_depth": ExactMetric.undefined(undefined),
        "ledger.market.quoted_spread": ExactMetric.undefined(undefined),
    }


def _midpoint(snapshot: MarketSnapshot) -> ExactMetric:
    _, metrics = _book_metrics(snapshot)
    bid = metrics["ledger.market.best_bid_price"]
    ask = metrics["ledger.market.best_ask_price"]
    if (
        bid.state is not ExactMetricState.DEFINED
        or ask.state is not ExactMetricState.DEFINED
    ):
        state = (
            ExactMetricState.UNDEFINED_EMPTY_BBO
            if bid.state is ExactMetricState.UNDEFINED_EMPTY_BBO
            else ExactMetricState.UNDEFINED_ONE_SIDED_BBO
        )
        return ExactMetric.undefined(state)
    assert bid.value is not None and ask.value is not None
    return ExactMetric.rational(
        bid.value.numerator + ask.value.numerator,
        2,
    )


def _fee(notional: int, rate_bps: int) -> int:
    return notional * rate_bps // FEE_DENOMINATOR_BPS


def _settle_trade(
    trade: TradeLogEntry,
    replay: ReplayResult,
) -> TradeSettlement:
    if trade.sequence < 0 or trade.price <= 0 or trade.quantity <= 0:
        _fail("trade fields are outside the exact settlement domain")
    order_owners: dict[str, tuple[str, Side]] = {
        item.order_id: (item.trader_id, item.side)
        for item in replay.opening_state.orders
    }
    scheduled_by_event = {
        item.event_key: item.command for item in replay.scheduled_commands
    }
    for scheduled in replay.scheduled_commands:
        command = scheduled.command
        if isinstance(command, LimitOrder | MarketOrder):
            if command.order_id in order_owners:
                _fail("retained replay contains duplicate order identities")
            order_owners[command.order_id] = (command.trader_id, command.side)
    maker = order_owners.get(trade.maker_order_id)
    taker = order_owners.get(trade.taker_order_id)
    cause = scheduled_by_event.get(trade.cause_event_key)
    if (
        maker is None
        or taker is None
        or not isinstance(cause, LimitOrder | MarketOrder)
        or cause.order_id != trade.taker_order_id
        or cause.trader_id != trade.taker_trader_id
        or maker[0] != trade.maker_trader_id
        or taker[0] != trade.taker_trader_id
    ):
        _fail("trade identities do not reconcile to retained opening and commands")
    if trade.aggressor_side is Side.BUY:
        expected_buyer = trade.taker_trader_id
        expected_seller = trade.maker_trader_id
        if taker[1] is not Side.BUY or maker[1] is not Side.SELL:
            _fail("buy-aggressor trade sides do not reconcile to retained orders")
    elif trade.aggressor_side is Side.SELL:
        expected_buyer = trade.maker_trader_id
        expected_seller = trade.taker_trader_id
        if taker[1] is not Side.SELL or maker[1] is not Side.BUY:
            _fail("sell-aggressor trade sides do not reconcile to retained orders")
    else:
        _fail("trade aggressor side is unsupported")
    if trade.buyer_id != expected_buyer or trade.seller_id != expected_seller:
        _fail("trade buyer and seller do not reconcile to maker/taker sides")
    notional = trade.price * trade.quantity
    maker_fee = _fee(notional, replay.config.maker_fee_bps)
    taker_fee = _fee(notional, replay.config.taker_fee_bps)
    if trade.maker_fee != maker_fee or trade.taker_fee != taker_fee:
        _fail("retained trade fees differ from independently recomputed fees")
    buyer_fee = taker_fee if trade.buyer_id == trade.taker_trader_id else maker_fee
    seller_fee = maker_fee if trade.seller_id == trade.maker_trader_id else taker_fee
    expected_trade_id = "ptt1:" + sha256_bytes(
        canonical_json_bytes(
            {
                "cause_event_key": trade.cause_event_key,
                "fill_index": trade.fill_index,
                "maker_order_id": trade.maker_order_id,
                "quantity": trade.quantity,
                "taker_order_id": trade.taker_order_id,
            }
        )
    )
    if trade.trade_id != expected_trade_id:
        _fail("retained trade ID differs from independently derived identity")
    posting_material = (
        (
            SettlementPartyKind.ACCOUNT,
            trade.buyer_id,
            SettlementCommodity.CASH,
            "notional",
            -notional,
        ),
        (
            SettlementPartyKind.ACCOUNT,
            trade.seller_id,
            SettlementCommodity.CASH,
            "notional",
            notional,
        ),
        (
            SettlementPartyKind.ACCOUNT,
            trade.buyer_id,
            SettlementCommodity.CASH,
            "buyer_fee",
            -buyer_fee,
        ),
        (
            SettlementPartyKind.VENUE,
            None,
            SettlementCommodity.CASH,
            "buyer_fee",
            buyer_fee,
        ),
        (
            SettlementPartyKind.ACCOUNT,
            trade.seller_id,
            SettlementCommodity.CASH,
            "seller_fee",
            -seller_fee,
        ),
        (
            SettlementPartyKind.VENUE,
            None,
            SettlementCommodity.CASH,
            "seller_fee",
            seller_fee,
        ),
        (
            SettlementPartyKind.ACCOUNT,
            trade.buyer_id,
            SettlementCommodity.ASSET,
            "asset",
            trade.quantity,
        ),
        (
            SettlementPartyKind.ACCOUNT,
            trade.seller_id,
            SettlementCommodity.ASSET,
            "asset",
            -trade.quantity,
        ),
    )
    postings = tuple(
        SettlementPosting(
            trade_id=trade.trade_id,
            posting_index=index,
            party_kind=party_kind,
            account_id=account_id,
            commodity=commodity,
            component=component,
            delta=delta,
        )
        for index, (party_kind, account_id, commodity, component, delta) in enumerate(
            posting_material
        )
    )
    return TradeSettlement(
        sequence=trade.sequence,
        trade_id=trade.trade_id,
        notional=notional,
        buyer_fee=buyer_fee,
        seller_fee=seller_fee,
        postings=postings,
    )


def _closures(
    opening: OpeningState,
    snapshot: MarketSnapshot,
    settlements: tuple[TradeSettlement, ...],
) -> tuple[tuple[AccountClosure, ...], VenueClosure]:
    opening_accounts = {
        item.trader_id: (item.cash, item.asset) for item in opening.accounts
    }
    replay_accounts = {
        item.trader_id: (item.cash, item.asset) for item in snapshot.accounts
    }
    if set(opening_accounts) != set(replay_accounts):
        _fail("replayed account universe differs from the frozen opening state")
    posted = {
        identifier: [cash, asset]
        for identifier, (cash, asset) in opening_accounts.items()
    }
    venue_cash = opening.venue_cash
    for settlement in settlements:
        for posting in settlement.postings:
            if posting.party_kind is SettlementPartyKind.VENUE:
                if posting.commodity is not SettlementCommodity.CASH:
                    _fail("venue received a non-cash settlement posting")
                venue_cash += posting.delta
                continue
            assert posting.account_id is not None
            if posting.account_id not in posted:
                _fail("settlement posting references an unknown account")
            field = 0 if posting.commodity is SettlementCommodity.CASH else 1
            posted[posting.account_id][field] += posting.delta
    closures = tuple(
        AccountClosure(
            account_id=identifier,
            opening_cash=opening_accounts[identifier][0],
            opening_asset=opening_accounts[identifier][1],
            posted_cash=posted[identifier][0],
            posted_asset=posted[identifier][1],
            replay_cash=replay_accounts[identifier][0],
            replay_asset=replay_accounts[identifier][1],
        )
        for identifier in sorted(opening_accounts)
    )
    venue = VenueClosure(
        opening_cash=opening.venue_cash,
        posted_cash=venue_cash,
        replay_cash=snapshot.venue_cash,
    )
    opening_cash_total = (
        sum(item.cash for item in opening.accounts) + opening.venue_cash
    )
    closing_cash_total = (
        sum(item.cash for item in snapshot.accounts) + snapshot.venue_cash
    )
    opening_asset_total = sum(item.asset for item in opening.accounts)
    closing_asset_total = sum(item.asset for item in snapshot.accounts)
    if (
        opening_cash_total != closing_cash_total
        or opening_asset_total != closing_asset_total
    ):
        _fail("independent horizon closure violates cash or asset conservation")
    return closures, venue


@dataclass(frozen=True, slots=True)
class _OrderSource:
    trader_id: str
    side: Side
    order_type: str
    quantity: int
    source_event_key: str
    price: int | None


def _lifecycle_and_reservation_closures(
    replay: ReplayResult,
) -> tuple[
    tuple[OrderLifecycleClosure, ...],
    tuple[OrderReservationClosure, ...],
    tuple[AccountReservationClosure, ...],
]:
    """Reconcile lifecycle and reserves without trusting engine invariant flags."""
    sources: dict[str, _OrderSource] = {}
    for opening_order in replay.opening_state.orders:
        if opening_order.order_id in sources:
            _fail("opening state contains duplicate order identities")
        sources[opening_order.order_id] = _OrderSource(
            trader_id=opening_order.trader_id,
            side=opening_order.side,
            order_type="opening_limit",
            quantity=opening_order.quantity,
            source_event_key=opening_order.event_key.value,
            price=opening_order.price,
        )
    cancels: dict[str, tuple[str, str]] = {}
    for scheduled in replay.scheduled_commands:
        command = scheduled.command
        if isinstance(command, LimitOrder | MarketOrder):
            if command.order_id in sources:
                _fail("command prefix contains duplicate order identities")
            sources[command.order_id] = _OrderSource(
                trader_id=command.trader_id,
                side=command.side,
                order_type=("limit" if isinstance(command, LimitOrder) else "market"),
                quantity=command.quantity,
                source_event_key=scheduled.event_key,
                price=command.price if isinstance(command, LimitOrder) else None,
            )
        elif isinstance(command, CancelOrder):
            if command.target_order_id in cancels:
                _fail("command prefix contains repeated cancellation of one order")
            cancels[command.target_order_id] = (
                command.trader_id,
                scheduled.event_key,
            )

    record_map = {item.order_id: item for item in replay.state.order_records}
    if len(record_map) != len(replay.state.order_records) or set(record_map) != set(
        sources
    ):
        _fail("order lifecycle registry differs from opening and submitted orders")
    resting_map = {item.order_id: item for item in replay.state.resting_orders}
    if len(resting_map) != len(replay.state.resting_orders):
        _fail("active order book contains duplicate order identities")
    if not set(resting_map).issubset(sources):
        _fail("active order book contains an order without an immutable source")
    scheduled_index: dict[str, int] = {}
    for index, scheduled in enumerate(replay.scheduled_commands):
        if (
            scheduled.event_key in scheduled_index
            or scheduled.command.event_key.value != scheduled.event_key
        ):
            _fail("scheduled command keys are duplicated or internally inconsistent")
        scheduled_index[scheduled.event_key] = index
    filled_by_order: Counter[str] = Counter()
    for trade in replay.trades:
        if (
            trade.maker_order_id == trade.taker_order_id
            or trade.maker_order_id not in sources
            or trade.taker_order_id not in sources
        ):
            _fail("trade ledger references invalid order identities")
        maker_source = sources[trade.maker_order_id]
        taker_source = sources[trade.taker_order_id]
        cause_index = scheduled_index.get(trade.cause_event_key)
        maker_index = scheduled_index.get(maker_source.source_event_key, -1)
        if (
            cause_index is None
            or taker_source.source_event_key != trade.cause_event_key
            or maker_index >= cause_index
        ):
            _fail("trade cause does not follow the resting maker source")
        filled_by_order[trade.maker_order_id] += trade.quantity
        filled_by_order[trade.taker_order_id] += trade.quantity

    lifecycle: list[OrderLifecycleClosure] = []
    canceled_ids: set[str] = set()
    for order_id in sorted(sources):
        source = sources[order_id]
        record = record_map[order_id]
        resting = resting_map.get(order_id)
        if (
            record.trader_id != source.trader_id
            or record.side is not source.side
            or record.order_type != source.order_type
            or record.original_quantity != source.quantity
        ):
            _fail("order lifecycle fields differ from their immutable source command")
        if resting is not None and (
            resting.trader_id != source.trader_id
            or resting.side is not source.side
            or source.price is None
            or resting.price != source.price
            or resting.event_key != source.source_event_key
            or resting.remaining_quantity != record.remaining_quantity
        ):
            _fail("active order fields differ from their immutable source command")
        cancel = cancels.get(order_id)
        if record.status is OrderStatus.CANCELED:
            if cancel is None or cancel[0] != source.trader_id:
                _fail("canceled lifecycle is not justified by an owner cancel command")
            cancel_index = scheduled_index[cancel[1]]
            source_index = scheduled_index.get(source.source_event_key, -1)
            if source_index >= cancel_index:
                _fail("cancel event does not follow the target order submission")
            for trade in replay.trades:
                if order_id in (trade.maker_order_id, trade.taker_order_id):
                    cause_index = scheduled_index.get(trade.cause_event_key)
                    if cause_index is None or cause_index >= cancel_index:
                        _fail("canceled order has a fill at or after its cancel event")
            canceled_ids.add(order_id)
        elif cancel is not None:
            _fail("cancel command does not close to a canceled lifecycle record")
        lifecycle.append(
            OrderLifecycleClosure(
                order_id=order_id,
                trader_id=source.trader_id,
                source_event_key=source.source_event_key,
                side=source.side,
                order_type=source.order_type,
                status=record.status,
                original_quantity=source.quantity,
                ledger_filled_quantity=filled_by_order[order_id],
                record_filled_quantity=record.filled_quantity,
                record_remaining_quantity=record.remaining_quantity,
                is_resting=resting is not None,
                cancel_event_key=None if cancel is None else cancel[1],
            )
        )
    if set(cancels) != canceled_ids:
        _fail("cancel command registry differs from canceled lifecycle records")

    expected_cash: Counter[str] = Counter()
    expected_asset: Counter[str] = Counter()
    order_reservations: list[OrderReservationClosure] = []
    for order_id in sorted(resting_map):
        resting_order = resting_map[order_id]
        if resting_order.side is Side.BUY:
            notional = resting_order.price * resting_order.remaining_quantity
            cash = notional + _fee(notional, replay.config.maker_fee_bps)
            asset = 0
        elif resting_order.side is Side.SELL:
            cash = 0
            asset = resting_order.remaining_quantity
        else:
            _fail("active order has unsupported side")
        expected_cash[resting_order.trader_id] += cash
        expected_asset[resting_order.trader_id] += asset
        order_reservations.append(
            OrderReservationClosure(
                order_id=order_id,
                trader_id=resting_order.trader_id,
                side=resting_order.side,
                remaining_quantity=resting_order.remaining_quantity,
                expected_reserved_cash=cash,
                expected_reserved_asset=asset,
                replay_reserved_cash=resting_order.reserved_cash,
                replay_reserved_asset=resting_order.reserved_asset,
            )
        )
    accounts = {item.trader_id: item for item in replay.state.accounts}
    if len(accounts) != len(replay.state.accounts):
        _fail("account snapshot contains duplicate identities")
    if any(item.trader_id not in accounts for item in replay.state.resting_orders):
        _fail("active order reserve references an unknown account")
    account_reservations = tuple(
        AccountReservationClosure(
            account_id=account_id,
            gross_cash=account.cash,
            gross_asset=account.asset,
            expected_reserved_cash=expected_cash[account_id],
            expected_reserved_asset=expected_asset[account_id],
            replay_reserved_cash=account.reserved_cash,
            replay_reserved_asset=account.reserved_asset,
        )
        for account_id, account in sorted(accounts.items())
    )
    return tuple(lifecycle), tuple(order_reservations), account_reservations


def _replay_boundary(
    run: IntegratedSyntheticRunArtifact,
    boundary: IntegratedHorizonReplayBoundary,
) -> ReplayResult:
    by_key = {
        item.event_key: item.command for item in run.market_replay.scheduled_commands
    }
    if len(by_key) != len(run.market_replay.scheduled_commands):
        _fail("terminal market replay contains duplicate command event keys")
    try:
        commands = tuple(by_key[key] for key in boundary.command_event_keys)
    except KeyError as error:
        raise LedgerDerivedEvidenceError(
            "registered horizon references a command absent from the retained replay"
        ) from error
    try:
        engine = CentralLimitOrderBook(
            run.market_replay.config, run.market_replay.opening_state
        )
        if commands:
            engine.schedule_all(commands)
        replay = engine.run()
        engine.assert_invariants()
    except Exception as error:
        raise LedgerDerivedEvidenceError(
            "registered horizon failed independent market replay"
        ) from error
    if (
        tuple(item.event_key for item in replay.scheduled_commands)
        != boundary.command_event_keys
        or replay.sha256 != boundary.replay_sha256
        or replay.state.sha256 != boundary.state_sha256
    ):
        _fail("independent horizon replay differs from its retained commitment")
    return replay


def _cohort_bindings(
    artifact: ProtocolBoundIntegratedSyntheticRunArtifact,
) -> tuple[tuple[str, CohortBasis, str, tuple[str, ...]], ...]:
    protocol_cohorts = {item.cohort_id: item for item in artifact.protocol.cohorts}
    bindings: list[tuple[str, CohortBasis, str, tuple[str, ...]]] = []
    for group in artifact.pair_input.groups:
        members = artifact.config.members_for(group.group_id)
        expected = GroupSpec.from_members(group.group_id, members)
        cohort = protocol_cohorts.get(group.group_id)
        if (
            expected != group
            or cohort is None
            or cohort.definition_sha256 != group.cohort_sha256
            or cohort.basis is CohortBasis.POST_TREATMENT_DESCRIPTIVE
        ):
            _fail("cohort membership is not frozen and protocol-bound")
        bindings.append((group.group_id, cohort.basis, group.cohort_sha256, members))
    return tuple(sorted(bindings, key=lambda item: item[0]))


def _group_outcome(
    *,
    binding: tuple[str, CohortBasis, str, tuple[str, ...]],
    replay: ReplayResult,
    opening_midpoint: ExactMetric,
    settlements: tuple[TradeSettlement, ...],
) -> LedgerGroupOutcome:
    group_id, cohort_basis, cohort_sha256, members_tuple = binding
    members = frozenset(members_tuple)
    accounts = tuple(
        item for item in replay.state.accounts if item.trader_id in members
    )
    if {item.trader_id for item in accounts} != members:
        _fail("frozen cohort is not fully represented in the replay account universe")
    cash = sum(item.cash for item in accounts)
    inventory = sum(item.asset for item in accounts)
    submitted_orders = tuple(
        item.command
        for item in replay.scheduled_commands
        if isinstance(item.command, LimitOrder | MarketOrder)
        and item.command.trader_id in members
    )
    submitted_order_ids = {item.order_id for item in submitted_orders}
    submitted_quantity = sum(item.quantity for item in submitted_orders)
    gross_filled = sum(
        trade.quantity
        * (int(trade.buyer_id in members) + int(trade.seller_id in members))
        for trade in replay.trades
    )
    submitted_filled = sum(
        trade.quantity
        * (
            int(trade.maker_order_id in submitted_order_ids)
            + int(trade.taker_order_id in submitted_order_ids)
        )
        for trade in replay.trades
    )
    fees_paid = sum(
        settlement.buyer_fee * int(trade.buyer_id in members)
        + settlement.seller_fee * int(trade.seller_id in members)
        for settlement, trade in zip(settlements, replay.trades, strict=True)
    )
    midpoint = _midpoint(replay.state)
    if midpoint.state is ExactMetricState.DEFINED:
        assert midpoint.value is not None
        wealth_value = ExactRational.from_ratio(cash).plus(
            midpoint.value.times_integer(inventory)
        )
        midpoint_wealth = ExactMetric(ExactMetricState.DEFINED, wealth_value)
    else:
        midpoint_wealth = ExactMetric.undefined(midpoint.state)
    opening_accounts = tuple(
        item for item in replay.opening_state.accounts if item.trader_id in members
    )
    opening_cash = sum(item.cash for item in opening_accounts)
    opening_inventory = sum(item.asset for item in opening_accounts)
    if opening_midpoint.state is not ExactMetricState.DEFINED:
        midpoint_pnl = ExactMetric.undefined(ExactMetricState.UNDEFINED_OPENING_BBO)
    elif midpoint_wealth.state is not ExactMetricState.DEFINED:
        midpoint_pnl = ExactMetric.undefined(midpoint_wealth.state)
    else:
        assert opening_midpoint.value is not None and midpoint_wealth.value is not None
        opening_wealth = ExactRational.from_ratio(opening_cash).plus(
            opening_midpoint.value.times_integer(opening_inventory)
        )
        midpoint_pnl = ExactMetric(
            ExactMetricState.DEFINED, midpoint_wealth.value.minus(opening_wealth)
        )
    conditional_fill_rate = (
        ExactMetric.undefined(ExactMetricState.UNDEFINED_ZERO_DENOMINATOR)
        if submitted_quantity == 0
        else ExactMetric.rational(submitted_filled, submitted_quantity)
    )
    return LedgerGroupOutcome(
        group_id=group_id,
        cohort_basis=cohort_basis,
        cohort_sha256=cohort_sha256,
        members=members_tuple,
        metrics=_make_metrics(
            {
                "ledger.group.cash_level": ExactMetric.integer(cash),
                "ledger.group.conditional_fill_rate": conditional_fill_rate,
                "ledger.group.gross_filled_quantity": ExactMetric.integer(gross_filled),
                "ledger.group.inventory_level": ExactMetric.integer(inventory),
                "ledger.group.midpoint_pnl": midpoint_pnl,
                "ledger.group.midpoint_wealth": midpoint_wealth,
                "ledger.group.submitted_filled_quantity": ExactMetric.integer(
                    submitted_filled
                ),
                "ledger.group.submitted_order_count": ExactMetric.integer(
                    len(submitted_orders)
                ),
                "ledger.group.submitted_order_quantity": ExactMetric.integer(
                    submitted_quantity
                ),
                "ledger.group.trading_fees_paid": ExactMetric.integer(fees_paid),
            }
        ),
    )


def _market_outcome(
    replay: ReplayResult,
    venue: VenueClosure,
) -> LedgerMarketOutcome:
    state, bbo = _book_metrics(replay.state)
    values = {
        **bbo,
        "ledger.market.trade_count": ExactMetric.integer(len(replay.trades)),
        "ledger.market.volume": ExactMetric.integer(
            sum(item.quantity for item in replay.trades)
        ),
        "ledger.market.venue_cash_level": ExactMetric.integer(venue.replay_cash),
        "ledger.market.venue_fee_revenue": ExactMetric.integer(
            venue.replay_cash - venue.opening_cash
        ),
    }
    return LedgerMarketOutcome(book_state=state, metrics=_make_metrics(values))


def _world_evidence(
    artifact: ProtocolBoundIntegratedSyntheticRunArtifact,
    run: IntegratedSyntheticRunArtifact,
    bindings: tuple[tuple[str, CohortBasis, str, tuple[str, ...]], ...],
) -> LedgerWorldEvidence:
    opening_engine = CentralLimitOrderBook(
        run.market_replay.config, run.market_replay.opening_state
    )
    opening_replay = opening_engine.run()
    opening_engine.assert_invariants()
    opening_midpoint = _midpoint(opening_replay.state)
    pair_horizons = {
        item.horizon_id: item.end_step for item in artifact.pair_input.horizons
    }
    evidence: list[LedgerHorizonEvidence] = []
    for boundary in run.horizon_boundaries:
        if pair_horizons.get(boundary.horizon_id) != boundary.end_step:
            _fail("retained horizon differs from the frozen paired input")
        replay = _replay_boundary(run, boundary)
        settlements = tuple(_settle_trade(item, replay) for item in replay.trades)
        if tuple(item.sequence for item in replay.trades) != tuple(
            range(len(replay.trades))
        ):
            _fail("retained trade ledger is incomplete or reordered")
        closures, venue = _closures(replay.opening_state, replay.state, settlements)
        lifecycle, order_reservations, account_reservations = (
            _lifecycle_and_reservation_closures(replay)
        )
        groups = tuple(
            _group_outcome(
                binding=binding,
                replay=replay,
                opening_midpoint=opening_midpoint,
                settlements=settlements,
            )
            for binding in bindings
        )
        evidence.append(
            LedgerHorizonEvidence(
                horizon_id=boundary.horizon_id,
                end_step=boundary.end_step,
                command_event_keys=boundary.command_event_keys,
                replay_sha256=replay.sha256,
                state_sha256=replay.state.sha256,
                settlements=settlements,
                account_closures=closures,
                venue_closure=venue,
                order_lifecycle_closures=lifecycle,
                order_reservation_closures=order_reservations,
                account_reservation_closures=account_reservations,
                market=_market_outcome(replay, venue),
                groups=groups,
            )
        )
    if set(pair_horizons) != {item.horizon_id for item in evidence}:
        _fail("ledger evidence does not cover every frozen horizon")
    return LedgerWorldEvidence(
        role=run.role,
        policy_sha256=run.policy_sha256,
        integrated_run_sha256=run.sha256,
        horizons=tuple(evidence),
    )


def _paired_differences(
    baseline: LedgerWorldEvidence,
    candidate: LedgerWorldEvidence,
) -> tuple[PairedLedgerDifference, ...]:
    baseline_horizons = {item.horizon_id: item for item in baseline.horizons}
    candidate_horizons = {item.horizon_id: item for item in candidate.horizons}
    if set(baseline_horizons) != set(candidate_horizons):
        _fail("paired ledger worlds have different registered horizons")
    differences: list[PairedLedgerDifference] = []
    for horizon_id in sorted(baseline_horizons):
        base_horizon = baseline_horizons[horizon_id]
        cand_horizon = candidate_horizons[horizon_id]
        base_market = _metric_map(base_horizon.market.metrics)
        cand_market = _metric_map(cand_horizon.market.metrics)
        if set(base_market) != set(cand_market):
            _fail("paired market metric registries differ")
        for metric_id in sorted(base_market):
            differences.append(
                PairedLedgerDifference(
                    horizon_id=horizon_id,
                    scope=DifferenceScope.MARKET,
                    group_id=None,
                    metric_id=metric_id,
                    baseline=base_market[metric_id],
                    candidate=cand_market[metric_id],
                    difference=_subtract_metrics(
                        cand_market[metric_id], base_market[metric_id]
                    ),
                )
            )
        base_groups = {item.group_id: item for item in base_horizon.groups}
        cand_groups = {item.group_id: item for item in cand_horizon.groups}
        if set(base_groups) != set(cand_groups):
            _fail("paired frozen cohort registries differ")
        for group_id in sorted(base_groups):
            if (
                base_groups[group_id].cohort_sha256
                != cand_groups[group_id].cohort_sha256
                or base_groups[group_id].members != cand_groups[group_id].members
            ):
                _fail("paired worlds changed frozen cohort membership")
            base_metrics = _metric_map(base_groups[group_id].metrics)
            cand_metrics = _metric_map(cand_groups[group_id].metrics)
            if set(base_metrics) != set(cand_metrics):
                _fail("paired cohort metric registries differ")
            for metric_id in sorted(base_metrics):
                differences.append(
                    PairedLedgerDifference(
                        horizon_id=horizon_id,
                        scope=DifferenceScope.GROUP,
                        group_id=group_id,
                        metric_id=metric_id,
                        baseline=base_metrics[metric_id],
                        candidate=cand_metrics[metric_id],
                        difference=_subtract_metrics(
                            cand_metrics[metric_id], base_metrics[metric_id]
                        ),
                    )
                )
    return tuple(differences)


def ledger_evidence_verifier_implementation_sha256() -> str:
    """Commit to the full live module callable bundle, not a helper allowlist."""
    return _ledger_live_callable_bundle_sha256()


def _build_result(
    *,
    artifact: ProtocolBoundIntegratedSyntheticRunArtifact,
    baseline: LedgerWorldEvidence,
    candidate: LedgerWorldEvidence,
    differences: tuple[PairedLedgerDifference, ...],
) -> LedgerDerivedEvaluation:
    result = object.__new__(LedgerDerivedEvaluation)
    values: tuple[tuple[str, object], ...] = (
        ("source_artifact_sha256", artifact.sha256),
        ("protocol_sha256", artifact.protocol.sha256),
        ("config_sha256", artifact.config_sha256),
        ("input_sha256", artifact.input_sha256),
        ("baseline", baseline),
        ("candidate", candidate),
        ("differences", differences),
        ("metric_registry", ledger_metric_registry()),
        ("metric_registry_sha256", ledger_metric_registry_sha256()),
        (
            "verifier_implementation_id",
            LEDGER_EVIDENCE_VERIFIER_IMPLEMENTATION_ID,
        ),
        (
            "verifier_implementation_source_sha256",
            ledger_evidence_verifier_implementation_sha256(),
        ),
        ("claim_scope", LEDGER_DERIVED_CLAIM_SCOPE),
        ("metric_registry_externally_registered", False),
        ("source_protocol_metric_registry_reused", False),
        ("synthetic_only", True),
        ("ledger_derived_group_outcomes", True),
        ("ledger_derived_market_outcomes", True),
        ("formal_welfare_evidence", False),
        ("integrated_mechanism_verified", False),
        ("schema_version", LEDGER_DERIVED_SCHEMA_VERSION),
    )
    for field, value in values:
        object.__setattr__(result, field, value)
    return result


def derive_protocol_bound_ledger_evidence(
    artifact: ProtocolBoundIntegratedSyntheticRunArtifact,
) -> LedgerDerivedEvaluation:
    """Derive exact ledger outcomes from one fully bound synthetic pair.

    The sole input is the high-level typed artifact.  Caller-provided outcomes,
    evidence flags, cohort members, and pass/fail booleans are intentionally not
    accepted.
    """
    if type(artifact) is not ProtocolBoundIntegratedSyntheticRunArtifact:
        raise TypeError(
            "artifact must be an exact ProtocolBoundIntegratedSyntheticRunArtifact"
        )
    try:
        verified = require_ledger_derivation_source_artifact(artifact)
        verified.paired_outcome.integrity_audit.require_passed()
    except Exception as error:
        raise LedgerDerivedEvidenceError(
            "protocol-bound integrated artifact failed fresh verification"
        ) from error
    if (
        verified.synthetic_only is not True
        or verified.baseline_run.integrated_mechanism_verified is not False
        or verified.candidate_run.integrated_mechanism_verified is not False
        or verified.baseline_run.ledger_derived_group_outcomes is not False
        or verified.candidate_run.ledger_derived_group_outcomes is not False
    ):
        _fail("source artifact evidence scope was unexpectedly broadened")
    bindings = _cohort_bindings(verified)
    baseline = _world_evidence(verified, verified.baseline_run, bindings)
    candidate = _world_evidence(verified, verified.candidate_run, bindings)
    differences = _paired_differences(baseline, candidate)
    if verified.pair_input.candidate_policy.is_noop:
        if verified.baseline_run.market_replay.canonical_bytes != (
            verified.candidate_run.market_replay.canonical_bytes
        ):
            _fail("no-op paired worlds do not have identical retained market replay")
        for cell in differences:
            if cell.baseline != cell.candidate:
                _fail("no-op paired ledger metric changed between worlds")
            if (
                cell.difference.state is ExactMetricState.DEFINED
                and cell.difference.value != ExactRational.from_ratio(0)
            ):
                _fail("no-op paired ledger difference is not exactly zero")
    return _build_result(
        artifact=verified,
        baseline=baseline,
        candidate=candidate,
        differences=differences,
    )


def _stat_fingerprint(value: os.stat_result) -> tuple[int, ...]:
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


def load_ledger_derived_evaluation(
    path: Path,
    *,
    expected_sha256: str,
    source_artifact: ProtocolBoundIntegratedSyntheticRunArtifact,
) -> LedgerDerivedEvaluation:
    """Load canonical evidence under external bytes and typed-source authority."""
    if not isinstance(path, Path):
        _fail("ledger-derived result path must be a Path")
    expected = _sha256(expected_sha256, "expected_sha256")
    candidate = Path(os.path.abspath(path.expanduser()))
    current = Path(candidate.anchor)
    try:
        for part in candidate.parts[1:]:
            current /= part
            if current.is_symlink():
                _fail("ledger-derived result path contains a symbolic link")
        descriptor = os.open(
            candidate,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
        )
    except OSError as error:
        raise LedgerDerivedEvidenceError("cannot open ledger-derived result") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > _MAX_ARTIFACT_BYTES
        ):
            _fail("ledger-derived result is not a bounded regular file")
        chunks: list[bytes] = []
        total = 0
        while total <= _MAX_ARTIFACT_BYTES:
            chunk = os.read(
                descriptor,
                min(1024 * 1024, _MAX_ARTIFACT_BYTES + 1 - total),
            )
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        if total > _MAX_ARTIFACT_BYTES:
            _fail("ledger-derived result exceeds its byte limit")
        after = os.fstat(descriptor)
        content = b"".join(chunks)
    finally:
        os.close(descriptor)
    if (
        _stat_fingerprint(before) != _stat_fingerprint(after)
        or len(content) != before.st_size
    ):
        _fail("ledger-derived result changed while it was read")
    try:
        linked = candidate.stat(follow_symlinks=False)
    except OSError as error:
        raise LedgerDerivedEvidenceError(
            "cannot restat ledger-derived result"
        ) from error
    if not stat.S_ISREG(linked.st_mode) or _stat_fingerprint(
        linked
    ) != _stat_fingerprint(after):
        _fail("ledger-derived result path identity changed while it was read")
    if sha256_bytes(content) != expected:
        _fail("ledger-derived result differs from its external commitment")
    try:
        parsed = read_json_bytes_object(content, source=str(candidate))
    except Exception as error:
        raise LedgerDerivedEvidenceError(
            "cannot parse ledger-derived result"
        ) from error
    if canonical_json_bytes(parsed) != content:
        _fail("ledger-derived result must use canonical JSON encoding")
    result = LedgerDerivedEvaluation.from_mapping(
        parsed,
        source_artifact=source_artifact,
    )
    if result.sha256 != expected:
        _fail("ledger-derived semantic replay differs from its raw commitment")
    return result


__all__ = [
    "LEDGER_DERIVED_CLAIM_SCOPE",
    "LEDGER_DERIVED_SCHEMA_VERSION",
    "LEDGER_EVIDENCE_VERIFIER_IMPLEMENTATION_ID",
    "LEDGER_METRIC_REGISTRY_VERSION",
    "AccountClosure",
    "AccountReservationClosure",
    "BookState",
    "DifferenceScope",
    "ExactMetric",
    "ExactMetricState",
    "ExactRational",
    "LedgerDerivedEvaluation",
    "LedgerDerivedEvidenceError",
    "LedgerGroupOutcome",
    "LedgerHorizonEvidence",
    "LedgerMarketOutcome",
    "LedgerMetric",
    "LedgerMetricDefinition",
    "LedgerMetricScope",
    "LedgerWorldEvidence",
    "OrderLifecycleClosure",
    "OrderReservationClosure",
    "PairedLedgerDifference",
    "SettlementCommodity",
    "SettlementPartyKind",
    "SettlementPosting",
    "TradeSettlement",
    "VenueClosure",
    "derive_protocol_bound_ledger_evidence",
    "ledger_metric_registry",
    "ledger_metric_registry_sha256",
    "ledger_evidence_verifier_implementation_sha256",
    "load_ledger_derived_evaluation",
]
