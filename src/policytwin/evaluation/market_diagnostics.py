"""Frozen, synthetic-only market-microstructure diagnostics.

The formulas in this module are deliberately explicit.  The module consumes
already frozen observations, does not estimate a market model, and cannot
authorize a real-data claim.  Real inputs remain fail-closed until an opaque
temporal-firewall receipt is verified at this boundary.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import StrEnum
from statistics import fmean
from typing import NoReturn

from policytwin.provenance import canonical_json_bytes, sha256_bytes

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class MarketDiagnosticError(ValueError):
    """Raised when an extended market diagnostic is not well-defined."""


def _fail(message: str) -> NoReturn:
    raise MarketDiagnosticError(message)


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        _fail(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{field} must be finite")
    return result


def _positive(value: object, field: str, *, allow_zero: bool = False) -> float:
    result = _finite(value, field)
    if result < 0.0 or (result == 0.0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        _fail(f"{field} must be {qualifier}")
    return result


def _integer(value: object, field: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"{field} must be an integer >= {minimum}")
    return value


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        _fail(f"{field} must be a stable identifier")
    return value


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail(f"{field} must be a lowercase SHA-256 digest")
    return value


def _tuple_of(
    value: object,
    expected: type[object],
    field: str,
    *,
    minimum: int = 1,
) -> tuple[object, ...]:
    if (
        not isinstance(value, tuple)
        or len(value) < minimum
        or not all(isinstance(item, expected) for item in value)
    ):
        _fail(f"{field} must be an immutable tuple of {expected.__name__}")
    return value


def _mean(values: tuple[float, ...], field: str) -> float:
    if not values:
        _fail(f"{field} is undefined for an empty sample")
    return fmean(values)


def _population_variance(values: tuple[float, ...], field: str) -> float:
    if len(values) < 2:
        _fail(f"{field} requires at least two observations")
    mean = fmean(values)
    return fmean((value - mean) ** 2 for value in values)


def _correlation(
    left: tuple[float, ...], right: tuple[float, ...], field: str
) -> float:
    if len(left) != len(right) or len(left) < 2:
        _fail(f"{field} requires aligned samples of length at least two")
    left_mean = fmean(left)
    right_mean = fmean(right)
    left_ss = math.fsum((value - left_mean) ** 2 for value in left)
    right_ss = math.fsum((value - right_mean) ** 2 for value in right)
    if left_ss == 0.0 or right_ss == 0.0:
        _fail(f"{field} is undefined for a zero-variance sample")
    covariance_sum = math.fsum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True)
    )
    result = covariance_sum / math.sqrt(left_ss * right_ss)
    return max(-1.0, min(1.0, result))


class DiagnosticTradeSide(StrEnum):
    """Trade direction used by execution and impact formulas."""

    BUY = "buy"
    SELL = "sell"

    @property
    def sign(self) -> float:
        """Return +1 for a buy and -1 for a sell."""
        return 1.0 if self is DiagnosticTradeSide.BUY else -1.0


@dataclass(frozen=True, slots=True)
class DepthLevel:
    """One displayed price level in a point-in-time order-book snapshot."""

    price: float
    displayed_quantity: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "price", _positive(self.price, "depth price"))
        object.__setattr__(
            self,
            "displayed_quantity",
            _positive(
                self.displayed_quantity,
                "depth displayed_quantity",
                allow_zero=True,
            ),
        )

    def to_mapping(self) -> dict[str, float]:
        return {"displayed_quantity": self.displayed_quantity, "price": self.price}


@dataclass(frozen=True, slots=True)
class DepthSnapshot:
    """Visible book levels around one strictly positive midpoint."""

    snapshot_id: str
    time: float
    midpoint: float
    bids: tuple[DepthLevel, ...]
    asks: tuple[DepthLevel, ...]

    def __post_init__(self) -> None:
        _identifier(self.snapshot_id, "depth snapshot_id")
        object.__setattr__(self, "time", _finite(self.time, "depth time"))
        object.__setattr__(self, "midpoint", _positive(self.midpoint, "midpoint"))
        _tuple_of(self.bids, DepthLevel, "bids")
        _tuple_of(self.asks, DepthLevel, "asks")
        bids = self.bids
        asks = self.asks
        if any(level.price >= self.midpoint for level in bids):
            _fail("every bid depth level must be strictly below the midpoint")
        if any(level.price <= self.midpoint for level in asks):
            _fail("every ask depth level must be strictly above the midpoint")
        bid_prices = tuple(level.price for level in bids)
        ask_prices = tuple(level.price for level in asks)
        if any(
            later >= earlier
            for earlier, later in zip(bid_prices, bid_prices[1:], strict=False)
        ):
            _fail("bid levels must be strictly decreasing in price")
        if any(
            later <= earlier
            for earlier, later in zip(ask_prices, ask_prices[1:], strict=False)
        ):
            _fail("ask levels must be strictly increasing in price")

    def to_mapping(self) -> dict[str, object]:
        return {
            "asks": [level.to_mapping() for level in self.asks],
            "bids": [level.to_mapping() for level in self.bids],
            "midpoint": self.midpoint,
            "snapshot_id": self.snapshot_id,
            "time": self.time,
        }


@dataclass(frozen=True, slots=True)
class SignedFlowObservation:
    """Signed net order flow and contemporaneous log midpoint return."""

    interval_id: str
    start_time: float
    end_time: float
    signed_quantity: float
    log_midpoint_return: float

    def __post_init__(self) -> None:
        _identifier(self.interval_id, "flow interval_id")
        object.__setattr__(self, "start_time", _finite(self.start_time, "start_time"))
        object.__setattr__(self, "end_time", _finite(self.end_time, "end_time"))
        if self.end_time <= self.start_time:
            _fail("flow end_time must be after start_time")
        object.__setattr__(
            self,
            "signed_quantity",
            _finite(self.signed_quantity, "signed_quantity"),
        )
        object.__setattr__(
            self,
            "log_midpoint_return",
            _finite(self.log_midpoint_return, "log_midpoint_return"),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "end_time": self.end_time,
            "interval_id": self.interval_id,
            "log_midpoint_return": self.log_midpoint_return,
            "signed_quantity": self.signed_quantity,
            "start_time": self.start_time,
        }


@dataclass(frozen=True, slots=True)
class ResiliencyResponse:
    """Quoted-spread recovery path after a liquidity shock."""

    event_id: str
    event_time: float
    baseline_spread: float
    times: tuple[float, ...]
    quoted_spreads: tuple[float, ...]

    def __post_init__(self) -> None:
        _identifier(self.event_id, "resiliency event_id")
        object.__setattr__(self, "event_time", _finite(self.event_time, "event_time"))
        object.__setattr__(
            self,
            "baseline_spread",
            _positive(self.baseline_spread, "baseline_spread"),
        )
        if not isinstance(self.times, tuple) or not isinstance(
            self.quoted_spreads, tuple
        ):
            _fail("resiliency paths must be immutable tuples")
        times = tuple(_finite(value, "resiliency time") for value in self.times)
        spreads = tuple(
            _positive(value, "resiliency quoted spread")
            for value in self.quoted_spreads
        )
        if len(times) != len(spreads) or len(times) < 2:
            _fail("resiliency response requires aligned paths of length at least two")
        if times[0] != self.event_time or any(
            later <= earlier for earlier, later in zip(times, times[1:], strict=False)
        ):
            _fail("resiliency times must start at event_time and strictly increase")
        if spreads[0] <= self.baseline_spread:
            _fail("initial spread must exceed the pre-shock baseline spread")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "quoted_spreads", spreads)

    def to_mapping(self) -> dict[str, object]:
        return {
            "baseline_spread": self.baseline_spread,
            "event_id": self.event_id,
            "event_time": self.event_time,
            "quoted_spreads": list(self.quoted_spreads),
            "times": list(self.times),
        }


@dataclass(frozen=True, slots=True)
class ExecutionFill:
    """One child fill contributing to a frozen parent-order benchmark."""

    price: float
    quantity: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "price", _positive(self.price, "fill price"))
        object.__setattr__(self, "quantity", _positive(self.quantity, "fill quantity"))

    def to_mapping(self) -> dict[str, float]:
        return {"price": self.price, "quantity": self.quantity}


@dataclass(frozen=True, slots=True)
class ParentOrderExecution:
    """Parent order with a decision-price and terminal opportunity benchmark."""

    order_id: str
    side: DiagnosticTradeSide
    decision_price: float
    terminal_price: float
    target_quantity: float
    fills: tuple[ExecutionFill, ...]

    def __post_init__(self) -> None:
        _identifier(self.order_id, "parent order_id")
        if not isinstance(self.side, DiagnosticTradeSide):
            _fail("parent order side must be a DiagnosticTradeSide")
        for field in ("decision_price", "terminal_price", "target_quantity"):
            object.__setattr__(self, field, _positive(getattr(self, field), field))
        _tuple_of(self.fills, ExecutionFill, "parent-order fills", minimum=0)
        fills = self.fills
        if math.fsum(fill.quantity for fill in fills) > self.target_quantity:
            _fail("parent-order fills must not exceed target_quantity")

    def to_mapping(self) -> dict[str, object]:
        return {
            "decision_price": self.decision_price,
            "fills": [fill.to_mapping() for fill in self.fills],
            "order_id": self.order_id,
            "side": self.side.value,
            "target_quantity": self.target_quantity,
            "terminal_price": self.terminal_price,
        }


@dataclass(frozen=True, slots=True)
class LiquidityDemand:
    """Requested quantity and quantity visible inside the frozen depth band."""

    demand_id: str
    requested_quantity: float
    available_quantity: float

    def __post_init__(self) -> None:
        _identifier(self.demand_id, "liquidity demand_id")
        object.__setattr__(
            self,
            "requested_quantity",
            _positive(self.requested_quantity, "requested_quantity"),
        )
        object.__setattr__(
            self,
            "available_quantity",
            _positive(self.available_quantity, "available_quantity", allow_zero=True),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "available_quantity": self.available_quantity,
            "demand_id": self.demand_id,
            "requested_quantity": self.requested_quantity,
        }


@dataclass(frozen=True, slots=True)
class ImpactObservation:
    """Immediate and fixed-long-horizon midquotes around signed flow."""

    impact_id: str
    side: DiagnosticTradeSide
    quantity: float
    pretrade_midquote: float
    immediate_midquote: float
    long_horizon_midquote: float

    def __post_init__(self) -> None:
        _identifier(self.impact_id, "impact_id")
        if not isinstance(self.side, DiagnosticTradeSide):
            _fail("impact side must be a DiagnosticTradeSide")
        for field in (
            "quantity",
            "pretrade_midquote",
            "immediate_midquote",
            "long_horizon_midquote",
        ):
            object.__setattr__(self, field, _positive(getattr(self, field), field))

    def to_mapping(self) -> dict[str, object]:
        return {
            "immediate_midquote": self.immediate_midquote,
            "impact_id": self.impact_id,
            "long_horizon_midquote": self.long_horizon_midquote,
            "pretrade_midquote": self.pretrade_midquote,
            "quantity": self.quantity,
            "side": self.side.value,
        }


@dataclass(frozen=True, slots=True)
class DiagnosticPricePath:
    """Strictly ordered price path used for return-based diagnostics."""

    path_id: str
    times: tuple[float, ...]
    prices: tuple[float, ...]

    def __post_init__(self) -> None:
        _identifier(self.path_id, "diagnostic price path_id")
        if not isinstance(self.times, tuple) or not isinstance(self.prices, tuple):
            _fail("diagnostic price paths must be immutable tuples")
        times = tuple(_finite(value, "diagnostic path time") for value in self.times)
        prices = tuple(
            _positive(value, "diagnostic path price") for value in self.prices
        )
        if len(times) != len(prices) or len(times) < 3:
            _fail("diagnostic price path requires at least three aligned prices")
        if any(
            later <= earlier for earlier, later in zip(times, times[1:], strict=False)
        ):
            _fail("diagnostic price path times must strictly increase")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "prices", prices)

    @property
    def log_returns(self) -> tuple[float, ...]:
        """Return consecutive, unannualized log returns."""
        return tuple(
            math.log(current / previous)
            for previous, current in zip(self.prices, self.prices[1:], strict=False)
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "path_id": self.path_id,
            "prices": list(self.prices),
            "times": list(self.times),
        }


@dataclass(frozen=True, slots=True)
class ExtendedMarketMetricConfig:
    """Immutable formula choices; no value may be selected after outcomes."""

    depth_band_bps: float
    resiliency_recovery_fraction: float
    volatility_clustering_lag: int
    return_autocorrelation_lag: int
    variance_ratio_horizon: int
    hill_tail_observation_count: int
    liquidity_shortfall_alpha: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "depth_band_bps",
            _positive(self.depth_band_bps, "depth_band_bps"),
        )
        recovery = _finite(
            self.resiliency_recovery_fraction,
            "resiliency_recovery_fraction",
        )
        if not 0.0 < recovery < 1.0:
            _fail("resiliency_recovery_fraction must lie strictly inside (0, 1)")
        object.__setattr__(self, "resiliency_recovery_fraction", recovery)
        for field in ("volatility_clustering_lag", "return_autocorrelation_lag"):
            object.__setattr__(
                self, field, _integer(getattr(self, field), field, minimum=1)
            )
        object.__setattr__(
            self,
            "variance_ratio_horizon",
            _integer(self.variance_ratio_horizon, "variance_ratio_horizon", minimum=2),
        )
        object.__setattr__(
            self,
            "hill_tail_observation_count",
            _integer(
                self.hill_tail_observation_count,
                "hill_tail_observation_count",
                minimum=1,
            ),
        )
        alpha = _finite(self.liquidity_shortfall_alpha, "liquidity_shortfall_alpha")
        if not 0.0 < alpha < 1.0:
            _fail("liquidity_shortfall_alpha must lie strictly inside (0, 1)")
        object.__setattr__(self, "liquidity_shortfall_alpha", alpha)

    def to_mapping(self) -> dict[str, object]:
        return {
            "depth_band_bps": self.depth_band_bps,
            "hill_tail_observation_count": self.hill_tail_observation_count,
            "liquidity_shortfall_alpha": self.liquidity_shortfall_alpha,
            "resiliency_recovery_fraction": self.resiliency_recovery_fraction,
            "return_autocorrelation_lag": self.return_autocorrelation_lag,
            "variance_ratio_horizon": self.variance_ratio_horizon,
            "volatility_clustering_lag": self.volatility_clustering_lag,
        }

    @property
    def sha256(self) -> str:
        """Commit the complete metric-definition configuration."""
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))


@dataclass(frozen=True, slots=True)
class ExtendedMarketInput:
    """Auditable synthetic observations for all expanded market diagnostics."""

    source_artifact_sha256: str
    study_protocol_sha256: str
    depth_snapshots: tuple[DepthSnapshot, ...]
    signed_flows: tuple[SignedFlowObservation, ...]
    resiliency_responses: tuple[ResiliencyResponse, ...]
    parent_orders: tuple[ParentOrderExecution, ...]
    price_paths: tuple[DiagnosticPricePath, ...]
    liquidity_demands: tuple[LiquidityDemand, ...]
    impacts: tuple[ImpactObservation, ...]
    synthetic_only: bool = True

    def __post_init__(self) -> None:
        _sha256(self.source_artifact_sha256, "source_artifact_sha256")
        _sha256(self.study_protocol_sha256, "study_protocol_sha256")
        fields: tuple[tuple[str, type[object], int], ...] = (
            ("depth_snapshots", DepthSnapshot, 1),
            ("signed_flows", SignedFlowObservation, 2),
            ("resiliency_responses", ResiliencyResponse, 1),
            ("parent_orders", ParentOrderExecution, 1),
            ("price_paths", DiagnosticPricePath, 1),
            ("liquidity_demands", LiquidityDemand, 1),
            ("impacts", ImpactObservation, 1),
        )
        for field, expected, minimum in fields:
            _tuple_of(getattr(self, field), expected, field, minimum=minimum)
        if not isinstance(self.synthetic_only, bool):
            _fail("synthetic_only must be a boolean")
        if not self.synthetic_only:
            _fail(
                "real market diagnostics require a typed temporal-firewall "
                "verifier receipt"
            )
        identifiers: tuple[tuple[str, tuple[str, ...]], ...] = (
            (
                "depth snapshot IDs",
                tuple(item.snapshot_id for item in self.depth_snapshots),
            ),
            (
                "flow interval IDs",
                tuple(item.interval_id for item in self.signed_flows),
            ),
            (
                "resiliency event IDs",
                tuple(item.event_id for item in self.resiliency_responses),
            ),
            ("parent order IDs", tuple(item.order_id for item in self.parent_orders)),
            ("price path IDs", tuple(item.path_id for item in self.price_paths)),
            (
                "liquidity demand IDs",
                tuple(item.demand_id for item in self.liquidity_demands),
            ),
            ("impact IDs", tuple(item.impact_id for item in self.impacts)),
        )
        for field, values in identifiers:
            if len(set(values)) != len(values):
                _fail(f"{field} must be unique")

    def to_mapping(self) -> dict[str, object]:
        return {
            "depth_snapshots": [item.to_mapping() for item in self.depth_snapshots],
            "impacts": [item.to_mapping() for item in self.impacts],
            "liquidity_demands": [item.to_mapping() for item in self.liquidity_demands],
            "parent_orders": [item.to_mapping() for item in self.parent_orders],
            "price_paths": [item.to_mapping() for item in self.price_paths],
            "resiliency_responses": [
                item.to_mapping() for item in self.resiliency_responses
            ],
            "signed_flows": [item.to_mapping() for item in self.signed_flows],
            "source_artifact_sha256": self.source_artifact_sha256,
            "study_protocol_sha256": self.study_protocol_sha256,
            "synthetic_only": self.synthetic_only,
        }

    @property
    def sha256(self) -> str:
        """Commit every observation and provenance field."""
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))


@dataclass(frozen=True, slots=True)
class ExtendedMarketMetrics:
    """Expanded metrics with units encoded in field names where material."""

    depth_within_band_quantity: float
    kyle_lambda_log_return_per_quantity: float
    mean_observed_spread_resiliency_recovery_time: float | None
    spread_resiliency_observed_fraction: float
    implementation_shortfall_bps: float
    volatility_clustering_autocorrelation: float
    hill_absolute_return_tail_index: float
    mean_liquidity_shortfall_fraction: float
    liquidity_shortfall_quantile: float
    return_autocorrelation: float
    variance_ratio: float
    temporary_impact_bps: float
    permanent_impact_bps: float
    total_immediate_impact_bps: float
    config_sha256: str
    input_sha256: str
    synthetic_only: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "depth_within_band_quantity",
            _positive(
                self.depth_within_band_quantity,
                "depth_within_band_quantity",
                allow_zero=True,
            ),
        )
        for field in (
            "kyle_lambda_log_return_per_quantity",
            "implementation_shortfall_bps",
            "temporary_impact_bps",
            "permanent_impact_bps",
            "total_immediate_impact_bps",
        ):
            _finite(getattr(self, field), field)
        for field in (
            "spread_resiliency_observed_fraction",
            "mean_liquidity_shortfall_fraction",
            "liquidity_shortfall_quantile",
        ):
            value = _finite(getattr(self, field), field)
            if not 0.0 <= value <= 1.0:
                _fail(f"{field} must lie in [0, 1]")
        for field in (
            "volatility_clustering_autocorrelation",
            "return_autocorrelation",
        ):
            value = _finite(getattr(self, field), field)
            if not -1.0 <= value <= 1.0:
                _fail(f"{field} must lie in [-1, 1]")
        _positive(
            self.hill_absolute_return_tail_index,
            "hill_absolute_return_tail_index",
        )
        _positive(self.variance_ratio, "variance_ratio", allow_zero=True)
        if self.mean_observed_spread_resiliency_recovery_time is not None:
            _positive(
                self.mean_observed_spread_resiliency_recovery_time,
                "mean_observed_spread_resiliency_recovery_time",
                allow_zero=True,
            )
        _sha256(self.config_sha256, "config_sha256")
        _sha256(self.input_sha256, "input_sha256")
        if not math.isclose(
            self.total_immediate_impact_bps,
            self.temporary_impact_bps + self.permanent_impact_bps,
            rel_tol=1e-12,
            abs_tol=1e-9,
        ):
            _fail("temporary plus permanent impact must equal immediate impact")
        if self.synthetic_only is not True:
            _fail("extended market metrics are currently synthetic-only")

    def to_mapping(self) -> dict[str, object]:
        return {
            "config_sha256": self.config_sha256,
            "depth_within_band_quantity": self.depth_within_band_quantity,
            "hill_absolute_return_tail_index": self.hill_absolute_return_tail_index,
            "implementation_shortfall_bps": self.implementation_shortfall_bps,
            "input_sha256": self.input_sha256,
            "kyle_lambda_log_return_per_quantity": (
                self.kyle_lambda_log_return_per_quantity
            ),
            "liquidity_shortfall_quantile": self.liquidity_shortfall_quantile,
            "mean_liquidity_shortfall_fraction": (
                self.mean_liquidity_shortfall_fraction
            ),
            "permanent_impact_bps": self.permanent_impact_bps,
            "return_autocorrelation": self.return_autocorrelation,
            "spread_resiliency_observed_fraction": (
                self.spread_resiliency_observed_fraction
            ),
            "mean_observed_spread_resiliency_recovery_time": (
                self.mean_observed_spread_resiliency_recovery_time
            ),
            "synthetic_only": self.synthetic_only,
            "temporary_impact_bps": self.temporary_impact_bps,
            "total_immediate_impact_bps": self.total_immediate_impact_bps,
            "variance_ratio": self.variance_ratio,
            "volatility_clustering_autocorrelation": (
                self.volatility_clustering_autocorrelation
            ),
        }


def _quantile(values: tuple[float, ...], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + weight * (ordered[upper] - ordered[lower])


def _lagged_correlation(values: tuple[float, ...], lag: int, field: str) -> float:
    if len(values) <= lag + 1:
        _fail(f"{field} requires more than lag + 1 observations")
    return _correlation(values[:-lag], values[lag:], field)


def _variance_ratio(returns: tuple[float, ...], horizon: int) -> float:
    if len(returns) <= horizon:
        _fail("variance ratio requires more returns than its horizon")
    one_step_variance = _population_variance(returns, "one-step return variance")
    if one_step_variance == 0.0:
        _fail("variance ratio is undefined for zero-variance one-step returns")
    overlapping = tuple(
        math.fsum(returns[start : start + horizon])
        for start in range(len(returns) - horizon + 1)
    )
    horizon_variance = _population_variance(
        overlapping, "overlapping horizon-return variance"
    )
    return horizon_variance / (horizon * one_step_variance)


def _hill_tail_index(absolute_returns: tuple[float, ...], count: int) -> float:
    positive = tuple(value for value in absolute_returns if value > 0.0)
    if len(positive) <= count:
        _fail("Hill tail index requires more positive returns than tail observations")
    ordered = sorted(positive)
    threshold = ordered[-count - 1]
    denominator = math.fsum(math.log(value / threshold) for value in ordered[-count:])
    if threshold <= 0.0 or denominator <= 0.0:
        _fail("Hill tail index is undefined for a tied or non-positive tail threshold")
    return count / denominator


def evaluate_extended_market_metrics(
    data: ExtendedMarketInput,
    config: ExtendedMarketMetricConfig,
) -> ExtendedMarketMetrics:
    """Evaluate every preregistered diagnostic without searching parameters."""
    if not isinstance(data, ExtendedMarketInput) or not isinstance(
        config, ExtendedMarketMetricConfig
    ):
        _fail("extended metrics require validated immutable input and config")

    depths = tuple(
        math.fsum(
            level.displayed_quantity
            for level in (*snapshot.bids, *snapshot.asks)
            if abs(level.price - snapshot.midpoint) / snapshot.midpoint * 10_000.0
            <= config.depth_band_bps
        )
        for snapshot in data.depth_snapshots
    )

    signed_quantities = tuple(item.signed_quantity for item in data.signed_flows)
    log_returns = tuple(item.log_midpoint_return for item in data.signed_flows)
    quantity_mean = fmean(signed_quantities)
    return_mean = fmean(log_returns)
    quantity_ss = math.fsum((value - quantity_mean) ** 2 for value in signed_quantities)
    if quantity_ss == 0.0:
        _fail("Kyle lambda is undefined for zero-variance signed quantity")
    kyle_lambda = (
        math.fsum(
            (quantity - quantity_mean) * (value - return_mean)
            for quantity, value in zip(signed_quantities, log_returns, strict=True)
        )
        / quantity_ss
    )

    recovery_times: list[float] = []
    for response in data.resiliency_responses:
        initial_excess = response.quoted_spreads[0] - response.baseline_spread
        threshold = response.baseline_spread + (
            config.resiliency_recovery_fraction * initial_excess
        )
        for time, spread in zip(response.times, response.quoted_spreads, strict=True):
            if spread <= threshold:
                recovery_times.append(time - response.event_time)
                break

    implementation_numerator = 0.0
    implementation_denominator = 0.0
    for order in data.parent_orders:
        filled_quantity = math.fsum(fill.quantity for fill in order.fills)
        unfilled_quantity = order.target_quantity - filled_quantity
        executed_and_opportunity_value = (
            math.fsum(fill.price * fill.quantity for fill in order.fills)
            + unfilled_quantity * order.terminal_price
        )
        benchmark_value = order.target_quantity * order.decision_price
        implementation_numerator += order.side.sign * (
            executed_and_opportunity_value - benchmark_value
        )
        implementation_denominator += benchmark_value
    if implementation_denominator <= 0.0:
        _fail("implementation shortfall requires positive benchmark notional")

    clustering_values: list[float] = []
    autocorrelation_values: list[float] = []
    variance_ratios: list[float] = []
    absolute_returns: list[float] = []
    for path in data.price_paths:
        returns = path.log_returns
        squared = tuple(value * value for value in returns)
        clustering_values.append(
            _lagged_correlation(
                squared,
                config.volatility_clustering_lag,
                "volatility clustering",
            )
        )
        autocorrelation_values.append(
            _lagged_correlation(
                returns,
                config.return_autocorrelation_lag,
                "return autocorrelation",
            )
        )
        variance_ratios.append(_variance_ratio(returns, config.variance_ratio_horizon))
        absolute_returns.extend(abs(value) for value in returns)

    shortfalls = tuple(
        max(demand.requested_quantity - demand.available_quantity, 0.0)
        / demand.requested_quantity
        for demand in data.liquidity_demands
    )

    impact_quantity = math.fsum(item.quantity for item in data.impacts)
    if impact_quantity <= 0.0:
        _fail("impact decomposition requires positive aggregate quantity")
    permanent_impact = (
        math.fsum(
            item.quantity
            * item.side.sign
            * (item.long_horizon_midquote - item.pretrade_midquote)
            / item.pretrade_midquote
            * 10_000.0
            for item in data.impacts
        )
        / impact_quantity
    )
    temporary_impact = (
        math.fsum(
            item.quantity
            * item.side.sign
            * (item.immediate_midquote - item.long_horizon_midquote)
            / item.pretrade_midquote
            * 10_000.0
            for item in data.impacts
        )
        / impact_quantity
    )
    immediate_impact = (
        math.fsum(
            item.quantity
            * item.side.sign
            * (item.immediate_midquote - item.pretrade_midquote)
            / item.pretrade_midquote
            * 10_000.0
            for item in data.impacts
        )
        / impact_quantity
    )

    return ExtendedMarketMetrics(
        depth_within_band_quantity=_mean(depths, "depth within band"),
        kyle_lambda_log_return_per_quantity=kyle_lambda,
        mean_observed_spread_resiliency_recovery_time=(
            fmean(recovery_times) if recovery_times else None
        ),
        spread_resiliency_observed_fraction=(
            len(recovery_times) / len(data.resiliency_responses)
        ),
        implementation_shortfall_bps=(
            implementation_numerator / implementation_denominator * 10_000.0
        ),
        volatility_clustering_autocorrelation=fmean(clustering_values),
        hill_absolute_return_tail_index=_hill_tail_index(
            tuple(absolute_returns), config.hill_tail_observation_count
        ),
        mean_liquidity_shortfall_fraction=fmean(shortfalls),
        liquidity_shortfall_quantile=_quantile(
            shortfalls, config.liquidity_shortfall_alpha
        ),
        return_autocorrelation=fmean(autocorrelation_values),
        variance_ratio=fmean(variance_ratios),
        temporary_impact_bps=temporary_impact,
        permanent_impact_bps=permanent_impact,
        total_immediate_impact_bps=immediate_impact,
        config_sha256=config.sha256,
        input_sha256=data.sha256,
    )
