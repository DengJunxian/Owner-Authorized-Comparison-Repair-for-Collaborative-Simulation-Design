"""Self-verifying, synthetic-only market and distribution metric artifacts.

The convenience evaluators in :mod:`policytwin.evaluation.policy` return the
metric value objects used by existing callers.  The artifacts in this module
are the persistence boundary: they retain the complete typed inputs and frozen
configuration, and they recompute every metric during construction and JSON
mapping ingestion.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import NoReturn, cast

from policytwin.counterfactual.events import SemanticEventKey
from policytwin.evaluation.policy import (
    AgentOutcome,
    DistributionMetricConfig,
    DistributionMetrics,
    DistributionSampleArtifact,
    DistributionSampleRole,
    GroupMetrics,
    InformationResponse,
    IntervalEstimate,
    IntervalKind,
    MarketEvaluationInput,
    MarketMetricConfig,
    MarketMetrics,
    ObservationUniverseManifest,
    OrderOutcome,
    PolicyEvaluationSchemaError,
    PricePath,
    QuoteObservation,
    ScenarioAggregation,
    ScenarioUniverseManifest,
    TradeObservation,
    TradeSide,
    evaluate_distribution_metrics,
    evaluate_group_metrics,
    evaluate_market_metrics,
)
from policytwin.protocol import CohortBasis, FrozenCohort
from policytwin.provenance import canonical_json_bytes, sha256_bytes

GROUP_EVALUATION_CLAIM_SCOPE = "synthetic_group_summary_arithmetic_only"
GROUP_OUTCOME_SOURCE_AUTHORITY = "unverified_caller_authored_synthetic_agent_outcomes"
MARKET_EVALUATION_CLAIM_SCOPE = "synthetic_market_metric_arithmetic_only"
MARKET_OBSERVATION_SOURCE_AUTHORITY = (
    "unverified_caller_authored_synthetic_market_observations"
)


def _fail(message: str) -> NoReturn:
    raise PolicyEvaluationSchemaError(message)


def _object(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        _fail(f"{field} must be an object with string keys")
    return cast(Mapping[str, object], value)


def _fields(value: Mapping[str, object], expected: frozenset[str], field: str) -> None:
    if frozenset(value) != expected:
        _fail(f"{field} has missing or unknown fields")


def _array(value: object, field: str) -> list[object]:
    if not isinstance(value, list):
        _fail(f"{field} must be an array")
    return cast(list[object], value)


def _string(value: object, field: str) -> str:
    if not isinstance(value, str):
        _fail(f"{field} must be a string")
    return value


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        _fail(f"{field} must be a boolean")
    return value


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{field} must be an integer")
    return value


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        _fail(f"{field} must be numeric")
    return float(value)


def _optional_number(value: object, field: str) -> float | None:
    return None if value is None else _number(value, field)


def _datetime(value: object, field: str) -> datetime:
    text = _string(value, field)
    try:
        return datetime.fromisoformat(text)
    except ValueError as error:
        raise PolicyEvaluationSchemaError(f"{field} must be ISO 8601") from error


def _event_key(value: object, field: str) -> SemanticEventKey:
    try:
        return SemanticEventKey.from_mapping(_object(value, field))
    except ValueError as error:
        raise PolicyEvaluationSchemaError(f"{field} is invalid") from error


def _enum_value[T](enum_type: type[T], value: object, field: str) -> T:
    text = _string(value, field)
    try:
        return enum_type(text)  # type: ignore[call-arg]
    except ValueError as error:
        raise PolicyEvaluationSchemaError(f"{field} is unsupported") from error


def _market_config_to_mapping(config: MarketMetricConfig) -> dict[str, object]:
    return config.to_mapping()


def _market_config_from_mapping(value: object) -> MarketMetricConfig:
    mapping = _object(value, "market metric config")
    _fields(
        mapping,
        frozenset(MarketMetricConfig.__dataclass_fields__),
        "market metric config",
    )
    return MarketMetricConfig(
        risk_alpha=_number(mapping["risk_alpha"], "risk_alpha"),
        jump_log_return_threshold=_number(
            mapping["jump_log_return_threshold"], "jump_log_return_threshold"
        ),
        flash_crash_drawdown_threshold=_number(
            mapping["flash_crash_drawdown_threshold"],
            "flash_crash_drawdown_threshold",
        ),
        flash_crash_window=_number(mapping["flash_crash_window"], "flash_crash_window"),
        realized_spread_horizon=_number(
            mapping["realized_spread_horizon"], "realized_spread_horizon"
        ),
        time_unit=_string(mapping["time_unit"], "time_unit"),
        price_unit=_string(mapping["price_unit"], "price_unit"),
        return_sampling_interval=_number(
            mapping["return_sampling_interval"], "return_sampling_interval"
        ),
        volatility_annualization_factor=_number(
            mapping["volatility_annualization_factor"],
            "volatility_annualization_factor",
        ),
        risk_return_horizon=_number(
            mapping["risk_return_horizon"], "risk_return_horizon"
        ),
        risk_return_sampling_unit=_string(
            mapping["risk_return_sampling_unit"], "risk_return_sampling_unit"
        ),
        portfolio_value_sampling_interval=_number(
            mapping["portfolio_value_sampling_interval"],
            "portfolio_value_sampling_interval",
        ),
        maximum_drawdown_horizon=_number(
            mapping["maximum_drawdown_horizon"], "maximum_drawdown_horizon"
        ),
        scenario_aggregation=_enum_value(
            ScenarioAggregation,
            mapping["scenario_aggregation"],
            "scenario_aggregation",
        ),
    )


_MARKET_INPUT_FIELDS = frozenset(MarketEvaluationInput.__dataclass_fields__)


def _market_input_to_mapping(data: MarketEvaluationInput) -> dict[str, object]:
    return {
        "information_responses": [
            {
                "absolute_deviations": list(item.absolute_deviations),
                "event_id": item.event_id,
                "event_time": item.event_time,
                "scenario_id": item.scenario_id,
                "source_event_key": item.source_event_key.to_mapping(),
                "times": list(item.times),
            }
            for item in data.information_responses
        ],
        "information_response_manifest": (
            data.information_response_manifest.to_mapping()
        ),
        "orders": [
            {
                "filled_quantity": item.filled_quantity,
                "filled_time": item.filled_time,
                "order_id": item.order_id,
                "scenario_id": item.scenario_id,
                "source_event_key": item.source_event_key.to_mapping(),
                "submitted_quantity": item.submitted_quantity,
                "submitted_time": item.submitted_time,
            }
            for item in data.orders
        ],
        "order_outcome_manifest": data.order_outcome_manifest.to_mapping(),
        "portfolio_return_scenario_ids": list(data.portfolio_return_scenario_ids),
        "portfolio_return_source_event_keys": [
            item.to_mapping() for item in data.portfolio_return_source_event_keys
        ],
        "portfolio_returns": list(data.portfolio_returns),
        "portfolio_value_paths": [list(path) for path in data.portfolio_value_paths],
        "portfolio_value_scenario_ids": list(data.portfolio_value_scenario_ids),
        "portfolio_value_scenario_manifest": (
            data.portfolio_value_scenario_manifest.to_mapping()
        ),
        "portfolio_value_source_event_keys": [
            item.to_mapping() for item in data.portfolio_value_source_event_keys
        ],
        "portfolio_value_times": [list(path) for path in data.portfolio_value_times],
        "price_path_scenario_manifest": data.price_path_scenario_manifest.to_mapping(),
        "price_paths": [
            {
                "prices": list(item.prices),
                "scenario_id": item.scenario_id,
                "source_event_key": item.source_event_key.to_mapping(),
                "times": list(item.times),
            }
            for item in data.price_paths
        ],
        "quotes": [
            {
                "ask": item.ask,
                "ask_depth": item.ask_depth,
                "bid": item.bid,
                "bid_depth": item.bid_depth,
                "latent_fundamental": item.latent_fundamental,
                "scenario_id": item.scenario_id,
                "source_event_key": item.source_event_key.to_mapping(),
                "time": item.time,
            }
            for item in data.quotes
        ],
        "quote_observation_manifest": data.quote_observation_manifest.to_mapping(),
        "risk_return_horizon": data.risk_return_horizon,
        "risk_return_sampling_unit": data.risk_return_sampling_unit,
        "risk_return_scenario_manifest": (
            data.risk_return_scenario_manifest.to_mapping()
        ),
        "trades": [
            {
                "midquote_after_horizon": item.midquote_after_horizon,
                "midquote_at_execution": item.midquote_at_execution,
                "price": item.price,
                "quantity": item.quantity,
                "realized_spread_horizon": item.realized_spread_horizon,
                "scenario_id": item.scenario_id,
                "side": item.side.value,
                "source_event_key": item.source_event_key.to_mapping(),
            }
            for item in data.trades
        ],
        "trade_observation_manifest": data.trade_observation_manifest.to_mapping(),
    }


def _market_input_from_mapping(value: object) -> MarketEvaluationInput:
    mapping = _object(value, "market evaluation input")
    _fields(mapping, _MARKET_INPUT_FIELDS, "market evaluation input")

    quotes: list[QuoteObservation] = []
    for raw in _array(mapping["quotes"], "quotes"):
        item = _object(raw, "quote")
        _fields(item, frozenset(QuoteObservation.__dataclass_fields__), "quote")
        quotes.append(
            QuoteObservation(
                scenario_id=_string(item["scenario_id"], "quote scenario_id"),
                time=_number(item["time"], "quote time"),
                bid=_number(item["bid"], "quote bid"),
                ask=_number(item["ask"], "quote ask"),
                bid_depth=_number(item["bid_depth"], "quote bid_depth"),
                ask_depth=_number(item["ask_depth"], "quote ask_depth"),
                latent_fundamental=_optional_number(
                    item["latent_fundamental"], "quote latent_fundamental"
                ),
                source_event_key=_event_key(
                    item["source_event_key"], "quote source_event_key"
                ),
            )
        )

    trades: list[TradeObservation] = []
    for raw in _array(mapping["trades"], "trades"):
        item = _object(raw, "trade")
        _fields(item, frozenset(TradeObservation.__dataclass_fields__), "trade")
        trades.append(
            TradeObservation(
                scenario_id=_string(item["scenario_id"], "trade scenario_id"),
                side=_enum_value(TradeSide, item["side"], "trade side"),
                price=_number(item["price"], "trade price"),
                quantity=_number(item["quantity"], "trade quantity"),
                midquote_at_execution=_number(
                    item["midquote_at_execution"], "trade midquote_at_execution"
                ),
                midquote_after_horizon=_number(
                    item["midquote_after_horizon"], "trade midquote_after_horizon"
                ),
                realized_spread_horizon=_number(
                    item["realized_spread_horizon"], "trade realized_spread_horizon"
                ),
                source_event_key=_event_key(
                    item["source_event_key"], "trade source_event_key"
                ),
            )
        )

    orders: list[OrderOutcome] = []
    for raw in _array(mapping["orders"], "orders"):
        item = _object(raw, "order")
        _fields(item, frozenset(OrderOutcome.__dataclass_fields__), "order")
        orders.append(
            OrderOutcome(
                order_id=_string(item["order_id"], "order_id"),
                submitted_time=_number(item["submitted_time"], "submitted_time"),
                submitted_quantity=_number(
                    item["submitted_quantity"], "submitted_quantity"
                ),
                filled_quantity=_number(item["filled_quantity"], "filled_quantity"),
                filled_time=_optional_number(item["filled_time"], "filled_time"),
                scenario_id=_string(item["scenario_id"], "order scenario_id"),
                source_event_key=_event_key(
                    item["source_event_key"], "order source_event_key"
                ),
            )
        )

    paths: list[PricePath] = []
    for raw in _array(mapping["price_paths"], "price_paths"):
        item = _object(raw, "price path")
        _fields(item, frozenset(PricePath.__dataclass_fields__), "price path")
        paths.append(
            PricePath(
                scenario_id=_string(item["scenario_id"], "price path scenario_id"),
                times=tuple(
                    _number(number, "price path time")
                    for number in _array(item["times"], "price path times")
                ),
                prices=tuple(
                    _number(number, "price path price")
                    for number in _array(item["prices"], "price path prices")
                ),
                source_event_key=_event_key(
                    item["source_event_key"], "price path source_event_key"
                ),
            )
        )

    responses: list[InformationResponse] = []
    for raw in _array(mapping["information_responses"], "information_responses"):
        item = _object(raw, "information response")
        _fields(
            item,
            frozenset(InformationResponse.__dataclass_fields__),
            "information response",
        )
        responses.append(
            InformationResponse(
                event_id=_string(item["event_id"], "information event_id"),
                event_time=_number(item["event_time"], "information event_time"),
                times=tuple(
                    _number(number, "information time")
                    for number in _array(item["times"], "information times")
                ),
                absolute_deviations=tuple(
                    _number(number, "information deviation")
                    for number in _array(
                        item["absolute_deviations"], "information deviations"
                    )
                ),
                scenario_id=_string(item["scenario_id"], "information scenario_id"),
                source_event_key=_event_key(
                    item["source_event_key"],
                    "information response source_event_key",
                ),
            )
        )

    result = MarketEvaluationInput(
        quotes=tuple(quotes),
        quote_observation_manifest=ObservationUniverseManifest.from_quotes(
            tuple(quotes)
        ),
        trades=tuple(trades),
        trade_observation_manifest=ObservationUniverseManifest.from_trades(
            tuple(trades)
        ),
        orders=tuple(orders),
        order_outcome_manifest=ObservationUniverseManifest.from_orders(tuple(orders)),
        price_paths=tuple(paths),
        price_path_scenario_manifest=ScenarioUniverseManifest.from_price_paths(
            tuple(paths)
        ),
        portfolio_returns=tuple(
            _number(item, "portfolio return")
            for item in _array(mapping["portfolio_returns"], "portfolio_returns")
        ),
        portfolio_return_scenario_ids=tuple(
            _string(item, "portfolio return scenario_id")
            for item in _array(
                mapping["portfolio_return_scenario_ids"],
                "portfolio_return_scenario_ids",
            )
        ),
        portfolio_return_source_event_keys=tuple(
            _event_key(item, "portfolio return source event key")
            for item in _array(
                mapping["portfolio_return_source_event_keys"],
                "portfolio_return_source_event_keys",
            )
        ),
        risk_return_scenario_manifest=(_rebuild_risk_manifest(mapping)),
        risk_return_horizon=_number(
            mapping["risk_return_horizon"], "risk_return_horizon"
        ),
        risk_return_sampling_unit=_string(
            mapping["risk_return_sampling_unit"], "risk_return_sampling_unit"
        ),
        portfolio_value_paths=tuple(
            tuple(
                _number(item, "portfolio value")
                for item in _array(path, "portfolio value path")
            )
            for path in _array(
                mapping["portfolio_value_paths"], "portfolio_value_paths"
            )
        ),
        portfolio_value_times=tuple(
            tuple(
                _number(item, "portfolio value time")
                for item in _array(path, "portfolio value times")
            )
            for path in _array(
                mapping["portfolio_value_times"], "portfolio_value_times"
            )
        ),
        portfolio_value_scenario_ids=tuple(
            _string(item, "portfolio value scenario_id")
            for item in _array(
                mapping["portfolio_value_scenario_ids"], "portfolio_value_scenario_ids"
            )
        ),
        portfolio_value_source_event_keys=tuple(
            _event_key(item, "portfolio value source event key")
            for item in _array(
                mapping["portfolio_value_source_event_keys"],
                "portfolio_value_source_event_keys",
            )
        ),
        portfolio_value_scenario_manifest=_rebuild_portfolio_value_manifest(mapping),
        information_responses=tuple(responses),
        information_response_manifest=(
            ObservationUniverseManifest.from_information_responses(tuple(responses))
        ),
    )
    for field in (
        "price_path_scenario_manifest",
        "risk_return_scenario_manifest",
        "portfolio_value_scenario_manifest",
        "quote_observation_manifest",
        "trade_observation_manifest",
        "order_outcome_manifest",
        "information_response_manifest",
    ):
        supplied = _object(mapping[field], field)
        if supplied != getattr(result, field).to_mapping():
            _fail(f"{field} does not reproduce from retained observations")
    return result


def _rebuild_risk_manifest(
    mapping: Mapping[str, object],
) -> ScenarioUniverseManifest:
    return ScenarioUniverseManifest.from_risk_returns(
        tuple(
            _string(item, "portfolio return scenario_id")
            for item in _array(
                mapping["portfolio_return_scenario_ids"],
                "portfolio_return_scenario_ids",
            )
        ),
        tuple(
            _number(item, "portfolio return")
            for item in _array(mapping["portfolio_returns"], "portfolio_returns")
        ),
        tuple(
            _event_key(item, "portfolio return source event key")
            for item in _array(
                mapping["portfolio_return_source_event_keys"],
                "portfolio_return_source_event_keys",
            )
        ),
        horizon=_number(mapping["risk_return_horizon"], "risk_return_horizon"),
        sampling_unit=_string(
            mapping["risk_return_sampling_unit"], "risk_return_sampling_unit"
        ),
    )


def _rebuild_portfolio_value_manifest(
    mapping: Mapping[str, object],
) -> ScenarioUniverseManifest:
    return ScenarioUniverseManifest.from_portfolio_value_paths(
        tuple(
            _string(item, "portfolio value scenario_id")
            for item in _array(
                mapping["portfolio_value_scenario_ids"], "portfolio_value_scenario_ids"
            )
        ),
        tuple(
            tuple(
                _number(item, "portfolio value time")
                for item in _array(path, "portfolio value times")
            )
            for path in _array(
                mapping["portfolio_value_times"], "portfolio_value_times"
            )
        ),
        tuple(
            tuple(
                _number(item, "portfolio value")
                for item in _array(path, "portfolio value path")
            )
            for path in _array(
                mapping["portfolio_value_paths"], "portfolio_value_paths"
            )
        ),
        tuple(
            _event_key(item, "portfolio value source event key")
            for item in _array(
                mapping["portfolio_value_source_event_keys"],
                "portfolio_value_source_event_keys",
            )
        ),
    )


def _market_metrics_from_mapping(
    value: object, data: MarketEvaluationInput
) -> MarketMetrics:
    mapping = _object(value, "market metrics")
    _fields(mapping, frozenset(MarketMetrics.__dataclass_fields__), "market metrics")
    universe_fields = (
        "price_path_scenario_universe",
        "risk_return_scenario_universe",
        "portfolio_value_scenario_universe",
        "quote_observation_universe",
        "trade_observation_universe",
        "order_outcome_universe",
        "information_response_universe",
    )
    source_mappings = (
        data.price_path_scenario_manifest.to_mapping(),
        data.risk_return_scenario_manifest.to_mapping(),
        data.portfolio_value_scenario_manifest.to_mapping(),
        data.quote_observation_manifest.to_mapping(),
        data.trade_observation_manifest.to_mapping(),
        data.order_outcome_manifest.to_mapping(),
        data.information_response_manifest.to_mapping(),
    )
    for field, source_mapping in zip(universe_fields, source_mappings, strict=True):
        if _object(mapping[field], field) != source_mapping:
            _fail(f"{field} differs from the retained input")
    return MarketMetrics(
        quoted_spread=_number(mapping["quoted_spread"], "quoted_spread"),
        effective_spread=_number(mapping["effective_spread"], "effective_spread"),
        realized_spread=_number(mapping["realized_spread"], "realized_spread"),
        displayed_depth=_number(mapping["displayed_depth"], "displayed_depth"),
        fill_probability=_number(mapping["fill_probability"], "fill_probability"),
        mean_time_to_fill_among_filled=_number(
            mapping["mean_time_to_fill_among_filled"],
            "mean_time_to_fill_among_filled",
        ),
        price_impact=_number(mapping["price_impact"], "price_impact"),
        realized_volatility=_number(
            mapping["realized_volatility"], "realized_volatility"
        ),
        jump_frequency=_number(mapping["jump_frequency"], "jump_frequency"),
        value_at_risk=_number(mapping["value_at_risk"], "value_at_risk"),
        expected_shortfall=_number(mapping["expected_shortfall"], "expected_shortfall"),
        maximum_drawdown=_number(mapping["maximum_drawdown"], "maximum_drawdown"),
        flash_crash_probability=_number(
            mapping["flash_crash_probability"], "flash_crash_probability"
        ),
        mean_relative_fundamental_deviation=_number(
            mapping["mean_relative_fundamental_deviation"],
            "mean_relative_fundamental_deviation",
        ),
        mean_information_response_half_life_among_recovered=_optional_number(
            mapping["mean_information_response_half_life_among_recovered"],
            "mean_information_response_half_life_among_recovered",
        ),
        information_half_life_observed_fraction=_number(
            mapping["information_half_life_observed_fraction"],
            "information_half_life_observed_fraction",
        ),
        metric_config_sha256=_string(
            mapping["metric_config_sha256"], "metric_config_sha256"
        ),
        scenario_aggregation=_enum_value(
            ScenarioAggregation,
            mapping["scenario_aggregation"],
            "scenario_aggregation",
        ),
        price_path_scenario_universe=data.price_path_scenario_manifest,
        risk_return_scenario_universe=data.risk_return_scenario_manifest,
        portfolio_value_scenario_universe=data.portfolio_value_scenario_manifest,
        quote_observation_universe=data.quote_observation_manifest,
        trade_observation_universe=data.trade_observation_manifest,
        order_outcome_universe=data.order_outcome_manifest,
        information_response_universe=data.information_response_manifest,
    )


def _distribution_sample_from_mapping(value: object) -> DistributionSampleArtifact:
    mapping = _object(value, "distribution sample")
    _fields(
        mapping,
        frozenset(DistributionSampleArtifact.__dataclass_fields__),
        "distribution sample",
    )
    return DistributionSampleArtifact(
        schema_version=_integer(mapping["schema_version"], "sample schema_version"),
        artifact_id=_string(mapping["artifact_id"], "sample artifact_id"),
        role=_enum_value(DistributionSampleRole, mapping["role"], "sample role"),
        values=tuple(
            _number(item, "sample value")
            for item in _array(mapping["values"], "sample values")
        ),
        values_sha256=_string(mapping["values_sha256"], "sample values_sha256"),
        data_manifest_sha256=_string(
            mapping["data_manifest_sha256"], "sample data_manifest_sha256"
        ),
        study_protocol_sha256=_string(
            mapping["study_protocol_sha256"], "sample study_protocol_sha256"
        ),
        available_at=_datetime(mapping["available_at"], "sample available_at"),
        frozen_at=_datetime(mapping["frozen_at"], "sample frozen_at"),
        estimand_id=_string(mapping["estimand_id"], "sample estimand_id"),
        sampling_unit=_string(mapping["sampling_unit"], "sample sampling_unit"),
        synthetic_only=_boolean(mapping["synthetic_only"], "sample synthetic_only"),
    )


def _distribution_config_from_mapping(value: object) -> DistributionMetricConfig:
    mapping = _object(value, "distribution metric config")
    _fields(
        mapping,
        frozenset(DistributionMetricConfig.__dataclass_fields__),
        "distribution metric config",
    )
    return DistributionMetricConfig(
        quantile_levels=tuple(
            _number(item, "quantile level")
            for item in _array(mapping["quantile_levels"], "quantile_levels")
        ),
        interval_level=_number(mapping["interval_level"], "interval_level"),
        tail_event_threshold=_number(
            mapping["tail_event_threshold"], "tail_event_threshold"
        ),
        reported_interval_levels=tuple(
            _number(item, "reported interval level")
            for item in _array(
                mapping["reported_interval_levels"], "reported_interval_levels"
            )
        ),
        interval_kind=_enum_value(
            IntervalKind, mapping["interval_kind"], "interval_kind"
        ),
        interval_method_id=_string(mapping["interval_method_id"], "interval_method_id"),
        estimand_id=_string(mapping["estimand_id"], "estimand_id"),
        sampling_unit=_string(mapping["sampling_unit"], "sampling_unit"),
        minimum_valid_draws=_integer(
            mapping["minimum_valid_draws"], "minimum_valid_draws"
        ),
        mmd_bandwidth=_number(mapping["mmd_bandwidth"], "mmd_bandwidth"),
    )


def _interval_from_mapping(value: object) -> IntervalEstimate:
    mapping = _object(value, "interval")
    _fields(mapping, frozenset(IntervalEstimate.__dataclass_fields__), "interval")
    return IntervalEstimate(
        level=_number(mapping["level"], "interval level"),
        lower=_number(mapping["lower"], "interval lower"),
        upper=_number(mapping["upper"], "interval upper"),
        interval_score=_number(mapping["interval_score"], "interval score"),
        empirical_coverage=_number(
            mapping["empirical_coverage"], "interval empirical coverage"
        ),
    )


def _distribution_metrics_from_mapping(value: object) -> DistributionMetrics:
    mapping = _object(value, "distribution metrics")
    _fields(
        mapping,
        frozenset(DistributionMetrics.__dataclass_fields__),
        "distribution metrics",
    )
    quantiles: list[tuple[float, float]] = []
    for raw in _array(mapping["quantiles"], "quantiles"):
        item = _object(raw, "quantile")
        _fields(item, frozenset({"level", "value"}), "quantile")
        quantiles.append(
            (
                _number(item["level"], "quantile level"),
                _number(item["value"], "quantile value"),
            )
        )
    return DistributionMetrics(
        quantiles=tuple(quantiles),
        sign_probability=_number(mapping["sign_probability"], "sign_probability"),
        crps=_number(mapping["crps"], "crps"),
        energy_score=_number(mapping["energy_score"], "energy_score"),
        interval_lower=_number(mapping["interval_lower"], "interval_lower"),
        interval_upper=_number(mapping["interval_upper"], "interval_upper"),
        interval_score=_number(mapping["interval_score"], "interval_score"),
        empirical_coverage=_number(mapping["empirical_coverage"], "empirical_coverage"),
        wasserstein_distance=_number(
            mapping["wasserstein_distance"], "wasserstein_distance"
        ),
        tail_event_probability=_number(
            mapping["tail_event_probability"], "tail_event_probability"
        ),
        brier_score=_number(mapping["brier_score"], "brier_score"),
        intervals=tuple(
            _interval_from_mapping(item)
            for item in _array(mapping["intervals"], "intervals")
        ),
        interval_kind=_enum_value(
            IntervalKind, mapping["interval_kind"], "interval_kind"
        ),
        interval_method_id=_string(mapping["interval_method_id"], "interval_method_id"),
        estimand_id=_string(mapping["estimand_id"], "estimand_id"),
        sampling_unit=_string(mapping["sampling_unit"], "sampling_unit"),
        predictive_draw_count=_integer(
            mapping["predictive_draw_count"], "predictive_draw_count"
        ),
        empirical_draw_count=_integer(
            mapping["empirical_draw_count"], "empirical_draw_count"
        ),
        mean_absolute_error=_number(
            mapping["mean_absolute_error"], "mean_absolute_error"
        ),
        root_mean_squared_error=_number(
            mapping["root_mean_squared_error"], "root_mean_squared_error"
        ),
        direction_accuracy=_number(mapping["direction_accuracy"], "direction_accuracy"),
        worst_five_percent_mean=_number(
            mapping["worst_five_percent_mean"], "worst_five_percent_mean"
        ),
        maximum_mean_discrepancy_squared=_number(
            mapping["maximum_mean_discrepancy_squared"],
            "maximum_mean_discrepancy_squared",
        ),
        predictive_sample_artifact_sha256=_string(
            mapping["predictive_sample_artifact_sha256"],
            "predictive_sample_artifact_sha256",
        ),
        held_out_sample_artifact_sha256=_string(
            mapping["held_out_sample_artifact_sha256"],
            "held_out_sample_artifact_sha256",
        ),
    )


def _agent_outcome_to_mapping(outcome: AgentOutcome) -> dict[str, object]:
    return {
        "agent_id": outcome.agent_id,
        "cohort": outcome.cohort.to_mapping(),
        "crowded_out": outcome.crowded_out,
        "execution_cost": outcome.execution_cost,
        "fees_paid": outcome.fees_paid,
        "filled_quantity": outcome.filled_quantity,
        "inventory_path": list(outcome.inventory_path),
        "pnl": outcome.pnl,
        "submitted_quantity": outcome.submitted_quantity,
        "survived": outcome.survived,
        "taxes_paid": outcome.taxes_paid,
        "transfer_received": outcome.transfer_received,
        "utility_equivalent_change": outcome.utility_equivalent_change,
        "wealth_path": list(outcome.wealth_path),
    }


def _agent_outcome_from_mapping(value: object) -> AgentOutcome:
    mapping = _object(value, "agent outcome")
    _fields(mapping, frozenset(AgentOutcome.__dataclass_fields__), "agent outcome")
    cohort_mapping = _object(mapping["cohort"], "agent outcome cohort")
    try:
        cohort = FrozenCohort.from_mapping(cohort_mapping)
    except ValueError as error:
        raise PolicyEvaluationSchemaError("agent outcome cohort is invalid") from error
    return AgentOutcome(
        agent_id=_string(mapping["agent_id"], "agent_id"),
        cohort=cohort,
        pnl=_number(mapping["pnl"], "pnl"),
        execution_cost=_number(mapping["execution_cost"], "execution_cost"),
        submitted_quantity=_number(mapping["submitted_quantity"], "submitted_quantity"),
        filled_quantity=_number(mapping["filled_quantity"], "filled_quantity"),
        inventory_path=tuple(
            _number(item, "inventory path value")
            for item in _array(mapping["inventory_path"], "inventory_path")
        ),
        wealth_path=tuple(
            _number(item, "wealth path value")
            for item in _array(mapping["wealth_path"], "wealth_path")
        ),
        survived=_boolean(mapping["survived"], "survived"),
        utility_equivalent_change=_number(
            mapping["utility_equivalent_change"], "utility_equivalent_change"
        ),
        fees_paid=_number(mapping["fees_paid"], "fees_paid"),
        taxes_paid=_number(mapping["taxes_paid"], "taxes_paid"),
        transfer_received=_number(mapping["transfer_received"], "transfer_received"),
        crowded_out=_boolean(mapping["crowded_out"], "crowded_out"),
    )


def _group_metric_from_mapping(value: object) -> GroupMetrics:
    mapping = _object(value, "group metric")
    _fields(mapping, frozenset(GroupMetrics.__dataclass_fields__), "group metric")
    return GroupMetrics(
        group_id=_string(mapping["group_id"], "group_id"),
        cohort_basis=_enum_value(CohortBasis, mapping["cohort_basis"], "cohort_basis"),
        cohort_definition_sha256=_string(
            mapping["cohort_definition_sha256"], "cohort_definition_sha256"
        ),
        agent_count=_integer(mapping["agent_count"], "agent_count"),
        mean_pnl=_number(mapping["mean_pnl"], "mean_pnl"),
        mean_execution_cost=_number(
            mapping["mean_execution_cost"], "mean_execution_cost"
        ),
        fill_rate=_number(mapping["fill_rate"], "fill_rate"),
        mean_inventory_risk=_number(
            mapping["mean_inventory_risk"], "mean_inventory_risk"
        ),
        mean_drawdown=_number(mapping["mean_drawdown"], "mean_drawdown"),
        survival_probability=_number(
            mapping["survival_probability"], "survival_probability"
        ),
        mean_utility_equivalent_change=_number(
            mapping["mean_utility_equivalent_change"],
            "mean_utility_equivalent_change",
        ),
        mean_wealth_change=_number(mapping["mean_wealth_change"], "mean_wealth_change"),
        mean_return_to_rms_risk=_number(
            mapping["mean_return_to_rms_risk"], "mean_return_to_rms_risk"
        ),
        mean_fee_burden=_number(mapping["mean_fee_burden"], "mean_fee_burden"),
        mean_tax_burden=_number(mapping["mean_tax_burden"], "mean_tax_burden"),
        crowd_out_probability=_number(
            mapping["crowd_out_probability"], "crowd_out_probability"
        ),
        mean_transfer_received=_number(
            mapping["mean_transfer_received"], "mean_transfer_received"
        ),
        total_fees_paid=_number(mapping["total_fees_paid"], "total_fees_paid"),
        total_taxes_paid=_number(mapping["total_taxes_paid"], "total_taxes_paid"),
    )


@dataclass(frozen=True, slots=True)
class MarketEvaluationArtifact:
    """Versioned market metrics that must reproduce from retained inputs."""

    schema_version: int
    artifact_id: str
    data: MarketEvaluationInput
    config: MarketMetricConfig
    metrics: MarketMetrics
    source_authority: str = MARKET_OBSERVATION_SOURCE_AUTHORITY
    claim_scope: str = MARKET_EVALUATION_CLAIM_SCOPE
    formal_market_evidence: bool = False
    synthetic_only: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 3:
            _fail("market evaluation artifact schema_version must equal 3")
        if not self.artifact_id or not isinstance(self.artifact_id, str):
            _fail("market evaluation artifact_id must be a non-empty string")
        if not isinstance(self.data, MarketEvaluationInput):
            _fail("market evaluation artifact data must be typed")
        if not isinstance(self.config, MarketMetricConfig):
            _fail("market evaluation artifact config must be typed")
        if not isinstance(self.metrics, MarketMetrics):
            _fail("market evaluation artifact metrics must be typed")
        if self.source_authority != MARKET_OBSERVATION_SOURCE_AUTHORITY:
            _fail("market observation source authority cannot be caller-promoted")
        if self.claim_scope != MARKET_EVALUATION_CLAIM_SCOPE:
            _fail("market evaluation claim_scope cannot be caller-promoted")
        if self.formal_market_evidence is not False:
            _fail("caller-authored market observations are not formal evidence")
        if self.synthetic_only is not True:
            _fail(
                "non-synthetic market evaluation requires a future typed "
                "temporal-firewall authority"
            )
        expected = evaluate_market_metrics(self.data, self.config)
        if self.metrics != expected:
            _fail("market metrics do not reproduce from retained input and config")

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def to_mapping(self) -> dict[str, object]:
        return {
            "artifact_id": self.artifact_id,
            "claim_scope": self.claim_scope,
            "config": _market_config_to_mapping(self.config),
            "data": _market_input_to_mapping(self.data),
            "formal_market_evidence": self.formal_market_evidence,
            "metrics": self.metrics.to_mapping(),
            "schema_version": self.schema_version,
            "source_authority": self.source_authority,
            "synthetic_only": self.synthetic_only,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> MarketEvaluationArtifact:
        mapping = _object(value, "market evaluation artifact")
        _fields(
            mapping, frozenset(cls.__dataclass_fields__), "market evaluation artifact"
        )
        data = _market_input_from_mapping(mapping["data"])
        return cls(
            schema_version=_integer(mapping["schema_version"], "schema_version"),
            artifact_id=_string(mapping["artifact_id"], "artifact_id"),
            data=data,
            config=_market_config_from_mapping(mapping["config"]),
            metrics=_market_metrics_from_mapping(mapping["metrics"], data),
            source_authority=_string(mapping["source_authority"], "source_authority"),
            claim_scope=_string(mapping["claim_scope"], "claim_scope"),
            formal_market_evidence=_boolean(
                mapping["formal_market_evidence"], "formal_market_evidence"
            ),
            synthetic_only=_boolean(mapping["synthetic_only"], "synthetic_only"),
        )


@dataclass(frozen=True, slots=True)
class GroupEvaluationArtifact:
    """Reproducible synthetic group arithmetic with a non-authorizing source label."""

    schema_version: int
    artifact_id: str
    outcomes: tuple[AgentOutcome, ...]
    metrics: tuple[GroupMetrics, ...]
    source_authority: str = GROUP_OUTCOME_SOURCE_AUTHORITY
    claim_scope: str = GROUP_EVALUATION_CLAIM_SCOPE
    formal_welfare_evidence: bool = False
    synthetic_only: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            _fail("group evaluation artifact schema_version must equal 1")
        if not isinstance(self.artifact_id, str) or not self.artifact_id:
            _fail("group evaluation artifact_id must be a non-empty string")
        if (
            not isinstance(self.outcomes, tuple)
            or not self.outcomes
            or not all(isinstance(item, AgentOutcome) for item in self.outcomes)
        ):
            _fail("group evaluation outcomes must be a non-empty typed tuple")
        if tuple(item.agent_id for item in self.outcomes) != tuple(
            sorted(item.agent_id for item in self.outcomes)
        ):
            _fail("group evaluation outcomes must use canonical agent-ID order")
        if (
            not isinstance(self.metrics, tuple)
            or not self.metrics
            or not all(isinstance(item, GroupMetrics) for item in self.metrics)
        ):
            _fail("group evaluation metrics must be a non-empty typed tuple")
        if self.source_authority != GROUP_OUTCOME_SOURCE_AUTHORITY:
            _fail("group outcome source authority is not repository-verifiable")
        if self.claim_scope != GROUP_EVALUATION_CLAIM_SCOPE:
            _fail("group evaluation claim scope is not permitted")
        if self.formal_welfare_evidence is not False:
            _fail("caller-authored group outcomes cannot authorize welfare claims")
        if self.synthetic_only is not True:
            _fail(
                "non-synthetic group evaluation requires a future paired-ledger, "
                "cohort, and utility authority"
            )
        expected = evaluate_group_metrics(self.outcomes)
        if self.metrics != expected:
            _fail("group metrics do not reproduce from retained outcomes")

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def to_mapping(self) -> dict[str, object]:
        return {
            "artifact_id": self.artifact_id,
            "claim_scope": self.claim_scope,
            "formal_welfare_evidence": self.formal_welfare_evidence,
            "metrics": [item.to_mapping() for item in self.metrics],
            "outcomes": [_agent_outcome_to_mapping(item) for item in self.outcomes],
            "schema_version": self.schema_version,
            "source_authority": self.source_authority,
            "synthetic_only": self.synthetic_only,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> GroupEvaluationArtifact:
        mapping = _object(value, "group evaluation artifact")
        _fields(
            mapping,
            frozenset(cls.__dataclass_fields__),
            "group evaluation artifact",
        )
        return cls(
            schema_version=_integer(mapping["schema_version"], "schema_version"),
            artifact_id=_string(mapping["artifact_id"], "artifact_id"),
            outcomes=tuple(
                _agent_outcome_from_mapping(item)
                for item in _array(mapping["outcomes"], "outcomes")
            ),
            metrics=tuple(
                _group_metric_from_mapping(item)
                for item in _array(mapping["metrics"], "metrics")
            ),
            source_authority=_string(mapping["source_authority"], "source_authority"),
            claim_scope=_string(mapping["claim_scope"], "claim_scope"),
            formal_welfare_evidence=_boolean(
                mapping["formal_welfare_evidence"], "formal_welfare_evidence"
            ),
            synthetic_only=_boolean(mapping["synthetic_only"], "synthetic_only"),
        )


@dataclass(frozen=True, slots=True)
class DistributionEvaluationArtifact:
    """Versioned distribution scores that must reproduce from retained samples."""

    schema_version: int
    artifact_id: str
    predictive_sample: DistributionSampleArtifact
    held_out_sample: DistributionSampleArtifact
    config: DistributionMetricConfig
    metrics: DistributionMetrics
    synthetic_only: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            _fail("distribution evaluation artifact schema_version must equal 1")
        if not self.artifact_id or not isinstance(self.artifact_id, str):
            _fail("distribution evaluation artifact_id must be a non-empty string")
        if not isinstance(
            self.predictive_sample, DistributionSampleArtifact
        ) or not isinstance(self.held_out_sample, DistributionSampleArtifact):
            _fail("distribution evaluation samples must be typed")
        if not isinstance(self.config, DistributionMetricConfig):
            _fail("distribution evaluation config must be typed")
        if not isinstance(self.metrics, DistributionMetrics):
            _fail("distribution evaluation metrics must be typed")
        if self.synthetic_only is not True:
            _fail(
                "non-synthetic distribution evaluation requires a typed temporal "
                "and study-protocol authority"
            )
        expected = evaluate_distribution_metrics(
            self.predictive_sample, self.held_out_sample, self.config
        )
        if self.metrics != expected:
            _fail(
                "distribution metrics do not reproduce from retained samples and config"
            )

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def to_mapping(self) -> dict[str, object]:
        return {
            "artifact_id": self.artifact_id,
            "config": self.config.to_mapping(),
            "held_out_sample": self.held_out_sample.to_mapping(),
            "metrics": self.metrics.to_mapping(),
            "predictive_sample": self.predictive_sample.to_mapping(),
            "schema_version": self.schema_version,
            "synthetic_only": self.synthetic_only,
        }

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> DistributionEvaluationArtifact:
        mapping = _object(value, "distribution evaluation artifact")
        _fields(
            mapping,
            frozenset(cls.__dataclass_fields__),
            "distribution evaluation artifact",
        )
        return cls(
            schema_version=_integer(mapping["schema_version"], "schema_version"),
            artifact_id=_string(mapping["artifact_id"], "artifact_id"),
            predictive_sample=_distribution_sample_from_mapping(
                mapping["predictive_sample"]
            ),
            held_out_sample=_distribution_sample_from_mapping(
                mapping["held_out_sample"]
            ),
            config=_distribution_config_from_mapping(mapping["config"]),
            metrics=_distribution_metrics_from_mapping(mapping["metrics"]),
            synthetic_only=_boolean(mapping["synthetic_only"], "synthetic_only"),
        )


def build_market_evaluation_artifact(
    artifact_id: str,
    data: MarketEvaluationInput,
    config: MarketMetricConfig,
) -> MarketEvaluationArtifact:
    """Compute and seal one synthetic market metric artifact."""
    return MarketEvaluationArtifact(
        schema_version=3,
        artifact_id=artifact_id,
        data=data,
        config=config,
        metrics=evaluate_market_metrics(data, config),
        synthetic_only=True,
    )


def build_distribution_evaluation_artifact(
    artifact_id: str,
    predictive_sample: DistributionSampleArtifact,
    held_out_sample: DistributionSampleArtifact,
    config: DistributionMetricConfig,
) -> DistributionEvaluationArtifact:
    """Compute and seal one synthetic distribution-scoring artifact."""
    return DistributionEvaluationArtifact(
        schema_version=1,
        artifact_id=artifact_id,
        predictive_sample=predictive_sample,
        held_out_sample=held_out_sample,
        config=config,
        metrics=evaluate_distribution_metrics(
            predictive_sample, held_out_sample, config
        ),
        synthetic_only=True,
    )


def build_group_evaluation_artifact(
    artifact_id: str,
    outcomes: tuple[AgentOutcome, ...],
) -> GroupEvaluationArtifact:
    """Seal synthetic group arithmetic without upgrading its evidence authority."""
    canonical_outcomes = tuple(sorted(outcomes, key=lambda item: item.agent_id))
    return GroupEvaluationArtifact(
        schema_version=1,
        artifact_id=artifact_id,
        outcomes=canonical_outcomes,
        metrics=evaluate_group_metrics(canonical_outcomes),
    )


__all__ = [
    "DistributionEvaluationArtifact",
    "GROUP_EVALUATION_CLAIM_SCOPE",
    "GROUP_OUTCOME_SOURCE_AUTHORITY",
    "GroupEvaluationArtifact",
    "MarketEvaluationArtifact",
    "build_distribution_evaluation_artifact",
    "build_group_evaluation_artifact",
    "build_market_evaluation_artifact",
]
