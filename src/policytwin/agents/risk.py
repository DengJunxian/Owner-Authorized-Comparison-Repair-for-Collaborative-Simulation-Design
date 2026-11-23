"""Atomic pre-trade risk gateway for structured agent actions."""

from __future__ import annotations

from dataclasses import dataclass

from policytwin.agents.models import (
    ConstraintCheck,
    ConstraintCode,
    DecisionAction,
    OrderKind,
    OrderPolicyConfig,
    OwnPortfolioState,
    PublicMarketObservation,
    RiskLimits,
)


@dataclass(frozen=True, slots=True)
class RiskAuthorization:
    """Maximum safe quantities and executable reference prices by side."""

    buy_quantity: int
    sell_quantity: int
    buy_price: int
    sell_price: int
    buy_limit_price: int
    sell_limit_price: int
    checks: tuple[ConstraintCheck, ...]


def _floor_tick(price: int, tick_size: int) -> int:
    return max(tick_size, price - price % tick_size)


def _ceil_tick(price: int, tick_size: int) -> int:
    remainder = price % tick_size
    return price if remainder == 0 else price + tick_size - remainder


def _max_affordable_quantity(*, cash: int, price: int, fee_bps: int, upper: int) -> int:
    """Find the exact largest quantity affordable under integer fee accounting."""
    lower = 0
    high = upper
    while lower < high:
        candidate = (lower + high + 1) // 2
        notional = price * candidate
        total = notional + notional * fee_bps // 10_000
        if total <= cash:
            lower = candidate
        else:
            high = candidate - 1
    return lower


class RiskGateway:
    """Apply cash, inventory, reservation, notional, and order-count limits."""

    def authorize(
        self,
        *,
        portfolio: OwnPortfolioState,
        observation: PublicMarketObservation,
        policy: OrderPolicyConfig,
        limits: RiskLimits,
        order_kind: OrderKind,
    ) -> RiskAuthorization:
        """Return side capacities without mutating any account or market state."""
        buy_limit_base = observation.best_bid or observation.mid_price
        sell_limit_base = observation.best_ask or observation.mid_price
        buy_limit = _floor_tick(
            max(
                policy.tick_size,
                buy_limit_base - policy.limit_offset_ticks * policy.tick_size,
            ),
            policy.tick_size,
        )
        sell_limit = _ceil_tick(
            sell_limit_base + policy.limit_offset_ticks * policy.tick_size,
            policy.tick_size,
        )
        if order_kind is OrderKind.LIMIT:
            buy_reference = buy_limit
            sell_reference = sell_limit
        else:
            buy_reference = observation.best_ask or _ceil_tick(
                observation.mid_price, policy.tick_size
            )
            sell_reference = observation.best_bid or _floor_tick(
                observation.mid_price, policy.tick_size
            )

        requested = policy.base_order_quantity
        buy_quantity = requested
        sell_quantity = requested
        checks: list[ConstraintCheck] = []

        def cap(
            *,
            action: DecisionAction,
            current: int,
            maximum: int,
            code: ConstraintCode,
            detail: str,
            force_record: bool = False,
        ) -> int:
            permitted = min(current, max(0, maximum))
            if permitted < current or force_record:
                checks.append(
                    ConstraintCheck(
                        code=code,
                        action=action,
                        requested_quantity=requested,
                        permitted_quantity=permitted,
                        detail=detail,
                    )
                )
            return permitted

        buy_quantity = cap(
            action=DecisionAction.BUY,
            current=buy_quantity,
            maximum=limits.max_order_quantity,
            code=ConstraintCode.ORDER_QUANTITY_LIMIT,
            detail="buy quantity capped by max_order_quantity",
        )
        sell_quantity = cap(
            action=DecisionAction.SELL,
            current=sell_quantity,
            maximum=limits.max_order_quantity,
            code=ConstraintCode.ORDER_QUANTITY_LIMIT,
            detail="sell quantity capped by max_order_quantity",
        )
        buy_quantity = cap(
            action=DecisionAction.BUY,
            current=buy_quantity,
            maximum=limits.max_order_notional // buy_reference,
            code=ConstraintCode.ORDER_NOTIONAL_LIMIT,
            detail="buy quantity capped by max_order_notional",
        )
        sell_quantity = cap(
            action=DecisionAction.SELL,
            current=sell_quantity,
            maximum=limits.max_order_notional // sell_reference,
            code=ConstraintCode.ORDER_NOTIONAL_LIMIT,
            detail="sell quantity capped by max_order_notional",
        )

        remaining_position = (
            limits.max_inventory - portfolio.inventory - portfolio.pending_buy_quantity
        )
        buy_quantity = cap(
            action=DecisionAction.BUY,
            current=buy_quantity,
            maximum=remaining_position,
            code=ConstraintCode.POSITION_LIMIT,
            detail="buy quantity capped by inventory plus pending buys",
        )
        sell_quantity = cap(
            action=DecisionAction.SELL,
            current=sell_quantity,
            maximum=portfolio.available_inventory,
            code=ConstraintCode.INVENTORY_AVAILABLE,
            detail="sell quantity capped by unreserved inventory",
        )

        spendable_cash = max(0, portfolio.available_cash - limits.cash_buffer)
        affordable = _max_affordable_quantity(
            cash=spendable_cash,
            price=buy_reference,
            fee_bps=limits.fee_bps,
            upper=buy_quantity,
        )
        buy_quantity = cap(
            action=DecisionAction.BUY,
            current=buy_quantity,
            maximum=affordable,
            code=ConstraintCode.CASH_BUDGET,
            detail="buy quantity capped by unreserved cash, buffer, and fees",
        )

        if portfolio.open_order_count >= limits.max_open_orders:
            buy_quantity = cap(
                action=DecisionAction.BUY,
                current=buy_quantity,
                maximum=0,
                code=ConstraintCode.OPEN_ORDER_LIMIT,
                detail="buy blocked by max_open_orders",
                force_record=True,
            )
            sell_quantity = cap(
                action=DecisionAction.SELL,
                current=sell_quantity,
                maximum=0,
                code=ConstraintCode.OPEN_ORDER_LIMIT,
                detail="sell blocked by max_open_orders",
                force_record=True,
            )

        if order_kind is OrderKind.MARKET:
            buy_quantity = cap(
                action=DecisionAction.BUY,
                current=buy_quantity,
                maximum=0,
                code=ConstraintCode.MARKET_ORDER_DISABLED,
                detail="market buy blocked without a committed depth bound",
                force_record=True,
            )
            sell_quantity = cap(
                action=DecisionAction.SELL,
                current=sell_quantity,
                maximum=0,
                code=ConstraintCode.MARKET_ORDER_DISABLED,
                detail="market sell blocked without a committed depth bound",
                force_record=True,
            )

        return RiskAuthorization(
            buy_quantity=buy_quantity,
            sell_quantity=sell_quantity,
            buy_price=buy_reference,
            sell_price=sell_reference,
            buy_limit_price=buy_limit,
            sell_limit_price=sell_limit,
            checks=tuple(checks),
        )


__all__ = ["RiskAuthorization", "RiskGateway"]
