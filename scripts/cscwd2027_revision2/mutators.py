"""Explicit common-mode TCB stress hooks, used only in in-process diagnostics."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from typing import cast
from unittest.mock import patch

from policytwin.coordination import workload
from policytwin.coordination.schema import AgentView
from policytwin.market.models import ReplayResult, Side
from policytwin.market.session import ReplayMarketSession

MUTANTS = (
    "quantity_rotation",
    "side_flip",
    "omit_balance_feedback",
    "trade_price_shift",
)


@contextmanager
def common_mode(kind: str | None) -> Iterator[dict[str, int]]:
    """Apply the same defect during execution and replay, then restore functions."""
    calls = {"calls": 0}
    if kind is None:
        yield calls
        return
    if kind not in MUTANTS:
        raise ValueError("unknown common-mode defect")
    original = workload.action
    original_property = cast(property, vars(ReplayMarketSession)["replay"])

    def bad_action(view: AgentView, side: int, quantity: int) -> tuple[Side, int]:
        calls["calls"] += 1
        selected, size = original(view, side, quantity)
        if kind == "quantity_rotation":
            return selected, 1 + size % 9
        if kind == "side_flip":
            return Side.SELL if selected is Side.BUY else Side.BUY, size
        signal = (
            3 * view.bid_quantity
            + 5 * view.ask_quantity
            + view.best_bid
            + view.best_ask
        )
        return Side.BUY if (side + signal) % 2 == 0 else Side.SELL, 1 + (
            quantity + signal
        ) % 9

    def bad_trades(session: ReplayMarketSession) -> ReplayResult:
        calls["calls"] += 1
        getter = original_property.fget
        if getter is None:
            raise RuntimeError("replay getter unavailable")
        value = cast(ReplayResult, getter(session))
        return replace(
            value, trades=tuple(replace(t, price=t.price + 1) for t in value.trades)
        )

    if kind == "trade_price_shift":
        with patch.object(ReplayMarketSession, "replay", property(bad_trades)):
            yield calls
    else:
        with patch.object(workload, "action", bad_action):
            yield calls
