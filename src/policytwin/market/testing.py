"""Scripted synthetic participant for order-book tests and smoke scenarios."""

from __future__ import annotations

from dataclasses import dataclass

from policytwin.counterfactual import EventKey
from policytwin.market.models import CancelOrder, LimitOrder, MarketOrder, Side


@dataclass(frozen=True, slots=True)
class ScriptedTestAgent:
    """Create commands from explicit test instructions without strategy logic."""

    agent_id: str
    experiment_id: str
    seed: str
    replicate_id: int = 0

    def event_key(self, *, step: int, purpose: str) -> EventKey:
        """Create one semantic event key for a scripted instruction."""
        return EventKey(
            experiment_id=self.experiment_id,
            seed=self.seed,
            replicate_id=self.replicate_id,
            simulation_step=step,
            purpose=f"test_agent:{self.agent_id}:{purpose}",
        )

    def limit_order(
        self,
        *,
        step: int,
        submitted_at: int,
        order_id: str,
        side: Side,
        quantity: int,
        price: int,
    ) -> LimitOrder:
        """Build an explicit synthetic limit-order command."""
        return LimitOrder(
            event_key=self.event_key(step=step, purpose=f"limit:{order_id}"),
            submitted_at=submitted_at,
            order_id=order_id,
            trader_id=self.agent_id,
            side=side,
            quantity=quantity,
            price=price,
        )

    def market_order(
        self,
        *,
        step: int,
        submitted_at: int,
        order_id: str,
        side: Side,
        quantity: int,
    ) -> MarketOrder:
        """Build an explicit synthetic immediate-or-cancel market command."""
        return MarketOrder(
            event_key=self.event_key(step=step, purpose=f"market:{order_id}"),
            submitted_at=submitted_at,
            order_id=order_id,
            trader_id=self.agent_id,
            side=side,
            quantity=quantity,
        )

    def cancel_order(
        self,
        *,
        step: int,
        submitted_at: int,
        target_order_id: str,
        instruction_id: str,
    ) -> CancelOrder:
        """Build an explicit cancellation with a separately named instruction."""
        return CancelOrder(
            event_key=self.event_key(step=step, purpose=f"cancel:{instruction_id}"),
            submitted_at=submitted_at,
            trader_id=self.agent_id,
            target_order_id=target_order_id,
        )


__all__ = ["ScriptedTestAgent"]
