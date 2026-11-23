"""Explicit opening-liquidity recipes over the unchanged integer CLOB runtime."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Self
from unittest.mock import patch

from policytwin.coordination import auditor, coordinator, replay, workload
from policytwin.coordination.plan import RunSpec, compile_contract
from policytwin.coordination.release_gate import implementation_sha256
from policytwin.coordination.schema import commitment
from policytwin.market.models import MarketConfig, OpeningState

METRIC = "policy-minus-baseline total traded notional; integer price times quantity"


def source_identity() -> str:
    """Bind inherited scientific code and every new runtime/collector source byte."""
    paths = sorted(Path(__file__).parent.glob("*.py"))
    return commitment(
        {
            "inherited": implementation_sha256(),
            "comparison_adapter": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
            },
        }
    )


@dataclass(frozen=True, slots=True)
class SimulationRecipe:
    """A complete declared scientific run, independent of permitted worker plans."""

    spec: RunSpec
    liquidity_per_level: int

    def __post_init__(self) -> None:
        if type(self.liquidity_per_level) is not int or self.liquidity_per_level < 1:
            raise ValueError("liquidity must be a positive integer")

    def to_mapping(self) -> dict[str, Any]:
        """Serialize the added scientific configuration explicitly."""
        return {
            "spec": self.spec.to_mapping(),
            "liquidity_per_level": self.liquidity_per_level,
        }

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> Self:
        """Parse a closed recipe schema."""
        if set(value) != {"spec", "liquidity_per_level"}:
            raise ValueError("unknown recipe fields")
        return cls(RunSpec.from_mapping(value["spec"]), value["liquidity_per_level"])

    def opening(self) -> OpeningState:
        """Change only declared external book depth; retain the target resting order."""
        value = workload.opening_state(self.spec)
        return replace(
            value,
            orders=tuple(
                replace(order, quantity=self.liquidity_per_level)
                if order.trader_id in {"book-bid", "book-ask"}
                else order
                for order in value.orders
            ),
        )

    def authority(self) -> dict[str, str]:
        """Pre-execution binding additional to the unmodified v1 Registration."""
        return {
            "recipe_sha256": commitment(self.to_mapping()),
            "opening_sha256": commitment(self.opening().to_mapping()),
            "source_sha256": source_identity(),
            "metric": METRIC,
        }

    @property
    def scientific_key(self) -> str:
        """Exact declared scientific equivalence, not arbitrary program equivalence."""
        return commitment(
            {
                **self.authority(),
                "contract_sha256": compile_contract(self.spec).sha256,
                "market_config": MarketConfig(tick_size=1).to_mapping(),
            }
        )


@contextmanager
def configured(recipe: SimulationRecipe) -> Iterator[None]:
    """Scope the same opening state to execution, replay AND conformance checks.

    Frozen v1 source files are not edited. Worker children read the serialized
    immutable views, not this factory. The parent is single-threaded; separate
    runner shards are separate OS processes. All aliases are restored on exit.
    """

    def opening(spec: RunSpec) -> OpeningState:
        if spec != recipe.spec:
            raise ValueError("recipe/request scientific configuration differs")
        return recipe.opening()

    with ExitStack() as stack:
        for module in (coordinator, auditor, replay):
            stack.enter_context(patch.object(module, "opening_state", opening))
        yield
