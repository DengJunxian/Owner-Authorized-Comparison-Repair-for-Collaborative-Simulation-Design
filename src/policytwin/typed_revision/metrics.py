"""Closed pure metrics over a complete, locally verified pair capsule."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self

from policytwin.coordination.schema import commitment


@dataclass(frozen=True, slots=True)
class MetricSpec:
    """A metric's units and inclusion window are part of scientific identity."""

    quantity: str
    first_epoch: int = 0

    def __post_init__(self) -> None:
        if self.quantity not in {"notional", "volume"}:
            raise ValueError("unsupported metric quantity")
        if type(self.first_epoch) is not int or self.first_epoch < 0:
            raise ValueError("invalid metric epoch")

    def to_mapping(self) -> dict[str, Any]:
        return {"schema": "paired-trade-sum.v1", **asdict(self)}

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> Self:
        if set(value) != {"schema", "quantity", "first_epoch"}:
            raise ValueError("metric fields differ")
        if value["schema"] != "paired-trade-sum.v1":
            raise ValueError("unknown metric schema")
        return cls(value["quantity"], value["first_epoch"])


def implementation_hash() -> str:
    """Version the evaluator independently of simulator source."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def evaluate(capsule: dict[str, Any], metric: MetricSpec) -> dict[str, int]:
    """Compute world sums and their paired difference without changing dynamics.

    Each trade is assigned to its causing command's recorded epoch, rather than
    inferring an epoch from a numeric timestamp. Opening events have no trades
    in this workload. Unknown causes and incomplete traces are rejected.
    """
    horizon = capsule["spec"]["epoch_count"]
    if metric.first_epoch >= horizon:
        raise ValueError("metric window lies outside the recorded trajectory")
    values = {}
    if set(capsule["worlds"]) != {"baseline", "policy"}:
        raise ValueError("both worlds are required")
    for name, world in capsule["worlds"].items():
        if not world["completed"] or len(world["epochs"]) != horizon:
            raise ValueError("metric requires a complete trajectory")
        causes = {}
        for epoch in world["epochs"]:
            for command in epoch["applied_commands"]:
                key = command["event_key"]["value"]
                if key in causes:
                    raise ValueError("duplicate command cause")
                causes[key] = epoch["epoch"]
        total = 0
        for trade in world["trades"]:
            if trade["cause_event_key"] not in causes:
                raise ValueError("trade is not bound to a recorded command")
            if causes[trade["cause_event_key"]] < metric.first_epoch:
                continue
            if type(trade["quantity"]) is not int or trade["quantity"] < 1:
                raise ValueError("invalid trade quantity")
            if type(trade["price"]) is not int or trade["price"] < 1:
                raise ValueError("invalid trade price")
            total += trade["quantity"] * (
                trade["price"] if metric.quantity == "notional" else 1
            )
        if (
            metric.quantity == "notional"
            and metric.first_epoch == 0
            and total != world["traded_notional"]
        ):
            raise ValueError("trade projection and native notional differ")
        values[name] = total
    return {**values, "contrast": values["policy"] - values["baseline"]}


def evaluation_key(trajectory_key: str, metric: MetricSpec) -> str:
    """Bind a pure evaluation rule and implementation to a scientific trajectory."""
    return commitment(
        {
            "trajectory_key": trajectory_key,
            "metric": metric.to_mapping(),
            "metric_source_sha256": implementation_hash(),
        }
    )
