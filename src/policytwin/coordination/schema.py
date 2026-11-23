"""Closed wire records shared by coordinator, workers and verifier."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Self, cast

from policytwin.provenance import canonical_json_bytes, sha256_bytes

SCHEMA = "policytwin.coordination.v1"
WORLDS = ("baseline", "policy")


class CoordinationError(ValueError):
    """A malformed record or unsatisfied protocol precondition."""


def object_fields(value: object, fields: set[str]) -> dict[str, Any]:
    """Require exactly the fields in a closed JSON object."""
    if not isinstance(value, dict) or set(value) != fields:
        raise CoordinationError(f"closed fields required: {sorted(fields)}")
    try:
        canonical_json_bytes(value)
    except (TypeError, ValueError) as error:
        raise CoordinationError("wire object is not canonical JSON") from error
    return cast(dict[str, Any], value)


def integer(value: object, label: str, minimum: int = 0) -> int:
    """Validate an exact non-Boolean integer."""
    if type(value) is not int or value < minimum:
        raise CoordinationError(f"invalid {label}")
    return value


def identifier(value: object, label: str) -> str:
    """Validate stable ASCII protocol identifiers."""
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", value) is None
    ):
        raise CoordinationError(f"invalid {label}")
    return value


def commitment(value: object) -> str:
    """Hash the canonical JSON encoding used by the existing repository."""
    return sha256_bytes(canonical_json_bytes(value))


def sha256(value: object, label: str) -> str:
    """Require a lower-case SHA-256 commitment."""
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise CoordinationError(f"invalid {label}")
    return value


@dataclass(frozen=True, slots=True)
class AgentView:
    """Sufficient view for the inherited feedback rule, plus full state binding."""

    state_sha256: str
    best_bid: int
    best_ask: int
    bid_quantity: int
    ask_quantity: int
    cash: int
    asset: int

    def __post_init__(self) -> None:
        sha256(self.state_sha256, "state_sha256")
        for name in self.__dataclass_fields__:
            if name != "state_sha256":
                integer(getattr(self, name), name)

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        """Parse a closed snapshot projection."""
        return cls(**object_fields(value, set(cls.__dataclass_fields__)))

    def to_mapping(self) -> dict[str, Any]:
        """Serialize the exact worker-visible view."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TaskRequest:
    """One immutable task dispatched from an independently compiled plan."""

    run_id: str
    task_id: str
    world: str
    epoch: int
    agent: str
    plan_sha256: str
    view: AgentView
    schema_version: str = SCHEMA

    def __post_init__(self) -> None:
        for name in ("run_id", "task_id", "agent"):
            identifier(getattr(self, name), name)
        if self.world not in WORLDS or self.schema_version != SCHEMA:
            raise CoordinationError("invalid task world or schema")
        integer(self.epoch, "epoch")
        sha256(self.plan_sha256, "plan_sha256")
        if not isinstance(self.view, AgentView):
            raise CoordinationError("task requires a validated view")

    def to_mapping(self) -> dict[str, Any]:
        """Return the closed task wire object."""
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        """Reject unknown fields before constructing a worker request."""
        obj = dict(object_fields(value, set(cls.__dataclass_fields__)))
        obj["view"] = AgentView.from_mapping(obj["view"])
        return cls(**obj)


@dataclass(frozen=True, slots=True)
class TaskResult:
    """A worker response; ownership and commitments are checked by the auditor."""

    run_id: str
    task_id: str
    message_id: str
    world: str
    epoch: int
    agent: str
    worker_id: int
    worker_pid: int
    plan_sha256: str
    view_sha256: str
    status: str
    draws: tuple[dict[str, Any], ...]
    commands: tuple[dict[str, Any], ...]
    payload_sha256: str
    schema_version: str = SCHEMA

    def __post_init__(self) -> None:
        for name in ("run_id", "task_id", "message_id", "agent"):
            identifier(getattr(self, name), name)
        if self.world not in WORLDS or self.schema_version != SCHEMA:
            raise CoordinationError("invalid result world or schema")
        if self.status not in ("active", "inactive"):
            raise CoordinationError("invalid task result status")
        for name in ("epoch", "worker_id", "worker_pid"):
            integer(getattr(self, name), name)
        for name in ("plan_sha256", "view_sha256", "payload_sha256"):
            sha256(getattr(self, name), name)
        for draw in self.draws:
            row = object_fields(
                draw, {"event_identity", "address", "state", "value", "world"}
            )
            identifier(row["event_identity"], "event_identity")
            if row["world"] not in WORLDS or row["state"] not in ("consumed", "masked"):
                raise CoordinationError("invalid draw state or world")
            if row["state"] == "consumed":
                integer(row["value"], "draw value")
            elif row["value"] is not None:
                raise CoordinationError("masked draw has a value")
            if not isinstance(row["address"], dict):
                raise CoordinationError("draw address must be an object")
        for command in self.commands:
            canonical_json_bytes(command)

    def payload(self) -> dict[str, Any]:
        """Content covered by the message commitment (PID is diagnostic only)."""
        obj = asdict(self)
        obj.pop("payload_sha256")
        obj.pop("worker_pid")
        return obj

    def to_mapping(self) -> dict[str, Any]:
        """Serialize a complete worker result."""
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        """Parse a result without silently correcting an invalid commitment."""
        obj = dict(object_fields(value, set(cls.__dataclass_fields__)))
        for name in ("draws", "commands"):
            if not isinstance(obj[name], list | tuple):
                raise CoordinationError(f"{name} must be an array")
            obj[name] = tuple(obj[name])
        return cls(**obj)
