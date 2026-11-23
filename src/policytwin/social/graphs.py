"""Immutable, canonical weighted directed social graphs."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal, Protocol, runtime_checkable

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class SocialGraphError(ValueError):
    """Raised when a social graph is malformed or temporally unavailable."""


class SocialGraphTemporalError(SocialGraphError):
    """Raised when an observed graph crosses a knowledge-time boundary."""


class SocialGraphKind(StrEnum):
    """Supported graph provenance classes."""

    OBSERVED = "observed"
    SYNTHETIC = "synthetic"


def _identifier(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise SocialGraphError(f"{field_name} must be a stable anonymous identifier")
    return value


def _aware_utc(value: object, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise SocialGraphError(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise SocialGraphError(f"{field_name} must include an explicit timezone")
    try:
        return value.astimezone(UTC)
    except (OverflowError, ValueError) as error:
        raise SocialGraphError(f"{field_name} is outside the UTC range") from error


def _sha256(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise SocialGraphError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class WeightedDirectedEdge:
    """A normalized directed influence edge with weight in ``(0, 1]``."""

    source_id: str
    target_id: str
    weight: float

    def __post_init__(self) -> None:
        source_id = _identifier(self.source_id, "source_id")
        target_id = _identifier(self.target_id, "target_id")
        if source_id == target_id:
            raise SocialGraphError("self-loop edges are forbidden")
        if isinstance(self.weight, bool) or not isinstance(self.weight, int | float):
            raise SocialGraphError("edge weight must be a finite real number")
        normalized_weight = float(self.weight)
        if not math.isfinite(normalized_weight) or not 0.0 < normalized_weight <= 1.0:
            raise SocialGraphError("edge weight must be finite and in (0, 1]")
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "target_id", target_id)
        object.__setattr__(self, "weight", normalized_weight)

    def to_mapping(self) -> dict[str, object]:
        """Return the canonical edge representation."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "weight": self.weight,
        }


def _canonical_graph(
    nodes: object, edges: object
) -> tuple[tuple[str, ...], tuple[WeightedDirectedEdge, ...]]:
    if not isinstance(nodes, tuple):
        raise SocialGraphError("nodes must be an immutable tuple")
    normalized_nodes = tuple(_identifier(node, "node_id") for node in nodes)
    if len(set(normalized_nodes)) != len(normalized_nodes):
        raise SocialGraphError("duplicate node identifiers are forbidden")
    canonical_nodes = tuple(sorted(normalized_nodes))
    node_set = frozenset(canonical_nodes)

    if not isinstance(edges, tuple):
        raise SocialGraphError("edges must be an immutable tuple")
    normalized_edges: list[WeightedDirectedEdge] = []
    seen_pairs: set[tuple[str, str]] = set()
    for edge in edges:
        if not isinstance(edge, WeightedDirectedEdge):
            raise SocialGraphError("every edge must be a WeightedDirectedEdge")
        pair = (edge.source_id, edge.target_id)
        if pair in seen_pairs:
            raise SocialGraphError(
                "duplicate directed edge is forbidden: "
                f"{edge.source_id}->{edge.target_id}"
            )
        if edge.source_id not in node_set or edge.target_id not in node_set:
            raise SocialGraphError("every edge endpoint must appear in nodes")
        seen_pairs.add(pair)
        normalized_edges.append(edge)
    canonical_edges = tuple(
        sorted(
            normalized_edges,
            key=lambda edge: (edge.source_id, edge.target_id, edge.weight),
        )
    )
    return canonical_nodes, canonical_edges


def _outgoing_edges(
    nodes: tuple[str, ...],
    edges: tuple[WeightedDirectedEdge, ...],
    node_id: str,
) -> tuple[WeightedDirectedEdge, ...]:
    normalized_node_id = _identifier(node_id, "node_id")
    if normalized_node_id not in nodes:
        raise SocialGraphError(f"unknown node_id: {normalized_node_id}")
    return tuple(edge for edge in edges if edge.source_id == normalized_node_id)


@runtime_checkable
class WeightedDirectedGraph(Protocol):
    """Read-only weighted directed structure independent of provenance."""

    @property
    def nodes(self) -> tuple[str, ...]:
        """Return canonical anonymous node identifiers."""
        ...

    @property
    def edges(self) -> tuple[WeightedDirectedEdge, ...]:
        """Return canonical weighted directed edges."""
        ...

    def outgoing_edges(self, node_id: str) -> tuple[WeightedDirectedEdge, ...]:
        """Return canonical outgoing edges for one node."""
        ...

    def to_mapping(self) -> dict[str, object]:
        """Return a JSON-compatible canonical representation."""
        ...


@runtime_checkable
class SocialGraph(WeightedDirectedGraph, Protocol):
    """Weighted directed graph with observed or synthetic provenance."""

    @property
    def graph_kind(self) -> SocialGraphKind:
        """Return whether the graph is observed or synthetic."""
        ...


@dataclass(frozen=True, slots=True)
class ObservedSocialGraph:
    """An observed graph proven available no later than a trusted cutoff."""

    nodes: tuple[str, ...]
    edges: tuple[WeightedDirectedEdge, ...]
    available_time: datetime
    trusted_cutoff: datetime
    source_manifest_sha256: str
    lineage_sha256: str
    cutoff_policy_sha256: str

    def __post_init__(self) -> None:
        canonical_nodes, canonical_edges = _canonical_graph(self.nodes, self.edges)
        available_time = _aware_utc(self.available_time, "available_time")
        trusted_cutoff = _aware_utc(self.trusted_cutoff, "trusted_cutoff")
        if available_time > trusted_cutoff:
            raise SocialGraphTemporalError(
                "observed graph available_time is after trusted_cutoff"
            )
        object.__setattr__(self, "nodes", canonical_nodes)
        object.__setattr__(self, "edges", canonical_edges)
        object.__setattr__(self, "available_time", available_time)
        object.__setattr__(self, "trusted_cutoff", trusted_cutoff)
        object.__setattr__(
            self,
            "source_manifest_sha256",
            _sha256(self.source_manifest_sha256, "source_manifest_sha256"),
        )
        object.__setattr__(
            self,
            "lineage_sha256",
            _sha256(self.lineage_sha256, "lineage_sha256"),
        )
        object.__setattr__(
            self,
            "cutoff_policy_sha256",
            _sha256(self.cutoff_policy_sha256, "cutoff_policy_sha256"),
        )

    @property
    def graph_kind(self) -> SocialGraphKind:
        return SocialGraphKind.OBSERVED

    def validate_available(self, *, as_of_time: datetime) -> None:
        """Fail closed unless this graph was available at the requested time."""
        normalized_as_of = _aware_utc(as_of_time, "as_of_time")
        if normalized_as_of > self.trusted_cutoff:
            raise SocialGraphTemporalError("as_of_time is after trusted_cutoff")
        if self.available_time > normalized_as_of:
            raise SocialGraphTemporalError(
                "observed graph was not available at as_of_time"
            )

    def outgoing_edges(self, node_id: str) -> tuple[WeightedDirectedEdge, ...]:
        """Return canonical outgoing influence edges for one node."""
        return _outgoing_edges(self.nodes, self.edges, node_id)

    def to_mapping(self) -> dict[str, object]:
        """Return graph structure and complete temporal provenance."""
        return {
            "available_time": self.available_time.isoformat(),
            "cutoff_policy_sha256": self.cutoff_policy_sha256,
            "edges": [edge.to_mapping() for edge in self.edges],
            "graph_kind": self.graph_kind.value,
            "lineage_sha256": self.lineage_sha256,
            "nodes": list(self.nodes),
            "source_manifest_sha256": self.source_manifest_sha256,
            "trusted_cutoff": self.trusted_cutoff.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class SyntheticSocialGraph:
    """A synthetic graph that cannot be mistaken for calibrated evidence."""

    nodes: tuple[str, ...]
    edges: tuple[WeightedDirectedEdge, ...]
    synthetic_uncalibrated: Literal[True] = True

    def __post_init__(self) -> None:
        if self.synthetic_uncalibrated is not True:
            raise SocialGraphError(
                "synthetic graph must be marked synthetic_uncalibrated"
            )
        canonical_nodes, canonical_edges = _canonical_graph(self.nodes, self.edges)
        object.__setattr__(self, "nodes", canonical_nodes)
        object.__setattr__(self, "edges", canonical_edges)

    @property
    def graph_kind(self) -> SocialGraphKind:
        return SocialGraphKind.SYNTHETIC

    def outgoing_edges(self, node_id: str) -> tuple[WeightedDirectedEdge, ...]:
        """Return canonical outgoing influence edges for one node."""
        return _outgoing_edges(self.nodes, self.edges, node_id)

    def to_mapping(self) -> dict[str, object]:
        """Return structure with an explicit non-empirical marker."""
        return {
            "edges": [edge.to_mapping() for edge in self.edges],
            "graph_kind": self.graph_kind.value,
            "nodes": list(self.nodes),
            "synthetic_uncalibrated": self.synthetic_uncalibrated,
        }


__all__ = [
    "ObservedSocialGraph",
    "SocialGraph",
    "SocialGraphError",
    "SocialGraphKind",
    "SocialGraphTemporalError",
    "SyntheticSocialGraph",
    "WeightedDirectedEdge",
    "WeightedDirectedGraph",
]
