"""Immutable schemas for synthetic simulation-based calibration.

The first calibration backend is deliberately synthetic-only.  Real inputs
must later arrive through a temporal-firewall-issued capability; accepting an
arbitrary mapping here would make the policy-post exclusion unauditable.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import NoReturn

from policytwin.data.temporal_firewall import DataPartition, TemporalRecord
from policytwin.provenance import canonical_json_bytes, sha256_bytes

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EVENT_KEY = re.compile(r"^ptk1:[0-9a-f]{64}$")


class CalibrationError(ValueError):
    """Raised when calibration would violate a scientific invariant."""


def _fail(message: str) -> NoReturn:
    raise CalibrationError(message)


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        _fail(f"{field} must be a stable ASCII identifier")
    return value


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        _fail(f"{field} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{field} must be finite")
    return result


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _fail(f"{field} must be a positive integer")
    return value


def _aware_utc(value: object, field: str) -> datetime:
    if not isinstance(value, datetime):
        _fail(f"{field} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        _fail(f"{field} must include an explicit timezone")
    return value.astimezone(UTC)


def _named_values(
    values: tuple[tuple[str, float], ...], field: str
) -> tuple[tuple[str, float], ...]:
    if not isinstance(values, tuple) or not values:
        _fail(f"{field} must be a non-empty immutable tuple")
    normalized: list[tuple[str, float]] = []
    for name, value in values:
        normalized.append((_identifier(name, f"{field} name"), _finite(value, name)))
    names = tuple(name for name, _ in normalized)
    if len(set(names)) != len(names):
        _fail(f"{field} names must be unique")
    return tuple(sorted(normalized))


class TransformKind(StrEnum):
    """Supported one-dimensional parameter transforms."""

    IDENTITY = "identity"
    LOG = "log"
    LOGIT = "logit"


@dataclass(frozen=True, slots=True)
class ParameterTransform:
    """Invertible transform between constrained and unconstrained spaces."""

    kind: TransformKind = TransformKind.IDENTITY

    def __post_init__(self) -> None:
        if not isinstance(self.kind, TransformKind):
            _fail("kind must be a TransformKind")

    def forward(self, value: float, *, lower: float, upper: float) -> float:
        """Map one in-support parameter value to unconstrained space."""
        x = _finite(value, "value")
        lo = _finite(lower, "lower")
        hi = _finite(upper, "upper")
        if not lo < hi:
            _fail("lower must be below upper")
        if self.kind is TransformKind.IDENTITY:
            if not lo <= x <= hi:
                _fail("identity value lies outside the declared support")
            return x
        if self.kind is TransformKind.LOG:
            if not lo <= x <= hi or x <= 0.0:
                _fail("log transform requires a positive in-support value")
            return math.log(x)
        if not lo < x < hi:
            _fail("logit values must lie strictly inside finite bounds")
        ratio = (x - lo) / (hi - x)
        return math.log(ratio)

    def inverse(self, value: float, *, lower: float, upper: float) -> float:
        """Map an unconstrained value back to the declared support."""
        z = _finite(value, "value")
        lo = _finite(lower, "lower")
        hi = _finite(upper, "upper")
        if not lo < hi:
            _fail("lower must be below upper")
        if self.kind is TransformKind.IDENTITY:
            if not lo <= z <= hi:
                _fail("identity inverse lies outside the declared support")
            return z
        if self.kind is TransformKind.LOG:
            result = math.exp(z)
            if not lo <= result <= hi:
                _fail("log inverse lies outside the declared support")
            return result
        if z >= 0.0:
            factor = math.exp(-z)
            unit = 1.0 / (1.0 + factor)
        else:
            factor = math.exp(z)
            unit = factor / (1.0 + factor)
        return lo + (hi - lo) * unit

    def to_mapping(self) -> dict[str, object]:
        return {"kind": self.kind.value}


@dataclass(frozen=True, slots=True)
class PriorSpec:
    """One bounded continuous uniform prior used by the ABC baseline."""

    name: str
    lower: float
    upper: float
    transform: ParameterTransform = ParameterTransform()

    def __post_init__(self) -> None:
        _identifier(self.name, "prior name")
        lo = _finite(self.lower, "lower")
        hi = _finite(self.upper, "upper")
        if not lo < hi:
            _fail("prior lower must be below upper")
        if self.transform.kind is TransformKind.LOG and lo <= 0.0:
            _fail("a log-transformed prior must have positive support")
        object.__setattr__(self, "lower", lo)
        object.__setattr__(self, "upper", hi)

    @property
    def mean(self) -> float:
        return (self.lower + self.upper) / 2.0

    @property
    def standard_deviation(self) -> float:
        return (self.upper - self.lower) / math.sqrt(12.0)

    def to_mapping(self) -> dict[str, object]:
        return {
            "distribution": "uniform",
            "lower": self.lower,
            "name": self.name,
            "transform": self.transform.to_mapping(),
            "upper": self.upper,
        }


class SummaryLevel(StrEnum):
    """Scientific aggregation levels for calibration summaries."""

    MICRO = "micro"
    MESO = "meso"
    MACRO = "macro"


class SummaryRole(StrEnum):
    """Disjoint uses of a frozen summary statistic."""

    CALIBRATION = "calibration"
    HELD_OUT_VALIDATION = "held_out_validation"


@dataclass(frozen=True, slots=True)
class SummaryStatisticSpec:
    """Frozen statistic semantics, source design, role, and weight.

    ``source_design_sha256`` commits the preregistered source *design* (for
    example, the TRAIN versus VALIDATION trajectory family).  It is not an
    assertion about the concrete records used by a run.  Concrete source-set
    commitments are verifier-derived by :class:`SummarySourcePartition` from
    the complete record, EventKey, and transform-lineage manifest.
    """

    name: str
    level: SummaryLevel
    role: SummaryRole
    definition_sha256: str
    source_design_sha256: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        _identifier(self.name, "summary name")
        if not isinstance(self.level, SummaryLevel):
            _fail("level must be a SummaryLevel")
        if not isinstance(self.role, SummaryRole):
            _fail("role must be a SummaryRole")
        for field in ("definition_sha256", "source_design_sha256"):
            if _SHA256.fullmatch(getattr(self, field)) is None:
                _fail(f"{field} must be a lowercase SHA-256 digest")
        weight = _finite(self.weight, "weight")
        if weight <= 0.0:
            _fail("summary weight must be positive")
        object.__setattr__(self, "weight", weight)

    def to_mapping(self) -> dict[str, object]:
        return {
            "definition_sha256": self.definition_sha256,
            "level": self.level.value,
            "name": self.name,
            "role": self.role.value,
            "source_design_sha256": self.source_design_sha256,
            "weight": self.weight,
        }


@dataclass(frozen=True, slots=True)
class SummaryStatisticsConfig:
    """Pre-registered, disjoint calibration and held-out statistics."""

    statistics: tuple[SummaryStatisticSpec, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.statistics, tuple) or not self.statistics:
            _fail("statistics must be a non-empty immutable tuple")
        if not all(isinstance(item, SummaryStatisticSpec) for item in self.statistics):
            _fail("statistics contain an invalid specification")
        names = tuple(item.name for item in self.statistics)
        if len(set(names)) != len(names):
            _fail("a statistic cannot be reused for calibration and validation")
        roles = {item.role for item in self.statistics}
        if roles != {SummaryRole.CALIBRATION, SummaryRole.HELD_OUT_VALIDATION}:
            _fail("at least one calibration and one held-out statistic are required")
        levels = {item.level for item in self.statistics}
        if levels != {SummaryLevel.MICRO, SummaryLevel.MESO, SummaryLevel.MACRO}:
            _fail("micro, meso, and macro summary levels must all be represented")
        calibration = tuple(
            item for item in self.statistics if item.role is SummaryRole.CALIBRATION
        )
        held_out = tuple(
            item
            for item in self.statistics
            if item.role is SummaryRole.HELD_OUT_VALIDATION
        )
        for role_name, role_statistics in (
            ("calibration", calibration),
            ("held-out", held_out),
        ):
            if len({item.source_design_sha256 for item in role_statistics}) != 1:
                _fail(
                    f"all {role_name} statistics must share one frozen source "
                    "design commitment"
                )
        for field in ("definition_sha256", "source_design_sha256"):
            calibration_values = {getattr(item, field) for item in calibration}
            held_out_values = {getattr(item, field) for item in held_out}
            if calibration_values & held_out_values:
                _fail(
                    "calibration and held-out statistics must not share "
                    f"{field} commitments"
                )

    @property
    def calibration(self) -> tuple[SummaryStatisticSpec, ...]:
        return tuple(
            item for item in self.statistics if item.role is SummaryRole.CALIBRATION
        )

    @property
    def held_out(self) -> tuple[SummaryStatisticSpec, ...]:
        return tuple(
            item
            for item in self.statistics
            if item.role is SummaryRole.HELD_OUT_VALIDATION
        )

    def to_mapping(self) -> dict[str, object]:
        return {"statistics": [item.to_mapping() for item in self.statistics]}


@dataclass(frozen=True, slots=True)
class IdentifiabilityThresholds:
    """Frozen warning thresholds, never adjusted from current outcomes."""

    minimum_contraction: float = 0.15
    boundary_fraction: float = 0.15
    boundary_mass_threshold: float = 0.35
    correlation_threshold: float = 0.9
    histogram_bins: int = 10

    def __post_init__(self) -> None:
        for field in (
            "minimum_contraction",
            "boundary_fraction",
            "boundary_mass_threshold",
            "correlation_threshold",
        ):
            value = _finite(getattr(self, field), field)
            if not 0.0 < value < 1.0:
                _fail(f"{field} must lie strictly between zero and one")
            object.__setattr__(self, field, value)
        _positive_int(self.histogram_bins, "histogram_bins")
        if self.histogram_bins < 4:
            _fail("histogram_bins must be at least four")

    def to_mapping(self) -> dict[str, object]:
        return {
            "boundary_fraction": self.boundary_fraction,
            "boundary_mass_threshold": self.boundary_mass_threshold,
            "correlation_threshold": self.correlation_threshold,
            "histogram_bins": self.histogram_bins,
            "minimum_contraction": self.minimum_contraction,
        }


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    """Complete immutable configuration for one synthetic ABC run."""

    schema_version: int
    study_id: str
    cutoff: datetime
    seed: str
    priors: tuple[PriorSpec, ...]
    summaries: SummaryStatisticsConfig
    simulation_count: int
    accepted_particles: int
    posterior_draws: int
    ppc_draws: int
    sbc_replicates: int
    sbc_posterior_draws: int
    identifiability: IdentifiabilityThresholds = IdentifiabilityThresholds()
    synthetic_only: bool = True

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            _fail("unsupported calibration config schema version")
        _identifier(self.study_id, "study_id")
        object.__setattr__(self, "cutoff", _aware_utc(self.cutoff, "cutoff"))
        if not isinstance(self.seed, str) or not self.seed or len(self.seed) > 512:
            _fail("seed must contain between 1 and 512 characters")
        if not isinstance(self.priors, tuple) or not self.priors:
            _fail("priors must be a non-empty immutable tuple")
        if not all(isinstance(item, PriorSpec) for item in self.priors):
            _fail("priors contain an invalid specification")
        names = tuple(item.name for item in self.priors)
        if len(set(names)) != len(names):
            _fail("prior names must be unique")
        if not isinstance(self.summaries, SummaryStatisticsConfig):
            _fail("summaries must be a SummaryStatisticsConfig")
        for field in (
            "simulation_count",
            "accepted_particles",
            "posterior_draws",
            "ppc_draws",
            "sbc_replicates",
            "sbc_posterior_draws",
        ):
            _positive_int(getattr(self, field), field)
        if self.accepted_particles >= self.simulation_count:
            _fail("accepted_particles must be smaller than simulation_count")
        if self.accepted_particles < 2:
            _fail("accepted_particles must retain a distribution, not one best point")
        if self.posterior_draws < 2 or self.ppc_draws < 2:
            _fail("posterior and PPC outputs each require at least two draws")
        if self.sbc_posterior_draws < 2:
            _fail("SBC requires at least two posterior draws per replicate")
        if not isinstance(self.identifiability, IdentifiabilityThresholds):
            _fail("identifiability thresholds are invalid")
        if self.synthetic_only is not True:
            _fail(
                "the current backend is synthetic-only; real calibration requires "
                "a future temporal-firewall-issued input capability"
            )

    def to_mapping(self) -> dict[str, object]:
        return {
            "accepted_particles": self.accepted_particles,
            "cutoff": self.cutoff.isoformat(),
            "identifiability": self.identifiability.to_mapping(),
            "posterior_draws": self.posterior_draws,
            "ppc_draws": self.ppc_draws,
            "priors": [item.to_mapping() for item in self.priors],
            "sbc_posterior_draws": self.sbc_posterior_draws,
            "sbc_replicates": self.sbc_replicates,
            "schema_version": self.schema_version,
            "seed": self.seed,
            "simulation_count": self.simulation_count,
            "study_id": self.study_id,
            "summaries": self.summaries.to_mapping(),
            "synthetic_only": self.synthetic_only,
        }

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))


@dataclass(frozen=True, slots=True)
class SummaryVector:
    """Complete named simulator summary vector."""

    values: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _named_values(self.values, "summary values"))

    def as_dict(self) -> dict[str, float]:
        return dict(self.values)

    def to_mapping(self) -> dict[str, object]:
        return {"values": {name: value for name, value in self.values}}

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))


@dataclass(frozen=True, slots=True)
class TransformLineageNode:
    """One complete transform-lineage node for a summary source set.

    Node and record IDs are local aliases.  The verifier resolves them before
    hashing, so merely renaming an alias cannot disguise shared upstream data.
    """

    node_id: str
    operation: str
    implementation_sha256: str
    parent_node_ids: tuple[str, ...] = ()
    source_record_ids: tuple[str, ...] = ()
    event_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.node_id, "lineage node_id")
        _identifier(self.operation, "lineage operation")
        if _SHA256.fullmatch(self.implementation_sha256) is None:
            _fail("lineage implementation_sha256 must be a SHA-256 digest")
        for field in ("parent_node_ids", "source_record_ids"):
            values = getattr(self, field)
            if not isinstance(values, tuple) or not all(
                isinstance(value, str) and _IDENTIFIER.fullmatch(value) is not None
                for value in values
            ):
                _fail(f"{field} must be an immutable stable-identifier tuple")
            if len(set(values)) != len(values):
                _fail(f"{field} must not contain duplicates")
            object.__setattr__(self, field, tuple(sorted(values)))
        if not isinstance(self.event_keys, tuple) or not all(
            isinstance(value, str) and _EVENT_KEY.fullmatch(value) is not None
            for value in self.event_keys
        ):
            _fail("lineage event_keys must be immutable ptk1 SHA-256 keys")
        if len(set(self.event_keys)) != len(self.event_keys):
            _fail("lineage event_keys must not contain duplicates")
        object.__setattr__(self, "event_keys", tuple(sorted(self.event_keys)))
        if not (self.parent_node_ids or self.source_record_ids or self.event_keys):
            _fail("a lineage node must bind a parent, source record, or EventKey")
        if self.node_id in self.parent_node_ids:
            _fail("a lineage node cannot directly reference itself")

    def to_mapping(self) -> dict[str, object]:
        return {
            "event_keys": list(self.event_keys),
            "implementation_sha256": self.implementation_sha256,
            "node_id": self.node_id,
            "operation": self.operation,
            "parent_node_ids": list(self.parent_node_ids),
            "source_record_ids": list(self.source_record_ids),
        }


def _record_manifest_mapping(
    record: TemporalRecord, *, include_partition: bool
) -> dict[str, object]:
    mapping = record.to_mapping()
    del mapping["record_id"]
    if not include_partition:
        del mapping["partition"]
    return mapping


@dataclass(frozen=True, slots=True)
class TransformLineage:
    """Connected acyclic transform graph covering every record and EventKey."""

    root_node_id: str
    nodes: tuple[TransformLineageNode, ...]

    def __post_init__(self) -> None:
        _identifier(self.root_node_id, "lineage root_node_id")
        if (
            not isinstance(self.nodes, tuple)
            or not self.nodes
            or not all(isinstance(node, TransformLineageNode) for node in self.nodes)
        ):
            _fail("transform lineage requires a non-empty immutable node tuple")
        ordered = tuple(sorted(self.nodes, key=lambda node: node.node_id))
        if len({node.node_id for node in ordered}) != len(ordered):
            _fail("transform lineage node IDs must be unique")
        by_id = {node.node_id: node for node in ordered}
        if self.root_node_id not in by_id:
            _fail("transform lineage root node is missing")
        for node in ordered:
            missing = set(node.parent_node_ids) - set(by_id)
            if missing:
                _fail(
                    f"transform lineage references missing parents: {sorted(missing)!r}"
                )

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node_id: str) -> None:
            if node_id in visiting:
                _fail("transform lineage contains a cycle")
            if node_id in visited:
                return
            visiting.add(node_id)
            for parent_id in by_id[node_id].parent_node_ids:
                visit(parent_id)
            visiting.remove(node_id)
            visited.add(node_id)

        visit(self.root_node_id)
        if visited != set(by_id):
            _fail("transform lineage contains nodes disconnected from its root")
        object.__setattr__(self, "nodes", ordered)

    def _validated_context(
        self,
        records: tuple[TemporalRecord, ...],
        event_keys: tuple[str, ...],
    ) -> tuple[dict[str, TemporalRecord], dict[str, TransformLineageNode]]:
        by_record_id = {record.record_id: record for record in records}
        if len(by_record_id) != len(records):
            _fail("source record IDs must be unique")
        referenced_records = tuple(
            record_id for node in self.nodes for record_id in node.source_record_ids
        )
        if len(set(referenced_records)) != len(referenced_records):
            _fail("each source record must appear exactly once in transform lineage")
        missing_records = set(referenced_records) - set(by_record_id)
        if missing_records:
            _fail(
                "transform lineage references unknown source records: "
                f"{sorted(missing_records)!r}"
            )
        if set(referenced_records) != set(by_record_id):
            _fail(
                "transform lineage does not cover the complete source record manifest"
            )

        referenced_keys = tuple(key for node in self.nodes for key in node.event_keys)
        if len(set(referenced_keys)) != len(referenced_keys):
            _fail("each EventKey must appear exactly once in transform lineage")
        if set(referenced_keys) != set(event_keys):
            _fail("transform lineage does not cover the complete EventKey manifest")
        return by_record_id, {node.node_id: node for node in self.nodes}

    def _validated_design_context(
        self,
        record_partitions: tuple[tuple[str, DataPartition], ...],
        event_keys: tuple[str, ...],
    ) -> tuple[dict[str, DataPartition], dict[str, TransformLineageNode]]:
        if not isinstance(record_partitions, tuple) or not record_partitions:
            _fail("source design requires a non-empty record-partition tuple")
        by_record_id: dict[str, DataPartition] = {}
        for record_id, partition in record_partitions:
            _identifier(record_id, "source design record_id")
            if not isinstance(partition, DataPartition):
                _fail("source design partition must be a DataPartition")
            if record_id in by_record_id:
                _fail("source design record IDs must be unique")
            by_record_id[record_id] = partition
        if (
            not isinstance(event_keys, tuple)
            or not event_keys
            or not all(
                isinstance(key, str) and _EVENT_KEY.fullmatch(key) is not None
                for key in event_keys
            )
        ):
            _fail("source design requires a non-empty ptk1 EventKey tuple")
        if len(set(event_keys)) != len(event_keys):
            _fail("source design EventKeys must be unique")
        referenced_records = tuple(
            record_id for node in self.nodes for record_id in node.source_record_ids
        )
        if len(set(referenced_records)) != len(referenced_records):
            _fail("each source-design record must appear exactly once in lineage")
        if set(referenced_records) != set(by_record_id):
            _fail("transform lineage does not cover the source-design records")
        referenced_keys = tuple(key for node in self.nodes for key in node.event_keys)
        if len(set(referenced_keys)) != len(referenced_keys):
            _fail("each source-design EventKey must appear exactly once in lineage")
        if set(referenced_keys) != set(event_keys):
            _fail("transform lineage does not cover the source-design EventKeys")
        return by_record_id, {node.node_id: node for node in self.nodes}

    def source_design_sha256(
        self,
        record_partitions: tuple[tuple[str, DataPartition], ...],
        event_keys: tuple[str, ...],
    ) -> str:
        """Hash transform semantics and source roles without run-specific aliases."""
        by_record_id, by_node_id = self._validated_design_context(
            record_partitions, event_keys
        )
        memo: dict[str, str] = {}

        def fingerprint(node_id: str) -> str:
            existing = memo.get(node_id)
            if existing is not None:
                return existing
            node = by_node_id[node_id]
            value = sha256_bytes(
                canonical_json_bytes(
                    {
                        "event_key_count": len(node.event_keys),
                        "implementation_sha256": node.implementation_sha256,
                        "operation": node.operation,
                        "parent_design_fingerprints": sorted(
                            fingerprint(parent_id) for parent_id in node.parent_node_ids
                        ),
                        "source_record_partitions": sorted(
                            by_record_id[record_id].value
                            for record_id in node.source_record_ids
                        ),
                    }
                )
            )
            memo[node_id] = value
            return value

        root_fingerprint = fingerprint(self.root_node_id)
        return sha256_bytes(
            canonical_json_bytes(
                {
                    "lineage_node_design_fingerprints": sorted(memo.values()),
                    "root_lineage_design_fingerprint": root_fingerprint,
                    "schema_version": "policytwin.calibration-source-design.v1",
                }
            )
        )

    def semantic_fingerprints(
        self,
        records: tuple[TemporalRecord, ...],
        event_keys: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Return alias- and order-invariant fingerprints for every lineage node."""
        by_record_id, by_node_id = self._validated_context(records, event_keys)
        memo: dict[str, str] = {}

        def fingerprint(node_id: str) -> str:
            existing = memo.get(node_id)
            if existing is not None:
                return existing
            node = by_node_id[node_id]
            value = sha256_bytes(
                canonical_json_bytes(
                    {
                        "event_keys": sorted(node.event_keys),
                        "implementation_sha256": node.implementation_sha256,
                        "operation": node.operation,
                        "parent_fingerprints": sorted(
                            fingerprint(parent_id) for parent_id in node.parent_node_ids
                        ),
                        "source_record_fingerprints": sorted(
                            sha256_bytes(
                                canonical_json_bytes(
                                    _record_manifest_mapping(
                                        by_record_id[record_id],
                                        include_partition=False,
                                    )
                                )
                            )
                            for record_id in node.source_record_ids
                        ),
                    }
                )
            )
            memo[node_id] = value
            return value

        fingerprint(self.root_node_id)
        return tuple(sorted(memo.values()))

    def source_set_sha256(
        self,
        records: tuple[TemporalRecord, ...],
        event_keys: tuple[str, ...],
    ) -> str:
        """Derive the concrete source-set commitment from complete manifests."""
        _, by_node_id = self._validated_context(records, event_keys)
        fingerprints = self.semantic_fingerprints(records, event_keys)
        fingerprint_by_id: dict[str, str] = {}

        def resolve(node_id: str) -> str:
            existing = fingerprint_by_id.get(node_id)
            if existing is not None:
                return existing
            node = by_node_id[node_id]
            by_record_id = {record.record_id: record for record in records}
            value = sha256_bytes(
                canonical_json_bytes(
                    {
                        "event_keys": sorted(node.event_keys),
                        "implementation_sha256": node.implementation_sha256,
                        "operation": node.operation,
                        "parent_fingerprints": sorted(
                            resolve(parent_id) for parent_id in node.parent_node_ids
                        ),
                        "source_record_fingerprints": sorted(
                            sha256_bytes(
                                canonical_json_bytes(
                                    _record_manifest_mapping(
                                        by_record_id[record_id],
                                        include_partition=False,
                                    )
                                )
                            )
                            for record_id in node.source_record_ids
                        ),
                    }
                )
            )
            fingerprint_by_id[node_id] = value
            return value

        root_fingerprint = resolve(self.root_node_id)
        return sha256_bytes(
            canonical_json_bytes(
                {
                    "event_keys": sorted(event_keys),
                    "lineage_node_fingerprints": list(fingerprints),
                    "record_manifest": sorted(
                        (
                            _record_manifest_mapping(record, include_partition=True)
                            for record in records
                        ),
                        key=lambda item: canonical_json_bytes(item),
                    ),
                    "root_lineage_fingerprint": root_fingerprint,
                    "schema_version": "policytwin.calibration-source-set.v1",
                }
            )
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "nodes": [node.to_mapping() for node in self.nodes],
            "root_node_id": self.root_node_id,
        }


def derive_source_set_sha256(
    *,
    records: tuple[TemporalRecord, ...],
    event_keys: tuple[str, ...],
    transform_lineage: TransformLineage,
) -> str:
    """Public verifier for an alias- and order-normalized concrete source set."""
    if (
        not isinstance(records, tuple)
        or not records
        or not all(isinstance(record, TemporalRecord) for record in records)
    ):
        _fail("source commitment requires a non-empty TemporalRecord tuple")
    if (
        not isinstance(event_keys, tuple)
        or not event_keys
        or not all(
            isinstance(key, str) and _EVENT_KEY.fullmatch(key) is not None
            for key in event_keys
        )
    ):
        _fail("source commitment requires a non-empty ptk1 EventKey tuple")
    if len(set(event_keys)) != len(event_keys):
        _fail("source commitment EventKeys must be unique")
    if not isinstance(transform_lineage, TransformLineage):
        _fail("source commitment requires a TransformLineage")
    return transform_lineage.source_set_sha256(records, event_keys)


def derive_source_design_sha256(
    *,
    record_partitions: tuple[tuple[str, DataPartition], ...],
    event_keys: tuple[str, ...],
    transform_lineage: TransformLineage,
) -> str:
    """Derive the preregistered transform/source-role design commitment."""
    if not isinstance(transform_lineage, TransformLineage):
        _fail("source design requires a TransformLineage")
    return transform_lineage.source_design_sha256(record_partitions, event_keys)


@dataclass(frozen=True, slots=True)
class SummarySourcePartition:
    """Actual role-typed records, EventKeys, and values for one summary split."""

    role: SummaryRole
    observed: SummaryVector
    source_design_sha256: str
    source_record_set_sha256: str
    records: tuple[TemporalRecord, ...]
    event_keys: tuple[str, ...]
    transform_lineage: TransformLineage

    def __post_init__(self) -> None:
        if not isinstance(self.role, SummaryRole):
            _fail("summary source role must be a SummaryRole")
        if not isinstance(self.observed, SummaryVector):
            _fail("partition observed must be a SummaryVector")
        for field in ("source_design_sha256", "source_record_set_sha256"):
            if _SHA256.fullmatch(getattr(self, field)) is None:
                _fail(f"partition {field} must be a SHA-256 digest")
        if not isinstance(self.records, tuple) or not self.records:
            _fail("partition records must be a non-empty immutable tuple")
        if not all(isinstance(item, TemporalRecord) for item in self.records):
            _fail("partition records contain invalid temporal metadata")
        ids = tuple(item.record_id for item in self.records)
        if len(set(ids)) != len(ids):
            _fail("partition record IDs must be unique")
        expected_partition = (
            DataPartition.TRAIN
            if self.role is SummaryRole.CALIBRATION
            else DataPartition.VALIDATION
        )
        if any(item.partition is not expected_partition for item in self.records):
            _fail("summary source role uses the wrong temporal data partition")
        if (
            not isinstance(self.event_keys, tuple)
            or not self.event_keys
            or not all(
                isinstance(item, str) and _EVENT_KEY.fullmatch(item) is not None
                for item in self.event_keys
            )
            or len(set(self.event_keys)) != len(self.event_keys)
        ):
            _fail("partition event keys must be unique ptk1 SHA-256 keys")
        object.__setattr__(
            self, "records", tuple(sorted(self.records, key=lambda x: x.record_id))
        )
        object.__setattr__(self, "event_keys", tuple(sorted(self.event_keys)))
        semantic_records = tuple(
            sha256_bytes(
                canonical_json_bytes(
                    _record_manifest_mapping(record, include_partition=False)
                )
            )
            for record in self.records
        )
        if len(set(semantic_records)) != len(semantic_records):
            _fail("partition contains renamed aliases of the same semantic record")
        expected_commitment = derive_source_set_sha256(
            records=self.records,
            event_keys=self.event_keys,
            transform_lineage=self.transform_lineage,
        )
        if self.source_record_set_sha256 != expected_commitment:
            _fail(
                "partition source_record_set_sha256 differs from the verifier-derived "
                "record/EventKey/transform-lineage commitment"
            )
        expected_design = derive_source_design_sha256(
            record_partitions=tuple(
                (record.record_id, record.partition) for record in self.records
            ),
            event_keys=self.event_keys,
            transform_lineage=self.transform_lineage,
        )
        if self.source_design_sha256 != expected_design:
            _fail(
                "partition source_design_sha256 differs from the verifier-derived "
                "partition/transform-lineage design commitment"
            )

    def to_mapping(self) -> dict[str, object]:
        return {
            "event_keys": list(self.event_keys),
            "observed": self.observed.to_mapping(),
            "records": [item.to_mapping() for item in self.records],
            "role": self.role.value,
            "source_design_sha256": self.source_design_sha256,
            "source_record_set_sha256": self.source_record_set_sha256,
            "transform_lineage": self.transform_lineage.to_mapping(),
        }


@dataclass(frozen=True, slots=True)
class CalibrationInput:
    """Synthetic observations with disjoint calibration/held-out sources."""

    observed: SummaryVector
    sources: tuple[SummarySourcePartition, ...]
    synthetic_only: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.observed, SummaryVector):
            _fail("observed must be a SummaryVector")
        if (
            not isinstance(self.sources, tuple)
            or len(self.sources) != 2
            or not all(
                isinstance(item, SummarySourcePartition) for item in self.sources
            )
        ):
            _fail("calibration input requires two typed summary source partitions")
        by_role = {item.role: item for item in self.sources}
        if set(by_role) != {
            SummaryRole.CALIBRATION,
            SummaryRole.HELD_OUT_VALIDATION,
        }:
            _fail("calibration input requires calibration and held-out source roles")
        ordered = tuple(by_role[role] for role in SummaryRole)
        object.__setattr__(self, "sources", ordered)
        calibration, held_out = ordered
        calibration_names = set(calibration.observed.as_dict())
        held_out_names = set(held_out.observed.as_dict())
        if calibration_names & held_out_names:
            _fail("calibration and held-out summary names must be disjoint")
        combined = {
            **calibration.observed.as_dict(),
            **held_out.observed.as_dict(),
        }
        if combined != self.observed.as_dict():
            _fail("role summaries do not exactly reconstruct the observed vector")
        calibration_record_ids = {item.record_id for item in calibration.records}
        held_out_record_ids = {item.record_id for item in held_out.records}
        if calibration_record_ids & held_out_record_ids:
            _fail("calibration and held-out records must be disjoint")
        if set(calibration.event_keys) & set(held_out.event_keys):
            _fail("calibration and held-out EventKeys must be disjoint")
        if calibration.source_record_set_sha256 == held_out.source_record_set_sha256:
            _fail("calibration and held-out verified source sets must be disjoint")
        calibration_record_fingerprints = {
            sha256_bytes(
                canonical_json_bytes(
                    _record_manifest_mapping(record, include_partition=False)
                )
            )
            for record in calibration.records
        }
        held_out_record_fingerprints = {
            sha256_bytes(
                canonical_json_bytes(
                    _record_manifest_mapping(record, include_partition=False)
                )
            )
            for record in held_out.records
        }
        calibration_content_hashes = {
            record.content_hash for record in calibration.records
        }
        held_out_content_hashes = {record.content_hash for record in held_out.records}
        if (
            calibration_content_hashes & held_out_content_hashes
            or calibration_record_fingerprints & held_out_record_fingerprints
        ):
            _fail(
                "calibration and held-out content-addressed source records overlap, "
                "including renamed aliases"
            )
        calibration_lineage = set(
            calibration.transform_lineage.semantic_fingerprints(
                calibration.records, calibration.event_keys
            )
        )
        held_out_lineage = set(
            held_out.transform_lineage.semantic_fingerprints(
                held_out.records, held_out.event_keys
            )
        )
        if calibration_lineage & held_out_lineage:
            _fail("calibration and held-out transform lineage overlaps")
        if self.synthetic_only is not True:
            _fail("this calibration input bridge is synthetic-only")

    @property
    def records(self) -> tuple[TemporalRecord, ...]:
        """Return both role-typed record sets for temporal validation."""
        return tuple(record for source in self.sources for record in source.records)

    @property
    def event_keys(self) -> tuple[str, ...]:
        """Return both disjoint EventKey manifests in stable role order."""
        return tuple(key for source in self.sources for key in source.event_keys)

    def source_for(self, role: SummaryRole) -> SummarySourcePartition:
        """Return the exact frozen source partition for one role."""
        if not isinstance(role, SummaryRole):
            _fail("source role must be a SummaryRole")
        return next(item for item in self.sources if item.role is role)

    def validate_for(self, config: CalibrationConfig) -> None:
        """Reject incomplete statistics and all policy-post/future records."""
        if not isinstance(config, CalibrationConfig):
            _fail("config must be a CalibrationConfig")
        expected = {item.name for item in config.summaries.statistics}
        actual = set(self.observed.as_dict())
        if actual != expected:
            _fail(
                "observed summaries do not exactly match the frozen statistic set: "
                f"missing={sorted(expected - actual)!r}, "
                f"extra={sorted(actual - expected)!r}"
            )
        for role, specifications in (
            (SummaryRole.CALIBRATION, config.summaries.calibration),
            (SummaryRole.HELD_OUT_VALIDATION, config.summaries.held_out),
        ):
            source = self.source_for(role)
            expected_names = {item.name for item in specifications}
            if set(source.observed.as_dict()) != expected_names:
                _fail(f"{role.value} source summary names differ from config")
            expected_source_designs = {
                item.source_design_sha256 for item in specifications
            }
            if expected_source_designs != {source.source_design_sha256}:
                _fail(f"{role.value} source design commitment differs from config")
            verified_source_hash = derive_source_set_sha256(
                records=source.records,
                event_keys=source.event_keys,
                transform_lineage=source.transform_lineage,
            )
            if verified_source_hash != source.source_record_set_sha256:
                _fail(f"{role.value} source-set verifier commitment changed")
        for record in self.records:
            if record.partition is DataPartition.POST_POLICY_EVALUATION:
                _fail(
                    "policy-post record is forbidden in calibration: "
                    f"{record.record_id}"
                )
            if record.available_time > config.cutoff:
                _fail(f"record available after calibration cutoff: {record.record_id}")
            if record.event_time > config.cutoff:
                _fail(
                    f"record event occurs after calibration cutoff: {record.record_id}"
                )

    def to_mapping(self) -> dict[str, object]:
        return {
            "observed": self.observed.to_mapping(),
            "sources": [item.to_mapping() for item in self.sources],
            "synthetic_only": self.synthetic_only,
        }

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))


@dataclass(frozen=True, slots=True)
class Particle:
    """One accepted ABC particle; no single particle is privileged as best."""

    parameters: tuple[tuple[str, float], ...]
    distance: float
    simulation_event_key: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "parameters", _named_values(self.parameters, "parameters")
        )
        distance = _finite(self.distance, "distance")
        if distance < 0.0:
            _fail("distance must not be negative")
        object.__setattr__(self, "distance", distance)
        if (
            not isinstance(self.simulation_event_key, str)
            or not self.simulation_event_key
        ):
            _fail("simulation_event_key must be non-empty")

    def to_mapping(self) -> dict[str, object]:
        return {
            "distance": self.distance,
            "parameters": dict(self.parameters),
            "simulation_event_key": self.simulation_event_key,
        }


@dataclass(frozen=True, slots=True)
class PosteriorDraw:
    """A keyed resample from the accepted empirical ABC posterior."""

    parameters: tuple[tuple[str, float], ...]
    source_particle: int
    event_key: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "parameters", _named_values(self.parameters, "parameters")
        )
        if (
            isinstance(self.source_particle, bool)
            or not isinstance(self.source_particle, int)
            or self.source_particle < 0
        ):
            _fail("source_particle must be a non-negative integer")
        if not isinstance(self.event_key, str) or not self.event_key:
            _fail("event_key must be non-empty")

    def to_mapping(self) -> dict[str, object]:
        return {
            "event_key": self.event_key,
            "parameters": dict(self.parameters),
            "source_particle": self.source_particle,
        }


@dataclass(frozen=True, slots=True)
class PosteriorSamples:
    """Accepted particles and explicit posterior draws."""

    backend_id: str
    particles: tuple[Particle, ...]
    draws: tuple[PosteriorDraw, ...]

    def __post_init__(self) -> None:
        _identifier(self.backend_id, "backend_id")
        if not self.particles or not all(
            isinstance(item, Particle) for item in self.particles
        ):
            _fail("posterior must contain accepted particles")
        if not self.draws or not all(
            isinstance(item, PosteriorDraw) for item in self.draws
        ):
            _fail("posterior must contain draws, not only one best parameter")
        if any(item.source_particle >= len(self.particles) for item in self.draws):
            _fail("posterior draw references an unknown particle")
        simulation_keys = tuple(item.simulation_event_key for item in self.particles)
        if len(set(simulation_keys)) != len(simulation_keys):
            _fail("posterior particles must have unique simulation event keys")
        draw_keys = tuple(item.event_key for item in self.draws)
        if len(set(draw_keys)) != len(draw_keys):
            _fail("posterior draws must have unique resampling event keys")
        for draw in self.draws:
            source = self.particles[draw.source_particle]
            if draw.parameters != source.parameters:
                _fail("posterior draw parameters do not match its source particle")

    def to_mapping(self) -> dict[str, object]:
        return {
            "backend_id": self.backend_id,
            "draws": [item.to_mapping() for item in self.draws],
            "particles": [item.to_mapping() for item in self.particles],
        }

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))


def parameter_mapping(values: Mapping[str, float]) -> tuple[tuple[str, float], ...]:
    """Normalize a public parameter mapping to an immutable representation."""
    return _named_values(tuple(values.items()), "parameters")


__all__ = [
    "CalibrationConfig",
    "CalibrationError",
    "CalibrationInput",
    "IdentifiabilityThresholds",
    "ParameterTransform",
    "Particle",
    "PosteriorDraw",
    "PosteriorSamples",
    "PriorSpec",
    "SummaryLevel",
    "SummaryRole",
    "SummarySourcePartition",
    "SummaryStatisticSpec",
    "SummaryStatisticsConfig",
    "SummaryVector",
    "TransformLineage",
    "TransformLineageNode",
    "TransformKind",
    "derive_source_design_sha256",
    "derive_source_set_sha256",
    "parameter_mapping",
]
