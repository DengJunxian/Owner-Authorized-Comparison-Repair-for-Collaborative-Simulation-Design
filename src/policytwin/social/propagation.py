"""Deterministic synchronous social-information propagation."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from policytwin.counterfactual import EventKey, ScenarioEventReceipt
from policytwin.social.audit import (
    SocialAuditLog,
    SocialAuditOutcome,
    SocialAuditRule,
    SocialAuditStage,
)
from policytwin.social.events import (
    InformationEvent,
    InformationEventError,
    InformationEventKind,
    InformationEventTemporalError,
    ScenarioInformationEvent,
)
from policytwin.social.graphs import (
    ObservedSocialGraph,
    SocialGraph,
    SocialGraphError,
    SocialGraphTemporalError,
    SyntheticSocialGraph,
)

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class SocialDiffusionError(ValueError):
    """Raised when deterministic social propagation cannot be trusted."""


class SocialDiffusionTemporalError(SocialDiffusionError):
    """Raised when graph or event knowledge crosses the configured boundary."""


class MissingCorrectionTargetError(SocialDiffusionError):
    """Raised when a correction targets evidence absent from one node."""


class AttentionOutcome(StrEnum):
    """Closed outcomes for a candidate exposure."""

    ACCEPTED = "accepted"
    CAPACITY_REJECTED = "capacity_rejected"
    DUPLICATE_PATH = "duplicate_path"
    ALREADY_EXPOSED = "already_exposed"


class BeliefUpdateKind(StrEnum):
    """Mechanism responsible for one belief transition."""

    DECAY = "decay"
    SIGNAL = "signal"
    CORRECTION = "correction"


def _identifier(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise SocialDiffusionError(
            f"{field_name} must be a stable anonymous identifier"
        )
    return value


def _sha256(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise SocialDiffusionError(f"{field_name} must be a lowercase SHA-256")
    return value


def _aware_utc(value: object, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise SocialDiffusionError(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise SocialDiffusionError(f"{field_name} must include a timezone")
    try:
        return value.astimezone(UTC)
    except (OverflowError, ValueError) as error:
        raise SocialDiffusionError(f"{field_name} is outside UTC range") from error


def _finite(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SocialDiffusionError(f"{field_name} must be a finite real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise SocialDiffusionError(f"{field_name} must be finite")
    return normalized


def _non_negative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SocialDiffusionError(f"{field_name} must be a non-negative integer")
    return value


def _commit(value: object) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SocialDiffusionError("propagation value is not canonical JSON") from error
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class PropagationConfig:
    """Immutable policy, time, and deterministic event-key configuration."""

    experiment_id: str
    shared_seed: str
    replicate_id: int
    policy_sha256: str
    as_of_time: datetime
    trusted_cutoff: datetime
    scenario_event_receipt: ScenarioEventReceipt | None = None
    social_enabled: bool = True
    max_hops: int = 3
    attention_capacity: int = 8
    repost_threshold: float = 0.25
    belief_decay: float = 0.0
    correction_strength: float = 0.5

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "experiment_id", _identifier(self.experiment_id, "experiment_id")
        )
        object.__setattr__(
            self, "shared_seed", _identifier(self.shared_seed, "shared_seed")
        )
        _non_negative_int(self.replicate_id, "replicate_id")
        object.__setattr__(
            self, "policy_sha256", _sha256(self.policy_sha256, "policy_sha256")
        )
        as_of_time = _aware_utc(self.as_of_time, "as_of_time")
        trusted_cutoff = _aware_utc(self.trusted_cutoff, "trusted_cutoff")
        scenario_receipt = self.scenario_event_receipt
        if as_of_time > trusted_cutoff:
            if not isinstance(scenario_receipt, ScenarioEventReceipt):
                raise SocialDiffusionTemporalError(
                    "post-cutoff propagation requires an engine scenario receipt"
                )
            if (
                scenario_receipt.scenario_time != as_of_time
                or scenario_receipt.trusted_cutoff != trusted_cutoff
                or scenario_receipt.scenario_id != self.experiment_id
                or scenario_receipt.generator_spec.cutoff_policy_sha256
                != self.policy_sha256
            ):
                raise SocialDiffusionTemporalError(
                    "scenario receipt does not match propagation time and policy"
                )
        elif scenario_receipt is not None:
            raise SocialDiffusionTemporalError(
                "scenario receipt is invalid for pre-cutoff observed propagation"
            )
        object.__setattr__(self, "as_of_time", as_of_time)
        object.__setattr__(self, "trusted_cutoff", trusted_cutoff)
        if not isinstance(self.social_enabled, bool):
            raise SocialDiffusionError("social_enabled must be boolean")
        _non_negative_int(self.max_hops, "max_hops")
        _non_negative_int(self.attention_capacity, "attention_capacity")
        repost_threshold = _finite(self.repost_threshold, "repost_threshold")
        if repost_threshold < 0.0:
            raise SocialDiffusionError("repost_threshold must be non-negative")
        object.__setattr__(self, "repost_threshold", repost_threshold)
        belief_decay = _finite(self.belief_decay, "belief_decay")
        if not 0.0 <= belief_decay <= 1.0:
            raise SocialDiffusionError("belief_decay must be in [0, 1]")
        object.__setattr__(self, "belief_decay", belief_decay)
        correction_strength = _finite(self.correction_strength, "correction_strength")
        if not 0.0 < correction_strength <= 1.0:
            raise SocialDiffusionError("correction_strength must be in (0, 1]")
        object.__setattr__(self, "correction_strength", correction_strength)

    def to_mapping(self) -> dict[str, object]:
        """Return the complete propagation policy without a world identifier."""
        return {
            "as_of_time": self.as_of_time.isoformat(),
            "attention_capacity": self.attention_capacity,
            "belief_decay": self.belief_decay,
            "correction_strength": self.correction_strength,
            "experiment_id": self.experiment_id,
            "max_hops": self.max_hops,
            "policy_sha256": self.policy_sha256,
            "replicate_id": self.replicate_id,
            "repost_threshold": self.repost_threshold,
            "shared_seed": self.shared_seed,
            "scenario_event_receipt": (
                None
                if self.scenario_event_receipt is None
                else self.scenario_event_receipt.to_mapping()
            ),
            "social_enabled": self.social_enabled,
            "trusted_cutoff": self.trusted_cutoff.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class SeededInformationEvent:
    """One information event injected directly at an anonymous seed node."""

    event: InformationEvent | ScenarioInformationEvent
    seed_node_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.event, InformationEvent | ScenarioInformationEvent):
            raise SocialDiffusionError(
                "event must be observed or engine-generated information"
            )
        object.__setattr__(
            self, "seed_node_id", _identifier(self.seed_node_id, "seed_node_id")
        )

    def to_mapping(self) -> dict[str, object]:
        return {"event": self.event.to_mapping(), "seed_node_id": self.seed_node_id}


@dataclass(frozen=True, slots=True)
class BeliefComponent:
    """One event-attributed belief component with retained correction lineage."""

    event_id: str
    contribution: float
    lineage_event_ids: tuple[str, ...]
    lineage_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", _identifier(self.event_id, "event_id"))
        object.__setattr__(
            self, "contribution", _finite(self.contribution, "contribution")
        )
        if not isinstance(self.lineage_event_ids, tuple) or not self.lineage_event_ids:
            raise SocialDiffusionError("lineage_event_ids must be a non-empty tuple")
        if not isinstance(self.lineage_sha256s, tuple) or not self.lineage_sha256s:
            raise SocialDiffusionError("lineage_sha256s must be a non-empty tuple")
        if len(self.lineage_event_ids) != len(self.lineage_sha256s):
            raise SocialDiffusionError("belief lineage identifiers and hashes differ")
        normalized_ids = tuple(
            _identifier(item, "lineage_event_id") for item in self.lineage_event_ids
        )
        normalized_hashes = tuple(
            _sha256(item, "lineage_sha256") for item in self.lineage_sha256s
        )
        if normalized_ids[0] != self.event_id:
            raise SocialDiffusionError("belief lineage must begin with event_id")
        object.__setattr__(self, "lineage_event_ids", normalized_ids)
        object.__setattr__(self, "lineage_sha256s", normalized_hashes)

    def to_mapping(self) -> dict[str, object]:
        return {
            "contribution": self.contribution,
            "event_id": self.event_id,
            "lineage_event_ids": list(self.lineage_event_ids),
            "lineage_sha256s": list(self.lineage_sha256s),
        }


@dataclass(frozen=True, slots=True)
class NodeBelief:
    """Canonical event-attributed belief state for one anonymous graph node."""

    node_id: str
    components: tuple[BeliefComponent, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _identifier(self.node_id, "node_id"))
        if not isinstance(self.components, tuple) or not all(
            isinstance(component, BeliefComponent) for component in self.components
        ):
            raise SocialDiffusionError(
                "components must be an immutable tuple of BeliefComponent values"
            )
        event_ids = [component.event_id for component in self.components]
        if len(set(event_ids)) != len(event_ids):
            raise SocialDiffusionError("node belief has duplicate event components")
        object.__setattr__(
            self,
            "components",
            tuple(sorted(self.components, key=lambda component: component.event_id)),
        )

    @property
    def belief(self) -> float:
        """Return a stable sum of canonical event contributions."""
        return math.fsum(component.contribution for component in self.components)

    def to_mapping(self) -> dict[str, object]:
        return {
            "belief": self.belief,
            "components": [component.to_mapping() for component in self.components],
            "node_id": self.node_id,
        }


@dataclass(frozen=True, slots=True)
class ExposureRecord:
    hop: int
    node_id: str
    source_node_id: str | None
    event_id: str
    event_key: str
    salience: float
    credibility: float
    path_weight: float

    def to_mapping(self) -> dict[str, object]:
        return {
            "credibility": self.credibility,
            "event_id": self.event_id,
            "event_key": self.event_key,
            "hop": self.hop,
            "node_id": self.node_id,
            "path_weight": self.path_weight,
            "salience": self.salience,
            "source_node_id": self.source_node_id,
        }


@dataclass(frozen=True, slots=True)
class AttentionRecord:
    hop: int
    node_id: str
    event_id: str
    event_key: str
    rank: int | None
    outcome: AttentionOutcome

    @property
    def accepted(self) -> bool:
        return self.outcome is AttentionOutcome.ACCEPTED

    def to_mapping(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "event_id": self.event_id,
            "event_key": self.event_key,
            "hop": self.hop,
            "node_id": self.node_id,
            "outcome": self.outcome.value,
            "rank": self.rank,
        }


@dataclass(frozen=True, slots=True)
class RepostRecord:
    hop: int
    node_id: str
    event_id: str
    event_key: str
    score: float
    reposted: bool

    def to_mapping(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "event_key": self.event_key,
            "hop": self.hop,
            "node_id": self.node_id,
            "reposted": self.reposted,
            "score": self.score,
        }


@dataclass(frozen=True, slots=True)
class BeliefUpdateRecord:
    hop: int
    node_id: str
    event_id: str | None
    event_key: str
    update_kind: BeliefUpdateKind
    belief_before: float
    belief_after: float
    corrected_event_id: str | None
    lineage_event_ids: tuple[str, ...]
    lineage_sha256s: tuple[str, ...]

    def to_mapping(self) -> dict[str, object]:
        return {
            "belief_after": self.belief_after,
            "belief_before": self.belief_before,
            "corrected_event_id": self.corrected_event_id,
            "event_id": self.event_id,
            "event_key": self.event_key,
            "hop": self.hop,
            "lineage_event_ids": list(self.lineage_event_ids),
            "lineage_sha256s": list(self.lineage_sha256s),
            "node_id": self.node_id,
            "update_kind": self.update_kind.value,
        }


@dataclass(frozen=True, slots=True)
class DiffusionResult:
    """Canonical state and transition records from one bounded diffusion run."""

    node_beliefs: tuple[NodeBelief, ...]
    exposures: tuple[ExposureRecord, ...]
    attention: tuple[AttentionRecord, ...]
    reposts: tuple[RepostRecord, ...]
    belief_updates: tuple[BeliefUpdateRecord, ...]
    max_hop_reached: int

    def to_mapping(self) -> dict[str, object]:
        return {
            "attention": [record.to_mapping() for record in self.attention],
            "belief_updates": [record.to_mapping() for record in self.belief_updates],
            "exposures": [record.to_mapping() for record in self.exposures],
            "max_hop_reached": self.max_hop_reached,
            "node_beliefs": [belief.to_mapping() for belief in self.node_beliefs],
            "reposts": [record.to_mapping() for record in self.reposts],
        }


@dataclass(frozen=True, slots=True)
class _Candidate:
    event: InformationEvent | ScenarioInformationEvent
    node_id: str
    source_node_id: str | None
    hop: int
    path_weight: float
    event_key: str

    def priority(self) -> tuple[float, float, float, str]:
        return (
            -self.event.magnitude,
            -self.event.credibility,
            -self.path_weight,
            self.event_key,
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "event_commitment": _commit(self.event.to_mapping()),
            "event_id": self.event.event_id,
            "event_key": self.event_key,
            "hop": self.hop,
            "node_id": self.node_id,
            "path_weight": self.path_weight,
            "source_node_id": self.source_node_id,
        }


def _semantic_event_key(
    config: PropagationConfig,
    *,
    hop: int,
    purpose: str,
) -> str:
    """Derive a pair-shared key whose schema has no world identifier."""
    return EventKey(
        experiment_id=config.experiment_id,
        seed=config.shared_seed,
        replicate_id=config.replicate_id,
        simulation_step=hop,
        purpose=f"social:{purpose}",
    ).value


def _candidate_key(
    config: PropagationConfig,
    *,
    event: InformationEvent | ScenarioInformationEvent,
    node_id: str,
    source_node_id: str | None,
    hop: int,
) -> str:
    source = "seed" if source_node_id is None else source_node_id
    purpose = f"exposure:{event.event_id}:{event.content_sha256}:{source}:{node_id}"
    return _semantic_event_key(config, hop=hop, purpose=purpose)


@dataclass(frozen=True, slots=True)
class SocialDiffusionEngine:
    """Synchronous, capacity-bounded, event-keyed diffusion engine."""

    graph: SocialGraph
    config: PropagationConfig
    audit_log: SocialAuditLog

    def __post_init__(self) -> None:
        if not isinstance(self.graph, ObservedSocialGraph | SyntheticSocialGraph):
            raise SocialDiffusionError("graph must be observed or synthetic")
        if not isinstance(self.config, PropagationConfig):
            raise SocialDiffusionError("config must be PropagationConfig")
        if not isinstance(self.audit_log, SocialAuditLog):
            raise SocialDiffusionError("audit_log must be SocialAuditLog")

    def _audit(
        self,
        *,
        stage: SocialAuditStage,
        hop: int,
        input_value: object,
        output_value: object,
        accepted_count: int,
        rejected_count: int,
        outcome: SocialAuditOutcome = SocialAuditOutcome.ALLOWED,
        rule_code: SocialAuditRule = SocialAuditRule.OK,
        event_key: str | None = None,
    ) -> None:
        key = event_key or _semantic_event_key(
            self.config, hop=hop, purpose=f"audit:{stage.value}"
        )
        self.audit_log.record(
            stage=stage,
            outcome=outcome,
            rule_code=rule_code,
            event_key=key,
            policy_sha256=self.config.policy_sha256,
            input_commitment=_commit(input_value),
            output_commitment=_commit(output_value),
            accepted_count=accepted_count,
            rejected_count=rejected_count,
        )

    def _validate_graph(self) -> None:
        if isinstance(self.graph, ObservedSocialGraph):
            if self.graph.cutoff_policy_sha256 != self.config.policy_sha256:
                raise SocialDiffusionTemporalError(
                    "observed graph cutoff policy does not match propagation policy"
                )
            if self.graph.trusted_cutoff != self.config.trusted_cutoff:
                raise SocialDiffusionTemporalError(
                    "observed graph cutoff does not match propagation cutoff"
                )
            graph_validation_time = (
                self.config.trusted_cutoff
                if self.config.scenario_event_receipt is not None
                else self.config.as_of_time
            )
            self.graph.validate_available(as_of_time=graph_validation_time)
        elif isinstance(self.graph, SyntheticSocialGraph):
            if self.graph.synthetic_uncalibrated is not True:
                raise SocialDiffusionError(
                    "synthetic graph provenance marker is absent"
                )
        else:
            raise SocialDiffusionError("unsupported social graph provenance")

    def _validate_event(
        self, event: InformationEvent | ScenarioInformationEvent
    ) -> None:
        if event.cutoff_policy_sha256 != self.config.policy_sha256:
            raise SocialDiffusionTemporalError(
                "event cutoff policy does not match propagation policy"
            )
        if event.as_of_time != self.config.as_of_time:
            raise SocialDiffusionTemporalError(
                "event as_of_time does not match propagation as_of_time"
            )
        if event.trusted_cutoff != self.config.trusted_cutoff:
            raise SocialDiffusionTemporalError(
                "event cutoff does not match propagation cutoff"
            )
        if event.event_time > self.config.as_of_time:
            raise SocialDiffusionTemporalError("event_time is after as_of_time")
        if isinstance(event, ScenarioInformationEvent):
            if (
                self.config.scenario_event_receipt is None
                or event.receipt != self.config.scenario_event_receipt
            ):
                raise SocialDiffusionTemporalError(
                    "generated event receipt does not match propagation authority"
                )
            return
        if self.config.scenario_event_receipt is not None:
            raise SocialDiffusionTemporalError(
                "an observed record cannot enter generated scenario propagation"
            )
        if event.available_time > self.config.as_of_time:
            raise SocialDiffusionTemporalError("available_time is after as_of_time")
        if event.available_time > self.config.trusted_cutoff:
            raise SocialDiffusionTemporalError("available_time is after cutoff")

    def _validated_inputs(
        self,
        seeded_events: tuple[SeededInformationEvent, ...],
        initial_beliefs: tuple[NodeBelief, ...],
    ) -> dict[str, NodeBelief]:
        if not isinstance(seeded_events, tuple) or not all(
            isinstance(item, SeededInformationEvent) for item in seeded_events
        ):
            raise SocialDiffusionError(
                "seeded_events must be a tuple of SeededInformationEvent values"
            )
        if not isinstance(initial_beliefs, tuple) or not all(
            isinstance(item, NodeBelief) for item in initial_beliefs
        ):
            raise SocialDiffusionError(
                "initial_beliefs must be a tuple of NodeBelief values"
            )
        self._validate_graph()
        node_set = frozenset(self.graph.nodes)
        seen_seeds: set[tuple[str, str]] = set()
        event_commitments: dict[str, str] = {}
        for seeded in seeded_events:
            if seeded.seed_node_id not in node_set:
                raise SocialDiffusionError("seed node is absent from social graph")
            self._validate_event(seeded.event)
            pair = (seeded.seed_node_id, seeded.event.event_id)
            if pair in seen_seeds:
                raise SocialDiffusionError("duplicate seeded node-event pair")
            seen_seeds.add(pair)
            commitment = _commit(seeded.event.to_mapping())
            prior = event_commitments.setdefault(seeded.event.event_id, commitment)
            if prior != commitment:
                raise SocialDiffusionError("one event_id has conflicting commitments")

        beliefs: dict[str, NodeBelief] = {
            node_id: NodeBelief(node_id=node_id) for node_id in self.graph.nodes
        }
        seen_belief_nodes: set[str] = set()
        for belief in initial_beliefs:
            if belief.node_id not in node_set:
                raise SocialDiffusionError("initial belief node is absent from graph")
            if belief.node_id in seen_belief_nodes:
                raise SocialDiffusionError("duplicate initial belief node")
            seen_belief_nodes.add(belief.node_id)
            beliefs[belief.node_id] = belief
        return beliefs

    def _attention(
        self,
        candidates: tuple[_Candidate, ...],
        seen: set[tuple[str, str]],
    ) -> tuple[tuple[_Candidate, ...], tuple[AttentionRecord, ...]]:
        grouped: dict[tuple[str, str], list[_Candidate]] = defaultdict(list)
        for candidate in candidates:
            grouped[(candidate.node_id, candidate.event.event_id)].append(candidate)

        winners: dict[str, list[_Candidate]] = defaultdict(list)
        records: list[AttentionRecord] = []
        for pair in sorted(grouped):
            paths = sorted(grouped[pair], key=lambda candidate: candidate.priority())
            if pair in seen:
                records.extend(
                    AttentionRecord(
                        hop=candidate.hop,
                        node_id=candidate.node_id,
                        event_id=candidate.event.event_id,
                        event_key=candidate.event_key,
                        rank=None,
                        outcome=AttentionOutcome.ALREADY_EXPOSED,
                    )
                    for candidate in paths
                )
                continue
            seen.add(pair)
            winners[pair[0]].append(paths[0])
            records.extend(
                AttentionRecord(
                    hop=candidate.hop,
                    node_id=candidate.node_id,
                    event_id=candidate.event.event_id,
                    event_key=candidate.event_key,
                    rank=None,
                    outcome=AttentionOutcome.DUPLICATE_PATH,
                )
                for candidate in paths[1:]
            )

        accepted: list[_Candidate] = []
        for node_id in sorted(winners):
            ranked = sorted(
                winners[node_id], key=lambda candidate: candidate.priority()
            )
            for index, candidate in enumerate(ranked, start=1):
                is_accepted = index <= self.config.attention_capacity
                records.append(
                    AttentionRecord(
                        hop=candidate.hop,
                        node_id=candidate.node_id,
                        event_id=candidate.event.event_id,
                        event_key=candidate.event_key,
                        rank=index,
                        outcome=(
                            AttentionOutcome.ACCEPTED
                            if is_accepted
                            else AttentionOutcome.CAPACITY_REJECTED
                        ),
                    )
                )
                if is_accepted:
                    accepted.append(candidate)
        return (
            tuple(
                sorted(
                    accepted,
                    key=lambda candidate: (candidate.node_id, candidate.priority()),
                )
            ),
            tuple(
                sorted(records, key=lambda record: (record.node_id, record.event_key))
            ),
        )

    def _decay_beliefs(
        self,
        beliefs: dict[str, NodeBelief],
        *,
        hop: int,
    ) -> tuple[dict[str, NodeBelief], list[BeliefUpdateRecord]]:
        if hop == 0 or self.config.belief_decay == 0.0:
            return dict(beliefs), []
        retained = 1.0 - self.config.belief_decay
        updated: dict[str, NodeBelief] = {}
        records: list[BeliefUpdateRecord] = []
        for node_id in sorted(beliefs):
            before = beliefs[node_id]
            components = tuple(
                BeliefComponent(
                    event_id=component.event_id,
                    contribution=component.contribution * retained,
                    lineage_event_ids=component.lineage_event_ids,
                    lineage_sha256s=component.lineage_sha256s,
                )
                for component in before.components
            )
            after = NodeBelief(node_id=node_id, components=components)
            updated[node_id] = after
            if before.components:
                lineage_ids = tuple(
                    item
                    for component in before.components
                    for item in component.lineage_event_ids
                )
                lineage_hashes = tuple(
                    item
                    for component in before.components
                    for item in component.lineage_sha256s
                )
                records.append(
                    BeliefUpdateRecord(
                        hop=hop,
                        node_id=node_id,
                        event_id=None,
                        event_key=_semantic_event_key(
                            self.config, hop=hop, purpose=f"belief_decay:{node_id}"
                        ),
                        update_kind=BeliefUpdateKind.DECAY,
                        belief_before=before.belief,
                        belief_after=after.belief,
                        corrected_event_id=None,
                        lineage_event_ids=lineage_ids,
                        lineage_sha256s=lineage_hashes,
                    )
                )
        return updated, records

    def _apply_candidate(
        self,
        belief: NodeBelief,
        candidate: _Candidate,
        prior_event_ids: frozenset[str],
    ) -> tuple[NodeBelief, BeliefUpdateRecord]:
        event = candidate.event
        components = {component.event_id: component for component in belief.components}
        corrected_event_id: str | None = None
        if event.event_kind is InformationEventKind.CORRECTION:
            corrected_event_id = event.corrects_event_id
            if corrected_event_id is None or corrected_event_id not in prior_event_ids:
                raise MissingCorrectionTargetError(
                    "correction target is absent from node belief"
                )
            target = components[corrected_event_id]
            fraction = (
                self.config.correction_strength
                * min(1.0, event.magnitude)
                * event.credibility
                * candidate.path_weight
            )
            updated_component = BeliefComponent(
                event_id=target.event_id,
                contribution=target.contribution * (1.0 - fraction),
                lineage_event_ids=(*target.lineage_event_ids, event.event_id),
                lineage_sha256s=(*target.lineage_sha256s, event.lineage_sha256),
            )
            components[target.event_id] = updated_component
            kind = BeliefUpdateKind.CORRECTION
            lineage_ids = updated_component.lineage_event_ids
            lineage_hashes = updated_component.lineage_sha256s
        else:
            increment = (
                event.sentiment
                * event.magnitude
                * event.credibility
                * candidate.path_weight
            )
            prior = components.get(event.event_id)
            if prior is None:
                updated_component = BeliefComponent(
                    event_id=event.event_id,
                    contribution=increment,
                    lineage_event_ids=(event.event_id,),
                    lineage_sha256s=(event.lineage_sha256,),
                )
            else:
                updated_component = BeliefComponent(
                    event_id=prior.event_id,
                    contribution=prior.contribution + increment,
                    lineage_event_ids=prior.lineage_event_ids,
                    lineage_sha256s=prior.lineage_sha256s,
                )
            components[event.event_id] = updated_component
            kind = BeliefUpdateKind.SIGNAL
            lineage_ids = updated_component.lineage_event_ids
            lineage_hashes = updated_component.lineage_sha256s

        updated = NodeBelief(
            node_id=belief.node_id,
            components=tuple(components.values()),
        )
        return updated, BeliefUpdateRecord(
            hop=candidate.hop,
            node_id=belief.node_id,
            event_id=event.event_id,
            event_key=candidate.event_key,
            update_kind=kind,
            belief_before=belief.belief,
            belief_after=updated.belief,
            corrected_event_id=corrected_event_id,
            lineage_event_ids=lineage_ids,
            lineage_sha256s=lineage_hashes,
        )

    def run(
        self,
        seeded_events: tuple[SeededInformationEvent, ...],
        *,
        initial_beliefs: tuple[NodeBelief, ...] = (),
    ) -> DiffusionResult:
        """Run seed ingestion and bounded synchronous propagation by hop."""
        valid_seed_schema = isinstance(seeded_events, tuple) and all(
            isinstance(item, SeededInformationEvent) for item in seeded_events
        )
        valid_belief_schema = isinstance(initial_beliefs, tuple) and all(
            isinstance(item, NodeBelief) for item in initial_beliefs
        )
        if not valid_seed_schema or not valid_belief_schema:
            rejected_input = {
                "config_commitment": _commit(self.config.to_mapping()),
                "graph_commitment": _commit(self.graph.to_mapping()),
                "initial_belief_schema_valid": valid_belief_schema,
                "seed_schema_valid": valid_seed_schema,
            }
            self._audit(
                stage=SocialAuditStage.PROPAGATION_INPUT,
                hop=0,
                input_value=rejected_input,
                output_value=rejected_input,
                accepted_count=0,
                rejected_count=1,
                outcome=SocialAuditOutcome.REJECTED,
                rule_code=SocialAuditRule.SCHEMA_VIOLATION,
            )
            raise SocialDiffusionError("propagation inputs have an invalid schema")

        seeded_events = tuple(
            sorted(
                seeded_events,
                key=lambda item: (
                    item.seed_node_id,
                    item.event.event_id,
                    _commit(item.event.to_mapping()),
                ),
            )
        )
        initial_beliefs = tuple(sorted(initial_beliefs, key=lambda item: item.node_id))
        input_value = {
            "config": self.config.to_mapping(),
            "graph": self.graph.to_mapping(),
            "initial_beliefs": [item.to_mapping() for item in initial_beliefs],
            "seeded_events": [item.to_mapping() for item in seeded_events],
        }
        try:
            beliefs = self._validated_inputs(seeded_events, initial_beliefs)
        except (
            InformationEventError,
            SocialDiffusionError,
            SocialGraphError,
        ) as error:
            temporal = isinstance(
                error,
                InformationEventTemporalError
                | SocialDiffusionTemporalError
                | SocialGraphTemporalError,
            )
            self._audit(
                stage=SocialAuditStage.PROPAGATION_INPUT,
                hop=0,
                input_value=input_value,
                output_value=input_value,
                accepted_count=0,
                rejected_count=len(seeded_events),
                outcome=SocialAuditOutcome.REJECTED,
                rule_code=(
                    SocialAuditRule.TEMPORAL_BOUNDARY
                    if temporal
                    else SocialAuditRule.SCHEMA_VIOLATION
                ),
            )
            raise

        self._audit(
            stage=SocialAuditStage.PROPAGATION_INPUT,
            hop=0,
            input_value=input_value,
            output_value={"input_commitment": _commit(input_value)},
            accepted_count=len(seeded_events),
            rejected_count=0,
        )
        candidates = tuple(
            _Candidate(
                event=seeded.event,
                node_id=seeded.seed_node_id,
                source_node_id=None,
                hop=0,
                path_weight=1.0,
                event_key=_candidate_key(
                    self.config,
                    event=seeded.event,
                    node_id=seeded.seed_node_id,
                    source_node_id=None,
                    hop=0,
                ),
            )
            for seeded in seeded_events
        )
        seen: set[tuple[str, str]] = set()
        exposures: list[ExposureRecord] = []
        attention: list[AttentionRecord] = []
        reposts: list[RepostRecord] = []
        belief_updates: list[BeliefUpdateRecord] = []
        max_hop_reached = 0

        for hop in range(self.config.max_hops + 1):
            if hop > 0 and not candidates:
                break
            max_hop_reached = hop
            canonical_candidates = tuple(
                sorted(
                    candidates,
                    key=lambda item: (item.node_id, item.priority()),
                )
            )
            hop_exposures = tuple(
                ExposureRecord(
                    hop=hop,
                    node_id=candidate.node_id,
                    source_node_id=candidate.source_node_id,
                    event_id=candidate.event.event_id,
                    event_key=candidate.event_key,
                    salience=candidate.event.magnitude,
                    credibility=candidate.event.credibility,
                    path_weight=candidate.path_weight,
                )
                for candidate in canonical_candidates
            )
            exposures.extend(hop_exposures)
            self._audit(
                stage=SocialAuditStage.EXPOSURE,
                hop=hop,
                input_value=[item.to_mapping() for item in canonical_candidates],
                output_value=[item.to_mapping() for item in hop_exposures],
                accepted_count=len(hop_exposures),
                rejected_count=0,
            )

            accepted, hop_attention = self._attention(canonical_candidates, seen)
            attention.extend(hop_attention)
            attention_rejected = sum(not record.accepted for record in hop_attention)
            capacity_rejected = sum(
                record.outcome is AttentionOutcome.CAPACITY_REJECTED
                for record in hop_attention
            )
            self._audit(
                stage=SocialAuditStage.ATTENTION,
                hop=hop,
                input_value=[item.to_mapping() for item in hop_exposures],
                output_value=[item.to_mapping() for item in hop_attention],
                accepted_count=len(accepted),
                rejected_count=attention_rejected,
                rule_code=(
                    SocialAuditRule.ATTENTION_CAPACITY
                    if capacity_rejected
                    else SocialAuditRule.OK
                ),
            )

            belief_input = [belief.to_mapping() for belief in beliefs.values()]
            beliefs, decay_records = self._decay_beliefs(beliefs, hop=hop)
            hop_updates = list(decay_records)
            prior_ids = {
                node_id: frozenset(
                    component.event_id for component in belief.components
                )
                for node_id, belief in beliefs.items()
            }
            try:
                for candidate in accepted:
                    updated, record = self._apply_candidate(
                        beliefs[candidate.node_id],
                        candidate,
                        prior_ids[candidate.node_id],
                    )
                    beliefs[candidate.node_id] = updated
                    hop_updates.append(record)
            except SocialDiffusionError as error:
                self._audit(
                    stage=SocialAuditStage.BELIEF_UPDATE,
                    hop=hop,
                    input_value=belief_input,
                    output_value=belief_input,
                    accepted_count=0,
                    rejected_count=1,
                    outcome=SocialAuditOutcome.REJECTED,
                    rule_code=(
                        SocialAuditRule.MISSING_CORRECTION_TARGET
                        if isinstance(error, MissingCorrectionTargetError)
                        else SocialAuditRule.SCHEMA_VIOLATION
                    ),
                    event_key=candidate.event_key,
                )
                raise
            belief_updates.extend(hop_updates)
            belief_output = [belief.to_mapping() for belief in beliefs.values()]
            self._audit(
                stage=SocialAuditStage.BELIEF_UPDATE,
                hop=hop,
                input_value=belief_input,
                output_value={
                    "beliefs": belief_output,
                    "updates": [item.to_mapping() for item in hop_updates],
                },
                accepted_count=len(hop_updates),
                rejected_count=0,
            )

            next_candidates: list[_Candidate] = []
            hop_reposts: list[RepostRecord] = []
            for candidate in accepted:
                score = (
                    candidate.event.magnitude
                    * candidate.event.credibility
                    * candidate.path_weight
                )
                reposted = (
                    self.config.social_enabled
                    and hop < self.config.max_hops
                    and score >= self.config.repost_threshold
                )
                hop_reposts.append(
                    RepostRecord(
                        hop=hop,
                        node_id=candidate.node_id,
                        event_id=candidate.event.event_id,
                        event_key=candidate.event_key,
                        score=score,
                        reposted=reposted,
                    )
                )
                if reposted:
                    for edge in self.graph.outgoing_edges(candidate.node_id):
                        path_weight = candidate.path_weight * edge.weight
                        next_candidates.append(
                            _Candidate(
                                event=candidate.event,
                                node_id=edge.target_id,
                                source_node_id=edge.source_id,
                                hop=hop + 1,
                                path_weight=path_weight,
                                event_key=_candidate_key(
                                    self.config,
                                    event=candidate.event,
                                    node_id=edge.target_id,
                                    source_node_id=edge.source_id,
                                    hop=hop + 1,
                                ),
                            )
                        )
            hop_reposts.sort(key=lambda record: (record.node_id, record.event_key))
            reposts.extend(hop_reposts)
            reposted_count = sum(record.reposted for record in hop_reposts)
            below_threshold_count = sum(
                record.score < self.config.repost_threshold for record in hop_reposts
            )
            self._audit(
                stage=SocialAuditStage.REPOST,
                hop=hop,
                input_value=[item.to_mapping() for item in accepted],
                output_value=[item.to_mapping() for item in hop_reposts],
                accepted_count=reposted_count,
                rejected_count=len(hop_reposts) - reposted_count,
                rule_code=(
                    SocialAuditRule.BELOW_THRESHOLD
                    if below_threshold_count
                    else SocialAuditRule.OK
                ),
            )
            if not self.config.social_enabled:
                break
            candidates = tuple(next_candidates)

        result = DiffusionResult(
            node_beliefs=tuple(beliefs[node_id] for node_id in sorted(beliefs)),
            exposures=tuple(exposures),
            attention=tuple(attention),
            reposts=tuple(reposts),
            belief_updates=tuple(belief_updates),
            max_hop_reached=max_hop_reached,
        )
        self._audit(
            stage=SocialAuditStage.PROPAGATION_OUTPUT,
            hop=max_hop_reached,
            input_value={
                "attention_count": len(attention),
                "belief_update_count": len(belief_updates),
                "exposure_count": len(exposures),
                "repost_count": len(reposts),
            },
            output_value=result.to_mapping(),
            accepted_count=len(result.node_beliefs),
            rejected_count=0,
        )
        return result


__all__ = [
    "AttentionOutcome",
    "AttentionRecord",
    "BeliefComponent",
    "BeliefUpdateKind",
    "BeliefUpdateRecord",
    "DiffusionResult",
    "ExposureRecord",
    "MissingCorrectionTargetError",
    "NodeBelief",
    "PropagationConfig",
    "RepostRecord",
    "SeededInformationEvent",
    "SocialDiffusionEngine",
    "SocialDiffusionError",
    "SocialDiffusionTemporalError",
]
