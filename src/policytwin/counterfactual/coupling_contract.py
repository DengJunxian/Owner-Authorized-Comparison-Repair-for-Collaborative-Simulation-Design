"""Declarative cross-world coupling contracts and independent consumption audit.

The scenario specification is frozen before either world executes.  It is the
only source of expected events.  Runtime code may submit observations, but it
cannot add to or revise the expected set used by :func:`audit_pair`.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import NoReturn, Self

COUPLING_CONTRACT_SCHEMA_VERSION = "policytwin.coupling-contract.v1"
EVENT_ONTOLOGY_VERSION = "policytwin.paired-market-events.v1"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class CouplingContractError(ValueError):
    """Raised when a contract or observation violates its closed schema."""


def _fail(message: str) -> NoReturn:
    raise CouplingContractError(message)


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        _fail(f"{field} must be a stable ASCII identifier")
    return value


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CouplingContractError("value is not canonical JSON") from error


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


class World(StrEnum):
    BASELINE = "baseline"
    POLICY = "policy"


class WorldScope(StrEnum):
    SHARED = "shared"
    BASELINE_ONLY = "baseline_only"
    POLICY_ONLY = "policy_only"

    def applies_to(self, world: World) -> bool:
        return (
            self is WorldScope.SHARED
            or (self is WorldScope.BASELINE_ONLY and world is World.BASELINE)
            or (self is WorldScope.POLICY_ONLY and world is World.POLICY)
        )


class Requirement(StrEnum):
    REQUIRED = "required"
    OPTIONAL = "optional"


class ReceiptState(StrEnum):
    CONSUMED = "consumed"
    MASKED = "masked"
    NOT_APPLICABLE = "not_applicable"
    MISSING = "missing"
    DUPLICATE = "duplicate"
    UNEXPECTED = "unexpected"


class ObservationState(StrEnum):
    CONSUMED = "consumed"
    MASKED = "masked"


class ConformanceViolationCode(StrEnum):
    """Machine-readable reasons why ``Conform(C, O0, O1)`` is false."""

    MISSING = "missing"
    DUPLICATE = "duplicate"
    UNEXPECTED = "unexpected"
    SEMANTIC_KEY_SUBSTITUTION = "semantic_key_substitution"
    EVENT_TYPE_MISMATCH = "event_type_mismatch"
    DRAW_ROLE_MISMATCH = "draw_role_mismatch"
    DISTRIBUTION_FAMILY_MISMATCH = "distribution_family_mismatch"
    DISTRIBUTION_PARAMETERS_MISMATCH = "distribution_parameters_mismatch"
    COUPLING_GROUP_MISMATCH = "coupling_group_mismatch"
    WORLD_SCOPE_MISMATCH = "world_scope_mismatch"
    REQUIREMENT_MISMATCH = "requirement_mismatch"
    OBSERVED_METADATA_MISSING = "observed_metadata_missing"
    INVALID_STATUS = "invalid_status"
    SHARED_VALUE_MISMATCH = "shared_value_mismatch"


@dataclass(frozen=True, slots=True)
class EventDeclaration:
    """One event template in a scenario specification."""

    event_type: str
    draw_role: str
    distribution_family: str
    distribution_parameters: tuple[tuple[str, int | float | str], ...]
    coupling_group: str
    world_scope: WorldScope
    requirement: Requirement

    def __post_init__(self) -> None:
        for field in (
            "event_type",
            "draw_role",
            "distribution_family",
            "coupling_group",
        ):
            _identifier(getattr(self, field), field)
        if type(self.distribution_parameters) is not tuple:
            _fail("distribution_parameters must be an exact tuple")
        names: list[str] = []
        for name, value in self.distribution_parameters:
            names.append(_identifier(name, "distribution parameter"))
            if isinstance(value, bool) or not isinstance(value, int | float | str):
                _fail("distribution parameter values must be scalar JSON values")
            if isinstance(value, float) and not math.isfinite(value):
                _fail("distribution parameter values must be finite")
        if len(names) != len(set(names)) or names != sorted(names):
            _fail("distribution parameters must be unique and sorted")

    def to_mapping(self) -> dict[str, object]:
        return {
            "coupling_group": self.coupling_group,
            "distribution_family": self.distribution_family,
            "distribution_parameters": dict(self.distribution_parameters),
            "draw_role": self.draw_role,
            "event_type": self.event_type,
            "required_or_optional": self.requirement.value,
            "world_scope": self.world_scope.value,
        }


@dataclass(frozen=True, slots=True)
class ScenarioSpecification:
    """Frozen generator input for an expected event set."""

    base_experiment_id: str
    scenario_namespace: str
    logical_epochs: tuple[int, ...]
    entity_ids: tuple[str, ...]
    declarations: tuple[EventDeclaration, ...]
    configuration_version: str
    event_ontology_version: str = EVENT_ONTOLOGY_VERSION
    semantic_version: str = "1.0.0"

    def __post_init__(self) -> None:
        for field in (
            "base_experiment_id",
            "scenario_namespace",
            "configuration_version",
            "event_ontology_version",
            "semantic_version",
        ):
            _identifier(getattr(self, field), field)
        if not self.logical_epochs or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in self.logical_epochs
        ):
            _fail("logical_epochs must be non-empty non-negative integers")
        if tuple(sorted(set(self.logical_epochs))) != self.logical_epochs:
            _fail("logical_epochs must be strictly increasing")
        if not self.entity_ids or any(
            _IDENTIFIER.fullmatch(item) is None for item in self.entity_ids
        ):
            _fail("entity_ids must be non-empty stable identifiers")
        if len(set(self.entity_ids)) != len(self.entity_ids):
            _fail("entity_ids must be unique")
        if not self.declarations or len(set(self.declarations)) != len(
            self.declarations
        ):
            _fail("declarations must be non-empty and unique")

    def to_mapping(self) -> dict[str, object]:
        return {
            "base_experiment_id": self.base_experiment_id,
            "configuration_version": self.configuration_version,
            "declarations": [item.to_mapping() for item in self.declarations],
            "entity_ids": list(self.entity_ids),
            "event_ontology_version": self.event_ontology_version,
            "logical_epochs": list(self.logical_epochs),
            "scenario_namespace": self.scenario_namespace,
            "semantic_version": self.semantic_version,
        }

    @property
    def sha256(self) -> str:
        return _digest(self.to_mapping())


@dataclass(frozen=True, slots=True)
class SemanticEventAddress:
    """Complete semantic identity declared before execution."""

    scenario_namespace: str
    replicate_id: int
    logical_epoch: int
    entity_id: str
    event_type: str
    draw_role: str
    distribution_family: str
    distribution_parameters: tuple[tuple[str, int | float | str], ...]
    coupling_group: str
    world_scope: WorldScope
    requirement: Requirement
    semantic_version: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.replicate_id, bool)
            or not isinstance(self.replicate_id, int)
            or self.replicate_id < 0
        ):
            _fail("replicate_id must be a non-negative integer")
        if (
            isinstance(self.logical_epoch, bool)
            or not isinstance(self.logical_epoch, int)
            or self.logical_epoch < 0
        ):
            _fail("logical_epoch must be a non-negative integer")
        EventDeclaration(
            event_type=self.event_type,
            draw_role=self.draw_role,
            distribution_family=self.distribution_family,
            distribution_parameters=self.distribution_parameters,
            coupling_group=self.coupling_group,
            world_scope=self.world_scope,
            requirement=self.requirement,
        )
        for field in ("scenario_namespace", "entity_id", "semantic_version"):
            _identifier(getattr(self, field), field)

    def to_mapping(self) -> dict[str, object]:
        return {
            "coupling_group": self.coupling_group,
            "distribution_family": self.distribution_family,
            "distribution_parameters": dict(self.distribution_parameters),
            "draw_role": self.draw_role,
            "entity_id": self.entity_id,
            "event_type": self.event_type,
            "logical_epoch": self.logical_epoch,
            "replicate_id": self.replicate_id,
            "required_or_optional": self.requirement.value,
            "scenario_namespace": self.scenario_namespace,
            "semantic_version": self.semantic_version,
            "world_scope": self.world_scope.value,
        }

    @property
    def identity(self) -> str:
        return f"pcc1:{_digest(self.to_mapping())}"


@dataclass(frozen=True, slots=True)
class CouplingContract:
    """Closed expected event set committed before world execution."""

    specification_sha256: str
    events: tuple[SemanticEventAddress, ...]
    base_experiment_id: str
    event_ontology_version: str
    configuration_version: str
    schema_version: str = COUPLING_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != COUPLING_CONTRACT_SCHEMA_VERSION:
            _fail("unsupported coupling contract schema version")
        if _SHA256.fullmatch(self.specification_sha256) is None:
            _fail("specification_sha256 must be a lowercase SHA-256")
        for field in (
            "base_experiment_id",
            "event_ontology_version",
            "configuration_version",
        ):
            _identifier(getattr(self, field), field)
        identities = [item.identity for item in self.events]
        if not identities or len(identities) != len(set(identities)):
            _fail("contract event identities must be non-empty and unique")
        if tuple(sorted(identities)) != tuple(identities):
            _fail("contract events must use canonical identity order")

    def to_mapping(self) -> dict[str, object]:
        return {
            "base_experiment_id": self.base_experiment_id,
            "configuration_version": self.configuration_version,
            "event_ontology_version": self.event_ontology_version,
            "events": [item.to_mapping() for item in self.events],
            "schema_version": self.schema_version,
            "specification_sha256": self.specification_sha256,
        }

    @property
    def sha256(self) -> str:
        return _digest(self.to_mapping())

    @property
    def experiment_id(self) -> str:
        return f"{self.base_experiment_id}-{self.sha256[:16]}"

    @property
    def by_identity(self) -> Mapping[str, SemanticEventAddress]:
        return MappingProxyType({item.identity: item for item in self.events})


def generate_contract(
    specification: ScenarioSpecification, replicate_id: int
) -> CouplingContract:
    """Generate expected events solely from a pre-execution specification."""
    if (
        isinstance(replicate_id, bool)
        or not isinstance(replicate_id, int)
        or replicate_id < 0
    ):
        _fail("replicate_id must be a non-negative integer")
    events = tuple(
        sorted(
            (
                SemanticEventAddress(
                    scenario_namespace=specification.scenario_namespace,
                    replicate_id=replicate_id,
                    logical_epoch=epoch,
                    entity_id=entity,
                    event_type=declaration.event_type,
                    draw_role=declaration.draw_role,
                    distribution_family=declaration.distribution_family,
                    distribution_parameters=declaration.distribution_parameters,
                    coupling_group=declaration.coupling_group,
                    world_scope=declaration.world_scope,
                    requirement=declaration.requirement,
                    semantic_version=specification.semantic_version,
                )
                for epoch in specification.logical_epochs
                for entity in specification.entity_ids
                for declaration in specification.declarations
            ),
            key=lambda item: item.identity,
        )
    )
    return CouplingContract(
        specification_sha256=specification.sha256,
        events=events,
        base_experiment_id=specification.base_experiment_id,
        event_ontology_version=specification.event_ontology_version,
        configuration_version=specification.configuration_version,
    )


@dataclass(frozen=True, slots=True)
class ConsumptionObservation:
    """Runtime-submitted observation; it cannot declare expected events."""

    world: World
    event_identity: str
    state: ObservationState
    value_commitment: str | None = None
    observed_event: SemanticEventAddress | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.event_identity, str) or not self.event_identity:
            _fail("event_identity must be non-empty")
        if self.state is ObservationState.CONSUMED:
            if (
                self.value_commitment is None
                or _SHA256.fullmatch(self.value_commitment) is None
            ):
                _fail("consumed observations require a value commitment")
        elif self.value_commitment is not None:
            _fail("masked observations must not contain a value commitment")

    @classmethod
    def consumed(
        cls, world: World, event: SemanticEventAddress, value: int | float | str
    ) -> Self:
        return cls(
            world=world,
            event_identity=event.identity,
            state=ObservationState.CONSUMED,
            value_commitment=_digest({"value": value}),
            observed_event=event,
        )

    @classmethod
    def masked(cls, world: World, event: SemanticEventAddress) -> Self:
        """Record an explicit non-consumption state for an optional event."""
        return cls(
            world=world,
            event_identity=event.identity,
            state=ObservationState.MASKED,
            observed_event=event,
        )


@dataclass(frozen=True, slots=True)
class ConsumptionReceipt:
    event_identity: str
    world: World
    state: ReceiptState
    occurrence_count: int
    value_commitments: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ConformanceViolation:
    code: ConformanceViolationCode
    world: World | None
    event_identity: str
    detail: str


@dataclass(frozen=True, slots=True)
class WorldConformance:
    world: World
    receipts: tuple[ConsumptionReceipt, ...]
    violations: tuple[ConformanceViolation, ...]
    passed: bool


@dataclass(frozen=True, slots=True)
class PairConformance:
    contract_sha256: str
    baseline: WorldConformance
    policy: WorldConformance
    shared_value_mismatches: tuple[str, ...]
    violations: tuple[ConformanceViolation, ...]
    passed: bool


def _metadata_violations(
    expected: SemanticEventAddress,
    observation: ConsumptionObservation,
) -> tuple[ConformanceViolation, ...]:
    observed = observation.observed_event
    if observed is None:
        return (
            ConformanceViolation(
                code=ConformanceViolationCode.OBSERVED_METADATA_MISSING,
                world=observation.world,
                event_identity=observation.event_identity,
                detail="expected-event observation omitted semantic metadata",
            ),
        )
    result: list[ConformanceViolation] = []
    if observed.identity != observation.event_identity:
        result.append(
            ConformanceViolation(
                code=ConformanceViolationCode.SEMANTIC_KEY_SUBSTITUTION,
                world=observation.world,
                event_identity=observation.event_identity,
                detail="observed semantic address does not commit to receipt identity",
            )
        )
    comparisons: tuple[
        tuple[str, object, object, ConformanceViolationCode], ...
    ] = (
        (
            "event_type",
            expected.event_type,
            observed.event_type,
            ConformanceViolationCode.EVENT_TYPE_MISMATCH,
        ),
        (
            "draw_role",
            expected.draw_role,
            observed.draw_role,
            ConformanceViolationCode.DRAW_ROLE_MISMATCH,
        ),
        (
            "distribution_family",
            expected.distribution_family,
            observed.distribution_family,
            ConformanceViolationCode.DISTRIBUTION_FAMILY_MISMATCH,
        ),
        (
            "distribution_parameters",
            expected.distribution_parameters,
            observed.distribution_parameters,
            ConformanceViolationCode.DISTRIBUTION_PARAMETERS_MISMATCH,
        ),
        (
            "coupling_group",
            expected.coupling_group,
            observed.coupling_group,
            ConformanceViolationCode.COUPLING_GROUP_MISMATCH,
        ),
        (
            "world_scope",
            expected.world_scope,
            observed.world_scope,
            ConformanceViolationCode.WORLD_SCOPE_MISMATCH,
        ),
        (
            "requirement",
            expected.requirement,
            observed.requirement,
            ConformanceViolationCode.REQUIREMENT_MISMATCH,
        ),
    )
    for field, declared, actual, code in comparisons:
        if declared != actual:
            result.append(
                ConformanceViolation(
                    code=code,
                    world=observation.world,
                    event_identity=observation.event_identity,
                    detail=f"{field} differs from the frozen contract",
                )
            )
    # The digest includes the remaining semantic address coordinates.  This
    # catches substitutions of epoch, entity, namespace, replicate, or version.
    if observed.to_mapping() != expected.to_mapping() and not result:
        result.append(
            ConformanceViolation(
                code=ConformanceViolationCode.SEMANTIC_KEY_SUBSTITUTION,
                world=observation.world,
                event_identity=observation.event_identity,
                detail="semantic address coordinates differ from the contract",
            )
        )
    return tuple(result)


def audit_world(
    contract: CouplingContract,
    world: World,
    observations: Sequence[ConsumptionObservation],
) -> WorldConformance:
    """Compare independent expected events with runtime observations."""
    by_expected = contract.by_identity
    grouped: dict[str, list[ConsumptionObservation]] = defaultdict(list)
    violations: list[ConformanceViolation] = []
    for item in observations:
        if item.world is not world:
            _fail("observations for different worlds cannot be mixed")
        grouped[item.event_identity].append(item)
    receipts: list[ConsumptionReceipt] = []
    for event in contract.events:
        selected = grouped.pop(event.identity, [])
        applies = event.world_scope.applies_to(world)
        commitments = tuple(
            item.value_commitment
            for item in selected
            if item.value_commitment is not None
        )
        if not applies:
            state = ReceiptState.UNEXPECTED if selected else ReceiptState.NOT_APPLICABLE
            if selected:
                violations.append(
                    ConformanceViolation(
                        code=ConformanceViolationCode.WORLD_SCOPE_MISMATCH,
                        world=world,
                        event_identity=event.identity,
                        detail=(
                            "event was observed in a world outside its declared scope"
                        ),
                    )
                )
        elif not selected:
            state = ReceiptState.MISSING
            violations.append(
                ConformanceViolation(
                    code=ConformanceViolationCode.MISSING,
                    world=world,
                    event_identity=event.identity,
                    detail="applicable event has no observation",
                )
            )
        elif len(selected) > 1:
            state = ReceiptState.DUPLICATE
            violations.append(
                ConformanceViolation(
                    code=ConformanceViolationCode.DUPLICATE,
                    world=world,
                    event_identity=event.identity,
                    detail="semantic address was observed more than once",
                )
            )
        elif selected[0].state is ObservationState.MASKED:
            state = ReceiptState.MASKED
        else:
            state = ReceiptState.CONSUMED
        for observation in selected:
            violations.extend(_metadata_violations(event, observation))
        if (
            applies
            and event.requirement is Requirement.REQUIRED
            and state is not ReceiptState.CONSUMED
            and state not in {ReceiptState.MISSING, ReceiptState.DUPLICATE}
        ):
            violations.append(
                ConformanceViolation(
                    code=ConformanceViolationCode.INVALID_STATUS,
                    world=world,
                    event_identity=event.identity,
                    detail="required applicable event must be consumed exactly once",
                )
            )
        receipts.append(
            ConsumptionReceipt(
                event_identity=event.identity,
                world=world,
                state=state,
                occurrence_count=len(selected),
                value_commitments=commitments,
            )
        )
    for identity, selected in sorted(grouped.items()):
        if identity in by_expected:
            raise AssertionError("expected identity remained after audit")
        receipts.append(
            ConsumptionReceipt(
                event_identity=identity,
                world=world,
                state=ReceiptState.UNEXPECTED,
                occurrence_count=len(selected),
                value_commitments=tuple(
                    item.value_commitment
                    for item in selected
                    if item.value_commitment is not None
                ),
            )
        )
        violations.append(
            ConformanceViolation(
                code=ConformanceViolationCode.UNEXPECTED,
                world=world,
                event_identity=identity,
                detail="observation has no declaration in the frozen contract",
            )
        )
    return WorldConformance(
        world=world,
        receipts=tuple(receipts),
        violations=tuple(violations),
        passed=not violations,
    )


def audit_pair(
    contract: CouplingContract,
    observations: Sequence[ConsumptionObservation],
) -> PairConformance:
    baseline = audit_world(
        contract,
        World.BASELINE,
        tuple(item for item in observations if item.world is World.BASELINE),
    )
    policy = audit_world(
        contract,
        World.POLICY,
        tuple(item for item in observations if item.world is World.POLICY),
    )
    baseline_by_id = {item.event_identity: item for item in baseline.receipts}
    policy_by_id = {item.event_identity: item for item in policy.receipts}
    mismatches: list[str] = []
    pair_violations = [*baseline.violations, *policy.violations]
    for event in contract.events:
        if event.world_scope is not WorldScope.SHARED:
            continue
        left = baseline_by_id[event.identity]
        right = policy_by_id[event.identity]
        if (
            left.state is ReceiptState.CONSUMED
            and right.state is ReceiptState.CONSUMED
            and left.value_commitments != right.value_commitments
        ):
            mismatches.append(event.identity)
            pair_violations.append(
                ConformanceViolation(
                    code=ConformanceViolationCode.SHARED_VALUE_MISMATCH,
                    world=None,
                    event_identity=event.identity,
                    detail="baseline and policy value commitments differ",
                )
            )
    return PairConformance(
        contract_sha256=contract.sha256,
        baseline=baseline,
        policy=policy,
        shared_value_mismatches=tuple(mismatches),
        violations=tuple(pair_violations),
        passed=not pair_violations,
    )


def conform(
    contract: CouplingContract,
    baseline_observations: Sequence[ConsumptionObservation],
    policy_observations: Sequence[ConsumptionObservation],
) -> bool:
    """Evaluate the executable predicate ``Conform(C, O0, O1)``.

    The inputs are kept world-separated so callers cannot accidentally treat a
    mixed trace as one world's evidence.  :func:`audit_pair` retains the full
    receipt and violation witness used to explain a false predicate.
    """
    if any(item.world is not World.BASELINE for item in baseline_observations):
        _fail("O0 must contain baseline observations only")
    if any(item.world is not World.POLICY for item in policy_observations):
        _fail("O1 must contain policy observations only")
    return audit_pair(
        contract, (*baseline_observations, *policy_observations)
    ).passed


class EventKeyedBackend:
    """Stateless random-draw backend keyed by a complete contract address."""

    def __init__(self, seed: str) -> None:
        self._seed = _identifier(seed, "seed")

    def draw_u64(self, event: SemanticEventAddress) -> int:
        payload = b"policytwin-coupling-contract-rng-v1\0" + _canonical_bytes(
            {"event": event.to_mapping(), "seed": self._seed}
        )
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")

    def draw(self, event: SemanticEventAddress) -> int | float:
        bits = self.draw_u64(event)
        parameters = dict(event.distribution_parameters)
        if event.distribution_family == "uniform_0_1":
            return bits / float(1 << 64)
        if event.distribution_family == "discrete_uniform_inclusive":
            lower = parameters.get("lower")
            upper = parameters.get("upper")
            if (
                isinstance(lower, bool)
                or isinstance(upper, bool)
                or not isinstance(lower, int)
                or not isinstance(upper, int)
                or lower > upper
            ):
                _fail("invalid discrete-uniform contract parameters")
            return lower + bits % (upper - lower + 1)
        _fail(f"unsupported distribution family: {event.distribution_family}")


class RandomTapeOracle:
    """Frozen address-to-value table used without runtime event-key hashing."""

    def __init__(self, values: Mapping[str, int | float]) -> None:
        if not values:
            _fail("random tape must not be empty")
        self._values = MappingProxyType(dict(values))
        self._consumption = Counter[str]()

    @classmethod
    def preallocate(
        cls,
        contract: CouplingContract,
        value_factory: Callable[[SemanticEventAddress], int | float],
    ) -> Self:
        """Materialize the table before world execution begins."""
        return cls({event.identity: value_factory(event) for event in contract.events})

    @property
    def sha256(self) -> str:
        return _digest(dict(self._values))

    def draw(self, event: SemanticEventAddress) -> int | float:
        try:
            value = self._values[event.identity]
        except KeyError as error:
            raise CouplingContractError(
                "event is absent from frozen random tape"
            ) from error
        self._consumption[event.identity] += 1
        return value

    @property
    def consumption_counts(self) -> Mapping[str, int]:
        return MappingProxyType(dict(self._consumption))


__all__ = [
    "COUPLING_CONTRACT_SCHEMA_VERSION",
    "ConsumptionObservation",
    "ConsumptionReceipt",
    "ConformanceViolation",
    "ConformanceViolationCode",
    "CouplingContract",
    "CouplingContractError",
    "EventDeclaration",
    "EventKeyedBackend",
    "ObservationState",
    "PairConformance",
    "RandomTapeOracle",
    "ReceiptState",
    "Requirement",
    "ScenarioSpecification",
    "SemanticEventAddress",
    "World",
    "WorldConformance",
    "WorldScope",
    "audit_pair",
    "audit_world",
    "conform",
    "generate_contract",
]
