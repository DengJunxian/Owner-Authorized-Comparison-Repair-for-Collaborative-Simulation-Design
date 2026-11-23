"""Auditable aggregation of structural-model paired-effect distributions.

The ensemble is frozen before target outcomes are available.  Every registered
member is required: a missing or failed member produces an explicit abstention
artifact, and completed-member weights are never silently renormalized.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from itertools import combinations
from statistics import fmean
from typing import NoReturn

from policytwin.provenance import canonical_json_bytes, sha256_bytes

ENSEMBLE_SCHEMA_VERSION = "policytwin.structural-ensemble.v1"
ENSEMBLE_ARTIFACT_SCHEMA_VERSION = "policytwin.structural-ensemble-artifact.v3"
ENSEMBLE_CLAIM_SCOPE = "synthetic_summary_arithmetic_only"
ENSEMBLE_MEMBER_EFFECT_AUTHORITY = "unverified_caller_authored_member_draws"

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EVENT_KEY = re.compile(r"^(?:pse1|ptk1):[0-9a-f]{64}$")


class StructuralEnsembleError(RuntimeError):
    """Base class for fail-closed structural-ensemble errors."""


class StructuralEnsembleSchemaError(StructuralEnsembleError, ValueError):
    """Raised when a frozen specification or member result is malformed."""


class StructuralEnsembleIntegrityError(StructuralEnsembleError):
    """Raised when registered members are not scientifically comparable."""


def _fail(message: str) -> NoReturn:
    raise StructuralEnsembleSchemaError(message)


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        _fail(f"{field} must be a stable ASCII identifier")
    return value


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail(f"{field} must be a lowercase SHA-256 digest")
    return value


def _event_key(value: object) -> str:
    if not isinstance(value, str) or _EVENT_KEY.fullmatch(value) is None:
        _fail("event_key must be a semantic or experiment event-key commitment")
    return value


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        _fail(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{field} must be finite")
    return result


def _probability(value: object, field: str, *, open_interval: bool = False) -> float:
    result = _finite(value, field)
    lower_ok = result > 0.0 if open_interval else result >= 0.0
    upper_ok = result < 1.0 if open_interval else result <= 1.0
    if not lower_ok or not upper_ok:
        interval = "(0, 1)" if open_interval else "[0, 1]"
        _fail(f"{field} must lie in {interval}")
    return result


def _aware_utc(value: object, field: str) -> datetime:
    if not isinstance(value, datetime):
        _fail(f"{field} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        _fail(f"{field} must include an explicit timezone")
    try:
        return value.astimezone(UTC)
    except (OverflowError, ValueError) as error:
        raise StructuralEnsembleSchemaError(
            f"{field} is outside the UTC range"
        ) from error


def _require_mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        _fail(f"{field} must be an object with string keys")
    return value


def _require_array(value: object, field: str) -> list[object]:
    if not isinstance(value, list):
        _fail(f"{field} must be an array")
    return value


def _require_fields(
    mapping: Mapping[str, object], expected: frozenset[str], field: str
) -> None:
    observed = frozenset(mapping)
    if observed != expected:
        _fail(
            f"{field} fields differ: missing={sorted(expected - observed)!r}, "
            f"extra={sorted(observed - expected)!r}"
        )


def _canonical_mapping_equal(
    observed: Mapping[str, object], expected: Mapping[str, object], field: str
) -> bool:
    try:
        return canonical_json_bytes(dict(observed)) == canonical_json_bytes(
            dict(expected)
        )
    except (TypeError, ValueError) as error:
        raise StructuralEnsembleSchemaError(
            f"{field} is not canonical JSON: {error}"
        ) from error


def _string(value: object, field: str) -> str:
    if not isinstance(value, str):
        _fail(f"{field} must be a string")
    return value


def _datetime_from_mapping(value: object, field: str) -> datetime:
    text = _string(value, field)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as error:
        raise StructuralEnsembleSchemaError(f"{field} is not ISO-8601") from error
    return _aware_utc(parsed, field)


def _population_variance(values: tuple[float, ...]) -> float:
    mean = fmean(values)
    return math.fsum((value - mean) ** 2 for value in values) / len(values)


def _weighted_inverse_cdf(
    values: tuple[tuple[float, float], ...], probability: float
) -> float:
    """Return the left-continuous weighted empirical inverse CDF."""
    if not values:
        _fail("weighted quantile requires at least one value")
    probability = _probability(probability, "quantile probability")
    ordered = sorted(values, key=lambda item: item[0])
    total = math.fsum(weight for _, weight in ordered)
    if total <= 0.0:
        _fail("weighted quantile mass must be positive")
    target = probability * total
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= target:
            return value
    return ordered[-1][0]


class ModelEvidenceStatus(StrEnum):
    """Declared evidence boundary for one registered structural model."""

    SYNTHETIC_ONLY = "synthetic_only"
    PRE_POLICY_REAL_ARTIFACT = "pre_policy_real_artifact"


class EnsembleWeighting(StrEnum):
    """Preregistered ensemble weighting rule."""

    EQUAL = "equal_weight"
    FROZEN = "frozen_member_weights"


class MemberResultStatus(StrEnum):
    """Terminal status of one registered member result."""

    COMPLETED = "completed"
    FAILED = "failed"


class MemberAuditStatus(StrEnum):
    """Coverage state retained in an ensemble artifact."""

    COMPLETED = "completed"
    FAILED = "failed"
    MISSING = "missing"


class StructuralEnsembleStatus(StrEnum):
    """Whether a complete ensemble distribution can be reported."""

    COMPLETED = "completed"
    ABSTAIN = "abstain"


@dataclass(frozen=True, slots=True)
class StructuralModelMember:
    """One preregistered model structure and its exact frozen artifacts."""

    model_id: str
    family: str
    config_sha256: str
    checkpoint_sha256: str
    evidence_status: ModelEvidenceStatus
    evidence_artifact_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.model_id, "model_id")
        _identifier(self.family, "model family")
        _sha256(self.config_sha256, "model config_sha256")
        _sha256(self.checkpoint_sha256, "model checkpoint_sha256")
        if not isinstance(self.evidence_status, ModelEvidenceStatus):
            _fail("evidence_status must be a ModelEvidenceStatus")
        _sha256(self.evidence_artifact_sha256, "evidence_artifact_sha256")
        if self.evidence_status is ModelEvidenceStatus.PRE_POLICY_REAL_ARTIFACT:
            _fail(
                "pre-policy real structural evidence requires typed temporal and "
                "lineage verifier authority; caller hashes and timestamps are not "
                "authority"
            )

    @property
    def sha256(self) -> str:
        """Return the commitment to model identity, artifacts, and evidence label."""
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    @property
    def structural_signature(self) -> tuple[str, str, str, str, str]:
        """Return identity fields that establish a distinct model structure."""
        return (
            self.family,
            self.config_sha256,
            self.checkpoint_sha256,
            self.evidence_status.value,
            self.evidence_artifact_sha256,
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "checkpoint_sha256": self.checkpoint_sha256,
            "config_sha256": self.config_sha256,
            "evidence_artifact_sha256": self.evidence_artifact_sha256,
            "evidence_status": self.evidence_status.value,
            "family": self.family,
            "model_id": self.model_id,
        }


@dataclass(frozen=True, slots=True)
class EnsembleMemberWeight:
    """One strictly positive preregistered structural-model weight."""

    model_id: str
    weight: float

    def __post_init__(self) -> None:
        _identifier(self.model_id, "weight model_id")
        weight = _probability(self.weight, "member weight", open_interval=True)
        object.__setattr__(self, "weight", weight)

    def to_mapping(self) -> dict[str, object]:
        return {"model_id": self.model_id, "weight": self.weight}


@dataclass(frozen=True, slots=True)
class StructuralEnsembleSpec:
    """Frozen complete member set, weights, and uncertainty summaries."""

    study_protocol_sha256: str
    members: tuple[StructuralModelMember, ...]
    weighting: EnsembleWeighting
    weights_frozen_at: datetime
    target_outcomes_available_at: datetime
    quantile_levels: tuple[float, ...]
    interval_level: float
    weights: tuple[EnsembleMemberWeight, ...] = ()
    schema_version: str = ENSEMBLE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _sha256(self.study_protocol_sha256, "study_protocol_sha256")
        if (
            not isinstance(self.members, tuple)
            or len(self.members) < 2
            or not all(isinstance(item, StructuralModelMember) for item in self.members)
        ):
            _fail("structural ensemble requires at least two immutable members")
        members = tuple(sorted(self.members, key=lambda item: item.model_id))
        if len({item.model_id for item in members}) != len(members):
            _fail("structural model IDs must be unique")
        if len({item.structural_signature for item in members}) != len(members):
            _fail(
                "structural model signatures must be unique; changing only "
                "model_id does not add structural uncertainty"
            )
        if len({item.family for item in members}) < 2:
            _fail(
                "structural uncertainty requires at least two distinct model families"
            )
        object.__setattr__(self, "members", members)
        if not isinstance(self.weighting, EnsembleWeighting):
            _fail("weighting must be an EnsembleWeighting")
        frozen_at = _aware_utc(self.weights_frozen_at, "weights_frozen_at")
        available_at = _aware_utc(
            self.target_outcomes_available_at, "target_outcomes_available_at"
        )
        if frozen_at >= available_at:
            _fail(
                "ensemble weights must be frozen before target outcomes are available"
            )
        object.__setattr__(self, "weights_frozen_at", frozen_at)
        object.__setattr__(self, "target_outcomes_available_at", available_at)
        if not isinstance(self.quantile_levels, tuple) or not self.quantile_levels:
            _fail("quantile_levels must be a non-empty immutable tuple")
        levels = tuple(
            sorted(
                _probability(value, "quantile level", open_interval=True)
                for value in self.quantile_levels
            )
        )
        if len(set(levels)) != len(levels):
            _fail("quantile levels must be unique")
        object.__setattr__(self, "quantile_levels", levels)
        object.__setattr__(
            self,
            "interval_level",
            _probability(self.interval_level, "interval_level", open_interval=True),
        )
        if not isinstance(self.weights, tuple) or not all(
            isinstance(item, EnsembleMemberWeight) for item in self.weights
        ):
            _fail("weights must be an immutable EnsembleMemberWeight tuple")
        weights = tuple(sorted(self.weights, key=lambda item: item.model_id))
        member_ids = tuple(item.model_id for item in members)
        if self.weighting is EnsembleWeighting.EQUAL:
            if weights:
                _fail("equal weighting must not include caller-supplied weights")
        else:
            if tuple(item.model_id for item in weights) != member_ids:
                _fail("frozen weights must cover every registered member exactly once")
            if not math.isclose(
                math.fsum(item.weight for item in weights),
                1.0,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                _fail("frozen member weights must sum to one")
        object.__setattr__(self, "weights", weights)
        if self.schema_version != ENSEMBLE_SCHEMA_VERSION:
            _fail("unsupported structural ensemble schema version")

    @property
    def member_weights(self) -> tuple[EnsembleMemberWeight, ...]:
        """Return all effective weights without result-dependent normalization."""
        if self.weighting is EnsembleWeighting.FROZEN:
            return self.weights
        equal_weight = 1.0 / len(self.members)
        return tuple(
            EnsembleMemberWeight(member.model_id, equal_weight)
            for member in self.members
        )

    @property
    def synthetic_only(self) -> bool:
        """Return true unless every member declares a real pre-policy artifact."""
        return any(
            item.evidence_status is ModelEvidenceStatus.SYNTHETIC_ONLY
            for item in self.members
        )

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def to_mapping(self) -> dict[str, object]:
        return {
            "interval_level": self.interval_level,
            "member_weights": [item.to_mapping() for item in self.member_weights],
            "members": [item.to_mapping() for item in self.members],
            "quantile_levels": list(self.quantile_levels),
            "schema_version": self.schema_version,
            "study_protocol_sha256": self.study_protocol_sha256,
            "synthetic_only": self.synthetic_only,
            "target_outcomes_available_at": (
                self.target_outcomes_available_at.isoformat()
            ),
            "weighting": self.weighting.value,
            "weights_frozen_at": self.weights_frozen_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class StructuralOutcomeCell:
    """Exact protocol, policy pair, estimand, cohort, and horizon for all members."""

    study_protocol_sha256: str
    baseline_policy_sha256: str
    candidate_policy_sha256: str
    estimand_id: str
    outcome_id: str
    cohort_id: str
    cohort_sha256: str
    horizon_id: str

    def __post_init__(self) -> None:
        _sha256(self.study_protocol_sha256, "cell study_protocol_sha256")
        _sha256(self.baseline_policy_sha256, "baseline_policy_sha256")
        _sha256(self.candidate_policy_sha256, "candidate_policy_sha256")
        _identifier(self.estimand_id, "estimand_id")
        _identifier(self.outcome_id, "outcome_id")
        _identifier(self.cohort_id, "cohort_id")
        _sha256(self.cohort_sha256, "cohort_sha256")
        _identifier(self.horizon_id, "horizon_id")

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def to_mapping(self) -> dict[str, object]:
        return {
            "baseline_policy_sha256": self.baseline_policy_sha256,
            "candidate_policy_sha256": self.candidate_policy_sha256,
            "cohort_id": self.cohort_id,
            "cohort_sha256": self.cohort_sha256,
            "estimand_id": self.estimand_id,
            "horizon_id": self.horizon_id,
            "outcome_id": self.outcome_id,
            "study_protocol_sha256": self.study_protocol_sha256,
        }


@dataclass(frozen=True, slots=True)
class StructuralPairedDraw:
    """One event-keyed candidate-minus-baseline outcome from one model."""

    event_key: str
    draw_id: str
    baseline_value: float
    candidate_value: float
    difference: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_key", _event_key(self.event_key))
        _identifier(self.draw_id, "draw_id")
        baseline = _finite(self.baseline_value, "baseline_value")
        candidate = _finite(self.candidate_value, "candidate_value")
        difference = _finite(self.difference, "difference")
        object.__setattr__(self, "baseline_value", baseline)
        object.__setattr__(self, "candidate_value", candidate)
        object.__setattr__(self, "difference", difference)
        if difference != candidate - baseline:
            _fail("paired difference must equal candidate minus baseline")

    def to_mapping(self) -> dict[str, object]:
        return {
            "baseline_value": self.baseline_value,
            "candidate_value": self.candidate_value,
            "difference": self.difference,
            "draw_id": self.draw_id,
            "event_key": self.event_key,
        }


@dataclass(frozen=True, slots=True)
class StructuralMemberResult:
    """Terminal paired-draw artifact for one registered structural member."""

    model_id: str
    member_sha256: str
    cell: StructuralOutcomeCell
    status: MemberResultStatus
    draws: tuple[StructuralPairedDraw, ...] = ()
    failure_code: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.model_id, "result model_id")
        _sha256(self.member_sha256, "result member_sha256")
        if not isinstance(self.cell, StructuralOutcomeCell):
            _fail("member result requires a StructuralOutcomeCell")
        if not isinstance(self.status, MemberResultStatus):
            _fail("member result status must be a MemberResultStatus")
        if not isinstance(self.draws, tuple) or not all(
            isinstance(item, StructuralPairedDraw) for item in self.draws
        ):
            _fail("member draws must be an immutable StructuralPairedDraw tuple")
        draws = tuple(
            sorted(self.draws, key=lambda item: (item.event_key, item.draw_id))
        )
        if len({item.event_key for item in draws}) != len(draws):
            _fail("member draw event keys must be unique")
        if len({item.draw_id for item in draws}) != len(draws):
            _fail("member draw IDs must be unique")
        object.__setattr__(self, "draws", draws)
        if self.status is MemberResultStatus.COMPLETED:
            if len(draws) < 2 or self.failure_code is not None:
                _fail("completed member requires at least two draws and no failure")
        else:
            if draws or self.failure_code is None:
                _fail("failed member requires a failure_code and no effect draws")
            _identifier(self.failure_code, "member failure_code")

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def to_mapping(self) -> dict[str, object]:
        return {
            "cell": self.cell.to_mapping(),
            "draws": [item.to_mapping() for item in self.draws],
            "failure_code": self.failure_code,
            "member_sha256": self.member_sha256,
            "model_id": self.model_id,
            "status": self.status.value,
        }


def _model_member_from_mapping(
    mapping: Mapping[str, object],
) -> StructuralModelMember:
    _require_fields(
        mapping,
        frozenset(
            {
                "checkpoint_sha256",
                "config_sha256",
                "evidence_artifact_sha256",
                "evidence_status",
                "family",
                "model_id",
            }
        ),
        "structural model member",
    )
    try:
        evidence_status = ModelEvidenceStatus(
            _string(mapping["evidence_status"], "member evidence_status")
        )
    except ValueError as error:
        raise StructuralEnsembleSchemaError(
            "member evidence_status is invalid"
        ) from error
    return StructuralModelMember(
        model_id=_string(mapping["model_id"], "member model_id"),
        family=_string(mapping["family"], "member family"),
        config_sha256=_string(mapping["config_sha256"], "member config_sha256"),
        checkpoint_sha256=_string(
            mapping["checkpoint_sha256"], "member checkpoint_sha256"
        ),
        evidence_status=evidence_status,
        evidence_artifact_sha256=_string(
            mapping["evidence_artifact_sha256"],
            "member evidence_artifact_sha256",
        ),
    )


def _member_weight_from_mapping(
    mapping: Mapping[str, object],
) -> EnsembleMemberWeight:
    _require_fields(mapping, frozenset({"model_id", "weight"}), "member weight")
    return EnsembleMemberWeight(
        model_id=_string(mapping["model_id"], "weight model_id"),
        weight=_finite(mapping["weight"], "member weight"),
    )


def _spec_from_mapping(mapping: Mapping[str, object]) -> StructuralEnsembleSpec:
    _require_fields(
        mapping,
        frozenset(
            {
                "interval_level",
                "member_weights",
                "members",
                "quantile_levels",
                "schema_version",
                "study_protocol_sha256",
                "synthetic_only",
                "target_outcomes_available_at",
                "weighting",
                "weights_frozen_at",
            }
        ),
        "structural ensemble spec",
    )
    try:
        weighting = EnsembleWeighting(
            _string(mapping["weighting"], "ensemble weighting")
        )
    except ValueError as error:
        raise StructuralEnsembleSchemaError("ensemble weighting is invalid") from error
    members = tuple(
        _model_member_from_mapping(_require_mapping(item, f"members[{index}]"))
        for index, item in enumerate(_require_array(mapping["members"], "members"))
    )
    effective_weights = tuple(
        _member_weight_from_mapping(_require_mapping(item, f"member_weights[{index}]"))
        for index, item in enumerate(
            _require_array(mapping["member_weights"], "member_weights")
        )
    )
    quantile_levels = tuple(
        _finite(item, f"quantile_levels[{index}]")
        for index, item in enumerate(
            _require_array(mapping["quantile_levels"], "quantile_levels")
        )
    )
    spec = StructuralEnsembleSpec(
        study_protocol_sha256=_string(
            mapping["study_protocol_sha256"], "study_protocol_sha256"
        ),
        members=members,
        weighting=weighting,
        weights_frozen_at=_datetime_from_mapping(
            mapping["weights_frozen_at"], "weights_frozen_at"
        ),
        target_outcomes_available_at=_datetime_from_mapping(
            mapping["target_outcomes_available_at"],
            "target_outcomes_available_at",
        ),
        quantile_levels=quantile_levels,
        interval_level=_finite(mapping["interval_level"], "interval_level"),
        weights=(effective_weights if weighting is EnsembleWeighting.FROZEN else ()),
        schema_version=_string(mapping["schema_version"], "spec schema_version"),
    )
    if not _canonical_mapping_equal(mapping, spec.to_mapping(), "ensemble spec"):
        raise StructuralEnsembleIntegrityError(
            "serialized structural ensemble spec is not canonical"
        )
    return spec


def _cell_from_mapping(mapping: Mapping[str, object]) -> StructuralOutcomeCell:
    fields = frozenset(
        {
            "baseline_policy_sha256",
            "candidate_policy_sha256",
            "cohort_id",
            "cohort_sha256",
            "estimand_id",
            "horizon_id",
            "outcome_id",
            "study_protocol_sha256",
        }
    )
    _require_fields(mapping, fields, "structural outcome cell")
    return StructuralOutcomeCell(
        study_protocol_sha256=_string(
            mapping["study_protocol_sha256"], "cell study_protocol_sha256"
        ),
        baseline_policy_sha256=_string(
            mapping["baseline_policy_sha256"], "baseline_policy_sha256"
        ),
        candidate_policy_sha256=_string(
            mapping["candidate_policy_sha256"], "candidate_policy_sha256"
        ),
        estimand_id=_string(mapping["estimand_id"], "estimand_id"),
        outcome_id=_string(mapping["outcome_id"], "outcome_id"),
        cohort_id=_string(mapping["cohort_id"], "cohort_id"),
        cohort_sha256=_string(mapping["cohort_sha256"], "cohort_sha256"),
        horizon_id=_string(mapping["horizon_id"], "horizon_id"),
    )


def _paired_draw_from_mapping(mapping: Mapping[str, object]) -> StructuralPairedDraw:
    _require_fields(
        mapping,
        frozenset(
            {
                "baseline_value",
                "candidate_value",
                "difference",
                "draw_id",
                "event_key",
            }
        ),
        "structural paired draw",
    )
    return StructuralPairedDraw(
        event_key=_string(mapping["event_key"], "draw event_key"),
        draw_id=_string(mapping["draw_id"], "draw_id"),
        baseline_value=_finite(mapping["baseline_value"], "baseline_value"),
        candidate_value=_finite(mapping["candidate_value"], "candidate_value"),
        difference=_finite(mapping["difference"], "difference"),
    )


def _member_result_from_mapping(
    mapping: Mapping[str, object],
) -> StructuralMemberResult:
    _require_fields(
        mapping,
        frozenset(
            {"cell", "draws", "failure_code", "member_sha256", "model_id", "status"}
        ),
        "structural member result",
    )
    try:
        status = MemberResultStatus(_string(mapping["status"], "member result status"))
    except ValueError as error:
        raise StructuralEnsembleSchemaError(
            "member result status is invalid"
        ) from error
    failure = mapping["failure_code"]
    if failure is not None and not isinstance(failure, str):
        _fail("member result failure_code must be a string or null")
    return StructuralMemberResult(
        model_id=_string(mapping["model_id"], "result model_id"),
        member_sha256=_string(mapping["member_sha256"], "result member_sha256"),
        cell=_cell_from_mapping(
            _require_mapping(mapping["cell"], "member result cell")
        ),
        status=status,
        draws=tuple(
            _paired_draw_from_mapping(
                _require_mapping(item, f"member result draws[{index}]")
            )
            for index, item in enumerate(
                _require_array(mapping["draws"], "member result draws")
            )
        ),
        failure_code=failure,
    )


@dataclass(frozen=True, slots=True)
class QuantileEstimate:
    """One explicitly levelled empirical quantile."""

    level: float
    value: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "level",
            _probability(self.level, "quantile level", open_interval=True),
        )
        object.__setattr__(self, "value", _finite(self.value, "quantile value"))

    def to_mapping(self) -> dict[str, object]:
        return {"level": self.level, "value": self.value}


@dataclass(frozen=True, slots=True)
class MemberDistribution:
    """Complete event-keyed distribution summary for one model member."""

    model_id: str
    weight: float
    draws: tuple[StructuralPairedDraw, ...]
    mean: float
    variance: float
    mean_sign: int
    quantiles: tuple[QuantileEstimate, ...]

    def to_mapping(self) -> dict[str, object]:
        return {
            "draws": [item.to_mapping() for item in self.draws],
            "mean": self.mean,
            "mean_sign": self.mean_sign,
            "model_id": self.model_id,
            "quantiles": [item.to_mapping() for item in self.quantiles],
            "variance": self.variance,
            "weight": self.weight,
        }


@dataclass(frozen=True, slots=True)
class WeightedPooledDraw:
    """One member draw and its unrenormalized mixture probability mass."""

    model_id: str
    event_key: str
    draw_id: str
    difference: float
    mixture_mass: float

    def to_mapping(self) -> dict[str, object]:
        return {
            "difference": self.difference,
            "draw_id": self.draw_id,
            "event_key": self.event_key,
            "mixture_mass": self.mixture_mass,
            "model_id": self.model_id,
        }


@dataclass(frozen=True, slots=True)
class VarianceDecomposition:
    """Law-of-total-variance decomposition across structural members."""

    pooled_mean: float
    within_model_variance: float
    between_model_variance: float
    total_variance: float

    def to_mapping(self) -> dict[str, object]:
        return {
            "between_model_variance": self.between_model_variance,
            "pooled_mean": self.pooled_mean,
            "total_variance": self.total_variance,
            "within_model_variance": self.within_model_variance,
        }


@dataclass(frozen=True, slots=True)
class WorstCaseQuantile:
    """Member envelope for one frozen quantile level."""

    level: float
    minimum: float
    minimizing_model_id: str
    maximum: float
    maximizing_model_id: str

    def to_mapping(self) -> dict[str, object]:
        return {
            "level": self.level,
            "maximum": self.maximum,
            "maximizing_model_id": self.maximizing_model_id,
            "minimum": self.minimum,
            "minimizing_model_id": self.minimizing_model_id,
        }


@dataclass(frozen=True, slots=True)
class WorstCaseInterval:
    """Envelope of member central intervals at one frozen level."""

    level: float
    lower: float
    lower_model_id: str
    upper: float
    upper_model_id: str

    def to_mapping(self) -> dict[str, object]:
        return {
            "level": self.level,
            "lower": self.lower,
            "lower_model_id": self.lower_model_id,
            "upper": self.upper,
            "upper_model_id": self.upper_model_id,
        }


@dataclass(frozen=True, slots=True)
class MemberAuditRecord:
    """Expected weight and terminal evidence retained for every member."""

    model_id: str
    status: MemberAuditStatus
    expected_weight: float
    result_sha256: str | None
    failure_code: str | None
    draw_count: int

    def to_mapping(self) -> dict[str, object]:
        return {
            "draw_count": self.draw_count,
            "expected_weight": self.expected_weight,
            "failure_code": self.failure_code,
            "model_id": self.model_id,
            "result_sha256": self.result_sha256,
            "status": self.status.value,
        }


@dataclass(frozen=True, slots=True)
class StructuralEnsembleArtifact:
    """Self-contained arithmetic summary or explicit abstention.

    ``StructuralMemberResult`` is a public value schema, not an execution
    receipt.  Until receipts are retained and live-replayed in this artifact,
    its disagreement statistics are useful for synthetic arithmetic tests only
    and cannot serve as formal model-disagreement evidence.

    The complete frozen specification and every supplied terminal member result
    are retained.  Construction therefore re-runs the same pure derivation used
    by :func:`evaluate_structural_ensemble`; summary fields are commitments to
    those inputs, not caller-authoritative values.
    """

    status: StructuralEnsembleStatus
    spec: StructuralEnsembleSpec
    spec_sha256: str
    cell: StructuralOutcomeCell
    member_results: tuple[StructuralMemberResult, ...]
    member_audit: tuple[MemberAuditRecord, ...]
    member_coverage: float
    weight_coverage: float
    member_distributions: tuple[MemberDistribution, ...]
    pooled_distribution: tuple[WeightedPooledDraw, ...]
    pooled_quantiles: tuple[QuantileEstimate, ...]
    variance_decomposition: VarianceDecomposition | None
    sign_disagreement: bool | None
    sign_disagreement_fraction: float | None
    worst_case_quantiles: tuple[WorstCaseQuantile, ...]
    worst_case_interval: WorstCaseInterval | None
    abstain_reasons: tuple[str, ...]
    synthetic_only: bool
    schema_version: str = ENSEMBLE_ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.status, StructuralEnsembleStatus):
            _fail("artifact status must be a StructuralEnsembleStatus")
        if not isinstance(self.spec, StructuralEnsembleSpec):
            _fail("ensemble artifact requires the retained structural spec")
        if self.spec_sha256 != self.spec.sha256:
            raise StructuralEnsembleIntegrityError(
                "ensemble spec hash does not match the retained structural spec"
            )
        if not isinstance(self.cell, StructuralOutcomeCell):
            _fail("ensemble artifact requires a StructuralOutcomeCell")
        if not isinstance(self.member_results, tuple) or not all(
            isinstance(item, StructuralMemberResult) for item in self.member_results
        ):
            _fail("ensemble artifact requires immutable retained member results")
        ordered_results = tuple(
            sorted(self.member_results, key=lambda item: item.model_id)
        )
        if self.member_results != ordered_results:
            raise StructuralEnsembleIntegrityError(
                "retained member results must use canonical model-ID order"
            )
        if self.synthetic_only is not True:
            _fail(
                "non-synthetic structural artifacts require typed temporal and "
                "lineage verifier authority"
            )
        if self.schema_version != ENSEMBLE_ARTIFACT_SCHEMA_VERSION:
            _fail("unsupported structural ensemble artifact schema version")
        derived = _derive_structural_ensemble(self.spec, self.cell, self.member_results)
        try:
            observed_summary = canonical_json_bytes(self._summary_mapping())
        except (AttributeError, TypeError, ValueError) as error:
            raise StructuralEnsembleIntegrityError(
                f"ensemble summary schema is invalid: {error}"
            ) from error
        if observed_summary != canonical_json_bytes(derived.to_mapping()):
            raise StructuralEnsembleIntegrityError(
                "ensemble summaries do not derive from retained spec and member results"
            )

    def _summary_mapping(self) -> dict[str, object]:
        return {
            "abstain_reasons": list(self.abstain_reasons),
            "member_audit": [item.to_mapping() for item in self.member_audit],
            "member_coverage": self.member_coverage,
            "member_distributions": [
                item.to_mapping() for item in self.member_distributions
            ],
            "pooled_distribution": [
                item.to_mapping() for item in self.pooled_distribution
            ],
            "pooled_quantiles": [item.to_mapping() for item in self.pooled_quantiles],
            "sign_disagreement": self.sign_disagreement,
            "sign_disagreement_fraction": self.sign_disagreement_fraction,
            "status": self.status.value,
            "synthetic_only": self.synthetic_only,
            "variance_decomposition": (
                None
                if self.variance_decomposition is None
                else self.variance_decomposition.to_mapping()
            ),
            "weight_coverage": self.weight_coverage,
            "worst_case_interval": (
                None
                if self.worst_case_interval is None
                else self.worst_case_interval.to_mapping()
            ),
            "worst_case_quantiles": [
                item.to_mapping() for item in self.worst_case_quantiles
            ],
        }

    @property
    def member_effect_authority(self) -> str:
        """Describe the missing authority behind retained member effects."""

        return ENSEMBLE_MEMBER_EFFECT_AUTHORITY

    @property
    def formal_model_disagreement_evidence(self) -> bool:
        """Return false until typed member execution receipts are retained."""

        return False

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def to_mapping(self) -> dict[str, object]:
        return {
            "abstain_reasons": list(self.abstain_reasons),
            "cell": self.cell.to_mapping(),
            "claim_scope": ENSEMBLE_CLAIM_SCOPE,
            "member_audit": [item.to_mapping() for item in self.member_audit],
            "member_coverage": self.member_coverage,
            "member_distributions": [
                item.to_mapping() for item in self.member_distributions
            ],
            "member_effect_authority": self.member_effect_authority,
            "member_results": [item.to_mapping() for item in self.member_results],
            "pooled_distribution": [
                item.to_mapping() for item in self.pooled_distribution
            ],
            "pooled_quantiles": [item.to_mapping() for item in self.pooled_quantiles],
            "schema_version": self.schema_version,
            "sign_disagreement": self.sign_disagreement,
            "sign_disagreement_fraction": self.sign_disagreement_fraction,
            "spec": self.spec.to_mapping(),
            "spec_sha256": self.spec_sha256,
            "status": self.status.value,
            "formal_model_disagreement_evidence": (
                self.formal_model_disagreement_evidence
            ),
            "synthetic_only": self.synthetic_only,
            "variance_decomposition": (
                None
                if self.variance_decomposition is None
                else self.variance_decomposition.to_mapping()
            ),
            "weight_coverage": self.weight_coverage,
            "worst_case_interval": (
                None
                if self.worst_case_interval is None
                else self.worst_case_interval.to_mapping()
            ),
            "worst_case_quantiles": [
                item.to_mapping() for item in self.worst_case_quantiles
            ],
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> StructuralEnsembleArtifact:
        """Reconstruct and fully rederive an untrusted serialized artifact."""
        mapping = _require_mapping(mapping, "structural ensemble artifact")
        _require_fields(
            mapping,
            frozenset(
                {
                    "abstain_reasons",
                    "cell",
                    "claim_scope",
                    "member_audit",
                    "member_coverage",
                    "member_distributions",
                    "member_effect_authority",
                    "member_results",
                    "pooled_distribution",
                    "pooled_quantiles",
                    "schema_version",
                    "sign_disagreement",
                    "sign_disagreement_fraction",
                    "spec",
                    "spec_sha256",
                    "status",
                    "formal_model_disagreement_evidence",
                    "synthetic_only",
                    "variance_decomposition",
                    "weight_coverage",
                    "worst_case_interval",
                    "worst_case_quantiles",
                }
            ),
            "structural ensemble artifact",
        )
        if mapping["claim_scope"] != ENSEMBLE_CLAIM_SCOPE:
            _fail("structural ensemble claim scope cannot be widened")
        if mapping["member_effect_authority"] != ENSEMBLE_MEMBER_EFFECT_AUTHORITY:
            _fail("structural member-effect authority cannot be widened")
        if mapping["formal_model_disagreement_evidence"] is not False:
            _fail(
                "caller-authored structural member draws cannot be formal "
                "model-disagreement evidence"
            )
        if mapping["schema_version"] != ENSEMBLE_ARTIFACT_SCHEMA_VERSION:
            _fail("unsupported structural ensemble artifact schema version")
        spec = _spec_from_mapping(
            _require_mapping(mapping["spec"], "structural ensemble spec")
        )
        cell = _cell_from_mapping(
            _require_mapping(mapping["cell"], "structural ensemble cell")
        )
        member_results = tuple(
            _member_result_from_mapping(
                _require_mapping(item, f"member_results[{index}]")
            )
            for index, item in enumerate(
                _require_array(mapping["member_results"], "member_results")
            )
        )
        expected = evaluate_structural_ensemble(spec, cell, member_results)
        if not _canonical_mapping_equal(
            mapping,
            expected.to_mapping(),
            "structural ensemble artifact",
        ):
            raise StructuralEnsembleIntegrityError(
                "serialized structural ensemble artifact does not rederive exactly"
            )
        return expected


def _member_distribution(
    result: StructuralMemberResult,
    weight: float,
    quantile_levels: tuple[float, ...],
) -> MemberDistribution:
    effects = tuple(item.difference for item in result.draws)
    mean = fmean(effects)
    sign = 1 if mean > 0.0 else (-1 if mean < 0.0 else 0)
    equally_weighted = tuple((value, 1.0 / len(effects)) for value in effects)
    return MemberDistribution(
        model_id=result.model_id,
        weight=weight,
        draws=result.draws,
        mean=mean,
        variance=_population_variance(effects),
        mean_sign=sign,
        quantiles=tuple(
            QuantileEstimate(level, _weighted_inverse_cdf(equally_weighted, level))
            for level in quantile_levels
        ),
    )


@dataclass(frozen=True, slots=True)
class _DerivedEnsemble:
    status: StructuralEnsembleStatus
    member_audit: tuple[MemberAuditRecord, ...]
    member_coverage: float
    weight_coverage: float
    member_distributions: tuple[MemberDistribution, ...]
    pooled_distribution: tuple[WeightedPooledDraw, ...]
    pooled_quantiles: tuple[QuantileEstimate, ...]
    variance_decomposition: VarianceDecomposition | None
    sign_disagreement: bool | None
    sign_disagreement_fraction: float | None
    worst_case_quantiles: tuple[WorstCaseQuantile, ...]
    worst_case_interval: WorstCaseInterval | None
    abstain_reasons: tuple[str, ...]
    synthetic_only: bool

    def to_mapping(self) -> dict[str, object]:
        return {
            "abstain_reasons": list(self.abstain_reasons),
            "member_audit": [item.to_mapping() for item in self.member_audit],
            "member_coverage": self.member_coverage,
            "member_distributions": [
                item.to_mapping() for item in self.member_distributions
            ],
            "pooled_distribution": [
                item.to_mapping() for item in self.pooled_distribution
            ],
            "pooled_quantiles": [item.to_mapping() for item in self.pooled_quantiles],
            "sign_disagreement": self.sign_disagreement,
            "sign_disagreement_fraction": self.sign_disagreement_fraction,
            "status": self.status.value,
            "synthetic_only": self.synthetic_only,
            "variance_decomposition": (
                None
                if self.variance_decomposition is None
                else self.variance_decomposition.to_mapping()
            ),
            "weight_coverage": self.weight_coverage,
            "worst_case_interval": (
                None
                if self.worst_case_interval is None
                else self.worst_case_interval.to_mapping()
            ),
            "worst_case_quantiles": [
                item.to_mapping() for item in self.worst_case_quantiles
            ],
        }


def _abstention_derivation(
    spec: StructuralEnsembleSpec,
    audit: tuple[MemberAuditRecord, ...],
    reasons: tuple[str, ...],
) -> _DerivedEnsemble:
    completed = tuple(
        item for item in audit if item.status is MemberAuditStatus.COMPLETED
    )
    return _DerivedEnsemble(
        status=StructuralEnsembleStatus.ABSTAIN,
        member_audit=audit,
        member_coverage=len(completed) / len(spec.members),
        weight_coverage=math.fsum(item.expected_weight for item in completed),
        member_distributions=(),
        pooled_distribution=(),
        pooled_quantiles=(),
        variance_decomposition=None,
        sign_disagreement=None,
        sign_disagreement_fraction=None,
        worst_case_quantiles=(),
        worst_case_interval=None,
        abstain_reasons=tuple(sorted(reasons)),
        synthetic_only=spec.synthetic_only,
    )


def _derive_structural_ensemble(
    spec: StructuralEnsembleSpec,
    cell: StructuralOutcomeCell,
    results: tuple[StructuralMemberResult, ...],
) -> _DerivedEnsemble:
    """Purely derive all summaries from the frozen spec, cell, and results.

    Input ordering is irrelevant.  Result-dependent member selection is not:
    every member in ``spec`` must appear exactly once and complete successfully.
    """
    if not isinstance(spec, StructuralEnsembleSpec):
        _fail("spec must be a StructuralEnsembleSpec")
    if not isinstance(cell, StructuralOutcomeCell):
        _fail("cell must be a StructuralOutcomeCell")
    if cell.study_protocol_sha256 != spec.study_protocol_sha256:
        raise StructuralEnsembleIntegrityError(
            "outcome cell is bound to a different study protocol"
        )
    if not isinstance(results, tuple) or not all(
        isinstance(item, StructuralMemberResult) for item in results
    ):
        _fail("results must be an immutable StructuralMemberResult tuple")
    result_ids = tuple(item.model_id for item in results)
    if len(set(result_ids)) != len(result_ids):
        raise StructuralEnsembleIntegrityError("duplicate structural member result")
    expected_members = {item.model_id: item for item in spec.members}
    extra = sorted(set(result_ids) - set(expected_members))
    if extra:
        raise StructuralEnsembleIntegrityError(
            f"unregistered structural member results: {extra!r}"
        )
    results_by_id = {item.model_id: item for item in results}
    weights = {item.model_id: item.weight for item in spec.member_weights}
    audit: list[MemberAuditRecord] = []
    abstain_reasons: list[str] = []
    completed_results: list[StructuralMemberResult] = []
    for model_id in sorted(expected_members):
        member = expected_members[model_id]
        result = results_by_id.get(model_id)
        if result is None:
            audit.append(
                MemberAuditRecord(
                    model_id=model_id,
                    status=MemberAuditStatus.MISSING,
                    expected_weight=weights[model_id],
                    result_sha256=None,
                    failure_code="member_result_missing",
                    draw_count=0,
                )
            )
            abstain_reasons.append(f"missing_member:{model_id}")
            continue
        if result.member_sha256 != member.sha256:
            raise StructuralEnsembleIntegrityError(
                f"member artifact commitment changed for {model_id!r}"
            )
        if result.cell != cell:
            raise StructuralEnsembleIntegrityError(
                f"estimand/outcome/cohort/horizon changed for {model_id!r}"
            )
        if result.status is MemberResultStatus.FAILED:
            audit.append(
                MemberAuditRecord(
                    model_id=model_id,
                    status=MemberAuditStatus.FAILED,
                    expected_weight=weights[model_id],
                    result_sha256=result.sha256,
                    failure_code=result.failure_code,
                    draw_count=0,
                )
            )
            abstain_reasons.append(f"failed_member:{model_id}:{result.failure_code}")
            continue
        completed_results.append(result)
        audit.append(
            MemberAuditRecord(
                model_id=model_id,
                status=MemberAuditStatus.COMPLETED,
                expected_weight=weights[model_id],
                result_sha256=result.sha256,
                failure_code=None,
                draw_count=len(result.draws),
            )
        )
    audit_tuple = tuple(audit)
    if abstain_reasons:
        return _abstention_derivation(spec, audit_tuple, tuple(abstain_reasons))

    ordered_results = tuple(sorted(completed_results, key=lambda item: item.model_id))
    reference_identity = tuple(
        (item.event_key, item.draw_id) for item in ordered_results[0].draws
    )
    for result in ordered_results[1:]:
        identity = tuple((item.event_key, item.draw_id) for item in result.draws)
        if len(identity) != len(reference_identity):
            raise StructuralEnsembleIntegrityError(
                "structural members have different paired draw counts"
            )
        if identity != reference_identity:
            raise StructuralEnsembleIntegrityError(
                "structural members do not share event keys and draw identities"
            )

    distributions = tuple(
        _member_distribution(result, weights[result.model_id], spec.quantile_levels)
        for result in ordered_results
    )
    pooled: list[WeightedPooledDraw] = []
    for distribution in distributions:
        mass = distribution.weight / len(distribution.draws)
        pooled.extend(
            WeightedPooledDraw(
                model_id=distribution.model_id,
                event_key=draw.event_key,
                draw_id=draw.draw_id,
                difference=draw.difference,
                mixture_mass=mass,
            )
            for draw in distribution.draws
        )
    pooled_distribution = tuple(
        sorted(pooled, key=lambda item: (item.model_id, item.event_key, item.draw_id))
    )
    weighted_values = tuple(
        (item.difference, item.mixture_mass) for item in pooled_distribution
    )
    pooled_mean = math.fsum(item.weight * item.mean for item in distributions)
    within = math.fsum(item.weight * item.variance for item in distributions)
    between = math.fsum(
        item.weight * (item.mean - pooled_mean) ** 2 for item in distributions
    )
    total = within + between
    variance = VarianceDecomposition(
        pooled_mean=pooled_mean,
        within_model_variance=within,
        between_model_variance=between,
        total_variance=total,
    )
    pooled_quantiles = tuple(
        QuantileEstimate(level, _weighted_inverse_cdf(weighted_values, level))
        for level in spec.quantile_levels
    )
    worst_quantiles: list[WorstCaseQuantile] = []
    for level in spec.quantile_levels:
        values = tuple(
            (
                next(
                    item.value for item in distribution.quantiles if item.level == level
                ),
                distribution.model_id,
            )
            for distribution in distributions
        )
        minimum, minimizing_model = min(values)
        maximum, maximizing_model = max(values)
        worst_quantiles.append(
            WorstCaseQuantile(
                level=level,
                minimum=minimum,
                minimizing_model_id=minimizing_model,
                maximum=maximum,
                maximizing_model_id=maximizing_model,
            )
        )
    tail = (1.0 - spec.interval_level) / 2.0
    member_intervals = tuple(
        (
            _weighted_inverse_cdf(
                tuple(
                    (draw.difference, 1.0 / len(distribution.draws))
                    for draw in distribution.draws
                ),
                tail,
            ),
            _weighted_inverse_cdf(
                tuple(
                    (draw.difference, 1.0 / len(distribution.draws))
                    for draw in distribution.draws
                ),
                1.0 - tail,
            ),
            distribution.model_id,
        )
        for distribution in distributions
    )
    lower, lower_model = min((item[0], item[2]) for item in member_intervals)
    upper, upper_model = max((item[1], item[2]) for item in member_intervals)
    worst_interval = WorstCaseInterval(
        level=spec.interval_level,
        lower=lower,
        lower_model_id=lower_model,
        upper=upper,
        upper_model_id=upper_model,
    )
    signs = tuple(item.mean_sign for item in distributions)
    sign_pairs = tuple(combinations(signs, 2))
    disagreement_count = sum(first != second for first, second in sign_pairs)
    sign_disagreement_fraction = disagreement_count / len(sign_pairs)
    return _DerivedEnsemble(
        status=StructuralEnsembleStatus.COMPLETED,
        member_audit=audit_tuple,
        member_coverage=1.0,
        weight_coverage=math.fsum(item.weight for item in spec.member_weights),
        member_distributions=distributions,
        pooled_distribution=pooled_distribution,
        pooled_quantiles=pooled_quantiles,
        variance_decomposition=variance,
        sign_disagreement=disagreement_count > 0,
        sign_disagreement_fraction=sign_disagreement_fraction,
        worst_case_quantiles=tuple(worst_quantiles),
        worst_case_interval=worst_interval,
        abstain_reasons=(),
        synthetic_only=spec.synthetic_only,
    )


def evaluate_structural_ensemble(
    spec: StructuralEnsembleSpec,
    cell: StructuralOutcomeCell,
    results: tuple[StructuralMemberResult, ...],
) -> StructuralEnsembleArtifact:
    """Derive synthetic summary arithmetic or an explicit abstention artifact."""
    derived = _derive_structural_ensemble(spec, cell, results)
    ordered_results = tuple(sorted(results, key=lambda item: item.model_id))
    return StructuralEnsembleArtifact(
        status=derived.status,
        spec=spec,
        spec_sha256=spec.sha256,
        cell=cell,
        member_results=ordered_results,
        member_audit=derived.member_audit,
        member_coverage=derived.member_coverage,
        weight_coverage=derived.weight_coverage,
        member_distributions=derived.member_distributions,
        pooled_distribution=derived.pooled_distribution,
        pooled_quantiles=derived.pooled_quantiles,
        variance_decomposition=derived.variance_decomposition,
        sign_disagreement=derived.sign_disagreement,
        sign_disagreement_fraction=derived.sign_disagreement_fraction,
        worst_case_quantiles=derived.worst_case_quantiles,
        worst_case_interval=derived.worst_case_interval,
        abstain_reasons=derived.abstain_reasons,
        synthetic_only=derived.synthetic_only,
    )
