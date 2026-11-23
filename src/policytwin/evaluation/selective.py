"""Frozen selective-prediction evaluation without target-outcome tuning."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from statistics import fmean
from typing import NoReturn

from policytwin.provenance import canonical_json_bytes, sha256_bytes

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SYNTHETIC_THRESHOLD_RECEIPT_KIND = "synthetic_threshold_source_receipt_v2"
_SYNTHETIC_THRESHOLD_VERIFIER_ID = "policytwin.synthetic_threshold_verifier.v1"
_SYNTHETIC_THRESHOLD_CLAIM_SCOPE = "synthetic_mechanics_only"
_SYNTHETIC_THRESHOLD_AUTHORITY_KIND = "self_issued_synthetic_fixture"


class SelectivePredictionError(ValueError):
    """Raised when a risk-coverage evaluation would be incomplete or tuned."""


def _fail(message: str) -> NoReturn:
    raise SelectivePredictionError(message)


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        _fail(f"{field} must be a stable identifier")
    return value


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail(f"{field} must be a lowercase SHA-256 digest")
    return value


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        _fail(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{field} must be finite")
    return result


def _aware(value: object, field: str) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        _fail(f"{field} must be a timezone-aware datetime")
    return value.astimezone(UTC)


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        _fail(f"{field} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        _fail(f"{field} keys must be strings")
    return value


def _exact_fields(
    mapping: Mapping[str, object], expected: frozenset[str], field: str
) -> None:
    observed = frozenset(mapping)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        _fail(f"{field} fields differ: missing={missing!r}, extra={extra!r}")


def _datetime_from_mapping(value: object, field: str) -> datetime:
    if not isinstance(value, str):
        _fail(f"{field} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        _fail(f"{field} must be an ISO-8601 datetime")
    return _aware(parsed, field)


def _tuple_from_list(value: object, field: str) -> tuple[object, ...]:
    if not isinstance(value, list):
        _fail(f"{field} must be a JSON array")
    return tuple(value)


def _string(value: object, field: str) -> str:
    if not isinstance(value, str):
        _fail(f"{field} must be a string")
    return value


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{field} must be an integer")
    return value


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        _fail(f"{field} must be boolean")
    return value


class ThresholdSource(StrEnum):
    """Permitted origins for an abstention threshold set."""

    PRE_POLICY_VALIDATION = "pre_policy_validation"
    SYNTHETIC_FIXTURE = "synthetic_fixture"


def _threshold_tuple(value: object) -> tuple[float, ...]:
    if not isinstance(value, tuple) or not value:
        _fail("uncertainty_thresholds must be a non-empty immutable tuple")
    thresholds = tuple(_finite(item, "uncertainty threshold") for item in value)
    if any(item < 0.0 for item in thresholds):
        _fail("uncertainty thresholds must be non-negative")
    if tuple(sorted(set(thresholds))) != thresholds:
        _fail("uncertainty thresholds must be unique and strictly increasing")
    return thresholds


def _synthetic_threshold_source_content(
    *,
    study_protocol_sha256: str,
    source_artifact_sha256: str,
    thresholds_frozen_at: datetime,
    target_outcomes_first_available_at: datetime,
    uncertainty_thresholds: tuple[float, ...],
) -> dict[str, object]:
    return {
        "source_artifact_sha256": source_artifact_sha256,
        "study_protocol_sha256": study_protocol_sha256,
        "synthetic_only": True,
        "target_outcomes_first_available_at": (
            target_outcomes_first_available_at.isoformat()
        ),
        "threshold_source": ThresholdSource.SYNTHETIC_FIXTURE.value,
        "thresholds_frozen_at": thresholds_frozen_at.isoformat(),
        "uncertainty_thresholds": list(uncertainty_thresholds),
    }


@dataclass(frozen=True, slots=True)
class SyntheticThresholdSourceReceipt:
    """Self-issued commitment to exact synthetic threshold fixture content.

    This receipt detects drift after construction.  Its public deterministic
    issuer and caller-supplied timestamps are intentionally recorded as
    non-independent authority, so it cannot establish preregistration.
    """

    schema_version: int
    receipt_kind: str
    verifier_id: str
    claim_scope: str
    authority_kind: str
    study_protocol_sha256: str
    source_artifact_sha256: str
    source_content_sha256: str
    thresholds_frozen_at: datetime
    target_outcomes_first_available_at: datetime
    uncertainty_thresholds: tuple[float, ...]
    synthetic_only: bool

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 2:
            _fail("synthetic threshold receipt schema_version must equal 2")
        if self.receipt_kind != _SYNTHETIC_THRESHOLD_RECEIPT_KIND:
            _fail("synthetic threshold receipt kind is unsupported")
        if self.verifier_id != _SYNTHETIC_THRESHOLD_VERIFIER_ID:
            _fail("synthetic threshold receipt implementation is unsupported")
        if self.claim_scope != _SYNTHETIC_THRESHOLD_CLAIM_SCOPE:
            _fail("synthetic threshold receipt claim scope is invalid")
        if self.authority_kind != _SYNTHETIC_THRESHOLD_AUTHORITY_KIND:
            _fail("synthetic threshold receipt authority kind is invalid")
        _sha256(self.study_protocol_sha256, "receipt study_protocol_sha256")
        _sha256(self.source_artifact_sha256, "receipt source_artifact_sha256")
        _sha256(self.source_content_sha256, "receipt source_content_sha256")
        frozen_at = _aware(self.thresholds_frozen_at, "receipt thresholds_frozen_at")
        available_at = _aware(
            self.target_outcomes_first_available_at,
            "receipt target_outcomes_first_available_at",
        )
        if frozen_at >= available_at:
            _fail("synthetic threshold receipt does not predate target outcomes")
        thresholds = _threshold_tuple(self.uncertainty_thresholds)
        if self.synthetic_only is not True:
            _fail("synthetic threshold receipt cannot authorize non-synthetic use")
        object.__setattr__(self, "thresholds_frozen_at", frozen_at)
        object.__setattr__(self, "target_outcomes_first_available_at", available_at)
        object.__setattr__(self, "uncertainty_thresholds", thresholds)
        expected_content_sha256 = sha256_bytes(
            canonical_json_bytes(
                _synthetic_threshold_source_content(
                    study_protocol_sha256=self.study_protocol_sha256,
                    source_artifact_sha256=self.source_artifact_sha256,
                    thresholds_frozen_at=frozen_at,
                    target_outcomes_first_available_at=available_at,
                    uncertainty_thresholds=thresholds,
                )
            )
        )
        if self.source_content_sha256 != expected_content_sha256:
            _fail("synthetic threshold receipt content commitment mismatch")

    @classmethod
    def issue(
        cls,
        *,
        study_protocol_sha256: str,
        source_artifact_sha256: str,
        thresholds_frozen_at: datetime,
        target_outcomes_first_available_at: datetime,
        uncertainty_thresholds: tuple[float, ...],
    ) -> SyntheticThresholdSourceReceipt:
        """Self-issue a mechanics-only receipt from complete fixture content."""

        protocol_sha256 = _sha256(
            study_protocol_sha256, "receipt study_protocol_sha256"
        )
        artifact_sha256 = _sha256(
            source_artifact_sha256, "receipt source_artifact_sha256"
        )
        frozen_at = _aware(thresholds_frozen_at, "receipt thresholds_frozen_at")
        available_at = _aware(
            target_outcomes_first_available_at,
            "receipt target_outcomes_first_available_at",
        )
        thresholds = _threshold_tuple(uncertainty_thresholds)
        content_sha256 = sha256_bytes(
            canonical_json_bytes(
                _synthetic_threshold_source_content(
                    study_protocol_sha256=protocol_sha256,
                    source_artifact_sha256=artifact_sha256,
                    thresholds_frozen_at=frozen_at,
                    target_outcomes_first_available_at=available_at,
                    uncertainty_thresholds=thresholds,
                )
            )
        )
        return cls(
            schema_version=2,
            receipt_kind=_SYNTHETIC_THRESHOLD_RECEIPT_KIND,
            verifier_id=_SYNTHETIC_THRESHOLD_VERIFIER_ID,
            claim_scope=_SYNTHETIC_THRESHOLD_CLAIM_SCOPE,
            authority_kind=_SYNTHETIC_THRESHOLD_AUTHORITY_KIND,
            study_protocol_sha256=protocol_sha256,
            source_artifact_sha256=artifact_sha256,
            source_content_sha256=content_sha256,
            thresholds_frozen_at=frozen_at,
            target_outcomes_first_available_at=available_at,
            uncertainty_thresholds=thresholds,
            synthetic_only=True,
        )

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> SyntheticThresholdSourceReceipt:
        """Load and revalidate an exact synthetic threshold-source receipt."""

        mapping = _mapping(value, "synthetic threshold receipt")
        _exact_fields(
            mapping,
            frozenset(
                {
                    "claim_scope",
                    "authority_kind",
                    "receipt_kind",
                    "schema_version",
                    "source_artifact_sha256",
                    "source_content_sha256",
                    "study_protocol_sha256",
                    "synthetic_only",
                    "target_outcomes_first_available_at",
                    "thresholds_frozen_at",
                    "uncertainty_thresholds",
                    "verifier_id",
                }
            ),
            "synthetic threshold receipt",
        )
        return cls(
            schema_version=_integer(mapping["schema_version"], "schema_version"),
            receipt_kind=_string(mapping["receipt_kind"], "receipt_kind"),
            verifier_id=_string(mapping["verifier_id"], "verifier_id"),
            claim_scope=_string(mapping["claim_scope"], "claim_scope"),
            authority_kind=_string(mapping["authority_kind"], "authority_kind"),
            study_protocol_sha256=_sha256(
                mapping["study_protocol_sha256"], "study_protocol_sha256"
            ),
            source_artifact_sha256=_sha256(
                mapping["source_artifact_sha256"], "source_artifact_sha256"
            ),
            source_content_sha256=_sha256(
                mapping["source_content_sha256"], "source_content_sha256"
            ),
            thresholds_frozen_at=_datetime_from_mapping(
                mapping["thresholds_frozen_at"], "thresholds_frozen_at"
            ),
            target_outcomes_first_available_at=_datetime_from_mapping(
                mapping["target_outcomes_first_available_at"],
                "target_outcomes_first_available_at",
            ),
            uncertainty_thresholds=_threshold_tuple(
                _tuple_from_list(
                    mapping["uncertainty_thresholds"], "uncertainty_thresholds"
                )
            ),
            synthetic_only=_boolean(mapping["synthetic_only"], "synthetic_only"),
        )

    @property
    def sha256(self) -> str:
        """Return the complete receipt commitment."""

        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def to_mapping(self) -> dict[str, object]:
        """Return the exact source content plus fixed verifier identity."""

        return {
            "authority_kind": self.authority_kind,
            "claim_scope": self.claim_scope,
            "receipt_kind": self.receipt_kind,
            "schema_version": self.schema_version,
            "source_artifact_sha256": self.source_artifact_sha256,
            "source_content_sha256": self.source_content_sha256,
            "study_protocol_sha256": self.study_protocol_sha256,
            "synthetic_only": self.synthetic_only,
            "target_outcomes_first_available_at": (
                self.target_outcomes_first_available_at.isoformat()
            ),
            "thresholds_frozen_at": self.thresholds_frozen_at.isoformat(),
            "uncertainty_thresholds": list(self.uncertainty_thresholds),
            "verifier_id": self.verifier_id,
        }


@dataclass(frozen=True, slots=True)
class SelectionRule:
    """Frozen lower-uncertainty rule with an explicit authority boundary."""

    schema_version: int
    rule_id: str
    study_protocol_sha256: str
    threshold_source: ThresholdSource
    threshold_source_artifact_sha256: str
    thresholds_frozen_at: datetime
    target_outcomes_first_available_at: datetime
    uncertainty_thresholds: tuple[float, ...]
    loss_metric_id: str
    minimum_retained: int
    monotonicity_tolerance: float = 0.0
    synthetic_only: bool = True
    threshold_source_receipt: SyntheticThresholdSourceReceipt | None = None

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 2:
            _fail("selection rule schema_version must equal 2")
        _identifier(self.rule_id, "rule_id")
        _sha256(self.study_protocol_sha256, "study_protocol_sha256")
        if not isinstance(self.threshold_source, ThresholdSource):
            _fail("threshold_source must be a ThresholdSource")
        _sha256(
            self.threshold_source_artifact_sha256,
            "threshold_source_artifact_sha256",
        )
        object.__setattr__(
            self,
            "thresholds_frozen_at",
            _aware(self.thresholds_frozen_at, "thresholds_frozen_at"),
        )
        object.__setattr__(
            self,
            "target_outcomes_first_available_at",
            _aware(
                self.target_outcomes_first_available_at,
                "target_outcomes_first_available_at",
            ),
        )
        if self.thresholds_frozen_at >= self.target_outcomes_first_available_at:
            _fail("selection thresholds must be frozen before target outcomes exist")
        thresholds = _threshold_tuple(self.uncertainty_thresholds)
        object.__setattr__(self, "uncertainty_thresholds", thresholds)
        _identifier(self.loss_metric_id, "loss_metric_id")
        if (
            isinstance(self.minimum_retained, bool)
            or not isinstance(self.minimum_retained, int)
            or self.minimum_retained < 1
        ):
            _fail("minimum_retained must be a positive integer")
        tolerance = _finite(self.monotonicity_tolerance, "monotonicity_tolerance")
        if tolerance < 0.0:
            _fail("monotonicity_tolerance must be non-negative")
        object.__setattr__(self, "monotonicity_tolerance", tolerance)
        if not isinstance(self.synthetic_only, bool):
            _fail("synthetic_only must be boolean")
        if self.synthetic_only is not True:
            _fail(
                "non-synthetic selection rules require a future typed verifier "
                "receipt; hashes and timestamps cannot authorize real evaluation"
            )
        if self.threshold_source is not ThresholdSource.SYNTHETIC_FIXTURE:
            _fail("synthetic marker and threshold source disagree")
        receipt = self.threshold_source_receipt
        if receipt is None:
            _fail(
                "threshold_source_receipt is required explicitly; selection-rule "
                "construction must never auto-sign a preregistration claim"
            )
        if type(receipt) is not SyntheticThresholdSourceReceipt:
            _fail("threshold_source_receipt must be a typed synthetic fixture receipt")
        revalidated_receipt = SyntheticThresholdSourceReceipt(
            schema_version=receipt.schema_version,
            receipt_kind=receipt.receipt_kind,
            verifier_id=receipt.verifier_id,
            claim_scope=receipt.claim_scope,
            authority_kind=receipt.authority_kind,
            study_protocol_sha256=receipt.study_protocol_sha256,
            source_artifact_sha256=receipt.source_artifact_sha256,
            source_content_sha256=receipt.source_content_sha256,
            thresholds_frozen_at=receipt.thresholds_frozen_at,
            target_outcomes_first_available_at=(
                receipt.target_outcomes_first_available_at
            ),
            uncertainty_thresholds=receipt.uncertainty_thresholds,
            synthetic_only=receipt.synthetic_only,
        )
        if revalidated_receipt != receipt:
            _fail("threshold source receipt is not canonical")
        receipt = revalidated_receipt
        expected_receipt = SyntheticThresholdSourceReceipt.issue(
            study_protocol_sha256=self.study_protocol_sha256,
            source_artifact_sha256=self.threshold_source_artifact_sha256,
            thresholds_frozen_at=self.thresholds_frozen_at,
            target_outcomes_first_available_at=(
                self.target_outcomes_first_available_at
            ),
            uncertainty_thresholds=self.uncertainty_thresholds,
        )
        if receipt != expected_receipt:
            _fail("threshold source receipt/content commitment drift")
        object.__setattr__(self, "threshold_source_receipt", receipt)

    @property
    def has_independent_preregistration_authority(self) -> bool:
        """Return false: the bundled receipt is a self-issued mechanics fixture."""

        return False

    @property
    def sha256(self) -> str:
        """Return the full frozen selection-rule commitment."""
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> SelectionRule:
        """Load an exact rule and revalidate its retained source receipt."""

        mapping = _mapping(value, "selection rule")
        _exact_fields(
            mapping,
            frozenset(
                {
                    "loss_metric_id",
                    "minimum_retained",
                    "monotonicity_tolerance",
                    "rule_id",
                    "schema_version",
                    "study_protocol_sha256",
                    "synthetic_only",
                    "target_outcomes_first_available_at",
                    "threshold_source",
                    "threshold_source_artifact_sha256",
                    "threshold_source_receipt",
                    "thresholds_frozen_at",
                    "uncertainty_thresholds",
                }
            ),
            "selection rule",
        )
        source_value = _string(mapping["threshold_source"], "threshold_source")
        try:
            source = ThresholdSource(source_value)
        except ValueError:
            _fail("threshold_source is not registered")
        receipt_mapping = _mapping(
            mapping["threshold_source_receipt"], "threshold_source_receipt"
        )
        return cls(
            schema_version=_integer(mapping["schema_version"], "schema_version"),
            rule_id=_identifier(mapping["rule_id"], "rule_id"),
            study_protocol_sha256=_sha256(
                mapping["study_protocol_sha256"], "study_protocol_sha256"
            ),
            threshold_source=source,
            threshold_source_artifact_sha256=_sha256(
                mapping["threshold_source_artifact_sha256"],
                "threshold_source_artifact_sha256",
            ),
            thresholds_frozen_at=_datetime_from_mapping(
                mapping["thresholds_frozen_at"], "thresholds_frozen_at"
            ),
            target_outcomes_first_available_at=_datetime_from_mapping(
                mapping["target_outcomes_first_available_at"],
                "target_outcomes_first_available_at",
            ),
            uncertainty_thresholds=_threshold_tuple(
                _tuple_from_list(
                    mapping["uncertainty_thresholds"], "uncertainty_thresholds"
                )
            ),
            loss_metric_id=_identifier(mapping["loss_metric_id"], "loss_metric_id"),
            minimum_retained=_integer(mapping["minimum_retained"], "minimum_retained"),
            monotonicity_tolerance=_finite(
                mapping["monotonicity_tolerance"], "monotonicity_tolerance"
            ),
            synthetic_only=_boolean(mapping["synthetic_only"], "synthetic_only"),
            threshold_source_receipt=SyntheticThresholdSourceReceipt.from_mapping(
                receipt_mapping
            ),
        )

    def to_mapping(self) -> dict[str, object]:
        """Return the canonical rule representation."""
        receipt = self.threshold_source_receipt
        if receipt is None:  # Defensive: __post_init__ always installs it.
            _fail("selection rule is missing its threshold source receipt")
        return {
            "loss_metric_id": self.loss_metric_id,
            "minimum_retained": self.minimum_retained,
            "monotonicity_tolerance": self.monotonicity_tolerance,
            "rule_id": self.rule_id,
            "schema_version": self.schema_version,
            "study_protocol_sha256": self.study_protocol_sha256,
            "synthetic_only": self.synthetic_only,
            "target_outcomes_first_available_at": (
                self.target_outcomes_first_available_at.isoformat()
            ),
            "threshold_source": self.threshold_source.value,
            "threshold_source_artifact_sha256": (self.threshold_source_artifact_sha256),
            "threshold_source_receipt": receipt.to_mapping(),
            "thresholds_frozen_at": self.thresholds_frozen_at.isoformat(),
            "uncertainty_thresholds": list(self.uncertainty_thresholds),
        }


@dataclass(frozen=True, slots=True)
class SelectivePredictionRecord:
    """One frozen prediction's pre-outcome uncertainty and later observed loss."""

    prediction_id: str
    uncertainty_score: float
    loss: float

    def __post_init__(self) -> None:
        _identifier(self.prediction_id, "prediction_id")
        uncertainty = _finite(self.uncertainty_score, "uncertainty_score")
        loss = _finite(self.loss, "loss")
        if uncertainty < 0.0:
            _fail("uncertainty_score must be non-negative")
        if loss < 0.0:
            _fail("loss must be non-negative")
        object.__setattr__(self, "uncertainty_score", uncertainty)
        object.__setattr__(self, "loss", loss)

    def to_mapping(self) -> dict[str, object]:
        """Return the complete retained prediction record."""

        return {
            "loss": self.loss,
            "prediction_id": self.prediction_id,
            "uncertainty_score": self.uncertainty_score,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> SelectivePredictionRecord:
        """Load one exact retained prediction record."""

        mapping = _mapping(value, "selective prediction record")
        _exact_fields(
            mapping,
            frozenset({"loss", "prediction_id", "uncertainty_score"}),
            "selective prediction record",
        )
        return cls(
            prediction_id=_identifier(mapping["prediction_id"], "prediction_id"),
            uncertainty_score=_finite(
                mapping["uncertainty_score"], "uncertainty_score"
            ),
            loss=_finite(mapping["loss"], "loss"),
        )


@dataclass(frozen=True, slots=True)
class RiskCoveragePoint:
    """Observed held-out loss for one preregistered retention threshold."""

    uncertainty_threshold: float
    retained_count: int
    total_count: int
    coverage: float
    mean_loss: float

    def __post_init__(self) -> None:
        threshold = _finite(self.uncertainty_threshold, "uncertainty_threshold")
        coverage = _finite(self.coverage, "coverage")
        mean_loss = _finite(self.mean_loss, "mean_loss")
        if threshold < 0.0 or not 0.0 < coverage <= 1.0 or mean_loss < 0.0:
            _fail("risk-coverage point values are outside their valid domains")
        if (
            isinstance(self.retained_count, bool)
            or not isinstance(self.retained_count, int)
            or isinstance(self.total_count, bool)
            or not isinstance(self.total_count, int)
            or not 0 < self.retained_count <= self.total_count
            or coverage != self.retained_count / self.total_count
        ):
            _fail("risk-coverage point counts and coverage are inconsistent")

    def to_mapping(self) -> dict[str, object]:
        """Return the canonical point representation."""
        return {
            "coverage": self.coverage,
            "mean_loss": self.mean_loss,
            "retained_count": self.retained_count,
            "total_count": self.total_count,
            "uncertainty_threshold": self.uncertainty_threshold,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> RiskCoveragePoint:
        """Load one exact point for subsequent full-curve rederivation."""

        mapping = _mapping(value, "risk-coverage point")
        _exact_fields(
            mapping,
            frozenset(
                {
                    "coverage",
                    "mean_loss",
                    "retained_count",
                    "total_count",
                    "uncertainty_threshold",
                }
            ),
            "risk-coverage point",
        )
        return cls(
            uncertainty_threshold=_finite(
                mapping["uncertainty_threshold"], "uncertainty_threshold"
            ),
            retained_count=_integer(mapping["retained_count"], "retained_count"),
            total_count=_integer(mapping["total_count"], "total_count"),
            coverage=_finite(mapping["coverage"], "coverage"),
            mean_loss=_finite(mapping["mean_loss"], "mean_loss"),
        )


@dataclass(frozen=True, slots=True)
class SelectivePredictionEvaluation:
    """Complete risk-coverage curve and an explicit monotonicity diagnostic."""

    rule: SelectionRule
    records: tuple[SelectivePredictionRecord, ...]
    rule_sha256: str
    record_manifest_sha256: str
    full_coverage_mean_loss: float
    points: tuple[RiskCoveragePoint, ...]
    monotone_selective_improvement: bool
    lowest_coverage_improves_over_full: bool
    synthetic_only: bool

    def __post_init__(self) -> None:
        rule = _revalidated_rule(self.rule)
        records = _validated_records(self.records)
        derived = _derive_selective_statistics(records, rule)
        _sha256(self.rule_sha256, "rule_sha256")
        _sha256(self.record_manifest_sha256, "record_manifest_sha256")
        full_loss = _finite(self.full_coverage_mean_loss, "full_coverage_mean_loss")
        if full_loss < 0.0:
            _fail("full_coverage_mean_loss must be non-negative")
        if not isinstance(self.points, tuple) or not self.points:
            _fail("selective evaluation requires an immutable point tuple")
        validated_points: list[RiskCoveragePoint] = []
        for point in self.points:
            if type(point) is not RiskCoveragePoint:
                _fail("selective evaluation contains an invalid point")
            validated_points.append(
                RiskCoveragePoint(
                    uncertainty_threshold=point.uncertainty_threshold,
                    retained_count=point.retained_count,
                    total_count=point.total_count,
                    coverage=point.coverage,
                    mean_loss=point.mean_loss,
                )
            )
        for field in (
            "monotone_selective_improvement",
            "lowest_coverage_improves_over_full",
            "synthetic_only",
        ):
            if not isinstance(getattr(self, field), bool):
                _fail(f"{field} must be boolean")
        if self.rule_sha256 != rule.sha256:
            _fail("selective evaluation rule hash mismatch")
        if self.record_manifest_sha256 != derived.record_manifest_sha256:
            _fail("selective evaluation record manifest hash mismatch")
        if full_loss != derived.full_coverage_mean_loss:
            _fail("selective evaluation full-coverage loss mismatch")
        if tuple(validated_points) != derived.points:
            _fail("selective evaluation curve points do not match retained records")
        if (
            self.monotone_selective_improvement
            is not derived.monotone_selective_improvement
        ):
            _fail("selective evaluation monotonicity diagnostic mismatch")
        if (
            self.lowest_coverage_improves_over_full
            is not derived.lowest_coverage_improves_over_full
        ):
            _fail("selective evaluation improvement diagnostic mismatch")
        if self.synthetic_only is not rule.synthetic_only:
            _fail("selective evaluation synthetic marker disagrees with its rule")
        object.__setattr__(self, "rule", rule)
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "points", tuple(validated_points))
        object.__setattr__(self, "full_coverage_mean_loss", full_loss)

    @property
    def sha256(self) -> str:
        """Return the evaluation commitment."""
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> SelectivePredictionEvaluation:
        """Load an artifact and rederive every result from its rule and records."""

        mapping = _mapping(value, "selective prediction evaluation")
        _exact_fields(
            mapping,
            frozenset(
                {
                    "full_coverage_mean_loss",
                    "lowest_coverage_improves_over_full",
                    "monotone_selective_improvement",
                    "points",
                    "records",
                    "record_manifest_sha256",
                    "rule",
                    "rule_sha256",
                    "schema_version",
                    "synthetic_only",
                }
            ),
            "selective prediction evaluation",
        )
        schema_version = _integer(mapping["schema_version"], "schema_version")
        if schema_version != 2:
            _fail("selective prediction evaluation schema_version must equal 2")
        rule = SelectionRule.from_mapping(_mapping(mapping["rule"], "rule"))
        record_items = _tuple_from_list(mapping["records"], "records")
        point_items = _tuple_from_list(mapping["points"], "points")
        records = tuple(
            SelectivePredictionRecord.from_mapping(_mapping(item, "record"))
            for item in record_items
        )
        points = tuple(
            RiskCoveragePoint.from_mapping(_mapping(item, "point"))
            for item in point_items
        )
        return cls(
            rule=rule,
            records=records,
            rule_sha256=_sha256(mapping["rule_sha256"], "rule_sha256"),
            record_manifest_sha256=_sha256(
                mapping["record_manifest_sha256"], "record_manifest_sha256"
            ),
            full_coverage_mean_loss=_finite(
                mapping["full_coverage_mean_loss"], "full_coverage_mean_loss"
            ),
            points=points,
            monotone_selective_improvement=_boolean(
                mapping["monotone_selective_improvement"],
                "monotone_selective_improvement",
            ),
            lowest_coverage_improves_over_full=_boolean(
                mapping["lowest_coverage_improves_over_full"],
                "lowest_coverage_improves_over_full",
            ),
            synthetic_only=_boolean(mapping["synthetic_only"], "synthetic_only"),
        )

    def to_mapping(self) -> dict[str, object]:
        """Return a machine-readable selective-evaluation artifact."""
        return {
            "full_coverage_mean_loss": self.full_coverage_mean_loss,
            "lowest_coverage_improves_over_full": (
                self.lowest_coverage_improves_over_full
            ),
            "monotone_selective_improvement": self.monotone_selective_improvement,
            "points": [point.to_mapping() for point in self.points],
            "records": [record.to_mapping() for record in self.records],
            "record_manifest_sha256": self.record_manifest_sha256,
            "rule": self.rule.to_mapping(),
            "rule_sha256": self.rule_sha256,
            "schema_version": 2,
            "synthetic_only": self.synthetic_only,
        }


@dataclass(frozen=True, slots=True)
class _DerivedSelectiveStatistics:
    record_manifest_sha256: str
    full_coverage_mean_loss: float
    points: tuple[RiskCoveragePoint, ...]
    monotone_selective_improvement: bool
    lowest_coverage_improves_over_full: bool


def _revalidated_rule(rule: object) -> SelectionRule:
    if type(rule) is not SelectionRule:
        _fail("rule must be a SelectionRule")
    assert isinstance(rule, SelectionRule)
    validated = SelectionRule(
        schema_version=rule.schema_version,
        rule_id=rule.rule_id,
        study_protocol_sha256=rule.study_protocol_sha256,
        threshold_source=rule.threshold_source,
        threshold_source_artifact_sha256=rule.threshold_source_artifact_sha256,
        thresholds_frozen_at=rule.thresholds_frozen_at,
        target_outcomes_first_available_at=rule.target_outcomes_first_available_at,
        uncertainty_thresholds=rule.uncertainty_thresholds,
        loss_metric_id=rule.loss_metric_id,
        minimum_retained=rule.minimum_retained,
        monotonicity_tolerance=rule.monotonicity_tolerance,
        synthetic_only=rule.synthetic_only,
        threshold_source_receipt=rule.threshold_source_receipt,
    )
    if validated != rule:
        _fail("selection rule is not canonical")
    return validated


def _validated_records(
    records: object,
) -> tuple[SelectivePredictionRecord, ...]:
    if not isinstance(records, tuple) or not records:
        _fail("records must be a non-empty immutable SelectivePredictionRecord tuple")
    validated: list[SelectivePredictionRecord] = []
    for record in records:
        if type(record) is not SelectivePredictionRecord:
            _fail(
                "records must be a non-empty immutable SelectivePredictionRecord tuple"
            )
        checked = SelectivePredictionRecord(
            prediction_id=record.prediction_id,
            uncertainty_score=record.uncertainty_score,
            loss=record.loss,
        )
        if checked != record:
            _fail("selective prediction record is not canonical")
        validated.append(checked)
    ordered = tuple(sorted(validated, key=lambda item: item.prediction_id))
    identifiers = tuple(record.prediction_id for record in ordered)
    if len(set(identifiers)) != len(identifiers):
        _fail("prediction IDs must be unique")
    return ordered


def _derive_selective_statistics(
    records: tuple[SelectivePredictionRecord, ...],
    rule: SelectionRule,
) -> _DerivedSelectiveStatistics:
    record_manifest = [record.to_mapping() for record in records]
    points: list[RiskCoveragePoint] = []
    for threshold in rule.uncertainty_thresholds:
        retained = tuple(
            record for record in records if record.uncertainty_score <= threshold
        )
        if len(retained) < rule.minimum_retained:
            _fail(
                "a preregistered threshold retained fewer than minimum_retained "
                f"records: threshold={threshold!r}"
            )
        points.append(
            RiskCoveragePoint(
                uncertainty_threshold=threshold,
                retained_count=len(retained),
                total_count=len(records),
                coverage=len(retained) / len(records),
                mean_loss=fmean(record.loss for record in retained),
            )
        )
    tolerance = rule.monotonicity_tolerance
    monotone = all(
        earlier.mean_loss <= later.mean_loss + tolerance
        for earlier, later in zip(points, points[1:], strict=False)
    )
    full_loss = fmean(record.loss for record in records)
    return _DerivedSelectiveStatistics(
        record_manifest_sha256=sha256_bytes(canonical_json_bytes(record_manifest)),
        full_coverage_mean_loss=full_loss,
        points=tuple(points),
        monotone_selective_improvement=monotone,
        lowest_coverage_improves_over_full=points[0].mean_loss < full_loss,
    )


def evaluate_selective_predictions(
    records: tuple[SelectivePredictionRecord, ...],
    rule: SelectionRule,
) -> SelectivePredictionEvaluation:
    """Evaluate only preregistered thresholds; never optimize on target losses."""
    validated_rule = _revalidated_rule(rule)
    ordered = _validated_records(records)
    derived = _derive_selective_statistics(ordered, validated_rule)
    return SelectivePredictionEvaluation(
        rule=validated_rule,
        records=ordered,
        rule_sha256=validated_rule.sha256,
        record_manifest_sha256=derived.record_manifest_sha256,
        full_coverage_mean_loss=derived.full_coverage_mean_loss,
        points=derived.points,
        monotone_selective_improvement=derived.monotone_selective_improvement,
        lowest_coverage_improves_over_full=(derived.lowest_coverage_improves_over_full),
        synthetic_only=validated_rule.synthetic_only,
    )


__all__ = [
    "RiskCoveragePoint",
    "SelectionRule",
    "SelectivePredictionError",
    "SelectivePredictionEvaluation",
    "SelectivePredictionRecord",
    "SyntheticThresholdSourceReceipt",
    "ThresholdSource",
    "evaluate_selective_predictions",
]
