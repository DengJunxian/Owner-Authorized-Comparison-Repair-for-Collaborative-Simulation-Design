"""Pure reporting derivation for the audited synthetic CLOB branch bundle.

This module does not execute or estimate a market model.  Its source boundary is
the verifier-owned :func:`audit_written_aamas_integrated_branch` receipt.  That
audit performs deterministic replay; only after it passes do the functions below
read every registered child and derive descriptive replicate statistics.

The reported quantiles describe the finite synthetic replicate distribution.
They are not confidence, credible, prediction, or tolerance intervals.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import stat
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from statistics import fmean
from typing import Literal, NoReturn, Self, cast

from policytwin.experiments.aamas_integrated_branch import (
    AAMAS_INTEGRATED_BRANCH_ARTIFACT_SCHEMA_VERSION,
    AAMAS_INTEGRATED_BRANCH_CHILD_SCHEMA_VERSION,
    AAMASIntegratedBranchAudit,
    AAMASIntegratedBranchError,
    AAMASIntegratedBranchPlan,
    audit_written_aamas_integrated_branch,
)
from policytwin.provenance import (
    MetadataValidationError,
    canonical_json_bytes,
    read_json_bytes_object,
    sha256_bytes,
    sha256_file,
    write_text_file,
)

AAMAS_INTEGRATED_BRANCH_REPORTING_SPEC_SCHEMA_VERSION = (
    "policytwin.aamas-integrated-branch-reporting-spec.v1"
)
AAMAS_INTEGRATED_BRANCH_REPORT_SCHEMA_VERSION = (
    "policytwin.aamas-integrated-branch-report.v1"
)
AAMAS_INTEGRATED_BRANCH_REPORT_CLAIM_SCOPE = (
    "synthetic_integrated_clob_branch_descriptive_statistics_only"
)
AAMAS_INTEGRATED_BRANCH_REPORT_FILENAME = "integrated-branch-report.json"
AAMAS_INTEGRATED_BRANCH_REPORT_SEAL_FILENAME = "integrated-branch-report.sha256"

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_EVENT_KEY = re.compile(r"^ptk1:[0-9a-f]{64}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_QUANTILE_INTERPRETATION = (
    "descriptive_finite_synthetic_replicate_quantiles_not_confidence_intervals"
)


class AAMASIntegratedBranchReportingError(RuntimeError):
    """Base error for integrated-branch reporting."""


class AAMASIntegratedBranchReportingSchemaError(
    AAMASIntegratedBranchReportingError, ValueError
):
    """Raised when a reporting object does not satisfy its frozen schema."""


class AAMASIntegratedBranchReportingIntegrityError(AAMASIntegratedBranchReportingError):
    """Raised when source or report bytes fail a non-bypassable audit."""


def _fail(message: str) -> NoReturn:
    raise AAMASIntegratedBranchReportingSchemaError(message)


def _integrity_fail(message: str) -> NoReturn:
    raise AAMASIntegratedBranchReportingIntegrityError(message)


def _strict_fields(
    value: Mapping[str, object], expected: frozenset[str], field: str
) -> None:
    observed = frozenset(value)
    if observed != expected:
        _fail(
            f"{field} fields differ: missing={sorted(expected - observed)!r}, "
            f"extra={sorted(observed - expected)!r}"
        )


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        _fail(f"{field} must be an object with string keys")
    return cast(Mapping[str, object], value)


def _sequence(value: object, field: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        _fail(f"{field} must be an array")
    return cast(Sequence[object], value)


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{field} must be a non-empty string")
    return value


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        _fail(f"{field} must be a stable ASCII identifier")
    return value


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail(f"{field} must be a lowercase SHA-256 digest")
    return value


def _stable_regular_bytes(path: Path, field: str) -> bytes:
    """Read one singly linked file and reject an identity-changing race."""

    try:
        before = path.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or before.st_nlink != 1
        ):
            _integrity_fail(f"{field} must be a singly linked regular file")
        payload = path.read_bytes()
        after = path.lstat()
    except AAMASIntegratedBranchReportingIntegrityError:
        raise
    except OSError as error:
        raise AAMASIntegratedBranchReportingIntegrityError(
            f"cannot read stable {field}"
        ) from error

    def identity(item: os.stat_result) -> tuple[int, int, int, int, int, int]:
        return (
            item.st_dev,
            item.st_ino,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
            item.st_nlink,
        )

    if identity(before) != identity(after) or len(payload) != before.st_size:
        _integrity_fail(f"{field} changed while being read")
    return payload


def _integer(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"{field} must be an integer >= {minimum}")
    return value


def _signed_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{field} must be an integer")
    return value


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        _fail(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{field} must be finite")
    return result


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        _fail(f"{field} must be boolean")
    return value


def _fraction(value: object, field: str) -> float:
    result = _finite(value, field)
    if not 0.0 <= result <= 1.0:
        _fail(f"{field} must lie in [0, 1]")
    return result


def _sample_variance(values: Sequence[float]) -> float:
    if len(values) < 2:
        _integrity_fail(
            "sample variance with ddof=1 requires at least two complete replicates"
        )
    mean = fmean(values)
    return sum((value - mean) ** 2 for value in values) / (len(values) - 1)


def _type7_quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        _integrity_fail("a descriptive quantile requires source draws")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    value = ordered[lower] + weight * (ordered[upper] - ordered[lower])
    return min(ordered[upper], max(ordered[lower], value))


def _rate(passed: int, total: int) -> float:
    if total < 1 or not 0 <= passed <= total:
        _integrity_fail("a reporting rate has an invalid numerator or denominator")
    return passed / total


@dataclass(frozen=True, slots=True)
class FrozenIntegratedBranchReportingSpec:
    """Versioned definitions fixed before inspecting formal output."""

    difference_definition: str = "candidate_minus_baseline"
    sample_variance_ddof: Literal[1] = 1
    mcse_definition: str = "sqrt_sample_variance_ddof1_over_replicate_count"
    quantile_method: str = "r_type_7_linear_interpolation"
    quantile_probabilities: tuple[float, float, float] = (0.05, 0.5, 0.95)
    quantile_interpretation: str = _QUANTILE_INTERPRETATION
    schema_version: str = AAMAS_INTEGRATED_BRANCH_REPORTING_SPEC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        expected: tuple[tuple[object, object], ...] = (
            (self.difference_definition, "candidate_minus_baseline"),
            (self.sample_variance_ddof, 1),
            (
                self.mcse_definition,
                "sqrt_sample_variance_ddof1_over_replicate_count",
            ),
            (self.quantile_method, "r_type_7_linear_interpolation"),
            (self.quantile_probabilities, (0.05, 0.5, 0.95)),
            (self.quantile_interpretation, _QUANTILE_INTERPRETATION),
            (
                self.schema_version,
                AAMAS_INTEGRATED_BRANCH_REPORTING_SPEC_SCHEMA_VERSION,
            ),
        )
        if any(observed != required for observed, required in expected):
            _fail("integrated branch reporting definitions are frozen by schema v1")

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def to_mapping(self) -> dict[str, object]:
        return {
            "difference_definition": self.difference_definition,
            "mcse_definition": self.mcse_definition,
            "quantile_interpretation": self.quantile_interpretation,
            "quantile_method": self.quantile_method,
            "quantile_probabilities": list(self.quantile_probabilities),
            "sample_variance_ddof": self.sample_variance_ddof,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(value, frozenset(cls.__annotations__), "reporting spec")
        probabilities = tuple(
            _finite(item, "quantile probability")
            for item in _sequence(
                value["quantile_probabilities"], "quantile_probabilities"
            )
        )
        if len(probabilities) != 3:
            _fail("quantile_probabilities must have exactly three values")
        return cls(
            difference_definition=_string(
                value["difference_definition"], "difference_definition"
            ),
            sample_variance_ddof=cast(
                Literal[1],
                _integer(value["sample_variance_ddof"], "sample_variance_ddof"),
            ),
            mcse_definition=_string(value["mcse_definition"], "mcse_definition"),
            quantile_method=_string(value["quantile_method"], "quantile_method"),
            quantile_probabilities=probabilities,
            quantile_interpretation=_string(
                value["quantile_interpretation"], "quantile_interpretation"
            ),
            schema_version=_string(value["schema_version"], "schema_version"),
        )


@dataclass(frozen=True, slots=True)
class IntegratedBranchDifference:
    """One retained candidate-minus-baseline source cell."""

    draw_id: str
    horizon_id: str
    group_id: str
    metric_name: str
    baseline_value: float
    candidate_value: float
    difference: float

    def __post_init__(self) -> None:
        for field in ("draw_id", "horizon_id", "group_id", "metric_name"):
            _identifier(getattr(self, field), field)
        baseline = _finite(self.baseline_value, "baseline_value")
        candidate = _finite(self.candidate_value, "candidate_value")
        difference = _finite(self.difference, "difference")
        if difference != candidate - baseline:
            _fail("reported difference must equal candidate minus baseline")

    @property
    def cell_key(self) -> tuple[str, str, str]:
        return self.horizon_id, self.group_id, self.metric_name

    def to_mapping(self) -> dict[str, object]:
        return {
            "baseline_value": self.baseline_value,
            "candidate_value": self.candidate_value,
            "difference": self.difference,
            "draw_id": self.draw_id,
            "group_id": self.group_id,
            "horizon_id": self.horizon_id,
            "metric_name": self.metric_name,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(value, frozenset(cls.__annotations__), "paired difference")
        return cls(
            draw_id=_identifier(value["draw_id"], "draw_id"),
            horizon_id=_identifier(value["horizon_id"], "horizon_id"),
            group_id=_identifier(value["group_id"], "group_id"),
            metric_name=_identifier(value["metric_name"], "metric_name"),
            baseline_value=_finite(value["baseline_value"], "baseline_value"),
            candidate_value=_finite(value["candidate_value"], "candidate_value"),
            difference=_finite(value["difference"], "difference"),
        )


@dataclass(frozen=True, slots=True)
class IntegratedBranchReplicateRow:
    """One complete source replicate; no failed row can be dropped."""

    replicate_id: int
    event_key: str
    child_path: str
    child_sha256: str
    shock_tape_sha256: str
    baseline_world_sha256: str
    baseline_run_sha256: str
    no_op_difference_sha256: str
    restriction_difference_sha256: str
    restriction_internal_event_delta: int
    restriction_nonzero_difference_count: int
    restriction_branch_observed: bool
    no_op_exact_zero: bool
    baseline_invariant: bool
    event_keyed_shocks_aligned: bool
    replay_byte_identical: bool
    pair_integrity_passed: bool
    no_op_differences: tuple[IntegratedBranchDifference, ...]
    restriction_differences: tuple[IntegratedBranchDifference, ...]

    def __post_init__(self) -> None:
        _integer(self.replicate_id, "replicate_id")
        if _EVENT_KEY.fullmatch(self.event_key) is None:
            _fail("event_key must use the exact ptk1 digest form")
        expected_path = PurePosixPath("runs", f"{self.replicate_id:06d}.json")
        if PurePosixPath(self.child_path) != expected_path:
            _fail("child_path is not the canonical replicate path")
        for field in (
            "child_sha256",
            "shock_tape_sha256",
            "baseline_world_sha256",
            "baseline_run_sha256",
            "no_op_difference_sha256",
            "restriction_difference_sha256",
        ):
            _sha256(getattr(self, field), field)
        _signed_integer(
            self.restriction_internal_event_delta,
            "restriction_internal_event_delta",
        )
        _integer(
            self.restriction_nonzero_difference_count,
            "restriction_nonzero_difference_count",
        )
        for field in (
            "restriction_branch_observed",
            "no_op_exact_zero",
            "baseline_invariant",
            "event_keyed_shocks_aligned",
            "replay_byte_identical",
            "pair_integrity_passed",
        ):
            _boolean(getattr(self, field), field)
        if not self.no_op_differences or not self.restriction_differences:
            _fail("a complete replicate must retain both difference registries")
        no_op_keys = tuple(item.cell_key for item in self.no_op_differences)
        restriction_keys = tuple(item.cell_key for item in self.restriction_differences)
        if no_op_keys != tuple(sorted(no_op_keys)) or len(set(no_op_keys)) != len(
            no_op_keys
        ):
            _fail("no-op cells must be complete, unique, and sorted")
        if restriction_keys != no_op_keys:
            _fail("restriction and no-op cells must have the same registry")
        if len({item.draw_id for item in self.no_op_differences}) != 1:
            _fail("no-op cells must retain one draw identifier")
        if tuple(item.draw_id for item in self.no_op_differences) != tuple(
            item.draw_id for item in self.restriction_differences
        ):
            _fail("restriction and no-op cells must retain the same draw identifier")
        if self.no_op_difference_sha256 != sha256_bytes(
            canonical_json_bytes([item.to_mapping() for item in self.no_op_differences])
        ):
            _fail("no-op difference commitment does not reproduce")
        if self.restriction_difference_sha256 != sha256_bytes(
            canonical_json_bytes(
                [item.to_mapping() for item in self.restriction_differences]
            )
        ):
            _fail("restriction difference commitment does not reproduce")
        exact_zero = all(item.difference == 0.0 for item in self.no_op_differences)
        nonzero_count = sum(
            item.difference != 0.0 for item in self.restriction_differences
        )
        branch_observed = (
            nonzero_count > 0 and self.restriction_internal_event_delta != 0
        )
        if self.no_op_exact_zero != exact_zero:
            _fail("no-op zero flag differs from the complete no-op registry")
        if self.restriction_nonzero_difference_count != nonzero_count:
            _fail("restriction nonzero count differs from the complete registry")
        if self.restriction_branch_observed != branch_observed:
            _fail("branch activation flag differs from the retained evidence")
        if not all(
            (
                self.no_op_exact_zero,
                self.baseline_invariant,
                self.event_keyed_shocks_aligned,
                self.replay_byte_identical,
                self.pair_integrity_passed,
            )
        ):
            _fail("an incomplete or failed source replicate cannot be reported")

    def to_mapping(self) -> dict[str, object]:
        return {
            "baseline_invariant": self.baseline_invariant,
            "baseline_run_sha256": self.baseline_run_sha256,
            "baseline_world_sha256": self.baseline_world_sha256,
            "child_path": self.child_path,
            "child_sha256": self.child_sha256,
            "event_key": self.event_key,
            "event_keyed_shocks_aligned": self.event_keyed_shocks_aligned,
            "no_op_differences": [item.to_mapping() for item in self.no_op_differences],
            "no_op_difference_sha256": self.no_op_difference_sha256,
            "no_op_exact_zero": self.no_op_exact_zero,
            "pair_integrity_passed": self.pair_integrity_passed,
            "replay_byte_identical": self.replay_byte_identical,
            "replicate_id": self.replicate_id,
            "restriction_branch_observed": self.restriction_branch_observed,
            "restriction_differences": [
                item.to_mapping() for item in self.restriction_differences
            ],
            "restriction_difference_sha256": self.restriction_difference_sha256,
            "restriction_internal_event_delta": (self.restriction_internal_event_delta),
            "restriction_nonzero_difference_count": (
                self.restriction_nonzero_difference_count
            ),
            "shock_tape_sha256": self.shock_tape_sha256,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(value, frozenset(cls.__annotations__), "replicate row")
        return cls(
            replicate_id=_integer(value["replicate_id"], "replicate_id"),
            event_key=_string(value["event_key"], "event_key"),
            child_path=_string(value["child_path"], "child_path"),
            child_sha256=_sha256(value["child_sha256"], "child_sha256"),
            shock_tape_sha256=_sha256(value["shock_tape_sha256"], "shock_tape_sha256"),
            baseline_world_sha256=_sha256(
                value["baseline_world_sha256"], "baseline_world_sha256"
            ),
            baseline_run_sha256=_sha256(
                value["baseline_run_sha256"], "baseline_run_sha256"
            ),
            no_op_difference_sha256=_sha256(
                value["no_op_difference_sha256"], "no_op_difference_sha256"
            ),
            restriction_difference_sha256=_sha256(
                value["restriction_difference_sha256"],
                "restriction_difference_sha256",
            ),
            restriction_internal_event_delta=_signed_integer(
                value["restriction_internal_event_delta"],
                "restriction_internal_event_delta",
            ),
            restriction_nonzero_difference_count=_integer(
                value["restriction_nonzero_difference_count"],
                "restriction_nonzero_difference_count",
            ),
            restriction_branch_observed=_boolean(
                value["restriction_branch_observed"],
                "restriction_branch_observed",
            ),
            no_op_exact_zero=_boolean(value["no_op_exact_zero"], "no_op_exact_zero"),
            baseline_invariant=_boolean(
                value["baseline_invariant"], "baseline_invariant"
            ),
            event_keyed_shocks_aligned=_boolean(
                value["event_keyed_shocks_aligned"],
                "event_keyed_shocks_aligned",
            ),
            replay_byte_identical=_boolean(
                value["replay_byte_identical"], "replay_byte_identical"
            ),
            pair_integrity_passed=_boolean(
                value["pair_integrity_passed"], "pair_integrity_passed"
            ),
            no_op_differences=tuple(
                IntegratedBranchDifference.from_mapping(
                    _mapping(item, "no-op difference")
                )
                for item in _sequence(value["no_op_differences"], "no_op_differences")
            ),
            restriction_differences=tuple(
                IntegratedBranchDifference.from_mapping(
                    _mapping(item, "restriction difference")
                )
                for item in _sequence(
                    value["restriction_differences"],
                    "restriction_differences",
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class IntegratedBranchDescriptiveQuantiles:
    """Type-7 quantiles with an explicit non-confidence interpretation."""

    q05: float
    q50: float
    q95: float
    method: Literal["r_type_7_linear_interpolation"] = "r_type_7_linear_interpolation"
    interpretation: str = _QUANTILE_INTERPRETATION

    def __post_init__(self) -> None:
        values = tuple(
            _finite(getattr(self, field), field) for field in ("q05", "q50", "q95")
        )
        if values != tuple(sorted(values)):
            _fail("descriptive quantiles must be ordered")
        if self.method != "r_type_7_linear_interpolation":
            _fail("unsupported quantile method")
        if self.interpretation != _QUANTILE_INTERPRETATION:
            _fail("quantiles cannot be labeled as confidence intervals")

    def to_mapping(self) -> dict[str, object]:
        return {
            "interpretation": self.interpretation,
            "method": self.method,
            "q05": self.q05,
            "q50": self.q50,
            "q95": self.q95,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(value, frozenset(cls.__annotations__), "quantiles")
        return cls(
            q05=_finite(value["q05"], "q05"),
            q50=_finite(value["q50"], "q50"),
            q95=_finite(value["q95"], "q95"),
            method=cast(
                Literal["r_type_7_linear_interpolation"],
                _string(value["method"], "method"),
            ),
            interpretation=_string(value["interpretation"], "interpretation"),
        )


@dataclass(frozen=True, slots=True)
class IntegratedBranchReportedScalar:
    """A finite scalar or an explicit zero-denominator unavailability."""

    status: Literal["defined", "undefined_zero_denominator"]
    value: float | None
    reason: str | None

    def __post_init__(self) -> None:
        if self.status == "defined":
            _finite(self.value, "reported scalar value")
            if self.reason is not None:
                _fail("a defined scalar cannot retain an unavailability reason")
            return
        if self.status == "undefined_zero_denominator":
            if self.value is not None:
                _fail("an unavailable scalar cannot use zero as a placeholder")
            if not isinstance(self.reason, str) or not self.reason:
                _fail("an unavailable scalar requires a reason")
            return
        _fail("unknown reported scalar status")

    @classmethod
    def defined(cls, value: float) -> Self:
        return cls(status="defined", value=value, reason=None)

    @classmethod
    def zero_denominator(cls, reason: str) -> Self:
        return cls(
            status="undefined_zero_denominator",
            value=None,
            reason=reason,
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "reason": self.reason,
            "status": self.status,
            "value": self.value,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(value, frozenset(cls.__annotations__), "reported scalar")
        raw_reason = value["reason"]
        if raw_reason is not None and not isinstance(raw_reason, str):
            _fail("reported scalar reason must be a string or null")
        raw_value = value["value"]
        return cls(
            status=cast(
                Literal["defined", "undefined_zero_denominator"],
                _string(value["status"], "reported scalar status"),
            ),
            value=None
            if raw_value is None
            else _finite(raw_value, "reported scalar value"),
            reason=raw_reason,
        )


@dataclass(frozen=True, slots=True)
class IntegratedBranchDifferenceSummary:
    """All restriction draws and frozen descriptive statistics for one cell."""

    horizon_id: str
    group_id: str
    metric_name: str
    replicate_ids: tuple[int, ...]
    draws: tuple[float, ...]
    mean: float
    sample_variance_ddof1: IntegratedBranchReportedScalar
    mcse: IntegratedBranchReportedScalar
    quantiles: IntegratedBranchDescriptiveQuantiles

    def __post_init__(self) -> None:
        for field in ("horizon_id", "group_id", "metric_name"):
            _identifier(getattr(self, field), field)
        if not self.replicate_ids:
            _fail("a summary requires at least one complete replicate")
        if self.replicate_ids != tuple(sorted(set(self.replicate_ids))):
            _fail("summary replicate identifiers must be unique and ordered")
        draws = tuple(_finite(value, "restriction draw") for value in self.draws)
        if len(draws) != len(self.replicate_ids):
            _fail("summary draws must cover every registered replicate")
        expected_mean = fmean(draws)
        if len(draws) < 2:
            reason = "ddof=1 sample variance requires at least two replicates"
            expected_variance = IntegratedBranchReportedScalar.zero_denominator(reason)
            expected_mcse = IntegratedBranchReportedScalar.zero_denominator(
                "MCSE is undefined because ddof=1 sample variance is unavailable"
            )
        else:
            variance = _sample_variance(draws)
            expected_variance = IntegratedBranchReportedScalar.defined(variance)
            expected_mcse = IntegratedBranchReportedScalar.defined(
                math.sqrt(variance / len(draws))
            )
        expected_quantiles = _descriptive_quantiles(draws)
        if (
            _finite(self.mean, "mean") != expected_mean
            or self.sample_variance_ddof1 != expected_variance
            or self.mcse != expected_mcse
            or self.quantiles != expected_quantiles
        ):
            _fail("summary statistics do not reproduce from their full draws")

    @property
    def cell_key(self) -> tuple[str, str, str]:
        return self.horizon_id, self.group_id, self.metric_name

    def to_mapping(self) -> dict[str, object]:
        return {
            "draws": list(self.draws),
            "group_id": self.group_id,
            "horizon_id": self.horizon_id,
            "mcse": self.mcse.to_mapping(),
            "mean": self.mean,
            "metric_name": self.metric_name,
            "quantiles": self.quantiles.to_mapping(),
            "replicate_ids": list(self.replicate_ids),
            "sample_variance_ddof1": self.sample_variance_ddof1.to_mapping(),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(value, frozenset(cls.__annotations__), "difference summary")
        return cls(
            horizon_id=_identifier(value["horizon_id"], "horizon_id"),
            group_id=_identifier(value["group_id"], "group_id"),
            metric_name=_identifier(value["metric_name"], "metric_name"),
            replicate_ids=tuple(
                _integer(item, "replicate_id")
                for item in _sequence(value["replicate_ids"], "replicate_ids")
            ),
            draws=tuple(
                _finite(item, "restriction draw")
                for item in _sequence(value["draws"], "draws")
            ),
            mean=_finite(value["mean"], "mean"),
            sample_variance_ddof1=IntegratedBranchReportedScalar.from_mapping(
                _mapping(value["sample_variance_ddof1"], "sample_variance_ddof1")
            ),
            mcse=IntegratedBranchReportedScalar.from_mapping(
                _mapping(value["mcse"], "mcse")
            ),
            quantiles=IntegratedBranchDescriptiveQuantiles.from_mapping(
                _mapping(value["quantiles"], "quantiles")
            ),
        )


def _descriptive_quantiles(
    draws: Sequence[float],
) -> IntegratedBranchDescriptiveQuantiles:
    return IntegratedBranchDescriptiveQuantiles(
        q05=_type7_quantile(draws, 0.05),
        q50=_type7_quantile(draws, 0.5),
        q95=_type7_quantile(draws, 0.95),
    )


@dataclass(frozen=True, slots=True)
class IntegratedBranchInvariantRates:
    """Collection rates for the non-bypassable paired-world checks."""

    no_op_exact_zero_rate: float
    baseline_invariance_rate: float
    event_keyed_shock_alignment_rate: float
    replay_byte_identity_rate: float

    def __post_init__(self) -> None:
        for field in self.__annotations__:
            _fraction(getattr(self, field), field)

    def to_mapping(self) -> dict[str, object]:
        return {
            "baseline_invariance_rate": self.baseline_invariance_rate,
            "event_keyed_shock_alignment_rate": (self.event_keyed_shock_alignment_rate),
            "no_op_exact_zero_rate": self.no_op_exact_zero_rate,
            "replay_byte_identity_rate": self.replay_byte_identity_rate,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(value, frozenset(cls.__annotations__), "invariant rates")
        return cls(
            no_op_exact_zero_rate=_fraction(
                value["no_op_exact_zero_rate"], "no_op_exact_zero_rate"
            ),
            baseline_invariance_rate=_fraction(
                value["baseline_invariance_rate"], "baseline_invariance_rate"
            ),
            event_keyed_shock_alignment_rate=_fraction(
                value["event_keyed_shock_alignment_rate"],
                "event_keyed_shock_alignment_rate",
            ),
            replay_byte_identity_rate=_fraction(
                value["replay_byte_identity_rate"],
                "replay_byte_identity_rate",
            ),
        )


@dataclass(frozen=True, slots=True)
class IntegratedBranchSourceCommitments:
    """External source commitments accepted by the replay audit."""

    benchmark_id: str
    plan_sha256: str
    artifact_sha256: str
    config_sha256: str
    replicate_count: int

    def __post_init__(self) -> None:
        _identifier(self.benchmark_id, "benchmark_id")
        for field in ("plan_sha256", "artifact_sha256", "config_sha256"):
            _sha256(getattr(self, field), field)
        _integer(self.replicate_count, "replicate_count", minimum=1)

    def to_mapping(self) -> dict[str, object]:
        return {
            "artifact_sha256": self.artifact_sha256,
            "benchmark_id": self.benchmark_id,
            "config_sha256": self.config_sha256,
            "plan_sha256": self.plan_sha256,
            "replicate_count": self.replicate_count,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(value, frozenset(cls.__annotations__), "source commitments")
        return cls(
            benchmark_id=_identifier(value["benchmark_id"], "benchmark_id"),
            plan_sha256=_sha256(value["plan_sha256"], "plan_sha256"),
            artifact_sha256=_sha256(value["artifact_sha256"], "artifact_sha256"),
            config_sha256=_sha256(value["config_sha256"], "config_sha256"),
            replicate_count=_integer(
                value["replicate_count"], "replicate_count", minimum=1
            ),
        )


@dataclass(frozen=True, slots=True)
class AAMASIntegratedBranchReportArtifact:
    """Immutable, synthetic-only report derived from every audited child."""

    reporting_spec: FrozenIntegratedBranchReportingSpec
    reporting_spec_sha256: str
    sources: IntegratedBranchSourceCommitments
    replicate_rows: tuple[IntegratedBranchReplicateRow, ...]
    branch_activation_count: int
    branch_activation_rate: float
    invariant_rates: IntegratedBranchInvariantRates
    restriction_summaries: tuple[IntegratedBranchDifferenceSummary, ...]
    formal_source_scale: bool
    integrity_valid: Literal[True] = True
    source_complete: Literal[True] = True
    synthetic_only: Literal[True] = True
    publication_eligible: Literal[False] = False
    quantiles_are_confidence_intervals: Literal[False] = False
    claim_scope: str = AAMAS_INTEGRATED_BRANCH_REPORT_CLAIM_SCOPE
    schema_version: str = AAMAS_INTEGRATED_BRANCH_REPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.reporting_spec, FrozenIntegratedBranchReportingSpec):
            _fail("reporting_spec must be frozen and typed")
        if self.reporting_spec_sha256 != self.reporting_spec.sha256:
            _fail("reporting spec digest does not reproduce")
        if not isinstance(self.sources, IntegratedBranchSourceCommitments):
            _fail("sources must use the typed commitment schema")
        if tuple(row.replicate_id for row in self.replicate_rows) != tuple(
            range(self.sources.replicate_count)
        ):
            _fail("replicate rows must be complete and canonically ordered")
        expected_branch_count = sum(
            row.restriction_branch_observed for row in self.replicate_rows
        )
        if self.branch_activation_count != expected_branch_count:
            _fail("branch activation count differs from complete replicate rows")
        if self.branch_activation_rate != _rate(
            expected_branch_count, self.sources.replicate_count
        ):
            _fail("branch activation rate differs from complete replicate rows")
        expected_rates = _invariant_rates(self.replicate_rows)
        if self.invariant_rates != expected_rates:
            _fail("invariant rates differ from complete replicate rows")
        expected_summaries = _summaries(self.replicate_rows)
        if self.restriction_summaries != expected_summaries:
            _fail("restriction summaries differ from complete replicate rows")
        if not isinstance(self.formal_source_scale, bool):
            _fail("formal_source_scale must be boolean")
        expected_flags: tuple[tuple[object, object], ...] = (
            (self.integrity_valid, True),
            (self.source_complete, True),
            (self.synthetic_only, True),
            (self.publication_eligible, False),
            (self.quantiles_are_confidence_intervals, False),
        )
        if any(observed is not expected for observed, expected in expected_flags):
            _fail("report claim flags cannot be broadened")
        if self.claim_scope != AAMAS_INTEGRATED_BRANCH_REPORT_CLAIM_SCOPE:
            _fail("report claim scope is invalid")
        if self.schema_version != AAMAS_INTEGRATED_BRANCH_REPORT_SCHEMA_VERSION:
            _fail("unsupported integrated branch report schema")

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_mapping())

    @property
    def sha256(self) -> str:
        return sha256_bytes(self.canonical_bytes)

    def to_mapping(self) -> dict[str, object]:
        return {
            "branch_activation_count": self.branch_activation_count,
            "branch_activation_rate": self.branch_activation_rate,
            "claim_scope": self.claim_scope,
            "formal_source_scale": self.formal_source_scale,
            "integrity_valid": self.integrity_valid,
            "invariant_rates": self.invariant_rates.to_mapping(),
            "publication_eligible": self.publication_eligible,
            "quantiles_are_confidence_intervals": (
                self.quantiles_are_confidence_intervals
            ),
            "replicate_rows": [item.to_mapping() for item in self.replicate_rows],
            "reporting_spec": self.reporting_spec.to_mapping(),
            "reporting_spec_sha256": self.reporting_spec_sha256,
            "restriction_summaries": [
                item.to_mapping() for item in self.restriction_summaries
            ],
            "schema_version": self.schema_version,
            "source_complete": self.source_complete,
            "sources": self.sources.to_mapping(),
            "synthetic_only": self.synthetic_only,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(value, frozenset(cls.__annotations__), "report artifact")
        return cls(
            reporting_spec=FrozenIntegratedBranchReportingSpec.from_mapping(
                _mapping(value["reporting_spec"], "reporting_spec")
            ),
            reporting_spec_sha256=_sha256(
                value["reporting_spec_sha256"], "reporting_spec_sha256"
            ),
            sources=IntegratedBranchSourceCommitments.from_mapping(
                _mapping(value["sources"], "sources")
            ),
            replicate_rows=tuple(
                IntegratedBranchReplicateRow.from_mapping(
                    _mapping(item, "replicate row")
                )
                for item in _sequence(value["replicate_rows"], "replicate_rows")
            ),
            branch_activation_count=_integer(
                value["branch_activation_count"], "branch_activation_count"
            ),
            branch_activation_rate=_fraction(
                value["branch_activation_rate"], "branch_activation_rate"
            ),
            invariant_rates=IntegratedBranchInvariantRates.from_mapping(
                _mapping(value["invariant_rates"], "invariant_rates")
            ),
            restriction_summaries=tuple(
                IntegratedBranchDifferenceSummary.from_mapping(
                    _mapping(item, "restriction summary")
                )
                for item in _sequence(
                    value["restriction_summaries"], "restriction_summaries"
                )
            ),
            formal_source_scale=_boolean(
                value["formal_source_scale"], "formal_source_scale"
            ),
            integrity_valid=cast(Literal[True], value["integrity_valid"]),
            source_complete=cast(Literal[True], value["source_complete"]),
            synthetic_only=cast(Literal[True], value["synthetic_only"]),
            publication_eligible=cast(Literal[False], value["publication_eligible"]),
            quantiles_are_confidence_intervals=cast(
                Literal[False], value["quantiles_are_confidence_intervals"]
            ),
            claim_scope=_string(value["claim_scope"], "claim_scope"),
            schema_version=_string(value["schema_version"], "schema_version"),
        )


def _invariant_rates(
    rows: Sequence[IntegratedBranchReplicateRow],
) -> IntegratedBranchInvariantRates:
    count = len(rows)
    return IntegratedBranchInvariantRates(
        no_op_exact_zero_rate=_rate(sum(row.no_op_exact_zero for row in rows), count),
        baseline_invariance_rate=_rate(
            sum(row.baseline_invariant for row in rows), count
        ),
        event_keyed_shock_alignment_rate=_rate(
            sum(row.event_keyed_shocks_aligned for row in rows), count
        ),
        replay_byte_identity_rate=_rate(
            sum(row.replay_byte_identical for row in rows), count
        ),
    )


def _summaries(
    rows: Sequence[IntegratedBranchReplicateRow],
) -> tuple[IntegratedBranchDifferenceSummary, ...]:
    if not rows:
        _integrity_fail("integrated branch reporting requires complete replicates")
    cell_draws: defaultdict[tuple[str, str, str], list[tuple[int, float]]]
    cell_draws = defaultdict(list)
    expected_cells = tuple(item.cell_key for item in rows[0].restriction_differences)
    for row in rows:
        observed_cells = tuple(item.cell_key for item in row.restriction_differences)
        if observed_cells != expected_cells:
            _integrity_fail(
                "a replicate has missing, extra, or reordered restriction cells"
            )
        for item in row.restriction_differences:
            cell_draws[item.cell_key].append((row.replicate_id, item.difference))
    summaries: list[IntegratedBranchDifferenceSummary] = []
    for horizon_id, group_id, metric_name in expected_cells:
        source = cell_draws[(horizon_id, group_id, metric_name)]
        replicate_ids = tuple(replicate_id for replicate_id, _ in source)
        draws = tuple(value for _, value in source)
        if len(draws) < 2:
            variance = IntegratedBranchReportedScalar.zero_denominator(
                "ddof=1 sample variance requires at least two replicates"
            )
            mcse = IntegratedBranchReportedScalar.zero_denominator(
                "MCSE is undefined because ddof=1 sample variance is unavailable"
            )
        else:
            variance_value = _sample_variance(draws)
            variance = IntegratedBranchReportedScalar.defined(variance_value)
            mcse = IntegratedBranchReportedScalar.defined(
                math.sqrt(variance_value / len(draws))
            )
        summaries.append(
            IntegratedBranchDifferenceSummary(
                horizon_id=horizon_id,
                group_id=group_id,
                metric_name=metric_name,
                replicate_ids=replicate_ids,
                draws=draws,
                mean=fmean(draws),
                sample_variance_ddof1=variance,
                mcse=mcse,
                quantiles=_descriptive_quantiles(draws),
            )
        )
    return tuple(summaries)


_SOURCE_ARTIFACT_FIELDS = frozenset(
    {
        "benchmark_id",
        "branch_replicate_count",
        "claim_scope",
        "config_sha256",
        "integrity_valid",
        "minimum_branch_replicates",
        "plan_sha256",
        "publication_eligible",
        "records",
        "schema_version",
        "scientifically_complete_for_claim",
        "synthetic_only",
    }
)
_SOURCE_RECORD_FIELDS = frozenset(
    {
        "baseline_invariant",
        "baseline_run_sha256",
        "baseline_world_sha256",
        "child_path",
        "child_sha256",
        "event_key",
        "event_keyed_shocks_aligned",
        "no_op_difference_sha256",
        "no_op_exact_zero",
        "pair_integrity_passed",
        "replay_byte_identical",
        "replicate_id",
        "restriction_branch_observed",
        "restriction_difference_sha256",
        "restriction_internal_event_delta",
        "restriction_nonzero_difference_count",
        "shock_tape_sha256",
    }
)
_SOURCE_CHILD_FIELDS = frozenset(
    {
        "audit",
        "claim_scope",
        "event_key",
        "no_op",
        "plan_sha256",
        "replicate_id",
        "restriction",
        "schema_version",
        "synthetic_only",
    }
)
_SOURCE_AUDIT_FIELDS = frozenset(
    {
        "baseline_invariant",
        "event_keyed_shocks_aligned",
        "no_op_exact_zero",
        "pair_integrity_passed",
        "replay_byte_identical",
        "restriction_branch_observed",
    }
)
_SOURCE_MODE_FIELDS = frozenset({"baseline_run", "candidate_run", "paired_outcome"})


def _read_canonical_mapping(path: Path, field: str) -> Mapping[str, object]:
    try:
        content = path.read_bytes()
        value = read_json_bytes_object(content, source=str(path))
    except (OSError, MetadataValidationError) as error:
        raise AAMASIntegratedBranchReportingIntegrityError(
            f"cannot read canonical {field}"
        ) from error
    if canonical_json_bytes(value) != content:
        _integrity_fail(f"{field} is not canonical JSON")
    return value


def _parse_differences(
    value: object, field: str
) -> tuple[IntegratedBranchDifference, ...]:
    return tuple(
        IntegratedBranchDifference.from_mapping(_mapping(item, field))
        for item in _sequence(value, field)
    )


def _shock_signature(
    run: Mapping[str, object], field: str
) -> tuple[tuple[str, str], ...]:
    steps = _sequence(run.get("steps"), f"{field}.steps")
    signature: list[tuple[str, str]] = []
    for index, raw_step in enumerate(steps):
        step = _mapping(raw_step, f"{field}.steps[{index}]")
        signature.append(
            (
                _string(step.get("shock_event_key"), "shock_event_key"),
                _sha256(
                    step.get("exogenous_command_sha256"),
                    "exogenous_command_sha256",
                ),
            )
        )
    return tuple(signature)


def _mode(value: object, field: str) -> Mapping[str, object]:
    mode = _mapping(value, field)
    _strict_fields(mode, _SOURCE_MODE_FIELDS, field)
    return mode


def _pair_outcome(mode: Mapping[str, object], field: str) -> Mapping[str, object]:
    return _mapping(mode["paired_outcome"], f"{field}.paired_outcome")


def _parse_replicate(
    source_dir: Path,
    record: Mapping[str, object],
    *,
    expected_replicate_id: int,
    expected_plan_sha256: str,
) -> IntegratedBranchReplicateRow:
    _strict_fields(record, _SOURCE_RECORD_FIELDS, "source record")
    replicate_id = _integer(record["replicate_id"], "replicate_id")
    if replicate_id != expected_replicate_id:
        _integrity_fail("source record registry is incomplete or reordered")
    child_path = _string(record["child_path"], "child_path")
    expected_path = f"runs/{replicate_id:06d}.json"
    if child_path != expected_path:
        _integrity_fail("source child path is not canonical")
    path = source_dir / PurePosixPath(child_path)
    child = _read_canonical_mapping(path, child_path)
    _strict_fields(child, _SOURCE_CHILD_FIELDS, "source child")
    child_sha256 = _sha256(record["child_sha256"], "child_sha256")
    if sha256_file(path) != child_sha256:
        _integrity_fail("source child digest differs from its registry")
    if (
        child["schema_version"] != AAMAS_INTEGRATED_BRANCH_CHILD_SCHEMA_VERSION
        or child["plan_sha256"] != expected_plan_sha256
        or child["replicate_id"] != replicate_id
        or child["event_key"] != record["event_key"]
        or child["synthetic_only"] is not True
    ):
        _integrity_fail("source child identity differs from its registry or plan")

    audit = _mapping(child["audit"], "source child audit")
    _strict_fields(audit, _SOURCE_AUDIT_FIELDS, "source child audit")
    no_op = _mode(child["no_op"], "no_op")
    restriction = _mode(child["restriction"], "restriction")
    no_op_pair = _pair_outcome(no_op, "no_op")
    restriction_pair = _pair_outcome(restriction, "restriction")
    no_op_differences = _parse_differences(
        no_op_pair.get("differences"), "no-op differences"
    )
    restriction_differences = _parse_differences(
        restriction_pair.get("differences"), "restriction differences"
    )
    no_op_difference_sha256 = sha256_bytes(
        canonical_json_bytes([item.to_mapping() for item in no_op_differences])
    )
    restriction_difference_sha256 = sha256_bytes(
        canonical_json_bytes([item.to_mapping() for item in restriction_differences])
    )
    if no_op_difference_sha256 != _sha256(
        record["no_op_difference_sha256"], "no_op_difference_sha256"
    ) or restriction_difference_sha256 != _sha256(
        record["restriction_difference_sha256"],
        "restriction_difference_sha256",
    ):
        _integrity_fail("source difference registry digest does not reproduce")

    no_op_baseline = _mapping(no_op_pair.get("baseline"), "no-op baseline")
    restriction_baseline = _mapping(
        restriction_pair.get("baseline"), "restriction baseline"
    )
    no_op_baseline_run = _mapping(no_op["baseline_run"], "no-op baseline run")
    restriction_baseline_run = _mapping(
        restriction["baseline_run"], "restriction baseline run"
    )
    baseline_invariant = canonical_json_bytes(no_op_baseline) == canonical_json_bytes(
        restriction_baseline
    ) and canonical_json_bytes(no_op_baseline_run) == canonical_json_bytes(
        restriction_baseline_run
    )
    if sha256_bytes(canonical_json_bytes(no_op_baseline)) != _sha256(
        record["baseline_world_sha256"], "baseline_world_sha256"
    ) or sha256_bytes(canonical_json_bytes(no_op_baseline_run)) != _sha256(
        record["baseline_run_sha256"], "baseline_run_sha256"
    ):
        _integrity_fail("source baseline commitment does not reproduce")

    modes = (no_op, restriction)
    signatures = tuple(
        _shock_signature(
            _mapping(mode[run_name], f"mode {run_name}"), f"mode {run_name}"
        )
        for mode in modes
        for run_name in ("baseline_run", "candidate_run")
    )
    event_keyed_shocks_aligned = all(
        signature == signatures[0] for signature in signatures[1:]
    )
    no_op_exact_zero = all(item.difference == 0.0 for item in no_op_differences)
    nonzero_count = sum(item.difference != 0.0 for item in restriction_differences)
    restriction_baseline_outcome = _mapping(
        restriction_pair.get("baseline"), "restriction baseline outcome"
    )
    restriction_candidate_outcome = _mapping(
        restriction_pair.get("candidate"), "restriction candidate outcome"
    )
    internal_delta = _integer(
        restriction_candidate_outcome.get("internal_event_count"),
        "candidate internal_event_count",
    ) - _integer(
        restriction_baseline_outcome.get("internal_event_count"),
        "baseline internal_event_count",
    )
    branch_observed = nonzero_count > 0 and internal_delta != 0
    source_flags = {
        "baseline_invariant": baseline_invariant,
        "event_keyed_shocks_aligned": event_keyed_shocks_aligned,
        "no_op_exact_zero": no_op_exact_zero,
        "restriction_branch_observed": branch_observed,
    }
    for name, derived in source_flags.items():
        if _boolean(audit[name], f"audit.{name}") != derived:
            _integrity_fail(f"source audit flag does not reproduce: {name}")
        if _boolean(record[name], f"record.{name}") != derived:
            _integrity_fail(f"source record flag does not reproduce: {name}")
    if (
        _signed_integer(
            record["restriction_internal_event_delta"],
            "restriction_internal_event_delta",
        )
        != internal_delta
        or _integer(
            record["restriction_nonzero_difference_count"],
            "restriction_nonzero_difference_count",
        )
        != nonzero_count
    ):
        _integrity_fail("source branch counts do not reproduce")
    for name in ("pair_integrity_passed", "replay_byte_identical"):
        if _boolean(audit[name], f"audit.{name}") is not True:
            _integrity_fail(f"source replicate failed required audit: {name}")
        if _boolean(record[name], f"record.{name}") is not True:
            _integrity_fail(f"source record failed required audit: {name}")
    shock_tape_sha256 = _sha256(
        no_op_pair.get("shock_tape_sha256"), "shock_tape_sha256"
    )
    if shock_tape_sha256 != _sha256(
        record["shock_tape_sha256"], "record shock_tape_sha256"
    ):
        _integrity_fail("source shock-tape commitment does not reproduce")
    return IntegratedBranchReplicateRow(
        replicate_id=replicate_id,
        event_key=_string(record["event_key"], "event_key"),
        child_path=child_path,
        child_sha256=child_sha256,
        shock_tape_sha256=shock_tape_sha256,
        baseline_world_sha256=_sha256(
            record["baseline_world_sha256"], "baseline_world_sha256"
        ),
        baseline_run_sha256=_sha256(
            record["baseline_run_sha256"], "baseline_run_sha256"
        ),
        no_op_difference_sha256=no_op_difference_sha256,
        restriction_difference_sha256=restriction_difference_sha256,
        restriction_internal_event_delta=internal_delta,
        restriction_nonzero_difference_count=nonzero_count,
        restriction_branch_observed=branch_observed,
        no_op_exact_zero=no_op_exact_zero,
        baseline_invariant=baseline_invariant,
        event_keyed_shocks_aligned=event_keyed_shocks_aligned,
        replay_byte_identical=True,
        pair_integrity_passed=True,
        no_op_differences=no_op_differences,
        restriction_differences=restriction_differences,
    )


def derive_aamas_integrated_branch_report(
    *,
    source_dir: Path,
    expected_plan_sha256: str,
    expected_artifact_sha256: str,
    reporting_spec: FrozenIntegratedBranchReportingSpec | None = None,
    allow_formal_scale: bool = False,
) -> AAMASIntegratedBranchReportArtifact:
    """Re-audit a private source snapshot, then derive finite-sample descriptions.

    The first replay audit identifies the caller-visible tree.  Reporting then
    copies that tree into a private directory, re-audits the copy against the
    same caller-held commitments, derives exclusively from the copied bytes,
    and re-audits once more after all reads.  This prevents an audit-then-read
    race from associating an old receipt with changed source values.
    """

    if not isinstance(source_dir, Path):
        raise TypeError("source_dir must be a Path")
    if not isinstance(allow_formal_scale, bool):
        raise TypeError("allow_formal_scale must be boolean")
    spec = reporting_spec or FrozenIntegratedBranchReportingSpec()
    if not isinstance(spec, FrozenIntegratedBranchReportingSpec):
        raise TypeError("reporting_spec must be frozen and typed")
    plan_sha256 = _sha256(expected_plan_sha256, "expected_plan_sha256")
    artifact_sha256 = _sha256(expected_artifact_sha256, "expected_artifact_sha256")

    def audit_source(directory: Path) -> AAMASIntegratedBranchAudit:
        try:
            return audit_written_aamas_integrated_branch(
                directory,
                expected_plan_sha256=plan_sha256,
                expected_artifact_sha256=artifact_sha256,
                allow_formal_scale=allow_formal_scale,
            )
        except (AAMASIntegratedBranchError, OSError, MetadataValidationError) as error:
            raise AAMASIntegratedBranchReportingIntegrityError(
                "integrated branch source replay audit failed"
            ) from error

    source_audit = audit_source(source_dir)
    with tempfile.TemporaryDirectory(
        prefix="policytwin-aamas-integrated-branch-reporting-"
    ) as raw:
        snapshot = Path(raw) / "integrated-branch"
        try:
            shutil.copytree(source_dir, snapshot, symlinks=True)
        except OSError as error:
            raise AAMASIntegratedBranchReportingIntegrityError(
                "cannot create private integrated branch source snapshot"
            ) from error
        snapshot_audit_before = audit_source(snapshot)
        if snapshot_audit_before != source_audit:
            _integrity_fail(
                "integrated branch source identity changed while creating the "
                "private snapshot"
            )

        plan_mapping = _read_canonical_mapping(snapshot / "plan.json", "source plan")
        try:
            plan = AAMASIntegratedBranchPlan.from_mapping(plan_mapping)
        except AAMASIntegratedBranchError as error:
            raise AAMASIntegratedBranchReportingIntegrityError(
                "integrated branch source plan failed typed validation"
            ) from error
        if plan.sha256 != plan_sha256:
            _integrity_fail("source plan differs from the caller-held plan commitment")
        source_mapping = _read_canonical_mapping(
            snapshot / "artifact.json", "source root artifact"
        )
        _strict_fields(source_mapping, _SOURCE_ARTIFACT_FIELDS, "source root artifact")
        if (
            source_mapping["schema_version"]
            != AAMAS_INTEGRATED_BRANCH_ARTIFACT_SCHEMA_VERSION
            or source_mapping["plan_sha256"] != plan_sha256
            or sha256_bytes(canonical_json_bytes(source_mapping)) != artifact_sha256
            or snapshot_audit_before.plan_sha256 != plan_sha256
            or snapshot_audit_before.artifact_sha256 != artifact_sha256
            or source_mapping["integrity_valid"] is not True
            or source_mapping["scientifically_complete_for_claim"] is not True
            or source_mapping["synthetic_only"] is not True
            or source_mapping["publication_eligible"] is not False
        ):
            _integrity_fail("source root identity or claim flags are invalid")
        raw_records = _sequence(source_mapping["records"], "source records")
        if (
            len(raw_records) != plan.repetitions
            or len(raw_records) != snapshot_audit_before.record_count
        ):
            _integrity_fail("source record registry is incomplete")
        rows = tuple(
            _parse_replicate(
                snapshot,
                _mapping(raw_record, "source record"),
                expected_replicate_id=index,
                expected_plan_sha256=plan_sha256,
            )
            for index, raw_record in enumerate(raw_records)
        )
        branch_count = sum(row.restriction_branch_observed for row in rows)
        if branch_count != _integer(
            source_mapping["branch_replicate_count"], "branch_replicate_count"
        ):
            _integrity_fail("source branch count differs from complete child rows")
        sources = IntegratedBranchSourceCommitments(
            benchmark_id=_identifier(source_mapping["benchmark_id"], "benchmark_id"),
            plan_sha256=plan_sha256,
            artifact_sha256=artifact_sha256,
            config_sha256=_sha256(source_mapping["config_sha256"], "config_sha256"),
            replicate_count=len(rows),
        )
        artifact = AAMASIntegratedBranchReportArtifact(
            reporting_spec=spec,
            reporting_spec_sha256=spec.sha256,
            sources=sources,
            replicate_rows=rows,
            branch_activation_count=branch_count,
            branch_activation_rate=_rate(branch_count, len(rows)),
            invariant_rates=_invariant_rates(rows),
            restriction_summaries=_summaries(rows),
            formal_source_scale=plan.is_formal_scale,
        )
        snapshot_audit_after = audit_source(snapshot)
        if snapshot_audit_after != snapshot_audit_before:
            _integrity_fail(
                "private integrated branch source snapshot changed after reads"
            )
        return artifact


def _expected_report_members() -> frozenset[str]:
    return frozenset(
        {
            AAMAS_INTEGRATED_BRANCH_REPORT_FILENAME,
            AAMAS_INTEGRATED_BRANCH_REPORT_SEAL_FILENAME,
        }
    )


def _audit_report_membership(directory: Path) -> None:
    if not directory.is_dir() or directory.is_symlink():
        _integrity_fail("integrated branch report root must be a real directory")
    observed: set[str] = set()
    for path in directory.rglob("*"):
        relative = path.relative_to(directory).as_posix()
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            _integrity_fail("integrated branch report must not contain symlinks")
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            _integrity_fail(
                "integrated branch report members must be singly linked regular files"
            )
        observed.add(relative)
    if observed != set(_expected_report_members()):
        _integrity_fail("integrated branch report membership changed")


def write_aamas_integrated_branch_report(
    output_dir: Path,
    artifact: AAMASIntegratedBranchReportArtifact,
) -> str:
    """Stage and atomically publish one exact-byte sealed reporting artifact."""

    if not isinstance(output_dir, Path):
        raise TypeError("output_dir must be a Path")
    if not isinstance(artifact, AAMASIntegratedBranchReportArtifact):
        raise TypeError("artifact must be an AAMASIntegratedBranchReportArtifact")
    if output_dir.exists() or output_dir.is_symlink():
        _integrity_fail("integrated branch report output already exists")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.with_name(f".{output_dir.name}.{artifact.sha256[:16]}.staging")
    if staging.exists() or staging.is_symlink():
        _integrity_fail("integrated branch report staging path already exists")
    try:
        staging.mkdir()
        report_path = staging / AAMAS_INTEGRATED_BRANCH_REPORT_FILENAME
        seal_path = staging / AAMAS_INTEGRATED_BRANCH_REPORT_SEAL_FILENAME
        write_text_file(report_path, artifact.canonical_bytes.decode("utf-8"))
        write_text_file(seal_path, artifact.sha256 + "\n")
        _audit_report_membership(staging)
        if sha256_file(report_path) != artifact.sha256:
            _integrity_fail("written report differs from its exact-byte digest")
        os.replace(staging, output_dir)
        _audit_report_membership(output_dir)
        return artifact.sha256
    except Exception:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise


@dataclass(frozen=True, slots=True)
class AAMASIntegratedBranchReportAudit:
    """Positive receipt for source replay and exact report re-derivation."""

    report_sha256: str
    source_plan_sha256: str
    source_artifact_sha256: str
    replicate_count: int
    summary_count: int
    passed: Literal[True] = True

    def __post_init__(self) -> None:
        for field in (
            "report_sha256",
            "source_plan_sha256",
            "source_artifact_sha256",
        ):
            _sha256(getattr(self, field), field)
        _integer(self.replicate_count, "replicate_count", minimum=1)
        _integer(self.summary_count, "summary_count", minimum=1)
        if self.passed is not True:
            _fail("only a positive report audit receipt may be constructed")


def audit_written_aamas_integrated_branch_report(
    *,
    output_dir: Path,
    source_dir: Path,
    expected_report_sha256: str,
    expected_plan_sha256: str,
    expected_artifact_sha256: str,
    expected_reporting_spec: FrozenIntegratedBranchReportingSpec,
    allow_formal_scale: bool = False,
) -> AAMASIntegratedBranchReportAudit:
    """Re-audit source, re-derive report, and verify fixed exact bytes."""

    if not isinstance(output_dir, Path) or not isinstance(source_dir, Path):
        raise TypeError("output_dir and source_dir must be Path values")
    if not isinstance(expected_reporting_spec, FrozenIntegratedBranchReportingSpec):
        raise TypeError("expected_reporting_spec must be frozen and typed")
    if not isinstance(allow_formal_scale, bool):
        raise TypeError("allow_formal_scale must be boolean")
    expected_report = _sha256(expected_report_sha256, "expected_report_sha256")
    _audit_report_membership(output_dir)
    live_report_path = output_dir / AAMAS_INTEGRATED_BRANCH_REPORT_FILENAME
    live_seal_path = output_dir / AAMAS_INTEGRATED_BRANCH_REPORT_SEAL_FILENAME
    live_report_bytes = _stable_regular_bytes(live_report_path, "written report")
    live_seal_bytes = _stable_regular_bytes(live_seal_path, "written report seal")

    with tempfile.TemporaryDirectory(
        prefix="policytwin-aamas-integrated-branch-report-audit-"
    ) as raw:
        snapshot = Path(raw) / "report"
        try:
            shutil.copytree(output_dir, snapshot, symlinks=True)
        except OSError as error:
            raise AAMASIntegratedBranchReportingIntegrityError(
                "cannot create private integrated branch report snapshot"
            ) from error
        _audit_report_membership(snapshot)
        report_path = snapshot / AAMAS_INTEGRATED_BRANCH_REPORT_FILENAME
        seal_path = snapshot / AAMAS_INTEGRATED_BRANCH_REPORT_SEAL_FILENAME
        report_bytes = _stable_regular_bytes(report_path, "snapshot report")
        seal_bytes = _stable_regular_bytes(seal_path, "snapshot report seal")
        if report_bytes != live_report_bytes or seal_bytes != live_seal_bytes:
            _integrity_fail(
                "integrated branch report changed while creating private snapshot"
            )
        try:
            stored_mapping = read_json_bytes_object(
                report_bytes, source=str(report_path)
            )
            seal = seal_bytes.decode("ascii")
        except (UnicodeError, MetadataValidationError) as error:
            raise AAMASIntegratedBranchReportingIntegrityError(
                "cannot parse written integrated branch report"
            ) from error
        if canonical_json_bytes(stored_mapping) != report_bytes:
            _integrity_fail("written integrated branch report is not canonical JSON")
        try:
            stored = AAMASIntegratedBranchReportArtifact.from_mapping(stored_mapping)
        except AAMASIntegratedBranchReportingSchemaError as error:
            raise AAMASIntegratedBranchReportingIntegrityError(
                "written integrated branch report failed typed validation"
            ) from error
        if (
            seal != expected_report + "\n"
            or sha256_bytes(report_bytes) != expected_report
        ):
            _integrity_fail("integrated branch report exact-byte seal is invalid")
        if stored.sha256 != expected_report:
            _integrity_fail("integrated branch report differs from external commitment")
        if stored.reporting_spec != expected_reporting_spec:
            _integrity_fail("reporting spec differs from its external commitment")
        replay = derive_aamas_integrated_branch_report(
            source_dir=source_dir,
            expected_plan_sha256=expected_plan_sha256,
            expected_artifact_sha256=expected_artifact_sha256,
            reporting_spec=expected_reporting_spec,
            allow_formal_scale=allow_formal_scale,
        )
        if stored.canonical_bytes != replay.canonical_bytes:
            _integrity_fail("written report differs from audited source re-derivation")
        _audit_report_membership(snapshot)
        if (
            _stable_regular_bytes(report_path, "snapshot report") != report_bytes
            or _stable_regular_bytes(seal_path, "snapshot report seal") != seal_bytes
        ):
            _integrity_fail("private integrated branch report snapshot changed")

        _audit_report_membership(output_dir)
        if (
            _stable_regular_bytes(live_report_path, "written report")
            != live_report_bytes
            or _stable_regular_bytes(live_seal_path, "written report seal")
            != live_seal_bytes
        ):
            _integrity_fail("integrated branch report changed during source replay")
        return AAMASIntegratedBranchReportAudit(
            report_sha256=stored.sha256,
            source_plan_sha256=stored.sources.plan_sha256,
            source_artifact_sha256=stored.sources.artifact_sha256,
            replicate_count=len(stored.replicate_rows),
            summary_count=len(stored.restriction_summaries),
        )


__all__ = [
    "AAMAS_INTEGRATED_BRANCH_REPORTING_SPEC_SCHEMA_VERSION",
    "AAMAS_INTEGRATED_BRANCH_REPORT_CLAIM_SCOPE",
    "AAMAS_INTEGRATED_BRANCH_REPORT_FILENAME",
    "AAMAS_INTEGRATED_BRANCH_REPORT_SCHEMA_VERSION",
    "AAMAS_INTEGRATED_BRANCH_REPORT_SEAL_FILENAME",
    "AAMASIntegratedBranchReportArtifact",
    "AAMASIntegratedBranchReportAudit",
    "AAMASIntegratedBranchReportingError",
    "AAMASIntegratedBranchReportingIntegrityError",
    "AAMASIntegratedBranchReportingSchemaError",
    "FrozenIntegratedBranchReportingSpec",
    "IntegratedBranchDescriptiveQuantiles",
    "IntegratedBranchDifference",
    "IntegratedBranchDifferenceSummary",
    "IntegratedBranchInvariantRates",
    "IntegratedBranchReplicateRow",
    "IntegratedBranchReportedScalar",
    "IntegratedBranchSourceCommitments",
    "audit_written_aamas_integrated_branch_report",
    "derive_aamas_integrated_branch_report",
    "write_aamas_integrated_branch_report",
]
