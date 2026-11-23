"""Auditable synthetic CLOB branch benchmark for paired policy worlds.

This benchmark is intentionally narrow.  It establishes that a non-no-op
policy can change an endogenous order/fill branch in the integrated synthetic
market without shifting later exogenous shocks.  A second, otherwise identical
no-op pair is required to have exactly zero paired differences.  The artifact
does not support an empirical SEC effect, welfare, or real-world causal claim.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Literal, NoReturn, Self, cast

from policytwin.counterfactual import EventKey
from policytwin.counterfactual.integrated_market import (
    IntegratedSyntheticRunArtifact,
    IntegratedSyntheticWorldFactory,
)
from policytwin.counterfactual.paired import PairedOutcome, WorldRole
from policytwin.experiments.integrated_synthetic import (
    integrated_synthetic_market_config,
    integrated_synthetic_pair,
)
from policytwin.provenance import (
    canonical_json_bytes,
    read_json_bytes_object,
    sha256_bytes,
    sha256_file,
    write_text_file,
)

AAMAS_INTEGRATED_BRANCH_PLAN_SCHEMA_VERSION = (
    "policytwin.aamas-integrated-branch-plan.v1"
)
AAMAS_INTEGRATED_BRANCH_ARTIFACT_SCHEMA_VERSION = (
    "policytwin.aamas-integrated-branch-artifact.v1"
)
AAMAS_INTEGRATED_BRANCH_CHILD_SCHEMA_VERSION = (
    "policytwin.aamas-integrated-branch-child.v1"
)
AAMAS_INTEGRATED_BRANCH_ATTEMPT_REGISTRY_SCHEMA_VERSION = (
    "policytwin.aamas-integrated-branch-attempt-registry.v1"
)
AAMAS_INTEGRATED_BRANCH_FAILURE_SCHEMA_VERSION = (
    "policytwin.aamas-integrated-branch-failure.v1"
)
AAMAS_INTEGRATED_BRANCH_CLAIM_SCOPE = (
    "synthetic_integrated_clob_branch_and_event_alignment_only"
)
AAMAS_INTEGRATED_BRANCH_FORMAL_REPETITIONS = 256
AAMAS_INTEGRATED_BRANCH_FORMAL_STEPS = 8

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EVENT_KEY = re.compile(r"^ptk1:[0-9a-f]{64}$")
_SECRET = re.compile(r"(?i)\b(?:sk|api[_-]?key|token)[-_:=]?[A-Za-z0-9._-]{8,}\b")
_ERROR_SUMMARY_LIMIT = 256


class AAMASIntegratedBranchError(RuntimeError):
    """Base error for the integrated branch benchmark."""


class AAMASIntegratedBranchSchemaError(AAMASIntegratedBranchError, ValueError):
    """Raised when a benchmark plan or artifact has an invalid schema."""


class AAMASIntegratedBranchIntegrityError(AAMASIntegratedBranchError):
    """Raised when written evidence differs from deterministic replay."""


class AAMASIntegratedBranchAttemptStatus(StrEnum):
    """Terminal state of one preregistered replicate attempt."""

    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED_UPSTREAM = "blocked_upstream"
    NOT_RUN = "not_run"


def _fail(message: str) -> NoReturn:
    raise AAMASIntegratedBranchSchemaError(message)


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        _fail(f"{field} must be a stable ASCII identifier")
    return value


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail(f"{field} must be a lowercase SHA-256 digest")
    return value


def _integer(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"{field} must be an integer >= {minimum}")
    return value


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        _fail(f"{field} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        _fail(f"{field} must be finite")
    return normalized


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        _fail(f"{field} must be an object with string keys")
    return cast(Mapping[str, object], value)


def _sequence(value: object, field: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        _fail(f"{field} must be an array")
    return cast(Sequence[object], value)


def _strict_fields(
    value: Mapping[str, object], expected: frozenset[str], field: str
) -> None:
    if frozenset(value) != expected:
        _fail(f"{field} has missing or unknown fields")


def _optional_integer(value: object, field: str) -> int | None:
    if value is None:
        return None
    return _integer(value, field)


def _optional_string(value: object, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        _fail(f"{field} must be null or a non-empty string")
    return value


def _bounded_error(error: BaseException) -> tuple[str, str, str, str]:
    """Return a bounded, redacted diagnostic plus a full-message fingerprint."""
    error_type = type(error).__name__
    if _IDENTIFIER.fullmatch(error_type) is None:
        error_type = "Exception"
    raw = str(error) or "exception_without_message"
    summary = " ".join(raw.split())
    summary = _SECRET.sub("[REDACTED]", summary)
    if len(summary) > _ERROR_SUMMARY_LIMIT:
        summary = summary[: _ERROR_SUMMARY_LIMIT - 3] + "..."
    return (
        "replicate_execution_failed",
        error_type,
        summary,
        sha256_bytes(canonical_json_bytes({"error_type": error_type, "message": raw})),
    )


@dataclass(frozen=True, slots=True)
class AAMASIntegratedBranchPlan:
    """Frozen execution inputs for the integrated branch benchmark."""

    benchmark_id: str
    seed: str
    repetitions: int
    step_count: int
    minimum_branch_replicates: int
    parameters: tuple[tuple[str, float], ...]
    schema_version: str = AAMAS_INTEGRATED_BRANCH_PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != AAMAS_INTEGRATED_BRANCH_PLAN_SCHEMA_VERSION:
            _fail("unsupported integrated branch plan schema")
        _identifier(self.benchmark_id, "benchmark_id")
        _identifier(self.seed, "seed")
        _integer(self.repetitions, "repetitions", minimum=1)
        _integer(self.step_count, "step_count", minimum=2)
        _integer(
            self.minimum_branch_replicates,
            "minimum_branch_replicates",
            minimum=1,
        )
        if self.minimum_branch_replicates > self.repetitions:
            _fail("minimum_branch_replicates cannot exceed repetitions")
        if not isinstance(self.parameters, tuple) or len(self.parameters) != 2:
            _fail("parameters must contain the two integrated model parameters")
        names: list[str] = []
        normalized: list[tuple[str, float]] = []
        for item in self.parameters:
            if not isinstance(item, tuple) or len(item) != 2:
                _fail("each parameter must be a name/value tuple")
            name = _identifier(item[0], "parameter name")
            value = _number(item[1], f"parameter {name}")
            names.append(name)
            normalized.append((name, value))
        if tuple(names) != ("behavior_scale", "signal_scale"):
            _fail("parameters must be sorted behavior_scale and signal_scale")
        if tuple(normalized) != self.parameters:
            _fail("parameter values must be normalized finite floats")

    @classmethod
    def formal(cls) -> Self:
        """Return the a-priori formal synthetic workload; execution is guarded."""
        return cls(
            benchmark_id="aamas-integrated-clob-branch-v1",
            seed="aamas-integrated-clob-branch-seed-v1",
            repetitions=AAMAS_INTEGRATED_BRANCH_FORMAL_REPETITIONS,
            step_count=AAMAS_INTEGRATED_BRANCH_FORMAL_STEPS,
            minimum_branch_replicates=1,
            parameters=(("behavior_scale", 0.8), ("signal_scale", 1.1)),
        )

    @property
    def execution_work(self) -> int:
        return self.repetitions * self.step_count

    @property
    def is_formal_scale(self) -> bool:
        formal_work = (
            AAMAS_INTEGRATED_BRANCH_FORMAL_REPETITIONS
            * AAMAS_INTEGRATED_BRANCH_FORMAL_STEPS
        )
        return self.execution_work >= formal_work

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def parameter_mapping(self) -> dict[str, float]:
        return dict(self.parameters)

    def to_mapping(self) -> dict[str, object]:
        return {
            "benchmark_id": self.benchmark_id,
            "minimum_branch_replicates": self.minimum_branch_replicates,
            "parameters": {name: value for name, value in self.parameters},
            "repetitions": self.repetitions,
            "schema_version": self.schema_version,
            "seed": self.seed,
            "step_count": self.step_count,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(
            value,
            frozenset(
                {
                    "benchmark_id",
                    "minimum_branch_replicates",
                    "parameters",
                    "repetitions",
                    "schema_version",
                    "seed",
                    "step_count",
                }
            ),
            "integrated branch plan",
        )
        raw_parameters = _mapping(value["parameters"], "parameters")
        return cls(
            benchmark_id=_identifier(value["benchmark_id"], "benchmark_id"),
            seed=_identifier(value["seed"], "seed"),
            repetitions=_integer(value["repetitions"], "repetitions", minimum=1),
            step_count=_integer(value["step_count"], "step_count", minimum=2),
            minimum_branch_replicates=_integer(
                value["minimum_branch_replicates"],
                "minimum_branch_replicates",
                minimum=1,
            ),
            parameters=tuple(
                (name, _number(raw_parameters[name], f"parameter {name}"))
                for name in sorted(raw_parameters)
            ),
            schema_version=cast(str, value["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class AAMASIntegratedBranchRecord:
    """One complete replicate reference and its non-bypassable invariants."""

    replicate_id: int
    event_key: str
    child_path: str
    child_sha256: str
    shock_tape_sha256: str
    baseline_world_sha256: str
    baseline_run_sha256: str
    no_op_difference_sha256: str
    restriction_difference_sha256: str
    restriction_nonzero_difference_count: int
    restriction_internal_event_delta: int
    restriction_branch_observed: bool
    no_op_exact_zero: Literal[True]
    baseline_invariant: Literal[True]
    event_keyed_shocks_aligned: Literal[True]
    replay_byte_identical: Literal[True]
    pair_integrity_passed: Literal[True]

    def __post_init__(self) -> None:
        _integer(self.replicate_id, "replicate_id")
        if (
            not isinstance(self.event_key, str)
            or _EVENT_KEY.fullmatch(self.event_key) is None
        ):
            _fail("event_key must use the exact ptk1 digest form")
        path = PurePosixPath(self.child_path)
        if (
            path.is_absolute()
            or path.parts != ("runs", f"{self.replicate_id:06d}.json")
            or ".." in path.parts
        ):
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
        _integer(
            self.restriction_nonzero_difference_count,
            "restriction_nonzero_difference_count",
            minimum=0,
        )
        if (
            isinstance(self.restriction_internal_event_delta, bool)
            or not isinstance(self.restriction_internal_event_delta, int)
        ):
            _fail("restriction_internal_event_delta must be an integer")
        if not isinstance(self.restriction_branch_observed, bool):
            _fail("restriction_branch_observed must be boolean")
        expected_branch = (
            self.restriction_nonzero_difference_count > 0
            and self.restriction_internal_event_delta != 0
        )
        if self.restriction_branch_observed is not expected_branch:
            _fail("restriction branch flag differs from the retained outcomes")
        for field in (
            "no_op_exact_zero",
            "baseline_invariant",
            "event_keyed_shocks_aligned",
            "replay_byte_identical",
            "pair_integrity_passed",
        ):
            if getattr(self, field) is not True:
                _fail(f"{field} must be proven true")

    def to_mapping(self) -> dict[str, object]:
        return {
            "baseline_invariant": self.baseline_invariant,
            "baseline_run_sha256": self.baseline_run_sha256,
            "baseline_world_sha256": self.baseline_world_sha256,
            "child_path": self.child_path,
            "child_sha256": self.child_sha256,
            "event_key": self.event_key,
            "event_keyed_shocks_aligned": self.event_keyed_shocks_aligned,
            "no_op_difference_sha256": self.no_op_difference_sha256,
            "no_op_exact_zero": self.no_op_exact_zero,
            "pair_integrity_passed": self.pair_integrity_passed,
            "replay_byte_identical": self.replay_byte_identical,
            "replicate_id": self.replicate_id,
            "restriction_difference_sha256": self.restriction_difference_sha256,
            "restriction_branch_observed": self.restriction_branch_observed,
            "restriction_internal_event_delta": self.restriction_internal_event_delta,
            "restriction_nonzero_difference_count": (
                self.restriction_nonzero_difference_count
            ),
            "shock_tape_sha256": self.shock_tape_sha256,
        }


@dataclass(frozen=True, slots=True)
class AAMASIntegratedBranchAttempt:
    """One terminal row in the fixed-size replicate attempt registry."""

    replicate_id: int
    event_key: str
    config_sha256: str
    status: AAMASIntegratedBranchAttemptStatus
    child_path: str | None = None
    child_sha256: str | None = None
    error_code: str | None = None
    error_type: str | None = None
    error_summary: str | None = None
    error_sha256: str | None = None
    blocked_by_replicate_id: int | None = None

    def __post_init__(self) -> None:
        _integer(self.replicate_id, "replicate_id")
        if (
            not isinstance(self.event_key, str)
            or _EVENT_KEY.fullmatch(self.event_key) is None
        ):
            _fail("attempt event_key must use the exact ptk1 digest form")
        _sha256(self.config_sha256, "config_sha256")
        if not isinstance(self.status, AAMASIntegratedBranchAttemptStatus):
            _fail("attempt status must be a typed terminal status")
        if self.status is AAMASIntegratedBranchAttemptStatus.COMPLETED:
            expected_path = f"runs/{self.replicate_id:06d}.json"
            if self.child_path != expected_path:
                _fail("completed attempt must reference its canonical child")
            _sha256(self.child_sha256, "child_sha256")
            if any(
                value is not None
                for value in (
                    self.error_code,
                    self.error_type,
                    self.error_summary,
                    self.error_sha256,
                    self.blocked_by_replicate_id,
                )
            ):
                _fail("completed attempt cannot contain failure metadata")
            return
        if self.child_path is not None or self.child_sha256 is not None:
            _fail("non-completed attempt cannot claim a result child")
        if self.status is AAMASIntegratedBranchAttemptStatus.FAILED:
            if self.error_code != "replicate_execution_failed":
                _fail("failed attempt requires the fixed failure code")
            _identifier(self.error_type, "error_type")
            if (
                not isinstance(self.error_summary, str)
                or not self.error_summary
                or len(self.error_summary) > _ERROR_SUMMARY_LIMIT
                or any(character in self.error_summary for character in "\r\n")
            ):
                _fail("failed attempt requires a bounded one-line error summary")
            _sha256(self.error_sha256, "error_sha256")
            if self.blocked_by_replicate_id is not None:
                _fail("failed attempt cannot be blocked by another replicate")
            return
        if any(
            value is not None
            for value in (
                self.error_code,
                self.error_type,
                self.error_summary,
                self.error_sha256,
            )
        ):
            _fail("unexecuted attempt cannot contain an invented error")
        if self.status is AAMASIntegratedBranchAttemptStatus.BLOCKED_UPSTREAM:
            blocked_by = _optional_integer(
                self.blocked_by_replicate_id, "blocked_by_replicate_id"
            )
            if blocked_by is None or blocked_by >= self.replicate_id:
                _fail("blocked attempt must identify an earlier failed replicate")
        elif self.blocked_by_replicate_id is not None:
            _fail("not-run attempt cannot claim an upstream failure")

    def to_mapping(self) -> dict[str, object]:
        return {
            "blocked_by_replicate_id": self.blocked_by_replicate_id,
            "child_path": self.child_path,
            "child_sha256": self.child_sha256,
            "config_sha256": self.config_sha256,
            "error_code": self.error_code,
            "error_sha256": self.error_sha256,
            "error_summary": self.error_summary,
            "error_type": self.error_type,
            "event_key": self.event_key,
            "replicate_id": self.replicate_id,
            "status": self.status.value,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(value, frozenset(cls.__annotations__), "branch attempt")
        raw_status = value["status"]
        if not isinstance(raw_status, str):
            _fail("attempt status must be a string")
        try:
            status = AAMASIntegratedBranchAttemptStatus(raw_status)
        except ValueError as error:
            raise AAMASIntegratedBranchSchemaError(
                "attempt status is not recognized"
            ) from error
        child_sha256 = _optional_string(value["child_sha256"], "child_sha256")
        error_sha256 = _optional_string(value["error_sha256"], "error_sha256")
        return cls(
            replicate_id=_integer(value["replicate_id"], "replicate_id"),
            event_key=cast(str, value["event_key"]),
            config_sha256=cast(str, value["config_sha256"]),
            status=status,
            child_path=_optional_string(value["child_path"], "child_path"),
            child_sha256=child_sha256,
            error_code=_optional_string(value["error_code"], "error_code"),
            error_type=_optional_string(value["error_type"], "error_type"),
            error_summary=_optional_string(
                value["error_summary"], "error_summary"
            ),
            error_sha256=error_sha256,
            blocked_by_replicate_id=_optional_integer(
                value["blocked_by_replicate_id"], "blocked_by_replicate_id"
            ),
        )


@dataclass(frozen=True, slots=True)
class AAMASIntegratedBranchAttemptRegistry:
    """Complete terminal registry; failures cannot disappear from denominators."""

    benchmark_id: str
    plan_sha256: str
    config_sha256: str
    expected_repetitions: int
    attempts: tuple[AAMASIntegratedBranchAttempt, ...]
    terminal_status: Literal["completed", "failed"]
    failure_replicate_id: int | None
    scientifically_complete_for_claim: bool
    schema_version: str = AAMAS_INTEGRATED_BRANCH_ATTEMPT_REGISTRY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _identifier(self.benchmark_id, "benchmark_id")
        _sha256(self.plan_sha256, "plan_sha256")
        _sha256(self.config_sha256, "config_sha256")
        _integer(self.expected_repetitions, "expected_repetitions", minimum=1)
        if (
            not isinstance(self.attempts, tuple)
            or len(self.attempts) != self.expected_repetitions
        ):
            _fail("attempt registry must contain every planned replicate")
        if tuple(item.replicate_id for item in self.attempts) != tuple(
            range(self.expected_repetitions)
        ):
            _fail("attempt registry replicate ids must be complete and ordered")
        if any(item.config_sha256 != self.config_sha256 for item in self.attempts):
            _fail("attempt config hashes differ from the registry")
        if (
            self.schema_version
            != AAMAS_INTEGRATED_BRANCH_ATTEMPT_REGISTRY_SCHEMA_VERSION
        ):
            _fail("unsupported attempt registry schema")
        failed = tuple(
            item.replicate_id
            for item in self.attempts
            if item.status is AAMASIntegratedBranchAttemptStatus.FAILED
        )
        if self.terminal_status == "completed":
            if self.failure_replicate_id is not None or failed:
                _fail("completed registry cannot contain a failed replicate")
            if any(
                item.status is not AAMASIntegratedBranchAttemptStatus.COMPLETED
                for item in self.attempts
            ):
                _fail("completed registry must complete every planned replicate")
            if self.scientifically_complete_for_claim is not True:
                _fail("completed registry must be scientifically complete")
            return
        if self.terminal_status != "failed":
            _fail("attempt registry terminal_status is invalid")
        if self.scientifically_complete_for_claim is not False:
            _fail("failed registry cannot be scientifically complete")
        if self.failure_replicate_id is None:
            if failed or any(
                item.status is not AAMASIntegratedBranchAttemptStatus.COMPLETED
                for item in self.attempts
            ):
                _fail("collection-level failure requires all attempts completed")
            return
        failure_id = _integer(self.failure_replicate_id, "failure_replicate_id")
        if failed != (failure_id,):
            _fail("failed registry must identify exactly its one failed replicate")
        for attempt in self.attempts:
            expected_status = (
                AAMASIntegratedBranchAttemptStatus.COMPLETED
                if attempt.replicate_id < failure_id
                else AAMASIntegratedBranchAttemptStatus.FAILED
                if attempt.replicate_id == failure_id
                else AAMASIntegratedBranchAttemptStatus.BLOCKED_UPSTREAM
            )
            if attempt.status is not expected_status:
                _fail("failed attempt propagation differs from ordered execution")
            if (
                attempt.status is AAMASIntegratedBranchAttemptStatus.BLOCKED_UPSTREAM
                and attempt.blocked_by_replicate_id != failure_id
            ):
                _fail("blocked attempt references the wrong upstream failure")

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_mapping())

    @property
    def sha256(self) -> str:
        return sha256_bytes(self.canonical_bytes)

    def to_mapping(self) -> dict[str, object]:
        return {
            "attempts": [item.to_mapping() for item in self.attempts],
            "benchmark_id": self.benchmark_id,
            "config_sha256": self.config_sha256,
            "expected_repetitions": self.expected_repetitions,
            "failure_replicate_id": self.failure_replicate_id,
            "plan_sha256": self.plan_sha256,
            "schema_version": self.schema_version,
            "scientifically_complete_for_claim": self.scientifically_complete_for_claim,
            "terminal_status": self.terminal_status,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(value, frozenset(cls.__annotations__), "attempt registry")
        raw_attempts = _sequence(value["attempts"], "attempts")
        complete = value["scientifically_complete_for_claim"]
        if not isinstance(complete, bool):
            _fail("scientifically_complete_for_claim must be boolean")
        terminal = value["terminal_status"]
        if terminal not in ("completed", "failed"):
            _fail("terminal_status must be completed or failed")
        return cls(
            benchmark_id=_identifier(value["benchmark_id"], "benchmark_id"),
            plan_sha256=_sha256(value["plan_sha256"], "plan_sha256"),
            config_sha256=_sha256(value["config_sha256"], "config_sha256"),
            expected_repetitions=_integer(
                value["expected_repetitions"], "expected_repetitions", minimum=1
            ),
            attempts=tuple(
                AAMASIntegratedBranchAttempt.from_mapping(
                    _mapping(item, "attempt")
                )
                for item in raw_attempts
            ),
            terminal_status=cast(Literal["completed", "failed"], terminal),
            failure_replicate_id=_optional_integer(
                value["failure_replicate_id"], "failure_replicate_id"
            ),
            scientifically_complete_for_claim=complete,
            schema_version=cast(str, value["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class AAMASIntegratedBranchArtifact:
    """Deterministic root registry for every integrated branch replicate."""

    benchmark_id: str
    plan_sha256: str
    config_sha256: str
    records: tuple[AAMASIntegratedBranchRecord, ...]
    minimum_branch_replicates: int
    branch_replicate_count: int
    integrity_valid: Literal[True] = True
    scientifically_complete_for_claim: Literal[True] = True
    publication_eligible: Literal[False] = False
    synthetic_only: Literal[True] = True
    claim_scope: str = AAMAS_INTEGRATED_BRANCH_CLAIM_SCOPE
    schema_version: str = AAMAS_INTEGRATED_BRANCH_ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _identifier(self.benchmark_id, "benchmark_id")
        _sha256(self.plan_sha256, "plan_sha256")
        _sha256(self.config_sha256, "config_sha256")
        if not isinstance(self.records, tuple) or not self.records:
            _fail("integrated branch artifact requires records")
        if tuple(item.replicate_id for item in self.records) != tuple(
            range(len(self.records))
        ):
            _fail("integrated branch records must be complete and ordered")
        _integer(
            self.minimum_branch_replicates,
            "minimum_branch_replicates",
            minimum=1,
        )
        _integer(self.branch_replicate_count, "branch_replicate_count", minimum=0)
        expected_branch_count = sum(
            item.restriction_branch_observed for item in self.records
        )
        if self.branch_replicate_count != expected_branch_count:
            _fail("branch replicate count differs from the complete registry")
        if self.branch_replicate_count < self.minimum_branch_replicates:
            _fail("the preregistered collection-level branch condition was not met")
        if any(
            value is not expected
            for value, expected in (
                (self.integrity_valid, True),
                (self.scientifically_complete_for_claim, True),
                (self.publication_eligible, False),
                (self.synthetic_only, True),
            )
        ):
            _fail("integrated branch claim flags cannot be broadened")
        if self.claim_scope != AAMAS_INTEGRATED_BRANCH_CLAIM_SCOPE:
            _fail("integrated branch claim scope is invalid")
        if self.schema_version != AAMAS_INTEGRATED_BRANCH_ARTIFACT_SCHEMA_VERSION:
            _fail("unsupported integrated branch artifact schema")

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_mapping())

    @property
    def sha256(self) -> str:
        return sha256_bytes(self.canonical_bytes)

    def to_mapping(self) -> dict[str, object]:
        return {
            "benchmark_id": self.benchmark_id,
            "branch_replicate_count": self.branch_replicate_count,
            "claim_scope": self.claim_scope,
            "config_sha256": self.config_sha256,
            "integrity_valid": self.integrity_valid,
            "minimum_branch_replicates": self.minimum_branch_replicates,
            "plan_sha256": self.plan_sha256,
            "publication_eligible": self.publication_eligible,
            "records": [item.to_mapping() for item in self.records],
            "schema_version": self.schema_version,
            "scientifically_complete_for_claim": self.scientifically_complete_for_claim,
            "synthetic_only": self.synthetic_only,
        }


@dataclass(frozen=True, slots=True)
class AAMASIntegratedBranchFailureArtifact:
    """Terminal evidence root for an incomplete benchmark execution."""

    benchmark_id: str
    plan_sha256: str
    config_sha256: str
    attempt_registry_sha256: str
    expected_repetitions: int
    completed_count: int
    failed_count: int
    blocked_count: int
    failure_replicate_id: int | None
    failure_code: Literal[
        "replicate_execution_failed", "minimum_branch_replicates_not_met"
    ]
    error_type: str
    error_summary: str
    error_sha256: str
    integrity_valid: Literal[True] = True
    scientifically_complete_for_claim: Literal[False] = False
    publication_eligible: Literal[False] = False
    synthetic_only: Literal[True] = True
    claim_scope: str = AAMAS_INTEGRATED_BRANCH_CLAIM_SCOPE
    schema_version: str = AAMAS_INTEGRATED_BRANCH_FAILURE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _identifier(self.benchmark_id, "benchmark_id")
        for field in (
            "plan_sha256",
            "config_sha256",
            "attempt_registry_sha256",
            "error_sha256",
        ):
            _sha256(getattr(self, field), field)
        repetitions = _integer(
            self.expected_repetitions, "expected_repetitions", minimum=1
        )
        completed = _integer(self.completed_count, "completed_count")
        failed = _integer(self.failed_count, "failed_count")
        blocked = _integer(self.blocked_count, "blocked_count")
        if completed + failed + blocked != repetitions:
            _fail("failure counts must cover every planned replicate")
        _identifier(self.error_type, "error_type")
        if (
            not isinstance(self.error_summary, str)
            or not self.error_summary
            or len(self.error_summary) > _ERROR_SUMMARY_LIMIT
            or any(character in self.error_summary for character in "\r\n")
        ):
            _fail("failure root requires a bounded one-line error summary")
        if self.failure_code == "replicate_execution_failed":
            failure_id = _optional_integer(
                self.failure_replicate_id, "failure_replicate_id"
            )
            if failure_id is None or failed != 1 or completed != failure_id:
                _fail("replicate failure counts do not match the failed id")
            if blocked != repetitions - failure_id - 1:
                _fail("replicate failure blocked count is inconsistent")
        elif self.failure_code == "minimum_branch_replicates_not_met":
            if (
                self.failure_replicate_id is not None
                or completed != repetitions
                or failed != 0
                or blocked != 0
            ):
                _fail("collection failure requires all replicate attempts")
        else:
            _fail("failure_code is invalid")
        if any(
            value is not expected
            for value, expected in (
                (self.integrity_valid, True),
                (self.scientifically_complete_for_claim, False),
                (self.publication_eligible, False),
                (self.synthetic_only, True),
            )
        ):
            _fail("failure root claim flags cannot be broadened")
        if self.claim_scope != AAMAS_INTEGRATED_BRANCH_CLAIM_SCOPE:
            _fail("failure root claim scope is invalid")
        if self.schema_version != AAMAS_INTEGRATED_BRANCH_FAILURE_SCHEMA_VERSION:
            _fail("unsupported integrated branch failure schema")

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_mapping())

    @property
    def sha256(self) -> str:
        return sha256_bytes(self.canonical_bytes)

    def to_mapping(self) -> dict[str, object]:
        return {
            "attempt_registry_sha256": self.attempt_registry_sha256,
            "benchmark_id": self.benchmark_id,
            "blocked_count": self.blocked_count,
            "claim_scope": self.claim_scope,
            "completed_count": self.completed_count,
            "config_sha256": self.config_sha256,
            "error_sha256": self.error_sha256,
            "error_summary": self.error_summary,
            "error_type": self.error_type,
            "expected_repetitions": self.expected_repetitions,
            "failed_count": self.failed_count,
            "failure_code": self.failure_code,
            "failure_replicate_id": self.failure_replicate_id,
            "integrity_valid": self.integrity_valid,
            "plan_sha256": self.plan_sha256,
            "publication_eligible": self.publication_eligible,
            "schema_version": self.schema_version,
            "scientifically_complete_for_claim": self.scientifically_complete_for_claim,
            "synthetic_only": self.synthetic_only,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(value, frozenset(cls.__annotations__), "failure artifact")
        failure_code = value["failure_code"]
        if failure_code not in (
            "replicate_execution_failed",
            "minimum_branch_replicates_not_met",
        ):
            _fail("failure_code is invalid")
        fixed_flags: dict[str, bool] = {}
        for field in (
            "integrity_valid",
            "scientifically_complete_for_claim",
            "publication_eligible",
            "synthetic_only",
        ):
            raw = value[field]
            if not isinstance(raw, bool):
                _fail(f"{field} must be boolean")
            fixed_flags[field] = raw
        return cls(
            benchmark_id=_identifier(value["benchmark_id"], "benchmark_id"),
            plan_sha256=_sha256(value["plan_sha256"], "plan_sha256"),
            config_sha256=_sha256(value["config_sha256"], "config_sha256"),
            attempt_registry_sha256=_sha256(
                value["attempt_registry_sha256"], "attempt_registry_sha256"
            ),
            expected_repetitions=_integer(
                value["expected_repetitions"], "expected_repetitions", minimum=1
            ),
            completed_count=_integer(value["completed_count"], "completed_count"),
            failed_count=_integer(value["failed_count"], "failed_count"),
            blocked_count=_integer(value["blocked_count"], "blocked_count"),
            failure_replicate_id=_optional_integer(
                value["failure_replicate_id"], "failure_replicate_id"
            ),
            failure_code=cast(
                Literal[
                    "replicate_execution_failed",
                    "minimum_branch_replicates_not_met",
                ],
                failure_code,
            ),
            error_type=_identifier(value["error_type"], "error_type"),
            error_summary=cast(str, value["error_summary"]),
            error_sha256=_sha256(value["error_sha256"], "error_sha256"),
            integrity_valid=cast(Literal[True], fixed_flags["integrity_valid"]),
            scientifically_complete_for_claim=cast(
                Literal[False], fixed_flags["scientifically_complete_for_claim"]
            ),
            publication_eligible=cast(
                Literal[False], fixed_flags["publication_eligible"]
            ),
            synthetic_only=cast(Literal[True], fixed_flags["synthetic_only"]),
            claim_scope=cast(str, value["claim_scope"]),
            schema_version=cast(str, value["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class AAMASIntegratedBranchRun:
    """In-memory artifact plus exact child bytes ready for publication."""

    plan: AAMASIntegratedBranchPlan
    artifact: AAMASIntegratedBranchArtifact
    children: tuple[tuple[str, bytes], ...]

    def __post_init__(self) -> None:
        if self.artifact.plan_sha256 != self.plan.sha256:
            _fail("run artifact is not bound to its plan")
        expected = tuple(item.child_path for item in self.artifact.records)
        if tuple(path for path, _ in self.children) != expected:
            _fail("run children differ from the complete record registry")
        if any(
            sha256_bytes(content) != record.child_sha256
            for record, (_, content) in zip(
                self.artifact.records, self.children, strict=True
            )
        ):
            _fail("run child bytes differ from their record commitments")

    @property
    def attempt_registry(self) -> AAMASIntegratedBranchAttemptRegistry:
        """Return the complete success registry without changing artifact v1."""
        return _success_attempt_registry(self)


@dataclass(frozen=True, slots=True)
class AAMASIntegratedBranchFailureRun:
    """In-memory terminal evidence retained after an interrupted run."""

    plan: AAMASIntegratedBranchPlan
    attempt_registry: AAMASIntegratedBranchAttemptRegistry
    artifact: AAMASIntegratedBranchFailureArtifact
    children: tuple[tuple[str, bytes], ...]

    def __post_init__(self) -> None:
        if self.attempt_registry.plan_sha256 != self.plan.sha256:
            _fail("failure registry is not bound to its plan")
        if self.artifact.plan_sha256 != self.plan.sha256:
            _fail("failure artifact is not bound to its plan")
        if self.artifact.attempt_registry_sha256 != self.attempt_registry.sha256:
            _fail("failure artifact is not bound to its attempt registry")
        completed = tuple(
            item
            for item in self.attempt_registry.attempts
            if item.status is AAMASIntegratedBranchAttemptStatus.COMPLETED
        )
        if tuple(path for path, _ in self.children) != tuple(
            cast(str, item.child_path) for item in completed
        ):
            _fail("failure children differ from completed attempt rows")
        if any(
            sha256_bytes(content) != item.child_sha256
            for item, (_, content) in zip(completed, self.children, strict=True)
        ):
            _fail("failure child bytes differ from attempt commitments")


class AAMASIntegratedBranchExecutionError(AAMASIntegratedBranchError):
    """Raised by the success-only runner while retaining failure evidence."""

    def __init__(self, evidence: AAMASIntegratedBranchFailureRun) -> None:
        self.evidence = evidence
        super().__init__(
            "integrated branch execution is scientifically incomplete; "
            f"terminal evidence sha256={evidence.artifact.sha256}"
        )


@dataclass(frozen=True, slots=True)
class AAMASIntegratedBranchAudit:
    """Independent replay decision for one written branch bundle."""

    plan_sha256: str
    artifact_sha256: str
    record_count: int
    passed: Literal[True] = True


@dataclass(frozen=True, slots=True)
class AAMASIntegratedBranchFailureAudit:
    """Audit receipt proving integrity, but not scientific completeness."""

    plan_sha256: str
    failure_sha256: str
    attempt_registry_sha256: str
    attempt_count: int
    completed_count: int
    failed_count: int
    blocked_count: int
    evidence_integrity_valid: Literal[True] = True
    scientifically_complete_for_claim: Literal[False] = False


AAMASIntegratedBranchEvidenceAudit = (
    AAMASIntegratedBranchAudit | AAMASIntegratedBranchFailureAudit
)


class AAMASIntegratedBranchPublishedFailure(AAMASIntegratedBranchError):
    """Raised after the compatibility writer publishes a failure bundle."""

    def __init__(
        self, output_dir: Path, receipt: AAMASIntegratedBranchFailureAudit
    ) -> None:
        self.output_dir = output_dir
        self.receipt = receipt
        super().__init__(
            "integrated branch execution failed; terminal evidence was published "
            f"at {output_dir} with sha256={receipt.failure_sha256}"
        )


@dataclass(frozen=True, slots=True)
class _ModeExecution:
    outcome: PairedOutcome
    baseline_run: IntegratedSyntheticRunArtifact
    candidate_run: IntegratedSyntheticRunArtifact


def _event_key(plan: AAMASIntegratedBranchPlan, replicate_id: int) -> EventKey:
    return EventKey(
        experiment_id=plan.benchmark_id,
        seed=plan.seed,
        replicate_id=replicate_id,
        simulation_step=0,
        purpose="integrated-clob-paired-world",
    )


def _execute_mode(
    plan: AAMASIntegratedBranchPlan,
    replicate_id: int,
    *,
    no_op: bool,
    replay_order: bool,
) -> _ModeExecution:
    config = integrated_synthetic_market_config()
    pair = integrated_synthetic_pair(
        event_key=_event_key(plan, replicate_id),
        parameters=plan.parameter_mapping(),
        config=config,
        candidate_allow_buy=no_op,
        candidate_allow_sell=True,
        candidate_is_noop=no_op,
        step_count=plan.step_count,
        include_all_groups=True,
    )
    factory = IntegratedSyntheticWorldFactory(config)
    outcome = pair.run(
        factory,
        parallel=replay_order,
        invocation_order=(
            (WorldRole.CANDIDATE, WorldRole.BASELINE)
            if replay_order
            else (WorldRole.BASELINE, WorldRole.CANDIDATE)
        ),
    )
    baseline = factory.runners[WorldRole.BASELINE].artifact
    candidate = factory.runners[WorldRole.CANDIDATE].artifact
    if baseline is None or candidate is None:
        raise AAMASIntegratedBranchIntegrityError(
            "integrated world omitted a complete CLOB run artifact"
        )
    return _ModeExecution(outcome, baseline, candidate)


def _differences_sha256(outcome: PairedOutcome) -> str:
    return sha256_bytes(
        canonical_json_bytes([item.to_mapping() for item in outcome.differences])
    )


def _shock_signature(
    run: IntegratedSyntheticRunArtifact,
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (step.shock_event_key, step.exogenous_command_sha256) for step in run.steps
    )


def _mode_mapping(mode: _ModeExecution) -> dict[str, object]:
    return {
        "baseline_run": mode.baseline_run.to_mapping(),
        "candidate_run": mode.candidate_run.to_mapping(),
        "paired_outcome": mode.outcome.to_mapping(),
    }


def _execute_replicate(
    plan: AAMASIntegratedBranchPlan, replicate_id: int
) -> tuple[AAMASIntegratedBranchRecord, bytes]:
    no_op = _execute_mode(plan, replicate_id, no_op=True, replay_order=False)
    no_op_replay = _execute_mode(plan, replicate_id, no_op=True, replay_order=True)
    restriction = _execute_mode(plan, replicate_id, no_op=False, replay_order=False)
    restriction_replay = _execute_mode(
        plan, replicate_id, no_op=False, replay_order=True
    )

    replay_byte_identical = all(
        left == right
        for left, right in (
            (no_op.outcome.canonical_bytes, no_op_replay.outcome.canonical_bytes),
            (
                no_op.baseline_run.canonical_bytes,
                no_op_replay.baseline_run.canonical_bytes,
            ),
            (
                no_op.candidate_run.canonical_bytes,
                no_op_replay.candidate_run.canonical_bytes,
            ),
            (
                restriction.outcome.canonical_bytes,
                restriction_replay.outcome.canonical_bytes,
            ),
            (
                restriction.baseline_run.canonical_bytes,
                restriction_replay.baseline_run.canonical_bytes,
            ),
            (
                restriction.candidate_run.canonical_bytes,
                restriction_replay.candidate_run.canonical_bytes,
            ),
        )
    )
    baseline_invariant = (
        canonical_json_bytes(no_op.outcome.baseline.to_mapping())
        == canonical_json_bytes(restriction.outcome.baseline.to_mapping())
        and no_op.baseline_run.canonical_bytes
        == restriction.baseline_run.canonical_bytes
    )
    signatures = (
        _shock_signature(no_op.baseline_run),
        _shock_signature(no_op.candidate_run),
        _shock_signature(restriction.baseline_run),
        _shock_signature(restriction.candidate_run),
    )
    event_keyed_shocks_aligned = all(item == signatures[0] for item in signatures[1:])
    no_op_exact_zero = all(item.difference == 0.0 for item in no_op.outcome.differences)
    pair_integrity_passed = (
        no_op.outcome.integrity_audit.passed
        and restriction.outcome.integrity_audit.passed
    )
    nonzero_count = sum(
        item.difference != 0.0 for item in restriction.outcome.differences
    )
    internal_delta = (
        restriction.outcome.candidate.internal_event_count
        - restriction.outcome.baseline.internal_event_count
    )
    checks = {
        "baseline_invariant": baseline_invariant,
        "event_keyed_shocks_aligned": event_keyed_shocks_aligned,
        "no_op_exact_zero": no_op_exact_zero,
        "pair_integrity_passed": pair_integrity_passed,
        "replay_byte_identical": replay_byte_identical,
        "restriction_branch_observed": nonzero_count > 0 and internal_delta != 0,
    }
    required_checks = {
        name: passed
        for name, passed in checks.items()
        if name != "restriction_branch_observed"
    }
    if not all(required_checks.values()):
        failed = sorted(name for name, passed in required_checks.items() if not passed)
        raise AAMASIntegratedBranchIntegrityError(
            "integrated branch replicate failed: " + ", ".join(failed)
        )

    key = _event_key(plan, replicate_id)
    child_mapping = {
        "audit": checks,
        "claim_scope": AAMAS_INTEGRATED_BRANCH_CLAIM_SCOPE,
        "event_key": key.value,
        "no_op": _mode_mapping(no_op),
        "plan_sha256": plan.sha256,
        "replicate_id": replicate_id,
        "restriction": _mode_mapping(restriction),
        "schema_version": AAMAS_INTEGRATED_BRANCH_CHILD_SCHEMA_VERSION,
        "synthetic_only": True,
    }
    child_bytes = canonical_json_bytes(child_mapping)
    child_path = f"runs/{replicate_id:06d}.json"
    record = AAMASIntegratedBranchRecord(
        replicate_id=replicate_id,
        event_key=key.value,
        child_path=child_path,
        child_sha256=sha256_bytes(child_bytes),
        shock_tape_sha256=no_op.outcome.shock_tape_sha256,
        baseline_world_sha256=no_op.outcome.baseline.sha256,
        baseline_run_sha256=no_op.baseline_run.sha256,
        no_op_difference_sha256=_differences_sha256(no_op.outcome),
        restriction_difference_sha256=_differences_sha256(restriction.outcome),
        restriction_nonzero_difference_count=nonzero_count,
        restriction_internal_event_delta=internal_delta,
        restriction_branch_observed=checks["restriction_branch_observed"],
        no_op_exact_zero=True,
        baseline_invariant=True,
        event_keyed_shocks_aligned=True,
        replay_byte_identical=True,
        pair_integrity_passed=True,
    )
    return record, child_bytes


def _completed_attempt(
    record: AAMASIntegratedBranchRecord,
    config_sha256: str,
) -> AAMASIntegratedBranchAttempt:
    return AAMASIntegratedBranchAttempt(
        replicate_id=record.replicate_id,
        event_key=record.event_key,
        config_sha256=config_sha256,
        status=AAMASIntegratedBranchAttemptStatus.COMPLETED,
        child_path=record.child_path,
        child_sha256=record.child_sha256,
    )


def _success_attempt_registry(
    run: AAMASIntegratedBranchRun,
) -> AAMASIntegratedBranchAttemptRegistry:
    return AAMASIntegratedBranchAttemptRegistry(
        benchmark_id=run.plan.benchmark_id,
        plan_sha256=run.plan.sha256,
        config_sha256=run.artifact.config_sha256,
        expected_repetitions=run.plan.repetitions,
        attempts=tuple(
            _completed_attempt(record, run.artifact.config_sha256)
            for record in run.artifact.records
        ),
        terminal_status="completed",
        failure_replicate_id=None,
        scientifically_complete_for_claim=True,
    )


def _failure_artifact(
    plan: AAMASIntegratedBranchPlan,
    registry: AAMASIntegratedBranchAttemptRegistry,
    *,
    failure_code: Literal[
        "replicate_execution_failed", "minimum_branch_replicates_not_met"
    ],
    error_type: str,
    error_summary: str,
    error_sha256: str,
) -> AAMASIntegratedBranchFailureArtifact:
    statuses = tuple(item.status for item in registry.attempts)
    return AAMASIntegratedBranchFailureArtifact(
        benchmark_id=plan.benchmark_id,
        plan_sha256=plan.sha256,
        config_sha256=registry.config_sha256,
        attempt_registry_sha256=registry.sha256,
        expected_repetitions=plan.repetitions,
        completed_count=statuses.count(AAMASIntegratedBranchAttemptStatus.COMPLETED),
        failed_count=statuses.count(AAMASIntegratedBranchAttemptStatus.FAILED),
        blocked_count=statuses.count(
            AAMASIntegratedBranchAttemptStatus.BLOCKED_UPSTREAM
        )
        + statuses.count(AAMASIntegratedBranchAttemptStatus.NOT_RUN),
        failure_replicate_id=registry.failure_replicate_id,
        failure_code=failure_code,
        error_type=error_type,
        error_summary=error_summary,
        error_sha256=error_sha256,
    )


def _capture_aamas_integrated_branch(
    plan: AAMASIntegratedBranchPlan,
    *,
    allow_formal_scale: bool,
) -> AAMASIntegratedBranchRun | AAMASIntegratedBranchFailureRun:
    if not isinstance(plan, AAMASIntegratedBranchPlan):
        raise TypeError("plan must be an AAMASIntegratedBranchPlan")
    if not isinstance(allow_formal_scale, bool):
        raise TypeError("allow_formal_scale must be boolean")
    if plan.is_formal_scale and not allow_formal_scale:
        raise AAMASIntegratedBranchError(
            "formal-scale integrated branch execution requires explicit authorization"
        )
    config_sha256 = integrated_synthetic_market_config().sha256
    records: list[AAMASIntegratedBranchRecord] = []
    children: list[tuple[str, bytes]] = []
    attempts: list[AAMASIntegratedBranchAttempt] = []
    for replicate_id in range(plan.repetitions):
        try:
            record, content = _execute_replicate(plan, replicate_id)
        except Exception as error:
            error_code, error_type, error_summary, error_sha256 = _bounded_error(error)
            attempts.append(
                AAMASIntegratedBranchAttempt(
                    replicate_id=replicate_id,
                    event_key=_event_key(plan, replicate_id).value,
                    config_sha256=config_sha256,
                    status=AAMASIntegratedBranchAttemptStatus.FAILED,
                    error_code=error_code,
                    error_type=error_type,
                    error_summary=error_summary,
                    error_sha256=error_sha256,
                )
            )
            attempts.extend(
                AAMASIntegratedBranchAttempt(
                    replicate_id=blocked_id,
                    event_key=_event_key(plan, blocked_id).value,
                    config_sha256=config_sha256,
                    status=AAMASIntegratedBranchAttemptStatus.BLOCKED_UPSTREAM,
                    blocked_by_replicate_id=replicate_id,
                )
                for blocked_id in range(replicate_id + 1, plan.repetitions)
            )
            registry = AAMASIntegratedBranchAttemptRegistry(
                benchmark_id=plan.benchmark_id,
                plan_sha256=plan.sha256,
                config_sha256=config_sha256,
                expected_repetitions=plan.repetitions,
                attempts=tuple(attempts),
                terminal_status="failed",
                failure_replicate_id=replicate_id,
                scientifically_complete_for_claim=False,
            )
            failure = _failure_artifact(
                plan,
                registry,
                failure_code="replicate_execution_failed",
                error_type=error_type,
                error_summary=error_summary,
                error_sha256=error_sha256,
            )
            return AAMASIntegratedBranchFailureRun(
                plan, registry, failure, tuple(children)
            )
        records.append(record)
        children.append((record.child_path, content))
        attempts.append(_completed_attempt(record, config_sha256))
    branch_count = sum(record.restriction_branch_observed for record in records)
    if branch_count < plan.minimum_branch_replicates:
        error_summary = (
            "preregistered minimum branch replicate condition was not met"
        )
        error_sha256 = sha256_bytes(
            canonical_json_bytes(
                {
                    "branch_replicate_count": branch_count,
                    "minimum_branch_replicates": plan.minimum_branch_replicates,
                }
            )
        )
        registry = AAMASIntegratedBranchAttemptRegistry(
            benchmark_id=plan.benchmark_id,
            plan_sha256=plan.sha256,
            config_sha256=config_sha256,
            expected_repetitions=plan.repetitions,
            attempts=tuple(attempts),
            terminal_status="failed",
            failure_replicate_id=None,
            scientifically_complete_for_claim=False,
        )
        failure = _failure_artifact(
            plan,
            registry,
            failure_code="minimum_branch_replicates_not_met",
            error_type="ScientificConditionFailure",
            error_summary=error_summary,
            error_sha256=error_sha256,
        )
        return AAMASIntegratedBranchFailureRun(
            plan, registry, failure, tuple(children)
        )
    artifact = AAMASIntegratedBranchArtifact(
        benchmark_id=plan.benchmark_id,
        plan_sha256=plan.sha256,
        config_sha256=config_sha256,
        records=tuple(records),
        minimum_branch_replicates=plan.minimum_branch_replicates,
        branch_replicate_count=branch_count,
    )
    return AAMASIntegratedBranchRun(plan, artifact, tuple(children))


def run_aamas_integrated_branch(
    plan: AAMASIntegratedBranchPlan,
    *,
    allow_formal_scale: bool = False,
) -> AAMASIntegratedBranchRun:
    """Execute all replicates or raise with a complete in-memory failure registry."""
    evidence = _capture_aamas_integrated_branch(
        plan, allow_formal_scale=allow_formal_scale
    )
    if isinstance(evidence, AAMASIntegratedBranchFailureRun):
        raise AAMASIntegratedBranchExecutionError(evidence)
    return evidence


def _expected_members(run: AAMASIntegratedBranchRun) -> frozenset[str]:
    return frozenset(
        {"artifact.json", "artifact.sha256", "attempts.json", "plan.json"}
        | {path for path, _ in run.children}
    )


def _expected_failure_members(
    run: AAMASIntegratedBranchFailureRun,
) -> frozenset[str]:
    return frozenset(
        {"attempts.json", "failure.json", "failure.sha256", "plan.json"}
        | {path for path, _ in run.children}
    )


def _audit_membership(root: Path, expected: frozenset[str]) -> None:
    if not root.is_dir() or root.is_symlink():
        raise AAMASIntegratedBranchIntegrityError(
            "integrated branch bundle root must be a real directory"
        )
    observed: set[str] = set()
    directories = {"runs"}
    observed_directories: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise AAMASIntegratedBranchIntegrityError(
                "integrated branch bundle must not contain symbolic links"
            )
        if stat.S_ISDIR(info.st_mode):
            if relative not in directories:
                raise AAMASIntegratedBranchIntegrityError(
                    "integrated branch bundle contains an unexpected directory"
                )
            observed_directories.add(relative)
            continue
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise AAMASIntegratedBranchIntegrityError(
                "integrated branch members must be singly linked regular files"
            )
        observed.add(relative)
    if observed_directories != directories or observed != set(expected):
        raise AAMASIntegratedBranchIntegrityError(
            "integrated branch bundle membership differs from its registry"
        )


def _load_canonical_plan(path: Path, expected_sha256: str) -> AAMASIntegratedBranchPlan:
    content = path.read_bytes()
    try:
        mapping = read_json_bytes_object(content, source=str(path))
    except Exception as error:
        raise AAMASIntegratedBranchIntegrityError(
            "integrated branch plan is not strict JSON"
        ) from error
    if canonical_json_bytes(mapping) != content:
        raise AAMASIntegratedBranchIntegrityError(
            "integrated branch plan is not canonical JSON"
        )
    plan = AAMASIntegratedBranchPlan.from_mapping(mapping)
    if plan.sha256 != _sha256(expected_sha256, "expected_plan_sha256"):
        raise AAMASIntegratedBranchIntegrityError(
            "integrated branch plan differs from its external commitment"
        )
    return plan


def _load_canonical_mapping(path: Path, field: str) -> Mapping[str, object]:
    content = path.read_bytes()
    try:
        mapping = read_json_bytes_object(content, source=str(path))
    except Exception as error:
        raise AAMASIntegratedBranchIntegrityError(
            f"integrated branch {field} is not strict JSON"
        ) from error
    if canonical_json_bytes(mapping) != content:
        raise AAMASIntegratedBranchIntegrityError(
            f"integrated branch {field} is not canonical JSON"
        )
    return mapping


def _load_attempt_registry(
    path: Path,
    *,
    plan: AAMASIntegratedBranchPlan,
) -> AAMASIntegratedBranchAttemptRegistry:
    try:
        registry = AAMASIntegratedBranchAttemptRegistry.from_mapping(
            _load_canonical_mapping(path, "attempt registry")
        )
    except AAMASIntegratedBranchSchemaError as error:
        raise AAMASIntegratedBranchIntegrityError(
            "integrated branch attempt registry failed typed validation"
        ) from error
    config_sha256 = integrated_synthetic_market_config().sha256
    if (
        registry.benchmark_id != plan.benchmark_id
        or registry.plan_sha256 != plan.sha256
        or registry.config_sha256 != config_sha256
        or registry.expected_repetitions != plan.repetitions
    ):
        raise AAMASIntegratedBranchIntegrityError(
            "integrated branch attempt registry differs from frozen execution inputs"
        )
    for attempt in registry.attempts:
        if attempt.event_key != _event_key(plan, attempt.replicate_id).value:
            raise AAMASIntegratedBranchIntegrityError(
                "integrated branch attempt event key differs from replay"
            )
    return registry


def audit_written_aamas_integrated_branch(
    output_dir: Path,
    *,
    expected_plan_sha256: str,
    expected_artifact_sha256: str,
    allow_formal_scale: bool = False,
) -> AAMASIntegratedBranchAudit:
    """Replay a fixed-membership bundle from an external plan/root commitment."""
    if not isinstance(output_dir, Path):
        raise TypeError("output_dir must be a Path")
    expected_root = _sha256(expected_artifact_sha256, "expected_artifact_sha256")
    plan = _load_canonical_plan(output_dir / "plan.json", expected_plan_sha256)
    expected = run_aamas_integrated_branch(
        plan, allow_formal_scale=allow_formal_scale
    )
    expected_registry = expected.attempt_registry
    members = _expected_members(expected)
    _audit_membership(output_dir, members)
    observed_registry = _load_attempt_registry(
        output_dir / "attempts.json", plan=plan
    )
    if (
        observed_registry.terminal_status != "completed"
        or observed_registry.canonical_bytes != expected_registry.canonical_bytes
    ):
        raise AAMASIntegratedBranchIntegrityError(
            "integrated branch success attempt registry differs from replay"
        )
    for relative, content in expected.children:
        path = output_dir / PurePosixPath(relative)
        if path.read_bytes() != content or sha256_file(path) != sha256_bytes(content):
            raise AAMASIntegratedBranchIntegrityError(
                f"integrated branch child differs from replay: {relative}"
            )
    artifact_path = output_dir / "artifact.json"
    if artifact_path.read_bytes() != expected.artifact.canonical_bytes:
        raise AAMASIntegratedBranchIntegrityError(
            "integrated branch root artifact differs from replay"
        )
    if expected.artifact.sha256 != expected_root:
        raise AAMASIntegratedBranchIntegrityError(
            "integrated branch root differs from its external commitment"
        )
    seal = (output_dir / "artifact.sha256").read_text(encoding="utf-8")
    if seal != expected_root + "\n":
        raise AAMASIntegratedBranchIntegrityError(
            "integrated branch root seal is invalid"
        )
    _audit_membership(output_dir, members)
    return AAMASIntegratedBranchAudit(
        plan_sha256=plan.sha256,
        artifact_sha256=expected.artifact.sha256,
        record_count=len(expected.artifact.records),
    )


def _audit_written_aamas_integrated_branch_failure(
    output_dir: Path,
    *,
    expected_plan_sha256: str,
    expected_failure_sha256: str,
    allow_formal_scale: bool,
) -> AAMASIntegratedBranchFailureAudit:
    expected_root = _sha256(expected_failure_sha256, "expected_failure_sha256")
    plan = _load_canonical_plan(output_dir / "plan.json", expected_plan_sha256)
    if plan.is_formal_scale and not allow_formal_scale:
        raise AAMASIntegratedBranchError(
            "formal-scale integrated branch audit requires explicit authorization"
        )
    registry = _load_attempt_registry(output_dir / "attempts.json", plan=plan)
    if (
        registry.terminal_status != "failed"
        or registry.scientifically_complete_for_claim
    ):
        raise AAMASIntegratedBranchIntegrityError(
            "failure bundle attempt registry does not have a failed terminal state"
        )
    completed_attempts = tuple(
        item
        for item in registry.attempts
        if item.status is AAMASIntegratedBranchAttemptStatus.COMPLETED
    )
    expected_members = frozenset(
        {"attempts.json", "failure.json", "failure.sha256", "plan.json"}
        | {cast(str, item.child_path) for item in completed_attempts}
    )
    _audit_membership(output_dir, expected_members)
    replayed_records: list[AAMASIntegratedBranchRecord] = []
    for attempt in completed_attempts:
        try:
            replayed_record, content = _execute_replicate(
                plan, attempt.replicate_id
            )
        except Exception as error:
            raise AAMASIntegratedBranchIntegrityError(
                "a completed failure-bundle row no longer replays"
            ) from error
        if (
            replayed_record.event_key != attempt.event_key
            or replayed_record.child_path != attempt.child_path
            or replayed_record.child_sha256 != attempt.child_sha256
        ):
            raise AAMASIntegratedBranchIntegrityError(
                "completed attempt commitment differs from replay"
            )
        child = output_dir / replayed_record.child_path
        if child.read_bytes() != content or sha256_file(child) != attempt.child_sha256:
            raise AAMASIntegratedBranchIntegrityError(
                "completed failure-bundle child differs from replay: "
                f"{attempt.child_path}"
            )
        replayed_records.append(replayed_record)
    try:
        failure = AAMASIntegratedBranchFailureArtifact.from_mapping(
            _load_canonical_mapping(output_dir / "failure.json", "failure root")
        )
    except AAMASIntegratedBranchSchemaError as error:
        raise AAMASIntegratedBranchIntegrityError(
            "integrated branch failure root failed typed validation"
        ) from error
    if (
        failure.benchmark_id != plan.benchmark_id
        or failure.plan_sha256 != plan.sha256
        or failure.config_sha256 != registry.config_sha256
        or failure.attempt_registry_sha256 != registry.sha256
        or failure.expected_repetitions != plan.repetitions
    ):
        raise AAMASIntegratedBranchIntegrityError(
            "failure root differs from its plan or attempt registry"
        )
    statuses = tuple(item.status for item in registry.attempts)
    counts = (
        statuses.count(AAMASIntegratedBranchAttemptStatus.COMPLETED),
        statuses.count(AAMASIntegratedBranchAttemptStatus.FAILED),
        statuses.count(AAMASIntegratedBranchAttemptStatus.BLOCKED_UPSTREAM)
        + statuses.count(AAMASIntegratedBranchAttemptStatus.NOT_RUN),
    )
    if counts != (
        failure.completed_count,
        failure.failed_count,
        failure.blocked_count,
    ):
        raise AAMASIntegratedBranchIntegrityError(
            "failure counts differ from the complete attempt registry"
        )
    if failure.failure_code == "replicate_execution_failed":
        if failure.failure_replicate_id != registry.failure_replicate_id:
            raise AAMASIntegratedBranchIntegrityError(
                "failure root identifies the wrong failed replicate"
            )
        failed_attempt = registry.attempts[cast(int, registry.failure_replicate_id)]
        if (
            failure.error_type != failed_attempt.error_type
            or failure.error_summary != failed_attempt.error_summary
            or failure.error_sha256 != failed_attempt.error_sha256
        ):
            raise AAMASIntegratedBranchIntegrityError(
                "failure root diagnostic differs from the failed attempt"
            )
    else:
        branch_count = sum(
            record.restriction_branch_observed for record in replayed_records
        )
        expected_summary = (
            "preregistered minimum branch replicate condition was not met"
        )
        expected_error_sha256 = sha256_bytes(
            canonical_json_bytes(
                {
                    "branch_replicate_count": branch_count,
                    "minimum_branch_replicates": plan.minimum_branch_replicates,
                }
            )
        )
        if (
            branch_count >= plan.minimum_branch_replicates
            or failure.failure_replicate_id is not None
            or failure.error_type != "ScientificConditionFailure"
            or failure.error_summary != expected_summary
            or failure.error_sha256 != expected_error_sha256
        ):
            raise AAMASIntegratedBranchIntegrityError(
                "collection-level branch failure differs from replay"
            )
    if failure.sha256 != expected_root:
        raise AAMASIntegratedBranchIntegrityError(
            "integrated branch failure root differs from its external commitment"
        )
    seal = (output_dir / "failure.sha256").read_text(encoding="utf-8")
    if seal != expected_root + "\n":
        raise AAMASIntegratedBranchIntegrityError(
            "integrated branch failure root seal is invalid"
        )
    _audit_membership(output_dir, expected_members)
    return AAMASIntegratedBranchFailureAudit(
        plan_sha256=plan.sha256,
        failure_sha256=failure.sha256,
        attempt_registry_sha256=registry.sha256,
        attempt_count=len(registry.attempts),
        completed_count=failure.completed_count,
        failed_count=failure.failed_count,
        blocked_count=failure.blocked_count,
    )


def audit_written_aamas_integrated_branch_evidence(
    output_dir: Path,
    *,
    expected_plan_sha256: str,
    expected_terminal_sha256: str,
    allow_formal_scale: bool = False,
) -> AAMASIntegratedBranchEvidenceAudit:
    """Audit either a complete result bundle or terminal failure evidence."""
    if not isinstance(output_dir, Path):
        raise TypeError("output_dir must be a Path")
    if not output_dir.is_dir() or output_dir.is_symlink():
        raise AAMASIntegratedBranchIntegrityError(
            "integrated branch bundle root must be a real directory"
        )
    has_success = (output_dir / "artifact.json").exists()
    has_failure = (output_dir / "failure.json").exists()
    if has_success == has_failure:
        raise AAMASIntegratedBranchIntegrityError(
            "bundle must contain exactly one success or failure terminal root"
        )
    if has_success:
        return audit_written_aamas_integrated_branch(
            output_dir,
            expected_plan_sha256=expected_plan_sha256,
            expected_artifact_sha256=expected_terminal_sha256,
            allow_formal_scale=allow_formal_scale,
        )
    return _audit_written_aamas_integrated_branch_failure(
        output_dir,
        expected_plan_sha256=expected_plan_sha256,
        expected_failure_sha256=expected_terminal_sha256,
        allow_formal_scale=allow_formal_scale,
    )


def write_aamas_integrated_branch_evidence(
    plan: AAMASIntegratedBranchPlan,
    output_dir: Path,
    *,
    allow_formal_scale: bool = False,
) -> AAMASIntegratedBranchEvidenceAudit:
    """Atomically publish either complete output or terminal failure evidence."""
    if not isinstance(output_dir, Path):
        raise TypeError("output_dir must be a Path")
    if output_dir.exists() or output_dir.is_symlink():
        raise AAMASIntegratedBranchIntegrityError(
            "integrated branch output must not already exist"
        )
    run = _capture_aamas_integrated_branch(
        plan, allow_formal_scale=allow_formal_scale
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.with_name(f".{output_dir.name}.{plan.sha256[:16]}.staging")
    if staging.exists() or staging.is_symlink():
        raise AAMASIntegratedBranchIntegrityError(
            "integrated branch staging path already exists"
        )
    try:
        (staging / "runs").mkdir(parents=True)
        write_text_file(
            staging / "plan.json",
            canonical_json_bytes(plan.to_mapping()).decode("utf-8"),
        )
        for relative, content in run.children:
            write_text_file(
                staging / PurePosixPath(relative), content.decode("utf-8")
            )
        registry = run.attempt_registry
        write_text_file(
            staging / "attempts.json", registry.canonical_bytes.decode("utf-8")
        )
        if isinstance(run, AAMASIntegratedBranchRun):
            write_text_file(
                staging / "artifact.json",
                run.artifact.canonical_bytes.decode("utf-8"),
            )
            write_text_file(staging / "artifact.sha256", run.artifact.sha256 + "\n")
        else:
            write_text_file(
                staging / "failure.json",
                run.artifact.canonical_bytes.decode("utf-8"),
            )
            write_text_file(staging / "failure.sha256", run.artifact.sha256 + "\n")
        audit = audit_written_aamas_integrated_branch_evidence(
            staging,
            expected_plan_sha256=plan.sha256,
            expected_terminal_sha256=run.artifact.sha256,
            allow_formal_scale=allow_formal_scale,
        )
        os.replace(staging, output_dir)
        expected_members = (
            _expected_members(run)
            if isinstance(run, AAMASIntegratedBranchRun)
            else _expected_failure_members(run)
        )
        _audit_membership(output_dir, expected_members)
        return audit
    except Exception:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise


def write_aamas_integrated_branch(
    plan: AAMASIntegratedBranchPlan,
    output_dir: Path,
    *,
    allow_formal_scale: bool = False,
) -> AAMASIntegratedBranchAudit:
    """Compatibility success API; failures are published before being raised."""
    receipt = write_aamas_integrated_branch_evidence(
        plan,
        output_dir,
        allow_formal_scale=allow_formal_scale,
    )
    if isinstance(receipt, AAMASIntegratedBranchFailureAudit):
        raise AAMASIntegratedBranchPublishedFailure(output_dir, receipt)
    return receipt


__all__ = [
    "AAMAS_INTEGRATED_BRANCH_ARTIFACT_SCHEMA_VERSION",
    "AAMAS_INTEGRATED_BRANCH_ATTEMPT_REGISTRY_SCHEMA_VERSION",
    "AAMAS_INTEGRATED_BRANCH_CHILD_SCHEMA_VERSION",
    "AAMAS_INTEGRATED_BRANCH_CLAIM_SCOPE",
    "AAMAS_INTEGRATED_BRANCH_FAILURE_SCHEMA_VERSION",
    "AAMAS_INTEGRATED_BRANCH_FORMAL_REPETITIONS",
    "AAMAS_INTEGRATED_BRANCH_FORMAL_STEPS",
    "AAMAS_INTEGRATED_BRANCH_PLAN_SCHEMA_VERSION",
    "AAMASIntegratedBranchArtifact",
    "AAMASIntegratedBranchAudit",
    "AAMASIntegratedBranchError",
    "AAMASIntegratedBranchAttempt",
    "AAMASIntegratedBranchAttemptRegistry",
    "AAMASIntegratedBranchAttemptStatus",
    "AAMASIntegratedBranchEvidenceAudit",
    "AAMASIntegratedBranchExecutionError",
    "AAMASIntegratedBranchFailureArtifact",
    "AAMASIntegratedBranchFailureAudit",
    "AAMASIntegratedBranchFailureRun",
    "AAMASIntegratedBranchIntegrityError",
    "AAMASIntegratedBranchPlan",
    "AAMASIntegratedBranchPublishedFailure",
    "AAMASIntegratedBranchRecord",
    "AAMASIntegratedBranchRun",
    "AAMASIntegratedBranchSchemaError",
    "audit_written_aamas_integrated_branch",
    "audit_written_aamas_integrated_branch_evidence",
    "run_aamas_integrated_branch",
    "write_aamas_integrated_branch",
    "write_aamas_integrated_branch_evidence",
]
