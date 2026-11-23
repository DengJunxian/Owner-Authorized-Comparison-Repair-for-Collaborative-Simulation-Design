"""Auditable phase-one runner for the synthetic AAMAS benchmark.

The phase-one artifact executes only the preregistered RNG-coupling smoke
cells and the integrated mechanism-ablation grid.  Calibration recovery and
selective-validity evaluation remain explicit blocked attempts.  Keeping them
in the expected-attempt registry prevents an incomplete benchmark from being
misrepresented as a completed three-question study.

Operational wall-clock measurements are written beside, but never included in,
the scientific artifact hash.  Requested resources and every failed or blocked
attempt remain part of the deterministic scientific record.
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Literal, NoReturn, Self, cast

from policytwin.counterfactual import EventKey
from policytwin.experiments.matrix import ResourceBudget
from policytwin.experiments.mechanism_ablations import (
    MECHANISM_ABLATION_CLAIM_SCOPE,
    MechanismAblationArtifact,
    MechanismAblationSpec,
    audit_written_mechanism_ablation_artifact,
    build_mechanism_ablation_protocol,
    issue_mechanism_ablation_authorization,
    run_mechanism_ablations,
    write_mechanism_ablation_artifact,
)
from policytwin.experiments.mechanism_smoke import (
    MECHANISM_CLAIM_SCOPE,
    MechanismSmokeArtifact,
    MechanismSmokeSpec,
    audit_mechanism_smoke_artifact,
    run_mechanism_smoke,
    write_mechanism_smoke_artifact,
)
from policytwin.provenance import (
    JsonlEventWriter,
    MetadataValidationError,
    canonical_json_bytes,
    read_json_object,
    read_jsonl_objects,
    sha256_bytes,
    write_json_file,
    write_text_file,
)

AAMAS_BENCHMARK_PLAN_SCHEMA_VERSION = "policytwin.aamas-benchmark-plan.v1"
AAMAS_BENCHMARK_ARTIFACT_SCHEMA_VERSION = "policytwin.aamas-benchmark.v1"
AAMAS_BENCHMARK_LOG_SCHEMA_VERSION = "policytwin.aamas-benchmark-log.v1"
AAMAS_BENCHMARK_CLAIM_SCOPE = "synthetic_rq1_mechanism_execution_only"
BLOCKED_PENDING_IMPLEMENTATION = "blocked_pending_implementation"

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EVENT_KEY = re.compile(r"^ptk1:[0-9a-f]{64}$")


class AAMASBenchmarkError(RuntimeError):
    """Base class for phase-one benchmark failures."""


class AAMASBenchmarkSchemaError(AAMASBenchmarkError, ValueError):
    """Raised when a plan, attempt, or artifact has an invalid schema."""


class AAMASBenchmarkIntegrityError(AAMASBenchmarkError):
    """Raised when a written benchmark does not reproduce from its plan."""


def _fail(message: str) -> NoReturn:
    raise AAMASBenchmarkSchemaError(message)


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


def _array(value: object, field: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        _fail(f"{field} must be an array")
    return cast(Sequence[object], value)


def _strict_fields(
    value: Mapping[str, object], expected: frozenset[str], context: str
) -> None:
    if frozenset(value) != expected:
        _fail(f"{context} has missing or unknown fields")


def _commit(value: object) -> str:
    return sha256_bytes(canonical_json_bytes(value))


class AAMASResearchQuestion(StrEnum):
    """Closed research-question registry for benchmark version one."""

    RQ1 = "rq1_random_coupling_and_mechanisms"
    RQ2 = "rq2_calibration_recovery"
    RQ3 = "rq3_selective_validity"


class AAMASAttemptKind(StrEnum):
    """Executable and deliberately blocked phase-one attempt kinds."""

    RNG_MECHANISM_SMOKE = "rng_mechanism_smoke"
    MECHANISM_ABLATION = "mechanism_ablation"
    CALIBRATION_RECOVERY = "calibration_recovery"
    SELECTIVE_VALIDITY = "selective_validity"


class AAMASAttemptStatus(StrEnum):
    """Terminal statuses; failed and blocked attempts are never missing data."""

    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED_PENDING_IMPLEMENTATION = BLOCKED_PENDING_IMPLEMENTATION


class AAMASSourceKind(StrEnum):
    """Typed scientific sub-artifacts accepted by the phase-one bundle."""

    RNG_MECHANISM_SMOKE = "rng_mechanism_smoke"
    MECHANISM_ABLATION = "mechanism_ablation"


class AAMASBenchmarkStatus(StrEnum):
    """Phase-one status without claiming completion of RQ2 or RQ3."""

    RQ1_COMPLETE_LATER_PHASES_BLOCKED = "rq1_complete_later_phases_blocked"
    INCOMPLETE = "incomplete"


@dataclass(frozen=True, slots=True)
class RNGStressCell:
    """One preregistered branch-location and branch-consumption stress cell."""

    cell_id: str
    branch_step: int
    branch_draw_count: int

    def __post_init__(self) -> None:
        _identifier(self.cell_id, "RNG stress cell_id")
        _integer(self.branch_step, "branch_step")
        _integer(self.branch_draw_count, "branch_draw_count", minimum=1)

    def to_mapping(self) -> dict[str, object]:
        return {
            "branch_draw_count": self.branch_draw_count,
            "branch_step": self.branch_step,
            "cell_id": self.cell_id,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(
            value,
            frozenset({"branch_draw_count", "branch_step", "cell_id"}),
            "RNG stress cell",
        )
        return cls(
            cell_id=_identifier(value["cell_id"], "cell_id"),
            branch_step=_integer(value["branch_step"], "branch_step"),
            branch_draw_count=_integer(
                value["branch_draw_count"], "branch_draw_count", minimum=1
            ),
        )


_FORMAL_RNG_CELLS = (
    RNGStressCell("early_light", branch_step=4, branch_draw_count=1),
    RNGStressCell("early_heavy", branch_step=4, branch_draw_count=5),
    RNGStressCell("late_light", branch_step=24, branch_draw_count=1),
    RNGStressCell("late_heavy", branch_step=24, branch_draw_count=5),
)


@dataclass(frozen=True, slots=True)
class ExpectedAAMASAttempt:
    """One exact attempt required by the frozen benchmark plan."""

    attempt_id: str
    research_question: AAMASResearchQuestion
    kind: AAMASAttemptKind
    config_sha256: str
    execution_expected: bool

    def __post_init__(self) -> None:
        _identifier(self.attempt_id, "expected attempt_id")
        if not isinstance(self.research_question, AAMASResearchQuestion):
            _fail("expected attempt research_question is invalid")
        if not isinstance(self.kind, AAMASAttemptKind):
            _fail("expected attempt kind is invalid")
        _sha256(self.config_sha256, "expected attempt config_sha256")
        if not isinstance(self.execution_expected, bool):
            _fail("execution_expected must be boolean")
        should_execute = self.research_question is AAMASResearchQuestion.RQ1
        if self.execution_expected is not should_execute:
            _fail("only RQ1 attempts may execute in benchmark phase one")

    def to_mapping(self) -> dict[str, object]:
        return {
            "attempt_id": self.attempt_id,
            "config_sha256": self.config_sha256,
            "execution_expected": self.execution_expected,
            "kind": self.kind.value,
            "research_question": self.research_question.value,
        }


def _blocked_config_sha256(kind: AAMASAttemptKind) -> str:
    return _commit(
        {
            "implementation_status": BLOCKED_PENDING_IMPLEMENTATION,
            "kind": kind.value,
            "schema_version": AAMAS_BENCHMARK_PLAN_SCHEMA_VERSION,
            "synthetic_only": True,
        }
    )


@dataclass(frozen=True, slots=True)
class AAMASBenchmarkPlan:
    """Immutable phase-one plan; defaults are the unexecuted formal scale."""

    benchmark_id: str = "aamas-synthetic-benchmark-v1"
    seed: str = "policytwin-aamas-synthetic-v1"
    rng_cells: tuple[RNGStressCell, ...] = _FORMAL_RNG_CELLS
    rng_replicate_count: int = 256
    rng_step_count: int = 32
    rng_policy_effect: int = 7
    mechanism_replicate_count: int = 64
    mechanism_step_count: int = 8
    resources: ResourceBudget = ResourceBudget(
        cpu_cores=1,
        memory_mb=2048,
        wall_time_seconds=900,
    )
    synthetic_only: Literal[True] = True
    schema_version: str = AAMAS_BENCHMARK_PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _identifier(self.benchmark_id, "benchmark_id")
        if not isinstance(self.seed, str) or not self.seed or len(self.seed) > 512:
            _fail("seed must contain between one and 512 characters")
        if (
            not isinstance(self.rng_cells, tuple)
            or len(self.rng_cells) != 4
            or not all(isinstance(item, RNGStressCell) for item in self.rng_cells)
        ):
            _fail("rng_cells must contain exactly four immutable stress cells")
        cell_ids = tuple(item.cell_id for item in self.rng_cells)
        if len(set(cell_ids)) != len(cell_ids):
            _fail("RNG stress cell IDs must be unique")
        _integer(self.rng_replicate_count, "rng_replicate_count", minimum=2)
        _integer(self.rng_step_count, "rng_step_count", minimum=2)
        if any(item.branch_step >= self.rng_step_count - 1 for item in self.rng_cells):
            _fail("every RNG branch must leave a later exogenous event")
        if isinstance(self.rng_policy_effect, bool) or not isinstance(
            self.rng_policy_effect, int
        ):
            _fail("rng_policy_effect must be an integer")
        _integer(
            self.mechanism_replicate_count,
            "mechanism_replicate_count",
            minimum=1,
        )
        _integer(self.mechanism_step_count, "mechanism_step_count", minimum=2)
        if not isinstance(self.resources, ResourceBudget):
            _fail("resources must be a ResourceBudget")
        if self.synthetic_only is not True:
            _fail("AAMAS phase-one benchmark must remain synthetic_only")
        if self.schema_version != AAMAS_BENCHMARK_PLAN_SCHEMA_VERSION:
            _fail("unsupported AAMAS benchmark plan schema")
        # Materialization validates composed identifiers and all delegated specs.
        if len(self.rng_specs) != len(self.rng_cells):
            _fail("RNG specification materialization changed cell cardinality")
        if (
            self.mechanism_ablation_spec.replicate_count
            != self.mechanism_replicate_count
        ):
            _fail("mechanism specification materialization changed replicate count")
        if len(self.expected_attempts) != 7:
            _fail("phase-one expected-attempt registry changed cardinality")

    @property
    def rng_specs(self) -> tuple[MechanismSmokeSpec, ...]:
        """Build four stress cells over one shared cross-cell exogenous tape."""

        return tuple(
            MechanismSmokeSpec(
                scenario_id=f"{self.benchmark_id}.rng.shared",
                seed=f"{self.seed}:rng:shared",
                replicate_count=self.rng_replicate_count,
                step_count=self.rng_step_count,
                branch_step=cell.branch_step,
                branch_draw_count=cell.branch_draw_count,
                policy_effect=self.rng_policy_effect,
            )
            for cell in self.rng_cells
        )

    @property
    def mechanism_ablation_spec(self) -> MechanismAblationSpec:
        """Build the exact integrated mechanism-ablation specification."""

        return MechanismAblationSpec(
            root_event_key=EventKey(
                experiment_id=f"{self.benchmark_id}.mechanism",
                seed=f"{self.seed}:mechanism",
                replicate_id=0,
                simulation_step=0,
                purpose="aamas.phase1.mechanism_ablation",
            ),
            replicate_count=self.mechanism_replicate_count,
            step_count=self.mechanism_step_count,
        )

    @property
    def expected_attempts(self) -> tuple[ExpectedAAMASAttempt, ...]:
        """Return the complete phase-one registry, including blocked work."""

        rng = tuple(
            ExpectedAAMASAttempt(
                attempt_id=f"rq1_rng_{cell.cell_id}",
                research_question=AAMASResearchQuestion.RQ1,
                kind=AAMASAttemptKind.RNG_MECHANISM_SMOKE,
                config_sha256=spec.sha256,
                execution_expected=True,
            )
            for cell, spec in zip(self.rng_cells, self.rng_specs, strict=True)
        )
        return (
            *rng,
            ExpectedAAMASAttempt(
                attempt_id="rq1_mechanism_ablations",
                research_question=AAMASResearchQuestion.RQ1,
                kind=AAMASAttemptKind.MECHANISM_ABLATION,
                config_sha256=self.mechanism_ablation_spec.sha256,
                execution_expected=True,
            ),
            ExpectedAAMASAttempt(
                attempt_id="rq2_calibration_recovery",
                research_question=AAMASResearchQuestion.RQ2,
                kind=AAMASAttemptKind.CALIBRATION_RECOVERY,
                config_sha256=_blocked_config_sha256(
                    AAMASAttemptKind.CALIBRATION_RECOVERY
                ),
                execution_expected=False,
            ),
            ExpectedAAMASAttempt(
                attempt_id="rq3_selective_validity",
                research_question=AAMASResearchQuestion.RQ3,
                kind=AAMASAttemptKind.SELECTIVE_VALIDITY,
                config_sha256=_blocked_config_sha256(
                    AAMASAttemptKind.SELECTIVE_VALIDITY
                ),
                execution_expected=False,
            ),
        )

    @property
    def is_formal_scale(self) -> bool:
        """Whether this plan exactly matches the unexecuted formal defaults."""

        return (
            self.rng_cells == _FORMAL_RNG_CELLS
            and self.rng_replicate_count == 256
            and self.rng_step_count == 32
            and self.rng_policy_effect == 7
            and self.mechanism_replicate_count == 64
            and self.mechanism_step_count == 8
        )

    @property
    def sha256(self) -> str:
        return _commit(self.to_mapping())

    def to_mapping(self) -> dict[str, object]:
        return {
            "benchmark_id": self.benchmark_id,
            "expected_attempts": [item.to_mapping() for item in self.expected_attempts],
            "mechanism_replicate_count": self.mechanism_replicate_count,
            "mechanism_step_count": self.mechanism_step_count,
            "resources": self.resources.to_mapping(),
            "rng_cells": [item.to_mapping() for item in self.rng_cells],
            "rng_policy_effect": self.rng_policy_effect,
            "rng_replicate_count": self.rng_replicate_count,
            "rng_step_count": self.rng_step_count,
            "schema_version": self.schema_version,
            "seed": self.seed,
            "synthetic_only": self.synthetic_only,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(
            value,
            frozenset(
                {
                    "benchmark_id",
                    "expected_attempts",
                    "mechanism_replicate_count",
                    "mechanism_step_count",
                    "resources",
                    "rng_cells",
                    "rng_policy_effect",
                    "rng_replicate_count",
                    "rng_step_count",
                    "schema_version",
                    "seed",
                    "synthetic_only",
                }
            ),
            "AAMAS benchmark plan",
        )
        if value["synthetic_only"] is not True:
            _fail("AAMAS benchmark plan must remain synthetic_only")
        seed = value["seed"]
        schema_version = value["schema_version"]
        if not isinstance(seed, str) or not isinstance(schema_version, str):
            _fail("plan seed and schema_version must be strings")
        plan = cls(
            benchmark_id=_identifier(value["benchmark_id"], "benchmark_id"),
            seed=seed,
            rng_cells=tuple(
                RNGStressCell.from_mapping(_mapping(item, "rng_cells item"))
                for item in _array(value["rng_cells"], "rng_cells")
            ),
            rng_replicate_count=_integer(
                value["rng_replicate_count"], "rng_replicate_count", minimum=2
            ),
            rng_step_count=_integer(
                value["rng_step_count"], "rng_step_count", minimum=2
            ),
            rng_policy_effect=_integer(
                value["rng_policy_effect"],
                "rng_policy_effect",
                minimum=-(1 << 63),
            ),
            mechanism_replicate_count=_integer(
                value["mechanism_replicate_count"],
                "mechanism_replicate_count",
                minimum=1,
            ),
            mechanism_step_count=_integer(
                value["mechanism_step_count"],
                "mechanism_step_count",
                minimum=2,
            ),
            resources=ResourceBudget.from_mapping(
                _mapping(value["resources"], "resources")
            ),
            synthetic_only=True,
            schema_version=schema_version,
        )
        raw_registry = tuple(
            dict(_mapping(item, "expected_attempts item"))
            for item in _array(value["expected_attempts"], "expected_attempts")
        )
        if raw_registry != tuple(item.to_mapping() for item in plan.expected_attempts):
            _fail("serialized expected-attempt registry differs from the frozen plan")
        if plan.to_mapping() != dict(value):
            _fail("AAMAS benchmark plan is not in canonical serialized form")
        return plan


@dataclass(frozen=True, slots=True)
class AAMASAttempt:
    """One terminal attempt with resources and any failure retained."""

    attempt_id: str
    research_question: AAMASResearchQuestion
    kind: AAMASAttemptKind
    config_sha256: str
    status: AAMASAttemptStatus
    requested_resources: ResourceBudget
    artifact_sha256: str | None
    completed_subattempts: int
    failed_subattempts: int
    failure_code: str | None = None
    failure_message: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.attempt_id, "attempt_id")
        if not isinstance(self.research_question, AAMASResearchQuestion):
            _fail("attempt research_question is invalid")
        if not isinstance(self.kind, AAMASAttemptKind):
            _fail("attempt kind is invalid")
        _sha256(self.config_sha256, "attempt config_sha256")
        if not isinstance(self.status, AAMASAttemptStatus):
            _fail("attempt status is invalid")
        if not isinstance(self.requested_resources, ResourceBudget):
            _fail("attempt requested_resources is invalid")
        if self.artifact_sha256 is not None:
            _sha256(self.artifact_sha256, "attempt artifact_sha256")
        completed = _integer(
            self.completed_subattempts, "completed_subattempts", minimum=0
        )
        failed = _integer(self.failed_subattempts, "failed_subattempts", minimum=0)
        if self.status is AAMASAttemptStatus.COMPLETED:
            if (
                self.artifact_sha256 is None
                or completed < 1
                or failed != 0
                or self.failure_code is not None
                or self.failure_message is not None
            ):
                _fail("completed attempt has inconsistent artifact or failure fields")
        elif self.status is AAMASAttemptStatus.FAILED:
            if failed < 1 or self.failure_code is None or self.failure_message is None:
                _fail("failed attempt must retain a failure and failed subattempt")
            _identifier(self.failure_code, "failure_code")
            if not self.failure_message.strip():
                _fail("failure_message must be non-empty")
        else:
            if (
                self.artifact_sha256 is not None
                or completed != 0
                or failed != 0
                or self.failure_code != BLOCKED_PENDING_IMPLEMENTATION
                or self.failure_message is None
                or not self.failure_message.strip()
            ):
                _fail("blocked attempt must retain the exact blocked contract")

    def to_mapping(self) -> dict[str, object]:
        return {
            "artifact_sha256": self.artifact_sha256,
            "attempt_id": self.attempt_id,
            "completed_subattempts": self.completed_subattempts,
            "config_sha256": self.config_sha256,
            "failed_subattempts": self.failed_subattempts,
            "failure_code": self.failure_code,
            "failure_message": self.failure_message,
            "kind": self.kind.value,
            "requested_resources": self.requested_resources.to_mapping(),
            "research_question": self.research_question.value,
            "status": self.status.value,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(
            value,
            frozenset(
                {
                    "artifact_sha256",
                    "attempt_id",
                    "completed_subattempts",
                    "config_sha256",
                    "failed_subattempts",
                    "failure_code",
                    "failure_message",
                    "kind",
                    "requested_resources",
                    "research_question",
                    "status",
                }
            ),
            "AAMAS attempt",
        )
        raw_rq = value["research_question"]
        raw_kind = value["kind"]
        raw_status = value["status"]
        if not all(isinstance(item, str) for item in (raw_rq, raw_kind, raw_status)):
            _fail("attempt enum values must be strings")
        try:
            research_question = AAMASResearchQuestion(cast(str, raw_rq))
            kind = AAMASAttemptKind(cast(str, raw_kind))
            status = AAMASAttemptStatus(cast(str, raw_status))
        except ValueError as error:
            raise AAMASBenchmarkSchemaError(
                "attempt enum value is unsupported"
            ) from error
        artifact = value["artifact_sha256"]
        failure_code = value["failure_code"]
        failure_message = value["failure_message"]
        if artifact is not None and not isinstance(artifact, str):
            _fail("artifact_sha256 must be a string or null")
        if failure_code is not None and not isinstance(failure_code, str):
            _fail("failure_code must be a string or null")
        if failure_message is not None and not isinstance(failure_message, str):
            _fail("failure_message must be a string or null")
        return cls(
            attempt_id=_identifier(value["attempt_id"], "attempt_id"),
            research_question=research_question,
            kind=kind,
            config_sha256=_sha256(value["config_sha256"], "config_sha256"),
            status=status,
            requested_resources=ResourceBudget.from_mapping(
                _mapping(value["requested_resources"], "requested_resources")
            ),
            artifact_sha256=artifact,
            completed_subattempts=_integer(
                value["completed_subattempts"], "completed_subattempts"
            ),
            failed_subattempts=_integer(
                value["failed_subattempts"], "failed_subattempts"
            ),
            failure_code=failure_code,
            failure_message=failure_message,
        )


@dataclass(frozen=True, slots=True)
class AAMASSourceArtifact:
    """One nested mechanism artifact committed by the benchmark result."""

    attempt_id: str
    kind: AAMASSourceKind
    relative_path: str
    semantic_sha256: str
    config_sha256: str
    claim_scope: str
    protocol_execution_receipt_sha256: str | None
    mechanism_execution: Literal[True] = True

    def __post_init__(self) -> None:
        _identifier(self.attempt_id, "source attempt_id")
        if not isinstance(self.kind, AAMASSourceKind):
            _fail("source artifact kind is invalid")
        path = PurePosixPath(self.relative_path)
        if (
            not self.relative_path
            or path.is_absolute()
            or path.as_posix() != self.relative_path
            or "\\" in self.relative_path
            or any(part in {"", ".", ".."} for part in path.parts)
            or path.suffix != ".json"
        ):
            _fail("source relative_path must be a safe JSON bundle path")
        _sha256(self.semantic_sha256, "source semantic_sha256")
        _sha256(self.config_sha256, "source config_sha256")
        _identifier(self.claim_scope, "source claim_scope")
        if self.kind is AAMASSourceKind.RNG_MECHANISM_SMOKE:
            if (
                self.claim_scope != MECHANISM_CLAIM_SCOPE
                or self.protocol_execution_receipt_sha256 is not None
            ):
                _fail("RNG source has invalid scope or protocol receipt")
        else:
            if self.claim_scope != MECHANISM_ABLATION_CLAIM_SCOPE:
                _fail("mechanism-ablation source claim scope changed")
            _sha256(
                self.protocol_execution_receipt_sha256,
                "protocol_execution_receipt_sha256",
            )
        if self.mechanism_execution is not True:
            _fail("source artifact must prove mechanism execution")

    def to_mapping(self) -> dict[str, object]:
        return {
            "attempt_id": self.attempt_id,
            "claim_scope": self.claim_scope,
            "config_sha256": self.config_sha256,
            "kind": self.kind.value,
            "mechanism_execution": self.mechanism_execution,
            "protocol_execution_receipt_sha256": (
                self.protocol_execution_receipt_sha256
            ),
            "relative_path": self.relative_path,
            "semantic_sha256": self.semantic_sha256,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(
            value,
            frozenset(
                {
                    "attempt_id",
                    "claim_scope",
                    "config_sha256",
                    "kind",
                    "mechanism_execution",
                    "protocol_execution_receipt_sha256",
                    "relative_path",
                    "semantic_sha256",
                }
            ),
            "AAMAS source artifact",
        )
        raw_kind = value["kind"]
        if not isinstance(raw_kind, str):
            _fail("source kind must be a string")
        try:
            kind = AAMASSourceKind(raw_kind)
        except ValueError as error:
            raise AAMASBenchmarkSchemaError("source kind is unsupported") from error
        path = value["relative_path"]
        claim_scope = value["claim_scope"]
        receipt = value["protocol_execution_receipt_sha256"]
        mechanism = value["mechanism_execution"]
        if not isinstance(path, str) or not isinstance(claim_scope, str):
            _fail("source path and claim scope must be strings")
        if receipt is not None and not isinstance(receipt, str):
            _fail("source protocol receipt hash must be a string or null")
        if mechanism is not True:
            _fail("source mechanism_execution must be true")
        return cls(
            attempt_id=_identifier(value["attempt_id"], "attempt_id"),
            kind=kind,
            relative_path=path,
            semantic_sha256=_sha256(value["semantic_sha256"], "semantic_sha256"),
            config_sha256=_sha256(value["config_sha256"], "config_sha256"),
            claim_scope=claim_scope,
            protocol_execution_receipt_sha256=receipt,
            mechanism_execution=True,
        )


def _derived_status(attempts: tuple[AAMASAttempt, ...]) -> AAMASBenchmarkStatus:
    rq1 = tuple(
        item for item in attempts if item.research_question is AAMASResearchQuestion.RQ1
    )
    if rq1 and all(item.status is AAMASAttemptStatus.COMPLETED for item in rq1):
        return AAMASBenchmarkStatus.RQ1_COMPLETE_LATER_PHASES_BLOCKED
    return AAMASBenchmarkStatus.INCOMPLETE


@dataclass(frozen=True, slots=True)
class AAMASBenchmarkArtifact:
    """Scientific phase-one root; wall-clock timing is deliberately absent."""

    plan: AAMASBenchmarkPlan
    attempts: tuple[AAMASAttempt, ...]
    source_artifacts: tuple[AAMASSourceArtifact, ...]
    status: AAMASBenchmarkStatus
    claim_scope: str = AAMAS_BENCHMARK_CLAIM_SCOPE
    synthetic_only: Literal[True] = True
    schema_version: str = AAMAS_BENCHMARK_ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.plan, AAMASBenchmarkPlan):
            _fail("artifact plan is invalid")
        if not isinstance(self.attempts, tuple) or not all(
            isinstance(item, AAMASAttempt) for item in self.attempts
        ):
            _fail("artifact attempts must be an immutable typed tuple")
        expected = self.plan.expected_attempts
        if tuple(item.attempt_id for item in self.attempts) != tuple(
            item.attempt_id for item in expected
        ):
            _fail("artifact omitted, duplicated, added, or reordered an attempt")
        for attempt, registered in zip(self.attempts, expected, strict=True):
            if (
                attempt.research_question is not registered.research_question
                or attempt.kind is not registered.kind
                or attempt.config_sha256 != registered.config_sha256
                or attempt.requested_resources != self.plan.resources
            ):
                _fail("attempt differs from its frozen registry entry")
            if registered.execution_expected:
                if attempt.status is AAMASAttemptStatus.BLOCKED_PENDING_IMPLEMENTATION:
                    _fail("an executable RQ1 attempt was silently marked blocked")
            elif (
                attempt.status is not AAMASAttemptStatus.BLOCKED_PENDING_IMPLEMENTATION
            ):
                _fail("RQ2/RQ3 cannot be promoted before their implementation exists")
        if not isinstance(self.source_artifacts, tuple) or not all(
            isinstance(item, AAMASSourceArtifact) for item in self.source_artifacts
        ):
            _fail("source_artifacts must be an immutable typed tuple")
        artifact_attempts = tuple(
            item for item in self.attempts if item.artifact_sha256 is not None
        )
        if tuple(item.attempt_id for item in self.source_artifacts) != tuple(
            item.attempt_id for item in artifact_attempts
        ):
            _fail("source artifacts do not exactly match artifact-bearing attempts")
        for source, attempt in zip(
            self.source_artifacts, artifact_attempts, strict=True
        ):
            registered = next(
                item for item in expected if item.attempt_id == attempt.attempt_id
            )
            expected_kind = (
                AAMASSourceKind.RNG_MECHANISM_SMOKE
                if attempt.kind is AAMASAttemptKind.RNG_MECHANISM_SMOKE
                else AAMASSourceKind.MECHANISM_ABLATION
            )
            if (
                source.kind is not expected_kind
                or source.relative_path != _source_path(registered)
                or source.semantic_sha256 != attempt.artifact_sha256
                or source.config_sha256 != attempt.config_sha256
            ):
                _fail("source artifact differs from its terminal attempt")
        if not isinstance(self.status, AAMASBenchmarkStatus):
            _fail("artifact status is invalid")
        if self.status is not _derived_status(self.attempts):
            _fail("artifact status does not derive from the complete attempt ledger")
        if self.claim_scope != AAMAS_BENCHMARK_CLAIM_SCOPE:
            _fail("phase-one claim scope cannot be widened")
        if self.synthetic_only is not True:
            _fail("AAMAS benchmark artifact must remain synthetic_only")
        if self.schema_version != AAMAS_BENCHMARK_ARTIFACT_SCHEMA_VERSION:
            _fail("unsupported AAMAS benchmark artifact schema")

    @property
    def completed_count(self) -> int:
        return sum(
            item.status is AAMASAttemptStatus.COMPLETED for item in self.attempts
        )

    @property
    def failed_count(self) -> int:
        return sum(item.status is AAMASAttemptStatus.FAILED for item in self.attempts)

    @property
    def blocked_count(self) -> int:
        return sum(
            item.status is AAMASAttemptStatus.BLOCKED_PENDING_IMPLEMENTATION
            for item in self.attempts
        )

    @property
    def sha256(self) -> str:
        return _commit(self.to_mapping())

    def to_mapping(self) -> dict[str, object]:
        return {
            "attempts": [item.to_mapping() for item in self.attempts],
            "blocked_count": self.blocked_count,
            "claim_scope": self.claim_scope,
            "completed_count": self.completed_count,
            "failed_count": self.failed_count,
            "plan": self.plan.to_mapping(),
            "plan_sha256": self.plan.sha256,
            "schema_version": self.schema_version,
            "source_artifacts": [item.to_mapping() for item in self.source_artifacts],
            "status": self.status.value,
            "synthetic_only": self.synthetic_only,
            "timing_in_scientific_hash": False,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(
            value,
            frozenset(
                {
                    "attempts",
                    "blocked_count",
                    "claim_scope",
                    "completed_count",
                    "failed_count",
                    "plan",
                    "plan_sha256",
                    "schema_version",
                    "source_artifacts",
                    "status",
                    "synthetic_only",
                    "timing_in_scientific_hash",
                }
            ),
            "AAMAS benchmark artifact",
        )
        if value["synthetic_only"] is not True:
            _fail("AAMAS artifact must remain synthetic_only")
        if value["timing_in_scientific_hash"] is not False:
            _fail("operational timing cannot enter the scientific hash")
        raw_status = value["status"]
        claim_scope = value["claim_scope"]
        schema_version = value["schema_version"]
        if not all(
            isinstance(item, str) for item in (raw_status, claim_scope, schema_version)
        ):
            _fail("artifact enum and schema fields must be strings")
        try:
            status = AAMASBenchmarkStatus(cast(str, raw_status))
        except ValueError as error:
            raise AAMASBenchmarkSchemaError("artifact status is unsupported") from error
        artifact = cls(
            plan=AAMASBenchmarkPlan.from_mapping(_mapping(value["plan"], "plan")),
            attempts=tuple(
                AAMASAttempt.from_mapping(_mapping(item, "attempts item"))
                for item in _array(value["attempts"], "attempts")
            ),
            source_artifacts=tuple(
                AAMASSourceArtifact.from_mapping(
                    _mapping(item, "source_artifacts item")
                )
                for item in _array(value["source_artifacts"], "source_artifacts")
            ),
            status=status,
            claim_scope=cast(str, claim_scope),
            synthetic_only=True,
            schema_version=cast(str, schema_version),
        )
        if artifact.plan.sha256 != _sha256(value["plan_sha256"], "plan_sha256"):
            _fail("artifact plan_sha256 does not match its retained plan")
        for field, observed in (
            ("completed_count", artifact.completed_count),
            ("failed_count", artifact.failed_count),
            ("blocked_count", artifact.blocked_count),
        ):
            if _integer(value[field], field) != observed:
                _fail(f"artifact {field} does not match attempts")
        if artifact.to_mapping() != dict(value):
            _fail("AAMAS benchmark artifact is not in canonical serialized form")
        return artifact


@dataclass(frozen=True, slots=True)
class AAMASOperationalTiming:
    """Non-scientific wall time retained without changing result identity."""

    attempt_id: str
    event_key: str
    wall_seconds: float

    def __post_init__(self) -> None:
        _identifier(self.attempt_id, "timing attempt_id")
        if (
            not isinstance(self.event_key, str)
            or _EVENT_KEY.fullmatch(self.event_key) is None
        ):
            _fail("timing event_key is invalid")
        elapsed = _number(self.wall_seconds, "wall_seconds")
        if elapsed < 0.0:
            _fail("wall_seconds must be non-negative")
        object.__setattr__(self, "wall_seconds", elapsed)

    def to_mapping(self) -> dict[str, object]:
        return {
            "attempt_id": self.attempt_id,
            "event": "observed",
            "event_key": self.event_key,
            "measurement_scope": "operational_non_scientific_process",
            "schema_version": AAMAS_BENCHMARK_LOG_SCHEMA_VERSION,
            "wall_seconds": self.wall_seconds,
        }


@dataclass(frozen=True, slots=True)
class AAMASBenchmarkRun:
    """In-memory result plus typed nested artifacts and operational timing."""

    artifact: AAMASBenchmarkArtifact
    rng_artifacts: tuple[tuple[str, MechanismSmokeArtifact], ...]
    mechanism_ablation_artifact: MechanismAblationArtifact | None
    timings: tuple[AAMASOperationalTiming, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, AAMASBenchmarkArtifact):
            _fail("run artifact is invalid")
        if not isinstance(self.rng_artifacts, tuple) or not all(
            isinstance(item, tuple)
            and len(item) == 2
            and isinstance(item[0], str)
            and isinstance(item[1], MechanismSmokeArtifact)
            for item in self.rng_artifacts
        ):
            _fail("run rng_artifacts are invalid")
        source_by_attempt = {
            item.attempt_id: item for item in self.artifact.source_artifacts
        }
        expected_rng_ids = tuple(
            item.attempt_id
            for item in self.artifact.source_artifacts
            if item.kind is AAMASSourceKind.RNG_MECHANISM_SMOKE
        )
        if tuple(item[0] for item in self.rng_artifacts) != expected_rng_ids:
            _fail("run RNG artifacts do not exactly match committed RNG sources")
        for attempt_id, nested in self.rng_artifacts:
            source = source_by_attempt.get(attempt_id)
            if (
                source is None
                or source.kind is not AAMASSourceKind.RNG_MECHANISM_SMOKE
                or source.semantic_sha256 != nested.sha256
                or source.config_sha256 != nested.config_sha256
            ):
                _fail("run RNG artifact differs from its source commitment")
        mechanism_source = next(
            (
                item
                for item in self.artifact.source_artifacts
                if item.kind is AAMASSourceKind.MECHANISM_ABLATION
            ),
            None,
        )
        if self.mechanism_ablation_artifact is None:
            if mechanism_source is not None:
                _fail("run omitted its committed mechanism-ablation artifact")
        elif (
            mechanism_source is None
            or mechanism_source.semantic_sha256
            != self.mechanism_ablation_artifact.sha256
            or mechanism_source.config_sha256
            != self.mechanism_ablation_artifact.spec.sha256
            or mechanism_source.protocol_execution_receipt_sha256
            != self.mechanism_ablation_artifact.protocol_execution_receipt.sha256
        ):
            _fail("run mechanism artifact differs from its source commitment")
        if (
            not isinstance(self.timings, tuple)
            or not all(
                isinstance(item, AAMASOperationalTiming) for item in self.timings
            )
            or tuple(item.attempt_id for item in self.timings)
            != tuple(item.attempt_id for item in self.artifact.attempts)
        ):
            _fail("run timings must cover every attempt exactly once in registry order")


def _source_path(attempt: ExpectedAAMASAttempt) -> str:
    if attempt.kind is AAMASAttemptKind.RNG_MECHANISM_SMOKE:
        return f"rq1/{attempt.attempt_id}.json"
    if attempt.kind is AAMASAttemptKind.MECHANISM_ABLATION:
        return "rq1/mechanism-ablation/mechanism-ablation.json"
    raise AssertionError("blocked attempts do not have source paths")


def _timing_key(plan: AAMASBenchmarkPlan, index: int, attempt_id: str) -> str:
    return EventKey(
        experiment_id=plan.benchmark_id,
        seed=plan.seed,
        replicate_id=index,
        simulation_step=0,
        purpose=f"aamas.timing.{attempt_id}",
    ).value


def _execution_failure(
    expected: ExpectedAAMASAttempt,
    resources: ResourceBudget,
    error: Exception,
) -> AAMASAttempt:
    message = str(error).strip() or "execution raised without a message"
    error_name = re.sub(r"[^A-Za-z0-9._-]", "_", type(error).__name__)[:96]
    failure_code = f"execution_{error_name or 'unknown_error'}"
    return AAMASAttempt(
        attempt_id=expected.attempt_id,
        research_question=expected.research_question,
        kind=expected.kind,
        config_sha256=expected.config_sha256,
        status=AAMASAttemptStatus.FAILED,
        requested_resources=resources,
        artifact_sha256=None,
        completed_subattempts=0,
        failed_subattempts=1,
        failure_code=failure_code,
        failure_message=message,
    )


def run_aamas_benchmark(
    plan: AAMASBenchmarkPlan,
    *,
    allow_formal_scale: bool = False,
) -> AAMASBenchmarkRun:
    """Execute phase one while retaining RQ2/RQ3 as explicit blocked attempts."""

    if not isinstance(plan, AAMASBenchmarkPlan):
        _fail("plan must be an AAMASBenchmarkPlan")
    if not isinstance(allow_formal_scale, bool):
        _fail("allow_formal_scale must be boolean")
    formal_rng_work = len(_FORMAL_RNG_CELLS) * 256 * 32
    requested_rng_work = (
        len(plan.rng_cells) * plan.rng_replicate_count * plan.rng_step_count
    )
    formal_ablation_work = len(plan.mechanism_ablation_spec.variants) * 64 * 8
    requested_ablation_work = (
        len(plan.mechanism_ablation_spec.variants)
        * plan.mechanism_replicate_count
        * plan.mechanism_step_count
    )
    requires_formal_authorization = (
        requested_rng_work >= formal_rng_work
        or requested_ablation_work >= formal_ablation_work
    )
    if requires_formal_authorization and not allow_formal_scale:
        raise AAMASBenchmarkIntegrityError(
            "formal AAMAS scale requires explicit allow_formal_scale=True"
        )
    expected_attempts = plan.expected_attempts
    rng_specs = iter(plan.rng_specs)
    attempts: list[AAMASAttempt] = []
    sources: list[AAMASSourceArtifact] = []
    rng_artifacts: list[tuple[str, MechanismSmokeArtifact]] = []
    mechanism_artifact: MechanismAblationArtifact | None = None
    timings: list[AAMASOperationalTiming] = []

    for index, expected in enumerate(expected_attempts):
        started = time.perf_counter()
        if not expected.execution_expected:
            attempt = AAMASAttempt(
                attempt_id=expected.attempt_id,
                research_question=expected.research_question,
                kind=expected.kind,
                config_sha256=expected.config_sha256,
                status=AAMASAttemptStatus.BLOCKED_PENDING_IMPLEMENTATION,
                requested_resources=plan.resources,
                artifact_sha256=None,
                completed_subattempts=0,
                failed_subattempts=0,
                failure_code=BLOCKED_PENDING_IMPLEMENTATION,
                failure_message=(
                    "Phase-one implementation intentionally does not execute "
                    f"{expected.kind.value}."
                ),
            )
        elif expected.kind is AAMASAttemptKind.RNG_MECHANISM_SMOKE:
            smoke_spec = next(rng_specs)
            try:
                nested = run_mechanism_smoke(smoke_spec)
                rng_artifacts.append((expected.attempt_id, nested))
                attempt = AAMASAttempt(
                    attempt_id=expected.attempt_id,
                    research_question=expected.research_question,
                    kind=expected.kind,
                    config_sha256=expected.config_sha256,
                    status=AAMASAttemptStatus.COMPLETED,
                    requested_resources=plan.resources,
                    artifact_sha256=nested.sha256,
                    completed_subattempts=3,
                    failed_subattempts=0,
                )
                sources.append(
                    AAMASSourceArtifact(
                        attempt_id=expected.attempt_id,
                        kind=AAMASSourceKind.RNG_MECHANISM_SMOKE,
                        relative_path=_source_path(expected),
                        semantic_sha256=nested.sha256,
                        config_sha256=smoke_spec.sha256,
                        claim_scope=nested.claim_scope,
                        protocol_execution_receipt_sha256=None,
                    )
                )
            except Exception as error:  # retained as a terminal scientific state
                attempt = _execution_failure(expected, plan.resources, error)
        else:
            ablation_spec = plan.mechanism_ablation_spec
            try:
                expected_protocol = build_mechanism_ablation_protocol(ablation_spec)
                protocol, authorization, execution_started_at = (
                    issue_mechanism_ablation_authorization(
                        ablation_spec,
                        execution_id=f"{plan.benchmark_id}.phase1.mechanism",
                    )
                )
                if protocol != expected_protocol:
                    raise AAMASBenchmarkIntegrityError(
                        "issued mechanism protocol differs from the public builder"
                    )
                nested_mechanism = run_mechanism_ablations(
                    spec=ablation_spec,
                    study_protocol=protocol,
                    protocol_authorization=authorization,
                    execution_started_at=execution_started_at,
                )
                mechanism_artifact = nested_mechanism
                status = (
                    AAMASAttemptStatus.COMPLETED
                    if nested_mechanism.failed_count == 0
                    else AAMASAttemptStatus.FAILED
                )
                attempt = AAMASAttempt(
                    attempt_id=expected.attempt_id,
                    research_question=expected.research_question,
                    kind=expected.kind,
                    config_sha256=expected.config_sha256,
                    status=status,
                    requested_resources=plan.resources,
                    artifact_sha256=nested_mechanism.sha256,
                    completed_subattempts=nested_mechanism.completed_count,
                    failed_subattempts=nested_mechanism.failed_count,
                    failure_code=(
                        None
                        if status is AAMASAttemptStatus.COMPLETED
                        else "mechanism_ablation_subattempt_failure"
                    ),
                    failure_message=(
                        None
                        if status is AAMASAttemptStatus.COMPLETED
                        else (
                            "Mechanism ablation retained one or more failed grid rows."
                        )
                    ),
                )
                sources.append(
                    AAMASSourceArtifact(
                        attempt_id=expected.attempt_id,
                        kind=AAMASSourceKind.MECHANISM_ABLATION,
                        relative_path=_source_path(expected),
                        semantic_sha256=nested_mechanism.sha256,
                        config_sha256=ablation_spec.sha256,
                        claim_scope=nested_mechanism.claim_scope,
                        protocol_execution_receipt_sha256=(
                            nested_mechanism.protocol_execution_receipt.sha256
                        ),
                    )
                )
            except Exception as error:  # retained as a terminal scientific state
                attempt = _execution_failure(expected, plan.resources, error)
        attempts.append(attempt)
        timings.append(
            AAMASOperationalTiming(
                attempt_id=expected.attempt_id,
                event_key=_timing_key(plan, index, expected.attempt_id),
                wall_seconds=time.perf_counter() - started,
            )
        )

    attempt_tuple = tuple(attempts)
    artifact = AAMASBenchmarkArtifact(
        plan=plan,
        attempts=attempt_tuple,
        source_artifacts=tuple(sources),
        status=_derived_status(attempt_tuple),
    )
    return AAMASBenchmarkRun(
        artifact=artifact,
        rng_artifacts=tuple(rng_artifacts),
        mechanism_ablation_artifact=mechanism_artifact,
        timings=tuple(timings),
    )


def _log_event_key(
    plan: AAMASBenchmarkPlan, index: int, attempt_id: str, purpose: str
) -> str:
    return EventKey(
        experiment_id=plan.benchmark_id,
        seed=plan.seed,
        replicate_id=index,
        simulation_step=0,
        purpose=f"aamas.{purpose}.{attempt_id}",
    ).value


def _resource_events(artifact: AAMASBenchmarkArtifact) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "artifact_sha256": attempt.artifact_sha256,
            "attempt_id": attempt.attempt_id,
            "config_sha256": attempt.config_sha256,
            "event": "requested",
            "event_key": _log_event_key(
                artifact.plan, index, attempt.attempt_id, "resources"
            ),
            "requested_resources": attempt.requested_resources.to_mapping(),
            "schema_version": AAMAS_BENCHMARK_LOG_SCHEMA_VERSION,
            "status": attempt.status.value,
        }
        for index, attempt in enumerate(artifact.attempts)
    )


def _failure_events(artifact: AAMASBenchmarkArtifact) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "artifact_sha256": attempt.artifact_sha256,
            "attempt_id": attempt.attempt_id,
            "event": "terminal_noncompletion",
            "event_key": _log_event_key(
                artifact.plan, index, attempt.attempt_id, "failures"
            ),
            "failure_code": attempt.failure_code,
            "failure_message": attempt.failure_message,
            "schema_version": AAMAS_BENCHMARK_LOG_SCHEMA_VERSION,
            "status": attempt.status.value,
        }
        for index, attempt in enumerate(artifact.attempts)
        if attempt.status is not AAMASAttemptStatus.COMPLETED
    )


def write_aamas_benchmark(
    directory: Path,
    run: AAMASBenchmarkRun,
) -> tuple[Path, Path]:
    """Write an immutable phase-one bundle and refuse every overwrite."""

    if not isinstance(directory, Path):
        _fail("directory must be a Path")
    if not isinstance(run, AAMASBenchmarkRun):
        _fail("run must be an AAMASBenchmarkRun")
    if directory.exists():
        raise AAMASBenchmarkIntegrityError(
            "AAMAS benchmark output directory already exists"
        )
    directory.mkdir(parents=True, exist_ok=False)
    write_json_file(directory / "plan.json", run.artifact.plan.to_mapping())
    write_text_file(directory / "plan.sha256", f"{run.artifact.plan.sha256}\n")

    rng_by_id = dict(run.rng_artifacts)
    for source in run.artifact.source_artifacts:
        path = directory / source.relative_path
        if source.kind is AAMASSourceKind.RNG_MECHANISM_SMOKE:
            nested = rng_by_id[source.attempt_id]
            if nested.sha256 != source.semantic_sha256:
                raise AAMASBenchmarkIntegrityError(
                    "RNG source changed before bundle publication"
                )
            path.parent.mkdir(parents=True, exist_ok=True)
            write_mechanism_smoke_artifact(path, nested)
        else:
            nested_mechanism = run.mechanism_ablation_artifact
            if (
                nested_mechanism is None
                or nested_mechanism.sha256 != source.semantic_sha256
            ):
                raise AAMASBenchmarkIntegrityError(
                    "mechanism source changed before bundle publication"
                )
            write_mechanism_ablation_artifact(path.parent, nested_mechanism)

    result_path = directory / "result.json"
    seal_path = directory / "result.sha256"
    write_json_file(result_path, run.artifact.to_mapping())
    write_text_file(seal_path, f"{run.artifact.sha256}\n")
    with JsonlEventWriter(directory / "resources.jsonl") as writer:
        for event in _resource_events(run.artifact):
            writer.write(event)
    with JsonlEventWriter(directory / "failures.jsonl") as writer:
        for event in _failure_events(run.artifact):
            writer.write(event)
    with JsonlEventWriter(directory / "timing.jsonl") as writer:
        for timing in run.timings:
            writer.write(timing.to_mapping())
    return result_path, seal_path


@dataclass(frozen=True, slots=True)
class AAMASBenchmarkAudit:
    """Positive audit receipt for one fully checked phase-one bundle."""

    artifact_sha256: str
    plan_sha256: str
    audited_source_count: int
    attempt_count: int
    completed_count: int
    failed_count: int
    blocked_count: int
    timing_excluded_from_scientific_hash: Literal[True] = True
    passed: Literal[True] = True

    def __post_init__(self) -> None:
        _sha256(self.artifact_sha256, "audit artifact_sha256")
        _sha256(self.plan_sha256, "audit plan_sha256")
        for field in (
            "audited_source_count",
            "attempt_count",
            "completed_count",
            "failed_count",
            "blocked_count",
        ):
            _integer(getattr(self, field), field)
        if (
            self.timing_excluded_from_scientific_hash is not True
            or self.passed is not True
        ):
            _fail("positive AAMAS audit flags must be true")


def _read_semantic_seal(path: Path, *, label: str) -> str:
    try:
        value = path.read_text(encoding="ascii")
    except (OSError, UnicodeError) as error:
        raise AAMASBenchmarkIntegrityError(f"cannot read {label} seal") from error
    if not value.endswith("\n") or value.count("\n") != 1:
        raise AAMASBenchmarkIntegrityError(f"{label} seal is not canonical")
    try:
        return _sha256(value[:-1], f"{label} seal")
    except AAMASBenchmarkSchemaError as error:
        raise AAMASBenchmarkIntegrityError(f"{label} seal is invalid") from error


def _expected_bundle_files(artifact: AAMASBenchmarkArtifact) -> frozenset[str]:
    expected = {
        "failures.jsonl",
        "plan.json",
        "plan.sha256",
        "resources.jsonl",
        "result.json",
        "result.sha256",
        "timing.jsonl",
    }
    for source in artifact.source_artifacts:
        expected.add(source.relative_path)
        if source.kind is AAMASSourceKind.RNG_MECHANISM_SMOKE:
            expected.add(f"{source.relative_path}.sha256")
        else:
            expected.add(
                str(
                    PurePosixPath(source.relative_path).with_name(
                        "mechanism-ablation.sha256"
                    )
                )
            )
    return frozenset(expected)


def _verify_bundle_membership(
    directory: Path, artifact: AAMASBenchmarkArtifact
) -> None:
    expected_files = _expected_bundle_files(artifact)
    actual_files: set[str] = set()
    try:
        entries = tuple(directory.rglob("*"))
    except OSError as error:
        raise AAMASBenchmarkIntegrityError(
            "cannot enumerate AAMAS benchmark bundle"
        ) from error
    for entry in entries:
        relative = entry.relative_to(directory).as_posix()
        if entry.is_symlink():
            raise AAMASBenchmarkIntegrityError(
                f"AAMAS benchmark bundle contains a symlink: {relative}"
            )
        try:
            if entry.is_dir():
                continue
            if not entry.is_file() or entry.stat().st_nlink != 1:
                raise AAMASBenchmarkIntegrityError(
                    "AAMAS benchmark member is not a single-link regular file: "
                    f"{relative}"
                )
        except OSError as error:
            raise AAMASBenchmarkIntegrityError(
                f"cannot inspect AAMAS benchmark member: {relative}"
            ) from error
        actual_files.add(relative)
    if actual_files != expected_files:
        raise AAMASBenchmarkIntegrityError(
            "AAMAS benchmark bundle membership changed: "
            f"missing={sorted(expected_files - actual_files)!r}, "
            f"unknown={sorted(actual_files - expected_files)!r}"
        )


def _audit_timing_events(
    events: tuple[dict[str, object], ...], artifact: AAMASBenchmarkArtifact
) -> None:
    if len(events) != len(artifact.attempts):
        raise AAMASBenchmarkIntegrityError(
            "operational timing log must cover every attempt exactly once"
        )
    expected_fields = frozenset(
        {
            "attempt_id",
            "event",
            "event_key",
            "measurement_scope",
            "schema_version",
            "wall_seconds",
        }
    )
    for index, (event, attempt) in enumerate(
        zip(events, artifact.attempts, strict=True)
    ):
        if frozenset(event) != expected_fields:
            raise AAMASBenchmarkIntegrityError("timing log schema changed")
        expected_key = _timing_key(artifact.plan, index, attempt.attempt_id)
        wall = event["wall_seconds"]
        if (
            event["attempt_id"] != attempt.attempt_id
            or event["event"] != "observed"
            or event["event_key"] != expected_key
            or event["measurement_scope"] != "operational_non_scientific_process"
            or event["schema_version"] != AAMAS_BENCHMARK_LOG_SCHEMA_VERSION
            or isinstance(wall, bool)
            or not isinstance(wall, int | float)
            or not math.isfinite(float(wall))
            or float(wall) < 0.0
        ):
            raise AAMASBenchmarkIntegrityError("timing log event is invalid")


def audit_aamas_benchmark(
    directory: Path,
    expected_plan: AAMASBenchmarkPlan,
) -> AAMASBenchmarkAudit:
    """Verify seals, complete attempts, logs, and live-replayed RQ1 artifacts."""

    if not isinstance(directory, Path):
        _fail("directory must be a Path")
    if not isinstance(expected_plan, AAMASBenchmarkPlan):
        _fail("expected_plan must be an AAMASBenchmarkPlan")
    try:
        stored_plan = AAMASBenchmarkPlan.from_mapping(
            read_json_object(directory / "plan.json")
        )
        artifact = AAMASBenchmarkArtifact.from_mapping(
            read_json_object(directory / "result.json")
        )
    except (OSError, ValueError, MetadataValidationError) as error:
        raise AAMASBenchmarkIntegrityError(
            "cannot parse AAMAS benchmark roots"
        ) from error
    if stored_plan != expected_plan or artifact.plan != expected_plan:
        raise AAMASBenchmarkIntegrityError(
            "stored benchmark plan differs from its external commitment"
        )
    if (
        _read_semantic_seal(directory / "plan.sha256", label="plan")
        != stored_plan.sha256
    ):
        raise AAMASBenchmarkIntegrityError("plan seal does not match")
    if (
        _read_semantic_seal(directory / "result.sha256", label="result")
        != artifact.sha256
    ):
        raise AAMASBenchmarkIntegrityError("result seal does not match")
    _verify_bundle_membership(directory, artifact)

    expected_by_id = {item.attempt_id: item for item in expected_plan.expected_attempts}
    rng_spec_by_attempt = {
        expected.attempt_id: spec
        for expected, spec in zip(
            (
                item
                for item in expected_plan.expected_attempts
                if item.kind is AAMASAttemptKind.RNG_MECHANISM_SMOKE
            ),
            expected_plan.rng_specs,
            strict=True,
        )
    }
    for source in artifact.source_artifacts:
        expected = expected_by_id[source.attempt_id]
        path = directory / source.relative_path
        if source.kind is AAMASSourceKind.RNG_MECHANISM_SMOKE:
            smoke_audit = audit_mechanism_smoke_artifact(
                path,
                rng_spec_by_attempt[source.attempt_id],
                expected_artifact_sha256=source.semantic_sha256,
            )
            if smoke_audit.config_sha256 != expected.config_sha256:
                raise AAMASBenchmarkIntegrityError(
                    "RNG audit config differs from its attempt"
                )
        else:
            receipt_sha256 = source.protocol_execution_receipt_sha256
            if receipt_sha256 is None:  # guarded by source construction
                raise AAMASBenchmarkIntegrityError(
                    "mechanism source lacks its protocol receipt commitment"
                )
            ablation_audit = audit_written_mechanism_ablation_artifact(
                path.parent,
                expected_spec=expected_plan.mechanism_ablation_spec,
                expected_protocol_receipt_sha256=receipt_sha256,
            )
            if ablation_audit.artifact_sha256 != source.semantic_sha256:
                raise AAMASBenchmarkIntegrityError(
                    "mechanism audit differs from its source commitment"
                )

    try:
        resources = read_jsonl_objects(directory / "resources.jsonl")
        failures = read_jsonl_objects(directory / "failures.jsonl")
        timings = read_jsonl_objects(directory / "timing.jsonl")
    except (OSError, MetadataValidationError) as error:
        raise AAMASBenchmarkIntegrityError("cannot read AAMAS audit logs") from error
    if resources != _resource_events(artifact):
        raise AAMASBenchmarkIntegrityError("resource log differs from attempts")
    if failures != _failure_events(artifact):
        raise AAMASBenchmarkIntegrityError(
            "failure log omitted or altered a failed/blocked attempt"
        )
    _audit_timing_events(timings, artifact)
    _verify_bundle_membership(directory, artifact)
    return AAMASBenchmarkAudit(
        artifact_sha256=artifact.sha256,
        plan_sha256=artifact.plan.sha256,
        audited_source_count=len(artifact.source_artifacts),
        attempt_count=len(artifact.attempts),
        completed_count=artifact.completed_count,
        failed_count=artifact.failed_count,
        blocked_count=artifact.blocked_count,
    )


__all__ = [
    "AAMAS_BENCHMARK_ARTIFACT_SCHEMA_VERSION",
    "AAMAS_BENCHMARK_CLAIM_SCOPE",
    "AAMAS_BENCHMARK_LOG_SCHEMA_VERSION",
    "AAMAS_BENCHMARK_PLAN_SCHEMA_VERSION",
    "BLOCKED_PENDING_IMPLEMENTATION",
    "AAMASAttempt",
    "AAMASAttemptKind",
    "AAMASAttemptStatus",
    "AAMASBenchmarkArtifact",
    "AAMASBenchmarkAudit",
    "AAMASBenchmarkError",
    "AAMASBenchmarkIntegrityError",
    "AAMASBenchmarkPlan",
    "AAMASBenchmarkRun",
    "AAMASBenchmarkSchemaError",
    "AAMASBenchmarkStatus",
    "AAMASOperationalTiming",
    "AAMASResearchQuestion",
    "AAMASSourceArtifact",
    "AAMASSourceKind",
    "ExpectedAAMASAttempt",
    "RNGStressCell",
    "audit_aamas_benchmark",
    "run_aamas_benchmark",
    "write_aamas_benchmark",
]
