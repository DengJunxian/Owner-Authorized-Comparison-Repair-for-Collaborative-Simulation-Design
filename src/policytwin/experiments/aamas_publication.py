"""Externally committed synthetic AAMAS publication bundle.

This module is the outermost evidence boundary for the synthetic AAMAS study.
It binds the core RQ1--RQ3 bundle, the integrated CLOB branch experiment, the
structural misspecification negative control, and both frozen reporting
derivations.  A complete synthetic evidence bundle still abstains from a
publication validity decision because it lacks independent real-data and
external verifier authority.

The caller must retain both the plan SHA-256 before execution and the root-seal
SHA-256 returned by the writer.  Auditing replays every executable child and
re-derives the core reporting artifact and root scientific summary.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import stat
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Literal, NoReturn, cast

from policytwin.data.temporal_firewall import (
    CutoffPolicy,
    TemporalAuditLog,
    TemporalFirewall,
)
from policytwin.evaluation.policy import (
    ValidityCertificate,
    ValidityEvidence,
    ValidityLevel,
    build_validity_certificate,
)
from policytwin.experiments.aamas_integrated_branch import (
    AAMASIntegratedBranchAudit,
    AAMASIntegratedBranchPlan,
    AAMASIntegratedBranchRun,
    audit_written_aamas_integrated_branch,
    run_aamas_integrated_branch,
    write_aamas_integrated_branch,
)
from policytwin.experiments.aamas_misspecification import (
    AAMASMisspecificationArtifact,
    AAMASMisspecificationSpec,
    MisspecificationAuditReport,
    NegativeControlDecision,
    audit_aamas_misspecification_artifact,
    run_aamas_misspecification_benchmark,
    write_aamas_misspecification_artifact,
)
from policytwin.experiments.aamas_submission import (
    AAMASSubmissionAuditReport,
    AAMASSubmissionPlan,
    AAMASSubmissionRun,
    audit_aamas_submission,
    run_aamas_submission,
    write_aamas_submission,
)
from policytwin.provenance import (
    EnvironmentMetadata,
    GitMetadata,
    RunMetadata,
    canonical_json_bytes,
    collect_environment_metadata,
    collect_git_metadata,
    read_json_bytes_object,
    read_json_object,
    read_jsonl_objects,
    sha256_bytes,
    sha256_file,
)
from policytwin.reporting.aamas_integrated_branch import (
    AAMAS_INTEGRATED_BRANCH_REPORT_FILENAME,
    AAMAS_INTEGRATED_BRANCH_REPORT_SEAL_FILENAME,
    AAMASIntegratedBranchReportArtifact,
    AAMASIntegratedBranchReportAudit,
    FrozenIntegratedBranchReportingSpec,
    audit_written_aamas_integrated_branch_report,
    derive_aamas_integrated_branch_report,
    write_aamas_integrated_branch_report,
)
from policytwin.reporting.aamas_submission import (
    AAMASSubmissionArtifact as AAMASCoreReportingArtifact,
)
from policytwin.reporting.aamas_submission import (
    AAMASSubmissionAudit as AAMASCoreReportingAudit,
)
from policytwin.reporting.aamas_submission import (
    FrozenAAMASReportingSpec,
    audit_aamas_submission_artifact,
    derive_aamas_submission_artifact,
    write_aamas_submission_artifact,
)

AAMAS_PUBLICATION_PLAN_SCHEMA_VERSION = "policytwin.aamas-publication-plan.v2"
AAMAS_PUBLICATION_RESULT_SCHEMA_VERSION = "policytwin.aamas-publication-result.v2"
AAMAS_PUBLICATION_MANIFEST_SCHEMA_VERSION = "policytwin.aamas-publication-manifest.v2"
AAMAS_PUBLICATION_SEAL_SCHEMA_VERSION = "policytwin.aamas-publication-seal.v2"
AAMAS_PUBLICATION_STUDY_COMMITMENT_SCHEMA_VERSION = (
    "policytwin.aamas-publication-study-commitment.v1"
)
AAMAS_PUBLICATION_CLAIM_SCOPE = "synthetic_aamas_evidence_bundle_only"

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_ROOT_DIRECTORIES = frozenset(
    {
        "core",
        "core-reporting",
        "integrated-branch",
        "integrated-branch-reporting",
        "misspecification",
    }
)
_NETWORK_FILES = (
    "network-core-run.jsonl",
    "network-integrated-branch-run.jsonl",
    "network-misspecification-run.jsonl",
    "network-core-publication.jsonl",
    "network-integrated-branch-publication.jsonl",
    "network-misspecification-publication.jsonl",
    "network-core-reporting.jsonl",
    "network-integrated-branch-reporting.jsonl",
)
_ROOT_FILES = frozenset(
    {
        "data-manifest.json",
        "attempt-registry.json",
        "failures.jsonl",
        "manifest.json",
        "metadata.json",
        "output-registry.json",
        "plan.json",
        "result.json",
        "seal.json",
        "source-manifest.json",
        "study-commitment.json",
        "timing.jsonl",
        "validity-certificate.json",
        *_NETWORK_FILES,
    }
)
_TIMING_COMPONENTS = (
    "core_execution",
    "integrated_branch_execution",
    "misspecification_execution",
    "core_publication",
    "integrated_branch_publication_and_replay_audit",
    "misspecification_publication_and_replay_audit",
    "core_reporting_derivation_and_audit",
    "integrated_branch_reporting_derivation_and_audit",
)


class AAMASPublicationError(RuntimeError):
    """Base error for an invalid publication plan or execution."""


class AAMASPublicationIntegrityError(AAMASPublicationError):
    """Raised when evidence differs from an external commitment."""


def _fail(message: str) -> NoReturn:
    raise AAMASPublicationError(message)


def _integrity_fail(message: str) -> NoReturn:
    raise AAMASPublicationIntegrityError(message)


def _identifier(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        _integrity_fail(f"{field_name} must be a stable ASCII identifier")
    return value


def _sha256(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _integrity_fail(f"{field_name} must be a lowercase SHA-256")
    return value


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        _integrity_fail(f"{field_name} must be an object with string keys")
    return cast(Mapping[str, object], value)


def _sequence(value: object, field_name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        _integrity_fail(f"{field_name} must be an array")
    return cast(Sequence[object], value)


def _strict_fields(
    value: Mapping[str, object], expected: frozenset[str], context: str
) -> None:
    if frozenset(value) != expected:
        _integrity_fail(f"{context} has missing or unknown fields")


def _integer(value: object, field_name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _integrity_fail(f"{field_name} must be an integer >= {minimum}")
    return value


def _number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        _integrity_fail(f"{field_name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        _integrity_fail(f"{field_name} must be finite")
    return result


def _pretty_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _write_new(path: Path, payload: bytes) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                _integrity_fail(f"short write while creating {path.name}")
            remaining = remaining[written:]
        os.fsync(descriptor)
    except OSError as error:
        raise AAMASPublicationIntegrityError(
            f"cannot exclusively create {path.name}: {error}"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        _integrity_fail(f"JSONL evidence cannot be empty: {path.name}")
    payload = b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)
    _write_new(path, payload)


def _component_definitions() -> tuple[dict[str, object], ...]:
    return (
        {
            "component_id": "core",
            "dependencies": [],
            "failure_propagation": "any_core_failure_blocks_root_completion",
            "path": "core",
        },
        {
            "component_id": "integrated_branch",
            "dependencies": [],
            "failure_propagation": (
                "any_replicate_or_replay_failure_blocks_root_completion"
            ),
            "path": "integrated-branch",
        },
        {
            "component_id": "misspecification",
            "dependencies": [],
            "failure_propagation": (
                "any_target_failure_or_undetected_negative_control_blocks_root_completion"
            ),
            "path": "misspecification",
        },
        {
            "component_id": "core_reporting",
            "dependencies": ["core"],
            "failure_propagation": "derivation_or_source_audit_failure_blocks_root",
            "path": "core-reporting",
        },
        {
            "component_id": "integrated_branch_reporting",
            "dependencies": ["integrated_branch"],
            "failure_propagation": "derivation_or_source_audit_failure_blocks_root",
            "path": "integrated-branch-reporting",
        },
    )


def _output_definitions() -> tuple[dict[str, object], ...]:
    return (
        {
            "output_id": "core_scientific_result",
            "path": "core/result.json",
            "producer": "core",
            "required_for_scientific_completion": True,
        },
        {
            "output_id": "integrated_branch_artifact",
            "path": "integrated-branch/artifact.json",
            "producer": "integrated_branch",
            "required_for_scientific_completion": True,
        },
        {
            "output_id": "misspecification_negative_control",
            "path": "misspecification/artifact.json",
            "producer": "misspecification",
            "required_for_scientific_completion": True,
        },
        {
            "output_id": "core_frozen_reporting",
            "path": "core-reporting/aamas-submission.json",
            "producer": "core_reporting",
            "required_for_scientific_completion": True,
        },
        {
            "output_id": "integrated_branch_frozen_reporting",
            "path": ("integrated-branch-reporting/integrated-branch-report.json"),
            "producer": "integrated_branch_reporting",
            "required_for_scientific_completion": True,
        },
        {
            "output_id": "root_validity_certificate",
            "path": "validity-certificate.json",
            "producer": "aamas_publication",
            "required_for_scientific_completion": True,
        },
    )


def _attempt_definitions() -> tuple[dict[str, object], ...]:
    """Return the immutable execution/publication/reporting attempt registry."""

    return (
        {
            "attempt_id": "core_execution",
            "component_id": "core",
            "dependencies": [],
            "stage": "execution",
        },
        {
            "attempt_id": "integrated_branch_execution",
            "component_id": "integrated_branch",
            "dependencies": [],
            "stage": "execution",
        },
        {
            "attempt_id": "misspecification_execution",
            "component_id": "misspecification",
            "dependencies": [],
            "stage": "execution",
        },
        {
            "attempt_id": "core_publication",
            "component_id": "core",
            "dependencies": ["core_execution"],
            "stage": "publication",
        },
        {
            "attempt_id": "integrated_branch_publication",
            "component_id": "integrated_branch",
            "dependencies": ["integrated_branch_execution"],
            "stage": "publication",
        },
        {
            "attempt_id": "misspecification_publication",
            "component_id": "misspecification",
            "dependencies": ["misspecification_execution"],
            "stage": "publication",
        },
        {
            "attempt_id": "core_reporting",
            "component_id": "core_reporting",
            "dependencies": ["core_publication"],
            "stage": "reporting",
        },
        {
            "attempt_id": "integrated_branch_reporting",
            "component_id": "integrated_branch_reporting",
            "dependencies": ["integrated_branch_publication"],
            "stage": "reporting",
        },
    )


class AAMASPublicationAttemptStatus(StrEnum):
    """Terminal state retained for every pre-registered outer attempt."""

    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED_UPSTREAM = "blocked_upstream"


@dataclass(frozen=True, slots=True)
class AAMASPublicationAttempt:
    """One immutable terminal row; failed attempts are evidence, not missing data."""

    attempt_id: str
    component_id: str
    stage: str
    status: AAMASPublicationAttemptStatus
    config_sha256: str
    output_sha256: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.attempt_id, "attempt_id")
        _identifier(self.component_id, "component_id")
        if self.stage not in {"execution", "publication", "reporting"}:
            _fail("attempt stage is invalid")
        if not isinstance(self.status, AAMASPublicationAttemptStatus):
            _fail("attempt status is invalid")
        _sha256(self.config_sha256, "attempt config_sha256")
        if self.output_sha256 is not None:
            _sha256(self.output_sha256, "attempt output_sha256")
        if self.status is AAMASPublicationAttemptStatus.COMPLETED:
            if self.error_type is not None or self.error_message is not None:
                _fail("completed attempt cannot retain failure fields")
        elif self.status is AAMASPublicationAttemptStatus.FAILED:
            if (
                not isinstance(self.error_type, str)
                or not self.error_type
                or not isinstance(self.error_message, str)
                or not self.error_message
                or self.output_sha256 is not None
            ):
                _fail("failed attempt must retain typed error evidence only")
        elif self.output_sha256 is not None:
            _fail("blocked attempt cannot retain an output digest")
        if self.reason is not None and (
            not isinstance(self.reason, str) or not self.reason
        ):
            _fail("attempt reason must be null or non-empty")

    def to_mapping(self) -> dict[str, object]:
        return {
            "attempt_id": self.attempt_id,
            "component_id": self.component_id,
            "config_sha256": self.config_sha256,
            "error_message": self.error_message,
            "error_type": self.error_type,
            "output_sha256": self.output_sha256,
            "reason": self.reason,
            "stage": self.stage,
            "status": self.status.value,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> AAMASPublicationAttempt:
        _strict_fields(
            value,
            frozenset(
                {
                    "attempt_id",
                    "component_id",
                    "config_sha256",
                    "error_message",
                    "error_type",
                    "output_sha256",
                    "reason",
                    "stage",
                    "status",
                }
            ),
            "publication attempt",
        )
        try:
            status = AAMASPublicationAttemptStatus(cast(str, value["status"]))
        except (TypeError, ValueError) as error:
            raise AAMASPublicationIntegrityError(
                "publication attempt status is invalid"
            ) from error

        def optional_text(field_name: str) -> str | None:
            raw = value[field_name]
            if raw is None:
                return None
            if not isinstance(raw, str) or not raw:
                _integrity_fail(f"{field_name} must be null or non-empty")
            return raw

        stage = value["stage"]
        if not isinstance(stage, str):
            _integrity_fail("attempt stage must be a string")
        return cls(
            attempt_id=_identifier(value["attempt_id"], "attempt_id"),
            component_id=_identifier(value["component_id"], "component_id"),
            stage=stage,
            status=status,
            config_sha256=_sha256(value["config_sha256"], "config_sha256"),
            output_sha256=(
                None
                if value["output_sha256"] is None
                else _sha256(value["output_sha256"], "output_sha256")
            ),
            error_type=optional_text("error_type"),
            error_message=optional_text("error_message"),
            reason=optional_text("reason"),
        )


def _attempt(
    attempt_id: str,
    *,
    config_sha256: str,
    status: AAMASPublicationAttemptStatus,
    output_sha256: str | None = None,
    error: Exception | None = None,
    reason: str | None = None,
) -> AAMASPublicationAttempt:
    definition = next(
        item for item in _attempt_definitions() if item["attempt_id"] == attempt_id
    )
    return AAMASPublicationAttempt(
        attempt_id=attempt_id,
        component_id=cast(str, definition["component_id"]),
        stage=cast(str, definition["stage"]),
        status=status,
        config_sha256=config_sha256,
        output_sha256=output_sha256,
        error_type=type(error).__name__ if error is not None else None,
        error_message=(str(error)[:2048] or "failed without a message")
        if error is not None
        else None,
        reason=reason,
    )


def _rq1_work(plan: AAMASSubmissionPlan) -> int:
    rq1 = plan.rq1_plan
    return (
        len(rq1.rng_cells) * rq1.rng_replicate_count * rq1.rng_step_count
        + len(rq1.mechanism_ablation_spec.variants)
        * rq1.mechanism_replicate_count
        * rq1.mechanism_step_count
    )


@dataclass(frozen=True, slots=True)
class AAMASPublicationPlan:
    """Immutable outer plan committed before any child execution."""

    publication_id: str = "policytwin-aamas-synthetic-publication-v1"
    core_plan: AAMASSubmissionPlan = field(default_factory=AAMASSubmissionPlan)
    integrated_branch_plan: AAMASIntegratedBranchPlan = field(
        default_factory=AAMASIntegratedBranchPlan.formal
    )
    misspecification_spec: AAMASMisspecificationSpec = field(
        default_factory=AAMASMisspecificationSpec
    )
    reporting_spec: FrozenAAMASReportingSpec = field(
        default_factory=FrozenAAMASReportingSpec
    )
    integrated_branch_reporting_spec: FrozenIntegratedBranchReportingSpec = field(
        default_factory=FrozenIntegratedBranchReportingSpec
    )
    synthetic_only: Literal[True] = True
    publication_eligible: Literal[False] = False
    schema_version: str = AAMAS_PUBLICATION_PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if _IDENTIFIER.fullmatch(self.publication_id) is None:
            _fail("publication_id must be a stable ASCII identifier")
        if not isinstance(self.core_plan, AAMASSubmissionPlan):
            _fail("core_plan must be an AAMASSubmissionPlan")
        if not isinstance(self.integrated_branch_plan, AAMASIntegratedBranchPlan):
            _fail("integrated_branch_plan must be an AAMASIntegratedBranchPlan")
        if not isinstance(self.misspecification_spec, AAMASMisspecificationSpec):
            _fail("misspecification_spec must be an AAMASMisspecificationSpec")
        if not isinstance(self.reporting_spec, FrozenAAMASReportingSpec):
            _fail("reporting_spec must be a FrozenAAMASReportingSpec")
        if not isinstance(
            self.integrated_branch_reporting_spec,
            FrozenIntegratedBranchReportingSpec,
        ):
            _fail("integrated_branch_reporting_spec must be a frozen typed spec")
        if self.synthetic_only is not True or self.publication_eligible is not False:
            _fail("publication plan cannot broaden synthetic-only claim authority")
        if self.schema_version != AAMAS_PUBLICATION_PLAN_SCHEMA_VERSION:
            _fail("unsupported AAMAS publication plan schema")

    @property
    def execution_work(self) -> int:
        return (
            self.core_plan.execution_work
            + self.integrated_branch_plan.execution_work
            + self.misspecification_spec.execution_work
        )

    @property
    def requires_formal_authorization(self) -> bool:
        formal = AAMASPublicationPlan()
        return (
            _rq1_work(self.core_plan) >= _rq1_work(formal.core_plan)
            or self.core_plan.calibration_spec.execution_work
            >= formal.core_plan.calibration_spec.execution_work
            or self.core_plan.execution_work >= formal.core_plan.execution_work
            or self.integrated_branch_plan.execution_work
            >= formal.integrated_branch_plan.execution_work
            or self.misspecification_spec.execution_work
            >= formal.misspecification_spec.execution_work
            or self.execution_work >= formal.execution_work
        )

    @property
    def is_formal_scale(self) -> bool:
        return (
            self.core_plan == AAMASSubmissionPlan()
            and self.integrated_branch_plan == AAMASIntegratedBranchPlan.formal()
            and self.misspecification_spec == AAMASMisspecificationSpec()
        )

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def to_mapping(self) -> dict[str, object]:
        return {
            "attempt_registry": list(_attempt_definitions()),
            "claim_scope": AAMAS_PUBLICATION_CLAIM_SCOPE,
            "component_registry": list(_component_definitions()),
            "core_plan": self.core_plan.to_mapping(),
            "failure_propagation": "fail_closed_no_complete_case_or_missing_output",
            "integrated_branch_plan": self.integrated_branch_plan.to_mapping(),
            "integrated_branch_reporting_spec": (
                self.integrated_branch_reporting_spec.to_mapping()
            ),
            "misspecification_spec": self.misspecification_spec.to_mapping(),
            "output_registry": list(_output_definitions()),
            "publication_eligible": self.publication_eligible,
            "publication_id": self.publication_id,
            "reporting_spec": self.reporting_spec.to_mapping(),
            "schema_version": self.schema_version,
            "synthetic_only": self.synthetic_only,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> AAMASPublicationPlan:
        _strict_fields(
            value,
            frozenset(
                {
                    "attempt_registry",
                    "claim_scope",
                    "component_registry",
                    "core_plan",
                    "failure_propagation",
                    "integrated_branch_plan",
                    "integrated_branch_reporting_spec",
                    "misspecification_spec",
                    "output_registry",
                    "publication_eligible",
                    "publication_id",
                    "reporting_spec",
                    "schema_version",
                    "synthetic_only",
                }
            ),
            "AAMAS publication plan",
        )
        if (
            value["claim_scope"] != AAMAS_PUBLICATION_CLAIM_SCOPE
            or value["failure_propagation"]
            != "fail_closed_no_complete_case_or_missing_output"
            or value["synthetic_only"] is not True
            or value["publication_eligible"] is not False
            or value["attempt_registry"] != list(_attempt_definitions())
            or value["component_registry"] != list(_component_definitions())
            or value["output_registry"] != list(_output_definitions())
        ):
            _integrity_fail("publication plan authority or frozen registry changed")
        schema = value["schema_version"]
        if not isinstance(schema, str):
            _integrity_fail("publication schema_version must be a string")
        result = cls(
            publication_id=_identifier(value["publication_id"], "publication_id"),
            core_plan=AAMASSubmissionPlan.from_mapping(
                _mapping(value["core_plan"], "core_plan")
            ),
            integrated_branch_plan=AAMASIntegratedBranchPlan.from_mapping(
                _mapping(value["integrated_branch_plan"], "integrated_branch_plan")
            ),
            integrated_branch_reporting_spec=(
                FrozenIntegratedBranchReportingSpec.from_mapping(
                    _mapping(
                        value["integrated_branch_reporting_spec"],
                        "integrated_branch_reporting_spec",
                    )
                )
            ),
            misspecification_spec=AAMASMisspecificationSpec.from_mapping(
                _mapping(value["misspecification_spec"], "misspecification_spec")
            ),
            reporting_spec=FrozenAAMASReportingSpec.from_mapping(
                _mapping(value["reporting_spec"], "reporting_spec")
            ),
            synthetic_only=True,
            publication_eligible=False,
            schema_version=schema,
        )
        if result.to_mapping() != dict(value):
            _integrity_fail("publication plan is not in canonical form")
        return result


@dataclass(frozen=True, slots=True)
class AAMASPublicationStudyCommitment:
    """Caller-retained pre-run commitment to code, plan, and lock bytes."""

    plan_sha256: str
    source_tree_sha256: str
    uv_lock_sha256: str
    git_commit: str | None
    git_dirty_at_precommit: bool
    source_scope: str = "AGENTS_pyproject_uv_lock_and_all_regular_src_policytwin_files"
    schema_version: str = AAMAS_PUBLICATION_STUDY_COMMITMENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _sha256(self.plan_sha256, "study commitment plan_sha256")
        _sha256(self.source_tree_sha256, "study commitment source_tree_sha256")
        _sha256(self.uv_lock_sha256, "study commitment uv_lock_sha256")
        if (
            self.git_commit is not None
            and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", self.git_commit) is None
        ):
            _fail("study commitment git_commit is invalid")
        if not isinstance(self.git_dirty_at_precommit, bool):
            _fail("study commitment git dirty flag must be boolean")
        if self.source_scope != (
            "AGENTS_pyproject_uv_lock_and_all_regular_src_policytwin_files"
        ):
            _fail("study commitment source scope cannot change")
        if self.schema_version != AAMAS_PUBLICATION_STUDY_COMMITMENT_SCHEMA_VERSION:
            _fail("unsupported AAMAS publication study commitment schema")

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def to_mapping(self) -> dict[str, object]:
        return {
            "git_commit": self.git_commit,
            "git_dirty_at_precommit": self.git_dirty_at_precommit,
            "plan_sha256": self.plan_sha256,
            "schema_version": self.schema_version,
            "source_scope": self.source_scope,
            "source_tree_sha256": self.source_tree_sha256,
            "uv_lock_sha256": self.uv_lock_sha256,
        }

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> AAMASPublicationStudyCommitment:
        _strict_fields(
            value,
            frozenset(
                {
                    "git_commit",
                    "git_dirty_at_precommit",
                    "plan_sha256",
                    "schema_version",
                    "source_scope",
                    "source_tree_sha256",
                    "uv_lock_sha256",
                }
            ),
            "AAMAS publication study commitment",
        )
        git_commit = value["git_commit"]
        if git_commit is not None and not isinstance(git_commit, str):
            _integrity_fail("study commitment git_commit must be null or a string")
        dirty = value["git_dirty_at_precommit"]
        if not isinstance(dirty, bool):
            _integrity_fail("study commitment git dirty flag must be boolean")
        source_scope = value["source_scope"]
        schema_version = value["schema_version"]
        if not isinstance(source_scope, str) or not isinstance(schema_version, str):
            _integrity_fail("study commitment scope and schema must be strings")
        result = cls(
            plan_sha256=_sha256(value["plan_sha256"], "plan_sha256"),
            source_tree_sha256=_sha256(
                value["source_tree_sha256"], "source_tree_sha256"
            ),
            uv_lock_sha256=_sha256(value["uv_lock_sha256"], "uv_lock_sha256"),
            git_commit=git_commit,
            git_dirty_at_precommit=dirty,
            source_scope=source_scope,
            schema_version=schema_version,
        )
        if result.to_mapping() != dict(value):
            _integrity_fail("study commitment is not in canonical form")
        return result


@dataclass(frozen=True, slots=True)
class AAMASPublicationTiming:
    """One bound operational duration excluded from scientific statistics."""

    component_id: str
    wall_seconds: float

    def __post_init__(self) -> None:
        if self.component_id not in _TIMING_COMPONENTS:
            _fail("publication timing component is not registered")
        if (
            isinstance(self.wall_seconds, bool)
            or not isinstance(self.wall_seconds, int | float)
            or not math.isfinite(float(self.wall_seconds))
            or float(self.wall_seconds) < 0.0
        ):
            _fail("publication timing must be finite and non-negative")
        object.__setattr__(self, "wall_seconds", float(self.wall_seconds))

    def to_mapping(self, plan_sha256: str) -> dict[str, object]:
        return {
            "component_id": self.component_id,
            "event": "observed",
            "event_key": sha256_bytes(
                canonical_json_bytes(
                    {
                        "component_id": self.component_id,
                        "plan_sha256": plan_sha256,
                        "purpose": "operational_wall_time",
                    }
                )
            ),
            "measurement_scope": "operational_non_scientific_process",
            "schema_version": 1,
            "wall_seconds": self.wall_seconds,
        }


def _network_policy() -> CutoffPolicy:
    return CutoffPolicy(
        schema_version=1,
        training_end=datetime(2020, 1, 1, tzinfo=UTC),
        validation_end=datetime(2020, 1, 2, tzinfo=UTC),
        policy_start=datetime(2020, 1, 3, tzinfo=UTC),
        cutoff=datetime(2020, 1, 4, tzinfo=UTC),
        allowed_license_tags=("synthetic",),
    )


def _run_frozen[T](
    action: Callable[[], T], *, prefix: str
) -> tuple[T, tuple[dict[str, object], ...]]:
    with tempfile.TemporaryDirectory(prefix=prefix) as temporary:
        audit = TemporalAuditLog(Path(temporary) / "network.jsonl")
        firewall = TemporalFirewall(_network_policy(), audit)
        with firewall.frozen_evaluation():
            result = action()
        events = tuple(dict(item) for item in audit.events())
    return result, events


def _capture_action[T](action: Callable[[], T]) -> tuple[T | None, Exception | None]:
    try:
        return action(), None
    except Exception as error:
        return None, error


def _capture_frozen[T](
    action: Callable[[], T], *, prefix: str
) -> tuple[T | None, Exception | None, tuple[dict[str, object], ...]]:
    with tempfile.TemporaryDirectory(prefix=prefix) as temporary:
        audit = TemporalAuditLog(Path(temporary) / "network.jsonl")
        firewall = TemporalFirewall(_network_policy(), audit)
        result: T | None = None
        failure: Exception | None = None
        with firewall.frozen_evaluation():
            result, failure = _capture_action(action)
        events = tuple(dict(item) for item in audit.events())
    return result, failure, events


def _validate_network_events(
    events: Sequence[Mapping[str, object]], *, context: str
) -> bool:
    if len(events) < 2:
        _integrity_fail(f"{context} no-network boundary is incomplete")
    if (
        events[0].get("action") != "enter_frozen_evaluation"
        or events[0].get("outcome") != "allowed"
        or events[-1].get("action") != "exit_frozen_evaluation"
        or events[-1].get("outcome") != "allowed"
    ):
        _integrity_fail(f"{context} no-network boundary is incomplete")
    denied = events[1:-1]
    if any(
        item.get("action") != "network_access"
        or item.get("outcome") != "rejected"
        or item.get("rule_code") != "NETWORK_DENIED"
        for item in denied
    ):
        _integrity_fail(f"{context} contains an unrecognized network audit event")
    policy_sha256 = _network_policy().sha256
    if any(item.get("policy_sha256") != policy_sha256 for item in events):
        _integrity_fail(f"{context} no-network policy changed")
    return bool(denied)


@dataclass(frozen=True, slots=True)
class AAMASPublicationRun:
    """In-memory children produced from one externally committed plan."""

    plan: AAMASPublicationPlan
    expected_plan_sha256: str
    study_commitment: AAMASPublicationStudyCommitment
    expected_study_commitment_sha256: str
    source_manifest_payload: bytes
    core_run: AAMASSubmissionRun | None
    integrated_branch_run: AAMASIntegratedBranchRun | None
    misspecification_artifact: AAMASMisspecificationArtifact | None
    execution_attempts: tuple[AAMASPublicationAttempt, ...]
    timings: tuple[AAMASPublicationTiming, ...]
    core_network_events: tuple[dict[str, object], ...]
    integrated_branch_network_events: tuple[dict[str, object], ...]
    misspecification_network_events: tuple[dict[str, object], ...]
    started_at: datetime
    finished_at: datetime
    formal_scale_authorized: bool

    def __post_init__(self) -> None:
        if not isinstance(self.plan, AAMASPublicationPlan):
            _fail("publication run plan has the wrong type")
        if self.expected_plan_sha256 != self.plan.sha256:
            _fail("publication run differs from the external plan commitment")
        if not isinstance(self.study_commitment, AAMASPublicationStudyCommitment):
            _fail("publication run lacks a typed pre-run study commitment")
        if (
            self.study_commitment.plan_sha256 != self.plan.sha256
            or self.expected_study_commitment_sha256 != self.study_commitment.sha256
        ):
            _fail("publication run differs from the pre-run study commitment")
        source = read_json_bytes_object(
            self.source_manifest_payload,
            source="in-memory pre-run source manifest",
        )
        if (
            source.get("tree_sha256") != self.study_commitment.source_tree_sha256
            or source.get("uv_lock_sha256") != self.study_commitment.uv_lock_sha256
        ):
            _fail("publication run source snapshot differs from study commitment")
        if self.core_run is not None and self.core_run.plan != self.plan.core_plan:
            _fail("core run differs from its committed plan")
        if (
            self.integrated_branch_run is not None
            and self.integrated_branch_run.plan != self.plan.integrated_branch_plan
        ):
            _fail("integrated branch run differs from its committed plan")
        if (
            self.misspecification_artifact is not None
            and self.misspecification_artifact.plan.spec
            != self.plan.misspecification_spec
        ):
            _fail("misspecification run differs from its committed spec")
        expected_attempt_ids = tuple(
            cast(str, item["attempt_id"]) for item in _attempt_definitions()[:3]
        )
        if tuple(item.attempt_id for item in self.execution_attempts) != (
            expected_attempt_ids
        ):
            _fail("execution attempts differ from the fixed registry")
        outputs = (
            self.core_run,
            self.integrated_branch_run,
            self.misspecification_artifact,
        )
        for attempt_row, output in zip(self.execution_attempts, outputs, strict=True):
            if (attempt_row.status is AAMASPublicationAttemptStatus.COMPLETED) is (
                output is None
            ):
                _fail("execution terminal state disagrees with retained output")
        if tuple(item.component_id for item in self.timings) != _TIMING_COMPONENTS[:3]:
            _fail("publication run timings have changed membership or order")
        core_network_denied = _validate_network_events(
            self.core_network_events,
            context="core execution",
        )
        branch_network_denied = _validate_network_events(
            self.integrated_branch_network_events,
            context="integrated branch execution",
        )
        misspecification_network_denied = _validate_network_events(
            self.misspecification_network_events,
            context="misspecification execution",
        )
        for denied, attempt_row in zip(
            (
                core_network_denied,
                branch_network_denied,
                misspecification_network_denied,
            ),
            self.execution_attempts,
            strict=True,
        ):
            if (
                denied
                and attempt_row.status is not AAMASPublicationAttemptStatus.FAILED
            ):
                _fail("denied network access lacks a failed terminal attempt")
        if not isinstance(self.formal_scale_authorized, bool):
            _fail("formal_scale_authorized must be boolean")
        if self.plan.requires_formal_authorization and not self.formal_scale_authorized:
            _fail("formal publication run lacks explicit authorization")
        for name in ("started_at", "finished_at"):
            value = getattr(self, name)
            if not isinstance(value, datetime) or value.tzinfo is None:
                _fail(f"{name} must be timezone aware")
            object.__setattr__(self, name, value.astimezone(UTC))
        if self.finished_at < self.started_at:
            _fail("publication run finished before it started")


def _require_execution_authority(
    plan: AAMASPublicationPlan, allow_formal_scale: bool
) -> None:
    if not isinstance(allow_formal_scale, bool):
        _fail("allow_formal_scale must be boolean")
    if plan.requires_formal_authorization and not allow_formal_scale:
        _integrity_fail(
            "formal or same-or-larger publication work requires allow_formal_scale=True"
        )


def run_aamas_publication(
    plan: AAMASPublicationPlan,
    *,
    expected_plan_sha256: str,
    study_commitment: AAMASPublicationStudyCommitment,
    expected_study_commitment_sha256: str,
    project_root: Path | None = None,
    allow_formal_scale: bool = False,
) -> AAMASPublicationRun:
    """Run all synthetic children, with branch controls under no-network guards."""

    if not isinstance(plan, AAMASPublicationPlan):
        _fail("plan must be an AAMASPublicationPlan")
    if _sha256(expected_plan_sha256, "expected_plan_sha256") != plan.sha256:
        _integrity_fail("publication plan differs from caller-held precommitment")
    _require_execution_authority(plan, allow_formal_scale)
    if not isinstance(study_commitment, AAMASPublicationStudyCommitment):
        _fail("run requires a typed caller-held study commitment")
    expected_study = _sha256(
        expected_study_commitment_sha256,
        "expected_study_commitment_sha256",
    )
    if (
        study_commitment.plan_sha256 != plan.sha256
        or expected_study != study_commitment.sha256
    ):
        _integrity_fail("run differs from caller-held study commitment")
    project = (
        Path(__file__).resolve().parents[3]
        if project_root is None
        else project_root.resolve()
    )
    source_mapping = _source_manifest(project)
    current_git = collect_git_metadata(project)
    if (
        source_mapping["tree_sha256"] != study_commitment.source_tree_sha256
        or source_mapping["uv_lock_sha256"] != study_commitment.uv_lock_sha256
        or current_git.commit != study_commitment.git_commit
        or current_git.dirty != study_commitment.git_dirty_at_precommit
    ):
        _integrity_fail("source, lock, or Git state changed after study commitment")
    source_manifest_payload = _pretty_json_bytes(source_mapping)
    started_at = datetime.now(UTC)
    timings: list[AAMASPublicationTiming] = []

    start = time.perf_counter()
    core, core_error, core_events = _capture_frozen(
        lambda: run_aamas_submission(
            plan.core_plan,
            expected_plan_sha256=plan.core_plan.sha256,
            allow_formal_scale=allow_formal_scale,
        ),
        prefix="policytwin-aamas-publication-core-",
    )
    timings.append(
        AAMASPublicationTiming("core_execution", time.perf_counter() - start)
    )

    start = time.perf_counter()
    branch, branch_error, branch_events = _capture_frozen(
        lambda: run_aamas_integrated_branch(
            plan.integrated_branch_plan,
            allow_formal_scale=allow_formal_scale,
        ),
        prefix="policytwin-aamas-publication-branch-",
    )
    timings.append(
        AAMASPublicationTiming(
            "integrated_branch_execution", time.perf_counter() - start
        )
    )

    start = time.perf_counter()
    misspecification, misspecification_error, misspecification_events = _capture_frozen(
        lambda: run_aamas_misspecification_benchmark(
            plan.misspecification_spec,
            allow_formal_scale=allow_formal_scale,
        ),
        prefix="policytwin-aamas-publication-misspec-",
    )
    timings.append(
        AAMASPublicationTiming(
            "misspecification_execution", time.perf_counter() - start
        )
    )
    execution_attempts = (
        _attempt(
            "core_execution",
            config_sha256=plan.core_plan.sha256,
            status=(
                AAMASPublicationAttemptStatus.COMPLETED
                if core_error is None
                else AAMASPublicationAttemptStatus.FAILED
            ),
            error=core_error,
        ),
        _attempt(
            "integrated_branch_execution",
            config_sha256=plan.integrated_branch_plan.sha256,
            status=(
                AAMASPublicationAttemptStatus.COMPLETED
                if branch_error is None
                else AAMASPublicationAttemptStatus.FAILED
            ),
            output_sha256=(branch.artifact.sha256 if branch is not None else None),
            error=branch_error,
        ),
        _attempt(
            "misspecification_execution",
            config_sha256=plan.misspecification_spec.sha256,
            status=(
                AAMASPublicationAttemptStatus.COMPLETED
                if misspecification_error is None
                else AAMASPublicationAttemptStatus.FAILED
            ),
            output_sha256=(
                misspecification.sha256 if misspecification is not None else None
            ),
            error=misspecification_error,
        ),
    )
    return AAMASPublicationRun(
        plan=plan,
        expected_plan_sha256=expected_plan_sha256,
        study_commitment=study_commitment,
        expected_study_commitment_sha256=expected_study,
        source_manifest_payload=source_manifest_payload,
        core_run=core,
        integrated_branch_run=branch,
        misspecification_artifact=misspecification,
        execution_attempts=execution_attempts,
        timings=tuple(timings),
        core_network_events=core_events,
        integrated_branch_network_events=branch_events,
        misspecification_network_events=misspecification_events,
        started_at=started_at,
        finished_at=datetime.now(UTC),
        formal_scale_authorized=allow_formal_scale,
    )


def _data_manifest() -> dict[str, object]:
    return {
        "claim_scope": "synthetic_empty_manifest",
        "manifest_id": "aamas-publication-synthetic-empty-v1",
        "real_record_count": 0,
        "records": [],
        "schema_version": 1,
        "synthetic_only": True,
    }


def _stable_file_row(project_root: Path, path: Path) -> dict[str, object]:
    before = path.lstat()
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_nlink != 1
    ):
        _integrity_fail("local source members must be singly linked regular files")
    content = path.read_bytes()
    after = path.lstat()

    def identity(item: os.stat_result) -> tuple[int, int, int, int, int, int]:
        return (
            item.st_dev,
            item.st_ino,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
            item.st_nlink,
        )

    if identity(before) != identity(after) or len(content) != before.st_size:
        _integrity_fail("local source changed while its manifest was built")
    return {
        "path": path.relative_to(project_root).as_posix(),
        "sha256": sha256_bytes(content),
        "size_bytes": len(content),
    }


def _source_paths(project_root: Path) -> tuple[Path, ...]:
    package = project_root / "src" / "policytwin"
    if not package.is_dir() or package.is_symlink():
        _integrity_fail("project_root lacks a real src/policytwin package")
    required = tuple(
        project_root / name for name in ("AGENTS.md", "pyproject.toml", "uv.lock")
    )
    for path in required:
        if not path.is_file() or path.is_symlink():
            _integrity_fail(f"required reproducibility input is missing: {path.name}")
    paths: list[Path] = list(required)
    for candidate in package.rglob("*"):
        details = candidate.lstat()
        if stat.S_ISLNK(details.st_mode):
            _integrity_fail("local package source contains a symlink")
        if stat.S_ISREG(details.st_mode):
            if "__pycache__" in candidate.parts or candidate.suffix in {".pyc", ".pyo"}:
                continue
            paths.append(candidate)
    return tuple(
        sorted(paths, key=lambda item: item.relative_to(project_root).as_posix())
    )


def _source_manifest(project_root: Path) -> dict[str, object]:
    rows = [
        _stable_file_row(project_root, path) for path in _source_paths(project_root)
    ]
    return {
        "authority": "local_worktree_bytes_not_external_attestation",
        "complete_for_scope": True,
        "files": rows,
        "schema_version": 1,
        "scope": "AGENTS_pyproject_uv_lock_and_all_regular_src_policytwin_files",
        "source_count": len(rows),
        "tree_sha256": sha256_bytes(canonical_json_bytes(rows)),
        "uv_lock_sha256": sha256_file(project_root / "uv.lock"),
    }


def build_aamas_publication_study_commitment(
    plan: AAMASPublicationPlan,
    *,
    project_root: Path | None = None,
) -> AAMASPublicationStudyCommitment:
    """Freeze the plan, source tree, lock, and Git commit before execution."""

    if not isinstance(plan, AAMASPublicationPlan):
        _fail("study commitment requires an AAMASPublicationPlan")
    project = (
        Path(__file__).resolve().parents[3]
        if project_root is None
        else project_root.resolve()
    )
    source = _source_manifest(project)
    git = collect_git_metadata(project)
    return AAMASPublicationStudyCommitment(
        plan_sha256=plan.sha256,
        source_tree_sha256=cast(str, source["tree_sha256"]),
        uv_lock_sha256=cast(str, source["uv_lock_sha256"]),
        git_commit=git.commit,
        git_dirty_at_precommit=git.dirty,
    )


def _portable_git_mapping(
    commitment: AAMASPublicationStudyCommitment,
) -> dict[str, object]:
    """Record generation Git identity without an absolute repository path."""

    return {
        "commit": commitment.git_commit,
        "dirty": commitment.git_dirty_at_precommit,
        "root": "." if commitment.git_commit is not None else None,
    }


def _require_git_commitment_match(
    git: GitMetadata,
    commitment: AAMASPublicationStudyCommitment,
    *,
    context: str,
) -> None:
    if (
        git.commit != commitment.git_commit
        or git.dirty != commitment.git_dirty_at_precommit
    ):
        _integrity_fail(f"{context} Git identity differs from study commitment")


def _audit_core_git_bindings(
    core_directory: Path,
    commitment: AAMASPublicationStudyCommitment,
) -> None:
    """Cross-bind child provenance to the outer pre-run Git commitment."""

    metadata = read_json_object(core_directory / "metadata.json")
    run_metadata = RunMetadata.from_mapping(
        _mapping(metadata.get("run_metadata"), "core run_metadata")
    )
    _require_git_commitment_match(
        run_metadata.git,
        commitment,
        context="core RunMetadata",
    )
    provenance_paths = tuple(
        sorted((core_directory / "calibration").rglob("provenance.json"))
    )
    if not provenance_paths:
        _integrity_fail("core calibration contains no provenance records")
    for path in provenance_paths:
        provenance = read_json_object(path)
        git = GitMetadata.from_mapping(
            _mapping(provenance.get("git"), f"{path.name} git metadata")
        )
        _require_git_commitment_match(
            git,
            commitment,
            context=f"core calibration provenance {path.relative_to(core_directory)}",
        )


def _tree_rows(directory: Path) -> tuple[dict[str, object], ...]:
    if not directory.is_dir() or directory.is_symlink():
        _integrity_fail("bundle tree root must be a real directory")
    rows: list[dict[str, object]] = []
    for path in sorted(directory.rglob("*")):
        details = path.lstat()
        if stat.S_ISLNK(details.st_mode):
            _integrity_fail("bundle tree contains a symbolic link")
        if stat.S_ISDIR(details.st_mode):
            continue
        if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
            _integrity_fail("bundle members must be singly linked regular files")
        relative = path.relative_to(directory).as_posix()
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != relative:
            _integrity_fail("bundle member path is unsafe")
        rows.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "size_bytes": details.st_size,
            }
        )
    return tuple(rows)


def _strict_audit_tree_rows(directory: Path) -> tuple[dict[str, object], ...]:
    """Capture a race-checked identity for every directory and file in a bundle."""

    def fingerprint(info: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
        return (
            info.st_dev,
            info.st_ino,
            info.st_mode,
            info.st_nlink,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        )

    try:
        root_before = directory.lstat()
    except OSError as error:
        raise AAMASPublicationIntegrityError(
            f"cannot inspect publication audit root: {error}"
        ) from error
    if not stat.S_ISDIR(root_before.st_mode) or stat.S_ISLNK(root_before.st_mode):
        _integrity_fail("publication audit root must be a real directory")
    initial_paths = tuple(
        sorted(
            directory.rglob("*"),
            key=lambda item: item.relative_to(directory).as_posix(),
        )
    )
    rows: list[dict[str, object]] = []
    for path in initial_paths:
        relative = path.relative_to(directory).as_posix()
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != relative:
            _integrity_fail("publication audit tree contains an unsafe path")
        try:
            before = path.lstat()
        except OSError as error:
            raise AAMASPublicationIntegrityError(
                f"cannot inspect publication audit member {relative}: {error}"
            ) from error
        if stat.S_ISLNK(before.st_mode):
            _integrity_fail("publication audit tree contains a symbolic link")
        if stat.S_ISDIR(before.st_mode):
            row: dict[str, object] = {
                "kind": "directory",
                "mode": stat.S_IMODE(before.st_mode),
                "path": relative,
            }
        elif stat.S_ISREG(before.st_mode) and before.st_nlink == 1:
            try:
                content = path.read_bytes()
            except OSError as error:
                raise AAMASPublicationIntegrityError(
                    f"cannot read publication audit member {relative}: {error}"
                ) from error
            if len(content) != before.st_size:
                _integrity_fail("publication audit member changed while read")
            row = {
                "kind": "file",
                "mode": stat.S_IMODE(before.st_mode),
                "path": relative,
                "sha256": sha256_bytes(content),
                "size_bytes": len(content),
            }
        else:
            _integrity_fail(
                "publication audit members must be directories or singly linked files"
            )
        try:
            after = path.lstat()
        except OSError as error:
            raise AAMASPublicationIntegrityError(
                f"cannot recheck publication audit member {relative}: {error}"
            ) from error
        if fingerprint(before) != fingerprint(after):
            _integrity_fail("publication audit member changed while captured")
        rows.append(row)
    try:
        root_after = directory.lstat()
        final_paths = tuple(
            sorted(
                item.relative_to(directory).as_posix()
                for item in directory.rglob("*")
            )
        )
    except OSError as error:
        raise AAMASPublicationIntegrityError(
            f"cannot finalize publication audit tree: {error}"
        ) from error
    initial_names = tuple(
        path.relative_to(directory).as_posix() for path in initial_paths
    )
    if (
        fingerprint(root_before) != fingerprint(root_after)
        or initial_names != final_paths
    ):
        _integrity_fail("publication audit tree changed while captured")
    return tuple(rows)


def _core_calibration_seal(core_directory: Path) -> str:
    manifest = read_json_object(core_directory / "manifest.json")
    rows = _sequence(manifest.get("subbundles"), "core subbundles")
    matches = [
        _mapping(row, "core subbundle")
        for row in rows
        if _mapping(row, "core subbundle").get("path") == "calibration"
    ]
    if len(matches) != 1:
        _integrity_fail("core manifest lacks one calibration subbundle")
    return _sha256(
        matches[0].get("external_seal_raw_sha256"),
        "core calibration seal",
    )


def _terminal_directory(directory: Path, attempt_row: AAMASPublicationAttempt) -> str:
    """Replace only staging-owned partial output with one immutable terminal row."""

    if directory.exists() or directory.is_symlink():
        if directory.is_symlink():
            _integrity_fail("cannot replace a symbolic-link child with a terminal row")
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=False)
    mapping = {
        "attempt": attempt_row.to_mapping(),
        "schema_version": 1,
        "terminal": True,
    }
    payload = _pretty_json_bytes(mapping)
    _write_new(directory / "terminal.json", payload)
    return sha256_bytes(canonical_json_bytes(mapping))


def _audit_terminal_directory(
    directory: Path, attempt_row: AAMASPublicationAttempt
) -> str:
    if directory.is_symlink() or not directory.is_dir():
        _integrity_fail("terminal child must be a real directory")
    if frozenset(item.name for item in directory.iterdir()) != {"terminal.json"}:
        _integrity_fail("terminal child fixed membership changed")
    path = directory / "terminal.json"
    details = path.lstat()
    if (
        not stat.S_ISREG(details.st_mode)
        or stat.S_ISLNK(details.st_mode)
        or details.st_nlink != 1
    ):
        _integrity_fail("terminal child must be singly linked and regular")
    expected = {
        "attempt": attempt_row.to_mapping(),
        "schema_version": 1,
        "terminal": True,
    }
    if read_json_object(path) != expected:
        _integrity_fail("terminal child differs from its attempt registry row")
    return sha256_bytes(canonical_json_bytes(expected))


def _attempt_registry_mapping(
    plan: AAMASPublicationPlan,
    attempts: Sequence[AAMASPublicationAttempt],
) -> dict[str, object]:
    expected_ids = tuple(
        cast(str, definition["attempt_id"]) for definition in _attempt_definitions()
    )
    if tuple(item.attempt_id for item in attempts) != expected_ids:
        _integrity_fail("terminal attempt registry differs from the frozen plan")
    return {
        "attempts": [item.to_mapping() for item in attempts],
        "definitions": list(_attempt_definitions()),
        "failure_propagation": "every_expected_attempt_has_one_terminal_state",
        "plan_sha256": plan.sha256,
        "schema_version": 1,
    }


def _load_attempt_registry(
    path: Path, plan: AAMASPublicationPlan
) -> tuple[AAMASPublicationAttempt, ...]:
    mapping = read_json_object(path)
    _strict_fields(
        mapping,
        frozenset(
            {
                "attempts",
                "definitions",
                "failure_propagation",
                "plan_sha256",
                "schema_version",
            }
        ),
        "publication attempt registry",
    )
    if (
        mapping["definitions"] != list(_attempt_definitions())
        or mapping["failure_propagation"]
        != "every_expected_attempt_has_one_terminal_state"
        or mapping["plan_sha256"] != plan.sha256
        or mapping["schema_version"] != 1
    ):
        _integrity_fail("publication attempt registry authority changed")
    attempts = tuple(
        AAMASPublicationAttempt.from_mapping(_mapping(item, "attempt row"))
        for item in _sequence(mapping["attempts"], "attempts")
    )
    if _attempt_registry_mapping(plan, attempts) != mapping:
        _integrity_fail("publication attempts do not match their fixed registry")
    expected_configs = {
        "core_execution": plan.core_plan.sha256,
        "integrated_branch_execution": plan.integrated_branch_plan.sha256,
        "misspecification_execution": plan.misspecification_spec.sha256,
        "core_publication": plan.core_plan.sha256,
        "integrated_branch_publication": plan.integrated_branch_plan.sha256,
        "misspecification_publication": plan.misspecification_spec.sha256,
        "core_reporting": plan.reporting_spec.sha256,
        "integrated_branch_reporting": (plan.integrated_branch_reporting_spec.sha256),
    }
    definitions = {item["attempt_id"]: item for item in _attempt_definitions()}
    for item in attempts:
        definition = definitions[item.attempt_id]
        if (
            item.config_sha256 != expected_configs[item.attempt_id]
            or item.component_id != definition["component_id"]
            or item.stage != definition["stage"]
        ):
            _integrity_fail("publication attempt identity or config changed")
    if any(
        item.status
        not in {
            AAMASPublicationAttemptStatus.COMPLETED,
            AAMASPublicationAttemptStatus.FAILED,
        }
        for item in attempts[:3]
    ):
        _integrity_fail("execution attempts must be completed or failed")
    for item in attempts[3:]:
        dependencies = cast(list[str], definitions[item.attempt_id]["dependencies"])
        upstream_completed = all(
            next(row for row in attempts if row.attempt_id == dependency).status
            is AAMASPublicationAttemptStatus.COMPLETED
            for dependency in dependencies
        )
        if not upstream_completed:
            if item.status is not AAMASPublicationAttemptStatus.BLOCKED_UPSTREAM:
                _integrity_fail("failed upstream must produce blocked terminal state")
        elif item.status not in {
            AAMASPublicationAttemptStatus.COMPLETED,
            AAMASPublicationAttemptStatus.FAILED,
        }:
            _integrity_fail("available attempt cannot claim an upstream block")
    return attempts


def _validity_certificate(
    plan: AAMASPublicationPlan,
    *,
    required_mechanism_receipt_valid: bool,
) -> ValidityCertificate:
    evidence = ValidityEvidence(
        temporal_leakage_detected=None,
        pre_policy_ppc_failure_rate=None,
        ood_policy_distance=None,
        model_disagreement=None,
        seed_instability=None,
        empirical_coverage_history=None,
        observed_group_labels=None,
        active_mechanisms=(
            plan.core_plan.validity_thresholds.required_mechanisms
            if required_mechanism_receipt_valid
            else None
        ),
        prompt_instability=None,
        parameter_support_failure_rate=None,
    )
    certificate = build_validity_certificate(
        evidence,
        plan.core_plan.validity_thresholds,
        synthetic_only=True,
    )
    if certificate.level is not ValidityLevel.ABSTAIN:
        _integrity_fail("outer validity certificate must authoritatively abstain")
    return certificate


@dataclass(frozen=True, slots=True)
class _PublicationSources:
    core: AAMASSubmissionAuditReport | None
    branch: AAMASIntegratedBranchAudit | None
    misspecification: MisspecificationAuditReport | None
    reporting: AAMASCoreReportingAudit | None
    integrated_branch_reporting: AAMASIntegratedBranchReportAudit | None
    branch_artifact: Mapping[str, object] | None
    misspecification_artifact: Mapping[str, object] | None
    reporting_artifact: AAMASCoreReportingArtifact | None
    integrated_branch_reporting_artifact: AAMASIntegratedBranchReportArtifact | None
    attempts: tuple[AAMASPublicationAttempt, ...]


@dataclass(frozen=True, slots=True)
class _PublicationDerivation:
    result: dict[str, object]
    output_registry: dict[str, object]
    failures: tuple[dict[str, object], ...]
    certificate: ValidityCertificate
    submission_evidence_complete: bool
    scientifically_complete: bool


def _derive_publication(
    plan: AAMASPublicationPlan, sources: _PublicationSources
) -> _PublicationDerivation:
    attempt_by_id = {item.attempt_id: item for item in sources.attempts}
    if tuple(attempt_by_id) != tuple(
        cast(str, item["attempt_id"]) for item in _attempt_definitions()
    ):
        _integrity_fail("derivation attempt registry changed")
    branch_count = 0
    branch_complete = False
    if sources.branch is not None and sources.branch_artifact is not None:
        branch_count = _integer(
            sources.branch_artifact.get("branch_replicate_count"),
            "branch_replicate_count",
        )
        branch_complete = (
            sources.branch_artifact.get("scientifically_complete_for_claim") is True
        )
    misspecification_failed = (
        sources.misspecification.failed_count
        if sources.misspecification is not None
        else plan.misspecification_spec.design_truth_count * 2
        + plan.misspecification_spec.target_truth_count * 2
    )
    decision: str | None = None
    if sources.misspecification_artifact is not None:
        assessment = _mapping(
            sources.misspecification_artifact.get("assessment"),
            "misspecification assessment",
        )
        raw_decision = assessment.get("decision")
        if raw_decision not in {item.value for item in NegativeControlDecision}:
            _integrity_fail("misspecification assessment decision is invalid")
        decision = cast(str, raw_decision)
    misspecification_complete = (
        sources.misspecification is not None and misspecification_failed == 0
    )
    misspecification_gate = (
        misspecification_complete and decision == NegativeControlDecision.DETECTED.value
    )
    certificate = _validity_certificate(
        plan,
        required_mechanism_receipt_valid=(branch_complete and branch_count > 0),
    )

    core_children_complete = (
        sources.core is not None and sources.core.core_registered_outputs_complete
    )
    core_reporting_complete = sources.reporting is not None
    branch_reporting_complete = sources.integrated_branch_reporting is not None

    def unavailable_status(attempt_id: str) -> str:
        status = attempt_by_id[attempt_id].status
        if status in {
            AAMASPublicationAttemptStatus.BLOCKED_UPSTREAM,
        }:
            return "blocked_upstream"
        return "failed"

    statuses: dict[str, str] = {
        "core_scientific_result": (
            "complete"
            if core_children_complete
            else unavailable_status("core_publication")
        ),
        "integrated_branch_artifact": (
            "complete"
            if branch_complete
            else unavailable_status("integrated_branch_publication")
        ),
        "misspecification_negative_control": (
            "complete_detected"
            if misspecification_gate
            else (
                "complete_not_detected"
                if misspecification_complete
                else unavailable_status("misspecification_publication")
            )
        ),
        "core_frozen_reporting": (
            "complete"
            if core_reporting_complete
            else unavailable_status("core_reporting")
        ),
        "integrated_branch_frozen_reporting": (
            "complete"
            if branch_reporting_complete
            else unavailable_status("integrated_branch_reporting")
        ),
        "root_validity_certificate": "abstain_missing_external_authority",
    }
    artifact_hashes: dict[str, str | None] = {
        "core_scientific_result": (
            sources.core.scientific_result_sha256 if sources.core is not None else None
        ),
        "integrated_branch_artifact": (
            sources.branch.artifact_sha256 if sources.branch is not None else None
        ),
        "misspecification_negative_control": (
            sources.misspecification.artifact_sha256
            if sources.misspecification is not None
            else None
        ),
        "core_frozen_reporting": (
            sources.reporting.artifact_sha256 if sources.reporting is not None else None
        ),
        "integrated_branch_frozen_reporting": (
            sources.integrated_branch_reporting.report_sha256
            if sources.integrated_branch_reporting is not None
            else None
        ),
        "root_validity_certificate": sha256_bytes(
            canonical_json_bytes(certificate.to_mapping())
        ),
    }
    output_rows: list[dict[str, object]] = []
    for definition in _output_definitions():
        output_id = cast(str, definition["output_id"])
        output_rows.append(
            {
                "artifact_sha256": artifact_hashes[output_id],
                "definition": definition,
                "output_id": output_id,
                "status": statuses[output_id],
            }
        )
    output_registry = {
        "entries": output_rows,
        "failure_propagation": "fail_closed_no_complete_case_or_missing_output",
        "plan_sha256": plan.sha256,
        "schema_version": 1,
    }

    failures: list[dict[str, object]] = []
    for item in sources.attempts:
        if item.status is AAMASPublicationAttemptStatus.COMPLETED:
            continue
        failures.append(
            {
                "attempt": item.to_mapping(),
                "component_id": item.component_id,
                "failure_id": f"terminal_{item.attempt_id}",
                "severity": "blocking",
                "status": item.status.value,
            }
        )
    if not core_children_complete:
        failures.append(
            {
                "component_id": "core",
                "failure_id": "core_scientific_incomplete",
                "severity": "blocking",
                "status": "failed",
            }
        )
    if not branch_complete:
        failures.append(
            {
                "component_id": "integrated_branch",
                "failure_id": "integrated_branch_claim_incomplete",
                "severity": "blocking",
                "status": "failed",
            }
        )
    if not misspecification_complete:
        failures.append(
            {
                "component_id": "misspecification",
                "failed_attempt_count": misspecification_failed,
                "failure_id": "misspecification_attempt_failure",
                "severity": "blocking",
                "status": "failed",
            }
        )
    elif not misspecification_gate:
        failures.append(
            {
                "component_id": "misspecification",
                "decision": decision,
                "failure_id": "negative_control_not_detected",
                "severity": "blocking",
                "status": "failed",
            }
        )
    failures.append(
        {
            "component_id": "validity_certificate",
            "failure_id": "missing_external_validity_authority",
            "severity": "blocking",
            "status": "abstain",
        }
    )
    submission_evidence_complete = (
        core_children_complete
        and branch_complete
        and misspecification_gate
        and core_reporting_complete
        and branch_reporting_complete
        and all(
            item.status is AAMASPublicationAttemptStatus.COMPLETED
            for item in sources.attempts
        )
    )
    scientifically_complete = False
    failed_attempts = tuple(
        item.to_mapping()
        for item in sources.attempts
        if item.status is not AAMASPublicationAttemptStatus.COMPLETED
    )
    result = {
        "claim_scope": AAMAS_PUBLICATION_CLAIM_SCOPE,
        "components": {
            "core": {
                "core_children_complete": core_children_complete,
                "scientific_result_sha256": artifact_hashes["core_scientific_result"],
            },
            "core_reporting": {
                "artifact_sha256": artifact_hashes["core_frozen_reporting"],
                "complete": core_reporting_complete,
                "output_count": (
                    sources.reporting.output_count
                    if sources.reporting is not None
                    else 0
                ),
                "reporting_spec_sha256": plan.reporting_spec.sha256,
            },
            "integrated_branch_reporting": {
                "artifact_sha256": artifact_hashes[
                    "integrated_branch_frozen_reporting"
                ],
                "complete": branch_reporting_complete,
                "reporting_spec_sha256": (plan.integrated_branch_reporting_spec.sha256),
            },
            "integrated_branch": {
                "artifact_sha256": artifact_hashes["integrated_branch_artifact"],
                "branch_replicate_count": branch_count,
                "record_count": (
                    sources.branch.record_count if sources.branch is not None else 0
                ),
                "required_mechanism_evidence": {
                    "evidence_source": (
                        "integrated-branch/artifact.json"
                        if sources.branch is not None
                        else None
                    ),
                    "non_noop_clob_branch_receipt": (
                        branch_complete and branch_count > 0
                    ),
                    "old_rq1_ablation_used_as_authority": False,
                },
                "scientifically_complete_for_claim": branch_complete,
            },
            "misspecification": {
                "artifact_sha256": artifact_hashes["misspecification_negative_control"],
                "attempt_count": (
                    sources.misspecification.attempt_count
                    if sources.misspecification is not None
                    else 0
                ),
                "decision": decision,
                "failed_count": misspecification_failed,
                "negative_control_gate_passed": misspecification_gate,
            },
        },
        "failure_registry_sha256": sha256_bytes(canonical_json_bytes(failures)),
        "failed_or_blocked_attempts": list(failed_attempts),
        "integrated_branch_reporting_status": statuses[
            "integrated_branch_frozen_reporting"
        ],
        "output_registry_sha256": sha256_bytes(canonical_json_bytes(output_registry)),
        "plan_sha256": plan.sha256,
        "publication_eligible": False,
        "schema_version": AAMAS_PUBLICATION_RESULT_SCHEMA_VERSION,
        "scientifically_complete": scientifically_complete,
        "submission_evidence_complete": submission_evidence_complete,
        "synthetic_only": True,
        "validity_level": certificate.level.value,
    }
    return _PublicationDerivation(
        result=result,
        output_registry=output_registry,
        failures=tuple(failures),
        certificate=certificate,
        submission_evidence_complete=submission_evidence_complete,
        scientifically_complete=scientifically_complete,
    )


def _child_manifest_row(
    *,
    root: Path,
    path: str,
    audit_anchor_kind: str,
    audit_anchor_sha256: str,
    scientific_artifact_sha256: str | None,
    terminal_state_sha256: str,
) -> dict[str, object]:
    rows = _tree_rows(root / path)
    return {
        "audit_anchor_kind": audit_anchor_kind,
        "audit_anchor_sha256": _sha256(audit_anchor_sha256, f"{path} audit anchor"),
        "file_count": len(rows),
        "path": path,
        "scientific_artifact_sha256": (
            None
            if scientific_artifact_sha256 is None
            else _sha256(scientific_artifact_sha256, f"{path} scientific artifact")
        ),
        "terminal_state_sha256": _sha256(
            terminal_state_sha256, f"{path} terminal state"
        ),
        "tree_sha256": sha256_bytes(canonical_json_bytes(rows)),
    }


@dataclass(frozen=True, slots=True)
class AAMASPublicationWriteResult:
    directory: Path
    plan_sha256: str
    study_commitment_sha256: str
    root_seal_sha256: str
    scientific_result_sha256: str
    integrity_valid_at_write: Literal[True] = True
    submission_evidence_complete: bool = False
    scientifically_complete: Literal[False] = False
    publication_eligible: Literal[False] = False


def write_aamas_publication(
    directory: Path,
    *,
    run: AAMASPublicationRun,
    expected_plan_sha256: str,
    expected_study_commitment_sha256: str,
    project_root: Path | None = None,
    allow_formal_scale: bool = False,
) -> AAMASPublicationWriteResult:
    """Stage, audit, seal, and atomically publish one immutable outer bundle."""

    if not isinstance(directory, Path) or not isinstance(run, AAMASPublicationRun):
        _fail("writer requires a Path and AAMASPublicationRun")
    expected_plan = _sha256(expected_plan_sha256, "expected_plan_sha256")
    if expected_plan != run.plan.sha256 or expected_plan != run.expected_plan_sha256:
        _integrity_fail("writer plan differs from caller-held precommitment")
    expected_study = _sha256(
        expected_study_commitment_sha256,
        "expected_study_commitment_sha256",
    )
    if (
        expected_study != run.study_commitment.sha256
        or expected_study != run.expected_study_commitment_sha256
    ):
        _integrity_fail("writer differs from caller-held study commitment")
    _require_execution_authority(run.plan, allow_formal_scale)
    if run.formal_scale_authorized is not allow_formal_scale:
        _integrity_fail("writer authorization differs from execution authorization")
    project = (
        Path(__file__).resolve().parents[3]
        if project_root is None
        else project_root.resolve()
    )
    if directory.exists() or directory.is_symlink():
        _integrity_fail("publication output already exists")
    if directory.resolve().is_relative_to(project / "src" / "policytwin"):
        _integrity_fail("publication output cannot be placed inside source scope")
    committed_source = read_json_bytes_object(
        run.source_manifest_payload,
        source="pre-run source manifest",
    )
    if (
        committed_source != _source_manifest(project)
        or committed_source.get("tree_sha256")
        != run.study_commitment.source_tree_sha256
        or committed_source.get("uv_lock_sha256") != run.study_commitment.uv_lock_sha256
    ):
        _integrity_fail("source or lock changed after execution began")
    _require_git_commitment_match(
        collect_git_metadata(project),
        run.study_commitment,
        context="writer pre-publication",
    )
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{directory.name}.staging-", dir=directory.parent)
    )
    writer_timings: list[AAMASPublicationTiming] = []
    try:
        attempts = list(run.execution_attempts)
        core_write_seal: str | None = None
        misspecification_root_seal: str | None = None
        core_audit: AAMASSubmissionAuditReport | None = None
        branch_audit: AAMASIntegratedBranchAudit | None = None
        misspecification_audit: MisspecificationAuditReport | None = None
        reporting_audit: AAMASCoreReportingAudit | None = None
        reporting_artifact: AAMASCoreReportingArtifact | None = None
        branch_reporting_audit: AAMASIntegratedBranchReportAudit | None = None
        branch_reporting_artifact: AAMASIntegratedBranchReportArtifact | None = None
        branch_mapping: Mapping[str, object] | None = None
        misspecification_mapping: Mapping[str, object] | None = None
        child_terminal_sha256: dict[str, str] = {}

        start = time.perf_counter()
        core_publication_log = TemporalAuditLog(
            staging / "network-core-publication.jsonl"
        )
        if run.core_run is None:
            with TemporalFirewall(
                _network_policy(), core_publication_log
            ).frozen_evaluation():
                core_attempt = _attempt(
                    "core_publication",
                    config_sha256=run.plan.core_plan.sha256,
                    status=AAMASPublicationAttemptStatus.BLOCKED_UPSTREAM,
                    reason="core_execution_failed",
                )
            child_terminal_sha256["core"] = _terminal_directory(
                staging / "core", core_attempt
            )
        else:
            try:
                core_git_metadata = collect_git_metadata(project)
                _require_git_commitment_match(
                    core_git_metadata,
                    run.study_commitment,
                    context="core publication pre-write",
                )
                with TemporalFirewall(
                    _network_policy(), core_publication_log
                ).frozen_evaluation():
                    core_write = write_aamas_submission(
                        staging / "core",
                        run=run.core_run,
                        expected_plan_sha256=run.plan.core_plan.sha256,
                        project_root=project,
                        git_metadata=core_git_metadata,
                    )
                    core_write_seal = core_write.seal_raw_sha256
                    core_audit = audit_aamas_submission(
                        staging / "core",
                        expected_plan=run.plan.core_plan,
                        expected_plan_sha256=run.plan.core_plan.sha256,
                        expected_seal_sha256=core_write_seal,
                        project_root=project,
                    )
                    _audit_core_git_bindings(
                        staging / "core",
                        run.study_commitment,
                    )
                core_attempt = _attempt(
                    "core_publication",
                    config_sha256=run.plan.core_plan.sha256,
                    status=AAMASPublicationAttemptStatus.COMPLETED,
                    output_sha256=core_audit.scientific_result_sha256,
                )
                child_terminal_sha256["core"] = sha256_bytes(
                    canonical_json_bytes(core_attempt.to_mapping())
                )
            except Exception as error:
                core_audit = None
                core_write_seal = None
                core_attempt = _attempt(
                    "core_publication",
                    config_sha256=run.plan.core_plan.sha256,
                    status=AAMASPublicationAttemptStatus.FAILED,
                    error=error,
                )
                child_terminal_sha256["core"] = _terminal_directory(
                    staging / "core", core_attempt
                )
        attempts.append(core_attempt)
        writer_timings.append(
            AAMASPublicationTiming("core_publication", time.perf_counter() - start)
        )

        start = time.perf_counter()
        branch_publication_log = TemporalAuditLog(
            staging / "network-integrated-branch-publication.jsonl"
        )
        if run.integrated_branch_run is None:
            with TemporalFirewall(
                _network_policy(), branch_publication_log
            ).frozen_evaluation():
                branch_attempt = _attempt(
                    "integrated_branch_publication",
                    config_sha256=run.plan.integrated_branch_plan.sha256,
                    status=AAMASPublicationAttemptStatus.BLOCKED_UPSTREAM,
                    reason="integrated_branch_execution_failed",
                )
            child_terminal_sha256["integrated-branch"] = _terminal_directory(
                staging / "integrated-branch", branch_attempt
            )
        else:
            try:
                with TemporalFirewall(
                    _network_policy(), branch_publication_log
                ).frozen_evaluation():
                    branch_audit = write_aamas_integrated_branch(
                        run.plan.integrated_branch_plan,
                        staging / "integrated-branch",
                        allow_formal_scale=allow_formal_scale,
                    )
                if (
                    branch_audit.artifact_sha256
                    != run.integrated_branch_run.artifact.sha256
                ):
                    _integrity_fail("published branch child differs from executed run")
                branch_mapping = read_json_object(
                    staging / "integrated-branch" / "artifact.json"
                )
                branch_attempt = _attempt(
                    "integrated_branch_publication",
                    config_sha256=run.plan.integrated_branch_plan.sha256,
                    status=AAMASPublicationAttemptStatus.COMPLETED,
                    output_sha256=branch_audit.artifact_sha256,
                )
                child_terminal_sha256["integrated-branch"] = sha256_bytes(
                    canonical_json_bytes(branch_attempt.to_mapping())
                )
            except Exception as error:
                branch_audit = None
                branch_mapping = None
                branch_attempt = _attempt(
                    "integrated_branch_publication",
                    config_sha256=run.plan.integrated_branch_plan.sha256,
                    status=AAMASPublicationAttemptStatus.FAILED,
                    error=error,
                )
                child_terminal_sha256["integrated-branch"] = _terminal_directory(
                    staging / "integrated-branch", branch_attempt
                )
        attempts.append(branch_attempt)
        writer_timings.append(
            AAMASPublicationTiming(
                "integrated_branch_publication_and_replay_audit",
                time.perf_counter() - start,
            )
        )

        start = time.perf_counter()
        misspecification_publication_log = TemporalAuditLog(
            staging / "network-misspecification-publication.jsonl"
        )
        if run.misspecification_artifact is None:
            with TemporalFirewall(
                _network_policy(), misspecification_publication_log
            ).frozen_evaluation():
                misspecification_attempt = _attempt(
                    "misspecification_publication",
                    config_sha256=run.plan.misspecification_spec.sha256,
                    status=AAMASPublicationAttemptStatus.BLOCKED_UPSTREAM,
                    reason="misspecification_execution_failed",
                )
            child_terminal_sha256["misspecification"] = _terminal_directory(
                staging / "misspecification", misspecification_attempt
            )
        else:
            try:
                with TemporalFirewall(
                    _network_policy(), misspecification_publication_log
                ).frozen_evaluation():
                    misspecification_receipt = write_aamas_misspecification_artifact(
                        staging / "misspecification",
                        run.misspecification_artifact,
                    )
                    misspecification_root_seal = (
                        misspecification_receipt.root_seal_sha256
                    )
                    misspecification_audit = audit_aamas_misspecification_artifact(
                        staging / "misspecification",
                        expected_plan_sha256=(misspecification_receipt.plan_sha256),
                        expected_root_seal_sha256=(misspecification_root_seal),
                        allow_formal_scale=allow_formal_scale,
                    )
                if (
                    misspecification_audit.artifact_sha256
                    != run.misspecification_artifact.sha256
                ):
                    _integrity_fail("published misspecification child differs from run")
                misspecification_mapping = read_json_object(
                    staging / "misspecification" / "artifact.json"
                )
                misspecification_attempt = _attempt(
                    "misspecification_publication",
                    config_sha256=run.plan.misspecification_spec.sha256,
                    status=AAMASPublicationAttemptStatus.COMPLETED,
                    output_sha256=misspecification_audit.artifact_sha256,
                )
                child_terminal_sha256["misspecification"] = sha256_bytes(
                    canonical_json_bytes(misspecification_attempt.to_mapping())
                )
            except Exception as error:
                misspecification_audit = None
                misspecification_mapping = None
                misspecification_root_seal = None
                misspecification_attempt = _attempt(
                    "misspecification_publication",
                    config_sha256=run.plan.misspecification_spec.sha256,
                    status=AAMASPublicationAttemptStatus.FAILED,
                    error=error,
                )
                child_terminal_sha256["misspecification"] = _terminal_directory(
                    staging / "misspecification", misspecification_attempt
                )
        attempts.append(misspecification_attempt)
        writer_timings.append(
            AAMASPublicationTiming(
                "misspecification_publication_and_replay_audit",
                time.perf_counter() - start,
            )
        )

        start = time.perf_counter()
        core_reporting_log = TemporalAuditLog(staging / "network-core-reporting.jsonl")
        if core_audit is None:
            with TemporalFirewall(
                _network_policy(), core_reporting_log
            ).frozen_evaluation():
                reporting_attempt = _attempt(
                    "core_reporting",
                    config_sha256=run.plan.reporting_spec.sha256,
                    status=AAMASPublicationAttemptStatus.BLOCKED_UPSTREAM,
                    reason="core_publication_failed",
                )
            child_terminal_sha256["core-reporting"] = _terminal_directory(
                staging / "core-reporting", reporting_attempt
            )
        else:
            try:
                with TemporalFirewall(
                    _network_policy(), core_reporting_log
                ).frozen_evaluation():
                    calibration_seal = _core_calibration_seal(staging / "core")
                    reporting_artifact = derive_aamas_submission_artifact(
                        rq1_directory=staging / "core" / "rq1",
                        rq1_plan=run.plan.core_plan.rq1_plan,
                        calibration_directory=staging / "core" / "calibration",
                        calibration_seal_sha256=calibration_seal,
                        reporting_spec=run.plan.reporting_spec,
                    )
                    write_aamas_submission_artifact(
                        staging / "core-reporting", reporting_artifact
                    )
                    reporting_audit = audit_aamas_submission_artifact(
                        directory=staging / "core-reporting",
                        rq1_directory=staging / "core" / "rq1",
                        rq1_plan=run.plan.core_plan.rq1_plan,
                        calibration_directory=staging / "core" / "calibration",
                        calibration_seal_sha256=calibration_seal,
                        expected_reporting_spec=run.plan.reporting_spec,
                    )
                reporting_attempt = _attempt(
                    "core_reporting",
                    config_sha256=run.plan.reporting_spec.sha256,
                    status=AAMASPublicationAttemptStatus.COMPLETED,
                    output_sha256=reporting_audit.artifact_sha256,
                )
                child_terminal_sha256["core-reporting"] = sha256_bytes(
                    canonical_json_bytes(reporting_attempt.to_mapping())
                )
            except Exception as error:
                reporting_audit = None
                reporting_artifact = None
                reporting_attempt = _attempt(
                    "core_reporting",
                    config_sha256=run.plan.reporting_spec.sha256,
                    status=AAMASPublicationAttemptStatus.FAILED,
                    error=error,
                )
                child_terminal_sha256["core-reporting"] = _terminal_directory(
                    staging / "core-reporting", reporting_attempt
                )
        attempts.append(reporting_attempt)
        writer_timings.append(
            AAMASPublicationTiming(
                "core_reporting_derivation_and_audit", time.perf_counter() - start
            )
        )

        start = time.perf_counter()
        branch_reporting_log = TemporalAuditLog(
            staging / "network-integrated-branch-reporting.jsonl"
        )
        if branch_audit is None:
            with TemporalFirewall(
                _network_policy(), branch_reporting_log
            ).frozen_evaluation():
                branch_reporting_attempt = _attempt(
                    "integrated_branch_reporting",
                    config_sha256=(run.plan.integrated_branch_reporting_spec.sha256),
                    status=AAMASPublicationAttemptStatus.BLOCKED_UPSTREAM,
                    reason="integrated_branch_publication_failed",
                )
            child_terminal_sha256["integrated-branch-reporting"] = _terminal_directory(
                staging / "integrated-branch-reporting",
                branch_reporting_attempt,
            )
        else:
            try:
                with TemporalFirewall(
                    _network_policy(), branch_reporting_log
                ).frozen_evaluation():
                    branch_reporting_artifact = derive_aamas_integrated_branch_report(
                        source_dir=staging / "integrated-branch",
                        expected_plan_sha256=(run.plan.integrated_branch_plan.sha256),
                        expected_artifact_sha256=(branch_audit.artifact_sha256),
                        reporting_spec=(run.plan.integrated_branch_reporting_spec),
                        allow_formal_scale=allow_formal_scale,
                    )
                    report_sha256 = write_aamas_integrated_branch_report(
                        staging / "integrated-branch-reporting",
                        branch_reporting_artifact,
                    )
                    branch_reporting_audit = (
                        audit_written_aamas_integrated_branch_report(
                            output_dir=(staging / "integrated-branch-reporting"),
                            source_dir=staging / "integrated-branch",
                            expected_report_sha256=report_sha256,
                            expected_plan_sha256=(
                                run.plan.integrated_branch_plan.sha256
                            ),
                            expected_artifact_sha256=(branch_audit.artifact_sha256),
                            expected_reporting_spec=(
                                run.plan.integrated_branch_reporting_spec
                            ),
                            allow_formal_scale=allow_formal_scale,
                        )
                    )
                branch_reporting_attempt = _attempt(
                    "integrated_branch_reporting",
                    config_sha256=(run.plan.integrated_branch_reporting_spec.sha256),
                    status=AAMASPublicationAttemptStatus.COMPLETED,
                    output_sha256=branch_reporting_audit.report_sha256,
                )
                child_terminal_sha256["integrated-branch-reporting"] = sha256_bytes(
                    canonical_json_bytes(branch_reporting_attempt.to_mapping())
                )
            except Exception as error:
                branch_reporting_audit = None
                branch_reporting_artifact = None
                branch_reporting_attempt = _attempt(
                    "integrated_branch_reporting",
                    config_sha256=(run.plan.integrated_branch_reporting_spec.sha256),
                    status=AAMASPublicationAttemptStatus.FAILED,
                    error=error,
                )
                child_terminal_sha256["integrated-branch-reporting"] = (
                    _terminal_directory(
                        staging / "integrated-branch-reporting",
                        branch_reporting_attempt,
                    )
                )
        attempts.append(branch_reporting_attempt)
        writer_timings.append(
            AAMASPublicationTiming(
                "integrated_branch_reporting_derivation_and_audit",
                time.perf_counter() - start,
            )
        )
        attempt_registry = _attempt_registry_mapping(run.plan, attempts)
        derived = _derive_publication(
            run.plan,
            _PublicationSources(
                core_audit,
                branch_audit,
                misspecification_audit,
                reporting_audit,
                branch_reporting_audit,
                branch_mapping,
                misspecification_mapping,
                reporting_artifact,
                branch_reporting_artifact,
                tuple(attempts),
            ),
        )

        plan_payload = _pretty_json_bytes(run.plan.to_mapping())
        result_payload = _pretty_json_bytes(derived.result)
        output_payload = _pretty_json_bytes(derived.output_registry)
        attempt_payload = _pretty_json_bytes(attempt_registry)
        certificate_payload = _pretty_json_bytes(derived.certificate.to_mapping())
        data_mapping = _data_manifest()
        data_payload = _pretty_json_bytes(data_mapping)
        source_mapping = read_json_bytes_object(
            run.source_manifest_payload,
            source="pre-run source manifest",
        )
        if source_mapping != _source_manifest(project):
            _integrity_fail("source or lock changed during bundle publication")
        source_payload = run.source_manifest_payload
        study_commitment_payload = _pretty_json_bytes(run.study_commitment.to_mapping())
        for name, payload in (
            ("attempt-registry.json", attempt_payload),
            ("plan.json", plan_payload),
            ("result.json", result_payload),
            ("output-registry.json", output_payload),
            ("validity-certificate.json", certificate_payload),
            ("data-manifest.json", data_payload),
            ("source-manifest.json", source_payload),
            ("study-commitment.json", study_commitment_payload),
        ):
            _write_new(staging / name, payload)
        _write_jsonl(staging / "failures.jsonl", derived.failures)
        _write_jsonl(
            staging / "network-core-run.jsonl",
            run.core_network_events,
        )
        _write_jsonl(
            staging / "network-integrated-branch-run.jsonl",
            run.integrated_branch_network_events,
        )
        _write_jsonl(
            staging / "network-misspecification-run.jsonl",
            run.misspecification_network_events,
        )
        all_timings = (*run.timings, *writer_timings)
        if tuple(item.component_id for item in all_timings) != _TIMING_COMPONENTS:
            _integrity_fail("complete timing registry changed")
        _write_jsonl(
            staging / "timing.jsonl",
            tuple(item.to_mapping(run.plan.sha256) for item in all_timings),
        )

        git_mapping = _portable_git_mapping(run.study_commitment)
        environment_mapping = collect_environment_metadata(project).to_mapping()
        metadata = {
            "caller_authorized_formal_scale": allow_formal_scale,
            "data_manifest_raw_sha256": sha256_bytes(data_payload),
            "environment": environment_mapping,
            "finished_at": run.finished_at.isoformat(),
            "git": git_mapping,
            "integrity_valid_at_write": True,
            "network_mode": "python_process_blocked_temporal_firewall",
            "network_os_isolation_attested": False,
            "plan_precommit_sha256": expected_plan,
            "publication_eligible": False,
            "schema_version": 1,
            "scientifically_complete": derived.scientifically_complete,
            "submission_evidence_complete": (derived.submission_evidence_complete),
            "source_attestation": "local_worktree_not_independent_custody",
            "source_manifest_raw_sha256": sha256_bytes(source_payload),
            "source_tree_sha256": source_mapping["tree_sha256"],
            "study_commitment_sha256": expected_study,
            "started_at": run.started_at.isoformat(),
            "synthetic_only": True,
            "uv_lock_sha256": source_mapping["uv_lock_sha256"],
            "validity_level": derived.certificate.level.value,
        }
        _write_new(staging / "metadata.json", _pretty_json_bytes(metadata))

        root_rows = [
            {
                "path": name,
                "sha256": sha256_file(staging / name),
                "size_bytes": (staging / name).stat().st_size,
            }
            for name in sorted(_ROOT_FILES - {"manifest.json", "seal.json"})
        ]
        child_rows = [
            _child_manifest_row(
                root=staging,
                path="core",
                audit_anchor_kind=(
                    "root_seal_raw_sha256"
                    if core_write_seal is not None
                    else "terminal_state_sha256"
                ),
                audit_anchor_sha256=(core_write_seal or child_terminal_sha256["core"]),
                scientific_artifact_sha256=(
                    core_audit.scientific_result_sha256
                    if core_audit is not None
                    else None
                ),
                terminal_state_sha256=child_terminal_sha256["core"],
            ),
            _child_manifest_row(
                root=staging,
                path="integrated-branch",
                audit_anchor_kind=(
                    "artifact_semantic_sha256"
                    if branch_audit is not None
                    else "terminal_state_sha256"
                ),
                audit_anchor_sha256=(
                    branch_audit.artifact_sha256
                    if branch_audit is not None
                    else child_terminal_sha256["integrated-branch"]
                ),
                scientific_artifact_sha256=(
                    branch_audit.artifact_sha256 if branch_audit is not None else None
                ),
                terminal_state_sha256=(child_terminal_sha256["integrated-branch"]),
            ),
            _child_manifest_row(
                root=staging,
                path="misspecification",
                audit_anchor_kind=(
                    "root_seal_raw_sha256"
                    if misspecification_root_seal is not None
                    else "terminal_state_sha256"
                ),
                audit_anchor_sha256=(
                    misspecification_root_seal
                    or child_terminal_sha256["misspecification"]
                ),
                scientific_artifact_sha256=(
                    misspecification_audit.artifact_sha256
                    if misspecification_audit is not None
                    else None
                ),
                terminal_state_sha256=(child_terminal_sha256["misspecification"]),
            ),
            _child_manifest_row(
                root=staging,
                path="core-reporting",
                audit_anchor_kind=(
                    "artifact_seal_file_raw_sha256"
                    if reporting_audit is not None
                    else "terminal_state_sha256"
                ),
                audit_anchor_sha256=(
                    sha256_file(staging / "core-reporting" / "aamas-submission.sha256")
                    if reporting_audit is not None
                    else child_terminal_sha256["core-reporting"]
                ),
                scientific_artifact_sha256=(
                    reporting_audit.artifact_sha256
                    if reporting_audit is not None
                    else None
                ),
                terminal_state_sha256=child_terminal_sha256["core-reporting"],
            ),
            _child_manifest_row(
                root=staging,
                path="integrated-branch-reporting",
                audit_anchor_kind=(
                    "artifact_seal_file_raw_sha256"
                    if branch_reporting_audit is not None
                    else "terminal_state_sha256"
                ),
                audit_anchor_sha256=(
                    sha256_file(
                        staging
                        / "integrated-branch-reporting"
                        / AAMAS_INTEGRATED_BRANCH_REPORT_SEAL_FILENAME
                    )
                    if branch_reporting_audit is not None
                    else child_terminal_sha256["integrated-branch-reporting"]
                ),
                scientific_artifact_sha256=(
                    branch_reporting_audit.report_sha256
                    if branch_reporting_audit is not None
                    else None
                ),
                terminal_state_sha256=(
                    child_terminal_sha256["integrated-branch-reporting"]
                ),
            ),
        ]
        manifest = {
            "operational_logs_bound": True,
            "plan_sha256": run.plan.sha256,
            "root_files": root_rows,
            "schema_version": AAMAS_PUBLICATION_MANIFEST_SCHEMA_VERSION,
            "subbundles": child_rows,
        }
        manifest_payload = _pretty_json_bytes(manifest)
        _write_new(staging / "manifest.json", manifest_payload)
        scientific_result_sha256 = sha256_bytes(canonical_json_bytes(derived.result))
        seal = {
            "caller_retained_digest_required": True,
            "integrated_branch_reporting_spec_sha256": (
                run.plan.integrated_branch_reporting_spec.sha256
            ),
            "manifest_raw_sha256": sha256_bytes(manifest_payload),
            "plan_raw_sha256": sha256_bytes(plan_payload),
            "plan_sha256": run.plan.sha256,
            "reporting_spec_sha256": run.plan.reporting_spec.sha256,
            "result_raw_sha256": sha256_bytes(result_payload),
            "schema_version": AAMAS_PUBLICATION_SEAL_SCHEMA_VERSION,
            "scientific_result_sha256": scientific_result_sha256,
            "study_commitment_sha256": expected_study,
        }
        seal_payload = _pretty_json_bytes(seal)
        _write_new(staging / "seal.json", seal_payload)
        _verify_root_membership(staging)
        root_seal_sha256 = sha256_bytes(seal_payload)
        prepublication_audit = audit_aamas_publication(
            staging,
            expected_plan=run.plan,
            expected_plan_sha256=run.plan.sha256,
            expected_study_commitment_sha256=expected_study,
            expected_root_seal_sha256=root_seal_sha256,
            project_root=project,
            allow_formal_scale=allow_formal_scale,
        )
        if (
            prepublication_audit.scientific_result_sha256 != scientific_result_sha256
            or prepublication_audit.study_commitment_sha256 != expected_study
        ):
            _integrity_fail("complete prepublication audit changed root identity")
        if directory.exists() or directory.is_symlink():
            _integrity_fail("publication output appeared before atomic publication")
        os.replace(staging, directory)
        return AAMASPublicationWriteResult(
            directory=directory,
            plan_sha256=run.plan.sha256,
            study_commitment_sha256=expected_study,
            root_seal_sha256=root_seal_sha256,
            scientific_result_sha256=scientific_result_sha256,
            submission_evidence_complete=(derived.submission_evidence_complete),
        )
    except BaseException:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise


def _verify_root_membership(directory: Path) -> None:
    if not directory.is_dir() or directory.is_symlink():
        _integrity_fail("publication root must be a real directory")
    entries = {item.name: item for item in directory.iterdir()}
    if frozenset(entries) != _ROOT_FILES | _ROOT_DIRECTORIES:
        _integrity_fail("publication root fixed membership changed")
    for name in _ROOT_FILES:
        details = entries[name].lstat()
        if (
            not stat.S_ISREG(details.st_mode)
            or stat.S_ISLNK(details.st_mode)
            or details.st_nlink != 1
        ):
            _integrity_fail(f"publication root file is not singly linked: {name}")
    for name in _ROOT_DIRECTORIES:
        details = entries[name].lstat()
        if not stat.S_ISDIR(details.st_mode) or stat.S_ISLNK(details.st_mode):
            _integrity_fail(f"publication child is not a real directory: {name}")
    _tree_rows(directory)


def _verify_manifest(
    directory: Path, manifest: Mapping[str, object]
) -> dict[str, Mapping[str, object]]:
    _strict_fields(
        manifest,
        frozenset(
            {
                "operational_logs_bound",
                "plan_sha256",
                "root_files",
                "schema_version",
                "subbundles",
            }
        ),
        "publication manifest",
    )
    if (
        manifest["schema_version"] != AAMAS_PUBLICATION_MANIFEST_SCHEMA_VERSION
        or manifest["operational_logs_bound"] is not True
    ):
        _integrity_fail("publication manifest authority changed")
    expected_root = _ROOT_FILES - {"manifest.json", "seal.json"}
    observed_root: set[str] = set()
    raw_root_rows = _sequence(manifest["root_files"], "manifest root_files")
    if tuple(
        _mapping(item, "manifest root row").get("path") for item in raw_root_rows
    ) != tuple(sorted(expected_root)):
        _integrity_fail("manifest root rows are not in canonical fixed order")
    for raw in raw_root_rows:
        row = _mapping(raw, "manifest root row")
        _strict_fields(row, frozenset({"path", "sha256", "size_bytes"}), "root row")
        path_value = row["path"]
        if (
            not isinstance(path_value, str)
            or path_value not in expected_root
            or path_value in observed_root
        ):
            _integrity_fail("manifest root path registry changed")
        observed_root.add(path_value)
        path = directory / path_value
        if (
            row["sha256"] != sha256_file(path)
            or row["size_bytes"] != path.stat().st_size
        ):
            _integrity_fail(f"sealed root member changed: {path_value}")
    if observed_root != expected_root:
        _integrity_fail("manifest omitted a root file")
    expected_kinds = {
        "core": {"root_seal_raw_sha256", "terminal_state_sha256"},
        "integrated-branch": {
            "artifact_semantic_sha256",
            "terminal_state_sha256",
        },
        "misspecification": {
            "root_seal_raw_sha256",
            "terminal_state_sha256",
        },
        "core-reporting": {
            "artifact_seal_file_raw_sha256",
            "terminal_state_sha256",
        },
        "integrated-branch-reporting": {
            "artifact_seal_file_raw_sha256",
            "terminal_state_sha256",
        },
    }
    child_rows: dict[str, Mapping[str, object]] = {}
    raw_child_rows = _sequence(manifest["subbundles"], "manifest subbundles")
    expected_child_order = (
        "core",
        "integrated-branch",
        "misspecification",
        "core-reporting",
        "integrated-branch-reporting",
    )
    if (
        tuple(
            _mapping(item, "manifest child row").get("path") for item in raw_child_rows
        )
        != expected_child_order
    ):
        _integrity_fail("manifest child rows are not in canonical fixed order")
    for raw in raw_child_rows:
        row = _mapping(raw, "manifest child row")
        _strict_fields(
            row,
            frozenset(
                {
                    "audit_anchor_kind",
                    "audit_anchor_sha256",
                    "file_count",
                    "path",
                    "scientific_artifact_sha256",
                    "terminal_state_sha256",
                    "tree_sha256",
                }
            ),
            "child row",
        )
        name = row["path"]
        if (
            not isinstance(name, str)
            or name not in expected_kinds
            or name in child_rows
        ):
            _integrity_fail("manifest child path registry changed")
        if row["audit_anchor_kind"] not in expected_kinds[name]:
            _integrity_fail("manifest child audit anchor kind changed")
        _sha256(row["audit_anchor_sha256"], "child audit anchor")
        _sha256(row["terminal_state_sha256"], "child terminal state")
        if row["scientific_artifact_sha256"] is not None:
            _sha256(
                row["scientific_artifact_sha256"],
                "child scientific artifact",
            )
        if (
            row["audit_anchor_kind"] == "terminal_state_sha256"
            and row["scientific_artifact_sha256"] is not None
        ):
            _integrity_fail("failed child cannot retain a scientific artifact")
        rows = _tree_rows(directory / name)
        if row["file_count"] != len(rows) or row["tree_sha256"] != sha256_bytes(
            canonical_json_bytes(rows)
        ):
            _integrity_fail(f"sealed child tree changed: {name}")
        child_rows[name] = row
    if frozenset(child_rows) != _ROOT_DIRECTORIES:
        _integrity_fail("manifest omitted or added a child bundle")
    return child_rows


def _audit_timing(rows: Sequence[Mapping[str, object]], plan_sha256: str) -> None:
    if len(rows) != len(_TIMING_COMPONENTS):
        _integrity_fail("timing registry changed cardinality")
    expected_fields = frozenset(
        {
            "component_id",
            "event",
            "event_key",
            "measurement_scope",
            "schema_version",
            "wall_seconds",
        }
    )
    for component, row in zip(_TIMING_COMPONENTS, rows, strict=True):
        _strict_fields(row, expected_fields, "timing row")
        expected = AAMASPublicationTiming(component, 0.0).to_mapping(plan_sha256)
        if any(
            row[field] != expected[field]
            for field in expected_fields - {"wall_seconds"}
        ):
            _integrity_fail("timing identity or formula changed")
        if _number(row["wall_seconds"], "wall_seconds") < 0.0:
            _integrity_fail("timing duration is negative")


def _audit_network_file(path: Path, *, context: str) -> bool:
    events = TemporalAuditLog(path).events()
    return _validate_network_events(events, context=context)


@dataclass(frozen=True, slots=True)
class AAMASPublicationAuditReport:
    integrity_valid: Literal[True]
    submission_evidence_complete: bool
    scientifically_complete: Literal[False]
    publication_eligible: Literal[False]
    validity_level: Literal[ValidityLevel.ABSTAIN]
    plan_sha256: str
    study_commitment_sha256: str
    root_seal_sha256: str
    scientific_result_sha256: str
    source_tree_sha256: str
    output_statuses: tuple[tuple[str, str], ...]
    incomplete_outputs: tuple[str, ...]
    failed_outputs: tuple[str, ...]
    blocked_outputs: tuple[str, ...]
    abstained_outputs: tuple[str, ...]


def _audit_aamas_publication_snapshot(
    directory: Path,
    *,
    expected_plan: AAMASPublicationPlan,
    expected_plan_sha256: str,
    expected_study_commitment_sha256: str,
    expected_root_seal_sha256: str,
    project_root: Path | None = None,
    allow_formal_scale: bool = False,
) -> AAMASPublicationAuditReport:
    """Audit one verifier-owned stable snapshot without copying it again."""

    if not isinstance(directory, Path) or not isinstance(
        expected_plan, AAMASPublicationPlan
    ):
        _fail("audit requires typed directory and publication plan")
    plan_sha256 = _sha256(expected_plan_sha256, "expected_plan_sha256")
    study_commitment_sha256 = _sha256(
        expected_study_commitment_sha256,
        "expected_study_commitment_sha256",
    )
    root_seal_sha256 = _sha256(expected_root_seal_sha256, "expected_root_seal_sha256")
    if plan_sha256 != expected_plan.sha256:
        _integrity_fail("external plan digest differs from expected plan")
    _require_execution_authority(expected_plan, allow_formal_scale)
    project = (
        Path(__file__).resolve().parents[3]
        if project_root is None
        else project_root.resolve()
    )
    _verify_root_membership(directory)
    if sha256_file(directory / "seal.json") != root_seal_sha256:
        _integrity_fail("root seal differs from caller-held digest")
    seal = read_json_object(directory / "seal.json")
    _strict_fields(
        seal,
        frozenset(
            {
                "caller_retained_digest_required",
                "integrated_branch_reporting_spec_sha256",
                "manifest_raw_sha256",
                "plan_raw_sha256",
                "plan_sha256",
                "reporting_spec_sha256",
                "result_raw_sha256",
                "schema_version",
                "scientific_result_sha256",
                "study_commitment_sha256",
            }
        ),
        "publication seal",
    )
    if (
        seal["schema_version"] != AAMAS_PUBLICATION_SEAL_SCHEMA_VERSION
        or seal["caller_retained_digest_required"] is not True
        or seal["plan_sha256"] != plan_sha256
        or seal["reporting_spec_sha256"] != expected_plan.reporting_spec.sha256
        or seal["integrated_branch_reporting_spec_sha256"]
        != expected_plan.integrated_branch_reporting_spec.sha256
        or seal["study_commitment_sha256"] != study_commitment_sha256
        or seal["manifest_raw_sha256"] != sha256_file(directory / "manifest.json")
        or seal["plan_raw_sha256"] != sha256_file(directory / "plan.json")
        or seal["result_raw_sha256"] != sha256_file(directory / "result.json")
    ):
        _integrity_fail("publication seal commitments changed")
    manifest = read_json_object(directory / "manifest.json")
    child_rows = _verify_manifest(directory, manifest)
    if manifest["plan_sha256"] != plan_sha256:
        _integrity_fail("publication manifest plan changed")
    stored_plan = AAMASPublicationPlan.from_mapping(
        read_json_object(directory / "plan.json")
    )
    if stored_plan != expected_plan:
        _integrity_fail("stored publication plan differs from external plan")
    study_commitment = AAMASPublicationStudyCommitment.from_mapping(
        read_json_object(directory / "study-commitment.json")
    )
    if (
        study_commitment.sha256 != study_commitment_sha256
        or study_commitment.plan_sha256 != plan_sha256
    ):
        _integrity_fail("stored study commitment differs from caller-held digest")

    network_denials = {
        name: _audit_network_file(directory / name, context=name)
        for name in _NETWORK_FILES
    }
    _audit_timing(read_jsonl_objects(directory / "timing.jsonl"), plan_sha256)

    attempts = _load_attempt_registry(
        directory / "attempt-registry.json", expected_plan
    )
    attempt_by_id = {item.attempt_id: item for item in attempts}
    network_attempts = {
        "network-core-run.jsonl": "core_execution",
        "network-integrated-branch-run.jsonl": "integrated_branch_execution",
        "network-misspecification-run.jsonl": "misspecification_execution",
        "network-core-publication.jsonl": "core_publication",
        "network-integrated-branch-publication.jsonl": (
            "integrated_branch_publication"
        ),
        "network-misspecification-publication.jsonl": ("misspecification_publication"),
        "network-core-reporting.jsonl": "core_reporting",
        "network-integrated-branch-reporting.jsonl": ("integrated_branch_reporting"),
    }
    if tuple(network_attempts) != _NETWORK_FILES:
        _integrity_fail("network-to-attempt registry changed")
    for name, attempt_id in network_attempts.items():
        if (
            network_denials[name]
            and attempt_by_id[attempt_id].status
            is not AAMASPublicationAttemptStatus.FAILED
        ):
            _integrity_fail("denied network access lacks a failed terminal attempt")
    core_publication = attempt_by_id["core_publication"]
    branch_publication = attempt_by_id["integrated_branch_publication"]
    misspecification_publication = attempt_by_id["misspecification_publication"]
    core_reporting_attempt = attempt_by_id["core_reporting"]
    branch_reporting_attempt = attempt_by_id["integrated_branch_reporting"]

    core_audit: AAMASSubmissionAuditReport | None = None
    branch_audit: AAMASIntegratedBranchAudit | None = None
    misspecification_audit: MisspecificationAuditReport | None = None
    reporting_audit: AAMASCoreReportingAudit | None = None
    branch_reporting_audit: AAMASIntegratedBranchReportAudit | None = None
    reporting_artifact: AAMASCoreReportingArtifact | None = None
    branch_reporting_artifact: AAMASIntegratedBranchReportArtifact | None = None
    branch_mapping: Mapping[str, object] | None = None
    misspecification_mapping: Mapping[str, object] | None = None

    core_row = child_rows["core"]
    if core_publication.status is AAMASPublicationAttemptStatus.COMPLETED:
        audited_core, core_audit_events = _run_frozen(
            lambda: audit_aamas_submission(
                directory / "core",
                expected_plan=expected_plan.core_plan,
                expected_plan_sha256=expected_plan.core_plan.sha256,
                expected_seal_sha256=cast(str, core_row["audit_anchor_sha256"]),
                project_root=project,
            ),
            prefix="policytwin-aamas-publication-audit-core-",
        )
        core_audit = audited_core
        _validate_network_events(core_audit_events, context="core audit replay")
        _audit_core_git_bindings(directory / "core", study_commitment)
        if (
            core_row["scientific_artifact_sha256"]
            != core_audit.scientific_result_sha256
            or core_publication.output_sha256 != core_audit.scientific_result_sha256
        ):
            _integrity_fail("core scientific result differs from terminal registry")
        core_terminal = sha256_bytes(
            canonical_json_bytes(core_publication.to_mapping())
        )
    else:
        core_terminal = _audit_terminal_directory(directory / "core", core_publication)
    if core_row["terminal_state_sha256"] != core_terminal:
        _integrity_fail("core terminal state differs from root manifest")

    branch_row = child_rows["integrated-branch"]
    if branch_publication.status is AAMASPublicationAttemptStatus.COMPLETED:
        audited_branch, branch_audit_events = _run_frozen(
            lambda: audit_written_aamas_integrated_branch(
                directory / "integrated-branch",
                expected_plan_sha256=expected_plan.integrated_branch_plan.sha256,
                expected_artifact_sha256=cast(str, branch_row["audit_anchor_sha256"]),
                allow_formal_scale=allow_formal_scale,
            ),
            prefix="policytwin-aamas-publication-audit-branch-",
        )
        branch_audit = audited_branch
        _validate_network_events(branch_audit_events, context="branch audit replay")
        if (
            branch_row["scientific_artifact_sha256"] != audited_branch.artifact_sha256
            or branch_publication.output_sha256 != audited_branch.artifact_sha256
        ):
            _integrity_fail("branch artifact differs from terminal registry")
        branch_mapping = read_json_object(
            directory / "integrated-branch" / "artifact.json"
        )
        branch_terminal = sha256_bytes(
            canonical_json_bytes(branch_publication.to_mapping())
        )
    else:
        branch_terminal = _audit_terminal_directory(
            directory / "integrated-branch", branch_publication
        )
    if branch_row["terminal_state_sha256"] != branch_terminal:
        _integrity_fail("branch terminal state differs from root manifest")

    misspecification_row = child_rows["misspecification"]
    if misspecification_publication.status is AAMASPublicationAttemptStatus.COMPLETED:
        audited_misspecification, misspecification_audit_events = _run_frozen(
            lambda: audit_aamas_misspecification_artifact(
                directory / "misspecification",
                expected_plan_sha256=sha256_bytes(
                    canonical_json_bytes(
                        read_json_object(directory / "misspecification" / "plan.json")
                    )
                ),
                expected_root_seal_sha256=cast(
                    str, misspecification_row["audit_anchor_sha256"]
                ),
                allow_formal_scale=allow_formal_scale,
            ),
            prefix="policytwin-aamas-publication-audit-misspec-",
        )
        misspecification_audit = audited_misspecification
        _validate_network_events(
            misspecification_audit_events,
            context="misspecification audit replay",
        )
        if (
            misspecification_row["scientific_artifact_sha256"]
            != audited_misspecification.artifact_sha256
            or misspecification_publication.output_sha256
            != audited_misspecification.artifact_sha256
        ):
            _integrity_fail("misspecification artifact differs from terminal registry")
        misspecification_mapping = read_json_object(
            directory / "misspecification" / "artifact.json"
        )
        misspecification_terminal = sha256_bytes(
            canonical_json_bytes(misspecification_publication.to_mapping())
        )
    else:
        misspecification_terminal = _audit_terminal_directory(
            directory / "misspecification", misspecification_publication
        )
    if misspecification_row["terminal_state_sha256"] != misspecification_terminal:
        _integrity_fail("misspecification terminal state differs from root manifest")

    reporting_row = child_rows["core-reporting"]
    if core_reporting_attempt.status is AAMASPublicationAttemptStatus.COMPLETED:
        if core_audit is None:
            _integrity_fail("core reporting completed without an audited core")
        calibration_seal = _core_calibration_seal(directory / "core")
        if reporting_row["audit_anchor_sha256"] != sha256_file(
            directory / "core-reporting" / "aamas-submission.sha256"
        ):
            _integrity_fail("core reporting seal differs from root manifest")
        audited_core_report, core_report_audit_events = _run_frozen(
            lambda: audit_aamas_submission_artifact(
                directory=directory / "core-reporting",
                rq1_directory=directory / "core" / "rq1",
                rq1_plan=expected_plan.core_plan.rq1_plan,
                calibration_directory=directory / "core" / "calibration",
                calibration_seal_sha256=calibration_seal,
                expected_reporting_spec=expected_plan.reporting_spec,
            ),
            prefix="policytwin-aamas-publication-audit-core-report-",
        )
        reporting_audit = audited_core_report
        _validate_network_events(
            core_report_audit_events,
            context="core reporting audit replay",
        )
        if (
            reporting_row["scientific_artifact_sha256"]
            != reporting_audit.artifact_sha256
            or core_reporting_attempt.output_sha256 != reporting_audit.artifact_sha256
        ):
            _integrity_fail("core report differs from terminal registry")
        reporting_artifact = AAMASCoreReportingArtifact.from_mapping(
            read_json_object(directory / "core-reporting" / "aamas-submission.json")
        )
        reporting_terminal = sha256_bytes(
            canonical_json_bytes(core_reporting_attempt.to_mapping())
        )
    else:
        reporting_terminal = _audit_terminal_directory(
            directory / "core-reporting", core_reporting_attempt
        )
    if reporting_row["terminal_state_sha256"] != reporting_terminal:
        _integrity_fail("core reporting terminal state differs from root manifest")

    branch_reporting_row = child_rows["integrated-branch-reporting"]
    if branch_reporting_attempt.status is AAMASPublicationAttemptStatus.COMPLETED:
        if branch_audit is None:
            _integrity_fail("branch reporting completed without audited source")
        if branch_reporting_row["audit_anchor_sha256"] != sha256_file(
            directory
            / "integrated-branch-reporting"
            / AAMAS_INTEGRATED_BRANCH_REPORT_SEAL_FILENAME
        ):
            _integrity_fail("branch reporting seal differs from root manifest")
        audited_branch_report, report_audit_events = _run_frozen(
            lambda: audit_written_aamas_integrated_branch_report(
                output_dir=directory / "integrated-branch-reporting",
                source_dir=directory / "integrated-branch",
                expected_report_sha256=cast(
                    str, branch_reporting_attempt.output_sha256
                ),
                expected_plan_sha256=expected_plan.integrated_branch_plan.sha256,
                expected_artifact_sha256=branch_audit.artifact_sha256,
                expected_reporting_spec=(
                    expected_plan.integrated_branch_reporting_spec
                ),
                allow_formal_scale=allow_formal_scale,
            ),
            prefix="policytwin-aamas-publication-audit-branch-report-",
        )
        branch_reporting_audit = audited_branch_report
        _validate_network_events(report_audit_events, context="branch reporting replay")
        if (
            branch_reporting_row["scientific_artifact_sha256"]
            != audited_branch_report.report_sha256
        ):
            _integrity_fail("branch report differs from root manifest")
        branch_reporting_artifact = AAMASIntegratedBranchReportArtifact.from_mapping(
            read_json_object(
                directory
                / "integrated-branch-reporting"
                / AAMAS_INTEGRATED_BRANCH_REPORT_FILENAME
            )
        )
        branch_reporting_terminal = sha256_bytes(
            canonical_json_bytes(branch_reporting_attempt.to_mapping())
        )
    else:
        branch_reporting_terminal = _audit_terminal_directory(
            directory / "integrated-branch-reporting",
            branch_reporting_attempt,
        )
    if branch_reporting_row["terminal_state_sha256"] != branch_reporting_terminal:
        _integrity_fail("branch reporting terminal state differs from manifest")

    derived = _derive_publication(
        expected_plan,
        _PublicationSources(
            core_audit,
            branch_audit,
            misspecification_audit,
            reporting_audit,
            branch_reporting_audit,
            branch_mapping,
            misspecification_mapping,
            reporting_artifact,
            branch_reporting_artifact,
            attempts,
        ),
    )
    if read_json_object(directory / "result.json") != derived.result:
        _integrity_fail("root scientific result does not rederive from child evidence")
    if read_json_object(directory / "output-registry.json") != derived.output_registry:
        _integrity_fail("root output registry does not rederive")
    if read_jsonl_objects(directory / "failures.jsonl") != derived.failures:
        _integrity_fail("root failure registry changed or omitted a failure")
    if (
        read_json_object(directory / "validity-certificate.json")
        != derived.certificate.to_mapping()
        or derived.certificate.level is not ValidityLevel.ABSTAIN
    ):
        _integrity_fail("outer validity certificate is not authoritative ABSTAIN")
    scientific_result_sha256 = sha256_bytes(canonical_json_bytes(derived.result))
    if seal["scientific_result_sha256"] != scientific_result_sha256:
        _integrity_fail("root scientific result semantic hash changed")

    if read_json_object(directory / "data-manifest.json") != _data_manifest():
        _integrity_fail("synthetic empty data manifest changed")
    source_mapping = read_json_object(directory / "source-manifest.json")
    if source_mapping != _source_manifest(project):
        _integrity_fail("live local source differs from bundled source manifest")
    if (
        source_mapping.get("tree_sha256") != study_commitment.source_tree_sha256
        or source_mapping.get("uv_lock_sha256") != study_commitment.uv_lock_sha256
    ):
        _integrity_fail("source manifest differs from pre-run study commitment")
    metadata = read_json_object(directory / "metadata.json")
    _strict_fields(
        metadata,
        frozenset(
            {
                "caller_authorized_formal_scale",
                "data_manifest_raw_sha256",
                "environment",
                "finished_at",
                "git",
                "integrity_valid_at_write",
                "network_mode",
                "network_os_isolation_attested",
                "plan_precommit_sha256",
                "publication_eligible",
                "schema_version",
                "scientifically_complete",
                "submission_evidence_complete",
                "source_attestation",
                "source_manifest_raw_sha256",
                "source_tree_sha256",
                "study_commitment_sha256",
                "started_at",
                "synthetic_only",
                "uv_lock_sha256",
                "validity_level",
            }
        ),
        "publication metadata",
    )
    try:
        started = datetime.fromisoformat(cast(str, metadata["started_at"]))
        finished = datetime.fromisoformat(cast(str, metadata["finished_at"]))
    except (TypeError, ValueError) as error:
        raise AAMASPublicationIntegrityError(
            "publication metadata timestamps are invalid"
        ) from error
    if started.tzinfo is None or finished.tzinfo is None or finished < started:
        _integrity_fail("publication metadata timestamps are not ordered and aware")
    GitMetadata.from_mapping(_mapping(metadata["git"], "metadata git"))
    environment = EnvironmentMetadata.from_mapping(
        _mapping(metadata["environment"], "metadata environment")
    )
    if metadata["git"] != _portable_git_mapping(study_commitment):
        _integrity_fail("generation Git identity differs from study commitment")
    if environment.lock_sha256 != study_commitment.uv_lock_sha256:
        _integrity_fail("generation environment lock differs from study commitment")
    expected_metadata_values = {
        "caller_authorized_formal_scale": allow_formal_scale,
        "data_manifest_raw_sha256": sha256_file(directory / "data-manifest.json"),
        "integrity_valid_at_write": True,
        "network_mode": "python_process_blocked_temporal_firewall",
        "network_os_isolation_attested": False,
        "plan_precommit_sha256": plan_sha256,
        "publication_eligible": False,
        "schema_version": 1,
        "scientifically_complete": derived.scientifically_complete,
        "submission_evidence_complete": derived.submission_evidence_complete,
        "source_attestation": "local_worktree_not_independent_custody",
        "source_manifest_raw_sha256": sha256_file(directory / "source-manifest.json"),
        "source_tree_sha256": source_mapping["tree_sha256"],
        "study_commitment_sha256": study_commitment_sha256,
        "synthetic_only": True,
        "uv_lock_sha256": sha256_file(project / "uv.lock"),
        "validity_level": ValidityLevel.ABSTAIN.value,
    }
    if any(metadata[key] != value for key, value in expected_metadata_values.items()):
        _integrity_fail("publication metadata commitments changed")
    output_statuses = tuple(
        (cast(str, row["output_id"]), cast(str, row["status"]))
        for row in cast(list[dict[str, object]], derived.output_registry["entries"])
    )
    incomplete = tuple(
        output_id
        for output_id, status in output_statuses
        if status not in {"complete", "complete_detected"}
    )
    failed = tuple(
        output_id
        for output_id, status in output_statuses
        if status == "complete_not_detected" or status.startswith("failed")
    )
    blocked = tuple(
        output_id
        for output_id, status in output_statuses
        if status.startswith("blocked")
    )
    abstained = tuple(
        output_id
        for output_id, status in output_statuses
        if status.startswith("abstain")
    )
    return AAMASPublicationAuditReport(
        integrity_valid=True,
        submission_evidence_complete=derived.submission_evidence_complete,
        scientifically_complete=False,
        publication_eligible=False,
        validity_level=ValidityLevel.ABSTAIN,
        plan_sha256=plan_sha256,
        study_commitment_sha256=study_commitment_sha256,
        root_seal_sha256=root_seal_sha256,
        scientific_result_sha256=scientific_result_sha256,
        source_tree_sha256=cast(str, source_mapping["tree_sha256"]),
        output_statuses=output_statuses,
        incomplete_outputs=incomplete,
        failed_outputs=failed,
        blocked_outputs=blocked,
        abstained_outputs=abstained,
    )


def audit_aamas_publication(
    directory: Path,
    *,
    expected_plan: AAMASPublicationPlan,
    expected_plan_sha256: str,
    expected_study_commitment_sha256: str,
    expected_root_seal_sha256: str,
    project_root: Path | None = None,
    allow_formal_scale: bool = False,
) -> AAMASPublicationAuditReport:
    """Audit a private snapshot and prove the live publication stayed unchanged.

    A complete initial replay establishes the caller-visible identity.  The tree
    is then copied with symbolic links preserved, audited and re-derived only from
    that private copy, and audited once more before return.  Strict tree rows on
    both sides prevent a positive receipt from being paired with bytes replaced
    while any child or root result was read.
    """

    if not isinstance(directory, Path) or not isinstance(
        expected_plan, AAMASPublicationPlan
    ):
        _fail("audit requires typed directory and publication plan")
    initial_rows = _strict_audit_tree_rows(directory)
    original_audit = _audit_aamas_publication_snapshot(
        directory,
        expected_plan=expected_plan,
        expected_plan_sha256=expected_plan_sha256,
        expected_study_commitment_sha256=expected_study_commitment_sha256,
        expected_root_seal_sha256=expected_root_seal_sha256,
        project_root=project_root,
        allow_formal_scale=allow_formal_scale,
    )
    stable_original_rows = _strict_audit_tree_rows(directory)
    if stable_original_rows != initial_rows:
        _integrity_fail("publication tree changed during its initial audit")

    with tempfile.TemporaryDirectory(
        prefix="policytwin-aamas-publication-audit-snapshot-"
    ) as temporary:
        snapshot = Path(temporary) / "publication"
        try:
            shutil.copytree(directory, snapshot, symlinks=True)
        except OSError as error:
            raise AAMASPublicationIntegrityError(
                "cannot create private publication audit snapshot"
            ) from error
        snapshot_rows = _strict_audit_tree_rows(snapshot)
        if snapshot_rows != stable_original_rows:
            _integrity_fail(
                "publication tree changed while its private snapshot was created"
            )
        snapshot_audit = _audit_aamas_publication_snapshot(
            snapshot,
            expected_plan=expected_plan,
            expected_plan_sha256=expected_plan_sha256,
            expected_study_commitment_sha256=expected_study_commitment_sha256,
            expected_root_seal_sha256=expected_root_seal_sha256,
            project_root=project_root,
            allow_formal_scale=allow_formal_scale,
        )
        if snapshot_audit != original_audit:
            _integrity_fail("publication snapshot audit differs from initial audit")
        if _strict_audit_tree_rows(snapshot) != snapshot_rows:
            _integrity_fail("private publication snapshot changed after derivation")
        final_snapshot_audit = _audit_aamas_publication_snapshot(
            snapshot,
            expected_plan=expected_plan,
            expected_plan_sha256=expected_plan_sha256,
            expected_study_commitment_sha256=expected_study_commitment_sha256,
            expected_root_seal_sha256=expected_root_seal_sha256,
            project_root=project_root,
            allow_formal_scale=allow_formal_scale,
        )
        if (
            final_snapshot_audit != snapshot_audit
            or _strict_audit_tree_rows(snapshot) != snapshot_rows
        ):
            _integrity_fail("private publication snapshot changed during final audit")
        if _strict_audit_tree_rows(directory) != stable_original_rows:
            _integrity_fail("live publication tree changed during private audit")
        return final_snapshot_audit


__all__ = [
    "AAMAS_PUBLICATION_CLAIM_SCOPE",
    "AAMAS_PUBLICATION_MANIFEST_SCHEMA_VERSION",
    "AAMAS_PUBLICATION_PLAN_SCHEMA_VERSION",
    "AAMAS_PUBLICATION_RESULT_SCHEMA_VERSION",
    "AAMAS_PUBLICATION_SEAL_SCHEMA_VERSION",
    "AAMAS_PUBLICATION_STUDY_COMMITMENT_SCHEMA_VERSION",
    "AAMASPublicationAuditReport",
    "AAMASPublicationAttemptStatus",
    "AAMASPublicationError",
    "AAMASPublicationIntegrityError",
    "AAMASPublicationPlan",
    "AAMASPublicationRun",
    "AAMASPublicationStudyCommitment",
    "AAMASPublicationTiming",
    "AAMASPublicationWriteResult",
    "audit_aamas_publication",
    "build_aamas_publication_study_commitment",
    "run_aamas_publication",
    "write_aamas_publication",
]
