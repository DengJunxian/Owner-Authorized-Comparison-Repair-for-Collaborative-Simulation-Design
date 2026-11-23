"""Externally committed root bundle for the synthetic AAMAS benchmark.

The two existing AAMAS executors deliberately publish independent artifacts.
This module composes them without weakening either audit boundary.  Scientific
results exclude wall-clock and machine metadata; a caller-retained root seal
binds those operational records, both nested trees, and the complete local
source manifest.

The bundle is synthetic-only.  Its validity certificate uses only diagnostics
mechanically derived from replay-audited child artifacts.  Missing verifier
evidence remains ``None`` and the current public certificate builder therefore
always returns the authoritative level ``ABSTAIN``.
"""

from __future__ import annotations

import math
import os
import shutil
import stat
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from statistics import fmean, variance
from typing import Literal, NoReturn, cast

from policytwin.calibration.schema import SummaryRole
from policytwin.data.temporal_firewall import (
    CutoffPolicy,
    TemporalAuditLog,
    TemporalFirewall,
)
from policytwin.evaluation.policy import (
    ValidityCertificate,
    ValidityEvidence,
    ValidityLevel,
    ValidityThresholds,
    build_validity_certificate,
)
from policytwin.experiments.aamas_benchmark import (
    AAMASAttemptKind,
    AAMASAttemptStatus,
    AAMASBenchmarkArtifact,
    AAMASBenchmarkPlan,
    AAMASBenchmarkRun,
    AAMASResearchQuestion,
    audit_aamas_benchmark,
    run_aamas_benchmark,
    write_aamas_benchmark,
)
from policytwin.experiments.aamas_calibration import (
    AAMASCalibrationRunCollection,
    AAMASCalibrationSpec,
    TruthRole,
    audit_aamas_calibration_collection,
    run_aamas_calibration_collection,
    write_aamas_calibration_collection,
)
from policytwin.experiments.aamas_calibration import (
    AttemptStatus as CalibrationAttemptStatus,
)
from policytwin.experiments.matrix import ResourceBudget
from policytwin.experiments.mechanism_smoke import (
    MechanismSmokeArtifact,
    RNGCouplingMode,
)
from policytwin.provenance import (
    ArtifactDigest,
    GitMetadata,
    JsonlEventWriter,
    RunMetadata,
    SourceArtifact,
    build_event_key_metadata,
    canonical_json_bytes,
    collect_environment_metadata,
    collect_git_metadata,
    read_json_object,
    read_jsonl_objects,
    sha256_bytes,
    sha256_file,
)

AAMAS_SUBMISSION_PLAN_SCHEMA_VERSION = "policytwin.aamas-submission-plan.v2"
AAMAS_SUBMISSION_RESULT_SCHEMA_VERSION = "policytwin.aamas-submission-result.v2"
AAMAS_SUBMISSION_MANIFEST_SCHEMA_VERSION = "policytwin.aamas-submission-manifest.v1"
AAMAS_SUBMISSION_SEAL_SCHEMA_VERSION = "policytwin.aamas-submission-seal.v1"
AAMAS_SUBMISSION_CLAIM_SCOPE = "synthetic_aamas_benchmark_only"

_ROOT_FILES = frozenset(
    {
        "attempt-registry.json",
        "data-manifest.json",
        "failures.jsonl",
        "manifest.json",
        "metadata.json",
        "network-audit.jsonl",
        "plan.json",
        "resources.jsonl",
        "result.json",
        "seal.json",
        "source-manifest.json",
        "timing.jsonl",
        "validity-certificate.json",
    }
)
_ROOT_DIRECTORIES = frozenset({"calibration", "rq1"})
_SHA256_CHARS = frozenset("0123456789abcdef")


class AAMASSubmissionError(RuntimeError):
    """Base error for an invalid submission plan or execution."""


class AAMASSubmissionIntegrityError(AAMASSubmissionError):
    """Raised when a root bundle differs from an external commitment."""


def _fail(message: str) -> NoReturn:
    raise AAMASSubmissionError(message)


def _integrity_fail(message: str) -> NoReturn:
    raise AAMASSubmissionIntegrityError(message)


def _sha256(value: object, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or not set(value) <= _SHA256_CHARS
    ):
        _integrity_fail(f"{field_name} must be a lowercase SHA-256")
    return value


def _object(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        _integrity_fail(f"{field_name} must be an object with string keys")
    return cast(Mapping[str, object], value)


def _array(value: object, field_name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        _integrity_fail(f"{field_name} must be an array")
    return cast(Sequence[object], value)


def _strict_fields(
    value: Mapping[str, object], expected: frozenset[str], context: str
) -> None:
    if frozenset(value) != expected:
        _integrity_fail(f"{context} has missing or unknown fields")


def _number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        _integrity_fail(f"{field_name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        _integrity_fail(f"{field_name} must be finite")
    return result


def _integer(value: object, field_name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _integrity_fail(f"{field_name} must be an integer >= {minimum}")
    return value


def _pretty_json_bytes(value: object) -> bytes:
    import json

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
            count = os.write(descriptor, remaining)
            if count <= 0:
                _integrity_fail(f"short write while creating {path.name}")
            remaining = remaining[count:]
        os.fsync(descriptor)
    except OSError as error:
        raise AAMASSubmissionIntegrityError(
            f"cannot exclusively create {path.name}: {error}"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _default_validity_thresholds() -> ValidityThresholds:
    """Return thresholds fixed independently of every benchmark outcome."""

    return ValidityThresholds(
        max_ppc_failure_green=0.10,
        max_ppc_failure_yellow=0.20,
        max_ood_distance_green=0.10,
        max_ood_distance_yellow=0.20,
        max_model_disagreement_green=0.10,
        max_model_disagreement_yellow=0.20,
        max_seed_instability_green=0.10,
        max_seed_instability_yellow=0.20,
        min_coverage_green=0.90,
        min_coverage_yellow=0.80,
        minimum_coverage_history=3,
        supported_group_labels=("latent_strategy",),
        required_mechanisms=(
            "event_keyed_exogenous_shocks",
            "integrated_mechanism_ablations",
        ),
    )


def _attempt_definitions() -> tuple[dict[str, object], ...]:
    return (
        {
            "attempt_id": "rq1_random_coupling_and_mechanisms",
            "dependencies": [],
            "failure_propagation": "any_registered_rq1_failure_fails_rq1",
            "research_question": AAMASResearchQuestion.RQ1.value,
        },
        {
            "attempt_id": "rq2_calibration_recovery",
            "dependencies": [],
            "failure_propagation": "no_complete_case_fallback",
            "research_question": AAMASResearchQuestion.RQ2.value,
        },
        {
            "attempt_id": "rq3_selective_validity",
            "dependencies": ["rq2_calibration_recovery"],
            "failure_propagation": "incomplete_rq2_or_target_rule_blocks_rq3",
            "research_question": AAMASResearchQuestion.RQ3.value,
        },
    )


def _output_definitions() -> tuple[dict[str, object], ...]:
    """Frozen formula registry; reporting code may not select alternatives."""

    return (
        {
            "ddof": 1,
            "denominator": "all_registered_replicates_in_each_rng_cell",
            "formula": (
                "cell_by_mode mean=sum(d)/n; sample_variance=sum((d-mean)^2)/(n-1); "
                "mcse=sqrt(sample_variance/n); policy_effect_bias=mean-policy_effect; "
                "post_branch_mismatch_rate=mismatch_count/[n*(steps-branch-1)]; "
                "mode_variance_ratio=mode_variance/independent_variance_or_null"
            ),
            "output_id": "rq1_rng_cell_mode_statistics",
        },
        {
            "ddof": None,
            "denominator": "complete_frozen_mechanism_grid_only",
            "formula": "retain audited child grid and its exact failure ledger",
            "output_id": "rq1_mechanism_ablation_grid",
        },
        {
            "ddof": 0,
            "denominator": "all_expected_truths_for_each_role_parameter",
            "formula": (
                "bias=mean(median-truth); mae=mean(abs(median-truth)); "
                "rmse=sqrt(mean((median-truth)^2)); coverage90=mean(covered); "
                "mean_width90=mean(upper90-lower90)"
            ),
            "output_id": "rq2_role_parameter_recovery",
        },
        {
            "ddof": None,
            "denominator": "all_held_out_ppc_cells_for_each_role_statistic",
            "formula": "held_out_coverage=covered_count/cell_count",
            "output_id": "rq2_held_out_ppc_coverage",
        },
        {
            "ddof": None,
            "denominator": "all_sbc_replicates_for_each_role_parameter",
            "formula": "sum_rank_histograms_and_weight_interval_coverage_by_replicates",
            "output_id": "rq2_sbc_histogram_and_coverage",
        },
        {
            "ddof": None,
            "denominator": "all_identifiability_cells_for_each_role_parameter",
            "formula": (
                "mean_boolean_rates_for_boundary_pileup_multimodality_"
                "confounding_and_nonidentifiability"
            ),
            "output_id": "rq2_identifiability_rates",
        },
        {
            "ddof": None,
            "denominator": "all_preregistered_target_truths_only",
            "formula": (
                "retain_n_coverage_and_mean_loss_at_each_fixed_threshold; "
                "no_design_truths_and_no_complete_case_fallback"
            ),
            "finite_sample_interval": None,
            "stability_claim_authorized": False,
            "output_id": "rq3_target_only_risk_coverage",
        },
        {
            "ddof": None,
            "denominator": "typed_replay_audited_evidence_only",
            "formula": (
                "pre_policy_ppc_failure=1-pooled_target_held_out_ppc_coverage; "
                "parameter_support_failure=pooled_target_boundary_pileup_rate; "
                "event_keyed_exogenous_shocks_present_only_if_every_registered_"
                "rq1_attempt_completed; integrated_mechanism_ablations_absent_"
                "from_core_evidence; "
                "all_other_unverified_evidence=null; then_apply_prefrozen_thresholds"
            ),
            "output_id": "rq3_validity_certificate",
        },
    )


def _rq1_work(plan: AAMASBenchmarkPlan) -> int:
    rng = len(plan.rng_cells) * plan.rng_replicate_count * plan.rng_step_count
    ablation = (
        len(plan.mechanism_ablation_spec.variants)
        * plan.mechanism_replicate_count
        * plan.mechanism_step_count
    )
    return rng + ablation


@dataclass(frozen=True, slots=True)
class AAMASSubmissionPlan:
    """Immutable three-question plan with pre-result validity thresholds."""

    submission_id: str = "policytwin-aamas-synthetic-submission-v1"
    rq1_plan: AAMASBenchmarkPlan = field(default_factory=AAMASBenchmarkPlan)
    calibration_spec: AAMASCalibrationSpec = field(default_factory=AAMASCalibrationSpec)
    validity_thresholds: ValidityThresholds = field(
        default_factory=_default_validity_thresholds
    )
    calibration_resources: ResourceBudget = ResourceBudget(
        cpu_cores=1,
        memory_mb=4096,
        wall_time_seconds=14_400,
    )
    synthetic_only: Literal[True] = True
    schema_version: str = AAMAS_SUBMISSION_PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            not isinstance(self.submission_id, str)
            or not self.submission_id
            or len(self.submission_id) > 128
        ):
            _fail("submission_id must be a non-empty bounded string")
        if not isinstance(self.rq1_plan, AAMASBenchmarkPlan):
            _fail("rq1_plan must be an AAMASBenchmarkPlan")
        if not isinstance(self.calibration_spec, AAMASCalibrationSpec):
            _fail("calibration_spec must be an AAMASCalibrationSpec")
        if not isinstance(self.validity_thresholds, ValidityThresholds):
            _fail("validity_thresholds must be a ValidityThresholds")
        if not isinstance(self.calibration_resources, ResourceBudget):
            _fail("calibration_resources must be a ResourceBudget")
        if self.synthetic_only is not True:
            _fail("AAMAS submission plan must remain synthetic_only")
        if self.schema_version != AAMAS_SUBMISSION_PLAN_SCHEMA_VERSION:
            _fail("unsupported AAMAS submission plan schema")

    @property
    def execution_work(self) -> int:
        return _rq1_work(self.rq1_plan) + self.calibration_spec.execution_work

    @property
    def is_formal_scale(self) -> bool:
        return (
            self.rq1_plan == AAMASBenchmarkPlan()
            and self.calibration_spec == AAMASCalibrationSpec()
        )

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def to_mapping(self) -> dict[str, object]:
        return {
            "attempt_registry": list(_attempt_definitions()),
            "calibration_resources": self.calibration_resources.to_mapping(),
            "calibration_spec": self.calibration_spec.to_mapping(),
            "claim_scope": AAMAS_SUBMISSION_CLAIM_SCOPE,
            "failure_propagation": "fail_closed_no_complete_case_fallback",
            "output_registry": list(_output_definitions()),
            "rq1_plan": self.rq1_plan.to_mapping(),
            "schema_version": self.schema_version,
            "submission_id": self.submission_id,
            "synthetic_only": self.synthetic_only,
            "timing_in_scientific_hash": False,
            "validity_thresholds": self.validity_thresholds.to_mapping(),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> AAMASSubmissionPlan:
        _strict_fields(
            value,
            frozenset(cls().to_mapping()),
            "AAMAS submission plan",
        )
        if (
            value["claim_scope"] != AAMAS_SUBMISSION_CLAIM_SCOPE
            or value["synthetic_only"] is not True
            or value["timing_in_scientific_hash"] is not False
            or value["failure_propagation"] != "fail_closed_no_complete_case_fallback"
        ):
            _integrity_fail("AAMAS submission plan authority boundary changed")
        identifier = value["submission_id"]
        schema = value["schema_version"]
        if not isinstance(identifier, str) or not isinstance(schema, str):
            _integrity_fail("submission plan identifiers must be strings")
        plan = cls(
            submission_id=identifier,
            rq1_plan=AAMASBenchmarkPlan.from_mapping(
                _object(value["rq1_plan"], "rq1_plan")
            ),
            calibration_spec=AAMASCalibrationSpec.from_mapping(
                _object(value["calibration_spec"], "calibration_spec")
            ),
            validity_thresholds=ValidityThresholds.from_mapping(
                _object(value["validity_thresholds"], "validity_thresholds")
            ),
            calibration_resources=ResourceBudget.from_mapping(
                _object(value["calibration_resources"], "calibration_resources")
            ),
            synthetic_only=True,
            schema_version=schema,
        )
        if value["attempt_registry"] != list(_attempt_definitions()):
            _integrity_fail("submission attempt registry changed")
        if value["output_registry"] != list(_output_definitions()):
            _integrity_fail("submission output formulas or denominators changed")
        if plan.to_mapping() != dict(value):
            _integrity_fail("submission plan is not in canonical form")
        return plan


@dataclass(frozen=True, slots=True)
class AAMASSubmissionTiming:
    component_id: str
    wall_seconds: float

    def __post_init__(self) -> None:
        if self.component_id not in {"rq1", "calibration"}:
            _fail("submission timing component_id is invalid")
        if (
            isinstance(self.wall_seconds, bool)
            or not isinstance(self.wall_seconds, int | float)
            or not math.isfinite(float(self.wall_seconds))
            or float(self.wall_seconds) < 0.0
        ):
            _fail("submission wall_seconds must be finite and non-negative")
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


@dataclass(frozen=True, slots=True)
class AAMASSubmissionRun:
    """Two child executions plus non-scientific timing/network evidence."""

    plan: AAMASSubmissionPlan
    expected_plan_sha256: str
    rq1_run: AAMASBenchmarkRun
    calibration_collection: AAMASCalibrationRunCollection
    timings: tuple[AAMASSubmissionTiming, ...]
    network_audit_events: tuple[dict[str, object], ...]
    started_at: datetime
    finished_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.plan, AAMASSubmissionPlan):
            _fail("submission run plan is invalid")
        if self.expected_plan_sha256 != self.plan.sha256:
            _fail("submission run differs from its external pre-run commitment")
        if self.rq1_run.artifact.plan != self.plan.rq1_plan:
            _fail("submission RQ1 run used a different plan")
        if self.calibration_collection.plan.spec != self.plan.calibration_spec:
            _fail("submission calibration run used a different spec")
        if tuple(item.component_id for item in self.timings) != (
            "rq1",
            "calibration",
        ):
            _fail("submission timings must cover both children in order")
        if not self.network_audit_events:
            _fail("submission run lacks no-network audit events")
        actions = tuple(item.get("action") for item in self.network_audit_events)
        if actions != ("enter_frozen_evaluation", "exit_frozen_evaluation"):
            _fail("submission run no-network boundary is incomplete")
        for field_name in ("started_at", "finished_at"):
            value = getattr(self, field_name)
            if not isinstance(value, datetime) or value.tzinfo is None:
                _fail(f"{field_name} must be timezone aware")
            object.__setattr__(self, field_name, value.astimezone(UTC))
        if self.finished_at < self.started_at:
            _fail("submission finished_at precedes started_at")


def _formal_authorization_required(plan: AAMASSubmissionPlan) -> bool:
    formal = AAMASSubmissionPlan()
    return (
        _rq1_work(plan.rq1_plan) >= _rq1_work(formal.rq1_plan)
        or plan.calibration_spec.execution_work
        >= formal.calibration_spec.execution_work
        or plan.execution_work >= formal.execution_work
    )


def run_aamas_submission(
    plan: AAMASSubmissionPlan,
    *,
    expected_plan_sha256: str,
    allow_formal_scale: bool = False,
) -> AAMASSubmissionRun:
    """Execute both synthetic children under one Python no-network boundary."""

    if not isinstance(plan, AAMASSubmissionPlan):
        _fail("plan must be an AAMASSubmissionPlan")
    if _sha256(expected_plan_sha256, "expected_plan_sha256") != plan.sha256:
        _integrity_fail("plan differs from the caller-retained pre-run commitment")
    if not isinstance(allow_formal_scale, bool):
        _fail("allow_formal_scale must be boolean")
    if _formal_authorization_required(plan) and not allow_formal_scale:
        _integrity_fail(
            "formal or larger AAMAS submission scale requires allow_formal_scale=True"
        )

    started_at = datetime.now(UTC)
    policy = _network_policy()
    with tempfile.TemporaryDirectory(prefix="policytwin-aamas-network-") as temporary:
        audit = TemporalAuditLog(Path(temporary) / "network.jsonl")
        firewall = TemporalFirewall(policy, audit)
        with firewall.frozen_evaluation():
            start = time.perf_counter()
            rq1 = run_aamas_benchmark(
                plan.rq1_plan,
                allow_formal_scale=allow_formal_scale,
            )
            rq1_timing = AAMASSubmissionTiming("rq1", time.perf_counter() - start)
            start = time.perf_counter()
            calibration = run_aamas_calibration_collection(
                plan.calibration_spec,
                allow_formal_scale=allow_formal_scale,
            )
            calibration_timing = AAMASSubmissionTiming(
                "calibration", time.perf_counter() - start
            )
        network_events = tuple(dict(item) for item in audit.events())
    return AAMASSubmissionRun(
        plan=plan,
        expected_plan_sha256=expected_plan_sha256,
        rq1_run=rq1,
        calibration_collection=calibration,
        timings=(rq1_timing, calibration_timing),
        network_audit_events=network_events,
        started_at=started_at,
        finished_at=datetime.now(UTC),
    )


class SubmissionAttemptStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED_UPSTREAM = "blocked_upstream"


@dataclass(frozen=True, slots=True)
class AAMASSubmissionAttempt:
    attempt_id: str
    research_question: AAMASResearchQuestion
    status: SubmissionAttemptStatus
    config_sha256: str
    source_scientific_sha256: str
    expected_subattempts: int
    completed_subattempts: int
    failed_subattempts: int
    reason: str | None

    def __post_init__(self) -> None:
        _sha256(self.config_sha256, "attempt config_sha256")
        _sha256(self.source_scientific_sha256, "attempt source_scientific_sha256")
        if self.expected_subattempts < 1:
            _fail("submission attempt expected_subattempts must be positive")
        if not 0 <= self.completed_subattempts <= self.expected_subattempts:
            _fail("submission attempt completed_subattempts is invalid")
        if not 0 <= self.failed_subattempts <= self.expected_subattempts:
            _fail("submission attempt failed_subattempts is invalid")
        if (
            self.completed_subattempts + self.failed_subattempts
            > self.expected_subattempts
        ):
            _fail("submission attempt terminal counts exceed the frozen denominator")
        if self.status is SubmissionAttemptStatus.COMPLETED:
            if (
                self.completed_subattempts != self.expected_subattempts
                or self.failed_subattempts != 0
                or self.reason is not None
            ):
                _fail("completed submission attempt is inconsistent")
        elif not self.reason:
            _fail("non-completed submission attempt must retain a reason")

    def to_mapping(self) -> dict[str, object]:
        return {
            "attempt_id": self.attempt_id,
            "completed_subattempts": self.completed_subattempts,
            "config_sha256": self.config_sha256,
            "expected_subattempts": self.expected_subattempts,
            "failed_subattempts": self.failed_subattempts,
            "reason": self.reason,
            "research_question": self.research_question.value,
            "source_scientific_sha256": self.source_scientific_sha256,
            "status": self.status.value,
        }


def _rq1_outputs(
    directory: Path,
    artifact: AAMASBenchmarkArtifact,
) -> dict[str, object]:
    source_by_attempt = {item.attempt_id: item for item in artifact.source_artifacts}
    rows: list[dict[str, object]] = []
    ratios: list[dict[str, object]] = []
    expected_rng = tuple(
        item
        for item in artifact.plan.expected_attempts
        if item.kind is AAMASAttemptKind.RNG_MECHANISM_SMOKE
    )
    for cell, expected in zip(artifact.plan.rng_cells, expected_rng, strict=True):
        source = source_by_attempt.get(expected.attempt_id)
        mode_variances: dict[RNGCouplingMode, float] = {}
        cell_rows: list[dict[str, object]] = []
        if source is None:
            for mode in RNGCouplingMode:
                cell_rows.append(
                    {
                        "cell_id": cell.cell_id,
                        "mode": mode.value,
                        "status": "blocked_missing_failed_source",
                        "statistics": None,
                    }
                )
        else:
            nested = MechanismSmokeArtifact.from_mapping(
                read_json_object(directory / source.relative_path)
            )
            for receipt in nested.receipts:
                differences = tuple(float(item) for item in receipt.differences)
                mean = fmean(differences)
                sample_variance = variance(differences)
                post_steps = nested.spec.step_count - nested.spec.branch_step - 1
                mismatch_count = 0
                for replicate in range(nested.spec.replicate_count):
                    start = (
                        replicate * nested.spec.step_count + nested.spec.branch_step + 1
                    )
                    end = (replicate + 1) * nested.spec.step_count
                    mismatch_count += sum(
                        left != right
                        for left, right in zip(
                            receipt.baseline_draws[start:end],
                            receipt.candidate_draws[start:end],
                            strict=True,
                        )
                    )
                mismatch_denominator = nested.spec.replicate_count * post_steps
                mode_variances[receipt.mode] = sample_variance
                cell_rows.append(
                    {
                        "cell_id": cell.cell_id,
                        "mode": receipt.mode.value,
                        "status": "complete",
                        "statistics": {
                            "difference_count": len(differences),
                            "mean_difference": mean,
                            "policy_effect_bias": mean - nested.spec.policy_effect,
                            "post_branch_mismatch_count": mismatch_count,
                            "post_branch_mismatch_denominator": mismatch_denominator,
                            "post_branch_mismatch_rate": (
                                mismatch_count / mismatch_denominator
                            ),
                            "sample_variance_ddof_1": sample_variance,
                            "standard_error": math.sqrt(
                                sample_variance / len(differences)
                            ),
                        },
                    }
                )
        rows.extend(cell_rows)
        independent = mode_variances.get(RNGCouplingMode.INDEPENDENT)
        event_ratio: float | None = None
        sequential_ratio: float | None = None
        if independent is not None and independent != 0.0:
            event_ratio = (
                mode_variances.get(RNGCouplingMode.EVENT_KEYED, 0.0) / independent
            )
            sequential_ratio = (
                mode_variances.get(RNGCouplingMode.SAME_SEQUENTIAL, 0.0) / independent
            )
        ratios.append(
            {
                "cell_id": cell.cell_id,
                "denominator_mode": RNGCouplingMode.INDEPENDENT.value,
                "event_keyed_to_independent": event_ratio,
                "same_sequential_to_independent": sequential_ratio,
            }
        )
    mechanism_attempt = next(
        item
        for item in artifact.attempts
        if item.kind is AAMASAttemptKind.MECHANISM_ABLATION
    )
    return {
        "mechanism_ablation": {
            "artifact_sha256": mechanism_attempt.artifact_sha256,
            "completed_subattempts": mechanism_attempt.completed_subattempts,
            "failed_subattempts": mechanism_attempt.failed_subattempts,
            "status": mechanism_attempt.status.value,
        },
        "rng_cell_mode_statistics": rows,
        "variance_ratios": ratios,
    }


def _calibration_attempt_counts(
    registry: Mapping[str, object],
) -> tuple[int, int, int]:
    raw = _array(registry.get("attempts"), "calibration attempt registry")
    completed = 0
    failed = 0
    for item in raw:
        row = _object(item, "calibration attempt row")
        status = row.get("status")
        if status == CalibrationAttemptStatus.COMPLETED.value:
            completed += 1
        elif status == CalibrationAttemptStatus.FAILED.value:
            failed += 1
        else:
            _integrity_fail("calibration attempt registry has an unknown status")
    return len(raw), completed, failed


def _rq2_outputs(
    plan: AAMASSubmissionPlan,
    calibration_results: Mapping[str, object],
    *,
    every_attempt_completed: bool,
) -> dict[str, object]:
    truth_rows = _array(calibration_results.get("truth_results"), "truth_results")
    expected_count = (
        plan.calibration_spec.design_truth_count
        + plan.calibration_spec.target_truth_count
    )
    if not every_attempt_completed or len(truth_rows) != expected_count:
        return {
            "aggregate_status": "blocked_no_complete_case_fallback",
            "completed_truth_count": len(truth_rows),
            "expected_truth_count": expected_count,
            "held_out_ppc_coverage": [],
            "identifiability_rates": [],
            "parameter_recovery": [],
            "sbc": [],
        }

    recovery_cells: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    ppc_cells: dict[tuple[str, str], list[bool]] = {}
    ident_cells: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    sbc_cells: dict[tuple[str, str], list[tuple[int, Mapping[str, object]]]] = {}
    for raw_truth in truth_rows:
        truth = _object(raw_truth, "truth result")
        role = truth.get("role")
        if role not in {TruthRole.DESIGN.value, TruthRole.TARGET.value}:
            _integrity_fail("truth result has an invalid role")
        role_text = cast(str, role)
        for raw_recovery in _array(truth.get("recovery"), "recovery"):
            recovery = _object(raw_recovery, "recovery cell")
            parameter = recovery.get("parameter")
            if not isinstance(parameter, str):
                _integrity_fail("recovery parameter must be a string")
            recovery_cells.setdefault((role_text, parameter), []).append(recovery)
        for raw_ppc in _array(truth.get("ppc"), "PPC"):
            ppc = _object(raw_ppc, "PPC cell")
            if ppc.get("role") != SummaryRole.HELD_OUT_VALIDATION.value:
                continue
            name = ppc.get("name")
            covered = ppc.get("covered")
            if not isinstance(name, str) or not isinstance(covered, bool):
                _integrity_fail("held-out PPC cell is malformed")
            ppc_cells.setdefault((role_text, name), []).append(covered)
        for raw_ident in _array(truth.get("identifiability"), "identifiability"):
            ident = _object(raw_ident, "identifiability cell")
            parameter = ident.get("parameter")
            if not isinstance(parameter, str):
                _integrity_fail("identifiability parameter must be a string")
            ident_cells.setdefault((role_text, parameter), []).append(ident)
        sbc = _object(truth.get("sbc"), "SBC")
        replicates = _integer(sbc.get("replicates"), "SBC replicates", minimum=1)
        for raw_parameter in _array(sbc.get("parameters"), "SBC parameters"):
            parameter_row = _object(raw_parameter, "SBC parameter")
            parameter = parameter_row.get("parameter")
            if not isinstance(parameter, str):
                _integrity_fail("SBC parameter must be a string")
            sbc_cells.setdefault((role_text, parameter), []).append(
                (replicates, parameter_row)
            )

    recovery_output: list[dict[str, object]] = []
    for (role, parameter), cells in sorted(recovery_cells.items()):
        errors = [
            _number(cell.get("posterior_median"), "posterior_median")
            - _number(cell.get("truth"), "truth")
            for cell in cells
        ]
        covered = [cell.get("covered") is True for cell in cells]
        widths = [
            _number(cell.get("upper_90"), "upper_90")
            - _number(cell.get("lower_90"), "lower_90")
            for cell in cells
        ]
        recovery_output.append(
            {
                "bias": fmean(errors),
                "coverage_90": fmean(covered),
                "denominator": len(cells),
                "mae": fmean(abs(item) for item in errors),
                "mean_width_90": fmean(widths),
                "parameter": parameter,
                "rmse": math.sqrt(fmean(item * item for item in errors)),
                "role": role,
            }
        )

    ppc_output = [
        {
            "covered_count": sum(values),
            "coverage": fmean(values),
            "denominator": len(values),
            "role": role,
            "statistic": statistic,
        }
        for (role, statistic), values in sorted(ppc_cells.items())
    ]
    ident_output: list[dict[str, object]] = []
    for (role, parameter), cells in sorted(ident_cells.items()):

        def rate(
            field_name: str,
            source_cells: Sequence[Mapping[str, object]] = cells,
        ) -> float:
            values = [cell.get(field_name) for cell in source_cells]
            if not all(isinstance(item, bool) for item in values):
                _integrity_fail(f"identifiability {field_name} must be boolean")
            return fmean(cast(Sequence[bool], values))

        ident_output.append(
            {
                "boundary_pileup_rate": rate("boundary_pileup"),
                "confounded_rate": rate("confounded"),
                "denominator": len(cells),
                "multimodal_rate": rate("multimodal"),
                "nonidentifiable_rate": 1.0 - rate("estimable_for_reporting"),
                "parameter": parameter,
                "role": role,
            }
        )

    sbc_output: list[dict[str, object]] = []
    for (role, parameter), sbc_group in sorted(sbc_cells.items()):
        histograms = [
            tuple(
                _integer(item, "SBC histogram count")
                for item in _array(cell.get("histogram"), "SBC histogram")
            )
            for _, cell in sbc_group
        ]
        if len({len(item) for item in histograms}) != 1:
            _integrity_fail("SBC histogram support changed across truths")
        histogram = [sum(items) for items in zip(*histograms, strict=True)]
        denominator = sum(replicates for replicates, _ in sbc_group)
        coverage_numerator = sum(
            replicates * _number(cell.get("interval_coverage_90"), "SBC coverage")
            for replicates, cell in sbc_group
        )
        sbc_output.append(
            {
                "coverage_90": coverage_numerator / denominator,
                "rank_count": sum(histogram),
                "rank_histogram": histogram,
                "replicate_denominator": denominator,
                "parameter": parameter,
                "role": role,
            }
        )
    return {
        "aggregate_status": "complete_all_expected_truths",
        "completed_truth_count": len(truth_rows),
        "expected_truth_count": expected_count,
        "held_out_ppc_coverage": ppc_output,
        "identifiability_rates": ident_output,
        "parameter_recovery": recovery_output,
        "sbc": sbc_output,
    }


def _certificate_from_sources(
    plan: AAMASSubmissionPlan,
    rq1_artifact: AAMASBenchmarkArtifact,
    rq2_outputs: Mapping[str, object],
) -> ValidityCertificate:
    rq1_attempts = tuple(
        item
        for item in rq1_artifact.attempts
        if item.research_question is AAMASResearchQuestion.RQ1
    )
    active_mechanisms: tuple[str, ...] | None = None
    if rq1_attempts and all(
        item.status is AAMASAttemptStatus.COMPLETED for item in rq1_attempts
    ):
        active_mechanisms = ("event_keyed_exogenous_shocks",)

    ppc_failure: float | None = None
    support_failure: float | None = None
    if rq2_outputs.get("aggregate_status") == "complete_all_expected_truths":
        ppc_rows = [
            _object(item, "PPC aggregate")
            for item in _array(
                rq2_outputs.get("held_out_ppc_coverage"), "PPC aggregates"
            )
            if _object(item, "PPC aggregate").get("role") == TruthRole.TARGET.value
        ]
        denominator = sum(
            _integer(item.get("denominator"), "PPC denominator", minimum=1)
            for item in ppc_rows
        )
        if denominator:
            covered = sum(
                _integer(item.get("covered_count"), "PPC covered count")
                for item in ppc_rows
            )
            ppc_failure = 1.0 - covered / denominator
        ident_rows = [
            _object(item, "identifiability aggregate")
            for item in _array(
                rq2_outputs.get("identifiability_rates"),
                "identifiability aggregates",
            )
            if _object(item, "identifiability aggregate").get("role")
            == TruthRole.TARGET.value
        ]
        ident_denominator = sum(
            _integer(item.get("denominator"), "identifiability denominator", minimum=1)
            for item in ident_rows
        )
        if ident_denominator:
            support_failure = (
                sum(
                    _number(item.get("boundary_pileup_rate"), "boundary pileup rate")
                    * _integer(
                        item.get("denominator"),
                        "identifiability denominator",
                        minimum=1,
                    )
                    for item in ident_rows
                )
                / ident_denominator
            )

    evidence = ValidityEvidence(
        temporal_leakage_detected=None,
        pre_policy_ppc_failure_rate=ppc_failure,
        ood_policy_distance=None,
        model_disagreement=None,
        seed_instability=None,
        empirical_coverage_history=None,
        observed_group_labels=None,
        active_mechanisms=active_mechanisms,
        prompt_instability=None,
        parameter_support_failure_rate=support_failure,
    )
    certificate = build_validity_certificate(
        evidence,
        plan.validity_thresholds,
        synthetic_only=True,
    )
    if certificate.level is not ValidityLevel.ABSTAIN:
        _integrity_fail("submission validity certificate exceeded its authority")
    return certificate


@dataclass(frozen=True, slots=True)
class _DerivedSubmission:
    result: dict[str, object]
    attempt_registry: dict[str, object]
    attempts: tuple[AAMASSubmissionAttempt, ...]
    certificate: ValidityCertificate
    resources: tuple[dict[str, object], ...]
    failures: tuple[dict[str, object], ...]
    core_registered_outputs_complete: bool


def _derive_submission(
    plan: AAMASSubmissionPlan,
    rq1_directory: Path,
    calibration_directory: Path,
) -> _DerivedSubmission:
    rq1_artifact = AAMASBenchmarkArtifact.from_mapping(
        read_json_object(rq1_directory / "result.json")
    )
    calibration_results = read_json_object(calibration_directory / "results.json")
    calibration_registry = read_json_object(
        calibration_directory / "attempt_registry.json"
    )
    expected_calibration, completed_calibration, failed_calibration = (
        _calibration_attempt_counts(calibration_registry)
    )
    every_calibration_completed = (
        completed_calibration == expected_calibration and failed_calibration == 0
    )
    rq1_outputs = _rq1_outputs(rq1_directory, rq1_artifact)
    rq2_outputs = _rq2_outputs(
        plan,
        calibration_results,
        every_attempt_completed=every_calibration_completed,
    )
    risk = _object(calibration_results.get("risk_coverage"), "risk_coverage")
    risk_records = _array(risk.get("records"), "risk coverage records")
    target_ids = tuple(
        item.get("prediction_id")
        for item in (_object(raw, "risk coverage record") for raw in risk_records)
    )
    expected_target_ids = tuple(
        f"target-{index:03d}"
        for index in range(plan.calibration_spec.target_truth_count)
    )
    rq3_complete = (
        every_calibration_completed
        and risk.get("status") == "complete_target_only"
        and risk.get("evaluation") is not None
        and target_ids == expected_target_ids
    )
    rq3_output = {
        "finite_sample_interval": None,
        "finite_sample_interval_status": "not_registered_in_schema_v1",
        "risk_coverage": dict(risk),
        "stability_claim_authorized": False,
        "status": "complete_target_only" if rq3_complete else "blocked_incomplete",
        "target_denominator": plan.calibration_spec.target_truth_count,
    }
    certificate = _certificate_from_sources(plan, rq1_artifact, rq2_outputs)

    rq1_attempts = tuple(
        item
        for item in rq1_artifact.attempts
        if item.research_question is AAMASResearchQuestion.RQ1
    )
    rq1_completed = sum(
        item.status is AAMASAttemptStatus.COMPLETED for item in rq1_attempts
    )
    rq1_failed = sum(item.status is AAMASAttemptStatus.FAILED for item in rq1_attempts)
    rq1_status = (
        SubmissionAttemptStatus.COMPLETED
        if rq1_completed == len(rq1_attempts) and rq1_failed == 0
        else SubmissionAttemptStatus.FAILED
    )
    rq1_reason = (
        None
        if rq1_status is SubmissionAttemptStatus.COMPLETED
        else ("one or more registered RQ1 attempts failed")
    )
    rq2_status = (
        SubmissionAttemptStatus.COMPLETED
        if every_calibration_completed
        else SubmissionAttemptStatus.FAILED
    )
    rq2_reason = (
        None
        if every_calibration_completed
        else ("one or more expected calibration truths failed; aggregation blocked")
    )
    if not every_calibration_completed:
        rq3_status = SubmissionAttemptStatus.BLOCKED_UPSTREAM
        rq3_reason = "RQ2 is incomplete"
    elif rq3_complete:
        rq3_status = SubmissionAttemptStatus.COMPLETED
        rq3_reason = None
    else:
        rq3_status = SubmissionAttemptStatus.FAILED
        rq3_reason = "registered target-only selective rule did not complete"

    calibration_scientific_sha256 = sha256_bytes(
        canonical_json_bytes(
            {
                "rq2": rq2_outputs,
                "rq3": rq3_output,
            }
        )
    )
    rq3_config_sha256 = sha256_bytes(
        canonical_json_bytes(
            {
                "calibration_spec_sha256": plan.calibration_spec.sha256,
                "output_definition": next(
                    item
                    for item in _output_definitions()
                    if item["output_id"] == "rq3_target_only_risk_coverage"
                ),
            }
        )
    )
    attempts = (
        AAMASSubmissionAttempt(
            "rq1_random_coupling_and_mechanisms",
            AAMASResearchQuestion.RQ1,
            rq1_status,
            plan.rq1_plan.sha256,
            rq1_artifact.sha256,
            len(rq1_attempts),
            rq1_completed,
            rq1_failed,
            rq1_reason,
        ),
        AAMASSubmissionAttempt(
            "rq2_calibration_recovery",
            AAMASResearchQuestion.RQ2,
            rq2_status,
            plan.calibration_spec.sha256,
            calibration_scientific_sha256,
            expected_calibration,
            completed_calibration,
            failed_calibration,
            rq2_reason,
        ),
        AAMASSubmissionAttempt(
            "rq3_selective_validity",
            AAMASResearchQuestion.RQ3,
            rq3_status,
            rq3_config_sha256,
            calibration_scientific_sha256,
            plan.calibration_spec.target_truth_count,
            len(risk_records) if every_calibration_completed else 0,
            0,
            rq3_reason,
        ),
    )
    core_registered_outputs_complete = all(
        item.status is SubmissionAttemptStatus.COMPLETED for item in attempts
    )
    attempt_registry = {
        "attempts": [item.to_mapping() for item in attempts],
        "failure_propagation": "fail_closed_no_complete_case_fallback",
        "plan_sha256": plan.sha256,
        "core_registered_outputs_complete": core_registered_outputs_complete,
        "schema_version": 2,
    }
    attempt_registry_sha256 = sha256_bytes(canonical_json_bytes(attempt_registry))
    result = {
        "attempt_registry_sha256": attempt_registry_sha256,
        "calibration_scientific_sha256": calibration_scientific_sha256,
        "claim_scope": AAMAS_SUBMISSION_CLAIM_SCOPE,
        "integrity_valid_at_write": True,
        "plan_sha256": plan.sha256,
        "publication_eligible": False,
        "real_policy_claim_authorized": False,
        "rq1": rq1_outputs,
        "rq1_scientific_sha256": rq1_artifact.sha256,
        "rq2": rq2_outputs,
        "rq3": rq3_output,
        "schema_version": AAMAS_SUBMISSION_RESULT_SCHEMA_VERSION,
        "core_registered_outputs_complete": core_registered_outputs_complete,
        "synthetic_only": True,
        "timing_in_scientific_hash": False,
        "validity_certificate_sha256": certificate.sha256,
        "validity_level": certificate.level.value,
    }
    resources = tuple(
        {
            "attempt_id": attempt.attempt_id,
            "event": "requested",
            "event_key": sha256_bytes(
                canonical_json_bytes(
                    {
                        "attempt_id": attempt.attempt_id,
                        "plan_sha256": plan.sha256,
                        "purpose": "resource_request",
                    }
                )
            ),
            "requested_resources": (
                plan.rq1_plan.resources.to_mapping()
                if attempt.research_question is AAMASResearchQuestion.RQ1
                else plan.calibration_resources.to_mapping()
            ),
            "schema_version": 1,
            "status": attempt.status.value,
        }
        for attempt in attempts
    )
    failure_rows: list[dict[str, object]] = [
        {
            "attempt_id": attempt.attempt_id,
            "event": "terminal_noncompletion",
            "event_key": sha256_bytes(
                canonical_json_bytes(
                    {
                        "attempt_id": attempt.attempt_id,
                        "plan_sha256": plan.sha256,
                        "purpose": "failure",
                    }
                )
            ),
            "reason": attempt.reason,
            "schema_version": 1,
            "status": attempt.status.value,
        }
        for attempt in attempts
        if attempt.status is not SubmissionAttemptStatus.COMPLETED
    ]
    failure_summary: dict[str, object] = {
        "event": "failure_summary",
        "event_key": sha256_bytes(
            canonical_json_bytes(
                {"plan_sha256": plan.sha256, "purpose": "failure_summary"}
            )
        ),
        "noncompleted_attempt_count": len(failure_rows),
        "schema_version": 1,
    }
    failures: tuple[dict[str, object], ...] = (
        failure_summary,
        *failure_rows,
    )
    return _DerivedSubmission(
        result,
        attempt_registry,
        attempts,
        certificate,
        resources,
        failures,
        core_registered_outputs_complete,
    )


def _data_manifest() -> dict[str, object]:
    return {
        "claim_scope": "synthetic_empty_manifest",
        "manifest_id": "aamas-synthetic-empty-v1",
        "real_record_count": 0,
        "records": [],
        "schema_version": 1,
        "synthetic_only": True,
    }


def _source_paths(project_root: Path) -> tuple[Path, ...]:
    package = project_root / "src" / "policytwin"
    paths: list[Path] = []
    for candidate in package.rglob("*"):
        if "__pycache__" in candidate.parts or candidate.suffix in {".pyc", ".pyo"}:
            continue
        if candidate.is_symlink():
            _integrity_fail("local source scope contains a symlink")
        if candidate.is_file():
            paths.append(candidate)
    for name in ("pyproject.toml", "uv.lock"):
        candidate = project_root / name
        if candidate.is_symlink():
            _integrity_fail(f"{name} must not be a symlink")
        if candidate.is_file():
            paths.append(candidate)
    return tuple(
        sorted(paths, key=lambda item: item.relative_to(project_root).as_posix())
    )


def _stable_file_row(project_root: Path, path: Path) -> dict[str, object]:
    before = path.stat()
    content = path.read_bytes()
    after = path.stat()
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if (
        not stat.S_ISREG(before.st_mode)
        or identity_before != identity_after
        or before.st_size != len(content)
    ):
        _integrity_fail("local source changed while its manifest was built")
    return {
        "path": path.relative_to(project_root).as_posix(),
        "sha256": sha256_bytes(content),
        "size_bytes": len(content),
    }


def _source_manifest(project_root: Path) -> dict[str, object]:
    rows = [
        _stable_file_row(project_root, path) for path in _source_paths(project_root)
    ]
    return {
        "authority": "local_worktree_bytes_not_external_attestation",
        "complete_for_scope": True,
        "files": rows,
        "schema_version": 1,
        "scope": "all_regular_src_policytwin_files_plus_pyproject_and_uv_lock",
        "source_count": len(rows),
        "tree_sha256": sha256_bytes(canonical_json_bytes(rows)),
    }


def _tree_rows(directory: Path) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            _integrity_fail("bundle tree contains a symlink")
        info = path.stat()
        if path.is_dir():
            if info.st_nlink < 2:
                _integrity_fail("bundle directory has an invalid link count")
            continue
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            _integrity_fail("bundle members must be single-link regular files")
        relative = path.relative_to(directory).as_posix()
        rows.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "size_bytes": info.st_size,
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
        raise AAMASSubmissionIntegrityError(
            f"cannot inspect submission audit root: {error}"
        ) from error
    if not stat.S_ISDIR(root_before.st_mode) or stat.S_ISLNK(root_before.st_mode):
        _integrity_fail("submission audit root must be a real directory")
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
            _integrity_fail("submission audit tree contains an unsafe path")
        try:
            before = path.lstat()
        except OSError as error:
            raise AAMASSubmissionIntegrityError(
                f"cannot inspect submission audit member {relative}: {error}"
            ) from error
        if stat.S_ISLNK(before.st_mode):
            _integrity_fail("submission audit tree contains a symbolic link")
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
                raise AAMASSubmissionIntegrityError(
                    f"cannot read submission audit member {relative}: {error}"
                ) from error
            if len(content) != before.st_size:
                _integrity_fail("submission audit member changed while read")
            row = {
                "kind": "file",
                "mode": stat.S_IMODE(before.st_mode),
                "path": relative,
                "sha256": sha256_bytes(content),
                "size_bytes": len(content),
            }
        else:
            _integrity_fail(
                "submission audit members must be directories or singly linked files"
            )
        try:
            after = path.lstat()
        except OSError as error:
            raise AAMASSubmissionIntegrityError(
                f"cannot recheck submission audit member {relative}: {error}"
            ) from error
        if fingerprint(before) != fingerprint(after):
            _integrity_fail("submission audit member changed while captured")
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
        raise AAMASSubmissionIntegrityError(
            f"cannot finalize submission audit tree: {error}"
        ) from error
    initial_names = tuple(
        path.relative_to(directory).as_posix() for path in initial_paths
    )
    if (
        fingerprint(root_before) != fingerprint(root_after)
        or initial_names != final_paths
    ):
        _integrity_fail("submission audit tree changed while captured")
    return tuple(rows)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with JsonlEventWriter(path) as writer:
        for row in rows:
            writer.write(row)


def _event_keys(run: AAMASSubmissionRun) -> tuple[tuple[str, ...], int]:
    occurrences: list[str] = []
    for _, artifact in run.rq1_run.rng_artifacts:
        for receipt in artifact.receipts:
            occurrences.extend(receipt.event_keys)
    for truth in run.calibration_collection.plan.truths:
        occurrences.extend(truth.event_keys)
    for attempt in run.calibration_collection.attempts:
        if attempt.run is not None:
            occurrences.extend(attempt.run.event_keys)
    unique = tuple(dict.fromkeys(occurrences))
    return unique, len(occurrences)


def _artifact_digest(root: Path, relative: str) -> ArtifactDigest:
    path = root / relative
    return ArtifactDigest(relative, sha256_file(path), path.stat().st_size)


@dataclass(frozen=True, slots=True)
class AAMASSubmissionWriteResult:
    directory: Path
    seal_raw_sha256: str
    plan_sha256: str
    scientific_result_sha256: str
    core_registered_outputs_complete: bool
    publication_eligible: Literal[False] = False


def write_aamas_submission(
    directory: Path,
    *,
    run: AAMASSubmissionRun,
    expected_plan_sha256: str,
    project_root: Path | None = None,
    git_metadata: GitMetadata | None = None,
) -> AAMASSubmissionWriteResult:
    """Replay child writers/audits, then atomically publish an immutable root."""

    if not isinstance(directory, Path) or not isinstance(run, AAMASSubmissionRun):
        _fail("directory and run must be typed submission values")
    expected_plan = _sha256(expected_plan_sha256, "expected_plan_sha256")
    if expected_plan != run.plan.sha256 or expected_plan != run.expected_plan_sha256:
        _integrity_fail("writer plan differs from the pre-run commitment")
    project = (
        Path(__file__).resolve().parents[3]
        if project_root is None
        else project_root.resolve()
    )
    git = collect_git_metadata(project) if git_metadata is None else git_metadata
    if not isinstance(git, GitMetadata):
        _fail("git_metadata must be GitMetadata when provided")
    if directory.exists() or directory.is_symlink():
        _integrity_fail("AAMAS submission output already exists")
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{directory.name}.staging-", dir=directory.parent)
    )
    try:
        rq1_directory = staging / "rq1"
        calibration_directory = staging / "calibration"
        _, rq1_seal_path = write_aamas_benchmark(rq1_directory, run.rq1_run)
        rq1_audit = audit_aamas_benchmark(rq1_directory, run.plan.rq1_plan)
        calibration_write = write_aamas_calibration_collection(
            calibration_directory,
            collection=run.calibration_collection,
            project_root=project,
            git_metadata=git,
        )
        calibration_audit = audit_aamas_calibration_collection(
            calibration_directory,
            expected_seal_sha256=calibration_write.seal_raw_sha256,
        )
        if not rq1_audit.passed or not calibration_audit.valid:
            _integrity_fail("a child writer failed its immediate replay audit")

        derived = _derive_submission(run.plan, rq1_directory, calibration_directory)
        plan_payload = _pretty_json_bytes(run.plan.to_mapping())
        result_payload = _pretty_json_bytes(derived.result)
        registry_payload = _pretty_json_bytes(derived.attempt_registry)
        certificate_payload = _pretty_json_bytes(derived.certificate.to_mapping())
        data_mapping = _data_manifest()
        data_payload = _pretty_json_bytes(data_mapping)
        source_mapping = _source_manifest(project)
        source_payload = _pretty_json_bytes(source_mapping)
        for name, payload in (
            ("plan.json", plan_payload),
            ("result.json", result_payload),
            ("attempt-registry.json", registry_payload),
            ("validity-certificate.json", certificate_payload),
            ("data-manifest.json", data_payload),
            ("source-manifest.json", source_payload),
        ):
            _write_new(staging / name, payload)
        _write_jsonl(staging / "resources.jsonl", derived.resources)
        _write_jsonl(staging / "failures.jsonl", derived.failures)
        _write_jsonl(
            staging / "timing.jsonl",
            tuple(item.to_mapping(run.plan.sha256) for item in run.timings),
        )
        _write_jsonl(staging / "network-audit.jsonl", run.network_audit_events)

        unique_event_keys, occurrence_count = _event_keys(run)
        run_metadata = RunMetadata(
            schema_version=1,
            run_id=run.plan.submission_id,
            experiment_id=run.plan.submission_id,
            claim_scope=AAMAS_SUBMISSION_CLAIM_SCOPE,
            started_at=run.started_at.isoformat(),
            finished_at=run.finished_at.isoformat(),
            git=git,
            config_sha256=run.plan.sha256,
            config_source=SourceArtifact(
                "plan.json",
                sha256_bytes(plan_payload),
                run.plan.sha256,
            ),
            data_manifest_id=cast(str, data_mapping["manifest_id"]),
            data_manifest=SourceArtifact(
                "data-manifest.json",
                sha256_bytes(data_payload),
                sha256_bytes(canonical_json_bytes(data_mapping)),
            ),
            model_checkpoint_sha256=None,
            environment=collect_environment_metadata(project),
            event_keys=build_event_key_metadata(
                schema_version="policytwin.aamas-submission-event-keys.v1",
                root_key=run.plan.sha256,
                keys=unique_event_keys,
            ),
            artifacts=tuple(
                _artifact_digest(staging, relative)
                for relative in (
                    "attempt-registry.json",
                    "calibration/results.json",
                    "calibration/seal.json",
                    "data-manifest.json",
                    "result.json",
                    "rq1/result.json",
                    "rq1/result.sha256",
                    "source-manifest.json",
                    "validity-certificate.json",
                )
            ),
        )
        metadata = {
            "integrity_valid_at_write": True,
            "local_source_manifest_raw_sha256": sha256_bytes(source_payload),
            "local_source_tree_sha256": source_mapping["tree_sha256"],
            "network_audit_path": "network-audit.jsonl",
            "network_mode": "python_process_blocked_temporal_firewall",
            "network_os_isolation_attested": False,
            "plan_precommit_sha256": expected_plan,
            "publication_eligible": False,
            "run_metadata": run_metadata.to_mapping(),
            "core_registered_outputs_complete": (
                derived.core_registered_outputs_complete
            ),
            "schema_version": 2,
            "source_attestation": "dirty_or_local_unattested",
            "synthetic_event_key_occurrence_count": occurrence_count,
        }
        _write_new(staging / "metadata.json", _pretty_json_bytes(metadata))

        root_file_names = _ROOT_FILES - {"manifest.json", "seal.json"}
        root_rows = [
            {
                "path": name,
                "sha256": sha256_file(staging / name),
                "size_bytes": (staging / name).stat().st_size,
            }
            for name in sorted(root_file_names)
        ]
        rq1_rows = _tree_rows(rq1_directory)
        calibration_rows = _tree_rows(calibration_directory)
        manifest = {
            "plan_sha256": run.plan.sha256,
            "root_files": root_rows,
            "schema_version": AAMAS_SUBMISSION_MANIFEST_SCHEMA_VERSION,
            "subbundles": [
                {
                    "external_seal_raw_sha256": sha256_file(rq1_seal_path),
                    "file_count": len(rq1_rows),
                    "path": "rq1",
                    "result_artifact_sha256": rq1_audit.artifact_sha256,
                    "tree_sha256": sha256_bytes(canonical_json_bytes(rq1_rows)),
                },
                {
                    "external_seal_raw_sha256": calibration_write.seal_raw_sha256,
                    "file_count": len(calibration_rows),
                    "path": "calibration",
                    "result_artifact_sha256": calibration_audit.results_sha256,
                    "tree_sha256": sha256_bytes(canonical_json_bytes(calibration_rows)),
                },
            ],
            "timing_in_scientific_hash": False,
        }
        manifest_payload = _pretty_json_bytes(manifest)
        _write_new(staging / "manifest.json", manifest_payload)
        result_scientific_sha256 = sha256_bytes(canonical_json_bytes(derived.result))
        seal = {
            "caller_retained_digest_required": True,
            "manifest_raw_sha256": sha256_bytes(manifest_payload),
            "plan_sha256": run.plan.sha256,
            "result_raw_sha256": sha256_bytes(result_payload),
            "schema_version": AAMAS_SUBMISSION_SEAL_SCHEMA_VERSION,
            "scientific_result_sha256": result_scientific_sha256,
            "timing_bound_by_external_seal": True,
        }
        seal_payload = _pretty_json_bytes(seal)
        _write_new(staging / "seal.json", seal_payload)
        if directory.exists() or directory.is_symlink():
            _integrity_fail("AAMAS submission output appeared before publication")
        os.replace(staging, directory)
        return AAMASSubmissionWriteResult(
            directory=directory,
            seal_raw_sha256=sha256_bytes(seal_payload),
            plan_sha256=run.plan.sha256,
            scientific_result_sha256=result_scientific_sha256,
            core_registered_outputs_complete=(derived.core_registered_outputs_complete),
        )
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _verify_root_membership(directory: Path) -> None:
    if directory.is_symlink() or not directory.is_dir():
        _integrity_fail("submission root must be a real directory")
    entries = {item.name: item for item in directory.iterdir()}
    expected = _ROOT_FILES | _ROOT_DIRECTORIES
    if frozenset(entries) != expected:
        _integrity_fail("submission root fixed membership changed")
    for name in _ROOT_FILES:
        path = entries[name]
        if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
            _integrity_fail(f"submission root file is not regular: {name}")
    for name in _ROOT_DIRECTORIES:
        path = entries[name]
        if path.is_symlink() or not path.is_dir():
            _integrity_fail(f"submission child is not a real directory: {name}")
    _tree_rows(directory)


def _verify_manifest(directory: Path, manifest: Mapping[str, object]) -> None:
    _strict_fields(
        manifest,
        frozenset(
            {
                "plan_sha256",
                "root_files",
                "schema_version",
                "subbundles",
                "timing_in_scientific_hash",
            }
        ),
        "submission manifest",
    )
    if (
        manifest["schema_version"] != AAMAS_SUBMISSION_MANIFEST_SCHEMA_VERSION
        or manifest["timing_in_scientific_hash"] is not False
    ):
        _integrity_fail("submission manifest authority boundary changed")
    expected_root = _ROOT_FILES - {"manifest.json", "seal.json"}
    seen_root: set[str] = set()
    for raw in _array(manifest["root_files"], "manifest root files"):
        row = _object(raw, "manifest root file")
        _strict_fields(row, frozenset({"path", "sha256", "size_bytes"}), "root row")
        path = row["path"]
        if not isinstance(path, str) or path not in expected_root or path in seen_root:
            _integrity_fail("manifest root path membership is invalid")
        seen_root.add(path)
        actual = directory / path
        if (
            row["sha256"] != sha256_file(actual)
            or row["size_bytes"] != actual.stat().st_size
        ):
            _integrity_fail(f"root file was modified: {path}")
    if seen_root != expected_root:
        _integrity_fail("manifest omitted a root file")
    subbundles = _array(manifest["subbundles"], "subbundles")
    if len(subbundles) != 2:
        _integrity_fail("manifest must retain exactly two child bundles")
    seen_children: set[str] = set()
    for raw in subbundles:
        row = _object(raw, "subbundle")
        _strict_fields(
            row,
            frozenset(
                {
                    "external_seal_raw_sha256",
                    "file_count",
                    "path",
                    "result_artifact_sha256",
                    "tree_sha256",
                }
            ),
            "subbundle row",
        )
        name = row["path"]
        if (
            not isinstance(name, str)
            or name not in _ROOT_DIRECTORIES
            or name in seen_children
            or PurePosixPath(name).as_posix() != name
        ):
            _integrity_fail("subbundle path is invalid")
        seen_children.add(name)
        rows = _tree_rows(directory / name)
        if row["file_count"] != len(rows) or row["tree_sha256"] != sha256_bytes(
            canonical_json_bytes(rows)
        ):
            _integrity_fail(f"subbundle tree changed: {name}")
        _sha256(row["external_seal_raw_sha256"], "child external seal")
        _sha256(row["result_artifact_sha256"], "child result artifact")
    if seen_children != _ROOT_DIRECTORIES:
        _integrity_fail("manifest omitted a child bundle")


def _audit_timings(rows: Sequence[Mapping[str, object]], plan_sha256: str) -> None:
    if len(rows) != 2:
        _integrity_fail("timing log must cover exactly two components")
    for expected_component, row in zip(("rq1", "calibration"), rows, strict=True):
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
        _strict_fields(row, expected_fields, "timing event")
        expected = AAMASSubmissionTiming(expected_component, 0.0).to_mapping(
            plan_sha256
        )
        if any(
            row[field] != expected[field]
            for field in expected_fields - {"wall_seconds"}
        ):
            _integrity_fail("timing event identity changed")
        wall = _number(row["wall_seconds"], "timing wall_seconds")
        if wall < 0.0:
            _integrity_fail("timing wall_seconds is negative")


@dataclass(frozen=True, slots=True)
class AAMASSubmissionAuditReport:
    integrity_valid: Literal[True]
    core_registered_outputs_complete: bool
    publication_eligible: Literal[False]
    plan_sha256: str
    scientific_result_sha256: str
    validity_level: ValidityLevel
    rq1_artifact_sha256: str
    calibration_results_sha256: str
    source_tree_sha256: str


def _audit_aamas_submission_snapshot(
    directory: Path,
    *,
    expected_plan: AAMASSubmissionPlan,
    expected_plan_sha256: str,
    expected_seal_sha256: str,
    project_root: Path | None = None,
) -> AAMASSubmissionAuditReport:
    """Audit one verifier-owned stable snapshot without copying it again."""

    if not isinstance(directory, Path) or not isinstance(
        expected_plan, AAMASSubmissionPlan
    ):
        _fail("audit requires typed directory and expected plan")
    plan_digest = _sha256(expected_plan_sha256, "expected_plan_sha256")
    seal_digest = _sha256(expected_seal_sha256, "expected_seal_sha256")
    if plan_digest != expected_plan.sha256:
        _integrity_fail("external plan digest differs from expected plan")
    project = (
        Path(__file__).resolve().parents[3]
        if project_root is None
        else project_root.resolve()
    )
    _verify_root_membership(directory)
    if sha256_file(directory / "seal.json") != seal_digest:
        _integrity_fail("root seal differs from caller-retained digest")
    seal = read_json_object(directory / "seal.json")
    _strict_fields(
        seal,
        frozenset(
            {
                "caller_retained_digest_required",
                "manifest_raw_sha256",
                "plan_sha256",
                "result_raw_sha256",
                "schema_version",
                "scientific_result_sha256",
                "timing_bound_by_external_seal",
            }
        ),
        "root seal",
    )
    if (
        seal["schema_version"] != AAMAS_SUBMISSION_SEAL_SCHEMA_VERSION
        or seal["caller_retained_digest_required"] is not True
        or seal["timing_bound_by_external_seal"] is not True
        or seal["plan_sha256"] != plan_digest
    ):
        _integrity_fail("root seal authority boundary changed")
    if seal["manifest_raw_sha256"] != sha256_file(directory / "manifest.json"):
        _integrity_fail("root manifest differs from root seal")
    if seal["result_raw_sha256"] != sha256_file(directory / "result.json"):
        _integrity_fail("root result raw bytes differ from root seal")
    manifest = read_json_object(directory / "manifest.json")
    _verify_manifest(directory, manifest)
    if manifest["plan_sha256"] != plan_digest:
        _integrity_fail("manifest plan differs from pre-run commitment")
    stored_plan = AAMASSubmissionPlan.from_mapping(
        read_json_object(directory / "plan.json")
    )
    if stored_plan != expected_plan:
        _integrity_fail("stored plan differs from external plan")

    child_rows = {
        cast(str, _object(item, "subbundle")["path"]): _object(item, "subbundle")
        for item in _array(manifest["subbundles"], "subbundles")
    }
    rq1_row = child_rows["rq1"]
    calibration_row = child_rows["calibration"]
    if rq1_row["external_seal_raw_sha256"] != sha256_file(
        directory / "rq1" / "result.sha256"
    ):
        _integrity_fail("RQ1 external seal changed")
    rq1_audit = audit_aamas_benchmark(directory / "rq1", expected_plan.rq1_plan)
    if rq1_row["result_artifact_sha256"] != rq1_audit.artifact_sha256:
        _integrity_fail("RQ1 audit result differs from root manifest")
    calibration_audit = audit_aamas_calibration_collection(
        directory / "calibration",
        expected_seal_sha256=cast(str, calibration_row["external_seal_raw_sha256"]),
    )
    if calibration_row["result_artifact_sha256"] != calibration_audit.results_sha256:
        _integrity_fail("calibration audit result differs from root manifest")

    derived = _derive_submission(
        expected_plan,
        directory / "rq1",
        directory / "calibration",
    )
    result = read_json_object(directory / "result.json")
    _strict_fields(result, frozenset(derived.result), "submission result")
    if result["schema_version"] != AAMAS_SUBMISSION_RESULT_SCHEMA_VERSION:
        _integrity_fail("submission result schema changed")
    if result != derived.result:
        _integrity_fail("root scientific result does not rederive from child artifacts")
    scientific_sha256 = sha256_bytes(canonical_json_bytes(derived.result))
    if seal["scientific_result_sha256"] != scientific_sha256:
        _integrity_fail("scientific result hash differs from root seal")
    if (
        read_json_object(directory / "attempt-registry.json")
        != derived.attempt_registry
    ):
        _integrity_fail("top-level attempt registry changed")
    if (
        read_json_object(directory / "validity-certificate.json")
        != derived.certificate.to_mapping()
    ):
        _integrity_fail("validity certificate does not rederive from sources")
    if read_jsonl_objects(directory / "resources.jsonl") != derived.resources:
        _integrity_fail("resource log changed")
    if read_jsonl_objects(directory / "failures.jsonl") != derived.failures:
        _integrity_fail("failure log omitted or changed a terminal state")
    _audit_timings(
        read_jsonl_objects(directory / "timing.jsonl"),
        expected_plan.sha256,
    )
    network_log = TemporalAuditLog(directory / "network-audit.jsonl")
    network_actions = tuple(item["action"] for item in network_log.events())
    if network_actions != ("enter_frozen_evaluation", "exit_frozen_evaluation"):
        _integrity_fail("no-network audit boundary is incomplete")

    data_mapping = read_json_object(directory / "data-manifest.json")
    if data_mapping != _data_manifest():
        _integrity_fail("synthetic empty data manifest changed")
    source_mapping = read_json_object(directory / "source-manifest.json")
    live_source_mapping = _source_manifest(project)
    if source_mapping != live_source_mapping:
        _integrity_fail("local source tree differs from its bundle manifest")
    metadata = read_json_object(directory / "metadata.json")
    _strict_fields(
        metadata,
        frozenset(
            {
                "integrity_valid_at_write",
                "local_source_manifest_raw_sha256",
                "local_source_tree_sha256",
                "network_audit_path",
                "network_mode",
                "network_os_isolation_attested",
                "plan_precommit_sha256",
                "publication_eligible",
                "run_metadata",
                "schema_version",
                "core_registered_outputs_complete",
                "source_attestation",
                "synthetic_event_key_occurrence_count",
            }
        ),
        "submission metadata",
    )
    if (
        metadata["schema_version"] != 2
        or metadata["integrity_valid_at_write"] is not True
        or metadata["publication_eligible"] is not False
        or metadata["network_os_isolation_attested"] is not False
        or metadata["plan_precommit_sha256"] != plan_digest
        or metadata["core_registered_outputs_complete"]
        is not derived.core_registered_outputs_complete
        or metadata["local_source_manifest_raw_sha256"]
        != sha256_file(directory / "source-manifest.json")
        or metadata["local_source_tree_sha256"] != source_mapping["tree_sha256"]
    ):
        _integrity_fail("submission metadata authority or commitments changed")
    run_metadata = RunMetadata.from_mapping(
        _object(metadata["run_metadata"], "run_metadata")
    )
    if (
        run_metadata.config_sha256 != plan_digest
        or run_metadata.config_source.source_sha256
        != sha256_file(directory / "plan.json")
        or run_metadata.data_manifest.source_sha256
        != sha256_file(directory / "data-manifest.json")
        or run_metadata.environment.lock_sha256
        != collect_environment_metadata(project).lock_sha256
    ):
        _integrity_fail("unified RunMetadata differs from frozen inputs or environment")
    for artifact in run_metadata.artifacts:
        path = directory / artifact.path
        if (
            path.is_symlink()
            or not path.is_file()
            or artifact.sha256 != sha256_file(path)
            or artifact.size_bytes != path.stat().st_size
        ):
            _integrity_fail(f"RunMetadata artifact changed: {artifact.path}")
    return AAMASSubmissionAuditReport(
        integrity_valid=True,
        core_registered_outputs_complete=derived.core_registered_outputs_complete,
        publication_eligible=False,
        plan_sha256=plan_digest,
        scientific_result_sha256=scientific_sha256,
        validity_level=derived.certificate.level,
        rq1_artifact_sha256=rq1_audit.artifact_sha256,
        calibration_results_sha256=calibration_audit.results_sha256,
        source_tree_sha256=cast(str, source_mapping["tree_sha256"]),
    )


def audit_aamas_submission(
    directory: Path,
    *,
    expected_plan: AAMASSubmissionPlan,
    expected_plan_sha256: str,
    expected_seal_sha256: str,
    project_root: Path | None = None,
) -> AAMASSubmissionAuditReport:
    """Audit a private snapshot and prove the caller-visible tree stayed unchanged.

    The original directory is first fully audited and captured as strict tree rows.
    It is then copied with links preserved so a link substitution remains visible to
    the snapshot audit.  All authoritative re-derivation occurs on that private
    copy.  A second snapshot audit and a final live-tree comparison prevent an old
    receipt from being returned for bytes changed during the audit.
    """

    if not isinstance(directory, Path) or not isinstance(
        expected_plan, AAMASSubmissionPlan
    ):
        _fail("audit requires typed directory and expected plan")
    initial_rows = _strict_audit_tree_rows(directory)
    original_audit = _audit_aamas_submission_snapshot(
        directory,
        expected_plan=expected_plan,
        expected_plan_sha256=expected_plan_sha256,
        expected_seal_sha256=expected_seal_sha256,
        project_root=project_root,
    )
    stable_original_rows = _strict_audit_tree_rows(directory)
    if stable_original_rows != initial_rows:
        _integrity_fail("submission tree changed during its initial audit")

    with tempfile.TemporaryDirectory(
        prefix="policytwin-aamas-submission-audit-"
    ) as temporary:
        snapshot = Path(temporary) / "submission"
        try:
            shutil.copytree(directory, snapshot, symlinks=True)
        except OSError as error:
            raise AAMASSubmissionIntegrityError(
                "cannot create private submission audit snapshot"
            ) from error
        snapshot_rows = _strict_audit_tree_rows(snapshot)
        if snapshot_rows != stable_original_rows:
            _integrity_fail(
                "submission tree changed while its private snapshot was created"
            )
        snapshot_audit = _audit_aamas_submission_snapshot(
            snapshot,
            expected_plan=expected_plan,
            expected_plan_sha256=expected_plan_sha256,
            expected_seal_sha256=expected_seal_sha256,
            project_root=project_root,
        )
        if snapshot_audit != original_audit:
            _integrity_fail("submission snapshot audit differs from the initial audit")
        if _strict_audit_tree_rows(snapshot) != snapshot_rows:
            _integrity_fail("private submission snapshot changed after derivation")
        final_snapshot_audit = _audit_aamas_submission_snapshot(
            snapshot,
            expected_plan=expected_plan,
            expected_plan_sha256=expected_plan_sha256,
            expected_seal_sha256=expected_seal_sha256,
            project_root=project_root,
        )
        if (
            final_snapshot_audit != snapshot_audit
            or _strict_audit_tree_rows(snapshot) != snapshot_rows
        ):
            _integrity_fail("private submission snapshot changed during final audit")
        if _strict_audit_tree_rows(directory) != stable_original_rows:
            _integrity_fail("live submission tree changed during private audit")
        return final_snapshot_audit


__all__ = [
    "AAMAS_SUBMISSION_CLAIM_SCOPE",
    "AAMAS_SUBMISSION_MANIFEST_SCHEMA_VERSION",
    "AAMAS_SUBMISSION_PLAN_SCHEMA_VERSION",
    "AAMAS_SUBMISSION_RESULT_SCHEMA_VERSION",
    "AAMAS_SUBMISSION_SEAL_SCHEMA_VERSION",
    "AAMASSubmissionAttempt",
    "AAMASSubmissionAuditReport",
    "AAMASSubmissionError",
    "AAMASSubmissionIntegrityError",
    "AAMASSubmissionPlan",
    "AAMASSubmissionRun",
    "AAMASSubmissionTiming",
    "AAMASSubmissionWriteResult",
    "SubmissionAttemptStatus",
    "audit_aamas_submission",
    "run_aamas_submission",
    "write_aamas_submission",
]
