"""Auditable synthetic calibration collection for the AAMAS benchmark.

This module is intentionally narrower than a real-data calibration study.  It
constructs prior-predictive synthetic truths without hand-selected parameters,
runs every preregistered attempt, seals each ordinary calibration bundle, and
only derives collection summaries after replay-verifying every expected bundle.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import stat
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import NoReturn

from policytwin.calibration.artifacts import (
    CALLER_SUPPLIED_GIT_IDENTITY_ATTESTATION,
    LOCAL_GIT_IDENTITY_ATTESTATION,
    CalibrationArtifactIntegrityError,
    load_calibration_artifact,
    write_calibration_artifact,
)
from policytwin.calibration.estimation import sample_prior
from policytwin.calibration.protocol import issue_synthetic_calibration_authorization
from policytwin.calibration.schema import CalibrationConfig
from policytwin.calibration.synthetic import (
    SyntheticMarketSimulator,
    generate_synthetic_observation,
    synthetic_calibration_config,
)
from policytwin.calibration.workflow import CalibrationRun, run_synthetic_calibration
from policytwin.evaluation.selective import (
    SelectionRule,
    SelectivePredictionError,
    SelectivePredictionEvaluation,
    SelectivePredictionRecord,
    SyntheticThresholdSourceReceipt,
    ThresholdSource,
    evaluate_selective_predictions,
)
from policytwin.provenance import (
    GitMetadata,
    canonical_json_bytes,
    read_json_bytes_object,
    sha256_bytes,
)

_CLAIM_SCOPE = "synthetic_aamas_calibration_only"
_TRUTH_NAMESPACE = "aamas_truth_v1"
_ROOT_FILES = frozenset(
    {
        "attempt_registry.json",
        "expected_attempt_registry.json",
        "results.json",
        "seal.json",
        "spec.json",
        "truth_manifest.json",
    }
)


class AAMASCalibrationError(RuntimeError):
    """Raised when the frozen collection design or execution is invalid."""


class AAMASCalibrationIntegrityError(AAMASCalibrationError):
    """Raised when collection bytes cannot be replay-verified."""


@dataclass(frozen=True, slots=True)
class AAMASCalibrationFailureReceipt:
    """Terminal standalone collection-writer failure evidence."""

    destination_name: str
    stage: str
    error_type: str
    error_message: str
    git_identity_attestation: str
    plan_sha256: str
    schema_version: int = 1
    status: str = "failed"
    synthetic_only: bool = True
    success_commitment: None = None

    def __post_init__(self) -> None:
        if (
            not self.destination_name
            or "/" in self.destination_name
            or "\\" in self.destination_name
            or not self.stage
            or not self.error_type
            or not self.error_message
        ):
            _integrity_fail("collection failure receipt fields are invalid")
        if self.git_identity_attestation not in {
            CALLER_SUPPLIED_GIT_IDENTITY_ATTESTATION,
            LOCAL_GIT_IDENTITY_ATTESTATION,
        }:
            _integrity_fail("collection failure Git attestation is invalid")
        _sha256(self.plan_sha256, "failure receipt plan_sha256")
        if (
            self.schema_version != 1
            or self.status != "failed"
            or self.synthetic_only is not True
            or self.success_commitment is not None
        ):
            _integrity_fail("collection failure receipt claims success")

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def to_mapping(self) -> dict[str, object]:
        return {
            "destination_name": self.destination_name,
            "error_message": self.error_message,
            "error_type": self.error_type,
            "git_identity_attestation": self.git_identity_attestation,
            "plan_sha256": self.plan_sha256,
            "schema_version": self.schema_version,
            "stage": self.stage,
            "status": self.status,
            "success_commitment": self.success_commitment,
            "synthetic_only": self.synthetic_only,
        }


class AAMASCalibrationWriteError(AAMASCalibrationIntegrityError):
    """Writer error carrying an explicit non-success terminal receipt."""

    def __init__(
        self,
        message: str,
        *,
        failure_receipt: AAMASCalibrationFailureReceipt,
        failure_receipt_path: Path | None,
    ) -> None:
        super().__init__(message)
        self.failure_receipt = failure_receipt
        self.failure_receipt_path = failure_receipt_path


def _fail(message: str) -> NoReturn:
    raise AAMASCalibrationError(message)


def _integrity_fail(message: str) -> NoReturn:
    raise AAMASCalibrationIntegrityError(message)


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _fail(f"{field} must be a positive integer")
    return value


def _nonempty(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 512:
        _fail(f"{field} must be a non-empty bounded string")
    return value


def _aware_utc(value: object, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        _fail(f"{field} must be a timezone-aware datetime")
    return value.astimezone(UTC)


def _sha256(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _integrity_fail(f"{field} must be a lowercase SHA-256")
    return value


def _mapping_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _integrity_fail(f"{field} must be an integer")
    return value


def _mapping_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        _integrity_fail(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        _integrity_fail(f"{field} must be finite")
    return result


def _mapping_string(value: object, field: str) -> str:
    if not isinstance(value, str):
        _integrity_fail(f"{field} must be a string")
    return value


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
    ).encode()


def _write_new_file(path: Path, payload: bytes) -> None:
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
                _integrity_fail(f"cannot complete collection write: {path.name}")
            remaining = remaining[written:]
        os.fsync(descriptor)
    except OSError as error:
        raise AAMASCalibrationIntegrityError(
            f"cannot create collection artifact {path.name}: {error}"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _git_identity_attestation(git_metadata: GitMetadata | None) -> str:
    return (
        LOCAL_GIT_IDENTITY_ATTESTATION
        if git_metadata is None
        else CALLER_SUPPLIED_GIT_IDENTITY_ATTESTATION
    )


def _remove_writer_tree(path: Path) -> None:
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return
    except OSError as error:
        raise AAMASCalibrationIntegrityError(
            "cannot inspect collection writer staging for cleanup"
        ) from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        _integrity_fail("collection writer cleanup target must be a real directory")
    try:
        shutil.rmtree(path)
    except OSError as error:
        raise AAMASCalibrationIntegrityError(
            "cannot remove collection writer staging after failure"
        ) from error


def _publish_collection_failure_receipt(
    directory: Path,
    receipt: AAMASCalibrationFailureReceipt,
) -> Path | None:
    parent = directory.parent
    failure_path = parent / f"{directory.name}.failure.json"
    if failure_path.exists() or failure_path.is_symlink():
        return None
    payload = canonical_json_bytes(receipt.to_mapping())
    descriptor: int | None = None
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{directory.name}.failure-staging-",
            dir=parent,
        )
        temporary_path = Path(temporary_name)
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                return None
            remaining = remaining[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        if failure_path.exists() or failure_path.is_symlink():
            return None
        os.replace(temporary_path, failure_path)
        temporary_path = None
        parent_descriptor = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
        return failure_path
    except OSError:
        return None
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_path is not None:
            with suppress(OSError):
                temporary_path.unlink(missing_ok=True)


class TruthRole(StrEnum):
    """Preregistered use of one independently keyed synthetic truth."""

    DESIGN = "design"
    TARGET = "target"


class AttemptStatus(StrEnum):
    """Terminal status retained for every expected calibration attempt."""

    COMPLETED = "completed"
    FAILED = "failed"


class CollectionStatus(StrEnum):
    """Whether all expected rows support the frozen aggregate."""

    COMPLETE = "complete"
    INCOMPLETE = "incomplete"


@dataclass(frozen=True, slots=True)
class AAMASCalibrationSpec:
    """Immutable calibration-collection design; defaults are the paper scale."""

    schema_version: int = 1
    study_id: str = "aamas-synthetic-calibration-v1"
    master_seed: str = "aamas-synthetic-calibration-master-v1"
    design_truth_count: int = 32
    target_truth_count: int = 256
    simulation_count: int = 350
    accepted_particles: int = 30
    posterior_draws: int = 160
    ppc_draws: int = 40
    sbc_replicates: int = 1
    sbc_posterior_draws: int = 40
    simulator_noise_scale: float = 0.002
    uncertainty_thresholds: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    minimum_retained: int = 64
    thresholds_frozen_at: datetime = datetime(2024, 12, 1, tzinfo=UTC)
    target_outcomes_first_available_at: datetime = datetime(2025, 1, 2, tzinfo=UTC)
    synthetic_only: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            _fail("AAMAS calibration spec schema_version must equal one")
        _nonempty(self.study_id, "study_id")
        _nonempty(self.master_seed, "master_seed")
        for field in (
            "design_truth_count",
            "target_truth_count",
            "simulation_count",
            "accepted_particles",
            "posterior_draws",
            "ppc_draws",
            "sbc_replicates",
            "sbc_posterior_draws",
            "minimum_retained",
        ):
            _positive_int(getattr(self, field), field)
        if self.minimum_retained > self.target_truth_count:
            _fail("minimum_retained cannot exceed the target truth count")
        if self.minimum_retained * 4 < self.target_truth_count:
            _fail("minimum_retained must cover at least 25% of target truths")
        if self.accepted_particles >= self.simulation_count:
            _fail("accepted_particles must be smaller than simulation_count")
        if self.accepted_particles < 2:
            _fail("calibration must retain a posterior distribution")
        if self.posterior_draws < 2 or self.ppc_draws < 2:
            _fail("posterior and PPC require at least two draws")
        if self.sbc_posterior_draws < 2:
            _fail("SBC requires at least two posterior draws")
        if isinstance(self.simulator_noise_scale, bool) or not isinstance(
            self.simulator_noise_scale, int | float
        ):
            _fail("simulator_noise_scale must be a finite real number")
        noise = float(self.simulator_noise_scale)
        if not math.isfinite(noise):
            _fail("simulator_noise_scale must be finite")
        if not 0.0 <= noise <= 1.0:
            _fail("simulator_noise_scale must lie in [0, 1]")
        object.__setattr__(self, "simulator_noise_scale", noise)
        if self.uncertainty_thresholds != (0.25, 0.5, 0.75, 1.0):
            _fail("AAMAS risk-coverage thresholds are frozen at (0.25,0.5,0.75,1.0)")
        frozen_at = _aware_utc(self.thresholds_frozen_at, "thresholds_frozen_at")
        available_at = _aware_utc(
            self.target_outcomes_first_available_at,
            "target_outcomes_first_available_at",
        )
        if frozen_at >= available_at:
            _fail("risk-coverage thresholds must predate target outcomes")
        if self.synthetic_only is not True:
            _fail("the AAMAS calibration collection is synthetic-only")
        object.__setattr__(self, "thresholds_frozen_at", frozen_at)
        object.__setattr__(self, "target_outcomes_first_available_at", available_at)

    def to_mapping(self) -> dict[str, object]:
        """Return the complete frozen design."""

        return {
            "accepted_particles": self.accepted_particles,
            "claim_scope": _CLAIM_SCOPE,
            "design_truth_count": self.design_truth_count,
            "master_seed": self.master_seed,
            "minimum_retained": self.minimum_retained,
            "posterior_draws": self.posterior_draws,
            "ppc_draws": self.ppc_draws,
            "real_policy_claim": False,
            "sbc_posterior_draws": self.sbc_posterior_draws,
            "sbc_replicates": self.sbc_replicates,
            "schema_version": self.schema_version,
            "simulation_count": self.simulation_count,
            "simulator_noise_scale": self.simulator_noise_scale,
            "study_id": self.study_id,
            "synthetic_only": self.synthetic_only,
            "target_outcomes_first_available_at": (
                self.target_outcomes_first_available_at.isoformat()
            ),
            "target_truth_count": self.target_truth_count,
            "thresholds_frozen_at": self.thresholds_frozen_at.isoformat(),
            "uncertainty_thresholds": list(self.uncertainty_thresholds),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> AAMASCalibrationSpec:
        """Load an exact spec, rejecting added or omitted design fields."""

        expected = frozenset(cls().to_mapping())
        if frozenset(value) != expected:
            _integrity_fail("AAMAS calibration spec fields changed")
        if value["claim_scope"] != _CLAIM_SCOPE:
            _integrity_fail("AAMAS calibration claim scope changed")
        if value["real_policy_claim"] is not False:
            _integrity_fail("synthetic collection cannot authorize a policy claim")
        thresholds = value["uncertainty_thresholds"]
        if not isinstance(thresholds, list):
            _integrity_fail("uncertainty_thresholds must be an array")
        try:
            frozen_at = datetime.fromisoformat(
                _mapping_string(value["thresholds_frozen_at"], "thresholds_frozen_at")
            )
            available_at = datetime.fromisoformat(
                _mapping_string(
                    value["target_outcomes_first_available_at"],
                    "target_outcomes_first_available_at",
                )
            )
            return cls(
                schema_version=_mapping_int(value["schema_version"], "schema_version"),
                study_id=_mapping_string(value["study_id"], "study_id"),
                master_seed=_mapping_string(value["master_seed"], "master_seed"),
                design_truth_count=_mapping_int(
                    value["design_truth_count"], "design_truth_count"
                ),
                target_truth_count=_mapping_int(
                    value["target_truth_count"], "target_truth_count"
                ),
                simulation_count=_mapping_int(
                    value["simulation_count"], "simulation_count"
                ),
                accepted_particles=_mapping_int(
                    value["accepted_particles"], "accepted_particles"
                ),
                posterior_draws=_mapping_int(
                    value["posterior_draws"], "posterior_draws"
                ),
                ppc_draws=_mapping_int(value["ppc_draws"], "ppc_draws"),
                sbc_replicates=_mapping_int(value["sbc_replicates"], "sbc_replicates"),
                sbc_posterior_draws=_mapping_int(
                    value["sbc_posterior_draws"], "sbc_posterior_draws"
                ),
                simulator_noise_scale=_mapping_float(
                    value["simulator_noise_scale"], "simulator_noise_scale"
                ),
                uncertainty_thresholds=tuple(
                    _mapping_float(item, "uncertainty threshold") for item in thresholds
                ),
                minimum_retained=_mapping_int(
                    value["minimum_retained"], "minimum_retained"
                ),
                thresholds_frozen_at=frozen_at,
                target_outcomes_first_available_at=available_at,
                synthetic_only=value["synthetic_only"] is True,
            )
        except (TypeError, ValueError) as error:
            raise AAMASCalibrationIntegrityError(
                f"invalid AAMAS calibration spec: {error}"
            ) from error

    @property
    def sha256(self) -> str:
        """Return the complete design commitment."""

        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    @property
    def execution_work(self) -> int:
        """Return a conservative deterministic work score for execution guards."""

        truth_count = self.design_truth_count + self.target_truth_count
        per_truth = (
            self.simulation_count
            + self.posterior_draws
            + self.ppc_draws
            + self.sbc_replicates * (self.simulation_count + self.sbc_posterior_draws)
        )
        return truth_count * per_truth

    @property
    def is_formal_scale(self) -> bool:
        """Whether this is the exact, intentionally unexecuted paper default."""

        return self == AAMASCalibrationSpec()


@dataclass(frozen=True, slots=True)
class AAMASTruth:
    """One prior-generated truth and every EventKey used to generate it."""

    truth_id: str
    role: TruthRole
    ordinal: int
    role_ordinal: int
    parameters: tuple[tuple[str, float], ...]
    event_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        _nonempty(self.truth_id, "truth_id")
        if not isinstance(self.role, TruthRole):
            _fail("truth role is invalid")
        if (
            isinstance(self.ordinal, bool)
            or not isinstance(self.ordinal, int)
            or isinstance(self.role_ordinal, bool)
            or not isinstance(self.role_ordinal, int)
            or self.ordinal < 0
            or self.role_ordinal < 0
        ):
            _fail("truth ordinals must be non-negative")
        if not isinstance(self.parameters, tuple) or not self.parameters:
            _fail("truth parameters must be a non-empty immutable tuple")
        for item in self.parameters:
            if (
                not isinstance(item, tuple)
                or len(item) != 2
                or not isinstance(item[0], str)
                or not item[0]
                or isinstance(item[1], bool)
                or not isinstance(item[1], int | float)
                or not math.isfinite(float(item[1]))
            ):
                _fail("truth parameters must contain named finite real values")
        if tuple(sorted(self.parameters)) != self.parameters:
            _fail("truth parameters must be sorted and immutable")
        if len(dict(self.parameters)) != len(self.parameters):
            _fail("truth parameters must be non-empty and unique")
        if (
            not isinstance(self.event_keys, tuple)
            or not self.event_keys
            or any(
                not isinstance(key, str)
                or len(key) != 69
                or not key.startswith("ptk1:")
                or any(character not in "0123456789abcdef" for character in key[5:])
                for key in self.event_keys
            )
            or len(set(self.event_keys)) != len(self.event_keys)
        ):
            _fail("truth EventKeys must be non-empty and unique")

    def to_mapping(self) -> dict[str, object]:
        return {
            "event_keys": list(self.event_keys),
            "ordinal": self.ordinal,
            "parameters": dict(self.parameters),
            "role": self.role.value,
            "role_ordinal": self.role_ordinal,
            "truth_id": self.truth_id,
        }

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))


@dataclass(frozen=True, slots=True)
class AAMASExpectedAttempt:
    """One preregistered calibration attempt, before any observation exists."""

    attempt_id: str
    truth_id: str
    role: TruthRole
    config_sha256: str
    truth_sha256: str
    bundle_path: str
    execution_id: str

    def to_mapping(self) -> dict[str, object]:
        return {
            "attempt_id": self.attempt_id,
            "bundle_path": self.bundle_path,
            "config_sha256": self.config_sha256,
            "execution_id": self.execution_id,
            "role": self.role.value,
            "truth_id": self.truth_id,
            "truth_sha256": self.truth_sha256,
        }


@dataclass(frozen=True, slots=True)
class AAMASCalibrationPlan:
    """Complete truth manifest and expected attempt registry."""

    spec: AAMASCalibrationSpec
    truths: tuple[AAMASTruth, ...]
    expected_attempts: tuple[AAMASExpectedAttempt, ...]

    def __post_init__(self) -> None:
        expected_count = self.spec.design_truth_count + self.spec.target_truth_count
        if (
            len(self.truths) != expected_count
            or len(self.expected_attempts) != expected_count
        ):
            _fail("truth manifest and expected attempt registry are incomplete")
        expected_roles = (TruthRole.DESIGN,) * self.spec.design_truth_count + (
            TruthRole.TARGET,
        ) * self.spec.target_truth_count
        if tuple(item.role for item in self.truths) != expected_roles:
            _fail("truth roles or their frozen design/target ordering changed")
        if tuple(item.ordinal for item in self.truths) != tuple(range(expected_count)):
            _fail("truth global ordinals must be complete and ordered")
        for role in TruthRole:
            role_truths = tuple(item for item in self.truths if item.role is role)
            if tuple(item.role_ordinal for item in role_truths) != tuple(
                range(len(role_truths))
            ):
                _fail("truth role ordinals must be complete and ordered")
        truth_ids = tuple(item.truth_id for item in self.truths)
        attempt_ids = tuple(item.attempt_id for item in self.expected_attempts)
        if len(set(truth_ids)) != len(truth_ids):
            _fail("truth manifest contains duplicate truth IDs")
        if len(set(attempt_ids)) != len(attempt_ids):
            _fail("expected attempt registry contains duplicate attempt IDs")
        if {item.truth_id for item in self.expected_attempts} != set(truth_ids):
            _fail("expected attempt registry does not cover every truth exactly")
        all_keys = tuple(key for truth in self.truths for key in truth.event_keys)
        if len(set(all_keys)) != len(all_keys):
            _fail("truth generation reused an EventKey")
        for truth, attempt in zip(self.truths, self.expected_attempts, strict=True):
            attempt_id = f"calibrate-{truth.truth_id}"
            expected = AAMASExpectedAttempt(
                attempt_id=attempt_id,
                truth_id=truth.truth_id,
                role=truth.role,
                config_sha256=_attempt_config(self.spec, truth).sha256,
                truth_sha256=truth.sha256,
                bundle_path=f"bundles/{attempt_id}",
                execution_id=f"aamas-{truth.role.value}-{truth.role_ordinal:03d}",
            )
            if attempt != expected:
                _fail("expected attempt row differs from its deterministic truth plan")

    @property
    def truth_manifest_mapping(self) -> dict[str, object]:
        return {
            "claim_scope": _CLAIM_SCOPE,
            "generator": "sample_prior",
            "generator_config_sha256": _base_config(
                self.spec, seed=self.spec.master_seed
            ).sha256,
            "master_seed_sha256": sha256_bytes(self.spec.master_seed.encode()),
            "namespace": _TRUTH_NAMESPACE,
            "schema_version": 1,
            "truths": [item.to_mapping() for item in self.truths],
        }

    @property
    def expected_attempt_registry_mapping(self) -> dict[str, object]:
        return {
            "attempts": [item.to_mapping() for item in self.expected_attempts],
            "expected_attempt_count": len(self.expected_attempts),
            "schema_version": 1,
            "spec_sha256": self.spec.sha256,
        }

    @property
    def sha256(self) -> str:
        return sha256_bytes(
            canonical_json_bytes(
                {
                    "expected_attempt_registry": self.expected_attempt_registry_mapping,
                    "spec_sha256": self.spec.sha256,
                    "truth_manifest": self.truth_manifest_mapping,
                }
            )
        )


def _base_config(spec: AAMASCalibrationSpec, *, seed: str) -> CalibrationConfig:
    config = synthetic_calibration_config(
        seed=seed,
        simulation_count=spec.simulation_count,
        accepted_particles=spec.accepted_particles,
        posterior_draws=spec.posterior_draws,
        ppc_draws=spec.ppc_draws,
        sbc_replicates=spec.sbc_replicates,
        sbc_posterior_draws=spec.sbc_posterior_draws,
    )
    return replace(config, study_id=spec.study_id)


def _attempt_seed(spec: AAMASCalibrationSpec, truth: AAMASTruth) -> str:
    """Derive inference randomness without any function of the hidden truth value."""

    return "aamas-run-v1:" + sha256_bytes(
        canonical_json_bytes(
            {
                "master_seed": spec.master_seed,
                "role": truth.role.value,
                "role_ordinal": truth.role_ordinal,
                "study_id": spec.study_id,
                "truth_id": truth.truth_id,
            }
        )
    )


def _attempt_config(spec: AAMASCalibrationSpec, truth: AAMASTruth) -> CalibrationConfig:
    return _base_config(spec, seed=_attempt_seed(spec, truth))


def build_aamas_calibration_plan(
    spec: AAMASCalibrationSpec | None = None,
) -> AAMASCalibrationPlan:
    """Generate all truths once from the fixed master seed without filtering."""

    spec = AAMASCalibrationSpec() if spec is None else spec
    if not isinstance(spec, AAMASCalibrationSpec):
        _fail("spec must be an AAMASCalibrationSpec")
    generator_config = _base_config(spec, seed=spec.master_seed)
    truths: list[AAMASTruth] = []
    role_counts = (
        (TruthRole.DESIGN, spec.design_truth_count),
        (TruthRole.TARGET, spec.target_truth_count),
    )
    ordinal = 0
    for role, count in role_counts:
        for role_ordinal in range(count):
            parameters, event_keys = sample_prior(
                generator_config,
                replicate=ordinal,
                namespace=_TRUTH_NAMESPACE,
                seed=spec.master_seed,
            )
            truths.append(
                AAMASTruth(
                    truth_id=f"{role.value}-{role_ordinal:03d}",
                    role=role,
                    ordinal=ordinal,
                    role_ordinal=role_ordinal,
                    parameters=tuple(sorted(parameters.items())),
                    event_keys=event_keys,
                )
            )
            ordinal += 1
    attempts: list[AAMASExpectedAttempt] = []
    for truth in truths:
        attempt_id = f"calibrate-{truth.truth_id}"
        attempts.append(
            AAMASExpectedAttempt(
                attempt_id=attempt_id,
                truth_id=truth.truth_id,
                role=truth.role,
                config_sha256=_attempt_config(spec, truth).sha256,
                truth_sha256=truth.sha256,
                bundle_path=f"bundles/{attempt_id}",
                execution_id=f"aamas-{truth.role.value}-{truth.role_ordinal:03d}",
            )
        )
    return AAMASCalibrationPlan(spec, tuple(truths), tuple(attempts))


@dataclass(frozen=True, slots=True)
class AAMASCalibrationAttempt:
    """In-memory terminal attempt, including failures rather than omitting them."""

    expected: AAMASExpectedAttempt
    status: AttemptStatus
    run: CalibrationRun | None
    error_type: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        if self.status is AttemptStatus.COMPLETED:
            if (
                self.run is None
                or self.error_type is not None
                or self.error_message is not None
            ):
                _fail("completed attempts require exactly one CalibrationRun")
            if self.run.config.sha256 != self.expected.config_sha256:
                _fail("completed attempt config differs from its expected registry row")
        elif self.status is AttemptStatus.FAILED:
            if self.run is not None or not self.error_type or not self.error_message:
                _fail("failed attempts require an explicit error and no run")
        else:
            _fail("attempt status is invalid")


@dataclass(frozen=True, slots=True)
class AAMASCalibrationRunCollection:
    """All expected attempts from one deterministic in-memory execution."""

    plan: AAMASCalibrationPlan
    attempts: tuple[AAMASCalibrationAttempt, ...]

    def __post_init__(self) -> None:
        expected_ids = tuple(item.attempt_id for item in self.plan.expected_attempts)
        actual_ids = tuple(item.expected.attempt_id for item in self.attempts)
        if actual_ids != expected_ids:
            _fail("attempt execution registry is missing, duplicated, or reordered")
        if any(
            actual.expected != expected
            for actual, expected in zip(
                self.attempts, self.plan.expected_attempts, strict=True
            )
        ):
            _fail("attempt execution registry drifted from the expected registry")
        truth_by_id = {truth.truth_id: truth for truth in self.plan.truths}
        for attempt in self.attempts:
            if attempt.run is None:
                continue
            truth = truth_by_id[attempt.expected.truth_id]
            if tuple(attempt.run.true_parameters) != truth.parameters:
                _fail("completed attempt truth differs from its frozen truth manifest")
        keys = tuple(key for truth in self.plan.truths for key in truth.event_keys)
        keys += tuple(
            key
            for attempt in self.attempts
            if attempt.run is not None
            for key in attempt.run.event_keys
        )
        if len(keys) != len(set(keys)):
            _fail("AAMAS calibration collection reused an EventKey")

    @property
    def status(self) -> CollectionStatus:
        if all(item.status is AttemptStatus.COMPLETED for item in self.attempts):
            return CollectionStatus.COMPLETE
        return CollectionStatus.INCOMPLETE

    @property
    def scientific_sha256(self) -> str:
        """Hash scientific mappings only, excluding filesystem provenance."""

        return sha256_bytes(
            canonical_json_bytes(
                {
                    "attempts": [
                        {
                            "attempt_id": item.expected.attempt_id,
                            "error_message": item.error_message,
                            "error_type": item.error_type,
                            "report": None
                            if item.run is None
                            else item.run.report.to_mapping(),
                            "status": item.status.value,
                        }
                        for item in self.attempts
                    ],
                    "plan_sha256": self.plan.sha256,
                }
            )
        )


def run_aamas_calibration_collection(
    spec: AAMASCalibrationSpec | None = None,
    *,
    allow_formal_scale: bool = False,
) -> AAMASCalibrationRunCollection:
    """Run every registered synthetic truth; failures remain terminal rows."""

    spec = AAMASCalibrationSpec() if spec is None else spec
    if not isinstance(spec, AAMASCalibrationSpec):
        _fail("spec must be an AAMASCalibrationSpec")
    if not isinstance(allow_formal_scale, bool):
        _fail("allow_formal_scale must be boolean")
    if (
        spec.execution_work >= AAMASCalibrationSpec().execution_work
        and not allow_formal_scale
    ):
        raise AAMASCalibrationIntegrityError(
            "formal AAMAS calibration scale requires explicit allow_formal_scale=True"
        )
    plan = build_aamas_calibration_plan(spec)
    simulator = SyntheticMarketSimulator(noise_scale=spec.simulator_noise_scale)
    truths = {item.truth_id: item for item in plan.truths}
    attempts: list[AAMASCalibrationAttempt] = []
    for expected in plan.expected_attempts:
        truth = truths[expected.truth_id]
        try:
            config = _attempt_config(spec, truth)
            observed = generate_synthetic_observation(
                config, simulator, dict(truth.parameters)
            )
            protocol, authorization, started_at = (
                issue_synthetic_calibration_authorization(
                    config,
                    observed,
                    execution_id=expected.execution_id,
                )
            )
            run = run_synthetic_calibration(
                config=config,
                observed=observed,
                simulator=simulator,
                true_parameters=dict(truth.parameters),
                study_protocol=protocol,
                protocol_authorization=authorization,
                execution_started_at=started_at,
            )
            attempts.append(
                AAMASCalibrationAttempt(expected, AttemptStatus.COMPLETED, run)
            )
        except Exception as error:  # Every expected row must reach a terminal state.
            attempts.append(
                AAMASCalibrationAttempt(
                    expected,
                    AttemptStatus.FAILED,
                    None,
                    type(error).__name__,
                    str(error)[:2048] or "calibration attempt failed without a message",
                )
            )
    return AAMASCalibrationRunCollection(plan, tuple(attempts))


@dataclass(frozen=True, slots=True)
class _FinalAttempt:
    expected: AAMASExpectedAttempt
    status: AttemptStatus
    manifest_raw_sha256: str | None
    report_sha256: str | None
    input_sha256: str | None
    event_keys_sha256: str | None
    error_type: str | None
    error_message: str | None

    def to_mapping(self) -> dict[str, object]:
        return {
            **self.expected.to_mapping(),
            "error_message": self.error_message,
            "error_type": self.error_type,
            "event_keys_sha256": self.event_keys_sha256,
            "input_sha256": self.input_sha256,
            "manifest_raw_sha256": self.manifest_raw_sha256,
            "report_sha256": self.report_sha256,
            "status": self.status.value,
        }


def _target_record(run: CalibrationRun, truth_id: str) -> SelectivePredictionRecord:
    priors = {prior.name: prior for prior in run.config.priors}
    normalized_widths: list[float] = []
    normalized_errors: list[float] = []
    for metric in run.recovery:
        prior = priors[metric.parameter]
        support_width = prior.upper - prior.lower
        normalized_widths.append((metric.upper_90 - metric.lower_90) / support_width)
        normalized_errors.append(metric.absolute_error / support_width)
    return SelectivePredictionRecord(
        prediction_id=truth_id,
        uncertainty_score=sum(normalized_widths) / len(normalized_widths),
        loss=sum(normalized_errors) / len(normalized_errors),
    )


def _truth_result_mapping(
    truth: AAMASTruth,
    expected: AAMASExpectedAttempt,
    run: CalibrationRun,
    manifest_raw_sha256: str,
) -> dict[str, object]:
    return {
        "attempt_id": expected.attempt_id,
        "bundle_manifest_raw_sha256": manifest_raw_sha256,
        "bundle_path": expected.bundle_path,
        "calibration_statistics": list(run.report.calibration_statistics),
        "held_out_statistics": list(run.report.held_out_statistics),
        "identifiability": [item.to_mapping() for item in run.identifiability],
        "posterior_draw_count": len(run.posterior.draws),
        "posterior_draws_retained_in_bundle": True,
        "posterior_sha256": run.posterior.sha256,
        "ppc": [item.to_mapping() for item in run.ppc.statistics],
        "recovery": [item.to_mapping() for item in run.recovery],
        "report_sha256": run.report.sha256,
        "role": truth.role.value,
        "sbc": run.sbc.to_mapping(),
        "truth_id": truth.truth_id,
    }


def _selection_rule(
    spec: AAMASCalibrationSpec,
    *,
    protocol_sha256: str,
) -> SelectionRule:
    receipt = SyntheticThresholdSourceReceipt.issue(
        study_protocol_sha256=protocol_sha256,
        source_artifact_sha256=spec.sha256,
        thresholds_frozen_at=spec.thresholds_frozen_at,
        target_outcomes_first_available_at=spec.target_outcomes_first_available_at,
        uncertainty_thresholds=spec.uncertainty_thresholds,
    )
    return SelectionRule(
        schema_version=2,
        rule_id="aamas_target_calibration_uncertainty_v1",
        study_protocol_sha256=protocol_sha256,
        threshold_source=ThresholdSource.SYNTHETIC_FIXTURE,
        threshold_source_artifact_sha256=spec.sha256,
        thresholds_frozen_at=spec.thresholds_frozen_at,
        target_outcomes_first_available_at=spec.target_outcomes_first_available_at,
        uncertainty_thresholds=spec.uncertainty_thresholds,
        loss_metric_id="mean_normalized_absolute_parameter_error",
        minimum_retained=spec.minimum_retained,
        synthetic_only=True,
        threshold_source_receipt=receipt,
    )


def _derive_results_mapping(
    plan: AAMASCalibrationPlan,
    attempts: tuple[_FinalAttempt, ...],
    runs: Mapping[str, CalibrationRun],
) -> dict[str, object]:
    blocking_reasons: list[str] = []
    if len(attempts) != len(plan.expected_attempts) or any(
        item.status is not AttemptStatus.COMPLETED for item in attempts
    ):
        blocking_reasons.append("not_every_expected_attempt_completed_and_replayed")
    truth_by_id = {item.truth_id: item for item in plan.truths}
    attempt_by_id = {item.expected.attempt_id: item for item in attempts}
    truth_results: list[dict[str, object]] = []
    for expected in plan.expected_attempts:
        final = attempt_by_id.get(expected.attempt_id)
        run = runs.get(expected.attempt_id)
        if (
            final is None
            or final.status is not AttemptStatus.COMPLETED
            or final.manifest_raw_sha256 is None
            or run is None
        ):
            continue
        truth_results.append(
            _truth_result_mapping(
                truth_by_id[expected.truth_id],
                expected,
                run,
                final.manifest_raw_sha256,
            )
        )

    selective_records: tuple[SelectivePredictionRecord, ...] = ()
    rule: SelectionRule | None = None
    evaluation: SelectivePredictionEvaluation | None = None
    selective_status = "blocked_incomplete_collection"
    selective_error: str | None = None
    if not blocking_reasons:
        target_attempts = tuple(
            expected
            for expected in plan.expected_attempts
            if expected.role is TruthRole.TARGET
        )
        selective_records = tuple(
            _target_record(runs[item.attempt_id], item.truth_id)
            for item in target_attempts
        )
        # The selection rule is frozen at collection-design time.  Per-attempt
        # protocol receipts remain in their bundles but are not substituted for
        # this collection-level commitment.
        rule = _selection_rule(spec=plan.spec, protocol_sha256=plan.sha256)
        try:
            evaluation = evaluate_selective_predictions(selective_records, rule)
            selective_status = "complete_target_only"
        except SelectivePredictionError as error:
            selective_status = "blocked_registered_threshold_retained_too_few"
            selective_error = str(error)
            blocking_reasons.append("a_registered_threshold_retained_too_few_targets")

    status = (
        CollectionStatus.COMPLETE
        if not blocking_reasons
        else CollectionStatus.INCOMPLETE
    )
    return {
        "authoritative_validity": False,
        "behavior_nine_parameter_ci_promoted": False,
        "blocking_reasons": blocking_reasons,
        "claim_scope": _CLAIM_SCOPE,
        "collection_status": status.value,
        "complete_case_fallback_used": False,
        "cross_truth_aggregation_status": (
            "complete" if status is CollectionStatus.COMPLETE else "blocked_incomplete"
        ),
        "real_policy_claim": False,
        "risk_coverage": {
            "error": selective_error,
            "evaluation": None if evaluation is None else evaluation.to_mapping(),
            "records": [item.to_mapping() for item in selective_records],
            "role": TruthRole.TARGET.value,
            "rule": None if rule is None else rule.to_mapping(),
            "status": selective_status,
        },
        "schema_version": 1,
        "synthetic_only": True,
        "truth_results": truth_results,
    }


@dataclass(frozen=True, slots=True)
class AAMASCalibrationWriteResult:
    """Caller-retained commitment for one newly written collection."""

    directory: Path
    seal_raw_sha256: str
    collection_status: CollectionStatus
    expected_attempt_count: int
    completed_attempt_count: int
    results_sha256: str


def _final_failed(
    expected: AAMASExpectedAttempt, error_type: str, error_message: str
) -> _FinalAttempt:
    return _FinalAttempt(
        expected=expected,
        status=AttemptStatus.FAILED,
        manifest_raw_sha256=None,
        report_sha256=None,
        input_sha256=None,
        event_keys_sha256=None,
        error_type=error_type,
        error_message=error_message[:2048],
    )


def _write_aamas_calibration_collection_staging(
    directory: Path,
    *,
    collection: AAMASCalibrationRunCollection,
    project_root: Path | None = None,
    git_metadata: GitMetadata | None = None,
) -> AAMASCalibrationWriteResult:
    """Populate one verifier-owned staging directory."""

    if not isinstance(collection, AAMASCalibrationRunCollection):
        _fail("collection must be an AAMASCalibrationRunCollection")
    try:
        (directory / "bundles").mkdir(exist_ok=False)
    except FileExistsError as error:
        raise AAMASCalibrationIntegrityError(
            "AAMAS calibration staging is not empty"
        ) from error

    finalized: list[_FinalAttempt] = []
    loaded_runs: dict[str, CalibrationRun] = {}
    for attempt in collection.attempts:
        expected = attempt.expected
        if attempt.status is AttemptStatus.FAILED:
            assert attempt.error_type is not None and attempt.error_message is not None
            finalized.append(
                _final_failed(expected, attempt.error_type, attempt.error_message)
            )
            continue
        assert attempt.run is not None
        bundle_directory = directory / expected.bundle_path
        try:
            manifest_path = write_calibration_artifact(
                bundle_directory,
                run=attempt.run,
                project_root=project_root,
                git_metadata=git_metadata,
            )
            manifest_digest = sha256_bytes(manifest_path.read_bytes())
            loaded = load_calibration_artifact(
                bundle_directory,
                expected_manifest_sha256=manifest_digest,
            )
        except (CalibrationArtifactIntegrityError, OSError) as error:
            raise AAMASCalibrationIntegrityError(
                f"cannot seal and replay bundle {expected.attempt_id}: {error}"
            ) from error
        if loaded.run.config.sha256 != expected.config_sha256:
            _integrity_fail("replayed bundle config differs from expected attempt")
        loaded_runs[expected.attempt_id] = loaded.run
        finalized.append(
            _FinalAttempt(
                expected=expected,
                status=AttemptStatus.COMPLETED,
                manifest_raw_sha256=manifest_digest,
                report_sha256=loaded.run.report.sha256,
                input_sha256=loaded.run.observed.sha256,
                event_keys_sha256=loaded.run.report.event_keys_sha256,
                error_type=None,
                error_message=None,
            )
        )

    finalized_tuple = tuple(finalized)
    results = _derive_results_mapping(collection.plan, finalized_tuple, loaded_runs)
    payloads = {
        "attempt_registry.json": _pretty_json_bytes(
            {
                "attempts": [item.to_mapping() for item in finalized_tuple],
                "schema_version": 1,
            }
        ),
        "expected_attempt_registry.json": _pretty_json_bytes(
            collection.plan.expected_attempt_registry_mapping
        ),
        "results.json": _pretty_json_bytes(results),
        "spec.json": _pretty_json_bytes(collection.plan.spec.to_mapping()),
        "truth_manifest.json": _pretty_json_bytes(
            collection.plan.truth_manifest_mapping
        ),
    }
    for name, payload in payloads.items():
        _write_new_file(directory / name, payload)
    files = [
        {"path": name, "sha256": sha256_bytes(payload), "size_bytes": len(payload)}
        for name, payload in sorted(payloads.items())
    ]
    bundle_manifests = [
        {
            "attempt_id": item.expected.attempt_id,
            "manifest_raw_sha256": item.manifest_raw_sha256,
            "path": f"{item.expected.bundle_path}/manifest.json",
        }
        for item in finalized_tuple
        if item.status is AttemptStatus.COMPLETED
    ]
    status = CollectionStatus(str(results["collection_status"]))
    seal = {
        "bundle_manifests": bundle_manifests,
        "claim_scope": _CLAIM_SCOPE,
        "collection_status": status.value,
        "expected_attempt_count": len(collection.plan.expected_attempts),
        "files": files,
        "git_identity_attestation": _git_identity_attestation(git_metadata),
        "plan_sha256": collection.plan.sha256,
        "schema_version": 2,
        "spec_sha256": collection.plan.spec.sha256,
        "synthetic_only": True,
    }
    seal_payload = _pretty_json_bytes(seal)
    _write_new_file(directory / "seal.json", seal_payload)
    return AAMASCalibrationWriteResult(
        directory=directory,
        seal_raw_sha256=sha256_bytes(seal_payload),
        collection_status=status,
        expected_attempt_count=len(collection.plan.expected_attempts),
        completed_attempt_count=sum(
            item.status is AttemptStatus.COMPLETED for item in finalized_tuple
        ),
        results_sha256=sha256_bytes(payloads["results.json"]),
    )


def write_aamas_calibration_collection(
    directory: Path,
    *,
    collection: AAMASCalibrationRunCollection,
    project_root: Path | None = None,
    git_metadata: GitMetadata | None = None,
) -> AAMASCalibrationWriteResult:
    """Replay-audit staging, then atomically publish an immutable collection."""

    if not isinstance(collection, AAMASCalibrationRunCollection):
        _fail("collection must be an AAMASCalibrationRunCollection")
    if directory.exists() or directory.is_symlink():
        raise AAMASCalibrationIntegrityError(
            "AAMAS calibration collections are immutable and cannot be overwritten"
        )
    try:
        directory.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise AAMASCalibrationIntegrityError(
            "cannot create AAMAS calibration collection parent"
        ) from error
    staging: Path | None = None
    published = False
    stage = "create_staging"
    attestation = _git_identity_attestation(git_metadata)
    try:
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{directory.name}.staging-",
                dir=directory.parent,
            )
        )
        stage = "write_collection"
        staged_write = _write_aamas_calibration_collection_staging(
            staging,
            collection=collection,
            project_root=project_root,
            git_metadata=git_metadata,
        )
        stage = "audit_staging"
        staged_audit = audit_aamas_calibration_collection(
            staging,
            expected_seal_sha256=staged_write.seal_raw_sha256,
        )
        if (
            not staged_audit.valid
            or staged_audit.git_identity_attestation != attestation
            or staged_audit.results_sha256 != staged_write.results_sha256
        ):
            _integrity_fail("staged collection immediate audit differs from writer")
        stage = "atomic_publish"
        if directory.exists() or directory.is_symlink():
            _integrity_fail(
                "AAMAS calibration collections are immutable and cannot be "
                "overwritten"
            )
        os.replace(staging, directory)
        staging = None
        published = True
        stage = "audit_published"
        final_audit = audit_aamas_calibration_collection(
            directory,
            expected_seal_sha256=staged_write.seal_raw_sha256,
        )
        if final_audit != staged_audit:
            _integrity_fail("published collection differs from audited staging")
        parent_descriptor = os.open(directory.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
        return AAMASCalibrationWriteResult(
            directory=directory,
            seal_raw_sha256=staged_write.seal_raw_sha256,
            collection_status=staged_write.collection_status,
            expected_attempt_count=staged_write.expected_attempt_count,
            completed_attempt_count=staged_write.completed_attempt_count,
            results_sha256=staged_write.results_sha256,
        )
    except Exception as error:
        cleanup_error: Exception | None = None
        cleanup_target = directory if published else staging
        if cleanup_target is not None:
            try:
                _remove_writer_tree(cleanup_target)
            except Exception as caught:
                cleanup_error = caught
        receipt = AAMASCalibrationFailureReceipt(
            destination_name=directory.name,
            stage=stage,
            error_type=type(error).__name__,
            error_message=str(error)[:2048] or "collection write failed",
            git_identity_attestation=attestation,
            plan_sha256=collection.plan.sha256,
        )
        receipt_path = _publish_collection_failure_receipt(directory, receipt)
        raised = AAMASCalibrationWriteError(
            "AAMAS calibration collection write failed with terminal evidence",
            failure_receipt=receipt,
            failure_receipt_path=receipt_path,
        )
        if cleanup_error is not None:
            raise raised from cleanup_error
        raise raised from error


@dataclass(frozen=True, slots=True)
class AAMASCalibrationAuditReport:
    """Positive replay result relative to a caller-retained seal digest."""

    valid: bool
    collection_status: CollectionStatus
    spec_sha256: str
    plan_sha256: str
    expected_attempt_count: int
    completed_attempt_count: int
    bundle_manifest_sha256s: tuple[str, ...]
    results_sha256: str
    git_identity_attestation: str
    claim_scope: str = _CLAIM_SCOPE
    authoritative_validity: bool = False
    real_policy_claim: bool = False


def _json_object(payload: bytes, field: str) -> dict[str, object]:
    try:
        return read_json_bytes_object(payload, source=field)
    except Exception as error:
        raise AAMASCalibrationIntegrityError(f"invalid {field}: {error}") from error


def _verify_root_membership(directory: Path) -> None:
    try:
        names = frozenset(item.name for item in directory.iterdir())
    except OSError as error:
        raise AAMASCalibrationIntegrityError(
            f"cannot enumerate AAMAS calibration collection: {error}"
        ) from error
    expected = _ROOT_FILES | {"bundles"}
    if names != expected:
        _integrity_fail(
            "collection root membership changed: "
            f"missing={sorted(expected - names)!r}, "
            f"unknown={sorted(names - expected)!r}"
        )
    if not (directory / "bundles").is_dir() or (directory / "bundles").is_symlink():
        _integrity_fail("collection bundles entry must be a real directory")


def _rebuild_final_attempts(
    mapping: Mapping[str, object],
    plan: AAMASCalibrationPlan,
) -> tuple[_FinalAttempt, ...]:
    raw_attempts = mapping.get("attempts")
    if mapping.get("schema_version") != 1 or not isinstance(raw_attempts, list):
        _integrity_fail("attempt registry schema changed")
    expected_fields = frozenset(
        {
            "attempt_id",
            "bundle_path",
            "config_sha256",
            "error_message",
            "error_type",
            "event_keys_sha256",
            "execution_id",
            "input_sha256",
            "manifest_raw_sha256",
            "report_sha256",
            "role",
            "status",
            "truth_id",
            "truth_sha256",
        }
    )
    if len(raw_attempts) != len(plan.expected_attempts):
        _integrity_fail("attempt registry omitted or added an expected attempt")
    final: list[_FinalAttempt] = []
    for expected, raw in zip(plan.expected_attempts, raw_attempts, strict=True):
        if not isinstance(raw, dict) or frozenset(raw) != expected_fields:
            _integrity_fail("attempt registry row fields changed")
        expected_mapping = expected.to_mapping()
        if any(raw[field] != value for field, value in expected_mapping.items()):
            _integrity_fail("attempt registry row drifted from expected attempt")
        try:
            status = AttemptStatus(str(raw["status"]))
        except ValueError as error:
            raise AAMASCalibrationIntegrityError("unknown attempt status") from error
        if status is AttemptStatus.COMPLETED:
            final.append(
                _FinalAttempt(
                    expected,
                    status,
                    _sha256(raw["manifest_raw_sha256"], "manifest_raw_sha256"),
                    _sha256(raw["report_sha256"], "report_sha256"),
                    _sha256(raw["input_sha256"], "input_sha256"),
                    _sha256(raw["event_keys_sha256"], "event_keys_sha256"),
                    None,
                    None,
                )
            )
            if raw["error_type"] is not None or raw["error_message"] is not None:
                _integrity_fail("completed attempt contains a failure message")
        else:
            if not isinstance(raw["error_type"], str) or not isinstance(
                raw["error_message"], str
            ):
                _integrity_fail("failed attempt omitted its explicit error")
            if not raw["error_type"] or not raw["error_message"]:
                _integrity_fail("failed attempt error fields must be non-empty")
            if any(
                raw[field] is not None
                for field in (
                    "manifest_raw_sha256",
                    "report_sha256",
                    "input_sha256",
                    "event_keys_sha256",
                )
            ):
                _integrity_fail("failed attempt claims completed bundle commitments")
            final.append(
                _final_failed(
                    expected,
                    str(raw["error_type"]),
                    str(raw["error_message"]),
                )
            )
    return tuple(final)


def _audit_aamas_calibration_collection_snapshot(
    directory: Path,
    *,
    expected_seal_sha256: str,
) -> AAMASCalibrationAuditReport:
    """Rebuild the plan and every aggregate from stable, committed bundles."""

    expected_seal = _sha256(expected_seal_sha256, "expected_seal_sha256")
    _verify_root_membership(directory)
    payloads: dict[str, bytes] = {}
    for name in _ROOT_FILES:
        path = directory / name
        if path.is_symlink() or not path.is_file():
            _integrity_fail(f"collection file is missing or not regular: {name}")
        try:
            payloads[name] = path.read_bytes()
        except OSError as error:
            raise AAMASCalibrationIntegrityError(
                f"cannot read collection file {name}: {error}"
            ) from error
    if sha256_bytes(payloads["seal.json"]) != expected_seal:
        _integrity_fail("collection seal differs from the caller-retained commitment")
    seal = _json_object(payloads["seal.json"], "seal.json")
    if frozenset(seal) != {
        "bundle_manifests",
        "claim_scope",
        "collection_status",
        "expected_attempt_count",
        "files",
        "git_identity_attestation",
        "plan_sha256",
        "schema_version",
        "spec_sha256",
        "synthetic_only",
    }:
        _integrity_fail("collection seal fields changed")
    if (
        seal.get("schema_version") != 2
        or seal.get("claim_scope") != _CLAIM_SCOPE
        or seal.get("synthetic_only") is not True
        or seal.get("git_identity_attestation")
        not in {
            CALLER_SUPPLIED_GIT_IDENTITY_ATTESTATION,
            LOCAL_GIT_IDENTITY_ATTESTATION,
        }
    ):
        _integrity_fail("collection seal authority boundary changed")
    file_rows = seal.get("files")
    if not isinstance(file_rows, list) or len(file_rows) != len(_ROOT_FILES) - 1:
        _integrity_fail("collection seal file registry is incomplete")
    expected_file_names = _ROOT_FILES - {"seal.json"}
    seen_files: set[str] = set()
    for row in file_rows:
        if not isinstance(row, dict) or frozenset(row) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            _integrity_fail("collection seal file row changed")
        name = row["path"]
        if (
            not isinstance(name, str)
            or name not in expected_file_names
            or name in seen_files
        ):
            _integrity_fail("collection seal file membership is invalid")
        seen_files.add(name)
        if row["sha256"] != sha256_bytes(payloads[name]) or row["size_bytes"] != len(
            payloads[name]
        ):
            _integrity_fail(f"collection top-level file was modified: {name}")
    if seen_files != expected_file_names:
        _integrity_fail("collection seal omitted a top-level file")

    spec = AAMASCalibrationSpec.from_mapping(
        _json_object(payloads["spec.json"], "spec.json")
    )
    plan = build_aamas_calibration_plan(spec)
    if (
        _json_object(payloads["truth_manifest.json"], "truth_manifest.json")
        != plan.truth_manifest_mapping
    ):
        _integrity_fail("truth manifest differs from deterministic prior generation")
    if (
        _json_object(
            payloads["expected_attempt_registry.json"], "expected_attempt_registry.json"
        )
        != plan.expected_attempt_registry_mapping
    ):
        _integrity_fail("expected attempt registry differs from the frozen plan")
    if seal.get("spec_sha256") != spec.sha256 or seal.get("plan_sha256") != plan.sha256:
        _integrity_fail("seal plan commitment mismatch")
    if seal.get("expected_attempt_count") != len(plan.expected_attempts):
        _integrity_fail("seal expected attempt count mismatch")

    finalized = _rebuild_final_attempts(
        _json_object(payloads["attempt_registry.json"], "attempt_registry.json"),
        plan,
    )
    bundle_rows = seal.get("bundle_manifests")
    if not isinstance(bundle_rows, list):
        _integrity_fail("seal bundle manifest registry is invalid")
    declared_bundles: dict[str, tuple[str, str]] = {}
    for row in bundle_rows:
        if not isinstance(row, dict) or frozenset(row) != {
            "attempt_id",
            "manifest_raw_sha256",
            "path",
        }:
            _integrity_fail("seal bundle manifest row changed")
        attempt_id = row["attempt_id"]
        path = row["path"]
        digest = row["manifest_raw_sha256"]
        if not isinstance(attempt_id, str) or not isinstance(path, str):
            _integrity_fail("seal bundle identifiers are invalid")
        if attempt_id in declared_bundles:
            _integrity_fail("seal contains a duplicate bundle attempt")
        declared_bundles[attempt_id] = (path, _sha256(digest, "bundle digest"))

    expected_bundle_directories = {
        item.expected.attempt_id
        for item in finalized
        if item.status is AttemptStatus.COMPLETED
    }
    try:
        actual_bundle_directories = {
            item.name for item in (directory / "bundles").iterdir()
        }
    except OSError as error:
        raise AAMASCalibrationIntegrityError(
            f"cannot enumerate calibration bundles: {error}"
        ) from error
    expected_directory_names = expected_bundle_directories
    if actual_bundle_directories != expected_directory_names:
        _integrity_fail("bundle directory membership differs from completed attempts")
    if set(declared_bundles) != expected_bundle_directories:
        _integrity_fail("seal bundle registry differs from completed attempts")

    loaded_runs: dict[str, CalibrationRun] = {}
    truth_by_id = {item.truth_id: item for item in plan.truths}
    all_keys = [key for truth in plan.truths for key in truth.event_keys]
    manifest_digests: list[str] = []
    for final in finalized:
        if final.status is not AttemptStatus.COMPLETED:
            continue
        expected = final.expected
        assert final.manifest_raw_sha256 is not None
        declared_path, declared_digest = declared_bundles[expected.attempt_id]
        expected_manifest_path = f"{expected.bundle_path}/manifest.json"
        if (
            declared_path != expected_manifest_path
            or declared_digest != final.manifest_raw_sha256
        ):
            _integrity_fail("bundle registry path or digest mismatch")
        bundle_directory = directory / expected.bundle_path
        try:
            loaded = load_calibration_artifact(
                bundle_directory,
                expected_manifest_sha256=declared_digest,
            )
        except CalibrationArtifactIntegrityError as error:
            raise AAMASCalibrationIntegrityError(
                f"calibration bundle failed replay: {expected.attempt_id}: {error}"
            ) from error
        run = loaded.run
        if (
            loaded.audit.git_identity_attestation
            != seal["git_identity_attestation"]
        ):
            _integrity_fail(
                "bundle Git identity attestation differs from collection boundary"
            )
        truth = truth_by_id[expected.truth_id]
        if run.config.sha256 != expected.config_sha256:
            _integrity_fail("replayed run config differs from expected registry")
        if tuple(run.true_parameters) != truth.parameters:
            _integrity_fail("replayed run truth differs from truth manifest")
        if (
            run.report.sha256 != final.report_sha256
            or run.observed.sha256 != final.input_sha256
            or run.report.event_keys_sha256 != final.event_keys_sha256
        ):
            _integrity_fail("replayed run commitments differ from attempt registry")
        if set(run.report.calibration_statistics) & set(run.report.held_out_statistics):
            _integrity_fail("calibration and held-out statistics overlap")
        loaded_runs[expected.attempt_id] = run
        all_keys.extend(run.event_keys)
        manifest_digests.append(declared_digest)
    if len(all_keys) != len(set(all_keys)):
        _integrity_fail("collection replay found reused EventKeys")

    expected_results = _derive_results_mapping(plan, finalized, loaded_runs)
    actual_results = _json_object(payloads["results.json"], "results.json")
    if actual_results != expected_results:
        _integrity_fail("collection results do not match complete replayed attempts")
    try:
        status = CollectionStatus(str(actual_results["collection_status"]))
    except ValueError as error:
        raise AAMASCalibrationIntegrityError("unknown collection status") from error
    if seal.get("collection_status") != status.value:
        _integrity_fail("seal collection status mismatch")
    return AAMASCalibrationAuditReport(
        valid=True,
        collection_status=status,
        spec_sha256=spec.sha256,
        plan_sha256=plan.sha256,
        expected_attempt_count=len(plan.expected_attempts),
        completed_attempt_count=len(loaded_runs),
        bundle_manifest_sha256s=tuple(manifest_digests),
        results_sha256=sha256_bytes(payloads["results.json"]),
        git_identity_attestation=str(seal["git_identity_attestation"]),
    )


def _collection_tree_rows(directory: Path) -> tuple[dict[str, object], ...]:
    """Capture a bounded, race-checked semantic identity of the complete tree."""

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
        raise AAMASCalibrationIntegrityError(
            f"cannot inspect collection audit root: {error}"
        ) from error
    if (
        not stat.S_ISDIR(root_before.st_mode)
        or stat.S_ISLNK(root_before.st_mode)
        or bool(root_before.st_mode & 0o7000)
    ):
        _integrity_fail("collection audit root must be a safe real directory")
    try:
        initial_paths = tuple(
            sorted(
                directory.rglob("*"),
                key=lambda item: item.relative_to(directory).as_posix(),
            )
        )
    except OSError as error:
        raise AAMASCalibrationIntegrityError(
            f"cannot enumerate collection audit tree: {error}"
        ) from error
    if len(initial_paths) > 100_000:
        _integrity_fail("collection audit tree exceeds path-count limit")
    rows: list[dict[str, object]] = []
    total_bytes = 0
    for path in initial_paths:
        relative = path.relative_to(directory).as_posix()
        pure = PurePosixPath(relative)
        if (
            pure.is_absolute()
            or ".." in pure.parts
            or pure.as_posix() != relative
            or "\\" in relative
            or any(ord(character) < 32 for character in relative)
        ):
            _integrity_fail("collection audit tree contains an unsafe path")
        try:
            before = path.lstat()
        except OSError as error:
            raise AAMASCalibrationIntegrityError(
                f"cannot inspect collection audit member {relative}: {error}"
            ) from error
        if stat.S_ISLNK(before.st_mode) or bool(before.st_mode & 0o7000):
            _integrity_fail("collection audit tree contains a linked or unsafe member")
        if stat.S_ISDIR(before.st_mode):
            row: dict[str, object] = {
                "kind": "directory",
                "path": relative,
            }
        elif stat.S_ISREG(before.st_mode) and before.st_nlink == 1:
            try:
                content = path.read_bytes()
            except OSError as error:
                raise AAMASCalibrationIntegrityError(
                    f"cannot read collection audit member {relative}: {error}"
                ) from error
            total_bytes += len(content)
            if total_bytes > 2 * 1024 * 1024 * 1024:
                _integrity_fail("collection audit tree exceeds byte limit")
            if len(content) != before.st_size:
                _integrity_fail("collection audit member changed while read")
            row = {
                "kind": "file",
                "path": relative,
                "sha256": sha256_bytes(content),
                "size_bytes": len(content),
            }
        else:
            _integrity_fail(
                "collection members must be directories or singly linked files"
            )
        try:
            after = path.lstat()
        except OSError as error:
            raise AAMASCalibrationIntegrityError(
                f"cannot recheck collection audit member {relative}: {error}"
            ) from error
        if fingerprint(before) != fingerprint(after):
            _integrity_fail("collection audit member changed while captured")
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
        raise AAMASCalibrationIntegrityError(
            f"cannot finalize collection audit tree: {error}"
        ) from error
    initial_names = tuple(
        path.relative_to(directory).as_posix() for path in initial_paths
    )
    if (
        fingerprint(root_before) != fingerprint(root_after)
        or initial_names != final_paths
    ):
        _integrity_fail("collection audit tree changed while captured")
    return tuple(rows)


def _collection_path_identity(directory: Path) -> tuple[tuple[object, ...], ...]:
    """Capture live inode identities separately from copy-portable semantic rows."""

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
        paths = (directory, *sorted(directory.rglob("*")))
        rows = tuple(
            (
                "." if path == directory else path.relative_to(directory).as_posix(),
                *fingerprint(path.lstat()),
            )
            for path in paths
        )
        final_names = (
            ".",
            *tuple(
                sorted(
                    path.relative_to(directory).as_posix()
                    for path in directory.rglob("*")
                )
            ),
        )
    except OSError as error:
        raise AAMASCalibrationIntegrityError(
            f"cannot capture collection path identity: {error}"
        ) from error
    if tuple(str(row[0]) for row in rows) != final_names:
        _integrity_fail("collection path identity changed while captured")
    return rows


def audit_aamas_calibration_collection(
    directory: Path,
    *,
    expected_seal_sha256: str,
) -> AAMASCalibrationAuditReport:
    """Replay only a verifier-owned snapshot and prove both trees stayed stable."""

    expected_seal = _sha256(expected_seal_sha256, "expected_seal_sha256")
    live_path_identity = _collection_path_identity(directory)
    live_rows = _collection_tree_rows(directory)
    with tempfile.TemporaryDirectory(
        prefix="policytwin-aamas-calibration-audit-"
    ) as temporary:
        snapshot = Path(temporary) / "collection"
        try:
            shutil.copytree(directory, snapshot, symlinks=True)
        except OSError as error:
            raise AAMASCalibrationIntegrityError(
                "cannot create private calibration collection snapshot"
            ) from error
        snapshot_rows = _collection_tree_rows(snapshot)
        if snapshot_rows != live_rows:
            _integrity_fail(
                "collection tree changed while its private snapshot was created"
            )
        snapshot_path_identity = _collection_path_identity(snapshot)
        report = _audit_aamas_calibration_collection_snapshot(
            snapshot,
            expected_seal_sha256=expected_seal,
        )
        if (
            _collection_tree_rows(snapshot) != snapshot_rows
            or _collection_path_identity(snapshot) != snapshot_path_identity
        ):
            _integrity_fail("private collection snapshot changed during replay")
        if (
            _collection_tree_rows(directory) != live_rows
            or _collection_path_identity(directory) != live_path_identity
        ):
            _integrity_fail("live collection tree changed during private audit")
        return report


__all__ = [
    "AAMASCalibrationAttempt",
    "AAMASCalibrationAuditReport",
    "AAMASCalibrationError",
    "AAMASCalibrationFailureReceipt",
    "AAMASExpectedAttempt",
    "AAMASCalibrationIntegrityError",
    "AAMASCalibrationPlan",
    "AAMASCalibrationRunCollection",
    "AAMASCalibrationSpec",
    "AAMASCalibrationWriteResult",
    "AAMASCalibrationWriteError",
    "AAMASTruth",
    "AttemptStatus",
    "CollectionStatus",
    "TruthRole",
    "audit_aamas_calibration_collection",
    "build_aamas_calibration_plan",
    "run_aamas_calibration_collection",
    "write_aamas_calibration_collection",
]
