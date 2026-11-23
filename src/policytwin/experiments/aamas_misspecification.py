"""Preregistered synthetic structural-misspecification negative control.

The misspecified data-generating process (DGP) adds the same frozen nonlinear
mechanism to both calibration and held-out trajectories.  Rejection ABC is
always fitted with the unmodified ``SyntheticMarketSimulator``, so the fitted
family cannot reproduce that mechanism.  Paired cells share all base EventKey
noise while their observations legitimately differ.  The resulting evidence
is only a synthetic test of misspecification diagnostics; it cannot authorize
real-market, policy, or causal claims.
"""

from __future__ import annotations

import json
import math
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import NoReturn, Self, cast

from policytwin.calibration.diagnostics import (
    ParameterIdentifiability,
    PosteriorPredictiveChecks,
    RecoveryMetric,
    identifiability_diagnostics,
    posterior_predictive_checks,
    synthetic_recovery,
)
from policytwin.calibration.estimation import (
    ABCPosteriorEstimator,
    EstimatorFit,
    calibration_event_key,
    sample_posterior,
    sample_prior,
)
from policytwin.calibration.schema import (
    CalibrationConfig,
    CalibrationInput,
    PosteriorSamples,
    SummaryRole,
    SummaryVector,
)
from policytwin.calibration.synthetic import (
    SyntheticMarketSimulator,
    synthetic_calibration_config,
    synthetic_calibration_input,
)
from policytwin.counterfactual import EventKey
from policytwin.provenance import (
    MetadataValidationError,
    canonical_json_bytes,
    read_json_bytes_object,
    sha256_bytes,
)

MISSPECIFICATION_SCHEMA_VERSION = "policytwin.aamas-misspecification.v2"
MISSPECIFICATION_CLAIM_SCOPE = (
    "synthetic_paired_dgp_misspecification_detection_only"
)
TRUTH_NAMESPACE = "aamas_misspecification_truth_v2"

_FORMULA_ID = "calibration_and_heldout_omitted_nonlinearity_v2"
_FORMULA_EXPRESSION = (
    "delta_s=strength*multiplier_s*(0.75+"
    "abs(signal_strength*liquidity)+signal_strength^2)"
)
_FORMULA_MULTIPLIERS = (
    ("macro_volatility", 1.5),
    ("macro_volume", 0.5),
    ("meso_depth", 0.75),
    ("meso_spread", 1.25),
    ("micro_order_arrival", 0.625),
    ("micro_order_size", 1.0),
)
_ROOT_FILES = frozenset(
    {
        "artifact.json",
        "expected_attempt_registry.json",
        "plan.json",
        "seal.json",
        "truth_manifest.json",
    }
)


class AAMASMisspecificationError(RuntimeError):
    """Raised when the preregistered design or execution is invalid."""


class AAMASMisspecificationIntegrityError(AAMASMisspecificationError):
    """Raised when persisted bytes fail a caller-anchored audit."""


def _fail(message: str) -> NoReturn:
    raise AAMASMisspecificationError(message)


def _integrity_fail(message: str) -> NoReturn:
    raise AAMASMisspecificationIntegrityError(message)


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _fail(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(f"{field} must be a non-negative integer")
    return value


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        _fail(f"{field} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{field} must be finite")
    return result


def _unit_interval(value: object, field: str) -> float:
    result = _finite(value, field)
    if not 0.0 <= result <= 1.0:
        _fail(f"{field} must lie in [0, 1]")
    return result


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 512:
        _fail(f"{field} must be a non-empty bounded string")
    return value


def _sha256(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _integrity_fail(f"{field} must be a lowercase SHA-256")
    return value


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        _integrity_fail(f"{field} must be an object with string keys")
    return cast(Mapping[str, object], value)


def _array(value: object, field: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        _integrity_fail(f"{field} must be an array")
    return cast(Sequence[object], value)


def _strict_fields(
    value: Mapping[str, object], expected: frozenset[str], field: str
) -> None:
    if frozenset(value) != expected:
        _integrity_fail(f"{field} has missing or unknown fields")


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
                _integrity_fail(f"could not complete write: {path.name}")
            remaining = remaining[written:]
        os.fsync(descriptor)
    except OSError as error:
        raise AAMASMisspecificationIntegrityError(
            f"could not create immutable artifact member {path.name}: {error}"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


class TruthRole(StrEnum):
    """Frozen use of an independently keyed prior truth."""

    DESIGN = "design"
    TARGET = "target"


class MisspecificationCell(StrEnum):
    """The two preregistered DGP cells."""

    WELL_SPECIFIED = "well_specified"
    MISSPECIFIED = "misspecified"


class AttemptStatus(StrEnum):
    """Terminal attempt status; failures are retained, never dropped."""

    COMPLETED = "completed"
    FAILED = "failed"


class NegativeControlDecision(StrEnum):
    """Frozen directional assessment of the negative control."""

    DETECTED = "detected"
    NOT_DETECTED = "not_detected"
    INCOMPLETE = "incomplete"


class CertificateLevel(StrEnum):
    """Only the scientifically permissible certificate level in this module."""

    ABSTAIN = "abstain"


@dataclass(frozen=True, slots=True)
class OmittedNonlinearTermSpec:
    """Code-frozen omitted term committed by the benchmark plan hash."""

    formula_id: str = _FORMULA_ID
    expression: str = _FORMULA_EXPRESSION
    fixed_strength: float = 1.0
    floor: float = 0.75
    interaction_absolute_coefficient: float = 1.0
    signal_squared_coefficient: float = 1.0
    statistic_multipliers: tuple[tuple[str, float], ...] = _FORMULA_MULTIPLIERS

    def __post_init__(self) -> None:
        if (
            self.formula_id != _FORMULA_ID
            or self.expression != _FORMULA_EXPRESSION
            or self.fixed_strength != 1.0
            or self.floor != 0.75
            or self.interaction_absolute_coefficient != 1.0
            or self.signal_squared_coefficient != 1.0
            or self.statistic_multipliers != _FORMULA_MULTIPLIERS
        ):
            _fail("the omitted nonlinear term is code-frozen for this benchmark")

    def shift(
        self,
        *,
        statistic: str,
        signal_strength: float,
        liquidity: float,
        strength: float,
    ) -> float:
        """Evaluate the preregistered deterministic shift."""

        multiplier = dict(self.statistic_multipliers).get(statistic)
        if multiplier is None:
            return 0.0
        nonlinear = (
            self.floor
            + self.interaction_absolute_coefficient * abs(signal_strength * liquidity)
            + self.signal_squared_coefficient * signal_strength * signal_strength
        )
        return strength * multiplier * nonlinear

    def to_mapping(self) -> dict[str, object]:
        return {
            "expression": self.expression,
            "fixed_strength": self.fixed_strength,
            "floor": self.floor,
            "formula_id": self.formula_id,
            "interaction_absolute_coefficient": (self.interaction_absolute_coefficient),
            "signal_squared_coefficient": self.signal_squared_coefficient,
            "statistic_multipliers": dict(self.statistic_multipliers),
        }


@dataclass(frozen=True, slots=True)
class MisspecificationThresholds:
    """Outcome-independent thresholds committed before target execution."""

    minimum_held_out_ppc_failure_gap: float = 0.25
    minimum_misspecified_ppc_failure_rate: float = 0.50
    maximum_normalized_recovery_loss: float = 0.35
    maximum_boundary_rate: float = 0.75
    maximum_multimodality_rate: float = 0.75

    def __post_init__(self) -> None:
        if self != MisspecificationThresholds._frozen():
            _fail("negative-control thresholds are frozen and cannot be tuned")

    @classmethod
    def _frozen(cls) -> MisspecificationThresholds:
        instance = object.__new__(cls)
        object.__setattr__(instance, "minimum_held_out_ppc_failure_gap", 0.25)
        object.__setattr__(instance, "minimum_misspecified_ppc_failure_rate", 0.50)
        object.__setattr__(instance, "maximum_normalized_recovery_loss", 0.35)
        object.__setattr__(instance, "maximum_boundary_rate", 0.75)
        object.__setattr__(instance, "maximum_multimodality_rate", 0.75)
        return instance

    def to_mapping(self) -> dict[str, object]:
        return {
            "maximum_boundary_rate": self.maximum_boundary_rate,
            "maximum_multimodality_rate": self.maximum_multimodality_rate,
            "maximum_normalized_recovery_loss": (self.maximum_normalized_recovery_loss),
            "minimum_held_out_ppc_failure_gap": (self.minimum_held_out_ppc_failure_gap),
            "minimum_misspecified_ppc_failure_rate": (
                self.minimum_misspecified_ppc_failure_rate
            ),
        }


@dataclass(frozen=True, slots=True)
class AAMASMisspecificationSpec:
    """Immutable design; defaults are guarded formal paper scale."""

    schema_version: int = 2
    study_id: str = "aamas-synthetic-misspecification-v2"
    master_seed: str = "aamas-synthetic-misspecification-master-v2"
    design_truth_count: int = 32
    target_truth_count: int = 256
    simulation_count: int = 350
    accepted_particles: int = 30
    posterior_draws: int = 160
    ppc_draws: int = 40
    simulator_noise_scale: float = 0.002
    omitted_term: OmittedNonlinearTermSpec = OmittedNonlinearTermSpec()
    thresholds: MisspecificationThresholds = MisspecificationThresholds()
    synthetic_only: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 2:
            _fail("misspecification spec schema_version must equal two")
        _text(self.study_id, "study_id")
        _text(self.master_seed, "master_seed")
        for field in (
            "design_truth_count",
            "target_truth_count",
            "simulation_count",
            "accepted_particles",
            "posterior_draws",
            "ppc_draws",
        ):
            _positive_int(getattr(self, field), field)
        if self.accepted_particles >= self.simulation_count:
            _fail("accepted_particles must be smaller than simulation_count")
        if self.accepted_particles < 2:
            _fail("the ABC posterior must retain at least two particles")
        if self.posterior_draws < 2 or self.ppc_draws < 2:
            _fail("posterior and PPC outputs require at least two draws")
        noise = _finite(self.simulator_noise_scale, "simulator_noise_scale")
        if not 0.0 <= noise <= 1.0:
            _fail("simulator_noise_scale must lie in [0, 1]")
        if not isinstance(self.omitted_term, OmittedNonlinearTermSpec):
            _fail("omitted_term must be an OmittedNonlinearTermSpec")
        if not isinstance(self.thresholds, MisspecificationThresholds):
            _fail("thresholds must be MisspecificationThresholds")
        if self.synthetic_only is not True:
            _fail("this negative control is synthetic-only")
        object.__setattr__(self, "simulator_noise_scale", noise)

    @property
    def execution_work(self) -> int:
        """Conservative deterministic work score for the explicit guard."""

        return (
            (self.design_truth_count + self.target_truth_count)
            * len(MisspecificationCell)
            * (self.simulation_count + self.posterior_draws + self.ppc_draws)
        )

    @property
    def is_formal_scale(self) -> bool:
        return self == AAMASMisspecificationSpec()

    def to_mapping(self) -> dict[str, object]:
        return {
            "accepted_particles": self.accepted_particles,
            "claim_scope": MISSPECIFICATION_CLAIM_SCOPE,
            "design_truth_count": self.design_truth_count,
            "master_seed": self.master_seed,
            "omitted_term": self.omitted_term.to_mapping(),
            "posterior_draws": self.posterior_draws,
            "ppc_draws": self.ppc_draws,
            "real_policy_claim": False,
            "schema_version": self.schema_version,
            "simulation_count": self.simulation_count,
            "simulator_noise_scale": self.simulator_noise_scale,
            "study_id": self.study_id,
            "synthetic_only": self.synthetic_only,
            "target_truth_count": self.target_truth_count,
            "thresholds": self.thresholds.to_mapping(),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        """Load an exact spec; formula and thresholds cannot drift."""

        _strict_fields(value, frozenset(cls().to_mapping()), "spec")
        if value["claim_scope"] != MISSPECIFICATION_CLAIM_SCOPE:
            _integrity_fail("spec claim scope changed")
        if value["real_policy_claim"] is not False:
            _integrity_fail("synthetic spec cannot authorize a real policy claim")
        omitted = _mapping(value["omitted_term"], "omitted_term")
        thresholds = _mapping(value["thresholds"], "thresholds")
        if omitted != OmittedNonlinearTermSpec().to_mapping():
            _integrity_fail("omitted nonlinear formula changed")
        if thresholds != MisspecificationThresholds().to_mapping():
            _integrity_fail("negative-control thresholds changed")
        try:
            return cls(
                schema_version=_raw_int(value["schema_version"], "schema_version"),
                study_id=_raw_string(value["study_id"], "study_id"),
                master_seed=_raw_string(value["master_seed"], "master_seed"),
                design_truth_count=_raw_int(
                    value["design_truth_count"], "design_truth_count"
                ),
                target_truth_count=_raw_int(
                    value["target_truth_count"], "target_truth_count"
                ),
                simulation_count=_raw_int(
                    value["simulation_count"], "simulation_count"
                ),
                accepted_particles=_raw_int(
                    value["accepted_particles"], "accepted_particles"
                ),
                posterior_draws=_raw_int(value["posterior_draws"], "posterior_draws"),
                ppc_draws=_raw_int(value["ppc_draws"], "ppc_draws"),
                simulator_noise_scale=_raw_float(
                    value["simulator_noise_scale"], "simulator_noise_scale"
                ),
                synthetic_only=value["synthetic_only"] is True,
            )
        except (AAMASMisspecificationError, TypeError, ValueError) as error:
            raise AAMASMisspecificationIntegrityError(
                f"invalid misspecification spec: {error}"
            ) from error

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))


def _raw_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _integrity_fail(f"{field} must be an integer")
    return value


def _raw_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        _integrity_fail(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        _integrity_fail(f"{field} must be finite")
    return result


def _raw_string(value: object, field: str) -> str:
    if not isinstance(value, str):
        _integrity_fail(f"{field} must be a string")
    return value


def _base_config(spec: AAMASMisspecificationSpec, *, seed: str) -> CalibrationConfig:
    return replace(
        synthetic_calibration_config(
            seed=seed,
            simulation_count=spec.simulation_count,
            accepted_particles=spec.accepted_particles,
            posterior_draws=spec.posterior_draws,
            ppc_draws=spec.ppc_draws,
            sbc_replicates=1,
            sbc_posterior_draws=2,
        ),
        study_id=spec.study_id,
    )


def derive_inference_seed(
    spec: AAMASMisspecificationSpec,
    *,
    role: TruthRole,
    role_ordinal: int,
) -> str:
    """Derive inference randomness without accepting truth values or hashes."""

    if not isinstance(spec, AAMASMisspecificationSpec):
        _fail("spec must be an AAMASMisspecificationSpec")
    if not isinstance(role, TruthRole):
        _fail("role must be a TruthRole")
    _nonnegative_int(role_ordinal, "role_ordinal")
    return "aamas-misspec-run-v2:" + sha256_bytes(
        canonical_json_bytes(
            {
                "master_seed": spec.master_seed,
                "role": role.value,
                "role_ordinal": role_ordinal,
                "study_id": spec.study_id,
            }
        )
    )


@dataclass(frozen=True, slots=True)
class MisspecificationTruth:
    """One prior-generated truth and its complete generation key manifest."""

    truth_id: str
    role: TruthRole
    ordinal: int
    role_ordinal: int
    parameters: tuple[tuple[str, float], ...]
    event_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(self.truth_id, "truth_id")
        if not isinstance(self.role, TruthRole):
            _fail("truth role is invalid")
        _nonnegative_int(self.ordinal, "ordinal")
        _nonnegative_int(self.role_ordinal, "role_ordinal")
        if (
            not isinstance(self.parameters, tuple)
            or not self.parameters
            or tuple(sorted(self.parameters)) != self.parameters
            or len(dict(self.parameters)) != len(self.parameters)
        ):
            _fail("truth parameters must be sorted, unique, and immutable")
        for name, value in self.parameters:
            _text(name, "parameter name")
            _finite(value, f"truth parameter {name}")
        if (
            not isinstance(self.event_keys, tuple)
            or not self.event_keys
            or len(set(self.event_keys)) != len(self.event_keys)
            or any(
                not isinstance(key, str)
                or len(key) != 69
                or not key.startswith("ptk1:")
                for key in self.event_keys
            )
        ):
            _fail("truth EventKeys must be unique ptk1 keys")

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
class ExpectedMisspecificationAttempt:
    """One exact cell/truth attempt frozen before observations are generated."""

    attempt_id: str
    truth_id: str
    role: TruthRole
    role_ordinal: int
    cell: MisspecificationCell
    dgp_strength: float
    inference_seed: str
    config_sha256: str
    truth_sha256: str
    calibration_observation_event_key: str
    held_out_observation_event_key: str

    def __post_init__(self) -> None:
        _text(self.attempt_id, "attempt_id")
        _text(self.truth_id, "truth_id")
        if not isinstance(self.role, TruthRole):
            _fail("attempt role is invalid")
        _nonnegative_int(self.role_ordinal, "role_ordinal")
        if not isinstance(self.cell, MisspecificationCell):
            _fail("attempt cell is invalid")
        strength = _finite(self.dgp_strength, "dgp_strength")
        expected_strength = (
            0.0 if self.cell is MisspecificationCell.WELL_SPECIFIED else 1.0
        )
        if strength != expected_strength:
            _fail("attempt DGP strength differs from its frozen cell")
        _text(self.inference_seed, "inference_seed")
        for value, field in (
            (self.config_sha256, "config_sha256"),
            (self.truth_sha256, "truth_sha256"),
        ):
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                _fail(f"{field} must be a lowercase SHA-256")
        for value, field in (
            (
                self.calibration_observation_event_key,
                "calibration_observation_event_key",
            ),
            (self.held_out_observation_event_key, "held_out_observation_event_key"),
        ):
            if not value.startswith("ptk1:") or len(value) != 69:
                _fail(f"{field} must be a ptk1 EventKey")

    def to_mapping(self) -> dict[str, object]:
        return {
            "attempt_id": self.attempt_id,
            "calibration_observation_event_key": (
                self.calibration_observation_event_key
            ),
            "cell": self.cell.value,
            "config_sha256": self.config_sha256,
            "dgp_strength": self.dgp_strength,
            "held_out_observation_event_key": self.held_out_observation_event_key,
            "inference_seed": self.inference_seed,
            "role": self.role.value,
            "role_ordinal": self.role_ordinal,
            "truth_id": self.truth_id,
            "truth_sha256": self.truth_sha256,
        }


def _config_for(
    spec: AAMASMisspecificationSpec, *, role: TruthRole, role_ordinal: int
) -> CalibrationConfig:
    return _base_config(
        spec,
        seed=derive_inference_seed(spec, role=role, role_ordinal=role_ordinal),
    )


def _expected_attempt(
    spec: AAMASMisspecificationSpec,
    truth: MisspecificationTruth,
    cell: MisspecificationCell,
) -> ExpectedMisspecificationAttempt:
    config = _config_for(spec, role=truth.role, role_ordinal=truth.role_ordinal)
    calibration_key = calibration_event_key(
        config,
        replicate=0,
        step=0,
        purpose="misspecification_observed:calibration",
    )
    held_out_key = calibration_event_key(
        config,
        replicate=0,
        step=1,
        purpose="misspecification_observed:held_out",
    )
    return ExpectedMisspecificationAttempt(
        attempt_id=f"{cell.value}-{truth.truth_id}",
        truth_id=truth.truth_id,
        role=truth.role,
        role_ordinal=truth.role_ordinal,
        cell=cell,
        dgp_strength=(
            0.0
            if cell is MisspecificationCell.WELL_SPECIFIED
            else spec.omitted_term.fixed_strength
        ),
        inference_seed=config.seed,
        config_sha256=config.sha256,
        truth_sha256=truth.sha256,
        calibration_observation_event_key=calibration_key.value,
        held_out_observation_event_key=held_out_key.value,
    )


@dataclass(frozen=True, slots=True)
class AAMASMisspecificationPlan:
    """Complete immutable truth tape and paired expected-attempt registry."""

    spec: AAMASMisspecificationSpec
    truths: tuple[MisspecificationTruth, ...]
    expected_attempts: tuple[ExpectedMisspecificationAttempt, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.spec, AAMASMisspecificationSpec):
            _fail("plan spec is invalid")
        truth_count = self.spec.design_truth_count + self.spec.target_truth_count
        if len(self.truths) != truth_count:
            _fail("truth manifest cardinality differs from the frozen spec")
        expected_roles = (TruthRole.DESIGN,) * self.spec.design_truth_count + (
            TruthRole.TARGET,
        ) * self.spec.target_truth_count
        if tuple(item.role for item in self.truths) != expected_roles:
            _fail("truth roles or ordering changed")
        if tuple(item.ordinal for item in self.truths) != tuple(range(truth_count)):
            _fail("truth ordinals must be complete and ordered")
        for role in TruthRole:
            rows = tuple(item for item in self.truths if item.role is role)
            if tuple(item.role_ordinal for item in rows) != tuple(range(len(rows))):
                _fail("role ordinals must be complete and ordered")
        if len({item.truth_id for item in self.truths}) != truth_count:
            _fail("truth IDs must be unique")
        truth_keys = tuple(key for item in self.truths for key in item.event_keys)
        if len(set(truth_keys)) != len(truth_keys):
            _fail("truth generator reused an EventKey")
        generator = _base_config(self.spec, seed=self.spec.master_seed)
        for truth in self.truths:
            parameters, event_keys = sample_prior(
                generator,
                replicate=truth.ordinal,
                namespace=TRUTH_NAMESPACE,
                seed=self.spec.master_seed,
            )
            expected_truth = MisspecificationTruth(
                truth_id=f"{truth.role.value}-{truth.role_ordinal:03d}",
                role=truth.role,
                ordinal=truth.ordinal,
                role_ordinal=truth.role_ordinal,
                parameters=tuple(sorted(parameters.items())),
                event_keys=event_keys,
            )
            if truth != expected_truth:
                _fail("truth manifest does not replay from sample_prior and its keys")

        exact_attempts = tuple(
            _expected_attempt(self.spec, truth, cell)
            for truth in self.truths
            for cell in MisspecificationCell
        )
        if self.expected_attempts != exact_attempts:
            _fail(
                "expected attempts differ from fixed truth membership, seeds, or cells"
            )
        for index in range(0, len(exact_attempts), 2):
            well, misspecified = exact_attempts[index : index + 2]
            if (
                well.inference_seed != misspecified.inference_seed
                or well.config_sha256 != misspecified.config_sha256
                or well.calibration_observation_event_key
                != misspecified.calibration_observation_event_key
                or well.held_out_observation_event_key
                != misspecified.held_out_observation_event_key
            ):
                _fail("paired cells do not share one inference and observation tape")

    @property
    def truth_manifest_mapping(self) -> dict[str, object]:
        return {
            "generator": "sample_prior",
            "master_seed_sha256": sha256_bytes(self.spec.master_seed.encode()),
            "namespace": TRUTH_NAMESPACE,
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

    def to_mapping(self) -> dict[str, object]:
        return {
            "claim_scope": MISSPECIFICATION_CLAIM_SCOPE,
            "expected_attempt_registry": self.expected_attempt_registry_mapping,
            "schema_version": MISSPECIFICATION_SCHEMA_VERSION,
            "spec": self.spec.to_mapping(),
            "truth_manifest": self.truth_manifest_mapping,
        }

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))


def build_aamas_misspecification_plan(
    spec: AAMASMisspecificationSpec | None = None,
) -> AAMASMisspecificationPlan:
    """Generate every truth exactly once without filtering or seed search."""

    spec = AAMASMisspecificationSpec() if spec is None else spec
    if not isinstance(spec, AAMASMisspecificationSpec):
        _fail("spec must be an AAMASMisspecificationSpec")
    generator = _base_config(spec, seed=spec.master_seed)
    truths: list[MisspecificationTruth] = []
    ordinal = 0
    for role, count in (
        (TruthRole.DESIGN, spec.design_truth_count),
        (TruthRole.TARGET, spec.target_truth_count),
    ):
        for role_ordinal in range(count):
            parameters, event_keys = sample_prior(
                generator,
                replicate=ordinal,
                namespace=TRUTH_NAMESPACE,
                seed=spec.master_seed,
            )
            truths.append(
                MisspecificationTruth(
                    truth_id=f"{role.value}-{role_ordinal:03d}",
                    role=role,
                    ordinal=ordinal,
                    role_ordinal=role_ordinal,
                    parameters=tuple(sorted(parameters.items())),
                    event_keys=event_keys,
                )
            )
            ordinal += 1
    attempts = tuple(
        _expected_attempt(spec, truth, cell)
        for truth in truths
        for cell in MisspecificationCell
    )
    return AAMASMisspecificationPlan(spec, tuple(truths), attempts)


@dataclass(frozen=True, slots=True)
class StructuralMisspecificationDGP:
    """DGP wrapper; the estimator never receives this adapter."""

    base: SyntheticMarketSimulator
    omitted_term: OmittedNonlinearTermSpec

    def simulate(
        self,
        parameters: Mapping[str, float],
        *,
        event_key: EventKey,
        cell: MisspecificationCell,
    ) -> SummaryVector:
        """Generate paired-noise summaries and apply the omitted DGP mechanism."""

        if not isinstance(event_key, EventKey):
            _fail("event_key must be an EventKey")
        if not isinstance(cell, MisspecificationCell):
            _fail("cell must be a MisspecificationCell")
        base = self.base.simulate(parameters, event_key=event_key)
        if cell is MisspecificationCell.WELL_SPECIFIED:
            return base
        signal = float(parameters["signal_strength"])
        liquidity = float(parameters["liquidity"])
        shifted = tuple(
            (
                name,
                value
                + self.omitted_term.shift(
                    statistic=name,
                    signal_strength=signal,
                    liquidity=liquidity,
                    strength=self.omitted_term.fixed_strength,
                ),
            )
            for name, value in base.values
        )
        return SummaryVector(shifted)

    def to_mapping(self) -> dict[str, object]:
        return {
            "base_simulator": self.base.to_mapping(),
            "fit_simulator_id": self.base.simulator_id,
            "omitted_term": self.omitted_term.to_mapping(),
            "omitted_term_trajectory_roles": [
                SummaryRole.CALIBRATION.value,
                SummaryRole.HELD_OUT_VALIDATION.value,
            ],
            "schema_version": 2,
            "synthetic_only": True,
        }


@dataclass(frozen=True, slots=True)
class MisspecificationAttemptOutput:
    """Complete low-level posterior, PPC, recovery, and key output."""

    config_sha256: str
    observed: CalibrationInput
    fit: EstimatorFit
    posterior: PosteriorSamples
    ppc: PosteriorPredictiveChecks
    recovery: tuple[RecoveryMetric, ...]
    identifiability: tuple[ParameterIdentifiability, ...]
    event_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.config_sha256) != 64:
            _fail("output config_sha256 must be a SHA-256")
        if not isinstance(self.observed, CalibrationInput):
            _fail("output observed input is invalid")
        if not isinstance(self.fit, EstimatorFit):
            _fail("output estimator fit is invalid")
        if not isinstance(self.posterior, PosteriorSamples):
            _fail("output posterior is invalid")
        if not isinstance(self.ppc, PosteriorPredictiveChecks):
            _fail("output PPC is invalid")
        if not self.recovery or not all(
            isinstance(item, RecoveryMetric) for item in self.recovery
        ):
            _fail("output recovery metrics are invalid")
        if not self.identifiability or not all(
            isinstance(item, ParameterIdentifiability) for item in self.identifiability
        ):
            _fail("output identifiability diagnostics are invalid")
        if not self.event_keys or len(set(self.event_keys)) != len(self.event_keys):
            _fail("attempt EventKeys must be non-empty and unique")
        expected_keys = (
            self.observed.event_keys
            + self.fit.event_keys
            + tuple(item.event_key for item in self.posterior.draws)
            + self.ppc.event_keys
        )
        if self.event_keys != expected_keys:
            _fail("attempt EventKey manifest is incomplete or reordered")

    def to_mapping(self) -> dict[str, object]:
        return {
            "config_sha256": self.config_sha256,
            "event_keys": list(self.event_keys),
            "fit": {
                "backend_id": self.fit.backend_id,
                "event_keys": list(self.fit.event_keys),
                "particles": [item.to_mapping() for item in self.fit.particles],
            },
            "identifiability": [item.to_mapping() for item in self.identifiability],
            "observed": self.observed.to_mapping(),
            "posterior": self.posterior.to_mapping(),
            "ppc": self.ppc.to_mapping(),
            "recovery": [item.to_mapping() for item in self.recovery],
        }

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))


@dataclass(frozen=True, slots=True)
class MisspecificationAttempt:
    """Terminal attempt retaining either complete output or explicit failure."""

    expected: ExpectedMisspecificationAttempt
    status: AttemptStatus
    output: MisspecificationAttemptOutput | None
    error_type: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.expected, ExpectedMisspecificationAttempt):
            _fail("attempt expected row is invalid")
        if self.status is AttemptStatus.COMPLETED:
            if (
                self.output is None
                or self.error_type is not None
                or self.error_message is not None
            ):
                _fail("completed attempt requires one output and no failure")
            if self.output.config_sha256 != self.expected.config_sha256:
                _fail("completed output differs from its frozen config")
        elif self.status is AttemptStatus.FAILED:
            if self.output is not None or not self.error_type or not self.error_message:
                _fail("failed attempt requires an explicit error and no output")
        else:
            _fail("attempt status is invalid")

    def to_mapping(self) -> dict[str, object]:
        return {
            "error_message": self.error_message,
            "error_type": self.error_type,
            "expected": self.expected.to_mapping(),
            "output": None if self.output is None else self.output.to_mapping(),
            "output_sha256": None if self.output is None else self.output.sha256,
            "status": self.status.value,
        }


@dataclass(frozen=True, slots=True)
class TargetCellAggregate:
    """Target-only diagnostics with fail-closed complete-case semantics."""

    cell: MisspecificationCell
    expected_target_attempts: int
    completed_target_attempts: int
    failed_target_attempt_ids: tuple[str, ...]
    held_out_statistic_count: int
    held_out_ppc_failures: int | None
    held_out_ppc_failure_rate: float | None
    normalized_recovery_loss: float | None
    boundary_rate: float | None
    multimodality_rate: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.cell, MisspecificationCell):
            _fail("aggregate cell is invalid")
        _positive_int(self.expected_target_attempts, "expected_target_attempts")
        _nonnegative_int(self.completed_target_attempts, "completed_target_attempts")
        _positive_int(self.held_out_statistic_count, "held_out_statistic_count")
        if self.completed_target_attempts > self.expected_target_attempts:
            _fail("completed target attempts exceed the frozen denominator")
        complete = self.completed_target_attempts == self.expected_target_attempts
        metrics = (
            self.held_out_ppc_failures,
            self.held_out_ppc_failure_rate,
            self.normalized_recovery_loss,
            self.boundary_rate,
            self.multimodality_rate,
        )
        if complete:
            if self.failed_target_attempt_ids or any(item is None for item in metrics):
                _fail("complete target aggregate omitted a metric")
            _nonnegative_int(self.held_out_ppc_failures, "held_out_ppc_failures")
            for field in (
                "held_out_ppc_failure_rate",
                "normalized_recovery_loss",
                "boundary_rate",
                "multimodality_rate",
            ):
                _unit_interval(getattr(self, field), field)
        elif not self.failed_target_attempt_ids or any(
            item is not None for item in metrics
        ):
            _fail("incomplete target aggregate must expose failures and no metrics")

    def to_mapping(self) -> dict[str, object]:
        return {
            "boundary_rate": self.boundary_rate,
            "cell": self.cell.value,
            "completed_target_attempts": self.completed_target_attempts,
            "expected_target_attempts": self.expected_target_attempts,
            "failed_target_attempt_ids": list(self.failed_target_attempt_ids),
            "held_out_ppc_failure_rate": self.held_out_ppc_failure_rate,
            "held_out_ppc_failures": self.held_out_ppc_failures,
            "held_out_statistic_count": self.held_out_statistic_count,
            "multimodality_rate": self.multimodality_rate,
            "normalized_recovery_loss": self.normalized_recovery_loss,
        }


def derive_target_aggregates(
    plan: AAMASMisspecificationPlan,
    attempts: tuple[MisspecificationAttempt, ...],
) -> tuple[TargetCellAggregate, ...]:
    """Aggregate only frozen target members; never drop a failed target row."""

    if tuple(item.expected for item in attempts) != plan.expected_attempts:
        _fail("attempts do not exactly match the frozen expected registry")
    prior_widths = {
        prior.name: prior.upper - prior.lower
        for prior in _base_config(plan.spec, seed=plan.spec.master_seed).priors
    }
    held_out_count = len(
        _base_config(plan.spec, seed=plan.spec.master_seed).summaries.held_out
    )
    results: list[TargetCellAggregate] = []
    for cell in MisspecificationCell:
        rows = tuple(
            item
            for item in attempts
            if item.expected.role is TruthRole.TARGET and item.expected.cell is cell
        )
        failed = tuple(
            item.expected.attempt_id
            for item in rows
            if item.status is AttemptStatus.FAILED
        )
        completed = tuple(item for item in rows if item.output is not None)
        if failed:
            results.append(
                TargetCellAggregate(
                    cell=cell,
                    expected_target_attempts=plan.spec.target_truth_count,
                    completed_target_attempts=len(completed),
                    failed_target_attempt_ids=failed,
                    held_out_statistic_count=held_out_count,
                    held_out_ppc_failures=None,
                    held_out_ppc_failure_rate=None,
                    normalized_recovery_loss=None,
                    boundary_rate=None,
                    multimodality_rate=None,
                )
            )
            continue
        ppc_rows = tuple(
            statistic
            for attempt in completed
            for statistic in cast(
                MisspecificationAttemptOutput, attempt.output
            ).ppc.statistics
            if statistic.role is SummaryRole.HELD_OUT_VALIDATION
        )
        recovery_rows = tuple(
            metric
            for attempt in completed
            for metric in cast(MisspecificationAttemptOutput, attempt.output).recovery
        )
        diagnostics = tuple(
            item
            for attempt in completed
            for item in cast(
                MisspecificationAttemptOutput, attempt.output
            ).identifiability
        )
        failures = sum(not item.covered for item in ppc_rows)
        normalized_loss = sum(
            item.absolute_error / prior_widths[item.parameter] for item in recovery_rows
        ) / len(recovery_rows)
        results.append(
            TargetCellAggregate(
                cell=cell,
                expected_target_attempts=plan.spec.target_truth_count,
                completed_target_attempts=len(completed),
                failed_target_attempt_ids=(),
                held_out_statistic_count=held_out_count,
                held_out_ppc_failures=failures,
                held_out_ppc_failure_rate=failures / len(ppc_rows),
                normalized_recovery_loss=normalized_loss,
                boundary_rate=sum(item.boundary_pileup for item in diagnostics)
                / len(diagnostics),
                multimodality_rate=sum(item.multimodal for item in diagnostics)
                / len(diagnostics),
            )
        )
    return tuple(results)


@dataclass(frozen=True, slots=True)
class NegativeControlAssessment:
    """Threshold-derived result; the thresholds remain in the plan."""

    decision: NegativeControlDecision
    held_out_ppc_failure_gap: float | None
    ppc_detection: bool | None
    recovery_threshold_exceeded: bool | None
    boundary_threshold_exceeded: bool | None
    multimodality_threshold_exceeded: bool | None

    def to_mapping(self) -> dict[str, object]:
        return {
            "boundary_threshold_exceeded": self.boundary_threshold_exceeded,
            "decision": self.decision.value,
            "held_out_ppc_failure_gap": self.held_out_ppc_failure_gap,
            "multimodality_threshold_exceeded": (self.multimodality_threshold_exceeded),
            "ppc_detection": self.ppc_detection,
            "recovery_threshold_exceeded": self.recovery_threshold_exceeded,
        }


def assess_negative_control(
    spec: AAMASMisspecificationSpec,
    aggregates: tuple[TargetCellAggregate, ...],
) -> NegativeControlAssessment:
    """Apply only thresholds committed in ``spec``."""

    by_cell = {item.cell: item for item in aggregates}
    if set(by_cell) != set(MisspecificationCell):
        _fail("negative-control assessment requires both frozen cells")
    well = by_cell[MisspecificationCell.WELL_SPECIFIED]
    misspecified = by_cell[MisspecificationCell.MISSPECIFIED]
    if (
        well.held_out_ppc_failure_rate is None
        or misspecified.held_out_ppc_failure_rate is None
        or misspecified.normalized_recovery_loss is None
        or misspecified.boundary_rate is None
        or misspecified.multimodality_rate is None
    ):
        return NegativeControlAssessment(
            NegativeControlDecision.INCOMPLETE,
            None,
            None,
            None,
            None,
            None,
        )
    gap = misspecified.held_out_ppc_failure_rate - well.held_out_ppc_failure_rate
    thresholds = spec.thresholds
    ppc_detection = (
        gap >= thresholds.minimum_held_out_ppc_failure_gap
        and misspecified.held_out_ppc_failure_rate
        >= thresholds.minimum_misspecified_ppc_failure_rate
    )
    return NegativeControlAssessment(
        decision=(
            NegativeControlDecision.DETECTED
            if ppc_detection
            else NegativeControlDecision.NOT_DETECTED
        ),
        held_out_ppc_failure_gap=gap,
        ppc_detection=ppc_detection,
        recovery_threshold_exceeded=(
            misspecified.normalized_recovery_loss
            > thresholds.maximum_normalized_recovery_loss
        ),
        boundary_threshold_exceeded=(
            misspecified.boundary_rate > thresholds.maximum_boundary_rate
        ),
        multimodality_threshold_exceeded=(
            misspecified.multimodality_rate > thresholds.maximum_multimodality_rate
        ),
    )


@dataclass(frozen=True, slots=True)
class MisspecificationValidityCertificate:
    """Non-authorizing certificate; absent typed evidence remains ``None``."""

    level: CertificateLevel
    negative_control_decision: NegativeControlDecision
    temporal_leakage_detected: None = None
    empirical_coverage_history: None = None
    real_policy_input_authority: None = None
    real_data_manifest_sha256: None = None
    reasons: tuple[str, ...] = (
        "Synthetic structural negative control only.",
        "No typed authority for real data, temporal leakage, or policy validity.",
    )

    def __post_init__(self) -> None:
        if self.level is not CertificateLevel.ABSTAIN:
            _fail("the synthetic negative control can only issue ABSTAIN")
        if not isinstance(self.negative_control_decision, NegativeControlDecision):
            _fail("certificate negative-control decision is invalid")
        if (
            self.temporal_leakage_detected is not None
            or self.empirical_coverage_history is not None
            or self.real_policy_input_authority is not None
            or self.real_data_manifest_sha256 is not None
        ):
            _fail("unsupported validity evidence must remain None")

    def to_mapping(self) -> dict[str, object]:
        return {
            "empirical_coverage_history": self.empirical_coverage_history,
            "level": self.level.value,
            "negative_control_decision": self.negative_control_decision.value,
            "real_data_manifest_sha256": self.real_data_manifest_sha256,
            "real_policy_input_authority": self.real_policy_input_authority,
            "reasons": list(self.reasons),
            "temporal_leakage_detected": self.temporal_leakage_detected,
        }


@dataclass(frozen=True, slots=True)
class AAMASMisspecificationArtifact:
    """Complete in-memory artifact with fixed membership and failed rows."""

    plan: AAMASMisspecificationPlan
    attempts: tuple[MisspecificationAttempt, ...]
    target_aggregates: tuple[TargetCellAggregate, ...]
    assessment: NegativeControlAssessment
    validity_certificate: MisspecificationValidityCertificate

    def __post_init__(self) -> None:
        if (
            tuple(item.expected for item in self.attempts)
            != self.plan.expected_attempts
        ):
            _fail("artifact attempts are missing, added, duplicated, or reordered")
        derived = derive_target_aggregates(self.plan, self.attempts)
        if self.target_aggregates != derived:
            _fail("target aggregates do not rederive from fixed target membership")
        assessment = assess_negative_control(self.plan.spec, derived)
        if self.assessment != assessment:
            _fail("negative-control assessment does not rederive from the plan")
        expected_certificate = MisspecificationValidityCertificate(
            CertificateLevel.ABSTAIN, assessment.decision
        )
        if self.validity_certificate != expected_certificate:
            _fail("validity certificate differs from the mandatory ABSTAIN result")
        by_truth: dict[str, list[MisspecificationAttempt]] = {}
        for attempt in self.attempts:
            by_truth.setdefault(attempt.expected.truth_id, []).append(attempt)
        for paired in by_truth.values():
            if len(paired) != 2:
                _fail("each truth must retain exactly two cell attempts")
            if all(item.output is not None for item in paired):
                left = cast(MisspecificationAttemptOutput, paired[0].output)
                right = cast(MisspecificationAttemptOutput, paired[1].output)
                if left.event_keys != right.event_keys:
                    _fail("paired cells drifted from their shared inference tape")
                if (
                    left.observed.source_for(SummaryRole.CALIBRATION).observed
                    == right.observed.source_for(SummaryRole.CALIBRATION).observed
                    or left.observed.source_for(
                        SummaryRole.HELD_OUT_VALIDATION
                    ).observed
                    == right.observed.source_for(
                        SummaryRole.HELD_OUT_VALIDATION
                    ).observed
                ):
                    _fail(
                        "paired DGP cells must differ in calibration and held-out "
                        "observations"
                    )

    @property
    def completed_count(self) -> int:
        return sum(item.status is AttemptStatus.COMPLETED for item in self.attempts)

    @property
    def failed_count(self) -> int:
        return sum(item.status is AttemptStatus.FAILED for item in self.attempts)

    def to_mapping(self) -> dict[str, object]:
        return {
            "assessment": self.assessment.to_mapping(),
            "attempts": [item.to_mapping() for item in self.attempts],
            "claim_scope": MISSPECIFICATION_CLAIM_SCOPE,
            "completed_count": self.completed_count,
            "failed_count": self.failed_count,
            "plan_sha256": self.plan.sha256,
            "schema_version": MISSPECIFICATION_SCHEMA_VERSION,
            "synthetic_only": True,
            "target_aggregates": [item.to_mapping() for item in self.target_aggregates],
            "validity_certificate": self.validity_certificate.to_mapping(),
        }

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))


def _execute_attempt(
    plan: AAMASMisspecificationPlan,
    expected: ExpectedMisspecificationAttempt,
) -> MisspecificationAttemptOutput:
    truth = next(item for item in plan.truths if item.truth_id == expected.truth_id)
    config = _config_for(plan.spec, role=truth.role, role_ordinal=truth.role_ordinal)
    base = SyntheticMarketSimulator(noise_scale=plan.spec.simulator_noise_scale)
    dgp = StructuralMisspecificationDGP(base, plan.spec.omitted_term)
    calibration_key = calibration_event_key(
        config,
        replicate=0,
        step=0,
        purpose="misspecification_observed:calibration",
    )
    held_out_key = calibration_event_key(
        config,
        replicate=0,
        step=1,
        purpose="misspecification_observed:held_out",
    )
    parameters = dict(truth.parameters)
    calibration_summary = dgp.simulate(
        parameters,
        event_key=calibration_key,
        cell=expected.cell,
    )
    held_out_summary = dgp.simulate(
        parameters,
        event_key=held_out_key,
        cell=expected.cell,
    )
    observed = synthetic_calibration_input(
        config,
        calibration_summary,
        held_out_summary,
        record_id=f"misspecification-{expected.cell.value}-{truth.truth_id}",
        calibration_generation_event_key=calibration_key.value,
        held_out_generation_event_key=held_out_key.value,
    )
    estimator = ABCPosteriorEstimator()
    fit = estimator.fit(
        config=config,
        observed=observed,
        simulator=base,
        namespace="structural_negative_control",
    )
    posterior, posterior_keys = sample_posterior(
        config=config,
        fit=fit,
        namespace="structural_negative_control",
    )
    ppc = posterior_predictive_checks(
        config=config,
        observed=observed,
        posterior=posterior,
        simulator=base,
        namespace="structural_negative_control",
    )
    recovery = synthetic_recovery(config, posterior, parameters)
    identifiability = identifiability_diagnostics(config, posterior)
    event_keys = observed.event_keys + fit.event_keys + posterior_keys + ppc.event_keys
    return MisspecificationAttemptOutput(
        config.sha256,
        observed,
        fit,
        posterior,
        ppc,
        recovery,
        identifiability,
        event_keys,
    )


def run_aamas_misspecification_benchmark(
    spec: AAMASMisspecificationSpec | None = None,
    *,
    allow_formal_scale: bool = False,
) -> AAMASMisspecificationArtifact:
    """Run every paired attempt; formal defaults require explicit authorization."""

    spec = AAMASMisspecificationSpec() if spec is None else spec
    if not isinstance(spec, AAMASMisspecificationSpec):
        _fail("spec must be an AAMASMisspecificationSpec")
    if not isinstance(allow_formal_scale, bool):
        _fail("allow_formal_scale must be boolean")
    if (
        spec.execution_work >= AAMASMisspecificationSpec().execution_work
        and not allow_formal_scale
    ):
        raise AAMASMisspecificationIntegrityError(
            "formal misspecification scale requires allow_formal_scale=True"
        )
    plan = build_aamas_misspecification_plan(spec)
    attempts: list[MisspecificationAttempt] = []
    for expected in plan.expected_attempts:
        try:
            output = _execute_attempt(plan, expected)
            attempts.append(
                MisspecificationAttempt(expected, AttemptStatus.COMPLETED, output)
            )
        except Exception as error:  # Each preregistered row reaches a terminal state.
            attempts.append(
                MisspecificationAttempt(
                    expected,
                    AttemptStatus.FAILED,
                    None,
                    type(error).__name__,
                    str(error)[:2048] or "attempt failed without a message",
                )
            )
    attempt_tuple = tuple(attempts)
    aggregates = derive_target_aggregates(plan, attempt_tuple)
    assessment = assess_negative_control(spec, aggregates)
    certificate = MisspecificationValidityCertificate(
        CertificateLevel.ABSTAIN, assessment.decision
    )
    return AAMASMisspecificationArtifact(
        plan,
        attempt_tuple,
        aggregates,
        assessment,
        certificate,
    )


@dataclass(frozen=True, slots=True)
class MisspecificationWriteReceipt:
    """External anchors required to audit a persisted directory."""

    plan_sha256: str
    root_seal_sha256: str
    artifact_sha256: str

    def __post_init__(self) -> None:
        for value in (
            self.plan_sha256,
            self.root_seal_sha256,
            self.artifact_sha256,
        ):
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                _fail("write receipt digests must be lowercase SHA-256 values")


def write_aamas_misspecification_artifact(
    directory: Path,
    artifact: AAMASMisspecificationArtifact,
) -> MisspecificationWriteReceipt:
    """Write a new fixed-layout artifact and return caller-retained anchors."""

    if not isinstance(artifact, AAMASMisspecificationArtifact):
        _fail("artifact must be an AAMASMisspecificationArtifact")
    payloads = {
        "artifact.json": _pretty_json_bytes(artifact.to_mapping()),
        "expected_attempt_registry.json": _pretty_json_bytes(
            artifact.plan.expected_attempt_registry_mapping
        ),
        "plan.json": _pretty_json_bytes(artifact.plan.to_mapping()),
        "truth_manifest.json": _pretty_json_bytes(artifact.plan.truth_manifest_mapping),
    }
    members = [
        {
            "path": name,
            "sha256": sha256_bytes(payload),
            "size_bytes": len(payload),
        }
        for name, payload in sorted(payloads.items())
    ]
    seal = {
        "artifact_sha256": artifact.sha256,
        "claim_scope": MISSPECIFICATION_CLAIM_SCOPE,
        "members": members,
        "plan_sha256": artifact.plan.sha256,
        "schema_version": 1,
    }
    seal_payload = _pretty_json_bytes(seal)
    try:
        directory.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise AAMASMisspecificationIntegrityError(
            "misspecification artifact directories are immutable"
        ) from error
    for name, payload in sorted(payloads.items()):
        _write_new_file(directory / name, payload)
    _write_new_file(directory / "seal.json", seal_payload)
    return MisspecificationWriteReceipt(
        artifact.plan.sha256,
        sha256_bytes(seal_payload),
        artifact.sha256,
    )


@dataclass(frozen=True, slots=True)
class MisspecificationAuditReport:
    """Positive fixed-layout audit against caller-supplied external anchors."""

    valid: bool
    plan_sha256: str
    root_seal_sha256: str
    artifact_sha256: str
    checked_files: tuple[str, ...]
    attempt_count: int
    failed_count: int


def _read_regular_file(path: Path) -> bytes:
    try:
        before = path.lstat()
    except OSError as error:
        raise AAMASMisspecificationIntegrityError(
            f"cannot stat artifact member {path.name}: {error}"
        ) from error
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_nlink != 1
    ):
        _integrity_fail(
            f"artifact member must be a singly linked regular file: {path.name}"
        )
    try:
        payload = path.read_bytes()
        after = path.lstat()
    except OSError as error:
        raise AAMASMisspecificationIntegrityError(
            f"cannot read artifact member {path.name}: {error}"
        ) from error
    def identity(details: os.stat_result) -> tuple[int, int, int, int, int, int]:
        return (
            details.st_dev,
            details.st_ino,
            details.st_size,
            details.st_mtime_ns,
            details.st_ctime_ns,
            details.st_nlink,
        )
    if identity(before) != identity(after) or len(payload) != before.st_size:
        _integrity_fail(f"artifact member changed while read: {path.name}")
    return payload


def _audit_artifact_membership(
    artifact: Mapping[str, object], plan: AAMASMisspecificationPlan
) -> tuple[tuple[MisspecificationAttempt, ...], int]:
    """Validate fixed membership and replay every persisted completed row."""

    _strict_fields(
        artifact,
        frozenset(
            {
                "assessment",
                "attempts",
                "claim_scope",
                "completed_count",
                "failed_count",
                "plan_sha256",
                "schema_version",
                "synthetic_only",
                "target_aggregates",
                "validity_certificate",
            }
        ),
        "artifact",
    )
    if (
        artifact["schema_version"] != MISSPECIFICATION_SCHEMA_VERSION
        or artifact["claim_scope"] != MISSPECIFICATION_CLAIM_SCOPE
        or artifact["synthetic_only"] is not True
        or artifact["plan_sha256"] != plan.sha256
    ):
        _integrity_fail("artifact header differs from the frozen plan")
    raw_attempts = _array(artifact["attempts"], "artifact attempts")
    if len(raw_attempts) != len(plan.expected_attempts):
        _integrity_fail("artifact attempt count differs from fixed membership")
    failed = 0
    replayed_attempts: list[MisspecificationAttempt] = []
    for raw, expected in zip(raw_attempts, plan.expected_attempts, strict=True):
        attempt = _mapping(raw, "attempt")
        _strict_fields(
            attempt,
            frozenset(
                {
                    "error_message",
                    "error_type",
                    "expected",
                    "output",
                    "output_sha256",
                    "status",
                }
            ),
            "attempt",
        )
        if attempt["expected"] != expected.to_mapping():
            _integrity_fail("artifact attempt differs from expected registry")
        status_value = attempt["status"]
        if status_value == AttemptStatus.COMPLETED.value:
            output = _mapping(attempt["output"], "completed output")
            digest = sha256_bytes(canonical_json_bytes(output))
            if attempt["output_sha256"] != digest:
                _integrity_fail("completed attempt output digest changed")
            if (
                attempt["error_type"] is not None
                or attempt["error_message"] is not None
            ):
                _integrity_fail("completed attempt contains failure fields")
            try:
                replayed_output = _execute_attempt(plan, expected)
            except Exception as error:
                raise AAMASMisspecificationIntegrityError(
                    "completed artifact output could not be reproduced by "
                    "deterministic frozen-plan replay"
                ) from error
            if replayed_output.to_mapping() != output:
                _integrity_fail(
                    "completed artifact output differs from deterministic "
                    "frozen-plan replay"
                )
            replayed_attempts.append(
                MisspecificationAttempt(
                    expected,
                    AttemptStatus.COMPLETED,
                    replayed_output,
                )
            )
        elif status_value == AttemptStatus.FAILED.value:
            failed += 1
            error_type = attempt["error_type"]
            error_message = attempt["error_message"]
            if (
                attempt["output"] is not None
                or attempt["output_sha256"] is not None
                or not isinstance(error_type, str)
                or not error_type
                or not isinstance(error_message, str)
                or not error_message
            ):
                _integrity_fail("failed attempt did not retain a valid failure")
            replayed_attempts.append(
                MisspecificationAttempt(
                    expected,
                    AttemptStatus.FAILED,
                    None,
                    error_type,
                    error_message,
                )
            )
        else:
            _integrity_fail("attempt status is unsupported")
    completed = len(raw_attempts) - failed
    if artifact["completed_count"] != completed or artifact["failed_count"] != failed:
        _integrity_fail("artifact status counts differ from attempt rows")
    attempt_tuple = tuple(replayed_attempts)
    aggregates = derive_target_aggregates(plan, attempt_tuple)
    assessment = assess_negative_control(plan.spec, aggregates)
    certificate = MisspecificationValidityCertificate(
        CertificateLevel.ABSTAIN,
        assessment.decision,
    )
    reconstructed = AAMASMisspecificationArtifact(
        plan,
        attempt_tuple,
        aggregates,
        assessment,
        certificate,
    )
    if reconstructed.to_mapping() != artifact:
        _integrity_fail(
            "artifact aggregates, assessment, certificate, or failure registry "
            "do not rederive from replayed completed rows"
        )
    return attempt_tuple, failed


def audit_aamas_misspecification_artifact(
    directory: Path,
    *,
    expected_plan_sha256: str,
    expected_root_seal_sha256: str,
    allow_formal_scale: bool = False,
) -> MisspecificationAuditReport:
    """Audit exact files and replay complete outputs from the frozen plan."""

    _sha256(expected_plan_sha256, "expected_plan_sha256")
    _sha256(expected_root_seal_sha256, "expected_root_seal_sha256")
    if not isinstance(allow_formal_scale, bool):
        _integrity_fail("allow_formal_scale must be boolean")
    try:
        root_details = directory.lstat()
    except OSError as error:
        raise AAMASMisspecificationIntegrityError(
            f"cannot stat artifact directory: {error}"
        ) from error
    if not stat.S_ISDIR(root_details.st_mode) or stat.S_ISLNK(root_details.st_mode):
        _integrity_fail("artifact root must be a real directory")
    try:
        names = frozenset(item.name for item in directory.iterdir())
    except OSError as error:
        raise AAMASMisspecificationIntegrityError(
            f"cannot enumerate artifact directory: {error}"
        ) from error
    if names != _ROOT_FILES:
        _integrity_fail("artifact root members differ from the fixed layout")
    payloads = {name: _read_regular_file(directory / name) for name in _ROOT_FILES}
    if sha256_bytes(payloads["seal.json"]) != expected_root_seal_sha256:
        _integrity_fail("root seal differs from the caller-retained digest")
    try:
        seal = read_json_bytes_object(payloads["seal.json"], source="seal.json")
        plan_mapping = read_json_bytes_object(payloads["plan.json"], source="plan.json")
        artifact_mapping = read_json_bytes_object(
            payloads["artifact.json"], source="artifact.json"
        )
        registry = read_json_bytes_object(
            payloads["expected_attempt_registry.json"],
            source="expected_attempt_registry.json",
        )
        truth_manifest = read_json_bytes_object(
            payloads["truth_manifest.json"], source="truth_manifest.json"
        )
    except MetadataValidationError as error:
        raise AAMASMisspecificationIntegrityError(str(error)) from error
    _strict_fields(
        seal,
        frozenset(
            {
                "artifact_sha256",
                "claim_scope",
                "members",
                "plan_sha256",
                "schema_version",
            }
        ),
        "seal",
    )
    if seal["claim_scope"] != MISSPECIFICATION_CLAIM_SCOPE:
        _integrity_fail("seal claim scope changed")
    members = _array(seal["members"], "seal members")
    expected_member_names = sorted(_ROOT_FILES - {"seal.json"})
    if len(members) != len(expected_member_names):
        _integrity_fail("seal member count changed")
    for raw, name in zip(members, expected_member_names, strict=True):
        row = _mapping(raw, "seal member")
        _strict_fields(row, frozenset({"path", "sha256", "size_bytes"}), "member")
        payload = payloads[name]
        if (
            row["path"] != name
            or row["sha256"] != sha256_bytes(payload)
            or row["size_bytes"] != len(payload)
        ):
            _integrity_fail(f"sealed member differs: {name}")
    _strict_fields(
        plan_mapping,
        frozenset(
            {
                "claim_scope",
                "expected_attempt_registry",
                "schema_version",
                "spec",
                "truth_manifest",
            }
        ),
        "plan",
    )
    spec = AAMASMisspecificationSpec.from_mapping(
        _mapping(plan_mapping["spec"], "plan spec")
    )
    expected_plan = build_aamas_misspecification_plan(spec)
    if plan_mapping != expected_plan.to_mapping():
        _integrity_fail("plan does not replay from its immutable spec")
    if expected_plan.sha256 != expected_plan_sha256:
        _integrity_fail("plan differs from the caller-retained digest")
    if (
        seal["plan_sha256"] != expected_plan.sha256
        or registry != expected_plan.expected_attempt_registry_mapping
        or truth_manifest != expected_plan.truth_manifest_mapping
    ):
        _integrity_fail("root manifests differ from the replayed plan")
    artifact_sha256 = sha256_bytes(canonical_json_bytes(artifact_mapping))
    if seal["artifact_sha256"] != artifact_sha256:
        _integrity_fail("artifact semantic digest differs from the root seal")
    if (
        spec.execution_work >= AAMASMisspecificationSpec().execution_work
        and not allow_formal_scale
    ):
        _integrity_fail(
            "formal misspecification audit requires allow_formal_scale=True"
        )
    replayed_attempts, failed_count = _audit_artifact_membership(
        artifact_mapping, expected_plan
    )
    return MisspecificationAuditReport(
        True,
        expected_plan.sha256,
        expected_root_seal_sha256,
        artifact_sha256,
        tuple(sorted(_ROOT_FILES)),
        len(replayed_attempts),
        failed_count,
    )


__all__ = [
    "AAMASMisspecificationArtifact",
    "AAMASMisspecificationError",
    "AAMASMisspecificationIntegrityError",
    "AAMASMisspecificationPlan",
    "AAMASMisspecificationSpec",
    "AttemptStatus",
    "CertificateLevel",
    "ExpectedMisspecificationAttempt",
    "MISSPECIFICATION_CLAIM_SCOPE",
    "MisspecificationAttempt",
    "MisspecificationAttemptOutput",
    "MisspecificationAuditReport",
    "MisspecificationCell",
    "MisspecificationThresholds",
    "MisspecificationTruth",
    "MisspecificationValidityCertificate",
    "MisspecificationWriteReceipt",
    "NegativeControlAssessment",
    "NegativeControlDecision",
    "OmittedNonlinearTermSpec",
    "StructuralMisspecificationDGP",
    "TargetCellAggregate",
    "TruthRole",
    "assess_negative_control",
    "audit_aamas_misspecification_artifact",
    "build_aamas_misspecification_plan",
    "derive_inference_seed",
    "derive_target_aggregates",
    "run_aamas_misspecification_benchmark",
    "write_aamas_misspecification_artifact",
]
