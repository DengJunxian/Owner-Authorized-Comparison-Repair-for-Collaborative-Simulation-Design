"""Verifier-owned local synthetic execution for policy-functional rank studies.

The phase-one :mod:`ledger_policy_functional_sbc` module intentionally accepts
caller-provided rows.  This module adds a narrower, process-local execution
envelope: it consumes one opaque plan authorization, draws synthetic prior
truths, constructs sealed pre-policy calibration runs, executes every oracle
and posterior policy-functional cell through the same integrated market path,
and only then projects the retained attempts into the phase-one arithmetic
core.

This is not an external preregistration service or a statistical power
certificate.  The ABC posterior is not asserted to be exactly exchangeable
with the prior truth, internal agent RNG is not capability-audited here, and a
dynamic content hash is not custody.  All real-world, coverage, causal,
publication, external chronology, and rooted-evidence claims remain false.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from threading import RLock
from typing import Literal, NoReturn
from weakref import WeakKeyDictionary, WeakValueDictionary, finalize

from policytwin.calibration import (
    CalibrationConfig,
    CalibrationError,
    CalibrationInput,
    CalibrationRun,
    IntegratedSyntheticCalibrationAdapter,
    run_synthetic_calibration,
    sample_prior,
    synthetic_calibration_input,
)
from policytwin.calibration.estimation import calibration_event_key
from policytwin.counterfactual import EventKey
from policytwin.counterfactual.integrated_market import IntegratedSyntheticMarketConfig
from policytwin.counterfactual.paired import PolicyWorldPair
from policytwin.evaluation.ledger_derived import (
    ExactMetricState,
    ExactRational,
    LedgerDerivedEvaluation,
    derive_protocol_bound_ledger_evidence,
)
from policytwin.evaluation.ledger_policy_functional_sbc import (
    CallerDeclaredRankThreshold,
    CanaryDetection,
    CanaryKind,
    MetaReplicateFailureStage,
    MetaReplicateStatus,
    PolicyFunctionalCanaryResult,
    PolicyFunctionalMetaReplicate,
    PolicyFunctionalOracleRow,
    PolicyFunctionalPosteriorRow,
    PolicyFunctionalSBCResult,
    PolicyFunctionalSBCSpec,
    derive_policy_functional_sbc_result,
)
from policytwin.evaluation.ledger_posterior import LedgerPosteriorOutputSpec
from policytwin.experiments.integrated_synthetic import (
    ProtocolBoundIntegratedSyntheticRunArtifact,
    integrated_synthetic_pair,
    integrated_synthetic_pair_input_sha256,
    run_protocol_bound_integrated_synthetic_pair,
)
from policytwin.protocol import (
    ClaimScope,
    FrozenDefinition,
    ProtocolExecutionAuthorization,
    ProtocolExecutionConsumer,
    ProtocolExecutionPurpose,
    ProtocolExecutionReceipt,
    RandomnessProtocol,
    StudyMode,
    StudyProtocol,
    SyntheticProtocolExecutionIssuer,
    consume_protocol_execution_authorization,
    validate_protocol_execution_receipt_binding,
)
from policytwin.provenance import canonical_json_bytes, sha256_bytes

POLICY_FUNCTIONAL_META_STUDY_SCHEMA_VERSION = (
    "policytwin.policy-functional-meta-study.v1"
)
POLICY_FUNCTIONAL_META_STUDY_CLAIM_SCOPE: Literal[
    "local_synthetic_prior_calibration_and_integrated_cell_execution_mechanics_only"
] = "local_synthetic_prior_calibration_and_integrated_cell_execution_mechanics_only"

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class PolicyFunctionalMetaStudyError(RuntimeError):
    """Fail-closed error with a stable audit code."""

    def __init__(self, message: str, *, code: str = "meta_study_contract_violation"):
        super().__init__(message)
        self.code = code


def _fail(message: str, *, code: str = "meta_study_contract_violation") -> NoReturn:
    raise PolicyFunctionalMetaStudyError(message, code=code)


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        _fail(f"{field} must be a stable ASCII identifier", code="invalid_identifier")
    return value


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail(f"{field} must be a lowercase SHA-256 digest", code="invalid_digest")
    return value


def _aware(value: object, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        _fail(f"{field} must be timezone-aware", code="invalid_chronology")
    return value.astimezone(UTC)


def _ordinal(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(f"{field} must be a non-negative integer", code="invalid_index")
    return value


def _positive(value: object, field: str, *, minimum: int = 1) -> int:
    result = _ordinal(value, field)
    if result < minimum:
        _fail(f"{field} must be at least {minimum}", code="invalid_dimension")
    return result


def _digest_mapping(value: Mapping[str, object]) -> str:
    return sha256_bytes(canonical_json_bytes(dict(value)))


def _require_digest_binding(
    *,
    expected: str,
    observed: str,
    code: str,
    message: str,
) -> None:
    _sha256(expected, "expected binding")
    _sha256(observed, "observed binding")
    if observed != expected:
        _fail(message, code=code)


class PolicyFunctionalCellRole(StrEnum):
    """Role of one functional evaluation row."""

    ORACLE = "oracle"
    POSTERIOR = "posterior"


class PolicyFunctionalCellStatus(StrEnum):
    """Terminal state of every planned attempt."""

    COMPLETED = "completed"
    FAILED = "failed"
    ABSTAINED_DEPENDENCY = "abstained_dependency"


@dataclass(frozen=True, slots=True)
class PolicyFunctionalCalibrationReceipt:
    """Recomputed closure for one synthetic pre-policy calibration run."""

    meta_replicate_index: int
    config_sha256: str
    input_sha256: str
    truth_parameter_sha256: str
    posterior_sha256: str
    posterior_draw_sha256s: tuple[str, ...]
    posterior_parameter_sha256s: tuple[str, ...]
    parameter_sbc_sha256: str
    report_sha256: str
    protocol_receipt_sha256: str
    prior_event_keys_sha256: str
    prior_event_key_count: int
    event_keys_sha256: str
    event_key_count: int
    synthetic_pre_policy_replay_verified: Literal[True] = True

    def __post_init__(self) -> None:
        _ordinal(self.meta_replicate_index, "meta_replicate_index")
        for field in (
            "config_sha256",
            "input_sha256",
            "truth_parameter_sha256",
            "posterior_sha256",
            "parameter_sbc_sha256",
            "report_sha256",
            "protocol_receipt_sha256",
            "prior_event_keys_sha256",
            "event_keys_sha256",
        ):
            _sha256(getattr(self, field), field)
        if (
            not isinstance(self.posterior_draw_sha256s, tuple)
            or not isinstance(self.posterior_parameter_sha256s, tuple)
            or not self.posterior_draw_sha256s
            or len(self.posterior_draw_sha256s) != len(self.posterior_parameter_sha256s)
        ):
            _fail("calibration receipt posterior manifest is invalid")
        for index, value in enumerate(self.posterior_draw_sha256s):
            _sha256(value, f"posterior_draw_sha256s[{index}]")
        for index, value in enumerate(self.posterior_parameter_sha256s):
            _sha256(value, f"posterior_parameter_sha256s[{index}]")
        if len(set(self.posterior_draw_sha256s)) != len(self.posterior_draw_sha256s):
            _fail("calibration receipt posterior draw identities are not unique")
        _positive(self.event_key_count, "event_key_count")
        _positive(self.prior_event_key_count, "prior_event_key_count")
        if self.synthetic_pre_policy_replay_verified is not True:
            _fail("calibration receipt claim boundary changed")

    def to_mapping(self) -> dict[str, object]:
        return {
            "config_sha256": self.config_sha256,
            "event_key_count": self.event_key_count,
            "event_keys_sha256": self.event_keys_sha256,
            "input_sha256": self.input_sha256,
            "meta_replicate_index": self.meta_replicate_index,
            "parameter_sbc_sha256": self.parameter_sbc_sha256,
            "posterior_sha256": self.posterior_sha256,
            "posterior_draw_sha256s": list(self.posterior_draw_sha256s),
            "posterior_parameter_sha256s": list(self.posterior_parameter_sha256s),
            "prior_event_key_count": self.prior_event_key_count,
            "prior_event_keys_sha256": self.prior_event_keys_sha256,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "report_sha256": self.report_sha256,
            "synthetic_pre_policy_replay_verified": (
                self.synthetic_pre_policy_replay_verified
            ),
            "truth_parameter_sha256": self.truth_parameter_sha256,
        }

    @property
    def sha256(self) -> str:
        return _digest_mapping(self.to_mapping())


@dataclass(frozen=True, slots=True)
class PolicyFunctionalCellAttempt:
    """One retained oracle or posterior scenario attempt, including failures."""

    meta_replicate_index: int
    role: PolicyFunctionalCellRole
    posterior_index: int | None
    scenario_replicate: int
    cell_id: str
    evaluation_event_key: str
    parameter_sha256: str
    posterior_draw_sha256: str | None
    scenario_commitment_sha256: str
    status: PolicyFunctionalCellStatus
    value: ExactRational | None
    source_artifact_sha256: str | None
    ledger_evidence_sha256: str | None
    baseline_shock_ledger_sha256: str | None
    candidate_shock_ledger_sha256: str | None
    mechanism_config_sha256: str
    failure_code: str | None = None
    failure_reason: str | None = None
    failure_evidence_sha256: str | None = None

    def __post_init__(self) -> None:
        _ordinal(self.meta_replicate_index, "meta_replicate_index")
        _ordinal(self.scenario_replicate, "scenario_replicate")
        if not isinstance(self.role, PolicyFunctionalCellRole):
            _fail("cell role has the wrong type")
        if self.role is PolicyFunctionalCellRole.ORACLE:
            if (
                self.posterior_index is not None
                or self.posterior_draw_sha256 is not None
            ):
                _fail("oracle cell cannot carry posterior identity")
        else:
            _ordinal(self.posterior_index, "posterior_index")
            _sha256(self.posterior_draw_sha256, "posterior_draw_sha256")
        _identifier(self.cell_id, "cell_id")
        if (
            not isinstance(self.evaluation_event_key, str)
            or not self.evaluation_event_key
        ):
            _fail("evaluation_event_key must be non-empty")
        for field in (
            "parameter_sha256",
            "scenario_commitment_sha256",
            "mechanism_config_sha256",
        ):
            _sha256(getattr(self, field), field)
        if not isinstance(self.status, PolicyFunctionalCellStatus):
            _fail("cell status has the wrong type")
        evidence_fields = (
            self.source_artifact_sha256,
            self.ledger_evidence_sha256,
            self.baseline_shock_ledger_sha256,
            self.candidate_shock_ledger_sha256,
        )
        if self.status is PolicyFunctionalCellStatus.COMPLETED:
            if not isinstance(self.value, ExactRational) or any(
                value is None for value in evidence_fields
            ):
                _fail("completed cell requires exact value and execution evidence")
            for index, value in enumerate(evidence_fields):
                _sha256(value, f"completed evidence digest {index}")
            if (
                self.failure_code is not None
                or self.failure_reason is not None
                or self.failure_evidence_sha256 is not None
            ):
                _fail("completed cell cannot carry failure evidence")
        else:
            if self.value is not None or any(
                value is not None for value in evidence_fields
            ):
                _fail("failed cell cannot carry completed execution evidence")
            if (
                self.failure_code is None
                or not self.failure_reason
                or self.failure_evidence_sha256 is None
            ):
                _fail("failed cell requires reason and evidence")
            _identifier(self.failure_code, "failure_code")
            _sha256(self.failure_evidence_sha256, "failure_evidence_sha256")

    def _fresh_validate(self) -> None:
        fresh = PolicyFunctionalCellAttempt(
            meta_replicate_index=self.meta_replicate_index,
            role=self.role,
            posterior_index=self.posterior_index,
            scenario_replicate=self.scenario_replicate,
            cell_id=self.cell_id,
            evaluation_event_key=self.evaluation_event_key,
            parameter_sha256=self.parameter_sha256,
            posterior_draw_sha256=self.posterior_draw_sha256,
            scenario_commitment_sha256=self.scenario_commitment_sha256,
            status=self.status,
            value=self.value,
            source_artifact_sha256=self.source_artifact_sha256,
            ledger_evidence_sha256=self.ledger_evidence_sha256,
            baseline_shock_ledger_sha256=self.baseline_shock_ledger_sha256,
            candidate_shock_ledger_sha256=self.candidate_shock_ledger_sha256,
            mechanism_config_sha256=self.mechanism_config_sha256,
            failure_code=self.failure_code,
            failure_reason=self.failure_reason,
            failure_evidence_sha256=self.failure_evidence_sha256,
        )
        if fresh != self:
            _fail("cell attempt was mutated outside validated construction")

    def to_mapping(self) -> dict[str, object]:
        self._fresh_validate()
        return {
            "baseline_shock_ledger_sha256": self.baseline_shock_ledger_sha256,
            "candidate_shock_ledger_sha256": self.candidate_shock_ledger_sha256,
            "cell_id": self.cell_id,
            "evaluation_event_key": self.evaluation_event_key,
            "failure_code": self.failure_code,
            "failure_evidence_sha256": self.failure_evidence_sha256,
            "failure_reason": self.failure_reason,
            "ledger_evidence_sha256": self.ledger_evidence_sha256,
            "mechanism_config_sha256": self.mechanism_config_sha256,
            "meta_replicate_index": self.meta_replicate_index,
            "parameter_sha256": self.parameter_sha256,
            "posterior_draw_sha256": self.posterior_draw_sha256,
            "posterior_index": self.posterior_index,
            "role": self.role.value,
            "scenario_commitment_sha256": self.scenario_commitment_sha256,
            "scenario_replicate": self.scenario_replicate,
            "source_artifact_sha256": self.source_artifact_sha256,
            "status": self.status.value,
            "value": None if self.value is None else self.value.to_mapping(),
        }

    @property
    def sha256(self) -> str:
        return _digest_mapping(self.to_mapping())


@dataclass(frozen=True, slots=True)
class PolicyFunctionalContractCanaryReceipt:
    """Actual local execution of one deterministic guard mutation."""

    kind: CanaryKind
    detection: CanaryDetection
    expected_error_code: str
    observed_error_code: str
    injected_input_sha256: str
    guard_evidence_sha256: str
    reason: str
    execution_attempted: Literal[True] = True
    statistical_power_evidence: Literal[False] = False

    def __post_init__(self) -> None:
        if not isinstance(self.kind, CanaryKind) or not isinstance(
            self.detection, CanaryDetection
        ):
            _fail("canary receipt has invalid enums")
        for field in ("expected_error_code", "observed_error_code"):
            _identifier(getattr(self, field), field)
        for field in ("injected_input_sha256", "guard_evidence_sha256"):
            _sha256(getattr(self, field), field)
        if not self.reason:
            _fail("canary receipt requires a reason")
        if (
            self.execution_attempted is not True
            or self.statistical_power_evidence is not False
        ):
            _fail("contract canary claim boundary changed")
        if (
            not self.detection.detected
            or self.expected_error_code != self.observed_error_code
        ):
            _fail("contract canary did not observe its predeclared rejection")

    def to_mapping(self) -> dict[str, object]:
        return {
            "detection": self.detection.value,
            "execution_attempted": self.execution_attempted,
            "expected_error_code": self.expected_error_code,
            "guard_evidence_sha256": self.guard_evidence_sha256,
            "injected_input_sha256": self.injected_input_sha256,
            "kind": self.kind.value,
            "observed_error_code": self.observed_error_code,
            "reason": self.reason,
            "statistical_power_evidence": self.statistical_power_evidence,
        }

    @property
    def sha256(self) -> str:
        return _digest_mapping(self.to_mapping())

    @property
    def receipt_sha256(self) -> str:
        return self.sha256


@dataclass(frozen=True, slots=True, init=False)
class PolicyFunctionalMetaStudyPlan:
    """Factory-built plan fixed before the local meta-run starts."""

    plan_id: str
    frozen_at: datetime
    outcomes_available_at: datetime
    calibration_configs: tuple[CalibrationConfig, ...]
    calibration_simulator: IntegratedSyntheticCalibrationAdapter
    market_config: IntegratedSyntheticMarketConfig
    event_key_seed: str
    scenario_replicate_count: int
    primary_output: LedgerPosteriorOutputSpec
    core_spec: PolicyFunctionalSBCSpec
    candidate_allow_buy: bool
    candidate_allow_sell: bool
    candidate_is_noop: bool
    step_count: int
    include_all_groups: bool
    scenario_event_keys: tuple[str, ...]
    pairs: tuple[PolicyWorldPair, ...]
    claim_scope: str = POLICY_FUNCTIONAL_META_STUDY_CLAIM_SCOPE
    schema_version: str = POLICY_FUNCTIONAL_META_STUDY_SCHEMA_VERSION

    def __init__(self) -> None:
        raise TypeError(
            "PolicyFunctionalMetaStudyPlan is factory-produced; use "
            "build_policy_functional_meta_study_plan"
        )

    def __copy__(self) -> NoReturn:
        raise TypeError("policy-functional meta-study plans cannot be copied")

    def __deepcopy__(self, memo: object) -> NoReturn:
        del memo
        raise TypeError("policy-functional meta-study plans cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("policy-functional meta-study plans cannot be serialized")

    @property
    def meta_replicate_count(self) -> int:
        return len(self.calibration_configs)

    @property
    def posterior_draw_count(self) -> int:
        return self.core_spec.posterior_draw_count

    @property
    def parameter_sbc_design_sha256(self) -> str:
        """Return the pre-outcome design commitment, never an observed result."""
        return self.core_spec.parameter_sbc_sha256

    @property
    def expected_cell_count(self) -> int:
        return (
            self.meta_replicate_count
            * (self.posterior_draw_count + 1)
            * self.scenario_replicate_count
        )

    @property
    def frozen_definition(self) -> FrozenDefinition:
        return FrozenDefinition(
            definition_id=self.plan_id,
            description="Local synthetic policy-functional meta-study plan.",
            definition_sha256=self.sha256,
        )

    def _fresh_validate(self) -> None:
        _identifier(self.plan_id, "plan_id")
        frozen = _aware(self.frozen_at, "frozen_at")
        available = _aware(self.outcomes_available_at, "outcomes_available_at")
        if frozen >= available:
            _fail(
                "plan time must precede declared outcomes availability",
                code="invalid_chronology",
            )
        if (
            not isinstance(self.calibration_configs, tuple)
            or not self.calibration_configs
        ):
            _fail("plan requires calibration configurations")
        if not all(
            type(item) is CalibrationConfig for item in self.calibration_configs
        ):
            _fail("plan calibration configurations have the wrong type")
        if len({item.sha256 for item in self.calibration_configs}) != len(
            self.calibration_configs
        ):
            _fail("meta-replicate calibration configurations must be unique")
        if any(
            item.synthetic_only is not True
            or item.cutoff != self.market_config.cutoff
            or item.posterior_draws != self.core_spec.posterior_draw_count
            for item in self.calibration_configs
        ):
            _fail("calibration configs differ from the meta-study dimensions or cutoff")
        if (
            type(self.calibration_simulator)
            is not IntegratedSyntheticCalibrationAdapter
        ):
            _fail("plan requires the exact integrated calibration adapter")
        if type(self.market_config) is not IntegratedSyntheticMarketConfig:
            _fail("plan requires the exact integrated market configuration")
        if self.market_config.scenario_generator_spec is None:
            _fail("meta-study requires the frozen forward scenario generator")
        _identifier(self.event_key_seed, "event_key_seed")
        _positive(self.scenario_replicate_count, "scenario_replicate_count")
        if type(self.primary_output) is not LedgerPosteriorOutputSpec:
            _fail("primary_output has the wrong type")
        if type(self.core_spec) is not PolicyFunctionalSBCSpec:
            _fail("core_spec has the wrong type")
        self.core_spec._fresh_validate()
        if (
            self.core_spec.meta_replicate_count != len(self.calibration_configs)
            or self.core_spec.scenario_replicate_count != self.scenario_replicate_count
            or self.core_spec.primary_output_id != self.primary_output.output_id
            or self.core_spec.primary_output_definition_sha256
            != _digest_mapping(self.primary_output.to_mapping())
        ):
            _fail("core specification dimensions differ from the plan")
        if (
            self.core_spec.threshold.declared_frozen_at > frozen
            or self.core_spec.threshold.declared_audit_outcomes_available_at
            != available
        ):
            _fail(
                "rank threshold chronology differs from the outer plan",
                code="invalid_chronology",
            )
        if not all(
            isinstance(value, bool)
            for value in (
                self.candidate_allow_buy,
                self.candidate_allow_sell,
                self.candidate_is_noop,
                self.include_all_groups,
            )
        ):
            _fail("policy and group switches must be boolean")
        _positive(self.step_count, "step_count", minimum=2)
        if (
            len(self.scenario_event_keys)
            != len(self.calibration_configs) * self.scenario_replicate_count
        ):
            _fail("scenario EventKey manifest has the wrong cardinality")
        if len(set(self.scenario_event_keys)) != len(self.scenario_event_keys):
            _fail("scenario EventKey manifest contains duplicates")
        expected_scenario_keys = tuple(
            _scenario_event_key(
                plan_id=self.plan_id,
                seed=self.event_key_seed,
                meta_replicate_index=replicate,
                scenario_replicate=scenario,
                scenario_replicate_count=self.scenario_replicate_count,
            ).value
            for replicate in range(len(self.calibration_configs))
            for scenario in range(self.scenario_replicate_count)
        )
        if self.scenario_event_keys != expected_scenario_keys:
            _fail(
                "scenario EventKey manifest differs from the frozen key design",
                code="rng_mismatch",
            )
        if (
            not isinstance(self.pairs, tuple)
            or len(self.pairs) != 1
            or type(self.pairs[0]) is not PolicyWorldPair
        ):
            _fail("plan requires one pre-outcome pair template")
        first = self.calibration_configs[0]
        template_parameters = {
            prior.name: (prior.lower + prior.upper) / 2.0 for prior in first.priors
        }
        expected_pair = integrated_synthetic_pair(
            event_key=_scenario_event_key(
                plan_id=self.plan_id,
                seed=self.event_key_seed,
                meta_replicate_index=0,
                scenario_replicate=0,
                scenario_replicate_count=self.scenario_replicate_count,
            ),
            parameters=template_parameters,
            config=self.market_config,
            candidate_allow_buy=self.candidate_allow_buy,
            candidate_allow_sell=self.candidate_allow_sell,
            candidate_is_noop=self.candidate_is_noop,
            step_count=self.step_count,
            include_all_groups=self.include_all_groups,
        )
        if self.pairs[0] != expected_pair:
            _fail(
                "pre-outcome pair template differs from the frozen plan",
                code="pair_template_mismatch",
            )
        if (
            self.claim_scope != POLICY_FUNCTIONAL_META_STUDY_CLAIM_SCOPE
            or self.schema_version != POLICY_FUNCTIONAL_META_STUDY_SCHEMA_VERSION
        ):
            _fail("meta-study schema or claim scope changed")

    def to_mapping(self) -> dict[str, object]:
        self._fresh_validate()
        return {
            "calibration_configs": [
                item.to_mapping() for item in self.calibration_configs
            ],
            "calibration_simulator": self.calibration_simulator.to_mapping(),
            "candidate_allow_buy": self.candidate_allow_buy,
            "candidate_allow_sell": self.candidate_allow_sell,
            "candidate_is_noop": self.candidate_is_noop,
            "claim_scope": self.claim_scope,
            "core_spec": self.core_spec.to_mapping(),
            "event_key_seed_sha256": sha256_bytes(self.event_key_seed.encode("utf-8")),
            "frozen_at": self.frozen_at.isoformat(),
            "include_all_groups": self.include_all_groups,
            "market_config": self.market_config.to_mapping(),
            "outcomes_available_at": self.outcomes_available_at.isoformat(),
            "pair_template": self.pairs[0].to_mapping(),
            "parameter_sbc_design_sha256": self.parameter_sbc_design_sha256,
            "plan_id": self.plan_id,
            "primary_output": self.primary_output.to_mapping(),
            "scenario_event_keys": list(self.scenario_event_keys),
            "scenario_replicate_count": self.scenario_replicate_count,
            "schema_version": self.schema_version,
            "step_count": self.step_count,
        }

    @property
    def sha256(self) -> str:
        return _digest_mapping(self.to_mapping())


@dataclass(frozen=True, slots=True, init=False, weakref_slot=True)
class PreparedPolicyFunctionalMetaStudy:
    """Opaque one-use, process-local plan/protocol capability."""

    plan: PolicyFunctionalMetaStudyPlan
    protocol: StudyProtocol
    execution_started_at: datetime
    protocol_authorization: ProtocolExecutionAuthorization

    def __init__(self) -> None:
        raise TypeError(
            "PreparedPolicyFunctionalMetaStudy is verifier-produced; use "
            "authorize_policy_functional_meta_study"
        )

    def __copy__(self) -> NoReturn:
        raise TypeError("prepared meta-study capability cannot be copied")

    def __deepcopy__(self, memo: object) -> NoReturn:
        del memo
        raise TypeError("prepared meta-study capability cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("PreparedPolicyFunctionalMetaStudy cannot be serialized")


_PREPARED_LOCK = RLock()
_PREPARED_VALUES: WeakValueDictionary[int, PreparedPolicyFunctionalMetaStudy] = (
    WeakValueDictionary()
)
_PREPARED_SEALS: dict[int, str] = {}


def _forget_prepared(identity: int) -> None:
    with _PREPARED_LOCK:
        _PREPARED_VALUES.pop(identity, None)
        _PREPARED_SEALS.pop(identity, None)


def _meta_authorization_input_sha256(
    plan: PolicyFunctionalMetaStudyPlan,
    protocol: StudyProtocol,
) -> str:
    return _digest_mapping(
        {
            "plan_sha256": plan.sha256,
            "protocol_sha256": protocol.sha256,
            "schema_version": 1,
        }
    )


def _prepared_seal(
    plan: PolicyFunctionalMetaStudyPlan,
    protocol: StudyProtocol,
    execution_started_at: datetime,
    protocol_authorization: ProtocolExecutionAuthorization,
) -> str:
    return _digest_mapping(
        {
            "authorization_receipt_sha256": (protocol_authorization.receipt.sha256),
            "execution_started_at": execution_started_at.isoformat(),
            "plan_sha256": plan.sha256,
            "protocol_sha256": protocol.sha256,
            "schema_version": 1,
        }
    )


def authorize_policy_functional_meta_study(
    *,
    plan: PolicyFunctionalMetaStudyPlan,
    protocol: StudyProtocol,
    execution_started_at: datetime,
) -> PreparedPolicyFunctionalMetaStudy:
    """Mint one local capability bound to the exact pre-execution plan."""
    if type(plan) is not PolicyFunctionalMetaStudyPlan:
        raise TypeError("plan must be a factory-produced PolicyFunctionalMetaStudyPlan")
    plan._fresh_validate()
    if type(protocol) is not StudyProtocol:
        raise TypeError("protocol must be an exact StudyProtocol")
    if (
        protocol.mode is not StudyMode.SYNTHETIC
        or protocol.claim_scope is not ClaimScope.SYNTHETIC_ONLY
    ):
        _fail(
            "meta-study accepts only a synthetic-only protocol",
            code="protocol_scope_mismatch",
        )
    if (
        protocol.knowledge_cutoff != plan.market_config.cutoff
        or protocol.randomness_protocol is not RandomnessProtocol.EVENT_KEYED_PAIRED
    ):
        _fail(
            "protocol cutoff or randomness differs from the plan",
            code="protocol_binding_mismatch",
        )
    if plan.frozen_definition not in protocol.evidence_plans:
        _fail(
            "protocol does not bind the exact meta-study plan",
            code="protocol_binding_mismatch",
        )
    started = _aware(execution_started_at, "execution_started_at")
    if (
        started < protocol.registered_at
        or started >= plan.outcomes_available_at
        or started >= protocol.evaluation_reveal_at
    ):
        _fail(
            "meta-study execution lies outside the pre-outcome local chronology",
            code="invalid_chronology",
        )
    protocol_authorization = SyntheticProtocolExecutionIssuer(protocol).issue(
        execution_id=f"{plan.plan_id}-meta-study",
        consumer=ProtocolExecutionConsumer.POLICY_FUNCTIONAL_META_EVALUATION,
        purpose=(ProtocolExecutionPurpose.SYNTHETIC_POLICY_FUNCTIONAL_META_EVALUATION),
        config_sha256=plan.sha256,
        input_sha256=_meta_authorization_input_sha256(plan, protocol),
        execution_started_at=started,
    )
    prepared = object.__new__(PreparedPolicyFunctionalMetaStudy)
    object.__setattr__(prepared, "plan", plan)
    object.__setattr__(prepared, "protocol", protocol)
    object.__setattr__(prepared, "execution_started_at", started)
    object.__setattr__(prepared, "protocol_authorization", protocol_authorization)
    identity = id(prepared)
    with _PREPARED_LOCK:
        _PREPARED_VALUES[identity] = prepared
        _PREPARED_SEALS[identity] = _prepared_seal(
            plan,
            protocol,
            started,
            protocol_authorization,
        )
    finalize(prepared, _forget_prepared, identity)
    return prepared


def _consume_prepared(
    prepared: PreparedPolicyFunctionalMetaStudy,
) -> tuple[
    PolicyFunctionalMetaStudyPlan,
    StudyProtocol,
    datetime,
    ProtocolExecutionReceipt,
]:
    if type(prepared) is not PreparedPolicyFunctionalMetaStudy:
        _fail("meta-study capability has the wrong type", code="invalid_authority")
    identity = id(prepared)
    with _PREPARED_LOCK:
        registered = _PREPARED_VALUES.pop(identity, None)
        seal = _PREPARED_SEALS.pop(identity, None)
    if registered is not prepared or seal is None:
        _fail(
            "meta-study capability was forged or already consumed",
            code="invalid_authority",
        )
    plan = prepared.plan
    protocol = prepared.protocol
    started = prepared.execution_started_at
    protocol_authorization = prepared.protocol_authorization
    if seal != _prepared_seal(
        plan,
        protocol,
        started,
        protocol_authorization,
    ):
        _fail(
            "meta-study capability differs from its mint binding",
            code="invalid_authority",
        )
    plan._fresh_validate()
    receipt = consume_protocol_execution_authorization(
        protocol_authorization,
        protocol=protocol,
        consumer=ProtocolExecutionConsumer.POLICY_FUNCTIONAL_META_EVALUATION,
        purpose=(ProtocolExecutionPurpose.SYNTHETIC_POLICY_FUNCTIONAL_META_EVALUATION),
        config_sha256=plan.sha256,
        input_sha256=_meta_authorization_input_sha256(plan, protocol),
        execution_started_at=started,
    )
    return plan, protocol, started, receipt


def _scenario_event_key(
    *,
    plan_id: str,
    seed: str,
    meta_replicate_index: int,
    scenario_replicate: int,
    scenario_replicate_count: int,
) -> EventKey:
    return EventKey(
        experiment_id=plan_id,
        seed=seed,
        replicate_id=(
            meta_replicate_index * scenario_replicate_count + scenario_replicate
        ),
        simulation_step=0,
        purpose="policy-functional-meta-shared-scenario",
    )


def _oracle_audit_event_key(plan_id: str, replicate: int) -> str:
    return EventKey(
        experiment_id=plan_id,
        seed="policy-functional-meta-oracle-audit-generation-v1",
        replicate_id=replicate,
        simulation_step=0,
        purpose="policy-functional-meta-oracle-audit-generation",
    ).value


def build_policy_functional_meta_study_plan(
    *,
    plan_id: str,
    frozen_at: datetime,
    outcomes_available_at: datetime,
    calibration_configs: tuple[CalibrationConfig, ...],
    calibration_simulator: IntegratedSyntheticCalibrationAdapter,
    market_config: IntegratedSyntheticMarketConfig,
    event_key_seed: str,
    scenario_replicate_count: int,
    primary_output: LedgerPosteriorOutputSpec,
    threshold: CallerDeclaredRankThreshold,
    parameter_sbc_design_sha256: str,
    randomized_tie_seed: str,
    candidate_allow_buy: bool,
    candidate_allow_sell: bool,
    candidate_is_noop: bool,
    step_count: int,
    include_all_groups: bool,
) -> PolicyFunctionalMetaStudyPlan:
    """Build the complete local plan without inspecting policy outcomes."""
    _identifier(plan_id, "plan_id")
    if not isinstance(calibration_configs, tuple) or not calibration_configs:
        _fail("calibration_configs must be a non-empty tuple")
    if not all(type(item) is CalibrationConfig for item in calibration_configs):
        _fail("calibration_configs contain an invalid type")
    posterior_counts = {item.posterior_draws for item in calibration_configs}
    if len(posterior_counts) != 1:
        _fail("all meta-replicates must retain the same posterior draw count")
    posterior_draw_count = next(iter(posterior_counts))
    if posterior_draw_count < 2:
        _fail("policy-functional ranks require at least two posterior draws")
    _positive(scenario_replicate_count, "scenario_replicate_count")
    if type(primary_output) is not LedgerPosteriorOutputSpec:
        raise TypeError("primary_output must be an exact LedgerPosteriorOutputSpec")
    primary_output_definition_sha256 = _digest_mapping(primary_output.to_mapping())
    core_spec = PolicyFunctionalSBCSpec(
        spec_id=plan_id,
        meta_replicate_count=len(calibration_configs),
        posterior_draw_count=posterior_draw_count,
        scenario_replicate_count=scenario_replicate_count,
        primary_output_id=primary_output.output_id,
        primary_output_definition_sha256=primary_output_definition_sha256,
        threshold=threshold,
        oracle_audit_generation_event_keys=tuple(
            _oracle_audit_event_key(plan_id, replicate)
            for replicate in range(len(calibration_configs))
        ),
        parameter_sbc_sha256=_sha256(
            parameter_sbc_design_sha256,
            "parameter_sbc_design_sha256",
        ),
        randomized_tie_seed=randomized_tie_seed,
    )
    scenario_keys = tuple(
        _scenario_event_key(
            plan_id=plan_id,
            seed=event_key_seed,
            meta_replicate_index=replicate,
            scenario_replicate=scenario,
            scenario_replicate_count=scenario_replicate_count,
        ).value
        for replicate in range(len(calibration_configs))
        for scenario in range(scenario_replicate_count)
    )
    first = calibration_configs[0]
    template_parameters = {
        prior.name: (prior.lower + prior.upper) / 2.0 for prior in first.priors
    }
    template_pair = integrated_synthetic_pair(
        event_key=_scenario_event_key(
            plan_id=plan_id,
            seed=event_key_seed,
            meta_replicate_index=0,
            scenario_replicate=0,
            scenario_replicate_count=scenario_replicate_count,
        ),
        parameters=template_parameters,
        config=market_config,
        candidate_allow_buy=candidate_allow_buy,
        candidate_allow_sell=candidate_allow_sell,
        candidate_is_noop=candidate_is_noop,
        step_count=step_count,
        include_all_groups=include_all_groups,
    )
    plan = object.__new__(PolicyFunctionalMetaStudyPlan)
    values: tuple[tuple[str, object], ...] = (
        ("plan_id", plan_id),
        ("frozen_at", _aware(frozen_at, "frozen_at")),
        (
            "outcomes_available_at",
            _aware(outcomes_available_at, "outcomes_available_at"),
        ),
        ("calibration_configs", calibration_configs),
        ("calibration_simulator", calibration_simulator),
        ("market_config", market_config),
        ("event_key_seed", event_key_seed),
        ("scenario_replicate_count", scenario_replicate_count),
        ("primary_output", primary_output),
        ("core_spec", core_spec),
        ("candidate_allow_buy", candidate_allow_buy),
        ("candidate_allow_sell", candidate_allow_sell),
        ("candidate_is_noop", candidate_is_noop),
        ("step_count", step_count),
        ("include_all_groups", include_all_groups),
        ("scenario_event_keys", scenario_keys),
        ("pairs", (template_pair,)),
        ("claim_scope", POLICY_FUNCTIONAL_META_STUDY_CLAIM_SCOPE),
        ("schema_version", POLICY_FUNCTIONAL_META_STUDY_SCHEMA_VERSION),
    )
    for field, value in values:
        object.__setattr__(plan, field, value)
    plan._fresh_validate()
    return plan


def _calibration_protocol(
    source: StudyProtocol,
    *,
    config: CalibrationConfig,
    plan: PolicyFunctionalMetaStudyPlan,
) -> StudyProtocol:
    return replace(
        source,
        study_id=config.study_id,
        knowledge_cutoff=config.cutoff,
        calibration_statistics=tuple(
            item.name for item in config.summaries.calibration
        ),
        heldout_validation_statistics=tuple(
            item.name for item in config.summaries.held_out
        ),
        evidence_plans=(plan.frozen_definition,),
    )


def _calibration_receipt_from_run(
    *,
    plan: PolicyFunctionalMetaStudyPlan,
    source_protocol: StudyProtocol,
    execution_started_at: datetime,
    meta_replicate_index: int,
    run: CalibrationRun,
) -> PolicyFunctionalCalibrationReceipt:
    """Replay and bind one retained calibration run to its planned prior cell."""
    config = plan.calibration_configs[meta_replicate_index]
    if type(run) is not CalibrationRun or run.config != config:
        _fail(
            "calibration run differs from its frozen meta-replicate config",
            code="calibration_config_mismatch",
        )
    try:
        replayed = replace(run)
    except (CalibrationError, TypeError, ValueError) as error:
        raise PolicyFunctionalMetaStudyError(
            "calibration run failed its sealed synthetic replay",
            code="calibration_replay_mismatch",
        ) from error
    expected_truth, prior_keys = sample_prior(
        config,
        replicate=0,
        namespace="policy_functional_meta:truth",
    )
    if dict(replayed.true_parameters) != expected_truth:
        _fail(
            "calibration truth does not replay from the frozen synthetic prior key",
            code="prior_truth_mismatch",
        )
    expected_protocol = _calibration_protocol(
        source_protocol,
        config=config,
        plan=plan,
    )
    protocol_receipt = replayed.protocol_execution_receipt
    if (
        protocol_receipt.protocol_sha256 != expected_protocol.sha256
        or protocol_receipt.config_sha256 != config.sha256
        or protocol_receipt.input_sha256 != replayed.observed.sha256
        or protocol_receipt.execution_started_at
        != _aware(execution_started_at, "execution_started_at")
    ):
        _fail(
            "calibration protocol receipt differs from the frozen input binding",
            code="calibration_input_mismatch",
        )
    posterior_draw_sha256s = tuple(
        _posterior_draw_sha256(draw) for draw in replayed.posterior.draws
    )
    posterior_parameter_sha256s = tuple(
        _parameter_sha256(dict(draw.parameters)) for draw in replayed.posterior.draws
    )
    return PolicyFunctionalCalibrationReceipt(
        meta_replicate_index=meta_replicate_index,
        config_sha256=config.sha256,
        input_sha256=replayed.observed.sha256,
        truth_parameter_sha256=replayed.true_parameters_sha256,
        posterior_sha256=replayed.posterior.sha256,
        posterior_draw_sha256s=posterior_draw_sha256s,
        posterior_parameter_sha256s=posterior_parameter_sha256s,
        parameter_sbc_sha256=replayed.sbc.sha256,
        report_sha256=replayed.report.sha256,
        protocol_receipt_sha256=protocol_receipt.sha256,
        prior_event_keys_sha256=sha256_bytes(canonical_json_bytes(list(prior_keys))),
        prior_event_key_count=len(prior_keys),
        event_keys_sha256=sha256_bytes(canonical_json_bytes(list(replayed.event_keys))),
        event_key_count=len(replayed.event_keys),
    )


def _run_one_calibration(
    *,
    plan: PolicyFunctionalMetaStudyPlan,
    source_protocol: StudyProtocol,
    execution_started_at: datetime,
    meta_replicate_index: int,
) -> tuple[CalibrationRun, PolicyFunctionalCalibrationReceipt]:
    config = plan.calibration_configs[meta_replicate_index]
    truth, prior_keys = sample_prior(
        config,
        replicate=0,
        namespace="policy_functional_meta:truth",
    )
    calibration_key = calibration_event_key(
        config,
        replicate=0,
        step=0,
        purpose="synthetic_observed:calibration",
    )
    heldout_key = calibration_event_key(
        config,
        replicate=0,
        step=1,
        purpose="synthetic_observed:held_out",
    )
    observed: CalibrationInput = synthetic_calibration_input(
        config,
        plan.calibration_simulator.simulate(truth, event_key=calibration_key),
        plan.calibration_simulator.simulate(truth, event_key=heldout_key),
        calibration_generation_event_key=calibration_key.value,
        held_out_generation_event_key=heldout_key.value,
    )
    calibration_protocol = _calibration_protocol(
        source_protocol,
        config=config,
        plan=plan,
    )
    issuer = SyntheticProtocolExecutionIssuer(calibration_protocol)
    authorization = issuer.issue(
        execution_id=f"{plan.plan_id}-calibration-{meta_replicate_index}",
        consumer=ProtocolExecutionConsumer.CALIBRATION_WORKFLOW,
        purpose=ProtocolExecutionPurpose.PRE_POLICY_CALIBRATION,
        config_sha256=config.sha256,
        input_sha256=observed.sha256,
        execution_started_at=execution_started_at,
    )
    run = run_synthetic_calibration(
        config=config,
        observed=observed,
        simulator=plan.calibration_simulator,
        true_parameters=truth,
        study_protocol=calibration_protocol,
        protocol_authorization=authorization,
        execution_started_at=execution_started_at,
    )
    # The receipt helper freshly replays the complete sealed built-in closure.
    if not prior_keys:
        _fail("synthetic prior did not emit explicit EventKeys")
    receipt = _calibration_receipt_from_run(
        plan=plan,
        source_protocol=source_protocol,
        execution_started_at=execution_started_at,
        meta_replicate_index=meta_replicate_index,
        run=run,
    )
    return run, receipt


@dataclass(frozen=True, slots=True)
class _CellTask:
    meta_replicate_index: int
    role: PolicyFunctionalCellRole
    posterior_index: int | None
    scenario_replicate: int
    pair: PolicyWorldPair
    parameter_sha256: str
    posterior_draw_sha256: str | None

    @property
    def cell_id(self) -> str:
        posterior = (
            "oracle"
            if self.posterior_index is None
            else f"posterior-{self.posterior_index}"
        )
        return (
            f"b{self.meta_replicate_index:04d}-{posterior}-"
            f"r{self.scenario_replicate:04d}"
        )


def _parameter_sha256(parameters: Mapping[str, float]) -> str:
    return _digest_mapping({"parameters": dict(sorted(parameters.items()))})


def _posterior_draw_sha256(draw: object) -> str:
    mapping = draw.to_mapping()  # type: ignore[attr-defined]
    return _digest_mapping(mapping)


def _cell_tasks(
    plan: PolicyFunctionalMetaStudyPlan,
    calibration_runs: Mapping[int, CalibrationRun],
) -> tuple[_CellTask, ...]:
    tasks: list[_CellTask] = []
    for meta_replicate_index, run in sorted(calibration_runs.items()):
        rows: list[
            tuple[
                PolicyFunctionalCellRole,
                int | None,
                dict[str, float],
                str | None,
                str,
            ]
        ] = [
            (
                PolicyFunctionalCellRole.ORACLE,
                None,
                dict(run.true_parameters),
                None,
                run.true_parameters_sha256,
            )
        ]
        rows.extend(
            (
                PolicyFunctionalCellRole.POSTERIOR,
                posterior_index,
                dict(draw.parameters),
                _posterior_draw_sha256(draw),
                _parameter_sha256(dict(draw.parameters)),
            )
            for posterior_index, draw in enumerate(run.posterior.draws)
        )
        for (
            role,
            posterior_index,
            parameters,
            draw_sha256,
            parameter_sha256,
        ) in rows:
            for scenario_replicate in range(plan.scenario_replicate_count):
                event_key = _scenario_event_key(
                    plan_id=plan.plan_id,
                    seed=plan.event_key_seed,
                    meta_replicate_index=meta_replicate_index,
                    scenario_replicate=scenario_replicate,
                    scenario_replicate_count=plan.scenario_replicate_count,
                )
                pair = integrated_synthetic_pair(
                    event_key=event_key,
                    parameters=parameters,
                    config=plan.market_config,
                    candidate_allow_buy=plan.candidate_allow_buy,
                    candidate_allow_sell=plan.candidate_allow_sell,
                    candidate_is_noop=plan.candidate_is_noop,
                    step_count=plan.step_count,
                    include_all_groups=plan.include_all_groups,
                )
                tasks.append(
                    _CellTask(
                        meta_replicate_index=meta_replicate_index,
                        role=role,
                        posterior_index=posterior_index,
                        scenario_replicate=scenario_replicate,
                        pair=pair,
                        parameter_sha256=parameter_sha256,
                        posterior_draw_sha256=draw_sha256,
                    )
                )
    return tuple(
        sorted(
            tasks,
            key=lambda item: (
                item.meta_replicate_index,
                item.role.value,
                -1 if item.posterior_index is None else item.posterior_index,
                item.scenario_replicate,
            ),
        )
    )


def _selected_difference(
    evidence: LedgerDerivedEvaluation,
    output: LedgerPosteriorOutputSpec,
) -> ExactRational:
    matches = tuple(
        item
        for item in evidence.differences
        if item.horizon_id == output.horizon_id
        and item.scope is output.scope
        and item.group_id == output.group_id
        and item.metric_id == output.metric_id
    )
    if len(matches) != 1:
        _fail(
            "primary output does not resolve to exactly one ledger difference",
            code="primary_output_resolution_failure",
        )
    metric = matches[0].difference
    if metric.state is not ExactMetricState.DEFINED or metric.value is None:
        _fail(
            "primary policy functional is undefined",
            code="primary_output_undefined",
        )
    return metric.value


def _run_integrated_cell(
    *,
    task: _CellTask,
    plan: PolicyFunctionalMetaStudyPlan,
    protocol: StudyProtocol,
    issuer: SyntheticProtocolExecutionIssuer,
    execution_started_at: datetime,
    parallel_worlds: bool,
) -> PolicyFunctionalCellAttempt:
    """Execute one source cell through the protocol-bound integrated runner."""
    input_sha256 = integrated_synthetic_pair_input_sha256(task.pair)
    authorization = issuer.issue(
        execution_id=f"{plan.plan_id}-{task.cell_id}",
        consumer=ProtocolExecutionConsumer.INTEGRATED_SYNTHETIC_SIMULATION,
        purpose=ProtocolExecutionPurpose.PAIRED_SYNTHETIC_POLICY_SIMULATION,
        config_sha256=plan.market_config.sha256,
        input_sha256=input_sha256,
        execution_started_at=execution_started_at,
    )
    artifact: ProtocolBoundIntegratedSyntheticRunArtifact = (
        run_protocol_bound_integrated_synthetic_pair(
            pair=task.pair,
            config=plan.market_config,
            protocol=protocol,
            authorization=authorization,
            execution_started_at=execution_started_at,
            parallel=parallel_worlds,
        )
    )
    evidence = derive_protocol_bound_ledger_evidence(artifact)
    value = _selected_difference(evidence, plan.primary_output)
    baseline_ledger = artifact.paired_outcome.baseline.shock_consumption_ledger
    candidate_ledger = artifact.paired_outcome.candidate.shock_consumption_ledger
    if baseline_ledger is None or candidate_ledger is None:
        _fail(
            "integrated cell omitted engine-owned shock ledgers",
            code="missing_rng_receipt",
        )
    tape_sha256 = task.pair.shock_tape.sha256
    if (
        baseline_ledger.shock_tape_sha256 != tape_sha256
        or candidate_ledger.shock_tape_sha256 != tape_sha256
        or baseline_ledger.declared_event_keys != candidate_ledger.declared_event_keys
        or tuple(item.event_key.value for item in baseline_ledger.consumptions)
        != tuple(item.event_key.value for item in candidate_ledger.consumptions)
    ):
        _fail(
            "paired worlds did not retain the same exogenous shock consumption",
            code="rng_mismatch",
        )
    return PolicyFunctionalCellAttempt(
        meta_replicate_index=task.meta_replicate_index,
        role=task.role,
        posterior_index=task.posterior_index,
        scenario_replicate=task.scenario_replicate,
        cell_id=task.cell_id,
        evaluation_event_key=_scenario_event_key(
            plan_id=plan.plan_id,
            seed=plan.event_key_seed,
            meta_replicate_index=task.meta_replicate_index,
            scenario_replicate=task.scenario_replicate,
            scenario_replicate_count=plan.scenario_replicate_count,
        ).value,
        parameter_sha256=task.parameter_sha256,
        posterior_draw_sha256=task.posterior_draw_sha256,
        scenario_commitment_sha256=tape_sha256,
        status=PolicyFunctionalCellStatus.COMPLETED,
        value=value,
        source_artifact_sha256=artifact.sha256,
        ledger_evidence_sha256=evidence.sha256,
        baseline_shock_ledger_sha256=baseline_ledger.sha256,
        candidate_shock_ledger_sha256=candidate_ledger.sha256,
        mechanism_config_sha256=plan.market_config.sha256,
    )


def _failed_attempt(
    *,
    task: _CellTask,
    plan: PolicyFunctionalMetaStudyPlan,
    error: Exception,
) -> PolicyFunctionalCellAttempt:
    reason = f"{type(error).__name__}: {error}"
    candidate_code = getattr(error, "code", type(error).__name__)
    failure_code = (
        candidate_code
        if isinstance(candidate_code, str)
        and _IDENTIFIER.fullmatch(candidate_code) is not None
        else "unclassified_failure"
    )
    return PolicyFunctionalCellAttempt(
        meta_replicate_index=task.meta_replicate_index,
        role=task.role,
        posterior_index=task.posterior_index,
        scenario_replicate=task.scenario_replicate,
        cell_id=task.cell_id,
        evaluation_event_key=_scenario_event_key(
            plan_id=plan.plan_id,
            seed=plan.event_key_seed,
            meta_replicate_index=task.meta_replicate_index,
            scenario_replicate=task.scenario_replicate,
            scenario_replicate_count=plan.scenario_replicate_count,
        ).value,
        parameter_sha256=task.parameter_sha256,
        posterior_draw_sha256=task.posterior_draw_sha256,
        scenario_commitment_sha256=task.pair.shock_tape.sha256,
        status=PolicyFunctionalCellStatus.FAILED,
        value=None,
        source_artifact_sha256=None,
        ledger_evidence_sha256=None,
        baseline_shock_ledger_sha256=None,
        candidate_shock_ledger_sha256=None,
        mechanism_config_sha256=plan.market_config.sha256,
        failure_code=failure_code,
        failure_reason=reason,
        failure_evidence_sha256=_digest_mapping(
            {
                "cell_id": task.cell_id,
                "error_type": type(error).__name__,
                "reason": str(error),
            }
        ),
    )


def _dependency_attempts(
    *,
    plan: PolicyFunctionalMetaStudyPlan,
    meta_replicate_index: int,
    error: Exception,
) -> tuple[PolicyFunctionalCellAttempt, ...]:
    result: list[PolicyFunctionalCellAttempt] = []
    reason = (
        f"pre-policy calibration dependency failed: {type(error).__name__}: {error}"
    )
    candidate_code = getattr(error, "code", type(error).__name__)
    failure_code = (
        candidate_code
        if isinstance(candidate_code, str)
        and _IDENTIFIER.fullmatch(candidate_code) is not None
        else "calibration_dependency_failure"
    )
    for role, posterior_index in (
        (PolicyFunctionalCellRole.ORACLE, None),
        *tuple(
            (PolicyFunctionalCellRole.POSTERIOR, index)
            for index in range(plan.posterior_draw_count)
        ),
    ):
        posterior_sha256 = (
            None
            if posterior_index is None
            else _digest_mapping(
                {
                    "meta_replicate_index": meta_replicate_index,
                    "posterior_index": posterior_index,
                    "state": "unavailable_calibration_dependency",
                }
            )
        )
        parameter_sha256 = _digest_mapping(
            {
                "meta_replicate_index": meta_replicate_index,
                "posterior_index": posterior_index,
                "role": role.value,
                "state": "unavailable_calibration_dependency",
            }
        )
        for scenario_replicate in range(plan.scenario_replicate_count):
            event_key = _scenario_event_key(
                plan_id=plan.plan_id,
                seed=plan.event_key_seed,
                meta_replicate_index=meta_replicate_index,
                scenario_replicate=scenario_replicate,
                scenario_replicate_count=plan.scenario_replicate_count,
            )
            posterior_label = (
                "oracle" if posterior_index is None else f"posterior-{posterior_index}"
            )
            cell_id = (
                f"b{meta_replicate_index:04d}-{posterior_label}-"
                f"r{scenario_replicate:04d}"
            )
            scenario_commitment = _digest_mapping(
                {
                    "event_key": event_key.value,
                    "state": "unexecuted_calibration_dependency",
                }
            )
            result.append(
                PolicyFunctionalCellAttempt(
                    meta_replicate_index=meta_replicate_index,
                    role=role,
                    posterior_index=posterior_index,
                    scenario_replicate=scenario_replicate,
                    cell_id=cell_id,
                    evaluation_event_key=event_key.value,
                    parameter_sha256=parameter_sha256,
                    posterior_draw_sha256=posterior_sha256,
                    scenario_commitment_sha256=scenario_commitment,
                    status=PolicyFunctionalCellStatus.ABSTAINED_DEPENDENCY,
                    value=None,
                    source_artifact_sha256=None,
                    ledger_evidence_sha256=None,
                    baseline_shock_ledger_sha256=None,
                    candidate_shock_ledger_sha256=None,
                    mechanism_config_sha256=plan.market_config.sha256,
                    failure_code=failure_code,
                    failure_reason=reason,
                    failure_evidence_sha256=_digest_mapping(
                        {
                            "cell_id": cell_id,
                            "dependency_error": str(error),
                            "dependency_type": type(error).__name__,
                        }
                    ),
                )
            )
    return tuple(result)


def _expected_coordinates(
    plan: PolicyFunctionalMetaStudyPlan,
) -> set[tuple[int, PolicyFunctionalCellRole, int | None, int]]:
    return {
        (meta, role, posterior, scenario)
        for meta in range(plan.meta_replicate_count)
        for role, posterior in (
            (PolicyFunctionalCellRole.ORACLE, None),
            *tuple(
                (PolicyFunctionalCellRole.POSTERIOR, index)
                for index in range(plan.posterior_draw_count)
            ),
        )
        for scenario in range(plan.scenario_replicate_count)
    }


def _validate_calibration_closure(
    *,
    plan: PolicyFunctionalMetaStudyPlan,
    protocol: StudyProtocol,
    execution_started_at: datetime,
    calibration_runs: tuple[CalibrationRun | None, ...],
    receipts: tuple[PolicyFunctionalCalibrationReceipt, ...],
) -> dict[int, PolicyFunctionalCalibrationReceipt]:
    if (
        not isinstance(calibration_runs, tuple)
        or len(calibration_runs) != plan.meta_replicate_count
        or not all(
            item is None or type(item) is CalibrationRun for item in calibration_runs
        )
    ):
        _fail(
            "calibration run ledger does not cover the frozen B axis",
            code="incomplete_calibration_ledger",
        )
    if not isinstance(receipts, tuple) or not all(
        type(item) is PolicyFunctionalCalibrationReceipt for item in receipts
    ):
        _fail("calibration receipt ledger has the wrong type")
    receipt_by_index = {item.meta_replicate_index: item for item in receipts}
    if len(receipt_by_index) != len(receipts):
        _fail(
            "calibration receipt ledger contains duplicate meta-replicates",
            code="duplicate_calibration_receipt",
        )
    successful_indices = {
        index for index, run in enumerate(calibration_runs) if run is not None
    }
    if set(receipt_by_index) != successful_indices:
        _fail(
            "calibration receipts do not exactly match successful runs",
            code="incomplete_calibration_ledger",
        )
    if tuple(receipts) != tuple(
        sorted(receipts, key=lambda item: item.meta_replicate_index)
    ):
        _fail(
            "calibration receipt ledger is not canonical",
            code="reordered_calibration_receipt",
        )
    for index in sorted(successful_indices):
        run = calibration_runs[index]
        if run is None:  # pragma: no cover - narrowed by successful_indices
            _fail("successful calibration index unexpectedly lacks a run")
        expected = _calibration_receipt_from_run(
            plan=plan,
            source_protocol=protocol,
            execution_started_at=execution_started_at,
            meta_replicate_index=index,
            run=run,
        )
        if receipt_by_index[index] != expected:
            code = (
                "calibration_input_mismatch"
                if receipt_by_index[index].input_sha256 != expected.input_sha256
                else "calibration_receipt_mismatch"
            )
            _fail(
                "calibration receipt differs from fresh sealed replay",
                code=code,
            )
    return receipt_by_index


def _validate_attempt_ledger(
    plan: PolicyFunctionalMetaStudyPlan,
    attempts: tuple[PolicyFunctionalCellAttempt, ...],
    *,
    calibration_runs: tuple[CalibrationRun | None, ...],
    calibration_receipts: tuple[PolicyFunctionalCalibrationReceipt, ...],
) -> None:
    if not isinstance(attempts, tuple) or not all(
        type(item) is PolicyFunctionalCellAttempt for item in attempts
    ):
        _fail("attempt ledger has the wrong type", code="invalid_attempt_ledger")
    for item in attempts:
        item._fresh_validate()
    coordinates = tuple(
        (
            item.meta_replicate_index,
            item.role,
            item.posterior_index,
            item.scenario_replicate,
        )
        for item in attempts
    )
    expected = _expected_coordinates(plan)
    if len(attempts) != plan.expected_cell_count or set(coordinates) != expected:
        _fail(
            "attempt ledger does not exactly cover B*(M+1)*R",
            code="incomplete_attempt_ledger",
        )
    if len(set(coordinates)) != len(coordinates) or len(
        {item.cell_id for item in attempts}
    ) != len(attempts):
        _fail(
            "attempt ledger contains duplicate coordinates or identities",
            code="duplicate_attempt_cell",
        )
    expected_order = tuple(
        sorted(
            attempts,
            key=lambda item: (
                item.meta_replicate_index,
                item.role.value,
                -1 if item.posterior_index is None else item.posterior_index,
                item.scenario_replicate,
            ),
        )
    )
    if attempts != expected_order:
        _fail("attempt ledger is not canonical", code="reordered_attempt_ledger")
    run_mapping = {
        index: run for index, run in enumerate(calibration_runs) if run is not None
    }
    task_mapping = {
        (
            task.meta_replicate_index,
            task.role,
            task.posterior_index,
            task.scenario_replicate,
        ): task
        for task in _cell_tasks(plan, run_mapping)
    }
    receipt_mapping = {item.meta_replicate_index: item for item in calibration_receipts}
    for item in attempts:
        coordinate = (
            item.meta_replicate_index,
            item.role,
            item.posterior_index,
            item.scenario_replicate,
        )
        expected_key = _scenario_event_key(
            plan_id=plan.plan_id,
            seed=plan.event_key_seed,
            meta_replicate_index=item.meta_replicate_index,
            scenario_replicate=item.scenario_replicate,
            scenario_replicate_count=plan.scenario_replicate_count,
        ).value
        posterior_label = (
            "oracle"
            if item.posterior_index is None
            else f"posterior-{item.posterior_index}"
        )
        expected_cell_id = (
            f"b{item.meta_replicate_index:04d}-{posterior_label}-"
            f"r{item.scenario_replicate:04d}"
        )
        if (
            item.cell_id != expected_cell_id
            or item.evaluation_event_key != expected_key
        ):
            _fail(
                "cell identity differs from the frozen coordinate",
                code="cell_identity_mismatch",
            )
        _require_digest_binding(
            expected=plan.market_config.sha256,
            observed=item.mechanism_config_sha256,
            code="mechanism_mismatch",
            message="cell mechanism config differs from the frozen market config",
        )
        task = task_mapping.get(coordinate)
        if task is None:
            if item.meta_replicate_index in receipt_mapping:
                _fail(
                    "successful calibration is missing its planned cell task",
                    code="incomplete_attempt_ledger",
                )
            expected_commitment = _digest_mapping(
                {
                    "event_key": expected_key,
                    "state": "unexecuted_calibration_dependency",
                }
            )
            if (
                item.status is not PolicyFunctionalCellStatus.ABSTAINED_DEPENDENCY
                or item.scenario_commitment_sha256 != expected_commitment
            ):
                _fail(
                    "failed calibration dependency was relabelled as a cell result",
                    code="calibration_dependency_mismatch",
                )
            continue
        _require_digest_binding(
            expected=task.parameter_sha256,
            observed=item.parameter_sha256,
            code="posterior_binding_mismatch",
            message="cell parameter identity differs from the retained posterior",
        )
        if item.posterior_draw_sha256 != task.posterior_draw_sha256:
            _fail(
                "cell posterior draw identity differs from the retained posterior",
                code="posterior_binding_mismatch",
            )
        _require_digest_binding(
            expected=task.pair.shock_tape.sha256,
            observed=item.scenario_commitment_sha256,
            code="rng_mismatch",
            message=("cell scenario commitment differs from the engine-owned tape"),
        )
        if item.status is PolicyFunctionalCellStatus.ABSTAINED_DEPENDENCY:
            _fail(
                "successful calibration cell cannot claim dependency abstention",
                code="calibration_dependency_mismatch",
            )
    for meta in range(plan.meta_replicate_count):
        for scenario in range(plan.scenario_replicate_count):
            column = tuple(
                item
                for item in attempts
                if item.meta_replicate_index == meta
                and item.scenario_replicate == scenario
            )
            if len({item.scenario_commitment_sha256 for item in column}) != 1:
                _fail(
                    "oracle and posterior rows do not share the scenario tape",
                    code="rng_mismatch",
                )
            if len({item.evaluation_event_key for item in column}) != 1:
                _fail(
                    "scenario column changed its EventKey across rows",
                    code="rng_mismatch",
                )
    for meta in range(plan.meta_replicate_count):
        for role, posterior in (
            (PolicyFunctionalCellRole.ORACLE, None),
            *tuple(
                (PolicyFunctionalCellRole.POSTERIOR, index)
                for index in range(plan.posterior_draw_count)
            ),
        ):
            row = tuple(
                item
                for item in attempts
                if item.meta_replicate_index == meta
                and item.role is role
                and item.posterior_index == posterior
            )
            if (
                len({item.parameter_sha256 for item in row}) != 1
                or len({item.posterior_draw_sha256 for item in row}) != 1
            ):
                _fail(
                    "one row changed parameter or posterior identity across scenarios",
                    code="posterior_binding_mismatch",
                )


def _all_completed_rng_receipts(
    attempts: tuple[PolicyFunctionalCellAttempt, ...],
) -> bool:
    return all(
        item.status is PolicyFunctionalCellStatus.COMPLETED
        and item.baseline_shock_ledger_sha256 is not None
        and item.candidate_shock_ledger_sha256 is not None
        for item in attempts
    )


def _attempt_source_evidence(
    attempts: tuple[PolicyFunctionalCellAttempt, ...],
) -> str:
    return _digest_mapping(
        {
            "attempt_sha256s": [item.sha256 for item in attempts],
            "schema_version": 1,
        }
    )


def _project_meta_replicates(
    *,
    plan: PolicyFunctionalMetaStudyPlan,
    attempts: tuple[PolicyFunctionalCellAttempt, ...],
    receipts: tuple[PolicyFunctionalCalibrationReceipt, ...],
) -> tuple[PolicyFunctionalMetaReplicate, ...]:
    receipt_by_index = {item.meta_replicate_index: item for item in receipts}
    projected: list[PolicyFunctionalMetaReplicate] = []
    for meta_index in range(plan.meta_replicate_count):
        group = tuple(
            item for item in attempts if item.meta_replicate_index == meta_index
        )
        incomplete = tuple(
            item
            for item in group
            if item.status is not PolicyFunctionalCellStatus.COMPLETED
        )
        if incomplete:
            dependency = any(
                item.status is PolicyFunctionalCellStatus.ABSTAINED_DEPENDENCY
                for item in incomplete
            )
            failed_oracle = any(
                item.role is PolicyFunctionalCellRole.ORACLE
                and item.status is PolicyFunctionalCellStatus.FAILED
                for item in incomplete
            )
            stage = (
                MetaReplicateFailureStage.PRE_POLICY_CALIBRATION
                if dependency
                else (
                    MetaReplicateFailureStage.SCENARIO_TAPE_INTEGRITY
                    if any(item.failure_code == "rng_mismatch" for item in incomplete)
                    else (
                        MetaReplicateFailureStage.ORACLE_FUNCTIONAL_INPUT
                        if failed_oracle
                        else MetaReplicateFailureStage.POSTERIOR_FUNCTIONAL_INPUT
                    )
                )
            )
            status = (
                MetaReplicateStatus.FAILED
                if any(
                    item.status is PolicyFunctionalCellStatus.FAILED
                    for item in incomplete
                )
                else MetaReplicateStatus.ABSTAINED
            )
            projected.append(
                PolicyFunctionalMetaReplicate(
                    replicate_index=meta_index,
                    status=status,
                    oracle=None,
                    posterior_rows=(),
                    status_reason=(
                        "retained incomplete source cell; complete-case rank "
                        "fallback is prohibited"
                    ),
                    status_evidence_sha256=_attempt_source_evidence(group),
                    failure_stage=stage,
                )
            )
            continue
        receipt = receipt_by_index.get(meta_index)
        if receipt is None:
            _fail(
                "completed meta-replicate lacks its calibration receipt",
                code="incomplete_calibration_ledger",
            )
        oracle_attempts = tuple(
            sorted(
                (
                    item
                    for item in group
                    if item.role is PolicyFunctionalCellRole.ORACLE
                ),
                key=lambda item: item.scenario_replicate,
            )
        )
        oracle = PolicyFunctionalOracleRow(
            truth_parameter_sha256=receipt.truth_parameter_sha256,
            scenario_values=tuple(
                item.value for item in oracle_attempts if item.value is not None
            ),
            scenario_commitments=tuple(
                item.scenario_commitment_sha256 for item in oracle_attempts
            ),
            source_evidence_sha256=_attempt_source_evidence(oracle_attempts),
        )
        posterior_rows: list[PolicyFunctionalPosteriorRow] = []
        for posterior_index in range(plan.posterior_draw_count):
            row_attempts = tuple(
                sorted(
                    (
                        item
                        for item in group
                        if item.role is PolicyFunctionalCellRole.POSTERIOR
                        and item.posterior_index == posterior_index
                    ),
                    key=lambda item: item.scenario_replicate,
                )
            )
            posterior_draw_sha256 = row_attempts[0].posterior_draw_sha256
            if posterior_draw_sha256 is None:
                _fail("completed posterior row lost its draw identity")
            posterior_rows.append(
                PolicyFunctionalPosteriorRow(
                    posterior_index=posterior_index,
                    posterior_draw_sha256=posterior_draw_sha256,
                    parameter_sha256=row_attempts[0].parameter_sha256,
                    scenario_values=tuple(
                        item.value for item in row_attempts if item.value is not None
                    ),
                    scenario_commitments=tuple(
                        item.scenario_commitment_sha256 for item in row_attempts
                    ),
                    source_evidence_sha256=_attempt_source_evidence(row_attempts),
                )
            )
        projected.append(
            PolicyFunctionalMetaReplicate(
                replicate_index=meta_index,
                status=MetaReplicateStatus.COMPLETED,
                oracle=oracle,
                posterior_rows=tuple(posterior_rows),
            )
        )
    return tuple(projected)


_EXPECTED_CANARY_ERROR_CODES: Mapping[CanaryKind, str] = {
    CanaryKind.POSTERIOR_SHIFT: "posterior_binding_mismatch",
    CanaryKind.IGNORED_DATA: "calibration_input_mismatch",
    CanaryKind.RNG_MISMATCH: "rng_mismatch",
    CanaryKind.OMITTED_MECHANISM: "mechanism_mismatch",
}


def _capture_contract_canary(
    *,
    kind: CanaryKind,
    injected_input_sha256: str,
    mutation: object,
) -> PolicyFunctionalContractCanaryReceipt:
    expected_code = _EXPECTED_CANARY_ERROR_CODES[kind]
    try:
        if not callable(mutation):
            _fail("canary mutation is not callable")
        mutation()
    except PolicyFunctionalMetaStudyError as error:
        if error.code != expected_code:
            raise PolicyFunctionalMetaStudyError(
                f"{kind.value} canary observed unexpected guard code {error.code}",
                code="canary_guard_mismatch",
            ) from error
        reason = (
            f"executed deterministic {kind.value} contract mutation and observed "
            f"fail-closed guard {error.code}"
        )
        evidence = _digest_mapping(
            {
                "expected_error_code": expected_code,
                "injected_input_sha256": injected_input_sha256,
                "kind": kind.value,
                "observed_error_code": error.code,
                "reason": str(error),
            }
        )
        return PolicyFunctionalContractCanaryReceipt(
            kind=kind,
            detection=CanaryDetection.DETECTED_FAIL,
            expected_error_code=expected_code,
            observed_error_code=error.code,
            injected_input_sha256=injected_input_sha256,
            guard_evidence_sha256=evidence,
            reason=reason,
        )
    _fail(
        f"{kind.value} contract canary was not rejected",
        code="canary_not_detected",
    )


def _execute_contract_canaries(
    *,
    plan: PolicyFunctionalMetaStudyPlan,
    protocol: StudyProtocol,
    execution_started_at: datetime,
    calibration_runs: tuple[CalibrationRun | None, ...],
    calibration_receipts: tuple[PolicyFunctionalCalibrationReceipt, ...],
    attempts: tuple[PolicyFunctionalCellAttempt, ...],
) -> tuple[PolicyFunctionalContractCanaryReceipt, ...]:
    successful_indices = {
        receipt.meta_replicate_index for receipt in calibration_receipts
    }
    posterior_position = next(
        (
            index
            for index, item in enumerate(attempts)
            if item.role is PolicyFunctionalCellRole.POSTERIOR
            and item.meta_replicate_index in successful_indices
        ),
        None,
    )
    source_position = next(
        (
            index
            for index, item in enumerate(attempts)
            if item.meta_replicate_index in successful_indices
        ),
        None,
    )

    def validate_attempts(
        changed: tuple[PolicyFunctionalCellAttempt, ...],
    ) -> None:
        _validate_attempt_ledger(
            plan,
            changed,
            calibration_runs=calibration_runs,
            calibration_receipts=calibration_receipts,
        )

    if posterior_position is None:
        posterior_injected = _digest_mapping(
            {"canary": CanaryKind.POSTERIOR_SHIFT.value}
        )

        def posterior_mutation() -> None:
            _require_digest_binding(
                expected=plan.parameter_sbc_design_sha256,
                observed=posterior_injected,
                code="posterior_binding_mismatch",
                message="posterior-shift canary changed a frozen digest binding",
            )

    else:
        shifted_attempt = replace(
            attempts[posterior_position],
            parameter_sha256=_digest_mapping(
                {"canary": CanaryKind.POSTERIOR_SHIFT.value}
            ),
        )
        shifted_attempts = (
            *attempts[:posterior_position],
            shifted_attempt,
            *attempts[posterior_position + 1 :],
        )
        posterior_injected = _digest_mapping(
            {
                "attempt_sha256s": [item.sha256 for item in shifted_attempts],
                "kind": CanaryKind.POSTERIOR_SHIFT.value,
            }
        )

        def posterior_mutation() -> None:
            validate_attempts(shifted_attempts)

    if calibration_receipts:
        ignored_receipt = replace(
            calibration_receipts[0],
            input_sha256=_digest_mapping({"canary": CanaryKind.IGNORED_DATA.value}),
        )
        ignored_receipts = (ignored_receipt, *calibration_receipts[1:])
        ignored_injected = _digest_mapping(
            {
                "kind": CanaryKind.IGNORED_DATA.value,
                "receipt_sha256s": [item.sha256 for item in ignored_receipts],
            }
        )

        def ignored_data_mutation() -> None:
            _validate_calibration_closure(
                plan=plan,
                protocol=protocol,
                execution_started_at=execution_started_at,
                calibration_runs=calibration_runs,
                receipts=ignored_receipts,
            )

    else:
        ignored_injected = _digest_mapping({"canary": CanaryKind.IGNORED_DATA.value})

        def ignored_data_mutation() -> None:
            _require_digest_binding(
                expected=plan.calibration_configs[0].sha256,
                observed=ignored_injected,
                code="calibration_input_mismatch",
                message="ignored-data canary changed a frozen input binding",
            )

    if source_position is None:
        rng_injected = _digest_mapping({"canary": CanaryKind.RNG_MISMATCH.value})
        mechanism_injected = _digest_mapping(
            {"canary": CanaryKind.OMITTED_MECHANISM.value}
        )

        def rng_mutation() -> None:
            _require_digest_binding(
                expected=plan.pairs[0].shock_tape.sha256,
                observed=rng_injected,
                code="rng_mismatch",
                message="RNG canary changed a frozen scenario-tape binding",
            )

        def mechanism_mutation() -> None:
            _require_digest_binding(
                expected=plan.market_config.sha256,
                observed=mechanism_injected,
                code="mechanism_mismatch",
                message="mechanism canary changed a frozen config binding",
            )

    else:
        rng_attempt = replace(
            attempts[source_position],
            scenario_commitment_sha256=_digest_mapping(
                {"canary": CanaryKind.RNG_MISMATCH.value}
            ),
        )
        rng_attempts = (
            *attempts[:source_position],
            rng_attempt,
            *attempts[source_position + 1 :],
        )
        mechanism_attempt = replace(
            attempts[source_position],
            mechanism_config_sha256=_digest_mapping(
                {"canary": CanaryKind.OMITTED_MECHANISM.value}
            ),
        )
        mechanism_attempts = (
            *attempts[:source_position],
            mechanism_attempt,
            *attempts[source_position + 1 :],
        )
        rng_injected = _digest_mapping(
            {
                "attempt_sha256s": [item.sha256 for item in rng_attempts],
                "kind": CanaryKind.RNG_MISMATCH.value,
            }
        )
        mechanism_injected = _digest_mapping(
            {
                "attempt_sha256s": [item.sha256 for item in mechanism_attempts],
                "kind": CanaryKind.OMITTED_MECHANISM.value,
            }
        )

        def rng_mutation() -> None:
            validate_attempts(rng_attempts)

        def mechanism_mutation() -> None:
            validate_attempts(mechanism_attempts)

    receipts = (
        _capture_contract_canary(
            kind=CanaryKind.POSTERIOR_SHIFT,
            injected_input_sha256=posterior_injected,
            mutation=posterior_mutation,
        ),
        _capture_contract_canary(
            kind=CanaryKind.IGNORED_DATA,
            injected_input_sha256=ignored_injected,
            mutation=ignored_data_mutation,
        ),
        _capture_contract_canary(
            kind=CanaryKind.RNG_MISMATCH,
            injected_input_sha256=rng_injected,
            mutation=rng_mutation,
        ),
        _capture_contract_canary(
            kind=CanaryKind.OMITTED_MECHANISM,
            injected_input_sha256=mechanism_injected,
            mutation=mechanism_mutation,
        ),
    )
    return tuple(sorted(receipts, key=lambda item: item.kind.value))


def _core_canaries(
    *,
    plan: PolicyFunctionalMetaStudyPlan,
    receipts: tuple[PolicyFunctionalContractCanaryReceipt, ...],
) -> tuple[PolicyFunctionalCanaryResult, ...]:
    return tuple(
        PolicyFunctionalCanaryResult(
            kind=receipt.kind,
            detection=receipt.detection,
            evidence_sha256=receipt.sha256,
            spec_sha256=plan.core_spec.sha256,
            primary_output_definition_sha256=(
                plan.core_spec.primary_output_definition_sha256
            ),
            reason=(
                "enclosing local runner retained a contract receipt; this "
                "phase-one arithmetic object does not inherit execution authority"
            ),
        )
        for receipt in receipts
    )


def _derive_core_result(
    *,
    plan: PolicyFunctionalMetaStudyPlan,
    attempts: tuple[PolicyFunctionalCellAttempt, ...],
    calibration_receipts: tuple[PolicyFunctionalCalibrationReceipt, ...],
    canary_receipts: tuple[PolicyFunctionalContractCanaryReceipt, ...],
) -> PolicyFunctionalSBCResult:
    return derive_policy_functional_sbc_result(
        spec=plan.core_spec,
        meta_replicates=_project_meta_replicates(
            plan=plan,
            attempts=attempts,
            receipts=calibration_receipts,
        ),
        canaries=_core_canaries(plan=plan, receipts=canary_receipts),
    )


def _calibration_run_commitment(run: CalibrationRun | None) -> object:
    if run is None:
        return None
    return {
        "config_sha256": run.config.sha256,
        "event_keys_sha256": sha256_bytes(canonical_json_bytes(list(run.event_keys))),
        "input_sha256": run.observed.sha256,
        "parameter_sbc_sha256": run.sbc.sha256,
        "posterior_sha256": run.posterior.sha256,
        "posterior_predictive_checks_sha256": run.ppc.sha256,
        "protocol_execution_receipt_sha256": (run.protocol_execution_receipt.sha256),
        "report_sha256": run.report.sha256,
        "simulator_config_sha256": sha256_bytes(run.simulator_config_json),
        "true_parameters_sha256": run.true_parameters_sha256,
    }


def _actual_parameter_sbc_manifest_sha256(
    *,
    plan: PolicyFunctionalMetaStudyPlan,
    receipts: tuple[PolicyFunctionalCalibrationReceipt, ...],
) -> str:
    by_index = {item.meta_replicate_index: item for item in receipts}
    return _digest_mapping(
        {
            "parameter_sbc_design_sha256": plan.parameter_sbc_design_sha256,
            "planned_meta_replicate_count": plan.meta_replicate_count,
            "results": [
                {
                    "meta_replicate_index": index,
                    "parameter_sbc_result_sha256": (
                        None
                        if index not in by_index
                        else by_index[index].parameter_sbc_sha256
                    ),
                    "status": (
                        "missing_dependency"
                        if index not in by_index
                        else "completed_synthetic"
                    ),
                }
                for index in range(plan.meta_replicate_count)
            ],
            "schema_version": 1,
        }
    )


def _validate_plan_protocol(
    *,
    plan: PolicyFunctionalMetaStudyPlan,
    protocol: StudyProtocol,
    execution_started_at: datetime,
) -> None:
    plan._fresh_validate()
    if type(protocol) is not StudyProtocol:
        _fail("meta-study result lost its exact protocol")
    if (
        protocol.mode is not StudyMode.SYNTHETIC
        or protocol.claim_scope is not ClaimScope.SYNTHETIC_ONLY
        or protocol.knowledge_cutoff != plan.market_config.cutoff
        or protocol.randomness_protocol is not RandomnessProtocol.EVENT_KEYED_PAIRED
        or plan.frozen_definition not in protocol.evidence_plans
    ):
        _fail(
            "meta-study protocol differs from the frozen synthetic plan",
            code="protocol_binding_mismatch",
        )
    started = _aware(execution_started_at, "execution_started_at")
    if (
        started < protocol.registered_at
        or started >= plan.outcomes_available_at
        or started >= protocol.evaluation_reveal_at
    ):
        _fail(
            "meta-study execution lies outside pre-outcome protocol chronology",
            code="invalid_chronology",
        )


@dataclass(frozen=True, slots=True, init=False, weakref_slot=True)
class SourceBoundPolicyFunctionalMetaEvaluation:
    """Process-local result from the closed synthetic meta-study executor."""

    plan: PolicyFunctionalMetaStudyPlan
    protocol: StudyProtocol
    execution_started_at: datetime
    meta_protocol_execution_receipt: ProtocolExecutionReceipt
    calibration_runs: tuple[CalibrationRun | None, ...]
    calibration_receipts: tuple[PolicyFunctionalCalibrationReceipt, ...]
    cell_attempts: tuple[PolicyFunctionalCellAttempt, ...]
    canary_receipts: tuple[PolicyFunctionalContractCanaryReceipt, ...]
    core_result: PolicyFunctionalSBCResult
    plan_sha256: str
    protocol_sha256: str
    actual_parameter_sbc_manifest_sha256: str
    complete_cell_attempt_ledger_rederived: Literal[True]
    exogenous_rng_receipts_checked_at_execution: bool
    actual_contract_canaries_executed: Literal[True]
    local_synthetic_calibration_replay_verified: bool
    local_integrated_cell_execution_retained: bool
    statistical_canary_power_verified: Literal[False]
    posterior_exchangeability_verified: Literal[False]
    internal_agent_rng_consumption_verified: Literal[False]
    externally_preregistered: Literal[False]
    external_custody_verified: Literal[False]
    independent_reproduction: Literal[False]
    real_world_coverage_claim: Literal[False]
    real_world_causal_claim: Literal[False]
    publication_eligible: Literal[False]
    synthetic_only: Literal[True]
    local_rank_arithmetic_abstention_required: bool
    scientific_abstention_required: Literal[True]
    claim_scope: str
    schema_version: str

    def __init__(self) -> None:
        raise TypeError(
            "SourceBoundPolicyFunctionalMetaEvaluation is verifier-produced; use "
            "run_prepared_policy_functional_meta_study"
        )

    def __copy__(self) -> NoReturn:
        raise TypeError("source-bound policy-functional results cannot be copied")

    def __deepcopy__(self, memo: object) -> NoReturn:
        del memo
        raise TypeError("source-bound policy-functional results cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("source-bound policy-functional results cannot be serialized")

    def _fresh_validate(self) -> None:
        _validate_plan_protocol(
            plan=self.plan,
            protocol=self.protocol,
            execution_started_at=self.execution_started_at,
        )
        if self.plan_sha256 != self.plan.sha256:
            _fail("result plan commitment changed", code="result_seal_mismatch")
        if self.protocol_sha256 != self.protocol.sha256:
            _fail("result protocol commitment changed", code="result_seal_mismatch")
        if self.actual_parameter_sbc_manifest_sha256 != (
            _actual_parameter_sbc_manifest_sha256(
                plan=self.plan,
                receipts=self.calibration_receipts,
            )
        ):
            _fail(
                "actual parameter-SBC result manifest changed",
                code="calibration_receipt_mismatch",
            )
        if type(self.meta_protocol_execution_receipt) is not ProtocolExecutionReceipt:
            _fail("result lost its outer meta-study protocol receipt")
        try:
            fresh_meta_receipt = ProtocolExecutionReceipt.from_mapping(
                self.meta_protocol_execution_receipt.to_mapping()
            )
            validate_protocol_execution_receipt_binding(
                self.protocol,
                fresh_meta_receipt,
            )
        except Exception as error:
            raise PolicyFunctionalMetaStudyError(
                "outer meta-study protocol receipt failed strict reconstruction",
                code="protocol_receipt_mismatch",
            ) from error
        if (
            fresh_meta_receipt != self.meta_protocol_execution_receipt
            or fresh_meta_receipt.consumer
            is not ProtocolExecutionConsumer.POLICY_FUNCTIONAL_META_EVALUATION
            or fresh_meta_receipt.purpose
            is not ProtocolExecutionPurpose.SYNTHETIC_POLICY_FUNCTIONAL_META_EVALUATION
            or fresh_meta_receipt.config_sha256 != self.plan.sha256
            or fresh_meta_receipt.input_sha256
            != _meta_authorization_input_sha256(self.plan, self.protocol)
            or fresh_meta_receipt.execution_started_at != self.execution_started_at
        ):
            _fail(
                "outer meta-study protocol receipt differs from its plan binding",
                code="protocol_receipt_mismatch",
            )
        _validate_calibration_closure(
            plan=self.plan,
            protocol=self.protocol,
            execution_started_at=self.execution_started_at,
            calibration_runs=self.calibration_runs,
            receipts=self.calibration_receipts,
        )
        _validate_attempt_ledger(
            self.plan,
            self.cell_attempts,
            calibration_runs=self.calibration_runs,
            calibration_receipts=self.calibration_receipts,
        )
        if not isinstance(self.canary_receipts, tuple) or not all(
            type(item) is PolicyFunctionalContractCanaryReceipt
            for item in self.canary_receipts
        ):
            _fail("contract canary receipt ledger has the wrong type")
        fresh_canaries = tuple(
            PolicyFunctionalContractCanaryReceipt(
                kind=item.kind,
                detection=item.detection,
                expected_error_code=item.expected_error_code,
                observed_error_code=item.observed_error_code,
                injected_input_sha256=item.injected_input_sha256,
                guard_evidence_sha256=item.guard_evidence_sha256,
                reason=item.reason,
                execution_attempted=item.execution_attempted,
                statistical_power_evidence=item.statistical_power_evidence,
            )
            for item in self.canary_receipts
        )
        if (
            fresh_canaries != self.canary_receipts
            or tuple(item.kind for item in fresh_canaries)
            != tuple(sorted(CanaryKind, key=lambda item: item.value))
            or any(
                item.expected_error_code != _EXPECTED_CANARY_ERROR_CODES[item.kind]
                for item in fresh_canaries
            )
        ):
            _fail(
                "contract canary receipts do not rederive",
                code="canary_receipt_mismatch",
            )
        expected_core = _derive_core_result(
            plan=self.plan,
            attempts=self.cell_attempts,
            calibration_receipts=self.calibration_receipts,
            canary_receipts=self.canary_receipts,
        )
        if (
            type(self.core_result) is not PolicyFunctionalSBCResult
            or self.core_result.to_mapping() != expected_core.to_mapping()
        ):
            _fail("embedded policy-functional core does not freshly rederive")
        expected_rng = _all_completed_rng_receipts(self.cell_attempts)
        expected_local_replay = all(run is not None for run in self.calibration_runs)
        expected_local_execution = all(
            item.status is PolicyFunctionalCellStatus.COMPLETED
            for item in self.cell_attempts
        )
        if (
            self.complete_cell_attempt_ledger_rederived is not True
            or self.exogenous_rng_receipts_checked_at_execution is not expected_rng
            or self.actual_contract_canaries_executed is not True
            or self.local_synthetic_calibration_replay_verified
            is not expected_local_replay
            or self.local_integrated_cell_execution_retained
            is not expected_local_execution
            or self.statistical_canary_power_verified is not False
            or self.posterior_exchangeability_verified is not False
            or self.internal_agent_rng_consumption_verified is not False
            or self.externally_preregistered is not False
            or self.external_custody_verified is not False
            or self.independent_reproduction is not False
            or self.real_world_coverage_claim is not False
            or self.real_world_causal_claim is not False
            or self.publication_eligible is not False
            or self.synthetic_only is not True
            or self.local_rank_arithmetic_abstention_required
            is not (
                self.core_result.decision.value
                != "synthetic_core_arithmetic_pass_unverified"
            )
            or self.scientific_abstention_required is not True
            or self.claim_scope != POLICY_FUNCTIONAL_META_STUDY_CLAIM_SCOPE
            or self.schema_version != POLICY_FUNCTIONAL_META_STUDY_SCHEMA_VERSION
        ):
            _fail("source-bound policy-functional claim boundary changed")

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_mapping())

    @property
    def sha256(self) -> str:
        return sha256_bytes(self.canonical_bytes)

    def to_mapping(self) -> dict[str, object]:
        self._fresh_validate()
        return {
            "actual_parameter_sbc_manifest_sha256": (
                self.actual_parameter_sbc_manifest_sha256
            ),
            "local_rank_arithmetic_abstention_required": (
                self.local_rank_arithmetic_abstention_required
            ),
            "scientific_abstention_required": self.scientific_abstention_required,
            "actual_contract_canaries_executed": (
                self.actual_contract_canaries_executed
            ),
            "calibration_receipts": [
                item.to_mapping() for item in self.calibration_receipts
            ],
            "calibration_run_commitments": [
                _calibration_run_commitment(item) for item in self.calibration_runs
            ],
            "canary_receipts": [item.to_mapping() for item in self.canary_receipts],
            "cell_attempts": [item.to_mapping() for item in self.cell_attempts],
            "claim_scope": self.claim_scope,
            "complete_cell_attempt_ledger_rederived": (
                self.complete_cell_attempt_ledger_rederived
            ),
            "core_result": self.core_result.to_mapping(),
            "execution_started_at": self.execution_started_at.isoformat(),
            "exogenous_rng_receipts_checked_at_execution": (
                self.exogenous_rng_receipts_checked_at_execution
            ),
            "external_custody_verified": self.external_custody_verified,
            "externally_preregistered": self.externally_preregistered,
            "independent_reproduction": self.independent_reproduction,
            "internal_agent_rng_consumption_verified": (
                self.internal_agent_rng_consumption_verified
            ),
            "local_integrated_cell_execution_retained": (
                self.local_integrated_cell_execution_retained
            ),
            "local_synthetic_calibration_replay_verified": (
                self.local_synthetic_calibration_replay_verified
            ),
            "meta_protocol_execution_receipt": (
                self.meta_protocol_execution_receipt.to_mapping()
            ),
            "plan": self.plan.to_mapping(),
            "plan_sha256": self.plan_sha256,
            "posterior_exchangeability_verified": (
                self.posterior_exchangeability_verified
            ),
            "protocol": self.protocol.to_mapping(),
            "protocol_sha256": self.protocol_sha256,
            "publication_eligible": self.publication_eligible,
            "real_world_causal_claim": self.real_world_causal_claim,
            "real_world_coverage_claim": self.real_world_coverage_claim,
            "schema_version": self.schema_version,
            "statistical_canary_power_verified": (
                self.statistical_canary_power_verified
            ),
            "synthetic_only": self.synthetic_only,
        }


class _MetaResultMintTicket:
    """Opaque one-use authority created after the complete execution closure."""

    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("policy-functional result mint tickets are verifier-issued")

    def __copy__(self) -> NoReturn:
        raise TypeError("policy-functional result mint tickets cannot be copied")

    def __deepcopy__(self, memo: object) -> NoReturn:
        del memo
        raise TypeError("policy-functional result mint tickets cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("policy-functional result mint tickets cannot be serialized")


@dataclass(frozen=True, slots=True)
class _MetaResultMintBinding:
    plan_sha256: str
    protocol_sha256: str
    protocol_receipt_sha256: str
    calibration_manifest_sha256: str
    attempt_manifest_sha256: str
    canary_manifest_sha256: str
    core_result_sha256: str


def _meta_result_mint_binding(
    result: SourceBoundPolicyFunctionalMetaEvaluation,
) -> _MetaResultMintBinding:
    return _MetaResultMintBinding(
        plan_sha256=result.plan_sha256,
        protocol_sha256=result.protocol_sha256,
        protocol_receipt_sha256=result.meta_protocol_execution_receipt.sha256,
        calibration_manifest_sha256=_digest_mapping(
            {
                "receipt_sha256s": [
                    item.sha256 for item in result.calibration_receipts
                ],
                "run_commitments": [
                    _calibration_run_commitment(item)
                    for item in result.calibration_runs
                ],
            }
        ),
        attempt_manifest_sha256=_digest_mapping(
            {"attempt_sha256s": [item.sha256 for item in result.cell_attempts]}
        ),
        canary_manifest_sha256=_digest_mapping(
            {"canary_sha256s": [item.sha256 for item in result.canary_receipts]}
        ),
        core_result_sha256=result.core_result.sha256,
    )


_META_RESULT_MINT_TICKETS: WeakKeyDictionary[
    _MetaResultMintTicket, _MetaResultMintBinding
] = WeakKeyDictionary()
_VERIFIED_META_RESULTS: WeakValueDictionary[
    int, SourceBoundPolicyFunctionalMetaEvaluation
] = WeakValueDictionary()
_VERIFIED_META_RESULT_SEALS: dict[int, str] = {}
_VERIFIED_META_RESULTS_LOCK = RLock()


def _forget_meta_result(identity: int) -> None:
    with _VERIFIED_META_RESULTS_LOCK:
        if _VERIFIED_META_RESULTS.get(identity) is None:
            _VERIFIED_META_RESULT_SEALS.pop(identity, None)


def _mint_meta_result(
    *,
    ticket: _MetaResultMintTicket,
    result: SourceBoundPolicyFunctionalMetaEvaluation,
) -> SourceBoundPolicyFunctionalMetaEvaluation:
    if type(ticket) is not _MetaResultMintTicket:
        _fail("meta result mint requires an exact verifier ticket", code="invalid_mint")
    with _VERIFIED_META_RESULTS_LOCK:
        binding = _META_RESULT_MINT_TICKETS.pop(ticket, None)
    if binding is None:
        _fail(
            "meta result mint ticket is invalid or already consumed",
            code="invalid_mint",
        )
    if binding != _meta_result_mint_binding(result):
        _fail(
            "meta result differs from its verifier mint ticket",
            code="invalid_mint",
        )
    result._fresh_validate()
    identity = id(result)
    seal = sha256_bytes(result.canonical_bytes)
    with _VERIFIED_META_RESULTS_LOCK:
        _VERIFIED_META_RESULTS[identity] = result
        _VERIFIED_META_RESULT_SEALS[identity] = seal
    finalize(result, _forget_meta_result, identity)
    return result


def require_source_bound_policy_functional_meta_evaluation(
    result: SourceBoundPolicyFunctionalMetaEvaluation,
) -> SourceBoundPolicyFunctionalMetaEvaluation:
    """Require same-process mint origin and unchanged canonical content."""
    if type(result) is not SourceBoundPolicyFunctionalMetaEvaluation:
        _fail("source-bound meta result has the wrong type", code="invalid_mint")
    try:
        before = sha256_bytes(result.canonical_bytes)
    except Exception as error:
        raise PolicyFunctionalMetaStudyError(
            "source-bound meta result cannot reproduce its mint seal",
            code="result_seal_mismatch",
        ) from error
    with _VERIFIED_META_RESULTS_LOCK:
        registered = _VERIFIED_META_RESULTS.get(id(result))
        mint_seal = _VERIFIED_META_RESULT_SEALS.get(id(result))
    if registered is not result or mint_seal != before:
        _fail(
            "source-bound meta result lacks its verifier mint or changed seal",
            code="invalid_mint",
        )
    result._fresh_validate()
    after = sha256_bytes(result.canonical_bytes)
    with _VERIFIED_META_RESULTS_LOCK:
        still_registered = _VERIFIED_META_RESULTS.get(id(result))
        current_seal = _VERIFIED_META_RESULT_SEALS.get(id(result))
    if still_registered is not result or before != after or current_seal != after:
        _fail(
            "source-bound meta result changed after verifier mint",
            code="result_seal_mismatch",
        )
    return result


def run_prepared_policy_functional_meta_study(
    prepared: PreparedPolicyFunctionalMetaStudy,
    *,
    parallel_cells: bool = False,
    parallel_worlds: bool = False,
) -> SourceBoundPolicyFunctionalMetaEvaluation:
    """Consume one capability and retain every planned synthetic attempt."""
    if not isinstance(parallel_cells, bool) or not isinstance(parallel_worlds, bool):
        raise TypeError("parallel switches must be boolean")
    (
        plan,
        protocol,
        execution_started_at,
        meta_protocol_execution_receipt,
    ) = _consume_prepared(prepared)
    _validate_plan_protocol(
        plan=plan,
        protocol=protocol,
        execution_started_at=execution_started_at,
    )

    runs: list[CalibrationRun | None] = [None] * plan.meta_replicate_count
    receipts: list[PolicyFunctionalCalibrationReceipt] = []
    calibration_errors: dict[int, Exception] = {}
    for meta_index in range(plan.meta_replicate_count):
        try:
            run, receipt = _run_one_calibration(
                plan=plan,
                source_protocol=protocol,
                execution_started_at=execution_started_at,
                meta_replicate_index=meta_index,
            )
        except Exception as error:  # retained as dependency abstention
            calibration_errors[meta_index] = error
        else:
            runs[meta_index] = run
            receipts.append(receipt)

    calibration_runs = tuple(runs)
    calibration_receipts = tuple(
        sorted(receipts, key=lambda item: item.meta_replicate_index)
    )
    run_mapping = {
        index: run for index, run in enumerate(calibration_runs) if run is not None
    }
    tasks = _cell_tasks(plan, run_mapping)
    issuer = SyntheticProtocolExecutionIssuer(protocol)

    def execute(task: _CellTask) -> PolicyFunctionalCellAttempt:
        try:
            return _run_integrated_cell(
                task=task,
                plan=plan,
                protocol=protocol,
                issuer=issuer,
                execution_started_at=execution_started_at,
                parallel_worlds=parallel_worlds,
            )
        except Exception as error:
            return _failed_attempt(task=task, plan=plan, error=error)

    if parallel_cells and tasks:
        with ThreadPoolExecutor(max_workers=min(32, len(tasks))) as executor:
            executed_attempts = tuple(executor.map(execute, tasks))
    else:
        executed_attempts = tuple(execute(task) for task in tasks)
    dependency_attempts = tuple(
        item
        for meta_index, error in sorted(calibration_errors.items())
        for item in _dependency_attempts(
            plan=plan,
            meta_replicate_index=meta_index,
            error=error,
        )
    )
    attempts = tuple(
        sorted(
            (*executed_attempts, *dependency_attempts),
            key=lambda item: (
                item.meta_replicate_index,
                item.role.value,
                -1 if item.posterior_index is None else item.posterior_index,
                item.scenario_replicate,
            ),
        )
    )
    _validate_calibration_closure(
        plan=plan,
        protocol=protocol,
        execution_started_at=execution_started_at,
        calibration_runs=calibration_runs,
        receipts=calibration_receipts,
    )
    _validate_attempt_ledger(
        plan,
        attempts,
        calibration_runs=calibration_runs,
        calibration_receipts=calibration_receipts,
    )
    canary_receipts = _execute_contract_canaries(
        plan=plan,
        protocol=protocol,
        execution_started_at=execution_started_at,
        calibration_runs=calibration_runs,
        calibration_receipts=calibration_receipts,
        attempts=attempts,
    )
    core_result = _derive_core_result(
        plan=plan,
        attempts=attempts,
        calibration_receipts=calibration_receipts,
        canary_receipts=canary_receipts,
    )
    result = object.__new__(SourceBoundPolicyFunctionalMetaEvaluation)
    values: tuple[tuple[str, object], ...] = (
        ("plan", plan),
        ("protocol", protocol),
        ("execution_started_at", execution_started_at),
        ("meta_protocol_execution_receipt", meta_protocol_execution_receipt),
        ("calibration_runs", calibration_runs),
        ("calibration_receipts", calibration_receipts),
        ("cell_attempts", attempts),
        ("canary_receipts", canary_receipts),
        ("core_result", core_result),
        ("plan_sha256", plan.sha256),
        ("protocol_sha256", protocol.sha256),
        (
            "actual_parameter_sbc_manifest_sha256",
            _actual_parameter_sbc_manifest_sha256(
                plan=plan,
                receipts=calibration_receipts,
            ),
        ),
        ("complete_cell_attempt_ledger_rederived", True),
        (
            "exogenous_rng_receipts_checked_at_execution",
            _all_completed_rng_receipts(attempts),
        ),
        ("actual_contract_canaries_executed", True),
        (
            "local_synthetic_calibration_replay_verified",
            all(run is not None for run in calibration_runs),
        ),
        (
            "local_integrated_cell_execution_retained",
            all(
                item.status is PolicyFunctionalCellStatus.COMPLETED for item in attempts
            ),
        ),
        ("statistical_canary_power_verified", False),
        ("posterior_exchangeability_verified", False),
        ("internal_agent_rng_consumption_verified", False),
        ("externally_preregistered", False),
        ("external_custody_verified", False),
        ("independent_reproduction", False),
        ("real_world_coverage_claim", False),
        ("real_world_causal_claim", False),
        ("publication_eligible", False),
        ("synthetic_only", True),
        (
            "local_rank_arithmetic_abstention_required",
            core_result.decision.value != "synthetic_core_arithmetic_pass_unverified",
        ),
        ("scientific_abstention_required", True),
        ("claim_scope", POLICY_FUNCTIONAL_META_STUDY_CLAIM_SCOPE),
        ("schema_version", POLICY_FUNCTIONAL_META_STUDY_SCHEMA_VERSION),
    )
    for field, value in values:
        object.__setattr__(result, field, value)
    result._fresh_validate()
    ticket = object.__new__(_MetaResultMintTicket)
    with _VERIFIED_META_RESULTS_LOCK:
        _META_RESULT_MINT_TICKETS[ticket] = _meta_result_mint_binding(result)
    return _mint_meta_result(ticket=ticket, result=result)


__all__ = [
    "POLICY_FUNCTIONAL_META_STUDY_CLAIM_SCOPE",
    "POLICY_FUNCTIONAL_META_STUDY_SCHEMA_VERSION",
    "PolicyFunctionalCalibrationReceipt",
    "PolicyFunctionalCellAttempt",
    "PolicyFunctionalCellRole",
    "PolicyFunctionalCellStatus",
    "PolicyFunctionalContractCanaryReceipt",
    "PolicyFunctionalMetaStudyError",
    "PolicyFunctionalMetaStudyPlan",
    "PreparedPolicyFunctionalMetaStudy",
    "SourceBoundPolicyFunctionalMetaEvaluation",
    "authorize_policy_functional_meta_study",
    "build_policy_functional_meta_study_plan",
    "require_source_bound_policy_functional_meta_evaluation",
    "run_prepared_policy_functional_meta_study",
]
