"""Offline distilled-policy interface sharing the strict planning contract."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from policytwin.agents.llm_contracts import (
    LLMContractError,
    LLMOutputError,
    LLMPlanResponse,
    LLMPrivacyError,
    LLMTemporalError,
    StrictLLMRequest,
)
from policytwin.agents.llm_firewall import (
    AdapterArtifactKind,
    AdapterAuthorization,
    LLMKnowledgeFirewallError,
)
from policytwin.cognition.planning import CognitivePlan
from policytwin.provenance import (
    MetadataValidationError,
    canonical_json_bytes,
    sha256_bytes,
)
from policytwin.social.audit import (
    NULL_COMMITMENT,
    SocialAuditLog,
    SocialAuditOutcome,
    SocialAuditRule,
    SocialAuditStage,
)
from policytwin.social.privacy import (
    Anonymizer,
    DeidentificationPolicy,
    UnsafeTextError,
)

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class DistilledPolicyError(RuntimeError):
    """Raised when a distilled policy cannot safely produce a plan."""


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise DistilledPolicyError(f"{field} must be a stable identifier")
    return value


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise DistilledPolicyError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _aware_utc(value: object, field: str) -> datetime:
    if not isinstance(value, datetime):
        raise DistilledPolicyError(f"{field} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise DistilledPolicyError(f"{field} must include an explicit timezone")
    try:
        return value.astimezone(UTC)
    except (OverflowError, ValueError) as error:
        raise DistilledPolicyError(f"{field} is outside the UTC range") from error


def _commit(value: object) -> str:
    try:
        return sha256_bytes(canonical_json_bytes(value))
    except (MetadataValidationError, TypeError, ValueError):
        descriptor: dict[str, object] = {
            "container_type": type(value).__qualname__,
            "is_mapping": isinstance(value, Mapping),
        }
        if isinstance(value, Mapping):
            descriptor["field_count"] = len(value)
        return sha256_bytes(canonical_json_bytes(descriptor))


@dataclass(frozen=True, slots=True)
class DistilledPolicyMetadata:
    """Content commitments required before a distilled policy may run."""

    adapter_id: str
    checkpoint_sha256: str
    feature_schema_sha256: str
    output_schema_sha256: str
    training_manifest_sha256: str
    cutoff_policy_sha256: str
    trained_through: datetime

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "adapter_id", _identifier(self.adapter_id, "adapter_id")
        )
        for field in (
            "checkpoint_sha256",
            "feature_schema_sha256",
            "output_schema_sha256",
            "training_manifest_sha256",
            "cutoff_policy_sha256",
        ):
            object.__setattr__(self, field, _sha256(getattr(self, field), field))
        object.__setattr__(
            self, "trained_through", _aware_utc(self.trained_through, "trained_through")
        )

    def to_mapping(self) -> dict[str, object]:
        """Return complete immutable distilled-policy provenance."""
        return {
            "adapter_id": self.adapter_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "cutoff_policy_sha256": self.cutoff_policy_sha256,
            "feature_schema_sha256": self.feature_schema_sha256,
            "output_schema_sha256": self.output_schema_sha256,
            "trained_through": self.trained_through.isoformat(),
            "training_manifest_sha256": self.training_manifest_sha256,
        }


@runtime_checkable
class DistilledPolicyAdapter(Protocol):
    """Interface for a local distilled substitute; no model loader is implied."""

    @property
    def metadata(self) -> DistilledPolicyMetadata:
        """Return the immutable checkpoint and schema commitments."""
        ...

    def infer(self, request: Mapping[str, object]) -> Mapping[str, object]:
        """Return an untrusted candidate response without network access."""
        ...


@dataclass(frozen=True, slots=True)
class DeterministicMockDistilledPolicy:
    """Deterministic synthetic fixture implementing the distilled interface."""

    metadata: DistilledPolicyMetadata
    action: str
    confidence: float
    review_step: int
    valid_until: datetime
    rationale_code: str

    def infer(self, request: Mapping[str, object]) -> Mapping[str, object]:
        """Return a fixed schema candidate bound to the request hash."""
        input_sha256 = request.get("input_sha256")
        if not isinstance(input_sha256, str):
            raise DistilledPolicyError("distilled request lacks input commitment")
        return {
            "action": self.action,
            "confidence": self.confidence,
            "input_sha256": input_sha256,
            "rationale_code": self.rationale_code,
            "review_step": self.review_step,
            "schema_version": 1,
            "valid_until": self.valid_until.isoformat(),
        }


class ValidatedDistilledPolicyGateway:
    """Apply the same privacy, cutoff, output, and audit gates as LLM planning."""

    def __init__(
        self,
        *,
        adapter: DistilledPolicyAdapter,
        authorization: AdapterAuthorization,
        deidentification_policy: DeidentificationPolicy,
        cutoff_policy_sha256: str,
        audit_log: SocialAuditLog,
    ) -> None:
        if not isinstance(adapter, DistilledPolicyAdapter):
            raise DistilledPolicyError("adapter must implement DistilledPolicyAdapter")
        if not isinstance(deidentification_policy, DeidentificationPolicy):
            raise DistilledPolicyError(
                "deidentification_policy must be DeidentificationPolicy"
            )
        if not isinstance(audit_log, SocialAuditLog):
            raise DistilledPolicyError("audit_log must be SocialAuditLog")
        if not isinstance(authorization, AdapterAuthorization):
            raise DistilledPolicyError("authorization must be AdapterAuthorization")
        metadata = adapter.metadata
        if authorization.provenance.adapter_id != metadata.adapter_id:
            raise DistilledPolicyError(
                "distilled adapter ID does not match authorization provenance"
            )
        if authorization.provenance.artifact_sha256 != metadata.checkpoint_sha256:
            raise DistilledPolicyError(
                "distilled checkpoint does not match authorization provenance"
            )
        if (
            authorization.provenance.training_manifest_sha256
            != metadata.training_manifest_sha256
        ):
            raise DistilledPolicyError(
                "distilled training manifest does not match authorization"
            )
        if authorization.provenance.training_data_cutoff != metadata.trained_through:
            raise DistilledPolicyError(
                "distilled training cutoff does not match authorization"
            )
        if (
            isinstance(adapter, DeterministicMockDistilledPolicy)
            and authorization.provenance.artifact_kind
            is not AdapterArtifactKind.DETERMINISTIC_MOCK
        ):
            raise DistilledPolicyError(
                "deterministic distilled fixture requires mock authorization"
            )
        if authorization.cutoff_policy_sha256 != cutoff_policy_sha256:
            raise DistilledPolicyError("authorization cutoff policy mismatches gateway")
        if (
            authorization.deidentification_policy_sha256
            != deidentification_policy.sha256
        ):
            raise DistilledPolicyError(
                "authorization deidentification policy mismatches gateway"
            )
        self._adapter = adapter
        self._authorization = authorization
        self._privacy_policy = deidentification_policy
        self._cutoff_policy_sha256 = _sha256(
            cutoff_policy_sha256, "cutoff_policy_sha256"
        )
        self._audit = audit_log

    def plan(self, request: StrictLLMRequest) -> CognitivePlan:
        """Produce one validated plan from a local distilled policy."""
        input_commitment = request.input_sha256
        try:
            self._validate_request(request)
        except (
            LLMContractError,
            LLMKnowledgeFirewallError,
            DistilledPolicyError,
            UnsafeTextError,
        ) as error:
            rule = (
                SocialAuditRule.PRIVACY_VIOLATION
                if isinstance(error, LLMPrivacyError | UnsafeTextError)
                else (
                    SocialAuditRule.INTEGRITY_FAILURE
                    if isinstance(error, LLMKnowledgeFirewallError)
                    else SocialAuditRule.TEMPORAL_BOUNDARY
                )
            )
            self._audit.record(
                stage=SocialAuditStage.DISTILLED_INPUT,
                outcome=SocialAuditOutcome.REJECTED,
                rule_code=rule,
                event_key=request.event_key.value,
                policy_sha256=self._authorization.sha256,
                input_commitment=input_commitment,
                output_commitment=NULL_COMMITMENT,
                accepted_count=0,
                rejected_count=1,
            )
            raise

        adapter_input = request.to_adapter_mapping()
        adapter_input_commitment = _commit(adapter_input)
        self._audit.record(
            stage=SocialAuditStage.DISTILLED_INPUT,
            outcome=SocialAuditOutcome.ALLOWED,
            rule_code=SocialAuditRule.OK,
            event_key=request.event_key.value,
            policy_sha256=self._authorization.sha256,
            input_commitment=input_commitment,
            output_commitment=adapter_input_commitment,
            accepted_count=1,
            rejected_count=0,
        )
        try:
            raw_response = self._adapter.infer(adapter_input)
        except Exception as error:
            self._reject_output(
                request=request,
                adapter_input_commitment=adapter_input_commitment,
                output_commitment=NULL_COMMITMENT,
            )
            raise DistilledPolicyError("distilled policy failed") from error

        output_commitment = _commit(raw_response)
        try:
            if not isinstance(raw_response, Mapping):
                raise LLMOutputError("distilled output must be a mapping")
            response = LLMPlanResponse.from_mapping(raw_response)
            response.validate_for(request)
            plan = CognitivePlan(
                agent_id=request.agent_commitment,
                topic_id=request.topic_id,
                as_of_time=request.as_of_time,
                valid_until=response.valid_until,
                trusted_cutoff=request.trusted_cutoff,
                allowed_actions=(response.action,),
                confidence=response.confidence,
                review_step=response.review_step,
                input_sha256=request.input_sha256,
                adapter_id=self._adapter.metadata.adapter_id,
                adapter_authorization_sha256=self._authorization.sha256,
                requested_use=request.requested_use,
                rationale_code=response.rationale_code,
            )
        except (LLMContractError, ValueError) as error:
            self._reject_output(
                request=request,
                adapter_input_commitment=adapter_input_commitment,
                output_commitment=output_commitment,
            )
            if isinstance(error, LLMContractError):
                raise
            raise LLMOutputError("distilled output cannot form a plan") from error

        self._audit.record(
            stage=SocialAuditStage.DISTILLED_OUTPUT,
            outcome=SocialAuditOutcome.ALLOWED,
            rule_code=SocialAuditRule.OK,
            event_key=request.event_key.value,
            policy_sha256=self._authorization.sha256,
            input_commitment=output_commitment,
            output_commitment=_commit(plan.to_mapping()),
            accepted_count=1,
            rejected_count=0,
        )
        return plan

    def _validate_request(self, request: StrictLLMRequest) -> None:
        if not isinstance(request, StrictLLMRequest):
            raise DistilledPolicyError("request must be StrictLLMRequest")
        self._authorization.validate_request_commitments(
            authorization_sha256=request.adapter_authorization_sha256,
            requested_use=request.requested_use,
            as_of_time=request.as_of_time,
            trusted_cutoff=request.trusted_cutoff,
            cutoff_policy_sha256=request.cutoff_policy_sha256,
            deidentification_policy_sha256=(request.deidentification_policy_sha256),
            prompt_template_sha256=request.prompt_template_sha256,
            sanitized_context_record_sha256=(request.sanitized_context_record.sha256),
        )
        metadata = self._adapter.metadata
        if (
            request.cutoff_policy_sha256 != self._cutoff_policy_sha256
            or metadata.cutoff_policy_sha256 != self._cutoff_policy_sha256
        ):
            raise LLMTemporalError("distilled cutoff policy commitment mismatches")
        if metadata.trained_through > request.as_of_time:
            raise LLMTemporalError("distilled policy was trained after as_of_time")
        if (
            request.deidentification_policy_sha256 != self._privacy_policy.sha256
            or request.sanitized_context.receipt.policy_sha256
            != self._privacy_policy.sha256
        ):
            raise LLMPrivacyError("distilled request privacy policy mismatches")
        Anonymizer(self._privacy_policy).assert_safe(request.sanitized_context)

    def _reject_output(
        self,
        *,
        request: StrictLLMRequest,
        adapter_input_commitment: str,
        output_commitment: str,
    ) -> None:
        self._audit.record(
            stage=SocialAuditStage.DISTILLED_OUTPUT,
            outcome=SocialAuditOutcome.REJECTED,
            rule_code=SocialAuditRule.ILLEGAL_ADAPTER_OUTPUT,
            event_key=request.event_key.value,
            policy_sha256=self._authorization.sha256,
            input_commitment=adapter_input_commitment,
            output_commitment=output_commitment,
            accepted_count=0,
            rejected_count=1,
        )


__all__ = [
    "DeterministicMockDistilledPolicy",
    "DistilledPolicyAdapter",
    "DistilledPolicyError",
    "DistilledPolicyMetadata",
    "ValidatedDistilledPolicyGateway",
]
