"""Fail-closed offline gateway for strict slow-planning adapters."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
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
from policytwin.counterfactual import EventKey
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
    DeidentificationError,
    DeidentificationPolicy,
    SanitizedText,
    UnsafeTextError,
)


class LLMGatewayError(RuntimeError):
    """Raised when the adapter boundary fails without exposing raw content."""


@runtime_checkable
class LLMAdapter(Protocol):
    """Raw offline adapter interface behind the strict validation gateway."""

    @property
    def adapter_id(self) -> str:
        """Return a stable non-sensitive adapter identifier."""
        ...

    def complete(self, request: Mapping[str, object]) -> Mapping[str, object]:
        """Return a candidate response; the gateway treats it as untrusted."""
        ...


def _safe_commitment(value: object) -> str:
    """Hash JSON input, or a non-sensitive type descriptor when it is invalid."""
    try:
        return sha256_bytes(canonical_json_bytes(value))
    except (MetadataValidationError, TypeError, ValueError):
        descriptor: dict[str, object] = {
            "container_type": type(value).__qualname__,
            "is_mapping": isinstance(value, Mapping),
        }
        if isinstance(value, Mapping):
            descriptor["field_count"] = len(value)
            descriptor["field_types"] = sorted(type(key).__qualname__ for key in value)
        return sha256_bytes(canonical_json_bytes(descriptor))


def _event_key_value(event_key: EventKey | None) -> str | None:
    if event_key is None:
        return None
    if not isinstance(event_key, EventKey):
        raise LLMGatewayError("event_key must be an EventKey")
    return event_key.value


def _audit_rule(error: Exception) -> SocialAuditRule:
    if isinstance(error, LLMKnowledgeFirewallError):
        return SocialAuditRule.INTEGRITY_FAILURE
    if isinstance(error, LLMPrivacyError | UnsafeTextError):
        return SocialAuditRule.PRIVACY_VIOLATION
    if isinstance(error, LLMTemporalError):
        return SocialAuditRule.TEMPORAL_BOUNDARY
    if isinstance(error, LLMOutputError):
        return SocialAuditRule.ILLEGAL_ADAPTER_OUTPUT
    return SocialAuditRule.SCHEMA_VIOLATION


@dataclass(frozen=True, slots=True)
class DeterministicMockLLMAdapter:
    """Deterministic offline mock; it performs no I/O and opens no network."""

    adapter_id: str
    action: str
    confidence: float
    review_step: int
    valid_until: datetime
    rationale_code: str

    def complete(self, request: Mapping[str, object]) -> Mapping[str, object]:
        """Return a fixed candidate bound to the supplied request commitment."""
        input_sha256 = request.get("input_sha256")
        if not isinstance(input_sha256, str):
            raise LLMGatewayError("mock request is missing input commitment")
        return {
            "action": self.action,
            "confidence": self.confidence,
            "input_sha256": input_sha256,
            "rationale_code": self.rationale_code,
            "review_step": self.review_step,
            "schema_version": 1,
            "valid_until": self.valid_until.isoformat(),
        }


DETERMINISTIC_MOCK_ADAPTER_ARTIFACT_SCHEMA_VERSION = (
    "policytwin.deterministic-mock-adapter-artifact.v1"
)


def _implementation_source_sha256(
    value: type[object] | Callable[..., object],
) -> str:
    try:
        source = inspect.getsource(value)
    except (OSError, TypeError) as error:
        raise LLMGatewayError(
            "deterministic mock implementation source is unavailable"
        ) from error
    return sha256_bytes(source.encode("utf-8"))


_ALLOWED_DETERMINISTIC_MOCK_COMPLETE = DeterministicMockLLMAdapter.complete
_ALLOWED_DETERMINISTIC_MOCK_CLASS_SOURCE_SHA256 = _implementation_source_sha256(
    DeterministicMockLLMAdapter
)
_ALLOWED_DETERMINISTIC_MOCK_COMPLETE_SOURCE_SHA256 = _implementation_source_sha256(
    DeterministicMockLLMAdapter.complete
)


def deterministic_mock_adapter_artifact_mapping(
    adapter: DeterministicMockLLMAdapter,
) -> dict[str, object]:
    """Return the verifier-owned implementation and complete setting manifest.

    The exact concrete type rule prevents a subclass from retaining the trusted
    adapter ID while overriding ``complete``.  The live method digest catches a
    runtime replacement after module import; the two allowed source digests are
    fixed when this verifier module is loaded.
    """
    if type(adapter) is not DeterministicMockLLMAdapter:
        raise LLMGatewayError(
            "deterministic mock authorization requires the exact trusted type"
        )
    live_class_sha256 = _implementation_source_sha256(type(adapter))
    live_complete = type(adapter).complete
    live_complete_sha256 = _implementation_source_sha256(live_complete)
    if (
        live_complete is not _ALLOWED_DETERMINISTIC_MOCK_COMPLETE
        or live_class_sha256 != _ALLOWED_DETERMINISTIC_MOCK_CLASS_SOURCE_SHA256
        or live_complete_sha256 != _ALLOWED_DETERMINISTIC_MOCK_COMPLETE_SOURCE_SHA256
    ):
        raise LLMGatewayError(
            "deterministic mock implementation differs from the allowed source"
        )
    return {
        "adapter_type": (
            f"{DeterministicMockLLMAdapter.__module__}."
            f"{DeterministicMockLLMAdapter.__qualname__}"
        ),
        "allowed_class_source_sha256": (
            _ALLOWED_DETERMINISTIC_MOCK_CLASS_SOURCE_SHA256
        ),
        "allowed_complete_source_sha256": (
            _ALLOWED_DETERMINISTIC_MOCK_COMPLETE_SOURCE_SHA256
        ),
        "schema_version": DETERMINISTIC_MOCK_ADAPTER_ARTIFACT_SCHEMA_VERSION,
        "settings": {
            "action": adapter.action,
            "adapter_id": adapter.adapter_id,
            "confidence": adapter.confidence,
            "rationale_code": adapter.rationale_code,
            "review_step": adapter.review_step,
            "valid_until": adapter.valid_until.isoformat(),
        },
    }


def deterministic_mock_adapter_artifact_sha256(
    adapter: DeterministicMockLLMAdapter,
) -> str:
    """Commit to the exact trusted type, source, and all response settings."""
    return sha256_bytes(
        canonical_json_bytes(deterministic_mock_adapter_artifact_mapping(adapter))
    )


@dataclass(frozen=True, slots=True)
class AuditedAnonymizer:
    """Required audited entry point for text used by a planning adapter."""

    anonymizer: Anonymizer
    audit_log: SocialAuditLog
    cutoff_policy_sha256: str

    def sanitize(
        self, raw_text: str, *, event_key: EventKey | None = None
    ) -> SanitizedText:
        """Sanitize text and persist commitments without logging either text."""
        input_commitment = self.anonymizer.policy.input_commitment(raw_text)
        try:
            sanitized = self.anonymizer.sanitize(raw_text)
        except DeidentificationError:
            self.audit_log.record(
                stage=SocialAuditStage.ANONYMIZATION,
                outcome=SocialAuditOutcome.REJECTED,
                rule_code=SocialAuditRule.PRIVACY_VIOLATION,
                event_key=_event_key_value(event_key),
                policy_sha256=self.cutoff_policy_sha256,
                input_commitment=input_commitment,
                output_commitment=NULL_COMMITMENT,
                accepted_count=0,
                rejected_count=1,
            )
            raise
        self.audit_log.record(
            stage=SocialAuditStage.ANONYMIZATION,
            outcome=SocialAuditOutcome.ALLOWED,
            rule_code=SocialAuditRule.OK,
            event_key=_event_key_value(event_key),
            policy_sha256=self.cutoff_policy_sha256,
            input_commitment=sanitized.receipt.input_sha256,
            output_commitment=sanitized.receipt.output_sha256,
            accepted_count=1,
            rejected_count=0,
        )
        return sanitized


class ValidatedLLMGateway:
    """Validate privacy, time, schema, and actions around one offline adapter."""

    def __init__(
        self,
        *,
        adapter: LLMAdapter,
        authorization: AdapterAuthorization,
        deidentification_policy: DeidentificationPolicy,
        cutoff_policy_sha256: str,
        audit_log: SocialAuditLog,
    ) -> None:
        if not isinstance(adapter, LLMAdapter):
            raise LLMGatewayError("adapter must implement the strict LLMAdapter")
        if not isinstance(deidentification_policy, DeidentificationPolicy):
            raise LLMGatewayError(
                "deidentification_policy must be DeidentificationPolicy"
            )
        if not isinstance(audit_log, SocialAuditLog):
            raise LLMGatewayError("audit_log must be SocialAuditLog")
        if not isinstance(authorization, AdapterAuthorization):
            raise LLMGatewayError("authorization must be AdapterAuthorization")
        if authorization.provenance.adapter_id != adapter.adapter_id:
            raise LLMGatewayError("adapter ID does not match authorization provenance")
        artifact_kind = authorization.provenance.artifact_kind
        if artifact_kind is AdapterArtifactKind.DETERMINISTIC_MOCK:
            if type(adapter) is not DeterministicMockLLMAdapter:
                raise LLMGatewayError(
                    "deterministic-mock authorization requires the exact trusted "
                    "adapter type"
                )
            if (
                authorization.provenance.artifact_sha256
                != deterministic_mock_adapter_artifact_sha256(adapter)
            ):
                raise LLMGatewayError(
                    "deterministic mock settings or implementation do not match "
                    "authorization provenance"
                )
        elif isinstance(adapter, DeterministicMockLLMAdapter):
            raise LLMGatewayError(
                "deterministic mock requires deterministic-mock authorization"
            )
        if authorization.cutoff_policy_sha256 != cutoff_policy_sha256:
            raise LLMGatewayError("authorization cutoff policy mismatches gateway")
        if (
            authorization.deidentification_policy_sha256
            != deidentification_policy.sha256
        ):
            raise LLMGatewayError(
                "authorization deidentification policy mismatches gateway"
            )
        self._adapter = adapter
        self._authorization = authorization
        self._policy = deidentification_policy
        self._cutoff_policy_sha256 = cutoff_policy_sha256
        self._audit = audit_log

    @property
    def adapter_id(self) -> str:
        """Return the backend's stable identifier."""
        return self._adapter.adapter_id

    def plan_from_mapping(self, mapping: Mapping[str, object]) -> CognitivePlan:
        """Parse an exact request before the backend can observe any field."""
        input_commitment = _safe_commitment(mapping)
        try:
            request = StrictLLMRequest.from_mapping(mapping)
        except LLMContractError as error:
            self._audit.record(
                stage=SocialAuditStage.LLM_INPUT,
                outcome=SocialAuditOutcome.REJECTED,
                rule_code=_audit_rule(error),
                event_key=None,
                policy_sha256=self._authorization.sha256,
                input_commitment=input_commitment,
                output_commitment=NULL_COMMITMENT,
                accepted_count=0,
                rejected_count=1,
            )
            raise
        return self.plan(request)

    def plan(self, request: StrictLLMRequest) -> CognitivePlan:
        """Return a schema-validated CognitivePlan from an untrusted response."""
        if not isinstance(request, StrictLLMRequest):
            raise LLMGatewayError("request must be a StrictLLMRequest")
        try:
            adapter_complete = self._validate_request(request)
        except (
            LLMContractError,
            LLMKnowledgeFirewallError,
            UnsafeTextError,
        ) as error:
            self._audit.record(
                stage=SocialAuditStage.LLM_INPUT,
                outcome=SocialAuditOutcome.REJECTED,
                rule_code=_audit_rule(error),
                event_key=request.event_key.value,
                policy_sha256=self._authorization.sha256,
                input_commitment=request.input_sha256,
                output_commitment=NULL_COMMITMENT,
                accepted_count=0,
                rejected_count=1,
            )
            raise

        adapter_mapping = request.to_adapter_mapping()
        adapter_input_commitment = _safe_commitment(adapter_mapping)
        self._audit.record(
            stage=SocialAuditStage.LLM_INPUT,
            outcome=SocialAuditOutcome.ALLOWED,
            rule_code=SocialAuditRule.OK,
            event_key=request.event_key.value,
            policy_sha256=self._authorization.sha256,
            input_commitment=request.input_sha256,
            output_commitment=adapter_input_commitment,
            accepted_count=1,
            rejected_count=0,
        )
        try:
            raw_response = adapter_complete(adapter_mapping)
        except Exception as error:
            self._audit.record(
                stage=SocialAuditStage.LLM_OUTPUT,
                outcome=SocialAuditOutcome.REJECTED,
                rule_code=SocialAuditRule.ILLEGAL_ADAPTER_OUTPUT,
                event_key=request.event_key.value,
                policy_sha256=self._authorization.sha256,
                input_commitment=adapter_input_commitment,
                output_commitment=NULL_COMMITMENT,
                accepted_count=0,
                rejected_count=1,
            )
            raise LLMGatewayError("offline adapter failed") from error

        output_commitment = _safe_commitment(raw_response)
        try:
            if not isinstance(raw_response, Mapping):
                raise LLMOutputError("adapter output must be a mapping")
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
                adapter_id=self._adapter.adapter_id,
                adapter_authorization_sha256=self._authorization.sha256,
                requested_use=request.requested_use,
                rationale_code=response.rationale_code,
            )
        except (LLMContractError, ValueError) as error:
            self._audit.record(
                stage=SocialAuditStage.LLM_OUTPUT,
                outcome=SocialAuditOutcome.REJECTED,
                rule_code=SocialAuditRule.ILLEGAL_ADAPTER_OUTPUT,
                event_key=request.event_key.value,
                policy_sha256=self._authorization.sha256,
                input_commitment=adapter_input_commitment,
                output_commitment=output_commitment,
                accepted_count=0,
                rejected_count=1,
            )
            if isinstance(error, LLMContractError):
                raise
            raise LLMOutputError("adapter output cannot form a plan") from error

        self._audit.record(
            stage=SocialAuditStage.LLM_OUTPUT,
            outcome=SocialAuditOutcome.ALLOWED,
            rule_code=SocialAuditRule.OK,
            event_key=request.event_key.value,
            policy_sha256=self._authorization.sha256,
            input_commitment=output_commitment,
            output_commitment=_safe_commitment(plan.to_mapping()),
            accepted_count=1,
            rejected_count=0,
        )
        return plan

    def _validate_request(
        self, request: StrictLLMRequest
    ) -> Callable[[Mapping[str, object]], Mapping[str, object]]:
        adapter_complete = self._validate_live_adapter_integrity()
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
        if request.cutoff_policy_sha256 != self._cutoff_policy_sha256:
            raise LLMTemporalError("request cutoff policy does not match the gateway")
        if (
            request.deidentification_policy_sha256 != self._policy.sha256
            or request.sanitized_context.receipt.policy_sha256 != self._policy.sha256
        ):
            raise LLMPrivacyError(
                "request deidentification policy does not match the gateway"
            )
        Anonymizer(self._policy).assert_safe(request.sanitized_context)
        return adapter_complete

    def _validate_live_adapter_integrity(
        self,
    ) -> Callable[[Mapping[str, object]], Mapping[str, object]]:
        """Recheck mock type, source, and settings immediately before every call."""
        if (
            self._authorization.provenance.artifact_kind
            is not AdapterArtifactKind.DETERMINISTIC_MOCK
        ):
            return self._adapter.complete
        if type(self._adapter) is not DeterministicMockLLMAdapter:
            raise LLMKnowledgeFirewallError(
                "live deterministic mock no longer has the exact trusted type"
            )
        try:
            artifact_sha256 = deterministic_mock_adapter_artifact_sha256(self._adapter)
        except LLMGatewayError as error:
            raise LLMKnowledgeFirewallError(
                "live deterministic mock implementation cannot be verified"
            ) from error
        if artifact_sha256 != self._authorization.provenance.artifact_sha256:
            raise LLMKnowledgeFirewallError(
                "live deterministic mock settings or implementation changed after "
                "gateway construction"
            )
        adapter = self._adapter

        def authorized_complete(
            mapping: Mapping[str, object],
        ) -> Mapping[str, object]:
            return _ALLOWED_DETERMINISTIC_MOCK_COMPLETE(adapter, mapping)

        return authorized_complete


__all__ = [
    "AuditedAnonymizer",
    "DETERMINISTIC_MOCK_ADAPTER_ARTIFACT_SCHEMA_VERSION",
    "DeterministicMockLLMAdapter",
    "LLMAdapter",
    "LLMGatewayError",
    "ValidatedLLMGateway",
    "deterministic_mock_adapter_artifact_mapping",
    "deterministic_mock_adapter_artifact_sha256",
]
