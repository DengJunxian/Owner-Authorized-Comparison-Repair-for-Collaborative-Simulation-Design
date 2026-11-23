"""Frozen deterministic-mock planning hook behind the knowledge firewall.

The hook is deliberately narrower than an LLM integration.  It can only hold
the repository's offline deterministic adapter, a synthetic-test
authorization, and an already sanitized context.  It performs no I/O and is
ineligible for real-model or post-cutoff policy evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from policytwin.agents.llm_contracts import LLMPlanResponse, SanitizedContextRecord
from policytwin.agents.llm_firewall import (
    AdapterArtifactKind,
    AdapterAuthorization,
    LLMKnowledgeFirewallError,
)
from policytwin.agents.llm_gateway import (
    DeterministicMockLLMAdapter,
    LLMGatewayError,
    ValidatedLLMGateway,
    deterministic_mock_adapter_artifact_sha256,
)
from policytwin.cognition.planning import PlanAction, RequestedUse
from policytwin.provenance import canonical_json_bytes, sha256_bytes
from policytwin.social.audit import SocialAuditLog
from policytwin.social.privacy import (
    Anonymizer,
    DeidentificationPolicy,
    SanitizedText,
)

DETERMINISTIC_MOCK_PLANNING_SCHEMA_VERSION = (
    "policytwin.deterministic-mock-planning-hook.v2"
)
_VERIFIER_CONTEXT_TEXT = "Anonymous synthetic public-state planning request."
_VERIFIER_CONTEXT_LINEAGE_SHA256 = sha256_bytes(
    b"integrated-synthetic-source-lineage-v1"
)
_VERIFIER_CONTEXT_EVENT_TIME = datetime(2024, 12, 30, tzinfo=UTC)
_VERIFIER_CONTEXT_AVAILABLE_TIME = datetime(2024, 12, 31, tzinfo=UTC)
_VERIFIER_CONTEXT_INGESTION_TIME = datetime(2025, 1, 1, tzinfo=UTC)
_CONTEXT_AUTHORITIES: dict[int, tuple[object, str, str, str]] = {}


def issue_verifier_owned_integrated_mock_context(
    policy: DeidentificationPolicy,
) -> tuple[SanitizedText, SanitizedContextRecord, object]:
    """Issue the one repository-owned anonymous context fixture.

    This closed issuer intentionally accepts no text, time, or lineage input.
    The returned identity capability is retained only in memory and is never
    serialized into an artifact.
    """
    if not isinstance(policy, DeidentificationPolicy):
        raise TypeError("policy must be a DeidentificationPolicy")
    context = Anonymizer(policy).sanitize(_VERIFIER_CONTEXT_TEXT)
    record = SanitizedContextRecord(
        record_id="integrated-mock-context-v1",
        source="synthetic-public-state-fixture",
        version="v1",
        event_time=_VERIFIER_CONTEXT_EVENT_TIME,
        available_time=_VERIFIER_CONTEXT_AVAILABLE_TIME,
        ingestion_time=_VERIFIER_CONTEXT_INGESTION_TIME,
        content_sha256=context.receipt.output_sha256,
        lineage_sha256=_VERIFIER_CONTEXT_LINEAGE_SHA256,
    )
    capability = object()
    receipt_sha256 = sha256_bytes(
        canonical_json_bytes(
            {
                "context_output_sha256": context.receipt.output_sha256,
                "context_record_sha256": record.sha256,
                "issuer": "policytwin.verifier-owned-integrated-mock-context.v1",
            }
        )
    )
    _CONTEXT_AUTHORITIES[id(capability)] = (
        capability,
        context.receipt.output_sha256,
        record.sha256,
        receipt_sha256,
    )
    return context, record, capability


@dataclass(frozen=True, slots=True)
class DeterministicMockPlanningHook:
    """Immutable, synthetic-only inputs for one validated offline gateway."""

    adapter: DeterministicMockLLMAdapter
    authorization: AdapterAuthorization
    deidentification_policy: DeidentificationPolicy
    sanitized_context: SanitizedText
    sanitized_context_record: SanitizedContextRecord
    _context_authority: object
    synthetic_only: Literal[True] = True
    schema_version: str = DETERMINISTIC_MOCK_PLANNING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.adapter) is not DeterministicMockLLMAdapter:
            raise LLMKnowledgeFirewallError(
                "integrated mock planning requires the exact trusted "
                "DeterministicMockLLMAdapter type"
            )
        if not isinstance(self.authorization, AdapterAuthorization):
            raise LLMKnowledgeFirewallError(
                "integrated mock planning requires AdapterAuthorization"
            )
        if not isinstance(self.deidentification_policy, DeidentificationPolicy):
            raise LLMKnowledgeFirewallError(
                "integrated mock planning requires DeidentificationPolicy"
            )
        if not isinstance(self.sanitized_context, SanitizedText):
            raise LLMKnowledgeFirewallError(
                "integrated mock planning requires an already sanitized context"
            )
        if not isinstance(self.sanitized_context_record, SanitizedContextRecord):
            raise LLMKnowledgeFirewallError(
                "integrated mock planning requires a sanitized context temporal record"
            )
        authority = _CONTEXT_AUTHORITIES.get(id(self._context_authority))
        if authority is None or authority[0] is not self._context_authority:
            raise LLMKnowledgeFirewallError(
                "mock context lacks a verifier-owned authority capability"
            )
        if (
            authority[1] != self.sanitized_context.receipt.output_sha256
            or authority[2] != self.sanitized_context_record.sha256
        ):
            raise LLMKnowledgeFirewallError(
                "mock context or temporal record differs from its issuer commitment"
            )
        provenance = self.authorization.provenance
        if (
            provenance.artifact_kind is not AdapterArtifactKind.DETERMINISTIC_MOCK
            or self.authorization.requested_use is not RequestedUse.SYNTHETIC_TEST
            or provenance.synthetic_only is not True
        ):
            raise LLMKnowledgeFirewallError(
                "integrated mock planning is eligible only for deterministic "
                "synthetic-test provenance"
            )
        if provenance.adapter_id != self.adapter.adapter_id:
            raise LLMKnowledgeFirewallError(
                "mock adapter ID differs from its authorization provenance"
            )
        try:
            adapter_artifact_sha256 = deterministic_mock_adapter_artifact_sha256(
                self.adapter
            )
        except LLMGatewayError as error:
            raise LLMKnowledgeFirewallError(
                "mock adapter implementation cannot be verified"
            ) from error
        if provenance.artifact_sha256 != adapter_artifact_sha256:
            raise LLMKnowledgeFirewallError(
                "mock adapter settings or implementation differ from authorization "
                "provenance"
            )
        if (
            self.authorization.deidentification_policy_sha256
            != self.deidentification_policy.sha256
        ):
            raise LLMKnowledgeFirewallError(
                "mock authorization and deidentification policy differ"
            )
        if (
            self.authorization.sanitized_context_record_sha256
            != self.sanitized_context_record.sha256
        ):
            raise LLMKnowledgeFirewallError(
                "mock authorization and context temporal record differ"
            )
        Anonymizer(self.deidentification_policy).assert_safe(self.sanitized_context)
        self.sanitized_context_record.validate_for(
            self.sanitized_context,
            as_of_time=self.authorization.trusted_cutoff,
            cutoff=self.authorization.trusted_cutoff,
            lineage_sha256=self.sanitized_context_record.lineage_sha256,
        )
        # Validate every fixed response field at configuration time.  The live
        # request still revalidates its action allowlist and input commitment.
        try:
            action = PlanAction(self.adapter.action)
        except (TypeError, ValueError) as error:
            raise LLMKnowledgeFirewallError(
                "deterministic mock has an unsupported action"
            ) from error
        LLMPlanResponse(
            action=action,
            confidence=self.adapter.confidence,
            review_step=self.adapter.review_step,
            valid_until=self.adapter.valid_until,
            input_sha256="0" * 64,
            rationale_code=self.adapter.rationale_code,
        )
        if self.adapter.valid_until > self.authorization.trusted_cutoff:
            raise LLMKnowledgeFirewallError(
                "deterministic mock valid_until is after the trusted cutoff"
            )
        if self.synthetic_only is not True:
            raise LLMKnowledgeFirewallError(
                "deterministic mock planning hook is synthetic-only"
            )
        if self.schema_version != DETERMINISTIC_MOCK_PLANNING_SCHEMA_VERSION:
            raise LLMKnowledgeFirewallError(
                "unsupported deterministic-mock planning schema version"
            )

    @property
    def sha256(self) -> str:
        """Commit to all safe hook settings and authorization evidence."""
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def to_mapping(self) -> dict[str, object]:
        """Return the complete safe configuration; no unsanitized text appears."""
        authority = _CONTEXT_AUTHORITIES.get(id(self._context_authority))
        if authority is None or authority[0] is not self._context_authority:
            raise LLMKnowledgeFirewallError("mock context authority is no longer valid")
        return {
            "adapter": {
                "action": self.adapter.action,
                "adapter_id": self.adapter.adapter_id,
                "confidence": self.adapter.confidence,
                "rationale_code": self.adapter.rationale_code,
                "review_step": self.adapter.review_step,
                "valid_until": self.adapter.valid_until.isoformat(),
            },
            "authorization": self.authorization.to_mapping(),
            "context_authority_receipt_sha256": authority[3],
            "deidentification_policy": self.deidentification_policy.safe_mapping(),
            "sanitized_context": self.sanitized_context.to_mapping(),
            "sanitized_context_record": self.sanitized_context_record.to_mapping(),
            "schema_version": self.schema_version,
            "synthetic_only": self.synthetic_only,
        }

    def gateway(self, audit_log: SocialAuditLog) -> ValidatedLLMGateway:
        """Construct an isolated validated gateway over the supplied audit chain."""
        if not isinstance(audit_log, SocialAuditLog):
            raise TypeError("audit_log must be a SocialAuditLog")
        # Recompute the artifact commitment here as well as at construction so
        # a runtime method replacement cannot reuse a previously valid hook.
        try:
            artifact_sha256 = deterministic_mock_adapter_artifact_sha256(self.adapter)
        except LLMGatewayError as error:
            raise LLMKnowledgeFirewallError(
                "mock adapter implementation cannot be verified at execution"
            ) from error
        if artifact_sha256 != self.authorization.provenance.artifact_sha256:
            raise LLMKnowledgeFirewallError("mock adapter changed after authorization")
        authority = _CONTEXT_AUTHORITIES.get(id(self._context_authority))
        if (
            authority is None
            or authority[0] is not self._context_authority
            or authority[1] != self.sanitized_context.receipt.output_sha256
            or authority[2] != self.sanitized_context_record.sha256
        ):
            raise LLMKnowledgeFirewallError(
                "mock context changed after verifier authorization"
            )
        return ValidatedLLMGateway(
            adapter=self.adapter,
            authorization=self.authorization,
            deidentification_policy=self.deidentification_policy,
            cutoff_policy_sha256=self.authorization.cutoff_policy_sha256,
            audit_log=audit_log,
        )


__all__ = [
    "DETERMINISTIC_MOCK_PLANNING_SCHEMA_VERSION",
    "DeterministicMockPlanningHook",
    "issue_verifier_owned_integrated_mock_context",
]
