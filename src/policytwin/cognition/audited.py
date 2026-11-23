"""Audited orchestration for immutable memory and semantic transitions."""

from __future__ import annotations

from dataclasses import dataclass

from policytwin.cognition.memory import (
    CognitiveMemoryError,
    CognitiveMemoryTemporalError,
    EpisodicMemory,
    SemanticState,
)
from policytwin.counterfactual import EventKey
from policytwin.provenance import canonical_json_bytes, sha256_bytes
from policytwin.social.audit import (
    NULL_COMMITMENT,
    SocialAuditLog,
    SocialAuditOutcome,
    SocialAuditRule,
    SocialAuditStage,
)
from policytwin.social.events import InformationEvent, InformationEventKind


class AuditedCognitionError(ValueError):
    """Raised when a cognitive snapshot or audited transition is inconsistent."""


def _commit(value: object) -> str:
    return sha256_bytes(canonical_json_bytes(value))


@dataclass(frozen=True, slots=True)
class CognitiveSnapshot:
    """One immutable memory and semantic state under a shared firewall."""

    episodic_memory: EpisodicMemory
    semantic_state: SemanticState

    def __post_init__(self) -> None:
        if not isinstance(self.episodic_memory, EpisodicMemory):
            raise AuditedCognitionError("episodic_memory must be EpisodicMemory")
        if not isinstance(self.semantic_state, SemanticState):
            raise AuditedCognitionError("semantic_state must be SemanticState")
        if (
            self.episodic_memory.trusted_cutoff != self.semantic_state.trusted_cutoff
            or self.episodic_memory.cutoff_policy_sha256
            != self.semantic_state.cutoff_policy_sha256
            or self.episodic_memory.lineage_sha256 != self.semantic_state.lineage_sha256
        ):
            raise AuditedCognitionError(
                "memory and semantic state firewall commitments must match"
            )

    def to_mapping(self) -> dict[str, object]:
        """Return the complete private scientific state for commitment only."""
        return {
            "episodic_memory": self.episodic_memory.to_mapping(),
            "semantic_state": self.semantic_state.to_mapping(),
        }


@dataclass(frozen=True, slots=True)
class AuditedCognitiveLayer:
    """Required run-facing gateway for memory and belief transitions."""

    audit_log: SocialAuditLog
    cutoff_policy_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.audit_log, SocialAuditLog):
            raise AuditedCognitionError("audit_log must be SocialAuditLog")
        if (
            not isinstance(self.cutoff_policy_sha256, str)
            or len(self.cutoff_policy_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.cutoff_policy_sha256
            )
        ):
            raise AuditedCognitionError(
                "cutoff_policy_sha256 must be a lowercase SHA-256 digest"
            )

    def attend(
        self,
        snapshot: CognitiveSnapshot,
        event: InformationEvent,
        *,
        event_key: EventKey,
        attention: float,
        strength: float,
        target_belief: float | None = None,
        correction_fraction: float | None = None,
    ) -> CognitiveSnapshot:
        """Audit and atomically derive memory plus semantic belief state."""
        if not isinstance(snapshot, CognitiveSnapshot):
            raise AuditedCognitionError("snapshot must be CognitiveSnapshot")
        if not isinstance(event, InformationEvent):
            raise AuditedCognitionError("event must be InformationEvent")
        if not isinstance(event_key, EventKey):
            raise AuditedCognitionError("event_key must be EventKey")
        input_value = {
            "attention": attention,
            "correction_fraction": correction_fraction,
            "event": event.to_mapping(),
            "snapshot": snapshot.to_mapping(),
            "strength": strength,
            "target_belief": target_belief,
        }
        input_commitment = _commit(input_value)
        try:
            if (
                snapshot.episodic_memory.cutoff_policy_sha256
                != self.cutoff_policy_sha256
            ):
                raise CognitiveMemoryTemporalError(
                    "snapshot cutoff policy does not match audited layer"
                )
            memory = snapshot.episodic_memory.append_information_event(
                event,
                attention=attention,
                strength=strength,
            )
            if event.event_kind is InformationEventKind.CORRECTION:
                if target_belief is not None or correction_fraction is None:
                    raise CognitiveMemoryError(
                        "correction requires correction_fraction only"
                    )
                semantic = snapshot.semantic_state.update(
                    event,
                    correction_fraction=correction_fraction,
                    attention=attention,
                )
            else:
                if target_belief is None or correction_fraction is not None:
                    raise CognitiveMemoryError(
                        "ordinary event requires target_belief only"
                    )
                semantic = snapshot.semantic_state.update(
                    event,
                    target_belief=target_belief,
                    attention=attention,
                )
            output = CognitiveSnapshot(memory, semantic)
        except CognitiveMemoryError as error:
            self.audit_log.record(
                stage=SocialAuditStage.MEMORY,
                outcome=SocialAuditOutcome.REJECTED,
                rule_code=(
                    SocialAuditRule.TEMPORAL_BOUNDARY
                    if isinstance(error, CognitiveMemoryTemporalError)
                    else SocialAuditRule.SCHEMA_VIOLATION
                ),
                event_key=event_key.value,
                policy_sha256=self.cutoff_policy_sha256,
                input_commitment=input_commitment,
                output_commitment=NULL_COMMITMENT,
                accepted_count=0,
                rejected_count=1,
            )
            raise
        self.audit_log.record(
            stage=SocialAuditStage.MEMORY,
            outcome=SocialAuditOutcome.ALLOWED,
            rule_code=SocialAuditRule.OK,
            event_key=event_key.value,
            policy_sha256=self.cutoff_policy_sha256,
            input_commitment=input_commitment,
            output_commitment=_commit(output.to_mapping()),
            accepted_count=1,
            rejected_count=0,
        )
        return output


__all__ = ["AuditedCognitionError", "AuditedCognitiveLayer", "CognitiveSnapshot"]
