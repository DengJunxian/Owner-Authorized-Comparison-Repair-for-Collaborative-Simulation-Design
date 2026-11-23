"""Outcome-knowledge exposure ledgers and fail-closed design temporality.

Temporal record cutoffs cannot erase facts already seen by a design-influencing
team.  This module records that separate scientific information boundary
without storing result text, values, or directions.  Locator-only entries prove
contamination conservatively; they do not verify the cited policy or result.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import NoReturn, cast

from policytwin.protocol import StudyMode, StudyProtocol
from policytwin.provenance import (
    MetadataValidationError,
    canonical_json_bytes,
    read_json_bytes_object,
    sha256_bytes,
)

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class OutcomeKnowledgeError(ValueError):
    """Raised when an outcome-knowledge declaration is unsafe or inconsistent."""


def _fail(message: str) -> NoReturn:
    raise OutcomeKnowledgeError(message)


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        _fail(f"{field} must be a stable identifier")
    return value


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{field} must be non-empty text")
    return value


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail(f"{field} must be a lowercase SHA-256 digest")
    return value


def _optional_sha256(value: object, field: str) -> str | None:
    if value is None:
        return None
    return _sha256(value, field)


def _aware(value: object, field: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as error:
            raise OutcomeKnowledgeError(f"{field} must be ISO 8601") from error
    else:
        _fail(f"{field} must be an aware datetime")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        _fail(f"{field} must include a timezone")
    return parsed.astimezone(UTC)


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        _fail(f"{field} must be an object with string keys")
    return cast(Mapping[str, object], value)


def _sequence(value: object, field: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        _fail(f"{field} must be an array")
    return cast(Sequence[object], value)


def _fields(
    mapping: Mapping[str, object], expected: frozenset[str], context: str
) -> None:
    missing = sorted(expected - frozenset(mapping))
    unknown = sorted(frozenset(mapping) - expected)
    if missing:
        _fail(f"{context} missing fields: {', '.join(missing)}")
    if unknown:
        _fail(f"{context} has unknown fields: {', '.join(unknown)}")


def _enum[T: StrEnum](enum_type: type[T], value: object, field: str) -> T:
    if not isinstance(value, str):
        _fail(f"{field} must be a string")
    try:
        return enum_type(value)
    except ValueError as error:
        allowed = ", ".join(item.value for item in enum_type)
        raise OutcomeKnowledgeError(f"{field} must be one of: {allowed}") from error


class KnowledgeAudience(StrEnum):
    """Who obtained the outcome information."""

    DESIGN_INFLUENCING_TEAM = "design_influencing_team"
    INDEPENDENT_BLIND_CUSTODIAN = "independent_blind_custodian"


class OutcomeKnowledgeFacet(StrEnum):
    """Result dimensions exposed, without retaining their content."""

    POST_POLICY_SUMMARY = "post_policy_summary"
    EFFECT_DIRECTION = "effect_direction"
    EFFECT_MAGNITUDE = "effect_magnitude"
    SUBGROUP_PATTERN = "subgroup_pattern"
    STATISTICAL_SIGNIFICANCE = "statistical_significance"


class SourceEvidenceStatus(StrEnum):
    """Whether an exposure source is merely located or locally verified."""

    LOCATOR_ONLY = "locator_only"
    ARCHIVED_HASH_VERIFIED = "archived_hash_verified"


class DisclosureStatus(StrEnum):
    """Authority boundary for the claimed completeness of one ledger."""

    SELF_REPORTED_INCOMPLETE = "self_reported_incomplete"
    EXTERNALLY_VERIFIED_COMPLETE = "externally_verified_complete"


class DesignTemporality(StrEnum):
    """Derived maximum interpretation of a study's knowledge chronology."""

    SYNTHETIC_ONLY = "synthetic_only"
    PSEUDO_PROSPECTIVE_BLINDED = "pseudo_prospective_blinded_at_registration"
    RETROSPECTIVE_OUTCOME_KNOWN = "retrospective_outcome_known_design"
    UNVERIFIED_ABSTAIN = "outcome_knowledge_unverified_abstain"


class DesignTemporalityReason(StrEnum):
    """Stable reason codes for machine-readable release gates."""

    SYNTHETIC_PROTOCOL = "synthetic_protocol"
    MANIFEST_REQUIRED = "outcome_knowledge_manifest_required"
    TARGET_EVENT_MISMATCH = "target_event_mismatch"
    PRE_REGISTRATION_TEAM_EXPOSURE = "pre_registration_team_exposure"
    REQUESTED_PSEUDO_CONTRADICTS_EXPOSURE = (
        "requested_pseudo_prospective_contradicts_exposure"
    )
    REQUESTED_RETROSPECTIVE_CONFIRMED = "requested_retrospective_confirmed"
    DISCLOSURE_NOT_EXTERNALLY_VERIFIED = "disclosure_not_externally_verified"
    EXTERNAL_VERIFIER_UNAVAILABLE = "external_registration_verifier_unavailable"
    REQUESTED_MODE_UNSUPPORTED = "requested_mode_unsupported"


@dataclass(frozen=True, slots=True)
class OutcomeKnowledgeExposure:
    """One conservative declaration that a result facet was known by a time."""

    schema_version: int
    exposure_id: str
    target_event_id: str
    exposure_known_by: datetime
    audience: KnowledgeAudience
    knowledge_facets: tuple[OutcomeKnowledgeFacet, ...]
    source_locator: str
    source_artifact_sha256: str | None
    source_evidence_status: SourceEvidenceStatus

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            _fail("outcome-knowledge exposure schema_version must equal 1")
        _identifier(self.exposure_id, "exposure_id")
        _identifier(self.target_event_id, "target_event_id")
        object.__setattr__(
            self,
            "exposure_known_by",
            _aware(self.exposure_known_by, "exposure_known_by"),
        )
        if not isinstance(self.audience, KnowledgeAudience):
            _fail("audience must be a KnowledgeAudience")
        if not isinstance(self.knowledge_facets, tuple) or not self.knowledge_facets:
            _fail("knowledge_facets must be a non-empty immutable tuple")
        if not all(
            isinstance(item, OutcomeKnowledgeFacet) for item in self.knowledge_facets
        ):
            _fail("knowledge_facets contains an invalid facet")
        values = tuple(item.value for item in self.knowledge_facets)
        if values != tuple(sorted(set(values))):
            _fail("knowledge_facets must be unique and canonically sorted")
        _text(self.source_locator, "source_locator")
        digest = _optional_sha256(self.source_artifact_sha256, "source_artifact_sha256")
        if not isinstance(self.source_evidence_status, SourceEvidenceStatus):
            _fail("source_evidence_status must be a SourceEvidenceStatus")
        if self.source_evidence_status is SourceEvidenceStatus.LOCATOR_ONLY:
            if digest is not None:
                _fail("locator-only exposure must not claim an artifact hash")
        elif digest is None:
            _fail("archived exposure requires a verified artifact hash")

    def to_mapping(self) -> dict[str, object]:
        """Return a text-free result-facet commitment."""
        return {
            "audience": self.audience.value,
            "exposure_id": self.exposure_id,
            "exposure_known_by": self.exposure_known_by.isoformat(),
            "knowledge_facets": [item.value for item in self.knowledge_facets],
            "schema_version": self.schema_version,
            "source_artifact_sha256": self.source_artifact_sha256,
            "source_evidence_status": self.source_evidence_status.value,
            "source_locator": self.source_locator,
            "target_event_id": self.target_event_id,
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> OutcomeKnowledgeExposure:
        """Strictly parse one exposure without defaults."""
        _fields(
            mapping,
            frozenset(
                {
                    "audience",
                    "exposure_id",
                    "exposure_known_by",
                    "knowledge_facets",
                    "schema_version",
                    "source_artifact_sha256",
                    "source_evidence_status",
                    "source_locator",
                    "target_event_id",
                }
            ),
            "outcome-knowledge exposure",
        )
        version = mapping["schema_version"]
        if isinstance(version, bool) or not isinstance(version, int):
            _fail("exposure schema_version must be an integer")
        return cls(
            schema_version=version,
            exposure_id=_identifier(mapping["exposure_id"], "exposure_id"),
            target_event_id=_identifier(mapping["target_event_id"], "target_event_id"),
            exposure_known_by=_aware(mapping["exposure_known_by"], "exposure_known_by"),
            audience=_enum(KnowledgeAudience, mapping["audience"], "audience"),
            knowledge_facets=tuple(
                _enum(OutcomeKnowledgeFacet, item, "knowledge_facets item")
                for item in _sequence(mapping["knowledge_facets"], "knowledge_facets")
            ),
            source_locator=_text(mapping["source_locator"], "source_locator"),
            source_artifact_sha256=_optional_sha256(
                mapping["source_artifact_sha256"], "source_artifact_sha256"
            ),
            source_evidence_status=_enum(
                SourceEvidenceStatus,
                mapping["source_evidence_status"],
                "source_evidence_status",
            ),
        )


@dataclass(frozen=True, slots=True)
class OutcomeKnowledgeManifest:
    """Immutable event-specific ledger with an explicit completeness boundary."""

    schema_version: int
    manifest_id: str
    target_event_id: str
    declared_complete_through: datetime
    disclosure_status: DisclosureStatus
    external_verification_receipt_sha256: str | None
    entries: tuple[OutcomeKnowledgeExposure, ...]

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            _fail("outcome-knowledge manifest schema_version must equal 1")
        _identifier(self.manifest_id, "manifest_id")
        _identifier(self.target_event_id, "target_event_id")
        object.__setattr__(
            self,
            "declared_complete_through",
            _aware(self.declared_complete_through, "declared_complete_through"),
        )
        if not isinstance(self.disclosure_status, DisclosureStatus):
            _fail("disclosure_status must be a DisclosureStatus")
        receipt = _optional_sha256(
            self.external_verification_receipt_sha256,
            "external_verification_receipt_sha256",
        )
        if self.disclosure_status is DisclosureStatus.SELF_REPORTED_INCOMPLETE:
            if receipt is not None:
                _fail("self-reported exposure ledger cannot claim an external receipt")
        elif receipt is None:
            _fail("externally verified disclosure requires a receipt commitment")
        if not isinstance(self.entries, tuple):
            _fail("entries must be an immutable tuple")
        if not all(isinstance(item, OutcomeKnowledgeExposure) for item in self.entries):
            _fail("entries contains an invalid exposure")
        ids = tuple(item.exposure_id for item in self.entries)
        if len(set(ids)) != len(ids):
            _fail("exposure IDs must be unique")
        if any(item.target_event_id != self.target_event_id for item in self.entries):
            _fail("every exposure must reference the manifest target event")
        ordering = tuple(
            (item.exposure_known_by, item.exposure_id) for item in self.entries
        )
        if ordering != tuple(sorted(ordering)):
            _fail("exposures must be ordered by known time and exposure ID")
        if any(
            item.exposure_known_by > self.declared_complete_through
            for item in self.entries
        ):
            _fail("an exposure follows the manifest completeness boundary")

    @property
    def sha256(self) -> str:
        """Return the canonical semantic manifest commitment."""
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def to_mapping(self) -> dict[str, object]:
        """Return a deterministic representation."""
        return {
            "declared_complete_through": self.declared_complete_through.isoformat(),
            "disclosure_status": self.disclosure_status.value,
            "entries": [item.to_mapping() for item in self.entries],
            "external_verification_receipt_sha256": (
                self.external_verification_receipt_sha256
            ),
            "manifest_id": self.manifest_id,
            "schema_version": self.schema_version,
            "target_event_id": self.target_event_id,
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> OutcomeKnowledgeManifest:
        """Strictly parse a complete exposure manifest."""
        _fields(
            mapping,
            frozenset(
                {
                    "declared_complete_through",
                    "disclosure_status",
                    "entries",
                    "external_verification_receipt_sha256",
                    "manifest_id",
                    "schema_version",
                    "target_event_id",
                }
            ),
            "outcome-knowledge manifest",
        )
        version = mapping["schema_version"]
        if isinstance(version, bool) or not isinstance(version, int):
            _fail("manifest schema_version must be an integer")
        return cls(
            schema_version=version,
            manifest_id=_identifier(mapping["manifest_id"], "manifest_id"),
            target_event_id=_identifier(mapping["target_event_id"], "target_event_id"),
            declared_complete_through=_aware(
                mapping["declared_complete_through"],
                "declared_complete_through",
            ),
            disclosure_status=_enum(
                DisclosureStatus,
                mapping["disclosure_status"],
                "disclosure_status",
            ),
            external_verification_receipt_sha256=_optional_sha256(
                mapping["external_verification_receipt_sha256"],
                "external_verification_receipt_sha256",
            ),
            entries=tuple(
                OutcomeKnowledgeExposure.from_mapping(
                    _mapping(item, "outcome-knowledge entry")
                )
                for item in _sequence(mapping["entries"], "entries")
            ),
        )


@dataclass(frozen=True, slots=True)
class LoadedOutcomeKnowledgeManifest:
    """Manifest plus exact source-byte and semantic commitments."""

    manifest: OutcomeKnowledgeManifest
    source_sha256: str
    semantic_sha256: str

    def __post_init__(self) -> None:
        _sha256(self.source_sha256, "source_sha256")
        _sha256(self.semantic_sha256, "semantic_sha256")
        if self.manifest.sha256 != self.semantic_sha256:
            _fail("loaded exposure manifest semantic hash mismatch")


def load_outcome_knowledge_manifest(
    path: Path, *, expected_source_sha256: str
) -> LoadedOutcomeKnowledgeManifest:
    """Load a manifest only when its exact bytes match an external commitment."""
    expected = _sha256(expected_source_sha256, "expected_source_sha256")
    try:
        content = path.read_bytes()
    except OSError as error:
        raise OutcomeKnowledgeError(
            f"cannot read exposure manifest: {error}"
        ) from error
    observed = sha256_bytes(content)
    if observed != expected:
        _fail("outcome-knowledge source hash does not match its commitment")
    try:
        mapping = read_json_bytes_object(content, source=str(path))
    except MetadataValidationError as error:
        raise OutcomeKnowledgeError(str(error)) from error
    manifest = OutcomeKnowledgeManifest.from_mapping(mapping)
    return LoadedOutcomeKnowledgeManifest(
        manifest=manifest,
        source_sha256=observed,
        semantic_sha256=manifest.sha256,
    )


@dataclass(frozen=True, slots=True)
class DesignTemporalityDecision:
    """Non-authorizing derived classification bound to protocol and ledger."""

    schema_version: int
    protocol_sha256: str
    target_event_id: str
    manifest_source_sha256: str | None
    manifest_semantic_sha256: str | None
    requested_mode: StudyMode
    derived_temporality: DesignTemporality
    pseudo_prospective_eligible: bool
    contaminating_exposure_ids: tuple[str, ...]
    reasons: tuple[DesignTemporalityReason, ...]

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            _fail("design-temporality decision schema_version must equal 1")
        _sha256(self.protocol_sha256, "protocol_sha256")
        _identifier(self.target_event_id, "target_event_id")
        _optional_sha256(self.manifest_source_sha256, "manifest_source_sha256")
        _optional_sha256(self.manifest_semantic_sha256, "manifest_semantic_sha256")
        if (self.manifest_source_sha256 is None) != (
            self.manifest_semantic_sha256 is None
        ):
            _fail("manifest source and semantic hashes must be present together")
        if not isinstance(self.requested_mode, StudyMode):
            _fail("requested_mode must be a StudyMode")
        if not isinstance(self.derived_temporality, DesignTemporality):
            _fail("derived_temporality must be a DesignTemporality")
        if not isinstance(self.pseudo_prospective_eligible, bool):
            _fail("pseudo_prospective_eligible must be boolean")
        if not isinstance(self.contaminating_exposure_ids, tuple):
            _fail("contaminating_exposure_ids must be immutable")
        for item in self.contaminating_exposure_ids:
            _identifier(item, "contaminating exposure ID")
        if tuple(sorted(set(self.contaminating_exposure_ids))) != (
            self.contaminating_exposure_ids
        ):
            _fail("contaminating exposure IDs must be unique and sorted")
        if not isinstance(self.reasons, tuple) or not self.reasons:
            _fail("design-temporality reasons must be a non-empty tuple")
        if not all(isinstance(item, DesignTemporalityReason) for item in self.reasons):
            _fail("design-temporality reasons contain an invalid code")
        if len(set(self.reasons)) != len(self.reasons):
            _fail("design-temporality reasons must be unique")
        if self.pseudo_prospective_eligible or (
            self.derived_temporality is DesignTemporality.PSEUDO_PROSPECTIVE_BLINDED
        ):
            _fail(
                "local design-temporality decisions cannot authorize a blinded "
                "pseudo-prospective label"
            )

        manifest_present = self.manifest_source_sha256 is not None
        if self.requested_mode is StudyMode.SYNTHETIC:
            if (
                self.derived_temporality is not DesignTemporality.SYNTHETIC_ONLY
                or manifest_present
                or self.contaminating_exposure_ids
                or self.reasons != (DesignTemporalityReason.SYNTHETIC_PROTOCOL,)
            ):
                _fail("synthetic design-temporality decision is inconsistent")
            return
        if self.derived_temporality is DesignTemporality.SYNTHETIC_ONLY:
            _fail("a real-study request cannot derive synthetic temporality")
        if self.derived_temporality is DesignTemporality.RETROSPECTIVE_OUTCOME_KNOWN:
            if not manifest_present or not self.contaminating_exposure_ids:
                _fail("retrospective temporality requires a committed exposure")
            expected_reason = (
                DesignTemporalityReason.REQUESTED_PSEUDO_CONTRADICTS_EXPOSURE
                if self.requested_mode is StudyMode.PSEUDO_PROSPECTIVE
                else DesignTemporalityReason.REQUESTED_RETROSPECTIVE_CONFIRMED
            )
            if self.reasons != (
                DesignTemporalityReason.PRE_REGISTRATION_TEAM_EXPOSURE,
                expected_reason,
            ):
                _fail("retrospective temporality reasons are inconsistent")
            return
        if self.derived_temporality is not DesignTemporality.UNVERIFIED_ABSTAIN:
            _fail("unsupported local design temporality")
        if self.contaminating_exposure_ids:
            _fail("an unverified decision cannot retain contaminating exposure IDs")
        if not manifest_present:
            if self.reasons != (DesignTemporalityReason.MANIFEST_REQUIRED,):
                _fail("missing-manifest temporality reasons are inconsistent")
            return
        allowed_reasons = {
            (DesignTemporalityReason.TARGET_EVENT_MISMATCH,),
            (DesignTemporalityReason.EXTERNAL_VERIFIER_UNAVAILABLE,),
            (
                DesignTemporalityReason.DISCLOSURE_NOT_EXTERNALLY_VERIFIED,
                DesignTemporalityReason.EXTERNAL_VERIFIER_UNAVAILABLE,
            ),
        }
        if self.reasons not in allowed_reasons:
            _fail("unverified temporality reasons are inconsistent")

    @property
    def sha256(self) -> str:
        """Return the complete decision commitment."""
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def to_mapping(self) -> dict[str, object]:
        """Return a deterministic non-authorizing audit record."""
        return {
            "contaminating_exposure_ids": list(self.contaminating_exposure_ids),
            "derived_temporality": self.derived_temporality.value,
            "manifest_semantic_sha256": self.manifest_semantic_sha256,
            "manifest_source_sha256": self.manifest_source_sha256,
            "protocol_sha256": self.protocol_sha256,
            "pseudo_prospective_eligible": self.pseudo_prospective_eligible,
            "reasons": [item.value for item in self.reasons],
            "requested_mode": self.requested_mode.value,
            "schema_version": self.schema_version,
            "target_event_id": self.target_event_id,
        }


def assess_design_temporality(
    protocol: StudyProtocol,
    *,
    target_event_id: str,
    manifest: LoadedOutcomeKnowledgeManifest | None,
) -> DesignTemporalityDecision:
    """Derive the maximum honest design label under current local authority.

    This repository has no external registration/knowledge verifier.  Therefore
    an empty or apparently complete manifest can never authorize a real
    pseudo-prospective label here.  A pre-registration team exposure is still
    sufficient to derive the conservative retrospective label.
    """
    if not isinstance(protocol, StudyProtocol):
        _fail("design temporality requires a typed StudyProtocol")
    event_id = _identifier(target_event_id, "target_event_id")
    if protocol.mode is StudyMode.SYNTHETIC:
        if manifest is not None:
            _fail("synthetic design temporality must not consume a real-event ledger")
        return DesignTemporalityDecision(
            schema_version=1,
            protocol_sha256=protocol.sha256,
            target_event_id=event_id,
            manifest_source_sha256=None,
            manifest_semantic_sha256=None,
            requested_mode=protocol.mode,
            derived_temporality=DesignTemporality.SYNTHETIC_ONLY,
            pseudo_prospective_eligible=False,
            contaminating_exposure_ids=(),
            reasons=(DesignTemporalityReason.SYNTHETIC_PROTOCOL,),
        )
    if manifest is None:
        return DesignTemporalityDecision(
            schema_version=1,
            protocol_sha256=protocol.sha256,
            target_event_id=event_id,
            manifest_source_sha256=None,
            manifest_semantic_sha256=None,
            requested_mode=protocol.mode,
            derived_temporality=DesignTemporality.UNVERIFIED_ABSTAIN,
            pseudo_prospective_eligible=False,
            contaminating_exposure_ids=(),
            reasons=(DesignTemporalityReason.MANIFEST_REQUIRED,),
        )
    ledger = manifest.manifest
    if ledger.target_event_id != event_id:
        return DesignTemporalityDecision(
            schema_version=1,
            protocol_sha256=protocol.sha256,
            target_event_id=event_id,
            manifest_source_sha256=manifest.source_sha256,
            manifest_semantic_sha256=manifest.semantic_sha256,
            requested_mode=protocol.mode,
            derived_temporality=DesignTemporality.UNVERIFIED_ABSTAIN,
            pseudo_prospective_eligible=False,
            contaminating_exposure_ids=(),
            reasons=(DesignTemporalityReason.TARGET_EVENT_MISMATCH,),
        )
    contaminating = tuple(
        sorted(
            item.exposure_id
            for item in ledger.entries
            if item.audience is KnowledgeAudience.DESIGN_INFLUENCING_TEAM
            and item.exposure_known_by <= protocol.registered_at
        )
    )
    if contaminating:
        reasons = [DesignTemporalityReason.PRE_REGISTRATION_TEAM_EXPOSURE]
        if protocol.mode is StudyMode.PSEUDO_PROSPECTIVE:
            reasons.append(
                DesignTemporalityReason.REQUESTED_PSEUDO_CONTRADICTS_EXPOSURE
            )
        else:
            reasons.append(DesignTemporalityReason.REQUESTED_RETROSPECTIVE_CONFIRMED)
        return DesignTemporalityDecision(
            schema_version=1,
            protocol_sha256=protocol.sha256,
            target_event_id=event_id,
            manifest_source_sha256=manifest.source_sha256,
            manifest_semantic_sha256=manifest.semantic_sha256,
            requested_mode=protocol.mode,
            derived_temporality=DesignTemporality.RETROSPECTIVE_OUTCOME_KNOWN,
            pseudo_prospective_eligible=False,
            contaminating_exposure_ids=contaminating,
            reasons=tuple(reasons),
        )
    reasons = [DesignTemporalityReason.EXTERNAL_VERIFIER_UNAVAILABLE]
    if ledger.disclosure_status is not DisclosureStatus.EXTERNALLY_VERIFIED_COMPLETE:
        reasons.insert(0, DesignTemporalityReason.DISCLOSURE_NOT_EXTERNALLY_VERIFIED)
    return DesignTemporalityDecision(
        schema_version=1,
        protocol_sha256=protocol.sha256,
        target_event_id=event_id,
        manifest_source_sha256=manifest.source_sha256,
        manifest_semantic_sha256=manifest.semantic_sha256,
        requested_mode=protocol.mode,
        derived_temporality=DesignTemporality.UNVERIFIED_ABSTAIN,
        pseudo_prospective_eligible=False,
        contaminating_exposure_ids=(),
        reasons=tuple(reasons),
    )


__all__ = [
    "DesignTemporality",
    "DesignTemporalityDecision",
    "DesignTemporalityReason",
    "DisclosureStatus",
    "KnowledgeAudience",
    "LoadedOutcomeKnowledgeManifest",
    "OutcomeKnowledgeError",
    "OutcomeKnowledgeExposure",
    "OutcomeKnowledgeFacet",
    "OutcomeKnowledgeManifest",
    "SourceEvidenceStatus",
    "assess_design_temporality",
    "load_outcome_knowledge_manifest",
]
