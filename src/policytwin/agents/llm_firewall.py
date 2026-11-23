"""Immutable knowledge-firewall authorization for optional planning adapters.

This module does not load or call a model.  It records the commitments and
closed eligibility decision that must exist before either an LLM-shaped
adapter or a distilled policy can observe a planning request.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Self, cast

from policytwin.cognition.planning import RequestedUse
from policytwin.provenance import canonical_json_bytes, sha256_bytes

KNOWLEDGE_FIREWALL_SCHEMA_VERSION = 2

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_PROVENANCE_FIELDS = frozenset(
    {
        "adapter_id",
        "anonymous_structured_inputs_only",
        "artifact_kind",
        "artifact_sha256",
        "environment_lock_sha256",
        "network_access_permitted",
        "provenance_manifest_sha256",
        "retrieval_permitted",
        "schema_version",
        "synthetic_only",
        "training_data_cutoff",
        "training_manifest_sha256",
        "verification_manifest_sha256",
        "weights_available_time",
        "weights_frozen_time",
    }
)
_PROBE_DEFINITION_FIELDS = frozenset(
    {"grading_rule_sha256", "input_sha256", "kind", "probe_id"}
)
_PROBE_MANIFEST_FIELDS = frozenset(
    {
        "definitions",
        "max_named_anonymous_sensitivity",
        "provenance_sha256",
        "schema_version",
        "trusted_cutoff",
    }
)
_PROBE_RESULT_FIELDS = frozenset(
    {
        "forbidden_revelation_detected",
        "input_sha256",
        "kind",
        "output_sha256",
        "probe_id",
        "sensitivity_score",
    }
)
_PROBE_REPORT_FIELDS = frozenset(
    {
        "manifest_sha256",
        "offline_enforced",
        "provenance_sha256",
        "results",
        "retrieval_disabled",
        "schema_version",
    }
)
_AUTHORIZATION_FIELDS = frozenset(
    {
        "cutoff_policy_sha256",
        "deidentification_policy_sha256",
        "leak_probe_manifest",
        "leak_probe_report",
        "prompt_template_sha256",
        "sanitized_context_record_sha256",
        "provenance",
        "requested_use",
        "schema_version",
        "trusted_cutoff",
    }
)


class LLMKnowledgeFirewallError(ValueError):
    """Raised when model knowledge provenance is not eligible for its use."""


class AdapterArtifactKind(StrEnum):
    """Closed provenance classes with distinct scientific permissions."""

    DETERMINISTIC_MOCK = "deterministic_mock"
    PRE_CUTOFF_MODEL = "pre_cutoff_model"
    DISTILLED_PRE_CUTOFF_POLICY = "distilled_pre_cutoff_policy"
    CONTEMPORARY_MODEL = "contemporary_model"


class LeakProbeKind(StrEnum):
    """Mandatory attacks against memorized or post-cutoff policy knowledge."""

    EVENT_KNOWLEDGE = "event_knowledge"
    TREATMENT_ASSIGNMENT = "treatment_assignment"
    POST_CUTOFF_OUTCOME = "post_cutoff_outcome"
    NAMED_VS_ANONYMOUS_SENSITIVITY = "named_vs_anonymous_sensitivity"


def _strict_fields(
    mapping: Mapping[str, object], expected: frozenset[str], context: str
) -> None:
    if any(not isinstance(key, str) for key in mapping):
        raise LLMKnowledgeFirewallError(f"{context} field names must be strings")
    actual = frozenset(mapping)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        raise LLMKnowledgeFirewallError(
            f"{context} is missing fields: {', '.join(missing)}"
        )
    if unknown:
        raise LLMKnowledgeFirewallError(
            f"{context} has unknown fields: {', '.join(unknown)}"
        )


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise LLMKnowledgeFirewallError(f"{field} must be an object")
    return cast(Mapping[str, object], value)


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise LLMKnowledgeFirewallError(f"{field} must be a stable identifier")
    return value


def _string(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise LLMKnowledgeFirewallError(f"{field} must be a string")
    return value


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise LLMKnowledgeFirewallError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise LLMKnowledgeFirewallError(f"{field} must be a boolean")
    return value


def _aware_utc(value: object, field: str) -> datetime:
    if not isinstance(value, datetime):
        raise LLMKnowledgeFirewallError(f"{field} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise LLMKnowledgeFirewallError(f"{field} must include a timezone")
    try:
        return value.astimezone(UTC)
    except (OverflowError, ValueError) as error:
        raise LLMKnowledgeFirewallError(f"{field} is outside the UTC range") from error


def _parse_datetime(value: object, field: str) -> datetime:
    if not isinstance(value, str):
        raise LLMKnowledgeFirewallError(f"{field} must be an ISO-8601 string")
    try:
        return _aware_utc(datetime.fromisoformat(value), field)
    except ValueError as error:
        raise LLMKnowledgeFirewallError(f"{field} must be valid ISO-8601") from error


def _schema_version(value: object) -> int:
    if value != KNOWLEDGE_FIREWALL_SCHEMA_VERSION:
        raise LLMKnowledgeFirewallError("unsupported knowledge-firewall schema version")
    return KNOWLEDGE_FIREWALL_SCHEMA_VERSION


def _unit_interval(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise LLMKnowledgeFirewallError(f"{field} must be a finite real number")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise LLMKnowledgeFirewallError(f"{field} must be in [0, 1]")
    return normalized


@dataclass(frozen=True, slots=True)
class ModelWeightProvenance:
    """Content-addressed, immutable weight and training provenance."""

    adapter_id: str
    artifact_kind: AdapterArtifactKind
    artifact_sha256: str
    training_manifest_sha256: str
    verification_manifest_sha256: str
    provenance_manifest_sha256: str
    environment_lock_sha256: str
    training_data_cutoff: datetime
    weights_frozen_time: datetime
    weights_available_time: datetime
    network_access_permitted: bool
    retrieval_permitted: bool
    anonymous_structured_inputs_only: bool
    synthetic_only: bool
    schema_version: int = KNOWLEDGE_FIREWALL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema_version(self.schema_version)
        object.__setattr__(
            self, "adapter_id", _identifier(self.adapter_id, "adapter_id")
        )
        if not isinstance(self.artifact_kind, AdapterArtifactKind):
            raise LLMKnowledgeFirewallError(
                "artifact_kind must be an AdapterArtifactKind"
            )
        for field in (
            "artifact_sha256",
            "training_manifest_sha256",
            "verification_manifest_sha256",
            "provenance_manifest_sha256",
            "environment_lock_sha256",
        ):
            object.__setattr__(self, field, _sha256(getattr(self, field), field))
        training_cutoff = _aware_utc(self.training_data_cutoff, "training_data_cutoff")
        frozen_time = _aware_utc(self.weights_frozen_time, "weights_frozen_time")
        available_time = _aware_utc(
            self.weights_available_time, "weights_available_time"
        )
        if training_cutoff > frozen_time:
            raise LLMKnowledgeFirewallError(
                "training_data_cutoff is after weights_frozen_time"
            )
        if frozen_time > available_time:
            raise LLMKnowledgeFirewallError(
                "weights_frozen_time is after weights_available_time"
            )
        object.__setattr__(self, "training_data_cutoff", training_cutoff)
        object.__setattr__(self, "weights_frozen_time", frozen_time)
        object.__setattr__(self, "weights_available_time", available_time)
        for field in (
            "network_access_permitted",
            "retrieval_permitted",
            "anonymous_structured_inputs_only",
            "synthetic_only",
        ):
            _bool(getattr(self, field), field)

    @property
    def sha256(self) -> str:
        """Return a content commitment to the complete provenance record."""
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def to_mapping(self) -> dict[str, object]:
        """Return the exact machine-readable provenance schema."""
        return {
            "adapter_id": self.adapter_id,
            "anonymous_structured_inputs_only": self.anonymous_structured_inputs_only,
            "artifact_kind": self.artifact_kind.value,
            "artifact_sha256": self.artifact_sha256,
            "environment_lock_sha256": self.environment_lock_sha256,
            "network_access_permitted": self.network_access_permitted,
            "provenance_manifest_sha256": self.provenance_manifest_sha256,
            "retrieval_permitted": self.retrieval_permitted,
            "schema_version": self.schema_version,
            "synthetic_only": self.synthetic_only,
            "training_data_cutoff": self.training_data_cutoff.isoformat(),
            "training_manifest_sha256": self.training_manifest_sha256,
            "verification_manifest_sha256": self.verification_manifest_sha256,
            "weights_available_time": self.weights_available_time.isoformat(),
            "weights_frozen_time": self.weights_frozen_time.isoformat(),
        }

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        """Parse provenance with exact-field and enum validation."""
        mapping = _mapping(value, "provenance")
        _strict_fields(mapping, _PROVENANCE_FIELDS, "provenance")
        try:
            artifact_kind = AdapterArtifactKind(
                _string(mapping["artifact_kind"], "artifact_kind")
            )
        except (TypeError, ValueError) as error:
            raise LLMKnowledgeFirewallError("unsupported artifact_kind") from error
        return cls(
            adapter_id=_identifier(mapping["adapter_id"], "adapter_id"),
            artifact_kind=artifact_kind,
            artifact_sha256=_sha256(mapping["artifact_sha256"], "artifact_sha256"),
            training_manifest_sha256=_sha256(
                mapping["training_manifest_sha256"], "training_manifest_sha256"
            ),
            verification_manifest_sha256=_sha256(
                mapping["verification_manifest_sha256"],
                "verification_manifest_sha256",
            ),
            provenance_manifest_sha256=_sha256(
                mapping["provenance_manifest_sha256"],
                "provenance_manifest_sha256",
            ),
            environment_lock_sha256=_sha256(
                mapping["environment_lock_sha256"], "environment_lock_sha256"
            ),
            training_data_cutoff=_parse_datetime(
                mapping["training_data_cutoff"], "training_data_cutoff"
            ),
            weights_frozen_time=_parse_datetime(
                mapping["weights_frozen_time"], "weights_frozen_time"
            ),
            weights_available_time=_parse_datetime(
                mapping["weights_available_time"], "weights_available_time"
            ),
            network_access_permitted=_bool(
                mapping["network_access_permitted"], "network_access_permitted"
            ),
            retrieval_permitted=_bool(
                mapping["retrieval_permitted"], "retrieval_permitted"
            ),
            anonymous_structured_inputs_only=_bool(
                mapping["anonymous_structured_inputs_only"],
                "anonymous_structured_inputs_only",
            ),
            synthetic_only=_bool(mapping["synthetic_only"], "synthetic_only"),
            schema_version=_schema_version(mapping["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class LeakProbeDefinition:
    """One committed attack input and its predeclared grading rule."""

    probe_id: str
    kind: LeakProbeKind
    input_sha256: str
    grading_rule_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "probe_id", _identifier(self.probe_id, "probe_id"))
        if not isinstance(self.kind, LeakProbeKind):
            raise LLMKnowledgeFirewallError("kind must be a LeakProbeKind")
        object.__setattr__(
            self, "input_sha256", _sha256(self.input_sha256, "input_sha256")
        )
        object.__setattr__(
            self,
            "grading_rule_sha256",
            _sha256(self.grading_rule_sha256, "grading_rule_sha256"),
        )

    def to_mapping(self) -> dict[str, object]:
        """Return the exact probe-definition schema."""
        return {
            "grading_rule_sha256": self.grading_rule_sha256,
            "input_sha256": self.input_sha256,
            "kind": self.kind.value,
            "probe_id": self.probe_id,
        }

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        mapping = _mapping(value, "probe definition")
        _strict_fields(mapping, _PROBE_DEFINITION_FIELDS, "probe definition")
        try:
            kind = LeakProbeKind(_string(mapping["kind"], "kind"))
        except (TypeError, ValueError) as error:
            raise LLMKnowledgeFirewallError("unsupported leak-probe kind") from error
        return cls(
            probe_id=_identifier(mapping["probe_id"], "probe_id"),
            kind=kind,
            input_sha256=_sha256(mapping["input_sha256"], "input_sha256"),
            grading_rule_sha256=_sha256(
                mapping["grading_rule_sha256"], "grading_rule_sha256"
            ),
        )


@dataclass(frozen=True, slots=True)
class LeakProbeManifest:
    """Frozen complete set of attacks required for one provenance record."""

    provenance_sha256: str
    trusted_cutoff: datetime
    max_named_anonymous_sensitivity: float
    definitions: tuple[LeakProbeDefinition, ...]
    schema_version: int = KNOWLEDGE_FIREWALL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema_version(self.schema_version)
        object.__setattr__(
            self,
            "provenance_sha256",
            _sha256(self.provenance_sha256, "provenance_sha256"),
        )
        object.__setattr__(
            self, "trusted_cutoff", _aware_utc(self.trusted_cutoff, "trusted_cutoff")
        )
        object.__setattr__(
            self,
            "max_named_anonymous_sensitivity",
            _unit_interval(
                self.max_named_anonymous_sensitivity,
                "max_named_anonymous_sensitivity",
            ),
        )
        if not isinstance(self.definitions, tuple) or not all(
            isinstance(item, LeakProbeDefinition) for item in self.definitions
        ):
            raise LLMKnowledgeFirewallError(
                "definitions must be an immutable LeakProbeDefinition tuple"
            )
        kinds = tuple(item.kind for item in self.definitions)
        if set(kinds) != set(LeakProbeKind) or len(kinds) != len(LeakProbeKind):
            raise LLMKnowledgeFirewallError(
                "manifest must contain exactly one definition for every leak probe"
            )
        probe_ids = tuple(item.probe_id for item in self.definitions)
        if len(set(probe_ids)) != len(probe_ids):
            raise LLMKnowledgeFirewallError("leak-probe IDs must be unique")
        object.__setattr__(
            self,
            "definitions",
            tuple(sorted(self.definitions, key=lambda item: item.kind.value)),
        )

    @property
    def sha256(self) -> str:
        """Return the frozen leak-probe manifest commitment."""
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def to_mapping(self) -> dict[str, object]:
        """Return the exact manifest schema without any probe text."""
        return {
            "definitions": [item.to_mapping() for item in self.definitions],
            "max_named_anonymous_sensitivity": self.max_named_anonymous_sensitivity,
            "provenance_sha256": self.provenance_sha256,
            "schema_version": self.schema_version,
            "trusted_cutoff": self.trusted_cutoff.isoformat(),
        }

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        mapping = _mapping(value, "leak-probe manifest")
        _strict_fields(mapping, _PROBE_MANIFEST_FIELDS, "leak-probe manifest")
        definitions = mapping["definitions"]
        if not isinstance(definitions, list):
            raise LLMKnowledgeFirewallError("definitions must be a JSON list")
        return cls(
            provenance_sha256=_sha256(
                mapping["provenance_sha256"], "provenance_sha256"
            ),
            trusted_cutoff=_parse_datetime(mapping["trusted_cutoff"], "trusted_cutoff"),
            max_named_anonymous_sensitivity=_unit_interval(
                mapping["max_named_anonymous_sensitivity"],
                "max_named_anonymous_sensitivity",
            ),
            definitions=tuple(
                LeakProbeDefinition.from_mapping(item) for item in definitions
            ),
            schema_version=_schema_version(mapping["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class LeakProbeResult:
    """Text-free result for one predeclared knowledge-leak attack."""

    probe_id: str
    kind: LeakProbeKind
    input_sha256: str
    output_sha256: str
    forbidden_revelation_detected: bool
    sensitivity_score: float | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "probe_id", _identifier(self.probe_id, "probe_id"))
        if not isinstance(self.kind, LeakProbeKind):
            raise LLMKnowledgeFirewallError("kind must be a LeakProbeKind")
        object.__setattr__(
            self, "input_sha256", _sha256(self.input_sha256, "input_sha256")
        )
        object.__setattr__(
            self, "output_sha256", _sha256(self.output_sha256, "output_sha256")
        )
        _bool(
            self.forbidden_revelation_detected,
            "forbidden_revelation_detected",
        )
        if self.kind is LeakProbeKind.NAMED_VS_ANONYMOUS_SENSITIVITY:
            if self.sensitivity_score is None:
                raise LLMKnowledgeFirewallError(
                    "named-vs-anonymous probe requires sensitivity_score"
                )
            object.__setattr__(
                self,
                "sensitivity_score",
                _unit_interval(self.sensitivity_score, "sensitivity_score"),
            )
        elif self.sensitivity_score is not None:
            raise LLMKnowledgeFirewallError(
                "sensitivity_score is only valid for named-vs-anonymous probe"
            )

    def to_mapping(self) -> dict[str, object]:
        """Return a result containing commitments, not prompts or completions."""
        return {
            "forbidden_revelation_detected": self.forbidden_revelation_detected,
            "input_sha256": self.input_sha256,
            "kind": self.kind.value,
            "output_sha256": self.output_sha256,
            "probe_id": self.probe_id,
            "sensitivity_score": self.sensitivity_score,
        }

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        mapping = _mapping(value, "leak-probe result")
        _strict_fields(mapping, _PROBE_RESULT_FIELDS, "leak-probe result")
        try:
            kind = LeakProbeKind(_string(mapping["kind"], "kind"))
        except (TypeError, ValueError) as error:
            raise LLMKnowledgeFirewallError("unsupported leak-probe kind") from error
        raw_score = mapping["sensitivity_score"]
        score = (
            None
            if raw_score is None
            else _unit_interval(raw_score, "sensitivity_score")
        )
        return cls(
            probe_id=_identifier(mapping["probe_id"], "probe_id"),
            kind=kind,
            input_sha256=_sha256(mapping["input_sha256"], "input_sha256"),
            output_sha256=_sha256(mapping["output_sha256"], "output_sha256"),
            forbidden_revelation_detected=_bool(
                mapping["forbidden_revelation_detected"],
                "forbidden_revelation_detected",
            ),
            sensitivity_score=score,
        )


@dataclass(frozen=True, slots=True)
class LeakProbeReport:
    """Committed complete attack results for one manifest and weight artifact."""

    manifest_sha256: str
    provenance_sha256: str
    offline_enforced: bool
    retrieval_disabled: bool
    results: tuple[LeakProbeResult, ...]
    schema_version: int = KNOWLEDGE_FIREWALL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema_version(self.schema_version)
        object.__setattr__(
            self, "manifest_sha256", _sha256(self.manifest_sha256, "manifest_sha256")
        )
        object.__setattr__(
            self,
            "provenance_sha256",
            _sha256(self.provenance_sha256, "provenance_sha256"),
        )
        _bool(self.offline_enforced, "offline_enforced")
        _bool(self.retrieval_disabled, "retrieval_disabled")
        if not isinstance(self.results, tuple) or not all(
            isinstance(item, LeakProbeResult) for item in self.results
        ):
            raise LLMKnowledgeFirewallError(
                "results must be an immutable LeakProbeResult tuple"
            )
        kinds = tuple(item.kind for item in self.results)
        if set(kinds) != set(LeakProbeKind) or len(kinds) != len(LeakProbeKind):
            raise LLMKnowledgeFirewallError(
                "report must contain exactly one result for every leak probe"
            )
        probe_ids = tuple(item.probe_id for item in self.results)
        if len(set(probe_ids)) != len(probe_ids):
            raise LLMKnowledgeFirewallError("leak-probe result IDs must be unique")
        object.__setattr__(
            self,
            "results",
            tuple(sorted(self.results, key=lambda item: item.kind.value)),
        )

    @property
    def sha256(self) -> str:
        """Return a content commitment to every probe result."""
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def assert_passed(self, manifest: LeakProbeManifest) -> None:
        """Fail closed unless the complete frozen manifest passed offline."""
        if self.manifest_sha256 != manifest.sha256:
            raise LLMKnowledgeFirewallError("leak-probe manifest commitment mismatches")
        if self.provenance_sha256 != manifest.provenance_sha256:
            raise LLMKnowledgeFirewallError(
                "leak-probe provenance commitment mismatches"
            )
        if not self.offline_enforced or not self.retrieval_disabled:
            raise LLMKnowledgeFirewallError(
                "leak probes must run with network and retrieval disabled"
            )
        definitions = {item.probe_id: item for item in manifest.definitions}
        for result in self.results:
            definition = definitions.get(result.probe_id)
            if definition is None or definition.kind is not result.kind:
                raise LLMKnowledgeFirewallError(
                    "leak-probe result does not match its manifest definition"
                )
            if definition.input_sha256 != result.input_sha256:
                raise LLMKnowledgeFirewallError(
                    "leak-probe input commitment mismatches"
                )
            if result.forbidden_revelation_detected:
                raise LLMKnowledgeFirewallError(
                    f"forbidden revelation detected by {result.kind.value} probe"
                )
            if result.kind is LeakProbeKind.NAMED_VS_ANONYMOUS_SENSITIVITY:
                sensitivity = result.sensitivity_score
                if (
                    sensitivity is None
                    or sensitivity > manifest.max_named_anonymous_sensitivity
                ):
                    raise LLMKnowledgeFirewallError(
                        "named-vs-anonymous sensitivity exceeds the frozen threshold"
                    )

    def to_mapping(self) -> dict[str, object]:
        """Return the exact report schema."""
        return {
            "manifest_sha256": self.manifest_sha256,
            "offline_enforced": self.offline_enforced,
            "provenance_sha256": self.provenance_sha256,
            "results": [item.to_mapping() for item in self.results],
            "retrieval_disabled": self.retrieval_disabled,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        mapping = _mapping(value, "leak-probe report")
        _strict_fields(mapping, _PROBE_REPORT_FIELDS, "leak-probe report")
        results = mapping["results"]
        if not isinstance(results, list):
            raise LLMKnowledgeFirewallError("results must be a JSON list")
        return cls(
            manifest_sha256=_sha256(mapping["manifest_sha256"], "manifest_sha256"),
            provenance_sha256=_sha256(
                mapping["provenance_sha256"], "provenance_sha256"
            ),
            offline_enforced=_bool(mapping["offline_enforced"], "offline_enforced"),
            retrieval_disabled=_bool(
                mapping["retrieval_disabled"], "retrieval_disabled"
            ),
            results=tuple(LeakProbeResult.from_mapping(item) for item in results),
            schema_version=_schema_version(mapping["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class AdapterAuthorization:
    """Fail-closed authorization binding provenance, use, cutoff, and probes."""

    provenance: ModelWeightProvenance
    requested_use: RequestedUse
    trusted_cutoff: datetime
    cutoff_policy_sha256: str
    deidentification_policy_sha256: str
    prompt_template_sha256: str
    sanitized_context_record_sha256: str
    leak_probe_manifest: LeakProbeManifest
    leak_probe_report: LeakProbeReport
    schema_version: int = KNOWLEDGE_FIREWALL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema_version(self.schema_version)
        if not isinstance(self.provenance, ModelWeightProvenance):
            raise LLMKnowledgeFirewallError("provenance must be ModelWeightProvenance")
        if not isinstance(self.requested_use, RequestedUse):
            raise LLMKnowledgeFirewallError("requested_use must be RequestedUse")
        cutoff = _aware_utc(self.trusted_cutoff, "trusted_cutoff")
        object.__setattr__(self, "trusted_cutoff", cutoff)
        for field in (
            "cutoff_policy_sha256",
            "deidentification_policy_sha256",
            "prompt_template_sha256",
            "sanitized_context_record_sha256",
        ):
            object.__setattr__(self, field, _sha256(getattr(self, field), field))
        if not isinstance(self.leak_probe_manifest, LeakProbeManifest):
            raise LLMKnowledgeFirewallError(
                "leak_probe_manifest must be LeakProbeManifest"
            )
        if not isinstance(self.leak_probe_report, LeakProbeReport):
            raise LLMKnowledgeFirewallError("leak_probe_report must be LeakProbeReport")
        if self.leak_probe_manifest.provenance_sha256 != self.provenance.sha256:
            raise LLMKnowledgeFirewallError(
                "manifest does not bind the authorized weight provenance"
            )
        if self.leak_probe_manifest.trusted_cutoff != cutoff:
            raise LLMKnowledgeFirewallError(
                "manifest trusted_cutoff does not match authorization"
            )
        self.leak_probe_report.assert_passed(self.leak_probe_manifest)
        self._assert_eligibility()

    def _assert_eligibility(self) -> None:
        provenance = self.provenance
        if provenance.network_access_permitted or provenance.retrieval_permitted:
            raise LLMKnowledgeFirewallError(
                "authorized adapters must disable network access and retrieval"
            )
        if not provenance.anonymous_structured_inputs_only:
            raise LLMKnowledgeFirewallError(
                "authorized adapters must accept anonymous structured inputs only"
            )
        kind = provenance.artifact_kind
        use = self.requested_use
        if kind is AdapterArtifactKind.DETERMINISTIC_MOCK:
            if use is not RequestedUse.SYNTHETIC_TEST or not provenance.synthetic_only:
                raise LLMKnowledgeFirewallError(
                    "deterministic mocks are eligible only for synthetic tests"
                )
            return
        if provenance.synthetic_only:
            raise LLMKnowledgeFirewallError(
                "non-mock model provenance cannot be marked synthetic-only"
            )
        if use is RequestedUse.SYNTHETIC_TEST:
            raise LLMKnowledgeFirewallError(
                "synthetic-test use requires deterministic mock provenance"
            )
        if kind is AdapterArtifactKind.CONTEMPORARY_MODEL:
            if use is not RequestedUse.ABLATION_ONLY:
                raise LLMKnowledgeFirewallError(
                    "contemporary or post-cutoff models are ablation-only"
                )
            return
        if (
            provenance.training_data_cutoff > self.trusted_cutoff
            or provenance.weights_frozen_time > self.trusted_cutoff
            or provenance.weights_available_time > self.trusted_cutoff
        ):
            raise LLMKnowledgeFirewallError(
                "pre-cutoff provenance contains a post-cutoff timestamp"
            )
        if use is RequestedUse.MAIN_POLICY and kind not in {
            AdapterArtifactKind.PRE_CUTOFF_MODEL,
            AdapterArtifactKind.DISTILLED_PRE_CUTOFF_POLICY,
        }:
            raise LLMKnowledgeFirewallError(
                "main policy use requires verified pre-cutoff weights or distillation"
            )
        if use is RequestedUse.MAIN_POLICY:
            raise LLMKnowledgeFirewallError(
                "main policy authorization is not implemented without an "
                "independently verified weight-loader and temporal-lineage capability"
            )

    @property
    def sha256(self) -> str:
        """Return the commitment stored in requests, plans, and audit events."""
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    @property
    def eligible_for_main_policy(self) -> bool:
        """Return false until a non-declarative main-path authority exists."""
        return False

    def validate_request_commitments(
        self,
        *,
        authorization_sha256: str,
        requested_use: RequestedUse,
        as_of_time: datetime,
        trusted_cutoff: datetime,
        cutoff_policy_sha256: str,
        deidentification_policy_sha256: str,
        prompt_template_sha256: str,
        sanitized_context_record_sha256: str,
    ) -> None:
        """Bind an individual request to this exact frozen authorization."""
        if _sha256(authorization_sha256, "authorization_sha256") != self.sha256:
            raise LLMKnowledgeFirewallError(
                "request authorization commitment does not match gateway"
            )
        if requested_use is not self.requested_use:
            raise LLMKnowledgeFirewallError("request use does not match authorization")
        normalized_as_of = _aware_utc(as_of_time, "as_of_time")
        if _aware_utc(trusted_cutoff, "trusted_cutoff") != self.trusted_cutoff:
            raise LLMKnowledgeFirewallError(
                "request cutoff does not match authorization"
            )
        if cutoff_policy_sha256 != self.cutoff_policy_sha256:
            raise LLMKnowledgeFirewallError(
                "request cutoff policy does not match authorization"
            )
        if deidentification_policy_sha256 != self.deidentification_policy_sha256:
            raise LLMKnowledgeFirewallError(
                "request deidentification policy does not match authorization"
            )
        if prompt_template_sha256 != self.prompt_template_sha256:
            raise LLMKnowledgeFirewallError(
                "request prompt template does not match authorization"
            )
        if sanitized_context_record_sha256 != self.sanitized_context_record_sha256:
            raise LLMKnowledgeFirewallError(
                "request context record does not match authorization"
            )
        if requested_use is RequestedUse.MAIN_POLICY and (
            self.provenance.training_data_cutoff > normalized_as_of
            or self.provenance.weights_frozen_time > normalized_as_of
            or self.provenance.weights_available_time > normalized_as_of
        ):
            raise LLMKnowledgeFirewallError(
                "main-path weights were not frozen and available at request time"
            )

    def to_mapping(self) -> dict[str, object]:
        """Return the complete authorization and embedded evidence."""
        return {
            "cutoff_policy_sha256": self.cutoff_policy_sha256,
            "deidentification_policy_sha256": self.deidentification_policy_sha256,
            "leak_probe_manifest": self.leak_probe_manifest.to_mapping(),
            "leak_probe_report": self.leak_probe_report.to_mapping(),
            "prompt_template_sha256": self.prompt_template_sha256,
            "provenance": self.provenance.to_mapping(),
            "requested_use": self.requested_use.value,
            "sanitized_context_record_sha256": (self.sanitized_context_record_sha256),
            "schema_version": self.schema_version,
            "trusted_cutoff": self.trusted_cutoff.isoformat(),
        }

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        """Revalidate eligibility when loading a serialized authorization."""
        mapping = _mapping(value, "adapter authorization")
        _strict_fields(mapping, _AUTHORIZATION_FIELDS, "adapter authorization")
        try:
            requested_use = RequestedUse(
                _string(mapping["requested_use"], "requested_use")
            )
        except (TypeError, ValueError) as error:
            raise LLMKnowledgeFirewallError("unsupported requested_use") from error
        return cls(
            provenance=ModelWeightProvenance.from_mapping(mapping["provenance"]),
            requested_use=requested_use,
            trusted_cutoff=_parse_datetime(mapping["trusted_cutoff"], "trusted_cutoff"),
            cutoff_policy_sha256=_sha256(
                mapping["cutoff_policy_sha256"], "cutoff_policy_sha256"
            ),
            deidentification_policy_sha256=_sha256(
                mapping["deidentification_policy_sha256"],
                "deidentification_policy_sha256",
            ),
            prompt_template_sha256=_sha256(
                mapping["prompt_template_sha256"], "prompt_template_sha256"
            ),
            sanitized_context_record_sha256=_sha256(
                mapping["sanitized_context_record_sha256"],
                "sanitized_context_record_sha256",
            ),
            leak_probe_manifest=LeakProbeManifest.from_mapping(
                mapping["leak_probe_manifest"]
            ),
            leak_probe_report=LeakProbeReport.from_mapping(
                mapping["leak_probe_report"]
            ),
            schema_version=_schema_version(mapping["schema_version"]),
        )


__all__ = [
    "KNOWLEDGE_FIREWALL_SCHEMA_VERSION",
    "AdapterArtifactKind",
    "AdapterAuthorization",
    "LLMKnowledgeFirewallError",
    "LeakProbeDefinition",
    "LeakProbeKind",
    "LeakProbeManifest",
    "LeakProbeReport",
    "LeakProbeResult",
    "ModelWeightProvenance",
]
