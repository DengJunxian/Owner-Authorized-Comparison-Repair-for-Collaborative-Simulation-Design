"""Strict anonymous contracts for optional slow-planning adapters."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Self, cast

from policytwin.cognition.planning import PlanAction, RequestedUse
from policytwin.counterfactual import EVENT_KEY_SCHEMA_VERSION, EventKey
from policytwin.provenance import canonical_json_bytes, sha256_bytes
from policytwin.social.privacy import (
    DeidentificationError,
    DeidentificationReceipt,
    RedactionCounts,
    SanitizedText,
)

LLM_REQUEST_SCHEMA_VERSION = 3
LLM_RESPONSE_SCHEMA_VERSION = 1

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_RATIONALE_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_EVENT_KEY_FIELDS = frozenset(
    {
        "experiment_id",
        "purpose",
        "replicate_id",
        "schema_version",
        "seed",
        "simulation_step",
    }
)
_FEATURE_FIELDS = frozenset(
    {"available_time", "code", "event_time", "lineage_sha256", "value"}
)
_RECEIPT_FIELDS = frozenset(
    {"counts", "input_sha256", "output_sha256", "policy_sha256"}
)
_COUNT_FIELDS = frozenset({"entity_count", "policy_count", "year_count"})
_SANITIZED_FIELDS = frozenset({"receipt", "text"})
_CONTEXT_RECORD_FIELDS = frozenset(
    {
        "available_time",
        "content_sha256",
        "event_time",
        "ingestion_time",
        "lineage_sha256",
        "record_id",
        "source",
        "version",
    }
)
_REQUEST_FIELDS = frozenset(
    {
        "adapter_authorization_sha256",
        "agent_commitment",
        "allowed_actions",
        "as_of_time",
        "cutoff_policy_sha256",
        "deidentification_policy_sha256",
        "event_key",
        "input_lineage_sha256",
        "output_schema_version",
        "prompt_template_sha256",
        "public_features",
        "requested_use",
        "sanitized_context",
        "sanitized_context_record",
        "schema_version",
        "topic_id",
        "trusted_cutoff",
    }
)
_RESPONSE_FIELDS = frozenset(
    {
        "action",
        "confidence",
        "input_sha256",
        "rationale_code",
        "review_step",
        "schema_version",
        "valid_until",
    }
)


class LLMContractError(ValueError):
    """Raised when an LLM request or response violates the closed schema."""


class LLMTemporalError(LLMContractError):
    """Raised when a request or response crosses the knowledge cutoff."""


class LLMPrivacyError(LLMContractError):
    """Raised when an adapter input is not proven anonymous."""


class LLMOutputError(LLMContractError):
    """Raised when an adapter emits an illegal or unauthorized output."""


class PublicFeatureCode(StrEnum):
    """Closed, bounded public features available to slow planning."""

    BELIEF = "belief"
    CONFIDENCE = "confidence"
    MEMORY_STRENGTH = "memory_strength"
    ATTENTION_LOAD = "attention_load"
    CASCADE_REACH = "cascade_reach"


@dataclass(frozen=True, slots=True)
class SanitizedContextRecord:
    """Temporal and lineage commitment for one sanitized context payload.

    A deidentification receipt proves only that the emitted text matches the
    anonymizer output.  This companion record proves which anonymous source
    record supplied that payload and whether it was actually available and
    ingested at the request boundary.
    """

    record_id: str
    source: str
    version: str
    event_time: datetime
    available_time: datetime
    ingestion_time: datetime
    content_sha256: str
    lineage_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "record_id", _identifier(self.record_id, "record_id"))
        object.__setattr__(self, "source", _identifier(self.source, "source"))
        object.__setattr__(self, "version", _identifier(self.version, "version"))
        event_time = _aware_utc(self.event_time, "event_time")
        available_time = _aware_utc(self.available_time, "available_time")
        ingestion_time = _aware_utc(self.ingestion_time, "ingestion_time")
        if event_time > available_time:
            raise LLMTemporalError("context event_time is after available_time")
        if available_time > ingestion_time:
            raise LLMTemporalError("context available_time is after ingestion_time")
        object.__setattr__(self, "event_time", event_time)
        object.__setattr__(self, "available_time", available_time)
        object.__setattr__(self, "ingestion_time", ingestion_time)
        object.__setattr__(
            self, "content_sha256", _sha256(self.content_sha256, "content_sha256")
        )
        object.__setattr__(
            self, "lineage_sha256", _sha256(self.lineage_sha256, "lineage_sha256")
        )

    @property
    def sha256(self) -> str:
        """Commit to the complete temporal provenance record."""
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def validate_for(
        self,
        context: SanitizedText,
        *,
        as_of_time: datetime,
        cutoff: datetime,
        lineage_sha256: str,
    ) -> None:
        """Reject a content, lineage, or temporal substitution."""
        normalized_as_of = _aware_utc(as_of_time, "as_of_time")
        normalized_cutoff = _aware_utc(cutoff, "cutoff")
        if not isinstance(context, SanitizedText):
            raise LLMPrivacyError("context record requires SanitizedText")
        if self.content_sha256 != context.receipt.output_sha256:
            raise LLMPrivacyError("context record does not bind the sanitized text")
        if self.lineage_sha256 != _sha256(lineage_sha256, "lineage_sha256"):
            raise LLMTemporalError("context record lineage does not match request")
        if self.event_time > normalized_as_of:
            raise LLMTemporalError("context event_time is after as_of_time")
        if self.available_time > normalized_as_of:
            raise LLMTemporalError("context was unavailable at as_of_time")
        if self.available_time > normalized_cutoff:
            raise LLMTemporalError("context was unavailable before cutoff")
        if self.ingestion_time > normalized_as_of:
            raise LLMTemporalError("context was not ingested at as_of_time")
        if self.ingestion_time > normalized_cutoff:
            raise LLMTemporalError("context was not ingested before cutoff")

    def to_mapping(self) -> dict[str, object]:
        """Return the exact anonymous temporal-provenance schema."""
        return {
            "available_time": self.available_time.isoformat(),
            "content_sha256": self.content_sha256,
            "event_time": self.event_time.isoformat(),
            "ingestion_time": self.ingestion_time.isoformat(),
            "lineage_sha256": self.lineage_sha256,
            "record_id": self.record_id,
            "source": self.source,
            "version": self.version,
        }

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        """Strictly parse a context provenance record."""
        mapping = _mapping(value, "sanitized_context_record")
        _strict_fields(mapping, _CONTEXT_RECORD_FIELDS, "sanitized_context_record")
        return cls(
            record_id=_identifier(mapping["record_id"], "record_id"),
            source=_identifier(mapping["source"], "source"),
            version=_identifier(mapping["version"], "version"),
            event_time=_parse_datetime(mapping["event_time"], "event_time"),
            available_time=_parse_datetime(mapping["available_time"], "available_time"),
            ingestion_time=_parse_datetime(mapping["ingestion_time"], "ingestion_time"),
            content_sha256=_sha256(mapping["content_sha256"], "content_sha256"),
            lineage_sha256=_sha256(mapping["lineage_sha256"], "lineage_sha256"),
        )


def _strict_fields(
    mapping: Mapping[str, object], expected: frozenset[str], context: str
) -> None:
    if any(not isinstance(key, str) for key in mapping):
        raise LLMContractError(f"{context} field names must be strings")
    actual = frozenset(mapping)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        raise LLMContractError(f"{context} is missing fields: {', '.join(missing)}")
    if unknown:
        raise LLMContractError(f"{context} has unknown fields: {', '.join(unknown)}")


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise LLMContractError(f"{field} must be an object with string fields")
    return cast(Mapping[str, object], value)


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise LLMContractError(f"{field} must be a stable anonymous identifier")
    return value


def _string(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise LLMContractError(f"{field} must be a string")
    return value


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise LLMContractError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _aware_utc(value: object, field: str) -> datetime:
    if not isinstance(value, datetime):
        raise LLMContractError(f"{field} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise LLMContractError(f"{field} must include an explicit timezone")
    try:
        return value.astimezone(UTC)
    except (OverflowError, ValueError) as error:
        raise LLMContractError(f"{field} is outside the UTC range") from error


def _parse_datetime(value: object, field: str) -> datetime:
    if not isinstance(value, str):
        raise LLMContractError(f"{field} must be an ISO-8601 string")
    try:
        return _aware_utc(datetime.fromisoformat(value), field)
    except ValueError as error:
        raise LLMContractError(f"{field} must be valid ISO-8601") from error


def _non_negative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LLMContractError(f"{field} must be a non-negative integer")
    return value


def _unit_interval(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise LLMContractError(f"{field} must be a finite real number")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise LLMContractError(f"{field} must be finite and in [0, 1]")
    return normalized


def _bounded_feature(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise LLMContractError("feature value must be a finite real number")
    normalized = float(value)
    if not math.isfinite(normalized) or not -1.0 <= normalized <= 1.0:
        raise LLMContractError("feature value must be finite and in [-1, 1]")
    return normalized


def _parse_event_key(value: object) -> EventKey:
    mapping = _mapping(value, "event_key")
    _strict_fields(mapping, _EVENT_KEY_FIELDS, "event_key")
    for field in ("experiment_id", "seed", "purpose"):
        _identifier(mapping[field], f"event_key.{field}")
    if mapping["schema_version"] != EVENT_KEY_SCHEMA_VERSION:
        raise LLMContractError("event_key schema_version is unsupported")
    return EventKey(
        experiment_id=cast(str, mapping["experiment_id"]),
        seed=cast(str, mapping["seed"]),
        replicate_id=_non_negative_int(
            mapping["replicate_id"], "event_key.replicate_id"
        ),
        simulation_step=_non_negative_int(
            mapping["simulation_step"], "event_key.simulation_step"
        ),
        purpose=cast(str, mapping["purpose"]),
        schema_version=EVENT_KEY_SCHEMA_VERSION,
    )


def _event_key_mapping(event_key: EventKey) -> dict[str, object]:
    return {
        "experiment_id": event_key.experiment_id,
        "purpose": event_key.purpose,
        "replicate_id": event_key.replicate_id,
        "schema_version": event_key.schema_version,
        "seed": event_key.seed,
        "simulation_step": event_key.simulation_step,
    }


def _parse_sanitized_text(value: object) -> SanitizedText:
    mapping = _mapping(value, "sanitized_context")
    _strict_fields(mapping, _SANITIZED_FIELDS, "sanitized_context")
    if not isinstance(mapping["text"], str):
        raise LLMPrivacyError("sanitized_context.text must be a string")
    receipt_mapping = _mapping(mapping["receipt"], "sanitized_context.receipt")
    _strict_fields(receipt_mapping, _RECEIPT_FIELDS, "sanitized_context.receipt")
    counts_mapping = _mapping(
        receipt_mapping["counts"], "sanitized_context.receipt.counts"
    )
    _strict_fields(counts_mapping, _COUNT_FIELDS, "sanitized_context.receipt.counts")
    counts = RedactionCounts(
        entity_count=_non_negative_int(counts_mapping["entity_count"], "entity_count"),
        policy_count=_non_negative_int(counts_mapping["policy_count"], "policy_count"),
        year_count=_non_negative_int(counts_mapping["year_count"], "year_count"),
    )
    receipt = DeidentificationReceipt(
        input_sha256=_sha256(receipt_mapping["input_sha256"], "input_sha256"),
        output_sha256=_sha256(receipt_mapping["output_sha256"], "output_sha256"),
        policy_sha256=_sha256(receipt_mapping["policy_sha256"], "policy_sha256"),
        counts=counts,
    )
    return SanitizedText(text=mapping["text"], receipt=receipt)


def _parse_actions(value: object) -> tuple[PlanAction, ...]:
    if not isinstance(value, list) or not value:
        raise LLMContractError("allowed_actions must be a non-empty JSON list")
    actions: list[PlanAction] = []
    for raw_action in value:
        if not isinstance(raw_action, str):
            raise LLMContractError("allowed_actions must contain strings")
        try:
            actions.append(PlanAction(raw_action))
        except ValueError as error:
            raise LLMContractError(
                f"unsupported allowed action: {raw_action}"
            ) from error
    return tuple(actions)


@dataclass(frozen=True, slots=True)
class PublicCognitiveFeature:
    """One bounded public feature with explicit availability lineage."""

    code: PublicFeatureCode
    value: float
    event_time: datetime
    available_time: datetime
    lineage_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.code, PublicFeatureCode):
            raise LLMContractError("feature code must be a PublicFeatureCode")
        object.__setattr__(self, "value", _bounded_feature(self.value))
        object.__setattr__(
            self, "event_time", _aware_utc(self.event_time, "event_time")
        )
        object.__setattr__(
            self,
            "available_time",
            _aware_utc(self.available_time, "available_time"),
        )
        object.__setattr__(
            self,
            "lineage_sha256",
            _sha256(self.lineage_sha256, "lineage_sha256"),
        )

    def validate_available(self, *, as_of_time: datetime, cutoff: datetime) -> None:
        """Reject a feature unavailable at the request boundary."""
        normalized_as_of = _aware_utc(as_of_time, "as_of_time")
        normalized_cutoff = _aware_utc(cutoff, "cutoff")
        if self.event_time > normalized_as_of:
            raise LLMTemporalError("feature event_time is after as_of_time")
        if self.available_time > normalized_as_of:
            raise LLMTemporalError("feature was unavailable at as_of_time")
        if self.available_time > normalized_cutoff:
            raise LLMTemporalError("feature was unavailable before cutoff")

    def to_mapping(self) -> dict[str, object]:
        """Return the exact public feature schema."""
        return {
            "available_time": self.available_time.isoformat(),
            "code": self.code.value,
            "event_time": self.event_time.isoformat(),
            "lineage_sha256": self.lineage_sha256,
            "value": self.value,
        }

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        """Strictly parse one public feature."""
        mapping = _mapping(value, "public feature")
        _strict_fields(mapping, _FEATURE_FIELDS, "public feature")
        try:
            code = PublicFeatureCode(_string(mapping["code"], "code"))
        except (TypeError, ValueError) as error:
            raise LLMContractError("public feature code is unsupported") from error
        return cls(
            code=code,
            value=_bounded_feature(mapping["value"]),
            event_time=_parse_datetime(mapping["event_time"], "event_time"),
            available_time=_parse_datetime(mapping["available_time"], "available_time"),
            lineage_sha256=_sha256(mapping["lineage_sha256"], "lineage_sha256"),
        )


@dataclass(frozen=True, slots=True)
class StrictLLMRequest:
    """Anonymous, pre-cutoff, allowlisted slow-planning request."""

    event_key: EventKey
    adapter_authorization_sha256: str
    requested_use: RequestedUse
    agent_commitment: str
    topic_id: str
    as_of_time: datetime
    trusted_cutoff: datetime
    cutoff_policy_sha256: str
    input_lineage_sha256: str
    prompt_template_sha256: str
    deidentification_policy_sha256: str
    sanitized_context: SanitizedText
    sanitized_context_record: SanitizedContextRecord
    public_features: tuple[PublicCognitiveFeature, ...]
    allowed_actions: tuple[PlanAction, ...]
    output_schema_version: int = LLM_RESPONSE_SCHEMA_VERSION
    schema_version: int = LLM_REQUEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != LLM_REQUEST_SCHEMA_VERSION:
            raise LLMContractError("unsupported LLM request schema version")
        if self.output_schema_version != LLM_RESPONSE_SCHEMA_VERSION:
            raise LLMContractError("unsupported requested output schema version")
        if not isinstance(self.event_key, EventKey):
            raise LLMContractError("event_key must be an EventKey")
        object.__setattr__(
            self,
            "adapter_authorization_sha256",
            _sha256(
                self.adapter_authorization_sha256,
                "adapter_authorization_sha256",
            ),
        )
        if not isinstance(self.requested_use, RequestedUse):
            raise LLMContractError("requested_use must be a RequestedUse")
        object.__setattr__(
            self,
            "agent_commitment",
            _sha256(self.agent_commitment, "agent_commitment"),
        )
        object.__setattr__(self, "topic_id", _identifier(self.topic_id, "topic_id"))
        as_of_time = _aware_utc(self.as_of_time, "as_of_time")
        cutoff = _aware_utc(self.trusted_cutoff, "trusted_cutoff")
        if as_of_time > cutoff:
            raise LLMTemporalError("as_of_time is after trusted_cutoff")
        object.__setattr__(self, "as_of_time", as_of_time)
        object.__setattr__(self, "trusted_cutoff", cutoff)
        for field in (
            "cutoff_policy_sha256",
            "input_lineage_sha256",
            "prompt_template_sha256",
            "deidentification_policy_sha256",
        ):
            object.__setattr__(self, field, _sha256(getattr(self, field), field))
        if not isinstance(self.sanitized_context, SanitizedText):
            raise LLMPrivacyError("sanitized_context must be SanitizedText")
        if (
            self.sanitized_context.receipt.policy_sha256
            != self.deidentification_policy_sha256
        ):
            raise LLMPrivacyError(
                "sanitized context policy does not match the request commitment"
            )
        if not isinstance(self.sanitized_context_record, SanitizedContextRecord):
            raise LLMPrivacyError(
                "sanitized_context_record must be SanitizedContextRecord"
            )
        self.sanitized_context_record.validate_for(
            self.sanitized_context,
            as_of_time=as_of_time,
            cutoff=cutoff,
            lineage_sha256=self.input_lineage_sha256,
        )
        if not isinstance(self.public_features, tuple) or not all(
            isinstance(feature, PublicCognitiveFeature)
            for feature in self.public_features
        ):
            raise LLMContractError("public_features must be an immutable feature tuple")
        feature_codes = tuple(feature.code for feature in self.public_features)
        if len(set(feature_codes)) != len(feature_codes):
            raise LLMContractError("public feature codes must be unique")
        for feature in self.public_features:
            feature.validate_available(as_of_time=as_of_time, cutoff=cutoff)
            if feature.lineage_sha256 != self.input_lineage_sha256:
                raise LLMTemporalError(
                    "public feature lineage does not match the request lineage"
                )
        object.__setattr__(
            self,
            "public_features",
            tuple(sorted(self.public_features, key=lambda item: item.code.value)),
        )
        if not isinstance(self.allowed_actions, tuple) or not self.allowed_actions:
            raise LLMContractError("allowed_actions must be a non-empty tuple")
        if not all(isinstance(action, PlanAction) for action in self.allowed_actions):
            raise LLMContractError("allowed_actions must contain PlanAction values")
        if len(set(self.allowed_actions)) != len(self.allowed_actions):
            raise LLMContractError("allowed_actions must be unique")
        object.__setattr__(
            self,
            "allowed_actions",
            tuple(sorted(self.allowed_actions, key=lambda action: action.value)),
        )

    @property
    def input_sha256(self) -> str:
        """Commit to the exact allowlisted request sent to an adapter."""
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def to_mapping(self) -> dict[str, object]:
        """Return the complete strict request, including only sanitized text."""
        return {
            "adapter_authorization_sha256": self.adapter_authorization_sha256,
            "agent_commitment": self.agent_commitment,
            "allowed_actions": [action.value for action in self.allowed_actions],
            "as_of_time": self.as_of_time.isoformat(),
            "cutoff_policy_sha256": self.cutoff_policy_sha256,
            "deidentification_policy_sha256": (self.deidentification_policy_sha256),
            "event_key": _event_key_mapping(self.event_key),
            "input_lineage_sha256": self.input_lineage_sha256,
            "output_schema_version": self.output_schema_version,
            "prompt_template_sha256": self.prompt_template_sha256,
            "public_features": [
                feature.to_mapping() for feature in self.public_features
            ],
            "requested_use": self.requested_use.value,
            "sanitized_context": self.sanitized_context.to_mapping(),
            "sanitized_context_record": self.sanitized_context_record.to_mapping(),
            "schema_version": self.schema_version,
            "topic_id": self.topic_id,
            "trusted_cutoff": self.trusted_cutoff.isoformat(),
        }

    def to_adapter_mapping(self) -> dict[str, object]:
        """Return the allowlisted payload exposed to the adapter backend."""
        mapping = self.to_mapping()
        mapping["event_key"] = self.event_key.value
        mapping["input_sha256"] = self.input_sha256
        return mapping

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> Self:
        """Strictly parse a request, rejecting every unknown field."""
        _strict_fields(mapping, _REQUEST_FIELDS, "LLM request")
        raw_features = mapping["public_features"]
        if not isinstance(raw_features, list):
            raise LLMContractError("public_features must be a JSON list")
        return cls(
            event_key=_parse_event_key(mapping["event_key"]),
            adapter_authorization_sha256=_sha256(
                mapping["adapter_authorization_sha256"],
                "adapter_authorization_sha256",
            ),
            requested_use=_parse_requested_use(mapping["requested_use"]),
            agent_commitment=_sha256(mapping["agent_commitment"], "agent_commitment"),
            topic_id=_identifier(mapping["topic_id"], "topic_id"),
            as_of_time=_parse_datetime(mapping["as_of_time"], "as_of_time"),
            trusted_cutoff=_parse_datetime(mapping["trusted_cutoff"], "trusted_cutoff"),
            cutoff_policy_sha256=_sha256(
                mapping["cutoff_policy_sha256"], "cutoff_policy_sha256"
            ),
            input_lineage_sha256=_sha256(
                mapping["input_lineage_sha256"], "input_lineage_sha256"
            ),
            prompt_template_sha256=_sha256(
                mapping["prompt_template_sha256"], "prompt_template_sha256"
            ),
            deidentification_policy_sha256=_sha256(
                mapping["deidentification_policy_sha256"],
                "deidentification_policy_sha256",
            ),
            sanitized_context=_parse_sanitized_context(mapping["sanitized_context"]),
            sanitized_context_record=SanitizedContextRecord.from_mapping(
                mapping["sanitized_context_record"]
            ),
            public_features=tuple(
                PublicCognitiveFeature.from_mapping(value) for value in raw_features
            ),
            allowed_actions=_parse_actions(mapping["allowed_actions"]),
            output_schema_version=_non_negative_int(
                mapping["output_schema_version"], "output_schema_version"
            ),
            schema_version=_non_negative_int(
                mapping["schema_version"], "schema_version"
            ),
        )


def _parse_requested_use(value: object) -> RequestedUse:
    if not isinstance(value, str):
        raise LLMContractError("requested_use must be a string")
    try:
        return RequestedUse(value)
    except ValueError as error:
        raise LLMContractError("requested_use is unsupported") from error


def _parse_sanitized_context(value: object) -> SanitizedText:
    try:
        return _parse_sanitized_text(value)
    except DeidentificationError as error:
        raise LLMPrivacyError("sanitized context failed receipt validation") from error


@dataclass(frozen=True, slots=True)
class LLMPlanResponse:
    """Closed adapter output before conversion into a CognitivePlan."""

    action: PlanAction
    confidence: float
    review_step: int
    valid_until: datetime
    input_sha256: str
    rationale_code: str
    schema_version: int = LLM_RESPONSE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != LLM_RESPONSE_SCHEMA_VERSION:
            raise LLMOutputError("unsupported LLM response schema version")
        if not isinstance(self.action, PlanAction):
            raise LLMOutputError("action must be a PlanAction")
        object.__setattr__(
            self, "confidence", _unit_interval(self.confidence, "confidence")
        )
        object.__setattr__(
            self, "review_step", _non_negative_int(self.review_step, "review_step")
        )
        object.__setattr__(
            self, "valid_until", _aware_utc(self.valid_until, "valid_until")
        )
        object.__setattr__(
            self, "input_sha256", _sha256(self.input_sha256, "input_sha256")
        )
        if (
            not isinstance(self.rationale_code, str)
            or _RATIONALE_PATTERN.fullmatch(self.rationale_code) is None
        ):
            raise LLMOutputError("rationale_code must be a closed uppercase code")

    def validate_for(self, request: StrictLLMRequest) -> None:
        """Bind output to the request, allowed actions, and time boundary."""
        if self.input_sha256 != request.input_sha256:
            raise LLMOutputError("response input commitment does not match request")
        if self.action not in request.allowed_actions:
            raise LLMOutputError("response action was not allowed by the request")
        if self.valid_until < request.as_of_time:
            raise LLMTemporalError("response valid_until is before as_of_time")
        if self.valid_until > request.trusted_cutoff:
            raise LLMTemporalError("response valid_until is after trusted_cutoff")

    def to_mapping(self) -> dict[str, object]:
        """Return the exact adapter response schema."""
        return {
            "action": self.action.value,
            "confidence": self.confidence,
            "input_sha256": self.input_sha256,
            "rationale_code": self.rationale_code,
            "review_step": self.review_step,
            "schema_version": self.schema_version,
            "valid_until": self.valid_until.isoformat(),
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> Self:
        """Reject unknown fields, illegal actions, and malformed values."""
        try:
            _strict_fields(mapping, _RESPONSE_FIELDS, "LLM response")
            action = PlanAction(_string(mapping["action"], "action"))
            response = cls(
                action=action,
                confidence=_unit_interval(mapping["confidence"], "confidence"),
                review_step=_non_negative_int(mapping["review_step"], "review_step"),
                valid_until=_parse_datetime(mapping["valid_until"], "valid_until"),
                input_sha256=_sha256(mapping["input_sha256"], "input_sha256"),
                rationale_code=_string(mapping["rationale_code"], "rationale_code"),
                schema_version=_non_negative_int(
                    mapping["schema_version"], "schema_version"
                ),
            )
        except (LLMContractError, TypeError, ValueError) as error:
            if isinstance(error, LLMOutputError):
                raise
            raise LLMOutputError(
                "adapter output failed strict schema validation"
            ) from error
        return response


__all__ = [
    "LLM_REQUEST_SCHEMA_VERSION",
    "LLM_RESPONSE_SCHEMA_VERSION",
    "LLMContractError",
    "LLMOutputError",
    "LLMPlanResponse",
    "LLMPrivacyError",
    "LLMTemporalError",
    "PublicCognitiveFeature",
    "PublicFeatureCode",
    "SanitizedContextRecord",
    "StrictLLMRequest",
]
