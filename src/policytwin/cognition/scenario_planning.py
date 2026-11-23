"""Receipt-bound slow plans for generated post-cutoff scenarios.

The ordinary :class:`~policytwin.cognition.planning.CognitivePlan` remains an
observed/pre-cutoff type.  This module deliberately does not add an exception
to that schema: a future plan is valid only for the exact engine-issued
scenario event receipt that authorized it.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime

from policytwin.cognition.planning import PlanAction, RequestedUse
from policytwin.counterfactual import ScenarioEventReceipt

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_RATIONALE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ACTION_ORDER = (PlanAction.HOLD, PlanAction.BUY, PlanAction.SELL)


class ScenarioCognitivePlanError(ValueError):
    """Raised when a generated-scenario plan is invalid or mis-bound."""


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ScenarioCognitivePlanError(
            f"{field} must be a stable anonymous identifier"
        )
    return value


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ScenarioCognitivePlanError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _actions(value: object) -> tuple[PlanAction, ...]:
    if not isinstance(value, tuple) or not value:
        raise ScenarioCognitivePlanError(
            "allowed_actions must be a non-empty immutable tuple"
        )
    if not all(isinstance(item, PlanAction) for item in value):
        raise ScenarioCognitivePlanError(
            "allowed_actions must contain only PlanAction values"
        )
    if len(set(value)) != len(value):
        raise ScenarioCognitivePlanError("allowed_actions contains duplicates")
    selected = frozenset(value)
    return tuple(item for item in _ACTION_ORDER if item in selected)


@dataclass(frozen=True, slots=True)
class ScenarioCognitivePlan:
    """Text-free slow plan authorized for exactly one generated future event."""

    agent_id: str
    topic_id: str
    receipt: ScenarioEventReceipt
    allowed_actions: tuple[PlanAction, ...]
    confidence: float
    review_step: int
    input_sha256: str
    adapter_id: str
    adapter_authorization_sha256: str
    requested_use: RequestedUse
    rationale_code: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "agent_id", _identifier(self.agent_id, "agent_id"))
        object.__setattr__(self, "topic_id", _identifier(self.topic_id, "topic_id"))
        if not isinstance(self.receipt, ScenarioEventReceipt):
            raise ScenarioCognitivePlanError(
                "scenario plan requires an engine-issued receipt"
            )
        object.__setattr__(self, "allowed_actions", _actions(self.allowed_actions))
        if isinstance(self.confidence, bool) or not isinstance(
            self.confidence, int | float
        ):
            raise ScenarioCognitivePlanError("confidence must be finite and in [0, 1]")
        confidence = float(self.confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ScenarioCognitivePlanError("confidence must be finite and in [0, 1]")
        object.__setattr__(self, "confidence", confidence)
        if (
            isinstance(self.review_step, bool)
            or not isinstance(self.review_step, int)
            or self.review_step < 0
        ):
            raise ScenarioCognitivePlanError(
                "review_step must be a non-negative integer"
            )
        object.__setattr__(
            self, "input_sha256", _sha256(self.input_sha256, "input_sha256")
        )
        object.__setattr__(
            self, "adapter_id", _identifier(self.adapter_id, "adapter_id")
        )
        object.__setattr__(
            self,
            "adapter_authorization_sha256",
            _sha256(
                self.adapter_authorization_sha256,
                "adapter_authorization_sha256",
            ),
        )
        if self.requested_use is not RequestedUse.SYNTHETIC_TEST:
            raise ScenarioCognitivePlanError(
                "generated scenario plans are synthetic-test only"
            )
        if (
            not isinstance(self.rationale_code, str)
            or _RATIONALE.fullmatch(self.rationale_code) is None
        ):
            raise ScenarioCognitivePlanError(
                "rationale_code must be a stable code without free text"
            )

    @property
    def scenario_id(self) -> str:
        return self.receipt.scenario_id

    @property
    def scenario_time(self) -> datetime:
        """Return scenario time without introducing an observed availability field."""
        return self.receipt.scenario_time

    @property
    def trusted_cutoff(self) -> datetime:
        return self.receipt.trusted_cutoff

    @property
    def cutoff_policy_sha256(self) -> str:
        return self.receipt.generator_spec.cutoff_policy_sha256

    @property
    def source_lineage_sha256(self) -> str:
        return self.receipt.generator_spec.source_lineage_sha256

    @property
    def generator_spec_sha256(self) -> str:
        return self.receipt.generator_spec.sha256

    def permits(self, action: PlanAction) -> bool:
        if not isinstance(action, PlanAction):
            raise ScenarioCognitivePlanError("action must be a PlanAction")
        return action in self.allowed_actions

    def is_valid_for(self, receipt: ScenarioEventReceipt) -> bool:
        """Require the exact engine receipt; time-only equality is insufficient."""
        if not isinstance(receipt, ScenarioEventReceipt):
            raise ScenarioCognitivePlanError(
                "plan validation requires an engine-issued receipt"
            )
        return receipt.engine_signature_sha256 == self.receipt.engine_signature_sha256

    def to_mapping(self) -> dict[str, object]:
        return {
            "adapter_authorization_sha256": self.adapter_authorization_sha256,
            "adapter_id": self.adapter_id,
            "agent_id": self.agent_id,
            "allowed_actions": [item.value for item in self.allowed_actions],
            "confidence": self.confidence,
            "input_sha256": self.input_sha256,
            "provenance": "model_generated_scenario",
            "rationale_code": self.rationale_code,
            "requested_use": self.requested_use.value,
            "review_step": self.review_step,
            "scenario_event_receipt": self.receipt.to_mapping(),
            "topic_id": self.topic_id,
        }


__all__ = ["ScenarioCognitivePlan", "ScenarioCognitivePlanError"]
