"""Executable synthetic structural-member generators.

These generators turn effect draws from two genuinely different mechanisms
into the exact paired-draw schema consumed by the structural ensemble.  They
are intentionally synthetic-only: one executes heterogeneous agent/social/
memory interactions and the other executes an order-arrival generator.  The
resulting receipts prove code execution and deterministic replay, not empirical
validity or real-market calibration.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NoReturn, Protocol, runtime_checkable

from policytwin.evaluation.baselines import (
    AuditedBaselineAdapter,
    BaselineInputBundle,
    BaselineMechanismReceipt,
    FullPolicyTwinSyntheticBaseline,
    OrderLevelGeneratorBaseline,
    audit_baseline_execution,
    execute_baseline_adapter,
)
from policytwin.evaluation.ensemble import (
    MemberResultStatus,
    ModelEvidenceStatus,
    StructuralEnsembleArtifact,
    StructuralEnsembleSpec,
    StructuralMemberResult,
    StructuralModelMember,
    StructuralOutcomeCell,
    StructuralPairedDraw,
    evaluate_structural_ensemble,
)
from policytwin.provenance import canonical_json_bytes, sha256_bytes, sha256_file

STRUCTURAL_GENERATOR_SCHEMA_VERSION = "policytwin.structural-generator.v1"
STRUCTURAL_GENERATOR_CLAIM_SCOPE = "synthetic_structural_mechanism_only"

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class StructuralGeneratorError(RuntimeError):
    """Raised when a generated member cannot be audited or compared."""


def _fail(message: str) -> NoReturn:
    raise StructuralGeneratorError(message)


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        _fail(f"{field} must be a stable identifier")
    return value


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail(f"{field} must be a lowercase SHA-256 digest")
    return value


def _source_sha256() -> str:
    path = Path(__file__).resolve()
    if not path.is_file():
        _fail("structural generator source is unavailable")
    return sha256_file(path)


@dataclass(frozen=True, slots=True)
class StructuralGeneratorDescriptor:
    """Identity of a generator wrapper and its required mechanism family."""

    generator_id: str
    family: str
    description: str
    source_sha256: str
    adapter_implementation_sha256: str
    implementation_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.generator_id, "generator_id")
        _identifier(self.family, "generator family")
        if not isinstance(self.description, str) or not self.description.strip():
            _fail("generator description must be non-empty")
        for field in (
            "source_sha256",
            "adapter_implementation_sha256",
            "implementation_sha256",
        ):
            _sha256(getattr(self, field), field)
        if self.implementation_sha256 != self.computed_implementation_sha256:
            _fail("generator implementation hash does not match source and adapter")

    @property
    def computed_implementation_sha256(self) -> str:
        return sha256_bytes(
            canonical_json_bytes(
                {
                    "adapter_implementation_sha256": (
                        self.adapter_implementation_sha256
                    ),
                    "description": self.description,
                    "family": self.family,
                    "generator_id": self.generator_id,
                    "source_sha256": self.source_sha256,
                }
            )
        )

    @classmethod
    def current(
        cls,
        *,
        generator_id: str,
        family: str,
        description: str,
        adapter_implementation_sha256: str,
    ) -> StructuralGeneratorDescriptor:
        source = _source_sha256()
        material = {
            "adapter_implementation_sha256": adapter_implementation_sha256,
            "description": description,
            "family": family,
            "generator_id": generator_id,
            "source_sha256": source,
        }
        return cls(
            generator_id=generator_id,
            family=family,
            description=description,
            source_sha256=source,
            adapter_implementation_sha256=adapter_implementation_sha256,
            implementation_sha256=sha256_bytes(canonical_json_bytes(material)),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "adapter_implementation_sha256": self.adapter_implementation_sha256,
            "description": self.description,
            "family": self.family,
            "generator_id": self.generator_id,
            "implementation_sha256": self.implementation_sha256,
            "source_sha256": self.source_sha256,
        }


@dataclass(frozen=True, slots=True)
class StructuralMemberExecutionReceipt:
    """Cross-binding of generated member, common input, code, and effect draws."""

    descriptor: StructuralGeneratorDescriptor
    input_bundle_sha256: str
    member_sha256: str
    cell_sha256: str
    result_sha256: str
    baseline_receipt: BaselineMechanismReceipt
    mechanism_execution: Literal[True] = True
    synthetic_only: Literal[True] = True
    claim_scope: str = STRUCTURAL_GENERATOR_CLAIM_SCOPE
    schema_version: str = STRUCTURAL_GENERATOR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.descriptor, StructuralGeneratorDescriptor):
            _fail("structural receipt descriptor is invalid")
        for field in (
            "input_bundle_sha256",
            "member_sha256",
            "cell_sha256",
            "result_sha256",
        ):
            _sha256(getattr(self, field), field)
        if not isinstance(self.baseline_receipt, BaselineMechanismReceipt):
            _fail("structural receipt requires an audited baseline receipt")
        if self.baseline_receipt.input_bundle_sha256 != self.input_bundle_sha256:
            _fail("structural and baseline receipts bind different common inputs")
        if self.mechanism_execution is not True or self.synthetic_only is not True:
            _fail("structural execution receipt must be executed and synthetic-only")
        if self.claim_scope != STRUCTURAL_GENERATOR_CLAIM_SCOPE:
            _fail("structural generator claim scope cannot be widened")
        if self.schema_version != STRUCTURAL_GENERATOR_SCHEMA_VERSION:
            _fail("unsupported structural generator receipt schema")

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def to_mapping(self) -> dict[str, object]:
        return {
            "baseline_receipt": self.baseline_receipt.to_mapping(),
            "cell_sha256": self.cell_sha256,
            "claim_scope": self.claim_scope,
            "descriptor": self.descriptor.to_mapping(),
            "input_bundle_sha256": self.input_bundle_sha256,
            "mechanism_execution": self.mechanism_execution,
            "member_sha256": self.member_sha256,
            "result_sha256": self.result_sha256,
            "schema_version": self.schema_version,
            "synthetic_only": self.synthetic_only,
        }


@dataclass(frozen=True, slots=True)
class StructuralMemberExecution:
    """One generated ensemble-member result and its non-detachable receipt."""

    result: StructuralMemberResult
    receipt: StructuralMemberExecutionReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.result, StructuralMemberResult) or not isinstance(
            self.receipt, StructuralMemberExecutionReceipt
        ):
            _fail("structural execution requires typed result and receipt")
        if self.result.status is not MemberResultStatus.COMPLETED:
            _fail("generated structural member must complete")
        if (
            self.receipt.result_sha256 != self.result.sha256
            or self.receipt.member_sha256 != self.result.member_sha256
            or self.receipt.cell_sha256 != self.result.cell.sha256
        ):
            _fail("generated result disagrees with its structural receipt")
        if tuple(draw.event_key for draw in self.result.draws) != (
            self.receipt.baseline_receipt.event_keys
        ):
            _fail("generated result event keys differ from common input keys")
        if tuple(draw.difference for draw in self.result.draws) != (
            self.receipt.baseline_receipt.draws
        ):
            _fail("generated member effects differ from baseline mechanism outputs")

    @property
    def sha256(self) -> str:
        return sha256_bytes(
            canonical_json_bytes(
                {
                    "receipt": self.receipt.to_mapping(),
                    "result": self.result.to_mapping(),
                }
            )
        )


@runtime_checkable
class StructuralMemberGenerator(Protocol):
    """Generator that consumes the same frozen input bundle as every baseline."""

    @property
    def member(self) -> StructuralModelMember: ...

    @property
    def adapter(self) -> AuditedBaselineAdapter: ...

    def descriptor(self) -> StructuralGeneratorDescriptor: ...

    def generate(
        self,
        inputs: BaselineInputBundle,
        cell: StructuralOutcomeCell,
    ) -> StructuralMemberExecution: ...


def _build_execution(
    *,
    member: StructuralModelMember,
    descriptor: StructuralGeneratorDescriptor,
    inputs: BaselineInputBundle,
    cell: StructuralOutcomeCell,
    baseline_receipt: BaselineMechanismReceipt,
) -> StructuralMemberExecution:
    if inputs.synthetic_only is not True:
        _fail("structural generators are synthetic-only")
    if cell.study_protocol_sha256 != inputs.study_protocol_sha256:
        _fail("structural outcome cell and common input use different protocols")
    draws = tuple(
        StructuralPairedDraw(
            event_key=event_key,
            draw_id=f"draw_{index:06d}",
            baseline_value=0.0,
            candidate_value=effect,
            difference=effect,
        )
        for index, (event_key, effect) in enumerate(
            zip(
                baseline_receipt.event_keys,
                baseline_receipt.draws,
                strict=True,
            )
        )
    )
    result = StructuralMemberResult(
        model_id=member.model_id,
        member_sha256=member.sha256,
        cell=cell,
        status=MemberResultStatus.COMPLETED,
        draws=draws,
    )
    receipt = StructuralMemberExecutionReceipt(
        descriptor=descriptor,
        input_bundle_sha256=inputs.sha256,
        member_sha256=member.sha256,
        cell_sha256=cell.sha256,
        result_sha256=result.sha256,
        baseline_receipt=baseline_receipt,
    )
    return StructuralMemberExecution(result=result, receipt=receipt)


@dataclass(frozen=True, slots=True)
class AgentInteractionStructuralGenerator:
    """Structural member backed by heterogeneous/social/memory interaction."""

    model_id: str
    adapter: FullPolicyTwinSyntheticBaseline

    def __post_init__(self) -> None:
        _identifier(self.model_id, "model_id")
        if not isinstance(self.adapter, FullPolicyTwinSyntheticBaseline):
            _fail("agent generator requires FullPolicyTwinSyntheticBaseline")

    @property
    def member(self) -> StructuralModelMember:
        return StructuralModelMember(
            model_id=self.model_id,
            family="heterogeneous_agent_interaction",
            config_sha256=self.adapter.spec.sha256,
            checkpoint_sha256=self.adapter.posterior_artifact_sha256,
            evidence_status=ModelEvidenceStatus.SYNTHETIC_ONLY,
            evidence_artifact_sha256=self.adapter.posterior_artifact_sha256,
        )

    def descriptor(self) -> StructuralGeneratorDescriptor:
        mechanism = self.adapter.describe_mechanism()
        return StructuralGeneratorDescriptor.current(
            generator_id="agent-interaction-generator.v1",
            family=self.member.family,
            description=(
                "Generate paired effects from heterogeneous private signals, "
                "social exposure, memory, and inventory restraint."
            ),
            adapter_implementation_sha256=mechanism.implementation_sha256,
        )

    def generate(
        self,
        inputs: BaselineInputBundle,
        cell: StructuralOutcomeCell,
    ) -> StructuralMemberExecution:
        _, baseline_receipt = execute_baseline_adapter(self.adapter, inputs)
        return _build_execution(
            member=self.member,
            descriptor=self.descriptor(),
            inputs=inputs,
            cell=cell,
            baseline_receipt=baseline_receipt,
        )


@dataclass(frozen=True, slots=True)
class OrderFlowStructuralGenerator:
    """Structural member backed by discrete order-arrival imbalance."""

    model_id: str
    adapter: OrderLevelGeneratorBaseline

    def __post_init__(self) -> None:
        _identifier(self.model_id, "model_id")
        if not isinstance(self.adapter, OrderLevelGeneratorBaseline):
            _fail("order-flow generator requires OrderLevelGeneratorBaseline")

    @property
    def member(self) -> StructuralModelMember:
        return StructuralModelMember(
            model_id=self.model_id,
            family="order_flow_generator",
            config_sha256=self.adapter.spec.sha256,
            checkpoint_sha256=self.adapter.generator_checkpoint_sha256,
            evidence_status=ModelEvidenceStatus.SYNTHETIC_ONLY,
            evidence_artifact_sha256=self.adapter.generator_checkpoint_sha256,
        )

    def descriptor(self) -> StructuralGeneratorDescriptor:
        mechanism = self.adapter.describe_mechanism()
        return StructuralGeneratorDescriptor.current(
            generator_id="order-flow-generator.v1",
            family=self.member.family,
            description=(
                "Generate paired effects from keyed buy/sell arrivals and "
                "depth-sensitive order imbalance impact."
            ),
            adapter_implementation_sha256=mechanism.implementation_sha256,
        )

    def generate(
        self,
        inputs: BaselineInputBundle,
        cell: StructuralOutcomeCell,
    ) -> StructuralMemberExecution:
        _, baseline_receipt = execute_baseline_adapter(self.adapter, inputs)
        return _build_execution(
            member=self.member,
            descriptor=self.descriptor(),
            inputs=inputs,
            cell=cell,
            baseline_receipt=baseline_receipt,
        )


def audit_structural_member_execution(
    generator: StructuralMemberGenerator,
    inputs: BaselineInputBundle,
    cell: StructuralOutcomeCell,
    artifact: StructuralMemberExecution,
) -> None:
    """Re-run a member generator and reject any altered result or receipt."""
    if not isinstance(generator, StructuralMemberGenerator):
        _fail("generator does not implement StructuralMemberGenerator")
    if not isinstance(artifact, StructuralMemberExecution):
        _fail("artifact must be a StructuralMemberExecution")
    audit_baseline_execution(
        generator.adapter,
        inputs,
        artifact.receipt.baseline_receipt,
    )
    if generator.generate(inputs, cell) != artifact:
        _fail("structural member execution does not reproduce")


def evaluate_generated_structural_ensemble(
    spec: StructuralEnsembleSpec,
    cell: StructuralOutcomeCell,
    executions: tuple[StructuralMemberExecution, ...],
) -> StructuralEnsembleArtifact:
    """Aggregate only complete generated members sharing one frozen input."""
    if (
        not isinstance(executions, tuple)
        or not executions
        or not all(isinstance(item, StructuralMemberExecution) for item in executions)
    ):
        _fail("executions must be a non-empty immutable generated-member tuple")
    input_hashes = {item.receipt.input_bundle_sha256 for item in executions}
    if len(input_hashes) != 1:
        _fail("generated structural members do not share one frozen input")
    return evaluate_structural_ensemble(
        spec,
        cell,
        tuple(item.result for item in executions),
    )


__all__ = [
    "AgentInteractionStructuralGenerator",
    "OrderFlowStructuralGenerator",
    "STRUCTURAL_GENERATOR_CLAIM_SCOPE",
    "STRUCTURAL_GENERATOR_SCHEMA_VERSION",
    "StructuralGeneratorDescriptor",
    "StructuralGeneratorError",
    "StructuralMemberExecution",
    "StructuralMemberExecutionReceipt",
    "StructuralMemberGenerator",
    "audit_structural_member_execution",
    "evaluate_generated_structural_ensemble",
]
