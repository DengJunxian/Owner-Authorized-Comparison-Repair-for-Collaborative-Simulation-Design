"""Finite comparison conformance and outcome-independent revision invalidation."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Any

from policytwin.coordination.schema import commitment

RULE = "maximize exact mean paired traded notional; lexicographic ties"


@dataclass(frozen=True, slots=True)
class ExpectedSlot:
    """One required design alternative/replicate scientific identity."""

    candidate: str
    replicate: int
    scientific_key: str

    def __post_init__(self) -> None:
        if not self.candidate or type(self.replicate) is not int or self.replicate < 0:
            raise ValueError("invalid expected coordinate")
        if not self.scientific_key:
            raise ValueError("missing scientific key")

    @property
    def coordinate(self) -> tuple[str, int]:
        """Return the set-level coordinate, independent of the received label order."""
        return self.candidate, self.replicate


@dataclass(frozen=True, slots=True)
class Evidence:
    """A trusted-store verification token, not an untrusted producer assertion."""

    object_id: str
    scientific_key: str
    plan_sha256: str
    capsule_sha256: str
    local_pass: bool
    value: int | None
    actual_candidate: str
    actual_replicate: int

    def __post_init__(self) -> None:
        if type(self.local_pass) is not bool:
            raise ValueError("local verification status must be boolean")
        if self.local_pass and type(self.value) is not int:
            raise ValueError("locally accepted evidence needs an integer metric")


@dataclass(frozen=True, slots=True)
class Member:
    """An evaluator's incoming label plus a separately verified evidence object."""

    candidate: str
    replicate: int
    evidence: Evidence

    def __post_init__(self) -> None:
        if not self.candidate or type(self.replicate) is not int or self.replicate < 0:
            raise ValueError("invalid incoming coordinate")

    @property
    def coordinate(self) -> tuple[str, int]:
        """Return the claimed comparison slot, not one inferred from runtime output."""
        return self.candidate, self.replicate


@dataclass(frozen=True, slots=True)
class ComparisonContract:
    """Expected scientific slots and authorized execution plans fixed before handoff."""

    session_id: str
    version: int
    slots: tuple[ExpectedSlot, ...]
    allowed_plans: tuple[str, ...]
    rule: str = RULE

    def __post_init__(self) -> None:
        if not self.slots or not self.allowed_plans:
            raise ValueError("empty comparison contract")
        if type(self.version) is not int or self.version < 1 or self.rule != RULE:
            raise ValueError("unsupported contract version or decision rule")
        if len({s.coordinate for s in self.slots}) != len(self.slots):
            raise ValueError("duplicate expected comparison slot")

    @property
    def sha256(self) -> str:
        """Bind scientific and operational permissions in the decision artifact."""
        return commitment(asdict(self))

    @property
    def candidates(self) -> tuple[str, ...]:
        """Use canonical candidate ordering for exact ties."""
        return tuple(sorted({s.candidate for s in self.slots}))


MODES = (
    "local_full_only",
    "spec_grouped",
    "strict_plan_lock",
    "comparison_contract",
    "conventional_relational",
    "minus_scientific_binding",
    "minus_slot_coverage",
)


def conform(
    contract: ComparisonContract, members: list[Member], mode: str
) -> list[str]:
    """Evaluate set-level obligations without ground-truth error labels."""
    if mode not in MODES:
        raise ValueError("unknown collector mode")
    findings = []
    if any(
        not m.evidence.local_pass or type(m.evidence.value) is not int for m in members
    ):
        findings.append("local_verification")
    if set(contract.candidates) != {m.candidate for m in members}:
        findings.append("candidate_presence")
    if mode == "local_full_only":
        return findings
    expected = {slot.coordinate: slot.scientific_key for slot in contract.slots}
    if mode == "conventional_relational":
        # Same obligations written as a relational multiset comparison, not an
        # external implementation or a claim of an exclusive contract advantage.
        wanted = Counter(
            (s.candidate, s.replicate, s.scientific_key) for s in contract.slots
        )
        seen = Counter(
            (m.candidate, m.replicate, m.evidence.scientific_key) for m in members
        )
        if wanted != seen:
            findings.append("relational_slot_science")
        if not all(m.evidence.plan_sha256 in contract.allowed_plans for m in members):
            findings.append("permitted_plan")
        return findings
    if mode != "minus_slot_coverage" and Counter(
        m.coordinate for m in members
    ) != Counter(expected.keys()):
        findings.append("slot_coverage")
    if mode not in ("minus_scientific_binding", "spec_grouped") and any(
        expected.get(m.coordinate) != m.evidence.scientific_key for m in members
    ):
        findings.append("scientific_binding")
    allowed = (
        contract.allowed_plans[:1]
        if mode == "strict_plan_lock"
        else contract.allowed_plans
    )
    if any(m.evidence.plan_sha256 not in allowed for m in members):
        findings.append("permitted_plan")
    return findings


def decide(
    contract: ComparisonContract, members: list[Member], mode: str
) -> dict[str, Any]:
    """Return a complete numerical decision only if the selected obligations pass."""
    incoming = members
    if mode != "local_full_only":
        members = [
            Member(m.evidence.actual_candidate, m.evidence.actual_replicate, m.evidence)
            for m in members
        ]
    findings = conform(contract, members, mode)
    binding = [
        {
            "candidate": m.candidate,
            "replicate": m.replicate,
            "object_id": m.evidence.object_id,
            "scientific_key": m.evidence.scientific_key,
            "plan_sha256": m.evidence.plan_sha256,
            "local_pass": m.evidence.local_pass,
        }
        for m in members
    ]
    normalizations = [
        {
            "from": list(a.coordinate),
            "to": list(b.coordinate),
            "object_id": b.evidence.object_id,
        }
        for a, b in zip(incoming, members, strict=True)
        if a.coordinate != b.coordinate
    ]
    if findings:
        return {
            "authorized": False,
            "findings": findings,
            "numerical": None,
            "normalizations": normalizations,
        }
    sums: dict[str, Fraction] = {}
    sizes = {}
    for candidate in contract.candidates:
        values = [m.evidence.value for m in members if m.candidate == candidate]
        if not values or any(type(v) is not int for v in values):
            raise RuntimeError("unreportable group passed conformance")
        sums[candidate] = Fraction(sum(v for v in values if v is not None), len(values))
        sizes[candidate] = len(values)
    order = sorted(contract.candidates, key=lambda c: (-sums[c], c))
    return {
        "authorized": True,
        "findings": [],
        "numerical": {
            "contract_sha256": contract.sha256,
            "mode": mode,
            "means": {c: [sums[c].numerator, sums[c].denominator] for c in order},
            "group_sizes": sizes,
            "top1": order[0],
            "top2": sorted(order[:2]),
            "member_capsules": [m.evidence.capsule_sha256 for m in members],
            "effective_bindings": binding,
            "normalizations": normalizations,
        },
        "normalizations": normalizations,
    }


def revision_plan(
    contract: ComparisonContract, available: list[Member]
) -> dict[str, Any]:
    """Find unsupported slots without inspecting numerical values or rankings."""
    reuse = []
    missing = []
    for slot in contract.slots:
        matching = sorted(
            m.evidence.object_id
            for m in available
            if (m.evidence.actual_candidate, m.evidence.actual_replicate)
            == slot.coordinate
            and m.evidence.local_pass
            and m.evidence.scientific_key == slot.scientific_key
            and m.evidence.plan_sha256 in contract.allowed_plans
        )
        if matching:
            reuse.append(
                {
                    "candidate": slot.candidate,
                    "replicate": slot.replicate,
                    "object_id": matching[0],
                }
            )
        else:
            missing.append({"candidate": slot.candidate, "replicate": slot.replicate})
    return {"contract_sha256": contract.sha256, "reuse": reuse, "rerun": missing}
