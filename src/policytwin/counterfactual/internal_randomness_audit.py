"""Protocol-lineage audit for paired internal random draw consumption.

This verifier compares only latent, event-addressed random draw receipts.  It
does not require equal probabilities, actions, plan outcomes, market states,
or welfare results across the two policy worlds.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from threading import RLock
from typing import Literal, NoReturn, Self
from weakref import WeakKeyDictionary, WeakValueDictionary, finalize

from policytwin.counterfactual.integrated_market import (
    IntegratedSyntheticMarketConfig,
    IntegratedSyntheticRunArtifact,
    RandomConsumptionEntry,
    RandomConsumptionLedger,
)
from policytwin.counterfactual.paired import PolicyWorldPair, WorldRole
from policytwin.counterfactual.randomness import (
    RNG_ALGORITHM_SPEC_COMMITMENT,
    RNG_IMPLEMENTATION_COMMITMENT,
    RNG_IMPLEMENTATION_ID,
    RandomDrawReceipt,
    assert_live_rng_implementation,
)
from policytwin.protocol import (
    ClaimScope,
    ProtocolExecutionConsumer,
    ProtocolExecutionPurpose,
    ProtocolExecutionReceipt,
    StudyMode,
    StudyProtocol,
    validate_protocol_execution_receipt_binding,
)
from policytwin.provenance import canonical_json_bytes, sha256_bytes

INTERNAL_RANDOM_COUPLING_AUDIT_SCHEMA_VERSION = (
    "policytwin.internal-random-coupling-audit.v1"
)
INTERNAL_RANDOM_COUPLING_CLAIM_SCOPE = (
    "protocol_bound_synthetic_internal_random_coupling_only"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class InternalRandomCouplingAuditError(RuntimeError):
    """Raised when paired latent draw evidence is incomplete or differs."""


class _InternalRandomAuditMintTicket:
    """Opaque, single-use authority issued inside protocol artifact minting."""

    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("internal random audit mint tickets are verifier-issued")


@dataclass(frozen=True, slots=True)
class _InternalRandomAuditMintBinding:
    baseline_run_sha256: str
    candidate_run_sha256: str
    pair_input_sha256: str
    config_sha256: str
    protocol_sha256: str
    protocol_execution_receipt_sha256: str


_AUDIT_LOCK = RLock()
_AUDIT_MINT_TICKETS: WeakKeyDictionary[
    _InternalRandomAuditMintTicket, _InternalRandomAuditMintBinding
] = WeakKeyDictionary()
_VERIFIER_OWNED_AUDITS: WeakValueDictionary[int, object] = WeakValueDictionary()
_VERIFIER_OWNED_AUDIT_SEALS: dict[int, str] = {}


def _fail(message: str) -> NoReturn:
    raise InternalRandomCouplingAuditError(message)


def _sha256(value: object, field: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        _fail(f"{field} must be an exact lowercase SHA-256 digest")
    return value


def _validated_protocol_context(
    *,
    protocol: StudyProtocol,
    config: IntegratedSyntheticMarketConfig,
    pair_input: PolicyWorldPair,
    protocol_execution_receipt: ProtocolExecutionReceipt,
) -> tuple[str, str, str, str]:
    """Return lineage only from exact typed synthetic protocol context."""
    if type(protocol) is not StudyProtocol:
        _fail("internal random audit requires the exact StudyProtocol type")
    if type(config) is not IntegratedSyntheticMarketConfig:
        _fail("internal random audit requires the exact integrated config type")
    if type(pair_input) is not PolicyWorldPair:
        _fail("internal random audit requires the exact PolicyWorldPair type")
    if type(protocol_execution_receipt) is not ProtocolExecutionReceipt:
        _fail("internal random audit requires the exact protocol receipt type")
    if (
        protocol.mode is not StudyMode.SYNTHETIC
        or protocol.claim_scope is not ClaimScope.SYNTHETIC_ONLY
        or pair_input.synthetic_only is not True
    ):
        _fail("internal random audit context must remain synthetic-only")
    receipt = protocol_execution_receipt
    if (
        receipt.consumer
        is not ProtocolExecutionConsumer.INTEGRATED_SYNTHETIC_SIMULATION
        or receipt.purpose
        is not ProtocolExecutionPurpose.PAIRED_SYNTHETIC_POLICY_SIMULATION
    ):
        _fail("internal random audit receipt has the wrong consumer or purpose")
    try:
        reconstructed_receipt = ProtocolExecutionReceipt.from_mapping(
            receipt.to_mapping()
        )
        validate_protocol_execution_receipt_binding(protocol, receipt)
    except (RuntimeError, TypeError, ValueError) as error:
        raise InternalRandomCouplingAuditError(
            "internal random audit protocol receipt failed strict reconstruction"
        ) from error
    if reconstructed_receipt != receipt:
        _fail("internal random audit protocol receipt changed after reconstruction")
    from policytwin.experiments.integrated_synthetic import (
        integrated_synthetic_pair_input_sha256,
    )

    pair_input_sha256 = integrated_synthetic_pair_input_sha256(pair_input)
    if (
        receipt.config_sha256 != config.sha256
        or receipt.input_sha256 != pair_input_sha256
    ):
        _fail("internal random audit receipt does not bind the typed config and pair")
    return protocol.sha256, config.sha256, pair_input_sha256, receipt.sha256


def _validate_run_context_bindings(
    *,
    baseline_run: IntegratedSyntheticRunArtifact,
    candidate_run: IntegratedSyntheticRunArtifact,
    config: IntegratedSyntheticMarketConfig,
    pair_input: PolicyWorldPair,
) -> None:
    """Bind both runs to every typed scalar supplied by the paired input."""
    if (
        type(baseline_run) is not IntegratedSyntheticRunArtifact
        or type(candidate_run) is not IntegratedSyntheticRunArtifact
    ):
        _fail("internal random audit requires exact integrated run artifacts")
    expected: tuple[tuple[str, object, object], ...] = (
        ("baseline role", baseline_run.role, WorldRole.BASELINE),
        ("candidate role", candidate_run.role, WorldRole.CANDIDATE),
        ("baseline config", baseline_run.config_sha256, config.sha256),
        ("candidate config", candidate_run.config_sha256, config.sha256),
        (
            "baseline parameter",
            baseline_run.parameter_sha256,
            pair_input.posterior_draw.parameter_sha256,
        ),
        (
            "candidate parameter",
            candidate_run.parameter_sha256,
            pair_input.posterior_draw.parameter_sha256,
        ),
        (
            "baseline shock tape",
            baseline_run.shock_tape_sha256,
            pair_input.shock_tape.sha256,
        ),
        (
            "candidate shock tape",
            candidate_run.shock_tape_sha256,
            pair_input.shock_tape.sha256,
        ),
        (
            "baseline initial state",
            baseline_run.initial_state_sha256,
            pair_input.initial_state.sha256,
        ),
        (
            "candidate initial state",
            candidate_run.initial_state_sha256,
            pair_input.initial_state.sha256,
        ),
        (
            "baseline policy",
            baseline_run.policy_sha256,
            pair_input.baseline_policy.sha256,
        ),
        (
            "candidate policy",
            candidate_run.policy_sha256,
            pair_input.candidate_policy.sha256,
        ),
    )
    for field, observed, wanted in expected:
        if observed != wanted:
            _fail(f"internal random audit {field} differs from typed context")


def _replay_exact_run(
    run: IntegratedSyntheticRunArtifact,
    *,
    role: WorldRole,
) -> RandomConsumptionLedger:
    if type(run) is not IntegratedSyntheticRunArtifact:
        _fail(f"{role.value} run must use the exact integrated artifact type")
    if run.role is not role:
        _fail(f"{role.value} run has the wrong world role")
    try:
        recomputed_run = IntegratedSyntheticRunArtifact.from_mapping(run.to_mapping())
    except (RuntimeError, TypeError, ValueError) as error:
        raise InternalRandomCouplingAuditError(
            f"{role.value} run failed complete deterministic reconstruction"
        ) from error
    if recomputed_run.canonical_bytes != run.canonical_bytes:
        _fail(f"{role.value} run changed after deterministic reconstruction")

    ledger = run.random_consumption_ledger
    if type(ledger) is not RandomConsumptionLedger:
        _fail(f"{role.value} random ledger must use the exact built-in type")
    try:
        recomputed_ledger = RandomConsumptionLedger.from_mapping(ledger.to_mapping())
    except (RuntimeError, TypeError, ValueError) as error:
        raise InternalRandomCouplingAuditError(
            f"{role.value} random ledger failed deterministic reconstruction"
        ) from error
    if recomputed_ledger.to_mapping() != ledger.to_mapping():
        _fail(f"{role.value} random ledger changed after reconstruction")
    if (
        ledger.single_world_random_consumption_complete is not True
        or ledger.cross_world_internal_random_coupling_verified is not False
        or run.cross_world_internal_random_coupling_verified is not False
    ):
        _fail(f"{role.value} evidence changed its single-world claim boundary")

    for entry in ledger.entries:
        if type(entry) is not RandomConsumptionEntry:
            _fail(f"{role.value} ledger entry must use the exact built-in type")
        if type(entry.receipt) is not RandomDrawReceipt:
            _fail(f"{role.value} draw receipt must use the exact built-in type")
        try:
            recomputed_receipt = RandomDrawReceipt.from_mapping(
                entry.receipt.to_mapping()
            )
        except (RuntimeError, TypeError, ValueError) as error:
            raise InternalRandomCouplingAuditError(
                f"{role.value} draw receipt failed independent recomputation"
            ) from error
        if recomputed_receipt.to_mapping() != entry.receipt.to_mapping():
            _fail(f"{role.value} draw receipt changed after recomputation")
    return ledger


@dataclass(frozen=True, slots=True, init=False, weakref_slot=True)
class InternalRandomCouplingAudit:
    """Verifier-produced proof of equal latent draws in one synthetic pair."""

    baseline_run_sha256: str
    candidate_run_sha256: str
    baseline_ledger: RandomConsumptionLedger
    candidate_ledger: RandomConsumptionLedger
    baseline_ledger_sha256: str
    candidate_ledger_sha256: str
    role_neutral_ledger_sha256: str
    pair_input_sha256: str
    config_sha256: str
    protocol_sha256: str
    protocol_execution_receipt_sha256: str
    expected_step_count: int
    expected_agent_ids: tuple[str, ...]
    expected_entry_count: int
    rng_implementation_id: str
    rng_algorithm_spec_commitment: str
    rng_implementation_commitment: str
    claim_scope: str
    cross_world_internal_random_coupling_verified: Literal[True]
    synthetic_only: Literal[True]
    external_validity_evidence: Literal[False]
    schema_version: str

    def __init__(self) -> None:
        raise TypeError(
            "InternalRandomCouplingAudit is verifier-produced; "
            "use the protocol-bound integrated runner"
        )

    @classmethod
    def _create(
        cls,
        *,
        baseline_run: IntegratedSyntheticRunArtifact,
        candidate_run: IntegratedSyntheticRunArtifact,
        pair_input_sha256: str,
        config_sha256: str,
        protocol_sha256: str,
        protocol_execution_receipt_sha256: str,
    ) -> Self:
        baseline_ledger = _replay_exact_run(
            baseline_run,
            role=WorldRole.BASELINE,
        )
        candidate_ledger = _replay_exact_run(
            candidate_run,
            role=WorldRole.CANDIDATE,
        )
        baseline_bytes = canonical_json_bytes(baseline_ledger.to_mapping())
        candidate_bytes = canonical_json_bytes(candidate_ledger.to_mapping())
        if baseline_bytes != candidate_bytes:
            _fail("baseline and candidate latent random consumption ledgers differ")
        if (
            baseline_ledger.expected_step_count != candidate_ledger.expected_step_count
            or baseline_ledger.expected_agent_ids != candidate_ledger.expected_agent_ids
            or len(baseline_ledger.entries) != len(candidate_ledger.entries)
        ):
            _fail("paired latent random ledger dimensions differ")

        live_implementation_commitment = assert_live_rng_implementation()
        lineage = (
            ("pair_input_sha256", pair_input_sha256),
            ("config_sha256", config_sha256),
            ("protocol_sha256", protocol_sha256),
            (
                "protocol_execution_receipt_sha256",
                protocol_execution_receipt_sha256,
            ),
        )
        for field, value in lineage:
            _sha256(value, field)

        instance = object.__new__(cls)
        values: tuple[tuple[str, object], ...] = (
            ("baseline_run_sha256", baseline_run.sha256),
            ("candidate_run_sha256", candidate_run.sha256),
            ("baseline_ledger", baseline_ledger),
            ("candidate_ledger", candidate_ledger),
            ("baseline_ledger_sha256", baseline_ledger.sha256),
            ("candidate_ledger_sha256", candidate_ledger.sha256),
            ("role_neutral_ledger_sha256", sha256_bytes(baseline_bytes)),
            ("pair_input_sha256", pair_input_sha256),
            ("config_sha256", config_sha256),
            ("protocol_sha256", protocol_sha256),
            (
                "protocol_execution_receipt_sha256",
                protocol_execution_receipt_sha256,
            ),
            ("expected_step_count", baseline_ledger.expected_step_count),
            ("expected_agent_ids", baseline_ledger.expected_agent_ids),
            ("expected_entry_count", len(baseline_ledger.entries)),
            ("rng_implementation_id", RNG_IMPLEMENTATION_ID),
            ("rng_algorithm_spec_commitment", RNG_ALGORITHM_SPEC_COMMITMENT),
            ("rng_implementation_commitment", live_implementation_commitment),
            ("claim_scope", INTERNAL_RANDOM_COUPLING_CLAIM_SCOPE),
            ("cross_world_internal_random_coupling_verified", True),
            ("synthetic_only", True),
            ("external_validity_evidence", False),
            ("schema_version", INTERNAL_RANDOM_COUPLING_AUDIT_SCHEMA_VERSION),
        )
        for attribute, attribute_value in values:
            object.__setattr__(instance, attribute, attribute_value)
        instance._validate()
        return instance

    def _validate(self) -> None:
        for field in (
            "baseline_run_sha256",
            "candidate_run_sha256",
            "baseline_ledger_sha256",
            "candidate_ledger_sha256",
            "role_neutral_ledger_sha256",
            "pair_input_sha256",
            "config_sha256",
            "protocol_sha256",
            "protocol_execution_receipt_sha256",
        ):
            _sha256(getattr(self, field), field)
        if (
            type(self.baseline_ledger) is not RandomConsumptionLedger
            or type(self.candidate_ledger) is not RandomConsumptionLedger
        ):
            _fail("audit ledgers must use exact built-in random ledger types")
        baseline_bytes = canonical_json_bytes(self.baseline_ledger.to_mapping())
        candidate_bytes = canonical_json_bytes(self.candidate_ledger.to_mapping())
        if baseline_bytes != candidate_bytes:
            _fail("audit latent random ledgers differ")
        if (
            self.baseline_ledger_sha256 != self.baseline_ledger.sha256
            or self.candidate_ledger_sha256 != self.candidate_ledger.sha256
            or self.role_neutral_ledger_sha256 != sha256_bytes(baseline_bytes)
        ):
            _fail("audit ledger commitments do not reproduce")
        if (
            type(self.expected_step_count) is not int
            or self.expected_step_count != self.baseline_ledger.expected_step_count
            or type(self.expected_agent_ids) is not tuple
            or self.expected_agent_ids != self.baseline_ledger.expected_agent_ids
            or type(self.expected_entry_count) is not int
            or self.expected_entry_count != len(self.baseline_ledger.entries)
            or self.expected_entry_count
            != self.expected_step_count * len(self.expected_agent_ids) * 2
        ):
            _fail("audit expected random ledger dimensions do not reproduce")
        if (
            self.rng_implementation_id != RNG_IMPLEMENTATION_ID
            or self.rng_algorithm_spec_commitment != RNG_ALGORITHM_SPEC_COMMITMENT
            or self.rng_implementation_commitment != RNG_IMPLEMENTATION_COMMITMENT
            or self.rng_implementation_commitment != assert_live_rng_implementation()
        ):
            _fail("audit RNG implementation lineage changed")
        if self.claim_scope != INTERNAL_RANDOM_COUPLING_CLAIM_SCOPE:
            _fail("audit claim scope changed")
        if self.cross_world_internal_random_coupling_verified is not True:
            _fail("audit must verify paired internal random coupling")
        if (
            self.synthetic_only is not True
            or self.external_validity_evidence is not False
        ):
            _fail("audit must remain synthetic-only and non-external")
        if self.schema_version != INTERNAL_RANDOM_COUPLING_AUDIT_SCHEMA_VERSION:
            _fail("unsupported internal random coupling audit schema")

    def _raw_mapping(self) -> dict[str, object]:
        return {
            "baseline_ledger": self.baseline_ledger.to_mapping(),
            "baseline_ledger_sha256": self.baseline_ledger_sha256,
            "baseline_run_sha256": self.baseline_run_sha256,
            "candidate_ledger": self.candidate_ledger.to_mapping(),
            "candidate_ledger_sha256": self.candidate_ledger_sha256,
            "candidate_run_sha256": self.candidate_run_sha256,
            "claim_scope": self.claim_scope,
            "config_sha256": self.config_sha256,
            "cross_world_internal_random_coupling_verified": (
                self.cross_world_internal_random_coupling_verified
            ),
            "expected_agent_ids": list(self.expected_agent_ids),
            "expected_entry_count": self.expected_entry_count,
            "expected_step_count": self.expected_step_count,
            "external_validity_evidence": self.external_validity_evidence,
            "pair_input_sha256": self.pair_input_sha256,
            "protocol_execution_receipt_sha256": (
                self.protocol_execution_receipt_sha256
            ),
            "protocol_sha256": self.protocol_sha256,
            "rng_algorithm_spec_commitment": self.rng_algorithm_spec_commitment,
            "rng_implementation_commitment": self.rng_implementation_commitment,
            "rng_implementation_id": self.rng_implementation_id,
            "role_neutral_ledger_sha256": self.role_neutral_ledger_sha256,
            "schema_version": self.schema_version,
            "synthetic_only": self.synthetic_only,
        }

    def to_mapping(self) -> dict[str, object]:
        """Return evidence only while this exact audit retains its mint seal."""
        _require_sealed_internal_random_coupling_audit(self)
        return self._raw_mapping()

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_mapping())

    @property
    def sha256(self) -> str:
        return sha256_bytes(self.canonical_bytes)


def _issue_internal_random_audit_mint_ticket(
    *,
    baseline_run: IntegratedSyntheticRunArtifact,
    candidate_run: IntegratedSyntheticRunArtifact,
    protocol: StudyProtocol,
    config: IntegratedSyntheticMarketConfig,
    pair_input: PolicyWorldPair,
    protocol_execution_receipt: ProtocolExecutionReceipt,
) -> _InternalRandomAuditMintTicket:
    """Issue one private ticket after the protocol artifact ticket is consumed."""
    (
        protocol_sha256,
        config_sha256,
        pair_input_sha256,
        protocol_execution_receipt_sha256,
    ) = _validated_protocol_context(
        protocol=protocol,
        config=config,
        pair_input=pair_input,
        protocol_execution_receipt=protocol_execution_receipt,
    )
    _validate_run_context_bindings(
        baseline_run=baseline_run,
        candidate_run=candidate_run,
        config=config,
        pair_input=pair_input,
    )
    binding = _InternalRandomAuditMintBinding(
        baseline_run_sha256=baseline_run.sha256,
        candidate_run_sha256=candidate_run.sha256,
        pair_input_sha256=_sha256(pair_input_sha256, "pair_input_sha256"),
        config_sha256=_sha256(config_sha256, "config_sha256"),
        protocol_sha256=_sha256(protocol_sha256, "protocol_sha256"),
        protocol_execution_receipt_sha256=_sha256(
            protocol_execution_receipt_sha256,
            "protocol_execution_receipt_sha256",
        ),
    )
    ticket = object.__new__(_InternalRandomAuditMintTicket)
    with _AUDIT_LOCK:
        _AUDIT_MINT_TICKETS[ticket] = binding
    return ticket


def _forget_internal_random_coupling_audit(identity: int) -> None:
    with _AUDIT_LOCK:
        if _VERIFIER_OWNED_AUDITS.get(identity) is None:
            _VERIFIER_OWNED_AUDIT_SEALS.pop(identity, None)


def _raw_audit_sha256(audit: InternalRandomCouplingAudit) -> str:
    return sha256_bytes(canonical_json_bytes(audit._raw_mapping()))


def _require_sealed_internal_random_coupling_audit(
    audit: InternalRandomCouplingAudit,
) -> InternalRandomCouplingAudit:
    if type(audit) is not InternalRandomCouplingAudit:
        _fail("internal random coupling audit has the wrong concrete type")
    try:
        current_seal = _raw_audit_sha256(audit)
        audit._validate()
    except Exception as error:
        raise InternalRandomCouplingAuditError(
            "internal random coupling audit cannot reproduce its mint seal"
        ) from error
    with _AUDIT_LOCK:
        registered = _VERIFIER_OWNED_AUDITS.get(id(audit))
        mint_seal = _VERIFIER_OWNED_AUDIT_SEALS.get(id(audit))
    if registered is not audit or mint_seal != current_seal:
        _fail(
            "internal random coupling audit was not verifier-minted or differs "
            "from its mint-time seal"
        )
    return audit


def _mint_internal_random_coupling_audit(
    *,
    mint_ticket: _InternalRandomAuditMintTicket,
    baseline_run: IntegratedSyntheticRunArtifact,
    candidate_run: IntegratedSyntheticRunArtifact,
    protocol: StudyProtocol,
    config: IntegratedSyntheticMarketConfig,
    pair_input: PolicyWorldPair,
    protocol_execution_receipt: ProtocolExecutionReceipt,
) -> InternalRandomCouplingAudit:
    if type(mint_ticket) is not _InternalRandomAuditMintTicket:
        _fail("internal random audit mint requires an exact verifier ticket")
    with _AUDIT_LOCK:
        binding = _AUDIT_MINT_TICKETS.pop(mint_ticket, None)
    if binding is None:
        _fail("internal random audit mint ticket is invalid or already consumed")
    (
        protocol_sha256,
        config_sha256,
        pair_input_sha256,
        protocol_execution_receipt_sha256,
    ) = _validated_protocol_context(
        protocol=protocol,
        config=config,
        pair_input=pair_input,
        protocol_execution_receipt=protocol_execution_receipt,
    )
    _validate_run_context_bindings(
        baseline_run=baseline_run,
        candidate_run=candidate_run,
        config=config,
        pair_input=pair_input,
    )
    expected_binding = _InternalRandomAuditMintBinding(
        baseline_run_sha256=baseline_run.sha256,
        candidate_run_sha256=candidate_run.sha256,
        pair_input_sha256=pair_input_sha256,
        config_sha256=config_sha256,
        protocol_sha256=protocol_sha256,
        protocol_execution_receipt_sha256=protocol_execution_receipt_sha256,
    )
    if binding != expected_binding:
        _fail("internal random audit mint fields differ from the ticket binding")
    audit = InternalRandomCouplingAudit._create(
        baseline_run=baseline_run,
        candidate_run=candidate_run,
        pair_input_sha256=pair_input_sha256,
        config_sha256=config_sha256,
        protocol_sha256=protocol_sha256,
        protocol_execution_receipt_sha256=protocol_execution_receipt_sha256,
    )
    seal = _raw_audit_sha256(audit)
    identity = id(audit)
    with _AUDIT_LOCK:
        _VERIFIER_OWNED_AUDITS[identity] = audit
        _VERIFIER_OWNED_AUDIT_SEALS[identity] = seal
    finalize(audit, _forget_internal_random_coupling_audit, identity)
    return audit


def require_verifier_owned_internal_random_coupling_audit(
    audit: InternalRandomCouplingAudit,
    *,
    baseline_run: IntegratedSyntheticRunArtifact,
    candidate_run: IntegratedSyntheticRunArtifact,
    protocol: StudyProtocol,
    config: IntegratedSyntheticMarketConfig,
    pair_input: PolicyWorldPair,
    protocol_execution_receipt: ProtocolExecutionReceipt,
) -> InternalRandomCouplingAudit:
    """Require a sealed audit and rederive it from actual typed run context."""
    _require_sealed_internal_random_coupling_audit(audit)
    (
        protocol_sha256,
        config_sha256,
        pair_input_sha256,
        protocol_execution_receipt_sha256,
    ) = _validated_protocol_context(
        protocol=protocol,
        config=config,
        pair_input=pair_input,
        protocol_execution_receipt=protocol_execution_receipt,
    )
    _validate_run_context_bindings(
        baseline_run=baseline_run,
        candidate_run=candidate_run,
        config=config,
        pair_input=pair_input,
    )
    baseline_ledger = _replay_exact_run(baseline_run, role=WorldRole.BASELINE)
    candidate_ledger = _replay_exact_run(candidate_run, role=WorldRole.CANDIDATE)
    if canonical_json_bytes(baseline_ledger.to_mapping()) != canonical_json_bytes(
        candidate_ledger.to_mapping()
    ):
        _fail("baseline and candidate latent random consumption ledgers differ")
    expected_scalars: tuple[tuple[str, object, object], ...] = (
        ("baseline_run_sha256", audit.baseline_run_sha256, baseline_run.sha256),
        ("candidate_run_sha256", audit.candidate_run_sha256, candidate_run.sha256),
        ("pair_input_sha256", audit.pair_input_sha256, pair_input_sha256),
        ("config_sha256", audit.config_sha256, config_sha256),
        ("protocol_sha256", audit.protocol_sha256, protocol_sha256),
        (
            "protocol_execution_receipt_sha256",
            audit.protocol_execution_receipt_sha256,
            protocol_execution_receipt_sha256,
        ),
    )
    for field, observed, expected in expected_scalars:
        if observed != _sha256(expected, field):
            _fail(f"internal random audit {field} differs from typed context")
    if (
        audit.baseline_ledger.to_mapping() != baseline_ledger.to_mapping()
        or audit.candidate_ledger.to_mapping() != candidate_ledger.to_mapping()
    ):
        _fail("internal random audit ledgers differ from typed run context")
    return audit


def verify_internal_random_coupling(
    *,
    baseline_run: IntegratedSyntheticRunArtifact,
    candidate_run: IntegratedSyntheticRunArtifact,
    pair_input_sha256: str,
    config_sha256: str,
    protocol_sha256: str,
    protocol_execution_receipt_sha256: str,
) -> NoReturn:
    """Hard-fail the removed digest-only claim factory.

    The name remains only so older modules can import and fail explicitly.  It
    cannot create, clone, serialize, or validate claim-bearing evidence.
    """
    _ = (
        baseline_run,
        candidate_run,
        pair_input_sha256,
        config_sha256,
        protocol_sha256,
        protocol_execution_receipt_sha256,
    )
    _fail(
        "digest-only internal random coupling verification is prohibited; "
        "use a verifier-owned protocol-bound integrated artifact"
    )


__all__ = [
    "INTERNAL_RANDOM_COUPLING_AUDIT_SCHEMA_VERSION",
    "INTERNAL_RANDOM_COUPLING_CLAIM_SCOPE",
    "InternalRandomCouplingAudit",
    "InternalRandomCouplingAuditError",
    "require_verifier_owned_internal_random_coupling_audit",
]
