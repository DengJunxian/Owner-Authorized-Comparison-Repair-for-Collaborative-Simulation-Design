"""Auditable ledgers and explicitly multi-objective policy trade-offs.

This module never treats a latent strategy cohort as an observed identity and
does not manufacture a universal scalar welfare measure.  Frozen preference
profiles may produce clearly labelled *preference scores* for sensitivity
analysis, while the underlying objective vectors and Pareto frontier remain
available in full.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import NoReturn

from policytwin.protocol import CohortBasis, FrozenCohort
from policytwin.provenance import canonical_json_bytes, sha256_bytes

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class WelfareEvaluationError(ValueError):
    """Raised when a ledger or multi-objective evaluation is ill-defined."""


def _fail(message: str) -> NoReturn:
    raise WelfareEvaluationError(message)


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        _fail(f"{field} must be a stable identifier")
    return value


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail(f"{field} must be a lowercase SHA-256 digest")
    return value


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        _fail(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{field} must be finite")
    return result


def _integer(value: object, field: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"{field} must be an integer >= {minimum}")
    return value


def _aware(value: object, field: str) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        _fail(f"{field} must be a timezone-aware datetime")
    return value.astimezone(UTC)


class LedgerPartyKind(StrEnum):
    """Closed account types for the institutional double-entry ledger."""

    AGENT = "agent"
    VENUE = "venue"
    GOVERNMENT = "government"


class LedgerFlowKind(StrEnum):
    """Closed transfer reasons with fixed payer/recipient semantics."""

    VENUE_FEE = "venue_fee"
    GOVERNMENT_TAX = "government_tax"
    GOVERNMENT_TRANSFER = "government_transfer"


@dataclass(frozen=True, slots=True)
class LedgerAccount:
    """One exact-minor-unit account; agents retain cohort provenance."""

    account_id: str
    kind: LedgerPartyKind
    opening_balance_minor: int
    closing_balance_minor: int
    cohort: FrozenCohort | None = None

    def __post_init__(self) -> None:
        _identifier(self.account_id, "ledger account_id")
        if not isinstance(self.kind, LedgerPartyKind):
            _fail("ledger account kind must be a LedgerPartyKind")
        _integer(self.opening_balance_minor, "opening_balance_minor", minimum=0)
        _integer(self.closing_balance_minor, "closing_balance_minor", minimum=0)
        if self.kind is LedgerPartyKind.AGENT:
            if not isinstance(self.cohort, FrozenCohort):
                _fail("agent ledger accounts require a FrozenCohort")
            if self.cohort.basis is CohortBasis.POST_TREATMENT_DESCRIPTIVE:
                _fail("agent ledger cohorts must be defined before treatment")
        elif self.cohort is not None:
            _fail("venue and government accounts must not carry an agent cohort")

    def to_mapping(self) -> dict[str, object]:
        return {
            "account_id": self.account_id,
            "closing_balance_minor": self.closing_balance_minor,
            "cohort": self.cohort.to_mapping() if self.cohort is not None else None,
            "kind": self.kind.value,
            "opening_balance_minor": self.opening_balance_minor,
        }


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    """One exact transfer recorded for both payer and recipient."""

    entry_id: str
    payer_account_id: str
    recipient_account_id: str
    amount_minor: int
    flow_kind: LedgerFlowKind

    def __post_init__(self) -> None:
        _identifier(self.entry_id, "ledger entry_id")
        _identifier(self.payer_account_id, "payer_account_id")
        _identifier(self.recipient_account_id, "recipient_account_id")
        if self.payer_account_id == self.recipient_account_id:
            _fail("ledger payer and recipient must differ")
        _integer(self.amount_minor, "amount_minor", minimum=1)
        if not isinstance(self.flow_kind, LedgerFlowKind):
            _fail("ledger flow_kind must be a LedgerFlowKind")

    def to_mapping(self) -> dict[str, object]:
        return {
            "amount_minor": self.amount_minor,
            "entry_id": self.entry_id,
            "flow_kind": self.flow_kind.value,
            "payer_account_id": self.payer_account_id,
            "recipient_account_id": self.recipient_account_id,
        }


@dataclass(frozen=True, slots=True)
class InstitutionalLedger:
    """A reconciled venue/government/agent ledger for one synthetic scenario."""

    ledger_id: str
    currency_minor_unit: str
    source_artifact_sha256: str
    study_protocol_sha256: str
    accounts: tuple[LedgerAccount, ...]
    entries: tuple[LedgerEntry, ...]
    synthetic_only: bool = True

    def __post_init__(self) -> None:
        _identifier(self.ledger_id, "ledger_id")
        _identifier(self.currency_minor_unit, "currency_minor_unit")
        _sha256(self.source_artifact_sha256, "source_artifact_sha256")
        _sha256(self.study_protocol_sha256, "study_protocol_sha256")
        if (
            not isinstance(self.accounts, tuple)
            or len(self.accounts) < 3
            or not all(isinstance(item, LedgerAccount) for item in self.accounts)
        ):
            _fail("ledger accounts must contain at least three typed accounts")
        if (
            not isinstance(self.entries, tuple)
            or not self.entries
            or not all(isinstance(item, LedgerEntry) for item in self.entries)
        ):
            _fail("ledger entries must be a non-empty immutable tuple")
        account_ids = tuple(account.account_id for account in self.accounts)
        entry_ids = tuple(entry.entry_id for entry in self.entries)
        if len(set(account_ids)) != len(account_ids):
            _fail("ledger account IDs must be unique")
        if len(set(entry_ids)) != len(entry_ids):
            _fail("ledger entry IDs must be unique")
        kinds = {account.kind for account in self.accounts}
        if kinds != {
            LedgerPartyKind.AGENT,
            LedgerPartyKind.VENUE,
            LedgerPartyKind.GOVERNMENT,
        }:
            _fail("ledger requires agent, venue, and government account types")
        if not isinstance(self.synthetic_only, bool):
            _fail("synthetic_only must be a boolean")
        if not self.synthetic_only:
            _fail("real welfare ledgers require a typed temporal-firewall receipt")

        accounts = {account.account_id: account for account in self.accounts}
        deltas = dict.fromkeys(account_ids, 0)
        allowed: dict[
            LedgerFlowKind,
            tuple[frozenset[LedgerPartyKind], frozenset[LedgerPartyKind]],
        ] = {
            LedgerFlowKind.VENUE_FEE: (
                frozenset({LedgerPartyKind.AGENT}),
                frozenset({LedgerPartyKind.VENUE}),
            ),
            LedgerFlowKind.GOVERNMENT_TAX: (
                frozenset({LedgerPartyKind.AGENT, LedgerPartyKind.VENUE}),
                frozenset({LedgerPartyKind.GOVERNMENT}),
            ),
            LedgerFlowKind.GOVERNMENT_TRANSFER: (
                frozenset({LedgerPartyKind.GOVERNMENT}),
                frozenset({LedgerPartyKind.AGENT}),
            ),
        }
        for entry in self.entries:
            payer = accounts.get(entry.payer_account_id)
            recipient = accounts.get(entry.recipient_account_id)
            if payer is None or recipient is None:
                _fail("every ledger entry must reference registered accounts")
            payer_kinds, recipient_kinds = allowed[entry.flow_kind]
            if payer.kind not in payer_kinds or recipient.kind not in recipient_kinds:
                _fail(f"invalid payer/recipient types for {entry.flow_kind.value}")
            deltas[payer.account_id] -= entry.amount_minor
            deltas[recipient.account_id] += entry.amount_minor
        for account in self.accounts:
            expected = account.opening_balance_minor + deltas[account.account_id]
            if account.closing_balance_minor != expected:
                _fail(f"ledger account {account.account_id!r} does not reconcile")
        if sum(account.opening_balance_minor for account in self.accounts) != sum(
            account.closing_balance_minor for account in self.accounts
        ):
            _fail("institutional ledger does not conserve cash")

    def to_mapping(self) -> dict[str, object]:
        return {
            "accounts": [account.to_mapping() for account in self.accounts],
            "currency_minor_unit": self.currency_minor_unit,
            "entries": [entry.to_mapping() for entry in self.entries],
            "ledger_id": self.ledger_id,
            "source_artifact_sha256": self.source_artifact_sha256,
            "study_protocol_sha256": self.study_protocol_sha256,
            "synthetic_only": self.synthetic_only,
        }

    @property
    def sha256(self) -> str:
        """Commit the complete reconciled ledger."""
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))


@dataclass(frozen=True, slots=True)
class CohortLedgerFlow:
    """Fee, tax, and transfer totals for an analytical frozen cohort."""

    cohort_id: str
    cohort_basis: CohortBasis
    cohort_definition_sha256: str
    fees_paid_minor: int
    taxes_paid_minor: int
    transfers_received_minor: int

    def __post_init__(self) -> None:
        _identifier(self.cohort_id, "cohort ledger cohort_id")
        if not isinstance(self.cohort_basis, CohortBasis):
            _fail("cohort ledger basis must be a CohortBasis")
        if self.cohort_basis is CohortBasis.POST_TREATMENT_DESCRIPTIVE:
            _fail("cohort ledger basis must be defined before treatment")
        _sha256(
            self.cohort_definition_sha256,
            "cohort ledger definition_sha256",
        )
        for field in (
            "fees_paid_minor",
            "taxes_paid_minor",
            "transfers_received_minor",
        ):
            _integer(getattr(self, field), field, minimum=0)

    def to_mapping(self) -> dict[str, object]:
        return {
            "cohort_basis": self.cohort_basis.value,
            "cohort_definition_sha256": self.cohort_definition_sha256,
            "cohort_id": self.cohort_id,
            "fees_paid_minor": self.fees_paid_minor,
            "taxes_paid_minor": self.taxes_paid_minor,
            "transfers_received_minor": self.transfers_received_minor,
        }


@dataclass(frozen=True, slots=True)
class InstitutionalLedgerSummary:
    """Separate institutional receipts and outlays; no net social-welfare claim."""

    ledger_sha256: str
    opening_total_cash_minor: int
    closing_total_cash_minor: int
    venue_fee_revenue_minor: int
    venue_tax_paid_minor: int
    venue_net_revenue_minor: int
    government_tax_revenue_minor: int
    government_transfers_minor: int
    government_net_revenue_minor: int
    cohort_flows: tuple[CohortLedgerFlow, ...]
    synthetic_only: bool = True

    def __post_init__(self) -> None:
        _sha256(self.ledger_sha256, "ledger_sha256")
        for field in (
            "opening_total_cash_minor",
            "closing_total_cash_minor",
            "venue_fee_revenue_minor",
            "venue_tax_paid_minor",
            "government_tax_revenue_minor",
            "government_transfers_minor",
        ):
            _integer(getattr(self, field), field, minimum=0)
        for field in ("venue_net_revenue_minor", "government_net_revenue_minor"):
            if isinstance(getattr(self, field), bool) or not isinstance(
                getattr(self, field), int
            ):
                _fail(f"{field} must be an integer")
        if self.opening_total_cash_minor != self.closing_total_cash_minor:
            _fail("ledger summary must conserve cash")
        if self.venue_net_revenue_minor != (
            self.venue_fee_revenue_minor - self.venue_tax_paid_minor
        ):
            _fail("venue net revenue does not reconcile")
        if self.government_net_revenue_minor != (
            self.government_tax_revenue_minor - self.government_transfers_minor
        ):
            _fail("government net revenue does not reconcile")
        if (
            not isinstance(self.cohort_flows, tuple)
            or not self.cohort_flows
            or not all(isinstance(item, CohortLedgerFlow) for item in self.cohort_flows)
        ):
            _fail("cohort_flows must be a non-empty immutable tuple")
        cohort_ids = tuple(item.cohort_id for item in self.cohort_flows)
        if len(set(cohort_ids)) != len(cohort_ids):
            _fail("cohort ledger summary IDs must be unique")
        if self.synthetic_only is not True:
            _fail("institutional ledger summaries are currently synthetic-only")

    def to_mapping(self) -> dict[str, object]:
        return {
            "closing_total_cash_minor": self.closing_total_cash_minor,
            "cohort_flows": [flow.to_mapping() for flow in self.cohort_flows],
            "government_net_revenue_minor": self.government_net_revenue_minor,
            "government_tax_revenue_minor": self.government_tax_revenue_minor,
            "government_transfers_minor": self.government_transfers_minor,
            "ledger_sha256": self.ledger_sha256,
            "opening_total_cash_minor": self.opening_total_cash_minor,
            "synthetic_only": self.synthetic_only,
            "venue_fee_revenue_minor": self.venue_fee_revenue_minor,
            "venue_net_revenue_minor": self.venue_net_revenue_minor,
            "venue_tax_paid_minor": self.venue_tax_paid_minor,
        }


def summarize_institutional_ledger(
    ledger: InstitutionalLedger,
) -> InstitutionalLedgerSummary:
    """Summarize an already reconciled ledger without netting across institutions."""
    if not isinstance(ledger, InstitutionalLedger):
        _fail("institutional summary requires a validated InstitutionalLedger")
    accounts = {account.account_id: account for account in ledger.accounts}
    venue_fees = sum(
        entry.amount_minor
        for entry in ledger.entries
        if entry.flow_kind is LedgerFlowKind.VENUE_FEE
    )
    government_taxes = sum(
        entry.amount_minor
        for entry in ledger.entries
        if entry.flow_kind is LedgerFlowKind.GOVERNMENT_TAX
    )
    venue_taxes = sum(
        entry.amount_minor
        for entry in ledger.entries
        if entry.flow_kind is LedgerFlowKind.GOVERNMENT_TAX
        and accounts[entry.payer_account_id].kind is LedgerPartyKind.VENUE
    )
    transfers = sum(
        entry.amount_minor
        for entry in ledger.entries
        if entry.flow_kind is LedgerFlowKind.GOVERNMENT_TRANSFER
    )
    cohort_accounts: dict[tuple[str, CohortBasis, str], set[str]] = {}
    for account in ledger.accounts:
        if account.kind is LedgerPartyKind.AGENT and account.cohort is not None:
            key = (
                account.cohort.cohort_id,
                account.cohort.basis,
                account.cohort.definition_sha256,
            )
            cohort_accounts.setdefault(key, set()).add(account.account_id)
    cohort_flows: list[CohortLedgerFlow] = []
    for key, member_ids in sorted(cohort_accounts.items(), key=lambda item: item[0][0]):
        cohort_id, basis, definition_sha256 = key
        cohort_flows.append(
            CohortLedgerFlow(
                cohort_id=cohort_id,
                cohort_basis=basis,
                cohort_definition_sha256=definition_sha256,
                fees_paid_minor=sum(
                    entry.amount_minor
                    for entry in ledger.entries
                    if entry.flow_kind is LedgerFlowKind.VENUE_FEE
                    and entry.payer_account_id in member_ids
                ),
                taxes_paid_minor=sum(
                    entry.amount_minor
                    for entry in ledger.entries
                    if entry.flow_kind is LedgerFlowKind.GOVERNMENT_TAX
                    and entry.payer_account_id in member_ids
                ),
                transfers_received_minor=sum(
                    entry.amount_minor
                    for entry in ledger.entries
                    if entry.flow_kind is LedgerFlowKind.GOVERNMENT_TRANSFER
                    and entry.recipient_account_id in member_ids
                ),
            )
        )
    opening = sum(account.opening_balance_minor for account in ledger.accounts)
    closing = sum(account.closing_balance_minor for account in ledger.accounts)
    return InstitutionalLedgerSummary(
        ledger_sha256=ledger.sha256,
        opening_total_cash_minor=opening,
        closing_total_cash_minor=closing,
        venue_fee_revenue_minor=venue_fees,
        venue_tax_paid_minor=venue_taxes,
        venue_net_revenue_minor=venue_fees - venue_taxes,
        government_tax_revenue_minor=government_taxes,
        government_transfers_minor=transfers,
        government_net_revenue_minor=government_taxes - transfers,
        cohort_flows=tuple(cohort_flows),
    )


@dataclass(frozen=True, slots=True)
class PairedParticipationOutcome:
    """Paired horizon participation with one shared pre-policy active state."""

    agent_id: str
    cohort: FrozenCohort
    initially_active: bool
    baseline_active_at_horizon: bool
    candidate_active_at_horizon: bool

    def __post_init__(self) -> None:
        _identifier(self.agent_id, "participation agent_id")
        if not isinstance(self.cohort, FrozenCohort):
            _fail("participation cohort must be a FrozenCohort")
        if self.cohort.basis is CohortBasis.POST_TREATMENT_DESCRIPTIVE:
            _fail("participation cohorts must be defined before treatment")
        for field in (
            "initially_active",
            "baseline_active_at_horizon",
            "candidate_active_at_horizon",
        ):
            if not isinstance(getattr(self, field), bool):
                _fail(f"{field} must be a boolean")

    @property
    def candidate_exit(self) -> bool:
        """Whether an initially active agent is inactive in the candidate world."""
        return self.initially_active and not self.candidate_active_at_horizon

    @property
    def baseline_exit(self) -> bool:
        """Whether an initially active agent is inactive in the baseline world."""
        return self.initially_active and not self.baseline_active_at_horizon

    @property
    def policy_crowd_out(self) -> bool:
        """Whether pairing shows baseline activity but candidate inactivity."""
        return self.baseline_active_at_horizon and not self.candidate_active_at_horizon

    @property
    def policy_induced_exit(self) -> bool:
        """Whether policy crowd-out occurs for an initially active agent."""
        return self.initially_active and self.policy_crowd_out

    def to_mapping(self) -> dict[str, object]:
        """Return all paired states and the complete cohort commitment."""
        return {
            "agent_id": self.agent_id,
            "baseline_active_at_horizon": self.baseline_active_at_horizon,
            "candidate_active_at_horizon": self.candidate_active_at_horizon,
            "cohort": self.cohort.to_mapping(),
            "initially_active": self.initially_active,
        }


@dataclass(frozen=True, slots=True)
class PairedParticipationInput:
    """Frozen horizon and provenance for paired participation outcomes."""

    source_artifact_sha256: str
    study_protocol_sha256: str
    horizon_id: str
    outcomes: tuple[PairedParticipationOutcome, ...]
    synthetic_only: bool = True

    def __post_init__(self) -> None:
        _sha256(self.source_artifact_sha256, "participation source_artifact_sha256")
        _sha256(self.study_protocol_sha256, "participation study_protocol_sha256")
        _identifier(self.horizon_id, "participation horizon_id")
        if (
            not isinstance(self.outcomes, tuple)
            or not self.outcomes
            or not all(
                isinstance(item, PairedParticipationOutcome) for item in self.outcomes
            )
        ):
            _fail("participation outcomes must be a non-empty immutable tuple")
        agent_ids = tuple(item.agent_id for item in self.outcomes)
        if len(set(agent_ids)) != len(agent_ids):
            _fail("participation agent IDs must be unique")
        if self.synthetic_only is not True:
            _fail("real participation inputs require a typed temporal-firewall receipt")

    def to_mapping(self) -> dict[str, object]:
        return {
            "horizon_id": self.horizon_id,
            "outcomes": [item.to_mapping() for item in self.outcomes],
            "source_artifact_sha256": self.source_artifact_sha256,
            "study_protocol_sha256": self.study_protocol_sha256,
            "synthetic_only": self.synthetic_only,
        }

    @property
    def sha256(self) -> str:
        """Commit the horizon, provenance, and every paired state."""
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))


@dataclass(frozen=True, slots=True)
class ParticipationMetrics:
    """Separate descriptive exits from paired policy crowd-out."""

    cohort_id: str
    cohort_basis: CohortBasis
    cohort_definition_sha256: str
    agent_count: int
    initially_active_count: int
    baseline_active_at_horizon_count: int
    baseline_exit_count: int
    candidate_exit_count: int
    policy_crowd_out_count: int
    policy_induced_exit_count: int
    baseline_exit_probability: float
    candidate_exit_probability: float
    policy_crowd_out_probability: float
    policy_induced_exit_probability: float
    input_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.cohort_id, "participation cohort_id")
        if not isinstance(self.cohort_basis, CohortBasis):
            _fail("participation cohort_basis must be a CohortBasis")
        if self.cohort_basis is CohortBasis.POST_TREATMENT_DESCRIPTIVE:
            _fail("participation cohort_basis must be defined before treatment")
        _sha256(self.cohort_definition_sha256, "cohort_definition_sha256")
        _sha256(self.input_sha256, "participation input_sha256")
        _integer(self.agent_count, "agent_count", minimum=1)
        for field in (
            "initially_active_count",
            "baseline_active_at_horizon_count",
            "baseline_exit_count",
            "candidate_exit_count",
            "policy_crowd_out_count",
            "policy_induced_exit_count",
        ):
            _integer(getattr(self, field), field, minimum=0)
        if not 0 < self.initially_active_count <= self.agent_count:
            _fail("initially_active_count must lie in [1, agent_count]")
        if not 0 < self.baseline_active_at_horizon_count <= self.agent_count:
            _fail("baseline active count must lie in [1, agent_count]")
        if self.baseline_exit_count > self.initially_active_count:
            _fail("baseline exits exceed the initially active denominator")
        if self.candidate_exit_count > self.initially_active_count:
            _fail("candidate exits exceed the initially active denominator")
        if self.policy_crowd_out_count > self.baseline_active_at_horizon_count:
            _fail("policy crowd-out exceeds the baseline-active denominator")
        if self.policy_induced_exit_count > min(
            self.initially_active_count, self.policy_crowd_out_count
        ):
            _fail("policy-induced exits exceed their defining counts")
        probability_checks = (
            (
                "baseline_exit_probability",
                self.baseline_exit_count / self.initially_active_count,
            ),
            (
                "candidate_exit_probability",
                self.candidate_exit_count / self.initially_active_count,
            ),
            (
                "policy_crowd_out_probability",
                self.policy_crowd_out_count / self.baseline_active_at_horizon_count,
            ),
            (
                "policy_induced_exit_probability",
                self.policy_induced_exit_count / self.initially_active_count,
            ),
        )
        for field, expected in probability_checks:
            value = _finite(getattr(self, field), field)
            if not 0.0 <= value <= 1.0 or value != expected:
                _fail(f"{field} does not match its registered count and denominator")

    def to_mapping(self) -> dict[str, object]:
        return {
            "agent_count": self.agent_count,
            "baseline_active_at_horizon_count": (self.baseline_active_at_horizon_count),
            "baseline_exit_count": self.baseline_exit_count,
            "baseline_exit_probability": self.baseline_exit_probability,
            "candidate_exit_count": self.candidate_exit_count,
            "candidate_exit_probability": self.candidate_exit_probability,
            "cohort_basis": self.cohort_basis.value,
            "cohort_definition_sha256": self.cohort_definition_sha256,
            "cohort_id": self.cohort_id,
            "initially_active_count": self.initially_active_count,
            "input_sha256": self.input_sha256,
            "policy_crowd_out_count": self.policy_crowd_out_count,
            "policy_crowd_out_probability": self.policy_crowd_out_probability,
            "policy_induced_exit_count": self.policy_induced_exit_count,
            "policy_induced_exit_probability": self.policy_induced_exit_probability,
        }


def evaluate_paired_participation(
    data: PairedParticipationInput,
) -> tuple[ParticipationMetrics, ...]:
    """Evaluate all frozen cohorts and preserve exit/crowd-out distinctions."""
    if not isinstance(data, PairedParticipationInput):
        _fail("participation evaluation requires a PairedParticipationInput")
    outcomes = data.outcomes
    input_sha256 = data.sha256
    results: list[ParticipationMetrics] = []
    for cohort_id in sorted({item.cohort.cohort_id for item in outcomes}):
        members = tuple(item for item in outcomes if item.cohort.cohort_id == cohort_id)
        commitments = {
            (item.cohort.basis, item.cohort.definition_sha256) for item in members
        }
        if len(commitments) != 1:
            _fail(f"cohort {cohort_id!r} has conflicting definitions")
        basis, definition_sha256 = next(iter(commitments))
        initially_active = sum(item.initially_active for item in members)
        baseline_active = sum(item.baseline_active_at_horizon for item in members)
        if initially_active == 0:
            _fail(f"cohort {cohort_id!r} has no initially active denominator")
        if baseline_active == 0:
            _fail(f"cohort {cohort_id!r} has no baseline-active crowd-out denominator")
        baseline_exits = sum(item.baseline_exit for item in members)
        candidate_exits = sum(item.candidate_exit for item in members)
        crowd_outs = sum(item.policy_crowd_out for item in members)
        induced_exits = sum(item.policy_induced_exit for item in members)
        results.append(
            ParticipationMetrics(
                cohort_id=cohort_id,
                cohort_basis=basis,
                cohort_definition_sha256=definition_sha256,
                agent_count=len(members),
                initially_active_count=initially_active,
                baseline_active_at_horizon_count=baseline_active,
                baseline_exit_count=baseline_exits,
                candidate_exit_count=candidate_exits,
                policy_crowd_out_count=crowd_outs,
                policy_induced_exit_count=induced_exits,
                baseline_exit_probability=baseline_exits / initially_active,
                candidate_exit_probability=candidate_exits / initially_active,
                policy_crowd_out_probability=crowd_outs / baseline_active,
                policy_induced_exit_probability=induced_exits / initially_active,
                input_sha256=input_sha256,
            )
        )
    return tuple(results)


class ObjectiveDirection(StrEnum):
    """Whether a larger or smaller raw objective is preferred."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


@dataclass(frozen=True, slots=True)
class ObjectiveDefinition:
    """Frozen unit and normalization bounds for one policy objective."""

    objective_id: str
    direction: ObjectiveDirection
    unit: str
    lower_bound: float
    upper_bound: float

    def __post_init__(self) -> None:
        _identifier(self.objective_id, "objective_id")
        if not isinstance(self.direction, ObjectiveDirection):
            _fail("objective direction must be an ObjectiveDirection")
        _identifier(self.unit, "objective unit")
        object.__setattr__(
            self, "lower_bound", _finite(self.lower_bound, "lower_bound")
        )
        object.__setattr__(
            self, "upper_bound", _finite(self.upper_bound, "upper_bound")
        )
        if self.upper_bound <= self.lower_bound:
            _fail("objective upper_bound must exceed lower_bound")

    def normalized_preference(self, value: float) -> float:
        """Map an in-bound raw value to [0, 1] in the preferred direction."""
        value = _finite(value, f"{self.objective_id} value")
        if not self.lower_bound <= value <= self.upper_bound:
            _fail(f"objective {self.objective_id!r} lies outside frozen bounds")
        ascending = (value - self.lower_bound) / (self.upper_bound - self.lower_bound)
        return (
            ascending
            if self.direction is ObjectiveDirection.MAXIMIZE
            else 1.0 - ascending
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "direction": self.direction.value,
            "lower_bound": self.lower_bound,
            "objective_id": self.objective_id,
            "unit": self.unit,
            "upper_bound": self.upper_bound,
        }


@dataclass(frozen=True, slots=True)
class ObjectiveValue:
    """One raw policy-objective value; no implicit normalization."""

    objective_id: str
    value: float

    def __post_init__(self) -> None:
        _identifier(self.objective_id, "objective value objective_id")
        object.__setattr__(self, "value", _finite(self.value, "objective value"))

    def to_mapping(self) -> dict[str, object]:
        return {"objective_id": self.objective_id, "value": self.value}


@dataclass(frozen=True, slots=True)
class PolicyObjectiveVector:
    """Complete raw objective vector from one versioned synthetic artifact."""

    policy_id: str
    source_artifact_sha256: str
    study_protocol_sha256: str
    values: tuple[ObjectiveValue, ...]
    synthetic_only: bool = True

    def __post_init__(self) -> None:
        _identifier(self.policy_id, "policy objective policy_id")
        _sha256(self.source_artifact_sha256, "source_artifact_sha256")
        _sha256(self.study_protocol_sha256, "study_protocol_sha256")
        if (
            not isinstance(self.values, tuple)
            or not self.values
            or not all(isinstance(item, ObjectiveValue) for item in self.values)
        ):
            _fail("policy objective values must be a non-empty immutable tuple")
        ids = tuple(item.objective_id for item in self.values)
        if len(set(ids)) != len(ids):
            _fail("policy objective IDs must be unique")
        if self.synthetic_only is not True:
            _fail("real policy objective vectors require a typed verifier receipt")

    def to_mapping(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "source_artifact_sha256": self.source_artifact_sha256,
            "study_protocol_sha256": self.study_protocol_sha256,
            "synthetic_only": self.synthetic_only,
            "values": [item.to_mapping() for item in self.values],
        }


@dataclass(frozen=True, slots=True)
class ObjectiveWeight:
    """One non-negative weight in a frozen preference profile."""

    objective_id: str
    weight: float

    def __post_init__(self) -> None:
        _identifier(self.objective_id, "weight objective_id")
        weight = _finite(self.weight, "objective weight")
        if weight < 0.0:
            _fail("objective weight must be non-negative")
        object.__setattr__(self, "weight", weight)

    def to_mapping(self) -> dict[str, object]:
        return {"objective_id": self.objective_id, "weight": self.weight}


@dataclass(frozen=True, slots=True)
class FrozenPreferenceProfile:
    """Pre-outcome preference weights used only for ranking sensitivity."""

    profile_id: str
    study_protocol_sha256: str
    weights: tuple[ObjectiveWeight, ...]
    frozen_at: datetime
    target_outcomes_available_at: datetime
    synthetic_only: bool = True

    def __post_init__(self) -> None:
        _identifier(self.profile_id, "preference profile_id")
        _sha256(self.study_protocol_sha256, "study_protocol_sha256")
        if (
            not isinstance(self.weights, tuple)
            or not self.weights
            or not all(isinstance(item, ObjectiveWeight) for item in self.weights)
        ):
            _fail("preference weights must be a non-empty immutable tuple")
        ids = tuple(item.objective_id for item in self.weights)
        if len(set(ids)) != len(ids):
            _fail("preference objective IDs must be unique")
        total = math.fsum(item.weight for item in self.weights)
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
            _fail("preference weights must sum to one")
        frozen = _aware(self.frozen_at, "preference frozen_at")
        available = _aware(
            self.target_outcomes_available_at,
            "target_outcomes_available_at",
        )
        if frozen >= available:
            _fail("preference weights must be frozen before target outcomes")
        object.__setattr__(self, "frozen_at", frozen)
        object.__setattr__(self, "target_outcomes_available_at", available)
        if self.synthetic_only is not True:
            _fail("real preference profiles require external pre-reveal evidence")

    def to_mapping(self) -> dict[str, object]:
        return {
            "frozen_at": self.frozen_at.isoformat(),
            "profile_id": self.profile_id,
            "study_protocol_sha256": self.study_protocol_sha256,
            "synthetic_only": self.synthetic_only,
            "target_outcomes_available_at": (
                self.target_outcomes_available_at.isoformat()
            ),
            "weights": [item.to_mapping() for item in self.weights],
        }


@dataclass(frozen=True, slots=True)
class PolicyPreferenceScore:
    """A preference-specific normalized score; explicitly not social welfare."""

    policy_id: str
    preference_score: float
    rank: int

    def __post_init__(self) -> None:
        _identifier(self.policy_id, "preference score policy_id")
        score = _finite(self.preference_score, "preference_score")
        if not 0.0 <= score <= 1.0:
            _fail("preference_score must lie in [0, 1]")
        _integer(self.rank, "preference rank", minimum=1)

    def to_mapping(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "preference_score": self.preference_score,
            "rank": self.rank,
        }


@dataclass(frozen=True, slots=True)
class PreferenceRanking:
    """Complete ranking under one frozen profile."""

    profile_id: str
    scores: tuple[PolicyPreferenceScore, ...]

    def __post_init__(self) -> None:
        _identifier(self.profile_id, "ranking profile_id")
        if (
            not isinstance(self.scores, tuple)
            or not self.scores
            or not all(isinstance(item, PolicyPreferenceScore) for item in self.scores)
        ):
            _fail("ranking scores must be a non-empty immutable tuple")
        policy_ids = tuple(item.policy_id for item in self.scores)
        if len(set(policy_ids)) != len(policy_ids):
            _fail("ranking policy IDs must be unique")
        if self.scores[0].rank != 1:
            _fail("the first policy in a ranking must have rank one")
        for position, (earlier, later) in enumerate(
            zip(self.scores, self.scores[1:], strict=False), start=2
        ):
            if later.preference_score > earlier.preference_score:
                _fail("ranking scores must be ordered from highest to lowest")
            expected_rank = (
                earlier.rank
                if later.preference_score == earlier.preference_score
                else position
            )
            if later.rank != expected_rank:
                _fail("ranking ranks do not match score order and exact ties")

    def to_mapping(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "scores": [score.to_mapping() for score in self.scores],
        }


@dataclass(frozen=True, slots=True)
class PolicyRankSensitivity:
    """Rank range and top-rank frequency across all frozen preferences."""

    policy_id: str
    minimum_rank: int
    maximum_rank: int
    top_rank_count: int
    profile_count: int

    def __post_init__(self) -> None:
        _identifier(self.policy_id, "rank sensitivity policy_id")
        for field in (
            "minimum_rank",
            "maximum_rank",
            "profile_count",
        ):
            _integer(getattr(self, field), field, minimum=1)
        _integer(self.top_rank_count, "top_rank_count", minimum=0)
        if self.maximum_rank < self.minimum_rank:
            _fail("maximum_rank must not be below minimum_rank")
        if self.top_rank_count > self.profile_count:
            _fail("top_rank_count must not exceed profile_count")

    def to_mapping(self) -> dict[str, object]:
        return {
            "maximum_rank": self.maximum_rank,
            "minimum_rank": self.minimum_rank,
            "policy_id": self.policy_id,
            "profile_count": self.profile_count,
            "top_rank_count": self.top_rank_count,
        }


@dataclass(frozen=True, slots=True)
class MultiObjectiveEvaluation:
    """Pareto and frozen-preference results retaining every raw objective."""

    objective_definition_sha256: str
    input_sha256: str
    objective_definitions: tuple[ObjectiveDefinition, ...]
    objective_vectors: tuple[PolicyObjectiveVector, ...]
    preference_profiles: tuple[FrozenPreferenceProfile, ...]
    pareto_frontier_policy_ids: tuple[str, ...]
    rankings: tuple[PreferenceRanking, ...]
    rank_sensitivity: tuple[PolicyRankSensitivity, ...]
    ranking_order_stable: bool
    synthetic_only: bool = True

    def __post_init__(self) -> None:
        _sha256(
            self.objective_definition_sha256,
            "objective_definition_sha256",
        )
        _sha256(self.input_sha256, "multi-objective input_sha256")
        typed_fields: tuple[tuple[str, type[object]], ...] = (
            ("objective_definitions", ObjectiveDefinition),
            ("objective_vectors", PolicyObjectiveVector),
            ("preference_profiles", FrozenPreferenceProfile),
            ("rankings", PreferenceRanking),
            ("rank_sensitivity", PolicyRankSensitivity),
        )
        for field, expected in typed_fields:
            value = getattr(self, field)
            if (
                not isinstance(value, tuple)
                or not value
                or not all(isinstance(item, expected) for item in value)
            ):
                _fail(f"{field} must be a non-empty immutable typed tuple")
        definition_ids = tuple(item.objective_id for item in self.objective_definitions)
        policy_tuple = tuple(item.policy_id for item in self.objective_vectors)
        profile_tuple = tuple(item.profile_id for item in self.preference_profiles)
        if len(set(definition_ids)) != len(definition_ids):
            _fail("multi-objective result has duplicate objective definitions")
        if len(set(policy_tuple)) != len(policy_tuple):
            _fail("multi-objective result has duplicate policies")
        if len(set(profile_tuple)) != len(profile_tuple):
            _fail("multi-objective result has duplicate preference profiles")
        if len(self.objective_definitions) < 2 or len(self.objective_vectors) < 2:
            _fail(
                "multi-objective result requires at least two objectives and policies"
            )
        protocols = {item.study_protocol_sha256 for item in self.objective_vectors} | {
            item.study_protocol_sha256 for item in self.preference_profiles
        }
        if len(protocols) != 1:
            _fail("retained policies and preferences must share one protocol")
        availability_times = {
            item.target_outcomes_available_at for item in self.preference_profiles
        }
        if len(availability_times) != 1:
            _fail("retained preferences must target one outcome availability time")
        weight_signatures = {
            tuple(
                sorted((weight.objective_id, weight.weight) for weight in item.weights)
            )
            for item in self.preference_profiles
        }
        if len(weight_signatures) != len(self.preference_profiles):
            _fail("retained preference profiles must have distinct weight vectors")
        definition_mapping = [item.to_mapping() for item in self.objective_definitions]
        expected_definition_hash = sha256_bytes(
            canonical_json_bytes(definition_mapping)
        )
        if self.objective_definition_sha256 != expected_definition_hash:
            _fail("objective_definition_sha256 does not match retained definitions")
        input_mapping = {
            "definitions": definition_mapping,
            "policies": [item.to_mapping() for item in self.objective_vectors],
            "preferences": [item.to_mapping() for item in self.preference_profiles],
        }
        if self.input_sha256 != sha256_bytes(canonical_json_bytes(input_mapping)):
            _fail("multi-objective input_sha256 does not match retained inputs")
        if (
            not isinstance(self.pareto_frontier_policy_ids, tuple)
            or not self.pareto_frontier_policy_ids
        ):
            _fail("Pareto frontier must be a non-empty immutable tuple")
        for policy_id in self.pareto_frontier_policy_ids:
            _identifier(policy_id, "Pareto frontier policy_id")
        policy_ids = set(policy_tuple)
        if not set(self.pareto_frontier_policy_ids) <= policy_ids:
            _fail("Pareto frontier contains an unregistered policy")
        if tuple(item.profile_id for item in self.rankings) != profile_tuple:
            _fail("rankings must cover every frozen preference profile")
        if {item.policy_id for item in self.rank_sensitivity} != policy_ids:
            _fail("rank sensitivity must cover every policy")
        definitions_by_id = {
            item.objective_id: item for item in self.objective_definitions
        }
        expected_objective_ids = set(definition_ids)
        policy_values: dict[str, dict[str, float]] = {}
        for vector in self.objective_vectors:
            values = {item.objective_id: item.value for item in vector.values}
            if set(values) != expected_objective_ids:
                _fail("retained policy vector has an incomplete objective set")
            policy_values[vector.policy_id] = values
        expected_frontier = tuple(
            sorted(
                policy_id
                for policy_id in policy_tuple
                if not any(
                    other_id != policy_id
                    and _dominates(
                        policy_values[other_id],
                        policy_values[policy_id],
                        definitions_by_id,
                    )
                    for other_id in policy_tuple
                )
            )
        )
        if self.pareto_frontier_policy_ids != expected_frontier:
            _fail("Pareto frontier does not match retained objective vectors")
        for profile, ranking in zip(
            self.preference_profiles, self.rankings, strict=True
        ):
            weights = {item.objective_id: item.weight for item in profile.weights}
            if set(weights) != expected_objective_ids:
                _fail("retained preference profile has an incomplete objective set")
            expected_scores = {
                policy_id: math.fsum(
                    weights[objective_id]
                    * definitions_by_id[objective_id].normalized_preference(value)
                    for objective_id, value in policy_values[policy_id].items()
                )
                for policy_id in policy_tuple
            }
            expected_order = tuple(
                sorted(policy_tuple, key=lambda item: (-expected_scores[item], item))
            )
            if tuple(item.policy_id for item in ranking.scores) != expected_order:
                _fail("preference ranking order does not match retained inputs")
            if any(
                score.preference_score != expected_scores[score.policy_id]
                for score in ranking.scores
            ):
                _fail("preference score does not match retained inputs")
        sensitivity_by_policy = {item.policy_id: item for item in self.rank_sensitivity}
        for policy_id in policy_tuple:
            ranks = tuple(
                score.rank
                for ranking in self.rankings
                for score in ranking.scores
                if score.policy_id == policy_id
            )
            sensitivity = sensitivity_by_policy[policy_id]
            if (
                sensitivity.minimum_rank != min(ranks)
                or sensitivity.maximum_rank != max(ranks)
                or sensitivity.top_rank_count != sum(rank == 1 for rank in ranks)
                or sensitivity.profile_count != len(self.rankings)
            ):
                _fail("policy rank sensitivity does not match retained rankings")
        orders = tuple(
            tuple(score.policy_id for score in ranking.scores)
            for ranking in self.rankings
        )
        if not isinstance(self.ranking_order_stable, bool):
            _fail("ranking_order_stable must be a boolean")
        if self.ranking_order_stable != (len(set(orders)) == 1):
            _fail("ranking_order_stable does not match retained rankings")
        if self.synthetic_only is not True:
            _fail("multi-objective evaluation is currently synthetic-only")

    def to_mapping(self) -> dict[str, object]:
        return {
            "input_sha256": self.input_sha256,
            "objective_definition_sha256": self.objective_definition_sha256,
            "objective_definitions": [
                item.to_mapping() for item in self.objective_definitions
            ],
            "objective_vectors": [item.to_mapping() for item in self.objective_vectors],
            "pareto_frontier_policy_ids": list(self.pareto_frontier_policy_ids),
            "preference_profiles": [
                item.to_mapping() for item in self.preference_profiles
            ],
            "rank_sensitivity": [item.to_mapping() for item in self.rank_sensitivity],
            "ranking_order_stable": self.ranking_order_stable,
            "rankings": [item.to_mapping() for item in self.rankings],
            "synthetic_only": self.synthetic_only,
        }


def _dominates(
    left: dict[str, float],
    right: dict[str, float],
    definitions: dict[str, ObjectiveDefinition],
) -> bool:
    weakly_better = True
    strictly_better = False
    for objective_id, definition in definitions.items():
        left_value = left[objective_id]
        right_value = right[objective_id]
        if definition.direction is ObjectiveDirection.MAXIMIZE:
            weakly_better = weakly_better and left_value >= right_value
            strictly_better = strictly_better or left_value > right_value
        else:
            weakly_better = weakly_better and left_value <= right_value
            strictly_better = strictly_better or left_value < right_value
    return weakly_better and strictly_better


def evaluate_multi_objective_policies(
    definitions: tuple[ObjectiveDefinition, ...],
    policies: tuple[PolicyObjectiveVector, ...],
    preferences: tuple[FrozenPreferenceProfile, ...],
) -> MultiObjectiveEvaluation:
    """Compute an unweighted Pareto frontier and frozen-weight sensitivity."""
    if (
        not isinstance(definitions, tuple)
        or len(definitions) < 2
        or not all(isinstance(item, ObjectiveDefinition) for item in definitions)
    ):
        _fail("multi-objective evaluation requires at least two definitions")
    if (
        not isinstance(policies, tuple)
        or len(policies) < 2
        or not all(isinstance(item, PolicyObjectiveVector) for item in policies)
    ):
        _fail("multi-objective evaluation requires at least two policies")
    if (
        not isinstance(preferences, tuple)
        or not preferences
        or not all(isinstance(item, FrozenPreferenceProfile) for item in preferences)
    ):
        _fail("multi-objective evaluation requires frozen preference profiles")
    definition_ids = tuple(item.objective_id for item in definitions)
    policy_ids = tuple(item.policy_id for item in policies)
    profile_ids = tuple(item.profile_id for item in preferences)
    if len(set(definition_ids)) != len(definition_ids):
        _fail("objective definition IDs must be unique")
    if len(set(policy_ids)) != len(policy_ids):
        _fail("policy IDs must be unique")
    if len(set(profile_ids)) != len(profile_ids):
        _fail("preference profile IDs must be unique")
    availability_times = {
        profile.target_outcomes_available_at for profile in preferences
    }
    if len(availability_times) != 1:
        _fail("preference profiles must target the same outcome availability time")
    weight_signatures = {
        tuple(sorted((item.objective_id, item.weight) for item in profile.weights))
        for profile in preferences
    }
    if len(weight_signatures) != len(preferences):
        _fail("preference profiles must not duplicate a weight vector")
    expected_ids = set(definition_ids)
    protocol_hashes = {policy.study_protocol_sha256 for policy in policies} | {
        profile.study_protocol_sha256 for profile in preferences
    }
    if len(protocol_hashes) != 1:
        _fail("policies and preferences must share one study protocol")
    definitions_by_id = {item.objective_id: item for item in definitions}
    policy_values: dict[str, dict[str, float]] = {}
    for policy in policies:
        values = {item.objective_id: item.value for item in policy.values}
        if set(values) != expected_ids:
            _fail(f"policy {policy.policy_id!r} does not have the exact objective set")
        for objective_id, value in values.items():
            definitions_by_id[objective_id].normalized_preference(value)
        policy_values[policy.policy_id] = values
    for profile in preferences:
        if {item.objective_id for item in profile.weights} != expected_ids:
            _fail(f"profile {profile.profile_id!r} does not weight every objective")

    frontier = tuple(
        sorted(
            policy_id
            for policy_id in policy_ids
            if not any(
                other_id != policy_id
                and _dominates(
                    policy_values[other_id],
                    policy_values[policy_id],
                    definitions_by_id,
                )
                for other_id in policy_ids
            )
        )
    )

    rankings: list[PreferenceRanking] = []
    for profile in preferences:
        weights = {item.objective_id: item.weight for item in profile.weights}
        raw_scores = tuple(
            (
                policy.policy_id,
                math.fsum(
                    weights[objective_id]
                    * definitions_by_id[objective_id].normalized_preference(value)
                    for objective_id, value in policy_values[policy.policy_id].items()
                ),
            )
            for policy in policies
        )
        ordered = sorted(raw_scores, key=lambda item: (-item[1], item[0]))
        score_rows: list[PolicyPreferenceScore] = []
        previous_score: float | None = None
        current_rank = 0
        for position, (policy_id, score) in enumerate(ordered, start=1):
            if previous_score is None or score != previous_score:
                current_rank = position
            score_rows.append(PolicyPreferenceScore(policy_id, score, current_rank))
            previous_score = score
        rankings.append(PreferenceRanking(profile.profile_id, tuple(score_rows)))

    sensitivity: list[PolicyRankSensitivity] = []
    for policy_id in sorted(policy_ids):
        ranks = tuple(
            score.rank
            for ranking in rankings
            for score in ranking.scores
            if score.policy_id == policy_id
        )
        sensitivity.append(
            PolicyRankSensitivity(
                policy_id=policy_id,
                minimum_rank=min(ranks),
                maximum_rank=max(ranks),
                top_rank_count=sum(rank == 1 for rank in ranks),
                profile_count=len(rankings),
            )
        )
    orders = tuple(
        tuple(score.policy_id for score in ranking.scores) for ranking in rankings
    )
    definition_mapping = [item.to_mapping() for item in definitions]
    input_mapping = {
        "definitions": definition_mapping,
        "policies": [item.to_mapping() for item in policies],
        "preferences": [item.to_mapping() for item in preferences],
    }
    return MultiObjectiveEvaluation(
        objective_definition_sha256=sha256_bytes(
            canonical_json_bytes(definition_mapping)
        ),
        input_sha256=sha256_bytes(canonical_json_bytes(input_mapping)),
        objective_definitions=definitions,
        objective_vectors=policies,
        preference_profiles=preferences,
        pareto_frontier_policy_ids=frontier,
        rankings=tuple(rankings),
        rank_sensitivity=tuple(sensitivity),
        ranking_order_stable=len(set(orders)) == 1,
    )
