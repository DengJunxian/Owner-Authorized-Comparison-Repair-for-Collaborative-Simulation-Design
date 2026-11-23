"""Exact finite integrated policy-functional calibration canary.

This module closes a deliberately small *synthetic* probability model.  A
uniform finite prior is combined with a finite, uniformly weighted support of
pre-policy observations.  Bayes' rule, posterior sampling, randomized-tie
ranks, conditional rank laws, and the unconditional rank law are all derived
with :class:`fractions.Fraction`; no Monte Carlo approximation enters those
calculations.

The policy functional is not a toy arithmetic value.  Every lookup-table cell
is obtained from the public protocol-bound integrated runner and then from the
verifier-derived exact ledger.  The finite posterior is nevertheless separate
from PolicyTwin's ABC posterior.  Passing this canary proves only exact
arithmetic and exchangeability for this closed finite synthetic model,
conditional on the executed integrated lookup table.  It does not validate
ABC, a continuous posterior, market mechanisms, real data, real causal claims,
external preregistration, or publication readiness.
"""

from __future__ import annotations

import math
import os
import re
import secrets
import stat
import sys
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from fractions import Fraction
from functools import lru_cache
from io import StringIO
from itertools import permutations, product
from pathlib import Path
from tempfile import TemporaryFile
from threading import RLock, get_ident
from typing import Literal, NoReturn, Self, TextIO, cast
from weakref import WeakValueDictionary, finalize

import policytwin.counterfactual.internal_randomness_audit as internal_rng_audit_module
import policytwin.evaluation.ledger_derived as ledger_derived_module
import policytwin.experiments.integrated_synthetic as integrated_synthetic_module
from policytwin.calibration.integrated import IntegratedSyntheticCalibrationAdapter
from policytwin.counterfactual.events import (
    SCENARIO_GENERATION_DRAW_SCHEMA_VERSION,
    SCENARIO_GENERATION_RECEIPT_SCHEMA_VERSION,
)
from policytwin.counterfactual.internal_randomness_audit import (
    InternalRandomCouplingAudit,
    require_verifier_owned_internal_random_coupling_audit,
)
from policytwin.counterfactual.paired import WorldRole
from policytwin.counterfactual.randomness import EventKey
from policytwin.evaluation.ledger_derived import (
    LEDGER_DERIVED_SCHEMA_VERSION,
    DifferenceScope,
    ExactMetricState,
    ExactRational,
    derive_protocol_bound_ledger_evidence,
    ledger_evidence_verifier_implementation_sha256,
)
from policytwin.evaluation.ledger_posterior import (
    LedgerBenefitDirection,
    LedgerPosteriorOutputSpec,
    UndefinedMetricPolicy,
    ledger_output_spec,
)
from policytwin.experiments.integrated_synthetic import (
    INTEGRATED_ESTIMAND_ID,
    INTEGRATED_METRIC_IDS,
    INTEGRATED_PAIR_INPUT_SCHEMA_VERSION,
    INTEGRATED_PROTOCOL_RUN_SCHEMA_VERSION,
    integrated_metric_definition_sha256,
    integrated_paired_estimand_definition_sha256,
    integrated_synthetic_forward_market_config,
    integrated_synthetic_pair,
    integrated_synthetic_pair_input_sha256,
    integrated_synthetic_shock_generator_implementation_sha256,
    run_protocol_bound_integrated_synthetic_pair,
)
from policytwin.protocol import (
    ClaimScope,
    CohortBasis,
    FailureThreshold,
    FrozenCohort,
    FrozenDefinition,
    OutputRole,
    ProtocolExecutionConsumer,
    ProtocolExecutionPurpose,
    RandomnessProtocol,
    RegisteredOutput,
    StudyMode,
    StudyProtocol,
    SyntheticProtocolExecutionIssuer,
)
from policytwin.provenance import (
    canonical_json_bytes,
    collect_git_metadata,
    read_json_bytes_object,
    sha256_bytes,
    sha256_file,
    write_json_file,
)

POLICY_FUNCTIONAL_EXACT_FINITE_SCHEMA_VERSION = (
    "policytwin.policy-functional-exact-finite.v1"
)
POLICY_FUNCTIONAL_EXACT_FINITE_CLAIM_SCOPE: Literal[
    "exact_closed_finite_synthetic_posterior_and_integrated_lookup_only"
] = "exact_closed_finite_synthetic_posterior_and_integrated_lookup_only"
POLICY_FUNCTIONAL_EXACT_FINITE_FUNCTIONAL: Literal[
    "mean_candidate_minus_baseline_terminal_all_latent_strategy_cash_level"
] = "mean_candidate_minus_baseline_terminal_all_latent_strategy_cash_level"
POLICY_FUNCTIONAL_EXACT_FINITE_SCENARIO_SELECTION_RULE: Literal[
    "lexicographically_first_candidate_with_distinct_state_functionals"
] = "lexicographically_first_candidate_with_distinct_state_functionals"
# Design-time-only bounded candidate list for the current, unexecuted s--x
# epoch.  A selected namespace may be frozen only by the one-time isolated
# public-runner audit.  The production canary never searches these candidates
# or chooses a namespace from its observed rank law.
POLICY_FUNCTIONAL_EXACT_FINITE_SCENARIO_CANDIDATE_SEEDS = (
    "exact-finite-policy-functional-grid-s-v1",
    "exact-finite-policy-functional-grid-t-v1",
    "exact-finite-policy-functional-grid-u-v1",
    "exact-finite-policy-functional-grid-v-v1",
    "exact-finite-policy-functional-grid-w-v1",
    "exact-finite-policy-functional-grid-x-v1",
)
_SCENARIO_AUDIT_EPOCH = "exact-finite-policy-functional-grid-s-x-v1"
_SELECTION_STATE_SCHEMA_VERSION = "policytwin.exact-finite-scenario-selection-state.v2"
_EXCLUDED_EXPOSED_PILOT_CANDIDATES = tuple(
    f"exact-finite-policy-functional-grid-{letter}-v1" for letter in "abcdef"
)
_EXCLUDED_STALE_AUDIT_CANDIDATES = tuple(
    f"exact-finite-policy-functional-grid-{letter}-v1" for letter in "ghijkl"
)
_EXCLUDED_SUPERSEDED_SOURCE_AUDIT_CANDIDATES = tuple(
    f"exact-finite-policy-functional-grid-{letter}-v1" for letter in "mnopqr"
)
_SUPERSEDED_M_R_FINAL_SOURCE_SHA256 = (
    "182aba10adde798c0585c1fa264d840ee2e15c61db97925674fc3fe5299279ff"
)
_SUPERSEDED_M_R_PENDING_SOURCE_SHA256 = (
    "feb36dfb16b90b575e7fe2724987b9cdcd35b9a153539a2f6bb4def6ec8948a1"
)
_SUPERSEDED_M_R_AUDITED_DESIGN_SHA256 = (
    "205c41d32695132567bd20baf12597a095b30e06846dfcf037779353200ae5af"
)
_SUPERSEDED_M_R_EXECUTION_RECEIPT_SHA256 = (
    "58c0f5e99203898fbfb247e755275a99ee3cf2e5b8433a4a1446bd5a3fc89cfe"
)
_SUPERSEDED_M_R_PLAN_SHA256 = (
    "e4848531f10ba73688f891188f49b56be47f36d36ceb8c4cd79712f0d3a57eed"
)
_SUPERSEDED_M_R_RESULT_SHA256 = (
    "b4429204099be9a4326c24974935a264629ee3d699925eae4375a394eca74c4c"
)
_SUPERSEDED_M_R_BUNDLE_ROOT_SHA256 = (
    "5666c7d544da644c8a1b907daa0260ea6766ef2d5cdc361e6f8d6f89018409f0"
)
_SCENARIO_AUDIT_CANARY_ID = "exact-finite-policy-functional-ci"
_SCENARIO_AUDIT_REPLICATE_COUNT = 4
_SCENARIO_AUDIT_STEP_COUNT = 3
_SCENARIO_AUDIT_CANDIDATE_ALLOW_BUY = False
_SCENARIO_AUDIT_CANDIDATE_ALLOW_SELL = True
_SCENARIO_AUDIT_CANDIDATE_IS_NOOP = False
# Issued only after the durable O_EXCL claim commits and consumed before the
# candidate loop begins.  A restart cannot recreate this process-local
# authority from the persisted claim hash, so a claimed epoch cannot be rerun.
_BLIND_AUDIT_ATTEMPT_CAPABILITIES: dict[int, tuple[object, str, int, int]] = {}
_BLIND_AUDIT_ACTIVATION_TICKETS: dict[int, tuple[object, str, int, int]] = {}
_ACTIVE_BLIND_AUDIT_EVALUATIONS: dict[int, tuple[object, str, int, int]] = {}
_BLIND_AUDIT_COMPLETION_CAPABILITIES: dict[
    int, tuple[object, str, str, tuple[str, ...], int, int]
] = {}
_BLIND_AUDIT_STDOUT_CAPABILITIES: dict[
    int, tuple[object, str, str, tuple[str, ...], str, int, int]
] = {}
_BLIND_AUDIT_CAPABILITY_LOCK = RLock()

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_EVENT_KEY = re.compile(r"^ptk1:[0-9a-f]{64}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_INTEGRATED_CELLS = 64
_MAX_INTEGRATED_CELL_STEPS = 256
_MAX_TIE_EXPANDED_ATOMS = 65_536
_MAX_PARAMETER_STATES = 8
_MAX_SUPPORT_SIZE = 64
_MAX_POSTERIOR_DRAWS = 8
_MAX_SCENARIOS = 32
_MAX_SAFE_INTEGER = 1 << 53


class ExactFinitePolicyFunctionalError(ValueError):
    """Raised when the exact finite canary contract fails closed."""


def _fail(message: str) -> NoReturn:
    raise ExactFinitePolicyFunctionalError(message)


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        _fail(f"{field} must be a stable ASCII identifier")
    return value


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail(f"{field} must be a lowercase SHA-256 digest")
    return value


def _event_key(value: object, field: str) -> str:
    if not isinstance(value, str) or _EVENT_KEY.fullmatch(value) is None:
        _fail(f"{field} must be a canonical EventKey commitment")
    return value


def _positive_int(value: object, field: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"{field} must be an integer of at least {minimum}")
    return value


def _aware(value: object, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        _fail(f"{field} must be timezone-aware")
    return value.astimezone(UTC)


def _fraction(value: ExactRational, field: str = "exact rational") -> Fraction:
    if type(value) is not ExactRational:
        raise TypeError(f"{field} must be an ExactRational")
    return Fraction(value.numerator, value.denominator)


def _exact(value: Fraction) -> ExactRational:
    return ExactRational.from_ratio(value.numerator, value.denominator)


def _uniform(size: int) -> Fraction:
    return Fraction(1, _positive_int(size, "uniform support size"))


def _digest(value: object) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _selection_state_path() -> Path:
    return Path(__file__).resolve().parents[1] / "exact_finite_scenario_selection.json"


def _selection_input_snapshot_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "exact_finite_scenario_selection_pending.json"
    )


def _blind_attempt_claim_path() -> Path:
    return (
        Path(__file__).resolve().parents[1] / "exact_finite_scenario_blind_attempt.json"
    )


def _superseded_m_r_final_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "exact_finite_scenario_selection_m_r_final.json"
    )


def _superseded_m_r_pending_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "exact_finite_scenario_selection_m_r_pending.json"
    )


def _read_stable_bounded_regular_bytes(
    path: Path,
    *,
    label: str,
    maximum_bytes: int = 65_536,
) -> bytes:
    """Read one regular file through no-follow descriptors with a stable stat."""
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if not isinstance(nofollow, int):
        _fail(f"{label} requires O_NOFOLLOW support")
    directory_flags = os.O_RDONLY | nofollow
    directory = getattr(os, "O_DIRECTORY", None)
    if isinstance(directory, int):
        directory_flags |= directory
    parent_fd: int | None = None
    file_fd: int | None = None
    try:
        parent_fd = os.open(path.parent, directory_flags)
        file_fd = os.open(path.name, os.O_RDONLY | nofollow, dir_fd=parent_fd)
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum_bytes:
            _fail(f"{label} must be a bounded regular file")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(file_fd, min(maximum_bytes + 1 - total, 65_536))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum_bytes:
                _fail(f"{label} exceeds its byte bound")
        content = b"".join(chunks)
        after = os.fstat(file_fd)
    except ExactFinitePolicyFunctionalError:
        raise
    except OSError as error:
        raise ExactFinitePolicyFunctionalError(
            f"{label} cannot be opened safely"
        ) from error
    finally:
        if file_fd is not None:
            os.close(file_fd)
        if parent_fd is not None:
            os.close(parent_fd)
    if before.st_size != len(content) or (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        _fail(f"{label} changed while it was read")
    return content


def _read_bounded_selection_state(
    path: Path,
    *,
    label: str,
) -> tuple[dict[str, object], str]:
    content = _read_stable_bounded_regular_bytes(path, label=label)
    state = read_json_bytes_object(content, source=str(path))
    expected_fields = frozenset(
        {
            "audit_epoch",
            "audited_design_sha256",
            "blind_audit_attempt_claim_sha256",
            "blind_audit_output",
            "blind_audit_output_receipt",
            "blind_audit_selection_input_sha256",
            "candidate_statuses",
            "manifest_selected_seed",
            "plan_scenario_seed",
            "production_authorized",
            "schema_version",
        }
    )
    if frozenset(state) != expected_fields:
        _fail("scenario selection state has missing or unknown fields")
    if state["schema_version"] != _SELECTION_STATE_SCHEMA_VERSION:
        _fail("scenario selection state schema changed")
    if state["audit_epoch"] != _SCENARIO_AUDIT_EPOCH:
        _fail("scenario selection state belongs to a different audit epoch")
    _identifier(state["plan_scenario_seed"], "selection plan_scenario_seed")
    selected = state["manifest_selected_seed"]
    if selected is not None:
        _identifier(selected, "selection manifest_selected_seed")
    for field in ("audited_design_sha256", "blind_audit_selection_input_sha256"):
        if state[field] is not None:
            _sha256(state[field], f"selection {field}")
    attempt_claim_sha256 = state["blind_audit_attempt_claim_sha256"]
    if attempt_claim_sha256 is not None:
        _sha256(
            attempt_claim_sha256,
            "selection blind_audit_attempt_claim_sha256",
        )
    if not isinstance(state["blind_audit_output"], str):
        _fail("selection blind_audit_output must be a string")
    receipt = state["blind_audit_output_receipt"]
    if type(receipt) is not dict or not all(type(key) is str for key in receipt):
        _fail("selection blind-audit receipt must be an exact object")
    statuses = state["candidate_statuses"]
    if type(statuses) is not list or not all(type(item) is str for item in statuses):
        _fail("selection candidate statuses must be a string array")
    if type(state["production_authorized"]) is not bool:
        _fail("selection production_authorized must be boolean")
    return state, sha256_bytes(content)


def _load_exact_finite_scenario_selection_state() -> tuple[dict[str, object], str]:
    return _read_bounded_selection_state(
        _selection_state_path(),
        label="scenario selection state",
    )


def _load_exact_finite_scenario_selection_input_snapshot() -> tuple[
    dict[str, object], str
]:
    state, source_sha256 = _read_bounded_selection_state(
        _selection_input_snapshot_path(),
        label="scenario selection pending-input snapshot",
    )
    pending_statuses = [
        "pending_post_freeze_blind_audit"
        for _ in POLICY_FUNCTIONAL_EXACT_FINITE_SCENARIO_CANDIDATE_SEEDS
    ]
    if state != {
        "audit_epoch": _SCENARIO_AUDIT_EPOCH,
        "audited_design_sha256": None,
        "blind_audit_attempt_claim_sha256": None,
        "blind_audit_output": "pending_post_freeze_blind_audit",
        "blind_audit_output_receipt": {"status": "pending_post_freeze_blind_audit"},
        "blind_audit_selection_input_sha256": None,
        "candidate_statuses": pending_statuses,
        "manifest_selected_seed": (
            POLICY_FUNCTIONAL_EXACT_FINITE_SCENARIO_CANDIDATE_SEEDS[0]
        ),
        "plan_scenario_seed": (
            POLICY_FUNCTIONAL_EXACT_FINITE_SCENARIO_CANDIDATE_SEEDS[0]
        ),
        "production_authorized": False,
        "schema_version": _SELECTION_STATE_SCHEMA_VERSION,
    }:
        _fail("scenario selection pending-input snapshot changed semantically")
    return state, source_sha256


_INITIAL_SELECTION_STATE, _INITIAL_SELECTION_SOURCE_SHA256 = (
    _load_exact_finite_scenario_selection_state()
)
_INITIAL_SELECTION_INPUT_SNAPSHOT, _INITIAL_SELECTION_INPUT_SHA256 = (
    _load_exact_finite_scenario_selection_input_snapshot()
)
if _INITIAL_SELECTION_STATE["blind_audit_output"] == (
    "pending_post_freeze_blind_audit"
) and (
    _INITIAL_SELECTION_STATE["production_authorized"] is not False
    or _INITIAL_SELECTION_SOURCE_SHA256 != _INITIAL_SELECTION_INPUT_SHA256
):
    _fail("pending scenario selection state differs from its immutable input snapshot")
POLICY_FUNCTIONAL_EXACT_FINITE_SELECTED_SCENARIO_SEED: str = cast(
    str, _INITIAL_SELECTION_STATE["plan_scenario_seed"]
)


def _superseded_m_r_disclosure() -> dict[str, object]:
    final_path = _superseded_m_r_final_path()
    pending_path = _superseded_m_r_pending_path()
    final_content = _read_stable_bounded_regular_bytes(
        final_path,
        label="superseded m--r final selection history",
    )
    pending_content = _read_stable_bounded_regular_bytes(
        pending_path,
        label="superseded m--r pending selection history",
    )
    final_sha256 = sha256_bytes(final_content)
    pending_sha256 = sha256_bytes(pending_content)
    if (
        final_sha256 != _SUPERSEDED_M_R_FINAL_SOURCE_SHA256
        or pending_sha256 != _SUPERSEDED_M_R_PENDING_SOURCE_SHA256
    ):
        _fail("superseded m--r selection history changed byte-for-byte")
    final_state = read_json_bytes_object(
        final_content,
        source=str(final_path),
    )
    expected_statuses = [
        "blind_audit_nondegenerate_selected",
        *("not_evaluated_after_first_eligible" for _ in range(5)),
    ]
    if (
        final_state.get("blind_audit_output")
        != "exact-finite-policy-functional-grid-m-v1"
        or final_state.get("manifest_selected_seed")
        != "exact-finite-policy-functional-grid-m-v1"
        or final_state.get("candidate_statuses") != expected_statuses
        or final_state.get("production_authorized") is not True
    ):
        _fail("superseded m--r final history changed semantically")
    pending_state = read_json_bytes_object(
        pending_content,
        source=str(pending_path),
    )
    if (
        pending_state.get("blind_audit_output") != "pending_post_freeze_blind_audit"
        or pending_state.get("production_authorized") is not False
        or pending_state.get("candidate_statuses")
        != ["pending_post_freeze_blind_audit"] * 6
    ):
        _fail("superseded m--r pending history changed semantically")
    return {
        "artifact_identifiers": {
            "evidence_bundle_root_sha256": _SUPERSEDED_M_R_BUNDLE_ROOT_SHA256,
            "exact_finite_plan_sha256": _SUPERSEDED_M_R_PLAN_SHA256,
            "exact_finite_result_sha256": _SUPERSEDED_M_R_RESULT_SHA256,
        },
        "candidate_grid": list(_EXCLUDED_SUPERSEDED_SOURCE_AUDIT_CANDIDATES),
        "candidate_statuses": expected_statuses,
        "initial_invalidation_trigger_paths": [
            "src/policytwin/calibration/diagnostics.py",
            "src/policytwin/counterfactual/randomness.py",
        ],
        "effect_values_or_diagnostics_reproduced_here": False,
        "external_custody_verified": False,
        "formal_result_persisted": True,
        "history_sources": {
            "final_file": final_path.name,
            "final_source_sha256": final_sha256,
            "pending_file": pending_path.name,
            "pending_source_sha256": pending_sha256,
        },
        "independent_reproduction": False,
        "previous_audited_design_sha256": (_SUPERSEDED_M_R_AUDITED_DESIGN_SHA256),
        "previous_execution_receipt_sha256": (_SUPERSEDED_M_R_EXECUTION_RECEIPT_SHA256),
        "prior_production_authorization_currently_valid": False,
        "reason": "required_source_correctness_fixes_changed_immutable_source_tree",
        "selected_seed": "exact-finite-policy-functional-grid-m-v1",
        "selection_and_result_were_exposed": True,
    }


def exact_finite_scenario_design_manifest() -> dict[str, object]:
    """Return the closed design-time fixture-selection disclosure.

    No policy-effect magnitude or rank diagnostic is stored here.  The a--f
    pilot, g--l stale audit, and source-superseded m--r audit are disjoint from
    the unexecuted s--x production grid.  Formal canary runs rederive all policy
    values through the public runner.
    """
    selection, selection_source_sha256 = _load_exact_finite_scenario_selection_state()
    raw_receipt = cast(dict[str, object], selection["blind_audit_output_receipt"])
    return {
        "audit_epoch": selection["audit_epoch"],
        "blind_audit_attempt_claim_sha256": selection[
            "blind_audit_attempt_claim_sha256"
        ],
        "blind_audit_output": selection["blind_audit_output"],
        "blind_audit_output_receipt": dict(raw_receipt),
        "blind_audit_selection_input_sha256": selection[
            "blind_audit_selection_input_sha256"
        ],
        "audited_design_sha256": (
            exact_finite_scenario_audited_design_sha256()
            if selection["audited_design_sha256"] is None
            else selection["audited_design_sha256"]
        ),
        "candidate_seeds": list(
            POLICY_FUNCTIONAL_EXACT_FINITE_SCENARIO_CANDIDATE_SEEDS
        ),
        "candidate_statuses": list(cast(list[str], selection["candidate_statuses"])),
        "excluded_exposed_pilot": {
            "candidate_grid": list(_EXCLUDED_EXPOSED_PILOT_CANDIDATES),
            "effect_values_exposed_for": [
                "exact-finite-policy-functional-grid-a-v1",
                "exact-finite-policy-functional-grid-b-v1",
            ],
            "reason": "effect_values_were_printed_so_pilot_cannot_select_fixture",
        },
        "excluded_superseded_source_audit": _superseded_m_r_disclosure(),
        "selection_uses_only_nondegeneracy": True,
        "excluded_stale_pre_freeze_audit": {
            "candidate_grid": list(_EXCLUDED_STALE_AUDIT_CANDIDATES),
            "emitted_seed_only": "exact-finite-policy-functional-grid-g-v1",
            "reason": (
                "internal_receipt_schema_was_not_frozen_and_grid_is_now_exposed"
            ),
        },
        "production_authorized": selection["production_authorized"],
        "production_runtime_search": False,
        "schema_version": "policytwin.exact-finite-scenario-design.v2",
        "selected_seed": selection["manifest_selected_seed"],
        "selection_input_snapshot_source_sha256": (
            _load_exact_finite_scenario_selection_input_snapshot()[1]
        ),
        "selection_state_source_sha256": selection_source_sha256,
        "selection_rule": POLICY_FUNCTIONAL_EXACT_FINITE_SCENARIO_SELECTION_RULE,
        "synthetic_test_fixture_design_only": True,
    }


def derive_exact_finite_scenario_audit_statuses(
    blind_audit_output: str,
) -> tuple[str, ...]:
    """Mechanically derive the bounded audit path from seed-only output.

    The blind process is allowed to emit only one frozen candidate seed or
    ``no-match``.  A selected seed therefore unambiguously commits every
    earlier candidate as evaluated-and-degenerate and every later candidate as
    not evaluated after the first eligible fixture.  No effect direction,
    magnitude, or rank diagnostic is needed to reconstruct this status vector.
    """
    if not isinstance(blind_audit_output, str):
        raise TypeError("blind_audit_output must be a string")
    candidates = POLICY_FUNCTIONAL_EXACT_FINITE_SCENARIO_CANDIDATE_SEEDS
    evaluated_degenerate = "blind_audit_evaluated_degenerate"
    if blind_audit_output == "no-match":
        return tuple(evaluated_degenerate for _ in candidates)
    if blind_audit_output not in candidates:
        _fail("blind audit output is outside the frozen candidate grid")
    selected_index = candidates.index(blind_audit_output)
    return tuple(
        evaluated_degenerate
        if index < selected_index
        else (
            "blind_audit_nondegenerate_selected"
            if index == selected_index
            else "not_evaluated_after_first_eligible"
        )
        for index in range(len(candidates))
    )


def derive_exact_finite_scenario_blind_audit_output_receipt(
    blind_audit_output: str,
    *,
    attempt_claim_sha256: str,
    selection_input_sha256: str | None = None,
) -> dict[str, object]:
    """Derive the static contract nested inside an execution-issued receipt."""
    statuses = derive_exact_finite_scenario_audit_statuses(blind_audit_output)
    _, frozen_input_sha256 = _load_exact_finite_scenario_selection_input_snapshot()
    if selection_input_sha256 is None:
        input_sha256 = frozen_input_sha256
    else:
        input_sha256 = _sha256(
            selection_input_sha256, "blind audit selection_input_sha256"
        )
    if input_sha256 != frozen_input_sha256:
        _fail("blind audit input digest differs from the frozen pending snapshot")
    return {
        "attempt_claim_sha256": _sha256(
            attempt_claim_sha256,
            "blind audit output attempt_claim_sha256",
        ),
        "candidate_grid_sha256": _digest(
            list(POLICY_FUNCTIONAL_EXACT_FINITE_SCENARIO_CANDIDATE_SEEDS)
        ),
        "audited_design_sha256": exact_finite_scenario_audited_design_sha256(),
        "candidate_statuses_sha256": _digest(list(statuses)),
        "selection_input_sha256": input_sha256,
        "effect_diagnostics_persistence_absence_verified": False,
        "external_custody_verified": False,
        "execution_lineage_sha256": _digest(
            _scenario_audit_execution_lineage_mapping()
        ),
        "independent_reproduction": False,
        "schema_version": "policytwin.exact-finite-blind-audit-output.v2",
        "selected_seed": (
            None if blind_audit_output == "no-match" else blind_audit_output
        ),
        "selection_rule": POLICY_FUNCTIONAL_EXACT_FINITE_SCENARIO_SELECTION_RULE,
        "stdout_sha256": sha256_bytes(f"{blind_audit_output}\n".encode()),
        "process_fd_output_observed_during_synchronous_evaluation": False,
        "python_stream_output_observed_during_synchronous_evaluation": False,
        "synthetic_test_fixture_design_only": True,
    }


def _blind_audit_cli_source_path() -> Path:
    return Path(__file__).resolve().parents[1] / "exact_finite_blind_audit.py"


def _canonical_blind_audit_orig_argv() -> tuple[str, ...]:
    """Return the sole locally authorized direct-script invocation shape."""
    script = _blind_audit_cli_source_path().resolve()
    main = sys.modules.get("__main__")
    main_file = getattr(main, "__file__", None)
    if not isinstance(main_file, str) or Path(main_file).resolve() != script:
        _fail("blind audit must be entered through the canonical CLI source")
    if len(sys.argv) != 1 or Path(sys.argv[0]).resolve() != script:
        _fail("blind audit CLI argv differs from the fixed no-argument entry")
    original = tuple(sys.orig_argv)
    if not (
        len(original) == 4
        and original[1:3] == ("-I", "-S")
        and Path(original[3]).resolve() == script
    ):
        _fail("blind audit orig_argv is not the canonical -I -S direct script")
    return original


def _source_only_loaded_manifest() -> tuple[dict[str, object], ...]:
    """Read the bootstrap importer's manifest and reject bytecode imports."""
    finders = tuple(
        finder
        for finder in sys.meta_path
        if getattr(finder, "policytwin_source_only_blind_loader", False) is True
    )
    if len(finders) != 1:
        _fail("blind audit requires exactly one source-only PolicyTwin importer")
    finder = finders[0]
    source_root = getattr(finder, "source_root", None)
    expected_source_root = Path(__file__).resolve().parents[2]
    if not isinstance(source_root, Path) or source_root != expected_source_root:
        _fail("blind audit source-only importer has the wrong source root")
    manifest_factory = getattr(finder, "loaded_source_manifest", None)
    if not callable(manifest_factory):
        _fail("blind audit source-only importer exposes no loaded manifest")
    raw_manifest = manifest_factory()
    if not isinstance(raw_manifest, tuple) or not raw_manifest:
        _fail("blind audit source-only loaded manifest is empty or invalid")
    expected_fields = frozenset({"module", "path", "sha256", "size_bytes"})
    records: list[dict[str, object]] = []
    seen_modules: set[str] = set()
    seen_paths: set[str] = set()
    for raw in raw_manifest:
        if type(raw) is not dict or frozenset(raw) != expected_fields:
            _fail("blind audit source-only loaded record has the wrong schema")
        module_name = raw["module"]
        path = raw["path"]
        size = raw["size_bytes"]
        if (
            not isinstance(module_name, str)
            or not (
                module_name == "policytwin" or module_name.startswith("policytwin.")
            )
            or module_name in seen_modules
            or not isinstance(path, str)
            or not path.startswith("src/policytwin/")
            or path in seen_paths
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or size > 2_097_152
        ):
            _fail("blind audit source-only loaded record is invalid")
        _sha256(raw["sha256"], "source-only loaded record sha256")
        module = sys.modules.get(module_name)
        module_file = getattr(module, "__file__", None)
        expected_path = expected_source_root.parent / path
        if (
            module is None
            or not isinstance(module_file, str)
            or Path(module_file).resolve() != expected_path.resolve()
            or getattr(module, "__cached__", None) is not None
        ):
            _fail("PolicyTwin module was not loaded solely from committed source")
        seen_modules.add(module_name)
        seen_paths.add(path)
        records.append(dict(raw))
    loaded_policytwin_modules = {
        name
        for name, module in sys.modules.items()
        if (name == "policytwin" or name.startswith("policytwin."))
        and isinstance(module, object)
        and getattr(module, "__file__", None) is not None
    }
    if seen_modules != loaded_policytwin_modules:
        _fail("source-only manifest does not cover every loaded PolicyTwin module")
    records.sort(key=lambda item: cast(str, item["module"]))
    return tuple(records)


def _require_loaded_policytwin_sources_match_disk() -> str:
    """Bind every compiled PolicyTwin source byte string to the tree snapshot."""
    loaded = _source_only_loaded_manifest()
    first = _scenario_audit_source_tree_manifest()
    second = _scenario_audit_source_tree_manifest()
    if first != second:
        _fail("immutable source tree changed across its complete snapshot passes")
    disk_by_path = {cast(str, item["path"]): item for item in first}
    for record in loaded:
        disk = disk_by_path.get(cast(str, record["path"]))
        if disk is None or (
            disk["sha256"] != record["sha256"]
            or disk["size_bytes"] != record["size_bytes"]
        ):
            _fail("loaded PolicyTwin code differs from committed source bytes")
    return _digest(list(loaded))


def _require_isolated_blind_audit_process() -> None:
    flags = sys.flags
    if not (
        flags.isolated == 1
        and flags.no_site == 1
        and flags.ignore_environment == 1
        and flags.safe_path is True
    ):
        _fail("blind audit CLI requires Python -I -S isolated execution")
    forbidden_modules = {"site", "sitecustomize", "usercustomize"} & set(sys.modules)
    if forbidden_modules:
        _fail("blind audit CLI loaded a forbidden site customization module")
    _canonical_blind_audit_orig_argv()
    _require_loaded_policytwin_sources_match_disk()


def _blind_audit_attempt_runtime_mapping() -> dict[str, object]:
    _require_isolated_blind_audit_process()
    original = _canonical_blind_audit_orig_argv()
    executable = Path(sys.executable).resolve()
    if not executable.is_file():
        _fail("blind audit Python executable is not a regular file")
    return {
        "claimed_at_utc": datetime.now(UTC).isoformat(),
        "execution_nonce_sha256": sha256_bytes(secrets.token_bytes(32)),
        "canonical_direct_cli_entry": True,
        "ignore_environment": True,
        "isolated": True,
        "loaded_source_manifest_sha256": (
            _require_loaded_policytwin_sources_match_disk()
        ),
        "no_site": True,
        "orig_argv_sha256": _digest(list(original)),
        "python_cache_tag": sys.implementation.cache_tag,
        "python_executable_sha256": sha256_file(executable),
        "python_implementation": sys.implementation.name,
        "python_version": sys.version,
        "safe_path": True,
        "site_modules_absent": True,
    }


def _build_blind_audit_attempt_claim(
    *, selection_input_sha256: str
) -> dict[str, object]:
    _, frozen_input_sha256 = _load_exact_finite_scenario_selection_input_snapshot()
    input_sha256 = _sha256(
        selection_input_sha256,
        "blind audit attempt selection_input_sha256",
    )
    if input_sha256 != frozen_input_sha256:
        _fail("blind audit attempt does not bind the frozen pending snapshot")
    return {
        "audit_epoch": _SCENARIO_AUDIT_EPOCH,
        "audited_design_sha256": exact_finite_scenario_audited_design_sha256(),
        "candidate_grid_sha256": _digest(
            list(POLICY_FUNCTIONAL_EXACT_FINITE_SCENARIO_CANDIDATE_SEEDS)
        ),
        "cli_source_sha256": sha256_file(_blind_audit_cli_source_path()),
        "external_custody_verified": False,
        "runtime": _blind_audit_attempt_runtime_mapping(),
        "schema_version": "policytwin.exact-finite-blind-attempt.v2",
        "selection_input_sha256": input_sha256,
    }


def _blind_audit_attempt_claim_bytes(claim: Mapping[str, object]) -> bytes:
    return canonical_json_bytes(dict(claim)) + b"\n"


def _validate_blind_audit_attempt_claim(
    claim: object,
    *,
    claim_sha256: str,
    selection_input_sha256: str,
) -> None:
    expected_fields = frozenset(
        {
            "audit_epoch",
            "audited_design_sha256",
            "candidate_grid_sha256",
            "cli_source_sha256",
            "external_custody_verified",
            "runtime",
            "schema_version",
            "selection_input_sha256",
        }
    )
    if type(claim) is not dict or frozenset(claim) != expected_fields:
        _fail("blind audit attempt claim has the wrong schema")
    expected_sha256 = _sha256(claim_sha256, "blind audit attempt claim_sha256")
    if sha256_bytes(_blind_audit_attempt_claim_bytes(claim)) != expected_sha256:
        _fail("blind audit attempt claim hash does not verify")
    if (
        claim["schema_version"] != "policytwin.exact-finite-blind-attempt.v2"
        or claim["audit_epoch"] != _SCENARIO_AUDIT_EPOCH
        or claim["external_custody_verified"] is not False
    ):
        _fail("blind audit attempt claim changed its authority or epoch")
    if claim["candidate_grid_sha256"] != _digest(
        list(POLICY_FUNCTIONAL_EXACT_FINITE_SCENARIO_CANDIDATE_SEEDS)
    ):
        _fail("blind audit attempt claim does not bind the frozen candidate grid")
    if claim["selection_input_sha256"] != _sha256(
        selection_input_sha256,
        "blind audit attempt expected selection_input_sha256",
    ):
        _fail("blind audit attempt claim does not bind H1")
    if claim["audited_design_sha256"] != exact_finite_scenario_audited_design_sha256():
        _fail("blind audit attempt claim audited design is stale")
    if claim["cli_source_sha256"] != sha256_file(_blind_audit_cli_source_path()):
        _fail("blind audit attempt claim CLI source is stale")
    runtime = claim["runtime"]
    expected_runtime_fields = frozenset(
        {
            "claimed_at_utc",
            "canonical_direct_cli_entry",
            "execution_nonce_sha256",
            "ignore_environment",
            "isolated",
            "loaded_source_manifest_sha256",
            "no_site",
            "orig_argv_sha256",
            "python_cache_tag",
            "python_executable_sha256",
            "python_implementation",
            "python_version",
            "safe_path",
            "site_modules_absent",
        }
    )
    if type(runtime) is not dict or frozenset(runtime) != expected_runtime_fields:
        _fail("blind audit attempt runtime has the wrong schema")
    for field in (
        "ignore_environment",
        "isolated",
        "canonical_direct_cli_entry",
        "no_site",
        "safe_path",
        "site_modules_absent",
    ):
        if runtime[field] is not True:
            _fail("blind audit attempt runtime was not isolated")
    for field in (
        "execution_nonce_sha256",
        "loaded_source_manifest_sha256",
        "orig_argv_sha256",
        "python_executable_sha256",
    ):
        _sha256(runtime[field], f"blind audit attempt runtime {field}")
    for field in ("python_cache_tag", "python_implementation", "python_version"):
        if not isinstance(runtime[field], str) or not runtime[field]:
            _fail(f"blind audit attempt runtime {field} must be nonempty")
    if (
        runtime["python_cache_tag"] != sys.implementation.cache_tag
        or runtime["python_implementation"] != sys.implementation.name
        or runtime["python_version"] != sys.version
    ):
        _fail("blind audit attempt Python runtime differs from validation runtime")
    executable = Path(sys.executable).resolve()
    if not executable.is_file() or runtime["python_executable_sha256"] != sha256_file(
        executable
    ):
        _fail("blind audit attempt Python executable commitment is stale")
    claimed_at = runtime["claimed_at_utc"]
    if not isinstance(claimed_at, str):
        _fail("blind audit attempt claimed_at_utc must be a string")
    try:
        claimed = datetime.fromisoformat(claimed_at)
    except ValueError as error:
        raise ExactFinitePolicyFunctionalError(
            "blind audit attempt claimed_at_utc is invalid"
        ) from error
    _aware(claimed, "blind audit attempt claimed_at_utc")


def _read_bounded_blind_audit_attempt_claim() -> tuple[dict[str, object], str]:
    path = _blind_attempt_claim_path()
    content = _read_stable_bounded_regular_bytes(
        path,
        label="blind audit attempt claim",
    )
    claim = read_json_bytes_object(content, source=str(path))
    return claim, sha256_bytes(content)


def _claim_blind_audit_attempt(
    *, selection_input_sha256: str
) -> tuple[dict[str, object], str, object]:
    """Consume the epoch exactly once before any candidate evaluator runs."""
    claim = _build_blind_audit_attempt_claim(
        selection_input_sha256=selection_input_sha256
    )
    content = _blind_audit_attempt_claim_bytes(claim)
    path = _blind_attempt_claim_path()
    if path.name in {"", ".", ".."}:
        _fail("blind audit attempt claim filename is invalid")
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if not isinstance(nofollow, int):
        _fail("blind audit attempt requires O_NOFOLLOW support")
    directory_flags = os.O_RDONLY | nofollow
    directory = getattr(os, "O_DIRECTORY", None)
    if isinstance(directory, int):
        directory_flags |= directory
    try:
        parent_fd = os.open(path.parent, directory_flags)
    except OSError as error:
        raise ExactFinitePolicyFunctionalError(
            "cannot open blind audit attempt parent directory safely"
        ) from error
    file_fd: int | None = None
    try:
        try:
            file_fd = os.open(
                path.name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow,
                0o600,
                dir_fd=parent_fd,
            )
        except FileExistsError as error:
            raise ExactFinitePolicyFunctionalError(
                "blind audit attempt claim already exists; this epoch is consumed "
                "and must not be rerun"
            ) from error
        offset = 0
        while offset < len(content):
            written = os.write(file_fd, content[offset:])
            if written <= 0:
                _fail("blind audit attempt claim write made no progress")
            offset += written
        os.fsync(file_fd)
        os.close(file_fd)
        file_fd = None
        os.fsync(parent_fd)
    except ExactFinitePolicyFunctionalError:
        raise
    except OSError as error:
        raise ExactFinitePolicyFunctionalError(
            "blind audit attempt claim could not be durably persisted"
        ) from error
    finally:
        if file_fd is not None:
            os.close(file_fd)
        os.close(parent_fd)
    persisted, claim_sha256 = _read_bounded_blind_audit_attempt_claim()
    if persisted != claim:
        _fail("blind audit attempt claim did not round-trip")
    _validate_blind_audit_attempt_claim(
        persisted,
        claim_sha256=claim_sha256,
        selection_input_sha256=selection_input_sha256,
    )
    capability = object()
    with _BLIND_AUDIT_CAPABILITY_LOCK:
        _BLIND_AUDIT_ATTEMPT_CAPABILITIES[id(capability)] = (
            capability,
            claim_sha256,
            os.getpid(),
            get_ident(),
        )
    return persisted, claim_sha256, capability


def _consume_blind_audit_attempt_capability(
    capability: object,
) -> tuple[str, object]:
    """Consume attempt authority and mint one claim-bound activation ticket."""
    with _BLIND_AUDIT_CAPABILITY_LOCK:
        registered = _BLIND_AUDIT_ATTEMPT_CAPABILITIES.get(id(capability))
        if (
            registered is None
            or registered[0] is not capability
            or registered[2:] != (os.getpid(), get_ident())
        ):
            _fail("blind audit evaluator requires an unconsumed attempt capability")
        del _BLIND_AUDIT_ATTEMPT_CAPABILITIES[id(capability)]
    ticket = object()
    with _BLIND_AUDIT_CAPABILITY_LOCK:
        _BLIND_AUDIT_ACTIVATION_TICKETS[id(ticket)] = (
            ticket,
            registered[1],
            os.getpid(),
            get_ident(),
        )
    return registered[1], ticket


def _activate_blind_audit_evaluation(ticket: object) -> object:
    """Consume a claim-issued ticket; there is no standalone token issuer."""
    with _BLIND_AUDIT_CAPABILITY_LOCK:
        registered = _BLIND_AUDIT_ACTIVATION_TICKETS.get(id(ticket))
        if (
            registered is None
            or registered[0] is not ticket
            or registered[2:] != (os.getpid(), get_ident())
        ):
            _fail("blind audit activation requires an unconsumed claim-bound ticket")
        del _BLIND_AUDIT_ACTIVATION_TICKETS[id(ticket)]
    _, durable_claim_sha256 = _read_bounded_blind_audit_attempt_claim()
    if durable_claim_sha256 != registered[1]:
        _fail("blind audit activation ticket differs from the durable claim")
    capability = object()
    with _BLIND_AUDIT_CAPABILITY_LOCK:
        _ACTIVE_BLIND_AUDIT_EVALUATIONS[id(capability)] = (
            capability,
            registered[1],
            os.getpid(),
            get_ident(),
        )
    return capability


def _discard_blind_audit_activation_ticket(ticket: object) -> None:
    """Consume an unactivated ticket on every pre-activation failure."""
    with _BLIND_AUDIT_CAPABILITY_LOCK:
        registered = _BLIND_AUDIT_ACTIVATION_TICKETS.get(id(ticket))
        if (
            registered is None
            or registered[0] is not ticket
            or registered[2:] != (os.getpid(), get_ident())
        ):
            _fail("blind audit activation ticket cleanup changed authority context")
        del _BLIND_AUDIT_ACTIVATION_TICKETS[id(ticket)]


def _require_active_blind_audit_evaluation(capability: object) -> str:
    with _BLIND_AUDIT_CAPABILITY_LOCK:
        registered = _ACTIVE_BLIND_AUDIT_EVALUATIONS.get(id(capability))
    if (
        registered is None
        or registered[0] is not capability
        or registered[2:] != (os.getpid(), get_ident())
    ):
        _fail("candidate evaluator requires an active blind-audit capability")
    return registered[1]


def _abort_blind_audit_evaluation(capability: object) -> None:
    """Consume active authority on every failed evaluation path."""
    with _BLIND_AUDIT_CAPABILITY_LOCK:
        registered = _ACTIVE_BLIND_AUDIT_EVALUATIONS.get(id(capability))
        if (
            registered is None
            or registered[0] is not capability
            or registered[2:] != (os.getpid(), get_ident())
        ):
            _fail("blind-audit evaluation capability lifecycle changed")
        del _ACTIVE_BLIND_AUDIT_EVALUATIONS[id(capability)]


def _expected_blind_audit_evaluated_candidates(
    blind_audit_output: str,
) -> tuple[str, ...]:
    candidates = POLICY_FUNCTIONAL_EXACT_FINITE_SCENARIO_CANDIDATE_SEEDS
    if blind_audit_output == "no-match":
        return candidates
    if blind_audit_output not in candidates:
        _fail("blind audit completion output is outside the frozen grid")
    return candidates[: candidates.index(blind_audit_output) + 1]


def _complete_blind_audit_evaluation(
    capability: object,
    *,
    blind_audit_output: str,
    observations: tuple[tuple[str, bool], ...],
) -> object:
    """Convert one active candidate loop into a single-use completion token."""
    expected_candidates = _expected_blind_audit_evaluated_candidates(blind_audit_output)
    expected_observations = tuple(
        (
            candidate,
            blind_audit_output != "no-match" and candidate == blind_audit_output,
        )
        for candidate in expected_candidates
    )
    if observations != expected_observations:
        _fail("blind audit completion does not prove the ordered first-match loop")
    with _BLIND_AUDIT_CAPABILITY_LOCK:
        registered = _ACTIVE_BLIND_AUDIT_EVALUATIONS.get(id(capability))
        if (
            registered is None
            or registered[0] is not capability
            or registered[2:] != (os.getpid(), get_ident())
        ):
            _fail("blind audit completion capability lifecycle changed")
        del _ACTIVE_BLIND_AUDIT_EVALUATIONS[id(capability)]
    claim_sha256 = registered[1]
    completion = object()
    with _BLIND_AUDIT_CAPABILITY_LOCK:
        _BLIND_AUDIT_COMPLETION_CAPABILITIES[id(completion)] = (
            completion,
            claim_sha256,
            blind_audit_output,
            expected_candidates,
            os.getpid(),
            get_ident(),
        )
    return completion


def _consume_blind_audit_completion_capability(
    capability: object,
) -> tuple[str, str, tuple[str, ...]]:
    with _BLIND_AUDIT_CAPABILITY_LOCK:
        registered = _BLIND_AUDIT_COMPLETION_CAPABILITIES.get(id(capability))
        if (
            registered is None
            or registered[0] is not capability
            or registered[4:] != (os.getpid(), get_ident())
        ):
            _fail("stdout finalization requires an unconsumed evaluation completion")
        del _BLIND_AUDIT_COMPLETION_CAPABILITIES[id(capability)]
    return registered[1], registered[2], registered[3]


def _discard_blind_audit_completion_capability_if_present(
    capability: object,
) -> None:
    """Remove an owner-bound completion after a failed handoff to stdout."""
    with _BLIND_AUDIT_CAPABILITY_LOCK:
        registered = _BLIND_AUDIT_COMPLETION_CAPABILITIES.get(id(capability))
        if registered is None:
            return
        if registered[0] is not capability or registered[4:] != (
            os.getpid(),
            get_ident(),
        ):
            _fail("blind audit completion cleanup changed authority context")
        del _BLIND_AUDIT_COMPLETION_CAPABILITIES[id(capability)]


def _blind_audit_execution_mapping_from_claim(
    claim: Mapping[str, object], *, blind_audit_output: str
) -> dict[str, object]:
    runtime = cast(dict[str, object], claim["runtime"])
    evaluated = _expected_blind_audit_evaluated_candidates(blind_audit_output)
    return {
        "candidate_loop_completed": True,
        "canonical_direct_cli_entry": runtime["canonical_direct_cli_entry"],
        "cli_source_sha256": claim["cli_source_sha256"],
        "evaluated_candidate_count": len(evaluated),
        "evaluated_candidates_sha256": _digest(list(evaluated)),
        "execution_nonce_sha256": runtime["execution_nonce_sha256"],
        "ignore_environment": runtime["ignore_environment"],
        "isolated": runtime["isolated"],
        "attempt_claimed_at_utc": runtime["claimed_at_utc"],
        "loaded_source_manifest_sha256": runtime["loaded_source_manifest_sha256"],
        "local_completion_capability_consumed": True,
        "no_site": runtime["no_site"],
        "orig_argv_sha256": runtime["orig_argv_sha256"],
        "python_cache_tag": runtime["python_cache_tag"],
        "python_executable_sha256": runtime["python_executable_sha256"],
        "python_implementation": runtime["python_implementation"],
        "python_version": runtime["python_version"],
        "safe_path": runtime["safe_path"],
        "selection_rule_order_verified": True,
        "site_modules_absent": runtime["site_modules_absent"],
        "stdout_write_and_flush_completed": True,
    }


def _issue_blind_audit_execution_receipt(
    expected_blind_audit_output: str,
    *,
    attempt_claim: Mapping[str, object],
    attempt_claim_sha256: str,
    finalization_capability: object,
) -> dict[str, object]:
    with _BLIND_AUDIT_CAPABILITY_LOCK:
        stdout_record = _BLIND_AUDIT_STDOUT_CAPABILITIES.get(
            id(finalization_capability)
        )
        if (
            stdout_record is None
            or stdout_record[0] is not finalization_capability
            or stdout_record[5:] != (os.getpid(), get_ident())
        ):
            _fail("receipt issuance requires the matching stdout completion")
        del _BLIND_AUDIT_STDOUT_CAPABILITIES[id(finalization_capability)]
    _, completion_claim_sha256, blind_audit_output, evaluated, stdout_sha256, _, _ = (
        stdout_record
    )
    expected_stdout_sha256 = sha256_bytes(f"{blind_audit_output}\n".encode())
    if (
        completion_claim_sha256 != attempt_claim_sha256
        or blind_audit_output != expected_blind_audit_output
        or evaluated != _expected_blind_audit_evaluated_candidates(blind_audit_output)
        or stdout_sha256 != expected_stdout_sha256
    ):
        _fail("stdout completion differs from receipt issuance inputs")
    durable_claim, durable_claim_sha256 = _read_bounded_blind_audit_attempt_claim()
    if durable_claim_sha256 != attempt_claim_sha256 or durable_claim != attempt_claim:
        _fail("receipt issuance inputs differ from the durable attempt claim")
    contract = derive_exact_finite_scenario_blind_audit_output_receipt(
        blind_audit_output,
        attempt_claim_sha256=attempt_claim_sha256,
    )
    execution = _blind_audit_execution_mapping_from_claim(
        attempt_claim,
        blind_audit_output=blind_audit_output,
    )
    payload: dict[str, object] = {
        "attempt_claim": dict(attempt_claim),
        "attempt_claim_sha256": _sha256(
            attempt_claim_sha256,
            "blind audit receipt attempt_claim_sha256",
        ),
        "contract": contract,
        "execution": execution,
        "execution_history_external_attestation_verified": False,
        "local_workflow_guard_only": True,
        "malicious_workspace_owner_resistance_verified": False,
        "retry_prevention_external_custody_verified": False,
        "schema_version": "policytwin.exact-finite-blind-audit-execution.v4",
    }
    return {**payload, "receipt_sha256": _digest(payload)}


def _write_blind_audit_stdout_bytes(
    payload: bytes, *, completion_capability: object
) -> object:
    completion_claim_sha256, blind_audit_output, evaluated = (
        _consume_blind_audit_completion_capability(completion_capability)
    )
    if not isinstance(payload, bytes) or not payload:
        raise TypeError("blind audit stdout payload must be nonempty bytes")
    expected_payload = f"{blind_audit_output}\n".encode()
    if payload != expected_payload:
        _fail("blind audit stdout payload differs from evaluator completion")
    stream = sys.stdout
    flush = getattr(stream, "flush", None)
    binary = getattr(stream, "buffer", None)
    write = getattr(binary, "write", None)
    binary_flush = getattr(binary, "flush", None)
    if not callable(flush) or not callable(write) or not callable(binary_flush):
        _fail("blind audit stdout must expose a writable binary buffer")
    flush()
    written = write(payload)
    binary_flush()
    if type(written) is not int or written != len(payload):
        _fail("blind audit stdout did not accept the complete byte payload")
    capability = object()
    with _BLIND_AUDIT_CAPABILITY_LOCK:
        _BLIND_AUDIT_STDOUT_CAPABILITIES[id(capability)] = (
            capability,
            completion_claim_sha256,
            blind_audit_output,
            evaluated,
            sha256_bytes(payload),
            os.getpid(),
            get_ident(),
        )
    return capability


def _validate_blind_audit_execution_receipt(
    receipt: object,
    *,
    attempt_claim_sha256: str,
    blind_audit_output: str,
    selection_input_sha256: str,
) -> None:
    if type(receipt) is not dict or frozenset(receipt) != frozenset(
        {
            "attempt_claim",
            "attempt_claim_sha256",
            "contract",
            "execution",
            "execution_history_external_attestation_verified",
            "local_workflow_guard_only",
            "malicious_workspace_owner_resistance_verified",
            "receipt_sha256",
            "retry_prevention_external_custody_verified",
            "schema_version",
        }
    ):
        _fail("blind audit execution receipt has the wrong schema")
    if (
        receipt["schema_version"] != "policytwin.exact-finite-blind-audit-execution.v4"
        or receipt["execution_history_external_attestation_verified"] is not False
        or receipt["local_workflow_guard_only"] is not True
        or receipt["malicious_workspace_owner_resistance_verified"] is not False
        or receipt["retry_prevention_external_custody_verified"] is not False
    ):
        _fail("blind audit execution receipt overstates its evidence authority")
    expected_attempt_claim_sha256 = _sha256(
        attempt_claim_sha256,
        "blind audit expected attempt_claim_sha256",
    )
    if receipt["attempt_claim_sha256"] != expected_attempt_claim_sha256:
        _fail("blind audit execution receipt binds a different attempt claim")
    attempt_claim = receipt["attempt_claim"]
    _validate_blind_audit_attempt_claim(
        attempt_claim,
        claim_sha256=expected_attempt_claim_sha256,
        selection_input_sha256=selection_input_sha256,
    )
    contract = receipt["contract"]
    if contract != derive_exact_finite_scenario_blind_audit_output_receipt(
        blind_audit_output,
        attempt_claim_sha256=expected_attempt_claim_sha256,
        selection_input_sha256=selection_input_sha256,
    ):
        _fail("blind audit execution receipt contract does not rederive")
    execution = receipt["execution"]
    expected_execution_fields = frozenset(
        {
            "candidate_loop_completed",
            "canonical_direct_cli_entry",
            "cli_source_sha256",
            "evaluated_candidate_count",
            "evaluated_candidates_sha256",
            "execution_nonce_sha256",
            "ignore_environment",
            "isolated",
            "attempt_claimed_at_utc",
            "loaded_source_manifest_sha256",
            "local_completion_capability_consumed",
            "no_site",
            "orig_argv_sha256",
            "python_cache_tag",
            "python_executable_sha256",
            "python_implementation",
            "python_version",
            "safe_path",
            "selection_rule_order_verified",
            "site_modules_absent",
            "stdout_write_and_flush_completed",
        }
    )
    if type(execution) is not dict or frozenset(execution) != expected_execution_fields:
        _fail("blind audit execution identity has the wrong schema")
    if execution != _blind_audit_execution_mapping_from_claim(
        cast(dict[str, object], attempt_claim),
        blind_audit_output=blind_audit_output,
    ):
        _fail("blind audit execution identity differs from its attempt claim")
    for field in (
        "ignore_environment",
        "isolated",
        "candidate_loop_completed",
        "canonical_direct_cli_entry",
        "local_completion_capability_consumed",
        "no_site",
        "safe_path",
        "selection_rule_order_verified",
        "site_modules_absent",
        "stdout_write_and_flush_completed",
    ):
        if execution[field] is not True:
            _fail("blind audit execution was not isolated and completed")
    for field in (
        "cli_source_sha256",
        "evaluated_candidates_sha256",
        "execution_nonce_sha256",
        "loaded_source_manifest_sha256",
        "orig_argv_sha256",
        "python_executable_sha256",
    ):
        _sha256(execution[field], f"blind audit execution {field}")
    if execution["cli_source_sha256"] != sha256_file(_blind_audit_cli_source_path()):
        _fail("blind audit execution CLI source commitment is stale")
    for field in ("python_cache_tag", "python_implementation", "python_version"):
        if not isinstance(execution[field], str) or not execution[field]:
            _fail(f"blind audit execution {field} must be a nonempty string")
    if (
        execution["python_cache_tag"] != sys.implementation.cache_tag
        or execution["python_implementation"] != sys.implementation.name
        or execution["python_version"] != sys.version
    ):
        _fail("blind audit execution Python runtime differs from validation runtime")
    live_executable = Path(sys.executable).resolve()
    if not live_executable.is_file() or execution[
        "python_executable_sha256"
    ] != sha256_file(live_executable):
        _fail("blind audit execution Python executable commitment is stale")
    claimed_at = execution["attempt_claimed_at_utc"]
    if not isinstance(claimed_at, str):
        _fail("blind audit execution attempt_claimed_at_utc must be a string")
    try:
        claimed = datetime.fromisoformat(claimed_at)
    except ValueError as error:
        raise ExactFinitePolicyFunctionalError(
            "blind audit execution attempt_claimed_at_utc is invalid"
        ) from error
    _aware(claimed, "blind audit execution attempt_claimed_at_utc")
    payload = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if receipt["receipt_sha256"] != _digest(payload):
        _fail("blind audit execution receipt seal does not verify")


def _require_durable_blind_attempt_matches_receipt(
    receipt: object,
    *,
    expected_claim_sha256: str,
) -> None:
    """Bind finalized authorization to the actual no-follow claim bytes."""
    claim, claim_sha256 = _read_bounded_blind_audit_attempt_claim()
    expected = _sha256(
        expected_claim_sha256,
        "finalized durable blind-audit attempt claim",
    )
    if claim_sha256 != expected:
        _fail("finalized durable blind-audit attempt claim hash changed")
    if type(receipt) is not dict or receipt.get("attempt_claim") != claim:
        _fail("finalized receipt differs from the durable blind-audit attempt claim")


def _require_pending_epoch_has_no_attempt_claim() -> None:
    """Reject a stale pending JSON after the durable epoch claim was consumed."""
    try:
        os.lstat(_blind_attempt_claim_path())
    except FileNotFoundError:
        return
    except OSError as error:
        raise ExactFinitePolicyFunctionalError(
            "cannot determine whether the pending blind-audit epoch was consumed"
        ) from error
    _fail(
        "pending selection has a durable attempt claim; the epoch is "
        "consumed_unfinalized and must not be rerun"
    )


def validate_exact_finite_scenario_design_manifest(
    manifest: Mapping[str, object],
    *,
    require_finalized: bool,
) -> bool:
    """Strictly validate the pending or finalized bounded-search manifest.

    The return value is ``True`` only for a finalized selected fixture that
    authorizes production.  A completed ``no-match`` audit validates but
    returns ``False``.  Production callers must pass ``require_finalized=True``
    and separately require the returned authorization Boolean.
    """
    if type(manifest) is not dict or not all(type(key) is str for key in manifest):
        _fail("scenario design manifest must be an object with string keys")
    if not isinstance(require_finalized, bool):
        raise TypeError("require_finalized must be boolean")
    expected_fields = frozenset(
        {
            "audit_epoch",
            "blind_audit_attempt_claim_sha256",
            "blind_audit_output",
            "blind_audit_output_receipt",
            "blind_audit_selection_input_sha256",
            "audited_design_sha256",
            "candidate_seeds",
            "candidate_statuses",
            "excluded_exposed_pilot",
            "excluded_stale_pre_freeze_audit",
            "excluded_superseded_source_audit",
            "production_authorized",
            "production_runtime_search",
            "schema_version",
            "selected_seed",
            "selection_input_snapshot_source_sha256",
            "selection_state_source_sha256",
            "selection_rule",
            "selection_uses_only_nondegeneracy",
            "synthetic_test_fixture_design_only",
        }
    )
    if frozenset(manifest) != expected_fields:
        _fail("scenario design manifest has missing or unknown fields")
    raw_candidates = manifest["candidate_seeds"]
    raw_statuses = manifest["candidate_statuses"]
    if (
        type(raw_candidates) is not list
        or tuple(raw_candidates)
        != POLICY_FUNCTIONAL_EXACT_FINITE_SCENARIO_CANDIDATE_SEEDS
    ):
        _fail("scenario design candidate order differs from the frozen grid")
    if (
        type(raw_statuses) is not list
        or len(raw_statuses) != len(raw_candidates)
        or not all(type(item) is str for item in raw_statuses)
    ):
        _fail("scenario design statuses have the wrong schema")
    statuses = tuple(raw_statuses)
    selection_state_source_sha256 = _sha256(
        manifest["selection_state_source_sha256"],
        "scenario selection_state_source_sha256",
    )
    _, live_selection_state_source_sha256 = (
        _load_exact_finite_scenario_selection_state()
    )
    if selection_state_source_sha256 != live_selection_state_source_sha256:
        _fail("scenario selection-state source commitment is stale")
    selection_input_snapshot_source_sha256 = _sha256(
        manifest["selection_input_snapshot_source_sha256"],
        "scenario selection_input_snapshot_source_sha256",
    )
    _, live_selection_input_snapshot_sha256 = (
        _load_exact_finite_scenario_selection_input_snapshot()
    )
    if selection_input_snapshot_source_sha256 != live_selection_input_snapshot_sha256:
        _fail("scenario pending-input snapshot commitment is stale")
    if (
        manifest["schema_version"] != "policytwin.exact-finite-scenario-design.v2"
        or manifest["audit_epoch"] != _SCENARIO_AUDIT_EPOCH
        or manifest["selection_rule"]
        != POLICY_FUNCTIONAL_EXACT_FINITE_SCENARIO_SELECTION_RULE
        or manifest["production_runtime_search"] is not False
        or manifest["selection_uses_only_nondegeneracy"] is not True
        or manifest["synthetic_test_fixture_design_only"] is not True
    ):
        _fail("scenario design scientific contract changed")

    # Validate the status/selection relationship before the whole-design digest.
    # This preserves a useful fail-closed witness when a stale historical
    # manifest is additionally forged: structural selection faults should not
    # be hidden behind the more general stale-design error.
    selected_seed = manifest["selected_seed"]
    pending_statuses = tuple("pending_post_freeze_blind_audit" for _ in raw_candidates)
    no_match_statuses = derive_exact_finite_scenario_audit_statuses("no-match")
    selected_status = "blind_audit_nondegenerate_selected"
    if statuses not in (pending_statuses, no_match_statuses):
        if statuses.count(selected_status) != 1:
            _fail("final scenario design must contain exactly one selected candidate")
        if selected_seed not in raw_candidates:
            _fail("final scenario seed is outside the frozen candidate grid")
        if manifest["production_authorized"] is not True:
            _fail("selected scenario design must explicitly authorize production")
        expected_statuses = derive_exact_finite_scenario_audit_statuses(selected_seed)
        if statuses != expected_statuses:
            _fail("scenario status vector does not align with the selected seed")
    if (
        manifest["audited_design_sha256"]
        != exact_finite_scenario_audited_design_sha256()
    ):
        _fail("scenario design target differs from the frozen audited design")
    exposed = manifest["excluded_exposed_pilot"]
    stale = manifest["excluded_stale_pre_freeze_audit"]
    superseded = manifest["excluded_superseded_source_audit"]
    if type(exposed) is not dict or exposed != {
        "candidate_grid": list(_EXCLUDED_EXPOSED_PILOT_CANDIDATES),
        "effect_values_exposed_for": [
            "exact-finite-policy-functional-grid-a-v1",
            "exact-finite-policy-functional-grid-b-v1",
        ],
        "reason": "effect_values_were_printed_so_pilot_cannot_select_fixture",
    }:
        _fail("excluded exposed pilot disclosure changed")
    if type(stale) is not dict or stale != {
        "candidate_grid": list(_EXCLUDED_STALE_AUDIT_CANDIDATES),
        "emitted_seed_only": "exact-finite-policy-functional-grid-g-v1",
        "reason": "internal_receipt_schema_was_not_frozen_and_grid_is_now_exposed",
    }:
        _fail("excluded stale audit disclosure changed")
    if type(superseded) is not dict or superseded != _superseded_m_r_disclosure():
        _fail("excluded source-superseded m--r audit disclosure changed")
    if set(raw_candidates) & (
        set(_EXCLUDED_EXPOSED_PILOT_CANDIDATES)
        | set(_EXCLUDED_STALE_AUDIT_CANDIDATES)
        | set(_EXCLUDED_SUPERSEDED_SOURCE_AUDIT_CANDIDATES)
    ):
        _fail("production scenario candidates overlap an exposed grid")
    raw_attempt_claim_sha256 = manifest["blind_audit_attempt_claim_sha256"]
    if statuses == pending_statuses:
        _require_pending_epoch_has_no_attempt_claim()
        if selected_seed != POLICY_FUNCTIONAL_EXACT_FINITE_SELECTED_SCENARIO_SEED:
            _fail("pending scenario seed differs from the source placeholder")
        if manifest["production_authorized"] is not False:
            _fail("pending scenario design cannot authorize production")
        if manifest["blind_audit_selection_input_sha256"] is not None:
            _fail("pending scenario design cannot claim a completed input digest")
        if raw_attempt_claim_sha256 is not None:
            _fail("pending scenario design cannot claim an audit attempt")
        if manifest["blind_audit_output"] != "pending_post_freeze_blind_audit":
            _fail("pending scenario design has an inconsistent audit output")
        if manifest["blind_audit_output_receipt"] != {
            "status": "pending_post_freeze_blind_audit"
        }:
            _fail("pending scenario design has an inconsistent output receipt")
        if selection_state_source_sha256 != selection_input_snapshot_source_sha256:
            _fail("pending scenario state differs from its immutable input snapshot")
        if require_finalized:
            _fail("production scenario design is still pending blind audit")
        return False
    if statuses == no_match_statuses:
        attempt_claim_sha256 = _sha256(
            raw_attempt_claim_sha256,
            "completed blind-audit attempt claim",
        )
        selection_input_sha256 = _sha256(
            manifest["blind_audit_selection_input_sha256"],
            "completed blind-audit selection input",
        )
        if selection_input_sha256 != selection_input_snapshot_source_sha256:
            _fail("completed blind audit does not bind the frozen pending input")
        if selected_seed is not None:
            _fail("completed no-match audit cannot name a selected seed")
        if manifest["blind_audit_output"] != "no-match":
            _fail("completed no-match audit has inconsistent stdout")
        _validate_blind_audit_execution_receipt(
            manifest["blind_audit_output_receipt"],
            attempt_claim_sha256=attempt_claim_sha256,
            blind_audit_output="no-match",
            selection_input_sha256=selection_input_sha256,
        )
        _require_durable_blind_attempt_matches_receipt(
            manifest["blind_audit_output_receipt"],
            expected_claim_sha256=attempt_claim_sha256,
        )
        if manifest["production_authorized"] is not False:
            _fail("completed no-match audit cannot authorize production")
        return False
    if statuses.count(selected_status) != 1:
        _fail("final scenario design must contain exactly one selected candidate")
    if selected_seed not in raw_candidates:
        _fail("final scenario seed is outside the frozen candidate grid")
    if selected_seed != POLICY_FUNCTIONAL_EXACT_FINITE_SELECTED_SCENARIO_SEED:
        _fail("scenario design selected seed differs from the frozen constant")
    if manifest["production_authorized"] is not True:
        _fail("selected scenario design must explicitly authorize production")
    selection_input_sha256 = _sha256(
        manifest["blind_audit_selection_input_sha256"],
        "completed blind-audit selection input",
    )
    if selection_input_sha256 != selection_input_snapshot_source_sha256:
        _fail("completed blind audit does not bind the frozen pending input")
    attempt_claim_sha256 = _sha256(
        raw_attempt_claim_sha256,
        "completed blind-audit attempt claim",
    )
    expected_statuses = derive_exact_finite_scenario_audit_statuses(selected_seed)
    if statuses != expected_statuses:
        _fail("scenario status vector does not align with the selected seed")
    if manifest["blind_audit_output"] != selected_seed:
        _fail("final blind-audit output does not equal the selected seed")
    _validate_blind_audit_execution_receipt(
        manifest["blind_audit_output_receipt"],
        attempt_claim_sha256=attempt_claim_sha256,
        blind_audit_output=selected_seed,
        selection_input_sha256=selection_input_sha256,
    )
    _require_durable_blind_attempt_matches_receipt(
        manifest["blind_audit_output_receipt"],
        expected_claim_sha256=attempt_claim_sha256,
    )
    return True


def _binary64_exact(value: ExactRational, field: str) -> tuple[float, str]:
    fraction = _fraction(value, field)
    as_float = float(fraction)
    if not math.isfinite(as_float) or Fraction.from_float(as_float) != fraction:
        _fail(f"{field} must be exactly representable as binary64")
    return as_float, as_float.hex()


def _exact_from_binary64(value: float, field: str) -> ExactRational:
    if not isinstance(value, float) or not math.isfinite(value):
        _fail(f"{field} must be a finite binary64 value")
    numerator, denominator = value.as_integer_ratio()
    return ExactRational.from_ratio(numerator, denominator)


class ExactFinitePosteriorMode(StrEnum):
    """Closed posterior implementations compared by the finite canary."""

    EXACT_BAYES = "exact_bayes"
    IGNORED_DATA_PRIOR = "ignored_data_prior"


class ExactFiniteCanaryDecision(StrEnum):
    """Engineering decision for the closed synthetic canary only."""

    PASS = "pass"
    FAIL = "fail"
    ABSTAIN = "abstain"


class ExactFiniteNoOpZeroCheck(StrEnum):
    """Cellwise exact-zero status for the no-op policy invariant."""

    NOT_APPLICABLE = "not_applicable"
    PASSED = "passed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ExactFiniteParameterState:
    """One finite parameter state with an exact binary64 transport."""

    state_id: str
    behavior_scale: ExactRational
    signal_scale: ExactRational
    behavior_scale_binary64_hex: str
    signal_scale_binary64_hex: str

    def __post_init__(self) -> None:
        _identifier(self.state_id, "state_id")
        behavior, behavior_hex = _binary64_exact(self.behavior_scale, "behavior_scale")
        signal, signal_hex = _binary64_exact(self.signal_scale, "signal_scale")
        if not 0.0 <= behavior <= 1.0:
            _fail("behavior_scale must lie in the integrated model domain [0, 1]")
        if not 0.0 < signal <= 5.0:
            _fail("signal_scale must lie in the integrated model domain (0, 5]")
        if self.behavior_scale_binary64_hex != behavior_hex:
            _fail("behavior_scale binary64 hex does not match its exact rational")
        if self.signal_scale_binary64_hex != signal_hex:
            _fail("signal_scale binary64 hex does not match its exact rational")

    @classmethod
    def from_exact(
        cls,
        *,
        state_id: str,
        behavior_scale: ExactRational,
        signal_scale: ExactRational,
    ) -> Self:
        """Build one state only when both values have lossless binary64 forms."""
        _, behavior_hex = _binary64_exact(behavior_scale, "behavior_scale")
        _, signal_hex = _binary64_exact(signal_scale, "signal_scale")
        return cls(
            state_id=state_id,
            behavior_scale=behavior_scale,
            signal_scale=signal_scale,
            behavior_scale_binary64_hex=behavior_hex,
            signal_scale_binary64_hex=signal_hex,
        )

    @property
    def parameters(self) -> dict[str, float]:
        return {
            "behavior_scale": float.fromhex(self.behavior_scale_binary64_hex),
            "signal_scale": float.fromhex(self.signal_scale_binary64_hex),
        }

    def to_mapping(self) -> dict[str, object]:
        return {
            "behavior_scale": self.behavior_scale.to_mapping(),
            "behavior_scale_binary64_hex": self.behavior_scale_binary64_hex,
            "signal_scale": self.signal_scale.to_mapping(),
            "signal_scale_binary64_hex": self.signal_scale_binary64_hex,
            "state_id": self.state_id,
        }


@dataclass(frozen=True, slots=True)
class ExactPosteriorMassRow:
    """Exact likelihood, marginal observation mass, and posterior for one y."""

    observation_value: int
    observation_probability: ExactRational
    likelihood_by_state: tuple[tuple[str, ExactRational], ...]
    posterior_by_state: tuple[tuple[str, ExactRational], ...]

    def __post_init__(self) -> None:
        if isinstance(self.observation_value, bool) or not isinstance(
            self.observation_value, int
        ):
            _fail("observation_value must be an integer category")
        if not Fraction() < _fraction(self.observation_probability) <= Fraction(1):
            _fail("observation_probability must lie in (0, 1]")
        for field in ("likelihood_by_state", "posterior_by_state"):
            values = getattr(self, field)
            if (
                not isinstance(values, tuple)
                or not values
                or not all(
                    isinstance(item, tuple)
                    and len(item) == 2
                    and isinstance(item[0], str)
                    and type(item[1]) is ExactRational
                    for item in values
                )
            ):
                _fail(f"{field} must be an immutable exact state-probability table")
            state_ids = tuple(item[0] for item in values)
            if len(set(state_ids)) != len(state_ids):
                _fail(f"{field} contains duplicate parameter states")
            if any(
                not Fraction() <= _fraction(item[1]) <= Fraction(1) for item in values
            ):
                _fail(f"{field} contains a value outside [0, 1]")
        if tuple(item[0] for item in self.likelihood_by_state) != tuple(
            item[0] for item in self.posterior_by_state
        ):
            _fail("likelihood and posterior state orders differ")
        if (
            sum((_fraction(item[1]) for item in self.posterior_by_state), Fraction())
            != 1
        ):
            _fail("posterior probabilities must sum exactly to one")

    def posterior(self, state_id: str) -> Fraction:
        try:
            return next(
                _fraction(value)
                for candidate, value in self.posterior_by_state
                if candidate == state_id
            )
        except StopIteration as error:
            raise ExactFinitePolicyFunctionalError(
                f"posterior row has no state {state_id!r}"
            ) from error

    def to_mapping(self) -> dict[str, object]:
        return {
            "likelihood_by_state": [
                {"probability": probability.to_mapping(), "state_id": state_id}
                for state_id, probability in self.likelihood_by_state
            ],
            "observation_probability": self.observation_probability.to_mapping(),
            "observation_value": self.observation_value,
            "posterior_by_state": [
                {"probability": probability.to_mapping(), "state_id": state_id}
                for state_id, probability in self.posterior_by_state
            ],
        }


@dataclass(frozen=True, slots=True)
class ExactRankEnumerationAtom:
    """One atom in the complete truth/data/draw/tie-position enumeration."""

    posterior_mode: ExactFinitePosteriorMode
    truth_state_id: str
    observation_support_index: int
    observation_value: int
    posterior_state_ids: tuple[str, ...]
    tie_position: int
    rank: int
    joint_probability: ExactRational

    def __post_init__(self) -> None:
        if not isinstance(self.posterior_mode, ExactFinitePosteriorMode):
            _fail("rank atom posterior mode has the wrong type")
        _identifier(self.truth_state_id, "truth_state_id")
        if (
            isinstance(self.observation_support_index, bool)
            or not isinstance(self.observation_support_index, int)
            or self.observation_support_index < 0
        ):
            _fail("observation_support_index must be non-negative")
        if (
            not isinstance(self.posterior_state_ids, tuple)
            or not self.posterior_state_ids
        ):
            _fail("rank atom must contain posterior state identities")
        for state_id in self.posterior_state_ids:
            _identifier(state_id, "posterior state_id")
        for value, field in ((self.tie_position, "tie_position"), (self.rank, "rank")):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                _fail(f"{field} must be non-negative")
        if not Fraction() < _fraction(self.joint_probability) <= Fraction(1):
            _fail("rank atom joint probability must lie in (0, 1]")

    def to_mapping(self) -> dict[str, object]:
        return {
            "joint_probability": self.joint_probability.to_mapping(),
            "observation_support_index": self.observation_support_index,
            "observation_value": self.observation_value,
            "posterior_mode": self.posterior_mode.value,
            "posterior_state_ids": list(self.posterior_state_ids),
            "rank": self.rank,
            "tie_position": self.tie_position,
            "truth_state_id": self.truth_state_id,
        }


@dataclass(frozen=True, slots=True)
class ExactConditionalRankLaw:
    """Exact randomized-rank PMF conditional on one observation category."""

    posterior_mode: ExactFinitePosteriorMode
    observation_value: int
    observation_probability: ExactRational
    rank_probabilities: tuple[ExactRational, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.posterior_mode, ExactFinitePosteriorMode):
            _fail("conditional law posterior mode has the wrong type")
        if (
            not isinstance(self.rank_probabilities, tuple)
            or len(self.rank_probabilities) < 2
        ):
            _fail("conditional law requires at least two rank categories")
        if sum((_fraction(item) for item in self.rank_probabilities), Fraction()) != 1:
            _fail("conditional rank probabilities must sum exactly to one")

    @property
    def is_discrete_uniform(self) -> bool:
        expected = Fraction(1, len(self.rank_probabilities))
        return all(_fraction(item) == expected for item in self.rank_probabilities)

    def to_mapping(self) -> dict[str, object]:
        return {
            "observation_probability": self.observation_probability.to_mapping(),
            "observation_value": self.observation_value,
            "posterior_mode": self.posterior_mode.value,
            "rank_probabilities": [
                item.to_mapping() for item in self.rank_probabilities
            ],
        }


@dataclass(frozen=True, slots=True)
class ExactFiniteRankLawDerivation:
    """Complete exact enumeration for one posterior implementation."""

    posterior_mode: ExactFinitePosteriorMode
    posterior_rows: tuple[ExactPosteriorMassRow, ...]
    atoms: tuple[ExactRankEnumerationAtom, ...]
    conditional_laws: tuple[ExactConditionalRankLaw, ...]
    unconditional_rank_probabilities: tuple[ExactRational, ...]
    posterior_draw_permutation_exchangeability_verified: bool
    conditional_truth_draw_exchangeability_exhaustively_verified: bool
    complete_support_probability: ExactRational

    def __post_init__(self) -> None:
        if not isinstance(self.posterior_mode, ExactFinitePosteriorMode):
            _fail("rank derivation posterior mode has the wrong type")
        if not self.posterior_rows or not self.atoms or not self.conditional_laws:
            _fail("rank derivation must retain every non-empty exact table")
        if not isinstance(
            self.posterior_draw_permutation_exchangeability_verified, bool
        ):
            _fail("exchangeability verification field must be boolean")
        if not isinstance(
            self.conditional_truth_draw_exchangeability_exhaustively_verified,
            bool,
        ):
            _fail("truth/draw exchangeability verification field must be boolean")
        if _fraction(self.complete_support_probability) != 1:
            _fail("rank derivation does not account for total probability one")
        if (
            sum(
                (_fraction(item) for item in self.unconditional_rank_probabilities),
                Fraction(),
            )
            != 1
        ):
            _fail("unconditional rank probabilities must sum exactly to one")

    @property
    def unconditional_is_discrete_uniform(self) -> bool:
        expected = Fraction(1, len(self.unconditional_rank_probabilities))
        return all(
            _fraction(item) == expected
            for item in self.unconditional_rank_probabilities
        )

    @property
    def all_conditionals_are_discrete_uniform(self) -> bool:
        return all(item.is_discrete_uniform for item in self.conditional_laws)

    def to_mapping(self) -> dict[str, object]:
        return {
            "atoms": [item.to_mapping() for item in self.atoms],
            "complete_support_probability": (
                self.complete_support_probability.to_mapping()
            ),
            "conditional_laws": [item.to_mapping() for item in self.conditional_laws],
            "conditional_truth_draw_exchangeability_exhaustively_verified": (
                self.conditional_truth_draw_exchangeability_exhaustively_verified
            ),
            "posterior_draw_permutation_exchangeability_verified": (
                self.posterior_draw_permutation_exchangeability_verified
            ),
            "posterior_mode": self.posterior_mode.value,
            "posterior_rows": [item.to_mapping() for item in self.posterior_rows],
            "unconditional_rank_probabilities": [
                item.to_mapping() for item in self.unconditional_rank_probabilities
            ],
        }


def _validate_exact_design_dimensions(
    *,
    parameter_state_count: int,
    observation_support_size: int,
    posterior_draw_count: int,
    scenario_replicate_count: int,
) -> None:
    state_count = _positive_int(
        parameter_state_count, "parameter_state_count", minimum=2
    )
    support_size = _positive_int(
        observation_support_size, "observation_support_size", minimum=2
    )
    draw_count = _positive_int(posterior_draw_count, "posterior_draw_count", minimum=1)
    scenario_count = _positive_int(
        scenario_replicate_count, "scenario_replicate_count", minimum=1
    )
    if state_count > _MAX_PARAMETER_STATES:
        _fail("parameter_state_count exceeds the exact finite resource bound")
    if support_size > _MAX_SUPPORT_SIZE:
        _fail("observation_support_size exceeds the exact finite resource bound")
    if draw_count > _MAX_POSTERIOR_DRAWS:
        _fail("posterior_draw_count exceeds the exact finite resource bound")
    if scenario_count > _MAX_SCENARIOS:
        _fail("scenario_replicate_count exceeds the integrated resource bound")
    integrated_cells = state_count * (support_size + scenario_count)
    if integrated_cells > _MAX_INTEGRATED_CELLS:
        _fail("exact finite design exceeds the total integrated-cell budget")
    atom_upper_bound = (
        2 * state_count * support_size * (state_count**draw_count) * (draw_count + 1)
    )
    if atom_upper_bound > _MAX_TIE_EXPANDED_ATOMS:
        _fail("exact finite design exceeds the exhaustive tie-atom budget")


def _validate_integrated_execution_budget(
    *,
    parameter_state_count: int,
    observation_support_size: int,
    scenario_replicate_count: int,
    step_count: int,
) -> None:
    states = _positive_int(
        parameter_state_count,
        "parameter_state_count",
        minimum=2,
    )
    support = _positive_int(
        observation_support_size,
        "observation_support_size",
        minimum=2,
    )
    scenarios = _positive_int(
        scenario_replicate_count,
        "scenario_replicate_count",
    )
    steps = _positive_int(step_count, "step_count", minimum=2)
    cell_steps = states * (support + scenarios) * steps
    if cell_steps > _MAX_INTEGRATED_CELL_STEPS:
        _fail("exact finite design exceeds the integrated cell-step budget")


def _posterior_rows(
    *,
    state_ids: tuple[str, ...],
    observation_categories: tuple[tuple[str, tuple[int, ...]], ...],
) -> tuple[ExactPosteriorMassRow, ...]:
    by_state = dict(observation_categories)
    support_size = len(by_state[state_ids[0]])
    all_categories = tuple(
        sorted({value for state_id in state_ids for value in by_state[state_id]})
    )
    rows: list[ExactPosteriorMassRow] = []
    for observation in all_categories:
        counts = tuple(by_state[state_id].count(observation) for state_id in state_ids)
        total_count = sum(counts)
        if total_count <= 0:
            _fail("internal observation category has zero probability")
        rows.append(
            ExactPosteriorMassRow(
                observation_value=observation,
                observation_probability=_exact(
                    Fraction(total_count, len(state_ids) * support_size)
                ),
                likelihood_by_state=tuple(
                    (state_id, _exact(Fraction(count, support_size)))
                    for state_id, count in zip(state_ids, counts, strict=True)
                ),
                posterior_by_state=tuple(
                    (state_id, _exact(Fraction(count, total_count)))
                    for state_id, count in zip(state_ids, counts, strict=True)
                ),
            )
        )
    if sum((_fraction(row.observation_probability) for row in rows), Fraction()) != 1:
        _fail("observation marginal probabilities do not sum to one")
    return tuple(rows)


def derive_exact_finite_rank_laws(
    *,
    observation_categories: tuple[tuple[str, tuple[int, ...]], ...],
    functional_values: tuple[tuple[str, ExactRational], ...],
    posterior_draw_count: int,
    posterior_mode: ExactFinitePosteriorMode,
) -> ExactFiniteRankLawDerivation:
    """Exhaust the exact truth/data/posterior/tie-position probability law.

    ``observation_categories`` declares the equally weighted finite support for
    each equally likely parameter state.  ``functional_values`` supplies one
    exact integrated policy functional per state.  No data are sampled or
    discarded by this function.
    """
    if not isinstance(posterior_mode, ExactFinitePosteriorMode):
        raise TypeError("posterior_mode must be an ExactFinitePosteriorMode")
    if (
        not isinstance(observation_categories, tuple)
        or len(observation_categories) < 2
        or not isinstance(functional_values, tuple)
    ):
        _fail("finite rank derivation requires immutable state tables")
    state_ids = tuple(item[0] for item in observation_categories)
    if len(set(state_ids)) != len(state_ids):
        _fail("observation table contains duplicate state IDs")
    for state_id in state_ids:
        _identifier(state_id, "observation state_id")
    if tuple(item[0] for item in functional_values) != state_ids:
        _fail("functional state order must exactly match observation state order")
    support_sizes: list[int] = []
    for state_id, support in observation_categories:
        if (
            not isinstance(support, tuple)
            or not support
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in support
            )
        ):
            _fail(f"observation support for {state_id!r} must contain integers")
        support_sizes.append(len(support))
    if len(set(support_sizes)) != 1:
        _fail("every state must use the same declared observation support size")
    for state_id, value in functional_values:
        _identifier(state_id, "functional state_id")
        _fraction(value, f"functional value for {state_id}")
    _validate_exact_design_dimensions(
        parameter_state_count=len(state_ids),
        observation_support_size=support_sizes[0],
        posterior_draw_count=posterior_draw_count,
        scenario_replicate_count=1,
    )
    rows = _posterior_rows(
        state_ids=state_ids,
        observation_categories=observation_categories,
    )
    row_by_observation = {row.observation_value: row for row in rows}
    functional = {
        state_id: _fraction(value, f"functional value for {state_id}")
        for state_id, value in functional_values
    }
    support_by_state = dict(observation_categories)
    state_prior = _uniform(len(state_ids))
    support_prior = _uniform(support_sizes[0])
    atoms: list[ExactRankEnumerationAtom] = []
    posterior_draws_exchangeable = True

    for truth_state_id in state_ids:
        truth_functional = functional[truth_state_id]
        for support_index, observation in enumerate(support_by_state[truth_state_id]):
            row = row_by_observation[observation]
            posterior = {
                state_id: (
                    row.posterior(state_id)
                    if posterior_mode is ExactFinitePosteriorMode.EXACT_BAYES
                    else state_prior
                )
                for state_id in state_ids
            }
            for draw_state_ids in product(state_ids, repeat=posterior_draw_count):
                draw_probability = math.prod(
                    posterior[state_id] for state_id in draw_state_ids
                )
                if draw_probability == 0:
                    continue
                for permuted in set(permutations(draw_state_ids)):
                    permuted_probability = math.prod(
                        posterior[state_id] for state_id in permuted
                    )
                    if permuted_probability != draw_probability:
                        posterior_draws_exchangeable = False
                less_count = sum(
                    functional[state_id] < truth_functional
                    for state_id in draw_state_ids
                )
                equal_count = sum(
                    functional[state_id] == truth_functional
                    for state_id in draw_state_ids
                )
                tie_mass = Fraction(1, equal_count + 1)
                joint = state_prior * support_prior * draw_probability * tie_mass
                for tie_position in range(equal_count + 1):
                    atoms.append(
                        ExactRankEnumerationAtom(
                            posterior_mode=posterior_mode,
                            truth_state_id=truth_state_id,
                            observation_support_index=support_index,
                            observation_value=observation,
                            posterior_state_ids=tuple(draw_state_ids),
                            tie_position=tie_position,
                            rank=less_count + tie_position,
                            joint_probability=_exact(joint),
                        )
                    )
    if len(atoms) > _MAX_TIE_EXPANDED_ATOMS:
        _fail("actual exhaustive enumeration exceeds the tie-atom resource bound")
    total_mass = sum((_fraction(item.joint_probability) for item in atoms), Fraction())
    if total_mass != 1:
        _fail("exhaustive rank atoms do not account for probability one")

    rank_count = posterior_draw_count + 1
    conditional_laws: list[ExactConditionalRankLaw] = []
    for row in rows:
        observation_mass = _fraction(row.observation_probability)
        probabilities = tuple(
            sum(
                (
                    _fraction(atom.joint_probability)
                    for atom in atoms
                    if atom.observation_value == row.observation_value
                    and atom.rank == rank
                ),
                Fraction(),
            )
            / observation_mass
            for rank in range(rank_count)
        )
        conditional_laws.append(
            ExactConditionalRankLaw(
                posterior_mode=posterior_mode,
                observation_value=row.observation_value,
                observation_probability=row.observation_probability,
                rank_probabilities=tuple(_exact(value) for value in probabilities),
            )
        )
    unconditional = tuple(
        sum(
            (_fraction(atom.joint_probability) for atom in atoms if atom.rank == rank),
            Fraction(),
        )
        for rank in range(rank_count)
    )
    truth_draw_exchangeable = True
    for row in rows:
        truth_distribution = {
            state_id: row.posterior(state_id) for state_id in state_ids
        }
        draw_distribution = {
            state_id: (
                row.posterior(state_id)
                if posterior_mode is ExactFinitePosteriorMode.EXACT_BAYES
                else state_prior
            )
            for state_id in state_ids
        }
        for joint_states in product(state_ids, repeat=posterior_draw_count + 1):
            probability = truth_distribution[joint_states[0]] * math.prod(
                draw_distribution[state_id] for state_id in joint_states[1:]
            )
            for permuted in set(permutations(joint_states)):
                permuted_probability = truth_distribution[permuted[0]] * math.prod(
                    draw_distribution[state_id] for state_id in permuted[1:]
                )
                if permuted_probability != probability:
                    truth_draw_exchangeable = False
    return ExactFiniteRankLawDerivation(
        posterior_mode=posterior_mode,
        posterior_rows=rows,
        atoms=tuple(atoms),
        conditional_laws=tuple(conditional_laws),
        unconditional_rank_probabilities=tuple(
            _exact(value) for value in unconditional
        ),
        posterior_draw_permutation_exchangeability_verified=(
            posterior_draws_exchangeable
        ),
        conditional_truth_draw_exchangeability_exhaustively_verified=(
            truth_draw_exchangeable
        ),
        complete_support_probability=_exact(total_mass),
    )


@dataclass(frozen=True, slots=True)
class ExactFiniteObservationCell:
    """One integrated pre-policy summary cell on the complete finite support."""

    state_id: str
    support_index: int
    event_key: str
    summary_sha256: str
    observation_value: int
    heldout_quoted_spread: ExactRational

    def __post_init__(self) -> None:
        _identifier(self.state_id, "observation cell state_id")
        if (
            isinstance(self.support_index, bool)
            or not isinstance(self.support_index, int)
            or self.support_index < 0
        ):
            _fail("observation support_index must be non-negative")
        _event_key(self.event_key, "observation event_key")
        _sha256(self.summary_sha256, "observation summary_sha256")
        if (
            isinstance(self.observation_value, bool)
            or not isinstance(self.observation_value, int)
            or abs(self.observation_value) > _MAX_SAFE_INTEGER
        ):
            _fail("observation category must be an exact binary64-safe integer")
        _fraction(self.heldout_quoted_spread, "heldout_quoted_spread")

    def to_mapping(self) -> dict[str, object]:
        return {
            "event_key": self.event_key,
            "heldout_quoted_spread": self.heldout_quoted_spread.to_mapping(),
            "observation_value": self.observation_value,
            "state_id": self.state_id,
            "summary_sha256": self.summary_sha256,
            "support_index": self.support_index,
        }


@dataclass(frozen=True, slots=True)
class ExactFinitePolicyFunctionalCell:
    """One protocol-bound integrated lookup-table cell."""

    state_id: str
    scenario_index: int
    event_key: str
    scenario_generation_receipt_sha256: str
    shock_tape_sha256: str
    declared_event_keys_sha256: str
    baseline_consumption_manifest_sha256: str
    candidate_consumption_manifest_sha256: str
    pair_integrity_audit_sha256: str
    protocol_execution_receipt_sha256: str
    protocol_bound_artifact_sha256: str
    ledger_evidence_sha256: str
    internal_random_coupling_audit_sha256: str
    internal_random_role_neutral_ledger_sha256: str
    internal_random_expected_entry_count: int
    functional_value: ExactRational
    protocol_bound_runner_used: Literal[True] = True
    scenario_generation_receipt_live_replayed: Literal[True] = True
    paired_shock_consumption_identical: Literal[True] = True
    cross_world_internal_random_coupling_verified: Literal[True] = True
    internal_random_coupling_external_validity_evidence: Literal[False] = False

    def __post_init__(self) -> None:
        _identifier(self.state_id, "policy cell state_id")
        if (
            isinstance(self.scenario_index, bool)
            or not isinstance(self.scenario_index, int)
            or self.scenario_index < 0
        ):
            _fail("scenario_index must be non-negative")
        _event_key(self.event_key, "policy cell event_key")
        for field in (
            "scenario_generation_receipt_sha256",
            "shock_tape_sha256",
            "declared_event_keys_sha256",
            "baseline_consumption_manifest_sha256",
            "candidate_consumption_manifest_sha256",
            "pair_integrity_audit_sha256",
            "protocol_execution_receipt_sha256",
            "protocol_bound_artifact_sha256",
            "ledger_evidence_sha256",
            "internal_random_coupling_audit_sha256",
            "internal_random_role_neutral_ledger_sha256",
        ):
            _sha256(getattr(self, field), field)
        _fraction(self.functional_value, "functional_value")
        if (
            isinstance(self.internal_random_expected_entry_count, bool)
            or not isinstance(self.internal_random_expected_entry_count, int)
            or self.internal_random_expected_entry_count <= 0
        ):
            _fail("internal random audit entry count must be positive")
        if (
            self.protocol_bound_runner_used is not True
            or self.scenario_generation_receipt_live_replayed is not True
            or self.paired_shock_consumption_identical is not True
            or self.cross_world_internal_random_coupling_verified is not True
            or self.internal_random_coupling_external_validity_evidence is not False
        ):
            _fail("policy cell cannot weaken its formal execution evidence")
        if (
            self.baseline_consumption_manifest_sha256
            != self.candidate_consumption_manifest_sha256
        ):
            _fail("paired worlds consumed different exogenous realizations")

    def to_mapping(self) -> dict[str, object]:
        return {
            "baseline_consumption_manifest_sha256": (
                self.baseline_consumption_manifest_sha256
            ),
            "candidate_consumption_manifest_sha256": (
                self.candidate_consumption_manifest_sha256
            ),
            "declared_event_keys_sha256": self.declared_event_keys_sha256,
            "event_key": self.event_key,
            "functional_value": self.functional_value.to_mapping(),
            "internal_random_coupling_audit_sha256": (
                self.internal_random_coupling_audit_sha256
            ),
            "internal_random_coupling_external_validity_evidence": (
                self.internal_random_coupling_external_validity_evidence
            ),
            "internal_random_expected_entry_count": (
                self.internal_random_expected_entry_count
            ),
            "internal_random_role_neutral_ledger_sha256": (
                self.internal_random_role_neutral_ledger_sha256
            ),
            "ledger_evidence_sha256": self.ledger_evidence_sha256,
            "pair_integrity_audit_sha256": self.pair_integrity_audit_sha256,
            "paired_shock_consumption_identical": (
                self.paired_shock_consumption_identical
            ),
            "protocol_bound_artifact_sha256": self.protocol_bound_artifact_sha256,
            "protocol_bound_runner_used": self.protocol_bound_runner_used,
            "protocol_execution_receipt_sha256": (
                self.protocol_execution_receipt_sha256
            ),
            "scenario_generation_receipt_live_replayed": (
                self.scenario_generation_receipt_live_replayed
            ),
            "scenario_generation_receipt_sha256": (
                self.scenario_generation_receipt_sha256
            ),
            "scenario_index": self.scenario_index,
            "shock_tape_sha256": self.shock_tape_sha256,
            "state_id": self.state_id,
            "cross_world_internal_random_coupling_verified": (
                self.cross_world_internal_random_coupling_verified
            ),
        }


def _default_parameter_states() -> tuple[ExactFiniteParameterState, ...]:
    return (
        ExactFiniteParameterState.from_exact(
            state_id="theta_low",
            behavior_scale=ExactRational.from_ratio(1, 4),
            signal_scale=ExactRational.from_ratio(1, 2),
        ),
        ExactFiniteParameterState.from_exact(
            state_id="theta_high",
            behavior_scale=ExactRational.from_ratio(3, 4),
            signal_scale=ExactRational.from_ratio(2),
        ),
    )


def _observation_event_key(
    *, canary_id: str, seed: str, support_index: int
) -> EventKey:
    return EventKey(
        experiment_id=f"{canary_id}-observation",
        seed=seed,
        replicate_id=support_index,
        simulation_step=0,
        purpose="exact-finite-pre-policy-observation",
    )


def _policy_event_key(*, canary_id: str, seed: str, scenario_index: int) -> EventKey:
    return EventKey(
        experiment_id=f"{canary_id}-policy",
        seed=seed,
        replicate_id=scenario_index,
        simulation_step=0,
        purpose="exact-finite-policy-functional-scenario",
    )


def _module_source_path(module: object, field: str) -> Path:
    raw_path = getattr(module, "__file__", None)
    if not isinstance(raw_path, str):
        _fail(f"{field} module has no stable source path")
    path = Path(raw_path).resolve()
    if not path.is_file() or path.suffix != ".py":
        _fail(f"{field} module source is not a regular Python file")
    return path


@lru_cache(maxsize=1)
def _scenario_audit_git_lineage() -> dict[str, object]:
    root = Path(__file__).resolve().parents[3]
    git = collect_git_metadata(root)
    return {
        "commit": git.commit,
        "repository_root_present": git.root is not None,
        "policytwin_python_tree_and_declared_inputs_bound_separately": True,
    }


def _scenario_audit_source_tree_manifest() -> tuple[dict[str, object], ...]:
    root = Path(__file__).resolve().parents[3]
    source_root = root / "src" / "policytwin"
    paths = tuple(sorted(source_root.rglob("*.py")))
    if not paths or len(paths) > 512:
        _fail("immutable source-tree audit exceeded its file-count bound")
    material: list[dict[str, object]] = []
    total_bytes = 0
    for path in paths:
        if path.is_symlink():
            _fail("immutable source-tree audit forbids symbolic links")
        before = path.stat()
        if not stat.S_ISREG(before.st_mode) or before.st_size > 2_097_152:
            _fail("immutable source tree contains an invalid or oversized file")
        content = path.read_bytes()
        after = path.stat()
        if before.st_size != len(content) or (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            _fail("immutable source tree changed while it was hashed")
        total_bytes += len(content)
        if total_bytes > 33_554_432:
            _fail("immutable source-tree audit exceeded its total-byte bound")
        material.append(
            {
                "mode": before.st_mode & 0o777,
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_bytes(content),
                "size_bytes": len(content),
            }
        )
    return tuple(material)


def _scenario_audit_immutable_source_tree_sha256() -> str:
    first = _scenario_audit_source_tree_manifest()
    second = _scenario_audit_source_tree_manifest()
    if first != second:
        _fail("immutable source tree changed across its complete snapshot passes")
    return _digest(first)


def _scenario_audit_execution_lineage_mapping() -> dict[str, object]:
    root = Path(__file__).resolve().parents[3]
    return {
        "blind_audit_cli_source_sha256": sha256_file(
            root / "src" / "policytwin" / "exact_finite_blind_audit.py"
        ),
        "environment_lock_sha256": sha256_file(root / "uv.lock"),
        "exact_blind_audit_contract_version": 4,
        "exact_canary_source_sha256": sha256_file(Path(__file__).resolve()),
        "exact_canary_schema_version": POLICY_FUNCTIONAL_EXACT_FINITE_SCHEMA_VERSION,
        "git": _scenario_audit_git_lineage(),
        "integrated_pair_input_schema_version": INTEGRATED_PAIR_INPUT_SCHEMA_VERSION,
        "integrated_protocol_run_schema_version": (
            INTEGRATED_PROTOCOL_RUN_SCHEMA_VERSION
        ),
        "integrated_runner_source_sha256": sha256_file(
            _module_source_path(integrated_synthetic_module, "integrated runner")
        ),
        "immutable_source_tree_sha256": (
            _scenario_audit_immutable_source_tree_sha256()
        ),
        "internal_random_audit_schema_version": (
            internal_rng_audit_module.INTERNAL_RANDOM_COUPLING_AUDIT_SCHEMA_VERSION
        ),
        "internal_random_audit_source_sha256": sha256_file(
            _module_source_path(internal_rng_audit_module, "internal random audit")
        ),
        "ledger_derived_schema_version": LEDGER_DERIVED_SCHEMA_VERSION,
        "ledger_verifier_implementation_sha256": (
            ledger_evidence_verifier_implementation_sha256()
        ),
        "ledger_verifier_source_sha256": sha256_file(
            _module_source_path(ledger_derived_module, "ledger verifier")
        ),
        "project_config_sha256": sha256_file(root / "pyproject.toml"),
        "python_cache_tag": sys.implementation.cache_tag,
        "python_implementation": sys.implementation.name,
        "python_version": sys.version,
        "python_version_file_sha256": sha256_file(root / ".python-version"),
        "scenario_generation_draw_schema_version": (
            SCENARIO_GENERATION_DRAW_SCHEMA_VERSION
        ),
        "scenario_generation_receipt_schema_version": (
            SCENARIO_GENERATION_RECEIPT_SCHEMA_VERSION
        ),
        "scenario_generator_implementation_sha256": (
            integrated_synthetic_shock_generator_implementation_sha256()
        ),
        "scenario_selection_pending_input_sha256": (
            _load_exact_finite_scenario_selection_input_snapshot()[1]
        ),
    }


def _scenario_audited_design_mapping(
    *,
    canary_id: str,
    parameter_states: tuple[ExactFiniteParameterState, ...],
    scenario_replicate_count: int,
    step_count: int,
    candidate_allow_buy: bool,
    candidate_allow_sell: bool,
    candidate_is_noop: bool,
) -> dict[str, object]:
    config = integrated_synthetic_forward_market_config()
    output = ledger_output_spec(
        output_id="exact_finite_terminal_cash_level",
        horizon_id="terminal",
        scope=DifferenceScope.GROUP,
        group_id="all-latent-strategies",
        metric_id="ledger.group.cash_level",
        config=config,
        role=OutputRole.PRIMARY,
        benefit_direction=LedgerBenefitDirection.DESCRIPTIVE,
        undefined_policy=UndefinedMetricPolicy.FAIL_CLOSED,
    )
    return {
        "candidate_allow_buy": candidate_allow_buy,
        "candidate_allow_sell": candidate_allow_sell,
        "candidate_is_noop": candidate_is_noop,
        "candidate_scenario_event_keys": [
            {
                "event_keys": [
                    _policy_event_key(
                        canary_id=canary_id,
                        seed=candidate_seed,
                        scenario_index=index,
                    ).value
                    for index in range(scenario_replicate_count)
                ],
                "scenario_seed": candidate_seed,
            }
            for candidate_seed in (
                POLICY_FUNCTIONAL_EXACT_FINITE_SCENARIO_CANDIDATE_SEEDS
            )
        ],
        "canary_id": canary_id,
        "functional_definition": POLICY_FUNCTIONAL_EXACT_FINITE_FUNCTIONAL,
        "include_all_groups": False,
        "execution_lineage": _scenario_audit_execution_lineage_mapping(),
        "market_config_sha256": config.sha256,
        "output": output.to_mapping(),
        "parameter_states": [item.to_mapping() for item in parameter_states],
        "parallel_worlds": False,
        "scenario_replicate_count": scenario_replicate_count,
        "schema_version": "policytwin.exact-finite-scenario-audited-design.v1",
        "step_count": step_count,
    }


def exact_finite_scenario_audited_design_sha256() -> str:
    """Return the exact target design used by the one-time fixture audit."""
    return _digest(
        _scenario_audited_design_mapping(
            canary_id=_SCENARIO_AUDIT_CANARY_ID,
            parameter_states=_default_parameter_states(),
            scenario_replicate_count=_SCENARIO_AUDIT_REPLICATE_COUNT,
            step_count=_SCENARIO_AUDIT_STEP_COUNT,
            candidate_allow_buy=_SCENARIO_AUDIT_CANDIDATE_ALLOW_BUY,
            candidate_allow_sell=_SCENARIO_AUDIT_CANDIDATE_ALLOW_SELL,
            candidate_is_noop=_SCENARIO_AUDIT_CANDIDATE_IS_NOOP,
        )
    )


def _definition(identifier: str, *, digest: str | None = None) -> FrozenDefinition:
    return FrozenDefinition(
        definition_id=identifier,
        description=f"Frozen exact-finite synthetic definition for {identifier}.",
        definition_sha256=_digest(identifier) if digest is None else digest,
    )


def _build_exact_finite_protocol(
    *,
    canary_id: str,
    frozen_at: datetime,
    parameter_states: tuple[ExactFiniteParameterState, ...],
    scenario_event_keys: tuple[str, ...],
    candidate_allow_buy: bool,
    candidate_allow_sell: bool,
    candidate_is_noop: bool,
    step_count: int,
) -> tuple[StudyProtocol, LedgerPosteriorOutputSpec, str]:
    config = integrated_synthetic_forward_market_config()
    template_key = _policy_event_key(
        canary_id=canary_id,
        seed="protocol-template-not-random-evidence",
        scenario_index=0,
    )
    template = integrated_synthetic_pair(
        event_key=template_key,
        parameters=parameter_states[0].parameters,
        config=config,
        candidate_allow_buy=candidate_allow_buy,
        candidate_allow_sell=candidate_allow_sell,
        candidate_is_noop=candidate_is_noop,
        step_count=step_count,
        include_all_groups=False,
    )
    receipt = template.scenario_generation_receipt
    if receipt is None:
        _fail(
            "the live integrated forward generator did not issue a "
            "ScenarioGenerationReceipt"
        )
    receipt.validate_live()
    horizons = tuple(
        _definition(
            horizon.horizon_id,
            digest=_digest(horizon.to_mapping()),
        )
        for horizon in template.horizons
    )
    cohorts = tuple(
        FrozenCohort(
            cohort_id=group.group_id,
            description=(
                "Frozen synthetic latent-strategy cohort; not an observed "
                "investor identity."
            ),
            definition_sha256=group.cohort_sha256,
            basis=CohortBasis.LATENT_STRATEGY,
        )
        for group in template.groups
    )
    output = ledger_output_spec(
        output_id="exact_finite_terminal_cash_level",
        horizon_id="terminal",
        scope=DifferenceScope.GROUP,
        group_id="all-latent-strategies",
        metric_id="ledger.group.cash_level",
        config=config,
        role=OutputRole.PRIMARY,
        benefit_direction=LedgerBenefitDirection.DESCRIPTIVE,
        undefined_policy=UndefinedMetricPolicy.FAIL_CLOSED,
    )
    ledger_evidence_plan = _definition(
        "exact-finite-ledger-policy-functional",
        digest=_digest(
            {
                "aggregation": "exact_arithmetic_mean_over_frozen_scenarios",
                "direct_registered_source": {
                    "cohort_id": "all-latent-strategies",
                    "horizon_id": "terminal",
                    "metric_id": "cash",
                    "role": OutputRole.PRIMARY.value,
                },
                "functional_definition": (POLICY_FUNCTIONAL_EXACT_FINITE_FUNCTIONAL),
                "ledger_output": output.to_mapping(),
                "ledger_verifier_implementation_sha256": (
                    ledger_evidence_verifier_implementation_sha256()
                ),
                "scenario_replicate_count": len(scenario_event_keys),
            }
        ),
    )
    outputs: list[RegisteredOutput] = []
    for horizon_index, horizon in enumerate(horizons):
        for cohort_index, cohort in enumerate(cohorts):
            for metric_index, metric_id in enumerate(INTEGRATED_METRIC_IDS):
                outputs.append(
                    RegisteredOutput(
                        output_id=(
                            f"out-{horizon_index}-{cohort_index}-{metric_index}"
                        ),
                        metric_id=metric_id,
                        cohort_id=cohort.cohort_id,
                        horizon_id=horizon.definition_id,
                        role=(
                            OutputRole.PRIMARY
                            if horizon.definition_id == "terminal"
                            and cohort.cohort_id == "all-latent-strategies"
                            and metric_id == "cash"
                            else OutputRole.SECONDARY
                        ),
                        interval_method="none",
                        interval_levels=(),
                    )
                )
    protocol = StudyProtocol(
        schema_version=2,
        study_id=canary_id,
        mode=StudyMode.SYNTHETIC,
        claim_scope=ClaimScope.SYNTHETIC_ONLY,
        registered_at=frozen_at,
        knowledge_cutoff=config.cutoff,
        model_frozen_at=frozen_at,
        policy_start=frozen_at + timedelta(minutes=30),
        policy_end=frozen_at + timedelta(days=1),
        evaluation_reveal_at=frozen_at + timedelta(days=2),
        estimand=_definition(
            INTEGRATED_ESTIMAND_ID,
            digest=integrated_paired_estimand_definition_sha256(),
        ),
        baseline_policy=_definition(
            template.baseline_policy.policy_id,
            digest=template.baseline_policy.sha256,
        ),
        candidate_policy=_definition(
            template.candidate_policy.policy_id,
            digest=template.candidate_policy.sha256,
        ),
        cohorts=cohorts,
        horizons=horizons,
        metrics=tuple(
            _definition(
                metric_id, digest=integrated_metric_definition_sha256(metric_id)
            )
            for metric_id in INTEGRATED_METRIC_IDS
        ),
        baselines=(_definition("zero_effect"),),
        outputs=tuple(outputs),
        calibration_statistics=("exact_finite_macro_net_worth",),
        heldout_validation_statistics=(
            "exact_finite_macro_quoted_spread_not_used_for_posterior",
        ),
        policy_evaluation_outcomes=(f"ledger-primary-{_digest(output.to_mapping())}",),
        failure_thresholds=(
            FailureThreshold(
                check_id="temporal_leakage",
                comparator="higher_is_worse",
                yellow_value=0.0,
                red_value=0.0,
                abstain_at_red=True,
            ),
        ),
        abstention_rule=_definition("exact_finite_scientific_abstention"),
        randomness_protocol=RandomnessProtocol.EVENT_KEYED_PAIRED,
        evidence_plans=(ledger_evidence_plan,),
    )
    generator_commitment = _digest(
        {
            "receipt_schema_version": receipt.schema_version,
            "scenario_event_keys": list(scenario_event_keys),
            "scenario_generator_spec": receipt.generator_spec.to_mapping(),
        }
    )
    return protocol, output, generator_commitment


@dataclass(frozen=True, slots=True, init=False, weakref_slot=True)
class ExactFiniteCanaryPlan:
    """Factory-produced immutable commitment to the complete finite design."""

    canary_id: str
    seed: str
    scenario_seed: str
    frozen_at: datetime
    parameter_states: tuple[ExactFiniteParameterState, ...]
    observation_support_size: int
    posterior_draw_count: int
    scenario_replicate_count: int
    step_count: int
    candidate_allow_buy: bool
    candidate_allow_sell: bool
    candidate_is_noop: bool
    observation_event_keys: tuple[str, ...]
    scenario_event_keys: tuple[str, ...]
    observation_simulator_sha256: str
    market_config_sha256: str
    scenario_generator_commitment_sha256: str
    scenario_design_manifest_sha256: str
    protocol: StudyProtocol
    output: LedgerPosteriorOutputSpec
    functional_definition: str
    claim_scope: str
    schema_version: str

    def __init__(self) -> None:
        raise TypeError(
            "ExactFiniteCanaryPlan is factory-produced; use "
            "build_exact_finite_policy_functional_plan"
        )

    def __copy__(self) -> NoReturn:
        raise TypeError("exact finite canary plans cannot be copied")

    def __deepcopy__(self, memo: object) -> NoReturn:
        del memo
        raise TypeError("exact finite canary plans cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("exact finite canary plans cannot be serialized")

    def _fresh_validate(self) -> None:
        _identifier(self.canary_id, "canary_id")
        if not isinstance(self.seed, str) or not self.seed or len(self.seed) > 512:
            _fail("seed must contain between 1 and 512 characters")
        if self.scenario_seed != POLICY_FUNCTIONAL_EXACT_FINITE_SELECTED_SCENARIO_SEED:
            _fail("scenario seed differs from the frozen design-time selection")
        frozen_at = _aware(self.frozen_at, "frozen_at")
        if frozen_at != self.frozen_at:
            _fail("frozen_at must be canonical UTC")
        if (
            not isinstance(self.parameter_states, tuple)
            or len(self.parameter_states) < 2
            or not all(
                type(item) is ExactFiniteParameterState
                for item in self.parameter_states
            )
        ):
            _fail("plan requires at least two exact finite parameter states")
        state_ids = tuple(item.state_id for item in self.parameter_states)
        if len(set(state_ids)) != len(state_ids):
            _fail("parameter state IDs must be unique")
        _validate_exact_design_dimensions(
            parameter_state_count=len(self.parameter_states),
            observation_support_size=self.observation_support_size,
            posterior_draw_count=self.posterior_draw_count,
            scenario_replicate_count=self.scenario_replicate_count,
        )
        _validate_integrated_execution_budget(
            parameter_state_count=len(self.parameter_states),
            observation_support_size=self.observation_support_size,
            scenario_replicate_count=self.scenario_replicate_count,
            step_count=self.step_count,
        )
        if not all(
            isinstance(value, bool)
            for value in (
                self.candidate_allow_buy,
                self.candidate_allow_sell,
                self.candidate_is_noop,
            )
        ):
            _fail("candidate policy switches must be boolean")
        if self.candidate_is_noop and not (
            self.candidate_allow_buy and self.candidate_allow_sell
        ):
            _fail("no-op policy must equal the baseline policy switches")
        expected_observation_keys = tuple(
            _observation_event_key(
                canary_id=self.canary_id,
                seed=self.seed,
                support_index=index,
            ).value
            for index in range(self.observation_support_size)
        )
        expected_scenario_keys = tuple(
            _policy_event_key(
                canary_id=self.canary_id,
                seed=self.scenario_seed,
                scenario_index=index,
            ).value
            for index in range(self.scenario_replicate_count)
        )
        if self.observation_event_keys != expected_observation_keys:
            _fail("observation EventKey grid does not freshly rederive")
        if self.scenario_event_keys != expected_scenario_keys:
            _fail("policy scenario EventKey grid does not freshly rederive")
        adapter = IntegratedSyntheticCalibrationAdapter(step_count=self.step_count)
        if self.observation_simulator_sha256 != _digest(adapter.to_mapping()):
            _fail("pre-policy observation simulator commitment is stale")
        config = integrated_synthetic_forward_market_config()
        if self.market_config_sha256 != config.sha256:
            _fail("integrated forward market configuration commitment is stale")
        protocol, output, generator_commitment = _build_exact_finite_protocol(
            canary_id=self.canary_id,
            frozen_at=self.frozen_at,
            parameter_states=self.parameter_states,
            scenario_event_keys=self.scenario_event_keys,
            candidate_allow_buy=self.candidate_allow_buy,
            candidate_allow_sell=self.candidate_allow_sell,
            candidate_is_noop=self.candidate_is_noop,
            step_count=self.step_count,
        )
        if self.protocol != protocol or self.output != output:
            _fail("protocol or ledger output commitment does not freshly rederive")
        if self.scenario_generator_commitment_sha256 != generator_commitment:
            _fail("scenario generator commitment does not freshly rederive")
        if self.scenario_design_manifest_sha256 != _digest(
            exact_finite_scenario_design_manifest()
        ):
            _fail("scenario design manifest commitment does not freshly rederive")
        validate_exact_finite_scenario_design_manifest(
            exact_finite_scenario_design_manifest(),
            require_finalized=False,
        )
        if self.functional_definition != POLICY_FUNCTIONAL_EXACT_FINITE_FUNCTIONAL:
            _fail("exact finite policy functional definition changed")
        if self.claim_scope != POLICY_FUNCTIONAL_EXACT_FINITE_CLAIM_SCOPE:
            _fail("exact finite canary claim scope changed")
        if self.schema_version != POLICY_FUNCTIONAL_EXACT_FINITE_SCHEMA_VERSION:
            _fail("unsupported exact finite canary schema version")

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_mapping())

    @property
    def sha256(self) -> str:
        return sha256_bytes(self.canonical_bytes)

    def to_mapping(self) -> dict[str, object]:
        self._fresh_validate()
        return {
            "candidate_allow_buy": self.candidate_allow_buy,
            "candidate_allow_sell": self.candidate_allow_sell,
            "candidate_is_noop": self.candidate_is_noop,
            "canary_id": self.canary_id,
            "claim_scope": self.claim_scope,
            "frozen_at": self.frozen_at.isoformat(),
            "functional_definition": self.functional_definition,
            "market_config_sha256": self.market_config_sha256,
            "observation_event_keys": list(self.observation_event_keys),
            "observation_simulator_sha256": self.observation_simulator_sha256,
            "observation_support_size": self.observation_support_size,
            "output": self.output.to_mapping(),
            "parameter_states": [item.to_mapping() for item in self.parameter_states],
            "posterior_draw_count": self.posterior_draw_count,
            "protocol": self.protocol.to_mapping(),
            "scenario_event_keys": list(self.scenario_event_keys),
            "scenario_design_manifest_sha256": self.scenario_design_manifest_sha256,
            "scenario_generator_commitment_sha256": (
                self.scenario_generator_commitment_sha256
            ),
            "scenario_replicate_count": self.scenario_replicate_count,
            "schema_version": self.schema_version,
            "scenario_seed": self.scenario_seed,
            "seed": self.seed,
            "step_count": self.step_count,
        }


def build_exact_finite_policy_functional_plan(
    *,
    canary_id: str = "exact-finite-policy-functional-ci",
    seed: str = "exact-finite-policy-functional-grid-v1",
    frozen_at: datetime | None = None,
    parameter_states: tuple[ExactFiniteParameterState, ...] | None = None,
    observation_support_size: int = 8,
    posterior_draw_count: int = 3,
    scenario_replicate_count: int = 4,
    step_count: int = 3,
    candidate_allow_buy: bool = False,
    candidate_allow_sell: bool = True,
    candidate_is_noop: bool = False,
) -> ExactFiniteCanaryPlan:
    """Freeze the default ``K=2,L=8,R=4,M=3`` exact finite canary."""
    identifier = _identifier(canary_id, "canary_id")
    if not isinstance(seed, str) or not seed or len(seed) > 512:
        _fail("seed must contain between 1 and 512 characters")
    states = (
        _default_parameter_states() if parameter_states is None else parameter_states
    )
    if not isinstance(states, tuple):
        _fail("parameter_states must be an immutable tuple")
    _validate_exact_design_dimensions(
        parameter_state_count=len(states),
        observation_support_size=observation_support_size,
        posterior_draw_count=posterior_draw_count,
        scenario_replicate_count=scenario_replicate_count,
    )
    _validate_integrated_execution_budget(
        parameter_state_count=len(states),
        observation_support_size=observation_support_size,
        scenario_replicate_count=scenario_replicate_count,
        step_count=step_count,
    )
    if not all(
        isinstance(value, bool)
        for value in (candidate_allow_buy, candidate_allow_sell, candidate_is_noop)
    ):
        _fail("candidate policy switches must be boolean")
    if candidate_is_noop and not (candidate_allow_buy and candidate_allow_sell):
        _fail("no-op policy must enable the baseline buy and sell permissions")
    config = integrated_synthetic_forward_market_config()
    freeze = config.cutoff if frozen_at is None else _aware(frozen_at, "frozen_at")
    if freeze != config.cutoff:
        _fail("exact finite protocol freeze must equal the trusted synthetic cutoff")
    observation_keys = tuple(
        _observation_event_key(
            canary_id=identifier,
            seed=seed,
            support_index=index,
        ).value
        for index in range(observation_support_size)
    )
    scenario_keys = tuple(
        _policy_event_key(
            canary_id=identifier,
            seed=POLICY_FUNCTIONAL_EXACT_FINITE_SELECTED_SCENARIO_SEED,
            scenario_index=index,
        ).value
        for index in range(scenario_replicate_count)
    )
    protocol, output, generator_commitment = _build_exact_finite_protocol(
        canary_id=identifier,
        frozen_at=freeze,
        parameter_states=states,
        scenario_event_keys=scenario_keys,
        candidate_allow_buy=candidate_allow_buy,
        candidate_allow_sell=candidate_allow_sell,
        candidate_is_noop=candidate_is_noop,
        step_count=step_count,
    )
    adapter = IntegratedSyntheticCalibrationAdapter(step_count=step_count)
    plan = object.__new__(ExactFiniteCanaryPlan)
    values: tuple[tuple[str, object], ...] = (
        ("canary_id", identifier),
        ("seed", seed),
        ("scenario_seed", POLICY_FUNCTIONAL_EXACT_FINITE_SELECTED_SCENARIO_SEED),
        ("frozen_at", freeze),
        ("parameter_states", states),
        ("observation_support_size", observation_support_size),
        ("posterior_draw_count", posterior_draw_count),
        ("scenario_replicate_count", scenario_replicate_count),
        ("step_count", step_count),
        ("candidate_allow_buy", candidate_allow_buy),
        ("candidate_allow_sell", candidate_allow_sell),
        ("candidate_is_noop", candidate_is_noop),
        ("observation_event_keys", observation_keys),
        ("scenario_event_keys", scenario_keys),
        ("observation_simulator_sha256", _digest(adapter.to_mapping())),
        ("market_config_sha256", config.sha256),
        ("scenario_generator_commitment_sha256", generator_commitment),
        (
            "scenario_design_manifest_sha256",
            _digest(exact_finite_scenario_design_manifest()),
        ),
        ("protocol", protocol),
        ("output", output),
        ("functional_definition", POLICY_FUNCTIONAL_EXACT_FINITE_FUNCTIONAL),
        ("claim_scope", POLICY_FUNCTIONAL_EXACT_FINITE_CLAIM_SCOPE),
        ("schema_version", POLICY_FUNCTIONAL_EXACT_FINITE_SCHEMA_VERSION),
    )
    for field, value in values:
        object.__setattr__(plan, field, value)
    plan._fresh_validate()
    _register_exact_finite_plan(plan)
    return plan


_EXACT_FINITE_PLANS: WeakValueDictionary[int, ExactFiniteCanaryPlan] = (
    WeakValueDictionary()
)
_EXACT_FINITE_PLAN_SEALS: dict[int, str] = {}
_EXACT_FINITE_PLAN_LOCK = RLock()


def _forget_exact_finite_plan(identity: int) -> None:
    with _EXACT_FINITE_PLAN_LOCK:
        if _EXACT_FINITE_PLANS.get(identity) is None:
            _EXACT_FINITE_PLAN_SEALS.pop(identity, None)


def _register_exact_finite_plan(plan: ExactFiniteCanaryPlan) -> None:
    seal = plan.sha256
    identity = id(plan)
    with _EXACT_FINITE_PLAN_LOCK:
        _EXACT_FINITE_PLANS[identity] = plan
        _EXACT_FINITE_PLAN_SEALS[identity] = seal
    finalize(plan, _forget_exact_finite_plan, identity)


def require_exact_finite_canary_plan(
    plan: ExactFiniteCanaryPlan,
) -> ExactFiniteCanaryPlan:
    """Require the same-process factory identity and its unchanged mint seal."""
    if type(plan) is not ExactFiniteCanaryPlan:
        raise TypeError("plan must be an exact ExactFiniteCanaryPlan")
    try:
        before = plan.sha256
    except Exception as error:
        raise ExactFinitePolicyFunctionalError(
            "exact finite plan cannot reproduce its mint seal"
        ) from error
    with _EXACT_FINITE_PLAN_LOCK:
        registered = _EXACT_FINITE_PLANS.get(id(plan))
        seal = _EXACT_FINITE_PLAN_SEALS.get(id(plan))
    if registered is not plan or seal != before:
        _fail("exact finite plan is unissued or differs from its factory mint seal")
    plan._fresh_validate()
    after = plan.sha256
    with _EXACT_FINITE_PLAN_LOCK:
        still_registered = _EXACT_FINITE_PLANS.get(id(plan))
        still_sealed = _EXACT_FINITE_PLAN_SEALS.get(id(plan))
    if still_registered is not plan or still_sealed != after or before != after:
        _fail("exact finite plan changed during fresh mint-seal validation")
    return plan


def exact_finite_plan_matches_scenario_audited_design(
    plan: ExactFiniteCanaryPlan,
) -> bool:
    """Return whether a minted plan exactly matches the audited fixture target."""
    plan = require_exact_finite_canary_plan(plan)
    actual = _digest(
        _scenario_audited_design_mapping(
            canary_id=plan.canary_id,
            parameter_states=plan.parameter_states,
            scenario_replicate_count=plan.scenario_replicate_count,
            step_count=plan.step_count,
            candidate_allow_buy=plan.candidate_allow_buy,
            candidate_allow_sell=plan.candidate_allow_sell,
            candidate_is_noop=plan.candidate_is_noop,
        )
    )
    return actual == exact_finite_scenario_audited_design_sha256()


def _observation_cell(
    plan: ExactFiniteCanaryPlan,
    state: ExactFiniteParameterState,
    support_index: int,
) -> ExactFiniteObservationCell:
    adapter = IntegratedSyntheticCalibrationAdapter(step_count=plan.step_count)
    key = _observation_event_key(
        canary_id=plan.canary_id,
        seed=plan.seed,
        support_index=support_index,
    )
    summary = adapter.simulate(state.parameters, event_key=key)
    values = summary.as_dict()
    if set(values) != {
        "macro_quoted_spread",
        "macro_net_worth",
        "meso_mean_buy_probability",
        "meso_memory_load",
        "micro_submission_rate",
        "micro_turnover_probability",
    }:
        _fail("integrated calibration adapter summary registry changed")
    raw_observation = values["macro_net_worth"]
    if (
        not isinstance(raw_observation, float)
        or not math.isfinite(raw_observation)
        or not raw_observation.is_integer()
        or abs(raw_observation) > _MAX_SAFE_INTEGER
    ):
        _fail("macro_net_worth is not an exact binary64-safe integer category")
    heldout = values["macro_quoted_spread"]
    if not isinstance(heldout, float):
        _fail("held-out quoted spread is not a binary64 value")
    return ExactFiniteObservationCell(
        state_id=state.state_id,
        support_index=support_index,
        event_key=key.value,
        summary_sha256=summary.sha256,
        observation_value=int(raw_observation),
        heldout_quoted_spread=_exact_from_binary64(heldout, "held-out quoted spread"),
    )


def _consumption_manifest(ledger: object, field: str) -> tuple[str, str]:
    if ledger is None:
        _fail(f"{field} shock-consumption ledger is missing")
    declared = getattr(ledger, "declared_event_keys", None)
    consumptions = getattr(ledger, "consumptions", None)
    violations = getattr(ledger, "violations", None)
    if (
        not isinstance(declared, tuple)
        or not isinstance(consumptions, tuple)
        or not isinstance(violations, tuple)
        or violations
    ):
        _fail(f"{field} shock-consumption ledger is incomplete or violated")
    declared_digest = _digest(list(declared))
    consumption_digest = _digest(
        [
            {
                "event_key": item.event_key.value,
                "realization_sha256": item.realization_sha256,
            }
            for item in consumptions
        ]
    )
    return declared_digest, consumption_digest


def _execute_protocol_bound_policy_cell(
    *,
    canary_id: str,
    frozen_at: datetime,
    state: ExactFiniteParameterState,
    scenario_seed: str,
    scenario_index: int,
    candidate_allow_buy: bool,
    candidate_allow_sell: bool,
    candidate_is_noop: bool,
    step_count: int,
    protocol: StudyProtocol,
    output: LedgerPosteriorOutputSpec,
    parallel_worlds: bool,
) -> ExactFinitePolicyFunctionalCell:
    """Execute the shared public protocol-bound integrated cell path."""
    config = integrated_synthetic_forward_market_config()
    key = _policy_event_key(
        canary_id=canary_id,
        seed=scenario_seed,
        scenario_index=scenario_index,
    )
    pair = integrated_synthetic_pair(
        event_key=key,
        parameters=state.parameters,
        config=config,
        candidate_allow_buy=candidate_allow_buy,
        candidate_allow_sell=candidate_allow_sell,
        candidate_is_noop=candidate_is_noop,
        step_count=step_count,
        include_all_groups=False,
    )
    generation_receipt = pair.scenario_generation_receipt
    if generation_receipt is None:
        _fail("formal integrated policy cell lacks a live ScenarioGenerationReceipt")
    generation_receipt.validate_live()
    issuer = SyntheticProtocolExecutionIssuer(protocol)
    scenario_namespace = _digest(scenario_seed)[:12]
    execution_id = f"ef-{scenario_namespace}-{state.state_id}-{scenario_index}"
    authorization = issuer.issue(
        execution_id=execution_id,
        consumer=ProtocolExecutionConsumer.INTEGRATED_SYNTHETIC_SIMULATION,
        purpose=ProtocolExecutionPurpose.PAIRED_SYNTHETIC_POLICY_SIMULATION,
        config_sha256=config.sha256,
        input_sha256=integrated_synthetic_pair_input_sha256(pair),
        execution_started_at=frozen_at,
    )
    artifact = run_protocol_bound_integrated_synthetic_pair(
        pair=pair,
        config=config,
        protocol=protocol,
        authorization=authorization,
        execution_started_at=frozen_at,
        parallel=parallel_worlds,
        invocation_order=(WorldRole.BASELINE, WorldRole.CANDIDATE),
    )
    artifact.paired_outcome.integrity_audit.require_passed()
    raw_internal_random_audit = artifact.internal_random_coupling_audit
    if type(raw_internal_random_audit) is not InternalRandomCouplingAudit:
        _fail("protocol-bound artifact lacks an exact internal random audit")
    internal_random_audit = require_verifier_owned_internal_random_coupling_audit(
        raw_internal_random_audit,
        baseline_run=artifact.baseline_run,
        candidate_run=artifact.candidate_run,
        protocol=artifact.protocol,
        config=config,
        pair_input=pair,
        protocol_execution_receipt=artifact.protocol_execution_receipt,
    )
    if (
        internal_random_audit is not raw_internal_random_audit
        or artifact.cross_world_internal_random_coupling_verified is not True
        or internal_random_audit.cross_world_internal_random_coupling_verified
        is not True
        or internal_random_audit.external_validity_evidence is not False
    ):
        _fail("protocol-bound internal random coupling audit does not rederive")
    evidence = derive_protocol_bound_ledger_evidence(artifact)
    matching = tuple(
        item
        for item in evidence.differences
        if item.horizon_id == output.horizon_id
        and item.scope is output.scope
        and item.group_id == output.group_id
        and item.metric_id == output.metric_id
    )
    if len(matching) != 1:
        _fail("ledger derivation did not produce exactly one frozen functional cell")
    metric = matching[0].difference
    if metric.state is not ExactMetricState.DEFINED or metric.value is None:
        _fail("the frozen exact policy functional is undefined")
    baseline_declared, baseline_consumed = _consumption_manifest(
        artifact.paired_outcome.baseline.shock_consumption_ledger,
        "baseline",
    )
    candidate_declared, candidate_consumed = _consumption_manifest(
        artifact.paired_outcome.candidate.shock_consumption_ledger,
        "candidate",
    )
    if baseline_declared != candidate_declared:
        _fail("paired worlds declared different exogenous event identities")
    if baseline_consumed != candidate_consumed:
        _fail("paired worlds consumed different exogenous shock realizations")
    if pair.shock_tape.sha256 != generation_receipt.shock_tape_sha256:
        _fail("scenario generation receipt does not bind the executed shock tape")
    return ExactFinitePolicyFunctionalCell(
        state_id=state.state_id,
        scenario_index=scenario_index,
        event_key=key.value,
        scenario_generation_receipt_sha256=generation_receipt.sha256,
        shock_tape_sha256=pair.shock_tape.sha256,
        declared_event_keys_sha256=baseline_declared,
        baseline_consumption_manifest_sha256=baseline_consumed,
        candidate_consumption_manifest_sha256=candidate_consumed,
        pair_integrity_audit_sha256=artifact.paired_outcome.integrity_audit.sha256,
        protocol_execution_receipt_sha256=(artifact.protocol_execution_receipt.sha256),
        protocol_bound_artifact_sha256=artifact.sha256,
        ledger_evidence_sha256=evidence.sha256,
        internal_random_coupling_audit_sha256=internal_random_audit.sha256,
        internal_random_role_neutral_ledger_sha256=(
            internal_random_audit.role_neutral_ledger_sha256
        ),
        internal_random_expected_entry_count=(
            internal_random_audit.expected_entry_count
        ),
        functional_value=metric.value,
    )


def _policy_cell(
    plan: ExactFiniteCanaryPlan,
    state: ExactFiniteParameterState,
    scenario_index: int,
    *,
    parallel_worlds: bool,
) -> ExactFinitePolicyFunctionalCell:
    return _execute_protocol_bound_policy_cell(
        canary_id=plan.canary_id,
        frozen_at=plan.frozen_at,
        state=state,
        scenario_seed=plan.scenario_seed,
        scenario_index=scenario_index,
        candidate_allow_buy=plan.candidate_allow_buy,
        candidate_allow_sell=plan.candidate_allow_sell,
        candidate_is_noop=plan.candidate_is_noop,
        step_count=plan.step_count,
        protocol=plan.protocol,
        output=plan.output,
        parallel_worlds=parallel_worlds,
    )


def _audit_scenario_seed_functional_nondegeneracy(
    scenario_seed: str,
    *,
    evaluation_capability: object,
) -> bool:
    """Evaluate one frozen fixture via the production cell implementation.

    This design-time helper intentionally returns only a Boolean.  It uses the
    same public protocol-bound integrated runner, verifier-derived ledger
    evidence, scenario receipt checks, exogenous-tape checks, and internal RNG
    coupling audit as a production policy-functional cell.
    """
    _require_active_blind_audit_evaluation(evaluation_capability)
    if scenario_seed not in POLICY_FUNCTIONAL_EXACT_FINITE_SCENARIO_CANDIDATE_SEEDS:
        _fail("blind audit scenario seed is outside the frozen candidate grid")
    canary_id = _SCENARIO_AUDIT_CANARY_ID
    states = _default_parameter_states()
    scenario_replicate_count = _SCENARIO_AUDIT_REPLICATE_COUNT
    step_count = _SCENARIO_AUDIT_STEP_COUNT
    config = integrated_synthetic_forward_market_config()
    scenario_keys = tuple(
        _policy_event_key(
            canary_id=canary_id,
            seed=scenario_seed,
            scenario_index=index,
        ).value
        for index in range(scenario_replicate_count)
    )
    protocol, output, _ = _build_exact_finite_protocol(
        canary_id=canary_id,
        frozen_at=config.cutoff,
        parameter_states=states,
        scenario_event_keys=scenario_keys,
        candidate_allow_buy=_SCENARIO_AUDIT_CANDIDATE_ALLOW_BUY,
        candidate_allow_sell=_SCENARIO_AUDIT_CANDIDATE_ALLOW_SELL,
        candidate_is_noop=_SCENARIO_AUDIT_CANDIDATE_IS_NOOP,
        step_count=step_count,
    )
    cells = tuple(
        _execute_protocol_bound_policy_cell(
            canary_id=canary_id,
            frozen_at=config.cutoff,
            state=state,
            scenario_seed=scenario_seed,
            scenario_index=scenario_index,
            candidate_allow_buy=_SCENARIO_AUDIT_CANDIDATE_ALLOW_BUY,
            candidate_allow_sell=_SCENARIO_AUDIT_CANDIDATE_ALLOW_SELL,
            candidate_is_noop=_SCENARIO_AUDIT_CANDIDATE_IS_NOOP,
            step_count=step_count,
            protocol=protocol,
            output=output,
            parallel_worlds=False,
        )
        for state in states
        for scenario_index in range(scenario_replicate_count)
    )
    for scenario_index in range(scenario_replicate_count):
        scenario_cells = tuple(
            item for item in cells if item.scenario_index == scenario_index
        )
        if len(scenario_cells) != len(states):
            _fail("blind audit policy lookup grid is incomplete")
        for field in (
            "event_key",
            "scenario_generation_receipt_sha256",
            "shock_tape_sha256",
            "declared_event_keys_sha256",
            "baseline_consumption_manifest_sha256",
            "candidate_consumption_manifest_sha256",
            "internal_random_role_neutral_ledger_sha256",
            "internal_random_expected_entry_count",
        ):
            if len({getattr(item, field) for item in scenario_cells}) != 1:
                _fail("blind audit scenario coupling does not verify")
    state_functionals = tuple(
        sum(
            (
                _fraction(cell.functional_value)
                for cell in cells
                if cell.state_id == state.state_id
            ),
            Fraction(),
        )
        / scenario_replicate_count
        for state in states
    )
    return len(set(state_functionals)) > 1


def run_exact_finite_scenario_blind_audit(
    *,
    attempt_capability: object,
    stdout: TextIO | None = None,
) -> tuple[str, object]:
    """Run the bounded synthetic-only s--x fixture evaluator.

    The only successful stdout payload is the first eligible frozen seed or
    ``no-match``, followed by one newline.  This API returns no candidate
    effect value, sign, direction, or rank diagnostic and writes none itself.
    Its receipt explicitly does not verify the absence of other file logging,
    delayed output, or external persistence.  Production canary execution
    never calls this design-time function.
    """
    _require_isolated_blind_audit_process()
    expected_claim_sha256, activation_ticket = _consume_blind_audit_attempt_capability(
        attempt_capability
    )
    activation_ticket_pending = True
    try:
        claim, persisted_claim_sha256 = _read_bounded_blind_audit_attempt_claim()
        if persisted_claim_sha256 != expected_claim_sha256:
            _fail("blind audit evaluator attempt claim differs from the durable claim")
        _, selection_input_sha256 = (
            _load_exact_finite_scenario_selection_input_snapshot()
        )
        _validate_blind_audit_attempt_claim(
            claim,
            claim_sha256=persisted_claim_sha256,
            selection_input_sha256=selection_input_sha256,
        )
        target = sys.stdout if stdout is None else stdout
        if not callable(getattr(target, "write", None)) or not callable(
            getattr(target, "flush", None)
        ):
            raise TypeError("stdout must be a writable text stream")
        activation_ticket_pending = False
        evaluation_capability = _activate_blind_audit_evaluation(activation_ticket)
    finally:
        if activation_ticket_pending:
            _discard_blind_audit_activation_ticket(activation_ticket)
    captured_stdout = StringIO()
    captured_stderr = StringIO()
    selected = "no-match"
    observations: list[tuple[str, bool]] = []
    evaluation_failed = False
    captured_process_bytes = b""
    evaluation_completed = False
    try:
        saved_stdout_fd = os.dup(1)
        try:
            saved_stderr_fd = os.dup(2)
        except Exception:
            os.close(saved_stdout_fd)
            raise
        try:
            with TemporaryFile(mode="w+b") as process_capture:
                try:
                    sys.stdout.flush()
                    sys.stderr.flush()
                    os.dup2(process_capture.fileno(), 1)
                    os.dup2(process_capture.fileno(), 2)
                    with (
                        redirect_stdout(captured_stdout),
                        redirect_stderr(captured_stderr),
                    ):
                        for (
                            candidate
                        ) in POLICY_FUNCTIONAL_EXACT_FINITE_SCENARIO_CANDIDATE_SEEDS:
                            eligible = _audit_scenario_seed_functional_nondegeneracy(
                                candidate,
                                evaluation_capability=evaluation_capability,
                            )
                            if type(eligible) is not bool:
                                _fail(
                                    "blind audit evaluator must return an exact Boolean"
                                )
                            observations.append((candidate, eligible))
                            if eligible:
                                selected = candidate
                                break
                except Exception:
                    evaluation_failed = True
                finally:
                    try:
                        sys.stdout.flush()
                        sys.stderr.flush()
                    except Exception:
                        evaluation_failed = True
                    os.dup2(saved_stdout_fd, 1)
                    os.dup2(saved_stderr_fd, 2)
                process_capture.seek(0)
                captured_process_bytes = process_capture.read()
        finally:
            try:
                os.close(saved_stdout_fd)
            finally:
                os.close(saved_stderr_fd)
        if evaluation_failed:
            raise ExactFinitePolicyFunctionalError(
                "blind audit candidate evaluation failed closed"
            ) from None
        if (
            captured_stdout.getvalue()
            or captured_stderr.getvalue()
            or captured_process_bytes
        ):
            _fail("blind audit evaluator emitted forbidden process output")
        _require_isolated_blind_audit_process()
        target_payload = f"{selected}\n"
        target_written = target.write(target_payload)
        if type(target_written) is not int or target_written != len(target_payload):
            _fail("blind audit target did not accept the complete seed-only payload")
        target.flush()
        completion = _complete_blind_audit_evaluation(
            evaluation_capability,
            blind_audit_output=selected,
            observations=tuple(observations),
        )
        evaluation_completed = True
        return selected, completion
    finally:
        if not evaluation_completed:
            _abort_blind_audit_evaluation(evaluation_capability)


def execute_exact_finite_scenario_blind_audit_and_finalize() -> str:
    """Run once in an isolated process and atomically finalize selection state.

    The immutable pending-input snapshot must byte-match the live pending state
    before any candidate executes.  A durable O_EXCL/O_NOFOLLOW attempt claim
    consumes the complete s--x epoch before the evaluator starts.  The selected
    seed is written and flushed before an execution-specific receipt is issued
    and the mutable state is replaced.  While the local durable claim remains
    under custody, any failure after its creation forbids retrying this epoch.
    This is not protection against a malicious workspace owner deleting state.
    """
    global POLICY_FUNCTIONAL_EXACT_FINITE_SELECTED_SCENARIO_SEED

    _require_isolated_blind_audit_process()
    pending_state, pending_sha256 = _load_exact_finite_scenario_selection_state()
    snapshot_state, snapshot_sha256 = (
        _load_exact_finite_scenario_selection_input_snapshot()
    )
    if pending_sha256 != snapshot_sha256 or pending_state != snapshot_state:
        _fail("blind audit live input is not the frozen pending snapshot")
    if validate_exact_finite_scenario_design_manifest(
        exact_finite_scenario_design_manifest(),
        require_finalized=False,
    ):
        _fail("blind audit selection state is already finalized")

    (
        attempt_claim,
        attempt_claim_sha256,
        attempt_capability,
    ) = _claim_blind_audit_attempt(selection_input_sha256=snapshot_sha256)
    internal_stdout = StringIO()
    selected, completion_capability = run_exact_finite_scenario_blind_audit(
        attempt_capability=attempt_capability,
        stdout=internal_stdout,
    )
    completion_handoff_finished = False
    try:
        expected_stdout_bytes = f"{selected}\n".encode()
        if internal_stdout.getvalue().encode() != expected_stdout_bytes:
            _fail("blind audit evaluator did not produce the exact seed-only payload")
        finalization_capability = _write_blind_audit_stdout_bytes(
            expected_stdout_bytes,
            completion_capability=completion_capability,
        )
        completion_handoff_finished = True
    finally:
        if not completion_handoff_finished:
            _discard_blind_audit_completion_capability_if_present(completion_capability)
    receipt = _issue_blind_audit_execution_receipt(
        selected,
        attempt_claim=attempt_claim,
        attempt_claim_sha256=attempt_claim_sha256,
        finalization_capability=finalization_capability,
    )
    selected_seed = None if selected == "no-match" else selected
    plan_seed = (
        POLICY_FUNCTIONAL_EXACT_FINITE_SCENARIO_CANDIDATE_SEEDS[0]
        if selected_seed is None
        else selected_seed
    )
    final_state: dict[str, object] = {
        "audit_epoch": _SCENARIO_AUDIT_EPOCH,
        "audited_design_sha256": exact_finite_scenario_audited_design_sha256(),
        "blind_audit_attempt_claim_sha256": attempt_claim_sha256,
        "blind_audit_output": selected,
        "blind_audit_output_receipt": receipt,
        "blind_audit_selection_input_sha256": snapshot_sha256,
        "candidate_statuses": list(
            derive_exact_finite_scenario_audit_statuses(selected)
        ),
        "manifest_selected_seed": selected_seed,
        "plan_scenario_seed": plan_seed,
        "production_authorized": selected_seed is not None,
        "schema_version": _SELECTION_STATE_SCHEMA_VERSION,
    }
    write_json_file(_selection_state_path(), final_state)
    persisted, _ = _load_exact_finite_scenario_selection_state()
    if persisted != final_state:
        _fail("blind audit finalized selection state did not round-trip")
    # The module necessarily imported the pending placeholder before the
    # one-time audit ran.  Promote the just-persisted plan seed before the
    # same-process finalized validator and any subsequent plan construction;
    # a fresh process derives the identical value directly from final state.
    POLICY_FUNCTIONAL_EXACT_FINITE_SELECTED_SCENARIO_SEED = plan_seed
    authorization = validate_exact_finite_scenario_design_manifest(
        exact_finite_scenario_design_manifest(),
        require_finalized=True,
    )
    if selected_seed is not None and authorization is not True:
        _fail("blind audit finalized state did not authorize its selected fixture")
    if selected_seed is None and authorization is not False:
        _fail("blind audit no-match state unexpectedly authorized production")
    return selected


def _functional_table(
    *,
    plan: ExactFiniteCanaryPlan,
    policy_cells: tuple[ExactFinitePolicyFunctionalCell, ...],
) -> tuple[tuple[str, ExactRational], ...]:
    values: list[tuple[str, ExactRational]] = []
    for state in plan.parameter_states:
        state_values = tuple(
            _fraction(cell.functional_value)
            for cell in policy_cells
            if cell.state_id == state.state_id
        )
        if len(state_values) != plan.scenario_replicate_count:
            _fail("policy lookup table is incomplete for one parameter state")
        values.append(
            (
                state.state_id,
                _exact(sum(state_values, Fraction()) / len(state_values)),
            )
        )
    return tuple(values)


def _observation_category_table(
    *,
    plan: ExactFiniteCanaryPlan,
    observation_cells: tuple[ExactFiniteObservationCell, ...],
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple(
        (
            state.state_id,
            tuple(
                cell.observation_value
                for cell in observation_cells
                if cell.state_id == state.state_id
            ),
        )
        for state in plan.parameter_states
    )


def _scenario_coupling_verified(
    *,
    plan: ExactFiniteCanaryPlan,
    cells: tuple[ExactFinitePolicyFunctionalCell, ...],
) -> bool:
    for scenario_index in range(plan.scenario_replicate_count):
        scenario_cells = tuple(
            item for item in cells if item.scenario_index == scenario_index
        )
        if len(scenario_cells) != len(plan.parameter_states):
            return False
        for field in (
            "event_key",
            "scenario_generation_receipt_sha256",
            "shock_tape_sha256",
            "declared_event_keys_sha256",
            "baseline_consumption_manifest_sha256",
            "candidate_consumption_manifest_sha256",
            "internal_random_role_neutral_ledger_sha256",
            "internal_random_expected_entry_count",
        ):
            if len({getattr(item, field) for item in scenario_cells}) != 1:
                return False
    return True


def _ignored_data_detection(
    *,
    exact: ExactFiniteRankLawDerivation,
    ignored: ExactFiniteRankLawDerivation,
) -> tuple[bool, bool]:
    state_count = len(exact.posterior_rows[0].posterior_by_state)
    prior = Fraction(1, state_count)
    informative_observations = {
        row.observation_value
        for row in exact.posterior_rows
        if any(_fraction(value) != prior for _, value in row.posterior_by_state)
    }
    conditional_detected = any(
        law.observation_value in informative_observations
        and not law.is_discrete_uniform
        for law in ignored.conditional_laws
    )
    marginal_blind_spot = (
        ignored.unconditional_is_discrete_uniform and conditional_detected
    )
    return conditional_detected, marginal_blind_spot


def _expected_result_components(
    *,
    plan: ExactFiniteCanaryPlan,
    observation_cells: tuple[ExactFiniteObservationCell, ...],
    policy_cells: tuple[ExactFinitePolicyFunctionalCell, ...],
) -> tuple[
    tuple[tuple[str, ExactRational], ...],
    ExactFiniteRankLawDerivation,
    ExactFiniteRankLawDerivation,
    bool,
    bool,
    bool,
    bool,
    ExactFiniteNoOpZeroCheck,
    ExactFiniteCanaryDecision,
]:
    functional_values = _functional_table(plan=plan, policy_cells=policy_cells)
    observation_categories = _observation_category_table(
        plan=plan,
        observation_cells=observation_cells,
    )
    exact = derive_exact_finite_rank_laws(
        observation_categories=observation_categories,
        functional_values=functional_values,
        posterior_draw_count=plan.posterior_draw_count,
        posterior_mode=ExactFinitePosteriorMode.EXACT_BAYES,
    )
    ignored = derive_exact_finite_rank_laws(
        observation_categories=observation_categories,
        functional_values=functional_values,
        posterior_draw_count=plan.posterior_draw_count,
        posterior_mode=ExactFinitePosteriorMode.IGNORED_DATA_PRIOR,
    )
    conditional_detected, marginal_blind_spot = _ignored_data_detection(
        exact=exact,
        ignored=ignored,
    )
    functional_nondegenerate = (
        len({_fraction(value) for _, value in functional_values}) > 1
    )
    scenario_audited_design_match = exact_finite_plan_matches_scenario_audited_design(
        plan
    )
    if plan.candidate_is_noop:
        noop_zero_check = (
            ExactFiniteNoOpZeroCheck.PASSED
            if all(_fraction(cell.functional_value) == 0 for cell in policy_cells)
            else ExactFiniteNoOpZeroCheck.FAILED
        )
    else:
        noop_zero_check = ExactFiniteNoOpZeroCheck.NOT_APPLICABLE
    exact_pass = (
        exact.posterior_draw_permutation_exchangeability_verified
        and exact.conditional_truth_draw_exchangeability_exhaustively_verified
        and exact.all_conditionals_are_discrete_uniform
        and exact.unconditional_is_discrete_uniform
        and _fraction(exact.complete_support_probability) == 1
    )
    ignored_joint_nonexchangeable = not (
        ignored.conditional_truth_draw_exchangeability_exhaustively_verified
    )
    posterior_integrity_pass = (
        exact_pass and ignored.posterior_draw_permutation_exchangeability_verified
    )
    if (
        noop_zero_check is ExactFiniteNoOpZeroCheck.FAILED
        or not posterior_integrity_pass
    ):
        decision = ExactFiniteCanaryDecision.FAIL
    elif (
        plan.candidate_is_noop
        or not functional_nondegenerate
        or not scenario_audited_design_match
    ):
        decision = ExactFiniteCanaryDecision.ABSTAIN
    elif (
        exact_pass
        and conditional_detected
        and marginal_blind_spot
        and ignored_joint_nonexchangeable
    ):
        decision = ExactFiniteCanaryDecision.PASS
    else:
        decision = ExactFiniteCanaryDecision.FAIL
    return (
        functional_values,
        exact,
        ignored,
        conditional_detected,
        marginal_blind_spot,
        functional_nondegenerate,
        scenario_audited_design_match,
        noop_zero_check,
        decision,
    )


@dataclass(frozen=True, slots=True, init=False, weakref_slot=True)
class ExactFinitePolicyFunctionalResult:
    """Factory-produced exhaustive result with conservative claim boundaries."""

    plan: ExactFiniteCanaryPlan
    observation_cells: tuple[ExactFiniteObservationCell, ...]
    policy_cells: tuple[ExactFinitePolicyFunctionalCell, ...]
    functional_values: tuple[tuple[str, ExactRational], ...]
    exact_bayes_rank_law: ExactFiniteRankLawDerivation
    ignored_data_rank_law: ExactFiniteRankLawDerivation
    decision: ExactFiniteCanaryDecision
    ignored_data_conditional_canary_detected: bool
    ignored_data_conditional_truth_draw_exchangeability_verified: bool
    ignored_data_conditional_truth_draw_nonexchangeability_detected: bool
    ignored_data_unconditional_marginal_blind_spot_exposed: bool
    functional_nondegenerate: bool
    scenario_audited_design_match: bool
    noop_policy_cellwise_zero_check: ExactFiniteNoOpZeroCheck
    exact_finite_discrete_posterior_rederived: Literal[True]
    conditional_exchangeability_exhaustively_verified: bool
    randomized_tie_rank_uniformity_exact: bool
    complete_finite_support_accounted: Literal[True]
    integrated_policy_functional_lookup_executed_and_freshly_verified: Literal[True]
    shared_exogenous_tape_receipts_verified: Literal[True]
    integrated_abc_posterior_exchangeability_verified: Literal[False]
    continuous_parameter_posterior_exact: Literal[False]
    fixed_key_grid_statistical_independence_verified: Literal[False]
    cross_world_internal_random_coupling_verified: Literal[True]
    integrated_market_mechanism_validity_verified: Literal[False]
    external_preregistration_verified: Literal[False]
    external_custody_verified: Literal[False]
    independent_reproduction: Literal[False]
    standalone_environment_provenance_complete: Literal[False]
    real_world_coverage_claim: Literal[False]
    real_world_causal_claim: Literal[False]
    publication_eligible: Literal[False]
    scientific_abstention_required: Literal[True]
    synthetic_only: Literal[True]
    claim_scope: str
    schema_version: str

    def __init__(self) -> None:
        raise TypeError(
            "ExactFinitePolicyFunctionalResult is factory-produced; use "
            "run_exact_finite_policy_functional_canary"
        )

    def __copy__(self) -> NoReturn:
        raise TypeError("exact finite canary results cannot be copied")

    def __deepcopy__(self, memo: object) -> NoReturn:
        del memo
        raise TypeError("exact finite canary results cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("exact finite canary results cannot be serialized")

    def _fresh_validate(self) -> None:
        if type(self.plan) is not ExactFiniteCanaryPlan:
            raise TypeError("exact finite result lost its exact plan")
        require_exact_finite_canary_plan(self.plan)
        expected_observation_coordinates = tuple(
            (state.state_id, support_index)
            for state in self.plan.parameter_states
            for support_index in range(self.plan.observation_support_size)
        )
        actual_observation_coordinates = tuple(
            (item.state_id, item.support_index) for item in self.observation_cells
        )
        if actual_observation_coordinates != expected_observation_coordinates:
            _fail("observation cells are incomplete or non-canonical")
        for observation_cell in self.observation_cells:
            expected_key = _observation_event_key(
                canary_id=self.plan.canary_id,
                seed=self.plan.seed,
                support_index=observation_cell.support_index,
            ).value
            if observation_cell.event_key != expected_key:
                _fail("observation cell EventKey differs from the frozen grid")
        expected_policy_coordinates = tuple(
            (state.state_id, scenario_index)
            for state in self.plan.parameter_states
            for scenario_index in range(self.plan.scenario_replicate_count)
        )
        actual_policy_coordinates = tuple(
            (item.state_id, item.scenario_index) for item in self.policy_cells
        )
        if actual_policy_coordinates != expected_policy_coordinates:
            _fail("policy-functional cells are incomplete or non-canonical")
        for policy_cell in self.policy_cells:
            expected_key = _policy_event_key(
                canary_id=self.plan.canary_id,
                seed=self.plan.scenario_seed,
                scenario_index=policy_cell.scenario_index,
            ).value
            if policy_cell.event_key != expected_key:
                _fail("policy-functional EventKey differs from the frozen grid")
        scenario_coupling = _scenario_coupling_verified(
            plan=self.plan,
            cells=self.policy_cells,
        )
        if not scenario_coupling:
            _fail("scenario generation or exogenous consumption differs across cells")
        (
            functional_values,
            exact,
            ignored,
            conditional_detected,
            marginal_blind_spot,
            functional_nondegenerate,
            scenario_audited_design_match,
            noop_zero_check,
            decision,
        ) = _expected_result_components(
            plan=self.plan,
            observation_cells=self.observation_cells,
            policy_cells=self.policy_cells,
        )
        if (
            self.functional_values != functional_values
            or self.exact_bayes_rank_law != exact
            or self.ignored_data_rank_law != ignored
            or self.decision is not decision
            or self.ignored_data_conditional_canary_detected is not conditional_detected
            or self.ignored_data_conditional_truth_draw_exchangeability_verified
            is not ignored.conditional_truth_draw_exchangeability_exhaustively_verified
            or self.ignored_data_conditional_truth_draw_nonexchangeability_detected
            is not (
                not ignored.conditional_truth_draw_exchangeability_exhaustively_verified
            )
            or self.ignored_data_unconditional_marginal_blind_spot_exposed
            is not marginal_blind_spot
            or self.functional_nondegenerate is not functional_nondegenerate
            or self.scenario_audited_design_match is not scenario_audited_design_match
            or self.noop_policy_cellwise_zero_check is not noop_zero_check
        ):
            _fail("exact finite posterior, functional, or rank laws do not rederive")
        if self.conditional_exchangeability_exhaustively_verified is not (
            exact.conditional_truth_draw_exchangeability_exhaustively_verified
        ):
            _fail("conditional truth/draw exchangeability claim does not rederive")
        if self.randomized_tie_rank_uniformity_exact is not (
            exact.all_conditionals_are_discrete_uniform
            and exact.unconditional_is_discrete_uniform
        ):
            _fail("randomized-tie rank-uniformity claim does not rederive")
        if self.complete_finite_support_accounted is not (
            _fraction(exact.complete_support_probability) == 1
        ):
            _fail("complete finite-support claim does not rederive")
        expected_true = (
            self.exact_finite_discrete_posterior_rederived,
            self.complete_finite_support_accounted,
            self.integrated_policy_functional_lookup_executed_and_freshly_verified,
            self.shared_exogenous_tape_receipts_verified,
            self.cross_world_internal_random_coupling_verified,
            self.scientific_abstention_required,
            self.synthetic_only,
        )
        if any(value is not True for value in expected_true):
            _fail("exact finite positive evidence flags were weakened")
        expected_false = (
            self.integrated_abc_posterior_exchangeability_verified,
            self.continuous_parameter_posterior_exact,
            self.fixed_key_grid_statistical_independence_verified,
            self.integrated_market_mechanism_validity_verified,
            self.external_preregistration_verified,
            self.external_custody_verified,
            self.independent_reproduction,
            self.standalone_environment_provenance_complete,
            self.real_world_coverage_claim,
            self.real_world_causal_claim,
            self.publication_eligible,
        )
        if any(value is not False for value in expected_false):
            _fail("exact finite scientific claim boundary was overstated")
        if self.claim_scope != POLICY_FUNCTIONAL_EXACT_FINITE_CLAIM_SCOPE:
            _fail("exact finite result claim scope changed")
        if self.schema_version != POLICY_FUNCTIONAL_EXACT_FINITE_SCHEMA_VERSION:
            _fail("unsupported exact finite result schema version")

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_mapping())

    @property
    def sha256(self) -> str:
        return sha256_bytes(self.canonical_bytes)

    def to_mapping(self) -> dict[str, object]:
        self._fresh_validate()
        return {
            "claim_scope": self.claim_scope,
            "complete_finite_support_accounted": (
                self.complete_finite_support_accounted
            ),
            "conditional_exchangeability_exhaustively_verified": (
                self.conditional_exchangeability_exhaustively_verified
            ),
            "continuous_parameter_posterior_exact": (
                self.continuous_parameter_posterior_exact
            ),
            "decision": self.decision.value,
            "exact_bayes_rank_law": self.exact_bayes_rank_law.to_mapping(),
            "exact_finite_discrete_posterior_rederived": (
                self.exact_finite_discrete_posterior_rederived
            ),
            "external_preregistration_verified": self.external_preregistration_verified,
            "external_custody_verified": self.external_custody_verified,
            "fixed_key_grid_statistical_independence_verified": (
                self.fixed_key_grid_statistical_independence_verified
            ),
            "functional_definition": self.plan.functional_definition,
            "functional_nondegenerate": self.functional_nondegenerate,
            "functional_values": [
                {"state_id": state_id, "value": value.to_mapping()}
                for state_id, value in self.functional_values
            ],
            "ignored_data_conditional_canary_detected": (
                self.ignored_data_conditional_canary_detected
            ),
            "ignored_data_conditional_truth_draw_exchangeability_verified": (
                self.ignored_data_conditional_truth_draw_exchangeability_verified
            ),
            "ignored_data_conditional_truth_draw_nonexchangeability_detected": (
                self.ignored_data_conditional_truth_draw_nonexchangeability_detected
            ),
            "ignored_data_rank_law": self.ignored_data_rank_law.to_mapping(),
            "ignored_data_unconditional_marginal_blind_spot_exposed": (
                self.ignored_data_unconditional_marginal_blind_spot_exposed
            ),
            "independent_reproduction": self.independent_reproduction,
            "noop_policy_cellwise_zero_check": (
                self.noop_policy_cellwise_zero_check.value
            ),
            "integrated_abc_posterior_exchangeability_verified": (
                self.integrated_abc_posterior_exchangeability_verified
            ),
            "integrated_market_mechanism_validity_verified": (
                self.integrated_market_mechanism_validity_verified
            ),
            "integrated_policy_functional_lookup_executed_and_freshly_verified": (
                self.integrated_policy_functional_lookup_executed_and_freshly_verified
            ),
            "cross_world_internal_random_coupling_verified": (
                self.cross_world_internal_random_coupling_verified
            ),
            "observation_cells": [item.to_mapping() for item in self.observation_cells],
            "plan": self.plan.to_mapping(),
            "plan_sha256": self.plan.sha256,
            "policy_cells": [item.to_mapping() for item in self.policy_cells],
            "publication_eligible": self.publication_eligible,
            "randomized_tie_rank_uniformity_exact": (
                self.randomized_tie_rank_uniformity_exact
            ),
            "real_world_causal_claim": self.real_world_causal_claim,
            "real_world_coverage_claim": self.real_world_coverage_claim,
            "schema_version": self.schema_version,
            "scenario_audited_design_match": self.scenario_audited_design_match,
            "scientific_abstention_required": self.scientific_abstention_required,
            "standalone_environment_provenance_complete": (
                self.standalone_environment_provenance_complete
            ),
            "shared_exogenous_tape_receipts_verified": (
                self.shared_exogenous_tape_receipts_verified
            ),
            "synthetic_only": self.synthetic_only,
        }


_EXACT_FINITE_RESULTS: WeakValueDictionary[int, ExactFinitePolicyFunctionalResult] = (
    WeakValueDictionary()
)
_EXACT_FINITE_RESULT_SEALS: dict[int, str] = {}
_EXACT_FINITE_RESULT_LOCK = RLock()


def _forget_exact_finite_result(identity: int) -> None:
    with _EXACT_FINITE_RESULT_LOCK:
        if _EXACT_FINITE_RESULTS.get(identity) is None:
            _EXACT_FINITE_RESULT_SEALS.pop(identity, None)


def _register_exact_finite_result(
    result: ExactFinitePolicyFunctionalResult,
) -> None:
    seal = result.sha256
    identity = id(result)
    with _EXACT_FINITE_RESULT_LOCK:
        _EXACT_FINITE_RESULTS[identity] = result
        _EXACT_FINITE_RESULT_SEALS[identity] = seal
    finalize(result, _forget_exact_finite_result, identity)


def require_exact_finite_policy_functional_result(
    result: ExactFinitePolicyFunctionalResult,
) -> ExactFinitePolicyFunctionalResult:
    """Require a live factory result and its unchanged source-bound seal."""
    if type(result) is not ExactFinitePolicyFunctionalResult:
        raise TypeError("result must be an exact finite policy-functional result")
    try:
        before = result.sha256
    except Exception as error:
        raise ExactFinitePolicyFunctionalError(
            "exact finite result cannot reproduce its mint seal"
        ) from error
    with _EXACT_FINITE_RESULT_LOCK:
        registered = _EXACT_FINITE_RESULTS.get(id(result))
        seal = _EXACT_FINITE_RESULT_SEALS.get(id(result))
    if registered is not result or seal != before:
        _fail("exact finite result is unissued or differs from its factory mint seal")
    result._fresh_validate()
    after = result.sha256
    with _EXACT_FINITE_RESULT_LOCK:
        still_registered = _EXACT_FINITE_RESULTS.get(id(result))
        still_sealed = _EXACT_FINITE_RESULT_SEALS.get(id(result))
    if still_registered is not result or still_sealed != after or before != after:
        _fail("exact finite result changed during fresh mint-seal validation")
    return result


def run_exact_finite_policy_functional_canary(
    plan: ExactFiniteCanaryPlan,
    *,
    parallel: bool = False,
) -> ExactFinitePolicyFunctionalResult:
    """Run every finite observation and protocol-bound policy lookup cell."""
    if type(plan) is not ExactFiniteCanaryPlan:
        raise TypeError("plan must be an exact ExactFiniteCanaryPlan")
    if not isinstance(parallel, bool):
        raise TypeError("parallel must be boolean")
    plan = require_exact_finite_canary_plan(plan)
    production_authorized = validate_exact_finite_scenario_design_manifest(
        exact_finite_scenario_design_manifest(), require_finalized=True
    )
    if not production_authorized:
        _fail("completed scenario design does not authorize production execution")
    observation_coordinates = tuple(
        (state, support_index)
        for state in plan.parameter_states
        for support_index in range(plan.observation_support_size)
    )
    policy_coordinates = tuple(
        (state, scenario_index)
        for state in plan.parameter_states
        for scenario_index in range(plan.scenario_replicate_count)
    )
    if parallel:
        with ThreadPoolExecutor(
            max_workers=min(8, len(observation_coordinates))
        ) as pool:
            observation_cells = tuple(
                pool.map(
                    lambda coordinate: _observation_cell(plan, *coordinate),
                    observation_coordinates,
                )
            )
        with ThreadPoolExecutor(max_workers=min(8, len(policy_coordinates))) as pool:
            policy_cells = tuple(
                pool.map(
                    lambda coordinate: _policy_cell(
                        plan,
                        *coordinate,
                        parallel_worlds=True,
                    ),
                    policy_coordinates,
                )
            )
    else:
        observation_cells = tuple(
            _observation_cell(plan, state, support_index)
            for state, support_index in observation_coordinates
        )
        policy_cells = tuple(
            _policy_cell(
                plan,
                state,
                scenario_index,
                parallel_worlds=False,
            )
            for state, scenario_index in policy_coordinates
        )
    (
        functional_values,
        exact,
        ignored,
        conditional_detected,
        marginal_blind_spot,
        functional_nondegenerate,
        scenario_audited_design_match,
        noop_zero_check,
        decision,
    ) = _expected_result_components(
        plan=plan,
        observation_cells=observation_cells,
        policy_cells=policy_cells,
    )
    result = object.__new__(ExactFinitePolicyFunctionalResult)
    values: tuple[tuple[str, object], ...] = (
        ("plan", plan),
        ("observation_cells", observation_cells),
        ("policy_cells", policy_cells),
        ("functional_values", functional_values),
        ("exact_bayes_rank_law", exact),
        ("ignored_data_rank_law", ignored),
        ("decision", decision),
        ("ignored_data_conditional_canary_detected", conditional_detected),
        (
            "ignored_data_conditional_truth_draw_exchangeability_verified",
            ignored.conditional_truth_draw_exchangeability_exhaustively_verified,
        ),
        (
            "ignored_data_conditional_truth_draw_nonexchangeability_detected",
            not ignored.conditional_truth_draw_exchangeability_exhaustively_verified,
        ),
        (
            "ignored_data_unconditional_marginal_blind_spot_exposed",
            marginal_blind_spot,
        ),
        ("functional_nondegenerate", functional_nondegenerate),
        ("scenario_audited_design_match", scenario_audited_design_match),
        ("noop_policy_cellwise_zero_check", noop_zero_check),
        ("exact_finite_discrete_posterior_rederived", True),
        (
            "conditional_exchangeability_exhaustively_verified",
            exact.conditional_truth_draw_exchangeability_exhaustively_verified,
        ),
        (
            "randomized_tie_rank_uniformity_exact",
            exact.all_conditionals_are_discrete_uniform
            and exact.unconditional_is_discrete_uniform,
        ),
        ("complete_finite_support_accounted", True),
        (
            "integrated_policy_functional_lookup_executed_and_freshly_verified",
            True,
        ),
        ("shared_exogenous_tape_receipts_verified", True),
        ("integrated_abc_posterior_exchangeability_verified", False),
        ("continuous_parameter_posterior_exact", False),
        ("fixed_key_grid_statistical_independence_verified", False),
        ("cross_world_internal_random_coupling_verified", True),
        ("integrated_market_mechanism_validity_verified", False),
        ("external_preregistration_verified", False),
        ("external_custody_verified", False),
        ("independent_reproduction", False),
        ("standalone_environment_provenance_complete", False),
        ("real_world_coverage_claim", False),
        ("real_world_causal_claim", False),
        ("publication_eligible", False),
        ("scientific_abstention_required", True),
        ("synthetic_only", True),
        ("claim_scope", POLICY_FUNCTIONAL_EXACT_FINITE_CLAIM_SCOPE),
        ("schema_version", POLICY_FUNCTIONAL_EXACT_FINITE_SCHEMA_VERSION),
    )
    for field, value in values:
        object.__setattr__(result, field, value)
    result._fresh_validate()
    _register_exact_finite_result(result)
    return result


def replay_exact_finite_policy_functional_canary(
    source: ExactFinitePolicyFunctionalResult,
    *,
    parallel: bool = False,
) -> ExactFinitePolicyFunctionalResult:
    """Re-execute every integrated cell and require byte-identical evidence."""
    if type(source) is not ExactFinitePolicyFunctionalResult:
        raise TypeError("source must be an exact finite canary result")
    source = require_exact_finite_policy_functional_result(source)
    replayed = run_exact_finite_policy_functional_canary(
        source.plan,
        parallel=parallel,
    )
    if replayed.canonical_bytes != source.canonical_bytes:
        _fail("exact finite canary replay differs from the source result")
    return replayed


__all__ = [
    "POLICY_FUNCTIONAL_EXACT_FINITE_CLAIM_SCOPE",
    "POLICY_FUNCTIONAL_EXACT_FINITE_FUNCTIONAL",
    "POLICY_FUNCTIONAL_EXACT_FINITE_SCENARIO_CANDIDATE_SEEDS",
    "POLICY_FUNCTIONAL_EXACT_FINITE_SCENARIO_SELECTION_RULE",
    "POLICY_FUNCTIONAL_EXACT_FINITE_SCHEMA_VERSION",
    "POLICY_FUNCTIONAL_EXACT_FINITE_SELECTED_SCENARIO_SEED",
    "ExactConditionalRankLaw",
    "ExactFiniteCanaryDecision",
    "ExactFiniteCanaryPlan",
    "ExactFiniteNoOpZeroCheck",
    "ExactFiniteObservationCell",
    "ExactFiniteParameterState",
    "ExactFinitePolicyFunctionalCell",
    "ExactFinitePolicyFunctionalError",
    "ExactFinitePolicyFunctionalResult",
    "ExactFinitePosteriorMode",
    "ExactFiniteRankLawDerivation",
    "ExactPosteriorMassRow",
    "ExactRankEnumerationAtom",
    "build_exact_finite_policy_functional_plan",
    "derive_exact_finite_scenario_blind_audit_output_receipt",
    "derive_exact_finite_scenario_audit_statuses",
    "derive_exact_finite_rank_laws",
    "exact_finite_plan_matches_scenario_audited_design",
    "exact_finite_scenario_audited_design_sha256",
    "exact_finite_scenario_design_manifest",
    "require_exact_finite_canary_plan",
    "require_exact_finite_policy_functional_result",
    "replay_exact_finite_policy_functional_canary",
    "run_exact_finite_policy_functional_canary",
    "validate_exact_finite_scenario_design_manifest",
]
