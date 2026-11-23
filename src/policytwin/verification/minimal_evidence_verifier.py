"""Standard-library-only verifier for a declarative evidence capsule.

This deliberately narrow verifier does not import the simulator, contract
compiler, replay implementation, or reporting gate.  It verifies a materialized
expected-event set and execution commitments.  A malicious or incorrect
contract generator remains outside its guarantee.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import NoReturn, cast

MINIMAL_EVIDENCE_CAPSULE_SCHEMA_VERSION = "policytwin.minimal-evidence-capsule.v1"
_WORLDS = ("baseline", "policy")
_SCOPES = {"shared", "baseline_only", "policy_only"}
_REQUIREMENTS = {"required", "optional"}
_STATES = {"consumed", "masked"}


class MinimalVerifierError(ValueError):
    """Raised when the capsule itself violates the closed interchange schema."""


def _fail(message: str) -> NoReturn:
    raise MinimalVerifierError(message)


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        _fail(f"{field} must be an object")
    return cast(Mapping[str, object], value)


def _sequence(value: object, field: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        _fail(f"{field} must be an array")
    return cast(Sequence[object], value)


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{field} must be a non-empty string")
    return value


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{field} must be an integer")
    return value


def canonical_json_bytes(value: object) -> bytes:
    """Return the single canonical byte representation accepted by the capsule."""
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise MinimalVerifierError("capsule is not canonical JSON data") from error


def capsule_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class MinimalVerificationResult:
    accepted: bool
    violations: tuple[str, ...]
    capsule_sha256: str

    def to_mapping(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "capsule_sha256": self.capsule_sha256,
            "violations": list(self.violations),
        }


def _applies(scope: str, world: str) -> bool:
    return scope == "shared" or scope == f"{world}_only"


def _event_rows(raw: object) -> tuple[Mapping[str, object], ...]:
    rows = tuple(
        _mapping(item, "contract event") for item in _sequence(raw, "contract.events")
    )
    if not rows:
        _fail("contract.events must not be empty")
    for row in rows:
        if frozenset(row) != {
            "event_identity",
            "metadata",
            "requirement",
            "world_scope",
        }:
            _fail("contract event has missing or unknown fields")
        scope = _string(row["world_scope"], "world_scope")
        requirement = _string(row["requirement"], "requirement")
        if scope not in _SCOPES or requirement not in _REQUIREMENTS:
            _fail("contract event has an unsupported scope or requirement")
        _mapping(row["metadata"], "event metadata")
    identities = tuple(_string(row["event_identity"], "event_identity") for row in rows)
    if len(set(identities)) != len(identities) or identities != tuple(
        sorted(identities)
    ):
        _fail("contract identities must be unique and canonically ordered")
    return rows


def _observation_rows(raw: object, world: str) -> tuple[Mapping[str, object], ...]:
    rows = tuple(
        _mapping(item, f"observations.{world} item")
        for item in _sequence(raw, f"observations.{world}")
    )
    for row in rows:
        if frozenset(row) != {
            "event_identity",
            "metadata",
            "state",
            "value_commitment",
        }:
            _fail("observation has missing or unknown fields")
        state = _string(row["state"], "observation.state")
        if state not in _STATES:
            _fail("observation has an unsupported state")
        commitment = row["value_commitment"]
        if state == "consumed" and (not isinstance(commitment, str) or not commitment):
            _fail("consumed observation requires a value commitment")
        if state == "masked" and commitment is not None:
            _fail("masked observation must not carry a value commitment")
        _mapping(row["metadata"], "observation metadata")
    return rows


def _verify_world(
    world: str,
    events: tuple[Mapping[str, object], ...],
    observations: tuple[Mapping[str, object], ...],
) -> list[str]:
    violations: list[str] = []
    by_identity = {
        _string(event["event_identity"], "event_identity"): event for event in events
    }
    counts = Counter(
        _string(row["event_identity"], "observation.event_identity")
        for row in observations
    )
    observed_by_identity: dict[str, list[Mapping[str, object]]] = {}
    for row in observations:
        observed_by_identity.setdefault(str(row["event_identity"]), []).append(row)
    for identity, count in counts.items():
        if identity not in by_identity:
            violations.append(f"unexpected:{world}:{identity}")
        if count > 1:
            violations.append(f"duplicate:{world}:{identity}")
    for identity, event in by_identity.items():
        applies = _applies(str(event["world_scope"]), world)
        selected = observed_by_identity.get(identity, [])
        if not applies and selected:
            violations.append(f"world_scope:{world}:{identity}")
            continue
        if not applies:
            continue
        if not selected:
            if event["requirement"] == "required":
                violations.append(f"missing:{world}:{identity}")
            continue
        for row in selected:
            if row["metadata"] != event["metadata"]:
                violations.append(f"metadata:{world}:{identity}")
            if event["requirement"] == "required" and row["state"] != "consumed":
                violations.append(f"required_status:{world}:{identity}")
    return violations


def verify_capsule(capsule: Mapping[str, object]) -> MinimalVerificationResult:
    """Verify event-set, replay, output, metric, and cutoff commitments."""
    if frozenset(capsule) != {
        "contract",
        "execution",
        "observations",
        "schema_version",
    }:
        _fail("capsule has missing or unknown top-level fields")
    if capsule["schema_version"] != MINIMAL_EVIDENCE_CAPSULE_SCHEMA_VERSION:
        _fail("unsupported minimal evidence capsule schema")
    contract = _mapping(capsule["contract"], "contract")
    if frozenset(contract) != {"events", "sha256"}:
        _fail("contract has missing or unknown fields")
    events = _event_rows(contract["events"])
    expected_contract_digest = capsule_sha256([dict(item) for item in events])
    violations: list[str] = []
    if contract["sha256"] != expected_contract_digest:
        violations.append("contract_commitment")

    observations = _mapping(capsule["observations"], "observations")
    if frozenset(observations) != set(_WORLDS):
        _fail("observations must contain baseline and policy")
    by_world = {
        world: _observation_rows(observations[world], world) for world in _WORLDS
    }
    for world in _WORLDS:
        violations.extend(_verify_world(world, events, by_world[world]))

    shared = {
        str(event["event_identity"])
        for event in events
        if event["world_scope"] == "shared"
    }
    baseline_values = {
        str(row["event_identity"]): row["value_commitment"]
        for row in by_world["baseline"]
        if row["state"] == "consumed"
    }
    policy_values = {
        str(row["event_identity"]): row["value_commitment"]
        for row in by_world["policy"]
        if row["state"] == "consumed"
    }
    for identity in sorted(shared & baseline_values.keys() & policy_values.keys()):
        if baseline_values[identity] != policy_values[identity]:
            violations.append(f"shared_value:{identity}")

    execution = _mapping(capsule["execution"], "execution")
    if frozenset(execution) != {
        "cutoff_epoch",
        "expected_command_sha256",
        "expected_manifest_sha256",
        "expected_output_sha256",
        "max_available_epoch",
        "observed_command_sha256",
        "observed_manifest_sha256",
        "observed_output_sha256",
        "present_metrics",
        "required_metrics",
    }:
        _fail("execution has missing or unknown fields")
    for stem in ("manifest", "command", "output"):
        if execution[f"expected_{stem}_sha256"] != execution[f"observed_{stem}_sha256"]:
            violations.append(f"{stem}_commitment")
    required_metrics = {
        _string(item, "required metric")
        for item in _sequence(execution["required_metrics"], "required_metrics")
    }
    present_metrics = {
        _string(item, "present metric")
        for item in _sequence(execution["present_metrics"], "present_metrics")
    }
    if missing := sorted(required_metrics - present_metrics):
        violations.extend(f"missing_metric:{item}" for item in missing)
    if _integer(execution["max_available_epoch"], "max_available_epoch") > _integer(
        execution["cutoff_epoch"], "cutoff_epoch"
    ):
        violations.append("post_cutoff")
    unique = tuple(sorted(set(violations)))
    return MinimalVerificationResult(
        accepted=not unique,
        violations=unique,
        capsule_sha256=capsule_sha256(dict(capsule)),
    )


__all__ = [
    "MINIMAL_EVIDENCE_CAPSULE_SCHEMA_VERSION",
    "MinimalVerificationResult",
    "MinimalVerifierError",
    "canonical_json_bytes",
    "capsule_sha256",
    "verify_capsule",
]
