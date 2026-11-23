"""Audited synthetic smoke run for the three RNG coupling mechanisms.

This module is deliberately separate from :mod:`policytwin.experiments.matrix`.
It executes only a small synthetic randomness-coupling diagnostic and makes no
claim that any market, agent, social, memory, calibration, or LLM mechanism ran.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Literal, NoReturn, Self, cast

import policytwin.counterfactual.events as events_module
import policytwin.counterfactual.randomness as randomness_module
from policytwin.counterfactual.events import (
    EventType,
    HashBasedRNG,
    SemanticEventKey,
)
from policytwin.counterfactual.randomness import EventKey, deterministic_int
from policytwin.provenance import (
    canonical_json_bytes,
    read_json_object,
    sha256_bytes,
    sha256_file,
    write_json_file,
    write_text_file,
)

MECHANISM_SMOKE_SCHEMA_VERSION = "policytwin.synthetic-rng-mechanism-smoke.v1"
MECHANISM_RECEIPT_SCHEMA_VERSION = "policytwin.synthetic-rng-receipt.v1"
MECHANISM_CLAIM_SCOPE = "synthetic_rng_coupling_mechanism_only"
MECHANISM_EXECUTOR_ID = "policytwin.synthetic_rng_coupling.v1"

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SEMANTIC_EVENT_KEY = re.compile(r"^pse1:[0-9a-f]{64}$")


class MechanismSmokeError(RuntimeError):
    """Base class for mechanism-smoke failures."""


class MechanismSmokeSchemaError(MechanismSmokeError, ValueError):
    """Raised when a mechanism specification or artifact is malformed."""


class MechanismSmokeIntegrityError(MechanismSmokeError):
    """Raised when a receipt cannot be reproduced from trusted inputs."""


def _fail(message: str) -> NoReturn:
    raise MechanismSmokeSchemaError(message)


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        _fail(f"{field} must be a stable ASCII identifier")
    return value


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail(f"{field} must be a lowercase SHA-256 digest")
    return value


def _string(value: object, field: str) -> str:
    if not isinstance(value, str):
        _fail(f"{field} must be a string")
    return value


def _integer(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"{field} must be an integer >= {minimum}")
    return value


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        _fail(f"{field} must be an object with string keys")
    return cast(Mapping[str, object], value)


def _array(value: object, field: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        _fail(f"{field} must be an array")
    return cast(Sequence[object], value)


def _strict_fields(
    value: Mapping[str, object], expected: frozenset[str], context: str
) -> None:
    if frozenset(value) != expected:
        _fail(f"{context} has missing or unknown fields")


class RNGCouplingMode(StrEnum):
    """The three actually executed synthetic coupling implementations."""

    INDEPENDENT = "independent"
    SAME_SEQUENTIAL = "same_sequential"
    EVENT_KEYED = "event_keyed"


class SeedCoupling(StrEnum):
    INDEPENDENT = "independent_world_seeds"
    SHARED = "shared_world_seed"


class DrawAddressing(StrEnum):
    SEQUENTIAL_CURSOR = "sequential_cursor"
    SEMANTIC_EVENT_KEY = "semantic_event_key"


@dataclass(frozen=True, slots=True)
class MechanismSmokeSpec:
    """Immutable preregistered input for a tiny synthetic diagnostic."""

    scenario_id: str = "synthetic-rng-mechanism-smoke"
    seed: str = "policytwin-mechanism-smoke-v1"
    replicate_count: int = 4
    step_count: int = 8
    branch_step: int = 3
    branch_draw_count: int = 3
    policy_effect: int = 7
    synthetic_only: Literal[True] = True

    def __post_init__(self) -> None:
        _identifier(self.scenario_id, "scenario_id")
        if not isinstance(self.seed, str) or not self.seed or len(self.seed) > 512:
            _fail("seed must contain between one and 512 characters")
        _integer(self.replicate_count, "replicate_count", minimum=2)
        _integer(self.step_count, "step_count", minimum=2)
        _integer(self.branch_step, "branch_step")
        if self.branch_step >= self.step_count - 1:
            _fail("branch_step must leave at least one later exogenous event")
        _integer(self.branch_draw_count, "branch_draw_count", minimum=1)
        if isinstance(self.policy_effect, bool) or not isinstance(
            self.policy_effect, int
        ):
            _fail("policy_effect must be an integer")
        if self.synthetic_only is not True:
            _fail("mechanism smoke input must be synthetic_only")

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def to_mapping(self) -> dict[str, object]:
        return {
            "branch_draw_count": self.branch_draw_count,
            "branch_step": self.branch_step,
            "policy_effect": self.policy_effect,
            "replicate_count": self.replicate_count,
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "step_count": self.step_count,
            "synthetic_only": self.synthetic_only,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(
            value,
            frozenset(
                {
                    "branch_draw_count",
                    "branch_step",
                    "policy_effect",
                    "replicate_count",
                    "scenario_id",
                    "seed",
                    "step_count",
                    "synthetic_only",
                }
            ),
            "mechanism smoke spec",
        )
        seed = value["seed"]
        synthetic_only = value["synthetic_only"]
        if not isinstance(seed, str):
            _fail("seed must be a string")
        if synthetic_only is not True:
            _fail("mechanism smoke spec must be synthetic_only")
        return cls(
            scenario_id=_identifier(value["scenario_id"], "scenario_id"),
            seed=seed,
            replicate_count=_integer(value["replicate_count"], "replicate_count"),
            step_count=_integer(value["step_count"], "step_count"),
            branch_step=_integer(value["branch_step"], "branch_step"),
            branch_draw_count=_integer(value["branch_draw_count"], "branch_draw_count"),
            policy_effect=_integer(
                value["policy_effect"], "policy_effect", minimum=-(1 << 63)
            ),
            synthetic_only=True,
        )


@dataclass(frozen=True, slots=True)
class MechanismSwitches:
    """Switches set by the executor, rather than copied from factor labels."""

    mode: RNGCouplingMode
    seed_coupling: SeedCoupling
    draw_addressing: DrawAddressing
    candidate_branch_consumes_stream_positions: bool

    def __post_init__(self) -> None:
        if not isinstance(self.mode, RNGCouplingMode):
            _fail("mode must be an RNGCouplingMode")
        if not isinstance(self.seed_coupling, SeedCoupling):
            _fail("seed_coupling must be a SeedCoupling")
        if not isinstance(self.draw_addressing, DrawAddressing):
            _fail("draw_addressing must be a DrawAddressing")
        if not isinstance(self.candidate_branch_consumes_stream_positions, bool):
            _fail("candidate branch cursor switch must be a boolean")

    def to_mapping(self) -> dict[str, object]:
        return {
            "candidate_branch_consumes_stream_positions": (
                self.candidate_branch_consumes_stream_positions
            ),
            "draw_addressing": self.draw_addressing.value,
            "mode": self.mode.value,
            "seed_coupling": self.seed_coupling.value,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(
            value,
            frozenset(
                {
                    "candidate_branch_consumes_stream_positions",
                    "draw_addressing",
                    "mode",
                    "seed_coupling",
                }
            ),
            "mechanism switches",
        )
        branch = value["candidate_branch_consumes_stream_positions"]
        if not isinstance(branch, bool):
            _fail("candidate branch cursor switch must be a boolean")
        try:
            return cls(
                mode=RNGCouplingMode(_string(value["mode"], "mode")),
                seed_coupling=SeedCoupling(
                    _string(value["seed_coupling"], "seed_coupling")
                ),
                draw_addressing=DrawAddressing(
                    _string(value["draw_addressing"], "draw_addressing")
                ),
                candidate_branch_consumes_stream_positions=branch,
            )
        except (TypeError, ValueError) as error:
            raise MechanismSmokeSchemaError("unknown mechanism switch") from error


def _switches(mode: RNGCouplingMode) -> MechanismSwitches:
    if mode is RNGCouplingMode.INDEPENDENT:
        return MechanismSwitches(
            mode=mode,
            seed_coupling=SeedCoupling.INDEPENDENT,
            draw_addressing=DrawAddressing.SEQUENTIAL_CURSOR,
            candidate_branch_consumes_stream_positions=True,
        )
    if mode is RNGCouplingMode.SAME_SEQUENTIAL:
        return MechanismSwitches(
            mode=mode,
            seed_coupling=SeedCoupling.SHARED,
            draw_addressing=DrawAddressing.SEQUENTIAL_CURSOR,
            candidate_branch_consumes_stream_positions=True,
        )
    return MechanismSwitches(
        mode=mode,
        seed_coupling=SeedCoupling.SHARED,
        draw_addressing=DrawAddressing.SEMANTIC_EVENT_KEY,
        candidate_branch_consumes_stream_positions=False,
    )


def _source_components() -> tuple[tuple[str, str], ...]:
    raw_paths = (
        ("policytwin.counterfactual.events", events_module.__file__),
        ("policytwin.counterfactual.randomness", randomness_module.__file__),
        ("policytwin.experiments.mechanism_smoke", __file__),
    )
    components: list[tuple[str, str]] = []
    for module_name, raw_path in raw_paths:
        if raw_path is None:
            raise MechanismSmokeIntegrityError(
                f"cannot locate executor source for {module_name}"
            )
        path = Path(raw_path)
        if not path.is_file():
            raise MechanismSmokeIntegrityError(
                f"executor source is unavailable for {module_name}"
            )
        components.append((module_name, sha256_file(path)))
    return tuple(sorted(components))


def _executor_code_sha256(components: tuple[tuple[str, str], ...]) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            [
                {"module": module, "source_sha256": digest}
                for module, digest in components
            ]
        )
    )


def _semantic_keys(
    spec: MechanismSmokeSpec, replicate_id: int
) -> tuple[SemanticEventKey, ...]:
    start = datetime(2000, 1, 1, tzinfo=UTC)
    return tuple(
        SemanticEventKey(
            scenario_id=spec.scenario_id,
            event_type=EventType.EXOGENOUS_SHOCK,
            entity_id=f"synthetic_fundamental_r{replicate_id}",
            timestamp=start + timedelta(seconds=step),
        )
        for step in range(spec.step_count)
    )


def _sequential_draw(
    spec: MechanismSmokeSpec, *, seed: str, replicate_id: int, cursor: int
) -> int:
    key = EventKey(
        experiment_id=spec.scenario_id,
        seed=seed,
        replicate_id=replicate_id,
        simulation_step=cursor,
        purpose="synthetic_sequential_stream",
    )
    return deterministic_int(key, -1_000, 1_000)


def _event_keyed_branch_draws(
    spec: MechanismSmokeSpec, *, replicate_id: int, rng: HashBasedRNG
) -> tuple[int, ...]:
    start = datetime(2000, 1, 1, tzinfo=UTC)
    return tuple(
        rng.integer(
            SemanticEventKey(
                scenario_id=spec.scenario_id,
                event_type=EventType.AGENT_DECISION,
                entity_id=f"candidate_branch_r{replicate_id}_{draw_index}",
                timestamp=start
                + timedelta(seconds=spec.branch_step, microseconds=draw_index + 1),
                agent_id="candidate_agent",
            ),
            -1_000,
            1_000,
            domain="synthetic_internal_branch",
        )
        for draw_index in range(spec.branch_draw_count)
    )


@dataclass(frozen=True, slots=True)
class MechanismReceipt:
    """Content-addressed evidence emitted by one actually executed RNG mode."""

    mode: RNGCouplingMode
    config_sha256: str
    executor_code_sha256: str
    code_components: tuple[tuple[str, str], ...]
    actual_switches: MechanismSwitches
    input_sha256: str
    event_keys_sha256: str
    output_sha256: str
    event_keys: tuple[str, ...]
    baseline_draws: tuple[int, ...]
    candidate_draws: tuple[int, ...]
    differences: tuple[int, ...]
    internal_branch_draws_sha256: tuple[str, ...]
    mechanism_execution: Literal[True] = True
    claim_scope: str = MECHANISM_CLAIM_SCOPE
    synthetic_only: Literal[True] = True
    schema_version: str = MECHANISM_RECEIPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.mode, RNGCouplingMode):
            _fail("receipt mode must be an RNGCouplingMode")
        for field in (
            "config_sha256",
            "executor_code_sha256",
            "input_sha256",
            "event_keys_sha256",
            "output_sha256",
        ):
            _sha256(getattr(self, field), field)
        if not isinstance(self.code_components, tuple) or not self.code_components:
            _fail("receipt code_components must be a non-empty tuple")
        normalized_components = tuple(sorted(self.code_components))
        if len({name for name, _ in normalized_components}) != len(
            normalized_components
        ):
            _fail("receipt code component modules must be unique")
        for name, digest in normalized_components:
            _identifier(name, "code component module")
            _sha256(digest, "code component source_sha256")
        object.__setattr__(self, "code_components", normalized_components)
        if self.executor_code_sha256 != _executor_code_sha256(normalized_components):
            _fail("executor_code_sha256 does not match code_components")
        if self.actual_switches != _switches(self.mode):
            _fail("actual_switches do not match the executed mode")
        lengths = {
            len(self.event_keys),
            len(self.baseline_draws),
            len(self.candidate_draws),
        }
        if len(lengths) != 1 or not self.event_keys:
            _fail("receipt event keys and draws must be non-empty and aligned")
        if any(
            not isinstance(key, str) or _SEMANTIC_EVENT_KEY.fullmatch(key) is None
            for key in self.event_keys
        ):
            _fail("receipt contains an invalid semantic event key")
        if len(set(self.event_keys)) != len(self.event_keys):
            _fail("receipt semantic event keys must be unique")
        if len(self.differences) == 0 or len(self.internal_branch_draws_sha256) != len(
            self.differences
        ):
            _fail("receipt replicate summaries are not aligned")
        for value in (*self.baseline_draws, *self.candidate_draws, *self.differences):
            if isinstance(value, bool) or not isinstance(value, int):
                _fail("receipt draws and differences must be integers")
        for digest in self.internal_branch_draws_sha256:
            _sha256(digest, "internal branch draws sha256")
        expected_event_hash = sha256_bytes(canonical_json_bytes(list(self.event_keys)))
        if self.event_keys_sha256 != expected_event_hash:
            _fail("event_keys_sha256 does not match event_keys")
        if self.output_sha256 != self._computed_output_sha256():
            _fail("output_sha256 does not match receipt output")
        if self.mechanism_execution is not True:
            _fail("executed mechanism receipt must set mechanism_execution=true")
        if self.claim_scope != MECHANISM_CLAIM_SCOPE:
            _fail("mechanism receipt claim scope cannot be widened")
        if self.synthetic_only is not True:
            _fail("mechanism receipt must be synthetic_only")
        if self.schema_version != MECHANISM_RECEIPT_SCHEMA_VERSION:
            _fail("unsupported mechanism receipt schema")

    def _computed_output_sha256(self) -> str:
        return sha256_bytes(
            canonical_json_bytes(
                {
                    "baseline_draws": list(self.baseline_draws),
                    "candidate_draws": list(self.candidate_draws),
                    "differences": list(self.differences),
                    "internal_branch_draws_sha256": list(
                        self.internal_branch_draws_sha256
                    ),
                }
            )
        )

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def to_mapping(self) -> dict[str, object]:
        return {
            "actual_switches": self.actual_switches.to_mapping(),
            "baseline_draws": list(self.baseline_draws),
            "candidate_draws": list(self.candidate_draws),
            "claim_scope": self.claim_scope,
            "code_components": [
                {"module": module, "source_sha256": digest}
                for module, digest in self.code_components
            ],
            "config_sha256": self.config_sha256,
            "differences": list(self.differences),
            "event_keys": list(self.event_keys),
            "event_keys_sha256": self.event_keys_sha256,
            "executor_code_sha256": self.executor_code_sha256,
            "input_sha256": self.input_sha256,
            "internal_branch_draws_sha256": list(self.internal_branch_draws_sha256),
            "mechanism_execution": self.mechanism_execution,
            "mode": self.mode.value,
            "output_sha256": self.output_sha256,
            "schema_version": self.schema_version,
            "synthetic_only": self.synthetic_only,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(
            value,
            frozenset(
                {
                    "actual_switches",
                    "baseline_draws",
                    "candidate_draws",
                    "claim_scope",
                    "code_components",
                    "config_sha256",
                    "differences",
                    "event_keys",
                    "event_keys_sha256",
                    "executor_code_sha256",
                    "input_sha256",
                    "internal_branch_draws_sha256",
                    "mechanism_execution",
                    "mode",
                    "output_sha256",
                    "schema_version",
                    "synthetic_only",
                }
            ),
            "mechanism receipt",
        )
        raw_components = _array(value["code_components"], "code_components")
        components: list[tuple[str, str]] = []
        for raw_component in raw_components:
            component = _mapping(raw_component, "code component")
            _strict_fields(
                component,
                frozenset({"module", "source_sha256"}),
                "code component",
            )
            components.append(
                (
                    _identifier(component["module"], "code component module"),
                    _sha256(component["source_sha256"], "code component source_sha256"),
                )
            )
        if value["mechanism_execution"] is not True:
            _fail("mechanism_execution must be true for an executed receipt")
        if value["synthetic_only"] is not True:
            _fail("mechanism receipt must be synthetic_only")
        mode_value = _string(value["mode"], "mode")
        try:
            mode = RNGCouplingMode(mode_value)
        except (TypeError, ValueError) as error:
            raise MechanismSmokeSchemaError("unknown receipt mode") from error
        return cls(
            mode=mode,
            config_sha256=_sha256(value["config_sha256"], "config_sha256"),
            executor_code_sha256=_sha256(
                value["executor_code_sha256"], "executor_code_sha256"
            ),
            code_components=tuple(components),
            actual_switches=MechanismSwitches.from_mapping(
                _mapping(value["actual_switches"], "actual_switches")
            ),
            input_sha256=_sha256(value["input_sha256"], "input_sha256"),
            event_keys_sha256=_sha256(value["event_keys_sha256"], "event_keys_sha256"),
            output_sha256=_sha256(value["output_sha256"], "output_sha256"),
            event_keys=tuple(
                cast(str, item) for item in _array(value["event_keys"], "event_keys")
            ),
            baseline_draws=tuple(
                _integer(item, "baseline draw", minimum=-1_000)
                for item in _array(value["baseline_draws"], "baseline_draws")
            ),
            candidate_draws=tuple(
                _integer(item, "candidate draw", minimum=-1_000)
                for item in _array(value["candidate_draws"], "candidate_draws")
            ),
            differences=tuple(
                _integer(item, "difference", minimum=-(1 << 63))
                for item in _array(value["differences"], "differences")
            ),
            internal_branch_draws_sha256=tuple(
                _sha256(item, "internal branch draws sha256")
                for item in _array(
                    value["internal_branch_draws_sha256"],
                    "internal_branch_draws_sha256",
                )
            ),
            mechanism_execution=True,
            claim_scope=_identifier(value["claim_scope"], "claim_scope"),
            synthetic_only=True,
            schema_version=cast(str, value["schema_version"]),
        )


def _execute_mode(
    spec: MechanismSmokeSpec,
    mode: RNGCouplingMode,
    components: tuple[tuple[str, str], ...],
) -> MechanismReceipt:
    all_keys: list[str] = []
    baseline_draws: list[int] = []
    candidate_draws: list[int] = []
    differences: list[int] = []
    branch_hashes: list[str] = []
    input_replicates: list[dict[str, object]] = []

    for replicate_id in range(spec.replicate_count):
        semantic_keys = _semantic_keys(spec, replicate_id)
        key_values = tuple(key.value for key in semantic_keys)
        all_keys.extend(key_values)
        input_replicates.append(
            {
                "event_keys": [key.to_mapping() for key in semantic_keys],
                "replicate_id": replicate_id,
            }
        )
        baseline: list[int] = []
        candidate: list[int] = []
        branch_draws: tuple[int, ...]

        if mode is RNGCouplingMode.EVENT_KEYED:
            rng = HashBasedRNG(spec.seed)
            branch_draws = ()
            for step, key in enumerate(semantic_keys):
                baseline.append(
                    rng.integer(key, -1_000, 1_000, domain="synthetic_exogenous")
                )
                candidate.append(
                    rng.integer(key, -1_000, 1_000, domain="synthetic_exogenous")
                )
                if step == spec.branch_step:
                    branch_draws = _event_keyed_branch_draws(
                        spec, replicate_id=replicate_id, rng=rng
                    )
        else:
            if mode is RNGCouplingMode.INDEPENDENT:
                baseline_seed = f"{spec.seed}:baseline"
                candidate_seed = f"{spec.seed}:candidate"
            else:
                baseline_seed = candidate_seed = f"{spec.seed}:shared"
            candidate_cursor = 0
            mutable_branch_draws: list[int] = []
            for step in range(spec.step_count):
                baseline.append(
                    _sequential_draw(
                        spec,
                        seed=baseline_seed,
                        replicate_id=replicate_id,
                        cursor=step,
                    )
                )
                candidate.append(
                    _sequential_draw(
                        spec,
                        seed=candidate_seed,
                        replicate_id=replicate_id,
                        cursor=candidate_cursor,
                    )
                )
                candidate_cursor += 1
                if step == spec.branch_step:
                    for _ in range(spec.branch_draw_count):
                        mutable_branch_draws.append(
                            _sequential_draw(
                                spec,
                                seed=candidate_seed,
                                replicate_id=replicate_id,
                                cursor=candidate_cursor,
                            )
                        )
                        candidate_cursor += 1
            branch_draws = tuple(mutable_branch_draws)

        baseline_draws.extend(baseline)
        candidate_draws.extend(candidate)
        differences.append(spec.policy_effect + sum(candidate) - sum(baseline))
        branch_hashes.append(sha256_bytes(canonical_json_bytes(list(branch_draws))))

    input_mapping = {
        "mode": mode.value,
        "replicates": input_replicates,
        "spec": spec.to_mapping(),
    }
    output_mapping = {
        "baseline_draws": baseline_draws,
        "candidate_draws": candidate_draws,
        "differences": differences,
        "internal_branch_draws_sha256": branch_hashes,
    }
    return MechanismReceipt(
        mode=mode,
        config_sha256=spec.sha256,
        executor_code_sha256=_executor_code_sha256(components),
        code_components=components,
        actual_switches=_switches(mode),
        input_sha256=sha256_bytes(canonical_json_bytes(input_mapping)),
        event_keys_sha256=sha256_bytes(canonical_json_bytes(all_keys)),
        output_sha256=sha256_bytes(canonical_json_bytes(output_mapping)),
        event_keys=tuple(all_keys),
        baseline_draws=tuple(baseline_draws),
        candidate_draws=tuple(candidate_draws),
        differences=tuple(differences),
        internal_branch_draws_sha256=tuple(branch_hashes),
    )


@dataclass(frozen=True, slots=True)
class MechanismSmokeArtifact:
    """A three-row immutable synthetic mechanism artifact."""

    spec: MechanismSmokeSpec
    config_sha256: str
    receipts: tuple[MechanismReceipt, ...]
    claim_scope: str = MECHANISM_CLAIM_SCOPE
    synthetic_only: Literal[True] = True
    schema_version: str = MECHANISM_SMOKE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.config_sha256 != self.spec.sha256:
            _fail("artifact config_sha256 does not match spec")
        if not isinstance(self.receipts, tuple) or len(self.receipts) != len(
            RNGCouplingMode
        ):
            _fail("artifact must contain exactly three RNG mechanism receipts")
        ordered = tuple(sorted(self.receipts, key=lambda item: item.mode.value))
        if {receipt.mode for receipt in ordered} != set(RNGCouplingMode):
            _fail("artifact must execute each RNG coupling mode exactly once")
        object.__setattr__(self, "receipts", ordered)
        for receipt in ordered:
            if receipt.config_sha256 != self.config_sha256:
                _fail("receipt config hash does not match artifact")
            expected_draw_count = self.spec.replicate_count * self.spec.step_count
            if len(receipt.event_keys) != expected_draw_count:
                _fail("receipt draw count does not match spec")
            if len(receipt.differences) != self.spec.replicate_count:
                _fail("receipt replicate count does not match spec")
            if receipt.mode is RNGCouplingMode.EVENT_KEYED:
                if receipt.baseline_draws != receipt.candidate_draws:
                    _fail("event-keyed worlds must share every exogenous realization")
                if any(
                    difference != self.spec.policy_effect
                    for difference in receipt.differences
                ):
                    _fail("event-keyed paired differences must equal policy effect")
            if receipt.mode is RNGCouplingMode.SAME_SEQUENTIAL:
                for replicate_id in range(self.spec.replicate_count):
                    start = replicate_id * self.spec.step_count
                    split = start + self.spec.branch_step + 1
                    end = start + self.spec.step_count
                    if (
                        receipt.baseline_draws[start:split]
                        != receipt.candidate_draws[start:split]
                    ):
                        _fail("same-sequential draws diverged before the branch")
                    if (
                        receipt.baseline_draws[split:end]
                        == receipt.candidate_draws[split:end]
                    ):
                        _fail("same-sequential branch did not shift later draws")
        if self.claim_scope != MECHANISM_CLAIM_SCOPE:
            _fail("artifact claim scope cannot be widened")
        if self.synthetic_only is not True:
            _fail("mechanism artifact must be synthetic_only")
        if self.schema_version != MECHANISM_SMOKE_SCHEMA_VERSION:
            _fail("unsupported mechanism smoke schema")

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def receipt(self, mode: RNGCouplingMode) -> MechanismReceipt:
        for receipt in self.receipts:
            if receipt.mode is mode:
                return receipt
        raise KeyError(mode)

    def to_mapping(self) -> dict[str, object]:
        return {
            "claim_scope": self.claim_scope,
            "config_sha256": self.config_sha256,
            "receipts": [receipt.to_mapping() for receipt in self.receipts],
            "schema_version": self.schema_version,
            "spec": self.spec.to_mapping(),
            "synthetic_only": self.synthetic_only,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(
            value,
            frozenset(
                {
                    "claim_scope",
                    "config_sha256",
                    "receipts",
                    "schema_version",
                    "spec",
                    "synthetic_only",
                }
            ),
            "mechanism smoke artifact",
        )
        if value["synthetic_only"] is not True:
            _fail("mechanism artifact must be synthetic_only")
        claim_scope = value["claim_scope"]
        schema_version = value["schema_version"]
        if not isinstance(claim_scope, str) or not isinstance(schema_version, str):
            _fail("artifact claim scope and schema version must be strings")
        return cls(
            spec=MechanismSmokeSpec.from_mapping(_mapping(value["spec"], "spec")),
            config_sha256=_sha256(value["config_sha256"], "config_sha256"),
            receipts=tuple(
                MechanismReceipt.from_mapping(_mapping(item, "receipt"))
                for item in _array(value["receipts"], "receipts")
            ),
            claim_scope=claim_scope,
            synthetic_only=True,
            schema_version=schema_version,
        )


def run_mechanism_smoke(
    spec: MechanismSmokeSpec | None = None,
) -> MechanismSmokeArtifact:
    """Execute the three small synthetic RNG coupling mechanisms."""
    if spec is None:
        spec = MechanismSmokeSpec()
    if not isinstance(spec, MechanismSmokeSpec):
        _fail("spec must be a MechanismSmokeSpec")
    components = _source_components()
    return MechanismSmokeArtifact(
        spec=spec,
        config_sha256=spec.sha256,
        receipts=tuple(
            _execute_mode(spec, mode, components) for mode in RNGCouplingMode
        ),
    )


def _seal_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.sha256")


def write_mechanism_smoke_artifact(
    path: Path, artifact: MechanismSmokeArtifact
) -> None:
    """Write one artifact and its semantic SHA-256 seal."""
    if not isinstance(path, Path):
        _fail("path must be a pathlib.Path")
    if not isinstance(artifact, MechanismSmokeArtifact):
        _fail("artifact must be a MechanismSmokeArtifact")
    write_json_file(path, artifact.to_mapping())
    write_text_file(_seal_path(path), f"{artifact.sha256}\n")


@dataclass(frozen=True, slots=True)
class MechanismAuditEntry:
    mode: RNGCouplingMode
    receipt_sha256: str
    mechanism_execution: Literal[True] = True

    def __post_init__(self) -> None:
        if not isinstance(self.mode, RNGCouplingMode):
            _fail("audit mode must be an RNGCouplingMode")
        _sha256(self.receipt_sha256, "receipt_sha256")
        if self.mechanism_execution is not True:
            _fail("successful mechanism audit must set mechanism_execution=true")


@dataclass(frozen=True, slots=True)
class MechanismSmokeAudit:
    artifact_sha256: str
    config_sha256: str
    entries: tuple[MechanismAuditEntry, ...]
    claim_scope: str = MECHANISM_CLAIM_SCOPE
    synthetic_only: Literal[True] = True

    def __post_init__(self) -> None:
        _sha256(self.artifact_sha256, "artifact_sha256")
        _sha256(self.config_sha256, "config_sha256")
        if len(self.entries) != len(RNGCouplingMode):
            _fail("audit must contain all three mechanism entries")
        if self.claim_scope != MECHANISM_CLAIM_SCOPE or self.synthetic_only is not True:
            _fail("audit scope must remain synthetic RNG only")

    @property
    def mechanism_execution(self) -> Literal[True]:
        """Return true only because every receipt was source-checked and replayed."""
        return True


def audit_mechanism_smoke_artifact(
    path: Path,
    expected_spec: MechanismSmokeSpec,
    *,
    expected_artifact_sha256: str | None = None,
) -> MechanismSmokeAudit:
    """Re-read, source-check, and fully replay a mechanism artifact."""
    if not isinstance(path, Path):
        raise MechanismSmokeIntegrityError("path must be a pathlib.Path")
    try:
        artifact = MechanismSmokeArtifact.from_mapping(read_json_object(path))
    except (OSError, ValueError) as error:
        raise MechanismSmokeIntegrityError("mechanism artifact is invalid") from error
    if artifact.spec != expected_spec:
        raise MechanismSmokeIntegrityError(
            "artifact spec does not match the authoritative expected spec"
        )
    try:
        seal = _seal_path(path).read_text(encoding="ascii")
    except (OSError, UnicodeError) as error:
        raise MechanismSmokeIntegrityError(
            "cannot read mechanism artifact seal"
        ) from error
    if seal != f"{artifact.sha256}\n":
        raise MechanismSmokeIntegrityError("mechanism artifact seal does not match")
    if expected_artifact_sha256 is not None:
        _sha256(expected_artifact_sha256, "expected_artifact_sha256")
        if artifact.sha256 != expected_artifact_sha256:
            raise MechanismSmokeIntegrityError(
                "mechanism artifact does not match the external commitment"
            )
    current_components = _source_components()
    current_code_sha256 = _executor_code_sha256(current_components)
    if any(
        receipt.code_components != current_components
        or receipt.executor_code_sha256 != current_code_sha256
        for receipt in artifact.receipts
    ):
        raise MechanismSmokeIntegrityError(
            "mechanism receipt executor code does not match current source"
        )
    expected = run_mechanism_smoke(expected_spec)
    if artifact.to_mapping() != expected.to_mapping():
        raise MechanismSmokeIntegrityError(
            "mechanism artifact does not match deterministic replay"
        )
    return MechanismSmokeAudit(
        artifact_sha256=artifact.sha256,
        config_sha256=artifact.config_sha256,
        entries=tuple(
            MechanismAuditEntry(mode=receipt.mode, receipt_sha256=receipt.sha256)
            for receipt in artifact.receipts
        ),
    )


__all__ = [
    "MECHANISM_CLAIM_SCOPE",
    "MECHANISM_EXECUTOR_ID",
    "MECHANISM_RECEIPT_SCHEMA_VERSION",
    "MECHANISM_SMOKE_SCHEMA_VERSION",
    "DrawAddressing",
    "MechanismAuditEntry",
    "MechanismReceipt",
    "MechanismSmokeArtifact",
    "MechanismSmokeAudit",
    "MechanismSmokeError",
    "MechanismSmokeIntegrityError",
    "MechanismSmokeSchemaError",
    "MechanismSmokeSpec",
    "MechanismSwitches",
    "RNGCouplingMode",
    "SeedCoupling",
    "audit_mechanism_smoke_artifact",
    "run_mechanism_smoke",
    "write_mechanism_smoke_artifact",
]
