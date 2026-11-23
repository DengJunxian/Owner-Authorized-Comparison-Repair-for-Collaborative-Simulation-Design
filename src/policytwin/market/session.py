"""Deterministic multi-step market sessions built from sealed CLOB replays.

The core :class:`CentralLimitOrderBook` deliberately seals one command batch.
Agent feedback, however, needs a trustworthy snapshot between decisions.  This
module obtains that snapshot without introducing a second matching engine: on
every accepted transition it replays the complete accumulated command prefix
from the frozen opening state.  The implementation is intentionally simple and
slow; correctness and exact replay are the acceptance criteria.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from policytwin.market.models import (
    MarketCommand,
    MarketConfig,
    MarketSnapshot,
    OpeningState,
    ReplayResult,
    command_to_mapping,
)
from policytwin.market.order_book import CentralLimitOrderBook
from policytwin.provenance import canonical_json_bytes, sha256_bytes

MARKET_SESSION_SCHEMA_VERSION = "policytwin.replay-market-session.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class MarketSessionError(RuntimeError):
    """Raised when a multi-step replay commitment cannot be verified."""


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise MarketSessionError(f"{field} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class MarketSessionTransition:
    """One committed command-prefix transition and its resulting market state."""

    sequence: int
    previous_replay_sha256: str
    command_sha256s: tuple[str, ...]
    command_count: int
    replay_sha256: str
    state_sha256: str
    transition_sha256: str
    schema_version: str = MARKET_SESSION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence < 0
        ):
            raise MarketSessionError("transition sequence must be non-negative")
        if (
            isinstance(self.command_count, bool)
            or not isinstance(self.command_count, int)
            or self.command_count <= 0
        ):
            raise MarketSessionError("transition command_count must be positive")
        if (
            not isinstance(self.command_sha256s, tuple)
            or len(self.command_sha256s) != self.command_count
        ):
            raise MarketSessionError("command commitments must match command_count")
        for index, value in enumerate(self.command_sha256s):
            _sha256(value, f"command_sha256s[{index}]")
        for field in (
            "previous_replay_sha256",
            "replay_sha256",
            "state_sha256",
            "transition_sha256",
        ):
            _sha256(getattr(self, field), field)
        if self.schema_version != MARKET_SESSION_SCHEMA_VERSION:
            raise MarketSessionError("unsupported market-session schema version")
        if sha256_bytes(canonical_json_bytes(self.core_mapping())) != (
            self.transition_sha256
        ):
            raise MarketSessionError("market-session transition hash mismatch")

    def core_mapping(self) -> dict[str, object]:
        """Return all transition fields covered by ``transition_sha256``."""
        return {
            "command_count": self.command_count,
            "command_sha256s": list(self.command_sha256s),
            "previous_replay_sha256": self.previous_replay_sha256,
            "replay_sha256": self.replay_sha256,
            "schema_version": self.schema_version,
            "sequence": self.sequence,
            "state_sha256": self.state_sha256,
        }

    def to_mapping(self) -> dict[str, object]:
        """Return the complete machine-readable transition."""
        return {**self.core_mapping(), "transition_sha256": self.transition_sha256}


class ReplayMarketSession:
    """Multi-step state facade whose only transition implementation is replay."""

    def __init__(self, config: MarketConfig, opening_state: OpeningState) -> None:
        if not isinstance(config, MarketConfig):
            raise MarketSessionError("config must be a MarketConfig")
        if not isinstance(opening_state, OpeningState):
            raise MarketSessionError("opening_state must be an OpeningState")
        self._config = config
        self._opening_state = opening_state
        self._commands: tuple[MarketCommand, ...] = ()
        self._transitions: tuple[MarketSessionTransition, ...] = ()
        self._replay = self._execute(self._commands)

    @property
    def config(self) -> MarketConfig:
        """Return the immutable market configuration."""
        return self._config

    @property
    def opening_state(self) -> OpeningState:
        """Return the frozen opening state."""
        return self._opening_state

    @property
    def commands(self) -> tuple[MarketCommand, ...]:
        """Return the exact accumulated command stream."""
        return self._commands

    @property
    def transitions(self) -> tuple[MarketSessionTransition, ...]:
        """Return the immutable transition ledger."""
        return self._transitions

    @property
    def replay(self) -> ReplayResult:
        """Return the latest complete sealed replay."""
        return self._replay

    @property
    def snapshot(self) -> MarketSnapshot:
        """Return the latest replay-verified market snapshot."""
        return self._replay.state

    @property
    def sha256(self) -> str:
        """Commit to config, opening state, commands, and every transition."""
        return sha256_bytes(
            canonical_json_bytes(
                {
                    "commands": [command_to_mapping(item) for item in self._commands],
                    "config_sha256": self._config.sha256,
                    "opening_state_sha256": self._replay.opening_state_sha256,
                    "replay_sha256": self._replay.sha256,
                    "schema_version": MARKET_SESSION_SCHEMA_VERSION,
                    "transitions": [item.to_mapping() for item in self._transitions],
                }
            )
        )

    def submit(self, command: MarketCommand) -> MarketSessionTransition:
        """Atomically append one command and materialize the next snapshot."""
        return self.submit_all((command,))

    def submit_all(
        self, commands: tuple[MarketCommand, ...]
    ) -> MarketSessionTransition:
        """Atomically append a non-empty immutable command batch.

        The session is updated only after the candidate prefix executes
        successfully.  A rejected command therefore cannot leave a partially
        advanced scientific state.
        """
        if not isinstance(commands, tuple) or not commands:
            raise MarketSessionError("commands must be a non-empty immutable tuple")
        command_hashes: list[str] = []
        for command in commands:
            try:
                mapping = command_to_mapping(command)
            except (AttributeError, TypeError) as error:
                raise MarketSessionError(
                    "commands contain an unsupported value"
                ) from error
            command_hashes.append(sha256_bytes(canonical_json_bytes(mapping)))
        candidate_commands = (*self._commands, *commands)
        previous = self._replay.sha256
        candidate_replay = self._execute(candidate_commands)
        prior_schedule = self._replay.scheduled_commands
        if candidate_replay.scheduled_commands[: len(prior_schedule)] != prior_schedule:
            raise MarketSessionError(
                "new commands would retroactively reorder the materialized market "
                "history"
            )
        core: dict[str, object] = {
            "command_count": len(commands),
            "command_sha256s": command_hashes,
            "previous_replay_sha256": previous,
            "replay_sha256": candidate_replay.sha256,
            "schema_version": MARKET_SESSION_SCHEMA_VERSION,
            "sequence": len(self._transitions),
            "state_sha256": candidate_replay.state.sha256,
        }
        transition = MarketSessionTransition(
            sequence=len(self._transitions),
            previous_replay_sha256=previous,
            command_sha256s=tuple(command_hashes),
            command_count=len(commands),
            replay_sha256=candidate_replay.sha256,
            state_sha256=candidate_replay.state.sha256,
            transition_sha256=sha256_bytes(canonical_json_bytes(core)),
        )
        self._commands = candidate_commands
        self._replay = candidate_replay
        self._transitions = (*self._transitions, transition)
        return transition

    def assert_replay(self) -> None:
        """Recompute every prefix and verify the complete transition ledger."""
        initial = self._execute(())
        prefix: tuple[MarketCommand, ...] = ()
        previous = initial.sha256
        prior_schedule = initial.scheduled_commands
        offset = 0
        for sequence, transition in enumerate(self._transitions):
            if transition.sequence != sequence:
                raise MarketSessionError("market-session transition order changed")
            batch = self._commands[offset : offset + transition.command_count]
            if len(batch) != transition.command_count:
                raise MarketSessionError("market-session transition exceeds commands")
            hashes = tuple(
                sha256_bytes(canonical_json_bytes(command_to_mapping(item)))
                for item in batch
            )
            prefix = (*prefix, *batch)
            replay = self._execute(prefix)
            if replay.scheduled_commands[: len(prior_schedule)] != prior_schedule:
                raise MarketSessionError(
                    "market-session transition retroactively reordered prior history"
                )
            if (
                transition.previous_replay_sha256 != previous
                or transition.command_sha256s != hashes
                or transition.replay_sha256 != replay.sha256
                or transition.state_sha256 != replay.state.sha256
            ):
                raise MarketSessionError("market-session replay ledger mismatch")
            previous = replay.sha256
            prior_schedule = replay.scheduled_commands
            offset += transition.command_count
        if offset != len(self._commands):
            raise MarketSessionError("market-session commands are not fully accounted")
        replay = self._execute(self._commands)
        if replay.canonical_bytes != self._replay.canonical_bytes:
            raise MarketSessionError("latest market replay differs from recomputation")

    def _execute(self, commands: tuple[MarketCommand, ...]) -> ReplayResult:
        engine = CentralLimitOrderBook(self._config, self._opening_state)
        if commands:
            engine.schedule_all(commands)
        result = engine.run()
        engine.assert_invariants()
        return result


__all__ = [
    "MARKET_SESSION_SCHEMA_VERSION",
    "MarketSessionError",
    "MarketSessionTransition",
    "ReplayMarketSession",
]
