"""Shared offline-only test fixtures."""

from __future__ import annotations

import json
import socket
from datetime import UTC, datetime
from pathlib import Path
from typing import NoReturn

import pytest

from policytwin.data.temporal_firewall import (
    CutoffPolicy,
    TemporalAuditLog,
    TemporalFirewall,
)


def _network_forbidden(*args: object, **kwargs: object) -> NoReturn:
    del args, kwargs
    raise AssertionError("tests must not access the network")


@pytest.fixture(autouse=True)
def deny_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail every test that attempts to create a network socket."""
    monkeypatch.setattr(socket, "socket", _network_forbidden)
    monkeypatch.setattr(socket, "create_connection", _network_forbidden)


@pytest.fixture
def cutoff_policy() -> CutoffPolicy:
    """Create a three-way synthetic temporal policy with an embargo."""
    return CutoffPolicy(
        schema_version=1,
        cutoff=datetime(2025, 1, 31, tzinfo=UTC),
        training_end=datetime(2025, 1, 10, tzinfo=UTC),
        validation_end=datetime(2025, 1, 20, tzinfo=UTC),
        policy_start=datetime(2025, 1, 21, tzinfo=UTC),
        allowed_license_tags=("synthetic-test",),
    )


@pytest.fixture
def temporal_firewall(tmp_path: Path, cutoff_policy: CutoffPolicy) -> TemporalFirewall:
    """Create a firewall with a real chained audit sink."""
    return TemporalFirewall(
        cutoff_policy, TemporalAuditLog(tmp_path / "temporal-audit.jsonl")
    )


@pytest.fixture
def smoke_config_path(tmp_path: Path) -> Path:
    """Create a complete synthetic config and zero-record manifest."""
    manifest = {
        "schema_version": 1,
        "manifest_id": "test-synthetic-manifest",
        "data_kind": "synthetic",
        "description": "Synthetic test manifest with no records.",
        "record_count": 0,
        "files": [],
    }
    config = {
        "schema_version": 1,
        "experiment_type": "synthetic_smoke",
        "experiment_id": "test-synthetic-smoke",
        "knowledge_cutoff": "2025-01-01T00:00:00+00:00",
        "data_manifest": "manifest.json",
        "replicates": 3,
        "horizon_steps": 5,
        "initial_value_ticks": 1000,
        "shock_radius_ticks": 2,
        "policy_shift_ticks": 1,
        "seed": "test-synthetic-seed",
    }
    manifest_path = tmp_path / "manifest.json"
    config_path = tmp_path / "config.json"
    manifest_path.write_text(
        f"{json.dumps(manifest, indent=2, sort_keys=True)}\n", encoding="utf-8"
    )
    config_path.write_text(
        f"{json.dumps(config, indent=2, sort_keys=True)}\n", encoding="utf-8"
    )
    return config_path
