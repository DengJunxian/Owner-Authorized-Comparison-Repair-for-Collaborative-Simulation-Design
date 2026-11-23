"""Manual-checkable fixtures for prefrozen grids, ground truth and statistics."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from pathlib import Path

import pytest

from policytwin.coordination.faults import Fault
from policytwin.coordination.plan import RunSpec
from scripts.cscwd2027 import run_studies
from scripts.cscwd2027.derive_results import bootstrap_mean, quantile, rate, summarize
from scripts.cscwd2027.study_design import entry, schedule, seed


def test_schedule_has_fixed_counts_and_unique_attempts() -> None:
    rows = schedule()
    assert len(rows) == 1874
    assert len({r["id"] for r in rows}) == len(rows)
    assert Counter(r["group"] for r in rows) == {
        "e1_reference": 192,
        "e1": 576,
        "e2_reference": 32,
        "e2": 256,
        "e2b_reference": 32,
        "e2b": 96,
        "e3": 690,
    }
    assert sum(r["group"] == "e3" and r["warmup"] for r in rows) == 90
    seen = set()
    for row in rows:
        if row["reference_id"]:
            assert row["reference_id"] in seen
        seen.add(row["id"])


def test_seed_pairing_and_mode_order() -> None:
    assert seed("e1-v1", 6, "cancellation", 0) == seed("e1-v1", 6, "cancellation", 0)
    assert seed("e1-v1", 6, "cancellation", 0) != seed("e2-v1", 6, "cancellation", 0)
    rows = [
        r
        for r in schedule()
        if r["group"] == "e3"
        and not r["warmup"]
        and r["spec"]["agent_count"] == 6
        and r["plan"]["worker_count"] == 1
    ]
    orders = []
    for block in range(20):
        cell = [r for r in rows if r["block"] == block]
        assert len({r["spec"]["seed"] for r in cell}) == 1
        assert len(cell) == 5
        orders.append(tuple(r["cost_mode"] for r in cell))
    assert len(set(orders)) > 1


def test_quantiles_and_paired_bootstrap_fixture() -> None:
    assert quantile([1.0, 2.0, 3.0, 4.0], 0.25) == 1.75
    assert quantile([2.0], 0.975) == 2.0
    assert bootstrap_mean([2.0] * 20) == {"n": 20, "mean": 2.0, "ci95": [2.0, 2.0]}
    assert bootstrap_mean([])["n"] == 0


def test_failed_attempt_is_not_removed_from_denominator() -> None:
    rows = [{"decisions": {"full": {"numerical_file_exists": True}}}, {}]
    assert rate(rows, "full", "numerical_file_exists") == {"count": 1, "denominator": 2}
    summary = summarize([], schedule())
    assert not summary["data_complete"]
    assert len(summary["missing_ids"]) == 1874


def test_runner_reference_fault_and_actual_release_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(run_studies, "RESULTS", tmp_path)
    spec = RunSpec(
        "pipeline-unit-only", "pipeline-unit-only", 0, 6, "cancellation", 4, 1
    )
    reference = entry("unit-reference", "e2_reference", spec, 2, processes=False)
    assert run_studies.execute(reference)["status"] == "completed"
    row = entry(
        "unit-fault",
        "e2",
        spec,
        2,
        processes=False,
        reference_id="unit-reference",
        fault=asdict(Fault("payload_mutation", "policy", 1, spec.agents[1])),
    )
    result = run_studies.execute(row)
    assert result["status"] == "completed", result.get("error")
    assert result["injection_triggered"]
    assert result["decisions"]["minus_replay"]["numerical_file_exists"]
    assert not result["decisions"]["full"]["numerical_file_exists"]
    assert (tmp_path / "unit-fault/injection_witness.json").is_file()
