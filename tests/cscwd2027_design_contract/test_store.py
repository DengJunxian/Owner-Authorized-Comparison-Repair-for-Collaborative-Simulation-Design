"""A complete per-run file/token engineering check outside the formal namespace."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

from policytwin.coordination.plan import CoordinationExecutionPlan, RunSpec
from policytwin.design_contract.runtime import SimulationRecipe
from scripts.cscwd2027_design_contract import store


def test_bound_recipe_run_and_actual_report_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(store, "RESULTS", tmp_path)
    recipe = SimulationRecipe(
        RunSpec("unit-store", "unit-store-not-formal", 0, 4, "cancellation", 2, 1), 24
    )
    row = {
        "id": "unit-store-0",
        "recipe": recipe.to_mapping(),
        "plan": asdict(CoordinationExecutionPlan(2)),
    }
    result = store.execute(row, "0" * 64)
    assert result["status"] == "completed" and result["local_pass"]
    token, capsule = store.load("unit-store-0")
    assert token.local_pass and token.scientific_key == recipe.scientific_key
    assert token.value == capsule["metrics"]["paired_traded_notional"]
    assert (
        tmp_path / "objects/unit-store-0/local_report/numerical_report.json"
    ).is_file()
