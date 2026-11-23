"""Hand-calculated witnesses and shared-defect checks; separate from v1 tests."""

from __future__ import annotations

import ast
import copy
from pathlib import Path
from typing import Any

import pytest

from policytwin.coordination.coordinator import run_pair
from policytwin.coordination.plan import CoordinationExecutionPlan, RunSpec
from policytwin.coordination.release_gate import register, report, seal
from policytwin.coordination_validation import witnesses
from policytwin.coordination_validation.gate import witnessed_report
from scripts.cscwd2027_revision2.design import CONFIG, schedule
from scripts.cscwd2027_revision2.mutators import MUTANTS, common_mode


def manual_fixture() -> dict[str, Any]:
    """Signal 6+15+90+110+35+161=417 gives sell, quantity 4 for zero bits."""
    names = [f"agent-{i:03d}" for i in range(6)] + ["book-bid", "book-ask"]
    view = {
        "bid_quantity": 2,
        "ask_quantity": 3,
        "best_bid": 90,
        "best_ask": 110,
        "asset": 5,
        "cash": 2300,
    }
    row = {
        "epoch": 0,
        "requests": [{"task_id": "t0", "view": view}],
        "results": [
            {
                "task_id": "t0",
                "agent": "agent-000",
                "status": "active",
                "draws": [
                    {"address": {"draw_role": r}, "state": "consumed", "value": 0}
                    for r in ("side", "quantity")
                ],
                "commands": [
                    {"command_type": "market_order", "side": "sell", "quantity": 4}
                ],
            }
        ],
    }
    world = {
        "epochs": [row],
        "trades": [],
        "traded_notional": 0,
        "final_state": {
            "accounts": [
                {"trader_id": name, "cash": 100_000_000, "asset": 1_000_000}
                for name in names
            ]
        },
    }
    return {
        "spec": {"agent_count": 6, "branch": "agent_exit", "branch_epoch": 1},
        "worlds": {w: copy.deepcopy(world) for w in ("baseline", "policy")},
    }


def test_manual_action_and_empty_ledger() -> None:
    result = witnesses.specification_witnesses(manual_fixture())
    assert result["pass"] and result["decisions_checked"] == 2
    assert result["accounts_checked"] == 16


@pytest.mark.parametrize(
    "field,value,code",
    [
        ("quantity", 5, "quantity_congruence"),
        ("side", "buy", "side_congruence"),
    ],
)
def test_manual_action_counterexamples(field: str, value: object, code: str) -> None:
    capsule = manual_fixture()
    capsule["worlds"]["policy"]["epochs"][0]["results"][0]["commands"][0][field] = value
    result = witnesses.specification_witnesses(capsule)
    assert [f["code"] for f in result["findings"]] == [code]


@pytest.mark.parametrize("corrupt", (False, True))
def test_manual_cash_asset_transfer(corrupt: bool) -> None:
    capsule = manual_fixture()
    world = capsule["worlds"]["policy"]
    world["trades"] = [
        {
            "buyer_id": "agent-000",
            "seller_id": "book-ask",
            "maker_trader_id": "book-ask",
            "taker_trader_id": "agent-000",
            "price": 110,
            "quantity": 2,
            "maker_fee": 0,
            "taker_fee": 0,
        }
    ]
    world["traded_notional"] = 220
    accounts = {a["trader_id"]: a for a in world["final_state"]["accounts"]}
    accounts["agent-000"].update(cash=99_999_780, asset=1_000_002)
    accounts["book-ask"].update(cash=100_000_220, asset=999_998)
    if corrupt:
        accounts["agent-000"]["cash"] += 1
    result = witnesses.specification_witnesses(capsule)
    assert result["pass"] is not corrupt
    assert len(result["findings"]) == int(corrupt)


def test_witness_has_no_simulator_or_policy_import() -> None:
    tree = ast.parse(Path(witnesses.__file__).read_text())
    imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert imports == ["__future__", "typing"]


@pytest.mark.parametrize("mutant", (None, *MUTANTS))
def test_actual_report_under_shared_defect(tmp_path: Path, mutant: str | None) -> None:
    spec = RunSpec("r2-unit-common", "r2-unit-seed", 0, 6, "cancellation", 3, 1)
    plan = CoordinationExecutionPlan(1)
    registration = register(spec, plan)
    with common_mode(mutant) as counter:
        capsule = seal(run_pair(spec, plan, processes=False), registration)
        full = report(spec, plan, capsule, registration, tmp_path / "full")
        extended = witnessed_report(
            spec, plan, capsule, registration, tmp_path / "extended"
        )
    assert full["numerical_file_exists"]
    assert extended["numerical_file_exists"] is (mutant is None)
    assert (counter["calls"] > 0) is (mutant is not None)


def test_fixed_grid_and_matched_inputs() -> None:
    rows = schedule()
    assert len(rows) == 336 and len({r["id"] for r in rows}) == 336
    assert sum(r["study"] == "extension" for r in rows) == 216
    assert sum(r["study"] == "common_mode" for r in rows) == 120
    lookup = {r["id"]: r for r in rows}
    for row in rows:
        if row["reference_id"]:
            assert row["spec"] == lookup[row["reference_id"]]["spec"]
    assert CONFIG["threshold"] == "exact equality, no fitted tolerance"
