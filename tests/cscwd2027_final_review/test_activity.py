"""Hand-verifiable descriptive-audit fixtures; no empirical observations."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from scripts.cscwd2027_final_review.activity import pair_activity, world_activity


def world() -> dict[str, Any]:
    """Two epochs, only the first containing a filled command."""
    return {
        "epochs": [
            {
                "applied_commands": [{"event_key": {"value": "a"}}],
                "results": [{"status": "active"}, {"status": "inactive"}],
                "state_after": {"state": 1},
            },
            {
                "applied_commands": [{"event_key": {"value": "b"}}],
                "results": [{"status": "active"}],
                "state_after": {"state": 2},
            },
        ],
        "trades": [{"cause_event_key": "a"}, {"cause_event_key": "a"}],
        "final_state": {"accounts": [{"trader_id": "x", "asset": 1, "cash": 2}]},
    }


def test_fill_epochs_count_causes_not_fills() -> None:
    """Two fills in one epoch count as one filled epoch, not two."""
    assert world_activity(world()) == {
        "world_epochs": 2,
        "fill_bearing_epochs": 1,
        "active_tasks": 2,
        "inactive_tasks": 1,
        "fills": 2,
    }


def test_equal_worlds_and_zero_contrast() -> None:
    """Exact equality is kept separate from presence of activity."""
    baseline = world()
    result = pair_activity(
        {
            "worlds": {"baseline": baseline, "policy": deepcopy(baseline)},
            "metrics": {"paired_traded_notional": 0},
        }
    )
    assert result["fill_bearing_epochs"] == 2
    assert result["different_world_traces"] == 0
    assert result["nonzero_paired_contrasts"] == 0


def test_trace_difference_can_have_zero_contrast() -> None:
    """A changed state with zero aggregate contrast is not discarded."""
    baseline, policy = world(), world()
    policy["epochs"][1]["state_after"] = {"state": 3}
    policy["final_state"]["accounts"][0]["cash"] = 1
    result = pair_activity(
        {
            "worlds": {"baseline": baseline, "policy": policy},
            "metrics": {"paired_traded_notional": 0},
        }
    )
    assert result["different_world_traces"] == 1
    assert result["different_terminal_balances"] == 1
    assert result["nonzero_paired_contrasts"] == 0
