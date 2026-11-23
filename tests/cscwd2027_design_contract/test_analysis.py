"""Manual release/abstention accounting fixtures, not study observations."""

from __future__ import annotations

from typing import Any

from scripts.cscwd2027_design_contract.analyze import aggregate


def row(*, local: bool, written: bool, valid: bool) -> dict[str, Any]:
    return {
        "case_id": "fixture",
        "nominal_valid_or_recoverable": True,
        "all_observed_local_valid": local,
        "reference_available": local,
        "decision": {"numerical_file_exists": written},
        "output_panel_valid": valid,
        "consequences": {
            "top1_disagreement": False if written else None,
            "top2_disagreement": False if written else None,
            "optimal_set_error": False if written else None,
            "zero_regret_disagreement": False if written else None,
            "assessable": written,
            "regret": [0, 1] if written else None,
        },
    }


def test_recovered_labels_are_not_invalid_releases() -> None:
    result = aggregate([row(local=True, written=True, valid=True)])
    assert result["valid_control_release"] == 1
    assert result["unauthorized_release"] == 0


def test_native_failure_is_not_unnecessary_abstention() -> None:
    result = aggregate([row(local=False, written=False, valid=False)])
    assert result["unavailable_controls"] == 1
    assert result["unnecessary_abstention"] == 0
    assert result["regret_mean"] is None
