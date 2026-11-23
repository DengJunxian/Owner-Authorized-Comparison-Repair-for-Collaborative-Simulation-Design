"""Additional verification fixtures, not changes to the frozen research grid."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from policytwin.coordination.coordinator import run_pair
from policytwin.coordination.plan import CoordinationExecutionPlan, RunSpec
from policytwin.coordination.release_gate import register, seal
from policytwin.coordination_validation.gate import witnessed_report
from policytwin.coordination_validation.witnesses import specification_witnesses
from policytwin.counterfactual.coupling_contract import EventKeyedBackend
from tests.cscwd2027_revision2.test_specification_witnesses import manual_fixture


@pytest.mark.parametrize("field", ("cash", "asset"))
def test_balanced_misattribution_is_not_hidden_by_total_conservation(
    field: str,
) -> None:
    capsule = manual_fixture()
    accounts = capsule["worlds"]["baseline"]["final_state"]["accounts"]
    original = sum(a[field] for a in accounts)
    accounts[0][field] += 1
    accounts[1][field] -= 1
    assert sum(a[field] for a in accounts) == original
    result = specification_witnesses(capsule)
    assert not result["pass"] and len(result["findings"]) == 2


def test_shared_sampler_remains_an_explicit_trusted_boundary(tmp_path: Path) -> None:
    spec = RunSpec(
        "r2-boundary-unit", "r2-boundary-unit-seed", 0, 6, "agent_exit", 3, 1
    )
    plan = CoordinationExecutionPlan(1)
    registration = register(spec, plan)
    with patch.object(EventKeyedBackend, "draw", return_value=0):
        capsule = seal(run_pair(spec, plan, processes=False), registration)
        result = witnessed_report(
            spec, plan, capsule, registration, tmp_path / "report"
        )
    assert result["numerical_file_exists"] and result["witnesses"]["pass"]
