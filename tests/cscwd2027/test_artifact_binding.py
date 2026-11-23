"""Post-freeze packaging tests; never change the formal scientific design."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from policytwin.coordination.plan import RunSpec
from policytwin.coordination.release_gate import implementation_sha256
from scripts.cscwd2027 import run_studies
from scripts.cscwd2027.study_design import entry
from scripts.cscwd2027.verify_artifacts import verify_attempt


@pytest.mark.parametrize("mutation", (None, "report_value", "delete_report"))
def test_file_level_binding_rejects_report_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str | None
) -> None:
    monkeypatch.setattr(run_studies, "RESULTS", tmp_path)
    spec = RunSpec("binding-unit", "binding-unit", 0, 6, "cancellation", 4, 1)
    design = entry("binding-unit", "e1_reference", spec, 1, processes=False)
    result = run_studies.execute(design)
    assert result["status"] == "completed"
    folder = tmp_path / design["id"]
    report = folder / "reports/full/numerical_report.json"
    if mutation == "report_value":
        content = json.loads(report.read_text())
        content["metrics"]["paired_traded_notional"] += 1
        report.write_text(json.dumps(content))
    elif mutation == "delete_report":
        report.unlink()
    arguments = (
        folder,
        design,
        None,
        "2020-01-01T00:00:00+00:00",
        implementation_sha256(),
    )
    if mutation is None:
        assert verify_attempt(*arguments)["reports_checked"] == 1
    else:
        with pytest.raises(ValueError, match="report"):
            verify_attempt(*arguments)
