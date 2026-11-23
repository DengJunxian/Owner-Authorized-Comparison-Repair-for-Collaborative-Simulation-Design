"""Evidence and manuscript checks for the reviewer-requested revision."""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path

import pytest

from policytwin.coordination.plan import CoordinationExecutionPlan, RunSpec
from policytwin.coordination.schema import CoordinationError

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper/cscwd2027_design_contract"
RESULT = ROOT / "results/cscwd2027/review_response_v1/summary.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_separate_diagnostic_sources_are_still_sealed():
    seal = read(ROOT / "configs/cscwd2027/review_response_v1_freeze.json")
    for path, expected in seal["members"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected
    summary = read(RESULT)
    manifest = read(RESULT.with_name("manifest.json"))
    freeze_path = ROOT / "configs/cscwd2027/review_response_v1_freeze.json"
    assert (
        hashlib.sha256(freeze_path.read_bytes()).hexdigest()
        == manifest["freeze_sha256"]
    )
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == manifest["summary_sha256"]
    assert summary["new_simulation_pairs"] == 0
    assert summary["sessions"] == len(
        read(ROOT / "configs/cscwd2027/design_contract_v1/sessions.json")
    )


def test_permission_results_change_no_other_obligation():
    rows = read(RESULT)["decisions"]
    assert all(r["all_local_valid"] for r in rows)
    for row in rows:
        if row["case"] == "permitted_control":
            assert row["authorized"] and row["output_panel_valid"]
        elif row["mode"] == "without_permission":
            assert row["authorized"] and not row["output_panel_valid"]
        else:
            assert not row["authorized"]
            assert row["decision"]["findings"] == ["permitted_plan"]


def test_matrix_cells_and_flips_match_complete_original_counts():
    source = read(ROOT / "results/cscwd2027/design_contract_v1/derived/summary.json")
    visual = read(PAPER / "generated/review_figure_data.json")
    assert len(visual["matrix"]) == len(source["mode_statistics"])
    for row in visual["matrix"]:
        values = source["mode_statistics"][row["mode"]]
        assert row["top1_flips"] == values["top1_disagreement"]
        assert row["assessable_reports"] == values["assessable_decisions"]
        assert len(row["cells"]) == len(source["case_statistics"])
        for cell in row["cells"]:
            values = source["case_statistics"][cell["case"]][row["mode"]]
            assert cell["released"] == values["written"]
            assert cell["unauthorized"] == values["unauthorized_release"]
            assert cell["count"] == values["sets"]
            if cell["state"] == "invalid_release":
                assert cell["unauthorized"] == cell["count"]
            elif cell["state"] == "false_rejection":
                assert (
                    cell["released"] == 0
                    and values["evaluable_controls"] == cell["count"]
                )
            else:
                assert cell["unauthorized"] == 0


def test_gap_analysis_uses_every_original_reference_without_resampling():
    rows = read(RESULT)["gaps"]
    assert len(rows) == 2 * read(RESULT)["sessions"]
    for row in rows:
        path = (
            ROOT
            / "results/cscwd2027/design_contract_v1/evaluation/references"
            / row["session"]
            / f"v{row['version']}/numerical_comparison.json"
        )
        reference = read(path)
        means = sorted(
            (Fraction(*v) for v in reference["means"].values()), reverse=True
        )
        assert row["means"] == reference["means"]
        assert Fraction(*row["top_two_gap"]) == means[0] - means[1]


def test_new_presentation_manifest_binds_real_inputs_and_outputs():
    manifest = read(PAPER / "generated/figure_manifest_review_response.json")
    for group in ("inputs", "source", "outputs"):
        for path, expected in manifest[group].items():
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected


def test_schedule_schema_and_cross_candidate_random_identity_are_explicit():
    with pytest.raises(CoordinationError, match="unsupported logical schedule"):
        CoordinationExecutionPlan(2, schedule="different-logical-order")
    left = RunSpec("fixture", "seed", 0, 6, "cancellation")
    right = RunSpec("fixture", "seed", 0, 6, "agent_exit")
    assert left.run_id != right.run_id
    main = (PAPER / "sections/evaluation.tex").read_text()
    assert "not across candidates" in main


def test_manuscript_is_comparison_focused_and_separates_new_measurements():
    text = "\n".join(p.read_text() for p in (PAPER / "sections").glob("*.tex"))
    assert "\\PartitionEqual" not in text and "\\FaultAttempts" not in text
    assert "independently fresh old slots and shared newly" in text
    assert "post-hoc" in text
    assert "paired reuse/fresh time ratios" in text
    assert "An unauthorized report" in text
    assert "population-level" in text
    assert "F_a(V_a" in (PAPER / "technical_supplement.tex").read_text()
