"""Second-review publication data must preserve universes and raw failures."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper/cscwd2027_design_contract"
RESULT = ROOT / "results/cscwd2027/review_consequences_v2"


def read(path):
    return json.loads(path.read_text())


def test_conditional_counts_preserve_original_denominators():
    original = read(ROOT / "results/cscwd2027/design_contract_v1/derived/summary.json")
    publication = read(RESULT / "publication_summary.json")
    for mode, row in publication["artifact_analysis"]["original_posthoc"].items():
        old = original["mode_statistics"][mode]
        assert row["all_report_flips"] == old["top1_disagreement"]
        assert row["all_assessable_reports"] == old["assessable_decisions"]
        assert row["unauthorized_reports"] == old["unauthorized_release"]
    assert (RESULT / "artifacts/summary.json").stat().st_size == 0
    receipt = publication["recovery_receipt"]
    assert receipt["new_simulation_pairs"] == 0
    assert receipt["comparison"] == "exact canonical serialized bytes"
    assert len(receipt["verified_original_outputs"]) > 0


def test_mixed_universe_and_negative_permission_result_are_retained():
    data = read(RESULT / "publication_summary.json")["artifact_analysis"]
    assert sum(data["operation_count_distribution"].values()) == data["mixed_panels"]
    assert set(data["operation_count_distribution"]) == {"0", "1", "2", "3"}
    for s in data["mixed"].values():
        assert s["panels"] == data["mixed_panels"]
    assert data["mixed"]["without_permission"]["unauthorized_reports"] > 0
    assert data["mixed"]["without_permission"]["conditional_flips"] == 0
    assert all(r["exact"] for r in data["localization"])


def test_workflow_counts_and_shared_reference_are_explicit():
    data = read(RESULT / "publication_summary.json")["workflow"]
    config = read(ROOT / "configs/cscwd2027/review_consequences_v2.json")
    assert data["new_pairs"] == config["workflow"]["maximum_new_pairs"]
    assert data["statuses"] == {"completed": data["new_pairs"]}
    assert data["locally_rejected"] == 0
    assert data["scientific_trace_matches"] == data["slot_comparisons"]
    assert data["draw_matches"] == data["slot_comparisons"]
    for row in data["rows"]:
        assert row["replicate_extension"]["shared_new_slot_reference"] is True
        assert row["add_candidate"]["new_execution_shared_with_timing"] is True
    supplement = (PAPER / "sections/review_consequences.tex").read_text()
    assert "not independent reexecution" in supplement


def test_new_manifest_and_display_merge_are_faithful():
    manifest = read(PAPER / "generated/figure_manifest_review_consequences.json")
    for group in ("inputs", "source", "outputs"):
        for path, expected in manifest[group].items():
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected
    visual = read(PAPER / "generated/review_consequence_plot_data.json")
    assert len(visual["matrix"]) == 6
    assert (
        visual["original"]["spec_grouped"]
        == visual["original"]["minus_scientific_binding"]
    )
    assert visual["original_denominators_preserved"] is True
