"""Post-run consistency checks; not additional scientific observations."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/cscwd2027/consent_projection_v1"
CONFIG = ROOT / "configs/cscwd2027/consent_projection_v1"
PAPER = ROOT / "paper/cscwd2027_design_contract"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_presealed_schedule_and_full_execution_accounting():
    seal = read(CONFIG / "freeze.json")
    started = read(RESULT / "run_started.json")
    ended = read(RESULT / "run_completed.json")
    assert started["freeze_sha256"] == sha(CONFIG / "freeze.json")
    assert seal["created_at_utc"] < started["started_at_utc"]
    assert started["started_at_utc"] < ended["completed_at_utc"]
    assert not ended["failures"]
    assert len(ended["sessions"]) == seal["scientific_blocks"]
    summary = read(RESULT / "summary.json")
    assert summary["planning_requests"] == seal["planning_requests"]
    assert summary["physical_requests"] == seal["physical_requests"]
    assert summary["method_panels"] == seal["physical_method_panels"]
    assert not summary["chronology_violations"]
    methods = read(CONFIG / "study.json")["methods"]
    assert summary["actual_paired_attempts"] == (
        summary["initial_pairs"]
        + summary["reference_pairs"]
        + sum(summary["methods"][m]["new_pairs"] for m in methods)
    )
    assert summary["actual_paired_attempts"] == len(
        list((RESULT / "objects").glob("*/result.json"))
    )


def test_exact_selection_parity_and_no_outcome_replacement():
    summary = read(RESULT / "summary.json")
    for method in ("certificates", "policy_sql", "full_key_memo"):
        assert summary["methods"][method]["selection_disagreements"] == 0
        assert summary["methods"][method]["false_abstentions"] == 0
    schedule = read(CONFIG / "schedule.json")
    expected = {
        (b["session"]["id"], t["request"]["kind"])
        for b in schedule
        for c in b["contexts"]
        for t in c["tickets"]
        if t["physical"]
    }
    planned = [read(p) for p in (RESULT / "planning").glob("*.json")]
    observed = {
        (e["request"]["session"]["id"], e["request"]["kind"])
        for e in planned
        if e["physical"]
    }
    assert observed == expected
    assert all(e["physical"] == (e["index"] == 0) for e in planned)


def test_call_counts_keep_selection_and_authorization_separate():
    rows = read(RESULT / "analysis/planning_rows.json")
    by_case = {(r["session"], r["request"], r["method"]): r for r in rows}
    for row in rows:
        assert row["selection_check_calls"] == (3 if row["selected"] else 0)
        if row["method"] == "policy_sql":
            assert row["search_predicate_calls"] is None
        if row["method"] == "certificates":
            full = by_case[row["session"], row["request"], "exhaustive"]
            assert row["search_predicate_calls"] <= full["search_predicate_calls"]
        if row["method"] == "full_key_memo":
            assert row["skipped_candidates"] == 0
    for row in read(RESULT / "analysis/physical_rows.json"):
        assert row["authorization_predicate_calls"] == (
            0 if row["status"] == "no_agreement" else 9
        )


def test_current_scientific_seal_and_raw_manifest_are_unchanged():
    for name, expected in read(CONFIG / "freeze.json")["members"].items():
        assert sha(ROOT / name) == expected, name
    for name, expected in read(RESULT / "manifest.json")["members"].items():
        assert sha(RESULT / name) == expected, name


def test_generated_presentation_matches_structured_evidence():
    manifest = read(PAPER / "generated/consent_projection_table_manifest.json")
    for group in ("inputs", "source", "outputs"):
        for name, expected in manifest[group].items():
            assert sha(ROOT / name) == expected, name
    numbers = dict(
        re.findall(
            r"\\newcommand\{\\([A-Za-z]+)\}\{([^{}]+)\}",
            (PAPER / "generated/consent_projection_numbers.tex").read_text(),
        )
    )
    assert numbers == {k: str(v) for k, v in manifest["numbers"].items()}
    summary = read(RESULT / "summary.json")
    assert int(numbers["ProjectionCases"]) == summary["planning_requests"]
    assert int(numbers["ProjectionPairs"]) == summary["actual_paired_attempts"]
    assert (
        "consent_projection_results"
        in (PAPER / "sections/negotiated_evaluation.tex").read_text()
    )
    assert (
        "haselbock1993exploiting"
        in (PAPER / "sections/consent_projection_method.tex").read_text()
    )


def test_original_registered_metadata_remains_locked():
    from scripts.check_registered_metadata import check_registered_metadata

    check_registered_metadata(
        ROOT / "paper/main_revised.tex", ROOT / "paper/registered_metadata_lock.json"
    )


def test_current_draft_metadata_resolves_generated_percent():
    import importlib.util

    path = ROOT / "ccfa-workfiles/writing/paired-comparison/update_metadata.py"
    spec = importlib.util.spec_from_file_location("draft_metadata", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    metadata = module.current_metadata(ROOT)
    expected = read(PAPER / "generated/consent_projection_table_manifest.json")[
        "numbers"
    ]["ProjectionCallReduction"]
    assert f"{expected}%" in metadata["abstract"]
    assert "\\" not in metadata["abstract"]
    assert metadata == read(PAPER / "cscwd_draft_metadata.json")
