"""Post-run accounting and manuscript bindings, separate from sealed tests."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/cscwd2027/owner_policy_v1"
PAPER = ROOT / "paper/cscwd2027_design_contract"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_execution_started_after_seal_and_completed_without_replacement():
    config = ROOT / "configs/cscwd2027/owner_policy_v1"
    freeze = read(config / "freeze.json")
    start = read(RESULT / "run_started.json")
    complete = read(RESULT / "run_completed.json")
    assert start["freeze_sha256"] == sha(config / "freeze.json")
    assert freeze["created_at_utc"] < start["started_at_utc"]
    assert start["started_at_utc"] < complete["completed_at_utc"]
    assert len(complete["sessions"]) == freeze["sessions"]
    assert complete["failures"] == []
    assert read(config / "preseal_tests.json")["tests"] == 10


def test_actual_acquisitions_and_exact_reports_partition():
    s = read(RESULT / "summary.json")
    assert s["actual_paired_attempts"] == len(
        list((RESULT / "objects").glob("*/result.json"))
    )
    assert s["actual_paired_attempts"] == (
        s["initial_pairs"]
        + s["reference_pairs"]
        + sum(m["new_pairs"] for m in s["methods"].values())
    )
    assert not s["chronology_violations"]
    assert not s["failed_or_rejected_attempts"]
    for m in s["methods"].values():
        assert m["requests"] == s["tickets"]
        assert m["released"] == m["exact_reports"] == m["assessable_reports"]
        assert m["matching_slots"] == m["compared_slots"]
        assert m["execution_abstentions"] == 0
        assert m["released"] + m["no_agreement"] == m["requests"]


def test_equal_obligation_sql_control_matches_targets_and_work():
    s = read(RESULT / "summary.json")
    rows = read(RESULT / "analysis/rows.json")
    assert s["sql_planner_matches"] == s["tickets"]
    for full in (r for r in rows if r["method"] == "owner_search"):
        sql = next(
            r
            for r in rows
            if r["method"] == "policy_csp"
            and (r["session"], r["case"]) == (full["session"], full["case"])
        )
        for key in ("selected", "authorized", "exact", "new_pairs", "reevaluations"):
            assert full[key] == sql[key]


def test_all_controls_and_combined_concessions_are_retained():
    rows = read(RESULT / "analysis/rows.json")
    assert len(rows) == read(RESULT / "summary.json")["method_panels"]
    for row in rows:
        if row["case"] == "empty_agreement":
            assert row["selected"] is None and not row["authorized"]
            assert not row["false_abstention"]
        if row["case"] == "permissive_control":
            assert row["authorized"] and row["exact"]
        if row["case"] == "multi_concession":
            if row["method"] in {"owner_search", "policy_csp"}:
                assert row["authorized"] and row["waivers_count"] == 3
            else:
                assert row["false_abstention"] and not row["authorized"]


def test_vote_ablations_retain_the_same_numerical_evidence():
    s = read(RESULT / "summary.json")
    for m in s["methods"].values():
        assert m["policy_mutations_rejected"] == m["policy_mutations"]
        assert m["omit_policy_invalid_release"] == m["policy_mutations"]


def test_immutable_result_and_presentation_manifests():
    for name, expected in read(RESULT / "manifest.json")["members"].items():
        assert sha(RESULT / name) == expected, name
    manifest = read(PAPER / "generated/owner_policy_table_manifest.json")
    for group in ("inputs", "outputs"):
        for name, expected in manifest[group].items():
            assert sha(ROOT / name) == expected, name
    figure = read(PAPER / "generated/figure_manifest_owner_policy_search.json")
    assert figure["brief_user_confirmed"]
    assert figure["lint_errors"] == figure["lint_warnings"] == 0
    for group in ("inputs", "source", "outputs"):
        for name, expected in figure[group].items():
            assert sha(ROOT / name) == expected, name


def test_published_denominators_and_historical_posthoc_label():
    macros = dict(
        re.findall(
            r"\\newcommand\{\\([A-Za-z]+)\}\{([^{}]+)\}",
            (PAPER / "generated/owner_policy_numbers.tex").read_text(),
        )
    )
    s = read(RESULT / "summary.json")
    assert int(macros["OwnerTickets"]) == s["tickets"]
    assert int(macros["OwnerPairs"]) == s["actual_paired_attempts"]
    assert int(macros["OwnerReleased"]) == s["methods"]["owner_search"]["released"]
    assert int(macros["ConcessionAll"]) == (
        int(macros["ConcessionAssessable"]) + int(macros["ConcessionAbstentions"])
    )
    text = (PAPER / "sections/owner_policy_results.tex").read_text()
    assert "post-hoc" in text
    assert r"\ConcessionTopOneSame" in text
    assert r"\OwnerSqlMatches" in text
