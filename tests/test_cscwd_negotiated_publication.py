"""Publication, accounting, authority and preservation checks for actual rows."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/cscwd2027/negotiated_repair_v1"
REPORT = ROOT / "reports/cscwd2027/negotiated_repair_v1"
PAPER = ROOT / "paper/cscwd2027_design_contract"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_new_seal_and_all_prior_scientific_inputs():
    from scripts.cscwd2027_negotiated_repair import verify

    seal = verify()
    assert read(RESULT / "run_started.json")["freeze_sha256"] == sha(
        ROOT / "configs/cscwd2027/negotiated_repair_v1/freeze.json"
    )
    assert seal["created_at_utc"] < read(RESULT / "run_started.json")["started_at_utc"]
    assert not read(RESULT / "run_completed.json")["failures"]


def test_actual_execution_partition_and_finite_controls():
    summary = read(RESULT / "summary.json")
    methods = summary["methods"]
    assert summary["actual_paired_attempts"] == summary["initial_pairs"] + summary[
        "reference_pairs"
    ] + sum(m["new_pairs"] for m in methods.values())
    assert summary["actual_paired_attempts"] == len(
        list((RESULT / "objects").glob("*/result.json"))
    )
    assert not summary["chronology_violations"]
    assert not summary["failed_or_rejected_attempts"]
    assert methods["frontier"]["released"] == methods["frontier"]["exact_reports"]
    assert (
        methods["frontier"]["matching_slots"] == methods["frontier"]["compared_slots"]
    )
    assert methods["frontier"] == methods["csp_dag"]
    assert methods["serial"]["concessions"] == methods["frontier"]["concessions"]
    assert all(m["execution_abstentions"] == 0 for m in methods.values())
    rows = read(RESULT / "analysis/rows.json")
    assert all(
        not r["authorized"] and r["new_pairs"] == 0
        for r in rows
        if r["kind"] == "nonnegotiable"
    )
    assert all(
        r["authorized"] and r["new_pairs"] == r["reevaluations"] == 0
        for r in rows
        if r["kind"] == "complete_control"
    )


def test_original_authority_and_new_artifact_manifests():
    manifest = read(RESULT / "manifest.json")
    for name, digest in manifest["members"].items():
        assert sha(RESULT / name) == digest, name
    figure = read(PAPER / "generated/figure_manifest_negotiated_repair.json")
    assert not figure["vector_exports_pending"]
    for group in ("inputs", "source", "outputs"):
        for name, digest in figure[group].items():
            assert sha(ROOT / name) == digest, name
    diagram = read(PAPER / "generated/figure_manifest_owner_counteroffer.json")
    assert diagram["brief_user_confirmed"]
    assert diagram["lint_errors"] == diagram["lint_warnings"] == 0
    for group in ("inputs", "source", "outputs"):
        for name, digest in diagram[group].items():
            assert sha(ROOT / name) == digest, name
    macros = dict(
        re.findall(
            r"\\newcommand\{\\([A-Za-z]+)\}\{([^{}]+)\}",
            (PAPER / "generated/negotiated_repair_numbers.tex").read_text(),
        )
    )
    assert macros == {k: str(v) for k, v in figure["numbers"].items()}


def test_all_concessions_and_designed_authority_nulls_are_retained():
    rows = read(RESULT / "analysis/rows.json")
    for row in rows:
        if row["selected"] is None:
            continue
        assert len(row["selected"]["waived"]) <= 1
        for mutation in row["authority_mutations"]:
            assert not mutation["full_release"]
            assert mutation["omit_consent_release"]
            assert mutation["wrong_authority_if_omitted"]
    pairs = read(RESULT / "analysis/paired_work.json")
    assert any(not r["same_target"] for r in pairs)
    assert all(
        r["same_target"] and r["new_pair_difference"] == 0
        for r in pairs
        if r["kind"].endswith("control")
    )


def test_post_run_sql_projection_is_complete():
    audit = read(REPORT / "metric_projection_audit.json")
    assert audit["status"] == "PASS" and not audit["discrepancies"]
    assert (
        audit["unique_parent_capsules"]
        == read(RESULT / "summary.json")["actual_paired_attempts"]
    )
    assert not audit["new_simulation_pairs"]


def test_active_manuscript_uses_new_results_and_preserves_original_registration():
    main = (PAPER / "main.tex").read_text(encoding="utf-8")
    comparison = (PAPER / "sections/comparison.tex").read_text(encoding="utf-8")
    assert r"\input{sections/negotiated_evaluation}" in main
    assert r"\input{sections/owner_authorized_repair}" in comparison
    # The author-approved current diagram separates revision validity scopes;
    # the full veto/continuation path is retained, with unchanged figure hashes.
    assert "generated/revision_lifetimes.pdf" in comparison
    assert "generated/owner_policy_search.pdf" in (
        PAPER / "technical_supplement.tex"
    ).read_text(encoding="utf-8")
    assert "generated/contract_repair.pdf" in (
        PAPER / "technical_supplement.tex"
    ).read_text(encoding="utf-8")
    assert r"\input{generated/negotiated_repair_numbers.tex}" in main
    assert r"\input{sections/owner_authorized_supplement}" in (
        PAPER / "technical_supplement.tex"
    ).read_text(encoding="utf-8")
    from scripts.check_registered_metadata import check_registered_metadata

    check_registered_metadata(
        ROOT / "paper/main_revised.tex", ROOT / "paper/registered_metadata_lock.json"
    )
