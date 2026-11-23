"""Publication checks; these do not change the pre-run scientific analysis."""

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/cscwd2027/mechanism_review_v1"
CONFIG = ROOT / "configs/cscwd2027/mechanism_review_v1"
PAPER = ROOT / "paper/cscwd2027_design_contract"


def read(path):
    return json.loads(path.read_text())


def test_pre_execution_seal_and_prior_evidence_are_unchanged():
    from scripts.cscwd2027_mechanism_review import protect_check, verify

    seal = verify()
    started = read(RESULT / "run_started.json")
    assert datetime.fromisoformat(seal["created_at_utc"]) < datetime.fromisoformat(
        started["created_at_utc"]
    )
    assert protect_check()["unchanged"]
    amendment = read(CONFIG / "analysis_schema_amendment.json")
    assert amendment["changes_numeric_fields"] is False
    assert amendment["changed_endpoints"] is False


def test_all_attempts_and_local_rejections_are_retained():
    summary = read(RESULT / "summary.json")["coupling"]
    attempts = read(CONFIG / "attempts.json")
    actual = [read(RESULT / "objects" / r["id"] / "result.json") for r in attempts]
    assert len(actual) == summary["pairs"]
    assert sum(r["local_pass"] for r in actual) == summary["local_pass"]
    assert {r["id"] for r in actual if not r["local_pass"]} == set(
        summary["local_reject_ids"]
    )
    assert {r["id"] for r in actual if r["status"] != "completed"} == set(
        summary["failed_ids"]
    )
    rows = read(RESULT / "analysis/coupling_rows.json")
    assert len(rows) == len(read(CONFIG / "contexts.json")) * 2 * len(
        read(CONFIG / "study.json")["panel_cases"]
    ) * len(read(CONFIG / "study.json")["collectors"])
    assert all(not r["effects"]["assessable"] for r in rows if not r["released"])


def test_switch_statistics_and_clean_control_denominators():
    summary = read(RESULT / "summary.json")["coupling"]["primary_switch"]
    rows = read(RESULT / "analysis/coupling_rows.json")
    for mode, expected in summary.items():
        group = [
            r
            for r in rows
            if "primary" in r["families"]
            and r["mode"] == mode
            and r["case"] == "coupling_switch_target"
        ]
        assert len(group) == expected["panels"]
        assert (
            sum(r["released"] and not r["valid"] for r in group)
            == expected["unauthorized"]
        )
        assert (
            sum(bool(r["effects"]["top1_disagreement"]) for r in group)
            == expected["top1_flips"]
        )
    assert summary["comparison_contract"]["unauthorized"] == 0
    assert summary["conventional_relational"]["unauthorized"] == 0


def test_composite_residuals_and_actual_repair_packet_totals():
    summary = read(RESULT / "summary.json")["repair"]
    rows = read(RESULT / "analysis/repair_rows.json")
    assert len(rows) == summary["panels"]
    assert (
        sum(r["score"]["exact_reason_set"] for r in rows)
        == summary["exact_reason_sets"]
    )
    assert (
        sum(r["score"]["correct_slots"] for r in rows)
        == summary["micro_slot_recall"][0]
    )
    for method, counts in summary["methods"].items():
        values = [next(v for v in r["repairs"] if v["method"] == method) for r in rows]
        for metric in (
            "new_pairs",
            "object_transfers",
            "round_trips",
            "request_response_bytes",
        ):
            assert sum(v[metric] for v in values) == counts[metric]
        for v in values:
            assert len(v["packets"]) == v["round_trips"]
            assert (
                sum(p["request_bytes"] + p["response_bytes"] for p in v["packets"])
                == v["request_response_bytes"]
            )
            assert v["new_pairs"] == 0


def test_new_presentation_manifest_and_locked_template():
    manifest = read(PAPER / "generated/figure_manifest_mechanism.json")
    for collection in ("inputs", "source", "outputs"):
        for path, digest in manifest[collection].items():
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    meta = read(PAPER / "cscwd_draft_metadata.json")
    assert meta["authors"] == "Anonymous authors"
    main = (PAPER / "main.tex").read_text()
    assert r"\documentclass[conference]{IEEEtran}" in main
    assert not re.search(
        r"\\(?:geometry|fontsize|linespread|baselinestretch|enlargethispage)", main
    )
