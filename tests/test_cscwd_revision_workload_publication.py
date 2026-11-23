"""Post-run publication checks, separate from frozen scientific observations."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/cscwd2027/revision_workload_v1"
CONFIG = ROOT / "configs/cscwd2027/revision_workload_v1"
PAPER = ROOT / "paper/cscwd2027_design_contract"
REPORT = ROOT / "reports/cscwd2027/revision_workload_v1"


def read(path: Path):
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_scientific_and_publication_hash_bindings():
    for name, digest in read(CONFIG / "freeze.json")["members"].items():
        assert sha(ROOT / name) == digest, name
    for name, digest in read(RESULT / "manifest.json")["members"].items():
        assert sha(RESULT / name) == digest, name
    for manifest_name in (
        "revision_workload_table_manifest.json",
        "figure_manifest_revision_workload.json",
        "figure_manifest_revision_lifetimes.json",
    ):
        manifest = read(PAPER / "generated" / manifest_name)
        for group in ("inputs", "source", "outputs"):
            for name, digest in manifest[group].items():
                assert sha(ROOT / name) == digest, name


def test_complete_preregistered_route_and_physical_denominators():
    summary, seal = read(RESULT / "summary.json"), read(CONFIG / "freeze.json")
    for field in ("planning_contexts", "route_measurements"):
        assert summary[field] == seal[field]
    assert summary["physical_panels"] == seal["physical_method_panels"]
    rows = read(RESULT / "physical_rows.json")
    expected = {
        (b["session"]["id"], e["episode"], s["stage"], m)
        for b in read(CONFIG / "schedule.json")
        for e in b["episodes"]
        for s in e["stages"]
        if s["physical"]
        for m in read(CONFIG / "study.json")["physical_methods"]
    }
    observed = {(r["session"], r["episode"], r["stage"], r["method"]) for r in rows}
    assert observed == expected
    attempts = read(RESULT / "attempt_rows.json")
    assert summary["actual_pairs"] == len(attempts)
    assert summary["failed_attempts"] == sum(
        r["status"] != "completed" or not r["local_pass"] for r in attempts
    )
    assert len(list((RESULT / "objects").glob("*/result.json"))) == len(attempts)


def test_all_timings_balanced_and_all_repeats_retained():
    rows = read(RESULT / "timing_rows.json")
    groups = defaultdict(list)
    for row in rows:
        key = tuple(
            row[k] for k in ("session", "domain_size", "episode", "stage", "method")
        )
        groups[key].append(row)
    assert len(rows) == read(RESULT / "summary.json")["route_measurements"]
    for values in groups.values():
        assert sorted(r["repeat"] for r in values) == list(range(5))
        assert sorted(r["position"] for r in values) == list(range(5))
        assert len({r["counts"]["search_predicate_calls"] for r in values}) == 1


def test_plotted_totals_are_complete_context_means_not_selected_routes():
    data = read(PAPER / "generated/revision_workload_plot_data.json")
    raw = read(RESULT / "timing_rows.json")
    for point in data["points"]:
        means = []
        for repeat in range(5):
            values = [
                r
                for r in raw
                if r["method"] == point["method"]
                and r["domain_size"] == point["domain_size"]
                and r["repeat"] == repeat
            ]
            assert len(values) == data["contexts_per_domain"]
            means.append(sum(r["wall_ns"] for r in values) / len(values) / 1e6)
        assert point["median_wall_ms"] == median(means)
        assert point["min_wall_ms"] == min(means)
        assert point["max_wall_ms"] == max(means)
    for row in data["phases"]:
        assert abs(sum(row["components"].values()) - row["total_ms"]) < 1e-8


def test_manuscript_numbers_resolve_to_complete_structured_summary():
    numbers = read(PAPER / "generated/revision_workload_numbers.json")
    macros = dict(
        re.findall(
            r"\\newcommand\{\\([A-Za-z]+)\}\{([^{}]+)\}",
            (PAPER / "generated/revision_workload_numbers.tex").read_text(),
        )
    )
    assert macros == {key: str(value) for key, value in numbers.items()}
    assert all("pending" not in str(value).lower() for value in numbers.values())
    summary = read(RESULT / "summary.json")
    study = read(CONFIG / "study.json")
    assert numbers["WorkloadEpochs"] == study["epochs"]
    assert numbers["WorkloadReplicates"] == study["replicates"]
    assert numbers["WorkloadSmallAgents"] == min(study["agents"])
    assert numbers["WorkloadLargeAgents"] == max(study["agents"])
    design = (PAPER / "sections/revision_workload_design.tex").read_text()
    assert r"\WorkloadEpochs{}" in design
    assert "ProjectionEpochs" not in design
    assert numbers["WorkloadActualPairs"] == summary["actual_pairs"]
    assert numbers["WorkloadFailures"] == summary["failed_attempts"]
    assert (
        numbers["WorkloadAgreements"] == summary["methods"]["persistent"]["agreements"]
    )
    assert numbers["WorkloadStaleAdmitted"] == summary["stale_approvals_admitted"]
    assert (
        "revision_workload_results"
        in (PAPER / "sections/negotiated_evaluation.tex").read_text()
    )


def test_record_audit_is_explicitly_bounded_and_passes():
    report = read(REPORT / "independent_record_check.json")
    assert report["status"] == "PASS"
    assert not report["errors"]


def test_original_registered_metadata_is_still_locked():
    from scripts.check_registered_metadata import check_registered_metadata

    check_registered_metadata(
        ROOT / "paper/main_revised.tex", ROOT / "paper/registered_metadata_lock.json"
    )
