"""Post-run publication checks; sealed runtime and endpoints are unchanged."""

import gzip
import hashlib
import importlib.util
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/cscwd2027/typed_revision_v1"
RESULT = ROOT / "results/cscwd2027/typed_revision_v1"
PAPER = ROOT / "paper/cscwd2027_design_contract"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_typed_seal_precedes_every_acquisition_and_protects_prior_inputs():
    from scripts.cscwd2027_typed_revision import verify

    seal = verify()
    started = read(RESULT / "run_started.json")
    assert datetime.fromisoformat(seal["created_at_utc"]) < datetime.fromisoformat(
        started["created_at_utc"]
    )
    assert read(RESULT / "run_completed.json")["failures"] == []


def test_counts_are_actual_attempts_and_recorded_actions():
    summary = read(RESULT / "summary.json")
    rows = read(RESULT / "analysis/rows.json")
    actual = [read(p) for p in (RESULT / "objects").glob("*/result.json")]
    assert len(actual) == summary["actual_paired_attempts"]
    assert sum(r["local_pass"] for r in actual) == summary["local_pass"]
    assert len(rows) == len(read(CONFIG / "schedule.json")) * 3 * 3 * 2
    for mode in ("typed", "opaque"):
        group = [r for r in rows if r["method"] == mode]
        assert sum(r["new_pairs"] for r in group) == summary[mode]["new_pairs"]
    for row in rows:
        folder = RESULT / "handoffs" / row["id"]
        order = read(folder / "producer_work_order.json")
        response = read(folder / "producer_response.json")
        assert order["contract_sha256"] == response["contract_sha256"]
        actions = Counter(a["action"] for a in order["actions"])
        assert actions["execute"] == row["new_pairs"]
        assert actions["evaluate"] == row["reevaluations"]
        assert len(response["actions"]) == len(order["actions"])
        ordered_at = datetime.fromisoformat(order["timestamp_before_acquisition"])
        for action in response["actions"]:
            if action["action"] == "execute":
                registration = read(
                    RESULT / "objects" / action["trace"]["object_id"] / "attempt.json"
                )["registration"]
                assert ordered_at <= datetime.fromisoformat(
                    registration["registered_at_utc"]
                )
        available = {tuple(slot) for slot in order["available_slots"]}
        for action in order["actions"]:
            if action["action"] != "execute":
                assert (action["candidate"], action["replicate"]) in available
        assert (folder / "metric_binding_ablation.json").is_file()


def test_partial_store_controls_and_relational_agreement():
    for row in read(RESULT / "analysis/rows.json"):
        if row["availability"] == 0:
            assert not row["unsafe_invalid_release"]
            assert row["new_pairs"] == 12
        if row["method"] == "typed":
            assert row["relational_agreement"]
            if row["final_release"]:
                assert row["comparison"]["exact"]
                assert row["repeat_request_all_reuse"]
        if not row["final_release"] or not row["fresh_release"]:
            assert row["comparison"]["regret"] is None


def test_complete_rows_macros_and_exports_are_bound():
    manifest = read(PAPER / "generated/figure_manifest_typed_revision.json")
    for category in ("inputs", "source", "outputs"):
        for name, digest in manifest[category].items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    plot = read(PAPER / "generated/typed_revision_plot_data.json")
    assert plot["rows"] == read(RESULT / "analysis/rows.json")
    assert plot["axis"]["y_min"] == 0
    macros = dict(
        re.findall(
            r"\\newcommand\{\\([A-Za-z]+)\}\{([^{}]*)\}",
            (PAPER / "generated/typed_revision_numbers.tex").read_text(),
        )
    )
    for key, value in plot["macro_values"].items():
        assert macros[key] == str(value)


def test_original_metadata_and_current_anonymous_template():
    from scripts.check_registered_metadata import check_registered_metadata

    check_registered_metadata(
        ROOT / "paper/main_revised.tex", ROOT / "paper/registered_metadata_lock.json"
    )
    main = (PAPER / "main.tex").read_text()
    assert r"\documentclass[conference]{IEEEtran}" in main
    assert "Anonymous authors" in main
    assert not re.search(
        r"\\(?:geometry|fontsize|linespread|baselinestretch|enlargethispage)", main
    )
    text = "\n".join(p.read_text() for p in (PAPER / "sections").glob("*.tex"))
    assert "\u2014" not in text
    assert "three-way" in text


def test_post_run_sql_projection_on_manual_fixture(tmp_path):
    path = (
        ROOT
        / "ccfa-workfiles/writing/paired-comparison/audit_typed_metric_projection.py"
    )
    spec = importlib.util.spec_from_file_location("typed_sql_audit_fixture", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.RESULT = tmp_path
    worlds = {}
    for world, quantities in (("baseline", (2, 3)), ("policy", (4, 2))):
        worlds[world] = {
            "completed": True,
            "epochs": [
                {
                    "epoch": epoch,
                    "applied_commands": [{"event_key": {"value": f"{world}-{epoch}"}}],
                }
                for epoch in range(4)
            ],
            "trades": [
                {
                    "cause_event_key": f"{world}-{epoch}",
                    "quantity": quantity,
                    "price": price,
                }
                for epoch, quantity, price in zip(
                    (0, 2), quantities, (10, 20), strict=True
                )
            ],
        }
    folder = tmp_path / "objects/fixture"
    folder.mkdir(parents=True)
    with gzip.open(folder / "capsule.json.gz", "wt", encoding="utf-8") as handle:
        json.dump({"spec": {"epoch_count": 4}, "worlds": worlds}, handle)
    values = module.sql_projection("fixture")
    assert values["notional", 0] == {"baseline": 80, "policy": 80, "contrast": 0}
    assert values["volume", 0] == {"baseline": 5, "policy": 6, "contrast": 1}
    assert values["volume", 2] == {"baseline": 3, "policy": 2, "contrast": -1}
