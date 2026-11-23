"""Presentation-only invariants for the author-approved CSCWD Figure 1."""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "ccfa-workfiles/figures/contract-overview"
PAPER = ROOT / "paper/cscwd2027_design_contract"
NS = {"svg": "http://www.w3.org/2000/svg"}


def test_confirmed_spec_nodes_and_edges_are_present_in_live_svg() -> None:
    spec = json.loads((WORK / "spec.json").read_text())
    svg = ET.parse(WORK / "fig.final.svg").getroot()
    elements = {element.get("id"): element for element in svg.iter()}
    for node in spec["nodes"]:
        group = elements[f"n:{node['id']}"]
        labels = [element.text for element in group.findall("svg:text", NS)]
        assert node["label"] in labels
        if "sublabel" in node:
            assert node["sublabel"] in labels
    for edge in spec["edges"]:
        group = elements[f"e:{edge['id']}"]
        assert group.get("data-from") == edge["from"]
        assert group.get("data-to") == edge["to"]
        path = group.find("svg:path", NS)
        assert path is not None and path.get("marker-end") == "url(#arrow)"
        assert bool(path.get("stroke-dasharray")) == (edge["style"] == "dashed")


def test_runtime_has_only_supporting_specification_inputs() -> None:
    spec = json.loads((WORK / "spec.json").read_text())
    inputs = [edge for edge in spec["edges"] if edge["to"] == "simulation-runtime"]
    assert {edge["from"] for edge in inputs} == {
        "stochastic-contract",
        "execution-plan",
    }
    assert all(edge["kind"] == "control" for edge in inputs)
    assert not any(edge["from"] == "reuse-slots" for edge in spec["edges"])
    assert {edge["to"] for edge in spec["edges"] if edge["from"] == "panel-checks"} == {
        "numerical-report",
        "reuse-slots",
    }


def test_manifest_binds_the_current_source_and_both_vector_exports() -> None:
    manifest = json.loads(
        (PAPER / "generated/figure_manifest_contract_overview.json").read_text()
    )
    assert manifest["author_confirmed_brief"] is True
    assert manifest["numeric_or_experimental_changes"] is False
    assert manifest["lint_errors"] == manifest["lint_warnings"] == 0
    for collection in ("sources", "outputs"):
        for path, expected in manifest[collection].items():
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected
    assert (PAPER / "generated/contract_overview.pdf").read_bytes() == (
        WORK / "fig.final.pdf"
    ).read_bytes()


def test_prior_generated_statistics_and_data_figures_remain_byte_identical() -> None:
    archive = ROOT / (
        "release/cscwd2027/review_20261025_design_contract_v2/manuscript_source.zip"
    )
    prefix = "paper/cscwd2027_design_contract/generated/"
    with ZipFile(archive) as zipped:
        members = [
            name
            for name in zipped.namelist()
            if name.startswith(prefix) and not name.endswith("/")
        ]
        assert members
        for name in members:
            assert (ROOT / name).read_bytes() == zipped.read(name), name
