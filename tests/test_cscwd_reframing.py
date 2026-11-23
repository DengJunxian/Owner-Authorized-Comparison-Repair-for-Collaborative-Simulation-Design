"""Checks for evidence-preserving CSCWD retargeting and scientific plots."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper/cscwd2027_design_contract"
WORK = ROOT / "ccfa-workfiles/writing/paired-comparison"


def test_current_draft_metadata_matches_the_rewritten_sources() -> None:
    path = WORK / "update_metadata.py"
    spec = importlib.util.spec_from_file_location("cscwd_draft_metadata", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    recorded = json.loads((PAPER / "cscwd_draft_metadata.json").read_text())
    assert recorded == module.current_metadata(ROOT)
    assert recorded["authors"] == "Anonymous authors"
    assert "not a registered submission" in recorded["status"]


def test_plot_counts_and_denominators_match_the_complete_study() -> None:
    source = json.loads(
        (
            ROOT / ("results/cscwd2027/design_contract_v1/derived/summary.json")
        ).read_text()
    )
    plotted = json.loads((PAPER / "generated/cscwd_plot_data.json").read_text())
    comparison = plotted["comparison"]
    assert {row["mode"] for row in comparison["rows"]} == set(source["mode_statistics"])
    for row in comparison["rows"]:
        original = source["mode_statistics"][row["mode"]]
        assert row["unauthorized"] == [
            original["unauthorized_release"],
            original["sets"],
        ]
        assert row["positive_regret"] == [
            original["optimal_set_error"],
            original["regret_denominator"],
        ]
        assert row["valid_retained"] == [
            original["valid_control_release"],
            original["evaluable_controls"],
        ]
    assert set(comparison["cases"]) == set(source["case_statistics"])
    for row, cells in zip(comparison["rows"], comparison["case_matrix"], strict=True):
        for case, cell in zip(comparison["cases"], cells, strict=True):
            original = source["case_statistics"][case][row["mode"]]
            assert cell == [original["unauthorized_release"], original["sets"]]


def test_all_original_cost_cells_and_quantiles_are_preserved() -> None:
    source = json.loads(
        (ROOT / "results/cscwd2027/derived_v1/summary.json").read_text()
    )
    plotted = json.loads((PAPER / "generated/cscwd_plot_data.json").read_text())
    assert plotted["cost_cells"] == source["cost"]


def test_merged_handoff_table_preserves_every_original_endpoint() -> None:
    source = json.loads(
        (
            ROOT / ("results/cscwd2027/design_contract_v1/derived/summary.json")
        ).read_text()
    )
    plotted = json.loads((PAPER / "generated/cscwd_plot_data.json").read_text())
    lines = (PAPER / "generated/handoff_summary_cscwd.tex").read_text().splitlines()
    for row in plotted["comparison"]["rows"]:
        matching = [line for line in lines if line.startswith(row["label"] + " & ")]
        assert len(matching) == 1
        cells = matching[0].removesuffix(r"\\").strip().split(" & ")[1:]
        s = source["mode_statistics"][row["mode"]]
        assert cells == [
            str(s["written"]),
            str(s["unauthorized_release"]),
            f"{s['valid_control_release']}/{s['evaluable_controls']}",
            str(s["top1_disagreement"]),
            str(s["top2_disagreement"]),
            str(s["optimal_set_error"]),
            str(s["zero_regret_disagreement"]),
        ]


def test_plot_manifest_binds_inputs_renderer_and_outputs() -> None:
    manifest = json.loads(
        (PAPER / "generated/figure_manifest_cscwd_reframing.json").read_text()
    )
    assert manifest["numeric_or_experimental_changes"] is False
    for group in ("inputs", "source", "outputs"):
        for path, expected in manifest[group].items():
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected


def test_panel_obligations_and_original_theory_remain_available() -> None:
    archive = WORK / "source/pre_reframing_source.zip"
    with ZipFile(archive) as baseline:
        paths = [
            p for p in baseline.namelist() if "/sections/" in p and p.endswith(".tex")
        ]
        old = "\n".join(baseline.read(path).decode() for path in paths)
    pattern = r"\\begin\{equation\}[\s\S]*?\\end\{equation\}"
    original_panel = next(
        x for x in re.findall(pattern, old) if r"\label{eq:panel}" in x
    )
    assert original_panel in old
    comparison = (PAPER / "sections/comparison.tex").read_text()
    for obligation in (
        r"\operatorname{slots}(B)=_{\rm multi}\mathcal S",
        r"\mathsf{LocalValid}(o)",
        r"\operatorname{key}(o)=K_D(\operatorname{slot}(o))",
        r"h(P_o)\in\mathcal P",
    ):
        assert obligation in comparison
    law = r"\\textbf\{Proposition 2[\s\S]*?(?=\\subsection\{Threat Model\})"
    with ZipFile(WORK / "source/pre_review_revision.zip") as previous:
        assert re.search(law, old)[0] in previous.read("sections/protocol.tex").decode()
    supplement = (PAPER / "technical_supplement.tex").read_text()
    assert "accepted law is generally conditional" in supplement
    assert "Almost-sure acceptance" in supplement
    assert "observed all-pass batch establishes neither condition" in supplement


def test_original_citations_and_template_are_preserved() -> None:
    with ZipFile(WORK / "source/pre_reframing_source.zip") as baseline:
        paths = [
            p for p in baseline.namelist() if "/sections/" in p and p.endswith(".tex")
        ]
        old = "\n".join(baseline.read(path).decode() for path in paths)
        new = "\n".join((ROOT / path).read_text() for path in paths)

        def citations(text: str) -> set[str]:
            return {
                key.strip()
                for group in re.findall(r"\\cite\{([^}]+)\}", text)
                for key in group.split(",")
            }

        assert citations(old) <= citations(
            new
            + (PAPER / "sections/related.tex").read_text()
            + (PAPER / "technical_supplement.tex").read_text()
        )
        path = "paper/cscwd2027_design_contract/IEEEtran.cls"
        assert baseline.read(path) == (ROOT / path).read_bytes()
        path = "paper/cscwd2027_design_contract/references.bib"
        assert (ROOT / path).read_bytes().startswith(baseline.read(path))
    main = (PAPER / "main.tex").read_text()
    assert "\\documentclass[conference]{IEEEtran}" in main
    assert "\\IEEEauthorblockN{Anonymous authors}" in main
    assert not re.search(
        r"\\(?:usepackage\{geometry\}|fontsize|setlength|addtolength)", main
    )
    assert main.index("sections/problem") < main.index("sections/comparison")
    assert "sections/protocol" in (PAPER / "sections/problem.tex").read_text()
